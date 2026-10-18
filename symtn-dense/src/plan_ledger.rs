//! Additive reservation ledger for the cuTENSOR plan cache bound.
//!
//! Pure host arithmetic, compiled and tested without the `cuda` feature. The
//! context applies the bound it computes to its Tenferro backend.

/// Entries of a cuTENSOR contraction plan cost about this much retained state.
/// Used only to bound how far the plan entry cap is
/// raised.
pub(crate) const CUTENSOR_PLAN_BYTES: usize = 14 * 1024;

/// Default ceiling on reserved cuTENSOR plan entries: 8 MiB at an estimated
/// 14 KB per plan, about 585 distinct operand signatures. Tenferro's own
/// default bound is 64, which a transform with more distinct block layouts than
/// that would thrash. The per-plan figure is an estimate of a Tenferro
/// internal, so it bounds only how far the cap is raised.
pub const DEFAULT_PLAN_CACHE_BUDGET_BYTES: usize = 8 * 1024 * 1024;

/// How many plan entries `budget_bytes` pays for: the requirement, truncated
/// to what the budget affords.
pub fn plan_cache_entries_for(required: usize, budget_bytes: usize) -> usize {
    required.min(budget_bytes / CUTENSOR_PLAN_BYTES)
}

/// Sum of every live consumer's plan-entry reservation.
///
/// Why additive rather than each consumer raising the cap to its own absolute
/// need: absolute raises from independent consumers take the maximum, not the
/// sum, so two working sets that each fit end up sharing one of their sizes
/// and evict each other. Consumers here reserve and release deltas, and the
/// cap is `base + reserved`, where `base` is the backend's bound when the
/// context was built and stays the pool for uncounted eager submissions.
#[derive(Debug)]
pub(crate) struct PlanEntryLedger {
    base: usize,
    limit: usize,
    reserved: usize,
    shortfall: u64,
}

impl PlanEntryLedger {
    pub(crate) fn new(base: usize, limit: usize) -> Self {
        Self {
            base,
            limit,
            reserved: 0,
            shortfall: 0,
        }
    }

    /// The part of `delta` the limit still affords.
    pub(crate) fn grant(&self, delta: usize) -> usize {
        delta.min(self.limit.saturating_sub(self.reserved))
    }

    /// The cap needed once `granted` more entries are reserved. The caller
    /// raises the backend to at least this, never lowers it, so a release
    /// followed by a re-reserve reuses headroom instead of growing the cap.
    pub(crate) fn cap_with(&self, granted: usize) -> usize {
        self.base
            .saturating_add(self.reserved)
            .saturating_add(granted)
    }

    /// Records a reservation once the cap covering it is in place.
    pub(crate) fn commit(&mut self, requested: usize, granted: usize) {
        self.reserved += granted;
        self.shortfall = self.shortfall.saturating_add((requested - granted) as u64);
    }

    pub(crate) fn release(&mut self, entries: usize) {
        self.reserved = self.reserved.saturating_sub(entries);
    }

    pub(crate) fn reserved(&self) -> usize {
        self.reserved
    }

    /// Entries requested but not granted, summed over every reservation.
    pub(crate) fn shortfall(&self) -> u64 {
        self.shortfall
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Applies `reserve` the way the context does: grant, raise the cap
    /// monotonically, commit.
    fn reserve(ledger: &mut PlanEntryLedger, cap: &mut usize, delta: usize) -> usize {
        let granted = ledger.grant(delta);
        *cap = (*cap).max(ledger.cap_with(granted));
        ledger.commit(delta, granted);
        granted
    }

    #[test]
    fn the_plan_cap_never_exceeds_the_byte_budget() {
        assert_eq!(plan_cache_entries_for(200, 300 * CUTENSOR_PLAN_BYTES), 200);
        assert_eq!(plan_cache_entries_for(200, 10 * CUTENSOR_PLAN_BYTES), 10);
        assert_eq!(plan_cache_entries_for(200, 0), 0);
    }

    #[test]
    fn two_reservations_add() {
        let mut ledger = PlanEntryLedger::new(64, 1000);
        let mut cap = 64;
        assert_eq!(reserve(&mut ledger, &mut cap, 20), 20);
        assert_eq!(reserve(&mut ledger, &mut cap, 80), 80);
        assert_eq!(ledger.reserved(), 100);
        assert_eq!(cap, 164);
        assert_eq!(ledger.shortfall(), 0);
    }

    #[test]
    fn release_then_re_reserve_does_not_grow_the_cap() {
        let mut ledger = PlanEntryLedger::new(64, 1000);
        let mut cap = 64;
        reserve(&mut ledger, &mut cap, 20);
        reserve(&mut ledger, &mut cap, 80);
        for _ in 0..5 {
            ledger.release(80);
            assert_eq!(ledger.reserved(), 20);
            reserve(&mut ledger, &mut cap, 80);
            assert_eq!(cap, 164, "a create/drop cycle crept the cap");
        }
        // Releasing never shrinks the cap either.
        ledger.release(100);
        assert_eq!(ledger.reserved(), 0);
        assert_eq!(cap, 164);
    }

    #[test]
    fn an_over_budget_reserve_is_truncated_and_reported() {
        let mut ledger = PlanEntryLedger::new(64, 100);
        let mut cap = 64;
        assert_eq!(reserve(&mut ledger, &mut cap, 70), 70);
        assert_eq!(reserve(&mut ledger, &mut cap, 50), 30);
        assert_eq!(ledger.shortfall(), 20);
        assert_eq!(reserve(&mut ledger, &mut cap, 5), 0);
        assert_eq!(ledger.shortfall(), 25);
        assert_eq!(ledger.reserved(), 100);
        assert_eq!(cap, 164);
        // Headroom a release frees is granted again.
        ledger.release(10);
        assert_eq!(reserve(&mut ledger, &mut cap, 10), 10);
        assert_eq!(cap, 164);
    }

    #[test]
    fn a_cap_raised_elsewhere_absorbs_nothing() {
        // A cap already above `base + reserved` (another backend user raised
        // it) does not reduce what a reservation counts: the ledger still
        // holds every consumer's entries, and the cap still covers their sum.
        let mut ledger = PlanEntryLedger::new(64, 1000);
        let mut cap = 150;
        reserve(&mut ledger, &mut cap, 20);
        assert_eq!(cap, 150);
        reserve(&mut ledger, &mut cap, 80);
        assert_eq!(ledger.reserved(), 100);
        assert_eq!(cap, 164);
    }
}
