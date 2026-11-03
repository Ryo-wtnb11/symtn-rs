//! The block kernels' shared scheduler: J independent dense jobs in canonical
//! order on one engine's execution context, split into lanes that each
//! reserve a nonzero share of the context's slots (summing to at most S).
//!
//! Every resource number comes from the caller's `Exec`: the slot count S
//! from the token's `width_for` (Auto) or the policy's `BatchDecision`
//! (forced); a forced policy's lane count is its decision's, and Auto's is at
//! most `L0 = min(S, J, batch_decision.lanes)`. Nothing here is calibrated or
//! timed.
//!
//! Under Auto, given a grant model, the scheduler scores the baseline
//! placement (L0 equal-share lanes, largest job first on the lane of least
//! accumulated serial time), an equal-share placement for every lane count
//! from L0 down to one, and a large-job reservation on uneven lanes (see
//! `Moldable::place`; discarded when more than L0 of its lanes hold jobs)
//! under one model, and keeps the lowest score; ties keep the baseline, then
//! the equal shares. The reservation is a candidate only under the default
//! [`LaneAllocation::Moldable`]; [`LaneAllocation::EqualShare`] scores the
//! baseline and the equal shares alone. A job's modeled time in a lane is its serial time when its provider
//! would run it on one thread there, otherwise its share of that time over
//! the provider's granted width plus the token's dispatch cost at that width.
//! The granted width is itself an estimate composed from the provider's
//! public decisions (for a Packed GEMM plan an optimistic upper bound on its
//! grid), and the model has no sublinear speedup, so a lower score is a
//! scheduling estimate, not a promise of a faster run. Without a grant model
//! (a QR batch, or a factorization batch holding an input whose batch product
//! is not one, including zero, see `driver::BlockOp::modeled`) the baseline
//! is returned.
//!
//! The search costs `O(J log J + G S + G' sum(W) + L0 J' + P J log L0)`,
//! counting the provider queries' width scans. Jobs with equal serial times
//! and grant keys share their grants, so only one of each of the G distinct
//! (time, key) groups is queried: once at the full S, and, for the G' groups
//! widened there, once per distinct equal-share width during the sweep (W,
//! with `sum(W) <= 2 S H(L0) + L0`; rebuilding the winner may repeat up to
//! two widths). A proved lower bound costs `O(J')` per lane count for the J'
//! widened jobs, the others entering it as one precomputed sum and maximum,
//! and only the `P <= L0` counts whose bound is below the best score so far
//! are placed, with one heap per width class. A searched allocation is kept
//! in the process-global [`StructureCacheKind::LaneAllocations`] cache, so a
//! call whose inputs were all searched before, on any executor, rebuilds it
//! in `O(J + P)` without a query.
//!
//! The reservation adds, for the fewer than S jobs above the mean load, one
//! query per width up to S (`O(S^3)` with the width scans), an
//! `O(S^2 log S)` sort and binary search of its targets, and one placement
//! of `O(J (R + log S))` steps for its R < S reserved lanes, querying each
//! widened (time, key) group once per reserved width it evaluates.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::Duration;

use symtn_core::{
    ErasedStructureCacheControl, StructureCache, StructureCacheEquivalent, StructureCacheInfo,
    StructureCacheKind,
};
use tcontract_exec::{BatchDecision, BatchQuery, BatchStrategy, Exec, WorkModel};

use super::driver::OpKind;
use super::gemm::ENTRY_OVERHEAD_BYTES;
use crate::LaneAllocation;

/// One lane: its reserved slot budget and its jobs in ascending canonical
/// order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Lane {
    /// Reserved, not effective: a job's provider call may use fewer slots.
    pub(crate) budget: usize,
    pub(crate) jobs: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Allocation {
    /// The caller's budget P.
    pub(crate) budget: usize,
    /// The slots S the lanes reserve from (`S <= P`); their budgets sum to at
    /// most S, and a slot no lane reserves stays idle.
    pub(crate) slots: usize,
    pub(crate) decision: BatchDecision,
    pub(crate) lanes: Vec<Lane>,
}

/// A nanosecond estimate as the dependency normalizes it (`tcontract-exec`
/// `exec.rs: ns_duration`): NaN and nonpositive are zero, huge saturates, so
/// cost ordering is a total order on `Duration`.
pub(crate) fn estimate(ns: f64) -> Duration {
    if ns.is_nan() || ns <= 0.0 {
        Duration::ZERO
    } else if ns >= 1e18 {
        Duration::from_secs(u64::MAX / 4)
    } else {
        Duration::from_secs_f64(ns * 1e-9)
    }
}

/// Serial time of `work` weighted multiply-adds under the engine's work model,
/// mirroring `tcontract-exec` `batch.rs: item_ns` (a rate that is not `> 0`,
/// NaN included, counts as 1).
pub(crate) fn work_model_ns(model: &WorkModel, work: f64) -> f64 {
    let rate = if model.muladds_per_ns > 0.0 {
        model.muladds_per_ns
    } else {
        1.0
    };
    model.item_overhead_ns + work / rate
}

/// The lane tables and LPT order of earlier allocations, kept by an owner
/// that schedules repeatedly so a warm allocation reuses their capacity.
#[derive(Debug, Default)]
pub(crate) struct AllocationScratch {
    lanes: Vec<Lane>,
    /// Lanes an allocation with fewer lanes set aside, kept so that a later
    /// allocation of more lanes (a baseline the search then narrows, every
    /// warm call) reuses their job capacity instead of allocating it again.
    spare: Vec<Lane>,
    order: Vec<usize>,
    load: Vec<Reverse<(Duration, usize, usize)>>,
    narrow: Vec<Reverse<(Duration, usize, usize)>>,
    columns: Columns,
    keys: Vec<GrantKey>,
    moldable: Moldable,
}

impl AllocationScratch {
    /// Takes `allocation`'s lane tables back for the next allocation.
    pub(crate) fn keep(&mut self, allocation: Allocation) {
        self.lanes = allocation.lanes;
    }

    /// The capacities kept, in bytes. Saturates.
    pub(crate) fn retained_bytes(&self) -> usize {
        use std::mem::size_of;
        self.lanes
            .iter()
            .chain(&self.spare)
            .map(|lane| lane.jobs.capacity().saturating_mul(size_of::<usize>()))
            .fold(
                self.lanes
                    .capacity()
                    .saturating_add(self.spare.capacity())
                    .saturating_mul(size_of::<Lane>()),
                usize::saturating_add,
            )
            .saturating_add(self.order.capacity().saturating_mul(size_of::<usize>()))
            .saturating_add(
                self.load
                    .capacity()
                    .saturating_add(self.narrow.capacity())
                    .saturating_mul(size_of::<Reverse<(Duration, usize, usize)>>()),
            )
            .saturating_add(self.columns.retained_bytes())
            .saturating_add(self.keys.capacity().saturating_mul(size_of::<GrantKey>()))
            .saturating_add(self.moldable.retained_bytes())
    }
}

/// What a job's grant depends on besides its serial time, the execution
/// context and its op's [`OpKind`]: jobs with equal keys and equal serial
/// times get equal grants at every width. Every op of one kind that reaches
/// the process-global [`lane_allocations`] cache must give different grant
/// inputs different keys.
pub(crate) type GrantKey = [u64; 2];

/// Everything a searched allocation is a function of, as the
/// [`StructureCacheKind::LaneAllocations`] cache keys it.
///
/// Why the key is complete: a searched allocation is a function of the
/// context's budget P, its work model (the strategy is Auto, the only one
/// that searches), the mean work (the batch decision), the token's dispatch
/// cost at every width `2..=P` (the slot count, the batch decision, every
/// grant and every item time read only those), whether the token carries a
/// team (the Faer grant's item parallelism), the jobs' serial times in
/// canonical order (the slot count, the order, the placements and the
/// scores), their grant keys (every other grant input) and the op kind,
/// which fixes the grant function: every GEMM block op grants through
/// `gemm::grant` (keys: the plan's algorithm and the work) and every
/// factorization through `factor::grant` (keys: the item's total and payback
/// work), so equal keys of different kinds never share an allocation, and
/// the selected [`LaneAllocation`], which fixes the candidates scored.
/// Floats are keyed by their bits, so equality is reflexive and a
/// differently signed zero only searches again.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LaneAllocationKey {
    family: OpKind,
    selector: LaneAllocation,
    budget: usize,
    model: [u64; 5],
    mean_work: u64,
    team: bool,
    costs: Box<[Duration]>,
    times: Box<[Duration]>,
    keys: Box<[GrantKey]>,
}

/// One call's [`LaneAllocationKey`], borrowed: a lookup reads the dispatch
/// costs from the token instead of staging them, so a hit allocates nothing.
struct LaneAllocationQuery<'a, 'e> {
    family: OpKind,
    selector: LaneAllocation,
    exec: &'a Exec<'e>,
    times: &'a [Duration],
    keys: &'a [GrantKey],
    mean_work: f64,
}

/// The work model's fields, by bits.
fn model_bits(model: &WorkModel) -> [u64; 5] {
    [
        model.item_overhead_ns.to_bits(),
        model.muladds_per_ns.to_bits(),
        model.min_work_per_lane_ns.to_bits(),
        model.min_items as u64,
        model.min_items_per_lane as u64,
    ]
}

/// The token's dispatch cost at every width `2..=budget`.
fn dispatch_costs<'a>(exec: &'a Exec<'_>) -> impl Iterator<Item = Duration> + 'a {
    (2..=exec.budget())
        .filter_map(NonZeroUsize::new)
        .map(|width| exec.token_ref().dispatch_cost(width))
}

/// The one hash of a key and of its borrowed query, field by field.
#[allow(clippy::too_many_arguments)]
fn hash_key<H: Hasher>(
    state: &mut H,
    (family, selector): (OpKind, LaneAllocation),
    budget: usize,
    model: [u64; 5],
    mean_work: u64,
    team: bool,
    costs: (usize, impl Iterator<Item = Duration>),
    times: &[Duration],
    keys: &[GrantKey],
) {
    (family, selector, budget, model, mean_work, team).hash(state);
    state.write_usize(costs.0);
    costs.1.for_each(|cost| cost.hash(state));
    times.hash(state);
    keys.hash(state);
}

impl Hash for LaneAllocationKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        hash_key(
            state,
            (self.family, self.selector),
            self.budget,
            self.model,
            self.mean_work,
            self.team,
            (self.costs.len(), self.costs.iter().copied()),
            &self.times,
            &self.keys,
        );
    }
}

impl LaneAllocationQuery<'_, '_> {
    fn model(&self) -> [u64; 5] {
        model_bits(&self.exec.batch_policy().model)
    }

    fn team(&self) -> bool {
        self.exec.token_ref().carries_team()
    }

    fn to_key(&self) -> LaneAllocationKey {
        LaneAllocationKey {
            family: self.family,
            selector: self.selector,
            budget: self.exec.budget(),
            model: self.model(),
            mean_work: self.mean_work.to_bits(),
            team: self.team(),
            costs: dispatch_costs(self.exec).collect(),
            times: self.times.into(),
            keys: self.keys.into(),
        }
    }
}

impl Hash for LaneAllocationQuery<'_, '_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let budget = self.exec.budget();
        hash_key(
            state,
            (self.family, self.selector),
            budget,
            self.model(),
            self.mean_work.to_bits(),
            self.team(),
            (budget.saturating_sub(1), dispatch_costs(self.exec)),
            self.times,
            self.keys,
        );
    }
}

impl StructureCacheEquivalent<LaneAllocationKey> for LaneAllocationQuery<'_, '_> {
    fn equivalent(&self, key: &LaneAllocationKey) -> bool {
        key.family == self.family
            && key.selector == self.selector
            && key.budget == self.exec.budget()
            && key.model == self.model()
            && key.mean_work == self.mean_work.to_bits()
            && key.team == self.team()
            && *key.times == *self.times
            && *key.keys == *self.keys
            && key.costs.len() + 1 == key.budget
            && key.costs.iter().copied().eq(dispatch_costs(self.exec))
    }
}

/// A searched allocation, immutable: the slot count, the batch decision,
/// each lane's budget and the end of its jobs in `jobs`.
#[derive(Debug)]
pub(crate) struct LaneAllocationEntry {
    slots: usize,
    decision: BatchDecision,
    lanes: Box<[(usize, usize)]>,
    jobs: Box<[usize]>,
}

pub(crate) type LaneAllocationCache = StructureCache<LaneAllocationKey, LaneAllocationEntry>;

/// The bytes charged for one cached allocation: the inline key and entry,
/// the entry overhead, and its boxed slices (`P - 1` costs, J times, keys
/// and job indices, one pair per lane). Saturates.
fn charge(key: &LaneAllocationKey, entry: &LaneAllocationEntry) -> u64 {
    use std::mem::size_of;
    [
        size_of::<LaneAllocationKey>(),
        size_of::<LaneAllocationEntry>(),
        ENTRY_OVERHEAD_BYTES,
        key.costs.len().saturating_mul(size_of::<Duration>()),
        key.times.len().saturating_mul(size_of::<Duration>()),
        key.keys.len().saturating_mul(size_of::<GrantKey>()),
        entry
            .lanes
            .len()
            .saturating_mul(size_of::<(usize, usize)>()),
        entry.jobs.len().saturating_mul(size_of::<usize>()),
    ]
    .into_iter()
    .fold(0, |total: u64, bytes| total.saturating_add(bytes as u64))
}

struct LaneAllocationsControl;

impl ErasedStructureCacheControl for LaneAllocationsControl {
    fn info(&self) -> StructureCacheInfo {
        lane_allocations().info()
    }

    fn clear(&self) {
        ErasedStructureCacheControl::clear(lane_allocations())
    }

    fn set_byte_budget(&self, bytes: u64) {
        ErasedStructureCacheControl::set_byte_budget(lane_allocations(), bytes)
    }
}

/// The process-global lane-allocation cache, registered with `symtn-core`'s
/// structure-cache registry on first use.
pub(crate) fn lane_allocations() -> &'static LaneAllocationCache {
    static CACHE: OnceLock<LaneAllocationCache> = OnceLock::new();
    CACHE.get_or_init(|| {
        // Refused only if another crate claimed the slot first. Then
        // `symtn::cache::clear` cannot reach this cache, so it admits
        // nothing: every batch searches per call, with identical results.
        let budget = symtn_core::register_structure_cache(
            StructureCacheKind::LaneAllocations,
            &LaneAllocationsControl,
        )
        .unwrap_or(0);
        StructureCache::new(StructureCacheKind::LaneAllocations, budget)
    })
}

/// `count` empty lanes, reusing `pool`'s lane and job capacity; lanes beyond
/// `count` move to `spare`, and missing ones come from it first.
fn lanes_from(mut pool: Vec<Lane>, spare: &mut Vec<Lane>, count: usize) -> Vec<Lane> {
    while pool.len() > count {
        spare.extend(pool.pop());
    }
    pool.reserve(count.saturating_sub(pool.len()));
    while pool.len() < count {
        pool.push(spare.pop().unwrap_or(Lane {
            budget: 0,
            jobs: Vec::new(),
        }));
    }
    for lane in &mut pool {
        lane.jobs.clear();
    }
    pool
}

/// [`allocate_in`] with every job's grant serial, where no equal-share
/// candidate can beat the baseline placement on jobs of equal time.
#[cfg(test)]
pub(crate) fn allocate(exec: &Exec<'_>, times: &[Duration], mean_work: f64) -> Allocation {
    allocate_in(
        exec,
        times,
        (OpKind::Gemm, LaneAllocation::Moldable, |_| [0; 2]),
        mean_work,
        &mut AllocationScratch::default(),
        &LaneAllocationCache::new(StructureCacheKind::LaneAllocations, 0),
        Some(|_, _, _| 1),
    )
}

/// A job's modeled time in a lane where its provider grants it `width`
/// slots: its serial time `serial` when the grant is one slot, otherwise
/// `serial / width` plus the token's dispatch cost at that width, in integer
/// nanoseconds.
pub(crate) fn item_time(exec: &Exec<'_>, serial: Duration, width: usize) -> Duration {
    match NonZeroUsize::new(width).filter(|w| w.get() > 1) {
        None => serial,
        Some(w) => (serial / u32::try_from(w.get()).unwrap_or(u32::MAX))
            .saturating_add(exec.token_ref().dispatch_cost(w)),
    }
}

/// Whether an allocation of jobs with serial times `times` and mean work
/// `mean_work` on `exec` can search or reuse a searched allocation: Auto, at
/// least two jobs, a batch decision of at least two lanes and at least two
/// slots (the lane cap and slot count [`allocate_in`] derives). Otherwise it
/// returns the baseline, whatever the grant.
pub(crate) fn searchable(exec: &Exec<'_>, times: &[Duration], mean_work: f64) -> bool {
    exec.batch_policy().strategy == BatchStrategy::Auto
        && exec.budget() >= 2
        && times.len() >= 2
        && exec
            .batch_decision(&BatchQuery::new(times.len(), mean_work))
            .lanes
            >= 2
        && exec
            .token_ref()
            .width_for(
                times
                    .iter()
                    .fold(Duration::ZERO, |sum, &t| sum.saturating_add(t)),
                1,
            )
            .get()
            >= 2
}

/// Lane allocation of jobs with serial times `times` (canonical order)
/// and mean weighted work `mean_work` (multiply-adds; it feeds the
/// `BatchQuery`, while `times` feed `width_for`, so neither is converted
/// twice), with its tables drawn from `scratch`.
///
/// `grant(j, times[j], w)` is the width job `j`'s provider takes in a lane of
/// `w` reserved slots, `key(j)` its other inputs (see [`GrantKey`]) and
/// `family` the op kind whose grant function `grant` is;
/// Auto scores the candidates `selector` admits with it (under
/// [`LaneAllocation::EqualShare`], the baseline and the equal shares only,
/// without the large-job reservation) (see the module documentation), and
/// takes a searched allocation from `cache` when one was computed from the
/// same inputs (see [`LaneAllocationKey`]), offering each new search for
/// admission. Without a grant, and under a forced strategy, the baseline
/// placement is returned and no candidate is scored; `key` is then not
/// called.
///
/// Why a lookup and a publish rather than the cache's single-flight build:
/// that build takes an owned key, which a hit would have to stage. Racing
/// misses of one key each search, and the losers count as duplicate builds.
pub(crate) fn allocate_in(
    exec: &Exec<'_>,
    times: &[Duration],
    (family, selector, key): (OpKind, LaneAllocation, impl Fn(usize) -> GrantKey),
    mean_work: f64,
    scratch: &mut AllocationScratch,
    cache: &LaneAllocationCache,
    grant: Option<impl Fn(usize, Duration, usize) -> usize>,
) -> Allocation {
    // The keys are staged only where a search can run, so a serial context
    // retains nothing for them and reads no grant input.
    let mut keys = std::mem::take(&mut scratch.keys);
    keys.clear();
    if grant.is_some() && searchable(exec, times, mean_work) {
        keys.extend((0..times.len()).map(key));
    }
    let allocation = allocate_with(
        exec,
        times,
        (family, selector, &keys),
        mean_work,
        scratch,
        cache,
        grant,
    );
    scratch.keys = keys;
    allocation
}

fn allocate_with(
    exec: &Exec<'_>,
    times: &[Duration],
    (family, selector, keys): (OpKind, LaneAllocation, &[GrantKey]),
    mean_work: f64,
    scratch: &mut AllocationScratch,
    cache: &LaneAllocationCache,
    grant: Option<impl Fn(usize, Duration, usize) -> usize>,
) -> Allocation {
    let jobs = times.len();
    let budget = exec.budget();
    let pool = std::mem::take(&mut scratch.lanes);
    // The keys are staged only where a search can run.
    let query = (grant.is_some()
        && exec.batch_policy().strategy == BatchStrategy::Auto
        && !keys.is_empty())
    .then_some(LaneAllocationQuery {
        family,
        selector,
        exec,
        times,
        keys,
        mean_work,
    });
    let epoch = symtn_core::core_reset_epoch();
    if let Some(entry) = query.as_ref().and_then(|query| cache.get(query)) {
        let mut lanes = lanes_from(pool, &mut scratch.spare, entry.lanes.len());
        let mut start = 0;
        for (lane, &(budget, end)) in lanes.iter_mut().zip(&entry.lanes) {
            lane.budget = budget;
            lane.jobs.extend_from_slice(&entry.jobs[start..end]);
            start = end;
        }
        return Allocation {
            budget,
            slots: entry.slots,
            decision: entry.decision,
            lanes,
        };
    }
    let decision = exec.batch_decision(&BatchQuery::new(jobs, mean_work));
    if jobs == 0 {
        return Allocation {
            budget,
            slots: 0,
            decision,
            lanes: lanes_from(pool, &mut scratch.spare, 0),
        };
    }
    let (slots, lanes) = if exec.batch_policy().strategy == BatchStrategy::Auto {
        let total = times
            .iter()
            .fold(Duration::ZERO, |sum, &t| sum.saturating_add(t));
        let slots = exec.token_ref().width_for(total, 1).get();
        (slots, slots.min(jobs).min(decision.lanes))
    } else {
        (
            decision.lanes * decision.item_threads,
            decision.lanes.min(jobs),
        )
    };
    let lanes = lanes.max(1);
    let mut assigned = lanes_from(pool, &mut scratch.spare, lanes);
    if lanes == 1 {
        assigned[0].budget = slots;
        assigned[0].jobs.extend(0..jobs);
        return Allocation {
            budget,
            slots,
            decision,
            lanes: assigned,
        };
    }

    let mut order = std::mem::take(&mut scratch.order);
    order.clear();
    order.extend(0..jobs);
    order.sort_unstable_by_key(|&j| (Reverse(times[j]), j));
    // LPT, ties to the lowest lane. The job count breaks load ties first so
    // zero-time estimates still leave no reserved lane empty.
    let mut load = std::mem::take(&mut scratch.load);
    load.clear();
    load.extend((0..lanes).map(|k| Reverse((Duration::ZERO, 0, k))));
    let mut load = BinaryHeap::from(load);
    for &j in &order {
        if let Some(Reverse((sum, count, k))) = load.pop() {
            assigned[k].jobs.push(j);
            load.push(Reverse((sum.saturating_add(times[j]), count + 1, k)));
        }
    }
    let mut load = load.into_vec();
    for (k, lane) in assigned.iter_mut().enumerate() {
        lane.jobs.sort_unstable();
        lane.budget = slots / lanes + usize::from(k < slots % lanes);
    }
    if let (BatchStrategy::Auto, Some(grant)) = (exec.batch_policy().strategy, grant) {
        let fill = |j: usize, width: usize| item_time(exec, times[j], grant(j, times[j], width));
        let mut columns = std::mem::take(&mut scratch.columns);
        let mut narrow = std::mem::take(&mut scratch.narrow);
        columns.begin(times, keys, |j| grant(j, times[j], slots) > 1);
        columns.prepare((slots, lanes), &fill);
        let baseline = assigned
            .iter()
            .map(|lane| {
                lane.jobs.iter().fold(Duration::ZERO, |sum, &j| {
                    sum.saturating_add(columns.get(j, lane.budget))
                })
            })
            .max()
            .unwrap_or(Duration::ZERO);
        let mut best = (baseline, None);
        // Ties keep the baseline, then the larger lane count. Shares only
        // widen as the count falls, so each width's column is filled once.
        for count in (1..=lanes).rev() {
            columns.prepare((slots, count), &fill);
            // A candidate that cannot score strictly lower is not placed:
            // the choice is the one the full sweep makes.
            if columns.lower_bound((slots, count)) >= best.0.as_nanos() {
                continue;
            }
            let item = |j: usize, width: usize| columns.get(j, width);
            let score = equal_shares(slots, count, &order, &item, &mut load, &mut narrow, None);
            if score < best.0 {
                best = (score, Some(count));
            }
        }
        if selector == LaneAllocation::Moldable {
            let mut moldable = std::mem::take(&mut scratch.moldable);
            let candidate = std::mem::take(&mut moldable.lanes);
            let provider = |r: usize, width: usize| grant(r, times[r], width);
            let item_of = |serial: Duration, width: usize| item_time(exec, serial, width);
            let (candidate, score) = moldable.place(
                (slots, lanes),
                &order,
                times,
                &columns,
                (&item_of, &provider),
                candidate,
                &mut scratch.spare,
            );
            // Ties keep the baseline and the equal shares.
            let wins = score.is_some_and(|score| score < best.0);
            moldable.lanes = if wins {
                best.1 = None;
                std::mem::replace(&mut assigned, candidate)
            } else {
                candidate
            };
            scratch.moldable = moldable;
        }
        if let Some(count) = best.1 {
            columns.prepare((slots, count), &fill);
            let item = |j: usize, width: usize| columns.get(j, width);
            assigned = lanes_from(assigned, &mut scratch.spare, count);
            equal_shares(
                slots,
                count,
                &order,
                &item,
                &mut load,
                &mut narrow,
                Some(&mut assigned),
            );
        }
        scratch.columns = columns;
        scratch.narrow = narrow;
        if let Some(query) = query {
            let key = query.to_key();
            let mut ends = Vec::with_capacity(assigned.len());
            let mut placed = Vec::with_capacity(jobs);
            for lane in &assigned {
                placed.extend_from_slice(&lane.jobs);
                ends.push((lane.budget, placed.len()));
            }
            let entry = LaneAllocationEntry {
                slots,
                decision,
                lanes: ends.into(),
                jobs: placed.into(),
            };
            let bytes = charge(&key, &entry);
            cache.publish(&key, Arc::new(entry), bytes, epoch);
        }
    }
    scratch.order = order;
    scratch.load = load;
    debug_assert!(
        assigned.iter().all(|lane| lane.budget > 0)
            && assigned.iter().map(|lane| lane.budget).sum::<usize>() <= slots
    );
    Allocation {
        budget,
        slots,
        decision,
        lanes: assigned,
    }
}

/// Every job's modeled item time at two lane widths, so a width the
/// candidates share is granted once per job, not once per candidate. Jobs
/// whose provider stays serial at the full slot count are queried once and
/// never again.
#[derive(Debug, Default)]
struct Columns {
    /// The width of each column; zero (no lane has it) when unfilled.
    width: [usize; 2],
    items: [Vec<Duration>; 2],
    /// The jobs the provider widens at the full slot count, ascending.
    widened: Vec<usize>,
    /// Each job's representative: the lowest job with its serial time and
    /// grant key, whose grants it shares at every width.
    rep: Vec<usize>,
    /// Jobs ordered by serial time, grant key and index, to find `rep`.
    by_key: Vec<usize>,
    /// Whether each job is widened at the full slot count.
    widens: Vec<bool>,
    /// The sum and the largest of the other jobs' serial times, in ns.
    serial_total: u128,
    serial_largest: u128,
}

impl Columns {
    /// Starts an allocation of jobs with serial times `times` and grant keys
    /// `keys`: both columns hold every job's serial time, each job's
    /// representative is the lowest job with its serial time and key (whose
    /// grants it shares at every width, so only representatives are
    /// queried), and `widens(j)` (the provider's grant at the full slot
    /// count S is above one), asked of representatives only, splits the jobs.
    ///
    /// Why a job serial at S is serial at every share `w <= S`: each step of
    /// a grant is a released width rule that starts at width one and moves
    /// only to a width `k` in `2..=b` that both pays and strictly lowers its
    /// objective, where the payoff and the objective at `k` depend on the
    /// token's dispatch cost at `k` alone, not on the ceiling `b`
    /// (threadteam `width_for_by`, reached through `Exec::width_for_within`
    /// and `batch_decision_within`). A ceiling below `b` scans a prefix of
    /// the same candidates, so if none qualifies under `b` none qualifies
    /// under it either; and when the minimum under `b` sits at `q <= b'`
    /// it is also the minimum under `b'`, so a step's result never grows
    /// as its ceiling falls. The factorization grant is one such step. The
    /// GEMM grant's second step runs within the first step's result, which
    /// is thus at most its value at S: a Faer plan's one-item decision is
    /// its inner width rule or one thread (`faer_auto`, one item, no lanes),
    /// and a Packed plan's is its narrowed width (`packed_auto`, one item,
    /// no lanes), both width rules within that smaller ceiling. Hence a
    /// grant of one at S is one at every share, and such a job's item time is
    /// its serial time at every candidate width.
    fn begin(&mut self, times: &[Duration], keys: &[GrantKey], widens: impl Fn(usize) -> bool) {
        self.width = [0; 2];
        for column in &mut self.items {
            column.clear();
            column.extend_from_slice(times);
        }
        self.by_key.clear();
        self.by_key.extend(0..times.len());
        self.by_key
            .sort_unstable_by_key(|&j| (times[j], keys[j], j));
        self.rep.clear();
        self.rep.resize(times.len(), 0);
        let mut first = None;
        for &j in &self.by_key {
            let start = match first {
                Some(f) if times[f] == times[j] && keys[f] == keys[j] => f,
                _ => j,
            };
            first = Some(start);
            self.rep[j] = start;
        }
        self.widened.clear();
        self.widens.clear();
        (self.serial_total, self.serial_largest) = (0, 0);
        for (j, t) in times.iter().enumerate() {
            let rep = self.rep[j];
            let wide = if rep == j {
                widens(j)
            } else {
                self.widens[rep]
            };
            self.widens.push(wide);
            if wide {
                self.widened.push(j);
            } else {
                self.serial_total = self.serial_total.saturating_add(t.as_nanos());
                self.serial_largest = self.serial_largest.max(t.as_nanos());
            }
        }
    }

    /// Keeps the columns of the widths of `count` equal shares of `slots`
    /// (`count <= slots`) and fills a missing one's widened jobs with
    /// `fill(job, width)`; the other jobs keep their serial time.
    fn prepare(
        &mut self,
        (slots, count): (usize, usize),
        fill: &impl Fn(usize, usize) -> Duration,
    ) {
        let narrow = slots / count;
        let widths = [narrow, narrow + 1];
        let widths = &widths[..1 + usize::from(slots % count != 0)];
        for &width in widths {
            if self.width.contains(&width) {
                continue;
            }
            let slot = usize::from(widths.contains(&self.width[0]));
            self.width[slot] = width;
            for &j in &self.widened {
                let rep = self.rep[j];
                self.items[slot][j] = if rep == j {
                    fill(j, width)
                } else {
                    self.items[slot][rep]
                };
            }
        }
    }

    /// A lower bound, in nanoseconds, on the score of the equal-share
    /// placement over `count` lanes of `slots` whose widths' columns are
    /// prepared: `max(max_j m_j, ceil(sum_j m_j / count))`, where `m_j` is
    /// job `j`'s smaller item time over the two share widths.
    ///
    /// Proof. Every job runs on exactly one lane, whose width is one of the
    /// two shares, so its item time there is at least `m_j` (the minimum,
    /// because the item time need not fall as the width grows). A lane's load
    /// is the sum of its jobs' item times, so the largest load is at least
    /// any one job's `m_j`, and at least the mean load, which is at least
    /// `sum_j m_j / count`; a load is whole nanoseconds, hence the ceiling.
    /// The placement saturates its loads at `Duration::MAX`, so its score is
    /// at least `min(bound, Duration::MAX)`; any score it is compared with
    /// is at most `Duration::MAX`, so `bound >= score` still proves the
    /// candidate cannot score strictly lower.
    /// The serial jobs' part is their precomputed sum and maximum, so the
    /// bound costs one step per widened job.
    fn lower_bound(&self, (slots, count): (usize, usize)) -> u128 {
        let narrow = slots / count;
        let wide = narrow + usize::from(slots % count != 0);
        let (mut largest, mut total) = (self.serial_largest, self.serial_total);
        for &job in &self.widened {
            let m = self.get(job, narrow).min(self.get(job, wide)).as_nanos();
            largest = largest.max(m);
            total = total.saturating_add(m);
        }
        largest.max(total.div_ceil(count as u128))
    }

    fn get(&self, job: usize, width: usize) -> Duration {
        debug_assert!(self.width.contains(&width));
        self.items[usize::from(self.width[0] != width)][job]
    }

    fn retained_bytes(&self) -> usize {
        self.items
            .iter()
            .map(|column| column.capacity())
            .fold(0, usize::saturating_add)
            .saturating_mul(std::mem::size_of::<Duration>())
            .saturating_add(
                (self.widened.capacity())
                    .saturating_add(self.rep.capacity())
                    .saturating_add(self.by_key.capacity())
                    .saturating_mul(std::mem::size_of::<usize>()),
            )
            .saturating_add(self.widens.capacity())
    }
}

/// Candidate M, large-job reservation: its target search tables and its
/// lanes, kept for reuse.
#[derive(Debug, Default)]
struct Moldable {
    /// Row `i` holds the item time of the `i`-th largest job at every width
    /// `1..=S`, for the jobs above the mean load only.
    table: Vec<Duration>,
    targets: Vec<Duration>,
    /// Each reserved lane's load and job count.
    wide: Vec<(Duration, usize)>,
    /// The width-one lanes, by `(load, jobs, lane)`.
    thin: Vec<Reverse<(Duration, usize, usize)>>,
    lanes: Vec<Lane>,
    grants: Grants,
}

/// The grants the reservation queried, each at most once per
/// (representative, width), in dense tables, so their capacities are their
/// whole allocation. A zero cell is not yet queried (no width rule grants
/// zero slots; such a grant would only be queried again).
#[derive(Debug, Default)]
struct Grants {
    /// S, the length of a `heavy` row.
    slots: usize,
    /// Each representative's row: below `heavy_rows` a row of `heavy`,
    /// otherwise of `tail`; `usize::MAX` for the jobs never queried.
    row: Vec<usize>,
    heavy_rows: usize,
    tail_rows: usize,
    /// The jobs above the mean load: every width `1..=S`, row-major.
    heavy: Vec<usize>,
    /// Each width's column of `tail`, or `usize::MAX`.
    column: Vec<usize>,
    /// The other widened jobs, column-major over the widths they meet.
    tail: Vec<usize>,
}

impl Grants {
    /// Gives a row to each widened representative: a `heavy` row to those
    /// of the `heavy` jobs, a `tail` row to the others.
    fn begin(&mut self, slots: usize, heavy: &[usize], columns: &Columns) {
        let jobs = columns.rep.len();
        self.slots = slots;
        self.row.clear();
        self.row.resize(jobs, usize::MAX);
        self.heavy_rows = 0;
        for &j in heavy {
            let r = columns.rep[j];
            if columns.widens[j] && self.row[r] == usize::MAX {
                self.row[r] = self.heavy_rows;
                self.heavy_rows += 1;
            }
        }
        self.tail_rows = 0;
        for j in 0..jobs {
            if columns.widens[j] && columns.rep[j] == j && self.row[j] == usize::MAX {
                self.row[j] = self.heavy_rows + self.tail_rows;
                self.tail_rows += 1;
            }
        }
        self.heavy.clear();
        self.heavy.resize(self.heavy_rows * slots, 0);
        self.column.clear();
        self.column.resize(slots + 1, usize::MAX);
        self.tail.clear();
    }

    /// Job `j`'s grant at `width <= S`: one slot for a job the provider keeps
    /// serial at S (see [`Columns::begin`]), otherwise its representative's
    /// grant, queried with `query(representative, width)` on first use.
    fn get(
        &mut self,
        columns: &Columns,
        j: usize,
        width: usize,
        query: &impl Fn(usize, usize) -> usize,
    ) -> usize {
        if !columns.widens[j] {
            return 1;
        }
        let r = columns.rep[j];
        let row = self.row[r];
        let cell = if row < self.heavy_rows {
            &mut self.heavy[row * self.slots + width - 1]
        } else {
            if self.column[width] == usize::MAX {
                self.column[width] = self.tail.len() / self.tail_rows;
                self.tail.resize(self.tail.len() + self.tail_rows, 0);
            }
            &mut self.tail[self.column[width] * self.tail_rows + row - self.heavy_rows]
        };
        if *cell == 0 {
            *cell = query(r, width);
        }
        *cell
    }

    fn retained_bytes(&self) -> usize {
        [&self.row, &self.heavy, &self.column, &self.tail]
            .into_iter()
            .map(Vec::capacity)
            .fold(0, usize::saturating_add)
            .saturating_mul(std::mem::size_of::<usize>())
    }
}

/// `n` nanoseconds as a duration, saturating.
fn from_nanos(n: u128) -> Duration {
    Duration::new(
        u64::try_from(n / 1_000_000_000).unwrap_or(u64::MAX),
        (n % 1_000_000_000) as u32,
    )
}

impl Moldable {
    /// Places the jobs, `order` largest first, on lanes of uneven widths
    /// over `slots`, and returns the lanes with the modeled makespan, or no
    /// score when more than `cap` lanes hold jobs (the batch decision's lane
    /// count, which the candidate must not exceed). `query(r, w)` is the
    /// provider's width for representative job `r` (see `columns`) on a lane
    /// of `w` slots, queried at most once per pair, and `item_of(t, q)` the
    /// modeled time of a job of serial time `t` granted `q` slots; `lanes` is
    /// empty on entry.
    ///
    /// 1. The target's lower bound is `LB = max(ceil(sum t / S), max_j
    ///    min_w item(j, w))`. Only jobs above the mean load can raise the
    ///    second term, and fewer than S jobs are above it.
    /// 2. For a target `T >= LB`, each job with `t_j > T` needs the least
    ///    width `p_j(T)` with `item(j, p_j(T)) <= T`; the others take one
    ///    slot. `T` passes when the needed widths sum to at most S, which is
    ///    monotone in `T`, so a binary search over `LB` and the item times
    ///    at least `LB` finds the least passing target (the largest job's
    ///    serial time, if above `LB`, passes with no reservation).
    /// 3. Each job above the target gets a lane of its width, the remaining
    ///    slots become one-slot lanes, and every other job, largest first,
    ///    goes to the lane minimising `(load + item, jobs, lane)`.
    /// 4. Empty one-slot lanes are dropped, and their slots widen the
    ///    reserved lanes, largest job first, while that job's grant still
    ///    grows. Any slot left stays idle.
    ///
    /// The score is the completed placement's, not the target: a passing
    /// target does not prove that the tail fits under it.
    #[allow(clippy::too_many_arguments)]
    fn place(
        &mut self,
        (slots, cap): (usize, usize),
        order: &[usize],
        times: &[Duration],
        columns: &Columns,
        (item_of, query): (
            &impl Fn(Duration, usize) -> Duration,
            &impl Fn(usize, usize) -> usize,
        ),
        mut lanes: Vec<Lane>,
        spare: &mut Vec<Lane>,
    ) -> (Vec<Lane>, Option<Duration>) {
        let Self {
            table,
            targets,
            wide,
            thin,
            grants,
            ..
        } = self;
        let grant = |g: &mut Grants, j: usize, w: usize| g.get(columns, j, w, query);
        let item = |g: &mut Grants, j: usize, w: usize| item_of(times[j], grant(g, j, w));
        let total = times
            .iter()
            .fold(0u128, |sum, t| sum.saturating_add(t.as_nanos()));
        let mean = total.div_ceil(slots as u128);
        let heavy = order
            .iter()
            .take_while(|&&j| times[j].as_nanos() > mean)
            .count();
        grants.begin(slots, &order[..heavy], columns);
        table.clear();
        for &j in &order[..heavy] {
            for width in 1..=slots {
                table.push(item(grants, j, width));
            }
        }
        let table = &*table;
        let row = |i: usize| &table[i * slots..(i + 1) * slots];
        let bound = (0..heavy)
            .filter_map(|i| row(i).iter().min())
            .fold(from_nanos(mean), |bound, &least| bound.max(least));
        let large = order[..heavy]
            .iter()
            .take_while(|&&j| times[j] > bound)
            .count();
        targets.clear();
        targets.push(bound);
        for i in 0..large {
            targets.extend(row(i).iter().filter(|&&t| t >= bound));
        }
        targets.sort_unstable();
        targets.dedup();
        // Every target is at least each large job's least item time.
        let need = |i: usize, target: Duration| {
            1 + row(i)
                .iter()
                .position(|&t| t <= target)
                .unwrap_or(slots - 1)
        };
        let reserved = |target: Duration| {
            order[..large]
                .iter()
                .take_while(|&&j| times[j] > target)
                .count()
        };
        let passes = |target: Duration| {
            (0..reserved(target))
                .map(|i| need(i, target))
                .fold(0usize, usize::saturating_add)
                <= slots
        };
        let target = targets[targets.partition_point(|&t| !passes(t))];
        let reserved = reserved(target);
        let used: usize = (0..reserved).map(|i| need(i, target)).sum();

        let count = reserved + (slots - used);
        lanes = lanes_from(lanes, spare, count);
        wide.clear();
        for (i, (lane, &j)) in lanes.iter_mut().zip(order).take(reserved).enumerate() {
            let width = need(i, target);
            lane.budget = width;
            lane.jobs.push(j);
            wide.push((row(i)[width - 1], 1));
        }
        let mut heap = BinaryHeap::from(std::mem::take(thin));
        heap.clear();
        for (k, lane) in lanes.iter_mut().enumerate().skip(reserved) {
            lane.budget = 1;
            heap.push(Reverse((Duration::ZERO, 0, k)));
        }
        for &j in &order[reserved..] {
            let mut best = heap.peek().map(|&Reverse((load, jobs, k))| {
                (load.saturating_add(item(grants, j, 1)), jobs, k)
            });
            for (k, &(load, jobs)) in wide.iter().enumerate() {
                let key = (
                    load.saturating_add(item(grants, j, lanes[k].budget)),
                    jobs,
                    k,
                );
                best = Some(best.map_or(key, |best| best.min(key)));
            }
            let Some((load, jobs, k)) = best else {
                unreachable!("a target passes with at least one lane")
            };
            if k < reserved {
                wide[k] = (load, jobs + 1);
            } else {
                heap.pop();
                heap.push(Reverse((load, jobs + 1, k)));
            }
            lanes[k].jobs.push(j);
        }
        let mut idle = heap
            .iter()
            .filter(|Reverse((_, jobs, _))| *jobs == 0)
            .count();
        let mut score = heap
            .iter()
            .map(|&Reverse((load, _, _))| load)
            .max()
            .unwrap_or(Duration::ZERO);
        *thin = heap.into_vec();
        if count - idle > cap {
            return (lanes, None);
        }
        for (lane, &j) in lanes.iter_mut().zip(order).take(reserved) {
            while idle > 0 && grant(grants, j, lane.budget + 1) > grant(grants, j, lane.budget) {
                lane.budget += 1;
                idle -= 1;
            }
            let load = lane.jobs.iter().fold(Duration::ZERO, |sum, &j| {
                sum.saturating_add(item(grants, j, lane.budget))
            });
            score = score.max(load);
        }
        let mut kept = 0;
        for k in 0..lanes.len() {
            if !lanes[k].jobs.is_empty() {
                lanes.swap(kept, k);
                lanes[kept].jobs.sort_unstable();
                kept += 1;
            }
        }
        spare.extend(lanes.drain(kept..));
        (lanes, Some(score))
    }

    fn retained_bytes(&self) -> usize {
        use std::mem::size_of;
        self.table
            .capacity()
            .saturating_add(self.targets.capacity())
            .saturating_mul(size_of::<Duration>())
            .saturating_add(
                self.wide
                    .capacity()
                    .saturating_mul(size_of::<(Duration, usize)>()),
            )
            .saturating_add(
                self.thin
                    .capacity()
                    .saturating_mul(size_of::<Reverse<(Duration, usize, usize)>>()),
            )
            .saturating_add(self.grants.retained_bytes())
            .saturating_add(
                self.lanes
                    .iter()
                    .map(|lane| lane.jobs.capacity().saturating_mul(size_of::<usize>()))
                    .fold(
                        self.lanes.capacity().saturating_mul(size_of::<Lane>()),
                        usize::saturating_add,
                    ),
            )
    }
}

/// The modeled makespan of the equal-share placement over `count` lanes of
/// `slots`: lane `k` reserves `slots / count + [k < slots % count]`, the jobs
/// in `order` seed one lane each (the largest on the widest), and every later
/// job goes to the lane minimising `(load + item, jobs, lane)`. With `record`,
/// the placement is written into its (empty) lanes, jobs ascending.
///
/// Lanes of one width share each job's item time, so each width class keeps
/// one heap ordered by `(load, jobs, lane)` and a job compares the two tops.
fn equal_shares(
    slots: usize,
    count: usize,
    order: &[usize],
    item: &impl Fn(usize, usize) -> Duration,
    wide_heap: &mut Vec<Reverse<(Duration, usize, usize)>>,
    narrow_heap: &mut Vec<Reverse<(Duration, usize, usize)>>,
    mut record: Option<&mut Vec<Lane>>,
) -> Duration {
    let (narrow, wide_lanes) = (slots / count, slots % count);
    let width = |k: usize| narrow + usize::from(k < wide_lanes);
    let mut wide = BinaryHeap::from(std::mem::take(wide_heap));
    let mut thin = BinaryHeap::from(std::mem::take(narrow_heap));
    wide.clear();
    thin.clear();
    for (k, &j) in order.iter().take(count).enumerate() {
        let entry = Reverse((item(j, width(k)), 1, k));
        if k < wide_lanes {
            wide.push(entry);
        } else {
            thin.push(entry);
        }
        if let Some(lanes) = record.as_deref_mut() {
            lanes[k].jobs.push(j);
        }
    }
    for &j in order.iter().skip(count) {
        let wide_key = wide
            .peek()
            .map(|&Reverse((load, jobs, k))| (load.saturating_add(item(j, narrow + 1)), jobs, k));
        let thin_key = thin
            .peek()
            .map(|&Reverse((load, jobs, k))| (load.saturating_add(item(j, narrow)), jobs, k));
        let (heap, key) = match (wide_key, thin_key) {
            (Some(a), Some(b)) if a <= b => (&mut wide, a),
            (Some(a), None) => (&mut wide, a),
            (_, Some(b)) => (&mut thin, b),
            (None, None) => unreachable!("count >= 1 seeds a lane"),
        };
        heap.pop();
        let (load, jobs, k) = key;
        heap.push(Reverse((load, jobs + 1, k)));
        if let Some(lanes) = record.as_deref_mut() {
            lanes[k].jobs.push(j);
        }
    }
    if let Some(lanes) = record {
        for (k, lane) in lanes.iter_mut().enumerate() {
            lane.jobs.sort_unstable();
            lane.budget = width(k);
        }
    }
    let score = wide
        .iter()
        .chain(thin.iter())
        .map(|&Reverse((load, _, _))| load)
        .max()
        .unwrap_or(Duration::ZERO);
    *wide_heap = wide.into_vec();
    *narrow_heap = thin.into_vec();
    score
}

/// The jobs of one lane, visited in ascending index order with exclusive
/// access to each job's state.
pub(crate) trait LaneJobs<S> {
    fn for_each(&mut self, visit: &mut dyn FnMut(usize, &mut S));
}

/// Every state of a one-lane run, at its own index.
struct AllJobs<'x, S>(&'x mut [S]);

impl<S> LaneJobs<S> for AllJobs<'_, S> {
    fn for_each(&mut self, visit: &mut dyn FnMut(usize, &mut S)) {
        for (j, state) in self.0.iter_mut().enumerate() {
            visit(j, state);
        }
    }
}

/// One lane's share of a multi-lane run.
struct PartJobs<'x, 'y, S>(&'x mut [(usize, &'y mut S)]);

impl<S> LaneJobs<S> for PartJobs<'_, '_, S> {
    fn for_each(&mut self, visit: &mut dyn FnMut(usize, &mut S)) {
        for (j, state) in self.0.iter_mut() {
            visit(*j, state);
        }
    }
}

/// [`run_with_lanes`] without lane-local state.
#[cfg(test)]
pub(crate) fn run<S: Send, E: Send>(
    exec: Exec<'_>,
    allocation: &Allocation,
    states: &mut [S],
    job: impl Fn(usize, &mut S, Exec<'_>) -> Result<(), E> + Sync,
) -> Result<Result<(), (usize, E)>, tenferro_cpu::threadteam::Error> {
    run_with_lanes(
        exec,
        allocation,
        states,
        |_, _| (),
        |j, state, (), exec| job(j, state, exec),
    )
}

/// Run every job of `allocation` on `exec` with exclusive access to its own
/// `states` entry, and return the lowest canonical failure.
///
/// Every job runs even after another fails, so each job below the returned
/// index has completed. Lane `k` gets a child token of exactly
/// `allocation.lanes[k].budget` slots (one threadteam `fork_sized` over S;
/// slots no lane reserves stay idle for the call) with the caller's workspace
/// and policy. Before its first job, each lane builds its own state once with
/// `lane` from its jobs and its child context; every job of that lane borrows
/// it, and it drops when the lane finishes, before the lanes join. A
/// panicking job unwinds after the lanes join; the engine's guarded entry
/// then marks itself broken.
///
/// The outer error is threadteam's refusal of the lane budgets, before any
/// job ran. An allocation's budgets are nonzero and sum to at most S by
/// construction, so it is unreachable; it is still returned, never replaced
/// by an equal split.
///
/// Why per-job state is lent by lane rather than behind a lock per job: a
/// lock per job costs an allocation per job on platforms whose mutex is
/// boxed, and a lane already owns its jobs exclusively.
pub(crate) fn run_with_lanes<S: Send, L, E: Send>(
    exec: Exec<'_>,
    allocation: &Allocation,
    states: &mut [S],
    lane: impl Fn(&mut dyn LaneJobs<S>, &Exec<'_>) -> L + Sync,
    job: impl Fn(usize, &mut S, &mut L, Exec<'_>) -> Result<(), E> + Sync,
) -> Result<Result<(), (usize, E)>, tenferro_cpu::threadteam::Error> {
    let lanes = &allocation.lanes;
    if lanes.is_empty() {
        return Ok(Ok(()));
    }
    let policy = *exec.batch_policy();
    let workspace = exec.workspace();
    // A lane's jobs ascend, so its first failure is its lowest one.
    let run_lane = |token: tenferro_cpu::threadteam::Exec<'_>, jobs: &mut dyn LaneJobs<S>| {
        let mut lane_exec = Exec::from_token(token, workspace, policy);
        let mut local = lane(jobs, &lane_exec);
        let mut first = None;
        jobs.for_each(&mut |j, state| {
            if let Err(error) = job(j, state, &mut local, lane_exec.reborrow()) {
                first.get_or_insert((j, error));
            }
        });
        first
    };
    if let [lane] = lanes.as_slice() {
        debug_assert_eq!(lane.jobs.len(), states.len());
        let width =
            NonZeroUsize::new(lane.budget).ok_or(tenferro_cpu::threadteam::Error::ZeroBudget)?;
        let token = exec.limit(width).into_token();
        return Ok(run_lane(token, &mut AllJobs(states)).map_or(Ok(()), Err));
    }
    let slots = NonZeroUsize::new(allocation.slots).unwrap_or(NonZeroUsize::MIN);
    let token = exec.limit(slots).into_token();
    let mut lane_of = vec![0; states.len()];
    let mut shares = Vec::with_capacity(lanes.len());
    for (k, lane) in lanes.iter().enumerate() {
        shares.push(lane.budget);
        for &j in &lane.jobs {
            lane_of[j] = k;
        }
    }
    let mut parts: Vec<Vec<(usize, &mut S)>> = lanes
        .iter()
        .map(|lane| Vec::with_capacity(lane.jobs.len()))
        .collect();
    for (j, state) in states.iter_mut().enumerate() {
        parts[lane_of[j]].push((j, state));
    }
    let parts: Vec<Mutex<_>> = parts
        .into_iter()
        .map(|part| Mutex::new((part, None)))
        .collect();
    token.fork_sized(&shares, |k, token| {
        let mut part = parts[k].lock().unwrap_or_else(PoisonError::into_inner);
        let (jobs, failure) = &mut *part;
        *failure = run_lane(token, &mut PartJobs(jobs));
    })?;
    Ok(parts
        .into_iter()
        .filter_map(|m| m.into_inner().unwrap_or_else(PoisonError::into_inner).1)
        .min_by_key(|&(j, _)| j)
        .map_or(Ok(()), Err))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tcontract_exec::BatchPolicy;
    use tenferro_cpu::{CpuEngine, Pool};

    fn engine(threads: usize, policy: BatchPolicy) -> CpuEngine {
        CpuEngine::new(Pool::builder().threads(threads).build().unwrap()).with_batch_policy(policy)
    }

    fn split(lanes: usize) -> BatchPolicy {
        BatchPolicy::new(BatchStrategy::Split {
            lanes: NonZeroUsize::new(lanes).unwrap(),
        })
    }

    /// Large enough that every width pays, so Auto's slots are the whole budget.
    fn secs(s: &[u64]) -> Vec<Duration> {
        s.iter().map(|&s| Duration::from_secs(s)).collect()
    }

    fn allocation(
        threads: usize,
        policy: BatchPolicy,
        times: &[Duration],
        work: f64,
    ) -> Allocation {
        engine(threads, policy)
            .with_contraction_exec(|exec, _, _| Ok(allocate(exec, times, work)))
            .unwrap()
    }

    fn budgets(a: &Allocation) -> Vec<usize> {
        a.lanes.iter().map(|l| l.budget).collect()
    }

    #[test]
    fn auto_p8_three_lanes_share_all_slots_largest_first() {
        let a = allocation(8, BatchPolicy::default(), &secs(&[1, 3, 2]), 1e9);
        assert_eq!((a.budget, a.slots, a.decision.lanes), (8, 8, 3));
        assert_eq!(budgets(&a), [3, 3, 2]);
        let jobs: Vec<_> = a.lanes.iter().map(|l| l.jobs.clone()).collect();
        assert_eq!(jobs, [vec![1], vec![2], vec![0]]);
    }

    #[test]
    fn forced_policies_reserve_their_decision() {
        let five = secs(&[5, 4, 3, 2, 1]);
        let a = allocation(8, split(3), &five, 1e9);
        assert_eq!((a.slots, budgets(&a)), (6, vec![2, 2, 2]));
        let a = allocation(5, split(2), &five, 1e9);
        assert_eq!((a.slots, budgets(&a)), (4, vec![2, 2]));
        let a = allocation(
            8,
            BatchPolicy::new(BatchStrategy::Outer),
            &secs(&[1, 1, 1]),
            1e9,
        );
        assert_eq!((a.slots, budgets(&a)), (3, vec![1, 1, 1]));
        let a = allocation(8, BatchPolicy::new(BatchStrategy::Inner), &five, 1e9);
        assert_eq!((a.slots, budgets(&a)), (8, vec![8]));
        assert_eq!(a.lanes[0].jobs, [0, 1, 2, 3, 4]);
        let a = allocation(8, BatchPolicy::new(BatchStrategy::Sequential), &five, 1e9);
        assert_eq!((a.slots, budgets(&a)), (1, vec![1]));
    }

    #[test]
    fn single_job_and_empty() {
        let a = allocation(8, BatchPolicy::default(), &secs(&[7]), 1e9);
        assert_eq!((a.slots, budgets(&a)), (8, vec![8]));
        let a = allocation(8, BatchPolicy::new(BatchStrategy::Outer), &secs(&[7]), 1e9);
        assert_eq!(budgets(&a), [1]);
        let zeros = allocation(
            8,
            BatchPolicy::new(BatchStrategy::Outer),
            &[Duration::ZERO; 3],
            0.0,
        );
        assert!(zeros.lanes.iter().all(|l| l.jobs.len() == 1));
        let a = allocation(8, BatchPolicy::default(), &[], 0.0);
        assert!(a.lanes.is_empty());
        let mut engine = engine(8, BatchPolicy::default());
        let calls = AtomicUsize::new(0);
        let result = engine.with_contraction_exec(|exec, _, _| {
            Ok(run(exec.reborrow(), &a, &mut [(); 0], |_, _, _| {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok::<(), ()>(())
            }))
        });
        assert_eq!(result.unwrap(), Ok(Ok(())));
        assert_eq!(calls.into_inner(), 0);
    }

    /// Review R1: the query takes mean multiply-adds and `width_for` takes
    /// nanoseconds. Two real 48x48 SVDs (w = 110592 each) at 15.6 per ns and a
    /// caller-set 1000 ns lane minimum give two lanes; converting the work to
    /// time a second time would give one.
    #[test]
    fn r1_min_work_lane_gate_uses_work_not_time() {
        let model = WorkModel::default()
            .with_muladds_per_ns(15.6)
            .with_item_overhead_ns(0.0)
            .with_min_work_per_lane_ns(1000.0);
        let work = 110_592.0;
        let t = estimate(work_model_ns(&model, work));
        assert_eq!(t.as_nanos(), 7089);
        let policy = BatchPolicy::default().with_model(model);
        let a = allocation(8, policy, &[t, t], work);
        assert_eq!(a.decision.lanes, 2);
        let double_converted = allocation(8, policy, &[t, t], work / 15.6);
        assert_eq!(double_converted.decision.lanes, 1);
    }

    #[test]
    fn rate_zero_or_nan_counts_as_one_and_nan_time_is_zero() {
        let rate = |r: f64| WorkModel::default().with_muladds_per_ns(r);
        let reference = work_model_ns(&rate(1.0), 4096.0);
        assert_eq!(work_model_ns(&rate(0.0), 4096.0), reference);
        assert_eq!(work_model_ns(&rate(f64::NAN), 4096.0), reference);
        assert_eq!(estimate(f64::NAN), Duration::ZERO);
        assert_eq!(estimate(-1.0), Duration::ZERO);
        assert_eq!(estimate(f64::INFINITY), Duration::from_secs(u64::MAX / 4));
        for r in [0.0, f64::NAN] {
            let policy = BatchPolicy::default().with_model(rate(r));
            let times = [
                estimate(f64::NAN),
                estimate(work_model_ns(&rate(r), 4096.0)),
            ];
            let a = allocation(4, policy, &times, 4096.0);
            let b = allocation(
                4,
                BatchPolicy::default().with_model(rate(1.0)),
                &times,
                4096.0,
            );
            assert_eq!(a, b);
        }
    }

    #[derive(Debug, PartialEq)]
    enum Injected {
        Validation(usize),
        Numeric(usize),
    }

    #[test]
    fn lowest_canonical_error_and_every_job_runs() {
        let times = secs(&[1, 6, 2, 5, 3, 4]);
        let mut engine = engine(4, BatchPolicy::new(BatchStrategy::Outer));
        let ran: Vec<AtomicUsize> = (0..times.len()).map(|_| AtomicUsize::new(0)).collect();
        let result = engine
            .with_contraction_exec(|exec, _, _| {
                let a = allocate(exec, &times, 1e9);
                assert_eq!(a.lanes.len(), 4);
                Ok(run(
                    exec.reborrow(),
                    &a,
                    &mut vec![(); times.len()],
                    |j, _, _| {
                        ran[j].fetch_add(1, Ordering::Relaxed);
                        match j {
                            4 => Err(Injected::Validation(j)),
                            2 => Err(Injected::Numeric(j)),
                            5 => Err(Injected::Validation(j)),
                            _ => Ok(()),
                        }
                    },
                ))
            })
            .unwrap();
        assert_eq!(result, Ok(Err((2, Injected::Numeric(2)))));
        assert!(ran.iter().all(|r| r.load(Ordering::Relaxed) == 1));
    }

    #[test]
    fn lane_panic_unwinds_after_join_and_breaks_the_engine() {
        let times = secs(&[1, 1, 1, 1]);
        let mut engine = engine(4, BatchPolicy::new(BatchStrategy::Outer));
        let ran = AtomicUsize::new(0);
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            engine.with_contraction_exec(|exec, _, _| {
                let a = allocate(exec, &times, 1e9);
                Ok(run(
                    exec.reborrow(),
                    &a,
                    &mut vec![(); times.len()],
                    |j, _, _| {
                        assert_ne!(j, 2, "injected lane panic");
                        ran.fetch_add(1, Ordering::Relaxed);
                        Ok::<(), ()>(())
                    },
                ))
            })
        }));
        assert!(unwound.is_err());
        assert_eq!(ran.into_inner(), 3);
        assert!(engine.is_broken());
        assert!(engine.with_contraction_exec(|_, _, _| Ok(())).is_err());
    }

    /// Every policy, budget and job count: lane budgets are nonzero and sum to
    /// at most S <= P, every job is placed and run once on a token of its
    /// lane's reserved budget, and the allocation is reproducible. Uneven
    /// budgets run as reserved: on four slots (flat 100 ns dispatch, a job
    /// taking its whole lane from 300 ns) jobs of 100, 100, 1000 and 2000 ns
    /// reserve three slots for the largest and one for the 1000 ns job, and
    /// the jobs see three and one slot, not an even two and two.
    #[test]
    fn allocation_is_total_legal_and_deterministic() {
        let mut seed = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % 1_000
        };
        for p in 1..=8 {
            let policies = [
                BatchPolicy::default(),
                BatchPolicy::new(BatchStrategy::Sequential),
                BatchPolicy::new(BatchStrategy::Inner),
                BatchPolicy::new(BatchStrategy::Outer),
                split(1),
                split(3),
                split(16),
            ];
            for policy in policies {
                let mut engine = engine(p, policy);
                for jobs in 0..=11 {
                    let times: Vec<_> = (0..jobs).map(|_| Duration::from_millis(next())).collect();
                    engine
                        .with_contraction_exec(|exec, _, _| {
                            let a = allocate(exec, &times, 1e6);
                            assert_eq!(a, allocate(exec, &times, 1e6));
                            check(&a, p, jobs);
                            let seen: Vec<AtomicUsize> =
                                (0..jobs).map(|_| AtomicUsize::new(0)).collect();
                            let result =
                                run(exec.reborrow(), &a, &mut vec![(); jobs], |j, _, child| {
                                    let lane =
                                        a.lanes.iter().find(|l| l.jobs.contains(&j)).unwrap();
                                    assert_eq!(child.budget(), lane.budget);
                                    assert_eq!(child.batch_policy(), &policy);
                                    seen[j].fetch_add(1, Ordering::Relaxed);
                                    Ok::<(), ()>(())
                                });
                            assert_eq!(result, Ok(Ok(())));
                            assert!(seen.iter().all(|s| s.load(Ordering::Relaxed) == 1));
                            Ok(())
                        })
                        .unwrap();
                }
            }
        }

        let t = nanos(&[100, 100, 1000, 2000]);
        flat(4, BatchPolicy::default(), 100)
            .with_contraction_exec(|exec, _, _| {
                let a = allocate_in(
                    exec,
                    &t,
                    (OpKind::Gemm, LaneAllocation::Moldable, |_| [0; 2]),
                    1e6,
                    &mut AllocationScratch::default(),
                    &uncached(),
                    Some(|_, t: Duration, w| if t.as_nanos() >= 300 { w } else { 1 }),
                );
                assert_eq!(lanes(&a), [(3, vec![0, 1, 3]), (1, vec![2])]);
                let seen: Vec<AtomicUsize> = (0..4).map(|_| AtomicUsize::new(0)).collect();
                let result = run(exec.reborrow(), &a, &mut [(); 4], |j, _, child| {
                    seen[j].store(child.budget(), Ordering::Relaxed);
                    Ok::<(), ()>(())
                });
                assert_eq!(result, Ok(Ok(())));
                let seen: Vec<usize> = seen.into_iter().map(AtomicUsize::into_inner).collect();
                assert_eq!(seen, [3, 3, 1, 3]);
                Ok(())
            })
            .unwrap();
    }

    /// An engine of `threads` whose dispatch cost is `rows` (anchors at 2, 4
    /// and 6, extrapolated above) for every dispatch kind.
    fn engine_with_costs(threads: usize, policy: BatchPolicy, rows: [u64; 3]) -> CpuEngine {
        use tenferro_cpu::threadteam::{DispatchCosts, WidthCosts};
        let row = WidthCosts::new(rows.map(Duration::from_nanos));
        let pool = Pool::builder()
            .threads(threads)
            .dispatch_costs(DispatchCosts::new(row, row, row))
            .build()
            .unwrap();
        CpuEngine::new(pool).with_batch_policy(policy)
    }

    /// A flat dispatch cost `d` ns at every width up to six.
    fn flat(threads: usize, policy: BatchPolicy, d: u64) -> CpuEngine {
        engine_with_costs(threads, policy, [d; 3])
    }

    fn nanos(t: &[u64]) -> Vec<Duration> {
        t.iter().map(|&t| Duration::from_nanos(t)).collect()
    }

    /// The allocation under the factorization grant (one width decision on
    /// the job's time), as the driver's default `BlockOp::grant` makes it.
    fn chosen(engine: &mut CpuEngine, times: &[Duration], mean_work: f64) -> Allocation {
        chosen_with(engine, times, mean_work, LaneAllocation::Moldable)
    }

    /// [`chosen`] under the lane allocation `selector`.
    fn chosen_with(
        engine: &mut CpuEngine,
        times: &[Duration],
        mean_work: f64,
        selector: LaneAllocation,
    ) -> Allocation {
        engine
            .with_contraction_exec(|exec, _, _| {
                let exec: &Exec<'_> = exec;
                Ok(allocate_in(
                    exec,
                    times,
                    (OpKind::Gemm, selector, |_| [0; 2]),
                    mean_work,
                    &mut AllocationScratch::default(),
                    &uncached(),
                    Some(|_, t: Duration, w| exec.width_for_within(w, t.as_nanos() as f64, 1)),
                ))
            })
            .unwrap()
    }

    fn lanes(a: &Allocation) -> Vec<(usize, Vec<usize>)> {
        a.lanes.iter().map(|l| (l.budget, l.jobs.clone())).collect()
    }

    /// A lane-allocation cache of its own, so a test sees no other test's
    /// entries; a budget of zero admits nothing.
    fn cache(budget: u64) -> LaneAllocationCache {
        LaneAllocationCache::new(StructureCacheKind::LaneAllocations, budget)
    }

    fn uncached() -> LaneAllocationCache {
        cache(0)
    }

    /// A test-local model of the scheduler for a flat dispatch cost `d` and
    /// times that are either below the payoff `3 d` or at least `4 d`: a job
    /// takes its whole lane when it pays, else one slot.
    mod oracle {
        pub(super) type Lanes = Vec<(usize, Vec<usize>)>;

        fn item(t: u64, w: usize, d: u64) -> u64 {
            if w >= 2 && t >= 3 * d {
                t / w as u64 + d
            } else {
                t
            }
        }

        fn shares(slots: usize, count: usize) -> Vec<usize> {
            (0..count)
                .map(|k| slots / count + usize::from(k < slots % count))
                .collect()
        }

        fn order(t: &[u64]) -> Vec<usize> {
            let mut order: Vec<usize> = (0..t.len()).collect();
            order.sort_by_key(|&j| (std::cmp::Reverse(t[j]), j));
            order
        }

        fn finish(
            widths: Vec<usize>,
            mut jobs: Vec<Vec<usize>>,
            t: &[u64],
            d: u64,
        ) -> (u64, Lanes) {
            let score = widths
                .iter()
                .zip(&jobs)
                .map(|(&w, js)| js.iter().map(|&j| item(t[j], w, d)).sum::<u64>())
                .max()
                .unwrap();
            jobs.iter_mut().for_each(|js| js.sort());
            (score, widths.into_iter().zip(jobs).collect())
        }

        /// Main's placement: least accumulated serial time, then fewest
        /// jobs, then lowest lane.
        pub(super) fn baseline(t: &[u64], slots: usize, count: usize, d: u64) -> (u64, Lanes) {
            let widths = shares(slots, count);
            let (mut serial, mut jobs) = (vec![0u64; count], vec![Vec::new(); count]);
            for j in order(t) {
                let k = (0..count)
                    .min_by_key(|&k| (serial[k], jobs[k].len(), k))
                    .unwrap();
                serial[k] += t[j];
                jobs[k].push(j);
            }
            finish(widths, jobs, t, d)
        }

        /// Equal shares: the first `count` jobs seed one lane each, then the
        /// least `(load + item, jobs, lane)`.
        pub(super) fn equal(t: &[u64], slots: usize, count: usize, d: u64) -> (u64, Lanes) {
            let widths = shares(slots, count);
            let (mut load, mut jobs) = (vec![0u64; count], vec![Vec::new(); count]);
            for (i, j) in order(t).into_iter().enumerate() {
                let k = if i < count {
                    i
                } else {
                    (0..count)
                        .min_by_key(|&k| (load[k] + item(t[j], widths[k], d), jobs[k].len(), k))
                        .unwrap()
                };
                load[k] += item(t[j], widths[k], d);
                jobs[k].push(j);
            }
            finish(widths, jobs, t, d)
        }

        /// The width a job's provider takes on a lane of `w` slots.
        fn grant(t: u64, w: usize, d: u64) -> usize {
            if w >= 2 && t >= 3 * d {
                w
            } else {
                1
            }
        }

        /// Large-job reservation: the least target T, from `LB = max(ceil(
        /// sum t / S), max_j min_w item)` and the item times above it, at
        /// which the jobs above T fit their least widths reaching T into S;
        /// those jobs take lanes of those widths, the rest of S one-slot
        /// lanes; the other jobs go, largest first, to the least `(load +
        /// item, jobs, lane)`; empty one-slot lanes then widen the reserved
        /// lanes, largest job first, while the grant grows.
        pub(super) fn moldable(t: &[u64], slots: usize, d: u64) -> (u64, Lanes) {
            let items = |j: usize| (1..=slots).map(move |w| item(t[j], w, d));
            let lb = (0..t.len())
                .map(|j| items(j).min().unwrap())
                .fold(t.iter().sum::<u64>().div_ceil(slots as u64), u64::max);
            let mut targets = vec![lb];
            for j in (0..t.len()).filter(|&j| t[j] > lb) {
                targets.extend(items(j).filter(|&x| x >= lb));
            }
            targets.sort();
            let need = |j: usize, target: u64| (1..=slots).find(|&w| item(t[j], w, d) <= target);
            let above = |target: u64| order(t).into_iter().filter(move |&j| t[j] > target);
            let target = targets
                .into_iter()
                .find(|&target| {
                    above(target)
                        .map(|j| need(j, target).unwrap())
                        .sum::<usize>()
                        <= slots
                })
                .unwrap();
            let big: Vec<usize> = above(target).collect();
            let mut widths: Vec<usize> = big.iter().map(|&j| need(j, target).unwrap()).collect();
            let used: usize = widths.iter().sum();
            widths.resize(big.len() + slots - used, 1);
            let mut load: Vec<u64> = big
                .iter()
                .zip(&widths)
                .map(|(&j, &w)| item(t[j], w, d))
                .collect();
            load.resize(widths.len(), 0);
            let mut jobs: Vec<Vec<usize>> = big.iter().map(|&j| vec![j]).collect();
            jobs.resize(widths.len(), Vec::new());
            for j in order(t).into_iter().skip(big.len()) {
                let k = (0..widths.len())
                    .min_by_key(|&k| (load[k] + item(t[j], widths[k], d), jobs[k].len(), k))
                    .unwrap();
                load[k] += item(t[j], widths[k], d);
                jobs[k].push(j);
            }
            let mut idle = jobs.iter().filter(|js| js.is_empty()).count();
            for (k, &j) in big.iter().enumerate() {
                while idle > 0 && grant(t[j], widths[k] + 1, d) > grant(t[j], widths[k], d) {
                    widths[k] += 1;
                    idle -= 1;
                }
            }
            let (widths, jobs) = widths
                .into_iter()
                .zip(jobs)
                .filter(|(_, js)| !js.is_empty())
                .unzip();
            finish(widths, jobs, t, d)
        }

        /// `best`, or the reservation when it scores strictly lower on at
        /// most `cap` lanes and `cap` allows fan-out.
        fn or_moldable(
            best: (u64, Lanes),
            t: &[u64],
            slots: usize,
            cap: usize,
            d: u64,
        ) -> (u64, Lanes) {
            let m = moldable(t, slots, d);
            if cap > 1 && m.1.len() <= cap && m.0 < best.0 {
                m
            } else {
                best
            }
        }

        /// The lowest score over the baseline, every equal-share count up to
        /// `cap` and the reservation; ties keep the baseline, then the larger
        /// count, then the equal shares.
        pub(super) fn choose(t: &[u64], slots: usize, cap: usize, d: u64) -> (u64, Lanes) {
            or_moldable(choose_equal(t, slots, cap, d), t, slots, cap, d)
        }

        /// The lowest score over the baseline and every equal-share count up
        /// to `cap`, without the reservation; ties keep the baseline, then
        /// the larger count.
        pub(super) fn choose_equal(t: &[u64], slots: usize, cap: usize, d: u64) -> (u64, Lanes) {
            let mut best = baseline(t, slots, cap, d);
            for count in (1..=cap).rev() {
                let candidate = equal(t, slots, count, d);
                if candidate.0 < best.0 {
                    best = candidate;
                }
            }
            best
        }

        /// [`choose`] that skips a count whose bound
        /// `max(max_j m_j, ceil(sum_j m_j / count))`, `m_j` the job's least
        /// item time over the count's share widths, is not below the best
        /// score so far; `placed` counts the placements it does make.
        pub(super) fn choose_pruned(
            t: &[u64],
            slots: usize,
            cap: usize,
            d: u64,
            placed: &mut usize,
        ) -> (u64, Lanes) {
            let mut best = baseline(t, slots, cap, d);
            for count in (1..=cap).rev() {
                let widths = shares(slots, count);
                let least = |j: usize| widths.iter().map(|&w| item(t[j], w, d)).min().unwrap();
                let total: u64 = (0..t.len()).map(least).sum();
                let largest = (0..t.len()).map(least).max().unwrap();
                if largest.max(total.div_ceil(count as u64)) >= best.0 {
                    continue;
                }
                *placed += 1;
                let candidate = equal(t, slots, count, d);
                if candidate.0 < best.0 {
                    best = candidate;
                }
            }
            or_moldable(best, t, slots, cap, d)
        }
    }

    /// Purpose: a lane width at which the provider stays serial charges the
    /// job its serial time, not a share plus a dispatch cost. With dispatch
    /// costs 3000/2000/1400 ns at widths 2/4/6, a 5000 ns job does not pay at
    /// width 2 (it needs 9000 ns) but pays at width 6.
    #[test]
    fn item_time_follows_provider_serial_decision() {
        let mut engine = engine_with_costs(6, BatchPolicy::default(), [3000, 2000, 1400]);
        let t = Duration::from_nanos(5000);
        let (two, six) = engine
            .with_contraction_exec(|exec, _, _| {
                let at = |w| item_time(exec, t, exec.width_for_within(w, 5000.0, 1));
                Ok((at(2), at(6)))
            })
            .unwrap();
        assert_eq!(two, t);
        assert_eq!(six, Duration::from_nanos(5000 / 6 + 1400));
    }

    /// Purpose: Auto takes fewer, wider lanes when one block dominates. On
    /// four slots (flat 100 ns dispatch) a 40000 ns block and three 100 ns
    /// blocks score 40000 ns on main's four lanes, 10400 ns on one lane.
    #[test]
    fn auto_beats_main_when_one_block_dominates() {
        let t = [40000, 100, 100, 100];
        let a = chosen(&mut flat(4, BatchPolicy::default(), 100), &nanos(&t), 1e9);
        assert_eq!((a.slots, a.decision.lanes), (4, 4));
        assert_eq!(oracle::baseline(&t, 4, 4, 100).0, 40000);
        assert_eq!(lanes(&a), [(4, vec![0, 1, 2, 3])]);
        assert_eq!(oracle::equal(&t, 4, 1, 100).0, 10000 + 100 + 300);
    }

    /// Purpose: the selected lane allocation is the one evaluated. On four
    /// slots (flat 100 ns dispatch), blocks of 100, 100, 1000 and 2000 ns
    /// score lowest with the large-block reservation, which only
    /// `Moldable` places; `EqualShare` keeps the best equal-share placement.
    #[test]
    fn selected_lane_allocation_is_the_one_evaluated() {
        let t = [100, 100, 1000, 2000];
        let moldable = oracle::choose(&t, 4, 4, 100);
        let equal = oracle::choose_equal(&t, 4, 4, 100);
        assert!(moldable.0 < equal.0);
        for (selector, expected) in [
            (LaneAllocation::Moldable, moldable.1),
            (LaneAllocation::EqualShare, equal.1),
        ] {
            let mut engine = flat(4, BatchPolicy::default(), 100);
            let a = chosen_with(&mut engine, &nanos(&t), 1e9, selector);
            assert_eq!(lanes(&a), expected, "{selector:?}");
        }
    }

    /// Purpose: main's allocation is kept unchanged when no equal-share
    /// placement scores strictly lower: on a tie (serial jobs, mixed zero
    /// times) and when main's placement scores lowest (S = 4 with a
    /// caller-set lane work minimum giving two lanes: 16850 ns against
    /// 16900 ns for two equal lanes and 17225 ns for one).
    #[test]
    fn auto_keeps_main_allocation_on_tie() {
        let tie = [200, 200, 0, 0];
        let a = chosen(&mut flat(4, BatchPolicy::default(), 100), &nanos(&tie), 1e9);
        assert_eq!(a.slots, 4);
        assert_eq!(lanes(&a), oracle::baseline(&tie, 4, 4, 100).1);
        assert!(a.lanes.iter().all(|l| l.jobs.len() == 1));

        let t = [15600, 2000, 6000, 6900, 13100, 15500, 4300, 2300];
        let model = WorkModel::default()
            .with_muladds_per_ns(1.0)
            .with_item_overhead_ns(0.0)
            .with_min_work_per_lane_ns(30000.0);
        let policy = BatchPolicy::default().with_model(model);
        let mean = t.iter().sum::<u64>() as f64 / t.len() as f64;
        let a = chosen(&mut flat(4, policy, 100), &nanos(&t), mean);
        assert_eq!((a.slots, a.decision.lanes), (4, 2));
        assert_eq!(oracle::baseline(&t, 4, 2, 100).0, 16850);
        assert_eq!(oracle::equal(&t, 4, 2, 100).0, 16900);
        assert_eq!(oracle::equal(&t, 4, 1, 100).0, 17225);
        assert_eq!(lanes(&a), [(2, vec![0, 2, 3, 6]), (2, vec![1, 4, 5, 7])]);
    }

    /// Purpose: the allocation scratch's reuse is visible and complete. A
    /// cached allocation is rebuilt into the scratch's kept lane tables, so
    /// a warm hit allocates and frees nothing, even where the search
    /// narrowed main's four lanes to one (a 40000 ns block and three 100 ns
    /// ones on four slots, flat 100 ns dispatch) or reserved uneven lanes
    /// (blocks of 100, 100, 1000 and 2000 ns: three slots and one); the
    /// batches alternate, so each hit replaces another's lanes. Dropping the
    /// scratch then frees at most its `retained_bytes`.
    #[test]
    fn warm_cached_allocation_allocates_nothing_and_scratch_is_charged() {
        use crate::tests::counting_alloc;
        let batches = [
            (nanos(&[40000, 100, 100, 100]), 1),
            (nanos(&[40000, 100, 100, 101]), 1),
            (nanos(&[100, 100, 1000, 2000]), 2),
        ];
        let keys = [[0; 2]; 4];
        let model = WorkModel::default()
            .with_muladds_per_ns(1.0)
            .with_item_overhead_ns(0.0);
        let mut engine = flat(4, BatchPolicy::default().with_model(model), 100);
        let cache = cache(1 << 20);
        engine
            .with_contraction_exec(|exec, _, _| {
                let exec: &Exec<'_> = exec;
                let mut scratch = AllocationScratch::default();
                let queries = std::cell::Cell::new(0);
                let mut call = |t: &[Duration]| {
                    let grant = |_, t: Duration, w| {
                        queries.set(queries.get() + 1);
                        exec.width_for_within(w, t.as_nanos() as f64, 1)
                    };
                    let a = allocate_in(
                        exec,
                        t,
                        (OpKind::Gemm, LaneAllocation::Moldable, |j| keys[j]),
                        10075.0,
                        &mut scratch,
                        &cache,
                        Some(grant),
                    );
                    let chosen = (a.decision.lanes, a.lanes.len());
                    scratch.keep(a);
                    chosen
                };
                for (t, lanes) in &batches {
                    assert_eq!(call(t), (4, *lanes));
                }
                for (t, lanes) in batches.iter().chain(&batches) {
                    queries.set(0);
                    let (chosen, allocs) = counting_alloc::measure(|| call(t));
                    assert_eq!(chosen, (4, *lanes));
                    assert_eq!(queries.get(), 0, "the call hit");
                    assert_eq!((allocs.calls, allocs.frees), (0, 0));
                }
                let charged = scratch.retained_bytes();
                let ((), dropped) = counting_alloc::measure(|| drop(scratch));
                assert!(
                    dropped.freed_bytes as usize <= charged,
                    "{} freed, {charged} charged",
                    dropped.freed_bytes
                );
                Ok(())
            })
            .unwrap();
    }

    /// Purpose: jobs with equal serial times and grant keys share every
    /// grant, so only one of them is queried, without changing the choice.
    /// Nine jobs in three groups of three (flat 100 ns dispatch, six slots)
    /// cost a third of the queries of the same jobs with distinct keys and
    /// get the same lanes; 200 serial blocks are queried once per distinct
    /// serial time.
    #[test]
    fn equal_grant_inputs_are_queried_once() {
        let queries = std::cell::Cell::new(0);
        let run = |engine: &mut CpuEngine, t: &[Duration], keys: &[GrantKey]| {
            queries.set(0);
            let a = engine
                .with_contraction_exec(|exec, _, _| {
                    let exec: &Exec<'_> = exec;
                    let grant = |_, t: Duration, w| {
                        queries.set(queries.get() + 1);
                        exec.width_for_within(w, t.as_nanos() as f64, 1)
                    };
                    Ok(allocate_in(
                        exec,
                        t,
                        (OpKind::Gemm, LaneAllocation::Moldable, |j| keys[j]),
                        1e9,
                        &mut AllocationScratch::default(),
                        &uncached(),
                        Some(grant),
                    ))
                })
                .unwrap();
            (lanes(&a), queries.get())
        };
        let mut engine = flat(6, BatchPolicy::default(), 100);
        let t = nanos(&[9000, 9000, 9000, 4000, 4000, 4000, 100, 100, 100]);
        let grouped: Vec<GrantKey> = (0..9).map(|j| [j as u64 / 3, 0]).collect();
        let distinct: Vec<GrantKey> = (0..9).map(|j| [j as u64, 0]).collect();
        let (shared, few) = run(&mut engine, &t, &grouped);
        let (separate, all) = run(&mut engine, &t, &distinct);
        assert_eq!(shared, separate);
        assert!(few > 0);
        assert_eq!(all, 3 * few);

        let tiny: Vec<u64> = (0..200).map(|j| 50 + (j * 37) % 200).collect();
        let times: std::collections::BTreeSet<u64> = tiny.iter().copied().collect();
        let (_, count) = run(&mut engine, &nanos(&tiny), &[[0; 2]; 200]);
        assert_eq!(count, times.len());
    }

    /// Purpose: a repeated allocation with every input unchanged is taken
    /// from the cache without a grant query and equals the searched one;
    /// changing any one input it depends on (a serial time, a grant key, the
    /// op kind, a factorization's total or payback work, the mean work, the
    /// work model, the dispatch costs, the budget, the token's team, the
    /// selected lane allocation) searches
    /// again; after a clear, and on a disabled cache
    /// at every call, it searches and finds the cached allocation.
    #[test]
    fn unchanged_inputs_hit_and_any_change_searches() {
        let t = nanos(&[40000, 100, 100, 100]);
        let keys = [[0; 2]; 4];
        let model = WorkModel::default()
            .with_muladds_per_ns(1.0)
            .with_item_overhead_ns(0.0);
        let policy = BatchPolicy::default().with_model(model);
        let cache = cache(1 << 20);
        let mut scratch = AllocationScratch::default();
        let queries = std::cell::Cell::new(0);
        let family = std::cell::Cell::new(OpKind::Gemm);
        let selector = std::cell::Cell::new(LaneAllocation::Moldable);
        // Runs one allocation of `family`'s kind on `exec` and returns its
        // lanes and whether it queried a grant.
        let mut run = |exec: &Exec<'_>, t: &[Duration], keys: &[GrantKey], mean: f64| {
            queries.set(0);
            let grant = |_, t: Duration, w| {
                queries.set(queries.get() + 1);
                exec.width_for_within(w, t.as_nanos() as f64, 1)
            };
            let a = allocate_in(
                exec,
                t,
                (family.get(), selector.get(), |j| keys[j]),
                mean,
                &mut scratch,
                &cache,
                Some(grant),
            );
            let result = (lanes(&a), queries.get() > 0);
            scratch.keep(a);
            result
        };
        let mut on = |engine: &mut CpuEngine, t: &[Duration], keys: &[GrantKey], mean: f64| {
            engine
                .with_contraction_exec(|exec, _, _| Ok(run(exec, t, keys, mean)))
                .unwrap()
        };
        let mut base = flat(4, policy, 100);
        let (first, searched) = on(&mut base, &t, &keys, 10075.0);
        assert!(searched);
        assert_eq!(on(&mut base, &t, &keys, 10075.0), (first.clone(), false));

        let mut other_time = t.clone();
        other_time[3] = Duration::from_nanos(101);
        assert!(on(&mut base, &other_time, &keys, 10075.0).1);
        let mut other_key = keys;
        other_key[2] = [1, 0];
        assert!(on(&mut base, &t, &other_key, 10075.0).1);
        assert!(on(&mut base, &t, &keys, 10076.0).1);
        let other_model = BatchPolicy::default().with_model(model.with_min_items(3));
        assert!(on(&mut flat(4, other_model, 100), &t, &keys, 10075.0).1);
        assert!(on(&mut flat(4, policy, 101), &t, &keys, 10075.0).1);
        assert!(on(&mut flat(5, policy, 100), &t, &keys, 10075.0).1);
        // The same keys under the factorizations' grant, then their per-job
        // total and payback work bits.
        family.set(OpKind::Factorization);
        assert!(on(&mut base, &t, &keys, 10075.0).1);
        let work: Vec<GrantKey> = [40000.0_f64, 100.0, 100.0, 100.0]
            .iter()
            .map(|&w| [w.to_bits(), (w * 6e-4).to_bits()])
            .collect();
        assert!(on(&mut base, &t, &work, 10075.0).1);
        assert!(!on(&mut base, &t, &work, 10075.0).1);
        let mut other_payback = work.clone();
        other_payback[1][1] = 0.07_f64.to_bits();
        assert!(on(&mut base, &t, &other_payback, 10075.0).1);
        let mut other_total = work.clone();
        other_total[1][0] = 100.5_f64.to_bits();
        assert!(on(&mut base, &t, &other_total, 10075.0).1);
        family.set(OpKind::Gemm);
        // The other lane allocation searches once, then hits its own entry.
        selector.set(LaneAllocation::EqualShare);
        let (equal, searched) = on(&mut base, &t, &keys, 10075.0);
        assert!(searched);
        assert_eq!(on(&mut base, &t, &keys, 10075.0), (equal, false));
        selector.set(LaneAllocation::Moldable);
        // The base entry survived the other keys' admissions.
        assert_eq!(on(&mut base, &t, &keys, 10075.0), (first.clone(), false));

        ErasedStructureCacheControl::clear(&cache);
        assert_eq!(on(&mut base, &t, &keys, 10075.0), (first.clone(), true));
        assert_eq!(on(&mut base, &t, &keys, 10075.0), (first.clone(), false));
        ErasedStructureCacheControl::set_byte_budget(&cache, 0);
        for _ in 0..2 {
            assert_eq!(on(&mut base, &t, &keys, 10075.0), (first.clone(), true));
        }
        ErasedStructureCacheControl::set_byte_budget(&cache, 1 << 20);

        // The same pool and dispatch costs, through a plain token without
        // the engine's team.
        use tenferro_cpu::threadteam::{DispatchCosts, WidthCosts};
        let row = WidthCosts::new([Duration::from_nanos(100); 3]);
        let pool = Pool::builder()
            .threads(4)
            .dispatch_costs(DispatchCosts::new(row, row, row))
            .build()
            .unwrap();
        let borrowed = tcontract_exec::Pool::borrow(&pool);
        let plain = Exec::new(&borrowed, 4).unwrap().with_batch_policy(policy);
        assert!(!plain.token_ref().carries_team());
        assert!(run(&plain, &t, &keys, 10075.0).1);
    }

    /// Purpose: the cache's retained bytes stay within its budget. Distinct
    /// batches beyond what a small budget holds evict older entries, and an
    /// entry above the largest admissible size is rejected, with the same
    /// allocation as a search.
    #[test]
    fn admission_respects_the_byte_budget() {
        let model = WorkModel::default()
            .with_muladds_per_ns(1.0)
            .with_item_overhead_ns(0.0);
        let mut engine = flat(4, BatchPolicy::default().with_model(model), 100);
        let keys = [[0; 2]; 4];
        let bounded = cache(4096);
        let tiny = cache(64);
        engine
            .with_contraction_exec(|exec, _, _| {
                let exec: &Exec<'_> = exec;
                let grant =
                    Some(|_, t: Duration, w| exec.width_for_within(w, t.as_nanos() as f64, 1));
                let batches = 100..164;
                for last in batches.clone() {
                    let t = nanos(&[40000, 100, 100, last]);
                    let mut scratch = AllocationScratch::default();
                    let cached = allocate_in(
                        exec,
                        &t,
                        (OpKind::Gemm, LaneAllocation::Moldable, |j| keys[j]),
                        10075.0,
                        &mut scratch,
                        &bounded,
                        grant,
                    );
                    let rejected = allocate_in(
                        exec,
                        &t,
                        (OpKind::Gemm, LaneAllocation::Moldable, |j| keys[j]),
                        10075.0,
                        &mut scratch,
                        &tiny,
                        grant,
                    );
                    assert_eq!(cached, rejected);
                    let info = bounded.info();
                    assert!(info.charged_bytes() <= info.byte_budget(), "{info:?}");
                }
                assert!(bounded.info().evictions() > 0);
                let info = tiny.info();
                assert_eq!(
                    (info.entries(), info.rejections()),
                    (0, batches.end - batches.start)
                );
                Ok(())
            })
            .unwrap();
    }

    /// Purpose: a caller's `min_items` above the job count forbids fan-out,
    /// and Auto then keeps main's single lane over every slot, even where two
    /// equal lanes would score lower, without reading a grant or a grant key.
    #[test]
    fn caller_min_items_keeps_one_lane() {
        let t = nanos(&[8200, 6500, 2800, 5600, 9500]);
        let policy = BatchPolicy::default().with_model(WorkModel::default().with_min_items(6));
        let (keys, grants) = (std::cell::Cell::new(0), std::cell::Cell::new(0));
        let a = flat(3, policy, 100)
            .with_contraction_exec(|exec, _, _| {
                let exec: &Exec<'_> = exec;
                let key = |_| {
                    keys.set(keys.get() + 1);
                    [0; 2]
                };
                let grant = |_, t: Duration, w| {
                    grants.set(grants.get() + 1);
                    exec.width_for_within(w, t.as_nanos() as f64, 1)
                };
                Ok(allocate_in(
                    exec,
                    &t,
                    (OpKind::Gemm, LaneAllocation::Moldable, key),
                    1e9,
                    &mut AllocationScratch::default(),
                    &uncached(),
                    Some(grant),
                ))
            })
            .unwrap();
        assert_eq!(a.decision.lanes, 1);
        assert_eq!(lanes(&a), [(3, vec![0, 1, 2, 3, 4])]);
        assert_eq!((keys.get(), grants.get()), (0, 0));
    }

    /// Purpose: one thread, one job, and many tiny blocks reproduce main's
    /// allocation.
    #[test]
    fn serial_cases_reproduce_main() {
        let t = [8200, 6500, 2800];
        let a = chosen(&mut flat(1, BatchPolicy::default(), 100), &nanos(&t), 1e9);
        assert_eq!(lanes(&a), [(1, vec![0, 1, 2])]);
        let a = chosen(
            &mut flat(4, BatchPolicy::default(), 100),
            &nanos(&[9000]),
            1e9,
        );
        assert_eq!(lanes(&a), [(4, vec![0])]);
        let tiny = [100; 200];
        let a = chosen(
            &mut flat(6, BatchPolicy::default(), 100),
            &nanos(&tiny),
            1e9,
        );
        assert_eq!(
            lanes(&a),
            oracle::baseline(&tiny, a.slots, a.slots.min(200).min(a.decision.lanes), 100).1
        );
    }

    /// Purpose: under the model, Auto's choice is the lowest score over
    /// main's allocation, every equal-share lane count up to the batch
    /// decision and the large-job reservation on at most that many lanes
    /// (so never above main's), and it is legal: no empty lane, nonzero
    /// shares summing to at most S, at most the decision's lanes. Includes
    /// the case a pruning bound would miss (S = 3: two lanes score 11100 ns
    /// against 11364 and 12100), mixed and all-zero times, non-divisible S,
    /// and the reservation's cases on flat 100 ns dispatch:
    /// - it wins: on S = 4, jobs of 100, 100, 1000 and 2000 ns score at
    ///   best 1100 ns on equal shares; the target 1000 ns reserves three
    ///   slots for the 2000 ns job (`2000 / 3 + 100 = 766`), the 1000 ns job
    ///   takes the one-slot lane, and the lanes score 1000 ns;
    /// - its target is not its score: on S = 4, jobs of 10000, 10000, 4000
    ///   and 1000 ns pass the target `ceil(25000 / 4) = 6250` with two
    ///   two-slot lanes (5100 ns each), but the 4000 ns job lifts one to
    ///   7200 ns, and one four-slot lane (6650 ns) is kept;
    /// - it exceeds the decision's lanes: on S = 6 with a 10000 ns lane work
    ///   minimum, one 10000 ns job and 120 of 100 ns give two lanes; under
    ///   this synthetic grant (a paying job takes its whole lane) the
    ///   reservation's four lanes (3900 ns) are discarded, and two equal
    ///   lanes (7733 ns) beat main's 11000 ns. (Under the real SVD grant
    ///   every one of these jobs is serial; see `block_svd`.)
    #[test]
    fn allocation_dominates_main_under_model() {
        let d = 100;
        let pruned = [8200, 6500, 2800, 5600, 9500];
        let a = chosen(
            &mut flat(3, BatchPolicy::default(), d),
            &nanos(&pruned),
            1e9,
        );
        assert_eq!(
            [1, 2, 3].map(|l| oracle::equal(&pruned, 3, l, d).0),
            [11364, 11100, 12100]
        );
        assert_eq!(lanes(&a), [(2, vec![1, 3, 4]), (1, vec![0, 2])]);
        let mut placed = 0;
        assert_eq!(
            oracle::choose_pruned(&pruned, 3, 3, d, &mut placed),
            oracle::choose(&pruned, 3, 3, d)
        );

        let flat4 = || flat(4, BatchPolicy::default(), d);
        let wins = [100, 100, 1000, 2000];
        assert_eq!(
            [1, 2, 3, 4].map(|l| oracle::equal(&wins, 4, l, d).0),
            [1150, 1100, 1100, 2000]
        );
        let reserved = (1000, vec![(3, vec![0, 1, 3]), (1, vec![2])]);
        assert_eq!(oracle::moldable(&wins, 4, d), reserved);
        assert_eq!(oracle::choose(&wins, 4, 4, d), reserved);
        assert_eq!(lanes(&chosen(&mut flat4(), &nanos(&wins), 1e9)), reserved.1);

        let tail = [10000, 10000, 4000, 1000];
        assert_eq!(
            oracle::moldable(&tail, 4, d),
            (7200, vec![(2, vec![0, 2]), (2, vec![1, 3])])
        );
        let one = oracle::equal(&tail, 4, 1, d);
        assert_eq!(one.0, 6650);
        assert_eq!(oracle::choose(&tail, 4, 4, d), one);
        assert_eq!(lanes(&chosen(&mut flat4(), &nanos(&tail), 1e9)), one.1);

        let mut many = vec![10000];
        many.extend([100; 120]);
        let model = WorkModel::default()
            .with_muladds_per_ns(1.0)
            .with_item_overhead_ns(0.0)
            .with_min_work_per_lane_ns(10000.0);
        let mean = many.iter().sum::<u64>() as f64 / many.len() as f64;
        let a = chosen(
            &mut flat(6, BatchPolicy::default().with_model(model), d),
            &nanos(&many),
            mean,
        );
        assert_eq!((a.slots, a.decision.lanes), (6, 2));
        let discarded = oracle::moldable(&many, 6, d);
        assert_eq!((discarded.0, discarded.1.len()), (3900, 4));
        assert_eq!(oracle::baseline(&many, 6, 2, d).0, 11000);
        let two = oracle::equal(&many, 6, 2, d);
        assert_eq!(two.0, 7733);
        assert_eq!(oracle::choose(&many, 6, 2, d), two);
        assert_eq!(lanes(&a), two.1);

        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        let lane_minimum = WorkModel::default()
            .with_muladds_per_ns(1.0)
            .with_item_overhead_ns(0.0)
            .with_min_work_per_lane_ns(20000.0);
        for p in 2..=6 {
            for policy in [
                BatchPolicy::default(),
                BatchPolicy::default().with_model(lane_minimum),
            ] {
                let mut engine = flat(p, policy, d);
                for jobs in 1..=12 {
                    for _ in 0..4 {
                        // Below the payoff (0..250) or paying (400..20000).
                        let t: Vec<u64> = (0..jobs)
                            .map(|_| match next(4) {
                                0 => 0,
                                1 => next(250),
                                _ => 400 + next(19600),
                            })
                            .collect();
                        let mean = t.iter().sum::<u64>() as f64 / jobs as f64;
                        let a = chosen(&mut engine, &nanos(&t), mean);
                        let cap = a.slots.min(jobs).min(a.decision.lanes).max(1);
                        let (score, want) = oracle::choose(&t, a.slots, cap, d);
                        let mut placed = 0;
                        assert_eq!(
                            oracle::choose_pruned(&t, a.slots, cap, d, &mut placed),
                            (score, want.clone())
                        );
                        assert!(score <= oracle::baseline(&t, a.slots, cap, d).0);
                        assert_eq!(lanes(&a), want, "p {p} times {t:?}");
                        assert!(a.lanes.len() <= cap);
                        check(&a, p, jobs);
                    }
                }
                let zeros = vec![0; 5];
                let a = chosen(&mut engine, &nanos(&zeros), 0.0);
                check(&a, p, 5);
            }
        }
    }

    /// Purpose: a job the provider widens only at some shares is still
    /// scored at each share while the serial jobs keep their serial time.
    /// With dispatch costs 3000/2000/1400 ns at widths 2/4/6, a 5000 ns job
    /// pays only from five slots up: one six-slot lane scores
    /// `5000 / 6 + 1400 + 5 * 100 = 2733` ns, every narrower share 5000 ns.
    #[test]
    fn widened_and_serial_jobs_mix() {
        let t = [5000, 100, 100, 100, 100, 100];
        let mut engine = engine_with_costs(6, BatchPolicy::default(), [3000, 2000, 1400]);
        let a = chosen(&mut engine, &nanos(&t), 1e9);
        assert_eq!((a.slots, a.decision.lanes), (6, 6));
        assert_eq!(lanes(&a), [(6, vec![0, 1, 2, 3, 4, 5])]);
    }

    /// Purpose: on many blocks no provider would widen (200 blocks below the
    /// payoff, six slots, flat 100 ns dispatch) Auto returns main's
    /// allocation, and neither the single full-width query that settles
    /// every serial block nor the lower bound that skips placing hopeless
    /// lane counts changes the choice: production equals the plain
    /// simulator, which agrees with the pruned one while that places fewer
    /// than every count.
    #[test]
    fn pruning_keeps_main_on_many_tiny_blocks() {
        let d = 100;
        let t: Vec<u64> = (0..200).map(|j| 50 + (j * 37) % 200).collect();
        let a = chosen(&mut flat(6, BatchPolicy::default(), d), &nanos(&t), 1e9);
        let cap = a.slots.min(t.len()).min(a.decision.lanes);
        assert_eq!((a.slots, cap), (6, 6));
        let (_, main) = oracle::baseline(&t, a.slots, cap, d);
        assert_eq!(lanes(&a), main);
        let mut placed = 0;
        let pruned = oracle::choose_pruned(&t, a.slots, cap, d, &mut placed);
        assert_eq!(pruned, oracle::choose(&t, a.slots, cap, d));
        assert_eq!(pruned.1, main);
        assert!(placed < cap, "placed {placed} of {cap}");
    }

    fn check(a: &Allocation, p: usize, jobs: usize) {
        assert_eq!(a.budget, p);
        if jobs == 0 {
            assert!(a.lanes.is_empty());
            return;
        }
        let l = a.lanes.len();
        assert!(l >= 1 && l <= a.slots.min(jobs) && a.slots <= p);
        assert!(a.lanes.iter().all(|x| x.budget > 0));
        assert!(a.lanes.iter().map(|x| x.budget).sum::<usize>() <= a.slots);
        let mut all: Vec<usize> = a.lanes.iter().flat_map(|x| x.jobs.clone()).collect();
        assert!(a.lanes.iter().all(|x| !x.jobs.is_empty()));
        assert!(a
            .lanes
            .iter()
            .all(|x| x.jobs.windows(2).all(|w| w[0] < w[1])));
        all.sort_unstable();
        assert_eq!(all, (0..jobs).collect::<Vec<_>>());
    }
}
