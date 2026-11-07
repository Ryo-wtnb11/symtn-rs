//! The one placement and the one runner for a batch of independent block
//! jobs on one engine's execution context.
//!
//! A job owns an exclusive destination and is one body ([`Work::run`]).
//! [`run_native`] runs a batch without a placement, split in two
//! recursively over a count of logical threads ([`Split`]) or each job alone
//! on one slot ([`Each`]), or on the lanes the block kernels' placement
//! chose ([`Placed`]).
//!
//! # Placement
//!
//! `allocate` places J jobs of serial times `t_j` (canonical order) and
//! mean weighted work `w` on a context of budget B. Every resource number
//! comes from the caller's `Exec`: the slot count S from the token's
//! `width_for` over `sum t_j` with one section (Auto) or the policy's
//! `BatchDecision` for `(J, w)` (forced: `lanes * item_threads`); a forced
//! policy's lane count is `min(decision.lanes, J)`, and Auto's is at most
//! `L0 = min(S, J, decision.lanes)`. Nothing here is calibrated or timed. A
//! single lane (one of the counts is one, or no job) takes every job on all
//! S slots, before any ordering.
//!
//! The baseline is `L0` lanes of equal shares of S (the remainder one slot
//! each to the lowest lanes), jobs largest serial time first on the lane of
//! least accumulated serial time, then fewest jobs, then lowest index.
//! Under Auto, given a grant model (the width `grant(j, t_j, w)` a job's
//! provider takes on a lane of `w` slots), the baseline is scored, then an
//! equal-share placement for every lane count from `L0` down to one, and a
//! large-job reservation on uneven lanes (discarded when more than `L0` of
//! its lanes hold jobs), all under one model, and the lowest score is kept;
//! ties keep the baseline, then the larger lane count, then the equal
//! shares. A job's modeled time on a lane is its serial time when its
//! provider would run it on one thread there, otherwise its time over the
//! provider's granted width plus the token's dispatch cost at that width.
//! The granted width is itself an estimate composed from the provider's
//! public decisions (for a Packed GEMM plan an optimistic upper bound on its
//! grid), and the model has no sublinear speedup, so a lower score is a
//! scheduling estimate, not a promise of a faster run. Without a grant
//! model (a QR batch, or a factorization batch holding an input whose batch
//! product is not one, including zero) and under a forced policy the
//! baseline is returned.
//!
//! The search costs `O(J log J + G S + G' sum(W) + L0 J' + P J log L0)`,
//! counting the provider queries' width scans. Jobs with equal serial times
//! and grant keys share their grants, so only one of each of the G distinct
//! (time, key) groups is queried: once at the full S, and, for the G'
//! groups widened there, once per distinct equal-share width during the
//! sweep (W, with `sum(W) <= 2 S H(L0) + L0`; rebuilding the winner may
//! repeat up to two widths). A proved lower bound costs `O(J')` per lane
//! count for the J' widened jobs, the others entering it as one precomputed
//! sum and maximum, and only the `P <= L0` counts whose bound is below the
//! best score so far are placed, with one heap per width class. The
//! reservation adds, for the fewer than S jobs above the mean load, one
//! query per width up to S (`O(S^3)` with the width scans), an
//! `O(S^2 log S)` sort and binary search of its targets, and one placement
//! of `O(J (R + log S))` steps for its R < S reserved lanes, querying each
//! widened (time, key) group once per reserved width it evaluates. The
//! winner is written as flat tables in `O(J + L)`.
//!
//! A searched placement is kept in the process-global
//! `StructureCacheKind::LaneAllocations` cache, keyed by every input the
//! search reads, so a call whose inputs were all searched before, on any
//! executor, rebuilds its tables in `O(J + B + L)` without a sort or a grant
//! query, and without allocating once the tables have the capacity.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::num::NonZeroUsize;
use std::sync::{Mutex, PoisonError, RwLock};
use std::time::Duration;

pub use tcontract_exec::{BatchPolicy, Exec};
use tcontract_exec::{BatchQuery, BatchStrategy};
use tenferro_cpu::threadteam;

mod placement;
mod runner;

pub use placement::flat_width;
#[cfg(test)]
pub(crate) use placement::item_time;
#[cfg(test)]
pub(crate) use placement::LaneAllocationCache;
pub(crate) use placement::{allocate, lane_allocations, searchable};
use placement::{parallel_split, words, PlaceScratch};
pub(crate) use runner::recycle;
use runner::Arena;
pub use runner::{run_native, Assignment, Cut, Each, Outcome, Placed, Refusal, Split, Work};

/// What a job's grant depends on besides its serial time, the execution
/// context and its op's kind: jobs with equal keys and equal serial times get
/// equal grants at every width.
pub(crate) type GrantKey = [u64; 2];

/// A placed batch: lanes of nonzero shares summing to at most the budget,
/// every job in exactly one lane, each lane's jobs ascending.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Placement {
    shares: Vec<usize>,
    lane_starts: Vec<usize>,
    lane_jobs: Vec<usize>,
    lane_of: Vec<usize>,
}

impl Placement {
    /// The number of lanes L (zero for an empty batch).
    pub fn lanes(&self) -> usize {
        self.shares.len()
    }

    /// The slots lane `lane` reserves.
    pub fn share(&self, lane: usize) -> usize {
        self.shares[lane]
    }

    /// Lane `lane`'s jobs, ascending.
    pub fn jobs(&self, lane: usize) -> &[usize] {
        &self.lane_jobs[self.lane_starts[lane]..self.lane_starts[lane + 1]]
    }

    fn clear(&mut self) {
        self.shares.clear();
        self.lane_starts.clear();
        self.lane_jobs.clear();
        self.lane_of.clear();
    }

    /// One lane of `share` slots holding every job.
    fn sequential(&mut self, share: usize, jobs: usize) {
        self.shares.clear();
        self.shares.push(share.max(1));
        self.lane_starts.clear();
        self.lane_starts.extend([0, jobs]);
        self.lane_jobs.clear();
        self.lane_jobs.extend(0..jobs);
        self.lane_of.clear();
        self.lane_of.resize(jobs, 0);
    }

    fn retained_bytes(&self) -> usize {
        words(
            self.shares.capacity()
                + self.lane_starts.capacity()
                + self.lane_jobs.capacity()
                + self.lane_of.capacity(),
        )
    }
}

/// The placement, its call-local working tables and the runner's tables,
/// kept by the caller's workspace so a warm call reuses every capacity.
#[derive(Debug, Default)]
pub struct LaneTables {
    placement: Placement,
    /// The block driver's per-job serial times, staged here between calls.
    times: Vec<Duration>,
    scratch: PlaceScratch,
    arena: Arena,
}

impl LaneTables {
    pub fn placement(&self) -> &Placement {
        &self.placement
    }

    /// The capacities kept, in bytes.
    pub fn retained_bytes(&self) -> usize {
        self.placement
            .retained_bytes()
            .saturating_add(
                self.times
                    .capacity()
                    .saturating_mul(std::mem::size_of::<Duration>()),
            )
            .saturating_add(self.scratch.retained_bytes())
            .saturating_add(self.arena.retained_bytes())
    }

    /// The staged serial times' buffer, emptied, for the caller to fill and
    /// hand back with [`keep_times`](Self::keep_times).
    pub(crate) fn take_times(&mut self) -> Vec<Duration> {
        let mut times = std::mem::take(&mut self.times);
        times.clear();
        times
    }

    pub(crate) fn keep_times(&mut self, times: Vec<Duration>) {
        self.times = times;
    }

    /// Tables holding `placement`, to run it.
    #[cfg(test)]
    pub(crate) fn from_placement(placement: Placement) -> Self {
        Self {
            placement,
            ..Self::default()
        }
    }
}
