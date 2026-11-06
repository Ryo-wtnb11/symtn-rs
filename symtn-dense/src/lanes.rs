//! The one placement and the one runner for a batch of independent block
//! jobs on one engine's execution context.
//!
//! A job owns an exclusive destination and is an ordered list of stages, each
//! one symtn-owned movement section or one provider call ([`StageKind`]).
//! [`place`] splits the context's budget B into lanes; [`run_native`] runs
//! every lane in one fork, and [`run_barrier`] runs the same stage
//! enumeration in rounds cut at each provider stage, handing those stages to
//! the caller's engine between rounds ([`Barrier`]).
//!
//! # Placement under `Auto`
//!
//! Every stage gets the provider's own public width rule, threadteam's
//! `width_for` over its serial estimate with its section count `s`
//! ([`Estimate::sections`], one for a provider call); a single-item
//! factorization's width is further capped by the width its payback estimate
//! pays for ([`Estimate::payback`]). The job width `w_j` is the widest stage.
//! A stage of width `x` is modelled as `E` at one slot and `E / x + s D(x)`
//! above, with `D` the token's dispatch cost: a linear-speedup approximation,
//! not a provider curve. A job's time on a share `a` sums its stages at
//! `min(a, w_k)`, grouped by stage width.
//!
//! With `B_eff = min(B, sum of w_j over nonzero jobs)` and `E_tot` the total
//! estimate, each job is allotted `a_j = min(w_j, floor(B_eff E_j / E_tot))`.
//! A job with `a_j >= 2` gets a lane of its own; the remaining slots become
//! one-slot lanes. Every other job, zero-estimate jobs included, goes in
//! source order to the lane on which it would finish first (ties to the
//! lowest lane). Empty lanes are dropped. The lanes are kept only when they
//! beat running every job on the whole budget ("sequential") under the score
//! of the form that will run, and their total estimate pays for an outer
//! section of their count:
//!
//! - native: the slowest lane plus `D(L)` (the outer fork publishes `L - 1`
//!   child leaders);
//! - barrier: per round the slowest lane's movement, plus every provider
//!   stage serialized at its whole-engine width, plus `R D(L)` for the `R`
//!   rounds the runner enters: per lane its provider stages, plus one when
//!   movement follows the last of them, maximized over lanes (`1 + max_l`
//!   provider stages when every lane ends in movement);
//! - sequential: every stage at its own width (the barrier form's provider
//!   stages are at their whole-engine width either way).
//!
//! All arithmetic is in integer nanoseconds; nothing is sorted or cached.
//! Native job times, for its lanes and its Sequential score alike, divide
//! each width bucket's summed estimate once (`sum E / x + n D(x)`); the
//! barrier walk divides per stage, for both of its candidates.
//! Per call it costs `O(B + G_w B + N_st + J log L + sum over queued
//! wide-capable jobs of (L + sigma beta_j))`, plus `R L` in the barrier form,
//! where `G_w` counts the distinct widening estimates (the width memo is
//! emptied by its own keys, amortized O(1) each), `sigma` the distinct
//! lane shares and `beta_j` job j's distinct stage widths.
//!
//! A budget of one, or no job, places without reading any estimate.
//!
//! # Forced policies
//!
//! A forced `BatchStrategy` computes no width. Its `BatchDecision` is
//! authoritative: one lane of `item_threads` slots (`Inner`: the whole
//! budget; `Sequential`: one), or `lanes` lanes of `item_threads` slots each
//! (`Outer`, `Split`), filled in source order by least accumulated estimate.
//! Every stage then runs at its lane's whole share.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::num::NonZeroUsize;
use std::sync::{Mutex, PoisonError};
use std::time::Duration;

pub use crate::lane_gemm::{gemm_estimate, LaneGemmBody, LaneGemmIo, LaneGemmShape, LaneGemms};
pub use tcontract_exec::{BatchPolicy, Exec};
use tcontract_exec::{BatchQuery, BatchStrategy};
use tenferro_cpu::threadteam;

mod placement;
mod runner;

pub use placement::{place, place_each};
use placement::{words, PlaceScratch};
pub(crate) use runner::recycle;
use runner::RunScratch;
pub use runner::{run_barrier, run_native, Barrier, Outcome, Work};

/// What one stage of a job is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageKind {
    /// A symtn-owned data movement section (pack, scatter, direct copies).
    Movement,
    /// One provider call (a GEMM, a factorization). The barrier form hands it
    /// to the engine between rounds.
    Provider,
}

/// One stage's provider estimates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Estimate {
    /// Serial time.
    pub serial: Duration,
    /// The payback estimate of a single-item factorization, which caps its
    /// width as tlinalg caps a batch of one; `None` for every other stage,
    /// including a multi-item factorization's intact envelope.
    pub payback: Option<Duration>,
    /// The parallel sections the stage opens (at least one): its width is
    /// queried with this count and each of them is priced one dispatch.
    /// A provider call is one; a composite that runs several provider calls
    /// in one body counts them, an aggregate envelope approximation of
    /// those calls, not a per-call model.
    pub sections: u32,
}

/// Which runner the placement is scored for, fixed by the executor before
/// placement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Form {
    /// Every stage runs inside the lanes ([`run_native`]).
    Native,
    /// Provider stages are handed to the engine between rounds
    /// ([`run_barrier`]).
    Barrier,
}

/// A batch's jobs as placement and the runners see them. Job and stage
/// indices are dense and stable for one placement and its run.
pub trait Jobs {
    /// The number of jobs J.
    fn count(&self) -> usize;
    /// The number of stages of `job`.
    fn stages(&self, job: usize) -> usize;
    fn kind(&self, job: usize, stage: usize) -> StageKind;
    /// Read only where placement needs a time: under `Auto` with a budget
    /// above one, and under a forced multi-lane decision.
    fn estimate(&self, job: usize, stage: usize) -> Estimate;
}

/// A placed batch: lanes of nonzero shares summing to at most the budget,
/// every job in exactly one lane, each lane's jobs ascending.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Placement {
    shares: Vec<usize>,
    lane_starts: Vec<usize>,
    lane_jobs: Vec<usize>,
    lane_of: Vec<usize>,
    /// Per job, where its stage widths start in `widths` (one past the last
    /// job at the end). Empty when no width was computed.
    stage_starts: Vec<usize>,
    /// Every stage's `Auto` width `w_k`, or empty: then each stage runs at
    /// its lane's whole share.
    widths: Vec<usize>,
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

    /// The width stage `stage` of `job` runs at on a lane of `share` slots:
    /// `min(share, w_k)` under `Auto`, otherwise the whole share.
    pub(crate) fn selected(&self, job: usize, stage: usize, share: usize) -> usize {
        match self.stage_starts.get(job) {
            Some(&start) => self.widths[start + stage].min(share),
            None => share,
        }
    }

    fn clear(&mut self) {
        self.shares.clear();
        self.lane_starts.clear();
        self.lane_jobs.clear();
        self.lane_of.clear();
        self.stage_starts.clear();
        self.widths.clear();
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
                + self.lane_of.capacity()
                + self.stage_starts.capacity()
                + self.widths.capacity(),
        )
    }
}

/// The placement, its call-local working tables and the runner's tables,
/// kept by the caller's workspace so a warm call reuses every capacity.
#[derive(Debug, Default)]
pub struct LaneTables {
    placement: Placement,
    scratch: PlaceScratch,
    run: RunScratch,
}

impl LaneTables {
    pub fn placement(&self) -> &Placement {
        &self.placement
    }

    /// The capacities kept, in bytes.
    pub fn retained_bytes(&self) -> usize {
        self.placement
            .retained_bytes()
            .saturating_add(self.scratch.retained_bytes())
            .saturating_add(self.run.retained_bytes())
    }

    /// The placement queue's live entries.
    #[cfg(test)]
    pub(crate) fn queued(&self) -> usize {
        self.scratch.heap.len()
    }

    /// The last `Auto` comparison's `(lanes, sequential)` scores, in ns.
    #[cfg(test)]
    pub(crate) fn scores(&self) -> Option<(u128, u128)> {
        self.scratch.scores
    }
}
