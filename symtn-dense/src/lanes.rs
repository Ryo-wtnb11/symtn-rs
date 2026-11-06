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
//! Every stage that widens ([`Estimate::widen`]) gets the provider's own
//! public width rule, threadteam's `width_for` over its serial estimate with
//! its section count `s` ([`Estimate::sections`], one for a provider call);
//! a single-item factorization's width is further capped by the width its
//! payback estimate pays for ([`Estimate::payback`]). A stage of width `x` is
//! modelled as `E` at one slot and `E / x + s D(x)` above, with `D` the
//! token's dispatch cost: a linear-speedup approximation, not a provider
//! curve. A stage that does not widen is modelled at one slot, and runs on
//! its lane's whole share, which the provider's own rule divides. A job's
//! time on a share `a` sums its stages at `min(a, w_k)`, grouped by stage
//! width.
//!
//! Admission is per call: with `E_tot` the total estimate, the call uses
//! `slots = width_for(E_tot, 1)` of the budget, and the baseline is
//! `L0 = min(slots, J)` lanes of equal shares (the remainder one slot each to
//! the lowest lanes), filled by the longest-processing-time rule: jobs by
//! descending estimate, then index, each to the lane of least modeled load,
//! then fewest jobs, then lowest index. The candidates are the equal shares
//! of every lane count from `L0` down to one (one lane is the whole call on
//! `slots`), and one reservation, an intentional heuristic: each job is
//! allotted `a_j = min(w_j, floor(slots E_j / E_tot))`, a job with
//! `a_j >= 2` gets a lane of its own, the remaining slots become one-slot
//! lanes, at most `L0` lanes in all, and every other job goes in source
//! order to the lane on which it would finish first. A lane count of two or
//! more is a candidate only when `E_tot` pays for an outer section of that
//! count. A candidate replaces the best so far only by scoring strictly
//! lower, so ties keep the baseline, then the larger lane count. The score
//! is that of the form that will run:
//!
//! - native: the slowest lane plus `D(L)` (the outer fork publishes `L - 1`
//!   child leaders); an equal-share candidate whose lower bound (its jobs'
//!   largest and mean least time) cannot beat the best is not built;
//! - barrier: per round the slowest lane's movement, plus every provider
//!   stage serialized at its whole-engine width, plus `R D(L)` for the `R`
//!   rounds the runner enters: per lane its provider stages, plus one when
//!   movement follows the last of them, maximized over lanes.
//!
//! All arithmetic is in integer nanoseconds and ties are total, so the
//! placement is repeatable for given estimates; nothing is cached. Native
//! job times divide each width bucket's summed estimate once
//! (`sum E / x + n D(x)`); the barrier walk divides per stage. Per call it
//! costs `O(B + G_w B + N_st + J log J + L0 (beta + J log L0))`, plus
//! `L0 (N_st + R L0)` in the barrier form, where `G_w` counts the distinct
//! widening estimates (the width memo is emptied by its own keys, amortized
//! O(1) each), `beta` the jobs' distinct stage widths summed, and the
//! reservation adds `sum over its queued wide-capable jobs of (L + sigma
//! beta_j)` with `sigma` the distinct lane shares.
//!
//! A movement stage owns the check of its own fork: once it has capped its
//! selected width to its final part count, under `Auto` it forks only when
//! its estimate pays for a section of that many parts on the token, and
//! thread, that forks (whose dispatch-cost row may differ from
//! placement's), and runs inline otherwise. Provider stages keep the
//! provider's own rule.
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
    /// Whether placement may credit the stage with the linear speedup of the
    /// width its estimate pays for. The estimate's producer sets it from the
    /// route it knows (provider, item count, composite form): false where
    /// the provider is not known to speed up at the widths the model would
    /// grant, which placement then scores at one slot. It decides scoring
    /// only: the stage still runs on its lane's whole share, and the
    /// provider's own rule decides how much of it to use.
    pub widen: bool,
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
    /// Every stage's `Auto` offer: its width `w_k`, or the budget for a stage
    /// that does not widen; empty without `Auto` widths, when each stage runs
    /// at its lane's whole share.
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

    /// The width stage `stage` of `job` is offered on a lane of `share`
    /// slots: `min(share, w_k)` under `Auto` (the whole share for a stage
    /// that does not widen), otherwise the whole share.
    pub(crate) fn selected(&self, job: usize, stage: usize, share: usize) -> usize {
        match self.stage_starts.get(job) {
            Some(&start) => self.widths[start + stage].min(share),
            None => share,
        }
    }

    fn clear(&mut self) {
        self.clear_lanes();
        self.stage_starts.clear();
        self.widths.clear();
    }

    fn clear_lanes(&mut self) {
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

    /// The last `Auto` placement's baseline and one-lane scores, in ns, each
    /// when it was scored.
    #[cfg(test)]
    pub(crate) fn scores(&self) -> (Option<u128>, Option<u128>) {
        self.scratch.scores
    }
}
