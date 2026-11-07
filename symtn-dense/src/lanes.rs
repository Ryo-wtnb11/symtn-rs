//! The one placement and the one runner for a batch of independent block
//! jobs on one engine's execution context.
//!
//! A job owns an exclusive destination and is an ordered list of stages, each
//! one symtn-owned movement section or one provider call ([`StageKind`]).
//! [`place`] splits the context's budget B into lanes; [`run_native`] runs
//! every lane in one fork ([`Placed`]). [`run_native`] also runs a batch
//! without a placement: split in two recursively over a count of logical
//! threads ([`Split`]), or each job alone on one slot ([`Each`]).
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
//! A lone job takes one lane of the widest offer of its stages: a lone
//! stage that does not widen is offered the whole budget B, which its
//! provider's own rule divides.
//!
//! In the native form, when no stage widens past one slot, no job's time
//! depends on its share, and nothing is searched: the jobs take one-slot
//! lanes, as many as the total estimate `E_tot` pays for as one flat
//! section, `width_for(E_tot, 1)`, at most `J`, and lowered to the largest
//! count whose outer fork pays (a dispatch row need not be monotone),
//! filled as the baseline below. A search would trade only the outer fork's
//! cost against how tightly the lanes pack the jobs, and so pack every lane
//! up to the longest job. The barrier form searches as below: its rounds
//! serialize the provider stages and pay the outer fork each. A barrier set
//! of exactly one movement stage per job is placed as the native form: its
//! runner enters one round and serializes nothing, and a one-stage job's
//! time is its stage's, so its score is the native one. Any other
//! provider-free set keeps the barrier search: native job times divide a
//! width bucket's summed estimate once and the barrier walk each stage
//! (below), and a set with no stage enters no round.
//!
//! Otherwise admission is the whole budget B, each lane at most `cap`, the widest
//! offer of any stage (a share no stage could use more of). With `E_tot`
//! the total estimate, the baseline is `L0 = min(B, J)` lanes of equal shares
//! (`B / L0`, the remainder one slot each to the lowest lanes, each capped),
//! filled by the longest-processing-time rule: jobs by descending estimate,
//! then index, each to the lane of least modeled load, then fewest jobs, then
//! lowest index. The candidates are the equal shares of every lane count
//! from `L0` down to one (one lane is the whole call on `cap` slots), and one
//! reservation, an intentional heuristic: each job is allotted
//! `a_j = min(w_j, floor(B E_j / E_tot))`, a job with `a_j >= 2` gets a lane
//! of its own, the remaining slots become one-slot lanes, at most `L0` lanes
//! in all, and every other job goes in source order to the lane on which it
//! would finish first. A lane count of two or more is a candidate only when
//! `E_tot` pays for an outer section of that count. The equal-share winner
//! is the least score, ties to the larger lane count (so to the baseline);
//! the reservation replaces it only by scoring strictly lower. The score is
//! that of the form that will run:
//!
//! - native: the slowest lane plus `D(L)` (the outer fork publishes `L - 1`
//!   child leaders). Candidates are built in the order of a lower bound of
//!   their score (every job takes at least its least time on the
//!   candidate's two shares, so the slowest lane takes at least the largest
//!   and the mean of those), and the search stops at the first bound that
//!   cannot beat the best;
//! - barrier: per round the slowest lane's movement, plus every provider
//!   stage serialized at its whole-engine width, plus `R D(L)` for the `R`
//!   rounds the runner enters: per lane its provider stages, plus one when
//!   movement follows the last of them, maximized over lanes. Every
//!   candidate is built.
//!
//! All arithmetic is in integer nanoseconds and ties are total, so the
//! placement is repeatable for given estimates; nothing is cached. Native
//! job times divide each width bucket's summed estimate once
//! (`sum E / x + n D(x)`); the barrier walk divides per stage. A job's time
//! is fixed once a share covers its widest stage, so a bound sums only the
//! jobs wider than the candidate's narrow share `q`, once per distinct `q`
//! (at most `2 sqrt(B)` of them), while every built candidate queries those
//! jobs' buckets again on its two shares. With `beta_q` the stage-width
//! buckets of the jobs wider than `q`, `C` the candidates built (in a set
//! that keeps the barrier search, every paying one) and `G_w` the distinct widening estimates
//! (the width memo is emptied by its own keys, amortized O(1) each), a call
//! costs `O(B + G_w B + N_st + J log J + L0 log L0 + sum over distinct q of
//! beta_q + sum over built candidates c of beta_{q_c} + C J log L0)`, the
//! winner's rebuild included; each barrier-scored candidate also walks its stages
//! and rounds, `O(N_st + J + R)`. The reservation adds `sum over its queued
//! wide-capable jobs of (L + sigma beta_j)` with `sigma` the distinct lane
//! shares. The native no-search case costs the stage pass, the width scans,
//! the sort and one fill.
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
use std::sync::{Mutex, PoisonError, RwLock};
use std::time::Duration;

pub use crate::lane_gemm::{gemm_estimate, LaneGemmBody, LaneGemmIo, LaneGemmShape, LaneGemms};
pub use tcontract_exec::{BatchPolicy, Exec};
use tcontract_exec::{BatchQuery, BatchStrategy};
use tenferro_cpu::threadteam;

mod placement;
mod runner;

pub use placement::{flat_width, place};
use placement::{parallel_split, words, PlaceScratch};
pub(crate) use runner::recycle;
use runner::Arena;
pub use runner::{run_native, Assignment, Cut, Each, Outcome, Placed, Refusal, Split, Work};

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
    /// Provider stages are handed to the engine between rounds, a form
    /// placement still scores; no runner of it remains.
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
            .saturating_add(self.scratch.retained_bytes())
            .saturating_add(self.arena.retained_bytes())
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
