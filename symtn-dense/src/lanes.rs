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

pub use crate::lane_gemm::{LaneGemmBody, LaneGemmIo, LaneGemmShape, LaneGemms};
pub use tcontract_exec::{BatchPolicy, Exec};
use tcontract_exec::{BatchQuery, BatchStrategy};
use tenferro_cpu::threadteam;

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
    pub fn selected(&self, job: usize, stage: usize, share: usize) -> usize {
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

#[derive(Debug, Default)]
struct PlaceScratch {
    /// `D(k)` in ns at index k, for `2 <= k <= B`.
    costs: Vec<u128>,
    /// Every stage's serial estimate in ns and kind, flat in job order.
    stage_ns: Vec<u128>,
    sections: Vec<u32>,
    kinds: Vec<StageKind>,
    /// Per job: E_j, w_j and the allotment a_j.
    job_ns: Vec<u128>,
    job_width: Vec<usize>,
    allot: Vec<usize>,
    /// The width of each distinct widening estimate, and this call's keys.
    ///
    /// Why the map is emptied key by key rather than cleared: `clear` resets
    /// every bucket of the retained table, a cost of the largest earlier call
    /// rather than of this one. Removing this call's keys costs O(G_w), plus
    /// hashbrown's in-place rehash of its deleted markers, at most once per
    /// table capacity of insertions (amortized O(1) per key).
    memo: HashMap<Estimate, usize>,
    memo_keys: Vec<Estimate>,
    /// The memo's full capacity, the charge's basis: the largest capacity it
    /// reported, sampled after every insertion. Why not `memo.capacity()`
    /// itself: a removal may leave a deleted marker, which lowers the
    /// reported capacity (by a hash-dependent amount) without freeing a
    /// bucket; a growth rehashes them away, so the sample right after it is
    /// the new table's full capacity.
    memo_capacity: usize,
    /// Per lane: accumulated time and job count.
    tau: Vec<u128>,
    count: Vec<usize>,
    heap: BinaryHeap<Reverse<(u128, usize)>>,
    /// The index of each distinct lane share (by share, `usize::MAX` if
    /// none), the distinct shares and a queued job's time on each.
    share_slot: Vec<usize>,
    distinct: Vec<usize>,
    distinct_time: Vec<u128>,
    /// One job's time summary: `(w, sum E, sum of sections)` per distinct
    /// stage width,
    /// indexed through `bucket_of` (by width, `usize::MAX` if none).
    bucket_of: Vec<usize>,
    buckets: Vec<(usize, u128, u128)>,
    /// The barrier walk's slowest movement per round.
    round_max: Vec<u128>,
    #[cfg(test)]
    scores: Option<(u128, u128)>,
}

impl PlaceScratch {
    fn retained_bytes(&self) -> usize {
        let wide = std::mem::size_of::<u128>();
        words(
            self.job_width.capacity()
                + self.allot.capacity()
                + self.count.capacity()
                + self.share_slot.capacity()
                + self.distinct.capacity()
                + self.bucket_of.capacity(),
        )
        .saturating_add(wide.saturating_mul(
            self.costs.capacity()
                + self.stage_ns.capacity()
                + self.job_ns.capacity()
                + self.tau.capacity()
                + self.distinct_time.capacity()
                + self.round_max.capacity(),
        ))
        .saturating_add(
            self.kinds
                .capacity()
                .saturating_mul(std::mem::size_of::<StageKind>()),
        )
        .saturating_add(
            self.sections
                .capacity()
                .saturating_mul(std::mem::size_of::<u32>()),
        )
        .saturating_add(memo_bytes(self.memo_capacity))
        .saturating_add(
            self.memo_keys
                .capacity()
                .saturating_mul(std::mem::size_of::<Estimate>()),
        )
        .saturating_add(
            self.heap
                .capacity()
                .saturating_mul(std::mem::size_of::<(u128, usize)>()),
        )
        .saturating_add(self.buckets.capacity().saturating_mul(std::mem::size_of::<(
            usize,
            u128,
            u128,
        )>()))
    }

    /// `D(x)` for a section of `x` participants; zero below two.
    fn dispatch(&self, x: usize) -> u128 {
        if x < 2 {
            0
        } else {
            self.costs[x]
        }
    }

    /// `t_k(x)` of stage `k`.
    fn stage_time(&self, k: usize, x: usize) -> u128 {
        let e = self.stage_ns[k];
        if x < 2 {
            e
        } else {
            (e / x as u128)
                .saturating_add(u128::from(self.sections[k]).saturating_mul(self.costs[x]))
        }
    }

    /// Fills `buckets` with job `j`'s time summary from `widths`.
    fn summarize(&mut self, starts: &[usize], widths: &[usize], j: usize) {
        self.buckets.clear();
        let range = starts[j]..starts[j + 1];
        for k in range {
            let (w, e, sections) = (widths[k], self.stage_ns[k], self.sections[k]);
            let slot = match self.bucket_of[w] {
                usize::MAX => {
                    self.bucket_of[w] = self.buckets.len();
                    self.buckets.push((w, 0, 0));
                    self.buckets.len() - 1
                }
                slot => slot,
            };
            let bucket = &mut self.buckets[slot];
            bucket.1 = bucket.1.saturating_add(e);
            bucket.2 += u128::from(sections);
        }
        for &(w, _, _) in &self.buckets {
            self.bucket_of[w] = usize::MAX;
        }
    }

    /// The summarized job's time on a share of `a` slots.
    fn job_time(&self, a: usize) -> u128 {
        self.buckets.iter().fold(0u128, |sum, &(w, e, n)| {
            let x = a.min(w);
            let t = if x < 2 {
                e
            } else {
                (e / x as u128).saturating_add(n.saturating_mul(self.costs[x]))
            };
            sum.saturating_add(t)
        })
    }
}

/// Upper bound of the width memo's retained bytes at `capacity` items:
/// hashbrown keeps at most `2 capacity` buckets (load factor 7/8, or one
/// spare below eight buckets), each a slot plus one control byte, and one
/// trailing control group of at most 16 bytes.
fn memo_bytes(capacity: usize) -> usize {
    if capacity == 0 {
        return 0;
    }
    let slot = std::mem::size_of::<(Estimate, usize)>() + 1;
    capacity.saturating_mul(2 * slot).saturating_add(16)
}

fn words(n: usize) -> usize {
    n.saturating_mul(std::mem::size_of::<usize>())
}

fn ns(duration: Duration) -> u128 {
    duration.as_nanos()
}

fn duration(ns: u128) -> Duration {
    Duration::from_nanos(u64::try_from(ns).unwrap_or(u64::MAX))
}

/// Places `jobs` on `exec`'s budget for the runner of `form`, into
/// `tables` (see the module documentation).
pub fn place(exec: &Exec<'_>, jobs: &dyn Jobs, form: Form, tables: &mut LaneTables) {
    let LaneTables {
        placement, scratch, ..
    } = tables;
    placement.clear();
    #[cfg(test)]
    {
        scratch.scores = None;
    }
    let count = jobs.count();
    if count == 0 {
        return;
    }
    if exec.batch_policy().strategy != BatchStrategy::Auto {
        return place_forced(exec, jobs, placement, scratch);
    }
    let budget = exec.budget();
    if budget == 1 {
        return placement.sequential(1, count);
    }
    let token = exec.token_ref();

    // Widths.
    scratch.costs.clear();
    scratch.costs.resize(budget + 1, 0);
    let mut cheapest = Duration::MAX;
    for k in 2..=budget {
        let cost = NonZeroUsize::new(k).map_or(Duration::MAX, |k| token.dispatch_cost(k));
        scratch.costs[k] = ns(cost);
        cheapest = cheapest.min(cost);
    }
    scratch.stage_ns.clear();
    scratch.sections.clear();
    scratch.kinds.clear();
    scratch.job_ns.clear();
    scratch.job_width.clear();
    // Empty after every completed call (`memo_keys`); this clear only costs
    // the table after an unwound one.
    scratch.memo.clear();
    scratch.memo_keys.clear();
    for j in 0..count {
        placement.stage_starts.push(placement.widths.len());
        let (mut e_j, mut w_j) = (0u128, 1usize);
        for k in 0..jobs.stages(j) {
            let estimate = jobs.estimate(j, k);
            let largest = estimate
                .payback
                .map_or(estimate.serial, |p| p.max(estimate.serial));
            // No width pays below `PAYS` sections of the cheapest dispatch.
            let sections = estimate.sections.max(1);
            let w = if !threadteam::pays_at(largest, sections, cheapest) {
                1
            } else {
                let w = *scratch.memo.entry(estimate).or_insert_with(|| {
                    scratch.memo_keys.push(estimate);
                    let w = token.width_for(estimate.serial, sections);
                    estimate
                        .payback
                        .map_or(w, |p| w.min(token.width_for(p, sections)))
                        .get()
                });
                scratch.memo_capacity = scratch.memo_capacity.max(scratch.memo.capacity());
                w
            };
            let e = ns(estimate.serial);
            placement.widths.push(w);
            scratch.stage_ns.push(e);
            scratch.sections.push(sections);
            scratch.kinds.push(jobs.kind(j, k));
            e_j = e_j.saturating_add(e);
            w_j = w_j.max(w);
        }
        scratch.job_ns.push(e_j);
        scratch.job_width.push(w_j);
    }
    placement.stage_starts.push(placement.widths.len());
    for key in scratch.memo_keys.drain(..) {
        scratch.memo.remove(&key);
    }
    let wide: usize = (0..count)
        .filter(|&j| scratch.job_ns[j] > 0)
        .map(|j| scratch.job_width[j])
        .fold(0, usize::saturating_add);
    let b_eff = budget.min(wide);
    if b_eff <= 1 || count == 1 {
        return placement.sequential(budget, count);
    }

    // Allot.
    let total = scratch
        .job_ns
        .iter()
        .fold(0u128, |sum, &e| sum.saturating_add(e));
    scratch.allot.clear();
    scratch.allot.extend((0..count).map(|j| {
        let proportional = (b_eff as u128).saturating_mul(scratch.job_ns[j]) / total.max(1);
        scratch.job_width[j].min(usize::try_from(proportional).unwrap_or(usize::MAX))
    }));

    // Lanes: dedicated ones in source order, then the one-slot lanes.
    placement.lane_of.clear();
    placement.lane_of.resize(count, usize::MAX);
    scratch.tau.clear();
    scratch.buckets.clear();
    scratch.bucket_of.clear();
    scratch.bucket_of.resize(budget + 1, usize::MAX);
    let mut dedicated = 0;
    for j in 0..count {
        let a = scratch.allot[j];
        if a >= 2 {
            placement.lane_of[j] = placement.shares.len();
            placement.shares.push(a);
            scratch.summarize(&placement.stage_starts, &placement.widths, j);
            let t = scratch.job_time(a);
            scratch.tau.push(t);
            dedicated += a;
        }
    }
    for _ in dedicated..b_eff {
        placement.shares.push(1);
        scratch.tau.push(0);
    }
    let lanes = placement.shares.len();
    scratch.share_slot.clear();
    scratch.share_slot.resize(budget + 1, usize::MAX);
    scratch.distinct.clear();
    for &share in &placement.shares {
        if scratch.share_slot[share] == usize::MAX {
            scratch.share_slot[share] = scratch.distinct.len();
            scratch.distinct.push(share);
        }
    }
    // The heap holds one entry per lane. A wide job's scan changes a lane's
    // time without updating it; the next narrow job then rebuilds it from
    // `tau`, O(L) the wide job's lane scan already paid.
    scratch.heap.clear();
    let mut stale = true;
    for j in 0..count {
        if scratch.allot[j] >= 2 {
            continue;
        }
        let lane = if scratch.job_width[j] == 1 {
            // A narrow job takes the same time on every lane.
            if stale {
                scratch.heap.clear();
                scratch
                    .heap
                    .extend((0..lanes).map(|l| Reverse((scratch.tau[l], l))));
                stale = false;
            }
            let lane = least_lane(&mut scratch.heap);
            scratch.tau[lane] = scratch.tau[lane].saturating_add(scratch.job_ns[j]);
            scratch.heap.push(Reverse((scratch.tau[lane], lane)));
            lane
        } else {
            scratch.summarize(&placement.stage_starts, &placement.widths, j);
            scratch.distinct_time.clear();
            for d in 0..scratch.distinct.len() {
                let t = scratch.job_time(scratch.distinct[d]);
                scratch.distinct_time.push(t);
            }
            let finish = |l: usize| {
                let slot = scratch.share_slot[placement.shares[l]];
                scratch.tau[l].saturating_add(scratch.distinct_time[slot])
            };
            let lane =
                (1..lanes).fold(0, |best, l| if finish(l) < finish(best) { l } else { best });
            let t = finish(lane);
            scratch.tau[lane] = t;
            stale = true;
            lane
        };
        placement.lane_of[j] = lane;
    }
    compact(placement, scratch, count);

    // Choose.
    let lanes = placement.lanes();
    let lanes_time = match form {
        Form::Native => scratch
            .tau
            .iter()
            .copied()
            .max()
            .unwrap_or(0)
            .saturating_add(scratch.dispatch(lanes)),
        Form::Barrier => barrier_time(placement, scratch),
    };
    // Each form prices Sequential by its lanes' rounding rule: the native job
    // summary, the barrier's per-stage walk.
    let mut sequential_time = 0u128;
    for j in 0..count {
        let t = match form {
            Form::Native => {
                scratch.summarize(&placement.stage_starts, &placement.widths, j);
                scratch.job_time(budget)
            }
            Form::Barrier => (placement.stage_starts[j]..placement.stage_starts[j + 1])
                .fold(0u128, |t, k| {
                    t.saturating_add(scratch.stage_time(k, placement.widths[k]))
                }),
        };
        sequential_time = sequential_time.saturating_add(t);
    }
    #[cfg(test)]
    {
        scratch.scores = Some((lanes_time, sequential_time));
    }
    let pays = NonZeroUsize::new(lanes).is_some_and(|l| token.pays(duration(total), 1, l));
    if !(lanes_time < sequential_time && pays) {
        placement.sequential(budget, count);
    }
}

/// A forced policy's placement: its batch decision, authoritative.
fn place_forced(
    exec: &Exec<'_>,
    jobs: &dyn Jobs,
    placement: &mut Placement,
    scratch: &mut PlaceScratch,
) {
    let count = jobs.count();
    let decision = exec.batch_decision(&BatchQuery::new(count, 0.0));
    let share = decision.item_threads.max(1);
    if decision.lanes <= 1 {
        return placement.sequential(share, count);
    }
    placement.shares.resize(decision.lanes, share);
    placement.lane_of.resize(count, usize::MAX);
    scratch.tau.clear();
    scratch.tau.resize(decision.lanes, 0);
    scratch.heap.clear();
    scratch
        .heap
        .extend((0..decision.lanes).map(|l| Reverse((0, l))));
    for j in 0..count {
        let e_j = (0..jobs.stages(j)).fold(0u128, |sum, k| {
            sum.saturating_add(ns(jobs.estimate(j, k).serial))
        });
        let lane = least_lane(&mut scratch.heap);
        scratch.tau[lane] = scratch.tau[lane].saturating_add(e_j);
        placement.lane_of[j] = lane;
        scratch.heap.push(Reverse((scratch.tau[lane], lane)));
    }
    compact(placement, scratch, count);
}

/// Places each of `count` jobs alone on a lane of one slot, reading no
/// estimate: the fan-out of a caller whose partition into jobs is already
/// its placement. Every stage then runs at one slot.
pub fn place_each(count: usize, tables: &mut LaneTables) {
    let placement = &mut tables.placement;
    placement.clear();
    placement.shares.resize(count, 1);
    placement.lane_starts.extend(0..=count);
    placement.lane_jobs.extend(0..count);
    placement.lane_of.extend(0..count);
}

/// The lane of least accumulated time, ties to the lowest, popped from
/// `heap`, which holds one current entry per lane; the caller pushes the lane
/// back with its new time.
fn least_lane(heap: &mut BinaryHeap<Reverse<(u128, usize)>>) -> usize {
    // A placement has at least one lane, so the heap is never empty here.
    heap.pop().map_or(0, |Reverse((_, l))| l)
}

/// Drops the empty lanes of `placement.shares` (with their `tau`) and builds
/// each remaining lane's ascending job list from `placement.lane_of`.
fn compact(placement: &mut Placement, scratch: &mut PlaceScratch, count: usize) {
    let lanes = placement.shares.len();
    scratch.count.clear();
    scratch.count.resize(lanes, 0);
    for &l in &placement.lane_of {
        scratch.count[l] += 1;
    }
    // `count` becomes each kept lane's new index.
    let mut kept = 0;
    for l in 0..lanes {
        if scratch.count[l] > 0 {
            placement.shares[kept] = placement.shares[l];
            scratch.tau[kept] = scratch.tau[l];
            placement.lane_starts.push(scratch.count[l]);
            scratch.count[l] = kept;
            kept += 1;
        }
    }
    placement.shares.truncate(kept);
    scratch.tau.truncate(kept);
    // Lane sizes to starts, then a source-order fill.
    let mut start = 0;
    for size in placement.lane_starts.iter_mut() {
        (*size, start) = (start, start + *size);
    }
    placement.lane_starts.push(start);
    placement.lane_jobs.clear();
    placement.lane_jobs.resize(count, 0);
    for j in 0..count {
        let l = scratch.count[placement.lane_of[j]];
        placement.lane_of[j] = l;
        placement.lane_jobs[placement.lane_starts[l]] = j;
        placement.lane_starts[l] += 1;
    }
    // The fill advanced each start to the next lane's.
    for l in (1..kept).rev() {
        placement.lane_starts[l] = placement.lane_starts[l - 1];
    }
    placement.lane_starts[0] = 0;
}

/// The barrier form's lanes score: per round the slowest lane's movement,
/// every provider stage serialized at its whole-engine width, and one outer
/// fork per round the runner enters.
fn barrier_time(placement: &Placement, scratch: &mut PlaceScratch) -> u128 {
    scratch.round_max.clear();
    let (mut serialized, mut rounds) = (0u128, 0usize);
    for l in 0..placement.lanes() {
        let share = placement.shares[l];
        // `trailing`: a movement stage follows the lane's last provider stage
        // (or the lane has none), so the runner enters one more round for it.
        let (mut round, mut movement, mut trailing) = (0usize, 0u128, false);
        for &j in placement.jobs(l) {
            for k in placement.stage_starts[j]..placement.stage_starts[j + 1] {
                let w = placement.widths[k];
                match scratch.kinds[k] {
                    StageKind::Movement => {
                        movement = movement.saturating_add(scratch.stage_time(k, share.min(w)));
                        trailing = true;
                    }
                    StageKind::Provider => {
                        raise(&mut scratch.round_max, round, movement);
                        serialized = serialized.saturating_add(scratch.stage_time(k, w));
                        (round, movement, trailing) = (round + 1, 0, false);
                    }
                }
            }
        }
        raise(&mut scratch.round_max, round, movement);
        rounds = rounds.max(round + usize::from(trailing));
    }
    let forks = (rounds as u128).saturating_mul(scratch.dispatch(placement.lanes()));
    scratch
        .round_max
        .iter()
        .fold(serialized, |sum, &t| sum.saturating_add(t))
        .saturating_add(forks)
}

fn raise(round_max: &mut Vec<u128>, round: usize, time: u128) {
    if round_max.len() <= round {
        round_max.resize(round + 1, 0);
    }
    round_max[round] = round_max[round].max(time);
}

/// The stage bodies of a placed batch.
pub trait Work: Jobs + Sync {
    /// One job's exclusive state.
    type Job: Send;
    /// One lane's state (its initialized source and destination buffers),
    /// kept for the whole call and lent to every stage the lane runs.
    type Lane: Send;
    type Error: Send;

    /// Runs stage `stage` of `job` on `exec`, whose budget is the stage's
    /// selected width ([`Placement::selected`]): a movement stage dispatches
    /// at most that many participants, and a provider stage hands `exec` to
    /// the provider as its whole offer. The barrier form never calls it for a
    /// provider stage.
    fn run(
        &self,
        job: usize,
        stage: usize,
        state: &mut Self::Job,
        lane: &mut Self::Lane,
        exec: Exec<'_>,
    ) -> Result<(), Self::Error>;
}

/// The engine side of the barrier form: the caller owns the engine and
/// enters it once per round, and submits each pending provider stage on the
/// engine's whole offer between rounds.
pub trait Barrier<W: Work + ?Sized> {
    /// A refused engine entry.
    type Refused;

    /// Enters the engine and calls `round` once with its whole context,
    /// returning `round`'s refusal of the lane split as the caller's own.
    fn region(
        &mut self,
        round: &mut dyn FnMut(Exec<'_>) -> Result<(), threadteam::Error>,
    ) -> Result<(), Self::Refused>;

    /// Submits provider stage `stage` of `job` over `lane`'s buffers, on the
    /// whole engine.
    fn submit(
        &mut self,
        job: usize,
        stage: usize,
        state: &mut W::Job,
        lane: &mut W::Lane,
    ) -> Result<(), W::Error>;
}

/// The job references and per-lane parts of earlier runs, kept empty
/// between calls so a warm run reuses their capacity.
#[derive(Debug, Default)]
struct RunScratch {
    refs: Vec<Vec<ParkedRef>>,
    parts: Vec<Mutex<Part<'static, 'static, (), ()>>>,
}

type ParkedRef = (usize, &'static mut ());

impl RunScratch {
    fn retained_bytes(&self) -> usize {
        let inner = self.refs.iter().fold(0usize, |sum, refs| {
            sum.saturating_add(
                refs.capacity()
                    .saturating_mul(std::mem::size_of::<ParkedRef>()),
            )
        });
        inner
            .saturating_add(
                self.refs
                    .capacity()
                    .saturating_mul(std::mem::size_of::<Vec<ParkedRef>>()),
            )
            .saturating_add(
                self.parts
                    .capacity()
                    .saturating_mul(std::mem::size_of::<Mutex<Part<'static, 'static, (), ()>>>()),
            )
    }

    /// Lends each lane its jobs' states, ascending, its lane state and a
    /// cursor at its first stage.
    fn lend<'a, 'b, S, L>(
        &mut self,
        placement: &Placement,
        states: &'b mut [S],
        lanes: &'a mut [L],
    ) -> Vec<Mutex<Part<'a, 'b, S, L>>> {
        let count = placement.lanes();
        assert_eq!(states.len(), placement.lane_of.len(), "one state per job");
        assert!(lanes.len() >= count, "one lane state per lane");
        let mut refs: Vec<Vec<(usize, &'b mut S)>> = std::mem::take(&mut self.refs)
            .into_iter()
            .map(recycle)
            .collect();
        if refs.len() < count {
            refs.resize_with(count, Vec::new);
        }
        // Each lane's vector reserved exactly, so a cold call grows each once.
        for (l, lane) in refs.iter_mut().take(count).enumerate() {
            lane.reserve(placement.jobs(l).len());
        }
        for (j, state) in states.iter_mut().enumerate() {
            refs[placement.lane_of[j]].push((j, state));
        }
        let mut parts: Vec<Mutex<Part<'a, 'b, S, L>>> = recycle(std::mem::take(&mut self.parts));
        parts.extend(
            refs.drain(..count)
                .zip(lanes.iter_mut())
                .map(|(jobs, lane)| {
                    Mutex::new(Part {
                        jobs,
                        lane,
                        pos: 0,
                        stage: 0,
                    })
                }),
        );
        self.refs = refs.into_iter().map(recycle).collect();
        parts
    }

    /// Returns each lane's reference vector to its lane position, ahead of
    /// the unused tail, so the next call's lane `l` reuses lane `l`'s
    /// capacity.
    fn park<S, L>(&mut self, mut parts: Vec<Mutex<Part<'_, '_, S, L>>>) {
        self.refs.splice(
            0..0,
            parts.drain(..).map(|part| {
                recycle(
                    part.into_inner()
                        .unwrap_or_else(PoisonError::into_inner)
                        .jobs,
                )
            }),
        );
        self.parts = recycle(parts);
    }
}

/// `vec`'s allocation, emptied, for elements of another lifetime or type of
/// the same layout.
///
/// Why not a transmute: this crate forbids unsafe code. std collects a mapped
/// `vec::IntoIter` in place when the layouts match; that is an optimization,
/// not a documented guarantee, so a regression would reallocate, never
/// misbehave.
fn recycle<A, B>(mut vec: Vec<A>) -> Vec<B> {
    vec.clear();
    vec.into_iter().map(|_| unreachable!("emptied")).collect()
}

/// One lane's jobs, its state and its cursor: the next stage `stage` of the
/// job at `pos`.
#[derive(Debug)]
struct Part<'a, 'b, S, L> {
    jobs: Vec<(usize, &'b mut S)>,
    lane: &'a mut L,
    pos: usize,
    stage: usize,
}

impl<S, L> Part<'_, '_, S, L> {
    /// Moves the cursor past finished jobs; the job index and stage it then
    /// points at, or `None` when the lane is done.
    fn next(&mut self, jobs: &dyn Jobs) -> Option<(usize, usize)> {
        while let Some(&(j, _)) = self.jobs.get(self.pos) {
            if self.stage < jobs.stages(j) {
                return Some((j, self.stage));
            }
            (self.pos, self.stage) = (self.pos + 1, 0);
        }
        None
    }

    /// Records the current job's outcome: the next stage, or its failure,
    /// which ends that job only.
    fn step<E>(&mut self, outcome: Result<(), E>, failure: &Mutex<Option<(usize, E)>>) {
        match outcome {
            Ok(()) => self.stage += 1,
            Err(error) => {
                let job = self.jobs[self.pos].0;
                let mut first = failure.lock().unwrap_or_else(PoisonError::into_inner);
                if first.as_ref().is_none_or(|&(j, _)| job < j) {
                    *first = Some((job, error));
                }
                (self.pos, self.stage) = (self.pos + 1, 0);
            }
        }
    }

    /// Runs the lane's stages on `exec` until it is done or, with `cut`, at
    /// a provider stage.
    fn advance<W: Work<Job = S, Lane = L>>(
        &mut self,
        work: &W,
        placement: &Placement,
        share: usize,
        exec: &mut Exec<'_>,
        cut: bool,
        failure: &Mutex<Option<(usize, W::Error)>>,
    ) {
        while let Some((j, k)) = self.next(work) {
            if cut && work.kind(j, k) == StageKind::Provider {
                return;
            }
            let width =
                NonZeroUsize::new(placement.selected(j, k, share)).unwrap_or(NonZeroUsize::MIN);
            let state = &mut *self.jobs[self.pos].1;
            let outcome = work.run(j, k, state, self.lane, exec.reborrow().limit(width));
            self.step(outcome, failure);
        }
    }
}

/// A run's job outcome: every job succeeded, or the lowest-indexed failing
/// job and its error.
pub type Outcome<E> = Result<(), (usize, E)>;

/// Runs every job of `tables`' placement on `exec` with exclusive access to
/// its own `states` entry, every stage inside its lane, and returns the
/// lowest-indexed job's failure.
///
/// Lane `l` is one `fork_sized` child of exactly its share, with `exec`'s
/// workspace and policy, and borrows `lanes[l]` for the call. A job stops at
/// its own first failing stage; every other job runs to its end. A panic
/// unwinds after the lanes join. The outer error is threadteam's refusal of
/// the lane shares, before any stage ran: unreachable for a placement made on
/// `exec`'s budget.
///
/// # Panics
///
/// When `states` does not hold one entry per placed job, or `lanes` fewer
/// entries than the placement has lanes.
pub fn run_native<W: Work>(
    exec: Exec<'_>,
    work: &W,
    tables: &mut LaneTables,
    states: &mut [W::Job],
    lanes: &mut [W::Lane],
) -> Result<Outcome<W::Error>, threadteam::Error> {
    let LaneTables { placement, run, .. } = tables;
    if placement.lanes() == 0 {
        return Ok(Ok(()));
    }
    let failure = Mutex::new(None);
    let mut parts = run.lend(placement, states, lanes);
    let forked = round(exec, work, placement, &mut parts, false, &failure);
    run.park(parts);
    forked?;
    let failure = failure.into_inner().unwrap_or_else(PoisonError::into_inner);
    Ok(failure.map_or(Ok(()), Err))
}

/// [`run_native`] in rounds, for an executor whose provider stages need the
/// engine itself: each round is one `host.region`, in which every lane runs
/// its stages up to its next provider stage; then, in lane order, each
/// pending provider stage is submitted alone through `host.submit` on the
/// whole engine over its lane's own state. The call takes, maximized over
/// lanes, each lane's provider stages plus one when movement follows the
/// last of them; no round is entered once every lane is done.
///
/// Every lane keeps its share and its `lanes` entry in every round. Failures
/// are job-local exactly as in [`run_native`], so the reported job does not
/// depend on the lane count; a failed submission is never retried. The outer
/// error is `host`'s refused entry, which stops the call.
///
/// # Panics
///
/// As [`run_native`].
pub fn run_barrier<W: Work, H: Barrier<W>>(
    host: &mut H,
    work: &W,
    tables: &mut LaneTables,
    states: &mut [W::Job],
    lanes: &mut [W::Lane],
) -> Result<Outcome<W::Error>, H::Refused> {
    let LaneTables { placement, run, .. } = tables;
    if placement.lanes() == 0 {
        return Ok(Ok(()));
    }
    let failure = Mutex::new(None);
    let mut parts = run.lend(placement, states, lanes);
    let outcome = loop {
        let pending = parts.iter_mut().any(|part| {
            part.get_mut()
                .unwrap_or_else(PoisonError::into_inner)
                .next(work)
                .is_some()
        });
        if !pending {
            break Ok(());
        }
        let entered =
            host.region(&mut |exec| round(exec, work, placement, &mut parts, true, &failure));
        if let Err(refused) = entered {
            break Err(refused);
        }
        for part in parts.iter_mut() {
            let part = part.get_mut().unwrap_or_else(PoisonError::into_inner);
            if let Some((j, k)) = part.next(work) {
                let state = &mut *part.jobs[part.pos].1;
                let submitted = host.submit(j, k, state, part.lane);
                part.step(submitted, &failure);
            }
        }
    };
    run.park(parts);
    outcome?;
    let failure = failure.into_inner().unwrap_or_else(PoisonError::into_inner);
    Ok(failure.map_or(Ok(()), Err))
}

/// Runs every lane's stages on `exec` (up to its next provider stage with
/// `cut`): one `fork_sized` child of exactly its share per lane.
///
/// Why one lane runs without the fork and its lock: the fork would run it
/// inline anyway, and a platform whose std mutex is boxed lazily (Apple's
/// pthread mutex) allocates on a fresh mutex's first lock.
fn round<W: Work>(
    exec: Exec<'_>,
    work: &W,
    placement: &Placement,
    parts: &mut [Mutex<Part<'_, '_, W::Job, W::Lane>>],
    cut: bool,
    failure: &Mutex<Option<(usize, W::Error)>>,
) -> Result<(), threadteam::Error> {
    let (workspace, policy) = (exec.workspace(), *exec.batch_policy());
    if let [part] = parts {
        let share = placement.shares[0];
        // `limit` clamps; refuse an oversized share as `fork_sized` would.
        if share > exec.budget() {
            return Err(threadteam::Error::SharesExceedBudget {
                budget: exec.budget(),
                sum: share,
            });
        }
        let width = NonZeroUsize::new(share).ok_or(threadteam::Error::ZeroBudget)?;
        let mut exec = Exec::from_token(exec.into_token().limit(width), workspace, policy);
        let part = part.get_mut().unwrap_or_else(PoisonError::into_inner);
        part.advance(work, placement, share, &mut exec, cut, failure);
        return Ok(());
    }
    let parts = &*parts;
    exec.into_token().fork_sized(&placement.shares, |l, token| {
        let mut part = parts[l].lock().unwrap_or_else(PoisonError::into_inner);
        let mut exec = Exec::from_token(token, workspace, policy);
        part.advance(
            work,
            placement,
            placement.shares[l],
            &mut exec,
            cut,
            failure,
        );
    })
}
