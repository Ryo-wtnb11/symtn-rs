//! The block kernels' shared scheduler: J independent dense jobs in canonical
//! order on one engine's execution context, split into equal-share lanes.
//!
//! Every resource number comes from the caller's `Exec`: the slot count from
//! the token's `width_for` (Auto) or the policy's `BatchDecision` (forced), the
//! lane count from `batch_decision`. Nothing here is calibrated or timed.
//!
//! Under Auto the lane count is at most the batch decision's: the scheduler
//! scores the baseline placement (largest job first on the lane of least
//! accumulated serial time) and an equal-share placement for every lane
//! count up to it under one model, and keeps the lowest score, the baseline
//! on ties. A job's modeled time in a lane is its serial time when its
//! provider would run it on one thread there, otherwise its share of that
//! time over the provider's granted width plus the token's dispatch cost at
//! that width. The model has no sublinear speedup, so a lower score is a
//! scheduling estimate, not a promise of a faster run.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::num::NonZeroUsize;
use std::sync::{Mutex, PoisonError};
use std::time::Duration;

use tcontract_exec::{BatchDecision, BatchQuery, BatchStrategy, Exec, WorkModel};

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
    /// The slots S the lanes share (`S <= P`).
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
    order: Vec<usize>,
    load: Vec<Reverse<(Duration, usize, usize)>>,
    narrow: Vec<Reverse<(Duration, usize, usize)>>,
    columns: Columns,
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
            .map(|lane| lane.jobs.capacity().saturating_mul(size_of::<usize>()))
            .fold(
                self.lanes.capacity().saturating_mul(size_of::<Lane>()),
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
    }
}

/// `count` empty lanes, reusing `pool`'s lane and job capacity.
fn lanes_from(mut pool: Vec<Lane>, count: usize) -> Vec<Lane> {
    pool.truncate(count);
    pool.resize_with(count, || Lane {
        budget: 0,
        jobs: Vec::new(),
    });
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
        mean_work,
        &mut AllocationScratch::default(),
        |_, _, _| 1,
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

/// Equal-split allocation of jobs with serial times `times` (canonical order)
/// and mean weighted work `mean_work` (multiply-adds; it feeds the
/// `BatchQuery`, while `times` feed `width_for`, so neither is converted
/// twice), with its tables drawn from `scratch`.
///
/// `grant(j, times[j], w)` is the width job `j`'s provider takes in a lane of
/// `w` reserved slots; Auto scores its candidates with it (see the module
/// documentation). Forced strategies keep their decision and never call it.
pub(crate) fn allocate_in(
    exec: &Exec<'_>,
    times: &[Duration],
    mean_work: f64,
    scratch: &mut AllocationScratch,
    grant: impl Fn(usize, Duration, usize) -> usize,
) -> Allocation {
    let jobs = times.len();
    let budget = exec.budget();
    let decision = exec.batch_decision(&BatchQuery::new(jobs, mean_work));
    let pool = std::mem::take(&mut scratch.lanes);
    if jobs == 0 {
        return Allocation {
            budget,
            slots: 0,
            decision,
            lanes: lanes_from(pool, 0),
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
    let mut assigned = lanes_from(pool, lanes);
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
    if exec.batch_policy().strategy == BatchStrategy::Auto {
        let fill = |j: usize, width: usize| item_time(exec, times[j], grant(j, times[j], width));
        let mut columns = std::mem::take(&mut scratch.columns);
        let mut narrow = std::mem::take(&mut scratch.narrow);
        columns.width = [0; 2];
        columns.prepare((slots, lanes), jobs, &fill);
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
            columns.prepare((slots, count), jobs, &fill);
            let item = |j: usize, width: usize| columns.get(j, width);
            let score = equal_shares(slots, count, &order, &item, &mut load, &mut narrow, None);
            if score < best.0 {
                best = (score, Some(count));
            }
        }
        if let Some(count) = best.1 {
            columns.prepare((slots, count), jobs, &fill);
            let item = |j: usize, width: usize| columns.get(j, width);
            assigned = lanes_from(assigned, count);
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
    }
    scratch.order = order;
    scratch.load = load;
    debug_assert!(assigned
        .iter()
        .enumerate()
        .all(|(k, lane)| lane.budget
            == slots / assigned.len() + usize::from(k < slots % assigned.len())));
    Allocation {
        budget,
        slots,
        decision,
        lanes: assigned,
    }
}

/// Every job's modeled item time at two lane widths, so a width the
/// candidates share is granted once per job, not once per candidate.
#[derive(Debug, Default)]
struct Columns {
    /// The width of each column; zero (no lane has it) when unfilled.
    width: [usize; 2],
    items: [Vec<Duration>; 2],
}

impl Columns {
    /// Keeps the columns of the widths of `count` equal shares of `slots`
    /// (`count <= slots`) and fills a missing one with `fill(job, width)`.
    fn prepare(
        &mut self,
        (slots, count): (usize, usize),
        jobs: usize,
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
            self.items[slot].clear();
            self.items[slot].extend((0..jobs).map(|j| fill(j, width)));
        }
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

/// Run every job of `allocation` on `exec` with exclusive access to its own
/// `states` entry, and return the lowest canonical failure.
///
/// Every job runs even after another fails, so each job below the returned
/// index has completed. Lane `k` gets a child token of exactly
/// `allocation.lanes[k].budget` slots (threadteam's even `fork` share over S)
/// with the caller's workspace and policy. A panicking job unwinds after the
/// lanes join; the engine's guarded entry then marks itself broken.
///
/// Why per-job state is lent by lane rather than behind a lock per job: a
/// lock per job costs an allocation per job on platforms whose mutex is
/// boxed, and a lane already owns its jobs exclusively.
pub(crate) fn run<S: Send, E: Send>(
    exec: Exec<'_>,
    allocation: &Allocation,
    states: &mut [S],
    job: impl Fn(usize, &mut S, Exec<'_>) -> Result<(), E> + Sync,
) -> Result<(), (usize, E)> {
    let lanes = &allocation.lanes;
    if lanes.is_empty() {
        return Ok(());
    }
    let policy = *exec.batch_policy();
    let workspace = exec.workspace();
    let slots = NonZeroUsize::new(allocation.slots).unwrap_or(NonZeroUsize::MIN);
    let token = exec.limit(slots).into_token();
    // A lane's jobs ascend, so its first failure is its lowest one.
    let run_lane = |token: tenferro_cpu::threadteam::Exec<'_>,
                    jobs: &mut dyn Iterator<Item = (usize, &mut S)>| {
        let mut lane_exec = Exec::from_token(token, workspace, policy);
        let mut first = None;
        for (j, state) in jobs {
            if let Err(error) = job(j, state, lane_exec.reborrow()) {
                first.get_or_insert((j, error));
            }
        }
        first
    };
    if let [lane] = lanes.as_slice() {
        debug_assert_eq!(lane.jobs.len(), states.len());
        return run_lane(token, &mut states.iter_mut().enumerate()).map_or(Ok(()), Err);
    }
    let mut lane_of = vec![0; states.len()];
    for (k, lane) in lanes.iter().enumerate() {
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
    token.fork(lanes.len(), |k, token| {
        let mut part = parts[k].lock().unwrap_or_else(PoisonError::into_inner);
        let (jobs, failure) = &mut *part;
        *failure = run_lane(token, &mut jobs.iter_mut().map(|(j, s)| (*j, &mut **s)));
    });
    parts
        .into_iter()
        .filter_map(|m| m.into_inner().unwrap_or_else(PoisonError::into_inner).1)
        .min_by_key(|&(j, _)| j)
        .map_or(Ok(()), Err)
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
        assert_eq!(result.unwrap(), Ok(()));
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
        assert_eq!(result, Err((2, Injected::Numeric(2))));
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

    /// Every policy, budget and job count: lanes conserve S <= P with shares
    /// differing by at most one, every job is placed and run once on a token of
    /// its lane's reserved budget, and the allocation is reproducible.
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
                            assert_eq!(result, Ok(()));
                            assert!(seen.iter().all(|s| s.load(Ordering::Relaxed) == 1));
                            Ok(())
                        })
                        .unwrap();
                }
            }
        }
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
        engine
            .with_contraction_exec(|exec, _, _| {
                let exec: &Exec<'_> = exec;
                Ok(allocate_in(
                    exec,
                    times,
                    mean_work,
                    &mut AllocationScratch::default(),
                    |_, t, w| exec.width_for_within(w, t.as_nanos() as f64, 1),
                ))
            })
            .unwrap()
    }

    fn lanes(a: &Allocation) -> Vec<(usize, Vec<usize>)> {
        a.lanes.iter().map(|l| (l.budget, l.jobs.clone())).collect()
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

        /// The lowest score over the baseline and every equal-share count up
        /// to `cap`; ties keep the baseline, then the larger count.
        pub(super) fn choose(t: &[u64], slots: usize, cap: usize, d: u64) -> (u64, Lanes) {
            let mut best = baseline(t, slots, cap, d);
            for count in (1..=cap).rev() {
                let candidate = equal(t, slots, count, d);
                if candidate.0 < best.0 {
                    best = candidate;
                }
            }
            best
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

    /// Purpose: a caller's `min_items` above the job count forbids fan-out,
    /// and Auto then keeps main's single lane over every slot, even where two
    /// equal lanes would score lower.
    #[test]
    fn caller_min_items_keeps_one_lane() {
        let t = [8200, 6500, 2800, 5600, 9500];
        let policy = BatchPolicy::default().with_model(WorkModel::default().with_min_items(6));
        let a = chosen(&mut flat(3, policy, 100), &nanos(&t), 1e9);
        assert_eq!(a.decision.lanes, 1);
        assert_eq!(lanes(&a), [(3, vec![0, 1, 2, 3, 4])]);
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
    /// main's allocation and every equal-share lane count up to the batch
    /// decision (so never above main's), and it is legal: no empty lane,
    /// shares summing to S, at most the decision's lanes. Includes the case
    /// a pruning bound would miss (S = 3: two lanes score 11100 ns against
    /// 11364 and 12100), mixed and all-zero times and non-divisible S.
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

    fn check(a: &Allocation, p: usize, jobs: usize) {
        assert_eq!(a.budget, p);
        if jobs == 0 {
            assert!(a.lanes.is_empty());
            return;
        }
        let l = a.lanes.len();
        assert!(l >= 1 && l <= a.slots.min(jobs) && a.slots <= p);
        assert_eq!(a.lanes.iter().map(|x| x.budget).sum::<usize>(), a.slots);
        assert!(a.lanes.windows(2).all(|w| w[0].budget >= w[1].budget));
        assert!(a.lanes[0].budget - a.lanes[l - 1].budget <= 1);
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
