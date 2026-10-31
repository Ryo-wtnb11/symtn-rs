//! The block kernels' shared scheduler: J independent dense jobs in canonical
//! order on one engine's execution context, split into equal-share lanes.
//!
//! Every resource number comes from the caller's `Exec`: the slot count from
//! the token's `width_for` (Auto) or the policy's `BatchDecision` (forced), the
//! lane count from `batch_decision`. Nothing here is calibrated or timed.

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
// The factorization kernels' cost; the GEMM kernel uses the contraction
// model instead.
#[cfg_attr(not(test), allow(dead_code))]
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
}

impl AllocationScratch {
    /// Takes `allocation`'s lane tables back for the next allocation.
    pub(crate) fn keep(&mut self, allocation: Allocation) {
        self.lanes = allocation.lanes;
    }

    /// The capacities kept, in bytes. Saturates.
    #[cfg_attr(feature = "cpu-blas-core", allow(dead_code))]
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
                    .saturating_mul(size_of::<Reverse<(Duration, usize, usize)>>()),
            )
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

/// Equal-split allocation of jobs with serial times `times` (canonical order)
/// and mean weighted work `mean_work` (multiply-adds; it feeds the
/// `BatchQuery`, while `times` feed `width_for`, so neither is converted twice).
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn allocate(exec: &Exec<'_>, times: &[Duration], mean_work: f64) -> Allocation {
    allocate_in(exec, times, mean_work, &mut AllocationScratch::default())
}

/// [`allocate`] with its tables drawn from `scratch`.
pub(crate) fn allocate_in(
    exec: &Exec<'_>,
    times: &[Duration],
    mean_work: f64,
    scratch: &mut AllocationScratch,
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
    scratch.order = order;
    scratch.load = load.into_vec();
    for (k, lane) in assigned.iter_mut().enumerate() {
        lane.jobs.sort_unstable();
        lane.budget = slots / lanes + usize::from(k < slots % lanes);
    }
    Allocation {
        budget,
        slots,
        decision,
        lanes: assigned,
    }
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
