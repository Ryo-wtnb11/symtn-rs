//! The block placement's contracts (`crate::lanes::allocate`), ported from
//! main's `block_kernel/schedule.rs` tests: legality, determinism, forced
//! decisions, the lane cap, the baseline, the equal-share sweep and the
//! large-job reservation against an independent model.

use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use tcontract_exec::{BatchDecision, BatchPolicy, BatchQuery, BatchStrategy, Exec, WorkModel};
use tenferro_cpu::{CpuEngine, Pool};

use symtn_core::{ErasedStructureCacheControl, StructureCacheKind};

use super::counting_alloc;
use crate::block_kernel::driver::{estimate, work_model_ns, OpKind};
use crate::lanes::{
    self, item_time, run_native, GrantKey, LaneAllocationCache, LaneTables, Placed, Placement, Work,
};

/// One lane of an [`Allocation`]: its share and its jobs, ascending.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Lane {
    budget: usize,
    jobs: Vec<usize>,
}

/// A placement read back as main's `Allocation`: the caller's budget P, the
/// slots S the lanes reserve from (computed here from the same public
/// `Exec` queries), the batch decision and the lanes.
#[derive(Clone, Debug, PartialEq)]
struct Allocation {
    budget: usize,
    slots: usize,
    decision: BatchDecision,
    lanes: Vec<Lane>,
    placement: Placement,
}

/// [`lanes::allocate`] of `times` with grant `grant`, keys `key` of op kind
/// `family`, into `tables` through `cache`, read back.
fn allocate_in(
    exec: &Exec<'_>,
    times: &[Duration],
    family_key: (OpKind, impl Fn(usize) -> GrantKey),
    mean_work: f64,
    tables: &mut LaneTables,
    cache: &LaneAllocationCache,
    grant: Option<impl Fn(usize, Duration, usize) -> usize>,
) -> Allocation {
    lanes::allocate(exec, times, family_key, mean_work, grant, cache, tables);
    read_back(exec, times, mean_work, tables)
}

/// [`allocate_in`] of a GEMM batch on fresh tables, uncached.
fn allocate_with(
    exec: &Exec<'_>,
    times: &[Duration],
    key: impl Fn(usize) -> GrantKey,
    mean_work: f64,
    grant: Option<impl Fn(usize, Duration, usize) -> usize>,
) -> Allocation {
    let tables = &mut LaneTables::default();
    allocate_in(
        exec,
        times,
        (OpKind::Gemm, key),
        mean_work,
        tables,
        &uncached(),
        grant,
    )
}

/// A lane-allocation cache of its own, so a test sees no other test's
/// entries; a budget of zero admits nothing.
fn cache(budget: u64) -> LaneAllocationCache {
    LaneAllocationCache::new(StructureCacheKind::LaneAllocations, budget)
}

fn uncached() -> LaneAllocationCache {
    cache(0)
}

/// `tables`' placement of `times` as an [`Allocation`].
fn read_back(
    exec: &Exec<'_>,
    times: &[Duration],
    mean_work: f64,
    tables: &LaneTables,
) -> Allocation {
    let decision = exec.batch_decision(&BatchQuery::new(times.len(), mean_work));
    let slots = if times.is_empty() {
        0
    } else if exec.batch_policy().strategy == BatchStrategy::Auto {
        let total = times
            .iter()
            .fold(Duration::ZERO, |s, &t| s.saturating_add(t));
        exec.token_ref().width_for(total, 1).get()
    } else {
        decision.lanes * decision.item_threads
    };
    let placement = tables.placement().clone();
    let lanes = (0..placement.lanes())
        .map(|l| Lane {
            budget: placement.share(l),
            jobs: placement.jobs(l).to_vec(),
        })
        .collect();
    Allocation {
        budget: exec.budget(),
        slots,
        decision,
        lanes,
        placement,
    }
}

/// [`allocate_with`] with every job's grant serial.
fn allocate(exec: &Exec<'_>, times: &[Duration], mean_work: f64) -> Allocation {
    allocate_with(exec, times, |_| [0; 3], mean_work, Some(|_, _, _| 1))
}

/// Jobs of one body `F` over states `S` failing with `E`, each with no lane
/// state.
struct Jobs<F, S, E> {
    count: usize,
    job: F,
    io: std::marker::PhantomData<fn(S) -> E>,
}

impl<S: Send, E: Send, F> Work for Jobs<F, S, E>
where
    F: Fn(usize, &mut S, Exec<'_>) -> Result<(), E> + Sync,
{
    type Job = S;
    type Lane = ();
    type Error = E;
    fn count(&self) -> usize {
        self.count
    }
    fn run(&self, job: usize, state: &mut S, _: &mut (), exec: Exec<'_>) -> Result<(), E> {
        (self.job)(job, state, exec)
    }
}

/// Runs every job of `placement` on its `Placed` lanes.
fn run<S: Send, E: Send>(
    exec: Exec<'_>,
    placement: &Placement,
    states: &mut [S],
    job: impl Fn(usize, &mut S, Exec<'_>) -> Result<(), E> + Sync,
) -> Result<Result<(), (usize, E)>, tenferro_cpu::threadteam::Error> {
    let mut tables = LaneTables::from_placement(placement.clone());
    let work = Jobs {
        count: states.len(),
        job,
        io: std::marker::PhantomData,
    };
    let mut lanes = vec![(); placement.lanes()];
    let tables = &mut tables;
    run_native(exec, &work, Placed { tables }, states, &mut lanes).map_err(
        |refused| match refused {
            crate::lanes::Refusal::Context(error) => error,
            other => panic!("supply refused: {other}"),
        },
    )
}

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

fn allocation(threads: usize, policy: BatchPolicy, times: &[Duration], work: f64) -> Allocation {
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
        Ok(run(
            exec.reborrow(),
            &a.placement,
            &mut [(); 0],
            |_, _, _| {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok::<(), ()>(())
            },
        ))
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
                &a.placement,
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
                &a.placement,
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
                        let result = run(
                            exec.reborrow(),
                            &a.placement,
                            &mut vec![(); jobs],
                            |j, _, child| {
                                let lane = a.lanes.iter().find(|l| l.jobs.contains(&j)).unwrap();
                                assert_eq!(child.budget(), lane.budget);
                                assert_eq!(child.batch_policy(), &policy);
                                seen[j].fetch_add(1, Ordering::Relaxed);
                                Ok::<(), ()>(())
                            },
                        );
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
            let a = allocate_with(
                exec,
                &t,
                |_| [0; 3],
                1e6,
                Some(|_, t: Duration, w| if t.as_nanos() >= 300 { w } else { 1 }),
            );
            assert_eq!(lanes(&a), [(3, vec![0, 1, 3]), (1, vec![2])]);
            let seen: Vec<AtomicUsize> = (0..4).map(|_| AtomicUsize::new(0)).collect();
            let result = run(
                exec.reborrow(),
                &a.placement,
                &mut [(); 4],
                |j, _, child| {
                    seen[j].store(child.budget(), Ordering::Relaxed);
                    Ok::<(), ()>(())
                },
            );
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
/// the job's time).
fn chosen(engine: &mut CpuEngine, times: &[Duration], mean_work: f64) -> Allocation {
    engine
        .with_contraction_exec(|exec, _, _| {
            let exec: &Exec<'_> = exec;
            let grant = |_, t: Duration, w| exec.width_for_within(w, t.as_nanos() as f64, 1);
            Ok(allocate_with(
                exec,
                times,
                |_| [0; 3],
                mean_work,
                Some(grant),
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

    fn finish(widths: Vec<usize>, mut jobs: Vec<Vec<usize>>, t: &[u64], d: u64) -> (u64, Lanes) {
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

    /// Exclusive reservation: the least target T, as [`moldable`]'s, at
    /// which the jobs above T fit their least widths into S and the other
    /// jobs' serial time fits the one-slot lanes left, `sum <= (S - used)
    /// T`; each job above T alone on a lane of its width, the others,
    /// largest first, on the least loaded one-slot lane (then fewest jobs,
    /// then lowest lane); empty one-slot lanes then widen the reserved
    /// lanes, largest job first, while the grant grows.
    pub(super) fn exclusive(t: &[u64], slots: usize, d: u64) -> (u64, Lanes) {
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
        let target = targets
            .into_iter()
            .find(|&target| {
                let (big, rest): (Vec<usize>, Vec<usize>) =
                    order(t).into_iter().partition(|&j| t[j] > target);
                let used: usize = big.iter().map(|&j| need(j, target).unwrap()).sum();
                let tail: u64 = rest.iter().map(|&j| t[j]).sum();
                used <= slots
                    && (rest.is_empty() || used < slots)
                    && tail as u128 <= (slots - used) as u128 * target as u128
            })
            .unwrap();
        let big: Vec<usize> = order(t).into_iter().filter(|&j| t[j] > target).collect();
        let mut widths: Vec<usize> = big.iter().map(|&j| need(j, target).unwrap()).collect();
        let used: usize = widths.iter().sum();
        let thin = slots - used;
        let mut jobs: Vec<Vec<usize>> = big.iter().map(|&j| vec![j]).collect();
        let (mut load, mut thin_jobs) = (vec![0u64; thin], vec![Vec::new(); thin]);
        for j in order(t).into_iter().skip(big.len()) {
            let k = (0..thin)
                .min_by_key(|&k| (load[k] + t[j], thin_jobs[k].len(), k))
                .unwrap();
            load[k] += t[j];
            thin_jobs[k].push(j);
        }
        let mut idle = thin_jobs.iter().filter(|js| js.is_empty()).count();
        for (k, &j) in big.iter().enumerate() {
            while idle > 0 && grant(t[j], widths[k] + 1, d) > grant(t[j], widths[k], d) {
                widths[k] += 1;
                idle -= 1;
            }
        }
        widths.extend(std::iter::repeat_n(1, thin));
        jobs.extend(thin_jobs);
        let (widths, jobs) = widths
            .into_iter()
            .zip(jobs)
            .filter(|(_, js)| !js.is_empty())
            .unzip();
        finish(widths, jobs, t, d)
    }

    /// `best`, or the better reservation (the exclusive one only when it
    /// scores strictly lower than the shared one) when it scores strictly
    /// lower on at most `cap` lanes and `cap` allows fan-out; a reservation
    /// on more than `cap` lanes is discarded.
    fn or_moldable(
        best: (u64, Lanes),
        t: &[u64],
        slots: usize,
        cap: usize,
        d: u64,
    ) -> (u64, Lanes) {
        let fits = |m: (u64, Lanes)| (m.1.len() <= cap).then_some(m);
        let m = match (fits(moldable(t, slots, d)), fits(exclusive(t, slots, d))) {
            (Some(m), Some(x)) => Some(if x.0 < m.0 { x } else { m }),
            (m, x) => m.or(x),
        };
        match m {
            Some(m) if cap > 1 && m.0 < best.0 => m,
            _ => best,
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
                [0; 3]
            };
            let grant = |_, t: Duration, w| {
                grants.set(grants.get() + 1);
                exec.width_for_within(w, t.as_nanos() as f64, 1)
            };
            Ok(allocate_with(exec, &t, key, 1e9, Some(grant)))
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

    // One large job and a tail: the shared reservation takes all six slots
    // (`30000 / 6 + 100 = 5100`) and queues the tail on that lane at width
    // six (`5 * (1000 / 6 + 100) = 1330`), 6430 like one equal share; the
    // exclusive one leaves a slot for the tail (`5 * 1000 = 5000 <= 1 *
    // 6100`) and scores `30000 / 5 + 100 = 6100`.
    let alone = [30000, 1000, 1000, 1000, 1000, 1000];
    let excl = (6100, vec![(5, vec![0]), (1, vec![1, 2, 3, 4, 5])]);
    assert_eq!(
        oracle::moldable(&alone, 6, d),
        (6430, vec![(6, vec![0, 1, 2, 3, 4, 5])])
    );
    assert_eq!(oracle::equal(&alone, 6, 1, d).0, 6430);
    assert_eq!(oracle::exclusive(&alone, 6, d), excl);
    assert_eq!(oracle::choose(&alone, 6, 6, d), excl);
    assert_eq!(
        lanes(&chosen(
            &mut flat(6, BatchPolicy::default(), d),
            &nanos(&alone),
            1e9
        )),
        excl.1
    );

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

/// Purpose: the GEMM lane decision reads the jobs' multiply-adds, not their
/// times. Two equal GEMMs of weighted work `W = 2^20` at tcontract's rate
/// take `d` ns each; with a model rate of `W / (100 d)` per ns, no overhead,
/// a `3 d` lane minimum, `min_items = 2` and `min_items_per_lane = 1` on two
/// slots of cheap dispatch, each lane carries `100 d` of work and the batch
/// places two lanes.
#[test]
fn gemm_lane_decision_reads_work_units() {
    use crate::block_kernel::driver::{cost, OpKind};
    let work = (1u64 << 20) as f64;
    let d = cost(OpKind::Gemm, work, &WorkModel::default());
    let ns = d.as_nanos() as f64;
    let model = WorkModel::default()
        .with_muladds_per_ns(work / (100.0 * ns))
        .with_item_overhead_ns(0.0)
        .with_min_work_per_lane_ns(3.0 * ns)
        .with_min_items(2)
        .with_min_items_per_lane(1);
    let a = engine_with_costs(2, BatchPolicy::default().with_model(model), [1; 3])
        .with_contraction_exec(|exec, _, _| Ok(allocate(exec, &[d, d], work)))
        .unwrap();
    assert_eq!(a.decision.lanes, 2);
    assert_eq!(lanes(&a), [(1, vec![0]), (1, vec![1])]);
}

/// Purpose: the placement tables' reuse is visible and complete. A cached
/// placement is rebuilt into the tables' kept capacity, so a warm hit
/// allocates and frees nothing, sorts nothing and queries no grant, even
/// where the search narrowed main's four lanes to one (a 40000 ns block and
/// three 100 ns ones on four slots, flat 100 ns dispatch) or reserved uneven
/// lanes (blocks of 100, 100, 1000 and 2000 ns: three slots and one); the
/// batches alternate, so each hit replaces another's lanes, and each hit
/// equals its cold search. Dropping the tables then frees at most their
/// `retained_bytes`.
#[test]
fn warm_cached_allocation_allocates_nothing_and_scratch_is_charged() {
    let batches = [
        (nanos(&[40000, 100, 100, 100]), 1),
        (nanos(&[40000, 100, 100, 101]), 1),
        (nanos(&[100, 100, 1000, 2000]), 2),
    ];
    let keys = [[0; 3]; 4];
    let model = WorkModel::default()
        .with_muladds_per_ns(1.0)
        .with_item_overhead_ns(0.0);
    let mut engine = flat(4, BatchPolicy::default().with_model(model), 100);
    let cache = cache(1 << 20);
    engine
        .with_contraction_exec(|exec, _, _| {
            let exec: &Exec<'_> = exec;
            let mut tables = LaneTables::default();
            let queries = std::cell::Cell::new(0);
            let call = |t: &[Duration], tables: &mut LaneTables| {
                let grant = |_, t: Duration, w| {
                    queries.set(queries.get() + 1);
                    exec.width_for_within(w, t.as_nanos() as f64, 1)
                };
                let key = (OpKind::Gemm, |j: usize| keys[j]);
                lanes::allocate(exec, t, key, 10075.0, Some(grant), &cache, tables);
            };
            let mut cold = Vec::new();
            for (t, lanes) in &batches {
                call(t, &mut tables);
                assert_eq!(tables.placement().lanes(), *lanes);
                cold.push(tables.placement().clone());
            }
            for (i, (t, _)) in batches.iter().enumerate().chain(batches.iter().enumerate()) {
                queries.set(0);
                let recording = crate::probe::record();
                let ((), allocs) = counting_alloc::measure(|| call(t, &mut tables));
                assert_eq!(*tables.placement(), cold[i], "hit == cold miss");
                assert_eq!(queries.get(), 0, "the call hit");
                assert_eq!(recording.get(crate::probe::Event::PlacementSort), 0);
                assert_eq!((allocs.calls, allocs.frees), (0, 0));
            }
            let charged = tables.retained_bytes();
            let ((), dropped) = counting_alloc::measure(|| drop(tables));
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
                Ok(allocate_with(exec, t, |j| keys[j], 1e9, Some(grant)))
            })
            .unwrap();
        (lanes(&a), queries.get())
    };
    let mut engine = flat(6, BatchPolicy::default(), 100);
    let t = nanos(&[9000, 9000, 9000, 4000, 4000, 4000, 100, 100, 100]);
    let grouped: Vec<GrantKey> = (0..9).map(|j| [j as u64 / 3, 0, 0]).collect();
    let distinct: Vec<GrantKey> = (0..9).map(|j| [j as u64, 0, 0]).collect();
    let (shared, few) = run(&mut engine, &t, &grouped);
    let (separate, all) = run(&mut engine, &t, &distinct);
    assert_eq!(shared, separate);
    assert!(few > 0);
    assert_eq!(all, 3 * few);

    let tiny: Vec<u64> = (0..200).map(|j| 50 + (j * 37) % 200).collect();
    let times: std::collections::BTreeSet<u64> = tiny.iter().copied().collect();
    let (_, count) = run(&mut engine, &nanos(&tiny), &[[0; 3]; 200]);
    assert_eq!(count, times.len());
}

/// Purpose: a repeated placement with every input unchanged is taken from
/// the cache without a grant query and equals the searched one; changing
/// any one input it depends on (a serial time, a grant key, the op kind, a
/// factorization's total or payback work, the mean work, the work model, a
/// dispatch cost, the budget, the token's team) searches again; after a
/// clear, and on a disabled cache at every call, it searches and finds the
/// cached placement.
#[test]
fn unchanged_inputs_hit_and_any_change_searches() {
    let t = nanos(&[40000, 100, 100, 100]);
    let keys = [[0; 3]; 4];
    let model = WorkModel::default()
        .with_muladds_per_ns(1.0)
        .with_item_overhead_ns(0.0);
    let policy = BatchPolicy::default().with_model(model);
    let cache = cache(1 << 20);
    let mut tables = LaneTables::default();
    let queries = std::cell::Cell::new(0);
    let family = std::cell::Cell::new(OpKind::Gemm);
    // Runs one placement of `family`'s kind on `exec` and returns its lanes
    // and whether it queried a grant.
    let mut run = |exec: &Exec<'_>, t: &[Duration], keys: &[GrantKey], mean: f64| {
        queries.set(0);
        let grant = |_, t: Duration, w| {
            queries.set(queries.get() + 1);
            exec.width_for_within(w, t.as_nanos() as f64, 1)
        };
        let key = (family.get(), |j: usize| keys[j]);
        let a = allocate_in(exec, t, key, mean, &mut tables, &cache, Some(grant));
        (lanes(&a), queries.get() > 0)
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
    other_key[2] = [1, 0, 0];
    assert!(on(&mut base, &t, &other_key, 10075.0).1);
    assert!(on(&mut base, &t, &keys, 10076.0).1);
    let other_model = BatchPolicy::default().with_model(model.with_min_items(3));
    assert!(on(&mut flat(4, other_model, 100), &t, &keys, 10075.0).1);
    assert!(on(&mut flat(4, policy, 101), &t, &keys, 10075.0).1);
    // One dispatch-cost entry alone: width 4's anchor.
    assert!(
        on(
            &mut engine_with_costs(4, policy, [100, 101, 100]),
            &t,
            &keys,
            10075.0
        )
        .1
    );
    assert!(on(&mut flat(5, policy, 100), &t, &keys, 10075.0).1);
    // The same keys under the factorizations' grant, then their per-job
    // total and payback work bits.
    family.set(OpKind::Factorization);
    assert!(on(&mut base, &t, &keys, 10075.0).1);
    let work: Vec<GrantKey> = [40000.0_f64, 100.0, 100.0, 100.0]
        .iter()
        .map(|&w| [w.to_bits(), (w * 6e-4).to_bits(), 0])
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
    // Admit the team-bearing base entry again, so the plain token below
    // differs from a resident entry in its team alone.
    assert_eq!(on(&mut base, &t, &keys, 10075.0), (first.clone(), true));
    assert_eq!(on(&mut base, &t, &keys, 10075.0), (first.clone(), false));

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

/// An engine of `threads` whose dispatch cost, for every kind, is the row
/// of the measured `(width, ns)` anchors.
fn engine_with_anchors(threads: usize, policy: BatchPolicy, anchors: &[(usize, u64)]) -> CpuEngine {
    use tenferro_cpu::threadteam::{DispatchCosts, WidthCosts};
    let anchors: Vec<_> = anchors
        .iter()
        .map(|&(w, ns)| (w, Duration::from_nanos(ns)))
        .collect();
    let row = WidthCosts::from_anchors(&anchors).unwrap();
    let pool = Pool::builder()
        .threads(threads)
        .dispatch_costs(DispatchCosts::new(row, row, row))
        .build()
        .unwrap();
    CpuEngine::new(pool).with_batch_policy(policy)
}

/// One searched GEMM placement of four jobs (a 40000 ns block and three of
/// 100 ns) on `engine` through `cache`: its lanes, whether it queried a
/// grant, and how many dispatch costs the cache key read.
fn keyed_run(
    engine: &mut CpuEngine,
    cache: &LaneAllocationCache,
) -> (Vec<(usize, Vec<usize>)>, bool, usize) {
    let t = nanos(&[40000, 100, 100, 100]);
    engine
        .with_contraction_exec(|exec, _, _| {
            let exec: &Exec<'_> = exec;
            let queries = std::cell::Cell::new(0);
            let grant = |_, t: Duration, w| {
                queries.set(queries.get() + 1);
                exec.width_for_within(w, t.as_nanos() as f64, 1)
            };
            let mut tables = LaneTables::default();
            let recording = crate::probe::record();
            let key = (OpKind::Gemm, |_: usize| [0; 3]);
            lanes::allocate(exec, &t, key, 10075.0, Some(grant), cache, &mut tables);
            let reads = recording.get(crate::probe::Event::PlacementCostRead);
            let a = read_back(exec, &t, 10075.0, &tables);
            Ok((lanes(&a), queries.get() > 0, reads))
        })
        .unwrap()
}

fn keyed_policy() -> BatchPolicy {
    BatchPolicy::default().with_model(
        WorkModel::default()
            .with_muladds_per_ns(1.0)
            .with_item_overhead_ns(0.0),
    )
}

/// Purpose: a cached lookup reads the token's dispatch-cost row once, not
/// once to hash it and again to compare it: a hit at budget B reads the
/// B - 1 costs it compares plus the two the hash covers (widths 2 and B), so
/// one more width costs one more read. The miss before it searched.
#[test]
fn a_cached_lookup_reads_the_cost_row_once() {
    for budget in [4, 16] {
        let mut engine = flat(budget, keyed_policy(), 100);
        let cache = cache(1 << 20);
        let (cold, searched, _) = keyed_run(&mut engine, &cache);
        assert!(searched);
        let (hit, searched, reads) = keyed_run(&mut engine, &cache);
        assert_eq!((hit, searched), (cold, false), "budget {budget} hit");
        assert_eq!(reads, (budget - 1) + 2, "budget {budget}");
    }
}

/// Purpose: the cache key is complete in the token's dispatch-cost row,
/// although its hash reads only the row's ends. Two pools whose rows agree
/// at widths 2 and 8 but not at width 4 (checked first, so the second
/// lookup does reach the comparison) never share a placement: the second
/// searches. A third pool with the first one's row, a different pool, hits
/// its entry, and the second row then hits its own.
#[test]
fn a_row_that_differs_only_inside_misses() {
    let a = [(2, 100), (4, 100), (8, 100)];
    let b = [(2, 100), (4, 150), (8, 100)];
    let costs = |engine: &mut CpuEngine| {
        engine
            .with_contraction_exec(|exec, _, _| {
                let at = |w| {
                    exec.token_ref()
                        .dispatch_cost(NonZeroUsize::new(w).unwrap())
                };
                Ok((exec.budget(), at(2), at(4), at(8)))
            })
            .unwrap()
    };
    let mut first = engine_with_anchors(8, keyed_policy(), &a);
    let mut second = engine_with_anchors(8, keyed_policy(), &b);
    let (ca, cb) = (costs(&mut first), costs(&mut second));
    assert_eq!((ca.0, ca.1, ca.3), (cb.0, cb.1, cb.3));
    assert_ne!(ca.2, cb.2);

    let cache = cache(1 << 20);
    assert!(keyed_run(&mut first, &cache).1);
    assert!(!keyed_run(&mut first, &cache).1);
    assert!(
        keyed_run(&mut second, &cache).1,
        "a row differing at width 4 searches"
    );
    let mut third = engine_with_anchors(8, keyed_policy(), &a);
    assert!(
        !keyed_run(&mut third, &cache).1,
        "an equal row of another pool hits"
    );
    assert!(!keyed_run(&mut second, &cache).1);
}

/// Purpose: the cache's retained bytes stay within its budget. Distinct
/// batches beyond what a small budget holds evict older entries, and an
/// entry above the largest admissible size is rejected before its key is
/// staged, with the same placement as a search. An evicted batch searches
/// again, is admitted again (a resident one only hits) and then hits.
#[test]
fn admission_respects_the_byte_budget() {
    let model = WorkModel::default()
        .with_muladds_per_ns(1.0)
        .with_item_overhead_ns(0.0);
    let mut engine = flat(4, BatchPolicy::default().with_model(model), 100);
    let keys = [[0; 3]; 4];
    let bounded = cache(4096);
    let tiny = cache(64);
    engine
        .with_contraction_exec(|exec, _, _| {
            let exec: &Exec<'_> = exec;
            let queries = std::cell::Cell::new(0);
            let grant = |_, t: Duration, w| {
                queries.set(queries.get() + 1);
                exec.width_for_within(w, t.as_nanos() as f64, 1)
            };
            let key = (OpKind::Gemm, |j: usize| keys[j]);
            let batches = 100..164;
            for last in batches.clone() {
                let t = nanos(&[40000, 100, 100, last]);
                let mut tables = LaneTables::default();
                let cached =
                    allocate_in(exec, &t, key, 10075.0, &mut tables, &bounded, Some(grant));
                let rejected = allocate_in(exec, &t, key, 10075.0, &mut tables, &tiny, Some(grant));
                assert_eq!(cached, rejected);
                let info = bounded.info();
                assert!(info.charged_bytes() <= info.byte_budget(), "{info:?}");
            }
            assert!(bounded.info().evictions() > 0);
            let info = tiny.info();
            assert_eq!(
                (info.entries(), info.admissions(), info.rejections()),
                (0, 0, batches.end - batches.start)
            );

            // An evicted batch: the first that searches again.
            let mut tables = LaneTables::default();
            let evicted = batches
                .clone()
                .map(|last| nanos(&[40000, 100, 100, last]))
                .find(|t| {
                    queries.set(0);
                    let admitted = bounded.info().admissions();
                    allocate_in(exec, t, key, 10075.0, &mut tables, &bounded, Some(grant));
                    let searched = queries.get() > 0;
                    assert_eq!(
                        bounded.info().admissions(),
                        admitted + usize::from(searched) as u64
                    );
                    searched
                })
                .expect("an evicted batch searches");
            queries.set(0);
            allocate_in(
                exec,
                &evicted,
                key,
                10075.0,
                &mut tables,
                &bounded,
                Some(grant),
            );
            assert_eq!(queries.get(), 0, "and hits once admitted again");
            Ok(())
        })
        .unwrap();
}

/// Purpose: racing misses of one batch each search and publish; the loser
/// finds the winner resident, counts one duplicate build, and both get the
/// same placement, which then hits. Each thread's first grant query waits
/// for the other's, so both have missed before either publishes.
#[test]
fn a_racing_duplicate_search_counts_and_keeps_the_winner() {
    let model = WorkModel::default()
        .with_muladds_per_ns(1.0)
        .with_item_overhead_ns(0.0);
    let policy = BatchPolicy::default().with_model(model);
    let t = nanos(&[100, 100, 1000, 2000]);
    let keys = [[0; 3]; 4];
    let cache = cache(1 << 20);
    let both = std::sync::Barrier::new(2);
    let search = || {
        flat(4, policy, 100)
            .with_contraction_exec(|exec, _, _| {
                let exec: &Exec<'_> = exec;
                let waited = std::cell::Cell::new(false);
                let grant = |_, t: Duration, w| {
                    if !waited.replace(true) {
                        both.wait();
                    }
                    exec.width_for_within(w, t.as_nanos() as f64, 1)
                };
                let key = (OpKind::Gemm, |j: usize| keys[j]);
                let tables = &mut LaneTables::default();
                Ok(allocate_in(
                    exec,
                    &t,
                    key,
                    10075.0,
                    tables,
                    &cache,
                    Some(grant),
                ))
            })
            .unwrap()
    };
    let (a, b) = std::thread::scope(|scope| {
        let a = scope.spawn(search);
        let b = scope.spawn(search);
        (a.join().unwrap(), b.join().unwrap())
    });
    assert_eq!(a, b);
    let info = cache.info();
    assert_eq!(
        (info.entries(), info.admissions(), info.duplicate_builds()),
        (1, 1, 1)
    );
    let hits = info.hits();
    let again = flat(4, policy, 100)
        .with_contraction_exec(|exec, _, _| {
            let key = (OpKind::Gemm, |j: usize| keys[j]);
            let tables = &mut LaneTables::default();
            let grant = Some(|_, _, _| unreachable!("the call hits"));
            Ok(allocate_in(exec, &t, key, 10075.0, tables, &cache, grant))
        })
        .unwrap();
    assert_eq!((again, cache.info().hits()), (a, hits + 1));
}
