//! The runner contracts of `crate::lanes`.

use std::num::NonZeroUsize;
use std::time::Duration;

use tcontract_exec::{BatchPolicy, BatchStrategy, Exec};
use tenferro_cpu::threadteam::{self, DispatchCosts, WidthCosts};
use tenferro_cpu::{CpuEngine, Pool};

use super::counting_alloc;
use crate::block_kernel::driver::OpKind;
use crate::lanes::{
    allocate, lane_allocations, run_native, Assignment, Cut, Each, LaneAllocationCache, LaneTables,
    Placed, Refusal, Split, Work,
};
use crate::DenseError;

/// A dispatch cost of `d` ns at widths 2, 4 and 6 for every dispatch kind
/// (growing linearly above 6, threadteam's extrapolation).
fn row(d: u64) -> WidthCosts {
    WidthCosts::new([Duration::from_nanos(d); 3])
}

fn engine(threads: usize, policy: BatchPolicy, costs: WidthCosts) -> CpuEngine {
    let pool = Pool::builder()
        .threads(threads)
        .dispatch_costs(DispatchCosts::new(costs, costs, costs))
        .build()
        .unwrap();
    CpuEngine::new(pool).with_batch_policy(policy)
}

/// `f` on `engine`'s contraction context narrowed to `budget` slots.
fn on<R>(engine: &mut CpuEngine, budget: usize, f: impl FnOnce(Exec<'_>) -> R) -> R {
    engine
        .with_contraction_exec(|exec, _, _| {
            Ok(f(exec.reborrow().limit(NonZeroUsize::new(budget).unwrap())))
        })
        .unwrap()
}

/// The baseline placement of jobs of serial times `times` into `tables`:
/// no grant, so no candidate is scored.
fn place(exec: &Exec<'_>, times: &[Duration], tables: &mut LaneTables) {
    let grant = None::<fn(usize, Duration, usize) -> usize>;
    // Without a grant the cache is not read.
    let cache = lane_allocations();
    allocate(
        exec,
        times,
        (OpKind::Gemm, |_| [0; 3]),
        1e9,
        grant,
        cache,
        tables,
    );
}

/// `count` jobs of `ns` each.
fn uniform(count: usize, ns: u64) -> Vec<Duration> {
    vec![Duration::from_nanos(ns); count]
}

/// Records the budget every job is offered; fails the listed jobs.
struct Spy {
    count: usize,
    fail: Vec<usize>,
}

fn injected(job: usize) -> DenseError {
    DenseError::Unsupported {
        op: "injected",
        message: job.to_string(),
    }
}

impl Work for Spy {
    type Job = Vec<usize>;
    type Lane = ();
    type Error = DenseError;
    fn count(&self) -> usize {
        self.count
    }
    fn run(
        &self,
        job: usize,
        seen: &mut Self::Job,
        _: &mut (),
        exec: Exec<'_>,
    ) -> Result<(), DenseError> {
        seen.push(exec.budget());
        if self.fail.contains(&job) {
            return Err(injected(job));
        }
        Ok(())
    }
}

/// Pack, GEMM and scatter of `ROWS x K` inputs against one `K x COLS`
/// right-hand side through each lane's own initialized buffers, `groups[j]`
/// times for job `j`.
struct Packed {
    groups: Vec<usize>,
    rhs: Vec<f64>,
    fail: Vec<usize>,
}

const ROWS: usize = 2;
const K: usize = 2;
const COLS: usize = 2;

struct Component {
    input: Vec<f64>,
    output: Vec<f64>,
}

#[derive(Default)]
struct LaneBuffers {
    source: Vec<f64>,
    destination: Vec<f64>,
    used_by: Vec<usize>,
    /// The budget each job on this lane was offered.
    offers: Vec<usize>,
}

fn gemm(dst: &mut [f64], lhs: &[f64], rhs: &[f64]) {
    for c in 0..COLS {
        for r in 0..ROWS {
            dst[r + ROWS * c] = (0..K).map(|k| lhs[r + ROWS * k] * rhs[k + K * c]).sum();
        }
    }
}

impl Work for Packed {
    type Job = Component;
    type Lane = LaneBuffers;
    type Error = DenseError;
    fn count(&self) -> usize {
        self.groups.len()
    }
    fn run(
        &self,
        job: usize,
        state: &mut Component,
        lane: &mut LaneBuffers,
        exec: Exec<'_>,
    ) -> Result<(), DenseError> {
        lane.offers.push(exec.budget());
        if self.fail.contains(&job) {
            return Err(injected(job));
        }
        for _ in 0..self.groups[job] {
            lane.source.copy_from_slice(&state.input);
            lane.used_by.push(job);
            gemm(&mut lane.destination, &lane.source, &self.rhs);
            for (out, d) in state.output.iter_mut().zip(&lane.destination) {
                *out += d;
            }
        }
        Ok(())
    }
}

fn packed(groups: &[usize], fail: Vec<usize>) -> Packed {
    Packed {
        groups: groups.to_vec(),
        rhs: vec![1.0, 2.0, 3.0, 4.0],
        fail,
    }
}

/// The serial times a `Packed` batch is placed by: a microsecond per group.
fn packed_times(work: &Packed) -> Vec<Duration> {
    work.groups
        .iter()
        .map(|&n| Duration::from_micros(n as u64))
        .collect()
}

fn components(count: usize) -> Vec<Component> {
    (0..count)
        .map(|j| Component {
            input: (0..ROWS * K).map(|i| (10 * j + i) as f64).collect(),
            output: vec![0.0; ROWS * COLS],
        })
        .collect()
}

fn lane_buffers(count: usize) -> Vec<LaneBuffers> {
    (0..count)
        .map(|_| LaneBuffers {
            source: vec![0.0; ROWS * K],
            destination: vec![0.0; ROWS * COLS],
            used_by: vec![],
            offers: vec![],
        })
        .collect()
}

/// Outcome, outputs and each lane's users of one run on `budget` slots
/// under `policy`.
type Ran = (
    Result<(), usize>,
    Vec<Component>,
    Vec<(Vec<usize>, Vec<usize>, Vec<usize>)>,
);

fn run_packed(work: &Packed, budget: usize, policy: BatchPolicy) -> Ran {
    let mut engine = engine(budget, policy, row(60));
    let mut tables = LaneTables::default();
    on(&mut engine, budget, |exec| {
        place(&exec, &packed_times(work), &mut tables)
    });
    let mut states = components(work.count());
    let mut lanes = lane_buffers(tables.placement().lanes());
    let outcome = on(&mut engine, budget, |exec| {
        run_native(
            exec,
            work,
            Placed {
                tables: &mut tables,
            },
            &mut states,
            &mut lanes,
        )
        .unwrap()
    });
    let p = tables.placement();
    let users = (0..p.lanes())
        .map(|l| {
            let lane = &mut lanes[l];
            (
                p.jobs(l).to_vec(),
                std::mem::take(&mut lane.used_by),
                std::mem::take(&mut lane.offers),
            )
        })
        .collect();
    (outcome.map_err(|(j, _)| j), states, users)
}

fn outer() -> BatchPolicy {
    BatchPolicy::new(BatchStrategy::Outer)
}

/// Each lane packs, multiplies and scatters through its own initialized
/// source and destination buffers, and the outputs are the hand products.
#[test]
fn lanes_run_their_jobs_through_their_own_buffers() {
    let work = packed(&[1, 2, 1], vec![]);
    let expected: Vec<Vec<f64>> = components(3)
        .iter()
        .zip([1.0, 2.0, 1.0])
        .map(|(c, groups)| {
            let mut out = vec![0.0; ROWS * COLS];
            gemm(&mut out, &c.input, &work.rhs);
            out.iter().map(|x| x * groups).collect()
        })
        .collect();
    let (outcome, states, users) = run_packed(&work, 2, outer());
    assert_eq!(outcome, Ok(()));
    let outputs: Vec<_> = states.into_iter().map(|c| c.output).collect();
    assert_eq!(outputs, expected);
    assert_eq!(users.len(), 2);
    for (jobs, used_by, _) in users {
        let expected_users: Vec<usize> = jobs
            .iter()
            .flat_map(|&j| std::iter::repeat_n(j, work.groups[j]))
            .collect();
        assert_eq!(used_by, expected_users);
    }
}

/// Components 0 and 1 fail: component 0's error is reported on one lane and
/// on two, and every job ran.
#[test]
fn a_failure_ends_only_its_own_job_and_the_lowest_index_wins() {
    let work = packed(&[2, 1], vec![0, 1]);
    for budget in [1, 2] {
        let (outcome, _, users) = run_packed(&work, budget, outer());
        assert_eq!(users.len(), budget);
        assert_eq!(outcome, Err(0), "budget {budget}");
        let offered: usize = users.iter().map(|(_, _, offers)| offers.len()).sum();
        assert_eq!(offered, 2, "budget {budget}");
    }
}

/// No job work: the runner's own tables are what a warm call reuses.
struct Noop(usize);

impl Work for Noop {
    type Job = ();
    type Lane = ();
    type Error = ();
    fn count(&self) -> usize {
        self.0
    }
    fn run(&self, _: usize, _: &mut (), _: &mut (), _: Exec<'_>) -> Result<(), ()> {
        Ok(())
    }
}

/// After a four-lane call warms the tables, a one-lane call and its repeat
/// reuse them: the repeat allocates nothing.
#[test]
fn a_repeated_call_after_fewer_lanes_allocates_nothing() {
    let mut times = uniform(3, 10_000);
    times.extend(uniform(32, 0));
    let work = Noop(times.len());
    let mut engine = engine(4, outer(), row(60));
    let mut tables = LaneTables::default();
    let mut states = vec![(); work.count()];
    let mut lanes = vec![(); 4];
    let mut call = |engine: &mut CpuEngine, budget| {
        on(engine, budget, |mut exec| {
            counting_alloc::measure(|| {
                place(&exec, &times, &mut tables);
                run_native(
                    exec.reborrow(),
                    &work,
                    Placed {
                        tables: &mut tables,
                    },
                    &mut states,
                    &mut lanes,
                )
                .unwrap()
                .unwrap();
            })
            .1
        })
    };
    call(&mut engine, 4);
    call(&mut engine, 1);
    assert_eq!(call(&mut engine, 1).calls, 0);
}

/// Dropping used tables frees at most what they charge, after a searched
/// placement that scored every candidate and reserved uneven lanes.
#[test]
fn the_tables_charge_covers_what_they_retain() {
    let times: Vec<Duration> = [100, 100, 1000, 2000, 30_000, 7, 900, 4000]
        .map(Duration::from_nanos)
        .to_vec();
    let work = Spy {
        count: times.len(),
        fail: vec![],
    };
    let mut engine = engine(8, BatchPolicy::default(), row(60));
    let mut tables = LaneTables::default();
    let mut seen = vec![vec![]; work.count()];
    on(&mut engine, 8, |mut exec| {
        let shared: &Exec<'_> = &exec;
        let grant = |_, t: Duration, w| shared.width_for_within(w, t.as_nanos() as f64, 1);
        let cache = LaneAllocationCache::new(symtn_core::StructureCacheKind::LaneAllocations, 0);
        let key = (OpKind::Gemm, |_| [0; 3]);
        allocate(shared, &times, key, 1e9, Some(grant), &cache, &mut tables);
        let lanes = tables.placement().lanes();
        run_native(
            exec.reborrow(),
            &work,
            Placed {
                tables: &mut tables,
            },
            &mut seen,
            &mut vec![(); lanes],
        )
        .unwrap()
        .unwrap();
    });
    let charge = tables.retained_bytes();
    let ((), freed) = counting_alloc::measure(|| drop(tables));
    assert!(
        freed.freed_bytes as usize <= charge,
        "{} > {charge}",
        freed.freed_bytes
    );
}

/// A one-lane placement run on a narrower context is refused before any
/// job runs, as the lane fork refuses it.
#[test]
fn a_share_above_the_running_budget_is_refused_before_any_job() {
    let spy = Spy {
        count: 1,
        fail: vec![],
    };
    let mut engine = engine(6, BatchPolicy::default(), row(60));
    let mut tables = LaneTables::default();
    on(&mut engine, 6, |exec| {
        place(&exec, &[Duration::from_secs(1)], &mut tables)
    });
    assert_eq!(
        (tables.placement().lanes(), tables.placement().share(0)),
        (1, 6)
    );
    let mut seen = vec![vec![]];
    let refused = on(&mut engine, 4, |exec| {
        run_native(
            exec,
            &spy,
            Placed {
                tables: &mut tables,
            },
            &mut seen,
            &mut [()],
        )
    });
    assert_eq!(
        refused.map(|_| ()),
        Err(Refusal::Context(threadteam::Error::SharesExceedBudget {
            budget: 4,
            sum: 6
        }))
    );
    assert_eq!(seen, [Vec::<usize>::new()]);
}

/// `count` jobs, failing at the listed jobs.
fn one_stage(count: usize, fail: &[usize]) -> Spy {
    Spy {
        count,
        fail: fail.to_vec(),
    }
}

type Assigned = Result<Result<(), usize>, Refusal<threadteam::Error>>;

/// Each job's offered budgets and the outcome of `spy` run on `budget`
/// slots under `assignment`, with `lanes` lane states.
fn run_assigned(
    engine: &mut CpuEngine,
    budget: usize,
    spy: &Spy,
    assignment: impl Assignment<Spy>,
    lanes: usize,
) -> (Vec<Vec<usize>>, Assigned) {
    let mut seen = vec![vec![]; spy.count()];
    let outcome = on(engine, budget, |exec| {
        run_native(exec, spy, assignment, &mut seen, &mut vec![(); lanes])
    });
    (seen, outcome.map(|ran| ran.map_err(|(j, _)| j)))
}

fn offered(seen: &[Vec<usize>]) -> Vec<usize> {
    seen.iter().flatten().copied().collect()
}

type UnitCut<E> = fn(&mut (), Cut) -> Result<(), E>;

fn split<E>(threads: usize) -> Split<UnitCut<E>> {
    Split {
        threads,
        split: |_, _| Ok(()),
    }
}

/// `Split` gives every job the token of main's recursive replay split
/// (`parallel_split`, `threads / 2` logical threads to the left, a two-child
/// fork halving the token, the odd slot to the left), by hand:
/// - 6 jobs, 6 threads, budget 6: 3 | 3 on tokens 3 | 3, each half 1 | 2 on
///   tokens 2 | 1, the two jobs on one slot sharing it: [2, 1, 1, 2, 1, 1];
/// - 3 jobs, 4 threads, budget 4: 2 | 1 on tokens 2 | 2, the left 1 | 1:
///   [1, 1, 2];
/// - 3 jobs, 3 threads, budget 8: 1 | 2 (1 | 2 threads) on tokens 4 | 4, the
///   right 1 | 1 on tokens 2 | 2: [4, 2, 2];
/// - 3 jobs, 4 threads, budget 7: 2 | 1 on tokens 4 | 3, the left 1 | 1 on
///   tokens 2 | 2: [2, 2, 3];
/// - 1 job, 8 threads, budget 8: a leaf on the whole token: [8].
#[test]
fn split_offers_each_job_the_token_of_mains_recursion() {
    let mut engine = engine(8, BatchPolicy::default(), row(60));
    for (jobs, threads, budget, expected) in [
        (6, 6, 6, vec![2, 1, 1, 2, 1, 1]),
        (3, 4, 4, vec![1, 1, 2]),
        (3, 3, 8, vec![4, 2, 2]),
        (3, 4, 7, vec![2, 2, 3]),
        (1, 8, 8, vec![8]),
    ] {
        let spy = one_stage(jobs, &[]);
        let (seen, outcome) = run_assigned(&mut engine, budget, &spy, split(threads), 1);
        assert_eq!(outcome, Ok(Ok(())));
        assert_eq!(
            offered(&seen),
            expected,
            "{jobs} jobs, {threads} threads, budget {budget}"
        );
    }
}

/// Six jobs over two threads split into leaves [0, 1, 2] and [3, 4, 5]:
/// with jobs 0 and 3 failing, each leaf stops at its failure, the sibling
/// still runs, and the left failure is returned.
#[test]
fn a_split_leaf_stops_at_its_first_failure_and_the_left_wins() {
    let spy = one_stage(6, &[0, 3]);
    let mut engine = engine(2, BatchPolicy::default(), row(60));
    let (seen, outcome) = run_assigned(&mut engine, 2, &spy, split(2), 1);
    assert_eq!(outcome, Ok(Err(0)));
    let attempted: Vec<usize> = (0..6).filter(|&j| !seen[j].is_empty()).collect();
    assert_eq!(attempted, [0, 3]);
}

/// Jobs that record their attempt and check that their part's lane, the
/// job range `split` handed it, holds them.
struct Ranged(usize);

impl Work for Ranged {
    type Job = bool;
    type Lane = std::ops::Range<usize>;
    type Error = usize;
    fn count(&self) -> usize {
        self.0
    }
    fn run(
        &self,
        job: usize,
        attempted: &mut bool,
        lane: &mut Self::Lane,
        _: Exec<'_>,
    ) -> Result<(), usize> {
        *attempted = true;
        if lane.contains(&job) {
            Ok(())
        } else {
            Err(usize::MAX)
        }
    }
}

/// `split` divides each part's lane at its cut: six jobs over four threads
/// cut at job 3, then the left `[0, 3)` over two threads at job 2 and the
/// right `[3, 6)` at job 5, so every job runs inside its own range. A cut
/// `split` refuses (job 2's) stops that part before either half runs and is
/// reported as the part's first job, while the right part still runs.
#[test]
fn a_split_lane_is_divided_at_each_cut() {
    let work = Ranged(6);
    let mut engine = engine(4, BatchPolicy::default(), row(60));
    for refused in [None, Some(2)] {
        let mut attempted = vec![false; 6];
        let cuts = std::sync::Mutex::new(vec![]);
        let outcome = on(&mut engine, 4, |exec| {
            let divide = |lane: &mut std::ops::Range<usize>, cut: Cut| {
                cuts.lock().unwrap().push((cut.job, cut.left, cut.threads));
                if Some(cut.job) == refused {
                    return Err(cut.job);
                }
                let right = cut.job..lane.end;
                lane.end = cut.job;
                Ok(right)
            };
            let split = Split {
                threads: 4,
                split: divide,
            };
            let root = 0..6;
            run_native(
                exec,
                &work,
                split,
                &mut attempted,
                std::slice::from_mut(&mut { root }),
            )
            .unwrap()
        });
        let mut cuts = cuts.into_inner().unwrap();
        cuts.sort_unstable();
        match refused {
            None => {
                assert_eq!(outcome, Ok(()));
                assert_eq!(attempted, [true; 6]);
                assert_eq!(cuts, [(2, 1, 2), (3, 2, 4), (5, 1, 2)]);
            }
            Some(_) => {
                assert_eq!(outcome, Err((0, 2)));
                assert_eq!(attempted, [false, false, false, true, true, true]);
            }
        }
    }
}

/// `Each` runs every job on one slot of its own and returns the lowest
/// failure; a context narrower than the jobs is refused before any job.
#[test]
fn each_runs_every_job_on_one_slot_and_refuses_too_few_slots() {
    let spy = one_stage(3, &[1, 2]);
    let mut engine = engine(4, BatchPolicy::default(), row(60));
    let (seen, outcome) = run_assigned(&mut engine, 4, &spy, Each, 3);
    assert_eq!(outcome, Ok(Err(1)));
    assert_eq!(seen, [[1]; 3]);
    let (seen, outcome) = run_assigned(&mut engine, 2, &spy, Each, 3);
    assert_eq!(
        outcome,
        Err(Refusal::Context(threadteam::Error::SharesExceedBudget {
            budget: 2,
            sum: 3
        }))
    );
    assert!(seen.iter().all(Vec::is_empty));
}

/// No job work, every job failing with a unit error.
struct Failing(usize);

impl Work for Failing {
    type Job = ();
    type Lane = ();
    type Error = ();
    fn count(&self) -> usize {
        self.0
    }
    fn run(&self, _: usize, _: &mut (), _: &mut (), _: Exec<'_>) -> Result<(), ()> {
        Err(())
    }
}

/// A warm `Split` lends on the stack: no heap allocation on the caller,
/// whether every job succeeds or every leaf fails.
#[test]
fn split_lends_without_allocating_on_success_and_failure() {
    let (noop, failing) = (Noop(64), Failing(64));
    let mut engine = engine(8, BatchPolicy::default(), row(60));
    let mut states = vec![(); 64];
    {
        let mut lanes = [()];
        let mut call = |engine: &mut CpuEngine, fails: bool| {
            on(engine, 8, |exec| {
                counting_alloc::measure(|| {
                    let outcome = if fails {
                        run_native(exec, &failing, split(8), &mut states, &mut lanes)
                    } else {
                        run_native(exec, &noop, split(8), &mut states, &mut lanes)
                    };
                    assert_eq!(outcome.unwrap().is_err(), fails);
                })
                .1
            })
        };
        for fails in [false, true] {
            call(&mut engine, fails);
            assert_eq!(call(&mut engine, fails).calls, 0, "failing {fails}");
        }
    }
}

/// `Send`-only job, lane and error state (`Cell`).
struct Counting(usize);

impl Work for Counting {
    type Job = std::cell::Cell<usize>;
    type Lane = std::cell::Cell<usize>;
    type Error = std::cell::Cell<usize>;
    fn count(&self) -> usize {
        self.0
    }
    fn run(
        &self,
        job: usize,
        state: &mut Self::Job,
        lane: &mut Self::Lane,
        _: Exec<'_>,
    ) -> Result<(), Self::Error> {
        state.set(state.get() + 1);
        lane.set(lane.get() + 1);
        if job == 3 {
            return Err(std::cell::Cell::new(job));
        }
        Ok(())
    }
}

/// `Send`-only job, lane and error state runs on `Placed` lanes and on
/// `Each`'s one-slot fan-out, each lending a job's state to one lane
/// exclusively.
#[test]
fn placed_and_each_lend_send_only_state() {
    let work = Counting(4);
    let mut engine = engine(4, forced_lanes(2), row(60));
    let mut tables = LaneTables::default();
    on(&mut engine, 4, |exec| {
        place(&exec, &uniform(4, 1000), &mut tables)
    });
    for each in [false, true] {
        let mut states = vec![std::cell::Cell::new(0); 4];
        let mut lanes = vec![std::cell::Cell::new(0); 4];
        let outcome = on(&mut engine, 4, |exec| {
            if each {
                run_native(exec, &work, Each, &mut states, &mut lanes)
            } else {
                let tables = &mut tables;
                run_native(exec, &work, Placed { tables }, &mut states, &mut lanes)
            }
        });
        assert_eq!(outcome.unwrap().map_err(|(job, _)| job), Err(3));
        assert!(states.iter().all(|state| state.get() == 1));
        assert_eq!(lanes.iter().map(std::cell::Cell::get).sum::<usize>(), 4);
    }
}

/// A refusal with its context's error forgotten, to compare across runners.
fn unit<R>(refusal: Refusal<R>) -> Refusal<()> {
    match refusal {
        Refusal::Context(_) => Refusal::Context(()),
        Refusal::States { jobs, given } => Refusal::States { jobs, given },
        Refusal::Lanes { needed, given } => Refusal::Lanes { needed, given },
    }
}

/// A state or lane supply short of what an assignment lends is refused
/// before any job runs: `Each` and `Split` against the work's jobs (a
/// nonempty `Split` lends one lane), `Placed` against the placement.
#[test]
fn short_supplies_are_refused_before_any_job() {
    let short = [
        (
            2,
            1,
            Refusal::Lanes {
                needed: 2,
                given: 1,
            },
        ),
        (1, 2, Refusal::States { jobs: 2, given: 1 }),
    ];
    let spy = one_stage(2, &[]);
    let mut pool = engine(2, BatchPolicy::default(), row(60));
    for (states, lanes, refusal) in short {
        let mut seen = vec![vec![]; states];
        let outcome = on(&mut pool, 2, |exec| {
            run_native(exec, &spy, Each, &mut seen, &mut vec![(); lanes])
        });
        assert_eq!(outcome.map(|_| ()).map_err(unit), Err(refusal), "Each");
        assert!(seen.iter().all(Vec::is_empty));
    }
    for (states, lanes, refusal) in [
        (1, 1, Refusal::States { jobs: 2, given: 1 }),
        (
            2,
            0,
            Refusal::Lanes {
                needed: 1,
                given: 0,
            },
        ),
    ] {
        let mut seen = vec![vec![]; states];
        let outcome = on(&mut pool, 2, |exec| {
            run_native(exec, &spy, split(2), &mut seen, &mut vec![(); lanes])
        });
        assert_eq!(outcome.map(|_| ()).map_err(unit), Err(refusal), "Split");
        assert!(seen.iter().all(Vec::is_empty));
    }

    let work = packed(&[1, 1], vec![]);
    let mut tables = LaneTables::default();
    let mut pool = engine(2, outer(), row(60));
    on(&mut pool, 2, |exec| {
        place(&exec, &packed_times(&work), &mut tables)
    });
    assert_eq!(tables.placement().lanes(), 2);
    for (states, lanes, refusal) in short {
        let (mut components, mut buffers) = (components(states), lane_buffers(lanes));
        let outcome = on(&mut pool, 2, |exec| {
            let tables = &mut tables;
            run_native(
                exec,
                &work,
                Placed { tables },
                &mut components,
                &mut buffers,
            )
        });
        assert_eq!(outcome.map(|_| ()).map_err(unit), Err(refusal), "Placed");
        assert!(buffers.iter().all(|lane| lane.offers.is_empty()));
    }
}

fn forced_lanes(lanes: usize) -> BatchPolicy {
    BatchPolicy::new(BatchStrategy::Split {
        lanes: NonZeroUsize::new(lanes).unwrap(),
    })
}

/// The runner's own allocations: those larger than the boxed OS lock a
/// platform's std `Mutex` may allocate on its first lock (64 bytes for
/// Apple's pthread mutex), which each lane part pays as main's runner does.
const RUNNER_TABLES: core::ops::RangeInclusive<usize> = 65..=usize::MAX;

/// A `Placed` run on fresh tables makes as many runner allocations at two
/// lanes as at eight.
#[test]
fn fresh_placed_allocations_do_not_depend_on_the_lane_count() {
    let work = Noop(16);
    let mut states = vec![(); 16];
    let fresh = |lanes: usize, states: &mut [()]| {
        let mut engine = engine(8, forced_lanes(lanes), row(60));
        let mut lane_states = vec![(); lanes];
        let mut placed = || {
            let mut tables = LaneTables::default();
            on(&mut engine, 8, |exec| {
                place(&exec, &uniform(16, 1000), &mut tables)
            });
            assert_eq!(tables.placement().lanes(), lanes);
            on(&mut engine, 8, |exec| {
                counting_alloc::measure_matching(RUNNER_TABLES, || {
                    run_native(
                        exec,
                        &work,
                        Placed {
                            tables: &mut tables,
                        },
                        &mut *states,
                        &mut lane_states,
                    )
                    .unwrap()
                    .unwrap();
                })
                .1
                .matched_calls
            })
        };
        // The first run also warms the engine's fork of this many lanes.
        placed();
        placed()
    };
    let (two, eight) = (fresh(2, &mut states), fresh(8, &mut states));
    counting_alloc::assert_same_calls("fresh Placed tables, lanes", two, eight);
}

/// After a 64-lane run, a narrower run lends only its own lane parts and
/// makes no runner allocation: no work scales with earlier calls' lanes.
#[test]
fn a_placed_run_touches_only_its_own_lanes_after_wider_calls() {
    let work = Noop(64);
    let mut states = vec![(); 64];
    let mut lane_states = vec![(); 64];
    let mut tables = LaneTables::default();
    let mut call = |lanes: usize| {
        let mut engine = engine(64, forced_lanes(lanes), row(60));
        on(&mut engine, 64, |exec| {
            place(&exec, &uniform(64, 1000), &mut tables)
        });
        let placed = tables.placement().lanes();
        let (lent, allocs) = on(&mut engine, 64, |exec| {
            let recording = crate::probe::record();
            let ((), allocs) = counting_alloc::measure_matching(RUNNER_TABLES, || {
                run_native(
                    exec,
                    &work,
                    Placed {
                        tables: &mut tables,
                    },
                    &mut states,
                    &mut lane_states,
                )
                .unwrap()
                .unwrap();
            });
            (recording.get(crate::probe::Event::RunnerLanes), allocs)
        });
        (placed, lent, allocs.matched_calls)
    };
    call(64);
    call(2);
    let (placed, lent, allocations) = call(2);
    assert_eq!((lent, allocations), (placed, 0));
}
