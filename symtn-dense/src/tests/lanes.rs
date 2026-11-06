//! The placement and runner contracts of `crate::lanes`.

use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use tcontract_exec::{BatchPolicy, BatchStrategy, Exec};
use tenferro_cpu::threadteam::{self, DispatchCosts, WidthCosts};
use tenferro_cpu::{CpuEngine, Pool};

use super::counting_alloc;
use crate::lanes::{
    place, run_barrier, run_native, Barrier, Estimate, Form, Jobs, LaneTables, Placement,
    StageKind, Work,
};
use crate::{
    DenseDotConfig, DenseError, DenseExecutor, DenseRead, DenseScalar, DenseTensor, DenseView,
    DenseViewMut, DenseWrite, Engine,
};

use StageKind::{Movement as M, Provider as P};

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

fn serial(ns: u64) -> Estimate {
    Estimate {
        serial: Duration::from_nanos(ns),
        payback: None,
        sections: 1,
        widen: true,
    }
}

fn capped(ns: u64, payback: u64) -> Estimate {
    Estimate {
        serial: Duration::from_nanos(ns),
        payback: Some(Duration::from_nanos(payback)),
        sections: 1,
        widen: true,
    }
}

/// Explicit stage lists; counts the estimates placement reads.
struct Batch {
    jobs: Vec<Vec<(StageKind, Estimate)>>,
    reads: AtomicUsize,
}

impl Batch {
    fn new(jobs: Vec<Vec<(StageKind, Estimate)>>) -> Self {
        Self {
            jobs,
            reads: AtomicUsize::new(0),
        }
    }
}

impl Jobs for Batch {
    fn count(&self) -> usize {
        self.jobs.len()
    }
    fn stages(&self, job: usize) -> usize {
        self.jobs[job].len()
    }
    fn kind(&self, job: usize, stage: usize) -> StageKind {
        self.jobs[job][stage].0
    }
    fn estimate(&self, job: usize, stage: usize) -> Estimate {
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.jobs[job][stage].1
    }
}

fn placed(exec: &Exec<'_>, batch: &Batch, form: Form) -> Placement {
    let mut tables = LaneTables::default();
    place(exec, batch, form, &mut tables);
    tables.placement().clone()
}

/// Nonzero shares summing to at most `budget`, every job in exactly one
/// lane, ascending within it, and every stage at a width its lane holds.
fn assert_legal(p: &Placement, batch: &Batch, budget: usize) {
    let jobs = batch.count();
    if jobs == 0 {
        assert_eq!(p.lanes(), 0);
        return;
    }
    let mut seen = vec![0; jobs];
    let mut sum = 0;
    for l in 0..p.lanes() {
        let share = p.share(l);
        assert!(share >= 1);
        sum += share;
        assert!(p.jobs(l).windows(2).all(|w| w[0] < w[1]));
        for &j in p.jobs(l) {
            seen[j] += 1;
            for k in 0..batch.stages(j) {
                assert!((1..=share).contains(&p.selected(j, k, share)));
            }
        }
    }
    assert!(sum <= budget, "shares {sum} exceed budget {budget}");
    assert!(seen.iter().all(|&n| n == 1));
}

fn fixtures() -> Vec<Batch> {
    let mixed = (0..20)
        .map(|j| {
            let e = (j * 37 % 11 + 1) * 300;
            match j % 3 {
                0 => vec![(P, serial(e))],
                1 => vec![(M, serial(e / 10)), (P, serial(e)), (M, serial(e / 10))],
                _ => vec![(P, capped(e * 4, e / 4))],
            }
        })
        .collect();
    vec![
        Batch::new(vec![]),
        Batch::new(vec![vec![(P, serial(5000))]]),
        // More jobs than any budget, of mixed stages and caps.
        Batch::new(mixed),
        // A zero-k GEMM batch: work 0 gives a zero estimate, which still runs.
        Batch::new(vec![vec![(P, serial(0))]; 3]),
        // Payback below the serial estimate.
        Batch::new(vec![
            vec![(P, capped(9000, 200))],
            vec![(P, capped(9000, 200))],
            vec![(P, serial(4000))],
            vec![(P, serial(0))],
        ]),
    ]
}

#[test]
fn placement_is_legal_for_every_policy_budget_and_batch() {
    let lanes = |n| BatchStrategy::Split {
        lanes: NonZeroUsize::new(n).unwrap(),
    };
    let strategies = [
        BatchStrategy::Auto,
        BatchStrategy::Sequential,
        BatchStrategy::Outer,
        BatchStrategy::Inner,
        lanes(2),
        lanes(3),
    ];
    for strategy in strategies {
        let mut engine = engine(8, BatchPolicy::new(strategy), row(60));
        for budget in 1..=8 {
            for batch in &fixtures() {
                for form in [Form::Native, Form::Barrier] {
                    let p = on(&mut engine, budget, |exec| placed(&exec, batch, form));
                    assert_legal(&p, batch, budget);
                }
            }
        }
    }
}

#[test]
fn placement_is_deterministic() {
    let mut engine = engine(8, BatchPolicy::default(), row(60));
    for batch in &fixtures() {
        on(&mut engine, 7, |exec| {
            let mut reused = LaneTables::default();
            place(&exec, batch, Form::Native, &mut reused);
            let first = reused.placement().clone();
            place(&exec, batch, Form::Native, &mut reused);
            assert_eq!(reused.placement(), &first);
            assert_eq!(placed(&exec, batch, Form::Native), first);
        });
    }
}

#[test]
fn zero_estimates_run_on_one_slot_and_empty_places_nothing() {
    let mut engine = engine(8, BatchPolicy::default(), row(60));
    on(&mut engine, 8, |exec| {
        let zero = Batch::new(vec![vec![(P, serial(0))]; 3]);
        let p = placed(&exec, &zero, Form::Native);
        assert_eq!((p.lanes(), p.share(0), p.jobs(0)), (1, 1, &[0, 1, 2][..]));
        assert!((0..3).all(|j| p.selected(j, 0, 8) == 1));
        assert_eq!(placed(&exec, &Batch::new(vec![]), Form::Native).lanes(), 0);
    });
}

#[test]
fn one_slot_budget_reads_no_estimate() {
    let mut engine = engine(8, BatchPolicy::default(), row(60));
    let batch = &fixtures()[2];
    let p = on(&mut engine, 1, |exec| placed(&exec, batch, Form::Native));
    assert_eq!(batch.reads.load(Ordering::Relaxed), 0);
    assert_eq!((p.lanes(), p.share(0)), (1, 1));
}

/// The placement as it runs: each lane's jobs and every stage's selected
/// width. (A lane's share past its stages' widths runs nothing: equal shares
/// of a larger budget only add such slots.)
fn as_run(p: &Placement, batch: &Batch) -> (Vec<Vec<usize>>, Vec<usize>) {
    let lanes = (0..p.lanes()).map(|l| p.jobs(l).to_vec()).collect();
    let mut widths = Vec::new();
    for l in 0..p.lanes() {
        for &j in p.jobs(l) {
            widths.extend((0..batch.stages(j)).map(|k| p.selected(j, k, p.share(l))));
        }
    }
    (lanes, widths)
}

/// Once every lane's share covers every stage's width (a budget of at least
/// the job count times the widest unrestricted stage query, serial or
/// payback), the stage widths, the candidates' scores and hence what runs
/// are fixed: placing on any budget from there to the pool gives the same
/// lanes and stage widths in both forms.
#[test]
fn placement_is_identical_on_every_budget_from_saturation_to_the_pool() {
    let pool = 112;
    let costs = row(60);
    let cost = |k: NonZeroUsize| costs.at(k);
    let full = NonZeroUsize::new(pool).unwrap();
    let rule = |e: Duration| threadteam::width_for_by(e, 1, full, cost).get();
    let mut engine = engine(pool, BatchPolicy::default(), costs);
    let mixed = vec![
        vec![(P, serial(1000))],
        vec![(P, capped(2000, 300))],
        vec![(M, serial(50)), (P, serial(900)), (M, serial(50))],
        vec![(P, serial(0))],
        vec![(M, serial(100))],
    ];
    let mut big_and_tiny = vec![vec![(P, serial(1000))]];
    big_and_tiny.extend(vec![vec![(P, serial(100))]; 10]);
    for jobs in [vec![vec![(P, serial(1000))]; 5], big_and_tiny, mixed] {
        let batch = Batch::new(jobs);
        let widest = batch
            .jobs
            .iter()
            .flatten()
            .map(|(_, e)| {
                e.payback
                    .map_or(rule(e.serial), |p| rule(e.serial).max(rule(p)))
            })
            .max()
            .unwrap();
        let saturation = batch.jobs.len() * widest;
        assert!(saturation < pool, "fixture saturates below the pool");
        for form in [Form::Native, Form::Barrier] {
            let mut at = |budget| on(&mut engine, budget, |exec| placed(&exec, &batch, form));
            let expected = as_run(&at(pool), &batch);
            if form == Form::Native {
                assert!(expected.0.len() > 1, "the fixture places lanes");
            }
            for budget in saturation..pool {
                assert_eq!(as_run(&at(budget), &batch), expected, "budget {budget}");
            }
        }
    }
}

/// Wide jobs queued between narrow ones leave the placement queue with one
/// entry per lane, not one per queued job.
#[test]
fn the_placement_queue_holds_one_entry_per_lane() {
    let mut jobs = vec![vec![(P, serial(1000))]; 64];
    jobs.push(vec![(P, serial(1))]);
    let batch = Batch::new(jobs);
    let budget = 2;
    let mut engine = engine(budget, BatchPolicy::default(), row(60));
    on(&mut engine, budget, |exec| {
        let mut tables = LaneTables::default();
        place(&exec, &batch, Form::Native, &mut tables);
        assert!(tables.queued() <= budget, "{} entries", tables.queued());
    });
}

/// Native job times divide each width bucket's summed estimate once: jobs
/// `[180, 180]` and `[185, 185]` (width 6) on six admitted slots at a flat
/// 60 ns take two lanes of 3 slots, `360/3 + 120 = 240` and `370/3 + 120 =
/// 243` (per stage it would be 242), scoring `243 + 60 = 303`.
#[test]
fn native_candidates_share_the_bucket_rounding() {
    let batch = Batch::new(vec![
        vec![(P, serial(180)), (P, serial(180))],
        vec![(P, serial(185)), (P, serial(185))],
    ]);
    let mut engine = engine(6, BatchPolicy::default(), row(60));
    on(&mut engine, 6, |exec| {
        let mut tables = LaneTables::default();
        place(&exec, &batch, Form::Native, &mut tables);
        assert_eq!(tables.scores().0, Some(303));
        let p = tables.placement();
        assert_eq!((p.lanes(), p.share(0), p.share(1)), (2, 3, 3));
    });
}

/// The call is admitted on the whole budget, each lane at most the widest
/// offer of any stage: eight 200 ns GEMMs of width six on sixteen slots take
/// eight lanes of two, `100 + 60 + D(8) = 240` (four lanes of four score
/// 280, one lane 744); two of them take two lanes of six, not of eight.
#[test]
fn admission_is_the_whole_budget_with_lanes_capped_at_the_widest_offer() {
    let mut engine = engine(16, BatchPolicy::default(), row(60));
    let mut shares = |jobs: usize| {
        let batch = Batch::new(vec![vec![(P, serial(200))]; jobs]);
        let p = on(&mut engine, 16, |exec| placed(&exec, &batch, Form::Native));
        (0..p.lanes()).map(|l| p.share(l)).collect::<Vec<_>>()
    };
    assert_eq!(shares(8), [2; 8]);
    assert_eq!(shares(2), [6, 6]);
}

/// Jobs that do not widen, of estimates `[100, 700, 300, 500, 200]` on two
/// slots, take the no-search baseline's two one-slot lanes (the 1,800 ns
/// total pays for a section of two) and fill them largest first: 700 and
/// 500 open the lanes, 300 joins 500, 200 joins 700 and 100 joins 800, so
/// lane 0 holds `[1, 4]` and lane 1 `[0, 2, 3]`.
#[test]
fn jobs_that_do_not_widen_fill_their_lanes_largest_first() {
    let flat = |ns| Estimate {
        widen: false,
        ..serial(ns)
    };
    let batch = Batch::new(
        [100, 700, 300, 500, 200]
            .into_iter()
            .map(|ns| vec![(P, flat(ns))])
            .collect(),
    );
    let mut engine = engine(2, BatchPolicy::default(), row(60));
    let p = on(&mut engine, 2, |exec| placed(&exec, &batch, Form::Native));
    assert_eq!(
        (0..p.lanes())
            .map(|l| p.jobs(l).to_vec())
            .collect::<Vec<_>>(),
        [vec![1, 4], vec![0, 2, 3]]
    );
}

/// A lane count strictly between the baseline and one wins: four 600 ns
/// GEMMs on six admitted slots at a flat 60 ns score 660 on the baseline's
/// four lanes `[2, 2, 1, 1]`, 780 on three, 580 on two lanes of 3 and 640 on
/// one; the reservation's four one-slot lanes score 660. Two lanes win, each
/// with two jobs, largest first, ties to the lowest lane.
#[test]
fn an_intermediate_lane_count_beats_both_endpoints() {
    let batch = Batch::new(vec![vec![(P, serial(600))]; 4]);
    let mut engine = engine(6, BatchPolicy::default(), row(60));
    let p = on(&mut engine, 6, |exec| placed(&exec, &batch, Form::Native));
    let lanes: Vec<_> = (0..p.lanes())
        .map(|l| (p.share(l), p.jobs(l).to_vec()))
        .collect();
    assert_eq!(lanes, [(3, vec![0, 2]), (3, vec![1, 3])]);
}

/// The native placement is the model's best equal-share candidate, ties to
/// the larger lane count, unless the reservation is strictly faster, and the
/// no-search baseline when no stage widens: over every two- to four-job
/// batch drawn from one-stage GEMMs that widen or not, a pack, GEMM and
/// scatter, and a non-widening stage followed by a widening one, on budgets
/// two to seven and flat, nonmonotone and uneven cost rows. Expected values
/// come from the model computed here: stage widths by threadteam's rule (one
/// for a stage that does not widen), job times bucketed by width,
/// largest-first fill, slowest lane plus `D(L)`, and for the baseline the
/// largest paying count up to `min(width_for(E_tot), J)`.
#[test]
fn native_placement_matches_the_model_on_a_mixed_matrix() {
    let kinds: [Vec<(StageKind, u64, bool)>; 6] = [
        vec![(P, 600, true)],
        vec![(P, 4000, true)],
        vec![(P, 50, true)],
        vec![(P, 1500, false)],
        vec![(M, 200, true), (P, 1500, true), (M, 200, true)],
        vec![(P, 600, false), (P, 4000, true)],
    ];
    let rows = [
        row(60),
        WidthCosts::new([100, 1, 1].map(Duration::from_nanos)),
        WidthCosts::new([10, 200, 30].map(Duration::from_nanos)),
    ];
    for costs in rows {
        let d = |x: usize| costs.at(NonZeroUsize::new(x).unwrap()).as_nanos() as u64;
        let pays = |e: u64, x: usize| {
            x < 2
                || threadteam::pays_at(
                    Duration::from_nanos(e),
                    1,
                    costs.at(NonZeroUsize::new(x).unwrap()),
                )
        };
        let mut engine = engine(7, BatchPolicy::default(), costs);
        for budget in 2..=7 {
            let rule = |e: u64| {
                threadteam::width_for_by(
                    Duration::from_nanos(e),
                    1,
                    NonZeroUsize::new(budget).unwrap(),
                    |k| costs.at(k),
                )
                .get()
            };
            for count in 2..=4u32 {
                for code in 0..6usize.pow(count) {
                    let jobs: Vec<&Vec<(StageKind, u64, bool)>> = (0..count)
                        .map(|i| &kinds[code / 6usize.pow(i) % 6])
                        .collect();
                    // Per job its width buckets `(w, sum E, stages)`.
                    let buckets: Vec<Vec<(usize, u64, u64)>> = jobs
                        .iter()
                        .map(|stages| {
                            let mut b: Vec<(usize, u64, u64)> = Vec::new();
                            for &(_, e, widen) in stages.iter() {
                                let w = if widen { rule(e) } else { 1 };
                                match b.iter_mut().find(|x| x.0 == w) {
                                    Some(x) => (x.1, x.2) = (x.1 + e, x.2 + 1),
                                    None => b.push((w, e, 1)),
                                }
                            }
                            b
                        })
                        .collect();
                    let time = |j: usize, a: usize| -> u64 {
                        buckets[j]
                            .iter()
                            .map(|&(w, e, n)| {
                                let x = a.min(w);
                                if x < 2 {
                                    e
                                } else {
                                    e / x as u64 + n * d(x)
                                }
                            })
                            .sum()
                    };
                    let e: Vec<u64> = jobs.iter().map(|s| s.iter().map(|x| x.1).sum()).collect();
                    let total: u64 = e.iter().sum();
                    let batch = Batch::new(
                        jobs.iter()
                            .map(|stages| {
                                stages
                                    .iter()
                                    .map(|&(k, ns, widen)| {
                                        (
                                            k,
                                            Estimate {
                                                widen,
                                                ..serial(ns)
                                            },
                                        )
                                    })
                                    .collect()
                            })
                            .collect(),
                    );
                    let p = on(&mut engine, budget, |exec| {
                        placed(&exec, &batch, Form::Native)
                    });
                    let what = format!("{e:?} on {budget}");
                    if buckets.iter().all(|b| b.iter().all(|x| x.0 == 1)) {
                        let lanes = (1..=rule(total).min(e.len()))
                            .rev()
                            .find(|&l| pays(total, l))
                            .unwrap();
                        assert_eq!(p.lanes(), lanes, "{what}");
                        assert!((0..p.lanes()).all(|l| p.share(l) == 1), "{what}");
                        continue;
                    }
                    let mut best = (u64::MAX, std::cmp::Reverse(0));
                    for lanes in 1..=budget.min(e.len()) {
                        if !pays(total, lanes) {
                            continue;
                        }
                        let share = |l: usize| budget / lanes + usize::from(l < budget % lanes);
                        let mut order: Vec<usize> = (0..e.len()).collect();
                        order.sort_by_key(|&j| (std::cmp::Reverse(e[j]), j));
                        let mut load = vec![(0u64, 0usize); lanes];
                        for j in order {
                            let l = (0..lanes).min_by_key(|&l| (load[l], l)).unwrap();
                            load[l] = (load[l].0 + time(j, share(l)), load[l].1 + 1);
                        }
                        let fork = if lanes < 2 { 0 } else { d(lanes) };
                        let score = load.iter().map(|&(t, _)| t).max().unwrap() + fork;
                        best = best.min((score, std::cmp::Reverse(lanes)));
                    }
                    let slowest = (0..p.lanes())
                        .map(|l| p.jobs(l).iter().map(|&j| time(j, p.share(l))).sum::<u64>())
                        .max()
                        .unwrap();
                    let fork = if p.lanes() < 2 { 0 } else { d(p.lanes()) };
                    let chosen = slowest + fork;
                    assert!(chosen <= best.0, "{what}: {chosen} > {}", best.0);
                    if chosen == best.0 {
                        assert_eq!(p.lanes(), best.1 .0, "{what}: tie to the larger count");
                    }
                }
            }
        }
    }
}

/// When no job's time depends on its share, there is no search: 96 jobs
/// that do not widen, of a ramp of estimates, take 96 one-slot lanes on 96
/// slots, one job each, although fewer lanes would cost a smaller fork for
/// the same longest job.
#[test]
fn jobs_whose_time_ignores_the_share_take_the_baseline_lanes() {
    let batch = Batch::new(
        (0..96u64)
            .map(|i| {
                let flat = Estimate {
                    widen: false,
                    ..serial(10_000 + 1_000 * i)
                };
                vec![(P, flat)]
            })
            .collect(),
    );
    let mut engine = engine(96, BatchPolicy::default(), row(60));
    let p = on(&mut engine, 96, |exec| placed(&exec, &batch, Form::Native));
    assert_eq!(p.lanes(), 96);
    assert!((0..96).all(|l| p.share(l) == 1 && p.jobs(l).len() == 1));
}

/// The no-search baseline's lanes are one slot each, whatever the budget:
/// two 10 us jobs that do not widen on eight slots take two lanes (the
/// 20 us total pays for a section of eight, capped at the two jobs), and
/// each provider stage is offered one slot, not four.
#[test]
fn no_search_lanes_offer_one_slot_each() {
    let flat = Estimate {
        widen: false,
        ..serial(10_000)
    };
    let spy = spy(vec![vec![(P, flat)]; 2]);
    let mut engine = engine(8, BatchPolicy::default(), row(60));
    let p = on(&mut engine, 8, |exec| {
        placed(&exec, &spy.batch, Form::Native)
    });
    assert_eq!((p.lanes(), p.share(0), p.share(1)), (2, 1, 1));
    assert_eq!(run_spy(&mut engine, 8, &spy), [[(0, 1)], [(0, 1)]]);
}

/// On a dispatch row of 1000 ns at two slots and 1 ns from four (501 ns at
/// three), three 100 ns jobs that do not widen on four slots pay for a
/// section of four (`300 >= 3 * 1`), capped at three lanes, whose fork does
/// not pay (`300 < 3 * 501`), nor does two (`300 < 3 * 1000`): one lane of
/// one slot.
#[test]
fn no_search_lowers_a_lane_count_whose_fork_does_not_pay() {
    let flat = Estimate {
        widen: false,
        ..serial(100)
    };
    let batch = Batch::new(vec![vec![(P, flat)]; 3]);
    let costs = WidthCosts::new([1000, 1, 1].map(Duration::from_nanos));
    let mut engine = engine(4, BatchPolicy::default(), costs);
    let p = on(&mut engine, 4, |exec| placed(&exec, &batch, Form::Native));
    assert_eq!((p.lanes(), p.share(0)), (1, 1));
}

/// The barrier form keeps its scored search for one-slot stages: two jobs of
/// a thousand 1 ns provider stages on two slots at 60 ns score
/// `2000 + 1000 * 60 = 62000` on two lanes (a fork per round) and 2000 on
/// one, which wins.
#[test]
fn barrier_jobs_of_one_slot_stages_keep_the_scored_search() {
    let batch = Batch::new(vec![vec![(P, serial(1)); 1000]; 2]);
    let mut engine = engine(2, BatchPolicy::default(), row(60));
    on(&mut engine, 2, |exec| {
        let mut tables = LaneTables::default();
        place(&exec, &batch, Form::Barrier, &mut tables);
        assert_eq!(tables.scores(), (Some(62000), Some(2000)));
        assert_eq!(tables.placement().lanes(), 1);
    });
}

/// The reservation replaces the equal shares only by scoring strictly lower:
/// on eight slots at dispatch costs of 1, 10 and 20 ns at two, four and six
/// slots, a 100 ns job (payback 5 ns, width 2) and a 250 ns job (width 8)
/// score 73 on two equal lanes of four and 108 on one lane; the reservation
/// gives them dedicated lanes of `min(2, floor(8 * 100 / 350)) = 2` and
/// `min(8, floor(8 * 250 / 350)) = 5`, scoring `max(51, 65) + 1 = 66`.
#[test]
fn a_strictly_faster_reservation_replaces_the_equal_shares() {
    let batch = Batch::new(vec![vec![(P, capped(100, 5))], vec![(P, serial(250))]]);
    let costs = WidthCosts::new([1, 10, 20].map(Duration::from_nanos));
    let mut engine = engine(8, BatchPolicy::default(), costs);
    on(&mut engine, 8, |exec| {
        let mut tables = LaneTables::default();
        place(&exec, &batch, Form::Native, &mut tables);
        assert_eq!(tables.scores().0, Some(73));
        let p = tables.placement();
        assert_eq!((p.lanes(), p.share(0), p.share(1)), (2, 2, 5));
    });
}

/// Many large blocks whose provider does not widen spread over one-slot
/// lanes rather than running one after another on the whole budget, as the
/// width each estimate pays for would suggest under the linear model.
#[test]
fn blocks_that_do_not_widen_run_on_balanced_lanes() {
    let block = Estimate {
        widen: false,
        ..serial(100_000)
    };
    let batch = Batch::new(vec![vec![(P, block)]; 96]);
    let mut engine = engine(8, BatchPolicy::default(), row(60));
    let p = on(&mut engine, 8, |exec| placed(&exec, &batch, Form::Native));
    assert_eq!(p.lanes(), 8);
    assert!((0..8).all(|l| p.share(l) == 1 && p.jobs(l).len() == 12));
}

/// Records the budget every stage is offered; fails the listed stages.
struct Spy {
    batch: Batch,
    fail: Vec<(usize, usize)>,
}

impl Jobs for Spy {
    fn count(&self) -> usize {
        self.batch.count()
    }
    fn stages(&self, job: usize) -> usize {
        self.batch.stages(job)
    }
    fn kind(&self, job: usize, stage: usize) -> StageKind {
        self.batch.kind(job, stage)
    }
    fn estimate(&self, job: usize, stage: usize) -> Estimate {
        self.batch.estimate(job, stage)
    }
}

fn injected(job: usize) -> DenseError {
    DenseError::Unsupported {
        op: "injected",
        message: job.to_string(),
    }
}

impl Work for Spy {
    type Job = Vec<(usize, usize)>;
    type Lane = ();
    type Error = DenseError;
    fn run(
        &self,
        job: usize,
        stage: usize,
        seen: &mut Self::Job,
        _: &mut (),
        exec: Exec<'_>,
    ) -> Result<(), DenseError> {
        seen.push((stage, exec.budget()));
        if self.fail.contains(&(job, stage)) {
            return Err(injected(job));
        }
        Ok(())
    }
}

fn spy(jobs: Vec<Vec<(StageKind, Estimate)>>) -> Spy {
    Spy {
        batch: Batch::new(jobs),
        fail: vec![],
    }
}

fn run_spy(engine: &mut CpuEngine, budget: usize, spy: &Spy) -> Vec<Vec<(usize, usize)>> {
    let mut seen = vec![vec![]; spy.count()];
    on(engine, budget, |mut exec| {
        let mut tables = LaneTables::default();
        place(&exec, spy, Form::Native, &mut tables);
        let lanes = tables.placement().lanes();
        run_native(
            exec.reborrow(),
            spy,
            &mut tables,
            &mut seen,
            &mut vec![(); lanes],
        )
        .unwrap()
        .unwrap();
    });
    seen
}

/// One pack, GEMM and scatter: the cheap moves run on one slot while the
/// GEMM's token is limited to its own width at the budget.
#[test]
fn movement_runs_at_its_width_and_the_gemm_token_at_its_own() {
    let spy = spy(vec![vec![
        (M, serial(46)),
        (P, serial(606)),
        (M, serial(46)),
    ]]);
    let mut engine = engine(6, BatchPolicy::default(), row(60));
    assert_eq!(run_spy(&mut engine, 6, &spy), [[(0, 1), (1, 6), (2, 1)]]);
    assert_eq!(run_spy(&mut engine, 4, &spy), [[(0, 1), (1, 4), (2, 1)]]);
}

/// A forced `Inner` offers every provider stage the whole budget, and reads
/// no estimate.
#[test]
fn forced_inner_offers_every_provider_stage_the_whole_budget() {
    let spy = spy(vec![vec![(P, serial(2 * 64))], vec![(P, serial(2 * 80))]]);
    let mut engine = engine(6, BatchPolicy::new(BatchStrategy::Inner), row(60));
    assert_eq!(run_spy(&mut engine, 6, &spy), [[(0, 6)], [(0, 6)]]);
    assert_eq!(spy.batch.reads.load(Ordering::Relaxed), 0);
}

/// Two components of a pack (46 ns), a GEMM (606 ns) and a scatter at a
/// flat 60 ns dispatch on six slots: native lanes overlap the GEMMs and win,
/// while the barrier form serializes them on the engine and the one-lane
/// run wins (534 against 506 ns).
#[test]
fn barrier_scoring_serializes_the_gemms() {
    let component = vec![(M, serial(46)), (P, serial(606)), (M, serial(46))];
    let batch = Batch::new(vec![component.clone(), component]);
    let mut engine = engine(6, BatchPolicy::default(), row(60));
    on(&mut engine, 6, |exec| {
        let mut tables = LaneTables::default();
        place(&exec, &batch, Form::Native, &mut tables);
        assert_eq!(tables.placement().lanes(), 2);
        place(&exec, &batch, Form::Barrier, &mut tables);
        assert_eq!(tables.scores(), (Some(534), Some(506)));
        assert_eq!(tables.placement().lanes(), 1);
    });
}

/// A lane ending in its provider stage needs no further round: two lone
/// 606 ns GEMMs price one fork, `2 (606/6 + 60) + 60 = 382`, against 322 on
/// one lane.
#[test]
fn barrier_scoring_prices_only_the_rounds_entered() {
    let batch = Batch::new(vec![vec![(P, serial(606))]; 2]);
    let mut engine = engine(6, BatchPolicy::default(), row(60));
    on(&mut engine, 6, |exec| {
        let mut tables = LaneTables::default();
        place(&exec, &batch, Form::Barrier, &mut tables);
        assert_eq!(tables.scores(), (Some(382), Some(322)));
    });
}

/// Pack, GEMM and scatter of `ROWS x K` inputs against one `K x COLS`
/// right-hand side through each lane's own initialized buffers.
struct Packed {
    batch: Batch,
    rhs: Vec<f64>,
    fail: Vec<(usize, usize)>,
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
    /// The budget each movement stage on this lane was offered.
    offers: Vec<usize>,
}

fn gemm(dst: &mut [f64], lhs: &[f64], rhs: &[f64]) {
    for c in 0..COLS {
        for r in 0..ROWS {
            dst[r + ROWS * c] = (0..K).map(|k| lhs[r + ROWS * k] * rhs[k + K * c]).sum();
        }
    }
}

impl Jobs for Packed {
    fn count(&self) -> usize {
        self.batch.count()
    }
    fn stages(&self, job: usize) -> usize {
        self.batch.stages(job)
    }
    fn kind(&self, job: usize, stage: usize) -> StageKind {
        self.batch.kind(job, stage)
    }
    fn estimate(&self, job: usize, stage: usize) -> Estimate {
        self.batch.estimate(job, stage)
    }
}

impl Work for Packed {
    type Job = Component;
    type Lane = LaneBuffers;
    type Error = DenseError;
    fn run(
        &self,
        job: usize,
        stage: usize,
        state: &mut Component,
        lane: &mut LaneBuffers,
        exec: Exec<'_>,
    ) -> Result<(), DenseError> {
        if self.kind(job, stage) == M {
            lane.offers.push(exec.budget());
        }
        if self.fail.contains(&(job, stage)) {
            return Err(injected(job));
        }
        match stage % 3 {
            0 => {
                lane.source.copy_from_slice(&state.input);
                lane.used_by.push(job);
            }
            1 => gemm(&mut lane.destination, &lane.source, &self.rhs),
            _ => {
                for (out, d) in state.output.iter_mut().zip(&lane.destination) {
                    *out += d;
                }
            }
        }
        Ok(())
    }
}

/// A custom executor on the default batch entry that records the budget the
/// engine offers each GEMM submission.
#[derive(Default)]
struct Recording {
    budgets: Vec<usize>,
}

impl DenseExecutor for Recording {
    fn svd(
        &mut self,
        _: &mut dyn Engine,
        _: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        unreachable!("not a factorization test")
    }
    fn qr(&mut self, _: &mut dyn Engine, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        unreachable!("not a factorization test")
    }
    fn eigh(
        &mut self,
        _: &mut dyn Engine,
        _: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        unreachable!("not a factorization test")
    }
    fn dot_general_into(
        &mut self,
        engine: &mut dyn Engine,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        _: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        let budget = tenferro_cpu::with_cpu_engine(engine, |cpu| {
            cpu.with_contraction_exec(|exec, _, _| Ok(exec.budget()))
                .unwrap()
        });
        self.budgets.push(budget.unwrap());
        let (DenseWrite::F64(mut out), DenseRead::F64(lhs), DenseRead::F64(rhs)) =
            (output, lhs, rhs)
        else {
            unreachable!("an f64 test")
        };
        gemm(out.data_mut(), lhs.data(), rhs.data());
        Ok(())
    }
}

/// The barrier form's engine: one contraction entry per round, and each GEMM
/// submitted alone through the executor over its lane's buffers.
struct Host {
    engine: CpuEngine,
    executor: Recording,
    rhs: Vec<f64>,
    fail: Vec<(usize, usize)>,
    attempts: Vec<(usize, usize)>,
    regions: usize,
}

impl Barrier<Packed> for Host {
    type Refused = tenferro_tensor::Error;
    fn region(
        &mut self,
        round: &mut dyn FnMut(Exec<'_>) -> Result<(), threadteam::Error>,
    ) -> Result<(), Self::Refused> {
        self.regions += 1;
        self.engine
            .with_contraction_exec(|exec, _, _| Ok(round(exec.reborrow())))
            .map(|forked| forked.unwrap())
    }
    fn submit(
        &mut self,
        job: usize,
        stage: usize,
        _: &mut Component,
        lane: &mut LaneBuffers,
    ) -> Result<(), DenseError> {
        self.attempts.push((job, stage));
        if self.fail.contains(&(job, stage)) {
            return Err(injected(job));
        }
        let (src_shape, rhs_shape, dst_shape) = ([ROWS, K], [K, COLS], [ROWS, COLS]);
        let (src_strides, rhs_strides, dst_strides) = ([1, ROWS], [1, K], [1, ROWS]);
        self.executor.matmul_axpby_into(
            &mut self.engine,
            DenseWrite::F64(DenseViewMut::new(
                &mut lane.destination,
                &dst_shape,
                &dst_strides,
                0,
            )?),
            DenseRead::F64(DenseView::new(&lane.source, &src_shape, &src_strides, 0)?),
            DenseRead::F64(DenseView::new(&self.rhs, &rhs_shape, &rhs_strides, 0)?),
            DenseScalar::F64(1.0),
            DenseScalar::F64(0.0),
        )
    }
}

fn packed(groups: &[usize], fail: Vec<(usize, usize)>) -> Packed {
    let group = [(M, serial(1000)), (P, serial(1000)), (M, serial(1000))];
    Packed {
        batch: Batch::new(groups.iter().map(|&n| group.repeat(n)).collect()),
        rhs: vec![1.0, 2.0, 3.0, 4.0],
        fail,
    }
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

/// Outcome, outputs, each lane's users and the host of one run on `budget`
/// slots under `policy`.
type Ran = (
    Result<(), usize>,
    Vec<Component>,
    Vec<(Vec<usize>, Vec<usize>, Vec<usize>)>,
    Option<Host>,
);

fn run_packed(work: &Packed, budget: usize, policy: BatchPolicy, form: Form) -> Ran {
    let mut engine = engine(budget, policy, row(60));
    let mut tables = LaneTables::default();
    on(&mut engine, budget, |exec| {
        place(&exec, work, form, &mut tables)
    });
    let mut states = components(work.count());
    let mut lanes = lane_buffers(tables.placement().lanes());
    let (outcome, host) = match form {
        Form::Native => {
            let outcome = on(&mut engine, budget, |exec| {
                run_native(exec, work, &mut tables, &mut states, &mut lanes).unwrap()
            });
            (outcome, None)
        }
        Form::Barrier => {
            let mut host = Host {
                engine,
                executor: Recording::default(),
                rhs: work.rhs.clone(),
                fail: work.fail.clone(),
                attempts: vec![],
                regions: 0,
            };
            let outcome = run_barrier(&mut host, work, &mut tables, &mut states, &mut lanes);
            (outcome.unwrap(), Some(host))
        }
    };
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
    (outcome.map_err(|(j, _)| j), states, users, host)
}

fn outer() -> BatchPolicy {
    BatchPolicy::new(BatchStrategy::Outer)
}

/// Each lane packs, multiplies and scatters through its own initialized
/// source and destination buffers, natively and in barrier rounds, and the
/// outputs are the hand products.
#[test]
fn lanes_run_their_jobs_through_their_own_buffers_in_both_forms() {
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
    for form in [Form::Native, Form::Barrier] {
        let (outcome, states, users, _) = run_packed(&work, 2, outer(), form);
        assert_eq!(outcome, Ok(()));
        let outputs: Vec<_> = states.into_iter().map(|c| c.output).collect();
        assert_eq!(outputs, expected);
        assert_eq!(users.len(), 2);
        for (jobs, used_by, _) in users {
            let expected_users: Vec<usize> = jobs
                .iter()
                .flat_map(|&j| std::iter::repeat_n(j, work.batch.stages(j) / 3))
                .collect();
            assert_eq!(used_by, expected_users);
        }
    }
}

/// Component 0's second GEMM and component 1's first GEMM fail: component
/// 0's error is reported on one lane and on two, and its second GEMM is
/// attempted in both, after component 1 failed.
#[test]
fn a_failure_ends_only_its_own_job_and_the_lowest_index_wins() {
    let fail = vec![(0, 4), (1, 1)];
    let work = packed(&[2, 1], fail.clone());
    for budget in [1, 2] {
        for form in [Form::Native, Form::Barrier] {
            let (outcome, _, users, host) = run_packed(&work, budget, outer(), form);
            assert_eq!(users.len(), budget);
            assert_eq!(outcome, Err(0), "budget {budget}, {form:?}");
            if let Some(host) = host {
                assert!(fail.iter().all(|f| host.attempts.contains(f)));
            }
        }
    }
}

/// In the barrier form every GEMM submission sees the engine's whole budget
/// (four slots), not its lane's share (two), on which the movement runs.
#[test]
fn barrier_submissions_get_the_whole_engine() {
    let work = packed(&[1, 1], vec![]);
    let split = BatchPolicy::new(BatchStrategy::Split {
        lanes: NonZeroUsize::new(2).unwrap(),
    });
    let (outcome, _, users, host) = run_packed(&work, 4, split, Form::Barrier);
    assert_eq!(outcome, Ok(()));
    assert_eq!(users.len(), 2);
    assert_eq!(host.unwrap().executor.budgets, [4, 4]);
    for (_, _, offers) in users {
        assert_eq!(offers, [2, 2]);
    }
}

/// The barrier runner enters one round per provider stage of its longest
/// lane, plus one only when movement follows the last of them.
#[test]
fn barrier_rounds_end_with_the_last_stage() {
    let provider_only = Packed {
        batch: Batch::new(vec![vec![(P, serial(1000))]; 2]),
        ..packed(&[], vec![])
    };
    let (_, _, users, host) = run_packed(&provider_only, 2, outer(), Form::Barrier);
    assert_eq!((users.len(), host.unwrap().regions), (2, 1));
    let (_, _, users, host) = run_packed(&packed(&[1, 1], vec![]), 2, outer(), Form::Barrier);
    assert_eq!((users.len(), host.unwrap().regions), (2, 2));
}

/// No stage work: the runner's own tables are what a warm call reuses.
struct Noop(Batch);

impl Jobs for Noop {
    fn count(&self) -> usize {
        self.0.count()
    }
    fn stages(&self, job: usize) -> usize {
        self.0.stages(job)
    }
    fn kind(&self, job: usize, stage: usize) -> StageKind {
        self.0.kind(job, stage)
    }
    fn estimate(&self, job: usize, stage: usize) -> Estimate {
        self.0.estimate(job, stage)
    }
}

impl Work for Noop {
    type Job = ();
    type Lane = ();
    type Error = ();
    fn run(&self, _: usize, _: usize, _: &mut (), _: &mut (), _: Exec<'_>) -> Result<(), ()> {
        Ok(())
    }
}

/// After a four-lane call warms the tables, a one-lane call and its repeat
/// reuse them: the repeat allocates nothing.
#[test]
fn a_repeated_call_after_fewer_lanes_allocates_nothing() {
    let mut jobs = vec![vec![(P, serial(10_000))]; 3];
    jobs.extend(vec![vec![(P, serial(0))]; 32]);
    let work = Noop(Batch::new(jobs));
    let mut engine = engine(4, outer(), row(60));
    let mut tables = LaneTables::default();
    let mut states = vec![(); work.count()];
    let mut lanes = vec![(); 4];
    let mut call = |engine: &mut CpuEngine, budget| {
        on(engine, budget, |mut exec| {
            counting_alloc::measure(|| {
                place(&exec, &work, Form::Native, &mut tables);
                run_native(exec.reborrow(), &work, &mut tables, &mut states, &mut lanes)
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

/// Dropping used tables frees at most what they charge.
#[test]
fn the_tables_charge_covers_what_they_retain() {
    let spy = spy(fixtures().swap_remove(2).jobs);
    let mut engine = engine(8, BatchPolicy::default(), row(60));
    let mut tables = LaneTables::default();
    let mut seen = vec![vec![]; spy.count()];
    on(&mut engine, 8, |mut exec| {
        place(&exec, &spy, Form::Native, &mut tables);
        let lanes = tables.placement().lanes();
        run_native(
            exec.reborrow(),
            &spy,
            &mut tables,
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
/// stage runs, as the lane fork refuses it.
#[test]
fn a_share_above_the_running_budget_is_refused_before_any_stage() {
    let spy = spy(vec![vec![(P, serial(606))]]);
    let mut engine = engine(6, BatchPolicy::default(), row(60));
    let mut tables = LaneTables::default();
    on(&mut engine, 6, |exec| {
        place(&exec, &spy, Form::Native, &mut tables)
    });
    assert_eq!(
        (tables.placement().lanes(), tables.placement().share(0)),
        (1, 6)
    );
    let mut seen = vec![vec![]];
    let refused = on(&mut engine, 4, |exec| {
        run_native(exec, &spy, &mut tables, &mut seen, &mut [()])
    });
    assert_eq!(
        refused.map(|_| ()),
        Err(threadteam::Error::SharesExceedBudget { budget: 4, sum: 6 })
    );
    assert_eq!(seen, [Vec::<(usize, usize)>::new()]);
}
