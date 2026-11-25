//! One movement model, with engine-local measurements and structural summaries.

use super::*;
use crate::movement::{maximum, Counts, Summary, CLASSES};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};
use symtn_dense::lanes;
use tenferro_cpu::threadteam::HostClass;

const ONLINE_BUDGET: Duration = Duration::from_millis(10);
const ONLINE_WORKING_SETS: [usize; 3] = [64 << 10, 1 << 20, 4 << 20];
const BUFFER_ELEMENTS: usize = ONLINE_WORKING_SETS[2] / std::mem::size_of::<f64>();

#[derive(Clone, Copy, Debug, PartialEq)]
struct Curve<const N: usize> {
    points: [(usize, f64); N],
    online: Option<[(usize, f64); 3]>,
}

fn interpolate(points: &[(usize, f64)], at: f64) -> f64 {
    for pair in points.windows(2) {
        if at <= pair[1].0 as f64 {
            let t = ((at.max(pair[0].0 as f64) / pair[0].0 as f64).ln()
                / (pair[1].0 as f64 / pair[0].0 as f64).ln())
            .clamp(0.0, 1.0);
            return pair[0].1 + t * (pair[1].1 - pair[0].1);
        }
    }
    points[points.len() - 1].1
}

impl<const N: usize> Curve<N> {
    fn at(&self, bytes: f64) -> f64 {
        match self.online {
            Some(points) if bytes <= points[2].0 as f64 => interpolate(&points, bytes),
            Some(points) => {
                points[2].1 * interpolate(&self.points, bytes)
                    / interpolate(&self.points, points[2].0 as f64)
            }
            None => interpolate(&self.points, bytes),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Row {
    a: f64,
    r: Curve<3>,
    e: Curve<12>,
}

// AMD EPYC 9V45, 2026-11-25, CI run 45550821471 (reused owned f64 replay).
// Rows come from that run's measured curves; its fit-acceptance flag was false.
// These fallback seeds do not establish validated phase predictions.
const CI_ROWS: [Row; 2] = [
    Row {
        a: 7.295822143356,
        r: Curve {
            online: None,
            points: [
                (65536, 1.960316249501),
                (4194304, 1.894785141212),
                (134217728, 0.8617609082357),
            ],
        },
        e: Curve {
            online: None,
            points: [
                (65536, 0.08914671031175),
                (131072, 0.08317755099417),
                (262144, 0.08159981907025),
                (524288, 0.06088424466089),
                (1048576, 0.1009711216944),
                (2097152, 0.1335119130773),
                (4194304, 0.1558782935818),
                (8388608, 0.1605964012099),
                (16777216, 0.2226114011787),
                (33554432, 0.3010221382176),
                (67108864, 0.4358547653704),
                (134217728, 0.3517858496209),
            ],
        },
    },
    Row {
        a: 7.295822143356,
        r: Curve {
            online: None,
            points: [
                (65536, 1.334474262286),
                (4194304, 1.230226508449),
                (134217728, 0.7420021698247),
            ],
        },
        e: Curve {
            online: None,
            points: [
                (65536, 0.2356881426186),
                (131072, 0.3327978436722),
                (262144, 0.2468692734975),
                (524288, 0.2339934921432),
                (1048576, 0.2468050807728),
                (2097152, 0.2466618085993),
                (4194304, 0.2645176847509),
                (8388608, 0.2622610504078),
                (16777216, 0.4279836838160),
                (33554432, 0.6088979324751),
                (67108864, 0.5667141949012),
                (134217728, 0.5945211305828),
            ],
        },
    },
];
const REFERENCES: [(HostClass, [Row; 2]); 1] = [(HostClass::X86_64Avx512, CI_ROWS)];

/// Structural table: unmeasured inactive rows use their same-stride real-map source.
pub(super) struct MovementConstants {
    rows: [Row; CLASSES],
    bandwidth: Vec<f64>,
    reserved_entries: usize,
    attempted: bool,
}

pub(super) type Constants = MovementConstants;

impl MovementConstants {
    fn reference(host: HostClass) -> Self {
        let rows = REFERENCES
            .iter()
            .find(|(class, _)| *class == host)
            .map_or(CI_ROWS, |(_, rows)| *rows);
        Self {
            rows: [rows[0], rows[1], rows[0], rows[1]],
            bandwidth: Vec::new(),
            reserved_entries: 0,
            attempted: false,
        }
    }

    fn row(&self, class: usize) -> Row {
        // There is one measured class per inner stride, so it is also the
        // slowest available entry for every unmeasured class of that stride.
        self.rows[class]
    }

    fn retained_bytes(&self) -> usize {
        // The slot already charges its inline value, including the rows.
        self.bandwidth.capacity().max(self.reserved_entries) * std::mem::size_of::<f64>()
    }

    fn bandwidth(&self, q: usize) -> f64 {
        self.bandwidth.get(q).copied().unwrap_or(f64::INFINITY)
    }
}

/// The retained value itself records the attempt, including a failed one:
/// untouched rows remain reference rows until this entry is evicted or trimmed.
pub(super) fn region<R>(
    engine: &mut dyn Engine,
    op: impl FnOnce(lanes::Exec<'_>, Option<&Constants>) -> R,
) -> Result<R, OperationError> {
    region_with(engine, calibrate, op)
}

fn region_with<R>(
    engine: &mut dyn Engine,
    measure: impl FnOnce(&mut lanes::Exec<'_>, Instant, &mut Constants),
    op: impl FnOnce(lanes::Exec<'_>, Option<&Constants>) -> R,
) -> Result<R, OperationError> {
    let budget = tenferro_cpu::with_cpu_engine(engine, |cpu| {
        (!cpu.is_broken()
            && cpu.num_threads() > 1
            && cpu.batch_policy().strategy == tenferro_cpu::CpuBatchStrategy::Auto)
            .then(|| cpu.num_threads())
    })
    .flatten();
    let Some(budget) = budget else {
        return crate::host_pool::lane_region(
            engine,
            crate::host_pool::HostPoolSite::Replay,
            |exec| op(exec, None),
        );
    };
    let mut retained = symtn_dense::engine_slot::checkout::<Constants>(engine);
    let deadline = retained.is_none().then(|| Instant::now() + ONLINE_BUDGET);
    if retained.is_none() {
        let mut candidate = Constants::reference(HostClass::detect());
        candidate.reserved_entries = budget + 1;
        symtn_dense::engine_slot::checkin(engine, candidate, Constants::retained_bytes);
        retained = symtn_dense::engine_slot::checkout::<Constants>(engine);
    }
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::host_pool::lane_region(
            engine,
            crate::host_pool::HostPoolSite::Replay,
            |mut exec| {
                if let Some(constants) = retained.as_mut().filter(|constants| !constants.attempted)
                {
                    constants.attempted = true;
                    constants.bandwidth = vec![f64::INFINITY; constants.reserved_entries];
                    measure(
                        &mut exec,
                        deadline.unwrap_or_else(|| Instant::now() + ONLINE_BUDGET),
                        constants,
                    );
                    interpolate_bandwidth(&mut constants.bandwidth);
                }
                op(exec, retained.as_ref())
            },
        )
    }));
    if let Some(constants) = retained {
        symtn_dense::engine_slot::checkin(engine, constants, Constants::retained_bytes);
    }
    match result {
        Ok(result) => result,
        Err(panic) => std::panic::resume_unwind(panic),
    }
}

/// The sole argmin, shared by fixed items and q-dependent executed ranges.
fn choose(
    p: usize,
    traffic: f64,
    phase: impl Fn(usize) -> (f64, f64),
    dispatch: impl Fn(usize) -> f64,
    bandwidth: impl Fn(usize) -> f64,
) -> usize {
    if p == 1 {
        return 1;
    }
    let (mut best, mut time) = (1, phase(1).0);
    for q in 2..=p {
        let (sum, largest) = phase(q);
        let candidate = dispatch(q) + largest.max(sum / q as f64).max(traffic / bandwidth(q));
        if candidate < time {
            best = q;
            time = candidate;
        }
    }
    best
}

fn member_cost(q: usize, n: usize, setup: f64, work: f64) -> (f64, f64) {
    (
        q as f64 * setup + n as f64 * work,
        setup + n.div_ceil(q) as f64 * work,
    )
}

pub(super) fn fill_cost(
    q: usize,
    elements: usize,
    columns: usize,
    a: f64,
    r: f64,
    e: f64,
) -> (f64, f64) {
    let leaves = q.min(columns).max(1);
    (
        leaves as f64 * a + leaves.min(elements) as f64 * r + elements as f64 * e,
        a + if elements == 0 {
            0.0
        } else {
            r + elements.div_ceil(leaves) as f64 * e
        },
    )
}

pub(super) type Costs = [(f64, f64, f64); CLASSES];

pub(super) fn components(counts: &[Counts; CLASSES], costs: &Costs) -> (f64, f64) {
    counts
        .iter()
        .zip(costs)
        .fold((0.0, 0.0), |(setup, work), (count, &(a, r, e))| {
            (
                setup + a * count.items as f64,
                work + r * count.runs as f64 + e * count.elements as f64,
            )
        })
}

pub(super) fn fixed(phase: &Summary, costs: &Costs) -> impl Fn(usize) -> (f64, f64) {
    let (setup, work) = components(&phase.counts, costs);
    let largest = phase
        .jobs
        .iter()
        .map(|job| {
            let (setup, work) = components(job, costs);
            setup + work
        })
        .fold(0.0, f64::max);
    move |_| (setup + work, largest)
}

/// Each group's pack columns have equal element counts, so each class hull
/// contains at most one point and a chunk aggregates these maxima in O(t).
pub(super) fn pack_maximum(phase: &Summary, costs: &Costs) -> f64 {
    phase
        .hulls
        .iter()
        .zip(costs)
        .filter(|(hull, _)| !hull.is_empty())
        .map(|(hull, &(a, r, e))| a + maximum(hull, r, e))
        .fold(0.0, f64::max)
}

pub(super) fn member_phase(
    phase: &Summary,
    costs: &Costs,
    members: usize,
) -> impl Fn(usize) -> (f64, f64) {
    let (setup, work) = components(&phase.counts, costs);
    move |q| member_cost(q, members, setup, work)
}

/// The phase supplies its executed-item costs; this chooser knows no phase kind.
pub(super) fn width<D: DenseRecouplingScalar, F: Fn(usize) -> (f64, f64)>(
    exec: &lanes::Exec<'_>,
    constants: Option<&Constants>,
    jobs: usize,
    working_set: u128,
    traffic: u128,
    prepare: impl FnOnce(&Costs) -> F,
) -> usize {
    let budget = exec.budget();
    if budget == 1 {
        return 1;
    }
    let jobs = jobs.max(1);
    if exec.batch_policy().strategy != tenferro_cpu::CpuBatchStrategy::Auto {
        let decision = exec.batch_decision(&lanes::BatchQuery::new(jobs, 0.0));
        return decision
            .lanes
            .saturating_mul(decision.item_threads)
            .clamp(1, budget.max(1))
            .min(jobs);
    }
    let reference;
    let constants = match constants {
        Some(constants) => constants,
        None => {
            reference = Constants::reference(HostClass::detect());
            &reference
        }
    };
    let bytes = std::mem::size_of::<D>() as f64;
    let working_set = working_set as f64 * bytes;
    let costs = core::array::from_fn(|c| {
        let row = constants.row(c);
        (row.a, row.r.at(working_set), row.e.at(working_set))
    });
    choose(
        budget.min(jobs),
        traffic as f64 * bytes,
        prepare(&costs),
        |q| {
            exec.token_ref()
                .dispatch_cost(NonZeroUsize::new(q).unwrap())
                .as_secs_f64()
                * 1e9
        },
        |q| constants.bandwidth(q),
    )
}

/// Fill every unmeasured width once, linearly between measured q knots and
/// flat outside them. Infinity remains only when the ladder has no measurement.
fn interpolate_bandwidth(values: &mut [f64]) {
    let Some(mut left) = values.iter().position(|x| x.is_finite() && *x > 0.0) else {
        return;
    };
    let first = values[left];
    values[..left].fill(first);
    for right in left + 1..values.len() {
        if !values[right].is_finite() || values[right] <= 0.0 {
            continue;
        }
        for q in left + 1..right {
            let t = (q - left) as f64 / (right - left) as f64;
            values[q] = values[left] + t * (values[right] - values[left]);
        }
        left = right;
    }
    let last = values[left];
    values[left + 1..].fill(last);
}

struct Fixture {
    transform: TreeTransformStructure<f64>,
    source_len: usize,
    destination_len: usize,
}

impl Fixture {
    fn new(elements: usize, runs: usize, inner: usize) -> Result<Self, OperationError> {
        use symtn_core::BlockSpec;
        let shape = vec![elements / runs, runs];
        let step = inner + 1;
        let dst = BlockStructure::from_blocks_with_rank(
            2,
            vec![BlockSpec::new(
                shape.clone(),
                vec![1, shape[0] + usize::from(runs > 1)],
                0,
            )?],
        )?;
        let src = BlockStructure::from_blocks_with_rank(
            2,
            vec![BlockSpec::new(
                shape.clone(),
                vec![step, step * shape[0] + usize::from(runs > 1)],
                0,
            )?],
        )?;
        let transform = TreeTransformStructure::compile_structures(
            &dst,
            &src,
            &[crate::TreeTransformBlockSpec::single(0, 0, -0.75)],
        )?;
        Ok(Self {
            transform,
            source_len: src.required_len()?,
            destination_len: dst.required_len()?,
        })
    }

    fn run(
        &self,
        exec: lanes::Exec<'_>,
        source: &[f64],
        destination: &mut [MaybeUninit<f64>],
    ) -> Result<(), OperationError> {
        let destination = destination
            .get_mut(..self.destination_len)
            .ok_or(OperationError::ElementCountOverflow)?;
        let source = source
            .get(..self.source_len)
            .ok_or(OperationError::ElementCountOverflow)?;
        let task = self.transform.task_view()?;
        let writer = OwnedWriter {
            task: ReplayParts::new(task),
            alpha: 1.0,
        };
        inline_direct(
            exec.into_token(),
            self.transform.parallel_schedule(),
            true,
            InlineSingles::Schedule,
            destination,
            source,
            &writer,
            &mut (),
            &mut Vec::new(),
            &mut [0; 2],
            &mut LeafGate::new(task.layouts().max_planned_elements()),
        )
    }
}

fn sample(
    exec: &mut lanes::Exec<'_>,
    deadline: Instant,
    elements: usize,
    runs: usize,
    inner: usize,
    source: &[f64],
    destination: &mut [MaybeUninit<f64>],
) -> Option<f64> {
    if Instant::now() >= deadline {
        return None;
    }
    let fixture = Fixture::new(elements, runs, inner).ok()?;
    let repetitions = if elements <= 16 {
        64
    } else if elements <= 4096 {
        4
    } else {
        1
    };
    let mut samples = [0.0; 3];
    for time in &mut samples {
        if Instant::now() >= deadline {
            return None;
        }
        let start = Instant::now();
        for _ in 0..repetitions {
            fixture
                .run(
                    exec.reborrow().limit(NonZeroUsize::MIN),
                    source,
                    destination,
                )
                .ok()?;
            std::hint::black_box(&*destination);
        }
        *time = start.elapsed().as_secs_f64() * 1e9 / repetitions as f64;
    }
    samples.sort_by(f64::total_cmp);
    (Instant::now() < deadline).then_some(samples[1])
}

fn measure_row(
    exec: &mut lanes::Exec<'_>,
    deadline: Instant,
    inner: usize,
    source: &[f64],
    destination: &mut [MaybeUninit<f64>],
) -> Option<Row> {
    let mut measure =
        |elements, runs| sample(exec, deadline, elements, runs, inner, source, destination);
    let empty = measure(0, 1)?;
    let low = measure(2, 1)?;
    let high = measure(16, 1)?;
    let mut one = [0.0; 3];
    let mut run = [(0, 0.0); 3];
    let working_sets = [ONLINE_WORKING_SETS, [1024, 8192, 65536]][inner];
    for (i, w) in working_sets.into_iter().enumerate() {
        let n = w / (2 * std::mem::size_of::<f64>());
        one[i] = measure(n, 1)?;
        let many_runs = (n / 2).min(128);
        let many = measure(n, many_runs)?;
        run[i] = (w, (many - one[i]) / (many_runs - 1) as f64);
    }
    let a = low - empty - (high - low) * (2.0 / 14.0) - run[0].1;
    let element = core::array::from_fn(|i| {
        let w = working_sets[i];
        (w, (one[i] - empty - a - run[i].1) / (w / 16) as f64)
    });
    if !std::iter::once(a)
        .chain(run.iter().map(|p| p.1))
        .chain(element.iter().map(|p| p.1))
        .all(|value| value.is_finite() && value >= 0.0)
    {
        return None;
    }
    Some(Row {
        a,
        r: Curve {
            points: CI_ROWS[inner].r.points,
            online: Some(run),
        },
        e: Curve {
            points: CI_ROWS[inner].e.points,
            online: Some(element),
        },
    })
}

fn calibrate(exec: &mut lanes::Exec<'_>, deadline: Instant, constants: &mut Constants) {
    if Instant::now() >= deadline {
        return;
    }
    let source = vec![1.0; BUFFER_ELEMENTS];
    let mut destination = Vec::<f64>::with_capacity(BUFFER_ELEMENTS);
    for inner in 0..2 {
        if let Some(row) = measure_row(
            exec,
            deadline,
            inner,
            &source,
            destination.spare_capacity_mut(),
        ) {
            constants.rows[inner] = row;
            constants.rows[inner + 2] = row;
        }
    }
    let p = exec.budget();
    if Instant::now() >= deadline {
        return;
    }
    for q in std::iter::successors(Some(2.min(p)), |&q| (q < p).then(|| (q * 2).min(p))) {
        if Instant::now() >= deadline {
            return;
        }
        let table = exec
            .token_ref()
            .dispatch_cost(NonZeroUsize::new(q).unwrap())
            .as_secs_f64()
            * 1e9;
        let n = (BUFFER_ELEMENTS / 2 / q).max(1);
        let Some(fixture) = Fixture::new(n, 1, 0).ok() else {
            continue;
        };
        let Some(destination) = destination.spare_capacity_mut().get_mut(..n * q) else {
            return;
        };
        let pieces: Vec<_> = destination.chunks_mut(n).map(Mutex::new).collect();
        let failed = AtomicBool::new(false);
        let (workspace, policy) = (exec.workspace(), *exec.batch_policy());
        if Instant::now() >= deadline {
            return;
        }
        let start = Instant::now();
        exec.reborrow().into_token().fork(q, |i, token| {
            let mut piece = pieces[i].lock().unwrap_or_else(PoisonError::into_inner);
            if fixture
                .run(
                    lanes::Exec::from_token(token, workspace, policy),
                    &source[i * n..(i + 1) * n],
                    &mut piece,
                )
                .is_err()
            {
                failed.store(true, Ordering::Relaxed);
            }
        });
        let elapsed = start.elapsed().as_secs_f64() * 1e9 - table;
        if Instant::now() < deadline
            && !failed.load(Ordering::Relaxed)
            && elapsed.is_finite()
            && elapsed > 0.0
        {
            constants.bandwidth[q] = (2 * n * q * std::mem::size_of::<f64>()) as f64 / elapsed;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    fn fixed(p: usize, sum: f64, largest: f64, dispatch: f64) -> usize {
        choose(
            p,
            0.0,
            |_| (sum, largest),
            |q| dispatch * q as f64,
            |_| f64::INFINITY,
        )
    }

    /// Purpose: distributed work pays for a split, while costly dispatch prevents it.
    #[test]
    fn dominant_item_with_small_items_can_split() {
        assert!(fixed(8, 1000.0 + 99.0 * 10.0, 1000.0, 10.0) > 1);
        assert_eq!(fixed(8, 1000.0 + 99.0 * 10.0, 1000.0, 1000.0), 1);
    }

    /// Purpose: many small items choose an interior width, unlike an inline-dominated phase.
    #[test]
    fn small_items_choose_an_interior_width() {
        assert_eq!(fixed(8, 3000.0, 30.0, 100.0), 5);
        assert_eq!(fixed(8, 30.0, 30.0, 100.0), 1);
    }

    /// Purpose: one indivisible item stays inline; splitting becomes useful with replication.
    #[test]
    fn one_large_item_stays_inline() {
        assert_eq!(fixed(8, 1000.0, 1000.0, 0.0), 1);
        assert!(fixed(8, 2000.0, 1000.0, 0.0) > 1);
    }

    /// Purpose: strict improvement preserves the smaller of tied integer widths.
    #[test]
    fn ties_choose_the_smaller_width() {
        assert_eq!(
            choose(
                3,
                0.0,
                |_| (12.0, 0.0),
                |q| if q == 2 { 0.0 } else { 2.0 },
                |_| f64::INFINITY
            ),
            2
        );
        assert_eq!(
            choose(
                3,
                0.0,
                |_| (12.0, 0.0),
                |q| if q == 2 { 1.0 } else { 2.0 },
                |_| f64::INFINITY
            ),
            3
        );
    }

    /// Purpose: width one never evaluates model data.
    #[test]
    fn serial_choice_reads_no_model_data() {
        assert_eq!(
            choose(
                1,
                0.0,
                |_| panic!("phase"),
                |_| panic!("dispatch"),
                |_| panic!("bandwidth")
            ),
            1
        );
        assert_eq!(fixed(2, 12.0, 1.0, 0.0), 2);
    }

    /// Purpose: the integer argmin agrees with an exact common-denominator oracle.
    #[test]
    fn small_ladders_match_an_exhaustive_integer_oracle() {
        for p in 2..=6 {
            let denominator: u64 = (1..=p as u64 + 1).product();
            for sum in 0..=12u64 {
                for largest in 0..=sum {
                    for traffic in [0, 24] {
                        let expected = (1..=p)
                            .min_by_key(|&q| {
                                if q == 1 {
                                    sum * denominator
                                } else {
                                    let dispatch = (q * 3 % 5) as u64;
                                    dispatch * denominator
                                        + (largest * denominator)
                                            .max(sum * denominator / q as u64)
                                            .max(traffic * denominator / (q + 1) as u64)
                                }
                            })
                            .unwrap();
                        assert_eq!(
                            choose(
                                p,
                                traffic as f64,
                                |_| (sum as f64, largest as f64),
                                |q| (q * 3 % 5) as f64,
                                |q| (q + 1) as f64
                            ),
                            expected
                        );
                    }
                }
            }
        }
    }

    /// Purpose: executed member ranges each pay setup and preserve the per-member work.
    #[test]
    fn member_ranges_price_setup_at_the_executed_width() {
        assert_eq!(member_cost(1, 4, 1.0, 1.0 + 2.0), (13.0, 13.0));
        assert_eq!(member_cost(2, 4, 1.0, 1.0 + 2.0), (14.0, 7.0));
        assert_eq!(member_cost(2, 4, 0.0, 1.0 + 2.0), (12.0, 6.0));
    }

    /// Purpose: the shared chooser handles downward ceiling steps without convexity assumptions.
    #[test]
    fn member_ceiling_steps_share_the_integer_argmin() {
        assert_eq!(
            choose(
                10,
                0.0,
                |q| member_cost(q, 10, 1.0, 1.0),
                |q| 0.1 * q as f64,
                |_| f64::INFINITY
            ),
            10
        );
        assert_eq!(
            choose(
                10,
                0.0,
                |q| member_cost(q, 10, 1.0, 1.0),
                |q| 10.0 * q as f64,
                |_| f64::INFINITY
            ),
            1
        );
    }

    /// Purpose: even empty executed cold-fill leaves pay setup, but only nonempty leaves pay a run.
    #[test]
    fn cold_fill_prices_the_executed_nonempty_runs() {
        assert_eq!(fill_cost(4, 1, 4, 1.0, 1.0, 1.0), (6.0, 3.0));
        assert_eq!(fill_cost(1, 1, 4, 1.0, 1.0, 1.0), (3.0, 3.0));
        assert_eq!(fill_cost(4, 4, 4, 1.0, 1.0, 1.0), (12.0, 3.0));
    }

    /// Purpose: an ownership job sums all its roles and setups across classes;
    /// changing the coefficients can change which whole job is the largest.
    #[test]
    fn mixed_class_jobs_keep_their_joint_maximum_and_setups() {
        use crate::movement::Point;
        let mut phase = Summary::default();
        phase
            .push(
                0,
                Point {
                    runs: 2,
                    elements: 3,
                },
            )
            .unwrap();
        phase
            .push(
                1,
                Point {
                    runs: 1,
                    elements: 4,
                },
            )
            .unwrap();
        phase
            .push(
                0,
                Point {
                    runs: 1,
                    elements: 1,
                },
            )
            .unwrap();
        phase
            .push(
                0,
                Point {
                    runs: 8,
                    elements: 1,
                },
            )
            .unwrap();
        let costs = [
            (2.0, 3.0, 5.0),
            (7.0, 11.0, 13.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ];
        assert_eq!(super::fixed(&phase, &costs)(1), (134.0, 70.0));
        phase.fuse_jobs(&[0, 3]).unwrap();
        assert_eq!(super::fixed(&phase, &costs)(1), (134.0, 103.0));
        let mut costs = costs;
        costs[0] = (0.0, 100.0, 0.0);
        assert_eq!(super::fixed(&phase, &costs)(1), (1170.0, 800.0));
    }

    /// Purpose: widths between measured knots retain the bandwidth roof;
    /// an entirely unmeasured ladder is the only infinity fallback.
    #[test]
    fn bandwidth_interpolates_the_measured_ladder() {
        let mut values = [f64::INFINITY; 9];
        for (q, value) in [(2, 10.0), (4, 30.0), (8, 70.0)] {
            values[q] = value;
        }
        interpolate_bandwidth(&mut values);
        assert_eq!(
            values,
            [10.0, 10.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
        );
        let mut empty = [f64::INFINITY; 3];
        interpolate_bandwidth(&mut empty);
        assert_eq!(empty, [f64::INFINITY; 3]);
    }

    /// Purpose: online run and element curves keep measured knots and follow
    /// the reference shape, scaled at the last measured knot, beyond them.
    #[test]
    fn online_curve_tails_preserve_reference_shape() {
        let close = |actual: f64, expected: f64| {
            assert!((actual - expected).abs() <= 32.0 * f64::EPSILON * expected.abs().max(1.0));
        };
        let mut run = CI_ROWS[1].r;
        run.online = Some([(1024, 1.0), (8192, 2.0), (65536, 4.0)]);
        close(run.at(8192.0), 2.0);
        close(run.at(65536.0), 4.0);
        close(run.at(4194304.0), 4.0 * 1.230226508449 / 1.334474262286);
        assert_ne!(run.at(4194304.0), 4.0);
        let mut element = CI_ROWS[0].e;
        element.online = Some([(65536, 0.5), (1048576, 1.0), (4194304, 2.0)]);
        close(element.at(1048576.0), 1.0);
        close(element.at(4194304.0), 2.0);
        close(
            element.at(134217728.0),
            2.0 * 0.3517858496209 / 0.1558782935818,
        );
        assert_ne!(element.at(134217728.0), 2.0);
    }

    /// Purpose: forced policies use the engine's batch decision without
    /// reading model data, and the executed-job cap still applies.
    #[test]
    fn forced_width_delegates_to_the_engine_and_caps_jobs() {
        use tenferro_cpu::{CpuBatchPolicy, CpuBatchStrategy};
        for strategy in [
            CpuBatchStrategy::Sequential,
            CpuBatchStrategy::Inner,
            CpuBatchStrategy::Outer,
            CpuBatchStrategy::Split {
                lanes: NonZeroUsize::new(3).unwrap(),
            },
        ] {
            let mut engine = tenferro_cpu::CpuEngine::new(
                tenferro_cpu::Pool::builder().threads(8).build().unwrap(),
            )
            .with_batch_policy(CpuBatchPolicy::new(strategy));
            region_with(
                &mut engine,
                |_, _, _| panic!("forced calibration"),
                |exec, constants| {
                    assert!(constants.is_none());
                    for jobs in [1, 3, 12] {
                        let decision = exec.batch_decision(&lanes::BatchQuery::new(jobs, 0.0));
                        let expected = decision
                            .lanes
                            .saturating_mul(decision.item_threads)
                            .clamp(1, exec.budget())
                            .min(jobs);
                        assert_eq!(
                            width::<f64, _>(
                                &exec,
                                constants,
                                jobs,
                                0,
                                0,
                                |_| -> fn(usize) -> (f64, f64) { panic!("forced model") }
                            ),
                            expected
                        );
                    }
                },
            )
            .unwrap();
        }
    }

    /// Purpose: admission reserves ladder bytes without allocating, and never undercharges actual capacity.
    #[test]
    fn ladder_reservation_is_charged_before_allocation() {
        let mut constants = Constants::reference(HostClass::Generic);
        constants.reserved_entries = 4;
        assert_eq!(constants.bandwidth.capacity(), 0);
        assert_eq!(
            constants.retained_bytes(),
            constants.reserved_entries * std::mem::size_of::<f64>()
        );
        constants.bandwidth = Vec::with_capacity(3 * constants.reserved_entries);
        assert_eq!(
            constants.retained_bytes(),
            constants.bandwidth.capacity() * std::mem::size_of::<f64>()
        );
    }

    /// Purpose: a failed retained attempt uses references once, and trimming enables a new attempt.
    #[test]
    fn failed_calibration_is_retained_until_trim() {
        let mut engine =
            tenferro_cpu::CpuEngine::new(tenferro_cpu::Pool::builder().threads(2).build().unwrap());
        let calls = Cell::new(0);
        for _ in 0..2 {
            region_with(
                &mut engine,
                |_, _, _| calls.set(calls.get() + 1),
                |_, constants| {
                    assert_eq!(
                        constants.unwrap().rows,
                        [CI_ROWS[0], CI_ROWS[1], CI_ROWS[0], CI_ROWS[1]]
                    );
                },
            )
            .unwrap();
        }
        assert_eq!(calls.get(), 1);
        symtn_dense::engine_slot::clear(&mut engine);
        region_with(
            &mut engine,
            |_, _, _| calls.set(calls.get() + 1),
            |_, constants| assert!(constants.is_some()),
        )
        .unwrap();
        assert_eq!(calls.get(), 2);
    }

    /// Purpose: accepted online rows outrank references and remain installed on a warm entry.
    #[test]
    fn accepted_online_row_is_reused() {
        let mut engine =
            tenferro_cpu::CpuEngine::new(tenferro_cpu::Pool::builder().threads(2).build().unwrap());
        let calls = Cell::new(0);
        for _ in 0..2 {
            region_with(
                &mut engine,
                |_, _, constants| {
                    calls.set(calls.get() + 1);
                    constants.rows[0].a += 1.0;
                },
                |_, constants| {
                    let rows = constants.unwrap().rows;
                    assert_eq!(rows[0].a, CI_ROWS[0].a + 1.0);
                    assert_ne!(rows[0].a, CI_ROWS[0].a);
                    assert_eq!(rows[1], CI_ROWS[1]);
                },
            )
            .unwrap();
        }
        assert_eq!(calls.get(), 1);
    }

    /// Purpose: serial engines skip calibration and retained state entirely.
    #[test]
    fn serial_engine_keeps_no_calibration_entry() {
        let mut engine = tenferro_cpu::CpuEngine::serial();
        region_with(
            &mut engine,
            |_, _, _| panic!("serial calibration"),
            |_, constants| assert!(constants.is_none()),
        )
        .unwrap();
        assert_eq!(symtn_dense::engine_slot::retained_bytes(&mut engine), 0);
    }
}
