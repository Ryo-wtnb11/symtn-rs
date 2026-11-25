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

    fn memory_knee(&self) -> f64 {
        let knots = || self.points.iter().chain(self.online.iter().flatten());
        let minimum = knots()
            .min_by(|a, b| self.at(a.0 as f64).total_cmp(&self.at(b.0 as f64)))
            .map_or(0, |point| point.0);
        let plateau = self.at(f64::INFINITY);
        knots()
            .filter(|&&(bytes, _)| bytes >= minimum && self.at(bytes as f64) >= plateau)
            .map(|point| point.0)
            .min()
            .map_or(f64::INFINITY, |bytes| bytes as f64)
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
    memory_knees: [f64; CLASSES],
    bandwidth: Vec<f64>,
    dispatch: Vec<Option<Duration>>,
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
            memory_knees: core::array::from_fn(|c| rows[c % 2].e.memory_knee()),
            bandwidth: Vec::new(),
            dispatch: Vec::new(),
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
            + self.dispatch.capacity().max(self.reserved_entries)
                * std::mem::size_of::<Option<Duration>>()
    }

    fn dispatch_cost(&self, q: usize, table: impl FnOnce() -> Duration) -> f64 {
        self.dispatch
            .get(q)
            .copied()
            .flatten()
            .unwrap_or_else(table)
            .as_secs_f64()
            * 1e9
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
    let mut op = Some(op);
    let mut serial = None;
    let budget = tenferro_cpu::with_cpu_engine(engine, |cpu| {
        if cpu.num_threads() == 1 {
            serial = Some(crate::host_pool::enter_cpu(
                cpu,
                crate::host_pool::HostPoolSite::Replay,
                true,
                |exec| op.take().expect("region body runs once")(exec, None),
            ));
            return None;
        }
        (!cpu.is_broken()
            && cpu.num_threads() > 1
            && cpu.batch_policy().strategy == tenferro_cpu::CpuBatchStrategy::Auto)
            .then(|| cpu.num_threads())
    })
    .flatten();
    if let Some(result) = serial {
        return result;
    }
    let Some(budget) = budget else {
        return crate::host_pool::lane_region(
            engine,
            crate::host_pool::HostPoolSite::Replay,
            |exec| op.take().expect("region body runs once")(exec, None),
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
                    constants.dispatch = vec![None; constants.reserved_entries];
                    measure(
                        &mut exec,
                        deadline.unwrap_or_else(|| Instant::now() + ONLINE_BUDGET),
                        constants,
                    );
                    constants.memory_knees =
                        core::array::from_fn(|c| constants.rows[c].e.memory_knee());
                    validate_bandwidth(
                        &mut constants.bandwidth,
                        (2 * std::mem::size_of::<f64>()) as f64
                            / constants.rows[0].e.at(f64::INFINITY),
                    );
                    interpolate_bandwidth(&mut constants.bandwidth);
                }
                op.take().expect("region body runs once")(exec, retained.as_ref())
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
    // Fused ownership jobs can mix strides: summing independent class maxima
    // bounds M above, so the conservative estimate may select fewer leaves.
    let largest = phase.maxima.as_ref().map_or(0.0, |maxima| {
        maxima
            .hulls
            .iter()
            .zip(&maxima.setups)
            .zip(costs)
            .map(|((hull, &k), &(a, r, e))| a * k + maximum(hull, r, e))
            .sum()
    });
    move |_| (setup + work, largest)
}

/// The executor submits one GEMM batch per consecutive chunk of q groups.
/// Summed chunk maxima bound its serial batch barriers' critical path below;
/// they are not a prediction of the provider's actual placement or runtime.
pub(super) fn group_phase<'a>(
    schedule: &'a TreeTransformParallelSchedule,
    costs: &Costs,
    gemm: impl Fn(usize) -> f64 + 'a,
) -> impl Fn(usize) -> (f64, f64) + 'a {
    let costs = *costs;
    move |q| {
        let (mut sum, mut critical) = (0.0, 0.0);
        for first in (0..schedule.pack_movement.len()).step_by(q) {
            let mut largest = 0.0_f64;
            for group in first..(first + q).min(schedule.pack_movement.len()) {
                let movement = [
                    &schedule.pack_movement[group],
                    &schedule.scatter_movement[group],
                ]
                .into_iter()
                .map(|summary| {
                    let (setup, work) = components(&summary.counts, &costs);
                    setup + work
                })
                .sum::<f64>();
                let item = movement + gemm(group);
                sum += item;
                largest = largest.max(item);
            }
            critical += largest;
        }
        (sum, critical)
    }
}

/// Each group's pack columns have equal element counts, so each class hull
/// contains at most one point and a chunk aggregates these maxima in O(t).
pub(super) fn pack_maximum(phase: &Summary, costs: &Costs) -> f64 {
    phase.maxima.as_ref().map_or(0.0, |maxima| {
        maxima
            .hulls
            .iter()
            .zip(costs)
            .filter(|(hull, _)| !hull.is_empty())
            .map(|(hull, &(a, r, e))| a + maximum(hull, r, e))
            .fold(0.0, f64::max)
    })
}

pub(super) fn pack_phase(
    summary: &Summary,
    groups: &[Summary],
    costs: &Costs,
    columns: usize,
    fill: Option<usize>,
) -> impl Fn(usize) -> (f64, f64) {
    let (setup, work) = components(&summary.counts, costs);
    let largest = groups
        .iter()
        .map(|phase| pack_maximum(phase, costs))
        .fold(0.0, f64::max);
    let (a, r, e) = costs[2];
    move |q| {
        let (fill_sum, fill_max) = fill.map_or((0.0, 0.0), |elements| {
            fill_cost(q, elements, columns, a, r, e)
        });
        (setup + work + fill_sum, largest + fill_max)
    }
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
    elements: impl FnOnce() -> [f64; CLASSES],
    prepare: impl Fn(&Costs) -> F,
) -> usize {
    let budget = exec.budget();
    if budget.min(jobs) <= 1 {
        return 1;
    }
    let jobs = jobs.max(1);
    if let Some(width) = lanes::forced_width(exec, jobs) {
        return width;
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
    let traffic = traffic as f64 * bytes;
    choose(
        budget.min(jobs),
        traffic,
        prepare(&costs),
        |q| {
            constants.dispatch_cost(q, || {
                exec.token_ref()
                    .dispatch_cost(NonZeroUsize::new(q).unwrap())
            })
        },
        phase_bandwidth(working_set, traffic, elements(), constants),
    )
}

fn phase_bandwidth<'a>(
    working_set: f64,
    traffic: f64,
    elements: [f64; CLASSES],
    constants: &'a Constants,
) -> impl Fn(usize) -> f64 + 'a {
    let rows = &constants.rows;
    let measured = &constants.bandwidth;
    let element_time = |bytes| {
        elements
            .iter()
            .zip(rows)
            .map(|(n, row)| n * row.e.at(bytes))
            .sum::<f64>()
    };
    let rate = |time: f64| {
        if time > 0.0 {
            traffic / time
        } else {
            f64::INFINITY
        }
    };
    let single = rate(element_time(working_set));
    let memory = rate(element_time(f64::INFINITY));
    // A mixed phase uses the memory roof only after every active curve reaches its plateau.
    let knee = elements
        .iter()
        .zip(constants.memory_knees)
        .filter(|(n, _)| **n > 0.0)
        .map(|(_, knee)| knee)
        .fold(0.0, f64::max);
    let first = measured
        .iter()
        .copied()
        .find(|value| value.is_finite() && *value >= memory.max(single))
        .unwrap_or(f64::INFINITY);
    move |q| {
        let saturated = if working_set >= knee {
            measured.get(q).copied().unwrap_or(f64::INFINITY).max(first)
        } else {
            f64::INFINITY
        };
        (q as f64 * single).min(saturated)
    }
}

/// A slower-than-serial or descending sample is not a saturated bandwidth estimate.
fn validate_bandwidth(values: &mut [f64], minimum: f64) {
    let mut previous = minimum;
    for value in values {
        if value.is_finite() && *value > 0.0 && *value >= previous {
            previous = *value;
        } else {
            *value = f64::INFINITY;
        }
    }
}

/// Fill missing widths between accepted q knots, flat outside them.
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

fn bandwidth_fixture_bytes(remaining_ns: f64, dispatch_ns: f64, ns_per_byte: f64) -> usize {
    if ns_per_byte > 0.0 && ns_per_byte.is_finite() {
        ((remaining_ns - dispatch_ns).max(0.0) / ns_per_byte) as usize
    } else {
        0
    }
}

fn validated_dispatch(measured: Duration, table: Duration) -> Option<Duration> {
    (measured <= table.saturating_mul(20)).then_some(measured)
}

fn measure_dispatch(exec: &mut lanes::Exec<'_>, deadline: Instant, constants: &mut Constants) {
    // ponytail: replace this interim measurement with the threadteam calibration hook;
    // delete it when threadteam can calibrate a running pool.
    for q in 2..=exec.budget() {
        if Instant::now() >= deadline {
            return;
        }
        let width = NonZeroUsize::new(q).unwrap();
        let table = exec.token_ref().dispatch_cost(width);
        exec.reborrow().limit(width).into_token().fork(q, |_, _| {});
        let mut samples = [Duration::ZERO; 5];
        for sample in &mut samples {
            if Instant::now() >= deadline {
                return;
            }
            let start = Instant::now();
            for _ in 0..8 {
                exec.reborrow().limit(width).into_token().fork(q, |_, _| {});
            }
            *sample = start.elapsed() / 8;
        }
        if Instant::now() >= deadline {
            return;
        }
        samples.sort_unstable();
        constants.dispatch[q] = validated_dispatch(samples[2], table);
    }
}

fn calibrate(exec: &mut lanes::Exec<'_>, deadline: Instant, constants: &mut Constants) {
    measure_dispatch(exec, deadline, constants);
    if Instant::now() >= deadline {
        return;
    }
    let setup_start = Instant::now();
    let source = vec![1.0; BUFFER_ELEMENTS];
    let mut destination = Vec::<f64>::with_capacity(BUFFER_ELEMENTS);
    let setup_ns_per_byte = setup_start.elapsed().as_secs_f64() * 1e9
        / (BUFFER_ELEMENTS * std::mem::size_of::<f64>()) as f64;
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
    let teardown_start = Instant::now();
    drop(source);
    drop(destination);
    let setup_ns_per_byte = setup_ns_per_byte
        + teardown_start.elapsed().as_secs_f64() * 1e9
            / (BUFFER_ELEMENTS * std::mem::size_of::<f64>()) as f64;
    let ladder = || std::iter::successors(Some(2.min(p)), |&q| (q < p).then(|| (q * 2).min(p)));
    let dispatch_ns = |q| {
        constants.dispatch_cost(q, || {
            exec.token_ref()
                .dispatch_cost(NonZeroUsize::new(q).unwrap())
        })
    };
    let memory_bytes = constants.rows[0].e.memory_knee() as usize;
    let ns_per_byte = setup_ns_per_byte
        + ladder().count() as f64 * constants.rows[0].e.at(memory_bytes as f64)
            / (2 * std::mem::size_of::<f64>()) as f64;
    let remaining_ns = deadline
        .saturating_duration_since(Instant::now())
        .as_secs_f64()
        * 1e9;
    let maximum_bytes =
        bandwidth_fixture_bytes(remaining_ns, ladder().map(dispatch_ns).sum(), ns_per_byte);
    let elements = memory_bytes.div_ceil(2 * std::mem::size_of::<f64>());
    // A smaller fixture would measure cache bandwidth, so leave B_sat unknown
    // when first-touch and the whole saturated ladder cannot fit the deadline.
    if (elements + p).saturating_mul(2 * std::mem::size_of::<f64>()) > maximum_bytes {
        return;
    }
    let source = vec![1.0; elements + p];
    let mut destination = Vec::<f64>::with_capacity(elements + p);
    for q in ladder() {
        let dispatch = constants.dispatch_cost(q, || {
            exec.token_ref()
                .dispatch_cost(NonZeroUsize::new(q).unwrap())
        });
        let remaining_ns = deadline
            .saturating_duration_since(Instant::now())
            .as_secs_f64()
            * 1e9;
        if bandwidth_fixture_bytes(remaining_ns, dispatch, ns_per_byte) < memory_bytes {
            return;
        }
        let n = elements.div_ceil(q).max(1);
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
        exec.reborrow()
            .limit(NonZeroUsize::new(q).unwrap())
            .into_token()
            .fork(q, |i, token| {
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
        let elapsed = start.elapsed().as_secs_f64() * 1e9 - dispatch;
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

    /// Purpose: accepted dispatch measurements drive the width and skip table reads;
    /// absent and rejected measurements are the table-route controls.
    #[test]
    fn movement_dispatch_uses_measurement_or_table() {
        let table = Duration::from_nanos(600);
        let mut engine = tenferro_cpu::CpuEngine::new(
            tenferro_cpu::Pool::builder()
                .threads(2)
                .dispatch_costs(tenferro_cpu::threadteam::DispatchCosts::new(
                    table, table, table,
                ))
                .build()
                .unwrap(),
        );
        for (sample, expected, table_route, expected_width) in [
            (Some(0), 0.0, false, 2),
            (Some(110), 110.0, false, 2),
            (Some(12_000), 12_000.0, false, 1),
            (Some(12_001), 600.0, true, 1),
            (None, 600.0, true, 1),
        ] {
            let mut constants = Constants::reference(HostClass::Generic);
            constants.dispatch = vec![None; 3];
            constants.dispatch[2] = sample
                .map(Duration::from_nanos)
                .and_then(|measured| validated_dispatch(measured, table));
            let reads = Cell::new(0);
            assert_eq!(
                constants.dispatch_cost(2, || {
                    reads.set(reads.get() + 1);
                    table
                }),
                expected
            );
            assert_eq!(reads.get(), usize::from(table_route));
            crate::host_pool::lane_region(
                &mut engine,
                crate::host_pool::HostPoolSite::Replay,
                |exec| {
                    assert_eq!(
                        width::<f64, _>(
                            &exec,
                            Some(&constants),
                            2,
                            0,
                            0,
                            || [0.0; CLASSES],
                            |_| |_| (400.0, 200.0),
                        ),
                        expected_width
                    );
                },
            )
            .unwrap();
        }
        assert_eq!(
            validated_dispatch(Duration::ZERO, Duration::ZERO),
            Some(Duration::ZERO)
        );
        assert_eq!(
            validated_dispatch(Duration::from_nanos(1), Duration::ZERO),
            None
        );
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

    fn compiled_jobs(jobs: &[[Counts; CLASSES]]) -> Summary {
        let mut phase = Summary::default();
        for job in jobs {
            for (count, &other) in phase.counts.iter_mut().zip(job) {
                *count = count.add(other).unwrap();
            }
            phase.record_job(job).unwrap();
        }
        phase.finish().unwrap();
        phase
    }

    fn linear_job_costs(jobs: &[[Counts; CLASSES]], costs: &Costs) -> (f64, f64) {
        jobs.iter()
            .map(|job| {
                job.iter()
                    .zip(costs)
                    .map(|(count, &(a, r, e))| {
                        a * count.items as f64 + r * count.runs as f64 + e * count.elements as f64
                    })
                    .sum::<f64>()
            })
            .fold((0.0, 0.0_f64), |(sum, largest), job| {
                (sum + job, largest.max(job))
            })
    }

    /// Purpose: independent class maxima bound mixed executed jobs above while
    /// exact compiled totals preserve S, including zero-cost coefficients.
    #[test]
    fn compiled_bound_covers_random_mixed_jobs() {
        let mut seed = 17u64;
        for len in 1..30 {
            let jobs = (0..len)
                .map(|_| {
                    core::array::from_fn(|_| {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                        let items = (seed >> 32) % 4;
                        Counts {
                            items,
                            runs: items * (1 + seed % 19),
                            elements: items * (1 + seed % 31),
                        }
                    })
                })
                .collect::<Vec<_>>();
            let phase = compiled_jobs(&jobs);
            for rate in 0..20 {
                let costs = core::array::from_fn(|c| {
                    (
                        (rate % 5) as f64,
                        ((rate + c) % 7) as f64,
                        ((rate + 2 * c) % 11) as f64,
                    )
                });
                let exact = linear_job_costs(&jobs, &costs);
                let (sum, bound) = super::fixed(&phase, &costs)(1);
                assert_eq!(sum, exact.0);
                assert!(bound >= exact.1);
            }
        }
    }

    /// Purpose: a single class with a fixed setup count has an exact maximum;
    /// varying setup counts are the control requiring a conservative bound.
    #[test]
    fn single_class_fixed_setups_have_an_exact_maximum() {
        for items in [1, 2] {
            let jobs = [
                [
                    Counts {
                        items,
                        runs: 4,
                        elements: 8,
                    },
                    Counts::default(),
                    Counts::default(),
                    Counts::default(),
                ],
                [
                    Counts {
                        items,
                        runs: 1,
                        elements: 100,
                    },
                    Counts::default(),
                    Counts::default(),
                    Counts::default(),
                ],
            ];
            for costs in [[(100.0, 0.0, 1.0); CLASSES], [(1.0, 100.0, 0.0); CLASSES]] {
                assert_eq!(
                    super::fixed(&compiled_jobs(&jobs), &costs)(1),
                    linear_job_costs(&jobs, &costs)
                );
            }
        }
        let jobs = [
            [
                Counts {
                    items: 2,
                    runs: 4,
                    elements: 8,
                },
                Counts::default(),
                Counts::default(),
                Counts::default(),
            ],
            [
                Counts {
                    items: 1,
                    runs: 1,
                    elements: 100,
                },
                Counts::default(),
                Counts::default(),
                Counts::default(),
            ],
        ];
        let costs = [(100.0, 0.0, 1.0); CLASSES];
        assert_eq!(linear_job_costs(&jobs, &costs), (408.0, 208.0));
        assert_eq!(
            super::fixed(&compiled_jobs(&jobs), &costs)(1),
            (408.0, 300.0)
        );
    }

    /// Purpose: a pack leaf pays for its column work and cold-fill share
    /// together; warm scratch is the control with no fill price.
    #[test]
    fn pack_columns_and_cold_fill_share_add_their_prices() {
        let column = [
            Counts {
                items: 1,
                runs: 1,
                elements: 3,
            },
            Counts::default(),
            Counts::default(),
            Counts::default(),
        ];
        let groups = [compiled_jobs(&[column; 4])];
        let summary = Summary::totals(&groups).unwrap();
        let costs = [(1.0, 1.0, 1.0); CLASSES];
        assert_eq!(
            pack_phase(&summary, &groups, &costs, 4, Some(8))(2),
            (32.0, 11.0)
        );
        assert_eq!(
            pack_phase(&summary, &groups, &costs, 4, None)(2),
            (20.0, 5.0)
        );
    }

    /// Purpose: dense-dominated groups choose their chunk width from the combined
    /// cost and serial batch barriers; movement alone is the inline control.
    #[test]
    fn group_chunks_price_gemm_and_serial_batch_barriers() {
        use crate::TreeTransformBlockSpec;
        let blocks = [100, 1, 100, 1]
            .into_iter()
            .flat_map(|rows| [vec![rows], vec![rows]])
            .collect::<Vec<_>>();
        let structure = BlockStructure::packed_column_major(1, blocks).unwrap();
        let transform = TreeTransformStructure::compile_structures(
            &structure,
            &structure,
            &(0..4)
                .map(|group| {
                    TreeTransformBlockSpec::multi(
                        vec![2 * group, 2 * group + 1],
                        vec![2 * group, 2 * group + 1],
                        vec![1.0, 0.0, 0.0, 1.0],
                    )
                })
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let schedule = transform.parallel_schedule();
        let costs = [(0.5, 0.0, 0.0); CLASSES];
        let dense = [100.0, 1.0, 100.0, 1.0];
        let phase = group_phase(schedule, &costs, |group| dense[group]);
        assert_eq!(
            [phase(1), phase(2), phase(3), phase(4)],
            [
                (210.0, 210.0),
                (210.0, 204.0),
                (210.0, 105.0),
                (210.0, 102.0)
            ]
        );
        let dispatch = |q| 10.0 * q as f64;
        assert_eq!(choose(4, 0.0, phase, dispatch, |_| f64::INFINITY), 3);
        assert_eq!(
            choose(
                4,
                0.0,
                group_phase(schedule, &costs, |_| 0.0),
                dispatch,
                |_| f64::INFINITY
            ),
            1
        );
        assert_eq!(
            choose(4, 0.0, |_| (210.0, 102.0), dispatch, |_| f64::INFINITY),
            2
        );
    }

    /// Purpose: one runnable job bypasses preparation even on a wider token;
    /// one slot is the matching control for a phase with multiple jobs.
    #[test]
    fn effective_single_width_skips_cost_preparation() {
        for (slots, jobs) in [(8, 1), (1, 8)] {
            let mut engine = tenferro_cpu::CpuEngine::new(
                tenferro_cpu::Pool::builder()
                    .threads(slots)
                    .build()
                    .unwrap(),
            );
            crate::host_pool::lane_region(
                &mut engine,
                crate::host_pool::HostPoolSite::Replay,
                |exec| {
                    assert_eq!(
                        width::<f64, _>(
                            &exec,
                            None,
                            jobs,
                            0,
                            0,
                            || panic!("element preparation"),
                            |_| -> fn(usize) -> (f64, f64) { panic!("cost preparation") }
                        ),
                        1
                    );
                },
            )
            .unwrap();
        }
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

    /// Purpose: a cache-resident phase scales its element rate at both budgets;
    /// the ungated memory measurement is the control that keeps two slots inline.
    #[test]
    fn cache_resident_many_items_split_at_the_budget() {
        let (sum, largest, traffic) = (136531.0, 227.0, 7376896.0);
        let mut constants = Constants::reference(HostClass::Generic);
        constants.bandwidth = vec![60.0; 5];
        let elements = [traffic / 16.0, 0.0, 0.0, 0.0];
        let dispatch = |q| [0.0, 0.0, 600.0, 650.0, 700.0][q];
        for budget in [2, 4] {
            assert_eq!(
                choose(
                    budget,
                    traffic,
                    |_| (sum, largest),
                    dispatch,
                    phase_bandwidth(traffic, traffic, elements, &constants)
                ),
                budget
            );
        }
        assert_eq!(
            choose(
                2,
                traffic,
                |_| (sum, largest),
                dispatch,
                |_| traffic / 288719.0
            ),
            1
        );
        let element_time = traffic / 16.0 * 0.15972158132790892;
        let bandwidth = phase_bandwidth(traffic, traffic, elements, &constants);
        assert!((traffic / bandwidth(4) - element_time / 4.0).abs() < 1e-9);
    }

    /// Purpose: memory-resident phases stop gaining width above the measured
    /// aggregate plateau; cache-resident work is the control without this cap.
    #[test]
    fn memory_resident_bandwidth_stops_at_saturation() {
        let mut constants = Constants::reference(HostClass::Generic);
        constants.bandwidth = vec![
            f64::INFINITY,
            f64::INFINITY,
            60.0,
            80.0,
            80.0,
            80.0,
            80.0,
            80.0,
            80.0,
        ];
        let traffic = (128 << 20) as f64;
        let elements = [traffic / 16.0, 0.0, 0.0, 0.0];
        let bandwidth = phase_bandwidth(traffic, traffic, elements, &constants);
        assert_eq!(bandwidth(4), 80.0);
        let sum = elements[0] * constants.rows[0].e.at(traffic);
        assert_eq!(choose(8, traffic, |_| (sum, 0.0), |_| 0.0, bandwidth), 3);
        assert!(phase_bandwidth((4 << 20) as f64, traffic, elements, &constants)(4) > 80.0);
    }

    /// Purpose: a too-slow or decreasing measurement is rejected before interpolation;
    /// a ladder meeting the same minimum and order is retained.
    #[test]
    fn invalid_saturated_measurements_are_rejected() {
        let mut values = [f64::INFINITY, f64::INFINITY, 40.0, 100.0, 80.0, 120.0];
        validate_bandwidth(&mut values, 50.0);
        assert_eq!(values[2], f64::INFINITY);
        assert_eq!(values[4], f64::INFINITY);
        interpolate_bandwidth(&mut values);
        assert_eq!(values, [100.0, 100.0, 100.0, 100.0, 110.0, 120.0]);
        let mut valid = [f64::INFINITY, f64::INFINITY, 60.0, 100.0, 100.0, 120.0];
        validate_bandwidth(&mut valid, 50.0);
        assert_eq!(&valid[2..], &[60.0, 100.0, 100.0, 120.0]);
    }

    /// Purpose: only active element curves determine the cache knee, and a
    /// phase-specific minimum rejects low prefix knots while retaining later ones.
    #[test]
    fn saturation_uses_active_curves_and_their_memory_rate() {
        let mut constants = Constants::reference(HostClass::Generic);
        assert_eq!(
            constants.memory_knees[..2],
            [(64 << 20) as f64, (32 << 20) as f64]
        );
        constants.bandwidth = vec![10.0, 10.0, 10.0, 80.0, 80.0];
        let elements = [0.0, 1000.0, 0.0, 0.0];
        assert_eq!(
            phase_bandwidth((48 << 20) as f64, 16000.0, elements, &constants)(4),
            80.0
        );
        assert!(phase_bandwidth((24 << 20) as f64, 16000.0, elements, &constants)(4) > 80.0);
        assert_eq!(
            phase_bandwidth((48 << 20) as f64, 16000.0, elements, &constants)(3),
            80.0
        );
        constants.bandwidth[3..].fill(10.0);
        assert!(phase_bandwidth((48 << 20) as f64, 16000.0, elements, &constants)(3) > 10.0);
    }

    /// Purpose: noise after the knee must not admit a cap below the current
    /// single-thread rate; a cap above that rate remains usable.
    #[test]
    fn saturated_cap_cannot_undercut_a_faster_curve_knot() {
        let mut constants = Constants::reference(HostClass::Generic);
        constants.bandwidth = vec![27.0; 5];
        let elements = [0.0, 1000.0, 0.0, 0.0];
        let actual = phase_bandwidth((64 << 20) as f64, 16000.0, elements, &constants)(4);
        assert!(actual > 27.0);
        constants.bandwidth.fill(40.0);
        assert_eq!(
            phase_bandwidth((64 << 20) as f64, 16000.0, elements, &constants)(4),
            40.0
        );
    }

    /// Purpose: mixed classes and logical read traffic use the same weighted
    /// element time as the serial model; a one-class phase is the control.
    #[test]
    fn mixed_curves_weight_the_serial_element_rate() {
        let constants = Constants::reference(HostClass::Generic);
        let working_set = 7376896.0;
        let elements = [100.0, 200.0, 300.0, 400.0];
        let traffic = 24000.0;
        let expected = 400.0 * constants.rows[0].e.at(working_set)
            + 600.0 * constants.rows[1].e.at(working_set);
        let actual = phase_bandwidth(working_set, traffic, elements, &constants)(4);
        assert!((traffic / actual - expected / 4.0).abs() < 1e-12);
        let one = phase_bandwidth(working_set, traffic, [1000.0, 0.0, 0.0, 0.0], &constants)(4);
        assert_ne!(actual, one);
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

    /// Purpose: remaining calibration time bounds fixture bytes before allocation;
    /// an insufficient deadline cannot admit a memory-resident fixture.
    #[test]
    fn saturated_fixture_must_fit_the_remaining_budget() {
        let memory_bytes = 64 << 20;
        assert!(bandwidth_fixture_bytes(1_000_000.0, 1000.0, 0.1) < memory_bytes);
        assert!(bandwidth_fixture_bytes(9_000_000.0, 1000.0, 0.1) >= memory_bytes);
        assert_eq!(bandwidth_fixture_bytes(1000.0, 1000.0, 0.1), 0);
        assert_eq!(bandwidth_fixture_bytes(1000.0, 0.0, f64::INFINITY), 0);
    }

    /// Purpose: saturated model counts still feed a finite cost and valid width;
    /// ordinary counts are the control exercising the same decision path.
    #[test]
    fn saturated_summary_keeps_a_valid_width() {
        for count in [1, usize::MAX] {
            let mut summary = Summary::default();
            for _ in 0..2 {
                let counts = summary
                    .push(
                        0,
                        crate::movement::Point {
                            runs: count,
                            elements: count,
                        },
                    )
                    .unwrap();
                summary
                    .record_job(&[
                        counts,
                        Counts::default(),
                        Counts::default(),
                        Counts::default(),
                    ])
                    .unwrap();
            }
            summary.finish().unwrap();
            let phase = super::fixed(&summary, &[(1.0, 1.0, 1.0); CLASSES]);
            assert!(phase(1).0.is_finite());
            assert!((1..=4).contains(&choose(4, 0.0, phase, |_| 1.0, |_| f64::INFINITY)));
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
            constants.reserved_entries
                * (std::mem::size_of::<f64>() + std::mem::size_of::<Option<Duration>>())
        );
        constants.bandwidth = Vec::with_capacity(3 * constants.reserved_entries);
        constants.dispatch = Vec::with_capacity(2 * constants.reserved_entries);
        assert_eq!(
            constants.retained_bytes(),
            constants.bandwidth.capacity() * std::mem::size_of::<f64>()
                + constants.dispatch.capacity() * std::mem::size_of::<Option<Duration>>()
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

    /// Purpose: accepted online constants outrank references and survive warm reentry.
    #[test]
    fn accepted_online_constants_are_reused() {
        let mut engine =
            tenferro_cpu::CpuEngine::new(tenferro_cpu::Pool::builder().threads(2).build().unwrap());
        let calls = Cell::new(0);
        for _ in 0..2 {
            region_with(
                &mut engine,
                |_, _, constants| {
                    calls.set(calls.get() + 1);
                    constants.rows[0].a += 1.0;
                    constants.dispatch[2] = Some(Duration::from_nanos(110));
                },
                |_, constants| {
                    let constants = constants.unwrap();
                    let rows = constants.rows;
                    assert_eq!(rows[0].a, CI_ROWS[0].a + 1.0);
                    assert_ne!(rows[0].a, CI_ROWS[0].a);
                    assert_eq!(rows[1], CI_ROWS[1]);
                    assert_eq!(
                        constants.dispatch_cost(2, || panic!("retained online dispatch")),
                        110.0
                    );
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
