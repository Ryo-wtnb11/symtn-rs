//! One movement model, with engine-local measurements and structural summaries.

use super::*;
use crate::movement::{job_range, Counts, Job, Summary, CLASSES};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
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
    generation: u64,
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
            generation: next_generation(),
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

// Entry creation and measurement both advance the revision, including after eviction.
fn next_generation() -> u64 {
    static GENERATION: AtomicU64 = AtomicU64::new(1);
    GENERATION.fetch_add(1, Ordering::Relaxed)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct DecisionKey {
    pub budget: usize,
    pub carries_team: bool,
    pub generation: u64,
    pub strategy: tenferro_cpu::CpuBatchStrategy,
    pub bytes: usize,
    pub reads_destination: bool,
    pub replays_plans: bool,
    pub destination_len: usize,
    pub destination_capacity: usize,
    pub members: usize,
}

/// A single schedule/key, with one usize per phase; no old key survives replacement.
#[derive(Clone, Debug, Default)]
pub(super) struct WidthMemo {
    key: Option<DecisionKey>,
    identity: Weak<()>,
    dispatch: Vec<Duration>,
    pub widths: Vec<usize>,
}

impl WidthMemo {
    pub fn update(
        &mut self,
        key: DecisionKey,
        identity: Weak<()>,
        dispatch: impl Iterator<Item = Duration> + Clone,
        build: impl FnOnce(&mut Vec<usize>) -> Result<(), OperationError>,
    ) -> Result<(), OperationError> {
        if self.key == Some(key)
            && self.identity.ptr_eq(&identity)
            && self.dispatch.iter().copied().eq(dispatch.clone())
        {
            return Ok(());
        }
        self.key = None;
        self.widths.clear();
        build(&mut self.widths)?;
        self.widths.shrink_to_fit();
        self.dispatch.clear();
        self.dispatch.extend(dispatch);
        self.dispatch.shrink_to_fit();
        self.identity = identity;
        self.key = Some(key);
        Ok(())
    }

    pub fn finish_destination(&mut self, len: usize, capacity: usize) {
        if let Some(key) = self.key.as_mut() {
            key.destination_len = len;
            key.destination_capacity = capacity;
        }
    }

    pub fn retained_bytes(&self) -> usize {
        self.widths.capacity() * core::mem::size_of::<usize>()
            + self.dispatch.capacity() * core::mem::size_of::<Duration>()
    }
}

pub(super) fn decision_key<D>(
    exec: &lanes::Exec<'_>,
    constants: Option<&Constants>,
    reads_destination: bool,
    replays_plans: bool,
    destination_len: usize,
    destination_capacity: usize,
    members: usize,
) -> DecisionKey {
    DecisionKey {
        budget: exec.budget(),
        carries_team: exec.token_ref().carries_team(),
        generation: constants.map_or(0, |constants| constants.generation),
        strategy: exec.batch_policy().strategy,
        bytes: core::mem::size_of::<D>(),
        reads_destination,
        replays_plans,
        destination_len,
        destination_capacity,
        members,
    }
}

pub(super) fn dispatch_key<'a>(
    exec: &'a lanes::Exec<'_>,
) -> impl Iterator<Item = Duration> + Clone + 'a {
    (2..=exec.budget()).map(|q| {
        exec.token_ref()
            .dispatch_cost(NonZeroUsize::new(q).unwrap())
    })
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
                    constants.generation = next_generation();
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

/// One candidate ladder and one expression, including inline execution.
fn choose(
    p: usize,
    jobs: usize,
    traffic: f64,
    phase: impl Fn(usize) -> f64,
    dispatch: impl Fn(usize) -> f64,
    bandwidth: impl Fn(usize) -> f64,
) -> usize {
    if p.min(jobs) <= 1 {
        return 1;
    }
    let candidates = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64];
    let mut best = 1;
    let mut time = f64::INFINITY;
    let roof = traffic / bandwidth(p);
    for q in candidates
        .into_iter()
        .chain((!candidates.contains(&p)).then_some(p))
    {
        if q > p.min(jobs) {
            continue;
        }
        let candidate = if q > 1 { dispatch(q) } else { 0.0 } + phase(q).max(roof);
        if candidate < time {
            best = q;
            time = candidate;
        }
    }
    best
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

pub(super) struct Prices<'a> {
    bytes: f64,
    dispatch: &'a dyn Fn(usize) -> f64,
    bandwidth: &'a dyn Fn(usize) -> f64,
    admits: &'a dyn Fn(usize, usize) -> bool,
}

impl Prices<'_> {
    fn leaf(&self, job: Job, minimum: usize, costs: &Costs, slots: usize) -> f64 {
        let (setup, work) = components(&job.counts, costs);
        let total = setup + work;
        if slots == 1 || minimum == 0 || !(self.admits)(slots, minimum) {
            return total;
        }
        let (setup, work) = components(&job.planned, costs);
        let planned = setup + work;
        let (count, traffic) = job.planned.iter().zip([2, 2, 1, 1]).fold(
            (0.0, 0.0),
            |(count, traffic), (item, reads)| {
                (
                    count + item.items as f64,
                    traffic + item.elements as f64 * reads as f64 * self.bytes,
                )
            },
        );
        total - planned
            + planned.min(
                count * (self.dispatch)(slots)
                    + (planned / slots as f64).max(traffic / (self.bandwidth)(slots)),
            )
    }
}

/// Mirrors lanes::Split, including its recursive job rounding and slot shares.
fn leaf_max(
    first: usize,
    jobs: usize,
    slots: usize,
    q: usize,
    fill: usize,
    leaf: &impl Fn(core::ops::Range<usize>, usize, usize) -> f64,
) -> f64 {
    if q <= 1 || jobs <= 1 {
        return leaf(first..first + jobs, slots, fill);
    }
    let left = q / 2;
    let middle = jobs.saturating_mul(left).div_ceil(q).clamp(1, jobs - 1);
    let left_slots = slots / q * left + (slots % q).min(left);
    let left_fill = fill / q * left + (fill % q) * left / q;
    leaf_max(first, middle, left_slots, left, left_fill, leaf).max(leaf_max(
        first + middle,
        jobs - middle,
        slots - left_slots,
        q - left,
        fill - left_fill,
        leaf,
    ))
}

pub(super) fn fixed<'a>(
    phase: &'a Summary,
    costs: &Costs,
) -> impl Fn(usize, usize, &Prices<'_>) -> f64 + 'a {
    prefix_phase(
        &phase.prefix,
        0,
        phase.prefix.len(),
        phase.planned_min,
        costs,
        None,
    )
}

pub(super) fn prefix_phase<'a>(
    prefix: &'a [Job],
    first: usize,
    jobs: usize,
    minimum: usize,
    costs: &Costs,
    fill: Option<usize>,
) -> impl Fn(usize, usize, &Prices<'_>) -> f64 + 'a {
    let costs = *costs;
    move |q, p, prices| {
        leaf_max(
            first,
            jobs,
            p,
            q,
            fill.unwrap_or(0),
            &|range, slots, fill_elements| {
                let movement = prices.leaf(job_range(prefix, range), minimum, &costs, slots);
                let (a, r, e) = costs[2];
                movement
                    + fill.map_or(0.0, |_| {
                        a + if fill_elements == 0 {
                            0.0
                        } else {
                            r + e * fill_elements as f64
                        }
                    })
            },
        )
    }
}

/// Consecutive GEMM batches have serial barriers; their dense estimate remains
/// the provider's, and their movement estimate conservatively withholds fan-out.
pub(super) fn group_phase<'a>(
    schedule: &'a TreeTransformParallelSchedule,
    costs: &Costs,
    gemm: impl Fn(usize) -> f64 + 'a,
) -> impl Fn(usize, usize, &Prices<'_>) -> f64 + 'a {
    let costs = *costs;
    move |q, _, _| {
        let mut critical = 0.0;
        for first in (0..schedule.pack_movement.len()).step_by(q) {
            critical += (first..(first + q).min(schedule.pack_movement.len()))
                .map(|group| {
                    [
                        &schedule.pack_movement[group],
                        &schedule.scatter_movement[group],
                    ]
                    .into_iter()
                    .map(|summary| {
                        let (a, w) = components(&summary.counts, &costs);
                        a + w
                    })
                    .sum::<f64>()
                        + gemm(group)
                })
                .fold(0.0, f64::max);
        }
        critical
    }
}

/// Member ranges fuse each role into one serial walk; no retained plan executes.
pub(super) fn member_phase(
    phase: &Summary,
    costs: &Costs,
    members: usize,
) -> impl Fn(usize, usize, &Prices<'_>) -> f64 {
    let (setup, work) = components(&phase.counts, costs);
    move |q, _, _| setup + members.div_ceil(q) as f64 * work
}

/// Each phase supplies its actual critical path; class remains table data.
#[allow(clippy::too_many_arguments)]
pub(super) fn width<D: DenseRecouplingScalar, F: Fn(usize, usize, &Prices<'_>) -> f64>(
    exec: &lanes::Exec<'_>,
    constants: Option<&Constants>,
    jobs: usize,
    working_set: u128,
    traffic: u128,
    elements: impl FnOnce() -> [f64; CLASSES],
    admits: impl Fn(usize, usize) -> bool,
    prepare: impl Fn(&Costs) -> F,
) -> usize {
    let budget = exec.budget();
    if budget.min(jobs) <= 1 {
        return 1;
    }
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
    let dispatch = |q| {
        constants.dispatch_cost(q, || {
            exec.token_ref()
                .dispatch_cost(NonZeroUsize::new(q).unwrap())
        })
    };
    width_with(
        budget,
        constants,
        std::mem::size_of::<D>(),
        jobs,
        working_set,
        traffic,
        elements,
        admits,
        prepare,
        dispatch,
    )
}

#[allow(clippy::too_many_arguments)]
fn width_with<F: Fn(usize, usize, &Prices<'_>) -> f64>(
    budget: usize,
    constants: &Constants,
    bytes: usize,
    jobs: usize,
    working_set: u128,
    traffic: u128,
    elements: impl FnOnce() -> [f64; CLASSES],
    admits: impl Fn(usize, usize) -> bool,
    prepare: impl Fn(&Costs) -> F,
    dispatch: impl Fn(usize) -> f64,
) -> usize {
    if budget.min(jobs) <= 1 {
        return 1;
    }
    #[cfg(any(test, feature = "testing"))]
    crate::probe::add(crate::probe::Event::MovementModelEvaluation, 1);
    let bytes = bytes as f64;
    let working_set = working_set as f64 * bytes;
    let costs = core::array::from_fn(|c| {
        let row = constants.row(c);
        (row.a, row.r.at(working_set), row.e.at(working_set))
    });
    let traffic = traffic as f64 * bytes;
    let phase = prepare(&costs);
    let bandwidth = phase_bandwidth(working_set, traffic, elements(), constants);
    let prices = Prices {
        bytes,
        dispatch: &dispatch,
        bandwidth: &bandwidth,
        admits: &admits,
    };
    choose(
        budget,
        jobs,
        traffic,
        |q| phase(q, budget, &prices),
        &dispatch,
        &bandwidth,
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
    // ponytail: replace this interim measurement with the threadteam calibration hook
    // (tensor4all/threadteam-rs#92);
    // delete it when threadteam can calibrate a running pool.
    let budget = exec.budget();
    for (q, slot) in constants
        .dispatch
        .iter_mut()
        .enumerate()
        .take(budget + 1)
        .skip(2)
    {
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
        *slot = validated_dispatch(samples[2], table);
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

    /// Purpose: tied candidates retain the smaller width; a strict improvement changes it.
    #[test]
    fn ties_choose_the_smaller_width() {
        for (last, expected) in [(6.0, 2), (5.0, 3)] {
            assert_eq!(
                choose(
                    3,
                    3,
                    0.0,
                    |q| [0.0, 12.0, 6.0, last][q],
                    |_| 0.0,
                    |_| f64::INFINITY
                ),
                expected
            );
        }
    }

    /// Purpose: serial or indivisible phases never prepare model data.
    #[test]
    fn serial_choice_reads_no_model_data() {
        for (slots, jobs) in [(1, 8), (8, 1)] {
            assert_eq!(
                choose(
                    slots,
                    jobs,
                    0.0,
                    |_| panic!("phase"),
                    |_| panic!("dispatch"),
                    |_| panic!("bandwidth")
                ),
                1
            );
        }
        assert_eq!(
            choose(2, 2, 0.0, |q| 12.0 / q as f64, |_| 0.0, |_| f64::INFINITY),
            2
        );
    }

    /// Purpose: accepted dispatch measurements drive the width and skip table reads;
    /// absent and rejected measurements are the table-route controls.
    #[test]
    fn movement_dispatch_uses_measurement_or_table() {
        let table = Duration::from_nanos(600);
        let mut engine = tenferro_cpu::CpuEngine::new(
            tenferro_cpu::Pool::builder()
                .threads(2)
                .dispatch_costs({
                    let flat = tenferro_cpu::threadteam::WidthCosts::flat(table);
                    tenferro_cpu::threadteam::DispatchCosts::new(flat, flat, flat)
                })
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
                            |_, _| false,
                            |_| |q, _, _| 400.0 / q as f64,
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

    fn memo_key(constants: &Constants) -> DecisionKey {
        DecisionKey {
            budget: 4,
            carries_team: true,
            generation: constants.generation,
            strategy: tenferro_cpu::CpuBatchStrategy::Auto,
            bytes: core::mem::size_of::<f64>(),
            reads_destination: false,
            replays_plans: true,
            destination_len: 0,
            destination_capacity: 0,
            members: 1,
        }
    }

    fn direct_phase_width(phase: &Summary, constants: &Constants, key: DecisionKey) -> usize {
        width_with(
            key.budget,
            constants,
            key.bytes,
            phase.prefix.len(),
            phase.working_set as u128,
            phase.logical_traffic(key.reads_destination) as u128,
            || phase.counts.map(|count| count.elements as f64),
            |slots, elements| key.carries_team && key.replays_plans && elements >= slots,
            |costs| fixed(phase, costs),
            |q| constants.dispatch_cost(q, || Duration::from_nanos((q * 23) as u64)),
        )
    }

    /// Purpose: memoized widths match direct model evaluation over varied
    /// structural plans and constants; both inline and split choices occur.
    #[test]
    fn memoized_widths_match_direct_random_plans_and_constants() {
        let mut seed = 31u64;
        let mut next = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            seed >> 32
        };
        let mut memo = WidthMemo::default();
        let mut split = false;
        let mut inline = false;
        for _ in 0..64 {
            let mut constants = Constants::reference(HostClass::Generic);
            for row in &mut constants.rows {
                row.a = next() as f64 % 100.0;
                for (_, value) in &mut row.r.points {
                    *value = next() as f64 % 30.0 + 0.1;
                }
                for (_, value) in &mut row.e.points {
                    *value = next() as f64 % 10.0 + 0.1;
                }
            }
            constants.memory_knees = core::array::from_fn(|c| constants.rows[c].e.memory_knee());
            constants.dispatch = (0..=8)
                .map(|_| Some(Duration::from_nanos(next() % 10000)))
                .collect();
            let mut key = memo_key(&constants);
            key.budget = next() as usize % 8 + 1;
            key.reads_destination = next() % 2 != 0;
            key.bytes = if next() % 2 == 0 { 4 } else { 16 };
            let phases: Vec<_> = (0..next() as usize % 5 + 1)
                .map(|_| {
                    let jobs: Vec<_> = (0..next() as usize % 16 + 1)
                        .map(|_| {
                            let counts = core::array::from_fn(|_| Counts {
                                items: next() % 3 + 1,
                                runs: next() % 50 + 1,
                                elements: next() % 1000 + 1,
                            });
                            Job {
                                planned: if next() % 2 == 0 {
                                    counts
                                } else {
                                    [Counts::default(); CLASSES]
                                },
                                counts,
                            }
                        })
                        .collect();
                    let mut phase = compiled_jobs(&jobs, 1);
                    phase.traffic = phase.counts.iter().map(|count| count.elements * 2).sum();
                    phase.working_set = phase.traffic;
                    phase.destination_reads = phase.traffic / 2;
                    phase
                })
                .collect();
            let expected: Vec<_> = phases
                .iter()
                .map(|phase| direct_phase_width(phase, &constants, key))
                .collect();
            split |= expected.iter().any(|&width| width > 1);
            inline |= expected.contains(&1);
            let identity = Arc::new(());
            for _ in 0..2 {
                memo.update(
                    key,
                    Arc::downgrade(&identity),
                    core::iter::once(Duration::ZERO),
                    |widths| {
                        widths.extend(
                            phases
                                .iter()
                                .map(|phase| direct_phase_width(phase, &constants, key)),
                        );
                        Ok(())
                    },
                )
                .unwrap();
                assert_eq!(memo.widths, expected);
            }
        }
        assert!(split && inline);
    }

    /// Purpose: budget and constants revisions rebuild every phase once;
    /// an identical key evaluates no model, and both changes alter a choice.
    #[test]
    fn memo_recomputes_only_when_budget_or_constants_change() {
        use std::cell::Cell;
        let phase = compiled_jobs(
            &vec![
                Job {
                    counts: [Counts {
                        items: 1,
                        runs: 1,
                        elements: 1000000
                    }; CLASSES],
                    ..Job::default()
                };
                8
            ],
            0,
        );
        let phases = [phase];
        let identity = Arc::new(());
        let mut constants = Constants::reference(HostClass::Generic);
        constants.dispatch = vec![Some(Duration::ZERO); 5];
        let mut key = memo_key(&constants);
        key.budget = 2;
        let mut memo = WidthMemo::default();
        let evaluations = Cell::new(0);
        let mut previous = None;
        for change in 0..3 {
            if change == 1 {
                key.budget = 4;
            }
            if change == 2 {
                constants.dispatch.fill(Some(Duration::from_secs(1)));
                constants.generation = next_generation();
                key.generation = constants.generation;
            }
            evaluations.set(0);
            for replay in 0..2 {
                memo.update(
                    key,
                    Arc::downgrade(&identity),
                    core::iter::once(Duration::ZERO),
                    |widths| {
                        for phase in &phases {
                            evaluations.set(evaluations.get() + 1);
                            widths.push(direct_phase_width(phase, &constants, key));
                        }
                        Ok(())
                    },
                )
                .unwrap();
                assert_eq!(
                    evaluations.replace(0),
                    if replay == 0 { phases.len() } else { 0 }
                );
            }
            assert_eq!(
                memo.widths,
                phases
                    .iter()
                    .map(|phase| direct_phase_width(phase, &constants, key))
                    .collect::<Vec<_>>()
            );
            if let Some(previous) = previous {
                assert_ne!(memo.widths, previous);
            }
            previous = Some(memo.widths.clone());
        }
    }

    /// Purpose: schedule, gate, policy, scalar, writer, fill and member inputs
    /// invalidate the single entry; each unchanged input is the warm control.
    #[test]
    fn memo_invalidates_every_other_decision_input() {
        let constants = Constants::reference(HostClass::Generic);
        let base = memo_key(&constants);
        let identity = Arc::new(());
        let other_identity = Arc::new(());
        let variants = [
            DecisionKey {
                carries_team: false,
                ..base
            },
            DecisionKey {
                strategy: tenferro_cpu::CpuBatchStrategy::Inner,
                ..base
            },
            DecisionKey {
                bytes: base.bytes * 2,
                ..base
            },
            DecisionKey {
                reads_destination: true,
                ..base
            },
            DecisionKey {
                replays_plans: false,
                ..base
            },
            DecisionKey {
                destination_len: 1,
                ..base
            },
            DecisionKey {
                destination_capacity: 1,
                ..base
            },
            DecisionKey { members: 2, ..base },
        ];
        for (key, id, cost) in variants
            .into_iter()
            .map(|key| (key, &identity, Duration::ZERO))
            .chain([
                (base, &other_identity, Duration::ZERO),
                (base, &identity, Duration::from_nanos(1)),
            ])
        {
            let mut memo = WidthMemo::default();
            memo.update(
                base,
                Arc::downgrade(&identity),
                core::iter::once(Duration::ZERO),
                |widths| {
                    widths.push(1);
                    Ok(())
                },
            )
            .unwrap();
            memo.update(key, Arc::downgrade(id), core::iter::once(cost), |widths| {
                widths.push(2);
                Ok(())
            })
            .unwrap();
            assert_eq!(memo.widths, [2]);
            memo.update(key, Arc::downgrade(id), core::iter::once(cost), |_| {
                panic!("warm key rebuilt")
            })
            .unwrap();
        }
    }

    fn compiled_jobs(jobs: &[Job], minimum: usize) -> Summary {
        let mut phase = Summary::default();
        phase.planned_min = minimum;
        for job in jobs {
            for (count, other) in phase.counts.iter_mut().zip(job.counts) {
                *count = count.add(other).unwrap();
            }
            for (count, other) in phase.planned.iter_mut().zip(job.planned) {
                *count = count.add(other).unwrap();
            }
            phase.record_job(&job.counts).unwrap();
        }
        phase.finish().unwrap();
        phase
    }

    fn uniform_job(items: u64, elements: u64, runs: u64, planned: bool) -> Job {
        let mut counts = [Counts::default(); CLASSES];
        counts[0] = Counts {
            items,
            runs: runs * items,
            elements: elements * items,
        };
        Job {
            counts,
            planned: if planned {
                counts
            } else {
                [Counts::default(); CLASSES]
            },
        }
    }

    fn reference_choice(phase: &Summary, budget: usize, traffic: f64, admit: bool) -> usize {
        let mut constants = Constants::reference(HostClass::Generic);
        constants.bandwidth = vec![
            5.303316889,
            5.303316889,
            6.580693628,
            7.873770916,
            9.166848204,
        ];
        let costs = core::array::from_fn(|c| {
            let row = constants.row(c);
            (row.a, row.r.at(traffic), row.e.at(traffic))
        });
        let dispatch = |q| [0.0, 0.0, 110.0, 200.0, 290.0][q];
        let bandwidth = phase_bandwidth(
            traffic,
            traffic,
            phase.counts.map(|count| count.elements as f64),
            &constants,
        );
        let admits = |_, _| admit;
        let prices = Prices {
            bytes: 8.0,
            dispatch: &dispatch,
            bandwidth: &bandwidth,
            admits: &admits,
        };
        let model = super::fixed(phase, &costs);
        choose(
            budget,
            phase.prefix.len(),
            traffic,
            |q| model(q, budget, &prices),
            dispatch,
            &bandwidth,
        )
    }

    /// Purpose: large admitted roles keep leaves with inner fan-out; withholding
    /// that credit is the serial-item control that exhausts the outer slots.
    #[test]
    fn large_planned_roles_choose_leaves_with_inner_fanout() {
        let phase = compiled_jobs(
            &[1, 4, 9, 4, 1].map(|items| uniform_job(items, 65536, 256, true)),
            65536,
        );
        let traffic = 19.0 * 65536.0 * 16.0;
        for budget in [2, 4] {
            let chosen = reference_choice(&phase, budget, traffic, true);
            assert!(budget / chosen >= 2);
            assert_eq!(reference_choice(&phase, budget, traffic, false), budget);
        }
    }

    /// Purpose: distributed serial work still pays for outer splitting; dispatch
    /// dominating its total time is the inline control.
    #[test]
    fn dominant_serial_job_and_small_jobs_split() {
        let mut jobs = vec![uniform_job(1, 10, 1, false); 100];
        jobs[0] = uniform_job(1, 1000, 1, false);
        let phase = compiled_jobs(&jobs, 0);
        let costs = [(0.0, 0.0, 1.0); CLASSES];
        for budget in [2, 4] {
            for (dispatch_ns, split) in [(10.0, true), (2000.0, false)] {
                let dispatch = |_| dispatch_ns;
                let bandwidth = |_| f64::INFINITY;
                let prices = Prices {
                    bytes: 8.0,
                    dispatch: &dispatch,
                    bandwidth: &bandwidth,
                    admits: &|_, _| true,
                };
                let model = super::fixed(&phase, &costs);
                let chosen = choose(
                    budget,
                    jobs.len(),
                    0.0,
                    |q| model(q, budget, &prices),
                    dispatch,
                    bandwidth,
                );
                assert_eq!(chosen > 1, split);
            }
        }
    }

    /// Purpose: a dominant admitted role retains inner fan-out while distributed
    /// work still splits at four slots; serial item pricing exhausts outer slots.
    #[test]
    fn dominant_admitted_roles_keep_a_split_with_inner_fanout() {
        let mut jobs = vec![uniform_job(1, 2304, 1, true); 201];
        jobs[0] = uniform_job(1, 589824, 1, true);
        let phase = compiled_jobs(&jobs, 2304);
        assert_eq!(reference_choice(&phase, 2, 16809984.0, true), 1);
        assert_eq!(reference_choice(&phase, 4, 16809984.0, true), 3);
        assert_eq!(reference_choice(&phase, 4, 16809984.0, false), 4);
    }

    /// Purpose: small gated roles use every outer slot on the measured cache
    /// curve; a single ownership job is the indivisible control.
    #[test]
    fn small_roles_choose_the_budget_without_inner_credit() {
        let mut jobs = vec![uniform_job(3, 256, 16, true); 597];
        jobs.extend([uniform_job(2, 256, 16, true); 4]);
        jobs.extend([uniform_job(1, 256, 16, true); 2]);
        let phase = compiled_jobs(&jobs, 256);
        for budget in [2, 4] {
            assert_eq!(reference_choice(&phase, budget, 7376896.0, false), budget);
            assert_eq!(
                reference_choice(
                    &compiled_jobs(&[uniform_job(1801, 256, 16, true)], 256),
                    budget,
                    7376896.0,
                    false
                ),
                1
            );
        }
    }

    /// Purpose: slot count cannot change a serial-only job; an admitted planned
    /// job is the control whose aggregate time decreases on wider leaves.
    #[test]
    fn serial_only_items_are_independent_of_leaf_slots() {
        let costs = [(1.0, 1.0, 1.0); CLASSES];
        let prices = Prices {
            bytes: 8.0,
            dispatch: &|_| 1.0,
            bandwidth: &|_| f64::INFINITY,
            admits: &|_, _| true,
        };
        let serial = uniform_job(2, 100, 4, false);
        let planned = uniform_job(2, 100, 4, true);
        for slots in [1, 2, 4] {
            assert_eq!(prices.leaf(serial, 100, &costs, slots), 210.0);
        }
        assert!(prices.leaf(planned, 100, &costs, 4) < prices.leaf(planned, 100, &costs, 2));
    }

    /// Purpose: cumulative job components reproduce independent linear scans for
    /// every range and preserve S_q under the same actual leaf partition.
    #[test]
    fn prefix_leaf_sums_match_random_linear_scans() {
        let mut seed = 17u64;
        for len in 1..30 {
            let jobs = (0..len)
                .map(|_| {
                    let counts = core::array::from_fn(|_| {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                        let items = 1 + (seed >> 32) % 4;
                        Counts {
                            items,
                            runs: items * (1 + seed % 19),
                            elements: items * (1 + seed % 31),
                        }
                    });
                    let planned = core::array::from_fn(|c| {
                        if c % 2 == 0 {
                            counts[c]
                        } else {
                            Counts::default()
                        }
                    });
                    Job { counts, planned }
                })
                .collect::<Vec<_>>();
            let phase = compiled_jobs(&jobs, 1);
            let costs = [(2.0, 3.0, 5.0); CLASSES];
            let prices = Prices {
                bytes: 8.0,
                dispatch: &|s| s as f64,
                bandwidth: &|s| 10.0 * s as f64,
                admits: &|_, _| true,
            };
            for q in 1..=len.min(8) {
                let (actual_sum, expected_sum) = (Cell::new(0.0), Cell::new(0.0));
                leaf_max(0, len, 8, q, 0, &|range, slots, _| {
                    let mut expected = Job::default();
                    for job in &jobs[range.clone()] {
                        for (left, right) in expected.counts.iter_mut().zip(job.counts) {
                            left.items += right.items;
                            left.runs += right.runs;
                            left.elements += right.elements;
                        }
                        for (left, right) in expected.planned.iter_mut().zip(job.planned) {
                            left.items += right.items;
                            left.runs += right.runs;
                            left.elements += right.elements;
                        }
                    }
                    let actual = job_range(&phase.prefix, range);
                    assert_eq!(actual, expected);
                    actual_sum.set(actual_sum.get() + prices.leaf(actual, 1, &costs, slots));
                    let linear_time = |counts: &[Counts; CLASSES]| {
                        counts
                            .iter()
                            .map(|c| (2 * c.items + 3 * c.runs + 5 * c.elements) as f64)
                            .sum::<f64>()
                    };
                    let total = linear_time(&expected.counts);
                    let planned = linear_time(&expected.planned);
                    let count = expected.planned.iter().map(|c| c.items).sum::<u64>() as f64;
                    let traffic = expected
                        .planned
                        .iter()
                        .enumerate()
                        .map(|(c, item)| item.elements as f64 * if c < 2 { 16.0 } else { 8.0 })
                        .sum::<f64>();
                    let linear = if slots == 1 {
                        total
                    } else {
                        total - planned
                            + planned.min(
                                count * slots as f64
                                    + (planned / slots as f64).max(traffic / (10 * slots) as f64),
                            )
                    };
                    expected_sum.set(expected_sum.get() + linear);
                    0.0
                });
                assert_eq!(actual_sum.get(), expected_sum.get());
            }
        }
    }

    /// Purpose: recursive splits use the executor's uneven job boundaries and
    /// whole-token remainders, rather than equally sized flat job chunks.
    #[test]
    fn leaf_partition_matches_recursive_ownership_boundaries() {
        let observed = std::cell::RefCell::new(Vec::new());
        leaf_max(0, 7, 8, 3, 0, &|range, slots, _| {
            observed.borrow_mut().push((range, slots));
            0.0
        });
        assert_eq!(*observed.borrow(), [(0..3, 3), (3..5, 3), (5..7, 2)]);
        assert_ne!(*observed.borrow(), [(0..3, 3), (3..6, 3), (6..7, 2)]);
    }

    /// Purpose: the total-slot bandwidth floor applies equally to inline and
    /// split candidates; removing the floor allows the widest choice.
    #[test]
    fn aggregate_bandwidth_uses_all_slots_at_every_width() {
        assert_eq!(
            choose(
                4,
                8,
                1000.0,
                |q| 100.0 / q as f64,
                |_| 0.0,
                |slots| {
                    assert_eq!(slots, 4);
                    1.0
                }
            ),
            1
        );
        assert_eq!(
            choose(
                4,
                8,
                1000.0,
                |q| 100.0 / q as f64,
                |_| 0.0,
                |_| f64::INFINITY
            ),
            4
        );
    }

    /// Purpose: only approved candidate widths and the actual token budget are
    /// considered, never an unlisted job-count cap.
    #[test]
    fn candidate_widths_use_the_approved_ladder_and_budget() {
        assert_eq!(
            choose(7, 5, 0.0, |q| 100.0 / q as f64, |_| 0.0, |_| f64::INFINITY),
            4
        );
        assert_eq!(
            choose(7, 7, 0.0, |q| 100.0 / q as f64, |_| 0.0, |_| f64::INFINITY),
            7
        );
    }

    /// Purpose: a pack leaf pays column work plus its actual cold-fill share;
    /// warm scratch is the control without setup or fill traffic.
    #[test]
    fn pack_columns_and_cold_fill_share_add_their_prices() {
        let phase = compiled_jobs(&[uniform_job(1, 3, 1, false); 4], 0);
        let costs = [(1.0, 1.0, 1.0); CLASSES];
        let prices = Prices {
            bytes: 8.0,
            dispatch: &|_| 0.0,
            bandwidth: &|_| f64::INFINITY,
            admits: &|_, _| false,
        };
        assert_eq!(
            prefix_phase(&phase.prefix, 0, 4, 0, &costs, Some(8))(2, 4, &prices),
            16.0
        );
        assert_eq!(
            prefix_phase(&phase.prefix, 0, 4, 0, &costs, None)(2, 4, &prices),
            10.0
        );
        let tiny = std::cell::RefCell::new(Vec::new());
        leaf_max(0, 4, 4, 4, 1, &|_, _, fill| {
            tiny.borrow_mut().push(fill);
            0.0
        });
        assert_eq!(*tiny.borrow(), [0, 0, 0, 1]);
    }

    /// Purpose: fused member ranges pay setup once per role and range; the
    /// per-member setup model is the control with a different total.
    #[test]
    fn member_ranges_price_setup_at_the_executed_width() {
        let phase = compiled_jobs(&[uniform_job(1, 2, 1, false)], 0);
        let costs = [(1.0, 1.0, 1.0); CLASSES];
        let prices = Prices {
            bytes: 8.0,
            dispatch: &|_| 0.0,
            bandwidth: &|_| f64::INFINITY,
            admits: &|_, _| false,
        };
        assert_eq!(member_phase(&phase, &costs, 4)(1, 4, &prices), 13.0);
        assert_eq!(member_phase(&phase, &costs, 4)(2, 4, &prices), 7.0);
        assert_ne!(member_phase(&phase, &costs, 4)(2, 4, &prices), 8.0);
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
        let dispatch = |q| 10.0 * q as f64;
        let prices = Prices {
            bytes: 8.0,
            dispatch: &dispatch,
            bandwidth: &|_| f64::INFINITY,
            admits: &|_, _| false,
        };
        assert_eq!(
            [
                phase(1, 4, &prices),
                phase(2, 4, &prices),
                phase(3, 4, &prices),
                phase(4, 4, &prices)
            ],
            [210.0, 204.0, 105.0, 102.0]
        );
        assert_eq!(
            choose(
                4,
                4,
                0.0,
                |q| phase(q, 4, &prices),
                dispatch,
                |_| f64::INFINITY
            ),
            3
        );
        assert_eq!(
            choose(
                4,
                4,
                0.0,
                |q| group_phase(schedule, &costs, |_| 0.0)(q, 4, &prices),
                dispatch,
                |_| f64::INFINITY
            ),
            1
        );
        assert_eq!(
            choose(
                4,
                4,
                0.0,
                |q| 102.0_f64.max(210.0 / q as f64),
                dispatch,
                |_| f64::INFINITY
            ),
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
                            |_, _| false,
                            |_| -> fn(usize, usize, &Prices<'_>) -> f64 {
                                panic!("cost preparation")
                            }
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

    /// Purpose: cache-resident bandwidth follows the element curve; applying
    /// the memory roof prematurely is the slower negative control.
    #[test]
    fn cache_resident_bandwidth_uses_the_element_curve() {
        let traffic = 7376896.0;
        let mut constants = Constants::reference(HostClass::Generic);
        constants.bandwidth = vec![60.0; 5];
        let elements = [traffic / 16.0, 0.0, 0.0, 0.0];
        let bandwidth = phase_bandwidth(traffic, traffic, elements, &constants);
        let element_time = traffic / 16.0 * 0.15972158132790892;
        assert!((traffic / bandwidth(4) - element_time / 4.0).abs() < 1e-9);
        assert!(bandwidth(2) > traffic / 288719.0);
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
        assert_eq!(
            choose(8, 8, traffic, |q| sum / q as f64, |_| 0.0, bandwidth),
            2
        );
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
            let prices = Prices {
                bytes: 8.0,
                dispatch: &|_| 1.0,
                bandwidth: &|_| f64::INFINITY,
                admits: &|_, _| false,
            };
            assert!(phase(1, 4, &prices).is_finite());
            assert!((1..=4).contains(&choose(
                4,
                2,
                0.0,
                |q| phase(q, 4, &prices),
                |_| 1.0,
                |_| f64::INFINITY
            )));
            assert!(job_range(&summary.prefix, 1..2).counts[0].elements > 0);
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
