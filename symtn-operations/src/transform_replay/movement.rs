//! One movement model, with engine-local measurements and structural summaries.

use super::*;
use crate::movement::{job_range, planned_range, Counts, Job, Planned, Summary, CLASSES};
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
    /// The working set the bandwidth ladder was measured at; infinite until then.
    bandwidth_bytes: f64,
    dispatch: Vec<Option<Duration>>,
    /// One empty engine entry's median cost in ns, when measured.
    entry: Option<f64>,
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
            bandwidth_bytes: f64::INFINITY,
            dispatch: Vec::new(),
            entry: None,
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

/// At most sixteen structures, most recently used first; dead identities are
/// reused before live entries. Each replacement keeps its vector capacities.
#[derive(Clone, Debug, Default)]
pub(super) struct WidthMemo {
    entries: Vec<WidthDecision>,
}

const MAX_DECISIONS: usize = 16;

#[derive(Clone, Debug, Default)]
pub(super) struct WidthDecision {
    key: Option<DecisionKey>,
    identity: Weak<()>,
    dispatch: Vec<Duration>,
    pub widths: Vec<usize>,
    pub chunks: Vec<chunk::ChunkMovement>,
}

impl WidthMemo {
    pub fn current(&self) -> &WidthDecision {
        &self.entries[0]
    }

    pub fn current_mut(&mut self) -> &mut WidthDecision {
        &mut self.entries[0]
    }

    pub fn update(
        &mut self,
        key: DecisionKey,
        identity: Weak<()>,
        dispatch: impl Iterator<Item = Duration> + Clone,
        build: impl FnOnce(
            &mut Vec<usize>,
            &mut Vec<chunk::ChunkMovement>,
        ) -> Result<(), OperationError>,
    ) -> Result<(), OperationError> {
        let index = if let Some(index) = self
            .entries
            .iter()
            .position(|entry| entry.identity.ptr_eq(&identity))
        {
            index
        } else if let Some(index) = self
            .entries
            .iter()
            .position(|entry| entry.identity.strong_count() == 0)
        {
            index
        } else if self.entries.len() < MAX_DECISIONS {
            if self.entries.is_empty() {
                self.entries.reserve_exact(MAX_DECISIONS);
            }
            self.entries.push(WidthDecision::default());
            self.entries.len() - 1
        } else {
            self.entries.len() - 1
        };
        self.entries[..=index].rotate_right(1);
        let entry = self.current_mut();
        if entry.key == Some(key)
            && entry.identity.ptr_eq(&identity)
            && entry.dispatch.iter().copied().eq(dispatch.clone())
        {
            return Ok(());
        }
        entry.key = None;
        entry.widths.clear();
        build(&mut entry.widths, &mut entry.chunks)?;
        entry.dispatch.clear();
        entry.dispatch.extend(dispatch);
        entry.identity = identity;
        entry.key = Some(key);
        Ok(())
    }

    pub fn finish_destination(&mut self, len: usize, capacity: usize) {
        if let Some(key) = self.current_mut().key.as_mut() {
            key.destination_len = len;
            key.destination_capacity = capacity;
        }
    }

    pub fn retained_bytes(&self) -> usize {
        self.entries.capacity() * core::mem::size_of::<WidthDecision>()
            + self
                .entries
                .iter()
                .map(|entry| {
                    // A dead Weak can be the last owner of its marker's Arc control block.
                    2 * core::mem::size_of::<usize>()
                        + entry.widths.capacity() * core::mem::size_of::<usize>()
                        + entry.dispatch.capacity() * core::mem::size_of::<Duration>()
                        + entry.chunks.capacity() * core::mem::size_of::<chunk::ChunkMovement>()
                        + entry
                            .chunks
                            .iter()
                            .map(chunk::ChunkMovement::retained_bytes)
                            .sum::<usize>()
                })
                .sum::<usize>()
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
    let budget = if exec.batch_policy().strategy == tenferro_cpu::CpuBatchStrategy::Auto {
        exec.budget()
    } else {
        1
    };
    (2..=budget).map(|q| {
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
    let mut deadline = retained.is_none().then(|| Instant::now() + ONLINE_BUDGET);
    if retained.is_none() {
        let mut candidate = Constants::reference(HostClass::detect());
        candidate.reserved_entries = budget + 1;
        symtn_dense::engine_slot::checkin(engine, candidate, Constants::retained_bytes);
        retained = symtn_dense::engine_slot::checkout::<Constants>(engine);
    }
    // An entry is timed from outside every region, so before entering one.
    let entry = match retained.as_ref() {
        Some(constants) if !constants.attempted => {
            let deadline = *deadline.get_or_insert_with(|| Instant::now() + ONLINE_BUDGET);
            measure_entry(engine, deadline)
        }
        _ => None,
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::host_pool::lane_region(
            engine,
            crate::host_pool::HostPoolSite::Replay,
            |mut exec| {
                if let Some(constants) = retained.as_mut().filter(|constants| !constants.attempted)
                {
                    constants.attempted = true;
                    constants.entry = entry;
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
                    complete_dispatch(&mut constants.dispatch, |q| {
                        exec.token_ref()
                            .dispatch_cost(NonZeroUsize::new(q).unwrap())
                    });
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

/// The candidate widths below the budget; the budget itself is the last.
const CANDIDATES: [usize; 12] = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64];

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
    let mut best = 1;
    let mut time = f64::INFINITY;
    let roof = traffic / bandwidth(p);
    for q in CANDIDATES
        .into_iter()
        .chain((!CANDIDATES.contains(&p)).then_some(p))
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
    /// A leaf of `slots` slots running `job`, whose planned roles have the
    /// extremes `roles`. The gate's admission is monotone in a role's
    /// elements, so the leaf runs every planned role with inner fan-out when
    /// it admits the smallest; otherwise, when it admits the largest, that
    /// role is credited alone, and the admitted roles between the two are
    /// priced serially (an upper bound).
    fn leaf(&self, job: Job, roles: Planned, costs: &Costs, slots: usize) -> f64 {
        let (setup, work) = components(&job.counts, costs);
        let total = setup + work;
        if slots == 1 || roles.min == usize::MAX {
            return total;
        }
        let admitted = if (self.admits)(slots, roles.min) {
            job.planned
        } else if (self.admits)(slots, roles.largest.1.elements) {
            let (class, point) = roles.largest;
            let mut largest = [Counts::default(); CLASSES];
            largest[class] = Counts {
                items: 1,
                runs: u64::try_from(point.runs).unwrap_or(u64::MAX),
                elements: u64::try_from(point.elements).unwrap_or(u64::MAX),
            };
            largest
        } else {
            return total;
        };
        let (setup, work) = components(&admitted, costs);
        let planned = setup + work;
        let (count, traffic) = admitted.iter().zip([2, 2, 1, 1]).fold(
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
        &phase.planned_jobs,
        costs,
        None,
    )
}

/// `planned` is the [`planned_tree`](crate::movement::planned_tree) over
/// `prefix`'s jobs, or empty for a phase that plans no role.
pub(super) fn prefix_phase<'a>(
    prefix: &'a [Job],
    first: usize,
    jobs: usize,
    planned: &'a [Planned],
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
                let roles = planned_range(planned, range.clone());
                let movement = prices.leaf(job_range(prefix, range), roles, &costs, slots);
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

/// Engine regions one plan-group chunk enters: its pack, its GEMM batch and
/// its scatter (`chunk::groups`).
const CHUNK_ENTRIES: f64 = 3.0;

/// The fixed cost of one plan-group chunk: its engine entries at the
/// engine's measured entry cost, or none when that was not measured.
pub(super) fn chunk_fixed(constants: Option<&Constants>) -> f64 {
    constants
        .and_then(|constants| constants.entry)
        .map_or(0.0, |entry| CHUNK_ENTRIES * entry)
}

/// Consecutive GEMM batches have serial barriers; their dense estimate remains
/// the provider's, and their movement estimate conservatively withholds fan-out.
/// Every chunk pays `fixed`, and every chunk of more than one group forks its
/// batch: the chooser charges the first chunk's D(q), which always holds q
/// groups (q never exceeds the group count), so the later chunks' forks are
/// charged here.
pub(super) fn group_phase<'a>(
    schedule: &'a TreeTransformParallelSchedule,
    costs: &Costs,
    gemm: impl Fn(usize) -> f64 + 'a,
    fixed: f64,
) -> impl Fn(usize, usize, &Prices<'_>) -> f64 + 'a {
    let costs = *costs;
    move |q, _, prices| {
        let groups = schedule.pack_movement.len();
        let mut critical = 0.0;
        for first in (0..groups).step_by(q) {
            let end = (first + q).min(groups);
            critical += fixed;
            if first > 0 && end - first > 1 {
                critical += (prices.dispatch)(end - first);
            }
            critical += (first..end)
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
    // A cap measured at a working set bounds the bandwidth of every larger one.
    let knee = knee.min(constants.bandwidth_bytes);
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
    let mut slope = [0.0; 3];
    let working_sets = [ONLINE_WORKING_SETS, [1024, 8192, 65536]][inner];
    for (i, w) in working_sets.into_iter().enumerate() {
        let n = w / (2 * std::mem::size_of::<f64>());
        one[i] = measure(n, 1)?;
        let many_runs = (n / 2).min(128);
        let many = measure(n, many_runs)?;
        slope[i] = (many - one[i]) / (many_runs - 1) as f64;
    }
    fit_row(inner, empty, low, high, one, slope)
}

/// The row of one class from its samples: `empty`, `low` and `high` are
/// one-run walks of 0, 2 and 16 elements, `one[i]` a one-run walk of the
/// class's `i`-th working set and `slope[i]` its measured time per extra run.
///
/// The physical row has `a >= 0`, `r >= 0` and `e > 0`. A negative setup or
/// per-run term means that term is below the measurement's resolution (a
/// contiguous run costs no measurable time per run), not that the host
/// differs from the samples: it is fixed at zero, and each one-run total is
/// kept and attributed to the remaining terms, which is the least-squares
/// fit of those terms on the same samples. Only a row that is still
/// non-physical (`e <= 0`) or whose samples are not finite is rejected, and
/// the caller then keeps the next source (reference table, then the slowest
/// same-stride row).
fn fit_row(
    inner: usize,
    empty: f64,
    low: f64,
    high: f64,
    one: [f64; 3],
    slope: [f64; 3],
) -> Option<Row> {
    let working_sets = [ONLINE_WORKING_SETS, [1024, 8192, 65536]][inner];
    let finite = |value: f64| value.is_finite().then_some(value);
    let run: [(usize, f64); 3] = core::array::from_fn(|i| {
        (
            working_sets[i],
            finite(slope[i]).map_or(f64::NAN, |r| r.max(0.0)),
        )
    });
    let a = finite(low - empty - (high - low) * (2.0 / 14.0) - run[0].1)?.max(0.0);
    let element: [(usize, f64); 3] = core::array::from_fn(|i| {
        let w = working_sets[i];
        (w, (one[i] - empty - a - run[i].1) / (w / 16) as f64)
    });
    if !run.iter().all(|p| p.1.is_finite()) || !element.iter().all(|p| p.1.is_finite() && p.1 > 0.0)
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

/// Completes the measured anchors of `values` (indexed by width) into one
/// non-decreasing row over every width from 2: linear between anchors, the
/// table scaled to the nearest anchor outside them, then the running
/// maximum, so measured and tabulated widths never mix into a dip. Without
/// an anchor every width keeps the table.
fn complete_dispatch(values: &mut [Option<Duration>], table: impl Fn(usize) -> Duration) {
    let anchor = |(q, value): (usize, &Option<Duration>)| value.map(|value| (q, value));
    let Some((first, first_value)) = values.iter().enumerate().skip(2).find_map(anchor) else {
        return;
    };
    let (last, last_value) = values
        .iter()
        .enumerate()
        .skip(2)
        .rev()
        .find_map(anchor)
        .unwrap_or((first, first_value));
    let scaled = |q: usize, anchor: usize, value: Duration| {
        let reference = table(anchor).as_secs_f64();
        if reference > 0.0 {
            table(q).mul_f64(value.as_secs_f64() / reference)
        } else {
            value
        }
    };
    for (q, value) in values.iter_mut().enumerate().take(first).skip(2) {
        *value = Some(scaled(q, first, first_value));
    }
    let mut left = first;
    while let Some(offset) = values[left + 1..=last].iter().position(Option::is_some) {
        let right = left + 1 + offset;
        let (Some(low), Some(high)) = (values[left], values[right]) else {
            break;
        };
        let (low, high) = (low.as_secs_f64(), high.as_secs_f64());
        for (q, value) in values.iter_mut().enumerate().take(right).skip(left + 1) {
            let t = (q - left) as f64 / (right - left) as f64;
            *value = Some(Duration::from_secs_f64(low + t * (high - low)));
        }
        left = right;
    }
    for (q, value) in values.iter_mut().enumerate().skip(last + 1) {
        *value = Some(scaled(q, last, last_value));
    }
    let mut floor = Duration::ZERO;
    for value in values.iter_mut().skip(2).flatten() {
        floor = floor.max(*value);
        *value = floor;
    }
}

fn measure_dispatch(exec: &mut lanes::Exec<'_>, deadline: Instant, constants: &mut Constants) {
    // ponytail: replace this interim measurement with threadteam's
    // `Exec::calibrate_dispatch_costs` (tensor4all/threadteam-rs#92, on
    // threadteam main, not in the pinned release); delete it once the pinned
    // threadteam calibrates a running pool.
    // The chooser's candidate widths, the budget last; `complete_dispatch`
    // interpolates the widths between them, as threadteam's own row does.
    let budget = exec.budget();
    let ladder = CANDIDATES
        .into_iter()
        .filter(|&q| (2..budget).contains(&q))
        .chain([budget]);
    for q in ladder {
        let Some(slot) = constants.dispatch.get_mut(q) else {
            continue;
        };
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

/// The median cost of one empty guarded engine entry, the entry every
/// replay region takes (`host_pool::lane_region`), timed directly so that no
/// region is observed.
fn measure_entry(engine: &mut dyn Engine, deadline: Instant) -> Option<f64> {
    tenferro_cpu::with_cpu_engine(engine, |cpu| {
        cpu.with_exec(|_| ()).ok()?;
        let mut samples = [0.0; 5];
        for sample in &mut samples {
            if Instant::now() >= deadline {
                return None;
            }
            let start = Instant::now();
            for _ in 0..8 {
                cpu.with_exec(|_| ()).ok()?;
            }
            *sample = start.elapsed().as_secs_f64() * 1e9 / 8.0;
        }
        samples.sort_by(f64::total_cmp);
        Some(samples[2])
    })
    .flatten()
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
    // The fixture is the knee, or what first-touch and the whole ladder fit
    // in the deadline: a cap measured at a working set bounds every larger
    // one, so `phase_bandwidth` applies it from the fixture's size on. Below
    // the largest row working set a sample is too short to time.
    let pair = 2 * std::mem::size_of::<f64>();
    let fixture_bytes = memory_bytes.min(maximum_bytes.saturating_sub(p * pair));
    if fixture_bytes < ONLINE_WORKING_SETS[2] {
        return;
    }
    let elements = fixture_bytes / pair;
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
        if bandwidth_fixture_bytes(remaining_ns, dispatch, ns_per_byte) < fixture_bytes {
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
            let bytes = (n * q * pair) as f64;
            constants.bandwidth[q] = bytes / elapsed;
            constants.bandwidth_bytes = if constants.bandwidth_bytes.is_finite() {
                constants.bandwidth_bytes.max(bytes)
            } else {
                bytes
            };
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
                    let mut phase = compiled_jobs(&jobs);
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
                    |widths, _| {
                        widths.extend(
                            phases
                                .iter()
                                .map(|phase| direct_phase_width(phase, &constants, key)),
                        );
                        Ok(())
                    },
                )
                .unwrap();
                assert_eq!(memo.current().widths, expected);
            }
        }
        assert!(split && inline);
    }

    /// Purpose: budget and constants revisions rebuild every phase once;
    /// an identical key evaluates no model, and both changes alter a choice.
    #[test]
    fn memo_recomputes_only_when_budget_or_constants_change() {
        use std::cell::Cell;
        let phase = compiled_jobs(&vec![
            Job {
                counts: [Counts {
                    items: 1,
                    runs: 1,
                    elements: 1000000
                }; CLASSES],
                ..Job::default()
            };
            8
        ]);
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
                    |widths, _| {
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
                memo.current().widths,
                phases
                    .iter()
                    .map(|phase| direct_phase_width(phase, &constants, key))
                    .collect::<Vec<_>>()
            );
            if let Some(previous) = previous {
                assert_ne!(memo.current().widths, previous);
            }
            previous = Some(memo.current().widths.clone());
        }
    }

    /// Purpose: schedule, gate, policy, scalar, writer, fill and member inputs
    /// invalidate that structure's entry; unchanged inputs are the warm control.
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
                |widths, _| {
                    widths.push(1);
                    Ok(())
                },
            )
            .unwrap();
            memo.update(
                key,
                Arc::downgrade(id),
                core::iter::once(cost),
                |widths, _| {
                    widths.push(2);
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(memo.current().widths, [2]);
            memo.update(key, Arc::downgrade(id), core::iter::once(cost), |_, _| {
                panic!("warm key rebuilt")
            })
            .unwrap();
        }
    }

    /// Purpose: a hit promotes its structure before bounded LRU replacement;
    /// the untouched least-recent entry is the eviction control.
    #[test]
    fn memo_promotes_hits_before_evicting_the_lru_structure() {
        let key = memo_key(&Constants::reference(HostClass::Generic));
        let identities: Vec<_> = (0..=MAX_DECISIONS).map(|_| Arc::new(())).collect();
        let mut memo = WidthMemo::default();
        for (index, identity) in identities[..MAX_DECISIONS].iter().enumerate() {
            memo.update(
                key,
                Arc::downgrade(identity),
                core::iter::empty(),
                |widths, _| {
                    widths.push(index);
                    Ok(())
                },
            )
            .unwrap();
        }
        memo.update(
            key,
            Arc::downgrade(&identities[0]),
            core::iter::empty(),
            |_, _| panic!("live structure rebuilt"),
        )
        .unwrap();
        memo.update(
            key,
            Arc::downgrade(&identities[MAX_DECISIONS]),
            core::iter::empty(),
            |widths, _| {
                widths.push(MAX_DECISIONS);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(memo.entries.len(), MAX_DECISIONS);
        memo.update(
            key,
            Arc::downgrade(&identities[0]),
            core::iter::empty(),
            |_, _| panic!("promoted structure evicted"),
        )
        .unwrap();
        let rebuilt = Cell::new(false);
        memo.update(
            key,
            Arc::downgrade(&identities[1]),
            core::iter::empty(),
            |_, _| {
                rebuilt.set(true);
                Ok(())
            },
        )
        .unwrap();
        assert!(rebuilt.get());
    }

    /// Purpose: a dead structure is replaced before a live LRU structure.
    #[test]
    fn memo_reuses_dead_structure_before_live_lru() {
        let key = memo_key(&Constants::reference(HostClass::Generic));
        let mut identities: Vec<_> = (0..MAX_DECISIONS).map(|_| Some(Arc::new(()))).collect();
        let mut memo = WidthMemo::default();
        for identity in &identities {
            memo.update(
                key,
                Arc::downgrade(identity.as_ref().unwrap()),
                core::iter::empty(),
                |widths, _| {
                    widths.push(1);
                    Ok(())
                },
            )
            .unwrap();
        }
        identities[MAX_DECISIONS - 1] = None;
        let replacement = Arc::new(());
        memo.update(
            key,
            Arc::downgrade(&replacement),
            core::iter::empty(),
            |widths, _| {
                widths.push(1);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(memo.entries.len(), MAX_DECISIONS);
        memo.update(
            key,
            Arc::downgrade(identities[0].as_ref().unwrap()),
            core::iter::empty(),
            |_, _| panic!("live LRU evicted before dead structure"),
        )
        .unwrap();
    }

    /// The planned extremes of a job whose roles of one class are equal.
    fn job_roles(job: &Job) -> Planned {
        let mut roles = Planned::default();
        for (class, count) in job.planned.iter().enumerate() {
            if count.items != 0 {
                roles.role(
                    class,
                    crate::movement::Point {
                        runs: (count.runs / count.items) as usize,
                        elements: (count.elements / count.items) as usize,
                    },
                );
            }
        }
        roles
    }

    fn compiled_jobs(jobs: &[Job]) -> Summary {
        let mut phase = Summary::default();
        for job in jobs {
            for (count, other) in phase.counts.iter_mut().zip(job.counts) {
                *count = count.add(other).unwrap();
            }
            for (count, other) in phase.planned.iter_mut().zip(job.planned) {
                *count = count.add(other).unwrap();
            }
            phase.record_job(&job.counts).unwrap();
            *phase.planned_jobs.last_mut().unwrap() = job_roles(job);
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

    fn synthetic_choice(phase: &Summary, budget: usize, admit: bool) -> (usize, Vec<f64>) {
        let costs = [(0.0, 0.0, 1.0); CLASSES];
        let dispatch = |_| 1.0;
        let bandwidth = |_| f64::INFINITY;
        let admits = |_, _| admit;
        let prices = Prices {
            bytes: 8.0,
            dispatch: &dispatch,
            bandwidth: &bandwidth,
            admits: &admits,
        };
        let model = super::fixed(phase, &costs);
        let chosen = choose(
            budget,
            phase.prefix.len(),
            0.0,
            |q| model(q, budget, &prices),
            dispatch,
            bandwidth,
        );
        let times = (1..=budget.min(phase.prefix.len()))
            .map(|q| if q > 1 { 1.0 } else { 0.0 } + model(q, budget, &prices))
            .collect();
        (chosen, times)
    }

    /// Purpose: large admitted roles keep leaves with inner fan-out; withholding
    /// that credit is the serial-item control that exhausts the outer slots.
    #[test]
    fn large_planned_roles_choose_leaves_with_inner_fanout() {
        let phase = compiled_jobs(&[1, 4, 9, 4, 1].map(|items| uniform_job(items, 24, 0, true)));
        assert_eq!(
            synthetic_choice(&phase, 2, true),
            (1, vec![19.0 + 456.0 / 2.0, 1.0 + 336.0])
        );
        assert_eq!(
            synthetic_choice(&phase, 4, true),
            (
                1,
                vec![
                    19.0 + 456.0 / 4.0,
                    1.0 + 14.0 + 336.0 / 2.0,
                    1.0 + 312.0,
                    1.0 + 216.0
                ]
            )
        );
        assert_eq!(
            synthetic_choice(&phase, 2, false),
            (2, vec![456.0, 1.0 + 336.0])
        );
        assert_eq!(
            synthetic_choice(&phase, 4, false),
            (4, vec![456.0, 1.0 + 336.0, 1.0 + 312.0, 1.0 + 216.0])
        );
    }

    /// Purpose: distributed serial work still pays for outer splitting; dispatch
    /// dominating its total time is the inline control.
    #[test]
    fn dominant_serial_job_and_small_jobs_split() {
        let mut jobs = vec![uniform_job(1, 10, 1, false); 100];
        jobs[0] = uniform_job(1, 1000, 1, false);
        let phase = compiled_jobs(&jobs);
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
        let mut jobs = vec![uniform_job(1, 20, 0, true); 12];
        jobs[0] = uniform_job(1, 100, 0, true);
        let phase = compiled_jobs(&jobs);
        assert_eq!(
            synthetic_choice(&phase, 2, true),
            (1, vec![12.0 + 320.0 / 2.0, 1.0 + 200.0])
        );
        assert_eq!(
            synthetic_choice(&phase, 4, true),
            (
                3,
                vec![
                    12.0 + 320.0 / 4.0,
                    1.0 + 6.0 + 200.0 / 2.0,
                    1.0 + (4.0_f64 + 160.0 / 2.0).max(80.0),
                    1.0 + 140.0
                ]
            )
        );
        assert_eq!(
            synthetic_choice(&phase, 4, false),
            (4, vec![320.0, 1.0 + 200.0, 1.0 + 160.0, 1.0 + 140.0])
        );
    }

    /// Purpose: small gated roles use every outer slot; a single ownership
    /// job is the indivisible control.
    #[test]
    fn small_roles_choose_the_budget_without_inner_credit() {
        let phase = compiled_jobs(&[uniform_job(1, 24, 0, true); 8]);
        for budget in [2, 4] {
            assert_eq!(
                synthetic_choice(&phase, budget, false),
                (
                    budget,
                    [
                        8.0 * 24.0,
                        1.0 + 4.0 * 24.0,
                        1.0 + 3.0 * 24.0,
                        1.0 + 2.0 * 24.0
                    ][..budget]
                        .to_vec()
                )
            );
            assert_eq!(
                synthetic_choice(
                    &compiled_jobs(&[uniform_job(8, 24, 0, true)]),
                    budget,
                    false
                ),
                (1, vec![8.0 * 24.0])
            );
        }
    }

    /// Purpose: a leaf credits inner fan-out for the planned roles its slots
    /// admit, not only when the phase's smallest role is admitted: one large
    /// role among many small ones keeps a leaf of at least two slots when that
    /// is predicted faster than the budget's one-slot leaves. Admitting no
    /// role (what gating on the phase's smallest role gave here) is the
    /// control that strands the large role on one slot.
    #[test]
    fn a_dominant_admitted_role_among_small_roles_keeps_inner_fanout() {
        let mut jobs = vec![uniform_job(1, 2304, 1, true); 201];
        jobs[0] = uniform_job(1, 589_824, 1, true);
        let phase = compiled_jobs(&jobs);
        let costs = [(7.3, 1.48, 0.2228); CLASSES];
        let dispatch = |q: usize| 500.0 + 120.0 * q as f64;
        let bandwidth = |_| f64::INFINITY;
        for budget in [8, 16] {
            for (admits_large, fans_out) in [(true, true), (false, false)] {
                let admits = |slots: usize, elements: usize| {
                    admits_large && slots >= 2 && elements >= 100_000
                };
                let prices = Prices {
                    bytes: 8.0,
                    dispatch: &dispatch,
                    bandwidth: &bandwidth,
                    admits: &admits,
                };
                let model = super::fixed(&phase, &costs);
                let time = |q| if q > 1 { dispatch(q) } else { 0.0 } + model(q, budget, &prices);
                let chosen = choose(
                    budget,
                    jobs.len(),
                    0.0,
                    |q| model(q, budget, &prices),
                    dispatch,
                    bandwidth,
                );
                let dominant = Cell::new(0);
                leaf_max(0, jobs.len(), budget, chosen, 0, &|range, slots, _| {
                    if range.contains(&0) {
                        dominant.set(slots);
                    }
                    0.0
                });
                assert_eq!(dominant.get() >= 2, fans_out);
                if fans_out {
                    assert!(time(chosen) < time(budget));
                } else {
                    assert_eq!(chosen, budget);
                }
            }
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
            assert_eq!(
                prices.leaf(serial, job_roles(&serial), &costs, slots),
                210.0
            );
        }
        let roles = job_roles(&planned);
        assert!(prices.leaf(planned, roles, &costs, 4) < prices.leaf(planned, roles, &costs, 2));
    }

    /// Purpose: cumulative job components and the planned-extremes tree
    /// reproduce independent linear scans for every range and preserve S_q
    /// under the same actual leaf partition.
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
            let phase = compiled_jobs(&jobs);
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
                    let roles = jobs[range.clone()]
                        .iter()
                        .fold(Planned::default(), |roles, job| roles.join(job_roles(job)));
                    assert_eq!(planned_range(&phase.planned_jobs, range.clone()), roles);
                    let actual = job_range(&phase.prefix, range);
                    assert_eq!(actual, expected);
                    actual_sum.set(actual_sum.get() + prices.leaf(actual, roles, &costs, slots));
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

    /// Purpose: modeled ownership ranges and slot shares match actual Split
    /// leaves; flat job chunks are the uneven-boundary negative control.
    #[test]
    fn leaf_partition_matches_recursive_ownership_boundaries() {
        use core::ops::Range;
        struct Observe(usize);
        impl lanes::Work for Observe {
            type Job = (Range<usize>, usize);
            type Lane = Range<usize>;
            type Error = ();

            fn count(&self) -> usize {
                self.0
            }

            fn run(
                &self,
                _: usize,
                state: &mut Self::Job,
                lane: &mut Self::Lane,
                exec: lanes::Exec<'_>,
            ) -> Result<(), Self::Error> {
                *state = (lane.clone(), exec.budget());
                Ok(())
            }
        }

        for p in 1..=9 {
            let mut engine = tenferro_cpu::CpuEngine::new(
                tenferro_cpu::Pool::builder().threads(p).build().unwrap(),
            );
            for jobs in 1..=17 {
                for q in 1..=p {
                    let mut observed = vec![(0..0, 0); jobs];
                    crate::host_pool::lane_region(
                        &mut engine,
                        crate::host_pool::HostPoolSite::Replay,
                        |exec| {
                            lanes::run_native(
                                exec,
                                &Observe(jobs),
                                lanes::Split {
                                    threads: q,
                                    split: |left: &mut Range<usize>, cut: lanes::Cut| {
                                        let right = cut.job..left.end;
                                        left.end = cut.job;
                                        Ok(right)
                                    },
                                },
                                &mut observed,
                                core::slice::from_mut(&mut (0..jobs)),
                            )
                            .unwrap()
                            .unwrap();
                        },
                    )
                    .unwrap();
                    observed.dedup();
                    let modeled = std::cell::RefCell::new(Vec::new());
                    leaf_max(0, jobs, p, q, 0, &|range, slots, _| {
                        modeled.borrow_mut().push((range, slots));
                        0.0
                    });
                    assert_eq!(*modeled.borrow(), observed, "jobs={jobs}, q={q}, p={p}");
                    if (jobs, q, p) == (7, 3, 8) {
                        assert_ne!(observed, [(0..3, 3), (3..6, 3), (6..7, 2)]);
                    }
                }
            }
        }
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
        let phase = compiled_jobs(&[uniform_job(1, 3, 1, false); 4]);
        let costs = [(1.0, 1.0, 1.0); CLASSES];
        let prices = Prices {
            bytes: 8.0,
            dispatch: &|_| 0.0,
            bandwidth: &|_| f64::INFINITY,
            admits: &|_, _| false,
        };
        assert_eq!(
            prefix_phase(&phase.prefix, 0, 4, &[], &costs, Some(8))(2, 4, &prices),
            16.0
        );
        assert_eq!(
            prefix_phase(&phase.prefix, 0, 4, &[], &costs, None)(2, 4, &prices),
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
        let phase = compiled_jobs(&[uniform_job(1, 2, 1, false)]);
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
    /// cost and serial batch barriers, a later chunk of two groups paying its
    /// own fork; movement alone is the inline control.
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
        let phase = group_phase(schedule, &costs, |group| dense[group], 0.0);
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
            [210.0, 204.0 + 20.0, 105.0, 102.0]
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
                |q| group_phase(schedule, &costs, |_| 0.0, 0.0)(q, 4, &prices),
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

    /// Purpose: a chunk's fixed cost is paid once per chunk and a later
    /// chunk's fork once per forking chunk, so a large fixed cost chooses few
    /// chunks; a zero fixed cost is the control that keeps one group per
    /// chunk when each fork costs more than the work it splits.
    #[test]
    fn group_chunks_pay_their_fixed_cost_and_forks_per_chunk() {
        use crate::TreeTransformBlockSpec;
        let structure = BlockStructure::packed_column_major(1, vec![vec![1]; 16]).unwrap();
        let transform = TreeTransformStructure::compile_structures(
            &structure,
            &structure,
            &(0..8)
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
        let costs = [(0.0, 0.0, 0.0); CLASSES];
        let dispatch = |q| 10.0 * q as f64;
        let prices = Prices {
            bytes: 8.0,
            dispatch: &dispatch,
            bandwidth: &|_| f64::INFINITY,
            admits: &|_, _| false,
        };
        let free = group_phase(schedule, &costs, |_| 1.0, 0.0);
        assert_eq!(free(2, 8, &prices), 4.0 + 3.0 * 20.0);
        let paid = group_phase(schedule, &costs, |_| 1.0, 100.0);
        assert_eq!(paid(2, 8, &prices), 4.0 * 101.0 + 3.0 * 20.0);
        for (fixed, expected) in [(0.0, 1), (100.0, 8)] {
            let phase = group_phase(schedule, &costs, |_| 1.0, fixed);
            assert_eq!(
                choose(
                    8,
                    8,
                    0.0,
                    |q| phase(q, 8, &prices),
                    dispatch,
                    |_| f64::INFINITY
                ),
                expected
            );
        }
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
        let traffic = 128.0 * 16.0;
        let mut constants = Constants::reference(HostClass::Generic);
        // Synthetic cache and memory rows: e=2 through 2048 bytes, then e=8.
        constants.rows = [Row {
            a: 0.0,
            r: Curve {
                points: [(1, 0.0), (2, 0.0), (4, 0.0)],
                online: None,
            },
            e: Curve {
                points: core::array::from_fn(|i| (64 << i, if i < 6 { 2.0 } else { 8.0 })),
                online: None,
            },
        }; CLASSES];
        constants.memory_knees = constants.rows.map(|row| row.e.memory_knee());
        constants.bandwidth = vec![16.0; 5];
        let elements = [128.0, 0.0, 0.0, 0.0];
        let bandwidth = phase_bandwidth(traffic, traffic, elements, &constants);
        assert_eq!(traffic / bandwidth(4), 128.0 * 2.0 / 4.0);
        assert!(traffic / constants.bandwidth[4] > traffic / bandwidth(4));
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

    /// One-run and many-run samples of the contiguous class with setup 7 ns,
    /// element costs shaped like a measured Xeon row (2.6-4x the reference)
    /// and per-run slopes `slope`, as `measure_row` takes them.
    fn contiguous_samples(element: [f64; 3], slope: [f64; 3]) -> (f64, f64, f64, [f64; 3]) {
        let (empty, a, small) = (100.0, 7.0, 0.2);
        let r0 = slope[0].max(0.0);
        let low = empty + a + r0 + 2.0 * small;
        let high = empty + a + r0 + 16.0 * small;
        let one = core::array::from_fn(|i| {
            empty + a + slope[i].max(0.0) + element[i] * (ONLINE_WORKING_SETS[i] / 16) as f64
        });
        (empty, low, high, one)
    }

    /// Purpose: a per-run slope below resolution (negative) fits r = 0 and
    /// keeps the measured setup and element costs, not the reference row's;
    /// the old all-terms-nonnegative rejection is the negative control that
    /// discarded this row. A non-positive element cost still rejects.
    #[test]
    fn negative_run_slopes_fit_zero_and_keep_measured_elements() {
        let close = |actual: f64, expected: f64| assert!((actual - expected).abs() < 1e-9);
        let element = [0.229, 0.424, 0.619];
        let slope = [1.9, -9.0, -42.9];
        let (empty, low, high, one) = contiguous_samples(element, slope);
        // The previous fit accepted a row only if every unconstrained term
        // was nonnegative, so these samples fell back to the reference row.
        assert!(!slope.iter().all(|&r| r >= 0.0));
        let row = fit_row(0, empty, low, high, one, slope).unwrap();
        close(row.a, 7.0);
        let run = row.r.online.unwrap();
        let fitted = row.e.online.unwrap();
        for (i, &expected) in element.iter().enumerate() {
            let bytes = ONLINE_WORKING_SETS[i] as f64;
            close(run[i].1, slope[i].max(0.0));
            close(fitted[i].1, expected);
            close(row.e.at(bytes), expected);
            assert!(row.e.at(bytes) > 2.0 * CI_ROWS[0].e.at(bytes));
        }

        let (empty, low, high, one) = contiguous_samples(element, [1.9, 3.0, 4.0]);
        // A setup below resolution (here -3 ns) fits a = 0.
        let negative_setup =
            fit_row(0, empty, low - 10.0, high - 10.0, one, [1.9, 3.0, 4.0]).unwrap();
        assert_eq!(negative_setup.a, 0.0);
        assert!(negative_setup.e.online.unwrap().iter().all(|p| p.1 > 0.0));
        assert!(fit_row(0, empty, low, high, one, [1.9, f64::NAN, 4.0]).is_none());

        let (empty, low, high, one) = contiguous_samples([0.229, -0.01, 0.619], slope);
        assert!(fit_row(0, empty, low, high, one, slope).is_none());
    }

    /// Purpose: measured dispatch anchors become one non-decreasing row:
    /// linear between anchors, the table scaled to the nearest anchor
    /// outside them, and a lower later anchor lifted to the running maximum;
    /// a row without anchors is the control that keeps the table everywhere.
    #[test]
    fn dispatch_anchors_complete_a_monotone_row() {
        let table = |q: usize| Duration::from_nanos(100 * q as u64);
        let ns = |value: Option<Duration>| value.unwrap().as_secs_f64() * 1e9;
        let close = |actual: f64, expected: f64| assert!((actual - expected).abs() < 1e-6);
        let mut values: Vec<Option<Duration>> = vec![None; 9];
        for (q, measured) in [(3, 600), (5, 1000), (6, 800)] {
            values[q] = Some(Duration::from_nanos(measured));
        }
        complete_dispatch(&mut values, table);
        assert!(values[..2].iter().all(Option::is_none));
        let expected = [
            200.0 * 600.0 / 300.0,
            600.0,
            800.0,
            1000.0,
            1000.0,
            1000.0,
            800.0 * 800.0 / 600.0,
        ];
        for (q, expected) in (2..9).zip(expected) {
            close(ns(values[q]), expected);
        }
        let mut unmeasured: Vec<Option<Duration>> = vec![None; 9];
        complete_dispatch(&mut unmeasured, table);
        assert!(unmeasured.iter().all(Option::is_none));
    }

    /// Purpose: a bandwidth ladder measured on a fixture smaller than the
    /// knee caps every phase at least that large; an unmeasured fixture size
    /// is the control that applies the cap only from the knee.
    #[test]
    fn bandwidth_cap_applies_from_the_measured_fixture_size() {
        let mut constants = Constants::reference(HostClass::Generic);
        constants.bandwidth = vec![f64::INFINITY, f64::INFINITY, 60.0, 80.0, 80.0];
        let working_set = (16 << 20) as f64;
        let elements = [working_set / 16.0, 0.0, 0.0, 0.0];
        assert!(working_set < constants.memory_knees[0]);
        constants.bandwidth_bytes = (8 << 20) as f64;
        assert_eq!(
            phase_bandwidth(working_set, working_set, elements, &constants)(4),
            80.0
        );
        assert!(phase_bandwidth((4 << 20) as f64, working_set, elements, &constants)(4) > 80.0);
        constants.bandwidth_bytes = f64::INFINITY;
        assert!(phase_bandwidth(working_set, working_set, elements, &constants)(4) > 80.0);
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
