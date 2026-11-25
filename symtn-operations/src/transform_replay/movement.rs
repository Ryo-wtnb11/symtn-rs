//! Replay-owned movement calibration and fixed-form phase pricing.
use super::*;
use crate::transform_structure::{MovementCounts, MovementSummary};
use std::fmt::Write as _;
use std::time::{Duration, Instant};

const WORKING_SETS: [usize; 3] = [64 * 1024, 1024 * 1024, 4 * 1024 * 1024];

#[derive(Clone, Copy)]
struct Curve<const N: usize> {
    points: [(usize, f64); N],
}

impl<const N: usize> Curve<N> {
    fn at(&self, bytes: f64) -> f64 {
        let points = &self.points;
        for pair in points.windows(2) {
            if bytes <= pair[1].0 as f64 {
                let t = ((bytes.max(pair[0].0 as f64) / pair[0].0 as f64).ln()
                    / (pair[1].0 as f64 / pair[0].0 as f64).ln())
                .clamp(0.0, 1.0);
                return pair[0].1 + t * (pair[1].1 - pair[0].1);
            }
        }
        points[N - 1].1
    }

    fn valid(&self) -> bool {
        self.points
            .iter()
            .all(|&(w, t)| w > 0 && t.is_finite() && t >= 0.0)
            && self
                .points
                .windows(2)
                .all(|pair| pair[0].0 < pair[1].0 && pair[0].1 * 0.9 <= pair[1].1)
    }

    fn monotone(&mut self) {
        for index in 1..N {
            self.points[index].1 = self.points[index].1.max(self.points[index - 1].1);
        }
    }
}

#[derive(Clone, Copy)]
struct Rates {
    setup: f64,
    runs: [Curve<2>; 4],
    elements: [Curve<3>; 4],
}

impl Rates {
    fn cost(&self, counts: MovementCounts, bytes: f64) -> f64 {
        self.setup * counts.roles.iter().sum::<usize>() as f64
            + (0..4)
                .map(|k| {
                    self.runs[k].at(bytes) * counts.runs[k] as f64
                        + self.elements[k].at(bytes) * counts.elements[k] as f64
                })
                .sum::<f64>()
    }

    fn valid(&self) -> bool {
        self.setup.is_finite()
            && self.setup >= 0.0
            && self.runs.iter().all(Curve::valid)
            && self.elements.iter().all(Curve::valid)
    }
}

/// One fixed-size payload type is used for the engine's movement slot.
/// Copy, scaled copy and zero-fill are measured; unavailable maps use base.
pub(super) struct Calibration {
    copy: Rates,
    fill: Rates,
}

fn dispatch_at(dispatch: &[(usize, Duration)], width: usize) -> f64 {
    if width <= 1 {
        return 0.0;
    }
    for pair in dispatch.windows(2) {
        if width <= pair[1].0 {
            let t = (width - pair[0].0) as f64 / (pair[1].0 - pair[0].0) as f64;
            return pair[0].1.as_secs_f64() * 1e9 * (1.0 - t) + pair[1].1.as_secs_f64() * 1e9 * t;
        }
    }
    dispatch.last().map_or(0.0, |p| p.1.as_secs_f64() * 1e9)
}

fn inline_time(
    rates: &Rates,
    phase: MovementSummary,
    bytes: f64,
    dispatch: f64,
    budget: usize,
    admits_largest: bool,
) -> f64 {
    let sum = rates.cost(phase.total, bytes);
    if !admits_largest {
        return sum;
    }
    let item = phase.largest;
    let largest = (rates.setup
        + rates.runs[item.kind].at(bytes) * item.runs as f64
        + rates.elements[item.kind].at(bytes) * item.elements as f64)
        .min(sum);
    sum - largest + largest.min(dispatch + largest / budget as f64)
}

/// Counts are read in O(1), without item traversal or allocation. Between
/// dispatch/curve knots, each split branch is convex in q; discrete bisection
/// finds its integer minimum, including max-branch crossings and smaller ties.
fn choose(
    rates: &Rates,
    phase: MovementSummary,
    dispatch: &[(usize, Duration)],
    budget: usize,
    admits_largest: bool,
) -> usize {
    if budget <= 1
        || phase.total.roles == [0; 4]
        || !phase.exact_working_set
        || !phase.supported_span
    {
        return 1;
    }
    let bytes = phase.working_set_elements as f64 * std::mem::size_of::<f64>() as f64;
    let mut best = 1;
    let mut time = inline_time(
        rates,
        phase,
        bytes,
        dispatch_at(dispatch, budget),
        budget,
        admits_largest,
    );
    let mut cuts = [2usize; usize::BITS as usize + 3 + 2 * 4 * (2 + 3)];
    let mut count = 0;
    let mut add = |q: usize| {
        cuts[count] = q.clamp(2, budget);
        count += 1;
    };
    add(2);
    add(budget);
    for &(q, _) in dispatch {
        add(q);
    }
    for k in 0..4 {
        for w in rates.runs[k]
            .points
            .iter()
            .map(|p| p.0)
            .chain(rates.elements[k].points.iter().map(|p| p.0))
        {
            let q = bytes / w as f64;
            add(q.floor() as usize);
            add(q.ceil() as usize);
        }
    }
    let cuts = &mut cuts[..count];
    cuts.sort_unstable();
    let mut dispatch_index = 0;
    for interval in cuts.windows(2) {
        let (mut lo, mut hi) = (interval[0], interval[1]);
        while dispatch_index + 2 < dispatch.len() && dispatch[dispatch_index + 1].0 <= lo {
            dispatch_index += 1;
        }
        let (left, right) = (dispatch[dispatch_index], dispatch[dispatch_index + 1]);
        let score = |q: usize| {
            let t = (q - left.0) as f64 / (right.0 - left.0) as f64;
            let d = left.1.as_secs_f64() * 1e9 * (1.0 - t) + right.1.as_secs_f64() * 1e9 * t;
            let w = bytes / q as f64;
            d + rates
                .cost(phase.max, w)
                .max(rates.cost(phase.total, w) / q as f64)
        };
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if score(mid) <= score(mid + 1) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        let split = score(lo);
        if split < time || (split == time && lo < best) {
            best = lo;
            time = split;
        }
    }
    best
}

pub(super) fn widths(
    calibration: &Calibration,
    dispatch: &[(usize, Duration)],
    singles: MovementSummary,
    inactive: MovementSummary,
    budget: usize,
    admits_largest_single: bool,
) -> [usize; 2] {
    [
        choose(
            &calibration.copy,
            singles,
            dispatch,
            budget,
            admits_largest_single,
        ),
        choose(&calibration.fill, inactive, dispatch, budget, false),
    ]
}

/// The raw replay already holds the CPU entry here. Re-entering it would
/// deadlock, so calibrate its actual inline traversal/writer on the serial token.
pub(super) fn calibrate(
    exec: &mut symtn_dense::lanes::Exec<'_>,
    deadline: Instant,
) -> Option<Calibration> {
    calibrate_report(exec, deadline, None)
}

fn calibrate_report(
    exec: &mut symtn_dense::lanes::Exec<'_>,
    deadline: Instant,
    mut report: Option<&mut String>,
) -> Option<Calibration> {
    if Instant::now() >= deadline {
        return None;
    }
    // One pair covers both kinds and all run counts, including source stride padding.
    let count = WORKING_SETS[2] / std::mem::size_of::<f64>();
    let source = vec![1.0; count];
    let mut destination = Vec::with_capacity(count);
    let timer_jitter = (0..8)
        .map(|_| Instant::now().elapsed().as_secs_f64() * 1e9)
        .fold(0.0, f64::max);
    let mut copy = measure(
        exec,
        deadline,
        false,
        &source,
        &mut destination,
        timer_jitter,
        &mut report,
    )?;
    measure_scaled(
        exec,
        deadline,
        &mut copy,
        &source,
        &mut destination,
        timer_jitter,
        &mut report,
    )?;
    let fill = measure(
        exec,
        deadline,
        true,
        &source,
        &mut destination,
        timer_jitter,
        &mut report,
    )?;
    (Instant::now() < deadline).then_some(Calibration { copy, fill })
}

/// Runs the production online attempt on this engine and reports raw fit evidence.
/// Intended only for the calibration example; it obeys admission, policy and retention.
#[cfg(any(test, feature = "testing"))]
#[doc(hidden)]
pub fn online_movement_calibration(engine: &mut dyn Engine) -> Result<String, OperationError> {
    let start = Instant::now();
    let mut evidence = String::new();
    let mut attempted = false;
    let accepted = symtn_dense::lanes::movement_region(
        engine,
        true,
        |exec, deadline| {
            attempted = true;
            calibrate_report(exec, deadline, Some(&mut evidence))
        },
        |_, payload| payload.is_some(),
    )
    .map_err(OperationError::Dense)?;
    Ok(format!(
        "online outcome={} elapsed_us={} payload_attempted={attempted}\n{evidence}",
        if accepted { "Some" } else { "None" },
        start.elapsed().as_micros(),
    ))
}

struct Fixture {
    transform: TreeTransformStructure<f64>,
    source_len: usize,
    count: usize,
}

impl Fixture {
    fn new(
        elements: usize,
        runs: usize,
        kind: usize,
        fill: bool,
        scale: f64,
    ) -> Result<Self, OperationError> {
        use symtn_core::BlockSpec;
        let inner = elements / runs;
        let step = kind + 1;
        let shape = vec![inner, runs];
        let dst_step = 1;
        let dst_strides = vec![dst_step, dst_step * inner + usize::from(runs > 1)];
        let src_strides = vec![step, step * inner + usize::from(runs > 1)];
        let dst = BlockStructure::from_blocks_with_rank(
            2,
            vec![BlockSpec::new(shape.clone(), dst_strides, 0)?],
        )?;
        let src = if fill {
            BlockStructure::from_blocks_with_rank(2, vec![])?
        } else {
            BlockStructure::from_blocks_with_rank(2, vec![BlockSpec::new(shape, src_strides, 0)?])?
        };
        let specs = if fill {
            vec![]
        } else {
            vec![crate::TreeTransformBlockSpec::single(0, 0, scale)]
        };
        let transform = TreeTransformStructure::compile_structures(&dst, &src, &specs)?;
        if elements > 0 {
            let phase = if fill {
                transform.parallel_schedule().inactive_movement
            } else {
                transform.parallel_schedule().single_movement
            };
            let expected_kind = if fill { 1 } else { kind };
            if phase.total.elements[expected_kind] != elements
                || phase.total.runs[expected_kind] != runs
            {
                return Err(OperationError::InvalidArgument {
                    message: "calibration traversal changed during normalization",
                });
            }
        }
        let count = dst.required_len()?;
        Ok(Self {
            transform,
            source_len: src.required_len()?,
            count,
        })
    }

    fn sample(
        &self,
        exec: &mut symtn_dense::lanes::Exec<'_>,
        deadline: Instant,
        repetitions: usize,
        source: &[f64],
        destination: &mut Vec<f64>,
    ) -> Option<f64> {
        if Instant::now() >= deadline {
            return None;
        }
        let task = self.transform.task_view().ok()?;
        let writer = OwnedWriter {
            task: ReplayParts::new(task),
            alpha: 1.0,
        };
        let mut gate = LeafGate::new(task.layouts().max_planned_elements());
        let mut index = [0; 2];
        let mut zeros = Vec::new();
        let source = source.get(..self.source_len)?;
        let destination = destination.spare_capacity_mut().get_mut(..self.count)?;
        let start = Instant::now();
        for _ in 0..repetitions {
            inline_direct(
                exec.reborrow()
                    .limit(core::num::NonZeroUsize::MIN)
                    .into_token(),
                self.transform.parallel_schedule(),
                true,
                InlineSingles::Schedule,
                destination,
                source,
                &writer,
                &mut (),
                &mut zeros,
                &mut index,
                &mut gate,
            )
            .ok()?;
            std::hint::black_box(&*destination);
        }
        let ns = start.elapsed().as_secs_f64() * 1e9 / repetitions as f64;
        (Instant::now() < deadline).then_some(ns)
    }
}

#[derive(Clone, Copy)]
struct Measurement {
    ns: f64,
    jitter: f64,
}

fn elements_at(working_set: usize, fill: bool) -> usize {
    working_set / if fill { 8 } else { 16 }
}

#[allow(clippy::too_many_arguments)]
fn measured(
    exec: &mut symtn_dense::lanes::Exec<'_>,
    deadline: Instant,
    elements: usize,
    runs: usize,
    kind: usize,
    fill: bool,
    scale: f64,
    source: &[f64],
    destination: &mut Vec<f64>,
    report: &mut Option<&mut String>,
) -> Option<Measurement> {
    if Instant::now() >= deadline {
        return None;
    }
    let fixture = Fixture::new(elements, runs, kind, fill, scale).ok()?;
    let working_set = elements * if fill { 8 } else { 16 };
    let repetitions = if elements <= 16 {
        128
    } else if working_set < WORKING_SETS[1] {
        4
    } else {
        1
    };
    if working_set < WORKING_SETS[1] {
        fixture.sample(exec, deadline, 1, source, destination)?;
    }
    let mut samples = [0.0; 3];
    // ponytail: two large samples bound traffic; offline medians establish precision.
    let sample_count = if working_set >= WORKING_SETS[1] {
        2
    } else {
        samples.len()
    };
    for sample in &mut samples[..sample_count] {
        *sample = fixture.sample(exec, deadline, repetitions, source, destination)?;
    }
    if let Some(report) = report.as_deref_mut() {
        let _ = writeln!(
            report,
            "{} kind={kind} n={elements} runs={runs} repetitions={repetitions} samples_ns={:?}",
            if fill {
                "fill"
            } else if scale == 1.0 {
                "copy"
            } else {
                "scaled"
            },
            &samples[..sample_count]
        );
    }
    samples[..sample_count].sort_by(f64::total_cmp);
    Some(Measurement {
        // The lower of two large samples excludes their shared buffer's first stores.
        ns: samples[(sample_count - 1) / 2],
        jitter: samples[sample_count - 1] - samples[0],
    })
}

#[allow(clippy::too_many_arguments)]
fn measure(
    exec: &mut symtn_dense::lanes::Exec<'_>,
    deadline: Instant,
    fill: bool,
    source: &[f64],
    destination: &mut Vec<f64>,
    timer_jitter: f64,
    report: &mut Option<&mut String>,
) -> Option<Rates> {
    let mut sample = |elements, runs, kind, report: &mut Option<&mut String>| {
        measured(
            exec,
            deadline,
            elements,
            runs,
            kind,
            fill,
            1.0,
            source,
            destination,
            report,
        )
    };
    let empty = sample(0, 1, 0, report)?.ns;
    let mut run_points = [[(0, 0.0); 2]; 4];
    let mut one_run = [[0.0; 3]; 4];
    for (k, points) in run_points
        .iter_mut()
        .enumerate()
        .take(2)
        .skip(usize::from(fill))
    {
        let n = elements_at(WORKING_SETS[0], fill);
        let one = sample(n, 1, k, report)?;
        let many_runs = n / 2;
        let many = sample(n, many_runs, k, report)?;
        let signal = many.ns - one.ns;
        let jitter = timer_jitter.max(one.jitter + many.jitter);
        if let Some(report) = report.as_deref_mut() {
            let _ = writeln!(report, "{} kind={k} r_signal_ns={signal} jitter_ns={jitter} raw_r_ns_per_run={} accepted={}", if fill { "fill" } else { "copy" }, signal / (many_runs - 1) as f64, signal.is_finite() && signal >= 20.0 * jitter);
        }
        if !signal.is_finite() || signal < 20.0 * jitter {
            return None;
        }
        let per_run = signal / (many_runs - 1) as f64;
        // ponytail: online r is cache-resident only; offline paired ladders resolve W dependence.
        *points = [(WORKING_SETS[0], per_run), (WORKING_SETS[2], per_run)];
        one_run[k][0] = one.ns;
        for (j, &w) in WORKING_SETS.iter().enumerate().skip(1) {
            one_run[k][j] = sample(elements_at(w, fill), 1, k, report)?.ns;
        }
    }
    if fill {
        run_points[0] = run_points[1];
    }
    run_points[2] = run_points[0];
    run_points[3] = run_points[1];
    let runs = run_points.map(|points| Curve { points });
    let low = sample(2, 1, 0, report)?.ns;
    let high = sample(16, 1, 0, report)?.ns;
    let slope = (high - low) / 14.0;
    let setup = low - empty - slope * 2.0 - runs[0].points[0].1;
    let mut element_points = [[(0, 0.0); 3]; 4];
    for (k, points) in element_points
        .iter_mut()
        .enumerate()
        .take(2)
        .skip(usize::from(fill))
    {
        for (j, w) in WORKING_SETS.into_iter().enumerate() {
            let n = elements_at(w, fill);
            let time = one_run[k][j];
            points[j] = (w, (time - empty - setup - runs[k].at(w as f64)) / n as f64);
        }
    }
    if fill {
        element_points[0] = element_points[1];
    }
    element_points[2] = element_points[0];
    element_points[3] = element_points[1];
    let mut rates = Rates {
        setup,
        runs,
        elements: element_points.map(|points| Curve { points }),
    };
    if let Some(report) = report.as_deref_mut() {
        let _ = writeln!(
            report,
            "{} raw_a_ns={} raw_r_ns_per_run={:?} raw_e_ns_per_element={:?} valid={}",
            if fill { "fill" } else { "copy" },
            rates.setup,
            run_points,
            element_points,
            rates.valid()
        );
    }
    if !rates.valid() {
        return None;
    }
    // Accepted <=10% noise is enveloped so the split search retains convex intervals.
    for curve in &mut rates.elements {
        curve.monotone();
    }
    if let Some(report) = report.as_deref_mut() {
        let fitted = rates.elements.map(|curve| curve.points);
        let _ = writeln!(
            report,
            "{} fitted_e_ns_per_element={fitted:?}",
            if fill { "fill" } else { "copy" }
        );
    }
    Some(rates)
}

/// ponytail: one cache-resident scale point supplies ratios for the copy curve;
/// a second full ladder needs separate offline validation and a larger budget.
#[allow(clippy::too_many_arguments)]
fn measure_scaled(
    exec: &mut symtn_dense::lanes::Exec<'_>,
    deadline: Instant,
    rates: &mut Rates,
    source: &[f64],
    destination: &mut Vec<f64>,
    timer_jitter: f64,
    report: &mut Option<&mut String>,
) -> Option<()> {
    let mut sample = |elements, runs, kind, report: &mut Option<&mut String>| {
        measured(
            exec,
            deadline,
            elements,
            runs,
            kind,
            false,
            -0.75,
            source,
            destination,
            report,
        )
    };
    let empty = sample(0, 1, 0, report)?.ns;
    let w = WORKING_SETS[0];
    let n = elements_at(w, false);
    for kind in 0..2 {
        let one = sample(n, 1, kind, report)?;
        let many_runs = n / 2;
        let many = sample(n, many_runs, kind, report)?;
        let signal = many.ns - one.ns;
        let jitter = timer_jitter.max(one.jitter + many.jitter);
        if let Some(report) = report.as_deref_mut() {
            let _ = writeln!(
                report,
                "scaled kind={kind} r_signal_ns={signal} jitter_ns={jitter} accepted={}",
                signal.is_finite() && signal >= 20.0 * jitter
            );
        }
        if !signal.is_finite() || signal < 20.0 * jitter {
            return None;
        }
        let run = signal / (many_runs - 1) as f64;
        let element = (one.ns - empty - rates.setup - run) / n as f64;
        let r_ratio = run / rates.runs[kind].at(w as f64);
        let e_ratio = element / rates.elements[kind].at(w as f64);
        if let Some(report) = report.as_deref_mut() {
            let _ = writeln!(report,
                "scaled kind={kind} W={w} scale=-0.75 r_ns_per_run={run} e_ns_per_element={element} r_ratio={r_ratio} e_ratio={e_ratio} signal_ns={signal} jitter_ns={jitter}");
        }
        if !r_ratio.is_finite() || r_ratio < 0.0 || !e_ratio.is_finite() || e_ratio < 0.0 {
            return None;
        }
        rates.runs[kind + 2] = Curve {
            points: rates.runs[kind].points.map(|(w, r)| (w, r * r_ratio)),
        };
        rates.elements[kind + 2] = Curve {
            points: rates.elements[kind].points.map(|(w, e)| (w, e * e_ratio)),
        };
    }
    rates.valid().then_some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(super) fn measured_rates() -> Rates {
        Rates {
            setup: 9.0,
            runs: [Curve {
                points: [(65536, 1.3), (16777216, 1.6)],
            }; 4],
            elements: [
                Curve {
                    points: [(65536, 0.07), (8388608, 0.21), (16777216, 0.22)],
                },
                Curve {
                    points: [(65536, 0.23), (8388608, 0.27), (16777216, 0.43)],
                },
                Curve {
                    points: [(65536, 0.07), (8388608, 0.21), (16777216, 0.22)],
                },
                Curve {
                    points: [(65536, 0.23), (8388608, 0.27), (16777216, 0.43)],
                },
            ],
        }
    }

    fn summary(
        big: usize,
        small: usize,
        small_count: usize,
        runs: usize,
        kind: usize,
    ) -> MovementSummary {
        let mut total = MovementCounts::default();
        total.roles[kind] = small_count + 1;
        total.runs[kind] = runs * (small_count + 1);
        total.elements[kind] = big + small_count * small;
        let mut max = MovementCounts::default();
        max.roles[kind] = 1;
        max.runs[kind] = runs;
        max.elements[kind] = big;
        MovementSummary {
            total,
            max,
            largest: crate::transform_structure::MovementItem {
                runs,
                elements: big,
                kind,
                planned: true,
            },
            working_set_elements: total.elements[kind] * 2,
            exact_working_set: true,
            supported_span: true,
        }
    }

    pub(super) fn ladder(anchors: &[(usize, f64)]) -> Vec<(usize, Duration)> {
        anchors
            .iter()
            .map(|&(q, ns)| (q, Duration::from_secs_f64(ns * 1e-9)))
            .collect()
    }

    /// A dominant item splits with measured item costs and rising dispatch;
    /// free flat dispatch is the control where whole-token inline wins.
    #[test]
    fn dominant_phase_splits_with_measured_costs() {
        let rates = measured_rates();
        let phase = summary(589824, 2304, 99, 1, 0);
        let dispatch = ladder(&[
            (1, 0.0),
            (2, 647.5),
            (4, 894.0),
            (8, 1429.0),
            (16, 2435.0),
            (32, 8927.5),
            (64, 30267.5),
            (96, 44956.5),
        ]);
        assert!(choose(&rates, phase, &dispatch, 96, true) > 1);
        let free = ladder(&[(1, 0.0), (96, 0.0)]);
        assert_eq!(choose(&rates, phase, &free, 96, true), 1);
    }

    /// Scaled dominant Singles use their own measured curves; expensive dispatch
    /// is the negative control retaining the whole-token inline route.
    #[test]
    fn scaled_dominant_phase_splits() {
        let rates = measured_rates();
        let phase = summary(589824, 2304, 99, 1, 0).scaled();
        let dispatch = ladder(&[
            (1, 0.0),
            (2, 647.5),
            (4, 894.0),
            (8, 1429.0),
            (16, 2435.0),
            (32, 8927.5),
            (64, 30267.5),
            (96, 44956.5),
        ]);
        assert!(choose(&rates, phase, &dispatch, 96, true) > 1);
        assert_eq!(
            choose(&rates, phase, &ladder(&[(1, 0.0), (96, 1e12)]), 96, true),
            1
        );
    }

    /// Mixed Singles add one cost from each map class, while removing scale
    /// applications is the negative control returning the copy-only price.
    #[test]
    fn mixed_singles_sum_copy_and_scale_costs() {
        let mut rates = measured_rates();
        rates.setup = 1.0;
        rates.runs = [Curve {
            points: [(1, 2.0), (2, 2.0)],
        }; 4];
        rates.elements = [Curve {
            points: [(1, 3.0), (2, 3.0), (3, 3.0)],
        }; 4];
        rates.runs[2] = Curve {
            points: [(1, 5.0), (2, 5.0)],
        };
        rates.elements[2] = Curve {
            points: [(1, 7.0), (2, 7.0), (3, 7.0)],
        };
        let mixed = MovementCounts {
            roles: [1, 0, 1, 0],
            runs: [1, 0, 2, 0],
            elements: [4, 0, 6, 0],
        };
        assert_eq!(rates.cost(mixed, 1.0), 2.0 + 2.0 + 12.0 + 10.0 + 42.0);
        let copy = MovementCounts {
            roles: [2, 0, 0, 0],
            runs: [3, 0, 0, 0],
            elements: [10, 0, 0, 0],
        };
        assert_eq!(rates.cost(copy, 1.0), 2.0 + 6.0 + 30.0);
    }

    /// The full machine-word dispatch ladder fits the stack cut buffer;
    /// a two-width budget is the control with the same free-cost inline tie.
    #[test]
    fn all_dispatch_anchors_fit_with_scaled_curve_knots() {
        let rates = Rates {
            setup: 0.0,
            runs: [Curve {
                points: [(1, 0.0), (2, 0.0)],
            }; 4],
            elements: [Curve {
                points: [(1, 0.0), (2, 0.0), (3, 0.0)],
            }; 4],
        };
        let dispatch: Vec<_> = std::iter::successors(Some(1usize), |&width| {
            (width < usize::MAX).then(|| width.saturating_mul(2))
        })
        .map(|width| (width, Duration::ZERO))
        .collect();
        let phase = summary(4, 2, 1, 1, 0).scaled();
        assert_eq!(choose(&rates, phase, &dispatch, usize::MAX, true), 1);
        assert_eq!(choose(&rates, phase, &dispatch[..2], 2, true), 1);
    }

    /// The larger population of small items splits with cheap dispatch;
    /// the smaller population is the negative control retaining inline.
    #[test]
    fn cheap_dispatch_respects_inline_competition() {
        let rates = measured_rates();
        let cheap = ladder(&[(1, 0.0), (2, 600.0), (4, 600.0)]);
        assert!(choose(&rates, summary(589824, 2304, 200, 1, 0), &cheap, 4, true) > 1);
        assert_eq!(
            choose(&rates, summary(589824, 2304, 99, 1, 0), &cheap, 4, true),
            1
        );
    }

    /// Many small run-heavy items use the available workers; prohibitive
    /// dispatch is the negative control selecting inline.
    #[test]
    fn small_run_heavy_phase_uses_workers() {
        let phase = summary(256, 256, 1800, 16, 1);
        let rates = measured_rates();
        assert_eq!(
            choose(
                &rates,
                phase,
                &ladder(&[(1, 0.0), (2, 100.0), (4, 100.0)]),
                4,
                true,
            ),
            4
        );
        assert_eq!(
            choose(
                &rates,
                phase,
                &ladder(&[(1, 0.0), (2, 1e9), (4, 1e9)]),
                4,
                true
            ),
            1
        );
    }

    /// The approved approximation equals the item-wise model for one large
    /// item plus serial small items; another medium item exposes its limit.
    #[test]
    fn inline_approximation_is_exact_for_one_large_item() {
        let rates = Rates {
            setup: 1.0,
            runs: [Curve {
                points: [(1, 1.0), (2, 1.0)],
            }; 4],
            elements: [Curve {
                points: [(1, 1.0), (2, 1.0), (3, 1.0)],
            }; 4],
        };
        let phase = summary(1000, 2, 3, 1, 0);
        let actual = inline_time(&rates, phase, 1.0, 10.0, 4, true);
        assert_eq!(actual, 12.0 + 10.0 + 1002.0 / 4.0);
        let medium = summary(1000, 100, 1, 1, 0);
        let approximation = inline_time(&rates, medium, 1.0, 10.0, 4, true);
        let itemwise = 10.0 + 1002.0 / 4.0 + 10.0 + 102.0 / 4.0;
        assert!(approximation > itemwise);
    }

    /// Fill roles always walk serially; granting the largest role fictitious
    /// fan-out is the negative control that incorrectly retains inline.
    #[test]
    fn serial_fill_dominance_selects_split() {
        let rates = Rates {
            setup: 1.0,
            runs: [Curve {
                points: [(1, 1.0), (2, 1.0)],
            }; 4],
            elements: [Curve {
                points: [(1, 1.0), (2, 1.0), (3, 1.0)],
            }; 4],
        };
        let phase = summary(1000, 2, 10, 1, 1);
        let dispatch = ladder(&[(1, 0.0), (2, 10.0), (4, 10.0)]);
        assert_eq!(inline_time(&rates, phase, 1.0, 10.0, 4, false), 1042.0);
        assert!(choose(&rates, phase, &dispatch, 4, false) > 1);
        assert_eq!(choose(&rates, phase, &dispatch, 4, true), 1);
    }

    /// The inline approximation widens one actual role; a larger synthetic
    /// hull bound remains relevant only to the split's load estimate.
    #[test]
    fn inline_uses_largest_role_instead_of_hull_maxima() {
        let rates = Rates {
            setup: 1.0,
            runs: [Curve {
                points: [(1, 1.0), (2, 1.0)],
            }; 4],
            elements: [Curve {
                points: [(1, 1.0), (2, 1.0), (3, 1.0)],
            }; 4],
        };
        let mut phase = summary(1000, 2, 3, 1, 0);
        phase.total = MovementCounts {
            roles: [1, 3, 0, 0],
            runs: [1, 3, 0, 0],
            elements: [1000, 6, 0, 0],
        };
        phase.max.roles[1] = 1;
        phase.max.runs[1] = 1;
        phase.max.elements[1] = 2;
        assert_eq!(
            inline_time(&rates, phase, 1.0, 10.0, 4, true),
            12.0 + 10.0 + 1002.0 / 4.0
        );
    }

    /// Interval minimization matches an independently enumerated model at
    /// every integer width, including curve and dispatch crossings.
    #[test]
    fn interval_search_matches_integer_oracle() {
        let rates = measured_rates();
        let dispatch = ladder(&[(1, 0.0), (2, 10.0), (8, 50.0), (32, 700.0), (96, 10000.0)]);
        for working_set in [65536, 1048576, 13086720, 33554432] {
            let mut phase = summary(4096, 256, 31, 16, 1);
            phase.working_set_elements = working_set / 8;
            for budget in 2..=96 {
                let bytes = working_set as f64;
                let small = |w: f64| {
                    rates.setup + rates.runs[1].at(w) * 16.0 + rates.elements[1].at(w) * 256.0
                };
                let large = |w: f64| {
                    rates.setup + rates.runs[1].at(w) * 16.0 + rates.elements[1].at(w) * 4096.0
                };
                let mut expected = 1;
                let mut best = 31.0 * small(bytes)
                    + large(bytes)
                        .min(dispatch_at(&dispatch, budget) + large(bytes) / budget as f64);
                for q in 2..=budget {
                    let w = bytes / q as f64;
                    let score = dispatch_at(&dispatch, q)
                        + large(w).max((large(w) + 31.0 * small(w)) / q as f64);
                    if score < best {
                        best = score;
                        expected = q;
                    }
                }
                assert_eq!(choose(&rates, phase, &dispatch, budget, true), expected);
            }
        }
        let empty = MovementSummary::default();
        assert_eq!(choose(&rates, empty, &dispatch, 96, true), 1);
    }

    /// Log interpolation uses the geometric midpoint and flat endpoint tails;
    /// a large decrease or invalid measured value is rejected.
    #[test]
    fn curves_interpolate_without_cache_regimes() {
        let curve = Curve {
            points: [(64, 2.0), (1024, 6.0)],
        };
        assert_eq!(curve.at(256.0), 4.0);
        assert_eq!(curve.at(1.0), 2.0);
        assert_eq!(curve.at(4096.0), 6.0);
        assert!(!Curve {
            points: [(64, 6.0), (1024, 2.0)]
        }
        .valid());
        assert!(!Curve {
            points: [(64, -1.0), (1024, 2.0)]
        }
        .valid());
        assert!(!Curve {
            points: [(64, 1.0), (1024, f64::INFINITY)]
        }
        .valid());
    }
    /// Equal split scores choose the smaller integer route; all-zero costs
    /// are the control whose tie retains inline route one.
    #[test]
    fn ties_keep_the_smaller_route() {
        let mut rates = Rates {
            setup: 1.0,
            runs: [Curve {
                points: [(1, 1.0), (2, 1.0)],
            }; 4],
            elements: [Curve {
                points: [(1, 1.0), (2, 1.0), (3, 1.0)],
            }; 4],
        };
        let dispatch = ladder(&[(1, 0.0), (4, 0.0)]);
        assert_eq!(
            choose(&rates, summary(2, 2, 1, 1, 0), &dispatch, 4, true),
            2
        );
        rates.setup = 0.0;
        for curve in &mut rates.runs {
            for point in &mut curve.points {
                point.1 = 0.0;
            }
        }
        for curve in &mut rates.elements {
            for point in &mut curve.points {
                point.1 = 0.0;
            }
        }
        assert_eq!(
            choose(&rates, summary(2, 2, 1, 1, 0), &dispatch, 4, true),
            1
        );
    }
}

#[cfg(test)]
mod calibration_tests {
    use super::*;

    /// A dominant-role shape with real recoupling factors reaches scale pricing;
    /// prohibitive dispatch is the control keeping its calibrated width inline.
    #[test]
    fn nonidentity_coefficient_schedule_admits_scaled_dominant_phase() {
        let shapes = std::iter::once(vec![589824]).chain((0..99).map(|_| vec![2304]));
        let space = BlockStructure::packed_column_major(1, shapes).unwrap();
        let specs: Vec<_> = (0..100)
            .map(|block| crate::TreeTransformBlockSpec::single(block, block, -0.75))
            .collect();
        let transform = TreeTransformStructure::compile_structures(&space, &space, &specs)
            .unwrap()
            .classify_single_coefficients();
        let writer = OwnedWriter {
            task: ReplayParts::new(transform.task_view().unwrap()),
            alpha: 1.0,
        };
        let [phase, _] = writer.movement_phases(transform.parallel_schedule());
        let phase = phase.expect("finite real scaled Singles are eligible");
        assert_eq!(phase.total.roles, [0, 0, specs.len(), 0]);
        let rates = super::tests::measured_rates();
        let dispatch = super::tests::ladder(&[
            (1, 0.0),
            (2, 647.5),
            (4, 894.0),
            (8, 1429.0),
            (16, 2435.0),
            (32, 8927.5),
            (64, 30267.5),
            (96, 44956.5),
        ]);
        assert!(choose(&rates, phase, &dispatch, 96, true) > 1);
        assert_eq!(
            choose(
                &rates,
                phase,
                &super::tests::ladder(&[(1, 0.0), (96, 1e12)]),
                96,
                true
            ),
            1
        );
    }

    /// The real direct replay records model eligibility for real scales;
    /// conjugation and complex payloads are controls staying on the base route.
    #[test]
    fn conjugated_and_complex_owned_singles_trace_base_routes() {
        use num_complex::Complex64;
        let space = Arc::new(BlockStructure::packed_column_major(1, [vec![32], vec![32]]).unwrap());
        let specs = [
            crate::TreeTransformBlockSpec::single(0, 0, -0.75),
            crate::TreeTransformBlockSpec::single(1, 1, 1.0),
        ];
        let mut engine =
            tenferro_cpu::CpuEngine::new(tenferro_cpu::Pool::builder().threads(2).build().unwrap());
        let mut dense = symtn_dense::DefaultDenseExecutor::new();
        for conjugate in [false, true] {
            let transform = TreeTransformStructure::compile_structures_with_storage_conjugation(
                &space, &space, &specs, conjugate,
            )
            .unwrap()
            .classify_single_coefficients();
            if !conjugate {
                for (alpha, expected) in [
                    (1.0, Some([1, 0, 1, 0])),
                    (-2.0, Some([0, 0, 2, 0])),
                    (0.0, None),
                ] {
                    let writer = OwnedWriter {
                        task: ReplayParts::new(transform.task_view().unwrap()),
                        alpha,
                    };
                    assert_eq!(
                        writer.movement_phases(transform.parallel_schedule())[0]
                            .map(|phase| phase.total.roles),
                        expected
                    );
                }
            }
            let mut real = TreeTransformWorkspace::<f64>::default();
            try_tree_transform_structure_overwrite_owned_raw(
                &mut engine,
                &mut dense,
                &mut real,
                &transform,
                &space,
                &space,
                1,
                &[2.0; 64],
                1.0,
                None,
            )
            .unwrap()
            .unwrap();
            assert_eq!(real.direct_routes[0].0, !conjugate);
            if conjugate {
                assert_eq!(real.direct_routes[0], (false, None));
            }
            let mut complex = TreeTransformWorkspace::<Complex64>::default();
            try_tree_transform_structure_overwrite_owned_raw(
                &mut engine,
                &mut dense,
                &mut complex,
                &transform,
                &space,
                &space,
                1,
                &[Complex64::new(2.0, 1.0); 64],
                Complex64::new(1.0, 0.0),
                None,
            )
            .unwrap()
            .unwrap();
            assert_eq!(complex.direct_routes[0], (false, None));
        }
    }

    /// Every tiny, one-run and cache-resident many-run fixture must retain
    /// the traversal class checked by Fixture::new after normalization.
    #[test]
    fn online_ladder_geometry_preserves_traversal_classes() {
        for fill in [false, true] {
            for n in [0, 2, 16] {
                Fixture::new(n, 1, 0, fill, 1.0).unwrap();
            }
            for kind in usize::from(fill)..2 {
                for w in WORKING_SETS {
                    Fixture::new(elements_at(w, fill), 1, kind, fill, 1.0).unwrap();
                }
                let n = elements_at(WORKING_SETS[0], fill);
                Fixture::new(n, n / 2, kind, fill, 1.0).unwrap();
            }
        }
    }

    /// Slight measured dips are accepted then enveloped for convex search;
    /// a larger dip and a negative coefficient remain rejected evidence.
    #[test]
    fn calibration_envelopes_only_accepted_noise() {
        let mut curve = Curve {
            points: [(1, 2.0), (2, 1.9), (4, 3.0)],
        };
        assert!(curve.valid());
        curve.monotone();
        assert_eq!(curve.points, [(1, 2.0), (2, 2.0), (4, 3.0)]);
        assert!(!Curve {
            points: [(1, 2.0), (2, 1.7)],
        }
        .valid());
        assert!(!Curve {
            points: [(1, -1.0), (2, 2.0)],
        }
        .valid());
    }

    /// Real timing must exercise measurement through payload construction;
    /// clock noise can legitimately reject a fit even without a short deadline.
    #[test]
    #[ignore = "real clock and data-quality rejection can legitimately yield no calibration"]
    fn real_online_calibration_produces_finite_nonnegative_payload() {
        let row =
            tenferro_cpu::threadteam::WidthCosts::from_anchors(&[(2, Duration::ZERO)]).unwrap();
        let mut engine = tenferro_cpu::CpuEngine::new(
            tenferro_cpu::Pool::builder()
                .threads(2)
                .dispatch_costs(tenferro_cpu::threadteam::DispatchCosts::new(row, row, row))
                .build()
                .unwrap(),
        );
        let payload = crate::host_pool::lane_region(
            &mut engine,
            crate::host_pool::HostPoolSite::Replay,
            |mut exec| calibrate(&mut exec, Instant::now() + Duration::from_secs(60)),
        )
        .unwrap()
        .expect("real measurement was rejected");
        for rates in [payload.copy, payload.fill] {
            assert!(std::iter::once(rates.setup)
                .chain(
                    rates
                        .runs
                        .iter()
                        .flat_map(|curve| curve.points.iter().map(|p| p.1))
                )
                .chain(
                    rates
                        .elements
                        .iter()
                        .flat_map(|curve| curve.points.iter().map(|p| p.1))
                )
                .all(|value| value.is_finite() && value >= 0.0));
        }
    }
}
