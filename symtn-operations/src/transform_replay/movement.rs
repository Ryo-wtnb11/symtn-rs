//! Replay-owned movement calibration and fixed-form phase pricing.
use super::*;
use crate::transform_structure::{MovementCounts, MovementSummary};
use std::time::{Duration, Instant};

const WORKING_SETS: [usize; 3] = [64 * 1024, 1024 * 1024, 16 * 1024 * 1024];

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
                .all(|pair| pair[0].0 < pair[1].0 && pair[0].1 <= pair[1].1)
    }
}

#[derive(Clone, Copy)]
struct Rates {
    setup: f64,
    runs: [Curve<2>; 2],
    elements: [Curve<3>; 2],
}

impl Rates {
    fn cost(&self, counts: MovementCounts, bytes: f64) -> f64 {
        self.setup * counts.roles.iter().sum::<usize>() as f64
            + (0..2)
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
/// Copy and zero-fill have independent measurements; unavailable maps use base.
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
) -> f64 {
    let sum = rates.cost(phase.total, bytes);
    let largest = rates.cost(phase.max, bytes).min(sum);
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
) -> usize {
    if budget <= 1
        || phase.total.roles == [0; 2]
        || !phase.exact_working_set
        || !phase.supported_span
    {
        return 1;
    }
    let bytes = phase.working_set_elements as f64 * std::mem::size_of::<f64>() as f64;
    let mut best = 1;
    let mut time = inline_time(rates, phase, bytes, dispatch_at(dispatch, budget), budget);
    let mut cuts = [2usize; usize::BITS as usize + 32];
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
    for k in 0..2 {
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
    schedule: &TreeTransformParallelSchedule,
    budget: usize,
) -> [usize; 2] {
    [
        choose(
            &calibration.copy,
            schedule.single_movement,
            dispatch,
            budget,
        ),
        choose(
            &calibration.fill,
            schedule.inactive_movement,
            dispatch,
            budget,
        ),
    ]
}

/// The raw replay already holds the CPU entry here. Re-entering it would
/// deadlock, so calibrate its actual inline traversal/writer on the serial token.
pub(super) fn calibrate(
    exec: &mut symtn_dense::lanes::Exec<'_>,
    deadline: Instant,
) -> Option<Calibration> {
    let copy = measure(exec, deadline, false)?;
    let fill = measure(exec, deadline, true)?;
    Some(Calibration { copy, fill }).filter(|_| Instant::now() < deadline)
}

struct Fixture {
    transform: TreeTransformStructure<f64>,
    source: Vec<f64>,
    destination: Vec<f64>,
    count: usize,
}

impl Fixture {
    fn new(elements: usize, runs: usize, kind: usize, fill: bool) -> Result<Self, OperationError> {
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
            vec![crate::TreeTransformBlockSpec::single(0, 0, 1.0)]
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
            source: vec![1.0; src.required_len()?],
            destination: Vec::with_capacity(count),
            count,
        })
    }

    fn sample(
        &mut self,
        exec: &mut symtn_dense::lanes::Exec<'_>,
        deadline: Instant,
        repetitions: usize,
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
        let mut index = vec![0; task.layouts().max_fused_rank()];
        let mut zeros = Vec::new();
        let start = Instant::now();
        for _ in 0..repetitions {
            inline_direct(
                exec.reborrow()
                    .limit(core::num::NonZeroUsize::MIN)
                    .into_token(),
                self.transform.parallel_schedule(),
                true,
                InlineSingles::Schedule,
                &mut self.destination.spare_capacity_mut()[..self.count],
                &self.source,
                &writer,
                &mut (),
                &mut zeros,
                &mut index,
                &mut gate,
            )
            .ok()?;
            std::hint::black_box(&self.destination.spare_capacity_mut()[..self.count]);
        }
        let ns = start.elapsed().as_secs_f64() * 1e9 / repetitions as f64;
        (Instant::now() < deadline).then_some(ns)
    }
}

fn measured(
    exec: &mut symtn_dense::lanes::Exec<'_>,
    deadline: Instant,
    elements: usize,
    runs: usize,
    kind: usize,
    fill: bool,
) -> Option<f64> {
    if Instant::now() >= deadline {
        return None;
    }
    let mut fixture = Fixture::new(elements, runs, kind, fill).ok()?;
    let repetitions = if elements <= 16 { 128 } else { 1 };
    fixture.sample(exec, deadline, 1)?;
    let mut samples = [0.0; 3];
    // ponytail: one large sample fits the online budget; use the offline campaign for stable medians.
    let sample_count = if elements >= 65536 { 1 } else { samples.len() };
    for sample in &mut samples[..sample_count] {
        *sample = fixture.sample(exec, deadline, repetitions)?;
    }
    samples[..sample_count].sort_by(f64::total_cmp);
    Some(samples[sample_count / 2])
}

fn measure(
    exec: &mut symtn_dense::lanes::Exec<'_>,
    deadline: Instant,
    fill: bool,
) -> Option<Rates> {
    let empty = measured(exec, deadline, 0, 1, 0, fill)?;
    let mut run_points = [[(0, 0.0); 2]; 2];
    let mut one_run = [[0.0; 3]; 2];
    for (k, points) in run_points.iter_mut().enumerate().skip(usize::from(fill)) {
        for (j, w) in [WORKING_SETS[0], WORKING_SETS[2]].into_iter().enumerate() {
            let n = w / if fill { 8 } else { 16 };
            let one = measured(exec, deadline, n, 1, k, fill)?;
            one_run[k][j * 2] = one;
            let many_runs = n / 16;
            let many = measured(exec, deadline, n, many_runs, k, fill)?;
            points[j] = (w, (many - one) / (many_runs - 1) as f64);
        }
    }
    if fill {
        run_points[0] = run_points[1];
    }
    let runs = run_points.map(|points| Curve { points });
    let low = measured(exec, deadline, 2, 1, 0, fill)?;
    let high = measured(exec, deadline, 16, 1, 0, fill)?;
    let slope = (high - low) / 14.0;
    let setup = low - empty - slope * 2.0 - runs[0].points[0].1;
    let mut element_points = [[(0, 0.0); 3]; 2];
    for (k, points) in element_points
        .iter_mut()
        .enumerate()
        .skip(usize::from(fill))
    {
        for (j, w) in WORKING_SETS.into_iter().enumerate() {
            let n = w / if fill { 8 } else { 16 };
            let time = if j == 1 {
                measured(exec, deadline, n, 1, k, fill)?
            } else {
                one_run[k][j]
            };
            points[j] = (w, (time - empty - setup - runs[k].at(w as f64)) / n as f64);
        }
    }
    if fill {
        element_points[0] = element_points[1];
    }
    let rates = Rates {
        setup,
        runs,
        elements: element_points.map(|points| Curve { points }),
    };
    rates.valid().then_some(rates)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn measured_rates() -> Rates {
        Rates {
            setup: 9.0,
            runs: [Curve {
                points: [(65536, 1.3), (16777216, 1.6)],
            }; 2],
            elements: [
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
            },
            working_set_elements: total.elements[kind] * 2,
            exact_working_set: true,
            supported_span: true,
        }
    }

    fn ladder(anchors: &[(usize, f64)]) -> Vec<(usize, Duration)> {
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
        assert!(choose(&rates, phase, &dispatch, 96) > 1);
        let free = ladder(&[(1, 0.0), (96, 0.0)]);
        assert_eq!(choose(&rates, phase, &free, 96), 1);
    }

    /// The larger population of small items splits with cheap dispatch;
    /// the smaller population is the negative control retaining inline.
    #[test]
    fn cheap_dispatch_respects_inline_competition() {
        let rates = measured_rates();
        let cheap = ladder(&[(1, 0.0), (2, 600.0), (4, 600.0)]);
        assert!(choose(&rates, summary(589824, 2304, 200, 1, 0), &cheap, 4) > 1);
        assert_eq!(
            choose(&rates, summary(589824, 2304, 99, 1, 0), &cheap, 4),
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
                4
            ),
            4
        );
        assert_eq!(
            choose(&rates, phase, &ladder(&[(1, 0.0), (2, 1e9), (4, 1e9)]), 4),
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
            }; 2],
            elements: [Curve {
                points: [(1, 1.0), (2, 1.0), (3, 1.0)],
            }; 2],
        };
        let phase = summary(1000, 2, 3, 1, 0);
        let actual = inline_time(&rates, phase, 1.0, 10.0, 4);
        assert_eq!(actual, 12.0 + 10.0 + 1002.0 / 4.0);
        let medium = summary(1000, 100, 1, 1, 0);
        let approximation = inline_time(&rates, medium, 1.0, 10.0, 4);
        let itemwise = 10.0 + 1002.0 / 4.0 + 10.0 + 102.0 / 4.0;
        assert!(approximation > itemwise);
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
                assert_eq!(choose(&rates, phase, &dispatch, budget), expected);
            }
        }
        let empty = MovementSummary::default();
        assert_eq!(choose(&rates, empty, &dispatch, 96), 1);
    }

    /// Log interpolation uses the geometric midpoint and flat endpoint tails;
    /// a decreasing measured curve is rejected rather than reshaped.
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
            }; 2],
            elements: [Curve {
                points: [(1, 1.0), (2, 1.0), (3, 1.0)],
            }; 2],
        };
        let dispatch = ladder(&[(1, 0.0), (4, 0.0)]);
        assert_eq!(choose(&rates, summary(2, 2, 1, 1, 0), &dispatch, 4), 2);
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
        assert_eq!(choose(&rates, summary(2, 2, 1, 1, 0), &dispatch, 4), 1);
    }
}
