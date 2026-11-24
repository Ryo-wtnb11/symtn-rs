use super::*;
use crate::{DenseBackend, DenseError, Engine};
use std::time::Instant;

/// Internal seam between workspace crates, not a user API: the width of a
/// movement phase whose serial estimate is `estimate` and whose independent
/// parts cap its width at `runnable`. Auto minimizes calibrated dispatch plus
/// `estimate / q`; forced policies retain their batch decision. A serial
/// engine does no calibration. Non-CPU and broken engines are refused.
///
/// Calibration uses the engine's narrowed tokens on first fork candidacy,
/// and is retained separately from contraction scratch under the engine's
/// extension-store limits. Eviction or `symtn::scratch::trim` makes it cold.
/// The chooser is O(ladder size), with no per-decision allocation.
#[doc(hidden)]
pub fn movement_width(
    engine: &mut dyn Engine,
    estimate: Duration,
    runnable: usize,
) -> Result<usize, DenseError> {
    let backend_error = |error: tenferro_tensor::Error| DenseError::Backend {
        backend: DenseBackend::Tenferro,
        op: "host_region",
        message: error.to_string(),
    };
    tenferro_cpu::with_cpu_engine(engine, |cpu| {
        if cpu.is_broken() {
            return cpu.exec().map(|_| 1).map_err(backend_error);
        }
        let budget = cpu.num_threads();
        let cap = budget.min(runnable.max(1));
        if cap <= 1 {
            return Ok(1);
        }
        if cpu.batch_policy().strategy != BatchStrategy::Auto {
            return cpu
                .with_exec(|exec| {
                    let decision = exec.batch_decision(&BatchQuery::new(runnable, 0.0));
                    decision
                        .lanes
                        .saturating_mul(decision.item_threads)
                        .clamp(1, cap)
                })
                .map_err(backend_error);
        }
        let rates = match crate::engine_slot::checkout_movement(cpu) {
            Some(rates) => rates,
            None => {
                // Even the lowest admissible calibration cannot beat serial:
                // tiny calls do not mint a token or pay for measurement.
                let mut lower = [(1, Duration::ZERO); usize::BITS as usize + 1];
                let mut len = 0;
                for width in widths(budget) {
                    let cost = if width == 1 {
                        Duration::ZERO
                    } else {
                        cpu.dispatch_costs().cost(
                            threadteam::DispatchKind::Team,
                            NonZeroUsize::new(width).unwrap(),
                        ) / 2
                    };
                    lower[len] = (width, cost);
                    len += 1;
                }
                if choose(&lower[..len], estimate, cap) == 1 {
                    return Ok(1);
                }
                cpu.with_exec(Calibration::measure).map_err(backend_error)?
            }
        };
        let width = choose(&rates.dispatch, estimate, cap);
        crate::engine_slot::checkin_movement(cpu, rates);
        Ok(width)
    })
    .unwrap_or_else(|| {
        Err(DenseError::Unsupported {
            op: "host_region",
            message: "a Host region needs the caller's CPU engine".to_string(),
        })
    })
}

pub(crate) struct Calibration {
    dispatch: Vec<(usize, Duration)>,
}

impl Calibration {
    pub(crate) fn retained_bytes(&self) -> usize {
        self.dispatch
            .capacity()
            .saturating_mul(std::mem::size_of::<(usize, Duration)>())
    }

    fn measure(exec: &mut Exec<'_>) -> Self {
        #[cfg(test)]
        crate::probe::add(crate::probe::Event::MovementCalibration, 1);
        // Median of five batches rejects interruptions without timing individual
        // sub-microsecond sections; 5 * 200 = 1000 empty sections per anchor.
        const BATCHES: usize = 5;
        const SECTIONS: u32 = 200;
        // Bound noise to [0.5, 20] times the token's threadteam table. A
        // PoolBuilder override takes precedence over the default class row;
        // in particular, an explicitly free dispatch remains free.
        const MAX_TABLE_MULTIPLE: u32 = 20;
        let mut dispatch = vec![(1, Duration::ZERO)];
        for width in widths(exec.budget()).skip(1) {
            let k = NonZeroUsize::new(width).unwrap();
            let baseline = exec.token_ref().dispatch_cost(k);
            let mut samples = [Duration::ZERO; BATCHES];
            exec.reborrow().limit(k).into_token().fork(width, |_, _| {});
            for sample in &mut samples {
                let start = Instant::now();
                for _ in 0..SECTIONS {
                    exec.reborrow().limit(k).into_token().fork(width, |_, _| {});
                }
                *sample = Instant::now()
                    .checked_duration_since(start)
                    .map(|elapsed| elapsed / SECTIONS)
                    .filter(|elapsed| !elapsed.is_zero())
                    .unwrap_or(baseline);
            }
            samples.sort_unstable();
            let measured = samples[BATCHES / 2]
                .max(dispatch.last().unwrap().1)
                .clamp(baseline / 2, baseline.saturating_mul(MAX_TABLE_MULTIPLE));
            dispatch.push((width, measured));
        }
        // threadteam 0.5 exposes neither prefix placement nor node boundaries;
        // interpolate between powers of two and the full budget. Topology and
        // bandwidth calibration belong to the flat vector-operation work.
        Self { dispatch }
    }
}

fn widths(budget: usize) -> impl Iterator<Item = usize> {
    std::iter::successors(Some(1), move |&width| {
        (width < budget).then(|| width.saturating_mul(2).min(budget))
    })
}

fn choose(dispatch: &[(usize, Duration)], estimate: Duration, cap: usize) -> usize {
    let serial = estimate.as_nanos();
    let (mut best, mut score) = (1, (serial, 0, 1));
    for pair in dispatch.windows(2) {
        let [(lo, low), (hi, high)] = [pair[0], pair[1]];
        let right = hi.min(cap);
        if right < lo {
            break;
        }
        let delta = high.as_nanos().abs_diff(low.as_nanos()) as f64 / (hi - lo) as f64;
        let slope = if high >= low { delta } else { -delta };
        // On a linear dispatch interval, a*q+b+serial/q is convex;
        // only its endpoints and the two neighbours of sqrt(serial/a)
        // can minimize it over integer widths.
        let optimum = if slope > 0.0 {
            (serial as f64 / slope).sqrt() as usize
        } else {
            right
        };
        for q in [
            lo,
            right,
            optimum.clamp(lo, right),
            optimum.saturating_add(1).clamp(lo, right),
        ] {
            let cost = model_time(low.as_nanos(), high.as_nanos(), hi - lo, q - lo, serial, q);
            let order = cost
                .0
                .cmp(&score.0)
                .then_with(|| compare_fractions((cost.1, cost.2), (score.1, score.2)));
            if order.is_lt() || (order.is_eq() && q < best) {
                (best, score) = (q, cost);
            }
        }
    }
    best
}

// An exact whole-nanosecond part and a proper fraction: rounding each
// interpolated cost would lose ties, including between adjacent widths.
fn model_time(
    low: u128,
    high: u128,
    span: usize,
    offset: usize,
    serial: u128,
    q: usize,
) -> (u128, u128, u128) {
    let (span, offset, q) = (span as u128, offset as u128, q as u128);
    let delta = high.abs_diff(low);
    // Only the remainders multiply: each is bounded by a usize width, so
    // their product fits u128 even when a Duration times a width would not.
    let product = (delta % span) * offset;
    let quotient = (delta / span) * offset + product / span;
    let remainder = product % span;
    let (whole, remainder) = if high >= low {
        (low + quotient, remainder)
    } else if remainder == 0 {
        (low - quotient, 0)
    } else {
        (low - quotient - 1, span - remainder)
    };
    let denominator = span * q;
    let (left, right) = (remainder * q, (serial % q) * span);
    let (carry, numerator) = if left >= denominator - right {
        (1, left - (denominator - right))
    } else {
        (0, left + right)
    };
    (whole + serial / q + carry, numerator, denominator)
}

// Continued quotients compare fractions exactly without cross-products.
fn compare_fractions(mut a: (u128, u128), mut b: (u128, u128)) -> std::cmp::Ordering {
    let mut reverse = false;
    loop {
        let order = (a.0 / a.1).cmp(&(b.0 / b.1));
        let (ar, br) = (a.0 % a.1, b.0 % b.1);
        let order = if !order.is_eq() {
            Some(order)
        } else {
            match (ar == 0, br == 0) {
                (true, true) => Some(std::cmp::Ordering::Equal),
                (true, false) => Some(std::cmp::Ordering::Less),
                (false, true) => Some(std::cmp::Ordering::Greater),
                (false, false) => None,
            }
        };
        if let Some(order) = order {
            return if reverse { order.reverse() } else { order };
        }
        (a, b) = ((a.1, ar), (b.1, br));
        reverse = !reverse;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe::{record, Event};

    /// Linear interpolation has an interior minimum, and integer ties keep
    /// the smaller width; the decision also respects each call's cap.
    #[test]
    fn movement_minimizes_the_interpolated_model_with_smaller_ties() {
        let rates = Calibration {
            dispatch: vec![
                (1, Duration::ZERO),
                (2, Duration::from_nanos(2)),
                (8, Duration::from_nanos(26)),
            ],
        };
        // q=4: 10+64/4=26; q=3: 6+64/3>27; q=5: 14+64/5>26.
        assert_eq!(choose(&rates.dispatch, Duration::from_nanos(64), 8), 4);
        // q=2: 2+24/2=14; q=3: 6+24/3=14.
        assert_eq!(choose(&rates.dispatch, Duration::from_nanos(24), 8), 2);
        let fractional = [
            (1, Duration::ZERO),
            (2, Duration::from_nanos(1)),
            (4, Duration::from_nanos(2)),
            (7, Duration::from_nanos(3)),
        ];
        // q=5 and q=6 both cost 13/3 ns, exactly.
        assert_eq!(choose(&fractional, Duration::from_nanos(10), 7), 5);
        let decreasing = [
            (1, Duration::ZERO),
            (2, Duration::from_nanos(100)),
            (4, Duration::from_nanos(1)),
        ];
        assert_eq!(choose(&decreasing, Duration::from_nanos(10), 4), 4);
        for (cap, expected) in [(1, 1), (2, 2), (3, 3), (8, 4)] {
            assert_eq!(
                choose(&rates.dispatch, Duration::from_nanos(64), cap),
                expected
            );
        }
    }

    /// Serial engines never create a calibration entry or run its probe.
    #[test]
    fn serial_movement_does_not_calibrate() {
        let mut engine = tenferro_cpu::CpuEngine::serial();
        let seen = record();
        assert_eq!(
            movement_width(&mut engine, Duration::from_secs(1), 8).unwrap(),
            1
        );
        assert_eq!(seen.get(Event::MovementCalibration), 0);
        assert_eq!(crate::engine_slot::retained_bytes(&mut engine), 0);
    }

    /// A lower-bound model that cannot beat serial leaves tiny phases cold.
    #[test]
    fn tiny_movement_does_not_calibrate() {
        let mut engine =
            tenferro_cpu::CpuEngine::new(tenferro_cpu::Pool::builder().threads(2).build().unwrap());
        let seen = record();
        assert_eq!(movement_width(&mut engine, Duration::ZERO, 2).unwrap(), 1);
        assert_eq!(seen.get(Event::MovementCalibration), 0);
        assert_eq!(crate::engine_slot::retained_bytes(&mut engine), 0);
    }

    /// A calibration is reused while retained; clearing the engine entries
    /// makes the next fork candidate cold again, without losing scratch.
    #[test]
    fn movement_calibration_is_reused_until_clear() {
        let mut engine =
            tenferro_cpu::CpuEngine::new(tenferro_cpu::Pool::builder().threads(2).build().unwrap());
        crate::engine_slot::checkin(&mut engine, vec![1u8], Vec::capacity);
        let seen = record();
        movement_width(&mut engine, Duration::from_secs(1), 2).unwrap();
        let cold = seen.get(Event::MovementCalibration);
        assert_eq!(cold, 1);
        movement_width(&mut engine, Duration::from_secs(1), 2).unwrap();
        assert_eq!(seen.get(Event::MovementCalibration), cold);
        assert_eq!(
            crate::engine_slot::checkout::<Vec<u8>>(&mut engine),
            Some(vec![1])
        );
        crate::engine_slot::clear(&mut engine);
        assert_eq!(crate::engine_slot::retained_bytes(&mut engine), 0);
        movement_width(&mut engine, Duration::from_secs(1), 2).unwrap();
        assert_eq!(seen.get(Event::MovementCalibration), cold * 2);
    }
}
