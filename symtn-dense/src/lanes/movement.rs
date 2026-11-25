use super::*;
use crate::{DenseBackend, DenseError, Engine};
use std::time::Instant;

/// Internal seam between workspace crates, not a user API: choose each
/// movement phase's width and run `op` on the same execution context.
/// Each phase has serial estimate `estimate` and `runnable` independent parts.
/// Auto compares split execution with sequential items on the whole token;
/// forced policies retain their batch decision. Non-CPU and broken engines
/// are refused, and a serial engine does no calibration.
///
/// Calibration forks only inside this guarded entry, sharing its root with
/// `op`. It is retained separately from contraction scratch under the engine's
/// extension-store limits. Refused admission uses the engine's dispatch table
/// without measurement; eviction or `symtn::scratch::trim` makes it cold.
/// The chooser is O(ladder size), with no per-decision allocation.
#[doc(hidden)]
pub fn movement_region<const N: usize, R>(
    engine: &mut dyn Engine,
    estimate: Duration,
    runnable: [usize; N],
    op: impl FnOnce(Exec<'_>, [usize; N]) -> R,
) -> Result<R, DenseError> {
    let backend_error = |error: tenferro_tensor::Error| DenseError::Backend {
        backend: DenseBackend::Tenferro,
        op: "host_region",
        message: error.to_string(),
    };
    tenferro_cpu::with_cpu_engine(engine, |cpu| {
        let budget = cpu.num_threads();
        let mut rates = None;
        let mut table = [(1, Duration::ZERO); usize::BITS as usize + 1];
        let mut len = 0;
        let mut measure = false;
        if !cpu.is_broken()
            && budget > 1
            && runnable.iter().any(|&parts| parts > 1)
            && cpu.batch_policy().strategy == BatchStrategy::Auto
        {
            rates = crate::engine_slot::checkout_movement(cpu);
            if rates.is_none() {
                let mut lower = table;
                for width in widths(budget) {
                    let cost = if width == 1 {
                        Duration::ZERO
                    } else {
                        cpu.dispatch_costs().cost(
                            threadteam::DispatchKind::Team,
                            NonZeroUsize::new(width).unwrap(),
                        )
                    };
                    table[len] = (width, cost);
                    lower[len] = (width, cost / 2);
                    len += 1;
                }
                // Never measure a ladder the slot cannot retain, or a tiny phase.
                measure = crate::engine_slot::movement_fits(cpu, Calibration::bytes(len))
                    && choose(&lower[..len], estimate, budget) > 1;
            }
        }
        let result = cpu
            .with_exec(|exec| {
                if measure {
                    rates = Some(Calibration::measure(exec));
                }
                let widths = runnable.map(|parts| {
                    if budget <= 1 || parts <= 1 {
                        1
                    } else if exec.batch_policy().strategy != BatchStrategy::Auto {
                        let decision = exec.batch_decision(&BatchQuery::new(parts, 0.0));
                        decision
                            .lanes
                            .saturating_mul(decision.item_threads)
                            .clamp(1, budget.min(parts))
                    } else if let Some(rates) = &rates {
                        choose_phase(&rates.dispatch, estimate, budget, parts)
                    } else {
                        choose_phase(&table[..len], estimate, budget, parts)
                    }
                });
                op(exec.reborrow(), widths)
            })
            .map_err(backend_error);
        if let Some(rates) = rates {
            crate::engine_slot::checkin_movement(cpu, rates);
        }
        result
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
        Self::bytes(self.dispatch.capacity())
    }

    fn bytes(anchors: usize) -> usize {
        anchors.saturating_mul(std::mem::size_of::<(usize, Duration)>())
    }

    fn measure(exec: &mut Exec<'_>) -> Self {
        #[cfg(test)]
        crate::probe::add(crate::probe::Event::MovementCalibration, 1);
        // Five batches share a 10 ms table-priced target across the anchors.
        const BATCHES: usize = 5;
        let anchors = widths(exec.budget()).count();
        let mut dispatch = Vec::with_capacity(anchors);
        dispatch.push((1, Duration::ZERO));
        for width in widths(exec.budget()).skip(1) {
            let k = NonZeroUsize::new(width).unwrap();
            let baseline = exec.token_ref().dispatch_cost(k);
            let sections = sections_per_batch(baseline, anchors - 1);
            let mut samples = [Duration::ZERO; BATCHES];
            exec.reborrow().limit(k).into_token().fork(width, |_, _| {});
            for sample in &mut samples {
                let start = Instant::now();
                for _ in 0..sections {
                    exec.reborrow().limit(k).into_token().fork(width, |_, _| {});
                }
                *sample = Instant::now()
                    .checked_duration_since(start)
                    .map(|elapsed| elapsed / sections)
                    .filter(|elapsed| !elapsed.is_zero())
                    .unwrap_or(baseline);
            }
            samples.sort_unstable();
            let measured =
                calibrated_dispatch(samples[BATCHES / 2], baseline, dispatch.last().unwrap().1);
            dispatch.push((width, measured));
        }
        // threadteam 0.5 exposes neither prefix placement nor node boundaries;
        // interpolate between powers of two and the full budget. Topology and
        // bandwidth calibration belong to the flat vector-operation work.
        Self { dispatch }
    }
}

// Faster-than-table measurements are valid; only excessive samples fall back.
// A PoolBuilder override remains authoritative, including explicitly free dispatch.
fn calibrated_dispatch(measured: Duration, table: Duration, previous: Duration) -> Duration {
    const MAX_TABLE_MULTIPLE: u32 = 20;
    let upper = table.saturating_mul(MAX_TABLE_MULTIPLE);
    if measured > upper {
        table
    } else {
        measured.max(previous).min(upper)
    }
}

fn sections_per_batch(cost: Duration, anchors: usize) -> u32 {
    // At most 200 sections per anchor, shared evenly by five median batches.
    let sections =
        (Duration::from_millis(10).as_nanos() / anchors.max(1) as u128 / cost.as_nanos().max(1))
            .clamp(20, 200);
    (sections / 5) as u32
}

fn choose_phase(
    dispatch: &[(usize, Duration)],
    estimate: Duration,
    budget: usize,
    runnable: usize,
) -> usize {
    let split = choose(dispatch, estimate, budget);
    if split == 1 {
        return 1;
    }
    let items = runnable.max(1) as u128;
    let serial = estimate.as_nanos();
    let inline = choose_sections(dispatch, estimate, budget, items);
    let inline_time = phase_time(dispatch, inline, serial, items);
    let split_time = phase_time(dispatch, split, serial, 1);
    let order = split_time.0.cmp(&inline_time.0).then_with(|| {
        compare_fractions((split_time.1, split_time.2), (inline_time.1, inline_time.2))
    });
    if order.is_lt() {
        split
    } else {
        1
    }
}

fn phase_time(
    dispatch: &[(usize, Duration)],
    width: usize,
    serial: u128,
    sections: u128,
) -> (u128, u128, u128) {
    if width == 1 {
        return (serial, 0, 1);
    }
    let pair = dispatch.windows(2).find(|pair| width <= pair[1].0).unwrap();
    model_time(
        pair[0].1.as_nanos().saturating_mul(sections),
        pair[1].1.as_nanos().saturating_mul(sections),
        pair[1].0 - pair[0].0,
        width - pair[0].0,
        serial,
        width,
    )
}

fn widths(budget: usize) -> impl Iterator<Item = usize> {
    std::iter::successors(Some(1), move |&width| {
        (width < budget).then(|| width.saturating_mul(2).min(budget))
    })
}

fn choose(dispatch: &[(usize, Duration)], estimate: Duration, cap: usize) -> usize {
    choose_sections(dispatch, estimate, cap, 1)
}

fn choose_sections(
    dispatch: &[(usize, Duration)],
    estimate: Duration,
    cap: usize,
    sections: u128,
) -> usize {
    let serial = estimate.as_nanos();
    let (mut best, mut score) = (1, (serial, 0, 1));
    for pair in dispatch.windows(2) {
        let [(lo, low), (hi, high)] = [pair[0], pair[1]];
        let right = hi.min(cap);
        if right < lo {
            break;
        }
        let (low, high) = (
            low.as_nanos().saturating_mul(sections),
            high.as_nanos().saturating_mul(sections),
        );
        let delta = high.abs_diff(low) as f64 / (hi - lo) as f64;
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
            let cost = model_time(low, high, hi - lo, q - lo, serial, q);
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
    (
        whole.saturating_add(serial / q).saturating_add(carry),
        numerator,
        denominator,
    )
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

    /// Dispatch measurements below half the table remain valid; only the upper bound rejects.
    #[test]
    fn dispatch_measurements_reject_only_above_the_table_multiple() {
        let table = Duration::from_nanos(600);
        for (measured, expected) in [(125, 125), (12_000, 12_000), (12_001, 600)] {
            assert_eq!(
                calibrated_dispatch(Duration::from_nanos(measured), table, Duration::ZERO),
                Duration::from_nanos(expected)
            );
        }
        assert_eq!(
            calibrated_dispatch(Duration::from_nanos(1), Duration::ZERO, table),
            Duration::ZERO
        );
    }

    /// Planned sections share the target and stay bounded even for free or costly tables.
    #[test]
    fn calibration_sections_share_the_time_target() {
        let anchors = 5;
        assert_eq!(sections_per_batch(Duration::from_micros(20), anchors), 20);
        assert_eq!(sections_per_batch(Duration::ZERO, anchors), 40);
        assert_eq!(sections_per_batch(Duration::from_secs(1), anchors), 4);
    }

    /// Inline items retain the full token; a split may offer more slots than items.
    #[test]
    fn movement_models_inline_fanout_and_surplus_slots() {
        let positive = [(1, Duration::ZERO), (8, Duration::from_nanos(7))];
        assert_eq!(
            choose_phase(&positive, Duration::from_nanos(10_000), 8, 1),
            1
        );
        for items in [2, 3, 8] {
            assert_eq!(
                choose_phase(&positive, Duration::from_nanos(10_000), 8, items),
                8
            );
        }
        let free = [(1, Duration::ZERO), (8, Duration::ZERO)];
        assert_eq!(choose_phase(&free, Duration::from_nanos(10_000), 8, 3), 1);
    }

    /// Saturated repeated-dispatch estimates cannot overflow or beat a finite phase.
    #[test]
    fn movement_repeated_dispatch_saturates_without_overflow() {
        let dispatch = [(1, Duration::ZERO), (2, Duration::MAX / 10)];
        assert_eq!(choose_phase(&dispatch, Duration::MAX, 2, usize::MAX), 2);
    }

    /// Refused bytes or entries use the table without measuring it repeatedly.
    #[test]
    fn refused_calibration_uses_the_table_without_measurement() {
        use tenferro_runtime::{ExecutionEngine, ExtensionCacheKey, ExtensionCacheLimits};
        for bytes in [true, false] {
            let mut engine = tenferro_cpu::CpuEngine::new(
                tenferro_cpu::Pool::builder().threads(2).build().unwrap(),
            );
            let mut store = engine.take_extension_caches();
            let limits = if bytes {
                ExtensionCacheLimits::default().with_max_retained_bytes(NonZeroUsize::MIN)
            } else {
                store.put(ExtensionCacheKey::new("test.foreign", "entry", 0), (), 0);
                ExtensionCacheLimits::new(NonZeroUsize::MIN)
            };
            store.set_limits(limits);
            engine.restore_extension_caches(store);
            let seen = record();
            for _ in 0..3 {
                movement_region(&mut engine, Duration::from_secs(1), [2], |_, [width]| width)
                    .unwrap();
            }
            assert_eq!(seen.get(Event::MovementCalibration), 0);
            let mut store = engine.take_extension_caches();
            store.clear();
            store.set_limits(ExtensionCacheLimits::default());
            engine.restore_extension_caches(store);
            movement_region(&mut engine, Duration::from_secs(1), [2], |_, [width]| width).unwrap();
            assert_eq!(seen.get(Event::MovementCalibration), 1);
        }
    }

    /// Serial engines never create a calibration entry or run its probe.
    #[test]
    fn serial_movement_does_not_calibrate() {
        let mut engine = tenferro_cpu::CpuEngine::serial();
        let seen = record();
        assert_eq!(
            movement_region(&mut engine, Duration::from_secs(1), [8], |_, [width]| width).unwrap(),
            1
        );
        assert_eq!(seen.get(Event::MovementCalibration), 0);
        assert_eq!(crate::engine_slot::retained_bytes(&mut engine), 0);
    }

    /// A single inline item can use the whole token without calibrating a split.
    #[test]
    fn single_item_movement_does_not_calibrate() {
        let mut engine =
            tenferro_cpu::CpuEngine::new(tenferro_cpu::Pool::builder().threads(2).build().unwrap());
        let seen = record();
        assert_eq!(
            movement_region(&mut engine, Duration::from_secs(1), [1], |_, [width]| width).unwrap(),
            1
        );
        assert_eq!(seen.get(Event::MovementCalibration), 0);
        assert_eq!(crate::engine_slot::retained_bytes(&mut engine), 0);
    }

    /// The optimistic screening model leaves provably tiny phases cold.
    #[test]
    fn tiny_movement_does_not_calibrate() {
        let mut engine =
            tenferro_cpu::CpuEngine::new(tenferro_cpu::Pool::builder().threads(2).build().unwrap());
        let seen = record();
        assert_eq!(
            movement_region(&mut engine, Duration::ZERO, [2], |_, [width]| width).unwrap(),
            1
        );
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
        movement_region(&mut engine, Duration::from_secs(1), [2], |_, [width]| width).unwrap();
        let cold = seen.get(Event::MovementCalibration);
        assert_eq!(cold, 1);
        movement_region(&mut engine, Duration::from_secs(1), [2], |_, [width]| width).unwrap();
        assert_eq!(seen.get(Event::MovementCalibration), cold);
        assert_eq!(
            crate::engine_slot::checkout::<Vec<u8>>(&mut engine),
            Some(vec![1])
        );
        crate::engine_slot::clear(&mut engine);
        assert_eq!(crate::engine_slot::retained_bytes(&mut engine), 0);
        movement_region(&mut engine, Duration::from_secs(1), [2], |_, [width]| width).unwrap();
        assert_eq!(seen.get(Event::MovementCalibration), cold * 2);
    }
}
