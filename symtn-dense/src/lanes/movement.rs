use super::*;
use crate::{DenseBackend, DenseError, Engine};
use std::time::Instant;

/// Internal guarded CPU movement entry. Replay owns the opaque, fixed-size
/// calibration payload; this crate owns dispatch and engine retention.
/// A rejected/incomplete payload is retained until eviction or scratch trim.
/// Width-one, forced-policy and ineligible calls perform no tuning.
#[doc(hidden)]
pub fn movement_region<T: Send + 'static, R>(
    engine: &mut dyn Engine,
    eligible: bool,
    calibrate: impl FnOnce(&mut Exec<'_>, Instant) -> Option<T>,
    op: impl FnOnce(Exec<'_>, Option<(&T, &[(usize, Duration)])>) -> R,
) -> Result<R, DenseError> {
    tenferro_cpu::with_cpu_engine(engine, |cpu| {
        let tune = eligible
            && !cpu.is_broken()
            && cpu.num_threads() > 1
            && cpu.batch_policy().strategy == BatchStrategy::Auto;
        let mut rates = tune
            .then(|| crate::engine_slot::checkout_movement::<T>(cpu))
            .flatten();
        let bytes = Calibration::<T>::bytes(cpu.num_threads());
        let measure = tune && rates.is_none() && crate::engine_slot::movement_fits::<T>(cpu, bytes);
        let result = cpu
            .with_exec(|exec| {
                if measure {
                    let start = Instant::now();
                    let deadline = start + Duration::from_millis(10);
                    let dispatch = measure_dispatch(exec, start + Duration::from_millis(2));
                    let payload = dispatch
                        .as_ref()
                        .and_then(|_| calibrate(exec, deadline))
                        .filter(|_| Instant::now() <= deadline);
                    #[cfg(test)]
                    {
                        use crate::probe::{add, Event};
                        add(
                            if payload.is_some() {
                                Event::MovementCalibrationSuccess
                            } else {
                                Event::MovementCalibrationFailure
                            },
                            1,
                        );
                        add(
                            Event::MovementCalibrationMicros,
                            start.elapsed().as_micros().min(usize::MAX as u128) as usize,
                        );
                    }
                    rates = Some(Calibration {
                        dispatch: dispatch.unwrap_or_default(),
                        payload,
                    });
                }
                let accepted = rates.as_ref().and_then(|rates| {
                    rates
                        .payload
                        .as_ref()
                        .map(|payload| (payload, rates.dispatch.as_slice()))
                });
                op(exec.reborrow(), accepted)
            })
            .map_err(|error| DenseError::Backend {
                backend: DenseBackend::Tenferro,
                op: "host_region",
                message: error.to_string(),
            });
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

pub(crate) struct Calibration<T> {
    dispatch: Vec<(usize, Duration)>,
    payload: Option<T>,
}

impl<T> Calibration<T> {
    pub(crate) fn retained_bytes(&self) -> usize {
        self.dispatch
            .capacity()
            .saturating_mul(std::mem::size_of::<(usize, Duration)>())
    }

    fn bytes(budget: usize) -> usize {
        widths(budget)
            .count()
            .saturating_mul(std::mem::size_of::<(usize, Duration)>())
    }
}

fn widths(budget: usize) -> impl Iterator<Item = usize> {
    std::iter::successors(Some(1), move |&width| {
        (width < budget).then(|| width.saturating_mul(2).min(budget))
    })
}

fn measure_dispatch(exec: &mut Exec<'_>, deadline: Instant) -> Option<Vec<(usize, Duration)>> {
    #[cfg(test)]
    crate::probe::add(crate::probe::Event::MovementCalibration, 1);
    let mut dispatch = Vec::with_capacity(widths(exec.budget()).count());
    dispatch.push((1, Duration::ZERO));
    for width in widths(exec.budget()).skip(1) {
        if Instant::now() >= deadline {
            return None;
        }
        let k = NonZeroUsize::new(width).unwrap();
        let table = exec.token_ref().dispatch_cost(k);
        if table.is_zero() {
            dispatch.push((width, table));
            continue;
        }
        let mut samples = [Duration::ZERO; 5];
        exec.reborrow().limit(k).into_token().fork(width, |_, _| {});
        for sample in &mut samples {
            if Instant::now() >= deadline {
                return None;
            }
            let start = Instant::now();
            exec.reborrow().limit(k).into_token().fork(width, |_, _| {});
            *sample = start.elapsed();
        }
        samples.sort_unstable();
        let measured = samples[2];
        let accepted = calibrated_dispatch(measured, table, dispatch.last().unwrap().1);
        dispatch.push((width, accepted));
    }
    (Instant::now() < deadline).then_some(dispatch)
}

fn calibrated_dispatch(measured: Duration, table: Duration, previous: Duration) -> Duration {
    if table.is_zero() || measured > table.saturating_mul(20) {
        table
    } else {
        measured.max(previous).min(table.saturating_mul(20))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe::{record, Event};

    /// A rejected sample preserves the table even below an accepted predecessor.
    #[test]
    fn rejected_dispatch_preserves_reference() {
        let ns = Duration::from_nanos;
        assert_eq!(calibrated_dispatch(ns(12_001), ns(600), ns(800)), ns(600));
        assert_eq!(
            calibrated_dispatch(ns(12_000), ns(600), ns(800)),
            ns(12_000)
        );
        assert_eq!(calibrated_dispatch(ns(1), ns(2), ns(800)), ns(40));
        assert_eq!(
            calibrated_dispatch(ns(1), Duration::ZERO, ns(800)),
            Duration::ZERO
        );
    }

    /// A failed payload is attempted once per retained entry; clear makes it cold.
    #[test]
    fn failed_payload_is_retained_until_clear() {
        let mut engine =
            tenferro_cpu::CpuEngine::new(tenferro_cpu::Pool::builder().threads(2).build().unwrap());
        let seen = record();
        for _ in 0..2 {
            movement_region::<(), _>(
                &mut engine,
                true,
                |_, _| None,
                |_, rates| assert!(rates.is_none()),
            )
            .unwrap();
        }
        assert_eq!(seen.get(Event::MovementCalibration), 1);
        crate::engine_slot::clear(&mut engine);
        movement_region::<(), _>(
            &mut engine,
            true,
            |_, _| None,
            |_, rates| assert!(rates.is_none()),
        )
        .unwrap();
        assert_eq!(seen.get(Event::MovementCalibration), 2);
        assert_eq!(
            seen.get(Event::MovementCalibrationFailure),
            seen.get(Event::MovementCalibration)
        );
        assert_eq!(seen.get(Event::MovementCalibrationSuccess), 0);
    }

    /// Ineligible calls never probe or retain, whereas admitted eligible calls do.
    #[test]
    fn ineligible_movement_stays_cold() {
        let mut engine =
            tenferro_cpu::CpuEngine::new(tenferro_cpu::Pool::builder().threads(2).build().unwrap());
        let seen = record();
        movement_region::<(), _>(
            &mut engine,
            false,
            |_, _| panic!("ineligible calibration"),
            |_, rates| assert!(rates.is_none()),
        )
        .unwrap();
        assert_eq!(seen.get(Event::MovementCalibration), 0);
        assert_eq!(crate::engine_slot::retained_bytes(&mut engine), 0);
        movement_region::<(), _>(&mut engine, true, |_, _| None, |_, _| ()).unwrap();
        assert_eq!(seen.get(Event::MovementCalibration), 1);
    }
    fn free_engine() -> tenferro_cpu::CpuEngine {
        let row = threadteam::WidthCosts::from_anchors(&[(2, Duration::ZERO)]).unwrap();
        tenferro_cpu::CpuEngine::new(
            tenferro_cpu::Pool::builder()
                .threads(2)
                .dispatch_costs(threadteam::DispatchCosts::new(row, row, row))
                .build()
                .unwrap(),
        )
    }

    /// Refused bytes or entries perform no probes; admission restoration is
    /// the negative control making the next eligible entry attempt calibration.
    #[test]
    fn refused_store_never_measures() {
        use tenferro_runtime::{ExecutionEngine, ExtensionCacheKey, ExtensionCacheLimits};
        for bytes in [true, false] {
            let mut engine = free_engine();
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
            for _ in 0..2 {
                movement_region::<(), _>(
                    &mut engine,
                    true,
                    |_, _| panic!("refused probe"),
                    |_, rates| assert!(rates.is_none()),
                )
                .unwrap();
            }
            assert_eq!(seen.get(Event::MovementCalibration), 0);
            let mut store = engine.take_extension_caches();
            store.clear();
            store.set_limits(ExtensionCacheLimits::default());
            engine.restore_extension_caches(store);
            movement_region::<(), _>(&mut engine, true, |_, _| None, |_, _| ()).unwrap();
            assert_eq!(seen.get(Event::MovementCalibration), 1);
        }
    }

    /// Physical serial and forced policies stay cold; Auto on the same
    /// nonserial engine is the control that admits a probe.
    #[test]
    fn serial_and_forced_policies_do_not_calibrate() {
        let seen = record();
        let mut serial = tenferro_cpu::CpuEngine::serial();
        movement_region::<(), _>(
            &mut serial,
            true,
            |_, _| panic!("serial probe"),
            |_, rates| assert!(rates.is_none()),
        )
        .unwrap();
        for strategy in [
            BatchStrategy::Sequential,
            BatchStrategy::Outer,
            BatchStrategy::Inner,
            BatchStrategy::Split {
                lanes: NonZeroUsize::MIN,
            },
        ] {
            let mut engine = free_engine().with_batch_policy(BatchPolicy::new(strategy));
            movement_region::<(), _>(
                &mut engine,
                true,
                |_, _| panic!("forced probe"),
                |_, rates| assert!(rates.is_none()),
            )
            .unwrap();
            assert_eq!(crate::engine_slot::retained_bytes(&mut engine), 0);
        }
        assert_eq!(seen.get(Event::MovementCalibration), 0);
        let mut engine = free_engine();
        movement_region::<(), _>(&mut engine, true, |_, _| None, |_, _| ()).unwrap();
        assert_eq!(seen.get(Event::MovementCalibration), 1);
    }

    /// A successful retained payload is reused unchanged; clearing the same
    /// engine removes it and makes one new attempt, which is the control.
    #[test]
    fn successful_payload_is_reused_until_clear() {
        let mut engine = free_engine();
        crate::engine_slot::checkin_movement(
            &mut engine,
            Calibration {
                dispatch: vec![(1, Duration::ZERO), (2, Duration::ZERO)],
                payload: Some(7u8),
            },
        );
        let seen = record();
        for _ in 0..2 {
            movement_region::<u8, _>(
                &mut engine,
                true,
                |_, _| panic!("retained probe"),
                |_, rates| {
                    assert_eq!(rates.map(|(value, _)| *value), Some(7));
                },
            )
            .unwrap();
        }
        assert_eq!(seen.get(Event::MovementCalibration), 0);
        crate::engine_slot::clear(&mut engine);
        movement_region::<u8, _>(
            &mut engine,
            true,
            |_, _| None,
            |_, rates| assert!(rates.is_none()),
        )
        .unwrap();
        assert_eq!(seen.get(Event::MovementCalibration), 1);
    }

    /// Each cold accepted payload contributes one success and no rejection;
    /// reuse is the control that contributes no second outcome.
    #[test]
    fn accepted_payload_outcomes_are_counted_only_when_cold() {
        let mut engine = free_engine();
        let seen = record();
        for _ in 0..2 {
            movement_region(
                &mut engine,
                true,
                |_, _| Some(()),
                |_, payload| assert!(payload.is_some()),
            )
            .unwrap();
        }
        assert_eq!(
            seen.get(Event::MovementCalibrationSuccess),
            seen.get(Event::MovementCalibration)
        );
        assert_eq!(seen.get(Event::MovementCalibrationFailure), 0);
    }

    /// Rejected calibration routes through the literal base all-or-one rule;
    /// one runnable item is the control remaining inline without a new probe.
    #[test]
    fn rejected_payload_routes_through_base() {
        let mut engine = free_engine();
        let seen = record();
        for parts in [2, 1] {
            let estimate = Duration::from_secs(1);
            let (width, expected) = movement_region::<(), _>(
                &mut engine,
                true,
                |_, _| None,
                |exec, rates| {
                    assert!(rates.is_none());
                    let expected = if exec.token_ref().pays(
                        estimate,
                        1,
                        NonZeroUsize::new(exec.budget()).unwrap(),
                    ) {
                        exec.budget().min(parts)
                    } else {
                        1
                    };
                    (flat_width(&exec, estimate, parts), expected)
                },
            )
            .unwrap();
            assert_eq!(width, expected);
        }
        assert_eq!(seen.get(Event::MovementCalibration), 1);
    }
}
