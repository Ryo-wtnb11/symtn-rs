use super::*;

#[test]
fn checked_compact_diagonal_svd_full_is_its_compact_svd_under_a_provider_bond_mismatch() {
    let engine = &mut crate::typed::CpuEngine::serial();
    let svd_calls = Arc::new(SpyCounts::default());
    let _dense = probe::inject_dense(Box::new(pinv_spy(&svd_calls, None, None)));
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0x70));
    let bond = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let on = |engine: &mut dyn Engine, bond: &GradedSpace<_>| -> TensorMap<_, f64> {
        TensorMap::diagonal(
            engine,
            &runtime,
            bond,
            [SectorSpectrum {
                sector: Label::X,
                values: vec![2.0, -1.0],
            }],
        )
        .unwrap()
    };
    let nondual = on(engine, &bond);
    let dual = on(engine, &bond.try_dual().unwrap());
    provider.extra_vacuum_channel.store(true, Ordering::Relaxed);

    // A nondual one-leg bond is its own SVD bond (`fuse(V) = V`, no fusion),
    // so the factors are direct on the input space and the provider's
    // coupled-dimension report is never consulted.
    reset_provider_queries(&provider);
    let Svd { u, s, vh } = nondual.svd_full(engine, &[0], &[1]).unwrap();
    assert_eq!(provider.queries_since_reset.load(Ordering::Relaxed), 0);
    assert_eq!(svd_calls.of(PINV_SVD), 0);
    for factor in [&u, &s, &vh] {
        assert_eq!(factor.codomain(), nondual.codomain());
    }

    // A dual bond needs the fresh nondual `fuse(V)`. Full SVD of a diagonal is
    // its compact SVD (TensorKit `svd_compact!(::DiagonalAlgorithm)` is
    // `svd_full!`), in every fusion mode (#1994): the compact route builds
    // `fuse(V)` and no square-full bond, so the provider's coupled-dimension
    // report is not consulted, and no dense SVD runs.
    let full = dual.svd_full(engine, &[0], &[1]).unwrap();
    let compact = dual.svd_compact(engine, &[0], &[1]).unwrap();
    assert_eq!(svd_calls.of(PINV_SVD), 0);
    assert!(full.s.dense_data().is_err());
    assert_eq!(
        full.s.diagview(engine).unwrap(),
        compact.s.diagview(engine).unwrap()
    );
    assert_eq!(full.s.codomain(), compact.s.codomain());
    for (full, compact) in [(&full.u, &compact.u), (&full.vh, &compact.vh)] {
        assert_eq!(full.codomain(), compact.codomain());
        assert_eq!(full.domain(), compact.domain());
        assert_eq!(full.dense_data().unwrap(), compact.dense_data().unwrap());
    }
}
#[test]
fn checked_compact_diagonal_svd_is_direct_for_subnormal_and_overflowing_values() {
    let engine = &mut crate::typed::CpuEngine::serial();
    let svd_calls = Arc::new(SpyCounts::default());
    let _dense = probe::inject_dense(Box::new(pinv_spy(&svd_calls, None, None)));
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0x71));
    let bond = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();

    // Hand-computed MAK `svd_full!(::DiagonalAlgorithm)`: `S = abs(a)`
    // exactly, with no dense solver, for a subnormal and the smallest normal.
    // Why not compare with the dense route: LAPACK may flush a subnormal
    // singular value to zero (f32), which is a solver limit, not the result.
    macro_rules! assert_real_magnitude {
        ($dtype:ty, $subnormal:expr, $minimum:expr) => {
            for (name, value) in [("subnormal", $subnormal), ("minimum normal", $minimum)] {
                let input: TensorMap<_, $dtype> = TensorMap::diagonal(
                    engine,
                    &runtime,
                    &bond,
                    [SectorSpectrum {
                        sector: Label::X,
                        values: vec![value],
                    }],
                )
                .unwrap();
                for full in [false, true] {
                    svd_calls.reset();
                    let Svd { s, .. } = if full {
                        input.svd_full(engine, &[0], &[1]).unwrap()
                    } else {
                        input.svd_compact(engine, &[0], &[1]).unwrap()
                    };
                    assert_eq!(svd_calls.of(PINV_SVD), 0, "{name}, full={full}");
                    assert_eq!(
                        s.diagview(engine).unwrap()[0].values,
                        [value],
                        "{name}, full={full}"
                    );
                }
            }
        };
    }
    assert_real_magnitude!(f32, f32::from_bits(1), f32::MIN_POSITIVE);
    assert_real_magnitude!(f64, f64::from_bits(1), f64::MIN_POSITIVE);

    // Complex overflow goes direct as well: MAK's `abs` overflows to `Inf`
    // (hand-computed: |MAX + MAX i| = sqrt(2) MAX exceeds MAX), with no
    // dense solver.
    macro_rules! assert_complex_overflow {
        ($dtype:ty, $value:expr, $infinite:expr) => {
            let overflow: TensorMap<_, $dtype> = TensorMap::diagonal(
                engine,
                &runtime,
                &bond,
                [SectorSpectrum {
                    sector: Label::X,
                    values: vec![$value],
                }],
            )
            .unwrap();
            for full in [false, true] {
                svd_calls.reset();
                let Svd { s, .. } = if full {
                    overflow.svd_full(engine, &[0], &[1]).unwrap()
                } else {
                    overflow.svd_compact(engine, &[0], &[1]).unwrap()
                };
                assert_eq!(svd_calls.of(PINV_SVD), 0, "full={full}");
                assert_eq!(
                    s.diagview(engine).unwrap()[0].values,
                    [$infinite],
                    "full={full}"
                );
            }
        };
    }
    assert_complex_overflow!(
        Complex32,
        Complex32::new(f32::MAX, f32::MAX),
        Complex32::new(f32::INFINITY, 0.0)
    );
    assert_complex_overflow!(
        Complex64,
        Complex64::new(f64::MAX, f64::MAX),
        Complex64::new(f64::INFINITY, 0.0)
    );
}
#[test]
fn checked_compact_diagonal_svd_rejects_nonfinite_and_skips_the_provider_when_finite() {
    let engine = &mut crate::typed::CpuEngine::serial();
    let svd_calls = Arc::new(SpyCounts::default());
    let _dense = probe::inject_dense(Box::new(pinv_spy(&svd_calls, Some(1), None)));
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0x72));
    let bond = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let nonfinite: TensorMap<_, f64> = TensorMap::diagonal(
        engine,
        &runtime,
        &bond,
        [SectorSpectrum {
            sector: Label::X,
            values: vec![f64::NAN, 1.0],
        }],
    )
    .unwrap();
    // The shared finite-input stage refuses a nonfinite diagonal with the
    // typed error, before any provider query or dense SVD.
    provider.fail_algebra.store(true, Ordering::Relaxed);
    let error = nonfinite.svd_compact(engine, &[0], &[1]).unwrap_err();
    assert!(
        matches!(
            &error,
            GenericTensorError::Plan(symtn::typed::CheckedGenericPlanError::Operation(
                symtn::typed::OperationError::InvalidArgument {
                    message: "svd input components must be finite"
                }
            ))
        ),
        "{error:?}"
    );
    assert_eq!(svd_calls.of(PINV_SVD), 0);

    provider.fail_algebra.store(false, Ordering::Relaxed);
    let finite: TensorMap<_, f64> = TensorMap::diagonal(
        engine,
        &runtime,
        &bond,
        [SectorSpectrum {
            sector: Label::X,
            values: vec![2.0, 1.0],
        }],
    )
    .unwrap();
    // A finite nondual bond is its own SVD bond (TensorKit `fuse(V) = V`):
    // `u`, `s` and `vh` live on the input space, so neither a style-changing
    // nor a failing provider is ever queried, and no dense SVD runs.
    for (invalid_style, fail_algebra) in [(true, false), (false, true)] {
        provider
            .invalid_style
            .store(invalid_style, Ordering::Relaxed);
        provider.fail_algebra.store(fail_algebra, Ordering::Relaxed);
        svd_calls.reset();
        reset_provider_queries(provider.as_ref());
        let factors = [
            finite.svd_compact(engine, &[0], &[1]).unwrap(),
            finite.svd_full(engine, &[0], &[1]).unwrap(),
        ];
        assert_eq!(provider.queries_since_reset.load(Ordering::Relaxed), 0);
        assert_eq!(svd_calls.of(PINV_SVD), 0);
        assert_eq!(svd_calls.total(), 0);
        for Svd { u, s, vh } in factors {
            for factor in [&u, &s, &vh] {
                assert_eq!(factor.codomain(), finite.codomain());
                assert_eq!(factor.domain(), finite.domain());
            }
        }
    }
}
/// A dual compact diagonal's SVD bond is the fresh nondual `fuse(V)`, built
/// by the provider, so a provider that changes its fusion style still fails
/// `svd_compact` with the typed plan error and runs no dense SVD.
#[test]
fn checked_dual_compact_diagonal_svd_compact_propagates_provider_errors() {
    let engine = &mut crate::typed::CpuEngine::serial();
    let svd_calls = Arc::new(SpyCounts::default());
    let _dense = probe::inject_dense(Box::new(pinv_spy(&svd_calls, None, None)));
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0x73));
    let bond = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)])
        .unwrap()
        .try_dual()
        .unwrap();
    let diagonal: TensorMap<_, f64> = TensorMap::diagonal(
        engine,
        &runtime,
        &bond,
        [SectorSpectrum {
            sector: Label::X,
            values: vec![2.0, 1.0],
        }],
    )
    .unwrap();
    provider.invalid_style.store(true, Ordering::Relaxed);
    let error = diagonal.svd_compact(engine, &[0], &[1]).unwrap_err();
    assert!(
        matches!(
            error,
            GenericTensorError::Plan(symtn::typed::CheckedGenericPlanError::Operation(
                symtn::typed::OperationError::Core(
                    symtn::typed::CoreError::UnsupportedFusionStyle { .. }
                )
            ))
        ),
        "{error:?}"
    );
    assert_eq!(svd_calls.total(), 0);
}
