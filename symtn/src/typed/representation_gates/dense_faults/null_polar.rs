use super::*;

#[test]
fn checked_compact_null_is_direct_and_agrees_with_dense() {
    let engine = &mut crate::typed::CpuEngine::serial();
    let calls = Arc::new(SpyCounts::default());
    let _dense = probe::inject_dense(Box::new(SpyExecutor::counting(&calls)));
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0x50));
    let bond = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> = TensorMap::diagonal(
        engine,
        &runtime,
        &bond,
        [SectorSpectrum {
            sector: Label::X,
            values: vec![1.0, 0.0],
        }],
    )
    .unwrap();
    let dense = source.materialize(engine).unwrap();

    // A square sector's shape-based null space is empty whatever its values:
    // read directly, with no dense kernel and the same single
    // coupled-dimension query as the dense route, which also runs none.
    reset_provider_queries(&provider);
    let compact = source.left_null(engine, &[0], &[1]).unwrap();
    let compact_queries = provider.queries_since_reset.load(Ordering::Relaxed);

    reset_provider_queries(&provider);
    let expected = dense.left_null(engine, &[0], &[1]).unwrap();
    assert_eq!(
        provider.queries_since_reset.load(Ordering::Relaxed),
        compact_queries
    );
    assert_eq!(calls.total(), 0);
    assert_eq!(compact.codomain(), expected.codomain());
    assert_eq!(compact.domain(), expected.domain());
    assert!(compact.dense_data().unwrap().is_empty());
    assert!(expected.dense_data().unwrap().is_empty());
}
#[test]
fn checked_generic_lazy_polar_second_svd_failure_keeps_parent_unchanged() {
    let engine = &mut crate::typed::CpuEngine::serial();
    for left in [true, false] {
        let svd_calls = Arc::new(SpyCounts::default());
        let gemm_calls = Arc::clone(&svd_calls);
        let _dense = probe::inject_dense(Box::new(pinv_spy(&svd_calls, Some(2), None)));
        let runtime = Runtime::builder().build().unwrap();
        let provider = Arc::new(CheckedOnlyToy::new(0x51));
        let bond = GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 1), (Label::X, 1)])
            .unwrap();
        let source: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&bond], [&bond], |trees, _| {
                if trees.coupled() == &Label::Vacuum {
                    2.0
                } else {
                    3.0
                }
            })
            .unwrap();
        let before = source.dense_data().unwrap().to_vec();
        let lazy = source.adjoint(engine).unwrap();
        let result = if left {
            lazy.left_polar(engine, &[0], &[1]).map(drop)
        } else {
            lazy.right_polar(engine, &[0], &[1]).map(drop)
        };

        assert!(matches!(
            result,
            Err(GenericTensorError::Plan(
                symtn::typed::CheckedGenericPlanError::Operation(_)
            ))
        ));
        // The first sector's W and P GEMMs precede the failing second SVD.
        assert_eq!(svd_calls.of(PINV_SVD), 2);
        assert_eq!(gemm_calls.of(Kernel::GEMM), 2);
        assert_eq!(gemm_calls.total(), 4);
        assert_eq!(source.dense_data().unwrap(), before.as_slice());
    }
}
#[test]
fn checked_generic_null_dense_failure_is_typed_and_nonpublishing() {
    let engine = &mut crate::typed::CpuEngine::serial();
    // What: a later sector QR failure crosses the public Generic facade as a
    // typed plan error without changing the source or returning a partial null.
    let calls = Arc::new(SpyCounts::default());
    let _dense = probe::inject_dense(Box::new(SpyExecutor::counting(&calls).failing(
        &[Kernel::QrInto],
        Some(2),
        "injected null QR failure",
    )));
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0x52));
    let long =
        GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 2), (Label::X, 2)]).unwrap();
    let short =
        GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 1), (Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> = TensorMap::zeros(engine, &runtime, [&long], [&short]).unwrap();
    let before = source.dense_data().unwrap().to_vec();
    assert!(matches!(
        source.left_null(engine, &[0], &[1]),
        Err(GenericTensorError::Plan(
            symtn::typed::CheckedGenericPlanError::Operation(_)
        ))
    ));
    assert_eq!(calls.of(&[Kernel::QrInto]), 2);
    assert_eq!(calls.of(PINV_SVD), 0);
    assert_eq!(source.dense_data().unwrap(), before);
    assert!(std::ptr::eq(source.provider(), provider.as_ref()));
}
#[test]
fn checked_generic_polar_provider_error_precedes_dense_work() {
    let engine = &mut crate::typed::CpuEngine::serial();
    // What: complete codomain dimensions are queried before domain,
    // direction, admission, and dense work.
    let svd_calls = Arc::new(SpyCounts::default());
    let gemm_calls = Arc::clone(&svd_calls);
    let _dense = probe::inject_dense(Box::new(pinv_spy(&svd_calls, None, None)));
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0x53));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> = TensorMap::zeros(engine, &runtime, [&leg], [&leg]).unwrap();
    let refused_compact: TensorMap<_, f64> = TensorMap::diagonal(
        engine,
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: Label::X,
            values: vec![f64::NAN, 1.0],
        }],
    )
    .unwrap();
    let before = source.dense_data().unwrap().to_vec();
    provider.fail_algebra.store(true, Ordering::Relaxed);
    assert!(matches!(
        source.left_polar(engine, &[0], &[1]),
        Err(GenericTensorError::Plan(
            symtn::typed::CheckedGenericPlanError::Provider(ToyError::Algebra)
        ))
    ));
    assert_eq!(svd_calls.of(PINV_SVD), 0);
    assert_eq!(gemm_calls.of(Kernel::GEMM), 0);
    assert_eq!(gemm_calls.total(), 0);
    assert_eq!(source.dense_data().unwrap(), before);
    // A nonfinite compact diagonal is refused by the shared finite-input
    // stage before any provider query, so the failing provider is never
    // reached.
    assert!(matches!(
        refused_compact.left_polar(engine, &[0], &[1]),
        Err(GenericTensorError::Plan(
            symtn::typed::CheckedGenericPlanError::Operation(
                symtn::typed::OperationError::InvalidArgument {
                    message: "polar input components must be finite"
                }
            )
        ))
    ));
    assert_eq!(svd_calls.of(PINV_SVD), 0);
    assert_eq!(gemm_calls.of(Kernel::GEMM), 0);
    assert_eq!(gemm_calls.total(), 0);
}
#[test]
fn checked_only_compact_diagonal_polar_is_direct_and_refuses_nonfinite_values() {
    let engine = &mut crate::typed::CpuEngine::serial();
    let svd_calls = Arc::new(SpyCounts::default());
    let gemm_calls = Arc::clone(&svd_calls);
    let _dense = probe::inject_dense(Box::new(pinv_spy(&svd_calls, None, None)));
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0x54));
    let bond =
        GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 2), (Label::X, 1)]).unwrap();
    let source: TensorMap<_, Complex64> = TensorMap::diagonal(
        engine,
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: Label::X,
                values: vec![Complex64::new(0.0, -2.0)],
            },
            SectorSpectrum {
                sector: Label::Vacuum,
                values: vec![Complex64::new(3.0, 4.0), Complex64::new(0.0, 0.0)],
            },
        ],
    )
    .unwrap();
    for left in [true, false] {
        let (w, p) = if left {
            let LeftPolar { w, p } = source.left_polar(engine, &[0], &[1]).unwrap();
            (w, p)
        } else {
            let RightPolar { p, wh } = source.right_polar(engine, &[0], &[1]).unwrap();
            (wh, p)
        };
        assert_eq!(svd_calls.of(PINV_SVD), 0);
        assert_eq!(gemm_calls.of(Kernel::GEMM), 0);
        assert_eq!(gemm_calls.total(), 0);
        assert!(std::ptr::eq(w.provider(), provider.as_ref()));
        assert!(std::ptr::eq(p.provider(), provider.as_ref()));
        assert_eq!(w.codomain(), source.codomain());
        assert_eq!(w.domain(), source.domain());
        assert!(w.dense_data().is_err());
        assert!(p.dense_data().is_err());
        let rebuilt = if left {
            w.compose(engine, &p)
        } else {
            p.compose(engine, &w)
        }
        .unwrap();
        assert_same_checked_generic_layout_and_close(&rebuilt, &source, |a, b| (a - b).norm());
    }

    let check_real = |engine: &mut dyn Engine,
                      source: &TensorMap<CheckedOnlyToy, f64>,
                      expected_phase: &[SectorSpectrum<Label, f64>],
                      expected_magnitude: &[SectorSpectrum<Label, f64>]| {
        for left in [true, false] {
            let (w, p) = if left {
                let LeftPolar { w, p } = source.left_polar(engine, &[0], &[1]).unwrap();
                (w, p)
            } else {
                let RightPolar { p, wh } = source.right_polar(engine, &[0], &[1]).unwrap();
                (wh, p)
            };
            assert!(std::ptr::eq(w.provider(), provider.as_ref()));
            assert!(std::ptr::eq(p.provider(), provider.as_ref()));
            assert_eq!(w.codomain(), source.codomain());
            assert_eq!(w.domain(), source.domain());
            assert_eq!(p.codomain(), source.domain());
            assert_eq!(p.domain(), source.domain());
            assert!(w.dense_data().is_err());
            assert!(p.dense_data().is_err());
            assert_eq!(w.diagview(engine).unwrap(), expected_phase);
            assert_eq!(p.diagview(engine).unwrap(), expected_magnitude);
            let rebuilt = if left {
                w.compose(engine, &p)
            } else {
                p.compose(engine, &w)
            }
            .unwrap();
            assert_same_checked_generic_layout_and_close(&rebuilt, source, |a, b| (a - b).abs());
        }
    };
    let real: TensorMap<_, f64> = TensorMap::diagonal(
        engine,
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: Label::X,
                values: vec![3.0],
            },
            SectorSpectrum {
                sector: Label::Vacuum,
                values: vec![-2.0, 0.0],
            },
        ],
    )
    .unwrap();
    check_real(
        engine,
        &real,
        &[
            SectorSpectrum {
                sector: Label::Vacuum,
                values: vec![-1.0, 1.0],
            },
            SectorSpectrum {
                sector: Label::X,
                values: vec![1.0],
            },
        ],
        &[
            SectorSpectrum {
                sector: Label::Vacuum,
                values: vec![2.0, 0.0],
            },
            SectorSpectrum {
                sector: Label::X,
                values: vec![3.0],
            },
        ],
    );
    let dual_bond = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)])
        .unwrap()
        .try_dual()
        .unwrap();
    let dual_sector = dual_bond.sectors().unwrap().remove(0);
    let dual: TensorMap<_, f64> = TensorMap::diagonal(
        engine,
        &runtime,
        &dual_bond,
        [SectorSpectrum {
            sector: dual_sector,
            values: vec![-4.0, 0.0],
        }],
    )
    .unwrap();
    check_real(
        engine,
        &dual,
        &[SectorSpectrum {
            sector: dual_sector,
            values: vec![-1.0, 1.0],
        }],
        &[SectorSpectrum {
            sector: dual_sector,
            values: vec![4.0, 0.0],
        }],
    );
    assert_eq!(svd_calls.of(PINV_SVD), 0);
    assert_eq!(gemm_calls.of(Kernel::GEMM), 0);
    assert_eq!(gemm_calls.total(), 0);

    // A NaN is refused by the shared finite-input stage; `MAX + MAX i` is
    // finite and direct: MAK's diagonal polar `W = sign_safe(a)`,
    // `P = abs(a)` gives the phase `(1 + i) / sqrt(2)` and `abs = Inf`.
    for bad_value in [
        Complex64::new(f64::NAN, 0.0),
        Complex64::new(f64::MAX, f64::MAX),
    ] {
        let bad: TensorMap<_, Complex64> = TensorMap::diagonal(
            engine,
            &runtime,
            &bond,
            [
                SectorSpectrum {
                    sector: Label::Vacuum,
                    values: vec![bad_value, Complex64::new(1.0, 0.0)],
                },
                SectorSpectrum {
                    sector: Label::X,
                    values: vec![Complex64::new(2.0, 0.0)],
                },
            ],
        )
        .unwrap();
        svd_calls.reset();
        if bad_value.re.is_nan() {
            let error = bad.left_polar(engine, &[0], &[1]).map(drop).unwrap_err();
            assert_eq!(
                cause::<symtn::typed::OperationError>(&error),
                Some(&symtn::typed::OperationError::InvalidArgument {
                    message: "polar input components must be finite"
                }),
                "{error:?}"
            );
            assert_eq!(svd_calls.total(), 0);
            continue;
        }
        let LeftPolar { w, p } = bad.left_polar(engine, &[0], &[1]).unwrap();
        assert_eq!(svd_calls.total(), 0);
        let (w, p) = (w.diagview(engine).unwrap(), p.diagview(engine).unwrap());
        assert_eq!(w[0].sector, Label::Vacuum);
        assert_eq!(&w[0].values[1..], [Complex64::new(1.0, 0.0)]);
        assert_eq!(&p[0].values[1..], [Complex64::new(1.0, 0.0)]);
        assert_eq!(w[1].values, [Complex64::new(1.0, 0.0)]);
        assert_eq!(p[1].values, [Complex64::new(2.0, 0.0)]);
        let (phase, magnitude) = (w[0].values[0], p[0].values[0]);
        let half = std::f64::consts::FRAC_1_SQRT_2;
        assert!((phase - Complex64::new(half, half)).norm() <= 4.0 * f64::EPSILON);
        assert_eq!(magnitude, Complex64::new(f64::INFINITY, 0.0));
    }
}
#[test]
fn checked_generic_polar_stages_svd_and_both_gemms_without_publication() {
    let engine = &mut crate::typed::CpuEngine::serial();
    // What: sectors stream (each SVD is followed by its W and P GEMMs, so
    // only one sector's factors are live), and any SVD or GEMM failure
    // returns no factor while preserving the source and provider authority.
    for (fail_svd, fail_gemm, expected_svd, expected_gemm) in [
        (None, None, 2, 4),
        (Some(1), None, 1, 0),
        (Some(2), None, 2, 2),
        (None, Some(1), 1, 1),
        (None, Some(2), 1, 2),
        (None, Some(3), 2, 3),
        (None, Some(4), 2, 4),
    ] {
        let svd_calls = Arc::new(SpyCounts::default());
        let gemm_calls = Arc::clone(&svd_calls);
        let _dense = probe::inject_dense(Box::new(pinv_spy(&svd_calls, fail_svd, fail_gemm)));
        let runtime = Runtime::builder().build().unwrap();
        let provider = Arc::new(CheckedOnlyToy::new(0x55));
        let bond = GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 1), (Label::X, 1)])
            .unwrap();
        let source: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&bond], [&bond], |trees, _| {
                if trees.coupled() == &Label::Vacuum {
                    2.0
                } else {
                    3.0
                }
            })
            .unwrap();
        let before = source.dense_data().unwrap().to_vec();
        let result = source.left_polar(engine, &[0], &[1]);
        if fail_svd.is_some() || fail_gemm.is_some() {
            assert!(matches!(
                result,
                Err(GenericTensorError::Plan(
                    symtn::typed::CheckedGenericPlanError::Operation(_)
                ))
            ));
        } else {
            let LeftPolar { w, p } = result.unwrap();
            assert!(std::ptr::eq(w.provider(), provider.as_ref()));
            assert!(std::ptr::eq(p.provider(), provider.as_ref()));
        }
        assert_eq!(svd_calls.of(PINV_SVD), expected_svd);
        assert_eq!(gemm_calls.of(Kernel::GEMM), expected_gemm);
        assert_eq!(svd_calls.total(), expected_svd + expected_gemm);
        assert_eq!(source.dense_data().unwrap(), before.as_slice());
    }
}
