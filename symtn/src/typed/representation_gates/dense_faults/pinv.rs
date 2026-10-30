use super::*;

#[test]
fn checked_compact_diagonal_pinv_keeps_a_checked_compact_output() {
    let engine = &mut crate::typed::CpuEngine::serial();
    let svd_calls = Arc::new(SpyCounts::default());
    let gemm_calls = Arc::clone(&svd_calls);
    let _dense = probe::inject_dense(Box::new(pinv_spy(&svd_calls, None, None)));
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0x60));
    let bond =
        GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 2), (Label::X, 3)]).unwrap();
    let input: TensorMap<_, f64> = TensorMap::diagonal(
        engine,
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: Label::X,
                values: vec![1.0, 2.0, 0.0],
            },
            SectorSpectrum {
                sector: Label::Vacuum,
                values: vec![8.0, -4.0],
            },
        ],
    )
    .unwrap();
    let before = input.diagview(engine).unwrap();
    let result = input.pinv(engine, &[0], &[1], 0.25).unwrap();
    assert_eq!(result.codomain(), input.domain());
    assert_eq!(result.domain(), input.codomain());
    assert!(std::ptr::eq(result.provider(), provider.as_ref()));
    assert_eq!(
        result.diagview(engine).unwrap(),
        [
            SectorSpectrum {
                sector: Label::Vacuum,
                values: vec![0.125, -0.25]
            },
            SectorSpectrum {
                sector: Label::X,
                values: vec![0.0, 0.0, 0.0]
            },
        ]
    );
    assert_eq!(input.diagview(engine).unwrap(), before);
    assert!(result.dense_data().is_err());
    assert_eq!(svd_calls.of(PINV_SVD), 0);
    assert_eq!(gemm_calls.of(Kernel::GEMM), 0);
    assert_eq!(gemm_calls.total(), 0);

    provider.invalid_style.store(true, Ordering::Relaxed);
    assert!(matches!(
        input.pinv(engine, &[0], &[1], f64::NAN),
        Err(GenericTensorError::Facade(
            symtn::typed::Error::InvalidArgument(_)
        ))
    ));
    // The compact output keeps the input space, so a style-changing provider
    // is never asked and cannot fail it (#1751).
    reset_provider_queries(&provider);
    let again = input.pinv(engine, &[0], &[1], 0.25).unwrap();
    assert_eq!(provider.queries_since_reset.load(Ordering::Relaxed), 0);
    assert_eq!(
        again.diagview(engine).unwrap(),
        result.diagview(engine).unwrap()
    );
    provider.invalid_style.store(false, Ordering::Relaxed);
    assert_eq!(svd_calls.of(PINV_SVD), 0);
    assert_eq!(gemm_calls.of(Kernel::GEMM), 0);
    assert_eq!(gemm_calls.total(), 0);

    let complex: TensorMap<_, Complex64> = TensorMap::diagonal(
        engine,
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: Label::Vacuum,
                values: vec![Complex64::new(8.0, 0.0), Complex64::new(2.0, 0.0)],
            },
            SectorSpectrum {
                sector: Label::X,
                values: vec![
                    Complex64::new(3.0, 4.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                ],
            },
        ],
    )
    .unwrap();
    let complex_result = complex.pinv(engine, &[0], &[1], 0.25).unwrap();
    assert_eq!(
        complex_result.diagview(engine).unwrap(),
        [
            SectorSpectrum {
                sector: Label::Vacuum,
                values: vec![Complex64::new(0.125, 0.0), Complex64::new(0.0, 0.0)],
            },
            SectorSpectrum {
                sector: Label::X,
                values: vec![
                    Complex64::new(0.12, -0.16),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, 0.0)
                ],
            },
        ]
    );
    assert!(complex_result.dense_data().is_err());
    assert_eq!(svd_calls.of(PINV_SVD), 0);
    assert_eq!(gemm_calls.of(Kernel::GEMM), 0);
    assert_eq!(gemm_calls.total(), 0);

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
            values: vec![-2.0, 0.0],
        }],
    )
    .unwrap();
    let dual_result = dual.pinv(engine, &[0], &[1], 0.0).unwrap();
    assert_eq!(dual_result.codomain(), dual.domain());
    assert_eq!(dual_result.domain(), dual.codomain());
    assert_eq!(dual_result.diagview(engine).unwrap()[0].values, [-0.5, 0.0]);
    assert!(dual_result.dense_data().is_err());
    assert_eq!(svd_calls.of(PINV_SVD), 0);
    assert_eq!(gemm_calls.of(Kernel::GEMM), 0);
    assert_eq!(gemm_calls.total(), 0);
}
#[test]
fn checked_generic_pinv_normalized_empty_skips_dense_execution() {
    let engine = &mut crate::typed::CpuEngine::serial();
    let svd_calls = Arc::new(SpyCounts::default());
    let gemm_calls = Arc::clone(&svd_calls);
    let _dense = probe::inject_dense(Box::new(pinv_spy(&svd_calls, None, None)));
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0x61));
    let empty = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 0)]).unwrap();
    let source: TensorMap<_, f64> = TensorMap::zeros(engine, &runtime, [&empty], [&empty]).unwrap();

    assert!(source.dense_data().unwrap().is_empty());
    let pseudo = source.pinv(engine, &[0], &[1], 0.0).unwrap();
    assert!(pseudo.dense_data().unwrap().is_empty());
    assert_eq!(pseudo.codomain(), source.domain());
    assert_eq!(pseudo.domain(), source.codomain());
    assert!(std::ptr::eq(pseudo.provider(), provider.as_ref()));
    assert_eq!(svd_calls.of(PINV_SVD), 0);
    assert_eq!(gemm_calls.of(Kernel::GEMM), 0);
    assert_eq!(gemm_calls.total(), 0);
}
#[test]
fn checked_generic_pinv_stages_svd_and_gemm_failures_without_publication() {
    let engine = &mut crate::typed::CpuEngine::serial();
    for (fail_svd, fail_gemm, expected_svd, expected_gemm) in [
        (None, None, 2, 2),
        (Some(1), None, 1, 0),
        (Some(2), None, 2, 0),
        (None, Some(2), 2, 2),
    ] {
        let svd_calls = Arc::new(SpyCounts::default());
        let gemm_calls = Arc::clone(&svd_calls);
        let _dense = probe::inject_dense(Box::new(pinv_spy(&svd_calls, fail_svd, fail_gemm)));
        let runtime = Runtime::builder().build().unwrap();
        let provider = Arc::new(CheckedOnlyToy::new(0x62));
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
        let result = source.pinv(engine, &[0], &[1], 0.0);
        if fail_svd.is_some() || fail_gemm.is_some() {
            assert!(matches!(
                result,
                Err(GenericTensorError::Facade(symtn::typed::Error::Operation(
                    _
                )))
            ));
        } else {
            assert!(result.is_ok());
        }
        assert_eq!(svd_calls.of(PINV_SVD), expected_svd);
        assert_eq!(gemm_calls.of(Kernel::GEMM), expected_gemm);
        assert_eq!(svd_calls.total(), expected_svd + expected_gemm);
        assert_eq!(source.dense_data().unwrap(), before.as_slice());
    }
}
