use super::*;

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_compact_diagonal_eigh_vals_reads_stored_real_spectrum() {
    let engine = &mut crate::typed::CpuEngine::serial();
    use symtn_core::LieGroupFusionRule;

    let calls = Arc::new(SpyCounts::default());
    let _dense = probe::inject_dense(Box::new(eigh_vals_spy(&calls)));
    let runtime = Runtime::builder().build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(vec![0, 0], 2), (vec![1, 0], 3)]).unwrap();
    let input: TensorMap<_, Complex64> = TensorMap::diagonal(
        engine,
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![
                    Complex64::new(2.0, 0.0),
                    Complex64::new(-2.0, 0.0),
                    Complex64::new(1.0, 0.0),
                ],
            },
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![Complex64::new(-4.0, 0.0), Complex64::new(0.0, 0.0)],
            },
        ],
    )
    .unwrap();
    let densified = probe::record();
    let got = input
        .eigh_vals(engine, &[0], &[1], HermitianTol::DEFAULT)
        .unwrap();
    assert_eq!(
        got,
        vec![
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![-4.0, 0.0],
            },
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![-2.0, 1.0, 2.0],
            },
        ]
    );
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    assert_eq!(calls.total(), 0);

    // Changed leg roles permute first; the swap keeps the diagonal compact
    // (#1866), so the stored spectrum is read without a solver call.
    let swapped = input
        .eigh_vals(engine, &[1], &[0], HermitianTol::DEFAULT)
        .unwrap();
    assert_eq!(calls.total(), 0);
    let swapped_dense = input
        .permute(engine, &[1], &[0])
        .unwrap()
        .materialize(engine)
        .unwrap();
    assert_eq!(
        swapped,
        swapped_dense
            .eigh_vals(engine, &[0], &[1], HermitianTol::DEFAULT)
            .unwrap()
    );
    let dense = input.materialize(engine).unwrap();
    assert_eq!(
        got,
        dense
            .eigh_vals(engine, &[0], &[1], HermitianTol::DEFAULT)
            .unwrap()
    );

    let real: TensorMap<_, f64> = TensorMap::diagonal(
        engine,
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![2.0, -2.0, 1.0],
            },
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![-4.0, 0.0],
            },
        ],
    )
    .unwrap();
    let dense_real = real.materialize(engine).unwrap();
    let densified = probe::record();
    assert_eq!(
        real.eigh_vals(engine, &[0], &[1], HermitianTol::DEFAULT)
            .unwrap(),
        dense_real
            .eigh_vals(engine, &[0], &[1], HermitianTol::DEFAULT)
            .unwrap()
    );
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);

    let narrow: TensorMap<_, num_complex::Complex32> = TensorMap::diagonal(
        engine,
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![
                    num_complex::Complex32::new(2.0, 0.0),
                    num_complex::Complex32::new(-2.0, 0.0),
                    num_complex::Complex32::new(1.0, 0.0),
                ],
            },
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![
                    num_complex::Complex32::new(-4.0, 0.0),
                    num_complex::Complex32::new(0.0, 0.0),
                ],
            },
        ],
    )
    .unwrap();
    let dense_narrow = narrow.materialize(engine).unwrap();
    let densified = probe::record();
    assert_eq!(
        narrow
            .eigh_vals(engine, &[0], &[1], HermitianTol::DEFAULT)
            .unwrap(),
        dense_narrow
            .eigh_vals(engine, &[0], &[1], HermitianTol::DEFAULT)
            .unwrap()
    );
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    calls.reset();

    let dual_leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 0], 2)])
        .unwrap()
        .try_dual()
        .unwrap();
    let dual_sector = dual_leg.sectors().unwrap().remove(0);
    let dual: TensorMap<_, f32> = TensorMap::diagonal(
        engine,
        &runtime,
        &dual_leg,
        [SectorSpectrum {
            sector: dual_sector.clone(),
            values: vec![-2.0, 2.0],
        }],
    )
    .unwrap();
    let densified = probe::record();
    assert_eq!(
        dual.eigh_vals(engine, &[0], &[1], HermitianTol::DEFAULT)
            .unwrap(),
        vec![SectorSpectrum {
            sector: dual_sector,
            values: vec![-2.0, 2.0],
        }]
    );
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    assert_eq!(calls.total(), 0);
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_compact_diagonal_eigh_vals_rejects_inconsistent_spectrum_admission() {
    let engine = &mut crate::typed::CpuEngine::serial();
    use symtn_core::LieGroupFusionRule;

    let runtime = Runtime::builder().build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg = GradedSpace::try_new(provider, [(vec![0, 0], 2), (vec![1, 0], 1)]).unwrap();
    let input: TensorMap<_, f64> = TensorMap::diagonal(
        engine,
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![1.0, -1.0],
            },
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![2.0],
            },
        ],
    )
    .unwrap();
    let spectrum = input.spectrum().unwrap();
    // A spectrum that does not cover the bond is misuse: a typed error.
    let mut admit = |entries: &[symtn_matrixalgebra::SectorSpectrum<f64>]| {
        symtn_matrixalgebra::seam::eigh_vals_from_source::<CheckedGenericAdmissionMode, _, _, _, _>(
            engine,
            &mut symtn_dense::DefaultDenseExecutor::new(),
            symtn_matrixalgebra::seam::FactorSource::Diagonal {
                space: input.logical_space(),
                spectrum: entries,
            },
            HermitianTol::DEFAULT,
        )
    };
    assert!(admit(spectrum).is_ok());
    let mut reversed = spectrum.to_vec();
    reversed.reverse();
    assert!(admit(&reversed).is_ok());
    assert!(
        matches!(admit(&spectrum[..1]), Err(symtn_matrixalgebra::seam::CheckedGenericFactorPlanError::Operation(OperationError::InvalidArgument { message })) if message.contains("sector count"))
    );
    let mut duplicate = spectrum.to_vec();
    duplicate[1].sector = duplicate[0].sector;
    assert!(
        matches!(admit(&duplicate), Err(symtn_matrixalgebra::seam::CheckedGenericFactorPlanError::Operation(OperationError::InvalidArgument { message })) if message.contains("twice"))
    );
    let mut wrong_size = spectrum.to_vec();
    wrong_size[0].values.push(0.0);
    assert!(
        matches!(admit(&wrong_size), Err(symtn_matrixalgebra::seam::CheckedGenericFactorPlanError::Operation(OperationError::InvalidArgument { message })) if message.contains("length"))
    );
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_compact_diagonal_eigh_vals_applies_the_dense_hermiticity_check_directly() {
    let engine = &mut crate::typed::CpuEngine::serial();
    use symtn_core::LieGroupFusionRule;

    let calls = Arc::new(SpyCounts::default());
    let _dense = probe::inject_dense(Box::new(eigh_vals_spy(&calls)));
    let runtime = Runtime::builder().build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg = GradedSpace::try_new(provider, [(vec![0, 0], 1)]).unwrap();
    let near: TensorMap<_, Complex64> = TensorMap::diagonal(
        engine,
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: vec![0, 0],
            values: vec![Complex64::new(1.0, 1e-15)],
        }],
    )
    .unwrap();
    // The dense route's relative Hermiticity check, applied to the diagonal
    // directly: within tolerance the eigenvalue is the real part.
    let densified = probe::record();
    assert_eq!(
        near.eigh_vals(engine, &[0], &[1], HermitianTol::DEFAULT)
            .unwrap()[0]
            .values,
        [1.0]
    );
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    assert_eq!(calls.total(), 0);

    let nonfinite: TensorMap<_, f64> = TensorMap::diagonal(
        engine,
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: vec![0, 0],
            values: vec![f64::NAN],
        }],
    )
    .unwrap();
    // NaN is refused first by the shared finite-input stage (#1986).
    let densified = probe::record();
    let compact_error = nonfinite
        .eigh_vals(engine, &[0], &[1], HermitianTol::DEFAULT)
        .unwrap_err();
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    assert!(
        is_finite_input_refusal(&compact_error, "eigh"),
        "{compact_error:?}"
    );
}

#[cfg(feature = "cgc-gen")]
fn assert_same_complex_multisets<S: PartialEq + std::fmt::Debug>(
    actual: &[SectorSpectrum<S, Complex64>],
    expected: &[SectorSpectrum<S, Complex64>],
    tol: f64,
) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.sector, expected.sector);
        assert!(actual.values.windows(2).all(|pair| pair[0]
            .re
            .total_cmp(&pair[1].re)
            .then(pair[0].im.total_cmp(&pair[1].im))
            .is_le()));
        let sorted = |values: &[Complex64]| {
            let mut values = values.to_vec();
            values.sort_by(|a, b| a.re.total_cmp(&b.re).then(a.im.total_cmp(&b.im)));
            values
        };
        let (left, right) = (sorted(&actual.values), sorted(&expected.values));
        assert_eq!(left.len(), right.len());
        for (left, right) in left.iter().zip(&right) {
            assert!((left - right).norm() <= tol, "{left} != {right}");
        }
    }
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_compact_diagonal_eig_vals_reads_stored_complex_spectrum() {
    let engine = &mut crate::typed::CpuEngine::serial();
    use symtn_core::LieGroupFusionRule;

    let calls = Arc::new(SpyCounts::default());
    let _dense = probe::inject_dense(Box::new(eig_vals_spy(&calls)));
    let runtime = Runtime::builder().build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(vec![0, 0], 2), (vec![1, 0], 3)]).unwrap();
    let input: TensorMap<_, Complex64> = TensorMap::diagonal(
        engine,
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![
                    Complex64::new(1.0, 1.0),
                    Complex64::new(-3.0, 4.0),
                    Complex64::new(0.0, 0.5),
                ],
            },
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![Complex64::new(-0.25, 0.0), Complex64::new(2.0, -2.0)],
            },
        ],
    )
    .unwrap();
    let saved = input.diagview(engine).unwrap();
    let densified = probe::record();
    let got = input.eig_vals(engine, &[0], &[1]).unwrap();
    assert_eq!(
        got,
        vec![
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![Complex64::new(-0.25, 0.0), Complex64::new(2.0, -2.0)],
            },
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![
                    Complex64::new(-3.0, 4.0),
                    Complex64::new(0.0, 0.5),
                    Complex64::new(1.0, 1.0),
                ],
            },
        ]
    );
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    assert_eq!(calls.total(), 0);
    assert_eq!(input.diagview(engine).unwrap(), saved);

    let dense = input.materialize(engine).unwrap();
    assert_same_complex_multisets(&got, &dense.eig_vals(engine, &[0], &[1]).unwrap(), 1e-12);
    calls.reset();

    // The swap keeps the diagonal compact (#1866): no solver call.
    let swapped = input.eig_vals(engine, &[1], &[0]).unwrap();
    assert_eq!(calls.total(), 0);
    let swapped_dense = input
        .permute(engine, &[1], &[0])
        .unwrap()
        .materialize(engine)
        .unwrap();
    assert_same_complex_multisets(
        &swapped,
        &swapped_dense.eig_vals(engine, &[0], &[1]).unwrap(),
        1e-12,
    );

    calls.reset();
    let lazy = dense.adjoint(engine).unwrap();
    assert!(matches!(lazy.repr, TypedTensorRepr::Adjoint(_)));
    // Materialized for the call: the values of `B^H` are `conj` of those of
    // `B` only as a multiset, so the order follows the logical input.
    assert_eq!(
        lazy.eig_vals(engine, &[0], &[1]).unwrap(),
        lazy.materialize(engine)
            .unwrap()
            .eig_vals(engine, &[0], &[1])
            .unwrap()
    );
    assert!(matches!(lazy.repr, TypedTensorRepr::Adjoint(_)));
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_compact_diagonal_eig_vals_widens_after_payload_rounding() {
    let engine = &mut crate::typed::CpuEngine::serial();
    use symtn_core::LieGroupFusionRule;

    let calls = Arc::new(SpyCounts::default());
    let _dense = probe::inject_dense(Box::new(eig_vals_spy(&calls)));
    let runtime = Runtime::builder().build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(vec![0, 0], 2), (vec![1, 0], 2)]).unwrap();
    let c = |re: f64, im: f64| Complex64::new(re, im);
    let r32 = |value: f32| f64::from(value);

    let real: TensorMap<_, f32> = TensorMap::diagonal(
        engine,
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![0.1, -0.7],
            },
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![1.3, 0.2],
            },
        ],
    )
    .unwrap();
    let densified = probe::record();
    let got = real.eig_vals(engine, &[0], &[1]).unwrap();
    assert_eq!(
        got,
        vec![
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![c(r32(-0.7), 0.0), c(r32(0.1), 0.0)],
            },
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![c(r32(0.2), 0.0), c(r32(1.3), 0.0)],
            },
        ]
    );
    assert_ne!(got[0].values[1].re, 0.1);
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    let dense = real
        .materialize(engine)
        .unwrap()
        .eig_vals(engine, &[0], &[1])
        .unwrap();
    assert_same_complex_multisets(&got, &dense, 1e-6);

    let real64: TensorMap<_, f64> = TensorMap::diagonal(
        engine,
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![0.1, -0.7],
            },
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![1.3, 0.2],
            },
        ],
    )
    .unwrap();
    let densified = probe::record();
    let got = real64.eig_vals(engine, &[0], &[1]).unwrap();
    assert_eq!(got[0].values, vec![c(-0.7, 0.0), c(0.1, 0.0)]);
    assert_eq!(got[1].values, vec![c(0.2, 0.0), c(1.3, 0.0)]);
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    let dense = real64
        .materialize(engine)
        .unwrap()
        .eig_vals(engine, &[0], &[1])
        .unwrap();
    assert_same_complex_multisets(&got, &dense, 1e-12);

    let narrow: TensorMap<_, num_complex::Complex32> = TensorMap::diagonal(
        engine,
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![
                    num_complex::Complex32::new(0.1, -0.3),
                    num_complex::Complex32::new(-0.7, 0.0),
                ],
            },
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![
                    num_complex::Complex32::new(0.0, 1.3),
                    num_complex::Complex32::new(0.2, 0.2),
                ],
            },
        ],
    )
    .unwrap();
    let densified = probe::record();
    let got = narrow.eig_vals(engine, &[0], &[1]).unwrap();
    assert_eq!(
        got,
        vec![
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![c(r32(-0.7), 0.0), c(r32(0.1), r32(-0.3))],
            },
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![c(0.0, r32(1.3)), c(r32(0.2), r32(0.2))],
            },
        ]
    );
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    let dense = narrow
        .materialize(engine)
        .unwrap()
        .eig_vals(engine, &[0], &[1])
        .unwrap();
    assert_same_complex_multisets(&got, &dense, 1e-6);
    assert_eq!(calls.total(), 6);
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_compact_diagonal_eig_vals_keeps_ties_and_dual_bond() {
    let engine = &mut crate::typed::CpuEngine::serial();
    use symtn_core::LieGroupFusionRule;

    let calls = Arc::new(SpyCounts::default());
    let _dense = probe::inject_dense(Box::new(eig_vals_spy(&calls)));
    let runtime = Runtime::builder().build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let c = |re: f64, im: f64| Complex64::new(re, im);
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 4)]).unwrap();
    let ties: TensorMap<_, Complex64> = TensorMap::diagonal(
        engine,
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: vec![1, 1],
            values: vec![c(3.0, 0.0), c(0.0, -3.0), c(-3.0, 0.0), c(1.0, 0.0)],
        }],
    )
    .unwrap();
    let densified = probe::record();
    let got = ties.eig_vals(engine, &[0], &[1]).unwrap();
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    assert_eq!(calls.total(), 0);
    assert_same_complex_multisets(
        &got,
        &[SectorSpectrum {
            sector: vec![1, 1],
            values: vec![c(-3.0, 0.0), c(0.0, -3.0), c(3.0, 0.0), c(1.0, 0.0)],
        }],
        0.0,
    );
    assert_same_complex_multisets(
        &got,
        &ties
            .materialize(engine)
            .unwrap()
            .eig_vals(engine, &[0], &[1])
            .unwrap(),
        1e-12,
    );
    calls.reset();

    let dual_leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 0], 2)])
        .unwrap()
        .try_dual()
        .unwrap();
    let dual_sector = dual_leg.sectors().unwrap().remove(0);
    let dual: TensorMap<_, f64> = TensorMap::diagonal(
        engine,
        &runtime,
        &dual_leg,
        [SectorSpectrum {
            sector: dual_sector.clone(),
            values: vec![-2.0, 5.0],
        }],
    )
    .unwrap();
    let densified = probe::record();
    let got = dual.eig_vals(engine, &[0], &[1]).unwrap();
    assert_eq!(
        got,
        vec![SectorSpectrum {
            sector: dual_sector,
            values: vec![c(-2.0, 0.0), c(5.0, 0.0)],
        }]
    );
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    assert_eq!(calls.total(), 0);
    assert_same_complex_multisets(
        &got,
        &dual
            .materialize(engine)
            .unwrap()
            .eig_vals(engine, &[0], &[1])
            .unwrap(),
        1e-12,
    );
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_compact_diagonal_eig_vals_rejects_inconsistent_spectrum_admission() {
    let engine = &mut crate::typed::CpuEngine::serial();
    use symtn_core::LieGroupFusionRule;

    let runtime = Runtime::builder().build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg = GradedSpace::try_new(provider, [(vec![0, 0], 2), (vec![1, 0], 1)]).unwrap();
    let input: TensorMap<_, Complex64> = TensorMap::diagonal(
        engine,
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![Complex64::new(1.0, 2.0), Complex64::new(-1.0, 0.0)],
            },
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![Complex64::new(0.0, 2.0)],
            },
        ],
    )
    .unwrap();
    let spectrum = input.spectrum().unwrap();
    // A spectrum that does not cover the bond is misuse: a typed error.
    let mut admit = |entries: &[symtn_matrixalgebra::SectorSpectrum<Complex64>]| {
        symtn_matrixalgebra::seam::eig_vals_from_source::<CheckedGenericAdmissionMode, _, _, _, _>(
            engine,
            &mut symtn_dense::DefaultDenseExecutor::new(),
            symtn_matrixalgebra::seam::FactorSource::Diagonal {
                space: input.logical_space(),
                spectrum: entries,
            },
        )
    };
    assert!(admit(spectrum).is_ok());
    let mut reversed = spectrum.to_vec();
    reversed.reverse();
    assert!(admit(&reversed).is_ok());
    assert!(
        matches!(admit(&spectrum[..1]), Err(symtn_matrixalgebra::seam::CheckedGenericFactorPlanError::Operation(OperationError::InvalidArgument { message })) if message.contains("sector count"))
    );
    let mut duplicate = spectrum.to_vec();
    duplicate[1].sector = duplicate[0].sector;
    assert!(
        matches!(admit(&duplicate), Err(symtn_matrixalgebra::seam::CheckedGenericFactorPlanError::Operation(OperationError::InvalidArgument { message })) if message.contains("twice"))
    );
    let mut wrong_size = spectrum.to_vec();
    wrong_size[0].values.push(Complex64::new(0.0, 0.0));
    assert!(
        matches!(admit(&wrong_size), Err(symtn_matrixalgebra::seam::CheckedGenericFactorPlanError::Operation(OperationError::InvalidArgument { message })) if message.contains("length"))
    );
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_compact_diagonal_eig_vals_refuses_nonfinite_values() {
    let engine = &mut crate::typed::CpuEngine::serial();
    use symtn_core::LieGroupFusionRule;

    let calls = Arc::new(SpyCounts::default());
    let _dense = probe::inject_dense(Box::new(eig_vals_spy(&calls)));
    let runtime = Runtime::builder().build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg = GradedSpace::try_new(provider, [(vec![0, 0], 1), (vec![1, 0], 1)]).unwrap();
    for bad in [
        Complex64::new(f64::NAN, 0.0),
        Complex64::new(0.0, f64::INFINITY),
        Complex64::new(f64::MAX, f64::MAX),
    ] {
        let input: TensorMap<_, Complex64> = TensorMap::diagonal(
            engine,
            &runtime,
            &leg,
            [
                SectorSpectrum {
                    sector: vec![0, 0],
                    values: vec![Complex64::new(1.0, 0.0)],
                },
                SectorSpectrum {
                    sector: vec![1, 0],
                    values: vec![bad],
                },
            ],
        )
        .unwrap();
        // A nonfinite value is refused by the shared finite-input stage, and
        // the overflowing `MAX + MAX i` by the dense route's eigenvalue check;
        // no dense solver either way.
        calls.reset();
        let densified = probe::record();
        let error = input.eig_vals(engine, &[0], &[1]).unwrap_err();
        assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
        assert_eq!(calls.total(), 0);
        let operation = match &error {
            GenericTensorError::Facade(Error::Operation(error)) => Some(&**error),
            GenericTensorError::Plan(CheckedGenericPlanError::Operation(error)) => Some(error),
            _ => None,
        };
        let expected_kind = if bad.re.is_finite() && bad.im.is_finite() {
            matches!(
                operation,
                Some(OperationError::Dense(
                    symtn_dense::DenseError::NumericalFailure { .. }
                ))
            )
        } else {
            matches!(operation, Some(OperationError::InvalidArgument { .. }))
        };
        assert!(expected_kind, "{error:?}");
    }
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_compact_diagonal_eigh_full_avoids_input_materialization_and_solver() {
    let engine = &mut crate::typed::CpuEngine::serial();
    use symtn_core::LieGroupFusionRule;

    let calls = Arc::new(SpyCounts::default());
    let _dense = probe::inject_dense(Box::new(eigh_full_spy(&calls)));
    let runtime = Runtime::builder().build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(vec![0, 0], 2), (vec![1, 0], 3)]).unwrap();
    macro_rules! check {
        ($engine:ident, $dtype:ty, $from:expr) => {{
            let from: fn(f64) -> $dtype = $from;
            let input: TensorMap<_, $dtype> = TensorMap::diagonal(
                $engine,
                &runtime,
                &leg,
                [
                    SectorSpectrum {
                        sector: vec![1, 0],
                        values: vec![from(2.0), from(-2.0), from(1.0)],
                    },
                    SectorSpectrum {
                        sector: vec![0, 0],
                        values: vec![from(0.0), from(-4.0)],
                    },
                ],
            )
            .unwrap();
            let densified = probe::record();
            calls.reset();
            let Eigh { d, v } = input
                .eigh_full($engine, &[0], &[1], HermitianTol::DEFAULT)
                .unwrap();
            assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
            assert_eq!(calls.total(), 0);
            assert!(std::ptr::eq(d.provider(), provider.as_ref()));
            assert!(std::ptr::eq(v.provider(), provider.as_ref()));
            assert!(d.dense_data().is_err());
            assert_eq!(
                d.diagview($engine).unwrap(),
                vec![
                    SectorSpectrum {
                        sector: vec![0, 0],
                        values: vec![from(-4.0), from(0.0)],
                    },
                    SectorSpectrum {
                        sector: vec![1, 0],
                        values: vec![from(-2.0), from(1.0), from(2.0)],
                    },
                ]
            );
            assert_eq!(v.codomain(), input.codomain());
            assert!(!v.domain()[0].is_dual());
            // Column j of each sector block is the unit vector of the stored
            // index that sorts into position j.
            let expected: TensorMap<_, $dtype> = TensorMap::from_subblock_fn(
                $engine,
                &runtime,
                v.codomain().iter(),
                v.domain().iter(),
                |trees, ij| {
                    let order: &[usize] = if trees.coupled() == &vec![0, 0] {
                        &[1, 0]
                    } else {
                        &[1, 2, 0]
                    };
                    from(f64::from(ij[0] == order[ij[1]]))
                },
            )
            .unwrap();
            assert_eq!(v.dense_data().unwrap(), expected.dense_data().unwrap());
        }};
    }
    check!(engine, f64, |x| x);
    check!(engine, Complex64, |x| Complex64::new(x, 0.0));
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_compact_diagonal_eigh_full_matches_independent_dense_oracle() {
    let engine = &mut crate::typed::CpuEngine::serial();
    use symtn_core::LieGroupFusionRule;

    let runtime = Runtime::builder().build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(vec![0, 0], 2), (vec![1, 0], 3)]).unwrap();
    let dual_leg = leg.try_dual().unwrap();
    let dual_sectors = dual_leg.sectors().unwrap();
    macro_rules! check {
        ($engine:ident, $dtype:ty, $from:expr, $tol:expr) => {{
            let from: fn(f64) -> $dtype = $from;
            let tol: f64 = $tol;
            let distance =
                |$engine: &mut dyn Engine, a: &TensorMap<_, $dtype>, b: &TensorMap<_, $dtype>| {
                    a.axpby($engine, from(1.0), b, from(-1.0))
                        .unwrap()
                        .norm($engine, 2.0)
                        .unwrap()
                };
            let build = |$engine: &mut dyn Engine,
                         space: &GradedSpace<_>,
                         sectors: [Vec<i64>; 2],
                         values: [Vec<f64>; 2]| {
                let [first, second] = values;
                let [s0, s1] = sectors;
                TensorMap::<_, $dtype>::diagonal(
                    $engine,
                    &runtime,
                    space,
                    [
                        SectorSpectrum {
                            sector: s0,
                            values: first.into_iter().map(from).collect(),
                        },
                        SectorSpectrum {
                            sector: s1,
                            values: second.into_iter().map(from).collect(),
                        },
                    ],
                )
                .unwrap()
            };
            let distinct = [vec![-2.0, 5.0], vec![3.0, -1.0, 0.5]];
            for (space, sectors) in [
                (&leg, [vec![0, 0], vec![1, 0]]),
                (
                    &dual_leg,
                    [dual_sectors[0].clone(), dual_sectors[1].clone()],
                ),
            ] {
                let input = build($engine, space, sectors, distinct.clone());
                // Oracle: an independently materialized dense payload through
                // the dense checked EIGH route.
                let dense = input.materialize($engine).unwrap();
                let expected = dense
                    .eigh_full($engine, &[0], &[1], HermitianTol::DEFAULT)
                    .unwrap();
                let densified = probe::record();
                let got = input
                    .eigh_full($engine, &[0], &[1], HermitianTol::DEFAULT)
                    .unwrap();
                assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
                assert_eq!(got.v.codomain(), expected.v.codomain());
                assert_eq!(got.v.domain(), expected.v.domain());
                assert_eq!(got.d.codomain(), expected.d.codomain());
                for (actual, oracle) in got
                    .d
                    .diagview($engine)
                    .unwrap()
                    .iter()
                    .zip(expected.d.diagview($engine).unwrap())
                {
                    assert_eq!(actual.sector, oracle.sector);
                    for (&a, &b) in actual.values.iter().zip(&oracle.values) {
                        assert!(ScalarOps::abs_value(a - b) <= tol);
                    }
                }
                assert!(distance($engine, &got.v, &expected.v) <= tol);

                // Changed leg roles go through an explicit permute.
                let swapped = input
                    .eigh_full($engine, &[1], &[0], HermitianTol::DEFAULT)
                    .unwrap();
                let swapped_oracle = input
                    .permute($engine, &[1], &[0])
                    .unwrap()
                    .materialize($engine)
                    .unwrap()
                    .eigh_full($engine, &[0], &[1], HermitianTol::DEFAULT)
                    .unwrap();
                assert_eq!(
                    swapped.d.diagview($engine).unwrap().len(),
                    swapped_oracle.d.diagview($engine).unwrap().len()
                );
                for (actual, oracle) in swapped
                    .d
                    .diagview($engine)
                    .unwrap()
                    .iter()
                    .zip(swapped_oracle.d.diagview($engine).unwrap())
                {
                    assert_eq!(actual.sector, oracle.sector);
                    for (&a, &b) in actual.values.iter().zip(&oracle.values) {
                        assert!(ScalarOps::abs_value(a - b) <= tol);
                    }
                }
                assert!(distance($engine, &swapped.v, &swapped_oracle.v) <= tol);
            }

            // Ties: exact equal values and ±x. Degenerate bases are not a
            // public guarantee, so check the factorization itself.
            let input = build(
                $engine,
                &leg,
                [vec![0, 0], vec![1, 0]],
                [vec![1.0, 1.0], vec![2.0, -2.0, 2.0]],
            );
            let dense = input.materialize($engine).unwrap();
            let oracle = dense
                .eigh_full($engine, &[0], &[1], HermitianTol::DEFAULT)
                .unwrap();
            let densified = probe::record();
            let Eigh { d, v } = input
                .eigh_full($engine, &[0], &[1], HermitianTol::DEFAULT)
                .unwrap();
            assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
            assert_eq!(
                d.diagview($engine).unwrap(),
                vec![
                    SectorSpectrum {
                        sector: vec![0, 0],
                        values: vec![from(1.0), from(1.0)],
                    },
                    SectorSpectrum {
                        sector: vec![1, 0],
                        values: vec![from(-2.0), from(2.0), from(2.0)],
                    },
                ]
            );
            let multiset = |spectrum: Vec<SectorSpectrum<Vec<i64>, $dtype>>| {
                spectrum
                    .into_iter()
                    .map(|entry| {
                        let mut values: Vec<f64> = entry
                            .values
                            .iter()
                            .map(|value| FactorScalar::widen_complex(*value).re)
                            .collect();
                        values.sort_by(f64::total_cmp);
                        (entry.sector, values)
                    })
                    .collect::<Vec<_>>()
            };
            let got_multiset = multiset(d.diagview($engine).unwrap());
            for ((sector, a), (oracle_sector, b)) in got_multiset
                .iter()
                .zip(multiset(oracle.d.diagview($engine).unwrap()))
            {
                assert_eq!(*sector, oracle_sector);
                for (a, b) in a.iter().zip(&b) {
                    assert!((a - b).abs() <= tol);
                }
            }
            let v_adjoint = &v.adjoint($engine).unwrap().materialize($engine).unwrap();
            let rebuilt = v
                .compose($engine, &d)
                .unwrap()
                .compose($engine, v_adjoint)
                .unwrap();
            assert!(distance($engine, &rebuilt, &dense) <= tol);
            let gram = v
                .adjoint($engine)
                .unwrap()
                .materialize($engine)
                .unwrap()
                .compose($engine, &v)
                .unwrap();
            let identity: TensorMap<_, $dtype> = TensorMap::from_subblock_fn(
                $engine,
                &runtime,
                gram.codomain().iter(),
                gram.domain().iter(),
                |_, ij| from(f64::from(ij[0] == ij[1])),
            )
            .unwrap();
            assert!(distance($engine, &gram, &identity) <= tol);
        }};
    }
    check!(engine, f32, |x| x as f32, 1e-5);
    check!(engine, f64, |x| x, 1e-12);
    check!(
        engine,
        num_complex::Complex32,
        |x| num_complex::Complex32::new(x as f32, 0.0),
        1e-5
    );
    check!(engine, Complex64, |x| Complex64::new(x, 0.0), 1e-12);
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_compact_diagonal_eig_full_avoids_input_materialization_and_solver() {
    let engine = &mut crate::typed::CpuEngine::serial();
    use symtn_core::LieGroupFusionRule;

    let calls = Arc::new(SpyCounts::default());
    let svd_vals_calls = Arc::clone(&calls);
    let _dense = probe::inject_dense(Box::new(eig_full_spy(&calls)));
    let runtime = Runtime::builder().build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(vec![0, 0], 2), (vec![1, 0], 3)]).unwrap();
    let c = |re: f64, im: f64| Complex64::new(re, im);
    macro_rules! check {
        ($engine:ident, $dtype:ty, $from:expr, [$($a:expr),*], [$($b:expr),*], [$($da:expr),*], [$($db:expr),*], $order:expr) => {{
            let from: fn(Complex64) -> $dtype = $from;
            let input: TensorMap<_, $dtype> = TensorMap::diagonal($engine,
                &runtime,
                &leg,
                [
                    SectorSpectrum {
                        sector: vec![1, 0],
                        values: vec![$(from($b)),*],
                    },
                    SectorSpectrum {
                        sector: vec![0, 0],
                        values: vec![$(from($a)),*],
                    },
                ],
            )
            .unwrap();
            let densified = probe::record();
            calls.reset();
            svd_vals_calls.reset();
            let Eig { d, v } = input.eig_full($engine, &[0], &[1]).unwrap();
            assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
            assert_eq!(calls.of(EIG_FULL), 0);
            assert_eq!(svd_vals_calls.get(Kernel::SvdVals), 0);
            assert!(std::ptr::eq(d.provider(), provider.as_ref()));
            assert!(std::ptr::eq(v.provider(), provider.as_ref()));
            assert!(d.dense_data().is_err());
            assert_eq!(
                d.diagview($engine).unwrap(),
                vec![
                    SectorSpectrum {
                        sector: vec![0, 0],
                        values: vec![$($da),*],
                    },
                    SectorSpectrum {
                        sector: vec![1, 0],
                        values: vec![$($db),*],
                    },
                ]
            );
            assert_eq!(v.codomain(), input.codomain());
            assert!(!v.domain()[0].is_dual());
            // Column j of each sector block is the unit vector of the stored
            // index that sorts into position j.
            let expected: TensorMap<_, Complex64> = TensorMap::from_subblock_fn($engine,
                &runtime,
                v.codomain().iter(),
                v.domain().iter(),
                |trees, ij| {
                    let order: &[usize] = if trees.coupled() == &vec![0, 0] {
                        &[0, 1]
                    } else {
                        &$order
                    };
                    c(f64::from(ij[0] == order[ij[1]]), 0.0)
                },
            )
            .unwrap();
            assert_eq!(v.dense_data().unwrap(), expected.dense_data().unwrap());

            // Changed leg roles permute first, and the swap keeps the
            // diagonal compact (#1866): no materialization and no EIG, as in
            // the multiplicity-free mode.
            let densified = probe::record();
            input.eig_full($engine, &[1], &[0]).unwrap();
            assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
            assert_eq!(calls.of(EIG_FULL), 0);
            assert_eq!(svd_vals_calls.get(Kernel::SvdVals), 0);
        }};
    }
    // Lexicographic (re, im): sector (1,0) -2+0.5i < -5i < 1+i, whose
    // magnitude order (5 > |-2+0.5i| > |1+i|) differs.
    check!(
        engine,
        Complex64,
        |z| z,
        [c(-4.0, 0.0), c(0.0, 0.25)],
        [c(1.0, 1.0), c(0.0, -5.0), c(-2.0, 0.5)],
        [c(-4.0, 0.0), c(0.0, 0.25)],
        [c(-2.0, 0.5), c(0.0, -5.0), c(1.0, 1.0)],
        [2, 1, 0]
    );
    check!(
        engine,
        f64,
        |z| z.re,
        [c(-4.0, 0.0), c(0.25, 0.0)],
        [c(1.0, 0.0), c(-5.0, 0.0), c(-2.0, 0.0)],
        [c(-4.0, 0.0), c(0.25, 0.0)],
        [c(-5.0, 0.0), c(-2.0, 0.0), c(1.0, 0.0)],
        [1, 2, 0]
    );
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_compact_diagonal_eig_full_matches_independent_dense_oracle() {
    let engine = &mut crate::typed::CpuEngine::serial();
    use symtn_core::LieGroupFusionRule;

    let runtime = Runtime::builder().build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(vec![0, 0], 2), (vec![1, 0], 3)]).unwrap();
    let dual_leg = leg.try_dual().unwrap();
    let dual_sectors = dual_leg.sectors().unwrap();
    macro_rules! check {
        ($engine:ident, $dtype:ty, $eig:ty, $from:expr, $complex:expr, $tol:expr) => {{
            // Payload values are rounded to `$dtype` first; `$eig` widening
            // is exact afterwards.
            let from: fn(f64, f64) -> $dtype = $from;
            let eig = |value: $dtype| -> $eig {
                <$eig as FactorScalar>::from_complex64(FactorScalar::widen_complex(value))
            };
            let tol: f64 = $tol;
            let im = if $complex { 1.0 } else { 0.0 };
            let distance =
                |$engine: &mut dyn Engine, a: &TensorMap<_, $eig>, b: &TensorMap<_, $eig>| {
                    a.axpby($engine, <$eig>::new(1.0, 0.0), b, <$eig>::new(-1.0, 0.0))
                        .unwrap()
                        .norm($engine, 2.0)
                        .unwrap()
                };
            let build = |$engine: &mut dyn Engine,
                         space: &GradedSpace<_>,
                         sectors: [Vec<i64>; 2],
                         values: [Vec<(f64, f64)>; 2]| {
                let [first, second] = values;
                let [s0, s1] = sectors;
                TensorMap::<_, $dtype>::diagonal(
                    $engine,
                    &runtime,
                    space,
                    [
                        SectorSpectrum {
                            sector: s0,
                            values: first.into_iter().map(|(re, i)| from(re, i * im)).collect(),
                        },
                        SectorSpectrum {
                            sector: s1,
                            values: second.into_iter().map(|(re, i)| from(re, i * im)).collect(),
                        },
                    ],
                )
                .unwrap()
            };
            let assert_d = |$engine: &mut dyn Engine,
                            got: &TensorMap<_, $eig>,
                            oracle: &TensorMap<_, $eig>| {
                let (got, oracle) = (
                    got.diagview($engine).unwrap(),
                    oracle.diagview($engine).unwrap(),
                );
                assert_eq!(got.len(), oracle.len());
                for (actual, oracle) in got.iter().zip(&oracle) {
                    assert_eq!(actual.sector, oracle.sector);
                    assert_eq!(actual.values.len(), oracle.values.len());
                    for (&a, &b) in actual.values.iter().zip(&oracle.values) {
                        assert!(ScalarOps::abs_value(a - b) <= tol, "{a} != {b}");
                    }
                }
            };
            // The published order, sorted here independently: lexicographic
            // (re, im) of the widened stored values, stable on ties.
            let lexicographic = |stored: &[_]| {
                let mut values: Vec<$eig> = stored.iter().map(|&value| eig(value)).collect();
                values.sort_by(|x, y| {
                    let (x, y) = (x.widen_complex(), y.widen_complex());
                    (x.re, x.im).partial_cmp(&(y.re, y.im)).unwrap()
                });
                values
            };
            // Distinct magnitudes: |0.1-0.3i|<0.7 and 0.2(+0.2i)<1.3(i)<2.9.
            let distinct = [
                vec![(0.1, -0.3), (-0.7, 0.0)],
                vec![(0.0, 1.3), (0.2, 0.2), (-2.9, 0.1)],
            ];
            let distinct = if $complex {
                distinct
            } else {
                [
                    vec![(0.1, 0.0), (-0.7, 0.0)],
                    vec![(1.3, 0.0), (0.2, 0.0), (-2.9, 0.0)],
                ]
            };
            for (space, sectors, dual) in [
                (&leg, [vec![0, 0], vec![1, 0]], false),
                (
                    &dual_leg,
                    [dual_sectors[0].clone(), dual_sectors[1].clone()],
                    true,
                ),
            ] {
                let input = build($engine, space, sectors, distinct.clone());
                // Oracle: an independently materialized dense payload through
                // the dense checked EIG route.
                let dense = input.materialize($engine).unwrap();
                let expected = dense.eig_full($engine, &[0], &[1]).unwrap();
                let densified = probe::record();
                let got = input.eig_full($engine, &[0], &[1]).unwrap();
                assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
                assert_eq!(got.v.codomain(), expected.v.codomain());
                assert_eq!(got.v.domain(), expected.v.domain());
                assert_eq!(got.d.codomain(), expected.d.codomain());
                assert_d($engine, &got.d, &expected.d);
                assert!(distance($engine, &got.v, &expected.v) <= tol);
                if !dual {
                    // Hand values: the rounded payload, widened exactly.
                    let stored = input.diagview($engine).unwrap();
                    let by = |sector: &[i64]| {
                        stored
                            .iter()
                            .find(|entry| entry.sector == sector)
                            .unwrap()
                            .values
                            .clone()
                    };
                    let (a, b) = (by(&[0, 0]), by(&[1, 0]));
                    assert_eq!(
                        got.d.diagview($engine).unwrap(),
                        vec![
                            SectorSpectrum {
                                sector: vec![0, 0],
                                values: lexicographic(&a),
                            },
                            SectorSpectrum {
                                sector: vec![1, 0],
                                values: lexicographic(&b),
                            },
                        ]
                    );
                }

                // Changed leg roles go through an explicit permute.
                let swapped = input.eig_full($engine, &[1], &[0]).unwrap();
                let swapped_oracle = input
                    .permute($engine, &[1], &[0])
                    .unwrap()
                    .materialize($engine)
                    .unwrap()
                    .eig_full($engine, &[0], &[1])
                    .unwrap();
                assert_d($engine, &swapped.d, &swapped_oracle.d);
                assert!(distance($engine, &swapped.v, &swapped_oracle.v) <= tol);
            }

            // Equal-magnitude ties (±2, conjugate-free imaginary ties) follow
            // the published order. Degenerate bases are not a public
            // guarantee, so check the factorization itself.
            let ties = if $complex {
                [
                    vec![(2.0, 0.0), (0.0, -2.0)],
                    vec![(-3.0, 0.0), (1.0, 0.0), (0.0, 3.0)],
                ]
            } else {
                [
                    vec![(2.0, 0.0), (-2.0, 0.0)],
                    vec![(-3.0, 0.0), (1.0, 0.0), (3.0, 0.0)],
                ]
            };
            let input = build($engine, &leg, [vec![0, 0], vec![1, 0]], ties);
            let stored = input.diagview($engine).unwrap();
            let by = |sector: &[i64]| {
                stored
                    .iter()
                    .find(|entry| entry.sector == sector)
                    .unwrap()
                    .values
                    .clone()
            };
            let dense = input.materialize($engine).unwrap();
            let oracle = dense.eig_full($engine, &[0], &[1]).unwrap();
            let densified = probe::record();
            let Eig { d, v } = input.eig_full($engine, &[0], &[1]).unwrap();
            assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
            let (a, b) = (by(&[0, 0]), by(&[1, 0]));
            assert_eq!(
                d.diagview($engine).unwrap(),
                vec![
                    SectorSpectrum {
                        sector: vec![0, 0],
                        values: lexicographic(&a),
                    },
                    SectorSpectrum {
                        sector: vec![1, 0],
                        values: lexicographic(&b),
                    },
                ]
            );
            let widen = |spectrum: Vec<SectorSpectrum<Vec<i64>, $eig>>| {
                spectrum
                    .into_iter()
                    .map(|entry| SectorSpectrum {
                        sector: entry.sector,
                        values: entry
                            .values
                            .into_iter()
                            .map(FactorScalar::widen_complex)
                            .collect(),
                    })
                    .collect::<Vec<_>>()
            };
            assert_same_complex_multisets(
                &widen(d.diagview($engine).unwrap()),
                &widen(oracle.d.diagview($engine).unwrap()),
                tol,
            );
            let v_inv = &v.inv($engine, &[0], &[1]).unwrap();
            let rebuilt = v
                .compose($engine, &d)
                .unwrap()
                .compose($engine, v_inv)
                .unwrap();
            let widened_input = TensorMap::<_, $eig>::diagonal(
                $engine,
                &runtime,
                &leg,
                stored.iter().map(|entry| SectorSpectrum {
                    sector: entry.sector.clone(),
                    values: entry.values.iter().map(|&value| eig(value)).collect(),
                }),
            )
            .unwrap()
            .materialize($engine)
            .unwrap();
            assert!(distance($engine, &rebuilt, &widened_input) <= tol);
        }};
    }
    check!(
        engine,
        f32,
        num_complex::Complex32,
        |re, _| re as f32,
        false,
        1e-5
    );
    check!(engine, f64, Complex64, |re, _| re, false, 1e-12);
    check!(
        engine,
        num_complex::Complex32,
        num_complex::Complex32,
        |re, im| num_complex::Complex32::new(re as f32, im as f32),
        true,
        1e-5
    );
    check!(engine, Complex64, Complex64, Complex64::new, true, 1e-12);
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_compact_diagonal_eig_full_refuses_nonfinite_values() {
    let engine = &mut crate::typed::CpuEngine::serial();
    use symtn_core::LieGroupFusionRule;

    let calls = Arc::new(SpyCounts::default());
    let _dense = probe::inject_dense(Box::new(eig_full_spy(&calls)));
    let runtime = Runtime::builder().build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg = GradedSpace::try_new(provider, [(vec![0, 0], 1), (vec![1, 0], 2)]).unwrap();
    for bad in [
        Complex64::new(f64::NAN, 0.0),
        Complex64::new(0.0, f64::INFINITY),
        Complex64::new(f64::MAX, f64::MAX),
    ] {
        let input: TensorMap<_, Complex64> = TensorMap::diagonal(
            engine,
            &runtime,
            &leg,
            [
                SectorSpectrum {
                    sector: vec![0, 0],
                    values: vec![Complex64::new(1.0, 0.0)],
                },
                SectorSpectrum {
                    sector: vec![1, 0],
                    values: vec![bad, Complex64::new(3.0, 0.0)],
                },
            ],
        )
        .unwrap();
        // The shared finite-input stage (the dense route's own checked
        // pre-check, same error), or the dense eigenvalue check for the
        // overflowing `MAX + MAX i`; no dense solver either way.
        let dense_error = input
            .materialize(engine)
            .unwrap()
            .eig_full(engine, &[0], &[1])
            .map(drop)
            .unwrap_err();
        calls.reset();
        let densified = probe::record();
        let compact_error = input.eig_full(engine, &[0], &[1]).map(drop).unwrap_err();
        assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
        assert_eq!(calls.total(), 0);
        assert_eq!(format!("{compact_error:?}"), format!("{dense_error:?}"));
    }

    let nonfinite: TensorMap<_, f64> = TensorMap::diagonal(
        engine,
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![1.0],
            },
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![f64::NAN, 2.0],
            },
        ],
    )
    .unwrap();
    let dense_error = nonfinite
        .materialize(engine)
        .unwrap()
        .eig_full(engine, &[0], &[1])
        .map(drop)
        .unwrap_err();
    let densified = probe::record();
    let compact_error = nonfinite
        .eig_full(engine, &[0], &[1])
        .map(drop)
        .unwrap_err();
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    assert_eq!(format!("{compact_error:?}"), format!("{dense_error:?}"));
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_compact_diagonal_eig_full_rejects_inconsistent_spectrum_admission() {
    let engine = &mut crate::typed::CpuEngine::serial();
    use symtn_core::LieGroupFusionRule;

    let runtime = Runtime::builder().build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg = GradedSpace::try_new(provider, [(vec![0, 0], 2), (vec![1, 0], 1)]).unwrap();
    let input: TensorMap<_, Complex64> = TensorMap::diagonal(
        engine,
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![Complex64::new(1.0, 2.0), Complex64::new(-1.0, 0.0)],
            },
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![Complex64::new(0.0, 2.0)],
            },
        ],
    )
    .unwrap();
    let spectrum = input.spectrum().unwrap();
    // A spectrum that does not cover the bond is misuse, refused with a typed
    // error; no dense executor is used.
    let mut admit = |entries: &[symtn_matrixalgebra::SectorSpectrum<Complex64>]| {
        symtn_matrixalgebra::seam::eig_full_from_source::<CheckedGenericAdmissionMode, _, _, _, _>(
            engine,
            &mut symtn_dense::DefaultDenseExecutor::new(),
            symtn_matrixalgebra::seam::FactorSource::Diagonal {
                space: input.logical_space(),
                spectrum: entries,
            },
        )
        .is_ok()
    };
    assert!(admit(spectrum));
    let mut reversed = spectrum.to_vec();
    reversed.reverse();
    assert!(admit(&reversed));
    assert!(!admit(&spectrum[..1]));
    let mut duplicate = spectrum.to_vec();
    duplicate[1].sector = duplicate[0].sector;
    assert!(!admit(&duplicate));
    let mut wrong_size = spectrum.to_vec();
    wrong_size[0].values.push(Complex64::new(0.0, 0.0));
    assert!(!admit(&wrong_size));
    for bad in [
        Complex64::new(f64::INFINITY, 0.0),
        Complex64::new(f64::MAX, f64::MAX),
    ] {
        // A nonfinite value is refused by the shared finite-input stage, and
        // an overflowing magnitude by the dense route's eigenvalue check.
        let mut nonfinite = spectrum.to_vec();
        nonfinite[0].values[0] = bad;
        assert!(!admit(&nonfinite));
    }
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_compact_diagonal_eigh_full_applies_the_dense_hermiticity_check_directly() {
    let engine = &mut crate::typed::CpuEngine::serial();
    use symtn_core::LieGroupFusionRule;

    let calls = Arc::new(SpyCounts::default());
    let _dense = probe::inject_dense(Box::new(eigh_full_spy(&calls)));
    let runtime = Runtime::builder().build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg = GradedSpace::try_new(provider, [(vec![0, 0], 1), (vec![1, 0], 2)]).unwrap();
    let complex = |engine: &mut dyn Engine, values: [Complex64; 2]| {
        TensorMap::<_, Complex64>::diagonal(
            engine,
            &runtime,
            &leg,
            [
                SectorSpectrum {
                    sector: vec![0, 0],
                    values: vec![values[0]],
                },
                SectorSpectrum {
                    sector: vec![1, 0],
                    values: vec![values[1], Complex64::new(3.0, 0.0)],
                },
            ],
        )
        .unwrap()
    };
    // A nonzero imaginary part meets the dense route's Hermiticity check,
    // applied to the diagonal directly.
    let near = complex(
        engine,
        [Complex64::new(1.0, 1e-15), Complex64::new(-1.0, 0.0)],
    );
    let densified = probe::record();
    let Eigh { d, .. } = near
        .eigh_full(engine, &[0], &[1], HermitianTol::DEFAULT)
        .unwrap();
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    assert_eq!(calls.total(), 0);
    assert_eq!(
        d.diagview(engine).unwrap()[0].values,
        [Complex64::new(1.0, 0.0)]
    );

    let non_hermitian = complex(
        engine,
        [Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.5)],
    );
    let dense_error = non_hermitian
        .materialize(engine)
        .unwrap()
        .eigh_full(engine, &[0], &[1], HermitianTol::DEFAULT)
        .unwrap_err();
    let densified = probe::record();
    let compact_error = non_hermitian
        .eigh_full(engine, &[0], &[1], HermitianTol::DEFAULT)
        .unwrap_err();
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    assert_eq!(compact_error.to_string(), dense_error.to_string());

    let nonfinite: TensorMap<_, f64> = TensorMap::diagonal(
        engine,
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![1.0],
            },
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![f64::NAN, 2.0],
            },
        ],
    )
    .unwrap();
    let densified = probe::record();
    let compact_error = nonfinite
        .eigh_full(engine, &[0], &[1], HermitianTol::DEFAULT)
        .map(drop)
        .unwrap_err();
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    assert!(
        is_finite_input_refusal(&compact_error, "eigh"),
        "{compact_error:?}"
    );
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_compact_diagonal_eigh_full_rejects_inconsistent_spectrum_admission() {
    let engine = &mut crate::typed::CpuEngine::serial();
    use symtn_core::LieGroupFusionRule;

    let runtime = Runtime::builder().build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg = GradedSpace::try_new(provider, [(vec![0, 0], 2), (vec![1, 0], 1)]).unwrap();
    let input: TensorMap<_, f64> = TensorMap::diagonal(
        engine,
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![1.0, -1.0],
            },
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![2.0],
            },
        ],
    )
    .unwrap();
    let spectrum = input.spectrum().unwrap();
    // A spectrum that does not cover the bond is misuse, refused with a typed
    // error; no dense executor is used.
    let mut admit = |entries: &[symtn_matrixalgebra::SectorSpectrum<f64>]| {
        symtn_matrixalgebra::seam::eigh_full_from_source::<CheckedGenericAdmissionMode, _, _, _, _>(
            engine,
            &mut symtn_dense::DefaultDenseExecutor::new(),
            symtn_matrixalgebra::seam::FactorSource::Diagonal {
                space: input.logical_space(),
                spectrum: entries,
            },
            HermitianTol::DEFAULT,
        )
        .is_ok()
    };
    assert!(admit(spectrum));
    let mut reversed = spectrum.to_vec();
    reversed.reverse();
    assert!(admit(&reversed));
    assert!(!admit(&spectrum[..1]));
    let mut duplicate = spectrum.to_vec();
    duplicate[1].sector = duplicate[0].sector;
    assert!(!admit(&duplicate));
    let mut wrong_size = spectrum.to_vec();
    wrong_size[0].values.push(0.0);
    assert!(!admit(&wrong_size));
    // A nonfinite value is refused by the shared finite-input stage.
    let mut complex = spectrum.to_vec();
    complex[0].values[0] = f64::INFINITY;
    assert!(!admit(&complex));
}
