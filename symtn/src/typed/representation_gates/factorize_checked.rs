use super::*;

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_compact_diagonal_svd_vals_skips_materialization_and_solver() {
    use symtn_core::LieGroupFusionRule;

    let solver_calls = Arc::new(SpyCounts::default());
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(fail_second_svd(&solver_calls)))
        .build()
        .unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(vec![0, 0], 2), (vec![1, 0], 3)]).unwrap();
    let real: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![3.0, -2.0, 1.0],
            },
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![-4.0, 0.0],
            },
        ],
    )
    .unwrap();
    let densified = probe::record();
    assert_eq!(
        real.svd_vals(&[0], &[1]).unwrap(),
        vec![
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![4.0, 0.0],
            },
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![3.0, 2.0, 1.0],
            },
        ]
    );
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    assert_eq!(solver_calls.total(), 0);

    // A spectrum that does not cover the bond is misuse: a typed error.
    let stored = real.spectrum().unwrap();
    let values_of = |spectrum| {
        symtn_matrixalgebra::seam::svd_vals_from_source::<CheckedGenericAdmissionMode, _, _, _, _>(
            &mut symtn_dense::DefaultDenseExecutor::new(),
            symtn_matrixalgebra::seam::FactorSource::Diagonal {
                space: &owned(&real).space,
                spectrum,
            },
        )
    };
    let mut missing = stored.to_vec();
    missing.pop();
    assert!(
        matches!(values_of(&missing), Err(symtn_matrixalgebra::seam::CheckedGenericFactorPlanError::Operation(OperationError::InvalidArgument { message })) if message.contains("sector count"))
    );
    let mut duplicate = stored.to_vec();
    duplicate[1].sector = duplicate[0].sector;
    assert!(
        matches!(values_of(&duplicate), Err(symtn_matrixalgebra::seam::CheckedGenericFactorPlanError::Operation(OperationError::InvalidArgument { message })) if message.contains("twice"))
    );
    let mut short = stored.to_vec();
    short[0].values.pop();
    assert!(
        matches!(values_of(&short), Err(symtn_matrixalgebra::seam::CheckedGenericFactorPlanError::Operation(OperationError::InvalidArgument { message })) if message.contains("length"))
    );

    let complex: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![
                    Complex64::new(-3.0, 4.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 1.0),
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
    let got = complex.svd_vals(&[0], &[1]).unwrap();
    assert_eq!(got[0].sector, vec![0, 0]);
    assert_eq!(got[0].values, [4.0, 0.0]);
    assert_eq!(got[1].sector, vec![1, 0]);
    assert_eq!(got[1].values[0], 5.0);
    assert!((got[1].values[1] - 2.0_f64.sqrt()).abs() < 1e-12);
    assert_eq!(got[1].values[2], 0.0);
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    assert_eq!(solver_calls.total(), 0);

    let dual_leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 0], 2)])
        .unwrap()
        .try_dual()
        .unwrap();
    let dual_sector = dual_leg.sectors().unwrap().remove(0);
    let dual: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &dual_leg,
        [SectorSpectrum {
            sector: dual_sector.clone(),
            values: vec![-2.0, 0.0],
        }],
    )
    .unwrap();
    let densified = probe::record();
    assert_eq!(
        dual.svd_vals(&[0], &[1]).unwrap(),
        vec![SectorSpectrum {
            sector: dual_sector,
            values: vec![2.0, 0.0]
        }]
    );
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    assert_eq!(solver_calls.total(), 0);
}

/// #1729: an admitted checked-Generic dual diagonal (non-self-dual SU(3)
/// labels) runs all four QR/LQ methods on `W = V` without materializing the
/// input or calling the dense QR kernel (the polar spy panics on QR).
#[cfg(feature = "cgc-gen")]
#[test]
fn checked_dual_diagonal_qr_lq_skips_materialization_and_dense_qr() {
    use symtn_core::LieGroupFusionRule;

    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(polar_spy(&Arc::default())))
        .build()
        .unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 0], 3), (vec![0, 0], 2)])
        .unwrap()
        .try_dual()
        .unwrap();
    let sectors = leg.sectors().unwrap();
    assert!(!sectors.contains(&vec![1, 0]));
    let input: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &leg,
        sectors.into_iter().map(|sector| {
            let values = (0..leg.degeneracy(&sector).unwrap())
                .map(|index| Complex64::new(index as f64 - 1.0, 2.0 - index as f64))
                .collect();
            SectorSpectrum { sector, values }
        }),
    )
    .unwrap();
    for operation in 0..4 {
        let densified = probe::record();
        let (left, right) = match operation {
            0 => input.qr_compact(&[0], &[1]).unwrap().pair(),
            1 => input.qr_full(&[0], &[1]).unwrap().pair(),
            2 => input.lq_compact(&[0], &[1]).unwrap().pair(),
            _ => input.lq_full(&[0], &[1]).unwrap().pair(),
        };
        assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
        for factor in [&left, &right] {
            assert_eq!(factor.codomain(), input.codomain());
            assert_eq!(factor.domain(), input.domain());
            assert!(factor.codomain()[0].is_dual());
            assert!(factor.diagview().is_ok());
            assert!(factor.dense_data().is_err());
        }
    }
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_compact_diagonal_svd_avoids_input_materialization_and_solver() {
    use symtn_core::LieGroupFusionRule;

    let svd_calls = Arc::new(SpyCounts::default());
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(polar_spy(&svd_calls)))
        .build()
        .unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let bond =
        GradedSpace::try_new(Arc::clone(&provider), [(vec![0, 0], 2), (vec![1, 0], 3)]).unwrap();
    let input: TensorMap<_, num_complex::Complex64> = TensorMap::diagonal(
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![
                    num_complex::Complex64::new(0.0, 0.0),
                    num_complex::Complex64::new(-2.0, 0.0),
                ],
            },
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![
                    num_complex::Complex64::new(1.0, 0.0),
                    num_complex::Complex64::new(0.0, 3.0),
                    num_complex::Complex64::new(-1.0, 0.0),
                ],
            },
        ],
    )
    .unwrap();
    let densified = probe::record();
    let Svd { u, s, vh } = input.svd_compact(&[0], &[1]).unwrap();
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    assert_eq!(svd_calls.of(POLAR_SVD), 0);
    assert!(std::ptr::eq(u.provider(), provider.as_ref()));
    assert!(std::ptr::eq(s.provider(), provider.as_ref()));
    assert!(std::ptr::eq(vh.provider(), provider.as_ref()));
    assert!(u.dense_data().is_ok());
    assert!(s.dense_data().is_err());
    assert!(vh.dense_data().is_ok());

    let densified = probe::record();
    svd_calls.reset();
    let Svd { u, s, vh } = input.svd_full(&[0], &[1]).unwrap();
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    assert_eq!(svd_calls.of(POLAR_SVD), 0);
    assert!(std::ptr::eq(u.provider(), provider.as_ref()));
    assert!(std::ptr::eq(s.provider(), provider.as_ref()));
    assert!(std::ptr::eq(vh.provider(), provider.as_ref()));
    assert!(u.dense_data().is_ok());
    assert!(s.dense_data().is_err());
    assert!(vh.dense_data().is_ok());
    assert_eq!(
        s.diagview().unwrap(),
        vec![
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![
                    num_complex::Complex64::new(2.0, 0.0),
                    num_complex::Complex64::new(0.0, 0.0),
                ],
            },
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![
                    num_complex::Complex64::new(3.0, 0.0),
                    num_complex::Complex64::new(1.0, 0.0),
                    num_complex::Complex64::new(1.0, 0.0),
                ],
            },
        ]
    );
    let expected_u: TensorMap<_, num_complex::Complex64> = TensorMap::from_subblock_fn(
        &runtime,
        u.codomain().iter(),
        u.domain().iter(),
        |trees, ij| {
            let order: &[usize] = if trees.coupled() == &vec![0, 0] {
                &[1, 0]
            } else {
                &[1, 0, 2]
            };
            num_complex::Complex64::new(f64::from(ij[0] == order[ij[1]]), 0.0)
        },
    )
    .unwrap();
    let expected_vh: TensorMap<_, num_complex::Complex64> = TensorMap::from_subblock_fn(
        &runtime,
        vh.codomain().iter(),
        vh.domain().iter(),
        |trees, ij| {
            let (order, phases): (&[usize], &[num_complex::Complex64]) =
                if trees.coupled() == &vec![0, 0] {
                    (
                        &[1, 0],
                        &[
                            num_complex::Complex64::new(-1.0, 0.0),
                            num_complex::Complex64::new(1.0, 0.0),
                        ],
                    )
                } else {
                    (
                        &[1, 0, 2],
                        &[
                            num_complex::Complex64::new(0.0, 1.0),
                            num_complex::Complex64::new(1.0, 0.0),
                            num_complex::Complex64::new(-1.0, 0.0),
                        ],
                    )
                };
            if ij[1] == order[ij[0]] {
                phases[ij[0]]
            } else {
                num_complex::Complex64::new(0.0, 0.0)
            }
        },
    )
    .unwrap();
    for (actual, expected) in [(&u, &expected_u), (&vh, &expected_vh)] {
        assert!(actual
            .dense_data()
            .unwrap()
            .iter()
            .zip(expected.dense_data().unwrap())
            .all(|(actual, expected)| (*actual - *expected).norm() <= 1e-12));
    }
    let rebuilt = u.compose(&s).unwrap().compose(&vh).unwrap();
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(input.materialize().unwrap().dense_data().unwrap())
        .all(|(actual, expected)| (*actual - *expected).norm() <= 1e-12));

    let changed = input.svd_full(&[1], &[0]).unwrap();
    let rebuilt = changed
        .u
        .compose(&changed.s)
        .unwrap()
        .compose(&changed.vh)
        .unwrap();
    let expected = input.permute(&[1], &[0]).unwrap().materialize().unwrap();
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(expected.dense_data().unwrap())
        .all(|(actual, expected)| (*actual - *expected).norm() <= 1e-12));

    let TypedData::Diagonal(spectrum) = owned(&input).data.as_ref() else {
        panic!("diagonal constructor must retain compact storage");
    };
    let source = &owned(&input).space;
    let mut missing = spectrum.clone();
    missing.pop();
    let mut duplicate = spectrum.clone();
    duplicate[1].sector = duplicate[0].sector;
    let mut short = spectrum.clone();
    short[0].values.pop();
    // A spectrum that does not cover the bond is misuse: a typed error, not a
    // dense fallback (MAK asserts the sizes). Full SVD of a diagonal is this
    // compact route (TensorKit `svd_compact!(::DiagonalAlgorithm)`).
    let diagonal = |spectrum| symtn_matrixalgebra::seam::FactorSource::Diagonal {
        space: source,
        spectrum,
    };
    let misuse = |error: &symtn_matrixalgebra::seam::CheckedGenericFactorPlanError<_>| {
        matches!(
            error,
            symtn_matrixalgebra::seam::CheckedGenericFactorPlanError::Operation(
                symtn_tensors::OperationError::InvalidArgument { .. }
            )
        )
    };
    for malformed in [&missing, &duplicate, &short] {
        let mut dense = symtn_dense::DefaultDenseExecutor::new();
        assert!(misuse(
            &symtn_matrixalgebra::seam::svd_compact_from_source::<
                CheckedGenericAdmissionMode,
                _,
                _,
                _,
                _,
            >(&mut dense, diagonal(malformed))
            .err()
            .unwrap()
        ));
    }
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_compact_diagonal_polar_returns_compact_hand_oracle_without_dense_work() {
    use symtn_core::LieGroupFusionRule;

    let svd_calls = Arc::new(SpyCounts::default());
    let gemm_calls = Arc::clone(&svd_calls);
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(polar_spy(&svd_calls)))
        .build()
        .unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let bond =
        GradedSpace::try_new(Arc::clone(&provider), [(vec![0, 0], 2), (vec![1, 0], 3)]).unwrap();
    let input: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![
                    Complex64::new(3.0, 4.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(0.0, -2.0),
                ],
            },
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![Complex64::new(-3.0, 0.0), Complex64::new(0.0, 1.0)],
            },
        ],
    )
    .unwrap();
    let dense_oracle = input.materialize().unwrap();

    for left in [true, false] {
        let (dense_w, dense_p) = if left {
            let LeftPolar { w, p } = dense_oracle.left_polar(&[0], &[1]).unwrap();
            (w, p)
        } else {
            let RightPolar { p, wh } = dense_oracle.right_polar(&[0], &[1]).unwrap();
            (wh, p)
        };
        let densified = probe::record();
        svd_calls.reset();
        gemm_calls.reset();
        let (w, p) = if left {
            let LeftPolar { w, p } = input.left_polar(&[0], &[1]).unwrap();
            (w, p)
        } else {
            let RightPolar { p, wh } = input.right_polar(&[0], &[1]).unwrap();
            (wh, p)
        };
        assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
        assert_eq!(svd_calls.of(POLAR_SVD), 0);
        assert_eq!(gemm_calls.of(Kernel::GEMM), 0);
        assert!(std::ptr::eq(w.provider(), provider.as_ref()));
        assert!(std::ptr::eq(p.provider(), provider.as_ref()));
        assert_eq!(w.logical_space().space(), dense_w.logical_space().space());
        assert_eq!(p.logical_space().space(), dense_p.logical_space().space());
        assert!(w.dense_data().is_err());
        assert!(p.dense_data().is_err());
        let phase = w.spectrum().unwrap();
        let magnitude = p.spectrum().unwrap();
        for source in input.spectrum().unwrap() {
            let actual_phase = phase
                .iter()
                .find(|entry| entry.sector == source.sector)
                .unwrap();
            let actual_magnitude = magnitude
                .iter()
                .find(|entry| entry.sector == source.sector)
                .unwrap();
            for ((&value, &got_phase), &got_magnitude) in source
                .values
                .iter()
                .zip(&actual_phase.values)
                .zip(&actual_magnitude.values)
            {
                let expected_magnitude = value.norm();
                let expected_phase = if expected_magnitude == 0.0 {
                    Complex64::new(1.0, 0.0)
                } else {
                    value / expected_magnitude
                };
                assert!((got_phase - expected_phase).norm() <= 1e-12);
                assert!((got_magnitude - Complex64::new(expected_magnitude, 0.0)).norm() <= 1e-12);
            }
        }
        let rebuilt = if left { w.compose(&p) } else { p.compose(&w) }.unwrap();
        assert!(rebuilt
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .zip(dense_oracle.dense_data().unwrap())
            .all(|(actual, expected)| (*actual - *expected).norm() <= 1e-12));
    }
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_compact_diagonal_null_is_empty_by_shape_without_dense_work() {
    // What: an SU(3) checked compact diagonal is square in every sector, so
    // both shape-based null spaces are empty whatever its zeros, read with
    // no materialization or dense kernel, keep the provider, and agree with
    // the dense route.
    use symtn_core::LieGroupFusionRule;

    macro_rules! check {
        ($dtype:ty) => {{
            let calls = Arc::new(SpyCounts::default());
            let runtime = Runtime::builder()
                .with_dense_executor(Box::new(polar_spy(&calls)))
                .build()
                .unwrap();
            let provider = Arc::new(
                LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap(),
            );
            let bond =
                GradedSpace::try_new(Arc::clone(&provider), [(vec![0, 0], 2), (vec![1, 0], 3)])
                    .unwrap();
            let input: TensorMap<_, $dtype> = TensorMap::diagonal(
                &runtime,
                &bond,
                [
                    SectorSpectrum {
                        sector: vec![1, 0],
                        values: vec![
                            <$dtype>::from_real(1.0),
                            <$dtype>::from_real(0.0),
                            <$dtype>::from_real(0.0),
                        ],
                    },
                    SectorSpectrum {
                        sector: vec![0, 0],
                        values: vec![<$dtype>::from_real(0.0), <$dtype>::from_real(-2.0)],
                    },
                ],
            )
            .unwrap();
            let dense_input = input.materialize().unwrap();
            let densified = probe::record();
            let left = input.left_null(&[0], &[1]).unwrap();
            let right = input.right_null(&[0], &[1]).unwrap();
            assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
            assert_eq!(calls.of(POLAR_SVD), 0);
            assert!(std::ptr::eq(left.provider(), provider.as_ref()));
            assert!(std::ptr::eq(right.provider(), provider.as_ref()));
            let dense_left = dense_input.left_null(&[0], &[1]).unwrap();
            let dense_right = dense_input.right_null(&[0], &[1]).unwrap();
            for factor in [&left, &right, &dense_left, &dense_right] {
                assert!(factor.dense_data().unwrap().is_empty());
            }
            assert!(left.domain()[0].degeneracies().is_empty());
            assert!(right.codomain()[0].degeneracies().is_empty());
            assert_eq!(left.domain(), dense_left.domain());
            assert_eq!(right.codomain(), dense_right.codomain());
        }};
    }

    check!(f32);
    check!(f64);
    check!(num_complex::Complex32);
    check!(Complex64);
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_compact_diagonal_svd_vals_rounds_at_payload_precision() {
    use symtn_core::LieGroupFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg = GradedSpace::try_new(provider, [(vec![1, 0], 2)]).unwrap();
    let real: TensorMap<_, f32> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: vec![1, 0],
            values: vec![-1.000_000_1, 0.0],
        }],
    )
    .unwrap();
    let densified = probe::record();
    assert_eq!(
        real.svd_vals(&[0], &[1]).unwrap()[0].values,
        [1.000_000_1_f32 as f64, 0.0]
    );
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);

    let complex: TensorMap<_, num_complex::Complex32> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: vec![1, 0],
            values: vec![
                num_complex::Complex32::new(1.0, 1.0),
                num_complex::Complex32::new(-3.0, 4.0),
            ],
        }],
    )
    .unwrap();
    let densified = probe::record();
    assert_eq!(
        complex.svd_vals(&[0], &[1]).unwrap()[0].values,
        [5.0, (2.0_f64.sqrt() as f32) as f64]
    );
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);

    for value in [
        num_complex::Complex32::new(f32::NAN, 0.0),
        num_complex::Complex32::new(f32::INFINITY, 0.0),
        num_complex::Complex32::new(f32::MAX * 0.75, f32::MAX * 0.75),
    ] {
        let input: TensorMap<_, num_complex::Complex32> = TensorMap::diagonal(
            &runtime,
            &leg,
            [SectorSpectrum {
                sector: vec![1, 0],
                values: vec![value, num_complex::Complex32::new(0.0, 0.0)],
            }],
        )
        .unwrap();
        // A nonfinite value is refused by the shared finite-input stage; a
        // finite one is MAK `svd_vals!(::DiagonalAlgorithm)`, `abs` then a
        // descending sort: `|0.75 MAX (1 + i)| = 1.06 MAX` overflows f32 to Inf.
        let densified = probe::record();
        let result = input.svd_vals(&[0], &[1]);
        assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
        if value.re.is_finite() {
            let values = result.unwrap().remove(0).values;
            assert_eq!(values, [f64::INFINITY, 0.0]);
        } else {
            let error = result.unwrap_err();
            assert!(is_finite_input_refusal(&error, "svd"), "{error:?}");
        }
    }
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_compact_diagonal_pinv_does_not_materialize_the_input() {
    use symtn_core::LieGroupFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let bond =
        GradedSpace::try_new(Arc::clone(&provider), [(vec![0, 0], 2), (vec![1, 0], 3)]).unwrap();
    let input: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![8.0, -4.0],
            },
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![1.0, 2.0, 0.0],
            },
        ],
    )
    .unwrap();
    let densified = probe::record();
    let output = input.pinv(&[0], &[1], 0.25).unwrap();
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    assert!(output.dense_data().is_err());
    assert_eq!(output.diagview().unwrap()[0].values, [0.125, -0.25]);

    let TypedData::Diagonal(spectrum) = owned(&input).data.as_ref() else {
        panic!("diagonal constructor must keep compact storage");
    };
    // A spectrum that does not cover its bond (missing, duplicate or short
    // sector) is misuse, refused by the shared admission before any value
    // is read (#1994).
    let admit = |spectrum: &[symtn_matrixalgebra::SectorSpectrum<f64>]| {
        symtn_matrixalgebra::seam::admit_compact_diagonal::<CheckedGenericAdmissionMode, _, _>(
            &owned(&input).space,
            spectrum,
        )
    };
    admit(spectrum).unwrap();
    let mut missing = spectrum.clone();
    missing.pop();
    let mut duplicate = spectrum.clone();
    duplicate[1].sector = duplicate[0].sector;
    let mut short = spectrum.clone();
    short[0].values.pop();
    for (malformed, expected) in [
        (missing, "sector count differs"),
        (duplicate, "bond sector twice"),
        (short, "length differs"),
    ] {
        assert!(
            matches!(admit(&malformed), Err(CheckedGenericFactorPlanError::Operation(OperationError::InvalidArgument { message })) if message.contains(expected))
        );
    }
}

/// #1735: checked `inv`/`exp`/`solve` of a compact diagonal map the spectrum
/// elementwise (TensorKit `inv`/`exp`/`\` on `DiagonalTensorMap`) and never
/// densify it; hand-computed values and a dense LU oracle are independent.
#[cfg(feature = "cgc-gen")]
#[test]
fn checked_compact_diagonal_inv_exp_solve_stay_compact() {
    use num_complex::Complex64;
    use symtn_core::LieGroupFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let bond =
        GradedSpace::try_new(Arc::clone(&provider), [(vec![0, 0], 2), (vec![1, 0], 3)]).unwrap();
    let column =
        GradedSpace::try_new(Arc::clone(&provider), [(vec![0, 0], 1), (vec![1, 0], 2)]).unwrap();

    macro_rules! compact_values {
        ($tensor:expr) => {{
            let tensor = $tensor;
            assert!(tensor.dense_data().is_err(), "result must stay compact");
            tensor
                .diagview()
                .unwrap()
                .into_iter()
                .map(|entry| entry.values)
                .collect::<Vec<_>>()
        }};
    }

    let real: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![2.0, -4.0],
            },
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![0.5, 1.0, -0.25],
            },
        ],
    )
    .unwrap();
    let densified = probe::record();
    let inverse = real.inv(&[0], &[1]).unwrap();
    let exponential = real.exp(&[0], &[1]).unwrap();
    let compact_solve = real.solve(&[0], &[1], &real, &[0], &[1]).unwrap();
    let rhs = TensorMap::<_, f64>::rand_with_seed(&runtime, [&bond], [&column], 7).unwrap();
    let dense_solve = real.solve(&[0], &[1], &rhs, &[0], &[1]).unwrap();
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    assert_eq!(
        compact_values!(&inverse),
        [vec![0.5, -0.25], vec![2.0, 1.0, -4.0]]
    );
    let expected_exp = [
        vec![2.0_f64.exp(), (-4.0_f64).exp()],
        vec![0.5_f64.exp(), 1.0_f64.exp(), (-0.25_f64).exp()],
    ];
    for (got, want) in compact_values!(&exponential).iter().zip(&expected_exp) {
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() <= 1e-15 * w.abs(), "{g} vs {w}");
        }
    }
    assert_eq!(
        compact_values!(&compact_solve),
        [vec![1.0, 1.0], vec![1.0, 1.0, 1.0]]
    );
    let lu = real
        .materialize()
        .unwrap()
        .solve(&[0], &[1], &rhs, &[0], &[1])
        .unwrap();
    assert_eq!(dense_solve.codomain(), lu.codomain());
    assert_eq!(dense_solve.domain(), lu.domain());
    for (g, w) in dense_solve
        .dense_data()
        .unwrap()
        .iter()
        .zip(lu.dense_data().unwrap())
    {
        assert!((g - w).abs() <= 1e-14 * (1.0 + w.abs()), "{g} vs {w}");
    }

    // A dual bond: the scaled leg's charge is the dual of the stored sector
    // label, so a mis-keyed scaling would disagree with the LU oracle.
    let dual = bond.try_dual().unwrap();
    let dual_spectrum = [
        SectorSpectrum {
            sector: vec![0, 0],
            values: vec![2.0, -4.0],
        },
        SectorSpectrum {
            sector: vec![0, 1],
            values: vec![0.5, 1.0, -0.25],
        },
    ];
    let dual_real: TensorMap<_, f64> = TensorMap::diagonal(&runtime, &dual, dual_spectrum).unwrap();
    let dual_rhs = TensorMap::<_, f64>::rand_with_seed(&runtime, [&dual], [&column], 11).unwrap();
    let densified = probe::record();
    let dual_solve = dual_real.solve(&[0], &[1], &dual_rhs, &[0], &[1]).unwrap();
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    let lu = dual_real
        .materialize()
        .unwrap()
        .solve(&[0], &[1], &dual_rhs, &[0], &[1])
        .unwrap();
    for (g, w) in dual_solve
        .dense_data()
        .unwrap()
        .iter()
        .zip(lu.dense_data().unwrap())
    {
        assert!((g - w).abs() <= 1e-14 * (1.0 + w.abs()), "{g} vs {w}");
    }

    let z = |re, im| Complex64::new(re, im);
    let complex: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![z(1.0, 1.0), z(0.0, -2.0)],
            },
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![z(4.0, 0.0), z(-0.5, 0.5), z(0.0, 0.25)],
            },
        ],
    )
    .unwrap();
    let source = compact_values!(&complex);
    let densified = probe::record();
    let inverse = complex.inv(&[0], &[1]).unwrap();
    let exponential = complex.exp(&[0], &[1]).unwrap();
    let rhs = TensorMap::<_, Complex64>::rand_with_seed(&runtime, [&bond], [&column], 9).unwrap();
    let dense_solve = complex.solve(&[0], &[1], &rhs, &[0], &[1]).unwrap();
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    for ((inv, exp), values) in compact_values!(&inverse)
        .iter()
        .zip(compact_values!(&exponential))
        .zip(&source)
    {
        for ((i, e), v) in inv.iter().zip(&exp).zip(values) {
            assert!((i - z(1.0, 0.0) / v).norm() <= 1e-15 * i.norm());
            assert!((e - v.exp()).norm() <= 1e-15 * e.norm());
        }
    }
    let lu = complex
        .materialize()
        .unwrap()
        .solve(&[0], &[1], &rhs, &[0], &[1])
        .unwrap();
    for (g, w) in dense_solve
        .dense_data()
        .unwrap()
        .iter()
        .zip(lu.dense_data().unwrap())
    {
        assert!((g - w).norm() <= 1e-14 * (1.0 + w.norm()), "{g} vs {w}");
    }
}

/// #1735: checked and multiplicity-free compact arms share one spectrum map,
/// so the same spectrum gives bitwise-equal values and equal errors.
#[cfg(feature = "cgc-gen")]
#[test]
fn checked_compact_diagonal_spectrum_maps_match_multiplicity_free() {
    use symtn_core::LieGroupFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let su3 = GradedSpace::try_new(
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap()),
        [(vec![0, 0], 2), (vec![1, 0], 3)],
    )
    .unwrap();
    let u1 = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 3)],
    )
    .unwrap();
    let cases: [[Vec<f64>; 2]; 3] = [
        [vec![2.0, -4.0], vec![0.5, 3.0, -0.25]],
        [vec![2.0, 0.0], vec![0.5, 3.0, -0.25]],
        [vec![2.0, f64::NAN], vec![0.5, f64::INFINITY, -0.25]],
    ];
    let bits = |result: Result<Vec<Vec<f64>>, String>| {
        result.map(|values| {
            values
                .into_iter()
                .map(|entry| entry.into_iter().map(f64::to_bits).collect::<Vec<_>>())
                .collect::<Vec<_>>()
        })
    };
    for [first, second] in cases {
        let checked: TensorMap<_, f64> = TensorMap::diagonal(
            &runtime,
            &su3,
            [
                SectorSpectrum {
                    sector: vec![0, 0],
                    values: first.clone(),
                },
                SectorSpectrum {
                    sector: vec![1, 0],
                    values: second.clone(),
                },
            ],
        )
        .unwrap();
        let free: TensorMap<_, f64> = TensorMap::diagonal(
            &runtime,
            &u1,
            [
                SectorSpectrum {
                    sector: U1Irrep::new(0),
                    values: first,
                },
                SectorSpectrum {
                    sector: U1Irrep::new(1),
                    values: second,
                },
            ],
        )
        .unwrap();
        macro_rules! spectrum_of {
            ($result:expr) => {
                $result
                    .map(|tensor| {
                        assert!(tensor.dense_data().is_err());
                        tensor
                            .diagview()
                            .unwrap()
                            .into_iter()
                            .map(|entry| entry.values)
                            .collect::<Vec<_>>()
                    })
                    .map_err(|error| error.to_string())
            };
        }
        let densified = probe::record();
        let checked_results = [
            spectrum_of!(checked.inv(&[0], &[1])),
            spectrum_of!(checked.exp(&[0], &[1])),
            spectrum_of!(checked.solve(&[0], &[1], &checked, &[0], &[1])),
        ];
        assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
        let free_results = [
            spectrum_of!(free.inv(&[0], &[1])),
            spectrum_of!(free.exp(&[0], &[1])),
            spectrum_of!(free.solve(&[0], &[1], &free, &[0], &[1])),
        ];
        for (checked, free) in checked_results.into_iter().zip(free_results) {
            assert_eq!(bits(checked), bits(free));
        }
    }
}
