use super::*;

#[test]
fn compact_diagonal_qr_lq_preserves_compact_factors_without_dense_kernels() {
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(polar_spy(&Arc::default())))
        .build()
        .unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(1), 3)])
        .unwrap()
        .try_dual()
        .unwrap();
    let input: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(-1),
            values: vec![-2.0, 0.0, 3.0],
        }],
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
            assert!(factor.diagview().is_ok());
            assert!(factor.dense_data().is_err());
            assert_eq!(
                factor.map_diagonal(|x| x).unwrap().diagview().unwrap(),
                factor.diagview().unwrap()
            );
        }
    }
}

#[test]
fn compact_diagonal_qr_lq_rejects_inconsistent_spectra_and_nonbond_spaces() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let input: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: U1Irrep::new(1),
                values: vec![2.0],
            },
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![-1.0, 0.0],
            },
        ],
    )
    .unwrap();
    let spectrum = input.spectrum().unwrap();
    let admit = |values: &[symtn_matrixalgebra::SectorSpectrum<f64>]| {
        qr_diagonal(input.logical_space(), values)
    };
    assert!(admit(spectrum).is_ok());
    let mut reversed = spectrum.to_vec();
    reversed.reverse();
    assert!(admit(&reversed).is_ok());
    // A spectrum that does not cover the bond is misuse (MAK asserts the
    // sizes), not a reason to try a dense route.
    let misuse = |result: Result<_, symtn_tensors::OperationError>| {
        matches!(
            result,
            Err(symtn_tensors::OperationError::InvalidArgument { .. })
        )
    };
    assert!(misuse(admit(&spectrum[..1])));
    let mut duplicate = spectrum.to_vec();
    duplicate[1].sector = duplicate[0].sector;
    assert!(misuse(admit(&duplicate)));
    let mut wrong = spectrum.to_vec();
    wrong[0].values.push(1.0);
    assert!(misuse(admit(&wrong)));
    // A NaN is refused by the shared finite-input stage.
    wrong = spectrum.to_vec();
    wrong[0].values[0] = f64::NAN;
    assert!(matches!(
        admit(&wrong),
        Err(symtn_tensors::OperationError::InvalidArgument {
            message: "qr input components must be finite"
        })
    ));
    let multileg: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&runtime, [&leg, &leg], [&leg, &leg], 1).unwrap();
    let space_mismatch = |result: Result<_, symtn_tensors::OperationError>| {
        matches!(
            result,
            Err(symtn_tensors::OperationError::SpaceMismatch { .. })
        )
    };
    assert!(space_mismatch(qr_diagonal(
        multileg.logical_space(),
        spectrum
    )));
    let dual = leg.try_dual().unwrap();
    let nonendo: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&runtime, [&leg], [&dual], 1).unwrap();
    assert!(space_mismatch(qr_diagonal(
        nonendo.logical_space(),
        spectrum
    )));
}

#[test]
fn compact_diagonal_polar_skips_input_materialization_svd_and_gemm() {
    let svd_calls = Arc::new(SpyCounts::default());
    let gemm_calls = Arc::clone(&svd_calls);
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(polar_spy(&svd_calls)))
        .build()
        .unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 3)]).unwrap();
    let input: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![
                Complex64::new(-2.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 3.0),
            ],
        }],
    )
    .unwrap();
    let densified = probe::record();
    let left = input.left_polar(&[0], &[1]).unwrap();
    assert!(left.w.diagview().is_ok());
    assert!(left.p.diagview().is_ok());
    assert!(left.w.dense_data().is_err());
    assert!(left.p.dense_data().is_err());
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    assert_eq!(svd_calls.of(POLAR_SVD), 0);
    assert_eq!(gemm_calls.of(Kernel::GEMM), 0);
    // Hand oracle: phase and magnitude of -2, 0 and 3i, with unit phase at 0.
    let zero = Complex64::new(0.0, 0.0);
    let phase = [
        Complex64::new(-1.0, 0.0),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 1.0),
    ];
    let magnitude = [2.0, 0.0, 3.0].map(|value| Complex64::new(value, 0.0));
    let w = left.w.materialize().unwrap();
    let p = left.p.materialize().unwrap();
    for (i, (&actual_w, &actual_p)) in w
        .dense_data()
        .unwrap()
        .iter()
        .zip(p.dense_data().unwrap())
        .enumerate()
    {
        let (row, col) = (i % 3, i / 3);
        let on_diagonal = row == col;
        assert_eq!(actual_w, if on_diagonal { phase[row] } else { zero });
        assert_eq!(actual_p, if on_diagonal { magnitude[row] } else { zero });
    }
    let densified = probe::record();
    let right = input.right_polar(&[0], &[1]).unwrap();
    assert!(right.wh.diagview().is_ok());
    assert!(right.p.diagview().is_ok());
    assert!(right.wh.dense_data().is_err());
    assert!(right.p.dense_data().is_err());
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    assert_eq!(svd_calls.of(POLAR_SVD), 0);
    assert_eq!(gemm_calls.of(Kernel::GEMM), 0);
}

#[test]
fn compact_diagonal_qr_lq_matches_hand_phase_and_absolute_value_all_scalars() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 4)]).unwrap();
    macro_rules! check {
        ($ty:ty, $values:expr, $phase:expr, $magnitude:expr, $tol:expr) => {{
            let input: TensorMap<_, $ty> = TensorMap::diagonal(
                &runtime,
                &leg,
                [SectorSpectrum {
                    sector: U1Irrep::new(0),
                    values: $values,
                }],
            )
            .unwrap();
            let saved = input.diagview().unwrap();
            for operation in 0..4 {
                let (w, p) = match operation {
                    0 => input.qr_compact(&[0], &[1]).unwrap().pair(),
                    1 => input.qr_full(&[0], &[1]).unwrap().pair(),
                    2 => {
                        let Lq { l, q } = input.lq_compact(&[0], &[1]).unwrap();
                        (q, l)
                    }
                    _ => {
                        let Lq { l, q } = input.lq_full(&[0], &[1]).unwrap();
                        (q, l)
                    }
                };
                assert_eq!(w.codomain(), input.codomain());
                assert_eq!(w.domain(), input.domain());
                assert_eq!(p.codomain(), input.domain());
                assert_eq!(p.domain(), p.codomain());
                assert!(w.dense_data().is_err());
                assert!(p.dense_data().is_err());
                assert!(w.diagview().is_ok());
                assert!(p.diagview().is_ok());
                let w = w.materialize().unwrap();
                let p = p.materialize().unwrap();
                for (i, (&actual_w, &actual_p)) in w
                    .dense_data()
                    .unwrap()
                    .iter()
                    .zip(p.dense_data().unwrap())
                    .enumerate()
                {
                    let row = i % 4;
                    let col = i / 4;
                    let expected_w = if row == col {
                        $phase[row]
                    } else {
                        Complex64::new(0.0, 0.0)
                    };
                    let expected_p = if row == col { $magnitude[row] } else { 0.0 };
                    assert!((actual_w.widen_complex() - expected_w).norm() <= $tol);
                    assert!((actual_p.widen_complex().re - expected_p).abs() <= $tol);
                }
                assert_typed_map_close(
                    &w.compose(&p).unwrap(),
                    &input.materialize().unwrap(),
                    $tol,
                );
                let unit = w.adjoint().unwrap().compose(&w).unwrap();
                for (i, &value) in unit.dense_data().unwrap().iter().enumerate() {
                    let expected = if i % 4 == i / 4 { 1.0 } else { 0.0 };
                    assert!((value.widen_complex() - Complex64::new(expected, 0.0)).norm() <= $tol);
                }
            }
            assert_eq!(input.diagview().unwrap(), saved);
        }};
    }
    let phase = [
        Complex64::new(-1.0, 0.0),
        Complex64::new(0.6, 0.8),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    let magnitude = [2.0, 5.0, 0.0, 1.0];
    check!(
        f32,
        vec![-2.0, 5.0, -0.0, 1.0],
        [phase[0], Complex64::new(1.0, 0.0), phase[2], phase[3]],
        magnitude,
        2e-6
    );
    check!(
        f64,
        vec![-2.0, 5.0, -0.0, 1.0],
        [phase[0], Complex64::new(1.0, 0.0), phase[2], phase[3]],
        magnitude,
        1e-12
    );
    check!(
        num_complex::Complex32,
        vec![
            num_complex::Complex32::new(-2.0, 0.0),
            num_complex::Complex32::new(3.0, 4.0),
            num_complex::Complex32::new(-0.0, 0.0),
            num_complex::Complex32::new(1.0, 0.0)
        ],
        phase,
        magnitude,
        2e-6
    );
    check!(
        Complex64,
        vec![
            Complex64::new(-2.0, 0.0),
            Complex64::new(3.0, 4.0),
            Complex64::new(-0.0, 0.0),
            Complex64::new(1.0, 0.0)
        ],
        phase,
        magnitude,
        1e-12
    );
}

#[test]
fn compact_diagonal_qr_lq_preserves_sectors_and_changed_roles() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    macro_rules! check {
        ($leg:expr, $spectra:expr) => {{
            let input: TensorMap<_, f64> = TensorMap::diagonal(&runtime, &$leg, $spectra).unwrap();
            for (rows, cols) in [([0], [1]), ([1], [0])] {
                let permuted = input.permute(&rows, &cols).unwrap();
                let dense = permuted.materialize().unwrap();
                for operation in 0..4 {
                    let (a, b) = match operation {
                        0 => input.qr_compact(&rows, &cols).unwrap().pair(),
                        1 => input.qr_full(&rows, &cols).unwrap().pair(),
                        2 => input.lq_compact(&rows, &cols).unwrap().pair(),
                        _ => input.lq_full(&rows, &cols).unwrap().pair(),
                    };
                    assert_eq!(a.codomain(), permuted.codomain());
                    assert_eq!(a.domain(), permuted.domain());
                    assert_eq!(b.codomain(), permuted.codomain());
                    assert_eq!(b.domain(), permuted.domain());
                    assert!(a.diagview().is_ok());
                    assert!(b.diagview().is_ok());
                    let reconstructed = a.compose(&b).unwrap().materialize().unwrap();
                    assert_typed_map_close(&reconstructed, &dense, 1e-12);
                    let q = if operation < 2 { &a } else { &b };
                    let gram = q.adjoint().unwrap().compose(q).unwrap();
                    let expected =
                        TensorMap::isomorphism(&runtime, &q.domain(), &q.domain()).unwrap();
                    assert_typed_map_close(
                        &gram.materialize().unwrap(),
                        &expected.materialize().unwrap(),
                        1e-12,
                    );
                }
            }
        }};
    }
    let u1 = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)],
    )
    .unwrap()
    .try_dual()
    .unwrap();
    check!(
        u1,
        [
            SectorSpectrum {
                sector: U1Irrep::new(-1),
                values: vec![-2.0, 0.0]
            },
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![1.0, -3.0, 2.0]
            },
        ]
    );
    let spin0 = SU2Irrep::from_twice_spin(0);
    let half = SU2Irrep::from_twice_spin(1);
    let su2 = GradedSpace::try_new(Arc::new(SU2FusionRule), [(spin0, 2), (half, 1)]).unwrap();
    check!(
        su2,
        [
            SectorSpectrum {
                sector: spin0,
                values: vec![2.0, -2.0]
            },
            SectorSpectrum {
                sector: half,
                values: vec![0.0]
            },
        ]
    );
    let even = product_sector(U1Irrep::new(0), Z2Irrep::EVEN);
    let odd = product_sector(U1Irrep::new(1), Z2Irrep::ODD);
    let fermion = GradedSpace::try_new(
        Arc::new(
            ProductFusionRule::<_, _>::try_new(U1FusionRule, FermionParityFusionRule).unwrap(),
        ),
        [(even, 2), (odd, 1)],
    )
    .unwrap();
    check!(
        fermion,
        [
            SectorSpectrum {
                sector: even,
                values: vec![-2.0, 0.0]
            },
            SectorSpectrum {
                sector: odd,
                values: vec![3.0]
            },
        ]
    );
    let zero = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 0), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    check!(
        zero,
        [SectorSpectrum {
            sector: U1Irrep::new(1),
            values: vec![0.0, -3.0],
        }]
    );
    let empty = GradedSpace::try_new(Arc::new(U1FusionRule), []).unwrap();
    check!(empty, Vec::<SectorSpectrum<_, f64>>::new());
}

#[test]
fn compact_diagonal_qr_lq_scales_complex_subnormals_before_normalizing() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 1)]).unwrap();
    macro_rules! check {
        ($ty:ty, $tiny:expr, $tol:expr) => {{
            let tiny = $tiny;
            let input: TensorMap<_, $ty> = TensorMap::diagonal(
                &runtime,
                &leg,
                [SectorSpectrum {
                    sector: U1Irrep::new(0),
                    values: vec![<$ty>::new(tiny, tiny)],
                }],
            )
            .unwrap();
            for operation in 0..4 {
                let (w, p) = match operation {
                    0 => input.qr_compact(&[0], &[1]).unwrap().pair(),
                    1 => input.qr_full(&[0], &[1]).unwrap().pair(),
                    2 => {
                        let Lq { l, q } = input.lq_compact(&[0], &[1]).unwrap();
                        (q, l)
                    }
                    _ => {
                        let Lq { l, q } = input.lq_full(&[0], &[1]).unwrap();
                        (q, l)
                    }
                };
                let phase = w.diagview().unwrap()[0].values[0].widen_complex();
                let magnitude = p.diagview().unwrap()[0].values[0].widen_complex().re;
                assert!((phase.re - std::f64::consts::FRAC_1_SQRT_2).abs() < $tol);
                assert!((phase.im - std::f64::consts::FRAC_1_SQRT_2).abs() < $tol);
                assert!(magnitude.is_finite() && magnitude > 0.0);
                assert!((phase.norm() - 1.0).abs() <= $tol);
                let rebuilt = phase * magnitude;
                let original = input.diagview().unwrap()[0].values[0].widen_complex();
                // The minimum subnormal rounds its magnitude to one ULP.
                assert!((rebuilt.re - original.re).abs() <= tiny as f64);
                assert!((rebuilt.im - original.im).abs() <= tiny as f64);
            }
        }};
    }
    check!(num_complex::Complex32, f32::from_bits(1), 2e-6);
    check!(Complex64, f64::from_bits(1), 1e-12);
}

/// Hand-computed MAK `sign_safe` / `abs` of the finite `MAX (1 + i)`: symtn's
/// scaled unit phase `(1 + i) / sqrt(2)`, and a magnitude that overflows to
/// `Inf`.
fn assert_overflow_phase_magnitude(phase: Complex64, magnitude: Complex64, tolerance: f64) {
    let half = std::f64::consts::FRAC_1_SQRT_2;
    assert!(
        (phase - Complex64::new(half, half)).norm() <= tolerance,
        "{phase}"
    );
    assert_eq!(magnitude, Complex64::new(f64::INFINITY, 0.0));
}

#[test]
fn compact_diagonal_qr_lq_refuse_nonfinite_and_are_direct_when_overflowing() {
    // A nonfinite value is refused by the shared finite-input stage, as on
    // the dense route (#1986). A finite overflowing value is MAK
    // `_diagonal_qr!` / `lq_diagonal!` (`positive = true`): `q` the phase,
    // `r` (`l`) the magnitude. No materialization and no dense QR.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 1)]).unwrap();
    for value in [
        Complex64::new(f64::NAN, 0.0),
        Complex64::new(f64::INFINITY, 1.0),
        Complex64::new(f64::MAX, f64::MAX),
    ] {
        let input: TensorMap<_, Complex64> = TensorMap::diagonal(
            &runtime,
            &leg,
            [SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![value],
            }],
        )
        .unwrap();
        let first =
            |tensor: &TensorMap<U1FusionRule, Complex64>| tensor.diagview().unwrap()[0].values[0];
        let densified = probe::record();
        if value.re.is_finite() {
            for Qr { q, r } in [
                input.qr_compact(&[0], &[1]).unwrap(),
                input.qr_full(&[0], &[1]).unwrap(),
            ] {
                assert_overflow_phase_magnitude(first(&q), first(&r), 4.0 * f64::EPSILON);
            }
            for Lq { l, q } in [
                input.lq_compact(&[0], &[1]).unwrap(),
                input.lq_full(&[0], &[1]).unwrap(),
            ] {
                assert_overflow_phase_magnitude(first(&q), first(&l), 4.0 * f64::EPSILON);
            }
        } else {
            for (error, family) in [
                (input.qr_compact(&[0], &[1]).map(drop).unwrap_err(), "qr"),
                (input.qr_full(&[0], &[1]).map(drop).unwrap_err(), "qr"),
                (input.lq_compact(&[0], &[1]).map(drop).unwrap_err(), "lq"),
                (input.lq_full(&[0], &[1]).map(drop).unwrap_err(), "lq"),
            ] {
                assert!(is_finite_input_refusal(&error, family), "{error:?}");
            }
        }
        assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    }
    let input: TensorMap<_, num_complex::Complex32> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![num_complex::Complex32::new(f32::MAX, f32::MAX)],
        }],
    )
    .unwrap();
    let widen = |tensor: &TensorMap<U1FusionRule, num_complex::Complex32>| {
        let value = tensor.diagview().unwrap()[0].values[0];
        Complex64::new(f64::from(value.re), f64::from(value.im))
    };
    let densified = probe::record();
    let Qr { q, r } = input.qr_compact(&[0], &[1]).unwrap();
    assert_overflow_phase_magnitude(widen(&q), widen(&r), 4.0 * f64::from(f32::EPSILON));
    let Lq { l, q } = input.lq_full(&[0], &[1]).unwrap();
    assert_overflow_phase_magnitude(widen(&q), widen(&l), 4.0 * f64::from(f32::EPSILON));
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
}

#[test]
fn compact_diagonal_polar_matches_hand_phase_and_absolute_value_all_scalars() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 4)]).unwrap();
    macro_rules! check {
        ($ty:ty, $values:expr, $phase:expr, $magnitude:expr, $tol:expr) => {{
            let input: TensorMap<_, $ty> = TensorMap::diagonal(
                &runtime,
                &leg,
                [SectorSpectrum {
                    sector: U1Irrep::new(0),
                    values: $values,
                }],
            )
            .unwrap();
            let saved = input.diagview().unwrap();
            for left in [true, false] {
                let (w, p) = if left {
                    let LeftPolar { w, p } = input.left_polar(&[0], &[1]).unwrap();
                    (w, p)
                } else {
                    let RightPolar { p, wh } = input.right_polar(&[0], &[1]).unwrap();
                    (wh, p)
                };
                assert_eq!(w.codomain(), input.codomain());
                assert_eq!(w.domain(), input.domain());
                assert_eq!(
                    p.codomain(),
                    if left {
                        input.domain()
                    } else {
                        input.codomain()
                    }
                );
                assert_eq!(p.domain(), p.codomain());
                assert!(w.dense_data().is_err());
                assert!(p.dense_data().is_err());
                assert!(w.diagview().is_ok());
                assert!(p.diagview().is_ok());
                let w = w.materialize().unwrap();
                let p = p.materialize().unwrap();
                for (i, (&actual_w, &actual_p)) in w
                    .dense_data()
                    .unwrap()
                    .iter()
                    .zip(p.dense_data().unwrap())
                    .enumerate()
                {
                    let row = i % 4;
                    let col = i / 4;
                    let expected_w = if row == col {
                        $phase[row]
                    } else {
                        Complex64::new(0.0, 0.0)
                    };
                    let expected_p = if row == col { $magnitude[row] } else { 0.0 };
                    assert!((actual_w.widen_complex() - expected_w).norm() <= $tol);
                    assert!((actual_p.widen_complex().re - expected_p).abs() <= $tol);
                }
                assert_typed_map_close(
                    &w.compose(&p).unwrap(),
                    &input.materialize().unwrap(),
                    $tol,
                );
                let unit = w.adjoint().unwrap().compose(&w).unwrap();
                for (i, &value) in unit
                    .materialize()
                    .unwrap()
                    .dense_data()
                    .unwrap()
                    .iter()
                    .enumerate()
                {
                    let expected = if i % 4 == i / 4 { 1.0 } else { 0.0 };
                    assert!((value.widen_complex() - Complex64::new(expected, 0.0)).norm() <= $tol);
                }
            }
            assert_eq!(input.diagview().unwrap(), saved);
        }};
    }
    let phase = [
        Complex64::new(-1.0, 0.0),
        Complex64::new(0.6, 0.8),
        Complex64::new(1.0, 0.0),
        Complex64::new(1.0, 0.0),
    ];
    let magnitude = [2.0, 5.0, 0.0, 1.0];
    check!(
        f32,
        vec![-2.0, 5.0, -0.0, 1.0],
        [phase[0], Complex64::new(1.0, 0.0), phase[2], phase[3]],
        magnitude,
        2e-6
    );
    check!(
        f64,
        vec![-2.0, 5.0, -0.0, 1.0],
        [phase[0], Complex64::new(1.0, 0.0), phase[2], phase[3]],
        magnitude,
        1e-12
    );
    check!(
        num_complex::Complex32,
        vec![
            num_complex::Complex32::new(-2.0, 0.0),
            num_complex::Complex32::new(3.0, 4.0),
            num_complex::Complex32::new(-0.0, 0.0),
            num_complex::Complex32::new(1.0, 0.0)
        ],
        phase,
        magnitude,
        2e-6
    );
    check!(
        Complex64,
        vec![
            Complex64::new(-2.0, 0.0),
            Complex64::new(3.0, 4.0),
            Complex64::new(-0.0, 0.0),
            Complex64::new(1.0, 0.0)
        ],
        phase,
        magnitude,
        1e-12
    );
}

#[test]
fn compact_diagonal_polar_preserves_sectors_and_changed_roles() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    macro_rules! check {
        ($leg:expr, $spectra:expr) => {{
            let input: TensorMap<_, f64> = TensorMap::diagonal(&runtime, &$leg, $spectra).unwrap();
            for (rows, cols) in [([0], [1]), ([1], [0])] {
                let permuted = input.permute(&rows, &cols).unwrap();
                let dense = permuted.materialize().unwrap();
                let expected_left = dense.left_polar(&[0], &[1]).unwrap();
                let expected_right = dense.right_polar(&[0], &[1]).unwrap();
                let densified = probe::record();
                let left = input.left_polar(&rows, &cols).unwrap();
                let right = input.right_polar(&rows, &cols).unwrap();
                assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
                if permuted.spectrum().is_some() {
                    for factor in [&left.w, &left.p, &right.p, &right.wh] {
                        assert!(factor.diagview().is_ok());
                        assert!(factor.dense_data().is_err());
                        assert_eq!(factor.codomain(), permuted.codomain());
                        assert_eq!(factor.domain(), permuted.domain());
                    }
                }
                assert_polar_factors(
                    &input,
                    &dense,
                    &(left.w, left.p),
                    &(expected_left.w, expected_left.p),
                    true,
                );
                assert_polar_factors(
                    &input,
                    &dense,
                    &(right.p, right.wh),
                    &(expected_right.p, expected_right.wh),
                    false,
                );
            }
        }};
    }
    let u1 = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)],
    )
    .unwrap()
    .try_dual()
    .unwrap();
    check!(
        u1,
        [
            SectorSpectrum {
                sector: U1Irrep::new(-1),
                values: vec![-2.0, 0.0]
            },
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![1.0, -3.0, 2.0]
            },
        ]
    );
    let spin0 = SU2Irrep::from_twice_spin(0);
    let half = SU2Irrep::from_twice_spin(1);
    let su2 = GradedSpace::try_new(Arc::new(SU2FusionRule), [(spin0, 2), (half, 1)]).unwrap();
    check!(
        su2,
        [
            SectorSpectrum {
                sector: spin0,
                values: vec![2.0, -2.0]
            },
            SectorSpectrum {
                sector: half,
                values: vec![0.0]
            },
        ]
    );
    let even = product_sector(U1Irrep::new(0), Z2Irrep::EVEN);
    let odd = product_sector(U1Irrep::new(1), Z2Irrep::ODD);
    let fermion = GradedSpace::try_new(
        Arc::new(
            ProductFusionRule::<_, _>::try_new(U1FusionRule, FermionParityFusionRule).unwrap(),
        ),
        [(even, 2), (odd, 1)],
    )
    .unwrap();
    check!(
        fermion,
        [
            SectorSpectrum {
                sector: even,
                values: vec![-2.0, 0.0]
            },
            SectorSpectrum {
                sector: odd,
                values: vec![3.0]
            },
        ]
    );
    let zero = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 0), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    check!(
        zero,
        [SectorSpectrum {
            sector: U1Irrep::new(1),
            values: vec![0.0, -3.0],
        }]
    );
    let empty = GradedSpace::try_new(Arc::new(U1FusionRule), []).unwrap();
    check!(empty, Vec::<SectorSpectrum<_, f64>>::new());
}

#[test]
fn compact_diagonal_polar_scales_complex_subnormals_before_normalizing() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 1)]).unwrap();
    macro_rules! check {
        ($ty:ty, $tiny:expr, $tol:expr) => {{
            let tiny = $tiny;
            let input: TensorMap<_, $ty> = TensorMap::diagonal(
                &runtime,
                &leg,
                [SectorSpectrum {
                    sector: U1Irrep::new(0),
                    values: vec![<$ty>::new(tiny, tiny)],
                }],
            )
            .unwrap();
            for left in [true, false] {
                let (w, p) = if left {
                    let LeftPolar { w, p } = input.left_polar(&[0], &[1]).unwrap();
                    (w, p)
                } else {
                    let RightPolar { p, wh } = input.right_polar(&[0], &[1]).unwrap();
                    (wh, p)
                };
                let phase = w.diagview().unwrap()[0].values[0].widen_complex();
                let magnitude = p.diagview().unwrap()[0].values[0].widen_complex().re;
                assert!((phase.re - std::f64::consts::FRAC_1_SQRT_2).abs() < $tol);
                assert!((phase.im - std::f64::consts::FRAC_1_SQRT_2).abs() < $tol);
                assert!(magnitude.is_finite() && magnitude > 0.0);
            }
        }};
    }
    check!(num_complex::Complex32, f32::from_bits(1), 2e-6);
    check!(Complex64, f64::from_bits(1), 1e-12);
}

#[test]
fn compact_diagonal_polar_refuses_nonfinite_and_is_direct_when_overflowing() {
    // A nonfinite value is refused by the shared finite-input stage; a finite
    // overflowing one is MAK's diagonal polar `W = sign_safe(a)`, `P = abs(a)`.
    // No materialization and no dense SVD.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 1)]).unwrap();
    for value in [
        Complex64::new(f64::NAN, 0.0),
        Complex64::new(f64::INFINITY, 1.0),
        Complex64::new(f64::MAX, f64::MAX),
    ] {
        let input: TensorMap<_, Complex64> = TensorMap::diagonal(
            &runtime,
            &leg,
            [SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![value],
            }],
        )
        .unwrap();
        let first =
            |tensor: &TensorMap<U1FusionRule, Complex64>| tensor.diagview().unwrap()[0].values[0];
        let densified = probe::record();
        if value.re.is_finite() {
            let LeftPolar { w, p } = input.left_polar(&[0], &[1]).unwrap();
            assert_overflow_phase_magnitude(first(&w), first(&p), 4.0 * f64::EPSILON);
            let RightPolar { p, wh } = input.right_polar(&[0], &[1]).unwrap();
            assert_overflow_phase_magnitude(first(&wh), first(&p), 4.0 * f64::EPSILON);
        } else {
            for error in [
                input.left_polar(&[0], &[1]).map(drop).unwrap_err(),
                input.right_polar(&[0], &[1]).map(drop).unwrap_err(),
            ] {
                assert!(is_finite_input_refusal(&error, "polar"), "{error:?}");
            }
        }
        assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    }
}

/// Multiplicity-free QR of a compact diagonal through the single entry.
fn qr_diagonal<R: MultiplicityFreeRigidSymbols<Scalar = f64>>(
    space: &BoundDynamicFusionMapSpace<R>,
    spectrum: &[symtn_matrixalgebra::SectorSpectrum<f64>],
) -> Result<Qr<symtn_matrixalgebra::seam::FactorOutput<R, f64>>, symtn_tensors::OperationError> {
    symtn_matrixalgebra::seam::qr_compact_from_source::<MultiplicityFreeAdmissionMode, _, _, _, _>(
        &mut symtn_dense::DefaultDenseExecutor::new(),
        symtn_matrixalgebra::seam::FactorSource::Diagonal { space, spectrum },
    )
}
