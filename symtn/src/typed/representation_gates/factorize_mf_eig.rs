use super::*;

#[test]
fn compact_diagonal_eigh_full_skips_dense_input_and_solver() {
    let calls = Arc::new(SpyCounts::default());
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(eigh_full_spy(&calls)))
        .build()
        .unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 3)]).unwrap();
    let input: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![
                Complex64::new(2.0, 0.0),
                Complex64::new(-2.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        }],
    )
    .unwrap();
    let densified = probe::record();
    let Eigh { d, v } = input.eigh_full(&[0], &[1], HermitianTol::DEFAULT).unwrap();
    assert_eq!(
        d.diagview().unwrap()[0].values,
        vec![
            Complex64::new(-2.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0)
        ]
    );
    assert_eq!(v.dense_data().unwrap().len(), 9);
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    assert_eq!(calls.total(), 0);
    let densified = probe::record();
    let _ = input.eigh_full(&[1], &[0], HermitianTol::DEFAULT).unwrap();
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    assert_eq!(calls.total(), 0);
}

#[test]
fn compact_diagonal_eigh_full_hand_permutation_all_scalars() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 3)]).unwrap();
    macro_rules! check {
        ($dtype:ty, $values:expr, $sorted:expr, $zero:expr, $one:expr) => {{
            let input: TensorMap<_, $dtype> = TensorMap::diagonal(
                &runtime,
                &leg,
                [SectorSpectrum {
                    sector: U1Irrep::new(0),
                    values: $values,
                }],
            )
            .unwrap();
            let saved = input.diagview().unwrap();
            let dense = input
                .materialize()
                .unwrap()
                .eigh_full(&[0], &[1], HermitianTol::DEFAULT)
                .unwrap();
            let densified = probe::record();
            let Eigh { d, v } = input.eigh_full(&[0], &[1], HermitianTol::DEFAULT).unwrap();
            assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
            assert_eq!(input.diagview().unwrap(), saved);
            assert_eq!(d.diagview().unwrap()[0].values, $sorted);
            assert_eq!(d.codomain(), v.domain());
            assert_eq!(d.domain(), v.domain());
            assert_eq!(v.codomain(), input.codomain());
            assert_eq!(v.domain(), dense.v.domain());
            assert!(!v.domain()[0].is_dual());
            let z: $dtype = $zero;
            let o: $dtype = $one;
            assert_eq!(v.dense_data().unwrap(), [z, o, z, z, z, o, o, z, z]);
            assert_eq!(
                input.compose(&v).unwrap().dense_data().unwrap(),
                v.compose(&d).unwrap().dense_data().unwrap()
            );
            assert_eq!(
                v.compose(&d)
                    .unwrap()
                    .compose(&v.adjoint().unwrap())
                    .unwrap()
                    .dense_data()
                    .unwrap(),
                input.materialize().unwrap().dense_data().unwrap()
            );
            assert_eq!(
                v.adjoint()
                    .unwrap()
                    .compose(&v)
                    .unwrap()
                    .dense_data()
                    .unwrap(),
                [o, z, z, z, o, z, z, z, o]
            );
        }};
    }
    check!(
        f32,
        vec![2.0_f32, -2.0, 1.0],
        vec![-2.0_f32, 1.0, 2.0],
        0.0_f32,
        1.0_f32
    );
    check!(
        f64,
        vec![2.0_f64, -2.0, 1.0],
        vec![-2.0_f64, 1.0, 2.0],
        0.0_f64,
        1.0_f64
    );
    check!(
        num_complex::Complex32,
        vec![
            num_complex::Complex32::new(2.0, 0.0),
            num_complex::Complex32::new(-2.0, 0.0),
            num_complex::Complex32::new(1.0, 0.0)
        ],
        vec![
            num_complex::Complex32::new(-2.0, 0.0),
            num_complex::Complex32::new(1.0, 0.0),
            num_complex::Complex32::new(2.0, 0.0)
        ],
        num_complex::Complex32::new(0.0, 0.0),
        num_complex::Complex32::new(1.0, 0.0)
    );
    check!(
        Complex64,
        vec![
            Complex64::new(2.0, 0.0),
            Complex64::new(-2.0, 0.0),
            Complex64::new(1.0, 0.0)
        ],
        vec![
            Complex64::new(-2.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0)
        ],
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0)
    );
}

#[test]
fn compact_diagonal_eigh_full_signed_zeros_remain_a_valid_eigenbasis() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let input: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![0.0, -0.0],
        }],
    )
    .unwrap();
    let dense = input
        .materialize()
        .unwrap()
        .eigh_full(&[0], &[1], HermitianTol::DEFAULT)
        .unwrap();
    let densified = probe::record();
    let Eigh { d, v } = input.eigh_full(&[0], &[1], HermitianTol::DEFAULT).unwrap();
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    assert_eq!(
        d.diagview().unwrap()[0].values,
        dense.d.diagview().unwrap()[0].values
    );
    let av = input.compose(&v).unwrap();
    let vd = v.compose(&d).unwrap();
    assert_eq!(av.dense_data().unwrap(), vd.dense_data().unwrap());
    assert_eq!(
        vd.compose(&v.adjoint().unwrap())
            .unwrap()
            .dense_data()
            .unwrap(),
        input.materialize().unwrap().dense_data().unwrap()
    );
    assert_eq!(
        v.adjoint()
            .unwrap()
            .compose(&v)
            .unwrap()
            .dense_data()
            .unwrap(),
        [1.0, 0.0, 0.0, 1.0]
    );
}

#[test]
fn compact_diagonal_eigh_full_preserves_sector_spaces_and_zero_regions() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    macro_rules! check {
        ($leg:expr, $spectra:expr, $expected:expr) => {{
            let input: TensorMap<_, f64> = TensorMap::diagonal(&runtime, &$leg, $spectra).unwrap();
            let saved = input.diagview().unwrap();
            let dense = input
                .materialize()
                .unwrap()
                .eigh_full(&[0], &[1], HermitianTol::DEFAULT)
                .unwrap();
            let densified = probe::record();
            let Eigh { d, v } = input.eigh_full(&[0], &[1], HermitianTol::DEFAULT).unwrap();
            assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
            assert_eq!(input.diagview().unwrap(), saved);
            assert_eq!(d.diagview().unwrap(), $expected);
            assert_eq!(v.codomain(), input.codomain());
            assert_eq!(v.domain(), dense.v.domain());
            assert!(!v.domain()[0].is_dual());
            assert_eq!(d.codomain(), v.domain());
            assert_eq!(d.domain(), v.domain());
            let av = input.compose(&v).unwrap();
            let vd = v.compose(&d).unwrap();
            for (&a, &b) in av
                .dense_data()
                .unwrap()
                .iter()
                .zip(vd.dense_data().unwrap())
            {
                assert!((a - b).abs() <= 1e-12);
            }
            let recon = vd.compose(&v.adjoint().unwrap()).unwrap();
            for (&a, &b) in recon
                .dense_data()
                .unwrap()
                .iter()
                .zip(input.materialize().unwrap().dense_data().unwrap())
            {
                assert!((a - b).abs() <= 1e-12);
            }
            let unit = v.adjoint().unwrap().compose(&v).unwrap();
            let bond = v.domain();
            let expected_unit = TensorMap::<_, f64>::from_subblock_fn(
                &runtime,
                [&bond[0]],
                [&bond[0]],
                |_, indices| f64::from(indices[0] == indices[1]),
            )
            .unwrap();
            for (&a, &b) in unit
                .dense_data()
                .unwrap()
                .iter()
                .zip(expected_unit.dense_data().unwrap())
            {
                assert!((a - b).abs() <= 1e-12);
            }
        }};
    }
    let u1 = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    let dual = u1.try_dual().unwrap();
    check!(
        dual,
        [
            SectorSpectrum {
                sector: U1Irrep::new(-1),
                values: vec![2.0, -2.0]
            },
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![1.0, -4.0, 0.0]
            },
        ],
        vec![
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![-4.0, 0.0, 1.0]
            },
            SectorSpectrum {
                sector: U1Irrep::new(-1),
                values: vec![-2.0, 2.0]
            },
        ]
    );
    let spin0 = SU2Irrep::from_twice_spin(0);
    let spin_half = SU2Irrep::from_twice_spin(1);
    let su2 = GradedSpace::try_new(Arc::new(SU2FusionRule), [(spin0, 2), (spin_half, 1)]).unwrap();
    check!(
        su2,
        [
            SectorSpectrum {
                sector: spin_half,
                values: vec![-3.0]
            },
            SectorSpectrum {
                sector: spin0,
                values: vec![2.0, -2.0]
            },
        ],
        vec![
            SectorSpectrum {
                sector: spin0,
                values: vec![-2.0, 2.0]
            },
            SectorSpectrum {
                sector: spin_half,
                values: vec![-3.0]
            },
        ]
    );
    let even = product_sector(U1Irrep::new(0), Z2Irrep::EVEN);
    let odd = product_sector(U1Irrep::new(1), Z2Irrep::ODD);
    let fermion = GradedSpace::try_new(
        Arc::new(U1FusionRule.product(FermionParityFusionRule)),
        [(even, 2), (odd, 1)],
    )
    .unwrap();
    check!(
        fermion,
        [
            SectorSpectrum {
                sector: odd,
                values: vec![-3.0]
            },
            SectorSpectrum {
                sector: even,
                values: vec![2.0, -2.0]
            },
        ],
        vec![
            SectorSpectrum {
                sector: even,
                values: vec![-2.0, 2.0]
            },
            SectorSpectrum {
                sector: odd,
                values: vec![-3.0]
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
            values: vec![2.0, -3.0]
        }],
        vec![SectorSpectrum {
            sector: U1Irrep::new(1),
            values: vec![-3.0, 2.0]
        },]
    );
    let empty = GradedSpace::try_new(Arc::new(U1FusionRule), []).unwrap();
    check!(
        empty,
        Vec::<SectorSpectrum<_, f64>>::new(),
        Vec::<SectorSpectrum<_, f64>>::new()
    );
}

#[test]
fn compact_diagonal_eigh_full_is_direct_for_near_hermitian_and_refuses_nonfinite_input() {
    let calls = Arc::new(SpyCounts::default());
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(eigh_full_spy(&calls)))
        .build()
        .unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 1)]).unwrap();
    let near: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![Complex64::new(1.0, 1e-15)],
        }],
    )
    .unwrap();
    // The dense route's relative Hermiticity check passes `1 + 1e-15 i`, and
    // the eigenvalue is its real part; the diagonal applies the same check
    // directly, with no materialization and no dense EIGH.
    let densified = probe::record();
    let Eigh { d, .. } = near.eigh_full(&[0], &[1], HermitianTol::DEFAULT).unwrap();
    assert_eq!(d.diagview().unwrap()[0].values, [Complex64::new(1.0, 0.0)]);
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    assert_eq!(calls.total(), 0);

    // A nonfinite diagonal is refused by the shared finite-input stage before
    // any dense work.
    for value in [f64::NAN, f64::INFINITY] {
        let input: TensorMap<_, f64> = TensorMap::diagonal(
            &runtime,
            &leg,
            [SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![value],
            }],
        )
        .unwrap();
        let densified = probe::record();
        let actual = input
            .eigh_full(&[0], &[1], HermitianTol::DEFAULT)
            .map(drop)
            .unwrap_err();
        assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
        assert!(is_finite_input_refusal(&actual, "eigh"), "{actual:?}");
    }
    let extreme: TensorMap<_, f32> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![f32::MAX],
        }],
    )
    .unwrap();
    let calls_before = calls.total();
    let densified = probe::record();
    assert_eq!(
        extreme
            .eigh_full(&[0], &[1], HermitianTol::DEFAULT)
            .unwrap()
            .d
            .diagview()
            .unwrap()[0]
            .values,
        [f32::MAX]
    );
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    assert_eq!(calls.total(), calls_before);
    let input: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![1.0],
        }],
    )
    .unwrap();
    let densified = probe::record();
    assert!(matches!(
        input.eigh_full(&[0, 1], &[], HermitianTol::DEFAULT),
        Err(Error::Operation(error))
            if matches!(error.as_ref(), symtn_tensors::OperationError::SpaceMismatch {
                message: "eigh requires an endomorphism (codomain == domain)"
            })
    ));
    assert_eq!(densified.get(Event::DiagonalMaterialization), 1);
}

#[test]
fn compact_diagonal_eigh_full_changed_roles_match_explicit_dense_permute() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 1), (U1Irrep::new(1), 2)],
    )
    .unwrap()
    .try_dual()
    .unwrap();
    let input: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: U1Irrep::new(-1),
                values: vec![-2.0, 3.0],
            },
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![1.0],
            },
        ],
    )
    .unwrap();
    let swapped = input.permute(&[1], &[0]).unwrap();
    assert_ne!(swapped.diagview().unwrap(), input.diagview().unwrap());
    let reference = swapped
        .materialize()
        .unwrap()
        .eigh_full(&[0], &[1], HermitianTol::DEFAULT)
        .unwrap();
    let densified = probe::record();
    let Eigh { d, v } = input.eigh_full(&[1], &[0], HermitianTol::DEFAULT).unwrap();
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    assert_eq!(d.diagview().unwrap(), reference.d.diagview().unwrap());
    assert_eq!(d.codomain(), reference.d.codomain());
    assert_eq!(d.domain(), reference.d.domain());
    assert_eq!(v.codomain(), reference.v.codomain());
    assert_eq!(v.domain(), reference.v.domain());
    assert!(!v.domain()[0].is_dual());
    let av = swapped.compose(&v).unwrap();
    let vd = v.compose(&d).unwrap();
    for (&a, &b) in av
        .dense_data()
        .unwrap()
        .iter()
        .zip(vd.dense_data().unwrap())
    {
        assert!((a - b).abs() <= 1e-12);
    }
    let recon = vd.compose(&v.adjoint().unwrap()).unwrap();
    for (&a, &b) in recon
        .dense_data()
        .unwrap()
        .iter()
        .zip(swapped.materialize().unwrap().dense_data().unwrap())
    {
        assert!((a - b).abs() <= 1e-12);
    }
}

#[test]
fn compact_diagonal_eig_full_skips_dense_input_and_solver() {
    let calls = Arc::new(SpyCounts::default());
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(eig_full_spy(&calls)))
        .build()
        .unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 3)]).unwrap();
    let input: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![
                Complex64::new(1.0, 1.0),
                Complex64::new(-4.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        }],
    )
    .unwrap();
    let densified = probe::record();
    let Eig { d, v } = input.eig_full(&[0], &[1]).unwrap();
    assert_eq!(
        d.diagview().unwrap()[0].values,
        vec![
            Complex64::new(-4.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 1.0)
        ]
    );
    assert_eq!(v.dense_data().unwrap().len(), 9);
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    assert_eq!(calls.of(EIG_FULL), 0);
    let densified = probe::record();
    let _ = input.eig_full(&[1], &[0]).unwrap();
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    assert_eq!(calls.of(EIG_FULL), 0);
}

#[test]
fn compact_diagonal_eig_full_hand_permutation_all_scalars() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 3)]).unwrap();
    macro_rules! check {
        ($dtype:ty, $eig:ty, $values:expr, $expected:expr, $zero:expr, $one:expr, $tol:expr, $promote:expr) => {{
            let input: TensorMap<_, $dtype> = TensorMap::diagonal(
                &runtime,
                &leg,
                [SectorSpectrum {
                    sector: U1Irrep::new(0),
                    values: $values,
                }],
            )
            .unwrap();
            let original = input.diagview().unwrap();
            let densified = probe::record();
            let Eig { d, v } = input.eig_full(&[0], &[1]).unwrap();
            assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
            assert_eq!(input.diagview().unwrap(), original);
            assert_eq!(d.diagview().unwrap()[0].values, $expected);
            assert_eq!(d.codomain(), v.domain());
            assert_eq!(d.domain(), v.domain());
            assert_eq!(v.codomain(), input.codomain());
            assert!(!v.domain()[0].is_dual());
            let z: $eig = $zero;
            let o: $eig = $one;
            assert_eq!(v.dense_data().unwrap(), [z, o, z, z, z, o, o, z, z]);
            let promoted = ($promote)(&input);
            let av = promoted.compose(&v).unwrap();
            let vd = v.compose(&d).unwrap();
            assert_typed_map_close(&av, &vd, $tol);
            assert_typed_map_close(
                &vd.compose(&v.adjoint().unwrap()).unwrap(),
                &promoted.materialize().unwrap(),
                $tol,
            );
            let unit = v.adjoint().unwrap().compose(&v).unwrap();
            assert_eq!(unit.dense_data().unwrap(), [o, z, z, z, o, z, z, z, o]);
        }};
    }
    check!(
        f32,
        num_complex::Complex32,
        vec![1.0_f32, -4.0, 0.25],
        vec![
            num_complex::Complex32::new(-4.0, 0.0),
            num_complex::Complex32::new(0.25, 0.0),
            num_complex::Complex32::new(1.0, 0.0)
        ],
        num_complex::Complex32::new(0.0, 0.0),
        num_complex::Complex32::new(1.0, 0.0),
        1e-5,
        |input: &TensorMap<U1FusionRule, f32>| input.convert::<num_complex::Complex32>()
    );
    check!(
        f64,
        Complex64,
        vec![1.0_f64, -4.0, 0.25],
        vec![
            Complex64::new(-4.0, 0.0),
            Complex64::new(0.25, 0.0),
            Complex64::new(1.0, 0.0)
        ],
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        1e-12,
        |input: &TensorMap<U1FusionRule, f64>| input.convert::<Complex64>()
    );
    check!(
        num_complex::Complex32,
        num_complex::Complex32,
        vec![
            num_complex::Complex32::new(1.0, 1.0),
            num_complex::Complex32::new(-4.0, 0.0),
            num_complex::Complex32::new(0.0, 0.25)
        ],
        vec![
            num_complex::Complex32::new(-4.0, 0.0),
            num_complex::Complex32::new(0.0, 0.25),
            num_complex::Complex32::new(1.0, 1.0)
        ],
        num_complex::Complex32::new(0.0, 0.0),
        num_complex::Complex32::new(1.0, 0.0),
        1e-5,
        |input: &TensorMap<U1FusionRule, num_complex::Complex32>| input.clone()
    );
    check!(
        Complex64,
        Complex64,
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(-4.0, 0.0),
            Complex64::new(0.0, 0.25)
        ],
        vec![
            Complex64::new(-4.0, 0.0),
            Complex64::new(0.0, 0.25),
            Complex64::new(1.0, 1.0)
        ],
        Complex64::new(0.0, 0.0),
        Complex64::new(1.0, 0.0),
        1e-12,
        |input: &TensorMap<U1FusionRule, Complex64>| input.clone()
    );
}

#[test]
fn compact_diagonal_eig_full_preserves_sector_spaces_and_zero_regions() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    macro_rules! check {
        ($leg:expr, $spectra:expr, $expected:expr) => {{
            let input: TensorMap<_, Complex64> =
                TensorMap::diagonal(&runtime, &$leg, $spectra).unwrap();
            let original = input.diagview().unwrap();
            let dense = input.materialize().unwrap().eig_full(&[0], &[1]).unwrap();
            let densified = probe::record();
            let Eig { d, v } = input.eig_full(&[0], &[1]).unwrap();
            assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
            assert_eq!(input.diagview().unwrap(), original);
            assert_eq!(d.diagview().unwrap(), $expected);
            assert_eq!(d.codomain(), v.domain());
            assert_eq!(d.domain(), v.domain());
            assert_eq!(v.codomain(), input.codomain());
            assert_eq!(v.domain(), dense.v.domain());
            assert!(!v.domain()[0].is_dual());
            let av = input.compose(&v).unwrap();
            let vd = v.compose(&d).unwrap();
            assert_typed_map_close(&av, &vd, 1e-12);
            assert_typed_map_close(
                &vd.compose(&v.adjoint().unwrap()).unwrap(),
                &input.materialize().unwrap(),
                1e-12,
            );
            let unit = v.adjoint().unwrap().compose(&v).unwrap();
            let bond = v.domain();
            let expected_unit = TensorMap::<_, Complex64>::from_subblock_fn(
                &runtime,
                [&bond[0]],
                [&bond[0]],
                |_, indices| Complex64::new(f64::from(indices[0] == indices[1]), 0.0),
            )
            .unwrap();
            assert_typed_map_close(&unit, &expected_unit, 1e-12);
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
                values: vec![Complex64::new(0.0, 2.0), Complex64::new(3.0, 0.0)],
            },
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, -4.0),
                    Complex64::new(0.0, 0.0)
                ],
            },
        ],
        vec![
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![
                    Complex64::new(0.0, -4.0),
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0)
                ],
            },
            SectorSpectrum {
                sector: U1Irrep::new(-1),
                values: vec![Complex64::new(0.0, 2.0), Complex64::new(3.0, 0.0)],
            },
        ]
    );
    let spin0 = SU2Irrep::from_twice_spin(0);
    let spin_half = SU2Irrep::from_twice_spin(1);
    let su2 = GradedSpace::try_new(Arc::new(SU2FusionRule), [(spin0, 2), (spin_half, 1)]).unwrap();
    check!(
        su2,
        [
            SectorSpectrum {
                sector: spin_half,
                values: vec![Complex64::new(0.0, -3.0)]
            },
            SectorSpectrum {
                sector: spin0,
                values: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 1.0)]
            },
        ],
        vec![
            SectorSpectrum {
                sector: spin0,
                values: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 1.0)]
            },
            SectorSpectrum {
                sector: spin_half,
                values: vec![Complex64::new(0.0, -3.0)]
            },
        ]
    );
    let even = product_sector(U1Irrep::new(0), Z2Irrep::EVEN);
    let odd = product_sector(U1Irrep::new(1), Z2Irrep::ODD);
    let fermion = GradedSpace::try_new(
        Arc::new(U1FusionRule.product(FermionParityFusionRule)),
        [(even, 2), (odd, 1)],
    )
    .unwrap();
    check!(
        fermion,
        [
            SectorSpectrum {
                sector: odd,
                values: vec![Complex64::new(0.0, -3.0)]
            },
            SectorSpectrum {
                sector: even,
                values: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 1.0)]
            },
        ],
        vec![
            SectorSpectrum {
                sector: even,
                values: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 1.0)]
            },
            SectorSpectrum {
                sector: odd,
                values: vec![Complex64::new(0.0, -3.0)]
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
            values: vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, -3.0)]
        }],
        vec![SectorSpectrum {
            sector: U1Irrep::new(1),
            values: vec![Complex64::new(0.0, -3.0), Complex64::new(1.0, 0.0)]
        }]
    );
    let empty = GradedSpace::try_new(Arc::new(U1FusionRule), []).unwrap();
    check!(
        empty,
        Vec::<SectorSpectrum<_, Complex64>>::new(),
        Vec::<SectorSpectrum<_, Complex64>>::new()
    );
}

#[test]
fn compact_diagonal_eig_full_changed_roles_match_explicit_dense_permute() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 1), (U1Irrep::new(1), 2)],
    )
    .unwrap()
    .try_dual()
    .unwrap();
    let input: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: U1Irrep::new(-1),
                values: vec![Complex64::new(-2.0, 0.5), Complex64::new(0.0, 3.0)],
            },
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![Complex64::new(1.0, 0.0)],
            },
        ],
    )
    .unwrap();
    let swapped = input.permute(&[1], &[0]).unwrap();
    assert_ne!(swapped.diagview().unwrap(), input.diagview().unwrap());
    let reference = swapped.materialize().unwrap().eig_full(&[0], &[1]).unwrap();
    let densified = probe::record();
    let Eig { d, v } = input.eig_full(&[1], &[0]).unwrap();
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    assert_eq!(d.diagview().unwrap(), reference.d.diagview().unwrap());
    assert_eq!(d.codomain(), reference.d.codomain());
    assert_eq!(d.domain(), reference.d.domain());
    assert_eq!(v.codomain(), reference.v.codomain());
    assert_eq!(v.domain(), reference.v.domain());
    assert!(!v.domain()[0].is_dual());
    let vd = v.compose(&d).unwrap();
    assert_typed_map_close(&swapped.compose(&v).unwrap(), &vd, 1e-12);
    assert_typed_map_close(
        &vd.compose(&v.adjoint().unwrap()).unwrap(),
        &swapped.materialize().unwrap(),
        1e-12,
    );
}

#[test]
fn compact_diagonal_eig_full_ties_and_signed_zeros_are_valid() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 4)]).unwrap();
    let input: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![
                Complex64::new(0.0, 2.0),
                Complex64::new(0.0, -2.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-0.0, 0.0),
            ],
        }],
    )
    .unwrap();
    let densified = probe::record();
    let Eig { d, v } = input.eig_full(&[0], &[1]).unwrap();
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    let values = &d.diagview().unwrap()[0].values;
    assert_eq!(values.len(), 4);
    // Lexicographic (re, im) under IEEE total order, as MAK's `isless`:
    // the `-0.0` real part precedes every `+0.0` one.
    assert_eq!(values[0].re.to_bits(), (-0.0_f64).to_bits());
    assert!(values.windows(2).all(|pair| pair[0]
        .re
        .total_cmp(&pair[1].re)
        .then(pair[0].im.total_cmp(&pair[1].im))
        .is_le()));
    assert_eq!(
        values
            .iter()
            .filter(|&&value| value == Complex64::new(0.0, 0.0))
            .count(),
        2
    );
    assert!(values.contains(&Complex64::new(0.0, 2.0)));
    assert!(values.contains(&Complex64::new(0.0, -2.0)));
    let vd = v.compose(&d).unwrap();
    assert_typed_map_close(&input.compose(&v).unwrap(), &vd, 1e-12);
    assert_typed_map_close(
        &vd.compose(&v.adjoint().unwrap()).unwrap(),
        &input.materialize().unwrap(),
        1e-12,
    );
    let unit = v.adjoint().unwrap().compose(&v).unwrap();
    assert_eq!(
        unit.dense_data()
            .unwrap()
            .iter()
            .filter(|&&x| x == Complex64::new(1.0, 0.0))
            .count(),
        4
    );
    assert_eq!(
        unit.dense_data()
            .unwrap()
            .iter()
            .filter(|&&x| x == Complex64::new(0.0, 0.0))
            .count(),
        12
    );
}

#[test]
fn compact_diagonal_eig_full_refuses_nonfinite_and_overflowing_values() {
    let calls = Arc::new(SpyCounts::default());
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(eig_full_spy(&calls)))
        .build()
        .unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 1)]).unwrap();
    for value in [
        Complex64::new(f64::NAN, 0.0),
        Complex64::new(f64::INFINITY, 0.0),
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
        // A nonfinite value is refused by the shared finite-input stage, and
        // the overflowing `MAX + MAX i` by the dense route's eigenvalue check;
        // no materialization or dense EIG either way.
        let before = calls.of(EIG_FULL);
        let densified = probe::record();
        let error = input.eig_full(&[0], &[1]).map(drop).unwrap_err();
        assert!(
            is_eig_value_error(&error, value.re.is_finite() && value.im.is_finite()),
            "{error:?}"
        );
        assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
        assert_eq!(calls.of(EIG_FULL), before);
    }
    let input: TensorMap<_, f32> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![f32::MAX],
        }],
    )
    .unwrap();
    let before = calls.of(EIG_FULL);
    let densified = probe::record();
    assert_eq!(
        input.eig_full(&[0], &[1]).unwrap().d.diagview().unwrap()[0].values,
        [num_complex::Complex32::new(f32::MAX, 0.0)]
    );
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    assert_eq!(calls.of(EIG_FULL), before);
    let densified = probe::record();
    assert!(matches!(
        input.eig_full(&[0, 1], &[]),
        Err(Error::Operation(error))
            if matches!(error.as_ref(), symtn_tensors::OperationError::SpaceMismatch {
                message: "eig requires an endomorphism (codomain == domain)"
            })
    ));
    assert_eq!(densified.get(Event::DiagonalMaterialization), 1);
}

#[test]
fn compact_diagonal_eig_vals_skips_dense_input_and_solver() {
    let calls = Arc::new(SpyCounts::default());
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(eig_vals_spy(&calls)))
        .build()
        .unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    let input: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: U1Irrep::new(1),
                values: vec![Complex64::new(0.0, 2.0), Complex64::new(-2.0, 0.0)],
            },
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![
                    Complex64::new(1.0, 1.0),
                    Complex64::new(-4.0, 0.0),
                    Complex64::new(0.0, 0.0),
                ],
            },
        ],
    )
    .unwrap();
    let saved = input.diagview().unwrap();
    let densified = probe::record();
    let result = input.eig_vals(&[0], &[1]).unwrap();
    assert_eq!(
        result[0].values,
        vec![
            Complex64::new(-4.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 1.0)
        ]
    );
    assert_eq!(
        result[1].values,
        vec![Complex64::new(-2.0, 0.0), Complex64::new(0.0, 2.0)]
    );
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    assert_eq!(calls.total(), 0);
    assert_eq!(input.diagview().unwrap(), saved);
}

#[test]
fn compact_diagonal_eig_vals_match_hand_spectra_across_scalars_and_sectors() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    macro_rules! check {
        ($leg:expr, $spectra:expr, $dtype:ty, $expected:expr, $tol:expr) => {{
            let input: TensorMap<_, $dtype> =
                TensorMap::diagonal(&runtime, &$leg, $spectra).unwrap();
            let saved = input.diagview().unwrap();
            let dense_values = input.materialize().unwrap().eig_vals(&[0], &[1]).unwrap();
            let densified = probe::record();
            let actual = input.eig_vals(&[0], &[1]).unwrap();
            assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
            assert_eq!(input.diagview().unwrap(), saved);
            let expected: Vec<Vec<Complex64>> = $expected;
            assert_eq!(actual.len(), expected.len());
            for ((actual, dense), expected) in actual.iter().zip(&dense_values).zip(expected) {
                assert_eq!(actual.sector, dense.sector);
                assert_eq!(actual.values.len(), expected.len());
                let mut actual_values = actual.values.clone();
                let mut dense_values = dense.values.clone();
                let mut expected_values = expected;
                let order = |a: &Complex64, b: &Complex64| {
                    a.re.total_cmp(&b.re).then(a.im.total_cmp(&b.im))
                };
                actual_values.sort_by(order);
                dense_values.sort_by(order);
                expected_values.sort_by(order);
                assert_eq!(actual_values, expected_values);
                for ((actual, dense), expected) in actual_values
                    .iter()
                    .zip(&dense_values)
                    .zip(&expected_values)
                {
                    assert_eq!(actual, expected);
                    assert!((*actual - *dense).norm() <= $tol);
                }
                assert_eq!(actual.values, actual_values);
            }
        }};
    }
    let u1 = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    let real32 = [1.0000001_f32, -3.25, 0.3];
    check!(
        u1,
        [
            SectorSpectrum {
                sector: U1Irrep::new(1),
                values: vec![2.0_f32, -2.0]
            },
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: real32.to_vec()
            },
        ],
        f32,
        vec![
            real32.map(|v| Complex64::new(v as f64, 0.0)).to_vec(),
            vec![Complex64::new(2.0, 0.0), Complex64::new(-2.0, 0.0)],
        ],
        1e-5
    );
    let complex32 = [
        num_complex::Complex32::new(1.0000001, 0.25),
        num_complex::Complex32::new(-3.25, 0.0),
        num_complex::Complex32::new(0.0, -0.3),
    ];
    check!(
        u1,
        [
            SectorSpectrum {
                sector: U1Irrep::new(1),
                values: vec![
                    num_complex::Complex32::new(0.0, 2.0),
                    num_complex::Complex32::new(-2.0, 0.0)
                ]
            },
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: complex32.to_vec()
            },
        ],
        num_complex::Complex32,
        vec![
            complex32
                .map(|v| Complex64::new(v.re as f64, v.im as f64))
                .to_vec(),
            vec![Complex64::new(0.0, 2.0), Complex64::new(-2.0, 0.0)],
        ],
        1e-5
    );
    let dual = u1.try_dual().unwrap();
    check!(
        dual,
        [
            SectorSpectrum {
                sector: U1Irrep::new(-1),
                values: vec![2.0_f64, -2.0]
            },
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![1.5, -4.0, 0.0]
            },
        ],
        f64,
        vec![
            vec![Complex64::new(2.0, 0.0), Complex64::new(-2.0, 0.0)],
            vec![
                Complex64::new(1.5, 0.0),
                Complex64::new(-4.0, 0.0),
                Complex64::new(0.0, 0.0)
            ],
        ],
        1e-12
    );
    let spin0 = SU2Irrep::from_twice_spin(0);
    let spin_half = SU2Irrep::from_twice_spin(1);
    let su2 = GradedSpace::try_new(Arc::new(SU2FusionRule), [(spin0, 2), (spin_half, 1)]).unwrap();
    check!(
        su2,
        [
            SectorSpectrum {
                sector: spin_half,
                values: vec![Complex64::new(0.0, -3.0)]
            },
            SectorSpectrum {
                sector: spin0,
                values: vec![Complex64::new(2.0, 1.0), Complex64::new(-2.0, -1.0)]
            },
        ],
        Complex64,
        vec![
            vec![Complex64::new(2.0, 1.0), Complex64::new(-2.0, -1.0)],
            vec![Complex64::new(0.0, -3.0)],
        ],
        1e-12
    );
    let even = product_sector(U1Irrep::new(0), Z2Irrep::EVEN);
    let odd = product_sector(U1Irrep::new(1), Z2Irrep::ODD);
    let fermion = GradedSpace::try_new(
        Arc::new(U1FusionRule.product(FermionParityFusionRule)),
        [(even, 2), (odd, 1)],
    )
    .unwrap();
    check!(
        fermion,
        [
            SectorSpectrum {
                sector: odd,
                values: vec![Complex64::new(0.0, -3.0)]
            },
            SectorSpectrum {
                sector: even,
                values: vec![Complex64::new(2.0, 1.0), Complex64::new(-2.0, -1.0)]
            },
        ],
        Complex64,
        vec![
            vec![Complex64::new(2.0, 1.0), Complex64::new(-2.0, -1.0)],
            vec![Complex64::new(0.0, -3.0)],
        ],
        1e-12
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
            values: vec![2.0_f64, -3.0]
        }],
        f64,
        vec![vec![Complex64::new(2.0, 0.0), Complex64::new(-3.0, 0.0)]],
        1e-12
    );
    let empty = GradedSpace::try_new(Arc::new(U1FusionRule), []).unwrap();
    let input: TensorMap<_, f64> =
        TensorMap::diagonal(&runtime, &empty, Vec::<SectorSpectrum<_, f64>>::new()).unwrap();
    let densified = probe::record();
    assert!(input.eig_vals(&[0], &[1]).unwrap().is_empty());
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
}

#[test]
fn compact_diagonal_eig_vals_refuses_nonfinite_and_overflowing_values() {
    let calls = Arc::new(SpyCounts::default());
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(eig_vals_spy(&calls)))
        .build()
        .unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 1)]).unwrap();
    for value in [
        Complex64::new(f64::NAN, 0.0),
        Complex64::new(f64::INFINITY, 0.0),
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
        // As `eig_full`: the shared finite-input stage, or the dense route's
        // eigenvalue check for the overflowing `MAX + MAX i`.
        let densified = probe::record();
        let error = input.eig_vals(&[0], &[1]).unwrap_err();
        assert!(
            is_eig_value_error(&error, value.re.is_finite() && value.im.is_finite()),
            "{error:?}"
        );
        assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    }
    assert_eq!(calls.total(), 0);
    let input: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![1.0],
        }],
    )
    .unwrap();
    let densified = probe::record();
    let error = input.eig_vals(&[0, 1], &[]).unwrap_err();
    assert!(matches!(
        error,
        Error::Operation(error)
            if matches!(
                error.as_ref(),
                symtn_tensors::OperationError::SpaceMismatch {
                    message: "eig requires an endomorphism (codomain == domain)"
                }
            )
    ));
    assert_eq!(densified.get(Event::DiagonalMaterialization), 1);
}

#[test]
fn compact_diagonal_eigh_vals_skips_dense_input_and_solver() {
    let calls = Arc::new(SpyCounts::default());
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(eigh_vals_spy(&calls)))
        .build()
        .unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    let source: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: U1Irrep::new(1),
                values: vec![2.0, -2.0],
            },
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![1.0, -4.0, 0.0],
            },
        ],
    )
    .unwrap();
    let densified = probe::record();
    let result = source.eigh_vals(&[0], &[1], HermitianTol::DEFAULT).unwrap();
    assert_eq!(result[0].values, vec![-4.0, 0.0, 1.0]);
    assert_eq!(result[1].values, vec![-2.0, 2.0]);
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    assert_eq!(calls.total(), 0);
}

#[test]
fn compact_diagonal_eigh_vals_match_hand_spectra_across_scalars_and_sectors() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    macro_rules! check {
        ($leg:expr, $spectra:expr, $dtype:ty, $expected:expr, $tol:expr) => {{
            let input: TensorMap<_, $dtype> =
                TensorMap::diagonal(&runtime, &$leg, $spectra).unwrap();
            let saved = input.diagview().unwrap();
            let dense = input.materialize().unwrap();
            let dense_values = dense.eigh_vals(&[0], &[1], HermitianTol::DEFAULT).unwrap();
            let densified = probe::record();
            let actual = input.eigh_vals(&[0], &[1], HermitianTol::DEFAULT).unwrap();
            assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
            assert_eq!(input.diagview().unwrap(), saved);
            let expected: Vec<Vec<f64>> = $expected;
            assert_eq!(actual.len(), expected.len());
            for ((actual, dense), expected) in actual.iter().zip(&dense_values).zip(expected) {
                assert_eq!(actual.sector, dense.sector);
                assert_eq!(actual.values.len(), expected.len());
                for ((&actual, &dense), expected) in
                    actual.values.iter().zip(&dense.values).zip(expected)
                {
                    assert!((actual - expected).abs() <= $tol);
                    assert!((actual - dense).abs() <= $tol);
                }
            }
        }};
    }
    let u1 = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    check!(
        u1,
        [
            SectorSpectrum {
                sector: U1Irrep::new(1),
                values: vec![2.0_f32, -2.0]
            },
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![1.0, -4.0, 0.0]
            },
        ],
        f32,
        vec![vec![-4.0, 0.0, 1.0], vec![-2.0, 2.0]],
        1e-5
    );
    check!(
        u1,
        [
            SectorSpectrum {
                sector: U1Irrep::new(1),
                values: vec![
                    num_complex::Complex32::new(2.0, 0.0),
                    num_complex::Complex32::new(-2.0, 0.0)
                ]
            },
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![
                    num_complex::Complex32::new(1.0, 0.0),
                    num_complex::Complex32::new(-4.0, 0.0),
                    num_complex::Complex32::new(0.0, 0.0)
                ]
            },
        ],
        num_complex::Complex32,
        vec![vec![-4.0, 0.0, 1.0], vec![-2.0, 2.0]],
        1e-5
    );
    let dual = u1.try_dual().unwrap();
    check!(
        dual,
        [
            SectorSpectrum {
                sector: U1Irrep::new(-1),
                values: vec![2.0_f64, -2.0]
            },
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![1.0, -4.0, 0.0]
            },
        ],
        f64,
        vec![vec![-2.0, 2.0], vec![-4.0, 0.0, 1.0]],
        1e-12
    );
    let spin0 = SU2Irrep::from_twice_spin(0);
    let spin_half = SU2Irrep::from_twice_spin(1);
    let su2 = GradedSpace::try_new(Arc::new(SU2FusionRule), [(spin0, 2), (spin_half, 1)]).unwrap();
    check!(
        su2,
        [
            SectorSpectrum {
                sector: spin_half,
                values: vec![Complex64::new(-3.0, 0.0)]
            },
            SectorSpectrum {
                sector: spin0,
                values: vec![Complex64::new(2.0, 0.0), Complex64::new(-2.0, 0.0)]
            },
        ],
        Complex64,
        vec![vec![-2.0, 2.0], vec![-3.0]],
        1e-12
    );
    let even = product_sector(U1Irrep::new(0), Z2Irrep::EVEN);
    let odd = product_sector(U1Irrep::new(1), Z2Irrep::ODD);
    let fermion = GradedSpace::try_new(
        Arc::new(U1FusionRule.product(FermionParityFusionRule)),
        [(even, 2), (odd, 1)],
    )
    .unwrap();
    check!(
        fermion,
        [
            SectorSpectrum {
                sector: odd,
                values: vec![-3.0_f64]
            },
            SectorSpectrum {
                sector: even,
                values: vec![2.0, -2.0]
            },
        ],
        f64,
        vec![vec![-2.0, 2.0], vec![-3.0]],
        1e-12
    );
    let zero_sector = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 0), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    check!(
        zero_sector,
        [SectorSpectrum {
            sector: U1Irrep::new(1),
            values: vec![2.0_f64, -3.0]
        }],
        f64,
        vec![vec![-3.0, 2.0]],
        1e-12
    );
    let empty = GradedSpace::try_new(Arc::new(U1FusionRule), []).unwrap();
    let input: TensorMap<_, f64> =
        TensorMap::diagonal(&runtime, &empty, Vec::<SectorSpectrum<_, f64>>::new()).unwrap();
    let densified = probe::record();
    assert!(input
        .eigh_vals(&[0], &[1], HermitianTol::DEFAULT)
        .unwrap()
        .is_empty());
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
}

#[test]
fn compact_diagonal_eigh_vals_applies_the_dense_hermiticity_check_directly() {
    let calls = Arc::new(SpyCounts::default());
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(eigh_vals_spy(&calls)))
        .build()
        .unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 1)]).unwrap();
    let near: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![Complex64::new(1.0, 1e-15)],
        }],
    )
    .unwrap();
    // Within the dense route's relative Hermiticity tolerance the eigenvalue
    // is the real part, read directly.
    let densified = probe::record();
    assert_eq!(
        near.eigh_vals(&[0], &[1], HermitianTol::DEFAULT).unwrap()[0].values,
        [1.0]
    );
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    assert_eq!(calls.total(), 0);

    let nonfinite: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![f64::NAN],
        }],
    )
    .unwrap();
    // NaN is refused by the shared finite-input stage.
    let densified = probe::record();
    let error = nonfinite
        .eigh_vals(&[0], &[1], HermitianTol::DEFAULT)
        .unwrap_err();
    assert!(is_finite_input_refusal(&error, "eigh"), "{error:?}");
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    assert_eq!(calls.total(), 0);
}

#[test]
fn compact_diagonal_eigh_vals_widens_stored_single_precision_values() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 3)]).unwrap();
    let values = [1.0000001_f32, -1.0000002_f32, 0.3_f32];
    let expected = vec![values[1] as f64, values[2] as f64, values[0] as f64];
    let real: TensorMap<_, f32> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: values.to_vec(),
        }],
    )
    .unwrap();
    let complex: TensorMap<_, num_complex::Complex32> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: values
                .map(|value| num_complex::Complex32::new(value, 0.0))
                .to_vec(),
        }],
    )
    .unwrap();
    let densified = probe::record();
    for actual in [
        real.eigh_vals(&[0], &[1], HermitianTol::DEFAULT),
        complex.eigh_vals(&[0], &[1], HermitianTol::DEFAULT),
    ] {
        assert_eq!(actual.unwrap()[0].values, expected);
    }
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
}

/// A finite diagonal entry whose eigenvalue overflows is the eigensolver's
/// numerical failure; a nonfinite entry is the finite-input precondition.
fn is_eig_value_error(error: &Error, finite_input: bool) -> bool {
    match error {
        Error::Operation(error) if finite_input => matches!(
            **error,
            OperationError::Dense(symtn_dense::DenseError::NumericalFailure { .. })
        ),
        Error::Operation(error) => matches!(**error, OperationError::InvalidArgument { .. }),
        _ => false,
    }
}
