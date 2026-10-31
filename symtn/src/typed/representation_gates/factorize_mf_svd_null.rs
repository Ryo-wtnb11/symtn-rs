use super::*;

#[test]
fn compact_diagonal_svd_full_preserves_spaces_without_materialization_or_solver() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let calls = Arc::new(SpyCounts::default());
    let _dense = probe::inject_dense(Box::new(polar_spy(&calls)));
    let runtime = Runtime::builder().build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(2), 3)])
        .unwrap()
        .try_dual()
        .unwrap();
    let input: TensorMap<_, f64> = TensorMap::diagonal(
        engine,
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(-2),
            values: vec![0.0, -4.0, 2.0],
        }],
    )
    .unwrap();
    let w = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(-2), 3)]).unwrap();
    let dense = input.materialize(engine).unwrap();
    let old = dense.svd_full(engine, &[0], &[1]).unwrap();
    assert!(calls.of(POLAR_SVD) > 0);
    calls.reset();
    let densified = probe::record();
    let out = input.svd_full(engine, &[0], &[1]).unwrap();
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    assert_eq!(calls.of(POLAR_SVD), 0);
    assert_eq!(out.u.codomain(), input.codomain());
    assert_eq!(out.vh.domain(), input.domain());
    assert_eq!(out.u.domain(), vec![w.clone()]);
    assert_eq!(out.s.codomain(), vec![w.clone()]);
    assert_eq!(out.s.domain(), vec![w.clone()]);
    assert_eq!(out.vh.codomain(), vec![w]);
    for (actual, expected) in [(&out.u, &old.u), (&out.s, &old.s), (&out.vh, &old.vh)] {
        assert_eq!(actual.codomain(), expected.codomain());
        assert_eq!(actual.domain(), expected.domain());
    }
    assert!(out.u.dense_data().is_ok());
    assert!(out.vh.dense_data().is_ok());
    assert!(out.s.dense_data().is_err());
    assert_eq!(
        out.s.diagview(engine).unwrap()[0].values,
        vec![4.0, 2.0, 0.0]
    );
    assert_eq!(
        out.s.diagonal_spectrum().unwrap().unwrap(),
        out.s.diagview(engine).unwrap()
    );
    let rebuilt = out
        .u
        .compose(engine, &out.s)
        .unwrap()
        .compose(engine, &out.vh)
        .unwrap();
    assert_eq!(rebuilt.dense_data().unwrap(), dense.dense_data().unwrap());
    let out_vh = &out.vh.adjoint(engine).unwrap();
    for (gram, identity) in [
        (
            out.u
                .adjoint(engine)
                .unwrap()
                .compose(engine, &out.u)
                .unwrap(),
            TensorMap::isomorphism(engine, &runtime, &out.u.domain(), &out.u.domain()).unwrap(),
        ),
        (
            out.vh.compose(engine, out_vh).unwrap(),
            TensorMap::isomorphism(engine, &runtime, &out.vh.codomain(), &out.vh.codomain())
                .unwrap(),
        ),
    ] {
        assert!(
            gram.axpby(engine, 1.0, &identity, -1.0)
                .unwrap()
                .norm(engine, 2.0)
                .unwrap()
                < 1e-12
        );
    }
    assert_eq!(
        out.s.materialize(engine).unwrap().dense_data().unwrap(),
        &[4.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0]
    );
}

#[test]
fn full_svd_compact_bond_requires_complete_exact_nondual_sector_legs() {
    let a = symtn_core::SectorId::new(1);
    let b = symtn_core::SectorId::new(2);
    let row = symtn_core::SectorLeg::new([(a, 2), (b, 3)], false);
    let exact = vec![
        symtn_matrixalgebra::SectorSpectrum {
            sector: a,
            values: vec![4.0, 1.0],
        },
        symtn_matrixalgebra::SectorSpectrum {
            sector: b,
            values: vec![3.0, 0.0, 0.0],
        },
    ];
    assert!(full_svd_spectrum_matches_bonds(&row, &row, &exact));
    let same_total = symtn_core::SectorLeg::new([(a, 3), (b, 2)], false);
    assert!(!full_svd_spectrum_matches_bonds(&row, &same_total, &exact));
    assert!(!full_svd_spectrum_matches_bonds(&row, &row, &exact[..1]));
    let dual = symtn_core::SectorLeg::new([(a, 2), (b, 3)], true);
    assert!(!full_svd_spectrum_matches_bonds(&dual, &dual, &exact));
    let duplicate = [exact[0].clone(), exact[0].clone()];
    assert!(!full_svd_spectrum_matches_bonds(&row, &row, &duplicate));
}

#[test]
fn dense_multisector_full_svd_publishes_compact_s_for_real_and_complex() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 3)],
    )
    .unwrap();
    let input: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, indices| {
            if indices[0] == indices[1] {
                (indices[0] + 2) as f64
            } else {
                0.25
            }
        })
        .unwrap();
    let Svd { u, s, vh } = input.svd_full(engine, &[0], &[1]).unwrap();
    assert!(s.dense_data().is_err());
    let values = s.diagview(engine).unwrap();
    assert_eq!(values.len(), 2);
    assert_eq!(values[0].sector, U1Irrep::new(0));
    assert_eq!(values[1].sector, U1Irrep::new(1));
    assert_eq!(values[0].values.len(), 2);
    assert_eq!(values[1].values.len(), 3);
    assert!(values
        .iter()
        .all(|entry| entry.values.windows(2).all(|pair| pair[0] >= pair[1])));
    assert_eq!(
        s.materialize(engine).unwrap().diagview(engine).unwrap(),
        s.diagview(engine).unwrap()
    );
    let rebuilt = u.compose(engine, &s).unwrap().compose(engine, &vh).unwrap();
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(input.dense_data().unwrap())
        .all(|(actual, expected)| (actual - expected).abs() < 1.0e-12));
    assert!(input
        .svd_full(engine, &[1], &[0])
        .unwrap()
        .s
        .dense_data()
        .is_err());

    let complex = input
        .convert::<Complex64>(engine)
        .unwrap()
        .scale(engine, Complex64::new(1.0, 0.5))
        .unwrap();
    let Svd { u, s, vh } = complex.svd_full(engine, &[0], &[1]).unwrap();
    assert!(s.dense_data().is_err());
    assert_eq!(s.diagview(engine).unwrap().len(), 2);
    let rebuilt = u.compose(engine, &s).unwrap().compose(engine, &vh).unwrap();
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(complex.dense_data().unwrap())
        .all(|(actual, expected)| (*actual - *expected).norm() < 1.0e-12));
}

/// The left and right null spaces of a square input are empty: the
/// codomain/domain is kept and the fresh one-leg bond has no sectors.
fn assert_empty_null_spaces<R, D>(
    engine: &mut dyn Engine,
    input: &TensorMap<R, D>,
    rows: &[usize],
    cols: &[usize],
) where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R> + TypedAdjointSpace<R> + TypedTensorTransformDispatch<R, D>,
    D: FactorizationScalar,
{
    let matrix = input.permute(engine, rows, cols).unwrap();
    let left = input.left_null(engine, rows, cols).unwrap();
    let right = input.right_null(engine, rows, cols).unwrap();
    assert_eq!(left.codomain(), matrix.codomain());
    assert_eq!(right.domain(), matrix.domain());
    for bond in [&left.domain()[0], &right.codomain()[0]] {
        assert!(!bond.is_dual());
        assert!(bond.sectors().unwrap().is_empty());
    }
    assert!(left.dense_data().unwrap().is_empty());
    assert!(right.dense_data().unwrap().is_empty());
}

#[test]
fn compact_diagonal_null_is_empty_by_shape_without_dense_work() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: a diagonal is square in every coupled sector, so its shape-based
    // null spaces are empty whatever its values (MatrixAlgebraKit
    // `qr_null!`/`lq_null!` with `DiagonalAlgorithm`), read with no
    // materialization and no dense solver, across scalars, U(1) with zero
    // and nonzero entries, SU(2), a fermionic product rule, and a dual leg.
    let calls = Arc::new(SpyCounts::default());
    let _dense = probe::inject_dense(Box::new(polar_spy(&calls)));
    let runtime = Runtime::builder().build().unwrap();
    let u1 = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    macro_rules! check_scalar {
        ($engine:ident, $scalar:ty) => {{
            let value = <$scalar as FactorScalar>::from_real;
            let input: TensorMap<_, $scalar> = TensorMap::diagonal(
                $engine,
                &runtime,
                &u1,
                [
                    SectorSpectrum {
                        sector: U1Irrep::new(0),
                        values: vec![value(3.0), value(0.0), value(-2.0)],
                    },
                    SectorSpectrum {
                        sector: U1Irrep::new(1),
                        values: vec![value(0.0), value(0.0)],
                    },
                ],
            )
            .unwrap();
            let densified = probe::record();
            assert_empty_null_spaces($engine, &input, &[0], &[1]);
            assert_empty_null_spaces($engine, &input, &[1], &[0]);
            assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
        }};
    }
    check_scalar!(engine, f32);
    check_scalar!(engine, f64);
    check_scalar!(engine, num_complex::Complex32);
    check_scalar!(engine, num_complex::Complex64);

    let spin0 = SU2Irrep::from_twice_spin(0);
    let half = SU2Irrep::from_twice_spin(1);
    let su2 = GradedSpace::try_new(Arc::new(SU2FusionRule), [(spin0, 3), (half, 2)]).unwrap();
    let input: TensorMap<_, f64> = TensorMap::diagonal(
        engine,
        &runtime,
        &su2,
        [
            SectorSpectrum {
                sector: spin0,
                values: vec![3.0, 0.0, -2.0],
            },
            SectorSpectrum {
                sector: half,
                values: vec![0.0, 4.0],
            },
        ],
    )
    .unwrap();
    let densified = probe::record();
    assert_empty_null_spaces(engine, &input, &[0], &[1]);
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);

    let provider = Arc::new(
        ProductFusionRule::<_, _>::try_new(U1FusionRule, FermionParityFusionRule).unwrap(),
    );
    let even = product_sector(U1Irrep::new(0), Z2Irrep::EVEN);
    let odd = product_sector(U1Irrep::new(1), Z2Irrep::ODD);
    let product = GradedSpace::try_new(Arc::clone(&provider), [(even, 2), (odd, 2)]).unwrap();
    let input: TensorMap<_, f64> = TensorMap::diagonal(
        engine,
        &runtime,
        &product,
        [
            SectorSpectrum {
                sector: even,
                values: vec![0.0, -2.0],
            },
            SectorSpectrum {
                sector: odd,
                values: vec![3.0, 0.0],
            },
        ],
    )
    .unwrap();
    let densified = probe::record();
    assert_empty_null_spaces(engine, &input, &[0], &[1]);
    assert_empty_null_spaces(engine, &input, &[1], &[0]);
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);

    let dual = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(1), 2)])
        .unwrap()
        .try_dual()
        .unwrap();
    let input: TensorMap<_, Complex64> = TensorMap::diagonal(
        engine,
        &runtime,
        &dual,
        [SectorSpectrum {
            sector: U1Irrep::new(-1),
            values: vec![Complex64::new(0.0, 0.0), Complex64::new(2.0, 3.0)],
        }],
    )
    .unwrap();
    let densified = probe::record();
    assert_empty_null_spaces(engine, &input, &[0], &[1]);
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    assert_eq!(calls.of(POLAR_SVD), 0);

    // The materialized lazy adjoint takes the dense route, which agrees.
    let lazy = input.materialize(engine).unwrap().adjoint(engine).unwrap();
    assert_empty_null_spaces(engine, &lazy, &[0], &[1]);
}

#[test]
fn compact_diagonal_null_refuses_nonfinite_input() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: an empty result does not skip the shared finite-input stage, so
    // dense and diagonal input refuse a NaN alike (#1986).
    let runtime = Runtime::builder().build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let nonfinite: TensorMap<_, f64> = TensorMap::diagonal(
        engine,
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![f64::NAN, 0.0],
        }],
    )
    .unwrap();
    let densified = probe::record();
    for error in [
        nonfinite.left_null(engine, &[0], &[1]).unwrap_err(),
        nonfinite.right_null(engine, &[0], &[1]).unwrap_err(),
    ] {
        assert!(is_finite_input_refusal(&error, "null"), "{error:?}");
    }
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
}

#[test]
fn compact_diagonal_svd_full_sorted_hand_oracle_all_scalars() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 4)]).unwrap();
    macro_rules! check {
        ($engine:ident, $ty:ty, $second:expr, $phase:expr, $tol:expr) => {{
            let input: TensorMap<_, $ty> = TensorMap::diagonal(
                $engine,
                &runtime,
                &leg,
                [SectorSpectrum {
                    sector: U1Irrep::new(0),
                    values: vec![
                        <$ty as FactorScalar>::from_real(0.0),
                        $second,
                        <$ty as FactorScalar>::from_real(2.0),
                        <$ty as FactorScalar>::from_real(-4.0),
                    ],
                }],
            )
            .unwrap();
            let out = input.svd_full($engine, &[0], &[1]).unwrap();
            assert_eq!(
                out.s.diagview($engine).unwrap()[0].values,
                [4.0, 2.0, 2.0, 0.0].map(<$ty as FactorScalar>::from_real)
            );
            // Hand order: magnitude 4, the two 2s in source order, then zero.
            let permutation = [3, 1, 2, 0];
            let phases = [
                Complex64::new(-1.0, 0.0),
                $phase,
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
            ];
            for col in 0..4 {
                for row in 0..4 {
                    let u = Complex64::new(if row == permutation[col] { 1.0 } else { 0.0 }, 0.0);
                    let vh = if col == permutation[row] {
                        phases[row]
                    } else {
                        Complex64::new(0.0, 0.0)
                    };
                    assert!(
                        (out.u.dense_data().unwrap()[row + 4 * col].widen_complex() - u).norm()
                            < $tol
                    );
                    assert!(
                        (out.vh.dense_data().unwrap()[row + 4 * col].widen_complex() - vh).norm()
                            < $tol
                    );
                }
            }
        }};
    }
    check!(engine, f32, -2.0, Complex64::new(-1.0, 0.0), 1e-6);
    check!(engine, f64, -2.0, Complex64::new(-1.0, 0.0), 1e-12);
    check!(
        engine,
        num_complex::Complex32,
        num_complex::Complex32::new(0.0, -2.0),
        Complex64::new(0.0, -1.0),
        1e-6
    );
    check!(
        engine,
        Complex64,
        Complex64::new(0.0, -2.0),
        Complex64::new(0.0, -1.0),
        1e-12
    );
}

#[test]
fn compact_diagonal_svd_full_subnormal_phase_and_nonfinite_refusal() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 1)]).unwrap();
    macro_rules! check {
        ($engine:ident, $ty:ty, $value:expr, $phase:expr, $tol:expr, $bad:expr) => {{
            let make = |$engine: &mut dyn Engine, value| {
                TensorMap::<_, $ty>::diagonal(
                    $engine,
                    &runtime,
                    &leg,
                    [SectorSpectrum {
                        sector: U1Irrep::new(0),
                        values: vec![value],
                    }],
                )
                .unwrap()
            };
            let input = make($engine, $value);
            let densified = probe::record();
            let out = input.svd_full($engine, &[0], &[1]).unwrap();
            assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
            assert!(
                out.s.diagview($engine).unwrap()[0].values[0]
                    .widen_complex()
                    .re
                    > 0.0
            );
            assert!(
                (out.u.dense_data().unwrap()[0].widen_complex() - Complex64::new(1.0, 0.0)).norm()
                    < $tol
            );
            assert!((out.vh.dense_data().unwrap()[0].widen_complex() - $phase).norm() < $tol);
            assert_eq!(
                out.u
                    .compose($engine, &out.s)
                    .unwrap()
                    .compose($engine, &out.vh)
                    .unwrap()
                    .dense_data()
                    .unwrap(),
                &[$value]
            );
            // A nonfinite value is refused by the shared finite-input stage;
            // a finite overflowing one is MAK `svd_full!(::DiagonalAlgorithm)`,
            // `S = abs(a) = Inf`. No materialization either way.
            for value in $bad {
                let input = make($engine, value);
                let densified = probe::record();
                let result = input.svd_full($engine, &[0], &[1]);
                assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
                let widened = value.widen_complex();
                if widened.re.is_finite() && widened.im.is_finite() {
                    let out = result.unwrap();
                    let singular = out.s.diagview($engine).unwrap()[0].values[0]
                        .widen_complex()
                        .re;
                    assert_eq!(singular, f64::INFINITY);
                } else {
                    let error = result.map(drop).unwrap_err();
                    assert!(is_finite_input_refusal(&error, "svd"), "{error:?}");
                }
            }
        }};
    }
    check!(
        engine,
        f32,
        -f32::from_bits(1),
        Complex64::new(-1.0, 0.0),
        1e-6,
        [f32::NAN, f32::INFINITY]
    );
    check!(
        engine,
        f64,
        -f64::from_bits(1),
        Complex64::new(-1.0, 0.0),
        1e-12,
        [f64::NAN, f64::INFINITY]
    );
    let phase = Complex64::new(0.5_f64.sqrt(), 0.5_f64.sqrt());
    check!(
        engine,
        num_complex::Complex32,
        num_complex::Complex32::new(f32::from_bits(1), f32::from_bits(1)),
        phase,
        1e-6,
        [
            num_complex::Complex32::new(f32::NAN, 0.0),
            num_complex::Complex32::new(f32::INFINITY, 0.0),
            num_complex::Complex32::new(f32::MAX, f32::MAX)
        ]
    );
    check!(
        engine,
        Complex64,
        Complex64::new(f64::from_bits(1), f64::from_bits(1)),
        phase,
        1e-12,
        [
            Complex64::new(f64::NAN, 0.0),
            Complex64::new(f64::INFINITY, 0.0),
            Complex64::new(f64::MAX, f64::MAX)
        ]
    );
}

#[test]
fn compact_diagonal_svd_uses_spectrum_without_dense_input() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    for full in [false, true] {
        let runtime = Runtime::builder().build().unwrap();
        let leg = GradedSpace::try_new(
            Arc::new(U1FusionRule),
            [(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)],
        )
        .unwrap();
        let source: TensorMap<_, f64> = TensorMap::diagonal(
            engine,
            &runtime,
            &leg,
            [
                SectorSpectrum {
                    sector: U1Irrep::new(1),
                    values: vec![3.0, -2.0],
                },
                SectorSpectrum {
                    sector: U1Irrep::new(0),
                    values: vec![1.0, -4.0, 0.0],
                },
            ],
        )
        .unwrap();
        let input = source
            .materialize(engine)
            .unwrap()
            .dense_data()
            .unwrap()
            .to_vec();
        let densified = probe::record();
        let Svd { u, s, vh } = (if full {
            source.svd_full(engine, &[0], &[1])
        } else {
            source.svd_compact(engine, &[0], &[1])
        })
        .unwrap();
        assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
        assert_eq!(
            s.diagview(engine)
                .unwrap()
                .iter()
                .map(|e| e.values.clone())
                .collect::<Vec<_>>(),
            vec![vec![4.0, 1.0, 0.0], vec![3.0, 2.0]]
        );
        let rebuilt = u.compose(engine, &s).unwrap().compose(engine, &vh).unwrap();
        assert!(rebuilt
            .dense_data()
            .unwrap()
            .iter()
            .zip(input)
            .all(|(a, b)| (a - b).abs() < 1e-12));
        assert_eq!(
            source.diagview(engine).unwrap()[0].values,
            vec![1.0, -4.0, 0.0]
        );
    }
}

#[test]
fn compact_diagonal_svd_vals_uses_only_the_stored_spectrum() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let solver_calls = Arc::new(SpyCounts::default());
    let _dense = probe::inject_dense(Box::new(fail_second_svd(&solver_calls)));
    let runtime = Runtime::builder().build().unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    let input: TensorMap<_, f64> = TensorMap::diagonal(
        engine,
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: U1Irrep::new(1),
                values: vec![-2.0, 3.0],
            },
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![-4.0, 0.0, 1.0],
            },
        ],
    )
    .unwrap();
    let before = input.diagview(engine).unwrap();
    let densified = probe::record();
    assert_eq!(
        input.svd_vals(engine, &[0], &[1]).unwrap(),
        vec![
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![4.0, 1.0, 0.0]
            },
            SectorSpectrum {
                sector: U1Irrep::new(1),
                values: vec![3.0, 2.0]
            },
        ]
    );
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    assert_eq!(input.diagview(engine).unwrap(), before);

    let dual_bond = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(1), 2)])
        .unwrap()
        .try_dual()
        .unwrap();
    let dual: TensorMap<_, f64> = TensorMap::diagonal(
        engine,
        &runtime,
        &dual_bond,
        [SectorSpectrum {
            sector: U1Irrep::new(-1),
            values: vec![-2.0, 1.0],
        }],
    )
    .unwrap();
    let densified = probe::record();
    assert_eq!(
        dual.svd_vals(engine, &[0], &[1]).unwrap()[0].values,
        [2.0, 1.0]
    );
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    assert_eq!(solver_calls.total(), 0);
}

#[test]
fn compact_diagonal_svd_vals_rounds_at_payload_precision() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let real: TensorMap<_, f32> = TensorMap::diagonal(
        engine,
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![-1.000_000_1, 0.0],
        }],
    )
    .unwrap();
    assert_eq!(
        real.svd_vals(engine, &[0], &[1]).unwrap()[0].values,
        [1.000_000_1_f32 as f64, 0.0]
    );

    let complex: TensorMap<_, num_complex::Complex32> = TensorMap::diagonal(
        engine,
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![
                num_complex::Complex32::new(1.0, 1.0),
                num_complex::Complex32::new(-3.0, 4.0),
            ],
        }],
    )
    .unwrap();
    assert_eq!(
        complex.svd_vals(engine, &[0], &[1]).unwrap()[0].values,
        [5.0, (2.0_f64.sqrt() as f32) as f64]
    );
}

#[test]
fn compact_diagonal_svd_vals_refuses_nonfinite_and_overflows_at_range_edges() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 1)]).unwrap();
    for value in [
        num_complex::Complex32::new(f32::NAN, 0.0),
        num_complex::Complex32::new(f32::INFINITY, 0.0),
        num_complex::Complex32::new(f32::MAX * 0.75, f32::MAX * 0.75),
    ] {
        let input: TensorMap<_, num_complex::Complex32> = TensorMap::diagonal(
            engine,
            &runtime,
            &leg,
            [SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![value],
            }],
        )
        .unwrap();
        // A nonfinite value is refused by the shared finite-input stage; a
        // finite one is MAK `svd_vals!(::DiagonalAlgorithm)`, `abs(a)`:
        // `|0.75 MAX (1 + i)| = 1.06 MAX` overflows f32 to Inf.
        let densified = probe::record();
        let result = input.svd_vals(engine, &[0], &[1]);
        assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
        if value.re.is_finite() {
            assert_eq!(result.unwrap()[0].values[0], f64::INFINITY);
        } else {
            let error = result.unwrap_err();
            assert!(is_finite_input_refusal(&error, "svd"), "{error:?}");
        }
    }
}

#[test]
fn compact_diagonal_svd_complex_permutation_and_phase() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    for full in [false, true] {
        let runtime = Runtime::builder().build().unwrap();
        let leg = GradedSpace::try_new(
            Arc::new(U1FusionRule),
            [(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)],
        )
        .unwrap();
        let z = Complex64::new;
        let input: TensorMap<_, Complex64> = TensorMap::diagonal(
            engine,
            &runtime,
            &leg,
            [
                SectorSpectrum {
                    sector: U1Irrep::new(1),
                    values: vec![z(-2.0, 0.0), z(4.0, 0.0)],
                },
                SectorSpectrum {
                    sector: U1Irrep::new(0),
                    values: vec![z(0.0, 0.0), z(0.0, -3.0), z(1.0, 1.0)],
                },
            ],
        )
        .unwrap();
        let densified = probe::record();
        let Svd { u, s, vh } = (if full {
            input.svd_full(engine, &[0], &[1])
        } else {
            input.svd_compact(engine, &[0], &[1])
        })
        .unwrap();
        assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
        let root2 = 2.0_f64.sqrt();
        assert_eq!(u.dense_data().unwrap().len(), 13);
        assert_eq!(vh.dense_data().unwrap().len(), 13);
        for (actual, expected) in u.dense_data().unwrap().iter().zip([
            z(0.0, 0.0),
            z(1.0, 0.0),
            z(0.0, 0.0),
            z(0.0, 0.0),
            z(0.0, 0.0),
            z(1.0, 0.0),
            z(1.0, 0.0),
            z(0.0, 0.0),
            z(0.0, 0.0),
            z(0.0, 0.0),
            z(1.0, 0.0),
            z(1.0, 0.0),
            z(0.0, 0.0),
        ]) {
            assert!((*actual - expected).norm() < 1e-12);
        }
        for (actual, expected) in vh.dense_data().unwrap().iter().zip([
            z(0.0, 0.0),
            z(0.0, 0.0),
            z(1.0, 0.0),
            z(0.0, -1.0),
            z(0.0, 0.0),
            z(0.0, 0.0),
            z(0.0, 0.0),
            z(1.0 / root2, 1.0 / root2),
            z(0.0, 0.0),
            z(0.0, 0.0),
            z(-1.0, 0.0),
            z(1.0, 0.0),
            z(0.0, 0.0),
        ]) {
            assert!((*actual - expected).norm() < 1e-12);
        }
        assert_eq!(
            s.diagview(engine)
                .unwrap()
                .iter()
                .map(|e| e.values.clone())
                .collect::<Vec<_>>(),
            vec![
                vec![z(3.0, 0.0), z(root2, 0.0), z(0.0, 0.0)],
                vec![z(4.0, 0.0), z(2.0, 0.0)]
            ]
        );
        let rebuilt = u.compose(engine, &s).unwrap().compose(engine, &vh).unwrap();
        assert!(rebuilt
            .dense_data()
            .unwrap()
            .iter()
            .zip(input.materialize(engine).unwrap().dense_data().unwrap())
            .all(|(a, b)| (*a - *b).norm() < 1e-12));
    }
}

#[test]
fn compact_diagonal_svd_other_host_scalars_and_sectors() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    for full in [false, true] {
        let runtime = Runtime::builder().build().unwrap();
        macro_rules! check {
            ($engine:ident, $rule:expr, $legs:expr, $spectra:expr, $dtype:ty, $tol:expr) => {{
                let leg = GradedSpace::try_new(Arc::new($rule), $legs).unwrap();
                let input: TensorMap<_, $dtype> =
                    TensorMap::diagonal($engine, &runtime, &leg, $spectra).unwrap();
                let before = input
                    .materialize($engine)
                    .unwrap()
                    .dense_data()
                    .unwrap()
                    .to_vec();
                let saved = input.diagview($engine).unwrap();
                let dense_values = input
                    .materialize($engine)
                    .unwrap()
                    .svd_vals($engine, &[0], &[1])
                    .unwrap();
                let densified = probe::record();
                let compact_values = input.svd_vals($engine, &[0], &[1]).unwrap();
                assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
                assert_eq!(compact_values.len(), dense_values.len());
                for (actual, expected) in compact_values.iter().zip(&dense_values) {
                    assert_eq!(actual.sector, expected.sector);
                    assert_eq!(actual.values.len(), expected.values.len());
                    for (&a, &b) in actual.values.iter().zip(&expected.values) {
                        assert!((a - b).abs() <= $tol * b.max(1.0));
                    }
                }
                let densified = probe::record();
                let Svd { u, s, vh } = (if full {
                    input.svd_full($engine, &[0], &[1])
                } else {
                    input.svd_compact($engine, &[0], &[1])
                })
                .unwrap();
                assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
                let rebuilt = u
                    .compose($engine, &s)
                    .unwrap()
                    .compose($engine, &vh)
                    .unwrap();
                assert!(rebuilt
                    .dense_data()
                    .unwrap()
                    .iter()
                    .zip(before)
                    .all(|(a, b)| FactorScalar::widen_complex(*a - b).norm() <= $tol));
                assert_eq!(input.diagview($engine).unwrap(), saved);
            }};
        }
        check!(
            engine,
            U1FusionRule,
            [(U1Irrep::new(0), 3)],
            [SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![-0.0_f32, -3.0, 1.0]
            }],
            f32,
            1e-5
        );
        check!(
            engine,
            U1FusionRule,
            [(U1Irrep::new(0), 2)],
            [SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![
                    num_complex::Complex32::new(1.0, 2.0),
                    num_complex::Complex32::new(-3.0, 1.0)
                ]
            }],
            num_complex::Complex32,
            1e-5
        );
        let spin0 = SU2Irrep::from_twice_spin(0);
        let spin_half = SU2Irrep::from_twice_spin(1);
        check!(
            engine,
            SU2FusionRule,
            [(spin0, 2), (spin_half, 1)],
            [
                SectorSpectrum {
                    sector: spin_half,
                    values: vec![-4.0_f64]
                },
                SectorSpectrum {
                    sector: spin0,
                    values: vec![2.0, -3.0]
                },
            ],
            f64,
            1e-12
        );
        let even = product_sector(U1Irrep::new(0), Z2Irrep::EVEN);
        let odd = product_sector(U1Irrep::new(1), Z2Irrep::ODD);
        check!(
            engine,
            ProductFusionRule::<_, _>::try_new(U1FusionRule, FermionParityFusionRule).unwrap(),
            [(even, 2), (odd, 1)],
            [
                SectorSpectrum {
                    sector: odd,
                    values: vec![-4.0_f64]
                },
                SectorSpectrum {
                    sector: even,
                    values: vec![2.0, -3.0]
                },
            ],
            f64,
            1e-12
        );
        check!(
            engine,
            U1FusionRule,
            [(U1Irrep::new(0), 0), (U1Irrep::new(1), 2)],
            [SectorSpectrum {
                sector: U1Irrep::new(1),
                values: vec![2.0_f64, -3.0]
            }],
            f64,
            1e-12
        );
        let empty = GradedSpace::try_new(Arc::new(U1FusionRule), []).unwrap();
        let input: TensorMap<_, f64> = TensorMap::diagonal(
            engine,
            &runtime,
            &empty,
            Vec::<SectorSpectrum<_, f64>>::new(),
        )
        .unwrap();
        let densified = probe::record();
        assert!(input.svd_vals(engine, &[0], &[1]).unwrap().is_empty());
        assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
        let Svd { u, s, vh } = (if full {
            input.svd_full(engine, &[0], &[1])
        } else {
            input.svd_compact(engine, &[0], &[1])
        })
        .unwrap();
        assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
        assert_eq!(
            u.dense_data().unwrap().len() + vh.dense_data().unwrap().len(),
            0
        );
        assert!(s.diagview(engine).unwrap().is_empty());
    }
}

#[test]
fn compact_diagonal_svd_at_c32_range_edge_overflows_like_mak() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    for full in [false, true] {
        let runtime = Runtime::builder().build().unwrap();
        let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 1)]).unwrap();
        let large = f32::MAX * 0.75;
        let input: TensorMap<_, num_complex::Complex32> = TensorMap::diagonal(
            engine,
            &runtime,
            &leg,
            [SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![num_complex::Complex32::new(large, large)],
            }],
        )
        .unwrap();
        // MAK `svd_full!(::DiagonalAlgorithm)`: `S = abs(a)` overflows f32
        // to Inf (`1.06 MAX`), and `Vh` keeps the finite unit phase.
        let Svd { s, vh, .. } = if full {
            input.svd_full(engine, &[0], &[1])
        } else {
            input.svd_compact(engine, &[0], &[1])
        }
        .unwrap();
        assert_eq!(
            s.diagview(engine).unwrap()[0].values,
            [num_complex::Complex32::new(f32::INFINITY, 0.0)]
        );
        let phase = vh.dense_data().unwrap()[0];
        let half = std::f32::consts::FRAC_1_SQRT_2;
        assert!((phase - num_complex::Complex32::new(half, half)).norm() <= 4.0 * f32::EPSILON);
    }
}

#[test]
fn compact_diagonal_svd_subnormal_complex_phase_is_unit() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    for full in [false, true] {
        let runtime = Runtime::builder().build().unwrap();
        let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 1)]).unwrap();
        let tiny = f64::from_bits(1);
        let input: TensorMap<_, Complex64> = TensorMap::diagonal(
            engine,
            &runtime,
            &leg,
            [SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![Complex64::new(tiny, tiny)],
            }],
        )
        .unwrap();
        let Svd { u, s, vh } = (if full {
            input.svd_full(engine, &[0], &[1])
        } else {
            input.svd_compact(engine, &[0], &[1])
        })
        .unwrap();
        assert!((vh.dense_data().unwrap()[0].norm() - 1.0).abs() < 1e-12);
        assert!((u.dense_data().unwrap()[0].norm() - 1.0).abs() < 1e-12);
        let rebuilt = u.compose(engine, &s).unwrap().compose(engine, &vh).unwrap();
        assert_eq!(rebuilt.dense_data().unwrap()[0], Complex64::new(tiny, tiny));
    }
}
