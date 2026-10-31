use super::*;

#[test]
fn checked_generic_pinv_rectangular_moore_penrose_and_validation_precedence() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    macro_rules! assert_moore_penrose {
        ($engine:ident, $input:expr, $pseudo:expr, $distance:expr) => {{
            let input = $input;
            let pseudo = $pseudo;
            let distance = $distance;
            let aa_plus = input.compose($engine, pseudo).unwrap();
            let a_plus_a = pseudo.compose($engine, input).unwrap();
            for (actual, expected) in aa_plus
                .compose($engine, input)
                .unwrap()
                .dense_data()
                .unwrap()
                .iter()
                .zip(input.dense_data().unwrap())
            {
                assert!(distance(*actual, *expected) < 1e-10);
            }
            for (actual, expected) in a_plus_a
                .compose($engine, pseudo)
                .unwrap()
                .dense_data()
                .unwrap()
                .iter()
                .zip(pseudo.dense_data().unwrap())
            {
                assert!(distance(*actual, *expected) < 1e-10);
            }
            for (actual, expected) in aa_plus
                .adjoint($engine)
                .unwrap()
                .materialize($engine)
                .unwrap()
                .dense_data()
                .unwrap()
                .iter()
                .zip(aa_plus.dense_data().unwrap())
            {
                assert!(distance(*actual, *expected) < 1e-10);
            }
            for (actual, expected) in a_plus_a
                .adjoint($engine)
                .unwrap()
                .materialize($engine)
                .unwrap()
                .dense_data()
                .unwrap()
                .iter()
                .zip(a_plus_a.dense_data().unwrap())
            {
                assert!(distance(*actual, *expected) < 1e-10);
            }
        }};
    }

    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let codomain = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let domain = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 3)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&codomain], [&domain], |_, index| {
            [[1.0, 0.0, 1.0], [0.0, 2.0, 1.0]][index[0]][index[1]]
        })
        .unwrap();
    let before = source.dense_data().unwrap().to_vec();
    reset_provider_queries(&provider);
    for rcond in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            source.pinv(engine, &[0], &[1], rcond),
            Err(GenericTensorError::Facade(
                symtn::typed::Error::InvalidArgument(_)
            ))
        ));
        assert_no_provider_queries(&provider);
    }
    assert_eq!(source.dense_data().unwrap(), before.as_slice());

    let pseudo = source.pinv(engine, &[0], &[1], 1e-12).unwrap();
    assert_eq!(pseudo.codomain(), source.domain());
    assert_eq!(pseudo.domain(), source.codomain());
    assert!(std::ptr::eq(pseudo.provider(), provider.as_ref()));
    assert_moore_penrose!(engine, &source, &pseudo, |actual: f64, expected: f64| {
        (actual - expected).abs()
    });

    let complex = source
        .convert::<Complex64>(engine)
        .unwrap()
        .scale(engine, Complex64::new(1.0, 0.25))
        .unwrap();
    let complex_pseudo = complex.pinv(engine, &[0], &[1], 1e-12).unwrap();
    assert_moore_penrose!(
        engine,
        &complex,
        &complex_pseudo,
        |actual: Complex64, expected: Complex64| (actual - expected).norm()
    );

    let tall: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&domain], [&codomain], |_, index| {
            [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]][index[0]][index[1]]
        })
        .unwrap();
    let tall_pseudo = tall.pinv(engine, &[0], &[1], 1e-12).unwrap();
    assert_moore_penrose!(engine, &tall, &tall_pseudo, |actual: f64, expected: f64| {
        (actual - expected).abs()
    });
    let complex_tall = tall
        .convert::<Complex64>(engine)
        .unwrap()
        .scale(engine, Complex64::new(1.0, 0.25))
        .unwrap();
    let complex_tall_pseudo = complex_tall.pinv(engine, &[0], &[1], 1e-12).unwrap();
    assert_moore_penrose!(
        engine,
        &complex_tall,
        &complex_tall_pseudo,
        |actual: Complex64, expected: Complex64| { (actual - expected).norm() }
    );

    let lazy = source.adjoint(engine).unwrap();
    let lazy_pseudo = lazy.pinv(engine, &[0], &[1], 1e-12).unwrap();
    for (actual, expected) in lazy_pseudo.dense_data().unwrap().iter().zip(
        pseudo
            .adjoint(engine)
            .unwrap()
            .materialize(engine)
            .unwrap()
            .dense_data()
            .unwrap(),
    ) {
        assert!((actual - expected).abs() < 1e-10);
    }
}

#[cfg(feature = "cgc-gen")]
macro_rules! assert_sun_polar_laws {
    ($engine:ident, $runtime:expr, $source:expr, $codomain:expr, $domain:expr, $left:expr) => {{
        let source = $source;
        let left: bool = $left;
        let (w, p) = if left {
            let LeftPolar { w, p } = source
                .left_polar($engine, &codomain_axes(&source), &domain_axes(&source))
                .unwrap();
            (w, p)
        } else {
            let RightPolar { p, wh } = source
                .right_polar($engine, &codomain_axes(&source), &domain_axes(&source))
                .unwrap();
            (wh, p)
        };
        let assert_close = |$engine: &mut dyn Engine,
                            actual: &TensorMap<_, _>,
                            expected: &TensorMap<_, _>,
                            what: &str| {
            let error = actual
                .axpby($engine, 1.0.into(), expected, (-1.0).into())
                .unwrap()
                .norm($engine, 2.0)
                .unwrap();
            assert!(
                error < 1e-9 * (1.0 + expected.norm($engine, 2.0).unwrap()),
                "{what}: {error}"
            );
        };
        // Checked Generic contraction reads owned tensors only; the lazy
        // adjoint's logical data is replayed into an owned map.
        let owned_adjoint = |$engine: &mut dyn Engine, tensor: &TensorMap<_, _>| {
            tensor
                .adjoint($engine)
                .unwrap()
                .materialize($engine)
                .unwrap()
        };
        // A rank-0 map names no leg to rebuild its adjoint from; checked eigh
        // admission is the Hermitian (real scalar) test there.
        let assert_hermitian = |$engine: &mut dyn Engine, tensor: &TensorMap<_, _>, what: &str| {
            if tensor.rank() == 0 {
                assert!(
                    tensor
                        .eigh_vals($engine, &[], &[], HermitianTol::DEFAULT)
                        .is_ok(),
                    "{what}"
                );
            } else {
                let operand = &owned_adjoint($engine, tensor);
                assert_close($engine, operand, tensor, what);
            }
        };
        let rebuilt = if left {
            w.compose($engine, &p)
        } else {
            p.compose($engine, &w)
        }
        .unwrap();
        assert_close($engine, &rebuilt, source, "A = WP / PW");
        let wh = owned_adjoint($engine, &w);
        let (gram, gram_space, square_oracle) = if left {
            (
                wh.compose($engine, &w).unwrap(),
                $domain,
                owned_adjoint($engine, source)
                    .compose($engine, source)
                    .unwrap(),
            )
        } else {
            let operand = &owned_adjoint($engine, source);
            (
                w.compose($engine, &wh).unwrap(),
                $codomain,
                source.compose($engine, operand).unwrap(),
            )
        };
        let spaces: Vec<&GradedSpace<_>> = gram_space.collect();
        assert_hermitian($engine, &p, "P Hermitian");
        let p_compose = &p.compose($engine, &p).unwrap();
        assert_close($engine, p_compose, &square_oracle, "P^2 = A^H A / A A^H");
        if spaces.is_empty() {
            // A rank-0 Gram map sum |w|^2 is real and nonnegative, so unit
            // norm is the scalar identity (no leg names the provider).
            assert_eq!(gram.dense_data().unwrap().len(), 1);
            assert!(
                (gram.norm($engine, 2.0).unwrap() - 1.0).abs() < 1e-9,
                "W isometry"
            );
        } else {
            let identity = TensorMap::from_subblock_fn(
                $engine,
                $runtime,
                spaces.iter().copied(),
                spaces.iter().copied(),
                |trees, ij| {
                    let same_tree = trees.codomain_uncoupled() == trees.domain_uncoupled()
                        && trees.codomain_innerlines() == trees.domain_innerlines()
                        && trees.codomain_vertices() == trees.domain_vertices();
                    let (row, col) = ij.split_at(spaces.len());
                    if same_tree && row == col { 1.0 } else { 0.0 }.into()
                },
            )
            .unwrap();
            assert_close($engine, &gram, &identity, "W isometry");
        }
        assert!(p
            .eigh_vals(
                $engine,
                &codomain_axes(&p),
                &domain_axes(&p),
                HermitianTol::DEFAULT
            )
            .unwrap()
            .iter()
            .flat_map(|entry| &entry.values)
            .all(|&value| value >= -1e-10));

        // pinv reads the same coupled-sector regions: Moore-Penrose laws.
        let pseudo = source
            .pinv(
                $engine,
                &codomain_axes(&source),
                &domain_axes(&source),
                1e-12,
            )
            .unwrap();
        let source_compose = &source
            .compose($engine, &pseudo)
            .unwrap()
            .compose($engine, source)
            .unwrap();
        assert_close($engine, source_compose, source, "A A+ A = A");
        let pseudo_compose = &pseudo
            .compose($engine, source)
            .unwrap()
            .compose($engine, &pseudo)
            .unwrap();
        assert_close($engine, pseudo_compose, &pseudo, "A+ A A+ = A+");
        for projector in [
            source.compose($engine, &pseudo).unwrap(),
            pseudo.compose($engine, source).unwrap(),
        ] {
            assert_hermitian($engine, &projector, "pinv projector Hermitian");
        }
    }};
}

#[cfg(feature = "cgc-gen")]
#[test]
fn sun_checked_generic_polar_and_pinv_accept_facade_layouts_of_rank_three_and_four() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: polar and pinv are defined for every TensorMap (TensorKit
    // left_polar / right_polar / pinv); multi-leg SU(3) facade layouts with
    // outer multiplicity satisfy A = WP / PW, W isometric, and P Hermitian
    // PSD with P^2 = A^H A / A A^H (which fixes P uniquely), and the
    // Moore-Penrose laws for pinv.
    use symtn::sector::LieGroupFusionRule;

    let runtime = Runtime::builder().build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 2), (vec![0, 0], 1)]).unwrap();
    let value = |salt: u64| ((salt.wrapping_mul(2_654_435_761) % 1009) as f64) / 1009.0 - 0.5;
    // Non-self-dual legs: T = 3 (x2) + 3bar and its dual T*, in mixed-dual
    // and all-dual rank-3 shapes. An all-dual map with one codomain and two
    // domain legs (or the reverse) violates the polar direction in some
    // sector, so the all-dual shapes put all three legs on one side.
    let t =
        GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 0], 2), (vec![0, 1], 1)]).unwrap();
    let t_dual = t.try_dual().unwrap();
    let cases: [(&[&GradedSpace<_>], &[&GradedSpace<_>], bool); 8] = [
        (&[&leg, &leg], &[&leg], true),
        (&[&leg], &[&leg, &leg], false),
        (&[&leg, &leg], &[&leg, &leg], true),
        (&[&leg, &leg], &[&leg, &leg], false),
        (&[&t, &t_dual], &[&t], true),
        (&[&t], &[&t, &t_dual], false),
        (&[&t_dual, &t_dual, &t_dual], &[], true),
        (&[], &[&t_dual, &t_dual, &t_dual], false),
    ];
    for (codomain, domain, left) in cases {
        let mut salt = 0;
        let real: TensorMap<_, f64> = TensorMap::from_subblock_fn(
            engine,
            &runtime,
            codomain.iter().copied(),
            domain.iter().copied(),
            |_, _| {
                salt += 1;
                value(salt)
            },
        )
        .unwrap();
        assert_sun_polar_laws!(
            engine,
            &runtime,
            &real,
            codomain.iter().copied(),
            domain.iter().copied(),
            left
        );
        let complex: TensorMap<_, Complex64> = TensorMap::from_subblock_fn(
            engine,
            &runtime,
            codomain.iter().copied(),
            domain.iter().copied(),
            |_, _| {
                salt += 2;
                Complex64::new(value(salt), value(salt + 1))
            },
        )
        .unwrap();
        assert_sun_polar_laws!(
            engine,
            &runtime,
            &complex,
            codomain.iter().copied(),
            domain.iter().copied(),
            left
        );
    }
}

#[test]
fn checked_generic_pinv_of_a_nan_tensor_is_invalid_argument() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: the staged checked-Generic pinv never publishes a finite answer
    // for a NaN payload; the finite-input stage answers `InvalidArgument`
    // before any provider SVD, so the kind does not depend on the provider.
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let bond =
        GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 1), (Label::X, 1)]).unwrap();
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let source: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&bond], [&bond], |trees, _| {
                if trees.coupled() == &Label::Vacuum {
                    4.0
                } else {
                    bad
                }
            })
            .unwrap();
        for (case, result) in [
            ("f64", source.pinv(engine, &[0], &[1], 0.5).map(|_| ())),
            (
                "c64",
                source
                    .convert::<Complex64>(engine)
                    .unwrap()
                    .pinv(engine, &[0], &[1], 0.5)
                    .map(|_| ()),
            ),
        ] {
            match result {
                Err(GenericTensorError::Facade(symtn::typed::Error::Operation(error))) => {
                    assert_eq!(
                        *error,
                        symtn::typed::OperationError::InvalidArgument {
                            message: "pinv singular values must be finite"
                        },
                        "{case}"
                    )
                }
                other => panic!("{case}: expected the finite-input refusal, got {other:?}"),
            }
        }
        // rcond validation precedes the finite scan.
        match source.pinv(engine, &[0], &[1], -1.0) {
            Err(GenericTensorError::Facade(symtn::typed::Error::InvalidArgument(message))) => {
                assert!(message.contains("rcond"), "{message}")
            }
            other => panic!("expected the rcond error, got {:?}", other.map(|_| ())),
        }
    }
}

#[test]
fn checked_generic_pinv_uses_a_strict_global_cutoff() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let bond =
        GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 1), (Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&bond], [&bond], |trees, _| {
            if trees.coupled() == &Label::Vacuum {
                4.0
            } else {
                2.0
            }
        })
        .unwrap();
    let pseudo = source.pinv(engine, &[0], &[1], 0.5).unwrap();
    // The cutoff drops the X block exactly; the kept value is 1/4.
    assert_eq!(pseudo.dense_data().unwrap()[1], 0.0);
    numerics::assert_close("pinv kept value", pseudo.dense_data().unwrap()[0], 0.25, 1);
}

/// A nonfinite compact entry is refused by the dense cutoff's typed
/// error, as in multiplicity-free mode, not sent to the dense SVD.
#[test]
fn checked_compact_diagonal_pinv_refuses_nonfinite_like_multiplicity_free() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let bond = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    for value in [f64::NAN, f64::INFINITY] {
        let input: TensorMap<_, f64> = TensorMap::diagonal(
            engine,
            &runtime,
            &bond,
            [SectorSpectrum {
                sector: Label::X,
                values: vec![value, 0.0],
            }],
        )
        .unwrap();
        assert!(matches!(
            input.pinv(engine, &[0], &[1], 0.5),
            Err(GenericTensorError::Facade(symtn::typed::Error::Operation(error)))
                if *error == symtn::typed::OperationError::InvalidArgument {
                    message: "pinv singular values must be finite"
                }
        ));
    }
}

#[test]
fn checked_compact_diagonal_pinv_precision_limits_follow_multiplicity_free_semantics() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let bond = GradedSpace::try_new(provider, [(Label::Vacuum, 2), (Label::X, 2)]).unwrap();

    let real: TensorMap<_, f32> = TensorMap::diagonal(
        engine,
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: Label::Vacuum,
                values: vec![8.0, 2.0],
            },
            SectorSpectrum {
                sector: Label::X,
                values: vec![-4.0, 0.0],
            },
        ],
    )
    .unwrap();
    let real_dense = real
        .materialize(engine)
        .unwrap()
        .pinv(engine, &[0], &[1], 0.25)
        .unwrap();
    let real_direct = real.pinv(engine, &[0], &[1], 0.25).unwrap();
    assert!(real_direct.dense_data().is_err());
    assert_eq!(
        real_direct.diagview(engine).unwrap()[0].values,
        [0.125, 0.0]
    );
    for (&actual, &expected) in real_direct
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(real_dense.dense_data().unwrap())
    {
        assert!((actual - expected).abs() <= 1e-6);
    }

    let complex: TensorMap<_, Complex32> = TensorMap::diagonal(
        engine,
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: Label::Vacuum,
                values: vec![Complex32::new(8.0, 0.0), Complex32::new(2.0, 0.0)],
            },
            SectorSpectrum {
                sector: Label::X,
                values: vec![Complex32::new(3.0, 4.0), Complex32::new(0.0, 0.0)],
            },
        ],
    )
    .unwrap();
    let complex_dense = complex
        .materialize(engine)
        .unwrap()
        .pinv(engine, &[0], &[1], 0.25)
        .unwrap();
    let complex_direct = complex.pinv(engine, &[0], &[1], 0.25).unwrap();
    assert!(complex_direct.dense_data().is_err());
    for (&actual, &expected) in complex_direct
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(complex_dense.dense_data().unwrap())
    {
        assert!((actual - expected).norm() <= 1e-6);
    }

    // Multiplicity-free semantics (TensorKit
    // `pinv(::DiagonalTensorMap)`): magnitudes are compared unrounded, so a
    // finite entry whose f32 magnitude overflows is retained and inverted;
    // a retained subnormal stays compact with its IEEE reciprocal.
    let large_value = Complex32::new(f32::MAX * 0.75, f32::MAX * 0.75);
    let large: TensorMap<_, Complex32> = TensorMap::diagonal(
        engine,
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: Label::Vacuum,
                values: vec![large_value, Complex32::new(0.0, 0.0)],
            },
            SectorSpectrum {
                sector: Label::X,
                values: vec![Complex32::new(0.0, 0.0); 2],
            },
        ],
    )
    .unwrap();
    let direct = large.pinv(engine, &[0], &[1], 0.0).unwrap();
    assert!(direct.dense_data().is_err());
    let inverse = direct.diagview(engine).unwrap()[0].values[0];
    let expected = Complex64::new(1.0, 0.0)
        / Complex64::new(f64::from(large_value.re), f64::from(large_value.im));
    assert!(
        (Complex64::new(f64::from(inverse.re), f64::from(inverse.im)) - expected).norm()
            <= 1e-6 * expected.norm(),
        "{inverse} != {expected}"
    );

    for value in [f32::from_bits(1), f32::from_bits(1 << 22)] {
        let tiny: TensorMap<_, f32> = TensorMap::diagonal(
            engine,
            &runtime,
            &bond,
            [
                SectorSpectrum {
                    sector: Label::Vacuum,
                    values: vec![value, 0.0],
                },
                SectorSpectrum {
                    sector: Label::X,
                    values: vec![0.0, 0.0],
                },
            ],
        )
        .unwrap();
        let direct = tiny.pinv(engine, &[0], &[1], 0.0).unwrap();
        assert!(direct.dense_data().is_err());
        assert_eq!(
            direct.diagview(engine).unwrap()[0].values,
            [1.0 / value, 0.0]
        );
    }

    for value in [f64::from_bits(1), 1e-308] {
        let tiny: TensorMap<_, f64> = TensorMap::diagonal(
            engine,
            &runtime,
            &bond,
            [
                SectorSpectrum {
                    sector: Label::Vacuum,
                    values: vec![value, 0.0],
                },
                SectorSpectrum {
                    sector: Label::X,
                    values: vec![0.0, 0.0],
                },
            ],
        )
        .unwrap();
        let direct = tiny.pinv(engine, &[0], &[1], 0.0).unwrap();
        assert!(direct.dense_data().is_err());
        assert_eq!(
            direct.diagview(engine).unwrap()[0].values,
            [1.0 / value, 0.0]
        );
    }

    let tiny = Complex32::new(f32::from_bits(1), 0.0);
    let tiny_complex: TensorMap<_, Complex32> = TensorMap::diagonal(
        engine,
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: Label::Vacuum,
                values: vec![tiny, Complex32::new(0.0, 0.0)],
            },
            SectorSpectrum {
                sector: Label::X,
                values: vec![Complex32::new(0.0, 0.0); 2],
            },
        ],
    )
    .unwrap();
    let direct = tiny_complex.pinv(engine, &[0], &[1], 0.0).unwrap();
    assert!(direct.dense_data().is_err());
    let inverse = direct.diagview(engine).unwrap()[0].values[0];
    assert!(inverse.re.is_infinite() && inverse.re > 0.0, "{inverse}");
}

#[cfg(feature = "cgc-gen")]
fn assert_sun_checked_generic_pinv<D>(
    engine: &mut dyn Engine,
    n: usize,
    label: Vec<i64>,
    off_diagonal: D,
    close: impl Fn(D, D) -> f64,
) where
    D: symtn::typed::AdvancedLinalgScalar + fmt::Debug + PartialEq,
{
    use symtn::sector::LieGroupFusionRule;

    let runtime = Runtime::builder().build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(n).unwrap()).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(label, 2)]).unwrap();
    let source: TensorMap<_, D> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&leg, &leg],
        [&leg, &leg],
        |trees, index| {
            let row = index[0] + 2 * index[1];
            let column = index[2] + 2 * index[3];
            if trees.codomain_vertices() == trees.domain_vertices() && row == column {
                D::from_real(2.0)
            } else if trees.codomain_vertices()[0].get() == 2
                && trees.domain_vertices()[0].get() == 1
                && row == column
            {
                off_diagonal
            } else {
                D::from_real(0.0)
            }
        },
    )
    .unwrap();
    assert!((0..source.subblock_count()).any(|index| {
        source
            .subblock_fusion_trees(index)
            .unwrap()
            .codomain_vertices()
            .iter()
            .chain(
                source
                    .subblock_fusion_trees(index)
                    .unwrap()
                    .domain_vertices(),
            )
            .any(|vertex| vertex.get() > 1)
    }));
    let pseudo = source.pinv(engine, &[0, 1], &[2, 3], 1e-12).unwrap();
    assert!(std::ptr::eq(pseudo.provider(), provider.as_ref()));
    assert_eq!(pseudo.codomain(), source.domain());
    assert_eq!(pseudo.domain(), source.codomain());
    let expected: TensorMap<_, D> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&leg, &leg],
        [&leg, &leg],
        |trees, index| {
            let row = index[0] + 2 * index[1];
            let column = index[2] + 2 * index[3];
            if trees.codomain_vertices() == trees.domain_vertices() && row == column {
                D::from_real(0.5)
            } else if trees.codomain_vertices()[0].get() == 2
                && trees.domain_vertices()[0].get() == 1
                && row == column
            {
                D::from_real(-0.25) * off_diagonal
            } else {
                D::from_real(0.0)
            }
        },
    )
    .unwrap();
    for index in 0..pseudo.subblock_count() {
        let actual = pseudo.subblock(index).unwrap();
        let expected_block = expected.subblock(index).unwrap();
        assert_eq!(actual.key(), expected_block.key());
        assert_eq!(actual.shape(), expected_block.shape());
        assert_eq!(actual.strides(), expected_block.strides());
    }
    for (actual, expected) in pseudo
        .dense_data()
        .unwrap()
        .iter()
        .zip(expected.dense_data().unwrap())
    {
        assert!(close(*actual, *expected) < 1e-9);
    }
    assert!((0..source.subblock_count()).any(|index| {
        let trees = source.subblock_fusion_trees(index).unwrap();
        trees.codomain_vertices()[0].get() == 2
            && trees.domain_vertices()[0].get() == 1
            && source.subblock(index).unwrap().shape() == [2, 2, 2, 2]
            && source.dense_data().unwrap()[source.subblock(index).unwrap().offset()]
                == off_diagonal
    }));
    let aa_plus = source.compose(engine, &pseudo).unwrap();
    let a_plus_a = pseudo.compose(engine, &source).unwrap();
    for (actual, expected) in aa_plus
        .compose(engine, &source)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(source.dense_data().unwrap())
    {
        assert!(close(*actual, *expected) < 1e-9);
    }
    for (actual, expected) in a_plus_a
        .compose(engine, &pseudo)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(pseudo.dense_data().unwrap())
    {
        assert!(close(*actual, *expected) < 1e-9);
    }
    for (actual, expected) in aa_plus
        .adjoint(engine)
        .unwrap()
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(aa_plus.dense_data().unwrap())
    {
        assert!(close(*actual, *expected) < 1e-9);
    }
    for (actual, expected) in a_plus_a
        .adjoint(engine)
        .unwrap()
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(a_plus_a.dense_data().unwrap())
    {
        assert!(close(*actual, *expected) < 1e-9);
    }
    let lazy = source.adjoint(engine).unwrap();
    let lazy_pseudo = lazy.pinv(engine, &[0, 1], &[2, 3], 1e-12).unwrap();
    for (actual, expected) in lazy_pseudo.dense_data().unwrap().iter().zip(
        pseudo
            .adjoint(engine)
            .unwrap()
            .materialize(engine)
            .unwrap()
            .dense_data()
            .unwrap(),
    ) {
        assert!(close(*actual, *expected) < 1e-9);
    }
}

#[cfg(feature = "cgc-gen")]
#[test]
fn sun_checked_generic_pinv_cross_mu_full_keys_for_both_dtypes() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    for (n, label) in [(3, vec![1, 1]), (4, vec![1, 0, 1])] {
        assert_sun_checked_generic_pinv::<f64>(
            engine,
            n,
            label.clone(),
            1.0,
            |actual, expected| (actual - expected).abs(),
        );
        assert_sun_checked_generic_pinv::<Complex64>(
            engine,
            n,
            label,
            Complex64::new(1.0, 0.5),
            |actual, expected| (actual - expected).norm(),
        );
    }
}

/// A compact diagonal's pseudo-inverse keeps the input space, dual included
/// (TensorKit `pinv(::DiagonalTensorMap)` returns `DiagonalTensorMap(_,
/// d.domain)`), so no provider query is made and a failing or style-changing
/// provider cannot fail it.
#[test]
fn checked_compact_diagonal_pinv_keeps_the_input_space_without_provider_queries() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    for dual in [false, true] {
        for fault in ["fail_algebra", "invalid_style"] {
            let provider = Arc::new(CheckedOnlyToy::new_product_probe(1));
            let mut bond = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
            if dual {
                bond = bond.try_dual().unwrap();
            }
            let diagonal: TensorMap<_, f64> = TensorMap::diagonal(
                engine,
                &runtime,
                &bond,
                [SectorSpectrum {
                    sector: Label::X,
                    values: vec![2.0, 0.0],
                }],
            )
            .unwrap();
            match fault {
                "fail_algebra" => provider.fail_algebra.store(true, Ordering::Relaxed),
                _ => provider.invalid_style.store(true, Ordering::Relaxed),
            }
            reset_provider_queries(&provider);
            let pseudo = diagonal.pinv(engine, &[0], &[1], 1e-12).unwrap();
            assert_eq!(provider.queries_since_reset.load(Ordering::Relaxed), 0);
            assert_eq!(pseudo.codomain(), diagonal.codomain());
            assert_eq!(pseudo.domain(), diagonal.domain());
        }
    }
}
