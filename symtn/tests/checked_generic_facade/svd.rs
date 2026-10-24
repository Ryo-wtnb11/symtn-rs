use super::*;

#[cfg(feature = "cgc-gen")]
#[test]
fn sun_checked_generic_compact_svd_preserves_provider_and_reconstructs() {
    use symtn::sector::LieGroupFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |trees, _| {
            trees.coupled().iter().sum::<i64>() as f64 + 1.0
        })
        .unwrap();

    let Svd { u, s, vh } = source.svd_compact(&[0], &[1]).unwrap();
    assert!(std::ptr::eq(u.provider(), provider.as_ref()));
    assert!(std::ptr::eq(s.provider(), provider.as_ref()));
    assert!(std::ptr::eq(vh.provider(), provider.as_ref()));
    let rebuilt = u.compose(&s).unwrap().compose(&vh).unwrap();
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(source.dense_data().unwrap())
        .all(|(actual, expected)| (actual - expected).abs() < 1.0e-10));

    let complex = source.convert::<Complex64>();
    let Svd {
        u: complex_u,
        s: complex_s,
        vh: complex_vh,
    } = complex.svd_compact(&[0], &[1]).unwrap();
    assert!(std::ptr::eq(complex_u.provider(), provider.as_ref()));
    assert!(std::ptr::eq(complex_s.provider(), provider.as_ref()));
    assert!(std::ptr::eq(complex_vh.provider(), provider.as_ref()));
    let complex_rebuilt = complex_u
        .compose(&complex_s)
        .unwrap()
        .compose(&complex_vh)
        .unwrap();
    assert!(complex_rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(complex.dense_data().unwrap())
        .all(|(actual, expected)| (*actual - *expected).norm() < 1.0e-10));
}

#[test]
fn checked_generic_dense_input_svd_publishes_compact_multisector_s() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 2), (Label::X, 3)]).unwrap();
    let real: TensorMap<_, f64> = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, ij| {
        if ij[0] == ij[1] {
            (ij[0] + 2) as f64
        } else {
            0.25
        }
    })
    .unwrap();
    let Svd { u, s, vh } = real.svd_compact(&[0], &[1]).unwrap();
    assert!(std::ptr::eq(s.provider(), provider.as_ref()));
    assert_eq!(s.codomain(), s.domain());
    assert!(s.dense_data().is_err());
    let spectrum = s.diagview().unwrap();
    assert_eq!(
        spectrum
            .iter()
            .map(|entry| entry.values.len())
            .sum::<usize>(),
        5
    );
    assert!(spectrum
        .iter()
        .all(|entry| entry.values.windows(2).all(|pair| pair[0] >= pair[1])));
    assert!(s.map_diagonal(f64::sqrt).is_ok());
    let rebuilt = u.compose(&s).unwrap().compose(&vh).unwrap();
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(real.dense_data().unwrap())
        .all(|(actual, expected)| (actual - expected).abs() < 1.0e-10));

    let complex = real
        .convert::<Complex64>()
        .scale(Complex64::new(1.0, 0.5))
        .unwrap();
    let Svd { u, s, vh } = complex.svd_compact(&[0], &[1]).unwrap();
    assert!(std::ptr::eq(s.provider(), provider.as_ref()));
    assert_eq!(s.codomain(), s.domain());
    assert!(s.dense_data().is_err());
    assert_eq!(
        s.diagview()
            .unwrap()
            .iter()
            .map(|entry| entry.values.len())
            .sum::<usize>(),
        5
    );
    let rebuilt = u.compose(&s).unwrap().compose(&vh).unwrap();
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(complex.dense_data().unwrap())
        .all(|(actual, expected)| (*actual - *expected).norm() < 1.0e-10));

    {
        let Svd { u, s, vh } = real.svd_full(&[0], &[1]).unwrap();
        assert!(std::ptr::eq(s.provider(), provider.as_ref()));
        assert_eq!(s.codomain(), s.domain());
        assert!(s.dense_data().is_err());
        assert!(
            symtn::typed::__network::network_reuse_class(&s, false) == NetworkReuseClass::Compact
        );
        let values = s.diagview().unwrap();
        assert_eq!(values.len(), 2);
        assert_eq!(values[0].sector, Label::Vacuum);
        assert_eq!(values[1].sector, Label::X);
        assert_eq!(values[0].values.len(), 2);
        assert_eq!(values[1].values.len(), 3);
        assert!(values
            .iter()
            .all(|entry| entry.values.windows(2).all(|pair| pair[0] >= pair[1])));
        assert_eq!(
            s.materialize().unwrap().diagview().unwrap(),
            s.diagview().unwrap()
        );
        let rebuilt = u.compose(&s).unwrap().compose(&vh).unwrap();
        assert!(rebuilt
            .dense_data()
            .unwrap()
            .iter()
            .zip(real.dense_data().unwrap())
            .all(|(actual, expected)| (actual - expected).abs() < 1.0e-10));
    }
    let Svd { u, s, vh } = complex.svd_full(&[0], &[1]).unwrap();
    assert!(s.dense_data().is_err());
    assert_eq!(s.diagview().unwrap().len(), 2);
    let rebuilt = u.compose(&s).unwrap().compose(&vh).unwrap();
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(complex.dense_data().unwrap())
        .all(|(actual, expected)| (*actual - *expected).norm() < 1.0e-10));
}

#[test]
fn checked_compact_diagonal_svd_matches_hand_permutation_and_phase() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let bond =
        GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 2), (Label::X, 3)]).unwrap();
    let input: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: Label::Vacuum,
                values: vec![Complex64::new(0.0, 0.0), Complex64::new(-2.0, 0.0)],
            },
            SectorSpectrum {
                sector: Label::X,
                values: vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.0, 3.0),
                    Complex64::new(-1.0, 0.0),
                ],
            },
        ],
    )
    .unwrap();
    let Svd { u, s, vh } = input.svd_compact(&[0], &[1]).unwrap();
    for factor in [&u, &s, &vh] {
        assert!(std::ptr::eq(factor.provider(), provider.as_ref()));
    }
    assert_eq!(
        s.diagview().unwrap()[0].values,
        [Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.0)]
    );
    assert_eq!(
        s.diagview().unwrap()[1].values,
        [
            Complex64::new(3.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0)
        ]
    );

    let expected_u: TensorMap<_, Complex64> = TensorMap::from_subblock_fn(
        &runtime,
        u.codomain().iter(),
        u.domain().iter(),
        |trees, ij| {
            let order: &[usize] = if trees.coupled() == &Label::Vacuum {
                &[1, 0]
            } else {
                &[1, 0, 2]
            };
            Complex64::new(f64::from(ij[0] == order[ij[1]]), 0.0)
        },
    )
    .unwrap();
    let expected_vh: TensorMap<_, Complex64> = TensorMap::from_subblock_fn(
        &runtime,
        vh.codomain().iter(),
        vh.domain().iter(),
        |trees, ij| {
            let (order, phases): (&[usize], &[Complex64]) = if trees.coupled() == &Label::Vacuum {
                (
                    &[1, 0],
                    &[Complex64::new(-1.0, 0.0), Complex64::new(1.0, 0.0)],
                )
            } else {
                (
                    &[1, 0, 2],
                    &[
                        Complex64::new(0.0, 1.0),
                        Complex64::new(1.0, 0.0),
                        Complex64::new(-1.0, 0.0),
                    ],
                )
            };
            if ij[1] == order[ij[0]] {
                phases[ij[0]]
            } else {
                Complex64::new(0.0, 0.0)
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
            .all(|(actual, expected)| (*actual - expected).norm() <= 1e-12));
    }
    for gram in [
        u.adjoint()
            .unwrap()
            .materialize()
            .unwrap()
            .compose(&u)
            .unwrap(),
        vh.compose(&vh.adjoint().unwrap().materialize().unwrap())
            .unwrap(),
    ] {
        let identity: TensorMap<_, Complex64> = TensorMap::from_subblock_fn(
            &runtime,
            gram.codomain().iter(),
            gram.domain().iter(),
            |_, ij| Complex64::new(f64::from(ij[0] == ij[1]), 0.0),
        )
        .unwrap();
        assert!(gram
            .dense_data()
            .unwrap()
            .iter()
            .zip(identity.dense_data().unwrap())
            .all(|(actual, expected)| (*actual - expected).norm() <= 1e-12));
    }
    let rebuilt = u.compose(&s).unwrap().compose(&vh).unwrap();
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(input.materialize().unwrap().dense_data().unwrap())
        .all(|(actual, expected)| (*actual - expected).norm() <= 1e-12));

    let Svd { u, s, vh } = input.svd_full(&[0], &[1]).unwrap();
    for factor in [&u, &s, &vh] {
        assert!(std::ptr::eq(factor.provider(), provider.as_ref()));
    }
    assert!(s.dense_data().is_err());
    for (actual, expected) in [(&u, &expected_u), (&vh, &expected_vh)] {
        assert!(actual
            .dense_data()
            .unwrap()
            .iter()
            .zip(expected.dense_data().unwrap())
            .all(|(actual, expected)| (*actual - expected).norm() <= 1e-12));
    }
    let rebuilt = u.compose(&s).unwrap().compose(&vh).unwrap();
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(input.materialize().unwrap().dense_data().unwrap())
        .all(|(actual, expected)| (*actual - expected).norm() <= 1e-12));

    let narrow = input.convert::<Complex32>();
    let Svd { u, s, vh } = narrow.svd_compact(&[0], &[1]).unwrap();
    assert_eq!(
        s.diagview().unwrap()[0].values,
        [Complex32::new(2.0, 0.0), Complex32::new(0.0, 0.0)]
    );
    let rebuilt = u.compose(&s).unwrap().compose(&vh).unwrap();
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(narrow.materialize().unwrap().dense_data().unwrap())
        .all(|(actual, expected)| (*actual - expected).norm() <= 1e-5));
    let Svd { u, s, vh } = narrow.svd_full(&[0], &[1]).unwrap();
    let rebuilt = u.compose(&s).unwrap().compose(&vh).unwrap();
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(narrow.materialize().unwrap().dense_data().unwrap())
        .all(|(actual, expected)| (*actual - expected).norm() <= 1e-5));
}

#[test]
fn checked_compact_diagonal_svd_rejects_nonfinite_and_skips_the_provider_when_finite() {
    let _cache = cache_shared();
    let svd_calls = Arc::new(SpyCounts::default());
    let runtime = Runtime::builder()
        .dense_threads(1)
        .with_dense_executor(Box::new(pinv_spy(&svd_calls, Some(1), None)))
        .build()
        .unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let bond = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let nonfinite: TensorMap<_, f64> = TensorMap::diagonal(
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
    let error = nonfinite.svd_compact(&[0], &[1]).unwrap_err();
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
            finite.svd_compact(&[0], &[1]).unwrap(),
            finite.svd_full(&[0], &[1]).unwrap(),
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

#[test]
fn checked_compact_diagonal_svd_is_direct_for_subnormal_and_overflowing_values() {
    let _cache = cache_shared();
    let svd_calls = Arc::new(SpyCounts::default());
    let runtime = Runtime::builder()
        .dense_threads(1)
        .with_dense_executor(Box::new(pinv_spy(&svd_calls, None, None)))
        .build()
        .unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let bond = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();

    // Hand-computed MAK `svd_full!(::DiagonalAlgorithm)`: `S = abs(a)`
    // exactly, with no dense solver, for a subnormal and the smallest normal.
    // Why not compare with the dense route: LAPACK may flush a subnormal
    // singular value to zero (f32), which is a solver limit, not the result.
    macro_rules! assert_real_magnitude {
        ($dtype:ty, $subnormal:expr, $minimum:expr) => {
            for (name, value) in [("subnormal", $subnormal), ("minimum normal", $minimum)] {
                let input: TensorMap<_, $dtype> = TensorMap::diagonal(
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
                        input.svd_full(&[0], &[1]).unwrap()
                    } else {
                        input.svd_compact(&[0], &[1]).unwrap()
                    };
                    assert_eq!(svd_calls.of(PINV_SVD), 0, "{name}, full={full}");
                    assert_eq!(
                        s.diagview().unwrap()[0].values,
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
                    overflow.svd_full(&[0], &[1]).unwrap()
                } else {
                    overflow.svd_compact(&[0], &[1]).unwrap()
                };
                assert_eq!(svd_calls.of(PINV_SVD), 0, "full={full}");
                assert_eq!(s.diagview().unwrap()[0].values, [$infinite], "full={full}");
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
fn checked_compact_diagonal_svd_preserves_dual_and_changed_roles() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let dual_bond = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)])
        .unwrap()
        .try_dual()
        .unwrap();
    let input: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &dual_bond,
        [SectorSpectrum {
            sector: Label::X,
            values: vec![-2.0, 0.0],
        }],
    )
    .unwrap();
    let Svd { u, s, vh } = input.svd_compact(&[0], &[1]).unwrap();
    assert!(std::ptr::eq(u.provider(), provider.as_ref()));
    assert_eq!(s.diagview().unwrap()[0].values, [2.0, 0.0]);
    let rebuilt = u.compose(&s).unwrap().compose(&vh).unwrap();
    assert_eq!(
        rebuilt.dense_data().unwrap(),
        input.materialize().unwrap().dense_data().unwrap()
    );
    let Svd { u, s, vh } = input.svd_full(&[0], &[1]).unwrap();
    assert!(std::ptr::eq(u.provider(), provider.as_ref()));
    assert_eq!(s.diagview().unwrap()[0].values, [2.0, 0.0]);
    let rebuilt = u.compose(&s).unwrap().compose(&vh).unwrap();
    assert_eq!(
        rebuilt.dense_data().unwrap(),
        input.materialize().unwrap().dense_data().unwrap()
    );
    // Changed roles enter the checked transform first; this provider rejects
    // that transform before SVD can inspect the compact payload.
    assert!(matches!(
        input.svd_compact(&[1], &[0]),
        Err(GenericTensorError::Plan(
            symtn::typed::CheckedGenericPlanError::Operation(
                symtn::typed::OperationError::EmptyTransformBlock
            )
        ))
    ));
    assert!(matches!(
        input.svd_full(&[1], &[0]),
        Err(GenericTensorError::Plan(
            symtn::typed::CheckedGenericPlanError::Operation(
                symtn::typed::OperationError::EmptyTransformBlock
            )
        ))
    ));
}

#[test]
fn checked_generic_full_svd_keeps_dense_s_for_equal_total_but_unequal_sector_bonds() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let rows =
        GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 2), (Label::X, 3)]).unwrap();
    let cols =
        GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 3), (Label::X, 2)]).unwrap();
    let input: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&rows], [&cols], |_, ij| {
            if ij[0] == ij[1] {
                2.0
            } else {
                0.25
            }
        })
        .unwrap();
    let Svd { u, s, vh } = input.svd_full(&[0], &[1]).unwrap();
    assert_ne!(s.codomain(), s.domain());
    assert_eq!(s.codomain(), u.domain());
    assert_eq!(s.domain(), vh.codomain());
    assert!(s.dense_data().is_ok());
    let rebuilt = u.compose(&s).unwrap().compose(&vh).unwrap();
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(input.dense_data().unwrap())
        .all(|(actual, expected)| (actual - expected).abs() < 1.0e-10));
}

#[test]
fn checked_compact_diagonal_svd_full_is_its_compact_svd_under_a_provider_bond_mismatch() {
    let _cache = cache_shared();
    let svd_calls = Arc::new(SpyCounts::default());
    let runtime = Runtime::builder()
        .dense_threads(1)
        .with_dense_executor(Box::new(pinv_spy(&svd_calls, None, None)))
        .build()
        .unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let bond = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let on = |bond: &GradedSpace<_>| -> TensorMap<_, f64> {
        TensorMap::diagonal(
            &runtime,
            bond,
            [SectorSpectrum {
                sector: Label::X,
                values: vec![2.0, -1.0],
            }],
        )
        .unwrap()
    };
    let nondual = on(&bond);
    let dual = on(&bond.try_dual().unwrap());
    provider.extra_vacuum_channel.store(true, Ordering::Relaxed);

    // A nondual one-leg bond is its own SVD bond (`fuse(V) = V`, no fusion),
    // so the factors are direct on the input space and the provider's
    // coupled-dimension report is never consulted.
    reset_provider_queries(&provider);
    let Svd { u, s, vh } = nondual.svd_full(&[0], &[1]).unwrap();
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
    let full = dual.svd_full(&[0], &[1]).unwrap();
    let compact = dual.svd_compact(&[0], &[1]).unwrap();
    assert_eq!(svd_calls.of(PINV_SVD), 0);
    assert!(full.s.dense_data().is_err());
    assert_eq!(full.s.diagview().unwrap(), compact.s.diagview().unwrap());
    assert_eq!(full.s.codomain(), compact.s.codomain());
    for (full, compact) in [(&full.u, &compact.u), (&full.vh, &compact.vh)] {
        assert_eq!(full.codomain(), compact.codomain());
        assert_eq!(full.domain(), compact.domain());
        assert_eq!(full.dense_data().unwrap(), compact.dense_data().unwrap());
    }
}

#[cfg(feature = "cgc-gen")]
#[test]
fn sun_checked_generic_full_svd_preserves_provider_reconstructs_and_accepts_lazy() {
    use symtn::sector::LieGroupFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |trees, _| {
            trees.coupled().iter().sum::<i64>() as f64 + 1.0
        })
        .unwrap();

    let Svd { u, s, vh } = source.svd_full(&[0], &[1]).unwrap();
    assert!(std::ptr::eq(u.provider(), provider.as_ref()));
    assert!(std::ptr::eq(s.provider(), provider.as_ref()));
    assert!(std::ptr::eq(vh.provider(), provider.as_ref()));
    assert!(s.dense_data().is_err());
    assert_eq!(s.diagview().unwrap().len(), 1);
    let rebuilt = u.compose(&s).unwrap().compose(&vh).unwrap();
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(source.dense_data().unwrap())
        .all(|(actual, expected)| (actual - expected).abs() < 1.0e-10));

    let complex = source.convert::<Complex64>();
    let Svd {
        u: complex_u,
        s: complex_s,
        vh: complex_vh,
    } = complex.svd_full(&[0], &[1]).unwrap();
    let complex_rebuilt = complex_u
        .compose(&complex_s)
        .unwrap()
        .compose(&complex_vh)
        .unwrap();
    assert!(complex_rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(complex.dense_data().unwrap())
        .all(|(actual, expected)| (*actual - *expected).norm() < 1.0e-10));

    // A lazy adjoint is factored from its parent, in the materialized
    // adjoint's spaces and gauge.
    let lazy = complex.adjoint().unwrap();
    let Svd {
        u: lazy_u,
        s: lazy_s,
        vh: lazy_vh,
    } = lazy.svd_full(&[0], &[1]).unwrap();
    let Svd {
        u: eager_u,
        s: eager_s,
        vh: eager_vh,
    } = lazy.materialize().unwrap().svd_full(&[0], &[1]).unwrap();
    for (lazy, eager) in [(lazy_u, eager_u), (lazy_s, eager_s), (lazy_vh, eager_vh)] {
        assert_eq!(lazy.codomain(), eager.codomain());
        assert_eq!(lazy.domain(), eager.domain());
        assert!(lazy
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .zip(eager.materialize().unwrap().dense_data().unwrap())
            .all(|(actual, expected)| (*actual - *expected).norm() < 1.0e-10));
    }
}

#[cfg(feature = "cgc-gen")]
#[test]
fn sun_checked_generic_svd_vals_matches_compact_spectrum() {
    use symtn::sector::LieGroupFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |trees, _| {
            trees.coupled().iter().sum::<i64>() as f64 + 1.0
        })
        .unwrap();
    let spectra = source.svd_vals(&[0], &[1]).unwrap();
    assert!(!spectra.is_empty());
    assert!(spectra.iter().all(|spectrum| spectrum
        .values
        .iter()
        .all(|value| value.is_finite() && *value >= 0.0)));

    let complex = source.convert::<Complex64>();
    let complex_spectra = complex.svd_vals(&[0], &[1]).unwrap();
    assert_eq!(complex_spectra, spectra);
}

#[test]
fn checked_compact_svd_vals_preserves_leg_roles_and_decode_failure() {
    use symtn::typed::{CheckedGenericPlanError, OperationError};

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 2), (Label::X, 3)]).unwrap();
    let compact: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: Label::X,
                values: vec![-3.0, 0.0, 2.0],
            },
            SectorSpectrum {
                sector: Label::Vacuum,
                values: vec![-4.0, 1.0],
            },
        ],
    )
    .unwrap();
    let expected = compact.materialize().unwrap();
    assert_eq!(
        compact.svd_vals(&[0], &[1]).unwrap(),
        expected.svd_vals(&[0], &[1]).unwrap()
    );
    // Changing the leg roles takes the checked transform boundary before SVD.
    assert!(matches!(
        compact.svd_vals(&[1], &[0]),
        Err(GenericTensorError::Plan(
            CheckedGenericPlanError::Operation(OperationError::EmptyTransformBlock)
        ))
    ));

    provider.fail_decode.store(true, Ordering::Relaxed);
    assert!(matches!(
        compact.svd_vals(&[0], &[1]),
        Err(GenericTensorError::Plan(CheckedGenericPlanError::Provider(
            _
        )))
    ));
    provider.fail_decode.store(false, Ordering::Relaxed);
}

#[test]
fn checked_generic_svd_truncation_reconstructs_and_preserves_provider() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
            if indices[0] == indices[1] {
                (indices[0] + 2) as f64
            } else {
                0.0
            }
        })
        .unwrap();
    // The truncated SVD is a composition (#1534).
    let Svd { u, s, vh } = source.svd_compact(&[0], &[1]).unwrap();
    let found = s.domain()[0]
        .find_truncated(&s.diagview().unwrap(), &Truncation::rank(1))
        .unwrap();
    let u = u
        .restrict_leg(&[(u.codomain_rank(), &found.selection)])
        .unwrap();
    let s = s
        .restrict_leg(&[(0, &found.selection), (1, &found.selection)])
        .unwrap();
    let vh = vh.restrict_leg(&[(0, &found.selection)]).unwrap();
    assert!(std::ptr::eq(u.provider(), provider.as_ref()));
    assert!(std::ptr::eq(s.provider(), provider.as_ref()));
    assert!(std::ptr::eq(vh.provider(), provider.as_ref()));
    let rebuilt = u.compose(&s).unwrap().compose(&vh).unwrap();
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .all(|value| value.is_finite()));
    assert!(s.diagview().unwrap().iter().all(|spectrum| {
        spectrum.values.len() <= 2 && spectrum.values.iter().all(|value| value.is_finite())
    }));
}

#[test]
fn checked_generic_compact_svd_failure_is_typed_and_nonpublishing() {
    let _cache = cache_exclusive();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(122));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg], |_, _| 2.0).unwrap();
    let before = source.dense_data().unwrap().to_vec();
    // The construction published the layouts this operation walks; a
    // provider whose answers change is not one identity, so the cached
    // layouts go first (#2030). The tag is this test's alone.
    forget_cached_structures();
    provider.fail_algebra.store(true, Ordering::Relaxed);
    let error = source.svd_compact(&[0, 1], &[2]).unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Plan(symtn::typed::CheckedGenericPlanError::Provider(
            ToyError::Algebra
        ))
    ));
    assert_eq!(source.dense_data().unwrap(), before.as_slice());
}

#[test]
fn checked_generic_full_svd_failure_is_typed_and_nonpublishing() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, ij| {
            if ij[0] == ij[1] {
                2.0
            } else {
                0.25
            }
        })
        .unwrap();
    let before = source.dense_data().unwrap().to_vec();
    provider.fail_algebra.store(true, Ordering::Relaxed);
    let error = source.svd_full(&[0], &[1]).unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Plan(symtn::typed::CheckedGenericPlanError::Provider(
            ToyError::Algebra
        ))
    ));
    assert_eq!(source.dense_data().unwrap(), before.as_slice());
}

/// A dual compact diagonal's SVD bond is the fresh nondual `fuse(V)`, built
/// by the provider, so a provider that changes its fusion style still fails
/// `svd_compact` with the typed plan error and runs no dense SVD.
#[test]
fn checked_dual_compact_diagonal_svd_compact_propagates_provider_errors() {
    let _cache = cache_shared();
    let svd_calls = Arc::new(SpyCounts::default());
    let runtime = Runtime::builder()
        .dense_threads(1)
        .with_dense_executor(Box::new(pinv_spy(&svd_calls, None, None)))
        .build()
        .unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let bond = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)])
        .unwrap()
        .try_dual()
        .unwrap();
    let diagonal: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &bond,
        [SectorSpectrum {
            sector: Label::X,
            values: vec![2.0, 1.0],
        }],
    )
    .unwrap();
    provider.invalid_style.store(true, Ordering::Relaxed);
    let error = diagonal.svd_compact(&[0], &[1]).unwrap_err();
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

/// QR/LQ/SVD and the value-only spectra of a facade-layout source (#1494),
/// each against an oracle that does not share the factorization's layout:
/// reconstruction, isometry against an explicit tree-basis identity, `S`
/// against `svd_vals`, and `eigh_vals`/`eig_vals` of `A Aᴴ` against the
/// squared singular values.
#[cfg(feature = "cgc-gen")]
macro_rules! assert_sun_compact_laws {
    ($runtime:expr, $source:expr) => {{
        let source = $source;
        let assert_close = |actual: &TensorMap<_, _>, expected: &TensorMap<_, _>, what: &str| {
            let error = actual
                .axpby(1.0.into(), expected, (-1.0).into())
                .unwrap()
                .norm(2.0)
                .unwrap();
            assert!(
                error < 1e-9 * (1.0 + expected.norm(2.0).unwrap()),
                "{what}: {error}"
            );
        };
        let owned_adjoint =
            |tensor: &TensorMap<_, _>| tensor.adjoint().unwrap().materialize().unwrap();
        let assert_identity = |gram: &TensorMap<_, _>, what: &str| {
            let (codomain, domain) = (gram.codomain(), gram.domain());
            let rank = codomain.iter().count();
            let identity = TensorMap::from_subblock_fn(
                $runtime,
                codomain.iter(),
                domain.iter(),
                |trees, ij| {
                    let same_tree = trees.codomain_uncoupled() == trees.domain_uncoupled()
                        && trees.codomain_innerlines() == trees.domain_innerlines()
                        && trees.codomain_vertices() == trees.domain_vertices();
                    let (row, col) = ij.split_at(rank);
                    if same_tree && row == col { 1.0 } else { 0.0 }.into()
                },
            )
            .unwrap();
            assert_close(gram, &identity, what);
        };
        let by_sector = |spectra: Vec<SectorSpectrum<_>>| {
            spectra
                .into_iter()
                .filter(|spectrum| !spectrum.values.is_empty())
                .map(|spectrum| {
                    let mut values = spectrum.values;
                    values.sort_by(|a, b| b.total_cmp(a));
                    (spectrum.sector, values)
                })
                .collect::<Vec<_>>()
        };
        fn find<S: PartialEq>(spectra: &[(S, Vec<f64>)], sector: &S) -> Vec<f64> {
            spectra
                .iter()
                .find(|(key, _)| key == sector)
                .map_or_else(Vec::new, |(_, values)| values.clone())
        }
        let assert_values = |actual: &[f64], expected: &[f64], what: &str| {
            let scale = 1.0 + expected.first().copied().unwrap_or(0.0).abs();
            for (index, &actual) in actual.iter().enumerate() {
                let expected = expected.get(index).copied().unwrap_or(0.0);
                assert!((actual - expected).abs() < 1e-9 * scale, "{what}");
            }
        };

        let Qr { q, r } = source
            .qr_compact(&codomain_axes(&source), &domain_axes(&source))
            .unwrap();
        assert_close(&q.compose(&r).unwrap(), source, "A = QR");
        assert_identity(&owned_adjoint(&q).compose(&q).unwrap(), "Qᴴ Q = 1");
        let Lq { l, q } = source
            .lq_compact(&codomain_axes(&source), &domain_axes(&source))
            .unwrap();
        assert_close(&l.compose(&q).unwrap(), source, "A = LQ");
        assert_identity(&q.compose(&owned_adjoint(&q)).unwrap(), "Q Qᴴ = 1");
        let Svd { u, s, vh } = source
            .svd_compact(&codomain_axes(&source), &domain_axes(&source))
            .unwrap();
        assert_close(
            &u.compose(&s).unwrap().compose(&vh).unwrap(),
            source,
            "A = U S Vh",
        );
        assert_identity(&owned_adjoint(&u).compose(&u).unwrap(), "Uᴴ U = 1");
        assert_identity(&vh.compose(&owned_adjoint(&vh)).unwrap(), "Vh Vhᴴ = 1");

        let singular = by_sector(
            source
                .svd_vals(&codomain_axes(&source), &domain_axes(&source))
                .unwrap(),
        );
        let diagonal = by_sector(s.eigh_vals(&[0], &[1], HermitianTol::DEFAULT).unwrap());
        assert_eq!(singular.len(), diagonal.len());
        for (sector, values) in &singular {
            assert_eq!(find(&diagonal, sector).len(), values.len());
            assert_values(&find(&diagonal, sector), values, "S = svd_vals");
        }
        let gram = source.compose(&owned_adjoint(source)).unwrap();
        let squared = singular
            .iter()
            .map(|(sector, values)| (sector.clone(), values.iter().map(|v| v * v).collect()))
            .collect::<Vec<_>>();
        for (sector, values) in by_sector(
            gram.eigh_vals(
                &codomain_axes(&gram),
                &domain_axes(&gram),
                HermitianTol::DEFAULT,
            )
            .unwrap(),
        ) {
            assert_values(&values, &find(&squared, &sector), "eigh_vals(A Aᴴ) = s²");
        }
        for spectrum in gram
            .eig_vals(&codomain_axes(&gram), &domain_axes(&gram))
            .unwrap()
        {
            let mut values = spectrum.values.iter().map(|v| v.re).collect::<Vec<_>>();
            assert!(spectrum
                .values
                .iter()
                .all(|v| v.im.abs() < 1e-9 * (1.0 + v.norm())));
            values.sort_by(|a, b| b.total_cmp(a));
            assert_values(
                &values,
                &find(&squared, &spectrum.sector),
                "eig_vals(A Aᴴ) = s²",
            );
        }
    }};
}

#[cfg(feature = "cgc-gen")]
#[test]
fn sun_checked_generic_compact_factors_and_spectra_on_facade_layouts() {
    use symtn::sector::LieGroupFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 2), (vec![0, 0], 1)]).unwrap();
    let t =
        GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 0], 2), (vec![0, 1], 1)]).unwrap();
    let t_dual = t.try_dual().unwrap();
    let value = |salt: u64| ((salt.wrapping_mul(2_654_435_761) % 1009) as f64) / 1009.0 - 0.5;
    let cases: [(&[&GradedSpace<_>], &[&GradedSpace<_>]); 7] = [
        (&[&leg, &leg], &[&leg]),
        (&[&leg], &[&leg, &leg]),
        (&[&leg, &leg], &[&leg, &leg]),
        (&[&t, &t_dual], &[&t]),
        (&[&t], &[&t, &t_dual]),
        (&[&t_dual, &t_dual], &[&t_dual]),
        (&[&t_dual, &t_dual, &t_dual], &[]),
    ];
    for (codomain, domain) in cases {
        let mut salt = 0;
        let real: TensorMap<_, f64> = TensorMap::from_subblock_fn(
            &runtime,
            codomain.iter().copied(),
            domain.iter().copied(),
            |_, _| {
                salt += 1;
                value(salt)
            },
        )
        .unwrap();
        assert_sun_compact_laws!(&runtime, &real);
        let complex: TensorMap<_, Complex64> = TensorMap::from_subblock_fn(
            &runtime,
            codomain.iter().copied(),
            domain.iter().copied(),
            |_, _| {
                salt += 2;
                Complex64::new(value(salt), value(salt + 1))
            },
        )
        .unwrap();
        assert_sun_compact_laws!(&runtime, &complex);
    }
}
