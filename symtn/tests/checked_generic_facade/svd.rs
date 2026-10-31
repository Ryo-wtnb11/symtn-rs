use super::*;

#[cfg(feature = "cgc-gen")]
#[test]
fn sun_checked_generic_compact_svd_preserves_provider_and_reconstructs() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    use symtn::sector::LieGroupFusionRule;

    let runtime = Runtime::builder().build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |trees, _| {
            trees.coupled().iter().sum::<i64>() as f64 + 1.0
        })
        .unwrap();

    let Svd { u, s, vh } = source.svd_compact(engine, &[0], &[1]).unwrap();
    assert!(std::ptr::eq(u.provider(), provider.as_ref()));
    assert!(std::ptr::eq(s.provider(), provider.as_ref()));
    assert!(std::ptr::eq(vh.provider(), provider.as_ref()));
    let rebuilt = u.compose(engine, &s).unwrap().compose(engine, &vh).unwrap();
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(source.dense_data().unwrap())
        .all(|(actual, expected)| (actual - expected).abs() < 1.0e-10));

    let complex = source.convert::<Complex64>(engine).unwrap();
    let Svd {
        u: complex_u,
        s: complex_s,
        vh: complex_vh,
    } = complex.svd_compact(engine, &[0], &[1]).unwrap();
    assert!(std::ptr::eq(complex_u.provider(), provider.as_ref()));
    assert!(std::ptr::eq(complex_s.provider(), provider.as_ref()));
    assert!(std::ptr::eq(complex_vh.provider(), provider.as_ref()));
    let complex_rebuilt = complex_u
        .compose(engine, &complex_s)
        .unwrap()
        .compose(engine, &complex_vh)
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
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 2), (Label::X, 3)]).unwrap();
    let real: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, ij| {
            if ij[0] == ij[1] {
                (ij[0] + 2) as f64
            } else {
                0.25
            }
        })
        .unwrap();
    let Svd { u, s, vh } = real.svd_compact(engine, &[0], &[1]).unwrap();
    assert!(std::ptr::eq(s.provider(), provider.as_ref()));
    assert_eq!(s.codomain(), s.domain());
    assert!(s.dense_data().is_err());
    let spectrum = s.diagview(engine).unwrap();
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
    assert!(s.map_diagonal(engine, f64::sqrt).is_ok());
    let rebuilt = u.compose(engine, &s).unwrap().compose(engine, &vh).unwrap();
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(real.dense_data().unwrap())
        .all(|(actual, expected)| (actual - expected).abs() < 1.0e-10));

    let complex = real
        .convert::<Complex64>(engine)
        .unwrap()
        .scale(engine, Complex64::new(1.0, 0.5))
        .unwrap();
    let Svd { u, s, vh } = complex.svd_compact(engine, &[0], &[1]).unwrap();
    assert!(std::ptr::eq(s.provider(), provider.as_ref()));
    assert_eq!(s.codomain(), s.domain());
    assert!(s.dense_data().is_err());
    assert_eq!(
        s.diagview(engine)
            .unwrap()
            .iter()
            .map(|entry| entry.values.len())
            .sum::<usize>(),
        5
    );
    let rebuilt = u.compose(engine, &s).unwrap().compose(engine, &vh).unwrap();
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(complex.dense_data().unwrap())
        .all(|(actual, expected)| (*actual - *expected).norm() < 1.0e-10));

    {
        let Svd { u, s, vh } = real.svd_full(engine, &[0], &[1]).unwrap();
        assert!(std::ptr::eq(s.provider(), provider.as_ref()));
        assert_eq!(s.codomain(), s.domain());
        assert!(s.dense_data().is_err());
        assert!(
            symtn::typed::__network::network_reuse_class(&s, false) == NetworkReuseClass::Compact
        );
        let values = s.diagview(engine).unwrap();
        assert_eq!(values.len(), 2);
        assert_eq!(values[0].sector, Label::Vacuum);
        assert_eq!(values[1].sector, Label::X);
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
            .zip(real.dense_data().unwrap())
            .all(|(actual, expected)| (actual - expected).abs() < 1.0e-10));
    }
    let Svd { u, s, vh } = complex.svd_full(engine, &[0], &[1]).unwrap();
    assert!(s.dense_data().is_err());
    assert_eq!(s.diagview(engine).unwrap().len(), 2);
    let rebuilt = u.compose(engine, &s).unwrap().compose(engine, &vh).unwrap();
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(complex.dense_data().unwrap())
        .all(|(actual, expected)| (*actual - *expected).norm() < 1.0e-10));
}

#[test]
fn checked_compact_diagonal_svd_matches_hand_permutation_and_phase() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let bond =
        GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 2), (Label::X, 3)]).unwrap();
    let input: TensorMap<_, Complex64> = TensorMap::diagonal(
        engine,
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
    let Svd { u, s, vh } = input.svd_compact(engine, &[0], &[1]).unwrap();
    for factor in [&u, &s, &vh] {
        assert!(std::ptr::eq(factor.provider(), provider.as_ref()));
    }
    assert_eq!(
        s.diagview(engine).unwrap()[0].values,
        [Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.0)]
    );
    assert_eq!(
        s.diagview(engine).unwrap()[1].values,
        [
            Complex64::new(3.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(1.0, 0.0)
        ]
    );

    let expected_u: TensorMap<_, Complex64> = TensorMap::from_subblock_fn(
        engine,
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
        engine,
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
    let vh_adjoint = &vh.adjoint(engine).unwrap().materialize(engine).unwrap();
    for gram in [
        u.adjoint(engine)
            .unwrap()
            .materialize(engine)
            .unwrap()
            .compose(engine, &u)
            .unwrap(),
        vh.compose(engine, vh_adjoint).unwrap(),
    ] {
        let identity: TensorMap<_, Complex64> = TensorMap::from_subblock_fn(
            engine,
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
    let rebuilt = u.compose(engine, &s).unwrap().compose(engine, &vh).unwrap();
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(input.materialize(engine).unwrap().dense_data().unwrap())
        .all(|(actual, expected)| (*actual - expected).norm() <= 1e-12));

    let Svd { u, s, vh } = input.svd_full(engine, &[0], &[1]).unwrap();
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
    let rebuilt = u.compose(engine, &s).unwrap().compose(engine, &vh).unwrap();
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(input.materialize(engine).unwrap().dense_data().unwrap())
        .all(|(actual, expected)| (*actual - expected).norm() <= 1e-12));

    let narrow = input.convert::<Complex32>(engine).unwrap();
    let Svd { u, s, vh } = narrow.svd_compact(engine, &[0], &[1]).unwrap();
    assert_eq!(
        s.diagview(engine).unwrap()[0].values,
        [Complex32::new(2.0, 0.0), Complex32::new(0.0, 0.0)]
    );
    let rebuilt = u.compose(engine, &s).unwrap().compose(engine, &vh).unwrap();
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(narrow.materialize(engine).unwrap().dense_data().unwrap())
        .all(|(actual, expected)| (*actual - expected).norm() <= 1e-5));
    let Svd { u, s, vh } = narrow.svd_full(engine, &[0], &[1]).unwrap();
    let rebuilt = u.compose(engine, &s).unwrap().compose(engine, &vh).unwrap();
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(narrow.materialize(engine).unwrap().dense_data().unwrap())
        .all(|(actual, expected)| (*actual - expected).norm() <= 1e-5));
}

#[test]
fn checked_compact_diagonal_svd_preserves_dual_and_changed_roles() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let dual_bond = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)])
        .unwrap()
        .try_dual()
        .unwrap();
    let input: TensorMap<_, f64> = TensorMap::diagonal(
        engine,
        &runtime,
        &dual_bond,
        [SectorSpectrum {
            sector: Label::X,
            values: vec![-2.0, 0.0],
        }],
    )
    .unwrap();
    let Svd { u, s, vh } = input.svd_compact(engine, &[0], &[1]).unwrap();
    assert!(std::ptr::eq(u.provider(), provider.as_ref()));
    assert_eq!(s.diagview(engine).unwrap()[0].values, [2.0, 0.0]);
    let rebuilt = u.compose(engine, &s).unwrap().compose(engine, &vh).unwrap();
    assert_eq!(
        rebuilt.dense_data().unwrap(),
        input.materialize(engine).unwrap().dense_data().unwrap()
    );
    let Svd { u, s, vh } = input.svd_full(engine, &[0], &[1]).unwrap();
    assert!(std::ptr::eq(u.provider(), provider.as_ref()));
    assert_eq!(s.diagview(engine).unwrap()[0].values, [2.0, 0.0]);
    let rebuilt = u.compose(engine, &s).unwrap().compose(engine, &vh).unwrap();
    assert_eq!(
        rebuilt.dense_data().unwrap(),
        input.materialize(engine).unwrap().dense_data().unwrap()
    );
    // Changed roles enter the checked transform first; this provider rejects
    // that transform before SVD can inspect the compact payload.
    assert!(matches!(
        input.svd_compact(engine, &[1], &[0]),
        Err(GenericTensorError::Plan(
            symtn::typed::CheckedGenericPlanError::Operation(
                symtn::typed::OperationError::EmptyTransformBlock
            )
        ))
    ));
    assert!(matches!(
        input.svd_full(engine, &[1], &[0]),
        Err(GenericTensorError::Plan(
            symtn::typed::CheckedGenericPlanError::Operation(
                symtn::typed::OperationError::EmptyTransformBlock
            )
        ))
    ));
}

#[test]
fn checked_generic_full_svd_keeps_dense_s_for_equal_total_but_unequal_sector_bonds() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let rows =
        GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 2), (Label::X, 3)]).unwrap();
    let cols =
        GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 3), (Label::X, 2)]).unwrap();
    let input: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&rows], [&cols], |_, ij| {
            if ij[0] == ij[1] {
                2.0
            } else {
                0.25
            }
        })
        .unwrap();
    let Svd { u, s, vh } = input.svd_full(engine, &[0], &[1]).unwrap();
    assert_ne!(s.codomain(), s.domain());
    assert_eq!(s.codomain(), u.domain());
    assert_eq!(s.domain(), vh.codomain());
    assert!(s.dense_data().is_ok());
    let rebuilt = u.compose(engine, &s).unwrap().compose(engine, &vh).unwrap();
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(input.dense_data().unwrap())
        .all(|(actual, expected)| (actual - expected).abs() < 1.0e-10));
}

#[cfg(feature = "cgc-gen")]
#[test]
fn sun_checked_generic_full_svd_preserves_provider_reconstructs_and_accepts_lazy() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    use symtn::sector::LieGroupFusionRule;

    let runtime = Runtime::builder().build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |trees, _| {
            trees.coupled().iter().sum::<i64>() as f64 + 1.0
        })
        .unwrap();

    let Svd { u, s, vh } = source.svd_full(engine, &[0], &[1]).unwrap();
    assert!(std::ptr::eq(u.provider(), provider.as_ref()));
    assert!(std::ptr::eq(s.provider(), provider.as_ref()));
    assert!(std::ptr::eq(vh.provider(), provider.as_ref()));
    assert!(s.dense_data().is_err());
    assert_eq!(s.diagview(engine).unwrap().len(), 1);
    let rebuilt = u.compose(engine, &s).unwrap().compose(engine, &vh).unwrap();
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(source.dense_data().unwrap())
        .all(|(actual, expected)| (actual - expected).abs() < 1.0e-10));

    let complex = source.convert::<Complex64>(engine).unwrap();
    let Svd {
        u: complex_u,
        s: complex_s,
        vh: complex_vh,
    } = complex.svd_full(engine, &[0], &[1]).unwrap();
    let complex_rebuilt = complex_u
        .compose(engine, &complex_s)
        .unwrap()
        .compose(engine, &complex_vh)
        .unwrap();
    assert!(complex_rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(complex.dense_data().unwrap())
        .all(|(actual, expected)| (*actual - *expected).norm() < 1.0e-10));

    // A lazy adjoint is factored from its parent, in the materialized
    // adjoint's spaces and gauge.
    let lazy = complex.adjoint(engine).unwrap();
    let Svd {
        u: lazy_u,
        s: lazy_s,
        vh: lazy_vh,
    } = lazy.svd_full(engine, &[0], &[1]).unwrap();
    let Svd {
        u: eager_u,
        s: eager_s,
        vh: eager_vh,
    } = lazy
        .materialize(engine)
        .unwrap()
        .svd_full(engine, &[0], &[1])
        .unwrap();
    for (lazy, eager) in [(lazy_u, eager_u), (lazy_s, eager_s), (lazy_vh, eager_vh)] {
        assert_eq!(lazy.codomain(), eager.codomain());
        assert_eq!(lazy.domain(), eager.domain());
        assert!(lazy
            .materialize(engine)
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .zip(eager.materialize(engine).unwrap().dense_data().unwrap())
            .all(|(actual, expected)| (*actual - *expected).norm() < 1.0e-10));
    }
}

#[cfg(feature = "cgc-gen")]
#[test]
fn sun_checked_generic_svd_vals_matches_compact_spectrum() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    use symtn::sector::LieGroupFusionRule;

    let runtime = Runtime::builder().build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |trees, _| {
            trees.coupled().iter().sum::<i64>() as f64 + 1.0
        })
        .unwrap();
    let spectra = source.svd_vals(engine, &[0], &[1]).unwrap();
    assert!(!spectra.is_empty());
    assert!(spectra.iter().all(|spectrum| spectrum
        .values
        .iter()
        .all(|value| value.is_finite() && *value >= 0.0)));

    let complex = source.convert::<Complex64>(engine).unwrap();
    let complex_spectra = complex.svd_vals(engine, &[0], &[1]).unwrap();
    assert_eq!(complex_spectra, spectra);
}

#[test]
fn checked_compact_svd_vals_preserves_leg_roles_and_decode_failure() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    use symtn::typed::{CheckedGenericPlanError, OperationError};

    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 2), (Label::X, 3)]).unwrap();
    let compact: TensorMap<_, f64> = TensorMap::diagonal(
        engine,
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
    let expected = compact.materialize(engine).unwrap();
    assert_eq!(
        compact.svd_vals(engine, &[0], &[1]).unwrap(),
        expected.svd_vals(engine, &[0], &[1]).unwrap()
    );
    // Changing the leg roles takes the checked transform boundary before SVD.
    assert!(matches!(
        compact.svd_vals(engine, &[1], &[0]),
        Err(GenericTensorError::Plan(
            CheckedGenericPlanError::Operation(OperationError::EmptyTransformBlock)
        ))
    ));

    provider.fail_decode.store(true, Ordering::Relaxed);
    assert!(matches!(
        compact.svd_vals(engine, &[0], &[1]),
        Err(GenericTensorError::Plan(CheckedGenericPlanError::Provider(
            _
        )))
    ));
    provider.fail_decode.store(false, Ordering::Relaxed);
}

#[test]
fn checked_generic_svd_truncation_reconstructs_and_preserves_provider() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, indices| {
            if indices[0] == indices[1] {
                (indices[0] + 2) as f64
            } else {
                0.0
            }
        })
        .unwrap();
    // The truncated SVD is a composition (#1534).
    let Svd { u, s, vh } = source.svd_compact(engine, &[0], &[1]).unwrap();
    let found = s.domain()[0]
        .find_truncated(&s.diagview(engine).unwrap(), &Truncation::rank(1))
        .unwrap();
    let u = u
        .restrict_leg(engine, &[(u.codomain_rank(), &found.selection)])
        .unwrap();
    let s = s
        .restrict_leg(engine, &[(0, &found.selection), (1, &found.selection)])
        .unwrap();
    let vh = vh.restrict_leg(engine, &[(0, &found.selection)]).unwrap();
    assert!(std::ptr::eq(u.provider(), provider.as_ref()));
    assert!(std::ptr::eq(s.provider(), provider.as_ref()));
    assert!(std::ptr::eq(vh.provider(), provider.as_ref()));
    let rebuilt = u.compose(engine, &s).unwrap().compose(engine, &vh).unwrap();
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .all(|value| value.is_finite()));
    assert!(s.diagview(engine).unwrap().iter().all(|spectrum| {
        spectrum.values.len() <= 2 && spectrum.values.iter().all(|value| value.is_finite())
    }));
}

#[test]
fn checked_generic_compact_svd_failure_is_typed_and_nonpublishing() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let _cache = cache_exclusive();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(122));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg], [&leg], |_, _| 2.0).unwrap();
    let before = source.dense_data().unwrap().to_vec();
    // The construction published the layouts this operation walks; a
    // provider whose answers change is not one identity, so the cached
    // layouts go first (#2030). The tag is this test's alone.
    forget_cached_structures();
    provider.fail_algebra.store(true, Ordering::Relaxed);
    let error = source.svd_compact(engine, &[0, 1], &[2]).unwrap_err();
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
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, ij| {
            if ij[0] == ij[1] {
                2.0
            } else {
                0.25
            }
        })
        .unwrap();
    let before = source.dense_data().unwrap().to_vec();
    provider.fail_algebra.store(true, Ordering::Relaxed);
    let error = source.svd_full(engine, &[0], &[1]).unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Plan(symtn::typed::CheckedGenericPlanError::Provider(
            ToyError::Algebra
        ))
    ));
    assert_eq!(source.dense_data().unwrap(), before.as_slice());
}

/// QR/LQ/SVD and the value-only spectra of a facade-layout source (#1494),
/// each against an oracle that does not share the factorization's layout:
/// reconstruction, isometry against an explicit tree-basis identity, `S`
/// against `svd_vals`, and `eigh_vals`/`eig_vals` of `A Aᴴ` against the
/// squared singular values.
#[cfg(feature = "cgc-gen")]
macro_rules! assert_sun_compact_laws {
    ($engine:ident, $runtime:expr, $source:expr) => {{
        let source = $source;
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
        let owned_adjoint = |$engine: &mut dyn Engine, tensor: &TensorMap<_, _>| {
            tensor
                .adjoint($engine)
                .unwrap()
                .materialize($engine)
                .unwrap()
        };
        let assert_identity = |$engine: &mut dyn Engine, gram: &TensorMap<_, _>, what: &str| {
            let (codomain, domain) = (gram.codomain(), gram.domain());
            let rank = codomain.iter().count();
            let identity = TensorMap::from_subblock_fn(
                $engine,
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
            assert_close($engine, gram, &identity, what);
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
            .qr_compact($engine, &codomain_axes(&source), &domain_axes(&source))
            .unwrap();
        let q_compose = &q.compose($engine, &r).unwrap();
        assert_close($engine, q_compose, source, "A = QR");
        let operand = &owned_adjoint($engine, &q).compose($engine, &q).unwrap();
        assert_identity($engine, operand, "Qᴴ Q = 1");
        let Lq { l, q } = source
            .lq_compact($engine, &codomain_axes(&source), &domain_axes(&source))
            .unwrap();
        let l_compose = &l.compose($engine, &q).unwrap();
        assert_close($engine, l_compose, source, "A = LQ");
        let operand = &owned_adjoint($engine, &q);
        let q_compose = q.compose($engine, operand).unwrap();
        let q_compose = &q_compose;
        assert_identity($engine, q_compose, "Q Qᴴ = 1");
        let Svd { u, s, vh } = source
            .svd_compact($engine, &codomain_axes(&source), &domain_axes(&source))
            .unwrap();
        let u_compose = &u
            .compose($engine, &s)
            .unwrap()
            .compose($engine, &vh)
            .unwrap();
        assert_close($engine, u_compose, source, "A = U S Vh");
        let operand = &owned_adjoint($engine, &u).compose($engine, &u).unwrap();
        assert_identity($engine, operand, "Uᴴ U = 1");
        let owned_adjoint_out = owned_adjoint($engine, &vh);
        let vh_compose = vh.compose($engine, &owned_adjoint_out).unwrap();
        assert_identity($engine, &vh_compose, "Vh Vhᴴ = 1");

        let singular = by_sector(
            source
                .svd_vals($engine, &codomain_axes(&source), &domain_axes(&source))
                .unwrap(),
        );
        let diagonal = by_sector(
            s.eigh_vals($engine, &[0], &[1], HermitianTol::DEFAULT)
                .unwrap(),
        );
        assert_eq!(singular.len(), diagonal.len());
        for (sector, values) in &singular {
            assert_eq!(find(&diagonal, sector).len(), values.len());
            assert_values(&find(&diagonal, sector), values, "S = svd_vals");
        }
        let owned_adjoint_out = owned_adjoint($engine, source);
        let gram = source.compose($engine, &owned_adjoint_out).unwrap();
        let squared = singular
            .iter()
            .map(|(sector, values)| (sector.clone(), values.iter().map(|v| v * v).collect()))
            .collect::<Vec<_>>();
        for (sector, values) in by_sector(
            gram.eigh_vals(
                $engine,
                &codomain_axes(&gram),
                &domain_axes(&gram),
                HermitianTol::DEFAULT,
            )
            .unwrap(),
        ) {
            assert_values(&values, &find(&squared, &sector), "eigh_vals(A Aᴴ) = s²");
        }
        for spectrum in gram
            .eig_vals($engine, &codomain_axes(&gram), &domain_axes(&gram))
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
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    use symtn::sector::LieGroupFusionRule;

    let runtime = Runtime::builder().build().unwrap();
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
        assert_sun_compact_laws!(engine, &runtime, &real);
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
        assert_sun_compact_laws!(engine, &runtime, &complex);
    }
}
