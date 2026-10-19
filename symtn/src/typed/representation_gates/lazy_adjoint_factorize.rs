use super::*;

/// Destination QR only (`qr` must not be reached); the second call fails.
fn fail_second_qr() -> SpyExecutor {
    SpyExecutor::default()
        .only(&[Kernel::QrInto], "QR must use the destination API")
        .failing(&[Kernel::QrInto], Some(2), "injected second-sector failure")
}

#[test]
fn svd_vals_reads_the_parent_without_materializing_the_adjoint() {
    // What: values-only SVD preserves typed sector spectra across cold,
    // repeated, cloned, and concurrent lazy-adjoint reads.
    macro_rules! assert_fixture {
        ($source:expr) => {{
            let source = $source;
            let expected = source.svd_vals(&[0, 1], &[2]).unwrap();
            let lazy = source.adjoint().unwrap();
            assert_eq!(lazy.svd_vals(&[0], &[1, 2]).unwrap(), expected);
            assert_eq!(lazy.svd_vals(&[0], &[1, 2]).unwrap(), expected);
            assert_eq!(lazy.clone().svd_vals(&[0], &[1, 2]).unwrap(), expected);
            let TypedTensorRepr::Adjoint(view) = &lazy.repr else {
                unreachable!()
            };
            assert!(Arc::ptr_eq(
                view.logical_space.provider_arc(),
                source.logical_space().provider_arc()
            ));
        }};
    }
    assert_fixture!(u1_lazy_fixture());
    assert_fixture!(u1_lazy_fixture().convert::<Complex64>());
    assert_fixture!(su2_lazy_fixture());
    assert_fixture!(su2_lazy_fixture().convert::<Complex64>());

    let source = u1_lazy_fixture().convert::<Complex64>();
    let expected = source.svd_vals(&[0, 1], &[2]).unwrap();
    let lazy = source.adjoint().unwrap();
    let threads: Vec<_> = (0..4)
        .map(|_| {
            let lazy = lazy.clone();
            std::thread::spawn(move || lazy.svd_vals(&[0], &[1, 2]).unwrap())
        })
        .collect();
    for thread in threads {
        assert_eq!(thread.join().unwrap(), expected);
    }
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}

fn assert_compact_svd_reads_parent<R, D>(source: &TensorMap<R, D>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: FactorizationScalar + core::fmt::Debug,
{
    let eager = eager_adjoint_oracle(source);
    let lazy = source.adjoint().unwrap();
    let actual = lazy.svd_compact(&[0], &[1, 2]).unwrap();
    let expected = eager.svd_compact(&[0], &[1, 2]).unwrap();

    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
    for (actual, expected) in [
        (&actual.u, &expected.u),
        (&actual.s, &expected.s),
        (&actual.vh, &expected.vh),
    ] {
        assert_eq!(
            actual.logical_space().space(),
            expected.logical_space().space()
        );
        assert!(Arc::ptr_eq(
            actual.logical_space().provider_arc(),
            source.logical_space().provider_arc()
        ));
        assert!(actual
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .zip(expected.materialize().unwrap().dense_data().unwrap())
            .all(|(&left, &right)| {
                (left.widen_complex() - right.widen_complex()).norm() < 1e-12
            }));
    }
    assert!(is_isometric!(actual.u, 1e-12));
    let rebuilt = actual
        .u
        .compose(&actual.s)
        .unwrap()
        .compose(&actual.vh)
        .unwrap();
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(eager.dense_data().unwrap())
        .all(|(&left, &right)| { (left.widen_complex() - right.widen_complex()).norm() < 1e-12 }));
}

#[test]
fn compact_svd_reads_the_parent_without_materializing_the_adjoint() {
    // What: typed compact factors keep the eager logical-adjoint semantics,
    // provider authority, final gauge, and reconstruction without an input copy.
    let u1 = u1_lazy_fixture();
    let su2 = su2_lazy_fixture();
    assert_compact_svd_reads_parent(&u1);
    assert_compact_svd_reads_parent(&genuinely_complex(&u1));
    assert_compact_svd_reads_parent(&su2);
    assert_compact_svd_reads_parent(&genuinely_complex(&su2));
}

fn assert_full_svd_reads_parent<R, D>(source: &TensorMap<R, D>, compare_factor_bytes: bool)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: FactorizationScalar + core::fmt::Debug,
{
    let eager = eager_adjoint_oracle(source);
    let lazy = source.adjoint().unwrap();
    let actual = lazy
        .svd_full(&codomain_axes(&lazy), &domain_axes(&lazy))
        .unwrap();
    let expected = eager
        .svd_full(&codomain_axes(&eager), &domain_axes(&eager))
        .unwrap();

    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
    for (actual, expected) in [
        (&actual.u, &expected.u),
        (&actual.s, &expected.s),
        (&actual.vh, &expected.vh),
    ] {
        assert_eq!(
            actual.logical_space().space(),
            expected.logical_space().space()
        );
        assert!(Arc::ptr_eq(
            actual.logical_space().provider_arc(),
            source.logical_space().provider_arc()
        ));
    }
    assert!(actual
        .s
        .dense_data()
        .unwrap()
        .iter()
        .zip(expected.s.dense_data().unwrap())
        .all(|(&left, &right)| { (left.widen_complex() - right.widen_complex()).norm() < 1e-12 }));
    if compare_factor_bytes {
        for (actual, expected) in [(&actual.u, &expected.u), (&actual.vh, &expected.vh)] {
            assert!(actual
                .dense_data()
                .unwrap()
                .iter()
                .zip(expected.dense_data().unwrap())
                .all(|(&left, &right)| {
                    (left.widen_complex() - right.widen_complex()).norm() < 1e-12
                }));
        }
    }
    assert!(is_isometric!(actual.u, 1e-12));
    assert!(is_isometric!(actual.vh, 1e-12));
    let rebuilt = actual
        .u
        .compose(&actual.s)
        .unwrap()
        .compose(&actual.vh)
        .unwrap();
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(eager.dense_data().unwrap())
        .all(|(&left, &right)| { (left.widen_complex() - right.widen_complex()).norm() < 1e-12 }));
}

#[test]
fn full_svd_adjoint_rectangular_matched_matches_materialized_oracle() {
    let matched = u1_matrix_fixture([(0, 2)], [(0, 3)]);
    assert_full_svd_reads_parent(&matched, true);
    assert_full_svd_reads_parent(&genuinely_complex(&matched), true);
}

#[test]
fn full_svd_adjoint_unmatched_row_only_matches_materialized_oracle() {
    let source = u1_matrix_fixture([(0, 2), (1, 1)], [(0, 3)]);
    assert_full_svd_reads_parent(&source, false);
}

#[test]
fn full_svd_adjoint_unmatched_column_only_matches_materialized_oracle() {
    let source = u1_matrix_fixture([(0, 2)], [(0, 3), (1, 1)]);
    assert_full_svd_reads_parent(&source, false);
}

#[test]
fn full_svd_adjoint_disjoint_matches_materialized_oracle() {
    let source = u1_matrix_fixture([(1, 2)], [(0, 3)]);
    assert_full_svd_reads_parent(&source, false);
}

#[test]
fn full_svd_adjoint_multitree_matches_materialized_oracle() {
    let multitree = su2_lazy_fixture();
    assert_full_svd_reads_parent(&multitree, false);
    assert_full_svd_reads_parent(&genuinely_complex(&multitree), false);
}

#[test]
fn full_svd_late_failure_does_not_publish_the_adjoint_cache() {
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(fail_second_svd(&Arc::default())))
        .build()
        .unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 2), (U1Irrep::new(1), 2)]).unwrap();
    let source = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        (indices.iter().sum::<usize>() + 1) as f64
    })
    .unwrap();
    let before = source.dense_data().unwrap().to_vec();
    let lazy = source.adjoint().unwrap();

    assert!(matches!(
        lazy.svd_full(&[0], &[1]),
        Err(Error::Operation(_))
    ));
    assert_eq!(source.dense_data().unwrap(), before);
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}

fn assert_null_redirect<R, D>(source: &TensorMap<R, D>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: FactorizationScalar + core::fmt::Debug,
{
    let target = eager_adjoint_oracle(source);
    let lazy = source.adjoint().unwrap();
    for (actual, expected, left) in [
        (
            lazy.left_null(&codomain_axes(&lazy), &domain_axes(&lazy))
                .unwrap(),
            target
                .left_null(&codomain_axes(&target), &domain_axes(&target))
                .unwrap(),
            true,
        ),
        (
            lazy.right_null(&codomain_axes(&lazy), &domain_axes(&lazy))
                .unwrap(),
            target
                .right_null(&codomain_axes(&target), &domain_axes(&target))
                .unwrap(),
            false,
        ),
    ] {
        assert!(actual.owned_body().is_some());
        assert_eq!(
            actual.logical_space().space(),
            expected.logical_space().space()
        );
        assert!(Arc::ptr_eq(
            actual.logical_space().provider_arc(),
            source.logical_space().provider_arc()
        ));
        let actual_projector = if left {
            actual.compose(&actual.adjoint().unwrap()).unwrap()
        } else {
            actual.adjoint().unwrap().compose(&actual).unwrap()
        };
        let expected_projector = if left {
            expected.compose(&expected.adjoint().unwrap()).unwrap()
        } else {
            expected.adjoint().unwrap().compose(&expected).unwrap()
        };
        assert!(actual_projector
            .dense_data()
            .unwrap()
            .iter()
            .zip(expected_projector.dense_data().unwrap())
            .all(|(&actual, &expected)| {
                (actual.widen_complex() - expected.widen_complex()).norm() < 1e-11
            }));
        let residual = if left {
            actual.adjoint().unwrap().compose(&target).unwrap()
        } else {
            target.compose(&actual.adjoint().unwrap()).unwrap()
        };
        assert!(residual.norm(2.0).unwrap() < 1e-10 * (1.0 + target.norm(2.0).unwrap()));
        assert!(if left {
            is_isometric!(actual, 1e-11)
        } else {
            is_isometric!(actual.adjoint().unwrap(), 1e-11)
        });
        let _ = actual.dense_data().unwrap();
    }
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}

#[test]
fn null_spaces_redirect_through_the_parent_without_materializing_the_adjoint() {
    let fixtures = [
        u1_matrix_fixture([(0, 3)], [(0, 2)]),
        u1_matrix_fixture([(0, 3), (1, 2)], [(0, 2)]),
        u1_matrix_fixture([(0, 3)], [(0, 2), (1, 2)]),
        u1_matrix_fixture([(1, 2)], [(0, 3)]),
        u1_matrix_fixture([(0, 3)], [(0, 3)]),
    ];
    for source in &fixtures {
        assert_null_redirect(source);
        assert_null_redirect(&genuinely_complex(source));
    }
    let multitree = su2_lazy_fixture();
    assert_null_redirect(&multitree);
    assert_null_redirect(&genuinely_complex(&multitree));
}

fn assert_null_late_failure(left: bool) {
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(fail_second_qr()))
        .build()
        .unwrap();
    let provider = Arc::new(U1FusionRule);
    let short = GradedSpace::try_new(
        Arc::clone(&provider),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    let long =
        GradedSpace::try_new(provider, [(U1Irrep::new(0), 3), (U1Irrep::new(1), 3)]).unwrap();
    // The adjoint's null space is the parent's opposite one, so the parent is
    // long on that side and both sectors run a QR.
    let (codomain, domain) = if left {
        (&short, &long)
    } else {
        (&long, &short)
    };
    let source = TensorMap::from_subblock_fn(&runtime, [codomain], [domain], |_, indices| {
        (indices.iter().sum::<usize>() + 1) as f64
    })
    .unwrap();
    let before = source.dense_data().unwrap().to_vec();
    let lazy = source.adjoint().unwrap();

    let result = if left {
        lazy.left_null(&[0], &[1])
    } else {
        lazy.right_null(&[0], &[1])
    };
    assert!(matches!(result, Err(Error::Operation(_))));
    assert_eq!(source.dense_data().unwrap(), before);
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}

#[test]
fn null_space_late_failure_leaves_parent_and_adjoint_cache_unchanged() {
    assert_null_late_failure(true);
    assert_null_late_failure(false);
}

fn assert_polar_redirect<R, D>(source: &TensorMap<R, D>, left: bool)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: FactorizationScalar + core::fmt::Debug,
{
    let target = eager_adjoint_oracle(source);
    let lazy = source.adjoint().unwrap();
    let actual = if left {
        lazy.left_polar(&[0], &[1]).unwrap().pair()
    } else {
        lazy.right_polar(&codomain_axes(&lazy), &domain_axes(&lazy))
            .unwrap()
            .pair()
    };
    let expected = if left {
        target.left_polar(&[0], &[1]).unwrap().pair()
    } else {
        target
            .right_polar(&codomain_axes(&target), &domain_axes(&target))
            .unwrap()
            .pair()
    };
    assert_polar_factors(source, &target, &actual, &expected, left);
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}

fn assert_rank_deficient_polar_support<R, D>(source: &TensorMap<R, D>, left: bool)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: AdvancedLinalgScalar + core::fmt::Debug,
{
    let target = eager_adjoint_oracle(source);
    let target_pinv = target.pinv(&[0], &[1], 1e-10).unwrap();
    let target_codomain = target.compose(&target_pinv).unwrap();
    let target_domain = target_pinv.compose(&target).unwrap();
    let lazy = source.adjoint().unwrap();
    let factors = if left {
        lazy.left_polar(&[0], &[1]).unwrap().pair()
    } else {
        lazy.right_polar(&[0], &[1]).unwrap().pair()
    };
    let (positive, isometry) = if left {
        (&factors.1, &factors.0)
    } else {
        (&factors.0, &factors.1)
    };
    let positive_pinv = positive.pinv(&[0], &[1], 1e-10).unwrap();
    if left {
        let support = positive_pinv.compose(positive).unwrap();
        assert_typed_map_close(&support, &target_domain, 1e-9);
        let image = isometry
            .compose(&support)
            .unwrap()
            .compose(&isometry.adjoint().unwrap())
            .unwrap();
        assert_typed_map_close(&image, &target_codomain, 1e-9);
    } else {
        let support = positive.compose(&positive_pinv).unwrap();
        assert_typed_map_close(&support, &target_codomain, 1e-9);
        let image = isometry
            .adjoint()
            .unwrap()
            .compose(&support)
            .unwrap()
            .compose(isometry)
            .unwrap();
        assert_typed_map_close(&image, &target_domain, 1e-9);
    }
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}

#[test]
fn polar_redirects_through_parent_with_owned_psd_factors_and_a_cold_receiver() {
    let tall = u1_matrix_fixture([(0, 3)], [(0, 2)]);
    assert_polar_redirect(&tall, false);
    assert_polar_redirect(&genuinely_complex(&tall), false);

    let wide = u1_matrix_fixture([(0, 2)], [(0, 3)]);
    assert_polar_redirect(&wide, true);

    let codomain_only = u1_matrix_fixture([(0, 2), (1, 2)], [(0, 2)]);
    assert_polar_redirect(&codomain_only, false);
    let domain_only = u1_matrix_fixture([(0, 2)], [(0, 2), (1, 2)]);
    assert_polar_redirect(&domain_only, true);

    let provider = Arc::new(SU2FusionRule);
    let half = GradedSpace::try_new(provider, [(SU2Irrep::from_twice_spin(1), 1)]).unwrap();
    let multitree = TensorMap::from_subblock_fn(
        &Runtime::builder().dense_threads(1).build().unwrap(),
        [&half, &half, &half],
        [&half],
        |_, indices| (indices.iter().sum::<usize>() + 1) as f64,
    )
    .unwrap();
    assert_eq!(
        multitree.logical_space().space().structure().block_count(),
        2
    );
    let multitree = genuinely_complex(&multitree);
    assert_polar_redirect(&multitree, false);

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 3)]).unwrap();
    let rank_deficient = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        let value = ((indices[0] + 1) * (indices[1] + 1)) as f64;
        num_complex::Complex64::new(value, value / 3.0)
    })
    .unwrap();
    assert_polar_redirect(&rank_deficient, true);
    assert_polar_redirect(&rank_deficient, false);
    assert_rank_deficient_polar_support(&rank_deficient, true);
    assert_rank_deficient_polar_support(&rank_deficient, false);
}

#[test]
fn polar_redirect_repeats_clones_and_runs_concurrently_without_warming_receiver() {
    let source = genuinely_complex(&u1_matrix_fixture([(0, 3)], [(0, 3)]));
    let target = eager_adjoint_oracle(&source);
    let lazy = source.adjoint().unwrap();
    for left in [true, false] {
        let expected = if left {
            target.left_polar(&[0], &[1]).unwrap().pair()
        } else {
            target.right_polar(&[0], &[1]).unwrap().pair()
        };
        for _ in 0..2 {
            let actual = if left {
                lazy.clone().left_polar(&[0], &[1]).unwrap().pair()
            } else {
                lazy.clone().right_polar(&[0], &[1]).unwrap().pair()
            };
            assert_polar_factors(&source, &target, &actual, &expected, left);
        }
        let calls = (0..4)
            .map(|_| {
                let clone = lazy.clone();
                std::thread::spawn(move || {
                    if left {
                        clone.left_polar(&[0], &[1]).unwrap().pair()
                    } else {
                        clone.right_polar(&[0], &[1]).unwrap().pair()
                    }
                })
            })
            .collect::<Vec<_>>();
        for call in calls {
            let actual = call.join().unwrap();
            assert_polar_factors(&source, &target, &actual, &expected, left);
        }
    }
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}

#[test]
fn polar_redirect_wrong_direction_keeps_requested_name_and_receiver_cold() {
    let source = u1_matrix_fixture([(0, 3)], [(0, 2)]);
    let lazy = source.adjoint().unwrap();
    let error = lazy.left_polar(&[0], &[1]).unwrap_err();
    assert!(matches!(
        error,
        Error::Operation(error)
            if matches!(
                error.as_ref(),
                symtn_tensors::OperationError::InvalidArgument { message }
                    if *message == "left_polar requires rows >= columns in every coupled-sector matrix"
            )
    ));

    let source = u1_matrix_fixture([(0, 2)], [(0, 3)]);
    let lazy = source.adjoint().unwrap();
    let error = lazy.right_polar(&[0], &[1]).unwrap_err();
    assert!(matches!(
        error,
        Error::Operation(error)
            if matches!(
                error.as_ref(),
                symtn_tensors::OperationError::InvalidArgument { message }
                    if *message == "right_polar requires columns >= rows in every coupled-sector matrix"
            )
    ));
}

#[test]
fn polar_redirect_late_failure_leaves_parent_and_receiver_unchanged() {
    for left in [true, false] {
        // Sectors stream, so the first sector's W/P GEMMs run before the
        // second SVD fails.
        let runtime = Runtime::builder()
            .with_dense_executor(Box::new(SpyExecutor::counting(&Arc::default()).failing(
                POLAR_SVD,
                Some(2),
                "injected second-sector failure",
            )))
            .build()
            .unwrap();
        let provider = Arc::new(U1FusionRule);
        let leg =
            GradedSpace::try_new(provider, [(U1Irrep::new(0), 2), (U1Irrep::new(1), 2)]).unwrap();
        let source = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
            (indices.iter().sum::<usize>() + 1) as f64
        })
        .unwrap();
        let before = source.dense_data().unwrap().to_vec();
        let lazy = source.adjoint().unwrap();
        let result = if left {
            lazy.left_polar(&[0], &[1]).map(drop)
        } else {
            lazy.right_polar(&[0], &[1]).map(drop)
        };
        assert!(matches!(result, Err(Error::Operation(_))));
        assert_eq!(source.dense_data().unwrap(), before);
        let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
            unreachable!()
        };
    }
}

fn assert_qr_lq_factors<R, D>(
    source: &TensorMap<R, D>,
    target: &TensorMap<R, D>,
    actual: &(TensorMap<R, D>, TensorMap<R, D>),
    expected: &(TensorMap<R, D>, TensorMap<R, D>),
    qr: bool,
    compare_gauge: bool,
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar + core::fmt::Debug,
{
    for (actual, expected) in [(&actual.0, &expected.0), (&actual.1, &expected.1)] {
        assert!(actual.owned_body().is_some());
        assert_eq!(
            actual.logical_space().space(),
            expected.logical_space().space()
        );
        assert!(Arc::ptr_eq(
            actual.logical_space().provider_arc(),
            source.logical_space().provider_arc()
        ));
        if compare_gauge {
            assert!(actual
                .dense_data()
                .unwrap()
                .iter()
                .zip(expected.dense_data().unwrap())
                .all(|(&left, &right)| {
                    (left.widen_complex() - right.widen_complex()).norm() < 1e-12
                }));
        }
    }
    let isometry = if qr {
        is_isometric!(actual.0, 1e-12)
    } else {
        is_isometric!(actual.1.adjoint().unwrap(), 1e-12)
    };
    assert!(isometry);
    let rebuilt = actual.0.compose(&actual.1).unwrap();
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(target.dense_data().unwrap())
        .all(|(&left, &right)| { (left.widen_complex() - right.widen_complex()).norm() < 1e-12 }));
}

fn assert_qr_lq_keeps_input_cache_cold<R, D>(source: &TensorMap<R, D>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: FactorizationScalar + core::fmt::Debug,
{
    let target = eager_adjoint_oracle(source);
    let lazy = source.adjoint().unwrap();

    let actual = lazy
        .qr_compact(&codomain_axes(&lazy), &domain_axes(&lazy))
        .unwrap()
        .pair();
    assert_qr_lq_factors(
        source,
        &target,
        &actual,
        &target
            .qr_compact(&codomain_axes(&target), &domain_axes(&target))
            .unwrap()
            .pair(),
        true,
        true,
    );
    let actual = lazy
        .lq_compact(&codomain_axes(&lazy), &domain_axes(&lazy))
        .unwrap()
        .pair();
    assert_qr_lq_factors(
        source,
        &target,
        &actual,
        &target
            .lq_compact(&codomain_axes(&target), &domain_axes(&target))
            .unwrap()
            .pair(),
        false,
        true,
    );
    let actual = lazy
        .qr_full(&codomain_axes(&lazy), &domain_axes(&lazy))
        .unwrap()
        .pair();
    assert_qr_lq_factors(
        source,
        &target,
        &actual,
        &target
            .qr_full(&codomain_axes(&target), &domain_axes(&target))
            .unwrap()
            .pair(),
        true,
        false,
    );
    let actual = lazy
        .lq_full(&codomain_axes(&lazy), &domain_axes(&lazy))
        .unwrap()
        .pair();
    assert_qr_lq_factors(
        source,
        &target,
        &actual,
        &target
            .lq_full(&codomain_axes(&target), &domain_axes(&target))
            .unwrap()
            .pair(),
        false,
        false,
    );

    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}

#[test]
fn qr_lq_adjoint_dispatch_covers_unique_simple_dtypes_and_rectangles() {
    // What: adjoint QR uses an operation-local logical copy while LQ uses
    // the parent QR, preserving compact gauge, full semantics, and provider authority.
    let tall = u1_matrix_fixture([(-2, 1), (0, 3)], [(-2, 1), (0, 1)]);
    let wide = u1_matrix_fixture([(-2, 1), (0, 1)], [(-2, 1), (0, 3)]);
    assert_qr_lq_keeps_input_cache_cold(&tall);
    assert_qr_lq_keeps_input_cache_cold(&genuinely_complex(&tall));
    assert_qr_lq_keeps_input_cache_cold(&wide);
    assert_qr_lq_keeps_input_cache_cold(&genuinely_complex(&wide));

    let multitree = su2_lazy_fixture();
    assert_qr_lq_keeps_input_cache_cold(&multitree);
    assert_qr_lq_keeps_input_cache_cold(&genuinely_complex(&multitree));
}

#[test]
fn full_qr_lq_adjoint_dispatch_handles_unmatched_and_disjoint_sectors() {
    for source in [
        u1_matrix_fixture([(0, 2), (1, 1)], [(0, 3)]),
        u1_matrix_fixture([(0, 2)], [(0, 3), (1, 1)]),
    ] {
        let target = eager_adjoint_oracle(&source);
        let lazy = source.adjoint().unwrap();
        let qr = lazy.qr_full(&[0], &[1]).unwrap().pair();
        assert_qr_lq_factors(
            &source,
            &target,
            &qr,
            &target.qr_full(&[0], &[1]).unwrap().pair(),
            true,
            false,
        );
        let lq = lazy.lq_full(&[0], &[1]).unwrap().pair();
        assert_qr_lq_factors(
            &source,
            &target,
            &lq,
            &target.lq_full(&[0], &[1]).unwrap().pair(),
            false,
            false,
        );
        let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
            unreachable!()
        };
    }
}

#[test]
fn qr_lq_adjoint_dispatch_handles_an_empty_homspace() {
    let source = u1_matrix_fixture([(1, 2)], [(0, 3)]);
    assert!(source.dense_data().unwrap().is_empty());
    assert_qr_lq_keeps_input_cache_cold(&source);
}

#[test]
fn qr_lq_uncached_owned_outputs_repeat_clone_and_run_concurrently() {
    let source = genuinely_complex(&su2_lazy_fixture());
    let target = eager_adjoint_oracle(&source);
    let lazy = source.adjoint().unwrap();
    let expected_qr = lazy.qr_compact(&[0], &[1, 2]).unwrap().pair();
    let expected_lq = lazy.lq_full(&[0], &[1, 2]).unwrap().pair();
    for _ in 0..2 {
        let qr = lazy.clone().qr_compact(&[0], &[1, 2]).unwrap().pair();
        let lq = lazy.clone().lq_full(&[0], &[1, 2]).unwrap().pair();
        assert_qr_lq_factors(&source, &target, &qr, &expected_qr, true, true);
        assert_qr_lq_factors(&source, &target, &lq, &expected_lq, false, false);
    }
    std::thread::scope(|scope| {
        let calls: Vec<_> = (0..4)
            .map(|_| {
                let lazy = lazy.clone();
                scope.spawn(move || {
                    let qr = lazy.qr_compact(&[0], &[1, 2]).unwrap().pair();
                    let lq = lazy.lq_full(&[0], &[1, 2]).unwrap().pair();
                    (qr, lq)
                })
            })
            .collect();
        for call in calls {
            let (qr, lq) = call.join().unwrap();
            assert_qr_lq_factors(&source, &target, &qr, &expected_qr, true, true);
            assert_qr_lq_factors(&source, &target, &lq, &expected_lq, false, false);
        }
    });
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}

fn assert_full_qr_lq_late_failure(qr: bool) {
    let runtime = Runtime::builder()
        .with_dense_executor(Box::new(fail_second_qr()))
        .build()
        .unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 2), (U1Irrep::new(1), 2)]).unwrap();
    let source = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        (indices.iter().sum::<usize>() + 1) as f64
    })
    .unwrap();
    let before = source.dense_data().unwrap().to_vec();
    let lazy = source.adjoint().unwrap();

    let result = if qr {
        lazy.qr_full(&[0], &[1]).map(drop)
    } else {
        lazy.lq_full(&[0], &[1]).map(drop)
    };
    assert!(matches!(result, Err(Error::Operation(_))));
    assert_eq!(source.dense_data().unwrap(), before);
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}

#[test]
fn full_qr_lq_late_failure_leaves_parent_and_adjoint_cache_unchanged() {
    assert_full_qr_lq_late_failure(true);
    assert_full_qr_lq_late_failure(false);
}

fn assert_truncated_svd_reads_parent<R, D>(source: &TensorMap<R, D>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: FactorizationScalar + SpectrumMagnitude + core::fmt::Debug,
{
    // The truncated SVD is the composition `svd_compact` -> `diagview` ->
    // `find_truncated` -> `restrict_*`.
    struct Truncated<R: SectorCodec, D> {
        u: TensorMap<R, D>,
        s: TensorMap<R, D>,
        vh: TensorMap<R, D>,
        singular_values: Vec<SectorSpectrum<R::Sector, f64>>,
        error: f64,
    }
    fn truncated<R, D>(tensor: &TensorMap<R, D>, truncation: &Truncation) -> Truncated<R, D>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
        D: FactorizationScalar + SpectrumMagnitude,
    {
        let Svd { u, s, vh } = tensor.svd_compact(&[0], &[1, 2]).unwrap();
        let found = s.domain()[0]
            .find_truncated(&s.diagview().unwrap(), truncation)
            .unwrap();
        let s = s
            .restrict_leg(&[(0, &found.selection), (1, &found.selection)])
            .unwrap();
        Truncated {
            u: u.restrict_leg(&[(u.codomain_rank(), &found.selection)])
                .unwrap(),
            singular_values: s
                .diagview()
                .unwrap()
                .into_iter()
                .map(|entry| SectorSpectrum {
                    sector: entry.sector,
                    values: entry.values.iter().map(|v| v.magnitude()).collect(),
                })
                .collect(),
            s,
            vh: vh.restrict_leg(&[(0, &found.selection)]).unwrap(),
            error: found.error,
        }
    }
    let eager = eager_adjoint_oracle(source);
    let lazy = source.adjoint().unwrap();
    let truncation = Truncation::rank(1);
    let actual = truncated(&lazy, &truncation);
    let expected = truncated(&eager, &truncation);

    // The lazy route factors the parent while the oracle factors the
    // materialized adjoint, so kept sectors and counts match exactly and
    // values under the tolerance rule; every payload entry can reach a
    // singular value, so `terms` is the payload length.
    let terms = source.dense_data().unwrap().len();
    assert_eq!(actual.singular_values.len(), expected.singular_values.len());
    for (actual, expected) in actual.singular_values.iter().zip(&expected.singular_values) {
        assert_eq!(actual.sector, expected.sector);
        crate::test_numerics::numerics::assert_slices_close(
            "kept singular values",
            &actual.values,
            &expected.values,
            terms,
        );
    }
    crate::test_numerics::numerics::assert_close(
        "truncation error",
        actual.error,
        expected.error,
        terms,
    );
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
    for (actual, expected) in [
        (&actual.u, &expected.u),
        (&actual.s, &expected.s),
        (&actual.vh, &expected.vh),
    ] {
        assert_eq!(
            actual.logical_space().space(),
            expected.logical_space().space()
        );
        assert!(Arc::ptr_eq(
            actual.logical_space().provider_arc(),
            source.logical_space().provider_arc()
        ));
        assert!(actual
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .zip(expected.materialize().unwrap().dense_data().unwrap())
            .all(|(&left, &right)| {
                (left.widen_complex() - right.widen_complex()).norm() < 1e-12
            }));
    }
    assert!(is_isometric!(actual.u, 1e-12));
}

#[test]
fn truncated_svd_reads_the_parent_without_materializing_the_adjoint() {
    // What: truncation selection, error, factor gauge, and typed provider
    // authority match an eager logical-adjoint oracle without an input copy.
    let u1 = u1_lazy_fixture();
    let su2 = su2_lazy_fixture();
    assert_truncated_svd_reads_parent(&u1);
    assert_truncated_svd_reads_parent(&genuinely_complex(&u1));
    assert_truncated_svd_reads_parent(&su2);
    assert_truncated_svd_reads_parent(&genuinely_complex(&su2));
}

#[test]
fn rejected_truncation_does_not_materialize_the_adjoint() {
    // What: the parent-native path preserves typed truncation errors
    // without publishing the logical-adjoint payload first.
    let source = u1_lazy_fixture();
    let foreign =
        GradedSpace::try_new(Arc::new(SU2FusionRule), [(SU2Irrep::from_twice_spin(0), 1)]).unwrap();
    let lazy = source.adjoint().unwrap();
    let Svd { s, .. } = lazy.svd_compact(&[0], &[1, 2]).unwrap();
    assert!(matches!(s.domain()[0]
        .find_truncated(
            &s.diagview().unwrap(),
            &Truncation::space(foreign.truncspace())
        ), Err(Error::Operation(e)) if matches!(*e, OperationError::InvalidArgument { message } if message.contains("different fusion rule"))));
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}
