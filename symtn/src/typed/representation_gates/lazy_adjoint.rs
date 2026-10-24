use super::*;

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_generic_lazy_transforms_do_not_materialize_uncached_input() {
    use symtn_core::LieGroupFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let fundamental = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 0], 2)]).unwrap();
    let antifundamental = GradedSpace::try_new(Arc::clone(&provider), [(vec![0, 1], 3)]).unwrap();
    let source: TensorMap<_, Complex64> = TensorMap::from_subblock_fn(
        &runtime,
        [&fundamental, &fundamental],
        [&antifundamental],
        |trees, indices| {
            Complex64::new(
                indices.iter().sum::<usize>() as f64,
                trees.coupled().iter().sum::<i64>() as f64 + 0.5,
            )
        },
    )
    .unwrap();
    let lazy = source.adjoint().unwrap();
    let eager = lazy.materialized_tensor_uncached().unwrap();
    let adjoint = probe::record();

    let outputs = [
        (
            lazy.permute(&[0, 2], &[1]).unwrap(),
            eager.permute(&[0, 2], &[1]).unwrap(),
        ),
        (
            lazy.braid(&[0, 2], &[1], &[0, 1, 2]).unwrap(),
            eager.braid(&[0, 2], &[1], &[0, 1, 2]).unwrap(),
        ),
        (lazy.repartition(2).unwrap(), eager.repartition(2).unwrap()),
        (
            lazy.transpose(&[2, 1], &[0]).unwrap(),
            eager.transpose(&[2, 1], &[0]).unwrap(),
        ),
        (
            lazy.transpose(&[0, 2], &[1]).unwrap(),
            eager.transpose(&[0, 2], &[1]).unwrap(),
        ),
    ];
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
    // Read before the checks below: a transformed adjoint stays lazy, and
    // they materialize it on purpose.
    assert_eq!(adjoint.get(Event::AdjointMaterialization), 0);
    for (actual, expected) in outputs {
        assert!(matches!(&actual.repr, TypedTensorRepr::Adjoint(_)));
        assert_eq!(
            actual.logical_space().space(),
            expected.logical_space().space()
        );
        assert!(actual.norm(2.0).unwrap().is_finite());
        assert!(actual.qr_compact(&[0, 1], &[2]).is_ok());
        let actual = actual.materialize().unwrap();
        assert_eq!(
            actual.dense_data().unwrap().len(),
            expected.dense_data().unwrap().len()
        );
        assert!(actual
            .dense_data()
            .unwrap()
            .iter()
            .zip(expected.dense_data().unwrap())
            .all(|(&actual, &expected)| (actual - expected).norm() < 1.0e-10));
    }
}

#[test]
fn coupled_block_reads_do_not_materialize_a_lazy_adjoint() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 3)],
    )
    .unwrap();
    let source: TensorMap<_, num_complex::Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg], |_, index| {
            num_complex::Complex64::new(index[0] as f64, index[2] as f64 + 1.0)
        })
        .unwrap();
    let lazy = source.adjoint().unwrap();
    let clone = lazy.clone();
    let mut entries = 0;
    for (sector, block) in lazy.blocks().unwrap() {
        let again = clone.block(&sector).unwrap();
        assert_eq!((again.rows(), again.cols()), (block.rows(), block.cols()));
        for row in 0..block.rows() {
            for col in 0..block.cols() {
                assert_eq!(block.get(row, col), again.get(row, col));
                entries += 1;
            }
        }
    }
    assert!(entries > 0);
}

#[test]
fn generic_lazy_adjoint_keeps_parent_storage_and_refuses_a_dense_borrow() {
    let source = u1_lazy_fixture();
    let parent: Arc<TypedTensorBody<_, _, NonCloneHost>> = Arc::new(TypedTensorBody::dense(
        source.logical_space().clone(),
        NonCloneHost(source.dense_data().unwrap().to_vec()),
    ));
    let logical_space = symtn_tensors::adjoint_bound_space_dyn(&parent.space).unwrap();
    let lazy = TensorMap {
        runtime: source.runtime.clone(),
        repr: TypedTensorRepr::Adjoint(Arc::new(TypedAdjointView::new(
            Arc::clone(&parent),
            logical_space,
        ))),
    };

    assert!(matches!(
        lazy.dense_data(),
        Err(Error::Unsupported {
            alternative: crate::error::Alternative::Materialize,
            ..
        })
    ));
    let TypedTensorRepr::Adjoint(view) = &lazy.repr else {
        unreachable!("fixture is a lazy adjoint")
    };
    assert!(Arc::ptr_eq(&parent, &view.parent));
}

fn assert_lazy_involution<R, D>(source: &TensorMap<R, D>)
where
    R: TypedSectorAdmission<
            Sector = <R as SectorCodec>::Sector,
            Error = FusionAlgebraError,
            Mode = MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: TensorScalar,
{
    let adjoint = source.adjoint().unwrap();
    let TypedTensorRepr::Adjoint(view) = &adjoint.repr else {
        panic!("dense adjoint must be lazy");
    };
    assert!(Arc::ptr_eq(&view.parent, owned(source)));
    assert!(Arc::ptr_eq(
        view.parent.space.provider_arc(),
        view.logical_space.provider_arc()
    ));
    assert!(std::ptr::eq(adjoint.provider(), source.provider()));

    let clone = adjoint.clone();
    let TypedTensorRepr::Adjoint(clone_view) = &clone.repr else {
        unreachable!()
    };
    assert!(Arc::ptr_eq(view, clone_view));

    let restored = adjoint.adjoint().unwrap();
    assert!(Arc::ptr_eq(owned(source), owned(&restored)));
    assert_eq!(
        source.dense_data().unwrap().as_ptr(),
        restored.dense_data().unwrap().as_ptr()
    );
}

#[test]
fn lazy_adjoint_representation_and_involution_cover_unique_simple_and_both_dtypes() {
    let u1_f64 = u1_lazy_fixture();
    let u1_c64 = u1_f64.convert::<Complex64>();
    let su2_f64 = su2_lazy_fixture();
    let su2_c64 = su2_f64.convert::<Complex64>();

    assert_lazy_involution(&u1_f64);
    assert_lazy_involution(&u1_c64);
    assert_lazy_involution(&su2_f64);
    assert_lazy_involution(&su2_c64);
}

#[test]
fn lazy_adjoint_metadata_is_logical_and_cold() {
    let source = u1_lazy_fixture();
    let adjoint = source.adjoint().unwrap();
    assert_eq!((source.codomain_rank(), source.domain_rank()), (2, 1));
    assert_eq!((adjoint.codomain_rank(), adjoint.domain_rank()), (1, 2));
    assert_eq!(adjoint.rank(), 3);
    let signature = |space: &GradedSpace<U1FusionRule>| {
        (
            space.sectors().unwrap(),
            space.degeneracies().to_vec(),
            space.is_dual(),
        )
    };
    let source_codomain = source.codomain();
    let source_domain = source.domain();
    let adjoint_codomain = adjoint.codomain();
    let adjoint_domain = adjoint.domain();
    assert_eq!(
        signature(&adjoint_codomain[0]),
        signature(&source_domain[0])
    );
    assert_eq!(
        signature(&adjoint_domain[0]),
        signature(&source_codomain[0])
    );
    assert_eq!(
        signature(&adjoint_domain[1]),
        signature(&source_codomain[1])
    );
    let source_dims = source.leg_dims().unwrap();
    assert_eq!(
        adjoint.leg_dims().unwrap(),
        [source_dims[2], source_dims[0], source_dims[1]]
    );
    assert_eq!(adjoint.subblock_count(), source.subblock_count());
    let expected = symtn_tensors::adjoint_bound_space_dyn(source.logical_space()).unwrap();
    for index in 0..adjoint.subblock_count() {
        let actual = adjoint.subblock(index).unwrap();
        let expected_block = expected.space().structure().block(index).unwrap();
        assert_eq!(actual.key(), expected_block.key());
        assert_eq!(actual.shape(), expected_block.shape());
        assert_eq!(actual.strides(), expected_block.strides());
        assert_eq!(actual.offset(), expected_block.offset());
        assert_eq!(
            adjoint.subblock_fusion_trees(index).unwrap(),
            decode_block_fusion_trees(adjoint.provider(), expected_block.key()).unwrap()
        );
    }
    assert_eq!(adjoint.logical_space().space(), expected.space());
    assert!(!format!("{adjoint:?}").is_empty());
    let TypedTensorRepr::Adjoint(_) = &adjoint.repr else {
        unreachable!()
    };
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_generic_inv_lazy_is_detached_and_keeps_receiver_cold() {
    use symtn_core::LieGroupFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, _| 2.0).unwrap();
    let lazy = source.adjoint().unwrap();
    let inverse = lazy.inv(&[0], &[1]).unwrap();

    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
    assert!(matches!(inverse.repr, TypedTensorRepr::Owned(_)));
    assert!(!std::ptr::eq(
        inverse.dense_data().unwrap().as_ptr(),
        source.dense_data().unwrap().as_ptr()
    ));
    assert!(inverse
        .dense_data()
        .unwrap()
        .iter()
        .all(|value| (*value - 0.5).abs() < 1.0e-12));
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_generic_null_lazy_redirects_are_owned_and_keep_receiver_cold() {
    // What: both null directions use the opposite operation on the owned
    // parent, detach the final adjoint, and never publish the lazy cache.
    use symtn_core::LieGroupFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, indices| {
            let row = indices[0] + 2 * indices[1];
            let column = indices[2] + 2 * indices[3];
            f64::from(row == column)
                * trees.codomain_vertices()[0].get() as f64
                * trees.domain_vertices()[0].get() as f64
        })
        .unwrap();
    let body = Arc::clone(owned(&source));
    let payload = Arc::clone(&body.data);
    let lazy = source.adjoint().unwrap();
    let expected_left = source
        .right_null(&[0, 1], &[2, 3])
        .unwrap()
        .adjoint()
        .unwrap()
        .materialized_tensor_uncached()
        .unwrap();
    let expected_right = source
        .left_null(&[0, 1], &[2, 3])
        .unwrap()
        .adjoint()
        .unwrap()
        .materialized_tensor_uncached()
        .unwrap();

    for (actual, expected) in [
        (lazy.left_null(&[0, 1], &[2, 3]).unwrap(), expected_left),
        (lazy.right_null(&[0, 1], &[2, 3]).unwrap(), expected_right),
    ] {
        assert!(matches!(&actual.repr, TypedTensorRepr::Owned(_)));
        assert_eq!(
            actual.logical_space().space(),
            expected.logical_space().space()
        );
        assert!(actual
            .dense_data()
            .unwrap()
            .iter()
            .zip(expected.dense_data().unwrap())
            .all(|(&actual, &expected)| (actual - expected).abs() < 1e-10));
        assert!(std::ptr::eq(actual.provider(), provider.as_ref()));
    }
    assert!(Arc::ptr_eq(owned(&source), &body));
    assert!(Arc::ptr_eq(&owned(&source).data, &payload));
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}

/// The shared rule table (#1755) in checked Generic: what each op
/// materializes for a lazy adjoint. `Parent` and `AdjointSeam` read the parent
/// in place (no materialization), `Materialize` copies the input once, and
/// `Redirect` detaches its adjointed output (one copy).
#[cfg(feature = "cgc-gen")]
#[test]
fn checked_generic_lazy_adjoint_factorizations_follow_the_rule_table() {
    use symtn_core::LieGroupFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg], |trees, indices| {
            trees.codomain_vertices()[0].get() as f64 + 0.25 * indices.iter().sum::<usize>() as f64
        })
        .unwrap();
    let lazy = source.adjoint().unwrap();
    let materializations = |op: &dyn Fn()| {
        let adjoint = probe::record();
        op();
        adjoint.get(Event::AdjointMaterialization)
    };

    assert_eq!(
        materializations(&|| drop(lazy.svd_vals(&[0], &[1, 2]).unwrap())),
        0
    );
    assert_eq!(
        materializations(&|| drop(lazy.svd_compact(&[0], &[1, 2]).unwrap())),
        0
    );
    assert_eq!(
        materializations(&|| drop(lazy.svd_full(&[0], &[1, 2]).unwrap())),
        0
    );
    assert_eq!(
        materializations(&|| drop(lazy.pinv(&[0], &[1, 2], 0.0).unwrap())),
        0
    );
    assert_eq!(
        materializations(&|| drop(lazy.qr_compact(&[0], &[1, 2]).unwrap())),
        1
    );
    assert_eq!(
        materializations(&|| drop(lazy.qr_full(&[0], &[1, 2]).unwrap())),
        1
    );
    assert_eq!(
        materializations(&|| drop(lazy.lq_compact(&[0], &[1, 2]).unwrap())),
        0
    );
    assert_eq!(
        materializations(&|| drop(lazy.lq_full(&[0], &[1, 2]).unwrap())),
        0
    );
    assert_eq!(
        materializations(&|| drop(lazy.left_null(&[0], &[1, 2]).unwrap())),
        1
    );
    assert_eq!(
        materializations(&|| drop(lazy.right_null(&[0], &[1, 2]).unwrap())),
        1
    );
    assert_eq!(
        materializations(&|| drop(lazy.right_polar(&[0], &[1, 2]).unwrap())),
        1
    );
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_generic_exp_lazy_is_owned_and_stays_cold() {
    use symtn_core::LieGroupFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, ij| f64::from(ij == [0, 1]))
            .unwrap();
    let lazy = source.adjoint().unwrap();
    let actual = lazy.exp(&[0], &[1]).unwrap();
    let expected = source.exp(&[0], &[1]).unwrap().adjoint().unwrap();
    assert!(matches!(actual.repr, TypedTensorRepr::Owned(_)));
    assert_eq!(
        actual.dense_data().unwrap(),
        expected.materialize().unwrap().dense_data().unwrap()
    );
    assert!(std::ptr::eq(actual.provider(), provider.as_ref()));
    assert!(actual.runtime().same_runtime(source.runtime()));
    assert_eq!(actual.codomain(), lazy.codomain());
    assert_eq!(actual.domain(), lazy.domain());
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_generic_lazy_flip_stays_cold_and_keeps_logical_duality() {
    use symtn_core::LieGroupFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![2, 2], 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg], |trees, _| {
            trees.codomain_vertices()[0].get() as f64
        })
        .unwrap();
    let lazy = source.adjoint().unwrap();

    let flipped = lazy.flip(&[1], Direction::Forward).unwrap();
    assert_eq!(flipped.domain()[0].is_dual(), !lazy.domain()[0].is_dual());

    let expected = source
        .flip(&[0], Direction::Inverse)
        .unwrap()
        .adjoint()
        .unwrap();
    assert_eq!(
        flipped.materialize().unwrap().dense_data().unwrap(),
        expected.materialize().unwrap().dense_data().unwrap()
    );
    assert_eq!(flipped.codomain(), expected.codomain());
    assert_eq!(flipped.domain(), expected.domain());
    assert!(std::ptr::eq(flipped.provider(), provider.as_ref()));
    assert!(flipped.runtime().same_runtime(expected.runtime()));
}

#[test]
fn inverse_twist_and_flip_keep_the_lazy_adjoint_materialization_boundary() {
    let source = fz2_fixture();
    let eager = eager_adjoint_oracle(&source);

    let lazy_twist = source.adjoint().unwrap();
    assert_eq!(
        lazy_twist
            .twist(&[0], Direction::Inverse)
            .unwrap()
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap(),
        eager
            .twist(&[0], Direction::Inverse)
            .unwrap()
            .dense_data()
            .unwrap()
    );

    let lazy_flip = source.adjoint().unwrap();
    let actual = lazy_flip.flip(&[1], Direction::Inverse).unwrap();
    let expected = eager.flip(&[1], Direction::Inverse).unwrap();
    assert_eq!(
        actual.materialize().unwrap().dense_data().unwrap(),
        expected.dense_data().unwrap()
    );
    assert_eq!(
        actual.logical_space().space(),
        expected.logical_space().space()
    );
}

#[test]
fn simple_lazy_observers_and_owned_outputs_do_not_publish() {
    let source = u1_matrix_fixture([(0, 2)], [(0, 2)]);
    let eager = eager_adjoint_oracle(&source);
    let lazy = source.adjoint().unwrap();

    assert_eq!(lazy.is_diagonal(0.0), eager.is_diagonal(0.0));
    assert!(lazy.is_diagonal(-1.0).is_err());
    assert_eq!(
        lazy.convert::<Complex64>()
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap()
            .to_vec(),
        eager
            .convert::<Complex64>()
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap()
            .to_vec()
    );
    assert_eq!(
        lazy.convert::<Complex64>()
            .re()
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap()
            .to_vec(),
        eager
            .convert::<Complex64>()
            .re()
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap()
            .to_vec()
    );
    assert_eq!(
        lazy.convert::<Complex64>()
            .im()
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap()
            .to_vec(),
        eager
            .convert::<Complex64>()
            .im()
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap()
            .to_vec()
    );
    assert_eq!(
        lazy.insert_unit(0, Side::Domain, Duality::Plain)
            .unwrap()
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap()
            .to_vec(),
        eager
            .insert_unit(0, Side::Domain, Duality::Plain)
            .unwrap()
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap()
            .to_vec()
    );

    let scalar = source.trace_pairs(&[(0, 1)]).unwrap();
    let scalar_eager = eager_adjoint_oracle(&scalar);
    let scalar_lazy = scalar.adjoint().unwrap();
    assert_eq!(
        scalar_lazy.scalar().unwrap(),
        scalar_eager.scalar().unwrap()
    );

    let complex = genuinely_complex(&source);
    let eager_complex = eager_adjoint_oracle(&complex);
    let lazy_complex = complex.adjoint().unwrap();
    assert_eq!(
        lazy_complex
            .re()
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap()
            .to_vec(),
        eager_complex
            .re()
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap()
            .to_vec()
    );
    assert_eq!(
        lazy_complex
            .im()
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap()
            .to_vec(),
        eager_complex
            .im()
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap()
            .to_vec()
    );

    let clone = lazy.clone();
    assert_eq!(
        lazy.materialize().unwrap().dense_data().unwrap().to_vec(),
        eager.materialize().unwrap().dense_data().unwrap().to_vec()
    );
    assert_eq!(
        clone.materialize().unwrap().dense_data().unwrap().to_vec(),
        eager.materialize().unwrap().dense_data().unwrap().to_vec()
    );
    // Explicit materialization is not an implicit build.
}

#[test]
fn lazy_otimes_orientations_and_deligne_inputs_stay_cold() {
    let lhs = u1_lazy_fixture();
    let rhs = lhs.scale(2.0).unwrap();
    let eager_lhs = eager_adjoint_oracle(&lhs);
    let eager_rhs = eager_adjoint_oracle(&rhs);
    let lazy_lhs = lhs.adjoint().unwrap();
    let lazy_rhs = rhs.adjoint().unwrap();

    for (actual, expected) in [
        (lhs.otimes(&rhs).unwrap(), lhs.otimes(&rhs).unwrap()),
        (
            lhs.otimes(&lazy_rhs).unwrap(),
            lhs.otimes(&eager_rhs).unwrap(),
        ),
        (
            lazy_lhs.otimes(&rhs).unwrap(),
            eager_lhs.otimes(&rhs).unwrap(),
        ),
        (
            lazy_lhs.otimes(&lazy_rhs).unwrap(),
            eager_lhs.otimes(&eager_rhs).unwrap(),
        ),
    ] {
        assert_eq!(actual.dense_data().unwrap(), expected.dense_data().unwrap());
        assert!(Arc::ptr_eq(
            actual.logical_space().provider_arc(),
            lhs.logical_space().provider_arc()
        ));
    }

    let deligne_lhs = u1_matrix_fixture([(0, 2)], [(0, 2)]);
    let deligne_rhs = deligne_lhs.scale(3.0).unwrap();
    let eager_deligne_lhs = eager_adjoint_oracle(&deligne_lhs);
    let eager_deligne_rhs = eager_adjoint_oracle(&deligne_rhs);
    let lazy_deligne_lhs = deligne_lhs.adjoint().unwrap();
    let lazy_deligne_rhs = deligne_rhs.adjoint().unwrap();
    let product = Arc::new(ProductFusionRule::<_, _>::try_new(U1FusionRule, U1FusionRule).unwrap());
    let actual = lazy_deligne_lhs
        .deligne_product(&lazy_deligne_rhs, Arc::clone(&product))
        .unwrap();
    let expected = eager_deligne_lhs
        .deligne_product(&eager_deligne_rhs, product)
        .unwrap();
    assert_eq!(actual.dense_data().unwrap(), expected.dense_data().unwrap());
}

#[test]
fn absorb_uses_operation_local_logical_payloads_without_warming_lazy_inputs() {
    let destination_parent = genuinely_complex(&u1_lazy_fixture());
    let source_parent = destination_parent
        .scale(num_complex::Complex64::new(2.0, -1.0))
        .unwrap();
    let eager_destination = eager_adjoint_oracle(&destination_parent);
    let eager_source = eager_adjoint_oracle(&source_parent);
    let expected = eager_destination.absorb(&eager_source).unwrap();
    let lazy_destination = destination_parent.adjoint().unwrap();
    let lazy_source = source_parent.adjoint().unwrap();

    let actual = lazy_destination.absorb(&lazy_source).unwrap();
    assert_typed_map_close(&actual, &expected, 1e-12);
}

fn assert_parent_native_transform<R, D>(
    source: &TensorMap<R, D>,
    eager: &TensorMap<R, D>,
    operation: impl Fn(&TensorMap<R, D>) -> Result<TensorMap<R, D>, FacadeError<R>>,
) where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorTransformDispatch<R, D> + TypedAdjointSpace<R>,
    D: TensorScalar + core::fmt::Debug,
{
    let lazy = source.adjoint().unwrap();
    let actual = operation(&lazy).unwrap();
    let expected = operation(eager).unwrap();
    let TypedTensorRepr::Adjoint(_) = &actual.repr else {
        panic!("a transformed lazy adjoint must remain parent-backed");
    };
    assert_eq!(
        actual.logical_space().space(),
        expected.logical_space().space()
    );
    assert!(Arc::ptr_eq(
        actual.logical_space().provider_arc(),
        source.logical_space().provider_arc()
    ));
    assert_eq!(
        actual.materialize().unwrap().dense_data().unwrap().len(),
        expected.dense_data().unwrap().len()
    );
    assert!(actual
        .materialize()
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(expected.dense_data().unwrap())
        .all(|(&actual, &expected)| {
            (actual.widen_complex() - expected.widen_complex()).norm() < 1e-12
        }));
}

/// `eager` is an independently built `source^H`.
fn assert_parent_native_transform_suite<R, D>(source: &TensorMap<R, D>, eager: &TensorMap<R, D>)
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorTransformDispatch<R, D> + TypedAdjointSpace<R>,
    D: TensorScalar + core::fmt::Debug,
{
    assert_parent_native_transform(source, eager, |tensor| tensor.permute(&[2, 0], &[1]));
    assert_parent_native_transform(source, eager, |tensor| {
        tensor.braid(&[2, 0], &[1], &[17, 3, 11])
    });
    assert_parent_native_transform(source, eager, |tensor| tensor.transpose(&[2, 1], &[0]));
    assert_parent_native_transform(source, eager, |tensor| tensor.repartition(2));
}

#[test]
fn nonidentity_adjoint_transforms_stay_parent_native_for_unique_simple_and_both_dtypes() {
    let u1 = u1_lazy_fixture();
    let su2 = su2_lazy_fixture();
    let u1_c64 = u1
        .convert::<Complex64>()
        .scale(num_complex::Complex64::new(1.0, 2.0))
        .unwrap();
    let su2_c64 = su2
        .convert::<Complex64>()
        .scale(num_complex::Complex64::new(1.0, 2.0))
        .unwrap();
    assert_parent_native_transform_suite(&u1, &eager_adjoint_oracle(&u1));
    assert_parent_native_transform_suite(&u1_c64, &eager_adjoint_oracle(&u1_c64));
    assert_parent_native_transform_suite(&su2, &eager_adjoint_oracle(&su2));
    assert_parent_native_transform_suite(&su2_c64, &eager_adjoint_oracle(&su2_c64));
}

#[cfg(feature = "cgc-gen")]
#[test]
fn nonidentity_adjoint_transforms_stay_parent_native_for_checked_generic_and_both_dtypes() {
    use symtn_core::LieGroupFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let su2 = GradedSpace::try_new(
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(2).unwrap()).unwrap()),
        [(vec![0], 2), (vec![1], 3), (vec![2], 2)],
    )
    .unwrap();
    // `8 ⊗ 8 → 8` has outer multiplicity two.
    let su3 = GradedSpace::try_new(
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap()),
        [(vec![0, 0], 2), (vec![1, 1], 2), (vec![1, 0], 1)],
    )
    .unwrap();
    for leg in [su2, su3] {
        let source: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg], |trees, indices| {
                indices.iter().sum::<usize>() as f64 + trees.coupled().iter().sum::<i64>() as f64
                    - 1.5
            })
            .unwrap();
        let source_c64 = source
            .convert::<Complex64>()
            .scale(num_complex::Complex64::new(1.0, 2.0))
            .unwrap();
        // Oracle (i): TensorKit `copy(t')`, which no transform route reads.
        let eager = source
            .adjoint()
            .unwrap()
            .materialized_tensor_uncached()
            .unwrap();
        assert_parent_native_transform_suite(&source, &eager);
        let eager_c64 = source_c64
            .adjoint()
            .unwrap()
            .materialized_tensor_uncached()
            .unwrap();
        assert_parent_native_transform_suite(&source_c64, &eager_c64);
    }
}

fn assert_parent_native_elementwise<R, D>(source: &TensorMap<R, D>, alpha: D, beta: D)
where
    R: TypedSectorAdmission<
            Sector = <R as SectorCodec>::Sector,
            Error = FusionAlgebraError,
            Mode = MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: TensorScalar + core::fmt::Debug,
{
    let lazy = source.adjoint().unwrap();
    let eager = eager_adjoint_oracle(source);

    let add = lazy.axpby(alpha, &eager, beta).unwrap();
    let expected_add = eager.axpby(alpha, &eager, beta).unwrap();
    assert!(add
        .dense_data()
        .unwrap()
        .iter()
        .zip(expected_add.dense_data().unwrap())
        .all(|(&actual, &expected)| {
            (actual.widen_complex() - expected.widen_complex()).norm() < 1e-12
        }));
    // Two lazy adjoints stay lazy (A10, #1874): the parents are combined.
    let add_both = lazy.axpby(alpha, &lazy, beta).unwrap();
    let TypedTensorRepr::Adjoint(_) = &add_both.repr else {
        panic!("adding two lazy adjoints must remain parent-backed");
    };
    assert!(add_both
        .materialize()
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(expected_add.dense_data().unwrap())
        .all(|(&actual, &expected)| {
            (actual.widen_complex() - expected.widen_complex()).norm() < 1e-12
        }));
    let add_rhs = eager.axpby(alpha, &lazy, beta).unwrap();
    assert!(add_rhs
        .dense_data()
        .unwrap()
        .iter()
        .zip(expected_add.dense_data().unwrap())
        .all(|(&actual, &expected)| {
            (actual.widen_complex() - expected.widen_complex()).norm() < 1e-12
        }));

    let scaled = lazy.scale(alpha).unwrap();
    let expected_scaled = eager.scale(alpha).unwrap();
    let TypedTensorRepr::Adjoint(_) = &scaled.repr else {
        panic!("scaling a lazy adjoint must remain parent-backed");
    };
    assert!(scaled
        .materialize()
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(expected_scaled.dense_data().unwrap())
        .all(|(&actual, &expected)| {
            (actual.widen_complex() - expected.widen_complex()).norm() < 1e-12
        }));

    // Why an absolute bound and not `numerics`: the inner products reach
    // magnitude 520, where the `numerics` bound would exceed the `1e-12` this
    // gate has always held them to.
    let eager_inner = eager.inner(&eager).unwrap().widen_complex();
    for (what, inner) in [
        ("lazy.inner(&eager)", lazy.inner(&eager).unwrap()),
        ("eager.inner(&lazy)", eager.inner(&lazy).unwrap()),
        ("lazy.inner(&lazy)", lazy.inner(&lazy).unwrap()),
    ] {
        assert!(
            (inner.widen_complex() - eager_inner).norm() < 1e-12,
            "{what}"
        );
    }
    assert!((lazy.norm(2.0).unwrap() - eager.norm(2.0).unwrap()).abs() < 1e-12);
    assert!((lazy.norm(f64::INFINITY).unwrap() - eager.norm(f64::INFINITY).unwrap()).abs() < 1e-12);
    assert!((lazy.norm(1.5).unwrap() - eager.norm(1.5).unwrap()).abs() < 1e-12);

    let normalized = lazy
        .scale(D::from_real(1.0 / lazy.norm(2.0).unwrap()))
        .unwrap();
    let TypedTensorRepr::Adjoint(_) = &normalized.repr else {
        panic!("normalizing a lazy adjoint must remain parent-backed");
    };
    assert!((normalized.norm(2.0).unwrap() - 1.0).abs() < 1e-12);
}

#[test]
fn adjoint_elementwise_and_reductions_stay_parent_native() {
    let u1 = u1_lazy_fixture();
    let su2 = su2_lazy_fixture();
    let u1_c64 = u1
        .convert::<Complex64>()
        .scale(num_complex::Complex64::new(1.0, 2.0))
        .unwrap();
    let su2_c64 = su2
        .convert::<Complex64>()
        .scale(num_complex::Complex64::new(1.0, 2.0))
        .unwrap();
    assert_parent_native_elementwise(&u1, 2.0, -0.5);
    assert_parent_native_elementwise(&su2, 2.0, -0.5);
    assert_parent_native_elementwise(
        &u1_c64,
        num_complex::Complex64::new(0.5, 1.0),
        num_complex::Complex64::new(-0.25, 0.75),
    );
    assert_parent_native_elementwise(
        &su2_c64,
        num_complex::Complex64::new(0.5, 1.0),
        num_complex::Complex64::new(-0.25, 0.75),
    );
}

fn assert_parent_native_trace_pairs<R, D>(source: &TensorMap<R, D>)
where
    R: TypedSectorAdmission<
            Sector = <R as SectorCodec>::Sector,
            Error = FusionAlgebraError,
            Mode = MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: TensorScalar + core::fmt::Debug,
{
    let lazy = source.adjoint().unwrap();
    let eager = eager_adjoint_oracle(source);
    let expected = eager.trace_pairs(&[(0, 1)]).unwrap();
    let actual = lazy.trace_pairs(&[(0, 1)]).unwrap();
    assert_eq!(
        actual.logical_space().space(),
        expected.logical_space().space()
    );
    assert!(Arc::ptr_eq(
        actual.logical_space().provider_arc(),
        source.logical_space().provider_arc()
    ));
    assert!(actual
        .dense_data()
        .unwrap()
        .iter()
        .zip(expected.dense_data().unwrap())
        .all(|(&actual, &expected)| {
            (actual.widen_complex() - expected.widen_complex()).norm() < 1e-12
        }));
    assert!(
        matches!(lazy.trace_pairs(&[(0, lazy.rank())]), Err(Error::Operation(e)) if matches!(*e, OperationError::InvalidAxisSet { tensor: "trace pairs", .. }))
    );
    assert!(
        matches!(lazy.trace_pairs(&[(0, 0)]), Err(Error::Operation(e)) if matches!(*e, OperationError::InvalidAxisSet { tensor: "trace pairs", .. }))
    );
}

#[test]
fn adjoint_trace_pairs_stays_parent_native() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let u1_provider = Arc::new(U1FusionRule);
    let u1_traced = GradedSpace::try_new(
        Arc::clone(&u1_provider),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let u1_open = GradedSpace::try_new(u1_provider, [(U1Irrep::new(-1), 1), (U1Irrep::new(0), 1)])
        .and_then(|space| space.try_dual())
        .unwrap();
    let u1 = TensorMap::from_subblock_fn(
        &runtime,
        [&u1_traced, &u1_open],
        [&u1_traced],
        |_, indices| indices.iter().sum::<usize>() as f64 + 1.0,
    )
    .unwrap();
    let su2_provider = Arc::new(SU2FusionRule);
    let su2_traced = GradedSpace::try_new(
        Arc::clone(&su2_provider),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 1),
        ],
    )
    .unwrap();
    let su2_open = GradedSpace::try_new(
        su2_provider,
        [
            (SU2Irrep::from_twice_spin(0), 1),
            (SU2Irrep::from_twice_spin(1), 1),
        ],
    )
    .and_then(|space| space.try_dual())
    .unwrap();
    let su2 = TensorMap::from_subblock_fn(
        &runtime,
        [&su2_traced, &su2_open, &su2_open],
        [&su2_traced],
        |_, indices| indices.iter().sum::<usize>() as f64 + 1.0,
    )
    .unwrap();
    assert!(su2.subblock_count() > 1);
    assert!(
        eager_adjoint_oracle(&su2)
            .trace_pairs(&[(0, 1)])
            .unwrap()
            .subblock_count()
            > 1
    );
    let u1_c64 = u1
        .convert::<Complex64>()
        .scale(num_complex::Complex64::new(1.0, 2.0))
        .unwrap();
    let su2_c64 = su2
        .convert::<Complex64>()
        .scale(num_complex::Complex64::new(1.0, 2.0))
        .unwrap();
    assert_parent_native_trace_pairs(&u1);
    assert_parent_native_trace_pairs(&u1_c64);
    assert_parent_native_trace_pairs(&su2);
    assert_parent_native_trace_pairs(&su2_c64);
}

fn assert_parent_native_tr<R, D>(source: &TensorMap<R, D>)
where
    R: TypedSectorAdmission<
            Sector = <R as SectorCodec>::Sector,
            Error = FusionAlgebraError,
            Mode = MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: TensorScalar + core::fmt::Debug + numerics::Numeric,
{
    let lazy = source.adjoint().unwrap();
    let eager = eager_adjoint_oracle(source);
    numerics::assert_close(
        "lazy.tr()",
        lazy.tr().unwrap(),
        eager.tr().unwrap(),
        GATE_TERMS,
    );
}

#[test]
fn adjoint_positive_trace_conjugates_the_parent_without_materializing() {
    let u1_source = u1_lazy_fixture();
    let u1_leg = u1_source.codomain().remove(0);
    let u1 =
        TensorMap::from_subblock_fn(u1_source.runtime(), [&u1_leg], [&u1_leg], |_, indices| {
            (indices[0] + 2 * indices[1]) as f64 + 1.0
        })
        .unwrap();
    let su2_source = su2_lazy_fixture();
    let su2_leg = su2_source.codomain().remove(0);
    let su2 = TensorMap::from_subblock_fn(
        su2_source.runtime(),
        [&su2_leg],
        [&su2_leg],
        |_, indices| (indices[0] + 2 * indices[1]) as f64 + 1.0,
    )
    .unwrap();
    let u1_c64 = u1
        .convert::<Complex64>()
        .scale(num_complex::Complex64::new(1.0, 2.0))
        .unwrap();
    let su2_c64 = su2
        .convert::<Complex64>()
        .scale(num_complex::Complex64::new(1.0, 2.0))
        .unwrap();
    assert_parent_native_tr(&u1);
    assert_parent_native_tr(&u1_c64);
    assert_parent_native_tr(&su2);
    assert_parent_native_tr(&su2_c64);
}

fn assert_parent_native_contract_and_compose<R, D>(source: &TensorMap<R, D>)
where
    R: TypedSectorAdmission<
            Sector = <R as SectorCodec>::Sector,
            Error = FusionAlgebraError,
            Mode = MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: TensorScalar + core::fmt::Debug,
{
    let lazy = source.adjoint().unwrap();
    let eager = eager_adjoint_oracle(source);
    for (lhs, rhs, expected_lhs, expected_rhs) in [
        (&lazy, &eager, &eager, &eager),
        (&eager, &lazy, &eager, &eager),
        (&lazy, &lazy, &eager, &eager),
    ] {
        let actual = lhs
            .contract(
                rhs,
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &[1],
                    domain: &[0],
                },
            )
            .unwrap();
        let expected = expected_lhs
            .contract(
                expected_rhs,
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &[1],
                    domain: &[0],
                },
            )
            .unwrap();
        assert_eq!(
            actual.logical_space().space(),
            expected.logical_space().space()
        );
        assert!(Arc::ptr_eq(
            actual.logical_space().provider_arc(),
            lhs.logical_space().provider_arc()
        ));
        assert!(actual
            .dense_data()
            .unwrap()
            .iter()
            .zip(expected.dense_data().unwrap())
            .all(|(&actual, &expected)| {
                (actual.widen_complex() - expected.widen_complex()).norm() < 1e-12
            }));

        let actual = lhs.compose(rhs).unwrap();
        let expected = expected_lhs.compose(expected_rhs).unwrap();
        assert_eq!(
            actual.logical_space().space(),
            expected.logical_space().space()
        );
        assert!(Arc::ptr_eq(
            actual.logical_space().provider_arc(),
            lhs.logical_space().provider_arc()
        ));
        assert!(actual
            .dense_data()
            .unwrap()
            .iter()
            .zip(expected.dense_data().unwrap())
            .all(|(&actual, &expected)| {
                (actual.widen_complex() - expected.widen_complex()).norm() < 1e-12
            }));
    }
    assert!(matches!(lazy
        .contract(
            &eager,
            &ContractSpec {
                lhs: &[2],
                rhs: &[0],
                codomain: &[0],
                domain: &[1]
            }
        ), Err(Error::Operation(e)) if matches!(*e, OperationError::InvalidAxisSet { tensor: "lhs", .. })));
}

#[test]
fn adjoint_contract_and_compose_stay_parent_native() {
    let u1_source = u1_lazy_fixture();
    let u1_leg = u1_source.codomain().remove(0);
    let u1 =
        TensorMap::from_subblock_fn(u1_source.runtime(), [&u1_leg], [&u1_leg], |_, indices| {
            (indices[0] + 2 * indices[1]) as f64 + 1.0
        })
        .unwrap();
    let su2_source = su2_lazy_fixture();
    let su2_leg = su2_source.codomain().remove(0);
    let su2 = TensorMap::from_subblock_fn(
        su2_source.runtime(),
        [&su2_leg],
        [&su2_leg],
        |_, indices| (indices[0] + 2 * indices[1]) as f64 + 1.0,
    )
    .unwrap();
    let u1_c64 = u1
        .convert::<Complex64>()
        .scale(num_complex::Complex64::new(1.0, 2.0))
        .unwrap();
    let su2_c64 = su2
        .convert::<Complex64>()
        .scale(num_complex::Complex64::new(1.0, 2.0))
        .unwrap();
    assert_parent_native_contract_and_compose(&u1);
    assert_parent_native_contract_and_compose(&u1_c64);
    assert_parent_native_contract_and_compose(&su2);
    assert_parent_native_contract_and_compose(&su2_c64);
}

/// `t'·t`, `t'∘t`, `t·t'` and `t'·t'` of a checked SU(3) tensor `t` read
/// the parent payload without materializing `t'`, and equal (i) the
/// contraction of `copy(t')` and, for `t'∘t`, (iii) the coupled-block
/// product `B_c(t)ᴴ B_c(t)` over every block, the `8 ⊗ 8 → 8` rows of outer
/// multiplicity two included.
#[cfg(feature = "cgc-gen")]
fn assert_checked_adjoint_contract_and_compose<D>(
    source: &TensorMap<symtn_core::LieGroupFusionRule, D>,
) where
    D: TensorScalar + core::fmt::Debug,
{
    let lazy = source.adjoint().unwrap();
    let eager = lazy.materialized_tensor_uncached().unwrap();
    let spec = |lhs, rhs, codomain, domain| ContractSpec {
        lhs,
        rhs,
        codomain,
        domain,
    };
    let (core, flip, both) = (
        spec(&[1, 2], &[0, 1], &[0], &[1]),
        spec(&[2], &[0], &[0, 1], &[2, 3]),
        spec(&[1], &[0], &[0, 1], &[2, 3]),
    );
    let adjoint = probe::record();
    let outputs = [
        (
            lazy.contract(source, &core).unwrap(),
            eager.contract(source, &core).unwrap(),
        ),
        (
            lazy.compose(source).unwrap(),
            eager.compose(source).unwrap(),
        ),
        (
            source.contract(&lazy, &flip).unwrap(),
            source.contract(&eager, &flip).unwrap(),
        ),
        (
            lazy.contract(&lazy, &both).unwrap(),
            eager.contract(&eager, &both).unwrap(),
        ),
    ];
    assert_eq!(adjoint.get(Event::AdjointMaterialization), 0);
    assert!(matches!(&lazy.repr, TypedTensorRepr::Adjoint(_)));
    let close = |actual: &TensorMap<_, D>, expected: &TensorMap<_, D>| {
        assert_eq!(
            actual.logical_space().space(),
            expected.logical_space().space()
        );
        let (actual, expected) = (actual.dense_data().unwrap(), expected.dense_data().unwrap());
        assert_eq!(actual.len(), expected.len());
        assert!(actual.iter().zip(expected).all(|(&actual, &expected)| {
            (actual.widen_complex() - expected.widen_complex()).norm() < 1e-10
        }));
    };
    for (actual, expected) in &outputs {
        close(actual, expected);
    }
    let mut multiplicity_rows = false;
    for composed in [&outputs[0].0, &outputs[1].0] {
        for (sector, block) in composed.blocks().unwrap() {
            let parent = source.block(&sector).unwrap();
            assert_eq!((block.rows(), block.cols()), (parent.cols(), parent.cols()));
            multiplicity_rows |= sector == vec![1, 1] && parent.rows() > parent.cols();
            for row in 0..block.rows() {
                for col in 0..block.cols() {
                    let expected: num_complex::Complex64 = (0..parent.rows())
                        .map(|k| {
                            parent.get(k, row).unwrap().widen_complex().conj()
                                * parent.get(k, col).unwrap().widen_complex()
                        })
                        .sum();
                    let actual = block.get(row, col).unwrap().widen_complex();
                    assert!((actual - expected).norm() < 1e-10, "{sector:?}");
                }
            }
        }
    }
    assert!(multiplicity_rows);
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_generic_su3_adjoint_contract_and_compose_read_the_parent() {
    use symtn_core::LieGroupFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    // `8 ⊗ 8 → 8` has outer multiplicity two.
    let leg = GradedSpace::try_new(
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap()),
        [(vec![0, 0], 2), (vec![1, 1], 2), (vec![1, 0], 1)],
    )
    .unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg], |trees, indices| {
            indices.iter().sum::<usize>() as f64 + trees.coupled().iter().sum::<i64>() as f64 - 1.5
        })
        .unwrap();
    assert_checked_adjoint_contract_and_compose(&source);
    assert_checked_adjoint_contract_and_compose(
        &source
            .convert::<Complex64>()
            .scale(num_complex::Complex64::new(1.0, 2.0))
            .unwrap(),
    );
}

fn assert_rank_three_su2_contract_and_compose<D>(source: &TensorMap<SU2FusionRule, D>)
where
    D: TensorScalar + core::fmt::Debug,
{
    assert!(source.subblock_count() > 1);
    let lazy = source.adjoint().unwrap();
    let eager = eager_adjoint_oracle(source);

    let actual = lazy
        .contract(
            source,
            &ContractSpec {
                lhs: &[2, 1],
                rhs: &[1, 0],
                codomain: &[1],
                domain: &[0],
            },
        )
        .unwrap();
    let expected = eager
        .contract(
            source,
            &ContractSpec {
                lhs: &[2, 1],
                rhs: &[1, 0],
                codomain: &[1],
                domain: &[0],
            },
        )
        .unwrap();
    assert!(actual.subblock_count() > 1);
    assert_eq!(
        actual.logical_space().space(),
        expected.logical_space().space()
    );
    assert!(Arc::ptr_eq(
        actual.logical_space().provider_arc(),
        source.logical_space().provider_arc()
    ));
    assert!(actual
        .dense_data()
        .unwrap()
        .iter()
        .zip(expected.dense_data().unwrap())
        .all(|(&actual, &expected)| {
            (actual.widen_complex() - expected.widen_complex()).norm() < 1e-12
        }));

    let actual = lazy.compose(source).unwrap();
    let expected = eager.compose(source).unwrap();
    assert!(actual.subblock_count() > 1);
    assert_eq!(
        actual.logical_space().space(),
        expected.logical_space().space()
    );
    assert!(Arc::ptr_eq(
        actual.logical_space().provider_arc(),
        source.logical_space().provider_arc()
    ));
    assert!(actual
        .dense_data()
        .unwrap()
        .iter()
        .zip(expected.dense_data().unwrap())
        .all(|(&actual, &expected)| {
            (actual.widen_complex() - expected.widen_complex()).norm() < 1e-12
        }));
}

#[test]
fn rank_three_su2_adjoint_contract_and_compose_use_oriented_recoupling() {
    let source = su2_lazy_fixture();
    let complex = source
        .convert::<Complex64>()
        .scale(num_complex::Complex64::new(1.0, 2.0))
        .unwrap();
    assert_rank_three_su2_contract_and_compose(&source);
    assert_rank_three_su2_contract_and_compose(&complex);
}

fn assert_fermionic_contract_and_compose_semantics<D>(
    source: &TensorMap<FermionParityFusionRule, D>,
) where
    D: TensorScalar + core::fmt::Debug,
{
    let lazy = source.adjoint().unwrap();
    let eager = eager_adjoint_oracle(source);
    let contract = lazy
        .contract(
            source,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    let expected_contract = eager
        .contract(
            source,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    let compose = lazy.compose(source).unwrap();
    let expected_compose = eager.compose(source).unwrap();
    assert!(contract
        .dense_data()
        .unwrap()
        .iter()
        .zip(expected_contract.dense_data().unwrap())
        .all(|(&actual, &expected)| {
            (actual.widen_complex() - expected.widen_complex()).norm() < 1e-12
        }));
    assert!(compose
        .dense_data()
        .unwrap()
        .iter()
        .zip(expected_compose.dense_data().unwrap())
        .all(|(&actual, &expected)| {
            (actual.widen_complex() - expected.widen_complex()).norm() < 1e-12
        }));
    assert!(contract
        .dense_data()
        .unwrap()
        .iter()
        .zip(compose.dense_data().unwrap())
        .any(|(&contract, &compose)| {
            (contract.widen_complex() - compose.widen_complex()).norm() > 1e-12
        }));
    assert!(Arc::ptr_eq(
        contract.logical_space().provider_arc(),
        source.logical_space().provider_arc()
    ));
    assert!(Arc::ptr_eq(
        compose.logical_space().provider_arc(),
        source.logical_space().provider_arc()
    ));
}

#[test]
fn fermionic_lazy_contract_keeps_the_supertrace_distinct_from_compose() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(FermionParityFusionRule);
    let leg = GradedSpace::try_new(provider, [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 2)])
        .and_then(|space| space.try_dual())
        .unwrap();
    let source = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |trees, indices| {
        let sector = trees.codomain_uncoupled[0];
        let parity_weight = if sector == Z2Irrep::ODD { 3.0 } else { 1.0 };
        parity_weight * (indices[0] + 2 * indices[1] + 1) as f64
    })
    .unwrap();
    let complex = source
        .convert::<Complex64>()
        .scale(num_complex::Complex64::new(1.0, 2.0))
        .unwrap();
    assert_fermionic_contract_and_compose_semantics(&source);
    assert_fermionic_contract_and_compose_semantics(&complex);
}

fn assert_same_error(actual: Error, expected: Error) {
    assert_eq!(
        core::mem::discriminant(&actual),
        core::mem::discriminant(&expected)
    );
    assert_eq!(actual.to_string(), expected.to_string());
}

#[test]
fn lazy_contract_preserves_validation_precedence_without_materializing() {
    let source = {
        let fixture = u1_lazy_fixture();
        let leg = fixture.codomain().remove(0);
        TensorMap::from_subblock_fn(fixture.runtime(), [&leg], [&leg], |_, indices| {
            (indices[0] + 2 * indices[1] + 1) as f64
        })
        .unwrap()
    };
    let eager = eager_adjoint_oracle(&source);
    for spec in [
        ContractSpec {
            lhs: &[2],
            rhs: &[0],
            codomain: &[0],
            domain: &[1],
        },
        ContractSpec {
            lhs: &[1, 1],
            rhs: &[0, 0],
            codomain: &[],
            domain: &[],
        },
    ] {
        let lazy = source.adjoint().unwrap();
        assert_same_error(
            lazy.contract(&eager, &spec).unwrap_err(),
            eager.contract(&eager, &spec).unwrap_err(),
        );
    }

    let bad_leg = GradedSpace::try_new(
        Arc::clone(source.logical_space().provider_arc()),
        [(U1Irrep::new(7), 1)],
    )
    .unwrap();
    let bad =
        TensorMap::from_subblock_fn(source.runtime(), [&bad_leg], [&bad_leg], |_, _| 1.0).unwrap();
    let lazy = source.adjoint().unwrap();
    assert_same_error(
        lazy.contract(
            &bad,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[0],
            },
        )
        .unwrap_err(),
        eager
            .contract(
                &bad,
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &[0],
                    domain: &[0],
                },
            )
            .unwrap_err(),
    );

    let other_runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let source_leg = source.codomain().remove(0);
    let other =
        TensorMap::from_subblock_fn(&other_runtime, [&source_leg], [&source_leg], |_, _| 1.0)
            .unwrap();
    let lazy = source.adjoint().unwrap();
    assert_same_error(
        lazy.contract(
            &other,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap_err(),
        eager
            .contract(
                &other,
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &[0],
                    domain: &[1],
                },
            )
            .unwrap_err(),
    );

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let z2 = Arc::new(ZNFusionRule::new(2).unwrap());
    let z3 = Arc::new(ZNFusionRule::new(3).unwrap());
    let z2_leg = GradedSpace::try_new(Arc::clone(&z2), [(z2.irrep(0), 2)]).unwrap();
    let z3_leg = GradedSpace::try_new(Arc::clone(&z3), [(z3.irrep(0), 2)]).unwrap();
    let z2_tensor =
        TensorMap::from_subblock_fn(&runtime, [&z2_leg], [&z2_leg], |_, _| 1.0).unwrap();
    let z3_tensor =
        TensorMap::from_subblock_fn(&runtime, [&z3_leg], [&z3_leg], |_, _| 1.0).unwrap();
    let eager = eager_adjoint_oracle(&z2_tensor);
    let lazy = z2_tensor.adjoint().unwrap();
    assert_same_error(
        lazy.contract(&z3_tensor, &RANK_TWO_COMPOSE).unwrap_err(),
        eager.contract(&z3_tensor, &RANK_TWO_COMPOSE).unwrap_err(),
    );
}

#[test]
fn lazy_binary_outputs_keep_the_lhs_provider_allocation() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let lhs_provider = Arc::new(U1FusionRule);
    let rhs_provider = Arc::new(U1FusionRule);
    assert!(!Arc::ptr_eq(&lhs_provider, &rhs_provider));
    let lhs_leg = GradedSpace::try_new(
        Arc::clone(&lhs_provider),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let rhs_leg = GradedSpace::try_new(
        Arc::clone(&rhs_provider),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let lhs = TensorMap::from_subblock_fn(&runtime, [&lhs_leg], [&lhs_leg], |_, indices| {
        (indices[0] + 2 * indices[1] + 1) as f64
    })
    .unwrap();
    let rhs = TensorMap::from_subblock_fn(&runtime, [&rhs_leg], [&rhs_leg], |_, indices| {
        (2 * indices[0] + indices[1] + 1) as f64
    })
    .unwrap();
    let lazy = lhs.adjoint().unwrap();
    let eager = eager_adjoint_oracle(&lhs);
    let rhs_lazy = rhs.adjoint().unwrap();
    let rhs_eager = eager_adjoint_oracle(&rhs);
    for (actual, expected) in [
        (
            lazy.contract(
                &rhs_lazy,
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &[0],
                    domain: &[1],
                },
            )
            .unwrap(),
            eager.contract(&rhs_eager, &RANK_TWO_COMPOSE).unwrap(),
        ),
        (
            lazy.compose(&rhs_lazy).unwrap(),
            eager.compose(&rhs_eager).unwrap(),
        ),
    ] {
        assert_eq!(actual.dense_data().unwrap(), expected.dense_data().unwrap());
        assert!(Arc::ptr_eq(
            actual.logical_space().provider_arc(),
            &lhs_provider
        ));
        assert!(!Arc::ptr_eq(
            actual.logical_space().provider_arc(),
            &rhs_provider
        ));
    }
}

#[test]
fn identity_transforms_preserve_the_cold_lazy_view() {
    let adjoint = u1_lazy_fixture().adjoint().unwrap();
    let TypedTensorRepr::Adjoint(view) = &adjoint.repr else {
        unreachable!()
    };

    let outputs = [
        adjoint.permute(&[0], &[1, 2]).unwrap(),
        adjoint.braid(&[0], &[1, 2], &[0, 1, 2]).unwrap(),
        adjoint.transpose(&[0], &[1, 2]).unwrap(),
        adjoint.repartition(1).unwrap(),
    ];
    for output in &outputs {
        let TypedTensorRepr::Adjoint(output_view) = &output.repr else {
            panic!("identity transform must preserve the lazy representation");
        };
        assert!(Arc::ptr_eq(view, output_view));
    }
    assert!(adjoint.braid(&[0], &[1, 2], &[]).is_err());
    assert!(
        matches!(adjoint.permute(&[0, 0], &[1]), Err(Error::Operation(e)) if matches!(*e, OperationError::InvalidPermutation { rank: 3, .. }))
    );
    assert!(
        matches!(adjoint.braid(&[0, 0], &[1], &[0, 1, 2]), Err(Error::Operation(e)) if matches!(*e, OperationError::InvalidPermutation { rank: 3, .. }))
    );
    assert!(
        matches!(adjoint.transpose(&[0, 0], &[1]), Err(Error::Operation(e)) if matches!(*e, OperationError::InvalidPermutation { rank: 3, .. }))
    );
    assert!(adjoint.repartition(4).is_err());

    let scalar = fixture().trace_pairs(&[(0, 1)]).unwrap();
    let scalar_adjoint = scalar.adjoint().unwrap();
    let TypedTensorRepr::Adjoint(scalar_view) = &scalar_adjoint.repr else {
        unreachable!()
    };
    let scalar_transpose = scalar_adjoint.transpose(&[], &[]).unwrap();
    let TypedTensorRepr::Adjoint(transpose_view) = &scalar_transpose.repr else {
        panic!("rank-zero transpose must preserve the lazy representation");
    };
    assert!(Arc::ptr_eq(scalar_view, transpose_view));
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "a lazy adjoint never holds a compact diagonal parent")]
fn lazy_adjoint_constructor_rejects_a_compact_diagonal_parent() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let bond = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let diagonal = TensorMap::diagonal(
        &runtime,
        &bond,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![1.0, 2.0],
        }],
    )
    .unwrap();
    let _ = TypedAdjointView::new(
        Arc::clone(owned(&diagonal)),
        diagonal.logical_space().clone(),
    );
}

#[test]
fn compact_adjoint_never_enters_the_lazy_representation() {
    let densified = probe::record();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let bond = GradedSpace::try_new(provider, [(U1Irrep::new(0), 2)]).unwrap();
    let real = TensorMap::diagonal(
        &runtime,
        &bond,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![1.0, 2.0],
        }],
    )
    .unwrap();
    let complex = TensorMap::diagonal(
        &runtime,
        &bond,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![
                num_complex::Complex64::new(1.0, 2.0),
                num_complex::Complex64::new(3.0, -4.0),
            ],
        }],
    )
    .unwrap();

    let real_adjoint = real.adjoint().unwrap();
    let complex_adjoint = complex.adjoint().unwrap();
    assert!(matches!(&real_adjoint.repr, TypedTensorRepr::Owned(_)));
    assert!(matches!(&complex_adjoint.repr, TypedTensorRepr::Owned(_)));
    assert!(matches!(
        owned(&real_adjoint).data.as_ref(),
        TypedData::Diagonal(_)
    ));
    assert_eq!(real_adjoint.spectrum().unwrap()[0].values, [1.0, 2.0]);
    // TensorKit's real `adjoint(d) = d`: the body is shared, not copied.
    assert!(Arc::ptr_eq(owned(&real_adjoint), owned(&real)));
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    let spectrum = complex_adjoint.spectrum().unwrap();
    assert_eq!(
        spectrum[0].values,
        [
            num_complex::Complex64::new(1.0, -2.0),
            num_complex::Complex64::new(3.0, 4.0)
        ]
    );
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);

    let complex_restored = complex_adjoint.adjoint().unwrap();
    assert!(matches!(&complex_restored.repr, TypedTensorRepr::Owned(_)));
    assert!(matches!(
        owned(&complex_restored).data.as_ref(),
        TypedData::Diagonal(_)
    ));
    assert_eq!(
        complex_restored.spectrum().unwrap()[0].values,
        [
            num_complex::Complex64::new(1.0, 2.0),
            num_complex::Complex64::new(3.0, -4.0)
        ]
    );
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
}

#[test]
fn lazy_cat_matches_eager_adjoint_cat_without_materializing_an_adjoint() {
    // What: `cat` of lazy adjoints on either side equals `cat` of the eager
    // adjoints, over two coupled sectors, without materializing an adjoint.
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = |degeneracies: [usize; 2]| {
        GradedSpace::try_new(
            Arc::clone(&provider),
            [
                (U1Irrep::new(0), degeneracies[0]),
                (U1Irrep::new(1), degeneracies[1]),
            ],
        )
        .unwrap()
    };
    let common = leg([3, 2]);
    let left = leg([2, 1]);
    let right = leg([4, 2]);
    let parent =
        |codomain: &GradedSpace<U1FusionRule>, domain: &GradedSpace<U1FusionRule>, seed| {
            TensorMap::<U1FusionRule, num_complex::Complex64>::rand_with_seed(
                &runtime,
                [codomain],
                [domain],
                seed,
            )
            .unwrap()
        };

    for (lhs, rhs, side) in [
        (
            parent(&left, &common, 773_101),
            parent(&right, &common, 773_102),
            Side::Domain,
        ),
        (
            parent(&common, &left, 773_103),
            parent(&common, &right, 773_104),
            Side::Codomain,
        ),
    ] {
        let expected = eager_adjoint_oracle(&lhs)
            .cat(&eager_adjoint_oracle(&rhs), side)
            .unwrap();
        let lazy_lhs = lhs.adjoint().unwrap();
        let lazy_rhs = rhs.adjoint().unwrap();
        let adjoint = probe::record();

        let actual = lazy_lhs.cat(&lazy_rhs, side).unwrap();

        assert_eq!(adjoint.get(Event::AdjointMaterialization), 0);
        assert_eq!(
            actual.logical_space().space(),
            expected.logical_space().space()
        );
        assert_eq!(actual.dense_data().unwrap(), expected.dense_data().unwrap());
    }
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_trace_nonselfdual_multiblock_lazy_matches_hand_weighted_trace() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(
        symtn_core::LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap(),
    );
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(vec![0, 0], 2), (vec![1, 0], 3)]).unwrap();
    let source = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, ij| {
        Complex64::new(
            (1 + ij[0] + 4 * ij[1]) as f64,
            0.5 + ij[0] as f64 - 2.0 * ij[1] as f64,
        )
    })
    .unwrap();
    assert_eq!(source.subblock_count(), 2);
    let direct = source.trace_pairs(&[(0, 1)]).unwrap();
    // Vacuum: 1+6. Fundamental: 3*(1+6+11), with the same
    // dimension weights on the imaginary diagonal components.
    let expected = Complex64::new(61.0, -4.5);
    assert!((direct.dense_data().unwrap()[0] - expected).norm() < 1e-12);
    let lazy = source.adjoint().unwrap();
    let TypedTensorRepr::Adjoint(view) = &lazy.repr else {
        panic!("expected lazy source")
    };
    let parent = Arc::clone(&view.parent);
    let result = lazy.trace_pairs(&[(0, 1)]).unwrap();
    assert!((result.dense_data().unwrap()[0] - expected.conj()).norm() < 1e-12);
    assert!(Arc::ptr_eq(
        result.logical_space().provider_arc(),
        &provider
    ));
    let TypedTensorRepr::Adjoint(view) = &lazy.repr else {
        panic!("trace changed source representation")
    };
    assert!(Arc::ptr_eq(&view.parent, &parent));
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_trace_nonselfdual_multiblock_real_matches_hand_weighted_trace() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(
        symtn_core::LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap(),
    );
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 0], 2), (vec![2, 0], 3)]).unwrap();
    let source = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, ij| {
        (1 + ij[0] + 4 * ij[1]) as f64
    })
    .unwrap();
    assert_eq!(source.subblock_count(), 2);
    // SU(3) fundamental and symmetric-square dimensions are 3 and 6:
    // 3*(1+6) + 6*(1+6+11) = 129. Off-diagonal values do not contribute.
    let expected = 129.0;
    let direct = source.trace_pairs(&[(0, 1)]).unwrap();
    assert_eq!(direct.dense_data().unwrap().len(), 1);
    assert!((direct.dense_data().unwrap()[0] - expected).abs() < 1e-12);
    let lazy = source.adjoint().unwrap();
    assert!(matches!(&lazy.repr, TypedTensorRepr::Adjoint(_)));
    let result = lazy.trace_pairs(&[(0, 1)]).unwrap();
    assert!((result.dense_data().unwrap()[0] - expected).abs() < 1e-12);
    assert!(Arc::ptr_eq(
        result.logical_space().provider_arc(),
        &provider
    ));
    assert!(matches!(&lazy.repr, TypedTensorRepr::Adjoint(_)));
}
