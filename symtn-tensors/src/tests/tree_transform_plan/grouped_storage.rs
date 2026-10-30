use super::*;

#[test]
fn explicit_keyed_replay_accepts_dense_and_opaque_namespaces() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    for key in [BlockKey::Dense, BlockKey::opaque([7, 11])] {
        let structure = BlockStructure::from_blocks(vec![BlockSpec::column_major_with_key(
            key.clone(),
            vec![1],
            0,
        )
        .unwrap()])
        .unwrap();
        let space = vec![1];
        let src =
            TestTensor::<f64>::from_vec_with_structure(vec![3.0], space.clone(), structure.clone())
                .unwrap();
        let mut dst =
            TestTensor::<f64>::from_vec_with_structure(vec![0.0], space, structure).unwrap();
        let replay = TreeTransformStructure::compile_keyed_shared_structures(
            Arc::clone(dst.structure()),
            Arc::clone(src.structure()),
            &[TreeTransformKeyBlockSpec::single(key.clone(), key, 2.0)],
            false,
        )
        .unwrap();
        let mut backend = DenseTreeTransformOperations::default();
        let mut workspace = TreeTransformWorkspace::default();

        replay_into(
            engine,
            &mut backend,
            &mut workspace,
            &replay,
            &mut dst,
            &src,
            1.0,
            0.0,
        )
        .unwrap();

        // What: explicit application-key replay remains namespace-neutral; only
        // categorical tree-transform planning requires FusionTreePairKey.
        assert_eq!(dst.data(), &[6.0]);
    }
}

#[test]
fn tree_transform_compile_grouped_lowers_to_replay_ready_structure() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let key10 = grouped_su2_test_pair(0);
    let key20 = grouped_su2_test_pair(2);
    let key100 = grouped_su2_test_pair(0);
    let key200 = grouped_su2_test_pair(2);
    let key300 = grouped_su2_test_pair(4);
    let src_space = vec![6, 1];
    let dst_space = vec![4, 1];
    let src_structure = packed_fixture_structure(
        2,
        [
            (key100.clone(), vec![2, 1]),
            (key300.clone(), vec![2, 1]),
            (key200.clone(), vec![2, 1]),
        ],
    )
    .unwrap();
    let dst_structure = packed_fixture_structure(
        2,
        [(key20.clone(), vec![2, 1]), (key10.clone(), vec![2, 1])],
    )
    .unwrap();
    let src = TestTensor::<f64>::from_vec_with_structure(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        src_space,
        src_structure,
    )
    .unwrap();
    let mut dst =
        TestTensor::<f64>::from_vec_with_structure(vec![0.0; 4], dst_space, dst_structure).unwrap();
    let structure = TreeTransformGroupPlan::new(vec![TreeTransformGroupBlockSpec::try_multi(
        [key10, key20],
        [key100, key200, key300],
        vec![10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0],
    )
    .unwrap()])
    .compile_structures(dst.structure(), src.structure())
    .unwrap();
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();

    replay_into(
        engine,
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &src,
        1.0,
        0.0,
    )
    .unwrap();

    assert_eq!(structure.block_count(), 1);
    assert_eq!(dst.data(), &[7020.0, 9240.0, 3510.0, 4620.0]);
    assert_eq!(workspace.source_len(), 6);
    assert_eq!(workspace.destination_len(), 4);
}

#[test]
fn keyed_and_grouped_compile_resolve_every_key_before_structural_validation() {
    let present = grouped_su2_test_pair(0);
    let missing = grouped_su2_test_pair(2);
    let dst_structure = packed_fixture_structure(2, [(present.clone(), vec![1, 1])]).unwrap();
    let src_structure = packed_fixture_structure(1, [(present.clone(), vec![1])]).unwrap();
    let structurally_invalid =
        TreeTransformGroupBlockSpec::try_multi([present.clone()], [present.clone()], vec![1.0_f64])
            .unwrap();
    let missing_later = TreeTransformGroupBlockSpec::single(missing.clone(), present.clone(), 1.0);

    let err = TreeTransformGroupPlan::new(vec![structurally_invalid.clone(), missing_later])
        .compile_structures(&dst_structure, &src_structure)
        .unwrap_err();
    assert_eq!(
        err,
        OperationError::MissingBlockKey {
            key: Box::new(BlockKey::from(missing))
        }
    );

    let err = TreeTransformGroupPlan::new(vec![structurally_invalid])
        .compile_structures(&dst_structure, &src_structure)
        .unwrap_err();
    assert_eq!(
        err,
        OperationError::StructureRankMismatch {
            expected: 2,
            actual: 1,
        }
    );

    let coefficient_mismatch =
        TreeTransformKeyBlockSpec::multi([present.clone()], [present.clone()], Vec::<f64>::new());
    let missing_later = TreeTransformKeyBlockSpec::single(BlockKey::opaque([2]), present, 1.0);
    let err = TreeTransformStructure::compile_keyed_shared_structures(
        Arc::new(dst_structure.clone()),
        Arc::new(src_structure.clone()),
        &[coefficient_mismatch, missing_later],
        false,
    )
    .unwrap_err();
    assert_eq!(
        err,
        OperationError::MissingBlockKey {
            key: Box::new(BlockKey::opaque([2]))
        }
    );
}

#[test]
fn grouped_storage_mapping_preserves_callback_error_order() {
    let present = grouped_su2_test_pair(0);
    let missing = grouped_su2_test_pair(2);
    let structure = Arc::new(packed_fixture_structure(1, [(present.clone(), vec![1])]).unwrap());
    let plan = TreeTransformGroupPlan::new(vec![
        TreeTransformGroupBlockSpec::single(present.clone(), present.clone(), 1.0_f64),
        TreeTransformGroupBlockSpec::single(missing, present, 1.0),
    ]);
    let axis_called = std::cell::Cell::new(false);

    let err = plan
        .compile_shared_structures_with_storage_mapping(
            Arc::clone(&structure),
            &structure,
            Arc::clone(&structure),
            |_| Err(OperationError::ElementCountOverflow),
            |_| {
                axis_called.set(true);
                Ok(0)
            },
            false,
        )
        .unwrap_err();

    assert_eq!(err, OperationError::ElementCountOverflow);
    assert!(!axis_called.get());
}

#[test]
fn grouped_storage_mapping_owns_coefficients_and_matches_direct_complex_replay() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let dst0 = grouped_su2_test_pair(0);
    let dst1 = grouped_su2_test_pair(2);
    let dst2 = grouped_su2_test_pair(4);
    let src0 = grouped_su2_test_pair(0);
    let src1 = grouped_su2_test_pair(2);
    let src2 = grouped_su2_test_pair(4);
    let dst_structure = Arc::new(
        packed_fixture_structure(
            2,
            [
                (dst0.clone(), vec![1, 2]),
                (dst1.clone(), vec![1, 2]),
                (dst2.clone(), vec![1, 2]),
            ],
        )
        .unwrap(),
    );
    let logical_src_structure = packed_fixture_structure(
        2,
        [
            (src0.clone(), vec![2, 1]),
            (src1.clone(), vec![2, 1]),
            (src2.clone(), vec![2, 1]),
        ],
    )
    .unwrap();
    let storage_src_structure = Arc::new(
        packed_fixture_structure(
            2,
            [
                (src1.clone(), vec![1, 2]),
                (src2.clone(), vec![1, 2]),
                (src0.clone(), vec![1, 2]),
            ],
        )
        .unwrap(),
    );
    let coefficients = vec![
        Complex64::new(0.5, -2.0),
        Complex64::new(1.0, 1.0),
        Complex64::new(2.0, -1.0),
        Complex64::new(-3.0, 0.5),
        Complex64::new(4.0, 2.0),
    ];
    let callback_trace = std::cell::RefCell::new(Vec::new());
    let grouped = {
        let plan = TreeTransformGroupPlan::new(vec![
            TreeTransformGroupBlockSpec::single(dst0, src0, coefficients[0])
                .with_source_axes([1, 0]),
            TreeTransformGroupBlockSpec::try_multi(
                [dst1, dst2],
                [src1, src2],
                coefficients[1..].to_vec(),
            )
            .unwrap()
            .with_source_axes([1, 0]),
        ]);
        plan.compile_shared_structures_with_storage_mapping(
            Arc::clone(&dst_structure),
            &logical_src_structure,
            Arc::clone(&storage_src_structure),
            |block| {
                callback_trace.borrow_mut().push(("block", block));
                Ok(match block {
                    0 => 2,
                    1 => 0,
                    2 => 1,
                    _ => unreachable!("logical source block is resolved from the structure"),
                })
            },
            |axis| {
                callback_trace.borrow_mut().push(("axis", axis));
                Ok(1 - axis)
            },
            true,
        )
        .unwrap()
    };
    let direct_specs = [
        TreeTransformBlockSpec::single(0, 2, coefficients[0]).with_source_axes([0, 1]),
        TreeTransformBlockSpec::multi(vec![1, 2], vec![0, 1], coefficients[1..].to_vec())
            .with_source_axes([0, 1]),
    ];
    let direct = TreeTransformStructure::compile_structures_with_storage_conjugation(
        &dst_structure,
        &storage_src_structure,
        &direct_specs,
        true,
    )
    .unwrap();

    assert_eq!(
        callback_trace.into_inner(),
        [
            ("block", 0),
            ("axis", 1),
            ("axis", 0),
            ("block", 1),
            ("block", 2),
            ("axis", 1),
            ("axis", 0),
        ]
    );
    assert_eq!(grouped, direct);
    assert_eq!(grouped.gathered_coefficients(), coefficients.as_slice());

    let src_space = vec![3, 2];
    let dst_space = vec![3, 2];
    let src = TestTensor::<Complex64>::from_vec_with_structure(
        vec![
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, 4.0),
            Complex64::new(5.0, 6.0),
            Complex64::new(7.0, 8.0),
            Complex64::new(9.0, 10.0),
            Complex64::new(11.0, 12.0),
        ],
        src_space,
        storage_src_structure.as_ref().clone(),
    )
    .unwrap();
    let mut grouped_dst = TestTensor::<Complex64>::from_vec_with_structure(
        vec![Complex64::new(0.0, 0.0); 6],
        dst_space.clone(),
        dst_structure.as_ref().clone(),
    )
    .unwrap();
    let mut direct_dst = TestTensor::<Complex64>::from_vec_with_structure(
        vec![Complex64::new(0.0, 0.0); 6],
        dst_space,
        dst_structure.as_ref().clone(),
    )
    .unwrap();
    let mut grouped_backend = DenseTreeTransformOperations::default();
    let mut grouped_workspace = TreeTransformWorkspace::default();
    let mut direct_backend = DenseTreeTransformOperations::default();
    let mut direct_workspace = TreeTransformWorkspace::default();

    replay_into(
        engine,
        &mut grouped_backend,
        &mut grouped_workspace,
        &grouped,
        &mut grouped_dst,
        &src,
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap();
    replay_into(
        engine,
        &mut direct_backend,
        &mut direct_workspace,
        &direct,
        &mut direct_dst,
        &src,
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap();

    assert_eq!(grouped_dst.data(), direct_dst.data());
}

#[test]
fn tree_transform_compile_grouped_rejects_missing_tree_block_key() {
    let src_space = vec![2, 2];
    let dst_space = vec![2, 2];
    let present_key = grouped_su2_test_pair(0);
    let missing_key = grouped_su2_test_pair(2);
    let src_structure = packed_fixture_structure(2, [(present_key.clone(), vec![2, 2])]).unwrap();
    let dst_structure = packed_fixture_structure(2, [(present_key.clone(), vec![2, 2])]).unwrap();
    let src =
        TestTensor::<f64>::from_vec_with_structure(vec![1.0; 4], src_space, src_structure).unwrap();
    let dst =
        TestTensor::<f64>::from_vec_with_structure(vec![0.0; 4], dst_space, dst_structure).unwrap();

    let err = TreeTransformGroupPlan::new(vec![TreeTransformGroupBlockSpec::single(
        missing_key.clone(),
        present_key,
        1.0,
    )])
    .compile_structures(dst.structure(), src.structure())
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::MissingBlockKey {
            key: Box::new(BlockKey::from(missing_key))
        }
    );
}

#[test]
fn tree_transform_group_block_spec_from_groups_uses_source_group_and_ordered_keys() {
    let src_key1 = fusion_tree_test_key([10, 20], [30], 5, [false, true], [true]);
    let src_key2 = fusion_tree_test_key([10, 20], [30], 6, [false, true], [true]);
    let dst_key1 = fusion_tree_test_key([20, 10], [30], 7, [true, false], [true]);
    let dst_key2 = fusion_tree_test_key([20, 10], [30], 8, [true, false], [true]);
    let src_structure = packed_fixture_structure(
        2,
        [
            (src_key1.clone(), vec![1, 1]),
            (src_key2.clone(), vec![1, 1]),
        ],
    )
    .unwrap();
    let dst_structure = packed_fixture_structure(
        2,
        [
            (dst_key1.clone(), vec![1, 1]),
            (dst_key2.clone(), vec![1, 1]),
        ],
    )
    .unwrap();
    let src_groups = src_structure.fusion_tree_groups();
    let dst_groups = dst_structure.fusion_tree_groups();

    let spec = TreeTransformGroupBlockSpec::from_block_groups(
        &dst_structure,
        &dst_groups[0],
        &src_structure,
        &src_groups[0],
        vec![1.0_f64, 2.0, 3.0, 4.0],
    )
    .unwrap();

    assert_eq!(&spec.group_key(), src_groups[0].group_key());
    assert_ne!(&spec.group_key(), dst_groups[0].group_key());
    assert_eq!(spec.src_keys(), &[src_key1, src_key2]);
    assert_eq!(spec.dst_keys(), &[dst_key1, dst_key2]);
    assert_eq!(
        spec.recoupling_coefficients_dst_src(),
        &[1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn grouped_spec_from_public_groups_rejects_repeated_block_indices() {
    let key = grouped_su2_test_pair(0);
    let structure = packed_fixture_structure(2, [(key, vec![1, 1])]).unwrap();
    let group = structure.fusion_tree_groups().pop().unwrap();
    let repeated = symtn_core::FusionTreeBlockGroup::new(group.group_key().clone(), vec![0, 0]);

    let src_err = TreeTransformGroupBlockSpec::from_block_groups(
        &structure,
        &group,
        &structure,
        &repeated,
        vec![1.0_f64, 2.0],
    )
    .unwrap_err();
    let dst_err = TreeTransformGroupBlockSpec::from_block_groups(
        &structure,
        &repeated,
        &structure,
        &group,
        vec![1.0_f64, 2.0],
    )
    .unwrap_err();

    // What: a caller-built group cannot duplicate a matrix column or row by
    // repeating one otherwise valid BlockStructure index.
    assert_eq!(
        src_err,
        OperationError::DuplicateTreeTransformKey {
            tensor: "src",
            index: 1,
        }
    );
    assert_eq!(
        dst_err,
        OperationError::DuplicateTreeTransformKey {
            tensor: "dst",
            index: 1,
        }
    );
}

#[test]
fn tree_transform_group_plan_compiles_across_degeneracy_shapes_without_layout_leakage() {
    let src_key1 = fusion_tree_test_key([10, 20], [30], 5, [false, true], [true]);
    let src_key2 = fusion_tree_test_key([10, 20], [30], 6, [false, true], [true]);
    let dst_key1 = fusion_tree_test_key([20, 10], [30], 7, [true, false], [true]);
    let dst_key2 = fusion_tree_test_key([20, 10], [30], 8, [true, false], [true]);
    let src_small = packed_fixture_structure(
        2,
        [
            (src_key1.clone(), vec![2, 1]),
            (src_key2.clone(), vec![2, 1]),
        ],
    )
    .unwrap();
    let dst_small = packed_fixture_structure(
        2,
        [
            (dst_key1.clone(), vec![2, 1]),
            (dst_key2.clone(), vec![2, 1]),
        ],
    )
    .unwrap();
    let src_large =
        packed_fixture_structure(2, [(src_key1, vec![3, 1]), (src_key2, vec![3, 1])]).unwrap();
    let dst_large =
        packed_fixture_structure(2, [(dst_key1, vec![3, 1]), (dst_key2, vec![3, 1])]).unwrap();
    let spec = TreeTransformGroupBlockSpec::from_block_groups(
        &dst_small,
        &dst_small.fusion_tree_groups()[0],
        &src_small,
        &src_small.fusion_tree_groups()[0],
        vec![1.0_f64, 0.0, 0.0, 1.0],
    )
    .unwrap();
    let plan = TreeTransformGroupPlan::new(vec![spec]);

    let small_structure = plan.compile_structures(&dst_small, &src_small).unwrap();
    let large_structure = plan.compile_structures(&dst_large, &src_large).unwrap();

    assert_eq!(plan.specs().len(), 1);
    assert_eq!(small_structure.block_count(), 1);
    assert_eq!(large_structure.block_count(), 1);
    assert_eq!(small_structure.workspace_lens(), (4, 4));
    assert_eq!(large_structure.workspace_lens(), (6, 6));
}

#[test]
fn tree_transform_structure_cache_key_tracks_concrete_layout() {
    let key = fusion_tree_test_key([10, 20], [30], 5, [false, true], [true]);
    let src = BlockStructure::from_blocks_with_rank(
        2,
        vec![BlockSpec::with_key(key.clone().into(), vec![2, 3], vec![1, 2], 0).unwrap()],
    )
    .unwrap();
    let shape_changed = BlockStructure::from_blocks_with_rank(
        2,
        vec![BlockSpec::with_key(key.clone().into(), vec![3, 2], vec![1, 3], 0).unwrap()],
    )
    .unwrap();
    let stride_changed = BlockStructure::from_blocks_with_rank(
        2,
        vec![BlockSpec::with_key(key.clone().into(), vec![2, 3], vec![2, 1], 0).unwrap()],
    )
    .unwrap();
    let offset_changed = BlockStructure::from_blocks_with_rank(
        2,
        vec![BlockSpec::with_key(key.clone().into(), vec![2, 3], vec![1, 2], 1).unwrap()],
    )
    .unwrap();
    let key = |dst: &BlockStructure, src: &BlockStructure, conjugate: bool| {
        crate::tree_transform::CompletedTransformerKey::tree_pair::<f64>(
            symtn_core::RuleIdentity::of_type::<()>(),
            &TreeTransformOperation::permute([0, 1], []),
            dst,
            src,
            conjugate,
        )
    };
    let base = key(&src, &src, false);

    assert_ne!(base, key(&src, &src, true));
    assert_ne!(base, key(&shape_changed, &src, false));
    assert_ne!(base, key(&stride_changed, &src, false));
    assert_ne!(base, key(&offset_changed, &src, false));
}

#[test]
fn tree_transform_group_block_spec_rejects_group_structure_mismatch() {
    let src_key = fusion_tree_test_key([10, 20], [30], 5, [false, true], [true]);
    let src_structure = packed_fixture_structure(2, [(src_key, vec![1, 1])]).unwrap();
    let dense_structure = BlockStructure::trivial(&[1, 1]).unwrap();
    let src_groups = src_structure.fusion_tree_groups();

    let err = TreeTransformGroupBlockSpec::<f64>::from_block_groups(
        &dense_structure,
        &src_groups[0],
        &src_structure,
        &src_groups[0],
        vec![1.0],
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::FusionTreeGroupMismatch {
            tensor: "dst",
            index: 0,
        }
    );
}

#[test]
fn tree_transform_rejects_incompatible_single_tree_shapes() {
    let src_space = vec![2, 2];
    let dst_space = vec![4, 1];
    let src = TestTensor::<f64>::from_vec(vec![1.0; 4], src_space).unwrap();
    let dst = TestTensor::<f64>::filled(0.0, dst_space).unwrap();

    let err = TreeTransformStructure::compile_structures(
        dst.structure(),
        src.structure(),
        &[TreeTransformBlockSpec::single(0, 0, 1.0)],
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::ShapeMismatch {
            dst: vec![4, 1],
            src: vec![2, 2],
        }
    );
}

#[test]
fn tree_transform_rejects_mismatched_multi_tree_element_count() {
    let src_space = vec![4, 2];
    let dst_space = vec![3, 2];
    let src_structure = BlockStructure::packed_column_major(2, [vec![2, 2], vec![2, 2]]).unwrap();
    let dst_structure = BlockStructure::packed_column_major(2, [vec![3, 1], vec![3, 1]]).unwrap();
    let src =
        TestTensor::<f64>::from_vec_with_structure(vec![1.0; 8], src_space, src_structure).unwrap();
    let dst =
        TestTensor::<f64>::from_vec_with_structure(vec![0.0; 6], dst_space, dst_structure).unwrap();

    let err = TreeTransformStructure::compile_structures(
        dst.structure(),
        src.structure(),
        &[TreeTransformBlockSpec::multi(
            vec![0, 1],
            vec![0, 1],
            vec![1.0, 0.0, 0.0, 1.0],
        )],
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::ElementCountMismatch {
            expected: 3,
            actual: 4,
        }
    );
}

/// Issue #1739: a Multi group packs each source in its own shape order and
/// scatters each column in the destination's, so equal element counts are
/// not enough; every layout must share one shape.
fn compile_multi_pair_shapes(
    dst_shapes: [Vec<usize>; 2],
    src_shapes: [Vec<usize>; 2],
) -> Result<TreeTransformStructure<f64>, OperationError> {
    let space = vec![4, 2];
    let src_structure = BlockStructure::packed_column_major(2, src_shapes).unwrap();
    let dst_structure = BlockStructure::packed_column_major(2, dst_shapes).unwrap();
    let src =
        TestTensor::<f64>::from_vec_with_structure(vec![1.0; 8], space.clone(), src_structure)
            .unwrap();
    let dst =
        TestTensor::<f64>::from_vec_with_structure(vec![0.0; 8], space, dst_structure).unwrap();
    TreeTransformStructure::compile_structures(
        dst.structure(),
        src.structure(),
        &[TreeTransformBlockSpec::multi(
            vec![0, 1],
            vec![0, 1],
            vec![1.0, 0.0, 0.0, 1.0],
        )],
    )
}

#[test]
fn tree_transform_rejects_incompatible_multi_tree_shapes() {
    assert_eq!(
        compile_multi_pair_shapes([vec![4, 1], vec![4, 1]], [vec![2, 2], vec![2, 2]]).unwrap_err(),
        OperationError::ShapeMismatch {
            dst: vec![4, 1],
            src: vec![2, 2],
        }
    );
}

#[test]
fn tree_transform_rejects_mixed_shape_multi_tree_group() {
    assert_eq!(
        compile_multi_pair_shapes([vec![2, 2], vec![2, 2]], [vec![2, 2], vec![4, 1]]).unwrap_err(),
        OperationError::ShapeMismatch {
            dst: vec![2, 2],
            src: vec![4, 1],
        }
    );
    assert_eq!(
        compile_multi_pair_shapes([vec![2, 2], vec![4, 1]], [vec![2, 2], vec![2, 2]]).unwrap_err(),
        OperationError::ShapeMismatch {
            dst: vec![2, 2],
            src: vec![4, 1],
        }
    );
    assert!(compile_multi_pair_shapes([vec![2, 2], vec![2, 2]], [vec![2, 2], vec![2, 2]]).is_ok());
}
