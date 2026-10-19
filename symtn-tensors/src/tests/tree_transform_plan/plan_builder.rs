use super::*;

pub(super) fn grouped_su2_test_pair(first_inner: usize) -> FusionTreePairKey {
    all_codomain_fusion_tree_test_key_for_rule(
        &SimpleSu2Rule,
        [4, 4, 4, 4],
        0,
        [false, false, false, false],
        [first_inner, 4],
        [1, 1, 1],
    )
}

#[test]
fn tree_transform_compile_keyed_pairs_tree_blocks_by_key_not_index_for_all_numeric_dtypes() {
    assert_tree_multi_keyed_dtype(
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0],
        vec![7020.0, 9240.0, 3510.0, 4620.0],
    );
    assert_tree_multi_keyed_dtype(
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0],
        vec![7020.0, 9240.0, 3510.0, 4620.0],
    );
    assert_tree_multi_keyed_dtype(
        vec![1_i32, 2, 3, 4, 5, 6],
        vec![10, 100, 1000, 20, 200, 2000],
        vec![7020, 9240, 3510, 4620],
    );
    assert_tree_multi_keyed_dtype(
        vec![1_i64, 2, 3, 4, 5, 6],
        vec![10, 100, 1000, 20, 200, 2000],
        vec![7020, 9240, 3510, 4620],
    );
    assert_tree_multi_keyed_dtype(
        vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(3.0, 0.0),
            Complex32::new(4.0, 0.0),
            Complex32::new(5.0, 0.0),
            Complex32::new(6.0, 0.0),
        ],
        vec![
            Complex32::new(10.0, 0.0),
            Complex32::new(100.0, 0.0),
            Complex32::new(1000.0, 0.0),
            Complex32::new(20.0, 0.0),
            Complex32::new(200.0, 0.0),
            Complex32::new(2000.0, 0.0),
        ],
        vec![
            Complex32::new(7020.0, 0.0),
            Complex32::new(9240.0, 0.0),
            Complex32::new(3510.0, 0.0),
            Complex32::new(4620.0, 0.0),
        ],
    );
    assert_tree_multi_keyed_dtype(
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(5.0, 0.0),
            Complex64::new(6.0, 0.0),
        ],
        vec![
            Complex64::new(10.0, 0.0),
            Complex64::new(100.0, 0.0),
            Complex64::new(1000.0, 0.0),
            Complex64::new(20.0, 0.0),
            Complex64::new(200.0, 0.0),
            Complex64::new(2000.0, 0.0),
        ],
        vec![
            Complex64::new(7020.0, 0.0),
            Complex64::new(9240.0, 0.0),
            Complex64::new(3510.0, 0.0),
            Complex64::new(4620.0, 0.0),
        ],
    );
}

#[test]
fn tree_transform_rejects_invalid_block_specs_at_compile_time() {
    let space = vec![4, 2];
    let structure = BlockStructure::packed_column_major(2, [vec![2, 2], vec![2, 2]]).unwrap();
    let src =
        TestTensor::<f64>::from_vec_with_structure(vec![1.0; 8], space.clone(), structure.clone())
            .unwrap();
    let dst = TestTensor::<f64>::from_vec_with_structure(vec![0.0; 8], space, structure).unwrap();

    let err = TreeTransformStructure::compile_structures(
        dst.structure(),
        src.structure(),
        &[TreeTransformBlockSpec::multi(
            vec![0, 1],
            vec![0, 1],
            vec![1.0, 2.0],
        )],
    )
    .unwrap_err();
    assert_eq!(
        err,
        OperationError::CoefficientCountMismatch {
            expected: 4,
            actual: 2,
        }
    );

    let err = TreeTransformStructure::compile_structures(
        dst.structure(),
        src.structure(),
        &[
            TreeTransformBlockSpec::single(0, 0, 1.0),
            TreeTransformBlockSpec::single(0, 1, 1.0),
        ],
    )
    .unwrap_err();
    assert_eq!(
        err,
        OperationError::DuplicateTransformDestination { dst_block: 0 }
    );
}

#[test]
fn tree_transform_compile_keyed_rejects_missing_tree_block_key() {
    let src_space = vec![2, 2];
    let dst_space = vec![2, 2];
    let key1 = BlockKey::opaque([1]);
    let key2 = BlockKey::opaque([2]);
    let src_structure = packed_fixture_structure(2, [(key1.clone(), vec![2, 2])]).unwrap();
    let dst_structure = packed_fixture_structure(2, [(key1.clone(), vec![2, 2])]).unwrap();
    let src =
        TestTensor::<f64>::from_vec_with_structure(vec![1.0; 4], src_space, src_structure).unwrap();
    let dst =
        TestTensor::<f64>::from_vec_with_structure(vec![0.0; 4], dst_space, dst_structure).unwrap();

    let err = TreeTransformStructure::compile_keyed_shared_structures(
        Arc::clone(dst.structure()),
        Arc::clone(src.structure()),
        &[TreeTransformKeyBlockSpec::single(key2.clone(), key1, 1.0)],
        false,
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::MissingBlockKey {
            key: Box::new(key2)
        }
    );
}

#[test]
fn tree_transform_group_block_spec_preserves_group_identity_and_ordered_keys() {
    let group_key = FusionTreeGroupKey::from_sector_ids([4, 4, 4, 4], [], [false; 4], []);
    let pair = grouped_su2_test_pair;
    let dst_key1 = pair(0);
    let dst_key2 = pair(2);
    let src_key = pair(4);
    let spec = TreeTransformGroupBlockSpec::try_multi(
        [dst_key1.clone(), dst_key2.clone()],
        [src_key.clone()],
        vec![2.0_f64, 3.0],
    )
    .unwrap();

    assert_eq!(&spec.group_key(), &group_key);
    assert_eq!(
        spec.group_key()
            .codomain_uncoupled()
            .iter()
            .map(|sector| sector.id())
            .collect::<Vec<_>>(),
        vec![4, 4, 4, 4]
    );
    assert!(spec.group_key().domain_uncoupled().is_empty());
    assert_eq!(spec.group_key().codomain_is_dual(), &[false; 4]);
    assert!(spec.group_key().domain_is_dual().is_empty());
    fn assert_categorical_keys(_: &[FusionTreePairKey]) {}
    assert_categorical_keys(spec.dst_keys());
    assert_categorical_keys(spec.src_keys());
    assert_eq!(spec.dst_keys(), &[dst_key1, dst_key2]);
    assert_eq!(spec.src_keys(), &[src_key]);
    assert_eq!(spec.recoupling_coefficients_dst_src(), &[2.0, 3.0]);
}

#[test]
fn grouped_spec_accepts_distinct_source_and_destination_cohorts() {
    let src1 = fusion_tree_test_key([10, 20], [30], 5, [false, true], [true]);
    let src2 = fusion_tree_test_key([10, 20], [30], 6, [false, true], [true]);
    let dst1 = fusion_tree_test_key([20, 10], [31], 7, [true, false], [false]);
    let dst2 = fusion_tree_test_key([20, 10], [31], 8, [true, false], [false]);

    let spec = TreeTransformGroupBlockSpec::try_multi(
        [dst1.clone(), dst2.clone()],
        [src1.clone(), src2.clone()],
        vec![1.0_f64, 2.0, 3.0, 4.0],
    )
    .unwrap();

    // What: the stored identity is derived from the authoritative source
    // cohort, while a transform may target a different coherent cohort.
    assert_eq!(&spec.group_key(), &src1.group_key());
    assert_ne!(&spec.group_key(), &dst1.group_key());
    assert_eq!(spec.dst_keys(), &[dst1, dst2]);
    assert_eq!(spec.src_keys(), &[src1, src2]);
}

#[test]
fn grouped_spec_rejects_mixed_source_and_destination_cohorts() {
    let src = fusion_tree_test_key([10, 20], [30], 5, [false, true], [true]);
    let other_src = fusion_tree_test_key([10, 21], [30], 6, [false, true], [true]);
    let dst = fusion_tree_test_key([20, 10], [31], 7, [true, false], [false]);
    let other_dst = fusion_tree_test_key([20, 11], [31], 8, [true, false], [false]);

    let src_err = TreeTransformGroupBlockSpec::try_multi(
        [dst.clone()],
        [src.clone(), other_src],
        vec![1.0_f64, 2.0],
    )
    .unwrap_err();
    let dst_err =
        TreeTransformGroupBlockSpec::try_multi([dst, other_dst], [src], vec![1.0_f64, 2.0])
            .unwrap_err();

    // What: one grouped matrix cannot silently mix either categorical cohort.
    assert_eq!(
        src_err,
        OperationError::FusionTreeGroupMismatch {
            tensor: "src",
            index: 1,
        }
    );
    assert_eq!(
        dst_err,
        OperationError::FusionTreeGroupMismatch {
            tensor: "dst",
            index: 1,
        }
    );
}

#[test]
fn grouped_spec_rejects_duplicate_source_columns_and_destination_rows() {
    let src = grouped_su2_test_pair(0);
    let dst = grouped_su2_test_pair(2);

    let src_err = TreeTransformGroupBlockSpec::try_multi(
        [dst.clone()],
        [src.clone(), src.clone()],
        vec![1.0_f64, 2.0],
    )
    .unwrap_err();
    let dst_err =
        TreeTransformGroupBlockSpec::try_multi([dst.clone(), dst], [src], vec![1.0_f64, 2.0])
            .unwrap_err();

    // What: each U[dst, src] row and column has one categorical basis identity,
    // while the same identity may still occur once on each opposite side.
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
fn grouped_spec_rejects_empty_sets_and_wrong_coefficient_count() {
    let src = grouped_su2_test_pair(0);
    let dst = grouped_su2_test_pair(2);

    let empty_dst = TreeTransformGroupBlockSpec::try_multi(
        Vec::<FusionTreePairKey>::new(),
        [src.clone()],
        Vec::<f64>::new(),
    )
    .unwrap_err();
    let empty_src = TreeTransformGroupBlockSpec::try_multi(
        [dst.clone()],
        Vec::<FusionTreePairKey>::new(),
        Vec::<f64>::new(),
    )
    .unwrap_err();
    let coefficient_err =
        TreeTransformGroupBlockSpec::try_multi([dst], [src], Vec::<f64>::new()).unwrap_err();

    // What: grouped matrices have at least one row and column and exactly one
    // row-major coefficient per destination/source pair.
    assert_eq!(empty_dst, OperationError::EmptyTransformBlock);
    assert_eq!(empty_src, OperationError::EmptyTransformBlock);
    assert_eq!(
        coefficient_err,
        OperationError::CoefficientCountMismatch {
            expected: 1,
            actual: 0,
        }
    );
}

#[test]
fn unique_tree_transform_plan_builder_creates_single_specs_in_source_order() {
    let key = |codomain| {
        FusionTreePairKey::try_pair_from_sector_ids(
            codomain,
            [1],
            1,
            [false, false],
            [false],
            [],
            [],
            [1],
            [],
        )
        .unwrap()
    };
    let src_key1 = key([1, 0]);
    let src_key2 = key([0, 1]);
    let dst_key1 = key([0, 1]);
    let dst_key2 = key([1, 0]);
    let src_tree1 = expect_tree_key(&src_key1);
    let src_tree2 = expect_tree_key(&src_key2);
    let dst_tree1 = expect_tree_key(&dst_key1);
    let dst_tree2 = expect_tree_key(&dst_key2);
    let src_structure = packed_fixture_structure(
        3,
        [
            (src_key1.clone(), vec![1, 1, 1]),
            (src_key2.clone(), vec![1, 1, 1]),
        ],
    )
    .unwrap();

    let plan = build_unique_tree_transform_group_plan(
        &UniqueZ2Rule,
        TreeTransformOperation::permute([1, 0], [2]),
        &src_structure,
        |src| {
            if src == &src_tree1 {
                Ok((dst_tree1.clone(), 2.0_f64))
            } else if src == &src_tree2 {
                Ok((dst_tree2.clone(), 3.0_f64))
            } else {
                panic!("unexpected source key {src:?}")
            }
        },
    )
    .unwrap();

    assert_eq!(plan.specs().len(), 2);
    assert_eq!(&plan.specs()[0].group_key(), &src_tree1.group_key());
    assert_eq!(plan.specs()[0].src_keys(), &[src_key1]);
    assert_eq!(plan.specs()[0].dst_keys(), &[dst_key1]);
    assert_eq!(plan.specs()[0].recoupling_coefficients_dst_src(), &[2.0]);
    assert_eq!(&plan.specs()[1].group_key(), &src_tree2.group_key());
    assert_eq!(plan.specs()[1].src_keys(), &[src_key2]);
    assert_eq!(plan.specs()[1].dst_keys(), &[dst_key2]);
    assert_eq!(plan.specs()[1].recoupling_coefficients_dst_src(), &[3.0]);
}

#[test]
fn single_output_unique_tree_transform_helper_rejects_simple_fusion() {
    let src_key = fusion_tree_test_key([1, 1, 1], [1], 1, [false, false, false], [false]);
    let src_structure = packed_fixture_structure(4, [(src_key, vec![1, 1, 1, 1])]).unwrap();
    let operation = TreeTransformOperation::transpose([2, 1, 0], [0]);

    let err = build_unique_tree_transform_group_plan(
        &SimpleSu2Rule,
        operation.clone(),
        &src_structure,
        |_| -> Result<(FusionTreePairKey, f64), OperationError> {
            unreachable!("non-Unique fusion must be rejected before transforming keys")
        },
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::UnsupportedFusionStyle {
            operation: Box::new(operation),
            style: FusionStyleKind::Simple,
        }
    );
}

#[test]
fn tree_transform_plan_builder_accepts_simple_multi_destination_callback() {
    let src_key0 = all_codomain_fusion_tree_test_key_for_rule(
        &SimpleSu2Rule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key_for_rule(
        &SimpleSu2Rule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let src_tree0 = expect_tree_key(&src_key0);
    let src_tree1 = expect_tree_key(&src_key1);
    let src_structure = packed_fixture_structure(
        4,
        [
            (src_key0.clone(), vec![1, 1, 1, 1]),
            (src_key1.clone(), vec![1, 1, 1, 1]),
        ],
    )
    .unwrap();
    let operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);

    let plan = build_tree_transform_group_plan(&SimpleSu2Rule, operation, &src_structure, |src| {
        if src == &src_tree0 {
            Ok(vec![
                (src_tree0.clone(), 0.5_f64),
                (src_tree1.clone(), 0.866_025_403_784_438_6),
            ])
        } else if src == &src_tree1 {
            Ok(vec![
                (src_tree0.clone(), 0.866_025_403_784_438_6),
                (src_tree1.clone(), -0.5),
            ])
        } else {
            panic!("unexpected source key {src:?}")
        }
    })
    .unwrap();

    assert_eq!(plan.specs().len(), 1);
    let spec = &plan.specs()[0];
    assert_eq!(spec.src_keys(), &[src_key0.clone(), src_key1.clone()]);
    assert_eq!(spec.dst_keys(), &[src_key0, src_key1]);
    assert_eq!(
        spec.recoupling_coefficients_dst_src(),
        &[0.5, 0.866_025_403_784_438_6, 0.866_025_403_784_438_6, -0.5]
    );
}

#[test]
fn tree_transform_plan_builder_lowers_injective_singleton_rows_in_source_order() {
    // What: a nonidentity monomial group is represented by ordered direct
    // specs even when its coefficients are neither inferred nor unit-valued.
    let src_key0 = all_codomain_fusion_tree_test_key_for_rule(
        &SimpleSu2Rule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key_for_rule(
        &SimpleSu2Rule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let src_tree0 = expect_tree_key(&src_key0);
    let src_tree1 = expect_tree_key(&src_key1);
    let src_structure = packed_fixture_structure(
        4,
        [
            (src_key0.clone(), vec![1, 1, 1, 1]),
            (src_key1.clone(), vec![1, 1, 1, 1]),
        ],
    )
    .unwrap();

    let plan = build_tree_transform_group_plan(
        &SimpleSu2Rule,
        TreeTransformOperation::braid([1, 0, 2, 3], [], [0, 1, 2, 3], []),
        &src_structure,
        |src| {
            if src == &src_tree0 {
                Ok(vec![(src_tree1.clone(), -2.0_f64)])
            } else if src == &src_tree1 {
                Ok(vec![(src_tree0.clone(), 3.0_f64)])
            } else {
                panic!("unexpected source key {src:?}")
            }
        },
    )
    .unwrap();

    assert_eq!(plan.specs().len(), 2);
    assert_eq!(plan.specs()[0].src_keys(), std::slice::from_ref(&src_key0));
    assert_eq!(plan.specs()[0].dst_keys(), std::slice::from_ref(&src_key1));
    assert_eq!(plan.specs()[0].recoupling_coefficients_dst_src(), &[-2.0]);
    assert_eq!(plan.specs()[1].src_keys(), std::slice::from_ref(&src_key1));
    assert_eq!(plan.specs()[1].dst_keys(), std::slice::from_ref(&src_key0));
    assert_eq!(plan.specs()[1].recoupling_coefficients_dst_src(), &[3.0]);
    assert!(plan
        .specs()
        .iter()
        .all(|spec| spec.source_axes() == Some([1, 0, 2, 3].as_slice())));

    let space = vec![1, 1, 1, 1];
    let src = TestTensor::<f64>::from_vec_with_structure(
        vec![5.0, 7.0],
        space.clone(),
        src_structure.clone(),
    )
    .unwrap();
    let mut dst =
        TestTensor::<f64>::from_vec_with_structure(vec![11.0, 13.0], space, src_structure.clone())
            .unwrap();
    let compiled = plan
        .compile_structures(&src_structure, &src_structure)
        .unwrap();
    assert!(!compiled.has_pack_gemm_scatter_blocks());
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();

    replay_into(
        &mut backend,
        &mut workspace,
        &compiled,
        &mut dst,
        &src,
        2.0,
        3.0,
    )
    .unwrap();

    // dst0 receives 3*src1 and dst1 receives -2*src0.
    assert_eq!(dst.data(), &[75.0, 19.0]);
    assert_eq!(
        (workspace.source_len(), workspace.destination_len()),
        (0, 0)
    );
}

#[test]
fn tree_transform_plan_builder_keeps_destination_collisions_in_multi() {
    // What: singleton rows are not direct when two sources contribute to the
    // same destination, because replay must preserve their sum.
    let src_key0 = all_codomain_fusion_tree_test_key_for_rule(
        &SimpleSu2Rule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key_for_rule(
        &SimpleSu2Rule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let dst_tree = expect_tree_key(&src_key0);
    let src_structure = packed_fixture_structure(
        4,
        [
            (src_key0.clone(), vec![1, 1, 1, 1]),
            (src_key1.clone(), vec![1, 1, 1, 1]),
        ],
    )
    .unwrap();

    let plan = build_tree_transform_group_plan(
        &SimpleSu2Rule,
        TreeTransformOperation::braid([1, 0, 2, 3], [], [0, 1, 2, 3], []),
        &src_structure,
        |_| Ok(vec![(dst_tree.clone(), 1.0_f64)]),
    )
    .unwrap();

    assert_eq!(plan.specs().len(), 1);
    assert_eq!(plan.specs()[0].src_keys(), &[src_key0, src_key1]);
    assert_eq!(plan.specs()[0].dst_keys().len(), 1);
}

#[test]
fn su2_first_pair_braid_lowers_nonidentity_monomial_group_to_singles() {
    // What: the first-vertex SU(2) R move is direct for every fusion channel,
    // while preserving the channel-dependent coefficients returned by core.
    let keys = [[0, 1], [2, 1]].map(|inner| {
        all_codomain_fusion_tree_test_key_for_rule(
            &SU2FusionRule,
            [1, 1, 1, 1],
            0,
            [false, false, false, false],
            inner,
            [1, 1, 1],
        )
    });
    let structure =
        packed_fixture_structure(4, keys.iter().cloned().map(|key| (key, vec![1usize; 4])))
            .unwrap();
    let operation = TreeTransformOperation::braid([1, 0, 2, 3], [], [0, 1, 2, 3], []);

    let plan =
        build_all_codomain_tree_transform_group_plan(&SU2FusionRule, operation.clone(), &structure)
            .unwrap();

    assert_eq!(plan.specs().len(), keys.len());
    for ((spec, key), expected_coefficient) in plan.specs().iter().zip(&keys).zip([-1.0, 1.0]) {
        assert_eq!(spec.src_keys(), std::slice::from_ref(key));
        assert_eq!(spec.dst_keys(), std::slice::from_ref(key));
        assert_eq!(
            spec.recoupling_coefficients_dst_src(),
            &[expected_coefficient]
        );
        assert_eq!(spec.source_axes(), Some([1, 0, 2, 3].as_slice()));
    }
    assert!(!plan
        .compile_structures(&structure, &structure)
        .unwrap()
        .has_pack_gemm_scatter_blocks());

    let serial =
        crate::tree_transform::build_multiplicity_free_all_codomain_tree_transform_group_plan(
            &SU2FusionRule,
            operation.clone(),
            &structure,
        )
        .unwrap();

    // What: direct replay honors alpha/beta and overwrite on strided blocks
    // without touching storage padding or allocating pack/GEMM/scatter jobs.
    let padded_structure = BlockStructure::from_blocks_with_rank(
        4,
        vec![
            BlockSpec::with_key(
                keys[0].clone().into(),
                vec![2, 2, 1, 1],
                vec![1, 2, 4, 4],
                1,
            )
            .unwrap(),
            BlockSpec::with_key(
                keys[1].clone().into(),
                vec![2, 2, 1, 1],
                vec![1, 2, 4, 4],
                7,
            )
            .unwrap(),
        ],
    )
    .unwrap();
    let space = vec![2, 2, 1, 1];
    let src = TestTensor::<f64>::from_vec_with_structure(
        vec![99.0, 1.0, 2.0, 3.0, 4.0, 98.0, 97.0, 9.0, 10.0, 11.0, 12.0],
        space.clone(),
        padded_structure.clone(),
    )
    .unwrap();
    let mut dst = TestTensor::<f64>::from_vec_with_structure(
        vec![91.0, 5.0, 6.0, 7.0, 8.0, 92.0, 93.0, 13.0, 14.0, 15.0, 16.0],
        space,
        padded_structure.clone(),
    )
    .unwrap();
    let compiled = serial
        .compile_structures(&padded_structure, &padded_structure)
        .unwrap();
    assert!(!compiled.has_pack_gemm_scatter_blocks());
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();
    replay_into(
        &mut backend,
        &mut workspace,
        &compiled,
        &mut dst,
        &src,
        2.0,
        3.0,
    )
    .unwrap();
    assert_eq!(
        dst.data(),
        &[91.0, 13.0, 12.0, 17.0, 16.0, 92.0, 93.0, 57.0, 64.0, 65.0, 72.0]
    );
    assert_eq!(
        (workspace.source_len(), workspace.destination_len()),
        (0, 0)
    );

    dst.data_mut().fill(f64::NAN);
    backend.set_recoupling_threads(std::num::NonZeroUsize::new(4).unwrap());
    backend.set_transform_parallel_min_len(0);
    replay_overwrite_into(
        &mut backend,
        &mut workspace,
        &compiled,
        &mut dst,
        &src,
        -2.0,
    )
    .unwrap();
    assert!(dst.data()[0].is_nan());
    assert_eq!(&dst.data()[1..5], &[2.0, 6.0, 4.0, 8.0]);
    assert!(dst.data()[5].is_nan() && dst.data()[6].is_nan());
    assert_eq!(&dst.data()[7..11], &[-18.0, -22.0, -20.0, -24.0]);
}

#[test]
fn nested_fz2_u1_su2_first_pair_braid_preserves_product_phases_in_singles() {
    // What: direct lowering preserves the product rule's fermionic sign times
    // the SU(2) channel phase instead of synthesizing a diagonal coefficient.
    let left_rule = FpU1Rule::default();
    let rule = FpU1Su2Rule::default();
    let left_sector =
        |parity, charge| left_rule.encode_component_ids(parity, U1Irrep::new(charge).sector_id());
    let sector = |parity, charge, twice_spin| {
        rule.encode_component_ids(
            left_sector(parity, charge),
            SU2Irrep::from_twice_spin(twice_spin).sector_id(),
        )
    };
    let odd_half = sector(SectorId::new(1), 0, 1);
    let even_singlet = sector(SectorId::new(0), 0, 0);
    let even_triplet = sector(SectorId::new(0), 0, 2);
    let odd_half_inner = sector(SectorId::new(1), 0, 1);
    let keys = [even_singlet, even_triplet].map(|first_inner| {
        FusionTreePairKey::pair(
            FusionTreeKey::try_new_for_rule(
                &rule,
                [odd_half; 4],
                even_singlet,
                [false; 4],
                [first_inner, odd_half_inner],
                [MultiplicityIndex::ONE; 3],
            )
            .unwrap(),
            FusionTreeKey::try_new_for_rule(&rule, [], even_singlet, [], [], []).unwrap(),
        )
    });
    let structure =
        packed_fixture_structure(4, keys.iter().cloned().map(|key| (key, vec![1usize; 4])))
            .unwrap();

    let plan = build_all_codomain_tree_transform_group_plan(
        &rule,
        TreeTransformOperation::braid([1, 0, 2, 3], [], [0, 1, 2, 3], []),
        &structure,
    )
    .unwrap();

    assert_eq!(plan.specs().len(), 2);
    for ((spec, key), expected_coefficient) in plan.specs().iter().zip(&keys).zip([1.0, -1.0]) {
        assert_eq!(spec.src_keys(), std::slice::from_ref(key));
        assert_eq!(spec.dst_keys(), std::slice::from_ref(key));
        assert_eq!(
            spec.recoupling_coefficients_dst_src(),
            &[expected_coefficient]
        );
    }
    assert!(!plan
        .compile_structures(&structure, &structure)
        .unwrap()
        .has_pack_gemm_scatter_blocks());
}

#[test]
fn multiplicity_free_su2_plan_builder_creates_generic_recoupling_block() {
    let src_key0 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let src_structure = packed_fixture_structure(
        4,
        [
            (src_key0.clone(), vec![1, 1, 1, 1]),
            (src_key1.clone(), vec![1, 1, 1, 1]),
        ],
    )
    .unwrap();
    let operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);

    let plan =
        build_all_codomain_tree_transform_group_plan(&SU2FusionRule, operation, &src_structure)
            .unwrap();

    assert_eq!(plan.specs().len(), 1);
    let spec = &plan.specs()[0];
    assert_eq!(spec.src_keys(), &[src_key0.clone(), src_key1.clone()]);
    assert_eq!(spec.dst_keys(), &[src_key0, src_key1]);
    let expected = [0.5, 0.866_025_403_784_438_6, 0.866_025_403_784_438_6, -0.5];
    assert_eq!(spec.recoupling_coefficients_dst_src().len(), expected.len());
    for (&actual, expected) in spec.recoupling_coefficients_dst_src().iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "coefficient {actual} != {expected}"
        );
    }

    let src_space = vec![1, 1, 1, 1];
    let dst_space = vec![1, 1, 1, 1];
    let src = TestTensor::<f64>::from_vec_with_structure(
        vec![10.0, 20.0],
        src_space,
        src_structure.clone(),
    )
    .unwrap();
    let mut dst = TestTensor::<f64>::from_vec_with_structure(
        vec![0.0, 0.0],
        dst_space,
        src_structure.clone(),
    )
    .unwrap();
    let structure = plan
        .compile_structures(&src_structure, &src_structure)
        .unwrap();
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();

    replay_into(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &src,
        1.0,
        0.0,
    )
    .unwrap();

    assert!(structure.has_pack_gemm_scatter_blocks());
    assert!((dst.data()[0] - 22.320_508_075_688_77).abs() < 1.0e-12);
    assert!((dst.data()[1] + 1.339_745_962_155_612_7).abs() < 1.0e-12);
}

#[test]
fn tree_pair_transform_public_helper_executes_su2_recoupling_block() {
    let src_key0 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let structure = packed_fixture_structure(
        4,
        [
            (src_key0.clone(), vec![1, 1, 1, 1]),
            (src_key1.clone(), vec![1, 1, 1, 1]),
        ],
    )
    .unwrap();
    let src_space = vec![1, 1, 1, 1];
    let dst_space = vec![1, 1, 1, 1];
    let src =
        TestTensor::<f64>::from_vec_with_structure(vec![10.0, 20.0], src_space, structure.clone())
            .unwrap();
    let mut dst =
        TestTensor::<f64>::from_vec_with_structure(vec![0.0, 0.0], dst_space, structure).unwrap();
    let operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);

    let compiled = uncached_tree_pair_structure(&SU2FusionRule, operation, &dst, &src).unwrap();
    assert!(compiled.has_pack_gemm_scatter_blocks());
    eager_tree_transform_into(
        &SU2FusionRule,
        TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []),
        &mut dst,
        &src,
        1.0,
        0.0,
    )
    .unwrap();

    assert!((dst.data()[0] - 22.320_508_075_688_77).abs() < 1.0e-12);
    assert!((dst.data()[1] + 1.339_745_962_155_612_7).abs() < 1.0e-12);
}

#[test]
fn tree_transform_recoupling_replays_complex_data_with_real_structural_coefficients() {
    let src_key0 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let structure = packed_fixture_structure(
        4,
        [
            (src_key0.clone(), vec![1, 1, 1, 1]),
            (src_key1.clone(), vec![1, 1, 1, 1]),
        ],
    )
    .unwrap();
    let src_space = vec![1, 1, 1, 1];
    let dst_space = vec![1, 1, 1, 1];
    let src = TestTensor::<Complex64>::from_vec_with_structure(
        vec![Complex64::new(10.0, 1.0), Complex64::new(20.0, -2.0)],
        src_space,
        structure.clone(),
    )
    .unwrap();
    let mut dst = TestTensor::<Complex64>::from_vec_with_structure(
        vec![Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
        dst_space,
        structure.clone(),
    )
    .unwrap();
    let operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
    let compiled = uncached_tree_pair_structure(&SU2FusionRule, operation, &dst, &src).unwrap();
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();

    replay_into(
        &mut backend,
        &mut workspace,
        &compiled,
        &mut dst,
        &src,
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap();
    let first = dst.data().to_vec();
    assert_eq!(workspace.source_len(), 2);
    assert_eq!(workspace.destination_len(), 2);

    dst.data_mut().fill(Complex64::new(0.0, 0.0));
    replay_into(
        &mut backend,
        &mut workspace,
        &compiled,
        &mut dst,
        &src,
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap();
    assert_eq!(dst.data(), first.as_slice());
    assert_eq!(workspace.source_len(), 2);
    assert_eq!(workspace.destination_len(), 2);
}

#[test]
fn tree_transform_structure_sorts_replay_blocks_by_tensorkit_weight() {
    let dst_structure =
        BlockStructure::packed_column_major(1, [vec![1], vec![3], vec![3]]).unwrap();
    let src_structure =
        BlockStructure::packed_column_major(1, [vec![1], vec![3], vec![3]]).unwrap();
    let structure = TreeTransformStructure::compile_structures(
        &dst_structure,
        &src_structure,
        &[
            TreeTransformBlockSpec::single(0, 0, 1.0),
            TreeTransformBlockSpec::multi(vec![1, 2], vec![1, 2], vec![1.0, 0.0, 0.0, 1.0]),
        ],
    )
    .unwrap();

    assert_eq!(structure.replay_weights(), vec![12, 1]);
}

#[test]
fn tree_transform_recoupling_plan_groups_same_shape_multi_blocks() {
    let dst_structure = BlockStructure::packed_column_major(
        1,
        [vec![2], vec![2], vec![1], vec![1], vec![1], vec![1]],
    )
    .unwrap();
    let src_structure = dst_structure.clone();
    let structure = TreeTransformStructure::compile_structures(
        &dst_structure,
        &src_structure,
        &[
            TreeTransformBlockSpec::multi(vec![2, 3], vec![2, 3], vec![1.0, 0.0, 0.0, 1.0]),
            TreeTransformBlockSpec::multi(vec![0, 1], vec![0, 1], vec![1.0, 0.0, 0.0, 1.0]),
            TreeTransformBlockSpec::multi(vec![4, 5], vec![4, 5], vec![1.0, 0.0, 0.0, 1.0]),
        ],
    )
    .unwrap();

    assert_eq!(structure.replay_weights(), vec![8, 4, 4]);
    let plan = structure.recoupling_plan();
    assert_eq!(plan.block_indices(), &[1, 2, 0]);
    let jobs = plan.jobs();
    assert_eq!(jobs.len(), 3);
    assert_eq!((jobs[0].rows, jobs[0].contracted, jobs[0].cols), (1, 2, 2));
    assert_eq!((jobs[1].rows, jobs[1].contracted, jobs[1].cols), (1, 2, 2));
    assert_eq!((jobs[2].rows, jobs[2].contracted, jobs[2].cols), (2, 2, 2));
    assert_eq!(jobs[1].lhs_offset - jobs[0].lhs_offset, 2);
    assert_eq!(jobs[1].rhs_offset - jobs[0].rhs_offset, 4);
    assert_eq!(jobs[1].dst_offset - jobs[0].dst_offset, 2);
}

#[test]
fn tree_transform_structure_replays_su2_recoupling_without_recompiling() {
    let src_key0 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let src_key1 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let block_structure = packed_fixture_structure(
        4,
        [
            (src_key0.clone(), vec![1, 1, 1, 1]),
            (src_key1.clone(), vec![1, 1, 1, 1]),
        ],
    )
    .unwrap();
    let src_space = vec![1, 1, 1, 1];
    let dst_space = vec![1, 1, 1, 1];
    let mut src = TestTensor::<f64>::from_vec_with_structure(
        vec![10.0, 20.0],
        src_space,
        block_structure.clone(),
    )
    .unwrap();
    let mut dst =
        TestTensor::<f64>::from_vec_with_structure(vec![0.0, 0.0], dst_space, block_structure)
            .unwrap();
    let operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
    let structure = uncached_tree_pair_structure(&SU2FusionRule, operation, &dst, &src).unwrap();
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();
    let expected = |initial: [f64; 2], source: [f64; 2], alpha: f64, beta: f64| {
        let c = 0.866_025_403_784_438_6;
        [
            beta * initial[0] + alpha * (0.5 * source[0] + c * source[1]),
            beta * initial[1] + alpha * (c * source[0] - 0.5 * source[1]),
        ]
    };

    assert!(structure.has_pack_gemm_scatter_blocks());
    replay_into(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &src,
        1.0,
        0.0,
    )
    .unwrap();
    let expected_first = expected([0.0, 0.0], [10.0, 20.0], 1.0, 0.0);
    assert!((dst.data()[0] - expected_first[0]).abs() < 1.0e-12);
    assert!((dst.data()[1] - expected_first[1]).abs() < 1.0e-12);
    assert_eq!(workspace.source_len(), 2);
    assert_eq!(workspace.destination_len(), 2);

    src.data_mut().copy_from_slice(&[3.0, -4.0]);
    dst.data_mut().copy_from_slice(&[1.0, 2.0]);
    replay_into(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &src,
        2.0,
        -1.0,
    )
    .unwrap();
    let expected_second = expected([1.0, 2.0], [3.0, -4.0], 2.0, -1.0);
    assert!((dst.data()[0] - expected_second[0]).abs() < 1.0e-12);
    assert!((dst.data()[1] - expected_second[1]).abs() < 1.0e-12);
}

#[test]
fn parallel_plan_compile_matches_serial_plan() {
    // What: shared group compilation produces the same transformer,
    // source order, and group order.
    use crate::tree_transform::{
        build_tree_pair_transform_group_plan_validated_with_threads,
        validate_multiplicity_free_tree_pair_preflight,
    };

    let key = |uncoupled: [usize; 4], inner: [usize; 2]| {
        all_codomain_fusion_tree_test_key_for_rule(
            &SU2FusionRule,
            uncoupled,
            0,
            [false, false, false, false],
            inner,
            [1, 1, 1],
        )
    };
    let keys = [
        key([1, 1, 1, 1], [0, 1]),
        key([1, 1, 1, 1], [2, 1]),
        key([1, 3, 3, 1], [2, 1]),
        key([1, 3, 3, 1], [4, 1]),
    ];
    let src_structure = packed_fixture_structure(
        4,
        keys.iter().map(|key| (key.clone(), vec![1usize, 1, 1, 1])),
    )
    .unwrap();
    let operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
    let build = |threads: usize| {
        let proof = validate_multiplicity_free_tree_pair_preflight(
            &SU2FusionRule,
            &operation,
            &src_structure,
        )
        .unwrap();
        build_tree_pair_transform_group_plan_validated_with_threads(
            &proof,
            operation.clone(),
            threads,
        )
        .unwrap()
    };

    let serial_plan = build(1);
    let parallel_plan = build(8);
    assert_eq!(parallel_plan, serial_plan);
}

#[test]
fn all_codomain_repeated_axis_is_an_invalid_permutation() {
    let keys = [1, 2].map(|sector| {
        FusionTreePairKey::pair(
            FusionTreeKey::try_new_for_rule(
                &SU2FusionRule,
                [SectorId::new(sector), SectorId::new(sector)],
                SectorId::new(0),
                [false, false],
                [],
                [MultiplicityIndex::ONE],
            )
            .unwrap(),
            empty_fusion_tree(),
        )
    });
    let structure =
        packed_fixture_structure(2, keys.into_iter().map(|key| (key, vec![1, 1]))).unwrap();
    let invalid_operation = TreeTransformOperation::braid([0, 0], [], [0, 1], []);
    let result =
        build_all_codomain_tree_transform_group_plan(&SU2FusionRule, invalid_operation, &structure);
    assert!(matches!(
        result,
        Err(OperationError::InvalidPermutation { .. })
    ));
}

#[test]
fn staged_group_scheduler_builds_bounded_balanced_contiguous_batches() {
    use crate::tree_transform::partition_staged_groups_for_test;

    for (group_count, threads, expected_sizes) in [
        (3, 2, vec![2, 1]),
        (5, 2, vec![3, 2]),
        (3, 8, vec![1, 1, 1]),
    ] {
        let batches = partition_staged_groups_for_test((0..group_count).collect(), threads);
        // What: awkward group/thread ratios still create exactly the bounded
        // task count rather than relying on Rayon's recursive split heuristic.
        assert_eq!(
            batches.iter().map(Vec::len).collect::<Vec<_>>(),
            expected_sizes
        );
        assert_eq!(
            batches.into_iter().flatten().collect::<Vec<_>>(),
            (0..group_count).collect::<Vec<_>>()
        );
    }
}

#[test]
fn split_only_repartition_plan_matches_serial_and_parallel_builders() {
    // What: a SU(2) 2|2 -> 3|1 split-only braid compiles to the direct
    // repartition row independently of the compile worker count.
    use crate::tree_transform::{
        build_tree_pair_transform_group_plan_validated_with_threads,
        validate_multiplicity_free_tree_pair_preflight,
    };

    let source = FusionTreePairKey::try_pair_from_sector_ids(
        [1, 2],
        [2, 1],
        1,
        [false, true],
        [true, false],
        [],
        [],
        [1],
        [1],
    )
    .unwrap();
    let source_key = BlockKey::from(source.clone());
    let source_structure = packed_fixture_structure(4, [(source_key, vec![1usize; 4])]).unwrap();
    let operation = TreeTransformOperation::braid([0, 1, 3], [2], [0, 1], [2, 3]);
    let expected = multiplicity_free_repartition_tree_pair(&SU2FusionRule, &source, 3).unwrap();
    let build = |threads| {
        let proof = validate_multiplicity_free_tree_pair_preflight(
            &SU2FusionRule,
            &operation,
            &source_structure,
        )
        .unwrap();
        build_tree_pair_transform_group_plan_validated_with_threads(
            &proof,
            operation.clone(),
            threads,
        )
        .unwrap()
    };

    let serial = build(1);
    let parallel = build(8);
    assert_eq!(parallel, serial);

    let spec = &serial.specs()[0];
    assert_eq!(spec.dst_keys(), &[expected[0].0.clone()]);
    assert!((spec.recoupling_coefficients_dst_src()[0] - expected[0].1).abs() < 1.0e-12);
}

#[test]
fn identity_group_plan_lowers_each_su2_tree_to_a_direct_single() {
    // What: an identity operation over a multi-tree SU2 fusion group compiles
    // to independent direct copies for every supported facade and worker count.
    use crate::tree_transform::{
        build_tree_pair_transform_group_plan_validated_with_threads,
        validate_multiplicity_free_tree_pair_preflight,
    };

    let key = |inner: [usize; 2]| {
        all_codomain_fusion_tree_test_key_for_rule(
            &SU2FusionRule,
            [3, 3, 3, 3],
            0,
            [false, false, false, false],
            inner,
            [1, 1, 1],
        )
    };
    let keys = [key([0, 3]), key([2, 3]), key([4, 3]), key([6, 3])];
    let structure =
        packed_fixture_structure(4, keys.iter().map(|key| (key.clone(), vec![1usize; 4]))).unwrap();
    let operation = TreeTransformOperation::braid([0, 1, 2, 3], [], [17, 3, 11, 5], []);
    let build = |threads| {
        let proof =
            validate_multiplicity_free_tree_pair_preflight(&SU2FusionRule, &operation, &structure)
                .unwrap();
        build_tree_pair_transform_group_plan_validated_with_threads(
            &proof,
            operation.clone(),
            threads,
        )
        .unwrap()
    };

    let serial = build(1);
    let parallel = build(8);
    assert_eq!(parallel, serial);
    assert_eq!(serial.specs().len(), keys.len());
    for spec in serial.specs() {
        assert_eq!(spec.src_keys().len(), 1);
        assert_eq!(spec.dst_keys(), spec.src_keys());
        assert_eq!(spec.recoupling_coefficients_dst_src(), &[1.0]);
        assert_eq!(spec.source_axes(), Some([0, 1, 2, 3].as_slice()));
    }
    assert!(!serial
        .compile_structures(&structure, &structure)
        .unwrap()
        .has_pack_gemm_scatter_blocks());

    let transpose = build_tree_pair_transform_group_plan(
        &SU2FusionRule,
        TreeTransformOperation::transpose([0, 1, 2, 3], []),
        &structure,
    )
    .unwrap();
    assert!(transpose.specs().iter().all(|spec| {
        spec.src_keys().len() == 1
            && spec.dst_keys() == spec.src_keys()
            && spec.recoupling_coefficients_dst_src() == [1.0]
    }));
    assert!(!transpose
        .compile_structures(&structure, &structure)
        .unwrap()
        .has_pack_gemm_scatter_blocks());

    let all_codomain =
        build_all_codomain_tree_transform_group_plan(&SU2FusionRule, operation, &structure)
            .unwrap();
    assert_eq!(all_codomain, serial);

    let tensor_space = vec![1, 1, 1, 1];
    let src = TestTensor::<f64>::from_vec_with_structure(
        vec![1.0, 2.0, 3.0, 4.0],
        tensor_space.clone(),
        structure.clone(),
    )
    .unwrap();
    let mut dst = TestTensor::<f64>::from_vec_with_structure(
        vec![5.0, 6.0, 7.0, 8.0],
        tensor_space,
        structure,
    )
    .unwrap();
    let compiled = serial
        .compile_structures(dst.structure(), src.structure())
        .unwrap();
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();
    replay_into(
        &mut backend,
        &mut workspace,
        &compiled,
        &mut dst,
        &src,
        2.0,
        3.0,
    )
    .unwrap();
    assert_eq!(dst.data(), &[17.0, 22.0, 27.0, 32.0]);

    dst.data_mut().fill(f64::NAN);
    backend.set_recoupling_threads(std::num::NonZeroUsize::new(4).unwrap());
    backend.set_transform_parallel_min_len(0);
    replay_overwrite_into(
        &mut backend,
        &mut workspace,
        &compiled,
        &mut dst,
        &src,
        -2.0,
    )
    .unwrap();
    assert_eq!(dst.data(), &[-2.0, -4.0, -6.0, -8.0]);
}

#[test]
fn same_split_transpose_is_direct_for_real_tree_pairs_but_split_change_is_not() {
    // What: exact 2|1 fZ2 and SU2 transposes preserve the source tree with a
    // unit coefficient, while a cyclic split change retains recoupling.
    let fz2_source = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &FermionParityFusionRule,
            [SectorId::new(1), SectorId::new(0)],
            SectorId::new(1),
            [false, true],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(
            &FermionParityFusionRule,
            [SectorId::new(1)],
            SectorId::new(1),
            [true],
            [],
            [],
        )
        .unwrap(),
    );
    let fz2_key = fz2_source;
    let fz2_structure = packed_fixture_structure(3, [(fz2_key.clone(), vec![1, 1, 1])]).unwrap();
    let exact = TreeTransformOperation::transpose([0, 1], [2]);
    let fz2_plan = build_tree_pair_transform_group_plan(
        &FermionParityFusionRule,
        exact.clone(),
        &fz2_structure,
    )
    .unwrap();
    assert_eq!(fz2_plan.specs().len(), 1);
    assert_eq!(
        fz2_plan.specs()[0].src_keys(),
        std::slice::from_ref(&fz2_key)
    );
    assert_eq!(fz2_plan.specs()[0].dst_keys(), &[fz2_key]);
    assert_eq!(
        fz2_plan.specs()[0].recoupling_coefficients_dst_src(),
        &[1.0]
    );
    assert!(!fz2_plan
        .compile_structures(&fz2_structure, &fz2_structure)
        .unwrap()
        .has_pack_gemm_scatter_blocks());

    let one = SU2Irrep::from_twice_spin(2).sector_id();
    let su2_source = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &SU2FusionRule,
            [one, one],
            one,
            [false, false],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(&SU2FusionRule, [one], one, [true], [], []).unwrap(),
    );
    let su2_rows = symtn_core::testing::multiplicity_free_transpose_tree_pair(
        &SU2FusionRule,
        &su2_source,
        &[0, 1],
        &[2],
    )
    .unwrap();
    assert_eq!(su2_rows, vec![(su2_source.clone(), 1.0)]);

    let su2_key = su2_source;
    let su2_structure = packed_fixture_structure(3, [(su2_key.clone(), vec![1, 1, 1])]).unwrap();
    let su2_plan =
        build_tree_pair_transform_group_plan(&SU2FusionRule, exact, &su2_structure).unwrap();
    assert_eq!(
        su2_plan.specs()[0].src_keys(),
        std::slice::from_ref(&su2_key)
    );
    assert_eq!(su2_plan.specs()[0].dst_keys(), &[su2_key]);
    assert_eq!(
        su2_plan.specs()[0].recoupling_coefficients_dst_src(),
        &[1.0]
    );
    assert!(!su2_plan
        .compile_structures(&su2_structure, &su2_structure)
        .unwrap()
        .has_pack_gemm_scatter_blocks());

    let control_keys = [0, 2, 4].map(|inner| {
        BlockKey::from(FusionTreePairKey::pair(
            FusionTreeKey::try_new_for_rule(
                &SU2FusionRule,
                [one, one, one],
                one,
                [false, false, false],
                [SectorId::new(inner)],
                [MultiplicityIndex::ONE, MultiplicityIndex::ONE],
            )
            .unwrap(),
            FusionTreeKey::try_new_for_rule(&SU2FusionRule, [one], one, [true], [], []).unwrap(),
        ))
    });
    let control_src = packed_fixture_structure(
        4,
        control_keys
            .iter()
            .cloned()
            .map(|key| (key, vec![1, 1, 1, 1])),
    )
    .unwrap();
    let split_change = TreeTransformOperation::transpose([3, 0, 1], [2]);
    assert!(!split_change.is_identity_for(3, 1));
    let control =
        build_tree_pair_transform_group_plan(&SU2FusionRule, split_change, &control_src).unwrap();
    assert_eq!(control.specs()[0].src_keys().len(), control_keys.len());
    let dst_structure = packed_fixture_structure(
        4,
        control
            .specs()
            .iter()
            .flat_map(|spec| spec.dst_keys().iter().cloned())
            .map(|key| (key, vec![1, 1, 1, 1])),
    )
    .unwrap();
    assert!(control
        .compile_structures(&dst_structure, &control_src)
        .unwrap()
        .has_pack_gemm_scatter_blocks());
}
