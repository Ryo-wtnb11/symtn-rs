use super::*;
use crate::probe::{self, Event};

#[test]
fn tree_pair_plan_builder_handles_su2_one_by_one_domain_crossing() {
    let src_key = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids([1], [1], 1, [false], [false], [], [], [], [])
            .unwrap(),
    );
    let expected_dst_key = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids([1], [1], 1, [true], [true], [], [], [], [])
            .unwrap(),
    );
    let src_structure = packed_fixture_structure(2, [(src_key.clone(), vec![1, 1])]).unwrap();
    let dst_structure =
        packed_fixture_structure(2, [(expected_dst_key.clone(), vec![1, 1])]).unwrap();

    let plan = build_tree_pair_transform_group_plan(
        &SU2FusionRule,
        TreeTransformOperation::permute([1], [0]),
        &src_structure,
    )
    .unwrap();

    assert_eq!(plan.specs().len(), 1);
    let spec = &plan.specs()[0];
    assert_eq!(spec.src_keys(), &[expect_tree_key(&src_key)]);
    assert_eq!(spec.dst_keys(), &[expect_tree_key(&expected_dst_key)]);
    assert_eq!(spec.recoupling_coefficients_dst_src().len(), 1);
    assert!((spec.recoupling_coefficients_dst_src()[0] - 1.0).abs() < 1.0e-12);
    plan.compile_structures(&dst_structure, &src_structure)
        .unwrap();
}

#[test]
fn tree_pair_transform_public_helper_executes_su2_domain_crossing() {
    let src_key = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids([1], [1], 1, [false], [false], [], [], [], [])
            .unwrap(),
    );
    let expected_dst_key = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids([1], [1], 1, [true], [true], [], [], [], [])
            .unwrap(),
    );
    let src_structure = packed_fixture_structure(2, [(src_key, vec![1, 1])]).unwrap();
    let dst_structure =
        packed_fixture_structure(2, [(expected_dst_key.clone(), vec![1, 1])]).unwrap();
    let src_space = vec![1, 1];
    let dst_space = vec![1, 1];
    let src =
        TestTensor::<f64>::from_vec_with_structure(vec![7.0], src_space, src_structure).unwrap();
    let mut dst =
        TestTensor::<f64>::from_vec_with_structure(vec![2.0], dst_space, dst_structure).unwrap();
    eager_tree_transform_into(
        &SU2FusionRule,
        TreeTransformOperation::permute([1], [0]),
        &mut dst,
        &src,
        3.0,
        5.0,
    )
    .unwrap();

    assert_eq!(dst.structure().block(0).unwrap().key(), &expected_dst_key);
    assert!((dst.data()[0] - 31.0).abs() < 1.0e-12);
}

#[test]
fn tree_pair_transform_public_helper_executes_su2_with_complex_data() {
    let src_key = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids([1], [1], 1, [false], [false], [], [], [], [])
            .unwrap(),
    );
    let expected_dst_key = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids([1], [1], 1, [true], [true], [], [], [], [])
            .unwrap(),
    );
    let src_structure = packed_fixture_structure(2, [(src_key, vec![1, 1])]).unwrap();
    let dst_structure =
        packed_fixture_structure(2, [(expected_dst_key.clone(), vec![1, 1])]).unwrap();
    let src_space = vec![1, 1];
    let dst_space = vec![1, 1];
    let src = TestTensor::<Complex64>::from_vec_with_structure(
        vec![Complex64::new(7.0, 1.0)],
        src_space,
        src_structure,
    )
    .unwrap();
    let mut dst = TestTensor::<Complex64>::from_vec_with_structure(
        vec![Complex64::new(2.0, -3.0)],
        dst_space,
        dst_structure,
    )
    .unwrap();
    let operation = TreeTransformOperation::permute([1], [0]);

    eager_tree_transform_into(
        &SU2FusionRule,
        operation,
        &mut dst,
        &src,
        Complex64::new(3.0, 0.0),
        Complex64::new(5.0, 0.0),
    )
    .unwrap();

    assert_eq!(dst.structure().block(0).unwrap().key(), &expected_dst_key);
    assert!((dst.data()[0] - Complex64::new(31.0, -12.0)).norm() < 1.0e-12);
}

#[test]
fn tree_pair_operation_key_uses_tensorkit_global_source_axes() {
    let src_key = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [1, 0],
            [1],
            1,
            [false, false],
            [false],
            [],
            [],
            [1],
            [],
        )
        .unwrap(),
    );
    let src_structure = packed_fixture_structure(3, [(src_key, vec![1, 1, 1])]).unwrap();

    let local_domain_identity = build_tree_pair_transform_group_plan(
        &Z2FusionRule,
        TreeTransformOperation::permute([1, 0], [0]),
        &src_structure,
    )
    .unwrap_err();
    assert_eq!(
        local_domain_identity,
        OperationError::InvalidPermutation {
            axes: vec![1, 0, 0],
            rank: 3
        }
    );

    build_tree_pair_transform_group_plan(
        &Z2FusionRule,
        TreeTransformOperation::permute([1, 0], [2]),
        &src_structure,
    )
    .unwrap();
}

#[test]
fn unique_tree_pair_reuses_completed_transformers_with_complete_storage_keys() {
    let src_key = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [1],
            [0, 1],
            1,
            [false],
            [false, true],
            [],
            [],
            [],
            [1],
        )
        .unwrap(),
    );
    // The cache API requires the built-in rule-key marker and rigid symbols;
    // use the production Z2 rule, whose semantics match the local UniqueZ2
    // oracle while satisfying both bounds without adding test-only adapters.
    let rule = Z2FusionRule;
    assert_eq!(rule.fusion_style(), FusionStyleKind::Unique);
    let src_tree = expect_tree_key(&src_key);
    let operation = TreeTransformOperation::permute([0, 2], [1]);
    let (dst_tree, _) = single_permuted_tree_pair(&rule, &src_tree, &[0, 2], &[1]);
    let src_structure = packed_fixture_structure(3, [(src_key, vec![1, 1, 1])]).unwrap();
    let dst_structure =
        packed_fixture_structure(3, [(BlockKey::from(dst_tree), vec![1, 1, 1])]).unwrap();
    let src_space = vec![1, 1, 1];
    let dst_space = vec![1, 1, 1];
    let src =
        TestTensor::<f64>::from_vec_with_structure(vec![7.0], src_space, src_structure).unwrap();
    let dst =
        TestTensor::<f64>::from_vec_with_structure(vec![0.0], dst_space, dst_structure).unwrap();
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    mark_canonical([dst.structure().as_ref(), src.structure().as_ref()]);

    let first =
        resolve_tree_pair(&rule, &operation, dst.structure(), src.structure(), false).unwrap();
    let second =
        resolve_tree_pair(&rule, &operation, dst.structure(), src.structure(), false).unwrap();
    assert!(same_core(&first, &second));

    let dst_structure = Arc::new(dst.structure().clone());
    let src_structure = Arc::new(src.structure().clone());
    let direct_plan = build_tree_pair_transform_group_plan(
        &rule,
        TreeTransformOperation::permute([0, 2], [1]),
        &src_structure,
    )
    .unwrap();
    let direct = direct_plan
        .compile_shared_structures_with_storage_conjugation(
            Arc::clone(&dst_structure),
            Arc::clone(&src_structure),
            true,
        )
        .unwrap();
    let permute = TreeTransformOperation::permute([0, 2], [1]);
    let first_storage =
        resolve_tree_pair(&rule, &permute, &dst_structure, &src_structure, true).unwrap();
    let cached_storage =
        resolve_tree_pair(&rule, &permute, &dst_structure, &src_structure, true).unwrap();
    assert_eq!(cached_storage, direct);
    assert!(same_core(&first_storage, &cached_storage));
    assert!(cached_storage.storage_conjugate());
    // What: the complete Unique transformer is reused (the wrapper clones
    // share content, hence content ids), while storage conjugation remains a
    // distinct completed-transformer key.
    assert!(same_core(
        &first,
        &resolve_tree_pair(&rule, &permute, &dst_structure, &src_structure, false).unwrap()
    ));
    assert!(!same_core(&first, &first_storage));
}

#[test]
fn fermionic_storage_conjugation_uses_distinct_reusable_structures() {
    let rule = FermionParityFusionRule;
    let odd = SectorId::new(1);
    let tree = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(&rule, [odd], odd, [false], [], []).unwrap(),
        FusionTreeKey::try_new_for_rule(&rule, [odd], odd, [true], [], []).unwrap(),
    );
    let structure =
        Arc::new(packed_fixture_structure(2, [(BlockKey::from(tree), vec![1, 1])]).unwrap());
    let operation = TreeTransformOperation::braid([1], [0], [0], [1]);
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    mark_canonical([structure.as_ref()]);
    let resolve = |conjugate| {
        resolve_tree_pair(&rule, &operation, &structure, &structure, conjugate).unwrap()
    };

    let plain = resolve(false);
    let conjugated = resolve(true);
    let plain_warm = resolve(false);
    let conjugated_warm = resolve(true);

    // What: fermionic storage conjugation is part of the transformer key;
    // each variant is reused.
    assert!(!same_core(&plain, &conjugated));
    assert!(same_core(&plain, &plain_warm));
    assert!(same_core(&conjugated, &conjugated_warm));
    assert!(!plain.storage_conjugate());
    assert!(conjugated.storage_conjugate());
}

#[test]
fn u1_unique_tree_pair_reuses_completed_transformer_but_expert_layouts_stay_eager() {
    let positive = U1Irrep::new(1).sector_id();
    let negative = U1Irrep::new(-1).sector_id();
    let vacuum = U1FusionRule.vacuum();
    let source = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &U1FusionRule,
            [positive, negative],
            vacuum,
            [false, false],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(&U1FusionRule, [], vacuum, [], [], []).unwrap(),
    );
    let destination = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &U1FusionRule,
            [negative, positive],
            vacuum,
            [false, false],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(&U1FusionRule, [], vacuum, [], [], []).unwrap(),
    );
    let src_structure = Arc::new(
        packed_fixture_structure(2, [(BlockKey::from(source.clone()), vec![1, 1])]).unwrap(),
    );
    let dst_structure = Arc::new(
        packed_fixture_structure(2, [(BlockKey::from(destination.clone()), vec![1, 1])]).unwrap(),
    );
    let operation = TreeTransformOperation::permute([1, 0], []);
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    mark_canonical([dst_structure.as_ref(), src_structure.as_ref()]);

    let first = resolve_tree_pair(
        &U1FusionRule,
        &operation,
        &dst_structure,
        &src_structure,
        false,
    )
    .unwrap();
    let second = resolve_tree_pair(
        &U1FusionRule,
        &operation,
        &dst_structure,
        &src_structure,
        false,
    )
    .unwrap();
    // What: U(1) reuses the completed transformer of canonical structures.
    assert!(same_core(&first, &second));

    // An equal expert layout (fresh content, never admitted) is lookup-only:
    // each resolution rebuilds an equal transformer.
    let expert_src =
        Arc::new(packed_fixture_structure(2, [(BlockKey::from(source), vec![1, 1])]).unwrap());
    let expert_dst =
        Arc::new(packed_fixture_structure(2, [(BlockKey::from(destination), vec![1, 1])]).unwrap());
    let first_eager =
        resolve_tree_pair(&U1FusionRule, &operation, &expert_dst, &expert_src, false).unwrap();
    let second_eager =
        resolve_tree_pair(&U1FusionRule, &operation, &expert_dst, &expert_src, false).unwrap();
    assert!(!same_core(&first_eager, &second_eager));
    assert_eq!(first_eager, second_eager);
    assert_eq!(first_eager.blocks(), first.blocks());
}

#[test]
fn tree_pair_transform_public_helper_executes_split_changing_permute() {
    let src_key = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [1],
            [0, 1],
            1,
            [false],
            [false, true],
            [],
            [],
            [],
            [1],
        )
        .unwrap(),
    );
    let src_tree = expect_tree_key(&src_key);
    let operation = TreeTransformOperation::permute([0, 2], [1]);
    let (dst_tree, coefficient) =
        single_permuted_tree_pair(&Z2FusionRule, &src_tree, &[0, 2], &[1]);
    let dst_key = BlockKey::from(dst_tree);
    let src_structure = packed_fixture_structure(3, [(src_key, vec![1, 1, 1])]).unwrap();
    let dst_structure = packed_fixture_structure(3, [(dst_key.clone(), vec![1, 1, 1])]).unwrap();
    let src_space = vec![1, 1, 1];
    let dst_space = vec![1, 1, 1];
    let src =
        TestTensor::<f64>::from_vec_with_structure(vec![7.0], src_space, src_structure).unwrap();
    let mut dst =
        TestTensor::<f64>::from_vec_with_structure(vec![2.0], dst_space, dst_structure).unwrap();

    eager_tree_transform_into(&Z2FusionRule, operation, &mut dst, &src, 3.0, 5.0).unwrap();

    assert_eq!(dst.structure().block(0).unwrap().key(), &dst_key);
    assert_eq!(dst.data(), &[3.0 * coefficient * 7.0 + 5.0 * 2.0]);
}

#[test]
fn tree_pair_transform_public_helper_compiles_against_actual_destination_structure() {
    let src_key = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [1],
            [0, 1],
            1,
            [false],
            [false, true],
            [],
            [],
            [],
            [1],
        )
        .unwrap(),
    );
    let src_tree = expect_tree_key(&src_key);
    let operation = TreeTransformOperation::permute([0, 2], [1]);
    let (dst_tree, _) = single_permuted_tree_pair(&Z2FusionRule, &src_tree, &[0, 2], &[1]);
    let expected_missing = BlockKey::from(dst_tree);
    let src_structure = packed_fixture_structure(3, [(src_key.clone(), vec![1, 1, 1])]).unwrap();
    let wrong_dst_structure = packed_fixture_structure(3, [(src_key, vec![1, 1, 1])]).unwrap();
    let src_space = vec![1, 1, 1];
    let dst_space = vec![1, 1, 1];
    let src =
        TestTensor::<f64>::from_vec_with_structure(vec![7.0], src_space, src_structure).unwrap();
    let dst = TestTensor::<f64>::from_vec_with_structure(vec![0.0], dst_space, wrong_dst_structure)
        .unwrap();

    let err = uncached_tree_pair_structure(&Z2FusionRule, operation, &dst, &src).unwrap_err();

    assert_eq!(
        err,
        OperationError::MissingBlockKey {
            key: Box::new(expected_missing),
        }
    );
}

#[test]
fn multiplicity_free_product_tree_pair_plan_builder_handles_fz2_u1_su2_blocks() {
    let (rule, src_space, dst_space, [c0, c1]) = fz2_u1_su2_tree_pair_fixture();
    let src_structure = src_space.structure();
    let dst_structure = dst_space.structure();

    let plan = build_tree_pair_transform_group_plan(
        &rule,
        TreeTransformOperation::permute([1, 0], [2]),
        src_structure,
    )
    .unwrap();

    assert_eq!(plan.specs().len(), 2);
    assert!((single_transform_coefficient_for_coupled(&plan, c0) - 1.0).abs() < 1.0e-12);
    assert!((single_transform_coefficient_for_coupled(&plan, c1) + 1.0).abs() < 1.0e-12);
    plan.compile_structures(dst_structure, src_structure)
        .unwrap();
}

#[test]
fn eager_product_tree_pair_plan_bypasses_row_callback_assembly() {
    let (rule, src_space, dst_space, _) = fz2_u1_su2_tree_pair_fixture();
    let operation = TreeTransformOperation::permute([1, 0], [2]);
    let group_count = src_space.structure().fusion_tree_groups().len();
    let row_callback = build_tree_transform_group_plan(
        &rule,
        operation.clone(),
        src_space.structure(),
        |source| {
            symtn_core::multiplicity_free_permute_tree_pair(&rule, source, &[1, 0], &[2])
                .map_err(OperationError::from_core_preserving_context)
        },
    )
    .unwrap();

    let tree_pair_lowering_calls_rec = probe::record();
    let direct =
        build_tree_pair_transform_group_plan(&rule, operation, src_space.structure()).unwrap();

    // What: the eager Simple builder reproduces the per-row assembler's plan while
    // consuming one core ordered map per whole group instead of rebuilding
    // full-key rows per source column.
    assert_eq!(direct, row_callback);
    assert_eq!(
        tree_pair_lowering_calls_rec.get(Event::OrderedTreePairLowering),
        group_count
    );
    direct
        .compile_structures(dst_space.structure(), src_space.structure())
        .unwrap();
}

#[test]
fn eager_simple_plan_prepares_once_per_distinct_source_split() {
    use crate::tree_transform::{
        build_tree_pair_transform_group_plan_validated_with_threads,
        validate_multiplicity_free_tree_pair_preflight,
    };

    let half_id = 1usize;
    let one_id = 2usize;
    let vacuum_id = 0usize;
    let half = SU2Irrep::from_twice_spin(half_id).sector_id();
    let vacuum = SU2FusionRule.vacuum();
    let group_a = [[vacuum_id, half_id], [one_id, half_id]].map(|inner| {
        all_codomain_fusion_tree_test_key_for_rule(
            &SU2FusionRule,
            [half_id; 4],
            vacuum_id,
            [false; 4],
            inner,
            [1, 1, 1],
        )
    });
    let group_b = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [half_id, half_id, one_id, one_id],
        vacuum_id,
        [false; 4],
        [vacuum_id, one_id],
        [1, 1, 1],
    );
    let split_group = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &SU2FusionRule,
            [half; 3],
            half,
            [false; 3],
            [vacuum],
            [MultiplicityIndex::ONE; 2],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(&SU2FusionRule, [half], half, [false], [], []).unwrap(),
    );
    let source_order = [group_a[0].clone(), group_b, group_a[1].clone(), split_group];
    let structure = packed_fixture_structure(
        4,
        source_order
            .iter()
            .cloned()
            .map(|key| (key, vec![1usize; 4])),
    )
    .unwrap();
    assert_eq!(
        structure
            .fusion_tree_group_slice()
            .iter()
            .map(|group| group.block_indices())
            .collect::<Vec<_>>(),
        vec![&[0, 2][..], &[1][..], &[3][..]]
    );
    let operation = TreeTransformOperation::permute([1, 0, 2], [3]);
    let oracle =
        build_tree_transform_group_plan(&SU2FusionRule, operation.clone(), &structure, |source| {
            symtn_core::multiplicity_free_permute_tree_pair(
                &SU2FusionRule,
                source,
                &[1, 0, 2],
                &[3],
            )
            .map_err(OperationError::from_core_preserving_context)
        })
        .unwrap();
    let proof =
        validate_multiplicity_free_tree_pair_preflight(&SU2FusionRule, &operation, &structure)
            .unwrap();

    for threads in [1, 4] {
        let tree_pair_operation_preparations_rec = probe::record();
        let prepared = build_tree_pair_transform_group_plan_validated_with_threads(
            &proof,
            operation.clone(),
            threads,
        )
        .unwrap();

        // What: serial and threaded lowering each prepare two source splits
        // across three interleaved groups while preserving the independent
        // per-source oracle's grouping, order, and coefficients.
        assert_eq!(
            tree_pair_operation_preparations_rec.get(Event::TreePairOperationPreparation),
            2
        );
        assert_eq!(prepared.specs().len(), oracle.specs().len());
        for (actual, expected) in prepared.specs().iter().zip(oracle.specs()) {
            assert_eq!(actual.group_key(), expected.group_key());
            assert_eq!(actual.src_keys(), expected.src_keys());
            assert_eq!(actual.dst_keys(), expected.dst_keys());
            assert_eq!(actual.source_axes(), expected.source_axes());
            assert_eq!(
                actual.recoupling_coefficients_dst_src().len(),
                expected.recoupling_coefficients_dst_src().len()
            );
            for (&actual, &expected) in actual
                .recoupling_coefficients_dst_src()
                .iter()
                .zip(expected.recoupling_coefficients_dst_src())
            {
                assert!((actual - expected).abs() <= 1.0e-12);
            }
        }
    }
}

#[test]
fn rank_nine_same_split_groups_prepare_once_and_lower_once_each() {
    // What: runtime-rank operation storage preserves the per-source tree,
    // coefficient, compiled-layout, and replay oracle above the old boundary.
    use crate::tree_transform::{
        build_tree_pair_transform_group_plan_validated_with_threads,
        validate_multiplicity_free_tree_pair_preflight,
    };

    let vacuum = SU2FusionRule.vacuum();
    let keys = [0usize, 1, 2].map(|twice_spin| {
        let mut uncoupled = [vacuum; 9];
        uncoupled[0] = SectorId::new(twice_spin);
        uncoupled[1] = SectorId::new(twice_spin);
        BlockKey::from(FusionTreePairKey::pair(
            FusionTreeKey::try_new_for_rule(
                &SU2FusionRule,
                uncoupled,
                vacuum,
                [false; 9],
                [vacuum; 7],
                [MultiplicityIndex::ONE; 8],
            )
            .unwrap(),
            empty_fusion_tree(),
        ))
    });
    let structure =
        packed_fixture_structure(9, keys.into_iter().map(|key| (key, vec![1usize; 9]))).unwrap();
    let operation = TreeTransformOperation::braid([1, 0, 2, 3, 4, 5, 6, 7, 8], [], 0..9, []);
    let oracle =
        build_tree_transform_group_plan(&SU2FusionRule, operation.clone(), &structure, |source| {
            symtn_core::multiplicity_free_braid_tree_pair(
                &SU2FusionRule,
                source,
                &[1, 0, 2, 3, 4, 5, 6, 7, 8],
                &[],
                &[0, 1, 2, 3, 4, 5, 6, 7, 8],
                &[],
            )
            .map_err(OperationError::from_core_preserving_context)
        })
        .unwrap();
    let proof =
        validate_multiplicity_free_tree_pair_preflight(&SU2FusionRule, &operation, &structure)
            .unwrap();

    let tree_pair_operation_preparations_rec = probe::record();
    let tree_pair_lowering_calls_rec = probe::record();
    let plan =
        build_tree_pair_transform_group_plan_validated_with_threads(&proof, operation, 1).unwrap();

    assert_eq!(structure.fusion_tree_groups().len(), 3);
    assert_eq!(
        tree_pair_operation_preparations_rec.get(Event::TreePairOperationPreparation),
        1
    );
    assert_eq!(
        tree_pair_lowering_calls_rec.get(Event::OrderedTreePairLowering),
        3
    );
    assert_eq!(plan.specs().len(), 3);
    assert_eq!(plan.specs().len(), oracle.specs().len());
    for (actual, expected) in plan.specs().iter().zip(oracle.specs()) {
        assert_eq!(actual.group_key(), expected.group_key());
        assert_eq!(actual.src_keys(), expected.src_keys());
        assert_eq!(actual.dst_keys(), expected.dst_keys());
        assert_eq!(actual.source_axes(), expected.source_axes());
        assert_eq!(
            actual
                .recoupling_coefficients_dst_src()
                .iter()
                .map(|coefficient| coefficient.to_bits())
                .collect::<Vec<_>>(),
            expected
                .recoupling_coefficients_dst_src()
                .iter()
                .map(|coefficient| coefficient.to_bits())
                .collect::<Vec<_>>()
        );
    }

    let compiled = plan.compile_structures(&structure, &structure).unwrap();
    let oracle_compiled = oracle.compile_structures(&structure, &structure).unwrap();
    assert_eq!(compiled.blocks(), oracle_compiled.blocks());
    assert_eq!(compiled.layouts(), oracle_compiled.layouts());
    assert_eq!(
        compiled
            .gathered_coefficients()
            .iter()
            .map(|coefficient| coefficient.to_bits())
            .collect::<Vec<_>>(),
        oracle_compiled
            .gathered_coefficients()
            .iter()
            .map(|coefficient| coefficient.to_bits())
            .collect::<Vec<_>>()
    );

    let space = [&[1; 9][..], &[][..]].concat();
    let source = TestTensor::<f64>::from_vec_with_structure(
        vec![1.0, 2.0, 3.0],
        space.clone(),
        structure.clone(),
    )
    .unwrap();
    let mut output =
        TestTensor::<f64>::from_vec_with_structure(vec![0.0; 3], space.clone(), structure.clone())
            .unwrap();
    let mut oracle_output =
        TestTensor::<f64>::from_vec_with_structure(vec![0.0; 3], space, structure).unwrap();
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();
    replay_into(
        &mut backend,
        &mut workspace,
        &compiled,
        &mut output,
        &source,
        1.0,
        0.0,
    )
    .unwrap();
    replay_into(
        &mut backend,
        &mut workspace,
        &oracle_compiled,
        &mut oracle_output,
        &source,
        1.0,
        0.0,
    )
    .unwrap();
    assert_eq!(
        output
            .data()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        oracle_output
            .data()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
fn eager_su2_dense_tree_pair_plan_matches_row_callback_replay() {
    let source0 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    );
    let source1 = all_codomain_fusion_tree_test_key_for_rule(
        &SU2FusionRule,
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [2, 1],
        [1, 1, 1],
    );
    let structure = packed_fixture_structure(
        4,
        [(source0, vec![1, 1, 1, 1]), (source1, vec![1, 1, 1, 1])],
    )
    .unwrap();
    let operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
    let row_callback =
        build_tree_transform_group_plan(&SU2FusionRule, operation.clone(), &structure, |source| {
            symtn_core::multiplicity_free_braid_tree_pair(
                &SU2FusionRule,
                source,
                &[0, 2, 1, 3],
                &[],
                &[0, 1, 2, 3],
                &[],
            )
            .map_err(OperationError::from_core_preserving_context)
        })
        .unwrap();

    let tree_pair_lowering_calls_rec = probe::record();
    let direct =
        build_tree_pair_transform_group_plan(&SU2FusionRule, operation, &structure).unwrap();
    // The per-source core oracle and the prepared lowering compute the F-move
    // coefficients by different routes, so the entry form, keys and axes
    // match exactly and the coefficients within the tolerance rule (a rank-4
    // recoupling chain).
    assert_eq!(direct.specs().len(), row_callback.specs().len());
    // The Single/Multi entry form is private to symtn-operations; its Debug
    // tag is the only test-visible witness, and it is compared, not parsed.
    let entry_form = |spec: &dyn std::fmt::Debug| {
        let text = format!("{spec:?}");
        if text.contains("entries: Single") {
            "Single"
        } else if text.contains("entries: Multi") {
            "Multi"
        } else {
            panic!("unrecognized spec entry form: {text}")
        }
    };
    for (actual, expected) in direct.specs().iter().zip(row_callback.specs()) {
        assert_eq!(entry_form(actual), entry_form(expected));
        assert_eq!(actual.group_key(), expected.group_key());
        assert_eq!(actual.src_keys(), expected.src_keys());
        assert_eq!(actual.dst_keys(), expected.dst_keys());
        assert_eq!(actual.source_axes(), expected.source_axes());
        crate::test_numerics::numerics::assert_slices_close(
            "direct vs row-callback recoupling coefficients",
            actual.recoupling_coefficients_dst_src(),
            expected.recoupling_coefficients_dst_src(),
            4,
        );
    }
    assert_eq!(
        tree_pair_lowering_calls_rec.get(Event::OrderedTreePairLowering),
        1
    );

    let space = vec![1, 1, 1, 1];
    let src = TestTensor::<f64>::from_vec_with_structure(
        vec![10.0, 20.0],
        space.clone(),
        structure.clone(),
    )
    .unwrap();
    let mut direct_dst = TestTensor::<f64>::from_vec_with_structure(
        vec![0.0, 0.0],
        space.clone(),
        structure.clone(),
    )
    .unwrap();
    let mut row_callback_dst =
        TestTensor::<f64>::from_vec_with_structure(vec![0.0, 0.0], space, structure.clone())
            .unwrap();
    let direct_structure = direct
        .compile_structures(direct_dst.structure(), src.structure())
        .unwrap();
    let row_callback_structure = row_callback
        .compile_structures(row_callback_dst.structure(), src.structure())
        .unwrap();
    let mut direct_backend = DenseTreeTransformOperations::default();
    let mut row_callback_backend = DenseTreeTransformOperations::default();
    let mut direct_workspace = TreeTransformWorkspace::default();
    let mut row_callback_workspace = TreeTransformWorkspace::default();
    replay_into(
        &mut direct_backend,
        &mut direct_workspace,
        &direct_structure,
        &mut direct_dst,
        &src,
        1.0,
        0.0,
    )
    .unwrap();
    replay_into(
        &mut row_callback_backend,
        &mut row_callback_workspace,
        &row_callback_structure,
        &mut row_callback_dst,
        &src,
        1.0,
        0.0,
    )
    .unwrap();
    // Each destination entry sums the two recoupled source trees.
    crate::test_numerics::numerics::assert_slices_close(
        "direct vs row-callback replay",
        direct_dst.data(),
        row_callback_dst.data(),
        2,
    );
}

#[test]
fn product_tree_pair_plan_is_thread_count_invariant() {
    use crate::tree_transform::{
        build_tree_pair_transform_group_plan_validated_with_threads,
        validate_multiplicity_free_tree_pair_preflight,
    };

    let (rule, src_space, dst_space, [c0, c1]) = fz2_u1_su2_tree_pair_fixture();
    let operation = TreeTransformOperation::permute([1, 0], [2]);
    let build = |threads| {
        let proof = validate_multiplicity_free_tree_pair_preflight(
            &rule,
            &operation,
            src_space.structure(),
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
    for threads in [2, 4] {
        let threaded = build(threads);
        // What: product-symmetry fermionic phases and plan order are identical
        // when scheduling the same fusion groups.
        assert_eq!(threaded, serial);
        threaded
            .compile_structures(dst_space.structure(), src_space.structure())
            .unwrap();
    }
    assert!((single_transform_coefficient_for_coupled(&serial, c0) - 1.0).abs() < 1.0e-12);
    assert!((single_transform_coefficient_for_coupled(&serial, c1) + 1.0).abs() < 1.0e-12);
}

#[test]
fn tree_pair_transform_public_helper_executes_product_fz2_u1_su2_blocks() {
    let (rule, src_space, dst_space, [c0, c1]) = fz2_u1_su2_tree_pair_fixture();
    let operation = TreeTransformOperation::permute([1, 0], [2]);
    let src =
        TestTensor::<f64>::from_vec_with_fusion_space(vec![10.0, 20.0], src_space.clone()).unwrap();
    let mut dst =
        TestTensor::<f64>::from_vec_with_fusion_space(vec![1.0, 2.0], dst_space.clone()).unwrap();
    let initial_dst = dst.data().to_vec();
    let plan =
        build_tree_pair_transform_group_plan(&rule, operation.clone(), src.structure()).unwrap();
    assert!((single_transform_coefficient_for_coupled(&plan, c0) - 1.0).abs() < 1.0e-12);
    assert!((single_transform_coefficient_for_coupled(&plan, c1) + 1.0).abs() < 1.0e-12);
    let mut expected = initial_dst
        .iter()
        .map(|value| 3.0 * value)
        .collect::<Vec<_>>();
    for spec in plan.specs() {
        let src_key = &spec.src_keys()[0];
        let dst_key = &spec.dst_keys()[0];
        let src_offset = block_offset_for_tree_pair(src.structure(), src_key);
        let dst_offset = block_offset_for_tree_pair(dst.structure(), dst_key);
        expected[dst_offset] +=
            2.0 * spec.recoupling_coefficients_dst_src()[0] * src.data()[src_offset];
    }

    eager_tree_transform_into(&rule, operation, &mut dst, &src, 2.0, 3.0).unwrap();

    assert_eq!(dst.structure(), dst_space.structure());
    for (actual, expected) in dst.data().iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual {actual} != expected {expected}"
        );
    }
}

#[test]
fn product_tree_transformers_are_shared_by_independent_resolutions() {
    let (rule, src_space, dst_space, _) = fz2_u1_su2_tree_pair_fixture();
    let operation = TreeTransformOperation::permute([1, 0], [2]);
    let src = TestTensor::<f64>::from_vec_with_fusion_space(vec![10.0, 20.0], src_space).unwrap();
    let dst = TestTensor::<f64>::from_vec_with_fusion_space(vec![1.0, 2.0], dst_space).unwrap();
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    mark_canonical([dst.structure().as_ref(), src.structure().as_ref()]);

    let cold =
        resolve_tree_pair(&rule, &operation, dst.structure(), src.structure(), false).unwrap();
    let warm =
        resolve_tree_pair(&rule, &operation, dst.structure(), src.structure(), false).unwrap();
    let mut context = TreeTransformExecutionContext::<f64, RuleIdentity>::default();
    let from_context = context
        .tree_structure(
            &rule,
            &operation,
            dst.structure(),
            crate::mode::TreeStructureSource::Stored {
                structure: src.structure(),
                storage_conjugate: false,
            },
        )
        .unwrap();

    // What: fZ2 x U(1) x SU(2) transformers are process-global: independent
    // resolutions, a fresh context included, share one core.
    assert!(same_core(&cold, &warm));
    assert!(same_core(&cold, &from_context));
    assert_eq!(cold, from_context);
}

#[test]
fn recoupling_threads_do_not_change_cached_tree_transform_result() {
    let (rule, src_space, dst_space, _) = fz2_u1_su2_tree_pair_fixture();
    let operation = TreeTransformOperation::permute([1, 0], [2]);
    let src = TestTensor::<f64>::from_vec_with_fusion_space(vec![10.0, 20.0], src_space).unwrap();
    let dst = TestTensor::<f64>::from_vec_with_fusion_space(vec![1.0, 2.0], dst_space).unwrap();

    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    mark_canonical([dst.structure().as_ref(), src.structure().as_ref()]);
    let mut planning = crate::tree_transform::TreeTransformPlanning::default();
    planning.set_recoupling_threads(std::num::NonZeroUsize::new(1).unwrap());
    let serial = planning
        .resolve_tree_pair(&rule, &operation, dst.structure(), src.structure(), false)
        .unwrap();
    planning.set_recoupling_threads(std::num::NonZeroUsize::new(4).unwrap());
    let cached = planning
        .resolve_tree_pair(&rule, &operation, dst.structure(), src.structure(), false)
        .unwrap();
    // The thread count is not a key determinant: the second call hits.
    assert!(same_core(&serial, &cached));

    // A build under 4 threads (an expert copy, so it cannot hit) equals it.
    let expert_dst = expert_copy(dst.structure());
    let expert_src = expert_copy(src.structure());
    let eager = planning
        .resolve_tree_pair(&rule, &operation, &expert_dst, &expert_src, false)
        .unwrap();
    assert!(!same_core(&eager, &cached));
    assert_eq!(eager.blocks(), cached.blocks());
    assert_eq!(eager.layouts(), cached.layouts());

    let mut cached_dst = TestTensor::<f64>::from_vec_with_dynamic_space(
        vec![1.0, 2.0],
        dst.dims().to_vec(),
        dst.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut eager_dst = TestTensor::<f64>::from_vec_with_dynamic_space(
        vec![1.0, 2.0],
        dst.dims().to_vec(),
        dst.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut cached_backend = DenseTreeTransformOperations::default();
    let mut eager_backend = DenseTreeTransformOperations::default();
    let mut cached_workspace = TreeTransformWorkspace::default();
    let mut eager_workspace = TreeTransformWorkspace::default();
    replay_into(
        &mut cached_backend,
        &mut cached_workspace,
        &cached,
        &mut cached_dst,
        &src,
        -2.0,
        3.0,
    )
    .unwrap();
    replay_into(
        &mut eager_backend,
        &mut eager_workspace,
        &eager,
        &mut eager_dst,
        &src,
        -2.0,
        3.0,
    )
    .unwrap();

    // What: `recoupling_threads` schedules plan compilation only; omitting it
    // from the cache key must not change the compiled transform or replay bits.
    assert_eq!(
        cached_dst
            .data()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        eager_dst
            .data()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
fn product_tree_transform_rebuilds_after_global_cache_reset_with_old_values_live() {
    // What: a reset may drop every semantic layout/transform artifact while old
    // product-symmetry tensors remain live; rebuilding preserves the fZ2 swap sign.
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let (rule, src_space, dst_space, _) = fz2_u1_su2_tree_pair_fixture();
    let src_hom = src_space.homspace().clone();
    let dst_hom = dst_space.homspace().clone();
    let operation = TreeTransformOperation::permute([1, 0], [2]);
    let src_data = vec![10.0, 20.0];
    let initial_dst = vec![1.0, 2.0];
    let src = TestTensor::<f64>::from_vec_with_fusion_space(src_data.clone(), src_space).unwrap();
    let mut dst =
        TestTensor::<f64>::from_vec_with_fusion_space(initial_dst.clone(), dst_space).unwrap();
    eager_tree_transform_into(&rule, operation.clone(), &mut dst, &src, 2.0, 3.0).unwrap();
    let expected = dst.data().to_vec();

    symtn_core::clear_structure_caches();
    let rebuilt_src_space = DynamicFusionMapSpace::from_final_homspace(&rule, src_hom).unwrap();
    let rebuilt_dst_space = DynamicFusionMapSpace::from_final_homspace(&rule, dst_hom).unwrap();
    let rebuilt_src =
        TestTensor::<f64>::from_vec_with_fusion_space(src_data, rebuilt_src_space).unwrap();
    let mut rebuilt_dst =
        TestTensor::<f64>::from_vec_with_fusion_space(initial_dst, rebuilt_dst_space).unwrap();
    eager_tree_transform_into(&rule, operation, &mut rebuilt_dst, &rebuilt_src, 2.0, 3.0).unwrap();

    assert_eq!(rebuilt_dst.data(), expected.as_slice());
}

#[test]
fn tree_pair_transform_public_helper_executes_product_with_complex_data() {
    let (rule, src_space, dst_space, [c0, c1]) = fz2_u1_su2_tree_pair_fixture();
    let operation = TreeTransformOperation::permute([1, 0], [2]);
    let src = TestTensor::<Complex64>::from_vec_with_fusion_space(
        vec![Complex64::new(10.0, 1.0), Complex64::new(20.0, -2.0)],
        src_space.clone(),
    )
    .unwrap();
    let mut dst = TestTensor::<Complex64>::from_vec_with_fusion_space(
        vec![Complex64::new(1.0, 3.0), Complex64::new(2.0, -4.0)],
        dst_space.clone(),
    )
    .unwrap();
    let initial_dst = dst.data().to_vec();
    let plan =
        build_tree_pair_transform_group_plan(&rule, operation.clone(), src.structure()).unwrap();
    assert!((single_transform_coefficient_for_coupled(&plan, c0) - 1.0).abs() < 1.0e-12);
    assert!((single_transform_coefficient_for_coupled(&plan, c1) + 1.0).abs() < 1.0e-12);
    let alpha = Complex64::new(2.0, 0.0);
    let beta = Complex64::new(3.0, 0.0);
    let mut expected = initial_dst
        .iter()
        .map(|value| *value * beta)
        .collect::<Vec<_>>();
    for spec in plan.specs() {
        let src_key = &spec.src_keys()[0];
        let dst_key = &spec.dst_keys()[0];
        let src_offset = block_offset_for_tree_pair(src.structure(), src_key);
        let dst_offset = block_offset_for_tree_pair(dst.structure(), dst_key);
        expected[dst_offset] += src.data()[src_offset]
            .scale_by_coefficient(spec.recoupling_coefficients_dst_src()[0])
            * alpha;
    }

    eager_tree_transform_into(&rule, operation, &mut dst, &src, alpha, beta).unwrap();

    assert_eq!(dst.structure(), dst_space.structure());
    assert_eq!(dst.data(), expected.as_slice());
}

#[test]
fn tree_transform_structure_replays_product_without_recompiling() {
    let (rule, src_space, dst_space, [c0, c1]) = fz2_u1_su2_tree_pair_fixture();
    let operation = TreeTransformOperation::permute([1, 0], [2]);
    let mut src =
        TestTensor::<f64>::from_vec_with_fusion_space(vec![10.0, 20.0], src_space.clone()).unwrap();
    let mut dst =
        TestTensor::<f64>::from_vec_with_fusion_space(vec![1.0, 2.0], dst_space.clone()).unwrap();
    let plan =
        build_tree_pair_transform_group_plan(&rule, operation.clone(), src.structure()).unwrap();
    let structure = uncached_tree_pair_structure(&rule, operation, &dst, &src).unwrap();
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();

    assert!((single_transform_coefficient_for_coupled(&plan, c0) - 1.0).abs() < 1.0e-12);
    assert!((single_transform_coefficient_for_coupled(&plan, c1) + 1.0).abs() < 1.0e-12);
    assert_eq!(structure.block_count(), 2);
    assert!(!structure.has_pack_gemm_scatter_blocks());
    let expected_first = expected_single_tree_pair_replay(
        &plan,
        dst.structure(),
        src.structure(),
        dst.data(),
        src.data(),
        2.0,
        3.0,
    );
    replay_into(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &src,
        2.0,
        3.0,
    )
    .unwrap();
    for (actual, expected) in dst.data().iter().zip(expected_first) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual {actual} != expected {expected}"
        );
    }
    assert_eq!(workspace.source_len(), 0);
    assert_eq!(workspace.destination_len(), 0);

    src.data_mut().copy_from_slice(&[4.0, 5.0]);
    dst.data_mut().copy_from_slice(&[6.0, 7.0]);
    let expected_second = expected_single_tree_pair_replay(
        &plan,
        dst.structure(),
        src.structure(),
        dst.data(),
        src.data(),
        -1.0,
        0.5,
    );
    replay_into(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &src,
        -1.0,
        0.5,
    )
    .unwrap();
    for (actual, expected) in dst.data().iter().zip(expected_second) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual {actual} != expected {expected}"
        );
    }
}

#[test]
fn tree_transform_overwrite_context_ignores_destination_bits() {
    let (rule, src_space, dst_space, _) = fz2_u1_su2_tree_pair_fixture();
    type RuleKey = <FpU1Su2Rule as TreeTransformRuleCacheKey>::Key;
    let operation = TreeTransformOperation::permute([1, 0], [2]);
    let src = TestTensor::<f64>::from_vec_with_fusion_space(vec![10.0, 20.0], src_space).unwrap();
    let mut expected = TestTensor::<f64>::from_vec_with_fusion_space(
        vec![0.0; dst_space.required_len().unwrap()],
        dst_space.clone(),
    )
    .unwrap();
    eager_tree_transform_into(&rule, operation.clone(), &mut expected, &src, 2.0, 0.0).unwrap();

    let mut cached = TestTensor::<f64>::from_vec_with_fusion_space(
        vec![f64::NAN; dst_space.required_len().unwrap()],
        dst_space,
    )
    .unwrap();
    let mut context = TreeTransformExecutionContext::<f64, RuleKey>::default();
    for _ in 0..2 {
        cached.data_mut().fill(f64::NAN);
        context_tree_transform_overwrite_into(
            &mut context,
            &rule,
            operation.clone(),
            &mut cached,
            &src,
            2.0,
        )
        .unwrap();
        assert_eq!(cached.data(), expected.data());
    }

    // What: the borrowed-operation overwrite entry point ignores destination bits
    // while reusing the same compiled structure on its warm invocation.
    let dst_structure = Arc::clone(cached.structure());
    let src_structure = Arc::clone(src.structure());
    for _ in 0..2 {
        cached.data_mut().fill(f64::NAN);
        context
            .tree_transform_dyn_overwrite_into_ref(
                &rule,
                &operation,
                &dst_structure,
                &src_structure,
                cached.data_mut(),
                src.data(),
                2.0,
            )
            .unwrap();
        assert_eq!(cached.data(), expected.data());
    }

    // What: the borrowed-operation accumulating entry point matches the eager
    // transform without adding another completed structure.
    cached.data_mut().fill(0.0);
    context
        .tree_transform_dyn_into_ref(
            &rule,
            &operation,
            &dst_structure,
            &src_structure,
            cached.data_mut(),
            src.data(),
            2.0,
            0.0,
        )
        .unwrap();
    assert_eq!(cached.data(), expected.data());
}

#[test]
fn product_resolution_compiles_distinct_degeneracy_shapes() {
    let (rule, src_space, dst_space, _) = fz2_u1_su2_tree_pair_fixture();
    let operation = TreeTransformOperation::permute([1, 0], [2]);
    let src =
        TestTensor::<f64>::from_vec_with_fusion_space(vec![10.0, 20.0], src_space.clone()).unwrap();
    let mut dst =
        TestTensor::<f64>::from_vec_with_fusion_space(vec![1.0, 2.0], dst_space.clone()).unwrap();
    let src_large_structure = column_major_structure_like(src_space.structure(), vec![2, 1, 1]);
    let dst_large_structure = column_major_structure_like(dst_space.structure(), vec![1, 2, 1]);
    let large_space = vec![2, 1, 1];
    let src_large = TestTensor::<f64>::from_vec_with_structure(
        vec![1.0, 2.0, 3.0, 4.0],
        large_space.clone(),
        src_large_structure,
    )
    .unwrap();
    let dst_large = TestTensor::<f64>::from_vec_with_structure(
        vec![0.0, 0.0, 0.0, 0.0],
        large_space,
        dst_large_structure,
    )
    .unwrap();
    let resolve = |dst: &TestTensor<f64>, src: &TestTensor<f64>| {
        resolve_tree_pair(&rule, &operation, dst.structure(), src.structure(), false).unwrap()
    };
    let small = resolve(&dst, &src);
    let large = resolve(&dst_large, &src_large);
    assert_eq!((small.block_count(), large.block_count()), (2, 2));
    // What: a degeneracy change is a distinct layout, hence a distinct core.
    assert!(!same_core(&small, &large));
    assert_ne!(small.layouts(), large.layouts());

    let structure = resolve(&dst, &src);
    let plan = build_tree_pair_transform_group_plan(
        &rule,
        TreeTransformOperation::permute([1, 0], [2]),
        src.structure(),
    )
    .unwrap();
    let expected = expected_single_tree_pair_replay(
        &plan,
        dst.structure(),
        src.structure(),
        dst.data(),
        src.data(),
        2.0,
        3.0,
    );
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();
    replay_into(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &src,
        2.0,
        3.0,
    )
    .unwrap();
    for (actual, expected) in dst.data().iter().zip(expected) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual {actual} != expected {expected}"
        );
    }
}

#[test]
fn tree_transform_execution_context_replays_product_tree_pair_transforms() {
    let (rule, src_space, dst_space, _) = fz2_u1_su2_tree_pair_fixture();
    let equivalent_rule = FpU1Su2Rule::try_new(
        FpU1Rule::try_new(FermionParityFusionRule, U1FusionRule).unwrap(),
        SU2FusionRule,
    )
    .unwrap();
    type RuleKey = <FpU1Su2Rule as TreeTransformRuleCacheKey>::Key;
    let operation = TreeTransformOperation::permute([1, 0], [2]);
    let mut src =
        TestTensor::<f64>::from_vec_with_fusion_space(vec![10.0, 20.0], src_space.clone()).unwrap();
    let mut dst =
        TestTensor::<f64>::from_vec_with_fusion_space(vec![1.0, 2.0], dst_space.clone()).unwrap();
    let plan =
        build_tree_pair_transform_group_plan(&rule, operation.clone(), src.structure()).unwrap();
    let mut context = TreeTransformExecutionContext::<f64, RuleKey>::default();
    let expected_first = expected_single_tree_pair_replay(
        &plan,
        dst.structure(),
        src.structure(),
        dst.data(),
        src.data(),
        2.0,
        3.0,
    );

    context_tree_transform_into(
        &mut context,
        &rule,
        operation.clone(),
        &mut dst,
        &src,
        2.0,
        3.0,
    )
    .unwrap();

    for (actual, expected) in dst.data().iter().zip(expected_first) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual {actual} != expected {expected}"
        );
    }

    src.data_mut().copy_from_slice(&[4.0, 5.0]);
    dst.data_mut().copy_from_slice(&[6.0, 7.0]);
    let expected_second = expected_single_tree_pair_replay(
        &plan,
        dst.structure(),
        src.structure(),
        dst.data(),
        src.data(),
        -1.0,
        0.5,
    );
    context_tree_transform_into(
        &mut context,
        &equivalent_rule,
        operation,
        &mut dst,
        &src,
        -1.0,
        0.5,
    )
    .unwrap();

    assert_eq!(rule.rule_identity(), equivalent_rule.rule_identity());
    for (actual, expected) in dst.data().iter().zip(expected_second) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "actual {actual} != expected {expected}"
        );
    }
}

#[test]
fn tree_transform_execution_context_separates_tree_pair_operations() {
    let (rule, src_space, dst_space, _) = fz2_u1_su2_tree_pair_fixture();
    type RuleKey = <FpU1Su2Rule as TreeTransformRuleCacheKey>::Key;
    let src =
        TestTensor::<f64>::from_vec_with_fusion_space(vec![10.0, 20.0], src_space.clone()).unwrap();
    let mut dst =
        TestTensor::<f64>::from_vec_with_fusion_space(vec![1.0, 2.0], dst_space.clone()).unwrap();
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    mark_canonical([dst.structure().as_ref(), src.structure().as_ref()]);
    let mut context = TreeTransformExecutionContext::<f64, RuleKey>::default();
    let permute = TreeTransformOperation::permute([1, 0], [2]);
    let braid = TreeTransformOperation::braid([1, 0], [2], [1, 0], [2]);
    for operation in [&permute, &braid] {
        dst.data_mut().copy_from_slice(&[1.0, 2.0]);
        context_tree_transform_into(
            &mut context,
            &rule,
            operation.clone(),
            &mut dst,
            &src,
            1.0,
            0.0,
        )
        .unwrap();
    }
    crate::probe::take_completed_transformer_activity();
    for operation in [&permute, &braid] {
        dst.data_mut().copy_from_slice(&[1.0, 2.0]);
        context_tree_transform_into(
            &mut context,
            &rule,
            operation.clone(),
            &mut dst,
            &src,
            1.0,
            0.0,
        )
        .unwrap();
    }

    // What: the operation is a key determinant: each operation has its own
    // published transformer, and the warm round hits both.
    let warm = crate::probe::take_completed_transformer_activity();
    assert_eq!((warm.builds, warm.hits), (0, 2));
    let resolve = |operation| {
        resolve_tree_pair(&rule, operation, dst.structure(), src.structure(), false).unwrap()
    };
    assert!(!same_core(&resolve(&permute), &resolve(&braid)));
}

// Regression (#1921): rank-5 SU(2) and Fibonacci transposes whose last
// compact step is a bend emit destination rows in first-appearance order, not
// HomSpace order. The compact plan must still replay to the same values as
// the per-pair plan in one shared destination structure.
// Why a macro, not a generic helper: the replay backend is selected per
// concrete scalar, and the two fixtures use f64 and Complex64.
macro_rules! assert_rank5_compact_transpose_replays_like_per_pair {
    ($rule:expr, $scalar:ty, $sectors:expr, $value:expr) => {{
        let rule = &$rule;
        let sectors: &[SectorId] = &$sectors;
        type T = $scalar;
        let (codomain_permutation, domain_permutation) = ([4usize, 3, 2], [1usize, 0]);
        let leg = || SectorLeg::new(sectors.iter().map(|&sector| (sector, 1)), false);
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg()]),
            FusionProductSpace::new([leg(), leg(), leg()]),
        );
        let sources = hom.fusion_tree_keys(rule).to_vec();
        let src_structure =
            packed_fixture_structure(5, sources.iter().cloned().map(|key| (key, vec![2usize; 5])))
                .unwrap();
        let operation = TreeTransformOperation::transpose(codomain_permutation, domain_permutation);
        let compact =
            build_tree_pair_transform_group_plan(rule, operation.clone(), &src_structure).unwrap();
        let per_pair = build_tree_transform_group_plan(rule, operation, &src_structure, |source| {
            symtn_core::testing::multiplicity_free_transpose_tree_pair(
                rule,
                source,
                &codomain_permutation,
                &domain_permutation,
            )
            .map_err(OperationError::from_core_preserving_context)
        })
        .unwrap();

        // One destination structure holding every destination either plan reaches.
        let mut destinations = std::collections::BTreeSet::new();
        for spec in compact.specs().iter().chain(per_pair.specs()) {
            destinations.extend(spec.dst_keys().iter().cloned());
        }
        let dst_structure = packed_fixture_structure(
            5,
            destinations.into_iter().map(|key| (key, vec![2usize; 5])),
        )
        .unwrap();
        let compact = compact
            .compile_structures(&dst_structure, &src_structure)
            .unwrap();
        let per_pair = per_pair
            .compile_structures(&dst_structure, &src_structure)
            .unwrap();

        let source_len = src_structure.required_len().unwrap();
        let source_data = (0..source_len)
            .map(|index| $value(((index * 37 + 11) % 101) as f64 / 7.0 - 5.0))
            .collect::<Vec<_>>();
        let source = TestTensor::<T>::from_vec_with_structure(
            source_data,
            [&[1; 2][..], &[1; 3][..]].concat(),
            src_structure.clone(),
        )
        .unwrap();
        let replay = |compiled: &symtn_operations::TreeTransformStructure<T>| {
            let mut output = TestTensor::<T>::from_vec_with_structure(
                vec![T::zero(); dst_structure.required_len().unwrap()],
                [&[1; 3][..], &[1; 2][..]].concat(),
                dst_structure.clone(),
            )
            .unwrap();
            replay_into(
                &mut DenseTreeTransformOperations::default(),
                &mut TreeTransformWorkspace::default(),
                compiled,
                &mut output,
                &source,
                T::one(),
                T::zero(),
            )
            .unwrap();
            output.data().to_vec()
        };
        let actual = replay(&compact);
        let expected = replay(&per_pair);
        assert_eq!(actual.len(), expected.len());
        let mut nonzero = 0;
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            let (actual, expected) = (Complex64::from(actual), Complex64::from(expected));
            nonzero += usize::from(expected.norm() > 1.0e-9);
            assert!(
                (actual - expected).norm() <= 1.0e-10 * (1.0 + expected.norm()),
                "element {index}: compact {actual} vs per-pair {expected}"
            );
        }
        assert!(nonzero > 0, "replay must exercise nonzero output");
    }};
}

#[test]
fn rank5_compact_transpose_with_trailing_bend_replays_like_per_pair() {
    assert_rank5_compact_transpose_replays_like_per_pair!(
        SU2FusionRule,
        f64,
        [SectorId::new(0), SectorId::new(1), SectorId::new(2)],
        |x: f64| x
    );
    assert_rank5_compact_transpose_replays_like_per_pair!(
        symtn_core::TableRule::fibonacci(),
        Complex64,
        [SectorId::new(0), SectorId::new(1)],
        |x: f64| Complex64::new(x, 0.5 - x / 3.0)
    );
}
