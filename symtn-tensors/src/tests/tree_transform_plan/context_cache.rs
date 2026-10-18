use super::*;
use crate::probe::{self, Event};

#[test]
fn expert_layout_transformers_are_rebuilt_per_call_and_never_published() {
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
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
    let space = vec![1, 1, 1, 1];
    let src = TestTensor::<f64>::from_vec_with_structure(
        vec![3.0, -4.0],
        space.clone(),
        block_structure.clone(),
    )
    .unwrap();
    let operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [23, 29, 31, 37], []);

    let calls = Arc::new(AtomicUsize::new(0));
    let rule = AdmissionCountingSu2Rule {
        nsymbol_calls: Arc::clone(&calls),
        fusion_style: None,
        braiding_style: None,
    };
    let mut first = TreeTransformExecutionContext::<f64, RuleIdentity>::default();
    crate::probe::take_completed_transformer_activity();
    let run = |context: &mut TreeTransformExecutionContext<f64, RuleIdentity>| {
        let mut dst = TestTensor::<f64>::from_vec_with_structure(
            vec![1.0, 2.0],
            space.clone(),
            block_structure.clone(),
        )
        .unwrap();
        context_tree_transform_into(context, &rule, operation.clone(), &mut dst, &src, 2.0, -1.0)
            .unwrap();
        dst.data().to_vec()
    };
    let first_data = run(&mut first);
    assert!(calls.load(Ordering::Relaxed) > 0);

    calls.store(0, Ordering::Relaxed);
    let again = run(&mut first);
    assert!(calls.load(Ordering::Relaxed) > 0);
    let mut second = TreeTransformExecutionContext::<f64, RuleIdentity>::default();
    let second_data = run(&mut second);

    // What: a packed expert layout is not canonical (no complete-HomSpace
    // entry owns it), so its keys are lookup-only: every call builds, the
    // same context included, nothing is published, and the result is the
    // same each time (only canonical content is publishable).
    let activity = crate::probe::take_completed_transformer_activity();
    assert_eq!(
        (activity.hits, activity.builds, activity.publications),
        (0, 3, 0)
    );
    assert_eq!(again, first_data);
    assert_eq!(second_data, first_data);
}

#[test]
fn su2_two_by_two_f_move_uses_one_completed_structure_miss_compiler() {
    // Composed-coefficient hits must not race a clearing sibling test.
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

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
    let src = TestTensor::<f64>::from_vec_with_structure(
        vec![10.0, 20.0],
        src_space,
        block_structure.clone(),
    )
    .unwrap();
    let dst =
        TestTensor::<f64>::from_vec_with_structure(vec![0.0, 0.0], dst_space, block_structure)
            .unwrap();
    let direct_operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
    let miss_operation = TreeTransformOperation::braid(vec![0, 2, 1, 3], [], vec![0, 1, 2, 3], []);
    let hit_operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
    assert_eq!(direct_operation, miss_operation);
    assert_eq!(miss_operation, hit_operation);
    let group_count = src.structure().fusion_tree_groups().len();
    let mut planning = crate::tree_transform::TreeTransformPlanning::default();

    let tree_pair_lowering_calls_rec = probe::record();
    let direct =
        uncached_tree_pair_structure(&SU2FusionRule, direct_operation, &dst, &src).unwrap();
    assert_eq!(
        tree_pair_lowering_calls_rec.get(Event::OrderedTreePairLowering),
        group_count
    );

    let tree_pair_lowering_calls_rec = probe::record();
    crate::probe::take_coefficient_group_activity();
    let miss = planning
        .resolve_tree_pair(
            &SU2FusionRule,
            &miss_operation,
            dst.structure(),
            src.structure(),
            false,
        )
        .unwrap();
    // One lowering per group the composed-coefficient cache missed (a
    // sibling test may already have published some of these groups).
    let groups = crate::probe::take_coefficient_group_activity();
    assert_eq!(groups.hits + groups.misses, group_count);
    assert_eq!(
        tree_pair_lowering_calls_rec.get(Event::OrderedTreePairLowering),
        groups.misses
    );

    let tree_pair_lowering_calls_rec = probe::record();
    let rebuilt = planning
        .resolve_tree_pair(
            &SU2FusionRule,
            &hit_operation,
            dst.structure(),
            src.structure(),
            false,
        )
        .unwrap();
    // The packed fixture is an expert layout, so the second resolution
    // rebuilds its transformer (lookup-only key), but every group's
    // coefficients come from the composed-coefficient cache: no lowering.
    assert_eq!(
        tree_pair_lowering_calls_rec.get(Event::OrderedTreePairLowering),
        0
    );
    let groups = crate::probe::take_coefficient_group_activity();
    assert_eq!((groups.hits, groups.misses), (group_count, 0));
    let hit = rebuilt;
    assert!(direct.has_pack_gemm_scatter_blocks());
    assert_eq!(direct.blocks(), miss.blocks());
    assert_eq!(direct.layouts(), miss.layouts());
    assert_eq!(miss.blocks(), hit.blocks());
    assert_eq!(miss.layouts(), hit.layouts());
    let coefficient_bits = |structure: &TreeTransformStructure<f64>| {
        structure
            .gathered_coefficients()
            .iter()
            .map(|coefficient| coefficient.to_bits())
            .collect::<Vec<_>>()
    };
    assert_eq!(coefficient_bits(&direct), coefficient_bits(&miss));
    assert_eq!(coefficient_bits(&miss), coefficient_bits(&hit));

    let replay = |structure: &TreeTransformStructure<f64>| {
        let mut output = dst.clone();
        let mut backend = DenseTreeTransformOperations::default();
        let mut workspace = TreeTransformWorkspace::default();
        replay_into(
            &mut backend,
            &mut workspace,
            structure,
            &mut output,
            &src,
            1.0,
            0.0,
        )
        .unwrap();
        output.data().to_vec()
    };
    let direct_output = replay(&direct);
    let miss_output = replay(&miss);
    let hit_output = replay(&hit);

    let replay_bits = |values: &[f64]| {
        values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    };
    assert_eq!(replay_bits(&direct_output), replay_bits(&miss_output));
    assert_eq!(replay_bits(&miss_output), replay_bits(&hit_output));
    assert!((direct_output[0] - 22.320_508_075_688_77).abs() < 1.0e-12);
    assert!((direct_output[1] + 1.339_745_962_155_612_7).abs() < 1.0e-12);
}

#[test]
fn all_codomain_plan_compiles_distinct_degeneracy_shapes() {
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
    let small_structure = packed_fixture_structure(
        4,
        [
            (src_key0.clone(), vec![1, 1, 1, 1]),
            (src_key1.clone(), vec![1, 1, 1, 1]),
        ],
    )
    .unwrap();
    let large_structure = packed_fixture_structure(
        4,
        [(src_key0, vec![2, 1, 1, 1]), (src_key1, vec![2, 1, 1, 1])],
    )
    .unwrap();
    let small_space = vec![1, 1, 1, 1];
    let large_space = vec![2, 1, 1, 1];
    let src = TestTensor::<f64>::from_vec_with_structure(
        vec![10.0, 20.0],
        small_space.clone(),
        small_structure.clone(),
    )
    .unwrap();
    let mut dst =
        TestTensor::<f64>::from_vec_with_structure(vec![0.0, 0.0], small_space, small_structure)
            .unwrap();
    let src_large = TestTensor::<f64>::from_vec_with_structure(
        vec![1.0, 2.0, 3.0, 4.0],
        large_space.clone(),
        large_structure.clone(),
    )
    .unwrap();
    let dst_large = TestTensor::<f64>::from_vec_with_structure(
        vec![0.0, 0.0, 0.0, 0.0],
        large_space,
        large_structure,
    )
    .unwrap();
    let operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
    let resolve = |dst: &TestTensor<f64>, src: &TestTensor<f64>| {
        uncached_all_codomain_structure(&SU2FusionRule, operation.clone(), dst, src).unwrap()
    };

    let small = resolve(&dst, &src);
    let large = resolve(&dst_large, &src_large);
    assert!(small.has_pack_gemm_scatter_blocks() && large.has_pack_gemm_scatter_blocks());
    // What: a degeneracy change is a distinct layout, hence a distinct core.
    assert_ne!(small.layouts(), large.layouts());

    let structure = resolve(&dst, &src);
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

    assert!((dst.data()[0] - 22.320_508_075_688_77).abs() < 1.0e-12);
    assert!((dst.data()[1] + 1.339_745_962_155_612_7).abs() < 1.0e-12);
}

#[test]
fn all_codomain_plan_replays_su2_recoupling() {
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
    let space = vec![1, 1, 1, 1];
    let mut src = TestTensor::<f64>::from_vec_with_structure(
        vec![10.0, 20.0],
        space.clone(),
        block_structure.clone(),
    )
    .unwrap();
    let mut dst =
        TestTensor::<f64>::from_vec_with_structure(vec![0.0, 0.0], space, block_structure).unwrap();
    let operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
    let structure = uncached_all_codomain_structure(&SU2FusionRule, operation, &dst, &src).unwrap();
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

    assert!((dst.data()[0] - 22.320_508_075_688_77).abs() < 1.0e-12);
    assert!((dst.data()[1] + 1.339_745_962_155_612_7).abs() < 1.0e-12);

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

    let c = 0.866_025_403_784_438_6;
    assert!((dst.data()[0] - (-1.0 + 2.0 * (0.5 * 3.0 + c * -4.0))).abs() < 1.0e-12);
    assert!((dst.data()[1] - (-2.0 + 2.0 * (c * 3.0 - 0.5 * -4.0))).abs() < 1.0e-12);
}

#[test]
fn tree_pair_and_all_codomain_transforms_agree_on_an_all_codomain_source() {
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
    let space = vec![1, 1, 1, 1];
    let src = TestTensor::<f64>::from_vec_with_structure(
        vec![10.0, 20.0],
        space.clone(),
        block_structure.clone(),
    )
    .unwrap();
    let mut dst =
        TestTensor::<f64>::from_vec_with_structure(vec![0.0, 0.0], space, block_structure).unwrap();
    let operation = TreeTransformOperation::braid([0, 2, 1, 3], [], [0, 1, 2, 3], []);
    let mut context = TreeTransformExecutionContext::<f64, RuleIdentity>::default();

    context_tree_transform_into(
        &mut context,
        &SU2FusionRule,
        operation.clone(),
        &mut dst,
        &src,
        1.0,
        0.0,
    )
    .unwrap();
    let tree_pair = dst.data().to_vec();

    dst.data_mut().copy_from_slice(&[0.0, 0.0]);
    let structure = uncached_all_codomain_structure(&SU2FusionRule, operation, &dst, &src).unwrap();
    replay_into(
        &mut DenseTreeTransformOperations::default(),
        &mut TreeTransformWorkspace::default(),
        &structure,
        &mut dst,
        &src,
        1.0,
        0.0,
    )
    .unwrap();

    // What: the tree-pair and all-codomain lowerings agree here.
    assert_eq!(tree_pair, dst.data());
    assert!((dst.data()[0] - 22.320_508_075_688_77).abs() < 1.0e-12);
    assert!((dst.data()[1] + 1.339_745_962_155_612_7).abs() < 1.0e-12);
}
