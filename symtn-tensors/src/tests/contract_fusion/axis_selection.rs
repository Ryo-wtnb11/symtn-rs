use super::*;
use crate::contract::FusionContractOrientation;
use crate::probe::{self, Event};
use symtn_dense::Engine;

fn scored_candidates(lhs: &[usize], rhs: &[usize]) -> usize {
    crate::contract::contracted_axis_order_candidates(lhs, rhs).len()
        * [
            FusionContractOrientation::LhsRhs,
            FusionContractOrientation::RhsLhs,
        ]
        .len()
}

#[test]
fn forced_axis_order_candidates_have_identical_u1_result() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: both paired caller orders and both operand orientations produce
    // the same U1 contraction result.
    let rule = U1FusionRule;
    let sectors = [
        U1Irrep::new(-1).sector_id(),
        U1Irrep::new(0).sector_id(),
        U1Irrep::new(1).sector_id(),
    ];
    let leg = || SectorLeg::new(sectors.map(|sector| (sector, 1)), false);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let space = DynamicFusionMapSpace::from_final_homspace(&rule, hom).unwrap();
    let len = space.structure().required_len().unwrap();
    let lhs = TestTensor::<f64>::from_vec_with_fusion_space(
        (0..len).map(|i| i as f64 + 1.0).collect(),
        space.clone(),
    )
    .unwrap();
    let rhs = TestTensor::<f64>::from_vec_with_fusion_space(
        (0..len).map(|i| 2.0 * i as f64 - 0.5).collect(),
        space.clone(),
    )
    .unwrap();
    let candidates = crate::contract::contracted_axis_order_candidates(&[3, 2], &[0, 1]);
    assert!(candidates.len() >= 2);
    let dst_dyn = space.clone();
    let lhs_dyn = space.clone();
    let rhs_dyn = space.clone();
    let mut outputs = Vec::new();
    for candidate in candidates.iter().take(2) {
        for orientation in [
            crate::contract::FusionContractOrientation::LhsRhs,
            crate::contract::FusionContractOrientation::RhsLhs,
        ] {
            let plan = crate::contract::prepare_tensorcontract_fusion_plan_dyn_raw_with_axis_order_and_orientation(
                &rule,
                &dst_dyn,
                &lhs_dyn,
                &rhs_dyn,
                TensorContractSpec::new(&[3, 2], &[0, 1], OutputAxisOrder::from_axes(&[0, 1, 2, 3])),
                candidate,
                orientation,
            )
            .unwrap();
            let mut dst =
                TestTensor::<f64>::from_vec_with_fusion_space(vec![0.0; len], space.clone())
                    .unwrap();
            crate::contract::execute_dynamic_tree_execution_artifact_for_test(
                engine, &rule, &plan, &mut dst, &lhs, &rhs, 1.0, 0.0,
            )
            .unwrap();
            outputs.push(dst);
        }
    }
    assert_eq!(outputs.len(), 4);
    for output in &outputs[1..] {
        assert_eq!(outputs[0].fusion_space(), output.fusion_space());
        for (a, b) in outputs[0].data().iter().zip(output.data()) {
            assert!((a - b).abs() < 1.0e-10, "candidate mismatch: {a} vs {b}");
        }
    }
    let selected = fixture_contract_plan(
        &rule,
        &space.clone(),
        &space.clone(),
        &space.clone(),
        TensorContractSpec::new(&[3, 2], &[0, 1], OutputAxisOrder::from_axes(&[0, 1, 2, 3])),
    )
    .unwrap();
    assert_eq!(
        selected.orientation(),
        crate::contract::FusionContractOrientation::LhsRhs
    );
    assert_eq!(
        selected.lhs_transform().kind(),
        TreeTransformOperationKind::Permute
    );
    assert_eq!(selected.lhs_transform().domain_permutation(), &[2, 3]);
    assert_eq!(
        selected.rhs_transform().kind(),
        TreeTransformOperationKind::Permute
    );
    assert_eq!(selected.rhs_transform().codomain_permutation(), &[1, 0]);
    let mut rhs_permuted =
        TestTensor::<f64>::from_vec_with_fusion_space(vec![0.0; len], space.clone()).unwrap();
    eager_tree_transform_into(
        &rule,
        TreeTransformOperation::permute([1, 0], [2, 3]),
        &mut rhs_permuted,
        &rhs,
        1.0,
        0.0,
    )
    .unwrap();
    let mut oracle =
        TestTensor::<f64>::from_vec_with_fusion_space(vec![0.0; len], space.clone()).unwrap();
    fusion_contract_into(
        engine,
        &rule,
        &mut oracle,
        &lhs,
        &rhs_permuted,
        TensorContractSpec::with_default_output_order(&[2, 3], &[0, 1]),
        1.0,
        0.0,
    )
    .unwrap();
    let mut normal = TestTensor::<f64>::from_vec_with_fusion_space(vec![0.0; len], space).unwrap();
    fusion_contract_into(
        engine,
        &rule,
        &mut normal,
        &lhs,
        &rhs,
        TensorContractSpec::new(&[3, 2], &[0, 1], OutputAxisOrder::from_axes(&[0, 1, 2, 3])),
        1.0,
        0.0,
    )
    .unwrap();
    // What: compiling the selected forward orientation preserves its eager bytes.
    assert_eq!(normal.data(), outputs[0].data());
    for ((actual, expected), fixed) in normal
        .data()
        .iter()
        .zip(oracle.data())
        .zip(outputs[0].data())
    {
        assert!((*actual - *expected).abs() < 1.0e-10);
        assert!((*actual - *fixed).abs() < 1.0e-10);
    }
}

#[test]
fn forced_orientations_match_asymmetric_u1_reduced_block_oracle() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rule = U1FusionRule;
    let neutral = U1Irrep::new(0).sector_id();
    let leg = |dimension| SectorLeg::new([(neutral, dimension)], false);
    let lhs_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(2), leg(3)]),
        FusionProductSpace::new([leg(4), leg(5)]),
    );
    let rhs_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(5), leg(4)]),
        FusionProductSpace::new([leg(6), leg(7)]),
    );
    let output_axes = [1, 0, 3, 2];
    let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        &lhs_hom,
        &rhs_hom,
        &[3, 2],
        &[0, 1],
        &output_axes,
        2,
    )
    .unwrap();
    let lhs_space = DynamicFusionMapSpace::from_final_homspace(&rule, lhs_hom).unwrap();
    let rhs_space = DynamicFusionMapSpace::from_final_homspace(&rule, rhs_hom).unwrap();
    let dst_space = DynamicFusionMapSpace::from_final_homspace(&rule, dst_hom).unwrap();
    let lhs = TestTensor::<f64>::from_vec_with_fusion_space(
        (0..lhs_space.required_len().unwrap())
            .map(|index| index as f64 * 0.125 - 2.0)
            .collect(),
        lhs_space,
    )
    .unwrap();
    let rhs = TestTensor::<f64>::from_vec_with_fusion_space(
        (0..rhs_space.required_len().unwrap())
            .map(|index| 1.5 - index as f64 * 0.0625)
            .collect(),
        rhs_space,
    )
    .unwrap();
    let initial = (0..dst_space.required_len().unwrap())
        .map(|index| index as f64 * 0.25 + 0.5)
        .collect::<Vec<_>>();
    let alpha = 1.25;
    let beta = -0.75;
    let lhs_block = lhs.structure().block(0).unwrap();
    let rhs_block = rhs.structure().block(0).unwrap();
    let dst_block = dst_space.structure().block(0).unwrap();
    let mut expected = initial.iter().map(|value| beta * value).collect::<Vec<_>>();
    for l0 in 0..2 {
        for l1 in 0..3 {
            for r2 in 0..6 {
                for r3 in 0..7 {
                    let mut sum = 0.0;
                    for c0 in 0..4 {
                        for c1 in 0..5 {
                            let lhs_offset = lhs_block.offset()
                                + [l0, l1, c0, c1]
                                    .iter()
                                    .zip(lhs_block.strides())
                                    .map(|(&index, &stride)| index * stride)
                                    .sum::<usize>();
                            let rhs_offset = rhs_block.offset()
                                + [c1, c0, r2, r3]
                                    .iter()
                                    .zip(rhs_block.strides())
                                    .map(|(&index, &stride)| index * stride)
                                    .sum::<usize>();
                            sum += lhs.data()[lhs_offset] * rhs.data()[rhs_offset];
                        }
                    }
                    let dst_offset = dst_block.offset()
                        + [l1, l0, r3, r2]
                            .iter()
                            .zip(dst_block.strides())
                            .map(|(&index, &stride)| index * stride)
                            .sum::<usize>();
                    expected[dst_offset] += alpha * sum;
                }
            }
        }
    }

    let axes = TensorContractSpec::new(&[3, 2], &[0, 1], OutputAxisOrder::from_axes(&output_axes));
    let candidates = crate::contract::contracted_axis_order_candidates(&[3, 2], &[0, 1]);
    let dst_dyn = dst_space.clone();
    let lhs_dyn = lhs.fusion_space().unwrap().as_ref().clone();
    let rhs_dyn = rhs.fusion_space().unwrap().as_ref().clone();
    for candidate in candidates.iter().take(2) {
        for orientation in [
            crate::contract::FusionContractOrientation::LhsRhs,
            crate::contract::FusionContractOrientation::RhsLhs,
        ] {
            let plan = crate::contract::prepare_tensorcontract_fusion_plan_dyn_raw_with_axis_order_and_orientation(
                &rule,
                &dst_dyn,
                &lhs_dyn,
                &rhs_dyn,
                axes,
                candidate,
                orientation,
            )
            .unwrap();
            let mut dst =
                TestTensor::<f64>::from_vec_with_fusion_space(initial.clone(), dst_space.clone())
                    .unwrap();
            crate::contract::execute_dynamic_tree_execution_artifact_for_test(
                engine, &rule, &plan, &mut dst, &lhs, &rhs, alpha, beta,
            )
            .unwrap();
            for (&actual, &expected) in dst.data().iter().zip(&expected) {
                assert!((actual - expected).abs() < 1.0e-10);
            }
        }
    }
}

#[test]
fn paired_axis_selector_scores_once_and_publishes_only_winner_replay() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = SU2FusionRule;
    let leg = || SectorLeg::new([(SU2Irrep::from_twice_spin(0).sector_id(), 1)], false);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let space = DynamicFusionMapSpace::from_final_homspace(&rule, hom).unwrap();
    let lhs = TestTensor::<f64>::from_vec_with_fusion_space(vec![2.0], space.clone()).unwrap();
    let rhs = TestTensor::<f64>::from_vec_with_fusion_space(vec![3.0], space.clone()).unwrap();
    let mut dst = TestTensor::<f64>::from_vec_with_fusion_space(vec![0.0], space).unwrap();
    let axes = TensorContractSpec::with_default_output_order(&[3, 2], &[0, 1]);
    let source_layout_homspace_id_comparisons_rec = probe::record();
    let _pure_plan = fixture_contract_plan(
        &rule,
        dst.fusion_space().unwrap(),
        lhs.fusion_space().unwrap(),
        rhs.fusion_space().unwrap(),
        axes,
    )
    .unwrap();
    // What: cold candidate scoring never enters the runtime HomSpace identity path.
    assert_eq!(
        source_layout_homspace_id_comparisons_rec.get(Event::SourceLayoutHomspaceIdComparison),
        0
    );
    symtn_core::clear_structure_caches();
    let candidate_score_calls_rec = probe::record();
    let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
    crate::probe::take_completed_transformer_activity();

    context_contract_into(
        engine,
        &mut context,
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        axes,
        1.0,
        0.0,
    )
    .unwrap();
    // What: eager resolution scores each candidate once and builds only the
    // reusable completed transformers, which it publishes.
    let scored = scored_candidates(&[3, 2], &[0, 1]);
    assert_eq!(candidate_score_calls_rec.get(Event::CandidateScore), scored);
    // The canonical fixture may be shared with a sibling test that already
    // published one of these keys.
    let cold = crate::probe::take_completed_transformer_activity();
    assert_eq!(cold.publications + cold.hits, 2, "{cold:?}");

    context_contract_into(
        engine,
        &mut context,
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        axes,
        1.0,
        0.0,
    )
    .unwrap();
    // What: an ordinary repeat resolves eagerly and hits the published
    // transformers instead of rebuilding them.
    assert_eq!(
        candidate_score_calls_rec.get(Event::CandidateScore),
        2 * scored
    );
    let warm = crate::probe::take_completed_transformer_activity();
    assert_eq!((warm.builds, warm.hits), (0, 2));
}

#[test]
fn reverse_winner_is_independent_of_first_cached_consumer() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    struct RejectingStorageGemm;

    impl symtn_operations::fusion_replay::StorageGemm<f64, Vec<f64>, Vec<f64>, Vec<f64>>
        for RejectingStorageGemm
    {
        fn matmul_range_into(
            &mut self,
            _dst: &mut Vec<f64>,
            _dst_offset: usize,
            _lhs: &Vec<f64>,
            _lhs_offset: usize,
            _rhs: &Vec<f64>,
            _rhs_offset: usize,
            _rows: usize,
            _contracted: usize,
            _cols: usize,
        ) -> Result<(), OperationError> {
            panic!("reverse storage-direct resolution must reject before GEMM")
        }
    }

    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = U1FusionRule;
    let space = |dimensions: [usize; 4]| {
        let homspace = FusionTreeHomSpace::from_sector_ids(
            [(0, dimensions[0]), (0, dimensions[1])],
            [(0, dimensions[2]), (0, dimensions[3])],
        );
        DynamicFusionMapSpace::from_final_homspace(&rule, homspace).unwrap()
    };
    let lhs_space = space([2, 3, 5, 7]);
    let rhs_space = space([11, 13, 3, 2]);
    let dst_space = space([11, 13, 5, 7]);
    let lhs = TestTensor::<f64>::from_vec_with_fusion_space(
        (0..lhs_space.required_len().unwrap())
            .map(|index| index as f64 * 0.125 - 3.0)
            .collect(),
        lhs_space,
    )
    .unwrap();
    let rhs = TestTensor::<f64>::from_vec_with_fusion_space(
        (0..rhs_space.required_len().unwrap())
            .map(|index| 2.0 - index as f64 * 0.03125)
            .collect(),
        rhs_space,
    )
    .unwrap();
    let initial = (0..dst_space.required_len().unwrap())
        .map(|index| index as f64 * 0.0625 - 1.0)
        .collect::<Vec<_>>();
    let alpha = 1.25;
    let beta = -0.5;
    let axes =
        || TensorContractSpec::new(&[1, 0], &[2, 3], OutputAxisOrder::from_axes(&[2, 3, 0, 1]));

    let lhs_block = lhs.structure().block(0).unwrap();
    let rhs_block = rhs.structure().block(0).unwrap();
    let dst_block = dst_space.structure().block(0).unwrap();
    let mut oracle = initial.iter().map(|value| beta * value).collect::<Vec<_>>();
    for r0 in 0..11 {
        for r1 in 0..13 {
            for l2 in 0..5 {
                for l3 in 0..7 {
                    let mut sum = 0.0;
                    for c0 in 0..2 {
                        for c1 in 0..3 {
                            let lhs_offset = lhs_block.offset()
                                + [c0, c1, l2, l3]
                                    .iter()
                                    .zip(lhs_block.strides())
                                    .map(|(&index, &stride)| index * stride)
                                    .sum::<usize>();
                            let rhs_offset = rhs_block.offset()
                                + [r0, r1, c1, c0]
                                    .iter()
                                    .zip(rhs_block.strides())
                                    .map(|(&index, &stride)| index * stride)
                                    .sum::<usize>();
                            sum += lhs.data()[lhs_offset] * rhs.data()[rhs_offset];
                        }
                    }
                    let dst_offset = dst_block.offset()
                        + [r0, r1, l2, l3]
                            .iter()
                            .zip(dst_block.strides())
                            .map(|(&index, &stride)| index * stride)
                            .sum::<usize>();
                    oracle[dst_offset] += alpha * sum;
                }
            }
        }
    }
    let assert_oracle = |actual: &[f64]| {
        for (index, (&actual, &expected)) in actual.iter().zip(&oracle).enumerate() {
            assert!(
                (actual - expected).abs() < 1.0e-9,
                "element {index}: {actual} != {expected}"
            );
        }
    };
    let lhs_dynamic = lhs.fusion_space().unwrap().as_ref().clone();
    let rhs_dynamic = rhs.fusion_space().unwrap().as_ref().clone();
    let dst_dynamic = dst_space.clone();
    let reverse_plan =
        crate::contract::prepare_tensorcontract_fusion_plan_dyn_raw_with_axis_order_and_orientation(
            &rule,
            &dst_dynamic,
            &lhs_dynamic,
            &rhs_dynamic,
            axes(),
            &crate::contract::contracted_axis_order_candidates(&[1, 0], &[2, 3])[1],
            crate::contract::FusionContractOrientation::RhsLhs,
        )
        .unwrap();
    // What: the reverse-orientation plan, executed step by step, matches the
    // reduced-block oracle.
    let mut explicit_dst =
        TestTensor::<f64>::from_vec_with_fusion_space(initial.clone(), dst_space.clone()).unwrap();
    plan_contract_into(
        &rule,
        &reverse_plan,
        &mut explicit_dst,
        &lhs,
        &rhs,
        alpha,
        beta,
    )
    .unwrap();
    assert_oracle(explicit_dst.data());

    let fresh_dst = || {
        TestTensor::<f64>::from_vec_with_fusion_space(initial.clone(), dst_space.clone()).unwrap()
    };

    let candidate_score_calls_rec = probe::record();
    let mut one_shot_dst = fresh_dst();
    fusion_contract_into(
        engine,
        &rule,
        &mut one_shot_dst,
        &lhs,
        &rhs,
        axes(),
        alpha,
        beta,
    )
    .unwrap();
    assert_eq!(
        candidate_score_calls_rec.get(Event::CandidateScore),
        scored_candidates(&[1, 0], &[2, 3])
    );
    assert_oracle(one_shot_dst.data());

    let provider = Arc::new(rule);
    let lhs_bound = BoundDynamicFusionMapSpace::bind_multiplicity_free(
        lhs.fusion_space().unwrap().as_ref().clone(),
        Arc::clone(&provider),
    )
    .unwrap();
    let rhs_bound = BoundDynamicFusionMapSpace::bind_multiplicity_free(
        rhs.fusion_space().unwrap().as_ref().clone(),
        Arc::clone(&provider),
    )
    .unwrap();
    let dst_bound = BoundDynamicFusionMapSpace::bind_multiplicity_free(
        dst_space.clone(),
        Arc::clone(&provider),
    )
    .unwrap();
    let unsupported = OperationError::UnsupportedTensorContractScope {
        message: "storage-direct contraction supports only canonical fully-direct oriented \
                  operands",
    };

    // The storage-direct call is context-free, so it only needs to decline
    // without touching its destination; ordering against the context cache
    // is covered by the typed/dynamic loops below.
    symtn_core::clear_structure_caches();
    let mut storage_dst = vec![0.0; initial.len()];
    assert_eq!(
        crate::contract::tensorcontract_fusion_dyn_prelowered_direct_on_storage(
            &mut RejectingStorageGemm,
            &dst_bound,
            &mut storage_dst,
            crate::FusionOperand::direct(lhs_bound.space()),
            &lhs.data().to_vec(),
            crate::FusionOperand::direct(rhs_bound.space()),
            &rhs.data().to_vec(),
            axes(),
        ),
        Err(unsupported)
    );
    assert_eq!(storage_dst, vec![0.0; initial.len()]);

    for typed_first in [true, false] {
        symtn_core::clear_structure_caches();
        let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
        let mut typed_dst =
            TestTensor::<f64>::from_vec_with_fusion_space(initial.clone(), dst_space.clone())
                .unwrap();
        let mut dynamic_dst = initial.clone();
        let typed_call = |engine: &mut dyn Engine,
                          context: &mut TensorContractFusionExecutionContext<f64, RuleIdentity>,
                          dst: &mut TestTensor<f64>| {
            context_contract_into(
                engine,
                context,
                provider.as_ref(),
                dst,
                &lhs,
                &rhs,
                axes(),
                alpha,
                beta,
            )
        };
        let dynamic_call =
            |engine: &mut dyn Engine,
             context: &mut TensorContractFusionExecutionContext<f64, RuleIdentity>,
             dst: &mut [f64]| {
                context.tensorcontract_prelowered_into(
                    engine,
                    &dst_bound,
                    dst,
                    crate::FusionOperand::direct(lhs_bound.space()),
                    lhs.data(),
                    crate::FusionOperand::direct(rhs_bound.space()),
                    rhs.data(),
                    axes(),
                    alpha,
                    crate::ContractDestinationInit::Axpby(beta),
                )
            };
        if typed_first {
            typed_call(engine, &mut context, &mut typed_dst).unwrap();
            dynamic_call(engine, &mut context, &mut dynamic_dst).unwrap();
        } else {
            dynamic_call(engine, &mut context, &mut dynamic_dst).unwrap();
            typed_call(engine, &mut context, &mut typed_dst).unwrap();
        }
        assert_oracle(typed_dst.data());
        assert_oracle(&dynamic_dst);
        assert_eq!(
            context.last_resolution_orientation(),
            Some(crate::contract::FusionContractOrientation::RhsLhs)
        );
    }

    for profiled_first in [true, false] {
        symtn_core::clear_structure_caches();
        let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
        let mut ordinary_dst =
            TestTensor::<f64>::from_vec_with_fusion_space(initial.clone(), dst_space.clone())
                .unwrap();
        let mut profiled_dst =
            TestTensor::<f64>::from_vec_with_fusion_space(initial.clone(), dst_space.clone())
                .unwrap();
        let ordinary_call =
            |engine: &mut dyn Engine,
             context: &mut TensorContractFusionExecutionContext<f64, RuleIdentity>,
             dst: &mut TestTensor<f64>| {
                context_contract_into(
                    engine,
                    context,
                    provider.as_ref(),
                    dst,
                    &lhs,
                    &rhs,
                    axes(),
                    alpha,
                    beta,
                )
            };
        let profiled_call =
            |engine: &mut dyn Engine,
             context: &mut TensorContractFusionExecutionContext<f64, RuleIdentity>,
             dst: &mut TestTensor<f64>| {
                context_contract_into(
                    engine,
                    context,
                    provider.as_ref(),
                    dst,
                    &lhs,
                    &rhs,
                    axes(),
                    alpha,
                    beta,
                )?;
                assert!(!context.last_resolution_is_core());
                assert!(context.last_resolution_orientation().is_some());
                Ok::<_, OperationError>(())
            };
        if profiled_first {
            profiled_call(engine, &mut context, &mut profiled_dst).unwrap();
            ordinary_call(engine, &mut context, &mut ordinary_dst).unwrap();
        } else {
            ordinary_call(engine, &mut context, &mut ordinary_dst).unwrap();
            profiled_call(engine, &mut context, &mut profiled_dst).unwrap();
        }
        assert_oracle(ordinary_dst.data());
        assert_oracle(profiled_dst.data());
        assert_eq!(
            context.last_resolution_orientation(),
            Some(crate::contract::FusionContractOrientation::RhsLhs)
        );
    }
}

#[test]
fn paired_axis_selector_rejects_invalid_axes_before_scoring_or_mutation() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = U1FusionRule;
    let leg = || SectorLeg::new([(U1Irrep::new(0).sector_id(), 1)], false);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let space = DynamicFusionMapSpace::from_final_homspace(&rule, hom).unwrap();
    let lhs = TestTensor::<f64>::from_vec_with_fusion_space(vec![2.0], space.clone()).unwrap();
    let rhs = TestTensor::<f64>::from_vec_with_fusion_space(vec![3.0], space.clone()).unwrap();
    let axes = TensorContractSpec::with_default_output_order(&[3, 3], &[0, 1]);
    let mut context_dst = TestTensor::<f64>::from_vec_with_fusion_space(vec![5.0], space).unwrap();
    let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
    let error = context_contract_into(
        engine,
        &mut context,
        &rule,
        &mut context_dst,
        &lhs,
        &rhs,
        axes,
        1.0,
        0.0,
    )
    .unwrap_err();
    // What: duplicate axes retain validation precedence and leave all buffers untouched.
    assert_eq!(
        error,
        OperationError::InvalidAxisSet {
            tensor: "lhs",
            axes: vec![3, 3],
            rank: 4,
        }
    );
    assert_eq!(lhs.data(), &[2.0]);
    assert_eq!(rhs.data(), &[3.0]);
    assert_eq!(context_dst.data(), &[5.0]);
}

#[test]
fn crossed_axis_selection_preserves_real_fermion_parity_complex_result() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: both crossed caller pair orders produce one odd FermionParity result.
    let rule = FermionParityFusionRule;
    let sectors = [SectorId::new(0), SectorId::new(1)];
    let leg = || SectorLeg::new(sectors.map(|sector| (sector, 1)), false);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let space = DynamicFusionMapSpace::from_final_homspace(&rule, hom).unwrap();
    let len = space.structure().required_len().unwrap();
    let lhs = TestTensor::<Complex64>::from_vec_with_fusion_space(
        (0..len)
            .map(|i| Complex64::new(i as f64 + 1.0, 0.25 * i as f64))
            .collect(),
        space.clone(),
    )
    .unwrap();
    let rhs = TestTensor::<Complex64>::from_vec_with_fusion_space(
        (0..len)
            .map(|i| Complex64::new(0.5 * i as f64 - 0.75, 1.0 - 0.1 * i as f64))
            .collect(),
        space.clone(),
    )
    .unwrap();
    let candidates = crate::contract::contracted_axis_order_candidates(&[3, 2], &[0, 1]);
    assert!(candidates.len() >= 2);
    let dst_dyn = space.clone();
    let lhs_dyn = space.clone();
    let rhs_dyn = space.clone();
    let mut outputs = Vec::new();
    for candidate in candidates.iter().take(2) {
        for orientation in [
            crate::contract::FusionContractOrientation::LhsRhs,
            crate::contract::FusionContractOrientation::RhsLhs,
        ] {
            let plan = crate::contract::prepare_tensorcontract_fusion_plan_dyn_raw_with_axis_order_and_orientation(
                &rule,
                &dst_dyn,
                &lhs_dyn,
                &rhs_dyn,
                TensorContractSpec::new(&[3, 2], &[0, 1], OutputAxisOrder::from_axes(&[0, 1, 2, 3])),
                candidate,
                orientation,
            )
            .unwrap();
            let mut dst = TestTensor::<Complex64>::from_vec_with_fusion_space(
                vec![Complex64::zero(); len],
                space.clone(),
            )
            .unwrap();
            let mut tree_backend = DenseTreeTransformOperations::default_executor();
            let mut tree_workspace = TreeTransformWorkspace::default();
            let mut contract_backend = DenseTreeTransformOperations::default_executor();
            let mut contract_workspace = TensorContractWorkspace::default();
            crate::contract::tensorcontract_fusion_dynamic_plan_into_with(
                engine,
                &mut tree_backend,
                &mut tree_workspace,
                &mut contract_backend,
                &mut contract_workspace,
                &rule,
                &plan,
                &mut dst,
                &lhs,
                &rhs,
                Complex64::one(),
                Complex64::zero(),
            )
            .unwrap();
            outputs.push(dst);
        }
    }
    assert_eq!(outputs.len(), 4);
    for output in &outputs[1..] {
        assert_eq!(outputs[0].fusion_space(), output.fusion_space());
        for (a, b) in outputs[0].data().iter().zip(output.data()) {
            assert!((a - b).norm() < 1.0e-10, "candidate mismatch: {a} vs {b}");
        }
    }
    let selected = fixture_contract_plan(
        &rule,
        &space.clone(),
        &space.clone(),
        &space.clone(),
        TensorContractSpec::new(&[3, 2], &[0, 1], OutputAxisOrder::from_axes(&[0, 1, 2, 3])),
    )
    .unwrap();
    assert_eq!(
        selected.lhs_transform().kind(),
        TreeTransformOperationKind::Permute
    );
    assert_eq!(selected.lhs_transform().domain_permutation(), &[2, 3]);
    assert_eq!(
        selected.rhs_transform().kind(),
        TreeTransformOperationKind::Permute
    );
    assert_eq!(selected.rhs_transform().codomain_permutation(), &[1, 0]);
    let mut rhs_permuted = TestTensor::<Complex64>::from_vec_with_fusion_space(
        vec![Complex64::zero(); len],
        space.clone(),
    )
    .unwrap();
    eager_tree_transform_into(
        &rule,
        TreeTransformOperation::permute([1, 0], [2, 3]),
        &mut rhs_permuted,
        &rhs,
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();
    let mut oracle = TestTensor::<Complex64>::from_vec_with_fusion_space(
        vec![Complex64::zero(); len],
        space.clone(),
    )
    .unwrap();
    fusion_contract_into(
        engine,
        &rule,
        &mut oracle,
        &lhs,
        &rhs_permuted,
        TensorContractSpec::with_default_output_order(&[2, 3], &[0, 1]),
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();
    let mut normal =
        TestTensor::<Complex64>::from_vec_with_fusion_space(vec![Complex64::zero(); len], space)
            .unwrap();
    fusion_contract_into(
        engine,
        &rule,
        &mut normal,
        &lhs,
        &rhs,
        TensorContractSpec::new(&[3, 2], &[0, 1], OutputAxisOrder::from_axes(&[0, 1, 2, 3])),
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();
    for ((actual, expected), fixed) in normal
        .data()
        .iter()
        .zip(oracle.data())
        .zip(outputs[0].data())
    {
        assert!((*actual - *expected).norm() < 1.0e-10);
        assert!((*actual - *fixed).norm() < 1.0e-10);
    }
}

#[test]
fn crossed_axis_selection_preserves_asymmetric_fz2_u1_su2_result() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let left_rule = FpU1Rule::try_new(FermionParityFusionRule, U1FusionRule).unwrap();
    let rule = FpU1Su2Rule::try_new(
        FpU1Rule::try_new(FermionParityFusionRule, U1FusionRule).unwrap(),
        SU2FusionRule,
    )
    .unwrap();
    let sector = rule.encode_component_ids(
        left_rule.encode_component_ids(SectorId::new(1), U1Irrep::new(0).sector_id()),
        SU2Irrep::from_twice_spin(1).sector_id(),
    );
    let build_space = |dimensions: [usize; 4]| {
        let leg = |dimension| SectorLeg::new([(sector, dimension)], false);
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(dimensions[0]), leg(dimensions[1])]),
            FusionProductSpace::new([leg(dimensions[2]), leg(dimensions[3])]),
        );
        DynamicFusionMapSpace::from_final_homspace(&rule, hom).unwrap()
    };
    let lhs_space = build_space([1, 1, 2, 3]);
    let rhs_space = build_space([3, 2, 4, 5]);
    let dst_space = build_space([1, 1, 4, 5]);
    let lhs_len = lhs_space.required_len().unwrap();
    let rhs_len = rhs_space.required_len().unwrap();
    let dst_len = dst_space.required_len().unwrap();
    let lhs = TestTensor::<Complex64>::from_vec_with_fusion_space(
        (0..lhs_len)
            .map(|index| Complex64::new(index as f64 + 0.5, -0.25))
            .collect(),
        lhs_space,
    )
    .unwrap();
    let rhs = TestTensor::<Complex64>::from_vec_with_fusion_space(
        (0..rhs_len)
            .map(|index| Complex64::new(1.0 - index as f64 * 0.125, 0.75))
            .collect(),
        rhs_space,
    )
    .unwrap();
    let candidates = crate::contract::contracted_axis_order_candidates(&[3, 2], &[0, 1]);
    let dst_dyn = dst_space.clone();
    let lhs_dyn = lhs.fusion_space().unwrap().as_ref().clone();
    let rhs_dyn = rhs.fusion_space().unwrap().as_ref().clone();
    let mut outputs = Vec::new();
    for candidate in &candidates {
        for orientation in [
            crate::contract::FusionContractOrientation::LhsRhs,
            crate::contract::FusionContractOrientation::RhsLhs,
        ] {
            let plan = crate::contract::prepare_tensorcontract_fusion_plan_dyn_raw_with_axis_order_and_orientation(
                &rule,
                &dst_dyn,
                &lhs_dyn,
                &rhs_dyn,
                TensorContractSpec::with_default_output_order(&[3, 2], &[0, 1]),
                candidate,
                orientation,
            )
            .unwrap();
            let mut dst = TestTensor::<Complex64>::from_vec_with_fusion_space(
                vec![Complex64::zero(); dst_len],
                dst_space.clone(),
            )
            .unwrap();
            let mut tree_backend = DenseTreeTransformOperations::default_executor();
            let mut tree_workspace = TreeTransformWorkspace::default();
            let mut contract_backend = DenseTreeTransformOperations::default_executor();
            let mut contract_workspace = TensorContractWorkspace::default();
            crate::contract::tensorcontract_fusion_dynamic_plan_into_with(
                engine,
                &mut tree_backend,
                &mut tree_workspace,
                &mut contract_backend,
                &mut contract_workspace,
                &rule,
                &plan,
                &mut dst,
                &lhs,
                &rhs,
                Complex64::one(),
                Complex64::zero(),
            )
            .unwrap();
            let mut artifact = TestTensor::<Complex64>::from_vec_with_fusion_space(
                vec![Complex64::zero(); dst_len],
                dst_space.clone(),
            )
            .unwrap();
            crate::contract::execute_dynamic_tree_execution_artifact_for_test(
                engine,
                &rule,
                &plan,
                &mut artifact,
                &lhs,
                &rhs,
                Complex64::one(),
                Complex64::zero(),
            )
            .unwrap();
            // What: compiled orientation preserves the eager destination
            // structure and every reduced-block value for the odd, charged,
            // half-spin product sector.
            assert_eq!(artifact.structure(), dst.structure());
            // Path-to-path agreement of two executors of one plan with SU2
            // recoupling, under the tolerance rule (terms bounded by
            // len(lhs) * len(rhs), recoupled trees included); the replay of the same artifact stays
            // exact.
            numerics::assert_slices_close(
                "compiled artifact vs direct plan replay",
                artifact.data(),
                dst.data(),
                lhs.data().len() * rhs.data().len(),
            );
            outputs.push(dst);
        }
    }
    // What: crossed pair order and operand orientation are immaterial for an
    // odd, charged, half-spin product sector.
    assert_eq!(outputs.len(), 4);
    for output in &outputs[1..] {
        for (lhs, rhs) in outputs[0].data().iter().zip(output.data()) {
            assert!((*lhs - *rhs).norm() < 1.0e-10);
        }
    }
    let selected = fixture_contract_plan(
        &rule,
        &dst_space.clone(),
        lhs.fusion_space().unwrap(),
        rhs.fusion_space().unwrap(),
        TensorContractSpec::with_default_output_order(&[3, 2], &[0, 1]),
    )
    .unwrap();
    assert_eq!(
        selected.lhs_transform().kind(),
        TreeTransformOperationKind::Permute
    );
    assert_eq!(selected.lhs_transform().domain_permutation(), &[3, 2]);
    assert_eq!(
        selected.rhs_transform().kind(),
        TreeTransformOperationKind::Permute
    );
    assert_eq!(selected.rhs_transform().codomain_permutation(), &[0, 1]);
    let mut lhs_permuted = TestTensor::<Complex64>::from_vec_with_fusion_space(
        vec![Complex64::zero(); lhs_len],
        build_space([1, 1, 3, 2]),
    )
    .unwrap();
    eager_tree_transform_into(
        &rule,
        TreeTransformOperation::permute([0, 1], [3, 2]),
        &mut lhs_permuted,
        &lhs,
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();
    let mut oracle = TestTensor::<Complex64>::from_vec_with_fusion_space(
        vec![Complex64::zero(); dst_len],
        dst_space.clone(),
    )
    .unwrap();
    fusion_contract_into(
        engine,
        &rule,
        &mut oracle,
        &lhs_permuted,
        &rhs,
        TensorContractSpec::with_default_output_order(&[2, 3], &[0, 1]),
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();
    let mut normal = TestTensor::<Complex64>::from_vec_with_fusion_space(
        vec![Complex64::zero(); dst_len],
        dst_space,
    )
    .unwrap();
    fusion_contract_into(
        engine,
        &rule,
        &mut normal,
        &lhs,
        &rhs,
        TensorContractSpec::with_default_output_order(&[3, 2], &[0, 1]),
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();
    for ((actual, expected), fixed) in normal
        .data()
        .iter()
        .zip(oracle.data())
        .zip(outputs[0].data())
    {
        assert!((*actual - *expected).norm() < 1.0e-10);
        assert!((*actual - *fixed).norm() < 1.0e-10);
    }
}
