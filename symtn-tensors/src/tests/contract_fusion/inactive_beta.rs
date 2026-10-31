use super::*;

#[test]
fn tensorcontract_fusion_block_replay_scales_inactive_dst_blocks_once() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rule = Z2FusionRule;
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let leg = || SectorLeg::new([(even, 1), (odd, 1)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg()]),
        FusionProductSpace::new([leg()]),
    );
    let keys = homspace.fusion_tree_keys(&rule);
    let key_for_sector = |sector| {
        keys.iter()
            .find(|key| key.codomain_tree().coupled() == sector)
            .cloned()
            .expect("Z2 one-leg homspace contains requested sector")
    };
    let even_key = key_for_sector(even);
    let odd_key = key_for_sector(odd);

    let lhs_space = DynamicFusionMapSpace::try_from_layout(
        homspace.clone(),
        packed_fixture_structure(2, [(even_key.clone(), vec![1, 1])]).unwrap(),
    )
    .unwrap()
    .try_bind_rule(&rule)
    .unwrap();
    let rhs_space = DynamicFusionMapSpace::try_from_layout(
        homspace.clone(),
        packed_fixture_structure(2, [(even_key.clone(), vec![1, 1])]).unwrap(),
    )
    .unwrap()
    .try_bind_rule(&rule)
    .unwrap();
    let dst_space = DynamicFusionMapSpace::try_from_layout(
        homspace,
        packed_fixture_structure(2, [(even_key, vec![1, 1]), (odd_key, vec![1, 1])]).unwrap(),
    )
    .unwrap()
    .try_bind_rule(&rule)
    .unwrap();

    let lhs = TestTensor::<f64>::from_vec_with_fusion_space(vec![2.0], lhs_space).unwrap();
    let rhs = TestTensor::<f64>::from_vec_with_fusion_space(vec![5.0], rhs_space).unwrap();
    let mut dst =
        TestTensor::<f64>::from_vec_with_fusion_space(vec![10.0, 20.0], dst_space).unwrap();
    let axes = TensorContractSpec::with_default_output_order(&[1], &[0]);
    let alpha = 2.0;
    let beta = 3.0;

    let specs = tensorcontract_fusion_block_specs(
        &rule,
        dst.fusion_space().unwrap(),
        lhs.fusion_space().unwrap(),
        rhs.fusion_space().unwrap(),
        axes,
    )
    .unwrap();
    assert_eq!(specs, vec![TensorContractBlockSpec::new(0, 0, 0)]);

    let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
    context_contract_into(
        engine,
        &mut context,
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        axes,
        alpha,
        beta,
    )
    .unwrap();
    // What: an irregular packed Core plan replays the active block and scales
    // the inactive one by beta.
    assert!(context.last_resolution_is_core());
    assert_eq!(dst.data(), &[50.0, 60.0]);
}

#[test]
fn self_dual_conjugate_sparse_source_scales_inactive_block_like_eager_adjoint_oracle() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = Z2FusionRule;
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let leg = || SectorLeg::new([(even, 1), (odd, 1)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg()]),
        FusionProductSpace::new([leg()]),
    );
    let keys = homspace.fusion_tree_keys(&rule);
    let key_for_sector = |sector| {
        keys.iter()
            .find(|key| key.codomain_tree().coupled() == sector)
            .cloned()
            .expect("Z2 one-leg homspace contains requested sector")
    };
    let even_key = key_for_sector(even);
    let odd_key = key_for_sector(odd);
    // What: the eager planner consumes a sparse physical subset
    // of the broad Z2 HomSpace, while the eager-adjoint oracle uses its
    // equivalent canonical even-only source.
    let source_leg = || SectorLeg::new([(even, 1)], false);
    let source_homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([source_leg()]),
        FusionProductSpace::new([source_leg()]),
    );
    let source_space = || {
        DynamicFusionMapSpace::try_from_layout(
            homspace.clone(),
            packed_fixture_structure(2, [(even_key.clone(), vec![1, 1])]).unwrap(),
        )
        .unwrap()
        .try_bind_rule(&rule)
        .unwrap()
    };
    let dst_space = DynamicFusionMapSpace::try_from_layout(
        homspace.clone(),
        packed_fixture_structure(2, [(even_key.clone(), vec![1, 1]), (odd_key, vec![1, 1])])
            .unwrap(),
    )
    .unwrap()
    .try_bind_rule(&rule)
    .unwrap();
    let lhs_space = source_space();
    let rhs_space = source_space();
    let oracle_source_space =
        || DynamicFusionMapSpace::from_final_homspace(&rule, source_homspace.clone()).unwrap();
    let lhs_dynamic = oracle_source_space();
    let rhs_dynamic = oracle_source_space();
    let lhs_data = vec![Complex64::new(2.0, 3.0)];
    let rhs_data = vec![Complex64::new(5.0, -1.0)];
    let initial = vec![Complex64::new(10.0, 4.0), Complex64::new(20.0, -2.0)];
    let alpha = Complex64::new(0.75, -0.25);
    let beta = Complex64::new(0.5, 0.125);

    let (adjoint_space, adjoint_data) =
        crate::adjoint::adjoint_dyn(&rule, &lhs_dynamic, &lhs_data).unwrap();
    let oracle_dst_space =
        crate::DynamicFusionMapSpace::contracted(&rule, &adjoint_space, &rhs_dynamic, &[1], &[0])
            .unwrap();
    let provider = Arc::new(rule);
    let oracle_dst_bound = crate::BoundDynamicFusionMapSpace::bind_multiplicity_free(
        oracle_dst_space,
        Arc::clone(&provider),
    )
    .unwrap();
    let adjoint_bound = crate::BoundDynamicFusionMapSpace::bind_multiplicity_free(
        adjoint_space,
        Arc::clone(&provider),
    )
    .unwrap();
    let rhs_bound = crate::BoundDynamicFusionMapSpace::bind_multiplicity_free(
        rhs_dynamic,
        Arc::clone(&provider),
    )
    .unwrap();
    let mut oracle = vec![initial[0]];
    crate::TensorContractFusionExecutionContext::<Complex64, _>::default()
        .tensorcontract_prelowered_into(
            engine,
            &oracle_dst_bound,
            &mut oracle,
            crate::FusionOperand::direct(adjoint_bound.space()),
            &adjoint_data,
            crate::FusionOperand::direct(rhs_bound.space()),
            &rhs_data,
            TensorContractSpec::with_default_output_order(&[1], &[0]),
            alpha,
            crate::ContractDestinationInit::Axpby(beta),
        )
        .unwrap();
    // What: the independent eager contraction produces the active block; the
    // whole-destination axpby oracle also scales the absent odd block.
    oracle.push(beta * initial[1]);

    let lhs = TestTensor::<Complex64>::from_vec_with_fusion_space(lhs_data, lhs_space).unwrap();
    let rhs = TestTensor::<Complex64>::from_vec_with_fusion_space(rhs_data, rhs_space).unwrap();
    let mut actual =
        TestTensor::<Complex64>::from_vec_with_fusion_space(initial, dst_space).unwrap();
    let conjugate_axes =
        TensorContractSpec::with_default_output_order_and_conjugation(&[1], &[0], true, false);
    let mut context = TensorContractFusionExecutionContext::<Complex64, RuleIdentity>::default();
    for _ in 0..2 {
        actual
            .data_mut()
            .copy_from_slice(&[Complex64::new(10.0, 4.0), Complex64::new(20.0, -2.0)]);
        context_contract_into(
            engine,
            &mut context,
            &rule,
            &mut actual,
            &lhs,
            &rhs,
            conjugate_axes,
            alpha,
            beta,
        )
        .unwrap();
        assert_eq!(actual.data(), oracle.as_slice());
    }
}

#[test]
fn tensorcontract_fusion_block_replay_scatter_beta_supports_dense_dtypes() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    assert_fusion_block_scatter_beta_dtype(engine, 2.0_f32, 5.0, 10.0, 20.0, 2.0, 3.0);
    assert_fusion_block_scatter_beta_dtype(engine, 2.0_f64, 5.0, 10.0, 20.0, 2.0, 3.0);
    assert_fusion_block_scatter_beta_dtype(
        engine,
        Complex32::new(2.0, 1.0),
        Complex32::new(5.0, -2.0),
        Complex32::new(10.0, 1.0),
        Complex32::new(20.0, -3.0),
        Complex32::new(2.0, -1.0),
        Complex32::new(-1.0, 0.5),
    );
    assert_fusion_block_scatter_beta_dtype(
        engine,
        Complex64::new(2.0, 1.0),
        Complex64::new(5.0, -2.0),
        Complex64::new(10.0, 1.0),
        Complex64::new(20.0, -3.0),
        Complex64::new(2.0, -1.0),
        Complex64::new(-1.0, 0.5),
    );
}

fn assert_fusion_block_scatter_beta_dtype<T>(
    engine: &mut dyn Engine,
    lhs_value: T,
    rhs_value: T,
    initial_even: T,
    initial_odd: T,
    alpha: T,
    beta: T,
) where
    T: DenseBlockScalar + DenseRecouplingScalar + RecouplingCoefficientAction<f64> + Debug,
{
    let rule = Z2FusionRule;
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let leg = || SectorLeg::new([(even, 1), (odd, 1)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg()]),
        FusionProductSpace::new([leg()]),
    );
    let keys = homspace.fusion_tree_keys(&rule);
    let key_for_sector = |sector| {
        keys.iter()
            .find(|key| key.codomain_tree().coupled() == sector)
            .cloned()
            .expect("Z2 one-leg homspace contains requested sector")
    };
    let even_key = key_for_sector(even);
    let odd_key = key_for_sector(odd);

    let lhs_space = DynamicFusionMapSpace::try_from_layout(
        homspace.clone(),
        packed_fixture_structure(2, [(even_key.clone(), vec![1, 1])]).unwrap(),
    )
    .unwrap()
    .try_bind_rule(&rule)
    .unwrap();
    let rhs_space = DynamicFusionMapSpace::try_from_layout(
        homspace.clone(),
        packed_fixture_structure(2, [(even_key.clone(), vec![1, 1])]).unwrap(),
    )
    .unwrap()
    .try_bind_rule(&rule)
    .unwrap();
    let dst_space = DynamicFusionMapSpace::try_from_layout(
        homspace,
        packed_fixture_structure(2, [(even_key, vec![1, 1]), (odd_key, vec![1, 1])]).unwrap(),
    )
    .unwrap()
    .try_bind_rule(&rule)
    .unwrap();

    let lhs = TestTensor::<T>::from_vec_with_fusion_space(vec![lhs_value], lhs_space).unwrap();
    let rhs = TestTensor::<T>::from_vec_with_fusion_space(vec![rhs_value], rhs_space).unwrap();
    let mut dst =
        TestTensor::<T>::from_vec_with_fusion_space(vec![initial_even, initial_odd], dst_space)
            .unwrap();

    fusion_contract_into(
        engine,
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        TensorContractSpec::with_default_output_order(&[1], &[0]),
        alpha,
        beta,
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[
            beta * initial_even + alpha * lhs_value * rhs_value,
            beta * initial_odd
        ]
    );
}
