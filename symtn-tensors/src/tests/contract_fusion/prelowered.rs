use super::*;

#[test]
fn prelowered_storage_layouts_and_execution_paths_match_oracle() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    symtn_core::clear_structure_caches();
    let rule = Z2FusionRule;
    let vacuum = SectorId::new(0);
    let leg = || SectorLeg::new([(vacuum, 2)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg()]),
        FusionProductSpace::new([leg()]),
    );
    let canonical_typed =
        DynamicFusionMapSpace::from_final_homspace(&rule, homspace.clone()).unwrap();
    let canonical = canonical_typed.clone();
    let canonical_block = canonical.structure().block(0).unwrap();
    let padded_structure = BlockStructure::from_blocks_with_rank(
        2,
        vec![
            BlockSpec::with_key(canonical_block.key().clone(), vec![2, 2], vec![1, 3], 1).unwrap(),
        ],
    )
    .unwrap();
    let padded = DynamicFusionMapSpace::try_from_layout(homspace, padded_structure)
        .unwrap()
        .try_bind_rule(&rule)
        .unwrap();
    assert_ne!(
        canonical.structure().content_id(),
        padded.structure().content_id()
    );

    let canonical_data = vec![1.0, 2.0, 3.0, 4.0];
    let padded_data = vec![0.0, 1.0, 2.0, 0.0, 3.0, 4.0];
    let rhs_data = vec![2.0, -1.0, 0.5, 3.0];
    let (logical_lhs, eager_lhs_data) =
        crate::adjoint::adjoint_dyn(&rule, &canonical, &canonical_data).unwrap();
    let rhs = canonical.clone();
    let dst =
        crate::DynamicFusionMapSpace::contracted(&rule, &logical_lhs, &rhs, &[1], &[0]).unwrap();
    let provider = Arc::new(rule);
    let logical_lhs_bound = crate::BoundDynamicFusionMapSpace::bind_multiplicity_free(
        logical_lhs.clone(),
        Arc::clone(&provider),
    )
    .unwrap();
    let rhs_bound = crate::BoundDynamicFusionMapSpace::bind_multiplicity_free(
        rhs.clone(),
        Arc::clone(&provider),
    )
    .unwrap();
    let dst_bound = crate::BoundDynamicFusionMapSpace::bind_multiplicity_free(
        dst.clone(),
        Arc::clone(&provider),
    )
    .unwrap();
    let canonical_operand = crate::FusionOperand::adjoint(&canonical);
    let padded_operand = crate::FusionOperand::adjoint(&padded);
    let rhs_operand = crate::FusionOperand::direct(&rhs);
    let prelowered_axes = || {
        TensorContractSpec::new_with_conjugation(
            &[1],
            &[0],
            crate::OutputAxisOrder::identity(),
            true,
            false,
        )
    };
    let ordinary_axes = || TensorContractSpec::with_default_output_order(&[1], &[0]);

    // What: A^T * B in column-major order, computed independently of every
    // symmetry-aware contraction route exercised below.
    let oracle = vec![0.0, 2.0, 6.5, 13.5];
    assert_eq!(dst.required_len().unwrap(), oracle.len());

    let swapped_dst_bound =
        crate::BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
            &logical_lhs_bound,
            &rhs_bound,
            &[1],
            &[0],
            crate::OutputAxisOrder::from_axes(&[1, 0]),
        )
        .unwrap();
    let swapped_axes =
        || TensorContractSpec::new(&[1], &[0], crate::OutputAxisOrder::from_axes(&[1, 0]));
    symtn_core::clear_structure_caches();
    let mut route_context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
    let mut ordinary_swapped = vec![0.0; swapped_dst_bound.space().required_len().unwrap()];
    route_context
        .tensorcontract_prelowered_into(
            engine,
            &swapped_dst_bound,
            &mut ordinary_swapped,
            crate::FusionOperand::direct(logical_lhs_bound.space()),
            &eager_lhs_data,
            crate::FusionOperand::direct(rhs_bound.space()),
            &rhs_data,
            swapped_axes(),
            1.0,
            crate::ContractDestinationInit::Axpby(0.0),
        )
        .unwrap();
    assert_eq!(ordinary_swapped, [0.0, 6.5, 2.0, 13.5]);
    // The plan takes the same lazy-adjoint form as `prelowered_axes`: the
    // contracted leg is the domain leg 1 of `A'`.
    let candidate = crate::contract::contracted_axis_order_candidates(&[1], &[0]).remove(0);
    let reverse_plan = crate::contract::prepare_tensorcontract_fusion_plan_dyn_raw_with_axis_order_and_orientation(
        provider.as_ref(),
        swapped_dst_bound.space(),
        &canonical,
        &rhs,
        TensorContractSpec::new_with_conjugation(
            &[1],
            &[0],
            crate::OutputAxisOrder::from_axes(&[1, 0]),
            true,
            false,
        ),
        &candidate,
        crate::contract::FusionContractOrientation::RhsLhs,
    )
    .unwrap();
    let mut padded_artifact_output = vec![0.0; swapped_dst_bound.space().required_len().unwrap()];
    crate::contract::execute_prelowered_dynamic_tree_execution_artifact_for_test(
        engine,
        provider.as_ref(),
        &reverse_plan,
        swapped_dst_bound.space(),
        &mut padded_artifact_output,
        padded_operand,
        &padded_data,
        rhs_operand,
        &rhs_data,
        1.0,
        0.0,
    )
    .unwrap();
    // What: reverse artifact execution reads the six-element padded physical
    // adjoint storage while contracting the distinct four-element logical layout.
    assert_eq!(padded_data.len(), 6);
    assert_eq!(logical_lhs.required_len().unwrap(), 4);
    // What: requested output order [1, 0] writes the hand-computed transpose
    // of the identity-order column-major oracle below.
    assert_eq!(padded_artifact_output, [0.0, 6.5, 2.0, 13.5]);

    let mut execute_prelowered =
        |context: &mut TensorContractFusionExecutionContext<f64, RuleIdentity>,
         lhs: crate::FusionOperand<'_>,
         lhs_data: &[f64]| {
            let mut output = vec![0.0; dst.required_len().unwrap()];
            context
                .tensorcontract_prelowered_into(
                    engine,
                    &dst_bound,
                    &mut output,
                    lhs,
                    lhs_data,
                    rhs_operand,
                    &rhs_data,
                    prelowered_axes(),
                    1.0,
                    crate::ContractDestinationInit::Axpby(0.0),
                )
                .unwrap();
            output
        };

    for (first, first_data, second, second_data) in [
        (
            canonical_operand,
            canonical_data.as_slice(),
            padded_operand,
            padded_data.as_slice(),
        ),
        (
            padded_operand,
            padded_data.as_slice(),
            canonical_operand,
            canonical_data.as_slice(),
        ),
    ] {
        symtn_core::clear_structure_caches();
        let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
        let first_output = execute_prelowered(&mut context, first, first_data);
        assert_eq!(first_output, oracle);
        let second_output = execute_prelowered(&mut context, second, second_data);
        // What: equal logical spaces accept distinct admissible physical
        // layouts in either call order.
        assert_eq!(second_output, oracle);

        let repeated = execute_prelowered(&mut context, second, second_data);
        // What: an identical prelowered call remains deterministic.
        assert_eq!(repeated, oracle);

        symtn_core::clear_structure_caches();
        let mut publisher = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
        assert_eq!(
            execute_prelowered(&mut publisher, first, first_data),
            oracle
        );
        let mut consumer = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
        // What: execution is independent of state built by another context.
        assert_eq!(
            execute_prelowered(&mut consumer, second, second_data),
            oracle
        );
        assert_eq!(
            execute_prelowered(&mut consumer, second, second_data),
            oracle
        );
    }

    symtn_core::clear_structure_caches();
    let mut namespace_context =
        TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
    let mut ordinary = vec![0.0; dst.required_len().unwrap()];
    namespace_context
        .tensorcontract_prelowered_into(
            engine,
            &dst_bound,
            &mut ordinary,
            crate::FusionOperand::direct(logical_lhs_bound.space()),
            &eager_lhs_data,
            crate::FusionOperand::direct(rhs_bound.space()),
            &rhs_data,
            ordinary_axes(),
            1.0,
            crate::ContractDestinationInit::Axpby(0.0),
        )
        .unwrap();
    assert_eq!(ordinary, oracle);

    let ordinary_tensor = TestTensor::<f64>::from_vec_with_fusion_space(
        eager_lhs_data,
        canonical_typed
            .validate_rule(&rule)
            .and_then(|()| canonical_typed.adjoint_view())
            .unwrap(),
    )
    .unwrap();
    // What: the ordinary tensor constructor aliases its physical structure to
    // its fusion subblock structure, so only the prelowered seam needs a
    // separate storage determinant.
    assert!(Arc::ptr_eq(
        ordinary_tensor.structure(),
        ordinary_tensor.fusion_space().unwrap().structure(),
    ));
}

#[test]
fn tensorcontract_fusion_prelowered_uniform_fermion_twist_takes_the_scaled_core_and_matches_eager()
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    use num_complex::Complex64;

    let rule = FermionParityFusionRule;
    let odd = SectorId::new(1);
    let make_space = |codomain_dual: bool, domain_dual: bool| {
        let hom = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(odd, 1)], codomain_dual)]),
            FusionProductSpace::new([SectorLeg::new([(odd, 1)], domain_dual)]),
        );
        crate::DynamicFusionMapSpace::from_final_homspace(&rule, hom).unwrap()
    };
    let lhs_space = make_space(true, false);
    // The externally dual RHS codomain requires the fermionic supertrace twist.
    let rhs_space = make_space(true, false);
    let lhs_data = vec![Complex64::new(2.0, 3.0)];
    let rhs_data = vec![Complex64::new(5.0, -1.0)];
    let (adj_space, adj_data) = crate::adjoint::adjoint_dyn(&rule, &lhs_space, &lhs_data).unwrap();
    let dst_space =
        crate::DynamicFusionMapSpace::contracted(&rule, &adj_space, &rhs_space, &[1], &[0])
            .unwrap();
    let provider = Arc::new(rule);
    let dst_bound = crate::BoundDynamicFusionMapSpace::bind_multiplicity_free(
        dst_space.clone(),
        Arc::clone(&provider),
    )
    .unwrap();
    let adj_bound =
        crate::BoundDynamicFusionMapSpace::bind_multiplicity_free(adj_space, Arc::clone(&provider))
            .unwrap();
    let lhs_bound =
        crate::BoundDynamicFusionMapSpace::bind_multiplicity_free(lhs_space, Arc::clone(&provider))
            .unwrap();
    let rhs_bound =
        crate::BoundDynamicFusionMapSpace::bind_multiplicity_free(rhs_space, Arc::clone(&provider))
            .unwrap();
    let mut context = crate::TensorContractFusionExecutionContext::<Complex64, _>::default();
    let mut eager = vec![Complex64::new(0.0, 0.0); dst_space.required_len().unwrap()];
    context
        .tensorcontract_prelowered_into(
            engine,
            &dst_bound,
            &mut eager,
            crate::FusionOperand::direct(adj_bound.space()),
            &adj_data,
            crate::FusionOperand::direct(rhs_bound.space()),
            &rhs_data,
            TensorContractSpec::with_default_output_order(&[1], &[0]),
            Complex64::new(1.0, 0.0),
            crate::ContractDestinationInit::Axpby(Complex64::new(0.0, 0.0)),
        )
        .unwrap();

    let mut lazy = vec![Complex64::new(0.0, 0.0); dst_space.required_len().unwrap()];
    context
        .tensorcontract_prelowered_into(
            engine,
            &dst_bound,
            &mut lazy,
            crate::FusionOperand::adjoint(lhs_bound.space()),
            &lhs_data,
            crate::FusionOperand::direct(rhs_bound.space()),
            &rhs_data,
            TensorContractSpec::new_with_conjugation(
                &[1],
                &[0],
                crate::OutputAxisOrder::identity(),
                true,
                false,
            ),
            Complex64::new(1.0, 0.0),
            crate::ContractDestinationInit::Axpby(Complex64::new(0.0, 0.0)),
        )
        .unwrap();

    assert_eq!(lazy, eager);

    let mut no_cache_context =
        crate::TensorContractFusionExecutionContext::<Complex64, _>::default();
    let mut no_cache_lazy = vec![Complex64::new(0.0, 0.0); dst_space.required_len().unwrap()];
    no_cache_context
        .tensorcontract_prelowered_into(
            engine,
            &dst_bound,
            &mut no_cache_lazy,
            crate::FusionOperand::adjoint(lhs_bound.space()),
            &lhs_data,
            crate::FusionOperand::direct(rhs_bound.space()),
            &rhs_data,
            TensorContractSpec::new_with_conjugation(
                &[1],
                &[0],
                crate::OutputAxisOrder::identity(),
                true,
                false,
            ),
            Complex64::new(1.0, 0.0),
            crate::ContractDestinationInit::Axpby(Complex64::new(0.0, 0.0)),
        )
        .unwrap();
    assert_complex64_bits_eq(
        "cached vs NoCache complex fermion twist",
        &no_cache_lazy,
        &lazy,
    );
    // What: a twist uniform per RHS coupled sector rides the core as per-job
    // GEMM alpha (#1858), where TensorKit copies an operand to twist it.
    assert!(context.last_resolution_is_core());
}

#[test]
fn nested_product_lowered_dynamic_execution_matches_independent_encoded_oracles() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: direct and lazy-adjoint contractions for a nested non-Abelian
    // product keep the encoded layout/data semantics with and without replay
    // caching, including source transforms and a nonidentity output transform.
    const ISOLATED_ENV: &str = "SYMTN_LOWERED_DYNAMIC_ORACLE_CHILD";
    if std::env::var_os(ISOLATED_ENV).is_none() {
        // What: cache resets used to make the two oracles independent cannot
        // change process-global cache generations observed by sibling tests.
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "tests::contract_fusion::prelowered::nested_product_lowered_dynamic_execution_matches_independent_encoded_oracles",
                "--nocapture",
            ])
            .env(ISOLATED_ENV, "1")
            .status()
            .unwrap();
        assert!(status.success());
        return;
    }
    use symtn_core::{
        FermionParityFusionRule, Fz2SectorLayout, PackedProductCodec, ProductFusionRule,
        ProductSectorCodec, ProductSectorLayout, SU2FusionRule, SU2Irrep, Su2SectorLayout,
        U1FusionRule, U1Irrep, U1SectorLayout, Z2Irrep,
    };

    type Fz2U1Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
    type Fz2U1Layout = ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>;
    type TripleCodec = PackedProductCodec<Fz2U1Layout, Su2SectorLayout>;
    type Fz2U1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule, Fz2U1Codec>;
    type TripleRule = ProductFusionRule<Fz2U1Rule, SU2FusionRule, TripleCodec>;
    type TripleRuleKey = <TripleRule as TreeTransformRuleCacheKey>::Key;

    let rule = TripleRule::try_new(
        Fz2U1Rule::try_new(FermionParityFusionRule, U1FusionRule).unwrap(),
        SU2FusionRule,
    )
    .unwrap();
    let sector = |parity: u8, charge: i32, twice_spin: usize| {
        TripleCodec::encode(
            Fz2U1Codec::encode(
                Z2Irrep::new(parity).sector_id(),
                U1Irrep::new(charge).sector_id(),
            ),
            SU2Irrep::from_twice_spin(twice_spin).sector_id(),
        )
    };
    let vacuum = sector(0, 0, 0);
    let charged = sector(1, 0, 1);
    let leg = |dual| SectorLeg::new([(vacuum, 1), (charged, 1)], dual);
    let provider = Arc::new(rule);
    let bind_encoded = |homspace: FusionTreeHomSpace| {
        BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
            Arc::clone(&provider),
            homspace,
        )
        .unwrap()
    };
    let bind_lowered = |homspace: FusionTreeHomSpace| {
        BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(
            Arc::clone(&provider),
            homspace,
        )
        .unwrap()
    };
    let lhs_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(false), leg(true)]),
        FusionProductSpace::new([leg(false)]),
    );
    let rhs_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(true)]),
        FusionProductSpace::new([leg(true), leg(false)]),
    );
    let encoded_lhs = bind_encoded(lhs_hom.clone());
    let encoded_rhs = bind_encoded(rhs_hom.clone());
    let lowered_lhs = bind_lowered(lhs_hom);
    let lowered_rhs = bind_lowered(rhs_hom);
    assert_eq!(encoded_lhs.space(), lowered_lhs.space());
    assert_eq!(encoded_rhs.space(), lowered_rhs.space());
    let lhs_data = (0..encoded_lhs.space().required_len().unwrap())
        .map(|index| index as f64 + 1.0)
        .collect::<Vec<_>>();
    let rhs_data = (0..encoded_rhs.space().required_len().unwrap())
        .map(|index| 0.5 * index as f64 - 2.0)
        .collect::<Vec<_>>();
    let direct_axes =
        TensorContractSpec::new(&[0], &[2], OutputAxisOrder::from_axes(&[2, 0, 3, 1]));

    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();
    let encoded_dst = BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
        &encoded_lhs,
        &encoded_rhs,
        direct_axes.lhs_contracting_axes(),
        direct_axes.rhs_contracting_axes(),
        direct_axes.output_permutation(),
    )
    .unwrap();
    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();
    let lowered_dst = BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
        &lowered_lhs,
        &lowered_rhs,
        direct_axes.lhs_contracting_axes(),
        direct_axes.rhs_contracting_axes(),
        direct_axes.output_permutation(),
    )
    .unwrap();
    assert_eq!(encoded_dst.space(), lowered_dst.space());

    for _ in 0..2 {
        symtn_core::clear_structure_caches();
        symtn_core::clear_structure_caches();
        let mut encoded = vec![0.0; encoded_dst.space().required_len().unwrap()];
        let mut encoded_context =
            TensorContractFusionExecutionContext::<f64, TripleRuleKey>::default();
        encoded_context
            .tensorcontract_prelowered_into(
                engine,
                &encoded_dst,
                &mut encoded,
                crate::FusionOperand::direct(encoded_lhs.space()),
                &lhs_data,
                crate::FusionOperand::direct(encoded_rhs.space()),
                &rhs_data,
                direct_axes,
                1.0,
                crate::ContractDestinationInit::Axpby(0.0),
            )
            .unwrap();

        symtn_core::clear_structure_caches();
        symtn_core::clear_structure_caches();
        let mut lowered = vec![0.0; lowered_dst.space().required_len().unwrap()];
        let mut lowered_context =
            TensorContractFusionExecutionContext::<f64, TripleRuleKey>::default();
        lowered_context
            .tensorcontract_prelowered_into(
                engine,
                &lowered_dst,
                &mut lowered,
                crate::FusionOperand::direct(lowered_lhs.space()),
                &lhs_data,
                crate::FusionOperand::direct(lowered_rhs.space()),
                &rhs_data,
                direct_axes,
                1.0,
                crate::ContractDestinationInit::Axpby(0.0),
            )
            .unwrap();
        // The encoded layout is the independent oracle for the lowered one;
        // SU2 recoupling makes the sums order-dependent, so the comparison
        // uses the tolerance rule (terms bounded by len(lhs) * len(rhs), which covers the recoupled
        // trees as well as the contracted length).
        numerics::assert_slices_close(
            "lowered vs encoded",
            &lowered,
            &encoded,
            lhs_data.len() * rhs_data.len(),
        );
        let mut warm = vec![0.0; lowered.len()];
        lowered_context
            .tensorcontract_prelowered_into(
                engine,
                &lowered_dst,
                &mut warm,
                crate::FusionOperand::direct(lowered_lhs.space()),
                &lhs_data,
                crate::FusionOperand::direct(lowered_rhs.space()),
                &rhs_data,
                direct_axes,
                1.0,
                crate::ContractDestinationInit::Axpby(0.0),
            )
            .unwrap();
        assert_eq!(warm, lowered);
    }

    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();
    let (eager_lhs, eager_lhs_data) = crate::adjoint_bound_dyn(&encoded_lhs, &lhs_data).unwrap();
    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();
    let lazy_lhs = crate::adjoint_bound_space_dyn(&lowered_lhs).unwrap();
    assert_eq!(eager_lhs.space(), lazy_lhs.space());
    let lazy_axes = TensorContractSpec::new_with_conjugation(
        &[1],
        &[1],
        OutputAxisOrder::from_axes(&[2, 0, 3, 1]),
        true,
        false,
    );
    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();
    let encoded_lazy_dst = BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
        &eager_lhs,
        &encoded_rhs,
        lazy_axes.lhs_contracting_axes(),
        lazy_axes.rhs_contracting_axes(),
        lazy_axes.output_permutation(),
    )
    .unwrap();
    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();
    let lazy_dst = BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
        &lazy_lhs,
        &lowered_rhs,
        lazy_axes.lhs_contracting_axes(),
        lazy_axes.rhs_contracting_axes(),
        lazy_axes.output_permutation(),
    )
    .unwrap();
    assert_eq!(encoded_lazy_dst.space(), lazy_dst.space());

    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();
    let mut eager = vec![0.0; lazy_dst.space().required_len().unwrap()];
    let mut eager_context = TensorContractFusionExecutionContext::<f64, TripleRuleKey>::default();
    eager_context
        .tensorcontract_prelowered_into(
            engine,
            &lazy_dst,
            &mut eager,
            crate::FusionOperand::direct(eager_lhs.space()),
            &eager_lhs_data,
            crate::FusionOperand::direct(encoded_rhs.space()),
            &rhs_data,
            TensorContractSpec::new(
                lazy_axes.lhs_contracting_axes(),
                lazy_axes.rhs_contracting_axes(),
                lazy_axes.output_permutation(),
            ),
            1.0,
            crate::ContractDestinationInit::Axpby(0.0),
        )
        .unwrap();

    for _ in 0..2 {
        symtn_core::clear_structure_caches();
        symtn_core::clear_structure_caches();
        let mut encoded_lazy = vec![0.0; encoded_lazy_dst.space().required_len().unwrap()];
        let mut encoded_lazy_context =
            TensorContractFusionExecutionContext::<f64, TripleRuleKey>::default();
        encoded_lazy_context
            .tensorcontract_prelowered_into(
                engine,
                &encoded_lazy_dst,
                &mut encoded_lazy,
                FusionOperand::adjoint(encoded_lhs.space()),
                &lhs_data,
                FusionOperand::direct(encoded_rhs.space()),
                &rhs_data,
                lazy_axes,
                1.0,
                crate::ContractDestinationInit::Axpby(0.0),
            )
            .unwrap();
        numerics::assert_slices_close(
            "encoded lazy adjoint vs eager adjoint oracle",
            &encoded_lazy,
            &eager,
            lhs_data.len() * rhs_data.len(),
        );

        symtn_core::clear_structure_caches();
        symtn_core::clear_structure_caches();
        let mut lazy = vec![0.0; lazy_dst.space().required_len().unwrap()];
        let mut lazy_context =
            TensorContractFusionExecutionContext::<f64, TripleRuleKey>::default();
        let mut execute_lazy =
            |context: &mut TensorContractFusionExecutionContext<f64, TripleRuleKey>,
             output: &mut [f64]| {
                context.tensorcontract_prelowered_into(
                    engine,
                    &lazy_dst,
                    output,
                    FusionOperand::adjoint(lowered_lhs.space()),
                    &lhs_data,
                    FusionOperand::direct(lowered_rhs.space()),
                    &rhs_data,
                    lazy_axes,
                    1.0,
                    crate::ContractDestinationInit::Axpby(0.0),
                )
            };
        execute_lazy(&mut lazy_context, &mut lazy).unwrap();
        numerics::assert_slices_close(
            "lowered lazy adjoint vs eager adjoint oracle",
            &lazy,
            &eager,
            lhs_data.len() * rhs_data.len(),
        );
        let mut warm = vec![0.0; lazy.len()];
        execute_lazy(&mut lazy_context, &mut warm).unwrap();
        assert_eq!(warm, lazy);
    }
}

#[test]
fn storage_direct_contraction_refuses_mis_stacked_trees_before_gemm() {
    // What (#1517): an operand whose coupled-sector columns are stacked in a
    // different tree order than the partner's rows has no positional GEMM;
    // the storage-direct seam must refuse it, not multiply mismatched trees.
    struct NoGemm;

    impl symtn_operations::fusion_replay::StorageGemm<f64, Vec<f64>, Vec<f64>, Vec<f64>> for NoGemm {
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
            panic!("mis-stacked storage must be refused before GEMM")
        }
    }

    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = Z2FusionRule;
    let leg = || SectorLeg::new([(SectorId::new(0), 1), (SectorId::new(1), 1)], false);
    let homspace = || {
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg()]),
            FusionProductSpace::new([leg(), leg()]),
        )
    };
    let keys = homspace().fusion_tree_keys(&rule);
    let canonical = DynamicFusionMapSpace::from_final_homspace(&rule, homspace()).unwrap();
    let mut blocks = keys
        .iter()
        .map(|key| (key.clone(), vec![1; 4]))
        .collect::<Vec<_>>();
    blocks.sort_by(|(a, _), (b, _)| {
        a.codomain_tree()
            .cmp(b.codomain_tree())
            .then(b.domain_tree().cmp(a.domain_tree()))
    });
    let mis_stacked = DynamicFusionMapSpace::try_from_layout(
        homspace(),
        BlockStructure::coupled_sector_matrix_with_keys(&rule, 2, 4, blocks).unwrap(),
    )
    .unwrap()
    .try_bind_rule(&rule)
    .unwrap();
    let provider = Arc::new(rule);
    let bind = |space: &DynamicFusionMapSpace| {
        BoundDynamicFusionMapSpace::bind_multiplicity_free((*space).clone(), Arc::clone(&provider))
            .unwrap()
    };
    let (canonical, mis_stacked) = (bind(&canonical), bind(&mis_stacked));
    let len = canonical.space().required_len().unwrap();
    let values = (0..len).map(|index| index as f64 + 1.0).collect::<Vec<_>>();
    symtn_core::clear_structure_caches();
    for (lhs, rhs) in [(&mis_stacked, &canonical), (&canonical, &mis_stacked)] {
        let mut dst = vec![0.0; len];
        let result = crate::contract::tensorcontract_fusion_dyn_prelowered_direct_on_storage(
            &mut NoGemm,
            &canonical,
            &mut dst,
            crate::FusionOperand::direct(lhs.space()),
            &values,
            crate::FusionOperand::direct(rhs.space()),
            &values,
            TensorContractSpec::new(&[2, 3], &[0, 1], crate::OutputAxisOrder::identity()),
        );
        assert!(
            matches!(
                result,
                Err(OperationError::UnsupportedTensorContractScope { .. })
            ),
            "{result:?}"
        );
    }
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_generic_compose_pairs_mis_stacked_multiplicity_trees_by_identity() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What (#1520): checked Generic compose on SU(3), where `8 ⊗ 8 → 8` has
    // vertex multiplicity 2, pairs coupled-sector rows and columns by tree
    // identity (vertex label included) when operands stack those trees in
    // different orders, through the checked irregular core of `plan_compose`.
    use std::collections::BTreeMap;
    use symtn_core::CheckedGenericFusion;
    use symtn_sectors::LieGroupFusionRule;

    type Entries = BTreeMap<(BlockKey, Vec<usize>), f64>;
    fn position(block: &symtn_core::BlockRef<'_>, coordinates: &[usize]) -> usize {
        block.offset()
            + coordinates
                .iter()
                .zip(block.strides())
                .map(|(&c, &s)| c * s)
                .sum::<usize>()
    }
    fn coordinates(shape: &[usize]) -> Vec<Vec<usize>> {
        (0..shape.iter().product::<usize>())
            .map(|mut linear| {
                shape
                    .iter()
                    .map(|&extent| {
                        let coordinate = linear % extent;
                        linear /= extent;
                        coordinate
                    })
                    .collect()
            })
            .collect()
    }
    fn entries(structure: &BlockStructure, data: &[f64]) -> Entries {
        let mut entries = Entries::new();
        for index in 0..structure.block_count() {
            let block = structure.block(index).unwrap();
            for coordinates in coordinates(block.shape()) {
                let value = data[position(&block, &coordinates)];
                entries.insert((block.key().clone(), coordinates), value);
            }
        }
        entries
    }

    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let adjoint = provider.encode_irrep(&[1, 1]).unwrap();
    assert_eq!(provider.try_nsymbol(adjoint, adjoint, adjoint).unwrap(), 2);
    let leg = || SectorLeg::new([(adjoint, 2)], false);
    let canonical = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&provider),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg()]),
            FusionProductSpace::new([leg(), leg()]),
        ),
    )
    .unwrap();
    let structure = canonical.space().structure();
    let len = canonical.space().required_len().unwrap();
    let data = (0..len)
        .map(|index| ((7 * index) % 11) as f64 - 4.5 + 0.01 * index as f64)
        .collect::<Vec<_>>();
    let good = entries(structure, &data);

    // Coupled-sector matrix layout with the row (codomain) and column
    // (domain) trees of every sector stacked in first-occurrence order,
    // optionally reversed.
    let restack = |reverse_rows: bool, reverse_columns: bool| {
        let mut sectors: Vec<(SectorId, Vec<usize>)> = Vec::new();
        for index in 0..structure.block_count() {
            let BlockKey::FusionTree(pair) = structure.block(index).unwrap().key().clone() else {
                unreachable!()
            };
            let coupled = pair.codomain_tree().coupled();
            match sectors.iter_mut().find(|(sector, _)| *sector == coupled) {
                Some((_, indices)) => indices.push(index),
                None => sectors.push((coupled, vec![index])),
            }
        }
        let mut specs = Vec::new();
        for (_, indices) in sectors {
            let blocks = indices
                .iter()
                .map(|&index| structure.block(index).unwrap())
                .collect::<Vec<_>>();
            let base = blocks.iter().map(|block| block.offset()).min().unwrap();
            let axis_trees = |codomain: bool, reverse: bool| {
                let mut trees: Vec<(FusionTreeKey, usize)> = Vec::new();
                for block in &blocks {
                    let BlockKey::FusionTree(pair) = block.key() else {
                        unreachable!()
                    };
                    let (tree, extents) = if codomain {
                        (pair.codomain_tree(), &block.shape()[..2])
                    } else {
                        (pair.domain_tree(), &block.shape()[2..])
                    };
                    if !trees.iter().any(|(known, _)| known == tree) {
                        trees.push((tree.clone(), extents.iter().product()));
                    }
                }
                if reverse {
                    trees.reverse();
                }
                trees
            };
            let (rows, columns) = (
                axis_trees(true, reverse_rows),
                axis_trees(false, reverse_columns),
            );
            let offset = |trees: &[(FusionTreeKey, usize)], tree: &FusionTreeKey| {
                trees
                    .iter()
                    .take_while(|(known, _)| known != tree)
                    .map(|(_, extent)| extent)
                    .sum::<usize>()
            };
            let height = rows.iter().map(|(_, extent)| extent).sum::<usize>();
            let rank_of = |trees: &[(FusionTreeKey, usize)], tree: &FusionTreeKey| {
                trees.iter().position(|(known, _)| known == tree).unwrap()
            };
            // Blocks are listed row by row in the new stacking, so each
            // operand's first-occurrence tree order is its storage order.
            let mut sector_specs = blocks
                .iter()
                .map(|block| {
                    let BlockKey::FusionTree(pair) = block.key() else {
                        unreachable!()
                    };
                    let shape = block.shape().to_vec();
                    let spec = BlockSpec::with_key(
                        block.key().clone(),
                        shape.clone(),
                        vec![1, shape[0], height, height * shape[2]],
                        base + offset(&rows, pair.codomain_tree())
                            + height * offset(&columns, pair.domain_tree()),
                    )
                    .unwrap();
                    (
                        (
                            rank_of(&rows, pair.codomain_tree()),
                            rank_of(&columns, pair.domain_tree()),
                        ),
                        spec,
                    )
                })
                .collect::<Vec<_>>();
            sector_specs.sort_by_key(|(position, _)| *position);
            specs.extend(sector_specs.into_iter().map(|(_, spec)| spec));
        }
        let restacked =
            canonical.with_test_structure(BlockStructure::from_blocks_with_rank(4, specs).unwrap());
        let mut restacked_data = vec![0.0; len];
        let layout = restacked.space().structure();
        for index in 0..layout.block_count() {
            let block = layout.block(index).unwrap();
            for coordinates in coordinates(block.shape()) {
                let at = position(&block, &coordinates);
                restacked_data[at] = good[&(block.key().clone(), coordinates)];
            }
        }
        assert_eq!(entries(layout, &restacked_data), good);
        (restacked, restacked_data)
    };

    // The layout model reproduces the canonical layout exactly.
    let (unchanged, unchanged_data) = restack(false, false);
    assert_eq!(unchanged_data, data);
    let layout = |structure: &BlockStructure| {
        (0..structure.block_count())
            .map(|index| {
                let block = structure.block(index).unwrap();
                (
                    block.key().clone(),
                    (block.offset(), block.strides().to_vec()),
                )
            })
            .collect::<BTreeMap<_, _>>()
    };
    assert_eq!(layout(unchanged.space().structure()), layout(structure));

    let tilings = [
        ("Canonical", (canonical.clone(), data.clone())),
        ("RowsReversed", restack(true, false)),
        ("ColumnsReversed", restack(false, true)),
        ("BothReversed", restack(true, true)),
    ];
    for (name, (_, tiling_data)) in &tilings[1..] {
        assert_ne!(
            tiling_data, &data,
            "{name} must restack a multi-tree sector"
        );
    }

    // Independent oracle: `(A B)[X, Z] = Σ_Y A[X, Y] B[Y, Z]` over tree
    // identity; composition crosses no leg, so no recoupling enters.
    let mut square = Entries::new();
    for ((lhs_key, lhs_index), lhs_value) in &good {
        let BlockKey::FusionTree(lhs_tree) = lhs_key else {
            unreachable!()
        };
        for ((rhs_key, rhs_index), rhs_value) in &good {
            let BlockKey::FusionTree(rhs_tree) = rhs_key else {
                unreachable!()
            };
            if lhs_tree.domain_tree() == rhs_tree.codomain_tree()
                && lhs_index[2..] == rhs_index[..2]
            {
                let key = BlockKey::FusionTree(FusionTreePairKey::pair(
                    lhs_tree.codomain_tree().clone(),
                    rhs_tree.domain_tree().clone(),
                ));
                let index = [&lhs_index[..2], &rhs_index[2..]].concat();
                *square.entry((key, index)).or_insert(0.0) += lhs_value * rhs_value;
            }
        }
    }

    let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
    for (lhs_name, (lhs_space, lhs_data)) in &tilings {
        for (rhs_name, (rhs_space, rhs_data)) in &tilings {
            let (space, product) = context
                .tensorcompose_checked_generic_in(
                    engine,
                    direct_side(lhs_space, lhs_data),
                    direct_side(rhs_space, rhs_data),
                )
                .unwrap();
            let actual = entries(space.space().structure(), &product);
            assert_eq!(
                actual.keys().collect::<Vec<_>>(),
                square.keys().collect::<Vec<_>>(),
                "{lhs_name}·{rhs_name}: keys"
            );
            for ((key, value), expected) in actual.iter().zip(square.values()) {
                assert!(
                    (value - expected).abs() <= 1e-10 * expected.abs().max(1.0),
                    "{lhs_name}·{rhs_name}: {key:?}: {value} != {expected}"
                );
            }
        }
    }

    // What (#1865): the lazy adjoint of each tiling composes as its
    // materialization does. Its logical space is canonical while the
    // parent's storage order is caller-chosen, so a restacked parent
    // reaches the checked irregular core, read with `MatrixOp::Adjoint`.
    // Oracle (TensorKit `subblock(t', (f₁, f₂)) = permutedims(conj(subblock(t,
    // (f₂, f₁))))`, real here): `(A'B)[X, Z] = Σ_Y A[Y, X] B[Y, Z]`.
    let mut adjoint_square = Entries::new();
    for ((lhs_key, lhs_index), lhs_value) in &good {
        let BlockKey::FusionTree(lhs_tree) = lhs_key else {
            unreachable!()
        };
        for ((rhs_key, rhs_index), rhs_value) in &good {
            let BlockKey::FusionTree(rhs_tree) = rhs_key else {
                unreachable!()
            };
            if lhs_tree.codomain_tree() == rhs_tree.codomain_tree()
                && lhs_index[..2] == rhs_index[..2]
            {
                let key = BlockKey::FusionTree(FusionTreePairKey::pair(
                    lhs_tree.domain_tree().clone(),
                    rhs_tree.domain_tree().clone(),
                ));
                let index = [&lhs_index[2..], &rhs_index[2..]].concat();
                *adjoint_square.entry((key, index)).or_insert(0.0) += lhs_value * rhs_value;
            }
        }
    }
    for (lhs_name, (lhs_space, lhs_data)) in &tilings {
        let logical = crate::adjoint_bound_space_dyn_generic_checked(lhs_space).unwrap();
        for (rhs_name, (rhs_space, rhs_data)) in &tilings {
            let (space, product) = context
                .tensorcompose_checked_generic_in(
                    engine,
                    (
                        &logical,
                        crate::FusionOperand::adjoint(lhs_space.space()),
                        lhs_data,
                    ),
                    direct_side(rhs_space, rhs_data),
                )
                .unwrap();
            let actual = entries(space.space().structure(), &product);
            assert_eq!(
                actual.keys().collect::<Vec<_>>(),
                adjoint_square.keys().collect::<Vec<_>>(),
                "{lhs_name}'·{rhs_name}: keys"
            );
            for ((key, value), expected) in actual.iter().zip(adjoint_square.values()) {
                assert!(
                    (value - expected).abs() <= 1e-10 * expected.abs().max(1.0),
                    "{lhs_name}'·{rhs_name}: {key:?}: {value} != {expected}"
                );
            }
        }
    }
}
