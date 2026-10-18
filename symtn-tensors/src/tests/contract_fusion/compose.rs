use super::*;

pub(super) fn z2_matrix_homspace() -> FusionTreeHomSpace {
    let leg = || SectorLeg::new([(SectorId::new(0), 2), (SectorId::new(1), 2)], false);
    FusionTreeHomSpace::new(
        FusionProductSpace::new([leg()]),
        FusionProductSpace::new([leg()]),
    )
}

pub(super) fn z2_matrix_space_with_homspace(homspace: FusionTreeHomSpace) -> DynamicFusionMapSpace {
    DynamicFusionMapSpace::from_final_homspace(&Z2FusionRule, homspace).unwrap()
}

pub(super) fn fermion_parity_matrix_homspace() -> FusionTreeHomSpace {
    let leg = || SectorLeg::new([(SectorId::new(0), 2), (SectorId::new(1), 2)], false);
    FusionTreeHomSpace::new(
        FusionProductSpace::new([leg()]),
        FusionProductSpace::new([leg()]),
    )
}

pub(super) fn fermion_parity_matrix_space_with_homspace(
    homspace: FusionTreeHomSpace,
) -> DynamicFusionMapSpace {
    DynamicFusionMapSpace::from_final_homspace(&FermionParityFusionRule, homspace).unwrap()
}

/// The device eager compose's route on Host storage: [`crate::plan_compose`]
/// for a direct-core executor, then the core's storage replay into a
/// zero-filled destination.
pub(super) fn compose_direct_on_storage<R, G>(
    gemm: &mut G,
    dst_space: &crate::BoundDynamicFusionMapSpace<R>,
    dst: &mut Vec<f64>,
    lhs: crate::FusionOperand<'_>,
    lhs_storage: &Vec<f64>,
    rhs: crate::FusionOperand<'_>,
    rhs_storage: &Vec<f64>,
) -> Result<(), OperationError>
where
    R: symtn_core::MultiplicityFreeRigidSymbols<Scalar = f64>,
    G: symtn_operations::fusion_replay::StorageGemm<f64, Vec<f64>, Vec<f64>, Vec<f64>>,
{
    let resolution = crate::plan_compose::<crate::DirectCoreExecutor, _>(dst_space, lhs, rhs)?;
    let (plan, swapped) = resolution.direct_core().expect("compose plans the core");
    assert!(!swapped);
    plan.execute_direct_on_storage_prezeroed(gemm, dst, lhs_storage, rhs_storage)
}

#[test]
fn bosonic_tensorcompose_matches_ordinary_contract() {
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // What: coefficient-free bosonic composition and ordinary contraction
    // share one compiled Core resolution instead of retaining parallel plans.
    let layout = z2_matrix_space_with_homspace(z2_matrix_homspace());
    let source = layout;
    let dst = crate::DynamicFusionMapSpace::contracted(&Z2FusionRule, &source, &source, &[1], &[0])
        .unwrap();
    let provider = Arc::new(Z2FusionRule);
    let source_bound = crate::BoundDynamicFusionMapSpace::bind_multiplicity_free(
        source.clone(),
        Arc::clone(&provider),
    )
    .unwrap();
    let dst_bound = crate::BoundDynamicFusionMapSpace::bind_multiplicity_free(
        dst.clone(),
        Arc::clone(&provider),
    )
    .unwrap();
    let lhs = vec![1.0; source.required_len().unwrap()];
    let rhs = vec![2.0; source.required_len().unwrap()];
    let mut context = crate::TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
    let mut ordinary = vec![0.0; dst.required_len().unwrap()];
    context
        .tensorcontract_fusion_dyn_into(
            &dst_bound,
            &mut ordinary,
            &source_bound,
            &lhs,
            &source_bound,
            &rhs,
            TensorContractSpec::new(&[1], &[0], crate::OutputAxisOrder::identity()),
            1.0,
            0.0,
        )
        .unwrap();
    assert!(context.last_resolution_is_core());

    let mut composed = vec![0.0; dst.required_len().unwrap()];
    context
        .tensorcompose_planned_into(
            &dst_bound,
            &mut composed,
            (crate::FusionOperand::direct(&source), &lhs),
            (crate::FusionOperand::direct(&source), &rhs),
            1.0,
            crate::ContractDestinationInit::Axpby(0.0),
        )
        .unwrap();

    assert_eq!(composed, ordinary);
    assert!(context.last_resolution_is_core());
}

#[test]
fn bosonic_tensorcompose_with_an_adjoint_operand_matches_the_materialized_adjoint() {
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // What: bosonic composition with a *conjugated* operand — the one arm of
    // the composition seam no public caller in this workspace reaches because
    // the typed facade passes direct operands. It is also the arm whose
    // inner call the non-lowered seam changed, so this is where the claim
    // "the two entry points compute the same thing" is actually pinned.
    //
    // Oracle: the lazy adjoint operand must produce exactly what the eagerly
    // materialized adjoint produces as a direct operand. The payload is
    // element-distinct, so the per-block transpose the adjoint performs is
    // visible in every element of the result.
    let layout = z2_matrix_space_with_homspace(z2_matrix_homspace());
    let source = layout;
    let provider = Arc::new(Z2FusionRule);
    let source_bound = crate::BoundDynamicFusionMapSpace::bind_multiplicity_free(
        source.clone(),
        Arc::clone(&provider),
    )
    .unwrap();
    let lhs: Vec<f64> = (0..source.required_len().unwrap())
        .map(|index| index as f64 + 1.0)
        .collect();
    let rhs: Vec<f64> = lhs.iter().map(|value| value * 0.5 + 3.0).collect();

    let (adjoint_bound, adjoint_data) = crate::adjoint_bound_dyn(&source_bound, &lhs).unwrap();
    let dst = crate::DynamicFusionMapSpace::contracted(
        &Z2FusionRule,
        adjoint_bound.space(),
        &source,
        &[1],
        &[0],
    )
    .unwrap();
    let dst_bound =
        crate::BoundDynamicFusionMapSpace::bind_multiplicity_free(dst.clone(), provider).unwrap();
    let mut context = crate::TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();

    let mut reference = vec![0.0; dst.required_len().unwrap()];
    context
        .tensorcompose_planned_into(
            &dst_bound,
            &mut reference,
            (
                crate::FusionOperand::direct(adjoint_bound.space()),
                &adjoint_data,
            ),
            (crate::FusionOperand::direct(&source), &rhs),
            1.0,
            crate::ContractDestinationInit::Axpby(0.0),
        )
        .unwrap();

    let mut lazy = vec![0.0; dst.required_len().unwrap()];
    context
        .tensorcompose_planned_into(
            &dst_bound,
            &mut lazy,
            (crate::FusionOperand::adjoint(&source), &lhs),
            (crate::FusionOperand::direct(&source), &rhs),
            1.0,
            crate::ContractDestinationInit::Axpby(0.0),
        )
        .unwrap();

    assert_eq!(lazy, reference);
    // Not the degenerate all-zero destination the assertion would also accept.
    assert!(lazy.iter().any(|value| *value != 0.0));
}

#[test]
fn tensorcompose_fusion_preflights_all_extents_before_mutating_destination() {
    // What: both cold and cached composition plans reject malformed source or
    // destination extents before beta scaling or grouped GEMM changes output.
    let rule = FermionParityFusionRule;
    let layout = fermion_parity_matrix_space_with_homspace(fermion_parity_matrix_homspace());
    let source = layout;
    let dst =
        crate::DynamicFusionMapSpace::contracted(&rule, &source, &source, &[1], &[0]).unwrap();
    let provider = Arc::new(rule);
    let dst_bound = crate::BoundDynamicFusionMapSpace::bind_multiplicity_free(
        dst.clone(),
        Arc::clone(&provider),
    )
    .unwrap();
    let lhs = vec![1.0; source.required_len().unwrap()];
    let rhs = vec![2.0; source.required_len().unwrap()];
    let operand = crate::FusionOperand::direct(&source);
    let mut context = crate::TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();

    let mut valid = vec![0.0; dst.required_len().unwrap()];
    context
        .tensorcompose_planned_into(
            &dst_bound,
            &mut valid,
            (operand, &lhs),
            (operand, &rhs),
            1.0,
            crate::ContractDestinationInit::Axpby(0.0),
        )
        .unwrap();

    let mut destination = vec![7.0; dst.required_len().unwrap()];
    let before = destination.clone();
    let error = context
        .tensorcompose_planned_into(
            &dst_bound,
            &mut destination,
            (operand, &lhs[..lhs.len() - 1]),
            (operand, &rhs),
            1.0,
            crate::ContractDestinationInit::Axpby(0.5),
        )
        .unwrap_err();
    assert!(matches!(error, OperationError::ElementCountMismatch { .. }));
    assert_eq!(destination, before);

    let mut short_destination = vec![9.0; dst.required_len().unwrap() - 1];
    let before = short_destination.clone();
    let error = context
        .tensorcompose_planned_into(
            &dst_bound,
            &mut short_destination,
            (operand, &lhs),
            (operand, &rhs),
            1.0,
            crate::ContractDestinationInit::Axpby(0.5),
        )
        .unwrap_err();
    assert!(matches!(error, OperationError::ElementCountMismatch { .. }));
    assert_eq!(short_destination, before);
}

#[test]
fn fermionic_tensorcompose_keeps_coefficient_free_semantics() {
    // What: a dual odd RHS leg gives ordinary contraction its supertrace sign,
    // while composition retains and reuses a separate direct Core plan.
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = FermionParityFusionRule;
    let odd = SectorId::new(1);
    let typed_space = |codomain_dual: bool, domain_dual: bool| {
        DynamicFusionMapSpace::from_final_homspace(
            &rule,
            FusionTreeHomSpace::new(
                FusionProductSpace::new([SectorLeg::new([(odd, 1)], codomain_dual)]),
                FusionProductSpace::new([SectorLeg::new([(odd, 1)], domain_dual)]),
            ),
        )
        .unwrap()
    };
    let lhs = typed_space(false, true);
    let rhs = typed_space(true, false);
    let dst = crate::DynamicFusionMapSpace::contracted(&rule, &lhs, &rhs, &[1], &[0]).unwrap();
    let provider = Arc::new(rule);
    let lhs_bound = crate::BoundDynamicFusionMapSpace::bind_multiplicity_free(
        lhs.clone(),
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
    let mut context = crate::TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
    let mut contracted = vec![0.0; dst.required_len().unwrap()];
    context
        .tensorcontract_fusion_dyn_into(
            &dst_bound,
            &mut contracted,
            &lhs_bound,
            &[2.0],
            &rhs_bound,
            &[3.0],
            TensorContractSpec::new(&[1], &[0], crate::OutputAxisOrder::identity()),
            1.0,
            0.0,
        )
        .unwrap();
    assert_eq!(contracted, [-6.0]);

    let compose = |context: &mut crate::TensorContractFusionExecutionContext<f64, RuleIdentity>| {
        let mut output = vec![0.0; dst.required_len().unwrap()];
        context
            .tensorcompose_planned_into(
                &dst_bound,
                &mut output,
                (crate::FusionOperand::direct(&lhs), &[2.0]),
                (crate::FusionOperand::direct(&rhs), &[3.0]),
                1.0,
                crate::ContractDestinationInit::Axpby(0.0),
            )
            .unwrap();
        output
    };
    assert_eq!(compose(&mut context), [6.0]);
    assert!(context.last_resolution_is_core());
    assert_eq!(compose(&mut context), [6.0]);

    #[derive(Default)]
    struct VecGemm {
        unit_calls: usize,
        scaled_alphas: Vec<f64>,
    }

    impl symtn_operations::fusion_replay::StorageGemm<f64, Vec<f64>, Vec<f64>, Vec<f64>> for VecGemm {
        fn supports_matmul_with_ops_scaled(&self, lhs_op: MatrixOp, rhs_op: MatrixOp) -> bool {
            lhs_op == MatrixOp::Identity && rhs_op == MatrixOp::Identity
        }

        fn matmul_range_into(
            &mut self,
            dst: &mut Vec<f64>,
            dst_offset: usize,
            lhs: &Vec<f64>,
            lhs_offset: usize,
            rhs: &Vec<f64>,
            rhs_offset: usize,
            rows: usize,
            contracted: usize,
            cols: usize,
        ) -> Result<(), OperationError> {
            self.unit_calls += 1;
            for col in 0..cols {
                for row in 0..rows {
                    dst[dst_offset + row + rows * col] = (0..contracted)
                        .map(|inner| {
                            lhs[lhs_offset + row + rows * inner]
                                * rhs[rhs_offset + inner + contracted * col]
                        })
                        .sum();
                }
            }
            Ok(())
        }

        fn matmul_range_with_ops_scaled_into(
            &mut self,
            dst: &mut Vec<f64>,
            dst_offset: usize,
            lhs: &Vec<f64>,
            lhs_offset: usize,
            rhs: &Vec<f64>,
            rhs_offset: usize,
            rows: usize,
            contracted: usize,
            cols: usize,
            lhs_op: MatrixOp,
            rhs_op: MatrixOp,
            alpha: f64,
        ) -> Result<(), OperationError> {
            assert_eq!(lhs_op, MatrixOp::Identity);
            assert_eq!(rhs_op, MatrixOp::Identity);
            self.scaled_alphas.push(alpha);
            for col in 0..cols {
                for row in 0..rows {
                    dst[dst_offset + row + rows * col] = alpha
                        * (0..contracted)
                            .map(|inner| {
                                lhs[lhs_offset + row + rows * inner]
                                    * rhs[rhs_offset + inner + contracted * col]
                            })
                            .sum::<f64>();
                }
            }
            Ok(())
        }
    }

    let lhs_values = vec![2.0];
    let rhs_values = vec![3.0];
    let mut direct_composed = vec![0.0; dst.required_len().unwrap()];
    compose_direct_on_storage(
        &mut VecGemm::default(),
        &dst_bound,
        &mut direct_composed,
        crate::FusionOperand::direct(lhs_bound.space()),
        &lhs_values,
        crate::FusionOperand::direct(rhs_bound.space()),
        &rhs_values,
    )
    .unwrap();
    assert_eq!(direct_composed, [6.0]);

    let mut direct_contracted = vec![0.0; dst.required_len().unwrap()];
    let mut contract_gemm = VecGemm::default();
    crate::contract::tensorcontract_fusion_dyn_prelowered_direct_on_storage(
        &mut contract_gemm,
        &dst_bound,
        &mut direct_contracted,
        crate::FusionOperand::direct(lhs_bound.space()),
        &lhs_values,
        crate::FusionOperand::direct(rhs_bound.space()),
        &rhs_values,
        TensorContractSpec::new(&[1], &[0], crate::OutputAxisOrder::identity()),
    )
    .unwrap();
    assert_eq!(direct_contracted, [-6.0]);
    assert_eq!(contract_gemm.scaled_alphas, [-1.0]);

    let mixed_space = |codomain_dual: bool, domain_dual: bool| {
        DynamicFusionMapSpace::from_final_homspace(
            &rule,
            FusionTreeHomSpace::new(
                FusionProductSpace::new([SectorLeg::new(
                    [(SectorId::new(0), 1), (odd, 1)],
                    codomain_dual,
                )]),
                FusionProductSpace::new([SectorLeg::new(
                    [(SectorId::new(0), 1), (odd, 1)],
                    domain_dual,
                )]),
            ),
        )
        .unwrap()
    };
    let mixed_lhs = mixed_space(false, true);
    let mixed_rhs = mixed_space(true, false);
    let mixed_dst =
        crate::DynamicFusionMapSpace::contracted(&rule, &mixed_lhs, &mixed_rhs, &[1], &[0])
            .unwrap();
    let mixed_lhs_bound =
        crate::BoundDynamicFusionMapSpace::bind_multiplicity_free(mixed_lhs, Arc::clone(&provider))
            .unwrap();
    let mixed_rhs_bound =
        crate::BoundDynamicFusionMapSpace::bind_multiplicity_free(mixed_rhs, Arc::clone(&provider))
            .unwrap();
    let mixed_dst_bound =
        crate::BoundDynamicFusionMapSpace::bind_multiplicity_free(mixed_dst, Arc::clone(&provider))
            .unwrap();
    let mut mixed_output = vec![0.0; mixed_dst_bound.space().required_len().unwrap()];
    let mixed_lhs_values = vec![2.0, 5.0];
    let mixed_rhs_values = vec![3.0, 7.0];
    let mut mixed_gemm = VecGemm::default();
    crate::contract::tensorcontract_fusion_dyn_prelowered_direct_on_storage(
        &mut mixed_gemm,
        &mixed_dst_bound,
        &mut mixed_output,
        crate::FusionOperand::direct(mixed_lhs_bound.space()),
        &mixed_lhs_values,
        crate::FusionOperand::direct(mixed_rhs_bound.space()),
        &mixed_rhs_values,
        TensorContractSpec::new(&[1], &[0], crate::OutputAxisOrder::identity()),
    )
    .unwrap();
    assert_eq!(mixed_output, [6.0, -35.0]);
    assert_eq!(mixed_gemm.unit_calls, 1);
    assert_eq!(mixed_gemm.scaled_alphas, [-1.0]);

    struct FailingGemm;

    impl symtn_operations::fusion_replay::StorageGemm<f64, Vec<f64>, Vec<f64>, Vec<f64>>
        for FailingGemm
    {
        fn supports_matmul_with_ops_scaled(&self, lhs_op: MatrixOp, rhs_op: MatrixOp) -> bool {
            lhs_op == MatrixOp::Identity && rhs_op == MatrixOp::Identity
        }

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
            Err(OperationError::UnsupportedTensorContractScope {
                message: "injected storage GEMM failure",
            })
        }

        fn matmul_range_with_ops_scaled_into(
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
            _lhs_op: MatrixOp,
            _rhs_op: MatrixOp,
            _alpha: f64,
        ) -> Result<(), OperationError> {
            Err(OperationError::UnsupportedTensorContractScope {
                message: "injected scaled storage GEMM failure",
            })
        }
    }

    let mut failed = vec![0.0; dst.required_len().unwrap()];
    assert!(
        crate::contract::tensorcontract_fusion_dyn_prelowered_direct_on_storage(
            &mut FailingGemm,
            &dst_bound,
            &mut failed,
            crate::FusionOperand::direct(lhs_bound.space()),
            &lhs_values,
            crate::FusionOperand::direct(rhs_bound.space()),
            &rhs_values,
            TensorContractSpec::new(&[1], &[0], crate::OutputAxisOrder::identity()),
        )
        .is_err()
    );
    assert_eq!(failed, [0.0]);
    assert_eq!(lhs_values, [2.0]);
    assert_eq!(rhs_values, [3.0]);
}
