use super::*;

#[test]
fn fusion_context_accepts_explicit_backend_type_arguments() {
    let _: TensorContractFusionExecutionContext<
        f64,
        RuleIdentity,
        DenseTreeTransformOperations,
        DenseTreeTransformOperations,
    > = Default::default();
}

#[test]
fn fibonacci_complex_provider_replays_crossing_free_prepared_contraction() {
    use symtn_core::{SectorCodec, TableRule, TableSector};

    // What: the contraction lane carries the provider's Complex64 coefficient type through the
    // public prepare/replay contraction lane. Anyonic crossings are intentionally
    // absent: ordinary contraction must not infer them; #633 owns explicit planar lowering.
    let rule = TableRule::fibonacci();
    let tau = rule.encode_sector(&TableSector::new(2)).unwrap();
    let leg = || SectorLeg::new([(tau, 1)], false);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg()]),
        FusionProductSpace::new([leg()]),
    );
    let space = DynamicFusionMapSpace::from_final_homspace(&rule, hom).unwrap();
    let lhs = TestTensor::<Complex64>::from_vec_with_fusion_space(
        vec![Complex64::new(2.0, 1.0)],
        space.clone(),
    )
    .unwrap();
    let rhs = TestTensor::<Complex64>::from_vec_with_fusion_space(
        vec![Complex64::new(3.0, -2.0)],
        space.clone(),
    )
    .unwrap();
    let mut dst =
        TestTensor::<Complex64>::from_vec_with_fusion_space(vec![Complex64::new(0.0, 0.0)], space)
            .unwrap();
    let axes = TensorContractSpec::with_default_output_order(&[1], &[0]);
    let mut context = TensorContractFusionExecutionContext::<
        Complex64,
        RuleIdentity,
        DenseTreeTransformOperations,
        DenseTreeTransformOperations,
        Complex64,
    >::default();
    let prepared = fixture_prepare_contract(&mut context, &rule, &dst, &lhs, &rhs, axes).unwrap();

    for _ in 0..2 {
        dst.data_mut()[0] = Complex64::new(0.0, 0.0);
        fixture_execute_prepared(
            &mut context,
            &prepared,
            &mut dst,
            &lhs,
            &rhs,
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        )
        .unwrap();
    }

    assert_eq!(dst.data(), &[Complex64::new(8.0, -1.0)]);
}

// Mutation oracle for unchecked internal rigid-symbol providers, not a
// physically coherent ribbon category.
#[derive(Clone, Copy, Debug)]
pub(super) struct AdversarialNonuniformTwistRule;

impl FusionRule for AdversarialNonuniformTwistRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Fermionic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        SectorId::new((3 - sector.id()) % 3)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        vec![SectorId::new((left.id() + right.id()) % 3)].into()
    }
}

impl MultiplicityFreeFusionRule for AdversarialNonuniformTwistRule {}

impl MultiplicityFreeFusionSymbols for AdversarialNonuniformTwistRule {
    type Scalar = f64;

    fn f_symbol_scalar(
        &self,
        _left: SectorId,
        _middle: SectorId,
        _right: SectorId,
        _coupled: SectorId,
        _left_coupled: SectorId,
        _right_coupled: SectorId,
    ) -> Self::Scalar {
        1.0
    }

    fn r_symbol_scalar(
        &self,
        _left: SectorId,
        _right: SectorId,
        _coupled: SectorId,
    ) -> Self::Scalar {
        1.0
    }
}

impl MultiplicityFreeRigidSymbols for AdversarialNonuniformTwistRule {
    fn dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn inv_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn twist_scalar(&self, sector: SectorId) -> Self::Scalar {
        if sector == SectorId::new(1) {
            -1.0
        } else {
            1.0
        }
    }

    fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }
}

#[derive(Default)]
pub(super) struct CpuOrientedGemm {
    pub(super) calls: Vec<(usize, usize, MatrixOp, MatrixOp, f64)>,
}

impl CpuOrientedGemm {
    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        dst: &mut [f64],
        dst_offset: usize,
        lhs: &[f64],
        lhs_offset: usize,
        rhs: &[f64],
        rhs_offset: usize,
        rows: usize,
        contracted: usize,
        cols: usize,
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
        alpha: f64,
    ) {
        self.calls
            .push((lhs_offset, rhs_offset, lhs_op, rhs_op, alpha));
        for col in 0..cols {
            for row in 0..rows {
                dst[dst_offset + row + rows * col] = alpha
                    * (0..contracted)
                        .map(|inner| {
                            let lhs_index = match lhs_op {
                                MatrixOp::Identity => row + rows * inner,
                                MatrixOp::Adjoint => inner + contracted * row,
                                MatrixOp::Transpose => {
                                    unreachable!("test does not admit transpose")
                                }
                            };
                            let rhs_index = match rhs_op {
                                MatrixOp::Identity => inner + contracted * col,
                                MatrixOp::Adjoint => col + cols * inner,
                                MatrixOp::Transpose => {
                                    unreachable!("test does not admit transpose")
                                }
                            };
                            lhs[lhs_offset + lhs_index] * rhs[rhs_offset + rhs_index]
                        })
                        .sum::<f64>();
            }
        }
    }
}

impl symtn_operations::fusion_replay::StorageGemm<f64, Vec<f64>, Vec<f64>, Vec<f64>>
    for CpuOrientedGemm
{
    fn supports_matmul_with_ops_scaled(&self, lhs_op: MatrixOp, rhs_op: MatrixOp) -> bool {
        let supported = |op| matches!(op, MatrixOp::Identity | MatrixOp::Adjoint);
        supported(lhs_op) && supported(rhs_op)
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
        self.run(
            dst,
            dst_offset,
            lhs,
            lhs_offset,
            rhs,
            rhs_offset,
            rows,
            contracted,
            cols,
            MatrixOp::Identity,
            MatrixOp::Identity,
            1.0,
        );
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
        self.run(
            dst, dst_offset, lhs, lhs_offset, rhs, rhs_offset, rows, contracted, cols, lhs_op,
            rhs_op, alpha,
        );
        Ok(())
    }
}

pub(super) fn assert_f64_bits_eq(label: &str, actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{label} bit mismatch at {index}: {actual} vs {expected}"
        );
    }
}

pub(super) fn assert_complex64_bits_eq(
    label: &str,
    actual: &[num_complex::Complex64],
    expected: &[num_complex::Complex64],
) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            (actual.re.to_bits(), actual.im.to_bits()),
            (expected.re.to_bits(), expected.im.to_bits()),
            "{label} bit mismatch at {index}: {actual} vs {expected}"
        );
    }
}

#[test]
fn tensor_contract_fusion_execution_context_reports_host_placement() {
    let context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();

    assert_eq!(context.tree_backend_placement(), Placement::Host);
    assert_eq!(context.tree_workspace_placement(), Placement::Host);
    assert_eq!(context.contract_backend_placement(), Placement::Host);
    assert_eq!(context.contract_workspace_placement(), Placement::Host);
    assert_eq!(context.fusion_block_workspace_placement(), Placement::Host);
    assert_eq!(
        context.fusion_scratch_workspace_placement(),
        Placement::Host
    );
    assert!(context.is_host_context());
}

#[test]
fn tensorcontract_fusion_block_specs_enumerate_z2_compose_blocks_and_replays() {
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = Z2FusionRule;
    let leg = || SectorLeg::new([(SectorId::new(0), 1), (SectorId::new(1), 1)], false);
    let lhs_space = DynamicFusionMapSpace::from_final_homspace(
        &rule,
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg()]),
            FusionProductSpace::new([leg()]),
        ),
    )
    .unwrap();
    let rhs_space = DynamicFusionMapSpace::from_final_homspace(
        &rule,
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg()]),
            FusionProductSpace::new([leg()]),
        ),
    )
    .unwrap();
    let dst_space = DynamicFusionMapSpace::from_final_homspace(
        &rule,
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg()]),
            FusionProductSpace::new([leg()]),
        ),
    )
    .unwrap();
    let lhs = TestTensor::<f64>::from_vec_with_fusion_space(vec![2.0, 3.0], lhs_space).unwrap();
    let rhs = TestTensor::<f64>::from_vec_with_fusion_space(vec![5.0, 7.0], rhs_space).unwrap();
    let mut dst =
        TestTensor::<f64>::from_vec_with_fusion_space(vec![10.0, 20.0], dst_space).unwrap();

    let specs = tensorcontract_fusion_block_specs(
        &rule,
        dst.fusion_space().unwrap(),
        lhs.fusion_space().unwrap(),
        rhs.fusion_space().unwrap(),
        TensorContractSpec::with_default_output_order(&[1], &[0]),
    )
    .unwrap();
    assert_eq!(
        specs,
        vec![
            TensorContractBlockSpec::new(0, 0, 0),
            TensorContractBlockSpec::new(1, 1, 1),
        ]
    );

    fusion_contract_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        TensorContractSpec::with_default_output_order(&[1], &[0]),
        2.0,
        3.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[50.0, 102.0]);

    let mut context_dst = TestTensor::<f64>::from_vec_with_dynamic_space(
        vec![10.0, 20.0],
        dst.dims().to_vec(),
        dst.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
    context_contract_into(
        &mut context,
        &rule,
        &mut context_dst,
        &lhs,
        &rhs,
        TensorContractSpec::with_default_output_order(&[1], &[0]),
        2.0,
        3.0,
    )
    .unwrap();
    assert_eq!(context_dst.data(), &[50.0, 102.0]);

    context_dst.data_mut().copy_from_slice(&[10.0, 20.0]);
    context_contract_into(
        &mut context,
        &rule,
        &mut context_dst,
        &lhs,
        &rhs,
        TensorContractSpec::with_default_output_order(&[1], &[0]),
        2.0,
        3.0,
    )
    .unwrap();
    assert_eq!(context_dst.data(), &[50.0, 102.0]);

    context_dst.data_mut().copy_from_slice(&[10.0, 20.0]);
    context_contract_into(
        &mut context,
        &rule,
        &mut context_dst,
        &lhs,
        &rhs,
        TensorContractSpec::with_default_output_order(&[1], &[0]),
        2.0,
        3.0,
    )
    .unwrap();
    assert_eq!(context_dst.data(), &[50.0, 102.0]);
    assert!(context.last_resolution_is_core());
}

#[test]
fn tensorcontract_fusion_su2_swap_matches_explicit_permute_then_compose() {
    // What: SU2 C[a b; g h] = A[a b; c d] * B[d c; g h] equals an explicit
    // RHS permutation followed by core composition.
    let rule = SU2FusionRule;
    let sectors = [
        SU2Irrep::from_twice_spin(0).sector_id(),
        SU2Irrep::from_twice_spin(1).sector_id(),
    ];
    let degeneracy = 2usize;
    let leg = || SectorLeg::new(sectors.map(|sector| (sector, degeneracy)), false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let space = |hom: &FusionTreeHomSpace| {
        DynamicFusionMapSpace::from_final_homspace(&rule, hom.clone()).unwrap()
    };
    let tensor_space = space(&homspace);
    let fill = |seed: f64| move |index: usize| 0.25 * seed + ((index * 7 + 3) % 11) as f64 - 5.0;
    let len = tensor_space.structure().required_len().unwrap();
    let lhs_data = (0..len).map(fill(1.0)).collect::<Vec<_>>();
    let rhs_data = (0..len).map(fill(2.0)).collect::<Vec<_>>();
    let lhs =
        TestTensor::<f64>::from_vec_with_fusion_space(lhs_data, tensor_space.clone()).unwrap();
    let rhs =
        TestTensor::<f64>::from_vec_with_fusion_space(rhs_data, tensor_space.clone()).unwrap();

    // Route under test: swap axes through the fusion contraction facade.
    let mut dst_swap =
        TestTensor::<f64>::from_vec_with_fusion_space(vec![0.0; len], tensor_space.clone())
            .unwrap();
    fusion_contract_into(
        &rule,
        &mut dst_swap,
        &lhs,
        &rhs,
        TensorContractSpec::new(&[3, 2], &[0, 1], OutputAxisOrder::from_axes(&[0, 1, 2, 3])),
        1.0,
        0.0,
    )
    .unwrap();

    // Reference: explicitly permute rhs codomain legs, then core compose.
    let permuted_space = space(&homspace);
    let mut rhs_permuted =
        TestTensor::<f64>::from_vec_with_fusion_space(vec![0.0; len], permuted_space).unwrap();
    eager_tree_transform_into(
        &rule,
        TreeTransformOperation::permute([1, 0], [2, 3]),
        &mut rhs_permuted,
        &rhs,
        1.0,
        0.0,
    )
    .unwrap();
    let mut dst_compose =
        TestTensor::<f64>::from_vec_with_fusion_space(vec![0.0; len], tensor_space.clone())
            .unwrap();
    fusion_contract_into(
        &rule,
        &mut dst_compose,
        &lhs,
        &rhs_permuted,
        TensorContractSpec::with_default_output_order(&[2, 3], &[0, 1]),
        1.0,
        0.0,
    )
    .unwrap();

    for (index, (&actual, &expected)) in dst_swap.data().iter().zip(dst_compose.data()).enumerate()
    {
        assert!(
            (actual - expected).abs() < 1.0e-10,
            "swap vs permute+compose mismatch at {index}: {actual} vs {expected}"
        );
    }

    let dst_dyn = dst_compose.fusion_space().unwrap().as_ref().clone();
    let lhs_dyn = lhs.fusion_space().unwrap().as_ref().clone();
    let rhs_dyn = rhs.fusion_space().unwrap().as_ref().clone();
    for candidate in crate::contract::contracted_axis_order_candidates(&[3, 2], &[0, 1]) {
        for orientation in [
            crate::contract::FusionContractOrientation::LhsRhs,
            crate::contract::FusionContractOrientation::RhsLhs,
        ] {
            let plan = crate::contract::prepare_tensorcontract_fusion_plan_dyn_raw_with_axis_order_and_orientation(
                &rule,
                &dst_dyn,
                &lhs_dyn,
                &rhs_dyn,
                TensorContractSpec::new(
                    &[3, 2],
                    &[0, 1],
                    OutputAxisOrder::from_axes(&[0, 1, 2, 3]),
                ),
                &candidate,
                orientation,
            )
            .unwrap();
            let mut actual =
                TestTensor::<f64>::from_vec_with_fusion_space(vec![0.0; len], tensor_space.clone())
                    .unwrap();
            let mut tree_backend = DenseTreeTransformOperations::default_executor();
            let mut tree_workspace = TreeTransformWorkspace::default();
            let mut contract_backend = DenseTreeTransformOperations::default_executor();
            let mut contract_workspace = TensorContractWorkspace::default();
            crate::contract::tensorcontract_fusion_dynamic_plan_into_with(
                &mut tree_backend,
                &mut tree_workspace,
                &mut contract_backend,
                &mut contract_workspace,
                &rule,
                &plan,
                &mut actual,
                &lhs,
                &rhs,
                1.0,
                0.0,
            )
            .unwrap();
            let mut artifact =
                TestTensor::<f64>::from_vec_with_fusion_space(vec![0.0; len], tensor_space.clone())
                    .unwrap();
            crate::contract::execute_dynamic_tree_execution_artifact_for_test(
                &rule,
                &plan,
                &mut artifact,
                &lhs,
                &rhs,
                1.0,
                0.0,
            )
            .unwrap();
            // What: either physical operand orientation compiles to the same
            // SU2 destination structure and reduced-block values as the
            // explicit permute-then-compose oracle.
            assert_eq!(artifact.structure(), dst_compose.structure());
            // The compiled artifact and the direct plan replay are two
            // executors of one plan whose SU2 recoupling sums may associate
            // differently, so their agreement is checked under the tolerance
            // rule (terms bounded by len(lhs) * len(rhs), which covers the recoupled
            // trees as well as the contracted length). Profiling replays
            // the same artifact and stays exact.
            numerics::assert_slices_close(
                "compiled artifact vs direct plan replay",
                artifact.data(),
                actual.data(),
                lhs.data().len() * rhs.data().len(),
            );
            for (&actual, &expected) in artifact.data().iter().zip(dst_compose.data()) {
                assert!((actual - expected).abs() < 1.0e-10);
            }
            for (&actual, &expected) in actual.data().iter().zip(dst_compose.data()) {
                assert!((actual - expected).abs() < 1.0e-10);
            }
        }
    }
}
