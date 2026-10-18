use super::super::scratch::DynamicFusionScratchWorkspace;
use super::*;
use crate::tests::TestTensor;

// Reference contraction path, test-only: the allocation/output reference the
// `run_host_reference` tests compare against.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn tensorcontract_fusion_dynamic_plan_into_context<RuleKey, BT, BC, R, D>(
    tree_context: &mut TreeTransformExecutionContext<D, RuleKey, f64, BT>,
    contract_backend: &mut BC,
    contract_workspace: &mut BC::Workspace,
    fusion_block_workspace: &mut FusionBlockContractWorkspace<D>,
    scratch: &mut DynamicFusionScratchWorkspace<D>,
    rule: &R,
    plan: &FusionContractPlan,
    dst: &mut TestTensor<D>,
    lhs: &TestTensor<D>,
    rhs: &TestTensor<D>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    RuleKey: 'static + Clone + Eq + std::hash::Hash + Send + Sync,
    BT: TreeTransformBackend<D, f64>,
    BT::Workspace: Default,
    BC: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
{
    let dst_space = dst.dynamic_space().clone();
    let lhs_space = lhs.dynamic_space().clone();
    let rhs_space = rhs.dynamic_space().clone();
    let dst_structure = std::sync::Arc::clone(dst.structure());
    let lhs_structure = std::sync::Arc::clone(lhs.structure());
    let rhs_structure = std::sync::Arc::clone(rhs.structure());
    tensorcontract_fusion_dynamic_plan_dyn_into_context(
        tree_context,
        contract_backend,
        contract_workspace,
        fusion_block_workspace,
        scratch,
        rule,
        encoded_layout_primer::<R>,
        plan,
        &dst_space,
        &dst_structure,
        dst.data_mut(),
        &lhs_space,
        &lhs_structure,
        lhs.data(),
        &rhs_space,
        &rhs_structure,
        rhs.data(),
        alpha,
        beta,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_dynamic_tree_execution_artifact_for_test<R, D>(
    rule: &R,
    plan: &FusionContractPlan,
    dst: &mut TestTensor<D>,
    lhs: &TestTensor<D>,
    rhs: &TestTensor<D>,
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
{
    let mut tree_context =
        TreeTransformExecutionContext::new(DenseTreeTransformOperations::default_executor());
    let mut contract_backend = DenseTreeTransformOperations::default();
    let mut contract_workspace = super::backend::TensorContractWorkspace::default();
    let mut fusion_block_workspace = FusionBlockContractWorkspace::default();
    let mut scratch = DynamicFusionScratchWorkspace::default();
    tensorcontract_fusion_dynamic_plan_into_context(
        &mut tree_context,
        &mut contract_backend,
        &mut contract_workspace,
        &mut fusion_block_workspace,
        &mut scratch,
        rule,
        plan,
        dst,
        lhs,
        rhs,
        alpha,
        beta,
    )
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_prelowered_dynamic_tree_execution_artifact_for_test<R, D>(
    rule: &R,
    plan: &FusionContractPlan,
    dst_space: &DynamicFusionMapSpace,
    dst_data: &mut [D],
    lhs: super::dynamic_space::FusionOperand<'_>,
    lhs_data: &[D],
    rhs: super::dynamic_space::FusionOperand<'_>,
    rhs_data: &[D],
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + TreeTransformRuleCacheKey<Key = crate::RuleIdentity>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
{
    let mut tree_context: TreeTransformExecutionContext<D, crate::RuleIdentity, f64, _> =
        TreeTransformExecutionContext::new(DenseTreeTransformOperations::default_executor());
    let mut contract_backend = DenseTreeTransformOperations::default();
    let mut contract_workspace = super::backend::TensorContractWorkspace::default();
    let mut fusion_block_workspace = FusionBlockContractWorkspace::default();
    let mut scratch = DynamicFusionScratchWorkspace::default();
    let layout_primer = encoded_layout_primer::<R>;
    let lhs_layout = lhs.prepare(rule, layout_primer)?;
    let rhs_layout = rhs.prepare(rule, layout_primer)?;
    let artifact = compile_prelowered_dynamic_tree_execution_artifact(
        tree_context.planning(),
        rule,
        layout_primer,
        plan,
        dst_space,
        &lhs_layout,
        &rhs_layout,
    )?;
    execute_dynamic_tree_execution_artifact(
        &mut tree_context,
        &mut contract_backend,
        &mut contract_workspace,
        &mut fusion_block_workspace,
        &mut scratch,
        &artifact,
        dst_space.structure(),
        dst_data,
        lhs_data,
        rhs_data,
        alpha,
        beta,
    )
}

/// Dynamic-rank core of the TensorKit `@tensor`-shaped route: source
/// tree-pair transforms, core coupled GEMM, optional output transform. All
/// operands are (space, storage structure, raw slice) triples.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn tensorcontract_fusion_dynamic_plan_dyn_into_context<RuleKey, BT, BC, R, D>(
    tree_context: &mut TreeTransformExecutionContext<D, RuleKey, f64, BT>,
    contract_backend: &mut BC,
    contract_workspace: &mut BC::Workspace,
    fusion_block_workspace: &mut FusionBlockContractWorkspace<D>,
    scratch: &mut DynamicFusionScratchWorkspace<D>,
    rule: &R,
    layout_primer: LayoutKeyBuilder<R>,
    plan: &FusionContractPlan,
    dst_space: &DynamicFusionMapSpace,
    dst_structure: &Arc<BlockStructure>,
    dst_data: &mut [D],
    lhs_space: &DynamicFusionMapSpace,
    lhs_structure: &Arc<BlockStructure>,
    lhs_data: &[D],
    rhs_space: &DynamicFusionMapSpace,
    rhs_structure: &Arc<BlockStructure>,
    rhs_data: &[D],
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    RuleKey: 'static + Clone + Eq + std::hash::Hash + Send + Sync,
    BT: TreeTransformBackend<D, f64>,
    BT::Workspace: Default,
    BC: TensorContractBackend<D, f64>,
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleKey>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64>,
{
    let artifact =
        compile_dynamic_tree_execution_artifact::<symtn_core::MultiplicityFreeAdmissionMode, _>(
            tree_context.planning(),
            rule,
            layout_primer,
            plan,
            dst_space,
            lhs_space,
            lhs_structure,
            rhs_space,
            rhs_structure,
        )?;
    execute_dynamic_tree_execution_artifact(
        tree_context,
        contract_backend,
        contract_workspace,
        fusion_block_workspace,
        scratch,
        &artifact,
        dst_structure,
        dst_data,
        lhs_data,
        rhs_data,
        alpha,
        beta,
    )
}

/// The eager `DynamicTree` route executor at one member, over caller-owned
/// test resources: the arm `execute_contract_route_host` runs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_dynamic_tree_execution_artifact<RuleKey, BT, BC, D, C>(
    tree_context: &mut TreeTransformExecutionContext<D, RuleKey, C, BT>,
    contract_backend: &mut BC,
    contract_workspace: &mut BC::Workspace,
    fusion_block_workspace: &mut FusionBlockContractWorkspace<D>,
    scratch: &mut DynamicFusionScratchWorkspace<D>,
    artifact: &DynamicTreeExecutionArtifact<C>,
    dst_structure: &Arc<BlockStructure>,
    dst_data: &mut [D],
    lhs_data: &[D],
    rhs_data: &[D],
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    RuleKey: 'static + Clone + Eq + std::hash::Hash + Send + Sync,
    BT: TreeTransformBackend<D, C>,
    BT::Workspace: Default,
    BC: TensorContractBackend<D, C>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    C: DenseBlockScalar,
{
    super::super::route_host::execute_dynamic_tree_route_host(
        &mut super::super::route_host::EagerTreeStage {
            backend: tree_context.backend_mut(),
            workspaces: &mut Default::default(),
        },
        &mut super::fusion_block::BackendRank2Gemm::new(contract_backend, contract_workspace),
        fusion_block_workspace,
        scratch.route_scratch(),
        artifact,
        (dst_structure, dst_data),
        lhs_data,
        rhs_data,
        1,
        alpha,
        symtn_operations::ContractDestinationInit::Axpby(beta),
    )
}
