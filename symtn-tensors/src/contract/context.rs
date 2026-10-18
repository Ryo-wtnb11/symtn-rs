use std::hash::Hash;
use std::sync::Arc;

use symtn_core::{
    BlockStructure, CheckedFusionAlgebra, MultiplicityFreeAdmissionMode,
    MultiplicityFreeRigidSymbols, Placement,
};

use crate::mode::{
    ContractRequest, ContractSide, ContractStaging, MultiplicityFreePlanningScalar,
    PlanningAlgebra, StagedContraction, TreeStructureSource,
};
use crate::tree_context::TreeTransformExecutionContext;
use crate::tree_transform::{TreeTransformPlanning, TreeTransformRuleCacheKey};
use crate::{
    DenseBlockScalar, DenseRecouplingScalar, DenseTreeTransformOperations, OperationError,
    RecouplingCoefficientAction, ReportsPlacement, TreeTransformBackend, ZeroBytes,
};
use symtn_operations::{
    ContractDestinationInit, OutputAxisOrder, TensorContractSpec, TreeTransformWorkspace,
};

use super::backend::TensorContractBackend;
#[cfg(test)]
use super::dynamic_space::encoded_layout_primer;
use super::dynamic_space::{
    BoundDynamicFusionMapSpace, DynamicFusionMapSpace, FusionOperand, LayoutKeyBuilder,
};
#[cfg(test)]
use super::fusion::FusionContractOrientation;
use super::fusion::{
    prepare_tensorcontract_fusion_plan_dyn_prelowered_canonical, FusionContractPlan,
};
use super::fusion_block::FusionBlockContractWorkspace;
use super::planar::PlanarContractOrders;
use super::resolution::try_compile_oriented_storage_contract_plan;
use super::resolution::{
    try_compile_oriented_storage_contract_candidate_plan, ContractKind, ContractRoute, CopyCRoute,
    CoreMiss, CoreRoute, ExecCaps, StorageContractResolution,
};
use super::scratch::DynamicFusionScratchWorkspace;
use crate::host_scratch::HostScratchBuffer;

pub struct TensorContractFusionExecutionContext<
    D,
    RuleKey,
    BT = DenseTreeTransformOperations,
    BC = DenseTreeTransformOperations,
    C = f64,
> where
    D: DenseBlockScalar + RecouplingCoefficientAction<C>,
    C: DenseBlockScalar,
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
    BT: TreeTransformBackend<D, C>,
    BC: TensorContractBackend<D, C>,
{
    tree_context: TreeTransformExecutionContext<D, RuleKey, C, BT>,
    /// One transform workspace per contraction stage, indexed by `Stage`.
    stage_workspaces: [BT::Workspace; 3],
    contract_backend: BC,
    contract_workspace: BC::Workspace,
    fusion_block_workspace: FusionBlockContractWorkspace<D>,
    fusion_scratch: DynamicFusionScratchWorkspace<D>,
    copy_c_scratch: HostScratchBuffer<D>,
    #[cfg(test)]
    last_top_level_resolution_was_core: bool,
    #[cfg(test)]
    last_top_level_resolution_orientation: Option<FusionContractOrientation>,
}

impl<D, RuleKey, C>
    TensorContractFusionExecutionContext<
        D,
        RuleKey,
        DenseTreeTransformOperations,
        DenseTreeTransformOperations,
        C,
    >
where
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    C: DenseBlockScalar,
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
{
    /// Replays an admitted route (Core, CopyC or DynamicTree) over uniform
    /// member-major Host stacks through the one Host route executor, with
    /// this lane's tree and contract resources and the caller's workspace:
    /// `dst = contract(lhs, rhs)` per member. `init` is
    /// [`ContractDestinationInit::Zeroed`] only when `dst`'s inactive blocks
    /// are already zero, else `Axpby(0)`.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn execute_storage_contract_members_host(
        &mut self,
        resolution: &StorageContractResolution<C>,
        dst: (&Arc<BlockStructure>, &mut [D]),
        lhs: (&Arc<BlockStructure>, &[D]),
        rhs: (&Arc<BlockStructure>, &[D]),
        workspace: &mut super::route_host::HostContractMembersWorkspace<D, C>,
        members: usize,
        init: ContractDestinationInit<D>,
    ) -> Result<(), OperationError>
    where
        C: std::ops::Neg<Output = C>,
    {
        let (tree_backend, _) = self.tree_context.backend_workspace_mut();
        super::route_host::execute_members_host(
            tree_backend,
            &mut self.contract_backend,
            &mut self.contract_workspace,
            &mut self.fusion_block_workspace,
            super::route_host::MemberRoute::Resolution {
                resolution,
                lhs: lhs.0,
                rhs: rhs.0,
            },
            dst.0,
            workspace,
            dst.1,
            lhs.1,
            rhs.1,
            members,
            init,
        )
    }
}

impl<D, RuleKey, BT, BC, C> TensorContractFusionExecutionContext<D, RuleKey, BT, BC, C>
where
    D: DenseBlockScalar + RecouplingCoefficientAction<C>,
    C: DenseBlockScalar,
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
    BT: TreeTransformBackend<D, C>,
    BC: TensorContractBackend<D, C>,
{
    pub fn with_parts(
        tree_context: TreeTransformExecutionContext<D, RuleKey, C, BT>,
        contract_backend: BC,
        contract_workspace: BC::Workspace,
    ) -> Self
    where
        BT::Workspace: Default,
    {
        Self {
            tree_context,
            stage_workspaces: Default::default(),
            contract_backend,
            contract_workspace,
            fusion_block_workspace: FusionBlockContractWorkspace::default(),
            fusion_scratch: DynamicFusionScratchWorkspace::default(),
            copy_c_scratch: HostScratchBuffer::default(),
            #[cfg(test)]
            last_top_level_resolution_was_core: false,
            #[cfg(test)]
            last_top_level_resolution_orientation: None,
        }
    }

    #[inline]
    pub fn tree_context(&self) -> &TreeTransformExecutionContext<D, RuleKey, C, BT> {
        &self.tree_context
    }

    #[inline]
    pub fn tree_context_mut(&mut self) -> &mut TreeTransformExecutionContext<D, RuleKey, C, BT> {
        &mut self.tree_context
    }

    #[inline]
    pub fn contract_backend(&self) -> &BC {
        &self.contract_backend
    }

    #[inline]
    pub fn contract_backend_mut(&mut self) -> &mut BC {
        &mut self.contract_backend
    }

    #[inline]
    pub fn contract_workspace(&self) -> &BC::Workspace {
        &self.contract_workspace
    }

    /// Lends the grow-only Host buffer that holds TensorKit
    /// `blas_contract!`'s `copyC` temporary: the default-output contraction
    /// an eager caller then permutes into its own result. It lives with this
    /// context's other execution scratch, so it is pooled and bounded the same
    /// way (largest temporary seen, per idle context). Return it with
    /// [`Self::restore_copy_c_scratch`]; a buffer lost to an error is simply
    /// reallocated by the next caller.
    #[doc(hidden)]
    pub fn take_copy_c_scratch(&mut self) -> HostScratchBuffer<D> {
        std::mem::take(&mut self.copy_c_scratch)
    }

    #[doc(hidden)]
    pub fn restore_copy_c_scratch(&mut self, scratch: HostScratchBuffer<D>) {
        self.copy_c_scratch = scratch;
    }

    /// Logical length of the retained `copyC` buffer.
    #[doc(hidden)]
    pub fn copy_c_scratch_len(&self) -> usize {
        self.copy_c_scratch.len()
    }

    #[cfg(test)]
    pub(crate) fn eager_coefficient_pack_builds(&self) -> [usize; 3]
    where
        BT: TreeTransformBackend<D, C, Workspace = TreeTransformWorkspace<D>>,
    {
        std::array::from_fn(|stage| self.stage_workspaces[stage].coefficient_pack_builds())
    }

    #[cfg(test)]
    pub(crate) fn stage_workspace_retained_bytes(&self) -> usize
    where
        BT: TreeTransformBackend<D, C, Workspace = TreeTransformWorkspace<D>>,
    {
        self.stage_workspaces
            .iter()
            .map(TreeTransformWorkspace::retained_bytes)
            .sum()
    }

    #[cfg(test)]
    pub(crate) fn last_resolution_is_core(&self) -> bool {
        self.last_top_level_resolution_was_core
    }

    #[cfg(test)]
    pub(crate) fn last_resolution_orientation(&self) -> Option<FusionContractOrientation> {
        self.last_top_level_resolution_orientation
    }

    #[cfg(test)]
    pub(super) fn record_contract_route(&mut self, resolution: &StorageContractResolution<C>) {
        self.last_top_level_resolution_was_core =
            matches!(resolution.route, ContractRoute::Core { .. });
        self.last_top_level_resolution_orientation = match &resolution.route {
            ContractRoute::DynamicTree(artifact) => Some(artifact.orientation()),
            ContractRoute::Core { swapped: true, .. } => Some(FusionContractOrientation::RhsLhs),
            ContractRoute::Core { swapped: false, .. } | ContractRoute::CopyC(_) => None,
        };
    }

    pub fn into_parts(
        self,
    ) -> (
        TreeTransformExecutionContext<D, RuleKey, C, BT>,
        BC,
        BC::Workspace,
    ) {
        (
            self.tree_context,
            self.contract_backend,
            self.contract_workspace,
        )
    }
}

impl<D, RuleKey, BT, BC, C> TensorContractFusionExecutionContext<D, RuleKey, BT, BC, C>
where
    D: DenseBlockScalar + RecouplingCoefficientAction<C>,
    C: DenseBlockScalar,
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
    BT: TreeTransformBackend<D, C, Workspace = TreeTransformWorkspace<D>>,
    BC: TensorContractBackend<D, C>,
{
    /// Allocated Host capacity retained by fusion lhs/rhs/destination scratch,
    /// the `copyC` temporary, and the tree-transform workspaces (the tree
    /// context's and one per contraction stage, coefficient packs included).
    #[doc(hidden)]
    pub fn retained_host_scratch_bytes(&self) -> usize {
        std::iter::once(self.tree_context.workspace())
            .chain(&self.stage_workspaces)
            .map(TreeTransformWorkspace::retained_bytes)
            .fold(
                self.fusion_scratch.retained_bytes().saturating_add(
                    self.copy_c_scratch
                        .capacity()
                        .saturating_mul(std::mem::size_of::<D>()),
                ),
                usize::saturating_add,
            )
    }

    /// Releases this context's retained Host fusion, `copyC` and
    /// tree-transform buffers.
    #[doc(hidden)]
    pub fn trim_host_scratch(&mut self) {
        self.fusion_scratch.clear();
        self.copy_c_scratch = HostScratchBuffer::default();
        *self.tree_context.backend_workspace_mut().1 = TreeTransformWorkspace::default();
        self.stage_workspaces = Default::default();
    }
}

impl<D, RuleKey, BT, BC, C> TensorContractFusionExecutionContext<D, RuleKey, BT, BC, C>
where
    D: DenseBlockScalar + RecouplingCoefficientAction<C>,
    C: DenseBlockScalar,
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
    BT: TreeTransformBackend<D, C> + ReportsPlacement,
    BT::Workspace: ReportsPlacement,
    BC: TensorContractBackend<D, C> + ReportsPlacement,
    BC::Workspace: ReportsPlacement,
{
    #[inline]
    pub fn tree_backend_placement(&self) -> Placement {
        self.tree_context.backend_placement()
    }

    #[inline]
    pub fn tree_workspace_placement(&self) -> Placement {
        self.tree_context.workspace_placement()
    }

    #[inline]
    pub fn contract_backend_placement(&self) -> Placement {
        self.contract_backend.placement()
    }

    #[inline]
    pub fn contract_workspace_placement(&self) -> Placement {
        self.contract_workspace.placement()
    }

    #[inline]
    pub fn fusion_block_workspace_placement(&self) -> Placement {
        self.fusion_block_workspace.placement()
    }

    #[inline]
    pub fn fusion_scratch_workspace_placement(&self) -> Placement {
        self.fusion_scratch.placement()
    }

    #[inline]
    pub fn is_host_context(&self) -> bool {
        self.tree_context.is_host_context()
            && self.contract_backend.is_host_placement()
            && self.contract_workspace.is_host_placement()
            && self.fusion_block_workspace.is_host_placement()
            && self.fusion_scratch.is_host_placement()
    }
}

impl<D, RuleKey, BT, BC, C> TensorContractFusionExecutionContext<D, RuleKey, BT, BC, C>
where
    D: DenseBlockScalar + RecouplingCoefficientAction<C>,
    C: DenseBlockScalar,
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
    BT: TreeTransformBackend<D, C>,
    BC: TensorContractBackend<D, C>,
    BT::Workspace: Default,
    BC::Workspace: Default,
{
    pub fn new(tree_backend: BT, contract_backend: BC) -> Self {
        Self::with_parts(
            TreeTransformExecutionContext::new(tree_backend),
            contract_backend,
            BC::Workspace::default(),
        )
    }
}

impl<D, RuleKey, BT, BC, C> Default for TensorContractFusionExecutionContext<D, RuleKey, BT, BC, C>
where
    D: DenseBlockScalar + RecouplingCoefficientAction<C>,
    C: DenseBlockScalar,
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
    BT: TreeTransformBackend<D, C> + Default,
    BC: TensorContractBackend<D, C> + Default,
    BT::Workspace: Default,
    BC::Workspace: Default,
{
    fn default() -> Self {
        Self::new(BT::default(), BC::default())
    }
}

impl<D, RuleKey, BT, BC, C> TensorContractFusionExecutionContext<D, RuleKey, BT, BC, C>
where
    D: DenseBlockScalar + RecouplingCoefficientAction<C>,
    C: DenseBlockScalar,
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
    BT: TreeTransformBackend<D, C>,
    BC: TensorContractBackend<D, C>,
{
    /// Dynamic-rank `tensorcontract!`: the eager Host contraction over
    /// [`BoundDynamicFusionMapSpace`] handles plus raw slices in the
    /// coupled-sector matrix layout. `dst_data` must be sized for
    /// `dst_space.required_len()`.
    #[allow(clippy::too_many_arguments)]
    pub fn tensorcontract_fusion_dyn_into<R>(
        &mut self,
        dst_space: &BoundDynamicFusionMapSpace<R>,
        dst_data: &mut [D],
        lhs_space: &BoundDynamicFusionMapSpace<R>,
        lhs_data: &[D],
        rhs_space: &BoundDynamicFusionMapSpace<R>,
        rhs_data: &[D],
        axes: TensorContractSpec<'_>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    {
        self.tensorcontract_fusion_dyn_into_with_init(
            dst_space,
            dst_data,
            lhs_space,
            lhs_data,
            rhs_space,
            rhs_data,
            axes,
            alpha,
            ContractDestinationInit::Axpby(beta),
        )
    }

    /// [`Self::tensorcontract_fusion_dyn_into`] with the destination
    /// initialisation made explicit: an owned output born all-zero passes
    /// [`ContractDestinationInit::Zeroed`] so its inactive blocks are never
    /// touched.
    #[allow(clippy::too_many_arguments)]
    pub fn tensorcontract_fusion_dyn_into_with_init<R>(
        &mut self,
        dst_space: &BoundDynamicFusionMapSpace<R>,
        dst_data: &mut [D],
        lhs_space: &BoundDynamicFusionMapSpace<R>,
        lhs_data: &[D],
        rhs_space: &BoundDynamicFusionMapSpace<R>,
        rhs_data: &[D],
        axes: TensorContractSpec<'_>,
        alpha: D,
        init: ContractDestinationInit<D>,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    {
        if !axes.lhs_conjugate() && !axes.rhs_conjugate() {
            return self.tensorcontract_prelowered_into(
                dst_space,
                dst_data,
                FusionOperand::direct(lhs_space.space()),
                lhs_data,
                FusionOperand::direct(rhs_space.space()),
                rhs_data,
                axes,
                alpha,
                init,
            );
        }
        let lowered = lower_legacy_axes(lhs_space.space(), rhs_space.space(), axes)?;
        self.tensorcontract_prelowered_into(
            dst_space,
            dst_data,
            legacy_operand(lhs_space.space(), axes.lhs_conjugate()),
            lhs_data,
            legacy_operand(rhs_space.space(), axes.rhs_conjugate()),
            rhs_data,
            lowered.as_spec(),
            alpha,
            init,
        )
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tensorcontract_fusion_dyn_into_raw<R>(
        &mut self,
        rule: &R,
        dst_space: &DynamicFusionMapSpace,
        dst_data: &mut [D],
        lhs_space: &DynamicFusionMapSpace,
        lhs_data: &[D],
        rhs_space: &DynamicFusionMapSpace,
        rhs_data: &[D],
        axes: TensorContractSpec<'_>,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    {
        let lowered = lower_legacy_axes(lhs_space, rhs_space, axes)?;
        let axes = lowered.as_spec();
        let output_axes = spec_output_axes(axes, dst_space.rank());
        self.tensorcontract_planned_raw_into::<MultiplicityFreeAdmissionMode, R>(
            &mut (),
            PlanTarget {
                rule,
                space: dst_space,
                authority: encoded_layout_primer::<R>,
            },
            dst_data,
            (legacy_operand(lhs_space, axes.lhs_conjugate()), lhs_data),
            (legacy_operand(rhs_space, axes.rhs_conjugate()), rhs_data),
            PlannedContraction::Contract((
                axes.lhs_contracting_axes(),
                axes.rhs_contracting_axes(),
                &output_axes,
            )),
            alpha,
            ContractDestinationInit::Axpby(beta),
        )
        .map(drop)
    }

    /// Test-only: the planning step of
    /// [`Self::tensorcontract_fusion_dyn_into_raw`] alone, for fixtures that
    /// replay the route through [`Self::execute_contract_resolution_host`].
    #[cfg(test)]
    pub(crate) fn plan_tensorcontract_fusion_dyn_raw<R>(
        &mut self,
        rule: &R,
        dst_space: &DynamicFusionMapSpace,
        lhs_space: &DynamicFusionMapSpace,
        rhs_space: &DynamicFusionMapSpace,
        axes: TensorContractSpec<'_>,
    ) -> Result<StorageContractResolution<C>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    {
        let lowered = lower_legacy_axes(lhs_space, rhs_space, axes)?;
        let axes = lowered.as_spec();
        let output_axes = spec_output_axes(axes, dst_space.rank());
        self.plan_contract_raw::<super::resolution::HostEagerExecutor, R>(
            PlanTarget {
                rule,
                space: dst_space,
                authority: encoded_layout_primer::<R>,
            },
            legacy_operand(lhs_space, axes.lhs_conjugate()),
            legacy_operand(rhs_space, axes.rhs_conjugate()),
            (
                axes.lhs_contracting_axes(),
                axes.rhs_contracting_axes(),
                &output_axes,
            ),
        )
    }

    /// The prelowered operand seam: logical spaces are the categorical
    /// authority, storage operands and slices the physical one. Why not a
    /// public entrypoint: arbitrary callers cannot establish the lazy-adjoint
    /// coherence contract; the typed facade enters through
    /// `tensorcontract_into_*_in` (#1870).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tensorcontract_prelowered_into<R>(
        &mut self,
        dst_space: &BoundDynamicFusionMapSpace<R>,
        dst_data: &mut [D],
        lhs: FusionOperand<'_>,
        lhs_data: &[D],
        rhs: FusionOperand<'_>,
        rhs_data: &[D],
        axes: TensorContractSpec<'_>,
        alpha: D,
        init: ContractDestinationInit<D>,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    {
        // The planner rejects operand flags that disagree with `axes`.
        if axes.lhs_conjugate() != lhs.storage_conjugate()
            || axes.rhs_conjugate() != rhs.storage_conjugate()
        {
            return Err(OperationError::InvalidArgument {
                message: "prelowered operand flags must match the contraction request",
            });
        }
        let output_axes = spec_output_axes(axes, dst_space.space().rank());
        self.tensorcontract_planned_into(
            dst_space,
            dst_data,
            (lhs, lhs_data),
            (rhs, rhs_data),
            axes.lhs_contracting_axes(),
            axes.rhs_contracting_axes(),
            &output_axes,
            alpha,
            init,
        )
    }

    /// Host eager contraction: [`Self::plan_contract`] for the Host eager
    /// executor, then its replay, `dst = alpha * contract(lhs, rhs) + beta *
    /// dst` (`init`). The one route authority Host eager, the device and
    /// `ContractPlan` share; a lazy adjoint is a storage-conjugate operand.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn tensorcontract_planned_into<R>(
        &mut self,
        dst_space: &BoundDynamicFusionMapSpace<R>,
        dst_data: &mut [D],
        (lhs, lhs_data): (FusionOperand<'_>, &[D]),
        (rhs, rhs_data): (FusionOperand<'_>, &[D]),
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_axes: &[usize],
        alpha: D,
        init: ContractDestinationInit<D>,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    {
        self.tensorcontract_planned_raw_into::<MultiplicityFreeAdmissionMode, R>(
            &mut (),
            PlanTarget::bound(dst_space),
            dst_data,
            (lhs, lhs_data),
            (rhs, rhs_data),
            PlannedContraction::Contract((lhs_axes, rhs_axes, output_axes)),
            alpha,
            init,
        )
        .map(drop)
    }

    /// The Host plan + execute path of every contraction and composition
    /// entry in either admission mode `M`: [`plan_contract_in`] (under the
    /// context's planning state or the call's transaction `txn`) or
    /// [`plan_compose_in`], then the route replay `dst = alpha · op +
    /// init(dst)` into an existing destination. Returns the replayed route,
    /// which a checked call's commit publishes over.
    #[allow(clippy::too_many_arguments)]
    fn tensorcontract_planned_raw_into<M, R>(
        &mut self,
        txn: &mut M::ContractTxn,
        target: PlanTarget<'_, R, M::SpaceAuthority<'_>>,
        dst_data: &mut [D],
        (lhs, lhs_data): (FusionOperand<'_>, &[D]),
        (rhs, rhs_data): (FusionOperand<'_>, &[D]),
        planned: PlannedContraction<'_>,
        alpha: D,
        init: ContractDestinationInit<D>,
    ) -> Result<StorageContractResolution<C>, M::Error>
    where
        M: PlanningAlgebra<R, Scalar = C>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    {
        let resolution = match planned {
            PlannedContraction::Contract(axes) => {
                plan_contract_in::<super::resolution::HostEagerExecutor, M, R>(
                    M::contract_cache(self.tree_context.planning(), txn),
                    target,
                    lhs,
                    rhs,
                    axes,
                )?
            }
            PlannedContraction::Compose => {
                plan_compose_in::<super::resolution::HostEagerExecutor, M, R>(target, lhs, rhs)?
            }
        };
        #[cfg(test)]
        self.record_contract_route(&resolution);
        self.execute_contract_route_host(
            &resolution,
            target.space.structure(),
            dst_data,
            (lhs.storage_space().structure(), lhs_data),
            (rhs.storage_space().structure(), rhs_data),
            alpha,
            init,
        )?;
        Ok(resolution)
    }

    /// Categorical map composition on the coupled-sector block matrices,
    /// TensorKit `mul!`, into an existing destination: `dst = alpha · (lhs ∘
    /// rhs) + init(dst)`, the composition sibling of
    /// [`Self::tensorcontract_planned_into`]. A's domain against B's
    /// codomain, no fermionic supertrace twist, every braiding style. It
    /// plans through [`plan_compose`], as `ComposePlan` does, for the Host
    /// eager executor, whose irregular core packs a non-canonical tiling.
    /// Lazy adjoints stay storage-conjugate operands; neither is
    /// materialized.
    #[doc(hidden)]
    pub fn tensorcompose_planned_into<R>(
        &mut self,
        dst_space: &BoundDynamicFusionMapSpace<R>,
        dst_data: &mut [D],
        (lhs, lhs_data): (FusionOperand<'_>, &[D]),
        (rhs, rhs_data): (FusionOperand<'_>, &[D]),
        alpha: D,
        init: ContractDestinationInit<D>,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    {
        self.tensorcontract_planned_raw_into::<MultiplicityFreeAdmissionMode, R>(
            &mut (),
            PlanTarget::bound(dst_space),
            dst_data,
            (lhs, lhs_data),
            (rhs, rhs_data),
            PlannedContraction::Compose,
            alpha,
            init,
        )
        .map(drop)
    }

    /// The one owned Host contraction and composition of both modes, the
    /// steps every storage runs ([`StagedContract`]): stage the destination
    /// (multiplicity-free derives it; checked Generic admits the operands and
    /// stages it), plan, replay into a fresh zeroed payload, then commit and
    /// publish (#2063).
    #[allow(clippy::type_complexity)]
    pub(super) fn tensorcontract_in<'a, St, R>(
        &mut self,
        (lhs_authority, lhs, lhs_data): (
            &'a BoundDynamicFusionMapSpace<R>,
            FusionOperand<'a>,
            &[D],
        ),
        (rhs_authority, rhs, rhs_data): (
            &'a BoundDynamicFusionMapSpace<R>,
            FusionOperand<'a>,
            &[D],
        ),
        request: ContractRequest<'a>,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), ContractStageError<'a, R, St>>
    where
        St: ContractStageParts<'a, R>,
        St::M: PlanningAlgebra<R, Scalar = C>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ZeroBytes,
    {
        let mut staged = stage_contract::<R, St>(
            (lhs_authority, lhs, lhs_data.len()),
            (rhs_authority, rhs, rhs_data.len()),
            request,
        )?;
        let resolution =
            self.plan_staged::<super::resolution::HostEagerExecutor, R, St>(&mut staged)?;
        let mut data = crate::zeroed_payload(staged_len(&staged));
        self.execute_contract_resolution_host(
            &resolution,
            (staged_destination(&staged).structure(), &mut data),
            (lhs.storage_space().structure(), lhs_data),
            (rhs.storage_space().structure(), rhs_data),
            D::one(),
            ContractDestinationInit::Zeroed,
        )?;
        let destination = staged_commit(staged, &resolution)?;
        Ok((destination, data))
    }

    /// [`Self::tensorcontract_in`] writing into an existing destination: the
    /// same staging, then the space check against `dst_space` before
    /// `dst_data` runs the caller's remaining checks, then `alpha · op +
    /// beta · dst` and the same commit. Any error before the write leaves the
    /// destination untouched; the commit follows the write, so a failed call
    /// publishes nothing (#2063).
    ///
    /// Why the match sits before planning in both modes: no contraction
    /// staging step is deferred to planning on a memo hit, so the reported
    /// error does not depend on cache history.
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    pub(super) fn tensorcontract_into_in<'a, 'd, St, R, E>(
        &mut self,
        (lhs_authority, lhs, lhs_data): (
            &'a BoundDynamicFusionMapSpace<R>,
            FusionOperand<'a>,
            &[D],
        ),
        (rhs_authority, rhs, rhs_data): (
            &'a BoundDynamicFusionMapSpace<R>,
            FusionOperand<'a>,
            &[D],
        ),
        request: ContractRequest<'a>,
        dst_space: &DynamicFusionMapSpace,
        dst_data: impl FnOnce() -> Result<&'d mut [D], E>,
        alpha: D,
        beta: D,
    ) -> Result<(), E>
    where
        St: ContractStageParts<'a, R>,
        St::M: PlanningAlgebra<R, Scalar = C>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + 'd,
        E: From<ContractStageError<'a, R, St>> + From<OperationError>,
    {
        let mut staged = stage_contract::<R, St>(
            (lhs_authority, lhs, lhs_data.len()),
            (rhs_authority, rhs, rhs_data.len()),
            request,
        )?;
        // Why `E` directly: a destination mismatch is the caller's argument
        // error in every mode, never a checked plan failure.
        staged_require_destination(&staged, dst_space)?;
        let dst = dst_data()?;
        let resolution =
            self.plan_staged::<super::resolution::HostEagerExecutor, R, St>(&mut staged)?;
        self.execute_contract_resolution_host(
            &resolution,
            (staged_destination(&staged).structure(), dst),
            (lhs.storage_space().structure(), lhs_data),
            (rhs.storage_space().structure(), rhs_data),
            alpha,
            ContractDestinationInit::Axpby(beta),
        )
        .map_err(ContractStageError::<'a, R, St>::from)?;
        staged_commit(staged, &resolution)?;
        Ok(())
    }

    /// The planner's ladder over a staged contraction: the lock-free
    /// canonical core, then the rest under this context's planning state or
    /// the call's transaction.
    fn plan_staged<'a, X: ExecCaps, R, St>(
        &mut self,
        staged: &mut StagedContract<'a, R, St>,
    ) -> Result<StorageContractResolution<C>, ContractStageError<'a, R, St>>
    where
        St: ContractStageParts<'a, R>,
        St::M: PlanningAlgebra<R, Scalar = C>,
    {
        match staged_try_core::<X, R, St>(staged)? {
            CoreRoute::Hit(core) => Ok(core),
            CoreRoute::Miss(miss) => {
                staged_plan_beyond_core::<X, R, St>(staged, miss, self.tree_context.planning())
            }
        }
    }

    /// Replays a planned route on the Host into `dst`: `dst = alpha ·
    /// contract(lhs, rhs) + init(dst)`, the Host storage's contraction
    /// kernel. `dst` is the staged destination structure the route was
    /// planned for; `lhs`/`rhs` are the operands' stored structures and
    /// payloads.
    ///
    /// This concrete cross-crate entrypoint is internal and unstable despite
    /// being public for `symtn`; downstream callers must not rely on it.
    #[doc(hidden)]
    pub fn execute_contract_resolution_host(
        &mut self,
        resolution: &StorageContractResolution<C>,
        dst: (&Arc<BlockStructure>, &mut [D]),
        lhs: (&Arc<BlockStructure>, &[D]),
        rhs: (&Arc<BlockStructure>, &[D]),
        alpha: D,
        init: ContractDestinationInit<D>,
    ) -> Result<(), OperationError>
    where
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    {
        #[cfg(test)]
        self.record_contract_route(resolution);
        self.execute_route_host(resolution, dst, lhs, rhs, alpha, init)
    }

    /// [`StagedContract::plan_beyond_core`] of a multiplicity-free
    /// contraction, under this context's planning state.
    ///
    /// This concrete cross-crate entrypoint is internal and unstable despite
    /// being public for `symtn`; downstream callers must not rely on it.
    #[doc(hidden)]
    pub fn plan_staged_contract_beyond_core<X: ExecCaps, R>(
        &mut self,
        staged: &mut StagedContract<'_, R, MultiplicityFreeContractStage<R>>,
        miss: CoreMiss,
    ) -> Result<StorageContractResolution<C>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C>
            + CheckedFusionAlgebra
            + TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        staged_plan_beyond_core::<X, R, MultiplicityFreeContractStage<R>>(
            staged,
            miss,
            self.tree_context.planning(),
        )
    }

    /// The contraction planner: TensorKit `contract!`
    /// (`tensoroperations.jl:314-357` @cfaa073) under `contract_memcost`
    /// (`:358-368`), for an executor with the capabilities `X`. Every
    /// contraction route — Host eager, the device, `ContractPlan` and prepared
    /// — is chosen here.
    ///
    /// The ladder: the canonical fully-direct core
    /// ([`try_compile_storage_contract_core_route`]), then
    /// [`Self::plan_contract_beyond_core`] — TensorKit's `copyC`, then the
    /// `DynamicTree` artifact. A caller that wants the canonical route
    /// without taking a context lock calls the two halves itself, in that
    /// order. The requested order's zero-copy candidates are walked once.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn plan_contract<X: ExecCaps, R>(
        &mut self,
        dst_space: &BoundDynamicFusionMapSpace<R>,
        lhs: FusionOperand<'_>,
        rhs: FusionOperand<'_>,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_axes: &[usize],
    ) -> Result<StorageContractResolution<C>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    {
        self.plan_contract_raw::<X, R>(
            PlanTarget::bound(dst_space),
            lhs,
            rhs,
            (lhs_axes, rhs_axes, output_axes),
        )
    }

    /// [`Self::plan_contract`] for a destination given as provider, space and
    /// layout primer, not bound: the test-only raw entries plan with their
    /// caller's rule without binding a space per call.
    pub(crate) fn plan_contract_raw<X: ExecCaps, R>(
        &mut self,
        target: PlanTarget<'_, R>,
        lhs: FusionOperand<'_>,
        rhs: FusionOperand<'_>,
        axes: (&[usize], &[usize], &[usize]),
    ) -> Result<StorageContractResolution<C>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    {
        plan_contract_in::<X, MultiplicityFreeAdmissionMode, R>(
            self.tree_context.planning(),
            target,
            lhs,
            rhs,
            axes,
        )
    }

    /// [`Self::plan_contract`] after its canonical core declined (`miss`, from
    /// [`try_compile_storage_contract_core_route`] on the same request):
    /// TensorKit's `copyC` (`blas_contract!`, `tensoroperations.jl:436-446`
    /// @cfaa073) when [`copy_c_order`](super::resolution::copy_c_order) takes it
    /// — a zero-copy core into a temporary in its own default output, then one
    /// permute — else the `DynamicTree` artifact.
    ///
    /// Why the core precedes `copyC`: a contraction that has the canonical
    /// core needs no output permute (`copy_c_order`
    /// returns `None` for it), so TensorKit's memcost never charges it `dim(C)`.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn plan_contract_beyond_core<X: ExecCaps, R>(
        &mut self,
        miss: CoreMiss,
        dst_space: &BoundDynamicFusionMapSpace<R>,
        lhs: FusionOperand<'_>,
        rhs: FusionOperand<'_>,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_axes: &[usize],
    ) -> Result<StorageContractResolution<C>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    {
        plan_contract_beyond_core_in::<X, MultiplicityFreeAdmissionMode, R>(
            self.tree_context.planning(),
            miss,
            PlanTarget::bound(dst_space),
            lhs,
            rhs,
            (lhs_axes, rhs_axes, output_axes),
        )
    }

    /// Test-only: [`try_compile_storage_contract_core_route`] followed, on a
    /// miss, by [`Self::compile_storage_contract_dynamic_tree`] — the
    /// planner's ladder without `CopyC`, for gates that pin those two routes.
    /// Production planning is [`Self::plan_contract`].
    #[cfg(test)]
    pub(crate) fn compile_storage_contract_resolution<R>(
        &mut self,
        dst_space: &BoundDynamicFusionMapSpace<R>,
        lhs: FusionOperand<'_>,
        rhs: FusionOperand<'_>,
        axes: TensorContractSpec<'_>,
    ) -> Result<StorageContractResolution<C>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    {
        match try_compile_storage_contract_core_route::<super::resolution::DirectCoreExecutor, R>(
            dst_space, lhs, rhs, axes,
        )? {
            CoreRoute::Hit(core) => Ok(core),
            CoreRoute::Miss(_) => {
                self.compile_storage_contract_dynamic_tree(dst_space, lhs, rhs, axes)
            }
        }
    }

    /// The last rung of [`Self::plan_contract`]: the
    /// Host `DynamicTree` artifact for a contraction that
    /// [`try_compile_storage_contract_core_route`] declined, compiled by the
    /// same entry the Host contraction uses for this representation — the
    /// owned compiler for two owned operands, the prelowered compiler when
    /// either is a lazy adjoint — with this context's space cache and tree
    /// context, so its transform structures are the Arcs a device executor's
    /// prepared cache is keyed by.
    ///
    /// Host eager and the device plan the same route (#1858); the one
    /// executor difference: owned core geometry over a non-canonical storage
    /// layout (an expert tiling, #1517) is a core with pack/scatter on the
    /// Host (`IRREGULAR_CORE`) and this artifact on the device.
    ///
    /// A fermionic contraction twist travels in the artifact, both as the
    /// Host's in-place scale actions and as the sorted per-block scale list
    /// the device folds into the twisted operand's source transform. A
    /// canonical contraction whose twist varies within one coupled sector
    /// reaches here (the core declines it); a uniform one never does.
    ///
    /// Validates the request itself (provider against the spaces, operand
    /// flags against `axes`), so it is sound to call without the first half.
    #[doc(hidden)]
    pub fn compile_storage_contract_dynamic_tree<R>(
        &mut self,
        dst_space: &BoundDynamicFusionMapSpace<R>,
        lhs: FusionOperand<'_>,
        rhs: FusionOperand<'_>,
        axes: TensorContractSpec<'_>,
    ) -> Result<StorageContractResolution<C>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    {
        compile_dynamic_tree_in::<MultiplicityFreeAdmissionMode, R>(
            self.tree_context.planning(),
            PlanTarget::bound(dst_space),
            lhs,
            rhs,
            axes,
            false,
            None,
        )
    }

    /// Replays a planned route ([`Self::plan_contract`] with
    /// [`HostEagerExecutor`](super::resolution::HostEagerExecutor)) on Host
    /// slices through the one Host route executor at one member:
    /// `dst = alpha * contract(lhs, rhs) + beta * dst`.
    ///
    /// `CopyC` is TensorKit `blas_contract!`'s `copyC`
    /// (`tensoroperations.jl:436-446` @cfaa073): `mul!(Cnew, A, B)` into the
    /// lane's pooled temporary, then `tensoradd!(C, Cnew, pAB, false, α, β)`.
    /// TensorKit allocates `Cnew` per call (`tensoralloc_add(..., Val(true))`);
    /// the pooled buffer plays that role, so a warm call allocates no
    /// output-sized temporary. Stale pooled values are never read: eager
    /// scratch carries no core slot, so the core writes the temporary with a
    /// strong-zero `beta = 0`, inactive blocks included.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn execute_contract_route_host(
        &mut self,
        resolution: &StorageContractResolution<C>,
        dst_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        lhs: (&Arc<BlockStructure>, &[D]),
        rhs: (&Arc<BlockStructure>, &[D]),
        alpha: D,
        init: ContractDestinationInit<D>,
    ) -> Result<(), OperationError>
    where
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    {
        self.execute_route_host(resolution, (dst_structure, dst_data), lhs, rhs, alpha, init)
    }

    /// The eager arm of every route: the one route executor at one member,
    /// over this context's tree workspace and scratch (the pooled `copyC`
    /// temporary as a CopyC core destination).
    #[allow(clippy::too_many_arguments)]
    fn execute_route_host(
        &mut self,
        resolution: &StorageContractResolution<C>,
        dst: (&Arc<BlockStructure>, &mut [D]),
        lhs: (&Arc<BlockStructure>, &[D]),
        rhs: (&Arc<BlockStructure>, &[D]),
        alpha: D,
        init: ContractDestinationInit<D>,
    ) -> Result<(), OperationError>
    where
        D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    {
        let Self {
            tree_context,
            stage_workspaces,
            contract_backend,
            contract_workspace,
            fusion_block_workspace,
            fusion_scratch,
            copy_c_scratch,
            ..
        } = self;
        let mut scratch = fusion_scratch.route_scratch();
        if resolution.copy_c().is_some() {
            scratch.core_dst = copy_c_scratch;
        }
        super::route_host::execute_route_host(
            &mut super::route_host::EagerTreeStage {
                backend: tree_context.backend_mut(),
                workspaces: stage_workspaces,
            },
            &mut super::fusion_block::BackendRank2Gemm::new(contract_backend, contract_workspace),
            fusion_block_workspace,
            scratch,
            resolution,
            dst,
            lhs,
            rhs,
            1,
            alpha,
            init,
        )
    }
}

/// A legacy conjugation flag on an owned space requests its categorical
/// adjoint, with `axes` indexing the stored tensor's legs (TensorOperations'
/// `conjA`): the lazy-adjoint operand of the space ([`legacy_operand`]), its
/// contracted axes mapped onto the adjoint's legs here. Without a flag the
/// axes are unchanged.
fn lower_legacy_axes(
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    axes: TensorContractSpec<'_>,
) -> Result<crate::lowering::LoweredTensorContractSpec, OperationError> {
    crate::lowering::lower_tensorcontract_adjoint_axes(
        lhs.nout(),
        lhs.nin(),
        rhs.nout(),
        rhs.nin(),
        axes,
    )
}

/// The operand an owned space and a legacy conjugation flag request: the flag
/// asks for the categorical adjoint, which is the lazy adjoint of the space.
fn legacy_operand(space: &DynamicFusionMapSpace, conjugate: bool) -> FusionOperand<'_> {
    if conjugate {
        FusionOperand::adjoint(space)
    } else {
        FusionOperand::direct(space)
    }
}

/// What [`TensorContractFusionExecutionContext::tensorcontract_planned_raw_into`]
/// plans: a contraction over `(lhs, rhs, output)` axes, or a composition.
#[derive(Clone, Copy)]
pub(crate) enum PlannedContraction<'s> {
    Contract((&'s [usize], &'s [usize], &'s [usize])),
    Compose,
}

/// The output permutation of `axes` over a rank-`rank` destination.
fn spec_output_axes(axes: TensorContractSpec<'_>, rank: usize) -> smallvec::SmallVec<[usize; 16]> {
    match axes.output_permutation() {
        symtn_operations::OutputAxisOrder::Axes(axes) => axes.iter().copied().collect(),
        symtn_operations::OutputAxisOrder::Identity => (0..rank).collect(),
    }
}

/// The destination a contraction is planned for: its provider, its space and
/// the mode's authority over spaces derived from it (the multiplicity-free
/// layout primer by default). Bound spaces carry all three; the test-only
/// raw entries pass their caller's rule.
pub(crate) struct PlanTarget<'a, R, A = LayoutKeyBuilder<R>> {
    pub(crate) rule: &'a R,
    pub(crate) space: &'a DynamicFusionMapSpace,
    pub(crate) authority: A,
}

// Why manual: a derive would bound `R: Copy`.
impl<R, A: Copy> Clone for PlanTarget<'_, R, A> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<R, A: Copy> Copy for PlanTarget<'_, R, A> {}

impl<'a, R> PlanTarget<'a, R> {
    pub(crate) fn bound(space: &'a BoundDynamicFusionMapSpace<R>) -> Self
    where
        R: MultiplicityFreeRigidSymbols,
    {
        Self {
            rule: space.provider(),
            space: space.space(),
            authority: space.layout_primer(),
        }
    }
}

/// One pairwise contraction or composition staged by its admission mode and
/// not yet planned: the destination derived (multiplicity-free) or staged
/// (checked Generic, published only by `commit` after the replay, #2063).
///
/// Every storage runs the same steps on it — `try_core` (lock-free), on a
/// miss `plan_beyond_core`, its own replay of the route into the staged
/// destination structure, then `commit` — so the route, the destination and
/// the publication do not depend on the storage. `St` is the mode's stage
/// ([`MultiplicityFreeContractStage`] or [`CheckedContractStage`]).
///
/// Internal and unstable despite being public for `symtn`.
#[doc(hidden)]
pub struct StagedContract<'a, R, St> {
    lhs: &'a BoundDynamicFusionMapSpace<R>,
    operands: (FusionOperand<'a>, FusionOperand<'a>),
    request: ContractRequest<'a>,
    stage: St,
}

/// A multiplicity-free contraction's derived destination and layout primer.
///
/// Internal and unstable despite being public for `symtn`.
#[doc(hidden)]
pub struct MultiplicityFreeContractStage<R>(
    StagedContraction<BoundDynamicFusionMapSpace<R>, LayoutKeyBuilder<R>, ()>,
);

/// What a contraction stage holds for its mode `M`.
pub(crate) type StageParts<'a, R, M> = StagedContraction<
    <M as ContractStaging<R>>::ContractStage,
    <M as PlanningAlgebra<R>>::SpaceAuthority<'a>,
    <M as PlanningAlgebra<R>>::ContractTxn,
>;

/// The admission mode a [`StagedContract`] stage belongs to, and its parts.
pub(crate) trait ContractStageParts<'a, R: 'a>: Sized {
    type M: ContractStaging<R>;
    fn wrap(parts: StageParts<'a, R, Self::M>) -> Self;
    fn parts(&self) -> &StageParts<'a, R, Self::M>;
    fn parts_mut(&mut self) -> &mut StageParts<'a, R, Self::M>;
    fn into_parts(self) -> StageParts<'a, R, Self::M>;
}

/// A contraction stage's mode error.
pub(crate) type ContractStageError<'a, R, St> =
    <<St as ContractStageParts<'a, R>>::M as PlanningAlgebra<R>>::Error;

/// A contraction stage's mode coefficient.
pub(crate) type ContractStageScalar<'a, R, St> =
    <<St as ContractStageParts<'a, R>>::M as PlanningAlgebra<R>>::Scalar;

/// A contraction's contracted axes and its output axes as the planner reads
/// them.
type PlannedAxes<'a> = (&'a [usize], &'a [usize], smallvec::SmallVec<[usize; 16]>);

impl<'a, R> ContractStageParts<'a, R> for MultiplicityFreeContractStage<R>
where
    R: MultiplicityFreeRigidSymbols + CheckedFusionAlgebra + TreeTransformRuleCacheKey + 'a,
    R::Scalar: MultiplicityFreePlanningScalar,
{
    type M = MultiplicityFreeAdmissionMode;
    fn wrap(parts: StageParts<'a, R, Self::M>) -> Self {
        Self(parts)
    }
    fn parts(&self) -> &StageParts<'a, R, Self::M> {
        &self.0
    }
    fn parts_mut(&mut self) -> &mut StageParts<'a, R, Self::M> {
        &mut self.0
    }
    fn into_parts(self) -> StageParts<'a, R, Self::M> {
        self.0
    }
}

pub(crate) fn stage_contract<'a, R, St>(
    lhs: ContractSide<'a, R>,
    rhs: ContractSide<'a, R>,
    request: ContractRequest<'a>,
) -> Result<StagedContract<'a, R, St>, ContractStageError<'a, R, St>>
where
    St: ContractStageParts<'a, R>,
{
    let parts = St::M::stage_contraction(lhs, rhs, request)?;
    Ok(StagedContract {
        lhs: lhs.0,
        operands: (lhs.1, rhs.1),
        request,
        stage: St::wrap(parts),
    })
}

/// The provider and destination space the plan targets.
pub(crate) fn staged_destination<'s, 'a, R, St>(
    staged: &'s StagedContract<'a, R, St>,
) -> &'s DynamicFusionMapSpace
where
    St: ContractStageParts<'a, R>,
{
    St::M::contract_destination(&staged.stage.parts().stage, staged.lhs).1
}

pub(crate) fn staged_len<'a, R, St>(staged: &StagedContract<'a, R, St>) -> usize
where
    St: ContractStageParts<'a, R>,
{
    staged.stage.parts().len
}

/// TensorKit `spacecheck_contract` against an existing destination.
pub(crate) fn staged_require_destination<'a, R, St>(
    staged: &StagedContract<'a, R, St>,
    dst: &DynamicFusionMapSpace,
) -> Result<(), OperationError>
where
    St: ContractStageParts<'a, R>,
{
    if St::M::contract_destination_matches(&staged.stage.parts().stage, dst) {
        return Ok(());
    }
    Err(OperationError::SpaceMismatch {
        message: "destination fusion space or block layout does not match the contraction result",
    })
}

/// The output axes of a contraction request, as the planner reads them.
fn staged_contract_axes<'a>(request: ContractRequest<'a>, rank: usize) -> Option<PlannedAxes<'a>> {
    match request {
        ContractRequest::Contract { axes, .. } | ContractRequest::Planar { axes, .. } => Some((
            axes.lhs_contracting_axes(),
            axes.rhs_contracting_axes(),
            spec_output_axes(axes, rank),
        )),
        ContractRequest::Compose => None,
    }
}

/// The lock-free first half of the planner ([`plan_contract_in`]'s core, or
/// the whole [`plan_compose_in`]) for an executor with capabilities `X`.
pub(crate) fn staged_try_core<'a, X: ExecCaps, R, St>(
    staged: &StagedContract<'a, R, St>,
) -> Result<CoreRoute<ContractStageScalar<'a, R, St>>, ContractStageError<'a, R, St>>
where
    St: ContractStageParts<'a, R>,
    ContractStageScalar<'a, R, St>: DenseBlockScalar,
{
    let parts = staged.stage.parts();
    let (rule, space) = St::M::contract_destination(&parts.stage, staged.lhs);
    let target = PlanTarget {
        rule,
        space,
        authority: parts.authority,
    };
    let (lhs, rhs) = staged.operands;
    if let ContractRequest::Planar { orders, .. } = staged.request {
        return try_planar_core_route::<X, St::M, R>(target, lhs, rhs, orders);
    }
    let Some((lhs_axes, rhs_axes, output_axes)) =
        staged_contract_axes(staged.request, space.rank())
    else {
        return plan_compose_in::<X, St::M, R>(target, lhs, rhs).map(CoreRoute::Hit);
    };
    let axes = TensorContractSpec::new_with_conjugation(
        lhs_axes,
        rhs_axes,
        OutputAxisOrder::from_axes(&output_axes),
        lhs.storage_conjugate(),
        rhs.storage_conjugate(),
    );
    try_compile_core_route::<X, St::M, R>(target, lhs, rhs, axes, ContractKind::Contract)
}

/// The rest of [`plan_contract_in`] after `staged_try_core` missed: `copyC`,
/// then the `DynamicTree` artifact, under the context's planning state or
/// the call's transaction.
pub(crate) fn staged_plan_beyond_core<'a, X: ExecCaps, R, St>(
    staged: &mut StagedContract<'a, R, St>,
    miss: CoreMiss,
    planning: &mut TreeTransformPlanning,
) -> Result<StorageContractResolution<ContractStageScalar<'a, R, St>>, ContractStageError<'a, R, St>>
where
    St: ContractStageParts<'a, R>,
    ContractStageScalar<'a, R, St>: DenseBlockScalar,
{
    let StagedContract {
        lhs: lhs_authority,
        operands: (lhs, rhs),
        request,
        stage,
    } = staged;
    let StagedContraction {
        stage,
        authority,
        txn,
        ..
    } = stage.parts_mut();
    let (rule, space) = St::M::contract_destination(stage, lhs_authority);
    if let ContractRequest::Planar { axes, orders, .. } = *request {
        let plan =
            FusionContractPlan::planar(orders, lhs.storage_conjugate(), rhs.storage_conjugate());
        return compile_dynamic_tree_in::<St::M, R>(
            St::M::contract_cache(planning, txn),
            PlanTarget {
                rule,
                space,
                authority: *authority,
            },
            *lhs,
            *rhs,
            axes,
            X::IRREGULAR_CORE,
            Some(&plan),
        );
    }
    let Some((lhs_axes, rhs_axes, output_axes)) = staged_contract_axes(*request, space.rank())
    else {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "a composition plans only its core",
        }
        .into());
    };
    plan_contract_beyond_core_in::<X, St::M, R>(
        St::M::contract_cache(planning, txn),
        miss,
        PlanTarget {
            rule,
            space,
            authority: *authority,
        },
        *lhs,
        *rhs,
        (lhs_axes, rhs_axes, &output_axes),
    )
}

/// Commits the staged destination after the replay of `resolution`, then
/// publishes what the call staged for it.
pub(crate) fn staged_commit<'a, R, St>(
    staged: StagedContract<'a, R, St>,
    resolution: &StorageContractResolution<ContractStageScalar<'a, R, St>>,
) -> Result<BoundDynamicFusionMapSpace<R>, ContractStageError<'a, R, St>>
where
    St: ContractStageParts<'a, R>,
{
    let lhs = staged.lhs;
    let parts = staged.stage.into_parts();
    St::M::commit_contraction(lhs, parts.stage, parts.txn, resolution)
}

impl<'a, R> StagedContract<'a, R, MultiplicityFreeContractStage<R>>
where
    R: MultiplicityFreeRigidSymbols + CheckedFusionAlgebra + TreeTransformRuleCacheKey + 'a,
    R::Scalar: DenseBlockScalar,
{
    /// Derives the destination of a multiplicity-free `request`: two direct
    /// operands through the owned derivation, a lazy adjoint through the
    /// oriented one. Each operand is its logical space, its storage operand
    /// (a lazy adjoint is storage-conjugate) and its stored length.
    #[allow(clippy::type_complexity)]
    pub fn multiplicity_free(
        lhs: (&'a BoundDynamicFusionMapSpace<R>, FusionOperand<'a>, usize),
        rhs: (&'a BoundDynamicFusionMapSpace<R>, FusionOperand<'a>, usize),
        request: ContractRequest<'a>,
    ) -> Result<Self, OperationError> {
        stage_contract(lhs, rhs, request)
    }

    /// [`MultiplicityFreeContractStage`]'s `spacecheck_contract`.
    pub fn require_destination(&self, dst: &DynamicFusionMapSpace) -> Result<(), OperationError> {
        staged_require_destination(self, dst)
    }

    /// The staged destination space the route writes.
    pub fn destination(&self) -> &DynamicFusionMapSpace {
        staged_destination(self)
    }

    /// The staged destination's payload length.
    pub fn destination_len(&self) -> usize {
        staged_len(self)
    }

    /// The canonical core route, resolved from the operands alone without
    /// any context: a [`CoreRoute::Miss`] continues with
    /// [`TensorContractFusionExecutionContext::plan_staged_contract_beyond_core`].
    pub fn try_core<X: ExecCaps>(&self) -> Result<CoreRoute<R::Scalar>, OperationError> {
        staged_try_core::<X, R, MultiplicityFreeContractStage<R>>(self)
    }

    /// The derived destination, after the replay of `resolution`.
    pub fn commit(
        self,
        resolution: &StorageContractResolution<R::Scalar>,
    ) -> Result<BoundDynamicFusionMapSpace<R>, OperationError> {
        staged_commit(self, resolution)
    }
}

/// The request checks both halves of the contraction route run: the provider
/// against the three spaces, and the operand conjugation flags against the
/// request.
fn validate_raw_contract_request<M, R>(
    target: PlanTarget<'_, R, M::SpaceAuthority<'_>>,
    lhs: FusionOperand<'_>,
    rhs: FusionOperand<'_>,
    axes: TensorContractSpec<'_>,
) -> Result<(), M::Error>
where
    M: PlanningAlgebra<R>,
{
    M::validate_spaces(
        target.rule,
        target.authority,
        target.space,
        lhs.storage_space(),
        rhs.storage_space(),
    )?;
    if axes.lhs_conjugate() != lhs.storage_conjugate()
        || axes.rhs_conjugate() != rhs.storage_conjugate()
    {
        return Err(OperationError::InvalidArgument {
            message: "prelowered operand flags must match the contraction request",
        }
        .into());
    }
    Ok(())
}

/// First half of the contraction route: validation and the canonical core
/// over the parent buffers — lazy adjoints as GEMM operand flags and a
/// uniform fermionic twist folded into per-job alpha. A [`CoreRoute::Miss`]
/// is the input of the rest of the planner
/// ([`TensorContractFusionExecutionContext::plan_contract_beyond_core`]).
///
/// Free function, not a context method: this half resolves entirely from
/// the operands, so the canonical device contraction takes no Runtime
/// context lock.
#[doc(hidden)]
pub fn try_compile_storage_contract_core_route<X: ExecCaps, R>(
    dst_space: &BoundDynamicFusionMapSpace<R>,
    lhs: FusionOperand<'_>,
    rhs: FusionOperand<'_>,
    axes: TensorContractSpec<'_>,
) -> Result<CoreRoute<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
{
    try_compile_core_route::<X, MultiplicityFreeAdmissionMode, R>(
        PlanTarget::bound(dst_space),
        lhs,
        rhs,
        axes,
        ContractKind::Contract,
    )
}

/// The planner for a composition, TensorKit `mul!(C, A, B)`
/// (`linalg.jl:330-370` @cfaa073): A's domain against B's codomain in
/// order, the identity output, no twist and no candidate walk, resolved to
/// `Core { swapped: false }` with unit job coefficients, or the packed core
/// of a non-canonical tiling for an executor with
/// [`ExecCaps::IRREGULAR_CORE`]. Any other composition is
/// `UnsupportedTensorContractScope`: `mul!` has no transform route, so
/// none is planned.
///
/// The braiding check stays with the caller: `mul!` admits every braiding
/// style, since composition has no braid-dependent step.
///
/// Free function, not a context method: the core resolves entirely from
/// the operands, so planning a composition takes no Runtime context lock.
#[doc(hidden)]
pub fn plan_compose<X: ExecCaps, R>(
    dst_space: &BoundDynamicFusionMapSpace<R>,
    lhs: FusionOperand<'_>,
    rhs: FusionOperand<'_>,
) -> Result<StorageContractResolution<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
{
    plan_compose_in::<X, MultiplicityFreeAdmissionMode, R>(PlanTarget::bound(dst_space), lhs, rhs)
}

/// [`plan_compose`] in either admission mode `M`.
pub(crate) fn plan_compose_in<X: ExecCaps, M, R>(
    target: PlanTarget<'_, R, M::SpaceAuthority<'_>>,
    lhs: FusionOperand<'_>,
    rhs: FusionOperand<'_>,
) -> Result<StorageContractResolution<M::Scalar>, M::Error>
where
    M: PlanningAlgebra<R>,
    M::Scalar: DenseBlockScalar,
{
    match compose_core_route::<X, M, R>(target, lhs, rhs)? {
        CoreRoute::Hit(resolution) => Ok(resolution),
        CoreRoute::Miss(_) => Err(OperationError::UnsupportedTensorContractScope {
            message:
                "storage-direct composition supports only canonical fully-direct oriented operands",
        }
        .into()),
    }
}

/// The canonical core of `mul!(C, lhs, rhs)`: A's domain against B's
/// codomain in order, the identity output, no twist.
fn compose_core_route<X: ExecCaps, M, R>(
    target: PlanTarget<'_, R, M::SpaceAuthority<'_>>,
    lhs: FusionOperand<'_>,
    rhs: FusionOperand<'_>,
) -> Result<CoreRoute<M::Scalar>, M::Error>
where
    M: PlanningAlgebra<R>,
    M::Scalar: DenseBlockScalar,
{
    let (left, right) = (lhs.oriented_homspace(), rhs.oriented_homspace());
    let lhs_axes: Vec<usize> = (left.nout()..left.rank()).collect();
    let rhs_axes: Vec<usize> = (0..right.nout()).collect();
    let axes = TensorContractSpec::new_with_conjugation(
        &lhs_axes,
        &rhs_axes,
        symtn_operations::OutputAxisOrder::identity(),
        lhs.storage_conjugate(),
        rhs.storage_conjugate(),
    );
    try_compile_core_route::<X, M, R>(target, lhs, rhs, axes, ContractKind::Compose)
}

/// The first rung of a planar contraction, TensorKit `planarcontract!`: when
/// no operand needs a transpose and the output is the core's, the
/// composition core (`mul!` on both operands as stored); any other planar
/// spec goes to the `DynamicTree` artifact of its fixed plan.
fn try_planar_core_route<X: ExecCaps, M, R>(
    target: PlanTarget<'_, R, M::SpaceAuthority<'_>>,
    lhs: FusionOperand<'_>,
    rhs: FusionOperand<'_>,
    orders: &PlanarContractOrders,
) -> Result<CoreRoute<M::Scalar>, M::Error>
where
    M: PlanningAlgebra<R>,
    M::Scalar: DenseBlockScalar,
{
    let shape = |operand: FusionOperand<'_>| {
        let homspace = operand.oriented_homspace();
        (homspace.nout(), homspace.rank())
    };
    if !orders.is_composition(shape(lhs), shape(rhs)) {
        return Ok(CoreRoute::Miss(CoreMiss {
            requested_zero_copy: false,
        }));
    }
    compose_core_route::<X, M, R>(target, lhs, rhs)
}

/// [`TensorContractFusionExecutionContext::plan_contract`] in admission mode
/// `M`, deriving spaces and transformers under `cache` (the context's
/// planning state, or a checked call's transaction).
pub(crate) fn plan_contract_in<X: ExecCaps, M, R>(
    cache: &mut M::StructureCache,
    target: PlanTarget<'_, R, M::SpaceAuthority<'_>>,
    lhs: FusionOperand<'_>,
    rhs: FusionOperand<'_>,
    (lhs_axes, rhs_axes, output_axes): (&[usize], &[usize], &[usize]),
) -> Result<StorageContractResolution<M::Scalar>, M::Error>
where
    M: PlanningAlgebra<R>,
    M::Scalar: DenseBlockScalar,
{
    let axes = TensorContractSpec::new_with_conjugation(
        lhs_axes,
        rhs_axes,
        symtn_operations::OutputAxisOrder::from_axes(output_axes),
        lhs.storage_conjugate(),
        rhs.storage_conjugate(),
    );
    let core = try_compile_core_route::<X, M, R>(target, lhs, rhs, axes, ContractKind::Contract)?;
    match core {
        CoreRoute::Hit(core) => Ok(core),
        CoreRoute::Miss(miss) => plan_contract_beyond_core_in::<X, M, R>(
            cache,
            miss,
            target,
            lhs,
            rhs,
            (lhs_axes, rhs_axes, output_axes),
        ),
    }
}

fn plan_contract_beyond_core_in<X: ExecCaps, M, R>(
    cache: &mut M::StructureCache,
    miss: CoreMiss,
    target: PlanTarget<'_, R, M::SpaceAuthority<'_>>,
    lhs: FusionOperand<'_>,
    rhs: FusionOperand<'_>,
    (lhs_axes, rhs_axes, output_axes): (&[usize], &[usize], &[usize]),
) -> Result<StorageContractResolution<M::Scalar>, M::Error>
where
    M: PlanningAlgebra<R>,
    M::Scalar: DenseBlockScalar,
{
    let order = super::resolution::copy_c_order(
        || M::contract_twist_possible(target.rule, target.authority),
        |homspace, axes| {
            super::resolution::contract_axes_require_twist::<M, R>(
                target.rule,
                target.authority,
                homspace,
                axes,
            )
        },
        &|space| M::storage_layout_is_canonical(target.rule, space),
        target.space,
        lhs,
        rhs,
        lhs_axes,
        rhs_axes,
        output_axes,
        Some(miss.requested_zero_copy),
        true,
    );
    if let Some(orientation) = order {
        if let Some(copy) = plan_copy_c_in::<X, M, R>(
            cache,
            target,
            lhs,
            rhs,
            (lhs_axes, rhs_axes, output_axes),
            orientation,
        )? {
            return Ok(copy);
        }
    }
    let axes = TensorContractSpec::new_with_conjugation(
        lhs_axes,
        rhs_axes,
        symtn_operations::OutputAxisOrder::from_axes(output_axes),
        lhs.storage_conjugate(),
        rhs.storage_conjugate(),
    );
    compile_dynamic_tree_in::<M, R>(cache, target, lhs, rhs, axes, X::IRREGULAR_CORE, None)
}

/// The `CopyC` route in `orientation`, or `None` when its temporary has
/// no core the executor runs (the `DynamicTree` artifact then applies).
fn plan_copy_c_in<X: ExecCaps, M, R>(
    cache: &mut M::StructureCache,
    target: PlanTarget<'_, R, M::SpaceAuthority<'_>>,
    lhs: FusionOperand<'_>,
    rhs: FusionOperand<'_>,
    (lhs_axes, rhs_axes, output_axes): (&[usize], &[usize], &[usize]),
    orientation: super::fusion::FusionContractOrientation,
) -> Result<Option<StorageContractResolution<M::Scalar>>, M::Error>
where
    M: PlanningAlgebra<R>,
    M::Scalar: DenseBlockScalar,
{
    let swapped = orientation == super::fusion::FusionContractOrientation::RhsLhs;
    let ((first, first_axes), (second, second_axes)) = if swapped {
        ((rhs, rhs_axes), (lhs, lhs_axes))
    } else {
        ((lhs, lhs_axes), (rhs, rhs_axes))
    };
    let identity = symtn_operations::OutputAxisOrder::identity();
    // The temporary is `first·second`'s own default-order result over the
    // oriented HomSpaces, as any contraction result is, derived under the
    // destination's authority (the request was validated against its
    // provider).
    let first_open = first.storage_space().rank() - first_axes.len();
    let default_axes: smallvec::SmallVec<[usize; 16]> =
        (0..first_open + second.storage_space().rank() - second_axes.len()).collect();
    let temporary = M::copy_c_temporary(
        cache,
        target.authority,
        target.rule,
        first,
        second,
        first_axes,
        second_axes,
        &default_axes,
        first_open,
    )?;
    let temporary_target = PlanTarget {
        rule: target.rule,
        space: &temporary,
        authority: target.authority,
    };
    let core = try_compile_core_route::<X, M, R>(
        temporary_target,
        first,
        second,
        TensorContractSpec::new_with_conjugation(
            first_axes,
            second_axes,
            identity,
            first.storage_conjugate(),
            second.storage_conjugate(),
        ),
        ContractKind::Contract,
    )?;
    let CoreRoute::Hit(core) = core else {
        return Ok(None);
    };
    let Some((core, core_swapped)) = core.direct_core() else {
        return Ok(None);
    };
    let operation = super::resolution::copy_c_output_transform(
        orientation,
        lhs.storage_space().rank(),
        rhs.storage_space().rank(),
        lhs_axes,
        rhs_axes,
        output_axes,
        target.space.nout(),
    );
    let temporary_structure = Arc::clone(temporary.structure());
    let transform = M::tree_structure(
        cache,
        target.rule,
        &operation,
        target.space.structure(),
        TreeStructureSource::Stored {
            structure: &temporary_structure,
            storage_conjugate: false,
        },
    )?;
    Ok(Some(StorageContractResolution::new(ContractRoute::CopyC(
        CopyCRoute {
            core: Arc::clone(core),
            swapped: core_swapped ^ swapped,
            temporary_len: temporary.required_len().map_err(OperationError::from)?,
            temporary: temporary_structure,
            transform,
        },
    ))))
}

/// The last rung of the planner: the Host `DynamicTree` artifact for a
/// contraction the core declined (see
/// [`TensorContractFusionExecutionContext::compile_storage_contract_dynamic_tree`]).
///
/// `irregular_core`: the executor packs and scatters a core plan that is
/// not fully direct ([`ExecCaps::IRREGULAR_CORE`]), as the artifact's core
/// over a non-canonical destination tiling (#1517) is. `plan` is the fixed
/// plan of a planar contraction; `None` selects the contraction's.
fn compile_dynamic_tree_in<M, R>(
    cache: &mut M::StructureCache,
    target: PlanTarget<'_, R, M::SpaceAuthority<'_>>,
    lhs: FusionOperand<'_>,
    rhs: FusionOperand<'_>,
    axes: TensorContractSpec<'_>,
    irregular_core: bool,
    plan: Option<&FusionContractPlan>,
) -> Result<StorageContractResolution<M::Scalar>, M::Error>
where
    M: PlanningAlgebra<R>,
    M::Scalar: DenseBlockScalar,
{
    // Re-run here, not only in the first half: this entry is callable on
    // its own, and the artifact compilers assume a validated request.
    validate_raw_contract_request::<M, R>(target, lhs, rhs, axes)?;
    let artifact = if !lhs.storage_conjugate() && !rhs.storage_conjugate() {
        let (lhs_space, rhs_space) = (lhs.storage_space(), rhs.storage_space());
        compile_stored_dynamic_tree::<M, R>(
            cache,
            target,
            (lhs_space, lhs_space.structure()),
            (rhs_space, rhs_space.structure()),
            axes,
            plan,
        )?
    } else {
        M::prelowered_dynamic_tree_artifact(cache, target, lhs, rhs, axes, plan)?
    };
    // Transformed sources are canonical coupled layouts, so only a
    // non-canonical destination tiling leaves the core plan not fully
    // direct; a direct-core executor reports it before any device work.
    if !irregular_core && !artifact.block_plan_is_fully_direct() {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "dynamic-tree core plan over transformed sources is not fully direct",
        }
        .into());
    }
    Ok(StorageContractResolution::new(ContractRoute::DynamicTree(
        Arc::new(artifact),
    )))
}

/// The `DynamicTree` artifact over two sources given as the logical space the
/// plan is selected on and the storage structure replay reads: a direct
/// operand's own space and structure, or (checked Generic) a lazy adjoint's
/// logical space over its parent's storage. `plan` fixes the plan instead of
/// selecting one for `axes`.
pub(crate) fn compile_stored_dynamic_tree<M, R>(
    cache: &mut M::StructureCache,
    target: PlanTarget<'_, R, M::SpaceAuthority<'_>>,
    (lhs_space, lhs_structure): (&DynamicFusionMapSpace, &Arc<BlockStructure>),
    (rhs_space, rhs_structure): (&DynamicFusionMapSpace, &Arc<BlockStructure>),
    axes: TensorContractSpec<'_>,
    plan: Option<&FusionContractPlan>,
) -> Result<super::dynamic::DynamicTreeExecutionArtifact<M::Scalar>, M::Error>
where
    M: PlanningAlgebra<R>,
    M::Scalar: DenseBlockScalar,
{
    let PlanTarget {
        rule,
        space: dst_space,
        authority,
    } = target;
    let selected;
    let plan = match plan {
        Some(plan) => plan,
        None => {
            selected =
                M::dynamic_tree_plan(rule, authority, dst_space, lhs_space, rhs_space, axes)?;
            &selected
        }
    };
    super::dynamic::compile_dynamic_tree_execution_artifact::<M, R>(
        cache,
        rule,
        authority,
        plan,
        dst_space,
        lhs_space,
        lhs_structure,
        rhs_space,
        rhs_structure,
    )
}

/// The multiplicity-free `DynamicTree` artifact of a contraction with a
/// lazy-adjoint operand: both operands prelowered to logical-key layouts
/// under the destination's primer; `plan` as in [`compile_stored_dynamic_tree`].
pub(crate) fn compile_prelowered_dynamic_tree<R>(
    planning: &mut TreeTransformPlanning,
    target: PlanTarget<'_, R>,
    lhs: FusionOperand<'_>,
    rhs: FusionOperand<'_>,
    axes: TensorContractSpec<'_>,
    plan: Option<&FusionContractPlan>,
) -> Result<super::dynamic::DynamicTreeExecutionArtifact<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols + TreeTransformRuleCacheKey,
    R::Scalar: DenseBlockScalar + MultiplicityFreePlanningScalar,
{
    let PlanTarget {
        rule,
        space: dst_space,
        authority: layout_primer,
    } = target;
    let lhs_layout = lhs.prepare(rule, layout_primer)?;
    let rhs_layout = rhs.prepare(rule, layout_primer)?;
    let selected;
    let plan = match plan {
        Some(plan) => plan,
        None => {
            selected = prepare_tensorcontract_fusion_plan_dyn_prelowered_canonical(
                rule,
                dst_space,
                &lhs_layout,
                &rhs_layout,
                axes,
                layout_primer,
            )?;
            &selected
        }
    };
    super::dynamic::compile_prelowered_dynamic_tree_execution_artifact(
        planning,
        rule,
        layout_primer,
        plan,
        dst_space,
        &lhs_layout,
        &rhs_layout,
    )
}

fn try_compile_core_route<X: ExecCaps, M, R>(
    target: PlanTarget<'_, R, M::SpaceAuthority<'_>>,
    lhs: FusionOperand<'_>,
    rhs: FusionOperand<'_>,
    axes: TensorContractSpec<'_>,
    kind: ContractKind,
) -> Result<CoreRoute<M::Scalar>, M::Error>
where
    M: PlanningAlgebra<R>,
    M::Scalar: DenseBlockScalar,
{
    validate_raw_contract_request::<M, R>(target, lhs, rhs, axes)?;
    let irregular = X::IRREGULAR_CORE;
    // A twist that is not uniform within one coupled-sector matrix has no
    // per-job alpha; the DynamicTree artifact applies it per block instead.
    let mut requested_zero_copy = false;
    let route = match kind {
        ContractKind::Contract => try_compile_oriented_storage_contract_candidate_plan::<M, R>(
            target,
            lhs,
            rhs,
            axes,
            &mut requested_zero_copy,
            irregular,
        )?,
        ContractKind::Compose => try_compile_oriented_storage_contract_plan::<M, R>(
            target, lhs, rhs, axes, kind, irregular,
        )?
        .filter(|plan| irregular || plan.is_fully_direct())
        .map(|plan| ContractRoute::Core {
            plan,
            swapped: false,
        }),
    };
    Ok(match route {
        Some(route) => CoreRoute::Hit(StorageContractResolution::new(route)),
        None => CoreRoute::Miss(CoreMiss {
            requested_zero_copy,
        }),
    })
}

#[cfg(test)]
/// Canonical storage contraction over parent buffers with lazy operand
/// orientation. A miss is unsupported; this device leaf never prepares
/// logical-key projections or source transforms.
///
/// Free function, not a context method: the storage-direct route resolves and
/// replays entirely from the operands, so a device operation calling it needs
/// no execution context and therefore no Runtime state lock.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn tensorcontract_fusion_dyn_prelowered_direct_on_storage<R, G, D, DDst, DLhs, DRhs>(
    gemm: &mut G,
    dst_space: &BoundDynamicFusionMapSpace<R>,
    dst: &mut DDst,
    lhs: FusionOperand<'_>,
    lhs_storage: &DLhs,
    rhs: FusionOperand<'_>,
    rhs_storage: &DRhs,
    axes: TensorContractSpec<'_>,
) -> Result<(), OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: DenseBlockScalar,
    D: DenseBlockScalar + RecouplingCoefficientAction<R::Scalar>,
    G: symtn_operations::fusion_replay::StorageGemm<D, DDst, DLhs, DRhs>,
    DDst: symtn_core::TensorStorage<D>,
    DLhs: symtn_core::TensorStorage<D>,
    DRhs: symtn_core::TensorStorage<D>,
{
    let target = PlanTarget::bound(dst_space);
    validate_raw_contract_request::<MultiplicityFreeAdmissionMode, R>(target, lhs, rhs, axes)?;
    let plan = try_compile_oriented_storage_contract_plan::<MultiplicityFreeAdmissionMode, R>(
        target,
        lhs,
        rhs,
        axes,
        ContractKind::Contract,
        false,
    )?
    .filter(|plan| plan.is_fully_direct())
    .ok_or_else(|| OperationError::UnsupportedTensorContractScope {
        message:
            "storage-direct contraction supports only canonical fully-direct oriented operands",
    })?;
    plan.execute_direct_on_storage_prezeroed(gemm, dst, lhs_storage, rhs_storage)
}
