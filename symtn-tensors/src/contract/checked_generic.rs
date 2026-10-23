use std::sync::Arc;

use num_traits::Zero;
use symtn_core::{
    BlockStructure, BraidingStyleKind, CheckedGenericAdmissionMode, CheckedGenericRigidSymbols,
    CoreError, FusionTreeHomSpace, PermutedMemoTicket, RuleIdentity,
    StructurallyValidatedFusionTreeSubset,
};
use symtn_operations::{OutputAxisOrder, TensorContractSpec, TreeTransformBackend};

use crate::mode::{ContractRequest, ContractSide, ContractStaging, StagedContraction};
use crate::tree_transform::{CheckedGenericPlanError, CheckedPendingCoefficients};
use crate::validate_oriented_fusion_layout;
use crate::{
    ConjugateValue, DenseRecouplingScalar, OperationError, RecouplingCoefficientAction, ZeroBytes,
};

use super::backend::TensorContractBackend;
use super::context::TensorContractFusionExecutionContext;
#[cfg(test)]
use super::context::{plan_contract_in, PlanTarget};
use super::context::{
    stage_contract, staged_commit, staged_destination, staged_len, staged_plan_beyond_core,
    staged_require_destination, staged_try_core, ContractStageParts, StageParts, StagedContract,
};
use super::dynamic_space::{
    BoundDynamicFusionMapSpace, DynamicFusionMapSpace, FusionOperand,
    PreparedCheckedGenericDynamicSpace,
};
#[cfg(test)]
use super::resolution::HostEagerExecutor;
use super::resolution::StorageContractResolution;
use super::resolution::{CoreMiss, CoreRoute, ExecCaps};
use super::structure::TensorContractAxisPlan;
use crate::tree_transform::TreeTransformPlanning;

type CheckedContractResult<P, D> = Result<
    (BoundDynamicFusionMapSpace<P>, Vec<D>),
    CheckedGenericPlanError<<P as symtn_core::CheckedGenericFusion>::Error>,
>;

/// The braiding boundary of general checked contraction.
pub(crate) const CHECKED_CONTRACTION_REQUIRES_BOSONIC: OperationError =
    OperationError::UnsupportedTensorContractScope {
        message: "checked Generic contraction requires Bosonic braiding",
    };

/// What the checked planner derives spaces under (its
/// `PlanningAlgebra::SpaceAuthority`): the left binding, whose provider
/// admission stages every derived space, the entry's answer to whether a
/// contraction twist exists, and each operand's logical space.
///
/// Why the answer and the logical spaces travel here rather than being
/// derived where the planner asks: a provider read inside the planner would
/// follow the destination's staging (#2046). A lazy adjoint's canonical
/// logical layout is the space its view already holds; re-enumerating it
/// needs the provider, and the multiplicity-free layout primer that derives
/// it there has no checked counterpart.
pub(crate) struct CheckedAuthority<'a, P> {
    pub(crate) binding: &'a BoundDynamicFusionMapSpace<P>,
    twist_free: bool,
    sides: [CheckedSide<'a>; 2],
}

/// One operand of a checked request: its storage operand and logical space.
pub(crate) type CheckedSide<'a> = (FusionOperand<'a>, &'a DynamicFusionMapSpace);

// Why manual: a derive would bound `P: Copy`.
impl<P> Clone for CheckedAuthority<'_, P> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<P> Copy for CheckedAuthority<'_, P> {}

impl<'a, P> CheckedAuthority<'a, P>
where
    P: CheckedGenericRigidSymbols<Scalar = f64>,
{
    /// Canonical composition (TensorKit `mul!`) crosses no legs, so it never
    /// asks the twist question and the authority holds no answer: asked, it
    /// fails closed.
    pub(crate) fn composition(
        binding: &'a BoundDynamicFusionMapSpace<P>,
        sides: [CheckedSide<'a>; 2],
    ) -> Self {
        Self {
            binding,
            twist_free: false,
            sides,
        }
    }

    /// General contraction may braid and twist; the checked engine
    /// implements neither TensorKit `blas_contract!`'s fermionic supertrace
    /// branch nor non-symmetric braiding, so it admits Bosonic braiding only,
    /// read once from the left provider.
    pub(crate) fn contraction(
        binding: &'a BoundDynamicFusionMapSpace<P>,
        sides: [CheckedSide<'a>; 2],
    ) -> Result<Self, CheckedGenericPlanError<P::Error>> {
        if binding.provider().braiding_style() != BraidingStyleKind::Bosonic {
            return Err(CHECKED_CONTRACTION_REQUIRES_BOSONIC.into());
        }
        Ok(Self {
            binding,
            twist_free: true,
            sides,
        })
    }

    /// Whether the entry established that no contraction twist exists.
    pub(crate) fn twist_free(self) -> bool {
        self.twist_free
    }

    /// The logical space of a request operand. Both operands may be one
    /// tensor; storage and orientation then name the same space.
    pub(crate) fn logical_of(
        self,
        operand: FusionOperand<'_>,
    ) -> Result<&'a DynamicFusionMapSpace, CheckedGenericPlanError<P::Error>> {
        self.sides
            .iter()
            .find(|(side, _)| {
                std::ptr::eq(side.storage_space(), operand.storage_space())
                    && side.storage_conjugate() == operand.storage_conjugate()
            })
            .map(|&(_, logical)| logical)
            .ok_or_else(|| unknown_operand().into())
    }

    /// The storage operand of a request operand's logical space.
    pub(crate) fn operand_of(
        self,
        logical: &DynamicFusionMapSpace,
    ) -> Result<FusionOperand<'a>, CheckedGenericPlanError<P::Error>> {
        self.sides
            .iter()
            .find(|&&(_, side)| std::ptr::eq(side, logical))
            .map(|&(operand, _)| operand)
            .ok_or_else(|| unknown_operand().into())
    }
}

fn unknown_operand() -> OperationError {
    OperationError::StructureMismatch {
        tensor: "checked contraction operand",
    }
}

#[cfg(test)]
impl<'a, P> PlanTarget<'a, P, CheckedAuthority<'a, P>> {
    /// The checked destination `space` (a staged preview) planned under
    /// `authority`, whose binding's provider is the rule.
    pub(crate) fn checked(
        authority: CheckedAuthority<'a, P>,
        space: &'a DynamicFusionMapSpace,
    ) -> Self {
        Self {
            rule: authority.binding.provider(),
            space,
            authority,
        }
    }
}

/// One checked contraction's transaction (#2063): its staged intermediate
/// spaces and its staged coefficients and transformers. Nothing staged is
/// published before [`Self::commit`]; dropping it publishes nothing.
pub(crate) struct CheckedContractTxn {
    coefficients: CheckedPendingCoefficients,
    // Why inline three: a contraction stages at most its two sources and its
    // core destination, or one copyC temporary. A derived source permute
    // carries its destination-memo publication (#2157).
    staged: smallvec::SmallVec<[StagedSpace; 3]>,
}

type StagedSpace = (
    PreparedCheckedGenericDynamicSpace,
    Arc<BlockStructure>,
    Option<PermutedMemoTicket>,
);

impl CheckedContractTxn {
    /// Captures the reset epoch: create it before the call's first build.
    pub(crate) fn new() -> Self {
        Self {
            coefficients: CheckedPendingCoefficients::new(),
            staged: smallvec::SmallVec::new(),
        }
    }

    pub(crate) fn coefficients(&mut self) -> &mut CheckedPendingCoefficients {
        &mut self.coefficients
    }

    /// Stages one derived space, with the destination-memo publication of a
    /// derived source permute, and returns the preview the planner reads.
    pub(crate) fn stage(
        &mut self,
        prepared: PreparedCheckedGenericDynamicSpace,
        memo: Option<PermutedMemoTicket>,
    ) -> DynamicFusionMapSpace {
        let preview = prepared.preview();
        self.staged
            .push((prepared, Arc::clone(preview.structure()), memo));
        preview
    }

    /// Publishes after the fallible destination commit (`destination`:
    /// its preview and committed structures): commits only the staged
    /// intermediates `resolution` replays over, in staging order, then
    /// flushes coefficients and transformers keyed by committed ids, then
    /// the committed source permutes' destination memos (#2157). A stage
    /// the planner declined is dropped unpublished.
    pub(crate) fn commit(
        self,
        destination: (Arc<BlockStructure>, Arc<BlockStructure>),
        resolution: &StorageContractResolution<f64>,
    ) {
        let used = resolution.derived_structures();
        let mut committed: smallvec::SmallVec<[_; 4]> = smallvec::smallvec![destination];
        let mut memos: smallvec::SmallVec<[_; 2]> = smallvec::SmallVec::new();
        for (prepared, preview, memo) in self.staged {
            if used
                .iter()
                .any(|structure| Arc::ptr_eq(structure, &preview))
            {
                let (homspace, structure) = prepared.commit_structure();
                if let (Some(memo), Some(homspace)) = (memo, homspace) {
                    memos.push((memo, homspace, Arc::clone(&structure)));
                }
                committed.push((preview, structure));
            }
        }
        self.coefficients.flush_committed(&committed);
        for (memo, homspace, structure) in memos {
            memo.publish(&homspace, &structure);
        }
    }

    /// Publishes after an eager transform's destination commit
    /// (`destination`: its preview and committed structures); a transform
    /// stages no intermediate.
    pub(crate) fn commit_transform(self, destination: (Arc<BlockStructure>, Arc<BlockStructure>)) {
        self.coefficients.flush_committed(&[destination]);
    }
}

fn validate_source_structure<E>(
    space: &DynamicFusionMapSpace,
) -> Result<(), CheckedGenericPlanError<E>> {
    StructurallyValidatedFusionTreeSubset::try_new(space.homspace(), space.structure())?;
    Ok(())
}

/// That `logical` is the logical space of `operand`: the storage space
/// itself, or for a lazy adjoint the swapped HomSpace whose every block maps
/// onto a parent block of the transposed shape. Provider-free.
pub(crate) fn validate_checked_operand_relation(
    logical: &DynamicFusionMapSpace,
    operand: FusionOperand<'_>,
) -> Result<(), OperationError> {
    let storage = operand.storage_space();
    if !operand.storage_conjugate() {
        return if std::ptr::eq(logical, storage) {
            Ok(())
        } else {
            Err(OperationError::StructureMismatch {
                tensor: "checked direct operand",
            })
        };
    }
    if logical.nout() != storage.nin()
        || logical.nin() != storage.nout()
        || logical.homspace().codomain() != storage.homspace().domain()
        || logical.homspace().domain() != storage.homspace().codomain()
    {
        return Err(OperationError::StructureMismatch {
            tensor: "checked adjoint relation",
        });
    }
    validate_oriented_fusion_layout(logical.structure(), operand)
}

fn validate_contract_local<P>(
    (lhs, lhs_operand, lhs_len): ContractSide<'_, P>,
    (rhs, rhs_operand, rhs_len): ContractSide<'_, P>,
    axes: TensorContractSpec<'_>,
    dst_nout: usize,
) -> Result<TensorContractAxisPlan, CheckedGenericPlanError<P::Error>>
where
    P: CheckedGenericRigidSymbols<Scalar = f64>,
{
    let output_rank = lhs
        .space()
        .rank()
        .checked_sub(axes.lhs_contracting_axes().len())
        .and_then(|rank| {
            rhs.space()
                .rank()
                .checked_sub(axes.rhs_contracting_axes().len())
                .and_then(|rhs| rank.checked_add(rhs))
        })
        .ok_or(OperationError::ElementCountOverflow)?;
    let axis_plan =
        TensorContractAxisPlan::compile(lhs.space().rank(), rhs.space().rank(), output_rank, axes)?;
    if dst_nout > output_rank {
        return Err(CoreError::StructureRankMismatch {
            expected: output_rank,
            actual: dst_nout,
        }
        .into());
    }
    for (len, logical, operand) in [(lhs_len, lhs, lhs_operand), (rhs_len, rhs, rhs_operand)] {
        let storage = operand.storage_space();
        let expected = storage.required_len()?;
        if len != expected {
            return Err(OperationError::ElementCountMismatch {
                expected,
                actual: len,
            }
            .into());
        }
        validate_source_structure::<P::Error>(storage)?;
        validate_checked_operand_relation(logical.space(), operand)?;
    }
    Ok(axis_plan)
}

/// A lazy adjoint is a storage-conjugate operand, as in the
/// multiplicity-free mode; its relation to its logical space is checked
/// locally. The staging order is the provider-event order: local
/// validation, pair admission, the contraction's braiding check, then the
/// staged destination, whose producer derives the HomSpace from the
/// logical spaces.
impl<P> ContractStaging<P> for CheckedGenericAdmissionMode
where
    P: CheckedGenericRigidSymbols<Scalar = f64>,
{
    /// The staged destination and the preview the planner targets.
    type ContractStage = (PreparedCheckedGenericDynamicSpace, DynamicFusionMapSpace);

    fn stage_contraction<'a>(
        (lhs, lhs_operand, lhs_len): ContractSide<'a, P>,
        (rhs, rhs_operand, rhs_len): ContractSide<'a, P>,
        request: ContractRequest<'_>,
    ) -> Result<
        StagedContraction<Self::ContractStage, CheckedAuthority<'a, P>, CheckedContractTxn>,
        CheckedGenericPlanError<P::Error>,
    > {
        let compose_axes;
        let (axes, codomain_rank) = match request {
            ContractRequest::Contract {
                axes,
                codomain_rank,
            } => (axes, codomain_rank),
            ContractRequest::Compose => {
                let lhs_nout = lhs.space().nout();
                compose_axes = (
                    (lhs_nout..lhs.space().rank()).collect::<Vec<_>>(),
                    (0..rhs.space().nout()).collect::<Vec<_>>(),
                );
                (
                    TensorContractSpec::with_default_output_order(&compose_axes.0, &compose_axes.1),
                    lhs_nout,
                )
            }
            // A Bosonic planar contraction is the ordinary one, which the
            // facade runs; the checked planner has no planar route.
            ContractRequest::Planar { .. } => {
                return Err(OperationError::UnsupportedTensorContractScope {
                    message: "checked Generic planar contraction requires Bosonic braiding",
                }
                .into())
            }
        };
        let axis_plan = validate_contract_local(
            (lhs, lhs_operand, lhs_len),
            (rhs, rhs_operand, rhs_len),
            axes,
            codomain_rank,
        )?;
        let provider = crate::admission::admit_checked_generic_pair(lhs, rhs)?;
        let sides = [(lhs_operand, lhs.space()), (rhs_operand, rhs.space())];
        let authority = match request {
            ContractRequest::Contract { .. } | ContractRequest::Planar { .. } => {
                CheckedAuthority::contraction(lhs, sides)?
            }
            ContractRequest::Compose => CheckedAuthority::composition(lhs, sides),
        };
        let destination = lhs.prepare_final_homspace_generic_from_checked(provider, || {
            FusionTreeHomSpace::try_tensorcontract_homspace_generic_checked(
                provider,
                lhs.space().homspace(),
                rhs.space().homspace(),
                axes.lhs_contracting_axes(),
                axes.rhs_contracting_axes(),
                &axis_plan.output_axes,
                codomain_rank,
            )
            .map_err(CheckedGenericPlanError::from)
        })?;
        let preview = destination.preview();
        let len = destination.required_len();
        Ok(StagedContraction {
            stage: (destination, preview),
            authority,
            txn: CheckedContractTxn::new(),
            len,
        })
    }

    fn contract_destination_matches(
        (destination, _): &Self::ContractStage,
        dst: &DynamicFusionMapSpace,
    ) -> bool {
        destination.matches(dst)
    }

    fn contract_destination<'s>(
        (_, preview): &'s Self::ContractStage,
        lhs: &'s BoundDynamicFusionMapSpace<P>,
    ) -> (&'s P, &'s DynamicFusionMapSpace) {
        (lhs.provider(), preview)
    }

    /// Only after the fallible destination commit: a failed call commits no
    /// intermediate and publishes nothing. These commits cannot fail; a lost
    /// or refused admission leaves its preview unpublishable.
    fn commit_contraction(
        lhs: &BoundDynamicFusionMapSpace<P>,
        (destination, preview): Self::ContractStage,
        txn: CheckedContractTxn,
        resolution: &StorageContractResolution<f64>,
    ) -> Result<BoundDynamicFusionMapSpace<P>, CheckedGenericPlanError<P::Error>> {
        let destination = lhs.commit_final_homspace_generic_bound_checked(destination)?;
        txn.commit(
            (
                Arc::clone(preview.structure()),
                Arc::clone(destination.space().structure()),
            ),
            resolution,
        );
        Ok(destination)
    }
}

/// A checked Generic contraction's staged destination and preview, its
/// planning authority and its call transaction, held opaque until
/// [`StagedContract::commit`].
///
/// Internal and unstable despite being public for `symtn`.
#[doc(hidden)]
pub struct CheckedContractStage<'a, P>(
    StagedContraction<
        (PreparedCheckedGenericDynamicSpace, DynamicFusionMapSpace),
        CheckedAuthority<'a, P>,
        CheckedContractTxn,
    >,
);

impl<'a, P> ContractStageParts<'a, P> for CheckedContractStage<'a, P>
where
    P: CheckedGenericRigidSymbols<Scalar = f64> + 'a,
{
    type M = CheckedGenericAdmissionMode;
    fn wrap(parts: StageParts<'a, P, Self::M>) -> Self {
        Self(parts)
    }
    fn parts(&self) -> &StageParts<'a, P, Self::M> {
        &self.0
    }
    fn parts_mut(&mut self) -> &mut StageParts<'a, P, Self::M> {
        &mut self.0
    }
    fn into_parts(self) -> StageParts<'a, P, Self::M> {
        self.0
    }
}

impl<'a, P> StagedContract<'a, P, CheckedContractStage<'a, P>>
where
    P: CheckedGenericRigidSymbols<Scalar = f64> + 'a,
{
    /// Admits checked Generic operands and stages the destination of
    /// `request`; operands as in [`StagedContract::multiplicity_free`]. The
    /// staging validates the stored lengths.
    #[allow(clippy::type_complexity)]
    pub fn checked_generic(
        lhs: (&'a BoundDynamicFusionMapSpace<P>, FusionOperand<'a>, usize),
        rhs: (&'a BoundDynamicFusionMapSpace<P>, FusionOperand<'a>, usize),
        request: ContractRequest<'a>,
    ) -> Result<Self, CheckedGenericPlanError<P::Error>> {
        stage_contract(lhs, rhs, request)
    }

    /// The staged preview's `spacecheck_contract`.
    pub fn require_destination(&self, dst: &DynamicFusionMapSpace) -> Result<(), OperationError> {
        staged_require_destination(self, dst)
    }

    /// The staged destination preview the route writes.
    pub fn destination(&self) -> &DynamicFusionMapSpace {
        staged_destination(self)
    }

    /// The staged destination's payload length.
    pub fn destination_len(&self) -> usize {
        staged_len(self)
    }

    /// The canonical core route, resolved without any context.
    pub fn try_core<X: ExecCaps>(
        &self,
    ) -> Result<CoreRoute<f64>, CheckedGenericPlanError<P::Error>> {
        staged_try_core::<X, P, CheckedContractStage<'a, P>>(self)
    }

    /// The rest of the planner after a core miss, under the call's own
    /// transaction (it needs no context).
    pub fn plan_beyond_core<X: ExecCaps>(
        &mut self,
        miss: CoreMiss,
    ) -> Result<StorageContractResolution<f64>, CheckedGenericPlanError<P::Error>> {
        // The checked planner never reads the context's planning state.
        staged_plan_beyond_core::<X, P, CheckedContractStage<'a, P>>(
            self,
            miss,
            &mut TreeTransformPlanning::default(),
        )
    }

    /// Commits the staged destination after the replay of `resolution`, then
    /// the intermediates and transformers that route used (#2063).
    pub fn commit(
        self,
        resolution: &StorageContractResolution<f64>,
    ) -> Result<BoundDynamicFusionMapSpace<P>, CheckedGenericPlanError<P::Error>> {
        staged_commit(self, resolution)
    }
}

impl<D, BT, BC> TensorContractFusionExecutionContext<D, RuleIdentity, BT, BC>
where
    D: DenseRecouplingScalar
        + RecouplingCoefficientAction<f64>
        + ConjugateValue
        + Copy
        + Zero
        + ZeroBytes,
    BT: TreeTransformBackend<D, f64>,
    BC: TensorContractBackend<D, f64>,
{
    /// Contracts two checked Generic tensors, the result split after its
    /// first `codomain_rank` output axes (TensorOperations `pAB`) and owned
    /// by the left provider allocation. Each operand is its logical space,
    /// its storage operand (a lazy adjoint is storage-conjugate) and its
    /// storage payload, as in the multiplicity-free entry; the conjugation
    /// flags are the operands'.
    ///
    /// It plans through the shared [`plan_contract`] ladder in the checked
    /// mode (the canonical core, TensorKit's `copyC`, the `DynamicTree`
    /// artifact) and replays on the Host route executor. The destination and
    /// every intermediate are staged and planned over as previews; only after
    /// the replay succeeds is the destination committed, then the
    /// intermediates the route used, then the staged transformers (#2063).
    ///
    /// This concrete cross-crate entrypoint is internal and unstable despite
    /// being public for `symtn`; downstream callers must not rely on it.
    ///
    /// [`plan_contract`]: TensorContractFusionExecutionContext::plan_contract
    #[doc(hidden)]
    #[allow(clippy::type_complexity)]
    pub fn tensorcontract_checked_generic_in<P>(
        &mut self,
        (lhs_space, lhs, lhs_data): (&BoundDynamicFusionMapSpace<P>, FusionOperand<'_>, &[D]),
        (rhs_space, rhs, rhs_data): (&BoundDynamicFusionMapSpace<P>, FusionOperand<'_>, &[D]),
        (lhs_axes, rhs_axes, output_order): (&[usize], &[usize], OutputAxisOrder<'_>),
        codomain_rank: usize,
    ) -> CheckedContractResult<P, D>
    where
        P: CheckedGenericRigidSymbols<Scalar = f64>,
    {
        self.tensorcontract_in::<CheckedContractStage<'_, P>, P>(
            (lhs_space, lhs, lhs_data),
            (rhs_space, rhs, rhs_data),
            ContractRequest::Contract {
                axes: TensorContractSpec::new_with_conjugation(
                    lhs_axes,
                    rhs_axes,
                    output_order,
                    lhs.storage_conjugate(),
                    rhs.storage_conjugate(),
                ),
                codomain_rank,
            },
        )
    }

    /// `dst = alpha · contract(lhs, rhs) + beta · dst` for checked Generic
    /// operands (as in [`Self::tensorcontract_checked_generic_in`]) and an
    /// existing destination: the same staging, the staged destination
    /// compared with `dst_space` before `dst_data` is asked for the buffer,
    /// the write, then the same commit and publication (#2063).
    ///
    /// This concrete cross-crate entrypoint is internal and unstable despite
    /// being public for `symtn`; downstream callers must not rely on it.
    #[doc(hidden)]
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    pub fn tensorcontract_into_checked_generic_in<'d, P, E>(
        &mut self,
        (lhs_space, lhs, lhs_data): (&BoundDynamicFusionMapSpace<P>, FusionOperand<'_>, &[D]),
        (rhs_space, rhs, rhs_data): (&BoundDynamicFusionMapSpace<P>, FusionOperand<'_>, &[D]),
        (lhs_axes, rhs_axes, output_order): (&[usize], &[usize], OutputAxisOrder<'_>),
        codomain_rank: usize,
        dst_space: &DynamicFusionMapSpace,
        dst_data: impl FnOnce() -> Result<&'d mut [D], E>,
        alpha: D,
        beta: D,
    ) -> Result<(), E>
    where
        P: CheckedGenericRigidSymbols<Scalar = f64>,
        E: From<CheckedGenericPlanError<P::Error>> + From<OperationError>,
        D: 'd,
    {
        self.tensorcontract_into_in::<CheckedContractStage<'_, P>, P, E>(
            (lhs_space, lhs, lhs_data),
            (rhs_space, rhs, rhs_data),
            ContractRequest::Contract {
                axes: TensorContractSpec::new_with_conjugation(
                    lhs_axes,
                    rhs_axes,
                    output_order,
                    lhs.storage_conjugate(),
                    rhs.storage_conjugate(),
                ),
                codomain_rank,
            },
            dst_space,
            dst_data,
            alpha,
            beta,
        )
    }

    /// Canonical composition (TensorKit `mul!`) of two checked Generic
    /// tensors: `lhs.domain` is glued to `rhs.codomain` in order; operands
    /// as in [`Self::tensorcontract_checked_generic_in`].
    ///
    /// It plans through the shared [`plan_compose`](super::plan_compose)
    /// rung in the checked mode and replays on the Host route executor: one
    /// coefficient-free block GEMM per coupled sector, or the checked
    /// irregular core for a non-canonical tiling. No leg crosses another,
    /// hence no braiding style is required. The destination is staged,
    /// planned over as a preview, and committed only after the replay
    /// succeeds (#2063).
    ///
    /// This concrete cross-crate entrypoint is internal and unstable despite
    /// being public for `symtn`; downstream callers must not rely on it.
    #[doc(hidden)]
    #[allow(clippy::type_complexity)]
    pub fn tensorcompose_checked_generic_in<P>(
        &mut self,
        lhs: (&BoundDynamicFusionMapSpace<P>, FusionOperand<'_>, &[D]),
        rhs: (&BoundDynamicFusionMapSpace<P>, FusionOperand<'_>, &[D]),
    ) -> CheckedContractResult<P, D>
    where
        P: CheckedGenericRigidSymbols<Scalar = f64>,
    {
        self.tensorcontract_in::<CheckedContractStage<'_, P>, P>(lhs, rhs, ContractRequest::Compose)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};

    use crate::tests::{direct_side, entry_axes, GenericMultiplicityRule};
    use symtn_core::{
        BraidingStyleKind, CheckedGenericFusion, CoupledSectorFold, FusionProductSpace, FusionRule,
        FusionStyleKind, GenericFArray, GenericRMatrix, InfallibleGeneric, RuleIdentity, SectorId,
        SectorLeg, SectorVec,
    };

    use super::*;
    use crate::contract::backend::TensorContractBackend;
    use crate::contract::structure::TensorContractStructure;
    use crate::DenseTreeTransformOperations;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum Query {
        Dual,
        Channel,
        N,
        F,
        R,
        Rigidity,
    }

    impl Query {
        const COUNT: usize = 6;
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum Event {
        Identity,
        Style,
        Braiding,
        Query(Query),
    }

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct SpyError(Query);

    impl std::fmt::Display for SpyError {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "injected {:?} failure", self.0)
        }
    }

    impl std::error::Error for SpyError {}

    /// Checked-only wrapper: deliberately does not implement `FusionRule`.
    struct CheckedGenericSpy {
        rule: GenericMultiplicityRule,
        calls: Cell<[usize; Query::COUNT]>,
        events: RefCell<Vec<Event>>,
        fail: Cell<Option<Query>>,
        malformed: Cell<Option<Query>>,
        braiding: Cell<BraidingStyleKind>,
    }

    impl CheckedGenericSpy {
        fn new() -> Self {
            Self {
                rule: GenericMultiplicityRule,
                calls: Cell::new([0; Query::COUNT]),
                events: RefCell::new(Vec::new()),
                fail: Cell::new(None),
                malformed: Cell::new(None),
                braiding: Cell::new(BraidingStyleKind::Bosonic),
            }
        }

        fn hit(&self, query: Query) -> Result<(), SpyError> {
            self.events.borrow_mut().push(Event::Query(query));
            let mut calls = self.calls.get();
            calls[query as usize] += 1;
            self.calls.set(calls);
            if self.fail.get() == Some(query) {
                Err(SpyError(query))
            } else {
                Ok(())
            }
        }

        fn reset(&self) {
            self.calls.set([0; Query::COUNT]);
            self.events.borrow_mut().clear();
            self.fail.set(None);
            self.malformed.set(None);
        }

        fn count(&self, query: Query) -> usize {
            self.calls.get()[query as usize]
        }

        fn algebra_calls(&self) -> usize {
            self.calls.get().into_iter().sum()
        }
    }

    impl symtn_core::testing::Provider for CheckedGenericSpy {}

    impl CheckedGenericFusion for CheckedGenericSpy {
        type Error = SpyError;

        // Why its own identity rather than the wrapped rule's: the composed
        // coefficients are cached per identity, and a sibling test's
        // publication under the shared rule's identity would skip this
        // spy's provider queries.
        fn rule_identity(&self) -> RuleIdentity {
            self.events.borrow_mut().push(Event::Identity);
            RuleIdentity::of_type::<Self>()
        }

        fn fusion_style(&self) -> FusionStyleKind {
            self.events.borrow_mut().push(Event::Style);
            FusionRule::fusion_style(&self.rule)
        }

        fn braiding_style(&self) -> BraidingStyleKind {
            self.events.borrow_mut().push(Event::Braiding);
            self.braiding.get()
        }

        fn vacuum(&self) -> SectorId {
            FusionRule::vacuum(&self.rule)
        }

        fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
            self.hit(Query::Dual)?;
            Ok(FusionRule::dual(&self.rule, sector))
        }

        fn try_fusion_channels(
            &self,
            left: SectorId,
            right: SectorId,
        ) -> Result<SectorVec, Self::Error> {
            self.hit(Query::Channel)?;
            Ok(FusionRule::fusion_channels(&self.rule, left, right))
        }

        fn try_fusion_channels_in_table(
            &self,
            left: SectorId,
            right: SectorId,
        ) -> Result<SectorVec, Self::Error> {
            self.hit(Query::Channel)?;
            Ok(FusionRule::fusion_channels(&self.rule, left, right))
        }

        // Counts one query for the fold itself, as the engine's failure
        // budgets are stated per provider call rather than per inner channel
        // lookup. The classification is the trait default over this rule.
        fn try_coupled_sector_fold(
            &self,
            effective: &[SectorId],
        ) -> Result<CoupledSectorFold, Self::Error> {
            self.hit(Query::Channel)?;
            match InfallibleGeneric::new(&self.rule).try_coupled_sector_fold(effective) {
                Ok(fold) => Ok(fold),
                Err(never) => match never {},
            }
        }

        fn try_nsymbol(
            &self,
            left: SectorId,
            right: SectorId,
            coupled: SectorId,
        ) -> Result<usize, Self::Error> {
            self.hit(Query::N)?;
            Ok(FusionRule::nsymbol(&self.rule, left, right, coupled))
        }
    }

    impl CheckedGenericRigidSymbols for CheckedGenericSpy {
        type Scalar = f64;

        fn try_sqrt_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
            self.hit(Query::Rigidity)?;
            let _ = sector;
            Ok(1.0)
        }

        fn try_inv_sqrt_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
            self.hit(Query::Rigidity)?;
            let _ = sector;
            Ok(1.0)
        }

        fn try_frobenius_schur_phase_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
            self.hit(Query::Rigidity)?;
            let _ = sector;
            Ok(1.0)
        }

        fn try_f_symbol_generic(
            &self,
            a: SectorId,
            b: SectorId,
            c: SectorId,
            d: SectorId,
            e: SectorId,
            f: SectorId,
        ) -> Result<GenericFArray<f64>, Self::Error> {
            self.hit(Query::F)?;
            let shape = (
                self.rule.nsymbol(a, b, e),
                self.rule.nsymbol(e, c, d),
                self.rule.nsymbol(b, c, f),
                self.rule.nsymbol(a, f, d),
            );
            let mut data = vec![0.0; shape.0 * shape.1 * shape.2 * shape.3];
            let cols = shape.0 * shape.1;
            let rows = shape.2 * shape.3;
            for index in 0..cols.min(rows) {
                data[index * rows + index] = 1.0;
            }
            let symbol = GenericFArray::new(data, shape);
            if self.malformed.get() == Some(Query::F) {
                Ok(GenericFArray::new(
                    symbol.data().to_vec(),
                    (1, 1, symbol.data().len(), 1),
                ))
            } else {
                Ok(symbol)
            }
        }

        fn try_r_symbol_generic(
            &self,
            a: SectorId,
            b: SectorId,
            c: SectorId,
        ) -> Result<GenericRMatrix<f64>, Self::Error> {
            self.hit(Query::R)?;
            let size = self.rule.nsymbol(a, b, c);
            let mut data = vec![0.0; size * size];
            for index in 0..size {
                data[index * size + index] = 1.0;
            }
            let symbol = GenericRMatrix::new(data, size, size);
            if self.malformed.get() == Some(Query::R) {
                Ok(GenericRMatrix::new(
                    symbol.data().to_vec(),
                    1,
                    symbol.data().len(),
                ))
            } else {
                Ok(symbol)
            }
        }
    }

    /// The default contract backend, failing its `fail_at`-th rank-2 GEMM
    /// job (counted from 0) when set. Batches run serially through the
    /// trait default, so every job is counted.
    #[derive(Default)]
    struct FailingBackend {
        inner: DenseTreeTransformOperations,
        jobs: usize,
        fail_at: Option<usize>,
    }

    impl TensorContractBackend<f64, f64> for FailingBackend {
        type Workspace =
            <DenseTreeTransformOperations as TensorContractBackend<f64, f64>>::Workspace;

        fn tensorcontract_structure_into_raw(
            &mut self,
            _workspace: &mut Self::Workspace,
            _structure: &TensorContractStructure<f64>,
            _dst_structure: &Arc<BlockStructure>,
            _lhs_structure: &Arc<BlockStructure>,
            _rhs_structure: &Arc<BlockStructure>,
            _dst_data: &mut [f64],
            _lhs_data: &[f64],
            _rhs_data: &[f64],
            _alpha: f64,
            _beta: f64,
        ) -> Result<(), OperationError> {
            unreachable!("the route executor runs rank-2 GEMM jobs only")
        }

        fn matmul_rank2_into_raw(
            &mut self,
            workspace: &mut Self::Workspace,
            dst_data: &mut [f64],
            lhs_data: &[f64],
            rhs_data: &[f64],
            rows: usize,
            contracted: usize,
            cols: usize,
        ) -> Result<(), OperationError> {
            self.matmul_rank2_axpby_into_raw(
                workspace, dst_data, lhs_data, rhs_data, rows, contracted, cols, 1.0, 0.0,
            )
        }

        fn matmul_rank2_axpby_into_raw(
            &mut self,
            workspace: &mut Self::Workspace,
            dst_data: &mut [f64],
            lhs_data: &[f64],
            rhs_data: &[f64],
            rows: usize,
            contracted: usize,
            cols: usize,
            alpha: f64,
            beta: f64,
        ) -> Result<(), OperationError> {
            let job = self.jobs;
            self.jobs += 1;
            if self.fail_at == Some(job) {
                return Err(OperationError::StridedKernel {
                    message: "injected checked Generic core failure".into(),
                });
            }
            TensorContractBackend::<f64, f64>::matmul_rank2_axpby_into_raw(
                &mut self.inner,
                workspace,
                dst_data,
                lhs_data,
                rhs_data,
                rows,
                contracted,
                cols,
                alpha,
                beta,
            )
        }

        /// A lazy adjoint's parent is read through its `MatrixOp`, one batch
        /// of jobs; they count and fail like the per-job GEMMs.
        #[allow(clippy::too_many_arguments)]
        fn matmul_rank2_batch_with_ops_axpby_into_raw(
            &mut self,
            workspace: &mut Self::Workspace,
            dst_data: &mut [f64],
            lhs_data: &[f64],
            rhs_data: &[f64],
            jobs: &[symtn_operations::fusion_replay::Rank2GemmBatchJob],
            runs: &[usize],
            lhs_op: symtn_operations::fusion_replay::MatrixOp,
            rhs_op: symtn_operations::fusion_replay::MatrixOp,
            alpha: f64,
            beta: f64,
        ) -> Result<(), OperationError> {
            use symtn_operations::fusion_replay::MatrixOp;
            if lhs_op == MatrixOp::Identity && rhs_op == MatrixOp::Identity {
                return self.matmul_rank2_batch_axpby_into_raw(
                    workspace, dst_data, lhs_data, rhs_data, jobs, runs, alpha, beta,
                );
            }
            let first = self.jobs;
            self.jobs += jobs.len();
            if self
                .fail_at
                .is_some_and(|job| (first..self.jobs).contains(&job))
            {
                return Err(OperationError::StridedKernel {
                    message: "injected checked Generic core failure".into(),
                });
            }
            TensorContractBackend::<f64, f64>::matmul_rank2_batch_with_ops_axpby_into_raw(
                &mut self.inner,
                workspace,
                dst_data,
                lhs_data,
                rhs_data,
                jobs,
                runs,
                lhs_op,
                rhs_op,
                alpha,
                beta,
            )
        }
    }

    type FailingContext = TensorContractFusionExecutionContext<
        f64,
        RuleIdentity,
        DenseTreeTransformOperations,
        FailingBackend,
    >;

    fn failing_context(fail_at: Option<usize>) -> FailingContext {
        let mut context = FailingContext::new(
            DenseTreeTransformOperations::default(),
            FailingBackend::default(),
        );
        context.contract_backend_mut().fail_at = fail_at;
        context
    }

    type SpySpace = BoundDynamicFusionMapSpace<CheckedGenericSpy>;

    /// One public checked contraction on a fresh default context.
    fn contract(
        (lhs, lhs_data): (&SpySpace, &[f64]),
        (rhs, rhs_data): (&SpySpace, &[f64]),
        axes: TensorContractSpec<'_>,
        codomain_rank: usize,
    ) -> CheckedContractResult<CheckedGenericSpy, f64> {
        let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
        context.tensorcontract_checked_generic_in(
            direct_side(lhs, lhs_data),
            direct_side(rhs, rhs_data),
            entry_axes(axes),
            codomain_rank,
        )
    }

    /// The route the last call of `context` planned.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum Route {
        Core,
        CopyC,
        DynamicTree,
    }

    fn last_route<BC>(
        context: &TensorContractFusionExecutionContext<
            f64,
            RuleIdentity,
            DenseTreeTransformOperations,
            BC,
        >,
    ) -> Route
    where
        BC: TensorContractBackend<f64, f64>,
    {
        if context.last_resolution_is_core() {
            Route::Core
        } else if context.last_resolution_orientation().is_some() {
            Route::DynamicTree
        } else {
            Route::CopyC
        }
    }

    fn ramp(len: usize, scale: f64, offset: f64) -> Vec<f64> {
        (0..len)
            .map(|index| index as f64 * scale + offset)
            .collect()
    }

    fn homspace(_rule: &GenericMultiplicityRule, nout: usize, nin: usize) -> FusionTreeHomSpace {
        let leg = || SectorLeg::new([(SectorId::new(1), 1)], false);
        FusionTreeHomSpace::new(
            FusionProductSpace::new((0..nout).map(|_| leg())),
            FusionProductSpace::new((0..nin).map(|_| leg())),
        )
    }

    #[allow(clippy::arc_with_non_send_sync)]
    fn bound_pair(
        nout: usize,
        nin: usize,
    ) -> (
        Arc<CheckedGenericSpy>,
        BoundDynamicFusionMapSpace<CheckedGenericSpy>,
        Arc<CheckedGenericSpy>,
        BoundDynamicFusionMapSpace<CheckedGenericSpy>,
    ) {
        let left = Arc::new(CheckedGenericSpy::new());
        let right = Arc::new(CheckedGenericSpy::new());
        let homspace = homspace(&left.rule, nout, nin);
        let lhs = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
            Arc::clone(&left),
            homspace.clone(),
        )
        .unwrap();
        let rhs = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
            Arc::clone(&right),
            homspace,
        )
        .unwrap();
        left.reset();
        right.reset();
        (left, lhs, right, rhs)
    }

    /// What: checked compose plans the shared Core route over a preview of
    /// its staged destination. Its provider events are the pair admission,
    /// the destination stage (identity read, admission style, and cold its
    /// structure queries) and the commit guard's style (#2046): planning
    /// and replay query nothing, and no second (core) destination is
    /// derived. The left binding owns the result; a failing destination
    /// query surfaces exactly and publishes nothing.
    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn checked_compose_plans_the_core_route_without_provider_events_after_staging() {
        const ISOLATED: &str = "SYMTN_CHECKED_GENERIC_COMPOSE_EVENTS_ISOLATED";
        if std::env::var_os(ISOLATED).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "contract::checked_generic::tests::checked_compose_plans_the_core_route_without_provider_events_after_staging",
                ])
                .env(ISOLATED, "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "isolated compose test failed:\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }

        let (left, lhs, right, rhs) = bound_pair(1, 1);
        symtn_core::clear_structure_caches();
        let lhs_data = vec![1.0; lhs.space().required_len().unwrap()];
        let rhs_data = vec![2.0; rhs.space().required_len().unwrap()];
        let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
        // A failing destination query surfaces exactly and publishes nothing.
        left.fail.set(Some(Query::Channel));
        let error = context
            .tensorcompose_checked_generic_in(
                direct_side(&lhs, &lhs_data),
                direct_side(&rhs, &rhs_data),
            )
            .unwrap_err();
        assert!(matches!(
            error,
            CheckedGenericPlanError::Provider(SpyError(Query::Channel))
        ));
        assert_eq!(
            symtn_core::structure_cache_info(symtn_core::StructureCacheKind::DegeneracyStructure)
                .entries(),
            0
        );
        use Event::{Identity, Style};
        let channel = Event::Query(Query::Channel);
        let expected = [
            (
                "cold",
                vec![Identity, Style, Identity, Style, channel, channel, Style],
            ),
            ("warm", vec![Identity, Style, Identity, Style, Style]),
        ];
        for (call, events) in expected {
            left.reset();
            right.reset();
            let (output, data) = context
                .tensorcompose_checked_generic_in(
                    direct_side(&lhs, &lhs_data),
                    direct_side(&rhs, &rhs_data),
                )
                .unwrap();
            assert!(context.last_resolution_is_core(), "{call}");
            assert!(Arc::ptr_eq(output.provider_arc(), &left), "{call}");
            assert_eq!(right.algebra_calls(), 0, "{call}");
            assert_eq!(data.len(), output.space().required_len().unwrap());
            assert_eq!(*left.events.borrow(), events, "{call}");
            assert_eq!(*right.events.borrow(), [Identity, Style], "{call}");
        }
    }

    const ISOLATED: &str = "SYMTN_CHECKED_GENERIC_CONTRACT_ISOLATED";

    /// Reruns test `name` alone in a child process, whose cache assertions
    /// then see no concurrent clear; true inside that child.
    fn isolated(name: &str) -> bool {
        if std::env::var_os(ISOLATED).is_some() {
            return true;
        }
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                &format!("contract::checked_generic::tests::{name}"),
            ])
            .env(ISOLATED, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "isolated {name} failed:\n{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        false
    }

    use Event::{Braiding, Identity, Style};

    /// What: a canonical contraction plans the shared Core rung (zero-copy
    /// candidate, identity output). Its provider events are the pair
    /// admission, the braiding read, the destination stage and the
    /// commit guard's style (#2046): planning and replay query nothing, and
    /// no second (core) destination is derived. The left binding owns the
    /// result; a failing destination query surfaces exactly and publishes
    /// nothing.
    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn checked_contraction_plans_the_core_route_without_provider_events_after_staging() {
        if !isolated(
            "checked_contraction_plans_the_core_route_without_provider_events_after_staging",
        ) {
            return;
        }
        let (left, lhs, right, rhs) = bound_pair(1, 1);
        symtn_core::clear_structure_caches();
        let lhs_data = ramp(lhs.space().required_len().unwrap(), 0.5, -1.0);
        let rhs_data = ramp(rhs.space().required_len().unwrap(), -0.25, 2.0);
        let axes = || TensorContractSpec::with_default_output_order(&[1], &[0]);
        let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
        left.fail.set(Some(Query::Channel));
        let error = context
            .tensorcontract_checked_generic_in(
                direct_side(&lhs, &lhs_data),
                direct_side(&rhs, &rhs_data),
                entry_axes(axes()),
                1,
            )
            .unwrap_err();
        assert!(matches!(
            error,
            CheckedGenericPlanError::Provider(SpyError(Query::Channel))
        ));
        assert_eq!(cache_entries(), (0, 0));
        let channel = Event::Query(Query::Channel);
        let expected = [
            (
                "cold",
                vec![
                    Identity, Style, Braiding, Identity, Style, channel, channel, Style,
                ],
            ),
            (
                "warm",
                vec![Identity, Style, Braiding, Identity, Style, Style],
            ),
        ];
        let mut first = None;
        for (call, events) in expected {
            left.reset();
            right.reset();
            let (output, data) = context
                .tensorcontract_checked_generic_in(
                    direct_side(&lhs, &lhs_data),
                    direct_side(&rhs, &rhs_data),
                    entry_axes(axes()),
                    1,
                )
                .unwrap();
            assert_eq!(last_route(&context), Route::Core, "{call}");
            assert!(Arc::ptr_eq(output.provider_arc(), &left), "{call}");
            assert!(!Arc::ptr_eq(output.provider_arc(), &right), "{call}");
            assert_eq!(right.algebra_calls(), 0, "{call}");
            assert_eq!(data.len(), output.space().required_len().unwrap());
            assert_eq!(*left.events.borrow(), events, "{call}");
            assert_eq!(*right.events.borrow(), [Identity, Style], "{call}");
            // The destination and nothing else is committed.
            assert_eq!(cache_entries(), (1, 0), "{call}");
            let bits = data.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(*first.get_or_insert_with(|| bits.clone()), bits, "{call}");
        }
    }

    /// What: local request errors (axes, adjoint relation, rank, payload
    /// length) are reported before any provider query.
    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn checked_contraction_rejects_local_errors_before_provider_queries() {
        let (left, lhs, right, rhs) = bound_pair(1, 1);
        // A parent whose adjoint is not `lhs`: rank 3 against rank 2.
        let (_, parent, _, _) = bound_pair(2, 1);
        let lhs_data = vec![0.0; lhs.space().required_len().unwrap()];
        let rhs_data = vec![0.0; rhs.space().required_len().unwrap()];
        let parent_data = vec![0.0; parent.space().required_len().unwrap()];
        let short = vec![0.0; rhs_data.len() - 1];
        let direct = direct_side(&lhs, &lhs_data);
        let adjoint = (
            &lhs,
            FusionOperand::adjoint(parent.space()),
            &parent_data[..],
        );
        let default = || TensorContractSpec::with_default_output_order(&[1], &[0]);
        let cases = [
            (
                "axis",
                direct,
                TensorContractSpec::with_default_output_order(&[9], &[0]),
                1,
                &rhs_data,
            ),
            ("adjoint relation", adjoint, default(), 1, &rhs_data),
            ("rank", direct, default(), 3, &rhs_data),
            ("length", direct, default(), 1, &short),
        ];
        for (name, lhs_side, axes, nout, rhs_data) in cases {
            left.reset();
            right.reset();
            let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
            let error = context
                .tensorcontract_checked_generic_in(
                    lhs_side,
                    direct_side(&rhs, rhs_data),
                    entry_axes(axes),
                    nout,
                )
                .unwrap_err();
            assert!(
                !matches!(error, CheckedGenericPlanError::Provider(_)),
                "{name}: {error:?}"
            );
            assert!(left.events.borrow().is_empty(), "{name}");
            assert!(right.events.borrow().is_empty(), "{name}");
        }
    }

    /// What: a contraction whose operands are not in core form takes the
    /// `DynamicTree` artifact; only the lhs is recoupled (F and R queries),
    /// the rhs borrowed in place.
    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn checked_contraction_executes_one_transforming_candidate() {
        if !isolated("checked_contraction_executes_one_transforming_candidate") {
            return;
        }
        let (left, lhs, right, rhs) = bound_pair(2, 2);
        symtn_core::clear_structure_caches();
        let lhs_data = vec![1.0; lhs.space().required_len().unwrap()];
        let rhs_data = vec![2.0; rhs.space().required_len().unwrap()];
        let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
        crate::probe::take_completed_transformer_activity();
        let (output, data) = context
            .tensorcontract_checked_generic_in(
                direct_side(&lhs, &lhs_data),
                direct_side(&rhs, &rhs_data),
                entry_axes(TensorContractSpec::with_default_output_order(
                    &[3, 2],
                    &[0, 1],
                )),
                2,
            )
            .unwrap();
        assert_eq!(last_route(&context), Route::DynamicTree);
        assert!(Arc::ptr_eq(output.provider_arc(), &left));
        assert_eq!(right.algebra_calls(), 0);
        assert!(left.count(Query::F) > 0);
        assert!(left.count(Query::R) > 0);
        let transformers = crate::probe::take_completed_transformer_activity();
        assert_eq!((transformers.builds, transformers.publications), (1, 1));
        assert_eq!(data.len(), output.space().required_len().unwrap());
    }

    /// The Core, CopyC and `DynamicTree` (one and three transforms) checked
    /// fixtures: `(name, nout = nin, lhs axes, rhs axes, output, codomain
    /// rank, route)`.
    type Fixture = (
        &'static str,
        usize,
        &'static [usize],
        &'static [usize],
        Option<&'static [usize]>,
        usize,
        Route,
    );

    const FIXTURES: [Fixture; 4] = [
        ("core", 1, &[1], &[0], None, 1, Route::Core),
        (
            "copy_c",
            2,
            &[2, 3],
            &[0, 1],
            Some(&[1, 0, 2, 3]),
            2,
            Route::CopyC,
        ),
        ("lhs_only", 2, &[3, 2], &[0, 1], None, 2, Route::DynamicTree),
        (
            "three_stages",
            2,
            &[3, 1],
            &[0, 3],
            Some(&[2, 0, 3, 1]),
            2,
            Route::DynamicTree,
        ),
    ];

    fn fixture_axes<'a>(
        lhs: &'a [usize],
        rhs: &'a [usize],
        output: Option<&'a [usize]>,
    ) -> TensorContractSpec<'a> {
        match output {
            None => TensorContractSpec::with_default_output_order(lhs, rhs),
            Some(output) => {
                TensorContractSpec::new(lhs, rhs, symtn_operations::OutputAxisOrder::Axes(output))
            }
        }
    }

    /// What: on every route a failing provider query (destination, source
    /// spaces and structures, core destination, CopyC temporary, output
    /// structure) or a malformed symbol surfaces exactly, and nothing is
    /// published: no layout (cache 2), no transformer (cache 3), no
    /// composed coefficient, no output.
    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn checked_contraction_preserves_late_provider_and_shape_errors_on_every_route() {
        if !isolated("checked_contraction_preserves_late_provider_and_shape_errors_on_every_route")
        {
            return;
        }
        for (name, n, lhs_axes, rhs_axes, output, nout, route) in FIXTURES {
            let (left, lhs, right, rhs) = bound_pair(n, n);
            let lhs_data = ramp(lhs.space().required_len().unwrap(), 0.25, -1.0);
            let rhs_data = ramp(rhs.space().required_len().unwrap(), -0.5, 1.5);
            let axes = || fixture_axes(lhs_axes, rhs_axes, output);
            let mut failed = 0;
            for query in [Query::Dual, Query::Channel, Query::N, Query::F, Query::R] {
                symtn_core::clear_structure_caches();
                left.reset();
                right.reset();
                left.fail.set(Some(query));
                crate::probe::take_completed_transformer_activity();
                match contract((&lhs, &lhs_data), (&rhs, &rhs_data), axes(), nout) {
                    Err(error) => {
                        failed += 1;
                        assert!(
                            matches!(
                                error,
                                CheckedGenericPlanError::Provider(SpyError(actual)) if actual == query
                            ),
                            "{name} {query:?}: {error:?}"
                        );
                    }
                    // A route that never asks this query succeeds.
                    Ok(_) => {
                        assert_eq!(left.count(query), 0, "{name} {query:?}");
                        continue;
                    }
                }
                assert_eq!(right.algebra_calls(), 0, "{name} {query:?}");
                assert_eq!(cache_entries(), (0, 0), "{name} {query:?}");
                assert_eq!(coefficient_admissions(), 0, "{name} {query:?}");
                let transformers = crate::probe::take_completed_transformer_activity();
                assert_eq!(transformers.publications, 0, "{name} {query:?}");
            }
            assert!(failed >= 1, "{name}: some query fails");
            if route == Route::DynamicTree {
                symtn_core::clear_structure_caches();
                left.reset();
                left.malformed.set(Some(Query::R));
                let error =
                    contract((&lhs, &lhs_data), (&rhs, &rhs_data), axes(), nout).unwrap_err();
                assert!(
                    matches!(
                        error,
                        CheckedGenericPlanError::SymbolShape { symbol: "R", .. }
                    ),
                    "{name}"
                );
                assert_eq!(cache_entries(), (0, 0), "{name}");
            }
        }
    }

    /// What: each fixture plans its route, and the GEMM failing on the first
    /// or a later job of the replay returns the error without publishing:
    /// no layout, transformer, composed coefficient or output, so a failed
    /// call leaves nothing behind (#2063). The pooled scratch a warm call
    /// sized is reused, not grown, by a failing call.
    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn checked_contraction_backend_failure_publishes_nothing_and_keeps_retained_scratch() {
        if !isolated(
            "checked_contraction_backend_failure_publishes_nothing_and_keeps_retained_scratch",
        ) {
            return;
        }
        for (name, n, lhs_axes, rhs_axes, output, nout, route) in FIXTURES {
            let (_left, lhs, _right, rhs) = bound_pair(n, n);
            let lhs_data = ramp(lhs.space().required_len().unwrap(), 0.25, -1.0);
            let rhs_data = ramp(rhs.space().required_len().unwrap(), -0.5, 1.5);
            let axes = || fixture_axes(lhs_axes, rhs_axes, output);

            // Cold failure: nothing resident afterwards.
            symtn_core::clear_structure_caches();
            crate::probe::take_completed_transformer_activity();
            let mut context = failing_context(Some(0));
            let error = context
                .tensorcontract_checked_generic_in(
                    direct_side(&lhs, &lhs_data),
                    direct_side(&rhs, &rhs_data),
                    entry_axes(axes()),
                    nout,
                )
                .unwrap_err();
            assert!(
                matches!(
                    error,
                    CheckedGenericPlanError::Operation(OperationError::StridedKernel { .. })
                ),
                "{name}"
            );
            assert_eq!(last_route(&context), route, "{name}");
            assert_eq!(cache_entries(), (0, 0), "{name}");
            assert_eq!(coefficient_admissions(), 0, "{name}");
            let transformers = crate::probe::take_completed_transformer_activity();
            assert_eq!(transformers.publications, 0, "{name}");

            // Warm failure on the last job, after a success sized the scratch.
            context.contract_backend_mut().fail_at = None;
            let reference = context
                .tensorcontract_checked_generic_in(
                    direct_side(&lhs, &lhs_data),
                    direct_side(&rhs, &rhs_data),
                    entry_axes(axes()),
                    nout,
                )
                .unwrap()
                .1;
            let jobs = context.contract_backend_mut().jobs;
            let resident = cache_entries();
            let retained = context.retained_host_scratch_bytes();
            let backend = context.contract_backend_mut();
            backend.fail_at = Some(backend.jobs + (jobs - 1) / 2);
            let error = context
                .tensorcontract_checked_generic_in(
                    direct_side(&lhs, &lhs_data),
                    direct_side(&rhs, &rhs_data),
                    entry_axes(axes()),
                    nout,
                )
                .unwrap_err();
            assert!(
                matches!(
                    error,
                    CheckedGenericPlanError::Operation(OperationError::StridedKernel { .. })
                ),
                "{name}"
            );
            assert_eq!(cache_entries(), resident, "{name}");
            assert_eq!(context.retained_host_scratch_bytes(), retained, "{name}");
            context.contract_backend_mut().fail_at = None;
            let repeat = context
                .tensorcontract_checked_generic_in(
                    direct_side(&lhs, &lhs_data),
                    direct_side(&rhs, &rhs_data),
                    entry_axes(axes()),
                    nout,
                )
                .unwrap()
                .1;
            assert_eq!(
                repeat.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{name}"
            );
        }
    }

    /// What: a successful canonical contraction commits its destination
    /// once under the left provider and publishes nothing else.
    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn checked_contraction_success_commits_once() {
        if !isolated("checked_contraction_success_commits_once") {
            return;
        }
        let (left, lhs, right, rhs) = bound_pair(1, 1);
        symtn_core::clear_structure_caches();
        let (output, _) = contract(
            (&lhs, &vec![1.0; lhs.space().required_len().unwrap()]),
            (&rhs, &vec![2.0; rhs.space().required_len().unwrap()]),
            TensorContractSpec::with_default_output_order(&[1], &[0]),
            1,
        )
        .unwrap();
        assert!(Arc::ptr_eq(output.provider_arc(), &left));
        assert_eq!(right.algebra_calls(), 0);
        assert_eq!(cache_entries(), (1, 0));
        // The commit guard's style is the last provider event: commit reuses
        // the admitted identity (#2046).
        assert_eq!(left.events.borrow().last(), Some(&Style));
    }

    /// What: the three-stage fixture's staged and output transforms build
    /// their groups, then the core GEMM fails before the commit. Nothing is
    /// published, so a retry replays the identical F/R provider ledger and
    /// error. A committed call then publishes its groups, its intermediates
    /// (cache 2, with the destination) and its three transformers (cache
    /// 3); a repeat binds those transformers, so it makes no group lookup
    /// and no F or R query and returns the same bits.
    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn checked_contraction_publishes_composed_coefficients_only_after_its_commit() {
        if !isolated("checked_contraction_publishes_composed_coefficients_only_after_its_commit") {
            return;
        }
        let (left, lhs, right, rhs) = bound_pair(2, 2);
        symtn_core::clear_structure_caches();
        let lhs_data = vec![1.0; lhs.space().required_len().unwrap()];
        let rhs_data = vec![2.0; rhs.space().required_len().unwrap()];
        let axes = || fixture_axes(&[3, 1], &[0, 3], Some(&[2, 0, 3, 1]));
        let run = |fail_at| {
            let mut context = failing_context(fail_at);
            let result = context.tensorcontract_checked_generic_in(
                direct_side(&lhs, &lhs_data),
                direct_side(&rhs, &rhs_data),
                entry_axes(axes()),
                2,
            );
            assert_eq!(last_route(&context), Route::DynamicTree);
            result
        };

        let mut ledgers = Vec::new();
        for _attempt in 0..2 {
            left.reset();
            right.reset();
            crate::probe::take_coefficient_group_activity();
            let error = run(Some(0)).unwrap_err();
            assert!(matches!(
                error,
                CheckedGenericPlanError::Operation(OperationError::StridedKernel { .. })
            ));
            let groups = crate::probe::take_coefficient_group_activity();
            assert!(groups.misses > 0);
            assert_eq!((groups.hits, groups.publications), (0, 0));
            assert_eq!(coefficient_admissions(), 0);
            let transformers = crate::probe::take_completed_transformer_activity();
            assert_eq!((transformers.builds, transformers.publications), (3, 0));
            assert_eq!(cache_entries(), (0, 0));
            assert!(left.count(Query::F) > 0 && left.count(Query::R) > 0);
            // The F/R ledger: layout walks may publish pure-data layouts
            // (#2030), so structural queries can shrink on a retry.
            ledgers.push(
                left.events
                    .borrow()
                    .iter()
                    .copied()
                    .filter(|event| matches!(event, Event::Query(Query::F | Query::R)))
                    .collect::<Vec<_>>(),
            );
        }
        assert_eq!(ledgers[0], ledgers[1]);

        left.reset();
        let (first_space, first) = run(None).unwrap();
        assert!(coefficient_admissions() > 0);
        let transformers = crate::probe::take_completed_transformer_activity();
        assert_eq!((transformers.builds, transformers.publications), (3, 3));
        // The destination and the three intermediates span three HomSpaces.
        let committed = cache_entries();
        assert_eq!(committed, (3, 3));
        left.reset();
        crate::probe::take_coefficient_group_activity();
        let (repeat_space, repeat) = run(None).unwrap();
        let groups = crate::probe::take_coefficient_group_activity();
        assert_eq!((groups.hits, groups.misses, groups.publications), (0, 0, 0));
        let transformers = crate::probe::take_completed_transformer_activity();
        assert_eq!(
            (
                transformers.hits,
                transformers.builds,
                transformers.publications
            ),
            (3, 0, 0)
        );
        assert_eq!(cache_entries(), committed);
        assert_eq!((left.count(Query::F), left.count(Query::R)), (0, 0));
        assert_eq!(first_space.space(), repeat_space.space());
        assert_eq!(
            first.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            repeat.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }

    /// What (#2063): the three-stage fixture
    /// plans a `DynamicTree` artifact that stages the lhs, the rhs and the
    /// core destination; two of its four committed spaces (the destination
    /// and three intermediates) share a HomSpace, so within the first call
    /// the later preview loses its admission to the earlier and its
    /// transformer publishes under the winner's id: the second call builds
    /// nothing.
    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn intermediates_sharing_a_homspace_converge_on_one_id_within_the_call() {
        if !isolated("intermediates_sharing_a_homspace_converge_on_one_id_within_the_call") {
            return;
        }
        let (_left, lhs, _right, rhs) = bound_pair(2, 2);
        symtn_core::clear_structure_caches();
        let lhs_data = ramp(lhs.space().required_len().unwrap(), 0.75, -1.5);
        let rhs_data = ramp(rhs.space().required_len().unwrap(), -0.5, 2.0);
        let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
        let mut run = || {
            let result = context
                .tensorcontract_checked_generic_in(
                    direct_side(&lhs, &lhs_data),
                    direct_side(&rhs, &rhs_data),
                    entry_axes(fixture_axes(&[3, 1], &[0, 3], Some(&[2, 0, 3, 1]))),
                    2,
                )
                .unwrap();
            assert_eq!(last_route(&context), Route::DynamicTree);
            result
        };
        crate::probe::take_completed_transformer_activity();
        let (cold_space, cold) = run();
        let transformers = crate::probe::take_completed_transformer_activity();
        assert_eq!((transformers.hits, transformers.builds), (0, 3));
        let committed = cache_entries();
        // Four committed spaces, three resident structures.
        assert_eq!(committed.0, 3);
        for _ in 0..3 {
            let (space, data) = run();
            let transformers = crate::probe::take_completed_transformer_activity();
            assert_eq!(
                (
                    transformers.hits,
                    transformers.builds,
                    transformers.publications
                ),
                (3, 0, 0)
            );
            assert_eq!(cache_entries(), committed);
            assert_eq!(space.space(), cold_space.space());
            assert_eq!(
                data.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                cold.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }

    /// What: general checked contraction with a fermionic or anyonic
    /// provider returns the typed `Unsupported` after the pair admission and
    /// one braiding read, before any structural query; canonical
    /// composition crosses no legs and stays admitted.
    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn checked_contraction_requires_bosonic_braiding_and_compose_does_not() {
        for braiding in [BraidingStyleKind::Fermionic, BraidingStyleKind::Anyonic] {
            let (left, lhs, right, rhs) = bound_pair(1, 1);
            left.braiding.set(braiding);
            right.braiding.set(braiding);
            let lhs_data = vec![1.0; lhs.space().required_len().unwrap()];
            let rhs_data = vec![2.0; rhs.space().required_len().unwrap()];
            let error = contract(
                (&lhs, &lhs_data),
                (&rhs, &rhs_data),
                TensorContractSpec::with_default_output_order(&[1], &[0]),
                1,
            )
            .unwrap_err();
            assert!(
                matches!(
                    error,
                    CheckedGenericPlanError::Operation(
                        OperationError::UnsupportedTensorContractScope {
                            message: "checked Generic contraction requires Bosonic braiding",
                        }
                    )
                ),
                "{braiding:?}: {error:?}"
            );
            assert_eq!(*left.events.borrow(), [Identity, Style, Braiding]);
            assert_eq!(*right.events.borrow(), [Identity, Style]);
            assert_eq!(left.algebra_calls(), 0);

            let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
            let (_, data) = context
                .tensorcompose_checked_generic_in(
                    direct_side(&lhs, &lhs_data),
                    direct_side(&rhs, &rhs_data),
                )
                .unwrap();
            assert!(data.iter().any(|&value| value != 0.0), "{braiding:?}");
        }
    }

    /// What: a staged space the planned route does not replay over, as a
    /// declined `copyC` temporary would be, is dropped by the transaction's
    /// commit unpublished; the temporary the CopyC route uses is committed.
    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn a_declined_staged_space_is_not_published() {
        if !isolated("a_declined_staged_space_is_not_published") {
            return;
        }
        let (_left, lhs, _right, rhs) = bound_pair(2, 2);
        symtn_core::clear_structure_caches();
        let provider = lhs.provider();
        let sides = [
            (FusionOperand::direct(lhs.space()), lhs.space()),
            (FusionOperand::direct(rhs.space()), rhs.space()),
        ];
        let authority = CheckedAuthority::contraction(&lhs, sides).unwrap();
        let (lhs_axes, rhs_axes, output) = ([2, 3], [0, 1], [1, 0, 2, 3]);
        let destination = lhs
            .prepare_final_homspace_generic_from_checked(provider, || {
                FusionTreeHomSpace::try_tensorcontract_homspace_generic_checked(
                    provider,
                    lhs.space().homspace(),
                    rhs.space().homspace(),
                    &lhs_axes,
                    &rhs_axes,
                    &output,
                    2,
                )
                .map_err(CheckedGenericPlanError::from)
            })
            .unwrap();
        let preview = destination.preview();
        let mut txn = CheckedContractTxn::new();
        let resolution = plan_contract_in::<HostEagerExecutor, CheckedGenericAdmissionMode, _>(
            &mut txn,
            PlanTarget::checked(authority, &preview),
            FusionOperand::direct(lhs.space()),
            FusionOperand::direct(rhs.space()),
            (&lhs_axes, &rhs_axes, &output),
        )
        .unwrap();
        assert!(resolution.copy_c().is_some());
        let declined = || {
            lhs.prepare_final_homspace_generic_from_checked(provider, || {
                lhs.space()
                    .homspace()
                    .try_permute_generic_checked(provider, &[0, 1, 2], &[3])
                    .map_err(CheckedGenericPlanError::from)
            })
            .unwrap()
        };
        drop(txn.stage(declined(), None));
        assert_eq!(cache_entries(), (0, 0));
        txn.commit(
            (
                Arc::clone(preview.structure()),
                Arc::clone(preview.structure()),
            ),
            &resolution,
        );
        // The temporary only. The declined space is dropped, and the output
        // transformer stays unpublished: its destination (this test's
        // uncommitted preview) is not canonical.
        assert_eq!(cache_entries(), (1, 0));
        // A complete-cache hit lends the one resident structure; a miss
        // previews a fresh copy each time.
        let resident = |prepared: PreparedCheckedGenericDynamicSpace| {
            Arc::ptr_eq(&prepared.shared_structure(), &prepared.shared_structure())
        };
        assert!(!resident(declined()));
        let temporary = lhs
            .prepare_final_homspace_generic_from_checked(provider, || {
                Ok::<_, CheckedGenericPlanError<SpyError>>(preview.homspace().clone())
            })
            .unwrap();
        assert!(resident(temporary));
    }

    /// One side of the checked entry: logical space, storage operand and
    /// storage payload.
    type Side<'a> = (&'a SpySpace, FusionOperand<'a>, &'a [f64]);

    /// The lazy adjoint of `parent`: its logical space and `copy(parent')`,
    /// the owned materialization through the eager storage-mapped transform
    /// (its own oracles in `tree_transform_plan::checked_generic`), which no
    /// contraction route reads.
    fn adjoint_of(parent: &SpySpace, data: &[f64]) -> (SpySpace, SpySpace, Vec<f64>) {
        let logical = crate::adjoint_bound_space_dyn_generic_checked(parent).unwrap();
        let (nout, rank) = (logical.space().nout(), logical.space().rank());
        let (owned, owned_data) =
            crate::TreeTransformExecutionContext::<f64, RuleIdentity, f64>::default()
                .tree_transform_owned_checked_generic_in(
                    &logical,
                    Some(parent),
                    data,
                    &crate::TreeTransformOperation::permute(0..nout, nout..rank),
                    1.0,
                )
                .unwrap();
        (logical, owned, owned_data)
    }

    fn assert_close(what: &str, actual: (&SpySpace, &[f64]), expected: (&SpySpace, &[f64])) {
        assert_eq!(actual.0.space(), expected.0.space(), "{what}");
        assert_eq!(actual.1.len(), expected.1.len(), "{what}");
        for (actual, expected) in actual.1.iter().zip(expected.1) {
            assert!(
                (actual - expected).abs() <= 1e-12,
                "{what}: {actual} vs {expected}"
            );
        }
    }

    /// What: a lazy-adjoint operand (lhs, rhs or both) takes the fixture's
    /// route — the canonical core over `MatrixOp::Adjoint`, CopyC, or the
    /// `DynamicTree` artifact, whose adjoint source compiles a storage-mapped
    /// conjugating transformer even when the core borrows it (so `lhs_only`,
    /// whose direct rhs derives none, derives one per adjoint source). Core
    /// and CopyC
    /// equal the contraction of the materialized adjoint; so does
    /// composition. Why not `DynamicTree` values here: the spy's symbols are
    /// not a category, so two valid candidates (the materialized operand may
    /// select another) disagree; `symtn/tests/lazy_adjoint_contract_both_modes.rs`
    /// checks them on SU(2) and SU(3).
    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn checked_adjoint_operands_take_every_route_and_match_the_materialized_parent() {
        if !isolated("checked_adjoint_operands_take_every_route_and_match_the_materialized_parent")
        {
            return;
        }
        for (name, n, lhs_axes, rhs_axes, output, nout, route) in FIXTURES {
            let (_left, lhs, _right, rhs) = bound_pair(n, n);
            let lhs_data = ramp(lhs.space().required_len().unwrap(), 0.25, -1.0);
            let rhs_data = ramp(rhs.space().required_len().unwrap(), -0.5, 1.5);
            let (lhs_logical, lhs_owned, lhs_owned_data) = adjoint_of(&lhs, &lhs_data);
            let (rhs_logical, rhs_owned, rhs_owned_data) = adjoint_of(&rhs, &rhs_data);
            let axes = || fixture_axes(lhs_axes, rhs_axes, output);
            let mut transformers = Vec::new();
            for (lazy_lhs, lazy_rhs) in [(false, false), (true, false), (false, true), (true, true)]
            {
                let what = format!("{name} lazy lhs {lazy_lhs} rhs {lazy_rhs}");
                let lhs_side: Side<'_> = if lazy_lhs {
                    (&lhs_logical, FusionOperand::adjoint(lhs.space()), &lhs_data)
                } else {
                    direct_side(&lhs, &lhs_data)
                };
                let rhs_side: Side<'_> = if lazy_rhs {
                    (&rhs_logical, FusionOperand::adjoint(rhs.space()), &rhs_data)
                } else {
                    direct_side(&rhs, &rhs_data)
                };
                let (lhs_oracle, rhs_oracle) = (
                    if lazy_lhs {
                        direct_side(&lhs_owned, &lhs_owned_data)
                    } else {
                        direct_side(&lhs, &lhs_data)
                    },
                    if lazy_rhs {
                        direct_side(&rhs_owned, &rhs_owned_data)
                    } else {
                        direct_side(&rhs, &rhs_data)
                    },
                );
                let mut context =
                    TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
                crate::probe::take_completed_transformer_activity();
                let (space, data) = context
                    .tensorcontract_checked_generic_in(lhs_side, rhs_side, entry_axes(axes()), nout)
                    .unwrap();
                let activity = crate::probe::take_completed_transformer_activity();
                transformers.push(activity.builds + activity.hits);
                assert_eq!(last_route(&context), route, "{what}");
                if route == Route::DynamicTree {
                    continue;
                }
                let (expected_space, expected) = TensorContractFusionExecutionContext::<
                    f64,
                    RuleIdentity,
                >::default()
                .tensorcontract_checked_generic_in(lhs_oracle, rhs_oracle, entry_axes(axes()), nout)
                .unwrap();
                assert_close(&what, (&space, &data), (&expected_space, &expected));
            }
            if name == "lhs_only" {
                assert_eq!(transformers[0], 1, "{name}: the rhs is borrowed");
                assert_eq!(
                    transformers[3], 2,
                    "{name}: adjoint sources derive transformers"
                );
            }
        }

        let (_left, lhs, _right, rhs) = bound_pair(1, 1);
        let lhs_data = ramp(lhs.space().required_len().unwrap(), 0.25, -1.0);
        let rhs_data = ramp(rhs.space().required_len().unwrap(), -0.5, 1.5);
        let (lhs_logical, lhs_owned, lhs_owned_data) = adjoint_of(&lhs, &lhs_data);
        let lazy: Side<'_> = (&lhs_logical, FusionOperand::adjoint(lhs.space()), &lhs_data);
        let owned = direct_side(&lhs_owned, &lhs_owned_data);
        for (what, actual, expected) in [
            (
                "lazy∘rhs",
                (lazy, direct_side(&rhs, &rhs_data)),
                (owned, direct_side(&rhs, &rhs_data)),
            ),
            (
                "rhs∘lazy",
                (direct_side(&rhs, &rhs_data), lazy),
                (direct_side(&rhs, &rhs_data), owned),
            ),
            (
                "lazy∘lhs (t'∘t)",
                (lazy, direct_side(&lhs, &lhs_data)),
                (owned, direct_side(&lhs, &lhs_data)),
            ),
        ] {
            let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
            let (space, data) = context
                .tensorcompose_checked_generic_in(actual.0, actual.1)
                .unwrap();
            assert_eq!(last_route(&context), Route::Core, "{what}");
            let (expected_space, expected) =
                TensorContractFusionExecutionContext::<f64, RuleIdentity>::default()
                    .tensorcompose_checked_generic_in(expected.0, expected.1)
                    .unwrap();
            assert_close(what, (&space, &data), (&expected_space, &expected));
        }
    }

    /// What (#2063): with a lazy-adjoint lhs, a replay failing on its first
    /// GEMM publishes nothing on any route; a successful call commits, and
    /// its warm repeat builds no transformer, stages nothing new and
    /// returns the same bits.
    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn checked_adjoint_operand_failure_publishes_nothing_and_a_warm_repeat_builds_nothing() {
        if !isolated(
            "checked_adjoint_operand_failure_publishes_nothing_and_a_warm_repeat_builds_nothing",
        ) {
            return;
        }
        for (name, n, lhs_axes, rhs_axes, output, nout, route) in FIXTURES {
            let (_left, lhs, _right, rhs) = bound_pair(n, n);
            let lhs_data = ramp(lhs.space().required_len().unwrap(), 0.25, -1.0);
            let rhs_data = ramp(rhs.space().required_len().unwrap(), -0.5, 1.5);
            symtn_core::clear_structure_caches();
            let logical = crate::adjoint_bound_space_dyn_generic_checked(&lhs).unwrap();
            let axes = || fixture_axes(lhs_axes, rhs_axes, output);
            let run = |context: &mut FailingContext| {
                context.tensorcontract_checked_generic_in(
                    (&logical, FusionOperand::adjoint(lhs.space()), &lhs_data),
                    direct_side(&rhs, &rhs_data),
                    entry_axes(axes()),
                    nout,
                )
            };
            let resident = cache_entries();
            let admissions = coefficient_admissions();
            crate::probe::take_completed_transformer_activity();
            let mut context = failing_context(Some(0));
            let error = run(&mut context).unwrap_err();
            assert!(
                matches!(
                    error,
                    CheckedGenericPlanError::Operation(OperationError::StridedKernel { .. })
                ),
                "{name}"
            );
            assert_eq!(last_route(&context), route, "{name}");
            assert_eq!(cache_entries(), resident, "{name}");
            assert_eq!(coefficient_admissions(), admissions, "{name}");
            let transformers = crate::probe::take_completed_transformer_activity();
            assert_eq!(transformers.publications, 0, "{name}");

            context.contract_backend_mut().fail_at = None;
            let (_, reference) = run(&mut context).unwrap();
            let committed = cache_entries();
            crate::probe::take_completed_transformer_activity();
            let (_, repeat) = run(&mut context).unwrap();
            let warm = crate::probe::take_completed_transformer_activity();
            assert_eq!(warm.builds, 0, "{name}");
            assert_eq!(cache_entries(), committed, "{name}");
            assert_eq!(
                repeat.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{name}"
            );
        }
    }

    /// Resident `(cache 2, cache 3)` entries.
    fn cache_entries() -> (usize, usize) {
        let entries = |kind| symtn_core::structure_cache_info(kind).entries();
        (
            entries(symtn_core::StructureCacheKind::DegeneracyStructure),
            entries(symtn_core::StructureCacheKind::CompletedTreeTransformer),
        )
    }

    fn coefficient_admissions() -> u64 {
        symtn_core::structure_cache_info(symtn_core::StructureCacheKind::TreeTransformCoefficients)
            .admissions()
    }

    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn a_warm_checked_generic_contraction_converts_at_most_one_pack_per_stage() {
        // Isolated: a concurrent cache clear would force a rebuild.
        const ISOLATED: &str = "SYMTN_CHECKED_GENERIC_CONTRACT_PACKS_ISOLATED";
        if std::env::var_os(ISOLATED).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "contract::checked_generic::tests::a_warm_checked_generic_contraction_converts_at_most_one_pack_per_stage",
                ])
                .env(ISOLATED, "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "isolated pack test failed:\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }

        // Why (#2101): each stage replays into its own context workspace, so
        // no stage evicts another's Multi pack: a warm call converts at most
        // one pack per stage (a shared workspace would convert three into one
        // slot). #2063 tightens this to none: a warm call binds the completed
        // transformers of the first call, whose packs are already converted.
        let (_left, lhs, _right, rhs) = bound_pair(2, 2);
        let lhs_data = (0..lhs.space().required_len().unwrap())
            .map(|index| index as f64 - 1.5)
            .collect::<Vec<_>>();
        let rhs_data = (0..rhs.space().required_len().unwrap())
            .map(|index| 2.0 - index as f64)
            .collect::<Vec<_>>();
        let mut context =
            TensorContractFusionExecutionContext::<f64, symtn_core::RuleIdentity>::default();
        let mut run = || {
            let (_, data) = context
                .tensorcontract_checked_generic_in(
                    direct_side(&lhs, &lhs_data),
                    direct_side(&rhs, &rhs_data),
                    entry_axes(TensorContractSpec::new(
                        &[3, 1],
                        &[0, 3],
                        symtn_operations::OutputAxisOrder::Axes(&[2, 0, 3, 1]),
                    )),
                    2,
                )
                .unwrap();
            (data, context.eager_coefficient_pack_builds())
        };
        let (cold_data, cold) = run();
        assert_eq!(cold, [1, 1, 1], "all three stages recouple");
        let mut previous = cold;
        for _ in 0..3 {
            let (data, builds) = run();
            assert_eq!(data, cold_data);
            assert_eq!(builds, previous, "a warm call converts no pack");
            previous = builds;
        }
    }

    /// Rank-1 legs with the listed `(sector, degeneracy)` pairs on each side.
    fn homspace_with(codomain: &[(usize, usize)], domain: &[(usize, usize)]) -> FusionTreeHomSpace {
        let leg = |sectors: &[(usize, usize)]| {
            SectorLeg::new(
                sectors
                    .iter()
                    .map(|&(sector, degeneracy)| (SectorId::new(sector), degeneracy)),
                false,
            )
        };
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(codomain)]),
            FusionProductSpace::new([leg(domain)]),
        )
    }

    #[allow(clippy::arc_with_non_send_sync)]
    fn bound_space(homspace: FusionTreeHomSpace) -> BoundDynamicFusionMapSpace<CheckedGenericSpy> {
        BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
            Arc::new(CheckedGenericSpy::new()),
            homspace,
        )
        .unwrap()
    }

    fn assert_inactive_sector_zero_and_active_product<D>(
        one: D,
        two: D,
        six: D,
        is_positive_zero: fn(D) -> bool,
    ) where
        D: DenseRecouplingScalar
            + RecouplingCoefficientAction<f64>
            + ConjugateValue
            + Copy
            + Zero
            + ZeroBytes
            + std::fmt::Debug,
    {
        // lhs `[0:1, 1:2] <- [1:3]` and rhs `[1:3] <- [0:1, 1:2]` couple only
        // to sector 1; the destination `[0:1, 1:2] <- [0:1, 1:2]` also has the
        // sector-0 block, which no GEMM writes.
        let lhs = bound_space(homspace_with(&[(0, 1), (1, 2)], &[(1, 3)]));
        let rhs = bound_space(homspace_with(&[(1, 3)], &[(0, 1), (1, 2)]));
        let lhs_data = vec![one; lhs.space().required_len().unwrap()];
        let rhs_data = vec![two; rhs.space().required_len().unwrap()];
        let mut context = TensorContractFusionExecutionContext::<D, RuleIdentity>::default();
        let (output, data) = context
            .tensorcontract_checked_generic_in(
                direct_side(&lhs, &lhs_data),
                direct_side(&rhs, &rhs_data),
                entry_axes(TensorContractSpec::with_default_output_order(&[1], &[0])),
                1,
            )
            .unwrap();
        let structure = output.space().structure();
        assert_eq!(structure.block_count(), 2);
        assert_eq!(data.len(), 1 + 4);
        for index in 0..structure.block_count() {
            // The sector-0 block is the 1x1 one, the sector-1 block the 2x2 one.
            let block = structure.block(index).unwrap();
            let range = block.offset()..block.offset() + block.element_count().unwrap();
            match range.len() {
                1 => assert!(data[range].iter().all(|&v| is_positive_zero(v))),
                4 => assert!(data[range].iter().all(|&v| v == six)),
                other => panic!("unexpected block length {other}"),
            }
        }
    }

    #[test]
    fn owned_checked_generic_contract_leaves_inactive_sectors_exactly_zero() {
        assert_inactive_sector_zero_and_active_product::<f64>(1.0, 2.0, 6.0, |v| v.to_bits() == 0);
        assert_inactive_sector_zero_and_active_product::<num_complex::Complex64>(
            num_complex::Complex64::new(1.0, 0.0),
            num_complex::Complex64::new(2.0, 0.0),
            num_complex::Complex64::new(6.0, 0.0),
            |v| v.re.to_bits() == 0 && v.im.to_bits() == 0,
        );
    }

    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn a_reset_straddling_the_publication_leaves_nothing_resident() {
        const ISOLATED: &str = "SYMTN_CHECKED_GENERIC_CONTRACT_RESET_STRADDLE_ISOLATED";
        if std::env::var_os(ISOLATED).is_none() {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "contract::checked_generic::tests::a_reset_straddling_the_publication_leaves_nothing_resident",
                ])
                .env(ISOLATED, "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "isolated reset-straddle test failed:\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }

        // What: a clear between the commits and the first transformer
        // publication refuses every staged publication (the call's epoch is
        // older), the result is unchanged, and the next call rebuilds.
        let (_left, lhs, _right, rhs) = bound_pair(2, 2);
        let lhs_data = (0..lhs.space().required_len().unwrap())
            .map(|index| index as f64 - 1.5)
            .collect::<Vec<_>>();
        let rhs_data = (0..rhs.space().required_len().unwrap())
            .map(|index| 2.0 - index as f64)
            .collect::<Vec<_>>();
        let mut context =
            TensorContractFusionExecutionContext::<f64, symtn_core::RuleIdentity>::default();
        let mut run = || {
            context
                .tensorcontract_checked_generic_in(
                    direct_side(&lhs, &lhs_data),
                    direct_side(&rhs, &rhs_data),
                    entry_axes(TensorContractSpec::new(
                        &[3, 1],
                        &[0, 3],
                        symtn_operations::OutputAxisOrder::Axes(&[2, 0, 3, 1]),
                    )),
                    2,
                )
                .unwrap()
                .1
        };
        symtn_core::clear_structure_caches();
        let reference = run();
        symtn_core::clear_structure_caches();
        crate::probe::take_completed_transformer_activity();
        crate::probe::set_before_publish_hook(Box::new(symtn_core::clear_structure_caches));
        let straddled = run();
        let transformers = crate::probe::take_completed_transformer_activity();
        assert_eq!((transformers.builds, transformers.publications), (3, 3));
        assert_eq!(cache_entries().1, 0);
        assert_eq!(straddled, reference);
        let rebuilt = run();
        let transformers = crate::probe::take_completed_transformer_activity();
        assert_eq!((transformers.hits, transformers.builds), (0, 3));
        assert_eq!(rebuilt, reference);
    }

    /// Bits of a payload with ±0 folded (platform-portable).
    fn folded_bits(data: &[f64]) -> Vec<u64> {
        data.iter()
            .map(|&value| if value == 0.0 { 0 } else { value.to_bits() })
            .collect()
    }

    /// One owned checked transform on a fresh default context.
    fn transform(
        logical: &SpySpace,
        adjoint_of: Option<&SpySpace>,
        data: &[f64],
        operation: &crate::TreeTransformOperation,
    ) -> Result<(SpySpace, Vec<f64>), CheckedGenericPlanError<SpyError>> {
        crate::TreeTransformExecutionContext::<f64, RuleIdentity, f64>::default()
            .tree_transform_owned_checked_generic_in(logical, adjoint_of, data, operation, 1.0)
    }

    /// What (#2154): a warm owned checked permute, braid or transpose, of a
    /// direct source or of a lazy adjoint, takes its destination from the
    /// source's memo and its transformer from the completed cache: its
    /// provider events are the identity read, the admission style and the
    /// commit style, with no algebra query of the cold call's, and its
    /// payload is the cold call's.
    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn a_warm_checked_transform_reads_only_its_identity_and_styles() {
        if !isolated("a_warm_checked_transform_reads_only_its_identity_and_styles") {
            return;
        }
        use crate::TreeTransformOperation;
        let operations = [
            ("permute", TreeTransformOperation::permute([1, 0], [2, 3])),
            (
                "braid",
                TreeTransformOperation::braid([2, 0], [1, 3], [0, 1], [2, 3]),
            ),
            (
                "transpose",
                TreeTransformOperation::transpose([0, 1, 3], [2]),
            ),
        ];
        for adjoint in [false, true] {
            for (name, operation) in &operations {
                let what = format!("{name} adjoint={adjoint}");
                symtn_core::clear_structure_caches();
                let (provider, parent, _, _) = bound_pair(2, 2);
                let data = ramp(parent.space().required_len().unwrap(), 0.5, -1.0);
                let lazy = adjoint
                    .then(|| crate::adjoint_bound_space_dyn_generic_checked(&parent).unwrap());
                let (logical, adjoint_of) = match &lazy {
                    Some(lazy) => (lazy, Some(&parent)),
                    None => (&parent, None),
                };
                let run = || {
                    provider.reset();
                    let (space, out) = transform(logical, adjoint_of, &data, operation).unwrap();
                    (space, out, provider.events.borrow().clone())
                };
                let (cold_space, cold_data, _) = run();
                assert!(
                    provider.algebra_calls() > 0,
                    "{what}: the cold call derives"
                );
                for call in 0..3 {
                    let (space, out, events) = run();
                    assert_eq!(events, [Identity, Style, Style], "{what} warm {call}");
                    assert_eq!(provider.algebra_calls(), 0, "{what} warm {call}");
                    assert_eq!(space.space(), cold_space.space(), "{what} warm {call}");
                    assert_eq!(folded_bits(&out), folded_bits(&cold_data), "{what} {call}");
                }
            }
        }
    }

    /// What (#2154): a memo hit whose operation has no completed transformer
    /// yet (a permute after a braid on the same axes) runs the source
    /// preflight inside the structure build. Its ledger is the derived
    /// path's without the destination's `Dual` queries, its payload is
    /// equal, and a non-symmetric braiding fails at the same event with the
    /// same error. Oracle: a space admitted before a reset retains no entry,
    /// so it neither publishes nor reads a memo.
    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn a_memo_hit_with_a_new_operation_runs_the_preflight_in_the_build() {
        if !isolated("a_memo_hit_with_a_new_operation_runs_the_preflight_in_the_build") {
            return;
        }
        use crate::TreeTransformOperation;
        // Legs cross between codomain and domain, so the derivation dualizes.
        let braid = TreeTransformOperation::braid([2, 0], [1, 3], [0, 1], [2, 3]);
        let permute = TreeTransformOperation::permute([2, 0], [1, 3]);
        for braiding in [BraidingStyleKind::Bosonic, BraidingStyleKind::Anyonic] {
            let [derived, memo] = [false, true].map(|memo| {
                if memo {
                    symtn_core::clear_structure_caches();
                }
                let (provider, space, _, _) = bound_pair(2, 2);
                if !memo {
                    symtn_core::clear_structure_caches();
                }
                provider.braiding.set(braiding);
                let data = ramp(space.space().required_len().unwrap(), 0.5, -1.0);
                transform(&space, None, &data, &braid).unwrap();
                provider.reset();
                let result = transform(&space, None, &data, &permute)
                    .map(|(space, out)| (space.space().clone(), folded_bits(&out)));
                let events = provider.events.borrow().clone();
                (result, events)
            });
            // The derived ledger is the memo's with one run of `Dual`
            // queries (the destination's derivation) where they diverge.
            let split = derived
                .1
                .iter()
                .zip(&memo.1)
                .take_while(|(derived, memo)| derived == memo)
                .count();
            let dropped = derived.1[split..]
                .iter()
                .take_while(|&&event| event == Event::Query(Query::Dual))
                .count();
            assert_eq!(derived.1[..split], memo.1[..split], "{braiding:?}");
            assert_eq!(
                derived.1[split + dropped..],
                memo.1[split..],
                "{braiding:?}"
            );
            match (derived.0, memo.0) {
                (Ok(derived), Ok(memo)) => {
                    assert_eq!(braiding, BraidingStyleKind::Bosonic);
                    assert!(dropped > 0);
                    assert_eq!(derived, memo);
                }
                (Err(derived), Err(memo)) => {
                    assert_eq!(braiding, BraidingStyleKind::Anyonic);
                    assert_eq!(dropped, 0);
                    assert!(
                        matches!(
                            &derived,
                            CheckedGenericPlanError::Operation(
                                OperationError::UnsupportedBraidingStyle {
                                    style: BraidingStyleKind::Anyonic,
                                    ..
                                }
                            )
                        ),
                        "{derived:?}"
                    );
                    assert_eq!(format!("{derived:?}"), format!("{memo:?}"));
                }
                (derived, memo) => panic!("{braiding:?}: {derived:?} vs {memo:?}"),
            }
        }
    }

    /// What (#2157): a warm checked contraction stages each permuted operand
    /// from the operand's destination memo (#2154), as a warm transform
    /// does, on every route, direct or with a lazy-adjoint lhs. Its provider
    /// events are the pair admission and braiding read, the destination
    /// stage, one identity read and admission style per staged permute (an
    /// identity read per transformer lookup) and the commit style; the only
    /// algebra queries left are the `Dual`s of the contraction destination's
    /// own derivation (`three_stages`), which no operand memo covers. The
    /// payload is the cold call's, and a fixture that stages an operand
    /// derives it on the cold call only.
    #[test]
    #[allow(clippy::arc_with_non_send_sync)]
    fn a_warm_checked_contraction_derives_no_staged_operand() {
        if !isolated("a_warm_checked_contraction_derives_no_staged_operand") {
            return;
        }
        use Event::Query as Q;
        let (i, s, b, d) = (Identity, Style, Braiding, Q(Query::Dual));
        // Per fixture: whether it stages an operand, then the warm
        // lhs-provider events, direct and adjoint.
        type Pin = (&'static str, bool, [Vec<Event>; 2]);
        let pins: [Pin; 4] = [
            (
                "core",
                false,
                [vec![i, s, b, i, s, s], vec![i, s, b, i, s, s]],
            ),
            (
                "copy_c",
                true,
                [
                    vec![i, s, b, i, s, i, s, i, s],
                    vec![i, s, b, i, s, i, s, i, s],
                ],
            ),
            (
                "lhs_only",
                // The adjoint row admits its source structure once (#2233).
                true,
                [
                    vec![i, s, b, i, s, i, s, i, s],
                    vec![i, s, b, i, s, i, i, s, i, s],
                ],
            ),
            (
                "three_stages",
                // Both operands share one canonical source structure, whose
                // trees are admitted once per call (#2233).
                true,
                [
                    vec![i, s, b, i, d, d, s, i, s, i, i, s, i, i, s, i, s],
                    vec![i, s, b, i, d, d, s, i, s, i, i, s, i, i, s, i, s],
                ],
            ),
        ];
        for adjoint in [false, true] {
            for ((name, n, lhs_axes, rhs_axes, output, nout, route), pin) in
                FIXTURES.iter().zip(&pins)
            {
                assert_eq!(*name, pin.0);
                let what = format!("{name} adjoint={adjoint}");
                symtn_core::clear_structure_caches();
                let (left, lhs, right, rhs) = bound_pair(*n, *n);
                let lhs_data = ramp(lhs.space().required_len().unwrap(), 0.25, -1.0);
                let rhs_data = ramp(rhs.space().required_len().unwrap(), -0.5, 1.5);
                let logical =
                    adjoint.then(|| crate::adjoint_bound_space_dyn_generic_checked(&lhs).unwrap());
                let lhs_side = match &logical {
                    Some(logical) => (logical, FusionOperand::adjoint(lhs.space()), &lhs_data[..]),
                    None => direct_side(&lhs, &lhs_data),
                };
                let run = || {
                    left.reset();
                    right.reset();
                    let mut context =
                        TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
                    let (space, out) = context
                        .tensorcontract_checked_generic_in(
                            lhs_side,
                            direct_side(&rhs, &rhs_data),
                            entry_axes(fixture_axes(lhs_axes, rhs_axes, *output)),
                            *nout,
                        )
                        .unwrap();
                    assert_eq!(last_route(&context), *route, "{what}");
                    assert_eq!(right.algebra_calls(), 0, "{what}");
                    assert_eq!(*right.events.borrow(), [Identity, Style], "{what}");
                    let events = left.events.borrow().clone();
                    (space, out, events)
                };
                let (cold_space, cold_data, _) = run();
                let cold_algebra = left.algebra_calls();
                for call in 0..3 {
                    let (space, out, events) = run();
                    let warm_algebra = left.algebra_calls();
                    if pin.1 {
                        assert!(warm_algebra < cold_algebra, "{what} warm {call}");
                    } else {
                        assert_eq!(warm_algebra, cold_algebra, "{what} warm {call}");
                    }
                    assert_eq!(events, pin.2[usize::from(adjoint)], "{what} warm {call}");
                    assert_eq!(space.space(), cold_space.space(), "{what} warm {call}");
                    assert_eq!(folded_bits(&out), folded_bits(&cold_data), "{what} {call}");
                }
            }
        }
    }
}
