use std::sync::Arc;

use symtn_core::{
    BlockStructure, CategoricalScalar, FusionTreeHomSpace, FusionTreePairOrientation,
    MultiplicityFreeRigidSymbols,
};

use crate::mode::{MultiplicityFreePlanningScalar, PlanningAlgebra, TreeStructureSource};
#[cfg(test)]
use crate::tree_context::TreeTransformExecutionContext;
use crate::tree_transform::TreeTransformPlanning;
#[cfg(test)]
use crate::DenseTreeTransformOperations;
use crate::{
    DenseBlockScalar, OperationError, TreeTransformOperation, TreeTransformOperationKind,
    TreeTransformRuleCacheKey, TreeTransformStructure,
};
#[cfg(test)]
use crate::{DenseRecouplingScalar, RecouplingCoefficientAction, TreeTransformBackend};
use symtn_core::MultiplicityFreeAdmissionMode;
use symtn_operations::fusion_replay::{FusionBlockContractPlan, MatrixOp};

#[cfg(test)]
use super::backend::TensorContractBackend;
#[cfg(test)]
use super::dynamic_space::encoded_layout_primer;
use super::dynamic_space::{DynamicFusionMapSpace, FusionOperandLayout, LayoutKeyBuilder};
use super::fusion::{
    contract_twist_on_physical_lhs, FusionContractOrientation, FusionContractPlan,
};
#[cfg(test)]
use super::fusion_block::FusionBlockContractWorkspace;
use super::resolution::{contract_axes_require_twist, rhs_contract_requires_twist};

#[cfg(feature = "cuda")]
pub(crate) mod cuda;

use super::fusion;
#[cfg(test)]
use super::fusion_block;
#[cfg(test)]
use super::{backend, dynamic_space};

mod artifact;
#[cfg(test)]
mod test_entry;
#[cfg(test)]
mod tests;
mod transformed_spaces;
mod twist;
// Why test-only: the typed plan-level executor is a reference oracle for the
// planner's DynamicTree artifact; production routes through `plan_contract`.
#[cfg(test)]
mod typed_eager;
pub(crate) use artifact::*;
#[cfg(test)]
pub(crate) use test_entry::*;
use transformed_spaces::*;
pub(crate) use transformed_spaces::{
    compile_core_dst, compile_transformed_source, DynamicFusionCoreDstEntry,
    DynamicFusionTransformedSourceEntry,
};
pub(crate) use twist::compile_contract_twist;
pub(super) use twist::*;
#[cfg(test)]
pub(crate) use typed_eager::*;

pub(super) fn source_layout_metadata_is_borrowable<HomSpaceMatches>(
    source_space: &DynamicFusionMapSpace,
    core_nout: usize,
    core_rank: usize,
    homspace_matches: HomSpaceMatches,
    operation: &TreeTransformOperation,
    source_conjugate: bool,
) -> bool
where
    HomSpaceMatches: FnOnce() -> bool,
{
    // An identity transpose is the identity map, as an identity permute is.
    operation.kind() != TreeTransformOperationKind::Braid
        && source_layout_permutation_is_borrowable(
            source_space,
            core_nout,
            core_rank,
            homspace_matches,
            operation.codomain_permutation(),
            operation.domain_permutation(),
            source_conjugate,
        )
}

/// Whether the core reads a source as the adjoint of its parent's storage: a
/// conjugated (lazy adjoint) source whose core transform is the identity of
/// its logical axes, `logical_nout` of `logical_rank` in its codomain
/// (TensorKit `has_shared_permute(::AdjointTensorMap)` delegating to the
/// parent). The one predicate behind both transformed-source compilers'
/// `core_is_storage_adjoint` and the Complete candidate scorer's free side;
/// the artifact borrows it only after `storage_adjoint_core_plan`'s
/// canonical coupled-region proof, which the scorer mirrors with
/// [`storage_layout_is_canonical`].
pub(crate) fn source_is_storage_adjoint(
    source_conjugate: bool,
    logical_nout: usize,
    logical_rank: usize,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> bool {
    source_conjugate
        && codomain_permutation.iter().copied().eq(0..logical_nout)
        && domain_permutation
            .iter()
            .copied()
            .eq(logical_nout..logical_rank)
}

/// [`source_layout_metadata_is_borrowable`] for a permutation given as its
/// two axis lists, before any `TreeTransformOperation` value exists.
pub(super) fn source_layout_permutation_is_borrowable<HomSpaceMatches>(
    source_space: &DynamicFusionMapSpace,
    core_nout: usize,
    core_rank: usize,
    homspace_matches: HomSpaceMatches,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
    source_conjugate: bool,
) -> bool
where
    HomSpaceMatches: FnOnce() -> bool,
{
    if source_conjugate {
        return false;
    }
    if !codomain_permutation
        .iter()
        .copied()
        .eq(0..source_space.nout())
        || !domain_permutation
            .iter()
            .copied()
            .eq(source_space.nout()..source_space.rank())
    {
        return false;
    }
    if core_nout != source_space.nout() || core_rank != source_space.rank() || !homspace_matches() {
        return false;
    }
    true
}

fn source_layout_homspaces_match_by_id(
    source_space: &DynamicFusionMapSpace,
    core_space: &DynamicFusionMapSpace,
) -> bool {
    #[cfg(test)]
    crate::probe::hit(crate::probe::Event::SourceLayoutHomspaceIdComparison);
    core_space.homspace().id() == source_space.homspace().id()
}

fn source_is_borrowable_core_layout(
    source_space: &DynamicFusionMapSpace,
    source_structure: &Arc<BlockStructure>,
    core_space: &DynamicFusionMapSpace,
    operation: &TreeTransformOperation,
    source_conjugate: bool,
) -> bool {
    if !source_layout_metadata_is_borrowable(
        source_space,
        core_space.nout(),
        core_space.rank(),
        || source_layout_homspaces_match_by_id(source_space, core_space),
        operation,
        source_conjugate,
    ) {
        return false;
    }
    // Why not compare only the source's declared structure: even identity axes
    // can complete a sparse fusion-tree grid with structural-zero core blocks.
    structure_is_layout(core_space.structure(), source_structure)
}

/// Whether `structure` is laid out exactly as `layout`: the executor's test
/// for reading a direct source in place, and (through
/// [`storage_layout_is_canonical`]) the candidate scorer's.
fn structure_is_layout(layout: &Arc<BlockStructure>, structure: &Arc<BlockStructure>) -> bool {
    Arc::ptr_eq(layout, structure)
        || layout.content_id() == structure.content_id()
        // Why not rely on content ids alone: an intern reset can assign a new
        // monotonic id to equal live content while an operation cache pins both.
        || layout.as_ref() == structure.as_ref()
}

/// Whether a multiplicity-free space's storage is its HomSpace's canonical
/// coupled-sector layout, the core layout an identity transform derives.
/// Under it the executor reads a direct source in place
/// ([`source_is_borrowable_core_layout`]), and a storage adjoint passes
/// `storage_adjoint_core_plan`'s proof against canonical partners; the
/// Complete candidate scorer asks the same question of each operand.
pub(crate) fn storage_layout_is_canonical<R>(
    rule: &R,
    space: &DynamicFusionMapSpace,
) -> Result<bool, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
{
    // Why the residency flag first: a warm eager call then reads one atomic,
    // not the complete-HomSpace cache. A flagged structure is (or was) the
    // resident canonical layout of a HomSpace with this Complete space's
    // trees and shapes, hence this one's.
    if space.structure().is_canonical() {
        return Ok(true);
    }
    let canonical = space
        .homspace()
        .coupled_subblock_structure_from_leg_degeneracies(rule)
        .map_err(OperationError::from_core_preserving_context)?;
    Ok(structure_is_layout(&canonical, space.structure()))
}

/// Which physical operands are read in place, and which one carries the
/// fermionic contraction twist when there is one.
#[derive(Clone, Copy, Debug)]
struct SourceBorrowing {
    lhs_borrowed: bool,
    rhs_borrowed: bool,
    twist_lhs: bool,
}

/// Borrowing once the fermionic contraction twist, if any, claims one
/// operand: the one the layout already copies when exactly one is copied
/// (so no extra materialization), otherwise the smaller
/// ([`contract_twist_on_physical_lhs`]). A twisted operand is never borrowed.
fn resolve_source_borrowing<R>(
    rule: &R,
    plan: &FusionContractPlan,
    lhs_core: &DynamicFusionMapSpace,
    rhs_core: &DynamicFusionMapSpace,
    lhs_layout_borrowable: bool,
    rhs_layout_borrowable: bool,
) -> Result<SourceBorrowing, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
{
    let reverse = plan.orientation() == FusionContractOrientation::RhsLhs;
    let core_right = if reverse { lhs_core } else { rhs_core };
    let twisted = plan.supertrace_twist()
        && rhs_contract_requires_twist(rule, core_right, plan.core_axes().as_spec())?;
    source_borrowing(
        twisted,
        plan,
        lhs_core,
        rhs_core,
        lhs_layout_borrowable,
        rhs_layout_borrowable,
    )
}

/// [`resolve_source_borrowing`] in admission mode `M`.
fn resolve_source_borrowing_in<M, R>(
    rule: &R,
    authority: M::SpaceAuthority<'_>,
    plan: &FusionContractPlan,
    lhs_core: &DynamicFusionMapSpace,
    rhs_core: &DynamicFusionMapSpace,
    lhs_layout_borrowable: bool,
    rhs_layout_borrowable: bool,
) -> Result<SourceBorrowing, M::Error>
where
    M: PlanningAlgebra<R>,
{
    let reverse = plan.orientation() == FusionContractOrientation::RhsLhs;
    let core_right = if reverse { lhs_core } else { rhs_core };
    let twisted = plan.supertrace_twist()
        && contract_axes_require_twist::<M, R>(
            rule,
            authority,
            core_right.homspace(),
            plan.core_axes().as_spec().rhs_contracting_axes(),
        )?;
    Ok(source_borrowing(
        twisted,
        plan,
        lhs_core,
        rhs_core,
        lhs_layout_borrowable,
        rhs_layout_borrowable,
    )?)
}

fn source_borrowing(
    core_right_requires_twist: bool,
    plan: &FusionContractPlan,
    lhs_core: &DynamicFusionMapSpace,
    rhs_core: &DynamicFusionMapSpace,
    lhs_layout_borrowable: bool,
    rhs_layout_borrowable: bool,
) -> Result<SourceBorrowing, OperationError> {
    if !core_right_requires_twist {
        return Ok(SourceBorrowing {
            lhs_borrowed: lhs_layout_borrowable,
            rhs_borrowed: rhs_layout_borrowable,
            twist_lhs: false,
        });
    }
    let reverse = plan.orientation() == FusionContractOrientation::RhsLhs;
    let required_len = |space: &DynamicFusionMapSpace| {
        space
            .required_len()
            .map_err(OperationError::from_core_preserving_context)
    };
    let twist_lhs = contract_twist_on_physical_lhs(
        reverse,
        !lhs_layout_borrowable,
        !rhs_layout_borrowable,
        || Ok((required_len(lhs_core)?, required_len(rhs_core)?)),
    )?;
    Ok(SourceBorrowing {
        lhs_borrowed: lhs_layout_borrowable && !twist_lhs,
        rhs_borrowed: rhs_layout_borrowable && twist_lhs,
        twist_lhs,
    })
}
