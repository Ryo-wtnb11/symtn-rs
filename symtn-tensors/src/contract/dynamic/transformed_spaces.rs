//! The multiplicity-free transformed-source and core-destination
//! derivations of a dynamic-tree contraction. Nothing here retains anything:
//! the derived spaces come from the complete-HomSpace cache and the
//! transformers from the process-global completed-transformer cache.

use super::*;
use crate::tree_transform::TreeTransformPlanning;
use symtn_dense::Engine;

#[derive(Clone, Debug)]
pub(crate) struct DynamicFusionTransformedSourceEntry<C = f64> {
    pub(crate) space: Arc<DynamicFusionMapSpace>,
    pub(crate) replay_structure: Arc<BlockStructure>,
    /// `None` only for a source the checked planner borrows without
    /// deriving anything (an identity permutation); every non-borrowed
    /// source has one (checked when the artifact is assembled).
    pub(crate) transform_structure: Option<TreeTransformStructure<C>>,
    /// The core operand is exactly the adjoint of the payload
    /// `replay_structure` describes: a conjugated source under an identity
    /// transform, which the core may read in place with a GEMM adjoint op
    /// (TensorKit `blas_contract!` without `copyA`/`copyB`).
    pub(crate) core_is_storage_adjoint: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct DynamicFusionCoreDstEntry<C = f64> {
    pub(crate) space: Arc<DynamicFusionMapSpace>,
    pub(crate) output_transform_structure: TreeTransformStructure<C>,
}

/// The core-layout space of a stored source and its transformer. A
/// conjugated source is read as its storage-ordered lazy adjoint, keyed on
/// the parent's storage: a per-call adjoint view would take a fresh content
/// id on every call and only evict live transformers.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compile_transformed_source<R>(
    engine: &mut dyn Engine,
    planning: &mut TreeTransformPlanning,
    rule: &R,
    src_space: &DynamicFusionMapSpace,
    src_storage_structure: &Arc<BlockStructure>,
    operation: &TreeTransformOperation,
    source_conjugate: bool,
    layout_primer: LayoutKeyBuilder<R>,
) -> Result<DynamicFusionTransformedSourceEntry<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols + TreeTransformRuleCacheKey,
    R::Scalar: MultiplicityFreePlanningScalar,
{
    if source_conjugate {
        let oriented = crate::FusionOperand::prepare_storage_ordered_adjoint(src_space, rule)?;
        return compile_transformed_source_oriented(
            engine,
            planning,
            rule,
            &oriented,
            operation,
            layout_primer,
        );
    }
    let space = src_space.transformed_with_primer(rule, operation, layout_primer)?;
    let dst_structure = Arc::clone(space.structure());
    let transform_structure =
        <MultiplicityFreeAdmissionMode as PlanningAlgebra<R>>::tree_structure(
            engine,
            planning,
            rule,
            operation,
            &dst_structure,
            TreeStructureSource::Stored {
                structure: src_storage_structure,
                storage_conjugate: false,
            },
        )?;
    Ok(DynamicFusionTransformedSourceEntry {
        space: Arc::new(space),
        replay_structure: Arc::clone(src_storage_structure),
        transform_structure: Some(transform_structure),
        core_is_storage_adjoint: false,
    })
}

/// The core-layout space of a lazy adjoint read through its parent's
/// storage, and its oriented transformer.
pub(super) fn compile_transformed_source_oriented<R>(
    engine: &mut dyn Engine,
    planning: &mut TreeTransformPlanning,
    rule: &R,
    source: &FusionOperandLayout<'_>,
    operation: &TreeTransformOperation,
    layout_primer: LayoutKeyBuilder<R>,
) -> Result<DynamicFusionTransformedSourceEntry<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols + TreeTransformRuleCacheKey,
    R::Scalar: MultiplicityFreePlanningScalar,
{
    let space = source.transformed_space(rule, operation, layout_primer)?;
    let dst_structure = Arc::clone(space.structure());
    let transform_structure =
        <MultiplicityFreeAdmissionMode as PlanningAlgebra<R>>::tree_structure(
            engine,
            planning,
            rule,
            operation,
            &dst_structure,
            TreeStructureSource::Oriented(source),
        )?;
    Ok(DynamicFusionTransformedSourceEntry {
        space: Arc::new(space),
        replay_structure: Arc::clone(source.storage_space().structure()),
        transform_structure: Some(transform_structure),
        core_is_storage_adjoint: source_is_storage_adjoint(
            !source.is_direct(),
            source.nout(),
            source.rank(),
            operation.codomain_permutation(),
            operation.domain_permutation(),
        ),
    })
}

/// The core destination space and the output transformer into `output_dst`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compile_core_dst<R>(
    engine: &mut dyn Engine,
    planning: &mut TreeTransformPlanning,
    rule: &R,
    lhs: &DynamicFusionMapSpace,
    rhs: &DynamicFusionMapSpace,
    plan: &FusionContractPlan,
    output_dst: &DynamicFusionMapSpace,
    layout_primer: LayoutKeyBuilder<R>,
) -> Result<DynamicFusionCoreDstEntry<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols + TreeTransformRuleCacheKey,
    R::Scalar: MultiplicityFreePlanningScalar,
{
    let space = DynamicFusionMapSpace::core_dst_with_primer(rule, lhs, rhs, plan, layout_primer)?;
    let src_structure = Arc::clone(space.structure());
    let output_transform_structure =
        <MultiplicityFreeAdmissionMode as PlanningAlgebra<R>>::tree_structure(
            engine,
            planning,
            rule,
            plan.output_transform(),
            output_dst.structure(),
            TreeStructureSource::Stored {
                structure: &src_structure,
                storage_conjugate: false,
            },
        )?;
    Ok(DynamicFusionCoreDstEntry {
        space: Arc::new(space),
        output_transform_structure,
    })
}
