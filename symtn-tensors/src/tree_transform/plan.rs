use core::ops::{Add, Mul};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::{borrow::Cow, collections::hash_map::Entry, sync::Arc};

use num_traits::Zero;
#[cfg(test)]
use symtn_core::{
    generic_braid_tree_pair_block_ordered, generic_permute_tree_pair_block_ordered,
    generic_transpose_tree_pair_block_ordered, GenericRigidSymbols,
};
use symtn_core::{
    multiplicity_free_braid_tree_pair_block_ordered_indexed,
    multiplicity_free_transpose_tree_pair_block_ordered_indexed, BlockKey, BlockKeyKind,
    BlockStructure, CategoricalScalar, CheckedGenericAdmittedFusionTreeBlockStructure,
    CheckedGenericFusion, CheckedGenericRigidSymbols, CheckedGenericStructureError,
    CheckedGenericSymbolError, CoreError, FusionRule, FusionStyleKind, FusionTreeBlockGroup,
    FusionTreeGroupKey, FusionTreeKey, FusionTreePairKey, FusionTreePairOrientation,
    LocallyValidatedFusionTreeBlockStructure, MultiplicityFreeFusionSymbols,
    MultiplicityFreeRigidSymbols, OrderedBlockLinearMap, OrderedBlockLinearStorage,
    PreparedTreePairOperation, RuleIdentity, UniqueTreePairScratch,
};

use crate::OperationError;

use super::cache::{CoefficientGroupReuse, GroupSlot, OrientedBasisOrder, SourceGroup};
use super::operation::{
    TreeTransformOperation, TreeTransformOperationKind, ValidateBraidingSupport,
};
use symtn_operations::transform_plan::SharedTreeTransformGroupPlan;

struct OrientedGroup<'k> {
    // Storage order uses the parent's group indices directly; canonical order
    // needs the logical-to-storage bijection already admitted by the operand.
    logical_indices: Option<Vec<usize>>,
    storage_indices: Cow<'k, [usize]>,
    source_split: (usize, usize),
}

impl OrientedGroup<'_> {
    fn logical_indices(&self) -> &[usize] {
        self.logical_indices
            .as_deref()
            .unwrap_or(&self.storage_indices)
    }
}

pub use symtn_operations::transform_plan::{
    TreeTransformBlockSpec, TreeTransformGroupBlockSpec, TreeTransformGroupPlan,
    TreeTransformKeyBlockSpec,
};

/// Error from fallible Generic-fusion plan compilation.
///
/// Provider failures preserve their concrete source.  Symbol-shape failures
/// are symtn validation failures, rather than provider failures: a returned
/// F/R block exists but does not have the categorical dimensions required by
/// its sector tuple.
#[derive(Debug)]
pub enum CheckedGenericPlanError<E> {
    Provider(E),
    SymbolShape {
        symbol: &'static str,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    Core(CoreError),
    Operation(OperationError),
}

impl<E> From<CoreError> for CheckedGenericPlanError<E> {
    fn from(error: CoreError) -> Self {
        // An axis-list rejection is the operations crate's variant (#1989).
        match OperationError::from(error) {
            OperationError::Core(error) => Self::Core(error),
            axis => Self::Operation(axis),
        }
    }
}

impl<E> From<OperationError> for CheckedGenericPlanError<E> {
    fn from(error: OperationError) -> Self {
        Self::Operation(error)
    }
}

impl<E> From<CheckedGenericStructureError<E>> for CheckedGenericPlanError<E> {
    fn from(error: CheckedGenericStructureError<E>) -> Self {
        match error {
            CheckedGenericStructureError::Provider(error) => Self::Provider(error),
            CheckedGenericStructureError::Core(error) => Self::Core(error),
        }
    }
}

fn map_checked_generic_symbol_error<E>(
    error: CheckedGenericSymbolError<E>,
) -> CheckedGenericPlanError<E> {
    match error {
        CheckedGenericSymbolError::Provider(error) => CheckedGenericPlanError::Provider(error),
        CheckedGenericSymbolError::Shape {
            symbol,
            expected,
            actual,
        } => CheckedGenericPlanError::SymbolShape {
            symbol,
            expected,
            actual,
        },
        CheckedGenericSymbolError::Core(error) => CheckedGenericPlanError::Core(error),
    }
}

fn map_checked_generic_structure_error<E>(
    error: CheckedGenericStructureError<E>,
) -> CheckedGenericPlanError<E> {
    error.into()
}

impl<E: core::fmt::Display> core::fmt::Display for CheckedGenericPlanError<E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Provider(_) => f.write_str("fusion provider failed"),
            Self::SymbolShape {
                symbol,
                expected,
                actual,
            } => write!(
                f,
                "{symbol} shape mismatch: expected {expected:?}, got {actual:?}"
            ),
            Self::Core(_) => f.write_str("invalid tensor structure"),
            Self::Operation(_) => f.write_str("tensor operation failed"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for CheckedGenericPlanError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Provider(error) => Some(error),
            Self::Core(error) => Some(error),
            Self::Operation(error) => Some(error),
            Self::SymbolShape { .. } => None,
        }
    }
}

pub(crate) fn validate_multiplicity_free_tree_transform_capability<R>(
    rule: &R,
    operation: &TreeTransformOperation,
) -> Result<(), OperationError>
where
    R: FusionRule,
{
    #[cfg(test)]
    crate::probe::hit(crate::probe::Event::MultiplicityFreeCapabilityValidation);
    if !rule.fusion_style().is_multiplicity_free() {
        return Err(OperationError::UnsupportedFusionStyle {
            operation: Box::new(operation.clone()),
            style: rule.fusion_style(),
        });
    }
    operation.validate_braiding_support(rule)
}

pub(crate) fn validate_fusion_tree_key_namespace(
    structure: &BlockStructure,
) -> Result<(), OperationError> {
    match structure.sector_structure().key_kind() {
        None | Some(BlockKeyKind::FusionTree) => Ok(()),
        Some(actual) => Err(OperationError::from_core_preserving_context(
            CoreError::ExpectedFusionTreePairKey { actual },
        )),
    }
}

pub(crate) fn validate_tree_transform_rank_syntax(
    operation: &TreeTransformOperation,
    total_rank: usize,
) -> Result<(), OperationError> {
    PreparedTreePairOperation::validate_rank_syntax(
        total_rank,
        operation.codomain_permutation(),
        operation.domain_permutation(),
    )
    .map_err(OperationError::from_core_preserving_context)
}

pub(crate) fn validate_tree_pair_namespace_before_cache(
    operation: &TreeTransformOperation,
    structure: &BlockStructure,
) -> Result<(), OperationError> {
    match structure.sector_structure().key_kind() {
        None | Some(BlockKeyKind::FusionTree) => Ok(()),
        Some(_) => {
            // Why not probe the categorical cache for application keys: no
            // successful cold compile can publish such an entry. Syntax still
            // precedes namespace rejection to preserve the public error order.
            validate_tree_transform_rank_syntax(operation, structure.rank())?;
            validate_fusion_tree_key_namespace(structure)
        }
    }
}

pub(crate) fn validate_all_codomain_namespace_before_cache(
    operation: &TreeTransformOperation,
    structure: &BlockStructure,
) -> Result<(), OperationError> {
    match structure.sector_structure().key_kind() {
        None | Some(BlockKeyKind::FusionTree) => Ok(()),
        Some(_) => {
            // Why not apply rank-only syntax to categorical sources here:
            // all-codomain admission has an established pair/scope/source
            // precedence that depends on the actual tree split. Noncategorical
            // storage has no split, so only scope and total-rank syntax can
            // precede its namespace rejection.
            validate_all_codomain_operation_scope(operation)?;
            validate_tree_transform_rank_syntax(operation, structure.rank())?;
            validate_fusion_tree_key_namespace(structure)
        }
    }
}

pub(crate) fn validate_multiplicity_free_tree_pair_preflight<'rule, 'structure, R>(
    rule: &'rule R,
    operation: &TreeTransformOperation,
    src_structure: &'structure BlockStructure,
) -> Result<LocallyValidatedFusionTreeBlockStructure<'rule, 'structure, R>, OperationError>
where
    R: FusionRule,
{
    validate_multiplicity_free_tree_transform_capability(rule, operation)?;
    validate_multiplicity_free_tree_pair_preflight_after_capability(rule, operation, src_structure)
}

pub(crate) fn validate_multiplicity_free_tree_pair_preflight_after_capability<
    'rule,
    'structure,
    R,
>(
    rule: &'rule R,
    operation: &TreeTransformOperation,
    src_structure: &'structure BlockStructure,
) -> Result<LocallyValidatedFusionTreeBlockStructure<'rule, 'structure, R>, OperationError>
where
    R: FusionRule,
{
    validate_tree_transform_rank_syntax(operation, src_structure.rank())?;
    validate_fusion_tree_key_namespace(src_structure)?;
    validate_tree_transform_operation_syntax(operation, src_structure)?;
    LocallyValidatedFusionTreeBlockStructure::try_new(rule, src_structure)
        .map_err(OperationError::from_core_preserving_context)
}

#[cfg(test)]
pub(crate) fn validate_generic_tree_pair_preflight<'rule, 'structure, R>(
    rule: &'rule R,
    operation: &TreeTransformOperation,
    src_structure: &'structure BlockStructure,
) -> Result<LocallyValidatedFusionTreeBlockStructure<'rule, 'structure, R>, OperationError>
where
    R: FusionRule,
{
    if !rule.fusion_style().has_multiplicity() {
        return Err(OperationError::UnsupportedFusionStyle {
            operation: Box::new(operation.clone()),
            style: rule.fusion_style(),
        });
    }
    operation.validate_braiding_support(rule)?;
    validate_tree_transform_rank_syntax(operation, src_structure.rank())?;
    validate_fusion_tree_key_namespace(src_structure)?;
    validate_tree_transform_operation_syntax(operation, src_structure)?;
    LocallyValidatedFusionTreeBlockStructure::try_new(rule, src_structure)
        .map_err(OperationError::from_core_preserving_context)
}

pub(crate) struct CheckedGenericTreePairPreflight<'provider, 'structure, P> {
    admitted: CheckedGenericAdmittedFusionTreeBlockStructure<'provider, 'structure, P>,
}

impl<'structure, P> CheckedGenericTreePairPreflight<'_, 'structure, P>
where
    P: CheckedGenericFusion,
{
    fn structure(&self) -> &'structure BlockStructure {
        self.admitted.structure()
    }
}

fn validate_checked_generic_tree_pair_preflight<'provider, 'structure, P>(
    provider: &'provider P,
    identity: Option<&RuleIdentity>,
    operation: &TreeTransformOperation,
    src_structure: &'structure BlockStructure,
) -> Result<
    CheckedGenericTreePairPreflight<'provider, 'structure, P>,
    CheckedGenericPlanError<P::Error>,
>
where
    P: CheckedGenericFusion,
{
    if !provider.fusion_style().has_multiplicity() {
        return Err(CheckedGenericPlanError::Operation(
            OperationError::UnsupportedFusionStyle {
                operation: Box::new(operation.clone()),
                style: provider.fusion_style(),
            },
        ));
    }
    if operation.requires_symmetric_braiding() && !provider.braiding_style().is_symmetric() {
        return Err(CheckedGenericPlanError::Operation(
            OperationError::UnsupportedBraidingStyle {
                operation: Box::new(operation.clone()),
                style: provider.braiding_style(),
            },
        ));
    }
    validate_tree_transform_rank_syntax(operation, src_structure.rank())?;
    validate_fusion_tree_key_namespace(src_structure)?;
    validate_tree_transform_operation_syntax(operation, src_structure)?;
    let admitted = match identity {
        Some(identity) => CheckedGenericAdmittedFusionTreeBlockStructure::try_new_once(
            provider,
            identity,
            src_structure,
        ),
        None => CheckedGenericAdmittedFusionTreeBlockStructure::try_new(provider, src_structure),
    }
    .map_err(map_checked_generic_structure_error)?;
    Ok(CheckedGenericTreePairPreflight { admitted })
}

/// The plan preflight of a caller that already read and checked `identity`,
/// the provider's: its source admission runs once per structure (#2233).
pub(crate) fn validate_checked_generic_tree_pair_plan_preflight<'provider, 'structure, P>(
    provider: &'provider P,
    identity: &RuleIdentity,
    operation: &TreeTransformOperation,
    src_structure: &'structure BlockStructure,
) -> Result<
    CheckedGenericTreePairPreflight<'provider, 'structure, P>,
    CheckedGenericPlanError<P::Error>,
>
where
    P: CheckedGenericFusion,
{
    validate_checked_generic_tree_pair_preflight(provider, Some(identity), operation, src_structure)
}

pub(crate) struct LocallyValidatedAllCodomainFusionTreeBlockStructure<'rule, 'structure, R> {
    proof: LocallyValidatedFusionTreeBlockStructure<'rule, 'structure, R>,
}

impl<'rule, 'structure, R> LocallyValidatedAllCodomainFusionTreeBlockStructure<'rule, 'structure, R>
where
    R: FusionRule,
{
    fn try_new(
        rule: &'rule R,
        operation: &TreeTransformOperation,
        src_structure: &'structure BlockStructure,
    ) -> Result<Self, OperationError> {
        validate_multiplicity_free_tree_transform_capability(rule, operation)?;
        Self::try_new_after_capability(rule, operation, src_structure)
    }

    fn try_new_after_capability(
        rule: &'rule R,
        operation: &TreeTransformOperation,
        src_structure: &'structure BlockStructure,
    ) -> Result<Self, OperationError> {
        validate_all_codomain_namespace_before_cache(operation, src_structure)?;
        let mut first_pair_mismatch = None;
        let mut first_source_restriction = None;
        let mut first_syntax_error = None;
        let mut prepared_splits = SmallVec::<[(usize, usize); 4]>::new();
        for index in 0..src_structure.block_count() {
            let block = src_structure.block(index)?;
            let BlockKey::FusionTree(key) = block.key() else {
                continue;
            };
            if key.codomain_tree().coupled() != key.domain_tree().coupled() {
                first_pair_mismatch.get_or_insert(OperationError::Core(
                    CoreError::MalformedFusionTree {
                        message: "fusion tree pair requires matching coupled sectors",
                    },
                ));
            }
            if first_source_restriction.is_none() {
                first_source_restriction =
                    validate_all_codomain_fusion_tree_block(rule, index, key).err();
            }
            let split = (
                key.codomain_tree().uncoupled().len(),
                key.domain_tree().uncoupled().len(),
            );
            if !prepared_splits.contains(&split) {
                if first_syntax_error.is_none() {
                    first_syntax_error = prepare_tree_pair_operation_syntax(operation, split).err();
                }
                prepared_splits.push(split);
            }
        }
        if let Some(error) = first_pair_mismatch {
            return Err(error);
        }
        validate_all_codomain_operation_scope(operation)?;
        if let Some(error) = first_source_restriction {
            return Err(error);
        }
        if let Some(error) = first_syntax_error {
            return Err(error);
        }
        let proof = LocallyValidatedFusionTreeBlockStructure::try_new(rule, src_structure)
            .map_err(OperationError::from_core_preserving_context)?;
        Ok(Self { proof })
    }

    fn proof(&self) -> &LocallyValidatedFusionTreeBlockStructure<'rule, 'structure, R> {
        &self.proof
    }

    fn rule(&self) -> &'rule R {
        self.proof.rule()
    }

    fn structure(&self) -> &'structure BlockStructure {
        self.proof.structure()
    }
}

pub(crate) fn validate_multiplicity_free_all_codomain_preflight<'rule, 'structure, R>(
    rule: &'rule R,
    operation: &TreeTransformOperation,
    src_structure: &'structure BlockStructure,
) -> Result<LocallyValidatedAllCodomainFusionTreeBlockStructure<'rule, 'structure, R>, OperationError>
where
    R: FusionRule,
{
    LocallyValidatedAllCodomainFusionTreeBlockStructure::try_new(rule, operation, src_structure)
}

fn validate_tree_transform_operation_syntax(
    operation: &TreeTransformOperation,
    src_structure: &BlockStructure,
) -> Result<(), OperationError> {
    let mut prepared_splits = SmallVec::<[(usize, usize); 4]>::new();
    for index in 0..src_structure.block_count() {
        let block = src_structure.block(index)?;
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        let split = (
            key.codomain_tree().uncoupled().len(),
            key.domain_tree().uncoupled().len(),
        );
        if prepared_splits.contains(&split) {
            continue;
        }
        prepare_tree_pair_operation_syntax(operation, split)?;
        prepared_splits.push(split);
    }
    Ok(())
}

#[cfg(test)]
/// Build a TensorKit-style grouped tree-transform plan for multiplicity-free
/// fusion rules.
///
/// This is the generic callback form: each source tree may map to multiple
/// destination trees, and duplicate destinations are accumulated into one
/// group-level recoupling matrix. `GenericFusion` with vertex multiplicities is
/// intentionally not represented by this scalar-coefficient API.
///
/// # Provider-domain precondition
///
/// Fusion-tree block keys in `src_structure`, and keys returned by `transform`,
/// follow [`symtn_core::FusionTreeKey::validate_for_rule`]'s provider-domain
/// precondition.
pub fn build_tree_transform_group_plan<T, R, F>(
    rule: &R,
    operation: TreeTransformOperation,
    src_structure: &BlockStructure,
    transform: F,
) -> Result<TreeTransformGroupPlan<T>, OperationError>
where
    R: FusionRule,
    T: Clone + Add<Output = T> + Zero,
    F: FnMut(&FusionTreePairKey) -> Result<Vec<(FusionTreePairKey, T)>, OperationError>,
{
    let source_proof =
        validate_multiplicity_free_tree_pair_preflight(rule, &operation, src_structure)?;
    let mut transform = transform;
    build_tree_transform_group_plan_validated(&source_proof, operation, |source| {
        let rows = transform(source)?;
        for (destination, _) in &rows {
            destination
                .validate_for_rule(source_proof.rule())
                .map_err(OperationError::from_core_preserving_context)?;
        }
        Ok(rows)
    })
}

#[cfg(test)]
fn build_tree_transform_group_plan_validated<T, R, F>(
    source_proof: &LocallyValidatedFusionTreeBlockStructure<'_, '_, R>,
    operation: TreeTransformOperation,
    mut transform: F,
) -> Result<TreeTransformGroupPlan<T>, OperationError>
where
    R: FusionRule,
    T: Clone + Add<Output = T> + Zero,
    F: FnMut(&FusionTreePairKey) -> Result<Vec<(FusionTreePairKey, T)>, OperationError>,
{
    let src_structure = source_proof.structure();
    let source_axes = operation_source_axes(&operation);

    let mut specs = Vec::new();
    for group in src_structure.fusion_tree_group_slice() {
        specs.extend(assemble_tree_pair_group_specs(
            src_structure,
            group,
            &source_axes,
            &mut |_, src_key| transform(src_key).map(Arc::new),
        )?);
    }

    Ok(TreeTransformGroupPlan::new(specs))
}

/// Standard all-codomain tree-transform builder for Unique and Simple
/// multiplicity-free rules.
///
/// # Provider-domain precondition
///
/// Fusion-tree block keys in `src_structure` follow
/// [`symtn_core::FusionTreeKey::validate_for_rule`]'s provider-domain
/// precondition.
pub fn build_all_codomain_tree_transform_group_plan<R>(
    rule: &R,
    operation: TreeTransformOperation,
    src_structure: &BlockStructure,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
{
    // The family split, stated once for this file. `Unique` fusion is a
    // tree-to-tree bijection with a scalar coefficient, so its builder is a
    // flat O(#trees) loop emitting `Single` specs; `Simple` fusion needs the
    // grouped path with its dedup, injectivity check and dense recoupling
    // rows. TensorKit makes the same split for the same invariant, as two
    // transformer types selected at one constructor
    // (`tensors/treetransformers.jl` `AbelianTreeTransformer` versus
    // `GenericTreeTransformer`).
    //
    // Why the predicate recurs below rather than being threaded as a value:
    // the stages are separate functions with separate bodies — unvalidated,
    // validated, and validated-with-threads, the last of which exists only on
    // the `Simple` side because the flat loop has nothing to parallelize — so
    // there is no single object here to carry the decision the way
    // TensorKit's transformer does. Collapsing them was tried and rejected:
    // it orphans the per-family entry points that tests use to pin which
    // builder a fixture exercises (issue #306's acceptance).
    if rule.fusion_style() == FusionStyleKind::Unique {
        build_unique_all_codomain_tree_transform_group_plan(rule, operation, src_structure)
    } else {
        build_multiplicity_free_all_codomain_tree_transform_group_plan(
            rule,
            operation,
            src_structure,
        )
    }
}

/// Standard full tree-pair transform builder for Unique and Simple
/// multiplicity-free rules.
///
/// # Provider-domain precondition
///
/// Fusion-tree block keys in `src_structure` follow
/// [`symtn_core::FusionTreeKey::validate_for_rule`]'s provider-domain
/// precondition.
pub fn build_tree_pair_transform_group_plan<R>(
    rule: &R,
    operation: TreeTransformOperation,
    src_structure: &BlockStructure,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
{
    if rule.fusion_style() == FusionStyleKind::Unique {
        build_unique_tree_pair_transform_group_plan(rule, operation, src_structure)
    } else {
        build_multiplicity_free_tree_pair_transform_group_plan(rule, operation, src_structure)
    }
}

/// The categorical half of a completed multiplicity-free tree-pair build:
/// source preflight, destination key proof and plan build. It reads only the
/// rule, the operation and the two structures' block keys.
pub(crate) fn build_multiplicity_free_tree_pair_plan_after_capability_with_threads<R>(
    rule: &R,
    operation: &TreeTransformOperation,
    dst_structure: &BlockStructure,
    src_structure: &BlockStructure,
    threads: usize,
    reuse: Option<&CoefficientGroupReuse<R::Scalar>>,
) -> Result<ComposedPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: 'static
        + Copy
        + Clone
        + Add<Output = R::Scalar>
        + Mul<Output = R::Scalar>
        + Zero
        + Send
        + Sync,
{
    let source_proof = validate_multiplicity_free_tree_pair_preflight_after_capability(
        rule,
        operation,
        src_structure,
    )?;
    LocallyValidatedFusionTreeBlockStructure::try_new(rule, dst_structure)
        .map_err(OperationError::from_core_preserving_context)?;
    build_tree_pair_transform_group_plan_validated_reusing(
        &source_proof,
        operation.clone(),
        threads,
        reuse,
    )
}

#[cfg(test)]
pub(crate) fn build_tree_pair_transform_group_plan_validated_with_threads<R>(
    source_proof: &LocallyValidatedFusionTreeBlockStructure<'_, '_, R>,
    operation: TreeTransformOperation,
    threads: usize,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar:
        'static + Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero + Send + Sync,
{
    build_tree_pair_transform_group_plan_validated_reusing(source_proof, operation, threads, None)
        .map(ComposedPlan::into_owned)
}

/// As [`build_tree_pair_transform_group_plan_validated_with_threads`], reusing
/// cached source groups. Unique fusion never reuses one: it is `NoCache`, as
/// TensorKit's `fsbraid`/`fstranspose` (and QSpace's abelian skip), since its
/// per-block rebuild allocates nothing ([`compile_unique_tree_pair_structure`]).
fn build_tree_pair_transform_group_plan_validated_reusing<R>(
    source_proof: &LocallyValidatedFusionTreeBlockStructure<'_, '_, R>,
    operation: TreeTransformOperation,
    threads: usize,
    reuse: Option<&CoefficientGroupReuse<R::Scalar>>,
) -> Result<ComposedPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar:
        'static + Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero + Send + Sync,
{
    if source_proof.rule().fusion_style() == FusionStyleKind::Unique {
        build_unique_tree_pair_transform_group_plan_validated(source_proof, operation)
            .map(ComposedPlan::Owned)
    } else {
        build_multiplicity_free_tree_pair_transform_group_plan_validated_with_threads(
            source_proof,
            operation,
            threads,
            reuse,
        )
    }
}

/// The Unique tree-pair transformer of a cache-3 miss, uncached as TensorKit's
/// `NoCache` `fsbraid`/`fstranspose` (`braiding_manipulations.jl:338-339`,
/// `duality_manipulations.jl:566-569`) under its `AbelianTreeTransformer`
/// (`treetransformers.jl:10-12`): each source block moves on one reused
/// scratch, its destination is found by view, and the spec stores block
/// indices, so the compile allocates nothing per block at any rank.
/// `scratch` is the execution context's, reused across compiles.
///
/// Specs follow the source groups, as the cached builder emitted them.
pub(crate) fn compile_unique_tree_pair_structure<R>(
    rule: &R,
    operation: &TreeTransformOperation,
    dst_structure: &Arc<BlockStructure>,
    src_structure: &Arc<BlockStructure>,
    storage_conjugate: bool,
    scratch: &mut UniqueTreePairScratch,
) -> Result<symtn_operations::TreeTransformStructure<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Copy + Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let source_proof = validate_multiplicity_free_tree_pair_preflight_after_capability(
        rule,
        operation,
        src_structure,
    )?;
    LocallyValidatedFusionTreeBlockStructure::try_new(rule, dst_structure)
        .map_err(OperationError::from_core_preserving_context)?;
    let source_axes = operation_source_axes(operation);
    let mut primary_prepared = None;
    let mut additional_prepared = None::<FxHashMap<(usize, usize), PreparedTreePairOperation<'_>>>;
    scratch.reserve_rank(src_structure.rank());
    let mut specs = Vec::with_capacity(src_structure.block_count());
    for group in src_structure.fusion_tree_group_slice() {
        for &index in group.block_indices() {
            let Some(src_key) = source_proof.fusion_tree_pair_key(index)? else {
                return Err(OperationError::ExpectedFusionTreeBlock {
                    tensor: "src",
                    index,
                });
            };
            let prepared = prepared_tree_pair_operation_for_split(
                &mut primary_prepared,
                &mut additional_prepared,
                rule,
                operation,
                tree_pair_split(src_key),
            )?;
            let coefficient = source_proof
                .execute_unique_rigid_for_block_index_into(index, prepared, scratch)
                .map_err(OperationError::from_core_preserving_context)?;
            specs.push(
                TreeTransformBlockSpec::single(
                    unique_destination_block(dst_structure, scratch)?,
                    index,
                    coefficient,
                )
                .with_shared_source_axes(Arc::clone(&source_axes)),
            );
        }
    }
    symtn_operations::TreeTransformStructure::compile_indexed_shared_structures(
        Arc::clone(dst_structure),
        Arc::clone(src_structure),
        &specs,
        storage_conjugate,
    )
}

/// [`compile_unique_tree_pair_structure`] for a source read through its
/// parent's storage blocks in `orientation` (a lazy adjoint). Each block moves
/// as the logical pair its storage block reads in `orientation`; the spec
/// sources are storage blocks and its axes the storage axes, shared.
///
/// Specs follow the order the cached oriented builder emitted: the parent's
/// groups for [`OrientedBasisOrder::Storage`], the logical keys otherwise
/// (a Unique group holds one tree pair, so first group appearance is key
/// order).
#[allow(clippy::too_many_arguments)]
pub(crate) fn compile_unique_oriented_tree_pair_structure<R, FAxis>(
    rule: &R,
    operation: &TreeTransformOperation,
    dst_structure: &Arc<BlockStructure>,
    logical_keys: &[FusionTreePairKey],
    storage_structure: &Arc<BlockStructure>,
    orientation: FusionTreePairOrientation,
    basis_order: OrientedBasisOrder,
    logical_rank: usize,
    storage_projection: &FxHashMap<&FusionTreePairKey, usize>,
    logical_to_storage_axis: FAxis,
    scratch: &mut UniqueTreePairScratch,
) -> Result<symtn_operations::TreeTransformStructure<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Copy + Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
    FAxis: Fn(usize) -> Result<usize, OperationError>,
{
    validate_tree_transform_rank_syntax(operation, logical_rank)?;
    let storage_axes = operation
        .codomain_permutation()
        .iter()
        .chain(operation.domain_permutation())
        .map(|&axis| logical_to_storage_axis(axis))
        .collect::<Result<Arc<[usize]>, _>>()?;
    let mut primary_prepared = None;
    let mut additional_prepared = None::<FxHashMap<(usize, usize), PreparedTreePairOperation<'_>>>;
    scratch.reserve_rank(logical_rank);
    let mut specs = Vec::with_capacity(logical_keys.len());
    let mut push = |storage_index: usize| -> Result<(), OperationError> {
        let BlockKey::FusionTree(storage_key) = storage_structure.block(storage_index)?.key()
        else {
            return Err(OperationError::ExpectedFusionTreeBlock {
                tensor: "src",
                index: storage_index,
            });
        };
        // Two handle clones, no allocation: the pair the oriented block reads.
        let source = match orientation {
            FusionTreePairOrientation::Direct => storage_key.clone(),
            FusionTreePairOrientation::Adjoint => FusionTreePairKey::pair(
                storage_key.domain_tree().clone(),
                storage_key.codomain_tree().clone(),
            ),
        };
        let prepared = prepared_tree_pair_operation_for_split(
            &mut primary_prepared,
            &mut additional_prepared,
            rule,
            operation,
            tree_pair_split(&source),
        )?;
        let coefficient = prepared
            .execute_unique_rigid_into(rule, &source, scratch)
            .map_err(OperationError::from_core_preserving_context)?;
        specs.push(
            TreeTransformBlockSpec::single(
                unique_destination_block(dst_structure, scratch)?,
                storage_index,
                coefficient,
            )
            .with_shared_source_axes(Arc::clone(&storage_axes)),
        );
        Ok(())
    };
    if basis_order == OrientedBasisOrder::Storage {
        for group in storage_structure.fusion_tree_group_slice() {
            for &index in group.block_indices() {
                push(index)?;
            }
        }
    } else {
        for key in logical_keys {
            push(
                *storage_projection
                    .get(key)
                    .ok_or(OperationError::StructureMismatch {
                        tensor: "oriented source projection",
                    })?,
            )?;
        }
    }
    symtn_operations::TreeTransformStructure::compile_indexed_shared_structures(
        Arc::clone(dst_structure),
        Arc::clone(storage_structure),
        &specs,
        orientation == FusionTreePairOrientation::Adjoint,
    )
}

fn tree_pair_split(key: &FusionTreePairKey) -> (usize, usize) {
    (
        key.codomain_tree().uncoupled().len(),
        key.domain_tree().uncoupled().len(),
    )
}

/// The destination block of the pair a Unique move left in `scratch`.
fn unique_destination_block(
    dst_structure: &BlockStructure,
    scratch: &UniqueTreePairScratch,
) -> Result<usize, OperationError> {
    let pair = scratch.pair_view();
    dst_structure
        .find_block_index_by_fusion_tree_pair_view(pair)
        .ok_or_else(|| OperationError::MissingBlockKey {
            key: Box::new(BlockKey::FusionTree(pair.to_key())),
        })
}

/// Source fusion-tree pair keys of `structure`, flattened in group order.
fn source_group_key_refs(
    structure: &BlockStructure,
) -> Result<Vec<&FusionTreePairKey>, OperationError> {
    let mut keys = Vec::with_capacity(structure.block_count());
    for group in structure.fusion_tree_group_slice() {
        for &index in group.block_indices() {
            let BlockKey::FusionTree(key) = structure.block(index)?.key() else {
                return Err(OperationError::ExpectedFusionTreeBlock {
                    tensor: "src",
                    index,
                });
            };
            keys.push(key);
        }
    }
    Ok(keys)
}

/// Splits flattened source keys into `(group key, ordered source keys)`.
fn split_source_groups<'a>(
    group_keys: impl IntoIterator<Item = (&'a FusionTreeGroupKey, usize)>,
    mut keys: &'a [&'a FusionTreePairKey],
) -> Vec<SourceGroup<'a>> {
    group_keys
        .into_iter()
        .map(|(group_key, len)| {
            let (group, rest) = keys.split_at(len);
            keys = rest;
            (group_key, group)
        })
        .collect()
}

/// Assembles a plan from its source groups in order. With `reuse`, only the
/// groups it misses are passed to `build_missing` (by index, ascending), and
/// the built groups are staged on it, all or none; without it every group
/// is built.
fn build_groups_reusing<T, E>(
    reuse: Option<(&CoefficientGroupReuse<T>, &[SourceGroup<'_>])>,
    group_count: usize,
    build_missing: impl FnOnce(Vec<usize>) -> Result<Vec<Vec<TreeTransformGroupBlockSpec<T>>>, E>,
) -> Result<ComposedPlan<T>, E>
where
    T: 'static + Clone + Send + Sync,
{
    let Some((reuse, groups)) = reuse else {
        let built = build_missing((0..group_count).collect())?;
        return Ok(ComposedPlan::Owned(TreeTransformGroupPlan::from_specs(
            built.into_iter().flatten(),
        )));
    };
    let slots = reuse.lookup(groups);
    let missing = slots
        .iter()
        .enumerate()
        .filter_map(|(index, slot)| matches!(slot, GroupSlot::Miss(_)).then_some(index))
        .collect();
    let mut built = build_missing(missing)?.into_iter();
    // Why shared groups rather than one flat spec vector: a degeneracy-only
    // miss whose groups all hit then copies no spec (one spec is hundreds of
    // bytes of tree keys), only one `Arc` per group, and shares the set's
    // coefficient payload through its first group (#1753).
    let mut shared = Vec::with_capacity(groups.len());
    for (&group, slot) in groups.iter().zip(slots) {
        let entry = match slot {
            GroupSlot::Hit(entry) => entry,
            GroupSlot::Miss(hash) => reuse.stage(
                hash,
                group,
                built.next().expect("every missing group was built"),
            ),
        };
        shared.push(entry);
    }
    Ok(ComposedPlan::Shared(SharedTreeTransformGroupPlan::new(
        shared,
    )))
}

/// A plan build's result: owned specs (an uncached build) or the
/// composed-coefficient cache's shared group slices.
pub(crate) enum ComposedPlan<T> {
    Owned(TreeTransformGroupPlan<T>),
    Shared(SharedTreeTransformGroupPlan<T>),
}

impl<T: Copy> ComposedPlan<T> {
    pub(crate) fn compile_shared_structures_with_storage_conjugation(
        &self,
        dst_structure: Arc<BlockStructure>,
        src_structure: Arc<BlockStructure>,
        storage_conjugate: bool,
    ) -> Result<symtn_operations::TreeTransformStructure<T>, OperationError> {
        match self {
            Self::Owned(plan) => plan.compile_shared_structures_with_storage_conjugation(
                dst_structure,
                src_structure,
                storage_conjugate,
            ),
            Self::Shared(plan) => plan.compile_shared_structures_with_storage_conjugation(
                dst_structure,
                src_structure,
                storage_conjugate,
            ),
        }
    }

    pub(crate) fn compile_shared_structures_with_storage_mapping<FBlock, FAxis>(
        &self,
        dst_structure: Arc<BlockStructure>,
        logical_src_structure: &BlockStructure,
        storage_src_structure: Arc<BlockStructure>,
        logical_to_storage_block: FBlock,
        logical_to_storage_axis: FAxis,
        storage_conjugate: bool,
    ) -> Result<symtn_operations::TreeTransformStructure<T>, OperationError>
    where
        FBlock: Fn(usize) -> Result<usize, OperationError>,
        FAxis: Fn(usize) -> Result<usize, OperationError>,
    {
        match self {
            Self::Owned(plan) => plan.compile_shared_structures_with_storage_mapping(
                dst_structure,
                logical_src_structure,
                storage_src_structure,
                logical_to_storage_block,
                logical_to_storage_axis,
                storage_conjugate,
            ),
            Self::Shared(plan) => plan.compile_shared_structures_with_storage_mapping(
                dst_structure,
                logical_src_structure,
                storage_src_structure,
                logical_to_storage_block,
                logical_to_storage_axis,
                storage_conjugate,
            ),
        }
    }

    pub(crate) fn compile_shared_structures_with_source_projection<FSource, FAxis>(
        &self,
        dst_structure: Arc<BlockStructure>,
        storage_src_structure: Arc<BlockStructure>,
        logical_rank: usize,
        source_index: FSource,
        logical_to_storage_axis: FAxis,
        storage_conjugate: bool,
    ) -> Result<symtn_operations::TreeTransformStructure<T>, OperationError>
    where
        FSource: Fn(&FusionTreePairKey) -> Result<usize, OperationError>,
        FAxis: Fn(usize) -> Result<usize, OperationError>,
    {
        match self {
            Self::Owned(plan) => plan.compile_shared_structures_with_source_projection(
                dst_structure,
                storage_src_structure,
                logical_rank,
                source_index,
                logical_to_storage_axis,
                storage_conjugate,
            ),
            Self::Shared(plan) => plan.compile_shared_structures_with_source_projection(
                dst_structure,
                storage_src_structure,
                logical_rank,
                source_index,
                logical_to_storage_axis,
                storage_conjugate,
            ),
        }
    }
}

#[cfg(test)]
impl<T: Clone> ComposedPlan<T> {
    /// The plan as owned specs, for comparison with an eager producer.
    pub(crate) fn into_owned(self) -> TreeTransformGroupPlan<T> {
        match self {
            Self::Owned(plan) => plan,
            Self::Shared(plan) => TreeTransformGroupPlan::from_specs(plan.specs().cloned()),
        }
    }
}

#[cfg(test)]
pub(crate) fn build_unique_tree_transform_group_plan<T, R, F>(
    rule: &R,
    operation: TreeTransformOperation,
    src_structure: &BlockStructure,
    transform: F,
) -> Result<TreeTransformGroupPlan<T>, OperationError>
where
    R: FusionRule,
    T: Clone + Add<Output = T> + Zero,
    F: FnMut(&FusionTreePairKey) -> Result<(FusionTreePairKey, T), OperationError>,
{
    // A capability guard, not a family choice: this entry point implements
    // only the bijective lowering, so a `Simple` rule is rejected rather than
    // rerouted.
    if rule.fusion_style() != FusionStyleKind::Unique {
        return Err(OperationError::UnsupportedFusionStyle {
            operation: Box::new(operation),
            style: rule.fusion_style(),
        });
    }
    let mut transform = transform;
    build_tree_transform_group_plan(rule, operation, src_structure, |source| {
        transform(source).map(|row| vec![row])
    })
}

pub(crate) fn build_unique_all_codomain_tree_transform_group_plan<R>(
    rule: &R,
    operation: TreeTransformOperation,
    src_structure: &BlockStructure,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
{
    // A capability guard, not a family choice: this entry point implements
    // only the bijective lowering, so a `Simple` rule is rejected rather than
    // rerouted.
    if rule.fusion_style() != FusionStyleKind::Unique {
        return Err(OperationError::UnsupportedFusionStyle {
            operation: Box::new(operation),
            style: rule.fusion_style(),
        });
    }
    let source_proof =
        validate_multiplicity_free_all_codomain_preflight(rule, &operation, src_structure)?;
    build_unique_all_codomain_tree_transform_group_plan_validated(&source_proof, operation)
}

fn build_unique_all_codomain_tree_transform_group_plan_validated<R>(
    source_proof: &LocallyValidatedAllCodomainFusionTreeBlockStructure<'_, '_, R>,
    operation: TreeTransformOperation,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
{
    debug_assert_eq!(source_proof.rule().fusion_style(), FusionStyleKind::Unique);
    let src_structure = source_proof.structure();
    let source_axes = operation_source_axes(&operation);
    let mut specs = Vec::with_capacity(src_structure.block_count());
    for index in 0..src_structure.block_count() {
        specs.extend(unique_all_codomain_spec(
            source_proof,
            &operation,
            &source_axes,
            index,
        )?);
    }
    Ok(TreeTransformGroupPlan::new(specs))
}

/// The one-term Unique all-codomain spec of source block `index`; `None`
/// for a block without a fusion-tree key.
fn unique_all_codomain_spec<R>(
    source_proof: &LocallyValidatedAllCodomainFusionTreeBlockStructure<'_, '_, R>,
    operation: &TreeTransformOperation,
    source_axes: &Arc<[usize]>,
    index: usize,
) -> Result<Option<TreeTransformGroupBlockSpec<R::Scalar>>, OperationError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
{
    let proof = source_proof.proof();
    let Some(src_key) = proof.fusion_tree_pair_key(index)? else {
        return Ok(None);
    };
    let mut rows = match operation.kind() {
        TreeTransformOperationKind::Permute => {
            proof.permute_codomain_rows_for_block_index(index, operation.codomain_permutation())
        }
        TreeTransformOperationKind::Braid => proof.braid_codomain_rows_for_block_index(
            index,
            operation.codomain_permutation(),
            operation.codomain_levels(),
        ),
        TreeTransformOperationKind::Transpose => {
            unreachable!("all-codomain admission rejected transpose")
        }
    }
    .map_err(OperationError::from_core_preserving_context)?;
    let Some((destination, coefficient)) = rows.pop() else {
        return Err(OperationError::EmptyTransformBlock);
    };
    if !rows.is_empty() {
        return Err(OperationError::StructureMismatch {
            tensor: "proof-bound Unique all-codomain cardinality",
        });
    }
    let dst_key = FusionTreePairKey::pair(destination, src_key.domain_tree().clone());
    Ok(Some(
        TreeTransformGroupBlockSpec::single(dst_key, src_key.clone(), coefficient)
            .with_shared_source_axes(Arc::clone(source_axes)),
    ))
}

pub(crate) fn build_multiplicity_free_all_codomain_tree_transform_group_plan<R>(
    rule: &R,
    operation: TreeTransformOperation,
    src_structure: &BlockStructure,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
{
    let source_proof =
        validate_multiplicity_free_all_codomain_preflight(rule, &operation, src_structure)?;
    build_multiplicity_free_all_codomain_tree_transform_group_plan_validated(
        &source_proof,
        operation,
    )
}

fn build_multiplicity_free_all_codomain_tree_transform_group_plan_validated<R>(
    source_proof: &LocallyValidatedAllCodomainFusionTreeBlockStructure<'_, '_, R>,
    operation: TreeTransformOperation,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
{
    let src_structure = source_proof.structure();
    let source_axes = operation_source_axes(&operation);
    let mut specs = Vec::new();
    for group in src_structure.fusion_tree_group_slice() {
        specs.extend(build_one_multiplicity_free_all_codomain_group(
            source_proof,
            &operation,
            &source_axes,
            group,
        )?);
    }
    Ok(TreeTransformGroupPlan::new(specs))
}

fn build_one_multiplicity_free_all_codomain_group<R>(
    source_proof: &LocallyValidatedAllCodomainFusionTreeBlockStructure<'_, '_, R>,
    operation: &TreeTransformOperation,
    source_axes: &Arc<[usize]>,
    group: &FusionTreeBlockGroup,
) -> Result<Vec<TreeTransformGroupBlockSpec<R::Scalar>>, OperationError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
{
    let mut unique_index_by_tree = FxHashMap::default();
    let mut unique_indices = SmallVec::<[usize; 8]>::new();
    let mut source_alignment = SmallVec::<[usize; 8]>::new();
    for &src_block_index in group.block_indices() {
        let Some(src_key) = source_proof.proof().fusion_tree_pair_key(src_block_index)? else {
            return Err(OperationError::ExpectedFusionTreeBlock {
                tensor: "src",
                index: src_block_index,
            });
        };
        let codomain_tree = src_key.codomain_tree();
        let unique_index = match unique_index_by_tree.get(codomain_tree) {
            Some(&index) => index,
            None => {
                let index = unique_indices.len();
                unique_index_by_tree.insert(codomain_tree, index);
                unique_indices.push(src_block_index);
                index
            }
        };
        source_alignment.push(unique_index);
    }

    let transformed = transform_all_codomain_rows_for_block_indices(
        source_proof,
        operation,
        &mut unique_indices.iter().copied(),
    )?;
    if transformed.len() != unique_indices.len() {
        return Err(OperationError::CoefficientCountMismatch {
            expected: unique_indices.len(),
            actual: transformed.len(),
        });
    }
    let transformed = transformed.into_iter().map(Arc::new).collect::<Vec<_>>();
    let mut source_cursor = 0usize;
    let mut rows_for = |_: usize, _: &FusionTreeKey| {
        let unique_index = source_alignment.get(source_cursor).copied();
        source_cursor += 1;
        unique_index
            .and_then(|index| transformed.get(index))
            .map(Arc::clone)
            .ok_or(OperationError::StructureMismatch {
                tensor: "compile-local all-codomain rows",
            })
    };
    let specs = if operation.is_identity_for(group.group_key().codomain_uncoupled().len(), 0) {
        assemble_identity_all_codomain_group_specs(
            source_proof.structure(),
            group,
            source_axes,
            &mut rows_for,
        )
    } else {
        assemble_all_codomain_group_specs(
            source_proof.structure(),
            group,
            source_axes,
            &mut rows_for,
        )
    }?;
    if source_cursor != source_alignment.len() {
        return Err(OperationError::StructureMismatch {
            tensor: "compile-local all-codomain row order",
        });
    }
    Ok(specs)
}

type TransformRows<K, T> = Vec<(K, T)>;

#[cfg(test)]
mod generic_preflight_tests {
    use super::{validate_generic_tree_pair_preflight, TreeTransformOperation};
    use crate::tests::GenericMultiplicityRule;
    use symtn_core::{
        BlockKey, BlockSpec, BlockStructure, CoreError, FusionTreeKey, FusionTreePairKey,
        MultiplicityIndex, SectorId,
    };
    use symtn_operations::OperationError;

    #[test]
    fn generic_preflight_accepts_valid_braid_before_categorical_admission() {
        let rule = GenericMultiplicityRule;
        let charge = SectorId::new(1);
        let vacuum = SectorId::new(0);
        let valid = FusionTreePairKey::pair(
            FusionTreeKey::try_new_for_rule(
                &rule,
                [charge, charge],
                vacuum,
                [false, false],
                [],
                [MultiplicityIndex::ONE],
            )
            .unwrap(),
            FusionTreeKey::try_new_for_rule(&rule, [], vacuum, [], [], []).unwrap(),
        );
        let valid_structure = BlockStructure::from_blocks(vec![BlockSpec::column_major_with_key(
            BlockKey::from(valid),
            vec![1, 1],
            0,
        )
        .unwrap()])
        .unwrap();

        validate_generic_tree_pair_preflight(
            &rule,
            &TreeTransformOperation::braid([1, 0], [], [0, 1], []),
            &valid_structure,
        )
        .unwrap();

        assert_eq!(
            FusionTreePairKey::try_pair_from_sector_ids(
                [charge.id(), charge.id()],
                [],
                vacuum.id(),
                [false, false],
                [],
                [],
                [],
                [0],
                [],
            )
            .unwrap_err(),
            CoreError::InvalidMultiplicityIndex { value: 0 }
        );

        let malformed = FusionTreePairKey::try_pair_from_sector_ids(
            [charge.id(), charge.id()],
            [],
            vacuum.id(),
            [false, false],
            [],
            [],
            [],
            [2],
            [],
        )
        .unwrap();
        let malformed_structure =
            BlockStructure::from_blocks(vec![BlockSpec::column_major_with_key(
                BlockKey::from(malformed),
                vec![1, 1],
                0,
            )
            .unwrap()])
            .unwrap();
        let error = match validate_generic_tree_pair_preflight(
            &rule,
            &TreeTransformOperation::braid([0, 0], [], [0, 1], []),
            &malformed_structure,
        ) {
            Ok(_) => panic!("invalid Generic operation unexpectedly admitted"),
            Err(error) => error,
        };

        // What: invalid operation syntax retains precedence over malformed
        // Generic categorical data.
        assert_eq!(
            error,
            OperationError::InvalidPermutation {
                axes: vec![0, 0],
                rank: 2
            }
        );
    }

    #[test]
    fn generic_preflight_checks_rank_syntax_before_opaque_namespace() {
        let rule = GenericMultiplicityRule;
        let structure = BlockStructure::from_blocks(vec![BlockSpec::column_major_with_key(
            BlockKey::opaque([4, 2]),
            vec![1, 1],
            0,
        )
        .unwrap()])
        .unwrap();

        let syntax_error = match validate_generic_tree_pair_preflight(
            &rule,
            &TreeTransformOperation::braid([0, 0], [], [0, 1], []),
            &structure,
        ) {
            Ok(_) => panic!("invalid operation unexpectedly admitted"),
            Err(error) => error,
        };
        assert_eq!(
            syntax_error,
            OperationError::InvalidPermutation {
                axes: vec![0, 0],
                rank: 2
            }
        );

        let namespace_error = match validate_generic_tree_pair_preflight(
            &rule,
            &TreeTransformOperation::braid([0, 1], [], [0, 1], []),
            &structure,
        ) {
            Ok(_) => panic!("opaque namespace unexpectedly admitted"),
            Err(error) => error,
        };
        // What: Generic and multiplicity-free cold admission share the same
        // operation-before-namespace boundary.
        assert_eq!(
            namespace_error,
            OperationError::Core(CoreError::ExpectedFusionTreePairKey {
                actual: symtn_core::BlockKeyKind::Opaque,
            })
        );
    }
}

fn execute_staged_groups<I, O, F>(
    inputs: Vec<I>,
    threads: usize,
    build: F,
) -> Result<Vec<O>, OperationError>
where
    I: Send,
    O: Send,
    F: Fn(I) -> Result<O, OperationError> + Send + Sync,
{
    // The requested degree is a cap within the operation's Host pool.
    let threads = if threads <= 1 || inputs.len() <= 1 {
        1
    } else {
        threads.min(symtn_operations::host_pool::current_threads())
    };
    if threads <= 1 {
        return inputs.into_iter().map(build).collect();
    }

    use std::sync::Mutex;
    use symtn_operations::host_pool::{fork, region, HostPoolSite};

    // One input and one result slot per batch: each job takes its input and
    // stores its result, so the fold below sees the batches in source order
    // whatever ran first.
    let inputs: Vec<Mutex<Option<Vec<I>>>> = partition_staged_groups(inputs, threads)
        .into_iter()
        .map(|batch| Mutex::new(Some(batch)))
        .collect();
    type Slot<O> = Mutex<Option<Result<Vec<O>, OperationError>>>;
    let results: Vec<Slot<O>> = inputs.iter().map(|_| Mutex::new(None)).collect();
    region(HostPoolSite::PlanCompile, |exec| {
        fork(exec, inputs.len(), |job, _child| {
            let batch = inputs[job].lock().unwrap_or_else(|p| p.into_inner()).take();
            if let Some(batch) = batch {
                let result = batch.into_iter().map(&build).collect();
                *results[job].lock().unwrap_or_else(|p| p.into_inner()) = Some(result);
            }
        });
    });
    let batches = results.into_iter().map(|slot| {
        // INVARIANT: `fork` runs every job before returning.
        slot.into_inner()
            .unwrap_or_else(|p| p.into_inner())
            .expect("every staged batch ran")
    });
    flatten_ordered_batch_results(batches)
}

fn flatten_ordered_batch_results<O, E>(
    batches: impl IntoIterator<Item = Result<Vec<O>, E>>,
) -> Result<Vec<O>, E> {
    let mut outputs = Vec::new();
    // Why not stop at the first failing job as it finishes: jobs finish in
    // scheduling order, so the winning error would be nondeterministic when
    // several batches fail. Folding ordered batch results here preserves the
    // serial source-group error precedence without publishing partial state.
    for batch in batches {
        outputs.extend(batch?);
    }
    Ok(outputs)
}

#[cfg(test)]
mod ordered_batch_tests {
    use super::flatten_ordered_batch_results;

    #[test]
    fn flatten_ordered_batches_selects_first_error_and_preserves_success_order() {
        // What: collector semantics follow source-batch order even when several
        // worker results contain distinct errors.
        let errors = vec![Ok(vec![0, 1]), Err("first"), Err("second")];
        assert_eq!(flatten_ordered_batch_results(errors), Err("first"));

        // What: successful batches flatten without reordering their rows.
        let successes: Vec<Result<Vec<usize>, &str>> =
            vec![Ok(vec![0, 1]), Ok(vec![2]), Ok(vec![3, 4])];
        assert_eq!(
            flatten_ordered_batch_results(successes),
            Ok(vec![0, 1, 2, 3, 4])
        );
    }
}

fn partition_staged_groups<I>(inputs: Vec<I>, threads: usize) -> Vec<Vec<I>> {
    let batch_count = threads.max(1).min(inputs.len());
    if batch_count == 0 {
        return Vec::new();
    }
    let minimum_size = inputs.len() / batch_count;
    let larger_batches = inputs.len() % batch_count;
    let mut inputs = inputs.into_iter();
    (0..batch_count)
        .map(|batch| {
            let size = minimum_size + usize::from(batch < larger_batches);
            inputs.by_ref().take(size).collect()
        })
        .collect()
}

#[cfg(test)]
pub(crate) fn partition_staged_groups_for_test<I>(inputs: Vec<I>, threads: usize) -> Vec<Vec<I>> {
    partition_staged_groups(inputs, threads)
}

fn transform_all_codomain_rows_for_block_indices<R>(
    proof: &LocallyValidatedAllCodomainFusionTreeBlockStructure<'_, '_, R>,
    operation: &TreeTransformOperation,
    block_indices: &mut dyn Iterator<Item = usize>,
) -> Result<Vec<TransformRows<FusionTreeKey, R::Scalar>>, OperationError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let Some(first_index) = block_indices.next() else {
        return Ok(Vec::new());
    };
    let indices = std::iter::once(first_index).chain(block_indices);
    let transformed = match operation.kind() {
        TreeTransformOperationKind::Permute => proof
            .proof()
            .permute_codomain_rows_for_block_indices(indices, operation.codomain_permutation()),
        TreeTransformOperationKind::Braid => proof.proof().braid_codomain_rows_for_block_indices(
            indices,
            operation.codomain_permutation(),
            operation.codomain_levels(),
        ),
        TreeTransformOperationKind::Transpose => {
            unreachable!("all-codomain operation scope validation rejected transpose")
        }
    };
    transformed.map_err(OperationError::from_core_preserving_context)
}

fn assemble_identity_all_codomain_group_specs<T, F>(
    src_structure: &BlockStructure,
    group: &FusionTreeBlockGroup,
    source_axes: &Arc<[usize]>,
    rows_for: &mut F,
) -> Result<Vec<TreeTransformGroupBlockSpec<T>>, OperationError>
where
    T: Clone,
    F: FnMut(usize, &FusionTreeKey) -> Result<Arc<Vec<(FusionTreeKey, T)>>, OperationError>,
{
    let mut specs = Vec::with_capacity(group.block_indices().len());
    for &src_block_index in group.block_indices() {
        let block = src_structure.block(src_block_index)?;
        let BlockKey::FusionTree(src_key) = block.key() else {
            return Err(OperationError::ExpectedFusionTreeBlock {
                tensor: "src",
                index: src_block_index,
            });
        };
        let transformed = rows_for(src_block_index, src_key.codomain_tree())?;
        let [(dst_codomain_tree, coefficient)] = transformed.as_slice() else {
            return Err(OperationError::EmptyTransformBlock);
        };
        let dst_key =
            FusionTreePairKey::pair(dst_codomain_tree.clone(), src_key.domain_tree().clone());
        specs.push(
            TreeTransformGroupBlockSpec::single(dst_key, src_key.clone(), coefficient.clone())
                .with_shared_source_axes(Arc::clone(source_axes)),
        );
    }
    Ok(specs)
}

struct DenseRecouplingRows<T> {
    coefficients: Vec<T>,
    src_count: usize,
}

impl<T> DenseRecouplingRows<T>
where
    T: Clone + Add<Output = T> + Zero,
{
    fn new(src_count: usize) -> Self {
        Self {
            coefficients: Vec::with_capacity(src_count),
            src_count,
        }
    }

    fn push_zero_row(&mut self) -> usize {
        let row = self.coefficients.len() / self.src_count;
        self.coefficients
            .resize_with(self.coefficients.len() + self.src_count, T::zero);
        row
    }

    fn add(&mut self, dst_row: usize, src_column: usize, coefficient: T) {
        let index = dst_row * self.src_count + src_column;
        self.coefficients[index] = self.coefficients[index].clone() + coefficient;
    }

    fn into_coefficients(self) -> Vec<T> {
        self.coefficients
    }
}

fn assemble_all_codomain_group_specs<T, F>(
    src_structure: &BlockStructure,
    group: &FusionTreeBlockGroup,
    source_axes: &Arc<[usize]>,
    rows_for: &mut F,
) -> Result<Vec<TreeTransformGroupBlockSpec<T>>, OperationError>
where
    T: Clone + Add<Output = T> + Zero,
    F: FnMut(usize, &FusionTreeKey) -> Result<Arc<Vec<(FusionTreeKey, T)>>, OperationError>,
{
    let src_block_indices = group.block_indices();
    let mut src_keys = Vec::<FusionTreePairKey>::with_capacity(src_block_indices.len());
    let mut dst_keys = Vec::<FusionTreePairKey>::new();
    let mut dst_index_by_key = FxHashMap::<FusionTreePairKey, usize>::default();
    let mut rows = DenseRecouplingRows::new(src_block_indices.len());
    let mut direct_rows = Vec::with_capacity(src_block_indices.len());
    let mut direct_dst_keys = FxHashSet::default();
    let mut is_injective_singleton = true;

    for (src_column, &src_block_index) in src_block_indices.iter().enumerate() {
        let block = src_structure.block(src_block_index)?;
        let BlockKey::FusionTree(src_key) = block.key() else {
            return Err(OperationError::ExpectedFusionTreeBlock {
                tensor: "src",
                index: src_block_index,
            });
        };
        src_keys.push(src_key.clone());

        let transformed = rows_for(src_block_index, src_key.codomain_tree())?;
        if let [(dst_codomain_tree, coefficient)] = transformed.as_slice() {
            let dst_key =
                FusionTreePairKey::pair(dst_codomain_tree.clone(), src_key.domain_tree().clone());
            if !direct_dst_keys.insert(dst_key.clone()) {
                is_injective_singleton = false;
            }
            direct_rows.push((src_key.clone(), dst_key, coefficient.clone()));
        } else {
            is_injective_singleton = false;
        }
        for (dst_codomain_tree, coefficient) in transformed.iter() {
            let dst_key =
                FusionTreePairKey::pair(dst_codomain_tree.clone(), src_key.domain_tree().clone());
            let dst_row = if let Some(&dst_row) = dst_index_by_key.get(&dst_key) {
                dst_row
            } else {
                let dst_row = dst_keys.len();
                dst_index_by_key.insert(dst_key.clone(), dst_row);
                dst_keys.push(dst_key);
                let allocated_row = rows.push_zero_row();
                debug_assert_eq!(allocated_row, dst_row);
                allocated_row
            };
            rows.add(dst_row, src_column, coefficient.clone());
        }
    }

    if let Some(direct_specs) = lower_injective_singleton_rows(
        source_axes,
        direct_rows,
        src_keys.len(),
        is_injective_singleton,
    ) {
        return Ok(direct_specs);
    }
    if dst_keys.is_empty() {
        return Err(OperationError::EmptyTransformBlock);
    }

    Ok(vec![TreeTransformGroupBlockSpec::try_multi(
        dst_keys,
        src_keys,
        rows.into_coefficients(),
    )?
    .with_shared_source_axes(Arc::clone(source_axes))])
}

pub(crate) fn build_multiplicity_free_tree_pair_transform_group_plan<R>(
    rule: &R,
    operation: TreeTransformOperation,
    src_structure: &BlockStructure,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
{
    let source_proof =
        validate_multiplicity_free_tree_pair_preflight(rule, &operation, src_structure)?;
    build_multiplicity_free_tree_pair_transform_group_plan_validated(&source_proof, operation)
}

fn build_multiplicity_free_tree_pair_transform_group_plan_validated<R>(
    source_proof: &LocallyValidatedFusionTreeBlockStructure<'_, '_, R>,
    operation: TreeTransformOperation,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
{
    let src_structure = source_proof.structure();
    let source_axes = operation_source_axes(&operation);
    let mut primary_prepared = None;
    let mut additional_prepared = None::<FxHashMap<(usize, usize), PreparedTreePairOperation<'_>>>;
    let mut specs = Vec::new();
    for group in src_structure.fusion_tree_group_slice() {
        let source_split = (
            group.group_key().codomain_uncoupled().len(),
            group.group_key().domain_uncoupled().len(),
        );
        let prepared = prepared_tree_pair_operation_for_split(
            &mut primary_prepared,
            &mut additional_prepared,
            source_proof.rule(),
            &operation,
            source_split,
        )?;
        specs.extend(build_one_multiplicity_free_tree_pair_group(
            source_proof,
            &operation,
            &source_axes,
            group,
            prepared,
        )?);
    }
    Ok(TreeTransformGroupPlan::new(specs))
}

fn build_one_multiplicity_free_tree_pair_group<R>(
    source_proof: &LocallyValidatedFusionTreeBlockStructure<'_, '_, R>,
    operation: &TreeTransformOperation,
    source_axes: &Arc<[usize]>,
    group: &FusionTreeBlockGroup,
    prepared: &PreparedTreePairOperation<'_>,
) -> Result<Vec<TreeTransformGroupBlockSpec<R::Scalar>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
{
    let source_split = (
        group.group_key().codomain_uncoupled().len(),
        group.group_key().domain_uncoupled().len(),
    );
    if operation.is_identity_for(source_split.0, source_split.1) {
        let mut rows_for = |index: usize, _: &FusionTreePairKey| {
            source_proof
                .execute_multiplicity_free_for_block_index(index, prepared)
                .map_err(OperationError::from_core_preserving_context)
                .map(Arc::new)
        };
        return assemble_identity_tree_pair_group_specs(
            source_proof.structure(),
            group,
            source_axes,
            &mut rows_for,
        );
    }

    let first_index = *group
        .block_indices()
        .first()
        .ok_or(OperationError::EmptyTransformBlock)?;
    source_proof.fusion_tree_pair_key(first_index)?.ok_or(
        OperationError::ExpectedFusionTreeBlock {
            tensor: "src",
            index: first_index,
        },
    )?;
    let indices = group.block_indices().iter().copied();
    let ordered = match operation.kind() {
        TreeTransformOperationKind::Transpose => source_proof
            .execute_multiplicity_free_transpose_ordered_for_block_indices_borrowed(
                indices, prepared,
            ),
        TreeTransformOperationKind::Permute | TreeTransformOperationKind::Braid => source_proof
            .execute_multiplicity_free_braid_ordered_for_block_indices_borrowed(indices, prepared),
    }
    .map_err(OperationError::from_core_preserving_context)?;
    assemble_ordered_tree_pair_group_specs(source_proof.structure(), group, source_axes, ordered)
}

fn build_multiplicity_free_tree_pair_transform_group_plan_validated_with_threads<R>(
    source_proof: &LocallyValidatedFusionTreeBlockStructure<'_, '_, R>,
    operation: TreeTransformOperation,
    threads: usize,
    reuse: Option<&CoefficientGroupReuse<R::Scalar>>,
) -> Result<ComposedPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar:
        'static + Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero + Send + Sync,
{
    let src_structure = source_proof.structure();
    let all_groups = src_structure.fusion_tree_group_slice();
    let flat_keys = reuse
        .map(|_| source_group_key_refs(src_structure))
        .transpose()?;
    let groups = flat_keys.as_deref().map(|keys| {
        split_source_groups(
            all_groups
                .iter()
                .map(|group| (group.group_key(), group.block_indices().len())),
            keys,
        )
    });
    build_groups_reusing(reuse.zip(groups.as_deref()), all_groups.len(), |missing| {
        build_multiplicity_free_tree_pair_groups(source_proof, &operation, threads, missing)
    })
}

/// Builds the listed source groups of `source_proof`, in order.
fn build_multiplicity_free_tree_pair_groups<R>(
    source_proof: &LocallyValidatedFusionTreeBlockStructure<'_, '_, R>,
    operation: &TreeTransformOperation,
    threads: usize,
    group_indices: Vec<usize>,
) -> Result<Vec<Vec<TreeTransformGroupBlockSpec<R::Scalar>>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero + Send + Sync,
{
    let all_groups = source_proof.structure().fusion_tree_group_slice();
    let source_axes = operation_source_axes(operation);
    let mut primary_prepared = None;
    let mut additional_prepared = None::<FxHashMap<(usize, usize), PreparedTreePairOperation<'_>>>;
    let mut staged_groups = Vec::with_capacity(group_indices.len());
    for group_index in group_indices {
        let group = &all_groups[group_index];
        let source_split = (
            group.group_key().codomain_uncoupled().len(),
            group.group_key().domain_uncoupled().len(),
        );
        prepared_tree_pair_operation_for_split(
            &mut primary_prepared,
            &mut additional_prepared,
            source_proof.rule(),
            operation,
            source_split,
        )?;
        // Why not stage the prepared value itself: ranks beyond its inline
        // step capacity would deep-clone spilled storage once per group.
        staged_groups.push((group, source_split));
    }

    execute_staged_groups(staged_groups, threads, |(group, source_split)| {
        let prepared = primary_prepared
            .as_ref()
            .filter(|(split, _)| *split == source_split)
            .map(|(_, prepared)| prepared)
            .or_else(|| {
                additional_prepared
                    .as_ref()
                    .and_then(|prepared| prepared.get(&source_split))
            })
            .expect("every staged source split was prepared");
        build_one_multiplicity_free_tree_pair_group(
            source_proof,
            operation,
            &source_axes,
            group,
            prepared,
        )
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "the oriented planner keeps logical keys, storage authority, projection, orientation, rank, and thread policy explicit"
)]
pub(crate) fn build_oriented_tree_pair_transform_group_plan_capability_validated<R>(
    rule: &R,
    operation: TreeTransformOperation,
    logical_keys: &[FusionTreePairKey],
    storage_structure: &BlockStructure,
    orientation: FusionTreePairOrientation,
    basis_order: OrientedBasisOrder,
    logical_rank: usize,
    storage_projection: &FxHashMap<&FusionTreePairKey, usize>,
    threads: usize,
    reuse: Option<&CoefficientGroupReuse<R::Scalar>>,
) -> Result<ComposedPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar:
        'static + Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero + Send + Sync,
{
    // The sole cache caller admits capability before lookup, for both hits
    // and misses; do not query mutable provider capability a second time here.
    validate_tree_transform_rank_syntax(&operation, logical_rank)?;
    let source_axes = operation_source_axes(&operation);
    // Why the logical keys suffice for a group: the adjoint orientation maps
    // each logical tree pair to exactly one storage tree pair, and that pair
    // is all the builder reads at the projected storage index.
    let mut group_keys = Vec::new();
    let mut staged_groups = Vec::<OrientedGroup<'_>>::new();
    if basis_order == OrientedBasisOrder::Storage {
        // Swapping whole trees is a bijection on group keys. Parent groups
        // therefore preserve logical first appearance and member order without
        // rebuilding a hash grouping or allocating per-group index vectors.
        // TensorKit likewise traverses an adjoint through its parent's blocks:
        // https://github.com/QuantumKitHub/TensorKit.jl/blob/cfaa073e4d1e3eb2167edcbdc3be9872f41e7d91/src/tensors/adjoint.jl#L29-L44
        // The sole production Storage constructor makes one swapped key per
        // present parent block and the identity storage-index projection.
        for group in storage_structure.fusion_tree_group_slice() {
            let indices = group.block_indices();
            let key = &logical_keys[indices[0]];
            if reuse.is_some() {
                group_keys.push(key.group_key());
            }
            staged_groups.push(OrientedGroup {
                logical_indices: None,
                storage_indices: Cow::Borrowed(indices),
                source_split: (
                    key.codomain_tree().uncoupled().len(),
                    key.domain_tree().uncoupled().len(),
                ),
            });
        }
    } else {
        let mut group_indices = FxHashMap::default();
        for (logical_index, key) in logical_keys.iter().enumerate() {
            let group_index = match group_indices.entry(key.group_key()) {
                Entry::Occupied(entry) => *entry.get(),
                Entry::Vacant(entry) => {
                    let group_index = staged_groups.len();
                    if reuse.is_some() {
                        group_keys.push(entry.key().clone());
                    }
                    entry.insert(group_index);
                    staged_groups.push(OrientedGroup {
                        logical_indices: Some(Vec::new()),
                        storage_indices: Cow::Owned(Vec::new()),
                        source_split: (
                            key.codomain_tree().uncoupled().len(),
                            key.domain_tree().uncoupled().len(),
                        ),
                    });
                    group_index
                }
            };
            staged_groups[group_index]
                .logical_indices
                .as_mut()
                .expect("canonical group")
                .push(logical_index);
            staged_groups[group_index].storage_indices.to_mut().push(
                *storage_projection
                    .get(key)
                    .ok_or(OperationError::StructureMismatch {
                        tensor: "oriented source projection",
                    })?,
            );
        }
    }

    let flat_keys: Option<Vec<&FusionTreePairKey>> = reuse.map(|_| {
        staged_groups
            .iter()
            .flat_map(|group| {
                group
                    .logical_indices()
                    .iter()
                    .map(|&index| &logical_keys[index])
            })
            .collect()
    });
    let groups = flat_keys.as_deref().map(|keys| {
        split_source_groups(
            group_keys
                .iter()
                .zip(&staged_groups)
                .map(|(group_key, group)| (group_key, group.storage_indices.len())),
            keys,
        )
    });
    let group_count = staged_groups.len();
    build_groups_reusing(reuse.zip(groups.as_deref()), group_count, |missing| {
        let mut staged_groups = staged_groups.iter();
        let mut next = 0;
        let staged_groups = missing
            .into_iter()
            .map(|index| {
                let group = staged_groups
                    .nth(index - next)
                    .expect("missing group index");
                next = index + 1;
                group
            })
            .collect::<Vec<_>>();
        build_oriented_tree_pair_groups(
            rule,
            &operation,
            storage_structure,
            orientation,
            &source_axes,
            logical_keys,
            staged_groups,
            threads,
        )
    })
}

/// Builds the listed staged oriented groups, in order.
#[allow(clippy::too_many_arguments)] // Borrow the existing source/projection state without a second owner.
fn build_oriented_tree_pair_groups<R>(
    rule: &R,
    operation: &TreeTransformOperation,
    storage_structure: &BlockStructure,
    orientation: FusionTreePairOrientation,
    source_axes: &Arc<[usize]>,
    logical_keys: &[FusionTreePairKey],
    staged_groups: Vec<&OrientedGroup<'_>>,
    threads: usize,
) -> Result<Vec<Vec<TreeTransformGroupBlockSpec<R::Scalar>>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero + Send + Sync,
{
    let mut primary_prepared = None;
    let mut additional_prepared = None::<FxHashMap<(usize, usize), PreparedTreePairOperation<'_>>>;
    for group in &staged_groups {
        prepared_tree_pair_operation_for_split(
            &mut primary_prepared,
            &mut additional_prepared,
            rule,
            operation,
            group.source_split,
        )?;
    }

    execute_staged_groups(staged_groups, threads, |group| {
        let storage_indices = group.storage_indices.as_ref();
        let source_split = group.source_split;
        let prepared = primary_prepared
            .as_ref()
            .filter(|(split, _)| *split == source_split)
            .map(|(_, prepared)| prepared)
            .or_else(|| {
                additional_prepared
                    .as_ref()
                    .and_then(|prepared| prepared.get(&source_split))
            })
            .expect("every oriented source split was prepared");
        let ordered = match operation.kind() {
            TreeTransformOperationKind::Transpose => {
                multiplicity_free_transpose_tree_pair_block_ordered_indexed(
                    rule,
                    storage_structure,
                    storage_indices,
                    orientation,
                    prepared,
                )
            }
            TreeTransformOperationKind::Permute | TreeTransformOperationKind::Braid => {
                multiplicity_free_braid_tree_pair_block_ordered_indexed(
                    rule,
                    storage_structure,
                    storage_indices,
                    orientation,
                    prepared,
                )
            }
        }
        .map_err(OperationError::from_core_preserving_context)?;
        assemble_ordered_tree_pair_group_specs_from_keys(
            group
                .logical_indices()
                .iter()
                .map(|&index| logical_keys[index].clone()),
            source_axes,
            ordered,
        )
    })
}

fn assemble_ordered_tree_pair_group_specs<T>(
    src_structure: &BlockStructure,
    group: &FusionTreeBlockGroup,
    source_axes: &Arc<[usize]>,
    ordered: OrderedBlockLinearMap<FusionTreePairKey, T>,
) -> Result<Vec<TreeTransformGroupBlockSpec<T>>, OperationError>
where
    T: Clone + Zero,
{
    #[cfg(test)]
    crate::probe::hit(crate::probe::Event::OrderedTreePairLowering);

    let mut src_keys = Vec::with_capacity(group.block_indices().len());
    for &src_block_index in group.block_indices() {
        let block = src_structure.block(src_block_index)?;
        let BlockKey::FusionTree(src_key) = block.key() else {
            return Err(OperationError::ExpectedFusionTreeBlock {
                tensor: "src",
                index: src_block_index,
            });
        };
        src_keys.push(src_key.clone());
    }

    assemble_ordered_tree_pair_group_specs_from_keys(src_keys.into_iter(), source_axes, ordered)
}

fn assemble_ordered_tree_pair_group_specs_from_keys<T, I>(
    src_keys: I,
    source_axes: &Arc<[usize]>,
    ordered: OrderedBlockLinearMap<FusionTreePairKey, T>,
) -> Result<Vec<TreeTransformGroupBlockSpec<T>>, OperationError>
where
    T: Clone + Zero,
    I: ExactSizeIterator<Item = FusionTreePairKey>,
{
    let (destinations, source_count, storage) = ordered.into_parts();
    if source_count != src_keys.len() {
        return Err(OperationError::StructureMismatch {
            tensor: "ordered tree-pair source columns",
        });
    }
    if destinations.is_empty() {
        return Err(OperationError::EmptyTransformBlock);
    }

    match storage {
        OrderedBlockLinearStorage::SingletonColumns {
            destination_rows,
            coefficients,
        } => {
            if destination_rows.len() != source_count || coefficients.len() != source_count {
                return Err(OperationError::StructureMismatch {
                    tensor: "ordered singleton tree-pair columns",
                });
            }
            if destinations.len() == source_count
                && destination_rows
                    .iter()
                    .enumerate()
                    .all(|(index, &row)| index == row)
            {
                return Ok(src_keys
                    .zip(destinations)
                    .zip(coefficients)
                    .map(|((source, destination), coefficient)| {
                        TreeTransformGroupBlockSpec::single(destination, source, coefficient)
                            .with_shared_source_axes(Arc::clone(source_axes))
                    })
                    .collect());
            }
            let mut destination_seen = vec![false; destinations.len()];
            let mut is_injective = true;
            for &destination_row in &destination_rows {
                let Some(seen) = destination_seen.get_mut(destination_row) else {
                    return Err(OperationError::StructureMismatch {
                        tensor: "ordered singleton destination row",
                    });
                };
                if std::mem::replace(seen, true) {
                    is_injective = false;
                }
            }
            if is_injective {
                let mut destination_slots = destinations.into_iter().map(Some).collect::<Vec<_>>();
                return Ok(src_keys
                    .zip(destination_rows)
                    .zip(coefficients)
                    .map(|((source, destination_row), coefficient)| {
                        TreeTransformGroupBlockSpec::single(
                            destination_slots[destination_row]
                                .take()
                                .expect("injective destination row is consumed once"),
                            source,
                            coefficient,
                        )
                        .with_shared_source_axes(Arc::clone(source_axes))
                    })
                    .collect());
            }

            let mut dense = vec![T::zero(); destinations.len().saturating_mul(source_count)];
            for (source, (destination_row, coefficient)) in
                destination_rows.into_iter().zip(coefficients).enumerate()
            {
                dense[destination_row * source_count + source] = coefficient;
            }
            Ok(vec![TreeTransformGroupBlockSpec::try_multi(
                destinations,
                src_keys,
                dense,
            )?
            .with_shared_source_axes(Arc::clone(source_axes))])
        }
        OrderedBlockLinearStorage::DenseDstSrc(coefficients) => {
            let expected = destinations.len().saturating_mul(source_count);
            if coefficients.len() != expected {
                return Err(OperationError::StructureMismatch {
                    tensor: "ordered dense tree-pair coefficients",
                });
            }
            let coefficients = coefficients
                .into_iter()
                .map(|coefficient| coefficient.unwrap_or_else(T::zero))
                .collect();
            Ok(vec![TreeTransformGroupBlockSpec::try_multi(
                destinations,
                src_keys,
                coefficients,
            )?
            .with_shared_source_axes(Arc::clone(source_axes))])
        }
    }
}

/// Compile a checked Generic permute, braid, or planar transpose plan.
///
/// The complete source structure is admitted before the first rigidity or F/R
/// lookup. A failure returns no plan. This standalone entry is uncached: it
/// neither reads nor publishes the composed-coefficient cache.
///
/// Each source group is composed as one whole block, step by step (TensorKit
/// `GenericTreeTransformer`, `treetransformers.jl:53-114` at `cfaa073e`). The
/// first provider failure is that of the lowest basis row of the lowest
/// failing elementary step of the lowest failing group (#1962, approved
/// step-major order). A grouped (Multi) spec lists destinations source-major
/// by first appearance; the order among one source's destinations follows
/// the block composer and is not the per-pair term order, so consumers
/// resolve destinations by key.
#[cfg(test)]
pub fn build_checked_generic_tree_pair_transform_group_plan<P>(
    provider: &P,
    operation: TreeTransformOperation,
    src_structure: &BlockStructure,
) -> Result<TreeTransformGroupPlan<P::Scalar>, CheckedGenericPlanError<P::Error>>
where
    P: CheckedGenericRigidSymbols,
    P::Scalar: CategoricalScalar + Zero,
{
    let source_proof =
        validate_checked_generic_tree_pair_preflight(provider, None, &operation, src_structure)?;
    let group_count = source_proof.structure().fusion_tree_group_slice().len();
    let groups = build_checked_generic_tree_pair_groups(
        &operation,
        &source_proof,
        (0..group_count).collect(),
    )?;
    Ok(TreeTransformGroupPlan::from_specs(
        groups.into_iter().flatten(),
    ))
}

/// The cached checked producer: the source groups the composed-coefficient
/// cache misses are built, in index order, and staged on `reuse`. Why not
/// the standalone entry above: erased entries need `'static + Send + Sync`
/// coefficients, which that public API does not require.
pub(crate) fn build_checked_generic_tree_pair_transform_group_plan_validated<P>(
    operation: TreeTransformOperation,
    source_proof: &CheckedGenericTreePairPreflight<'_, '_, P>,
    reuse: &CoefficientGroupReuse<P::Scalar>,
) -> Result<ComposedPlan<P::Scalar>, CheckedGenericPlanError<P::Error>>
where
    P: CheckedGenericRigidSymbols,
    P::Scalar: 'static + CategoricalScalar + Zero + Send + Sync,
{
    // The preflight admitted a fusion style with multiplicity, so the Unique
    // exclusion of `build_tree_pair_transform_group_plan_validated_reusing`
    // never applies here.
    let structure = source_proof.structure();
    let all_groups = structure.fusion_tree_group_slice();
    let flat_keys = source_group_key_refs(structure)?;
    let groups = split_source_groups(
        all_groups
            .iter()
            .map(|group| (group.group_key(), group.block_indices().len())),
        &flat_keys,
    );
    build_groups_reusing(Some((reuse, &groups)), all_groups.len(), |missing| {
        build_checked_generic_tree_pair_groups(&operation, source_proof, missing)
    })
}

type CheckedGroupSpecs<P> = Result<
    Vec<Vec<TreeTransformGroupBlockSpec<<P as CheckedGenericRigidSymbols>::Scalar>>>,
    CheckedGenericPlanError<<P as symtn_core::CheckedGenericFusion>::Error>,
>;

/// Builds the listed source groups, in order: the first provider error is
/// that of the lowest failing listed group.
fn build_checked_generic_tree_pair_groups<P>(
    operation: &TreeTransformOperation,
    source_proof: &CheckedGenericTreePairPreflight<'_, '_, P>,
    group_indices: Vec<usize>,
) -> CheckedGroupSpecs<P>
where
    P: CheckedGenericRigidSymbols,
    P::Scalar: CategoricalScalar + Zero,
{
    let all_groups = source_proof.structure().fusion_tree_group_slice();
    let source_axes = operation_source_axes(operation);
    group_indices
        .into_iter()
        .map(|index| {
            build_one_checked_generic_tree_pair_group(
                operation,
                &source_axes,
                source_proof,
                &all_groups[index],
            )
        })
        .collect()
}

fn build_one_checked_generic_tree_pair_group<P>(
    operation: &TreeTransformOperation,
    source_axes: &Arc<[usize]>,
    source_proof: &CheckedGenericTreePairPreflight<'_, '_, P>,
    group: &FusionTreeBlockGroup,
) -> Result<Vec<TreeTransformGroupBlockSpec<P::Scalar>>, CheckedGenericPlanError<P::Error>>
where
    P: CheckedGenericRigidSymbols,
    P::Scalar: CategoricalScalar + Zero,
{
    let admitted = &source_proof.admitted;
    let indices = group.block_indices();
    let ordered = match operation.kind() {
        TreeTransformOperationKind::Permute => admitted.generic_permute_ordered_for_block_indices(
            indices,
            operation.codomain_permutation(),
            operation.domain_permutation(),
        ),
        TreeTransformOperationKind::Braid => admitted.generic_braid_ordered_for_block_indices(
            indices,
            operation.codomain_permutation(),
            operation.domain_permutation(),
            operation.codomain_levels(),
            operation.domain_levels(),
        ),
        TreeTransformOperationKind::Transpose => admitted
            .generic_transpose_ordered_for_block_indices(
                indices,
                operation.codomain_permutation(),
                operation.domain_permutation(),
            ),
    }
    .map_err(map_checked_generic_symbol_error)?;
    assemble_ordered_tree_pair_group_specs(source_proof.structure(), group, source_axes, ordered)
        .map_err(CheckedGenericPlanError::Operation)
}

#[cfg(test)]
/// Generic-fusion (outer-multiplicity) tree-pair plan compile — the dispatch
/// receptacle for outer-multiplicity rules. Parallel entry to
/// `build_multiplicity_free_tree_pair_transform_group_plan`: non-identity groups
/// are compiled as one ordered block map over full tree-pair keys, while the
/// identity path keeps the existing identity assembly.
///
/// This is a SEPARATE entry rather than a runtime branch inside the mult-free
/// builder because the two are disjoint at the type level:
/// `GenericRigidSymbols` and `MultiplicityFreeRigidSymbols` are never both
/// implemented by a real rule, so a mult-free rule can never name this
/// function's bound, let alone reach its row-generation body. Both paths do
/// intentionally share group-spec assembly, including structural monomial
/// lowering, while retaining separate fusion-style guards and symbol APIs.
/// Why not call this a byte-for-byte or blanket zero-cost guarantee: changes to
/// the shared assembler are expected to affect both paths; the guarantee is
/// that multiplicity-free rules never execute generic F/R-symbol logic. The
/// provider-owned [`FusionStyleKind`] gate below defends against a
/// `GenericRigidSymbols` implementation that reports a multiplicity-free
/// style. Fusion style is not duplicated in individual tree keys.
///
/// # Provider-domain precondition
///
/// Fusion-tree block keys in `src_structure` follow
/// [`symtn_core::FusionTreeKey::validate_for_rule`]'s provider-domain
/// precondition.
#[cfg(test)]
pub fn build_generic_tree_pair_transform_group_plan<R>(
    rule: &R,
    operation: TreeTransformOperation,
    src_structure: &BlockStructure,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar + Zero,
{
    let source_proof = validate_generic_tree_pair_preflight(rule, &operation, src_structure)?;
    build_generic_tree_pair_transform_group_plan_validated(&source_proof, operation)
}

#[cfg(test)]
pub(crate) fn build_generic_tree_pair_transform_group_plan_validated<R>(
    source_proof: &LocallyValidatedFusionTreeBlockStructure<'_, '_, R>,
    operation: TreeTransformOperation,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar + Zero,
{
    let src_structure = source_proof.structure();
    let source_axes = operation_source_axes(&operation);

    let mut specs = Vec::new();
    for group in src_structure.fusion_tree_group_slice() {
        if operation.is_identity_for(
            group.group_key().codomain_uncoupled().len(),
            group.group_key().domain_uncoupled().len(),
        ) {
            let mut rows_for = |index: usize, _: &FusionTreePairKey| {
                let rows = match operation.kind() {
                    TreeTransformOperationKind::Permute => source_proof
                        .generic_permute_tree_pair_for_block_index(
                            index,
                            operation.codomain_permutation(),
                            operation.domain_permutation(),
                        ),
                    TreeTransformOperationKind::Braid => source_proof
                        .generic_braid_tree_pair_for_block_index(
                            index,
                            operation.codomain_permutation(),
                            operation.domain_permutation(),
                            operation.codomain_levels(),
                            operation.domain_levels(),
                        ),
                    TreeTransformOperationKind::Transpose => source_proof
                        .generic_transpose_tree_pair_for_block_index(
                            index,
                            operation.codomain_permutation(),
                            operation.domain_permutation(),
                        ),
                }
                .map_err(OperationError::from_core_preserving_context)?;
                Ok(Arc::new(rows))
            };
            specs.extend(assemble_identity_tree_pair_group_specs(
                src_structure,
                group,
                &source_axes,
                &mut rows_for,
            )?);
        } else {
            let mut src_keys = Vec::with_capacity(group.block_indices().len());
            for &src_block_index in group.block_indices() {
                let block = src_structure.block(src_block_index)?;
                let BlockKey::FusionTree(src_key) = block.key() else {
                    return Err(OperationError::ExpectedFusionTreeBlock {
                        tensor: "src",
                        index: src_block_index,
                    });
                };
                src_keys.push(src_key.clone());
            }
            let ordered = match operation.kind() {
                TreeTransformOperationKind::Permute => generic_permute_tree_pair_block_ordered(
                    source_proof.rule(),
                    &src_keys,
                    operation.codomain_permutation(),
                    operation.domain_permutation(),
                ),
                TreeTransformOperationKind::Braid => generic_braid_tree_pair_block_ordered(
                    source_proof.rule(),
                    &src_keys,
                    operation.codomain_permutation(),
                    operation.domain_permutation(),
                    operation.codomain_levels(),
                    operation.domain_levels(),
                ),
                TreeTransformOperationKind::Transpose => generic_transpose_tree_pair_block_ordered(
                    source_proof.rule(),
                    &src_keys,
                    operation.codomain_permutation(),
                    operation.domain_permutation(),
                ),
            }
            .map_err(OperationError::from_core_preserving_context)?;
            specs.extend(assemble_ordered_tree_pair_group_specs(
                src_structure,
                group,
                &source_axes,
                ordered,
            )?);
        }
    }

    Ok(TreeTransformGroupPlan::new(specs))
}

fn assemble_identity_tree_pair_group_specs<T, F>(
    src_structure: &BlockStructure,
    group: &FusionTreeBlockGroup,
    source_axes: &Arc<[usize]>,
    rows_for: &mut F,
) -> Result<Vec<TreeTransformGroupBlockSpec<T>>, OperationError>
where
    T: Clone,
    F: FnMut(usize, &FusionTreePairKey) -> Result<Arc<Vec<(FusionTreePairKey, T)>>, OperationError>,
{
    let mut specs = Vec::with_capacity(group.block_indices().len());
    for &src_block_index in group.block_indices() {
        let block = src_structure.block(src_block_index)?;
        let BlockKey::FusionTree(src_key) = block.key() else {
            return Err(OperationError::ExpectedFusionTreeBlock {
                tensor: "src",
                index: src_block_index,
            });
        };
        let transformed = rows_for(src_block_index, src_key)?;
        let [(dst_key, coefficient)] = transformed.as_slice() else {
            return Err(OperationError::EmptyTransformBlock);
        };
        // Identity rows are singleton by construction. Why not synthesize the
        // coefficient here: consuming the validated row preserves scalar
        // conversion semantics across serial and parallel builders.
        specs.push(
            TreeTransformGroupBlockSpec::single(
                dst_key.clone(),
                src_key.clone(),
                coefficient.clone(),
            )
            .with_shared_source_axes(Arc::clone(source_axes)),
        );
    }
    Ok(specs)
}

#[cfg(test)]
/// Assemble one group's block specs (destination-key dedup plus the
/// `U[dst, src]` recoupling coefficient matrix) from per-tree recoupling
/// rows. Groups are independent, which is what lets the parallel compile map
/// over them.
fn assemble_tree_pair_group_specs<T, F>(
    src_structure: &BlockStructure,
    group: &FusionTreeBlockGroup,
    source_axes: &Arc<[usize]>,
    rows_for: &mut F,
) -> Result<Vec<TreeTransformGroupBlockSpec<T>>, OperationError>
where
    T: Clone + Add<Output = T> + Zero,
    F: FnMut(usize, &FusionTreePairKey) -> Result<Arc<Vec<(FusionTreePairKey, T)>>, OperationError>,
{
    assemble_tree_pair_group_specs_result(src_structure, group, source_axes, rows_for)
}

#[cfg(test)]
fn assemble_tree_pair_group_specs_result<T, E, F>(
    src_structure: &BlockStructure,
    group: &FusionTreeBlockGroup,
    source_axes: &Arc<[usize]>,
    rows_for: &mut F,
) -> Result<Vec<TreeTransformGroupBlockSpec<T>>, E>
where
    T: Clone + Add<Output = T> + Zero,
    E: From<CoreError> + From<OperationError>,
    F: FnMut(usize, &FusionTreePairKey) -> Result<Arc<Vec<(FusionTreePairKey, T)>>, E>,
{
    let src_block_indices = group.block_indices();
    let mut src_keys = Vec::<FusionTreePairKey>::with_capacity(src_block_indices.len());
    let mut dst_keys = Vec::<FusionTreePairKey>::new();
    let mut dst_index_by_key = FxHashMap::<FusionTreePairKey, usize>::default();
    let mut rows = DenseRecouplingRows::new(src_block_indices.len());
    let mut direct_rows = Vec::with_capacity(src_block_indices.len());
    let mut direct_dst_keys = FxHashSet::default();
    let mut is_injective_singleton = true;

    for (src_column, &src_block_index) in src_block_indices.iter().enumerate() {
        let block = src_structure.block(src_block_index)?;
        let BlockKey::FusionTree(src_key) = block.key() else {
            return Err(OperationError::ExpectedFusionTreeBlock {
                tensor: "src",
                index: src_block_index,
            }
            .into());
        };
        src_keys.push(src_key.clone());

        let transformed = rows_for(src_block_index, src_key)?;
        if let [(dst_tree_key, coefficient)] = transformed.as_slice() {
            let dst_key = dst_tree_key.clone();
            if !direct_dst_keys.insert(dst_key.clone()) {
                is_injective_singleton = false;
            }
            direct_rows.push((src_key.clone(), dst_key, coefficient.clone()));
        } else {
            is_injective_singleton = false;
        }

        for (dst_tree_key, coefficient) in transformed.iter() {
            let dst_key = dst_tree_key.clone();
            let dst_row = if let Some(&dst_row) = dst_index_by_key.get(&dst_key) {
                dst_row
            } else {
                let dst_row = dst_keys.len();
                dst_index_by_key.insert(dst_key.clone(), dst_row);
                dst_keys.push(dst_key);
                let allocated_row = rows.push_zero_row();
                debug_assert_eq!(allocated_row, dst_row);
                allocated_row
            };
            rows.add(dst_row, src_column, coefficient.clone());
        }
    }

    if let Some(direct_specs) = lower_injective_singleton_rows(
        source_axes,
        direct_rows,
        src_keys.len(),
        is_injective_singleton,
    ) {
        return Ok(direct_specs);
    }
    if dst_keys.is_empty() {
        return Err(OperationError::EmptyTransformBlock.into());
    }

    Ok(vec![TreeTransformGroupBlockSpec::try_multi(
        dst_keys,
        src_keys,
        rows.into_coefficients(),
    )
    .map_err(E::from)?
    .with_shared_source_axes(Arc::clone(source_axes))])
}

fn lower_injective_singleton_rows<T>(
    source_axes: &Arc<[usize]>,
    direct_rows: Vec<(FusionTreePairKey, FusionTreePairKey, T)>,
    src_count: usize,
    is_injective_singleton: bool,
) -> Option<Vec<TreeTransformGroupBlockSpec<T>>> {
    if !is_injective_singleton || direct_rows.len() != src_count || direct_rows.is_empty() {
        return None;
    }

    // Row cardinality plus destination-key injectivity proves independent
    // replay. Why not inspect coefficient values: phases and non-unit scalars
    // are valid direct maps, while numerical zeros cannot prove structure.
    Some(
        direct_rows
            .into_iter()
            .map(|(src_key, dst_key, coefficient)| {
                TreeTransformGroupBlockSpec::single(dst_key, src_key, coefficient)
                    .with_shared_source_axes(Arc::clone(source_axes))
            })
            .collect(),
    )
}

pub(crate) fn build_unique_tree_pair_transform_group_plan<R>(
    rule: &R,
    operation: TreeTransformOperation,
    src_structure: &BlockStructure,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero,
{
    // A capability guard, not a family choice: this entry point implements
    // only the bijective lowering, so a `Simple` rule is rejected rather than
    // rerouted.
    if rule.fusion_style() != FusionStyleKind::Unique {
        return Err(OperationError::UnsupportedFusionStyle {
            operation: Box::new(operation),
            style: rule.fusion_style(),
        });
    }
    let source_proof =
        validate_multiplicity_free_tree_pair_preflight(rule, &operation, src_structure)?;
    build_unique_tree_pair_transform_group_plan_validated(&source_proof, operation)
}

fn build_unique_tree_pair_transform_group_plan_validated<R>(
    source_proof: &LocallyValidatedFusionTreeBlockStructure<'_, '_, R>,
    operation: TreeTransformOperation,
) -> Result<TreeTransformGroupPlan<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    debug_assert_eq!(source_proof.rule().fusion_style(), FusionStyleKind::Unique);
    let src_structure = source_proof.structure();
    let source_axes = operation_source_axes(&operation);
    let mut primary_prepared = None;
    let mut additional_prepared = None::<FxHashMap<(usize, usize), PreparedTreePairOperation<'_>>>;
    let mut specs = Vec::with_capacity(src_structure.block_count());
    let mut scratch = UniqueTreePairScratch::default();
    for index in 0..src_structure.block_count() {
        specs.extend(unique_tree_pair_spec(
            source_proof,
            &operation,
            &source_axes,
            &mut primary_prepared,
            &mut additional_prepared,
            &mut scratch,
            index,
        )?);
    }
    Ok(TreeTransformGroupPlan::new(specs))
}

/// The one-term Unique spec of source block `index`; `None` for a block
/// without a fusion-tree key.
fn unique_tree_pair_spec<'p, 'o, R>(
    source_proof: &LocallyValidatedFusionTreeBlockStructure<'_, '_, R>,
    operation: &'o TreeTransformOperation,
    source_axes: &Arc<[usize]>,
    primary_prepared: &'p mut Option<((usize, usize), PreparedTreePairOperation<'o>)>,
    additional_prepared: &'p mut Option<FxHashMap<(usize, usize), PreparedTreePairOperation<'o>>>,
    scratch: &mut UniqueTreePairScratch,
    index: usize,
) -> Result<Option<TreeTransformGroupBlockSpec<R::Scalar>>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    let Some(src_key) = source_proof.fusion_tree_pair_key(index)? else {
        return Ok(None);
    };
    let source_split = (
        src_key.codomain_tree().uncoupled().len(),
        src_key.domain_tree().uncoupled().len(),
    );
    let prepared = prepared_tree_pair_operation_for_split(
        primary_prepared,
        additional_prepared,
        source_proof.rule(),
        operation,
        source_split,
    )?;
    let transformed = source_proof
        .execute_unique_rigid_for_block_index(index, prepared, scratch)
        .map_err(OperationError::from_core_preserving_context)?;
    Ok(Some(
        TreeTransformGroupBlockSpec::single(transformed.0, src_key.clone(), transformed.1)
            .with_shared_source_axes(Arc::clone(source_axes)),
    ))
}

fn prepared_tree_pair_operation_for_split<'prepared, 'operation, R>(
    primary: &'prepared mut Option<((usize, usize), PreparedTreePairOperation<'operation>)>,
    additional: &'prepared mut Option<
        FxHashMap<(usize, usize), PreparedTreePairOperation<'operation>>,
    >,
    rule: &R,
    operation: &'operation TreeTransformOperation,
    source_split: (usize, usize),
) -> Result<&'prepared PreparedTreePairOperation<'operation>, OperationError>
where
    R: FusionRule,
{
    if primary.is_none() {
        *primary = Some((
            source_split,
            prepare_tree_pair_operation(rule, operation, source_split)?,
        ));
    }
    if let Some((primary_split, prepared)) = primary.as_ref() {
        if *primary_split == source_split {
            return Ok(prepared);
        }
    }
    let prepared_by_split = additional.get_or_insert_with(FxHashMap::default);
    let prepared = match prepared_by_split.entry(source_split) {
        Entry::Occupied(entry) => entry.into_mut(),
        Entry::Vacant(entry) => {
            entry.insert(prepare_tree_pair_operation(rule, operation, source_split)?)
        }
    };
    Ok(prepared)
}

fn prepare_tree_pair_operation<'operation, R>(
    rule: &R,
    operation: &'operation TreeTransformOperation,
    (source_codomain_rank, source_domain_rank): (usize, usize),
) -> Result<PreparedTreePairOperation<'operation>, OperationError>
where
    R: FusionRule,
{
    #[cfg(test)]
    crate::probe::hit(crate::probe::Event::TreePairOperationPreparation);
    let prepared = match operation.kind() {
        TreeTransformOperationKind::Permute => match operation.raw_axis_positions() {
            Some(raw_axis_positions) => {
                PreparedTreePairOperation::prepare_permute_with_raw_axis_positions(
                    rule,
                    source_codomain_rank,
                    source_domain_rank,
                    operation.codomain_permutation(),
                    operation.domain_permutation(),
                    raw_axis_positions,
                )
            }
            // Why not require this internal segment here: operation
            // construction deliberately accepts malformed metadata so the
            // established validation error remains observable at execution.
            None => PreparedTreePairOperation::prepare_permute(
                rule,
                source_codomain_rank,
                source_domain_rank,
                operation.codomain_permutation(),
                operation.domain_permutation(),
            ),
        },
        TreeTransformOperationKind::Braid => match operation.raw_axis_positions() {
            Some(raw_axis_positions) => {
                PreparedTreePairOperation::prepare_braid_with_raw_axis_positions(
                    rule,
                    source_codomain_rank,
                    source_domain_rank,
                    operation.codomain_permutation(),
                    operation.domain_permutation(),
                    operation.codomain_levels(),
                    operation.domain_levels(),
                    raw_axis_positions,
                )
            }
            None => PreparedTreePairOperation::prepare_braid(
                rule,
                source_codomain_rank,
                source_domain_rank,
                operation.codomain_permutation(),
                operation.domain_permutation(),
                operation.codomain_levels(),
                operation.domain_levels(),
            ),
        },
        TreeTransformOperationKind::Transpose => PreparedTreePairOperation::prepare_transpose(
            source_codomain_rank,
            source_domain_rank,
            operation.codomain_permutation(),
            operation.domain_permutation(),
        ),
    }
    .map_err(OperationError::from_core_preserving_context)?;
    Ok(prepared.into_compiler_owned_simple_braid())
}

fn prepare_tree_pair_operation_syntax(
    operation: &TreeTransformOperation,
    (source_codomain_rank, source_domain_rank): (usize, usize),
) -> Result<(), OperationError> {
    match operation.kind() {
        TreeTransformOperationKind::Permute => PreparedTreePairOperation::validate_permute_syntax(
            source_codomain_rank,
            source_domain_rank,
            operation.codomain_permutation(),
            operation.domain_permutation(),
        ),
        TreeTransformOperationKind::Braid => PreparedTreePairOperation::validate_braid_syntax(
            source_codomain_rank,
            source_domain_rank,
            operation.codomain_permutation(),
            operation.domain_permutation(),
            operation.codomain_levels(),
            operation.domain_levels(),
        ),
        TreeTransformOperationKind::Transpose => {
            PreparedTreePairOperation::validate_transpose_syntax(
                source_codomain_rank,
                source_domain_rank,
                operation.codomain_permutation(),
                operation.domain_permutation(),
            )
        }
    }
    .map_err(OperationError::from_core_preserving_context)
}

fn validate_all_codomain_operation_scope(
    operation: &TreeTransformOperation,
) -> Result<(), OperationError> {
    let scope_error = || OperationError::UnsupportedTreeTransformScope {
        operation: Box::new(operation.clone()),
        message: "all-codomain UniqueFusion lowering requires an empty domain operation",
    };

    match operation.kind() {
        TreeTransformOperationKind::Permute if operation.domain_permutation().is_empty() => Ok(()),
        TreeTransformOperationKind::Braid
            if operation.domain_permutation().is_empty() && operation.domain_levels().is_empty() =>
        {
            Ok(())
        }
        TreeTransformOperationKind::Permute | TreeTransformOperationKind::Braid => {
            Err(scope_error())
        }
        TreeTransformOperationKind::Transpose => Err(OperationError::UnsupportedTreeTransformScope {
            operation: Box::new(operation.clone()),
            message: "all-codomain UniqueFusion lowering supports explicit Permute or Braid operations",
        }),
    }
}

fn operation_source_axes(operation: &TreeTransformOperation) -> Arc<[usize]> {
    operation
        .codomain_permutation()
        .iter()
        .chain(operation.domain_permutation())
        .copied()
        .collect()
}

fn validate_all_codomain_fusion_tree_block<R>(
    rule: &R,
    index: usize,
    key: &FusionTreePairKey,
) -> Result<(), OperationError>
where
    R: FusionRule,
{
    let domain = key.domain_tree();
    let empty_domain_coupled_is_valid = domain.coupled() == rule.vacuum();
    if domain.uncoupled().is_empty()
        && empty_domain_coupled_is_valid
        && domain.is_dual().is_empty()
        && domain.innerlines().is_empty()
        && domain.vertices().is_empty()
    {
        return Ok(());
    }
    Err(OperationError::ExpectedAllCodomainFusionTree { index })
}
