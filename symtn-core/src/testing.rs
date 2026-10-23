//! One-shot tree operations that tests outside the crate's unit tests use:
//! symtn-tensors' plan tests compare against the tree-pair oracles, and
//! `tests/identity_transform_allocations.rs` pins the allocation contract of
//! the block and unique paths.
//!
//! Not part of the public API: none of these has a production caller, so they
//! are crate-private and only forwarded here under the `testing` feature
//! (#1805).

use super::*;

pub use symtn_sectors::testing::Provider;

pub fn generic_braid_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
    codomain_levels: &[usize],
    domain_levels: &[usize],
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    fusion_tree::generic_braid_tree_pair(
        rule,
        tree_pair,
        codomain_permutation,
        domain_permutation,
        codomain_levels,
        domain_levels,
    )
}

pub fn generic_permute_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: GenericRigidSymbols,
    R::Scalar: CategoricalScalar,
{
    fusion_tree::generic_permute_tree_pair(
        rule,
        tree_pair,
        codomain_permutation,
        domain_permutation,
    )
}

pub fn multiplicity_free_transpose_tree_pair<R>(
    rule: &R,
    tree_pair: &FusionTreePairKey,
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<Vec<(FusionTreePairKey, R::Scalar)>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    fusion_tree::multiplicity_free_transpose_tree_pair(
        rule,
        tree_pair,
        codomain_permutation,
        domain_permutation,
    )
}

/// The direct unique path's zero-allocation contract is pinned through this
/// entry point because the multiplicity-free one returns a `Vec`, so a
/// measurement through it would measure the container.
pub fn unique_permute_tree<R>(
    rule: &R,
    tree: &FusionTreeKey,
    permutation: &[usize],
) -> Result<(FusionTreeKey, R::Scalar), CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Mul<Output = R::Scalar>,
{
    fusion_tree::unique_permute_tree(rule, tree, permutation)
}

#[allow(clippy::type_complexity)]
pub fn multiplicity_free_braid_tree_block<R>(
    rule: &R,
    src_keys: &[FusionTreeKey],
    permutation: &[usize],
    levels: &[usize],
) -> Result<Vec<Vec<(FusionTreeKey, R::Scalar)>>, CoreError>
where
    R: MultiplicityFreeFusionSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    fusion_tree::multiplicity_free_braid_tree_block(rule, src_keys, permutation, levels)
}

#[allow(clippy::type_complexity)]
pub fn multiplicity_free_permute_tree_pair_block<R>(
    rule: &R,
    src_keys: &[FusionTreePairKey],
    codomain_permutation: &[usize],
    domain_permutation: &[usize],
) -> Result<Vec<Vec<(FusionTreePairKey, R::Scalar)>>, CoreError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Clone + Add<Output = R::Scalar> + Mul<Output = R::Scalar>,
{
    fusion_tree::multiplicity_free_permute_tree_pair_block(
        rule,
        src_keys,
        codomain_permutation,
        domain_permutation,
    )
}

/// Marks `structure`'s content canonical, as a complete-HomSpace admission
/// would: lets sibling tests drive completed-transformer publication with a
/// hand-built fixture layout. Production publication requires a real
/// admission.
pub fn mark_structure_canonical(structure: &BlockStructure) {
    structure.content_key().mark_canonical();
}
