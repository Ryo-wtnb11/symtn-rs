use super::*;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use symtn_core::testing::{generic_braid_tree_pair, generic_permute_tree_pair};
use symtn_core::{
    generic_transpose_tree_pair, CheckedGenericFusion, CheckedGenericRigidSymbols, GenericFArray,
    GenericFusionSymbols, GenericRMatrix, GenericRigidSymbols,
};

/// Test view of the logical coefficient payload (an explicit copy).
trait GatheredCoefficients<T> {
    fn gathered_coefficients(&self) -> Vec<T>;
}

impl<T: Copy> GatheredCoefficients<T> for symtn_operations::TreeTransformStructure<T> {
    fn gathered_coefficients(&self) -> Vec<T> {
        let mut coefficients = Vec::new();
        self.gather_recoupling_coefficients_into(&mut coefficients);
        coefficients
    }
}

/// Resolves through the process-global completed-transformer owner as an
/// unbound context does.
fn resolve_tree_pair<R>(
    rule: &R,
    operation: &TreeTransformOperation,
    dst: &Arc<BlockStructure>,
    src: &Arc<BlockStructure>,
    storage_conjugate: bool,
) -> Result<symtn_operations::TreeTransformStructure<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols,
    R::Scalar: Copy
        + core::ops::Add<Output = R::Scalar>
        + core::ops::Mul<Output = R::Scalar>
        + num_traits::Zero
        + Send
        + Sync
        + 'static,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    crate::tree_transform::TreeTransformPlanning::default().resolve_tree_pair(
        engine,
        rule,
        operation,
        dst,
        src,
        storage_conjugate,
    )
}

/// Whether two handles share one replay core (one cache entry).
fn same_core<T>(
    left: &symtn_operations::TreeTransformStructure<T>,
    right: &symtn_operations::TreeTransformStructure<T>,
) -> bool {
    Arc::ptr_eq(left.replay_core(), right.replay_core())
}

/// Marks hand-built fixture layouts canonical so their keys publish, as a
/// complete-HomSpace admission would.
fn mark_canonical<'a>(structures: impl IntoIterator<Item = &'a BlockStructure>) {
    for structure in structures {
        symtn_core::testing::mark_structure_canonical(structure);
    }
}

/// An equal layout under a fresh, never-admitted content id: an expert copy
/// whose transformer keys are lookup-only.
fn expert_copy(structure: &BlockStructure) -> Arc<BlockStructure> {
    let blocks = (0..structure.block_count())
        .map(|index| {
            let block = structure.block(index).unwrap();
            symtn_core::BlockSpec::with_key(
                block.key().clone(),
                block.shape().to_vec(),
                block.strides().to_vec(),
                block.offset(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    Arc::new(BlockStructure::from_blocks_with_rank(structure.rank(), blocks).unwrap())
}

mod admission;
#[cfg(feature = "cgc-gen")]
mod checked_composer_oracles;
#[cfg(feature = "cgc-gen")]
mod checked_failure_order;
mod checked_generic;
mod context_cache;
mod generic;
mod grouped_storage;
mod plan_builder;
mod tree_pair;
mod unique;

use admission::*;
use generic::*;
use plan_builder::*;
