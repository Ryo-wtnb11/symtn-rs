use std::sync::Arc;

use symtn_core::BlockStructure;

use crate::transform_structure::{TreeTransformLaneSchedule, TreeTransformParallelSchedule};
use crate::{
    OperationError, TreeTransformBlock, TreeTransformLayoutTable, TreeTransformRecouplingPlan,
    TreeTransformStructure,
};

/// Backend-neutral scratch sizes required by a completed tree transform.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct TreeTransformWorkspaceRequirements {
    pub packed_source_len: usize,
    pub packed_destination_len: usize,
    pub converted_coefficient_len: usize,
    pub fused_index_len_per_worker: usize,
}

/// Read-only synchronous replay view over a completed tree-transform structure.
///
/// This adapter contains no provider, fusion-tree, backend schedule, storage,
/// pointer, or mutable workspace state. Concrete executors remain responsible
/// for placement/capability checks, allocation-range disjointness, and ensuring
/// their workspace satisfies [`Self::workspace_requirements`] before dispatch.
/// Host replay consumes the borrowed descriptors directly; backend-specific
/// scheduling remains outside this view.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TreeTransformTaskView<'a, C> {
    structure: &'a TreeTransformStructure<C>,
    source_len: usize,
    destination_len: usize,
}

impl<C: Copy> TreeTransformStructure<C> {
    pub(crate) fn task_view(&self) -> Result<TreeTransformTaskView<'_, C>, OperationError> {
        let (source_len, destination_len) = self.replay_storage_lens()?;
        Ok(TreeTransformTaskView {
            structure: self,
            source_len,
            destination_len,
        })
    }
}

impl<'a, C: Copy> TreeTransformTaskView<'a, C> {
    /// Process-local identity for this immutable completed structure. Because
    /// layouts cannot change after compilation, this also identifies the exact
    /// layout ordering used by converted coefficients.
    #[inline]
    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn admission_identity(self) -> std::sync::Weak<()> {
        Arc::downgrade(self.structure.identity_marker())
    }

    /// Process-local identity of the coefficient payload, shared by every
    /// structure bound to one payload whatever its layouts: equal identities
    /// mean an equal payload in every dtype it converts to.
    #[inline]
    #[cfg(feature = "cuda")]
    pub(crate) fn coefficient_identity(self) -> std::sync::Weak<()> {
        Arc::downgrade(self.structure.coefficient_identity_marker())
    }
    #[inline]
    pub(crate) fn blocks(self) -> &'a [TreeTransformBlock] {
        self.structure.blocks()
    }

    #[inline]
    pub(crate) fn layouts(self) -> &'a TreeTransformLayoutTable {
        self.structure.layouts()
    }

    /// Scalars of the Single blocks, indexed by their `coefficient`.
    #[inline]
    pub(crate) fn single_coefficients(self) -> &'a [C] {
        self.structure.single_coefficients()
    }

    /// A Multi block's shared `U[dst, src]` matrix, read in place.
    #[inline]
    pub(crate) fn block_matrix(self, block_index: usize) -> Option<&'a [C]> {
        self.structure.block_matrix(block_index)
    }

    #[inline]
    pub(crate) fn coefficient_len(self) -> usize {
        self.structure.coefficient_len()
    }

    /// The logical payload, converted, gathered once per device upload.
    #[cfg(feature = "cuda")]
    pub(crate) fn gather_converted_coefficients_into<U>(
        self,
        out: &mut Vec<U>,
        convert: impl FnMut(C) -> U,
    ) {
        self.structure
            .gather_converted_coefficients_into(out, convert);
    }

    #[inline]
    pub(crate) fn parallel_schedule(self) -> &'a TreeTransformParallelSchedule {
        self.structure.parallel_schedule()
    }

    /// The multi-slot schedule of the structure, built on first use
    /// ([`TreeTransformParallelSchedule::lanes`]).
    pub(crate) fn lanes(self) -> Result<&'a TreeTransformLaneSchedule, OperationError> {
        let structure = self.structure;
        self.parallel_schedule().lanes(
            structure.blocks(),
            structure.layouts(),
            structure.recoupling_plan(),
        )
    }

    pub(crate) fn recoupling_plan(self) -> &'a TreeTransformRecouplingPlan {
        self.structure.recoupling_plan()
    }

    #[inline]
    pub(crate) fn inactive_destination_layouts(self) -> &'a [usize] {
        self.structure.inactive_destination_layouts()
    }

    #[inline]
    pub(crate) fn storage_conjugate(self) -> bool {
        self.structure.storage_conjugate()
    }

    pub(crate) fn workspace_requirements(self) -> TreeTransformWorkspaceRequirements {
        let plan = self.structure.recoupling_plan();
        TreeTransformWorkspaceRequirements {
            packed_source_len: plan.source_len(),
            packed_destination_len: plan.destination_len(),
            converted_coefficient_len: plan.coefficient_len(),
            fused_index_len_per_worker: self.structure.layouts().max_fused_rank(),
        }
    }

    pub(crate) fn validate_workspace_requirements<D>(self) -> Result<(), OperationError> {
        let requirements = self.workspace_requirements();
        for len in [
            requirements.packed_source_len,
            requirements.packed_destination_len,
            requirements.converted_coefficient_len,
        ] {
            core::alloc::Layout::array::<D>(len)
                .map_err(|_| OperationError::ElementCountOverflow)?;
        }
        core::alloc::Layout::array::<usize>(requirements.fused_index_len_per_worker)
            .map_err(|_| OperationError::ElementCountOverflow)?;
        Ok(())
    }

    pub(crate) fn validate_structures_and_lengths(
        self,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst_len: usize,
        src_len: usize,
    ) -> Result<(), OperationError> {
        self.validate_structures(dst_structure, src_structure)?;
        self.validate_lengths(dst_len, src_len)
    }

    pub(crate) fn validate_structures(
        self,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
    ) -> Result<(), OperationError> {
        self.structure
            .validate_replay_structures(dst_structure, src_structure)
    }

    pub(crate) fn validate_lengths(
        self,
        dst_len: usize,
        src_len: usize,
    ) -> Result<(), OperationError> {
        validate_exact_len(self.destination_len, dst_len)?;
        validate_exact_len(self.source_len, src_len)
    }
}

fn validate_exact_len(expected: usize, actual: usize) -> Result<(), OperationError> {
    if expected == actual {
        Ok(())
    } else {
        Err(OperationError::ElementCountMismatch { expected, actual })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TreeTransformBlockSpec;

    #[test]
    fn completed_view_exposes_lengths_and_workspace_requirements() {
        let structure = Arc::new(
            BlockStructure::packed_column_major(1, [vec![2], vec![2], vec![1], vec![1], vec![1]])
                .unwrap(),
        );
        let transform = TreeTransformStructure::compile_structures(
            &structure,
            &structure,
            &[
                TreeTransformBlockSpec::single(2, 2, -1.0_f64),
                TreeTransformBlockSpec::multi(vec![0, 1], vec![0, 1], vec![1.0, 2.0, 3.0, 4.0]),
                TreeTransformBlockSpec::multi(vec![3, 4], vec![3, 4], vec![5.0, 6.0, 7.0, 8.0]),
            ],
        )
        .unwrap();

        let view = transform.task_view().unwrap();
        view.validate_structures_and_lengths(&structure, &structure, 7, 7)
            .unwrap();
        assert_eq!(
            view.workspace_requirements(),
            TreeTransformWorkspaceRequirements {
                packed_source_len: 6,
                packed_destination_len: 6,
                converted_coefficient_len: 8,
                fused_index_len_per_worker: 1,
            }
        );
    }

    #[test]
    fn admission_rejects_structure_or_nonexact_length() {
        let structure = Arc::new(BlockStructure::packed_column_major(1, [vec![2]]).unwrap());
        let other = Arc::new(BlockStructure::packed_column_major(1, [vec![3]]).unwrap());
        let transform = TreeTransformStructure::compile_structures(
            &structure,
            &structure,
            &[TreeTransformBlockSpec::single(0, 0, 1.0_f64)],
        )
        .unwrap();
        let view = transform.task_view().unwrap();

        assert_eq!(
            view.validate_structures_and_lengths(&structure, &structure, 3, 2),
            Err(OperationError::ElementCountMismatch {
                expected: 2,
                actual: 3,
            })
        );
        assert_eq!(
            view.validate_structures_and_lengths(&other, &structure, 3, 2),
            Err(OperationError::StructureMismatch { tensor: "dst" })
        );
    }
}
