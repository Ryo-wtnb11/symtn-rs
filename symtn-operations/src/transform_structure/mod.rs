use std::ops::Range;
use std::sync::Arc;

use symtn_core::{validate_block_storage_injective, BlockStructure, CoreError};
use symtn_dense::DenseGemmBatchJob;

use crate::kernel_adapter::{normalize_fused_layout, BakedFusedLayout, FusedLayoutScratch};
use crate::strided::offset_to_isize;
use crate::structure_identity::validate_structure_identity;
use crate::transform_plan::{
    ResolvedTreeTransformBlockSpec, TreeTransformBlockSpec, TreeTransformKeyBlockSpec,
};
use crate::OperationError;

mod coefficients;
mod compile;
mod layout_table;
mod schedule;

pub(crate) use self::coefficients::*;
pub use self::compile::*;
pub use self::layout_table::*;
pub use self::schedule::*;

/// Replay-ready tree-transform descriptor: a cheap shared handle. Clone
/// copies three `Arc`s.
///
/// This is the TensorKit-style transformer-build boundary: construction resolves
/// tree keys, block layouts, offsets, and pack/scatter descriptors against
/// concrete source and destination structures. Matrix-valued categorical
/// coefficients may be shared across compatible layout bindings. Hot paths
/// should build this once and replay it through
/// [`crate::TreeTransformBackend::tree_transform_structure_into_raw`] while reusing a
/// backend and workspace.
///
/// The compiled payload is a [`TreeTransformReplay`] core that holds no
/// structure: TensorKit's `AbelianTreeTransformer`/`GenericTreeTransformer`
/// likewise keep only coefficients and `StridedStructure` tuples
/// (`treetransformers.jl:10-129` @cfaa073). The handle binds that core to the
/// caller's destination and source structures, so a process-global cache can
/// retain cores without retaining (or charging) structure content, and a hit
/// rebinds one to the caller's structures without allocating.
///
/// Why not expose mutable compiled fields: the recoupling plan, converted
/// coefficient cache, and threaded replay schedule all derive from the same
/// descriptors. Read them through [`Self::blocks`], [`Self::layouts`], and
/// [`Self::block_coefficients`] so those derived plans cannot go stale after
/// compilation.
///
/// Coefficient layout: `TreeTransformBlock::Single::coefficient` and
/// `TreeTransformBlock::Multi::coefficient_start` index one logical payload
/// holding every Single scalar (spec order) followed by every Multi matrix
/// (spec order). Only the scalars are stored contiguously; each Multi block
/// references its spec's shared matrix, and every binding of one categorical
/// plan shares that plan's payload, so binding never copies a coefficient.
///
/// Blocks and layouts are read through the [`Self::blocks`] and
/// [`Self::layouts`] accessors. Read one block's coefficients with
/// [`Self::block_coefficients`], or copy the whole payload explicitly with
/// [`Self::gather_recoupling_coefficients_into`]. A compiled structure is
/// immutable.
#[derive(Clone, Debug)]
pub struct TreeTransformStructure<T> {
    core: Arc<TreeTransformReplay<T>>,
    dst_structure: Arc<BlockStructure>,
    src_structure: Arc<BlockStructure>,
}

/// The structure-free compiled payload of a [`TreeTransformStructure`]:
/// what the completed-transformer cache retains.
///
/// Its identity marker keys device-side preparation, so every handle bound
/// to one core shares that preparation.
#[doc(hidden)]
#[derive(Debug, PartialEq)]
pub struct TreeTransformReplay<T> {
    rank: usize,
    storage_conjugate: bool,
    identity: Arc<()>,
    blocks: Vec<TreeTransformBlock>,
    layouts: TreeTransformLayoutTable,
    coefficients: Arc<TreeTransformCoefficients<T>>,
    inactive_dst_layouts: Vec<usize>,
    physical_overwrite_len: Option<usize>,
    recoupling_plan: TreeTransformRecouplingPlan,
    parallel_schedule: TreeTransformParallelSchedule,
}

impl<T> TreeTransformReplay<T> {
    /// Conservative retained-byte charge of this core: its own allocation
    /// (inline value and `Arc` control), every owned capacity and the
    /// identity control block. It retains no structure.
    ///
    /// Shared categorical recoupling matrices are conservatively charged in
    /// full for every bound core; aggregate unique-byte accounting belongs
    /// with a future retention policy, not this ownership split.
    pub fn charged_bytes(&self) -> usize {
        const ARC_CONTROL_BYTES: usize = 2 * core::mem::size_of::<usize>();

        let vector_bytes =
            |capacity: usize, element_size: usize| capacity.saturating_mul(element_size);
        core::mem::size_of::<Self>()
            .saturating_add(ARC_CONTROL_BYTES)
            // The identity marker's own allocation.
            .saturating_add(ARC_CONTROL_BYTES)
            .saturating_add(vector_bytes(
                self.blocks.capacity(),
                core::mem::size_of::<TreeTransformBlock>(),
            ))
            .saturating_add(vector_bytes(
                self.layouts.entries.capacity(),
                core::mem::size_of::<TreeTransformLayout>(),
            ))
            .saturating_add(vector_bytes(
                self.layouts.fused_dims.capacity(),
                core::mem::size_of::<usize>(),
            ))
            .saturating_add(vector_bytes(
                self.layouts.fused_dst_strides.capacity(),
                core::mem::size_of::<isize>(),
            ))
            .saturating_add(vector_bytes(
                self.layouts.fused_src_strides.capacity(),
                core::mem::size_of::<isize>(),
            ))
            .saturating_add(self.coefficients.charged_bytes())
            .saturating_add(vector_bytes(
                self.inactive_dst_layouts.capacity(),
                core::mem::size_of::<usize>(),
            ))
            .saturating_add(vector_bytes(
                self.recoupling_plan.block_indices.capacity(),
                core::mem::size_of::<usize>(),
            ))
            .saturating_add(vector_bytes(
                self.recoupling_plan.jobs.capacity(),
                core::mem::size_of::<DenseGemmBatchJob>(),
            ))
            .saturating_add(vector_bytes(
                self.parallel_schedule.singles.capacity(),
                core::mem::size_of::<u32>(),
            ))
            .saturating_add(vector_bytes(
                self.parallel_schedule.pack_columns.capacity(),
                core::mem::size_of::<TreeTransformPackReplay>(),
            ))
            .saturating_add(vector_bytes(
                self.parallel_schedule.scatter_columns.capacity(),
                core::mem::size_of::<TreeTransformScatterReplay>(),
            ))
            .saturating_add(vector_bytes(
                self.parallel_schedule.scatter_groups.capacity(),
                core::mem::size_of::<TreeTransformScatterGroupReplay>(),
            ))
    }
}

impl<T> TreeTransformStructure<T> {
    /// Conservative retained-byte charge of this handle and its core. The
    /// source and destination structures are separate canonical owners, so
    /// only the handle's inline `Arc`s are included through
    /// `size_of::<Self>()`.
    #[doc(hidden)]
    pub fn charged_payload_bytes(&self) -> usize {
        core::mem::size_of::<Self>().saturating_add(self.core.charged_bytes())
    }

    /// The shared structure-free core.
    #[doc(hidden)]
    #[inline]
    pub fn replay_core(&self) -> &Arc<TreeTransformReplay<T>> {
        &self.core
    }

    /// Binds a cached core to the caller's structures. O(1), no allocation.
    ///
    /// The caller proves that `dst_structure` and `src_structure` are the
    /// structures the core was compiled against (equal content); replay
    /// re-checks that identity against the structures it is given.
    #[doc(hidden)]
    #[inline]
    pub fn from_replay_core(
        core: Arc<TreeTransformReplay<T>>,
        dst_structure: Arc<BlockStructure>,
        src_structure: Arc<BlockStructure>,
    ) -> Self {
        Self {
            core,
            dst_structure,
            src_structure,
        }
    }

    /// Destination structure this handle is bound to.
    #[doc(hidden)]
    #[inline]
    pub fn dst_structure(&self) -> &Arc<BlockStructure> {
        &self.dst_structure
    }

    /// Source structure this handle is bound to.
    #[doc(hidden)]
    #[inline]
    pub fn src_structure(&self) -> &Arc<BlockStructure> {
        &self.src_structure
    }
}

// Why manual: the same core under content-equal structures is equal, and the
// shared core short-circuits the payload comparison.
impl<T: PartialEq> PartialEq for TreeTransformStructure<T> {
    fn eq(&self, other: &Self) -> bool {
        (Arc::ptr_eq(&self.core, &other.core) || self.core == other.core)
            && self.dst_structure == other.dst_structure
            && self.src_structure == other.src_structure
    }
}

impl<T: Copy> TreeTransformStructure<T> {
    #[inline]
    pub fn rank(&self) -> usize {
        self.core.rank
    }

    #[inline]
    pub fn block_count(&self) -> usize {
        self.core.blocks.len()
    }

    /// Immutable compiled block descriptors.
    #[inline]
    pub fn blocks(&self) -> &[TreeTransformBlock] {
        &self.core.blocks
    }

    /// Immutable compiled layout table: per entry its offset and element
    /// count. Its geometry is read through [`Self::layout_diagnostics`].
    #[inline]
    pub fn layouts(&self) -> &TreeTransformLayoutTable {
        &self.core.layouts
    }

    /// Owned copy of every layout entry's role and compiled normalized
    /// geometry, for diagnostics and tests.
    ///
    /// Cost: four allocations and O(entries + Σf) copies per call, where f is
    /// an entry's normalized rank; replay never calls it. The raw
    /// (source-permuted, unnormalized) shapes and strides are not available:
    /// a completed transform retains only the geometry replay executes.
    ///
    /// The raw geometry is the destination and source `BlockStructure`
    /// blocks with the spec's `source_axes` applied to the source;
    /// [`Self::layouts`] gives each entry's offset and element count.
    pub fn layout_diagnostics(&self) -> TreeTransformLayoutDiagnostics {
        self.core.layouts.diagnostics(&self.core.blocks)
    }

    /// Block `block_index`'s `U[dst, src]` coefficients, read in place: a
    /// Single block's scalar, or the Multi block's shared matrix.
    #[inline]
    pub fn block_coefficients(&self, block_index: usize) -> Option<&[T]> {
        match *self.core.blocks.get(block_index)? {
            TreeTransformBlock::Single { coefficient, .. } => self
                .core
                .coefficients
                .singles
                .get(coefficient..=coefficient),
            TreeTransformBlock::Multi { .. } => self.block_matrix(block_index),
        }
    }

    #[inline]
    pub(crate) fn block_matrix(&self, block_index: usize) -> Option<&[T]> {
        match *self.core.blocks.get(block_index)? {
            TreeTransformBlock::Single { .. } => None,
            TreeTransformBlock::Multi { matrix, .. } => self.core.coefficients.matrix(matrix),
        }
    }

    /// Scalars of the Single blocks, indexed by their `coefficient`.
    #[inline]
    pub(crate) fn single_coefficients(&self) -> &[T] {
        &self.core.coefficients.singles
    }

    /// Length of the logical coefficient payload the block offsets index.
    #[inline]
    pub fn coefficient_len(&self) -> usize {
        self.core.coefficients.len
    }

    /// Copies the logical destination-by-source coefficient payload into
    /// `out`, replacing its contents.
    ///
    /// This is an O([`Self::coefficient_len`]) copy; replay reads
    /// [`Self::block_coefficients`] in place instead.
    pub fn gather_recoupling_coefficients_into(&self, out: &mut Vec<T>)
    where
        T: Copy,
    {
        self.gather_converted_coefficients_into(out, |coefficient| coefficient);
    }

    /// The logical payload, converted: the scalars, then the Multi matrices in
    /// `coefficient_start` (= spec) order. The CUDA executor calls this once
    /// per device structure upload.
    pub(crate) fn gather_converted_coefficients_into<U>(
        &self,
        out: &mut Vec<U>,
        mut convert: impl FnMut(T) -> U,
    ) where
        T: Copy,
    {
        let coefficients = &self.core.coefficients;
        out.clear();
        out.reserve(coefficients.len);
        out.extend(coefficients.singles.iter().map(|&value| convert(value)));
        for matrix in &coefficients.matrices {
            out.extend(matrix.iter().map(|&value| convert(value)));
        }
    }

    pub fn workspace_lens(&self) -> (usize, usize) {
        self.core
            .blocks
            .iter()
            .fold((0, 0), |(max_src, max_dst), block| match block {
                TreeTransformBlock::Single { .. } => (max_src, max_dst),
                TreeTransformBlock::Multi {
                    dst_count,
                    src_count,
                    element_count,
                    ..
                } => (
                    max_src.max(element_count.saturating_mul(*src_count)),
                    max_dst.max(element_count.saturating_mul(*dst_count)),
                ),
            })
    }

    pub fn workspace_len(&self) -> usize {
        let (source, destination) = self.workspace_lens();
        source.max(destination)
    }

    pub fn has_pack_gemm_scatter_blocks(&self) -> bool {
        !self.core.recoupling_plan.is_empty()
    }

    /// The core's identity: shared by every handle bound to one core.
    #[inline]
    pub(crate) fn identity_marker(&self) -> &Arc<()> {
        &self.core.identity
    }

    /// The coefficient payload's identity: shared by every core bound to one
    /// payload.
    #[inline]
    #[cfg(feature = "cuda")]
    pub(crate) fn coefficient_identity_marker(&self) -> &Arc<()> {
        self.core.coefficients.identity()
    }

    /// Whether `self` and `other` share one coefficient payload allocation.
    #[doc(hidden)]
    pub fn shares_coefficients_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.core.coefficients, &other.core.coefficients)
    }

    #[cfg(test)]
    pub(crate) fn gathered_coefficients(&self) -> Vec<T>
    where
        T: Copy,
    {
        let mut coefficients = Vec::new();
        self.gather_recoupling_coefficients_into(&mut coefficients);
        coefficients
    }

    #[cfg(test)]
    pub(crate) fn shares_recoupling_matrices_with(&self, other: &Self) -> bool {
        !self.core.coefficients.matrices.is_empty()
            && Arc::ptr_eq(&self.core.coefficients, &other.core.coefficients)
    }

    #[inline]
    pub fn recoupling_plan(&self) -> &TreeTransformRecouplingPlan {
        &self.core.recoupling_plan
    }

    /// Test/diagnostic helper: per-block replay weights.
    #[doc(hidden)]
    pub fn replay_weights(&self) -> Vec<usize> {
        self.core
            .blocks
            .iter()
            .map(|block| tree_transform_block_weight(block, &self.core.layouts.entries))
            .collect()
    }

    #[inline]
    pub fn storage_conjugate(&self) -> bool {
        self.core.storage_conjugate
    }

    /// Scalar of a [`TreeTransformBlock::Single`] block: `index` is that
    /// block's `coefficient`. `None` when `index` is not a Single scalar
    /// (Multi matrices are read with [`Self::block_coefficients`]).
    #[inline]
    pub fn single_coefficient(&self, index: usize) -> Option<T> {
        self.core.coefficients.singles.get(index).copied()
    }

    #[inline]
    pub(crate) fn inactive_destination_layouts(&self) -> &[usize] {
        &self.core.inactive_dst_layouts
    }

    #[inline]
    pub(crate) fn physical_overwrite_len(&self) -> Option<usize> {
        self.core.physical_overwrite_len
    }

    #[inline]
    pub(crate) fn parallel_schedule(&self) -> &TreeTransformParallelSchedule {
        &self.core.parallel_schedule
    }

    pub fn validate_replay_structures(
        &self,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
    ) -> Result<(), OperationError> {
        validate_structure_identity("dst", &self.dst_structure, dst_structure)?;
        validate_structure_identity("src", &self.src_structure, src_structure)
    }

    pub(crate) fn replay_storage_lens(&self) -> Result<(usize, usize), OperationError> {
        Ok((
            self.src_structure.required_len()?,
            self.dst_structure.required_len()?,
        ))
    }

    /// Rebinds an already compiled descriptor to content-equal canonical
    /// structures without rebuilding its replay payload.
    #[doc(hidden)]
    pub fn with_canonical_structures(
        mut self,
        dst_structure: Arc<BlockStructure>,
        src_structure: Arc<BlockStructure>,
    ) -> Result<Self, OperationError> {
        validate_structure_identity("dst", &self.dst_structure, &dst_structure)?;
        validate_structure_identity("src", &self.src_structure, &src_structure)?;
        self.dst_structure = dst_structure;
        self.src_structure = src_structure;
        Ok(self)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum TreeTransformBlock {
    Single {
        dst_layout: usize,
        src_layout: usize,
        coefficient: usize,
    },
    Multi {
        dst_layout_start: usize,
        dst_count: usize,
        src_layout_start: usize,
        src_count: usize,
        coefficient_start: usize,
        element_count: usize,
        /// Ordinal of this block's matrix among the plan's Multi matrices
        /// (the number of Multi blocks before it).
        matrix: u32,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use symtn_core::{BlockKey, BlockSpec};

    /// Independent hand normalization of a two-sided role: drop extent-1
    /// axes, order by destination stride, fuse runs contiguous on both sides.
    /// Written here, not shared with the production normalizer.
    fn expected_role(
        shape: &[usize],
        dst: &[isize],
        src: &[isize],
    ) -> (Vec<usize>, Vec<isize>, Vec<isize>) {
        if shape.contains(&0) {
            return (vec![0], vec![0], vec![0]);
        }
        let mut axes = (0..shape.len())
            .filter(|&axis| shape[axis] != 1)
            .map(|axis| (dst[axis], src[axis], shape[axis]))
            .collect::<Vec<_>>();
        axes.sort_by_key(|&(dst, _, _)| dst);
        let mut fused: Vec<(isize, isize, usize)> = Vec::new();
        for (dst, src, extent) in axes {
            match fused.last_mut() {
                Some(last)
                    if last.0 * last.2 as isize == dst && last.1 * last.2 as isize == src =>
                {
                    last.2 *= extent
                }
                _ => fused.push((dst, src, extent)),
            }
        }
        if fused.is_empty() {
            return (vec![1], vec![0], vec![0]);
        }
        (
            fused.iter().map(|axis| axis.2).collect(),
            fused.iter().map(|axis| axis.0).collect(),
            fused.iter().map(|axis| axis.1).collect(),
        )
    }

    #[test]
    fn compiled_single_role_is_the_normalized_permuted_pair() {
        // What: a runtime-rank Single block retains one normalized pair — the
        // destination block's strides against the source block's strides
        // after `source_axes` — and its source entry retains no geometry.
        let shape = vec![2; 9];
        let strides = (0..9).map(|axis| 1usize << axis).collect::<Vec<_>>();
        let block = BlockSpec::with_key(BlockKey::ordinal(0), shape, strides, 0).unwrap();
        let structure = BlockStructure::from_blocks_with_rank(9, vec![block]).unwrap();
        let specs =
            vec![TreeTransformBlockSpec::single(0, 0, 1.0_f64).with_source_axes((0..9).rev())];
        let compiled =
            TreeTransformStructure::compile_structures(&structure, &structure, &specs).unwrap();
        assert!(!compiled.has_pack_gemm_scatter_blocks());
        assert_eq!(compiled.layouts().max_fused_rank(), 9);

        let dst = (0..9).map(|axis| 1isize << axis).collect::<Vec<_>>();
        let src = (0..9).rev().map(|axis| 1isize << axis).collect::<Vec<_>>();
        let (dims, dst, src) = expected_role(&[2; 9], &dst, &src);
        let role = compiled.layouts().role(0).unwrap();
        assert_eq!(
            (role.dims(), role.dst_strides(), role.src_strides()),
            (&dims[..], &dst[..], &src[..])
        );
        assert_eq!(
            compiled.layouts().role(1).unwrap_err(),
            missing_role_error()
        );

        let diagnostics = compiled.layout_diagnostics();
        assert_eq!(diagnostics.len(), 2);
        let destination = diagnostics.entry(0).unwrap();
        assert_eq!(
            destination.role,
            TreeTransformLayoutRole::Single { source_entry: 1 }
        );
        assert_eq!(
            (
                destination.dims,
                destination.block_strides,
                destination.partner_strides
            ),
            (&dims[..], &dst[..], Some(&src[..]))
        );
        let source = diagnostics.entry(1).unwrap();
        assert_eq!(
            source.role,
            TreeTransformLayoutRole::SingleSource {
                destination_entry: 0
            }
        );
        assert_eq!(
            (source.dims, source.block_strides, source.partner_strides),
            (&dims[..], &src[..], Some(&dst[..]))
        );
        assert_eq!((source.offset, source.element_count), (0, 512));
    }

    fn missing_role_error() -> OperationError {
        OperationError::InvalidArgument {
            message: "tree transform layout entry has no compiled role of this kind",
        }
    }

    #[test]
    fn retained_layout_arena_is_exact_and_charged_by_capacity() {
        // What: every retained layout vector is sized exactly — the arena to
        // the normalized ranks, not `rank` per role, and the Single schedule
        // to its entries — and the admission charge follows capacity.
        let blocks = vec![
            BlockSpec::with_key(BlockKey::ordinal(0), vec![2, 2], vec![1, 2], 0).unwrap(),
            BlockSpec::with_key(BlockKey::ordinal(1), vec![2, 2], vec![1, 2], 4).unwrap(),
            BlockSpec::with_key(BlockKey::ordinal(2), vec![1, 4], vec![1, 1], 8).unwrap(),
        ];
        let structure = BlockStructure::from_blocks_with_rank(2, blocks).unwrap();
        let mut compiled = TreeTransformStructure::compile_structures(
            &structure,
            &structure,
            &[
                // Contiguous on both sides: one fused axis of 4.
                TreeTransformBlockSpec::single(0, 1, 1.0_f64),
                // Transposed: two axes.
                TreeTransformBlockSpec::single(1, 0, 1.0_f64).with_source_axes([1, 0]),
            ],
        )
        .unwrap();
        let layouts = compiled.layouts();
        // Two-sided roles: 1 + 2 axes. Inactive block 2 (`[1, 4]`): 1 axis.
        assert_eq!(layouts.fused_src_strides.capacity(), 3);
        assert_eq!(layouts.fused_dims.capacity(), 4);
        assert_eq!(layouts.fused_dst_strides.capacity(), 4);
        assert_eq!(layouts.entries.capacity(), 5);
        assert_eq!(compiled.parallel_schedule().singles.capacity(), 2);
        assert_eq!(
            compiled.layouts().inactive_role(4).unwrap(),
            (&[4usize][..], &[1isize][..])
        );
        assert!(compiled.layouts().role(4).is_err());
        assert!(compiled.layouts().inactive_role(0).is_err());

        let charged = compiled.charged_payload_bytes();
        Arc::get_mut(&mut compiled.core)
            .unwrap()
            .layouts
            .fused_dims
            .reserve_exact(128);
        assert!(compiled.charged_payload_bytes() > charged);
    }

    #[test]
    fn layout_normalization_overflow_returns_a_typed_compile_error() {
        // What: legal layout metadata whose signed span cannot be represented
        // is rejected as an overflow instead of panicking during normalization.
        let max = isize::MAX as usize;
        let block =
            BlockSpec::with_key(BlockKey::ordinal(0), vec![2, 2], vec![max - 1, max], 0).unwrap();
        let structure = BlockStructure::from_blocks_with_rank(2, vec![block]).unwrap();
        let error = TreeTransformStructure::compile_structures(
            &structure,
            &structure,
            &[TreeTransformBlockSpec::single(0, 0, 1.0_f64)],
        )
        .unwrap_err();

        assert!(matches!(error, OperationError::ElementCountOverflow));
    }

    #[test]
    fn layout_range_covers_negative_stride_rank_zero_and_zero_extent() {
        let entry = |offset, element_count| TreeTransformLayout {
            role: RoleSlot::default(),
            offset,
            element_count,
        };
        let mut layouts = RawLayoutTable::new(1);
        layouts.table.entries = vec![entry(5, 3), entry(0, 0)];
        layouts.shapes = vec![3, 0];
        layouts.strides = vec![-2, 1];
        let mut scalar = RawLayoutTable::new(0);
        scalar.table.entries = vec![entry(7, 1)];

        assert_eq!(layout_index_range(&layouts, 0).unwrap(), Some((1, 5)));
        assert_eq!(layout_index_range(&scalar, 0).unwrap(), Some((7, 7)));
        assert_eq!(layout_index_range(&layouts, 1).unwrap(), None);
    }

    #[test]
    fn single_only_schedule_allocates_no_scatter_group_metadata() {
        let structure = BlockStructure::packed_column_major(1, [vec![2]]).unwrap();
        let transform = TreeTransformStructure::compile_structures(
            &structure,
            &structure,
            &[TreeTransformBlockSpec::single(0, 0, 1.0)],
        )
        .unwrap();
        let schedule = transform.parallel_schedule();

        // What: an empty recoupling plan owns no Multi replay metadata.
        assert!(schedule.pack_columns.is_empty());
        assert!(schedule.scatter_columns.is_empty());
        assert!(schedule.scatter_groups.is_empty());
        assert_eq!(schedule.scatter_groups.capacity(), 0);
    }

    #[test]
    fn compiled_parallel_schedule_is_stable_for_equivalent_structures() {
        let block = |sector, offset| {
            BlockSpec::with_key(BlockKey::ordinal(sector), vec![2], vec![1], offset).unwrap()
        };
        let dst = BlockStructure::from_blocks_with_rank(1, vec![block(0, 0), block(1, 2)]).unwrap();
        let src = BlockStructure::packed_column_major(1, [vec![2], vec![2]]).unwrap();
        let specs = [TreeTransformBlockSpec::multi(
            vec![0, 1],
            vec![0, 1],
            vec![1.0, 0.0, 0.0, 1.0],
        )];

        let first = TreeTransformStructure::compile_structures(&dst, &src, &specs).unwrap();
        let second = TreeTransformStructure::compile_structures(&dst, &src, &specs).unwrap();

        assert_eq!(first.parallel_schedule(), second.parallel_schedule());
        assert_eq!(first.parallel_schedule().pack_columns.len(), 2);
        assert_eq!(first.parallel_schedule().scatter_columns.len(), 2);
        assert_eq!(first.blocks(), second.blocks());
        assert_eq!(first.layouts(), second.layouts());
        assert_eq!(
            first.gathered_coefficients(),
            second.gathered_coefficients()
        );
    }

    #[test]
    fn mapped_layout_append_preserves_axis_order_and_bakes_column_major_pack() {
        let mut layouts = RawLayoutTable::new(3);
        layouts.push_block(&[4, 2, 3], &[1, 4, 8], 0).unwrap();
        let count = layouts
            .push_block_with_axes(&[2, 3, 4], &[1, 2, 6], 7, Some(&[2, 0, 1]))
            .unwrap();

        // What: source-axis permutation changes the raw shape and source
        // strides in the requested order.
        assert_eq!(count, 24);
        assert_eq!(layouts.shape(1), &[4, 2, 3]);
        assert_eq!(layouts.strides(1), &[6, 1, 2]);
        assert_eq!(layouts.entry(1).offset, 7);

        // What: the pack role pairs those strides with the column-major
        // strides `[1, 4, 8]` of the same permuted shape. By hand: ordered by
        // packed stride, axes 1 and 2 are contiguous on both sides
        // (4·2 = 8, 1·2 = 2), axes 0 and 1 are not (6·4 ≠ 1).
        let table = layouts
            .finish(
                &[TreeTransformBlock::Multi {
                    dst_layout_start: 0,
                    dst_count: 1,
                    src_layout_start: 1,
                    src_count: 1,
                    coefficient_start: 0,
                    element_count: 24,
                    matrix: 0,
                }],
                &[],
            )
            .unwrap();
        let pack = table.role(1).unwrap();
        assert_eq!(pack.dims(), &[4, 6]);
        assert_eq!(pack.dst_strides(), &[1, 4]);
        assert_eq!(pack.src_strides(), &[6, 1]);
        // The scatter of the unpermuted destination fuses completely.
        let scatter = table.role(0).unwrap();
        assert_eq!(
            (scatter.dims(), scatter.dst_strides(), scatter.src_strides()),
            (&[24usize][..], &[1isize][..], &[1isize][..])
        );
    }

    #[test]
    fn mapped_layout_validation_is_atomic_and_keeps_permuted_zero_extent_order() {
        let mut layouts = RawLayoutTable::new(3);

        let error = layouts
            .push_block_with_axes(&[2, 3, 4], &[1, 2, 6], 0, Some(&[0, 0, 2]))
            .unwrap_err();
        assert_eq!(
            error,
            OperationError::InvalidPermutation {
                axes: vec![0, 0, 2],
                rank: 3,
            }
        );
        assert_eq!(layouts.entry_count(), 0);
        assert!(layouts.shapes.is_empty() && layouts.strides.is_empty());

        let count = layouts
            .push_block_with_axes(&[usize::MAX, 2, 0], &[1, 1, 1], 0, Some(&[2, 0, 1]))
            .unwrap();
        // What: validation follows the materialized permutation's order, so a
        // leading zero extent keeps the same non-overflowing element count.
        assert_eq!(count, 0);
        assert_eq!(layouts.shape(0), &[0, usize::MAX, 2]);
    }

    #[test]
    fn high_rank_axis_validation_accepts_reverse_and_rejects_late_duplicate() {
        let rank = 257;
        let reverse = (0..rank).rev().collect::<Vec<_>>();
        let mut duplicate = reverse.clone();
        duplicate[rank - 1] = duplicate[rank - 2];

        // What: validation remains correct when rank exceeds inline metadata
        // capacity, including a duplicate discovered only at the final axis.
        assert_eq!(crate::axis::validate_permutation(&reverse, rank), Ok(()));
        assert_eq!(
            crate::axis::validate_permutation(&duplicate, rank),
            Err(OperationError::InvalidPermutation {
                axes: duplicate,
                rank,
            })
        );
    }

    #[test]
    fn physical_overwrite_coverage_includes_active_and_inactive_layouts() {
        // What: the compiled overwrite proof covers each destination byte once,
        // including blocks with no numerical source contribution.
        let dst = BlockStructure::packed_column_major(1, [vec![2], vec![2]]).unwrap();
        let src = BlockStructure::packed_column_major(1, [vec![2]]).unwrap();
        let transform = TreeTransformStructure::compile_structures(
            &dst,
            &src,
            &[TreeTransformBlockSpec::single(0, 0, 1.0)],
        )
        .unwrap();

        assert_eq!(transform.physical_overwrite_len(), Some(4));
    }

    #[test]
    fn physical_overwrite_coverage_rejects_holes() {
        // What: storage padding that belongs to no destination block cannot be
        // exposed as initialized owned tensor data.
        let block = |sector, offset| {
            BlockSpec::with_key(BlockKey::ordinal(sector), vec![2], vec![1], offset).unwrap()
        };
        let dst = BlockStructure::from_blocks_with_rank(1, vec![block(0, 0), block(1, 3)]).unwrap();
        let src = BlockStructure::packed_column_major(1, [vec![2], vec![2]]).unwrap();
        let transform = TreeTransformStructure::compile_structures(
            &dst,
            &src,
            &[
                TreeTransformBlockSpec::single(0, 0, 1.0),
                TreeTransformBlockSpec::single(1, 1, 1.0),
            ],
        )
        .unwrap();

        assert_eq!(transform.physical_overwrite_len(), None);
    }

    #[test]
    fn physical_overwrite_coverage_handles_rank_zero_and_zero_extent() {
        // What: scalar blocks each own one physical slot, while an empty
        // destination proves the empty range without manufacturing a write.
        let scalar =
            BlockStructure::packed_column_major(0, [Vec::<usize>::new(), Vec::<usize>::new()])
                .unwrap();
        let scalar_transform = TreeTransformStructure::compile_structures(
            &scalar,
            &scalar,
            &[
                TreeTransformBlockSpec::single(0, 0, 1.0),
                TreeTransformBlockSpec::single(1, 1, 1.0),
            ],
        )
        .unwrap();
        assert_eq!(scalar_transform.physical_overwrite_len(), Some(2));

        let empty = BlockStructure::packed_column_major(1, [vec![0]]).unwrap();
        let empty_transform = TreeTransformStructure::compile_structures(
            &empty,
            &empty,
            &[TreeTransformBlockSpec::single(0, 0, 1.0)],
        )
        .unwrap();
        assert_eq!(empty_transform.physical_overwrite_len(), Some(0));
    }
}
