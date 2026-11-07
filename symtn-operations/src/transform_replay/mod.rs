use core::mem::MaybeUninit;
use core::ops::{Add, Mul};
use std::sync::{Arc, Weak};

use crate::host_pool::Exec;
use symtn_dense::Engine;

use num_traits::{One, Zero};
#[cfg(any(test, feature = "testing"))]
use symtn_core::HostWritableStorage;
use symtn_core::{BlockStructure, Placement};
#[cfg(test)]
use symtn_dense::DefaultDenseExecutor;
use symtn_dense::{strided_batch_runs_into, DenseExecutor, DenseGemmBatchJob};

use crate::host_scratch::HostScratchBuffer;
use crate::kernel_adapter::movement::{span, with_payload_map, with_transform_map, Uninit, Write};
use crate::kernel_adapter::sealed::LeafGate;
use crate::kernel_adapter::{admitted_plan, planned_write_uninit};
use crate::owned_overwrite_buffer::initialize_owned;
use crate::storage_scratch::TreeTransformScratchBuffers;
use crate::strided::offset_to_isize;
use crate::task_view::TreeTransformTaskView;
use crate::transform_structure::TreeTransformParallelSchedule;
use crate::{
    ConjugateValue, DenseRecouplingScalar, HostKernelAdapter, OperationError,
    RecouplingCoefficientAction, ReportsPlacement, TransformScale, TreeTransformBlock,
    TreeTransformLayoutTable, TreeTransformReplayProfile, TreeTransformStructure,
};

mod batched;
#[cfg(feature = "tenferro")]
mod chunk;
mod coefficients;
mod entry;
mod kernels;
mod member;
mod owned;
#[cfg(test)]
mod tests;
mod writers;

use batched::*;
use coefficients::*;
pub use entry::*;
pub(crate) use kernels::*;
pub use member::*;
pub use owned::*;
use writers::{InitializedWriter, OwnedWriter, ReplayParts};

impl<D: Zero> DestinationMode<D> {
    /// Whether the write ignores the destination: only such a write may run
    /// on a compiled plan.
    #[inline]
    fn overwrites(&self) -> bool {
        match self {
            Self::Overwrite => true,
            Self::Axpby(beta) => beta.is_zero(),
        }
    }
}

#[derive(Clone, Copy)]
enum DestinationMode<D> {
    Axpby(D),
    // Why not use Axpby(D::zero()): IEEE arithmetic still reads NaN destination
    // values, whereas assignment APIs promise destination-independent output.
    Overwrite,
}

/// The caller scale of the move or scatter writing each destination block:
/// `alpha`, times `θ_b` for the destination block at offset `b` when
/// `scales` lists `(b, θ_b)` (strictly increasing offsets). The fermionic
/// contraction twist rides here, so the transform that materializes the
/// twisted operand writes it already scaled — the same fold as the device's
/// `replay_with_destination_scales`.
#[derive(Clone, Copy)]
struct DestinationAlpha<'a, D, C> {
    alpha: D,
    scales: &'a [(usize, C)],
}

impl<'a, D, C> DestinationAlpha<'a, D, C>
where
    D: Copy + RecouplingCoefficientAction<C>,
    C: Copy,
{
    fn uniform(alpha: D) -> Self {
        Self { alpha, scales: &[] }
    }

    fn new(alpha: D, scales: &'a [(usize, C)]) -> Result<Self, OperationError> {
        if scales.windows(2).any(|pair| pair[0].0 >= pair[1].0) {
            return Err(OperationError::InvalidArgument {
                message: "tree transform destination scales must have strictly increasing offsets",
            });
        }
        Ok(Self { alpha, scales })
    }

    #[inline]
    fn at(self, offset: isize) -> D {
        if self.scales.is_empty() {
            return self.alpha;
        }
        usize::try_from(offset)
            .ok()
            .and_then(|offset| {
                self.scales
                    .binary_search_by_key(&offset, |&(block, _)| block)
                    .ok()
            })
            .map_or(self.alpha, |index| {
                self.alpha.scale_by_coefficient(self.scales[index].1)
            })
    }
}

struct PhysicalOverwriteProof {
    required_len: usize,
}

impl PhysicalOverwriteProof {
    fn new<C: Copy>(
        structure: &TreeTransformStructure<C>,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        src_len: usize,
        nout: usize,
    ) -> Result<Option<Self>, OperationError> {
        structure.validate_replay_structures(dst_structure, src_structure)?;
        validate_replay_storage_len(src_structure, src_len)?;
        let required_len = dst_structure.required_len()?;
        if structure.physical_overwrite_len() != Some(required_len) || nout > dst_structure.rank() {
            return Ok(None);
        }
        let Some(regions) = dst_structure.coupled_sector_regions(nout)? else {
            return Ok(None);
        };
        let mut next = 0usize;
        for region in regions.iter() {
            let range = region.range();
            if range.start != next || range.end > required_len {
                return Ok(None);
            }
            next = range.end;
        }
        if next != required_len {
            return Ok(None);
        }
        Ok(Some(Self { required_len }))
    }
}

/// Host scratch/replay workspace backed by `Vec<T>`.
///
/// Raw replay methods using this workspace operate on host slices. Device
/// execution should use a separate device workspace instead of hiding device
/// storage behind this type.
#[derive(Clone, Debug)]
pub struct HostTreeTransformWorkspace<T> {
    zero_strides: Vec<isize>,
    packed: TreeTransformScratchBuffers<HostScratchBuffer<T>, HostScratchBuffer<T>>,
    // Recoupling matrices converted into the data scalar type and packed in
    // recoupling_plan().entries() execution order. This order is layout
    // dependent and supplies the GEMM RHS offsets; it is not canonical
    // categorical U storage (TensorKit's basistransform buffer).
    coefficient_scratch: Vec<T>,
    // Identity of the structure whose layout-ordered RHS pack is installed.
    coefficient_structure_identity: Option<Weak<()>>,
    /// Coefficient packs converted into this workspace so far.
    #[cfg(any(test, feature = "testing"))]
    coefficient_pack_builds: usize,
    chunk_jobs: Vec<DenseGemmBatchJob>,
    chunk_runs: Vec<usize>,
    // One chunk's non-empty scatter groups, in destination order.
    chunk_scatter_groups: Vec<usize>,
    // The fused traversal indices of the replay's lanes ([`fused_index_lanes`]):
    // inline up to eight, the rank metadata width, so a fresh per-call
    // workspace of low rank allocates none.
    fused_indices: smallvec::SmallVec<[usize; INLINE_FUSED_RANK]>,
    member_shape: Vec<usize>,
    member_dst_strides: Vec<isize>,
    member_src_strides: Vec<isize>,
    member_ranges: Vec<(usize, usize)>,
}

pub type TreeTransformWorkspace<T> = HostTreeTransformWorkspace<T>;

impl<T> Default for HostTreeTransformWorkspace<T> {
    fn default() -> Self {
        Self {
            zero_strides: Vec::new(),
            packed: TreeTransformScratchBuffers::default(),
            coefficient_scratch: Vec::new(),
            coefficient_structure_identity: None,
            #[cfg(any(test, feature = "testing"))]
            coefficient_pack_builds: 0,
            chunk_jobs: Vec::new(),
            chunk_runs: Vec::new(),
            chunk_scatter_groups: Vec::new(),
            fused_indices: smallvec::SmallVec::new(),
            member_shape: Vec::new(),
            member_dst_strides: Vec::new(),
            member_src_strides: Vec::new(),
            member_ranges: Vec::new(),
        }
    }
}

impl<T> HostTreeTransformWorkspace<T> {
    /// Host capacity retained by member replay and ordinary transform
    /// scratch.
    #[doc(hidden)]
    pub fn retained_bytes(&self) -> usize {
        let bytes = |capacity: usize, item: usize| capacity.saturating_mul(item);
        bytes(
            self.packed.source().capacity() + self.packed.destination().capacity(),
            std::mem::size_of::<T>(),
        ) + bytes(
            self.coefficient_scratch.capacity(),
            std::mem::size_of::<T>(),
        ) + bytes(
            self.chunk_jobs.capacity(),
            std::mem::size_of::<DenseGemmBatchJob>(),
        ) + bytes(
            self.chunk_runs.capacity()
                + self.chunk_scatter_groups.capacity()
                + self.fused_indices_heap_capacity()
                + self.member_shape.capacity(),
            std::mem::size_of::<usize>(),
        ) + bytes(
            self.zero_strides.capacity()
                + self.member_dst_strides.capacity()
                + self.member_src_strides.capacity(),
            std::mem::size_of::<isize>(),
        ) + bytes(
            self.member_ranges.capacity(),
            std::mem::size_of::<(usize, usize)>(),
        )
    }

    /// Coefficient packs this workspace has converted: a warm replay of the
    /// structure whose pack is installed converts none.
    #[doc(hidden)]
    #[cfg(any(test, feature = "testing"))]
    pub fn coefficient_pack_builds(&self) -> usize {
        self.coefficient_pack_builds
    }

    #[inline]
    pub fn placement(&self) -> Placement {
        Placement::Host
    }

    #[inline]
    pub fn is_host_workspace(&self) -> bool {
        self.placement() == Placement::Host
    }

    pub fn source_len(&self) -> usize {
        self.packed.source().len()
    }

    pub fn destination_len(&self) -> usize {
        self.packed.destination().len()
    }

    fn prepare_packed_buffers(&mut self, source_len: usize, destination_len: usize, zero: T)
    where
        T: Clone,
    {
        prepare_packed_buffer(self.packed.source_mut(), source_len, zero.clone());
        prepare_packed_buffer(self.packed.destination_mut(), destination_len, zero);
    }

    /// Sizes the packed destination alone, for a replay whose source pack
    /// overwrites its scratch without a fill.
    fn prepare_packed_destination(&mut self, len: usize, zero: T)
    where
        T: Clone,
    {
        prepare_packed_buffer(self.packed.destination_mut(), len, zero);
    }

    /// Heap elements `fused_indices` owns (none while inline).
    fn fused_indices_heap_capacity(&self) -> usize {
        if self.fused_indices.spilled() {
            self.fused_indices.capacity()
        } else {
            0
        }
    }

    fn prepare_fused_indices(
        &mut self,
        threads: usize,
        max_fused_rank: usize,
    ) -> Result<(), OperationError> {
        let len =
            checked_fused_index_len(fused_index_lanes(threads, max_fused_rank), max_fused_rank)?;
        if len > self.fused_indices.len() {
            self.fused_indices.resize(len, 0);
        }
        Ok(())
    }
}

/// Fused-index entries a lane keeps inline: the rank metadata width.
const INLINE_FUSED_RANK: usize = 8;

/// Lanes whose fused indices the shared workspace buffer holds: every lane
/// above the inline rank, only the calling lane otherwise.
///
/// Why not always one shared `threads * rank` buffer: an eager call's
/// workspace is per call, and that buffer outgrows its inline capacity as
/// soon as the engine is wider than one, so every parallel call allocated on
/// the caller thread what the serial call does not. Up to the inline rank a
/// forked lane keeps its index on its own stack instead ([`LaneIndex`]);
/// above it the index spills anyway, and one shared buffer keeps a warm
/// replay allocation-free.
fn fused_index_lanes(threads: usize, max_fused_rank: usize) -> usize {
    if max_fused_rank <= INLINE_FUSED_RANK {
        1
    } else {
        threads
    }
}

/// The fused-index scratch of one lane of a replay split: a slice of the
/// shared buffer when it holds every lane of the part, or the lane's own
/// inline index.
enum LaneIndex<'a> {
    Shared(&'a mut [usize]),
    Own(smallvec::SmallVec<[usize; INLINE_FUSED_RANK]>),
}

impl LaneIndex<'_> {
    fn get(&mut self) -> &mut [usize] {
        match self {
            Self::Shared(indices) => indices,
            Self::Own(indices) => indices,
        }
    }

    /// Splits one fork's fused-index scratch: the first `left_threads` lanes
    /// keep the front of `self`; the rest take the back when it holds every
    /// lane, and their own inline index otherwise.
    fn split_off(&mut self, left_threads: usize, threads: usize, max_fused_rank: usize) -> Self {
        match self {
            Self::Shared(indices) if indices.len() >= threads * max_fused_rank => {
                let (left, right) =
                    core::mem::take(indices).split_at_mut(left_threads * max_fused_rank);
                *indices = left;
                Self::Shared(right)
            }
            _ => Self::Own(smallvec::SmallVec::from_elem(0, max_fused_rank)),
        }
    }
}

/// Sizes one pack buffer to `len` zero-initialised elements.
///
/// Why not `resize_filled` when the buffer must grow past its capacity: that
/// reallocates, copies contents every reader overwrites, and then fills the
/// new tail element by element. A per-call workspace grows from empty on
/// every call, so that fill was one extra pass over the whole chunk scratch
/// before the pack overwrote it. A fresh `filled` allocation lets std zero
/// real scalars through the allocator (`alloc_zeroed`): fresh pages need no
/// pass, and a reused block gets one bulk zeroing instead of the element loop.
/// Complex scalars still take the element loop (std has no zeroed path for
/// them).
fn prepare_packed_buffer<T: Clone>(buffer: &mut HostScratchBuffer<T>, len: usize, zero: T) {
    if len > buffer.capacity() {
        *buffer = HostScratchBuffer::filled(len, zero);
    } else {
        buffer.resize_filled(len, zero);
    }
}

/// Refuses a forced `Inner`, `Outer` or `Split` batch policy on `engine`
/// when `dense` cannot honor it, before any write or executor call: the
/// replay refuses the forced decision rather than running another
/// placement. `Auto` and `Sequential` always run; so does every policy on
/// an executor that honors forced policies.
fn refuse_unhonored_policy<E: DenseExecutor>(
    engine: &mut dyn Engine,
    dense: &E,
) -> Result<(), OperationError> {
    #[cfg(feature = "tenferro")]
    if !dense.honors_forced_batch_policy() {
        use tenferro_cpu::CpuBatchStrategy;
        let strategy = crate::host_pool::batch_policy(engine).strategy;
        if matches!(
            strategy,
            CpuBatchStrategy::Inner | CpuBatchStrategy::Outer | CpuBatchStrategy::Split { .. }
        ) {
            return Err(OperationError::Dense(
                symtn_dense::DenseError::Unsupported {
                    op: "tree_transform_replay",
                    message: format!(
                        "the forced batch policy {strategy:?} needs an executor that honors it"
                    ),
                },
            ));
        }
    }
    #[cfg(not(feature = "tenferro"))]
    let _ = (engine, dense);
    Ok(())
}

// Why not reserve here: public drivers call this before beta mutation, while
// executors retain ownership of actual workspace growth and profiler attribution.
fn checked_fused_index_len(threads: usize, max_fused_rank: usize) -> Result<usize, OperationError> {
    let len = threads
        .max(1)
        .checked_mul(max_fused_rank)
        .ok_or(OperationError::ElementCountOverflow)?;
    core::alloc::Layout::array::<usize>(len).map_err(|_| OperationError::ElementCountOverflow)?;
    Ok(len)
}

impl<T> ReportsPlacement for HostTreeTransformWorkspace<T> {
    #[inline]
    fn placement(&self) -> Placement {
        Placement::Host
    }
}
