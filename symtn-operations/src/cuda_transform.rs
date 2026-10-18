//! Device replay of tree transforms on [`CudaStorage`].
//!
//! This is the second executor of the *same* `TreeTransformTaskView` the host
//! path consumes: everything above the completed `TreeTransformStructure` —
//! the operation descriptor, rule validation, the categorical group plan with
//! its F/R moves, the cache keyed by `RuleIdentity` — is reused unchanged, and
//! nothing categorical happens during replay. What is left below the structure
//! is exactly one scalar coefficient per block, a baked
//! `(dims, dst_strides, src_strides)` triple per block, a storage-conjugation
//! flag, and the destination mode.
//!
//! A Multi block is the one place where a block pair is not a single move: the
//! host packs each source layout into a compact column, multiplies the column
//! block by the transposed recoupling matrix, and scatters the result columns
//! back out. This executor submits exactly that sequence, in the compile-time
//! recoupling plan's own job order.
//!
//! Reference record for the replayed dataflow: TensorKit
//! `add_transform!` / `add_transform_kernel!`
//! (src/tensors/indexmanipulations.jl:547/:579/:648/:660 @cfaa073), whose
//! recoupling step is `mul!(dst, src, transpose(U))`
//! (indexmanipulations.jl:629/:701 @cfaa073), and QSpace `Permute`/`permute`
//! (Source/QSpace.hh:2837/:2890 @dd2cc7e). Host authority for the lowering is
//! `transform_replay/kernels.rs`: `pack_layout_into_column`, `recoupling_gemm_batch`,
//! `scatter_column_into_layout`.

use core::any::TypeId;
use std::sync::{Arc, Weak};

use symtn_core::{BlockStructure, Placement, TensorStorage};
pub use symtn_dense::DEFAULT_PLAN_CACHE_BUDGET_BYTES;
use symtn_dense::{
    cuda_copy_strided_into, cuda_matmul_region_into, cuda_region_axpby, cuda_region_scale,
    cuda_region_zero, plan_cache_entries_for, CudaDenseContext, CudaDenseStorage, CudaRegion,
    CudaRegionBeta, CudaRegionCoefficient, CudaScalar, DenseError,
};

use crate::cuda::CudaStorage;
use crate::cuda_transform_plan::{
    compile_device_plan, DeviceMoveSpec, DeviceTransformPlan, PlanSignatures, PreparedDeviceCache,
    SharedPayload, StructureKey, DEFAULT_STRUCTURE_CACHE_ENTRIES,
};
use crate::fusion_replay::{FusionScaleBlockLayout, FusionStridedBlockLayout};
use crate::opaque_admission::{
    validate_stage_a, validate_stage_c, AllocationIdentity, CoefficientReadiness, ContextIdentity,
    ExecutorSnapshot, StorageDomain, StorageRegion, StorageSnapshot, WorkspaceSnapshot,
};
use crate::task_view::TreeTransformTaskView;
use crate::{
    OperationError, RecouplingCoefficientAction, TreeTransformBlock, TreeTransformStructure,
};

/// A direct core's inactive destination blocks as device zero regions: the
/// one conversion from host block layouts that the eager, member and
/// prepared CUDA contractions share (#1776). A device region's strides and
/// offset are unsigned, so a negatively strided block is rejected.
#[doc(hidden)]
pub struct CudaMemberZeroRegions {
    regions: Vec<CudaRegion>,
    max_zero_len: usize,
}

impl CudaMemberZeroRegions {
    /// The blocks over a `members`-member stack: each region gains a trailing
    /// member axis of stride `member_len` and is validated within the stack
    /// and as a destination.
    pub fn prepare(
        layouts: &[FusionScaleBlockLayout],
        member_len: usize,
        members: usize,
    ) -> Result<Self, OperationError> {
        let total = member_len
            .checked_mul(members)
            .ok_or(OperationError::ElementCountOverflow)?;
        let mut regions = Vec::with_capacity(layouts.len());
        let mut max_zero_len = 0;
        for layout in layouts {
            let block = &layout.block;
            let (strides, offset) = unsigned_zero_layout(block)?;
            let mut dims = block.shape.clone();
            dims.push(members);
            let mut strides: Vec<usize> = strides.collect();
            strides.push(member_len);
            let region = CudaRegion::new(dims, strides, offset).map_err(OperationError::Dense)?;
            region
                .validate_within(total)
                .map_err(OperationError::Dense)?;
            region
                .validate_as_destination("CUDA member contraction zero fill")
                .map_err(OperationError::Dense)?;
            max_zero_len = max_zero_len.max(block.shape.iter().product::<usize>());
            regions.push(region);
        }
        Ok(Self {
            regions,
            max_zero_len,
        })
    }

    /// Checks, without building them, that every block has a device region:
    /// the admission a CUDA replay runs before any device work.
    pub fn admit(layouts: &[FusionScaleBlockLayout]) -> Result<(), OperationError> {
        for layout in layouts {
            let _ = unsigned_zero_layout(&layout.block)?;
        }
        Ok(())
    }

    /// The single-tensor (B = 1) case of [`Self::prepare`], without the
    /// member axis, written into `regions` in place. Why not `prepare`: an
    /// eager replay re-derives its regions every call, so it reuses one
    /// vector's allocations instead of building a new one.
    pub fn fill_single<'a>(
        regions: &'a mut Vec<CudaRegion>,
        layouts: &[FusionScaleBlockLayout],
    ) -> Result<&'a [CudaRegion], OperationError> {
        regions.truncate(layouts.len());
        for (index, layout) in layouts.iter().enumerate() {
            let block = &layout.block;
            let (strides, offset) = unsigned_zero_layout(block)?;
            match regions.get_mut(index) {
                Some(region) => region
                    .assign(&block.shape, strides, offset)
                    .map_err(OperationError::Dense)?,
                None => regions.push(
                    CudaRegion::new(block.shape.clone(), strides.collect(), offset)
                        .map_err(OperationError::Dense)?,
                ),
            }
        }
        Ok(regions)
    }

    pub fn regions(&self) -> &[CudaRegion] {
        &self.regions
    }

    pub fn into_regions(self) -> Vec<CudaRegion> {
        self.regions
    }

    /// Elements of the largest block, before the member axis.
    pub fn max_zero_len(&self) -> usize {
        self.max_zero_len
    }

    /// Sizes the dtype's zero template for the largest block over `members`
    /// members, and the ones template when any region exists, so the fills
    /// upload nothing.
    pub fn reserve_templates<D: CudaScalar>(
        &self,
        ctx: &mut CudaDenseContext,
        members: usize,
    ) -> Result<(), OperationError> {
        ctx.reserve_zero_template::<D>(
            self.max_zero_len
                .checked_mul(members)
                .ok_or(OperationError::ElementCountOverflow)?,
        )
        .map_err(OperationError::Dense)?;
        if !self.regions.is_empty() {
            ctx.reserve_ones_template::<D>(1)
                .map_err(OperationError::Dense)?;
        }
        Ok(())
    }
}

/// A block's strides and offset as a device region's unsigned ones.
fn unsigned_zero_layout(
    block: &FusionStridedBlockLayout,
) -> Result<(impl Iterator<Item = usize> + '_, usize), OperationError> {
    if block.offset < 0 || block.strides.iter().any(|&stride| stride < 0) {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "device contraction cannot zero a negatively strided destination block",
        });
    }
    // Checked non-negative just above, so the casts are exact.
    Ok((
        block.strides.iter().map(|&stride| stride as usize),
        block.offset as usize,
    ))
}

/// The validated B-appended views of a nonzero-Single CopyC output transform.
/// No tensor payload or backend handle is retained here.
#[doc(hidden)]
pub struct CudaSingleMemberRegions {
    moves: Vec<(CudaRegion, CudaRegion, f64, Option<usize>)>,
    zeros: Vec<CudaRegion>,
    max_zero_len: usize,
}

impl CudaSingleMemberRegions {
    /// Checks the entire completed task class before the temporary core runs.
    pub fn admit(structure: &TreeTransformStructure<f64>) -> Result<usize, OperationError> {
        Self::admit_scaled(structure, &[])
    }

    /// [`Self::admit`] with each move writing destination block offset `b`
    /// scaled by θ_b from `destination_scales` (strictly increasing offsets),
    /// the fermionic source twist folded as in
    /// [`CudaTreeTransformExecutor::replay_with_destination_scales`]. Returns
    /// the zero-fill and scaled-move plan entries a replay needs.
    pub fn admit_scaled(
        structure: &TreeTransformStructure<f64>,
        destination_scales: &[(usize, f64)],
    ) -> Result<usize, OperationError> {
        if destination_scales
            .windows(2)
            .any(|pair| pair[0].0 >= pair[1].0)
        {
            return Err(OperationError::InvalidArgument {
                message: "CUDA member transform scales must have strictly increasing offsets",
            });
        }
        let task = structure.task_view()?;
        if task.storage_conjugate()
            || task.blocks().iter().any(|block| match *block {
                TreeTransformBlock::Single { coefficient, .. } => task
                    .single_coefficients()
                    .get(coefficient)
                    .is_none_or(|&coefficient| coefficient == 0.0),
                TreeTransformBlock::Multi { .. } => true,
            })
        {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "CUDA member transform requires unconjugated nonzero Single tasks",
            });
        }
        let device = compile_device_plan(task)?;
        let mut scaled = 0;
        for entry in &device.moves {
            let coefficient = scaled_coefficient(task, entry, destination_scales)?;
            if coefficient == 0.0 {
                return Err(OperationError::UnsupportedTensorContractScope {
                    message: "CUDA member transform requires unconjugated nonzero Single tasks",
                });
            }
            scaled += usize::from(coefficient != 1.0);
        }
        Ok(device.zeros.len() + scaled)
    }

    /// Validates every source, destination and inactive output region for B members.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        structure: &TreeTransformStructure<f64>,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst_member_len: usize,
        src_member_len: usize,
        dst_len: usize,
        src_len: usize,
        members: usize,
    ) -> Result<Self, OperationError> {
        Self::prepare_scaled(
            structure,
            dst_structure,
            src_structure,
            dst_member_len,
            src_member_len,
            dst_len,
            src_len,
            members,
            &[],
        )
    }

    /// [`Self::prepare`] with the destination-block scales of [`Self::admit_scaled`].
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_scaled(
        structure: &TreeTransformStructure<f64>,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst_member_len: usize,
        src_member_len: usize,
        dst_len: usize,
        src_len: usize,
        members: usize,
        destination_scales: &[(usize, f64)],
    ) -> Result<Self, OperationError> {
        Self::admit_scaled(structure, destination_scales)?;
        let task = structure.task_view()?;
        task.validate_structures_and_lengths(
            dst_structure,
            src_structure,
            dst_member_len,
            src_member_len,
        )?;
        let device = compile_device_plan(task)?;
        let region = |dims: &[usize], strides: &[usize], offset: usize, member_stride| {
            let mut shape = dims.to_vec();
            shape.push(members);
            let mut steps = strides.to_vec();
            steps.push(member_stride);
            CudaRegion::new(shape, steps, offset).map_err(OperationError::Dense)
        };
        let mut moves = Vec::with_capacity(device.moves.len());
        let mut scaled_count = 0;
        for entry in &device.moves {
            // A full-stack bounds check alone could accept a block spilling
            // into the next member. Check the compiled member views first.
            let source_member = CudaRegion::new(
                entry.dims.clone(),
                entry.src_strides.clone(),
                entry.src_offset,
            )
            .map_err(OperationError::Dense)?;
            let destination_member = CudaRegion::new(
                entry.dims.clone(),
                entry.dst_strides.clone(),
                entry.dst_offset,
            )
            .map_err(OperationError::Dense)?;
            source_member
                .validate_within(src_member_len)
                .map_err(OperationError::Dense)?;
            destination_member
                .validate_within(dst_member_len)
                .map_err(OperationError::Dense)?;
            let source = region(
                &entry.dims,
                &entry.src_strides,
                entry.src_offset,
                src_member_len,
            )?;
            let destination = region(
                &entry.dims,
                &entry.dst_strides,
                entry.dst_offset,
                dst_member_len,
            )?;
            source
                .validate_within(src_len)
                .map_err(OperationError::Dense)?;
            destination
                .validate_within(dst_len)
                .map_err(OperationError::Dense)?;
            destination
                .validate_as_destination("cuda CopyC batch")
                .map_err(OperationError::Dense)?;
            let coefficient = scaled_coefficient(task, entry, destination_scales)?;
            let coefficient_offset = (coefficient != 1.0).then(|| {
                let offset = scaled_count;
                scaled_count += 1;
                offset
            });
            moves.push((source, destination, coefficient, coefficient_offset));
        }
        let mut zeros = Vec::with_capacity(device.zeros.len());
        for entry in &device.zeros {
            CudaRegion::new(entry.dims.clone(), entry.strides.clone(), entry.offset)
                .map_err(OperationError::Dense)?
                .validate_within(dst_member_len)
                .map_err(OperationError::Dense)?;
            let zero = region(&entry.dims, &entry.strides, entry.offset, dst_member_len)?;
            zero.validate_within(dst_len)
                .map_err(OperationError::Dense)?;
            zero.validate_as_destination("cuda CopyC batch zero")
                .map_err(OperationError::Dense)?;
            zeros.push(zero);
        }
        Ok(Self {
            moves,
            zeros,
            max_zero_len: device.max_zero_len,
        })
    }

    pub fn moves(&self) -> &[(CudaRegion, CudaRegion, f64, Option<usize>)] {
        &self.moves
    }
    pub fn zeros(&self) -> &[CudaRegion] {
        &self.zeros
    }
    pub fn max_zero_len(&self) -> usize {
        self.max_zero_len
    }
    pub fn has_scaled_moves(&self) -> bool {
        self.moves
            .iter()
            .any(|(_, _, _, coefficient)| coefficient.is_some())
    }
    pub fn scaled_coefficients<D: RecouplingCoefficientAction<f64>>(&self) -> Vec<D> {
        self.moves
            .iter()
            .filter_map(|(_, _, coefficient, offset)| {
                offset.map(|_| D::coefficient_as_data(*coefficient))
            })
            .collect()
    }

    /// Overwrites every active layout and zeros every inactive layout.
    pub fn execute_overwrite<D: CudaScalar + RecouplingCoefficientAction<f64>>(
        &self,
        ctx: &mut CudaDenseContext,
        source: &CudaStorage<D>,
        coefficients: Option<&CudaDenseStorage>,
        destination: &mut CudaStorage<D>,
    ) -> Result<(), OperationError> {
        for region in &self.zeros {
            cuda_region_zero::<D>(ctx, &mut destination.0, region)
                .map_err(OperationError::Dense)?;
        }
        for (src, dst, coefficient, coefficient_offset) in &self.moves {
            if *coefficient == 1.0 {
                cuda_copy_strided_into::<D>(ctx, &source.0, src, &mut destination.0, dst)
                    .map_err(OperationError::Dense)?;
            } else {
                let coefficients = coefficients.ok_or(OperationError::InvalidArgument {
                    message: "CUDA CopyC scaled move has no coefficient buffer",
                })?;
                cuda_region_axpby::<D>(
                    ctx,
                    &source.0,
                    src,
                    false,
                    D::ONE,
                    CudaRegionCoefficient::Buffer(
                        coefficients,
                        coefficient_offset.ok_or(OperationError::InvalidArgument {
                            message: "CUDA CopyC scaled move has no coefficient offset",
                        })?,
                    ),
                    CudaRegionBeta::Overwrite,
                    &mut destination.0,
                    dst,
                )
                .map_err(OperationError::Dense)?;
            }
        }
        Ok(())
    }
    pub fn retained_bytes(&self) -> usize {
        let inline = self
            .moves
            .capacity()
            .saturating_mul(std::mem::size_of::<(
                CudaRegion,
                CudaRegion,
                f64,
                Option<usize>,
            )>())
            .saturating_add(
                self.zeros
                    .capacity()
                    .saturating_mul(std::mem::size_of::<CudaRegion>()),
            );
        let metadata = self
            .moves
            .iter()
            .flat_map(|(a, b, _, _)| [a, b])
            .chain(self.zeros.iter());
        inline.saturating_add(
            metadata
                .map(CudaRegion::retained_heap_bytes)
                .fold(0usize, usize::saturating_add),
        )
    }
}

/// A Single move's coefficient times θ of the destination block it writes
/// (the lookup of [`destination_scaled`]).
fn scaled_coefficient(
    task: TreeTransformTaskView<'_, f64>,
    entry: &DeviceMoveSpec,
    destination_scales: &[(usize, f64)],
) -> Result<f64, OperationError> {
    let coefficient = entry
        .coefficient
        .and_then(|index| task.single_coefficients().get(index))
        .copied()
        .ok_or(OperationError::InvalidArgument {
            message: "CUDA CopyC Single move has no coefficient",
        })?;
    Ok(
        match destination_scales.binary_search_by_key(&entry.dst_offset, |&(offset, _)| offset) {
            Ok(index) => coefficient * destination_scales[index].1,
            Err(_) => coefficient,
        },
    )
}

/// How a device replay treats the destination it writes.
///
/// [`Self::Overwrite`] is the destination-independent mode: every inactive
/// destination layout is zeroed and every active one is assigned, so a
/// destination holding NaN comes back clean. [`Self::Axpby`] is
/// `dst = alpha * T(src) + beta * dst` for any `beta`, the mode the typed
/// `*_into` APIs run in; `Axpby(0)` is `Overwrite` — see
/// [`CudaTreeTransformExecutor::replay`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CudaTreeTransformDestination<D> {
    Overwrite,
    Axpby(D),
}

/// One block, pack column or scatter column as the region pair the primitive
/// takes, built when the structure is prepared so a replay allocates nothing on
/// the host either.
///
/// `coefficient` is the index of the 1x1 operand in the uploaded coefficient
/// vector, or `None` where the host copies unscaled and the context's shared
/// `1` is the operand. `zero_coefficient` marks a Single block whose
/// coefficient is zero: like the host (VectorInterface's `scale(x, 0) = 0`)
/// it leaves `beta * dst` (zeros under `Overwrite`), without reading the
/// source.
struct PreparedMove {
    source: CudaRegion,
    destination: CudaRegion,
    coefficient: Option<usize>,
    zero_coefficient: bool,
}

/// One Multi block's pack → `Uᵀ` GEMM → scatter sequence, with every offset
/// already resolved against the structure-wide device workspace.
struct PreparedRecoupling {
    packs: Vec<PreparedMove>,
    scatters: Vec<PreparedMove>,
    /// Workspace destination offset, workspace source offset, and the offset of
    /// this block's recoupling matrix inside the uploaded coefficient vector —
    /// the block's own `coefficient_start`, because the device holds the whole
    /// converted payload and needs no re-pack.
    dst_offset: usize,
    lhs_offset: usize,
    matrix_offset: usize,
    rows: usize,
    contracted: usize,
    cols: usize,
}

/// Device state prepared once per completed structure.
struct PreparedStructure {
    moves: Vec<PreparedMove>,
    recouplings: Vec<PreparedRecoupling>,
    /// Whether the recouplings address any workspace element at all. A Multi
    /// block of zero element count submits nothing, so no workspace is grown
    /// for it and none is looked up.
    needs_workspace: bool,
    zeros: Vec<CudaRegion>,
    max_zero_len: usize,
    /// The largest Single or scatter destination, the zero template a zero
    /// caller scale needs to write zeros over every written layout.
    max_write_len: usize,
    plan_signatures: PlanSignatures,
    /// Whether an Overwrite replay has zero-filled the written destinations on
    /// this entry's context, so `plan_signatures.zero_fill` is reserved too.
    /// Set by the first such replay, never cleared: the entry leaves the ledger
    /// whole when it is evicted or cleared.
    zero_fills_reserved: bool,
    /// Whether some Single block's coefficient is zero in the payload dtype,
    /// so every Overwrite replay zero-fills its destination, whatever the scale.
    zero_coefficient_moves: bool,
    /// The structure's whole coefficient payload converted to the payload dtype
    /// and uploaded as one device vector. Block `b`'s scalar is the 1x1 operand
    /// at its own index and Multi block `b`'s recoupling matrix is the
    /// `S_b x D_b` run at its own `coefficient_start`, both addressed exactly as
    /// the structure indexes them. `None` only for a structure with no
    /// coefficients at all.
    ///
    /// Shared by every entry of this dtype and context whose structure is bound
    /// to the same host payload (`coefficient_identity`): the vector is a
    /// function of that payload and the dtype alone, never of a layout, so
    /// structures differing only in degeneracies upload it once (#1753).
    coefficients: Option<Arc<CudaDenseStorage>>,
    /// The host payload `coefficients` was converted from. The `Weak` pins the
    /// payload's identity marker, so a live payload at the same address is
    /// this one, never a successor.
    coefficient_identity: Weak<()>,
    /// How many elements of the payload the recoupling matrices occupy, which
    /// is what Stage C requires of the converted-coefficient slot.
    recoupling_len: usize,
    /// The coefficient-readiness fact, recorded when the upload happened rather
    /// than rebuilt from the lookup key at replay time: Stage C then checks
    /// something this executor observed, not something it restated.
    readiness: CoefficientReadiness,
}

impl SharedPayload for PreparedStructure {
    fn shared_payload(&self) -> Option<*const ()> {
        self.coefficients
            .as_ref()
            .map(|payload| Arc::as_ptr(payload).cast())
    }
}

/// The device scratch a Multi replay packs into and scatters out of, held per
/// payload dtype and context and grown monotonically.
///
/// It is execution scratch, not a semantic cache: every column it holds is
/// fully written before it is read, so its contents never carry meaning across
/// replays and dropping it changes nothing but the cost of the next one. It is
/// two buffers rather than one because the GEMM borrows its source shared and
/// its destination exclusively.
struct DeviceWorkspace {
    scalar: TypeId,
    context: u64,
    source: CudaDenseStorage,
    destination: CudaDenseStorage,
    bytes: usize,
}

/// Placement-aware executor replaying tree transforms on device buffers.
///
/// It owns the per-structure device state that makes a warm replay free of
/// host↔device traffic: one uploaded coefficient-and-recoupling-matrix vector
/// per (structure, payload dtype, context), bounded by a byte budget and an
/// entry count and purged when the structure itself is dropped, plus the pack/
/// scatter workspace, grown monotonically per (payload dtype, context). Neither
/// is a semantic cache — the vector is a dtype conversion of data the structure
/// already owns and the workspace is fully rewritten before it is read — so
/// dropping them changes nothing but the cost of the next replay.
///
/// One executor belongs next to one device execution context. Entries of other
/// contexts can never be read back, because the context identity is part of
/// every key. Because a replay needs `&mut self` and `&mut CudaDenseContext`
/// at once, its owner is the runtime's device state beside the
/// `CudaDenseContext`, inside the same device mutex the device lease takes; it
/// is deliberately not global and not owned by any structure.
pub struct CudaTreeTransformExecutor {
    prepared: PreparedDeviceCache<PreparedStructure>,
    /// One transform workspace per (payload dtype, context) actually replayed;
    /// in practice one or two. It is keyed rather than shared because a device
    /// buffer carries its payload dtype, and it is owned here rather than by a
    /// structure so that two structures of the same dtype reuse one allocation.
    workspaces: Vec<DeviceWorkspace>,
    coefficient_budget_bytes: usize,
    structure_entries: usize,
    plan_cache_budget_bytes: usize,
    required_plan_entries: usize,
    /// Plan entries this executor holds on each context it has prepared or
    /// reserved on, what it owes back to that context's ledger.
    plan_reservations: Vec<PlanReservation>,
}

/// One context's share of an executor's plan-entry reservation.
///
/// Kept per context because the prepared-structure cache is keyed by context
/// and one executor may serve several: a reservation can only be moved on the
/// context that granted it, and a structure evicted by another context's
/// insert is released at this context's next refresh or clear.
#[derive(Debug)]
struct PlanReservation {
    context: u64,
    /// Granted for the structures prepared on this context.
    structures: usize,
    /// High-water reservation for submissions through this context that have
    /// no prepared structure (the device trace).
    additional: usize,
}

/// Default device budget for uploaded coefficient vectors: 16 MiB, i.e. two
/// million f64 coefficients held for reuse. A structure larger than the whole
/// budget is still prepared; the budget bounds what is *retained* for other
/// structures, and evicting the structure under replay would only re-upload it.
pub const DEFAULT_COEFFICIENT_BUDGET_BYTES: usize = 16 * 1024 * 1024;

impl Default for CudaTreeTransformExecutor {
    fn default() -> Self {
        Self::new(
            DEFAULT_COEFFICIENT_BUDGET_BYTES,
            DEFAULT_PLAN_CACHE_BUDGET_BYTES,
        )
    }
}

impl CudaTreeTransformExecutor {
    /// An executor whose prepared-structure cache keeps at most 256 structures
    /// (the host completed-transformer cache is byte-bounded only, so this is
    /// a device-side bound of its own). Use [`Self::with_structure_entries`]
    /// for another bound.
    pub fn new(coefficient_budget_bytes: usize, plan_cache_budget_bytes: usize) -> Self {
        Self::with_structure_entries(
            coefficient_budget_bytes,
            plan_cache_budget_bytes,
            DEFAULT_STRUCTURE_CACHE_ENTRIES,
        )
    }

    /// An executor whose prepared-structure cache keeps at most
    /// `structure_entries` structures.
    ///
    /// The bound belongs to the caller: a workload alternating more
    /// transformers than it re-uploads the excess per replay, so sizing it is
    /// a memory decision, never a correctness one.
    pub fn with_structure_entries(
        coefficient_budget_bytes: usize,
        plan_cache_budget_bytes: usize,
        structure_entries: usize,
    ) -> Self {
        Self {
            prepared: PreparedDeviceCache::new(coefficient_budget_bytes, structure_entries),
            workspaces: Vec::new(),
            coefficient_budget_bytes,
            structure_entries,
            plan_cache_budget_bytes,
            required_plan_entries: 0,
            plan_reservations: Vec::new(),
        }
    }

    /// Device bytes this executor retains for reuse: the uploaded coefficient
    /// and recoupling vectors, the transform workspaces, and the bytes the
    /// context's shared scalar operands (the ones and zero templates) pin on
    /// its behalf. This is the number the device workspace budget charges.
    pub fn retained_device_bytes(&self, ctx: &CudaDenseContext) -> usize {
        self.executor_device_bytes()
            .saturating_add(ctx.scalar_operand_bytes())
    }

    /// Device bytes the executor itself retains: the uploaded coefficient and
    /// recoupling vectors plus the transform workspaces, and nothing the
    /// context owns. This is [`Self::retained_device_bytes`] without the shared
    /// context operands, so a caller reporting both can count each once.
    pub fn executor_device_bytes(&self) -> usize {
        self.prepared
            .retained_bytes()
            .saturating_add(self.workspace_device_bytes())
    }

    /// Device bytes the transform workspaces hold. Separate from
    /// [`Self::retained_device_bytes`] so a test can pin that the workspace
    /// does not grow on a warm replay.
    pub fn workspace_device_bytes(&self) -> usize {
        self.workspaces.iter().fold(0usize, |total, workspace| {
            total.saturating_add(workspace.bytes)
        })
    }

    /// Structures with device state currently held.
    pub fn prepared_structures(&self) -> usize {
        self.prepared.entry_count()
    }

    /// Distinct cuTENSOR operand signatures the currently prepared structures
    /// submit together, i.e. the plan-cache size a thrash-free warm replay of
    /// all of them needs.
    ///
    /// A nonzero caller scale's value does not enter it: it swaps the buffer
    /// the 1x1 coefficient operand is read from, but not that operand's view
    /// metadata (`[1, 1]` extents and strides) nor its alignment, which is
    /// `size_of::<D>()` for every view, and the accumulation scalars are not in
    /// a contraction plan key at all. A zero scale does: it zero-fills the
    /// written destinations instead (#1438), and a structure's fill signatures
    /// are added from its first such replay on.
    pub fn required_plan_entries(&self) -> usize {
        self.required_plan_entries
    }

    /// Reserves plan entries for `signatures` distinct operand signatures a
    /// caller submits through `ctx` without a prepared structure (the device
    /// trace), under this executor's plan-cache budget.
    ///
    /// The reservation is a high-water mark: only growth over the largest
    /// count reserved before is added, so repeated calls do not accumulate.
    /// It is returned by [`Self::clear`].
    pub fn reserve_plan_entries_for_additional(
        &mut self,
        ctx: &mut CudaDenseContext,
        signatures: usize,
    ) -> Result<(), OperationError> {
        let target = plan_cache_entries_for(signatures, self.plan_cache_budget_bytes);
        let reservation = self.plan_reservation(ctx.identity());
        if target > reservation.additional {
            let granted = ctx
                .reserve_plan_entries(target - reservation.additional)
                .map_err(OperationError::Dense)?;
            reservation.additional += granted;
        }
        Ok(())
    }

    /// Drops every prepared structure and returns every plan entry this
    /// executor reserved on `ctx`. The next replay re-prepares whatever it
    /// needs, so this is a memory decision, never a correctness one.
    ///
    /// Reservations on other contexts stay recorded, since only their own
    /// context can take them back; their structures are gone, so each is
    /// released at that context's next refresh or clear.
    pub fn clear(&mut self, ctx: &mut CudaDenseContext) {
        self.prepared =
            PreparedDeviceCache::new(self.coefficient_budget_bytes, self.structure_entries);
        self.workspaces.clear();
        self.required_plan_entries = 0;
        let context = ctx.identity();
        if let Some(index) = self
            .plan_reservations
            .iter()
            .position(|reservation| reservation.context == context)
        {
            let reservation = self.plan_reservations.swap_remove(index);
            ctx.release_plan_entries(reservation.structures + reservation.additional);
        }
    }

    fn plan_reservation(&mut self, context: u64) -> &mut PlanReservation {
        let index = match self
            .plan_reservations
            .iter()
            .position(|reservation| reservation.context == context)
        {
            Some(index) => index,
            None => {
                self.plan_reservations.push(PlanReservation {
                    context,
                    structures: 0,
                    additional: 0,
                });
                self.plan_reservations.len() - 1
            }
        };
        &mut self.plan_reservations[index]
    }

    /// Replays `structure` from `src` into `dst` on `ctx`'s device.
    ///
    /// Semantics are the host executor's, for the same structure: each Single
    /// block writes `coefficient * [conj] src_block` into its destination
    /// layout, each Multi block packs its source layouts into compact columns,
    /// multiplies them by the transposed recoupling matrix and scatters the
    /// result columns into its destination layouts, `Overwrite` additionally
    /// zeroes every inactive destination layout, and the order is the
    /// structure's own block order followed by the recoupling plan's job order.
    /// The source conjugation baked into the structure (`storage_conjugate`) is
    /// honoured as a flag on the read, never as a materialized buffer.
    ///
    /// # Caller scale
    ///
    /// `alpha` is the caller's scale on the transformed source, the host's own
    /// `alpha` argument (`dst = alpha * T(src)` under [`Overwrite`], plus
    /// `beta * dst` under [`Axpby(beta)`]). It reaches exactly where the host applies
    /// it: every Single-block move and every Multi-block *scatter*, never the
    /// pack columns and never the recoupling GEMM, which the host runs with
    /// `1`. The inactive destination layouts ignore it, as they do on the
    /// host: they become `beta * dst` through a zero-source move with `beta`
    /// in its epilogue (a zero fill for `beta = 0`, so [`Overwrite`] still
    /// cleans a poisoned destination, and nothing for `beta = 1`).
    ///
    /// `beta` rides the epilogue of the move or scatter that writes each
    /// active layout; every destination layout is written by exactly one of
    /// them, so `beta` is applied once per element and never as a separate
    /// pass.
    ///
    /// `alpha == 0` (IEEE comparison, so `-0.0` is a zero scale) follows
    /// VectorInterface's `scale(x, 0) = zero(x) * 0`, as the host does
    /// (#1438): whatever the source holds, every written and inactive layout
    /// becomes `beta * dst` (region zero fills under `Overwrite`, nothing under
    /// `Axpby(1)`; no pack, GEMM or source read). A Single block whose
    /// structural coefficient is zero is treated the same way.
    ///
    /// Disclosed differences from the host's arithmetic, all within dtype
    /// tolerance and none of them a change of the written block set:
    ///
    /// - a Single block rounds as `alpha * (c * x)` where the host folds the
    ///   scales first and rounds as `(alpha * c) * x`; the overflow position
    ///   moves with it. Multi blocks agree exactly in order (`alpha * (U x)`);
    /// - a zero scale writes `+0` under `Overwrite` and leaves a `-0` under
    ///   `Axpby(1)`, where the host writes `0 * α′` and `dst + 0 * α′`: only
    ///   the sign of a zero may differ;
    /// - the caller scale is not part of any cache or plan key, so an
    ///   `alpha == 0` replay adds no prepared-structure entry, and its zero
    ///   fills reuse the region-zero plan signatures.
    ///
    /// [`Overwrite`]: CudaTreeTransformDestination::Overwrite
    /// [`Axpby(1)`]: CudaTreeTransformDestination::Axpby
    /// [`Axpby(beta)`]: CudaTreeTransformDestination::Axpby
    ///
    /// # Capability boundaries
    ///
    /// All of these are reported before any device work — no upload, no
    /// allocation, no plan-cache change:
    ///
    /// - a layout the device region primitive cannot express (a negative
    ///   stride or offset, or a destination that is not proven injective).
    ///
    /// # Cost
    ///
    /// One `dot_general` submission per non-empty Single block, per pack and
    /// scatter column and per inactive destination layout, plus one GEMM per
    /// recoupling job — the same counts as the host's strided passes and its
    /// `matmul_batch_axpby_into` jobs. A zero `alpha` submits only the
    /// zero-source moves over the written layouts (the first one may grow the
    /// zero template to the largest written layout, one upload) and nothing
    /// under `Axpby(1)`. The first replay of a structure uploads
    /// its coefficient vector, sizes the context zero template and grows the
    /// transform workspace to `Σ_b element_count_b × (src_count_b + dst_count_b)`
    /// elements; every later replay of the same structure on the same context
    /// and dtype transfers nothing and allocates no device buffer.
    #[expect(
        clippy::too_many_arguments,
        reason = "the device replay boundary keeps context, structure, replay structures, buffers and destination mode explicit, exactly as the host raw entry points do"
    )]
    pub fn replay<D, C>(
        &mut self,
        ctx: &mut CudaDenseContext,
        structure: &TreeTransformStructure<C>,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst: &mut CudaStorage<D>,
        src: &CudaStorage<D>,
        alpha: D,
        mode: CudaTreeTransformDestination<D>,
    ) -> Result<(), OperationError>
    where
        D: CudaScalar + RecouplingCoefficientAction<C> + PartialEq + 'static,
        C: Copy,
    {
        self.replay_with_destination_scales(
            ctx,
            structure,
            dst_structure,
            src_structure,
            dst,
            src,
            alpha,
            mode,
            &[],
        )
    }

    /// [`Self::replay`] with the caller scale of destination block `b`
    /// multiplied by `θ_b`: block `b` receives `alpha * θ_b * T(src)_b` —
    /// under [`Overwrite`], the plain replay followed by scaling block `b` by
    /// `θ_b` (up to the sign of the zeros it writes to inactive layouts); under
    /// [`Axpby(1)`], `dst_b + alpha * θ_b * T(src)_b`. `destination_scales`
    /// lists `(destination block offset, θ_b)` sorted by strictly increasing
    /// offset; a block it does not list has `θ_b = 1`. This is the fermionic
    /// contraction twist of a general contraction on whichever operand is
    /// materialized for it — TensorKit's `twist!` after `tensoradd!` in
    /// `blas_contract!` (tensoroperations.jl:419/429 @cfaa073), which QSpace
    /// folds into its per-block GEMM scalar (QSpace_aux.cc:105-115 @dd2cc7e)
    /// and the host folds into the same moves
    /// (`tree_transform_structure_overwrite_with_structural_recoupling_raw`).
    ///
    /// The θ reaches only the descriptor alpha of the Single move or Multi
    /// scatter that writes block `b`, which becomes `alpha * θ_b`. Packs, the
    /// recoupling GEMM and the inactive-layout zero fills stay unscaled, so the
    /// write pass also applies the twist: one pass fewer than TensorKit. A
    /// Multi block's GEMM is shared by its destination layouts, and scaling
    /// each scatter by its own θ is exact either way; the contraction
    /// compiler additionally asserts θ is uniform per Multi block, which is
    /// what makes this TensorKit's "scale after the transform".
    ///
    /// The prepared structure, its uploaded coefficient vector and its cache
    /// key are those of the unscaled replay — θ is in no key — and a scaled
    /// replay uploads nothing more: θ is a descriptor scalar, not an operand.
    ///
    /// `alpha == 0` takes the zero route of [`Self::replay`] and ignores θ:
    /// `beta * dst` everywhere — zeros under `Overwrite`, the destination
    /// unchanged under `Axpby(1)` — as on the host (#1438). A fermionic twist is `±1`, so `alpha * θ_b` is
    /// non-zero for every non-zero `alpha`; a θ that would make it zero is
    /// rejected.
    ///
    /// [`Overwrite`]: CudaTreeTransformDestination::Overwrite
    /// [`Axpby(1)`]: CudaTreeTransformDestination::Axpby
    ///
    /// # Errors
    ///
    /// Those of [`Self::replay`], and `InvalidArgument` — before any device
    /// work — for offsets that are not strictly increasing or, with a
    /// non-zero `alpha`, a θ for which `alpha * θ` is zero.
    #[doc(hidden)]
    #[expect(
        clippy::too_many_arguments,
        reason = "the replay boundary plus the per-destination-block scale list"
    )]
    pub fn replay_with_destination_scales<D, C>(
        &mut self,
        ctx: &mut CudaDenseContext,
        structure: &TreeTransformStructure<C>,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst: &mut CudaStorage<D>,
        src: &CudaStorage<D>,
        alpha: D,
        mode: CudaTreeTransformDestination<D>,
        destination_scales: &[(usize, C)],
    ) -> Result<(), OperationError>
    where
        D: CudaScalar + RecouplingCoefficientAction<C> + PartialEq + 'static,
        C: Copy,
    {
        if destination_scales
            .windows(2)
            .any(|pair| pair[0].0 >= pair[1].0)
        {
            return Err(OperationError::InvalidArgument {
                message: "device tree transform destination scales must have strictly \
                          increasing offsets",
            });
        }
        if alpha != D::ZERO
            && destination_scales
                .iter()
                .any(|&(_, theta)| alpha.scale_by_coefficient(theta) == D::ZERO)
        {
            return Err(OperationError::InvalidArgument {
                message: "device tree transform destination scale must not vanish",
            });
        }
        let beta = match mode {
            CudaTreeTransformDestination::Overwrite => CudaRegionBeta::Overwrite,
            CudaTreeTransformDestination::Axpby(beta) if beta == D::ZERO => {
                CudaRegionBeta::Overwrite
            }
            CudaTreeTransformDestination::Axpby(beta) if beta == D::ONE => {
                CudaRegionBeta::Accumulate
            }
            CudaTreeTransformDestination::Axpby(beta) => CudaRegionBeta::Scale(beta),
        };
        let task = structure.task_view()?;
        let context = ContextIdentity(ctx.identity());
        let executor = ExecutorSnapshot {
            placement: Placement::Cuda(ctx.device()),
            context,
            supports_strided: true,
            // Recoupling GEMMs are submitted through the region GEMM adapter.
            supports_matrix: true,
            scalar: TypeId::of::<D>(),
            // No host fused-index scratch: the device reads the structure's
            // baked fused layout directly, so it declares zero workers.
            fused_index_workers: 0,
        };
        let dst_snapshot = storage_snapshot::<D>(dst, executor);
        let src_snapshot = storage_snapshot::<D>(src, executor);
        let admission = validate_stage_a::<D, C>(
            task,
            dst_structure,
            src_structure,
            dst_snapshot,
            src_snapshot,
            executor,
        )?;

        let key = StructureKey {
            structure: task.admission_identity(),
            scalar: TypeId::of::<D>(),
            context: ctx.identity(),
        };
        // One key scan per replay: the entry is resolved to an index here and
        // everything below reads it through that index.
        let index = match self.prepared.position(&key) {
            Some(index) => index,
            None => {
                self.prepare::<D, C>(ctx, task, &key)?;
                self.prepared
                    .position(&key)
                    .ok_or(OperationError::InvalidArgument {
                        message: "device tree transform state was not prepared",
                    })?
            }
        };
        // Before any submission, so the fills below find their plans covered.
        // A scaling fill has the zero fill's region signature.
        if !matches!(beta, CudaRegionBeta::Accumulate) {
            if let Some(entry) = self.prepared.value_mut(index) {
                if !entry.zero_fills_reserved && (alpha == D::ZERO || entry.zero_coefficient_moves)
                {
                    // Set before the refresh because the refresh sums the
                    // flag; undone on failure so the next such replay retries
                    // instead of submitting unreserved fills.
                    entry.zero_fills_reserved = true;
                    if let Err(error) = self.refresh_plan_cache(ctx) {
                        if let Some(entry) = self.prepared.value_mut(index) {
                            entry.zero_fills_reserved = false;
                        }
                        return Err(error);
                    }
                }
            }
        }

        let Self {
            prepared,
            workspaces,
            ..
        } = self;
        let prepared = prepared
            .touch(index)
            .ok_or(OperationError::InvalidArgument {
                message: "device tree transform state disappeared between preparation and replay",
            })?;
        let scratch = workspaces.iter().position(|workspace| {
            workspace.scalar == TypeId::of::<D>() && workspace.context == key.context
        });
        let workspace =
            workspace_snapshot::<D>(prepared, scratch.map(|index| &workspaces[index]), executor);
        validate_stage_c::<D, C>(
            task,
            dst_snapshot,
            src_snapshot,
            &workspace,
            executor,
            &admission,
        )?;

        // VectorInterface's `scale(x, 0) = zero(x) * 0`, as on the host: a
        // zero caller scale leaves `beta * dst` over every written layout —
        // zeros under Overwrite, the destination as it is under Axpby(1) —
        // without reading the source. `-0.0 == 0.0` under IEEE comparison, so
        // `-0.0` is zero too.
        let zero_scale = alpha == D::ZERO;
        let accumulate = matches!(beta, CudaRegionBeta::Accumulate);
        let destination_beta = match beta {
            CudaRegionBeta::Overwrite => D::ZERO,
            CudaRegionBeta::Accumulate => D::ONE,
            CudaRegionBeta::Scale(beta) => beta,
        };

        // Idempotent once the template is long enough, so a warm replay uploads
        // nothing; sized from the structure's longest inactive layout rather
        // than grown one fill at a time. After Stage C, so no admission verdict
        // is still outstanding when it uploads — the design's "reserve before
        // Stage C" wording would upload ahead of a verdict that can still
        // reject, which the rejection-order contract forbids; reserving here is
        // still before every submission.
        let template_len = if zero_scale && !accumulate {
            prepared.max_zero_len.max(prepared.max_write_len)
        } else {
            prepared.max_zero_len
        };
        ctx.reserve_zero_template::<D>(template_len)
            .map_err(OperationError::Dense)?;

        let conjugate = task.storage_conjugate();
        // Inactive layouts receive no move to carry `beta`, so they take the
        // zero-source move with `beta` in its epilogue (a zero fill for
        // `beta = 0`, nothing for `beta = 1`).
        for zero in &prepared.zeros {
            cuda_region_scale::<D>(ctx, &mut dst.0, zero, destination_beta)
                .map_err(OperationError::Dense)?;
        }
        if zero_scale {
            let written = prepared.moves.iter().chain(
                prepared
                    .recouplings
                    .iter()
                    .flat_map(|recoupling| &recoupling.scatters),
            );
            for entry in written {
                cuda_region_scale::<D>(ctx, &mut dst.0, &entry.destination, destination_beta)
                    .map_err(OperationError::Dense)?;
            }
            return Ok(());
        }
        for entry in &prepared.moves {
            if entry.zero_coefficient {
                cuda_region_scale::<D>(ctx, &mut dst.0, &entry.destination, destination_beta)
                    .map_err(OperationError::Dense)?;
                continue;
            }
            submit_move::<D>(
                ctx,
                &src.0,
                prepared.coefficients.as_deref(),
                entry,
                conjugate,
                destination_scaled(alpha, destination_scales, entry),
                beta,
                &mut dst.0,
            )?;
        }
        if !prepared.needs_workspace {
            return Ok(());
        }
        let scratch = scratch.map(|index| &mut workspaces[index]).ok_or(
            OperationError::InvalidArgument {
                message:
                    "device tree transform workspace disappeared between preparation and replay",
            },
        )?;
        let coefficients =
            prepared
                .coefficients
                .as_deref()
                .ok_or(OperationError::InvalidArgument {
                    message: "device tree transform recoupling block has no uploaded matrix",
                })?;
        // Host order exactly: pack every source column of one job, multiply,
        // scatter every destination column, then move to the next job
        // (`transform_replay/batched.rs`, the serial `recoupling_plan.entries()` loop).
        for recoupling in &prepared.recouplings {
            for pack in &recoupling.packs {
                submit_move::<D>(
                    ctx,
                    &src.0,
                    None,
                    pack,
                    conjugate,
                    // The host packs and multiplies unscaled and applies the
                    // caller scale at the scatter alone.
                    D::ONE,
                    CudaRegionBeta::Overwrite,
                    &mut scratch.source,
                )?;
            }
            if recoupling.rows != 0 && recoupling.contracted != 0 && recoupling.cols != 0 {
                // `dst[rows x cols] = lhs[rows x contracted] * rhs[contracted x
                // cols]` with the row-major `U[dst, src]` buffer read as the
                // column-major `(src_count x dst_count)` matrix `Uᵀ`: leading
                // dimension `contracted` makes element `(s, d)` of the operand
                // the host's `U[d][s]`. This is the host's
                // `recoupling_gemm_batch` contract and TensorKit's
                // `mul!(dst, src, transpose(U))`.
                cuda_matmul_region_into::<D>(
                    ctx,
                    &mut scratch.destination,
                    recoupling.dst_offset,
                    &scratch.source,
                    recoupling.lhs_offset,
                    coefficients,
                    recoupling.matrix_offset,
                    recoupling.rows,
                    recoupling.contracted,
                    recoupling.cols,
                )
                .map_err(OperationError::Dense)?;
            }
            for scatter in &recoupling.scatters {
                submit_move::<D>(
                    ctx,
                    &scratch.destination,
                    None,
                    scatter,
                    // The host scatter copies the packed column unconjugated:
                    // the conjugation was already applied on the pack.
                    false,
                    destination_scaled(alpha, destination_scales, scatter),
                    beta,
                    &mut dst.0,
                )?;
            }
        }
        Ok(())
    }

    fn prepare<D, C>(
        &mut self,
        ctx: &mut CudaDenseContext,
        task: TreeTransformTaskView<'_, C>,
        key: &StructureKey,
    ) -> Result<(), OperationError>
    where
        D: CudaScalar + RecouplingCoefficientAction<C> + PartialEq + 'static,
        C: Copy,
    {
        // Everything that can reject the structure runs here, before the first
        // byte crosses to the device: every region's expressibility and
        // destination-writability, the recoupling plan's agreement with its
        // blocks, and the workspace size arithmetic.
        let plan = compile_device_plan(task)?;
        task.validate_workspace_requirements::<D>()?;
        let mut prepared = prepared_structure(
            &plan,
            CoefficientReadiness {
                structure_and_layout: key.structure.clone(),
                scalar: TypeId::of::<D>(),
                context: ContextIdentity(key.context),
            },
            task.workspace_requirements().converted_coefficient_len,
        )?;

        self.grow_workspace::<D>(
            ctx,
            key.context,
            plan.workspace_source_len,
            plan.workspace_destination_len,
        )?;

        // A move's coefficient index is a Single scalar's, and the scalars
        // lead the payload, so this reads the host scalars, not the upload.
        let singles = task.single_coefficients();
        for entry in &mut prepared.moves {
            entry.zero_coefficient = entry.coefficient.is_some_and(|index| {
                singles
                    .get(index)
                    .is_some_and(|&value| D::coefficient_as_data(value) == D::ZERO)
            });
        }
        prepared.zero_coefficient_moves = prepared.moves.iter().any(|entry| entry.zero_coefficient);
        let device_bytes = task
            .coefficient_len()
            .saturating_mul(core::mem::size_of::<D>());
        prepared.coefficient_identity = task.coefficient_identity();
        let resident = self
            .prepared
            .values_on(key.context)
            .find(|entry| {
                entry.readiness.scalar == key.scalar
                    && Weak::ptr_eq(&entry.coefficient_identity, &prepared.coefficient_identity)
            })
            .and_then(|entry| entry.coefficients.clone());
        prepared.coefficients = match resident {
            Some(payload) => Some(payload),
            None => {
                // The whole payload, converted once, indexed exactly as the
                // structure indexes it: a Single block's scalar at its own
                // coefficient index and a Multi block's matrix as the run at its
                // own `coefficient_start`. This gathers the shared per-group
                // matrices into the one upload; the gather is the conversion
                // copy the upload already needed.
                let mut values: Vec<D> = Vec::new();
                task.gather_converted_coefficients_into(&mut values, D::coefficient_as_data);
                if values.is_empty() {
                    None
                } else {
                    Some(Arc::new(
                        CudaDenseStorage::upload_owned::<D>(ctx, values)
                            .map_err(OperationError::Dense)?,
                    ))
                }
            }
        };
        self.prepared.insert(key.clone(), prepared, device_bytes);
        self.refresh_plan_cache(ctx)
    }

    /// Makes this (dtype, context)'s transform workspace hold at least
    /// `source_len` and `destination_len` elements.
    ///
    /// Growth is monotonic: a shorter requirement reuses the resident buffers,
    /// so a replay sequence pays at most one allocation per high-water mark and
    /// a warm replay pays none. A workspace is not shrunk, because the next
    /// structure of the same dtype would only have to allocate again.
    fn grow_workspace<D: CudaScalar + 'static>(
        &mut self,
        ctx: &CudaDenseContext,
        context: u64,
        source_len: usize,
        destination_len: usize,
    ) -> Result<(), OperationError> {
        if source_len == 0 && destination_len == 0 {
            return Ok(());
        }
        let scalar = TypeId::of::<D>();
        let resident = self
            .workspaces
            .iter()
            .position(|workspace| workspace.scalar == scalar && workspace.context == context);
        let (have_source, have_destination) = resident.map_or((0, 0), |index| {
            let workspace = &self.workspaces[index];
            (workspace.source.len(), workspace.destination.len())
        });
        if have_source >= source_len && have_destination >= destination_len {
            return Ok(());
        }
        let source_len = source_len.max(have_source);
        let destination_len = destination_len.max(have_destination);
        let bytes = source_len
            .checked_add(destination_len)
            .and_then(|len| len.checked_mul(core::mem::size_of::<D>()))
            .ok_or(OperationError::ElementCountOverflow)?;
        // Growth costs one host↔device transfer of zeros per buffer, the same
        // way every other device allocation in symtn is made (#740), rather
        // than through `cubecl::Session::alloc_zero_output` (tenferro-rs#1833).
        // The values are irrelevant — every column is fully written before it
        // is read — and a warm replay pays none of it.
        let source = CudaDenseStorage::upload_owned::<D>(ctx, vec![D::ZERO; source_len])
            .map_err(OperationError::Dense)?;
        let destination = CudaDenseStorage::upload_owned::<D>(ctx, vec![D::ZERO; destination_len])
            .map_err(OperationError::Dense)?;
        let grown = DeviceWorkspace {
            scalar,
            context,
            source,
            destination,
            bytes,
        };
        match resident {
            Some(index) => self.workspaces[index] = grown,
            None => self.workspaces.push(grown),
        }
        Ok(())
    }

    /// Moves this executor's plan-entry reservation on `ctx` to what the
    /// structures prepared on `ctx` need together, under this executor's byte
    /// budget.
    ///
    /// Counting is structural — the distinct baked fused signatures of a
    /// structure, computed once on the host when it is prepared — not a timing
    /// heuristic. The change is a signed delta against what the executor holds
    /// on `ctx`, so a total that falls (an insert evicted a structure) is
    /// released, and a shortfall left by an exhausted budget is requested again
    /// next time.
    fn refresh_plan_cache(&mut self, ctx: &mut CudaDenseContext) -> Result<(), OperationError> {
        let signatures = |values: &mut dyn Iterator<Item = &PreparedStructure>| {
            values.fold(0usize, |total, prepared| {
                let zero_fill = if prepared.zero_fills_reserved {
                    prepared.plan_signatures.zero_fill
                } else {
                    0
                };
                total
                    .saturating_add(prepared.plan_signatures.submitted)
                    .saturating_add(zero_fill)
            })
        };
        self.required_plan_entries = signatures(&mut self.prepared.values());
        let context = ctx.identity();
        let required = signatures(&mut self.prepared.values_on(context));
        let target = plan_cache_entries_for(required, self.plan_cache_budget_bytes);
        let reservation = self.plan_reservation(context);
        if target > reservation.structures {
            let granted = ctx
                .reserve_plan_entries(target - reservation.structures)
                .map_err(OperationError::Dense)?;
            reservation.structures += granted;
        } else {
            ctx.release_plan_entries(reservation.structures - target);
            reservation.structures = target;
        }
        Ok(())
    }
}

/// Turns the host plan into the device regions the primitive takes, rejecting
/// every destination layout the device cannot write before the caller's
/// structure has cost anything on device.
///
/// The layout verdict depends on the region alone, so checking it here is the
/// same answer the submission would give — only early enough to be an
/// all-or-nothing capability boundary instead of a partial overwrite. Pack
/// columns are checked too: they write the workspace, and a source layout the
/// device cannot read is caught by the same region construction.
fn prepared_structure(
    plan: &DeviceTransformPlan,
    readiness: CoefficientReadiness,
    recoupling_len: usize,
) -> Result<PreparedStructure, OperationError> {
    let mut moves = Vec::with_capacity(plan.moves.len());
    for entry in &plan.moves {
        moves.push(prepared_move(entry)?);
    }
    let mut recouplings = Vec::with_capacity(plan.recouplings.len());
    for entry in &plan.recouplings {
        let mut packs = Vec::with_capacity(entry.packs.len());
        for pack in &entry.packs {
            packs.push(prepared_move(pack)?);
        }
        let mut scatters = Vec::with_capacity(entry.scatters.len());
        for scatter in &entry.scatters {
            scatters.push(prepared_move(scatter)?);
        }
        recouplings.push(PreparedRecoupling {
            packs,
            scatters,
            dst_offset: entry.job.dst_offset,
            lhs_offset: entry.job.lhs_offset,
            matrix_offset: entry.matrix_offset,
            rows: entry.job.rows,
            contracted: entry.job.contracted,
            cols: entry.job.cols,
        });
    }
    let max_write_len = moves
        .iter()
        .chain(
            recouplings
                .iter()
                .flat_map(|recoupling| &recoupling.scatters),
        )
        .map(|entry| entry.destination.element_count())
        .try_fold(0usize, |largest, count| {
            count.map(|count| largest.max(count))
        })
        .map_err(OperationError::Dense)?;
    let mut zeros = Vec::with_capacity(plan.zeros.len());
    for zero in &plan.zeros {
        let region = CudaRegion::new(zero.dims.clone(), zero.strides.clone(), zero.offset)
            .map_err(unsupported_layout)?;
        validate_destination(&region)?;
        zeros.push(region);
    }
    Ok(PreparedStructure {
        moves,
        recouplings,
        needs_workspace: plan.workspace_source_len + plan.workspace_destination_len > 0,
        zeros,
        max_zero_len: plan.max_zero_len,
        max_write_len,
        plan_signatures: plan.plan_signatures,
        zero_fills_reserved: false,
        zero_coefficient_moves: false,
        coefficients: None,
        coefficient_identity: Weak::new(),
        recoupling_len,
        readiness,
    })
}

fn prepared_move(entry: &DeviceMoveSpec) -> Result<PreparedMove, OperationError> {
    let destination = CudaRegion::new(
        entry.dims.clone(),
        entry.dst_strides.clone(),
        entry.dst_offset,
    )
    .map_err(unsupported_layout)?;
    validate_destination(&destination)?;
    Ok(PreparedMove {
        source: CudaRegion::new(
            entry.dims.clone(),
            entry.src_strides.clone(),
            entry.src_offset,
        )
        .map_err(unsupported_layout)?,
        destination,
        coefficient: entry.coefficient,
        zero_coefficient: false,
    })
}

fn validate_destination(region: &CudaRegion) -> Result<(), OperationError> {
    region
        .validate_as_destination(OP)
        .map_err(unsupported_layout)
}

const OP: &str = "cuda_tree_transform";

/// A layout the device region primitive cannot express is a capability
/// boundary of *this* executor, not a dense-backend failure: the host replays
/// it, so the caller needs the typed device-transform error to decide to stay
/// on the host rather than a `Dense` error it cannot classify.
fn unsupported_layout(error: DenseError) -> OperationError {
    match error {
        DenseError::Unsupported { .. } => OperationError::UnsupportedDeviceTreeTransform {
            message: "device tree transform cannot write this destination layout",
        },
        DenseError::RankMismatch { .. } => OperationError::UnsupportedDeviceTreeTransform {
            message: "device tree transform requires equal layout and stride ranks",
        },
        error => OperationError::Dense(error),
    }
}

/// The caller scale of the submission writing `entry`'s destination block:
/// `alpha * θ_b` where `scales` lists that block. A non-zero `alpha` times θ
/// is non-zero, which the replay checked before any device work; a zero
/// `alpha` never reaches a submission.
fn destination_scaled<D, C>(alpha: D, scales: &[(usize, C)], entry: &PreparedMove) -> D
where
    D: RecouplingCoefficientAction<C>,
    C: Copy,
{
    match scales.binary_search_by_key(&entry.destination.offset(), |&(offset, _)| offset) {
        Ok(index) => alpha.scale_by_coefficient(scales[index].1),
        Err(_) => alpha,
    }
}

/// Submits one prepared move with a non-zero caller scale on the descriptor,
/// taking the 1x1 coefficient operand from the structure's uploaded vector
/// or, where the host copies unscaled, from the context's shared `1`.
#[allow(clippy::too_many_arguments)]
fn submit_move<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
    coefficients: Option<&CudaDenseStorage>,
    entry: &PreparedMove,
    conjugate: bool,
    alpha: D,
    beta: CudaRegionBeta<D>,
    dst: &mut CudaDenseStorage,
) -> Result<(), OperationError> {
    let coefficient = match entry.coefficient {
        Some(index) => {
            let Some(coefficients) = coefficients else {
                return Err(OperationError::InvalidArgument {
                    message: "device tree transform block has no uploaded coefficient",
                });
            };
            CudaRegionCoefficient::Buffer(coefficients, index)
        }
        None => CudaRegionCoefficient::One,
    };
    cuda_region_axpby::<D>(
        ctx,
        src,
        &entry.source,
        conjugate,
        alpha,
        coefficient,
        beta,
        dst,
        &entry.destination,
    )
    .map_err(OperationError::Dense)
}

/// Admission facts of one device buffer.
///
/// Placement is read from the buffer itself, so a buffer resident on another
/// device is rejected by Stage A rather than by the first primitive call —
/// after the coefficient upload.
///
/// The allocation identity is the address of the storage handle. That is a
/// sound identity for the duration of the call: `CudaDenseStorage` is neither
/// `Clone` nor refcounted and owns its device allocation exclusively, so two
/// live handles at the same address cannot exist, and the snapshot never
/// outlives the borrows it is taken from.
///
/// Context is the executor's: a device buffer carries no context tag, and the
/// device ordinal is what distinguishes the resources a wrong context would
/// reach. Tenferro re-checks the operand's device at submission either way.
fn storage_snapshot<D: CudaScalar>(
    storage: &CudaStorage<D>,
    executor: ExecutorSnapshot,
) -> StorageSnapshot {
    let len = TensorStorage::<D>::len(storage);
    let region = if len == 0 {
        StorageRegion::Empty
    } else {
        StorageRegion::Bytes {
            domain: StorageDomain(storage.0.device() as u64),
            allocation: AllocationIdentity(std::ptr::from_ref(&storage.0) as u64),
            start: 0,
            len: len.saturating_mul(core::mem::size_of::<D>()),
        }
    };
    StorageSnapshot {
        active_len: len,
        usable_capacity: len,
        placement: TensorStorage::<D>::placement(storage),
        context: executor.context,
        region,
    }
}

/// Describes the device workspace and the uploaded recoupling matrices for
/// Stage C, exactly as they exist at that point.
///
/// The whole resident workspace buffer is the slot's capacity: Stage C compares
/// it with what this structure requires, so reporting the high-water mark a
/// wider structure grew it to is the honest number. The coefficient slot is the
/// whole uploaded payload, of which the recoupling matrices are the part Stage C
/// counts.
fn workspace_snapshot<D: CudaScalar>(
    prepared: &PreparedStructure,
    scratch: Option<&DeviceWorkspace>,
    executor: ExecutorSnapshot,
) -> WorkspaceSnapshot {
    let slot = |storage: Option<&CudaDenseStorage>| {
        storage.map_or_else(
            || empty_workspace_snapshot(executor),
            |storage| device_slot_snapshot::<D>(storage, executor),
        )
    };
    WorkspaceSnapshot {
        packed_source: slot(scratch.map(|scratch| &scratch.source)),
        packed_destination: slot(scratch.map(|scratch| &scratch.destination)),
        converted_coefficients: if prepared.recoupling_len == 0 {
            empty_workspace_snapshot(executor)
        } else {
            slot(prepared.coefficients.as_deref())
        },
        fused_index_capacity: 0,
        fused_index_placement: executor.placement,
        coefficient_readiness: Some(prepared.readiness.clone()),
    }
}

/// Admission facts of a device buffer this executor owns: a transform workspace
/// buffer, or the uploaded coefficient payload.
fn device_slot_snapshot<D: CudaScalar>(
    storage: &CudaDenseStorage,
    executor: ExecutorSnapshot,
) -> StorageSnapshot {
    let capacity = storage.len();
    StorageSnapshot {
        active_len: capacity,
        usable_capacity: capacity,
        placement: executor.placement,
        context: executor.context,
        region: if capacity == 0 {
            StorageRegion::Empty
        } else {
            StorageRegion::Bytes {
                domain: StorageDomain(storage.device() as u64),
                allocation: AllocationIdentity(std::ptr::from_ref(storage) as u64),
                start: 0,
                len: capacity.saturating_mul(core::mem::size_of::<D>()),
            }
        },
    }
}

/// The workspace slots a Single-block replay does not use. Their required
/// lengths are zero, so an empty region is the honest description rather than
/// a dummy buffer.
fn empty_workspace_snapshot(executor: ExecutorSnapshot) -> StorageSnapshot {
    StorageSnapshot {
        active_len: 0,
        usable_capacity: 0,
        placement: executor.placement,
        context: executor.context,
        region: StorageRegion::Empty,
    }
}

#[cfg(test)]
mod zero_region_tests {
    use super::*;

    fn layout(shape: &[usize], strides: &[isize], offset: isize) -> FusionScaleBlockLayout {
        FusionScaleBlockLayout {
            block: FusionStridedBlockLayout {
                shape: shape.to_vec(),
                strides: strides.to_vec(),
                offset,
            },
        }
    }

    #[test]
    fn member_zero_regions_append_the_member_axis_and_track_the_largest_block() {
        // What: each inactive block gains a trailing member axis of stride
        // member_len, keeps its own strides and offset, and the largest block
        // (before the member axis) sizes the zero template. No device needed.
        let layouts = [layout(&[2, 3], &[1, 2], 4), layout(&[5], &[1], 10)];
        let zeros = CudaMemberZeroRegions::prepare(&layouts, 16, 3).unwrap();
        let regions: Vec<_> = zeros
            .regions()
            .iter()
            .map(|region| {
                (
                    region.dims().to_vec(),
                    region.strides().to_vec(),
                    region.offset(),
                )
            })
            .collect();
        assert_eq!(
            regions,
            vec![
                (vec![2, 3, 3], vec![1, 2, 16], 4),
                (vec![5, 3], vec![1, 16], 10),
            ]
        );
        assert_eq!(zeros.max_zero_len(), 6);
    }

    #[test]
    fn single_zero_regions_are_the_member_free_case_written_in_place() {
        // What: the single-tensor fill yields the same blocks without the
        // member axis and reuses (truncates) the caller's vector.
        let layouts = [layout(&[2, 3], &[1, 2], 4)];
        let mut regions = vec![CudaRegion::packed(&[7], 0).unwrap(); 3];
        let filled = CudaMemberZeroRegions::fill_single(&mut regions, &layouts).unwrap();
        assert_eq!(filled.len(), 1);
        assert_eq!(filled[0].dims(), &[2, 3]);
        assert_eq!(filled[0].strides(), &[1, 2]);
        assert_eq!(filled[0].offset(), 4);
    }

    #[test]
    fn zero_regions_reject_negative_strides_and_out_of_stack_blocks() {
        // What: a negatively strided block has no device region, and a block
        // reaching past its member slot is rejected before any device work.
        for layouts in [vec![layout(&[2], &[-1], 4)], vec![layout(&[2], &[1], -1)]] {
            assert!(matches!(
                CudaMemberZeroRegions::prepare(&layouts, 8, 2),
                Err(OperationError::UnsupportedTensorContractScope { .. })
            ));
            assert!(matches!(
                CudaMemberZeroRegions::fill_single(&mut Vec::new(), &layouts),
                Err(OperationError::UnsupportedTensorContractScope { .. })
            ));
            assert!(matches!(
                CudaMemberZeroRegions::admit(&layouts),
                Err(OperationError::UnsupportedTensorContractScope { message })
                    if message == "device contraction cannot zero a negatively strided destination block"
            ));
        }
        assert!(CudaMemberZeroRegions::prepare(&[layout(&[4], &[1], 6)], 8, 2).is_err());
    }
}
