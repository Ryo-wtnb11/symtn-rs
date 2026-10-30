//! The one CUDA executor of a Host-compiled contraction route (#1859).
//!
//! Eager calls and member-batched replays of every route (Core, CopyC,
//! DynamicTree) run one stage sequence over the Host's [`RouteView`]: admit,
//! source transforms, core GEMMs, output transform. An eager call is one
//! member. What differs between the two is a provider ([`CudaRouteStage`])
//! and only at the seams that are genuinely different:
//!
//! - admission and resources: eager replays through the runtime's tree
//!   transform executor and contraction scratch; a member replay through the
//!   caller-owned [`CudaContractMembersWorkspace`], prepared once per route
//!   and member count, so a warm member call never depends on the runtime's
//!   bounded structure cache, and admitted once per plan
//!   ([`StorageContractResolution::admit_cuda_members`]);
//! - kernels: eager transforms admit Single/Multi, conjugation, alpha and
//!   beta, and its GEMMs Adjoint operands and axpby; member transforms are
//!   prepared nonzero Single overwrite moves and its GEMMs batch a trailing
//!   member axis, so the submission count does not depend on `B`.
//!
//! The core step ([`core_step`]) is not a provider seam: the choice between
//! the overwrite and the axpby GEMMs, which inactive blocks still need
//! `beta * dst`, and the fill kernel are one policy.
//!
//! Reference step order: TensorKit `blas_contract!`
//! (`tensoroperations.jl:378-447` @cfaa073): `tensoradd!`/`twist!` the
//! sources into temporaries, `mul!` into the destination or a temporary,
//! `tensoradd!(C, Cnew, pAB, false, α, β)`. QSpace `QSpace::contract`
//! (`QSpace.cc:4141-4263` @dd2cc7e): `permute_to` both operands, grouped
//! GEMMs, `Permute` of the result. Neither has a device or a member axis; at
//! `B = 1` the device runs the reference sequence.

use std::sync::Arc;

use symtn_core::{BlockStructure, TensorStorage};
use symtn_dense::{
    cuda_region_scale, CudaDenseContext, CudaDenseStorage, CudaRegion, CudaScalar,
    CudaScalarOperands,
};
use symtn_operations::cuda::{CudaStackedStorageGemm, CudaStorage, CudaStorageGemm};
use symtn_operations::cuda_transform::{CudaMemberZeroRegions, CudaSingleMemberRegions};
use symtn_operations::fusion_replay::{FusionBlockContractPlan, StorageGemm};
use symtn_operations::stacked::{StackedStorageView, StackedStorageViewMut};
use symtn_operations::{CudaTreeTransformDestination, CudaTreeTransformExecutor};

use super::dynamic::cuda::{grow, scratch_entry, CudaContractScratch, ScratchBuffers};
use super::resolution::StorageContractResolution;
use super::route_host::{RouteView, SourceStage, Stage};
use crate::{
    ContractDestinationInit, DenseBlockScalar, OperationError, RecouplingCoefficientAction,
    TreeTransformStructure,
};

/// The core GEMMs' operands: the core destination (the caller's, or a
/// stage-owned one), the left and right operands in core order, the inactive
/// destination regions [`core_step`] may fill, and how the provider's GEMM
/// seam addresses the members (`P`).
pub(super) struct CoreIo<'a, D: CudaScalar, P> {
    dst: &'a mut CudaStorage<D>,
    left: &'a CudaStorage<D>,
    right: &'a CudaStorage<D>,
    regions: &'a [CudaRegion],
    members: P,
}

/// Which storage-direct entry the core GEMMs replay.
#[derive(Clone, Copy)]
pub(super) enum CoreMode<D> {
    /// `dst = core` over the active blocks (`alpha = 1, beta = 0`).
    Overwrite,
    /// `dst = alpha * core + beta * dst` over the active blocks.
    Axpby(D, D),
}

/// One provider of the stage sequence. Crate-private and statically
/// dispatched: the entry point picks the provider, never a branch in
/// [`execute_route_cuda`].
pub(super) trait CudaRouteStage<D: CudaScalar, C> {
    /// How the GEMM seam addresses the operands: nothing for one member, the
    /// member extent and member lengths for a stacked replay.
    type Members;
    /// Whether a stage-owned core destination can hold stale inactive
    /// blocks, so [`core_step`] must zero them. The member stack is born zero
    /// and only GEMMs write it (#1746), so it never does.
    const STALE_CORE_DESTINATION: bool;

    /// Every check of this call that precedes the first submission.
    /// `core_init` is the destination init [`core_step`] will apply to the
    /// core destination, so a provider sizes its inactive-region list from
    /// the same decision the fill uses.
    #[allow(clippy::too_many_arguments)]
    fn admit(
        &mut self,
        ctx: &mut CudaDenseContext,
        route: &RouteView<'_, C>,
        dst: (&Arc<BlockStructure>, &CudaStorage<D>),
        lhs: &CudaStorage<D>,
        rhs: &CudaStorage<D>,
        members: usize,
        alpha: D,
        init: ContractDestinationInit<D>,
        core_init: ContractDestinationInit<D>,
    ) -> Result<(), OperationError>;

    /// Overwrite-moves one non-borrowed source into its stage-owned buffer,
    /// the fermionic twist folded into the move.
    fn source(
        &mut self,
        ctx: &mut CudaDenseContext,
        side: Stage,
        source: &SourceStage<'_, C>,
        core: &Arc<BlockStructure>,
        payload: &CudaStorage<D>,
    ) -> Result<(), OperationError>;

    /// The core operands: `caller` when the core writes the caller's
    /// destination, else the stage-owned core destination; each copied
    /// source's buffer, else the caller's payload.
    fn core_io<'a>(
        &'a mut self,
        ctx: &mut CudaDenseContext,
        route: &RouteView<'_, C>,
        caller: Option<&'a mut CudaStorage<D>>,
        lhs: &'a CudaStorage<D>,
        rhs: &'a CudaStorage<D>,
    ) -> Result<CoreIo<'a, D, Self::Members>, OperationError>;

    /// The provider's GEMM seam over `io`.
    fn gemms(
        ctx: &mut CudaDenseContext,
        plan: &FusionBlockContractPlan<C>,
        io: &mut CoreIo<'_, D, Self::Members>,
        mode: CoreMode<D>,
    ) -> Result<(), OperationError>;

    /// The output transform from the stage-owned core destination into `dst`.
    #[allow(clippy::too_many_arguments)]
    fn output(
        &mut self,
        ctx: &mut CudaDenseContext,
        transform: &TreeTransformStructure<C>,
        dst_structure: &Arc<BlockStructure>,
        core_structure: &Arc<BlockStructure>,
        dst: &mut CudaStorage<D>,
        alpha: D,
        init: ContractDestinationInit<D>,
    ) -> Result<(), OperationError>;
}

/// Replays `route` over `members` members on `ctx`'s device:
/// `dst = alpha * contract(lhs, rhs) + init(dst)` per member.
///
/// Every check that can reject the call runs in `admit`, before the first
/// submission; the last stage is the only one that writes `dst`.
///
/// Why alpha rides the output transform whenever there is one, and the core
/// otherwise: that is TensorKit's `copyC` placement (`mul!` into the
/// temporary, then `tensoradd!(C, Cnew, α, β)`), which the device has always
/// used. The Host DynamicTree puts alpha on its core instead; both are equal
/// within tolerance, and moving either changes numerics outside #1859.
#[allow(clippy::too_many_arguments)]
pub(super) fn execute_route_cuda<St, D, C>(
    stage: &mut St,
    ctx: &mut CudaDenseContext,
    route: RouteView<'_, C>,
    (dst_structure, dst): (&Arc<BlockStructure>, &mut CudaStorage<D>),
    lhs: &CudaStorage<D>,
    rhs: &CudaStorage<D>,
    members: usize,
    alpha: D,
    init: ContractDestinationInit<D>,
) -> Result<(), OperationError>
where
    St: CudaRouteStage<D, C>,
    D: CudaScalar + num_traits::Zero,
    C: DenseBlockScalar,
{
    let (core_alpha, core_init) = match route.output {
        None => (alpha, init),
        Some(_) if St::STALE_CORE_DESTINATION => (D::ONE, ContractDestinationInit::Axpby(D::ZERO)),
        Some(_) => (D::ONE, ContractDestinationInit::Zeroed),
    };
    stage.admit(
        ctx,
        &route,
        (dst_structure, dst),
        lhs,
        rhs,
        members,
        alpha,
        init,
        core_init,
    )?;
    for (side, source, core, payload) in [
        (Stage::Lhs, &route.lhs_source, route.lhs_core, lhs),
        (Stage::Rhs, &route.rhs_source, route.rhs_core, rhs),
    ] {
        if let Some(source) = source {
            stage.source(ctx, side, source, core, payload)?;
        }
    }
    let caller = route.output.is_none().then_some(&mut *dst);
    let io = stage.core_io(ctx, &route, caller, lhs, rhs)?;
    core_step::<St, D, C>(ctx, route.plan, io, core_alpha, core_init)?;
    if let Some((transform, core_structure)) = route.output {
        stage.output(
            ctx,
            transform,
            dst_structure,
            core_structure,
            dst,
            alpha,
            init,
        )?;
    }
    Ok(())
}

/// The core step of every provider: the GEMMs write the active blocks
/// (overwrite for `alpha = 1, beta = 0`, else axpby), then the inactive
/// blocks of a destination that is not born zero become `beta * dst` (a
/// zero fill for `beta = 0`, nothing for `beta = 1`).
///
/// The fill follows the GEMMs: the plan's inactive blocks are disjoint from
/// every block its GEMMs write, so either order gives the same values and
/// the same submissions, and after keeps the plan's own range validation
/// ahead of the first write to a caller's destination.
fn core_step<St, D, C>(
    ctx: &mut CudaDenseContext,
    plan: &FusionBlockContractPlan<C>,
    mut io: CoreIo<'_, D, St::Members>,
    alpha: D,
    init: ContractDestinationInit<D>,
) -> Result<(), OperationError>
where
    St: CudaRouteStage<D, C>,
    D: CudaScalar + num_traits::Zero,
{
    let beta = init.active_beta();
    let mode = if alpha == D::ONE && beta == D::ZERO {
        CoreMode::Overwrite
    } else {
        CoreMode::Axpby(alpha, beta)
    };
    St::gemms(ctx, plan, &mut io, mode)?;
    if inactive_blocks_need_beta(init) {
        for region in io.regions {
            cuda_region_scale::<D>(ctx, &mut io.dst.0, region, beta)
                .map_err(OperationError::Dense)?;
        }
    }
    Ok(())
}

/// Whether a core destination's inactive blocks need `beta * dst`: not when
/// it is born zero, and not for `beta = 1`.
fn inactive_blocks_need_beta<D: CudaScalar>(init: ContractDestinationInit<D>) -> bool {
    matches!(init, ContractDestinationInit::Axpby(beta) if beta != D::ONE)
}

/// The storage-direct replay `mode` selects, over any GEMM seam.
fn replay_core<G, D, C, DDst, DLhs, DRhs>(
    plan: &FusionBlockContractPlan<C>,
    gemm: &mut G,
    dst: &mut DDst,
    lhs: &DLhs,
    rhs: &DRhs,
    mode: CoreMode<D>,
) -> Result<(), OperationError>
where
    G: StorageGemm<D, DDst, DLhs, DRhs>,
    D: RecouplingCoefficientAction<C>,
    C: Copy + PartialEq + num_traits::One,
    DDst: TensorStorage<D>,
    DLhs: TensorStorage<D>,
    DRhs: TensorStorage<D>,
{
    match mode {
        CoreMode::Overwrite => plan.execute_direct_on_storage_prezeroed(gemm, dst, lhs, rhs),
        CoreMode::Axpby(alpha, beta) => {
            plan.execute_direct_on_storage_axpby(gemm, dst, lhs, rhs, alpha, beta)
        }
    }
}

fn transform_destination<D>(init: ContractDestinationInit<D>) -> CudaTreeTransformDestination<D> {
    match init {
        ContractDestinationInit::Zeroed => CudaTreeTransformDestination::Overwrite,
        ContractDestinationInit::Axpby(beta) => CudaTreeTransformDestination::Axpby(beta),
    }
}

fn materialized<D: CudaScalar>(
    slot: Option<&CudaStorage<D>>,
) -> Result<&CudaStorage<D>, OperationError> {
    slot.ok_or(OperationError::InvalidArgument {
        message: "device contraction source was not materialized",
    })
}

// ---------------------------------------------------------------------------
// Eager provider
// ---------------------------------------------------------------------------

const EAGER_IS_ONE_MEMBER: OperationError = OperationError::UnsupportedTensorContractScope {
    message: "eager CUDA contraction replays exactly one member",
};

/// The eager provider: the runtime's tree transform executor and
/// contraction scratch.
struct EagerCudaStage<'a> {
    transforms: &'a mut CudaTreeTransformExecutor,
    scratch: &'a mut CudaContractScratch,
}

impl<D, C> CudaRouteStage<D, C> for EagerCudaStage<'_>
where
    D: CudaScalar + RecouplingCoefficientAction<C> + PartialEq + 'static,
    C: DenseBlockScalar,
{
    type Members = ();
    /// The scratch is shared by every contraction of its dtype and context.
    const STALE_CORE_DESTINATION: bool = true;

    fn admit(
        &mut self,
        _: &mut CudaDenseContext,
        route: &RouteView<'_, C>,
        _: (&Arc<BlockStructure>, &CudaStorage<D>),
        _: &CudaStorage<D>,
        _: &CudaStorage<D>,
        members: usize,
        _: D,
        _: ContractDestinationInit<D>,
        core_init: ContractDestinationInit<D>,
    ) -> Result<(), OperationError> {
        if members != 1 {
            return Err(EAGER_IS_ONE_MEMBER);
        }
        let inactive = route.plan.inactive_destination_regions();
        CudaMemberZeroRegions::admit(inactive)?;
        // The region list the core step fills, converted before the first
        // submission like every other check.
        if inactive_blocks_need_beta(core_init) {
            CudaMemberZeroRegions::fill_single(&mut self.scratch.zero_regions, inactive)?;
        } else {
            self.scratch.zero_regions.clear();
        }
        Ok(())
    }

    fn source(
        &mut self,
        ctx: &mut CudaDenseContext,
        side: Stage,
        source: &SourceStage<'_, C>,
        core: &Arc<BlockStructure>,
        payload: &CudaStorage<D>,
    ) -> Result<(), OperationError> {
        let (bytes, buffers) = scratch_entry::<D>(&mut self.scratch.entries, ctx.identity())?;
        let slot = match side {
            Stage::Lhs => &mut buffers.lhs,
            Stage::Rhs => &mut buffers.rhs,
            Stage::Output => {
                return Err(OperationError::InvalidArgument {
                    message: "an output transform is not a source stage",
                })
            }
        };
        let buffer = grow(ctx, slot, bytes, core.required_len()?)?;
        self.transforms.replay_with_destination_scales(
            ctx,
            source.transform,
            core,
            source.payload_structure,
            buffer,
            payload,
            D::ONE,
            CudaTreeTransformDestination::Overwrite,
            source.scales,
        )
    }

    fn core_io<'a>(
        &'a mut self,
        ctx: &mut CudaDenseContext,
        route: &RouteView<'_, C>,
        caller: Option<&'a mut CudaStorage<D>>,
        lhs: &'a CudaStorage<D>,
        rhs: &'a CudaStorage<D>,
    ) -> Result<CoreIo<'a, D, ()>, OperationError> {
        let CudaContractScratch {
            entries,
            zero_regions,
        } = &mut *self.scratch;
        let (dst, lhs, rhs) = match (
            caller,
            route.lhs_source.is_some() || route.rhs_source.is_some(),
        ) {
            // The Core route touches no scratch entry.
            (Some(caller), false) => (caller, lhs, rhs),
            (caller, _) => {
                let (bytes, buffers) = scratch_entry::<D>(entries, ctx.identity())?;
                let ScratchBuffers {
                    lhs: lhs_slot,
                    rhs: rhs_slot,
                    dst: dst_slot,
                } = buffers;
                let dst = match (caller, route.output) {
                    (Some(caller), _) => caller,
                    (None, Some((_, core))) => grow(ctx, dst_slot, bytes, core.required_len()?)?,
                    (None, None) => {
                        return Err(OperationError::InvalidArgument {
                            message: "device contraction core destination is missing",
                        })
                    }
                };
                let lhs = if route.lhs_source.is_some() {
                    materialized(lhs_slot.as_ref())?
                } else {
                    lhs
                };
                let rhs = if route.rhs_source.is_some() {
                    materialized(rhs_slot.as_ref())?
                } else {
                    rhs
                };
                (dst, lhs, rhs)
            }
        };
        let (left, right) = route.core_order(lhs, rhs);
        Ok(CoreIo {
            dst,
            left,
            right,
            regions: zero_regions,
            members: (),
        })
    }

    fn gemms(
        ctx: &mut CudaDenseContext,
        plan: &FusionBlockContractPlan<C>,
        io: &mut CoreIo<'_, D, ()>,
        mode: CoreMode<D>,
    ) -> Result<(), OperationError> {
        replay_core(
            plan,
            &mut CudaStorageGemm::new(ctx),
            io.dst,
            io.left,
            io.right,
            mode,
        )
    }

    fn output(
        &mut self,
        ctx: &mut CudaDenseContext,
        transform: &TreeTransformStructure<C>,
        dst_structure: &Arc<BlockStructure>,
        core_structure: &Arc<BlockStructure>,
        dst: &mut CudaStorage<D>,
        alpha: D,
        init: ContractDestinationInit<D>,
    ) -> Result<(), OperationError> {
        let (_, buffers) = scratch_entry::<D>(&mut self.scratch.entries, ctx.identity())?;
        let core = materialized(buffers.dst.as_ref())?;
        self.transforms.replay(
            ctx,
            transform,
            dst_structure,
            core_structure,
            dst,
            core,
            alpha,
            transform_destination(init),
        )
    }
}

/// Replays a Host-compiled storage contraction route on `ctx`'s device into
/// `dst`: `dst = alpha * contract(lhs, rhs) + beta * dst`, as one member of
/// the stage sequence ([`execute_route_cuda`]).
///
/// `init` is [`ContractDestinationInit::Zeroed`] when the caller provides
/// `dst` zero-filled (the returning path's fresh #740 output), and otherwise
/// `Axpby(beta)` for a retained destination (`contract_into`). `beta` rides
/// whatever writes each element: the core GEMMs when they write `dst`
/// directly, or the output transform. Only the core plan's inactive blocks of
/// a directly written `dst` have no writer; they become `beta * dst` through
/// a zero-source region move. No pass reads or clears the whole destination.
///
/// A borrowed source is read in place, as on the host: an identity
/// transform of an operand already in core layout, or of a lazy adjoint
/// whose parent tiles canonical coupled-sector matrices, which the core
/// GEMM reads in its parent's storage with an adjoint op (#2160). The fermionic
/// contraction twist is folded into the source move writing each twisted
/// block (`replay_with_destination_scales`), as the Host folds it.
///
/// Every check that can reject the route runs before the first device
/// submission.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn execute_storage_contract_resolution_on_cuda<D, C>(
    ctx: &mut CudaDenseContext,
    transforms: &mut CudaTreeTransformExecutor,
    scratch: &mut CudaContractScratch,
    resolution: &StorageContractResolution<C>,
    dst: (&Arc<BlockStructure>, &mut CudaStorage<D>),
    (lhs_structure, lhs): (&Arc<BlockStructure>, &CudaStorage<D>),
    (rhs_structure, rhs): (&Arc<BlockStructure>, &CudaStorage<D>),
    alpha: D,
    init: ContractDestinationInit<D>,
) -> Result<(), OperationError>
where
    D: CudaScalar + RecouplingCoefficientAction<C> + PartialEq + num_traits::Zero + 'static,
    C: DenseBlockScalar,
{
    execute_route_cuda(
        &mut EagerCudaStage {
            transforms,
            scratch,
        },
        ctx,
        RouteView::of(&resolution.route, lhs_structure, rhs_structure),
        dst,
        lhs,
        rhs,
        1,
        alpha,
        init,
    )
}

// ---------------------------------------------------------------------------
// Member provider
// ---------------------------------------------------------------------------

/// Caller-owned device state of member-batched replays of one route:
/// validated member regions and core zero fills for the current `B`,
/// scaled-move coefficients, and the transformed-source and
/// core-destination stacks at their high-water member capacity. `S` is the
/// plan's device storage, so the holder needs no payload bound.
///
/// Retained capacity has no reference counterpart: TensorKit `cfaa073e`
/// `blas_contract!` takes its temporaries from a per-call allocator and QSpace
/// `dd2cc7e` `contract_matchAB_groupC` builds its buffers per call; neither
/// has a member axis to size.
#[doc(hidden)]
pub struct CudaContractMembersWorkspace<S> {
    /// The route (its resolution) and `B` the state below was prepared for.
    /// The resolution identifies every structure the state depends on: the
    /// core plan, each transform and borrow decision, and the core
    /// destination.
    prepared: Option<(Arc<StorageContractResolution<f64>>, usize)>,
    /// The engine ([`CudaDenseContext::identity`]) every device buffer below
    /// belongs to; a call on another engine starts from an empty workspace
    /// instead of submitting this engine's buffers.
    engine: Option<u64>,
    /// Members every stack in `buffers` holds; never below the prepared `B`.
    capacity: usize,
    /// lhs source, rhs source and output transform regions, by [`Stage`].
    stages: [Option<CudaSingleMemberRegions>; 3],
    /// Scaled-move coefficients per stage; structure-only, kept across `B`.
    coefficients: [Option<CudaDenseStorage>; 3],
    /// Transformed lhs, transformed rhs and core destination stacks; the core
    /// destination is a DynamicTree's or the CopyC temporary, born zero and
    /// written only by the core GEMMs (#1746). A member's offset is
    /// `B`-independent, so the first `B` members are the active stack and the
    /// rest is idle capacity.
    buffers: [Option<S>; 3],
    /// Core inactive blocks of a directly written destination, per member.
    core_zeros: Vec<CudaRegion>,
    /// Elements of the largest `core_zeros` block, before the member axis.
    core_zero_len: usize,
    /// Member lengths of the core destination, the core's left operand and
    /// its right operand.
    lens: [usize; 3],
    /// The zero and ones templates of this workspace's fills, lent to each
    /// call's context so a warm call uploads none of them. Not counted by
    /// [`Self::retained_bytes`], which reports the stacks and region
    /// metadata the member count sizes.
    operands: CudaScalarOperands,
    /// Test seam: the next output transform fails before it submits.
    #[cfg(any(test, feature = "testing"))]
    fail_before_output: bool,
}

impl<S> Default for CudaContractMembersWorkspace<S> {
    fn default() -> Self {
        Self {
            prepared: None,
            engine: None,
            capacity: 0,
            stages: [None, None, None],
            coefficients: [None, None, None],
            buffers: [None, None, None],
            core_zeros: Vec::new(),
            core_zero_len: 0,
            lens: [0; 3],
            operands: CudaScalarOperands::default(),
            #[cfg(any(test, feature = "testing"))]
            fail_before_output: false,
        }
    }
}

#[cfg(any(test, feature = "testing"))]
impl<S> CudaContractMembersWorkspace<S> {
    /// Test seam: makes the next replay of the prepared route fail after its
    /// core GEMMs and before its output transform, as a backend error there
    /// would, so failure recovery is testable after device work.
    #[doc(hidden)]
    pub fn fail_next_before_output(&mut self) {
        self.fail_before_output = true;
    }
}

impl<D: CudaScalar> CudaContractMembersWorkspace<CudaStorage<D>> {
    /// Retained device payload bytes, at stack capacity, plus Host region
    /// metadata.
    pub fn retained_bytes(&self) -> usize {
        let elements = self
            .buffers
            .iter()
            .flatten()
            .map(|buffer| buffer.0.len())
            .chain(
                self.coefficients
                    .iter()
                    .flatten()
                    .map(CudaDenseStorage::len),
            )
            .fold(0usize, usize::saturating_add);
        let regions = self
            .stages
            .iter()
            .flatten()
            .map(CudaSingleMemberRegions::retained_bytes)
            .chain(self.core_zeros.iter().map(CudaRegion::retained_heap_bytes))
            .fold(
                self.core_zeros
                    .capacity()
                    .saturating_mul(std::mem::size_of::<CudaRegion>()),
                usize::saturating_add,
            );
        elements
            .saturating_mul(std::mem::size_of::<D>())
            .saturating_add(regions)
    }
}

/// The per-call member contract every route's member replay admits, checked
/// without a device: overwrite only (`alpha = 1`, a born-zero destination or
/// `Axpby(0)`), at least one member.
fn admit_member_call<D: CudaScalar>(
    alpha: D,
    init: ContractDestinationInit<D>,
    members: usize,
) -> Result<(), OperationError> {
    if members == 0 {
        return Err(OperationError::InvalidArgument {
            message: "a CUDA member contraction needs at least one member",
        });
    }
    let overwrite = match init {
        ContractDestinationInit::Zeroed => true,
        ContractDestinationInit::Axpby(beta) => beta == D::ZERO,
    };
    if alpha != D::ONE || !overwrite {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "CUDA member contraction only overwrites its destination with alpha = 1",
        });
    }
    Ok(())
}

fn stack_len(member_len: usize, members: usize) -> Result<usize, OperationError> {
    member_len
        .checked_mul(members)
        .ok_or(OperationError::ElementCountOverflow)
}

/// How a stacked GEMM addresses `B` members: member lengths of the core
/// destination, left and right operands, each also the member stride.
#[derive(Clone, Copy)]
struct Stacks {
    members: usize,
    lens: [usize; 3],
}

/// The member provider: one workspace bound to one resolution.
struct MemberCudaStage<'a, D: CudaScalar> {
    workspace: &'a mut CudaContractMembersWorkspace<CudaStorage<D>>,
    resolution: &'a Arc<StorageContractResolution<f64>>,
}

impl<D> MemberCudaStage<'_, D>
where
    D: CudaScalar + RecouplingCoefficientAction<f64>,
{
    /// Validates every structure and region for `members` and
    /// installs them; nothing is submitted. A changed resolution or engine
    /// empties the workspace first, so a failure leaves nothing a later call
    /// trusts and no buffer of another engine is ever submitted.
    /// Stacks are replaced only above the high-water capacity (#1746).
    fn prepare(
        &mut self,
        ctx: &mut CudaDenseContext,
        route: &RouteView<'_, f64>,
        dst_structure: &Arc<BlockStructure>,
        members: usize,
    ) -> Result<(), OperationError> {
        let workspace = &mut *self.workspace;
        if workspace.engine != Some(ctx.identity())
            || !workspace
                .prepared
                .as_ref()
                .is_some_and(|(prepared, _)| Arc::ptr_eq(prepared, self.resolution))
        {
            *workspace = CudaContractMembersWorkspace::default();
            workspace.engine = Some(ctx.identity());
        }
        let dst_len = dst_structure.required_len()?;
        let core_structure = route.output.map_or(dst_structure, |(_, core)| core);
        let core_len = core_structure.required_len()?;
        let (left, right) = route.core_order(route.lhs_core, route.rhs_core);
        route
            .plan
            .validate_replay_structures(core_structure, left, right)?;
        let mut stages = [None, None, None];
        for (side, source, core) in [
            (Stage::Lhs, &route.lhs_source, route.lhs_core),
            (Stage::Rhs, &route.rhs_source, route.rhs_core),
        ] {
            let Some(source) = source else {
                continue;
            };
            let core_len = core.required_len()?;
            let source_len = source.payload_structure.required_len()?;
            stages[side as usize] = Some(CudaSingleMemberRegions::prepare_scaled(
                source.transform,
                core,
                source.payload_structure,
                core_len,
                source_len,
                stack_len(core_len, members)?,
                stack_len(source_len, members)?,
                members,
                source.scales,
            )?);
        }
        if let Some((transform, core_structure)) = route.output {
            stages[Stage::Output as usize] = Some(CudaSingleMemberRegions::prepare(
                transform,
                dst_structure,
                core_structure,
                dst_len,
                core_len,
                stack_len(dst_len, members)?,
                stack_len(core_len, members)?,
                members,
            )?);
        }
        let mut core_zeros = Vec::new();
        let mut core_zero_len = 0;
        if route.output.is_none() {
            let zeros = CudaMemberZeroRegions::prepare(
                route.plan.inactive_destination_regions(),
                dst_len,
                members,
            )?;
            core_zero_len = zeros.max_zero_len();
            core_zeros = zeros.into_regions();
        }
        for (slot, stage) in workspace.coefficients.iter_mut().zip(&stages) {
            if let Some(stage) = stage.as_ref().filter(|stage| stage.has_scaled_moves()) {
                if slot.is_none() {
                    *slot = Some(
                        CudaStorage::<D>::upload_owned(ctx, stage.scaled_coefficients::<D>())?.0,
                    );
                }
            }
        }
        // Fresh device stacks cost one zero upload each (#740); their
        // contents are irrelevant except a core stack's inactive blocks.
        if members > workspace.capacity {
            let zeros = |present: bool, len: usize| -> Result<_, OperationError> {
                present
                    .then(|| {
                        CudaStorage::upload_members(
                            ctx,
                            vec![D::ZERO; stack_len(len, members)?],
                            len,
                            members,
                        )
                    })
                    .transpose()
            };
            workspace.buffers = [
                zeros(route.lhs_source.is_some(), route.lhs_core.required_len()?)?,
                zeros(route.rhs_source.is_some(), route.rhs_core.required_len()?)?,
                zeros(route.output.is_some(), core_len)?,
            ];
            workspace.capacity = members;
        }
        workspace.stages = stages;
        workspace.core_zeros = core_zeros;
        workspace.core_zero_len = core_zero_len;
        workspace.lens = [core_len, left.required_len()?, right.required_len()?];
        workspace.prepared = Some((Arc::clone(self.resolution), members));
        Ok(())
    }
}

const UNPREPARED: OperationError = OperationError::InvalidArgument {
    message: "CUDA member stage is unprepared",
};

impl<D> CudaRouteStage<D, f64> for MemberCudaStage<'_, D>
where
    D: CudaScalar + RecouplingCoefficientAction<f64>,
{
    type Members = Stacks;
    const STALE_CORE_DESTINATION: bool = false;

    fn admit(
        &mut self,
        ctx: &mut CudaDenseContext,
        route: &RouteView<'_, f64>,
        (dst_structure, dst): (&Arc<BlockStructure>, &CudaStorage<D>),
        lhs: &CudaStorage<D>,
        rhs: &CudaStorage<D>,
        members: usize,
        alpha: D,
        init: ContractDestinationInit<D>,
        core_init: ContractDestinationInit<D>,
    ) -> Result<(), OperationError> {
        admit_member_call(alpha, init, members)?;
        // A copied source reads the caller's payload layout; a borrowed one is
        // the core layout.
        let lhs_payload = route
            .lhs_source
            .as_ref()
            .map_or(route.lhs_core, |source| source.payload_structure);
        let rhs_payload = route
            .rhs_source
            .as_ref()
            .map_or(route.rhs_core, |source| source.payload_structure);
        for (storage, structure) in [(dst, dst_structure), (lhs, lhs_payload), (rhs, rhs_payload)] {
            let expected = stack_len(structure.required_len()?, members)?;
            if storage.0.len() != expected {
                return Err(OperationError::ElementCountMismatch {
                    expected,
                    actual: storage.0.len(),
                });
            }
        }
        let prepared = self
            .workspace
            .prepared
            .as_ref()
            .is_some_and(|(resolution, prepared)| {
                *prepared == members && Arc::ptr_eq(resolution, self.resolution)
            })
            && self.workspace.engine == Some(ctx.identity());
        if !prepared {
            self.prepare(ctx, route, dst_structure, members)?;
        }
        // The templates of the fills this call submits, so they upload
        // nothing. The transform stages fill on every call; the core fills
        // only for `Axpby`, so a caller that never fills it pins no template
        // for it (#2123).
        let workspace = &*self.workspace;
        let core_fill = (inactive_blocks_need_beta(core_init) && !workspace.core_zeros.is_empty())
            .then_some(workspace.core_zero_len);
        let stages = workspace.stages.iter().flatten();
        let zero_len = stages
            .clone()
            .map(CudaSingleMemberRegions::max_zero_len)
            .chain(core_fill)
            .max()
            .unwrap_or(0);
        ctx.reserve_zero_template::<D>(stack_len(zero_len, members)?)
            .map_err(OperationError::Dense)?;
        if core_fill.is_some() || stages.into_iter().any(|stage| !stage.zeros().is_empty()) {
            ctx.reserve_ones_template::<D>(1)
                .map_err(OperationError::Dense)?;
        }
        Ok(())
    }

    fn source(
        &mut self,
        ctx: &mut CudaDenseContext,
        side: Stage,
        _: &SourceStage<'_, f64>,
        _: &Arc<BlockStructure>,
        payload: &CudaStorage<D>,
    ) -> Result<(), OperationError> {
        let workspace = &mut *self.workspace;
        let index = side as usize;
        let (Some(stage), Some(buffer)) = (&workspace.stages[index], &mut workspace.buffers[index])
        else {
            return Err(UNPREPARED);
        };
        stage.execute_overwrite(ctx, payload, workspace.coefficients[index].as_ref(), buffer)
    }

    fn core_io<'a>(
        &'a mut self,
        _: &mut CudaDenseContext,
        route: &RouteView<'_, f64>,
        caller: Option<&'a mut CudaStorage<D>>,
        lhs: &'a CudaStorage<D>,
        rhs: &'a CudaStorage<D>,
    ) -> Result<CoreIo<'a, D, Stacks>, OperationError> {
        let CudaContractMembersWorkspace {
            prepared,
            buffers: [lhs_buffer, rhs_buffer, core_buffer],
            core_zeros,
            lens,
            ..
        } = &mut *self.workspace;
        let members = prepared.as_ref().ok_or(UNPREPARED)?.1;
        let dst = match caller {
            Some(caller) => caller,
            None => core_buffer.as_mut().ok_or(UNPREPARED)?,
        };
        let lhs = if route.lhs_source.is_some() {
            lhs_buffer.as_ref().ok_or(UNPREPARED)?
        } else {
            lhs
        };
        let rhs = if route.rhs_source.is_some() {
            rhs_buffer.as_ref().ok_or(UNPREPARED)?
        } else {
            rhs
        };
        let (left, right) = route.core_order(lhs, rhs);
        Ok(CoreIo {
            dst,
            left,
            right,
            regions: core_zeros,
            members: Stacks {
                members,
                lens: *lens,
            },
        })
    }

    fn gemms(
        ctx: &mut CudaDenseContext,
        plan: &FusionBlockContractPlan<f64>,
        io: &mut CoreIo<'_, D, Stacks>,
        mode: CoreMode<D>,
    ) -> Result<(), OperationError> {
        let Stacks {
            members,
            lens: [dst_len, left_len, right_len],
        } = io.members;
        let left = StackedStorageView::new::<D>(io.left, left_len, members, left_len)?;
        let right = StackedStorageView::new::<D>(io.right, right_len, members, right_len)?;
        let mut dst = StackedStorageViewMut::new::<D>(io.dst, dst_len, members, dst_len)?;
        replay_core(
            plan,
            &mut CudaStackedStorageGemm::new(ctx),
            &mut dst,
            &left,
            &right,
            mode,
        )
    }

    fn output(
        &mut self,
        ctx: &mut CudaDenseContext,
        _: &TreeTransformStructure<f64>,
        _: &Arc<BlockStructure>,
        _: &Arc<BlockStructure>,
        dst: &mut CudaStorage<D>,
        _: D,
        _: ContractDestinationInit<D>,
    ) -> Result<(), OperationError> {
        let workspace = &mut *self.workspace;
        #[cfg(any(test, feature = "testing"))]
        if std::mem::take(&mut workspace.fail_before_output) {
            return Err(OperationError::InvalidArgument {
                message: "injected CUDA member failure before the output transform",
            });
        }
        let output = Stage::Output as usize;
        let (Some(stage), Some(core)) = (&workspace.stages[output], &workspace.buffers[output])
        else {
            return Err(UNPREPARED);
        };
        stage.execute_overwrite(ctx, core, workspace.coefficients[output].as_ref(), dst)
    }
}

/// Overwrites each of `members` member-major destination members with its
/// contraction, through the stage sequence of every eager call. `init` is
/// [`ContractDestinationInit::Zeroed`] for a stack whose core inactive
/// blocks are already zero (they get no fill, and the call reserves no
/// template for one) or `Axpby(0)`; anything else, and `alpha`
/// other than one, is not a member contract. The caller admits the route
/// once, for every `B`, with [`StorageContractResolution::admit_cuda_members`].
///
/// Every structural and region check finishes before the first device
/// submission; a warm call at an unchanged `B` transfers nothing, and a
/// changed `B` at or below the high-water capacity allocates no stack. The
/// submission count does not depend on `B`.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn execute_storage_contract_members_cuda<D>(
    ctx: &mut CudaDenseContext,
    resolution: &Arc<StorageContractResolution<f64>>,
    dst: (&Arc<BlockStructure>, &mut CudaStorage<D>),
    (lhs_structure, lhs): (&Arc<BlockStructure>, &CudaStorage<D>),
    (rhs_structure, rhs): (&Arc<BlockStructure>, &CudaStorage<D>),
    workspace: &mut CudaContractMembersWorkspace<CudaStorage<D>>,
    members: usize,
    init: ContractDestinationInit<D>,
) -> Result<(), OperationError>
where
    D: CudaScalar + RecouplingCoefficientAction<f64> + num_traits::Zero,
{
    ctx.swap_scalar_operands(&mut workspace.operands);
    let result = execute_route_cuda(
        &mut MemberCudaStage {
            workspace,
            resolution,
        },
        ctx,
        RouteView::of(&resolution.route, lhs_structure, rhs_structure),
        dst,
        lhs,
        rhs,
        members,
        D::ONE,
        init,
    );
    // Back on every outcome: a failed call keeps what it already uploaded.
    ctx.swap_scalar_operands(&mut workspace.operands);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The member contract is checked without a device, before any
    /// workspace or context is touched.
    #[test]
    fn member_admission_rejects_non_overwrite_calls_without_a_device() {
        let unsupported = |result: Result<(), OperationError>| {
            matches!(
                result,
                Err(OperationError::UnsupportedTensorContractScope { .. })
            )
        };
        assert!(admit_member_call(1.0, ContractDestinationInit::Zeroed, 1).is_ok());
        assert!(admit_member_call(1.0, ContractDestinationInit::Axpby(0.0), 17).is_ok());
        assert!(admit_member_call(1.0, ContractDestinationInit::Axpby(-0.0), 2).is_ok());
        assert!(unsupported(admit_member_call(
            2.5,
            ContractDestinationInit::Zeroed,
            1
        )));
        assert!(unsupported(admit_member_call(
            1.0,
            ContractDestinationInit::Axpby(1.0),
            1
        )));
        assert!(unsupported(admit_member_call(
            1.0,
            ContractDestinationInit::Axpby(2.0),
            1
        )));
        assert!(matches!(
            admit_member_call(1.0, ContractDestinationInit::Zeroed, 0),
            Err(OperationError::InvalidArgument { .. })
        ));
    }
}
