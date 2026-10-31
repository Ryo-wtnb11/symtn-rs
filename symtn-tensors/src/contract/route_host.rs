//! The one Host executor of every contraction route — Core, CopyC and
//! DynamicTree (#1859).
//!
//! Eager and member replays run the same stage sequence: install the core
//! slot, source transforms, core GEMMs, output transform. An eager call is a
//! one-member replay. The member count `B` matters only at two seams, each a
//! structural identity at `B = 1` (a trailing member axis of extent one
//! addresses the same elements):
//!
//! - the core: `B = 1` replays the plan's own jobs (`execute_raw*`); `B > 1`
//!   replays its member expansion ([`StackedDirectReplay`]);
//! - the transforms: `B = 1` is the ordinary replay; `B > 1` its member
//!   expansion, which only the dense tree backend provides
//!   ([`HostTreeStage`]).
//!
//! Reference step order: TensorKit `blas_contract!`
//! (`tensoroperations.jl:378-447` @cfaa073): copy and twist the sources,
//! `mul!` into the destination or a temporary, `tensoradd!` the temporary
//! into the destination. QSpace `contract` (`QSpace.cc:4141-4263`
//! @dd2cc7e) has no member axis either; `B = 1` is the reference sequence.

use std::ops::Neg;
use std::sync::Arc;
use symtn_dense::Engine;

use num_traits::{One, Zero};
use symtn_core::{BlockStructure, Placement, TensorStorage};
use symtn_dense::DenseExecutor;
use symtn_operations::fusion_replay::FusionBlockContractPlan;

use symtn_operations::stacked::{StackedDirectReplay, StackedStorageView, StackedStorageViewMut};
use symtn_operations::{
    admit_tree_transform_members_overwrite_raw, tree_transform_members_overwrite_raw,
    ContractDestinationInit, DenseTreeTransformOperations, Rank2Gemm, TreeTransformWorkspace,
};

use crate::host_scratch::HostScratchBuffer;
use crate::{
    DenseBlockScalar, DenseRecouplingScalar, OperationError, RecouplingCoefficientAction,
    TreeTransformBackend, TreeTransformStructure,
};

use super::backend::TensorContractWorkspace;
use super::dynamic::{DynamicFusionTransformedSourceEntry, DynamicTreeExecutionArtifact};
use super::fusion_block::{BackendRank2Gemm, FusionBlockContractWorkspace};
use super::resolution::{ContractRoute, StorageContractResolution};

/// One tree-transform stage of a route; each owns its member workspace.
#[derive(Clone, Copy, Debug)]
pub(super) enum Stage {
    Lhs = 0,
    Rhs = 1,
    Output = 2,
}

/// The tree-transform half of a Host route replay.
///
/// Why a trait and not a branch: eager entries are generic over the public
/// [`TreeTransformBackend`], while the member expansion (`B > 1`) needs the
/// dense executor only [`DenseTreeTransformOperations`] exposes. Widening the
/// public trait for one internal caller is not justified.
pub(super) trait HostTreeStage<D, C>
where
    D: DenseRecouplingScalar,
    C: Copy,
{
    type Backend: TreeTransformBackend<D, C>;

    /// The backend and the workspace stage `stage` replays into at `B = 1`.
    fn ordinary(
        &mut self,
        stage: Stage,
    ) -> (
        &mut Self::Backend,
        &mut <Self::Backend as TreeTransformBackend<D, C>>::Workspace,
    );

    /// Admits one `B > 1` overwrite stage before any write.
    #[allow(clippy::too_many_arguments)]
    fn admit_members(
        &mut self,
        stage: Stage,
        structure: &TreeTransformStructure<C>,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst_len: usize,
        src_len: usize,
        members: usize,
    ) -> Result<(), OperationError>;

    /// One `B > 1` overwrite stage over member-major stacks.
    #[allow(clippy::too_many_arguments)]
    fn overwrite_members(
        &mut self,
        engine: &mut dyn Engine,
        stage: Stage,
        structure: &TreeTransformStructure<C>,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst: &mut [D],
        src: &[D],
        members: usize,
        destination_scales: &[(usize, C)],
    ) -> Result<(), OperationError>;

    /// The `B > 1` member expansion of `plan`.
    fn stacked_replay(
        &mut self,
        plan: &Arc<FusionBlockContractPlan<C>>,
        members: usize,
    ) -> Result<StackedDirectReplay<C>, OperationError>;

    /// Replays a `B > 1` core over member-major stacks.
    #[allow(clippy::too_many_arguments)]
    fn stacked_core<G: Rank2Gemm<D>>(
        &mut self,
        engine: &mut dyn Engine,
        replay: &StackedDirectReplay<C>,
        gemm: &mut G,
        dst: &mut [D],
        left: &[D],
        right: &[D],
        zero_inactive: bool,
    ) -> Result<(), OperationError>;
}

/// The destination of one route replay.
pub(super) enum RouteDestination<'d, D> {
    /// `dst = alpha * contract(lhs, rhs) + init(dst)` into an existing
    /// payload.
    Into(&'d mut [D], ContractDestinationInit<D>),
    /// `alpha * contract(lhs, rhs)` into a fresh payload stored in `output`
    /// on success (one member only). `zeroed` allocates the zero payload a
    /// route with an output transform overwrites.
    Owned {
        output: &'d mut Vec<D>,
        zeroed: fn(usize) -> Vec<D>,
    },
}

const EAGER_IS_ONE_MEMBER: OperationError = OperationError::UnsupportedTensorContractScope {
    message: "eager Host contraction replays exactly one member",
};

/// The eager stage: each stage replays into its own workspace, as the member
/// stage does, so a warm call finds each stage's coefficient pack installed.
///
/// Why not one shared workspace: its pack is keyed by one structure identity,
/// so the lhs, rhs and output stages would evict each other every call
/// (#2101). Why not a structure cache: the pack is the completed
/// transformer's matrices converted to the payload scalar `D` in its
/// layout-ordered job order; the composed-coefficient key has no layout, and
/// the transformer entry is typed by the coefficient scalar only.
pub(super) struct EagerTreeStage<'a, B, W> {
    pub(super) backend: &'a mut B,
    pub(super) workspaces: &'a mut [W; 3],
}

impl<D, C, BT> HostTreeStage<D, C> for EagerTreeStage<'_, BT, BT::Workspace>
where
    D: DenseRecouplingScalar,
    C: Copy,
    BT: TreeTransformBackend<D, C>,
{
    type Backend = BT;

    fn ordinary(&mut self, stage: Stage) -> (&mut BT, &mut BT::Workspace) {
        (self.backend, &mut self.workspaces[stage as usize])
    }

    fn admit_members(
        &mut self,
        _: Stage,
        _: &TreeTransformStructure<C>,
        _: &Arc<BlockStructure>,
        _: &Arc<BlockStructure>,
        _: usize,
        _: usize,
        _: usize,
    ) -> Result<(), OperationError> {
        Err(EAGER_IS_ONE_MEMBER)
    }

    fn overwrite_members(
        &mut self,
        _engine: &mut dyn Engine,
        _: Stage,
        _: &TreeTransformStructure<C>,
        _: &Arc<BlockStructure>,
        _: &Arc<BlockStructure>,
        _: &mut [D],
        _: &[D],
        _: usize,
        _: &[(usize, C)],
    ) -> Result<(), OperationError> {
        Err(EAGER_IS_ONE_MEMBER)
    }

    fn stacked_replay(
        &mut self,
        _: &Arc<FusionBlockContractPlan<C>>,
        _: usize,
    ) -> Result<StackedDirectReplay<C>, OperationError> {
        Err(EAGER_IS_ONE_MEMBER)
    }

    fn stacked_core<G: Rank2Gemm<D>>(
        &mut self,
        _: &mut dyn Engine,
        _: &StackedDirectReplay<C>,
        _: &mut G,
        _: &mut [D],
        _: &[D],
        _: &[D],
        _: bool,
    ) -> Result<(), OperationError> {
        Err(EAGER_IS_ONE_MEMBER)
    }
}

/// The member stage: each stage replays into its own workspace at every `B`,
/// so a warm replay finds each stage's coefficient pack installed.
pub(super) struct MemberTreeStage<'a, E, D> {
    backend: &'a mut DenseTreeTransformOperations<E>,
    workspaces: &'a mut [TreeTransformWorkspace<D>; 3],
}

impl<E, D, C> HostTreeStage<D, C> for MemberTreeStage<'_, E, D>
where
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    C: DenseBlockScalar + Neg<Output = C>,
{
    type Backend = DenseTreeTransformOperations<E>;

    fn ordinary(
        &mut self,
        stage: Stage,
    ) -> (
        &mut DenseTreeTransformOperations<E>,
        &mut TreeTransformWorkspace<D>,
    ) {
        (self.backend, &mut self.workspaces[stage as usize])
    }

    fn admit_members(
        &mut self,
        stage: Stage,
        structure: &TreeTransformStructure<C>,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst_len: usize,
        src_len: usize,
        members: usize,
    ) -> Result<(), OperationError> {
        admit_tree_transform_members_overwrite_raw::<D, C>(
            &mut self.workspaces[stage as usize],
            structure,
            dst_structure,
            src_structure,
            dst_len,
            src_len,
            members,
        )
    }

    fn overwrite_members(
        &mut self,
        engine: &mut dyn Engine,
        stage: Stage,
        structure: &TreeTransformStructure<C>,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst: &mut [D],
        src: &[D],
        members: usize,
        destination_scales: &[(usize, C)],
    ) -> Result<(), OperationError> {
        // The engine's own cost rule decides the fan-out of the stacked replay.
        let threads = symtn_operations::host_pool::copy_width(engine, dst.len())?;
        tree_transform_members_overwrite_raw(
            engine,
            &mut crate::StridedHostKernelAdapter::default(),
            self.backend.dense_mut(),
            &mut self.workspaces[stage as usize],
            structure,
            dst_structure,
            src_structure,
            dst,
            src,
            members,
            threads,
            destination_scales,
        )
    }

    fn stacked_replay(
        &mut self,
        plan: &Arc<FusionBlockContractPlan<C>>,
        members: usize,
    ) -> Result<StackedDirectReplay<C>, OperationError> {
        StackedDirectReplay::new_signed(Arc::clone(plan), members)
    }

    fn stacked_core<G: Rank2Gemm<D>>(
        &mut self,
        engine: &mut dyn Engine,
        replay: &StackedDirectReplay<C>,
        gemm: &mut G,
        dst: &mut [D],
        left: &[D],
        right: &[D],
        zero_inactive: bool,
    ) -> Result<(), OperationError> {
        let members = replay.members();
        let [dst_len, left_len, right_len] = replay.member_lens();
        let (left, right) = (ReadSlice(left), ReadSlice(right));
        let left = StackedStorageView::new::<D>(&left, left_len, members, left_len)?;
        let right = StackedStorageView::new::<D>(&right, right_len, members, right_len)?;
        let mut dst = WriteSlice(dst);
        let mut dst = StackedStorageViewMut::new::<D>(&mut dst, dst_len, members, dst_len)?;
        replay.execute_signed_host(
            engine,
            &mut crate::StridedHostKernelAdapter::default(),
            gemm,
            &mut dst,
            &left,
            &right,
            zero_inactive,
        )
    }
}

struct ReadSlice<'a, D>(&'a [D]);
struct WriteSlice<'a, D>(&'a mut [D]);

impl<D> TensorStorage<D> for ReadSlice<'_, D> {
    fn len(&self) -> usize {
        self.0.len()
    }
    fn placement(&self) -> Placement {
        Placement::Host
    }
}
impl<D> symtn_core::HostReadableStorage<D> for ReadSlice<'_, D> {
    fn as_slice(&self) -> &[D] {
        self.0
    }
}
impl<D> TensorStorage<D> for WriteSlice<'_, D> {
    fn len(&self) -> usize {
        self.0.len()
    }
    fn placement(&self) -> Placement {
        Placement::Host
    }
}
impl<D> symtn_core::HostReadableStorage<D> for WriteSlice<'_, D> {
    fn as_slice(&self) -> &[D] {
        self.0
    }
}
impl<D> symtn_core::HostWritableStorage<D> for WriteSlice<'_, D> {
    fn as_mut_slice(&mut self) -> &mut [D] {
        self.0
    }
}

/// The core replay installed in a workspace: the plan itself at `B = 1`,
/// its member expansion at `B > 1`.
enum CoreReplay<C> {
    Empty,
    Plan(Arc<FusionBlockContractPlan<C>>),
    Stacked(StackedDirectReplay<C>),
}

impl<C: Copy + PartialEq + One> CoreReplay<C> {
    fn replays(&self, plan: &Arc<FusionBlockContractPlan<C>>, members: usize) -> bool {
        match self {
            Self::Empty => false,
            Self::Plan(installed) => members == 1 && Arc::ptr_eq(installed, plan),
            Self::Stacked(replay) => {
                replay.members() == members && Arc::ptr_eq(replay.plan(), plan)
            }
        }
    }
}

/// A caller-owned core replay and the proof that its workspace core
/// destination's inactive blocks are zero (#1747).
///
/// `inactive_zero` is cleared whenever a different replay is installed and
/// set only after a successful core replay into the workspace core
/// destination. GEMMs never write inactive blocks, so a later failure keeps
/// it valid. A core writing the caller's destination never consults it.
pub(super) struct CoreSlot<C> {
    replay: CoreReplay<C>,
    inactive_zero: bool,
    #[cfg(test)]
    builds: usize,
    #[cfg(test)]
    inactive_fills: usize,
    /// `B = 1` output transforms that ran the overwrite kernel (`Zeroed`).
    #[cfg(test)]
    output_overwrites: usize,
    /// Test fault point: the next replay dirties the core destination (as a
    /// previous plan's active results would) and fails before the core.
    #[cfg(test)]
    poison_dst_then_fail: bool,
}

impl<C> Default for CoreSlot<C> {
    fn default() -> Self {
        Self {
            replay: CoreReplay::Empty,
            inactive_zero: false,
            #[cfg(test)]
            builds: 0,
            #[cfg(test)]
            inactive_fills: 0,
            #[cfg(test)]
            output_overwrites: 0,
            #[cfg(test)]
            poison_dst_then_fail: false,
        }
    }
}

impl<C: Copy + PartialEq + One> CoreSlot<C> {
    fn install(&mut self, replay: CoreReplay<C>) {
        self.replay = replay;
        self.inactive_zero = false;
        #[cfg(test)]
        {
            self.builds += 1;
        }
    }

    fn retained_bytes(&self) -> usize {
        match &self.replay {
            CoreReplay::Stacked(replay) => replay.retained_bytes(),
            CoreReplay::Empty | CoreReplay::Plan(_) => 0,
        }
    }
}

/// The Host scratch one route replay writes: owned source copies, the core
/// destination of a non-identity output, and — for a caller-owned member
/// workspace — the core slot. Eager scratch carries no slot, so its core
/// destination is zero-filled on every call.
pub(super) struct HostRouteScratch<'a, D, C> {
    pub(super) lhs: &'a mut HostScratchBuffer<D>,
    pub(super) rhs: &'a mut HostScratchBuffer<D>,
    pub(super) core_dst: &'a mut HostScratchBuffer<D>,
    pub(super) core: Option<&'a mut CoreSlot<C>>,
}

/// Exact capacity when cold, then grow-only without clearing: every reader
/// first runs a replay that writes the logical length itself.
pub(super) fn prepare<D: Clone + Zero>(buffer: &mut HostScratchBuffer<D>, len: usize) {
    if buffer.capacity() == 0 {
        *buffer = HostScratchBuffer::filled(len, D::zero());
    } else {
        buffer.resize_filled(len, D::zero());
    }
}

fn member_len<D>(structure: &Arc<BlockStructure>, members: usize) -> Result<usize, OperationError> {
    let len = structure.required_len()?;
    let total = len
        .checked_mul(members)
        .ok_or(OperationError::ElementCountOverflow)?;
    core::alloc::Layout::array::<D>(total).map_err(|_| OperationError::ElementCountOverflow)?;
    Ok(total)
}

fn exact_len(actual: usize, expected: usize) -> Result<(), OperationError> {
    if actual == expected {
        Ok(())
    } else {
        Err(OperationError::ElementCountMismatch { expected, actual })
    }
}

/// `overwrite` one stage: `dst = T(src)`, the move writing destination block
/// `b` scaled by `θ_b` from `destination_scales`.
#[allow(clippy::too_many_arguments)]
fn overwrite_stage<S, D, C>(
    engine: &mut dyn Engine,
    stage: &mut S,
    slot: Stage,
    structure: &TreeTransformStructure<C>,
    dst_structure: &Arc<BlockStructure>,
    src_structure: &Arc<BlockStructure>,
    dst: &mut [D],
    src: &[D],
    members: usize,
    destination_scales: &[(usize, C)],
) -> Result<(), OperationError>
where
    S: HostTreeStage<D, C>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    C: Copy,
{
    if members != 1 {
        return stage.overwrite_members(
            engine,
            slot,
            structure,
            dst_structure,
            src_structure,
            dst,
            src,
            members,
            destination_scales,
        );
    }
    let (backend, workspace) = stage.ordinary(slot);
    backend.tree_transform_structure_overwrite_into_raw(
        engine,
        workspace,
        structure,
        dst_structure,
        src_structure,
        dst,
        src,
        D::one(),
        destination_scales,
    )
}

/// An owned source copy: the transform writing a core operand from the
/// caller's payload (TensorKit `blas_contract!`'s `copyA`/`copyB`).
pub(super) struct SourceStage<'r, C> {
    pub(super) transform: &'r TreeTransformStructure<C>,
    pub(super) payload_structure: &'r Arc<BlockStructure>,
    pub(super) scales: &'r [(usize, C)],
}

/// The stage sequence of one route, read from its resolution: which
/// operands are copied, the core plan and operand order, and the output
/// transform out of a workspace core destination, if any.
pub(super) struct RouteView<'r, C> {
    /// The structures the core GEMMs read for the physical lhs and rhs.
    pub(super) lhs_core: &'r Arc<BlockStructure>,
    pub(super) rhs_core: &'r Arc<BlockStructure>,
    /// `None`: the core reads the caller's payload in place.
    pub(super) lhs_source: Option<SourceStage<'r, C>>,
    pub(super) rhs_source: Option<SourceStage<'r, C>>,
    /// The core's left operand is the physical rhs.
    swapped: bool,
    pub(super) plan: &'r Arc<FusionBlockContractPlan<C>>,
    /// `(output transform, core-destination structure)`; `None` when the
    /// core writes the caller's destination.
    pub(super) output: Option<(&'r TreeTransformStructure<C>, &'r Arc<BlockStructure>)>,
    /// Alpha rides the output transform (CopyC: `mul!` into the temporary,
    /// then `tensoradd!(C, Cnew, α, β)`), else the core GEMMs.
    output_alpha: bool,
}

impl<'r, C: DenseBlockScalar> RouteView<'r, C> {
    fn dynamic_tree(artifact: &'r DynamicTreeExecutionArtifact<C>) -> Self {
        let [lhs_scales, rhs_scales] = artifact.stage_scales();
        let source = |entry: &'r DynamicFusionTransformedSourceEntry<C>,
                      borrowed: bool,
                      scales: &'r [(usize, C)]| {
            // A copied source always has its transformer (checked when the
            // artifact is assembled).
            entry
                .transform_structure
                .as_ref()
                .filter(|_| !borrowed)
                .map(|transform| SourceStage {
                    transform,
                    payload_structure: &entry.replay_structure,
                    scales,
                })
        };
        Self {
            lhs_core: artifact.core_source_structure(true),
            rhs_core: artifact.core_source_structure(false),
            lhs_source: source(&artifact.lhs_transform, artifact.lhs_borrowed, lhs_scales),
            rhs_source: source(&artifact.rhs_transform, artifact.rhs_borrowed, rhs_scales),
            swapped: artifact.core_order(false, true).0,
            plan: &artifact.block_plan,
            output: artifact
                .core_dst
                .as_ref()
                .map(|entry| (&entry.output_transform_structure, entry.space.structure())),
            output_alpha: false,
        }
    }

    pub(super) fn of(
        route: &'r ContractRoute<C>,
        lhs: &'r Arc<BlockStructure>,
        rhs: &'r Arc<BlockStructure>,
    ) -> Self {
        let direct = |plan, swapped, output| Self {
            lhs_core: lhs,
            rhs_core: rhs,
            lhs_source: None,
            rhs_source: None,
            swapped,
            plan,
            output,
            // CopyC (an output transform) carries alpha in its `tensoradd!`.
            output_alpha: output.is_some(),
        };
        match route {
            ContractRoute::Core { plan, swapped } => direct(plan, *swapped, None),
            ContractRoute::CopyC(copy) => direct(
                &copy.core,
                copy.swapped,
                Some((&copy.transform, &copy.temporary)),
            ),
            ContractRoute::DynamicTree(artifact) => Self::dynamic_tree(artifact),
        }
    }

    #[inline]
    pub(super) fn core_order<T>(&self, lhs: T, rhs: T) -> (T, T) {
        if self.swapped {
            (rhs, lhs)
        } else {
            (lhs, rhs)
        }
    }
}

/// Replays one route over `members` member-major stacks:
/// `dst = alpha * contract(lhs, rhs) + init(dst)` per member, as TensorKit
/// `blas_contract!` orders it — copy (and twist) the sources, `mul!` into
/// the destination or a temporary, `tensoradd!` the temporary into the
/// destination.
///
/// `B > 1` admits only the member overwrite contract (`alpha = 1`,
/// `Axpby(0)` or `Zeroed`) with a caller-owned core slot, and checks every structure,
/// length and transform before the first write. At `B = 1` each stage
/// validates its inputs before its first write and only the last stage writes
/// `dst`, so a validation error leaves `dst` unchanged at every `B`.
///
/// Beta rides the stage that writes `dst`: the core for Core and an identity
/// output, else the output transform. Alpha rides the core GEMMs, except
/// for CopyC, where it rides the output transform.
#[allow(clippy::too_many_arguments)]
fn execute_route_view<S, G, D, C>(
    engine: &mut dyn Engine,
    stage: &mut S,
    gemm: &mut G,
    fusion_block_workspace: &mut FusionBlockContractWorkspace<D>,
    scratch: HostRouteScratch<'_, D, C>,
    route: RouteView<'_, C>,
    (dst_structure, destination): (&Arc<BlockStructure>, RouteDestination<'_, D>),
    lhs_data: &[D],
    rhs_data: &[D],
    members: usize,
    alpha: D,
) -> Result<(), OperationError>
where
    S: HostTreeStage<D, C>,
    G: Rank2Gemm<D>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    C: DenseBlockScalar,
{
    let (mut dst_data, mut owned, init) = match destination {
        RouteDestination::Into(data, init) => (Some(data), None, init),
        RouteDestination::Owned { output, zeroed } => {
            if members != 1 {
                return Err(EAGER_IS_ONE_MEMBER);
            }
            (
                None,
                Some((output, zeroed)),
                ContractDestinationInit::Zeroed,
            )
        }
    };
    let HostRouteScratch {
        lhs: lhs_scratch,
        rhs: rhs_scratch,
        core_dst,
        mut core,
    } = scratch;
    let (lhs_core, rhs_core) = (route.lhs_core, route.rhs_core);
    let core_dst_structure = route
        .output
        .map_or(dst_structure, |(_, structure)| structure);
    let plan = route.plan;
    let lhs_core_len = member_len::<D>(lhs_core, members)?;
    let rhs_core_len = member_len::<D>(rhs_core, members)?;
    let core_dst_len = member_len::<D>(core_dst_structure, members)?;
    let (core_alpha, output_alpha) = if route.output_alpha {
        (D::one(), alpha)
    } else {
        (alpha, D::one())
    };

    let mut stacked = None;
    if members != 1 {
        let overwrite = match init {
            ContractDestinationInit::Zeroed => true,
            ContractDestinationInit::Axpby(beta) => beta.is_zero(),
        };
        if alpha != D::one() || !overwrite {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "member contraction overwrites its destination with unit alpha",
            });
        }
        let slot = core
            .as_deref_mut()
            .ok_or(OperationError::UnsupportedTensorContractScope {
                message: "member contraction needs a caller-owned core slot",
            })?;
        let dst_len = member_len::<D>(dst_structure, members)?;
        for (source, core, data) in [
            (&route.lhs_source, lhs_core, lhs_data),
            (&route.rhs_source, rhs_core, rhs_data),
        ] {
            let payload = source
                .as_ref()
                .map_or(core, |source| source.payload_structure);
            exact_len(data.len(), member_len::<D>(payload, members)?)?;
        }
        exact_len(dst_data.as_deref().map_or(0, <[D]>::len), dst_len)?;
        if !slot.replay.replays(plan, members) {
            stacked = Some(stage.stacked_replay(plan, members)?);
        }
        let replay = match (&stacked, &slot.replay) {
            (Some(replay), _) | (None, CoreReplay::Stacked(replay)) => replay,
            (None, _) => unreachable!("a matching B > 1 replay is stacked"),
        };
        let (left_structure, right_structure) = route.core_order(lhs_core, rhs_core);
        replay.plan().validate_replay_structures(
            core_dst_structure,
            left_structure,
            right_structure,
        )?;
        let (left_len, right_len) = route.core_order(lhs_core_len, rhs_core_len);
        if replay.member_lens().map(|len| len * members) != [core_dst_len, left_len, right_len] {
            return Err(OperationError::InvalidArgument {
                message: "member route core structure does not match its replay",
            });
        }
        for (slot, source, core, core_len, data_len) in [
            (
                Stage::Lhs,
                &route.lhs_source,
                lhs_core,
                lhs_core_len,
                lhs_data.len(),
            ),
            (
                Stage::Rhs,
                &route.rhs_source,
                rhs_core,
                rhs_core_len,
                rhs_data.len(),
            ),
        ] {
            if let Some(source) = source {
                stage.admit_members(
                    slot,
                    source.transform,
                    core,
                    source.payload_structure,
                    core_len,
                    data_len,
                    members,
                )?;
            }
        }
        if let Some((transform, _)) = route.output {
            stage.admit_members(
                Stage::Output,
                transform,
                dst_structure,
                core_dst_structure,
                dst_len,
                core_dst_len,
                members,
            )?;
        }
    }

    // A different replay is installed before any write and clears the proof.
    if let Some(slot) = core.as_deref_mut() {
        if let Some(replay) = stacked {
            slot.install(CoreReplay::Stacked(replay));
        } else if !slot.replay.replays(plan, members) {
            slot.install(CoreReplay::Plan(Arc::clone(plan)));
        }
    }

    if route.lhs_source.is_some() {
        prepare(lhs_scratch, lhs_core_len);
    }
    if route.rhs_source.is_some() {
        prepare(rhs_scratch, rhs_core_len);
    }
    // Workspace core destination: refill its inactive blocks unless the slot
    // proves them zero for this replay and length.
    let core_init = if route.output.is_none() {
        init
    } else {
        let proved = core
            .as_deref()
            .is_some_and(|slot| slot.inactive_zero && core_dst.len() == core_dst_len);
        prepare(core_dst, core_dst_len);
        if proved {
            ContractDestinationInit::Zeroed
        } else {
            ContractDestinationInit::Axpby(D::zero())
        }
    };
    #[cfg(test)]
    if let Some(slot) = core.as_deref_mut() {
        if std::mem::take(&mut slot.poison_dst_then_fail) {
            core_dst.fill(D::zero() + D::one());
            return Err(OperationError::InvalidArgument {
                message: "injected fault before the core replay",
            });
        }
    }

    if let Some(source) = &route.lhs_source {
        overwrite_stage(
            engine,
            stage,
            Stage::Lhs,
            source.transform,
            lhs_core,
            source.payload_structure,
            lhs_scratch.as_mut_slice(),
            lhs_data,
            members,
            source.scales,
        )?;
    }
    if let Some(source) = &route.rhs_source {
        overwrite_stage(
            engine,
            stage,
            Stage::Rhs,
            source.transform,
            rhs_core,
            source.payload_structure,
            rhs_scratch.as_mut_slice(),
            rhs_data,
            members,
            source.scales,
        )?;
    }

    let physical_lhs = if route.lhs_source.is_some() {
        lhs_scratch.as_slice()
    } else {
        lhs_data
    };
    let physical_rhs = if route.rhs_source.is_some() {
        rhs_scratch.as_slice()
    } else {
        rhs_data
    };
    let (left, right) = route.core_order(physical_lhs, physical_rhs);
    let (left_structure, right_structure) = route.core_order(lhs_core, rhs_core);
    // `None`: the core writes a fresh owned output.
    let core_out = if route.output.is_some() {
        Some(core_dst.as_mut_slice())
    } else {
        dst_data.as_deref_mut()
    };
    #[cfg(test)]
    if let Some(slot) = core.as_deref_mut() {
        slot.inactive_fills += usize::from(matches!(core_init, ContractDestinationInit::Axpby(_)));
    }
    if members == 1 {
        let mut kernels = crate::StridedHostKernelAdapter::default();
        let Some(core_out) = core_out else {
            let (output, _) = owned.as_mut().ok_or(OperationError::InvalidArgument {
                message: "contraction route has no destination",
            })?;
            #[cfg(test)]
            crate::probe::hit(crate::probe::Event::OwnedCoreReplay);
            **output = plan.execute_owned(
                engine,
                &mut kernels,
                gemm,
                fusion_block_workspace,
                core_dst_structure,
                left_structure,
                left,
                right_structure,
                right,
                core_alpha,
            )?;
            return Ok(());
        };
        match core_init {
            ContractDestinationInit::Zeroed => plan.execute_raw_zeroed(
                engine,
                &mut kernels,
                gemm,
                fusion_block_workspace,
                core_dst_structure,
                core_out,
                left_structure,
                left,
                right_structure,
                right,
                core_alpha,
            ),
            ContractDestinationInit::Axpby(beta) => plan.execute_raw(
                engine,
                &mut kernels,
                gemm,
                fusion_block_workspace,
                core_dst_structure,
                core_out,
                left_structure,
                left,
                right_structure,
                right,
                core_alpha,
                beta,
            ),
        }?;
    } else {
        let Some(CoreReplay::Stacked(replay)) = core.as_deref().map(|slot| &slot.replay) else {
            unreachable!("B > 1 installed its stacked replay");
        };
        let Some(core_out) = core_out else {
            return Err(EAGER_IS_ONE_MEMBER);
        };
        stage.stacked_core(
            engine,
            replay,
            gemm,
            core_out,
            left,
            right,
            matches!(core_init, ContractDestinationInit::Axpby(_)),
        )?;
    }

    let Some((transform, _)) = route.output else {
        return Ok(());
    };
    if let Some(slot) = core {
        slot.inactive_zero = true;
        #[cfg(test)]
        {
            slot.output_overwrites +=
                usize::from(members == 1 && matches!(init, ContractDestinationInit::Zeroed));
        }
    }
    // L2 keeps the zeroed output plus overwrite transform for an owned
    // output-transform route; the owned transform writer is a separate step.
    let mut zeroed_output = None;
    let dst_data = match (dst_data.take(), owned.as_ref()) {
        (Some(data), _) => data,
        (None, Some((_, zeroed))) => {
            zeroed_output.insert(zeroed(member_len::<D>(dst_structure, 1)?))
        }
        (None, None) => {
            return Err(OperationError::InvalidArgument {
                message: "contraction route has no destination",
            })
        }
    };
    if members != 1 {
        return stage.overwrite_members(
            engine,
            Stage::Output,
            transform,
            dst_structure,
            core_dst_structure,
            dst_data,
            core_dst.as_slice(),
            members,
            &[],
        );
    }
    let (backend, workspace) = stage.ordinary(Stage::Output);
    let source = core_dst.as_slice();
    let written = match init {
        ContractDestinationInit::Zeroed => backend.tree_transform_structure_overwrite_into_raw(
            engine,
            workspace,
            transform,
            dst_structure,
            core_dst_structure,
            dst_data,
            source,
            output_alpha,
            &[],
        ),
        ContractDestinationInit::Axpby(beta) => backend.tree_transform_structure_into_raw(
            engine,
            workspace,
            transform,
            dst_structure,
            core_dst_structure,
            dst_data,
            source,
            output_alpha,
            beta,
        ),
    };
    written?;
    if let (Some((output, _)), Some(data)) = (owned, zeroed_output) {
        *output = data;
    }
    Ok(())
}

/// Replays `resolution` through the one stage sequence of every route; the
/// physical operands' structures are read only by a route whose core reads
/// the caller's payload in place.
#[allow(clippy::too_many_arguments)]
pub(super) fn execute_route_host<S, G, D, C>(
    engine: &mut dyn Engine,
    stage: &mut S,
    gemm: &mut G,
    fusion_block_workspace: &mut FusionBlockContractWorkspace<D>,
    scratch: HostRouteScratch<'_, D, C>,
    resolution: &StorageContractResolution<C>,
    dst: (&Arc<BlockStructure>, RouteDestination<'_, D>),
    (lhs_structure, lhs_data): (&Arc<BlockStructure>, &[D]),
    (rhs_structure, rhs_data): (&Arc<BlockStructure>, &[D]),
    members: usize,
    alpha: D,
) -> Result<(), OperationError>
where
    S: HostTreeStage<D, C>,
    G: Rank2Gemm<D>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    C: DenseBlockScalar,
{
    execute_route_view(
        engine,
        stage,
        gemm,
        fusion_block_workspace,
        scratch,
        RouteView::of(&resolution.route, lhs_structure, rhs_structure),
        dst,
        lhs_data,
        rhs_data,
        members,
        alpha,
    )
}

/// [`execute_route_host`] of one `DynamicTree` artifact, whose sources are
/// read through its own core structures.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(super) fn execute_dynamic_tree_route_host<S, G, D, C>(
    engine: &mut dyn Engine,
    stage: &mut S,
    gemm: &mut G,
    fusion_block_workspace: &mut FusionBlockContractWorkspace<D>,
    scratch: HostRouteScratch<'_, D, C>,
    artifact: &DynamicTreeExecutionArtifact<C>,
    dst: (&Arc<BlockStructure>, &mut [D]),
    lhs_data: &[D],
    rhs_data: &[D],
    members: usize,
    alpha: D,
    init: ContractDestinationInit<D>,
) -> Result<(), OperationError>
where
    S: HostTreeStage<D, C>,
    G: Rank2Gemm<D>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    C: DenseBlockScalar,
{
    execute_route_view(
        engine,
        stage,
        gemm,
        fusion_block_workspace,
        scratch,
        RouteView::dynamic_tree(artifact),
        (dst.0, RouteDestination::Into(dst.1, init)),
        lhs_data,
        rhs_data,
        members,
        alpha,
    )
}

/// Caller-owned Host payload and replay state for the member replays of one
/// contraction route at varying `B`: owned source copies, the workspace core
/// destination (a DynamicTree core destination or the CopyC temporary), the
/// core slot, and one transform workspace per stage. Dense GEMM plans are not
/// workspace state: the block GEMM kernel keeps them in the process-global
/// geometry-plan cache, shared with eager calls.
#[doc(hidden)]
pub struct HostContractMembersWorkspace<D, C = f64> {
    lhs: HostScratchBuffer<D>,
    rhs: HostScratchBuffer<D>,
    core_dst: HostScratchBuffer<D>,
    core: CoreSlot<C>,
    /// Per-stage workspaces, indexed by [`Stage`].
    transforms: [TreeTransformWorkspace<D>; 3],
}

impl<D, C> Default for HostContractMembersWorkspace<D, C> {
    fn default() -> Self {
        Self {
            lhs: HostScratchBuffer::default(),
            rhs: HostScratchBuffer::default(),
            core_dst: HostScratchBuffer::default(),
            core: CoreSlot::default(),
            transforms: Default::default(),
        }
    }
}

impl<D, C: Copy + PartialEq + One> HostContractMembersWorkspace<D, C> {
    /// Owned scratch and replay state: payload scratch, the core slot's
    /// member expansion and the transform workspaces. Saturates at
    /// `usize::MAX`.
    pub fn retained_bytes(&self) -> usize {
        (self.lhs.capacity() + self.rhs.capacity() + self.core_dst.capacity())
            .saturating_mul(std::mem::size_of::<D>())
            .saturating_add(self.core.retained_bytes())
            .saturating_add(
                self.transforms
                    .iter()
                    .map(TreeTransformWorkspace::retained_bytes)
                    .fold(0, usize::saturating_add),
            )
    }

    #[cfg(test)]
    pub(crate) fn core_replay_builds(&self) -> usize {
        self.core.builds
    }

    #[cfg(test)]
    pub(crate) fn core_inactive_fills(&self) -> usize {
        self.core.inactive_fills
    }

    #[cfg(test)]
    pub(crate) fn output_overwrites(&self) -> usize {
        self.core.output_overwrites
    }

    #[cfg(test)]
    pub(crate) fn poison_dst_then_fail(&mut self) {
        self.core.poison_dst_then_fail = true;
    }

    #[cfg(test)]
    pub(crate) fn coefficient_pack_builds(&self) -> [usize; 3] {
        std::array::from_fn(|stage| self.transforms[stage].coefficient_pack_builds())
    }
}

/// The member route view: the resolved route, or a test's lone artifact.
pub(super) enum MemberRoute<'r, C> {
    Resolution {
        resolution: &'r StorageContractResolution<C>,
        lhs: &'r Arc<BlockStructure>,
        rhs: &'r Arc<BlockStructure>,
    },
    #[cfg(test)]
    DynamicTree(&'r DynamicTreeExecutionArtifact<C>),
}

/// Replays one route over uniform member-major Host stacks through the
/// route executor, with this workspace's slot and per-stage transform
/// workspaces: `dst = contract(lhs, rhs)` per member. `init` is `Zeroed`
/// only for a destination whose inactive blocks are already zero, else
/// `Axpby(0)`.
#[allow(clippy::too_many_arguments)]
pub(super) fn execute_members_host<E, EC, D, C>(
    engine: &mut dyn Engine,
    tree_backend: &mut DenseTreeTransformOperations<E>,
    contract_backend: &mut DenseTreeTransformOperations<EC>,
    contract_workspace: &mut TensorContractWorkspace<D>,
    fusion_block_workspace: &mut FusionBlockContractWorkspace<D>,
    route: MemberRoute<'_, C>,
    dst_structure: &Arc<BlockStructure>,
    workspace: &mut HostContractMembersWorkspace<D, C>,
    dst: &mut [D],
    lhs: &[D],
    rhs: &[D],
    members: usize,
    init: ContractDestinationInit<D>,
) -> Result<(), OperationError>
where
    E: DenseExecutor,
    EC: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    C: DenseBlockScalar + Neg<Output = C>,
{
    let HostContractMembersWorkspace {
        lhs: lhs_scratch,
        rhs: rhs_scratch,
        core_dst,
        core,
        transforms,
    } = workspace;
    let route = match route {
        MemberRoute::Resolution {
            resolution,
            lhs,
            rhs,
        } => RouteView::of(&resolution.route, lhs, rhs),
        #[cfg(test)]
        MemberRoute::DynamicTree(artifact) => RouteView::dynamic_tree(artifact),
    };
    execute_route_view(
        engine,
        &mut MemberTreeStage {
            backend: tree_backend,
            workspaces: transforms,
        },
        &mut BackendRank2Gemm::<_, _, C>::new(contract_backend, contract_workspace),
        fusion_block_workspace,
        HostRouteScratch {
            lhs: lhs_scratch,
            rhs: rhs_scratch,
            core_dst,
            core: Some(core),
        },
        route,
        (dst_structure, RouteDestination::Into(dst, init)),
        lhs,
        rhs,
        members,
        D::one(),
    )
}

/// The member replay of one DynamicTree artifact (a test seam).
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_dynamic_tree_execution_artifact_members_host<E, EC, D, C>(
    engine: &mut dyn Engine,
    tree_backend: &mut DenseTreeTransformOperations<E>,
    contract_backend: &mut DenseTreeTransformOperations<EC>,
    contract_workspace: &mut TensorContractWorkspace<D>,
    fusion_block_workspace: &mut FusionBlockContractWorkspace<D>,
    artifact: &DynamicTreeExecutionArtifact<C>,
    dst_structure: &Arc<BlockStructure>,
    workspace: &mut HostContractMembersWorkspace<D, C>,
    dst: &mut [D],
    lhs: &[D],
    rhs: &[D],
    members: usize,
) -> Result<(), OperationError>
where
    E: DenseExecutor,
    EC: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    C: DenseBlockScalar + Neg<Output = C>,
{
    execute_members_host(
        engine,
        tree_backend,
        contract_backend,
        contract_workspace,
        fusion_block_workspace,
        MemberRoute::DynamicTree(artifact),
        dst_structure,
        workspace,
        dst,
        lhs,
        rhs,
        members,
        ContractDestinationInit::Axpby(D::zero()),
    )
}
