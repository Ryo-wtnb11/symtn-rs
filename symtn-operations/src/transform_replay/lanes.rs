//! Tree replays on the lane runner ([`symtn_dense::lanes`]).
//!
//! Every joint component of the compiled schedule, in every member of a
//! member replay, is one job: it owns its atoms of its member's destination
//! for the whole call, so each destination element has one writer. Job
//! `member * components + component` is that member's component; an
//! ordinary replay is the one-member case, whose jobs are its components.
//! (One worker orders a member batch differently: plan job outer, member
//! inner.) Its stages are `Direct` (its inactive layouts and Singles, in
//! destination address order), then for each of its Multi groups in
//! recoupling-plan order `Pack`, `Gemm` and `Scatter`. Placement prices each stage by its provider
//! estimate (Strided's copy estimate of the elements a movement stage moves,
//! dense's own GEMM estimate) and runs it at its selected width `x`:
//!
//! - a movement stage on `b = min(x, units)` band sub-slots, one flat fork of
//!   `b` over the component's band-to-sub-slot map ([`TreeWork::band`]); at
//!   `b = 1` every item is written whole on the `x`-slot token, so a planned
//!   role is offered `x` slots and Strided's own width rule decides inside;
//!   a component whose one item is an overwritten Single whose plan Strided
//!   admits on the stage's token is never banded;
//! - a pack on one flat fork over `min(x, columns)` contiguous column ranges;
//! - a GEMM into the lane's initialized packed destination: with an executor
//!   that runs lane GEMMs, on the stage's `x`-slot context inside the lanes
//!   ([`symtn_dense::lanes::LaneGemms::run`], [`lane_replay`]); otherwise in
//!   the runner's barrier form ([`barrier_replay`]), handed to the engine
//!   between rounds as one `matmul_axpby_into` on the whole engine.
//!
//! The bands and pack ranges first cap the selected width; under `Auto` the
//! actual-fork pays check may then choose inline execution; a forced policy
//! keeps its parts.
//!
//! Each lane keeps grow-only scratch (a packed source buffer, an initialized
//! packed destination, traversal indices and the band tables), retained in
//! the replay workspace with the runner's tables; both forms use the same.
//! The caller only reserves the packed buffers' capacity before the fork;
//! each pack overwrites its source uninitialized, and each job zero-fills
//! the destination beyond its initialized length on its own lane, so the
//! fill runs in parallel and a retained workspace fills nothing once grown.

use std::sync::{PoisonError, RwLock};

use strided_kernel::execution::KernelClass;
use symtn_dense::lanes::{
    self, Estimate, Form, Jobs, LaneGemmShape, LaneGemms, LaneTables, StageKind, Work,
};
use symtn_dense::{DenseDType, DenseScalar, DenseView, DenseViewMut};

use super::writers::{Band, DestinationWriter, ReplayParts};
use super::*;
use crate::host_scratch::PackColumnWriter;
use crate::kernel_adapter::movement::with_conjugation;
use crate::transform_structure::TreeTransformPackReplay;

mod pieces;

use pieces::{PackCell, Piece, SubCell};

/// One lane's scratch, kept between calls; the band and pack tables are
/// empty between stages and keep only their capacity.
///
/// Why `RwLock` cells for a fork's sub-slots: `fork` shares one `Fn` with
/// every child, so each child takes its own `&mut` share from a cell; a std
/// `RwLock` locks without allocating, where a fresh std `Mutex` allocates on
/// its first lock on Apple targets.
pub(super) struct TreeLane<D> {
    source: HostScratchBuffer<D>,
    /// Initialized on the lane and only where it grows; every GEMM overwrites
    /// what it uses.
    ///
    /// Why initialized although every GEMM writes it with `beta = 0` before
    /// any read: the GEMM APIs take the output as an initialized `&mut [D]`,
    /// which safe Rust cannot form over spare capacity.
    destination: HostScratchBuffer<D>,
    index: Vec<usize>,
    pieces: Vec<Piece<'static, u8>>,
    subs: Vec<SubCell<'static, 'static, u8>>,
    packs: Vec<PackCell<'static, u8>>,
}

impl<D> Default for TreeLane<D> {
    fn default() -> Self {
        Self {
            source: HostScratchBuffer::default(),
            destination: HostScratchBuffer::default(),
            index: Vec::new(),
            pieces: Vec::new(),
            subs: Vec::new(),
            packs: Vec::new(),
        }
    }
}

impl<D> TreeLane<D> {
    fn retained_bytes(&self) -> usize {
        use core::mem::size_of;
        [
            self.source.capacity().saturating_mul(size_of::<D>()),
            self.destination.capacity().saturating_mul(size_of::<D>()),
            self.index.capacity().saturating_mul(size_of::<usize>()),
            self.pieces
                .capacity()
                .saturating_mul(size_of::<Piece<'static, u8>>()),
            self.subs
                .capacity()
                .saturating_mul(size_of::<SubCell<'static, 'static, u8>>()),
            self.packs
                .capacity()
                .saturating_mul(size_of::<PackCell<'static, u8>>()),
        ]
        .into_iter()
        .fold(0, usize::saturating_add)
    }
}

/// The lane replay's state retained by a workspace: the GEMM shapes, the
/// runner's tables, every lane's scratch, and the (empty) capacities of the
/// atom-slice and component tables.
pub(crate) struct TreeLaneScratch<D> {
    shapes: Vec<LaneGemmShape>,
    tables: LaneTables,
    lanes: Vec<TreeLane<D>>,
    atoms: Vec<&'static mut [u8]>,
    components: Vec<&'static mut [&'static mut [u8]]>,
}

impl<D> Default for TreeLaneScratch<D> {
    fn default() -> Self {
        Self {
            shapes: Vec::new(),
            tables: LaneTables::default(),
            lanes: Vec::new(),
            atoms: Vec::new(),
            components: Vec::new(),
        }
    }
}

/// Scratch only: a clone starts empty.
impl<D> Clone for TreeLaneScratch<D> {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl<D> core::fmt::Debug for TreeLaneScratch<D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TreeLaneScratch")
            .field("lanes", &self.lanes.len())
            .field("retained_bytes", &self.retained_bytes())
            .finish()
    }
}

impl<D> TreeLaneScratch<D> {
    pub(crate) fn retained_bytes(&self) -> usize {
        use core::mem::size_of;
        let lanes = self.lanes.iter().fold(0usize, |sum, lane| {
            sum.saturating_add(lane.retained_bytes())
        });
        [
            self.shapes
                .capacity()
                .saturating_mul(size_of::<LaneGemmShape>()),
            self.tables.retained_bytes(),
            self.lanes
                .capacity()
                .saturating_mul(size_of::<TreeLane<D>>()),
            lanes,
            self.atoms.capacity().saturating_mul(size_of::<&mut [u8]>()),
            self.components
                .capacity()
                .saturating_mul(size_of::<&mut [&mut [u8]]>()),
        ]
        .into_iter()
        .fold(0, usize::saturating_add)
    }

    /// The last call's placement (read by the lane tests).
    #[cfg(test)]
    pub(crate) fn placement(&self) -> &lanes::Placement {
        self.tables.placement()
    }
}

/// `vec`'s allocation, emptied, for elements of another lifetime or type of
/// the same layout.
///
/// Why not a transmute: replay forbids unsafe code. std collects a mapped
/// `vec::IntoIter` in place when the layouts match; that is an optimization,
/// not a documented guarantee, so the warm replay allocation test pins it (a
/// regression would reallocate, never misbehave).
fn recycle<A, B>(mut vec: Vec<A>) -> Vec<B> {
    vec.clear();
    vec.into_iter().map(|_| unreachable!("emptied")).collect()
}

/// The dense dtype of `D`'s GEMMs.
fn dense_dtype<D: DenseRecouplingScalar>() -> DenseDType {
    match D::zero().dense_scalar() {
        DenseScalar::F32(_) => DenseDType::F32,
        DenseScalar::F64(_) => DenseDType::F64,
        DenseScalar::C32(_) => DenseDType::C32,
        DenseScalar::C64(_) => DenseDType::C64,
    }
}

/// Replays `task` into `dst` through `writer` on the lane runner, inside one
/// [`DenseExecutor::with_lane_gemms`] entry of `dense` (see the module
/// documentation). Returns `false`, having written nothing, when placement
/// resolves to one lane of one slot: that is one worker, which keeps the
/// source order of the one-worker replay.
///
/// Errors follow the runner: every job is attempted, a job stops at its own
/// first failing stage, and the lowest-indexed failing job's error is
/// returned; an initialized destination may then be partly written. A
/// refused entry writes nothing.
#[allow(clippy::too_many_arguments)]
pub(super) fn lane_replay<E, W, T, D, C>(
    engine: &mut dyn Engine,
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    task: TreeTransformTaskView<'_, C>,
    schedule: &TreeTransformParallelSchedule,
    dst: &mut [T],
    src: &[D],
    writer: &W,
    profile: Option<&mut TreeTransformReplayProfile>,
    members: usize,
) -> Result<bool, OperationError>
where
    E: DenseExecutor,
    W: DestinationWriter<T, D>,
    T: Send + Sync,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    let start = profile.as_ref().map(|_| std::time::Instant::now());
    let HostTreeTransformWorkspace {
        lanes: scratch,
        coefficient_scratch,
        ..
    } = workspace;
    let TreeLaneScratch {
        shapes,
        tables,
        lanes,
        atoms,
        components,
    } = scratch;
    shapes.clear();
    shapes.extend(task.recoupling_plan().jobs().iter().map(lane_gemm_shape));
    let mut target = Some(dst);
    let mut outcome = None;
    let entered = dense.with_lane_gemms(engine, dense_dtype::<D>(), shapes, &mut |exec, gemms| {
        let Some(dst) = target.take() else {
            return;
        };
        let replay = Replay::new(
            task,
            schedule,
            Some(gemms),
            writer,
            src,
            coefficient_scratch,
            members,
        );
        outcome = Some(replay.run(
            Runner::<E, D>::Native(exec),
            tables,
            lanes,
            atoms,
            components,
            dst,
        ));
    });
    entered.map_err(OperationError::Dense)?;
    let ran = outcome.unwrap_or(Err(OperationError::InvalidArgument {
        message: "the lane GEMM entry returned without running its body",
    }))?;
    if let (true, Some(profile), Some(start)) = (ran, profile, start) {
        record_lanes(profile, task, schedule, members, start.elapsed());
    }
    Ok(ran)
}

/// [`lane_replay`] in the barrier form of the runner, for an executor
/// without lane GEMMs (the BLAS provider, custom executors): each round is
/// one Host region of the engine in which every lane runs its components'
/// stages up to its next GEMM; between rounds each pending GEMM is
/// submitted alone, in lane order, through `dense`'s
/// [`DenseExecutor::matmul_axpby_into`] (`alpha = 1`, `beta = 0`) over its
/// lane's packed source and initialized packed destination, on the whole
/// engine. Placement, stages, bands, lane scratch and errors are those of
/// [`lane_replay`]; a failed submission fails its own component only and is
/// never retried. Returns `false`, having written nothing, when placement
/// resolves to one lane of one slot.
///
/// Why one submission per GEMM, not one batch per round: each result then
/// belongs to exactly one component, and no batch operand spans several
/// lanes' buffers.
#[allow(clippy::too_many_arguments)]
pub(super) fn barrier_replay<E, W, T, D, C>(
    engine: &mut dyn Engine,
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    task: TreeTransformTaskView<'_, C>,
    schedule: &TreeTransformParallelSchedule,
    dst: &mut [T],
    src: &[D],
    writer: &W,
    profile: Option<&mut TreeTransformReplayProfile>,
    members: usize,
) -> Result<bool, OperationError>
where
    E: DenseExecutor,
    W: DestinationWriter<T, D>,
    T: Send + Sync,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    let start = profile.as_ref().map(|_| std::time::Instant::now());
    let HostTreeTransformWorkspace {
        lanes: scratch,
        coefficient_scratch,
        ..
    } = workspace;
    let TreeLaneScratch {
        tables,
        lanes,
        atoms,
        components,
        ..
    } = scratch;
    let replay = Replay::new(
        task,
        schedule,
        None,
        writer,
        src,
        coefficient_scratch,
        members,
    );
    let mut host = EngineGemms {
        engine,
        dense,
        schedule,
        jobs: task.recoupling_plan().jobs(),
        coefficients: coefficient_scratch,
        matmul: profile.as_ref().map(|_| core::time::Duration::ZERO),
    };
    let ran = replay.run(
        Runner::Barrier(&mut host),
        tables,
        lanes,
        atoms,
        components,
        dst,
    )?;
    if let (true, Some(profile), Some(start)) = (ran, profile, start) {
        record_lanes(profile, task, schedule, members, start.elapsed());
        let matmul = host.matmul.unwrap_or_default();
        profile.multi_dense_matmul_call += matmul;
        profile.multi_matmul_total += matmul;
    }
    Ok(ran)
}

/// A lane run's profile: its whole wall as `fused_lanes`, and its blocks and
/// columns.
fn record_lanes<C: Copy>(
    profile: &mut TreeTransformReplayProfile,
    task: TreeTransformTaskView<'_, C>,
    schedule: &TreeTransformParallelSchedule,
    members: usize,
    elapsed: core::time::Duration,
) {
    profile.fused_lanes += elapsed;
    profile.single_blocks += members * schedule.single_block_count;
    profile.multi_blocks += members * task.recoupling_plan().jobs().len();
    profile.packed_columns += members * schedule.packed_column_count;
    profile.scattered_columns += members * schedule.scattered_column_count;
}

fn lane_gemm_shape(job: &DenseGemmBatchJob) -> LaneGemmShape {
    LaneGemmShape {
        rows: job.rows,
        k: job.contracted,
        n: job.cols,
    }
}

/// Which runner a lane replay runs on: the native one on the lane-GEMM
/// entry's context, or the barrier form on `E`'s engine.
enum Runner<'x, 'e, E, D> {
    Native(lanes::Exec<'x>),
    Barrier(&'x mut EngineGemms<'e, E, D>),
}

/// The barrier form's engine side: the caller's engine and executor, and the
/// recoupling GEMMs it submits between rounds.
struct EngineGemms<'e, E, D> {
    engine: &'e mut dyn Engine,
    dense: &'e mut E,
    schedule: &'e TreeTransformParallelSchedule,
    jobs: &'e [DenseGemmBatchJob],
    coefficients: &'e [D],
    /// The summed wall of the submissions, when profiling.
    matmul: Option<core::time::Duration>,
}

/// The read-only parts of one lane replay.
struct Replay<'a, W, D, C> {
    parts: ReplayParts<'a, C>,
    schedule: &'a TreeTransformParallelSchedule,
    /// The lane GEMMs of a native run; `None` in the barrier form, whose
    /// GEMMs the engine runs between rounds.
    gemms: Option<&'a dyn LaneGemms>,
    writer: &'a W,
    src: &'a [D],
    coefficients: &'a [D],
    rank: usize,
    /// The members replayed, each `member_src` source elements long.
    members: usize,
    member_src: usize,
}

impl<'a, W, D, C> Replay<'a, W, D, C> {
    /// The member and component of job `job`.
    fn job(&self, job: usize) -> (usize, usize) {
        let components = self.schedule.component_count();
        (job / components, job % components)
    }

    /// Member `member`'s source.
    fn source(&self, member: usize) -> &'a [D] {
        &self.src[member * self.member_src..][..self.member_src]
    }
}

impl<'a, W, D, C> Replay<'a, W, D, C>
where
    W: Sync,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    fn new(
        task: TreeTransformTaskView<'a, C>,
        schedule: &'a TreeTransformParallelSchedule,
        gemms: Option<&'a dyn LaneGemms>,
        writer: &'a W,
        src: &'a [D],
        coefficients: &'a [D],
        members: usize,
    ) -> Self {
        Self {
            parts: ReplayParts::new(task),
            schedule,
            gemms,
            writer,
            src,
            coefficients,
            rank: task.layouts().max_fused_rank(),
            members,
            member_src: src.len() / members,
        }
    }

    /// Places the components for `runner`'s form, sizes each lane's scratch
    /// for its components, splits `dst` into atom slices and runs every
    /// lane; `false` without a write when the placement is one lane of one
    /// slot.
    fn run<T, E>(
        &self,
        mut runner: Runner<'_, '_, E, D>,
        tables: &mut LaneTables,
        lane_scratch: &mut Vec<TreeLane<D>>,
        atoms_kept: &mut Vec<&'static mut [u8]>,
        components_kept: &mut Vec<&'static mut [&'static mut [u8]]>,
        dst: &mut [T],
    ) -> Result<bool, OperationError>
    where
        W: DestinationWriter<T, D>,
        T: Send + Sync,
        E: DenseExecutor,
    {
        let schedule = self.schedule;
        // Placement reads only the jobs' estimates, before any atom slice
        // exists.
        let jobs = TreeWork::<'_, '_, '_, '_, W, T, D, C> {
            replay: self,
            target: core::marker::PhantomData,
        };
        match &mut runner {
            Runner::Native(exec) => lanes::place(exec, &jobs, Form::Native, tables),
            Runner::Barrier(host) => crate::host_pool::lane_region(
                host.engine,
                crate::host_pool::HostPoolSite::Replay,
                |exec| lanes::place(&exec, &jobs, Form::Barrier, tables),
            )?,
        }
        let placement = tables.placement();
        let count = placement.lanes();
        if count == 1 && placement.share(0) == 1 {
            return Ok(false);
        }
        if lane_scratch.len() < count {
            lane_scratch.resize_with(count, TreeLane::default);
        }
        for (l, lane) in lane_scratch.iter_mut().take(count).enumerate() {
            let (source, destination) =
                placement
                    .jobs(l)
                    .iter()
                    .fold((0usize, 0usize), |(source, destination), &job| {
                        let component = &schedule.components[self.job(job).1];
                        (
                            source.max(component.packed_source),
                            destination.max(component.packed_destination),
                        )
                    });
            // Capacity only: nothing is written here, on the caller.
            lane.source.reserve_overwrite(source);
            if destination > lane.destination.capacity() {
                lane.destination.reserve_overwrite(destination);
            }
            let index = placement.share(l).saturating_mul(self.rank);
            if lane.index.len() < index {
                lane.index.resize(index, 0);
            }
        }

        let mut atoms: Vec<&mut [T]> = recycle(core::mem::take(atoms_kept));
        let result = split_atoms(&mut atoms, schedule, dst, self.members).and_then(|()| {
            let mut components: Vec<&mut [&mut [T]]> = recycle(core::mem::take(components_kept));
            let mut rest = &mut atoms[..];
            for _ in 0..self.members {
                for c in 0..schedule.component_count() {
                    let (own, tail) =
                        core::mem::take(&mut rest).split_at_mut(schedule.component_atoms(c).len());
                    components.push(own);
                    rest = tail;
                }
            }
            let work = TreeWork {
                replay: self,
                target: core::marker::PhantomData,
            };
            let lane_scratch = &mut lane_scratch[..count];
            let ran = match runner {
                Runner::Native(exec) => lanes::run_native(
                    exec,
                    &work,
                    lanes::Placed { tables },
                    &mut components,
                    lane_scratch,
                )
                .map_err(|refusal| refused(refusal, |_| oversized())),
                Runner::Barrier(host) => {
                    lanes::run_barrier(host, &work, tables, &mut components, lane_scratch)
                        .map_err(|refusal| refused(refusal, |error| error))
                }
            };
            *components_kept = recycle(components);
            match ran? {
                Ok(()) => Ok(true),
                Err((_, error)) => Err(error),
            }
        });
        *atoms_kept = recycle(atoms);
        result
    }
}

/// The runner's refusal as this replay's error: the context's through
/// `context`, a short state or lane supply as an invalid lane split (each
/// component and lane is given its state, so neither is reachable).
fn refused<R>(
    refusal: lanes::Refusal<R>,
    context: impl FnOnce(R) -> OperationError,
) -> OperationError {
    match refusal {
        lanes::Refusal::Context(refused) => context(refused),
        lanes::Refusal::States { .. } | lanes::Refusal::Lanes { .. } => {
            OperationError::InvalidArgument {
                message: "tree transform lanes lack a component or lane state",
            }
        }
    }
}

/// A placement on the context's own budget fits it, so the runner never
/// refuses its lane shares.
fn oversized() -> OperationError {
    OperationError::InvalidArgument {
        message: "tree transform lanes exceed the engine budget",
    }
}

/// Each of the `members` equal members of `dst` split into the schedule's
/// joint atoms, held member after member in slot order (so each
/// component's atoms in a member are one contiguous run), in `atoms`'
/// emptied capacity. O(members atoms).
fn split_atoms<'d, T>(
    atoms: &mut Vec<&'d mut [T]>,
    schedule: &TreeTransformParallelSchedule,
    dst: &'d mut [T],
    members: usize,
) -> Result<(), OperationError> {
    let count = schedule.atoms.len();
    atoms.clear();
    atoms.resize_with(members * count, Default::default);
    let member_len = dst.len() / members;
    let mut members_rest = dst;
    for member in atoms.chunks_mut(count.max(1)).take(members) {
        let (own, tail) = core::mem::take(&mut members_rest).split_at_mut(member_len);
        members_rest = tail;
        split_member(member, schedule, own)?;
    }
    Ok(())
}

/// One member's `dst` split into the schedule's atoms, by slot.
fn split_member<'d, T>(
    atoms: &mut [&'d mut [T]],
    schedule: &TreeTransformParallelSchedule,
    dst: &'d mut [T],
) -> Result<(), OperationError> {
    let outside = || OperationError::InvalidArgument {
        message: "tree transform atom lies outside the destination",
    };
    let mut rest = dst;
    let mut rest_start = 0isize;
    for &slot in &schedule.address_order {
        let atom = &schedule.atoms[slot as usize];
        let skip = atom
            .lo
            .checked_sub(rest_start)
            .and_then(|skip| usize::try_from(skip).ok())
            .filter(|&skip| {
                skip.checked_add(atom.len)
                    .is_some_and(|end| end <= rest.len())
            })
            .ok_or_else(outside)?;
        let (_, tail) = core::mem::take(&mut rest).split_at_mut(skip);
        let (data, tail) = tail.split_at_mut(atom.len);
        atoms[slot as usize] = data;
        rest = tail;
        rest_start = isize::try_from(atom.len)
            .ok()
            .and_then(|len| atom.lo.checked_add(len))
            .ok_or_else(outside)?;
    }
    Ok(())
}

/// What a component's movement stage writes.
#[derive(Clone, Copy)]
enum Movement<'p, D> {
    /// Its inactive layouts and Singles, from its member's source `src`.
    Direct { src: &'p [D] },
    /// Group `group`'s scatter columns from its packed destination.
    Scatter { group: usize, packed: &'p [D] },
}

/// The runner's view of one replay: placement's jobs and the stage bodies.
/// `T` is the destination element, `'s` and `'d` the component tables' and
/// the atom slices' borrows.
struct TreeWork<'r, 'a, 's, 'd, W, T, D, C> {
    replay: &'r Replay<'a, W, D, C>,
    target: core::marker::PhantomData<fn(&'s mut [&'d mut [T]])>,
}

/// A movement stage's estimate: Strided's copy estimate of its elements.
fn copy(elements: usize) -> Estimate {
    Estimate {
        serial: KernelClass::Copy.estimate(elements),
        payback: None,
        sections: 1,
        widen: true,
    }
}

impl<W, T, D, C> Jobs for TreeWork<'_, '_, '_, '_, W, T, D, C>
where
    D: DenseRecouplingScalar,
    C: Copy,
{
    fn count(&self) -> usize {
        self.replay.members * self.replay.schedule.component_count()
    }

    fn stages(&self, job: usize) -> usize {
        let (_, component) = self.replay.job(job);
        1 + 3 * self.replay.schedule.component_group_list(component).len()
    }

    fn kind(&self, _: usize, stage: usize) -> StageKind {
        if stage > 0 && (stage - 1) % 3 == 1 {
            StageKind::Provider
        } else {
            StageKind::Movement
        }
    }

    fn estimate(&self, job: usize, stage: usize) -> Estimate {
        let schedule = self.replay.schedule;
        let (_, component) = self.replay.job(job);
        if stage == 0 {
            return copy(schedule.components[component].direct_weight);
        }
        let group = schedule.component_group_list(component)[(stage - 1) / 3] as usize;
        let shape = &self.replay.parts.plan.jobs()[group];
        match (stage - 1) % 3 {
            0 => copy(shape.rows.saturating_mul(shape.contracted)),
            1 => match self.replay.gemms {
                Some(gemms) => gemms.estimate(group),
                None => lanes::gemm_estimate(dense_dtype::<D>(), lane_gemm_shape(shape)),
            },
            _ => copy(shape.rows.saturating_mul(shape.cols)),
        }
    }
}

impl<'s, 'd, W, T, D, C> Work for TreeWork<'_, '_, 's, 'd, W, T, D, C>
where
    W: DestinationWriter<T, D>,
    T: Send + Sync,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    type Job = &'s mut [&'d mut [T]];
    type Lane = TreeLane<D>;
    type Error = OperationError;

    fn run(
        &self,
        job: usize,
        stage: usize,
        atoms: &mut &'s mut [&'d mut [T]],
        lane: &mut TreeLane<D>,
        exec: lanes::Exec<'_>,
    ) -> Result<(), OperationError> {
        let replay = self.replay;
        let TreeLane {
            source,
            destination,
            index,
            pieces,
            subs,
            packs,
        } = lane;
        let (member, component) = replay.job(job);
        if stage == 0 {
            // Within the capacity the caller reserved: fills, never allocates.
            let packed = replay.schedule.components[component].packed_destination;
            if destination.len() < packed {
                destination.resize_filled(packed, D::zero());
            }
            let src = replay.source(member);
            let direct = Movement::Direct { src };
            let estimate = self.estimate(job, stage);
            return replay.movement(
                component, direct, atoms, pieces, subs, index, estimate, exec,
            );
        }
        let group = replay.schedule.component_group_list(component)[(stage - 1) / 3] as usize;
        let shape = &replay.parts.plan.jobs()[group];
        let packed = shape.rows * shape.cols;
        match (stage - 1) % 3 {
            0 => {
                let estimate = self.estimate(job, stage);
                replay.pack(
                    group,
                    replay.source(member),
                    source,
                    packs,
                    index,
                    estimate,
                    exec,
                )
            }
            1 => {
                // The barrier form submits its GEMMs between rounds.
                let gemms = replay.gemms.ok_or(OperationError::InvalidArgument {
                    message: "a barrier-form lane ran a GEMM stage",
                })?;
                let coefficients = coefficients_of(replay.coefficients, shape)?;
                let destination = destination
                    .as_mut_slice()
                    .get_mut(..packed)
                    .ok_or(scratch_short())?;
                gemms
                    .run(
                        exec,
                        group,
                        D::dense_lane_gemm_io(source.as_slice(), coefficients, destination),
                    )
                    .map_err(OperationError::Dense)
            }
            _ => {
                let packed = destination
                    .as_slice()
                    .get(..packed)
                    .ok_or(scratch_short())?;
                replay.movement(
                    component,
                    Movement::Scatter { group, packed },
                    atoms,
                    pieces,
                    subs,
                    index,
                    self.estimate(job, stage),
                    exec,
                )
            }
        }
    }
}

/// Group `shape`'s recoupling coefficients, `k x n` column-major.
fn coefficients_of<'c, D>(
    coefficients: &'c [D],
    shape: &DenseGemmBatchJob,
) -> Result<&'c [D], OperationError> {
    coefficients
        .get(shape.rhs_offset..shape.rhs_offset + shape.contracted * shape.cols)
        .ok_or(OperationError::InvalidArgument {
            message: "tree transform recoupling coefficients are shorter than a job",
        })
}

impl<'s, 'd, W, T, D, C, E> lanes::Barrier<TreeWork<'_, '_, 's, 'd, W, T, D, C>>
    for EngineGemms<'_, E, D>
where
    E: DenseExecutor,
    W: DestinationWriter<T, D>,
    T: Send + Sync,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    type Refused = OperationError;

    fn region(
        &mut self,
        round: &mut dyn FnMut(lanes::Exec<'_>) -> Result<(), tenferro_cpu::threadteam::Error>,
    ) -> Result<(), OperationError> {
        crate::host_pool::lane_region(self.engine, crate::host_pool::HostPoolSite::Replay, round)?
            .map_err(|_| oversized())
    }

    fn submit(
        &mut self,
        job: usize,
        stage: usize,
        _: &mut &'s mut [&'d mut [T]],
        lane: &mut TreeLane<D>,
    ) -> Result<(), OperationError> {
        let start = self.matmul.map(|_| std::time::Instant::now());
        let component = job % self.schedule.component_count();
        let group = self.schedule.component_group_list(component)[(stage - 1) / 3] as usize;
        let shape = &self.jobs[group];
        let (rows, k, n) = (shape.rows, shape.contracted, shape.cols);
        let (lhs_shape, rhs_shape, dst_shape) = ([rows, k], [k, n], [rows, n]);
        let (rows_major, rhs_strides) = ([1, rows], [1, k]);
        let rhs = coefficients_of(self.coefficients, shape)?;
        let destination = lane
            .destination
            .as_mut_slice()
            .get_mut(..rows * n)
            .ok_or(scratch_short())?;
        let result = (|| {
            self.dense.matmul_axpby_into(
                self.engine,
                D::dense_write(DenseViewMut::new(destination, &dst_shape, &rows_major, 0)?),
                D::dense_read(DenseView::new(
                    lane.source.as_slice(),
                    &lhs_shape,
                    &rows_major,
                    0,
                )?),
                D::dense_read(DenseView::new(rhs, &rhs_shape, &rhs_strides, 0)?),
                D::one().dense_scalar(),
                D::zero().dense_scalar(),
            )
        })()
        .map_err(OperationError::Dense);
        if let (Some(matmul), Some(start)) = (self.matmul.as_mut(), start) {
            *matmul += start.elapsed();
        }
        result
    }
}

fn scratch_short() -> OperationError {
    OperationError::InvalidArgument {
        message: "tree transform lane scratch is shorter than its group",
    }
}
