//! Native tree replays on the lane runner ([`symtn_dense::lanes`]).
//!
//! Every joint component of the compiled schedule is one job: it owns its
//! atoms of the destination for the whole call, so each destination element
//! has one writer. Its stages are `Direct` (its inactive layouts and
//! Singles), then for each of its Multi groups in recoupling-plan order
//! `Pack`, `Gemm` and `Scatter`. Placement prices each stage by its provider
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
//! - a GEMM through the executor's lane GEMMs on the stage's `x`-slot
//!   context, into the lane's initialized packed destination
//!   ([`symtn_dense::lanes::LaneGemms::run`]).
//!
//! Each lane keeps grow-only scratch (a packed source buffer, an initialized
//! packed destination, traversal indices and the band tables), retained in
//! the replay workspace with the runner's tables.

use std::sync::{PoisonError, RwLock};

use strided_kernel::execution::KernelClass;
use symtn_dense::lanes::{
    self, Estimate, Form, Jobs, LaneGemmShape, LaneGemms, LaneTables, StageKind, Work,
};
use symtn_dense::{DenseDType, DenseScalar};

use super::writers::{Band, DestinationWriter, ReplayParts};
use super::*;
use crate::host_scratch::PackColumnWriter;
use crate::kernel_adapter::movement::with_conjugation;
use crate::transform_structure::TreeTransformPackReplay;

/// One sub-slot's share of one atom of a component stage: the whole atom,
/// or a run of its bands.
pub(super) struct Piece<'a, X> {
    slot: u32,
    sub: u32,
    band: Option<Band>,
    /// First destination element of `data`.
    start: isize,
    data: &'a mut [X],
}

/// One band sub-slot's pieces, traversal index and outcome.
pub(super) struct SubSlot<'c, 'd, X> {
    pieces: &'c mut [Piece<'d, X>],
    index: &'c mut [usize],
    result: Result<(), OperationError>,
}

type SubCell<'c, 'd, X> = RwLock<Option<SubSlot<'c, 'd, X>>>;

/// One pack range's writer, columns, traversal index and outcome.
pub(super) struct PackSlot<'c, X> {
    writer: PackColumnWriter<'c, X>,
    columns: &'c [TreeTransformPackReplay],
    /// Packed offset of the writer's first slot.
    start: usize,
    index: &'c mut [usize],
    result: Result<(), OperationError>,
}

type PackCell<'c, X> = RwLock<Option<PackSlot<'c, X>>>;

/// One lane's scratch, kept between calls; the band and pack tables are
/// empty between stages and keep only their capacity.
///
/// Why `RwLock` cells for a fork's sub-slots: `fork` shares one `Fn` with
/// every child, so each child takes its own `&mut` share from a cell; a std
/// `RwLock` locks without allocating, where a fresh std `Mutex` allocates on
/// its first lock on Apple targets.
pub(super) struct TreeLane<D> {
    source: HostScratchBuffer<D>,
    /// Initialized once and only where it grows; every GEMM overwrites what
    /// it uses.
    destination: Vec<D>,
    index: Vec<usize>,
    pieces: Vec<Piece<'static, u8>>,
    subs: Vec<SubCell<'static, 'static, u8>>,
    packs: Vec<PackCell<'static, u8>>,
}

impl<D> Default for TreeLane<D> {
    fn default() -> Self {
        Self {
            source: HostScratchBuffer::default(),
            destination: Vec::new(),
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

    /// The last call's placement (read by the native provider's lane tests).
    #[cfg(all(
        test,
        feature = "cpu-faer",
        not(any(
            feature = "cpu-blas",
            feature = "blas-accelerate",
            feature = "blas-openblas",
            feature = "blas-mkl",
            feature = "provider-inject"
        ))
    ))]
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
/// Errors follow the runner: every component is attempted, a component
/// stops at its own first failing stage, and the lowest-indexed failing
/// component's error is returned; an initialized destination may then be
/// partly written. A refused entry writes nothing.
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
    shapes.extend(
        task.recoupling_plan()
            .jobs()
            .iter()
            .map(|job| LaneGemmShape {
                rows: job.rows,
                k: job.contracted,
                n: job.cols,
            }),
    );
    let mut target = Some(dst);
    let mut outcome = None;
    let entered = dense.with_lane_gemms(engine, dense_dtype::<D>(), shapes, &mut |exec, gemms| {
        let Some(dst) = target.take() else {
            return;
        };
        let replay = Replay {
            parts: ReplayParts::new(task),
            schedule,
            gemms,
            writer,
            src,
            coefficients: coefficient_scratch,
            rank: task.layouts().max_fused_rank(),
        };
        outcome = Some(replay.run(exec, tables, lanes, atoms, components, dst));
    });
    entered.map_err(OperationError::Dense)?;
    let ran = outcome.unwrap_or(Err(OperationError::InvalidArgument {
        message: "the lane GEMM entry returned without running its body",
    }))?;
    if let (true, Some(profile), Some(start)) = (ran, profile, start) {
        profile.fused_lanes += start.elapsed();
        profile.single_blocks += schedule.single_block_count;
        profile.multi_blocks += task.recoupling_plan().jobs().len();
        profile.packed_columns += schedule.packed_column_count;
        profile.scattered_columns += schedule.scattered_column_count;
    }
    Ok(ran)
}

/// The read-only parts of one lane replay.
struct Replay<'a, W, D, C> {
    parts: ReplayParts<'a, C>,
    schedule: &'a TreeTransformParallelSchedule,
    gemms: &'a dyn LaneGemms,
    writer: &'a W,
    src: &'a [D],
    coefficients: &'a [D],
    rank: usize,
}

impl<'a, W, D, C> Replay<'a, W, D, C>
where
    W: Sync,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    /// Places the components on `exec`, sizes each lane's scratch for its
    /// components, splits `dst` into atom slices and runs every lane; `false`
    /// without a write when the placement is one lane of one slot.
    fn run<T>(
        &self,
        exec: lanes::Exec<'_>,
        tables: &mut LaneTables,
        lane_scratch: &mut Vec<TreeLane<D>>,
        atoms_kept: &mut Vec<&'static mut [u8]>,
        components_kept: &mut Vec<&'static mut [&'static mut [u8]]>,
        dst: &mut [T],
    ) -> Result<bool, OperationError>
    where
        W: DestinationWriter<T, D>,
        T: Send + Sync,
    {
        let schedule = self.schedule;
        // Placement reads only the jobs' estimates, before any atom slice
        // exists.
        let jobs = TreeWork::<'_, '_, '_, '_, W, T, D, C> {
            replay: self,
            target: core::marker::PhantomData,
        };
        lanes::place(&exec, &jobs, Form::Native, tables);
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
                    .fold((0usize, 0usize), |(source, destination), &c| {
                        let component = &schedule.components[c];
                        (
                            source.max(component.packed_source),
                            destination.max(component.packed_destination),
                        )
                    });
            lane.source.reserve_overwrite(source);
            if lane.destination.len() < destination {
                lane.destination.resize(destination, D::zero());
            }
            let index = placement.share(l).saturating_mul(self.rank);
            if lane.index.len() < index {
                lane.index.resize(index, 0);
            }
        }

        let mut atoms: Vec<&mut [T]> = recycle(core::mem::take(atoms_kept));
        let result = split_atoms(&mut atoms, schedule, dst).and_then(|()| {
            let mut components: Vec<&mut [&mut [T]]> = recycle(core::mem::take(components_kept));
            let mut rest = &mut atoms[..];
            for c in 0..schedule.component_count() {
                let (own, tail) =
                    core::mem::take(&mut rest).split_at_mut(schedule.component_atoms(c).len());
                components.push(own);
                rest = tail;
            }
            let work = TreeWork {
                replay: self,
                target: core::marker::PhantomData,
            };
            let ran = lanes::run_native(
                exec,
                &work,
                tables,
                &mut components,
                &mut lane_scratch[..count],
            );
            *components_kept = recycle(components);
            match ran {
                Ok(Ok(())) => Ok(true),
                Ok(Err((_, error))) => Err(error),
                // A placement on `exec`'s own budget fits it.
                Err(_) => Err(OperationError::InvalidArgument {
                    message: "tree transform lanes exceed the engine budget",
                }),
            }
        });
        *atoms_kept = recycle(atoms);
        result
    }
}

/// `dst` split into the schedule's joint atoms, held in slot order (so each
/// component's atoms are one contiguous run), in `atoms`' emptied capacity.
/// O(atoms).
fn split_atoms<'d, T>(
    atoms: &mut Vec<&'d mut [T]>,
    schedule: &TreeTransformParallelSchedule,
    dst: &'d mut [T],
) -> Result<(), OperationError> {
    atoms.clear();
    atoms.resize_with(schedule.atoms.len(), Default::default);
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
    /// Its inactive layouts and Singles.
    Direct,
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
    }
}

impl<W, T, D, C> Jobs for TreeWork<'_, '_, '_, '_, W, T, D, C>
where
    D: DenseRecouplingScalar,
    C: Copy,
{
    fn count(&self) -> usize {
        self.replay.schedule.component_count()
    }

    fn stages(&self, job: usize) -> usize {
        1 + 3 * self.replay.schedule.component_group_list(job).len()
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
        if stage == 0 {
            return copy(schedule.components[job].direct_weight);
        }
        let group = schedule.component_group_list(job)[(stage - 1) / 3] as usize;
        let shape = &self.replay.parts.plan.jobs()[group];
        match (stage - 1) % 3 {
            0 => copy(shape.rows.saturating_mul(shape.contracted)),
            1 => self.replay.gemms.estimate(group),
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
        if stage == 0 {
            return replay.movement(job, Movement::Direct, atoms, pieces, subs, index, exec);
        }
        let group = replay.schedule.component_group_list(job)[(stage - 1) / 3] as usize;
        let shape = &replay.parts.plan.jobs()[group];
        let packed = shape.rows * shape.cols;
        match (stage - 1) % 3 {
            0 => replay.pack(group, source, packs, index, exec),
            1 => {
                let coefficients = replay
                    .coefficients
                    .get(shape.rhs_offset..shape.rhs_offset + shape.contracted * shape.cols)
                    .ok_or(OperationError::InvalidArgument {
                        message: "tree transform recoupling coefficients are shorter than a job",
                    })?;
                let destination = destination.get_mut(..packed).ok_or(scratch_short())?;
                replay
                    .gemms
                    .run(
                        exec,
                        group,
                        D::dense_lane_gemm_io(source.as_slice(), coefficients, destination),
                    )
                    .map_err(OperationError::Dense)
            }
            _ => {
                let packed = destination.get(..packed).ok_or(scratch_short())?;
                replay.movement(
                    job,
                    Movement::Scatter { group, packed },
                    atoms,
                    pieces,
                    subs,
                    index,
                    exec,
                )
            }
        }
    }
}

fn scratch_short() -> OperationError {
    OperationError::InvalidArgument {
        message: "tree transform lane scratch is shorter than its group",
    }
}

impl<W, D, C> Replay<'_, W, D, C>
where
    W: Sync,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    /// Whether an overwrite of component `component` runs whole on its one
    /// planned Single on `exec`, never banded: a group-free component whose
    /// one item is a Single with a compiled plan that [`admitted_plan`]
    /// admits on `exec` (the same check its writer then applies).
    fn whole_single(&self, component: usize, exec: &Exec<'_>) -> bool {
        let schedule = self.schedule;
        if !schedule.component_group_list(component).is_empty() {
            return false;
        }
        let [TreeTransformDirectItem::Single { block, .. }] = *schedule.component_direct(component)
        else {
            return false;
        };
        let layouts = self.parts.layouts;
        scheduled_single(self.parts.blocks, block)
            .and_then(|(dst_layout, _, _)| {
                let elements = layouts.entry(dst_layout).element_count;
                layouts
                    .role(dst_layout)
                    .map(|role| admitted_plan(role, elements, exec).is_some())
            })
            .unwrap_or(false)
    }

    /// Movement stage `movement` of component `component` on `exec`.
    #[allow(clippy::too_many_arguments)]
    fn movement<T>(
        &self,
        component: usize,
        movement: Movement<'_, D>,
        atoms: &mut [&mut [T]],
        pieces: &mut Vec<Piece<'static, u8>>,
        subs: &mut Vec<SubCell<'static, 'static, u8>>,
        index: &mut [usize],
        exec: lanes::Exec<'_>,
    ) -> Result<(), OperationError>
    where
        W: DestinationWriter<T, D>,
        T: Send + Sync,
    {
        let schedule = self.schedule;
        let empty = match movement {
            Movement::Direct => schedule.component_direct(component).is_empty(),
            Movement::Scatter { group, .. } => schedule.scatter_groups[group].columns.is_empty(),
        };
        if empty {
            return Ok(());
        }
        let token = exec.into_token();
        let width = token.budget().get();
        let bands = if self.writer.overwrites() && self.whole_single(component, &token) {
            1
        } else {
            width.min(schedule.components[component].units).max(1)
        };
        let mut list: Vec<Piece<'_, T>> = recycle(core::mem::take(pieces));
        let result = self
            .band(component, bands, atoms, &mut list)
            .and_then(|()| {
                if bands == 1 {
                    return self.write_pieces(
                        component,
                        movement,
                        &mut list,
                        &mut index[..self.rank],
                        token,
                    );
                }
                let mut cells: Vec<SubCell<'_, '_, T>> = recycle(core::mem::take(subs));
                let mut rest = &mut list[..];
                let mut indices = &mut index[..bands * self.rank];
                for sub in 0..bands {
                    let own = rest
                        .iter()
                        .take_while(|piece| piece.sub as usize == sub)
                        .count();
                    let (own, tail) = core::mem::take(&mut rest).split_at_mut(own);
                    rest = tail;
                    let (traversal, tail) = core::mem::take(&mut indices).split_at_mut(self.rank);
                    indices = tail;
                    cells.push(RwLock::new(Some(SubSlot {
                        pieces: own,
                        index: traversal,
                        result: Ok(()),
                    })));
                }
                token.fork(bands, |sub, child| {
                    let mut cell = cells[sub].write().unwrap_or_else(PoisonError::into_inner);
                    if let Some(SubSlot {
                        pieces,
                        index,
                        result,
                    }) = cell.as_mut()
                    {
                        *result = self.write_pieces(component, movement, pieces, index, child);
                    }
                });
                let mut first = Ok(());
                for cell in &mut cells {
                    if let Some(slot) = cell.get_mut().unwrap_or_else(PoisonError::into_inner) {
                        if first.is_ok() {
                            first = core::mem::replace(&mut slot.result, Ok(()));
                        }
                    }
                }
                *subs = recycle(cells);
                first
            });
        *pieces = recycle(list);
        result
    }

    /// Writes `movement`'s items of each of `pieces` on `exec`: the items of
    /// the piece's atom, cropped to its bands.
    fn write_pieces<T>(
        &self,
        component: usize,
        movement: Movement<'_, D>,
        pieces: &mut [Piece<'_, T>],
        index: &mut [usize],
        mut exec: Exec<'_>,
    ) -> Result<(), OperationError>
    where
        W: DestinationWriter<T, D>,
    {
        let schedule = self.schedule;
        // Why a fresh adapter and stride vector per stage, not lane-retained
        // ones: every write here passes its compiled role, the compiled zero
        // source strides and this sub-slot's traversal index, so neither is
        // ever grown, and both are built without allocating.
        let mut kernels = self.writer.kernels();
        let mut zero_strides = Vec::new();
        for piece in pieces {
            match movement {
                Movement::Direct => {
                    let direct = schedule.component_direct(component);
                    let first = direct.partition_point(|item| item.slot() < piece.slot);
                    for &item in direct[first..]
                        .iter()
                        .take_while(|item| item.slot() == piece.slot)
                    {
                        self.writer.direct(
                            &mut kernels,
                            &mut zero_strides,
                            item,
                            piece.data,
                            piece.start,
                            piece.band,
                            exec.reborrow(),
                            index,
                        )?;
                    }
                }
                Movement::Scatter { group, packed } => {
                    let job = &self.parts.plan.jobs()[group];
                    let columns =
                        &schedule.scatter_columns[schedule.scatter_groups[group].columns.clone()];
                    let first = columns.partition_point(|column| column.slot < piece.slot);
                    for column in columns[first..]
                        .iter()
                        .take_while(|column| column.slot == piece.slot)
                    {
                        self.writer.scatter(
                            &mut kernels,
                            &mut zero_strides,
                            column.dst_layout,
                            packed,
                            column.packed_offset - job.dst_offset,
                            piece.data,
                            piece.start,
                            piece.band,
                            exec.reborrow(),
                            index,
                        )?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Splits component `component`'s atom slices into `pieces` for `bands`
    /// sub-slots, reborrowing them for one stage.
    ///
    /// At one sub-slot every atom is one whole piece. Otherwise the
    /// component's band units (each band of a bandable atom, each
    /// indivisible atom) are numbered in address order; unit `u`, with `P`
    /// destination elements in the units before it out of `total`, goes to
    /// sub-slot `min(bands - 1, floor(P bands / total))`: contiguous ranges
    /// balanced by written elements, the same map for every stage of the
    /// component at that sub-slot count. A run of one atom's units on one
    /// sub-slot is one piece. O(atoms + bands).
    fn band<'x, T>(
        &self,
        component: usize,
        bands: usize,
        atoms: &'x mut [&mut [T]],
        pieces: &mut Vec<Piece<'x, T>>,
    ) -> Result<(), OperationError> {
        let schedule = self.schedule;
        let range = schedule.component_atoms(component);
        pieces.reserve(range.len() + bands);
        if bands == 1 {
            for (slot, data) in range.zip(atoms.iter_mut()) {
                pieces.push(Piece {
                    slot: slot as u32,
                    sub: 0,
                    band: None,
                    start: schedule.atoms[slot].lo,
                    data,
                });
            }
            return Ok(());
        }
        let total = schedule.atoms[range.clone()]
            .iter()
            .fold(0u128, |sum, atom| sum + atom.weight as u128)
            .max(1);
        let width = bands as u128;
        let sub_of = |before: u128| ((before * width / total) as usize).min(bands - 1);
        let mut before = 0u128;
        for (slot, data) in range.zip(atoms.iter_mut()) {
            let atom = schedule.atoms[slot];
            let band = schedule.bands[slot];
            let data: &'x mut [T] = data;
            if band.extent < 2 {
                pieces.push(Piece {
                    slot: slot as u32,
                    sub: sub_of(before) as u32,
                    band: None,
                    start: atom.lo,
                    data,
                });
                before += atom.weight as u128;
                continue;
            }
            let (extent, stride) = (band.extent, band.stride);
            let unit = (atom.weight / extent) as u128;
            let mut rest = data;
            let mut k0 = 0usize;
            while k0 < extent {
                let sub = sub_of(before + k0 as u128 * unit);
                // The first band of a later sub-slot: the smallest `k` with
                // `(before + k unit) bands >= (sub + 1) total`.
                let k1 = if sub + 1 >= bands || unit == 0 {
                    extent
                } else {
                    let need = (sub as u128 + 1) * total;
                    let have = before * width;
                    let step = unit * width;
                    (need.saturating_sub(have).div_ceil(step) as usize).clamp(k0 + 1, extent)
                };
                let end = (k1 - k0)
                    .checked_mul(stride)
                    .map_or(rest.len(), |len| len.min(rest.len()));
                let (data, tail) = core::mem::take(&mut rest).split_at_mut(end);
                rest = tail;
                pieces.push(Piece {
                    slot: slot as u32,
                    sub: sub as u32,
                    band: Some((k0, k1)),
                    start: atom.lo + offset_to_isize(k0 * stride)?,
                    data,
                });
                k0 = k1;
            }
            before += atom.weight as u128;
        }
        Ok(())
    }

    /// Group `group`'s pack into `source` on `exec`: one flat fork over
    /// `min(width, columns)` contiguous column ranges, inline at one.
    fn pack(
        &self,
        group: usize,
        source: &mut HostScratchBuffer<D>,
        packs: &mut Vec<PackCell<'static, u8>>,
        index: &mut [usize],
        exec: lanes::Exec<'_>,
    ) -> Result<(), OperationError> {
        let job = &self.parts.plan.jobs()[group];
        let len = job.rows * job.contracted;
        let columns = &self.schedule.pack_columns[self.schedule.scatter_groups[group].pack.clone()];
        let token = exec.into_token();
        let parts = token.budget().get().min(columns.len());
        source.overwrite_with_columns(len, |mut writer| {
            if parts <= 1 {
                return self.pack_columns(&mut writer, columns, job.lhs_offset, index);
            }
            let mut cells: Vec<PackCell<'_, D>> = recycle(core::mem::take(packs));
            let mut indices = &mut index[..parts * self.rank];
            let mut push = |writer, range: core::ops::Range<usize>| {
                let columns = &columns[range];
                let (traversal, tail) = core::mem::take(&mut indices).split_at_mut(self.rank);
                indices = tail;
                cells.push(RwLock::new(Some(PackSlot {
                    writer,
                    columns,
                    start: columns[0].packed_offset,
                    index: traversal,
                    result: Ok(()),
                })));
            };
            let (mut rest, mut lo) = (writer, 0);
            for part in 1..parts {
                let hi = part * columns.len() / parts;
                let (own, tail) =
                    rest.split_at(columns[hi].packed_offset - columns[lo].packed_offset)?;
                push(own, lo..hi);
                (rest, lo) = (tail, hi);
            }
            push(rest, lo..columns.len());
            token.fork(parts, |part, _| {
                let mut cell = cells[part].write().unwrap_or_else(PoisonError::into_inner);
                if let Some(PackSlot {
                    writer,
                    columns,
                    start,
                    index,
                    result,
                }) = cell.as_mut()
                {
                    *result = self.pack_columns(writer, columns, *start, index);
                }
            });
            let mut first = Ok(());
            for cell in &mut cells {
                if let Some(slot) = cell.get_mut().unwrap_or_else(PoisonError::into_inner) {
                    if first.is_ok() {
                        first = core::mem::replace(&mut slot.result, Ok(()));
                    }
                }
            }
            *packs = recycle(cells);
            first
        })
    }

    /// Packs `columns` through `writer`, whose first slot is packed offset
    /// `start`.
    fn pack_columns(
        &self,
        writer: &mut PackColumnWriter<'_, D>,
        columns: &[TreeTransformPackReplay],
        start: usize,
        index: &mut [usize],
    ) -> Result<(), OperationError> {
        let layouts = self.parts.layouts;
        with_conjugation!(D, self.parts.storage_conjugate, |map| {
            for column in columns {
                writer.write_column(
                    column.packed_offset - start,
                    layouts.role(column.src_layout)?,
                    self.src,
                    layouts.entry(column.src_layout).offset,
                    index,
                    map,
                )?;
            }
            Ok(())
        })
    }
}
