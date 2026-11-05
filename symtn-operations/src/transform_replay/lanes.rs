//! The lane route: one destination writer map per call. Every joint
//! component runs on lane slots of its own, its inactive writes, Singles and
//! Multi groups (pack, the dense backend's GEMM on the lane, scatter) in
//! order, and a wide component bands its destination so that every stage
//! writes each destination element from the same slot.

use super::writers::{Band, DestinationWriter, ReplayParts};
use super::*;
use crate::transform_structure::{OneSlotStep, TreeTransformDirectItem};
use core::num::NonZeroUsize;
use core::ops::Range;
use core::time::Duration;
use symtn_dense::{DenseDType, DenseScalar, LaneGemmShape, LaneGemms, LaneToken};

/// One component's placement: its width and the times the placement
/// compares, and the packed scratch its largest group needs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct LanePlacement {
    /// The slots the component runs on (`width_c`).
    pub(super) width: usize,
    /// Its modeled time on one slot (`time_c(1)`).
    pub(super) serial: Duration,
    /// Its modeled time at `width`.
    pub(super) timed: Duration,
    /// Largest `rows * k` and `rows * n` of its groups.
    pub(super) source: usize,
    pub(super) destination: usize,
}

/// One lane of the split tier: its components (a range of the split
/// parts), its slots, and its scratch carving (prefix sums over the lanes
/// before it).
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct LaneLeaf {
    parts: (usize, usize),
    budget: usize,
    source_before: usize,
    destination_before: usize,
    pieces_before: usize,
}

/// A split-tier component and the serial time of the split components
/// before it, in nanoseconds (the balance prefix).
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct LanePart {
    component: u32,
    serial_before: usize,
}

/// One sub-slot's share of one atom of a banded component.
pub(super) struct Piece<'a, T> {
    slot: u32,
    sub: u32,
    band: Option<Band>,
    /// First destination element of `data`.
    start: isize,
    data: &'a mut [T],
}

/// One cost class's placement, which every member shares: equal job-shape
/// multisets also give equal packed scratch maxima.
pub(super) type ClassPlacement = LanePlacement;

/// A lane placement's admission rule for components that may run whole
/// on their plan ([`TreeTransformJointComponent::planned_single`]): only an
/// overwrite may. The width such a component takes is Strided's own copy
/// width on the queried token ([`LeafGate::width`]), asked directly: the
/// placement scans no gate, and each writer's gate lives on the thread that
/// runs it (one caller gate for the alone tier, one per split leaf).
#[derive(Clone, Copy, Debug)]
pub(super) struct Admission {
    /// Whether the call's writer overwrites.
    overwrites: bool,
}

impl Admission {
    /// Admission for one call of a writer that `overwrites` or not.
    pub(super) fn new(overwrites: bool) -> Self {
        Self { overwrites }
    }

    /// Whether component `component` may run whole on its plan in this
    /// call, before the width question.
    fn eligible(&self, schedule: &TreeTransformParallelSchedule, component: usize) -> bool {
        self.overwrites && schedule.components[component].planned_single
    }
}

/// The layout of an [`Piece`], for the workspace to keep the piece list's
/// capacity between calls (always empty there).
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub(super) struct ParkedPiece {
    slot: u32,
    sub: u32,
    band: Option<Band>,
    start: isize,
    data: &'static [u8],
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

/// Replays `task` on the lane route of `dense` (see
/// [`super::lane_route`]): inside one [`DenseExecutor::with_lane_gemms`]
/// entry, the placement below runs every joint component's writes on lane
/// slots it owns for the whole call.
///
/// Placement on the entry's token, of budget `B = min(threads, token)`:
/// each component `c` takes the width `width_c` minimising its modeled time
/// `time_c(w)` over the powers of two up to `min(B, units_c)` and that cap
/// (ties to the smaller); `time_c` is Strided's copy estimate of the
/// component's moved elements over `w` plus `1 + 2 g_c` dispatches of a
/// `w`-wide section (none at one slot), plus the dense backend's own time
/// of each of its `g_c` GEMMs at `w` ([`LaneGemms::time`]); an overwrite
/// of a planned Single that Strided widens on `w` slots runs whole on its
/// plan instead and costs one section of Strided's width `w'` there, and
/// then occupies `width_c = w'` slots, not the `w` it was offered
/// ([`component_time`]). The search runs once per cost class
/// ([`place_components`]). Components with `width_c = B` run alone on the
/// whole token, banded at `B` (or whole on their plan), in index order,
/// under one gate on the calling thread; the
/// rest split recursively by [`balanced_boundary`] over their
/// serial times with [`crate::host_pool::lane_share`] shares, on
/// `min(B, Σ width_c)` slots, when that split is cheaper than running them
/// one after another on the whole token ([`split_time`] with the same
/// times); a lane runs its components one after another, each banded at
/// `min(lane slots, width_c)`, under the lane's own gate. Cost: O(C) plus
/// O(Σ_q (1 + g_q) log B) time evaluations over the classes `q` (each
/// Θ(B) dispatch-cost reads, an admitted class's Strided width search
/// included), the split's reduced widths once per (class, width),
/// O(components + lanes log components) for the split, no allocation once
/// the workspace is warm. Gate scans happen in the writers: at most one on
/// the calling thread for the alone tier, and one per split
/// leaf and token shape it runs, on the worker that runs it.
///
/// One budget slot runs [`TreeTransformParallelSchedule::one_slot_order`]
/// on the whole destination instead.
///
/// The map is a function of the compiled structure, the dtype, the engine
/// width and dispatch costs and the dense model, so identical calls write
/// each element from the same slot. Errors: the alone tier stops at its
/// first failing component, the split returns its lowest failing
/// component's (left before right); an initialised destination may then be
/// partly written.
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
    kernels: &mut W::Kernels,
    threads: usize,
    mut profile: Option<&mut TreeTransformReplayProfile>,
) -> Result<(), OperationError>
where
    E: DenseExecutor,
    W: DestinationWriter<T, D>,
    T: Send + Sync,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    let start = profile.as_ref().map(|_| std::time::Instant::now());
    let mut shapes = core::mem::take(&mut workspace.lane_shapes);
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
    if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
        profile.multi_workspace_prepare += start.elapsed();
    }
    let mut inputs = Some((&mut *workspace, dst, kernels));
    let mut outcome = None;
    let start = profile.as_ref().map(|_| std::time::Instant::now());
    let entered =
        dense.with_lane_gemms(engine, dense_dtype::<D>(), &shapes, &mut |token, gemms| {
            let Some((workspace, dst, kernels)) = inputs.take() else {
                return;
            };
            outcome = Some(run(
                token, gemms, workspace, task, schedule, dst, src, writer, kernels, threads,
            ));
        });
    if let (Some(profile), Some(start)) = (profile, start) {
        profile.fused_lanes += start.elapsed();
    }
    workspace.lane_shapes = shapes;
    entered.map_err(OperationError::Dense)?;
    outcome.unwrap_or(Err(OperationError::InvalidArgument {
        message: "the lane GEMM entry returned without running its body",
    }))
}

/// The shared, read-only part of one lane replay.
struct Lanes<'a, W, D, C> {
    writer: &'a W,
    task: ReplayParts<'a, C>,
    schedule: &'a TreeTransformParallelSchedule,
    src: &'a [D],
    coefficients: &'a [D],
    gemms: &'a dyn LaneGemms,
    rank: usize,
    gate_bound: usize,
    /// Whether the writer overwrites ([`DestinationWriter::overwrites`]).
    overwrites: bool,
}

/// A component's destination: the whole destination of a one-lane replay,
/// or atom slices in slot order from slot `base`.
enum Target<'s, 'a, T> {
    Whole(&'s mut [T]),
    Atoms(&'s mut [&'a mut [T]], usize),
}

impl<T> Target<'_, '_, T> {
    /// The slice holding atom `slot` and its first destination element.
    fn at(&mut self, schedule: &TreeTransformParallelSchedule, slot: u32) -> (&mut [T], isize) {
        match self {
            Self::Whole(dst) => (&mut **dst, 0),
            Self::Atoms(slots, base) => (
                &mut *slots[slot as usize - *base],
                schedule.atoms[slot as usize].lo,
            ),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run<W, T, D, C>(
    mut token: LaneToken<'_>,
    gemms: &dyn LaneGemms,
    workspace: &mut TreeTransformWorkspace<D>,
    task: TreeTransformTaskView<'_, C>,
    schedule: &TreeTransformParallelSchedule,
    dst: &mut [T],
    src: &[D],
    writer: &W,
    kernels: &mut W::Kernels,
    threads: usize,
) -> Result<(), OperationError>
where
    W: DestinationWriter<T, D>,
    T: Send + Sync,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    let budget = threads.clamp(1, token.budget().get());
    let layouts = task.layouts();
    let rank = layouts.max_fused_rank();
    workspace.prepare_fused_indices(budget, rank)?;
    let components = schedule.component_count();
    let jobs = task.recoupling_plan().jobs();
    let group_sizes = |component: usize| {
        schedule.component_group_list(component).iter().fold(
            (0usize, 0usize),
            |(source, destination), &group| {
                let job = &jobs[group as usize];
                (
                    source.max(job.rows.saturating_mul(job.contracted)),
                    destination.max(job.rows.saturating_mul(job.cols)),
                )
            },
        )
    };

    if budget == 1 {
        // The one-lane iteration: components in index order on the whole
        // destination, one packed source buffer and destination region, no
        // slices table.
        let (source, destination) = (0..components).fold((0, 0), |(s, d), component| {
            let (source, destination) = group_sizes(component);
            (s.max(source), d.max(destination))
        });
        let HostTreeTransformWorkspace {
            packed,
            lane_sources,
            coefficient_scratch,
            fused_indices,
            zero_strides,
            ..
        } = &mut *workspace;
        // A structure without Multi groups packs nothing and keeps no
        // source buffer.
        let mut unused = HostScratchBuffer::default();
        let source_buffer = if task.recoupling_plan().is_empty() {
            &mut unused
        } else {
            if lane_sources.is_empty() {
                lane_sources.push(HostScratchBuffer::default());
            }
            &mut lane_sources[0]
        };
        source_buffer.reserve_overwrite(source);
        let destination_region = packed.destination_mut().uninit_mut(destination);
        let lanes = Lanes {
            writer,
            task: ReplayParts::new(task),
            schedule,
            src,
            coefficients: coefficient_scratch,
            gemms,
            rank,
            gate_bound: layouts.max_planned_elements(),
            overwrites: writer.overwrites(),
        };
        let gate = LeafGate::new(lanes.gate_bound);
        let exec = token.limit(NonZeroUsize::MIN);
        let mut exec = exec;
        // Source block order (`one_slot_order`), each component with Multi
        // groups whole at its first block.
        for &step in &schedule.one_slot_order {
            match step {
                OneSlotStep::Direct(index) => writer.direct(
                    kernels,
                    zero_strides,
                    schedule.direct[index as usize],
                    &mut *dst,
                    0,
                    None,
                    exec.reborrow(),
                    &gate,
                    &mut fused_indices[..rank],
                )?,
                OneSlotStep::Component(component) => lanes.run_narrow(
                    component as usize,
                    exec.reborrow(),
                    &mut Target::Whole(&mut *dst),
                    &mut *source_buffer,
                    &mut *destination_region,
                    &mut fused_indices[..rank],
                    kernels,
                    zero_strides,
                    &gate,
                )?,
            }
        }
        return Ok(());
    }

    // Placement.
    let mut placements = core::mem::take(&mut workspace.lane_placements);
    let mut classes = core::mem::take(&mut workspace.lane_classes);
    let mut class_widths = core::mem::take(&mut workspace.lane_class_widths);
    let admission = Admission::new(writer.overwrites());
    place_components(
        &mut token,
        gemms,
        schedule,
        budget,
        &group_sizes,
        admission,
        &mut classes,
        &mut placements,
    );

    // The split tier, its balance prefix and its lanes.
    let mut parts = core::mem::take(&mut workspace.lane_parts);
    parts.clear();
    parts.reserve_exact(components + 1);
    let mut serial_before = 0usize;
    let mut widths = 0usize;
    for (component, placement) in placements.iter().enumerate() {
        if placement.width == budget {
            continue;
        }
        parts.push(LanePart {
            component: component as u32,
            serial_before,
        });
        serial_before = serial_before.saturating_add(nanos(placement.serial));
        widths = widths.saturating_add(placement.width);
    }
    let listed = parts.len();
    parts.push(LanePart {
        component: u32::MAX,
        serial_before,
    });
    let width_of = |part: usize| placements[parts[part].component as usize].width;
    let split_width = budget.min(widths);
    let (mut lanes_count, mut split_budget) = (1, budget);
    if listed >= 2 && split_width >= 2 {
        let fork = token.dispatch_cost(NonZeroUsize::new(2).unwrap_or(NonZeroUsize::MIN));
        let sequential = (0..listed).fold(Duration::ZERO, |sum, part| {
            sum.saturating_add(placements[parts[part].component as usize].timed)
        });
        class_widths.clear();
        let cell = core::cell::RefCell::new((token.reborrow(), &mut class_widths));
        // A reduced width is evaluated once per (class, width) of the call.
        let time = |part: usize, at: usize| {
            let width = width_of(part).min(at).max(1);
            let component = parts[part].component as usize;
            if width == width_of(part) {
                return placements[component].timed;
            }
            if width == 1 {
                return placements[component].serial;
            }
            let class = schedule.component_class[component];
            let (token, memo) = &mut *cell.borrow_mut();
            *memo.entry((class, width as u32)).or_insert_with(|| {
                component_time(token, gemms, schedule, component, width, admission).0
            })
        };
        if split_time(
            0..listed,
            split_width,
            split_width,
            &|part| parts[part].serial_before,
            &time,
            fork,
        ) < sequential
        {
            (lanes_count, split_budget) = (split_width, split_width);
        }
    }
    let mut leaves = core::mem::take(&mut workspace.lane_leaves);
    leaves.clear();
    leaves.reserve_exact(listed.min(lanes_count) + 1);
    if listed > 0 {
        collect_leaves(
            0..listed,
            lanes_count,
            split_budget,
            &|part| parts[part].serial_before,
            &mut leaves,
        );
    }
    let atoms_of = |component: usize| schedule.component_atoms(component).len();
    let (mut source_before, mut destination_before, mut pieces_before) = (0usize, 0usize, 0usize);
    for leaf in leaves.iter_mut() {
        let (mut source, mut destination, mut pieces) = (0usize, 0usize, 0usize);
        for part in &parts[leaf.parts.0..leaf.parts.1] {
            let component = part.component as usize;
            let placement = &placements[component];
            source = source.max(placement.source);
            destination = destination.max(placement.destination);
            let width = placement.width.min(leaf.budget);
            if width >= 2 {
                pieces = pieces.max(atoms_of(component).saturating_add(width));
            }
        }
        leaf.source_before = source_before;
        leaf.destination_before = destination_before;
        leaf.pieces_before = pieces_before;
        source_before = source_before.saturating_add(source);
        destination_before = destination_before.saturating_add(destination);
        pieces_before = pieces_before.saturating_add(pieces);
    }
    leaves.push(LaneLeaf {
        parts: (listed, listed),
        budget: 0,
        source_before,
        destination_before,
        pieces_before,
    });
    // The alone tier runs before the split and reuses the same regions.
    let (mut alone_source, mut alone_destination, mut alone_pieces) = (0usize, 0usize, 0usize);
    for (component, placement) in placements.iter().enumerate() {
        if placement.width == budget {
            alone_source = alone_source.max(placement.source);
            alone_destination = alone_destination.max(placement.destination);
            alone_pieces = alone_pieces.max(atoms_of(component).saturating_add(budget));
        }
    }

    let result = {
        let HostTreeTransformWorkspace {
            packed,
            lane_sources,
            coefficient_scratch,
            fused_indices,
            atom_slices,
            lane_pieces,
            ..
        } = &mut *workspace;
        // One packed source buffer per split lane (the alone tier reuses the
        // first), each sized for its largest group before the first fork.
        // A structure without Multi groups packs nothing and keeps none.
        let lanes_listed = if task.recoupling_plan().is_empty() {
            0
        } else {
            (leaves.len() - 1).max(1)
        };
        if lane_sources.len() < lanes_listed {
            lane_sources.resize_with(lanes_listed, HostScratchBuffer::default);
        }
        let mut unused = HostScratchBuffer::default();
        if lanes_listed > 0 {
            for (buffer, pair) in lane_sources.iter_mut().zip(leaves.windows(2)) {
                buffer.reserve_overwrite(pair[1].source_before - pair[0].source_before);
            }
            let first_lane = leaves
                .get(1)
                .map_or(0, |next| next.source_before - leaves[0].source_before);
            lane_sources[0].reserve_overwrite(alone_source.max(first_lane));
        }

        let destination_region = packed
            .destination_mut()
            .uninit_mut(alone_destination.max(destination_before));
        let mut pieces: Vec<Option<Piece<'_, T>>> = recycle(core::mem::take(lane_pieces));
        pieces.resize_with(alone_pieces.max(pieces_before), || None);
        let mut slots = split_atoms(recycle(core::mem::take(atom_slices)), schedule, dst)?;
        let lanes = Lanes {
            writer,
            task: ReplayParts::new(task),
            schedule,
            src,
            coefficients: coefficient_scratch,
            gemms,
            rank,
            gate_bound: layouts.max_planned_elements(),
            overwrites: writer.overwrites(),
        };
        let result = (|| {
            // The alone tier runs on the calling thread: one gate scans the
            // whole token at most once for all of its components.
            let gate = LeafGate::new(lanes.gate_bound);
            for (component, placement) in placements.iter().enumerate() {
                if placement.width != budget {
                    continue;
                }
                lanes.run_component(
                    component,
                    budget,
                    token.reborrow(),
                    &gate,
                    &mut Target::Atoms(&mut slots[..], 0),
                    match lane_sources[..lanes_listed].first_mut() {
                        Some(buffer) => buffer,
                        None => &mut unused,
                    },
                    &mut *destination_region,
                    &mut pieces,
                    fused_indices.as_mut_slice(),
                )?;
            }
            if listed == 0 {
                return Ok(());
            }
            let exec = token
                .reborrow()
                .limit(NonZeroUsize::new(split_budget).unwrap_or(NonZeroUsize::MIN));
            lanes.run_split(
                &parts,
                &placements,
                &leaves,
                0..listed,
                lanes_count,
                split_budget,
                exec,
                &mut slots,
                0,
                &mut lane_sources[..lanes_listed],
                &mut destination_region[..destination_before],
                &mut pieces[..pieces_before],
                fused_indices.as_mut_slice(),
            )
        })();
        *atom_slices = recycle(slots);
        *lane_pieces = recycle(pieces);
        result
    };
    workspace.lane_placements = placements;
    workspace.lane_classes = classes;
    workspace.lane_class_widths = class_widths;
    workspace.lane_parts = parts;
    workspace.lane_leaves = leaves;
    result
}

/// Every component's [`LanePlacement`] into `placements` (see
/// [`lane_replay`]), evaluated once per cost class
/// ([`TreeTransformParallelSchedule::component_class`]): a class's members
/// have equal modeled times at every width and, having equal job-shape
/// multisets, equal packed scratch maxima, so the first member's search and
/// its `O(g_q)` scratch fold serve them all. Cost: O(C) plus, per class
/// `q`, `O(g_q)` and `(1 + g_q)` time evaluations at each of its O(log B)
/// candidate widths, each a dense `time` query of Θ(B) dispatch-cost reads
/// (an admitted class adds one Strided width search of the same order).
#[allow(clippy::too_many_arguments)]
pub(super) fn place_components(
    token: &mut LaneToken<'_>,
    gemms: &dyn LaneGemms,
    schedule: &TreeTransformParallelSchedule,
    budget: usize,
    group_sizes: &impl Fn(usize) -> (usize, usize),
    admission: Admission,
    classes: &mut Vec<Option<ClassPlacement>>,
    placements: &mut Vec<LanePlacement>,
) {
    let components = schedule.component_count();
    // Each table is sized once per call, so a fresh workspace allocates it
    // once however many components the structure has.
    placements.clear();
    placements.reserve_exact(components);
    classes.clear();
    classes.resize(schedule.class_count, None);
    for component in 0..components {
        let class = &mut classes[schedule.component_class[component] as usize];
        let placement = *class.get_or_insert_with(|| {
            let (source, destination) = group_sizes(component);
            let cap = budget.min(schedule.components[component].units).max(1);
            let (serial, _) = component_time(token, gemms, schedule, component, 1, admission);
            let (mut width, mut timed) = (1, serial);
            let mut candidate = 2;
            while candidate <= cap {
                let (time, occupied) =
                    component_time(token, gemms, schedule, component, candidate, admission);
                if time < timed {
                    (width, timed) = (occupied, time);
                }
                if candidate == cap {
                    break;
                }
                candidate = if candidate.saturating_mul(2) > cap {
                    cap
                } else {
                    candidate * 2
                };
            }
            LanePlacement {
                width,
                serial,
                timed,
                source,
                destination,
            }
        });
        placements.push(placement);
    }
}

/// `duration` in whole nanoseconds, saturating.
fn nanos(duration: Duration) -> usize {
    usize::try_from(duration.as_nanos()).unwrap_or(usize::MAX)
}

/// `time_c(width)` of component `component` (see [`lane_replay`]) and the
/// slots it occupies there.
///
/// A component that may run whole on its plan ([`Admission::eligible`])
/// and whose role Strided widens on the `width`-slot token runs as one
/// section of Strided's own width `w'` there, so it costs
/// `estimate / w' + dispatch(w')` and occupies `w'` slots; every other one
/// is banded over all `width`.
///
/// Why the occupied width and not the offered one: a component recorded at
/// the offered `width = B` while Strided runs it on `w' < B` would sit in
/// the alone tier, one after another on `w'` of `B` slots, and never enter
/// the split comparison that values it with this same time.
pub(super) fn component_time(
    token: &mut LaneToken<'_>,
    gemms: &dyn LaneGemms,
    schedule: &TreeTransformParallelSchedule,
    component: usize,
    width: usize,
    admission: Admission,
) -> (Duration, usize) {
    let groups = schedule.component_group_list(component);
    let moved = schedule.components[component].moved;
    let estimate = strided_kernel::execution::KernelClass::Copy.estimate(moved);
    if width >= 2 && admission.eligible(schedule, component) {
        let exec = granted_width(token.reborrow(), width);
        let whole = LeafGate::width(&exec, moved);
        if let Some(whole) = NonZeroUsize::new(whole).filter(|whole| whole.get() > 1) {
            let time = (estimate / u32::try_from(whole.get()).unwrap_or(u32::MAX))
                .saturating_add(exec.dispatch_cost(whole));
            return (time, whole.get());
        }
    }
    let width = NonZeroUsize::new(width).unwrap_or(NonZeroUsize::MIN);
    let movement = if width.get() == 1 {
        estimate
    } else {
        let sections =
            u32::try_from(groups.len().saturating_mul(2).saturating_add(1)).unwrap_or(u32::MAX);
        (estimate / u32::try_from(width.get()).unwrap_or(u32::MAX))
            .saturating_add(token.dispatch_cost(width).saturating_mul(sections))
    };
    let time = groups.iter().fold(movement, |time, &group| {
        time.saturating_add(gemms.time(token, group as usize, width))
    });
    (time, width.get())
}

/// The split tier's lanes, in order: the recursion of [`Lanes::run_split`].
fn collect_leaves(
    parts: Range<usize>,
    lanes: usize,
    budget: usize,
    weight_before: &impl Fn(usize) -> usize,
    leaves: &mut Vec<LaneLeaf>,
) {
    if lanes <= 1 || parts.len() <= 1 {
        leaves.push(LaneLeaf {
            parts: (parts.start, parts.end),
            budget,
            ..Default::default()
        });
        return;
    }
    let middle = parts.start
        + balanced_boundary(parts.len(), lanes, |part| weight_before(parts.start + part));
    let left_lanes = lanes / 2;
    let left_budget = if budget >= 2 {
        crate::host_pool::lane_share(budget, left_lanes, lanes)
    } else {
        budget
    };
    let right_budget = if budget >= 2 {
        budget - left_budget
    } else {
        budget
    };
    collect_leaves(
        parts.start..middle,
        left_lanes,
        left_budget,
        weight_before,
        leaves,
    );
    collect_leaves(
        middle..parts.end,
        lanes - left_lanes,
        right_budget,
        weight_before,
        leaves,
    );
}

impl<W, D, C> Lanes<'_, W, D, C>
where
    W: Sync,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    /// The split tier over `range` of `parts` on `lanes` lanes sharing
    /// `budget` slots of `exec`, with the regions of exactly the leaves of
    /// that range; mirrors [`collect_leaves`].
    #[allow(clippy::too_many_arguments)]
    fn run_split<'a, T>(
        &self,
        parts: &[LanePart],
        placements: &[LanePlacement],
        leaves: &[LaneLeaf],
        range: Range<usize>,
        lanes: usize,
        budget: usize,
        exec: Exec<'_>,
        slots: &mut [&'a mut [T]],
        slot_base: usize,
        sources: &mut [HostScratchBuffer<D>],
        destination: &mut [MaybeUninit<D>],
        pieces: &mut [Option<Piece<'a, T>>],
        fused: &mut [usize],
    ) -> Result<(), OperationError>
    where
        W: DestinationWriter<T, D>,
        T: Send + Sync,
    {
        if lanes <= 1 || range.len() <= 1 {
            // No buffer: a structure without Multi groups packs nothing.
            let mut unused = HostScratchBuffer::default();
            let source = match sources.first_mut() {
                Some(buffer) => buffer,
                None => &mut unused,
            };
            let mut exec = exec;
            // The lane's own gate, on the thread that runs it: it scans only
            // the lane's token shapes.
            let gate = LeafGate::new(self.gate_bound);
            for part in range {
                let component = parts[part].component as usize;
                let width = placements[component].width.min(budget).max(1);
                self.run_component(
                    component,
                    width,
                    exec.reborrow(),
                    &gate,
                    &mut Target::Atoms(&mut *slots, slot_base),
                    &mut *source,
                    &mut *destination,
                    &mut *pieces,
                    &mut *fused,
                )?;
            }
            return Ok(());
        }
        let middle = range.start
            + balanced_boundary(range.len(), lanes, |part| {
                parts[range.start + part].serial_before
            });
        let left_lanes = lanes / 2;
        let (left_budget, right_budget) = if budget >= 2 {
            let left = crate::host_pool::lane_share(budget, left_lanes, lanes);
            (left, budget - left)
        } else {
            (budget, budget)
        };
        let first = leaves.partition_point(|leaf| leaf.parts.0 < range.start);
        let cut = leaves.partition_point(|leaf| leaf.parts.0 < middle);
        let (left_source, right_source) = sources.split_at_mut((cut - first).min(sources.len()));
        let (left_destination, right_destination) = destination
            .split_at_mut(leaves[cut].destination_before - leaves[first].destination_before);
        let (left_pieces, right_pieces) =
            pieces.split_at_mut(leaves[cut].pieces_before - leaves[first].pieces_before);
        let boundary =
            self.schedule.components[parts[middle].component as usize].atom_start as usize;
        let (left_slots, right_slots) = slots.split_at_mut(boundary - slot_base);
        let (left_fused, mut right_fused) = split_lane_indices(fused, left_lanes, lanes, self.rank);
        let (left, right) = replay_join(
            exec,
            Some((left_lanes, lanes)),
            |exec| {
                self.run_split(
                    parts,
                    placements,
                    leaves,
                    range.start..middle,
                    left_lanes,
                    left_budget,
                    exec,
                    left_slots,
                    slot_base,
                    left_source,
                    left_destination,
                    left_pieces,
                    left_fused,
                )
            },
            |exec| {
                self.run_split(
                    parts,
                    placements,
                    leaves,
                    middle..range.end,
                    lanes - left_lanes,
                    right_budget,
                    exec,
                    right_slots,
                    boundary,
                    right_source,
                    right_destination,
                    right_pieces,
                    right_fused.get(),
                )
            },
        );
        left?;
        right
    }

    /// Component `component` on `width` slots of `exec` (which holds at
    /// least that many): whole items on one slot, banded otherwise. `gate`
    /// is the calling thread's; a banded sub-slot on another worker builds
    /// its own.
    #[allow(clippy::too_many_arguments)]
    fn run_component<'a, T>(
        &self,
        component: usize,
        width: usize,
        exec: Exec<'_>,
        gate: &LeafGate,
        target: &mut Target<'_, 'a, T>,
        source: &mut HostScratchBuffer<D>,
        destination: &mut [MaybeUninit<D>],
        pieces: &mut [Option<Piece<'a, T>>],
        fused: &mut [usize],
    ) -> Result<(), OperationError>
    where
        W: DestinationWriter<T, D>,
        T: Send + Sync,
    {
        if width <= 1 {
            let mut kernels = self.writer.kernels();
            return self.run_narrow(
                component,
                granted_width(exec, 1),
                target,
                source,
                destination,
                &mut fused[..self.rank],
                &mut kernels,
                &mut Vec::new(),
                gate,
            );
        }
        let mut exec = granted_width(exec, width);
        if let Some(item) = self.whole_single(component, &exec, gate) {
            // One stage, one item: the admitted plan partitions the role
            // over the token itself, deterministically for its width.
            let (dst, dst_start) = target.at(self.schedule, item.slot());
            let mut kernels = self.writer.kernels();
            return self.writer.direct(
                &mut kernels,
                &mut Vec::new(),
                item,
                dst,
                dst_start,
                None,
                exec,
                gate,
                &mut fused[..self.rank],
            );
        }
        let Target::Atoms(slots, base) = target else {
            return Err(OperationError::InvalidArgument {
                message: "a banded tree transform component needs its atom slices",
            });
        };
        let count = self.band(component, width, slots, *base, pieces)?;
        let pieces = &mut pieces[..count];
        let schedule = self.schedule;
        let direct = schedule.component_direct(component);
        let rank = self.rank;
        for_sub_slots(
            pieces,
            0..width,
            fused,
            rank,
            exec.reborrow(),
            &|pieces, index, exec| {
                let mut kernels = self.writer.kernels();
                let mut zero_strides = Vec::new();
                let gate = LeafGate::new(self.gate_bound);
                let mut exec = exec;
                for piece in pieces.iter_mut().flatten() {
                    let first = direct.partition_point(|item| item.slot() < piece.slot);
                    for &item in direct[first..]
                        .iter()
                        .take_while(|item| item.slot() == piece.slot)
                    {
                        self.writer.direct(
                            &mut kernels,
                            &mut zero_strides,
                            item,
                            &mut *piece.data,
                            piece.start,
                            piece.band,
                            exec.reborrow(),
                            &gate,
                            &mut *index,
                        )?;
                    }
                }
                Ok(())
            },
        )?;
        for &group in schedule.component_group_list(component) {
            let group = group as usize;
            let packed = self.gemm(group, width, exec.reborrow(), source, destination, fused)?;
            let job = &self.task.plan.jobs()[group];
            let columns = &schedule.scatter_columns[schedule.scatter_groups[group].columns.clone()];
            let packed = &*packed;
            for_sub_slots(
                pieces,
                0..width,
                fused,
                rank,
                exec.reborrow(),
                &|pieces, index, exec| {
                    let mut kernels = self.writer.kernels();
                    let mut zero_strides = Vec::new();
                    let gate = LeafGate::new(self.gate_bound);
                    let mut exec = exec;
                    for piece in pieces.iter_mut().flatten() {
                        let first = columns.partition_point(|column| column.slot < piece.slot);
                        for column in columns[first..]
                            .iter()
                            .take_while(|column| column.slot == piece.slot)
                        {
                            self.writer.scatter(
                                &mut kernels,
                                &mut zero_strides,
                                column,
                                packed,
                                column.packed_offset - job.dst_offset,
                                &mut *piece.data,
                                piece.start,
                                piece.band,
                                exec.reborrow(),
                                &gate,
                                &mut *index,
                            )?;
                        }
                    }
                    Ok(())
                },
            )?;
        }
        Ok(())
    }

    /// Component `component`'s one Single when it runs whole on `exec`:
    /// an overwrite of a group-free component whose one item is a Single
    /// with a compiled plan that `gate` admits on `exec` (the same
    /// `admitted_plan` its writer then applies). Accumulations, planless
    /// roles, inactive or coalesced atoms and Multi components stay banded.
    fn whole_single(
        &self,
        component: usize,
        exec: &Exec<'_>,
        gate: &LeafGate,
    ) -> Option<TreeTransformDirectItem> {
        if !self.overwrites || !self.schedule.components[component].planned_single {
            return None;
        }
        let [item @ TreeTransformDirectItem::Single { block, .. }] =
            *self.schedule.component_direct(component)
        else {
            return None;
        };
        let (dst_layout, _, _) = scheduled_single(self.task.blocks, block).ok()?;
        let role = self.task.layouts.role(dst_layout).ok()?;
        let elements = self.task.layouts.entry(dst_layout).element_count;
        admitted_plan(role, elements, exec, gate).map(|_| item)
    }

    /// Component `component` on one slot: every item whole, on its compiled
    /// path, in order (inactive and Singles, then each group's pack, GEMM
    /// and scatter).
    #[allow(clippy::too_many_arguments)]
    fn run_narrow<T>(
        &self,
        component: usize,
        mut exec: Exec<'_>,
        target: &mut Target<'_, '_, T>,
        source: &mut HostScratchBuffer<D>,
        destination: &mut [MaybeUninit<D>],
        index: &mut [usize],
        kernels: &mut W::Kernels,
        zero_strides: &mut Vec<isize>,
        gate: &LeafGate,
    ) -> Result<(), OperationError>
    where
        W: DestinationWriter<T, D>,
        T: Send + Sync,
    {
        let schedule = self.schedule;
        for &item in schedule.component_direct(component) {
            let (dst, dst_start) = target.at(schedule, item.slot());
            self.writer.direct(
                kernels,
                zero_strides,
                item,
                dst,
                dst_start,
                None,
                exec.reborrow(),
                gate,
                &mut *index,
            )?;
        }
        for &group in schedule.component_group_list(component) {
            let group = group as usize;
            let packed = self.gemm(group, 1, exec.reborrow(), source, destination, index)?;
            let job = &self.task.plan.jobs()[group];
            for column in &schedule.scatter_columns[schedule.scatter_groups[group].columns.clone()]
            {
                let (dst, dst_start) = target.at(schedule, column.slot);
                self.writer.scatter(
                    kernels,
                    zero_strides,
                    column,
                    packed,
                    column.packed_offset - job.dst_offset,
                    dst,
                    dst_start,
                    None,
                    exec.reborrow(),
                    gate,
                    &mut *index,
                )?;
            }
        }
        Ok(())
    }

    /// Packs group `group`'s source columns into `source` over `threads`
    /// slots of `exec`, through the buffer's write-before-read column
    /// protocol (split on column boundaries as the staged pack splits), then
    /// runs its GEMM on `exec` from the packed source into `destination`,
    /// returning the initialised packed destination.
    fn gemm<'d>(
        &self,
        group: usize,
        threads: usize,
        mut exec: Exec<'_>,
        source: &mut HostScratchBuffer<D>,
        destination: &'d mut [MaybeUninit<D>],
        fused: &mut [usize],
    ) -> Result<&'d mut [D], OperationError> {
        let job = &self.task.plan.jobs()[group];
        let source_len = job.rows * job.contracted;
        let destination_len = job.rows * job.cols;
        let Some(destination) = destination.get_mut(..destination_len) else {
            return Err(OperationError::InvalidArgument {
                message: "tree transform lane scratch is shorter than its group",
            });
        };
        let columns = &self.schedule.pack_columns;
        let first = columns.partition_point(|column| column.packed_offset < job.lhs_offset);
        let end =
            columns.partition_point(|column| column.packed_offset < job.lhs_offset + source_len);
        source.overwrite_with_columns(source_len, |writer| {
            replay_pack_columns(
                fused,
                self.rank,
                self.task.layouts,
                &columns[first..end],
                writer,
                job.lhs_offset,
                self.src,
                self.task.storage_conjugate,
                threads,
                exec.reborrow(),
            )
        })?;
        let coefficients = self
            .coefficients
            .get(job.rhs_offset..job.rhs_offset + job.contracted * job.cols)
            .ok_or(OperationError::InvalidArgument {
                message: "tree transform recoupling coefficients are shorter than a job",
            })?;
        let mut written = None;
        self.gemms
            .run(
                exec,
                group,
                D::dense_lane_gemm_io(source.as_slice(), coefficients, destination, &mut written),
            )
            .map_err(OperationError::Dense)?;
        written.ok_or(OperationError::InvalidArgument {
            message: "a lane GEMM reported success without its destination",
        })
    }

    /// Splits component `component`'s atom slices (in `slots` from slot
    /// `base`) into `pieces` for `width` sub-slots, returning their count.
    ///
    /// The component's band units (each band of a bandable atom, each
    /// indivisible atom) are numbered in address order; unit `u`, with `P`
    /// destination elements in the units before it out of `total`, goes to
    /// sub-slot `min(width - 1, floor(P width / total))`: contiguous ranges
    /// balanced by moved elements. A run of one atom's units on one sub-slot
    /// is one piece, cut from the atom's slice at its first band. O(atoms +
    /// width), no allocation.
    fn band<'a, T>(
        &self,
        component: usize,
        width: usize,
        slots: &mut [&'a mut [T]],
        base: usize,
        pieces: &mut [Option<Piece<'a, T>>],
    ) -> Result<usize, OperationError> {
        let atoms = &self.schedule.atoms;
        let range = self.schedule.component_atoms(component);
        let total = atoms[range.clone()]
            .iter()
            .fold(0u128, |sum, atom| sum + atom.weight as u128)
            .max(1);
        let width_u = width as u128;
        let sub_of = |before: u128| ((before * width_u / total) as usize).min(width - 1);
        let mut count = 0usize;
        let mut push = |piece: Piece<'a, T>| -> Result<(), OperationError> {
            let slot = pieces
                .get_mut(count)
                .ok_or(OperationError::InvalidArgument {
                    message: "tree transform band pieces exceed their reservation",
                })?;
            *slot = Some(piece);
            count += 1;
            Ok(())
        };
        let mut before = 0u128;
        for slot in range {
            let atom = atoms[slot];
            let data = core::mem::take(&mut slots[slot - base]);
            if !atom.bandable() {
                push(Piece {
                    slot: slot as u32,
                    sub: sub_of(before) as u32,
                    band: None,
                    start: atom.lo,
                    data,
                })?;
                before += atom.weight as u128;
                continue;
            }
            let extent = atom.band_extent;
            let unit = (atom.weight / extent) as u128;
            let stride = atom.band_stride;
            let mut rest = data;
            let mut k0 = 0usize;
            while k0 < extent {
                let sub = sub_of(before + k0 as u128 * unit);
                // The first band of a later sub-slot: the smallest `k` with
                // `(before + k unit) width >= (sub + 1) total`.
                let k1 = if sub + 1 >= width || unit == 0 {
                    extent
                } else {
                    let need = (sub as u128 + 1) * total;
                    let have = before * width_u;
                    let step = unit * width_u;
                    (need.saturating_sub(have).div_ceil(step) as usize).clamp(k0 + 1, extent)
                };
                let end = (k1 - k0)
                    .checked_mul(stride)
                    .map_or(rest.len(), |len| len.min(rest.len()));
                let (data, tail) = core::mem::take(&mut rest).split_at_mut(end);
                rest = tail;
                push(Piece {
                    slot: slot as u32,
                    sub: sub as u32,
                    band: Some((k0, k1)),
                    start: atom.lo + offset_to_isize(k0 * stride)?,
                    data,
                })?;
                k0 = k1;
            }
            before += atom.weight as u128;
        }
        Ok(count)
    }
}

/// Runs `body` once per sub-slot of `subs` over its pieces, forking `exec`
/// so that sub-slot `s` always runs on slot `s` of its `subs.len()`-slot
/// token: the same recursion, and so the same band-to-slot map, for every
/// stage.
fn for_sub_slots<'a, T: Send + Sync>(
    pieces: &mut [Option<Piece<'a, T>>],
    subs: Range<usize>,
    fused: &mut [usize],
    rank: usize,
    exec: Exec<'_>,
    body: &(impl Fn(&mut [Option<Piece<'a, T>>], &mut [usize], Exec<'_>) -> Result<(), OperationError>
          + Sync),
) -> Result<(), OperationError> {
    if subs.len() <= 1 {
        return body(pieces, &mut fused[..rank], exec);
    }
    let lanes = subs.len();
    let left_lanes = lanes / 2;
    let middle = subs.start + left_lanes;
    let at = pieces.partition_point(|piece| {
        piece
            .as_ref()
            .is_some_and(|piece| (piece.sub as usize) < middle)
    });
    let (left_pieces, right_pieces) = pieces.split_at_mut(at);
    let (left_fused, mut right_fused) = split_lane_indices(fused, left_lanes, lanes, rank);
    let (left, right) = replay_join(
        exec,
        Some((left_lanes, lanes)),
        |exec| {
            for_sub_slots(
                left_pieces,
                subs.start..middle,
                left_fused,
                rank,
                exec,
                body,
            )
        },
        |exec| {
            for_sub_slots(
                right_pieces,
                middle..subs.end,
                right_fused.get(),
                rank,
                exec,
                body,
            )
        },
    );
    left?;
    right
}
