//! The parallel tree replay: main's threaded replay
//! (`tree_transform_blocks_with_batched_recoupling_parallel`) on the lane
//! runner's [`Split`] assignment.
//!
//! TensorKit `_add_abelian_kernel_threaded!` / `_add_general_kernel_threaded!`
//! (indexmanipulations.jl) precedent: the Singles apply in parallel when
//! their destinations are slice-disjoint, and the Multi groups replay in
//! chunks of `t` consecutive groups, each a parallel pack, one grouped GEMM
//! between its pack and scatter phases, and a scatter split over the chunk's
//! groups. Disjointness is structural: every split cuts the destination (or
//! pack scratch) at the start of a hull run (see
//! [`TreeTransformParallelSchedule`]) with `split_at_mut`, and a part writes
//! its runs whole, so interleaved layouts whose bounding slices overlap share
//! one writer; a phase of one run stays inline on the caller.
//!
//! Why runs rather than TensorKit's per-group tasks: distinct blocks never
//! share an element, but safe Rust can only hand workers disjoint contiguous
//! slices, and a run is the smallest contiguous owner of overlapping hulls.
//!
//! A split part of `n` logical threads and `k` items splits at
//! `parallel_split(k, n)` items, `n / 2` threads to the left, as a two-child
//! fork of its token; a part of one thread or one item is a leaf, which
//! writes its items whole in order on its token through a gate of its own
//! and stops at its first failure. The left failure wins.

use symtn_dense::lanes::{self, Cut, Split, Work};

use super::writers::{scheduled_single, DestinationWriter};
use super::*;
use crate::host_scratch::PackColumnWriter;
use crate::kernel_adapter::movement::with_conjugation;
use crate::kernel_adapter::sealed::GateKeys;
use crate::transform_structure::{
    TreeTransformDirectItem, TreeTransformGroupSpan, TreeTransformPackReplay,
};

/// The direct items of a replay in one engine entry, and the replay width
/// `t` it admits there, with whether the policy is a forced `Sequential`.
///
/// `t` is [`lanes::flat_width`] of a copy of the whole destination over the
/// compiled runnable parts (main's `effective_tree_transform_threads`):
/// under `Auto` the engine's width when that copy pays for it, at most the
/// runnable parts, else one; a forced policy's decision, at most the runnable
/// parts. It fixes the chunks of the plan groups, and the Singles' split
/// when they are slice-disjoint.
///
/// The inactive layouts run first: split over their hull runs when there
/// are two or more and the run width `t_A` (the same rule over the new run
/// routes' runnable parts) exceeds one, inline on the caller's token
/// otherwise. Then the Singles: split over `t` workers when `t > 1` and they
/// are slice-disjoint (main's split); split over their runs at `t_A` when
/// they form at least two runs that are not all singletons and the caller's
/// gate would not admit the largest planned Single on the caller's token
/// (see [`inline_admits`]); inline in [`InlineSingles::ordered`]'s order
/// otherwise.
///
/// The call's gate cost keys are prepared here, before any split, for
/// every leaf gate of the call (none when no role retains a plan).
#[allow(clippy::too_many_arguments)]
pub(super) fn direct<W, T, D, C>(
    engine: &mut dyn Engine,
    workspace: &mut TreeTransformWorkspace<D>,
    task: TreeTransformTaskView<'_, C>,
    schedule: &TreeTransformParallelSchedule,
    dst: &mut [T],
    src: &[D],
    writer: &W,
    kernels: &mut W::Kernels,
    gate: &mut LeafGate<'_>,
) -> Result<(usize, bool), OperationError>
where
    W: DestinationWriter<T, D>,
    T: Send + Sync,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    let layouts = task.layouts();
    let rank = layouts.max_fused_rank();
    let estimate = strided_kernel::execution::KernelClass::Copy.estimate(dst.len());
    let inactive_runs = schedule.inactive_runs.len();
    let singles_disjoint = schedule.singles_slice_disjoint();
    // Runs of the new routes: the inactive layouts, and Singles that are
    // not slice-disjoint but still form two or more runs.
    let single_runs = if singles_disjoint {
        0
    } else {
        schedule.single_runs.len()
    };
    let admitted = crate::host_pool::lane_region(
        engine,
        crate::host_pool::HostPoolSite::Replay,
        |mut exec| {
            let sequential =
                exec.batch_policy().strategy == tenferro_cpu::CpuBatchStrategy::Sequential;
            let threads = lanes::flat_width(&exec, estimate, schedule.replay_runnable);
            let run_runnable = [inactive_runs, single_runs]
                .into_iter()
                .filter(|&runs| runs >= 2)
                .max()
                .unwrap_or(0);
            let run_threads = if run_runnable >= 2 {
                lanes::flat_width(&exec, estimate, run_runnable)
            } else {
                1
            };
            workspace.prepare_fused_indices(threads.max(run_threads), rank)?;
            workspace.gate_keys.prepare(
                &mut exec.reborrow().into_token(),
                layouts.max_planned_elements(),
            );
            #[cfg(test)]
            workspace.gate_keys.watch_as(gate);
            let split_inactive = run_threads > 1 && inactive_runs >= 2;
            let (split_singles, singles_width) = if singles_disjoint {
                (threads > 1, threads)
            } else {
                let split = run_threads > 1
                    && single_runs >= 2
                    && !inline_admits(gate, exec.token_ref(), schedule.singles_planned);
                (split, run_threads)
            };
            let HostTreeTransformWorkspace {
                zero_strides,
                fused_indices,
                gate_keys,
                ..
            } = &mut *workspace;
            if split_inactive {
                let items = Items::Direct {
                    list: &schedule.inactive,
                    runs: &schedule.inactive_runs,
                    singles: false,
                    blocks: task.blocks(),
                    src,
                };
                run_moves(
                    exec.reborrow(),
                    writer,
                    layouts,
                    schedule,
                    gate_keys,
                    items,
                    run_threads,
                    dst,
                    fused_indices,
                )?;
            }
            let token = exec.reborrow().into_token();
            let token = if sequential {
                token.limit(core::num::NonZeroUsize::MIN)
            } else {
                token
            };
            let inline = if split_singles {
                InlineSingles::None
            } else {
                InlineSingles::ordered(W::FIRST_TOUCH, task, threads)
            };
            inline_direct(
                token,
                schedule,
                !split_inactive,
                inline,
                dst,
                src,
                writer,
                kernels,
                zero_strides,
                fused_indices,
                gate,
            )?;
            if split_singles {
                let items = Items::Direct {
                    list: &schedule.singles,
                    runs: &schedule.single_runs,
                    singles: true,
                    blocks: task.blocks(),
                    src,
                };
                run_moves(
                    exec,
                    writer,
                    layouts,
                    schedule,
                    gate_keys,
                    items,
                    singles_width,
                    dst,
                    fused_indices,
                )?;
            }
            Ok::<_, OperationError>((threads, sequential, split_inactive || split_singles))
        },
    )??;
    #[cfg(any(test, feature = "testing"))]
    if admitted.0 > 1 || admitted.2 {
        workspace.movement_phases += 1;
    }
    Ok((admitted.0, admitted.1))
}

/// The plan groups of a replay at width `threads > 1`, in chunks of
/// `threads` consecutive groups (main's `replay_multi_chunk` and its
/// scatter): per chunk the rebased jobs, a pack split over `threads`
/// workers, one grouped GEMM, and the scatter of the chunk's non-empty
/// groups in destination order: split by group when every group and every
/// pair of them is slice-disjoint (main's split), over their hull runs when
/// there are two or more and the caller's gate would not admit the chunk's
/// largest planned column on the caller's token, and inline on the caller's
/// token through the caller gate otherwise. The first failing phase or GEMM
/// stops the replay.
///
/// The packed scratch is sized once for the largest chunk. A packed
/// destination too small for it is initialized inside the first chunk's
/// pack phase, split with the pack, and nowhere else; one long enough is not
/// touched; one with the capacity but not the length has its tail filled on
/// the caller.
#[allow(clippy::too_many_arguments)]
pub(super) fn groups<E, W, T, D, C>(
    engine: &mut dyn Engine,
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    task: TreeTransformTaskView<'_, C>,
    schedule: &TreeTransformParallelSchedule,
    dst: &mut [T],
    src: &[D],
    writer: &W,
    kernels: &mut W::Kernels,
    gate: &mut LeafGate<'_>,
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
    let layouts = task.layouts();
    let jobs = task.recoupling_plan().jobs();
    let (source_len, scratch_destination) = chunk_scratch(jobs, threads)?;
    workspace.packed.source_mut().reserve_overwrite(source_len);
    let chunk_len = threads.min(jobs.len());
    workspace.chunk_jobs.clear();
    workspace.chunk_jobs.reserve(chunk_len);
    workspace.chunk_runs.clear();
    workspace.chunk_runs.reserve(chunk_len);
    workspace.chunk_scatter_groups.clear();
    workspace.chunk_scatter_groups.reserve(chunk_len);
    let packed_destination = workspace.packed.destination_mut();
    let mut fill = packed_destination.len() < scratch_destination;
    if fill && packed_destination.capacity() >= scratch_destination {
        #[cfg(any(test, feature = "testing"))]
        let tail = scratch_destination - packed_destination.len();
        packed_destination.resize_filled(scratch_destination, D::zero());
        fill = false;
        #[cfg(any(test, feature = "testing"))]
        {
            workspace.destination_fills += tail;
        }
    }

    for (index, chunk) in jobs.chunks(threads).enumerate() {
        let first_group = index * threads;
        let last_group = first_group + chunk.len() - 1;
        let source_start = chunk[0].lhs_offset;
        let destination_start = chunk[0].dst_offset;
        let end = |offset: usize, rows: usize, cols: usize| {
            rows.checked_mul(cols)
                .and_then(|len| offset.checked_add(len))
                .ok_or(OperationError::ElementCountOverflow)
        };
        let last = &chunk[chunk.len() - 1];
        let source_len = end(last.lhs_offset, last.rows, last.contracted)? - source_start;
        let destination_len = end(last.dst_offset, last.rows, last.cols)? - destination_start;
        workspace.chunk_jobs.clear();
        workspace
            .chunk_jobs
            .extend(chunk.iter().map(|job| DenseGemmBatchJob {
                dst_offset: job.dst_offset - destination_start,
                lhs_offset: job.lhs_offset - source_start,
                ..*job
            }));
        strided_batch_runs_into(&workspace.chunk_jobs, &mut workspace.chunk_runs);

        let start = profile.as_ref().map(|_| std::time::Instant::now());
        let pack = &schedule.pack_columns[schedule.scatter_groups[first_group].pack.start
            ..schedule.scatter_groups[last_group].pack.end];
        let filled = fill.then_some(scratch_destination);
        pack_chunk(
            engine,
            workspace,
            task,
            pack,
            source_start,
            source_len,
            filled,
            src,
            threads,
        )?;
        fill = false;
        #[cfg(any(test, feature = "testing"))]
        {
            workspace.movement_phases += 1;
            workspace.destination_fills += filled.unwrap_or(0);
        }
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            profile.multi_blocks += chunk.len();
            profile.packed_columns += chunk.iter().map(|job| job.contracted).sum::<usize>();
            profile.multi_pack += start.elapsed();
        }

        let start = profile.as_ref().map(|_| std::time::Instant::now());
        let (source, destination) = workspace.packed.source_and_destination_mut();
        recoupling_gemm_batch(
            engine,
            dense,
            &mut destination.as_mut_slice()[..destination_len],
            source.as_slice(),
            &workspace.coefficient_scratch,
            &workspace.chunk_jobs,
            &workspace.chunk_runs,
        )?;
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            let elapsed = start.elapsed();
            profile.multi_dense_matmul_call += elapsed;
            profile.multi_matmul_total += elapsed;
        }

        let start = profile.as_ref().map(|_| std::time::Instant::now());
        let HostTreeTransformWorkspace {
            packed,
            zero_strides,
            fused_indices,
            gate_keys,
            chunk_scatter_groups: ordered,
            chunk_scatter_runs: runs,
            ..
        } = &mut *workspace;
        let spans = &schedule.group_spans;
        ordered.clear();
        ordered.extend(
            (first_group..=last_group)
                .filter(|&group| !schedule.scatter_groups[group].columns.is_empty()),
        );
        // Why not sort every scatter column: only the at-most-t group indices
        // need destination order for one safe split tree.
        ordered.sort_unstable_by_key(|&group| spans[group].lo);
        group_runs(ordered, spans, runs);
        let packed = &packed.destination().as_slice()[..destination_len];
        let groups = &ordered[..];
        let runs = &runs[..];
        // Main's split when every group is slice-disjoint and so is every
        // pair of them; otherwise a split over two or more runs unless main's
        // inline route would admit the chunk's largest planned column on
        // the caller's token.
        let main_disjoint =
            runs.len() == groups.len() && groups.iter().all(|&group| spans[group].slice_disjoint);
        let planned = groups
            .iter()
            .map(|&group| spans[group].planned)
            .max()
            .unwrap_or(0);
        crate::host_pool::lane_region(engine, crate::host_pool::HostPoolSite::Replay, |exec| {
            let split = main_disjoint
                || (runs.len() >= 2 && !inline_admits(gate, exec.token_ref(), planned));
            if split {
                let items = Items::Groups {
                    groups,
                    runs,
                    packed,
                    packed_start: destination_start,
                };
                return run_moves(
                    exec,
                    writer,
                    layouts,
                    schedule,
                    gate_keys,
                    items,
                    threads,
                    dst,
                    fused_indices,
                );
            }
            let mut exec = exec.into_token();
            for &group in groups {
                for column in
                    &schedule.scatter_columns[schedule.scatter_groups[group].columns.clone()]
                {
                    writer.scatter(
                        kernels,
                        zero_strides,
                        column.dst_layout,
                        packed,
                        column.packed_offset - destination_start,
                        dst,
                        0,
                        exec.reborrow(),
                        gate,
                        fused_indices,
                    )?;
                }
            }
            Ok(())
        })??;
        #[cfg(any(test, feature = "testing"))]
        {
            workspace.movement_phases += 1;
        }
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            profile.scattered_columns += chunk.iter().map(|job| job.cols).sum::<usize>();
            profile.multi_scatter += start.elapsed();
        }
    }
    Ok(())
}

/// Whether main's inline route would write an item of `planned` elements
/// (the largest role with a retained Strided plan, zero for none) through
/// its plan on the caller's token `exec`, by the caller's gate.
///
/// Why a new run split defers to it: inline, such an item takes the
/// caller's whole token and its plan fans the move out; on a run split
/// leaf's smaller token the gate rejects the plan and the item takes the
/// serial walk, so the split can only lose against main there. Without
/// such an item main writes the phase serially, which a split never loses
/// to.
fn inline_admits(
    gate: &mut LeafGate<'_>,
    exec: &crate::host_pool::Exec<'_>,
    planned: usize,
) -> bool {
    planned > 0 && gate.widens(exec, planned)
}

/// The start of each hull run of `ordered` (groups ascending by hull start)
/// into `runs`: a run ends where the next group starts after the largest
/// hull end so far. O(`ordered.len()`).
fn group_runs(ordered: &[usize], spans: &[TreeTransformGroupSpan], runs: &mut Vec<u32>) {
    runs.clear();
    let mut end = None;
    for (index, &group) in ordered.iter().enumerate() {
        let span = spans[group];
        match end {
            Some(last) if span.lo <= last => end = Some(span.hi.max(last)),
            _ => {
                // At most `threads` groups per chunk, so the index fits.
                runs.push(index as u32);
                end = Some(span.hi);
            }
        }
    }
}

/// The largest chunk's packed source and destination lengths over chunks
/// of `threads` consecutive jobs (main's `reserve_multi_chunk_scratch`).
///
/// Why sized once: chunks are visited in GEMM-run order, not by size, so
/// growing per chunk would reallocate once per larger chunk.
fn chunk_scratch(
    jobs: &[DenseGemmBatchJob],
    threads: usize,
) -> Result<(usize, usize), OperationError> {
    let span = |start: usize, end: Option<usize>| {
        end.and_then(|end| end.checked_sub(start))
            .ok_or(OperationError::ElementCountOverflow)
    };
    let (mut source_len, mut destination_len) = (0usize, 0usize);
    for chunk in jobs.chunks(threads) {
        let (first, last) = (&chunk[0], &chunk[chunk.len() - 1]);
        let source_end = last
            .rows
            .checked_mul(last.contracted)
            .and_then(|len| last.lhs_offset.checked_add(len));
        let destination_end = last
            .rows
            .checked_mul(last.cols)
            .and_then(|len| last.dst_offset.checked_add(len));
        source_len = source_len.max(span(first.lhs_offset, source_end)?);
        destination_len = destination_len.max(span(first.dst_offset, destination_end)?);
    }
    Ok((source_len, destination_len))
}

/// One chunk's pack phase in one engine entry: `columns` packed into the
/// source scratch (`source_len` elements from packed offset `source_start`),
/// split over `threads` workers; with `fill`, the packed destination is
/// initialized to `fill` zeros alongside, each split part its own range.
#[allow(clippy::too_many_arguments)]
fn pack_chunk<D, C>(
    engine: &mut dyn Engine,
    workspace: &mut TreeTransformWorkspace<D>,
    task: TreeTransformTaskView<'_, C>,
    columns: &[TreeTransformPackReplay],
    source_start: usize,
    source_len: usize,
    fill: Option<usize>,
    src: &[D],
    threads: usize,
) -> Result<(), OperationError>
where
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    let layouts = task.layouts();
    let conjugate = task.storage_conjugate();
    let HostTreeTransformWorkspace {
        packed,
        fused_indices,
        ..
    } = &mut *workspace;
    let (source, destination) = packed.split_mut();
    let mut run = |writer: PackColumnWriter<'_, D>, fill: Option<PackColumnWriter<'_, D>>| {
        crate::host_pool::lane_region(engine, crate::host_pool::HostPoolSite::Replay, |exec| {
            let work = Pack {
                columns,
                layouts,
                src,
                conjugate,
                rank: layouts.max_fused_rank(),
                lane: core::marker::PhantomData,
            };
            let mut lane = PackLane {
                writer,
                start: source_start,
                index: LaneIndex::Shared(fused_indices.as_mut_slice()),
                fill,
            };
            let mut states = vec![(); columns.len()];
            let split = split(threads, |lane, cut| work.cut(lane, cut));
            let lanes = core::slice::from_mut(&mut lane);
            let ran = lanes::run_native(exec, &work, split, &mut states, lanes);
            // A chunk without columns runs no leaf, which would have filled.
            if let Some(mut fill) = lane.fill.take() {
                fill.fill(D::zero());
            }
            outcome(ran)
        })?
    };
    match fill {
        None => source.overwrite_with_columns(source_len, |writer| run(writer, None)),
        Some(len) => destination.overwrite_with_columns(len, |fill| {
            source.overwrite_with_columns(source_len, |writer| run(writer, Some(fill)))
        }),
    }
}

/// The runner's result as the replay's: a job's error as it failed, and a
/// refusal, which a split with one lane per call never meets, as an invalid
/// argument.
pub(super) fn outcome(
    ran: Result<lanes::Outcome<OperationError>, lanes::Refusal<tenferro_cpu::threadteam::Error>>,
) -> Result<(), OperationError> {
    match ran {
        Ok(Ok(())) => Ok(()),
        Ok(Err((_, error))) => Err(error),
        Err(_) => Err(OperationError::InvalidArgument {
            message: "tree transform replay split refused its supply",
        }),
    }
}

/// The items of one split destination phase, in hull runs: a job of the
/// phase is one run, `runs[job]..runs[job + 1]` of its list (the last run
/// ends with the list), and a part writes its runs whole.
#[derive(Clone, Copy)]
enum Items<'a, D> {
    /// Singles by block index (`singles`) or inactive layouts by layout
    /// index, in destination address order, from `src`.
    Direct {
        list: &'a [u32],
        runs: &'a [u32],
        singles: bool,
        blocks: &'a [TreeTransformBlock],
        src: &'a [D],
    },
    /// Scatter groups in destination order, from the chunk's packed
    /// destination whose first element is packed offset `packed_start`.
    Groups {
        groups: &'a [usize],
        runs: &'a [u32],
        packed: &'a [D],
        packed_start: usize,
    },
}

impl<D> Items<'_, D> {
    fn runs(&self) -> (&[u32], usize) {
        match *self {
            Items::Direct { list, runs, .. } => (runs, list.len()),
            Items::Groups { groups, runs, .. } => (runs, groups.len()),
        }
    }

    /// The list positions of run `job`.
    fn run(&self, job: usize) -> core::ops::Range<usize> {
        let (runs, len) = self.runs();
        let end = runs.get(job + 1).map_or(len, |&end| end as usize);
        runs[job] as usize..end
    }
}

/// The direct item at position `index` of a direct list.
fn direct_item(list: &[u32], singles: bool, index: usize) -> TreeTransformDirectItem {
    if singles {
        TreeTransformDirectItem::Single { block: list[index] }
    } else {
        TreeTransformDirectItem::Inactive {
            layout: list[index] as usize,
        }
    }
}

/// A split destination phase: each item written whole through `writer`.
struct Moves<'a, 'd, W, T, D> {
    writer: &'a W,
    layouts: &'a TreeTransformLayoutTable,
    schedule: &'a TreeTransformParallelSchedule,
    keys: &'d GateKeys,
    items: Items<'a, D>,
    rank: usize,
    target: core::marker::PhantomData<fn(&'d mut [T])>,
}

/// One worker of a split destination phase: its part of the destination,
/// which starts at destination element `dst_start`, its kernel state, its
/// traversal scratch and its gate.
struct MoveLane<'d, T, K> {
    dst: &'d mut [T],
    dst_start: isize,
    kernels: K,
    zero_strides: Vec<isize>,
    index: LaneIndex<'d>,
    gate: LeafGate<'d>,
}

/// Runs `items` split over `threads` workers on `exec`, into `dst`.
#[allow(clippy::too_many_arguments)]
fn run_moves<W, T, D>(
    exec: lanes::Exec<'_>,
    writer: &W,
    layouts: &TreeTransformLayoutTable,
    schedule: &TreeTransformParallelSchedule,
    keys: &GateKeys,
    items: Items<'_, D>,
    threads: usize,
    dst: &mut [T],
    fused_indices: &mut [usize],
) -> Result<(), OperationError>
where
    W: DestinationWriter<T, D>,
    T: Send + Sync,
    D: Copy + Sync,
{
    let work = Moves {
        writer,
        layouts,
        schedule,
        keys,
        items,
        rank: layouts.max_fused_rank(),
        target: core::marker::PhantomData,
    };
    let mut lane = work.lane(dst, 0, LaneIndex::Shared(fused_indices));
    let mut states = vec![(); work.count()];
    let split = split(threads, |lane, cut| work.cut(lane, cut));
    outcome(lanes::run_native(
        exec,
        &work,
        split,
        &mut states,
        core::slice::from_mut(&mut lane),
    ))
}

/// A [`Split`] over `threads` logical threads whose cut is `cut`; the
/// signature makes the closure generic over its lane borrow.
pub(super) fn split<L, E, F>(threads: usize, cut: F) -> Split<F>
where
    F: Fn(&mut L, Cut) -> Result<L, E> + Sync,
{
    Split {
        threads,
        split: cut,
    }
}

impl<'d, W, T, D> Moves<'_, 'd, W, T, D>
where
    W: DestinationWriter<T, D>,
{
    fn lane(
        &self,
        dst: &'d mut [T],
        dst_start: isize,
        index: LaneIndex<'d>,
    ) -> MoveLane<'d, T, W::Kernels> {
        MoveLane {
            dst,
            dst_start,
            kernels: self.writer.kernels(),
            zero_strides: Vec::new(),
            index,
            gate: LeafGate::shared(self.keys),
        }
    }

    /// The first destination element run `job` writes: its first item's,
    /// since a run's items ascend by hull start.
    fn lo(&self, job: usize) -> Result<isize, OperationError> {
        let first = self.items.run(job).start;
        match self.items {
            Items::Direct {
                list,
                singles: true,
                blocks,
                ..
            } => self
                .layouts
                .destination_lo(scheduled_single(blocks, list[first])?.0),
            // Destination strides are nonnegative, so an inactive layout
            // starts at its offset.
            Items::Direct { list, .. } => Ok(self.layouts.entry(list[first] as usize).offset),
            Items::Groups { groups, .. } => Ok(self.schedule.group_spans[groups[first]].lo),
        }
    }

    /// Cuts `lane`'s destination before item `cut.job`: `lane` keeps the
    /// left part and its first `cut.left` workers' traversal scratch.
    fn cut(
        &self,
        lane: &mut MoveLane<'d, T, W::Kernels>,
        cut: Cut,
    ) -> Result<MoveLane<'d, T, W::Kernels>, OperationError> {
        let boundary = self.lo(cut.job)?;
        let at = boundary
            .checked_sub(lane.dst_start)
            .and_then(|at| usize::try_from(at).ok())
            .filter(|&at| at <= lane.dst.len())
            .ok_or(OperationError::ElementCountOverflow)?;
        let (left, right) = core::mem::take(&mut lane.dst).split_at_mut(at);
        lane.dst = left;
        let index = lane.index.split_off(cut.left, cut.threads, self.rank);
        Ok(self.lane(right, boundary, index))
    }
}

impl<'d, W, T, D> Work for Moves<'_, 'd, W, T, D>
where
    W: DestinationWriter<T, D>,
    T: Send + Sync,
    D: Copy + Sync,
{
    type Job = ();
    type Lane = MoveLane<'d, T, W::Kernels>;
    type Error = OperationError;

    fn count(&self) -> usize {
        self.items.runs().0.len()
    }

    fn run(
        &self,
        job: usize,
        _: &mut (),
        lane: &mut Self::Lane,
        exec: lanes::Exec<'_>,
    ) -> Result<(), OperationError> {
        let MoveLane {
            dst,
            dst_start,
            kernels,
            zero_strides,
            index,
            gate,
        } = lane;
        let index = &mut index.get()[..self.rank];
        let mut exec = exec.into_token();
        let run = self.items.run(job);
        match self.items {
            Items::Direct {
                list, singles, src, ..
            } => {
                for position in run {
                    self.writer.direct(
                        kernels,
                        zero_strides,
                        direct_item(list, singles, position),
                        src,
                        dst,
                        *dst_start,
                        exec.reborrow(),
                        gate,
                        index,
                    )?;
                }
                Ok(())
            }
            Items::Groups {
                groups,
                packed,
                packed_start,
                ..
            } => {
                for &group in &groups[run] {
                    let columns = self.schedule.scatter_groups[group].columns.clone();
                    for column in &self.schedule.scatter_columns[columns] {
                        self.writer.scatter(
                            kernels,
                            zero_strides,
                            column.dst_layout,
                            packed,
                            column.packed_offset - packed_start,
                            dst,
                            *dst_start,
                            exec.reborrow(),
                            gate,
                            index,
                        )?;
                    }
                }
                Ok(())
            }
        }
    }
}

/// A split pack phase: each column written through its part's writer.
struct Pack<'a, 'l, D> {
    columns: &'a [TreeTransformPackReplay],
    layouts: &'a TreeTransformLayoutTable,
    src: &'a [D],
    conjugate: bool,
    rank: usize,
    lane: core::marker::PhantomData<fn(&'l mut [D])>,
}

/// One worker of a split pack: its range of the pack scratch, which starts
/// at packed offset `start`, its traversal scratch, and its range of the
/// packed destination to initialize, if any.
struct PackLane<'a, D> {
    writer: PackColumnWriter<'a, D>,
    start: usize,
    index: LaneIndex<'a>,
    fill: Option<PackColumnWriter<'a, D>>,
}

impl<'l, D: Copy> Pack<'_, 'l, D> {
    /// Cuts `lane` before column `cut.job`: the pack scratch at the column's
    /// packed offset, the destination fill in proportion to the threads.
    fn cut(&self, lane: &mut PackLane<'l, D>, cut: Cut) -> Result<PackLane<'l, D>, OperationError> {
        let boundary = self.columns[cut.job].packed_offset;
        let writer = lane.writer.split_off(boundary - lane.start)?;
        let fill = match lane.fill.as_mut() {
            Some(fill) => {
                let mid = fill.len().saturating_mul(cut.left) / cut.threads;
                Some(fill.split_off(mid)?)
            }
            None => None,
        };
        Ok(PackLane {
            writer,
            start: boundary,
            index: lane.index.split_off(cut.left, cut.threads, self.rank),
            fill,
        })
    }
}

impl<'l, D> Work for Pack<'_, 'l, D>
where
    D: Copy + Send + Sync + Zero + ConjugateValue,
{
    type Job = ();
    type Lane = PackLane<'l, D>;
    type Error = OperationError;

    fn count(&self) -> usize {
        self.columns.len()
    }

    fn run(
        &self,
        job: usize,
        _: &mut (),
        lane: &mut Self::Lane,
        _: lanes::Exec<'_>,
    ) -> Result<(), OperationError> {
        if let Some(mut fill) = lane.fill.take() {
            fill.fill(D::zero());
        }
        let column = &self.columns[job];
        let index = &mut lane.index.get()[..self.rank];
        with_conjugation!(D, self.conjugate, |map| lane.writer.write_column(
            column.packed_offset - lane.start,
            self.layouts.role(column.src_layout)?,
            self.src,
            self.layouts.entry(column.src_layout).offset,
            index,
            map,
        ))
    }
}
