//! Movement phases use one cost model and one balanced whole-token split.
//! TensorKit's threaded index-manipulation kernels likewise parallelize
//! independent writes around grouped recoupling. Safe Rust assigns each
//! overlapping destination hull run to one writer; the width is capped by
//! those runnable ownership jobs.

use symtn_dense::lanes::{self, Cut, Split, Work};

use super::writers::{scheduled_single, DestinationWriter};
use super::*;
use crate::host_scratch::PackColumnWriter;
use crate::kernel_adapter::movement::with_conjugation;
use crate::kernel_adapter::sealed::GateKeys;
use crate::transform_structure::{
    TreeTransformDirectItem, TreeTransformGroupSpan, TreeTransformPackReplay,
};

/// One-slot, Sequential and Inner replays use the serial item loop without the
/// movement model; only Sequential narrows its token. Wider Auto replays price
/// direct fills and Singles, and chunk plan groups by the token's width. Other
/// forced policies keep
/// their widths, and execution gates decide whether one inline item fans out.
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
) -> Result<(usize, bool, bool), OperationError>
where
    W: DestinationWriter<T, D>,
    T: Send + Sync,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    let layouts = task.layouts();
    let rank = layouts.max_fused_rank();
    let serial = tenferro_cpu::with_cpu_engine(engine, |cpu| {
        let strategy = cpu.batch_policy().strategy;
        let sequential = strategy == tenferro_cpu::CpuBatchStrategy::Sequential;
        if cpu.num_threads() > 1 && !sequential && strategy != tenferro_cpu::CpuBatchStrategy::Inner
        {
            return None;
        }
        Some(crate::host_pool::enter_cpu(
            cpu,
            crate::host_pool::HostPoolSite::Replay,
            true,
            |mut exec| {
                workspace.prepare_fused_indices(1, rank)?;
                workspace.gate_keys.prepare(
                    &mut exec.reborrow().into_token(),
                    layouts.max_planned_elements(),
                );
                #[cfg(test)]
                workspace.gate_keys.watch_as(gate);
                let HostTreeTransformWorkspace {
                    zero_strides,
                    fused_indices,
                    ..
                } = &mut *workspace;
                let token = exec.reborrow().into_token();
                let token = if sequential {
                    token.limit(core::num::NonZeroUsize::MIN)
                } else {
                    token
                };
                inline_direct(
                    token,
                    schedule,
                    true,
                    InlineSingles::ordered(W::FIRST_TOUCH, task, 1),
                    dst,
                    src,
                    writer,
                    kernels,
                    zero_strides,
                    fused_indices,
                    gate,
                )?;
                Ok::<_, OperationError>((1, sequential, false))
            },
        ))
    })
    .flatten();
    if let Some(serial) = serial {
        return serial?;
    }
    let admitted = movement::region(engine, |mut exec, constants| {
        let modeled = exec.budget() > 1
            && exec.batch_policy().strategy == tenferro_cpu::CpuBatchStrategy::Auto;
        workspace.gate_keys.prepare(
            &mut exec.reborrow().into_token(),
            layouts.max_planned_elements(),
        );
        #[cfg(test)]
        workspace.gate_keys.watch_as(gate);
        prepare_decisions::<W, T, D, C>(workspace, task, schedule, writer, &exec, constants)?;
        let [threads, singles_width, inactive_width] = workspace.movement_memo.current().widths
            [..3]
            .try_into()
            .expect("direct phases");
        let serial = workspace.movement_memo.current().serial;
        workspace.prepare_fused_indices(threads.max(singles_width).max(inactive_width), rank)?;
        let split_inactive = inactive_width > 1;
        let split_singles = singles_width > 1;
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
                inactive_width,
                dst,
                fused_indices,
            )?;
        }
        let token = exec.reborrow().into_token();
        let token = if serial {
            token.limit(core::num::NonZeroUsize::MIN)
        } else {
            token
        };
        let inline = if split_singles {
            InlineSingles::None
        } else {
            InlineSingles::ordered(W::FIRST_TOUCH, task, singles_width)
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
        Ok::<_, OperationError>((threads, false, split_inactive || split_singles, modeled))
    })??;
    #[cfg(any(test, feature = "testing"))]
    if admitted.0 > 1 || admitted.2 {
        workspace.movement_phases += 1;
    }
    Ok((admitted.0, admitted.1, admitted.3))
}

/// Only the selected group width fixes chunk boundaries, so these tables are
/// rebuilt on a decision-key change, never on a warm replay.
#[derive(Clone, Debug, Default)]
pub(super) struct ChunkMovement {
    groups: Vec<usize>,
    runs: Vec<u32>,
    prefix: Vec<crate::movement::Job>,
    planned: Vec<crate::movement::Planned>,
    warm_pack_width: usize,
}

impl ChunkMovement {
    pub fn retained_bytes(&self) -> usize {
        self.groups.capacity() * core::mem::size_of::<usize>()
            + self.runs.capacity() * core::mem::size_of::<u32>()
            + self.prefix.capacity() * core::mem::size_of::<crate::movement::Job>()
            + self.planned.capacity() * core::mem::size_of::<crate::movement::Planned>()
    }
}

fn prepare_decisions<W, T, D, C>(
    workspace: &mut TreeTransformWorkspace<D>,
    task: TreeTransformTaskView<'_, C>,
    schedule: &TreeTransformParallelSchedule,
    writer: &W,
    exec: &lanes::Exec<'_>,
    constants: Option<&movement::Constants>,
) -> Result<(), OperationError>
where
    W: DestinationWriter<T, D>,
    T: Send + Sync,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    let key = movement::decision_key::<D>(
        exec,
        constants,
        writer.reads_destination(),
        writer.replays_plans(),
        workspace.packed.destination().len(),
        workspace.packed.destination().capacity(),
        1,
    );
    let HostTreeTransformWorkspace {
        movement_memo,
        gate_keys,
        ..
    } = workspace;
    movement_memo.update(
        key,
        task.admission_identity(),
        movement::dispatch_key(exec),
        |widths, movement_chunks| {
            let fixed = |summary: &crate::movement::Summary| {
                movement::width::<D, _>(
                    exec,
                    constants,
                    summary.prefix.len(),
                    summary.working_set as u128,
                    summary.logical_traffic(key.reads_destination) as u128,
                    || summary.counts.map(|count| count.elements as f64),
                    |slots, elements| key.replays_plans && gate_keys.admits(slots, elements),
                    |costs| movement::fixed(summary, costs),
                )
            };
            let jobs = task.recoupling_plan().jobs();
            // Why the widest chunk the token allows, not a modeled width:
            // chunks run one after another, each paying its engine entries,
            // pack, one dense batch call and scatter, so fewer chunks never
            // add symtn-side cost; how the batch call runs its jobs is the
            // dense backend's decision, which symtn neither sees from a
            // region nor duplicates. Only the token's width bounds a chunk,
            // as on main; a wider chunk needs only larger packed scratch
            // (`chunk_scratch`).
            let threads = lanes::forced_width(exec, jobs.len())
                .unwrap_or_else(|| exec.budget().min(jobs.len()))
                .max(1);
            widths.extend([
                threads,
                fixed(&schedule.single_movement),
                fixed(&schedule.inactive_movement),
            ]);
            let (_, destination_len) = chunk_scratch(jobs, threads)?;
            let mut fill =
                key.destination_len < destination_len && key.destination_capacity < destination_len;
            for (index, chunk) in jobs.chunks(threads).enumerate() {
                let first_group = index * threads;
                let last_group = first_group + chunk.len() - 1;
                let first = schedule.scatter_groups[first_group].pack.start;
                let end = schedule.scatter_groups[last_group].pack.end;
                let columns = end - first;
                let extra = (fill && columns != 0).then_some(destination_len);
                fill &= columns == 0;
                let summary = crate::movement::Summary::totals(
                    &schedule.pack_movement[first_group..=last_group],
                )?;
                let pack_width = |extra: Option<usize>| {
                    movement::width::<D, _>(
                        exec,
                        constants,
                        columns,
                        summary.working_set as u128 + extra.unwrap_or(0) as u128,
                        summary.traffic as u128 + extra.unwrap_or(0) as u128,
                        || {
                            let mut elements = summary.counts.map(|count| count.elements as f64);
                            elements[2] += extra.unwrap_or(0) as f64;
                            elements
                        },
                        |_, _| false,
                        |costs| {
                            movement::prefix_phase(
                                &schedule.pack_prefix,
                                first,
                                columns,
                                &[],
                                costs,
                                extra,
                            )
                        },
                    )
                };
                let cold_pack_width = pack_width(extra);
                let warm_pack_width = if extra.is_some() {
                    pack_width(None)
                } else {
                    cold_pack_width
                };
                if index == movement_chunks.len() {
                    movement_chunks.push(ChunkMovement::default());
                }
                let ChunkMovement {
                    groups,
                    runs,
                    prefix,
                    planned,
                    warm_pack_width: retained_warm_width,
                } = &mut movement_chunks[index];
                *retained_warm_width = warm_pack_width;
                groups.clear();
                groups.extend(
                    (first_group..=last_group)
                        .filter(|&group| !schedule.scatter_groups[group].columns.is_empty()),
                );
                groups.sort_unstable_by_key(|&group| schedule.group_spans[group].lo);
                group_runs(groups, &schedule.group_spans, runs);
                let summary = crate::movement::Summary::totals(
                    groups
                        .iter()
                        .map(|&group| &schedule.scatter_movement[group]),
                )?;
                prefix.clear();
                let mut cumulative = crate::movement::Job::default();
                for (run, &start) in runs.iter().enumerate() {
                    let end = runs.get(run + 1).map_or(groups.len(), |&end| end as usize);
                    let mut roles = crate::movement::Planned::default();
                    for &group in &groups[start as usize..end] {
                        cumulative = cumulative.add(schedule.scatter_movement[group].job())?;
                        roles = roles.join(schedule.scatter_movement[group].planned_roles);
                    }
                    let mut job = cumulative;
                    job.roles = roles;
                    prefix.push(job);
                }
                crate::movement::planned_tree(&prefix[..], planned);
                let scatter_width = movement::width::<D, _>(
                    exec,
                    constants,
                    runs.len(),
                    summary.working_set as u128,
                    summary.logical_traffic(key.reads_destination) as u128,
                    || summary.counts.map(|count| count.elements as f64),
                    |slots, elements| key.replays_plans && gate_keys.admits(slots, elements),
                    |costs| movement::prefix_phase(prefix, 0, runs.len(), planned, costs, None),
                );
                widths.extend([cold_pack_width, scatter_width]);
            }
            Ok(())
        },
    )?;
    movement_memo.current_mut().serial = movement::fails_closed(exec, constants);
    Ok(())
}

/// Execute compiled chunks of plan groups, each with an independently priced
/// pack, one grouped GEMM, and an independently priced scatter. A cold packed
/// destination is first-written inside the first nonempty pack; warm scratch
/// needs no fill. Each failure stops the replay.
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
    let scratch_width = crate::host_pool::threads(engine);
    workspace.prepare_fused_indices(scratch_width, layouts.max_fused_rank())?;
    let (source_len, scratch_destination) = chunk_scratch(jobs, threads)?;
    workspace.packed.source_mut().reserve_overwrite(source_len);
    let chunk_len = threads.min(jobs.len());
    workspace.chunk_jobs.clear();
    workspace.chunk_jobs.reserve(chunk_len);
    workspace.chunk_runs.clear();
    workspace.chunk_runs.reserve(chunk_len);
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
        let filled = (fill && !pack.is_empty()).then_some(scratch_destination);
        let pack_width = workspace.movement_memo.current().widths[3 + index * 2];
        pack_chunk(
            engine,
            workspace,
            task,
            pack,
            source_start,
            source_len,
            filled,
            src,
            pack_width,
        )?;
        fill &= pack.is_empty();
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
            fused_indices,
            gate_keys,
            movement_memo,
            ..
        } = &mut *workspace;
        let packed = &packed.destination().as_slice()[..destination_len];
        let movement = movement_memo.current();
        let groups = &movement.chunks[index].groups[..];
        let runs = &movement.chunks[index].runs[..];
        let width = movement.widths[4 + index * 2];
        let serial = movement.serial;
        crate::host_pool::lane_region(engine, crate::host_pool::HostPoolSite::Replay, |exec| {
            let exec = if serial {
                exec.limit(core::num::NonZeroUsize::MIN)
            } else {
                exec
            };
            let items = Items::Groups {
                groups,
                runs,
                packed,
                packed_start: destination_start,
            };
            run_moves(
                exec,
                writer,
                layouts,
                schedule,
                gate_keys,
                items,
                width,
                dst,
                fused_indices,
            )
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
    let movement = workspace.movement_memo.current_mut();
    for index in 0..jobs.chunks(threads).len() {
        movement.widths[3 + index * 2] = movement.chunks[index].warm_pack_width;
    }
    workspace.movement_memo.finish_destination(
        workspace.packed.destination().len(),
        workspace.packed.destination().capacity(),
    );
    Ok(())
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
/// split at the model's selected width; with `fill`, the packed destination is
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
    let mut run = |mut writer: PackColumnWriter<'_, D>,
                   mut fill: Option<PackColumnWriter<'_, D>>| {
        crate::host_pool::lane_region(engine, crate::host_pool::HostPoolSite::Replay, |exec| {
            if threads == 1 {
                if let Some(fill) = fill.as_mut() {
                    fill.fill(D::zero());
                }
                return with_conjugation!(D, conjugate, |map| {
                    for column in columns {
                        writer.write_column(
                            column.packed_offset - source_start,
                            layouts.role(column.src_layout)?,
                            src,
                            layouts.entry(column.src_layout).offset,
                            fused_indices.as_mut_slice(),
                            map,
                        )?;
                    }
                    Ok(())
                });
            }
            #[cfg(test)]
            replay_paths::pack_runner();
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
                first: 0,
                end: columns.len(),
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
    #[cfg(test)]
    replay_paths::moves();
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
    first: usize,
    end: usize,
}

impl<'l, D: Copy> Pack<'_, 'l, D> {
    /// Cuts `lane` before column `cut.job`: the pack scratch at the column's
    /// packed offset, the destination fill in proportion to active leaves.
    fn cut(&self, lane: &mut PackLane<'l, D>, cut: Cut) -> Result<PackLane<'l, D>, OperationError> {
        let boundary = self.columns[cut.job].packed_offset;
        let writer = lane.writer.split_off(boundary - lane.start)?;
        let fill = match lane.fill.as_mut() {
            Some(fill) => {
                let left = cut.left.min(cut.job - lane.first);
                let right = (cut.threads - cut.left).min(lane.end - cut.job);
                let mid = fill.len() / (left + right) * left
                    + (fill.len() % (left + right)) * left / (left + right);
                Some(fill.split_off(mid)?)
            }
            None => None,
        };
        let end = lane.end;
        lane.end = cut.job;
        Ok(PackLane {
            writer,
            start: boundary,
            index: lane.index.split_off(cut.left, cut.threads, self.rank),
            fill,
            first: cut.job,
            end,
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

#[cfg(test)]
pub(super) mod replay_paths {
    use std::cell::Cell;

    std::thread_local! {
        static COUNTS: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
    }

    pub fn pack_runner() {
        COUNTS.with(|counts| counts.set((counts.get().0 + 1, counts.get().1)));
    }

    pub fn moves() {
        COUNTS.with(|counts| counts.set((counts.get().0, counts.get().1 + 1)));
    }

    pub fn take() -> (usize, usize) {
        COUNTS.with(|counts| counts.replace((0, 0)))
    }
}

#[cfg(test)]
mod memo_capacity_tests {
    use super::*;

    /// Purpose: evicting a same-sized structure reuses every decision buffer
    /// without allocating, freeing buffers, or growing its capacity charge.
    #[test]
    fn movement_eviction_reuses_chunk_buffer_capacities() {
        let _serial = crate::counting_alloc::serial();
        let identities: Vec<_> = (0..=16).map(|_| Arc::new(())).collect();
        let key = movement::DecisionKey {
            budget: 2,
            carries_team: true,
            generation: 0,
            strategy: tenferro_cpu::CpuBatchStrategy::Auto,
            bytes: core::mem::size_of::<f64>(),
            reads_destination: false,
            replays_plans: false,
            destination_len: 0,
            destination_capacity: 0,
            members: 1,
        };
        let build = |widths: &mut Vec<usize>, chunks: &mut Vec<ChunkMovement>| {
            widths.extend([1, 2, 3]);
            if chunks.is_empty() {
                chunks.push(ChunkMovement::default());
            }
            let chunk = &mut chunks[0];
            chunk.groups.clear();
            chunk.groups.extend([0, 1]);
            chunk.runs.clear();
            chunk.runs.extend([0, 1]);
            chunk.prefix.clear();
            chunk.prefix.extend([crate::movement::Job::default(); 2]);
            Ok(())
        };
        let mut memo = movement::WidthMemo::default();
        for identity in &identities[..identities.len() - 1] {
            memo.update(
                key,
                Arc::downgrade(identity),
                core::iter::once(std::time::Duration::ZERO),
                build,
            )
            .unwrap();
        }
        let retained = memo.retained_bytes();
        let ((), allocations) = crate::counting_alloc::measure(|| {
            memo.update(
                key,
                Arc::downgrade(identities.last().unwrap()),
                core::iter::once(std::time::Duration::ZERO),
                build,
            )
            .unwrap();
        });
        assert_eq!(allocations.calls, 0);
        assert_eq!(allocations.frees, 0);
        assert_eq!(memo.retained_bytes(), retained);
    }
}
