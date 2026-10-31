use super::*;

pub(super) fn effective_tree_transform_threads(
    engine: &mut dyn Engine,
    schedule: &TreeTransformParallelSchedule,
    requested: usize,
) -> usize {
    let singles = if schedule.singles_slice_disjoint {
        schedule.singles.len()
    } else {
        usize::from(!schedule.singles.is_empty())
    };
    let runnable = singles.max(schedule.pack_columns.len()).max(1);
    let requested = requested.max(1);

    // Why answer before reading the engine width: a serial replay needs no
    // engine lookup at all.
    if requested == 1 || runnable == 1 {
        return 1;
    }

    // Why not trust the requested count: the caller's engine cannot execute
    // more workers than it has, and scratch for phantom workers turns a harmless
    // large hint into an allocation overflow before any replay begins.
    requested
        .min(crate::host_pool::threads(engine))
        .min(runnable)
}

/// Runs a replay phase split over `threads` workers with the operation's
/// engine token. A serial phase runs on a serial token, but the engine is
/// still admitted first: a broken or CUDA engine never reaches Host payload.
pub(super) fn replay_region<R>(
    engine: &mut dyn Engine,
    threads: usize,
    op: impl FnOnce(Exec<'_>) -> R,
) -> Result<R, OperationError> {
    if threads <= 1 {
        crate::host_pool::region(engine, crate::host_pool::HostPoolSite::Replay, |_| ())?;
        return Ok(op(Exec::serial()));
    }
    crate::host_pool::region(engine, crate::host_pool::HostPoolSite::Replay, op)
}

/// One replay fork: both halves run as jobs of `exec`. The split itself (and
/// so every per-worker scratch slice and the left-before-right error order)
/// is fixed by the replay's `threads`, not by where the token places a job.
pub(super) fn replay_join<A, B, RA, RB>(exec: Exec<'_>, left: A, right: B) -> (RA, RB)
where
    A: FnOnce(Exec<'_>) -> RA + Send + Sync,
    B: FnOnce(Exec<'_>) -> RB + Send + Sync,
    RA: Send + Sync,
    RB: Send + Sync,
{
    pool_join(exec, left, right)
}

/// Executes a validated tree-transform block list against a dense executor.
/// Single blocks apply directly; each Multi block reuses one source/destination
/// pack pair for `destination = source * U^T`.
///
/// Inlined into both the plain and profiled entry points so the
/// `Option<&mut profile>` checks constant-fold away in the unprofiled copy.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) fn tree_transform_blocks_with_batched_recoupling<A, E, D, C>(
    engine: &mut dyn Engine,
    kernels: &mut A,
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    task: TreeTransformTaskView<'_, C>,
    structure_identity: &Arc<()>,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: DestinationAlpha<'_, D, C>,
    mode: DestinationMode<D>,
    mut profile: Option<&mut TreeTransformReplayProfile>,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy,
{
    let layouts = task.layouts();
    let recoupling_plan = task.recoupling_plan();
    workspace.prepare_fused_indices(1, layouts.max_fused_rank())?;

    // All-Single structures (abelian recoupling is diagonal) skip the batch
    // machinery entirely: no pack scratch, no job list, no scatter pass.
    if recoupling_plan.is_empty() {
        return crate::host_pool::with_exec(engine, |mut exec| {
            for block in task.blocks() {
                let TreeTransformBlock::Single {
                    dst_layout,
                    src_layout,
                    coefficient,
                } = *block
                else {
                    unreachable!("checked above: no Multi blocks");
                };
                let start = profile.as_ref().map(|_| std::time::Instant::now());
                tree_transform_single_with_strided_kernel(
                    exec.reborrow(),
                    kernels,
                    &mut workspace.zero_strides,
                    Some(workspace.fused_indices.as_mut_slice()),
                    layouts,
                    dst_layout,
                    src_layout,
                    task.single_coefficients()[coefficient],
                    task.storage_conjugate(),
                    dst_data,
                    src_data,
                    alpha.at(layouts.entry(dst_layout).offset),
                    mode,
                )?;
                if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
                    let elapsed = start.elapsed();
                    profile.single_blocks += 1;
                    profile.single_total += elapsed;
                    profile.strided_kernel += elapsed;
                }
            }
            Ok(())
        });
    }

    let start = profile.as_ref().map(|_| std::time::Instant::now());
    let converted = ensure_recoupling_coefficients(workspace, task, structure_identity)?;
    if converted {
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            profile.multi_coefficient_prepare += start.elapsed();
        }
    }

    // Singles apply directly in replay order. Multi blocks are packed through
    // the compile-time recoupling entries, whose order is chosen to form
    // same-shape strided GEMM runs.
    crate::host_pool::with_exec(engine, |mut exec| {
        for block in task.blocks() {
            let TreeTransformBlock::Single {
                dst_layout,
                src_layout,
                coefficient,
            } = *block
            else {
                continue;
            };
            // Timestamps only under profiling: the per-block clock reads are
            // measurable against microsecond replays.
            let start = profile.as_ref().map(|_| std::time::Instant::now());
            tree_transform_single_with_strided_kernel(
                exec.reborrow(),
                kernels,
                &mut workspace.zero_strides,
                Some(workspace.fused_indices.as_mut_slice()),
                layouts,
                dst_layout,
                src_layout,
                task.single_coefficients()[coefficient],
                task.storage_conjugate(),
                dst_data,
                src_data,
                alpha.at(layouts.entry(dst_layout).offset),
                mode,
            )?;
            if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
                let elapsed = start.elapsed();
                profile.single_blocks += 1;
                profile.single_total += elapsed;
                profile.strided_kernel += elapsed;
            }
        }
        Ok(())
    })?;

    // Size the pack scratch for the largest Multi block once, so a fresh
    // per-call workspace allocates each buffer once instead of growing it
    // block by block.
    let mut largest = (0usize, 0usize);
    for (block_index, _) in recoupling_plan.entries() {
        if let TreeTransformBlock::Multi {
            dst_count,
            src_count,
            element_count,
            ..
        } = *recoupling_multi_block(task, block_index)?
        {
            largest.0 = largest.0.max(
                element_count
                    .checked_mul(src_count)
                    .ok_or_else(|| OperationError::ElementCountOverflow)?,
            );
            largest.1 = largest.1.max(
                element_count
                    .checked_mul(dst_count)
                    .ok_or_else(|| OperationError::ElementCountOverflow)?,
            );
        }
    }
    workspace.prepare_packed_buffers(largest.0, largest.1, D::zero());

    for (block_index, job) in recoupling_plan.entries() {
        let TreeTransformBlock::Multi {
            dst_layout_start,
            dst_count,
            src_layout_start,
            src_count,
            element_count,
            ..
        } = *recoupling_multi_block(task, block_index)?
        else {
            unreachable!("recoupling_multi_block only returns Multi blocks");
        };
        debug_assert_eq!(job.rows, element_count);
        debug_assert_eq!(job.contracted, src_count);
        debug_assert_eq!(job.cols, dst_count);
        let source_len = element_count
            .checked_mul(src_count)
            .ok_or_else(|| OperationError::ElementCountOverflow)?;
        let destination_len = element_count
            .checked_mul(dst_count)
            .ok_or_else(|| OperationError::ElementCountOverflow)?;
        let start = profile.as_ref().map(|_| std::time::Instant::now());
        workspace.prepare_packed_buffers(source_len, destination_len, D::zero());
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            profile.multi_workspace_prepare += start.elapsed();
        }
        let start = profile.as_ref().map(|_| std::time::Instant::now());
        crate::host_pool::with_exec(engine, |mut exec| {
            for src_index in 0..src_count {
                pack_layout_into_column(
                    exec.reborrow(),
                    kernels,
                    Some(workspace.fused_indices.as_mut_slice()),
                    layouts,
                    src_layout_start + src_index,
                    src_data,
                    workspace.packed.source_mut().as_mut_slice(),
                    src_index * element_count,
                    task.storage_conjugate(),
                )?;
            }
            Ok(())
        })?;
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            profile.multi_blocks += 1;
            profile.packed_columns += src_count;
            profile.multi_pack += start.elapsed();
        }

        let start = profile.as_ref().map(|_| std::time::Instant::now());
        {
            let local_job = DenseGemmBatchJob {
                dst_offset: 0,
                lhs_offset: 0,
                ..*job
            };
            let (source, destination) = workspace.packed.source_and_destination_mut();
            recoupling_gemm_batch(
                engine,
                dense,
                destination.as_mut_slice(),
                source.as_slice(),
                &workspace.coefficient_scratch,
                core::slice::from_ref(&local_job),
                &[1],
            )?;
        }
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            let elapsed = start.elapsed();
            profile.multi_dense_matmul_call += elapsed;
            profile.multi_matmul_total += elapsed;
        }

        let start = profile.as_ref().map(|_| std::time::Instant::now());
        crate::host_pool::with_exec(engine, |mut exec| {
            for dst_index in 0..dst_count {
                scatter_column_into_layout(
                    exec.reborrow(),
                    kernels,
                    &mut workspace.zero_strides,
                    Some(workspace.fused_indices.as_mut_slice()),
                    layouts,
                    dst_layout_start + dst_index,
                    workspace.packed.destination().as_slice(),
                    dst_index * element_count,
                    dst_data,
                    alpha.at(layouts.entry(dst_layout_start + dst_index).offset),
                    mode,
                )?;
            }
            Ok(())
        })?;
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            profile.scattered_columns += dst_count;
            profile.multi_scatter += start.elapsed();
        }
    }
    Ok(())
}

pub(super) fn parallel_split(items: usize, threads: usize) -> usize {
    let left_threads = threads / 2;
    items
        .saturating_mul(left_threads)
        .div_ceil(threads)
        .clamp(1, items - 1)
}

#[allow(clippy::too_many_arguments)]
fn replay_pack_columns<A, D>(
    mut kernels: A,
    fused_indices: &mut [usize],
    max_fused_rank: usize,
    layouts: &TreeTransformLayoutTable,
    items: &[TreeTransformPackReplay],
    packed_source: &mut [D],
    packed_start: usize,
    src_data: &[D],
    storage_conjugate: bool,
    threads: usize,
    mut exec: Exec<'_>,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D> + Clone + Send + Sync,
    D: DenseRecouplingScalar + ConjugateValue,
{
    if items.is_empty() {
        return Ok(());
    }
    if threads <= 1 || items.len() == 1 {
        let fused_index = &mut fused_indices[..max_fused_rank];
        for item in items {
            pack_layout_into_column(
                exec.reborrow(),
                &mut kernels,
                Some(&mut *fused_index),
                layouts,
                item.src_layout,
                src_data,
                packed_source,
                item.packed_offset - packed_start,
                storage_conjugate,
            )?;
        }
        return Ok(());
    }

    let middle = parallel_split(items.len(), threads);
    let boundary = items[middle].packed_offset;
    let (left_data, right_data) = packed_source.split_at_mut(boundary - packed_start);
    let (left_items, right_items) = items.split_at(middle);
    let left_threads = threads / 2;
    let right_threads = threads - left_threads;
    let (left_indices, mut right_indices) =
        split_lane_indices(fused_indices, left_threads, threads, max_fused_rank);
    let right_kernels = kernels.clone();
    let (left, right) = replay_join(
        exec,
        |exec| {
            replay_pack_columns(
                kernels,
                left_indices,
                max_fused_rank,
                layouts,
                left_items,
                left_data,
                packed_start,
                src_data,
                storage_conjugate,
                left_threads,
                exec,
            )
        },
        |exec| {
            replay_pack_columns(
                right_kernels,
                right_indices.get(),
                max_fused_rank,
                layouts,
                right_items,
                right_data,
                boundary,
                src_data,
                storage_conjugate,
                right_threads,
                exec,
            )
        },
    );
    left?;
    right
}

#[allow(clippy::too_many_arguments)]
pub(super) fn replay_single_blocks<A, D, C>(
    mut kernels: A,
    fused_indices: &mut [usize],
    max_fused_rank: usize,
    blocks: &[TreeTransformBlock],
    layouts: &TreeTransformLayoutTable,
    coefficients: &[C],
    storage_conjugate: bool,
    items: &[u32],
    dst_data: &mut [D],
    dst_start: isize,
    src_data: &[D],
    alpha: DestinationAlpha<'_, D, C>,
    mode: DestinationMode<D>,
    threads: usize,
    mut exec: Exec<'_>,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D> + Clone + Send + Sync,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    if items.is_empty() {
        return Ok(());
    }
    if threads <= 1 || items.len() == 1 {
        let mut zero_strides = Vec::new();
        let fused_index = &mut fused_indices[..max_fused_rank];
        for &item in items {
            let (dst_layout, src_layout, coefficient) = scheduled_single(blocks, item)?;
            let role = layouts.role(dst_layout)?;
            let scale = TransformScale::new(
                alpha.at(layouts.entry(dst_layout).offset),
                coefficients[coefficient],
            );
            kernels.transform_strided_baked(
                exec.reborrow(),
                &mut zero_strides,
                dst_data,
                src_data,
                role.dims(),
                role.dst_strides(),
                role.src_strides(),
                layouts.entry(dst_layout).offset - dst_start,
                layouts.entry(src_layout).offset,
                storage_conjugate,
                scale,
                match mode {
                    DestinationMode::Axpby(beta) => Some(beta),
                    DestinationMode::Overwrite => None,
                },
                Some(role),
                Some(&mut *fused_index),
            )?;
        }
        return Ok(());
    }

    let middle = parallel_split(items.len(), threads);
    let boundary = layouts.destination_lo(scheduled_single(blocks, items[middle])?.0)?;
    let split =
        usize::try_from(boundary - dst_start).map_err(|_| OperationError::ElementCountOverflow)?;
    let (left_data, right_data) = dst_data.split_at_mut(split);
    let (left_items, right_items) = items.split_at(middle);
    let left_threads = threads / 2;
    let right_threads = threads - left_threads;
    let (left_indices, mut right_indices) =
        split_lane_indices(fused_indices, left_threads, threads, max_fused_rank);
    let right_kernels = kernels.clone();
    let (left, right) = replay_join(
        exec,
        |exec| {
            replay_single_blocks(
                kernels,
                left_indices,
                max_fused_rank,
                blocks,
                layouts,
                coefficients,
                storage_conjugate,
                left_items,
                left_data,
                dst_start,
                src_data,
                alpha,
                mode,
                left_threads,
                exec,
            )
        },
        |exec| {
            replay_single_blocks(
                right_kernels,
                right_indices.get(),
                max_fused_rank,
                blocks,
                layouts,
                coefficients,
                storage_conjugate,
                right_items,
                right_data,
                boundary,
                src_data,
                alpha,
                mode,
                right_threads,
                exec,
            )
        },
    );
    left?;
    right
}

/// The `(dst_layout, src_layout, coefficient)` of the Single block a
/// parallel schedule entry names.
#[inline]
pub(super) fn scheduled_single(
    blocks: &[TreeTransformBlock],
    item: u32,
) -> Result<(usize, usize, usize), OperationError> {
    match blocks.get(item as usize) {
        Some(&TreeTransformBlock::Single {
            dst_layout,
            src_layout,
            coefficient,
        }) => Ok((dst_layout, src_layout, coefficient)),
        _ => Err(OperationError::InvalidArgument {
            message: "tree transform single schedule names a non-Single block",
        }),
    }
}

#[allow(clippy::too_many_arguments)]
fn replay_scatter_columns<A, D, C>(
    mut kernels: A,
    fused_indices: &mut [usize],
    max_fused_rank: usize,
    layouts: &TreeTransformLayoutTable,
    items: &[TreeTransformScatterReplay],
    dst_data: &mut [D],
    dst_start: isize,
    packed_destination: &[D],
    packed_start: usize,
    alpha: DestinationAlpha<'_, D, C>,
    mode: DestinationMode<D>,
    threads: usize,
    mut exec: Exec<'_>,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D> + Clone + Send + Sync,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    if items.is_empty() {
        return Ok(());
    }
    if threads <= 1 || items.len() == 1 {
        let fused_index = &mut fused_indices[..max_fused_rank];
        for item in items {
            let role = layouts.role(item.dst_layout)?;
            let offset = layouts.entry(item.dst_layout).offset - dst_start;
            match mode {
                DestinationMode::Axpby(beta) => kernels.axpby_strided_baked(
                    exec.reborrow(),
                    dst_data,
                    packed_destination,
                    role.dims(),
                    role.dst_strides(),
                    role.src_strides(),
                    offset,
                    offset_to_isize(item.packed_offset - packed_start)?,
                    alpha.at(layouts.entry(item.dst_layout).offset),
                    beta,
                    Some(role),
                    Some(&mut *fused_index),
                )?,
                DestinationMode::Overwrite => kernels.copy_scale_strided_baked(
                    exec.reborrow(),
                    dst_data,
                    packed_destination,
                    role.dims(),
                    role.dst_strides(),
                    role.src_strides(),
                    offset,
                    offset_to_isize(item.packed_offset - packed_start)?,
                    false,
                    alpha.at(layouts.entry(item.dst_layout).offset),
                    Some(role),
                    Some(&mut *fused_index),
                )?,
            }
        }
        return Ok(());
    }

    let middle = parallel_split(items.len(), threads);
    let boundary = items[middle].dst_lo;
    let split =
        usize::try_from(boundary - dst_start).map_err(|_| OperationError::ElementCountOverflow)?;
    let (left_data, right_data) = dst_data.split_at_mut(split);
    let (left_items, right_items) = items.split_at(middle);
    let left_threads = threads / 2;
    let right_threads = threads - left_threads;
    let (left_indices, mut right_indices) =
        split_lane_indices(fused_indices, left_threads, threads, max_fused_rank);
    let right_kernels = kernels.clone();
    let (left, right) = replay_join(
        exec,
        |exec| {
            replay_scatter_columns(
                kernels,
                left_indices,
                max_fused_rank,
                layouts,
                left_items,
                left_data,
                dst_start,
                packed_destination,
                packed_start,
                alpha,
                mode,
                left_threads,
                exec,
            )
        },
        |exec| {
            replay_scatter_columns(
                right_kernels,
                right_indices.get(),
                max_fused_rank,
                layouts,
                right_items,
                right_data,
                boundary,
                packed_destination,
                packed_start,
                alpha,
                mode,
                right_threads,
                exec,
            )
        },
    );
    left?;
    right
}

#[allow(clippy::too_many_arguments)]
pub(super) fn replay_scatter_groups<A, D, C>(
    kernels: A,
    fused_indices: &mut [usize],
    max_fused_rank: usize,
    layouts: &TreeTransformLayoutTable,
    scatter_columns: &[TreeTransformScatterReplay],
    scatter_groups: &[TreeTransformScatterGroupReplay],
    groups: &[usize],
    dst_data: &mut [D],
    dst_start: isize,
    packed_destination: &[D],
    packed_start: usize,
    alpha: DestinationAlpha<'_, D, C>,
    mode: DestinationMode<D>,
    threads: usize,
    exec: Exec<'_>,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D> + Clone + Send + Sync,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    if groups.is_empty() {
        return Ok(());
    }
    if threads <= 1 || groups.len() == 1 {
        for &group in groups {
            let columns = scatter_groups[group].columns.clone();
            replay_scatter_columns(
                kernels.clone(),
                fused_indices,
                max_fused_rank,
                layouts,
                &scatter_columns[columns],
                dst_data,
                dst_start,
                packed_destination,
                packed_start,
                alpha,
                mode,
                1,
                Exec::serial(),
            )?;
        }
        return Ok(());
    }

    let middle = parallel_split(groups.len(), threads);
    let boundary = scatter_columns[scatter_groups[groups[middle]].columns.start].dst_lo;
    let split =
        usize::try_from(boundary - dst_start).map_err(|_| OperationError::ElementCountOverflow)?;
    let (left_data, right_data) = dst_data.split_at_mut(split);
    let (left_groups, right_groups) = groups.split_at(middle);
    let left_threads = threads / 2;
    let right_threads = threads - left_threads;
    let (left_indices, mut right_indices) =
        split_lane_indices(fused_indices, left_threads, threads, max_fused_rank);
    let right_kernels = kernels.clone();
    let (left, right) = replay_join(
        exec,
        |exec| {
            replay_scatter_groups(
                kernels,
                left_indices,
                max_fused_rank,
                layouts,
                scatter_columns,
                scatter_groups,
                left_groups,
                left_data,
                dst_start,
                packed_destination,
                packed_start,
                alpha,
                mode,
                left_threads,
                exec,
            )
        },
        |exec| {
            replay_scatter_groups(
                right_kernels,
                right_indices.get(),
                max_fused_rank,
                layouts,
                scatter_columns,
                scatter_groups,
                right_groups,
                right_data,
                boundary,
                packed_destination,
                packed_start,
                alpha,
                mode,
                right_threads,
                exec,
            )
        },
    );
    left?;
    right
}

/// One concurrency-bounded chunk of Multi jobs up to and including its grouped
/// GEMM: rebases the jobs onto chunk scratch, packs the chunk's source columns
/// (in parallel when `threads > 1`), runs the GEMM into
/// `workspace.packed.destination()`, and orders the chunk's non-empty scatter
/// groups by destination offset in `workspace.chunk_scatter_groups`.
///
/// Shared by the initialised parallel replay and the owned uninitialised
/// writer, which differ only in how they scatter the packed destination.
/// Returns the chunk's packed destination base offset and whether the ordered
/// scatter groups may be split by destination slice.
#[allow(clippy::too_many_arguments)]
pub(super) fn replay_multi_chunk<A, E, D, C>(
    engine: &mut dyn Engine,
    kernels: &mut A,
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    task: TreeTransformTaskView<'_, C>,
    schedule: &TreeTransformParallelSchedule,
    first_group: usize,
    chunk: &[DenseGemmBatchJob],
    pack_cursor: &mut usize,
    src_data: &[D],
    threads: usize,
    mut profile: Option<&mut TreeTransformReplayProfile>,
) -> Result<(usize, bool), OperationError>
where
    A: HostKernelAdapter<D> + Clone + Send + Sync,
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    let layouts = task.layouts();
    let max_fused_rank = layouts.max_fused_rank();
    let storage_conjugate = task.storage_conjugate();
    let source_start = chunk[0].lhs_offset;
    let destination_start = chunk[0].dst_offset;
    let source_end = chunk
        .last()
        .and_then(|job| {
            job.rows
                .checked_mul(job.contracted)
                .and_then(|len| job.lhs_offset.checked_add(len))
        })
        .ok_or_else(|| OperationError::ElementCountOverflow)?;
    let destination_end = chunk
        .last()
        .and_then(|job| {
            job.rows
                .checked_mul(job.cols)
                .and_then(|len| job.dst_offset.checked_add(len))
        })
        .ok_or_else(|| OperationError::ElementCountOverflow)?;

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
    workspace.prepare_packed_buffers(
        source_end - source_start,
        destination_end - destination_start,
        D::zero(),
    );
    if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
        profile.multi_workspace_prepare += start.elapsed();
    }

    let pack_start = *pack_cursor;
    while *pack_cursor < schedule.pack_columns.len()
        && schedule.pack_columns[*pack_cursor].packed_offset < source_end
    {
        *pack_cursor += 1;
    }
    let start = profile.as_ref().map(|_| std::time::Instant::now());
    replay_region(engine, threads, |exec| {
        replay_pack_columns(
            kernels.clone(),
            workspace.fused_indices.as_mut_slice(),
            max_fused_rank,
            layouts,
            &schedule.pack_columns[pack_start..*pack_cursor],
            workspace.packed.source_mut().as_mut_slice(),
            source_start,
            src_data,
            storage_conjugate,
            threads,
            exec,
        )
    })??;
    if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
        profile.packed_columns += chunk.iter().map(|job| job.contracted).sum::<usize>();
        profile.multi_pack += start.elapsed();
    }

    let start = profile.as_ref().map(|_| std::time::Instant::now());
    let (source, destination) = workspace.packed.source_and_destination_mut();
    recoupling_gemm_batch(
        engine,
        dense,
        destination.as_mut_slice(),
        source.as_slice(),
        &workspace.coefficient_scratch,
        &workspace.chunk_jobs,
        &workspace.chunk_runs,
    )?;
    if let (Some(profile), Some(start)) = (profile, start) {
        let elapsed = start.elapsed();
        profile.multi_dense_matmul_call += elapsed;
        profile.multi_matmul_total += elapsed;
    }

    workspace.chunk_scatter_groups.clear();
    workspace.chunk_scatter_groups.extend(
        (first_group..first_group + chunk.len())
            .filter(|&group| !schedule.scatter_groups[group].columns.is_empty()),
    );
    // Why not sort/copy every scatter descriptor: only the at-most-T group
    // indices need destination order for one safe replay split tree.
    workspace
        .chunk_scatter_groups
        .sort_unstable_by_key(|&group| {
            schedule.scatter_columns[schedule.scatter_groups[group].columns.start].dst_lo
        });
    let scatter_slice_disjoint = workspace
        .chunk_scatter_groups
        .iter()
        .all(|&group| schedule.scatter_groups[group].slice_disjoint)
        && workspace.chunk_scatter_groups.windows(2).all(|groups| {
            let left_end = schedule.scatter_groups[groups[0]].columns.end;
            let right_start = schedule.scatter_groups[groups[1]].columns.start;
            schedule.scatter_columns[left_end - 1].dst_hi
                < schedule.scatter_columns[right_start].dst_lo
        });
    Ok((destination_start, scatter_slice_disjoint))
}

/// Threaded variant of [`tree_transform_blocks_with_batched_recoupling`]
/// (TensorKit `_add_abelian_kernel_threaded!` / `_add_general_kernel_threaded!`
/// precedent, indexmanipulations.jl:520-738):
///
/// - Singles apply in parallel when their compiled slices are disjoint. Multi
///   blocks replay in concurrency-bounded chunks whose pack and scatter phases
///   each take the engine token once. Work items are independent because the compile step
///   rejects duplicate destination blocks
///   (`OperationError::DuplicateTransformDestination`) and pack columns are
///   disjoint scratch ranges by construction; the workspace forbids `unsafe`,
///   so disjointness is realized structurally by recursively splitting the
///   buffers at compiled boundaries (`split_at_mut`) and rebasing offsets,
///   instead of TensorKit-style shared writes. Interleaved layouts whose
///   bounding slices overlap stay serial: exact element disjointness is not
///   enough to create independent Rust slices without unsafe code.
/// - Each chunk is one serial grouped GEMM call between its pack and scatter
///   phases. The dense executor owns its own parallelism, so no nesting arises.
///
/// Parallel copy scheduling forks the operation's engine token recursively
/// (`replay_join`), capped by the configured worker count. Replay descriptors and safe split
/// boundaries are compiled into the structure.
///
/// Per-task traversal indices are disjoint slices of one workspace arena, split
/// by the same worker budget as the data. Kernel-adapter clones therefore carry
/// configuration only on compiled replay; numerical and chunk-descriptor
/// storage also stays in the reused workspace.
///
/// Profiling attribution is phase-level; per-item clocks across workers would
/// measure contention, not work.
#[allow(clippy::too_many_arguments)]
pub(super) fn tree_transform_blocks_with_batched_recoupling_parallel<A, E, D, C>(
    engine: &mut dyn Engine,
    kernels: &mut A,
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    task: TreeTransformTaskView<'_, C>,
    schedule: &TreeTransformParallelSchedule,
    structure_identity: &Arc<()>,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: DestinationAlpha<'_, D, C>,
    mode: DestinationMode<D>,
    threads: usize,
    mut profile: Option<&mut TreeTransformReplayProfile>,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D> + Clone + Send + Sync,
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    let layouts = task.layouts();
    let recoupling_plan = task.recoupling_plan();
    let max_fused_rank = layouts.max_fused_rank();
    workspace.prepare_fused_indices(threads, max_fused_rank)?;

    let single_count = schedule.single_block_count;
    let multi_count = recoupling_plan.jobs().len();
    let start = profile.as_ref().map(|_| std::time::Instant::now());
    let converted = ensure_recoupling_coefficients(workspace, task, structure_identity)?;
    if converted {
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            profile.multi_coefficient_prepare += start.elapsed();
        }
    }
    if let Some(profile) = profile.as_deref_mut() {
        profile.single_blocks += single_count;
        profile.multi_blocks += multi_count;
    }

    let storage_conjugate = task.storage_conjugate();

    // Single destinations are independent of every Multi destination.
    {
        let start = profile.as_ref().map(|_| std::time::Instant::now());
        if schedule.singles_slice_disjoint {
            replay_region(engine, threads, |exec| {
                replay_single_blocks(
                    kernels.clone(),
                    workspace.fused_indices.as_mut_slice(),
                    max_fused_rank,
                    task.blocks(),
                    layouts,
                    task.single_coefficients(),
                    storage_conjugate,
                    &schedule.singles,
                    dst_data,
                    0,
                    src_data,
                    alpha,
                    mode,
                    threads,
                    exec,
                )
            })??;
        } else {
            let mut zero_strides = Vec::new();
            crate::host_pool::with_exec(engine, |mut exec| {
                for &item in &schedule.singles {
                    let (dst_layout, src_layout, coefficient) =
                        scheduled_single(task.blocks(), item)?;
                    tree_transform_single_with_strided_kernel(
                        exec.reborrow(),
                        kernels,
                        &mut zero_strides,
                        Some(workspace.fused_indices.as_mut_slice()),
                        layouts,
                        dst_layout,
                        src_layout,
                        task.single_coefficients()[coefficient],
                        storage_conjugate,
                        dst_data,
                        src_data,
                        alpha.at(layouts.entry(dst_layout).offset),
                        mode,
                    )?;
                }
                Ok(())
            })?;
        }

        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            let elapsed = start.elapsed();
            profile.single_total += elapsed;
            profile.strided_kernel += elapsed;
        }
    }

    // Why not retain one pack arena for the whole plan: TensorKit owns scratch
    // per concurrently executing fusion block. The existing replay-thread
    // budget is also the natural bound for a grouped dense submission.
    let chunk_size = threads.max(1);
    let mut pack_cursor = 0;
    for (chunk_index, chunk) in recoupling_plan.jobs().chunks(chunk_size).enumerate() {
        let scattered_column_count = chunk.iter().map(|job| job.cols).sum::<usize>();
        let (destination_start, scatter_slice_disjoint) = replay_multi_chunk(
            engine,
            kernels,
            dense,
            workspace,
            task,
            schedule,
            chunk_index * chunk_size,
            chunk,
            &mut pack_cursor,
            src_data,
            threads,
            profile.as_deref_mut(),
        )?;
        let packed_destination = workspace.packed.destination().as_slice();

        let start = profile.as_ref().map(|_| std::time::Instant::now());
        if scatter_slice_disjoint {
            replay_region(engine, threads, |exec| {
                replay_scatter_groups(
                    kernels.clone(),
                    workspace.fused_indices.as_mut_slice(),
                    max_fused_rank,
                    layouts,
                    &schedule.scatter_columns,
                    &schedule.scatter_groups,
                    &workspace.chunk_scatter_groups,
                    dst_data,
                    0,
                    packed_destination,
                    destination_start,
                    alpha,
                    mode,
                    threads,
                    exec,
                )
            })??;
        } else {
            let mut zero_strides = Vec::new();
            crate::host_pool::with_exec(engine, |mut exec| {
                for &group in &workspace.chunk_scatter_groups {
                    for item in
                        &schedule.scatter_columns[schedule.scatter_groups[group].columns.clone()]
                    {
                        scatter_column_into_layout(
                            exec.reborrow(),
                            kernels,
                            &mut zero_strides,
                            Some(workspace.fused_indices.as_mut_slice()),
                            layouts,
                            item.dst_layout,
                            packed_destination,
                            item.packed_offset - destination_start,
                            dst_data,
                            alpha.at(layouts.entry(item.dst_layout).offset),
                            mode,
                        )?;
                    }
                }
                Ok(())
            })?;
        }
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            profile.scattered_columns += scattered_column_count;
            profile.multi_scatter += start.elapsed();
        }
    }
    debug_assert_eq!(pack_cursor, schedule.pack_columns.len());
    Ok(())
}

#[cfg(test)]
mod bounded_workspace_tests {
    use super::*;
    use crate::{StridedHostKernelAdapter, TreeTransformBlockSpec};
    use num_complex::Complex64;

    fn many_group_fixture() -> (Arc<BlockStructure>, TreeTransformStructure<f64>, Vec<f64>) {
        const GROUPS: usize = 32;
        let mut shapes = Vec::with_capacity(2 * GROUPS);
        let mut specs = Vec::with_capacity(GROUPS);
        let mut source = Vec::new();
        for group in 0..GROUPS {
            let elements = group % 8 + 1;
            shapes.push(vec![elements]);
            shapes.push(vec![elements]);
            let first = 2 * group;
            specs.push(TreeTransformBlockSpec::multi(
                vec![first, first + 1],
                vec![first, first + 1],
                vec![1.0, 0.0, 0.0, 1.0],
            ));
            let base = source.len();
            source.extend((0..2 * elements).map(|index| (base + index + 1) as f64));
        }
        let structure = Arc::new(BlockStructure::packed_column_major(1, shapes).unwrap());
        let transform =
            TreeTransformStructure::compile_structures(&structure, &structure, &specs).unwrap();
        (structure, transform, source)
    }

    #[test]
    fn fused_index_arena_is_grow_only_and_reports_overflow() {
        // What: execution scratch retains its largest checked worker/rank arena
        // and reports impossible products without allocating or panicking.
        // Up to the inline rank it holds the calling lane only, at any width:
        // forked lanes keep their own index, so a wide eager call stays inline.
        let mut workspace = TreeTransformWorkspace::<f64>::default();
        workspace.prepare_fused_indices(4, 8).unwrap();
        assert_eq!(workspace.fused_indices.len(), 8);
        assert!(!workspace.fused_indices.spilled());
        workspace.prepare_fused_indices(4, 9).unwrap();
        assert_eq!(workspace.fused_indices.len(), 36);
        workspace.prepare_fused_indices(1, 2).unwrap();
        assert_eq!(workspace.fused_indices.len(), 36);
        assert_eq!(
            workspace.prepare_fused_indices(usize::MAX, 9),
            Err(OperationError::ElementCountOverflow)
        );
        assert_eq!(
            workspace.prepare_fused_indices(usize::MAX / 8, 10),
            Err(OperationError::ElementCountOverflow)
        );
    }

    #[test]
    fn many_groups_bound_serial_and_parallel_pack_scratch_by_concurrency() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let (structure, transform, source) = many_group_fixture();
        let total_source = transform.recoupling_plan().source_len();
        let total_destination = transform.recoupling_plan().destination_len();

        for threads in [1, 3] {
            let mut destination = vec![0.0; source.len()];
            let mut workspace = TreeTransformWorkspace::default();
            tree_transform_structure_with_structural_recoupling_raw(
                engine,
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut workspace,
                &transform,
                &structure,
                &structure,
                &mut destination,
                &source,
                1.0,
                0.0,
                threads,
            )
            .unwrap();

            // What: 32 independent groups replay exactly, while retained pack
            // capacity follows at most `threads` largest groups rather than all.
            assert_eq!(destination, source);
            let (source_capacity, destination_capacity) = workspace.packed_capacities();
            let chunk_source_bound = (threads * 2 * 8).next_power_of_two();
            let chunk_destination_bound = (threads * 2 * 8).next_power_of_two();
            assert!(source_capacity <= chunk_source_bound);
            assert!(destination_capacity <= chunk_destination_bound);
            assert!(source_capacity < total_source);
            assert!(destination_capacity < total_destination);
        }
    }

    #[test]
    fn bounded_group_replay_preserves_complex_alpha_beta() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let (structure, transform, source) = many_group_fixture();
        let source = source
            .into_iter()
            .map(|value| Complex64::new(value, -value))
            .collect::<Vec<_>>();
        let initial = Complex64::new(2.0, -3.0);
        let alpha = Complex64::new(0.5, 0.25);
        let beta = Complex64::new(-0.5, 0.75);
        let mut destination = vec![initial; source.len()];

        tree_transform_structure_with_structural_recoupling_raw(
            engine,
            &mut StridedHostKernelAdapter::default(),
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            &transform,
            &structure,
            &structure,
            &mut destination,
            &source,
            alpha,
            beta,
            3,
        )
        .unwrap();

        // What: chunk boundaries do not change complex recoupling or axpby.
        let expected = source
            .iter()
            .map(|&value| alpha * value + beta * initial)
            .collect::<Vec<_>>();
        assert_eq!(destination, expected);
    }
}
