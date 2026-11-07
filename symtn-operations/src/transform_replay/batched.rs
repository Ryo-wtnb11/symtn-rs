use super::writers::DestinationWriter;
use super::*;
use crate::kernel_adapter::movement::with_conjugation;
use crate::transform_structure::TreeTransformDirectItem;

/// Replays a validated task into `dst` through `writer` (main's replay):
///
/// - first the direct items, in one engine entry that also fixes the replay
///   width `t` ([`super::chunk::direct`]): the inactive layouts inline, then
///   the Singles, split over `t` workers when they are slice-disjoint and
///   inline otherwise;
/// - then the plan groups: at `t > 1` in chunks of `t` consecutive groups
///   ([`super::chunk::groups`]), each a parallel pack, one grouped GEMM and
///   a scatter split over `t` workers when the chunk's groups are
///   slice-disjoint (inline otherwise); at `t = 1` one group at a time
///   ([`replay_serial_groups`]).
///
/// Every inline move runs on the caller's whole token (one slot under a
/// forced `Sequential`) and is admitted through one caller gate; a split
/// leaf builds its own. The first failure stops the replay.
///
/// Its callers refuse a forced policy `dense` cannot honor first
/// ([`refuse_unhonored_policy`]), before any coefficient preparation.
#[allow(clippy::too_many_arguments)]
pub(super) fn replay<E, W, T, D, C>(
    engine: &mut dyn Engine,
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    task: TreeTransformTaskView<'_, C>,
    schedule: &TreeTransformParallelSchedule,
    dst: &mut [T],
    src: &[D],
    writer: &W,
    kernels: &mut W::Kernels,
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
    let mut gate = LeafGate::new(layouts.max_planned_elements());
    let start = profile.as_ref().map(|_| std::time::Instant::now());
    #[cfg(feature = "tenferro")]
    let (threads, sequential) = super::chunk::direct(
        engine, workspace, task, schedule, dst, src, writer, kernels, &mut gate,
    )?;
    #[cfg(not(feature = "tenferro"))]
    let (threads, sequential) = {
        workspace.prepare_fused_indices(1, layouts.max_fused_rank())?;
        let HostTreeTransformWorkspace {
            zero_strides,
            fused_indices,
            ..
        } = &mut *workspace;
        crate::host_pool::with_exec(engine, |exec| {
            inline_direct(
                exec,
                schedule,
                InlineSingles::ordered(W::FIRST_TOUCH, task, 1),
                dst,
                src,
                writer,
                kernels,
                zero_strides,
                fused_indices,
                &mut gate,
            )
        })?;
        (1, false)
    };
    if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
        let elapsed = start.elapsed();
        profile.single_blocks += schedule.single_block_count;
        profile.single_total += elapsed;
        profile.strided_kernel += elapsed;
    }
    if task.recoupling_plan().is_empty() {
        return Ok(());
    }
    if threads > 1 {
        #[cfg(feature = "tenferro")]
        return super::chunk::groups(
            engine, dense, workspace, task, schedule, dst, src, writer, kernels, &mut gate,
            threads, profile,
        );
    }
    replay_serial_groups(
        engine, dense, workspace, task, dst, src, writer, kernels, &mut gate, sequential, profile,
    )
}

/// The Singles an inline direct replay writes, and their order: main's
/// order for the destination kind and the replay width.
#[derive(Clone, Copy)]
pub(super) enum InlineSingles<'a> {
    /// None: a split writes them.
    None,
    /// The schedule's, in destination address order.
    Schedule,
    /// The non-empty Singles of `blocks`, in block (source) order.
    Blocks(&'a [TreeTransformBlock], &'a TreeTransformLayoutTable),
}

impl<'a> InlineSingles<'a> {
    /// Block order for a caller-provided destination at width one (main's
    /// one-worker replay, `for block in task.blocks()`), the schedule's
    /// address order otherwise: on a first-touch destination (main's
    /// `split_join_uninit`) and at a width above one (main's parallel
    /// replay).
    ///
    /// Why not one order for every destination: address order on the
    /// one-worker caller-provided `permute_into` measured 1.08-1.09x main,
    /// block order 0.99-1.02x; block order on the owned path loses the
    /// owned permute's write-miss reduction that address order gave it.
    pub(super) fn ordered<C: Copy>(
        first_touch: bool,
        task: TreeTransformTaskView<'a, C>,
        threads: usize,
    ) -> Self {
        if first_touch || threads > 1 {
            Self::Schedule
        } else {
            Self::Blocks(task.blocks(), task.layouts())
        }
    }
}

/// The direct items on the caller's token `exec`: the inactive layouts in
/// destination address order, then the `singles`, each through the caller
/// gate.
#[allow(clippy::too_many_arguments)]
pub(super) fn inline_direct<W, T, D>(
    mut exec: Exec<'_>,
    schedule: &TreeTransformParallelSchedule,
    singles: InlineSingles<'_>,
    dst: &mut [T],
    src: &[D],
    writer: &W,
    kernels: &mut W::Kernels,
    zero_strides: &mut Vec<isize>,
    index: &mut [usize],
    gate: &mut LeafGate,
) -> Result<(), OperationError>
where
    W: DestinationWriter<T, D>,
{
    let (scheduled, blocks) = match singles {
        InlineSingles::None => (&[][..], None),
        InlineSingles::Schedule => (&schedule.singles[..], None),
        InlineSingles::Blocks(blocks, layouts) => {
            u32::try_from(blocks.len()).map_err(|_| OperationError::ElementCountOverflow)?;
            (&[][..], Some((blocks, layouts)))
        }
    };
    let inactive = schedule
        .inactive
        .iter()
        .map(|&layout| TreeTransformDirectItem::Inactive {
            layout: layout as usize,
        });
    let scheduled = scheduled
        .iter()
        .map(|&block| TreeTransformDirectItem::Single { block });
    let blocks = blocks.into_iter().flat_map(|(blocks, layouts)| {
        blocks
            .iter()
            .enumerate()
            .filter_map(move |(block, item)| match *item {
                TreeTransformBlock::Single { dst_layout, .. }
                    if layouts.entry(dst_layout).element_count != 0 =>
                {
                    Some(TreeTransformDirectItem::Single {
                        block: block as u32,
                    })
                }
                _ => None,
            })
    });
    // Why one chain, not a loop per order: `direct` is inlined at each call
    // site, and a loop per order made the owned one-worker U1s permute 1.03x
    // slower than one call site.
    for item in inactive.chain(scheduled).chain(blocks) {
        writer.direct(
            kernels,
            zero_strides,
            item,
            src,
            dst,
            0,
            exec.reborrow(),
            gate,
            index,
        )?;
    }
    Ok(())
}

/// The plan groups of a one-worker replay, each in plan order: its pack, its
/// GEMM on the caller's engine and its scatter, one pack buffer pair reused.
/// Every move runs on the caller's whole token, one slot when `sequential`,
/// through the caller gate.
///
/// TensorKit `_add_general_kernel_nonthreaded!` (indexmanipulations.jl):
/// each Multi block is `destination = source * U^T` over packed columns,
/// here as one dense GEMM per block.
#[allow(clippy::too_many_arguments)]
fn replay_serial_groups<E, W, T, D, C>(
    engine: &mut dyn Engine,
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    task: TreeTransformTaskView<'_, C>,
    dst: &mut [T],
    src: &[D],
    writer: &W,
    kernels: &mut W::Kernels,
    gate: &mut LeafGate,
    sequential: bool,
    mut profile: Option<&mut TreeTransformReplayProfile>,
) -> Result<(), OperationError>
where
    E: DenseExecutor,
    W: DestinationWriter<T, D>,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    let layouts = task.layouts();
    let recoupling_plan = task.recoupling_plan();
    // Size the pack scratch for the largest Multi block once, so a fresh
    // per-call workspace allocates each buffer once instead of growing it
    // block by block.
    let (source_len, destination_len) = recoupling_plan.jobs().iter().try_fold(
        (0usize, 0usize),
        |(source, destination), job| {
            let packed = |cols: usize| {
                job.rows
                    .checked_mul(cols)
                    .ok_or(OperationError::ElementCountOverflow)
            };
            Ok::<_, OperationError>((
                source.max(packed(job.contracted)?),
                destination.max(packed(job.cols)?),
            ))
        },
    )?;
    workspace.packed.source_mut().reserve_overwrite(source_len);
    workspace.prepare_packed_destination(destination_len, D::zero());
    let slots = core::num::NonZeroUsize::MIN;

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
        let source_len = element_count * src_count;
        let destination_len = element_count * dst_count;
        let start = profile.as_ref().map(|_| std::time::Instant::now());
        workspace.prepare_packed_destination(destination_len, D::zero());
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            profile.multi_workspace_prepare += start.elapsed();
        }
        let start = profile.as_ref().map(|_| std::time::Instant::now());
        let HostTreeTransformWorkspace {
            packed,
            fused_indices,
            ..
        } = &mut *workspace;
        let storage_conjugate = task.storage_conjugate();
        packed
            .source_mut()
            .overwrite_with_columns(source_len, |mut columns| {
                with_conjugation!(D, storage_conjugate, |map| {
                    for src_index in 0..src_count {
                        let entry = src_layout_start + src_index;
                        columns.write_column(
                            src_index * element_count,
                            layouts.role(entry)?,
                            src,
                            layouts.entry(entry).offset,
                            fused_indices.as_mut_slice(),
                            map,
                        )?;
                    }
                    Ok(())
                })
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
        let HostTreeTransformWorkspace {
            packed,
            zero_strides,
            fused_indices,
            ..
        } = &mut *workspace;
        crate::host_pool::with_exec(engine, |exec| {
            let mut exec = if sequential { exec.limit(slots) } else { exec };
            for dst_index in 0..dst_count {
                writer.scatter(
                    kernels,
                    zero_strides,
                    dst_layout_start + dst_index,
                    packed.destination().as_slice(),
                    dst_index * element_count,
                    dst,
                    0,
                    exec.reborrow(),
                    gate,
                    fused_indices,
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
