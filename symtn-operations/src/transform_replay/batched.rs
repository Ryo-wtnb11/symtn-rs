use super::writers::DestinationWriter;
use super::*;
use crate::kernel_adapter::movement::with_conjugation;
use crate::transform_structure::TreeTransformDirectItem;

/// Replays a validated task into `dst` through `writer`, on a route chosen
/// before any write:
///
/// - the lane runner when the engine has more than one slot and placement
///   gives more than one slot, every joint component one job placed on a
///   lane: natively ([`super::lanes::lane_replay`]) when `dense` runs lane
///   GEMMs, otherwise in the barrier form ([`super::lanes::barrier_replay`]),
///   which hands each GEMM to the engine between rounds;
/// - otherwise the one-worker replay ([`replay_serial`]).
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
    profile: Option<&mut TreeTransformReplayProfile>,
) -> Result<(), OperationError>
where
    E: DenseExecutor,
    W: DestinationWriter<T, D>,
    T: Send + Sync,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    refuse_unhonored_policy(engine, dense)?;
    #[cfg(feature = "tenferro")]
    let profile = {
        let mut profile = profile;
        if crate::host_pool::threads(engine) > 1 {
            let lanes = if dense.supports_lane_gemms() {
                super::lanes::lane_replay
            } else {
                super::lanes::barrier_replay
            };
            if lanes(
                engine,
                dense,
                workspace,
                task,
                schedule,
                dst,
                src,
                writer,
                profile.as_deref_mut(),
                1,
            )? {
                return Ok(());
            }
        }
        profile
    };
    replay_serial(
        engine, dense, workspace, task, schedule, dst, src, writer, kernels, profile,
    )
}

/// The one-worker replay: the schedule's direct items (inactive
/// destinations and Singles) in its component order, which is destination
/// address order except where components interleave, then each recoupling
/// job in plan order, its pack, its GEMM on the caller's engine and its
/// scatter, one pack buffer pair reused. Every move runs on one slot. The
/// first failure is returned and nothing after it runs.
///
/// Why the schedule's order and not block order: an owned destination is
/// written for the first time here, and walking it in address order keeps
/// the first-touch write stream sequential; block order scatters it.
///
/// TensorKit `_add_general_kernel_nonthreaded!` (indexmanipulations.jl):
/// Singles apply directly and each Multi block is `destination = source *
/// U^T` over packed columns, here as one dense GEMM per block.
#[allow(clippy::too_many_arguments)]
fn replay_serial<E, W, T, D, C>(
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
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    let layouts = task.layouts();
    let rank = layouts.max_fused_rank();
    workspace.prepare_fused_indices(rank)?;
    let one = core::num::NonZeroUsize::MIN;
    let gate = LeafGate::new(layouts.max_planned_elements());

    let HostTreeTransformWorkspace {
        zero_strides,
        fused_indices,
        ..
    } = &mut *workspace;
    crate::host_pool::with_exec(engine, |exec| {
        let mut exec = exec.limit(one);
        if let Some(profile) = profile.as_deref_mut() {
            profile.single_blocks += schedule.single_block_count;
        }
        for &item in &schedule.direct {
            // Timestamps only under profiling: the per-item clock reads are
            // measurable against microsecond replays.
            let start = profile.as_ref().map(|_| std::time::Instant::now());
            writer.direct(
                kernels,
                zero_strides,
                item,
                src,
                dst,
                0,
                None,
                exec.reborrow(),
                &gate,
                fused_indices,
            )?;
            if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
                let elapsed = start.elapsed();
                if matches!(item, TreeTransformDirectItem::Single { .. }) {
                    profile.single_total += elapsed;
                }
                profile.strided_kernel += elapsed;
            }
        }
        Ok(())
    })?;

    let recoupling_plan = task.recoupling_plan();
    if recoupling_plan.is_empty() {
        return Ok(());
    }
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
            let mut exec = exec.limit(one);
            for dst_index in 0..dst_count {
                writer.scatter(
                    kernels,
                    zero_strides,
                    dst_layout_start + dst_index,
                    packed.destination().as_slice(),
                    dst_index * element_count,
                    dst,
                    0,
                    None,
                    exec.reborrow(),
                    &gate,
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
