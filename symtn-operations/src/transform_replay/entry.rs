use super::*;

/// Reference oracle: per-block pack, scalar recoupling, scatter. Production
/// replay is the batched-GEMM path in `tree_transform_structure_with_structural_recoupling_raw`.
#[cfg(any(test, feature = "testing"))]
#[expect(
    clippy::too_many_arguments,
    reason = "the raw kernel boundary keeps adapter, workspace, replay structures, buffers, and alpha/beta explicit"
)]
pub fn tree_transform_structure_with_strided_kernel_raw<A, D, C>(
    exec: crate::host_pool::Exec<'_>,
    kernels: &mut A,
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst_structure: &Arc<BlockStructure>,
    src_structure: &Arc<BlockStructure>,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    beta: D,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + Send
        + Sync
        + RecouplingCoefficientAction<C>,
    C: Copy,
{
    tree_transform_structure_with_strided_kernel_raw_mode(
        exec,
        kernels,
        workspace,
        structure,
        dst_structure,
        src_structure,
        dst_data,
        src_data,
        alpha,
        DestinationMode::Axpby(beta),
    )
}

#[cfg(any(test, feature = "testing"))]
#[expect(
    clippy::too_many_arguments,
    reason = "the overwrite kernel boundary keeps adapter, workspace, replay structures, buffers, and alpha explicit"
)]
pub fn tree_transform_structure_overwrite_with_strided_kernel_raw<A, D, C>(
    exec: crate::host_pool::Exec<'_>,
    kernels: &mut A,
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst_structure: &Arc<BlockStructure>,
    src_structure: &Arc<BlockStructure>,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + Send
        + Sync
        + RecouplingCoefficientAction<C>,
    C: Copy,
{
    tree_transform_structure_with_strided_kernel_raw_mode(
        exec,
        kernels,
        workspace,
        structure,
        dst_structure,
        src_structure,
        dst_data,
        src_data,
        alpha,
        DestinationMode::Overwrite,
    )
}

#[cfg(any(test, feature = "testing"))]
#[expect(
    clippy::too_many_arguments,
    reason = "the shared replay helper preserves the raw kernel inputs and destination update mode explicitly"
)]
fn tree_transform_structure_with_strided_kernel_raw_mode<A, D, C>(
    mut exec: crate::host_pool::Exec<'_>,
    kernels: &mut A,
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst_structure: &Arc<BlockStructure>,
    src_structure: &Arc<BlockStructure>,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    mode: DestinationMode<D>,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    D: Copy
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + Send
        + Sync
        + RecouplingCoefficientAction<C>,
    C: Copy,
{
    structure.validate_replay_structures(dst_structure, src_structure)?;
    validate_replay_storage_len(dst_structure, dst_data.len())?;
    validate_replay_storage_len(src_structure, src_data.len())?;
    let task = structure.task_view()?;
    workspace.prepare_fused_indices(1, structure.layouts().max_fused_rank())?;
    scale_inactive_destinations(
        exec.reborrow(),
        kernels,
        &mut workspace.zero_strides,
        task,
        dst_data,
        mode,
    )?;
    for (block_index, block) in structure.blocks().iter().enumerate() {
        match *block {
            TreeTransformBlock::Single {
                dst_layout,
                src_layout,
                coefficient,
            } => tree_transform_single_with_strided_kernel(
                exec.reborrow(),
                kernels,
                &mut workspace.zero_strides,
                Some(workspace.fused_indices.as_mut_slice()),
                structure.layouts(),
                dst_layout,
                src_layout,
                structure.single_coefficients()[coefficient],
                structure.storage_conjugate(),
                dst_data,
                src_data,
                alpha,
                mode,
            )?,
            TreeTransformBlock::Multi {
                dst_layout_start,
                dst_count,
                src_layout_start,
                src_count,
                element_count,
                ..
            } => tree_transform_multi_with_pack_gemm_scatter(
                exec.reborrow(),
                kernels,
                workspace,
                structure.layouts(),
                dst_layout_start,
                dst_count,
                src_layout_start,
                src_count,
                element_count,
                structure.block_matrix(block_index).unwrap_or_default(),
                structure.storage_conjugate(),
                dst_data,
                src_data,
                alpha,
                mode,
            )?,
        }
    }
    Ok(())
}

/// Replays a prepared structural-recoupling tree transform on host slices.
///
/// `threads` selects the replay parallelism (a property of the executing
/// backend, not of the cached structure): `<= 1` reuses one pack buffer and
/// submits each Multi group independently; `> 1` runs Single applies, Multi
/// pack columns and Multi scatter columns as independent work items over up to
/// `threads` work-stealing workers. Multi blocks submit bounded grouped
/// recoupling batches between their pack and scatter phases.
#[allow(clippy::too_many_arguments)]
pub(crate) fn tree_transform_structure_with_structural_recoupling_raw<A, E, D, C>(
    engine: &mut dyn Engine,
    kernels: &mut A,
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst_structure: &Arc<BlockStructure>,
    src_structure: &Arc<BlockStructure>,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    beta: D,
    threads: usize,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D> + Clone + Send + Sync,
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    tree_transform_structure_with_structural_recoupling_raw_mode(
        engine,
        kernels,
        dense,
        workspace,
        structure,
        dst_structure,
        src_structure,
        dst_data,
        src_data,
        DestinationAlpha::uniform(alpha),
        DestinationMode::Axpby(beta),
        threads,
    )
}

/// Overwrite form of the structural-recoupling replay. The Single move or
/// Multi scatter writing destination block `b` uses `alpha * θ_b` when
/// `destination_scales` lists `(b, θ_b)` (strictly increasing block offsets;
/// an unlisted block has `θ_b = 1`); packs, recoupling GEMMs and inactive
/// zero fills stay unscaled, as on the device.
#[allow(clippy::too_many_arguments)]
pub fn tree_transform_structure_overwrite_with_structural_recoupling_raw<A, E, D, C>(
    engine: &mut dyn Engine,
    kernels: &mut A,
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst_structure: &Arc<BlockStructure>,
    src_structure: &Arc<BlockStructure>,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    destination_scales: &[(usize, C)],
    threads: usize,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D> + Clone + Send + Sync,
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    tree_transform_structure_with_structural_recoupling_raw_mode(
        engine,
        kernels,
        dense,
        workspace,
        structure,
        dst_structure,
        src_structure,
        dst_data,
        src_data,
        DestinationAlpha::new(alpha, destination_scales)?,
        DestinationMode::Overwrite,
        threads,
    )
}

#[allow(clippy::too_many_arguments)]
fn tree_transform_structure_with_structural_recoupling_raw_mode<A, E, D, C>(
    engine: &mut dyn Engine,
    kernels: &mut A,
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst_structure: &Arc<BlockStructure>,
    src_structure: &Arc<BlockStructure>,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: DestinationAlpha<'_, D, C>,
    mode: DestinationMode<D>,
    threads: usize,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D> + Clone + Send + Sync,
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    let task = structure.task_view()?;
    task.validate_structures_and_lengths(
        dst_structure,
        src_structure,
        dst_data.len(),
        src_data.len(),
    )?;
    let schedule = structure.parallel_schedule();
    let threads = effective_tree_transform_threads(engine, task, threads)?;
    let requirements = task.workspace_requirements();
    task.validate_workspace_requirements::<D>()?;
    checked_fused_index_len(threads, requirements.fused_index_len_per_worker)?;
    if requirements.converted_coefficient_len != 0 {
        // Admission converts coefficients before the first destination write.
        // The shared serial/parallel helpers call this again, but the
        // structure-identity check makes that call a no-work cache hit.
        ensure_recoupling_coefficients(workspace, task, structure.identity_marker())?;
    }
    replay_initialized(
        engine,
        kernels,
        dense,
        workspace,
        task,
        schedule,
        structure.identity_marker(),
        dst_data,
        src_data,
        alpha,
        mode,
        threads,
        None,
    )
}

/// The initialised replay on the route `dense` selects before any write:
/// the lane route for an executor with lane GEMMs, the staged route
/// otherwise. `threads` is the effective width.
#[allow(clippy::too_many_arguments)]
fn replay_initialized<A, E, D, C>(
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
    profile: Option<&mut TreeTransformReplayProfile>,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D> + Clone + Send + Sync,
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    let writer = InitializedWriter {
        prototype: kernels.clone(),
        task: writers::ReplayParts::new(task),
        src: src_data,
        alpha,
        mode,
    };
    #[cfg(feature = "tenferro")]
    if lane_route(dense) {
        if !task.recoupling_plan().is_empty() {
            ensure_recoupling_coefficients(workspace, task, structure_identity)?;
        }
        return replay_on_lanes(
            engine, dense, workspace, task, schedule, dst_data, src_data, &writer, kernels,
            threads, profile,
        );
    }
    staged_replay(
        engine,
        dense,
        workspace,
        task,
        schedule,
        structure_identity,
        dst_data,
        src_data,
        &writer,
        kernels,
        threads,
        profile,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn tree_transform_structure_with_structural_recoupling_raw_profiled<A, E, D, C>(
    engine: &mut dyn Engine,
    kernels: &mut A,
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst_structure: &Arc<BlockStructure>,
    src_structure: &Arc<BlockStructure>,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    beta: D,
    threads: usize,
    profile: &mut TreeTransformReplayProfile,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D> + Clone + Send + Sync,
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    tree_transform_structure_with_structural_recoupling_raw_profiled_mode(
        engine,
        kernels,
        dense,
        workspace,
        structure,
        dst_structure,
        src_structure,
        dst_data,
        src_data,
        DestinationAlpha::uniform(alpha),
        DestinationMode::Axpby(beta),
        threads,
        profile,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn tree_transform_structure_overwrite_with_structural_recoupling_raw_profiled<
    A,
    E,
    D,
    C,
>(
    engine: &mut dyn Engine,
    kernels: &mut A,
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst_structure: &Arc<BlockStructure>,
    src_structure: &Arc<BlockStructure>,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    destination_scales: &[(usize, C)],
    threads: usize,
    profile: &mut TreeTransformReplayProfile,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D> + Clone + Send + Sync,
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    tree_transform_structure_with_structural_recoupling_raw_profiled_mode(
        engine,
        kernels,
        dense,
        workspace,
        structure,
        dst_structure,
        src_structure,
        dst_data,
        src_data,
        DestinationAlpha::new(alpha, destination_scales)?,
        DestinationMode::Overwrite,
        threads,
        profile,
    )
}

#[allow(clippy::too_many_arguments)]
fn tree_transform_structure_with_structural_recoupling_raw_profiled_mode<A, E, D, C>(
    engine: &mut dyn Engine,
    kernels: &mut A,
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst_structure: &Arc<BlockStructure>,
    src_structure: &Arc<BlockStructure>,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: DestinationAlpha<'_, D, C>,
    mode: DestinationMode<D>,
    threads: usize,
    profile: &mut TreeTransformReplayProfile,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D> + Clone + Send + Sync,
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    let total_start = std::time::Instant::now();

    let start = std::time::Instant::now();
    let task = structure.task_view()?;
    task.validate_structures_and_lengths(
        dst_structure,
        src_structure,
        dst_data.len(),
        src_data.len(),
    )?;

    let schedule = structure.parallel_schedule();
    let threads = effective_tree_transform_threads(engine, task, threads)?;
    let requirements = task.workspace_requirements();
    task.validate_workspace_requirements::<D>()?;
    checked_fused_index_len(threads, requirements.fused_index_len_per_worker)?;
    profile.validate += start.elapsed();

    if requirements.converted_coefficient_len != 0 {
        let start = std::time::Instant::now();
        // Attribute the admission conversion here; the replay helper's second
        // call is the same no-work identity check as in the unprofiled path.
        if ensure_recoupling_coefficients(workspace, task, structure.identity_marker())? {
            profile.multi_coefficient_prepare += start.elapsed();
        }
    }

    replay_initialized(
        engine,
        kernels,
        dense,
        workspace,
        task,
        schedule,
        structure.identity_marker(),
        dst_data,
        src_data,
        alpha,
        mode,
        threads,
        Some(profile),
    )?;

    profile.total += total_start.elapsed();
    Ok(())
}
