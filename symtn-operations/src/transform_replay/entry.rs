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

/// Replays a prepared structural-recoupling tree transform on host slices:
/// `dst = alpha * T(src) + beta * dst`.
///
/// The engine's width and policy fix the route before any write: the direct
/// items, then the Multi groups in chunks of the replay width, or one at a
/// time on one worker (see [`replay`]).
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
        None,
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
        None,
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
    profile: &mut TreeTransformReplayProfile,
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
        Some(profile),
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
    profile: &mut TreeTransformReplayProfile,
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
        Some(profile),
    )
}

/// The shared body of the four entries above: the refusal of a forced policy
/// `dense` cannot honor, validation, the coefficient pack, and the replay on
/// its route.
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
    mut profile: Option<&mut TreeTransformReplayProfile>,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D> + Clone + Send + Sync,
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    refuse_unhonored_policy(engine, dense)?;
    let total_start = profile.as_ref().map(|_| std::time::Instant::now());
    let task = structure.task_view()?;
    task.validate_structures_and_lengths(
        dst_structure,
        src_structure,
        dst_data.len(),
        src_data.len(),
    )?;
    let requirements = task.workspace_requirements();
    task.validate_workspace_requirements::<D>()?;
    checked_fused_index_len(1, requirements.fused_index_len_per_worker)?;
    if let (Some(profile), Some(start)) = (profile.as_deref_mut(), total_start) {
        profile.validate += start.elapsed();
    }
    if requirements.converted_coefficient_len != 0 {
        // Converted before the first destination write.
        let start = profile.as_ref().map(|_| std::time::Instant::now());
        if ensure_recoupling_coefficients(workspace, task, structure.identity_marker())? {
            if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
                profile.multi_coefficient_prepare += start.elapsed();
            }
        }
    }
    let writer = InitializedWriter {
        prototype: kernels.clone(),
        task: writers::ReplayParts::new(task),
        alpha,
        mode,
    };
    replay(
        engine,
        dense,
        workspace,
        task,
        structure.parallel_schedule(),
        dst_data,
        src_data,
        &writer,
        kernels,
        profile.as_deref_mut(),
    )?;
    if let (Some(profile), Some(start)) = (profile, total_start) {
        profile.total += start.elapsed();
    }
    Ok(())
}
