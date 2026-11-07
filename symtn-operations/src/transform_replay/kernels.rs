use super::*;

pub(crate) fn validate_replay_storage_len(
    structure: &BlockStructure,
    actual_len: usize,
) -> Result<(), OperationError> {
    let expected = structure
        .required_len()
        .map_err(OperationError::from_core_preserving_context)?;
    if actual_len != expected {
        return Err(OperationError::ElementCountMismatch {
            expected,
            actual: actual_len,
        });
    }
    Ok(())
}

pub(crate) fn zero_tree_transform_destination<A, D>(
    mut exec: crate::host_pool::Exec<'_>,
    kernels: &mut A,
    zero_strides: &mut Vec<isize>,
    dst_structure: &BlockStructure,
    dst_data: &mut [D],
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    D: Copy + Zero + One,
{
    validate_replay_storage_len(dst_structure, dst_data.len())?;
    let zero = [D::zero()];
    let mut dst_strides = Vec::new();
    for block_index in 0..dst_structure.block_count() {
        let block = dst_structure.block(block_index)?;
        zero_strides.clear();
        zero_strides.resize(block.shape().len(), 0);
        dst_strides.clear();
        for &stride in block.strides() {
            dst_strides
                .push(isize::try_from(stride).map_err(|_| OperationError::ElementCountOverflow)?);
        }
        kernels.copy_scale_strided_baked(
            exec.reborrow(),
            dst_data,
            &zero,
            block.shape(),
            &dst_strides,
            zero_strides,
            offset_to_isize(block.offset())?,
            0,
            false,
            D::one(),
            None,
            None,
        )?;
    }
    Ok(())
}

/// One Single block of compiled replay on the caller's token share, into
/// `dst_data` whose first element is destination element `dst_start`: the
/// adapter's planned overwrite when Strided widens the block on that share
/// ([`admitted_plan`]), the strided kernel otherwise.
#[allow(clippy::too_many_arguments)]
pub(super) fn tree_transform_single_on_token<A, D, C>(
    mut exec: crate::host_pool::Exec<'_>,
    gate: &mut LeafGate,
    kernels: &mut A,
    zero_strides: &mut Vec<isize>,
    fused_index: Option<&mut [usize]>,
    layouts: &TreeTransformLayoutTable,
    dst_index: usize,
    src_index: usize,
    coefficient: C,
    source_conjugate: bool,
    dst_data: &mut [D],
    dst_start: isize,
    src_data: &[D],
    alpha: D,
    mode: DestinationMode<D>,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    D: Copy
        + Send
        + Sync
        + Add<D, Output = D>
        + Mul<D, Output = D>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + RecouplingCoefficientAction<C>,
    C: Copy + Sync,
{
    let role = layouts.role(dst_index)?;
    let dst = layouts.entry(dst_index);
    let dst_offset = dst.offset - dst_start;
    let src_offset = layouts.entry(src_index).offset;
    let scale = TransformScale::new(alpha, coefficient);
    if mode.overwrites() {
        if let Some(plan) = admitted_plan(role, dst.element_count, &exec, gate) {
            if let Some(result) = kernels.planned_transform(
                exec.reborrow(),
                plan,
                role,
                dst_data,
                src_data,
                dst_offset,
                src_offset,
                source_conjugate,
                scale,
            ) {
                return result;
            }
        }
    }
    kernels.transform_strided_baked(
        exec,
        zero_strides,
        dst_data,
        src_data,
        role.dims(),
        role.dst_strides(),
        role.src_strides(),
        dst_offset,
        src_offset,
        source_conjugate,
        scale,
        match mode {
            DestinationMode::Axpby(beta) => Some(beta),
            DestinationMode::Overwrite => None,
        },
        Some(role),
        fused_index,
    )
}

#[cfg(any(test, feature = "testing"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn tree_transform_single_with_strided_kernel<A, D, C>(
    exec: crate::host_pool::Exec<'_>,
    kernels: &mut A,
    zero_strides: &mut Vec<isize>,
    fused_index: Option<&mut [usize]>,
    layouts: &TreeTransformLayoutTable,
    dst_index: usize,
    src_index: usize,
    coefficient: C,
    source_conjugate: bool,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    mode: DestinationMode<D>,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    D: Copy + One + PartialEq + RecouplingCoefficientAction<C>,
    C: Copy,
{
    let role = layouts.role(dst_index)?;
    kernels.transform_strided_baked(
        exec,
        zero_strides,
        dst_data,
        src_data,
        role.dims(),
        role.dst_strides(),
        role.src_strides(),
        layouts.entry(dst_index).offset,
        layouts.entry(src_index).offset,
        source_conjugate,
        TransformScale::new(alpha, coefficient),
        match mode {
            DestinationMode::Axpby(beta) => Some(beta),
            DestinationMode::Overwrite => None,
        },
        Some(role),
        fused_index,
    )
}

/// Applies a batch of Multi-block recoupling matrices over shared flat scratch
/// buffers: per job, the column-major
/// (element_count x dst_count) destination block receives `source_block *
/// U^T`, with `recoupling_coefficients_dst_src` (row-major `U[dst, src]`)
/// reinterpreted as the column-major (src_count x dst_count) matrix `U^T`.
/// This is TensorKit's `_add_transform_multi!` `mul!` step submitted as one
/// grouped call; the naive per-element loop in the kernel adapter remains
/// only for adapters without a dense executor. Job offsets are relative to the
/// supplied chunk scratch, matching the trusted-view validation contract.
pub(super) fn recoupling_gemm_batch<E, D>(
    engine: &mut dyn Engine,
    dense: &mut E,
    destination: &mut [D],
    source: &[D],
    coefficients: &[D],
    jobs: &[DenseGemmBatchJob],
    runs: &[usize],
) -> Result<(), OperationError>
where
    E: DenseExecutor,
    D: DenseRecouplingScalar,
{
    let dst_shape = [destination.len()];
    let lhs_shape = [source.len()];
    let rhs_shape = [coefficients.len()];
    let flat_strides = [1];
    let lhs = D::dense_read(symtn_dense::DenseView::new_trusted(
        source,
        &lhs_shape,
        &flat_strides,
        0,
    ));
    let rhs = D::dense_read(symtn_dense::DenseView::new_trusted(
        coefficients,
        &rhs_shape,
        &flat_strides,
        0,
    ));
    let output = D::dense_write(symtn_dense::DenseViewMut::new_trusted(
        destination,
        &dst_shape,
        &flat_strides,
        0,
    ));
    dense
        .matmul_batch_axpby_into(
            engine,
            output,
            lhs,
            rhs,
            jobs,
            runs,
            D::one().dense_scalar(),
            D::zero().dense_scalar(),
        )
        .map_err(OperationError::Dense)
}

#[cfg(any(test, feature = "testing"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn tree_transform_multi_with_pack_gemm_scatter<A, D, C>(
    exec: crate::host_pool::Exec<'_>,
    kernels: &mut A,
    workspace: &mut TreeTransformWorkspace<D>,
    layouts: &TreeTransformLayoutTable,
    dst_layout_start: usize,
    dst_count: usize,
    src_layout_start: usize,
    src_count: usize,
    element_count: usize,
    recoupling_coefficients_dst_src: &[C],
    source_conjugate: bool,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    mode: DestinationMode<D>,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    D: Copy + Zero + One + RecouplingCoefficientAction<C>,
    C: Copy,
{
    let source_len = element_count
        .checked_mul(src_count)
        .ok_or_else(|| OperationError::ElementCountOverflow)?;
    let destination_len = element_count
        .checked_mul(dst_count)
        .ok_or_else(|| OperationError::ElementCountOverflow)?;
    workspace.prepare_packed_buffers(source_len, destination_len, D::zero());
    let HostTreeTransformWorkspace {
        zero_strides,
        packed,
        fused_indices,
        ..
    } = workspace;
    tree_transform_multi_with_scratch_buffers(
        exec,
        kernels,
        zero_strides,
        Some(fused_indices.as_mut_slice()),
        packed,
        layouts,
        dst_layout_start,
        dst_count,
        src_layout_start,
        src_count,
        element_count,
        recoupling_coefficients_dst_src,
        source_conjugate,
        dst_data,
        src_data,
        alpha,
        mode,
    )
}

#[cfg(any(test, feature = "testing"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn tree_transform_multi_with_scratch_buffers<
    A,
    D,
    C,
    SourceScratch,
    DestinationScratch,
>(
    mut exec: crate::host_pool::Exec<'_>,
    kernels: &mut A,
    zero_strides: &mut Vec<isize>,
    mut fused_index: Option<&mut [usize]>,
    scratch: &mut TreeTransformScratchBuffers<SourceScratch, DestinationScratch>,
    layouts: &TreeTransformLayoutTable,
    dst_layout_start: usize,
    dst_count: usize,
    src_layout_start: usize,
    src_count: usize,
    element_count: usize,
    recoupling_coefficients_dst_src: &[C],
    source_conjugate: bool,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    mode: DestinationMode<D>,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    D: Copy + One + RecouplingCoefficientAction<C>,
    C: Copy,
    SourceScratch: HostWritableStorage<D>,
    DestinationScratch: HostWritableStorage<D>,
{
    for src_index in 0..src_count {
        pack_layout_into_column(
            exec.reborrow(),
            kernels,
            fused_index.as_deref_mut(),
            layouts,
            src_layout_start + src_index,
            src_data,
            scratch.source_mut().as_mut_slice(),
            src_index * element_count,
            source_conjugate,
        )?;
    }

    {
        let (source, destination) = scratch.source_and_destination_mut();
        kernels.recoupling_src_times_u_transpose(
            exec.reborrow(),
            destination.as_mut_slice(),
            source.as_slice(),
            recoupling_coefficients_dst_src,
            element_count,
            src_count,
            dst_count,
        )?;
    }

    for dst_index in 0..dst_count {
        scatter_column_into_layout(
            exec.reborrow(),
            kernels,
            zero_strides,
            fused_index.as_deref_mut(),
            layouts,
            dst_layout_start + dst_index,
            scratch.destination().as_slice(),
            dst_index * element_count,
            dst_data,
            alpha,
            mode,
        )?;
    }
    Ok(())
}

#[cfg(any(test, feature = "testing"))]
#[expect(
    clippy::too_many_arguments,
    reason = "the packing kernel keeps layout authority, source and destination slices, offset, and conjugation explicit"
)]
pub(super) fn pack_layout_into_column<A, T>(
    exec: crate::host_pool::Exec<'_>,
    kernels: &mut A,
    fused_index: Option<&mut [usize]>,
    layouts: &TreeTransformLayoutTable,
    entry_index: usize,
    src_data: &[T],
    packed: &mut [T],
    packed_offset: usize,
    source_conjugate: bool,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<T>,
    T: Copy + One,
{
    let role = layouts.role(entry_index)?;
    let packed_offset = offset_to_isize(packed_offset)?;
    kernels.copy_scale_strided_baked(
        exec,
        packed,
        src_data,
        role.dims(),
        role.dst_strides(),
        role.src_strides(),
        packed_offset,
        layouts.entry(entry_index).offset,
        source_conjugate,
        T::one(),
        Some(role),
        fused_index,
    )
}

/// One recoupling scatter column of compiled replay on the caller's token
/// share, into `dst_data` whose first element is destination element
/// `dst_start`: the adapter's planned overwrite when Strided widens the
/// column on that share ([`admitted_plan`]), the strided scatter otherwise.
#[allow(clippy::too_many_arguments)]
pub(super) fn scatter_column_on_token<A, T>(
    mut exec: crate::host_pool::Exec<'_>,
    gate: &mut LeafGate,
    kernels: &mut A,
    zero_strides: &mut Vec<isize>,
    fused_index: Option<&mut [usize]>,
    layouts: &TreeTransformLayoutTable,
    entry_index: usize,
    packed: &[T],
    packed_offset: usize,
    dst_data: &mut [T],
    dst_start: isize,
    alpha: T,
    mode: DestinationMode<T>,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<T>,
    T: Copy
        + Send
        + Sync
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + ConjugateValue,
{
    let role = layouts.role(entry_index)?;
    let dst = layouts.entry(entry_index);
    let dst_offset = dst.offset - dst_start;
    let packed_offset = offset_to_isize(packed_offset)?;
    if mode.overwrites() {
        if let Some(plan) = admitted_plan(role, dst.element_count, &exec, gate) {
            if let Some(result) = kernels.planned_scatter(
                exec.reborrow(),
                plan,
                role,
                dst_data,
                packed,
                dst_offset,
                packed_offset,
                alpha,
            ) {
                return result;
            }
        }
    }
    scatter_role(
        exec,
        kernels,
        zero_strides,
        fused_index,
        role,
        dst_offset,
        packed,
        packed_offset,
        dst_data,
        alpha,
        mode,
    )
}

#[cfg(any(test, feature = "testing"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn scatter_column_into_layout<A, T>(
    exec: crate::host_pool::Exec<'_>,
    kernels: &mut A,
    zero_strides: &mut Vec<isize>,
    fused_index: Option<&mut [usize]>,
    layouts: &TreeTransformLayoutTable,
    entry_index: usize,
    packed: &[T],
    packed_offset: usize,
    dst_data: &mut [T],
    alpha: T,
    mode: DestinationMode<T>,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<T>,
    T: Copy,
{
    let role = layouts.role(entry_index)?;
    scatter_role(
        exec,
        kernels,
        zero_strides,
        fused_index,
        role,
        layouts.entry(entry_index).offset,
        packed,
        offset_to_isize(packed_offset)?,
        dst_data,
        alpha,
        mode,
    )
}

/// A scatter column's strided kernel over its resolved role and offsets.
#[allow(clippy::too_many_arguments)]
pub(super) fn scatter_role<A, T>(
    exec: crate::host_pool::Exec<'_>,
    kernels: &mut A,
    zero_strides: &mut Vec<isize>,
    fused_index: Option<&mut [usize]>,
    role: crate::BakedFusedLayout<'_>,
    offset: isize,
    packed: &[T],
    packed_offset: isize,
    dst_data: &mut [T],
    alpha: T,
    mode: DestinationMode<T>,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<T>,
    T: Copy,
{
    match mode {
        DestinationMode::Axpby(beta) => {
            zero_strides.clear();
            kernels.axpby_strided_baked(
                exec,
                dst_data,
                packed,
                role.dims(),
                role.dst_strides(),
                role.src_strides(),
                offset,
                packed_offset,
                alpha,
                beta,
                Some(role),
                fused_index,
            )
        }
        DestinationMode::Overwrite => kernels.copy_scale_strided_baked(
            exec,
            dst_data,
            packed,
            role.dims(),
            role.dst_strides(),
            role.src_strides(),
            offset,
            packed_offset,
            false,
            alpha,
            Some(role),
            fused_index,
        ),
    }
}

pub(super) fn recoupling_multi_block<C: Copy>(
    task: TreeTransformTaskView<'_, C>,
    block_index: usize,
) -> Result<&TreeTransformBlock, OperationError> {
    // Why not `.ok_or`: this runs per block on the hot replay path, and an
    // eager error value would be built on every success.
    let block =
        task.blocks()
            .get(block_index)
            .ok_or_else(|| OperationError::BlockIndexOutOfBounds {
                tensor: "recoupling block",
                index: block_index,
                count: task.blocks().len(),
            })?;
    match block {
        TreeTransformBlock::Multi { .. } => Ok(block),
        TreeTransformBlock::Single { .. } => Err(OperationError::BlockIndexOutOfBounds {
            tensor: "recoupling block",
            index: block_index,
            count: task.blocks().len(),
        }),
    }
}

#[cfg(any(test, feature = "testing"))]
pub(super) fn scale_inactive_destinations<A, D, C>(
    mut exec: crate::host_pool::Exec<'_>,
    kernels: &mut A,
    zero_strides: &mut Vec<isize>,
    task: TreeTransformTaskView<'_, C>,
    dst_data: &mut [D],
    mode: DestinationMode<D>,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    D: Copy + PartialEq + Zero + One,
    C: Copy,
{
    match mode {
        DestinationMode::Axpby(beta) => {
            if beta == D::one() {
                return Ok(());
            }
            // Scaling the complete storage would also mutate padding not owned by any
            // block, so compile only the destination layouts with no active replay.
            for &layout_index in task.inactive_destination_layouts() {
                let (dims, strides) = task.layouts().inactive_role(layout_index)?;
                kernels.scale_strided(
                    exec.reborrow(),
                    dst_data,
                    dims,
                    strides,
                    task.layouts().entry(layout_index).offset,
                    beta,
                )?;
            }
        }
        DestinationMode::Overwrite => {
            let zero = [D::zero()];
            for &layout_index in task.inactive_destination_layouts() {
                let (dims, strides) = task.layouts().inactive_role(layout_index)?;
                zero_strides.clear();
                zero_strides.resize(dims.len(), 0);
                kernels.copy_scale_strided_baked(
                    exec.reborrow(),
                    dst_data,
                    &zero,
                    dims,
                    strides,
                    zero_strides,
                    task.layouts().entry(layout_index).offset,
                    0,
                    false,
                    D::one(),
                    None,
                    None,
                )?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod allocation_replay_tests {
    use super::*;
    use crate::{StridedHostKernelAdapter, TreeTransformBlockSpec};
    use symtn_dense::{
        DenseDotConfig, DenseError, DenseRead, DenseScalar, DenseTensor, DenseWrite,
    };

    #[derive(Default)]
    struct NoAllocDenseExecutor;

    impl DenseExecutor for NoAllocDenseExecutor {
        fn svd(
            &mut self,
            _engine: &mut dyn Engine,
            _input: DenseRead<'_>,
        ) -> Result<Vec<DenseTensor>, DenseError> {
            unreachable!("tree replay does not call SVD")
        }

        fn qr(
            &mut self,
            _engine: &mut dyn Engine,
            _input: DenseRead<'_>,
        ) -> Result<Vec<DenseTensor>, DenseError> {
            unreachable!("tree replay does not call QR")
        }

        fn eigh(
            &mut self,
            _engine: &mut dyn Engine,
            _input: DenseRead<'_>,
        ) -> Result<Vec<DenseTensor>, DenseError> {
            unreachable!("tree replay does not call EIGH")
        }

        fn dot_general_into(
            &mut self,
            _engine: &mut dyn Engine,
            _output: DenseWrite<'_>,
            _lhs: DenseRead<'_>,
            _rhs: DenseRead<'_>,
            _config: &DenseDotConfig,
        ) -> Result<(), DenseError> {
            unreachable!("tree replay uses the accumulate-form matmul")
        }

        fn matmul_axpby_into(
            &mut self,
            _engine: &mut dyn Engine,
            output: DenseWrite<'_>,
            lhs: DenseRead<'_>,
            rhs: DenseRead<'_>,
            _alpha: DenseScalar,
            _beta: DenseScalar,
        ) -> Result<(), DenseError> {
            let (mut output, lhs, rhs) = match (output, lhs, rhs) {
                (DenseWrite::F64(output), DenseRead::F64(lhs), DenseRead::F64(rhs)) => {
                    (output, lhs, rhs)
                }
                _ => unreachable!("allocation oracle uses f64"),
            };
            let [rows, contracted] = *lhs.shape() else {
                unreachable!("a matrix product")
            };
            let cols = rhs.shape()[1];
            let (output_offset, lhs_offset, rhs_offset) =
                (output.offset(), lhs.offset(), rhs.offset());
            let output_data = output.data_mut();
            for col in 0..cols {
                for row in 0..rows {
                    let mut value = 0.0;
                    for k in 0..contracted {
                        value += lhs.data()[lhs_offset + row + k * rows]
                            * rhs.data()[rhs_offset + k + col * contracted];
                    }
                    output_data[output_offset + row + col * rows] = value;
                }
            }
            Ok(())
        }
    }

    #[test]
    fn warm_custom_executor_replay_on_a_wide_engine_has_no_owned_allocations() {
        let block_structure = Arc::new(
            BlockStructure::packed_column_major(1, [vec![4], vec![4], vec![4], vec![4]]).unwrap(),
        );
        let structure = TreeTransformStructure::compile_structures(
            &block_structure,
            &block_structure,
            &[
                TreeTransformBlockSpec::multi(vec![0, 1], vec![0, 1], vec![1.0, 0.0, 0.0, 1.0]),
                TreeTransformBlockSpec::single(2, 2, 1.0),
                TreeTransformBlockSpec::single(3, 3, -1.0),
            ],
        )
        .unwrap();
        let src = (1..=16).map(f64::from).collect::<Vec<_>>();
        let expected = (1..=16)
            .map(|value| {
                if value <= 12 {
                    f64::from(value)
                } else {
                    -f64::from(value)
                }
            })
            .collect::<Vec<_>>();
        let mut dst = vec![0.0; 16];
        let mut kernels = StridedHostKernelAdapter::default();
        let mut dense = NoAllocDenseExecutor;
        let mut workspace = TreeTransformWorkspace::default();
        let mut pool = crate::host_pool::test_pool(3);

        let mut replay = |engine: &mut dyn Engine| {
            tree_transform_structure_with_structural_recoupling_raw(
                engine,
                &mut kernels,
                &mut dense,
                &mut workspace,
                &structure,
                &block_structure,
                &block_structure,
                &mut dst,
                &src,
                1.0,
                0.0,
            )
            .unwrap();
        };

        pool.install(&mut replay);
        let ((), allocations) =
            pool.install(|engine| crate::counting_alloc::measure(|| replay(engine)));
        assert_eq!(allocations.calls, 0);
        assert_eq!(dst, expected);
    }

    #[test]
    fn warm_custom_executor_overwrite_on_a_wide_engine_has_no_owned_allocations() {
        let block_structure = Arc::new(
            BlockStructure::packed_column_major(1, [vec![4], vec![4], vec![4], vec![4]]).unwrap(),
        );
        let structure = TreeTransformStructure::compile_structures(
            &block_structure,
            &block_structure,
            &[
                TreeTransformBlockSpec::multi(vec![0, 1], vec![0, 1], vec![1.0, 0.0, 0.0, 1.0]),
                TreeTransformBlockSpec::single(2, 2, 1.0),
                TreeTransformBlockSpec::single(3, 3, -1.0),
            ],
        )
        .unwrap();
        let src = (1..=16).map(f64::from).collect::<Vec<_>>();
        let expected = (1..=16)
            .map(|value| {
                if value <= 12 {
                    f64::from(value)
                } else {
                    -f64::from(value)
                }
            })
            .collect::<Vec<_>>();
        let mut dst = vec![f64::NAN; 16];
        let mut kernels = StridedHostKernelAdapter::default();
        let mut dense = NoAllocDenseExecutor;
        let mut workspace = TreeTransformWorkspace::default();
        let mut pool = crate::host_pool::test_pool(3);

        let mut replay = |engine: &mut dyn Engine| {
            dst.fill(f64::NAN);
            tree_transform_structure_overwrite_with_structural_recoupling_raw(
                engine,
                &mut kernels,
                &mut dense,
                &mut workspace,
                &structure,
                &block_structure,
                &block_structure,
                &mut dst,
                &src,
                1.0,
                &[],
            )
            .unwrap();
        };

        pool.install(&mut replay);
        let ((), allocations) =
            pool.install(|engine| crate::counting_alloc::measure(|| replay(engine)));
        assert_eq!(allocations.calls, 0);
        assert_eq!(dst, expected);
    }

    #[test]
    fn warm_wide_overwrite_of_a_rank_nine_single_has_no_owned_allocations() {
        // What: a Single whose permutation leaves nine unfusable axes,
        // above Strided's inline fused rank, replays warm without an
        // allocation on an engine wide enough to widen it, and matches the
        // index oracle `dst(c) = -1.5 * src(reversed c)`.
        const RANK: usize = 9;
        let shape = [2; RANK];
        let block_structure =
            Arc::new(BlockStructure::packed_column_major(RANK, vec![shape.to_vec()]).unwrap());
        let structure = TreeTransformStructure::compile_structures(
            &block_structure,
            &block_structure,
            &[TreeTransformBlockSpec::single(0, 0, -1.5).with_source_axes((0..RANK).rev())],
        )
        .unwrap();
        assert_eq!(structure.layouts().max_fused_rank(), RANK);

        let elements = 1 << RANK;
        let src = (0..elements)
            .map(|value| value as f64 + 0.25)
            .collect::<Vec<_>>();
        let expected = (0..elements)
            .map(|dst_linear| {
                let src_linear = (0..RANK).fold(0usize, |offset, axis| {
                    offset + ((dst_linear >> axis) & 1) * (1 << (RANK - 1 - axis))
                });
                -1.5 * src[src_linear]
            })
            .collect::<Vec<_>>();
        let mut dst = vec![f64::NAN; elements];
        let mut kernels = StridedHostKernelAdapter::default();
        let mut dense = NoAllocDenseExecutor;
        let mut workspace = TreeTransformWorkspace::default();
        let mut pool = crate::host_pool::eager_test_pool(3);

        let mut replay = |engine: &mut dyn Engine| {
            dst.fill(f64::NAN);
            tree_transform_structure_overwrite_with_structural_recoupling_raw(
                engine,
                &mut kernels,
                &mut dense,
                &mut workspace,
                &structure,
                &block_structure,
                &block_structure,
                &mut dst,
                &src,
                1.0,
                &[],
            )
            .unwrap();
        };

        pool.install(&mut replay);
        let ((), allocations) =
            pool.install(|engine| crate::counting_alloc::measure(|| replay(engine)));
        assert_eq!(allocations.calls, 0);
        assert_eq!(dst, expected);
    }

    #[test]
    fn rank_ten_multi_replay_matches_literal_oracle_without_owned_allocations() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        const RANK: usize = 10;
        const GROUPS: usize = 4;
        const BLOCKS: usize = 2 * GROUPS;
        let shape = [2, 2, 2, 2, 2, 2, 2, 2, 2, 1];
        let axes = [8, 7, 6, 5, 4, 3, 2, 1, 0, 9];
        let block_structure = Arc::new(
            BlockStructure::packed_column_major(RANK, vec![shape.to_vec(); BLOCKS]).unwrap(),
        );
        let specs = (0..GROUPS)
            .map(|group| {
                let first = 2 * group;
                TreeTransformBlockSpec::multi(
                    vec![first, first + 1],
                    vec![first, first + 1],
                    vec![1.0, 0.0, 0.0, 1.0],
                )
                .with_source_axes(axes)
            })
            .collect::<Vec<_>>();
        let structure =
            TreeTransformStructure::compile_structures(&block_structure, &block_structure, &specs)
                .unwrap();
        assert!(structure.has_pack_gemm_scatter_blocks());
        assert_eq!(structure.recoupling_plan().jobs().len(), GROUPS);

        let elements = shape.iter().product::<usize>();
        let src = (0..BLOCKS * elements)
            .map(|value| value as f64 + 0.25)
            .collect::<Vec<_>>();
        let strides = block_structure.block(0).unwrap().strides();
        let mut expected = vec![0.0; src.len()];
        for block in 0..BLOCKS {
            let base = block * elements;
            for dst_linear in 0..elements {
                let src_linear = (0..RANK).fold(0usize, |offset, axis| {
                    let coordinate = (dst_linear / strides[axis]) % shape[axis];
                    offset + coordinate * strides[axes[axis]]
                });
                expected[base + dst_linear] = src[base + src_linear];
            }
        }

        let mut serial = vec![0.0; src.len()];
        tree_transform_structure_with_structural_recoupling_raw(
            engine,
            &mut StridedHostKernelAdapter::default(),
            &mut NoAllocDenseExecutor,
            &mut TreeTransformWorkspace::default(),
            &structure,
            &block_structure,
            &block_structure,
            &mut serial,
            &src,
            1.0,
            0.0,
        )
        .unwrap();
        assert_eq!(serial, expected);

        let mut threaded = vec![0.0; src.len()];
        let mut kernels = StridedHostKernelAdapter::default();
        let mut dense = NoAllocDenseExecutor;
        let mut workspace = TreeTransformWorkspace::default();
        let mut pool = crate::host_pool::test_pool(3);
        pool.install(|engine| {
            tree_transform_structure_with_structural_recoupling_raw(
                engine,
                &mut kernels,
                &mut dense,
                &mut workspace,
                &structure,
                &block_structure,
                &block_structure,
                &mut threaded,
                &src,
                1.0,
                0.0,
            )
            .unwrap();
        });

        assert_eq!(threaded, expected);
        threaded.fill(0.0);
        let ((), allocations) = pool.install(|engine| {
            crate::counting_alloc::measure(|| {
                tree_transform_structure_with_structural_recoupling_raw(
                    engine,
                    &mut kernels,
                    &mut dense,
                    &mut workspace,
                    &structure,
                    &block_structure,
                    &block_structure,
                    &mut threaded,
                    &src,
                    1.0,
                    0.0,
                )
                .unwrap();
            })
        });
        assert_eq!(allocations.calls, 0);
        assert_eq!(threaded, expected);
    }
}
