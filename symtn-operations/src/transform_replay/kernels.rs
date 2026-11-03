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

/// One Single block of compiled replay on the caller's token share: the
/// adapter's planned overwrite when `gate` (built from that share) widens the
/// block, the strided kernel otherwise. `dst_data` may be a part of the
/// destination starting at absolute offset `dst_start`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn tree_transform_single_on_token<A, D, C>(
    mut exec: crate::host_pool::Exec<'_>,
    gate: &LeafGate,
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
/// share: the adapter's planned overwrite when `gate` (built from that share)
/// widens the column, the strided scatter otherwise. `dst_data` may be a part
/// of the destination starting at absolute offset `dst_start`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn scatter_column_on_token<A, T>(
    mut exec: crate::host_pool::Exec<'_>,
    gate: &LeafGate,
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
    T: Copy + Send + Sync + Add<T, Output = T> + Mul<T, Output = T> + PartialEq + Zero + One,
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
#[inline]
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
    // Lazy error construction: recoupling_multi_block is called per block on the
    // hot replay path (pack/recouple/scatter). Eager .ok_or built the
    // BlockIndexOutOfBounds struct on every success too, which the d=4 bisect
    // (see issue #103) attributed to the compose regression. .ok_or_else only
    // builds it on the never-taken out-of-bounds path.
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
            unreachable!("tree replay uses the batched matmul entry point")
        }

        fn matmul_batch_axpby_into(
            &mut self,
            _engine: &mut dyn Engine,
            output: DenseWrite<'_>,
            lhs: DenseRead<'_>,
            rhs: DenseRead<'_>,
            jobs: &[DenseGemmBatchJob],
            _runs: &[usize],
            _alpha: DenseScalar,
            _beta: DenseScalar,
        ) -> Result<(), DenseError> {
            let (mut output, lhs, rhs) = match (output, lhs, rhs) {
                (DenseWrite::F64(output), DenseRead::F64(lhs), DenseRead::F64(rhs)) => {
                    (output, lhs, rhs)
                }
                _ => unreachable!("allocation oracle uses f64"),
            };
            let output_offset = output.offset();
            let lhs_offset = lhs.offset();
            let rhs_offset = rhs.offset();
            let output_data = output.data_mut();
            for job in jobs {
                for col in 0..job.cols {
                    for row in 0..job.rows {
                        let mut value = 0.0;
                        for contracted in 0..job.contracted {
                            value += lhs.data()
                                [lhs_offset + job.lhs_offset + row + contracted * job.rows]
                                * rhs.data()[rhs_offset
                                    + job.rhs_offset
                                    + contracted
                                    + col * job.contracted];
                        }
                        output_data[output_offset + job.dst_offset + row + col * job.rows] = value;
                    }
                }
            }
            Ok(())
        }
    }

    #[test]
    fn warm_threaded_replay_has_no_owned_allocations() {
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
                3,
            )
            .unwrap();
        };

        pool.install(&mut replay);
        let (joins, allocations) = allocation_oracle::with_session(|| {
            pool.install(|engine| {
                allocation_oracle::with_measurement(|| {
                    replay(engine);
                    allocation_oracle::join_entries()
                })
            })
        });
        assert_eq!(allocations, 0);
        assert!(joins > 0);
        assert_eq!(dst, expected);
    }

    #[test]
    fn multi_pack_halves_its_token_evenly() {
        // What: the Multi pack's column split halves its token evenly at
        // each fork, so on three slots over three lanes the two-lane side
        // holds one slot and packs inline: one dispatched fork, where a
        // lane-proportional share (one slot left, two right) would dispatch
        // a second. One Multi group of three blocks has no Single phase and
        // one scatter component, so the pack makes every join.
        let block_structure =
            Arc::new(BlockStructure::packed_column_major(1, vec![vec![1]; 3]).unwrap());
        let matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let structure = TreeTransformStructure::compile_structures(
            &block_structure,
            &block_structure,
            &[TreeTransformBlockSpec::multi(
                vec![0, 1, 2],
                vec![0, 1, 2],
                matrix,
            )],
        )
        .unwrap();
        assert_eq!(structure.parallel_schedule().pack_columns.len(), 3);
        let src = [1.0, 2.0, 3.0];
        let mut dst = [0.0; 3];
        let mut pool = crate::host_pool::test_pool(3);
        let (dispatched, _) = allocation_oracle::with_session(|| {
            pool.install(|engine| {
                allocation_oracle::with_measurement(|| {
                    tree_transform_structure_with_structural_recoupling_raw(
                        engine,
                        &mut StridedHostKernelAdapter::default(),
                        &mut crate::transform_replay::tests::Staged::new(),
                        &mut TreeTransformWorkspace::default(),
                        &structure,
                        &block_structure,
                        &block_structure,
                        &mut dst,
                        &src,
                        1.0,
                        0.0,
                        3,
                    )
                    .unwrap();
                    allocation_oracle::dispatched_joins()
                })
            })
        });
        assert_eq!(dispatched, 1);
        // dst_i = sum_j U[i][j] src_j for the row-major `matrix`.
        assert_eq!(dst, [14.0, 32.0, 50.0]);
    }

    #[test]
    fn warm_threaded_overwrite_replay_has_no_owned_allocations() {
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
                3,
            )
            .unwrap();
        };

        pool.install(&mut replay);
        let (_, allocations) = allocation_oracle::with_session(|| {
            pool.install(|engine| allocation_oracle::with_measurement(|| replay(engine)))
        });
        assert_eq!(allocations, 0);
        assert_eq!(dst, expected);
    }

    #[test]
    fn warm_component_replay_has_no_owned_allocations() {
        // What: once a workspace has replayed a structure whose Multi groups
        // own disconnected atoms, a threaded replay splits the destination
        // into those atoms again without allocating.
        let (structure, transform) = super::super::tests::su2_interleaved_fixture();
        assert!(transform.parallel_schedule().component_count() > 1);
        let src = (1..=60).map(f64::from).collect::<Vec<_>>();
        let expected = super::super::tests::su2_oracle(&src, 1.0);
        let mut dst = vec![f64::NAN; 60];
        let mut kernels = StridedHostKernelAdapter::default();
        let mut dense = NoAllocDenseExecutor;
        let mut workspace = TreeTransformWorkspace::default();
        let mut pool = crate::host_pool::test_pool(3);

        let mut replay = |engine: &mut dyn Engine| {
            tree_transform_structure_overwrite_with_structural_recoupling_raw(
                engine,
                &mut kernels,
                &mut dense,
                &mut workspace,
                &transform,
                &structure,
                &structure,
                &mut dst,
                &src,
                1.0,
                &[],
                3,
            )
            .unwrap();
        };

        pool.install(&mut replay);
        let (joins, allocations) = allocation_oracle::with_session(|| {
            pool.install(|engine| {
                allocation_oracle::with_measurement(|| {
                    replay(engine);
                    allocation_oracle::join_entries()
                })
            })
        });
        assert_eq!(allocations, 0);
        assert!(joins > 0);
        assert_eq!(dst, expected);
    }

    /// The native executor whose lane GEMMs each run on one slot of their
    /// lane token; placement still sees the native times.
    ///
    /// Why not the lane's whole token: on x86_64 a GEMM given two slots or
    /// more takes faer's team kernel, which allocates its job tables on
    /// every call outside the engine workspace (tensor4all/faer#223), so the
    /// row would measure that dependency rather than the replay.
    struct OneSlotLaneGemms(DefaultDenseExecutor);

    struct OneSlot<'g>(&'g dyn symtn_dense::LaneGemms);

    impl symtn_dense::LaneGemms for OneSlot<'_> {
        fn run<'d>(
            &self,
            token: symtn_dense::LaneToken<'_>,
            job: usize,
            io: symtn_dense::LaneGemmIo<'_, 'd>,
        ) -> Result<(), DenseError> {
            self.0
                .run(token.limit(core::num::NonZeroUsize::MIN), job, io)
        }

        fn time(
            &self,
            token: &mut symtn_dense::LaneToken<'_>,
            job: usize,
            width: core::num::NonZeroUsize,
        ) -> std::time::Duration {
            self.0.time(token, job, width)
        }
    }

    impl DenseExecutor for OneSlotLaneGemms {
        fn svd(
            &mut self,
            engine: &mut dyn Engine,
            input: DenseRead<'_>,
        ) -> Result<Vec<DenseTensor>, DenseError> {
            self.0.svd(engine, input)
        }

        fn qr(
            &mut self,
            engine: &mut dyn Engine,
            input: DenseRead<'_>,
        ) -> Result<Vec<DenseTensor>, DenseError> {
            self.0.qr(engine, input)
        }

        fn eigh(
            &mut self,
            engine: &mut dyn Engine,
            input: DenseRead<'_>,
        ) -> Result<Vec<DenseTensor>, DenseError> {
            self.0.eigh(engine, input)
        }

        fn dot_general_into(
            &mut self,
            engine: &mut dyn Engine,
            output: DenseWrite<'_>,
            lhs: DenseRead<'_>,
            rhs: DenseRead<'_>,
            config: &DenseDotConfig,
        ) -> Result<(), DenseError> {
            self.0.dot_general_into(engine, output, lhs, rhs, config)
        }

        fn supports_lane_gemms(&self) -> bool {
            true
        }

        fn with_lane_gemms(
            &mut self,
            engine: &mut dyn Engine,
            dtype: symtn_dense::DenseDType,
            jobs: &[symtn_dense::LaneGemmShape],
            body: &mut symtn_dense::LaneGemmBody<'_>,
        ) -> Result<(), DenseError> {
            self.0
                .with_lane_gemms(engine, dtype, jobs, &mut |token, gemms| {
                    body(token, &OneSlot(gemms))
                })
        }
    }

    #[test]
    #[cfg_attr(
        any(
            feature = "cpu-blas",
            feature = "blas-accelerate",
            feature = "blas-openblas",
            feature = "blas-mkl",
            feature = "provider-inject"
        ),
        ignore = "lane GEMMs are the native provider's; a BLAS provider runs the staged route"
    )]
    fn warm_lane_replay_allocates_nothing_on_any_lane() {
        // What: once a workspace and an executor with lane GEMMs have
        // replayed a structure on an engine, a second identical call
        // allocates nothing on any lane (the lanes' GEMMs warm their engine
        // workspace slots in the first call), at one worker and banded on
        // three: an initialised overwrite, an initialised overwrite whose
        // groups have distinct GEMM geometries, and an owned replay, which
        // allocates its one output.
        let (structure, transform) = super::super::tests::su2_interleaved_fixture();
        let src = (1..=60).map(f64::from).collect::<Vec<_>>();
        let expected = super::super::tests::su2_oracle(&src, 1.0);
        // Groups of one to four elements per block: four geometries.
        let mixed = Arc::new(
            BlockStructure::packed_column_major(1, (1..=4).flat_map(|n| [vec![n], vec![n]]))
                .unwrap(),
        );
        let mixed_transform = TreeTransformStructure::compile_structures(
            &mixed,
            &mixed,
            &(0..4)
                .map(|g| {
                    TreeTransformBlockSpec::multi(
                        vec![2 * g, 2 * g + 1],
                        vec![2 * g, 2 * g + 1],
                        vec![0.0, 1.0, 1.0, 0.0],
                    )
                })
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let mixed_src = (1..=20).map(f64::from).collect::<Vec<_>>();
        let mut mixed_expected = Vec::new();
        for (start, n) in [(0, 1), (2, 2), (6, 3), (12, 4)] {
            mixed_expected.extend_from_slice(&mixed_src[start + n..start + 2 * n]);
            mixed_expected.extend_from_slice(&mixed_src[start..start + n]);
        }
        for (index, mut pool) in [
            crate::host_pool::test_pool(1),
            crate::host_pool::test_pool(3),
            crate::host_pool::eager_test_pool(3),
        ]
        .into_iter()
        .enumerate()
        {
            let mut dst = vec![f64::NAN; 60];
            let mut mixed_dst = vec![f64::NAN; 20];
            let mut kernels = StridedHostKernelAdapter::default();
            let mut dense = OneSlotLaneGemms(DefaultDenseExecutor::new());
            let mut workspace = TreeTransformWorkspace::default();
            let mut replay = |engine: &mut dyn Engine| {
                tree_transform_structure_overwrite_with_structural_recoupling_raw(
                    engine,
                    &mut kernels,
                    &mut dense,
                    &mut workspace,
                    &transform,
                    &structure,
                    &structure,
                    &mut dst,
                    &src,
                    1.0,
                    &[],
                    3,
                )
                .unwrap();
                tree_transform_structure_overwrite_with_structural_recoupling_raw(
                    engine,
                    &mut kernels,
                    &mut dense,
                    &mut workspace,
                    &mixed_transform,
                    &mixed,
                    &mixed,
                    &mut mixed_dst,
                    &mixed_src,
                    1.0,
                    &[],
                    3,
                )
                .unwrap();
            };
            pool.install(&mut replay);
            let (_, allocations) = allocation_oracle::with_session(|| {
                pool.install(|engine| allocation_oracle::with_measurement(|| replay(engine)))
            });
            assert_eq!(allocations, 0, "pool #{index}");
            assert_eq!(dst, expected);
            assert_eq!(mixed_dst, mixed_expected);

            let mut owned_workspace = TreeTransformWorkspace::default();
            let mut owned = |engine: &mut dyn Engine| {
                try_tree_transform_structure_overwrite_owned_raw(
                    engine,
                    &mut dense,
                    &mut owned_workspace,
                    &transform,
                    &structure,
                    &structure,
                    2,
                    &src,
                    1.0,
                    3,
                    None,
                )
                .unwrap()
                .unwrap()
            };
            pool.install(&mut owned);
            let (result, allocations) = allocation_oracle::with_session(|| {
                pool.install(|engine| allocation_oracle::with_measurement(|| owned(engine)))
            });
            assert_eq!(allocations, 1, "pool #{index}: the owned output only");
            assert_eq!(result, expected);
        }
    }

    #[test]
    fn lane_scratch_meets_each_buffers_requirement_and_is_charged_by_capacity() {
        // What: two sequential components whose groups need `(rows k, rows
        // n)` of `(200, 2)` and `(2, 200)` reserve 200 elements in each of
        // the two packed buffers (not 202 in all), the retained charge
        // covers both capacities, and a later smaller call keeps and still
        // charges them; the values are `U * source`.
        let sources = 101;
        let structure =
            Arc::new(BlockStructure::packed_column_major(1, vec![vec![2]; 101]).unwrap());
        let transform = TreeTransformStructure::compile_structures(
            &structure,
            &structure,
            &[
                TreeTransformBlockSpec::multi(vec![0], (0..100).collect(), vec![1.0; 100]),
                TreeTransformBlockSpec::multi(
                    (1..101).collect(),
                    vec![100],
                    (1..=100).map(f64::from).collect(),
                ),
            ],
        )
        .unwrap();
        let src = (0..2 * sources).map(|v| v as f64).collect::<Vec<_>>();
        let mut expected = vec![0.0; 2 * sources];
        for row in 0..2 {
            expected[row] = (0..100).map(|block| src[2 * block + row]).sum();
            for block in 1..101 {
                expected[2 * block + row] = block as f64 * src[200 + row];
            }
        }
        let mut workspace = TreeTransformWorkspace::default();
        let mut dense = DefaultDenseExecutor::new();
        let engine = &mut symtn_dense::CpuEngine::serial();
        let mut dst = vec![f64::NAN; 2 * sources];
        tree_transform_structure_overwrite_with_structural_recoupling_raw(
            engine,
            &mut StridedHostKernelAdapter::default(),
            &mut dense,
            &mut workspace,
            &transform,
            &structure,
            &structure,
            &mut dst,
            &src,
            1.0,
            &[],
            1,
        )
        .unwrap();
        assert_eq!(dst, expected);
        let (source, destination) = workspace.packed_capacities();
        assert!(source >= 200 && destination >= 200);
        let charged = workspace.retained_bytes();
        assert!(charged >= (source + destination) * std::mem::size_of::<f64>());

        let small = Arc::new(BlockStructure::packed_column_major(1, vec![vec![2]; 2]).unwrap());
        let small_transform = TreeTransformStructure::compile_structures(
            &small,
            &small,
            &[TreeTransformBlockSpec::multi(
                vec![0, 1],
                vec![0, 1],
                vec![1.0, 0.0, 0.0, 1.0],
            )],
        )
        .unwrap();
        let small_src = [1.0, 2.0, 3.0, 4.0];
        let mut small_dst = [f64::NAN; 4];
        tree_transform_structure_overwrite_with_structural_recoupling_raw(
            engine,
            &mut StridedHostKernelAdapter::default(),
            &mut dense,
            &mut workspace,
            &small_transform,
            &small,
            &small,
            &mut small_dst,
            &small_src,
            1.0,
            &[],
            1,
        )
        .unwrap();
        assert_eq!(small_dst, small_src);
        assert_eq!(workspace.packed_capacities(), (source, destination));
        assert!(workspace.retained_bytes() >= (source + destination) * std::mem::size_of::<f64>());
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
                3,
            )
            .unwrap();
        };

        pool.install(&mut replay);
        let (_, allocations) = allocation_oracle::with_session(|| {
            pool.install(|engine| allocation_oracle::with_measurement(|| replay(engine)))
        });
        assert_eq!(allocations, 0);
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
            1,
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
                3,
            )
            .unwrap();
        });

        assert_eq!(threaded, expected);
        threaded.fill(0.0);
        let (_, allocations) = allocation_oracle::with_session(|| {
            pool.install(|engine| {
                allocation_oracle::with_measurement(|| {
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
                        3,
                    )
                    .unwrap();
                })
            })
        });
        assert_eq!(allocations, 0);
        assert_eq!(threaded, expected);
    }
}
