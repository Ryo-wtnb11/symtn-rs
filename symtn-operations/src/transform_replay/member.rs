use super::*;

fn checked_len<D>(len: usize) -> Result<isize, OperationError> {
    core::alloc::Layout::array::<D>(len).map_err(|_| OperationError::ElementCountOverflow)?;
    offset_to_isize(len)
}

fn checked_view(
    offset: isize,
    shape: &[usize],
    strides: &[isize],
    member_len: usize,
) -> Result<(), OperationError> {
    if shape.len() != strides.len() {
        return Err(OperationError::InvalidArgument {
            message: "member transform layout rank mismatch",
        });
    }
    if shape.contains(&0) {
        return Ok(());
    }
    let (mut lo, mut hi) = (offset, offset);
    for (&extent, &stride) in shape.iter().zip(strides) {
        let span = isize::try_from(extent - 1)
            .map_err(|_| OperationError::ElementCountOverflow)?
            .checked_mul(stride)
            .ok_or(OperationError::ElementCountOverflow)?;
        if span < 0 {
            lo = lo
                .checked_add(span)
                .ok_or(OperationError::ElementCountOverflow)?;
        } else {
            hi = hi
                .checked_add(span)
                .ok_or(OperationError::ElementCountOverflow)?;
        }
    }
    if lo < 0 || hi >= checked_len::<u8>(member_len)? {
        return Err(OperationError::InvalidArgument {
            message: "member transform layout exceeds its member",
        });
    }
    Ok(())
}

fn checked_range(start: usize, len: usize, total: usize) -> Result<(), OperationError> {
    if start.checked_add(len).is_some_and(|end| end <= total) {
        Ok(())
    } else {
        Err(OperationError::InvalidArgument {
            message: "member transform packed job exceeds its buffer",
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn admit<D, C>(
    task: TreeTransformTaskView<'_, C>,
    dst_structure: &Arc<BlockStructure>,
    src_structure: &Arc<BlockStructure>,
    dst_len: usize,
    src_len: usize,
    members: usize,
    destinations: &mut Vec<(usize, usize)>,
) -> Result<(), OperationError>
where
    D: DenseRecouplingScalar,
    C: Copy,
{
    if members == 0 {
        return Err(OperationError::InvalidArgument {
            message: "member transform requires at least one member",
        });
    }
    offset_to_isize(members)?;
    task.validate_structures(dst_structure, src_structure)?;
    let dst_member = dst_structure.required_len()?;
    let src_member = src_structure.required_len()?;
    task.validate_lengths(dst_member, src_member)?;
    for (actual, per_member) in [(dst_len, dst_member), (src_len, src_member)] {
        let expected = per_member
            .checked_mul(members)
            .ok_or(OperationError::ElementCountOverflow)?;
        checked_len::<D>(expected)?;
        if actual != expected {
            return Err(OperationError::ElementCountMismatch { expected, actual });
        }
    }
    // Compilation already proves destination block injectivity. Exact
    // structure admission above carries that proof to every member; the
    // checked member stride keeps their physical ranges separate.
    let layouts = task.layouts();
    let plan = task.recoupling_plan();
    task.validate_workspace_requirements::<D>()?;
    checked_len::<D>(
        plan.source_len()
            .checked_mul(members)
            .ok_or(OperationError::ElementCountOverflow)?,
    )?;
    checked_len::<D>(
        plan.destination_len()
            .checked_mul(members)
            .ok_or(OperationError::ElementCountOverflow)?,
    )?;
    core::alloc::Layout::array::<DenseGemmBatchJob>(
        plan.jobs()
            .len()
            .checked_mul(members)
            .ok_or(OperationError::ElementCountOverflow)?,
    )
    .map_err(|_| OperationError::ElementCountOverflow)?;
    // Every view below is a compiled normalized role: it addresses exactly
    // the raw block's element set, so its minimum and maximum, and hence
    // these bounds, are the raw layout's.
    for &index in task.inactive_destination_layouts() {
        let (dims, strides) = layouts.inactive_role(index)?;
        checked_view(layouts.entry(index).offset, dims, strides, dst_member)?;
    }
    for block in task.blocks() {
        if let TreeTransformBlock::Single {
            dst_layout,
            src_layout,
            coefficient,
        } = *block
        {
            let role = layouts.role(dst_layout)?;
            checked_view(
                layouts.entry(dst_layout).offset,
                role.dims(),
                role.dst_strides(),
                dst_member,
            )?;
            checked_view(
                layouts.entry(src_layout).offset,
                role.dims(),
                role.src_strides(),
                src_member,
            )?;
            if coefficient >= task.single_coefficients().len() {
                return Err(OperationError::CoefficientCountMismatch {
                    expected: coefficient + 1,
                    actual: task.single_coefficients().len(),
                });
            }
        }
    }
    destinations.clear();
    destinations.reserve(plan.jobs().len());
    for (block_index, job) in plan.entries() {
        let &TreeTransformBlock::Multi {
            dst_layout_start,
            dst_count,
            src_layout_start,
            src_count,
            element_count,
            coefficient_start,
            ..
        } = recoupling_multi_block(task, block_index)?
        else {
            unreachable!()
        };
        if (job.rows, job.contracted, job.cols) != (element_count, src_count, dst_count) {
            return Err(OperationError::InvalidArgument {
                message: "member transform job geometry mismatch",
            });
        }
        let src_span = element_count
            .checked_mul(src_count)
            .ok_or(OperationError::ElementCountOverflow)?;
        let dst_span = element_count
            .checked_mul(dst_count)
            .ok_or(OperationError::ElementCountOverflow)?;
        let coefficient_span = src_count
            .checked_mul(dst_count)
            .ok_or(OperationError::ElementCountOverflow)?;
        checked_range(job.lhs_offset, src_span, plan.source_len())?;
        checked_range(job.dst_offset, dst_span, plan.destination_len())?;
        checked_range(job.rhs_offset, coefficient_span, plan.coefficient_len())?;
        checked_range(coefficient_start, coefficient_span, task.coefficient_len())?;
        destinations.push((job.dst_offset, job.dst_offset + dst_span));
        for column in 0..src_count {
            let entry = src_layout_start + column;
            let role = layouts.role(entry)?;
            checked_view(
                layouts.entry(entry).offset,
                role.dims(),
                role.src_strides(),
                src_member,
            )?;
            checked_view(0, role.dims(), role.dst_strides(), element_count)?;
            checked_range(
                job.lhs_offset + column * element_count,
                element_count,
                plan.source_len(),
            )?;
        }
        for column in 0..dst_count {
            let entry = dst_layout_start + column;
            let role = layouts.role(entry)?;
            checked_view(
                layouts.entry(entry).offset,
                role.dims(),
                role.dst_strides(),
                dst_member,
            )?;
            checked_view(0, role.dims(), role.src_strides(), element_count)?;
            checked_range(
                job.dst_offset + column * element_count,
                element_count,
                plan.destination_len(),
            )?;
        }
    }
    destinations.sort_unstable();
    if destinations.windows(2).any(|pair| pair[0].1 > pair[1].0) {
        return Err(OperationError::InvalidArgument {
            message: "member transform packed destinations overlap",
        });
    }
    Ok(())
}

/// Read-only admission for a member replay. It validates bounds and prepares
/// fallible coefficient conversion before the caller starts any stage.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn admit_tree_transform_members_overwrite_raw<D, C>(
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst_structure: &Arc<BlockStructure>,
    src_structure: &Arc<BlockStructure>,
    dst_len: usize,
    src_len: usize,
    members: usize,
) -> Result<(), OperationError>
where
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    C: Copy,
{
    let task = structure.task_view()?;
    admit::<D, C>(
        task,
        dst_structure,
        src_structure,
        dst_len,
        src_len,
        members,
        &mut workspace.member_ranges,
    )?;
    ensure_recoupling_coefficients(workspace, task, structure.identity_marker())?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn move_members<A, D>(
    exec: crate::host_pool::Exec<'_>,
    kernels: &mut A,
    shape: &mut Vec<usize>,
    dst_strides: &mut Vec<isize>,
    src_strides: &mut Vec<isize>,
    dst: &mut [D],
    src: &[D],
    local_shape: &[usize],
    local_dst_strides: &[isize],
    local_src_strides: &[isize],
    dst_member_stride: isize,
    src_member_stride: isize,
    dst_offset: isize,
    src_offset: isize,
    members: usize,
    conjugate: bool,
    scale: D,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    D: Copy,
{
    shape.clear();
    shape.extend_from_slice(local_shape);
    shape.push(members);
    dst_strides.clear();
    dst_strides.extend_from_slice(local_dst_strides);
    dst_strides.push(dst_member_stride);
    src_strides.clear();
    src_strides.extend_from_slice(local_src_strides);
    src_strides.push(src_member_stride);
    kernels.copy_scale_strided_baked(
        exec,
        dst,
        src,
        shape,
        dst_strides,
        src_strides,
        dst_offset,
        src_offset,
        conjugate,
        scale,
        None,
        None,
    )
}

/// Overwrite a fixed-stride Host batch with one completed tree transform.
/// Each member uses the exact structure passed at compilation; the workspace
/// owns only execution scratch and can be reused at a different batch size.
/// `threads` preserves the ordinary structural schedule for `members == 1`;
/// larger batches submit member-expanded dense jobs together without an outer
/// member thread pool. `destination_scales` folds `θ_b` into the moves
/// writing destination block `b` of every member, as in
/// [`tree_transform_structure_overwrite_with_structural_recoupling_raw`].
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn tree_transform_members_overwrite_raw<A, E, D, C>(
    engine: &mut dyn Engine,
    kernels: &mut A,
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst_structure: &Arc<BlockStructure>,
    src_structure: &Arc<BlockStructure>,
    dst_data: &mut [D],
    src_data: &[D],
    members: usize,
    threads: usize,
    destination_scales: &[(usize, C)],
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D> + Clone + Send + Sync,
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    if members == 1 {
        // The ordinary entry retains its structural parallel schedule and
        // already admits the single member before writing its destination.
        return tree_transform_structure_overwrite_with_structural_recoupling_raw(
            engine,
            kernels,
            dense,
            workspace,
            structure,
            dst_structure,
            src_structure,
            dst_data,
            src_data,
            D::one(),
            destination_scales,
            threads,
        );
    }
    let alpha = DestinationAlpha::new(D::one(), destination_scales)?;
    let task = structure.task_view()?;
    admit::<D, C>(
        task,
        dst_structure,
        src_structure,
        dst_data.len(),
        src_data.len(),
        members,
        &mut workspace.member_ranges,
    )?;
    ensure_recoupling_coefficients(workspace, task, structure.identity_marker())?;
    let layouts = task.layouts();
    let plan = task.recoupling_plan();
    let src_stride = offset_to_isize(src_structure.required_len()?)?;
    let dst_stride = offset_to_isize(dst_structure.required_len()?)?;
    let packed_src_stride = offset_to_isize(plan.source_len())?;
    let packed_dst_stride = offset_to_isize(plan.destination_len())?;
    let packed_src_len = plan.source_len() * members;
    let packed_dst_len = plan.destination_len() * members;
    workspace.prepare_packed_buffers(packed_src_len, packed_dst_len, D::zero());
    workspace.chunk_jobs.clear();
    for (_, job) in plan.entries() {
        for member in 0..members {
            workspace.chunk_jobs.push(DenseGemmBatchJob {
                dst_offset: member * plan.destination_len() + job.dst_offset,
                lhs_offset: member * plan.source_len() + job.lhs_offset,
                ..*job
            });
        }
    }
    strided_batch_runs_into(&workspace.chunk_jobs, &mut workspace.chunk_runs);
    let (shape, dst_strides, src_strides) = (
        &mut workspace.member_shape,
        &mut workspace.member_dst_strides,
        &mut workspace.member_src_strides,
    );
    // Host moves before and after the one grouped GEMM, each phase on the
    // caller engine's token.
    crate::host_pool::with_exec(engine, |mut exec| {
        for &index in task.inactive_destination_layouts() {
            let (dims, strides) = layouts.inactive_role(index)?;
            let zero = [D::zero()];
            workspace.zero_strides.clear();
            workspace.zero_strides.resize(dims.len(), 0);
            move_members(
                exec.reborrow(),
                kernels,
                shape,
                dst_strides,
                src_strides,
                dst_data,
                &zero,
                dims,
                strides,
                &workspace.zero_strides,
                dst_stride,
                0,
                layouts.entry(index).offset,
                0,
                members,
                false,
                D::one(),
            )?;
        }
        for block in task.blocks() {
            let TreeTransformBlock::Single {
                dst_layout,
                src_layout,
                coefficient,
            } = *block
            else {
                continue;
            };
            let role = layouts.role(dst_layout)?;
            let dst = layouts.entry(dst_layout);
            let src = layouts.entry(src_layout);
            shape.clear();
            shape.extend_from_slice(role.dims());
            shape.push(members);
            dst_strides.clear();
            dst_strides.extend_from_slice(role.dst_strides());
            dst_strides.push(dst_stride);
            src_strides.clear();
            src_strides.extend_from_slice(role.src_strides());
            src_strides.push(src_stride);
            kernels.transform_strided_baked(
                exec.reborrow(),
                &mut workspace.zero_strides,
                dst_data,
                src_data,
                shape,
                dst_strides,
                src_strides,
                dst.offset,
                src.offset,
                task.storage_conjugate(),
                TransformScale::new(
                    alpha.at(dst.offset),
                    task.single_coefficients()[coefficient],
                ),
                None,
                None,
                None,
            )?;
        }
        for (block_index, job) in plan.entries() {
            let &TreeTransformBlock::Multi {
                src_layout_start,
                src_count,
                element_count,
                ..
            } = recoupling_multi_block(task, block_index)?
            else {
                unreachable!()
            };
            for column in 0..src_count {
                let entry = src_layout_start + column;
                let role = layouts.role(entry)?;
                move_members(
                    exec.reborrow(),
                    kernels,
                    shape,
                    dst_strides,
                    src_strides,
                    workspace.packed.source_mut().as_mut_slice(),
                    src_data,
                    role.dims(),
                    role.dst_strides(),
                    role.src_strides(),
                    packed_src_stride,
                    src_stride,
                    offset_to_isize(job.lhs_offset + column * element_count)?,
                    layouts.entry(entry).offset,
                    members,
                    task.storage_conjugate(),
                    D::one(),
                )?;
            }
        }
        Ok(())
    })?;
    if !workspace.chunk_jobs.is_empty() {
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
    }
    crate::host_pool::with_exec(engine, |mut exec| {
        for (block_index, job) in plan.entries() {
            let &TreeTransformBlock::Multi {
                dst_layout_start,
                dst_count,
                element_count,
                ..
            } = recoupling_multi_block(task, block_index)?
            else {
                unreachable!()
            };
            for column in 0..dst_count {
                let entry = dst_layout_start + column;
                let role = layouts.role(entry)?;
                move_members(
                    exec.reborrow(),
                    kernels,
                    shape,
                    dst_strides,
                    src_strides,
                    dst_data,
                    workspace.packed.destination().as_slice(),
                    role.dims(),
                    role.dst_strides(),
                    role.src_strides(),
                    dst_stride,
                    packed_dst_stride,
                    layouts.entry(entry).offset,
                    offset_to_isize(job.dst_offset + column * element_count)?,
                    members,
                    false,
                    alpha.at(layouts.entry(entry).offset),
                )?;
            }
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{StridedHostKernelAdapter, TreeTransformBlockSpec};
    use num_complex::Complex64;
    use symtn_dense::{
        DenseDotConfig, DenseError, DenseRead, DenseScalar, DenseTensor, DenseWrite,
    };

    fn fixture() -> (Arc<BlockStructure>, TreeTransformStructure<f64>) {
        let blocks = Arc::new(
            BlockStructure::packed_column_major(1, [vec![1], vec![1], vec![1], vec![1]]).unwrap(),
        );
        let transform = TreeTransformStructure::compile_structures(
            &blocks,
            &blocks,
            &[
                TreeTransformBlockSpec::multi(vec![0, 1], vec![0, 1], vec![1.0, 2.0, 3.0, 4.0]),
                TreeTransformBlockSpec::single(2, 2, -2.0),
            ],
        )
        .unwrap();
        (blocks, transform)
    }

    struct CountingDense {
        inner: DefaultDenseExecutor,
        submissions: usize,
        jobs: usize,
    }

    impl Default for CountingDense {
        fn default() -> Self {
            Self {
                inner: DefaultDenseExecutor::new(),
                submissions: 0,
                jobs: 0,
            }
        }
    }

    impl CountingDense {
        fn one_thread_faer() -> Self {
            Self {
                inner: DefaultDenseExecutor::new(),
                submissions: 0,
                jobs: 0,
            }
        }
    }

    impl DenseExecutor for CountingDense {
        fn svd(
            &mut self,
            engine: &mut dyn Engine,
            input: DenseRead<'_>,
        ) -> Result<Vec<DenseTensor>, DenseError> {
            self.inner.svd(engine, input)
        }
        fn qr(
            &mut self,
            engine: &mut dyn Engine,
            input: DenseRead<'_>,
        ) -> Result<Vec<DenseTensor>, DenseError> {
            self.inner.qr(engine, input)
        }
        fn eigh(
            &mut self,
            engine: &mut dyn Engine,
            input: DenseRead<'_>,
        ) -> Result<Vec<DenseTensor>, DenseError> {
            self.inner.eigh(engine, input)
        }
        fn dot_general_into(
            &mut self,
            engine: &mut dyn Engine,
            output: DenseWrite<'_>,
            lhs: DenseRead<'_>,
            rhs: DenseRead<'_>,
            config: &DenseDotConfig,
        ) -> Result<(), DenseError> {
            self.inner
                .dot_general_into(engine, output, lhs, rhs, config)
        }
        fn matmul_batch_axpby_into(
            &mut self,
            engine: &mut dyn Engine,
            output: DenseWrite<'_>,
            lhs: DenseRead<'_>,
            rhs: DenseRead<'_>,
            jobs: &[DenseGemmBatchJob],
            runs: &[usize],
            alpha: DenseScalar,
            beta: DenseScalar,
        ) -> Result<(), DenseError> {
            self.submissions += 1;
            self.jobs += jobs.len();
            self.inner
                .matmul_batch_axpby_into(engine, output, lhs, rhs, jobs, runs, alpha, beta)
        }
    }

    #[test]
    fn member_overwrite_matches_hand_recoupling_and_clears_inactive() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let (blocks, transform) = fixture();
        let mut workspace = TreeTransformWorkspace::<f64>::default();
        for members in [1, 2, 17, 2] {
            let source: Vec<f64> = (0..members)
                .flat_map(|member| {
                    let base = member as f64 + 1.0;
                    [base, base + 2.0, base + 4.0, base + 6.0]
                })
                .collect();
            let mut destination = vec![f64::NAN; source.len()];
            let mut dense = CountingDense::default();
            tree_transform_members_overwrite_raw(
                engine,
                &mut StridedHostKernelAdapter::default(),
                &mut dense,
                &mut workspace,
                &transform,
                &blocks,
                &blocks,
                &mut destination,
                &source,
                members,
                2,
                &[],
            )
            .unwrap();
            assert_eq!(dense.submissions, 1);
            assert_eq!(dense.jobs, members);
            if members > 1 {
                let plan = transform.recoupling_plan();
                let base = plan.jobs()[0];
                assert_eq!(workspace.chunk_jobs.len(), members);
                for (member, job) in workspace.chunk_jobs.iter().enumerate() {
                    assert_eq!(
                        job.dst_offset,
                        member * plan.destination_len() + base.dst_offset
                    );
                    assert_eq!(job.lhs_offset, member * plan.source_len() + base.lhs_offset);
                    assert_eq!(job.rhs_offset, base.rhs_offset);
                }
                assert!(workspace
                    .chunk_jobs
                    .windows(2)
                    .all(|jobs| jobs[0].dst_offset + jobs[0].rows * jobs[0].cols
                        <= jobs[1].dst_offset));
            }
            for member in 0..members {
                let [a, b, c, _] = source[4 * member..][..4].try_into().unwrap();
                assert_eq!(
                    &destination[4 * member..4 * member + 4],
                    &[a + 2.0 * b, 3.0 * a + 4.0 * b, -2.0 * c, 0.0]
                );
            }
        }
    }

    #[test]
    fn complex_members_keep_real_structural_scale_and_grouped_rhs() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let (blocks, transform) = fixture();
        let source = [
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, -1.0),
            Complex64::new(5.0, 4.0),
            Complex64::new(7.0, 1.0),
            Complex64::new(2.0, -3.0),
            Complex64::new(-1.0, 5.0),
            Complex64::new(4.0, 2.0),
            Complex64::new(6.0, 2.0),
        ];
        let mut dst = vec![Complex64::new(f64::NAN, f64::NAN); source.len()];
        let mut dense = CountingDense::default();
        tree_transform_members_overwrite_raw(
            engine,
            &mut StridedHostKernelAdapter::default(),
            &mut dense,
            &mut TreeTransformWorkspace::default(),
            &transform,
            &blocks,
            &blocks,
            &mut dst,
            &source,
            2,
            1,
            &[],
        )
        .unwrap();
        assert_eq!(dense.submissions, 1);
        assert_eq!(dense.jobs, 2);
        for member in 0..2 {
            let a = source[4 * member];
            let b = source[4 * member + 1];
            let c = source[4 * member + 2];
            let expected = [
                a + 2.0 * b,
                3.0 * a + 4.0 * b,
                -2.0 * c,
                Complex64::new(0.0, 0.0),
            ];
            assert_eq!(&dst[4 * member..4 * member + 4], &expected);
        }
    }

    #[test]
    fn short_last_member_rejects_before_any_write_or_dense_submission() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let (blocks, transform) = fixture();
        let source = [1.0; 7];
        let mut dst = [f64::NAN; 8];
        let mut dense = CountingDense::default();
        let error = tree_transform_members_overwrite_raw(
            engine,
            &mut StridedHostKernelAdapter::default(),
            &mut dense,
            &mut TreeTransformWorkspace::default(),
            &transform,
            &blocks,
            &blocks,
            &mut dst,
            &source,
            2,
            1,
            &[],
        )
        .unwrap_err();
        assert!(matches!(
            error,
            OperationError::ElementCountMismatch {
                expected: 8,
                actual: 7
            }
        ));
        assert!(dst.iter().all(|value| value.is_nan()));
        assert_eq!(dense.submissions, 0);
    }

    #[test]
    fn heterogeneous_multi_blocks_share_one_dense_submission() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let blocks = Arc::new(
            BlockStructure::packed_column_major(
                1,
                [vec![2], vec![2], vec![1], vec![1], vec![1], vec![1]],
            )
            .unwrap(),
        );
        let transform = TreeTransformStructure::compile_structures(
            &blocks,
            &blocks,
            &[
                TreeTransformBlockSpec::multi(vec![0, 1], vec![0, 1], vec![1.0, 2.0, -1.0, 3.0]),
                TreeTransformBlockSpec::multi(vec![2, 3], vec![2, 3], vec![2.0, -1.0, 4.0, 1.0]),
                TreeTransformBlockSpec::single(4, 4, 5.0),
            ],
        )
        .unwrap();
        let members = 17;
        let source: Vec<f64> = (0..members)
            .flat_map(|member| {
                let t = member as f64;
                [
                    1.0 + t,
                    2.0 + t,
                    3.0 + t,
                    4.0 + t,
                    5.0 + t,
                    6.0 + t,
                    7.0 + t,
                    8.0 + t,
                ]
            })
            .collect();
        let mut destination = vec![f64::NAN; source.len()];
        let mut dense = CountingDense::default();
        tree_transform_members_overwrite_raw(
            engine,
            &mut StridedHostKernelAdapter::default(),
            &mut dense,
            &mut TreeTransformWorkspace::default(),
            &transform,
            &blocks,
            &blocks,
            &mut destination,
            &source,
            members,
            1,
            &[],
        )
        .unwrap();
        assert_eq!(dense.submissions, 1);
        assert_eq!(dense.jobs, 2 * members);
        for member in 0..members {
            let [a, b, c, d, e, f, g, _] = source[8 * member..8 * member + 8].try_into().unwrap();
            assert_eq!(
                &destination[8 * member..8 * member + 8],
                &[
                    a + 2.0 * c,
                    b + 2.0 * d,
                    -a + 3.0 * c,
                    -b + 3.0 * d,
                    2.0 * e - f,
                    4.0 * e + f,
                    5.0 * g,
                    0.0,
                ]
            );
        }
    }

    #[test]
    fn conjugated_complex_source_uses_the_task_view_flag() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let blocks = Arc::new(BlockStructure::packed_column_major(1, [vec![1]]).unwrap());
        let transform = TreeTransformStructure::compile_structures_with_storage_conjugation(
            &blocks,
            &blocks,
            &[TreeTransformBlockSpec::single(0, 0, -2.0_f64)],
            true,
        )
        .unwrap();
        let source = [Complex64::new(1.0, 2.0), Complex64::new(-3.0, 4.0)];
        let mut destination = [Complex64::new(f64::NAN, f64::NAN); 2];
        tree_transform_members_overwrite_raw(
            engine,
            &mut StridedHostKernelAdapter::default(),
            &mut CountingDense::default(),
            &mut TreeTransformWorkspace::default(),
            &transform,
            &blocks,
            &blocks,
            &mut destination,
            &source,
            2,
            1,
            &[],
        )
        .unwrap();
        assert_eq!(
            destination,
            [Complex64::new(-2.0, 4.0), Complex64::new(6.0, 8.0)]
        );
    }

    #[test]
    fn conjugated_complex_multi_pack_matches_hand_matrix_action() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let blocks = Arc::new(BlockStructure::packed_column_major(1, [vec![1], vec![1]]).unwrap());
        let transform = TreeTransformStructure::compile_structures_with_storage_conjugation(
            &blocks,
            &blocks,
            &[TreeTransformBlockSpec::multi(
                vec![0, 1],
                vec![0, 1],
                vec![1.0_f64, 2.0, 3.0, 4.0],
            )],
            true,
        )
        .unwrap();
        let source = [
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, -1.0),
            Complex64::new(-2.0, 4.0),
            Complex64::new(5.0, 3.0),
        ];
        let mut destination = [Complex64::new(f64::NAN, f64::NAN); 4];
        let mut dense = CountingDense::default();
        tree_transform_members_overwrite_raw(
            engine,
            &mut StridedHostKernelAdapter::default(),
            &mut dense,
            &mut TreeTransformWorkspace::default(),
            &transform,
            &blocks,
            &blocks,
            &mut destination,
            &source,
            2,
            1,
            &[],
        )
        .unwrap();
        assert_eq!(dense.submissions, 1);
        assert_eq!(dense.jobs, 2);
        for member in 0..2 {
            let a = source[2 * member].conj();
            let b = source[2 * member + 1].conj();
            assert_eq!(
                &destination[2 * member..2 * member + 2],
                &[a + 2.0 * b, 3.0 * a + 4.0 * b]
            );
        }
    }

    #[test]
    fn invalid_batch_count_rejects_before_writes() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let (blocks, transform) = fixture();
        let mut destination = [f64::NAN; 8];
        let mut dense = CountingDense::default();
        for members in [0, usize::MAX] {
            assert!(tree_transform_members_overwrite_raw(
                engine,
                &mut StridedHostKernelAdapter::default(),
                &mut dense,
                &mut TreeTransformWorkspace::default(),
                &transform,
                &blocks,
                &blocks,
                &mut destination,
                &[1.0; 8],
                members,
                1,
                &[],
            )
            .is_err());
        }
        assert!(destination.iter().all(|value| value.is_nan()));
        assert_eq!(dense.submissions, 0);
    }

    #[test]
    #[ignore = "benchmark: run via benchmarks.yml"]
    fn member_batch_measurement() {
        use std::time::Instant;

        let (blocks, transform) = fixture();
        for members in [1, 2, 17] {
            let source: Vec<f64> = (0..members)
                .flat_map(|index| [index as f64 + 1.0, 2.0, 3.0, 4.0])
                .collect();
            let mut batch_destination = vec![f64::NAN; source.len()];
            let mut ordinary_destination = batch_destination.clone();
            let mut batch_workspace = TreeTransformWorkspace::<f64>::default();
            let mut ordinary_workspace = TreeTransformWorkspace::<f64>::default();
            let mut batch_dense = CountingDense::one_thread_faer();
            let mut ordinary_dense = CountingDense::one_thread_faer();
            let mut batch_engine = symtn_dense::CpuEngine::serial();
            let mut ordinary_engine = symtn_dense::CpuEngine::serial();
            let mut batch = || {
                tree_transform_members_overwrite_raw(
                    &mut batch_engine,
                    &mut StridedHostKernelAdapter::default(),
                    &mut batch_dense,
                    &mut batch_workspace,
                    &transform,
                    &blocks,
                    &blocks,
                    &mut batch_destination,
                    &source,
                    members,
                    1,
                    &[],
                )
                .unwrap();
            };
            let mut ordinary = || {
                for member in 0..members {
                    tree_transform_structure_overwrite_with_structural_recoupling_raw(
                        &mut ordinary_engine,
                        &mut StridedHostKernelAdapter::default(),
                        &mut ordinary_dense,
                        &mut ordinary_workspace,
                        &transform,
                        &blocks,
                        &blocks,
                        &mut ordinary_destination[4 * member..4 * member + 4],
                        &source[4 * member..4 * member + 4],
                        1.0,
                        &[],
                        1,
                    )
                    .unwrap();
                }
            };
            let measure = |work: &mut dyn FnMut()| {
                allocation_oracle::with_session(|| {
                    let start = Instant::now();
                    allocation_oracle::with_measurement(work);
                    (start.elapsed(), allocation_oracle::allocated_bytes())
                })
            };
            let ((batch_cold, batch_cold_bytes), batch_cold_calls) = measure(&mut batch);
            let ((ordinary_cold, ordinary_cold_bytes), ordinary_cold_calls) =
                measure(&mut ordinary);
            let ((batch_warm, batch_warm_bytes), batch_warm_calls) = measure(&mut batch);
            let ((ordinary_warm, ordinary_warm_bytes), ordinary_warm_calls) =
                measure(&mut ordinary);
            assert_eq!(batch_destination, ordinary_destination);
            let retained = |w: &TreeTransformWorkspace<f64>| {
                core::mem::size_of::<f64>()
                    * (w.packed.source().capacity()
                        + w.packed.destination().capacity()
                        + w.coefficient_scratch.capacity())
                    + core::mem::size_of::<DenseGemmBatchJob>() * w.chunk_jobs.capacity()
                    + core::mem::size_of::<(usize, usize)>() * w.member_ranges.capacity()
                    + core::mem::size_of::<usize>()
                        * (w.chunk_runs.capacity()
                            + w.chunk_scatter_groups.capacity()
                            + w.fused_indices_heap_capacity()
                            + w.member_shape.capacity())
                    + core::mem::size_of::<isize>()
                        * (w.zero_strides.capacity()
                            + w.member_dst_strides.capacity()
                            + w.member_src_strides.capacity())
            };
            eprintln!("B={members} batch cold={batch_cold:?} {batch_cold_calls}calls {batch_cold_bytes}bytes warm={batch_warm:?} {batch_warm_calls}calls {batch_warm_bytes}bytes submissions={} retained={}B; ordinary cold={ordinary_cold:?} {ordinary_cold_calls}calls {ordinary_cold_bytes}bytes warm={ordinary_warm:?} {ordinary_warm_calls}calls {ordinary_warm_bytes}bytes submissions={} retained={}B",
                batch_dense.submissions, retained(&batch_workspace), ordinary_dense.submissions, retained(&ordinary_workspace));
        }
    }
}
