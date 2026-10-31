use super::*;

impl<C> FusionBlockContractPlan<C>
where
    C: Copy + PartialEq + One,
{
    #[allow(clippy::too_many_arguments)]
    pub fn execute_raw<A, G, D>(
        &self,
        engine: &mut dyn Engine,
        kernels: &mut A,
        gemm: &mut G,
        fusion_workspace: &mut FusionBlockContractWorkspace<D>,
        dst_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        lhs_structure: &Arc<BlockStructure>,
        lhs_data: &[D],
        rhs_structure: &Arc<BlockStructure>,
        rhs_data: &[D],
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        A: HostKernelAdapter<D>,
        G: Rank2Gemm<D>,
        D: DenseBlockScalar + RecouplingCoefficientAction<C>,
    {
        self.execute_host(
            engine,
            kernels,
            gemm,
            fusion_workspace,
            dst_structure,
            dst_data,
            lhs_structure,
            lhs_data,
            rhs_structure,
            rhs_data,
            alpha,
            ContractDestinationInit::Axpby(beta),
        )
    }

    /// [`Self::execute_raw`] with `beta = 0` for a destination the caller
    /// proves is all-zero ([`ContractDestinationInit::Zeroed`]): the active
    /// blocks are overwritten by their GEMM/scatter job and the inactive
    /// blocks are left untouched.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_raw_zeroed<A, G, D>(
        &self,
        engine: &mut dyn Engine,
        kernels: &mut A,
        gemm: &mut G,
        fusion_workspace: &mut FusionBlockContractWorkspace<D>,
        dst_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        lhs_structure: &Arc<BlockStructure>,
        lhs_data: &[D],
        rhs_structure: &Arc<BlockStructure>,
        rhs_data: &[D],
        alpha: D,
    ) -> Result<(), OperationError>
    where
        A: HostKernelAdapter<D>,
        G: Rank2Gemm<D>,
        D: DenseBlockScalar + RecouplingCoefficientAction<C>,
    {
        self.execute_host(
            engine,
            kernels,
            gemm,
            fusion_workspace,
            dst_structure,
            dst_data,
            lhs_structure,
            lhs_data,
            rhs_structure,
            rhs_data,
            alpha,
            ContractDestinationInit::Zeroed,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_host<A, G, D>(
        &self,
        engine: &mut dyn Engine,
        kernels: &mut A,
        gemm: &mut G,
        fusion_workspace: &mut FusionBlockContractWorkspace<D>,
        dst_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        lhs_structure: &Arc<BlockStructure>,
        lhs_data: &[D],
        rhs_structure: &Arc<BlockStructure>,
        rhs_data: &[D],
        alpha: D,
        init: ContractDestinationInit<D>,
    ) -> Result<(), OperationError>
    where
        A: HostKernelAdapter<D>,
        G: Rank2Gemm<D>,
        D: DenseBlockScalar + RecouplingCoefficientAction<C>,
    {
        let beta = init.active_beta();
        self.validate_replay_inputs(
            dst_structure,
            dst_data.len(),
            lhs_structure,
            lhs_data.len(),
            rhs_structure,
            rhs_data.len(),
        )?;

        if self.max_irregular_scratch_len != 0 {
            fusion_workspace
                .scratch
                .resize_filled(self.max_irregular_scratch_len, D::zero());
        } else {
            fusion_workspace.scratch.resize_filled(0, D::zero());
        }

        crate::host_pool::with_exec(engine, |exec| {
            self.init_inactive_blocks(
                exec,
                kernels,
                &mut fusion_workspace.zero_strides,
                dst_data,
                init,
            )
        })?;

        if !self.direct_batch.is_empty() {
            self.execute_direct_batches(engine, gemm, dst_data, lhs_data, rhs_data, alpha, beta)?;
        }

        for execution in &self.irregular {
            self.execute_irregular_group(
                engine,
                kernels,
                gemm,
                fusion_workspace,
                execution,
                dst_data,
                lhs_data,
                rhs_data,
                alpha,
                beta,
            )?;
        }
        Ok(())
    }

    /// `direct_batch` as one batch per job coefficient: the jobs of one
    /// coefficient are contiguous and its runs end at the next coefficient
    /// (plan-time invariant), so an unscaled plan is exactly one batch. The
    /// batches write disjoint destination blocks, so `beta` reaches each
    /// written element once.
    #[allow(clippy::too_many_arguments)]
    fn execute_direct_batches<G, D>(
        &self,
        engine: &mut dyn Engine,
        gemm: &mut G,
        dst: &mut [D],
        lhs: &[D],
        rhs: &[D],
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        G: Rank2Gemm<D>,
        D: DenseBlockScalar + RecouplingCoefficientAction<C>,
    {
        if self.direct_batch.len() != self.direct_batch_alpha.len() {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "direct plan has misaligned GEMM coefficients",
            });
        }
        let (mut start, mut run) = (0, 0);
        while start < self.direct_batch.len() {
            let coefficient = self.direct_batch_alpha[start];
            let mut end = start;
            let first_run = run;
            while end < self.direct_batch.len() && self.direct_batch_alpha[end] == coefficient {
                let len = self.direct_batch_runs.get(run).copied().ok_or(
                    OperationError::UnsupportedTensorContractScope {
                        message: "direct plan runs do not cover its jobs",
                    },
                )?;
                end += len;
                run += 1;
            }
            if end > self.direct_batch.len()
                || self.direct_batch_alpha[start..end]
                    .iter()
                    .any(|&alpha| alpha != coefficient)
            {
                return Err(OperationError::UnsupportedTensorContractScope {
                    message: "direct plan runs cross a coefficient change",
                });
            }
            let job_alpha = if coefficient == C::one() {
                alpha
            } else {
                alpha.scale_by_coefficient(coefficient)
            };
            self.execute_batch(
                engine,
                gemm,
                dst,
                lhs,
                rhs,
                &self.direct_batch[start..end],
                &self.direct_batch_runs[first_run..run],
                job_alpha,
                beta,
            )?;
            start = end;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_irregular_group<A, G, D>(
        &self,
        engine: &mut dyn Engine,
        kernels: &mut A,
        gemm: &mut G,
        fusion_workspace: &mut FusionBlockContractWorkspace<D>,
        execution: &FusionIrregularGroupExecution,
        dst_data: &mut [D],
        lhs_data: &[D],
        rhs_data: &[D],
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        A: HostKernelAdapter<D>,
        G: Rank2Gemm<D>,
        D: DenseBlockScalar + RecouplingCoefficientAction<C>,
    {
        let group = &self.groups[execution.group_index];
        let scratch = &mut fusion_workspace.scratch.as_mut_slice()[..execution.scratch.total_len];
        // `execute_batch` never reads the packed operands at a zero `alpha`.
        let forms_product = !alpha.is_zero();
        if forms_product && execution.class.packs_lhs() {
            let lhs = &mut scratch[..execution.scratch.lhs_len];
            if group.lhs.needs_clear {
                lhs.fill(D::zero());
            }
            crate::host_pool::with_exec(engine, |exec| {
                pack_group(exec, kernels, &group.lhs, lhs_data, lhs)
            })?;
        }
        if forms_product && execution.class.packs_rhs() {
            let start_index = execution.scratch.rhs_offset;
            let end = start_index + execution.scratch.rhs_len;
            let rhs = &mut scratch[start_index..end];
            if group.rhs.needs_clear {
                rhs.fill(D::zero());
            }
            crate::host_pool::with_exec(engine, |exec| {
                pack_group(exec, kernels, &group.rhs, rhs_data, rhs)
            })?;
        }

        if execution.class.scatters_dst() {
            let (inputs, destination) = scratch.split_at_mut(execution.scratch.dst_offset);
            let destination = &mut destination[..execution.scratch.dst_len];
            let lhs = if execution.class.packs_lhs() {
                &inputs[..execution.scratch.lhs_len]
            } else {
                lhs_data
            };
            let rhs = if execution.class.packs_rhs() {
                let start = execution.scratch.rhs_offset;
                &inputs[start..start + execution.scratch.rhs_len]
            } else {
                rhs_data
            };
            self.execute_batch(
                engine,
                gemm,
                destination,
                lhs,
                rhs,
                std::slice::from_ref(&execution.job),
                &[1],
                alpha,
                D::zero(),
            )?;
            crate::host_pool::with_exec(engine, |exec| {
                scatter_group(exec, kernels, &group.dst, dst_data, destination, beta)
            })?;
        } else {
            let lhs = if execution.class.packs_lhs() {
                &scratch[..execution.scratch.lhs_len]
            } else {
                lhs_data
            };
            let rhs = if execution.class.packs_rhs() {
                let start = execution.scratch.rhs_offset;
                &scratch[start..start + execution.scratch.rhs_len]
            } else {
                rhs_data
            };
            self.execute_batch(
                engine,
                gemm,
                dst_data,
                lhs,
                rhs,
                std::slice::from_ref(&execution.job),
                &[1],
                alpha,
                beta,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_batch<G, D>(
        &self,
        engine: &mut dyn Engine,
        gemm: &mut G,
        dst: &mut [D],
        lhs: &[D],
        rhs: &[D],
        jobs: &[Rank2GemmBatchJob],
        runs: &[usize],
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        G: Rank2Gemm<D>,
        D: Copy + PartialEq + Zero + One + Mul<D, Output = D>,
    {
        // BLAS `gemm` at `alpha == 0` leaves `beta * C` without forming
        // `A * B`, as TensorKit's `mul!` reaches it; a backend that multiplies
        // through (faer) would turn `0 * Inf` into NaN (#1442).
        if alpha.is_zero() {
            for job in jobs {
                let block = direct_slice_mut(dst, job.dst_offset, job.rows, job.cols)?;
                if beta.is_zero() {
                    block.fill(D::zero());
                } else if !beta.is_one() {
                    block.iter_mut().for_each(|value| *value = beta * *value);
                }
            }
            return Ok(());
        }
        if self.lhs_op == MatrixOp::Identity && self.rhs_op == MatrixOp::Identity {
            gemm.matmul_rank2_batch(engine, dst, lhs, rhs, jobs, runs, alpha, beta)
        } else {
            gemm.matmul_rank2_batch_with_ops(
                engine,
                dst,
                lhs,
                rhs,
                jobs,
                runs,
                self.lhs_op,
                self.rhs_op,
                alpha,
                beta,
            )
        }
    }

    pub(crate) fn direct_batch(&self) -> &[Rank2GemmBatchJob] {
        &self.direct_batch
    }

    /// How many distinct `(rows, contracted, cols)` shapes the direct GEMM
    /// jobs have: the dense plans a replay of this plan needs at one batch
    /// extent.
    #[doc(hidden)]
    pub fn distinct_direct_gemm_shapes(&self) -> usize {
        self.direct_batch
            .iter()
            .map(|job| (job.rows, job.contracted, job.cols))
            .collect::<HashSet<_>>()
            .len()
    }

    /// The per-member payload lengths `[dst, lhs, rhs]` this plan's
    /// structures require.
    pub(crate) fn member_lens(&self) -> Result<[usize; 3], OperationError> {
        let len = |structure: &BlockStructure| {
            structure
                .required_len()
                .map_err(OperationError::from_core_preserving_context)
        };
        Ok([
            len(&self.dst_structure)?,
            len(&self.lhs_structure)?,
            len(&self.rhs_structure)?,
        ])
    }

    /// The preconditions of an unscaled identity-orientation direct replay:
    /// every group direct, both operands untransformed, every job coefficient
    /// one.
    #[doc(hidden)]
    pub fn require_identity_direct_replay(&self) -> Result<(), OperationError> {
        self.require_fully_direct_storage()?;
        self.require_identity_storage_ops()?;
        self.require_unit_direct_batch_alpha()
    }

    /// The stacked Host route accepts only identity operands and exact signs.
    #[doc(hidden)]
    pub fn require_identity_signed_direct_replay(&self) -> Result<(), OperationError>
    where
        C: std::ops::Neg<Output = C>,
    {
        self.require_fully_direct_storage()?;
        self.require_identity_storage_ops()?;
        if self.direct_batch.len() == self.direct_batch_alpha.len()
            && self
                .direct_batch_alpha
                .iter()
                .all(|&alpha| alpha == C::one() || alpha == -C::one())
        {
            return Ok(());
        }
        Err(OperationError::UnsupportedTensorContractScope {
            message: "signed stacked replay requires exact unit-magnitude coefficients",
        })
    }

    pub(crate) fn direct_batch_alphas(&self) -> &[C] {
        &self.direct_batch_alpha
    }

    fn require_fully_direct_storage(&self) -> Result<(), OperationError> {
        if self.is_fully_direct() {
            Ok(())
        } else {
            Err(OperationError::UnsupportedTensorContractScope {
                message: NON_COUPLED_OPERAND_MESSAGE,
            })
        }
    }

    fn require_identity_storage_ops(&self) -> Result<(), OperationError> {
        if self.lhs_op == MatrixOp::Identity && self.rhs_op == MatrixOp::Identity {
            return Ok(());
        }
        Err(OperationError::UnsupportedTensorContractScope {
            message: "storage-handle core replay does not expose transpose/adjoint matrix views",
        })
    }

    fn require_unit_direct_batch_alpha(&self) -> Result<(), OperationError> {
        if self
            .direct_batch_alpha
            .iter()
            .all(|&alpha| alpha == C::one())
        {
            return Ok(());
        }
        Err(OperationError::UnsupportedTensorContractScope {
            message: "scaled storage plan cannot execute through unscaled host replay",
        })
    }

    /// Plan-time run partition of [`Self::direct_batch`]; handed to the backend
    /// alongside the jobs so it routes runs without recomputing the partition.
    #[cfg(test)]
    pub(super) fn direct_batch_runs(&self) -> &[usize] {
        &self.direct_batch_runs
    }

    /// Executes the contraction purely over storage handles, for a
    /// destination the caller guarantees is zero-filled.
    ///
    /// This is the device-side replay seam: the bounds require only
    /// [`TensorStorage`], so no host-slice contract leaks into the path. It
    /// supports exactly the fully-direct coupled-layout case with one scalar
    /// per GEMM job; every other case must use the host replay paths until the
    /// corresponding device kernels (pack/scatter, scale, tree transforms)
    /// exist behind their own seams. Destination blocks with no contributing
    /// GEMM (the ones the host path scales by `beta = 0`) are left untouched,
    /// which is exactly the overwrite semantics on a zeroed buffer. The active
    /// blocks are still fully overwritten.
    pub fn execute_direct_on_storage_prezeroed<G, D, DDst, DLhs, DRhs>(
        &self,
        gemm: &mut G,
        dst: &mut DDst,
        lhs: &DLhs,
        rhs: &DRhs,
    ) -> Result<(), OperationError>
    where
        G: StorageGemm<D, DDst, DLhs, DRhs>,
        D: RecouplingCoefficientAction<C>,
        DDst: TensorStorage<D>,
        DLhs: TensorStorage<D>,
        DRhs: TensorStorage<D>,
    {
        self.require_fully_direct_storage()?;
        if self.direct_batch.len() != self.direct_batch_alpha.len() {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "storage-direct plan has misaligned GEMM coefficients",
            });
        }
        let needs_general_gemm = self.lhs_op != MatrixOp::Identity
            || self.rhs_op != MatrixOp::Identity
            || self
                .direct_batch_alpha
                .iter()
                .any(|&alpha| alpha != C::one());
        if needs_general_gemm && !gemm.supports_matmul_with_ops_scaled(self.lhs_op, self.rhs_op) {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "storage GEMM backend does not implement scaled replay",
            });
        }
        for job in &self.direct_batch {
            validate_storage_range(lhs.len(), job.lhs_offset, job.rows, job.contracted)?;
            validate_storage_range(rhs.len(), job.rhs_offset, job.contracted, job.cols)?;
            validate_storage_range(dst.len(), job.dst_offset, job.rows, job.cols)?;
        }
        for (job, &alpha) in self.direct_batch.iter().zip(&self.direct_batch_alpha) {
            if alpha == C::one()
                && self.lhs_op == MatrixOp::Identity
                && self.rhs_op == MatrixOp::Identity
            {
                gemm.matmul_range_into(
                    dst,
                    job.dst_offset,
                    lhs,
                    job.lhs_offset,
                    rhs,
                    job.rhs_offset,
                    job.rows,
                    job.contracted,
                    job.cols,
                )?;
            } else {
                gemm.matmul_range_with_ops_scaled_into(
                    dst,
                    job.dst_offset,
                    lhs,
                    job.lhs_offset,
                    rhs,
                    job.rhs_offset,
                    job.rows,
                    job.contracted,
                    job.cols,
                    self.lhs_op,
                    self.rhs_op,
                    D::coefficient_as_data(alpha),
                )?;
            }
        }
        Ok(())
    }

    /// `dst = alpha * (lhs · rhs) + beta * dst` over the blocks the GEMM jobs
    /// write, purely over storage handles: each job's GEMM carries
    /// `alpha * job_alpha` and `beta` in its own epilogue, so `beta` is applied
    /// once per written element and never as a separate pass (TensorKit
    /// `mul!(C, A, B, α, β)` per coupled sector). The jobs write disjoint
    /// destination blocks, as [`Self::execute_direct_on_storage_prezeroed`]
    /// already relies on. Blocks no job writes are the caller's: they are
    /// [`Self::inactive_destination_regions`], which it scales by `beta`.
    ///
    /// Every job goes through the scaled GEMM entry; a caller with
    /// `alpha = 1, beta = 0` keeps [`Self::execute_direct_on_storage_prezeroed`].
    #[allow(clippy::too_many_arguments)]
    pub fn execute_direct_on_storage_axpby<G, D, DDst, DLhs, DRhs>(
        &self,
        gemm: &mut G,
        dst: &mut DDst,
        lhs: &DLhs,
        rhs: &DRhs,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        G: StorageGemm<D, DDst, DLhs, DRhs>,
        D: RecouplingCoefficientAction<C>,
        DDst: TensorStorage<D>,
        DLhs: TensorStorage<D>,
        DRhs: TensorStorage<D>,
    {
        self.require_fully_direct_storage()?;
        if self.direct_batch.len() != self.direct_batch_alpha.len() {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "storage-direct plan has misaligned GEMM coefficients",
            });
        }
        for job in &self.direct_batch {
            validate_storage_range(lhs.len(), job.lhs_offset, job.rows, job.contracted)?;
            validate_storage_range(rhs.len(), job.rhs_offset, job.contracted, job.cols)?;
            validate_storage_range(dst.len(), job.dst_offset, job.rows, job.cols)?;
        }
        for (job, &job_alpha) in self.direct_batch.iter().zip(&self.direct_batch_alpha) {
            gemm.matmul_range_axpby_with_ops_into(
                dst,
                job.dst_offset,
                lhs,
                job.lhs_offset,
                rhs,
                job.rhs_offset,
                job.rows,
                job.contracted,
                job.cols,
                self.lhs_op,
                self.rhs_op,
                alpha.scale_by_coefficient(job_alpha),
                beta,
            )?;
        }
        Ok(())
    }

    /// Read-only structure admission for a composite replay preflight.
    #[doc(hidden)]
    pub fn validate_replay_structures(
        &self,
        dst_structure: &Arc<BlockStructure>,
        lhs_structure: &Arc<BlockStructure>,
        rhs_structure: &Arc<BlockStructure>,
    ) -> Result<(), OperationError> {
        validate_structure_identity("dst", &self.dst_structure, dst_structure)?;
        validate_structure_identity("lhs", &self.lhs_structure, lhs_structure)?;
        validate_structure_identity("rhs", &self.rhs_structure, rhs_structure)
    }

    fn validate_replay_inputs(
        &self,
        dst_structure: &Arc<BlockStructure>,
        dst_len: usize,
        lhs_structure: &Arc<BlockStructure>,
        lhs_len: usize,
        rhs_structure: &Arc<BlockStructure>,
        rhs_len: usize,
    ) -> Result<(), OperationError> {
        self.validate_replay_structures(dst_structure, lhs_structure, rhs_structure)?;
        validate_storage_len(dst_structure, dst_len)?;
        validate_storage_len(lhs_structure, lhs_len)?;
        validate_storage_len(rhs_structure, rhs_len)
    }
}

pub(super) fn pack_group<A, T, C>(
    mut exec: crate::host_pool::Exec<'_>,
    kernels: &mut A,
    group: &FusionBlockMatrixGroup<C>,
    data: &[T],
    packed: &mut [T],
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<T>,
    T: Copy + One + PartialEq + RecouplingCoefficientAction<C>,
    C: Copy,
{
    for layout in &group.subblocks {
        kernels.transform_strided_baked(
            exec.reborrow(),
            &mut Vec::new(),
            packed,
            data,
            &layout.block.shape,
            &layout.matrix_strides,
            &layout.block.strides,
            layout.matrix_offset,
            layout.block.offset,
            false,
            TransformScale::Structural(layout.coefficient),
            None,
            None,
            None,
        )?;
    }
    Ok(())
}

pub(super) fn scatter_group<A, T, C>(
    mut exec: crate::host_pool::Exec<'_>,
    kernels: &mut A,
    group: &FusionBlockMatrixGroup<C>,
    data: &mut [T],
    packed: &[T],
    beta: T,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<T>,
    T: Copy + One + PartialEq + RecouplingCoefficientAction<C>,
    C: Copy,
{
    for layout in &group.subblocks {
        kernels.transform_strided_baked(
            exec.reborrow(),
            &mut Vec::new(),
            data,
            packed,
            &layout.block.shape,
            &layout.block.strides,
            &layout.matrix_strides,
            layout.block.offset,
            layout.matrix_offset,
            false,
            TransformScale::Structural(layout.coefficient),
            Some(beta),
            None,
            None,
        )?;
    }
    Ok(())
}

/// Checked mutable `rows x cols` column-major matrix slice at `base`; the
/// bounds-checking counterpart of the batch-job offset addressing.
pub fn direct_slice_mut<T>(
    data: &mut [T],
    base: usize,
    rows: usize,
    cols: usize,
) -> Result<&mut [T], OperationError> {
    let len = direct_matrix_len(rows, cols)?;
    let end = base
        .checked_add(len)
        .ok_or_else(|| OperationError::ElementCountOverflow)?;
    let actual = data.len();
    data.get_mut(base..end)
        .ok_or_else(|| OperationError::ElementCountMismatch {
            expected: end,
            actual,
        })
}

impl<C> FusionBlockContractPlan<C> {
    /// Initialises the destination blocks no job writes (see
    /// [`ContractDestinationInit`]). `zero_strides` is the caller's reusable
    /// stride scratch for the zero-source assignment.
    fn init_inactive_blocks<A, T>(
        &self,
        mut exec: crate::host_pool::Exec<'_>,
        kernels: &mut A,
        zero_strides: &mut Vec<isize>,
        data: &mut [T],
        init: ContractDestinationInit<T>,
    ) -> Result<(), OperationError>
    where
        A: HostKernelAdapter<T>,
        T: Copy + Zero + One + PartialEq,
    {
        let beta = match init {
            ContractDestinationInit::Zeroed => return Ok(()),
            ContractDestinationInit::Axpby(beta) if beta.is_one() => return Ok(()),
            ContractDestinationInit::Axpby(beta) => beta,
        };
        if beta.is_zero() {
            let zero = [T::zero()];
            for layout in &self.inactive_dst_scale_blocks {
                zero_strides.clear();
                zero_strides.resize(layout.block.shape.len(), 0);
                kernels.copy_scale_strided_baked(
                    exec.reborrow(),
                    data,
                    &zero,
                    &layout.block.shape,
                    &layout.block.strides,
                    zero_strides,
                    layout.block.offset,
                    0,
                    false,
                    T::one(),
                    None,
                    None,
                )?;
            }
            return Ok(());
        }
        for layout in &self.inactive_dst_scale_blocks {
            kernels.scale_strided(
                exec.reborrow(),
                data,
                &layout.block.shape,
                &layout.block.strides,
                layout.block.offset,
                beta,
            )?;
        }
        Ok(())
    }
}

/// Checked shared `rows x cols` column-major matrix slice at `base`; the
/// bounds-checking counterpart of the batch-job offset addressing.
pub fn direct_slice<T>(
    data: &[T],
    base: usize,
    rows: usize,
    cols: usize,
) -> Result<&[T], OperationError> {
    let len = direct_matrix_len(rows, cols)?;
    let end = base
        .checked_add(len)
        .ok_or_else(|| OperationError::ElementCountOverflow)?;
    data.get(base..end)
        .ok_or_else(|| OperationError::ElementCountMismatch {
            expected: end,
            actual: data.len(),
        })
}
