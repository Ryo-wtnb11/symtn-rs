use num_complex::{Complex32, Complex64};

use core::mem::MaybeUninit;

use crate::executor::{batch_offset, strided_batch_run_len};
use crate::fresh::{
    dispatch_fresh, split_single_fresh_segment, zero_piece, FreshParts, FreshScalar, FRESH_OP,
};
use crate::layout::strides_to_isize;
use crate::{
    DenseBackend, DenseDotConfig, DenseError, DenseExecutor, DenseFreshSegments, DenseGemmBatchJob,
    DenseOwned, DenseRead, DenseScalar, DenseTensor, DenseView, DenseViewMut, DenseWrite, MatrixOp,
};

use tenferro_cpu::{with_cpu_engine, CpuEngine};
use tenferro_linalg::{LinalgBackend, LinalgOps};
use tenferro_tensor::{
    Dest, DotGeneralConfig, Engine, SessionCachedDot, TensorRead, TensorView, TensorViewMut,
    TensorWrite, TypedTensorView, TypedTensorViewMut,
};

#[derive(Clone, Copy, Hash)]
struct StridedBatchRunLayout {
    lhs_shape: [usize; 3],
    lhs_strides: [usize; 3],
    lhs_job_offset: usize,
    rhs_shape: [usize; 3],
    rhs_strides: [usize; 3],
    rhs_job_offset: usize,
    dst_shape: [usize; 3],
    dst_strides: [usize; 3],
    dst_job_offset: usize,
}

/// The engine plan-cache slot of one strided-entry GEMM: its full descriptor
/// (scalar type, operand layouts with base offsets, operand ops) hashed over
/// the engine's slot capacity.
///
/// Why not one fixed slot: the engine keeps one plan per slot and rebuilds it
/// whenever the descriptor changes, so a replay issuing one GEMM per block of
/// differing shape rebuilt every block's plan on every warm call
/// (tenferro-rs#2349). Why not a caller-supplied identity: this executor seam
/// carries none, and the descriptor is the plan's identity anyway. A collision
/// only costs a rebuild: the engine compares the full descriptor before it
/// reuses a slot's plan.
fn descriptor_slot<T: 'static>(
    cpu: &CpuEngine,
    layout: &StridedBatchRunLayout,
    base_offsets: [usize; 3],
    ops: [MatrixOp; 2],
) -> usize {
    use std::hash::{Hash, Hasher};
    let capacity = cpu.plan_cache().contraction_capacity().max(1);
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    core::any::TypeId::of::<T>().hash(&mut hasher);
    layout.hash(&mut hasher);
    base_offsets.hash(&mut hasher);
    ops.hash(&mut hasher);
    (hasher.finish() % capacity as u64) as usize
}

#[derive(Debug)]
pub struct DefaultDenseExecutor {
    matmul_config: DotGeneralConfig,
    strided_batch_matmul_config: DotGeneralConfig,
    // The native block kernel's staging, reused across this executor's
    // calls: a prepared workspace's executor replays without allocating.
    gemm_scratch: crate::block_kernel::gemm::GemmScratch,
    // Test-only count of low-level batch seam submissions: one per strided
    // call and one per job in the serial route.
    #[cfg(test)]
    seam_dispatches: usize,
    // Test-only: jobs of the last batch routed where the grouped entry was.
    #[cfg(test)]
    grouped_route_jobs: usize,
}

impl DefaultDenseExecutor {
    /// An executor of descriptors and reusable job scratch only: it owns no
    /// threads or engine. Every call runs on the CPU engine the caller passes.
    pub fn new() -> Self {
        Self {
            matmul_config: DotGeneralConfig {
                lhs_contracting_dims: [1].into_iter().collect(),
                rhs_contracting_dims: [0].into_iter().collect(),
                lhs_batch_dims: Default::default(),
                rhs_batch_dims: Default::default(),
            },
            strided_batch_matmul_config: DotGeneralConfig {
                lhs_contracting_dims: [1].into_iter().collect(),
                rhs_contracting_dims: [0].into_iter().collect(),
                lhs_batch_dims: [2].into_iter().collect(),
                rhs_batch_dims: [2].into_iter().collect(),
            },
            gemm_scratch: Default::default(),
            #[cfg(test)]
            seam_dispatches: 0,
            #[cfg(test)]
            grouped_route_jobs: 0,
        }
    }

    /// Bytes of reusable call scratch this executor keeps between calls (the
    /// block GEMM kernel's staging), a conservative charge of its capacities. It holds no plan: GEMM plans
    /// live in the process-global geometry-plan cache. Unstable plumbing for
    /// symtn's prepared workspaces' `retained_bytes`.
    #[doc(hidden)]
    pub fn retained_bytes(&self) -> usize {
        self.gemm_scratch.retained_bytes()
    }

    #[cfg(test)]
    pub(crate) fn reset_seam_dispatches(&mut self) {
        self.seam_dispatches = 0;
    }

    #[cfg(test)]
    pub(crate) fn seam_dispatches(&self) -> usize {
        self.seam_dispatches
    }

    #[cfg(test)]
    pub(crate) fn staged_grouped_jobs(&self) -> usize {
        self.grouped_route_jobs
    }

    #[allow(clippy::too_many_arguments)]
    fn matmul_batch_axpby_dispatch(
        &mut self,
        engine: &mut dyn Engine,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        jobs: &[DenseGemmBatchJob],
        runs: &[usize],
        ops: [MatrixOp; 2],
        alpha: DenseScalar,
        beta: DenseScalar,
    ) -> Result<(), DenseError> {
        let op = if ops == [MatrixOp::Identity; 2] {
            "matmul_batch_axpby_into"
        } else {
            "matmul_batch_axpby_with_ops_into"
        };
        let (out_dtype, lhs_dtype, rhs_dtype) = (output.dtype(), lhs.dtype(), rhs.dtype());
        host_cpu(engine, op, |cpu| match (output, lhs, rhs) {
            (DenseWrite::F32(mut out), DenseRead::F32(lhs), DenseRead::F32(rhs)) => self
                .matmul_batch_axpby_route_typed(
                    cpu,
                    &mut out,
                    lhs,
                    rhs,
                    jobs,
                    runs,
                    ops,
                    alpha,
                    beta,
                    |view: DenseViewMut<'_, f32>| DenseWrite::F32(view),
                    |view: DenseView<'_, f32>| DenseRead::F32(view),
                ),
            (DenseWrite::F64(mut out), DenseRead::F64(lhs), DenseRead::F64(rhs)) => self
                .matmul_batch_axpby_route_typed(
                    cpu,
                    &mut out,
                    lhs,
                    rhs,
                    jobs,
                    runs,
                    ops,
                    alpha,
                    beta,
                    |view: DenseViewMut<'_, f64>| DenseWrite::F64(view),
                    |view: DenseView<'_, f64>| DenseRead::F64(view),
                ),
            (DenseWrite::C32(mut out), DenseRead::C32(lhs), DenseRead::C32(rhs)) => self
                .matmul_batch_axpby_route_typed(
                    cpu,
                    &mut out,
                    lhs,
                    rhs,
                    jobs,
                    runs,
                    ops,
                    alpha,
                    beta,
                    |view: DenseViewMut<'_, Complex32>| DenseWrite::C32(view),
                    |view: DenseView<'_, Complex32>| DenseRead::C32(view),
                ),
            (DenseWrite::C64(mut out), DenseRead::C64(lhs), DenseRead::C64(rhs)) => self
                .matmul_batch_axpby_route_typed(
                    cpu,
                    &mut out,
                    lhs,
                    rhs,
                    jobs,
                    runs,
                    ops,
                    alpha,
                    beta,
                    |view: DenseViewMut<'_, Complex64>| DenseWrite::C64(view),
                    |view: DenseView<'_, Complex64>| DenseRead::C64(view),
                ),
            _ => Err(crate::executor::batch_operand_error(
                op, out_dtype, lhs_dtype, rhs_dtype,
            )),
        })
    }

    /// Routes one typed batch by what the strided entry and the grouped
    /// kernel can represent, not by a run-length cutoff. The strided entry is one rank-3
    /// contraction, so it can carry a batch iff the batch is one job (batch
    /// extent one; TensorKit's per-block `mul!`) or ONE affine run:
    /// `runs == [jobs.len()]`, at least two jobs (a step needs two offsets),
    /// every job sharing job 0's shape and the job-0-to-job-1 offset steps
    /// (`strided_batch_run_len`, O(J): `runs` is a caller claim, and a wrong
    /// one would apply job 0's shape and steps to every job), and
    /// a destination step of at least `rows * cols`, which is the O(1) proof
    /// that the rank-3 destination view is self-disjoint (tenferro's dot path
    /// does not check destination overlap; the grouped kernel does). An
    /// op-bearing run also needs nonzero extents and a rank-3 layout the views
    /// admit. The grouped kernel — TensorKit `mul!` per matched sector, QSpace
    /// `contract_matchAB_groupC` -> `contractDATA_group`, a destination-grouped
    /// GEMM list with no cutoff — represents every batch, so everything else
    /// goes there as ONE submission built from `jobs` (never from `runs`, so a
    /// malformed partition cannot skip or over-index a job).
    ///
    /// Why not send one job to the grouped kernel too: a grouped submission
    /// stages its job records, geometry index and lane table and enters the
    /// scheduler, so a recoupling replay that submits one GEMM per Multi
    /// block would pay that per block (#2184), and an eager operation's
    /// executor, built per call, allocates the staging; the strided entry
    /// allocates nothing once its engine cache is warm.
    ///
    /// Why not a per-run cutoff: it would be an empirical constant in tensor
    /// semantics. The disadvantage accepted here: a long run plus residual
    /// jobs goes grouped, paying the grouped preparation per unbound call
    /// (O(J log J) staging, and under Auto the placement's lane search:
    /// one provider grant query per distinct (time, plan algorithm, work)
    /// group at the full slot count S, further queries per distinct share
    /// width only for the groups widened there, a lower bound per lane count,
    /// a placement for each lane count the bound does not rule out, and the
    /// two large-block reservations: a query at every width up to S for the
    /// (fewer than S) blocks above the mean load, shared by both, and per
    /// reservation a target search and one placement of the other blocks
    /// over the reserved and one-slot lanes;
    /// see `lanes` for the complexity) instead of the cached O(1) strided
    /// analysis plus the residual's own preparation.
    #[allow(clippy::too_many_arguments)]
    fn matmul_batch_axpby_route_typed<T, W, R>(
        &mut self,
        cpu: &mut CpuEngine,
        output: &mut DenseViewMut<'_, T>,
        lhs: DenseView<'_, T>,
        rhs: DenseView<'_, T>,
        jobs: &[DenseGemmBatchJob],
        runs: &[usize],
        ops: [MatrixOp; 2],
        alpha: DenseScalar,
        beta: DenseScalar,
        wrap_write: W,
        wrap_read: R,
    ) -> Result<(), DenseError>
    where
        T: GroupedScalar,
        W: for<'x> Fn(DenseViewMut<'x, T>) -> DenseWrite<'x> + Copy,
        R: for<'x> Fn(DenseView<'x, T>) -> DenseRead<'x> + Copy,
    {
        if jobs.is_empty() {
            return Ok(());
        }
        let identity = ops == [MatrixOp::Identity; 2];
        // An op-bearing run additionally needs nonzero extents and a layout
        // the views admit: its transposed rank-3 view is otherwise refused,
        // while the grouped kernel represents every such batch. The extents
        // are compared one by one, not multiplied: they are unvalidated here,
        // and the checked layout and range helpers own size overflow.
        let strided = crate::executor::single_strided_batch(jobs, runs)
            && (identity
                || jobs[0].rows != 0
                    && jobs[0].contracted != 0
                    && jobs[0].cols != 0
                    && self
                        .strided_batch_run_layout(jobs, ops[0], ops[1])
                        .is_ok_and(|layout| {
                            self.strided_batch_run_layout_admitted(
                                (output.data(), output.offset()),
                                lhs,
                                rhs,
                                &layout,
                            )
                        }));
        if strided {
            return self.matmul_strided_batch_run_typed(
                cpu,
                output,
                lhs,
                rhs,
                jobs,
                ops[0],
                ops[1],
                alpha,
                beta,
                if identity {
                    "strided_batch_gemm"
                } else {
                    "matmul_batch_axpby_with_ops_into"
                },
                wrap_write,
                wrap_read,
            );
        }
        #[cfg(test)]
        {
            self.grouped_route_jobs = jobs.len();
            self.seam_dispatches += 1;
        }
        let _ = (wrap_write, wrap_read);
        let (Some(alpha), Some(beta)) = (T::from_dense(alpha), T::from_dense(beta)) else {
            return Err(grouped_error(
                "alpha/beta dtype differs from the operands".to_string(),
            ));
        };
        crate::block_kernel::gemm::gemm_batch_into(
            cpu,
            &mut self.gemm_scratch,
            output,
            lhs,
            rhs,
            jobs,
            ops,
            alpha,
            beta,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn matmul_strided_batch_run_typed<T, W, R>(
        &mut self,
        cpu: &mut CpuEngine,
        output: &mut DenseViewMut<'_, T>,
        lhs: DenseView<'_, T>,
        rhs: DenseView<'_, T>,
        run: &[DenseGemmBatchJob],
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
        alpha: DenseScalar,
        beta: DenseScalar,
        error_op: &'static str,
        wrap_write: W,
        wrap_read: R,
    ) -> Result<(), DenseError>
    where
        T: 'static,
        W: for<'x> Fn(DenseViewMut<'x, T>) -> DenseWrite<'x>,
        R: for<'x> Fn(DenseView<'x, T>) -> DenseRead<'x>,
    {
        let layout = self.strided_batch_run_layout(run, lhs_op, rhs_op)?;
        let cache_slot = descriptor_slot::<T>(
            cpu,
            &layout,
            [lhs.offset(), rhs.offset(), output.offset()],
            [lhs_op, rhs_op],
        );
        self.matmul_strided_batch_run_layout_typed(
            cpu, output, lhs, rhs, layout, cache_slot, lhs_op, rhs_op, alpha, beta, error_op,
            wrap_write, wrap_read,
        )
    }

    fn strided_batch_run_layout(
        &self,
        run: &[DenseGemmBatchJob],
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
    ) -> Result<StridedBatchRunLayout, DenseError> {
        let first = &run[0];
        let run_len = run.len();
        // Two jobs define the structural batch stride. A single job never
        // steps it, so its stride is the matrix extent: a dense, self-disjoint
        // value every view validator accepts.
        let (lhs_batch_stride, rhs_batch_stride, dst_batch_stride) = match run.get(1) {
            Some(next) => {
                let step = |next: usize, first: usize| {
                    next.checked_sub(first)
                        .ok_or(DenseError::OffsetOverflow { value: first })
                };
                (
                    step(next.lhs_offset, first.lhs_offset)?,
                    step(next.rhs_offset, first.rhs_offset)?,
                    step(next.dst_offset, first.dst_offset)?,
                )
            }
            None => {
                let extent = |a: usize, b: usize, offset: usize| {
                    a.checked_mul(b)
                        .ok_or(DenseError::OffsetOverflow { value: offset })
                };
                (
                    extent(first.rows, first.contracted, first.lhs_offset)?,
                    extent(first.contracted, first.cols, first.rhs_offset)?,
                    extent(first.rows, first.cols, first.dst_offset)?,
                )
            }
        };
        let lhs_shape = [first.rows, first.contracted, run_len];
        let lhs_strides = match lhs_op {
            MatrixOp::Identity => [1, first.rows, lhs_batch_stride],
            MatrixOp::Transpose | MatrixOp::Adjoint => [first.contracted, 1, lhs_batch_stride],
        };
        let rhs_shape = [first.contracted, first.cols, run_len];
        let rhs_strides = match rhs_op {
            MatrixOp::Identity => [1, first.contracted, rhs_batch_stride],
            MatrixOp::Transpose | MatrixOp::Adjoint => [first.cols, 1, rhs_batch_stride],
        };
        let dst_shape = [first.rows, first.cols, run_len];
        let dst_strides = [1, first.rows, dst_batch_stride];
        Ok(StridedBatchRunLayout {
            lhs_shape,
            lhs_strides,
            lhs_job_offset: first.lhs_offset,
            rhs_shape,
            rhs_strides,
            rhs_job_offset: first.rhs_offset,
            dst_shape,
            dst_strides,
            dst_job_offset: first.dst_offset,
        })
    }

    fn strided_batch_run_layout_admitted<T, U>(
        &self,
        output: (&[U], usize),
        lhs: DenseView<'_, T>,
        rhs: DenseView<'_, T>,
        layout: &StridedBatchRunLayout,
    ) -> bool {
        let Some(lhs_offset) = lhs.offset().checked_add(layout.lhs_job_offset) else {
            return false;
        };
        let Some(rhs_offset) = rhs.offset().checked_add(layout.rhs_job_offset) else {
            return false;
        };
        let Some(dst_offset) = output.1.checked_add(layout.dst_job_offset) else {
            return false;
        };
        let offsets_fit = [lhs_offset, rhs_offset, dst_offset]
            .into_iter()
            .all(|offset| isize::try_from(offset).is_ok());
        let strides_fit = layout
            .lhs_strides
            .into_iter()
            .chain(layout.rhs_strides)
            .chain(layout.dst_strides)
            .all(|stride| isize::try_from(stride).is_ok());
        offsets_fit
            && strides_fit
            && DenseView::new(
                lhs.data(),
                &layout.lhs_shape,
                &layout.lhs_strides,
                lhs_offset,
            )
            .is_ok()
            && DenseView::new(
                rhs.data(),
                &layout.rhs_shape,
                &layout.rhs_strides,
                rhs_offset,
            )
            .is_ok()
            && DenseView::new(output.0, &layout.dst_shape, &layout.dst_strides, dst_offset).is_ok()
    }

    #[allow(clippy::too_many_arguments)]
    fn matmul_strided_batch_run_layout_typed<T, W, R>(
        &mut self,
        cpu: &mut CpuEngine,
        output: &mut DenseViewMut<'_, T>,
        lhs: DenseView<'_, T>,
        rhs: DenseView<'_, T>,
        layout: StridedBatchRunLayout,
        cache_slot: usize,
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
        alpha: DenseScalar,
        beta: DenseScalar,
        error_op: &'static str,
        wrap_write: W,
        wrap_read: R,
    ) -> Result<(), DenseError>
    where
        T: 'static,
        W: for<'x> Fn(DenseViewMut<'x, T>) -> DenseWrite<'x>,
        R: for<'x> Fn(DenseView<'x, T>) -> DenseRead<'x>,
    {
        let lhs_offset = batch_offset(lhs.offset(), layout.lhs_job_offset)?;
        let lhs_view = DenseView::new(
            lhs.data(),
            &layout.lhs_shape,
            &layout.lhs_strides,
            lhs_offset,
        )?;
        let rhs_offset = batch_offset(rhs.offset(), layout.rhs_job_offset)?;
        let rhs_view = DenseView::new(
            rhs.data(),
            &layout.rhs_shape,
            &layout.rhs_strides,
            rhs_offset,
        )?;
        let dst_offset = batch_offset(output.offset(), layout.dst_job_offset)?;
        let dst_view = DenseViewMut::new(
            output.data_mut(),
            &layout.dst_shape,
            &layout.dst_strides,
            dst_offset,
        )?;
        let lhs = TensorRead::from_view(tenferro_view(wrap_read(lhs_view))?);
        let rhs = TensorRead::from_view(tenferro_view(wrap_read(rhs_view))?);
        let output = TensorWrite::from_view(tenferro_view_mut(wrap_write(dst_view))?);
        let accumulation = tenferro_tensor::DotGeneralAccumulation {
            lhs_conj: lhs_op == MatrixOp::Adjoint,
            rhs_conj: rhs_op == MatrixOp::Adjoint,
            alpha: tenferro_scalar(alpha),
            beta: tenferro_scalar(beta),
        };
        #[cfg(test)]
        {
            self.seam_dispatches += 1;
        }
        cpu.dot_general_read_into_accum_cached(
            Some(cache_slot),
            lhs,
            rhs,
            &self.strided_batch_matmul_config,
            accumulation,
            output,
        )
        .map_err(|err| tenferro_error(error_op, err))
    }
}

impl Default for DefaultDenseExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl DenseExecutor for DefaultDenseExecutor {
    fn supports_svd_full(&self) -> bool {
        #[cfg(feature = "cpu-faer")]
        {
            // The faer provider factorizes the full matrices natively (its
            // `SvdFull` batch on the lanes), so it never reaches the
            // `[U1 | I]` QR completion.
            true
        }
        #[cfg(not(feature = "cpu-faer"))]
        {
            false
        }
    }

    fn svd_full_owned(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseOwned,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        if !self.supports_svd_full() {
            return Err(DenseError::Unsupported {
                op: "svd_full_owned",
                message: "executor does not implement owned full-matrices SVD".to_string(),
            });
        }
        let expected = rows
            .checked_mul(cols)
            .ok_or(DenseError::ElementCountOverflow)?;
        let actual = match &input {
            DenseOwned::F32(data) => data.len(),
            DenseOwned::F64(data) => data.len(),
            DenseOwned::C32(data) => data.len(),
            DenseOwned::C64(data) => data.len(),
        };
        if actual != expected {
            return Err(DenseError::ShapeMismatch {
                op: "svd_full_owned",
                expected: vec![expected],
                actual: vec![actual],
            });
        }
        if rows == 0 || cols == 0 {
            return Err(DenseError::Unsupported {
                op: "svd_full_owned",
                message: "zero-extent full-matrices SVD is unsupported".to_string(),
            });
        }
        let input = match input {
            DenseOwned::F32(data) => {
                tenferro_tensor::Tensor::from_vec_col_major(vec![rows, cols], data)
            }
            DenseOwned::F64(data) => {
                tenferro_tensor::Tensor::from_vec_col_major(vec![rows, cols], data)
            }
            DenseOwned::C32(data) => {
                tenferro_tensor::Tensor::from_vec_col_major(vec![rows, cols], data)
            }
            DenseOwned::C64(data) => {
                tenferro_tensor::Tensor::from_vec_col_major(vec![rows, cols], data)
            }
        }
        .map_err(|err| tenferro_error("svd_full_owned", err))?;
        #[cfg(all(test, feature = "cpu-faer"))]
        {
            fn slice_ptr<T: tenferro_tensor::TensorScalar>(
                input: &tenferro_tensor::Tensor,
            ) -> Option<usize> {
                input
                    .as_typed::<T>()
                    .map(|tensor| tensor.as_slice().unwrap().as_ptr() as usize)
            }
            let pointer = slice_ptr::<f32>(&input)
                .or_else(|| slice_ptr::<f64>(&input))
                .or_else(|| slice_ptr::<num_complex::Complex32>(&input))
                .or_else(|| slice_ptr::<num_complex::Complex64>(&input))
                .expect("DenseOwned only contains supported full-SVD dtypes");
            crate::probe::owned_full_svd_input(pointer);
        }
        let outputs = host_cpu(engine, "svd_full_owned", |cpu| {
            LinalgBackend::svd_full(cpu, &input)
                .map_err(|err| tenferro_error("svd_full_owned", err))
        })?;
        if outputs.len() != 3 {
            return Err(crate::executor::arity_mismatch(
                "svd_full_owned",
                3,
                outputs.len(),
            ));
        }
        outputs
            .into_iter()
            .map(DenseTensor::from_tenferro)
            .collect()
    }

    fn svd(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        let input = tenferro_view(input)?;
        with_cpu_linalg(engine, "svd_read", |exec| {
            TensorRead::from_view(input).svd(exec)
        })
        .map_err(|err| tenferro_error("svd_read", err))
        .and_then(|svd| {
            vec![svd.u, svd.s, svd.vh]
                .into_iter()
                .map(DenseTensor::from_tenferro)
                .collect()
        })
    }

    fn qr(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        let input = tenferro_view(input)?;
        with_cpu_linalg(engine, "qr_read", |exec| {
            TensorRead::from_view(input).qr(exec)
        })
        .map_err(|err| tenferro_error("qr_read", err))
        .and_then(|qr| {
            vec![qr.q, qr.r]
                .into_iter()
                .map(DenseTensor::from_tenferro)
                .collect()
        })
    }

    fn eig(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        eig_read(engine, input)
    }

    fn eigh(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        let input = tenferro_view(input)?;
        with_cpu_linalg(engine, "eigh_read", |exec| {
            TensorRead::from_view(input).eigh(exec)
        })
        .map_err(|err| tenferro_error("eigh_read", err))
        .and_then(|eigh| {
            vec![eigh.values, eigh.vectors]
                .into_iter()
                .map(DenseTensor::from_tenferro)
                .collect()
        })
    }

    // The native provider's factorizations run the blocks on the lane driver
    // (`block_kernel::factor`); a build without `cpu-faer` runs them one
    // after another (`factorize_batch_serial`).
    fn factorize_batch(
        &mut self,
        engine: &mut dyn Engine,
        op: crate::DenseFactorization,
        inputs: &[DenseRead<'_>],
        sink: &mut dyn FnMut(usize, Result<Vec<DenseTensor>, DenseError>) -> Result<(), DenseError>,
    ) -> Result<(), DenseError> {
        crate::executor::refuse_destination_only(op)?;
        if inputs.is_empty() {
            return Ok(());
        }
        #[cfg(feature = "cpu-faer")]
        return crate::block_kernel::factor::factorize(engine, op, inputs, sink);
        #[cfg(not(feature = "cpu-faer"))]
        {
            if matches!(
                op,
                crate::DenseFactorization::QrFull | crate::DenseFactorization::QrNull
            ) {
                // Without `cpu-faer`: the executor default's `[X I]` augmentation
                // through this executor's own `qr_into`, one input after
                // another.
                let null = op == crate::DenseFactorization::QrNull;
                for (index, &input) in inputs.iter().enumerate() {
                    let outcome = crate::executor::full_qr_by_qr_into(self, engine, input, null);
                    sink(index, outcome)?;
                }
                return Ok(());
            }
            if op == crate::DenseFactorization::SvdFull {
                // Without `cpu-faer`: the executor default's per-matrix
                // `svd_full_owned`, one input after another.
                for (index, &input) in inputs.iter().enumerate() {
                    let outcome = crate::executor::full_svd_by_entries(self, engine, input);
                    sink(index, outcome)?;
                }
                return Ok(());
            }
            factorize_batch_serial(engine, op, inputs, sink)
        }
    }

    // Without `cpu-faer`, the trait default's routes (the owned
    // `factorize_batch`, and the polar and exponential composites of
    // per-matrix calls).
    #[cfg(not(feature = "cpu-faer"))]
    fn factorize_batch_into(
        &mut self,
        engine: &mut dyn Engine,
        op: crate::DenseFactorization,
        inputs: &[DenseRead<'_>],
        dests: &mut [crate::FactorDest<'_>],
        sink: &mut dyn FnMut(
            usize,
            &mut crate::FactorDest<'_>,
            Result<(), crate::FactorFailure>,
        ) -> Result<(), DenseError>,
    ) -> Result<(), DenseError> {
        crate::factor_dest::validate(op, inputs, dests)?;
        crate::factor_dest::factorize_batch_into_validated(self, engine, op, inputs, dests, sink)
    }

    // The native provider writes each block's factors into its destination
    // on the lanes.
    #[cfg(feature = "cpu-faer")]
    fn factorize_batch_into(
        &mut self,
        engine: &mut dyn Engine,
        op: crate::DenseFactorization,
        inputs: &[DenseRead<'_>],
        dests: &mut [crate::FactorDest<'_>],
        sink: &mut dyn FnMut(
            usize,
            &mut crate::FactorDest<'_>,
            Result<(), crate::FactorFailure>,
        ) -> Result<(), DenseError>,
    ) -> Result<(), DenseError> {
        crate::factor_dest::validate(op, inputs, dests)?;
        if inputs.is_empty() {
            return Ok(());
        }
        crate::block_kernel::factor::factorize_into(engine, op, inputs, dests, sink)
    }

    // The native provider solves the valid compact rank-two systems on the
    // lanes.
    #[cfg(feature = "cpu-faer")]
    fn solve_batch_into(
        &mut self,
        engine: &mut dyn Engine,
        a: &[DenseRead<'_>],
        b: &[DenseRead<'_>],
        x: &mut [DenseWrite<'_>],
    ) -> Result<(), DenseError> {
        crate::executor::check_solve_batch(a, b, x)?;
        crate::block_kernel::solve::solve_batch_into(self, engine, a, b, x)
    }

    fn solve_into(
        &mut self,
        engine: &mut dyn Engine,
        a: DenseRead<'_>,
        b: DenseRead<'_>,
        x: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        if x.dtype() != b.dtype() {
            return Err(DenseError::DTypeMismatch {
                op: "solve_into",
                expected: b.dtype(),
                actual: x.dtype(),
            });
        }
        if x.shape() != b.shape() {
            return Err(DenseError::ShapeMismatch {
                op: "solve_into",
                expected: b.shape().to_vec(),
                actual: x.shape().to_vec(),
            });
        }
        let a = tenferro_view(a)?;
        let b = tenferro_view(b)?;
        let x = tenferro_view_mut(x)?;
        with_cpu_linalg(engine, "solve_into", |exec| {
            TensorRead::from_view(a).solve_into(
                &TensorRead::from_view(b),
                exec,
                Dest::overwrite(TensorWrite::from_view(x)),
            )
        })
        .map_err(tenferro_solve_error)
    }

    // SVD and EIGH values-only calls use Tenferro's documented borrowed
    // no-vector extensions. General EIG below uses native owned `eigvals`.
    fn svd_vals(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<DenseTensor, DenseError> {
        let input = tenferro_view(input)?;
        with_cpu_linalg(engine, "svd_values", |exec| {
            TensorRead::from_view(input).svdvals(exec)
        })
        .map_err(|err| tenferro_error("svd_values", err))
        .and_then(DenseTensor::from_tenferro)
    }

    fn eigh_vals(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<DenseTensor, DenseError> {
        let input = tenferro_view(input)?;
        with_cpu_linalg(engine, "eigh_values", |exec| {
            TensorRead::from_view(input).eigvalsh(exec)
        })
        .map_err(|err| tenferro_error("eigh_values", err))
        .and_then(DenseTensor::from_tenferro)
    }

    fn eig_vals(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<DenseTensor, DenseError> {
        let input = tenferro_view(input)?;
        // One engine lock for the contiguity pre-pass and the values call.
        with_cpu_linalg(engine, "eig_values", |exec| {
            let owned = exec.to_contiguous_read(TensorRead::from_view(input))?;
            owned.eigvals(exec)
        })
        .map_err(|err| tenferro_error("eig_values", err))
        .and_then(DenseTensor::from_tenferro)
    }

    fn dot_general_into(
        &mut self,
        engine: &mut dyn Engine,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        config: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        let lhs = TensorRead::from_view(tenferro_view(lhs)?);
        let rhs = TensorRead::from_view(tenferro_view(rhs)?);
        let output = TensorWrite::from_view(tenferro_view_mut(output)?);
        let dot_config = tenferro_dot_config(config);
        // Non-conjugating path stays byte-identical to the plain read_into
        // (which itself just wraps an overwrite accumulation). Conjugation is
        // folded into the kernel via the accumulation's conj flags — no
        // conjugated operand copy — instead of falling back to a scalar loop.
        if config.lhs_conj() || config.rhs_conj() {
            let mut accumulation = tenferro_tensor::DotGeneralAccumulation::overwrite(lhs.dtype())
                .map_err(|err| tenferro_error("dot_general_accum", err))?;
            accumulation.lhs_conj = config.lhs_conj();
            accumulation.rhs_conj = config.rhs_conj();
            with_cpu_linalg(engine, "dot_general_accum", |cpu| {
                cpu.dot_general_read_into_accum(lhs, rhs, &dot_config, accumulation, output)
            })
            .map_err(|err| tenferro_error("dot_general_accum", err))
        } else {
            with_cpu_linalg(engine, "dot_general_read_into", |cpu| {
                cpu.dot_general_read_into(lhs, rhs, &dot_config, output)
            })
            .map_err(|err| tenferro_error("dot_general_read_into", err))
        }
    }

    fn matmul_into(
        &mut self,
        engine: &mut dyn Engine,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
    ) -> Result<(), DenseError> {
        // GEMM backend selection is owned by tenferro; this seam only
        // lowers views and reuses the cached rank-2 contraction config.
        let lhs = TensorRead::from_view(tenferro_view(lhs)?);
        let rhs = TensorRead::from_view(tenferro_view(rhs)?);
        let output = TensorWrite::from_view(tenferro_view_mut(output)?);
        let config = &self.matmul_config;
        with_cpu_linalg(engine, "dot_general_read_into", |cpu| {
            cpu.dot_general_read_into(lhs, rhs, config, output)
        })
        .map_err(|err| tenferro_error("dot_general_read_into", err))
    }

    fn matmul_axpby_into(
        &mut self,
        engine: &mut dyn Engine,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        alpha: DenseScalar,
        beta: DenseScalar,
    ) -> Result<(), DenseError> {
        // Overwrite case keeps the cached-config fast path.
        if alpha.is_one() && beta.is_zero() {
            return self.matmul_into(engine, output, lhs, rhs);
        }
        let lhs = TensorRead::from_view(tenferro_view(lhs)?);
        let rhs = TensorRead::from_view(tenferro_view(rhs)?);
        let output = TensorWrite::from_view(tenferro_view_mut(output)?);
        let accumulation = tenferro_tensor::DotGeneralAccumulation {
            lhs_conj: false,
            rhs_conj: false,
            alpha: tenferro_scalar(alpha),
            beta: tenferro_scalar(beta),
        };
        let config = &self.matmul_config;
        with_cpu_linalg(engine, "dot_general_accum", |cpu| {
            cpu.dot_general_read_into_accum(lhs, rhs, config, accumulation, output)
        })
        .map_err(|err| tenferro_error("dot_general_accum", err))
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
        self.matmul_batch_axpby_dispatch(
            engine,
            output,
            lhs,
            rhs,
            jobs,
            runs,
            [MatrixOp::Identity; 2],
            alpha,
            beta,
        )
    }

    #[allow(clippy::redundant_closure)]
    fn matmul_batch_axpby_with_ops_into(
        &mut self,
        engine: &mut dyn Engine,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        jobs: &[DenseGemmBatchJob],
        runs: &[usize],
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
        alpha: DenseScalar,
        beta: DenseScalar,
    ) -> Result<(), DenseError> {
        if lhs_op == MatrixOp::Identity && rhs_op == MatrixOp::Identity {
            return self.matmul_batch_axpby_into(engine, output, lhs, rhs, jobs, runs, alpha, beta);
        }
        self.matmul_batch_axpby_dispatch(
            engine,
            output,
            lhs,
            rhs,
            jobs,
            runs,
            [lhs_op, rhs_op],
            alpha,
            beta,
        )
    }

    fn matmul_batch_fresh<'a>(
        &mut self,
        engine: &mut dyn Engine,
        segments: DenseFreshSegments<'_, 'a>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        jobs: &[DenseGemmBatchJob],
        ops: [MatrixOp; 2],
        alpha: DenseScalar,
    ) -> Result<(), DenseError> {
        host_cpu(engine, FRESH_OP, |cpu| {
            dispatch_fresh!(FRESH_OP, segments, lhs, rhs, |parts, lhs, rhs| {
                self.matmul_batch_fresh_typed(cpu, parts, lhs, rhs, jobs, ops, alpha)
            })
        })
    }

    // The grouped GEMMs are placed by the engine's batch decision.
    fn honors_forced_batch_policy(&self) -> bool {
        true
    }
}

impl DefaultDenseExecutor {
    /// The fresh batch, routed as [`Self::matmul_batch_axpby_route_typed`]
    /// routes the same batch, so a fresh destination changes no kernel
    /// choice except where stated:
    ///
    /// - one job, or one self-disjoint affine run whose outputs are gapless
    ///   and inside one segment, takes the initialized strided call it takes
    ///   today on its piece, zero-filled first through the carrier (the
    ///   strided entry has no uninitialized form);
    /// - an affine run with gaps between its outputs goes grouped: separate
    ///   pieces cannot form the one rank-3 destination view;
    /// - everything else goes grouped: the block kernel's lanes with
    ///   tcontract's fresh entry.
    ///
    /// On every route `alpha == 0` gives zero outputs whatever the operands
    /// hold.
    #[allow(clippy::too_many_arguments)]
    fn matmul_batch_fresh_typed<'a, T: GroupedScalar + FreshScalar>(
        &mut self,
        cpu: &mut CpuEngine,
        parts: FreshParts<'_, 'a, T>,
        lhs: DenseView<'_, T>,
        rhs: DenseView<'_, T>,
        jobs: &[DenseGemmBatchJob],
        ops: [MatrixOp; 2],
        alpha: DenseScalar,
    ) -> Result<(), DenseError> {
        let Some(typed_alpha) = T::from_dense(alpha) else {
            return Err(grouped_error(
                "alpha dtype differs from the operands".to_string(),
            ));
        };
        let Some(&first) = jobs.first() else {
            return Ok(());
        };
        let beta = T::scalar(T::default());
        let identity = ops == [MatrixOp::Identity; 2];
        if jobs.len() == 1 {
            // Why not `split_fresh_segments`: its staged order would be this
            // route's only allocation on a per-call executor.
            let Some(piece) = split_single_fresh_segment(parts.segments, &first)? else {
                (parts.written)(0, &mut []);
                return Ok(());
            };
            let output = zero_piece(parts.zeroed, piece)?;
            let (shape, strides) = ([output.len()], [1]);
            self.matmul_batch_initialized_typed(
                cpu,
                &mut DenseViewMut::new(&mut *output, &shape, &strides, 0)?,
                lhs,
                rhs,
                &[DenseGemmBatchJob {
                    dst_offset: 0,
                    ..first
                }],
                &[1],
                ops,
                alpha,
                beta,
            )?;
            (parts.written)(0, output);
            return Ok(());
        }
        let block = first.rows.checked_mul(first.cols);
        let gapless_run = strided_batch_run_len(jobs, 0) == jobs.len()
            && block.is_some_and(|block| block != 0)
            && jobs[1].dst_offset.checked_sub(first.dst_offset) == block;
        if let (true, Some(block)) = (gapless_run, block) {
            let len = block
                .checked_mul(jobs.len())
                .ok_or(DenseError::ElementCountOverflow)?;
            let start = first.dst_offset;
            let layout = StridedBatchRunLayout {
                dst_job_offset: 0,
                ..self.strided_batch_run_layout(jobs, ops[0], ops[1])?
            };
            let segment = parts.segments.iter().position(|(base, memory)| {
                start >= *base
                    && (start - base)
                        .checked_add(len)
                        .is_some_and(|end| end <= memory.len())
            });
            let admitted = |memory: &[MaybeUninit<T>]| {
                identity
                    || first.rows != 0
                        && first.contracted != 0
                        && first.cols != 0
                        && self.strided_batch_run_layout_admitted((memory, 0), lhs, rhs, &layout)
            };
            if let Some(segment) = segment.filter(|&segment| {
                let (base, memory) = &parts.segments[segment];
                admitted(&memory[start - base..start - base + len])
            }) {
                let (base, memory) = &mut parts.segments[segment];
                let (_, rest) = std::mem::take(memory).split_at_mut(start - *base);
                let output = zero_piece(parts.zeroed, rest.split_at_mut(len).0)?;
                let (shape, strides) = ([len], [1]);
                let cache_slot =
                    descriptor_slot::<T>(cpu, &layout, [lhs.offset(), rhs.offset(), 0], ops);
                self.matmul_strided_batch_run_layout_typed(
                    cpu,
                    &mut DenseViewMut::new(&mut *output, &shape, &strides, 0)?,
                    lhs,
                    rhs,
                    layout,
                    cache_slot,
                    ops[0],
                    ops[1],
                    alpha,
                    beta,
                    if identity {
                        "strided_batch_gemm"
                    } else {
                        "matmul_batch_axpby_with_ops_into"
                    },
                    T::write,
                    T::read,
                )?;
                for (j, piece) in output.chunks_exact_mut(block).enumerate() {
                    (parts.written)(j, piece);
                }
                return Ok(());
            }
        }
        #[cfg(test)]
        {
            self.grouped_route_jobs = jobs.len();
            self.seam_dispatches += 1;
        }
        crate::block_kernel::gemm::gemm_batch_fresh(
            cpu,
            &mut self.gemm_scratch,
            parts,
            lhs,
            rhs,
            jobs,
            ops,
            typed_alpha,
        )
    }

    /// The initialized batch route of [`Self::matmul_batch_axpby_with_ops_into`]
    /// for typed operands.
    #[allow(clippy::too_many_arguments)]
    fn matmul_batch_initialized_typed<T: GroupedScalar + FreshScalar>(
        &mut self,
        cpu: &mut CpuEngine,
        output: &mut DenseViewMut<'_, T>,
        lhs: DenseView<'_, T>,
        rhs: DenseView<'_, T>,
        jobs: &[DenseGemmBatchJob],
        runs: &[usize],
        ops: [MatrixOp; 2],
        alpha: DenseScalar,
        beta: DenseScalar,
    ) -> Result<(), DenseError> {
        self.matmul_batch_axpby_route_typed(
            cpu,
            output,
            lhs,
            rhs,
            jobs,
            runs,
            ops,
            alpha,
            beta,
            T::write,
            T::read,
        )
    }
}

impl DefaultDenseExecutor {}

/// The four GEMM dtypes of the grouped route, with their [`DenseScalar`]
/// projection (a mismatched coefficient dtype is refused before any write).
trait GroupedScalar: tcontract::api::Scalar + Default {
    fn from_dense(value: DenseScalar) -> Option<Self>;
}

macro_rules! grouped_scalar {
    ($($ty:ty => $variant:ident);+) => {$(
        impl GroupedScalar for $ty {
            fn from_dense(value: DenseScalar) -> Option<Self> {
                match value {
                    DenseScalar::$variant(value) => Some(value),
                    _ => None,
                }
            }
        }
    )+};
}
grouped_scalar!(f32 => F32; f64 => F64; Complex32 => C32; Complex64 => C64);

pub(crate) fn grouped_error(message: String) -> DenseError {
    DenseError::Backend {
        backend: DenseBackend::Tenferro,
        op: "grouped_gemm",
        message,
    }
}

/// The logical extent of a column-major contiguous view: its elements are
/// `data[offset..offset + len]` in job-offset order.
fn grouped_span(shape: &[usize], strides: &[usize], role: &str) -> Result<usize, DenseError> {
    let mut len = 1usize;
    for (&dim, &stride) in shape.iter().zip(strides) {
        if dim > 1 && stride != len {
            return Err(DenseError::Unsupported {
                op: "grouped_gemm",
                message: format!("grouped GEMM {role} must be column-major contiguous"),
            });
        }
        len = len
            .checked_mul(dim)
            .ok_or(DenseError::ElementCountOverflow)?;
    }
    Ok(len)
}

/// Checks every job's lhs/rhs/output range in job-index order, so the first
/// out-of-range job reported is the lowest-indexed one (`GroupedPlan` checks
/// in its sorted output order instead). Output overlap is left to the
/// caller (the block kernel), which rejects it before any write.
pub(crate) fn grouped_validate<T>(
    output: &DenseViewMut<'_, T>,
    lhs: DenseView<'_, T>,
    rhs: DenseView<'_, T>,
    jobs: &[DenseGemmBatchJob],
) -> Result<[usize; 3], DenseError> {
    let lens = [
        grouped_span(lhs.shape(), lhs.strides(), "lhs")?,
        grouped_span(rhs.shape(), rhs.strides(), "rhs")?,
        grouped_span(output.shape(), output.strides(), "output")?,
    ];
    grouped_validate_jobs(lens, jobs)?;
    Ok(lens)
}

/// [`grouped_validate`]'s operand checks for a fresh destination, whose job
/// placement the fresh segment split checks instead.
pub(crate) fn grouped_operand_lens<T>(
    lhs: DenseView<'_, T>,
    rhs: DenseView<'_, T>,
    jobs: &[DenseGemmBatchJob],
) -> Result<[usize; 2], DenseError> {
    let lens = [
        grouped_span(lhs.shape(), lhs.strides(), "lhs")?,
        grouped_span(rhs.shape(), rhs.strides(), "rhs")?,
    ];
    grouped_validate_jobs([lens[0], lens[1], usize::MAX], jobs)?;
    Ok(lens)
}

fn grouped_validate_jobs(lens: [usize; 3], jobs: &[DenseGemmBatchJob]) -> Result<(), DenseError> {
    for (index, job) in jobs.iter().enumerate() {
        for (role, offset, rows, cols, len) in [
            ("lhs", job.lhs_offset, job.rows, job.contracted, lens[0]),
            ("rhs", job.rhs_offset, job.contracted, job.cols, lens[1]),
            ("output", job.dst_offset, job.rows, job.cols, lens[2]),
        ] {
            let size = rows
                .checked_mul(cols)
                .ok_or(DenseError::ElementCountOverflow)?;
            let end = offset
                .checked_add(size)
                .ok_or(DenseError::OffsetOverflow { value: offset })?;
            if size != 0 && end > len {
                return Err(grouped_error(format!(
                    "job {index} {role} range {offset}..{end} exceeds buffer length {len}"
                )));
            }
        }
    }
    Ok(())
}

fn tenferro_scalar(value: DenseScalar) -> tenferro_tensor::ContractionScalar {
    match value {
        DenseScalar::F32(value) => tenferro_tensor::ContractionScalar::F32(value),
        DenseScalar::F64(value) => tenferro_tensor::ContractionScalar::F64(value),
        DenseScalar::C32(value) => tenferro_tensor::ContractionScalar::C32(value),
        DenseScalar::C64(value) => tenferro_tensor::ContractionScalar::C64(value),
    }
}

pub(crate) fn tenferro_view(input: DenseRead<'_>) -> Result<TensorView<'_>, DenseError> {
    match input {
        DenseRead::F32(view) => typed_tenferro_view(view).map(TensorView::F32),
        DenseRead::F64(view) => typed_tenferro_view(view).map(TensorView::F64),
        DenseRead::I32(view) => typed_tenferro_view(view).map(TensorView::I32),
        DenseRead::I64(view) => typed_tenferro_view(view).map(TensorView::I64),
        DenseRead::Bool(view) => typed_tenferro_view(view).map(TensorView::Bool),
        DenseRead::C32(view) => typed_tenferro_view(view).map(TensorView::C32),
        DenseRead::C64(view) => typed_tenferro_view(view).map(TensorView::C64),
    }
}

fn tenferro_view_mut(output: DenseWrite<'_>) -> Result<TensorViewMut<'_>, DenseError> {
    match output {
        DenseWrite::F32(view) => typed_tenferro_view_mut(view).map(TensorViewMut::F32),
        DenseWrite::F64(view) => typed_tenferro_view_mut(view).map(TensorViewMut::F64),
        DenseWrite::I32(view) => typed_tenferro_view_mut(view).map(TensorViewMut::I32),
        DenseWrite::I64(view) => typed_tenferro_view_mut(view).map(TensorViewMut::I64),
        DenseWrite::Bool(view) => typed_tenferro_view_mut(view).map(TensorViewMut::Bool),
        DenseWrite::C32(view) => typed_tenferro_view_mut(view).map(TensorViewMut::C32),
        DenseWrite::C64(view) => typed_tenferro_view_mut(view).map(TensorViewMut::C64),
    }
}

fn typed_tenferro_view<'a, T: 'static>(
    view: DenseView<'a, T>,
) -> Result<TypedTensorView<'a, T>, DenseError> {
    let offset = isize::try_from(view.offset()).map_err(|_| DenseError::OffsetOverflow {
        value: view.offset(),
    })?;
    if view.shape().len() == 1 && view.strides() == [1] {
        return TypedTensorView::from_slice([view.shape()[0]], [1_isize], offset, view.data())
            .map_err(|err| tenferro_error("TypedTensorView::from_slice", err));
    }
    let strides = strides_to_isize(view.strides())?;
    TypedTensorView::from_slice(view.shape(), strides, offset, view.data())
        .map_err(|err| tenferro_error("TypedTensorView::from_slice", err))
}

fn typed_tenferro_view_mut<'a, T: 'static>(
    view: DenseViewMut<'a, T>,
) -> Result<TypedTensorViewMut<'a, T>, DenseError> {
    let DenseViewMut {
        data,
        shape,
        strides,
        offset,
    } = view;
    let offset =
        isize::try_from(offset).map_err(|_| DenseError::OffsetOverflow { value: offset })?;
    if shape.len() == 1 && strides == [1] {
        return TypedTensorViewMut::from_slice([shape[0]], [1_isize], offset, data)
            .map_err(|err| tenferro_error("TypedTensorViewMut::from_slice", err));
    }
    let strides = strides_to_isize(strides)?;
    TypedTensorViewMut::from_slice(shape, strides, offset, data)
        .map_err(|err| tenferro_error("TypedTensorViewMut::from_slice", err))
}

fn tenferro_dot_config(config: &DenseDotConfig) -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: config.lhs_contracting_dims().iter().copied().collect(),
        rhs_contracting_dims: config.rhs_contracting_dims().iter().copied().collect(),
        lhs_batch_dims: config.lhs_batch_dims().iter().copied().collect(),
        rhs_batch_dims: config.rhs_batch_dims().iter().copied().collect(),
    }
}

pub(crate) fn tenferro_error(op: &'static str, err: tenferro_tensor::Error) -> DenseError {
    if err.kind() == tenferro_tensor::ErrorKind::Unsupported {
        return DenseError::Unsupported {
            op,
            message: err.to_string(),
        };
    }
    DenseError::Backend {
        backend: DenseBackend::Tenferro,
        op,
        message: err.to_string(),
    }
}

pub(crate) fn tenferro_solve_error(err: tenferro_tensor::Error) -> DenseError {
    if err.kind() == tenferro_tensor::ErrorKind::NumericalFailure {
        DenseError::NumericalFailure {
            backend: DenseBackend::Tenferro,
            op: "solve_into",
            message: err.to_string(),
        }
    } else {
        tenferro_error("solve_into", err)
    }
}

pub(crate) const NOT_A_CPU_ENGINE: &str =
    "DefaultDenseExecutor runs Host views and needs the caller's CPU engine";

/// Runs `f` on the CPU engine behind the caller's erased engine; any other
/// engine is refused before work.
fn with_cpu_linalg<R>(
    engine: &mut dyn Engine,
    op: &'static str,
    f: impl FnOnce(&mut dyn Engine) -> tenferro_tensor::Result<R>,
) -> tenferro_tensor::Result<R> {
    with_cpu_engine(engine, |cpu| f(cpu))
        .unwrap_or_else(|| Err(tenferro_tensor::Error::unsupported(op, NOT_A_CPU_ENGINE)))
}

/// [`with_cpu_linalg`] for symtn-side batch routes that need the concrete
/// engine (its execution bundle).
fn host_cpu<R>(
    engine: &mut dyn Engine,
    op: &'static str,
    f: impl FnOnce(&mut CpuEngine) -> Result<R, DenseError>,
) -> Result<R, DenseError> {
    with_cpu_engine(engine, f).unwrap_or_else(|| {
        Err(DenseError::Unsupported {
            op,
            message: NOT_A_CPU_ENGINE.to_string(),
        })
    })
}

/// tenferro's general eigendecomposition of one input on the whole engine:
/// the per-matrix [`DenseExecutor::eig`] entry.
pub(crate) fn eig_read(
    engine: &mut dyn Engine,
    input: DenseRead<'_>,
) -> Result<Vec<DenseTensor>, DenseError> {
    let input = tenferro_view(input)?;
    with_cpu_linalg(engine, "eig_read", |exec| {
        TensorRead::from_view(input).eig(exec)
    })
    .map_err(|err| tenferro_error("eig_read", err))
    .and_then(|eig| {
        vec![eig.values, eig.vectors]
            .into_iter()
            .map(DenseTensor::from_tenferro)
            .collect()
    })
}

/// A batch of per-matrix factorizations under one engine lock: the
/// per-matrix calls the per-call entries make, in the same order, each
/// input's outcome delivered to `sink` before the next input is solved.
#[cfg(not(feature = "cpu-faer"))]
fn factorize_batch_serial(
    engine: &mut dyn Engine,
    op: crate::DenseFactorization,
    inputs: &[DenseRead<'_>],
    sink: &mut dyn FnMut(usize, Result<Vec<DenseTensor>, DenseError>) -> Result<(), DenseError>,
) -> Result<(), DenseError> {
    use crate::DenseFactorization;
    let entry = match op {
        DenseFactorization::Svd => "svd_read",
        DenseFactorization::Qr => "qr_read",
        DenseFactorization::Eigh => "eigh_read",
        DenseFactorization::EighVals => "eigh_values",
        DenseFactorization::SvdVals => "svd_values",
        DenseFactorization::Eig => "eig_read",
        DenseFactorization::EigVals => "eig_values",
        DenseFactorization::Lq => "qr_read",
        DenseFactorization::QrFull | DenseFactorization::QrNull => {
            unreachable!("full QR and its null basis run through `qr_into` (`factorize_batch`)")
        }
        DenseFactorization::SvdFull => {
            unreachable!("full SVD runs through `svd_full_owned` (`factorize_batch`)")
        }
        DenseFactorization::LeftPolar
        | DenseFactorization::RightPolar
        | DenseFactorization::ExpViaEigh
        | DenseFactorization::Inv => {
            unreachable!("`factorize_batch` refuses the destination-only ops")
        }
    };
    let mut lq = crate::executor::AdjointQrLq::for_inputs(inputs);
    let solve =
        |exec: &mut dyn Engine, input: DenseRead<'_>| -> Result<Vec<DenseTensor>, DenseError> {
            let input = TensorRead::from_view(tenferro_view(input)?);
            let tensors = match op {
                DenseFactorization::Svd => input.svd_read(exec).map(|(u, s, vt)| vec![u, s, vt]),
                // LQ reaches this only with its staged adjoint (below); full
                // QR never reaches it (see `entry`).
                DenseFactorization::Qr
                | DenseFactorization::Lq
                | DenseFactorization::QrFull
                | DenseFactorization::QrNull => input.qr_read(exec).map(|(q, r)| vec![q, r]),
                DenseFactorization::SvdFull => {
                    unreachable!("full SVD never reaches the serial entries (see `entry`)")
                }
                DenseFactorization::LeftPolar
                | DenseFactorization::RightPolar
                | DenseFactorization::ExpViaEigh
                | DenseFactorization::Inv => {
                    unreachable!("`factorize_batch` refuses the destination-only ops")
                }
                DenseFactorization::Eigh => input.eigh_read(exec).map(|(w, v)| vec![w, v]),
                DenseFactorization::EighVals => input.eigvalsh_read(exec).map(|w| vec![w]),
                DenseFactorization::SvdVals => input.svdvals_read(exec).map(|s| vec![s]),
                DenseFactorization::Eig => input.eig_read(exec).map(|(w, v)| vec![w, v]),
                // The per-matrix `eig_vals` route: a contiguity pre-pass, then
                // the owned values call.
                DenseFactorization::EigVals => exec
                    .to_contiguous_read(input)
                    .and_then(|owned| owned.eigvals(exec))
                    .map(|w| vec![w]),
            }
            .map_err(|err| tenferro_error(entry, err))?;
            tensors
                .into_iter()
                .map(DenseTensor::from_tenferro)
                .collect()
        };
    with_cpu_linalg(engine, entry, |exec| {
        let mut run = || -> Result<(), DenseError> {
            for (index, &input) in inputs.iter().enumerate() {
                // LQ as the executor default runs it: QR of the staged
                // adjoint.
                let outcome = match op {
                    DenseFactorization::Lq => {
                        lq.factor(input, |adjoint| solve(&mut *exec, adjoint))
                    }
                    _ => solve(&mut *exec, input),
                };
                sink(index, outcome)?;
            }
            Ok(())
        };
        Ok(run())
    })
    .map_err(|err| tenferro_error(entry, err))?
}
