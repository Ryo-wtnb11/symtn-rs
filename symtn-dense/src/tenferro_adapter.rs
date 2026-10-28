use num_complex::{Complex32, Complex64};

use crate::executor::{batch_offset, strided_batch_run_len};
use crate::layout::strides_to_isize;
use crate::{
    DenseBackend, DenseDotConfig, DenseError, DenseExecutor, DenseGemmBatchJob, DenseOwned,
    DenseRead, DenseScalar, DenseTensor, DenseView, DenseViewMut, DenseWrite, MatrixOp,
};

use std::sync::Arc;

use std::sync::{Mutex, MutexGuard};
use tenferro_cpu::{CpuEngine, Pool};
#[cfg(not(feature = "provider-inject"))]
use tenferro_linalg::{LinalgBackend, TensorLinalgExt, TensorReadLinalgExt};
use tenferro_tensor::{
    DotGeneralConfig, Engine, SessionCachedDot, TensorDot, TensorRead, TensorView, TensorViewMut,
    TensorWrite, TypedTensorView, TypedTensorViewMut,
};

#[derive(Clone, Copy)]
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

fn run_partition_covers(jobs: &[DenseGemmBatchJob], runs: &[usize]) -> bool {
    let mut start = 0usize;
    for &run_len in runs {
        if run_len == 0 {
            return false;
        }
        let Some(end) = start.checked_add(run_len) else {
            return false;
        };
        if end > jobs.len() {
            return false;
        }
        start = end;
    }
    start == jobs.len()
}

/// One CPU engine shared by runtime-built executors: a tenferro
/// [`CpuEngine`] that owns the runtime's `threadteam` pool, behind a mutex.
///
/// Experimental (tenferro main d346af90): tenferro removed its process-wide
/// execution arbiter, and a `threadteam::Pool` belongs to exactly one engine,
/// so the runtime's executors share one engine and serialize on its lock for
/// each dense call — the same serialization the removed arbiter imposed.
// ponytail: global-per-runtime engine mutex. The approved end state (engine
// option A) passes `&mut CpuEngine` through every op instead.
#[derive(Clone, Debug)]
pub struct SharedCpuContext {
    inner: Arc<SharedEngine>,
}

#[derive(Debug)]
struct SharedEngine {
    engine: Mutex<CpuEngine>,
    threads: usize,
}

impl SharedCpuContext {
    fn from_engine(engine: CpuEngine) -> Self {
        let threads = engine.num_threads();
        Self {
            inner: Arc::new(SharedEngine {
                engine: Mutex::new(engine),
                threads,
            }),
        }
    }

    /// `RAYON_NUM_THREADS` if set, else the machine's available parallelism.
    pub fn from_env() -> Self {
        let threads = std::env::var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .filter(|&threads| threads > 0)
            .unwrap_or_else(available_parallelism);
        Self::with_threads(threads).unwrap_or_else(|_| Self::from_engine(CpuEngine::serial()))
    }

    /// Fixed thread count; `1` builds no pool at all (fully serial).
    pub fn with_threads(threads: usize) -> Result<Self, DenseError> {
        if threads <= 1 {
            return Ok(Self::from_engine(CpuEngine::serial()));
        }
        Pool::builder()
            .threads(threads)
            .build()
            .map(|pool| Self::from_engine(CpuEngine::new(pool)))
            .map_err(|err| DenseError::Backend {
                backend: DenseBackend::Tenferro,
                op: "SharedCpuContext::with_threads",
                message: err.to_string(),
            })
    }

    pub fn num_threads(&self) -> usize {
        self.inner.threads
    }

    /// One thread per CPU the process may run on.
    pub fn with_available_parallelism() -> Result<Self, DenseError> {
        Self::with_threads(available_parallelism())
    }

    fn lock(&self) -> MutexGuard<'_, CpuEngine> {
        // The engine tracks its own broken state after a panic.
        self.inner
            .engine
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Runs `op` on this context's pool (the engine's `threadteam` pool, which
    /// Rayon work inside `op` then uses). A serial context runs `op` inline.
    ///
    /// Holds the engine lock for the region, so `op` must not call a dense
    /// executor on this context, and a caller already on a pool worker must
    /// run inline instead of calling this (it would deadlock).
    #[doc(hidden)]
    pub fn install<R: Send>(&self, op: impl FnOnce() -> R + Send) -> R {
        if self.inner.threads <= 1 {
            return op();
        }
        let mut engine = self.lock();
        let result = match engine.exec() {
            Ok(exec) => exec.into_token().external(|_width| op()),
            // A broken engine refuses tokens; the region itself is still
            // correct on the calling thread.
            Err(_) => op(),
        };
        result
    }

    /// Identity check for regression tests: do two handles share one engine?
    #[doc(hidden)]
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// Address identity of the shared context, for observation probes only.
    #[doc(hidden)]
    pub fn identity(&self) -> usize {
        Arc::as_ptr(&self.inner) as usize
    }
}

fn available_parallelism() -> usize {
    std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
}

#[derive(Debug)]
pub struct DefaultDenseExecutor {
    ctx: SharedCpuContext,
    // Whether `ctx` came from `with_shared_context` (a runtime's engine).
    shared: bool,
    matmul_config: DotGeneralConfig,
    strided_batch_matmul_config: DotGeneralConfig,
    // Reused job list of the grouped route: scratch, not a plan cache.
    #[cfg(not(feature = "cpu-blas-core"))]
    grouped_jobs: Vec<cpueinsum::GroupedGemmJob>,
    // Test-only count of low-level batch seam submissions: one per strided
    // call and one per job in the serial route.
    #[cfg(test)]
    seam_dispatches: usize,
    // Test-only: jobs of the last batch routed where the grouped entry was.
    #[cfg(test)]
    grouped_route_jobs: usize,
}

impl DefaultDenseExecutor {
    pub fn new() -> Self {
        Self::from_context(SharedCpuContext::from_env(), false)
    }

    pub fn with_threads(threads: usize) -> Result<Self, DenseError> {
        SharedCpuContext::with_threads(threads).map(|ctx| Self::from_context(ctx, false))
    }

    /// Builds an executor on the runtime's shared engine.
    pub fn with_shared_context(ctx: &SharedCpuContext) -> Self {
        Self::from_context(ctx.clone(), true)
    }

    /// Regression-test hook: does this executor run on `ctx`'s engine?
    #[doc(hidden)]
    pub fn shares_cpu_context(&self, ctx: &SharedCpuContext) -> bool {
        self.shared && self.ctx.ptr_eq(ctx)
    }

    fn from_context(ctx: SharedCpuContext, shared: bool) -> Self {
        Self {
            ctx,
            shared,
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
            #[cfg(not(feature = "cpu-blas-core"))]
            grouped_jobs: Vec::new(),
            #[cfg(test)]
            seam_dispatches: 0,
            #[cfg(test)]
            grouped_route_jobs: 0,
        }
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

    /// Routes one typed identity batch by what the two tenferro entries can
    /// represent, not by a run-length cutoff. The strided entry is one rank-3
    /// contraction, so it can carry a batch iff the batch is one job (batch
    /// extent one; TensorKit's per-block `mul!`) or ONE affine run:
    /// `runs == [jobs.len()]`, at least two jobs (a step needs two offsets),
    /// every job sharing job 0's shape and the job-0-to-job-1 offset steps
    /// (`strided_batch_run_len`, O(J): `runs` is a caller claim, and a wrong
    /// one would apply job 0's shape and steps to every job), and
    /// a destination step of at least `rows * cols`, which is the O(1) proof
    /// that the rank-3 destination view is self-disjoint (tenferro's dot path
    /// does not check destination overlap; its grouped validator does). The
    /// grouped entry — TensorKit `mul!` per matched sector, QSpace
    /// `contract_matchAB_groupC` -> `contractDATA_group`, a destination-grouped
    /// GEMM list with no cutoff — represents every batch, so everything else
    /// goes there as ONE submission built from `jobs` (never from `runs`, so a
    /// malformed partition cannot skip or over-index a job).
    ///
    /// Why not send one job to the grouped entry too: tenferro's grouped
    /// validator allocates its O(J) range list on every call, so a recoupling
    /// replay that submits one GEMM per Multi block would allocate once per
    /// block on every warm call (#2184); the strided entry allocates nothing.
    ///
    /// Why not a per-run cutoff: it would be an empirical constant in tensor
    /// semantics. The disadvantage accepted here: a long run plus residual
    /// jobs goes grouped, paying tenferro's
    /// O(J log J) grouped validation per call instead of the cached O(1)
    /// strided analysis plus O(S log S) for the residual.
    #[allow(clippy::too_many_arguments)]
    fn matmul_batch_axpby_route_typed<T, W, R>(
        &mut self,
        output: &mut DenseViewMut<'_, T>,
        lhs: DenseView<'_, T>,
        rhs: DenseView<'_, T>,
        jobs: &[DenseGemmBatchJob],
        runs: &[usize],
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
        let single_self_disjoint_run = runs == [jobs.len()]
            && jobs.len() >= 2
            && strided_batch_run_len(jobs, 0) == jobs.len()
            && jobs[1]
                .dst_offset
                .checked_sub(jobs[0].dst_offset)
                .zip(jobs[0].rows.checked_mul(jobs[0].cols))
                .is_some_and(|(dst_step, block)| dst_step >= block);
        if jobs.len() == 1 || single_self_disjoint_run {
            return self.matmul_strided_batch_run_typed(
                output,
                lhs,
                rhs,
                jobs,
                0,
                MatrixOp::Identity,
                MatrixOp::Identity,
                alpha,
                beta,
                "strided_batch_gemm",
                wrap_write,
                wrap_read,
            );
        }
        #[cfg(test)]
        {
            self.grouped_route_jobs = jobs.len();
            self.seam_dispatches += 1;
        }
        #[cfg(not(feature = "cpu-blas-core"))]
        {
            let _ = (wrap_write, wrap_read);
            let mut engine = self.ctx.lock();
            grouped_gemm_into(
                &mut engine,
                &mut self.grouped_jobs,
                output,
                lhs,
                rhs,
                jobs,
                alpha,
                beta,
            )
        }
        // ponytail: a BLAS build keeps its provider, so it runs the jobs in
        // order through the engine's (BLAS) dot instead of the native
        // GroupedPlan; `cpueinsum-blas::GroupedPlan` would restore lane
        // parallelism but is a dependency nobody has approved yet.
        #[cfg(feature = "cpu-blas-core")]
        {
            grouped_validate(output, lhs, rhs, jobs)?;
            grouped_reject_overlap(jobs)?;
            #[cfg(test)]
            let dispatches = self.seam_dispatches;
            let result = self.matmul_batch_axpby_ops_serial_typed(
                output,
                lhs,
                rhs,
                jobs,
                0,
                MatrixOp::Identity,
                MatrixOp::Identity,
                alpha,
                beta,
                wrap_write,
                wrap_read,
            );
            // One grouped submission, however many jobs it ran.
            #[cfg(test)]
            {
                self.seam_dispatches = dispatches;
            }
            result
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn matmul_batch_axpby_ops_typed<T, W, R>(
        &mut self,
        output: &mut DenseViewMut<'_, T>,
        lhs: DenseView<'_, T>,
        rhs: DenseView<'_, T>,
        jobs: &[DenseGemmBatchJob],
        runs: &[usize],
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
        alpha: DenseScalar,
        beta: DenseScalar,
        wrap_write: W,
        wrap_read: R,
    ) -> Result<(), DenseError>
    where
        T: 'static,
        W: for<'x> Fn(DenseViewMut<'x, T>) -> DenseWrite<'x> + Copy,
        R: for<'x> Fn(DenseView<'x, T>) -> DenseRead<'x> + Copy,
    {
        // A valid covering partition with one entry per job contains only
        // singletons; malformed equal-count partitions need the same fallback.
        if runs.len() == jobs.len() {
            return self.matmul_batch_axpby_ops_serial_typed(
                output, lhs, rhs, jobs, 0, lhs_op, rhs_op, alpha, beta, wrap_write, wrap_read,
            );
        }
        if !run_partition_covers(jobs, runs) {
            return self.matmul_batch_axpby_ops_serial_typed(
                output, lhs, rhs, jobs, 0, lhs_op, rhs_op, alpha, beta, wrap_write, wrap_read,
            );
        }

        self.matmul_batch_axpby_ops_partition_typed(
            output, lhs, rhs, jobs, runs, lhs_op, rhs_op, alpha, beta, wrap_write, wrap_read,
        )
    }

    /// Issues a covering partition in job order: each batchable run as one
    /// strided call, and each maximal stretch of adjacent non-batchable runs
    /// as one serial call (one engine lock) instead of one call per run. Plan-cache
    /// slots are absolute job indices on both routes, so every GEMM keeps the
    /// slot, kernel and order it had with one serial call per run.
    #[allow(clippy::too_many_arguments)]
    fn matmul_batch_axpby_ops_partition_typed<T, W, R>(
        &mut self,
        output: &mut DenseViewMut<'_, T>,
        lhs: DenseView<'_, T>,
        rhs: DenseView<'_, T>,
        jobs: &[DenseGemmBatchJob],
        runs: &[usize],
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
        alpha: DenseScalar,
        beta: DenseScalar,
        wrap_write: W,
        wrap_read: R,
    ) -> Result<(), DenseError>
    where
        T: 'static,
        W: for<'x> Fn(DenseViewMut<'x, T>) -> DenseWrite<'x> + Copy,
        R: for<'x> Fn(DenseView<'x, T>) -> DenseRead<'x> + Copy,
    {
        let mut start = 0usize;
        let mut serial_start = 0usize;
        for &run_len in runs {
            let end = start + run_len;
            let run = &jobs[start..end];
            let can_batch = run_len >= 2
                && run[0].rows != 0
                && run[0].contracted != 0
                && run[0].cols != 0
                && strided_batch_run_len(run, 0) == run_len;
            let layout = can_batch
                .then(|| self.strided_batch_run_layout(run, lhs_op, rhs_op))
                .and_then(Result::ok)
                .filter(|layout| self.strided_batch_run_layout_admitted(output, lhs, rhs, layout));
            if let Some(layout) = layout {
                self.matmul_batch_axpby_ops_serial_typed(
                    output,
                    lhs,
                    rhs,
                    &jobs[serial_start..start],
                    serial_start,
                    lhs_op,
                    rhs_op,
                    alpha,
                    beta,
                    wrap_write,
                    wrap_read,
                )?;
                self.matmul_strided_batch_run_layout_typed(
                    output,
                    lhs,
                    rhs,
                    layout,
                    start,
                    lhs_op,
                    rhs_op,
                    alpha,
                    beta,
                    "matmul_batch_axpby_with_ops_into",
                    wrap_write,
                    wrap_read,
                )?;
                serial_start = end;
            }
            start = end;
        }
        self.matmul_batch_axpby_ops_serial_typed(
            output,
            lhs,
            rhs,
            &jobs[serial_start..],
            serial_start,
            lhs_op,
            rhs_op,
            alpha,
            beta,
            wrap_write,
            wrap_read,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn matmul_batch_axpby_ops_serial_typed<T, W, R>(
        &mut self,
        output: &mut DenseViewMut<'_, T>,
        lhs: DenseView<'_, T>,
        rhs: DenseView<'_, T>,
        jobs: &[DenseGemmBatchJob],
        cache_start: usize,
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
        alpha: DenseScalar,
        beta: DenseScalar,
        wrap_write: W,
        wrap_read: R,
    ) -> Result<(), DenseError>
    where
        T: 'static,
        W: for<'x> Fn(DenseViewMut<'x, T>) -> DenseWrite<'x>,
        R: for<'x> Fn(DenseView<'x, T>) -> DenseRead<'x>,
    {
        if jobs.is_empty() {
            // Do not take the engine lock for an empty loop.
            return Ok(());
        }
        let output_base = output.offset();
        // One engine lock for the whole job loop; each job keeps its own
        // plan-cache slot, order, alpha/beta and conjugation flags.
        let Self {
            ctx,
            matmul_config,
            #[cfg(test)]
            seam_dispatches,
            ..
        } = self;
        let mut engine = ctx.lock();
        Self::run_batch_jobs_in_session(
            &mut engine,
            matmul_config,
            #[cfg(test)]
            seam_dispatches,
            output,
            output_base,
            lhs,
            rhs,
            jobs,
            cache_start,
            lhs_op,
            rhs_op,
            alpha,
            beta,
            wrap_write,
            wrap_read,
        )
    }

    /// The numerical body of [`Self::matmul_batch_axpby_ops_serial_typed`],
    /// under one engine lock held by the caller.
    #[allow(clippy::too_many_arguments)]
    fn run_batch_jobs_in_session<T, W, R>(
        session: &mut CpuEngine,
        matmul_config: &DotGeneralConfig,
        #[cfg(test)] seam_dispatches: &mut usize,
        output: &mut DenseViewMut<'_, T>,
        output_base: usize,
        lhs: DenseView<'_, T>,
        rhs: DenseView<'_, T>,
        jobs: &[DenseGemmBatchJob],
        cache_start: usize,
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
        alpha: DenseScalar,
        beta: DenseScalar,
        wrap_write: W,
        wrap_read: R,
    ) -> Result<(), DenseError>
    where
        T: 'static,
        W: for<'x> Fn(DenseViewMut<'x, T>) -> DenseWrite<'x>,
        R: for<'x> Fn(DenseView<'x, T>) -> DenseRead<'x>,
    {
        for (relative_slot, job) in jobs.iter().enumerate() {
            let lhs_shape = [job.rows, job.contracted];
            let lhs_strides = match lhs_op {
                MatrixOp::Identity => [1, job.rows],
                MatrixOp::Transpose | MatrixOp::Adjoint => [job.contracted, 1],
            };
            let rhs_shape = [job.contracted, job.cols];
            let rhs_strides = match rhs_op {
                MatrixOp::Identity => [1, job.contracted],
                MatrixOp::Transpose | MatrixOp::Adjoint => [job.cols, 1],
            };
            let output_shape = [job.rows, job.cols];
            let output_strides = [1, job.rows];
            let lhs_view = DenseView::new(
                lhs.data(),
                &lhs_shape,
                &lhs_strides,
                batch_offset(lhs.offset(), job.lhs_offset)?,
            )?;
            let rhs_view = DenseView::new(
                rhs.data(),
                &rhs_shape,
                &rhs_strides,
                batch_offset(rhs.offset(), job.rhs_offset)?,
            )?;
            let output_view = DenseViewMut::new(
                output.data_mut(),
                &output_shape,
                &output_strides,
                batch_offset(output_base, job.dst_offset)?,
            )?;
            let lhs = TensorRead::from_view(tenferro_view(wrap_read(lhs_view))?);
            let rhs = TensorRead::from_view(tenferro_view(wrap_read(rhs_view))?);
            let output = TensorWrite::from_view(tenferro_view_mut(wrap_write(output_view))?);
            let accumulation = tenferro_tensor::DotGeneralAccumulation {
                lhs_conj: lhs_op == MatrixOp::Adjoint,
                rhs_conj: rhs_op == MatrixOp::Adjoint,
                alpha: tenferro_scalar(alpha),
                beta: tenferro_scalar(beta),
            };
            #[cfg(test)]
            {
                *seam_dispatches += 1;
            }
            session
                .dot_general_read_into_accum_cached(
                    Some(cache_start + relative_slot),
                    lhs,
                    rhs,
                    matmul_config,
                    accumulation,
                    output,
                )
                .map_err(|err| tenferro_error("matmul_batch_axpby_with_ops_into", err))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn matmul_strided_batch_run_typed<T, W, R>(
        &mut self,
        output: &mut DenseViewMut<'_, T>,
        lhs: DenseView<'_, T>,
        rhs: DenseView<'_, T>,
        run: &[DenseGemmBatchJob],
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
        let layout = self.strided_batch_run_layout(run, lhs_op, rhs_op)?;
        self.matmul_strided_batch_run_layout_typed(
            output, lhs, rhs, layout, cache_slot, lhs_op, rhs_op, alpha, beta, error_op,
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

    fn strided_batch_run_layout_admitted<T>(
        &self,
        output: &DenseViewMut<'_, T>,
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
        let Some(dst_offset) = output.offset().checked_add(layout.dst_job_offset) else {
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
            && DenseView::new(
                output.data(),
                &layout.dst_shape,
                &layout.dst_strides,
                dst_offset,
            )
            .is_ok()
    }

    #[allow(clippy::too_many_arguments)]
    fn matmul_strided_batch_run_layout_typed<T, W, R>(
        &mut self,
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
        self.ctx
            .lock()
            .dot_general_read_into_accum_cached(
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
        #[cfg(all(
            any(feature = "cpu-faer", feature = "cpu-blas-core"),
            not(feature = "provider-inject")
        ))]
        {
            // Both compiled CPU providers factorize the full matrices
            // natively, so neither reaches the `[U1 | I]` QR completion.
            true
        }
        #[cfg(any(
            not(any(feature = "cpu-faer", feature = "cpu-blas-core")),
            feature = "provider-inject"
        ))]
        {
            false
        }
    }

    fn svd_full_owned(
        &mut self,
        input: DenseOwned,
        rows: usize,
        cols: usize,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        #[cfg(feature = "provider-inject")]
        {
            let _ = (input, rows, cols);
            Err(DenseError::Unsupported {
                op: "svd_full_owned",
                message: "executor does not implement owned full-matrices SVD".to_string(),
            })
        }
        #[cfg(not(feature = "provider-inject"))]
        {
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
            #[cfg(all(
                test,
                any(feature = "cpu-faer", feature = "cpu-blas-core"),
                not(feature = "provider-inject")
            ))]
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
            let outputs = LinalgBackend::svd_full(&mut *self.ctx.lock(), &input)
                .map_err(|err| tenferro_error("svd_full_owned", err))?;
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
    }

    fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        #[cfg(feature = "provider-inject")]
        {
            let _ = input;
            Err(linalg_unavailable("svd"))
        }
        #[cfg(not(feature = "provider-inject"))]
        {
            let input = tenferro_view(input)?;
            with_cpu_linalg(&self.ctx, |exec| {
                TensorRead::from_view(input).svd_read(exec)
            })
            .map_err(|err| tenferro_error("svd_read", err))
            .and_then(|(u, s, vt)| {
                vec![u, s, vt]
                    .into_iter()
                    .map(DenseTensor::from_tenferro)
                    .collect()
            })
        }
    }

    fn qr(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        #[cfg(feature = "provider-inject")]
        {
            let _ = input;
            Err(linalg_unavailable("qr"))
        }
        #[cfg(not(feature = "provider-inject"))]
        {
            let input = tenferro_view(input)?;
            with_cpu_linalg(&self.ctx, |exec| TensorRead::from_view(input).qr_read(exec))
                .map_err(|err| tenferro_error("qr_read", err))
                .and_then(|(q, r)| {
                    vec![q, r]
                        .into_iter()
                        .map(DenseTensor::from_tenferro)
                        .collect()
                })
        }
    }

    fn eig(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        #[cfg(feature = "provider-inject")]
        {
            let _ = input;
            Err(linalg_unavailable("eig"))
        }
        #[cfg(not(feature = "provider-inject"))]
        {
            let input = tenferro_view(input)?;
            with_cpu_linalg(&self.ctx, |exec| {
                TensorRead::from_view(input).eig_read(exec)
            })
            .map_err(|err| tenferro_error("eig_read", err))
            .and_then(|(values, vectors)| {
                vec![values, vectors]
                    .into_iter()
                    .map(DenseTensor::from_tenferro)
                    .collect()
            })
        }
    }

    fn eigh(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        #[cfg(feature = "provider-inject")]
        {
            let _ = input;
            Err(linalg_unavailable("eigh"))
        }
        #[cfg(not(feature = "provider-inject"))]
        {
            let input = tenferro_view(input)?;
            with_cpu_linalg(&self.ctx, |exec| {
                TensorRead::from_view(input).eigh_read(exec)
            })
            .map_err(|err| tenferro_error("eigh_read", err))
            .and_then(|(values, vectors)| {
                vec![values, vectors]
                    .into_iter()
                    .map(DenseTensor::from_tenferro)
                    .collect()
            })
        }
    }

    // One engine lock for the whole batch: the per-matrix calls the per-call
    // entries make, in the same order, plus only the pure view lowering;
    // output wrapping stays outside it. Outputs land in one flat vector so the
    // batch adds no per-matrix container beyond the returned one.
    #[cfg(not(feature = "provider-inject"))]
    fn factorize_batch(
        &mut self,
        op: crate::DenseFactorization,
        inputs: &[DenseRead<'_>],
    ) -> Result<Vec<Vec<DenseTensor>>, DenseError> {
        use crate::DenseFactorization;
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let (arity, error_op) = match op {
            DenseFactorization::Svd => (3, "svd_read"),
            DenseFactorization::Qr => (2, "qr_read"),
        };
        let mut flat = Vec::with_capacity(arity * inputs.len());
        with_cpu_linalg(&self.ctx, |exec| {
            let mut run = || -> Result<(), DenseError> {
                for &input in inputs {
                    let input = TensorRead::from_view(tenferro_view(input)?);
                    let backend = |err| tenferro_error(error_op, err);
                    match op {
                        DenseFactorization::Svd => {
                            let (u, s, vt) = input.svd_read(&mut *exec).map_err(backend)?;
                            flat.extend([u, s, vt]);
                        }
                        DenseFactorization::Qr => {
                            let (q, r) = input.qr_read(&mut *exec).map_err(backend)?;
                            flat.extend([q, r]);
                        }
                    }
                }
                Ok(())
            };
            Ok(run())
        })
        .map_err(|err| tenferro_error(error_op, err))??;
        let mut flat = flat.into_iter();
        (0..inputs.len())
            .map(|_| {
                flat.by_ref()
                    .take(arity)
                    .map(DenseTensor::from_tenferro)
                    .collect()
            })
            .collect()
    }

    fn solve_into(
        &mut self,
        a: DenseRead<'_>,
        b: DenseRead<'_>,
        x: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        #[cfg(feature = "provider-inject")]
        {
            let _ = (a, b, x);
            Err(linalg_unavailable("solve_into"))
        }
        #[cfg(not(feature = "provider-inject"))]
        {
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
            with_cpu_linalg(&self.ctx, |exec| {
                TensorRead::from_view(a).solve_read_into(
                    TensorRead::from_view(b),
                    TensorWrite::from_view(x),
                    exec,
                )
            })
            .map_err(tenferro_solve_error)
        }
    }

    // SVD and EIGH values-only calls use Tenferro's documented borrowed
    // no-vector extensions. General EIG below uses native owned `eigvals`.
    fn svd_vals(&mut self, input: DenseRead<'_>) -> Result<DenseTensor, DenseError> {
        #[cfg(feature = "provider-inject")]
        {
            let _ = input;
            Err(linalg_unavailable("svd_vals"))
        }
        #[cfg(not(feature = "provider-inject"))]
        {
            let input = tenferro_view(input)?;
            with_cpu_linalg(&self.ctx, |exec| {
                TensorRead::from_view(input).svdvals_read(exec)
            })
            .map_err(|err| tenferro_error("svd_values", err))
            .and_then(DenseTensor::from_tenferro)
        }
    }

    fn eigh_vals(&mut self, input: DenseRead<'_>) -> Result<DenseTensor, DenseError> {
        #[cfg(feature = "provider-inject")]
        {
            let _ = input;
            Err(linalg_unavailable("eigh_vals"))
        }
        #[cfg(not(feature = "provider-inject"))]
        {
            let input = tenferro_view(input)?;
            with_cpu_linalg(&self.ctx, |exec| {
                TensorRead::from_view(input).eigvalsh_read(exec)
            })
            .map_err(|err| tenferro_error("eigh_values", err))
            .and_then(DenseTensor::from_tenferro)
        }
    }

    fn eig_vals(&mut self, input: DenseRead<'_>) -> Result<DenseTensor, DenseError> {
        #[cfg(feature = "provider-inject")]
        {
            let _ = input;
            Err(linalg_unavailable("eig_vals"))
        }
        #[cfg(not(feature = "provider-inject"))]
        {
            let input = tenferro_view(input)?;
            // One engine lock for the contiguity pre-pass and the values call.
            with_cpu_linalg(&self.ctx, |exec| {
                let owned = exec.to_contiguous_read(TensorRead::from_view(input))?;
                owned.eigvals(exec)
            })
            .map_err(|err| tenferro_error("eig_values", err))
            .and_then(DenseTensor::from_tenferro)
        }
    }

    fn dot_general_into(
        &mut self,
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
            self.ctx
                .lock()
                .dot_general_read_into_accum(lhs, rhs, &dot_config, accumulation, output)
                .map_err(|err| tenferro_error("dot_general_accum", err))
        } else {
            self.ctx
                .lock()
                .dot_general_read_into(lhs, rhs, &dot_config, output)
                .map_err(|err| tenferro_error("dot_general_read_into", err))
        }
    }

    fn matmul_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
    ) -> Result<(), DenseError> {
        // GEMM backend selection is owned by tenferro; this seam only
        // lowers views and reuses the cached rank-2 contraction config.
        let lhs = TensorRead::from_view(tenferro_view(lhs)?);
        let rhs = TensorRead::from_view(tenferro_view(rhs)?);
        let output = TensorWrite::from_view(tenferro_view_mut(output)?);
        self.ctx
            .lock()
            .dot_general_read_into(lhs, rhs, &self.matmul_config, output)
            .map_err(|err| tenferro_error("dot_general_read_into", err))
    }

    fn matmul_axpby_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        alpha: DenseScalar,
        beta: DenseScalar,
    ) -> Result<(), DenseError> {
        // Overwrite case keeps the cached-config fast path.
        if alpha.is_one() && beta.is_zero() {
            return self.matmul_into(output, lhs, rhs);
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
        self.ctx
            .lock()
            .dot_general_read_into_accum(lhs, rhs, &self.matmul_config, accumulation, output)
            .map_err(|err| tenferro_error("dot_general_accum", err))
    }

    fn matmul_batch_axpby_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        jobs: &[DenseGemmBatchJob],
        runs: &[usize],
        alpha: DenseScalar,
        beta: DenseScalar,
    ) -> Result<(), DenseError> {
        let (out_dtype, lhs_dtype, rhs_dtype) = (output.dtype(), lhs.dtype(), rhs.dtype());
        match (output, lhs, rhs) {
            (DenseWrite::F32(mut out), DenseRead::F32(lhs), DenseRead::F32(rhs)) => self
                .matmul_batch_axpby_route_typed(
                    &mut out,
                    lhs,
                    rhs,
                    jobs,
                    runs,
                    alpha,
                    beta,
                    |view: DenseViewMut<'_, f32>| DenseWrite::F32(view),
                    |view: DenseView<'_, f32>| DenseRead::F32(view),
                ),
            (DenseWrite::F64(mut out), DenseRead::F64(lhs), DenseRead::F64(rhs)) => self
                .matmul_batch_axpby_route_typed(
                    &mut out,
                    lhs,
                    rhs,
                    jobs,
                    runs,
                    alpha,
                    beta,
                    |view: DenseViewMut<'_, f64>| DenseWrite::F64(view),
                    |view: DenseView<'_, f64>| DenseRead::F64(view),
                ),
            (DenseWrite::C32(mut out), DenseRead::C32(lhs), DenseRead::C32(rhs)) => self
                .matmul_batch_axpby_route_typed(
                    &mut out,
                    lhs,
                    rhs,
                    jobs,
                    runs,
                    alpha,
                    beta,
                    |view: DenseViewMut<'_, Complex32>| DenseWrite::C32(view),
                    |view: DenseView<'_, Complex32>| DenseRead::C32(view),
                ),
            (DenseWrite::C64(mut out), DenseRead::C64(lhs), DenseRead::C64(rhs)) => self
                .matmul_batch_axpby_route_typed(
                    &mut out,
                    lhs,
                    rhs,
                    jobs,
                    runs,
                    alpha,
                    beta,
                    |view: DenseViewMut<'_, Complex64>| DenseWrite::C64(view),
                    |view: DenseView<'_, Complex64>| DenseRead::C64(view),
                ),
            _ => Err(crate::executor::batch_operand_error(
                "matmul_batch_axpby_into",
                out_dtype,
                lhs_dtype,
                rhs_dtype,
            )),
        }
    }

    #[allow(clippy::redundant_closure)]
    fn matmul_batch_axpby_with_ops_into(
        &mut self,
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
            return self.matmul_batch_axpby_into(output, lhs, rhs, jobs, runs, alpha, beta);
        }
        let (out_dtype, lhs_dtype, rhs_dtype) = (output.dtype(), lhs.dtype(), rhs.dtype());
        match (output, lhs, rhs) {
            (DenseWrite::F32(mut out), DenseRead::F32(lhs), DenseRead::F32(rhs)) => self
                .matmul_batch_axpby_ops_typed(
                    &mut out,
                    lhs,
                    rhs,
                    jobs,
                    runs,
                    lhs_op,
                    rhs_op,
                    alpha,
                    beta,
                    // Constructor functions capture one concrete lifetime; closures
                    // remain generic over each temporary view borrowed below.
                    |view| DenseWrite::F32(view),
                    |view| DenseRead::F32(view),
                ),
            (DenseWrite::F64(mut out), DenseRead::F64(lhs), DenseRead::F64(rhs)) => self
                .matmul_batch_axpby_ops_typed(
                    &mut out,
                    lhs,
                    rhs,
                    jobs,
                    runs,
                    lhs_op,
                    rhs_op,
                    alpha,
                    beta,
                    |view| DenseWrite::F64(view),
                    |view| DenseRead::F64(view),
                ),
            (DenseWrite::C32(mut out), DenseRead::C32(lhs), DenseRead::C32(rhs)) => self
                .matmul_batch_axpby_ops_typed(
                    &mut out,
                    lhs,
                    rhs,
                    jobs,
                    runs,
                    lhs_op,
                    rhs_op,
                    alpha,
                    beta,
                    |view| DenseWrite::C32(view),
                    |view| DenseRead::C32(view),
                ),
            (DenseWrite::C64(mut out), DenseRead::C64(lhs), DenseRead::C64(rhs)) => self
                .matmul_batch_axpby_ops_typed(
                    &mut out,
                    lhs,
                    rhs,
                    jobs,
                    runs,
                    lhs_op,
                    rhs_op,
                    alpha,
                    beta,
                    |view| DenseWrite::C64(view),
                    |view| DenseRead::C64(view),
                ),
            _ => Err(crate::executor::batch_operand_error(
                "matmul_batch_axpby_with_ops_into",
                out_dtype,
                lhs_dtype,
                rhs_dtype,
            )),
        }
    }
}

/// The four GEMM dtypes of the grouped route, with their [`DenseScalar`]
/// projection (a mismatched coefficient dtype is refused before any write).
// A BLAS build runs the grouped jobs through the BLAS dot and reads neither.
#[cfg_attr(feature = "cpu-blas-core", allow(dead_code))]
trait GroupedScalar: cpueinsum::Scalar {
    /// Real flops per multiply-add (2 real, 8 complex), as `GroupedPlan`
    /// counts them.
    const FLOPS_PER_MULADD: f64;
    fn from_dense(value: DenseScalar) -> Option<Self>;
}

macro_rules! grouped_scalar {
    ($($ty:ty => $variant:ident, $flops:expr);+) => {$(
        impl GroupedScalar for $ty {
            const FLOPS_PER_MULADD: f64 = $flops;
            fn from_dense(value: DenseScalar) -> Option<Self> {
                match value {
                    DenseScalar::$variant(value) => Some(value),
                    _ => None,
                }
            }
        }
    )+};
}
grouped_scalar!(f32 => F32, 2.0; f64 => F64, 2.0; Complex32 => C32, 8.0; Complex64 => C64, 8.0);

fn grouped_error(message: String) -> DenseError {
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
/// in its sorted output order instead). Output overlap is left to
/// `GroupedPlan::new`, which rejects it before any write.
fn grouped_validate<T>(
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
    Ok(lens)
}

/// The output-overlap check `GroupedPlan::new` makes, for the BLAS route that
/// builds no plan: O(J log J) over the nonempty output ranges, before any
/// write.
#[cfg(feature = "cpu-blas-core")]
fn grouped_reject_overlap(jobs: &[DenseGemmBatchJob]) -> Result<(), DenseError> {
    let mut ranges = jobs
        .iter()
        .enumerate()
        .filter(|(_, job)| job.rows != 0 && job.cols != 0)
        .map(|(index, job)| (job.dst_offset, job.dst_offset + job.rows * job.cols, index))
        .collect::<Vec<_>>();
    ranges.sort_unstable();
    for pair in ranges.windows(2) {
        if pair[0].1 > pair[1].0 {
            return Err(grouped_error(format!(
                "grouped output jobs {} and {} overlap",
                pair[0].2, pair[1].2
            )));
        }
    }
    Ok(())
}

/// symtn's block-contraction kernel (#2337): one `cpueinsum::GroupedPlan` over
/// the batch, run on the engine's execution token. Every bound and the output
/// overlap are checked before the first write; jobs keep their caller index,
/// and each output range is written by exactly one job, so lane order cannot
/// change a result.
// ponytail: plans per call, like the retired tenferro grouped entry did
// (`cache_slot: None`): O(J) bytes and, in cpueinsum 0.7.0, O(log J)
// allocations per call. The executor has no structure key for a batch, so a
// plan memo would be a new implicit cache; add one only with that key.
#[cfg(not(feature = "cpu-blas-core"))]
#[allow(clippy::too_many_arguments)]
fn grouped_gemm_into<T: GroupedScalar>(
    engine: &mut CpuEngine,
    grouped_jobs: &mut Vec<cpueinsum::GroupedGemmJob>,
    output: &mut DenseViewMut<'_, T>,
    lhs: DenseView<'_, T>,
    rhs: DenseView<'_, T>,
    jobs: &[DenseGemmBatchJob],
    alpha: DenseScalar,
    beta: DenseScalar,
) -> Result<(), DenseError> {
    let (Some(alpha), Some(beta)) = (T::from_dense(alpha), T::from_dense(beta)) else {
        return Err(grouped_error(
            "alpha/beta dtype differs from the operands".to_string(),
        ));
    };
    let [lhs_len, rhs_len, out_len] = grouped_validate(output, lhs, rhs, jobs)?;
    grouped_jobs.clear();
    grouped_jobs.extend(jobs.iter().map(|job| {
        cpueinsum::GroupedGemmJob::new(
            job.lhs_offset,
            job.rhs_offset,
            job.dst_offset,
            job.rows,
            job.contracted,
            job.cols,
        )
    }));
    let plan = cpueinsum::GroupedPlan::<T>::new(grouped_jobs, &Default::default())
        .map_err(|err| grouped_error(err.to_string()))?;
    let lhs = &lhs.data()[lhs.offset()..lhs.offset() + lhs_len];
    let rhs = &rhs.data()[rhs.offset()..rhs.offset() + rhs_len];
    let offset = output.offset();
    let out = &mut output.data_mut()[offset..offset + out_len];
    let mut exec = engine
        .exec()
        .map_err(|err| tenferro_error("grouped_gemm", err))?;
    // GroupedPlan's `Auto` only fans the jobs out over lanes, each job serial,
    // so a batch whose largest job pays for a team on its own would run that
    // job on one thread. tcontract-exec's faer rule (`batch.rs:faer_auto`)
    // gives such an item the whole team: run the jobs in turn, each on the
    // full token.
    // ponytail: one rule for the whole batch, so the small jobs of a mixed
    // batch also get the team; split the batch by job when that shows up.
    let max_muladds = jobs
        .iter()
        .map(|job| job.rows as f64 * job.contracted as f64 * job.cols as f64)
        .fold(0.0, f64::max);
    let max_job_ns = max_muladds * T::FLOPS_PER_MULADD * cpueinsum::tcontract::ns_per_flop();
    if exec.width_for(max_job_ns, 1) > 1 {
        exec = exec.with_batch_policy(cpueinsum::BatchPolicy::new(cpueinsum::BatchStrategy::Inner));
    }
    plan.execute_into_accum(&mut exec, alpha, lhs, rhs, beta, out)
        .map(|_route| ())
        .map_err(|err| grouped_error(err.to_string()))
}

fn tenferro_scalar(value: DenseScalar) -> tenferro_tensor::ContractionScalar {
    match value {
        DenseScalar::F32(value) => tenferro_tensor::ContractionScalar::F32(value),
        DenseScalar::F64(value) => tenferro_tensor::ContractionScalar::F64(value),
        DenseScalar::C32(value) => tenferro_tensor::ContractionScalar::C32(value),
        DenseScalar::C64(value) => tenferro_tensor::ContractionScalar::C64(value),
    }
}

fn tenferro_view(input: DenseRead<'_>) -> Result<TensorView<'_>, DenseError> {
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

#[cfg(not(feature = "provider-inject"))]
fn tenferro_solve_error(err: tenferro_tensor::Error) -> DenseError {
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

#[cfg(not(feature = "provider-inject"))]
fn with_cpu_linalg<R>(
    ctx: &SharedCpuContext,
    f: impl FnOnce(&mut dyn Engine) -> tenferro_tensor::Result<R>,
) -> tenferro_tensor::Result<R> {
    f(&mut *ctx.lock())
}

#[cfg(feature = "provider-inject")]
fn linalg_unavailable(op: &'static str) -> DenseError {
    DenseError::Unsupported {
        op,
        message: "provider-inject does not route factorizations or solves; they are unsupported in this configuration".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_context_executor_runs_on_that_engine() {
        let context = SharedCpuContext::with_threads(1).expect("serial context");
        let executor = DefaultDenseExecutor::with_shared_context(&context);
        assert!(executor.shares_cpu_context(&context));
    }
}
