use num_complex::{Complex32, Complex64};

use crate::executor::{batch_offset, strided_batch_run_len};
use crate::layout::strides_to_isize;
use crate::{
    DenseBackend, DenseDotConfig, DenseError, DenseExecutor, DenseGemmBatchJob, DenseOwned,
    DenseRead, DenseScalar, DenseTensor, DenseView, DenseViewMut, DenseWrite, MatrixOp,
};

use std::sync::Arc;

use std::sync::atomic::{AtomicU64, Ordering};

static SESSIONS_OPENED: AtomicU64 = AtomicU64::new(0);
static ADMISSIONS: AtomicU64 = AtomicU64::new(0);

thread_local! {
    // Set on the thread that runs a `with_linalg_scope` body while the
    // Tenferro execution scope is entered, so sessions opened there reuse its
    // admission and are not counted as admissions of their own.
    static IN_LINALG_SCOPE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// A snapshot of the process-wide symtn CPU-session observation counters.
///
/// Observability only: nothing here reads the value back, so no execution
/// decision, dispatch, or capability depends on it. It exists so a test or a
/// benchmark can attribute Tenferro session entries — a process-global
/// execution permit plus, on faer, a Rayon pool handoff — to a measured phase.
///
/// Scope: `sessions_opened` counts the sessions [`DefaultDenseExecutor`] opens
/// *itself* (the linear-algebra scope, the op-bearing serial GEMM batch, the
/// owned full SVD). Sessions that Tenferro opens internally for a plain
/// backend-level dot call are invisible at this seam and are **not** counted,
/// so this is a lower bound on the sessions a phase actually enters.
///
/// `admissions` counts the execution-permit acquisitions of those sessions
/// and of the backend-level calls of [`DenseExecutor::dot_general_into`] and
/// of the strided and grouped batch GEMM routes: one per such entry made
/// outside an execution scope, plus one per entered scope, whose entries share
/// its permit and pool handoff. Scopes are entered by
/// [`DenseExecutor::with_linalg_scope`] and by an op-bearing batch GEMM whose
/// covering run partition has more than one run. Other backend-level dot entries are not
/// counted. Under a multi-threaded CPU layout each admission is one Rayon pool
/// hop.
///
/// The counters are `Relaxed` and process-wide: a snapshot taken while another
/// thread executes is a sample, not a global instant.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CpuSessionStats {
    pub sessions_opened: u64,
    pub admissions: u64,
}

/// Reads the CPU-session observation counters. See [`CpuSessionStats`].
pub fn cpu_session_stats() -> CpuSessionStats {
    CpuSessionStats {
        sessions_opened: SESSIONS_OPENED.load(Ordering::Relaxed),
        admissions: ADMISSIONS.load(Ordering::Relaxed),
    }
}

/// Zeroes the CPU-session observation counters. See [`CpuSessionStats`].
pub fn reset_cpu_session_stats() {
    SESSIONS_OPENED.store(0, Ordering::Relaxed);
    ADMISSIONS.store(0, Ordering::Relaxed);
}

fn note_session_opened() {
    SESSIONS_OPENED.fetch_add(1, Ordering::Relaxed);
    note_backend_admission();
}

fn note_backend_admission() {
    if !IN_LINALG_SCOPE.get() {
        ADMISSIONS.fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(not(feature = "provider-inject"))]
use tenferro_cpu::with_cpu_exec_session;
use tenferro_cpu::{CpuBackend, CpuBackendKind, CpuContext};
#[cfg(not(feature = "provider-inject"))]
use tenferro_linalg::{LinalgBackend, TensorLinalgExt, TensorReadLinalgExt};
use tenferro_tensor::backend::{
    BackendSession, BackendSessionHost, GroupedGemmConfig, GroupedGemmJob,
};
use tenferro_tensor::{
    BackendCachedDot, BackendRuntimeCache, DotGeneralConfig, TensorDot, TensorRead, TensorView,
    TensorViewMut, TensorWrite, TypedTensorView, TypedTensorViewMut,
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

/// One CPU execution context — a parallelism hint plus resources for
/// multi-threaded CPU work — shared by runtime-built executors using the
/// compiled default provider kind. An explicitly requested nondefault kind
/// uses a private provider context, while a custom executor owns its own
/// configuration. The wrapper keeps callers independent of Tenferro types.
///
/// Each executor keeps its own `CpuBackend` and `BufferPool`; scratch reuse,
/// synchronization, and thread behavior remain provider concerns.
#[derive(Clone, Debug)]
pub struct SharedCpuContext {
    ctx: Arc<CpuContext>,
}

impl SharedCpuContext {
    /// Environment-driven context (`RAYON_NUM_THREADS`, else the machine's
    /// available parallelism) — matches what `CpuBackend::new` reads per call.
    pub fn from_env() -> Self {
        Self {
            ctx: Arc::new(CpuContext::from_env()),
        }
    }

    /// Fixed thread count; `1` builds no pool at all (fully serial), same as
    /// the per-executor constructors it replaces.
    pub fn with_threads(threads: usize) -> Result<Self, DenseError> {
        CpuContext::with_threads(threads)
            .map(|ctx| Self { ctx: Arc::new(ctx) })
            .map_err(|err| tenferro_error("SharedCpuContext::with_threads", err))
    }

    pub fn num_threads(&self) -> usize {
        self.ctx.num_threads()
    }

    /// One thread per CPU the process may run on (affinity-aware), and no
    /// environment read: the default size of a runtime's CPU pool.
    pub fn with_available_parallelism() -> Result<Self, DenseError> {
        Self::with_threads(tenferro_cpu::available_parallelism())
    }

    /// Runs `op` on this context's Rayon pool, the pool its dense kernels use.
    /// A one-thread context has no pool and runs `op` on the calling thread.
    /// Called from a worker of the same pool, `op` runs inline.
    ///
    /// Only for backend-free parallel regions (symtn's `host_pool`). Why not a
    /// whole operation: the pool's workers are registered with this context's
    /// Tenferro execution scope, so a dense backend call started on a worker
    /// while another operation's dense call on the same context is active
    /// panics with Tenferro's re-entry error (`BACKEND_REENTRY_PANIC`), even
    /// without work stealing.
    #[doc(hidden)]
    pub fn install<R: Send>(&self, op: impl FnOnce() -> R + Send) -> R {
        self.ctx.install(op)
    }

    /// Identity check for regression tests: do two handles share one context
    /// (hence one rayon pool)?
    #[doc(hidden)]
    pub fn ptr_eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.ctx, &other.ctx)
    }

    /// Address identity of the shared context, for observation probes only.
    #[doc(hidden)]
    pub fn identity(&self) -> usize {
        Arc::as_ptr(&self.ctx) as usize
    }
}

#[derive(Debug)]
pub struct DefaultDenseExecutor {
    backend: CpuBackend,
    matmul_config: DotGeneralConfig,
    strided_batch_matmul_config: DotGeneralConfig,
    grouped_cache: <CpuBackend as BackendRuntimeCache>::RuntimeCache,
    grouped_jobs: Vec<GroupedGemmJob>,
    // The runtime-shared context this executor was built on, if any (see
    // `with_shared_context`). Kept here rather than read back off the backend:
    // tenferro's `CpuBackend::linalg_context` accessor is `cpu-faer`-gated, so
    // a hook based on it does not compile on BLAS-provider builds.
    shared_ctx: Option<SharedCpuContext>,
    // Test-only count of low-level batch seam submissions: one per grouped or
    // strided call and one per job in the op-bearing serial fallback.
    #[cfg(test)]
    seam_dispatches: usize,
}

impl DefaultDenseExecutor {
    pub fn new() -> Self {
        Self::from_backend(CpuBackend::new())
    }

    pub fn with_threads(threads: usize) -> Result<Self, DenseError> {
        // `.into()`: since tenferro #1376 the BLAS-provider builds of these
        // constructors return `CpuBackendError`, while faer builds return the
        // crate `Error`; `Into` is identity on the latter, so one spelling
        // compiles under every provider feature.
        CpuBackend::with_threads(threads)
            .map(Self::from_backend)
            .map_err(|err| tenferro_error("CpuBackend::with_threads", err.into()))
    }

    /// Builds an executor on a specific CPU linear-algebra provider
    /// ([`CpuBackendKind::Faer`] or [`CpuBackendKind::Blas`]). Fails if the
    /// requested provider was not compiled in (e.g. `Blas` without a
    /// `cpu-blas`/`blas-*` feature) — the check happens here, not at first use.
    pub fn with_kind(kind: CpuBackendKind) -> Result<Self, DenseError> {
        CpuBackend::with_kind(kind)
            .map(Self::from_backend)
            .map_err(|err| tenferro_error("CpuBackend::with_kind", err.into()))
    }

    /// [`Self::with_kind`] plus an explicit thread count for the provider.
    pub fn with_threads_and_kind(threads: usize, kind: CpuBackendKind) -> Result<Self, DenseError> {
        CpuBackend::with_threads_and_kind(threads, kind)
            .map(Self::from_backend)
            .map_err(|err| tenferro_error("CpuBackend::with_threads_and_kind", err.into()))
    }

    /// Builds an executor with an optional provider kind. `None` uses Tenferro's
    /// resolved compiled default — BLAS when its CPU build enables `cpu-blas`,
    /// otherwise faer — on the runtime's shared [`SharedCpuContext`]; an
    /// explicit nondefault kind uses a private provider context. Each executor
    /// retains its own backend and buffer pool; this does not promise lock-free
    /// scratch access or a provider thread policy.
    pub fn with_shared_context(
        ctx: &SharedCpuContext,
        kind: Option<CpuBackendKind>,
    ) -> Result<Self, DenseError> {
        match kind {
            // ponytail: tenferro has no pub context+kind constructor, so the
            // one non-default-kind combination (explicit Faer while a BLAS
            // provider is compiled in) keeps a private context/pool exactly as
            // before this seam existed (`shared_ctx` stays `None`). Lift when
            // tenferro exposes `from_context` with a kind.
            Some(kind) if kind != CpuBackendKind::default_compiled() => {
                Self::with_threads_and_kind(ctx.num_threads(), kind)
            }
            // `from_context` fixes the kind at `default_compiled`, so it also
            // covers an explicit request FOR that default.
            _ => {
                let mut executor =
                    Self::from_backend(CpuBackend::from_context(Arc::clone(&ctx.ctx)));
                executor.shared_ctx = Some(ctx.clone());
                Ok(executor)
            }
        }
    }

    /// Regression-test hook: does this executor run on `ctx`'s rayon pool?
    #[doc(hidden)]
    pub fn shares_cpu_context(&self, ctx: &SharedCpuContext) -> bool {
        self.shared_ctx.as_ref().is_some_and(|own| own.ptr_eq(ctx))
    }

    fn from_backend(backend: CpuBackend) -> Self {
        Self {
            backend,
            matmul_config: DotGeneralConfig {
                lhs_contracting_dims: vec![1],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: Vec::new(),
                rhs_batch_dims: Vec::new(),
            },
            strided_batch_matmul_config: DotGeneralConfig {
                lhs_contracting_dims: vec![1],
                rhs_contracting_dims: vec![0],
                lhs_batch_dims: vec![2],
                rhs_batch_dims: vec![2],
            },
            grouped_cache: <CpuBackend as BackendRuntimeCache>::RuntimeCache::default(),
            grouped_jobs: Vec::new(),
            shared_ctx: None,
            #[cfg(test)]
            seam_dispatches: 0,
        }
    }

    /// Runs `body` once inside one Tenferro execution scope on this
    /// executor's backend, so every session it opens shares one permit and
    /// pool install. Inside an active symtn scope it runs inline; where
    /// Tenferro cannot enter a scope it runs unscoped, paying one admission
    /// per session as without a scope.
    #[cfg(not(feature = "provider-inject"))]
    fn in_execution_scope(
        &mut self,
        mut body: impl FnMut(&mut Self) -> Result<(), DenseError> + Send,
    ) -> Result<(), DenseError> {
        if IN_LINALG_SCOPE.get() {
            return body(self);
        }
        // `with_execution_scope` borrows its backend while the body needs
        // `&mut self`; a clone keeps the runtime identity, engine and domain,
        // which is what the scope matches sessions against.
        let witness = self.backend.clone();
        let mut entered = false;
        let scoped = witness.with_execution_scope(|| {
            entered = true;
            ADMISSIONS.fetch_add(1, Ordering::Relaxed);
            IN_LINALG_SCOPE.set(true);
            struct Leave;
            impl Drop for Leave {
                fn drop(&mut self) {
                    IN_LINALG_SCOPE.set(false);
                }
            }
            let _leave = Leave;
            body(&mut *self)
        });
        match scoped {
            Ok(result) => result,
            // Not entered because the domain is caller-owned (`Unsupported`)
            // or a Tenferro execution is already active on this thread
            // (`RuntimeState`): per-session admission is still correct, only
            // slower, so run the body unscoped.
            Err(err)
                if !entered
                    && matches!(
                        err.kind(),
                        tenferro_tensor::ErrorKind::Unsupported
                            | tenferro_tensor::ErrorKind::RuntimeState
                    ) =>
            {
                body(self)
            }
            Err(err) => Err(tenferro_error("with_execution_scope", err)),
        }
    }

    // provider-inject has no linear-algebra scope (see `with_linalg_scope`);
    // each session admits itself.
    #[cfg(feature = "provider-inject")]
    fn in_execution_scope(
        &mut self,
        mut body: impl FnMut(&mut Self) -> Result<(), DenseError> + Send,
    ) -> Result<(), DenseError> {
        body(self)
    }

    #[cfg(test)]
    pub(crate) fn reset_seam_dispatches(&mut self) {
        self.seam_dispatches = 0;
    }

    #[cfg(test)]
    pub(crate) fn seam_dispatches(&self) -> usize {
        self.seam_dispatches
    }

    /// Jobs staged for the last grouped submission; zero on a fresh executor
    /// whose identity batches all took the strided route.
    #[cfg(test)]
    pub(crate) fn staged_grouped_jobs(&self) -> usize {
        self.grouped_jobs.len()
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
        T: 'static,
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
        self.grouped_jobs.clear();
        self.grouped_jobs.extend(jobs.iter().map(|job| {
            GroupedGemmJob::new(
                job.dst_offset,
                job.lhs_offset,
                job.rhs_offset,
                job.rows,
                job.contracted,
                job.cols,
            )
        }));
        self.matmul_grouped_bundle_typed(output, lhs, rhs, alpha, beta, wrap_write, wrap_read)
    }

    /// Single grouped-gemm call over the jobs staged in `self.grouped_jobs`.
    /// One seam dispatch regardless of how many runs fed it.
    #[allow(clippy::too_many_arguments)]
    fn matmul_grouped_bundle_typed<T, W, R>(
        &mut self,
        output: &mut DenseViewMut<'_, T>,
        lhs: DenseView<'_, T>,
        rhs: DenseView<'_, T>,
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
        #[cfg(test)]
        {
            self.seam_dispatches += 1;
        }
        // shape/strides carry the view's own 'a lifetime, so capture them before
        // the mutable data borrow when rebuilding the full-buffer write view.
        let shape = output.shape();
        let strides = output.strides();
        let offset = output.offset();
        let out_view = DenseViewMut::new(output.data_mut(), shape, strides, offset)?;
        let lhs = TensorRead::from_view(tenferro_view(wrap_read(lhs))?);
        let rhs = TensorRead::from_view(tenferro_view(wrap_read(rhs))?);
        let output = TensorWrite::from_view(tenferro_view_mut(wrap_write(out_view))?);
        let accumulation = tenferro_tensor::DotGeneralAccumulation {
            lhs_conj: false,
            rhs_conj: false,
            alpha: tenferro_scalar(alpha),
            beta: tenferro_scalar(beta),
        };
        let config = GroupedGemmConfig::new(&self.grouped_jobs, accumulation);
        note_backend_admission();
        BackendCachedDot::grouped_gemm_cached(
            &mut self.backend,
            &mut self.grouped_cache,
            None,
            lhs,
            rhs,
            &config,
            output,
        )
        .map_err(|err| tenferro_error("grouped_gemm", err))
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
        // `Send` and the shared-reference `Sync`: the serial route below runs
        // its job loop inside a Tenferro session, whose closure Tenferro may
        // install on its own Rayon worker.
        T: 'static + Send + Sync,
        W: for<'x> Fn(DenseViewMut<'x, T>) -> DenseWrite<'x> + Copy + Send,
        R: for<'x> Fn(DenseView<'x, T>) -> DenseRead<'x> + Copy + Send,
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

        // Every run of the partition is one dispatch into its own Tenferro
        // session, and each session takes the process-wide execution permit
        // (on faer, also a Rayon pool install). One execution scope makes the
        // whole phase pay that once. The permit is then held across the
        // phase's between-run view setup too, so a short Tenferro call on
        // another thread can wait for the whole phase rather than one run.
        // A single run is one dispatch already, so a scope would save nothing.
        if runs.len() == 1 {
            return self.matmul_batch_axpby_ops_partition_typed(
                output, lhs, rhs, jobs, runs, lhs_op, rhs_op, alpha, beta, wrap_write, wrap_read,
            );
        }
        self.in_execution_scope(move |this| {
            this.matmul_batch_axpby_ops_partition_typed(
                output, lhs, rhs, jobs, runs, lhs_op, rhs_op, alpha, beta, wrap_write, wrap_read,
            )
        })
    }

    /// Issues a covering partition in job order: each batchable run as one
    /// strided call, and each maximal stretch of adjacent non-batchable runs
    /// as one serial call (one session) instead of one call per run. Plan-cache
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
        T: 'static + Send + Sync,
        W: for<'x> Fn(DenseViewMut<'x, T>) -> DenseWrite<'x> + Copy + Send,
        R: for<'x> Fn(DenseView<'x, T>) -> DenseRead<'x> + Copy + Send,
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
        T: 'static + Send + Sync,
        W: for<'x> Fn(DenseViewMut<'x, T>) -> DenseWrite<'x> + Send,
        R: for<'x> Fn(DenseView<'x, T>) -> DenseRead<'x> + Send,
    {
        if jobs.is_empty() {
            // A session is a process-wide critical section; do not take it for
            // an empty loop. The pre-change path opened zero sessions here.
            return Ok(());
        }
        let output_base = output.offset();
        // One session for the whole job loop instead of one per job: Tenferro
        // charges execution-domain admission, the engine mutex, and (on faer) a
        // synchronous Rayon worker handoff per session, and that cost is what
        // dominates a batch of small GEMMs. The jobs, their kernels, their
        // submission order, their alpha/beta and their conjugation flags are
        // unchanged, and each still uses the executor's own plan-cache slot, so
        // the results are bitwise identical to the per-job path.
        //
        // A CPU session is a process-wide critical section: it holds the global
        // execution permit (faer: CpuSet compatibility; BLAS: provider
        // exclusive, so every other thread's Tenferro call blocks) and, on
        // faer, runs this closure on Tenferro's Rayon pool for the whole scope.
        // The scope therefore covers the numerical loop only. Nothing inside it
        // compiles a plan, runs a provider callback, reads an
        // allocation-observing thread-local, forks with `rayon::join`, or waits
        // on another thread, and nothing inside it calls a backend-level entry
        // — that re-entry panics in release.
        let Self {
            backend,
            matmul_config,
            grouped_cache,
            #[cfg(test)]
            seam_dispatches,
            ..
        } = self;
        note_session_opened();
        #[cfg(test)]
        let (key, caller) = (session_key(backend), std::thread::current().id());
        backend.with_backend_session_cached(grouped_cache, |session| {
            #[cfg(test)]
            record_session_thread(key, caller);
            Self::run_batch_jobs_in_session(
                session,
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
        })
    }

    /// The numerical body of [`Self::matmul_batch_axpby_ops_serial_typed`],
    /// running inside one already-entered Tenferro session. Every dot goes
    /// through the session's own entry point: a backend-level call here would
    /// re-enter the execution arbiter and panic.
    #[allow(clippy::too_many_arguments)]
    fn run_batch_jobs_in_session<T, W, R>(
        session: &mut dyn BackendSession,
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
        note_backend_admission();
        BackendCachedDot::dot_general_read_into_accum_cached(
            &mut self.backend,
            &mut self.grouped_cache,
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
            matches!(
                self.backend.kind(),
                CpuBackendKind::Faer | CpuBackendKind::Blas
            )
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
            note_session_opened();
            let outputs = self
                .backend
                .with_backend_session(|session| {
                    with_cpu_exec_session(session, |exec| exec.svd_full(&input))
                })
                .ok_or_else(|| DenseError::Unsupported {
                    op: "svd_full_owned",
                    message: "CPU backend session unavailable".to_string(),
                })?
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
            with_cpu_linalg(&mut self.backend, |exec| {
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
            with_cpu_linalg(&mut self.backend, |exec| {
                TensorRead::from_view(input).qr_read(exec)
            })
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
            with_cpu_linalg(&mut self.backend, |exec| {
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
            with_cpu_linalg(&mut self.backend, |exec| {
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

    // One session for the whole batch: the per-session admission (global
    // execution permit, engine mutex, and on faer a Rayon pool handoff) is
    // otherwise paid once per coupled sector. The scope holds the per-matrix
    // calls the per-call entries make, in the same order, plus only the pure
    // view lowering; output wrapping stays outside it. Outputs land in one flat
    // vector so the batch adds no per-matrix container beyond the returned one.
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
        with_cpu_linalg(&mut self.backend, |exec| {
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

    // One Tenferro execution scope for the whole body: its sessions skip the
    // per-session permit and pool install (tenferro-cpu 0.6.0
    // `CpuOperationEntry::enter`, `is_entered`), so a streaming per-block loop
    // pays one admission per call while keeping one block in flight. The
    // permit is held across the body's own between-block work too.
    #[cfg(not(feature = "provider-inject"))]
    fn with_linalg_scope(
        &mut self,
        body: &mut crate::DenseLinalgScopeBody<'_>,
    ) -> Result<(), DenseError> {
        self.in_execution_scope(|this| body(this))
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
            with_cpu_linalg(&mut self.backend, |exec| {
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
            with_cpu_linalg(&mut self.backend, |exec| {
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
            with_cpu_linalg(&mut self.backend, |exec| {
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
            // One session for the contiguity pre-pass and the values call: a
            // Tenferro CPU session is a process-wide critical section (global
            // execution permit, engine mutex, and on faer a Rayon pool
            // handoff), so this scope holds only the numerical stage. Nothing
            // inside it compiles a plan, runs a provider callback, reads an
            // allocation-observing thread-local, forks with `rayon::join`, or
            // waits on another thread, and nothing inside it calls a
            // backend-level entry (that re-entry panics in release).
            with_cpu_linalg(&mut self.backend, |exec| {
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
        note_backend_admission();
        // Non-conjugating path stays byte-identical to the plain read_into
        // (which itself just wraps an overwrite accumulation). Conjugation is
        // folded into the kernel via the accumulation's conj flags — no
        // conjugated operand copy — instead of falling back to a scalar loop.
        if config.lhs_conj() || config.rhs_conj() {
            let mut accumulation = tenferro_tensor::DotGeneralAccumulation::overwrite(lhs.dtype())
                .map_err(|err| tenferro_error("dot_general_accum", err))?;
            accumulation.lhs_conj = config.lhs_conj();
            accumulation.rhs_conj = config.rhs_conj();
            self.backend
                .dot_general_read_into_accum(lhs, rhs, &dot_config, accumulation, output)
                .map_err(|err| tenferro_error("dot_general_accum", err))
        } else {
            self.backend
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
        self.backend
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
        self.backend
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
        lhs_contracting_dims: config.lhs_contracting_dims().to_vec(),
        rhs_contracting_dims: config.rhs_contracting_dims().to_vec(),
        lhs_batch_dims: config.lhs_batch_dims().to_vec(),
        rhs_batch_dims: config.rhs_batch_dims().to_vec(),
    }
}

pub(crate) fn tenferro_error(op: &'static str, err: tenferro_tensor::Error) -> DenseError {
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
fn with_cpu_linalg<R: Send>(
    backend: &mut CpuBackend,
    f: impl FnOnce(&mut dyn BackendSession) -> tenferro_tensor::Result<R> + Send,
) -> tenferro_tensor::Result<R> {
    note_session_opened();
    #[cfg(test)]
    {
        let (key, caller) = (session_key(backend), std::thread::current().id());
        backend.with_backend_session(|s| {
            record_session_thread(key, caller);
            f(s)
        })
    }
    #[cfg(not(test))]
    backend.with_backend_session(f)
}

// Test-only record of (backend, caller thread, thread the Tenferro session
// closure ran on) for each `with_cpu_linalg` and serial-batch session. It is
// process-wide because the closure may run on a pool worker, so tests read it
// filtered by their own executor's backend.
#[cfg(test)]
type SessionThreads = Vec<(usize, std::thread::ThreadId, std::thread::ThreadId)>;

#[cfg(test)]
static SESSION_THREADS: std::sync::Mutex<SessionThreads> = std::sync::Mutex::new(Vec::new());

#[cfg(test)]
fn session_key(backend: &CpuBackend) -> usize {
    std::ptr::from_ref(backend) as usize
}

#[cfg(test)]
fn record_session_thread(key: usize, caller: std::thread::ThreadId) {
    SESSION_THREADS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .push((key, caller, std::thread::current().id()));
}

#[cfg(feature = "provider-inject")]
fn linalg_unavailable(op: &'static str) -> DenseError {
    DenseError::Unsupported {
        op,
        message: "provider-inject does not route factorizations or solves; they are unsupported in this configuration".to_string(),
    }
}

#[cfg(all(
    test,
    any(feature = "cpu-faer", feature = "cpu-blas-core"),
    not(feature = "provider-inject")
))]
mod linalg_scope_tests {
    use super::*;

    const ROWS: usize = 5;
    const COLS: usize = 3;

    fn matrix(seed: usize) -> Vec<f64> {
        (0..ROWS * COLS)
            .map(|i| ((i * 7 + seed * 3) % 11) as f64 * 0.25 - 1.0)
            .collect()
    }

    /// Every output bit of an SVD then a QR of each of three matrices.
    fn factor_bits(dense: &mut dyn DenseExecutor) -> Vec<u64> {
        let mut bits = Vec::new();
        for seed in 0..3 {
            let data = matrix(seed);
            let read =
                || DenseRead::F64(DenseView::new(&data, &[ROWS, COLS], &[1, ROWS], 0).unwrap());
            for output in dense
                .svd(read())
                .unwrap()
                .into_iter()
                .chain(dense.qr(read()).unwrap())
            {
                bits.extend(output.as_f64_slice().unwrap().iter().map(|x| x.to_bits()));
            }
        }
        bits
    }

    /// Runs [`factor_bits`] in a scope; also reports whether the Tenferro
    /// execution scope was entered for the body.
    fn scoped_bits(executor: &mut DefaultDenseExecutor) -> (bool, Vec<u64>) {
        let mut entered = None;
        let mut bits = Vec::new();
        executor
            .with_linalg_scope(&mut |dense| {
                entered = Some(IN_LINALG_SCOPE.get());
                bits = factor_bits(dense);
                Ok(())
            })
            .unwrap();
        (entered.expect("scope body runs"), bits)
    }

    fn executors() -> [DefaultDenseExecutor; 2] {
        [
            DefaultDenseExecutor::new(),
            DefaultDenseExecutor::with_threads(1).unwrap(),
        ]
    }

    #[test]
    fn managed_scope_is_entered_and_bitwise_matches_per_call_sessions() {
        for mut executor in executors() {
            let unscoped = factor_bits(&mut executor);
            let (entered, scoped) = scoped_bits(&mut executor);
            assert!(entered);
            assert_eq!(scoped, unscoped);
            assert!(!IN_LINALG_SCOPE.get(), "scope flag leaks past the call");
        }
    }

    /// (caller, session) threads of the sessions the executor whose backend
    /// is `key` opens during `run`.
    fn session_threads_of(
        key: usize,
        run: impl FnOnce(),
    ) -> Vec<(std::thread::ThreadId, std::thread::ThreadId)> {
        let lock = || SESSION_THREADS.lock().unwrap_or_else(|p| p.into_inner());
        let before = lock().len();
        run();
        lock()[before..]
            .iter()
            .filter(|&&(recorded, ..)| recorded == key)
            .map(|&(_, caller, session)| (caller, session))
            .collect()
    }

    /// Tenferro-side evidence, not symtn bookkeeping: inside the scope every
    /// session closure runs on the body's own thread, so no session installs
    /// onto the pool again; outside it each session hops to a pool worker.
    #[test]
    fn scoped_sessions_run_on_the_body_thread_and_unscoped_ones_hop() {
        let mut executor = DefaultDenseExecutor::new();
        let key = session_key(&executor.backend);
        let mut body_thread = None;
        let mut scoped = Vec::new();
        executor
            .with_linalg_scope(&mut |dense| {
                body_thread = Some(std::thread::current().id());
                scoped = session_threads_of(key, || {
                    factor_bits(dense);
                });
                Ok(())
            })
            .unwrap();
        let body_thread = body_thread.unwrap();
        assert_ne!(
            body_thread,
            std::thread::current().id(),
            "body runs on a pool worker"
        );
        assert_eq!(scoped.len(), 6);
        assert!(scoped
            .iter()
            .all(|&(caller, session)| caller == body_thread && session == body_thread));

        let unscoped = session_threads_of(key, || {
            factor_bits(&mut executor);
        });
        assert_eq!(unscoped.len(), 6);
        assert!(unscoped
            .iter()
            .all(|&(caller, session)| caller == std::thread::current().id() && session != caller));
    }

    #[test]
    fn nested_scope_runs_inline_in_the_outer_scope() {
        for mut executor in executors() {
            let unscoped = factor_bits(&mut executor);
            let mut inner = None;
            executor
                .with_linalg_scope(&mut |dense| {
                    dense.with_linalg_scope(&mut |dense| {
                        inner = Some((IN_LINALG_SCOPE.get(), factor_bits(dense)));
                        Ok(())
                    })
                })
                .unwrap();
            assert_eq!(inner, Some((true, unscoped)));
        }
    }

    #[test]
    fn active_tenferro_scope_falls_back_to_per_call_sessions() {
        for mut executor in executors() {
            let unscoped = factor_bits(&mut executor);
            let witness = executor.backend.clone();
            let (entered, scoped) = witness
                .with_execution_scope(|| scoped_bits(&mut executor))
                .unwrap();
            assert!(
                !entered,
                "Tenferro rejects a nested scope; body runs unscoped"
            );
            assert_eq!(scoped, unscoped);
        }
    }

    /// A `DefaultDenseExecutor` on a caller-owned CPU domain, or `None` where
    /// the provider refuses such a domain with the typed error it must use.
    fn caller_owned_executor() -> Option<DefaultDenseExecutor> {
        use std::num::NonZeroUsize;
        use tenferro_cpu::{
            discover_cpu_topology, CpuPlacementGuarantee, ExternalCpuDomain, ResolvedCpuPlacement,
        };
        let id = tenferro_tensor::CpuDomainId::new(1389);
        let domain = ExternalCpuDomain::new(
            id,
            ResolvedCpuPlacement::AllAllowed {
                cpus: discover_cpu_topology().unwrap().allowed_cpus().clone(),
            },
            Arc::new(CpuContext::with_threads(2).unwrap()),
            NonZeroUsize::new(2).unwrap(),
            CpuPlacementGuarantee::AdvisoryDeclared,
        )
        .unwrap();
        match CpuBackend::from_external_managed_domains(id, [domain]) {
            Ok(backend) => Some(DefaultDenseExecutor::from_backend(backend)),
            // A provider whose thread count is process-global (Accelerate)
            // cannot serve a caller-owned domain at all, so there is no
            // fallback to exercise; the refusal itself must be typed.
            Err(tenferro_cpu::CpuBackendError::Tensor(error)) => {
                let source = std::error::Error::source(&error)
                    .and_then(|source| {
                        source.downcast_ref::<tenferro_cpu::CpuProviderBundleInstallError>()
                    })
                    .expect("caller-owned domain refusal carries its provider install error");
                assert!(
                    matches!(
                        source,
                        tenferro_cpu::CpuProviderBundleInstallError::IncompatibleDomain {
                            source:
                                tenferro_cpu::CpuProviderDomainError::ThreadCountNotEnforceable { .. },
                            ..
                        }
                    ),
                    "unexpected refusal: {source}"
                );
                None
            }
            Err(error) => panic!("caller-owned domain construction failed: {error}"),
        }
    }

    #[test]
    fn caller_owned_cpu_domain_falls_back_to_per_call_sessions() {
        let Some(mut executor) = caller_owned_executor() else {
            return;
        };
        let (entered, scoped) = scoped_bits(&mut executor);
        assert!(
            !entered,
            "Tenferro scopes need a managed domain; body runs unscoped"
        );
        assert_eq!(scoped, factor_bits(&mut DefaultDenseExecutor::new()));
    }

    const PARTITIONS: [&[usize]; 2] = [&[4, 1, 1, 1], &[1, 4, 1, 4]];

    /// An op-bearing c64 GEMM phase with run partition `pattern`: a run of
    /// length >= 2 is equal-shape, constant-stride (batchable) jobs, and a
    /// singleton differs in shape from its neighbours.
    struct Partition {
        jobs: Vec<DenseGemmBatchJob>,
        runs: Vec<usize>,
        lhs: Vec<Complex64>,
        rhs: Vec<Complex64>,
        dst: Vec<Complex64>,
    }

    impl Partition {
        fn new(pattern: &[usize]) -> Self {
            let mut jobs = Vec::new();
            let (mut l, mut r, mut d) = (0usize, 0usize, 0usize);
            for (index, &len) in pattern.iter().enumerate() {
                let (rows, contracted, cols) = if len >= 2 {
                    (3, 4, 2)
                } else {
                    (2 + index % 2, 3, 4)
                };
                for _ in 0..len {
                    jobs.push(DenseGemmBatchJob {
                        dst_offset: d,
                        lhs_offset: l,
                        rhs_offset: r,
                        rows,
                        contracted,
                        cols,
                    });
                    l += rows * contracted;
                    r += contracted * cols;
                    d += rows * cols;
                }
            }
            let value = |i: usize| {
                Complex64::new(
                    ((i * 7 + 3) % 11) as f64 * 0.25 - 1.0,
                    ((i * 5) % 7) as f64 * 0.125,
                )
            };
            Self {
                jobs,
                runs: pattern.to_vec(),
                lhs: (0..l).map(value).collect(),
                rhs: (0..r).map(|i| value(i + 5)).collect(),
                dst: (0..d).map(|i| value(i + 9)).collect(),
            }
        }

        /// Output bits after one call with `jobs`/`runs`, from the fixture's
        /// accumulator, with nonzero alpha and beta.
        fn run(
            &self,
            dense: &mut dyn DenseExecutor,
            jobs: &[DenseGemmBatchJob],
            runs: &[usize],
            dst: &mut [Complex64],
        ) {
            let s = [1usize];
            let dl = dst.len();
            dense
                .matmul_batch_axpby_with_ops_into(
                    DenseWrite::C64(DenseViewMut::new(dst, &[dl], &s, 0).unwrap()),
                    DenseRead::C64(DenseView::new(&self.lhs, &[self.lhs.len()], &s, 0).unwrap()),
                    DenseRead::C64(DenseView::new(&self.rhs, &[self.rhs.len()], &s, 0).unwrap()),
                    jobs,
                    runs,
                    MatrixOp::Adjoint,
                    MatrixOp::Identity,
                    DenseScalar::C64(Complex64::new(0.75, -0.5)),
                    DenseScalar::C64(Complex64::new(-0.25, 0.125)),
                )
                .unwrap();
        }

        fn whole(&self, dense: &mut dyn DenseExecutor) -> Vec<u64> {
            let mut dst = self.dst.clone();
            self.run(dense, &self.jobs, &self.runs, &mut dst);
            bits_c64(&dst)
        }

        /// The same GEMMs issued one run per call.
        fn per_run(&self, dense: &mut dyn DenseExecutor) -> Vec<u64> {
            let mut dst = self.dst.clone();
            let mut start = 0;
            for &len in &self.runs {
                self.run(dense, &self.jobs[start..start + len], &[len], &mut dst);
                start += len;
            }
            bits_c64(&dst)
        }
    }

    fn bits_c64(values: &[Complex64]) -> Vec<u64> {
        values
            .iter()
            .flat_map(|v| [v.re.to_bits(), v.im.to_bits()])
            .collect()
    }

    /// Tenferro-side evidence for the scoped partition: at default threads
    /// every serial session of the phase runs on the scope body's thread (no
    /// second pool install), while one call per run hops for each session.
    #[test]
    fn mixed_partition_sessions_run_in_one_scope_without_a_pool_hop() {
        for pattern in PARTITIONS {
            let fixture = Partition::new(pattern);
            let mut executor = DefaultDenseExecutor::new();
            let key = session_key(&executor.backend);
            let here = std::thread::current().id();
            let mut whole = Vec::new();
            let scoped = session_threads_of(key, || whole = fixture.whole(&mut executor));
            let serial_stretches = if pattern == [4, 1, 1, 1] { 1 } else { 2 };
            assert_eq!(scoped.len(), serial_stretches, "{pattern:?}");
            assert!(
                scoped
                    .iter()
                    .all(|&(caller, session)| caller != here && session == caller),
                "{pattern:?}: sessions must run inline on the scope body's worker"
            );

            let mut per_run = Vec::new();
            let unscoped = session_threads_of(key, || per_run = fixture.per_run(&mut executor));
            assert_eq!(
                unscoped.len(),
                pattern.iter().filter(|&&len| len == 1).count()
            );
            assert!(unscoped
                .iter()
                .all(|&(caller, session)| caller == here && session != caller));
            assert_eq!(whole, per_run, "{pattern:?}");
        }
    }

    /// A single-run partition is one dispatch, so it enters no scope: its one
    /// serial session is admitted (and hops to the pool) like any other.
    #[test]
    fn single_run_partition_enters_no_scope() {
        let singletons = Partition::new(&[1, 1]);
        // Two jobs of different shapes as one non-batchable run.
        let fixture = Partition {
            runs: vec![2],
            ..singletons
        };
        let mut executor = DefaultDenseExecutor::new();
        let key = session_key(&executor.backend);
        let here = std::thread::current().id();
        let mut whole = Vec::new();
        let sessions = session_threads_of(key, || whole = fixture.whole(&mut executor));
        assert_eq!(sessions.len(), 1);
        assert!(sessions
            .iter()
            .all(|&(caller, session)| caller == here && session != caller));
        let mut reference = fixture.dst.clone();
        for job in &fixture.jobs {
            fixture.run(
                &mut executor,
                std::slice::from_ref(job),
                &[1],
                &mut reference,
            );
        }
        assert_eq!(whole, bits_c64(&reference));
    }

    /// A caller-owned domain cannot enter a scope: the phase runs unscoped on
    /// the calling thread, still coalesced, with the per-run result.
    #[test]
    fn mixed_partition_on_a_caller_owned_domain_runs_unscoped() {
        for pattern in PARTITIONS {
            let Some(mut executor) = caller_owned_executor() else {
                return;
            };
            let fixture = Partition::new(pattern);
            let key = session_key(&executor.backend);
            let here = std::thread::current().id();
            let mut whole = Vec::new();
            let sessions = session_threads_of(key, || whole = fixture.whole(&mut executor));
            let serial_stretches = if pattern == [4, 1, 1, 1] { 1 } else { 2 };
            assert_eq!(sessions.len(), serial_stretches, "{pattern:?}");
            assert!(sessions.iter().all(|&(caller, _)| caller == here));
            assert_eq!(whole, fixture.per_run(&mut DefaultDenseExecutor::new()));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_context_without_kind_uses_the_compiled_provider_default() {
        let context = SharedCpuContext::with_threads(1).expect("serial context");
        let executor =
            DefaultDenseExecutor::with_shared_context(&context, None).expect("default provider");

        assert_eq!(executor.backend.kind(), CpuBackendKind::default_compiled());
    }
}
