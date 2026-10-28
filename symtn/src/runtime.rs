//! User-layer runtime: owns shared execution and cache state so everyday
//! tensor code never passes explicit contexts around.

use std::any::Any;
use std::hash::Hash;
use std::num::NonZeroUsize;
#[cfg(feature = "cuda")]
use std::sync::PoisonError;
use std::sync::{Arc, Mutex, MutexGuard, Weak};

use num_complex::{Complex32, Complex64};
use symtn_core::{HomSpaceId, RuleIdentity};
use symtn_tensors::{
    BoundDynamicFusionMapSpace, DenseTreeTransformOperations, TensorContractFusionExecutionContext,
    TreeTransformOperation,
};

use crate::error::Error;
use crate::typed::ScalarOps;

#[cfg(feature = "cuda")]
mod cuda;
mod host_scratch;
mod lanes;
mod leases;
#[cfg(test)]
mod tests;

#[cfg(feature = "cuda")]
pub use self::cuda::CudaTreeTransformStats;
#[cfg(feature = "cuda")]
pub(crate) use self::cuda::*;
pub(crate) use self::lanes::*;
pub use self::lanes::{Ctx, Ctxs};
pub(crate) use self::leases::*;

/// Type-keyed storage for state a downstream crate keeps on a [`Runtime`].
///
/// Each value lives under its own type, so a lookup returns the value of the
/// requested type or nothing. Why not one `Option<Box<dyn Any>>`: a value of
/// another type in that single slot left its owner a choice between a panic
/// and silently dropping someone else's state.
#[doc(hidden)]
#[derive(Default)]
pub struct ExtensionSlot {
    values: Vec<Box<dyn Any + Send>>,
}

impl ExtensionSlot {
    /// The stored value of type `T`, if one was inserted.
    pub fn get_mut<T: Any + Send>(&mut self) -> Option<&mut T> {
        self.values
            .iter_mut()
            .find_map(|value| value.as_mut().downcast_mut::<T>())
    }

    /// The stored value of type `T`, inserting `init()` first if there is none.
    pub fn get_or_insert_with<T: Any + Send>(&mut self, init: impl FnOnce() -> T) -> &mut T {
        let index = match self
            .values
            .iter()
            .position(|value| value.as_ref().is::<T>())
        {
            Some(index) => index,
            None => {
                self.values.push(Box::new(init()));
                self.values.len() - 1
            }
        };
        self.values[index]
            .as_mut()
            .downcast_mut::<T>()
            .expect("the value at `index` was found or inserted as a `T`")
    }
}

struct RuntimeInner {
    // Standalone CPU ops normally lease from `context_pool`/`executor_pool`,
    // while network execution uses per-plan workspace pools. Pool checkout,
    // cache access, provider resources, and a non-mintable injected executor
    // can still synchronize. Device operations do not take this lock: they
    // serialize on the device-local `cuda` mutex below instead.
    state: Mutex<RuntimeState>,
    execution_config: RuntimeExecutionConfig,
    /// Standalone operations lease an execution context and, for factorization,
    /// a dense executor. Both pools mint on empty, bound their idle count, and
    /// quarantine a resource that unwinds while leased.
    ///
    /// `DenseExecutor` takes `&mut self` for per-call scratch, so leases own
    /// mutable executor state rather than sharing one executor by `&`.
    context_pool: Mutex<Vec<PooledContext>>,
    executor_pool: Mutex<Vec<Box<dyn symtn_dense::DenseExecutor + Send>>>,
    /// `false` when a caller injected a custom (non-mintable) executor via
    /// `with_dense_executor`: the pool cannot reproduce it, so factorizations
    /// fall back to the `state` lock and its single executor.
    executor_mintable: bool,
    /// Idle-pool cap derived from available parallelism, with a minimum of two
    /// and a fallback of four when the system does not report a value.
    max_idle: usize,
    /// Contraction-plan cache behind its own mutex, separate from `state` but
    /// still synchronized while cache work is in progress.
    extensions: Mutex<ExtensionSlot>,
    /// The single CUDA context of this runtime and its device's process-wide
    /// lock. Device operations serialize on both (see [`CudaLease`]) instead
    /// of on `state`.
    #[cfg(feature = "cuda")]
    cuda: Option<CudaHome>,
    /// Device ordinal of `cuda`, fixed at build time and readable without a
    /// lock so placement preflight takes no device lock at all.
    #[cfg(feature = "cuda")]
    cuda_device: Option<usize>,
}

/// Pool entry for `RuntimeInner::context_pool`. Boxed: a context owns both the
/// multiplicity-free and Generic-fusion lanes, each with `f64` and `c64`
/// state. `Vec::pop`/`push` must move only a pointer rather than that complete
/// execution state; profiling showed by-value moves were ~70% of a small
/// standalone op's cost (issue #228). The pointer-size canary test below
/// prevents that regression.
type PooledContext = Box<TensorExecutionContext>;

/// Mints a dense executor identical to the one `RuntimeBuilder::build` created
/// from the same config. Only called when no custom executor was injected
/// (`executor_mintable`), and that config already built successfully once, so a
/// re-mint cannot fail. ponytail: CPU executors are cheap to construct; the pool
/// exists to skip per-call construction, not because minting is expensive.
fn mint_dense(config: &RuntimeExecutionConfig) -> Box<dyn symtn_dense::DenseExecutor + Send> {
    Box::new(symtn_dense::DefaultDenseExecutor::with_shared_context(
        &config.shared_ctx,
    ))
}

// No `dense_threads` field: the thread count is baked into `shared_ctx` at
// build time, so executors minted later cannot drift from it.
#[derive(Clone)]
pub(crate) struct RuntimeExecutionConfig {
    pub(crate) recoupling_threads: Option<NonZeroUsize>,
    /// Runtime CPU context shared by built-in executors: the state, mintable
    /// executor pool, and transform backends. An injected executor owns its
    /// own configuration.
    pub(crate) shared_ctx: symtn_dense::SharedCpuContext,
}

/// Execution runtime for the user-layer [`crate::typed::TensorMap`] API.
///
/// A `Runtime` is built once via [`Runtime::builder`] and then carried
/// implicitly by every tensor created from it; operations reuse the
/// Runtime-owned completed tree-transform store without explicit context
/// arguments. Context-local operation caches stay disabled; cloning a
/// `Runtime` clones a shared handle, not the state.
///
/// Concurrency: standalone tensor operations can lease independent execution
/// contexts instead of holding the coarse state mutex for their full duration.
/// Pool checkout, shared stores, the plan cache, and dense providers retain
/// their own synchronization and capability limits. Device operations take
/// only the device lock (below) and a mutex over this runtime's single CUDA
/// context, not the state mutex, so Host work is not blocked by device work.
///
/// CUDA: every device operation of every `Runtime` built for the same device
/// ordinal holds one process-wide lock for that device while it submits work,
/// so device operations on one device serialize their host-side enqueue across
/// Runtimes (the GPU work itself stays asynchronous; the lock adds no host
/// sync). A host sync that already happens under a lease — `to_host`, scalar
/// and spectrum downloads — therefore
/// stalls every Runtime on the device, not only the caller. This is what makes
/// a fresh device output, returned by any operation that creates a new
/// tensor, safe to read through another Runtime on the same device (#1384).
///
/// A buffer bound in an earlier lease and written again later (a
/// `*_into` destination, or reused scratch) is ordered by the
/// device's single CubeCL stream instead (#1391): the first device Runtime of
/// the process sets CubeCL's `streaming.max_streams` to 1, so all device work
/// of every thread, symtn or not, runs in enqueue order and never overlaps on
/// the GPU. Building a device Runtime fails with an unsupported error if
/// CubeCL's configuration was already loaded with more streams.
/// Process-wide side effects: a `cubecl.toml` `streaming.max_streams` value is
/// overridden without notice, the setting stays fixed even if opening the
/// device then fails, every other CubeCL client in the process (wgpu
/// included) also gets one stream, and a later `CubeClRuntimeConfig::set`
/// panics. One ordering residual, independent of the stream count, is tracked
/// in tensor4all/tenferro-rs#1868 (see `docs/backend_policy.md`).
///
/// # Examples
///
/// ```
/// use std::sync::Arc;
/// use symtn::sector::{Z2FusionRule, Z2Irrep};
/// use symtn::typed::{GradedSpace, Runtime, TensorMap};
///
/// let rt = Runtime::builder().build()?;
/// let v = GradedSpace::try_new(
///     Arc::new(Z2FusionRule),
///     [(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 1)],
/// )?;
/// let a: TensorMap<_, f64> = TensorMap::zeros(&rt, [&v], [&v])?;
/// assert_eq!(a.norm(2.0)?, 0.0);
/// # Ok::<(), symtn::typed::Error>(())
/// ```
#[derive(Clone)]
pub struct Runtime {
    inner: Arc<RuntimeInner>,
}

/// Non-owning identity for internal state parked outside an active execution.
///
/// Equality and hashing use the Runtime allocation's address. The held `Weak`
/// keeps that allocation reserved, so a later Runtime can never reuse the
/// address while this identity is live.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct RuntimeIdentity {
    inner: Weak<RuntimeInner>,
}

impl PartialEq for RuntimeIdentity {
    fn eq(&self, other: &Self) -> bool {
        Weak::ptr_eq(&self.inner, &other.inner)
    }
}

impl Eq for RuntimeIdentity {}

impl std::hash::Hash for RuntimeIdentity {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.inner.as_ptr().hash(state);
    }
}

impl RuntimeIdentity {
    #[doc(hidden)]
    pub fn matches(&self, runtime: &Runtime) -> bool {
        self.inner
            .upgrade()
            .is_some_and(|inner| Arc::ptr_eq(&inner, &runtime.inner))
    }

    #[doc(hidden)]
    pub fn is_alive(&self) -> bool {
        self.inner.strong_count() != 0
    }
}

impl Runtime {
    /// Starts building a runtime. The default runtime uses the CPU backend;
    /// feature-gated device options such as CUDA are attached through the
    /// builder so tensor construction keeps the same shape.
    pub fn builder() -> RuntimeBuilder {
        RuntimeBuilder::default()
    }

    pub(crate) fn lock(&self) -> MutexGuard<'_, RuntimeState> {
        // ponytail: poisoning treated as fatal; no operation leaves the
        // caches half-written in a way worth recovering from.
        self.inner
            .state
            .lock()
            .expect("symtn runtime state poisoned")
    }

    pub(crate) fn same_runtime(&self, other: &Runtime) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    pub(crate) fn identity(&self) -> RuntimeIdentity {
        RuntimeIdentity {
            inner: Arc::downgrade(&self.inner),
        }
    }

    #[cfg(test)]
    pub(crate) fn execution_config(&self) -> &RuntimeExecutionConfig {
        &self.inner.execution_config
    }

    /// The resolved transformer of a typed Host `*_into` request whose spaces
    /// were proved to match once before (the exact-layout memo on the
    /// process-global entry of coefficient type `C`); `None` takes the full
    /// path.
    pub(crate) fn exact_layout_tree_pair_hit_of<R, C: 'static + Send + Sync>(
        rule: &RuleIdentity,
        operation: symtn_tensors::TreeTransformOperationView<'_>,
        source: &BoundDynamicFusionMapSpace<R>,
        destination: &BoundDynamicFusionMapSpace<R>,
    ) -> Option<symtn_tensors::TreeTransformStructure<C>> {
        let (source_homspace, source_layout) = bound_layout_identity(source);
        let (destination_homspace, destination_layout) = bound_layout_identity(destination);
        symtn_tensors::exact_layout_tree_pair_hit::<C>(
            rule,
            operation,
            destination.space().structure(),
            source.space().structure(),
            (&source_homspace, source_layout),
            (&destination_homspace, destination_layout),
        )
    }

    pub(crate) fn admit_exact_tree_pair_layout_of<R, C: 'static + Send + Sync>(
        rule: &RuleIdentity,
        operation: &TreeTransformOperation,
        source: &BoundDynamicFusionMapSpace<R>,
        destination: &BoundDynamicFusionMapSpace<R>,
    ) -> bool {
        let (source_homspace, source_layout) = bound_layout_identity(source);
        let (destination_homspace, destination_layout) = bound_layout_identity(destination);
        symtn_tensors::admit_exact_tree_pair_layout::<C>(
            rule,
            operation,
            destination.space().structure(),
            source.space().structure(),
            (&source_homspace, source_layout),
            (&destination_homspace, destination_layout),
        )
    }

    /// Enters this runtime's CPU pool as the calling thread's Host pool until
    /// the guard drops. Every Host eager operation runs inside one: through
    /// its execution lease, or directly for the strided-only operations.
    pub(crate) fn enter_host_pool(&self) -> HostPoolEntry {
        HostPoolEntry {
            _guard: symtn_operations::host_pool::enter_host_pool(
                &self.inner.execution_config.shared_ctx,
            ),
            #[cfg(test)]
            _probe: HostPoolDepth::enter(),
        }
    }

    /// Leases an execution context for one standalone op: pop an idle one or
    /// mint a fresh config-bound one. Each context owns one
    /// `RuleIdentity`-keyed multiplicity-free lane and a separate Generic-fusion
    /// lane. The coarse `state` lock is not held during that lease, but pool
    /// checkout and other shared resources remain synchronized.
    pub(crate) fn lease_context(&self) -> Result<ContextLease<'_>, Error> {
        let pooled = self
            .inner
            .context_pool
            .lock()
            .expect("context pool poisoned")
            .pop();
        let context = match pooled {
            Some(context) => context,
            None => Box::new(TensorExecutionContext::for_config(
                &self.inner.execution_config,
            )?),
        };
        Ok(ContextLease {
            pool: &self.inner.context_pool,
            max_idle: self.inner.max_idle,
            context: Some(context),
            _host_pool: self.enter_host_pool(),
        })
    }

    /// Leases a dense executor for one factorization (#155). Pooled for a
    /// mintable config; otherwise falls back to the `state` lock and its single
    /// injected executor (which cannot be reproduced for a pool).
    pub(crate) fn lease_dense(&self) -> DenseLease<'_> {
        #[cfg(test)]
        crate::typed::probe::hit(crate::typed::probe::Event::DenseLease);
        if !self.inner.executor_mintable {
            return DenseLease::Locked {
                state: self.lock(),
                _host_pool: self.enter_host_pool(),
            };
        }
        let executor = self
            .inner
            .executor_pool
            .lock()
            .expect("executor pool poisoned")
            .pop()
            .unwrap_or_else(|| mint_dense(&self.inner.execution_config));
        DenseLease::Pooled {
            pool: &self.inner.executor_pool,
            max_idle: self.inner.max_idle,
            executor: Some(executor),
            _host_pool: self.enter_host_pool(),
        }
    }

    /// Locked access to the type-erased downstream extension slot (the
    /// contraction-plan cache and its configuration: the types live in
    /// `symtn-network`, which depends on this crate and claims and downcasts
    /// its entry on first use). Behind its own mutex rather than the coarse
    /// `state` one (#155), so the network hot path never contends with
    /// standalone ops. Expert seam for `symtn-network`; the mutex is held for
    /// the duration of `f`, so do not run tensor operations inside it.
    pub(crate) fn with_extension_slot<R>(&self, f: impl FnOnce(&mut ExtensionSlot) -> R) -> R {
        f(&mut self
            .inner
            .extensions
            .lock()
            .expect("symtn extension slot poisoned"))
    }

    /// CUDA device ordinal fixed when this runtime was built. Lock-free: the
    /// ordinal is immutable, so placement preflight needs no device lock.
    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_device_ordinal(&self) -> Option<usize> {
        self.inner.cuda_device
    }

    /// The device ordinal, or the missing-device error every device operation
    /// reports. Preflight-only sites use this instead of a lease.
    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_device_ordinal_checked(&self) -> Result<usize, Error> {
        self.inner.cuda_device.ok_or_else(missing_cuda_device)
    }

    /// Leases the device state for a path that only drops or reads it.
    ///
    /// Unlike [`Self::lease_cuda`], a poisoned mutex is recovered rather than
    /// re-panicked: clearing prepared device state and reading its byte
    /// counters are safe whatever a panicking device operation left behind —
    /// every entry is a dtype conversion of data the structure still owns, and
    /// the workspaces are fully rewritten before they are read. Turning
    /// another thread's panic into a panic in a cache-clear or a statistics
    /// read would be the worse contract.
    #[cfg(feature = "cuda")]
    pub(crate) fn lease_cuda_for_maintenance(&self) -> Option<CudaLease<'_>> {
        self.inner.cuda.as_ref().map(|cuda| {
            let device = lock_device(cuda.device_lock);
            CudaLease {
                state: cuda.state.lock().unwrap_or_else(PoisonError::into_inner),
                _device: device,
            }
        })
    }

    /// Leases this runtime's single CUDA context for one device operation.
    ///
    /// Device operations validate first, then lease, then execute: they hold
    /// only the device lock and this Runtime's device state, never the coarse
    /// `state` lock, so Host work on the same runtime is not blocked by device
    /// work. Device operations of every Runtime on this device serialize here.
    #[cfg(feature = "cuda")]
    pub(crate) fn lease_cuda(&self) -> Result<CudaLease<'_>, Error> {
        // ponytail: poisoning treated as fatal, as for `lock()` and the pools.
        self.inner
            .cuda
            .as_ref()
            .ok_or_else(missing_cuda_device)
            .map(|cuda| {
                let device = lock_device(cuda.device_lock);
                CudaLease {
                    state: cuda.state.lock().expect("symtn cuda context poisoned"),
                    _device: device,
                }
            })
    }

    /// Device state this Runtime's tree-transform executor and device context
    /// retain, or `None` when the Runtime has no device.
    ///
    /// Read-only observation; it takes the device lease, so it waits behind a
    /// device operation in progress. A device lock poisoned by an earlier
    /// panic is recovered and the recovered state's counters are returned,
    /// rather than panicking inside a statistics read.
    #[cfg(feature = "cuda")]
    pub fn cuda_tree_transform_stats(&self) -> Option<CudaTreeTransformStats> {
        let mut lease = self.lease_cuda_for_maintenance()?;
        let (dense, executor) = lease.split();
        Some(CudaTreeTransformStats {
            prepared_structures: executor.prepared_structures(),
            executor_bytes: executor.executor_device_bytes(),
            workspace_bytes: executor.workspace_device_bytes(),
            context_scalar_operand_bytes: dense.scalar_operand_bytes(),
            required_plan_entries: executor.required_plan_entries(),
        })
    }

    /// Device bytes the general contraction's operand and core-destination
    /// scratch holds, or `None` when the Runtime has no device.
    ///
    /// Read-only and grow-only: the scratch keeps its high-water allocation
    /// per payload dtype until [`Self::clear_cuda_caches`]. It is
    /// reported apart from [`CudaTreeTransformStats`] so each device byte is
    /// counted once, and is not charged to
    /// `symtn-network`'s `PlanCacheConfig::workspace_budget_bytes`. Takes the device lease like
    /// [`Self::cuda_tree_transform_stats`].
    #[cfg(feature = "cuda")]
    pub fn cuda_contract_scratch_bytes(&self) -> Option<usize> {
        let mut lease = self.lease_cuda_for_maintenance()?;
        Some(lease.split_contract().2.device_bytes())
    }

    /// Snapshot of the cuTENSOR plan cache of this Runtime's device context,
    /// or `Ok(None)` when the Runtime has no device.
    ///
    /// Read-only observation for warm-cost contracts: every region move,
    /// zero fill and GEMM a device operation submits builds or reuses one
    /// cuTENSOR plan per operand signature, so `misses` not growing across a
    /// warm call shows it rebuilt no plan, and `evictions` growing shows
    /// thrash. Nothing reads it back to decide execution. Takes the device
    /// lease like [`Self::cuda_tree_transform_stats`]; an error is the
    /// backend's own statistics query failing.
    #[cfg(feature = "cuda")]
    pub fn cuda_plan_cache_stats(&self) -> Result<Option<symtn_dense::CudaPlanCacheStats>, Error> {
        let Some(lease) = self.lease_cuda_for_maintenance() else {
            return Ok(None);
        };
        lease
            .plan_cache_stats()
            .map(Some)
            .map_err(|error| Error::from(symtn_operations::OperationError::Dense(error)))
    }

    /// Drops this Runtime's retained device execution state; a no-op when the
    /// Runtime has no device.
    ///
    /// Clears the tree-transform executor's prepared structures (uploaded
    /// coefficient and recoupling vectors), its pack/scatter workspaces and
    /// the plan-entry reservation it holds on the device context, and the
    /// general contraction's operand and core-destination scratch. After it,
    /// [`CudaTreeTransformStats`]'s `prepared_structures`, `executor_bytes`,
    /// `workspace_bytes` and `required_plan_entries` and
    /// [`Self::cuda_contract_scratch_bytes`] read 0; the next device operation
    /// re-prepares what it needs, so this is a memory decision, never a
    /// correctness one.
    ///
    /// It does not touch the process-global semantic caches shared by every
    /// Runtime ([`crate::cache::clear`] does that, and leaves device state
    /// alone), the device context's shared scalar operands
    /// (`context_scalar_operand_bytes`) or its cuTENSOR plan cache, which the
    /// context owns and bounds itself, or `symtn-network`'s per-Runtime plan
    /// cache (`clear_plan_cache`).
    ///
    /// Takes the device lease, so it waits behind a device operation in
    /// progress. A device lock poisoned by an earlier panic is recovered
    /// rather than propagated: dropping prepared device state is always safe.
    #[cfg(feature = "cuda")]
    pub fn clear_cuda_caches(&self) {
        if let Some(mut lease) = self.lease_cuda_for_maintenance() {
            let (dense, executor, scratch) = lease.split_contract();
            executor.clear(dense);
            scratch.clear();
        }
    }
}

fn bound_layout_identity<R>(space: &BoundDynamicFusionMapSpace<R>) -> (HomSpaceId, [usize; 3]) {
    let space = space.space();
    (
        space.homspace().id(),
        [
            space.structure().content_id(),
            space.homspace().codomain().len(),
            space.homspace().domain().len(),
        ],
    )
}

impl std::fmt::Debug for Runtime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Runtime").finish_non_exhaustive()
    }
}

/// Contradictory or invalid [`RuntimeBuilder`] settings, rejected by
/// [`RuntimeBuilder::build`] instead of being silently adjusted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RuntimeConfigError {
    /// [`RuntimeBuilder::dense_threads`] was given zero.
    ZeroDenseThreads,
    /// [`RuntimeBuilder::recoupling_threads`] was given zero.
    ZeroRecouplingThreads,
}

impl std::fmt::Display for RuntimeConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::ZeroDenseThreads => "dense_threads must be at least 1",
            Self::ZeroRecouplingThreads => "recoupling_threads must be at least 1",
        })
    }
}

impl std::error::Error for RuntimeConfigError {}

/// Builder for [`Runtime`]; see [`Runtime::builder`].
///
/// Not `Clone`/`Debug`-derivable: an injected dense executor
/// ([`Self::with_dense_executor`]) is a `Box<dyn DenseExecutor>`, which is
/// neither cloneable nor `Debug`. A manual [`std::fmt::Debug`] is provided that
/// reports the executor's presence without touching it.
#[derive(Default)]
pub struct RuntimeBuilder {
    #[cfg(feature = "cuda")]
    cuda_device: Option<usize>,
    dense_threads: Option<usize>,
    recoupling_threads: Option<usize>,
    /// User-injected CPU linear-algebra backend. When absent, the compiled
    /// provider is used.
    dense_executor: Option<Box<dyn symtn_dense::DenseExecutor + Send>>,
}

impl std::fmt::Debug for RuntimeBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("RuntimeBuilder");
        #[cfg(feature = "cuda")]
        s.field("cuda_device", &self.cuda_device);
        s.field("dense_threads", &self.dense_threads)
            .field("recoupling_threads", &self.recoupling_threads)
            .field("dense_executor", &self.dense_executor.is_some())
            .finish()
    }
}

impl RuntimeBuilder {
    /// Attaches a CUDA device (by ordinal) to the runtime. Tensors stay on
    /// the host until moved explicitly with
    /// [`crate::typed::TensorMap::to_cuda`]; there are no implicit
    /// transfers. Device initialization happens in [`Self::build`].
    #[cfg(feature = "cuda")]
    pub fn cuda(mut self, device: usize) -> Self {
        self.cuda_device = Some(device);
        self
    }

    /// Sets the worker count of this runtime's CPU pool. The runtime owns one
    /// pool; its Host eager operations run dense kernels, tree-transform
    /// replay, plan compile and strided kernels on it, and building a runtime
    /// never touches Rayon's process-global pool. Runtimes with different
    /// counts coexist in one process.
    ///
    /// Unset, the pool uses the process's available parallelism; the runtime
    /// reads no environment variable. A value of 1 creates no worker pool:
    /// every Host operation of this runtime runs on its calling thread. Zero
    /// makes [`Self::build`] fail with [`RuntimeConfigError::ZeroDenseThreads`].
    /// The count does not configure provider-internal (BLAS) or
    /// injected-executor threads.
    pub fn dense_threads(mut self, threads: usize) -> Self {
        self.dense_threads = Some(threads);
        self
    }

    /// Selects the CPU factorization backend (SVD / QR / eigh / eig / inv / exp
    /// on the coupled-sector matrices) by injecting a [`crate::expert::DenseExecutor`].
    /// When no executor is injected, the provider is the one compiled in by
    /// cargo feature: BLAS/LAPACK with a `cpu-blas`/`blas-*` feature, otherwise
    /// faer. Implement `DenseExecutor` and pass it here for any other backend;
    /// no operator or decomposition code changes. Contraction GEMM is not
    /// delegated to the injected executor; it always runs on the compiled
    /// provider.
    ///
    /// The injected executor owns its thread configuration. [`Self::dense_threads`]
    /// sizes the runtime CPU pool, which still runs this runtime's replay,
    /// plan compile and strided work; it does not reconfigure the injected
    /// backend.
    pub fn with_dense_executor(
        mut self,
        executor: Box<dyn symtn_dense::DenseExecutor + Send>,
    ) -> Self {
        self.dense_executor = Some(executor);
        self
    }

    /// Sets the CPU worker count for symmetry recoupling replays
    /// (permute/braid/transpose tree transforms — the cold-path cost of
    /// SU(2) workloads; **not** BLAS threads) and for tree-transform plan
    /// compile. Default is 1 (serial); values above 1 request parallelism
    /// past the backend's size gate, capped by and run on this runtime's CPU
    /// pool ([`Self::dense_threads`]). Zero makes [`Self::build`] fail with
    /// [`RuntimeConfigError::ZeroRecouplingThreads`].
    pub fn recoupling_threads(mut self, threads: usize) -> Self {
        self.recoupling_threads = Some(threads);
        self
    }

    /// Finishes the build; fails with [`RuntimeConfigError`] on contradictory
    /// or invalid settings, and when a requested backend (e.g. the CUDA
    /// device) cannot be initialized.
    pub fn build(self) -> Result<Runtime, Error> {
        if self.dense_threads == Some(0) {
            return Err(RuntimeConfigError::ZeroDenseThreads.into());
        }
        let recoupling_threads = self
            .recoupling_threads
            .map(|threads| {
                NonZeroUsize::new(threads).ok_or(RuntimeConfigError::ZeroRecouplingThreads)
            })
            .transpose()?;
        // A custom injected executor cannot be re-minted for the pool; those
        // runtimes fall back to the state lock for factorizations (#155).
        let executor_mintable = self.dense_executor.is_none();
        // Built-in executors share this runtime CPU context; injected executors
        // own their configuration.
        let shared_ctx = match self.dense_threads {
            Some(threads) => symtn_dense::SharedCpuContext::with_threads(threads),
            None => symtn_dense::SharedCpuContext::with_available_parallelism(),
        }
        .map_err(symtn_tensors::OperationError::Dense)?;
        // Injected backend wins; otherwise the compiled provider on the shared
        // context.
        let dense: Box<dyn symtn_dense::DenseExecutor + Send> = match self.dense_executor {
            Some(executor) => executor,
            None => Box::new(symtn_dense::DefaultDenseExecutor::with_shared_context(
                &shared_ctx,
            )),
        };
        let mut state = RuntimeState::with_config(dense, &shared_ctx)?;
        if let Some(threads) = recoupling_threads {
            state.set_recoupling_threads(threads);
        }
        #[cfg(feature = "cuda")]
        let cuda = match self.cuda_device {
            // The backend libraries (cuTENSOR, cuSOLVER/cuBLAS) initialize
            // lazily inside tenferro, so without this the first user operation
            // on this Runtime pays a one-time ~0.2 s unrelated to its size.
            // Construction is where that cost belongs; a failure here is a
            // build failure, never a half-initialized Runtime.
            Some(device) => {
                let mut cuda = symtn_dense::CudaDenseContext::new(device)
                    .map_err(symtn_tensors::OperationError::Dense)?;
                // Registered only once the ordinal opened, so the registry
                // stays bounded by real devices. Warm-up submits work, so it
                // runs under the device lock like any other submission.
                let device_lock = cuda_device_lock(device);
                {
                    let _device = lock_device(device_lock);
                    cuda.warm_up()
                        .map_err(symtn_tensors::OperationError::Dense)?;
                }
                // The executor's prepared-structure count is a device-side
                // bound of its own: the host completed-transformer cache is
                // process-global and byte-bounded only. It bounds memory,
                // never correctness.
                Some(CudaHome {
                    device_lock,
                    state: Mutex::new(CudaDeviceState {
                        dense: cuda,
                        tree_transform: symtn_operations::CudaTreeTransformExecutor::new(
                            symtn_operations::DEFAULT_COEFFICIENT_BUDGET_BYTES,
                            symtn_operations::DEFAULT_PLAN_CACHE_BUDGET_BYTES,
                        ),
                        contract_scratch: symtn_tensors::CudaContractScratch::default(),
                    }),
                })
            }
            None => None,
        };
        // One warm context/executor per core covers a thread-per-core driver;
        // fall back to a small count if the core count is unavailable.
        let max_idle = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .max(2);
        Ok(Runtime {
            inner: Arc::new(RuntimeInner {
                state: Mutex::new(state),
                execution_config: RuntimeExecutionConfig {
                    recoupling_threads,
                    shared_ctx,
                },
                context_pool: Mutex::new(Vec::new()),
                executor_pool: Mutex::new(Vec::new()),
                executor_mintable,
                max_idle,
                extensions: Mutex::new(ExtensionSlot::default()),
                #[cfg(feature = "cuda")]
                cuda,
                #[cfg(feature = "cuda")]
                cuda_device: self.cuda_device,
            }),
        })
    }
}
