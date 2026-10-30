//! Host parallel regions on the caller's engine.
//!
//! Every numerical entry receives the caller's `engine: &mut dyn Engine`.
//! A backend-free Host region (tree-transform replay fan-out, plan compile,
//! strided kernels) takes that engine's placement token through [`region`]
//! and splits it with [`fork`] / [`join`]; leaf kernels take a token argument
//! and pass it to strided, whose own cost rule decides whether to fan out. symtn schedules no work of its own and keeps no
//! engine, pool or token anywhere: the token lives only inside [`region`]'s
//! callback, so a region cannot call the dense backend on the same engine,
//! and the next dense call can borrow the engine again once it returns.
//!
//! A CPU engine supplies its team's token (a one-thread engine a serial
//! one). Without a compiled CPU provider there is no CPU engine type, so
//! regions run on a serial token. A CUDA engine has no Host pool: the Host
//! structural metadata a device operation prepares
//! ([`HostPoolSite::PlanCompile`]) runs inline on an explicit serial token,
//! and its payload work stays on the device. Every other site moves Host
//! payload and refuses a CUDA engine, as it refuses any other engine, with a
//! typed error before the region runs.

#[cfg(any(test, feature = "testing"))]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(any(test, feature = "testing"))]
use std::sync::Mutex;
use std::sync::RwLock;

pub use strided_kernel::Exec;
use symtn_dense::{DenseError, Engine};

use crate::OperationError;

/// Runs `op` with the placement token of the caller's CPU engine. `op` must
/// not call the dense backend; it returns before the engine is used again.
#[doc(hidden)]
pub fn region<R>(
    engine: &mut dyn Engine,
    site: HostPoolSite,
    op: impl FnOnce(Exec<'_>) -> R,
) -> Result<R, OperationError> {
    #[cfg(feature = "tenferro")]
    {
        let mut op = Some(op);
        let ran = tenferro_cpu::with_cpu_engine(engine, |cpu| {
            let exec = cpu.exec().map_err(|err| {
                OperationError::Dense(DenseError::Backend {
                    backend: symtn_dense::DenseBackend::Tenferro,
                    op: "host_region",
                    message: err.to_string(),
                })
            })?;
            let exec = exec.into_token();
            observe(site, exec.budget().get());
            let op = op.take().expect("region body runs once");
            Ok(op(exec))
        });
        match ran {
            Some(result) => result,
            // Why only plan compilation: the other sites move Host payload,
            // which a CUDA engine does not execute.
            #[cfg(feature = "cuda")]
            None if site == HostPoolSite::PlanCompile && is_cuda_engine(engine) => {
                observe(site, 1);
                let op = op.take().expect("region body runs once");
                Ok(op(Exec::serial()))
            }
            None => Err(OperationError::Dense(DenseError::Unsupported {
                op: "host_region",
                message: "Host regions run on the caller's CPU or CUDA engine".to_string(),
            })),
        }
    }
    #[cfg(not(feature = "tenferro"))]
    {
        let _ = engine;
        observe(site, 1);
        Ok(op(Exec::serial()))
    }
}

/// Whether `engine` is a CUDA engine, whose Host structural work runs on a
/// serial token. Recognition only: the context is dropped unused.
#[cfg(feature = "cuda")]
fn is_cuda_engine(engine: &mut dyn Engine) -> bool {
    symtn_dense::CudaDenseContext::new(engine).is_ok()
}

/// [`region`] for a fallible Host phase between dense calls (pack, scatter,
/// strided block moves): one token for the phase, its error flattened.
#[doc(hidden)]
pub fn with_exec<R>(
    engine: &mut dyn Engine,
    op: impl FnOnce(Exec<'_>) -> Result<R, OperationError>,
) -> Result<R, OperationError> {
    region(engine, HostPoolSite::Strided, op)?
}

/// Worker count of the caller's CPU engine (one for a serial engine or a
/// build without a CPU provider); a degree cap for scratch sizing.
pub fn threads(engine: &mut dyn Engine) -> usize {
    #[cfg(feature = "tenferro")]
    {
        tenferro_cpu::with_cpu_engine(engine, |cpu| cpu.num_threads())
            .unwrap_or(1)
            .max(1)
    }
    #[cfg(not(feature = "tenferro"))]
    {
        let _ = engine;
        1
    }
}

/// Worker count for a Host replay that writes `elements` destination
/// elements once: the caller engine's full width when the engine's own cost
/// rule admits the parallel section (threadteam's `pays` on strided's `Copy`
/// estimate for `elements`, against the engine pool's dispatch cost), and 1
/// otherwise. A serial engine never forks; the caller sets parallelism only
/// through its engine (pool width and dispatch costs). A broken or non-CPU
/// engine (CUDA included: the replay moves Host payload) is a typed error.
pub fn copy_width(engine: &mut dyn Engine, elements: usize) -> Result<usize, OperationError> {
    #[cfg(feature = "tenferro")]
    {
        let estimate = strided_kernel::execution::KernelClass::Copy.estimate(elements);
        let width = tenferro_cpu::with_cpu_engine(engine, |cpu| {
            let exec = cpu.exec().map_err(|err| {
                OperationError::Dense(DenseError::Backend {
                    backend: symtn_dense::DenseBackend::Tenferro,
                    op: "host_region",
                    message: err.to_string(),
                })
            })?;
            let exec = exec.into_token();
            let budget = exec.budget();
            Ok(if exec.pays(estimate, 1, budget) {
                budget.get()
            } else {
                1
            })
        });
        match width {
            Some(width) => width,
            None => Err(OperationError::Dense(DenseError::Unsupported {
                op: "host_region",
                message: "Host regions run on the caller's CPU or CUDA engine".to_string(),
            })),
        }
    }
    #[cfg(not(feature = "tenferro"))]
    {
        let _ = (engine, elements);
        Ok(1)
    }
}

/// Runs `jobs` jobs over `exec` (see `Exec::fork`); job `i` gets a child
/// token.
pub fn fork(exec: Exec<'_>, jobs: usize, f: impl Fn(usize, Exec<'_>) + Sync) {
    exec.fork(jobs, f);
}

/// Runs `left` and `right` as two jobs of `exec`, each with its half of the
/// budget, and returns both results.
pub fn join<A, B, RA, RB>(exec: Exec<'_>, left: A, right: B) -> (RA, RB)
where
    A: FnOnce(Exec<'_>) -> RA + Send + Sync,
    B: FnOnce(Exec<'_>) -> RB + Send + Sync,
    RA: Send + Sync,
    RB: Send + Sync,
{
    // Why the slots: `fork` takes one `Fn` for every job, and each side is a
    // `FnOnce` that owns its half of the split buffers. Why `RwLock`, not
    // `Mutex`: each slot is touched by one job only, and std's `Mutex` boxes
    // an OS lock on first use on some targets (macOS), which would put an
    // allocation on every replay fork; `RwLock` does not allocate.
    let left = RwLock::new(Some(left));
    let right = RwLock::new(Some(right));
    let left_out = RwLock::new(None);
    let right_out = RwLock::new(None);
    fork(exec, 2, |job, child| {
        if job == 0 {
            let run = take(&left);
            *write(&left_out) = Some(run(child));
        } else {
            let run = take(&right);
            *write(&right_out) = Some(run(child));
        }
    });
    (take(&left_out), take(&right_out))
}

fn write<T>(slot: &RwLock<T>) -> std::sync::RwLockWriteGuard<'_, T> {
    slot.write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn take<T>(slot: &RwLock<Option<T>>) -> T {
    // INVARIANT: `fork` runs each of the two jobs exactly once before
    // returning (a panicking job unwinds through `fork`), so every slot is
    // filled once and taken once.
    write(slot).take().expect("join slot filled exactly once")
}

/// Where a Host region observation was taken.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostPoolSite {
    Replay,
    PlanCompile,
    Strided,
}

/// One observation of the token a Host region ran with.
#[cfg(any(test, feature = "testing"))]
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostPoolObservation {
    pub site: HostPoolSite,
    /// Budget of the token the region ran with: the caller engine's width.
    pub token_budget: usize,
}

#[cfg(any(test, feature = "testing"))]
static OBSERVING: AtomicBool = AtomicBool::new(false);
#[cfg(any(test, feature = "testing"))]
static OBSERVATIONS: Mutex<Vec<HostPoolObservation>> = Mutex::new(Vec::new());

/// Test hook: starts or stops recording [`HostPoolObservation`]s. Off by
/// default; when off a site pays one relaxed atomic load. Only test builds
/// (`cfg(test)` or the `testing` feature) record; others compile the sites
/// to nothing.
#[cfg(any(test, feature = "testing"))]
#[doc(hidden)]
pub fn observe_host_pools(enabled: bool) {
    OBSERVING.store(enabled, Ordering::Relaxed);
}

/// Test hook: drains the recorded observations.
#[cfg(any(test, feature = "testing"))]
#[doc(hidden)]
pub fn take_host_pool_observations() -> Vec<HostPoolObservation> {
    std::mem::take(
        &mut *OBSERVATIONS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()),
    )
}

#[cfg(not(any(test, feature = "testing")))]
#[inline(always)]
fn observe(_site: HostPoolSite, _budget: usize) {}

#[cfg(any(test, feature = "testing"))]
fn observe(site: HostPoolSite, token_budget: usize) {
    if !OBSERVING.load(Ordering::Relaxed) {
        return;
    }
    OBSERVATIONS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .push(HostPoolObservation { site, token_budget });
}

/// Test support: a caller-owned CPU engine of `threads` workers.
#[cfg(all(any(test, feature = "testing"), feature = "tenferro"))]
#[doc(hidden)]
pub struct TestPool(tenferro_cpu::CpuEngine);

#[cfg(all(any(test, feature = "testing"), feature = "tenferro"))]
#[doc(hidden)]
pub fn test_pool(threads: usize) -> TestPool {
    TestPool(test_engine(threads))
}

/// Test support: a CPU engine of `threads` workers (serial at one).
#[cfg(all(any(test, feature = "testing"), feature = "tenferro"))]
#[doc(hidden)]
pub fn test_engine(threads: usize) -> tenferro_cpu::CpuEngine {
    if threads <= 1 {
        tenferro_cpu::CpuEngine::serial()
    } else {
        tenferro_cpu::CpuEngine::new(
            tenferro_cpu::Pool::builder()
                .threads(threads)
                .build()
                .expect("test pool"),
        )
    }
}

/// Test support: a `threads`-worker engine whose pool declares zero dispatch
/// cost, so its own cost rule admits every parallel section and small
/// fixtures take the parallel replay (serial at one).
#[cfg(all(any(test, feature = "testing"), feature = "tenferro"))]
#[doc(hidden)]
pub fn eager_test_engine(threads: usize) -> tenferro_cpu::CpuEngine {
    use tenferro_cpu::threadteam::{DispatchCosts, WidthCosts};
    if threads <= 1 {
        return tenferro_cpu::CpuEngine::serial();
    }
    let free = WidthCosts::flat(std::time::Duration::ZERO);
    tenferro_cpu::CpuEngine::new(
        tenferro_cpu::Pool::builder()
            .threads(threads)
            .dispatch_costs(DispatchCosts::new(free, free, free))
            .build()
            .expect("test pool"),
    )
}

/// Test support: [`eager_test_engine`] as a [`TestPool`].
#[cfg(all(any(test, feature = "testing"), feature = "tenferro"))]
#[doc(hidden)]
pub fn eager_test_pool(threads: usize) -> TestPool {
    TestPool(eager_test_engine(threads))
}

#[cfg(all(any(test, feature = "testing"), feature = "tenferro"))]
impl TestPool {
    /// Runs `op` with this engine.
    pub fn install<R>(&mut self, op: impl FnOnce(&mut dyn Engine) -> R) -> R {
        op(&mut self.0)
    }

    /// Runs `op` with this engine's whole token.
    pub fn with_exec<R>(&mut self, op: impl FnOnce(Exec<'_>) -> R) -> R {
        region(&mut self.0, HostPoolSite::Replay, op).expect("test engine token")
    }
}
