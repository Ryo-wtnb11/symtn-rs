//! The Host CPU pool that an eager operation's parallel regions run on.
//!
//! A `Runtime` owns one CPU context ([`SharedCpuContext`]): a Tenferro
//! `CpuEngine` whose `threadteam` pool runs the runtime's dense kernels. An eager operation enters that context on
//! the calling thread with `with_host_pool`. Every parallel region below it
//! (tree-transform replay fan-out, plan compile, strided kernels) then runs
//! inside the same pool through [`install_region`] or [`strided`], and every
//! degree cap reads [`current_threads`].
//!
//! Why the regions are installed, not the whole operation: a Rayon worker that
//! waits in `join` or in a nested `install` runs other queued jobs of its
//! pool. A whole operation picked up that way could enter Tenferro's backend
//! while another operation's dense call owns the same pool (Tenferro panics on
//! that re-entry), or block on a runtime lock held lower on the same stack.
//! Regions hold no lock and never call the dense backend, so running one of
//! them on a waiting worker is harmless.
//!
//! With no entered pool (direct callers of this crate without a `Runtime`),
//! regions use the ambient Rayon pool as before.

use std::cell::RefCell;
use std::marker::PhantomData;
#[cfg(any(test, feature = "testing"))]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(any(test, feature = "testing"))]
use std::sync::Mutex;

use strided_kernel::{with_execution_policy, ExecutionPolicy};
use symtn_dense::SharedCpuContext;

thread_local! {
    static CURRENT: RefCell<Option<SharedCpuContext>> = const { RefCell::new(None) };
}

/// Keeps a pool entered on the calling thread until dropped (see
/// [`enter_host_pool`]). Not `Send`: the entry is thread-local.
#[must_use = "the pool is entered only while the guard lives"]
pub struct HostPoolGuard {
    entered: usize,
    previous: Option<SharedCpuContext>,
    _thread_bound: PhantomData<*const ()>,
}

impl Drop for HostPoolGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        CURRENT.with(|current| {
            let mut current = current.borrow_mut();
            // Why the identity check: a guard dropped out of nesting order must
            // not re-enter a pool that an inner guard has already left.
            if current.as_ref().map(SharedCpuContext::identity) == Some(self.entered) {
                *current = previous;
            }
        });
    }
}

/// Enters `pool` as the calling thread's Host pool until the guard drops.
/// Entries nest. Used by owners whose lifetime is lexical but not a closure,
/// such as a runtime's execution leases.
pub fn enter_host_pool(pool: &SharedCpuContext) -> HostPoolGuard {
    HostPoolGuard {
        entered: pool.identity(),
        previous: CURRENT.with(|current| current.replace(Some(pool.clone()))),
        _thread_bound: PhantomData,
    }
}

/// Runs `op` with `pool` as the calling thread's Host pool.
pub(crate) fn with_host_pool<R>(pool: &SharedCpuContext, op: impl FnOnce() -> R) -> R {
    let _entered = enter_host_pool(pool);
    op()
}

fn current_pool() -> Option<SharedCpuContext> {
    CURRENT.with(|current| current.borrow().clone())
}

/// Worker count a parallel region may use: the entered pool's size, or the
/// ambient Rayon pool's when no pool is entered.
pub fn current_threads() -> usize {
    CURRENT
        .with(|current| current.borrow().as_ref().map(SharedCpuContext::num_threads))
        .unwrap_or_else(|| {
            audit_unentered(HostPoolSite::Threads);
            rayon::current_num_threads()
        })
        .max(1)
}

/// Coverage audit (`--cfg symtn_scope_audit`, a CI row): a Host site reached
/// on a caller thread with no entered pool means an eager operation that
/// neither leases nor enters its runtime's pool, so its work would run on the
/// ambient (global) pool. On a Rayon worker the ambient pool is the one being
/// worked in, which is correct.
#[inline(always)]
fn audit_unentered(site: HostPoolSite) {
    #[cfg(symtn_scope_audit)]
    if rayon::current_thread_index().is_none() {
        panic!("Host {site:?} site reached with no entered runtime pool");
    }
    let _ = site;
}

/// Runs a parallel region inside the entered pool. A one-thread pool runs it
/// on the calling thread with strided fan-out disabled. `op` must not take a
/// runtime lock or call the dense backend (see the module docs).
pub fn install_region<R: Send>(site: HostPoolSite, op: impl FnOnce() -> R + Send) -> R {
    match current_pool() {
        None => {
            audit_unentered(site);
            observe(site, None);
            op()
        }
        Some(pool) if pool.num_threads() == 1 => {
            with_execution_policy(ExecutionPolicy::Sequential, || {
                observe(site, Some(&pool));
                op()
            })
        }
        // Already on a pool worker (a nested region): `install` holds the
        // engine lock for the outer region, so re-entering it would deadlock;
        // the nested Rayon work stays on the current pool anyway.
        // ponytail: any Rayon worker counts, so a caller inside a foreign Rayon
        // pool runs the region there; check the engine pool identity once
        // threadteam exposes it.
        Some(pool) if rayon::current_thread_index().is_some() => {
            observe(site, Some(&pool));
            op()
        }
        Some(pool) => pool.install(|| {
            with_host_pool(&pool, || {
                observe(site, Some(&pool));
                op()
            })
        }),
    }
}

/// Runs one strided-kernel call of `len` elements. At or below strided's own
/// threading gate the kernel is serial whichever pool is current, so it runs
/// in place without an install; above it the call is a parallel region.
pub fn strided<R: Send>(len: usize, op: impl FnOnce() -> R + Send) -> R {
    if len <= strided_kernel::execution::MINTHREADLENGTH {
        return match CURRENT.with(|current| current.borrow().is_some()) {
            // Why Sequential rather than ambient: a strided entry may query the
            // ambient pool before its own size gate, which would initialize
            // Rayon's global pool from inside a runtime operation.
            true => with_execution_policy(ExecutionPolicy::Sequential, op),
            false => {
                audit_unentered(HostPoolSite::Strided);
                op()
            }
        };
    }
    install_region(HostPoolSite::Strided, op)
}

/// Where a Host pool observation was taken.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostPoolSite {
    /// A degree cap read (`current_threads`); audit only, never observed.
    Threads,
    Replay,
    PlanCompile,
    Strided,
    Dense,
}

/// One observation of the pool a Host parallel site ran in.
#[cfg(any(test, feature = "testing"))]
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostPoolObservation {
    pub site: HostPoolSite,
    /// [`SharedCpuContext::identity`] of the entered pool, if any.
    pub entered: Option<usize>,
    /// [`SharedCpuContext::num_threads`] of the entered pool: the width its
    /// engine runs parallel work at (one when it has no team).
    pub entered_threads: Option<usize>,
    /// `rayon::current_num_threads()` when the site ran on a Rayon worker;
    /// `None` when it ran on the calling (non-pool) thread.
    pub worker_pool_threads: Option<usize>,
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
    std::mem::take(&mut *OBSERVATIONS.lock().unwrap_or_else(|p| p.into_inner()))
}

/// Records where a dense linear-algebra stage runs: on the calling thread,
/// which holds the entered runtime's engine; the engine's own team does the
/// parallel work. A no-op outside test builds; it
/// stays callable there because production dense stages call it.
#[doc(hidden)]
#[inline]
pub fn observe_dense_site() {
    #[cfg(any(test, feature = "testing"))]
    observe(HostPoolSite::Dense, current_pool().as_ref());
}

#[cfg(not(any(test, feature = "testing")))]
#[inline(always)]
fn observe(_site: HostPoolSite, _pool: Option<&SharedCpuContext>) {}

#[cfg(any(test, feature = "testing"))]
fn observe(site: HostPoolSite, pool: Option<&SharedCpuContext>) {
    if !OBSERVING.load(Ordering::Relaxed) {
        return;
    }
    let observation = HostPoolObservation {
        site,
        entered: pool.map(SharedCpuContext::identity),
        entered_threads: pool.map(SharedCpuContext::num_threads),
        worker_pool_threads: rayon::current_thread_index().map(|_| rayon::current_num_threads()),
    };
    OBSERVATIONS
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .push(observation);
}
