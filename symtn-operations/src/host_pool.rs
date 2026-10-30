//! The Host CPU engine that an eager operation's parallel regions run on.
//!
//! A `Runtime` owns one CPU context ([`SharedCpuContext`]): a Tenferro
//! `CpuEngine` whose `threadteam` team runs the runtime's dense kernels. An
//! eager operation enters that context on the calling thread with
//! [`enter_host_pool`]. Every parallel region below it (tree-transform replay
//! fan-out, plan compile, strided kernels) takes the engine's placement token
//! through [`region`] or [`strided`] and splits it with [`fork`] / [`join`];
//! every degree cap reads [`current_threads`]. symtn schedules no work of its
//! own: the token decides where each task runs.
//!
//! A region holds the engine for its duration, so it must not call the dense
//! backend. Inside a region (on the calling thread and on every forked task)
//! the context is marked busy: a nested region or strided call runs serially
//! within the task that reached it instead of asking the held engine again.
//!
//! With no entered context (direct callers of this crate without a `Runtime`),
//! regions run serially on the calling thread.
// ponytail: the entered context is thread-local until engine option A passes
// `&mut` engine through every operation; the token then comes from that
// argument and this lookup goes away.

use std::cell::RefCell;
use std::marker::PhantomData;
#[cfg(any(test, feature = "testing"))]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(any(test, feature = "testing"))]
use std::sync::Mutex;
use std::sync::RwLock;

use strided_kernel::execution::KernelClass;
pub use strided_kernel::Exec;
use symtn_dense::SharedCpuContext;

#[derive(Clone)]
enum Entry {
    Pool(SharedCpuContext),
    /// Inside a region that holds the entered engine's token.
    Busy,
}

thread_local! {
    static CURRENT: RefCell<Option<Entry>> = const { RefCell::new(None) };
}

/// Keeps a pool entered on the calling thread until dropped (see
/// [`enter_host_pool`]). Not `Send`: the entry is thread-local.
#[must_use = "the pool is entered only while the guard lives"]
pub struct HostPoolGuard {
    entered: usize,
    previous: Option<Entry>,
    _thread_bound: PhantomData<*const ()>,
}

impl Drop for HostPoolGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        CURRENT.with(|current| {
            let mut current = current.borrow_mut();
            // Why the identity check: a guard dropped out of nesting order must
            // not re-enter a pool that an inner guard has already left.
            if matches!(&*current, Some(Entry::Pool(pool)) if pool.identity() == self.entered) {
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
        previous: CURRENT.with(|current| current.replace(Some(Entry::Pool(pool.clone())))),
        _thread_bound: PhantomData,
    }
}

fn current_entry() -> Option<Entry> {
    CURRENT.with(|current| current.borrow().clone())
}

/// Marks the calling thread busy until dropped, restoring its entry.
struct BusyGuard(Option<Entry>);

impl BusyGuard {
    fn enter() -> Self {
        Self(CURRENT.with(|current| current.replace(Some(Entry::Busy))))
    }
}

impl Drop for BusyGuard {
    fn drop(&mut self) {
        let previous = self.0.take();
        CURRENT.with(|current| *current.borrow_mut() = previous);
    }
}

/// Worker count a parallel region may use: the entered engine's width; one
/// inside a region or with no entered engine.
pub fn current_threads() -> usize {
    match current_entry() {
        Some(Entry::Pool(pool)) => pool.num_threads().max(1),
        Some(Entry::Busy) => 1,
        None => {
            audit_unentered(HostPoolSite::Threads);
            1
        }
    }
}

/// Coverage audit (`--cfg symtn_scope_audit`, a CI row): a Host site reached
/// with no entered engine means an eager operation that neither leases nor
/// enters its runtime's engine, so its work would run serially.
#[inline(always)]
fn audit_unentered(site: HostPoolSite) {
    #[cfg(symtn_scope_audit)]
    panic!("Host {site:?} site reached with no entered runtime pool");
    #[cfg(not(symtn_scope_audit))]
    let _ = site;
}

/// Runs a parallel region with the entered engine's token. A one-thread
/// engine, a nested region, or no entered engine hands `op` a serial token.
/// `op` must not take a runtime lock or call the dense backend (see the
/// module docs).
pub fn region<R>(site: HostPoolSite, op: impl FnOnce(Exec<'_>) -> R) -> R {
    match current_entry() {
        Some(Entry::Pool(pool)) => pool.with_exec(|exec| {
            observe(site, Some(&pool), exec.budget().get());
            let _busy = BusyGuard::enter();
            op(exec)
        }),
        Some(Entry::Busy) => op(Exec::serial()),
        None => {
            audit_unentered(site);
            observe(site, None, 1);
            op(Exec::serial())
        }
    }
}

/// Runs one strided-kernel call of `len` copied elements. When no parallel
/// section over `len` copies can pay on the entered engine, strided would run
/// serially whatever token it got, so it gets a serial one without taking the
/// engine; above that the call is a parallel region.
pub fn strided<R>(len: usize, op: impl FnOnce(Exec<'_>) -> R) -> R {
    match current_entry() {
        Some(Entry::Pool(pool)) if pool.section_pays(KernelClass::Copy.estimate(len)) => {
            region(HostPoolSite::Strided, op)
        }
        Some(_) => op(Exec::serial()),
        None => {
            audit_unentered(HostPoolSite::Strided);
            op(Exec::serial())
        }
    }
}

/// Runs `jobs` jobs over `exec` (see `Exec::fork`); job `i` gets a child
/// token. Each job runs marked busy, so strided calls inside it use their own
/// child token, never the held engine.
pub fn fork(exec: Exec<'_>, jobs: usize, f: impl Fn(usize, Exec<'_>) + Sync) {
    exec.fork(jobs, |job, child| {
        let _busy = BusyGuard::enter();
        f(job, child);
    });
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

#[cfg(any(test, feature = "testing"))]
fn lock<T>(slot: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
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

/// One observation of the engine a Host parallel site ran with.
#[cfg(any(test, feature = "testing"))]
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostPoolObservation {
    pub site: HostPoolSite,
    /// [`SharedCpuContext::identity`] of the entered engine, if any.
    pub entered: Option<usize>,
    /// [`SharedCpuContext::num_threads`] of the entered engine: the width its
    /// team runs parallel work at (one when it has no team).
    pub entered_threads: Option<usize>,
    /// Budget of the placement token the region ran with: the entered
    /// engine's token for a region, one for a serial region; `None` for a
    /// dense stage, which takes its token inside the dense backend.
    pub token_budget: Option<usize>,
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
    std::mem::take(&mut *lock(&OBSERVATIONS))
}

/// Records where a dense linear-algebra stage runs: on the calling thread,
/// which holds the entered runtime's engine; the engine's own team does the
/// parallel work. A no-op outside test builds; it stays callable there
/// because production dense stages call it.
#[doc(hidden)]
#[inline]
pub fn observe_dense_site() {
    #[cfg(any(test, feature = "testing"))]
    if OBSERVING.load(Ordering::Relaxed) {
        let pool = match current_entry() {
            Some(Entry::Pool(pool)) => Some(pool),
            _ => None,
        };
        record(HostPoolObservation {
            site: HostPoolSite::Dense,
            entered: pool.as_ref().map(SharedCpuContext::identity),
            entered_threads: pool.as_ref().map(SharedCpuContext::num_threads),
            token_budget: None,
        });
    }
}

#[cfg(not(any(test, feature = "testing")))]
#[inline(always)]
fn observe(_site: HostPoolSite, _pool: Option<&SharedCpuContext>, _budget: usize) {}

#[cfg(any(test, feature = "testing"))]
fn observe(site: HostPoolSite, pool: Option<&SharedCpuContext>, budget: usize) {
    if !OBSERVING.load(Ordering::Relaxed) {
        return;
    }
    record(HostPoolObservation {
        site,
        entered: pool.map(SharedCpuContext::identity),
        entered_threads: pool.map(SharedCpuContext::num_threads),
        token_budget: Some(budget),
    });
}

#[cfg(any(test, feature = "testing"))]
fn record(observation: HostPoolObservation) {
    lock(&OBSERVATIONS).push(observation);
}

/// Test support: an engine of `threads` workers that [`TestPool::install`]
/// enters on the calling thread, in place of a runtime.
#[cfg(any(test, feature = "testing"))]
#[doc(hidden)]
pub struct TestPool(SharedCpuContext);

#[cfg(any(test, feature = "testing"))]
#[doc(hidden)]
pub fn test_pool(threads: usize) -> TestPool {
    TestPool(SharedCpuContext::with_threads(threads).expect("test engine"))
}

#[cfg(any(test, feature = "testing"))]
impl TestPool {
    /// Runs `op` with this engine entered as the calling thread's Host pool.
    pub fn install<R>(&self, op: impl FnOnce() -> R) -> R {
        let _entered = enter_host_pool(&self.0);
        op()
    }

    pub fn context(&self) -> &SharedCpuContext {
        &self.0
    }
}
