//! A worker pool whose threads have all started, for process-wide
//! allocation windows (`counting_alloc::measure_process`).
//!
//! Why: Rayon starts a pool's worker threads asynchronously, and each one
//! allocates its own job queues (a `crossbeam_deque` block) on its own thread
//! before it runs any job. On a loaded machine a worker can still be starting
//! when a measured call runs, so a process-wide count charges that start-up
//! to the call. The pool's start hook runs after those allocations; waiting
//! for it on every worker keeps start-up out of every later window.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use symtn_dense::Pool;

/// A pool of `threads` workers, returned once every worker has started.
pub fn started_pool(threads: usize) -> Pool {
    let started = Arc::new(AtomicUsize::new(0));
    let count = Arc::clone(&started);
    let pool = Pool::builder()
        .threads(threads)
        .on_start(move |_| {
            count.fetch_add(1, Ordering::SeqCst);
        })
        .build()
        .unwrap();
    while started.load(Ordering::SeqCst) < threads {
        std::thread::yield_now();
    }
    pool
}
