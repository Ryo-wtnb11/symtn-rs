//! A worker pool every thread of which has run one job, for process-wide
//! allocation windows (`counting_alloc::measure_process`).
//!
//! Why: Rayon starts a pool's worker threads asynchronously, and each worker
//! allocates on its own thread before and while it first enters its work
//! loop: its job queues (a `crossbeam_deque` block) at start-up, then its
//! Crossbeam epoch registration on the first steal attempt. A worker the
//! measured calls never need can reach those points late, during a window,
//! and a process-wide count charges them to the call. A broadcast on a plain
//! pool token runs one job on every worker thread and returns only after
//! all of them did, so each worker has passed both first uses before the pool
//! is handed out. Later lazy initialization a measured call itself triggers
//! (for example a provider's first use of a worker) is not excluded.

use symtn_dense::Pool;
use tenferro_cpu::threadteam;

/// A pool of `threads` workers, returned once each has run a job.
pub fn started_pool(threads: usize) -> Pool {
    let pool = Pool::builder().threads(threads).build().unwrap();
    // A plain token, not a team: a team's broadcast reaches only its
    // reserved members, while this one reaches every pool worker.
    threadteam::Exec::new(&pool, threads)
        .unwrap()
        .broadcast(|_| {})
        .unwrap();
    pool
}
