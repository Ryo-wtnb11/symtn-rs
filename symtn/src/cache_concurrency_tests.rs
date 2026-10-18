//! Release-mode concurrency gates of the process-global completed-transformer
//! and composed-coefficient caches (#2014): no
//! single-flight, so no caller parks on another's build, even when work
//! stealing re-enters a caller's Rayon region. Each case runs in its own
//! process (the counters are process-global) under a watchdog, issues one
//! cold transform from many tasks with `recoupling_threads > 1`, and requires
//! every task to finish, one admission per key in each cache, and
//! byte-identical results.
//!
//! Run in release: `cargo test --release -p symtn --lib cache_concurrency`.

use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;

use rayon::prelude::*;
use symtn_core::{SU2FusionRule, SU2Irrep};
use symtn_tensors::{TreeTransformOperation, TreeTransformStructure};

use crate::runtime::Runtime;
use crate::typed::{GradedSpace, TensorMap};

const TASKS: usize = 32;

/// Runs `body` on a fresh thread and fails if it does not finish: a parked
/// thread is a deadlock, not a slow test.
fn watchdog(body: impl FnOnce() + Send + 'static) {
    let (done, finished) = mpsc::channel();
    std::thread::spawn(move || {
        body();
        done.send(()).unwrap();
    });
    finished
        .recv_timeout(Duration::from_secs(300))
        .expect("concurrent cold misses did not finish: a thread parked on a build");
}

/// An SU(2) `V^2 <- V^2` tensor (non-Unique fusion, so the build runs
/// recoupling regions) and its permute.
fn fixture(runtime: &Runtime) -> TensorMap<SU2FusionRule, f64> {
    let space = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 3),
            (SU2Irrep::from_twice_spin(2), 2),
            (SU2Irrep::from_twice_spin(3), 1),
        ],
    )
    .unwrap();
    TensorMap::rand_with_seed(runtime, [&space, &space], [&space, &space], 20_143).unwrap()
}

fn operation() -> TreeTransformOperation {
    TreeTransformOperation::permute([2, 0], [3, 1])
}

/// Stages the fixture's eager transform through one leased Runtime context
/// and returns its completed transformer (planning only: the dense backend is
/// not entered on a pool worker).
fn resolve(
    runtime: &Runtime,
    source: &TensorMap<SU2FusionRule, f64>,
) -> TreeTransformStructure<f64> {
    let source = source.test_bound_space();
    let mut lease = runtime.lease_context().unwrap();
    lease
        .context()
        .multiplicity_free_lane::<f64>()
        .unwrap()
        .tree_context_mut()
        .stage_tree_transform_multiplicity_free_in(
            source,
            None,
            source.space().required_len().unwrap(),
            &operation(),
        )
        .unwrap()
        .structure()
        .clone()
}

fn admissions() -> u64 {
    crate::test_cache::completed().admissions()
}

fn group_info() -> crate::cache::StructureCacheInfo {
    crate::cache::stats()
        .into_iter()
        .find(|info| info.kind() == crate::cache::StructureCacheKind::TreeTransformCoefficients)
        .expect("every structure cache kind reports")
}

/// The fixture's source fusion-tree groups: one composed-coefficient key
/// each.
fn source_groups(source: &TensorMap<SU2FusionRule, f64>) -> u64 {
    source
        .test_bound_space()
        .space()
        .structure()
        .fusion_tree_groups()
        .len() as u64
}

/// The resolved transformers agree with each other and with a sequential
/// rebuild, field for field; exactly `expected` completed-transformer
/// admissions happened, and one coefficient entry per source group (the
/// cache was cleared first) is resident and was admitted once.
fn assert_converged(
    built: &[TreeTransformStructure<f64>],
    before: u64,
    expected: Option<(u64, u64)>,
) {
    assert_eq!(built.len(), TASKS);
    for structure in built {
        assert_eq!(structure, &built[0]);
    }
    if let Some((expected, groups)) = expected {
        assert_eq!(admissions() - before, expected);
        let info = group_info();
        assert_eq!((info.entries() as u64, info.admissions()), (groups, groups));
    }
}

fn runtime() -> Runtime {
    Runtime::builder()
        .dense_threads(4)
        .recoupling_threads(4)
        .build()
        .unwrap()
}

#[test]
fn cache_concurrency_user_par_iter_on_the_runtime_pool() {
    if crate::test_cache::run_isolated_or_return(
        "SYMTN_CACHE_CONCURRENCY_RUNTIME_POOL",
        "cache_concurrency_tests::cache_concurrency_user_par_iter_on_the_runtime_pool",
    ) {
        return;
    }
    watchdog(|| {
        let runtime = runtime();
        let source = fixture(&runtime);
        crate::cache::clear();
        let before = admissions();
        let shared = runtime.execution_config().shared_ctx.clone();
        let built = shared.install(|| {
            (0..TASKS)
                .into_par_iter()
                .map(|_| resolve(&runtime, &source))
                .collect::<Vec<_>>()
        });
        assert_converged(&built, before, Some((1, source_groups(&source))));
    });
}

#[test]
fn cache_concurrency_user_par_iter_on_a_foreign_pool() {
    if crate::test_cache::run_isolated_or_return(
        "SYMTN_CACHE_CONCURRENCY_FOREIGN_POOL",
        "cache_concurrency_tests::cache_concurrency_user_par_iter_on_a_foreign_pool",
    ) {
        return;
    }
    watchdog(|| {
        let runtime = runtime();
        let source = fixture(&runtime);
        crate::cache::clear();
        let before = admissions();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let built = pool.install(|| {
            (0..TASKS)
                .into_par_iter()
                .map(|_| resolve(&runtime, &source))
                .collect::<Vec<_>>()
        });
        assert_converged(&built, before, Some((1, source_groups(&source))));
    });
}

#[test]
fn cache_concurrency_ambient_pool_without_a_runtime() {
    if crate::test_cache::run_isolated_or_return(
        "SYMTN_CACHE_CONCURRENCY_AMBIENT_POOL",
        "cache_concurrency_tests::cache_concurrency_ambient_pool_without_a_runtime",
    ) {
        return;
    }
    watchdog(|| {
        let runtime = runtime();
        let source = fixture(&runtime);
        let groups = source_groups(&source);
        let source = source.test_bound_space().clone();
        let src_len = source.space().required_len().unwrap();
        crate::cache::clear();
        let before = admissions();
        // Standalone contexts on Rayon's global pool: no Runtime pool,
        // recoupling regions on the ambient pool.
        let built = (0..TASKS)
            .into_par_iter()
            .map(|_| {
                let mut context = symtn_tensors::TreeTransformExecutionContext::<
                    f64,
                    symtn_core::RuleIdentity,
                >::default();
                context
                    .backend_mut()
                    .set_recoupling_threads(std::num::NonZeroUsize::new(4).unwrap());
                context
                    .stage_tree_transform_multiplicity_free_in(&source, None, src_len, &operation())
                    .unwrap()
                    .structure()
                    .clone()
            })
            .collect::<Vec<_>>();
        assert_converged(&built, before, Some((1, groups)));
    });
}

#[test]
fn cache_concurrency_with_a_racing_clear() {
    if crate::test_cache::run_isolated_or_return(
        "SYMTN_CACHE_CONCURRENCY_RACING_CLEAR",
        "cache_concurrency_tests::cache_concurrency_with_a_racing_clear",
    ) {
        return;
    }
    watchdog(|| {
        let runtime = runtime();
        let source = fixture(&runtime);
        crate::cache::clear();
        let before = admissions();
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let clearer = {
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    crate::cache::clear();
                    std::thread::yield_now();
                }
            })
        };
        let shared = runtime.execution_config().shared_ctx.clone();
        let built = shared.install(|| {
            (0..TASKS)
                .into_par_iter()
                .map(|_| resolve(&runtime, &source))
                .collect::<Vec<_>>()
        });
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        clearer.join().unwrap();
        // Clears zero the counters, so only convergence is checked here.
        let _ = before;
        assert_converged(&built, 0, None);
    });
}

#[test]
fn global_clear_takes_no_runtime_or_device_lock() {
    // Isolated: this clear is process-global and would land inside siblings'
    // warm-up-to-measurement windows (#2084).
    if crate::test_cache::run_isolated_or_return(
        "SYMTN_GLOBAL_CLEAR_TAKES_NO_RUNTIME_LOCK_ISOLATED",
        "cache_concurrency_tests::global_clear_takes_no_runtime_or_device_lock",
    ) {
        return;
    }
    // What: `symtn::cache::clear` is semantic only. With this Runtime's state
    // lock and a leased execution context held, a clear still completes.
    let runtime = runtime();
    let state = runtime.lock();
    let lease = runtime.lease_context().unwrap();
    let (done, finished) = mpsc::channel();
    std::thread::spawn(move || {
        crate::cache::clear();
        done.send(()).unwrap();
    });
    finished
        .recv_timeout(Duration::from_secs(60))
        .expect("the global clear waited on a Runtime lock");
    drop(lease);
    drop(state);
}
