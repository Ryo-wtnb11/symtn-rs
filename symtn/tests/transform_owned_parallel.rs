//! Owned transforms on a multi-threaded recoupling backend (#1217): the
//! uninitialised owned writer runs under the parallel schedule, so the result
//! is bit-identical to the serial owned result and the caller thread makes the
//! same allocations as the serial path, with no allocator-zeroed payload (no
//! `vec![zero; P]` fallback).

use std::hint::black_box;
use std::sync::Arc;

use num_complex::Complex64;
use symtn::sector::{product_sector, SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use symtn::typed::Runtime;
use symtn::typed::{GradedSpace, TensorMap};

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

/// Re-executes exactly one test, alone, in a child process. `counting_alloc::serial()`
/// only serializes this file's own three tests against each other; it does
/// not stop a preceding test's dropped `Runtime`/pool from leaving
/// process-global structural-cache work (symtn-core's fusion-tree-layout and
/// complete-HomSpace caches) still in flight when the lock is re-acquired.
/// `parallel_owned_permute_allocates_like_the_serial_owned_path` shares its
/// `u1_su2_space` fixture (same sectors and degeneracies) with
/// `u1_su2_owned_transforms_are_bit_identical_across_thread_counts`, so a
/// rebuild landing in that window changes this thread's allocation count (8
/// against 6, reported during #1570 verification even with the lock held).
/// Same technique as symtn-tensors #649/#650 and symtn-core/symtn-tensors'
/// #1598/#1606 fix.
///
/// Call at the top of the `#[test]` fn with a name-unique env var and the
/// test's libtest path; when it returns `true`, the child already ran the
/// real body and the caller must return immediately.
fn run_isolated_or_return(isolated_env: &str, test_path: &str) -> bool {
    if std::env::var_os(isolated_env).is_some() {
        return false;
    }
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test_path, "--include-ignored"])
        .env(isolated_env, "1")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success() && stdout.contains("test result: ok. 1 passed; 0 failed;"),
        "isolated test did not execute exactly once: {}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status
    );
    true
}

/// (allocations, bytes, zeroed allocations) on the caller thread.
fn measure<T>(f: impl FnOnce() -> T) -> (T, usize, usize, usize) {
    let (output, allocs) = counting_alloc::measure(f);
    (
        output,
        allocs.calls as usize,
        allocs.bytes as usize,
        allocs.zeroed_calls as usize,
    )
}

const THREADS: usize = 3;

/// The caller's pool. The parallel runtime replays on its own pool, pinned
/// to `THREADS` workers in [`runtimes`], so the effective worker count does not
/// depend on the host's core count or Rayon's global pool.
fn pool() -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(THREADS)
        .build()
        .unwrap()
}

fn runtimes() -> (Runtime, Runtime) {
    (
        Runtime::builder().recoupling_threads(1).build().unwrap(),
        Runtime::builder()
            .dense_threads(THREADS)
            .recoupling_threads(THREADS)
            .build()
            .unwrap(),
    )
}

type U1Su2 = symtn::sector::ProductFusionRule<U1FusionRule, SU2FusionRule>;

/// U(1) x SU(2): every non-trivial permutation recouples (Multi scatter
/// groups) and the degeneracies put the payload past the backend's parallel
/// size gate.
fn u1_su2_space(provider: &Arc<U1Su2>) -> GradedSpace<U1Su2> {
    GradedSpace::try_new(
        Arc::clone(provider),
        [
            (
                product_sector(U1Irrep::new(0), SU2Irrep::from_twice_spin(0)),
                18,
            ),
            (
                product_sector(U1Irrep::new(1), SU2Irrep::from_twice_spin(1)),
                14,
            ),
            (
                product_sector(U1Irrep::new(-1), SU2Irrep::from_twice_spin(1)),
                14,
            ),
            (
                product_sector(U1Irrep::new(0), SU2Irrep::from_twice_spin(2)),
                12,
            ),
        ],
    )
    .unwrap()
}

fn u1_space(provider: &Arc<U1FusionRule>) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::clone(provider),
        [
            (U1Irrep::new(-1), 36),
            (U1Irrep::new(0), 48),
            (U1Irrep::new(1), 36),
        ],
    )
    .unwrap()
}

macro_rules! transforms {
    ($source:expr) => {
        vec![
            ("permute", $source.permute(&[1], &[2, 0]).unwrap()),
            ("braid", $source.braid(&[1, 0], &[2], &[0, 2, 1]).unwrap()),
            ("repartition", $source.repartition(1).unwrap()),
            ("full transpose", $source.transpose(&[2], &[1, 0]).unwrap()),
            ("transpose", $source.transpose(&[1, 2], &[0]).unwrap()),
        ]
    };
}

macro_rules! assert_bit_identical {
    ($expected:expr, $actual:expr) => {
        for ((name, expected), (_, actual)) in $expected.iter().zip(&$actual) {
            assert_eq!(
                expected.dense_data().unwrap().len(),
                actual.dense_data().unwrap().len(),
                "{name}"
            );
            assert_eq!(expected.subblock_count(), actual.subblock_count(), "{name}");
            assert!(
                expected.dense_data().unwrap() == actual.dense_data().unwrap(),
                "{name}: payloads differ"
            );
        }
    };
}

#[test]
fn u1_su2_owned_transforms_are_bit_identical_across_thread_counts() {
    let _measurement = counting_alloc::serial();
    let (serial, parallel) = runtimes();
    let provider = Arc::new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(U1FusionRule, SU2FusionRule).unwrap(),
    );
    let space = u1_su2_space(&provider);
    let real_serial: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&serial, [&space, &space], [&space], 1217).unwrap();
    let real_parallel: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&parallel, [&space, &space], [&space], 1217).unwrap();
    assert!(real_serial.dense_data().unwrap() == real_parallel.dense_data().unwrap());
    assert!(
        real_serial.dense_data().unwrap().len() > 1 << 15,
        "fixture must exceed the backend's parallel size gate: {}",
        real_serial.dense_data().unwrap().len()
    );
    let complex_serial: TensorMap<_, Complex64> =
        TensorMap::rand_with_seed(&serial, [&space, &space], [&space], 1218).unwrap();
    let complex_parallel: TensorMap<_, Complex64> =
        TensorMap::rand_with_seed(&parallel, [&space, &space], [&space], 1218).unwrap();
    assert!(complex_serial.dense_data().unwrap() == complex_parallel.dense_data().unwrap());

    let expected_real = transforms!(real_serial);
    let expected_complex = transforms!(complex_serial);
    let (actual_real, actual_complex) =
        pool().install(|| (transforms!(real_parallel), transforms!(complex_parallel)));
    assert_bit_identical!(expected_real, actual_real);
    assert_bit_identical!(expected_complex, actual_complex);
}

#[test]
fn u1_owned_transforms_are_bit_identical_across_thread_counts() {
    let _measurement = counting_alloc::serial();
    let (serial, parallel) = runtimes();
    let provider = Arc::new(U1FusionRule);
    let space = u1_space(&provider);
    let real_serial: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&serial, [&space, &space], [&space], 1219).unwrap();
    let real_parallel: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&parallel, [&space, &space], [&space], 1219).unwrap();
    assert!(real_serial.dense_data().unwrap() == real_parallel.dense_data().unwrap());
    assert!(
        real_serial.dense_data().unwrap().len() > 1 << 15,
        "fixture must exceed the backend's parallel size gate: {}",
        real_serial.dense_data().unwrap().len()
    );
    let complex_serial: TensorMap<_, Complex64> =
        TensorMap::rand_with_seed(&serial, [&space, &space], [&space], 1220).unwrap();
    let complex_parallel: TensorMap<_, Complex64> =
        TensorMap::rand_with_seed(&parallel, [&space, &space], [&space], 1220).unwrap();

    let expected_real = transforms!(real_serial);
    let expected_complex = transforms!(complex_serial);
    let (actual_real, actual_complex) =
        pool().install(|| (transforms!(real_parallel), transforms!(complex_parallel)));
    assert_bit_identical!(expected_real, actual_real);
    assert_bit_identical!(expected_complex, actual_complex);
}

#[test]
fn parallel_owned_permute_allocates_like_the_serial_owned_path() {
    if run_isolated_or_return(
        "SYMTN_PARALLEL_OWNED_PERMUTE_ALLOCATES_LIKE_SERIAL_ISOLATED",
        "parallel_owned_permute_allocates_like_the_serial_owned_path",
    ) {
        return;
    }
    let _measurement = counting_alloc::serial();
    // What: with `recoupling_threads > 1` a warmed owned permute makes the
    // same caller-thread allocations as the serial owned path and none of them
    // is allocator-zeroed: the multi-threaded backend writes the owned output
    // directly, with no `vec![0.0; P]` fallback and destination-reading
    // replay (#1217).
    let (serial, parallel) = runtimes();
    let provider = Arc::new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(U1FusionRule, SU2FusionRule).unwrap(),
    );
    let space = u1_su2_space(&provider);
    let source_serial: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&serial, [&space, &space], [&space], 1221).unwrap();
    let source_parallel: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&parallel, [&space, &space], [&space], 1221).unwrap();
    let payload_bytes = size_of_val(source_serial.dense_data().unwrap());
    assert!(
        source_serial.dense_data().unwrap().len() > 1 << 15,
        "fixture must exceed the backend's parallel size gate: {}",
        source_serial.dense_data().unwrap().len()
    );

    let warm_serial = source_serial.permute(&[1], &[2, 0]).unwrap();
    let (result_serial, serial_allocations, serial_bytes, serial_zeroed) =
        measure(|| source_serial.permute(&[1], &[2, 0]).unwrap());
    black_box(result_serial.dense_data().unwrap());

    let pool = pool();
    let warm_parallel = pool.install(|| source_parallel.permute(&[1], &[2, 0]).unwrap());
    assert!(warm_serial.dense_data().unwrap() == warm_parallel.dense_data().unwrap());
    // This worker waits in a cross-pool install into the runtime's pool. Under
    // load a wait can be the first to box a lazily created OS lock (macOS std
    // allocates each Mutex/Condvar on first use, once per process) inside the
    // window (#1964): this thread's own sleep mutex and condvar, and, through
    // Rayon's wake of the runtime pool, each of its `THREADS` workers' sleep
    // mutexes. With at most `THREADS + 2` such allocations, `THREADS + 3`
    // windows on the same worker include a clean one, and the cleanest is the
    // steady-state count; a per-call allocation would appear in all of them.
    // Why not more warm-up: nothing in Rayon's public API forces a sleep.
    let (result_parallel, parallel_allocations, parallel_bytes, parallel_zeroed) =
        pool.install(|| {
            (0..THREADS + 3)
                .map(|_| measure(|| source_parallel.permute(&[1], &[2, 0]).unwrap()))
                .min_by_key(|&(_, calls, bytes, _)| (calls, bytes))
                .unwrap()
        });
    black_box(result_parallel.dense_data().unwrap());

    assert!(result_serial.dense_data().unwrap() == result_parallel.dense_data().unwrap());
    assert!(
        serial_bytes >= payload_bytes && parallel_bytes >= payload_bytes,
        "owned outputs allocate at least the payload: {serial_bytes} / {parallel_bytes}"
    );
    assert_eq!(
        (parallel_allocations, parallel_bytes, parallel_zeroed),
        (serial_allocations, serial_bytes, 0),
        "serial owned path: {serial_allocations} allocations, {serial_bytes} bytes, {serial_zeroed} zeroed"
    );
}
