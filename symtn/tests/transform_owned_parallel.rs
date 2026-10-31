//! Owned transforms on a multi-threaded engine (#1217): the uninitialised
//! owned writer runs under the parallel schedule, so the result is
//! bit-identical to the serial owned result and the caller thread makes the
//! same allocations as the serial path, with no allocator-zeroed payload (no
//! `vec![zero; P]` fallback).

use std::hint::black_box;
use std::sync::Arc;

use num_complex::Complex64;
use symtn::sector::{product_sector, SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use symtn::typed::{CpuEngine, Runtime};
use symtn::typed::{GradedSpace, TensorMap};

use crate::counting_alloc;

/// Re-executes exactly one test, alone, in a child process. `counting_alloc::serial()`
/// serializes the allocation target's test bodies against each other; it does
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
        .args([
            "--exact",
            &format!("transform_owned_parallel::{test_path}"),
            "--include-ignored",
        ])
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

/// The parallel engine: `THREADS` workers, independent of the host's core
/// count, whose pool declares zero dispatch cost so every parallel section
/// forks on every host. Why not the host-default costs: they come from the
/// detected host class, so a section forked on one CI runner stayed serial
/// on another and the parallel path went unmeasured there.
fn parallel_engine() -> CpuEngine {
    symtn_operations::host_pool::eager_test_engine(THREADS)
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
    ($engine:ident, $source:expr) => {
        vec![
            ("permute", $source.permute($engine, &[1], &[2, 0]).unwrap()),
            (
                "braid",
                $source.braid($engine, &[1, 0], &[2], &[0, 2, 1]).unwrap(),
            ),
            ("repartition", $source.repartition($engine, 1).unwrap()),
            (
                "full transpose",
                $source.transpose($engine, &[2], &[1, 0]).unwrap(),
            ),
            (
                "transpose",
                $source.transpose($engine, &[1, 2], &[0]).unwrap(),
            ),
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
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let (serial, parallel) = (&runtime, &runtime);
    let parallel_engine = &mut parallel_engine();
    let provider = Arc::new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(U1FusionRule, SU2FusionRule).unwrap(),
    );
    let space = u1_su2_space(&provider);
    let real_serial: TensorMap<_, f64> =
        TensorMap::rand_with_seed(engine, serial, [&space, &space], [&space], 1217).unwrap();
    let real_parallel: TensorMap<_, f64> =
        TensorMap::rand_with_seed(engine, parallel, [&space, &space], [&space], 1217).unwrap();
    assert!(real_serial.dense_data().unwrap() == real_parallel.dense_data().unwrap());
    let complex_serial: TensorMap<_, Complex64> =
        TensorMap::rand_with_seed(engine, serial, [&space, &space], [&space], 1218).unwrap();
    let complex_parallel: TensorMap<_, Complex64> =
        TensorMap::rand_with_seed(engine, parallel, [&space, &space], [&space], 1218).unwrap();
    assert!(complex_serial.dense_data().unwrap() == complex_parallel.dense_data().unwrap());

    let expected_real = transforms!(engine, real_serial);
    let expected_complex = transforms!(engine, complex_serial);
    let (actual_real, actual_complex) = (
        transforms!(parallel_engine, real_parallel),
        transforms!(parallel_engine, complex_parallel),
    );
    assert_bit_identical!(expected_real, actual_real);
    assert_bit_identical!(expected_complex, actual_complex);
}

#[test]
fn u1_owned_transforms_are_bit_identical_across_thread_counts() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let (serial, parallel) = (&runtime, &runtime);
    let parallel_engine = &mut parallel_engine();
    let provider = Arc::new(U1FusionRule);
    let space = u1_space(&provider);
    let real_serial: TensorMap<_, f64> =
        TensorMap::rand_with_seed(engine, serial, [&space, &space], [&space], 1219).unwrap();
    let real_parallel: TensorMap<_, f64> =
        TensorMap::rand_with_seed(engine, parallel, [&space, &space], [&space], 1219).unwrap();
    assert!(real_serial.dense_data().unwrap() == real_parallel.dense_data().unwrap());
    let complex_serial: TensorMap<_, Complex64> =
        TensorMap::rand_with_seed(engine, serial, [&space, &space], [&space], 1220).unwrap();
    let complex_parallel: TensorMap<_, Complex64> =
        TensorMap::rand_with_seed(engine, parallel, [&space, &space], [&space], 1220).unwrap();

    let expected_real = transforms!(engine, real_serial);
    let expected_complex = transforms!(engine, complex_serial);
    let (actual_real, actual_complex) = (
        transforms!(parallel_engine, real_parallel),
        transforms!(parallel_engine, complex_parallel),
    );
    assert_bit_identical!(expected_real, actual_real);
    assert_bit_identical!(expected_complex, actual_complex);
}

#[test]
fn parallel_owned_permute_allocates_like_the_serial_owned_path() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    if run_isolated_or_return(
        "SYMTN_PARALLEL_OWNED_PERMUTE_ALLOCATES_LIKE_SERIAL_ISOLATED",
        "parallel_owned_permute_allocates_like_the_serial_owned_path",
    ) {
        return;
    }
    // What: on a multi-threaded engine a warmed owned permute makes the
    // same caller-thread allocations as the serial owned path and none of them
    // is allocator-zeroed: the multi-threaded backend writes the owned output
    // directly, with no `vec![0.0; P]` fallback and destination-reading
    // replay (#1217).
    let runtime = Runtime::builder().build().unwrap();
    let (serial, parallel) = (&runtime, &runtime);
    let parallel_engine = &mut parallel_engine();
    let provider = Arc::new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(U1FusionRule, SU2FusionRule).unwrap(),
    );
    let space = u1_su2_space(&provider);
    let source_serial: TensorMap<_, f64> =
        TensorMap::rand_with_seed(engine, serial, [&space, &space], [&space], 1221).unwrap();
    let source_parallel: TensorMap<_, f64> =
        TensorMap::rand_with_seed(engine, parallel, [&space, &space], [&space], 1221).unwrap();
    let payload_bytes = size_of_val(source_serial.dense_data().unwrap());

    let warm_serial = source_serial.permute(engine, &[1], &[2, 0]).unwrap();
    let (result_serial, serial_allocations, serial_bytes, serial_zeroed) =
        measure(|| source_serial.permute(engine, &[1], &[2, 0]).unwrap());
    black_box(result_serial.dense_data().unwrap());

    let warm_parallel = source_parallel
        .permute(parallel_engine, &[1], &[2, 0])
        .unwrap();
    assert!(warm_serial.dense_data().unwrap() == warm_parallel.dense_data().unwrap());
    // The caller thread may box a lazily created OS lock (macOS std allocates
    // each Mutex/Condvar on first use, once per process) inside the window the
    // first time it waits for the engine's workers (#1964); the cleanest of a
    // few windows is the steady-state count, and a per-call allocation would
    // appear in all of them.
    let (result_parallel, parallel_allocations, parallel_bytes, parallel_zeroed) = (0..THREADS + 3)
        .map(|_| {
            measure(|| {
                source_parallel
                    .permute(parallel_engine, &[1], &[2, 0])
                    .unwrap()
            })
        })
        .min_by_key(|&(_, calls, bytes, _)| (calls, bytes))
        .unwrap();
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
