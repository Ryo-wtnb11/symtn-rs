//! Public checked-Generic transform timing (SU(3)/SU(4) permute, braid,
//! repartition, plus an SU(2) control). Run with `cargo bench -p symtn
//! --features cgc-gen --bench checked_generic_transform`.
//!
//! Each case runs in a child process so provider and Runtime state start
//! cold. Allocation counts cover requested bytes on the caller thread.
//! Timings are evidence, never a CI threshold.

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

use std::sync::Arc;

use symtn::sector::{LieGroupFusionRule, SU2FusionRule, SU2Irrep};
use symtn::typed::{GradedSpace, Runtime, TensorMap};

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

fn measure_value<T>(f: impl FnOnce() -> T) -> (T, usize, usize, u128) {
    let ((output, elapsed), allocs) = counting_alloc::measure(|| {
        let started = std::time::Instant::now();
        let output = f();
        (output, started.elapsed().as_nanos())
    });
    (
        output,
        allocs.calls as usize,
        allocs.bytes as usize,
        elapsed,
    )
}

fn assert_same_checked_tensor(
    actual: &TensorMap<LieGroupFusionRule, f64>,
    expected: &TensorMap<LieGroupFusionRule, f64>,
) {
    assert_eq!(actual.subblock_count(), expected.subblock_count());
    assert_eq!(
        actual.dense_data().unwrap().len(),
        expected.dense_data().unwrap().len()
    );
    for index in 0..actual.subblock_count() {
        assert_eq!(
            actual.subblock_fusion_trees(index).unwrap(),
            expected.subblock_fusion_trees(index).unwrap()
        );
    }
    for (actual, expected) in actual
        .dense_data()
        .unwrap()
        .iter()
        .zip(expected.dense_data().unwrap())
    {
        assert!((actual - expected).abs() <= 1e-12);
    }
}

fn completed_transformers() -> symtn::cache::StructureCacheInfo {
    symtn::cache::stats()
        .into_iter()
        .find(|info| info.kind() == symtn::cache::StructureCacheKind::CompletedTreeTransformer)
        .expect("every structure cache kind reports")
}

fn main() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    const CASE_ENV: &str = "SYMTN_CHECKED_GENERIC_MEASUREMENT_CASE";

    let Some(case) = std::env::var_os(CASE_ENV) else {
        for case in [
            "su3_permute",
            "su3_braid",
            "su3_repartition",
            "su4_permute",
            "su4_braid",
            "su4_repartition",
            "su2_permute_control",
        ] {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .env(CASE_ENV, case)
                .output()
                .unwrap();
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                output.status.success(),
                "case={case} status={}\nstdout:\n{stdout}\nstderr:\n{stderr}",
                output.status
            );
            print!("{stdout}");
        }
        return;
    };
    let case = case.to_str().unwrap();
    println!("case={case} allocation_scope=caller_thread_requested");

    if case == "su2_permute_control" {
        let runtime = Runtime::builder().build().unwrap();
        let provider = Arc::new(SU2FusionRule);
        let half = SU2Irrep::from_twice_spin(1);
        let half_leg = GradedSpace::try_new(Arc::clone(&provider), [(half, 1)]).unwrap();
        let coupled_leg = GradedSpace::try_new(
            Arc::clone(&provider),
            [
                (SU2Irrep::from_twice_spin(0), 1),
                (SU2Irrep::from_twice_spin(2), 1),
            ],
        )
        .unwrap();
        let source: TensorMap<_, f64> = TensorMap::rand_with_seed(
            engine,
            &runtime,
            [&half_leg, &half_leg],
            [&coupled_leg],
            783,
        )
        .unwrap();
        assert_eq!(source.subblock_count(), 2);
        let runtime_before = completed_transformers();
        let (first, first_allocations, first_bytes, first_ns) =
            measure_value(|| source.permute(engine, &[1, 0], &[2]).unwrap());
        let runtime_after_first = completed_transformers();
        let mut repeat_ns = Vec::with_capacity(7);
        let mut repeat_allocations = Vec::with_capacity(7);
        let mut repeat_bytes = Vec::with_capacity(7);
        for _ in 0..7 {
            let (repeated, allocations, bytes, ns) =
                measure_value(|| source.permute(engine, &[1, 0], &[2]).unwrap());
            assert_eq!(repeated.dense_data().unwrap(), first.dense_data().unwrap());
            repeat_ns.push(ns);
            repeat_allocations.push(allocations);
            repeat_bytes.push(bytes);
        }
        let runtime_after_repeat = completed_transformers();
        let mut sorted_ns = repeat_ns.clone();
        sorted_ns.sort_unstable();
        println!(
            "case={case} phase=public_first ns={first_ns} allocations={first_allocations} requested_bytes={first_bytes} runtime_before={runtime_before:?} runtime_after={runtime_after_first:?}"
        );
        println!(
            "case={case} phase=public_repeat samples_ns={repeat_ns:?} median_ns={} allocations={repeat_allocations:?} requested_bytes={repeat_bytes:?} runtime_after={runtime_after_repeat:?}",
            sorted_ns[sorted_ns.len() / 2]
        );
        assert!(runtime_after_first.entries() > runtime_before.entries());
        assert!(runtime_after_repeat.hits() > runtime_after_first.hits());
        return;
    }

    let (n, adjoint, operation) = match case {
        "su3_permute" => (3, vec![1, 1], "permute"),
        "su3_braid" => (3, vec![1, 1], "braid"),
        "su3_repartition" => (3, vec![1, 1], "repartition"),
        "su4_permute" => (4, vec![1, 0, 1], "permute"),
        "su4_braid" => (4, vec![1, 0, 1], "braid"),
        "su4_repartition" => (4, vec![1, 0, 1], "repartition"),
        _ => panic!("unknown measurement case: {case}"),
    };
    let runtime = Runtime::builder().build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(n).unwrap()).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(adjoint, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg], [&leg], |trees, _| {
            trees.codomain_vertices()[0].get() as f64
        })
        .unwrap();
    let mut apply = |source: &TensorMap<LieGroupFusionRule, f64>| match operation {
        "permute" => source.permute(engine, &[1, 0], &[2]).unwrap(),
        "braid" => source.braid(engine, &[1, 0], &[2], &[0, 1, 2]).unwrap(),
        "repartition" => source.repartition(engine, 1).unwrap(),
        _ => unreachable!(),
    };

    let runtime_before = completed_transformers();
    let (first, first_allocations, first_bytes, first_ns) = measure_value(|| apply(&source));
    let runtime_after_first = completed_transformers();
    let mut repeat_ns = Vec::with_capacity(7);
    let mut repeat_allocations = Vec::with_capacity(7);
    let mut repeat_bytes = Vec::with_capacity(7);
    for _ in 0..7 {
        let (repeated, allocations, bytes, ns) = measure_value(|| apply(&source));
        assert_same_checked_tensor(&repeated, &first);
        assert!(std::ptr::eq(repeated.provider(), provider.as_ref()));
        repeat_ns.push(ns);
        repeat_allocations.push(allocations);
        repeat_bytes.push(bytes);
    }
    let runtime_after_repeat = completed_transformers();
    let mut sorted_ns = repeat_ns.clone();
    sorted_ns.sort_unstable();
    println!(
        "case={case} phase=public_first_after_source_construction coefficient_caches=cold ns={first_ns} allocations={first_allocations} requested_bytes={first_bytes} runtime_before={runtime_before:?} runtime_after={runtime_after_first:?}"
    );
    println!(
        "case={case} phase=public_repeat_provider_warm samples_ns={repeat_ns:?} median_ns={} allocations={repeat_allocations:?} requested_bytes={repeat_bytes:?} runtime_after={runtime_after_repeat:?}",
        sorted_ns[sorted_ns.len() / 2]
    );
    assert!(runtime_after_first.entries() > runtime_before.entries());
    assert!(runtime_after_repeat.hits() > runtime_after_first.hits());
    assert!(std::ptr::eq(first.provider(), provider.as_ref()));
}
