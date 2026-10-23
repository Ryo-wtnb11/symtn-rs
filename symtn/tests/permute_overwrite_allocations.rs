use std::hint::black_box;
use std::sync::Arc;

use symtn::sector::{product_sector, SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use symtn::typed::Runtime;
use symtn::typed::{GradedSpace, TensorMap};

#[cfg(feature = "cgc-gen")]
use symtn::sector::SUNFusionRule;

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

fn measure(f: impl FnOnce()) -> (usize, usize) {
    let ((), allocs) = counting_alloc::measure(f);
    (allocs.calls as usize, allocs.bytes as usize)
}

#[cfg(feature = "cgc-gen")]
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

#[test]
fn cached_permute_overwrite_does_not_allocate_on_the_caller_thread() {
    let _measurement = counting_alloc::serial();
    // What: a warmed multiplicity-free non-Abelian permutation reuses its
    // compiled plan and replay workspace without allocating on the caller.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(U1FusionRule, SU2FusionRule).unwrap(),
    );
    let space = GradedSpace::try_new(
        Arc::clone(&provider),
        [
            (
                product_sector(U1Irrep::new(0), SU2Irrep::from_twice_spin(0)),
                8,
            ),
            (
                product_sector(U1Irrep::new(1), SU2Irrep::from_twice_spin(1)),
                6,
            ),
            (
                product_sector(U1Irrep::new(-1), SU2Irrep::from_twice_spin(1)),
                6,
            ),
            (
                product_sector(U1Irrep::new(0), SU2Irrep::from_twice_spin(2)),
                4,
            ),
        ],
    )
    .unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&runtime, [&space, &space], [&space], 197).unwrap();
    let mut destination = source.permute(&[1], &[2, 0]).unwrap();

    source
        .permute_into(&[1], &[2, 0], &mut destination, 1.0, 0.0)
        .unwrap();
    let destination_data = destination.dense_data().unwrap().as_ptr();

    let cost = measure(|| {
        source
            .permute_into(&[1], &[2, 0], &mut destination, 1.0, 0.0)
            .unwrap();
    });
    black_box(destination.dense_data().unwrap());

    assert_eq!(cost, (0, 0));
    assert_eq!(destination.dense_data().unwrap().as_ptr(), destination_data);
    assert!(std::ptr::eq(destination.provider(), provider.as_ref()));
}

#[test]
fn cached_u1_permute_overwrite_does_not_allocate_on_the_caller_thread() {
    let _measurement = counting_alloc::serial();
    // What: a warmed UniqueFusion permutation reuses its completed transformer
    // and replay workspace without allocating on the caller.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let space = GradedSpace::try_new(
        Arc::clone(&provider),
        [
            (U1Irrep::new(-1), 4),
            (U1Irrep::new(0), 8),
            (U1Irrep::new(1), 4),
        ],
    )
    .unwrap();
    let source: TensorMap<U1FusionRule, f64> =
        TensorMap::rand_with_seed(&runtime, [&space, &space], [&space], 418).unwrap();
    let mut destination = source.permute(&[1], &[2, 0]).unwrap();

    source
        .permute_into(&[1], &[2, 0], &mut destination, 1.0, 0.0)
        .unwrap();
    let destination_data = destination.dense_data().unwrap().as_ptr();

    let cost = measure(|| {
        source
            .permute_into(&[1], &[2, 0], &mut destination, 1.0, 0.0)
            .unwrap();
    });
    black_box(destination.dense_data().unwrap());

    assert_eq!(cost, (0, 0));
    assert_eq!(destination.dense_data().unwrap().as_ptr(), destination_data);
    assert!(std::ptr::eq(destination.provider(), provider.as_ref()));
}

/// Caller-thread allocation calls of one warm `into` call, after three
/// warm-up calls.
fn warm_calls(mut into: impl FnMut()) -> usize {
    for _ in 0..3 {
        into();
    }
    measure(&mut into).0
}

#[test]
fn warm_recoupling_permute_into_allocations_do_not_grow_with_block_count() {
    let _measurement = counting_alloc::serial();
    // What: a warm multiplicity-free non-Abelian `permute_into` submits each
    // Multi block's recoupling GEMM without a per-block allocation, so its
    // caller-thread allocation calls are the same for B and > 3B blocks
    // (#2184). Degeneracies stay small so no GEMM reaches the dense provider's
    // own packing workspace.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SU2FusionRule);
    let warm = |spins: usize| {
        let space = GradedSpace::try_new(
            Arc::clone(&provider),
            (0..spins).map(|twice_spin| (SU2Irrep::from_twice_spin(twice_spin), 2)),
        )
        .unwrap();
        let source: TensorMap<_, f64> =
            TensorMap::rand_with_seed(&runtime, [&space, &space], [&space, &space], 2184).unwrap();
        let mut destination = source.permute(&[1, 3], &[0, 2]).unwrap();
        let calls = warm_calls(|| {
            source
                .permute_into(&[1, 3], &[0, 2], &mut destination, 1.0, 0.0)
                .unwrap()
        });
        black_box(destination.dense_data().unwrap());
        (source.subblock_count(), calls)
    };
    let (small_blocks, small_calls) = warm(3);
    let (large_blocks, large_calls) = warm(4);
    assert!(large_blocks > 3 * small_blocks);
    assert_eq!(large_calls, small_calls);
}

/// The checked Generic counterpart of the multiplicity-free relation above:
/// SU(3) with fusion multiplicity.
#[cfg(feature = "cgc-gen")]
#[test]
fn warm_checked_generic_permute_into_allocations_do_not_grow_with_block_count() {
    let _measurement = counting_alloc::serial();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let warm = |labels: &[Vec<i64>]| {
        let space = GradedSpace::try_new(
            Arc::clone(&provider),
            labels.iter().map(|label| (label.clone(), 2)),
        )
        .unwrap();
        let source: TensorMap<_, f64> =
            TensorMap::rand_with_seed(&runtime, [&space, &space], [&space, &space], 2184).unwrap();
        let mut destination = source.permute(&[1, 3], &[0, 2]).unwrap();
        let calls = warm_calls(|| {
            source
                .permute_into(&[1, 3], &[0, 2], &mut destination, 1.0, 0.0)
                .unwrap()
        });
        black_box(destination.dense_data().unwrap());
        (source.subblock_count(), calls)
    };
    let (small_blocks, small_calls) = warm(&[vec![1, 0], vec![0, 1]]);
    let (large_blocks, large_calls) = warm(&[vec![1, 0], vec![0, 1], vec![1, 1]]);
    assert!(large_blocks > 3 * small_blocks);
    assert_eq!(large_calls, small_calls);
}

/// A source that outlives its structure-cache entry (here by `clear`, as by
/// eviction) loses the destination memo, so every warm permute derives its
/// destination; its trees were admitted once and are not re-admitted per
/// call (#2233). Each SU(3) admission query allocates in Racah, so per-block
/// re-admission would make the large case allocate more.
#[cfg(feature = "cgc-gen")]
#[test]
fn warm_checked_generic_permute_of_an_evicted_source_does_not_readmit_its_trees() {
    let _measurement = counting_alloc::serial();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let warm = |labels: &[Vec<i64>]| {
        let space = GradedSpace::try_new(
            Arc::clone(&provider),
            labels.iter().map(|label| (label.clone(), 2)),
        )
        .unwrap();
        let source: TensorMap<_, f64> =
            TensorMap::rand_with_seed(&runtime, [&space, &space], [&space, &space], 2233).unwrap();
        symtn::cache::clear();
        let calls = warm_calls(|| drop(black_box(source.permute(&[1, 3], &[0, 2]).unwrap())));
        (source.subblock_count(), calls)
    };
    let (small_blocks, small_calls) = warm(&[vec![1, 0], vec![0, 1]]);
    let (large_blocks, large_calls) = warm(&[vec![1, 0], vec![0, 1], vec![1, 1]]);
    assert!(large_blocks > 3 * small_blocks);
    assert_eq!(large_calls, small_calls);
}

#[test]
fn cached_planar_overwrites_do_not_allocate_on_the_caller_thread() {
    let _measurement = counting_alloc::serial();
    // What: the shared typed destination seam also reuses admitted full,
    // explicit, and repartition transpose operations without caller allocation.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let space = GradedSpace::try_new(
        Arc::clone(&provider),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 3),
            (U1Irrep::new(1), 2),
        ],
    )
    .unwrap();
    let source: TensorMap<U1FusionRule, f64> =
        TensorMap::rand_with_seed(&runtime, [&space, &space], [&space], 779).unwrap();

    let mut full = source.transpose(&[2], &[1, 0]).unwrap();
    source
        .transpose_into(&[2], &[1, 0], &mut full, 1.0, 0.0)
        .unwrap();
    assert_eq!(
        measure(|| {
            source
                .transpose_into(&[2], &[1, 0], &mut full, 1.0, 0.0)
                .unwrap()
        }),
        (0, 0)
    );

    let mut explicit = source.transpose(&[1, 2], &[0]).unwrap();
    source
        .transpose_into(&[1, 2], &[0], &mut explicit, 1.0, 0.0)
        .unwrap();
    assert_eq!(
        measure(|| {
            source
                .transpose_into(&[1, 2], &[0], &mut explicit, 1.0, 0.0)
                .unwrap()
        }),
        (0, 0)
    );

    let mut repartitioned = source.repartition(1).unwrap();
    source
        .repartition_into(&mut repartitioned, 1.0, 0.0)
        .unwrap();
    assert_eq!(
        measure(|| {
            source
                .repartition_into(&mut repartitioned, 1.0, 0.0)
                .unwrap()
        }),
        (0, 0)
    );
    assert!(std::ptr::eq(full.provider(), provider.as_ref()));
    assert!(std::ptr::eq(explicit.provider(), provider.as_ref()));
    assert!(std::ptr::eq(repartitioned.provider(), provider.as_ref()));
}

#[cfg(feature = "cgc-gen")]
fn assert_same_checked_tensor(
    actual: &TensorMap<SUNFusionRule, f64>,
    expected: &TensorMap<SUNFusionRule, f64>,
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

/// Measurement only: each child process starts with cold provider and Runtime
/// state. Allocation counts cover requested bytes on the caller thread, which
/// is the same scope as the established overwrite allocation gates above.
#[cfg(feature = "cgc-gen")]
#[test]
#[ignore = "benchmark: run via benchmarks.yml"]
fn checked_generic_public_transform_measurement() {
    const CASE_ENV: &str = "SYMTN_CHECKED_GENERIC_MEASUREMENT_CASE";
    const TEST_NAME: &str = "checked_generic_public_transform_measurement";

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
                .args([
                    "--ignored",
                    "--exact",
                    TEST_NAME,
                    "--nocapture",
                    "--test-threads=1",
                ])
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
    let _measurement = counting_alloc::serial();
    println!("case={case} allocation_scope=caller_thread_requested");

    if case == "su2_permute_control" {
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
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
        let source: TensorMap<_, f64> =
            TensorMap::rand_with_seed(&runtime, [&half_leg, &half_leg], [&coupled_leg], 783)
                .unwrap();
        assert_eq!(source.subblock_count(), 2);
        let runtime_before = completed_transformers();
        let (first, first_allocations, first_bytes, first_ns) =
            measure_value(|| source.permute(&[1, 0], &[2]).unwrap());
        let runtime_after_first = completed_transformers();
        let mut repeat_ns = Vec::with_capacity(7);
        let mut repeat_allocations = Vec::with_capacity(7);
        let mut repeat_bytes = Vec::with_capacity(7);
        for _ in 0..7 {
            let (repeated, allocations, bytes, ns) =
                measure_value(|| source.permute(&[1, 0], &[2]).unwrap());
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
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(n).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(adjoint, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg], |trees, _| {
            trees.codomain_vertices()[0].get() as f64
        })
        .unwrap();
    let apply = |source: &TensorMap<SUNFusionRule, f64>| match operation {
        "permute" => source.permute(&[1, 0], &[2]).unwrap(),
        "braid" => source.braid(&[1, 0], &[2], &[0, 1, 2]).unwrap(),
        "repartition" => source.repartition(1).unwrap(),
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

/// The process-global completed-transformer cache (`symtn::cache`).
#[cfg_attr(not(feature = "cgc-gen"), allow(dead_code))]
fn completed_transformers() -> symtn::cache::StructureCacheInfo {
    symtn::cache::stats()
        .into_iter()
        .find(|info| info.kind() == symtn::cache::StructureCacheKind::CompletedTreeTransformer)
        .expect("every structure cache kind reports")
}
