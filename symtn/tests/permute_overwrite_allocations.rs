use std::hint::black_box;
use std::sync::Arc;

use symtn::sector::{product_sector, SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use symtn::typed::Runtime;
use symtn::typed::{GradedSpace, TensorMap};

#[cfg(feature = "cgc-gen")]
use symtn::sector::LieGroupFusionRule;

use crate::counting_alloc;

fn measure(f: impl FnOnce()) -> (usize, usize) {
    let ((), allocs) = counting_alloc::measure(f);
    (allocs.calls as usize, allocs.bytes as usize)
}

#[test]
fn cached_permute_overwrite_does_not_allocate_on_the_caller_thread() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: a warmed multiplicity-free non-Abelian permutation reuses its
    // compiled plan and replay workspace without allocating on the caller.
    let runtime = Runtime::builder().build().unwrap();
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
        TensorMap::rand_with_seed(engine, &runtime, [&space, &space], [&space], 197).unwrap();
    let mut destination = source.permute(engine, &[1], &[2, 0]).unwrap();

    source
        .permute_into(engine, &[1], &[2, 0], &mut destination, 1.0, 0.0)
        .unwrap();
    let destination_data = destination.dense_data().unwrap().as_ptr();

    let cost = measure(|| {
        source
            .permute_into(engine, &[1], &[2, 0], &mut destination, 1.0, 0.0)
            .unwrap();
    });
    black_box(destination.dense_data().unwrap());

    assert_eq!(cost, (0, 0));
    assert_eq!(destination.dense_data().unwrap().as_ptr(), destination_data);
    assert!(std::ptr::eq(destination.provider(), provider.as_ref()));
}

#[test]
fn cached_u1_permute_overwrite_does_not_allocate_on_the_caller_thread() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: a warmed UniqueFusion permutation reuses its completed transformer
    // and replay workspace without allocating on the caller.
    let runtime = Runtime::builder().build().unwrap();
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
        TensorMap::rand_with_seed(engine, &runtime, [&space, &space], [&space], 418).unwrap();
    let mut destination = source.permute(engine, &[1], &[2, 0]).unwrap();

    source
        .permute_into(engine, &[1], &[2, 0], &mut destination, 1.0, 0.0)
        .unwrap();
    let destination_data = destination.dense_data().unwrap().as_ptr();

    let cost = measure(|| {
        source
            .permute_into(engine, &[1], &[2, 0], &mut destination, 1.0, 0.0)
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
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: a warm multiplicity-free non-Abelian `permute_into` submits each
    // Multi block's recoupling GEMM without a per-block allocation, so its
    // caller-thread allocation calls are the same for B and > 3B blocks.
    // Degeneracies stay small so no GEMM reaches the dense provider's
    // own packing workspace.
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(SU2FusionRule);
    let mut warm = |spins: usize| {
        let space = GradedSpace::try_new(
            Arc::clone(&provider),
            (0..spins).map(|twice_spin| (SU2Irrep::from_twice_spin(twice_spin), 2)),
        )
        .unwrap();
        let source: TensorMap<_, f64> =
            TensorMap::rand_with_seed(engine, &runtime, [&space, &space], [&space, &space], 2184)
                .unwrap();
        let mut destination = source.permute(engine, &[1, 3], &[0, 2]).unwrap();
        let calls = warm_calls(|| {
            source
                .permute_into(engine, &[1, 3], &[0, 2], &mut destination, 1.0, 0.0)
                .unwrap()
        });
        black_box(destination.dense_data().unwrap());
        (source.subblock_count(), calls)
    };
    let (small_blocks, small_calls) = warm(3);
    let (large_blocks, large_calls) = warm(4);
    assert!(large_blocks > 3 * small_blocks);
    // Why raw equality holds for a BLAS provider too: each distinct GEMM
    // descriptor keeps its plan in its own engine cache slot, so the warm-up
    // calls build every block's plan and the measured call rebuilds none. A
    // vendor plan rebuild allocates, so a plan miss fails this test just as a
    // per-block symtn temporary does.
    assert_eq!(large_calls, small_calls);
}

/// The checked Generic counterpart of the multiplicity-free relation above:
/// SU(3) with fusion multiplicity.
#[cfg(feature = "cgc-gen")]
#[test]
fn warm_checked_generic_permute_into_allocations_do_not_grow_with_block_count() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let mut warm = |labels: &[Vec<i64>]| {
        let space = GradedSpace::try_new(
            Arc::clone(&provider),
            labels.iter().map(|label| (label.clone(), 2)),
        )
        .unwrap();
        let source: TensorMap<_, f64> =
            TensorMap::rand_with_seed(engine, &runtime, [&space, &space], [&space, &space], 2184)
                .unwrap();
        let mut destination = source.permute(engine, &[1, 3], &[0, 2]).unwrap();
        let calls = warm_calls(|| {
            source
                .permute_into(engine, &[1, 3], &[0, 2], &mut destination, 1.0, 0.0)
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
/// call. Each SU(3) admission query allocates in Racah, so per-block
/// re-admission would make the large case allocate more.
#[cfg(feature = "cgc-gen")]
#[test]
fn warm_checked_generic_permute_of_an_evicted_source_does_not_readmit_its_trees() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let mut warm = |labels: &[Vec<i64>]| {
        let space = GradedSpace::try_new(
            Arc::clone(&provider),
            labels.iter().map(|label| (label.clone(), 2)),
        )
        .unwrap();
        let source: TensorMap<_, f64> =
            TensorMap::rand_with_seed(engine, &runtime, [&space, &space], [&space, &space], 2233)
                .unwrap();
        symtn::cache::clear();
        let calls =
            warm_calls(|| drop(black_box(source.permute(engine, &[1, 3], &[0, 2]).unwrap())));
        (source.subblock_count(), calls)
    };
    let (small_blocks, small_calls) = warm(&[vec![1, 0], vec![0, 1]]);
    let (large_blocks, large_calls) = warm(&[vec![1, 0], vec![0, 1], vec![1, 1]]);
    assert!(large_blocks > 3 * small_blocks);
    assert_eq!(large_calls, small_calls);
}

#[test]
fn cached_planar_overwrites_do_not_allocate_on_the_caller_thread() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: the shared typed destination seam also reuses admitted full,
    // explicit, and repartition transpose operations without caller allocation.
    let runtime = Runtime::builder().build().unwrap();
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
        TensorMap::rand_with_seed(engine, &runtime, [&space, &space], [&space], 779).unwrap();

    let mut full = source.transpose(engine, &[2], &[1, 0]).unwrap();
    source
        .transpose_into(engine, &[2], &[1, 0], &mut full, 1.0, 0.0)
        .unwrap();
    assert_eq!(
        measure(|| {
            source
                .transpose_into(engine, &[2], &[1, 0], &mut full, 1.0, 0.0)
                .unwrap()
        }),
        (0, 0)
    );

    let mut explicit = source.transpose(engine, &[1, 2], &[0]).unwrap();
    source
        .transpose_into(engine, &[1, 2], &[0], &mut explicit, 1.0, 0.0)
        .unwrap();
    assert_eq!(
        measure(|| {
            source
                .transpose_into(engine, &[1, 2], &[0], &mut explicit, 1.0, 0.0)
                .unwrap()
        }),
        (0, 0)
    );

    let mut repartitioned = source.repartition(engine, 1).unwrap();
    source
        .repartition_into(engine, &mut repartitioned, 1.0, 0.0)
        .unwrap();
    assert_eq!(
        measure(|| {
            source
                .repartition_into(engine, &mut repartitioned, 1.0, 0.0)
                .unwrap()
        }),
        (0, 0)
    );
    assert!(std::ptr::eq(full.provider(), provider.as_ref()));
    assert!(std::ptr::eq(explicit.provider(), provider.as_ref()));
    assert!(std::ptr::eq(repartitioned.provider(), provider.as_ref()));
}
