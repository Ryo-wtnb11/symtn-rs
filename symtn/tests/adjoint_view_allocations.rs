use std::hint::black_box;
use std::sync::Arc;
use symtn::typed::Engine;

use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::Runtime;
use symtn::typed::{GradedSpace, TensorMap};

use crate::counting_alloc;

fn measure(f: impl FnOnce()) -> (u64, u64) {
    let (allocations, bytes, _) = measure_peak(f);
    (allocations, bytes)
}

fn measure_peak(f: impl FnOnce()) -> (u64, u64, u64) {
    let ((), allocs) = counting_alloc::measure(f);
    (allocs.calls, allocs.bytes, allocs.peak_live_bytes)
}

fn tensor(
    runtime: &Runtime,
    sectors: impl IntoIterator<Item = (i32, usize)>,
    rank: usize,
) -> TensorMap<U1FusionRule, num_complex::Complex64> {
    let engine = &mut symtn_dense::CpuEngine::serial();
    assert_eq!(rank % 2, 0);
    let provider = Arc::new(U1FusionRule);
    let space = GradedSpace::try_new(
        provider,
        sectors
            .into_iter()
            .map(|(charge, degeneracy)| (U1Irrep::new(charge), degeneracy)),
    )
    .unwrap();
    TensorMap::rand_with_seed(
        engine,
        runtime,
        std::iter::repeat_n(&space, rank / 2),
        std::iter::repeat_n(&space, rank / 2),
        261 + rank as u64,
    )
    .unwrap()
}

#[test]
fn adjoint_creation_allocates_metadata_not_a_receiver_sized_payload() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: the fallible typed constructor transactionally admits the logical
    // adjoint layout, so metadata may scale with rank and tree count. It must
    // not scale with degeneracy: that would mean copying the parent payload.
    let runtime = Runtime::builder().build().unwrap();
    for (rank, radius) in [(2, 0), (2, 2), (2, 6), (2, 12), (4, 4), (6, 1), (8, 0)] {
        let large_degeneracy = if rank == 2 { 16 } else { 2 };
        let small = tensor(&runtime, (-radius..=radius).map(|charge| (charge, 1)), rank);
        let large = tensor(
            &runtime,
            (-radius..=radius).map(|charge| (charge, large_degeneracy)),
            rank,
        );
        let small_layout = small.adjoint(engine).unwrap();
        let small_cost = measure(|| {
            black_box(small.adjoint(engine).unwrap());
        });
        let large_layout = large.adjoint(engine).unwrap();
        let large_cost = measure(|| {
            black_box(large.adjoint(engine).unwrap());
        });
        assert_eq!(
            small_cost, large_cost,
            "rank={rank}, sector radius={radius}: adjoint creation copied payload-sized state"
        );
        black_box((small_layout, large_layout));
    }
}

#[test]
fn adjoint_involution_does_not_allocate() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: the second dagger restores the parent body without allocating.
    let runtime = Runtime::builder().build().unwrap();
    let source = tensor(&runtime, (-4..=4).map(|charge| (charge, 2)), 4);
    let adjoint = source.adjoint(engine).unwrap();

    let cost = measure(|| {
        black_box(adjoint.adjoint(engine).unwrap());
    });

    assert_eq!(cost, (0, 0));
}

#[test]
fn labelled_block_inspection_refuses_a_lazy_adjoint_without_copying() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let parent = tensor(&runtime, [(0, 64)], 2);
    let parent_block = parent.subblock(0).unwrap();
    let adjoint = parent.adjoint(engine).unwrap();
    let payload_bytes = std::mem::size_of_val(parent.dense_data().unwrap()) as u64;

    let refused = measure(|| {
        assert!(adjoint.subblocks().is_err());
    });
    assert!(
        refused.1 < payload_bytes,
        "block inspection materialized the lazy adjoint: {refused:?}"
    );

    let materialized = adjoint.materialize(engine).unwrap();
    let borrowed = measure(|| {
        let blocks = materialized.subblocks().unwrap().collect::<Vec<_>>();
        assert_eq!(blocks.len(), 1);
        let (_, values) = &blocks[0];
        assert_eq!(
            values.data().as_ptr(),
            materialized.dense_data().unwrap().as_ptr()
        );
        let expected = parent.dense_data().unwrap()
            [parent_block.offset() + 5 * parent_block.strides()[0] + 3 * parent_block.strides()[1]]
            .conj();
        assert_eq!(values.get(&[3, 5]).copied(), Some(expected));
    });
    assert!(
        borrowed.1 < payload_bytes,
        "block inspection copied the materialized payload: {borrowed:?}"
    );
}

#[test]
fn rectangular_multisector_adjoint_svd_keeps_total_and_peak_below_materialization() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: compact and full SVD of a lazy adjoint factor the parent directly
    // and never hide a receiver-sized logical-adjoint allocation around it.
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let left = GradedSpace::try_new(
        Arc::clone(&provider),
        (-2..=2).map(|charge| (U1Irrep::new(charge), 24)),
    )
    .unwrap();
    let right =
        GradedSpace::try_new(provider, (-2..=2).map(|charge| (U1Irrep::new(charge), 16))).unwrap();
    let parent: TensorMap<_, num_complex::Complex64> =
        TensorMap::rand_with_seed(engine, &runtime, [&left], [&right], 693_698).unwrap();
    black_box(parent.svd_compact(engine, &[0], &[1]).unwrap());
    black_box(parent.svd_full(engine, &[0], &[1]).unwrap());

    let operation = |engine: &mut dyn Engine,
                     tensor: &TensorMap<U1FusionRule, num_complex::Complex64>,
                     full| {
        if full {
            tensor.svd_full(engine, &[0], &[1])
        } else {
            tensor.svd_compact(engine, &[0], &[1])
        }
    };
    for full in [false, true] {
        let name = if full { "full" } else { "compact" };
        let optimized = parent.adjoint(engine).unwrap();
        let baseline = parent.adjoint(engine).unwrap();
        black_box(operation(engine, &optimized, full).unwrap());
        let baseline_materialize = &baseline.materialize(engine).unwrap();
        black_box(operation(engine, baseline_materialize, full).unwrap());
        let actual = measure_peak(|| {
            black_box(operation(engine, &optimized, full).unwrap());
        });
        let expected = measure_peak(|| {
            let baseline_materialize = &baseline.materialize(engine).unwrap();
            black_box(operation(engine, baseline_materialize, full).unwrap());
        });
        assert!(
            actual.1 <= expected.1,
            "{name} total: {actual:?} vs {expected:?}"
        );
        assert!(
            actual.2 < expected.2,
            "{name} peak: {actual:?} vs {expected:?}"
        );
    }
}

#[test]
fn lazy_scale_and_add_allocate_only_one_input_sized_payload() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let space =
        GradedSpace::try_new(provider, (-4..=4).map(|charge| (U1Irrep::new(charge), 8))).unwrap();
    let parent: TensorMap<_, num_complex::Complex64> =
        TensorMap::rand_with_seed(engine, &runtime, [&space, &space], [&space], 666_401).unwrap();
    let other_parent =
        TensorMap::rand_with_seed(engine, &runtime, [&space, &space], [&space], 666_402).unwrap();
    let owned =
        TensorMap::rand_with_seed(engine, &runtime, [&space], [&space, &space], 666_403).unwrap();
    let payload_bytes = std::mem::size_of_val(parent.dense_data().unwrap()) as u64;
    let alpha = num_complex::Complex64::new(0.5, 0.0);
    let beta = num_complex::Complex64::new(-0.25, 0.0);

    let warm = parent.adjoint(engine).unwrap();
    let warm_other = other_parent.adjoint(engine).unwrap();
    black_box(warm.scale(engine, alpha).unwrap());
    black_box(warm.axpby(engine, alpha, &owned, beta).unwrap());
    black_box(warm.axpby(engine, alpha, &warm_other, beta).unwrap());

    let scale_lazy = parent.adjoint(engine).unwrap();
    let mixed_lazy = parent.adjoint(engine).unwrap();
    let pair_lazy = parent.adjoint(engine).unwrap();
    let other_pair_lazy = other_parent.adjoint(engine).unwrap();
    for (_, bytes) in [
        measure(|| {
            black_box(scale_lazy.scale(engine, alpha).unwrap());
        }),
        measure(|| {
            black_box(mixed_lazy.axpby(engine, alpha, &owned, beta).unwrap());
        }),
        measure(|| {
            black_box(
                pair_lazy
                    .axpby(engine, alpha, &other_pair_lazy, beta)
                    .unwrap(),
            );
        }),
    ] {
        assert!(
            bytes >= payload_bytes && bytes < 2 * payload_bytes,
            "expected one payload allocation ({payload_bytes} bytes), observed {bytes} bytes"
        );
    }
}

#[test]
fn lazy_add_allocation_count_does_not_depend_on_the_block_count() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: `add` with a lazy-adjoint operand copies each block through one
    // bounds-checked strided call (#1399) and allocates only its output and
    // the tensor wrapping it, the same count as the per-element kernel it
    // replaced, for one block or many.
    let runtime = Runtime::builder().build().unwrap();
    let alpha = num_complex::Complex64::new(0.5, 0.0);
    let beta = num_complex::Complex64::new(-0.25, 0.0);
    let mut counts = Vec::new();
    for (rank, radius, degeneracy) in [(2, 0, 8), (2, 6, 2), (4, 3, 2), (6, 1, 2)] {
        let sectors = || (-radius..=radius).map(|charge| (charge, degeneracy));
        let parent = tensor(&runtime, sectors(), rank);
        let other = tensor(&runtime, sectors(), rank);
        let owned = other.clone();
        let lazy = parent.adjoint(engine).unwrap();
        let other_lazy = other.adjoint(engine).unwrap();
        black_box(lazy.axpby(engine, alpha, &owned, beta).unwrap());
        black_box(lazy.axpby(engine, alpha, &other_lazy, beta).unwrap());
        let mixed = measure(|| {
            black_box(lazy.axpby(engine, alpha, &owned, beta).unwrap());
        })
        .0;
        let pair = measure(|| {
            black_box(lazy.axpby(engine, alpha, &other_lazy, beta).unwrap());
        })
        .0;
        counts.push((mixed, pair));
    }
    // Two lazy operands give a lazy adjoint of the parents' sum (A10), so
    // their count may differ from the mixed one, but neither depends on the
    // block count.
    assert!(
        counts.iter().all(|&count| count == counts[0]),
        "(mixed, pair) per block count: {counts:?}"
    );
}

#[test]
fn block_stride_buffers_spill_only_past_rank_sixteen() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: the per-op block layout (`CheckedBlockLayout`, #1401) holds up
    // to 16 non-unit axes inline, so at rank 10 lazy `add` and lazy-adjoint
    // materialization allocate as at rank 2, for one block or many. Past 16
    // axes its four stride buffers spill at most once per op.
    let runtime = Runtime::builder().build().unwrap();
    let alpha = num_complex::Complex64::new(0.5, 0.0);
    let beta = num_complex::Complex64::new(-0.25, 0.0);
    let mut counts = Vec::new();
    for (sectors, rank) in [
        (vec![(0, 2)], 2),
        (vec![(0, 2)], 10),
        (vec![(0, 2), (1, 1), (-1, 1)], 10),
        (vec![(0, 2)], 18),
    ] {
        let parent = tensor(&runtime, sectors.clone(), rank);
        let owned = tensor(&runtime, sectors, rank);
        let other_lazy = owned.adjoint(engine).unwrap();
        let lazy = parent.adjoint(engine).unwrap();
        black_box(lazy.axpby(engine, alpha, &owned, beta).unwrap());
        black_box(lazy.axpby(engine, alpha, &other_lazy, beta).unwrap());
        let mixed = measure(|| {
            black_box(lazy.axpby(engine, alpha, &owned, beta).unwrap());
        })
        .0;
        let pair = measure(|| {
            black_box(lazy.axpby(engine, alpha, &other_lazy, beta).unwrap());
        })
        .0;
        let fresh = parent.adjoint(engine).unwrap();
        let materialize = measure(|| {
            black_box(
                fresh
                    .materialize(engine)
                    .unwrap()
                    .dense_data()
                    .unwrap()
                    .len(),
            );
        })
        .0;
        counts.push([mixed, pair, materialize]);
    }
    let [inline, one_block, many_blocks, spilled] = counts[..] else {
        unreachable!()
    };
    assert_eq!(one_block, inline, "rank 10, one block, against rank 2");
    assert_eq!(many_blocks, inline, "rank 10, three blocks, against rank 2");
    // Two lazy operands combine their parents element-wise with no layout
    // (A10).
    assert_eq!(spilled[1], inline[1], "rank 18 lazy pair");
    for (route, index) in [("mixed add", 0), ("materialize", 2)] {
        assert!(
            spilled[index] <= inline[index] + 4,
            "rank 18 {route}: {spilled:?} against {inline:?}"
        );
    }
}

#[test]
fn identity_adjoint_transform_cost_is_independent_of_rank_and_block_count() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: identity permute, braid, and repartition share the lazy view with
    // zero allocations across increasing rank and U1 sector counts.
    let runtime = Runtime::builder().build().unwrap();
    for (rank, radius) in [(2, 0), (2, 6), (4, 4), (6, 1), (8, 0), (10, 0)] {
        let source = tensor(&runtime, (-radius..=radius).map(|charge| (charge, 2)), rank);
        let adjoint = source.adjoint(engine).unwrap();
        let split = adjoint.codomain_rank();
        let codomain_axes = (0..split).collect::<Vec<_>>();
        let domain_axes = (split..rank).collect::<Vec<_>>();
        let levels = (0..rank).map(|axis| rank - axis).collect::<Vec<_>>();

        for cost in [
            measure(|| {
                black_box(
                    adjoint
                        .permute(engine, &codomain_axes, &domain_axes)
                        .unwrap(),
                );
            }),
            measure(|| {
                black_box(
                    adjoint
                        .braid(engine, &codomain_axes, &domain_axes, &levels)
                        .unwrap(),
                );
            }),
            measure(|| {
                black_box(adjoint.repartition(engine, split).unwrap());
            }),
        ] {
            assert_eq!(cost, (0, 0), "rank={rank}, sector radius={radius}");
        }
    }
}

#[test]
fn ordinary_tensor_clone_does_not_allocate() {
    let _serial = crate::counting_alloc::serial();
    // What: the representation split keeps an owned tensor's value-like Arc clone cost.
    let runtime = Runtime::builder().build().unwrap();
    let source = tensor(&runtime, (-4..=4).map(|charge| (charge, 2)), 4);

    let cost = measure(|| {
        black_box(source.clone());
    });

    assert_eq!(cost, (0, 0));
}

fn measure_eager_lazy_core_compose(
    rows: &[(i32, usize)],
    contracted: &[(i32, usize)],
    cols: &[(i32, usize)],
    seed: u64,
) -> u64 {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let make_space = |sectors: &[(i32, usize)]| {
        GradedSpace::try_new(
            Arc::clone(&provider),
            sectors
                .iter()
                .map(|&(charge, degeneracy)| (U1Irrep::new(charge), degeneracy)),
        )
        .unwrap()
    };
    let rows = make_space(rows);
    let contracted = make_space(contracted);
    let cols = make_space(cols);
    let parent: TensorMap<_, num_complex::Complex64> =
        TensorMap::rand_with_seed(engine, &runtime, [&contracted], [&rows], seed).unwrap();
    let rhs =
        TensorMap::rand_with_seed(engine, &runtime, [&contracted], [&cols], seed + 1).unwrap();
    let lhs = parent.adjoint(engine).unwrap();
    black_box(lhs.compose(engine, &rhs).unwrap());
    measure(|| {
        black_box(lhs.compose(engine, &rhs).unwrap());
    })
    .0
}

/// Purpose: a warm typed lazy-adjoint compose costs per call, not per
/// coupled group or block: one group and five groups cost the same.
#[test]
fn warm_typed_lazy_compose_does_not_scale_with_groups() {
    let _serial = crate::counting_alloc::serial();
    let single = measure_eager_lazy_core_compose(&[(0, 3)], &[(0, 2)], &[(0, 4)], 272_001);
    let charges = || -2..=2;
    let many = measure_eager_lazy_core_compose(
        &charges().map(|charge| (charge, 3)).collect::<Vec<_>>(),
        &charges().map(|charge| (charge, 2)).collect::<Vec<_>>(),
        &charges().map(|charge| (charge, 4)).collect::<Vec<_>>(),
        272_011,
    );
    counting_alloc::assert_same_calls("coupled groups 1 -> 5", single, many);
}

#[test]
fn materialize_allocates_one_fresh_payload_per_call() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: `materialize` (#1545) of an owned dense tensor or a lazy adjoint
    // costs one payload allocation plus fixed body wrappers, whatever the
    // block count, on every call: the result never shares a payload, so a
    // repeated call cannot be free.
    let runtime = Runtime::builder().build().unwrap();
    let mut counts = Vec::new();
    for (rank, radius, degeneracy) in [(2, 0, 8), (2, 6, 2), (4, 3, 2), (6, 1, 2)] {
        let source = tensor(
            &runtime,
            (-radius..=radius).map(|charge| (charge, degeneracy)),
            rank,
        );
        assert!(rank == 2 && radius == 0 || source.subblock_count() > 1);
        let payload_bytes = std::mem::size_of_val(source.dense_data().unwrap()) as u64;
        let lazy = source.adjoint(engine).unwrap();
        for input in [&source, &lazy, &source, &lazy] {
            let (allocations, bytes) = measure(|| {
                black_box(input.materialize(engine).unwrap());
            });
            counts.push(allocations);
            assert!(
                bytes >= payload_bytes && bytes < payload_bytes + 256,
                "rank={rank}, radius={radius}: {bytes} bytes for a {payload_bytes}-byte payload"
            );
        }
    }
    assert!(
        counts.iter().all(|&count| count == counts[0]),
        "allocations per block count and call: {counts:?}"
    );
}

/// Purpose: a warm checked Generic adjoint costs the same however many
/// fusion trees its HomSpace enumerates.
#[cfg(feature = "cgc-gen")]
#[test]
fn warm_checked_generic_adjoint_does_not_scale_with_fusion_trees() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What (#1942): a warm checked Generic adjoint of a complete tensor costs
    // a constant number of header allocations, independent of how many fusion
    // trees the HomSpace enumerates. Re-enumerating the adjoint trees with
    // provider queries cost 810 and 1407 calls for the two spaces below.
    use symtn::sector::LieGroupFusionRule;
    let runtime = Runtime::builder().build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let space = |sectors: &[(Vec<i64>, usize)]| {
        GradedSpace::try_new(Arc::clone(&provider), sectors.iter().cloned()).unwrap()
    };
    let small = space(&[(vec![0, 0], 2), (vec![1, 0], 1), (vec![1, 1], 2)]);
    let wide = space(&[
        (vec![0, 0], 1),
        (vec![1, 0], 1),
        (vec![0, 1], 1),
        (vec![1, 1], 1),
    ]);
    let mut costs = Vec::new();
    for leg in [&small, &wide] {
        let tensor: TensorMap<LieGroupFusionRule, f64> =
            TensorMap::rand_with_seed(engine, &runtime, [leg, leg], [leg, leg], 1942).unwrap();
        black_box(tensor.adjoint(engine).unwrap());
        let (calls, _) = measure(|| {
            black_box(tensor.adjoint(engine).unwrap());
        });
        costs.push(calls);
    }
    counting_alloc::assert_same_calls("fusion trees small -> wide", costs[0], costs[1]);
}
