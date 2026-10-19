use std::hint::black_box;
use std::sync::Arc;

use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::Runtime;
use symtn::typed::{GradedSpace, TensorMap};

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

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
        runtime,
        std::iter::repeat_n(&space, rank / 2),
        std::iter::repeat_n(&space, rank / 2),
        261 + rank as u64,
    )
    .unwrap()
}

#[test]
fn adjoint_creation_allocates_metadata_not_a_receiver_sized_payload() {
    let _measurement = counting_alloc::serial();
    // What: the fallible typed constructor transactionally admits the logical
    // adjoint layout, so metadata may scale with rank and tree count. It must
    // not scale with degeneracy: that would mean copying the parent payload.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    for (rank, radius) in [(2, 0), (2, 2), (2, 6), (2, 12), (4, 4), (6, 1), (8, 0)] {
        let large_degeneracy = if rank == 2 { 16 } else { 2 };
        let small = tensor(&runtime, (-radius..=radius).map(|charge| (charge, 1)), rank);
        let large = tensor(
            &runtime,
            (-radius..=radius).map(|charge| (charge, large_degeneracy)),
            rank,
        );
        let small_layout = small.adjoint().unwrap();
        let small_cost = measure(|| {
            black_box(small.adjoint().unwrap());
        });
        let large_layout = large.adjoint().unwrap();
        let large_cost = measure(|| {
            black_box(large.adjoint().unwrap());
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
    let _measurement = counting_alloc::serial();
    // What: the second dagger restores the parent body without allocating.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let source = tensor(&runtime, (-4..=4).map(|charge| (charge, 2)), 4);
    let adjoint = source.adjoint().unwrap();

    let cost = measure(|| {
        black_box(adjoint.adjoint().unwrap());
    });

    assert_eq!(cost, (0, 0));
}

#[test]
fn labelled_block_inspection_refuses_a_lazy_adjoint_without_copying() {
    let _measurement = counting_alloc::serial();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let parent = tensor(&runtime, [(0, 64)], 2);
    let parent_block = parent.subblock(0).unwrap();
    let adjoint = parent.adjoint().unwrap();
    let payload_bytes = std::mem::size_of_val(parent.dense_data().unwrap()) as u64;

    let refused = measure(|| {
        assert!(adjoint.subblocks().is_err());
    });
    assert!(
        refused.1 < payload_bytes,
        "block inspection materialized the lazy adjoint: {refused:?}"
    );

    let materialized = adjoint.materialize().unwrap();
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
    let _measurement = counting_alloc::serial();
    // What: compact and full SVD of a lazy adjoint factor the parent directly
    // and never hide a receiver-sized logical-adjoint allocation around it.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let left = GradedSpace::try_new(
        Arc::clone(&provider),
        (-2..=2).map(|charge| (U1Irrep::new(charge), 24)),
    )
    .unwrap();
    let right =
        GradedSpace::try_new(provider, (-2..=2).map(|charge| (U1Irrep::new(charge), 16))).unwrap();
    let parent: TensorMap<_, num_complex::Complex64> =
        TensorMap::rand_with_seed(&runtime, [&left], [&right], 693_698).unwrap();
    black_box(parent.svd_compact(&[0], &[1]).unwrap());
    black_box(parent.svd_full(&[0], &[1]).unwrap());

    let operation = |tensor: &TensorMap<U1FusionRule, num_complex::Complex64>, full| {
        if full {
            tensor.svd_full(&[0], &[1])
        } else {
            tensor.svd_compact(&[0], &[1])
        }
    };
    for full in [false, true] {
        let name = if full { "full" } else { "compact" };
        let optimized = parent.adjoint().unwrap();
        let baseline = parent.adjoint().unwrap();
        black_box(operation(&optimized, full).unwrap());
        black_box(operation(&baseline.materialize().unwrap(), full).unwrap());
        let actual = measure_peak(|| {
            black_box(operation(&optimized, full).unwrap());
        });
        let expected = measure_peak(|| {
            black_box(operation(&baseline.materialize().unwrap(), full).unwrap());
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
    let _measurement = counting_alloc::serial();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let space =
        GradedSpace::try_new(provider, (-4..=4).map(|charge| (U1Irrep::new(charge), 8))).unwrap();
    let parent: TensorMap<_, num_complex::Complex64> =
        TensorMap::rand_with_seed(&runtime, [&space, &space], [&space], 666_401).unwrap();
    let other_parent =
        TensorMap::rand_with_seed(&runtime, [&space, &space], [&space], 666_402).unwrap();
    let owned = TensorMap::rand_with_seed(&runtime, [&space], [&space, &space], 666_403).unwrap();
    let payload_bytes = std::mem::size_of_val(parent.dense_data().unwrap()) as u64;
    let alpha = num_complex::Complex64::new(0.5, 0.0);
    let beta = num_complex::Complex64::new(-0.25, 0.0);

    let warm = parent.adjoint().unwrap();
    let warm_other = other_parent.adjoint().unwrap();
    black_box(warm.scale(alpha).unwrap());
    black_box(warm.axpby(alpha, &owned, beta).unwrap());
    black_box(warm.axpby(alpha, &warm_other, beta).unwrap());

    let scale_lazy = parent.adjoint().unwrap();
    let mixed_lazy = parent.adjoint().unwrap();
    let pair_lazy = parent.adjoint().unwrap();
    let other_pair_lazy = other_parent.adjoint().unwrap();
    for (_, bytes) in [
        measure(|| {
            black_box(scale_lazy.scale(alpha).unwrap());
        }),
        measure(|| {
            black_box(mixed_lazy.axpby(alpha, &owned, beta).unwrap());
        }),
        measure(|| {
            black_box(pair_lazy.axpby(alpha, &other_pair_lazy, beta).unwrap());
        }),
    ] {
        assert!(
            bytes >= payload_bytes && bytes < 2 * payload_bytes,
            "expected one payload allocation ({payload_bytes} bytes), observed {bytes} bytes"
        );
    }
}

#[test]
fn lazy_add_allocation_count_is_pinned_across_block_counts() {
    let _measurement = counting_alloc::serial();
    // What: `add` with a lazy-adjoint operand copies each block through one
    // bounds-checked strided call (#1399) and allocates only its output and
    // the tensor wrapping it, the same count as the per-element kernel it
    // replaced, for one block or many.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let alpha = num_complex::Complex64::new(0.5, 0.0);
    let beta = num_complex::Complex64::new(-0.25, 0.0);
    let mut counts = Vec::new();
    for (rank, radius, degeneracy) in [(2, 0, 8), (2, 6, 2), (4, 3, 2), (6, 1, 2)] {
        let sectors = || (-radius..=radius).map(|charge| (charge, degeneracy));
        let parent = tensor(&runtime, sectors(), rank);
        let other = tensor(&runtime, sectors(), rank);
        let owned = other.clone();
        let lazy = parent.adjoint().unwrap();
        let other_lazy = other.adjoint().unwrap();
        black_box(lazy.axpby(alpha, &owned, beta).unwrap());
        black_box(lazy.axpby(alpha, &other_lazy, beta).unwrap());
        let mixed = measure(|| {
            black_box(lazy.axpby(alpha, &owned, beta).unwrap());
        })
        .0;
        let pair = measure(|| {
            black_box(lazy.axpby(alpha, &other_lazy, beta).unwrap());
        })
        .0;
        counts.push((mixed, pair));
    }
    // Mixed: payload `Vec`, `Arc<TypedData>` and `Arc<TypedTensorBody>`,
    // three, the same as the per-element kernel (#1399). Two
    // lazy operands give a lazy adjoint of the parents' sum (A10): those
    // three plus its `Arc<TypedAdjointView>` header.
    assert_eq!(counts, vec![(3, 4); 4]);
}

#[test]
fn block_stride_buffers_spill_only_past_rank_sixteen() {
    let _measurement = counting_alloc::serial();
    // What: the per-op block layout (`CheckedBlockLayout`, #1401) holds up
    // to 16 non-unit axes inline, so at rank 10 lazy `add` and lazy-adjoint
    // materialization allocate only their output, as main's per-element
    // kernel did, for one block or many. Past 16 axes its stride buffers
    // spill once per op.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let alpha = num_complex::Complex64::new(0.5, 0.0);
    let beta = num_complex::Complex64::new(-0.25, 0.0);
    let mut counts = Vec::new();
    for (sectors, rank) in [
        (vec![(0, 2)], 10),
        (vec![(0, 2), (1, 1), (-1, 1)], 10),
        (vec![(0, 2)], 18),
    ] {
        let parent = tensor(&runtime, sectors.clone(), rank);
        let owned = tensor(&runtime, sectors, rank);
        let other_lazy = owned.adjoint().unwrap();
        let lazy = parent.adjoint().unwrap();
        black_box(lazy.axpby(alpha, &owned, beta).unwrap());
        black_box(lazy.axpby(alpha, &other_lazy, beta).unwrap());
        let mixed = measure(|| {
            black_box(lazy.axpby(alpha, &owned, beta).unwrap());
        })
        .0;
        let pair = measure(|| {
            black_box(lazy.axpby(alpha, &other_lazy, beta).unwrap());
        })
        .0;
        let fresh = parent.adjoint().unwrap();
        let materialize = measure(|| {
            black_box(fresh.materialize().unwrap().dense_data().unwrap().len());
        })
        .0;
        counts.push((rank, parent.subblock_count() > 1, mixed, pair, materialize));
    }
    // Output only: 3 calls. At rank 18 the mixed `add` spills its four layout
    // buffers (extents and three operands' strides) and materialization its
    // three. Two lazy operands combine their parents element-wise with no
    // layout (A10): the output plus the lazy view header, 4 at every rank.
    assert_eq!(
        counts,
        vec![
            (10, false, 3, 4, 3),
            (10, true, 3, 4, 3),
            (18, false, 7, 4, 6)
        ]
    );
}

#[test]
fn identity_adjoint_transform_cost_is_independent_of_rank_and_block_count() {
    let _measurement = counting_alloc::serial();
    // What: identity permute, braid, and repartition share the lazy view with
    // zero allocations across increasing rank and U1 sector counts.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    for (rank, radius) in [(2, 0), (2, 6), (4, 4), (6, 1), (8, 0), (10, 0)] {
        let source = tensor(&runtime, (-radius..=radius).map(|charge| (charge, 2)), rank);
        let adjoint = source.adjoint().unwrap();
        let split = adjoint.codomain_rank();
        let codomain_axes = (0..split).collect::<Vec<_>>();
        let domain_axes = (split..rank).collect::<Vec<_>>();
        let levels = (0..rank).map(|axis| rank - axis).collect::<Vec<_>>();

        for cost in [
            measure(|| {
                black_box(adjoint.permute(&codomain_axes, &domain_axes).unwrap());
            }),
            measure(|| {
                black_box(
                    adjoint
                        .braid(&codomain_axes, &domain_axes, &levels)
                        .unwrap(),
                );
            }),
            measure(|| {
                black_box(adjoint.repartition(split).unwrap());
            }),
        ] {
            assert_eq!(cost, (0, 0), "rank={rank}, sector radius={radius}");
        }
    }
}

#[test]
fn ordinary_tensor_clone_does_not_allocate() {
    let _measurement = counting_alloc::serial();
    // What: the representation split keeps an owned tensor's value-like Arc clone cost.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
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
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
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
        TensorMap::rand_with_seed(&runtime, [&contracted], [&rows], seed).unwrap();
    let rhs = TensorMap::rand_with_seed(&runtime, [&contracted], [&cols], seed + 1).unwrap();
    let lhs = parent.adjoint().unwrap();
    black_box(lhs.compose(&rhs).unwrap());
    measure(|| {
        black_box(lhs.compose(&rhs).unwrap());
    })
    .0
}

#[test]
fn typed_single_group_lazy_compose_stays_below_the_measured_engine_margin() {
    let _measurement = counting_alloc::serial();
    // What: warmed typed lazy-adjoint compose measured 40 allocations when
    // admitted here; 64 leaves platform headroom without inheriting the erased
    // facade's former 128-allocation ceiling.
    let calls = measure_eager_lazy_core_compose(&[(0, 3)], &[(0, 2)], &[(0, 4)], 272_001);
    assert!(calls <= 64, "typed lazy compose allocated {calls} times");
}

#[test]
fn typed_multigroup_lazy_compose_stays_below_the_measured_engine_margin() {
    let _measurement = counting_alloc::serial();
    // What: the corresponding multigroup route measured 52 allocations; 80
    // keeps explicit headroom while still killing a return to per-term replay.
    let calls = measure_eager_lazy_core_compose(
        &[(-1, 2), (0, 3), (1, 1)],
        &[(-1, 1), (0, 2), (1, 3)],
        &[(-1, 3), (0, 1), (1, 2)],
        272_011,
    );
    assert!(
        calls <= 80,
        "typed multigroup compose allocated {calls} times"
    );
}

#[test]
fn materialize_allocates_one_fresh_payload_per_call() {
    let _measurement = counting_alloc::serial();
    // What: `materialize` (#1545) of an owned dense tensor or a lazy adjoint
    // costs one payload allocation plus the two fixed body wrappers, whatever
    // the block count, on every call: the result never shares a payload, so a
    // repeated call cannot be free.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    for (rank, radius, degeneracy) in [(2, 0, 8), (2, 6, 2), (4, 3, 2), (6, 1, 2)] {
        let source = tensor(
            &runtime,
            (-radius..=radius).map(|charge| (charge, degeneracy)),
            rank,
        );
        assert!(rank == 2 && radius == 0 || source.subblock_count() > 1);
        let payload_bytes = std::mem::size_of_val(source.dense_data().unwrap()) as u64;
        let lazy = source.adjoint().unwrap();
        for input in [&source, &lazy, &source, &lazy] {
            let (allocations, bytes) = measure(|| {
                black_box(input.materialize().unwrap());
            });
            // Payload `Vec`, its `Arc<TypedData>` and the `Arc<TypedTensorBody>`.
            assert_eq!(allocations, 3, "rank={rank}, radius={radius}");
            assert!(
                bytes >= payload_bytes && bytes < payload_bytes + 256,
                "rank={rank}, radius={radius}: {bytes} bytes for a {payload_bytes}-byte payload"
            );
        }
    }
}

#[cfg(feature = "cgc-gen")]
#[test]
fn warm_checked_generic_adjoint_does_not_scale_with_fusion_trees() {
    let _measurement = counting_alloc::serial();
    // What (#1942): a warm checked Generic adjoint of a complete tensor costs
    // a constant number of header allocations, independent of how many fusion
    // trees the HomSpace enumerates. Re-enumerating the adjoint trees with
    // provider queries cost 810 and 1407 calls for the two spaces below.
    use symtn::sector::SUNFusionRule;
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
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
        let tensor: TensorMap<SUNFusionRule, f64> =
            TensorMap::rand_with_seed(&runtime, [leg, leg], [leg, leg], 1942).unwrap();
        black_box(tensor.adjoint().unwrap());
        let (calls, _) = measure(|| {
            black_box(tensor.adjoint().unwrap());
        });
        costs.push(calls);
    }
    assert!(
        costs.iter().all(|&calls| calls <= 8) && costs[0] == costs[1],
        "warm checked Generic adjoint calls: {costs:?}"
    );
}
