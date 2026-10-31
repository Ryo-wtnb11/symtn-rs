//! Allocation contract of `restrict_leg` and of the re-routed internal
//! network restriction.
//!
//! The contract is *not* a total call count: each call rebuilds the small
//! structural objects of the destination hom-space. What is pinned is that
//! the output payload is never zero-filled — its blocks tile it, so it is
//! allocated uninitialized and overwritten once — and that everything
//! else is independent of the degeneracy dimensions — two tensors with the
//! same sector structure and different degeneracies must allocate the same
//! number of non-payload blocks.

use std::hint::black_box;
use std::sync::Arc;

use symtn::sector::TypedSectorAdmission;
use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::__network::{self, NetworkDegeneracyRestriction};
use symtn::typed::{GradedSpace, LegSelection, Runtime, TensorMap};
use symtn::typed::{StackedTensorMap, Svd};

use crate::counting_alloc;

#[derive(Debug)]
struct Measurement {
    allocations: usize,
    bytes: usize,
    /// Allocations, and the allocator-zeroed ones, of exactly the
    /// `payload_bytes` passed to `measure`.
    payloads: u64,
    zeroed_payloads: u64,
}

fn measure(payload_bytes: usize, operation: impl FnOnce()) -> Measurement {
    let ((), allocs) = counting_alloc::measure_matching(payload_bytes..=payload_bytes, operation);
    Measurement {
        allocations: allocs.calls as usize,
        bytes: allocs.bytes as usize,
        payloads: allocs.matched_calls,
        zeroed_payloads: allocs.matched_zeroed_calls,
    }
}

fn u1(provider: &Arc<U1FusionRule>, pairs: &[(i32, usize)]) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::clone(provider),
        pairs
            .iter()
            .map(|&(charge, degeneracy)| (U1Irrep::new(charge), degeneracy)),
    )
    .unwrap()
}

/// One `restrict_leg` on a leg whose degeneracies are `scale` times a base
/// shape: the measurement and the payload byte count for that scale.
fn restrict_measurement(scale: usize) -> (Measurement, usize) {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = u1(
        &provider,
        &[(-1, 2 * scale), (0, 3 * scale), (1, 2 * scale)],
    );
    let other = u1(&provider, &[(-1, 1), (0, 2), (1, 1)]);
    let source: TensorMap<_, f64> =
        TensorMap::rand_with_seed(engine, &runtime, [&leg], [&other, &other], 41).unwrap();
    let selection = LegSelection::try_new(
        &leg,
        [
            (U1Irrep::new(-1), 0..scale),
            (U1Irrep::new(0), scale..3 * scale),
        ],
    )
    .unwrap();

    // Warm every structural cache the call can hit before measuring.
    let warm = source.restrict_leg(engine, &[(0, &selection)]).unwrap();
    let payload_bytes = std::mem::size_of_val(warm.dense_data().unwrap());

    let mut output = None;
    let measurement = measure(payload_bytes, || {
        output = Some(black_box(
            source.restrict_leg(engine, &[(0, &selection)]).unwrap(),
        ));
    });
    assert_eq!(
        output.unwrap().dense_data().unwrap(),
        warm.dense_data().unwrap()
    );
    (measurement, payload_bytes)
}

/// Purpose: `restrict_leg` allocates its one output payload unfilled, and
/// everything else it allocates is independent of the degeneracies.
#[test]
fn restrict_leg_allocates_one_unfilled_payload_and_degeneracy_independent_scratch() {
    let _serial = crate::counting_alloc::serial();
    let (small, small_bytes) = restrict_measurement(1);
    let (large, large_bytes) = restrict_measurement(8);
    assert!(large_bytes > small_bytes);

    for (measurement, bytes) in [(&small, small_bytes), (&large, large_bytes)] {
        assert_eq!(
            (measurement.payloads, measurement.zeroed_payloads),
            (1, 0),
            "one unfilled payload-sized allocation, got {measurement:?} for {bytes} bytes"
        );
    }
    // Structural work does not grow with the degeneracy dimensions. The byte
    // budget pins the payload to one buffer and rules out a second
    // payload-sized buffer taken through plain `alloc`/`realloc`.
    assert_eq!(small.allocations, large.allocations);
    assert_eq!(
        small.bytes - small_bytes,
        large.bytes - large_bytes,
        "non-payload bytes must not depend on the degeneracy dimensions: \
         {small:?} vs {large:?} bytes for payloads {small_bytes}/{large_bytes}",
        small = small.bytes,
        large = large.bytes
    );
}

/// One restriction of all three legs of a tensor whose degeneracies are
/// `scale` times a base shape: the measurement and the payload byte count.
/// `sequential` restricts one leg per call instead, the negative control.
fn three_axis_measurement(scale: usize, sequential: bool) -> (Measurement, usize) {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = u1(
        &provider,
        &[(-1, 2 * scale), (0, 3 * scale), (1, 2 * scale)],
    );
    let dual = leg.try_dual().unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::rand_with_seed(engine, &runtime, [&leg, &dual], [&leg], 53).unwrap();
    let on_leg = LegSelection::try_new(
        &leg,
        [
            (U1Irrep::new(-1), 0..scale),
            (U1Irrep::new(0), scale..3 * scale),
        ],
    )
    .unwrap();
    let on_dual = LegSelection::try_new(
        &dual,
        [
            (U1Irrep::new(1), scale..2 * scale),
            (U1Irrep::new(0), 0..2 * scale),
        ],
    )
    .unwrap();
    let set = [(2, &on_leg), (0, &on_leg), (1, &on_dual)];
    let mut restrict = || {
        if sequential {
            set.iter().fold(source.clone(), |tensor, &pair| {
                tensor.restrict_leg(engine, &[pair]).unwrap()
            })
        } else {
            source.restrict_leg(engine, &set).unwrap()
        }
    };

    let warm = restrict();
    let payload_bytes = std::mem::size_of_val(warm.dense_data().unwrap());
    let mut output = None;
    let measurement = measure(payload_bytes, || {
        output = Some(black_box(restrict()));
    });
    assert_eq!(
        output.unwrap().dense_data().unwrap(),
        warm.dense_data().unwrap()
    );
    (measurement, payload_bytes)
}

#[test]
fn restricting_three_legs_is_one_payload_pass() {
    let _serial = crate::counting_alloc::serial();
    // What: `k` legs are restricted in one strided pass into one output
    // payload. Sequential one-leg passes would each allocate an
    // intermediate payload whose size grows with the degeneracies, so the
    // non-payload bytes would not stay fixed across scales.
    let (small, small_bytes) = three_axis_measurement(1, false);
    let (large, large_bytes) = three_axis_measurement(8, false);
    assert!(large_bytes > small_bytes);
    for (measurement, bytes) in [(&small, small_bytes), (&large, large_bytes)] {
        assert_eq!(
            measurement.zeroed_payloads, 0,
            "no payload-sized zeroed allocation, got {measurement:?} for {bytes} bytes"
        );
    }
    assert_eq!(small.allocations, large.allocations);
    assert_eq!(
        small.bytes - small_bytes,
        large.bytes - large_bytes,
        "only the output payload may grow with the degeneracies: \
         {small:?} vs {large:?} bytes for payloads {small_bytes}/{large_bytes}",
        small = small.bytes,
        large = large.bytes
    );

    // Negative control: the same three legs restricted one call at a time
    // allocate two intermediate payloads, so the assertion above would fail.
    let (small, small_bytes) = three_axis_measurement(1, true);
    let (large, large_bytes) = three_axis_measurement(8, true);
    assert_ne!(
        small.bytes - small_bytes,
        large.bytes - large_bytes,
        "sequential passes must show degeneracy-dependent non-payload bytes"
    );
}

#[test]
fn the_network_restriction_of_several_axes_fills_no_payload() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let zero = U1Irrep::new(0);
    let rows = u1(&provider, &[(0, 6)]);
    let columns = u1(&provider, &[(0, 8)]);
    let source: TensorMap<_, f64> =
        TensorMap::rand_with_seed(engine, &runtime, [&rows], [&columns], 43).unwrap();
    let zero_id = TypedSectorAdmission::try_encode_label(provider.as_ref(), &zero).unwrap();
    let restrictions = [
        NetworkDegeneracyRestriction {
            effective_axis: 0,
            authority_sector: zero_id,
            range: 1..4,
            partner: false,
        },
        NetworkDegeneracyRestriction {
            effective_axis: 1,
            authority_sector: zero_id,
            range: 2..6,
            partner: false,
        },
    ];

    let warm = __network::network_restrict_degeneracies(&source, false, &restrictions).unwrap();
    let payload_bytes = std::mem::size_of_val(warm.dense_data().unwrap());
    assert_eq!(warm.dense_data().unwrap().len(), 3 * 4);

    let mut output = None;
    let measurement = measure(payload_bytes, || {
        output = Some(black_box(
            __network::network_restrict_degeneracies(&source, false, &restrictions).unwrap(),
        ));
    });
    assert_eq!(
        output.unwrap().dense_data().unwrap(),
        warm.dense_data().unwrap()
    );
    assert_eq!(
        measurement.zeroed_payloads, 0,
        "two restricted axes write one unfilled payload, got {measurement:?}"
    );
    assert!(measurement.bytes >= payload_bytes);
}

/// One compact two-leg `restrict_leg` on an `s : bond <- bond` whose degeneracies
/// are `scale` times a base shape, keeping a fixed prefix of each sector.
fn compact_restrict_measurement(scale: usize) -> Measurement {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = u1(&provider, &[(-1, 2 * scale), (0, 3 * scale)]);
    let source: TensorMap<_, f64> =
        TensorMap::rand_with_seed(engine, &runtime, [&leg], [&leg], 47).unwrap();
    let Svd { s, .. } = source.svd_compact(engine, &[0], &[1]).unwrap();
    let bond = s.domain()[0].clone();
    let selection =
        LegSelection::try_new(&bond, [(U1Irrep::new(-1), 0..2), (U1Irrep::new(0), 0..3)]).unwrap();

    let warm = s
        .restrict_leg(engine, &[(0, &selection), (1, &selection)])
        .unwrap();
    assert!(
        symtn::expert::diagonal_spectrum(engine, &warm)
            .unwrap()
            .is_some(),
        "a compact receiver must stay compact"
    );

    let mut output = None;
    // No dense payload to watch: callers compare only calls and bytes.
    let measurement = measure(0, || {
        output = Some(black_box(
            s.restrict_leg(engine, &[(0, &selection), (1, &selection)])
                .unwrap(),
        ));
    });
    assert_eq!(
        symtn::expert::diagonal_spectrum(engine, &output.unwrap()).unwrap(),
        symtn::expert::diagonal_spectrum(engine, &warm).unwrap()
    );
    measurement
}

#[test]
fn compact_restrict_leg_costs_the_kept_values_and_nothing_per_discarded_one() {
    let _serial = crate::counting_alloc::serial();
    // The kept prefix is the same in both; only the discarded tail grows. A
    // compact restriction must therefore cost exactly the same, which is the
    // `O(sum_c k'_c)` contract: no dense block is ever materialized.
    let small = compact_restrict_measurement(1);
    let large = compact_restrict_measurement(16);
    assert_eq!(small.allocations, large.allocations);
    assert_eq!(small.bytes, large.bytes);
}

#[test]
fn lazy_adjoint_add_and_materialization_fill_no_payload() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: the owned outputs of a lazy-adjoint `add` and of the first
    // lazy-adjoint `data()` tile their storage, so neither zero-fills its
    // payload before overwriting every block. For `f64` the base
    // fill was `vec![0.0; len]`, an allocator-zeroed allocation.
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = u1(&provider, &[(-1, 2), (0, 3), (1, 2)]);
    let other = u1(&provider, &[(-1, 1), (0, 2), (1, 3)]);
    let square: TensorMap<_, f64> =
        TensorMap::rand_with_seed(engine, &runtime, [&leg, &other], [&leg, &other], 53).unwrap();
    let partner: TensorMap<_, f64> =
        TensorMap::rand_with_seed(engine, &runtime, [&leg, &other], [&leg, &other], 59).unwrap();
    let lazy = square.adjoint(engine).unwrap();
    let payload_bytes = std::mem::size_of_val(square.dense_data().unwrap());
    assert!(square.subblock_count() > 1);

    let mut sum = None;
    let add = measure(payload_bytes, || {
        sum = Some(black_box(lazy.axpby(engine, 0.5, &partner, -2.0).unwrap()));
    });
    assert_eq!(
        add.zeroed_payloads, 0,
        "lazy add zero-filled its payload: {add:?}"
    );

    let materialize = measure(payload_bytes, || {
        black_box(
            lazy.materialize(engine)
                .unwrap()
                .dense_data()
                .unwrap()
                .len(),
        );
    });
    assert_eq!(
        materialize.zeroed_payloads, 0,
        "lazy materialization zero-filled its payload: {materialize:?}"
    );
    // Oracle: the elementwise sum over the materialized adjoint payload,
    // which shares the partner's logical layout.
    let expected: Vec<u64> = lazy
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(partner.dense_data().unwrap())
        .map(|(&x, &y)| (0.5 * x + -2.0 * y).to_bits())
        .collect();
    let actual: Vec<u64> = sum
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .map(|x| x.to_bits())
        .collect();
    assert_eq!(actual, expected);
}

/// Steady-state `restrict_leg` on `[leg, other] <- [leg, other]`: every
/// coupled sector has several row trees, so the destination blocks are
/// strided sub-matrices; `other` carries charges `-reach..=reach`, so the
/// coupled-sector count grows with `reach`. Returns the measurement and
/// payload bytes of one warm call whose previous outputs were dropped.
fn steady_multi_row_restrict(scale: usize, reach: i32) -> (Measurement, usize) {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = u1(
        &provider,
        &[(-1, 2 * scale), (0, 3 * scale), (1, 2 * scale)],
    );
    let other = u1(
        &provider,
        &(-reach..=reach)
            .map(|charge| (charge, if charge == 0 { 2 } else { 1 }))
            .collect::<Vec<_>>(),
    );
    let source: TensorMap<_, f64> =
        TensorMap::rand_with_seed(engine, &runtime, [&leg, &other], [&leg, &other], 61).unwrap();
    let selection = LegSelection::try_new(
        &leg,
        [
            (U1Irrep::new(-1), 0..scale),
            (U1Irrep::new(0), scale..3 * scale),
        ],
    )
    .unwrap();
    let warm = source.restrict_leg(engine, &[(0, &selection)]).unwrap();
    let expected = warm.dense_data().unwrap().to_vec();
    let payload_bytes = std::mem::size_of_val(warm.dense_data().unwrap());
    drop(warm);
    drop(black_box(
        source.restrict_leg(engine, &[(0, &selection)]).unwrap(),
    ));
    let mut output = None;
    let measurement = measure(payload_bytes, || {
        output = Some(black_box(
            source.restrict_leg(engine, &[(0, &selection)]).unwrap(),
        ));
    });
    assert_eq!(output.unwrap().dense_data().unwrap(), expected.as_slice());
    (measurement, payload_bytes)
}

/// Purpose: a steady-state `restrict_leg` with several row trees per coupled
/// sector pays no per-call layout proof.
#[test]
fn steady_state_restrict_leg_with_several_row_trees_pays_no_layout_proof() {
    let _serial = crate::counting_alloc::serial();
    let (small, small_bytes) = steady_multi_row_restrict(1, 1);
    let (large, large_bytes) = steady_multi_row_restrict(4, 1);
    let (wide, wide_bytes) = steady_multi_row_restrict(1, 3);
    for (measurement, bytes) in [
        (&small, small_bytes),
        (&large, large_bytes),
        (&wide, wide_bytes),
    ] {
        assert_eq!(
            measurement.zeroed_payloads, 0,
            "no payload-sized zeroed allocation, got {measurement:?} for {bytes} bytes"
        );
    }
    // Proving the tiling on the per-call destination wrapper would cost per
    // coupled sector; the proof is a property recorded when the canonical
    // layout is built, so the count moves with neither the degeneracies nor
    // the coupled sectors.
    counting_alloc::assert_same_calls(
        "degeneracy scale 1 -> 4",
        small.allocations as u64,
        large.allocations as u64,
    );
    counting_alloc::assert_same_calls(
        "charge reach 1 -> 3",
        small.allocations as u64,
        wide.allocations as u64,
    );
}

/// One stacked `restrict_leg` over `members` members `[leg, other] <- [leg,
/// other]` (several row trees per coupled sector): the measurement of a warm
/// call and its payload bytes.
fn stacked_restrict_measurement(members: usize) -> (Measurement, usize) {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = u1(&provider, &[(-1, 2), (0, 3), (1, 2)]);
    let other = u1(&provider, &[(-1, 1), (0, 2), (1, 1)]);
    let tensors: Vec<TensorMap<_, f64>> = (0..members)
        .map(|seed| {
            TensorMap::rand_with_seed(
                engine,
                &runtime,
                [&leg, &other],
                [&leg, &other],
                67 + seed as u64,
            )
            .unwrap()
        })
        .collect();
    let stack = StackedTensorMap::pack(&tensors).unwrap();
    let selection =
        LegSelection::try_new(&leg, [(U1Irrep::new(-1), 0..1), (U1Irrep::new(0), 1..3)]).unwrap();
    let warm = stack.restrict_leg(engine, &[(0, &selection)]).unwrap();
    let payload_bytes =
        members * std::mem::size_of_val(warm.member(engine, 0).unwrap().dense_data().unwrap());
    drop(warm);
    let mut output = None;
    let measurement = measure(payload_bytes, || {
        output = Some(black_box(
            stack.restrict_leg(engine, &[(0, &selection)]).unwrap(),
        ));
    });
    let output = output.unwrap();
    for (index, tensor) in tensors.iter().enumerate() {
        assert_eq!(
            output.member(engine, index).unwrap().dense_data().unwrap(),
            tensor
                .restrict_leg(engine, &[(0, &selection)])
                .unwrap()
                .dense_data()
                .unwrap()
        );
    }
    (measurement, payload_bytes)
}

#[test]
fn stacked_restrict_leg_allocates_one_unfilled_payload_independent_of_the_member_count() {
    let _serial = crate::counting_alloc::serial();
    let (small, small_bytes) = stacked_restrict_measurement(2);
    let (large, large_bytes) = stacked_restrict_measurement(16);
    for (measurement, bytes) in [(&small, small_bytes), (&large, large_bytes)] {
        assert_eq!(
            measurement.zeroed_payloads, 0,
            "no payload-sized zeroed allocation, got {measurement:?} for {bytes} bytes"
        );
    }
    // One output buffer of `B L'` elements; everything else is the result's
    // structural objects, which do not depend on `B`.
    assert_eq!(small.allocations, large.allocations);
    assert_eq!(
        small.bytes - small_bytes,
        large.bytes - large_bytes,
        "non-payload bytes must not depend on the member count"
    );
}

/// One restriction and one embedding of a leg whose kept positions are every
/// other degeneracy index (`Σ_c ceil(n_c / 2)` runs, one per kept position,
/// the `eig` worst case: 4 at scale 1, 28 at scale 8) with degeneracies
/// `scale` times a base shape: the two warm measurements and the restricted
/// and embedded payload bytes.
fn alternate_runs_measurement(scale: usize) -> (Measurement, Measurement, usize, usize) {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = u1(
        &provider,
        &[(-1, 2 * scale), (0, 3 * scale), (1, 2 * scale)],
    );
    let other = u1(&provider, &[(-1, 1), (0, 2), (1, 1)]);
    let source: TensorMap<_, f64> =
        TensorMap::rand_with_seed(engine, &runtime, [&leg, &other], [&other], 71).unwrap();
    let every_other = |degeneracy: usize| (0..degeneracy).step_by(2).collect::<Vec<_>>();
    let selection = LegSelection::try_new(
        &leg,
        [
            (U1Irrep::new(-1), every_other(2 * scale)),
            (U1Irrep::new(0), every_other(3 * scale)),
            (U1Irrep::new(1), every_other(2 * scale)),
        ],
    )
    .unwrap();

    let warm = source.restrict_leg(engine, &[(0, &selection)]).unwrap();
    let warm_embedded = warm.embed_leg(engine, 0, &selection).unwrap();
    let payload_bytes = std::mem::size_of_val(warm.dense_data().unwrap());
    let mut output = None;
    let restrict = measure(payload_bytes, || {
        output = Some(black_box(
            source.restrict_leg(engine, &[(0, &selection)]).unwrap(),
        ));
    });
    assert_eq!(
        output.unwrap().dense_data().unwrap(),
        warm.dense_data().unwrap()
    );
    let embedded_bytes = std::mem::size_of_val(warm_embedded.dense_data().unwrap());
    let mut embedded = None;
    let embed = measure(embedded_bytes, || {
        embedded = Some(black_box(warm.embed_leg(engine, 0, &selection).unwrap()));
    });
    assert_eq!(
        embedded.unwrap().dense_data().unwrap(),
        warm_embedded.dense_data().unwrap()
    );
    (restrict, embed, payload_bytes, embedded_bytes)
}

/// Purpose: a selection's run count (4 -> 28 runs) costs no allocation in
/// the restriction or the embedding.
#[test]
fn a_many_run_selection_allocates_nothing_per_run() {
    let _serial = crate::counting_alloc::serial();
    // What: the run gather and scatter walk their pieces in place,
    // so the number of runs (here growing with the degeneracies) costs no
    // allocation: the restriction pays exactly what a contiguous one does,
    // and the embedding is independent of the run count.
    let (small_restrict, small_embed, small_bytes, small_embedded) = alternate_runs_measurement(1);
    let (large_restrict, large_embed, large_bytes, large_embedded) = alternate_runs_measurement(8);
    assert!(large_bytes > small_bytes);
    for (measurement, bytes) in [
        (&small_restrict, small_bytes),
        (&large_restrict, large_bytes),
    ] {
        assert_eq!(
            measurement.zeroed_payloads, 0,
            "no payload-sized zeroed allocation, got {measurement:?} for {bytes} bytes"
        );
    }
    assert_eq!(small_restrict.allocations, large_restrict.allocations);
    assert_eq!(
        small_restrict.bytes - small_bytes,
        large_restrict.bytes - large_bytes,
        "non-payload bytes must not depend on the run count"
    );
    assert_eq!(small_embed.allocations, large_embed.allocations);
    assert_eq!(
        small_embed.bytes - small_embedded,
        large_embed.bytes - large_embedded,
        "embedding non-payload bytes must not depend on the run count"
    );
}
