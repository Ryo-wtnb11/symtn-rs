//! Allocation gates for the typed `cat`/`braid`/`absorb`, alongside the compact-storage gates in
//! `typed_diagonal_allocations.rs`: bytes counted through a global allocator
//! while one warmed operation runs.
//!
//! The compact diagonal gates require one dense output allocation and no
//! source-sized dense intermediate per compact operand.

use std::hint::black_box;
use std::sync::Arc;
use symtn::typed::Side;

use symtn::sector::{U1FusionRule, U1Irrep, Z2FusionRule, Z2Irrep};
use symtn::typed::{Complex64, Runtime};
use symtn::typed::{GradedSpace, SectorSpectrum, TensorMap};

use crate::counting_alloc;

fn measured_allocations<T>(payload_bytes: usize, operation: impl FnOnce() -> T) -> (u64, usize) {
    let (output, allocs) =
        counting_alloc::measure_matching(payload_bytes..=payload_bytes, || black_box(operation()));
    black_box(output);
    (allocs.bytes, allocs.matched_calls as usize)
}

/// Small fixed allowance for descriptor/layout bookkeeping (copy plans,
/// derived-space metadata).
const STRUCTURAL_TOLERANCE: u64 = 128 * 1024;

fn u1_leg(provider: &Arc<U1FusionRule>, pairs: &[(i32, usize)]) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::clone(provider),
        pairs
            .iter()
            .map(|&(charge, degeneracy)| (U1Irrep::new(charge), degeneracy)),
    )
    .unwrap()
}

fn pseudo_random(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 33) as f64) / (u32::MAX as f64) - 0.5
}

#[test]
fn typed_cat_uses_one_output_allocation_without_scratch() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: a warmed typed catdomain owns exactly one output-sized payload
    // and no per-block allocations — the plan executes into the single final
    // buffer.
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let codomain = u1_leg(&provider, &[(0, 127)]);
    let left = u1_leg(&provider, &[(0, 251)]);
    let right = u1_leg(&provider, &[(0, 263)]);
    let mut state = 0x5eed_0580u64;
    let lhs: TensorMap<U1FusionRule, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&codomain], [&left], |_, _| {
            pseudo_random(&mut state)
        })
        .unwrap();
    let mut state = 0x5eed_0581u64;
    let rhs: TensorMap<U1FusionRule, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&codomain], [&right], |_, _| {
            pseudo_random(&mut state)
        })
        .unwrap();
    let warm: TensorMap<U1FusionRule, f64> = lhs.cat(engine, &rhs, Side::Domain).unwrap();
    let output_payload = std::mem::size_of_val(warm.dense_data().unwrap());

    let (allocated, payload_allocations) = measured_allocations(output_payload, || {
        lhs.cat(engine, &rhs, Side::Domain).unwrap()
    });

    assert_eq!(payload_allocations, 1);
    assert!(
        allocated <= output_payload as u64 + STRUCTURAL_TOLERANCE,
        "typed cat allocated {allocated} B for a {output_payload} B output"
    );
}

#[test]
fn typed_absorb_clones_the_destination_once() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: a warmed typed absorb allocates the destination clone and nothing
    // per block — the merge-join writes prefixes in place.
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let mut state = 0x5eed_0582u64;
    let destination: TensorMap<U1FusionRule, f64> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&u1_leg(&provider, &[(0, 251)])],
        [&u1_leg(&provider, &[(0, 127)])],
        |_, _| pseudo_random(&mut state),
    )
    .unwrap();
    let mut state = 0x5eed_0583u64;
    let source: TensorMap<U1FusionRule, f64> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&u1_leg(&provider, &[(0, 263)])],
        [&u1_leg(&provider, &[(0, 101)])],
        |_, _| pseudo_random(&mut state),
    )
    .unwrap();
    black_box(destination.absorb(engine, &source).unwrap());
    let output_payload = std::mem::size_of_val(destination.dense_data().unwrap());

    let (allocated, payload_allocations) = measured_allocations(output_payload, || {
        destination.absorb(engine, &source).unwrap()
    });

    assert_eq!(payload_allocations, 1);
    assert!(
        allocated <= output_payload as u64 + STRUCTURAL_TOLERANCE,
        "typed absorb allocated {allocated} B for a {output_payload} B destination clone"
    );
}

#[test]
fn typed_cat_reads_compact_operands_without_source_sized_buffers() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    const DEGENERACY: usize = 128;
    let runtime = Runtime::builder().build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(Z2FusionRule), [(Z2Irrep::EVEN, DEGENERACY)]).unwrap();
    let mut state = 0x5eed_0584u64;
    let tensor: TensorMap<Z2FusionRule, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, _| {
            pseudo_random(&mut state)
        })
        .unwrap();
    // Warm every layout cache with a throwaway spectrum factor.
    let warmup: TensorMap<Z2FusionRule, f64> = tensor.svd_compact(engine, &[0], &[1]).unwrap().s;
    black_box(warmup.cat(engine, &warmup, Side::Domain).unwrap());

    let s: TensorMap<Z2FusionRule, f64> = tensor.svd_compact(engine, &[0], &[1]).unwrap().s;
    let dense_payload = DEGENERACY * DEGENERACY * std::mem::size_of::<f64>();
    let output_payload = 2 * dense_payload;
    let expected = output_payload as u64;

    for call in ["first", "second"] {
        let (bytes, payload_allocations) = measured_allocations(output_payload, || {
            black_box(s.cat(engine, &s, Side::Domain).unwrap());
        });
        assert_eq!(payload_allocations, 1, "{call} cat");
        assert!(
            (expected..=expected + STRUCTURAL_TOLERANCE).contains(&bytes),
            "{call} cat allocated {bytes} B; expected one owned output ({expected} B)"
        );
    }
}

#[test]
fn typed_braid_and_absorb_allocate_only_one_dense_output_for_compact_input() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    const DEGENERACY: usize = 128;
    let runtime = Runtime::builder().build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(Z2FusionRule), [(Z2Irrep::EVEN, DEGENERACY)]).unwrap();
    let diagonal: TensorMap<_, f64> = TensorMap::diagonal(
        engine,
        &runtime,
        &leg,
        [symtn::typed::SectorSpectrum {
            sector: Z2Irrep::EVEN,
            values: vec![2.0; DEGENERACY],
        }],
    )
    .unwrap();
    black_box(diagonal.braid(engine, &[1], &[0], &[0, 1]).unwrap());
    black_box(diagonal.absorb(engine, &diagonal).unwrap());
    let payload = DEGENERACY * DEGENERACY * std::mem::size_of::<f64>();
    let mut counts = [0; 2];
    for (operation, name) in ["braid", "absorb"].into_iter().enumerate() {
        let (bytes, payload_allocations) = measured_allocations(payload, || {
            if operation == 0 {
                black_box(diagonal.braid(engine, &[1], &[0], &[0, 1]).unwrap());
            } else {
                black_box(diagonal.absorb(engine, &diagonal).unwrap());
            }
        });
        counts[operation] = payload_allocations;
        assert!(
            bytes >= payload as u64,
            "{name} allocated fewer than the required output: {bytes} B"
        );
    }
    assert_eq!(counts, [1, 1], "braid/absorb P-sized buffer counts");
}

#[test]
fn typed_absorb_smaller_compact_source_avoids_its_dense_buffer() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(Z2FusionRule);
    let wide = GradedSpace::try_new(Arc::clone(&provider), [(Z2Irrep::EVEN, 128)]).unwrap();
    let narrow = GradedSpace::try_new(provider, [(Z2Irrep::EVEN, 64)]).unwrap();
    let receiver: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&wide], [&wide], |_, index| {
            1.0 + index[0] as f64 + index[1] as f64
        })
        .unwrap();
    let source: TensorMap<_, f64> = TensorMap::diagonal(
        engine,
        &runtime,
        &narrow,
        [symtn::typed::SectorSpectrum {
            sector: Z2Irrep::EVEN,
            values: vec![2.0; 64],
        }],
    )
    .unwrap();
    black_box(receiver.absorb(engine, &source).unwrap());
    let output_bytes = 128 * 128 * std::mem::size_of::<f64>();
    let source_bytes = 64 * 64 * std::mem::size_of::<f64>();
    let (allocated, output_allocations) =
        measured_allocations(output_bytes, || receiver.absorb(engine, &source).unwrap());
    assert_eq!(output_allocations, 1);
    assert!(
        allocated < (output_bytes + source_bytes / 2) as u64,
        "absorb allocated {allocated} B; a compact source must not add its {source_bytes} B dense buffer"
    );
}

#[test]
fn typed_lazy_adjoint_cat_allocates_only_the_output_payload() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let common = u1_leg(&provider, &[(0, 64)]);
    let left = u1_leg(&provider, &[(0, 192)]);
    let right = u1_leg(&provider, &[(0, 320)]);
    let lhs_parent: TensorMap<U1FusionRule, Complex64> =
        TensorMap::rand_with_seed(engine, &runtime, [&left], [&common], 388_001).unwrap();
    let rhs_parent: TensorMap<U1FusionRule, Complex64> =
        TensorMap::rand_with_seed(engine, &runtime, [&right], [&common], 388_002).unwrap();

    let warm_lhs = lhs_parent.adjoint(engine).unwrap();
    let warm_rhs = rhs_parent.adjoint(engine).unwrap();
    let warm_output = warm_lhs.cat(engine, &warm_rhs, Side::Domain).unwrap();
    let output_payload = std::mem::size_of_val(warm_output.dense_data().unwrap());
    let input_payload = (lhs_parent.dense_data().unwrap().len()
        + rhs_parent.dense_data().unwrap().len()) as u64
        * std::mem::size_of::<Complex64>() as u64;

    let fast_lhs = lhs_parent.adjoint(engine).unwrap();
    let fast_rhs = rhs_parent.adjoint(engine).unwrap();
    let (fast_bytes, payload_allocations) = measured_allocations(output_payload, || {
        fast_lhs.cat(engine, &fast_rhs, Side::Domain).unwrap()
    });

    let eager_lhs = lhs_parent.adjoint(engine).unwrap();
    let eager_rhs = rhs_parent.adjoint(engine).unwrap();
    let (eager_bytes, _) = measured_allocations(output_payload, || {
        let lhs = eager_lhs.materialize(engine).unwrap();
        let rhs = eager_rhs.materialize(engine).unwrap();
        lhs.cat(engine, &rhs, Side::Domain).unwrap()
    });

    assert_eq!(payload_allocations, 1);
    assert!(
        fast_bytes >= output_payload as u64,
        "lazy cat allocated {fast_bytes} B, below its {output_payload} B owned output"
    );
    assert!(
        fast_bytes <= output_payload as u64 + STRUCTURAL_TOLERANCE,
        "lazy cat allocated {fast_bytes} B for a {output_payload} B output"
    );
    assert!(
        eager_bytes + STRUCTURAL_TOLERANCE >= fast_bytes + input_payload,
        "lazy cat saved {} B, below the {input_payload} B input payloads outside the \
         {STRUCTURAL_TOLERANCE} B structural allowance",
        eager_bytes.saturating_sub(fast_bytes)
    );
}

/// Warm allocation calls of one compact-diagonal braid on `$leg`.
macro_rules! warm_compact_braid_calls {
    ($engine:ident, $leg:expr, $spectrum:expr $(,)?) => {{
        let runtime = Runtime::builder().build().unwrap();
        let diagonal: TensorMap<_, f64> =
            TensorMap::diagonal($engine, &runtime, $leg, $spectrum).unwrap();
        black_box(diagonal.braid($engine, &[1], &[0], &[0, 1]).unwrap());
        black_box(diagonal.braid($engine, &[1], &[0], &[0, 1]).unwrap());
        let (output, allocs) = counting_alloc::measure(|| {
            black_box(diagonal.braid($engine, &[1], &[0], &[0, 1]).unwrap())
        });
        assert!(output.dense_data().is_ok());
        allocs.calls
    }};
}

#[test]
fn compact_braid_to_dual_sectors_allocates_like_the_self_dual_braid() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What (#2011): the compact braid sends sector c to its dual. For U(1)
    // charges ±1 that moves each destination to the other ordinal, and the
    // destination block is still found without an allocation, so the warm
    // call costs exactly what the self-dual Z2 braid of the same shape does.
    let z2 = GradedSpace::try_new(
        Arc::new(Z2FusionRule),
        [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 3)],
    )
    .unwrap();
    let self_dual = warm_compact_braid_calls!(
        engine,
        &z2,
        vec![
            SectorSpectrum {
                sector: Z2Irrep::EVEN,
                values: vec![1.0, 2.0],
            },
            SectorSpectrum {
                sector: Z2Irrep::ODD,
                values: vec![3.0, 4.0, 5.0],
            },
        ],
    );
    let u1 = u1_leg(&Arc::new(U1FusionRule), &[(-1, 2), (1, 3)]);
    let dual_moving = warm_compact_braid_calls!(
        engine,
        &u1,
        vec![
            SectorSpectrum {
                sector: U1Irrep::new(-1),
                values: vec![1.0, 2.0],
            },
            SectorSpectrum {
                sector: U1Irrep::new(1),
                values: vec![3.0, 4.0, 5.0],
            },
        ],
    );
    eprintln!("warm compact braid calls: Z2 {self_dual}, U(1) ±1 {dual_moving}");
    assert_eq!(dual_moving, self_dual);
}
