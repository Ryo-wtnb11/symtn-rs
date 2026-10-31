//! Allocation contract of the owned eager contraction output (#1212): the
//! payload is one allocator-zeroed allocation of `P * size_of::<D>()` bytes
//! for both dtypes, and nothing else on the warmed path allocates zeroed
//! memory or re-touches the inactive blocks.

use std::hint::black_box;
use std::sync::Arc;
use symtn::typed::ContractSpec;

use num_complex::{Complex32, Complex64};
use symtn::sector::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use symtn::typed::Runtime;
use symtn::typed::{GradedSpace, TensorMap, TensorScalar};

use crate::counting_alloc;

/// Allocations of `f` on this thread; `matched_zeroed_calls` counts the
/// allocator-zeroed ones of exactly `payload_bytes`.
fn measure(payload_bytes: usize, f: impl FnOnce()) -> counting_alloc::Allocs {
    counting_alloc::measure_matching(payload_bytes..=payload_bytes, f).1
}

fn u1_space(provider: &Arc<U1FusionRule>, sectors: &[(i32, usize)]) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::clone(provider),
        sectors
            .iter()
            .map(|&(charge, degeneracy)| (U1Irrep::new(charge), degeneracy)),
    )
    .unwrap()
}

fn su2_space(
    provider: &Arc<SU2FusionRule>,
    sectors: &[(usize, usize)],
) -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        Arc::clone(provider),
        sectors
            .iter()
            .map(|&(twice_spin, degeneracy)| (SU2Irrep::from_twice_spin(twice_spin), degeneracy)),
    )
    .unwrap()
}

/// The contracted bond carries only charge 0, so the destination `[a] <- [c]`
/// has coupled charges -1 and 1 that no GEMM writes (inactive blocks).
fn u1_contract_measurement<D: TensorScalar + std::fmt::Debug>(seed: u64) {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let open = u1_space(&provider, &[(-1, 2), (0, 3), (1, 2)]);
    let bond = u1_space(&provider, &[(0, 2)]);
    let lhs: TensorMap<_, D> =
        TensorMap::rand_with_seed(engine, &runtime, [&open], [&bond], seed).unwrap();
    let rhs: TensorMap<_, D> =
        TensorMap::rand_with_seed(engine, &runtime, [&bond], [&open], seed + 1).unwrap();
    let warm = lhs
        .contract(
            engine,
            &rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    let payload_len = warm.dense_data().unwrap().len();
    assert_eq!(payload_len, 2 * 2 + 3 * 3 + 2 * 2);
    assert_eq!(warm.subblock_count(), 3);

    let payload_bytes = std::mem::size_of_val(warm.dense_data().unwrap());
    let mut output = None;
    let measurement = measure(payload_bytes, || {
        output = Some(black_box(
            lhs.contract(
                engine,
                &rhs,
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &[0],
                    domain: &[1],
                },
            )
            .unwrap(),
        ));
    });
    let output = output.unwrap();
    assert_eq!(output.dense_data().unwrap(), warm.dense_data().unwrap());

    assert_eq!(
        measurement.matched_zeroed_calls,
        1,
        "the owned output must be exactly one allocator-zeroed payload of {payload_bytes} bytes; got {:?}",
        measurement
    );
}

/// Purpose: a warm owned f64 U(1) contraction allocates its output as one
/// allocator-zeroed payload.
#[test]
fn owned_f64_contraction_output_is_one_allocator_zeroed_payload() {
    let _serial = crate::counting_alloc::serial();
    u1_contract_measurement::<f64>(1_212_001);
}

/// Purpose: the Complex64 twin of the contract above.
#[test]
fn owned_c64_contraction_output_is_one_allocator_zeroed_payload() {
    let _serial = crate::counting_alloc::serial();
    u1_contract_measurement::<Complex64>(1_212_002);
}

/// SU(2) compose: non-abelian recoupling, so this is the fixture where the
/// coefficient scratch is converted per structure identity.
fn su2_compose_measurement<D: TensorScalar>(seed: u64) -> u64 {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(SU2FusionRule);
    // Destination `[a, c] <- [d, e]` couples to spins 1/2 and 3/2; the bond
    // `b` carries only spin 1/2, so the spin-3/2 block is inactive.
    let a = su2_space(&provider, &[(0, 2), (2, 3)]);
    let c = su2_space(&provider, &[(1, 2)]);
    let b = su2_space(&provider, &[(1, 3)]);
    let lhs: TensorMap<_, D> =
        TensorMap::rand_with_seed(engine, &runtime, [&a, &c], [&b], seed).unwrap();
    let rhs: TensorMap<_, D> =
        TensorMap::rand_with_seed(engine, &runtime, [&b], [&a, &c], seed + 1).unwrap();
    let warm = lhs.compose(engine, &rhs).unwrap();
    let payload_bytes = std::mem::size_of_val(warm.dense_data().unwrap());
    let measurement = measure(payload_bytes, || {
        black_box(lhs.compose(engine, &rhs).unwrap());
    });
    assert_eq!(
        measurement.matched_zeroed_calls, 1,
        "the owned output must be exactly one allocator-zeroed payload of \
         {payload_bytes} bytes; got {:?}",
        measurement
    );
    measurement.calls
}

/// Purpose: a warm owned SU(2) compose with an inactive coupled sector
/// allocates its output as one allocator-zeroed payload.
#[test]
fn owned_su2_contraction_with_inactive_sectors_is_one_allocator_zeroed_payload() {
    let _serial = crate::counting_alloc::serial();
    su2_compose_measurement::<f64>(1_212_003);
}

/// #1315: a single-precision payload must run the same algorithm on the
/// recoupling path, not a different one. Same fixture, same call count; only
/// the zeroed payload is half the size, which
/// `zeroed_allocations_of(size_of_val(data))` checks per dtype.
#[test]
fn owned_su2_single_precision_contraction_allocates_like_double() {
    let _serial = crate::counting_alloc::serial();
    let double = su2_compose_measurement::<f64>(1_315_003);
    let single = su2_compose_measurement::<f32>(1_315_003);
    assert_eq!(
        single, double,
        "f32 SU(2) compose allocated {single} times against {double} for f64"
    );
    let complex_double = su2_compose_measurement::<Complex64>(1_315_005);
    let complex_single = su2_compose_measurement::<Complex32>(1_315_005);
    assert_eq!(complex_single, complex_double);
}
