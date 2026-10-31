//! Real-device gates for `twist` in both directions on CUDA tensors
//! (issue #1330).
//!
//! A twist is not a tree transform: the Host builds a per-block ribbon-twist
//! factor at call time and scales a copy, so the evidence chain is not the
//! transform executor's. It is:
//!
//! 1. the value domain of the factor — `±1` for every provider this device
//!    impl admits — is gated without a device in `typed_transform_host_side.rs`,
//!    which ordinary CI runs;
//! 2. the lowering is gated here, by running the same public call on the Host
//!    and on the device; the Host side carries its own TensorKit and
//!    hand-computed oracles (`typed_facade.rs`, `semantic_suite.rs`,
//!    `checked_generic_braiding.rs`);
//! 3. the transfer, allocation and rejection contracts are in
//!    `typed_cuda_transform_contracts.rs`, which checks the thread-local
//!    transfer counters of the calling thread.
//!
//! Every fermionic fixture asserts non-vacuity: at least one block really is
//! negated, so a device path that copied its source would fail.
//!
//! Run with `cargo test -p symtn --features cuda,cpu-faer --test \
//! integration typed_cuda_twist:: -- --ignored` on a CUDA host.

#![cfg(feature = "cuda")]

use std::sync::Arc;
use symtn::typed::Direction;

use num_complex::Complex64;
use symtn::sector::{
    product_sector, FermionParityFusionRule, SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep,
    Z2Irrep,
};
use symtn::typed::{GradedSpace, TensorMap};
use symtn::typed::{Runtime, TensorScalar};

trait Payload: TensorScalar + Copy + PartialEq + std::fmt::Debug {
    fn negated(self) -> Self;
}

impl Payload for f64 {
    fn negated(self) -> Self {
        -self
    }
}

impl Payload for Complex64 {
    fn negated(self) -> Self {
        -self
    }
}

/// Runs one twist on the Host and on the device and asserts the device answer
/// is the Host answer exactly: every admitted factor is `±1`, so a device
/// result that differs in a single bit differs semantically.
macro_rules! device_matches_host {
    ($engine:ident, $cuda:ident, $what:expr, $host:expr, |$e:ident, $t:ident| $body:expr) => {{
        let host_source = &$host;
        let device_source = host_source.to_cuda($cuda).unwrap();
        let expected = {
            let $e: &mut dyn symtn::typed::Engine = &mut *$engine;
            let $t = host_source;
            $body
        }
        .unwrap();
        let actual = {
            let $e: &mut dyn symtn::typed::Engine = &mut *$cuda;
            let $t = &device_source;
            $body
        }
        .unwrap()
        .to_host($cuda)
        .unwrap();
        let what: &str = $what;
        assert_eq!(
            actual.dense_data().unwrap(),
            expected.dense_data().unwrap(),
            "{what}: payload"
        );
        assert_eq!(actual.rank(), expected.rank(), "{what}: rank");
        assert_eq!(
            actual.codomain_rank(),
            expected.codomain_rank(),
            "{what}: split"
        );
        actual
    }};
}

/// Non-vacuity: the twist negated at least one entry, and negated *nothing
/// else* — the `±1` value domain the descriptor-scale lowering relies on.
fn assert_signs_only<D: Payload>(source: &[D], twisted: &[D], what: &str) {
    assert_eq!(source.len(), twisted.len(), "{what}: length");
    let mut negated = 0usize;
    for (index, (&before, &after)) in source.iter().zip(twisted).enumerate() {
        if after == before {
            continue;
        }
        assert_eq!(
            after,
            before.negated(),
            "{what}: element {index} is neither kept nor negated"
        );
        negated += 1;
    }
    assert!(
        negated > 0,
        "{what}: no entry changed sign, so the fixture proves nothing"
    );
}

fn runtime() -> Runtime {
    Runtime::builder().build().unwrap()
}

fn real_fill(_trees: &symtn::typed::BlockFusionTrees<impl std::fmt::Debug>, idx: &[usize]) -> f64 {
    let mut value = 0.37;
    for (axis, &index) in idx.iter().enumerate() {
        value = value * 1.7 + (index as f64 + 1.0) * (axis as f64 + 2.0);
    }
    value
}

fn complex_fill(
    trees: &symtn::typed::BlockFusionTrees<impl std::fmt::Debug>,
    idx: &[usize],
) -> Complex64 {
    let re = real_fill(trees, idx);
    Complex64::new(re, -0.5 * re + 0.25)
}

fn fz2_leg(dual: bool) -> GradedSpace<FermionParityFusionRule> {
    let space = GradedSpace::try_new(
        Arc::new(FermionParityFusionRule),
        [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 1)],
    )
    .unwrap();
    if dual {
        space.try_dual().unwrap()
    } else {
        space
    }
}

fn u1_leg() -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 1),
            (U1Irrep::new(1), 2),
        ],
    )
    .unwrap()
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_twist_matches_the_host_on_fermion_parity() {
    let cuda = &mut crate::common::cuda_engine();
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = runtime();
    let leg = fz2_leg(false);
    let dual = fz2_leg(true);
    // A dual leg in the codomain and a plain one in the domain, so the leg
    // indices below reach both sides and both dualities.
    let real: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg, &dual], [&leg, &leg], real_fill)
            .unwrap();
    let complex: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg, &dual], [&leg, &leg], complex_fill)
            .unwrap();
    assert!(real.subblock_count() >= 2, "multi-block fixture");

    // Single codomain leg, single dual codomain leg, single domain leg.
    for legs in [&[0usize][..], &[1][..], &[3][..]] {
        let what = format!("fZ2/f64 twist {legs:?}");
        let twisted = device_matches_host!(engine, cuda, &what, real, |e, t| t.twist(
            e,
            legs,
            Direction::Forward
        ));
        assert_signs_only(
            real.dense_data().unwrap(),
            twisted.dense_data().unwrap(),
            &what,
        );
    }
    // Multi-axis, across the split and repeating both sides.
    for legs in [&[0usize, 3][..], &[1, 2, 3][..], &[0, 1, 1, 3][..]] {
        let what = format!("fZ2/f64 twist {legs:?}");
        let twisted = device_matches_host!(engine, cuda, &what, real, |e, t| t.twist(
            e,
            legs,
            Direction::Forward
        ));
        assert_signs_only(
            real.dense_data().unwrap(),
            twisted.dense_data().unwrap(),
            &what,
        );
    }
    let what = "fZ2/c64 twist [1, 2]";
    let twisted = device_matches_host!(engine, cuda, what, complex, |e, t| t.twist(
        e,
        &[1, 2],
        Direction::Forward
    ));
    assert_signs_only(
        complex.dense_data().unwrap(),
        twisted.dense_data().unwrap(),
        what,
    );

    // the inverse `twist` is the conjugate factor, which is the same `±1` here —
    // gated against the Host rather than assumed.
    let what = "fZ2/f64 inverse twist [0, 3]";
    let inverted = device_matches_host!(engine, cuda, what, real, |e, t| t.twist(
        e,
        &[0, 3],
        Direction::Inverse
    ));
    assert_signs_only(
        real.dense_data().unwrap(),
        inverted.dense_data().unwrap(),
        what,
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_twist_matches_the_host_for_product_providers() {
    let cuda = &mut crate::common::cuda_engine();
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = runtime();

    // fZ2 x U(1): fermionic signs on top of a charge grading.
    let rule = Arc::new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule)
            .unwrap(),
    );
    let leg = GradedSpace::try_new(
        Arc::clone(&rule),
        [
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
            (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 1),
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(2)), 1),
        ],
    )
    .unwrap();
    let host: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg], [&leg, &leg], real_fill)
            .unwrap();
    for legs in [&[1usize][..], &[2][..], &[0, 2][..]] {
        let what = format!("fZ2xU1 twist {legs:?}");
        let twisted = device_matches_host!(engine, cuda, &what, host, |e, t| t.twist(
            e,
            legs,
            Direction::Forward
        ));
        assert_signs_only(
            host.dense_data().unwrap(),
            twisted.dense_data().unwrap(),
            &what,
        );
    }

    // fZ2 (x) SU(2): fermionic signs with non-Abelian degeneracies.
    let rule = Arc::new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, SU2FusionRule)
            .unwrap(),
    );
    let leg = GradedSpace::try_new(
        Arc::clone(&rule),
        [
            (
                product_sector(Z2Irrep::EVEN, SU2Irrep::from_twice_spin(0)),
                2,
            ),
            (
                product_sector(Z2Irrep::ODD, SU2Irrep::from_twice_spin(1)),
                1,
            ),
            (
                product_sector(Z2Irrep::EVEN, SU2Irrep::from_twice_spin(2)),
                1,
            ),
        ],
    )
    .unwrap();
    let host: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg], [&leg, &leg], complex_fill)
            .unwrap();
    for legs in [&[0usize][..], &[3][..], &[1, 3][..]] {
        let what = format!("fZ2xSU2 twist {legs:?}");
        let twisted = device_matches_host!(engine, cuda, &what, host, |e, t| t.twist(
            e,
            legs,
            Direction::Forward
        ));
        assert_signs_only(
            host.dense_data().unwrap(),
            twisted.dense_data().unwrap(),
            &what,
        );
    }
    let what = "fZ2xSU2 inverse twist [1, 3]";
    let inverted = device_matches_host!(engine, cuda, what, host, |e, t| t.twist(
        e,
        &[1, 3],
        Direction::Inverse
    ));
    assert_signs_only(
        host.dense_data().unwrap(),
        inverted.dense_data().unwrap(),
        what,
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_twist_round_trips_bitwise_and_short_circuits() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut crate::common::cuda_engine();
    let runtime = runtime();
    let leg = fz2_leg(false);
    let host: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg], [&leg, &leg], complex_fill)
            .unwrap();
    let device = host.to_cuda(cuda).unwrap();

    // twist ∘ inverse twist == id, bitwise: both factors are exactly ±1.
    let round_trip = device
        .twist(cuda, &[0, 2, 3], Direction::Forward)
        .unwrap()
        .twist(cuda, &[0, 2, 3], Direction::Inverse)
        .unwrap()
        .to_host(cuda)
        .unwrap();
    assert_eq!(
        round_trip.dense_data().unwrap(),
        host.dense_data().unwrap(),
        "twist round trip is exact"
    );
    let twisted = device
        .twist(cuda, &[0, 2, 3], Direction::Forward)
        .unwrap()
        .to_host(cuda)
        .unwrap();
    assert_signs_only(
        host.dense_data().unwrap(),
        twisted.dense_data().unwrap(),
        "twist [0, 2, 3]",
    );

    // An empty leg list is a clone on both sides.
    let empty = device
        .twist(cuda, &[], Direction::Forward)
        .unwrap()
        .to_host(cuda)
        .unwrap();
    assert_eq!(
        empty.dense_data().unwrap(),
        host.dense_data().unwrap(),
        "empty twist is a clone"
    );

    // A bosonic provider twists by 1 on every block: a clone, never a scale.
    let u1 = u1_leg();
    let bosonic: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&u1, &u1], [&u1, &u1], real_fill).unwrap();
    let bosonic_device = bosonic.to_cuda(cuda).unwrap();
    for legs in [&[0usize][..], &[1, 3][..]] {
        assert_eq!(
            bosonic_device
                .twist(cuda, legs, Direction::Forward)
                .unwrap()
                .to_host(cuda)
                .unwrap()
                .dense_data()
                .unwrap(),
            bosonic.dense_data().unwrap(),
            "bosonic twist {legs:?} must be a clone"
        );
        assert_eq!(
            bosonic_device
                .twist(cuda, legs, Direction::Inverse)
                .unwrap()
                .to_host(cuda)
                .unwrap()
                .dense_data()
                .unwrap(),
            bosonic.dense_data().unwrap(),
            "bosonic inverse twist {legs:?} must be a clone"
        );
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_twist_on_a_lazy_adjoint_matches_the_host() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut crate::common::cuda_engine();
    let runtime = runtime();
    let leg = fz2_leg(false);
    let dual = fz2_leg(true);
    let host: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg, &dual], [&leg], complex_fill).unwrap();
    let device = host.to_cuda(cuda).unwrap();

    for legs in [&[0usize][..], &[2][..], &[0, 1, 2][..]] {
        let expected = host
            .adjoint(engine)
            .unwrap()
            .twist(engine, legs, Direction::Forward)
            .unwrap();
        let actual = device
            .adjoint(cuda)
            .unwrap()
            .twist(cuda, legs, Direction::Forward)
            .unwrap()
            .to_host(cuda)
            .unwrap();
        assert_eq!(
            actual.materialize(engine).unwrap().dense_data().unwrap(),
            expected.materialize(engine).unwrap().dense_data().unwrap(),
            "adjoint twist {legs:?}"
        );
        assert_eq!(
            actual.rank(),
            expected.rank(),
            "adjoint twist {legs:?} rank"
        );
        let inverse_expected = host
            .adjoint(engine)
            .unwrap()
            .twist(engine, legs, Direction::Inverse)
            .unwrap();
        let inverse_actual = device
            .adjoint(cuda)
            .unwrap()
            .twist(cuda, legs, Direction::Inverse)
            .unwrap()
            .to_host(cuda)
            .unwrap();
        assert_eq!(
            inverse_actual
                .materialize(engine)
                .unwrap()
                .dense_data()
                .unwrap(),
            inverse_expected
                .materialize(engine)
                .unwrap()
                .dense_data()
                .unwrap(),
            "adjoint inverse twist {legs:?}"
        );
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_twist_handles_a_space_with_no_coupled_sector_and_rejects_a_leg_past_the_rank() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut crate::common::cuda_engine();
    let runtime = runtime();
    // No coupled sector: zero blocks, zero elements. With no block the
    // identity detection is vacuously true, so this short-circuits to a clone
    // rather than reaching the zero-length upload — it proves the degenerate
    // space is handled, not that path.
    let even =
        GradedSpace::try_new(Arc::new(FermionParityFusionRule), [(Z2Irrep::EVEN, 2)]).unwrap();
    let odd = GradedSpace::try_new(Arc::new(FermionParityFusionRule), [(Z2Irrep::ODD, 1)]).unwrap();
    let empty: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&even], [&odd], real_fill).unwrap();
    assert!(empty.dense_data().unwrap().is_empty(), "empty fixture");
    let device = empty.to_cuda(cuda).unwrap();
    assert!(device
        .twist(cuda, &[0, 1], Direction::Forward)
        .unwrap()
        .to_host(cuda)
        .unwrap()
        .dense_data()
        .unwrap()
        .is_empty());

    // The range check precedes everything, on both sides, with the Host text.
    let leg = fz2_leg(false);
    let host: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], real_fill).unwrap();
    let device = host.to_cuda(cuda).unwrap();
    assert_eq!(
        device.twist(cuda, &[2], Direction::Forward).unwrap_err(),
        host.twist(engine, &[2], Direction::Forward).unwrap_err(),
    );
    assert_eq!(
        device.twist(cuda, &[7], Direction::Inverse).unwrap_err(),
        host.twist(engine, &[7], Direction::Inverse).unwrap_err(),
    );
}
