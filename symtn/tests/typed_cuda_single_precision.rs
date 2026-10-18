//! Device twins of the `f64`/`Complex64` base-family device suites, run at
//! every admitted device payload (#1336).
//!
//! Each gate here is one generic body instantiated for all four dtypes, so the
//! double-precision instantiation is the control: a helper that silently did
//! nothing would fail against `f64` first. The oracle is always the **host**
//! result *at the same payload dtype* — never the `f64` host result — so what
//! these prove is that the device agrees with symtn's own host semantics at
//! the payload's own precision. The host single-precision semantics themselves
//! are gated independently by `symtn/tests/single_precision_base.rs` (#1315).
//!
//! Tolerances are stated per family, not shared:
//!
//! * transfer is **bit exact** at every dtype — a byte-for-byte round trip;
//! * elementwise arithmetic and the structural transforms are compared at
//!   `8 * sqrt(n) * eps(real(D)) * scale`, `n` the number of terms combined;
//! * the reductions accumulate in `f64` on the device like the host
//!   (`weighted_inner_cuda`; #1344 for `norm`, #1383 for `inner`), so
//!   `inner` is compared under the workspace rule of
//!   `docs/testing_numerics.md`; the range cases are pinned by
//!   `device_norm_matches_the_host_where_a_payload_sum_would_overflow_or_underflow`
//!   and `device_inner_matches_the_host_where_single_precision_products_overflow`.
//!
//! Fixture entries are dyadic rationals, so every fixture is exactly
//! representable in `f32` and no fixture value is itself a rounding of the
//! double-precision one.
//!
//! Run with `cargo test -p symtn --features cuda,cpu-faer --test \
//! typed_cuda_single_precision -- --ignored --test-threads=1` on a CUDA host.
//! The counter gate reads the process-wide transfer counters, hence
//! `--test-threads=1`.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use num_complex::{Complex32, Complex64};

use symtn::expert::cuda_transfer_stats;
use symtn::sector::{
    product_sector, FermionParityFusionRule, ProductFusionRule, ProductFusionRuleExt,
    SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep, Z2Irrep,
};
use symtn::typed::{BlockFusionTrees, ContractSpec, GradedSpace, Runtime, TensorMap};

mod common;

#[path = "../../tests/support/numerics.rs"]
mod numerics;

use common::{DevicePayload, DeviceRule};

// ---------------------------------------------------------------------------
// Tolerances
// ---------------------------------------------------------------------------

/// A result that combined `terms` values of magnitude at most `scale`.
#[derive(Clone, Copy)]
struct Combined {
    terms: usize,
    scale: f64,
}

fn combined(terms: usize, scale: f64) -> Combined {
    Combined { terms, scale }
}

/// Absolute bound `2 * terms * eps(real(D)) * scale` for the `norm` check,
/// `scale` an upper bound on `sum |conj(a_i) * b_i|`.
/// It is looser than the `f64` accumulation (#1344, #1383) needs; `inner` uses the workspace
/// rule instead.
fn reduction_tolerance<D: DevicePayload>(terms: usize, scale: f64) -> f64 {
    2.0 * (terms as f64) * D::EPS * scale.max(1.0)
}

/// The `numerics` rule over `combined.terms`. Why the extra assertion: this
/// gate's earlier per-entry bound, `8 sqrt(terms) eps(D) scale (1 + |want_i|)`,
/// was relative per entry, while the rule's bound is one per slice; the
/// assertion keeps the rule's bound within the earlier one on the fixture
/// actually compared, so moving to the shared rule never loosens the gate.
#[track_caller]
fn assert_matches_host<D: DevicePayload>(got: &[D], want: &[D], combined: Combined, what: &str) {
    let what = format!("{what} [{}]", D::NAME);
    let largest = want.iter().map(|v| v.magnitude()).fold(0.0_f64, f64::max);
    let smallest = want
        .iter()
        .map(|v| v.magnitude())
        .fold(f64::INFINITY, f64::min);
    let former = 8.0
        * (combined.terms as f64).sqrt()
        * <D as numerics::Numeric>::EPS
        * combined.scale.max(1.0)
        * (1.0 + smallest.min(largest));
    assert!(
        numerics::tolerance::<D>(combined.terms, largest) <= former,
        "{what}: the rule's bound would exceed the gate's former bound {former:e}"
    );
    numerics::assert_slices_close(&what, got, want, combined.terms);
}

fn assert_bit_exact<D: DevicePayload>(actual: &[D], expected: &[D], what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what} [{}]: length", D::NAME);
    for (index, (left, right)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(left, right, "{what} [{}]: element {index}", D::NAME);
    }
}

/// Non-vacuity: the payload actually moved, so a device replay that wrote
/// nothing could not pass.
fn assert_moved<D: DevicePayload>(source: &[D], transformed: &[D], what: &str) {
    assert!(
        source.iter().zip(transformed).any(|(a, b)| a != b),
        "{what} [{}]: the operation left the payload identical",
        D::NAME
    );
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

#[path = "../../tests/support/fixtures.rs"]
mod fixtures;

use fixtures::cuda_runtime;

/// Dyadic, index-dependent, and complex-nontrivial where the dtype allows it.
fn fill<D: DevicePayload, S>(seed: f64) -> impl FnMut(&BlockFusionTrees<S>, &[usize]) -> D {
    let mut ordinal = 0.0_f64;
    move |_, indices| {
        ordinal += 1.0;
        let position: f64 = indices.iter().map(|&i| i as f64).sum();
        D::entry(
            seed + 0.25 * position + 0.125 * (ordinal % 7.0),
            0.5 - 0.25 * (ordinal % 5.0) + 0.125 * position,
        )
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

fn su2_leg() -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(2), 1),
        ],
    )
    .unwrap()
}

type FermionSu2 = ProductFusionRule<FermionParityFusionRule, SU2FusionRule>;

/// `fZ2 ⊠ SU(2)`: fermionic signs *and* non-Abelian recoupling in one provider.
fn fz2_su2_leg() -> GradedSpace<FermionSu2> {
    let rule = Arc::new(FermionParityFusionRule.product(SU2FusionRule));
    GradedSpace::try_new(
        rule,
        [
            (
                product_sector(Z2Irrep::EVEN, SU2Irrep::from_twice_spin(0)),
                2,
            ),
            (
                product_sector(Z2Irrep::ODD, SU2Irrep::from_twice_spin(2)),
                1,
            ),
        ],
    )
    .unwrap()
}

// ---------------------------------------------------------------------------
// Transfer
// ---------------------------------------------------------------------------

fn assert_round_trip_is_bit_exact<R, D>(host: &TensorMap<R, D>)
where
    R: DeviceRule,
    D: DevicePayload,
{
    let device = host.to_cuda().unwrap();
    let back = device.to_host().unwrap();
    assert_bit_exact(
        back.materialize().unwrap().dense_data().unwrap(),
        host.materialize().unwrap().dense_data().unwrap(),
        "round trip",
    );
    assert_eq!(back.rank(), host.rank());
    assert_eq!(back.subblock_count(), host.subblock_count());
    assert_eq!(back.leg_dims().unwrap(), host.leg_dims().unwrap());

    // A lazy adjoint transfers only its parent and rebuilds a lazy view; the
    // logical payload is read by materializing both sides the same way.
    let one = D::entry(1.0, 0.0);
    let lazy = host.adjoint().unwrap();
    let lazy_back = lazy.to_cuda().unwrap().to_host().unwrap();
    assert_bit_exact(
        lazy_back
            .scale(one)
            .unwrap()
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap(),
        lazy.scale(one)
            .unwrap()
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap(),
        "lazy adjoint round trip",
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_transfer_round_trips_bit_exactly_at_every_payload() {
    let runtime = cuda_runtime();
    let u1 = u1_leg();
    let su2 = su2_leg();
    let fz2su2 = fz2_su2_leg();

    fn run<R: DeviceRule, D: DevicePayload>(runtime: &Runtime, leg: &GradedSpace<R>) {
        let host =
            TensorMap::<R, D>::from_subblock_fn(runtime, [leg, leg], [leg], fill::<D, _>(1.5))
                .unwrap();
        assert_round_trip_is_bit_exact(&host);
    }

    run::<_, f64>(&runtime, &u1);
    run::<_, Complex64>(&runtime, &u1);
    run::<_, f32>(&runtime, &u1);
    run::<_, Complex32>(&runtime, &u1);

    run::<_, f64>(&runtime, &su2);
    run::<_, f32>(&runtime, &su2);
    run::<_, Complex32>(&runtime, &su2);

    run::<_, f32>(&runtime, &fz2su2);
    run::<_, Complex32>(&runtime, &fz2su2);
}

// ---------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------

fn assert_arithmetic_matches_host<R, D>(runtime: &Runtime, leg: &GradedSpace<R>)
where
    R: DeviceRule,
    D: DevicePayload,
{
    let a = TensorMap::<R, D>::from_subblock_fn(runtime, [leg], [leg], fill::<D, _>(1.5)).unwrap();
    let b =
        TensorMap::<R, D>::from_subblock_fn(runtime, [leg], [leg], fill::<D, _>(-2.25)).unwrap();
    let alpha = D::entry(2.0, -3.0);
    let beta = D::entry(-0.5, 1.25);
    let terms = a.materialize().unwrap().dense_data().unwrap().len();
    let tolerance = combined(2, 8.0);

    let device_a = a.to_cuda().unwrap();
    let device_b = b.to_cuda().unwrap();

    let scaled = device_a.scale(alpha).unwrap().to_host().unwrap();
    assert_matches_host(
        scaled.materialize().unwrap().dense_data().unwrap(),
        a.scale(alpha)
            .unwrap()
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap(),
        tolerance,
        "scale",
    );
    assert_moved(
        a.materialize().unwrap().dense_data().unwrap(),
        scaled.materialize().unwrap().dense_data().unwrap(),
        "scale",
    );

    let summed = device_a
        .axpby(alpha, &device_b, beta)
        .unwrap()
        .to_host()
        .unwrap();
    assert_matches_host(
        summed.materialize().unwrap().dense_data().unwrap(),
        a.axpby(alpha, &b, beta).unwrap().dense_data().unwrap(),
        tolerance,
        "add",
    );

    let zeros = device_a.zeros_like().unwrap().to_host().unwrap();
    assert_bit_exact(
        zeros.materialize().unwrap().dense_data().unwrap(),
        &vec![D::entry(0.0, 0.0); terms],
        "zeros_like",
    );

    // The lazy fold: `alpha A^H + beta B^H` over the device parents.
    let lazy_a = device_a.adjoint().unwrap();
    let lazy_b = device_b.adjoint().unwrap();
    let host_fold = a
        .adjoint()
        .unwrap()
        .axpby(alpha, &b.adjoint().unwrap(), beta)
        .unwrap();
    assert_matches_host(
        lazy_a
            .axpby(alpha, &lazy_b, beta)
            .unwrap()
            .to_host()
            .unwrap()
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap(),
        host_fold.materialize().unwrap().dense_data().unwrap(),
        tolerance,
        "lazy adjoint fold",
    );
    assert_matches_host(
        lazy_a
            .scale(alpha)
            .unwrap()
            .to_host()
            .unwrap()
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap(),
        a.adjoint()
            .unwrap()
            .scale(alpha)
            .unwrap()
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap(),
        tolerance,
        "lazy adjoint scale",
    );

    // One lazy and one owned operand at every dtype, both orders (#1874).
    assert_matches_host(
        lazy_a
            .axpby(alpha, &device_b, beta)
            .unwrap()
            .to_host()
            .unwrap()
            .dense_data()
            .unwrap(),
        a.adjoint()
            .unwrap()
            .axpby(alpha, &b, beta)
            .unwrap()
            .dense_data()
            .unwrap(),
        tolerance,
        "lazy + owned",
    );
    assert_matches_host(
        device_b
            .axpby(alpha, &lazy_a, beta)
            .unwrap()
            .to_host()
            .unwrap()
            .dense_data()
            .unwrap(),
        b.axpby(alpha, &a.adjoint().unwrap(), beta)
            .unwrap()
            .dense_data()
            .unwrap(),
        tolerance,
        "owned + lazy",
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_arithmetic_matches_the_host_at_every_payload() {
    let runtime = cuda_runtime();
    let u1 = u1_leg();
    let su2 = su2_leg();

    assert_arithmetic_matches_host::<_, f64>(&runtime, &u1);
    assert_arithmetic_matches_host::<_, Complex64>(&runtime, &u1);
    assert_arithmetic_matches_host::<_, f32>(&runtime, &u1);
    assert_arithmetic_matches_host::<_, Complex32>(&runtime, &u1);
    assert_arithmetic_matches_host::<_, f32>(&runtime, &su2);
    assert_arithmetic_matches_host::<_, Complex32>(&runtime, &su2);
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

fn assert_reductions_match_host<R, D>(runtime: &Runtime, leg: &GradedSpace<R>)
where
    R: DeviceRule,
    D: DevicePayload + numerics::Numeric,
{
    let a = TensorMap::<R, D>::from_subblock_fn(runtime, [leg], [leg], fill::<D, _>(1.5)).unwrap();
    let b =
        TensorMap::<R, D>::from_subblock_fn(runtime, [leg], [leg], fill::<D, _>(-2.25)).unwrap();
    let terms = a.dense_data().unwrap().len();
    // An upper bound on `sum |conj(a_i) * b_i|`, which is what the device
    // reduction's error bound is relative to.
    let peak = |tensor: &TensorMap<R, D>| {
        tensor
            .dense_data()
            .unwrap()
            .iter()
            .map(|value| value.magnitude())
            .fold(0.0_f64, f64::max)
    };
    let scale = peak(&a) * peak(&b) * terms as f64;
    let tolerance = reduction_tolerance::<D>(terms, scale);

    let device_a = a.to_cuda().unwrap();
    let device_b = b.to_cuda().unwrap();

    let host_norm = a.norm(2.0).unwrap();
    let device_norm = device_a.norm(2.0).unwrap();
    // `norm` takes a square root of the reduction, so its error is the
    // reduction's own bound or the rounding of a result of that magnitude,
    // whichever is larger. Both halves are absolute and documented.
    assert!(
        (device_norm - host_norm).abs()
            <= tolerance.max(host_norm * <D as numerics::Numeric>::EPS * terms as f64),
        "norm [{}]: device {device_norm}, host {host_norm}",
        D::NAME
    );
    assert!(host_norm > 0.0, "norm [{}] fixture is vacuous", D::NAME);

    // `terms`: an inner product reaches its scalar through the `terms`
    // products `conj(a_i) * b_i`, each weighted once.
    numerics::assert_close(
        &format!("inner [{}]", D::NAME),
        device_a.inner(&device_b).unwrap(),
        a.inner(&b).unwrap(),
        terms,
    );

    // Conjugate linearity in the first argument, which is what distinguishes
    // `inner` from an unconjugated bilinear form for the complex payloads.
    let phase = D::entry(0.0, 1.0);
    if phase != D::entry(0.0, 0.0) {
        let scaled_lhs = device_a.scale(phase).unwrap();
        numerics::assert_close(
            &format!("inner conjugate linearity [{}]", D::NAME),
            scaled_lhs.inner(&device_b).unwrap(),
            a.scale(phase).unwrap().inner(&b).unwrap(),
            terms,
        );
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_reductions_match_the_host() {
    let runtime = cuda_runtime();
    let u1 = u1_leg();
    let su2 = su2_leg();

    assert_reductions_match_host::<_, f64>(&runtime, &u1);
    assert_reductions_match_host::<_, Complex64>(&runtime, &u1);
    assert_reductions_match_host::<_, f32>(&runtime, &u1);
    assert_reductions_match_host::<_, Complex32>(&runtime, &u1);
    // SU(2): several coupled sectors carrying different quantum dimensions, so
    // the wide *cross-sector* half of the contract is exercised too.
    assert_reductions_match_host::<_, f32>(&runtime, &su2);
    assert_reductions_match_host::<_, Complex32>(&runtime, &su2);
}

/// `[leg] <- [leg]` with every entry of one magnitude, `|x| = magnitude(n)`
/// for `n` stored entries, and that magnitude as actually stored (after the
/// payload conversion).
fn constant_magnitude<R, D>(
    runtime: &Runtime,
    leg: &GradedSpace<R>,
    magnitude: fn(usize) -> f64,
) -> (TensorMap<R, D>, f64)
where
    R: DeviceRule,
    D: DevicePayload,
{
    let unit = D::entry(1.0, 1.0).magnitude();
    let n = TensorMap::<R, D>::from_subblock_fn(runtime, [leg], [leg], |_, _| D::entry(1.0, 1.0))
        .unwrap()
        .dense_data()
        .unwrap()
        .len();
    let part = magnitude(n) / unit;
    let tensor =
        TensorMap::<R, D>::from_subblock_fn(runtime, [leg], [leg], |_, _| D::entry(part, part))
            .unwrap();
    let stored = tensor.dense_data().unwrap()[0].magnitude();
    (tensor, stored)
}

/// #1344: the device `norm` accumulates in `f64` like the Host, so it neither
/// saturates where every entry is finite nor underflows where every square is
/// below the payload's subnormal range.
///
/// Two fixtures per provider, both derived from the `f32` lane's own limits:
///
/// * **near `f32::MAX / sqrt(n)`** — `|x| = f32::MAX / (2 sqrt(n))`, so every
///   single square already exceeds `f32::MAX` while the norm (at most
///   `f32::MAX * sqrt(max dim) / 2`) is representable;
/// * **tiny** — `|x| = sqrt(f32::MIN_POSITIVE) * f32::EPSILON`, whose square
///   is below the smallest `f32` subnormal and rounds to zero.
///
/// Oracles: the Host `norm` (wide accumulation, independent code path) and a
/// hand value `|x| * sqrt(W)`, `W = sum_c dim(c) * len_c`, which the caller
/// computes from the leg's sectors and degeneracies. For `[leg] <- [leg]`
/// every coupled sector `c` is one leg sector of degeneracy `d_c`, so its
/// block is `d_c x d_c` and `W = sum_c dim(c) * d_c^2`.
/// `inner(t, t)` narrows its `f64` total to the payload like the Host, so on
/// these fixtures it saturates / underflows exactly where the Host's does.
/// `f64`/`Complex64` run the same fixtures as the control.
fn assert_norm_is_overflow_and_underflow_safe<R, D>(
    runtime: &Runtime,
    leg: &GradedSpace<R>,
    hand_weight: f64,
) where
    R: DeviceRule,
    D: DevicePayload,
{
    let single = D::EPS > f64::EPSILON;
    let big: fn(usize) -> f64 = |n| f64::from(f32::MAX) / (2.0 * (n as f64).sqrt());
    let tiny: fn(usize) -> f64 = |_| f64::from(f32::MIN_POSITIVE).sqrt() * f64::from(f32::EPSILON);
    let fixtures = [(big, "big", true), (tiny, "tiny", false)];
    for (fixture, what, overflows) in fixtures {
        let (host, magnitude) = constant_magnitude::<R, D>(runtime, leg, fixture);
        let n = host.dense_data().unwrap().len();
        let device = host.to_cuda().unwrap();
        // Both norms are f64 sums of exactly widened squares.
        let wide_tolerance = 4.0 * n as f64 * f64::EPSILON;

        let host_norm = host.norm(2.0).unwrap();
        let hand = magnitude * hand_weight.sqrt();
        let device_norm = device.norm(2.0).unwrap();
        assert!(
            device_norm.is_finite() && device_norm > 0.0,
            "{what} norm [{}] must be finite and nonzero, got {device_norm}",
            D::NAME
        );
        assert!(
            (host_norm - hand).abs() <= wide_tolerance * hand,
            "{what} norm [{}]: host {host_norm}, hand {hand}",
            D::NAME
        );
        for (oracle, name) in [(host_norm, "host"), (hand, "hand")] {
            assert!(
                (device_norm - oracle).abs() <= wide_tolerance * oracle,
                "{what} norm [{}]: device {device_norm}, {name} {oracle}",
                D::NAME
            );
        }

        if single {
            let square = magnitude as f32 * magnitude as f32;
            let expected = if overflows { f32::INFINITY } else { 0.0 };
            assert_eq!(square, expected, "{what} [{}]: squares leave f32", D::NAME);
            // The wide total `|x|^2 * W` itself leaves the `f32` range, so the
            // one narrowing saturates / flushes both sides identically.
            assert_eq!(
                device.inner(&device).unwrap(),
                host.inner(&host).unwrap(),
                "{what} inner [{}] must narrow like the host",
                D::NAME
            );
        }
    }
}

type FermionU1 = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;

/// `fZ2 ⊠ U(1)`: fermionic grading on top of an abelian charge.
fn fz2_u1_leg() -> GradedSpace<FermionU1> {
    GradedSpace::try_new(
        Arc::new(FermionParityFusionRule.product(U1FusionRule)),
        [
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
            (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 1),
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(2)), 2),
        ],
    )
    .unwrap()
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_norm_matches_the_host_where_a_payload_sum_would_overflow_or_underflow() {
    let runtime = cuda_runtime();
    let u1 = u1_leg();
    let su2 = su2_leg();
    let fz2_u1 = fz2_u1_leg();

    // Hand weights `W = sum_c dim(c) * d_c^2` from the leg definitions:
    // u1_leg: charges -1, 0, 1 of degeneracy 2, 1, 2 -> 4 + 1 + 4;
    // su2_leg: spin 0 (dim 1) x2 and spin 1 (dim 3) x1 -> 1 * 4 + 3 * 1;
    // fz2_u1_leg: degeneracies 2, 1, 2, all dim 1 -> 4 + 1 + 4.
    let (u1_weight, su2_weight, fz2_u1_weight) = (9.0, 7.0, 9.0);
    assert_norm_is_overflow_and_underflow_safe::<_, f64>(&runtime, &u1, u1_weight);
    assert_norm_is_overflow_and_underflow_safe::<_, Complex64>(&runtime, &u1, u1_weight);
    assert_norm_is_overflow_and_underflow_safe::<_, f32>(&runtime, &u1, u1_weight);
    assert_norm_is_overflow_and_underflow_safe::<_, Complex32>(&runtime, &u1, u1_weight);
    // SU(2): quantum-dimension weights in the cross-sector combine.
    assert_norm_is_overflow_and_underflow_safe::<_, f64>(&runtime, &su2, su2_weight);
    assert_norm_is_overflow_and_underflow_safe::<_, f32>(&runtime, &su2, su2_weight);
    assert_norm_is_overflow_and_underflow_safe::<_, Complex32>(&runtime, &su2, su2_weight);
    assert_norm_is_overflow_and_underflow_safe::<_, f32>(&runtime, &fz2_u1, fz2_u1_weight);
    assert_norm_is_overflow_and_underflow_safe::<_, Complex32>(&runtime, &fz2_u1, fz2_u1_weight);
}

/// #1383: the device `inner` accumulates in `f64` like the Host, so a
/// cancelling sum whose every product exceeds `f32::MAX` stays finite.
///
/// Every coupled block of `[leg] <- [leg]` is `2 x 2` (all degeneracies two).
/// `a` holds `x (1 + i)` everywhere, `x = 2^64`, so every `|a_i|^2` and every
/// product `conj(a_i) b_i` overflows `f32`; `b` holds
/// `x (1 + i) (-1)^(i+j) (1 + 2^-10 [i = j = 0])`. Per block the four products
/// cancel to `u x^2 2^-10 = u 2^118`, `u = |1 + i|^2` in the payload (1 for a
/// real payload, whose entries drop the imaginary part, 2 for a complex one),
/// so the hand oracle is `u 2^118 sum_c dim(c)`. Every partial sum is a small
/// multiple of `2^118`, exact in `f64` in any order, and the result is exact
/// in `f32`. A payload-dtype accumulation instead meets `inf - inf` and
/// returns NaN.
fn assert_inner_survives_overflowing_products<R, D>(
    runtime: &Runtime,
    leg: &GradedSpace<R>,
    dim_sum: f64,
) where
    R: DeviceRule,
    D: DevicePayload + numerics::Numeric,
{
    let x = 2.0_f64.powi(64);
    let a =
        TensorMap::<R, D>::from_subblock_fn(runtime, [leg], [leg], |_, _| D::entry(x, x)).unwrap();
    let b = TensorMap::<R, D>::from_subblock_fn(runtime, [leg], [leg], |_, indices| {
        let sign = if (indices[0] + indices[1]) % 2 == 0 {
            1.0
        } else {
            -1.0
        };
        let bump = if indices == [0, 0] {
            1.0 + 2.0_f64.powi(-10)
        } else {
            1.0
        };
        D::entry(x * sign * bump, x * sign * bump)
    })
    .unwrap();
    // `re^2 + im^2`, exact, where `magnitude().powi(2)` rounds through a root.
    let (re, im) = D::entry(1.0, 1.0).parts();
    let unit = re * re + im * im;
    let hand = D::entry(unit * 2.0_f64.powi(118) * dim_sum, 0.0);
    assert!(
        (x as f32) * (x as f32) == f32::INFINITY,
        "the fixture products must overflow f32"
    );

    let host = a.inner(&b).unwrap();
    let device = a.to_cuda().unwrap().inner(&b.to_cuda().unwrap()).unwrap();
    assert_eq!(
        host,
        hand,
        "host inner [{}] against the hand value",
        D::NAME
    );
    assert!(
        device.magnitude().is_finite(),
        "device inner [{}] must stay finite, got {device:?}",
        D::NAME
    );
    // `terms`: the `a.data().len()` products reaching the scalar.
    numerics::assert_close(
        &format!("device inner [{}]", D::NAME),
        device,
        hand,
        a.dense_data().unwrap().len(),
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_inner_matches_the_host_where_single_precision_products_overflow() {
    let runtime = cuda_runtime();
    let u1 = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 2),
            (U1Irrep::new(1), 2),
        ],
    )
    .unwrap();
    let su2 = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(2), 2),
        ],
    )
    .unwrap();
    // `sum_c dim(c)`: three U(1) charges; SU(2) spins 0 and 1, 1 + 3.
    assert_inner_survives_overflowing_products::<_, f32>(&runtime, &u1, 3.0);
    assert_inner_survives_overflowing_products::<_, Complex32>(&runtime, &u1, 3.0);
    assert_inner_survives_overflowing_products::<_, f32>(&runtime, &su2, 4.0);
    assert_inner_survives_overflowing_products::<_, Complex32>(&runtime, &su2, 4.0);
    // Control: the double-precision payloads never overflowed.
    assert_inner_survives_overflowing_products::<_, f64>(&runtime, &su2, 4.0);
    assert_inner_survives_overflowing_products::<_, Complex64>(&runtime, &su2, 4.0);
}

/// What the widening costs, warm: a single-precision reduction runs the same
/// device schedule as its double-precision twin plus exactly one device
/// allocation per distinct widened operand (#1344, #1383): one for `norm` and
/// `inner(a, a)`, two for `inner(a, b)`. Transferred bytes are *equal*, not
/// halved, because the per-sector partials are downloaded in the wide lane.
/// The double-precision schedule itself is pinned by hand: one partials
/// upload, one download, one GEMM per nonempty coupled sector (three for
/// `u1_leg`).
#[test]
#[ignore = "requires a real CUDA device"]
fn a_warm_single_precision_reduction_costs_one_extra_device_allocation_per_operand() {
    type Stats = (u64, u64, u64, u64, u64, u64);
    fn measure<D: DevicePayload>(runtime: &Runtime, leg: &GradedSpace<U1FusionRule>) -> [Stats; 3] {
        let tensor = |seed| {
            TensorMap::<U1FusionRule, D>::from_subblock_fn(
                runtime,
                [leg],
                [leg],
                fill::<D, _>(seed),
            )
            .unwrap()
            .to_cuda()
            .unwrap()
        };
        let (a, b) = (tensor(1.5), tensor(-2.25));
        let warm = |op: &dyn Fn()| {
            op();
            let before = cuda_transfer_stats();
            op();
            let after = cuda_transfer_stats();
            (
                after.h2d_calls - before.h2d_calls,
                after.h2d_bytes - before.h2d_bytes,
                after.d2h_calls - before.d2h_calls,
                after.d2h_bytes - before.d2h_bytes,
                after.device_allocs - before.device_allocs,
                after.gemm_calls - before.gemm_calls,
            )
        };
        [
            warm(&|| {
                a.norm(2.0).unwrap();
            }),
            warm(&|| {
                a.inner(&a).unwrap();
            }),
            warm(&|| {
                a.inner(&b).unwrap();
            }),
        ]
    }

    let runtime = cuda_runtime();
    let leg = u1_leg();
    for (single, double, what) in [
        (
            measure::<f32>(&runtime, &leg),
            measure::<f64>(&runtime, &leg),
            "f32 against f64",
        ),
        (
            measure::<Complex32>(&runtime, &leg),
            measure::<Complex64>(&runtime, &leg),
            "c32 against c64",
        ),
    ] {
        for ((single, double), (op, widened)) in single.into_iter().zip(double).zip([
            ("norm", 1),
            ("inner(a, a)", 1),
            ("inner(a, b)", 2),
        ]) {
            eprintln!("{what} {op}: single {single:?} double {double:?}");
            assert_eq!(
                (double.0, double.2, double.4, double.5),
                (1, 1, 1, 3),
                "{what} {op}: double-precision (h2d_calls, d2h_calls, device_allocs, gemm_calls)"
            );
            assert_eq!(
                single,
                (
                    double.0,
                    double.1,
                    double.2,
                    double.3,
                    double.4 + widened,
                    double.5
                ),
                "{what} {op}: single precision adds exactly the widened allocations"
            );
        }
    }
}

/// Carried from the #1336 review: `alpha = 0` (and `-0`) on an
/// `*_into` transform must **overwrite**, not scale-and-accumulate.
///
/// The finite-poison loop in `assert_transforms_match_host` cannot tell the
/// two apart — `0 * 7 == 0` either way. A `NaN`-poisoned destination can:
/// `0 * NaN` is `NaN`, so any route that reads the destination before writing
/// leaves `NaN` behind, while a true overwrite leaves an exact `+0`.
#[test]
#[ignore = "requires a real CUDA device"]
fn a_zero_scale_into_clears_a_nan_poisoned_destination() {
    fn assert_cleared<R, D>(runtime: &Runtime, leg: &GradedSpace<R>)
    where
        R: DeviceRule,
        D: DevicePayload,
    {
        let host =
            TensorMap::<R, D>::from_subblock_fn(runtime, [leg, leg], [leg], fill::<D, _>(1.5))
                .unwrap();
        let device = host.to_cuda().unwrap();
        let host_permuted = host.permute(&[1, 2], &[0]).unwrap();
        let poisoned = host_permuted.scale(D::entry(f64::NAN, 0.0)).unwrap();
        assert!(
            !poisoned.dense_data().unwrap().is_empty()
                && poisoned
                    .dense_data()
                    .unwrap()
                    .iter()
                    .all(|value| value.magnitude().is_nan()),
            "the poison fixture [{}] must be entirely NaN",
            D::NAME
        );

        for alpha in [D::entry(0.0, 0.0), D::entry(-0.0, -0.0)] {
            let mut destination = poisoned.to_cuda().unwrap();
            device
                .permute_into(&[1, 2], &[0], &mut destination, alpha, D::from_real(0.0))
                .unwrap();
            for (index, &value) in destination
                .to_host()
                .unwrap()
                .dense_data()
                .unwrap()
                .iter()
                .enumerate()
            {
                assert_eq!(
                    value,
                    D::entry(0.0, 0.0),
                    "permute_into [{}] at alpha {alpha:?} left entry {index} as \
                     {value:?}",
                    D::NAME
                );
            }
        }
    }

    let runtime = cuda_runtime();
    let u1 = u1_leg();
    assert_cleared::<_, f64>(&runtime, &u1);
    assert_cleared::<_, Complex64>(&runtime, &u1);
    assert_cleared::<_, f32>(&runtime, &u1);
    assert_cleared::<_, Complex32>(&runtime, &u1);
}

// ---------------------------------------------------------------------------
// Contraction and compose
// ---------------------------------------------------------------------------

fn assert_contract_and_compose_match_host<R, D>(runtime: &Runtime, leg: &GradedSpace<R>)
where
    R: DeviceRule,
    D: DevicePayload,
{
    let a = TensorMap::<R, D>::from_subblock_fn(runtime, [leg], [leg], fill::<D, _>(1.5)).unwrap();
    let b =
        TensorMap::<R, D>::from_subblock_fn(runtime, [leg], [leg], fill::<D, _>(-2.25)).unwrap();
    let tolerance = combined(a.dense_data().unwrap().len(), 64.0);

    let device_a = a.to_cuda().unwrap();
    let device_b = b.to_cuda().unwrap();

    let host_contract = a
        .contract(
            &b,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    assert_matches_host(
        device_a
            .contract(
                &device_b,
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &[0],
                    domain: &[1],
                },
            )
            .unwrap()
            .to_host()
            .unwrap()
            .dense_data()
            .unwrap(),
        host_contract.dense_data().unwrap(),
        tolerance,
        "contract",
    );

    let host_compose = a.compose(&b).unwrap();
    assert_matches_host(
        device_a
            .compose(&device_b)
            .unwrap()
            .to_host()
            .unwrap()
            .dense_data()
            .unwrap(),
        host_compose.dense_data().unwrap(),
        tolerance,
        "compose",
    );

    // A lazy adjoint operand: `A^H . B` on both sides.
    let host_lazy = a.adjoint().unwrap().compose(&b).unwrap();
    assert_matches_host(
        device_a
            .adjoint()
            .unwrap()
            .compose(&device_b)
            .unwrap()
            .to_host()
            .unwrap()
            .dense_data()
            .unwrap(),
        host_lazy.dense_data().unwrap(),
        tolerance,
        "lazy adjoint compose",
    );

    // `contract_into` writes the same values the returning form does.
    let mut destination = device_a
        .contract(
            &device_b,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap()
        .zeros_like()
        .unwrap();
    device_a
        .contract_into(
            &device_b,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
            &mut destination,
            D::entry(1.0, 0.0),
            D::entry(0.0, 0.0),
        )
        .unwrap();
    assert_matches_host(
        destination.to_host().unwrap().dense_data().unwrap(),
        host_contract.dense_data().unwrap(),
        tolerance,
        "contract_into",
    );

    // A second write over the same destination overwrites rather than
    // accumulates, at every dtype.
    device_a
        .contract_into(
            &device_b,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
            &mut destination,
            D::entry(1.0, 0.0),
            D::entry(0.0, 0.0),
        )
        .unwrap();
    assert_matches_host(
        destination.to_host().unwrap().dense_data().unwrap(),
        host_contract.dense_data().unwrap(),
        tolerance,
        "contract_into is idempotent",
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_contract_and_compose_match_the_host_at_every_payload() {
    let runtime = cuda_runtime();
    let u1 = u1_leg();
    let su2 = su2_leg();
    let fz2su2 = fz2_su2_leg();

    assert_contract_and_compose_match_host::<_, f64>(&runtime, &u1);
    assert_contract_and_compose_match_host::<_, Complex64>(&runtime, &u1);
    assert_contract_and_compose_match_host::<_, f32>(&runtime, &u1);
    assert_contract_and_compose_match_host::<_, Complex32>(&runtime, &u1);
    assert_contract_and_compose_match_host::<_, f32>(&runtime, &su2);
    assert_contract_and_compose_match_host::<_, Complex32>(&runtime, &su2);
    assert_contract_and_compose_match_host::<_, f32>(&runtime, &fz2su2);
    assert_contract_and_compose_match_host::<_, Complex32>(&runtime, &fz2su2);
}

/// The hand-computed fermionic sign fixture of the `f64` suite, at every
/// payload: `contract` carries the `-1` the leg crossing produces and
/// `compose` does not. The values are small integers, so this is an exact
/// comparison at every dtype — a sign that the single-precision lowering lost
/// could not hide inside a tolerance.
#[test]
#[ignore = "requires a real CUDA device"]
fn device_fermionic_signs_are_exact_at_every_payload() {
    let runtime = cuda_runtime();
    let provider = Arc::new(FermionParityFusionRule);
    let odd = || GradedSpace::try_new(Arc::clone(&provider), [(Z2Irrep::ODD, 1)]).unwrap();
    let odd_dual = || odd().try_dual().unwrap();

    fn run<D: DevicePayload>(
        runtime: &Runtime,
        lhs_codomain: &GradedSpace<FermionParityFusionRule>,
        lhs_domain: &GradedSpace<FermionParityFusionRule>,
        rhs_codomain: &GradedSpace<FermionParityFusionRule>,
        rhs_domain: &GradedSpace<FermionParityFusionRule>,
    ) {
        let lhs =
            TensorMap::<_, D>::from_subblock_fn(runtime, [lhs_codomain], [lhs_domain], |_, _| {
                D::entry(2.0, 0.0)
            })
            .unwrap();
        let rhs =
            TensorMap::<_, D>::from_subblock_fn(runtime, [rhs_codomain], [rhs_domain], |_, _| {
                D::entry(3.0, 0.0)
            })
            .unwrap();
        let device_lhs = lhs.to_cuda().unwrap();
        let device_rhs = rhs.to_cuda().unwrap();

        assert_bit_exact(
            device_lhs
                .contract(
                    &device_rhs,
                    &ContractSpec {
                        lhs: &[1],
                        rhs: &[0],
                        codomain: &[0],
                        domain: &[1],
                    },
                )
                .unwrap()
                .to_host()
                .unwrap()
                .dense_data()
                .unwrap(),
            &[D::entry(-6.0, 0.0)],
            "fermionic contract",
        );
        assert_bit_exact(
            device_lhs
                .compose(&device_rhs)
                .unwrap()
                .to_host()
                .unwrap()
                .dense_data()
                .unwrap(),
            &[D::entry(6.0, 0.0)],
            "fermionic compose",
        );
        // The host agrees, at this dtype, with the same hand value.
        assert_bit_exact(
            lhs.contract(
                &rhs,
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &[0],
                    domain: &[1],
                },
            )
            .unwrap()
            .dense_data()
            .unwrap(),
            &[D::entry(-6.0, 0.0)],
            "fermionic contract, host",
        );
    }

    run::<f64>(&runtime, &odd(), &odd_dual(), &odd_dual(), &odd());
    run::<Complex64>(&runtime, &odd(), &odd_dual(), &odd_dual(), &odd());
    run::<f32>(&runtime, &odd(), &odd_dual(), &odd_dual(), &odd());
    run::<Complex32>(&runtime, &odd(), &odd_dual(), &odd_dual(), &odd());
}

// ---------------------------------------------------------------------------
// Structural transforms
// ---------------------------------------------------------------------------

fn assert_transforms_match_host<R, D>(runtime: &Runtime, leg: &GradedSpace<R>)
where
    R: DeviceRule,
    D: DevicePayload,
{
    let host =
        TensorMap::<R, D>::from_subblock_fn(runtime, [leg, leg], [leg], fill::<D, _>(1.5)).unwrap();
    let device = host.to_cuda().unwrap();
    // A recoupling combines at most the tree cardinality of one block; three
    // legs over these fixtures keeps that small, so `16` is generous.
    let tolerance = combined(16, 16.0);

    let host_permuted = host.permute(&[1, 2], &[0]).unwrap();
    let device_permuted = device.permute(&[1, 2], &[0]).unwrap().to_host().unwrap();
    assert_matches_host(
        device_permuted.dense_data().unwrap(),
        host_permuted.dense_data().unwrap(),
        tolerance,
        "permute",
    );
    assert_moved(
        host.dense_data().unwrap(),
        device_permuted.dense_data().unwrap(),
        "permute",
    );

    let levels: Vec<usize> = (0..host.rank()).collect();
    assert_matches_host(
        device
            .braid(&[1, 2], &[0], &levels)
            .unwrap()
            .to_host()
            .unwrap()
            .dense_data()
            .unwrap(),
        host.braid(&[1, 2], &[0], &levels)
            .unwrap()
            .dense_data()
            .unwrap(),
        tolerance,
        "braid",
    );

    assert_matches_host(
        device
            .repartition(1)
            .unwrap()
            .to_host()
            .unwrap()
            .dense_data()
            .unwrap(),
        host.repartition(1).unwrap().dense_data().unwrap(),
        tolerance,
        "repartition",
    );

    assert_matches_host(
        device
            .transpose(&[2], &[1, 0])
            .unwrap()
            .to_host()
            .unwrap()
            .dense_data()
            .unwrap(),
        host.transpose(&[2], &[1, 0]).unwrap().dense_data().unwrap(),
        tolerance,
        "transpose",
    );

    // The `*_into` forms of the same three planar-or-permute transforms (#1339) write
    // what their returning twins return, at every caller scale.
    // Each destination starts poisoned with a different multiple of the
    // expected payload, so an implementation that wrote nothing at `alpha = 1`
    // could not pass either.
    let poison = D::entry(7.0, -3.0);
    for alpha in [
        D::entry(1.0, 0.0),
        D::entry(-2.5, 0.5),
        D::entry(0.0, 0.0),
        D::entry(-0.0, -0.0),
    ] {
        let mut destination = host_permuted.scale(poison).unwrap().to_cuda().unwrap();
        device
            .permute_into(&[1, 2], &[0], &mut destination, alpha, D::from_real(0.0))
            .unwrap();
        assert_matches_host(
            destination.to_host().unwrap().dense_data().unwrap(),
            host_permuted.scale(alpha).unwrap().dense_data().unwrap(),
            tolerance,
            "permute_into",
        );

        let host_bent = host.repartition(1).unwrap();
        let mut bent = host_bent.scale(poison).unwrap().to_cuda().unwrap();
        device
            .repartition_into(&mut bent, alpha, D::from_real(0.0))
            .unwrap();
        assert_matches_host(
            bent.to_host().unwrap().dense_data().unwrap(),
            host_bent.scale(alpha).unwrap().dense_data().unwrap(),
            tolerance,
            "repartition_into",
        );

        let host_cyclic = host.transpose(&[2], &[1, 0]).unwrap();
        let mut cyclic = host_cyclic.scale(poison).unwrap().to_cuda().unwrap();
        device
            .transpose_into(&[2], &[1, 0], &mut cyclic, alpha, D::from_real(0.0))
            .unwrap();
        assert_matches_host(
            cyclic.to_host().unwrap().dense_data().unwrap(),
            host_cyclic.scale(alpha).unwrap().dense_data().unwrap(),
            tolerance,
            "transpose_into",
        );
    }

    // Round trip: permuting back reproduces the source at the payload's own
    // precision.
    assert_matches_host(
        device
            .permute(&[1, 2], &[0])
            .unwrap()
            .permute(&[2, 0], &[1])
            .unwrap()
            .to_host()
            .unwrap()
            .dense_data()
            .unwrap(),
        host.dense_data().unwrap(),
        tolerance,
        "permute round trip",
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_transforms_match_the_host_at_every_payload() {
    let runtime = cuda_runtime();
    let u1 = u1_leg();
    let su2 = su2_leg();
    let fz2su2 = fz2_su2_leg();

    assert_transforms_match_host::<_, f64>(&runtime, &u1);
    assert_transforms_match_host::<_, f32>(&runtime, &u1);
    assert_transforms_match_host::<_, Complex32>(&runtime, &u1);
    // SU(2): real recoupling coefficients acting on a single-precision payload.
    assert_transforms_match_host::<_, f64>(&runtime, &su2);
    assert_transforms_match_host::<_, f32>(&runtime, &su2);
    assert_transforms_match_host::<_, Complex32>(&runtime, &su2);
    // fZ2 ⊠ SU(2): fermionic signs and recoupling together.
    assert_transforms_match_host::<_, f32>(&runtime, &fz2su2);
    assert_transforms_match_host::<_, Complex32>(&runtime, &fz2su2);
}

/// The warm `*_into` contract of #1339 does not depend on the
/// payload dtype: a warm replay over a caller-owned destination transfers
/// nothing and allocates nothing at single precision too. The zero-scale route
/// uploads at most one *element* of the payload's own zero template, once per
/// context — half the bytes of the double-precision template, never a buffer.
#[test]
#[ignore = "requires a real CUDA device"]
fn a_warm_single_precision_into_transfers_nothing() {
    let runtime = cuda_runtime();
    let leg = u1_leg();

    fn run<D: DevicePayload>(runtime: &Runtime, leg: &GradedSpace<U1FusionRule>) {
        let host = TensorMap::<U1FusionRule, D>::from_subblock_fn(
            runtime,
            [leg, leg],
            [leg],
            fill::<D, _>(1.5),
        )
        .unwrap();
        let device = host.to_cuda().unwrap();
        let mut destination = host.permute(&[1, 2], &[0]).unwrap().to_cuda().unwrap();

        // Cold: prepares the structure for this pair and this dtype.
        device
            .permute_into(
                &[1, 2],
                &[0],
                &mut destination,
                D::entry(1.0, 0.0),
                D::entry(0.0, 0.0),
            )
            .unwrap();
        let cold = runtime.cuda_tree_transform_stats().unwrap();

        let before = cuda_transfer_stats();
        device
            .permute_into(
                &[1, 2],
                &[0],
                &mut destination,
                D::entry(-2.5, 0.5),
                D::entry(0.0, 0.0),
            )
            .unwrap();
        let after = cuda_transfer_stats();
        assert_eq!(
            (
                after.h2d_calls - before.h2d_calls,
                after.h2d_bytes - before.h2d_bytes,
                after.d2h_calls - before.d2h_calls,
                after.device_allocs - before.device_allocs,
            ),
            (0, 0, 0, 0),
            "warm overwrite [{}] must transfer and allocate nothing",
            D::NAME
        );
        assert_eq!(
            runtime.cuda_tree_transform_stats().unwrap(),
            cold,
            "a warm overwrite [{}] grows no device state",
            D::NAME
        );

        // The zero-scale route writes zero fills over the written layouts
        // (#1438), so its first call may size the context zero template to
        // the largest written layout and the `1` operand of *this* payload
        // once: at most two uploads, bounded by the destination.
        let before = cuda_transfer_stats();
        device
            .permute_into(
                &[1, 2],
                &[0],
                &mut destination,
                D::entry(0.0, 0.0),
                D::entry(0.0, 0.0),
            )
            .unwrap();
        let after = cuda_transfer_stats();
        let destination_len = host
            .permute(&[1, 2], &[0])
            .unwrap()
            .dense_data()
            .unwrap()
            .len();
        let bound = ((destination_len + 1) * std::mem::size_of::<D>()) as u64;
        assert!(
            after.h2d_calls - before.h2d_calls <= 2 && after.h2d_bytes - before.h2d_bytes <= bound,
            "the zero template [{}] is bounded by the destination, uploaded once",
            D::NAME
        );

        let before = cuda_transfer_stats();
        device
            .permute_into(
                &[1, 2],
                &[0],
                &mut destination,
                D::entry(-0.0, -0.0),
                D::entry(0.0, 0.0),
            )
            .unwrap();
        let after = cuda_transfer_stats();
        assert_eq!(
            (
                after.h2d_calls - before.h2d_calls,
                after.d2h_calls - before.d2h_calls,
                after.device_allocs - before.device_allocs,
            ),
            (0, 0, 0),
            "a warm zero-scale overwrite [{}] transfers nothing",
            D::NAME
        );
    }

    run::<f64>(&runtime, &leg);
    run::<f32>(&runtime, &leg);
    run::<Complex32>(&runtime, &leg);
}

// ---------------------------------------------------------------------------
// Cost contracts
// ---------------------------------------------------------------------------

/// Same device work, half the bytes.
///
/// The comparison is relative and same-process: the `f32` run of a fixture is
/// compared against the `f64` run of the *same* fixture in the same test
/// binary, so no absolute constant of a particular platform appears. Call
/// counts must be equal (the schedule does not depend on the payload dtype)
/// and transferred bytes must be exactly half (the element size does).
#[test]
#[ignore = "requires a real CUDA device"]
fn single_precision_costs_the_same_device_calls_and_half_the_bytes() {
    let runtime = cuda_runtime();
    let leg = u1_leg();

    fn measure<D: DevicePayload>(
        runtime: &Runtime,
        leg: &GradedSpace<U1FusionRule>,
    ) -> (u64, u64, u64, u64, u64, u64) {
        let a = TensorMap::<U1FusionRule, D>::from_subblock_fn(
            runtime,
            [leg],
            [leg],
            fill::<D, _>(1.5),
        )
        .unwrap();
        let b = TensorMap::<U1FusionRule, D>::from_subblock_fn(
            runtime,
            [leg],
            [leg],
            fill::<D, _>(-2.25),
        )
        .unwrap();
        let before = cuda_transfer_stats();
        let device_a = a.to_cuda().unwrap();
        let device_b = b.to_cuda().unwrap();
        let product = device_a
            .contract(
                &device_b,
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &[0],
                    domain: &[1],
                },
            )
            .unwrap();
        let _ = product.to_host().unwrap();
        let after = cuda_transfer_stats();
        (
            after.h2d_calls - before.h2d_calls,
            after.h2d_bytes - before.h2d_bytes,
            after.d2h_calls - before.d2h_calls,
            after.d2h_bytes - before.d2h_bytes,
            after.device_allocs - before.device_allocs,
            after.gemm_calls - before.gemm_calls,
        )
    }

    // Warm both lanes first: the first call of a dtype builds its execution
    // lane and its device scalar operands, which are one-time costs and not
    // part of the contract under test.
    let _ = measure::<f64>(&runtime, &leg);
    let _ = measure::<f32>(&runtime, &leg);
    let _ = measure::<Complex64>(&runtime, &leg);
    let _ = measure::<Complex32>(&runtime, &leg);

    for (single, double, what) in [
        (
            measure::<f32>(&runtime, &leg),
            measure::<f64>(&runtime, &leg),
            "f32 against f64",
        ),
        (
            measure::<Complex32>(&runtime, &leg),
            measure::<Complex64>(&runtime, &leg),
            "c32 against c64",
        ),
    ] {
        assert_eq!(
            (single.0, single.2, single.4, single.5),
            (double.0, double.2, double.4, double.5),
            "{what}: (h2d_calls, d2h_calls, device_allocs, gemm_calls) must be equal"
        );
        eprintln!("{what}: single {single:?} double {double:?}");
        assert!(double.1 > 0 && double.3 > 0, "{what}: vacuous byte counts");
        assert_eq!(single.1 * 2, double.1, "{what}: h2d bytes must be halved");
        assert_eq!(single.3 * 2, double.3, "{what}: d2h bytes must be halved");
    }
}

// ---------------------------------------------------------------------------
// Device-free gates (no `#[ignore]`: a `cuda` build runs them anywhere)
// ---------------------------------------------------------------------------

/// The admission table itself, as a compile-time fact: the base and
/// factorization device families (device QR included, #1271) are open for all
/// four payloads.
#[test]
fn the_device_admission_markers_hold_exactly_where_the_table_says() {
    fn device_payload<D: symtn::typed::CudaPayload>() {}
    fn device_factorization_payload<D: symtn::typed::CudaFactorizationPayload>() {}

    device_payload::<f64>();
    device_payload::<Complex64>();
    device_payload::<f32>();
    device_payload::<Complex32>();

    device_factorization_payload::<f64>();
    device_factorization_payload::<Complex64>();
    device_factorization_payload::<f32>();
    device_factorization_payload::<Complex32>();
}

/// The rejection order does not depend on the payload dtype: a runtime built
/// without a CUDA device rejects an upload with the same typed error at every
/// payload, before any device work.
#[test]
fn a_device_less_runtime_rejects_every_payload_the_same_way() {
    let runtime = Runtime::builder().build().unwrap();
    let leg = u1_leg();

    fn reject<D: DevicePayload>(runtime: &Runtime, leg: &GradedSpace<U1FusionRule>) -> String {
        let host = TensorMap::<U1FusionRule, D>::from_subblock_fn(
            runtime,
            [leg],
            [leg],
            fill::<D, _>(1.5),
        )
        .unwrap();
        match host.to_cuda() {
            Ok(_) => panic!("a device-less runtime must not upload [{}]", D::NAME),
            Err(error) => format!("{error}"),
        }
    }

    let double = reject::<f64>(&runtime, &leg);
    assert_eq!(reject::<f32>(&runtime, &leg), double);
    assert_eq!(reject::<Complex32>(&runtime, &leg), double);
    assert_eq!(reject::<Complex64>(&runtime, &leg), double);
}
