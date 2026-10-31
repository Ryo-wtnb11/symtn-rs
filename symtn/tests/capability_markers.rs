//! Static-assertion side of the payload-dtype capability split (#1308).
//!
//! Each helper below is a *positive* assertion: the family's representative
//! entry points are callable with exactly the marker named in its bound, and
//! with nothing stronger. The matching *negative* assertions — that a
//! `D: TensorScalar` caller cannot reach a factorization or a matrix function,
//! and that a `D: FactorizationScalar` caller cannot reach a matrix function —
//! are `compile_fail` doctests on the marker traits themselves
//! (`symtn::typed::FactorizationScalar`, `symtn::typed::AdvancedLinalgScalar`),
//! because a `compile_fail` alone can pass for an unrelated reason: the pair is
//! what pins the boundary.
//!
//! The helpers are never executed. They are instantiated as function items in
//! the test below, which is what forces their bodies to be type-checked.

use symtn::sector::U1FusionRule;
use symtn::typed::Direction;
use symtn::typed::HermitianTol;
use symtn::typed::{AdvancedLinalgScalar, FactorizationScalar, TensorMap, TensorScalar};

/// Base family: everything admitted by [`TensorScalar`] alone.
fn base_family<D: TensorScalar>(tensor: &TensorMap<U1FusionRule, D>) {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _ = tensor.norm(engine, 2.0);
    let _ = tensor.norm(engine, f64::INFINITY);
    let _ = tensor.inner(engine, tensor);
    let _ = tensor.tr(engine);
    let _ = tensor.adjoint(engine);
    let _ = tensor.permute(engine, &[0], &[1]);
    let _ = tensor.transpose(engine, &[1], &[0]);
    let _ = tensor.repartition(engine, 1);
    let _ = tensor.twist(engine, &[0], Direction::Forward);
    let _ = tensor.compose(engine, tensor);
    let _ = tensor.otimes(engine, tensor);
    let _ = tensor.trace_pairs(engine, &[(0, 1)]);
    let _ = tensor.diagview(engine);
    let _ = tensor.map_diagonal(engine, |value| value);
    let _ = tensor.subblocks();
}

/// Factorization family: needs [`FactorizationScalar`], not more.
fn factorization_family<D: FactorizationScalar>(tensor: &TensorMap<U1FusionRule, D>) {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _ = tensor.qr_compact(engine, &[0], &[1]);
    let _ = tensor.qr_full(engine, &[0], &[1]);
    let _ = tensor.lq_compact(engine, &[0], &[1]);
    let _ = tensor.lq_full(engine, &[0], &[1]);
    let _ = tensor.svd_compact(engine, &[0], &[1]);
    let _ = tensor.svd_full(engine, &[0], &[1]);
    let _ = tensor.svd_vals(engine, &[0], &[1]);
    let _ = tensor.eigh_full(engine, &[0], &[1], HermitianTol::DEFAULT);
    let _ = tensor.eigh_vals(engine, &[0], &[1], HermitianTol::DEFAULT);
    let _ = tensor.left_null(engine, &[0], &[1]);
    let _ = tensor.right_null(engine, &[0], &[1]);
    let _ = tensor.left_polar(engine, &[0], &[1]);
    let _ = tensor.right_polar(engine, &[0], &[1]);
}

/// Advanced linear algebra: needs [`AdvancedLinalgScalar`].
fn advanced_family<D: AdvancedLinalgScalar>(tensor: &TensorMap<U1FusionRule, D>) {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _ = tensor.exp(engine, &[0], &[1]);
    let _ = tensor.inv(engine, &[0], &[1]);
    let _ = tensor.pinv(engine, &[0], &[1], 0.0);
    let _ = tensor.solve(engine, &[0], &[1], tensor, &[0], &[1]);
}

/// The general (non-Hermitian) eigendecomposition also needs its factor dtype
/// `D::Eig` to be a payload, so it gets its own helper rather than widening
/// the marker.
fn general_eig_family<D>(tensor: &TensorMap<U1FusionRule, D>)
where
    D: AdvancedLinalgScalar,
    D::Eig: TensorScalar,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _ = tensor.eig_full(engine, &[0], &[1]);
    let _ = tensor.eig_vals(engine, &[0], &[1]);
}

/// Instantiating the helpers for every admitted payload dtype is the
/// assertion: it type-checks each body under exactly one marker bound.
#[test]
fn each_family_is_callable_under_exactly_its_marker() {
    let _: fn(&TensorMap<U1FusionRule, f64>) = base_family;
    let _: fn(&TensorMap<U1FusionRule, num_complex::Complex64>) = base_family;
    // Single precision reaches every family: base (#1315), factorization
    // (#1324) and advanced (#1459).
    let _: fn(&TensorMap<U1FusionRule, f32>) = base_family;
    let _: fn(&TensorMap<U1FusionRule, num_complex::Complex32>) = base_family;
    let _: fn(&TensorMap<U1FusionRule, f64>) = factorization_family;
    let _: fn(&TensorMap<U1FusionRule, num_complex::Complex64>) = factorization_family;
    let _: fn(&TensorMap<U1FusionRule, f32>) = factorization_family;
    let _: fn(&TensorMap<U1FusionRule, num_complex::Complex32>) = factorization_family;
    let _: fn(&TensorMap<U1FusionRule, f64>) = advanced_family;
    let _: fn(&TensorMap<U1FusionRule, num_complex::Complex64>) = advanced_family;
    let _: fn(&TensorMap<U1FusionRule, f64>) = general_eig_family;
    let _: fn(&TensorMap<U1FusionRule, num_complex::Complex64>) = general_eig_family;
    let _: fn(&TensorMap<U1FusionRule, f32>) = advanced_family;
    let _: fn(&TensorMap<U1FusionRule, num_complex::Complex32>) = advanced_family;
    let _: fn(&TensorMap<U1FusionRule, f32>) = general_eig_family;
    let _: fn(&TensorMap<U1FusionRule, num_complex::Complex32>) = general_eig_family;
}
