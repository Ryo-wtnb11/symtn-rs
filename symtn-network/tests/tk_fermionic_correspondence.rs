//! Numeric correspondence against TensorKit for fermionic (FZ2) contractions,
//! including a diagonal `S` from an SVD used in a closed loop with dual legs
//! (which fires the supertrace twist). The reference numbers were checked
//! against TensorKit 0.17.1 on identical degeneracy-1 FZ2 maps; every value here is
//! a physical, basis-independent scalar, so a mismatch is a genuine convention
//! divergence — not a layout/gauge artifact. Guards that symtn's fermionic
//! contract, and the diagonal-`contract` fast path on top of it, agree with
//! TensorKit.

use std::sync::Arc;
use symtn::typed::ContractSpec;
use symtn::typed::Direction;
use symtn::typed::Engine;

use symtn::sector::{FermionParityFusionRule, Z2Irrep};
use symtn::typed::Complex64;
use symtn::typed::{Eig, GradedSpace, Runtime, Svd, TensorMap};

use crate::network_support;
use network_support::{net, op};

/// FZ2 map `V <- V`, degeneracy 1, with `even`/`odd` block values — the exact
/// tensor `build(f)` produces in the Julia reference.
fn fz2_map(
    engine: &mut dyn Engine,
    rt: &Runtime,
    v: &GradedSpace<FermionParityFusionRule>,
    even: f64,
    odd: f64,
) -> TensorMap<FermionParityFusionRule, f64> {
    TensorMap::from_subblock_fn(engine, rt, [v], [v], move |trees, _| {
        if *trees.coupled() == Z2Irrep::EVEN {
            even
        } else {
            odd
        }
    })
    .unwrap()
}

fn scalar(engine: &mut dyn Engine, t: TensorMap<FermionParityFusionRule, f64>) -> f64 {
    t.scalar(engine).unwrap()
}

#[test]
fn fz2_contractions_match_tensorkit() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let rt = Runtime::builder().build().unwrap();
    let v = GradedSpace::try_new(
        Arc::new(FermionParityFusionRule),
        [(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 1)],
    )
    .unwrap();
    let a = fz2_map(engine, &rt, &v, 1.0, 4.0);
    let b = fz2_map(engine, &rt, &v, 2.0, 1.5);
    let c = fz2_map(engine, &rt, &v, 0.5, 2.5);
    let t = fz2_map(engine, &rt, &v, 3.0, 2.0);

    // S from an SVD is a compact diagonal factor; singular values = |T| per sector.
    let Svd { s, .. } = t.svd_compact(engine, &[0], &[1]).unwrap();
    let sv = s.svd_vals(engine, &[0], &[1]).unwrap();
    for entry in &sv {
        let expect = if entry.sector == Z2Irrep::EVEN {
            3.0
        } else {
            2.0
        };
        assert!(
            (entry.values[0] - expect).abs() < 1e-12,
            "singular value for sector {:?}: {} vs {expect}",
            entry.sector,
            entry.values[0]
        );
    }

    // What: TensorKit `tr` is the positive ordinary trace, while an explicit
    // parity twist (or a closed `@tensor` loop) is the fermionic supertrace.
    let d = fz2_map(engine, &rt, &v, 2.0, 3.0);
    let ordinary = d.tr(engine).unwrap();
    let twisted = d
        .twist(engine, &[0], Direction::Forward)
        .unwrap()
        .tr(engine)
        .unwrap();
    assert!((ordinary - 5.0).abs() < 1e-12, "ordinary = {ordinary}");
    assert!((twisted - (-1.0)).abs() < 1e-12, "twisted = {twisted}");

    // The SVD factor uses compact diagonal storage and must keep the same
    // ordinary trace without materializing its dense block matrices.
    let Svd { s: compact_d, .. } = d.svd_compact(engine, &[0], &[1]).unwrap();
    let compact_ordinary = compact_d.tr(engine).unwrap();
    assert!(
        (compact_ordinary - 5.0).abs() < 1e-12,
        "compact ordinary = {compact_ordinary}"
    );
    let Svd { s: compact_c64, .. } = d
        .convert::<Complex64>(engine)
        .unwrap()
        .svd_compact(engine, &[0], &[1])
        .unwrap();
    let compact_c64_ordinary = compact_c64.tr(engine).unwrap();
    assert!(
        (compact_c64_ordinary - Complex64::new(5.0, 0.0)).norm() < 1e-12,
        "compact c64 ordinary = {compact_c64_ordinary}"
    );

    // What: `eig_full().d` uses genuine complex diagonal storage. Its ordinary
    // trace sums both sectors without the odd parity twist.
    let complex_source = TensorMap::from_subblock_fn(engine, &rt, [&v], [&v], |trees, _| {
        if *trees.coupled() == Z2Irrep::EVEN {
            Complex64::new(2.0, 1.0)
        } else {
            Complex64::new(3.0, 4.0)
        }
    })
    .unwrap();
    let Eig { d: complex_d, .. } = complex_source.eig_full(engine, &[0], &[1]).unwrap();
    let complex_trace = complex_d.tr(engine).unwrap();
    let expected_complex_trace = Complex64::new(5.0, 5.0);
    assert!(
        (complex_trace - expected_complex_trace).norm() < 1e-12,
        "complex ordinary = {complex_trace}"
    );

    let ordinary_ab = a.compose(engine, &b).unwrap().tr(engine).unwrap();
    assert!(
        (ordinary_ab - 8.0).abs() < 1e-12,
        "ordinary_ab = {ordinary_ab}"
    );
    let operand = net(&[op(&["i"], &["j"]), op(&["j"], &["i"])], &[], &[])
        .contract(engine, &[&a, &b])
        .unwrap();

    // (1) dense supertrace loop tr(A B) = 2 - 6 = -4.
    let s1 = scalar(engine, operand);
    assert!((s1 - (-4.0)).abs() < 1e-12, "s1 = {s1}");
    let operand = net(
        &[op(&["i"], &["j"]), op(&["j"], &["k"]), op(&["k"], &["i"])],
        &[],
        &[],
    )
    .contract(engine, &[&a, &b, &c])
    .unwrap();

    // (2) dense three-map loop tr(A B C) = 1 - 15 = -14.
    let s2 = scalar(engine, operand);
    assert!((s2 - (-14.0)).abs() < 1e-12, "s2 = {s2}");
    let operand = net(
        &[op(&["i"], &["j"]), op(&["j"], &["k"]), op(&["k"], &["i"])],
        &[],
        &[],
    )
    .contract(engine, &[&a, &s, &b])
    .unwrap();

    // (3) diagonal S in a closed fermionic loop A[a;b] S[b;c] B[c;a] = 6 - 12 = -6.
    let s3 = scalar(engine, operand);
    assert!((s3 - (-6.0)).abs() < 1e-12, "s3 = {s3}");
    let operand = net(
        &[op(&["i"], &["j"]), op(&["j"], &["k"]), op(&["k"], &["i"])],
        &[],
        &[],
    )
    .contract(engine, &[&s, &a, &b])
    .unwrap();

    // (4) S on the leading side: S[a;b] A[b;c] B[c;a] = -6.
    let s4 = scalar(engine, operand);
    assert!((s4 - (-6.0)).abs() < 1e-12, "s4 = {s4}");

    // Force the single-axis diagonal `contract` fast path explicitly (rather than
    // whatever order the planner picks): A[i;j] · S[j;k] goes through scale+permute
    // + the fermionic twist fold, and closing with B must still give TK's -6.
    let as_ = a
        .contract(
            engine,
            &s,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    let operand = net(&[op(&["i"], &["k"]), op(&["k"], &["i"])], &[], &[])
        .contract(engine, &[&as_, &b])
        .unwrap();
    let s3_fast = scalar(engine, operand);
    assert!((s3_fast - (-6.0)).abs() < 1e-12, "s3_fast = {s3_fast}");
    // And the leading (D * A) order on `s`.
    let sa = s
        .contract(
            engine,
            &a,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    let operand = net(&[op(&["i"], &["k"]), op(&["k"], &["i"])], &[], &[])
        .contract(engine, &[&sa, &b])
        .unwrap();
    let s4_fast = scalar(engine, operand);
    assert!((s4_fast - (-6.0)).abs() < 1e-12, "s4_fast = {s4_fast}");
}
