//! Warm eager `contract` with a non-default output order or codomain/domain
//! split costs no more than `contract` in the default order and split
//! followed by `permute` (#1461, #1549), and returns the same tensor. TensorKit
//! `blas_contract!` takes the same shape of work when the output layout is not
//! the GEMM's: a temporary plus a permuting `tensoradd!` (its `copyC` path),
//! inside the one call.

use std::hint::black_box;
use symtn::typed::Engine;

use symtn::sector::{
    FermionParityFusionRule, ProductFusionRule, ProductSector, SU2FusionRule, SU2Irrep,
    U1FusionRule, U1Irrep, Z2Irrep,
};
use symtn::typed::{ContractSpec, GradedSpace, Runtime, TensorMap};

use crate::numerics;

use crate::counting_alloc;

/// (calls, bytes, allocations of exactly `payload` bytes) made while `f`
/// runs; the result is dropped afterwards.
fn allocations<T>(payload: usize, f: impl FnOnce() -> T) -> ((u64, u64, u64), T) {
    let (value, allocs) = counting_alloc::measure_matching(payload..=payload, f);
    ((allocs.calls, allocs.bytes, allocs.matched_calls), value)
}

/// Warm cost and value of `contract` onto `codomain ← domain` against the
/// default order and split followed by `permute(codomain, domain)`.
macro_rules! assert_no_costlier_than_permute {
    ($engine:ident, $runtime:expr, $a:expr, $b:expr, $lhs_axes:expr, $rhs_axes:expr,
     $codomain:expr, $domain:expr $(,)?) => {{
        let (lhs_axes, rhs_axes): (&[usize], &[usize]) = ($lhs_axes, $rhs_axes);
        let (codomain, domain): (&[usize], &[usize]) = ($codomain, $domain);
        let identity: Vec<usize> = (0..codomain.len() + domain.len()).collect();
        let (default_codomain, default_domain) = identity.split_at($a.rank() - lhs_axes.len());
        let spec = ContractSpec {
            lhs: lhs_axes,
            rhs: rhs_axes,
            codomain,
            domain,
        };
        let default = ContractSpec {
            codomain: default_codomain,
            domain: default_domain,
            ..spec
        };
        let fused = |$engine: &mut dyn Engine| black_box($a).contract($engine, $b, &spec).unwrap();
        let separate = |$engine: &mut dyn Engine| {
            black_box($a)
                .contract($engine, $b, &default)
                .unwrap()
                .permute($engine, codomain, domain)
                .unwrap()
        };
        let payload = std::mem::size_of_val(fused($engine).dense_data().unwrap());
        separate($engine);
        let misses = completed_transformers().misses();
        let (fused_cost, fused_value) = allocations(payload, || fused($engine));
        let (separate_cost, separate_value) = allocations(payload, || separate($engine));

        // What: the warm call compiles no new transform plan.
        assert_eq!(completed_transformers().misses(), misses);
        // What: the same tensor, bit for bit.
        assert_eq!(
            fused_value.dense_data().unwrap(),
            separate_value.dense_data().unwrap()
        );
        assert_eq!(fused_value.codomain_rank(), codomain.len());
        eprintln!(
            "{lhs_axes:?}/{rhs_axes:?} {codomain:?} <- {domain:?}: contract {fused_cost:?}, \
             contract + permute {separate_cost:?}"
        );
        // What: no more allocation calls or output-sized buffers than the
        // two-step route. Equality is TensorKit's `copyC`: a zero-copy
        // candidate contracts into a per-call temporary that one permute moves
        // into the result. Total bytes are not compared here: the copyC route
        // builds its temporary's space (a few hundred bytes of metadata) where
        // the two-step route builds the permute's.
        assert!(
            fused_cost.0 <= separate_cost.0 && fused_cost.2 <= separate_cost.2,
            "contract({lhs_axes:?}, {rhs_axes:?}; {codomain:?} <- {domain:?}) {fused_cost:?} \
             vs contract + permute {separate_cost:?}"
        );
        // Why not fewer output buffers on the DynamicTree route: its single
        // output transform replaces the two-step route's second owned output,
        // but its transformed operand is per-call scratch (the Runtime keeps
        // no execution pool), so the two routes allocate as many output-sized
        // buffers; the bound above is what stays.
    }};
}

/// `a: V⊗V ← W⊗W` with `W = V` or `W = V*`, `b: W⊗W ← V`, `m: V ← V`.
/// Contracting `a`'s whole domain with `b`'s whole codomain is the core-GEMM
/// form (dual `W` adds the fermionic supertrace twist); `a`'s last leg with
/// `m` is not. The specs move open legs of both operands across the split and
/// change the split's size.
macro_rules! assert_output_axes_cost {
    ($engine:ident, $provider:expr, $sectors:expr $(,)?) => {{
        let runtime = Runtime::builder().build().unwrap();
        let v = GradedSpace::try_new(
            std::sync::Arc::new($provider),
            $sectors.into_iter().map(|s| (s, 3)),
        )
        .unwrap();
        for w in [v.clone(), v.try_dual().unwrap()] {
            let a = TensorMap::<_, f64>::rand_with_seed($engine, &runtime, [&v, &v], [&w, &w], 1)
                .unwrap();
            let b =
                TensorMap::<_, f64>::rand_with_seed($engine, &runtime, [&w, &w], [&v], 2).unwrap();
            let m = TensorMap::<_, f64>::rand_with_seed($engine, &runtime, [&w], [&w], 3).unwrap();
            let x = TensorMap::<_, f64>::rand_with_seed($engine, &runtime, [&w, &w], [&v, &v], 4)
                .unwrap();
            // What: the value matches an independent route. Pre-permuting the
            // left operand to `[0, 2] <- [1, 3]` puts its contracted legs at
            // `[1, 3]`, off the core form, so the one-call source-transform
            // route computes it (TensorKit's copyA). Covered for an owned and
            // a lazy-adjoint left operand. Bound: an entry is bilinear in the
            // operands, so `len(lhs) * len(rhs)` terms.
            for lhs in [a.clone(), x.adjoint($engine).unwrap()] {
                let got = lhs
                    .contract(
                        $engine,
                        &b,
                        &ContractSpec {
                            lhs: &[2, 3],
                            rhs: &[0, 1],
                            codomain: &[0, 2],
                            domain: &[1],
                        },
                    )
                    .unwrap();
                let want = lhs
                    .permute($engine, &[0, 2], &[1, 3])
                    .unwrap()
                    .contract(
                        $engine,
                        &b,
                        &ContractSpec {
                            lhs: &[1, 3],
                            rhs: &[0, 1],
                            codomain: &[0, 2],
                            domain: &[1],
                        },
                    )
                    .unwrap();
                assert_eq!(got.codomain_rank(), want.codomain_rank());
                assert_eq!(got.leg_dims().unwrap(), want.leg_dims().unwrap());
                numerics::assert_slices_close(
                    "contract(output_axes) vs pre-permuted operand",
                    got.dense_data().unwrap(),
                    want.dense_data().unwrap(),
                    a.dense_data().unwrap().len() * b.dense_data().unwrap().len(),
                );
            }
            // Every split size, with legs of both operands on each side.
            for (codomain, domain) in [
                (&[0, 2][..], &[1][..]),
                (&[0, 2, 1][..], &[][..]),
                (&[2][..], &[1, 0][..]),
                (&[][..], &[1, 2, 0][..]),
            ] {
                assert_no_costlier_than_permute!(
                    $engine,
                    runtime,
                    &a,
                    &b,
                    &[2, 3],
                    &[0, 1],
                    codomain,
                    domain,
                );
            }
            for (codomain, domain) in [
                (&[0, 3, 1][..], &[2][..]),
                (&[3][..], &[0, 2, 1][..]),
                (&[0, 3][..], &[1, 2][..]),
                (&[1, 2, 3, 0][..], &[][..]),
            ] {
                assert_no_costlier_than_permute!(
                    $engine,
                    runtime,
                    &a,
                    &m,
                    &[3],
                    &[0],
                    codomain,
                    domain,
                );
            }
        }
    }};
}

fn centered(count: i32) -> impl Iterator<Item = i32> {
    (0..count).map(move |i| i - (count - 1) / 2)
}

#[test]
fn warm_output_axes_u1_costs_no_more_than_contract_then_permute() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    assert_output_axes_cost!(engine, U1FusionRule, centered(3).map(U1Irrep::new));
}

#[test]
fn warm_output_axes_su2_costs_no_more_than_contract_then_permute() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    assert_output_axes_cost!(engine, SU2FusionRule, (0..3).map(SU2Irrep::from_twice_spin));
}

#[test]
fn warm_output_axes_fz2_u1_costs_no_more_than_contract_then_permute() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let parity = |q: i32| {
        if q.rem_euclid(2) == 0 {
            Z2Irrep::EVEN
        } else {
            Z2Irrep::ODD
        }
    };
    assert_output_axes_cost!(
        engine,
        ProductFusionRule::<FermionParityFusionRule, U1FusionRule>::try_new(
            FermionParityFusionRule,
            U1FusionRule
        )
        .unwrap(),
        centered(3).map(|q| ProductSector::new(parity(q), U1Irrep::new(q))),
    );
}

/// The process-global completed-transformer cache (`symtn::cache`).
fn completed_transformers() -> symtn::cache::StructureCacheInfo {
    symtn::cache::stats()
        .into_iter()
        .find(|info| info.kind() == symtn::cache::StructureCacheKind::CompletedTreeTransformer)
        .expect("every structure cache kind reports")
}
