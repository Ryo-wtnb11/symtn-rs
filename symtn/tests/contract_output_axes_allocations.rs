//! Warm eager `contract` with a non-default output order or codomain/domain
//! split costs no more than `contract` in the default order and split
//! followed by `permute` (#1461, #1549), and returns the same tensor. TensorKit
//! `blas_contract!` takes the same shape of work when the output layout is not
//! the GEMM's: a temporary plus a permuting `tensoradd!` (its `copyC` path),
//! inside the one call.

use std::hint::black_box;

use num_complex::{Complex32, Complex64};
use symtn::sector::{
    FermionParityFusionRule, ProductFusionRule, ProductSector, SU2FusionRule, SU2Irrep,
    U1FusionRule, U1Irrep, Z2Irrep,
};
use symtn::typed::{ContractSpec, GradedSpace, Runtime, TensorMap};

#[path = "../../tests/support/numerics.rs"]
mod numerics;

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

/// (calls, bytes) allocated while `f` runs; the result is dropped afterwards.
fn allocations<T>(f: impl FnOnce() -> T) -> ((u64, u64), T) {
    let (value, allocs) = counting_alloc::measure(f);
    ((allocs.calls, allocs.bytes), value)
}

/// Warm cost and value of `contract` onto `codomain ← domain` against the
/// default order and split followed by `permute(codomain, domain)`.
/// `strict` also requires fewer bytes: the route has no separate output copy.
macro_rules! assert_no_costlier_than_permute {
    ($runtime:expr, $a:expr, $b:expr, $lhs_axes:expr, $rhs_axes:expr,
     $codomain:expr, $domain:expr, $strict:expr $(,)?) => {{
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
        let fused = || black_box($a).contract($b, &spec).unwrap();
        let separate = || {
            black_box($a)
                .contract($b, &default)
                .unwrap()
                .permute(codomain, domain)
                .unwrap()
        };
        fused();
        separate();
        let misses = completed_transformers().misses();
        let (fused_cost, fused_value) = allocations(fused);
        let (separate_cost, separate_value) = allocations(separate);

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
        // What: no more allocation calls or bytes than the two-step route.
        // Equality is TensorKit's `copyC`: a zero-copy candidate contracts
        // into a temporary that one permute moves into the result.
        assert!(
            fused_cost.0 <= separate_cost.0 && fused_cost.1 <= separate_cost.1,
            "contract({lhs_axes:?}, {rhs_axes:?}; {codomain:?} <- {domain:?}) {fused_cost:?} \
             vs contract + permute {separate_cost:?}"
        );
        // What: the DynamicTree route's single output transform replaces the
        // two-step route's second owned output, so it moves strictly fewer
        // bytes. An upper bound, not an exact count.
        assert!(
            !$strict || fused_cost.1 < separate_cost.1,
            "contract({lhs_axes:?}, {rhs_axes:?}; {codomain:?} <- {domain:?}) {fused_cost:?} \
             must allocate fewer bytes than contract + permute {separate_cost:?}"
        );
    }};
}

/// `a: V⊗V ← W⊗W` with `W = V` or `W = V*`, `b: W⊗W ← V`, `m: V ← V`.
/// Contracting `a`'s whole domain with `b`'s whole codomain is the core-GEMM
/// form (dual `W` adds the fermionic supertrace twist); `a`'s last leg with
/// `m` is not. The specs move open legs of both operands across the split and
/// change the split's size.
macro_rules! assert_output_axes_cost {
    ($provider:expr, $sectors:expr $(,)?) => {{
        let _guard = counting_alloc::serial();
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let v = GradedSpace::try_new(
            std::sync::Arc::new($provider),
            $sectors.into_iter().map(|s| (s, 3)),
        )
        .unwrap();
        for w in [v.clone(), v.try_dual().unwrap()] {
            let a = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v, &v], [&w, &w], 1).unwrap();
            let b = TensorMap::<_, f64>::rand_with_seed(&runtime, [&w, &w], [&v], 2).unwrap();
            let m = TensorMap::<_, f64>::rand_with_seed(&runtime, [&w], [&w], 3).unwrap();
            let x = TensorMap::<_, f64>::rand_with_seed(&runtime, [&w, &w], [&v, &v], 4).unwrap();
            // What: the value matches an independent route. Pre-permuting the
            // left operand to `[0, 2] <- [1, 3]` puts its contracted legs at
            // `[1, 3]`, off the core form, so the one-call source-transform
            // route computes it (TensorKit's copyA). Covered for an owned and
            // a lazy-adjoint left operand. Bound: an entry is bilinear in the
            // operands, so `len(lhs) * len(rhs)` terms.
            for lhs in [a.clone(), x.adjoint().unwrap()] {
                let got = lhs
                    .contract(
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
                    .permute(&[0, 2], &[1, 3])
                    .unwrap()
                    .contract(
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
                    runtime,
                    &a,
                    &b,
                    &[2, 3],
                    &[0, 1],
                    codomain,
                    domain,
                    // TensorKit's `copyC`: equal to the two-step route.
                    false,
                );
            }
            for (codomain, domain) in [
                (&[0, 3, 1][..], &[2][..]),
                (&[3][..], &[0, 2, 1][..]),
                (&[0, 3][..], &[1, 2][..]),
                (&[1, 2, 3, 0][..], &[][..]),
            ] {
                assert_no_costlier_than_permute!(
                    runtime,
                    &a,
                    &m,
                    &[3],
                    &[0],
                    codomain,
                    domain,
                    true,
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
    assert_output_axes_cost!(U1FusionRule, centered(3).map(U1Irrep::new));
}

#[test]
fn warm_output_axes_su2_costs_no_more_than_contract_then_permute() {
    assert_output_axes_cost!(SU2FusionRule, (0..3).map(SU2Irrep::from_twice_spin));
}

#[test]
fn warm_output_axes_fz2_u1_costs_no_more_than_contract_then_permute() {
    let parity = |q: i32| {
        if q.rem_euclid(2) == 0 {
            Z2Irrep::EVEN
        } else {
            Z2Irrep::ODD
        }
    };
    assert_output_axes_cost!(
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
