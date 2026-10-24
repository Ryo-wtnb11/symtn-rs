//! #2147 performance contract: on the DynamicTree route, a lazy adjoint whose
//! own tree transform is the identity reaches the core GEMM as its parent's
//! borrowed blocks with an adjoint op, not as an operand-sized copy
//! (TensorKit `cfaa073e` `blas_contract!`, `tensoroperations.jl:383-450`:
//! `mul!` over `block(A, c)'` when `A'` needs no `copyA`; QSpace `dd2cc7e1`
//! `wbarray.cc:3472-3482`: `conj` folded into the GEMM `'C'` op).
//!
//! Fixture `permuted`: `a = p'` with `p: [V, V] ← [V]` contracted over its
//! whole domain against legs 0 and 2 of `u: [V, X, V] ← []`, so `u` is
//! permuted and the route is DynamicTree while `a` is already in core form.
//!
//! Fixture `reversed` (#2159, candidate selection): `a` contracted over its
//! domain `(1, 2)` against `u: [V, V] ← [X]`'s codomain in reverse order
//! `(1, 0)`, `|u| < |p|`. TensorKit `contract_memcost`
//! (`tensoroperations.jl:364-374`) prices m1 (keep `a`'s order) at `|u|`
//! and m2 (keep `u`'s order) at `|p|`, since `isblascontractable(a, pA)` is
//! `has_shared_permute(::AdjointTensorMap)` on the parent (free). Charging
//! `a` as a copy priced m1 at `|p| + |u|` and took m2, copying `a`.
//!
//! In both, `|p| ≠ |u|`, so an allocation of exactly `|p|` bytes is the copy
//! of `a` and one of `|u|` bytes is the (still required) copy of `u`, the
//! positive control. This is symtn's own operand copy only: the dense
//! backend's per-GEMM buffers for a strided adjoint operand (Tenferro
//! `dot_general` over the per-job ops path, one per coupled block) are
//! outside it and are disclosed in the #2147/#2159 records.

use std::hint::black_box;
use std::sync::Arc;

use symtn::sector::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use symtn::typed::{ContractSpec, GradedSpace, Runtime, TensorMap};

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

/// `(u, spec)` of a fixture over `(runtime, V, X, fill)`.
macro_rules! fixture {
    (permuted, $runtime:expr, $v:expr, $x:expr, $value:expr) => {
        (
            TensorMap::from_subblock_fn($runtime, [$v, $x, $v], [], $value).unwrap(),
            ContractSpec {
                lhs: &[1, 2],
                rhs: &[0, 2],
                codomain: &[0],
                domain: &[1],
            },
        )
    };
    (reversed, $runtime:expr, $v:expr, $x:expr, $value:expr) => {
        (
            TensorMap::from_subblock_fn($runtime, [$v, $v], [$x], $value).unwrap(),
            ContractSpec {
                lhs: &[1, 2],
                rhs: &[1, 0],
                codomain: &[0],
                domain: &[1],
            },
        )
    };
}

/// The reduced entries of `$got` and `$want` agree (both nonzero).
macro_rules! assert_close {
    ($got:expr, $want:expr) => {{
        let entries = |t: &TensorMap<_, f64>| {
            t.subblocks()
                .unwrap()
                .flat_map(|(_, view)| {
                    let shape = view.shape().to_vec();
                    (0..shape.iter().product::<usize>())
                        .map(|mut linear| {
                            let index: Vec<usize> = shape
                                .iter()
                                .map(|&extent| {
                                    let i = linear % extent;
                                    linear /= extent;
                                    i
                                })
                                .collect();
                            *view.get(&index).unwrap()
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<f64>>()
        };
        let (got, want) = (entries($got), entries($want));
        assert_eq!(got.len(), want.len());
        assert!(want.iter().any(|&w| w != 0.0));
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() <= 1e-10 * (1.0 + w.abs()), "{g} vs {w}");
        }
    }};
}

/// Asserts, at two degeneracy scales, that one call on a trimmed runtime
/// allocates nothing of exactly `|p|` bytes and at least one buffer of
/// `|u|` bytes, and (oracle: materialize `a`, then contract) that its value
/// is unchanged; `$spaces(scale)` gives `(V, X)`.
macro_rules! assert_borrowed {
    ($name:expr, $fixture:ident, $spaces:expr) => {{
        let _serial = counting_alloc::serial();
        for scale in [1, 4] {
            let (v, x) = $spaces(scale);
            let runtime = Runtime::builder().dense_threads(1).build().unwrap();
            let value = |_: &_, index: &[usize]| 0.25 + index.iter().sum::<usize>() as f64 * 0.125;
            let p: TensorMap<_, f64> =
                TensorMap::from_subblock_fn(&runtime, [&v, &v], [&v], value).unwrap();
            let (u, spec): (TensorMap<_, f64>, _) = fixture!($fixture, &runtime, &v, &x, value);
            let bytes = |t: &TensorMap<_, f64>| {
                t.subblocks()
                    .unwrap()
                    .map(|(_, view)| view.shape().iter().product::<usize>())
                    .sum::<usize>()
                    * std::mem::size_of::<f64>()
            };
            let (p_bytes, u_bytes) = (bytes(&p), bytes(&u));
            assert_ne!(p_bytes, u_bytes);
            // `reversed` selects the candidate copying the smaller operand.
            if stringify!($fixture) == "reversed" {
                assert!(u_bytes < p_bytes);
            }
            let a = p.adjoint().unwrap();
            let call = || {
                black_box(a.contract(&u, &spec).unwrap());
            };
            let got = a.contract(&u, &spec).unwrap();
            let want = a.materialize().unwrap().contract(&u, &spec).unwrap();
            assert_close!(&got, &want);
            call();
            call();
            runtime.trim_host_contract_scratch();
            let ((), p_allocs) = counting_alloc::measure_matching(p_bytes..=p_bytes, call);
            let retained = runtime.host_contract_scratch_bytes();
            runtime.trim_host_contract_scratch();
            let ((), u_allocs) = counting_alloc::measure_matching(u_bytes..=u_bytes, call);
            let (p_copies, u_copies) = (p_allocs.matched_calls, u_allocs.matched_calls);
            eprintln!(
                "{} {} x{scale}: |p| {p_bytes} B copied {p_copies}x, |u| {u_bytes} B copied \
                 {u_copies}x; call {} allocs {} B peak {} B; retained scratch {retained} B",
                $name,
                stringify!($fixture),
                p_allocs.calls,
                p_allocs.bytes,
                p_allocs.peak_live_bytes,
            );
            // What: no allocation the size of `a`'s payload, while the
            // permuted `u` is still copied into its own scratch (the probe
            // sees operand copies).
            let what = format!("{} {} x{scale}", $name, stringify!($fixture));
            assert_eq!(p_copies, 0, "{what}");
            assert!(u_copies >= 1, "{what}");
        }
    }};
}

fn su2_spaces(scale: usize) -> (GradedSpace<SU2FusionRule>, GradedSpace<SU2FusionRule>) {
    let space = |degs: [usize; 3]| {
        GradedSpace::try_new(
            Arc::new(SU2FusionRule),
            [0, 1, 2]
                .into_iter()
                .zip(degs)
                .map(|(twice, deg)| (SU2Irrep::from_twice_spin(twice), deg * scale)),
        )
        .unwrap()
    };
    (space([2, 3, 2]), space([1, 1, 3]))
}

fn u1_spaces(scale: usize) -> (GradedSpace<U1FusionRule>, GradedSpace<U1FusionRule>) {
    let space = |degs: [usize; 3]| {
        GradedSpace::try_new(
            Arc::new(U1FusionRule),
            [-1, 0, 1]
                .into_iter()
                .zip(degs)
                .map(|(charge, deg)| (U1Irrep::new(charge), deg * scale)),
        )
        .unwrap()
    };
    (space([2, 3, 2]), space([1, 1, 3]))
}

#[cfg(feature = "cgc-gen")]
fn checked_su2_spaces(
    scale: usize,
) -> (
    GradedSpace<symtn::sector::LieGroupFusionRule>,
    GradedSpace<symtn::sector::LieGroupFusionRule>,
) {
    let space = |degs: [usize; 3]| {
        GradedSpace::try_new(
            Arc::new(
                symtn::sector::LieGroupFusionRule::new(
                    fusion_symbols::lie::GroupId::su(2).unwrap(),
                )
                .unwrap(),
            ),
            [0i64, 1, 2]
                .into_iter()
                .zip(degs)
                .map(|(twice, deg)| (vec![twice], deg * scale)),
        )
        .unwrap()
    };
    (space([2, 3, 2]), space([1, 1, 3]))
}

#[test]
fn su2_identity_adjoint_is_borrowed_on_the_dynamic_tree_route() {
    assert_borrowed!("SU(2)", permuted, su2_spaces);
}

#[test]
fn u1_identity_adjoint_is_borrowed_on_the_dynamic_tree_route() {
    assert_borrowed!("U(1)", permuted, u1_spaces);
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_su2_identity_adjoint_is_borrowed_on_the_dynamic_tree_route() {
    assert_borrowed!("checked SU(2)", permuted, checked_su2_spaces);
}

#[test]
fn su2_selection_prices_a_borrowable_identity_adjoint_as_free() {
    assert_borrowed!("SU(2)", reversed, su2_spaces);
}

#[test]
fn u1_selection_prices_a_borrowable_identity_adjoint_as_free() {
    assert_borrowed!("U(1)", reversed, u1_spaces);
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_su2_selection_prices_a_borrowable_identity_adjoint_as_free() {
    assert_borrowed!("checked SU(2)", reversed, checked_su2_spaces);
}
