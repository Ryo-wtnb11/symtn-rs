//! A repeated compact factorization of one structure reuses every cached
//! structure and builds its factor plan at a cost independent of the
//! degeneracies; only the dense kernels and the owned outputs scale with
//! them, and those are measured elsewhere.

use std::hint::black_box;
use std::sync::Arc;
use symtn::typed::HermitianTol;

use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::Runtime;
use symtn::typed::{GradedSpace, TensorMap};

use crate::counting_alloc;

/// Structure-cache misses so far, over every kind.
fn structure_misses() -> u64 {
    symtn::cache::stats()
        .into_iter()
        .map(|info| info.misses())
        .sum()
}

/// Rank-4 U(1) tensor with five coupled sectors (charges -2..=2) and
/// degeneracy `degeneracy` per leg sector, so every factor region table is
/// nontrivial.
fn tensor(runtime: &Runtime, degeneracy: usize) -> TensorMap<U1FusionRule, f64> {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let space = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [-1, 0, 1].map(|charge| (U1Irrep::new(charge), degeneracy)),
    )
    .unwrap();
    TensorMap::rand_with_seed(engine, runtime, [&space, &space], [&space, &space], 1216).unwrap()
}

/// Allocation calls and structure-cache misses of the second call of
/// `factor` (the first warms the caches and its factors stay alive, so both
/// factor structures are live).
fn second_call<T>(mut factor: impl FnMut() -> T) -> (u64, u64) {
    let first = factor();
    let misses = structure_misses();
    let (second, allocs) = counting_alloc::measure(&mut factor);
    let misses = structure_misses() - misses;
    black_box((&first, &second));
    (allocs.calls, misses)
}

/// `(name, (allocation calls, structure misses))` of the second compact
/// SVD, QR and Hermitian eigendecomposition of the degeneracy-2 and the
/// degeneracy-3 fixture.
fn second_factorizations() -> Vec<[(&'static str, (u64, u64)); 3]> {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    [2, 3]
        .into_iter()
        .map(|degeneracy| {
            let a = tensor(&runtime, degeneracy);
            let hermitian = a.adjoint(engine).unwrap().compose(engine, &a).unwrap();
            [
                (
                    "svd_compact",
                    second_call(|| a.svd_compact(engine, &[0, 1], &[2, 3]).unwrap()),
                ),
                (
                    "qr_compact",
                    second_call(|| a.qr_compact(engine, &[0, 1], &[2, 3]).unwrap()),
                ),
                (
                    "eigh_full",
                    second_call(|| {
                        hermitian
                            .eigh_full(engine, &[0, 1], &[2, 3], HermitianTol::DEFAULT)
                            .unwrap()
                    }),
                ),
            ]
        })
        .collect()
}

/// Purpose: a second compact SVD, QR and Hermitian eigendecomposition of one
/// structure rebuilds no cached structure.
#[test]
fn second_compact_factorization_rebuilds_no_structure() {
    let _serial = crate::counting_alloc::serial();
    for (name, (_, misses)) in second_factorizations().into_iter().flatten() {
        assert_eq!(misses, 0, "{name}: a warm call rebuilt a structure");
    }
}

/// Purpose: the factor plan a compact factorization builds per call is a
/// per-structure constant: the second call costs the same at degeneracy 2
/// and 3.
#[test]
#[cfg_attr(
    any(
        feature = "cpu-blas",
        feature = "blas-accelerate",
        feature = "blas-openblas",
        feature = "blas-mkl",
        feature = "provider-inject"
    ),
    ignore = "a LAPACK provider sizes its own workspace allocations by the block extents"
)]
fn second_compact_factorization_costs_independently_of_degeneracy() {
    let _serial = crate::counting_alloc::serial();
    let rows = second_factorizations();
    for ((name, (small, _)), (_, (large, _))) in rows[0].into_iter().zip(rows[1]) {
        counting_alloc::assert_same_calls(name, small, large);
    }
}
