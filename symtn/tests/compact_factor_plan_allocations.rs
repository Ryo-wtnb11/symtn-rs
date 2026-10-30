use std::hint::black_box;
use std::sync::Arc;
use symtn::typed::HermitianTol;

use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::Runtime;
use symtn::typed::{Eigh, GradedSpace, Qr, Svd, TensorMap};

use crate::counting_alloc;

fn measured<T>(operation: impl FnOnce() -> T) -> (T, u64) {
    let (value, allocs) = counting_alloc::measure(operation);
    (value, allocs.calls)
}

/// Rank-4 U(1) tensor with five coupled sectors (charges -2..=2) and
/// degeneracy 2 per leg sector, so every factor region table is nontrivial.
fn tensor(runtime: &Runtime) -> TensorMap<U1FusionRule, f64> {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let space = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 2),
            (U1Irrep::new(1), 2),
        ],
    )
    .unwrap();
    TensorMap::rand_with_seed(engine, runtime, [&space, &space], [&space, &space], 1216).unwrap()
}

// Allocation-call upper bounds on the SECOND compact factorization of one
// tensor (first call warms the structure caches; the first call's factors are
// kept alive so both factor structures stay live), single dense thread.
// Everything outside `compact_factor_plan` (dense kernels, output storage,
// spectra) is fixed, so the bound pins the plan's per-call constant
// (about 6 allocations, #1216). Measured with the pinned faer provider:
// svd_compact 144, qr_compact 111, eigh_full 120.
#[test]
fn second_compact_factorization_builds_the_plan_with_a_bounded_constant() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let a = tensor(&runtime);
    let warm_svd = a.svd_compact(engine, &[0, 1], &[2, 3]).unwrap();
    let (Svd { u, s, vh }, svd_calls) =
        measured(|| a.svd_compact(engine, &[0, 1], &[2, 3]).unwrap());
    black_box((&u, &s, &vh));
    assert!(
        svd_calls <= 144,
        "second svd_compact allocated {svd_calls} times"
    );
    drop(warm_svd);

    let warm_qr = a.qr_compact(engine, &[0, 1], &[2, 3]).unwrap();
    let (Qr { q, r }, qr_calls) = measured(|| a.qr_compact(engine, &[0, 1], &[2, 3]).unwrap());
    black_box((&q, &r));
    assert!(
        qr_calls <= 111,
        "second qr_compact allocated {qr_calls} times"
    );
    drop(warm_qr);

    let hermitian = a.adjoint(engine).unwrap().compose(engine, &a).unwrap();
    let warm_eigh = hermitian
        .eigh_full(engine, &[0, 1], &[2, 3], HermitianTol::DEFAULT)
        .unwrap();
    let (Eigh { d, v }, eigh_calls) = measured(|| {
        hermitian
            .eigh_full(engine, &[0, 1], &[2, 3], HermitianTol::DEFAULT)
            .unwrap()
    });
    black_box((&d, &v));
    assert!(
        eigh_calls <= 120,
        "second eigh_full allocated {eigh_calls} times"
    );
    drop(warm_eigh);
}
