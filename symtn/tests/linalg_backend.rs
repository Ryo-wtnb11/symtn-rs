//! Selectable CPU linear-algebra backend (#64): `RuntimeBuilder::linalg_backend`
//! chooses the dense SVD/QR/eigh provider. Backend choice is performance-only —
//! results stay identical across providers to numerical precision.

// Every test selects the Faer provider explicitly, so the file only exists in
// builds that compile it (#2107).
#![cfg(feature = "cpu-faer")]

use std::sync::Arc;

use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{GradedSpace, LinalgBackend, Runtime, RuntimeBuilder, TensorMap};

/// Whether this build links a BLAS/LAPACK provider.
const BLAS_COMPILED: bool = cfg!(any(
    feature = "cpu-blas",
    feature = "blas-accelerate",
    feature = "blas-openblas",
    feature = "blas-mkl"
));

/// Builds a runtime that selects `Blas`: the build succeeds exactly when a
/// BLAS provider is compiled, and fails in `build` otherwise (documented on
/// `RuntimeBuilder::linalg_backend`/`gemm_backend`).
fn blas_runtime(builder: RuntimeBuilder) -> Option<Runtime> {
    let built = builder.build();
    assert_eq!(
        built.is_ok(),
        BLAS_COMPILED,
        "Blas provider build with BLAS compiled = {BLAS_COMPILED}: {:?}",
        built.as_ref().err()
    );
    built.ok()
}

fn u1_space() -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 3),
            (U1Irrep::new(1), 2),
        ],
    )
    .unwrap()
}

fn spectra(rt: &Runtime) -> Vec<f64> {
    let v = u1_space();
    let t = TensorMap::<U1FusionRule, f64>::rand_with_seed(rt, [&v, &v], [&v, &v], 101).unwrap();
    let mut all: Vec<f64> = t
        .svd_vals(&[0, 1], &[2, 3])
        .unwrap()
        .into_iter()
        .flat_map(|s| s.values)
        .collect();
    all.sort_by(|a, b| b.partial_cmp(a).unwrap());
    all
}

/// Frobenius norm of a coupled-block contraction — exercises the `gemm_backend`
/// (contraction GEMM) seam rather than the factorization seam.
fn contraction_norm(rt: &Runtime) -> f64 {
    let v = u1_space();
    let a = TensorMap::<U1FusionRule, f64>::rand_with_seed(rt, [&v], [&v], 11).unwrap();
    let b = TensorMap::<U1FusionRule, f64>::rand_with_seed(rt, [&v], [&v], 12).unwrap();
    a.compose(&b).unwrap().norm(2.0).unwrap()
}

#[test]
fn faer_backend_builds_and_computes() {
    // The explicit Faer provider must build and produce a
    // sane spectrum (descending, all non-negative singular values).
    let rt = Runtime::builder()
        .linalg_backend(LinalgBackend::Faer)
        .build()
        .unwrap();
    let s = spectra(&rt);
    assert!(!s.is_empty());
    assert!(s.iter().all(|&x| x >= 0.0));
    for pair in s.windows(2) {
        assert!(pair[0] >= pair[1] - 1e-12);
    }
}

#[test]
fn linalg_backend_composes_with_dense_threads() {
    // Selecting a provider together with an explicit thread count routes through
    // the threads-and-kind constructor.
    let rt = Runtime::builder()
        .linalg_backend(LinalgBackend::Faer)
        .dense_threads(1)
        .build()
        .unwrap();
    assert!(!spectra(&rt).is_empty());
}

#[test]
fn gemm_backend_builds_and_contracts() {
    // The contraction-GEMM provider is selected independently of the
    // factorization provider; Faer requested explicitly.
    let rt = Runtime::builder()
        .gemm_backend(LinalgBackend::Faer)
        .build()
        .unwrap();
    assert!(contraction_norm(&rt) > 0.0);
}

#[test]
fn gemm_backend_composes_with_dense_threads() {
    let rt = Runtime::builder()
        .gemm_backend(LinalgBackend::Faer)
        .dense_threads(1)
        .build()
        .unwrap();
    assert!(contraction_norm(&rt) > 0.0);
}

#[test]
fn linalg_and_gemm_backends_are_independent() {
    // Factorization on one provider, contraction GEMM on another — the two
    // seams are wired separately and must not collide. (Both faer here, but the
    // call sets each independently.)
    let rt = Runtime::builder()
        .linalg_backend(LinalgBackend::Faer)
        .gemm_backend(LinalgBackend::Faer)
        .build()
        .unwrap();
    assert!(!spectra(&rt).is_empty());
    assert!(contraction_norm(&rt) > 0.0);
}

#[test]
fn faer_and_blas_agree_when_blas_is_available() {
    let faer = Runtime::builder()
        .linalg_backend(LinalgBackend::Faer)
        .build()
        .unwrap();
    let Some(blas) = blas_runtime(Runtime::builder().linalg_backend(LinalgBackend::Blas)) else {
        return;
    };
    let faer_s = spectra(&faer);
    let blas_s = spectra(&blas);
    assert_eq!(faer_s.len(), blas_s.len());
    for (a, b) in faer_s.iter().zip(&blas_s) {
        assert!((a - b).abs() <= 1e-10, "faer {a} vs blas {b}");
    }
}

#[test]
fn gemm_faer_and_blas_agree_when_blas_is_available() {
    // Same idea for the contraction-GEMM seam: a contraction on faer and on the
    // linked BLAS must agree to numerical precision.
    let faer = Runtime::builder()
        .gemm_backend(LinalgBackend::Faer)
        .build()
        .unwrap();
    let Some(blas) = blas_runtime(Runtime::builder().gemm_backend(LinalgBackend::Blas)) else {
        return;
    };
    let f = contraction_norm(&faer);
    let b = contraction_norm(&blas);
    assert!((f - b).abs() <= 1e-10, "faer {f} vs blas {b}");
}
