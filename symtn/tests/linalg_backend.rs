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
fn every_backend_selection_builds_and_computes() {
    // The factorization and contraction-GEMM providers are selected
    // independently and compose with an explicit thread count; every
    // combination builds and yields a sane descending non-negative spectrum
    // and a nonzero contraction.
    let runtimes = [
        (
            "linalg",
            Runtime::builder()
                .linalg_backend(LinalgBackend::Faer)
                .build()
                .unwrap(),
        ),
        (
            "linalg + threads",
            Runtime::builder()
                .linalg_backend(LinalgBackend::Faer)
                .dense_threads(1)
                .build()
                .unwrap(),
        ),
        (
            "gemm",
            Runtime::builder()
                .gemm_backend(LinalgBackend::Faer)
                .build()
                .unwrap(),
        ),
        (
            "gemm + threads",
            Runtime::builder()
                .gemm_backend(LinalgBackend::Faer)
                .dense_threads(1)
                .build()
                .unwrap(),
        ),
        (
            "linalg + gemm",
            Runtime::builder()
                .linalg_backend(LinalgBackend::Faer)
                .gemm_backend(LinalgBackend::Faer)
                .build()
                .unwrap(),
        ),
    ];
    for (name, rt) in runtimes {
        let s = spectra(&rt);
        assert!(!s.is_empty(), "{name}");
        assert!(s.iter().all(|&x| x >= 0.0), "{name}");
        for pair in s.windows(2) {
            assert!(pair[0] >= pair[1] - 1e-12, "{name}");
        }
        assert!(contraction_norm(&rt) > 0.0, "{name}");
    }
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
