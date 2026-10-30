//! The one test predicate for "the compiled CPU provider is faer".
//!
//! The provider is selected by cargo feature: every symtn feature below
//! enables the dense layer's BLAS/LAPACK provider (`cpu-blas-core`), and any
//! other build compiles faer. Gates that only hold on faer (allocation
//! budgets, bit pins) branch on this at run time.

pub fn faer_is_default_backend() -> bool {
    !cfg!(any(
        feature = "cpu-blas",
        feature = "blas-accelerate",
        feature = "blas-openblas",
        feature = "blas-mkl",
        feature = "provider-inject"
    ))
}
