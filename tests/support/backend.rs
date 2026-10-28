//! The one test predicate for "the compiled CPU provider is faer".
//!
//! The provider is a compile-time choice: every BLAS-provider feature of
//! symtn enables the dense layer's `cpu-blas-core`, and Tenferro's `native`
//! and `blas` providers are mutually exclusive, so faer is compiled iff none
//! of them is enabled. Gates that only hold on faer (allocation budgets, bit
//! pins) branch on this.

pub fn faer_is_default_backend() -> bool {
    !cfg!(any(
        feature = "cpu-blas",
        feature = "blas-accelerate",
        feature = "blas-openblas",
        feature = "blas-mkl",
        feature = "provider-inject"
    ))
}
