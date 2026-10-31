//! Export firewall for `symtn-dense`.
//!
//! Pins the exact names `lib.rs` re-exports, under every feature combination
//! at once, so adding or removing one fails here and is reviewed as an API
//! change. `DensePlacement` (one variant, `Host`, with four accessors that
//! always returned it), the `CUTENSOR_PLAN_BYTES` estimate and the test-only
//! `cuda_is_hermitian_region` were removed from this surface.
//!
//! Why a source scan and not an import list: an import list only proves
//! presence, not absence.

use std::collections::BTreeSet;

use crate::export_scan::reexports;

/// Leaf names re-exported by `pub use` in `source`.
fn exported_names(source: &str) -> Result<BTreeSet<String>, String> {
    let exports = reexports(source)?.without_modules()?;
    Ok(exports.names.into_keys().collect())
}

#[test]
fn firewall_rejects_every_inline_public_item() {
    for item in [
        "pub mod m {}",
        "pub fn f() {}",
        "pub struct S;",
        "pub enum E {}",
        "pub trait T {}",
        "pub type A = u8;",
        "pub const C: u8 = 0;",
        "pub static S: u8 = 0;",
        "pub union U { a: u8 }",
        "pub unsafe fn f() {}",
        "pub extern crate core;",
        "#[macro_export]\nmacro_rules! m { () => {} }",
    ] {
        let source = format!("pub use a::B;\n{item}\n");
        assert!(exported_names(&source).is_err(), "accepted `{item}`");
    }
}

#[test]
fn firewall_ignores_comments_strings_and_restricted_visibility() {
    let source = r#"
        //! pub fn doc() {}
        /* pub enum Hidden {} */
        #[path = "pub fn x.rs"]
        pub(crate) mod inner;
        pub(super) fn helper() {}
        pub use a::{B, c::D as E}; // pub struct Trailing;
        pub use f::G;
    "#;
    let names = exported_names(source).unwrap();
    assert_eq!(names, ["B", "E", "G"].map(str::to_owned).into());
}

#[test]
fn symtn_dense_exports_exactly_the_pinned_names() {
    let expected = [
        "CpuEngine",
        "CudaDenseContext",
        "CudaDenseStorage",
        "CudaDeviceId",
        "CudaEngine",
        "CudaRealScalar",
        "CudaRegion",
        "CudaRegionBeta",
        "CudaRegionCoefficient",
        "CudaScalar",
        "CudaScalarOperands",
        "CudaSpectrum",
        "CudaSvdGaugeWeights",
        "CudaSvdPhases",
        "CudaTransferStats",
        "DefaultDenseExecutor",
        "DenseBackend",
        "DenseDType",
        "DenseDotConfig",
        "DenseError",
        "DenseExecutor",
        "DenseFactorization",
        "DenseFreshSegments",
        "DenseGemmBatchJob",
        "DenseLinalgScopeBody",
        "DenseOwned",
        "DenseRead",
        "DenseScalar",
        "DenseTensor",
        "DenseView",
        "DenseViewMut",
        "DenseWrite",
        "Engine",
        "MatrixOp",
        "Pool",
        "arity_mismatch",
        "check_contiguous_output",
        "cuda_all_finite",
        "cuda_conj",
        "cuda_copy_region_into",
        "cuda_copy_spectrum_into",
        "cuda_copy_strided_into",
        "cuda_download_batched_spectra",
        "cuda_download_spectra",
        "cuda_eigh_region",
        "cuda_eigh_region_batched",
        "cuda_gather_columns_batched_into",
        "cuda_gather_elements",
        "cuda_gather_member_elements",
        "cuda_gather_members",
        "cuda_gemm_region_batched_into",
        "cuda_gemm_region_batched_scaled_into",
        "cuda_gemm_region_into",
        "cuda_gemm_region_with_ops_into",
        "cuda_hermitian_regions",
        "cuda_hermitian_regions_batched",
        "cuda_matmul_region_into",
        "cuda_qr_region",
        "cuda_region_axpby",
        "cuda_region_scale",
        "cuda_region_trace_accumulate",
        "cuda_region_zero",
        "cuda_svd_gauge_phases",
        "cuda_svd_region",
        "cuda_transfer_stats",
        "cuda_widen",
        "reset_cuda_transfer_stats",
        "strided_batch_runs",
        "strided_batch_runs_into",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<BTreeSet<_>>();
    assert_eq!(
        exported_names(include_str!("../src/lib.rs")).unwrap(),
        expected
    );
}
