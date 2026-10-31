//! Export firewall for `symtn-dense` (#1805).
//!
//! Pins the exact names `lib.rs` re-exports, under every feature combination
//! at once, so adding or removing one fails here and is reviewed as an API
//! change. `DensePlacement` (one variant, `Host`, with four accessors that
//! always returned it), the `CUTENSOR_PLAN_BYTES` estimate and the test-only
//! `cuda_is_hermitian_region` were removed from this surface (#1805).
//!
//! Why a source scan and not an import list: an import list only proves
//! presence, not absence.

use std::collections::BTreeSet;

/// Drops comments (line, doc and `/* */`) and blanks string literals so that
/// neither can hide or fake an item.
fn strip(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut rest = source;
    while let Some(ch) = rest.chars().next() {
        if rest.starts_with("//") {
            rest = &rest[rest.find('\n').unwrap_or(rest.len())..];
        } else if rest.starts_with("/*") {
            rest = &rest[rest.find("*/").map_or(rest.len(), |end| end + 2)..];
            out.push(' ');
        } else if ch == '"' {
            let mut escaped = false;
            let end = rest[1..]
                .find(|c: char| {
                    let close = c == '"' && !escaped;
                    escaped = c == '\\' && !escaped;
                    close
                })
                .map_or(rest.len(), |end| end + 2);
            rest = &rest[end..];
            out.push_str("\"\"");
        } else {
            out.push(ch);
            rest = &rest[ch.len_utf8()..];
        }
    }
    out
}

/// Leaf names re-exported by `pub use` in `source`, or an error for any
/// other inline public item (`pub fn`, `pub enum`, `pub const`, ...) or
/// `#[macro_export]`, which this firewall would otherwise not see.
fn exported_names(source: &str) -> Result<BTreeSet<String>, String> {
    let code = strip(source);
    if code.contains("macro_export") {
        return Err("lib.rs exports a macro_rules! macro".into());
    }
    let mut names = BTreeSet::new();
    for (at, _) in code.match_indices("pub") {
        let before = code[..at].chars().next_back();
        if before.is_some_and(|c| c.is_alphanumeric() || c == '_') {
            continue;
        }
        let after = &code[at + 3..];
        if after.starts_with('(') || after.starts_with(|c: char| c.is_alphanumeric() || c == '_') {
            continue; // `pub(crate)` etc., or an identifier such as `public`
        }
        let item = after.trim_start();
        let Some(tree) = item
            .strip_prefix("use")
            .filter(|t| t.starts_with(char::is_whitespace))
        else {
            let keyword = item.split_whitespace().next().unwrap_or("");
            return Err(format!("lib.rs gained an inline `pub {keyword}` item"));
        };
        let tree = tree[..tree.find(';').ok_or("unterminated pub use")?].trim();
        let leaves = match tree.find('{') {
            Some(open) => {
                let inner = tree[open + 1..]
                    .trim_end()
                    .strip_suffix('}')
                    .ok_or("bad use tree")?;
                if inner.contains('{') {
                    return Err(format!("nested use tree: {tree}"));
                }
                inner.split(',').collect::<Vec<_>>()
            }
            None => vec![tree],
        };
        for leaf in leaves
            .iter()
            .map(|leaf| leaf.trim())
            .filter(|l| !l.is_empty())
        {
            let name = leaf.rsplit(" as ").next().unwrap();
            names.insert(name.rsplit("::").next().unwrap().trim().to_owned());
        }
    }
    Ok(names)
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
