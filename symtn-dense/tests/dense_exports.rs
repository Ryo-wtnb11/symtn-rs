//! Export firewall for `symtn-dense`.
//!
//! Pins the exact names `lib.rs` re-exports and its one public module
//! (`lanes`), under every feature combination at once, so adding or removing
//! one fails here and is reviewed as an API change.
//!
//! Why a source scan and not an import list: an import list only proves
//! presence, not absence.

use std::collections::BTreeSet;

use crate::export_scan::{not_unrestricted_pub, reexports, strip, use_leaves};

/// Leaf names re-exported by `pub use` in `source`.
fn exported_names(source: &str) -> Result<BTreeSet<String>, String> {
    let exports = reexports(source)?;
    // The one public module: the `#[doc(hidden)]` lane runner that
    // symtn-operations and symtn-tensors drive their block jobs through,
    // whose own declarations `the_lanes_module_declares_exactly_the_pinned_names`
    // pins.
    if let Some(module) = exports.modules.iter().find(|module| *module != "lanes") {
        return Err(format!("lib.rs gained a `pub mod {module}`"));
    }
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
        "EIGH_EIGENVALUE_CHECK",
        "Engine",
        "FactorDest",
        "FactorFailure",
        "FactorPlacement",
        "MatrixOp",
        "Pool",
        "arity_mismatch",
        "check_contiguous_output",
        "finite_eigenvalues",
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

/// Every unrestricted public declaration of `source` in source order, as
/// `kind name`: a `pub use` leaf as `use Name`, a public field as
/// `field name`. Methods and fields count, so a new public method or field
/// of a lane type fails the pin as a new free item does.
fn public_declarations(source: &str) -> Vec<String> {
    let code = strip(source);
    let mut out = Vec::new();
    for (at, _) in code.match_indices("pub") {
        if not_unrestricted_pub(&code, at) {
            continue;
        }
        let item = code[at + 3..].trim_start();
        let mut words = item
            .split(|c: char| !(c.is_alphanumeric() || c == '_'))
            .filter(|word| !word.is_empty())
            .peekable();
        let mut kind = words.next().unwrap_or_default();
        while matches!(kind, "unsafe" | "async" | "extern")
            || (kind == "const" && words.peek() == Some(&"fn"))
        {
            kind = words.next().unwrap_or_default();
        }
        match kind {
            "use" => {
                let tree = item["use".len()..item.find(';').expect("unterminated pub use")].trim();
                for leaf in use_leaves(tree).unwrap() {
                    out.push(format!("use {leaf}"));
                }
            }
            "fn" | "struct" | "enum" | "trait" | "type" | "const" | "static" | "mod" | "union" => {
                out.push(format!("{kind} {}", words.next().unwrap_or_default()));
            }
            field => out.push(format!("field {field}")),
        }
    }
    out
}

#[test]
fn declaration_scan_sees_methods_fields_and_uses_only_when_unrestricted() {
    let source = r#"
        // pub fn hidden() {}
        pub use a::{B, c::D as E};
        pub struct S { pub x: u8, pub(crate) y: u8 }
        impl S { pub const fn f<T>() {} pub(super) fn g() {} }
        pub(crate) fn narrowed() {}
    "#;
    assert_eq!(
        public_declarations(source),
        ["use B", "use E", "struct S", "field x", "fn f"]
    );
}

/// The lane runner's public subtree, file by file, is exactly the pinned
/// declarations: `lib.rs` exempts the `lanes` module from its own pin, so
/// this is where a lane API change is reviewed.
#[test]
fn the_lanes_module_declares_exactly_the_pinned_names() {
    let facade: &[&str] = &[
        "use BatchPolicy",
        "use Exec",
        "use flat_width",
        "use run_native",
        "use Assignment",
        "use Cut",
        "use Each",
        "use Outcome",
        "use Refusal",
        "use Split",
        "use Work",
    ];
    for (file, source, expected) in [
        ("lanes.rs", include_str!("../src/lanes.rs"), facade),
        (
            "lanes/placement.rs",
            include_str!("../src/lanes/placement.rs"),
            &["fn flat_width"],
        ),
        (
            "lanes/runner.rs",
            include_str!("../src/lanes/runner.rs"),
            &[
                "trait Work",
                "type Outcome",
                "trait Assignment",
                // The sealed supertrait, in a private module.
                "trait Assign",
                "struct Split",
                "field threads",
                "field split",
                "struct Cut",
                "field job",
                "field left",
                "field threads",
                "struct Each",
                "fn run_native",
                "enum Refusal",
            ],
        ),
    ] {
        assert_eq!(public_declarations(source), expected, "{file}");
    }
}
