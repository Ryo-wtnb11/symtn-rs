//! Export firewall for symtn-matrixalgebra.
//!
//! The documented crate root carries the spectrum, scalar, factor and result
//! types; every entry point the `symtn` facade dispatches to sits in the
//! doc-hidden `seam` module. This test pins the exact names both export,
//! under every feature combination at once, so adding or removing one fails
//! here and is reviewed as an API change. The typed const-generic layer
//! (`svd_compact`, `BoundTensorMap`, ...) and the dense compact-diagonal
//! polar seam were removed because nothing outside this crate's tests
//! called them.
//!
//! Why a source scan and not an import list: an import list only proves
//! presence, not absence. Modelled on `symtn-network/tests/network_exports.rs`.

use std::collections::BTreeSet;

use crate::export_scan::{reexports, strip};

/// `pub use` leaves and every `pub mod <name>;`.
fn exported_names(code: &str) -> Result<BTreeSet<String>, String> {
    let exports = reexports(code)?;
    Ok(exports.names.into_keys().chain(exports.modules).collect())
}

/// Root and `seam` exports of `source`: the one inline module allowed is
/// `pub mod seam { .. }`, whose body is scanned on its own and replaced by
/// `pub mod seam;` in the root scan.
fn exports(source: &str) -> Result<(BTreeSet<String>, BTreeSet<String>), String> {
    const OPEN: &str = "pub mod seam {";
    let code = strip(source);
    let start = code.find(OPEN).ok_or("lib.rs declares no `pub mod seam`")?;
    let body_start = start + OPEN.len();
    let mut depth = 1;
    let body_len = code[body_start..]
        .find(|c: char| {
            depth += match c {
                '{' => 1,
                '}' => -1,
                _ => 0,
            };
            depth == 0
        })
        .ok_or("unterminated `pub mod seam`")?;
    let root = format!(
        "{}pub mod seam;{}",
        &code[..start],
        &code[body_start + body_len + 1..]
    );
    Ok((
        exported_names(&root)?,
        exported_names(&code[body_start..body_start + body_len])?,
    ))
}

fn names(list: &str) -> BTreeSet<String> {
    list.split_whitespace().map(str::to_string).collect()
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
        let root = format!("pub use a::B;\n{item}\npub mod seam {{ pub use c::D; }}\n");
        assert!(exports(&root).is_err(), "accepted `{item}` at the root");
        let seam = format!("pub use a::B;\npub mod seam {{ pub use c::D;\n{item}\n}}\n");
        assert!(exports(&seam).is_err(), "accepted `{item}` in seam");
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
        pub mod listed;
        pub use a::{B, c::D as E}; // pub struct Trailing;
        pub mod seam {
            pub use f::G;
            #[cfg(feature = "x")]
            pub use h::{I, J};
        }
        #[cfg(test)]
        use k::L;
    "#;
    let (root, seam) = exports(source).unwrap();
    assert_eq!(root, names("listed B E seam"));
    assert_eq!(seam, names("G I J"));
}

#[test]
fn crate_root_and_seam_export_exactly_the_consumed_items() {
    let (root, seam) = exports(include_str!("../src/lib.rs")).unwrap();
    assert_eq!(
        root,
        names(
            "BoundDynFactor FactorScalar HermitianTol SectorSpectrum SpectrumMagnitude \
             Eig Eigh LeftPolar Lq Qr RightPolar Svd \
             select_truncation Truncation TruncationDecision TruncationError TruncationSpace \
             WeightedSpectrum truncation seam"
        )
    );
    assert_eq!(
        seam,
        names(
            "admit_compact_diagonal compact_factor_routes coupled_sector_block_dimensions_generic_checked decide_bond_truncation \
             ascending_eigh_order diagonal_bond_data eig_full_from_source \
             eig_vals_from_source \
             eigh_full_dyn eigh_full_from_source eigh_sector_into \
             eigh_vals_from_source \
             in_linalg_scope left_null_from_source \
             left_polar_adjoint_from_parent left_polar_from_source \
             lq_compact_adjoint_from_parent lq_compact_from_source \
             lq_full_adjoint_from_parent lq_full_from_source pinv_diagonal_spectrum \
             qr_compact_from_source qr_full_from_source rectangular_diagonal_bond_tensor \
             require_finite_eigh_input nonfinite_factor_input FactorFamily \
             right_null_from_source \
             right_polar_adjoint_from_parent right_polar_from_source \
             scale_axis_by_spectrum_mapped axpby_axis_by_spectrum_into spectrum_bond \
             svd_compact_adjoint_from_parent svd_compact_dyn_checked_generic \
             svd_compact_from_source svd_full_adjoint_from_parent svd_full_from_source \
             svd_vals_dyn svd_vals_from_source validate_endomorphism_region_stacking validate_hermitian_regions \
             BoundDynamicTensorRef CheckedGenericFactorPlanError CompactFactorPlan CompactFactorRoute EigFullDyn EighFullDyn EighScratch \
             EIGH_EIGENVALUE_CHECK EIGH_NOT_HERMITIAN EIGH_NOT_SQUARE EndomorphismFamily ExecutorLease FactorMode FactorOutput FactorSpaceAuthority FactorSource \
             SvdFactorsDyn \
             exp_direct_into_dyn inv_direct_into_dyn \
             pinv_adjoint_parent_direct_into_dyn pinv_direct_into_dyn \
             solve_left_direct_into_dyn rescaled_power_norm"
        )
    );
}
