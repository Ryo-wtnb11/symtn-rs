//! Export firewall for `symtn-tensors`: nothing is exported behind the
//! `testing` feature. The exact public names are recorded by the API
//! snapshot (`api/`).
//!
//! Why a source scan and not an import list: an import list only proves
//! presence, not absence.

use std::collections::{BTreeMap, BTreeSet};

use crate::export_scan::reexports;

/// Leaf names re-exported by `pub use` in `source`, each with whether it is
/// gated behind the `testing` feature.
fn exported_names(source: &str) -> Result<BTreeMap<String, bool>, String> {
    Ok(reexports(source)?.without_modules()?.names)
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
        #[cfg(any(test, feature = "testing"))]
        pub use h::Gated;
    "#;
    let names = exported_names(source).unwrap();
    assert_eq!(
        names,
        [("B", false), ("E", false), ("G", false), ("Gated", true)]
            .map(|(name, gated)| (name.to_owned(), gated))
            .into()
    );
}

/// The production (ungated) or `testing`-gated names of `lib.rs`.
fn exported(testing: bool) -> BTreeSet<String> {
    exported_names(include_str!("../src/lib.rs"))
        .unwrap()
        .into_iter()
        .filter(|&(_, gated)| gated == testing)
        .map(|(name, _)| name)
        .collect()
}

/// Nothing is exported behind `testing`: production Generic is checked only,
/// and the infallible-Generic oracles left with the static-rank
/// facade.
#[test]
fn no_entries_are_exported_only_behind_testing() {
    assert_eq!(exported(true), BTreeSet::new());
}
