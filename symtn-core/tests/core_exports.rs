//! Export firewall for `symtn-core`: the `testing` gate.
//!
//! The exact public names are recorded by the API snapshot (`api/`); this
//! file checks what the snapshot does not state as a rule: which crate-root
//! names exist only behind the `testing` feature, and that no other feature
//! uses the `any(test, feature = ..)` gate shape the scan relies on.
//!
//! Why a source scan and not an import list: an import list only proves
//! presence, not absence. The crate root re-exports its private modules with
//! `pub use module::*`, so the scan follows each glob into the module's file
//! and collects that file's top-level (column 0, rustfmt-formatted) `pub`
//! items; `pub(crate)` and indented items (impl members, nested modules) are
//! not exports.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::export_scan::{not_unrestricted_pub, strip, testing_gated, use_leaves};

/// Names a module file exports at its top level, each with whether it is
/// gated behind the `testing` feature. `pub use child::*` of a
/// child module recurses into the child's file (`dir/child.rs` from `lib.rs`,
/// `dir/<module>/child.rs` otherwise); `pub mod m` exports `m` without
/// recursing (its items live under `m::`).
fn exported_names(dir: &Path, file: &str) -> Result<BTreeMap<String, bool>, String> {
    let path = dir.join(file);
    let source = std::fs::read_to_string(&path).map_err(|e| format!("{path:?}: {e}"))?;
    let code = strip(&source);
    if code.contains("macro_export") {
        return Err(format!("{file} exports a macro_rules! macro"));
    }
    let child_dir = match file {
        "lib.rs" | "mod.rs" => dir.to_path_buf(),
        _ => dir.join(file.trim_end_matches(".rs")),
    };
    let mut names = BTreeMap::new();
    for (at, _) in code.match_indices("pub") {
        if at != 0 && !code[..at].ends_with('\n') {
            continue; // not a top-level item
        }
        if not_unrestricted_pub(&code, at) {
            continue; // `pub(crate)` etc.
        }
        let gated = testing_gated(&code, at);
        let item = code[at + 3..].trim_start();
        let mut words = item.split(|c: char| !(c.is_alphanumeric() || c == '_'));
        let keyword = words.next().unwrap_or("");
        if keyword != "use" {
            let name = match keyword {
                "fn" | "struct" | "enum" | "trait" | "type" | "static" | "union" | "mod" => {
                    words.find(|w| !w.is_empty())
                }
                "const" | "unsafe" | "async" | "extern" => {
                    words.find(|w| !w.is_empty() && *w != "fn" && *w != "unsafe")
                }
                _ => None,
            }
            .ok_or_else(|| format!("{file}: unrecognised `pub {keyword}` item"))?;
            names.insert(name.to_owned(), gated);
            continue;
        }
        let tree = &item[3..];
        let tree = tree[..tree.find(';').ok_or("unterminated pub use")?].trim();
        if let Some(module) = tree.strip_suffix("::*") {
            for (name, inner) in exported_names(&child_dir, &format!("{module}.rs"))? {
                names.insert(name, gated || inner);
            }
            continue;
        }
        for name in use_leaves(tree)? {
            names.insert(name, gated);
        }
    }
    Ok(names)
}

/// The production (ungated) or `testing`-gated names of `file`.
fn exported(dir: &Path, file: &str, testing: bool) -> BTreeSet<String> {
    exported_names(dir, file)
        .unwrap()
        .into_iter()
        .filter(|&(_, gated)| gated == testing)
        .map(|(name, _)| name)
        .collect()
}

fn src() -> &'static Path {
    Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/src"))
}

#[test]
fn firewall_follows_globs_and_ignores_restricted_and_nested_items() {
    let dir = std::env::temp_dir().join(format!("symtn-core-exports-{}", std::process::id()));
    std::fs::create_dir_all(dir.join("inner")).unwrap();
    std::fs::write(
        dir.join("lib.rs"),
        "//! pub fn doc() {}\nmod inner;\npub use inner::*;\npub(crate) use hidden::*;\n\
         pub use a::{B, c::D as E};\npub const fn konst() {}\n\
         #[cfg(any(test, feature = \"testing\"))]\n#[doc(hidden)]\npub fn gated() {}\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("inner.rs"),
        "pub struct Shown;\npub(crate) fn hidden() {}\nimpl Shown {\n    pub fn method() {}\n}\n\
         mod nested {\n    pub fn deep() {}\n}\n/* pub enum Gone {} */\nmod leaf;\npub use leaf::*;\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("inner/leaf.rs"),
        "pub type Leaf = u8;\npub mod testing;\n",
    )
    .unwrap();
    let names = exported(&dir, "lib.rs", false);
    let gated = exported(&dir, "lib.rs", true);
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(gated, ["gated"].map(str::to_owned).into());
    assert_eq!(
        names,
        ["B", "E", "Leaf", "Shown", "konst", "testing"]
            .map(str::to_owned)
            .into()
    );
}

/// The infallible Generic symbol tier is test-only: production Generic is
/// checked only.
#[test]
fn infallible_generic_tier_is_exported_only_behind_testing() {
    let expected = [
        // Measurement counters of the Generic block composer.
        "GenericBlockStepCounts",
        "GenericFusionSymbols",
        "GenericRigidSymbols",
        "generic_braid_tree_pair_block_ordered",
        "generic_permute_tree_pair_block_ordered",
        "generic_transpose_tree_pair",
        "generic_transpose_tree_pair_block_ordered",
        "take_generic_block_step_counts",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<BTreeSet<_>>();
    assert_eq!(exported(src(), "lib.rs", true), expected);
}

#[test]
fn only_testing_uses_the_test_or_feature_gate() {
    fn visit(dir: &Path) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                visit(&path);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                let source = std::fs::read_to_string(&path).unwrap();
                for (at, _) in source.match_indices("any(test, feature = ") {
                    let rest = &source[at + "any(test, feature = ".len()..];
                    assert!(rest.starts_with("\"testing\")"), "{path:?}");
                }
            }
        }
    }
    visit(src());
}
