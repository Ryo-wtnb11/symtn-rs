//! Export firewall for `symtn-operations` (#1804).
//!
//! Two pins, both over every feature combination at once:
//! - the exact public modules and `pub use` leaves of `lib.rs`, so adding or
//!   removing a crate-root export fails here and is reviewed as an API change;
//! - the raw kernels, fusion-tree helpers, unrouted replay entries and compile
//!   constructors narrowed in #1804, which live inside public or glob-exported
//!   modules and so are invisible to the `lib.rs` pin. No workspace crate
//!   calls them, and keeping them public froze signatures that the
//!   one-pipeline refactors (#1851) must change.
//!
//! Why a source scan and not an import list: an import list only proves
//! presence, not absence.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

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

/// Public names of `lib.rs`: `mod <name>` for each `pub mod <name>;`, the
/// leaf of each `pub use` (`<path>::*` for a glob), or an error for any other
/// inline public item (`pub fn`, `pub mod m { .. }`, ...) or
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
        if let Some(module) = item
            .strip_prefix("mod")
            .filter(|t| t.starts_with(char::is_whitespace))
        {
            let end = module.find([';', '{']).ok_or("unterminated pub mod")?;
            if module[end..].starts_with('{') {
                return Err("lib.rs gained an inline `pub mod` body".into());
            }
            names.insert(format!("mod {}", module[..end].trim()));
            continue;
        }
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
            if leaf.ends_with('*') {
                names.insert(leaf.split_whitespace().collect());
                continue;
            }
            let name = leaf.rsplit(" as ").next().unwrap();
            names.insert(name.rsplit("::").next().unwrap().trim().to_owned());
        }
    }
    Ok(names)
}

/// Whether stripped `code` defines `name` as an unrestricted `pub fn`.
fn defines_public_fn(code: &str, name: &str) -> bool {
    ['(', '<']
        .iter()
        .any(|open| code.contains(&format!("pub fn {name}{open}")))
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
        pub mod visible;
        pub use a::{B, c::D as E}; // pub struct Trailing;
        pub use f::G;
        pub use h::*;
    "#;
    let names = exported_names(source).unwrap();
    assert_eq!(
        names,
        ["B", "E", "G", "h::*", "mod visible"]
            .map(str::to_owned)
            .into()
    );
}

/// Bodies of the inherent `impl ... <owner><...> { .. }` blocks in stripped
/// `code` (trait impls, `impl Trait for Owner`, are skipped).
fn inherent_impl_bodies<'a>(code: &'a str, owner: &str) -> Vec<&'a str> {
    let mut bodies = Vec::new();
    for (at, _) in code.match_indices("impl") {
        let Some(open) = code[at..].find('{').map(|open| at + open) else {
            break;
        };
        let header = &code[at..open];
        if !header.contains(&format!(" {owner}<")) || header.contains(" for ") {
            continue;
        }
        let mut depth = 0usize;
        for (offset, ch) in code[open..].char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        bodies.push(&code[open..open + offset]);
                        break;
                    }
                }
                _ => {}
            }
        }
    }
    bodies
}

#[test]
fn inherent_impl_scan_matches_only_the_owner() {
    let code = strip(
        "impl<T> Owner<T> { pub fn a() {} }\nimpl<T> Other<T> { pub fn b() {} }\nimpl<T> Tr for Owner<T> { fn c() {} }",
    );
    let bodies = inherent_impl_bodies(&code, "Owner");
    assert_eq!(bodies.len(), 1);
    assert!(defines_public_fn(bodies[0], "a"));
    assert!(!defines_public_fn(bodies[0], "b"));
}

#[test]
fn tree_transform_structure_has_no_panicking_coefficient_accessor() {
    // `coefficient(index)` panicked out of range; `single_coefficient`
    // returns `Option<T>`.
    let mut files = Vec::new();
    rust_sources(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
        &mut files,
    );
    let mut owner_blocks = 0;
    for file in &files {
        let code = strip(&fs::read_to_string(file).unwrap());
        for body in inherent_impl_bodies(&code, "TreeTransformStructure") {
            owner_blocks += 1;
            assert!(
                !defines_public_fn(body, "coefficient"),
                "{}: TreeTransformStructure::coefficient is public again",
                file.display()
            );
        }
    }
    assert!(owner_blocks > 0, "no TreeTransformStructure impl found");
}

#[test]
fn internal_fn_scan_sees_only_unrestricted_definitions() {
    let code = strip(
        "// pub fn hidden()\npub(crate) fn narrowed() {}\npub fn open<T>() {}\nlet _ = \"pub fn s()\";",
    );
    assert!(!defines_public_fn(&code, "hidden"));
    assert!(!defines_public_fn(&code, "narrowed"));
    assert!(!defines_public_fn(&code, "s"));
    assert!(defines_public_fn(&code, "open"));
}

#[test]
fn symtn_operations_root_exports_exactly_the_pinned_names() {
    let expected = [
        // public modules
        "mod axis",
        "mod cuda",
        "mod cuda_transform",
        "mod fusion_replay",
        "mod host_pool",
        "mod host_scratch",
        "mod kernel_adapter",
        "mod replay_backend",
        "mod stacked",
        "mod storage_scratch",
        "mod strided",
        "mod structure_identity",
        "mod transform_plan",
        "mod transform_replay",
        "mod transform_structure",
        // glob re-exports
        "axis::*",
        "replay_backend::*",
        "transform_replay::*",
        "transform_structure::*",
        // checked block layout (hidden)
        "take_checked_block_passes",
        "CheckedBlockLayout",
        "CheckedBlockPasses",
        // cuda transform
        "CudaTreeTransformDestination",
        "CudaTreeTransformExecutor",
        "DEFAULT_COEFFICIENT_BUDGET_BYTES",
        // error
        "OperationError",
        // fusion replay
        "fusion_scale_block_layouts_excluding",
        "ContractDestinationInit",
        "FusionBlockContractGroupPlan",
        "FusionBlockContractPlan",
        "FusionBlockContractWorkspace",
        "FusionBlockMatrixGroup",
        "FusionScaleBlockLayout",
        "FusionStridedBlockLayout",
        "FusionSubblockMatrixLayout",
        "HostFusionBlockContractWorkspace",
        "Rank2Gemm",
        "StorageGemm",
        // routed host scalar kernels
        "axpby_raw_strided_kernel_trusted",
        "scale_raw_strided_kernel_trusted",
        "tensoradd_raw_strided_kernel",
        "bilinear_raw_strided_kernel_mapped",
        "tensortrace_raw_strided_kernel_add_with_coefficient_trusted",
        // kernel adapter
        "BakedFusedLayout",
        "HostKernelAdapter",
        "StridedHostKernelAdapter",
        // placement / profile
        "ReportsPlacement",
        // scalar
        "scale_value",
        "ConjugateValue",
        "DenseBlockScalar",
        "DenseRecouplingScalar",
        "RealStructuralCoefficient",
        "RecouplingCoefficientAction",
        "TransformScale",
        "TreeTransformScalar",
        "WideScalar",
        // transform key / plan / profile
        "TreeTransformOperation",
        "TreeTransformOperationKind",
        "TreeTransformBlockSpec",
        "TreeTransformGroupBlockSpec",
        "TreeTransformGroupPlan",
        "TreeTransformKeyBlockSpec",
        "TreeTransformReplayProfile",
        // owned buffers (hidden)
        "take_owned_block_prefills",
        "overwrite_owned_blocks",
        "overwrite_owned_member_blocks",
        "BlockOverwrite",
        "AxisRuns",
        "check_axis_runs",
        "for_each_run_piece",
        "zeroed_payload",
        "ZeroBytes",
        "try_cat_owned_c64_raw",
        "try_cat_owned_mixed_raw",
        "try_cat_owned_raw",
        "OwnedCatC64Source",
        "OwnedCatCopy",
        "OwnedCatSide",
        "try_tensortrace_owned_raw",
        "OwnedTraceTerm",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<BTreeSet<_>>();
    assert_eq!(
        exported_names(include_str!("../src/lib.rs")).unwrap(),
        expected
    );
}

const CRATE_INTERNAL: &[&str] = &[
    // Raw kernels; the `_trusted` drivers are the routed entries.
    "axpby_raw_strided_kernel",
    "tensoradd_raw_strided_kernel_trusted",
    "tensortrace_raw_strided_kernel_add_with_coefficient",
    // Fusion-tree helpers.
    "block_indices_for_keys",
    "duplicate_fusion_tree_pair_index",
    "duplicate_fusion_tree_pair_indices",
    "fusion_tree_group_block_keys",
    "fusion_tree_pair_matches_group",
    "fusion_tree_pairs_share_group",
    // Misc.
    "direct_group_matrix_offset",
    "validate_recoupling_lens",
    "with_host_pool",
    // Unrouted replay entries.
    "tree_transform_structure_overwrite_with_structural_recoupling",
    "tree_transform_structure_with_structural_recoupling_raw",
    "tree_transform_structure_with_structural_recoupling_raw_profiled",
    "tree_transform_structure_overwrite_with_structural_recoupling_raw_profiled",
    // Compile constructors without a production caller; the keyed entry is
    // `compile_keyed_shared_structures` (shares the structures by `Arc`) and
    // grouped specs compile through `TreeTransformGroupPlan`.
    "compile_keyed",
    "compile_keyed_structures",
    "compile_keyed_structures_with_storage_conjugation",
    "compile_grouped",
    "compile_grouped_structures",
    "compile_grouped_structures_with_storage_conjugation",
    "compile_grouped_shared_structures",
];

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn crate_internal_functions_stay_out_of_the_public_api() {
    let mut files = Vec::new();
    rust_sources(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
        &mut files,
    );
    let mut escaped = Vec::new();
    for file in &files {
        let code = strip(&fs::read_to_string(file).unwrap());
        for name in CRATE_INTERNAL {
            if defines_public_fn(&code, name) {
                escaped.push(format!("{}: {name}", file.display()));
            }
        }
    }
    assert!(escaped.is_empty(), "public again: {escaped:#?}");
}
