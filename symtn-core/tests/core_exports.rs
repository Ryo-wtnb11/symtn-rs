//! Export firewall for `symtn-core` (#1805).
//!
//! Pins the exact names reachable at the crate root, under every feature
//! combination at once, so adding or removing one fails here and is reviewed
//! as an API change. Test-only tree operations (`generic_braid_tree`,
//! `multiplicity_free_*_block`, `unique_permute_tree`, ...) are crate-private;
//! the few that tests outside the crate use are forwarded only through the
//! `testing` feature's `symtn_core::testing`, pinned separately below.
//! `merge_fusion_trees_generic` was deleted, and so were the static-rank
//! `TensorMap` / `Tensor` and their `Trivial` marker (#2222) and the
//! static-rank `TensorMapSpace` / `ProductSpace` / `FusionTensorMapSpace`
//! (#2253).
//!
//! Why a source scan and not an import list: an import list only proves
//! presence, not absence. The crate root re-exports its private modules with
//! `pub use module::*`, so the scan follows each glob into the module's file
//! and collects that file's top-level (column 0, rustfmt-formatted) `pub`
//! items; `pub(crate)` and indented items (impl members, nested modules) are
//! not exports.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

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

/// The `testing` gate after [`strip`] blanks its string literal. No other
/// feature uses the `any(test, feature = ..)` shape
/// (`only_testing_uses_the_test_or_feature_gate`).
const TESTING_GATE: &str = "#[cfg(any(test, feature = \"\"))]";

/// Whether the item starting at `at` carries [`TESTING_GATE`] among the
/// attribute lines directly above it.
fn testing_gated(code: &str, at: usize) -> bool {
    code[..at]
        .lines()
        .rev()
        .map(str::trim)
        .take_while(|line| line.starts_with("#["))
        .any(|line| line == TESTING_GATE)
}

/// Names a module file exports at its top level, each with whether it is
/// gated behind the `testing` feature (#1854). `pub use child::*` of a
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
        let gated = testing_gated(&code, at);
        let after = &code[at + 3..];
        if after.starts_with('(') || after.starts_with(|c: char| c.is_alphanumeric() || c == '_') {
            continue; // `pub(crate)` etc.
        }
        let item = after.trim_start();
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
        for leaf in leaves.iter().map(|l| l.trim()).filter(|l| !l.is_empty()) {
            let name = leaf.rsplit(" as ").next().unwrap();
            names.insert(name.rsplit("::").next().unwrap().trim().to_owned(), gated);
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

#[test]
fn symtn_core_exports_exactly_the_pinned_names() {
    let expected = [
        "BlockKey",
        "BlockKeyKind",
        "BlockLayout",
        "BlockRef",
        "BlockSourceColumns",
        "BlockSpec",
        "BlockStructure",
        "BlockStructureContent",
        "BlockView",
        "BlockViewMut",
        "BraidingStyleKind",
        "CU1FusionRule",
        "CU1Irrep",
        "CU1_MAX_TWICE_CHARGE",
        "CanonicalUnitAdmission",
        "CanonicalUnitFusionRule",
        "CategoricalScalar",
        "CategoryAdapterError",
        "CategoryAdmissionError",
        "CategoryCodec",
        "CategoryFailure",
        "CategoryOperation",
        "CategoryProvider",
        "CategoryRule",
        "CheckedCanonicalUnitFusionRule",
        "CheckedFusionAlgebra",
        "CheckedFusionSpaceError",
        "CheckedGenericAdmissionMode",
        "CheckedGenericAdmittedFusionTreeBlockStructure",
        "CheckedGenericFusion",
        "CheckedGenericPivotal",
        "CheckedGenericRigidSymbols",
        "CheckedGenericStructureError",
        "CheckedGenericSymbolError",
        "StructureCacheInfo",
        "StructureCache",
        "EnumeratedCodec",
        "ErasedStructureCacheControl",
        "StructureCacheEquivalent",
        "CoreError",
        "CoupledSectorFold",
        "CoupledSectorFoldBuilder",
        "CoupledMatricizationBuilder",
        "CoupledMatrixSide",
        "CoupledTreePlacement",
        "CoupledMatrixPlacement",
        "CoupledSectorRegion",
        "CoupledTreeExtent",
        "DedupTreeArenaBuilder",
        "DegeneracyBlock",
        "DegeneracyBlockRef",
        "DegeneracyStructure",
        "DimVec",
        "DualVec",
        "FermionParityFusionRule",
        "FusionAlgebraError",
        "FusionProductSpace",
        "FusionRule",
        "FusionSpaceAdmission",
        "FusionStyleKind",
        "FusionTreeBlockGroup",
        "FusionTreeGroupKey",
        "FusionTreeGroupRef",
        "FusionTreeHomSpace",
        "FusionTreeKey",
        "FusionTreePairKey",
        "FusionTreePairOrientation",
        "FusionTreePairRef",
        "FusionTreeRef",
        "Fz2SectorLayout",
        "GenericFArray",
        "GenericRMatrix",
        "HomSpaceId",
        "HostReadableStorage",
        "HostStorage",
        "HostWritableStorage",
        "InfallibleGeneric",
        "LocallyValidatedFusionTreeBlockStructure",
        "MultiplicityFreeAdmissionMode",
        "MultiplicityFreeFusionRule",
        "MultiplicityFreeFusionSymbols",
        "MultiplicityFreeRigidSymbols",
        "MultiplicityIndex",
        "MultiplicityVec",
        "OpaqueBlockKey",
        "OrderedBlockLinearMap",
        "OrderedBlockLinearStorage",
        "OrientedFusionTreeHomSpace",
        "PackedProductCodec",
        "PackedSectorLayout",
        "Permuted",
        "PermutedMemoTicket",
        "PhysicalBasisError",
        "PhysicalFusionBasis",
        "Placement",
        "PreparedBlockStructure",
        "PreparedFusionTreeLayout",
        "PreparedTreePairOperation",
        "ProductAdmission",
        "ProductFusionRule",
        "ProductFusionRuleError",
        "ProductSector",
        "ProductSectorCodec",
        "ProductSectorCodecError",
        "ProductSectorComponent",
        "ProductSectorLayout",
        "RuleIdentity",
        "SU2CoefficientError",
        "SU2FusionRule",
        "SU2Irrep",
        "SU2_MAX_DOUBLED_SPIN",
        "SUNFusionRule",
        "SUNFusionRuleError",
        "SUNSymbolError",
        "SectorBlock",
        "SectorCodec",
        "SectorId",
        "SectorLeg",
        "SectorLegConstructionError",
        "SectorOrderKey",
        "SectorStructure",
        "SectorVec",
        "SimilarStorage",
        "StructurallyValidatedFusionTreeSubset",
        "Su2SectorLayout",
        "SymbolShapeError",
        "TableRule",
        "TableRuleError",
        "TableSector",
        "TensorKitProductCodec",
        "TensorStorage",
        "TreeRehomer",
        "TreeScratch",
        "TypedSectorAdmission",
        "U1FusionRule",
        "U1Irrep",
        "U1SectorLayout",
        "UncertifiedUnitAdmission",
        "UniqueTreePairScratch",
        "UnitLegInsertion",
        "WeakHomSpaceId",
        "Z2FusionRule",
        "Z2Irrep",
        "ZNFusionRule",
        "ZNIrrep",
        "axes",
        "checked_product",
        "column_major_strides",
        "structure_cache_info",
        "StructureCacheKind",
        "set_structure_cache_byte_budget",
        "structure_cache_infos",
        "register_structure_cache",
        "StructureCacheRegistrationError",
        "core_reset_epoch",
        "generic_braid_tree_pair_checked",
        "generic_permute_tree_pair_block_indexed_checked",
        "generic_permute_tree_pair_checked",
        "generic_transpose_tree_pair_checked",
        "merge_fusion_trees_generic_checked",
        "merge_fusion_trees_multiplicity_free",
        "multiplicity_free_braid_tree_pair",
        "multiplicity_free_braid_tree_pair_block_ordered_indexed",
        "multiplicity_free_permute_tree_pair",
        "multiplicity_free_permute_tree_pair_block_indexed",
        "multiplicity_free_repartition_tree_pair",
        "multiplicity_free_transpose_tree_pair_block_ordered_indexed",
        "product_sector",
        "clear_structure_caches",
        "split_fusion_tree",
        "split_fusion_tree_generic_checked",
        "split_fusion_tree_generic_checked_into",
        "split_fusion_tree_into",
        "testing",
        "try_for_each_column_major_stride",
        "validate_block_storage_injective",
        "validate_generic_fusion_tree_pair_checked",
        "validate_unit_layout_correspondence",
        "validate_unit_layout_correspondence_checked",
        "validate_unit_layout_correspondence_generic_checked",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<BTreeSet<_>>();
    assert_eq!(exported(src(), "lib.rs", false), expected);
}

/// The infallible Generic symbol tier is test-only: production Generic is
/// checked only (#1854).
#[test]
fn infallible_generic_tier_is_exported_only_behind_testing() {
    let expected = [
        // Measurement counters of the Generic block composer (#1962).
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

#[test]
fn testing_feature_forwards_exactly_the_pinned_oracles() {
    let expected = [
        // The provider seal native test fixtures implement.
        "Provider",
        "generic_braid_tree_pair",
        "generic_permute_tree_pair",
        "mark_structure_canonical",
        "multiplicity_free_braid_tree_block",
        "multiplicity_free_permute_tree_pair_block",
        "multiplicity_free_transpose_tree_pair",
        "unique_permute_tree",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<BTreeSet<_>>();
    assert_eq!(exported(src(), "testing.rs", false), expected);
}
