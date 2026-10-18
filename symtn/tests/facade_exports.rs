//! Export firewall for the `symtn` facade (#1565).
//!
//! Each typed operation has exactly one public path, the method on
//! `symtn::typed::TensorMap`; lower-layer free functions and whole-crate
//! re-exports are not reachable from `symtn`. This test pins the exact names
//! each public module exports, under every feature combination at once, so
//! adding or removing one fails here and is reviewed as an API change.
//!
//! Why a source scan and not a compile-time import list: an import list only
//! proves presence, not absence, and the workspace has no parser dependency.

use std::collections::BTreeSet;
use std::path::Path;

/// Removes comments and string and char literals, so that brace depth and
/// item keywords can be read from what remains.
fn strip(source: &str) -> String {
    let bytes = source.as_bytes();
    let mut out = String::with_capacity(source.len());
    let mut i = 0;
    while i < bytes.len() {
        let rest = &source[i..];
        if rest.starts_with("//") {
            i += rest.find('\n').unwrap_or(rest.len());
        } else if rest.starts_with("/*") {
            i += rest.find("*/").expect("unterminated block comment") + 2;
        } else if let Some((open, hashes)) = raw_string_open(rest).filter(|_| !ends_in_word(&out)) {
            let close = format!("\"{}", "#".repeat(hashes));
            i += open + rest[open..].find(&close).expect("unterminated raw string") + close.len();
            out.push_str("\"\"");
        } else if bytes[i] == b'"' {
            i += 1;
            while bytes[i] != b'"' {
                i += if bytes[i] == b'\\' { 2 } else { 1 };
            }
            i += 1;
            out.push_str("\"\"");
        } else if let Some(len) = char_literal_len(rest) {
            i += len;
            out.push_str("' '");
        } else {
            let ch = rest.chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

fn ends_in_word(text: &str) -> bool {
    text.ends_with(|c: char| c.is_alphanumeric() || c == '_')
}

/// Byte length and hash count of a raw-string opener (`r"`, `r#"`,
/// `br##"`, ...) at the start of `rest`.
fn raw_string_open(rest: &str) -> Option<(usize, usize)> {
    let tail = rest.strip_prefix("br").or_else(|| rest.strip_prefix('r'))?;
    let hashes = tail.len() - tail.trim_start_matches('#').len();
    tail[hashes..]
        .starts_with('"')
        .then_some((rest.len() - tail.len() + hashes + 1, hashes))
}

/// Byte length of a char literal at the start of `rest`; `None` for a
/// lifetime or anything else.
fn char_literal_len(rest: &str) -> Option<usize> {
    let body = rest.strip_prefix('\'')?;
    if body.starts_with('\\') {
        return body[2..].find('\'').map(|end| end + 4);
    }
    let first = body.chars().next()?;
    body[first.len_utf8()..]
        .starts_with('\'')
        .then_some(first.len_utf8() + 2)
}

/// Leaf names of a `use` tree such as `a::{b, c::{d, e as f}}`.
fn use_leaves(tree: &str, out: &mut BTreeSet<String>) {
    let tree = tree.trim();
    if let Some(inner) = tree.strip_suffix('}') {
        let open = inner.find('{').expect("use group");
        let mut depth = 0;
        let mut start = open + 1;
        for (index, ch) in inner.char_indices().skip(open + 1) {
            match ch {
                '{' => depth += 1,
                '}' => depth -= 1,
                ',' if depth == 0 => {
                    use_leaves(&inner[start..index], out);
                    start = index + 1;
                }
                _ => {}
            }
        }
        use_leaves(&inner[start..], out);
    } else if !tree.is_empty() {
        let leaf = tree.rsplit(" as ").next().unwrap();
        let leaf = leaf.rsplit("::").next().unwrap().trim();
        assert_ne!(leaf, "*", "glob re-export in the facade: {tree}");
        out.insert(leaf.to_owned());
    }
}

/// Public item names declared at depth zero of `body`, and the bodies of its
/// inline `pub mod` blocks.
fn exports(body: &str) -> (BTreeSet<String>, Vec<(String, String)>) {
    let mut names = BTreeSet::new();
    let mut inline_modules = Vec::new();
    let mut depth = 0usize;
    let mut i = 0;
    while i < body.len() {
        let rest = &body[i..];
        let ch = rest.chars().next().unwrap();
        match ch {
            '{' | '(' | '[' => depth += 1,
            '}' | ')' | ']' => depth -= 1,
            _ => {}
        }
        let at_word = !ends_in_word(&body[..i]);
        if depth == 0 && at_word && rest.starts_with("pub ") {
            let decl = rest["pub ".len()..].trim_start();
            let (mut keyword, mut tail) = decl.split_once(char::is_whitespace).unwrap();
            if keyword == "use" {
                use_leaves(&tail[..tail.find(';').unwrap()], &mut names);
                i += rest.find(';').unwrap() + 1;
                continue;
            }
            if keyword == "unsafe" || (keyword == "const" && tail.starts_with("fn ")) {
                (keyword, tail) = tail.trim_start().split_once(char::is_whitespace).unwrap();
            }
            assert!(
                matches!(
                    keyword,
                    "fn" | "struct" | "enum" | "trait" | "type" | "const" | "static" | "mod"
                ),
                "unrecognized public item `pub {keyword}` in the facade"
            );
            let tail = tail.trim_start();
            let name: String = tail
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            let after = tail[name.len()..].trim_start();
            if keyword == "mod" && after.starts_with('{') {
                let open = rest.len() - after.len();
                let close = open + matching_brace(&rest[open..]);
                inline_modules.push((name.clone(), rest[open + 1..close].to_owned()));
                names.insert(name);
                i += close + 1;
                continue;
            }
            names.insert(name);
        }
        i += ch.len_utf8();
    }
    (names, inline_modules)
}

/// Byte index of the brace closing the one that opens `text`.
fn matching_brace(text: &str) -> usize {
    let mut depth = 0;
    for (index, ch) in text.char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return index;
                }
            }
            _ => {}
        }
    }
    panic!("unbalanced braces")
}

fn module_exports(file: &str) -> (BTreeSet<String>, Vec<(String, String)>) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src").join(file);
    exports(&strip(&std::fs::read_to_string(path).unwrap()))
}

fn assert_exports(module: &str, actual: &BTreeSet<String>, expected: &[&str]) {
    let expected: BTreeSet<String> = expected.iter().map(|name| (*name).to_owned()).collect();
    let added: Vec<_> = actual.difference(&expected).collect();
    let removed: Vec<_> = expected.difference(actual).collect();
    assert!(
        added.is_empty() && removed.is_empty(),
        "`{module}` exports changed: added {added:?}, removed {removed:?}"
    );
}

#[test]
fn facade_exports_are_exactly_the_reviewed_names() {
    let (root, inline) = module_exports("lib.rs");
    assert_exports("symtn", &root, ROOT);
    let inline: Vec<&str> = inline.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(inline, ["mathematics"], "inline modules of `symtn`");
    assert_exports("symtn::typed", &module_exports("typed.rs").0, TYPED);
    assert_exports("symtn::sector", &module_exports("sector.rs").0, SECTOR);
    assert_exports("symtn::expert", &module_exports("expert.rs").0, EXPERT);
    assert_exports("symtn::cache", &module_exports("cache.rs").0, CACHE);
    assert_exports(
        "symtn::typed::__network",
        &module_exports("typed/__network.rs").0,
        NETWORK,
    );
}

#[test]
fn network_seam_pin_rejects_an_added_or_a_missing_item() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/typed/__network.rs");
    let source = std::fs::read_to_string(path).unwrap();
    let pinned = |source: &str| {
        let names = exports(&strip(source)).0;
        std::panic::catch_unwind(|| assert_exports("symtn::typed::__network", &names, NETWORK))
            .is_ok()
    };
    assert!(pinned(&source), "control: the real seam matches its pin");
    let added = format!("{source}\npub fn smuggled() {{}}\n");
    assert!(!pinned(&added), "an added seam item went unnoticed");
    let removed = source.replacen("pub fn runtime_identity", "fn runtime_identity", 1);
    assert_ne!(removed, source, "control: the edit applies");
    assert!(!pinned(&removed), "a removed seam item went unnoticed");
}

#[test]
fn each_name_has_one_path() {
    let modules = [
        ("symtn", ROOT),
        ("typed", TYPED),
        ("sector", SECTOR),
        ("expert", EXPERT),
        ("cache", CACHE),
        ("typed::__network", NETWORK),
    ];
    for (index, (left, left_names)) in modules.iter().enumerate() {
        for (right, right_names) in &modules[index + 1..] {
            let shared: Vec<_> = left_names
                .iter()
                .filter(|name| right_names.contains(name))
                .collect();
            assert!(
                shared.is_empty(),
                "names exported by both `{left}` and `{right}`: {shared:?}"
            );
        }
    }
}

#[test]
fn scanner_reads_nested_groups_renames_and_inline_modules() {
    let source = strip(
        "pub use a::{b, c::{d, e as f}};\n\
         // pub fn commented() {}\n\
         pub(crate) fn private() { let _ = \"pub fn in_string\"; let _ = '{'; }\n\
         pub const fn g<'a>() {}\n\
         pub mod m { pub struct Inner; }\n\
         impl X { pub fn method(&self) {} }\n",
    );
    let (names, inline) = exports(&source);
    let expected: BTreeSet<String> = ["b", "d", "f", "g", "m"].map(String::from).into();
    assert_eq!(names, expected);
    assert_eq!(
        exports(&inline[0].1).0,
        BTreeSet::from(["Inner".to_owned()])
    );
}

const ROOT: &[&str] = &[
    "cache",
    "expert",
    "krylov",
    "mathematics",
    "sector",
    "typed",
];
const TYPED: &[&str] = &[
    "AdvancedLinalgScalar",
    "Alternative",
    "BatchError",
    "BatchMemberRepresentation",
    "BlockFusionTrees",
    "BlockKey",
    "BlockKeyKind",
    "CheckedFusionSpaceError",
    "CheckedGenericPlanError",
    "CheckedGenericStructureError",
    "CheckedGenericTensorProductError",
    "Complex32",
    "Complex64",
    "ContractSpec",
    "CoreError",
    "CoupledBlock",
    "CoupledBlockPayload",
    "CudaFactorizationPayload",
    "CudaPayload",
    "CudaRealScalar",
    "CudaScalar",
    "CudaStorage",
    "CudaTreeTransformStats",
    "DecodeError",
    "DecodeLimits",
    "Direction",
    "Duality",
    "Eig",
    "Eigh",
    "ElementwiseExec",
    "EighFullPlan",
    "EighFullWorkspace",
    "EighStackOutput",
    "EncodeError",
    "Error",
    "FacadeError",
    "FactorExec",
    "FactorScalar",
    "FactorizationScalar",
    "FusionAlgebraError",
    "FusionMode",
    "FusionTreeGroupKey",
    "FusionTreeKey",
    "FusionTreeLabels",
    "FusionTreePairKey",
    "GenericTensorError",
    "GradedSpace",
    "LeftPolar",
    "LegSelection",
    "LinalgBackend",
    "Lq",
    "MaterializeExec",
    "MemberFault",
    "ComposePlan",
    "ContractExec",
    "ContractPlan",
    "ContractReplay",
    "ContractWorkspace",
    "TraceExec",
    "MultiplicityIndex",
    "NON_SYMMETRIC_CONTRACTION_UNSUPPORTED",
    "OpaqueBlockKey",
    "OperationError",
    "PayloadConversion",
    "PersistedScalar",
    "PhysicalDense",
    "PhysicalDenseError",
    "PreparedExec",
    "Qr",
    "RecouplingCoefficientAction",
    "ReduceExec",
    "HermitianTol",
    "RightPolar",
    "Runtime",
    "RuntimeBuilder",
    "RuntimeConfigError",
    "SectorSpectrum",
    "Side",
    "SignatureField",
    "SpectrumMagnitude",
    "StackExec",
    "StackedTensorMap",
    "StructureSignature",
    "Svd",
    "TensorMap",
    "TensorRef",
    "TensorScalar",
    "TransformExec",
    "TransformExecError",
    "TransformIntoSource",
    "TreeReplay",
    "TruncatedSelection",
    "Truncation",
    "TruncationError",
    "TruncationSpace",
    "TwistExec",
    "TypedPersistenceCodec",
    "TypedAdjointSpace",
    "BondOutput",
    "BondTransform",
    "TypedSpaceModeDispatch",
    "TypedStorage",
    "TypedTensorConstructionDispatch",
    "TypedTensorContractDispatch",
    "TypedTensorFlipDispatch",
    "TypedTensorModeDispatch",
    "TypedTensorProductDispatch",
    "TypedTensorRootDispatch",
    "TypedTensorTraceDispatch",
    "TypedTensorTransformDispatch",
    "TypedTensorTwistDispatch",
    "TypedTensorUnitDispatch",
    "TypedTruncationDispatch",
    "__network",
    "reject_non_symmetric_contraction",
];
/// `symtn::typed::__network`: the hidden seam `symtn-network` drives. It is
/// pinned like a public module so a new seam item is reviewed, not slipped in.
const NETWORK: &[&str] = &[
    "ExtensionSlot",
    "NetworkDegeneracyRestriction",
    "NetworkPayloadStorage",
    "NetworkReuseClass",
    "RuntimeDetachedTensorMap",
    "RuntimeIdentity",
    "cuda_device_ordinal",
    "detach_runtime",
    "network_has_compact_payload",
    "network_input_metadata_matches",
    "network_owned_payload",
    "network_restrict_degeneracies",
    "network_reuse_class",
    "network_scatter_add_assign",
    "network_sector_leg",
    "network_source_leg",
    "network_zeros_from_effective_legs",
    "runtime_identity",
    "with_extension_slot",
];
const SECTOR: &[&str] = &[
    "BraidingStyleKind",
    "CU1FusionRule",
    "CU1Irrep",
    "CanonicalUnitFusionRule",
    "CategoricalScalar",
    "CheckedCanonicalUnitFusionRule",
    "CheckedFusionAlgebra",
    "CheckedGenericAdmissionMode",
    "CheckedGenericFusion",
    "CheckedGenericPivotal",
    "CheckedGenericRigidSymbols",
    "CoupledSectorFold",
    "FermionParityFusionRule",
    "FusionRule",
    "FusionStyleKind",
    "Fz2SectorLayout",
    "GenericFArray",
    "GenericRMatrix",
    "MultiplicityFreeAdmissionMode",
    "MultiplicityFreeFusionRule",
    "MultiplicityFreeFusionSymbols",
    "MultiplicityFreeRigidSymbols",
    "PackedProductCodec",
    "PackedSectorLayout",
    "PhysicalBasisError",
    "PhysicalFusionBasis",
    "ProductFusionRule",
    "ProductFusionRuleExt",
    "ProductSector",
    "ProductSectorCodec",
    "ProductSectorCodecError",
    "ProductSectorComponent",
    "ProductSectorLayout",
    "PromoteCoefficientScalar",
    "RuleIdentity",
    "SU2CoefficientError",
    "SU2FusionRule",
    "SU2Irrep",
    "SUNFusionRule",
    "SUNFusionRuleError",
    "SUNSymbolError",
    "SectorCodec",
    "SectorId",
    "SectorOrderKey",
    "SectorVec",
    "Su2SectorLayout",
    "SymbolShapeError",
    "TableRule",
    "TableRuleError",
    "TableSector",
    "TensorKitProductCodec",
    "TypedSectorAdmission",
    "U1FusionRule",
    "U1Irrep",
    "U1SectorLayout",
    "Z2FusionRule",
    "Z2Irrep",
    "ZNFusionRule",
    "ZNIrrep",
    "product_fusion_rule",
    "product_fusion_rule_with_codec",
    "product_sector",
];
const EXPERT: &[&str] = &[
    "BlockLayout",
    "BlockRef",
    "BlockView",
    "CpuBackendKind",
    "CpuSessionStats",
    "CudaDenseContext",
    "CudaDenseStorage",
    "CudaPlanCacheStats",
    "CudaTransferStats",
    "DefaultDenseExecutor",
    "DenseBackend",
    "DenseDType",
    "DenseDotConfig",
    "DenseError",
    "DenseExecutor",
    "DenseFactorization",
    "DenseGemmBatchJob",
    "DenseLinalgScopeBody",
    "DenseOwned",
    "DenseRead",
    "DenseScalar",
    "DenseTensor",
    "DenseView",
    "DenseViewMut",
    "DenseWrite",
    "HostReadableStorage",
    "MatrixOp",
    "Placement",
    "SectorLeg",
    "SectorLegConstructionError",
    "SharedCpuContext",
    "TensorStorage",
    "TreeTransformOperation",
    "TreeTransformOperationKind",
    "cpu_session_stats",
    "cuda_transfer_stats",
    "diagonal_spectrum",
    "is_diagonal",
    "reset_cpu_session_stats",
    "reset_cuda_transfer_stats",
];
/// The one public control of the process-global structure caches.
const CACHE: &[&str] = &[
    "StructureCacheInfo",
    "StructureCacheKind",
    "clear",
    "configure_budgets",
    "stats",
];
