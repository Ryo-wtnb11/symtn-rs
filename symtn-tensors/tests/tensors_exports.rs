//! Export firewall for `symtn-tensors`.
//!
//! Pins the exact names `lib.rs` re-exports, under every feature combination
//! at once, so adding or removing one fails here and is reviewed as an API
//! change. Every name below is used by another workspace crate, by an
//! integration test or example of this crate, or by a retained public
//! signature. There is no static-rank layer: every entry takes a
//! dynamic space. The structure caches, the dynamic-rank `*_dyn_owned` /
//! `*_dyn_into` test entries, the re-exports of `symtn_operations` internals
//! and every name without a consumer outside this crate (other than an
//! integration test or example of this crate) are `pub(crate)`.
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

#[test]
fn symtn_tensors_exports_exactly_the_pinned_names() {
    let expected = [
        // adjoint
        "adjoint_bound_dyn",
        "adjoint_bound_space_dyn",
        "materialize_adjoint_data_dyn",
        "AdjointBlockMap",
        // backend_trace
        // bound_tensor
        "BoundDynamicTensorRef",
        // tree_transform: the process-global completed-transformer owner
        // (doc-hidden cross-crate seams)
        "TreeTransformOperationView",
        "admit_exact_tree_pair_layout",
        "exact_layout_tree_pair_hit",
        // contract
        "execute_storage_contract_members_cuda",
        "execute_storage_contract_resolution_on_cuda",
        "plan_compose",
        "prepare_tensorcontract_fusion_plan_dyn",
        "tensorcontract_fusion_block_specs",
        "try_compile_storage_contract_core_route",
        "CopyCRoute",
        "CoreRoute",
        "DirectCoreExecutor",
        "ExecCaps",
        "HostEagerExecutor",
        "BoundDynamicFusionMapSpace",
        "ContractRequest",
        "CudaContractScratch",
        "CudaContractMembersWorkspace",
        "DynamicFusionMapSpace",
        "HostContractMembersWorkspace",
        "FusionContractPlan",
        "FusionOperand",
        "HostTensorContractBackend",
        "StagedContract",
        "StorageContractResolution",
        "TensorContractBackend",
        "TensorContractBlockSpec",
        "TensorContractFusionExecutionContext",
        "ValidatedDynamicFusionLayout",
        // facade
        // oriented_elementwise
        "fusion_scatter_add_assign",
        "oriented_fusion_add_owned",
        "oriented_fusion_axpby_into",
        "oriented_fusion_inner",
        "oriented_fusion_inner_with",
        "oriented_fusion_restrict_owned",
        "stacked_fusion_restrict_owned",
        "SectorRunTable",
        "SelectedRuns",
        // physical
        "expand_physical_host",
        "project_physical_host",
        "PhysicalConversionError",
        "PivotalCoefficientAlgebra",
        "PlanarContractOrders",
        // symtn_core
        "RuleIdentity",
        // symtn_operations
        "cuda",
        "host_pool",
        "try_cat_owned_raw",
        "zeroed_payload",
        "ConjugateValue",
        "ContractDestinationInit",
        "DenseBlockScalar",
        "DenseRecouplingScalar",
        "DenseTreeTransformOperations",
        "OperationError",
        "OutputAxisOrder",
        "OwnedCatCopy",
        "OwnedCatSide",
        "RealStructuralCoefficient",
        "RecouplingCoefficientAction",
        "RigidCoefficientAlgebra",
        "ReportsPlacement",
        "TensorContractSpec",
        "TensorTraceAxisSpec",
        "TreeTransformBackend",
        "TreeTransformScalar",
        "TreeTransformStructure",
        "TreeTransformWorkspace",
        "WideScalar",
        "ZeroBytes",
        // tensortrace
        "tensortrace_fusion_dyn_preflight_checked",
        "tensortrace_fusion_dyn_preflight_generic_checked",
        "tensortrace_stage_multiplicity_free",
        "tensortrace_stage_checked_generic",
        "tensortrace_multiplicity_free_with",
        "tensortrace_checked_generic_with",
        "tensortrace_into_multiplicity_free_with",
        "tensortrace_into_checked_generic_with",
        "tensortrace_fusion_structure_owned_host",
        "tensortrace_fusion_structure_into_host",
        "TracePreflight",
        "tensortrace_fusion_structure_into_on_cuda",
        "NON_SYMMETRIC_CONTRACTION_UNSUPPORTED",
        "SymmetricBraidingOp",
        "admit_checked_generic_pair",
        "reject_non_symmetric_contraction",
        "require_symmetric_braiding",
        "TensorTraceFusionStructure",
        // tree_context
        "CoefficientAlgebra",
        "StagedTreeTransform",
        "StructureOutcome",
        "TreeTransformExecutionContext",
        // tree_transform
        "build_all_codomain_tree_transform_group_plan",
        "build_tree_pair_transform_group_plan",
        "CheckedGenericPlanError",
        "TreeTransformBlockSpec",
        "TreeTransformGroupBlockSpec",
        "TreeTransformGroupPlan",
        "TreeTransformKeyBlockSpec",
        "TreeTransformOperation",
        "TreeTransformOperationKind",
        "TreeTransformRuleCacheKey",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<BTreeSet<_>>();
    assert_eq!(exported(false), expected);
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
