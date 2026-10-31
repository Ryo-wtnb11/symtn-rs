//! Export firewall for `symtn-network`.
//!
//! Pins the exact names `lib.rs` re-exports, under every feature combination
//! at once, so adding or removing one fails here and is reviewed as an API
//! change. The block-sparse planner (`BlockSparseCostModel`,
//! `GreedyBlockSparseOptimizer`, ...) was removed because nothing fed it
//! `TensorMap` sectors; it must not come back without a sector-aware design.
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
fn symtn_network_exports_exactly_the_pinned_names() {
    let expected = [
        // cost
        "DenseCostModel",
        "DenseTensorInfo",
        // cotengra-python
        "CotengraPythonOptimizer",
        "CotengraMinimize",
        "CotengraPythonConfig",
        "CotengraPythonMethod",
        "CotengraSlicingConfig",
        // error
        "ContractError",
        "Result",
        "SliceError",
        "SliceResult",
        "SymmetricSliceExecutionError",
        "SymmetricSliceLowerError",
        // ir / labels
        "HyperEdge",
        "NetworkIR",
        "TensorNode",
        "LabelOccurrence",
        "TemporaryLabel",
        "TensorAxis",
        "TensorId",
        // network
        "Network",
        "NetworkExecutionWorkspace",
        "NetworkOperand",
        "PlannedNetwork",
        "SymmetricSliceStats",
        // optimizer
        "ContractionStep",
        "DenseContractionOptimizer",
        "DensePlanCostReport",
        "GreedyDenseOptimizer",
        "LabelOrderDenseOptimizer",
        // opt-path
        "BranchLevel",
        "DpObjective",
        "OptEinsumPathOptimizer",
        "PathMemoryLimit",
        "PathStrategy",
        // plan
        "active_pair_path_from_steps",
        "active_pair_path_from_tree",
        "dense_steps_from_active_pair_path",
        "ActivePair",
        "ContractionPlan",
        // plancache
        "clear_plan_cache",
        "configure_plan_cache",
        "load_plan_cache",
        "plan_cache_config",
        "plan_cache_stats",
        "save_plan_cache",
        "Optimizer",
        "PlanCacheConfig",
        "PlanCacheStats",
        "ReplanPolicy",
        "DEFAULT_PLAN_CACHE_CAPACITY",
        "DEFAULT_REPLAN_DRIFT_FACTOR",
        "DEFAULT_WORKSPACE_BUDGET_BYTES",
        // slice
        "best_next_internal_index",
        "best_next_slice_index",
        "contraction_width",
        "greedy_slice",
        "slice_plan_for",
        "DegeneracyRange",
        "SectorSlice",
        "SliceKind",
        "SliceLabels",
        "SlicePlan",
        "SlicedPlan",
        "SymmetricIndexSlice",
        "SymmetricSlicePlan",
        "SymmetricSliceSpec",
        "SymmetricSlicedPlan",
        // tree
        "ContractionTree",
        // index-notation front-end (symtn-macros)
        "contract",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<BTreeSet<_>>();
    assert_eq!(
        exported_names(include_str!("../src/lib.rs")).unwrap(),
        expected
    );
}
