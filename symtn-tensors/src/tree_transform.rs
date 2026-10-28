mod cache;
mod operation;
mod plan;

pub use cache::{
    admit_exact_tree_pair_layout, exact_layout_tree_pair_hit, TreeTransformOperationView,
};
pub(crate) use cache::{
    lookup_bound, lookup_trace, publish_staged_trace, publishable, CheckedPendingCoefficients,
    CoefficientGroupReuse, CompletedTransformerKey, OrientedBasisOrder, PendingCoefficientGroups,
    TraceKeyRef, TraceSplit, TraceTermReuse, TransformerMode, TreeTransformPlanning,
    TreeTransformScope,
};
pub use operation::{
    TreeTransformOperation, TreeTransformOperationKind, TreeTransformRuleCacheKey,
};
pub use plan::{
    build_all_codomain_tree_transform_group_plan, build_tree_pair_transform_group_plan,
    CheckedGenericPlanError, TreeTransformBlockSpec, TreeTransformGroupBlockSpec,
    TreeTransformGroupPlan, TreeTransformKeyBlockSpec,
};
#[cfg(test)]
pub(crate) use plan::{
    build_checked_generic_tree_pair_transform_group_plan,
    build_oriented_tree_pair_transform_group_plan_capability_validated,
};
pub(crate) use plan::{
    build_checked_generic_tree_pair_transform_group_plan_validated,
    validate_checked_generic_tree_pair_plan_preflight, CheckedGenericTreePairPreflight,
};
#[cfg(test)]
pub use plan::{build_generic_tree_pair_transform_group_plan, build_tree_transform_group_plan};
#[cfg(test)]
pub(crate) use plan::{
    build_multiplicity_free_all_codomain_tree_transform_group_plan,
    build_multiplicity_free_tree_pair_transform_group_plan,
    build_tree_pair_transform_group_plan_validated_with_threads,
    build_unique_all_codomain_tree_transform_group_plan,
    build_unique_tree_pair_transform_group_plan, build_unique_tree_transform_group_plan,
    partition_staged_groups_for_test, validate_multiplicity_free_tree_pair_preflight,
};
