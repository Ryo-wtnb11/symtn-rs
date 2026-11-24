mod block_specs;
mod plan;
pub(crate) use plan::contracted_axis_order_candidates;
#[cfg(test)]
pub(crate) use plan::ContractAxisOrderCandidate;
pub use plan::FusionContractOrientation;
pub(crate) use plan::{
    contract_twist_on_physical_lhs, min_dynamic_tree_materialized_elements,
    select_complete_tensorcontract_fusion_plan, CACHED_ORIENTATIONS,
};
#[cfg(test)]
pub(crate) use plan::{
    prepare_tensorcontract_fusion_candidate_facts_dyn_raw,
    prepare_tensorcontract_fusion_plan_dyn_raw_with_axis_order_and_orientation,
    FusionContractCandidateFacts,
};

#[cfg(test)]
pub(crate) use block_specs::contracted_fusion_tree_basis_matches;
pub use block_specs::tensorcontract_fusion_block_specs;
pub(crate) use block_specs::{
    contract_twist_factor, external_axis_is_dual, uniform_contract_twist,
};
#[cfg(test)]
pub(crate) use plan::prepare_tensorcontract_fusion_plan_dyn_raw;
pub use plan::{prepare_tensorcontract_fusion_plan_dyn, FusionContractPlan};
pub(crate) use plan::{
    prepare_tensorcontract_fusion_plan_dyn_prelowered_canonical,
    prepare_tensorcontract_fusion_plan_dyn_raw_canonical,
};
