mod backend;
mod checked_generic;
pub(crate) use checked_generic::{
    validate_checked_operand_relation, CheckedAuthority, CheckedContractTxn,
    CHECKED_CONTRACTION_REQUIRES_BOSONIC,
};
mod context;
pub(crate) use context::{
    compile_prelowered_dynamic_tree, compile_stored_dynamic_tree, PlanTarget,
};
mod dynamic;
#[cfg(feature = "cuda")]
#[doc(hidden)]
pub use dynamic::cuda::CudaContractScratch;
pub(crate) use dynamic::{
    compile_contract_twist, compile_core_dst, compile_transformed_source,
    source_is_storage_adjoint, storage_layout_is_canonical, DynamicFusionCoreDstEntry,
    DynamicFusionTransformedSourceEntry, DynamicTreeExecutionArtifact,
};
#[cfg(test)]
pub(crate) use dynamic::{
    execute_dynamic_tree_execution_artifact_for_test,
    execute_prelowered_dynamic_tree_execution_artifact_for_test,
    tensorcontract_fusion_dynamic_plan_into_with,
};
#[cfg(feature = "cuda")]
#[doc(hidden)]
pub use route_cuda::{
    execute_storage_contract_members_cuda, execute_storage_contract_resolution_on_cuda,
    CudaContractMembersWorkspace,
};
#[doc(hidden)]
pub use route_host::HostContractMembersWorkspace;
mod dynamic_space;
mod fusion;
#[cfg(test)]
pub(crate) use fusion::contracted_axis_order_candidates;
#[cfg(test)]
pub(crate) use fusion::FusionContractOrientation;
#[cfg(test)]
pub(crate) use fusion::{
    prepare_tensorcontract_fusion_candidate_facts_dyn_raw,
    prepare_tensorcontract_fusion_plan_dyn_raw,
    prepare_tensorcontract_fusion_plan_dyn_raw_with_axis_order_and_orientation,
    FusionContractCandidateFacts,
};
mod fusion_block;
pub(crate) use fusion_block::{
    compile_checked_generic_core_plan_general,
    compile_fusion_block_contract_plan_prelowered_validated, compile_operand_core_plan_general,
    core_homspace_matches, validate_fusion_contract_rule, ValidatedCoreContract,
};
mod planar;
pub use planar::PlanarContractOrders;
mod resolution;
pub(crate) use resolution::{compile_derived_core_plan, contract_axes_require_twist};
#[cfg(feature = "cuda")]
mod route_cuda;
mod route_host;
#[doc(hidden)]
pub use resolution::{
    CopyCRoute, CoreRoute, DirectCoreExecutor, ExecCaps, HostEagerExecutor,
    StorageContractResolution,
};
#[cfg(test)]
mod candidate_core_tests;
mod scratch;
#[cfg(test)]
mod storage_contract_tests;
mod structure;

#[cfg(test)]
pub(crate) use backend::tensorcontract_structure_with_dense_executor_raw;
#[cfg(test)]
pub(crate) use backend::TensorContractWorkspace;
pub use backend::{HostTensorContractBackend, TensorContractBackend};
#[cfg(test)]
pub use context::tensorcontract_fusion_dyn_prelowered_direct_on_storage;
pub use context::StagedContract;
pub use context::{
    plan_compose, try_compile_storage_contract_core_route, TensorContractFusionExecutionContext,
};
#[cfg(test)]
pub(crate) use dynamic_space::{
    checked_layout_primer, checked_metadata_dispatcher, MetadataOutput, MetadataRequest,
};
pub(crate) use dynamic_space::{
    dispatch_prepare, tree_transform_operation_axes, FusionOperandLayout, LayoutKeyBuilder,
};
pub use dynamic_space::{
    BoundDynamicFusionMapSpace, DynamicFusionMapSpace, FusionOperand,
    PreparedCheckedGenericDynamicSpace, ValidatedDynamicFusionLayout,
};
#[cfg(test)]
pub(crate) use fusion::contracted_fusion_tree_basis_matches;
pub(crate) use fusion::rhs_contract_twist_factor_oriented;
pub use fusion::{
    prepare_tensorcontract_fusion_plan_dyn, tensorcontract_fusion_block_specs, FusionContractPlan,
};
pub(crate) use fusion::{
    prepare_tensorcontract_fusion_plan_dyn_raw_canonical,
    select_complete_tensorcontract_fusion_plan,
};
pub use structure::TensorContractBlockSpec;
#[cfg(test)]
pub(crate) use structure::TensorContractDenseRouteKind;
#[cfg(test)]
pub(crate) use structure::TensorContractStructure;

#[cfg(test)]
pub(crate) use dynamic_space::encoded_layout_primer;
