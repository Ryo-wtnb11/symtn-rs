#![forbid(unsafe_code)]

//! Symmetric-tensor execution for symtn (the operational half of what
//! TensorKit.jl does in Julia): fusion-tree transforms (F/R recoupling,
//! braids, permutes), symmetric contraction routes, and the basic tensor
//! operations, lowered onto strided-rs kernels and the dense executor the
//! same way TensorKit lowers onto Strided.jl and dense backends. The
//! symmetry-agnostic einsum layer (TensorOperations.jl's role) lives below
//! this crate in strided-rs / tenferro; the structural data layer lives in
//! `symtn-core`.

mod adjoint;
mod admission;
mod bound_tensor;
mod contract;
mod lowering;
mod mode;
mod oriented_elementwise;
mod physical;
#[cfg(test)]
mod probe;
mod tensortrace;
#[cfg(test)]
mod test_support;
mod tree_context;
mod tree_transform;

pub use symtn_core::RuleIdentity;

pub(crate) use adjoint::adjoint_bound_space_dyn_generic_checked;
pub use adjoint::{
    adjoint_bound_dyn, adjoint_bound_space_dyn, materialize_adjoint_data_dyn, AdjointBlockMap,
};
#[doc(hidden)]
pub use admission::{
    admit_checked_generic_pair, reject_non_symmetric_contraction, require_symmetric_braiding,
    SymmetricBraidingOp, NON_SYMMETRIC_CONTRACTION_UNSUPPORTED,
};
pub use bound_tensor::BoundDynamicTensorRef;
#[cfg(test)]
pub(crate) use contract::tensorcontract_fusion_dyn_prelowered_direct_on_storage;
#[cfg(test)]
pub(crate) use contract::{contracted_fusion_tree_basis_matches, TensorContractDenseRouteKind};
#[cfg(feature = "cuda")]
#[doc(hidden)]
pub use contract::{
    execute_storage_contract_members_cuda, execute_storage_contract_resolution_on_cuda,
    CudaContractMembersWorkspace, CudaContractScratch,
};
#[doc(hidden)]
pub use contract::{
    plan_compose, try_compile_storage_contract_core_route, HostContractMembersWorkspace,
    PlanarContractOrders, StagedContract, StorageContractResolution,
};
pub use contract::{
    prepare_tensorcontract_fusion_plan_dyn, tensorcontract_fusion_block_specs, FusionContractPlan,
    HostTensorContractBackend, TensorContractBackend, TensorContractBlockSpec,
    TensorContractFusionExecutionContext,
};
pub use contract::{
    BoundDynamicFusionMapSpace, DynamicFusionMapSpace, FusionOperand, ValidatedDynamicFusionLayout,
};
#[doc(hidden)]
pub use contract::{CopyCRoute, CoreRoute, DirectCoreExecutor, ExecCaps, HostEagerExecutor};
#[cfg(test)]
pub(crate) use contract::{
    PreparedCheckedGenericDynamicSpace, TensorContractStructure, TensorContractWorkspace,
};
#[doc(hidden)]
pub use mode::ContractRequest;
#[doc(hidden)]
pub use mode::{CoefficientAlgebra, PivotalCoefficientAlgebra, RigidCoefficientAlgebra};
pub(crate) use oriented_elementwise::validate_oriented_fusion_layout;
#[doc(hidden)]
pub use oriented_elementwise::{
    fusion_scatter_add_assign, oriented_fusion_add_owned, oriented_fusion_axpby_into,
    oriented_fusion_inner, oriented_fusion_inner_with, oriented_fusion_restrict_owned,
    stacked_fusion_restrict_owned, SectorRunTable, SelectedRuns,
};
#[doc(hidden)]
pub use physical::{expand_physical_host, project_physical_host, PhysicalConversionError};
/// CUDA storage and GEMM seams (flat device buffers, never host-readable).
#[cfg(feature = "cuda")]
pub use symtn_operations::cuda;
#[doc(hidden)]
pub use symtn_operations::host_pool;
pub use symtn_operations::ContractDestinationInit;
pub use symtn_operations::OperationError;
pub use symtn_operations::ReportsPlacement;
pub(crate) use symtn_operations::StridedHostKernelAdapter;
pub(crate) use symtn_operations::TreeTransformReplayProfile;
pub use symtn_operations::TreeTransformStructure;
pub use symtn_operations::TreeTransformWorkspace;
pub(crate) use symtn_operations::{host_scratch, strided};
#[doc(hidden)]
pub use symtn_operations::{try_cat_owned_raw, OwnedCatCopy, OwnedCatSide};
#[doc(hidden)]
pub use symtn_operations::{zeroed_payload, ZeroBytes};
pub use symtn_operations::{ConjugateValue, RealStructuralCoefficient, TreeTransformScalar};
pub use symtn_operations::{
    DenseBlockScalar, DenseRecouplingScalar, RecouplingCoefficientAction, WideScalar,
};
pub use symtn_operations::{DenseTreeTransformOperations, TreeTransformBackend};
pub use symtn_operations::{OutputAxisOrder, TensorContractSpec, TensorTraceAxisSpec};
#[cfg(feature = "cuda")]
#[doc(hidden)]
pub use tensortrace::tensortrace_fusion_structure_into_on_cuda;
#[cfg(all(test, feature = "cgc-gen"))]
pub(crate) use tensortrace::{
    tensortrace_checked_generic_in, tensortrace_fusion_dyn_owned_generic_checked,
};
pub use tensortrace::{
    tensortrace_checked_generic_with, tensortrace_fusion_dyn_preflight_checked,
    tensortrace_fusion_dyn_preflight_generic_checked, tensortrace_fusion_structure_into_host,
    tensortrace_fusion_structure_owned_host, tensortrace_into_checked_generic_with,
    tensortrace_into_multiplicity_free_with, tensortrace_multiplicity_free_with,
    tensortrace_stage_checked_generic, tensortrace_stage_multiplicity_free,
    TensorTraceFusionStructure, TracePreflight,
};
#[cfg(test)]
pub(crate) use tensortrace::{tensortrace_fusion_dyn_into, tensortrace_fusion_dyn_owned};
#[cfg(test)]
pub(crate) use tensortrace::{
    tensortrace_fusion_dyn_into_checked, tensortrace_fusion_dyn_owned_checked,
    tensortrace_fusion_dyn_structure_into_raw, tensortrace_fusion_dyn_structure_owned,
    tensortrace_multiplicity_free_in, TensorTraceStage,
};
pub use tree_context::{StagedTreeTransform, StructureOutcome, TreeTransformExecutionContext};
#[doc(hidden)]
pub use tree_transform::{
    admit_exact_tree_pair_layout, exact_layout_tree_pair_hit, TreeTransformOperationView,
};
pub use tree_transform::{
    build_all_codomain_tree_transform_group_plan, build_tree_pair_transform_group_plan,
    CheckedGenericPlanError, TreeTransformBlockSpec, TreeTransformGroupBlockSpec,
    TreeTransformGroupPlan, TreeTransformKeyBlockSpec, TreeTransformOperation,
    TreeTransformOperationKind, TreeTransformRuleCacheKey,
};
#[cfg(test)]
pub(crate) use tree_transform::{
    build_checked_generic_tree_pair_transform_group_plan,
    build_generic_tree_pair_transform_group_plan, build_tree_transform_group_plan,
};
#[cfg(test)]
pub(crate) use tree_transform::{
    build_multiplicity_free_all_codomain_tree_transform_group_plan,
    build_multiplicity_free_tree_pair_transform_group_plan,
    build_unique_all_codomain_tree_transform_group_plan,
    build_unique_tree_pair_transform_group_plan, build_unique_tree_transform_group_plan,
};

#[cfg(test)]
mod tests;

/// The workspace tolerance rule for arithmetic test comparisons
/// (`docs/testing_numerics.md`).
#[cfg(test)]
#[path = "../../tests/support"]
mod test_numerics {
    use num_complex::{Complex32, Complex64};
    pub(crate) mod numerics;
}
