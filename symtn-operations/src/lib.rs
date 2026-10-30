#![deny(unsafe_code)]

//! Symmetry-free execution layer for symtn (the TensorOperations.jl role):
//! scalar contracts, strided kernels and the host kernel adapter, scratch and
//! workspace reuse, and the replay-facing error and profile types. Nothing in
//! this crate consumes fusion rules or enumerates fusion trees; the symmetric
//! compile layer (`symtn-tensors`) hands down already-enumerated fusion-tree
//! pair identities and scalar coefficients, which are resolved to offsets and
//! strides before kernel execution.
//!
//! The `testing` feature is unstable and internal: it exposes the per-block
//! strided tree-transform driver and scalar recoupling as a reference oracle
//! (and adds a method to `HostKernelAdapter`). Do not enable it in applications.

#[cfg(not(any(
    feature = "cpu-faer",
    feature = "cpu-blas",
    feature = "blas-accelerate",
    feature = "blas-openblas",
    feature = "blas-mkl",
    feature = "provider-inject"
)))]
compile_error!(
    "symtn-operations requires a host execution backend; enable cpu-faer, cpu-blas, a blas-* provider, or provider-inject (cuda still requires host replay)"
);

pub mod axis;
mod checked_block_layout;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub mod cuda_transform;
// Host-side planning only: compiled (and tested) without the `cuda` feature so
// CI, which merely `cargo check`s that feature, still executes these rules.
#[cfg(any(feature = "cuda", test))]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
mod cuda_transform_plan;
mod error;
pub mod fusion_replay;
#[doc(hidden)]
pub mod host_pool;
mod host_scalar_kernels;
pub mod host_scratch;
pub mod kernel_adapter;
mod placement;
pub mod replay_backend;
mod scalar;
#[doc(hidden)]
pub mod stacked;
pub mod storage_scratch;
pub mod strided;
pub mod structure_identity;
mod task_view;
mod transform_helpers;
mod transform_key;
pub mod transform_plan;
pub mod transform_replay;
pub mod transform_structure;
mod tree_profile;

pub use axis::*;
#[doc(hidden)]
pub use checked_block_layout::CheckedBlockLayout;
#[cfg(feature = "cuda")]
pub use cuda_transform::{
    CudaTreeTransformDestination, CudaTreeTransformExecutor, DEFAULT_COEFFICIENT_BUDGET_BYTES,
};
pub use error::OperationError;
pub use fusion_replay::{
    fusion_scale_block_layouts_excluding, ContractDestinationInit, FusionBlockContractGroupPlan,
    FusionBlockContractPlan, FusionBlockContractWorkspace, FusionBlockMatrixGroup,
    FusionScaleBlockLayout, FusionStridedBlockLayout, FusionSubblockMatrixLayout,
    HostFusionBlockContractWorkspace, Rank2Gemm, StorageGemm,
};
pub(crate) use host_scalar_kernels::tensoradd_raw_strided_kernel_trusted;
pub use host_scalar_kernels::{
    axpby_raw_strided_kernel_trusted, scale_raw_strided_kernel_trusted,
    tensoradd_raw_strided_kernel,
};
#[doc(hidden)]
pub use host_scalar_kernels::{
    bilinear_raw_strided_kernel_mapped, tensortrace_raw_strided_kernel_add_with_coefficient_trusted,
};
pub use kernel_adapter::{BakedFusedLayout, HostKernelAdapter, StridedHostKernelAdapter};
pub use placement::ReportsPlacement;
#[cfg(any(test, feature = "testing"))]
#[doc(hidden)]
pub use probe::{take_checked_block_passes, take_owned_block_prefills, CheckedBlockPasses};
pub use replay_backend::*;
pub use scalar::{
    scale_value, ConjugateValue, DenseBlockScalar, DenseRecouplingScalar,
    RealStructuralCoefficient, RecouplingCoefficientAction, TransformScale, TreeTransformScalar,
    WideScalar,
};
pub use transform_key::{TreeTransformOperation, TreeTransformOperationKind};
pub use transform_plan::{
    TreeTransformBlockSpec, TreeTransformGroupBlockSpec, TreeTransformGroupPlan,
    TreeTransformKeyBlockSpec,
};
pub use transform_replay::*;
pub use transform_structure::*;
pub use tree_profile::TreeTransformReplayProfile;
// Why not retain crate-level `forbid`: Rust does not permit a narrower module
// to override it. `deny` keeps every other module unsafe-free while this one
// owns the audited Vec length transition.
#[cfg(any(feature = "cuda", test))]
mod opaque_admission;
mod owned_blocks;
mod owned_cat;
#[allow(unsafe_code)]
mod owned_overwrite_buffer;
#[doc(hidden)]
pub use owned_blocks::{
    check_axis_runs, for_each_run_piece, overwrite_owned_blocks, overwrite_owned_member_blocks,
    AxisRuns, BlockOverwrite,
};
#[doc(hidden)]
pub use owned_overwrite_buffer::{zeroed_payload, ZeroBytes};
mod owned_trace;
#[cfg(any(test, feature = "testing"))]
mod probe;
#[doc(hidden)]
pub use owned_cat::{
    try_cat_owned_c64_raw, try_cat_owned_mixed_raw, try_cat_owned_raw, OwnedCatC64Source,
    OwnedCatCopy, OwnedCatSide,
};
#[doc(hidden)]
pub use owned_trace::{try_tensortrace_owned_raw, OwnedTraceTerm};

/// The workspace tolerance rule for arithmetic test comparisons
/// (`docs/testing_numerics.md`).
#[cfg(test)]
#[path = "../../tests/support"]
mod test_numerics {
    use num_complex::{Complex32, Complex64};
    pub(crate) mod numerics;
}
