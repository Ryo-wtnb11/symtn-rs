//! Expert layer: how a [`crate::typed::TensorMap`] is stored and executed,
//! rather than what it is mathematically.
//!
//! This is the one module for representation queries, block views, storage
//! and placement, the injectable dense executor, and process-wide transfer
//! observability; structure caches are controlled through [`crate::cache`].
//! Every tensor operation has its single public path on
//! [`crate::typed::TensorMap`]; nothing here is a second path to one.
//! [`crate::typed::TensorMap::diagview`] is the user-layer reader of a
//! diagonal, whatever the storage.

/// Block views returned by `TensorMap::subblock(s)`, and the placement and
/// storage contracts of a tensor's buffer.
pub use symtn_core::{
    BlockLayout, BlockRef, BlockView, HostReadableStorage, Placement, SectorLeg,
    SectorLegConstructionError, TensorStorage,
};
/// Per-thread CUDA boundary counters and the plan-cache snapshot returned by
/// `Runtime::cuda_plan_cache_stats`.
#[cfg(feature = "cuda")]
pub use symtn_dense::{
    cuda_transfer_stats, reset_cuda_transfer_stats, CudaPlanCacheStats, CudaTransferStats,
};
/// The device context a CUDA runtime dereferences to, and the device buffer
/// inside a `CudaStorage`.
#[cfg(feature = "cuda")]
pub use symtn_dense::{CudaDenseContext, CudaDenseStorage};
/// The dense executor seam of `RuntimeBuilder::with_dense_executor` and the
/// types its methods exchange.
pub use symtn_dense::{
    DefaultDenseExecutor, DenseBackend, DenseDType, DenseDotConfig, DenseError, DenseExecutor,
    DenseFactorization, DenseGemmBatchJob, DenseLinalgScopeBody, DenseOwned, DenseRead,
    DenseScalar, DenseTensor, DenseView, DenseViewMut, DenseWrite, MatrixOp, SharedCpuContext,
};
/// The tree-transform descriptor carried by `OperationError` variants.
pub use symtn_tensors::{TreeTransformOperation, TreeTransformOperationKind};

use crate::sector::{
    CheckedFusionAlgebra, MultiplicityFreeRigidSymbols, SectorCodec, TypedSectorAdmission,
};
use crate::typed::{
    Error, FacadeError, SectorSpectrum, TensorMap, TensorScalar, TypedTensorRootDispatch,
};

/// Returns the compact diagonal spectrum without materializing dense data.
///
/// [`None`] means the representation has no directly stored compact spectrum
/// (it is dense or a lazy adjoint); otherwise it clones only the
/// `O(Σ_c k_c)` compact values in canonical bond-sector order.
#[expect(
    clippy::type_complexity,
    reason = "the compact readback exposes provider-labelled sector spectra"
)]
pub fn diagonal_spectrum<R, D>(
    tensor: &TensorMap<R, D>,
) -> Result<Option<Vec<SectorSpectrum<R::Sector, D>>>, FacadeError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorRootDispatch<R>,
    D: TensorScalar,
{
    tensor.diagonal_spectrum()
}

/// Tests a rank-one map for blockwise diagonality without materializing
/// compact storage.
///
/// Compact storage is diagonal by construction and returns `true` without a
/// scan. This matches TensorKit `isdiag` for finite data at `tol = 0`;
/// positive tolerance uses `max_offdiag <= tol * max(norm(Inf), 1)`.
/// Negative and non-finite tolerances are rejected before every shortcut.
/// Scale `tol` to the payload dtype, as
/// [`FactorizationScalar`](crate::typed::FactorizationScalar) describes.
pub fn is_diagonal<R, D>(tensor: &TensorMap<R, D>, tol: f64) -> Result<bool, Error>
where
    R: MultiplicityFreeRigidSymbols + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    tensor.is_diagonal(tol)
}
