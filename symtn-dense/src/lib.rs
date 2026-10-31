#![forbid(unsafe_code)]

//! Dense block execution boundary for symtn.
//!
//! Symmetric tensor algorithms lower to this crate through symtn-owned storage
//! views and executors. The storage placement determines the execution path:
//! host views use host kernels, and future device views should use device
//! kernels without exposing concrete runtimes to TensorMap-level code.

mod dot;
mod dtype;
mod error;
mod executor;
mod fresh;
mod layout;
mod scalar;
mod tensor;
mod view;

#[cfg(feature = "cuda")]
mod cuda_adapter;
// Pure host arithmetic: compiled (and tested) without the `cuda` feature for
// the same reason `cuda_region` is.
#[cfg(any(feature = "cuda", test))]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
mod cuda_hermitian;
// Layout metadata only: compiled (and tested) without the `cuda` feature so
// CI, which merely `cargo check`s that feature, still executes these rules.
#[cfg(any(feature = "cuda", test))]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
mod cuda_region;
#[cfg(feature = "tenferro")]
mod block_kernel {
    // The BLAS provider keeps its own grouped scheduler for initialized
    // outputs; it runs fresh outputs here when a forced lane policy rules out
    // its serial vendor loop.
    #[cfg_attr(feature = "cpu-blas-core", allow(dead_code))]
    pub(crate) mod gemm;
    // The one lane driver, cost function and block-op trait every native
    // kernel runs through.
    #[cfg_attr(feature = "cpu-blas-core", allow(dead_code))]
    pub(crate) mod driver;
    // The native provider's block factorizations; BLAS keeps its serial
    // per-block vendor calls.
    #[cfg(all(feature = "cpu-faer", not(feature = "cpu-blas-core")))]
    pub(crate) mod factor;
    #[cfg_attr(feature = "cpu-blas-core", allow(dead_code))]
    pub(crate) mod schedule;
}
#[cfg(test)]
mod probe;
#[cfg(feature = "tenferro")]
mod tenferro_adapter;
#[cfg(test)]
mod tests;

pub use dot::DenseDotConfig;
pub use dtype::{DenseBackend, DenseDType};
pub use error::DenseError;
pub use executor::{
    arity_mismatch, check_contiguous_output, strided_batch_runs, strided_batch_runs_into,
    DenseExecutor, DenseFactorization, DenseGemmBatchJob, DenseLinalgScopeBody, DenseOwned,
    MatrixOp,
};
#[doc(hidden)]
pub use fresh::DenseFreshSegments;
pub use scalar::DenseScalar;
pub use tensor::DenseTensor;
pub use view::{DenseRead, DenseView, DenseViewMut, DenseWrite};

#[cfg(feature = "tenferro")]
pub use tenferro_adapter::DefaultDenseExecutor;

/// The execution resource every numerical entry takes as
/// `engine: &mut dyn Engine`: tenferro's engine trait, re-exported so callers
/// and custom executors need no direct tenferro dependency. The caller owns
/// the engine; symtn stores none.
pub use tenferro_tensor::Engine;

/// The caller-owned CPU engine ([`CpuEngine::serial`], or
/// [`CpuEngine::new`] over a [`Pool`]) and its thread pool, re-exported from
/// tenferro.
#[cfg(feature = "tenferro")]
pub use tenferro_cpu::{CpuEngine, Pool};

#[cfg(feature = "cuda")]
pub use cuda_adapter::{
    cuda_all_finite, cuda_conj, cuda_copy_strided_into, cuda_download_batched_spectra,
    cuda_eigh_region_batched, cuda_gather_columns_batched_into, cuda_gather_elements,
    cuda_gather_member_elements, cuda_gather_members, cuda_hermitian_regions_batched,
};
#[cfg(feature = "cuda")]
pub use cuda_adapter::{
    cuda_copy_region_into, cuda_eigh_region, cuda_gemm_region_batched_into,
    cuda_gemm_region_batched_scaled_into, cuda_gemm_region_into, cuda_gemm_region_with_ops_into,
    cuda_hermitian_regions, cuda_matmul_region_into, cuda_qr_region, cuda_region_axpby,
    cuda_region_scale, cuda_region_trace_accumulate, cuda_region_zero, cuda_svd_region,
    cuda_transfer_stats, cuda_widen, reset_cuda_transfer_stats, CudaDenseContext, CudaDenseStorage,
    CudaRealScalar, CudaRegionBeta, CudaRegionCoefficient, CudaScalar, CudaScalarOperands,
    CudaTransferStats,
};
#[cfg(feature = "cuda")]
pub use cuda_adapter::{
    cuda_copy_spectrum_into, cuda_download_spectra, cuda_svd_gauge_phases, CudaSpectrum,
    CudaSvdGaugeWeights, CudaSvdPhases,
};
#[cfg(feature = "cuda")]
pub use cuda_region::CudaRegion;

/// The caller-owned CUDA engine and the device id it opens on
/// (`CudaEngine::open(CudaDeviceId::from_ordinal(0))`), re-exported from
/// tenferro. Every device operation borrows the engine for its own duration;
/// symtn stores none.
#[cfg(feature = "cuda")]
pub use tenferro_gpu::cuda::{CudaDeviceId, CudaEngine};
