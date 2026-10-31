//! CUDA dense boundary: flat device buffers and offset-addressed matrix
//! GEMM, delegated to tenferro-gpu. This module is the only place in the
//! symtn workspace that touches tenferro GPU types; upper layers see opaque
//! storage handles and `DenseError`.
//!
//! Split by responsibility: device context and transfer stats stay here with
//! the scalar-admission traits every child uses; `context`, `storage`,
//! `gemm`, `elementwise`, `hermitian` and `factorization` each own one
//! device-boundary concern.

use std::cell::Cell;

use num_complex::{Complex32, Complex64};
use tenferro_gpu::cuda::raw::with_cuda_engine;
use tenferro_gpu::cuda::{download_tensor, upload_tensor, CudaRuntime};
use tenferro_linalg::{QrGauge, QrOptions, TensorReadLinalgExt};
use tenferro_tensor::{
    AllocationDomainId, CompareDir, ContractionScalar, DotGeneralAccumulation, DotGeneralConfig,
    Engine, Tensor, TensorRead, TensorScalar as TenferroScalar, TensorView, TensorViewMut,
    TensorWrite, TypedTensor,
};

use super::{DenseBackend, DenseDType, DenseError, MatrixOp};
use crate::cuda_hermitian::{
    input_stage, residual_stage, scaled_hermitian_residual_accepts, StageOutcome,
};
use crate::cuda_region::{
    validate_destination_layout, validate_gather_rows, validate_region, CudaRegion,
};
use crate::tensor::dense_dtype_from_tenferro;

mod context;
mod elementwise;
mod factorization;
mod finite;
mod gemm;
mod hermitian;
mod storage;
#[cfg(test)]
mod tests;

// Re-exports so `crate::cuda_adapter::Item` (what `lib.rs`'s `pub use
// cuda_adapter::{...}` and every other in-crate caller names) still resolves
// once the item's *definition* moves into one of the child files above.
pub use context::{CudaDenseContext, CudaScalarOperands};
pub use elementwise::{
    cuda_conj, cuda_copy_region_into, cuda_copy_strided_into, cuda_gather_elements,
    cuda_gather_member_elements, cuda_gather_members, cuda_region_axpby, cuda_region_scale,
    cuda_region_trace_accumulate, cuda_region_zero, cuda_widen, CudaRegionBeta,
    CudaRegionCoefficient,
};
pub use factorization::{
    cuda_copy_spectrum_into, cuda_download_batched_spectra, cuda_download_spectra,
    cuda_eigh_region, cuda_eigh_region_batched, cuda_gather_columns_batched_into, cuda_qr_region,
    cuda_svd_gauge_phases, cuda_svd_region, CudaSpectrum, CudaSvdGaugeWeights, CudaSvdPhases,
};
pub use finite::cuda_all_finite;
pub use gemm::{
    cuda_gemm_region_batched_into, cuda_gemm_region_batched_scaled_into, cuda_gemm_region_into,
    cuda_gemm_region_with_ops_into, cuda_matmul_region_into,
};
pub use hermitian::{cuda_hermitian_regions, cuda_hermitian_regions_batched};
pub use storage::CudaDenseStorage;

// `context::warm_up` submits an axpby the same way every region call does.
use elementwise::submit_region_axpby;
// The SVD host-gauge (factorization) validates its payload the same way
// every other region primitive does.
use elementwise::ensure_payload_dtype;

// Cross-child visibility for private helpers the test module needs: it is a
// sibling of these children, so the items it reaches through a *different*
// file need `pub(super)` there and a re-export here for its `use super::*`
// to pick them up. Gated
// to `cfg(test)` so a non-test build does not carry an unused import.
#[cfg(test)]
use context::SCALAR_OPERAND_SLOTS;
#[cfg(test)]
use elementwise::reject_zero_alpha;
#[cfg(test)]
use factorization::{
    ensure_spectra_device, validate_eigh_factor_shapes, validate_qr_factor_shapes,
    validate_svd_factor_shapes,
};

mod cuda_scalar_sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for num_complex::Complex32 {}
    impl Sealed for num_complex::Complex64 {}
}
/// Payload dtypes a symtn CUDA buffer may own.
///
/// Sealed to the four floating payloads `f32`, `f64`, [`Complex32`] and
/// [`Complex64`]: structural (fusion-tree) coefficients stay real, so only the
/// *payload* varies. Conjugation is never a payload property here — it is
/// carried as a GEMM operand flag.
///
/// Every payload names a **real lane** ([`TenferroScalar::Real`]: `f32` for
/// `f32` and [`Complex32`], `f64` for `f64` and [`Complex64`]). Real device
/// metadata — a spectrum, a reduction, the Hermitian test's normalizers — is
/// produced and consumed in that lane rather than in `f64` by assumption:
/// Tenferro answers a single-precision payload with `F32` metadata and
/// rejects an `F64` operand against it
/// (`benchmarks/history/cuda-single-precision-probe-2026-09-20.md`). A lane
/// value enters the payload dtype through `From` (`(s, 0)` for a complex
/// payload) when it must meet the payload in one elementwise op. Metadata
/// symtn decides on is widened to `f64` at the download, so the host-side
/// spectrum contract is unchanged.
///
/// Admitting a dtype here does not by itself open a typed tensor API for it;
/// `symtn`'s `CudaPayload` admits all four (#1336).
pub trait CudaScalar:
    TenferroScalar<Real: CudaRealScalar>
    + From<<Self as TenferroScalar>::Real>
    + PartialEq
    + cuda_scalar_sealed::Sealed
{
    /// symtn-side dtype tag, used for [`DenseError::DTypeMismatch`].
    const DTYPE: DenseDType;
    /// Additive identity, for `beta = 0` overwriting GEMMs.
    const ZERO: Self;
    /// Multiplicative identity, for unscaled GEMMs.
    const ONE: Self;
    /// Whether this payload has an imaginary part. Used to skip the
    /// value-preserving `conj`/`abs` passes on the real dtype; it is a dtype
    /// invariant, not a size or workload heuristic.
    const IS_COMPLEX: bool;

    /// Which of the context's lazily created scalar-operand slots this dtype
    /// owns. One slot per admitted dtype: the operands are *payload*-typed, so
    /// a slot shared between `f32` and `f64` would hand a buffer of the wrong
    /// dtype to the next call.
    #[doc(hidden)]
    const OPERAND_SLOT: usize;

    /// The backend's dtype-erased GEMM coefficient for this payload.
    fn contraction_scalar(self) -> ContractionScalar;

    /// The value's bit pattern, real then imaginary part: a total order key
    /// for grouping equal scales (the float order is only partial).
    #[doc(hidden)]
    fn bit_pattern(self) -> [u64; 2] {
        match self.contraction_scalar() {
            ContractionScalar::F32(value) => [u64::from(value.to_bits()), 0],
            ContractionScalar::F64(value) => [value.to_bits(), 0],
            ContractionScalar::C32(value) => {
                [u64::from(value.re.to_bits()), u64::from(value.im.to_bits())]
            }
            ContractionScalar::C64(value) => [value.re.to_bits(), value.im.to_bits()],
        }
    }

    /// The typed tensor behind a dtype-erased device buffer, if the dtypes agree.
    fn typed(tensor: &Tensor) -> Option<&TypedTensor<Self>> {
        tensor.as_typed::<Self>()
    }

    /// Mutable counterpart of [`CudaScalar::typed`].
    fn typed_mut(tensor: &mut Tensor) -> Option<&mut TypedTensor<Self>> {
        tensor.as_typed_mut::<Self>()
    }
}
/// The real lane of a [`CudaScalar`] payload.
///
/// Sealed to `f32` and `f64`, which are exactly the
/// [`TenferroScalar::Real`] types of the admitted payloads. It exists so the
/// adapter's real-metadata paths — spectrum and reduction downloads, the
/// Hermitian normalizers, the Hermitian tolerance — are typed by the payload
/// instead of assuming double precision, while the values symtn's host-side
/// decisions consume stay `f64`.
pub trait CudaRealScalar: TenferroScalar + cuda_scalar_sealed::Sealed {
    /// This lane's machine epsilon, in the `f64` the host decides in.
    const EPSILON: f64;

    /// Widens one downloaded metadata value to the host decision type. Exact:
    /// every `f32` is an `f64`.
    fn widen(self) -> f64;

    /// Narrows a host scalar to this lane, for a device operand. The
    /// only values narrowed here are powers of two inside this lane's range,
    /// so the value round-trips.
    fn narrow(value: f64) -> Self;

    /// This lane's `MAX_EXP`: its largest normal power of two is
    /// `2^(MAX_EXP - 1)` and its smallest is `2^-(MAX_EXP - 2)`.
    const MAX_EXP: i32;
}
impl CudaRealScalar for f32 {
    const EPSILON: f64 = f32::EPSILON as f64;
    const MAX_EXP: i32 = f32::MAX_EXP;

    fn widen(self) -> f64 {
        f64::from(self)
    }

    fn narrow(value: f64) -> Self {
        value as Self
    }
}
impl CudaRealScalar for f64 {
    const EPSILON: f64 = f64::EPSILON;
    const MAX_EXP: i32 = f64::MAX_EXP;

    fn widen(self) -> f64 {
        self
    }

    fn narrow(value: f64) -> Self {
        value
    }
}
impl CudaScalar for f32 {
    const DTYPE: DenseDType = DenseDType::F32;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const IS_COMPLEX: bool = false;
    const OPERAND_SLOT: usize = 0;

    fn contraction_scalar(self) -> ContractionScalar {
        ContractionScalar::F32(self)
    }
}
impl CudaScalar for Complex32 {
    const DTYPE: DenseDType = DenseDType::C32;
    const ZERO: Self = Complex32::new(0.0, 0.0);
    const ONE: Self = Complex32::new(1.0, 0.0);
    const IS_COMPLEX: bool = true;
    const OPERAND_SLOT: usize = 2;

    fn contraction_scalar(self) -> ContractionScalar {
        ContractionScalar::C32(self)
    }
}
impl CudaScalar for f64 {
    const DTYPE: DenseDType = DenseDType::F64;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const IS_COMPLEX: bool = false;
    const OPERAND_SLOT: usize = 1;

    fn contraction_scalar(self) -> ContractionScalar {
        ContractionScalar::F64(self)
    }
}
impl CudaScalar for Complex64 {
    const DTYPE: DenseDType = DenseDType::C64;
    const ZERO: Self = Complex64::new(0.0, 0.0);
    const ONE: Self = Complex64::new(1.0, 0.0);
    const IS_COMPLEX: bool = true;
    const OPERAND_SLOT: usize = 3;

    fn contraction_scalar(self) -> ContractionScalar {
        ContractionScalar::C64(self)
    }
}
fn dtype_mismatch<D: CudaScalar>(op: &'static str, tensor: &Tensor) -> DenseError {
    match dense_dtype_from_tenferro(tensor.dtype()) {
        Ok(actual) => DenseError::DTypeMismatch {
            op,
            expected: D::DTYPE,
            actual,
        },
        Err(err) => err,
    }
}
// Why thread-local rather than process-wide or per-context: every record site
// runs on the thread that called into this module (the caller's CUDA engine is
// borrowed on the caller's thread, and no worker or pool thread submits device
// work), so a thread-local counter
// attributes exactly the caller's work. A process-wide counter mixes in
// concurrent callers (#1568); a per-context counter would need the context at
// sites such as `CudaDenseStorage::download` that do not have it.
std::thread_local! {
    static TRANSFER_STATS: Cell<CudaTransferStats> = const {
        Cell::new(CudaTransferStats {
            h2d_calls: 0,
            h2d_bytes: 0,
            d2h_calls: 0,
            d2h_bytes: 0,
            device_allocs: 0,
            gemm_calls: 0,
            solver_calls: 0,
            copy_calls: 0,
            gauge_ops: 0,
        })
    };
}
fn record(update: impl FnOnce(&mut CudaTransferStats)) {
    TRANSFER_STATS.with(|cell| {
        let mut stats = cell.get();
        update(&mut stats);
        cell.set(stats);
    });
}
/// A snapshot of the calling thread's CUDA boundary observation counters.
///
/// Observability only: nothing in this module reads these values back, so no
/// execution decision, dispatch, or capability depends on them. They exist so
/// a benchmark or a device test can attribute host/device traffic and backend
/// submissions to a measured phase without re-deriving them from an external
/// profiler.
///
/// Scope, as counted at *this* seam:
///
/// - `h2d_*` / `d2h_*`: every host buffer this module uploads or downloads,
///   including the one-element scalar uploads and reduction/spectrum
///   downloads. Bytes are the host-side payload bytes.
/// - `device_allocs`: device buffers this module creates or takes ownership
///   of, i.e. uploads plus tenferro-produced tensors wrapped as
///   [`CudaDenseStorage`] (factorization factors). Tenferro's own solver
///   workspaces and the intermediate tensors of `cuda_hermitian_regions`
///   (`abs`, `div`, `sub`, the reductions) allocate on device but are not
///   visible as buffers here and are therefore not counted.
///   Nor are the solvers' device spectra ([`CudaSpectrum`]) or the
///   `concatenate` output that [`cuda_download_spectra`] gathers them into.
/// - `gemm_calls`: `dot_general` submissions, from
///   `cuda_gemm_region_strided_into` and from the region primitives
///   ([`cuda_region_axpby`], [`cuda_region_zero`]) alike.
/// - `solver_calls`: cuSOLVER region calls (SVD, QR, EIGH).
/// - `copy_calls`: `cuda_copy_region_into` calls that move data.
/// - `gauge_ops`: Tenferro op submissions of the compact-SVD sign/phase
///   gauge ([`cuda_svd_gauge_phases`] and [`CudaSvdPhases`]).
///
/// The counters are per thread: they count the work this module performs on
/// the thread that reads them, and never work submitted by another thread. A
/// device operation runs on its caller's thread, so a delta taken around a
/// call attributes exactly that call's work even while other threads use the
/// same or another device. To total several threads, read on each thread.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CudaTransferStats {
    pub h2d_calls: u64,
    pub h2d_bytes: u64,
    pub d2h_calls: u64,
    pub d2h_bytes: u64,
    pub device_allocs: u64,
    pub gemm_calls: u64,
    pub solver_calls: u64,
    pub copy_calls: u64,
    pub gauge_ops: u64,
}
/// Reads the calling thread's CUDA boundary observation counters. See
/// [`CudaTransferStats`].
pub fn cuda_transfer_stats() -> CudaTransferStats {
    TRANSFER_STATS.with(Cell::get)
}
/// Zeroes the calling thread's CUDA boundary observation counters; other
/// threads' counters are untouched. See [`CudaTransferStats`].
pub fn reset_cuda_transfer_stats() {
    TRANSFER_STATS.with(|cell| cell.set(CudaTransferStats::default()));
}
fn record_h2d(bytes: usize) {
    record(|stats| {
        stats.h2d_calls += 1;
        stats.h2d_bytes += bytes as u64;
        stats.device_allocs += 1;
    });
}
fn record_d2h(bytes: usize) {
    record(|stats| {
        stats.d2h_calls += 1;
        stats.d2h_bytes += bytes as u64;
    });
}
fn cuda_error(op: &'static str, err: impl std::fmt::Display) -> DenseError {
    DenseError::Backend {
        backend: DenseBackend::Cuda,
        op,
        message: err.to_string(),
    }
}
fn cuda_operand_view(op: MatrixOp, rows: usize, cols: usize) -> ([usize; 2], bool) {
    match op {
        MatrixOp::Identity => ([1, rows], false),
        MatrixOp::Transpose => ([cols, 1], false),
        MatrixOp::Adjoint => ([cols, 1], true),
    }
}
/// Where a device buffer lives: the device ordinal, kept as metadata, and the
/// allocation domain of the engine that owns it, which is what decides
/// compatibility. Two engines on one ordinal have distinct domains, and a
/// buffer of one must never be submitted through the other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct CudaPlacement {
    pub(super) ordinal: usize,
    pub(super) domain: AllocationDomainId,
}
/// Validates that every operand belongs to the context's engine (its
/// allocation domain, not only its device ordinal), so a buffer from another
/// engine is refused before any submission instead of failing late inside
/// tenferro/CUDA or running against the wrong runtime.
fn ensure_cuda_device(
    ctx_device: CudaPlacement,
    op: &'static str,
    operands: &[(&str, CudaPlacement)],
) -> Result<(), DenseError> {
    for (name, device) in operands {
        if *device != ctx_device {
            return Err(cuda_error(
                op,
                format!(
                    "operand `{name}` belongs to another CUDA engine (device {}, {:?}) than the \
                     supplied one (device {}, {:?})",
                    device.ordinal, device.domain, ctx_device.ordinal, ctx_device.domain
                ),
            ));
        }
    }
    Ok(())
}
fn isize_strides(strides: &[usize]) -> Result<Vec<isize>, DenseError> {
    strides
        .iter()
        .map(|&stride| {
            isize::try_from(stride).map_err(|_| DenseError::StrideOverflow { value: stride })
        })
        .collect()
}
/// Downloads a small real device tensor of the payload's lane `R` as host
/// values, widened to `f64`. Only used for spectra / diagonals / reductions —
/// the sole tensor-shaped data that is allowed to cross the device boundary
/// implicitly (truncation decisions are host scalar logic).
///
/// The lane is the caller's, not `f64` by assumption: a single-precision
/// payload's singular values, eigenvalues and reductions come back as `F32`,
/// and demanding `F64` here would make every such call an error. Widening happens on
/// the host after the transfer, so the bytes moved are the lane's own and
/// symtn's spectrum contract stays `f64`.
fn download_values<R: CudaRealScalar>(
    ctx: &CudaDenseContext,
    tensor: &Tensor,
) -> Result<Vec<f64>, DenseError> {
    let host =
        download_tensor(&ctx.runtime, tensor).map_err(|err| cuda_error("cuda_download", err))?;
    if host.dtype() != R::dtype() {
        return Err(cuda_error(
            "cuda_download",
            format!("expected {:?} values, got {:?}", R::dtype(), host.dtype()),
        ));
    }
    let typed = R::into_typed(host).map_err(|err| cuda_error("cuda_download", err))?;
    let values = typed
        .into_host_vec()
        .map_err(|err| cuda_error("cuda_download", err))?;
    let bytes = std::mem::size_of_val(values.as_slice());
    #[cfg(test)]
    crate::probe::add(crate::probe::Event::CudaMetadataDownloadBytes, bytes);
    record_d2h(bytes);
    Ok(values.into_iter().map(R::widen).collect())
}
