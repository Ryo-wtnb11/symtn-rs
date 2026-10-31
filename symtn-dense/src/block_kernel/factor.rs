//! The native provider's compact factorizations of independent blocks (SVD
//! and QR): one tlinalg call per block, the blocks spread over the lane
//! driver's lanes on the caller's engine.
//!
//! Each block reaches tlinalg exactly as tenferro's per-matrix `svd_read` or
//! `qr_read` does (same provider, same validation, same error values), but on
//! its lane's child token instead of the whole engine, so a block's faer
//! width is the engine's decision within its lane's reserved share. Each
//! factor is allocated at its exact size by the provider's `Vec` entry.

use std::marker::PhantomData;

use num_complex::{Complex32, Complex64};
use strided_view::RawStridedRef;
use tcontract_exec::Exec;
use tenferro_cpu::{with_cpu_engine, CpuEngine};
use tenferro_tensor::{StrideVec, Tensor, TensorScalar};

use super::driver::{self, BlockOp, Entry, LaneScratch, OpKind};
use crate::layout::strides_to_isize;
use crate::tenferro_adapter::{tenferro_error, tenferro_view, NOT_A_CPU_ENGINE};
use crate::{DenseError, DenseRead, DenseTensor, DenseView};

/// One compact factorization the provider runs per block. The shared code
/// owns validation, costing and scheduling; an implementation owns only its
/// provider call and its factors' shapes.
pub(crate) trait Factor {
    /// The per-matrix entry's `DenseError` label.
    const ENTRY: &'static str;
    /// The provider op, whose name labels the error payloads.
    const OP: tlinalg::Op;

    /// tlinalg's work estimate for one real `m x n` matrix, in multiply-adds.
    fn work(m: f64, n: f64) -> f64;

    /// The factors of one input of matrix extents `dims = [m, n]` and batch
    /// dimensions `batch`, from the provider, or empty ones when `raw` is
    /// `None` (a zero extent).
    fn factor<T: Scalar>(
        raw: Option<RawStridedRef<'_, T>>,
        dims: [usize; 2],
        batch: &[usize],
        exec: &mut Exec<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError>;
}

/// Compact SVD: `[U, S, Vh]` of shapes `[m, k, b..]`, `[k, b..]`,
/// `[k, n, b..]` with `k = min(m, n)`; `S` is real.
pub(crate) struct Svd;

impl Factor for Svd {
    const ENTRY: &'static str = "svd_read";
    const OP: tlinalg::Op = tlinalg::Op::Svd;

    fn work(m: f64, n: f64) -> f64 {
        m * n * m.min(n)
    }

    fn factor<T: Scalar>(
        raw: Option<RawStridedRef<'_, T>>,
        [m, n]: [usize; 2],
        batch: &[usize],
        exec: &mut Exec<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        let k = m.min(n);
        let (mut u, mut s, mut vt) = (Vec::new(), Vec::new(), Vec::new());
        if let Some(raw) = raw {
            tlinalg::svd::svd(Self::OP, raw, false, &mut u, &mut s, &mut vt, exec)
                .map_err(|err| tenferro_error(Self::ENTRY, provider_error(Self::OP, err)))?;
        }
        Ok(vec![
            output::<Self, T>(&[m, k], batch, u)?,
            output::<Self, <T as TensorScalar>::Real>(&[k], batch, T::real_values(s))?,
            output::<Self, T>(&[k, n], batch, vt)?,
        ])
    }
}

/// Compact QR: `[Q, R]` of shapes `[m, k, b..]`, `[k, n, b..]` with
/// `k = min(m, n)`, in the provider's gauge.
pub(crate) struct Qr;

impl Factor for Qr {
    const ENTRY: &'static str = "qr_read";
    const OP: tlinalg::Op = tlinalg::Op::Qr;

    fn work(m: f64, n: f64) -> f64 {
        let k = m.min(n);
        m * n * k - k.powi(3) / 3.0
    }

    fn factor<T: Scalar>(
        raw: Option<RawStridedRef<'_, T>>,
        [m, n]: [usize; 2],
        batch: &[usize],
        exec: &mut Exec<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        let k = m.min(n);
        let (mut q, mut r) = (Vec::new(), Vec::new());
        if let Some(raw) = raw {
            tlinalg::qr::qr(Self::OP, raw, &mut q, &mut r, exec)
                .map_err(|err| tenferro_error(Self::ENTRY, provider_error(Self::OP, err)))?;
        }
        Ok(vec![
            output::<Self, T>(&[m, k], batch, q)?,
            output::<Self, T>(&[k, n], batch, r)?,
        ])
    }
}

/// A column-major factor of shape `dims` followed by `batch`.
fn output<F: Factor, T: TensorScalar>(
    dims: &[usize],
    batch: &[usize],
    data: Vec<T>,
) -> Result<DenseTensor, DenseError> {
    Tensor::from_vec_col_major([dims, batch].concat(), data)
        .map_err(|err| tenferro_error(F::ENTRY, err))
        .and_then(DenseTensor::from_tenferro)
}

/// A scalar the provider factors; its singular values are in
/// [`TensorScalar::Real`].
pub(crate) trait Scalar: tlinalg::FaerScalar + TensorScalar {
    /// tlinalg returns the singular values in `Self`; tenferro publishes
    /// their real part.
    fn real_values(values: Vec<Self>) -> Vec<<Self as TensorScalar>::Real>;
}

impl Scalar for f32 {
    fn real_values(values: Vec<f32>) -> Vec<f32> {
        values
    }
}

impl Scalar for f64 {
    fn real_values(values: Vec<f64>) -> Vec<f64> {
        values
    }
}

impl Scalar for Complex32 {
    fn real_values(values: Vec<Complex32>) -> Vec<f32> {
        values.into_iter().map(|value| value.re).collect()
    }
}

impl Scalar for Complex64 {
    fn real_values(values: Vec<Complex64>) -> Vec<f64> {
        values.into_iter().map(|value| value.re).collect()
    }
}

/// One block: its validated input, its signed strides and, once run, its
/// factors.
pub(crate) struct Job<'a> {
    input: DenseRead<'a>,
    strides: StrideVec,
    factors: Option<Vec<DenseTensor>>,
}

/// Factorization `F` of each block on the lane driver.
struct Blocks<'a, F>(PhantomData<fn() -> (&'a (), F)>);

impl<'a, F: Factor> BlockOp for Blocks<'a, F> {
    type Job = Job<'a>;
    const ENTRY: Entry = Entry::Linalg;
    const KIND: OpKind = OpKind::Factorization;
    const LABEL: &'static str = F::ENTRY;

    fn work(&self, _: usize, job: &Job<'a>) -> f64 {
        work::<F>(job.input)
    }

    fn run(&self, _: usize, job: &mut Job<'a>, mut exec: Exec<'_>) -> Result<(), DenseError> {
        job.factors = Some(factor::<F>(job.input, &job.strides, &mut exec)?);
        Ok(())
    }
}

/// Factorization `F` of every input, returning the factors per input
/// exactly as the per-matrix entry ([`crate::DenseExecutor::svd`] or
/// [`crate::DenseExecutor::qr`]) returns them, or the error of the
/// lowest-indexed failing input.
///
/// Checks run in the per-matrix entry's order: the engine kind, then per
/// input the view and the dtype (before the engine is entered), then rank,
/// layout and the factorization (inside it). Inputs above the first input
/// that fails a pre-entry check cannot change the result and are not run.
pub(crate) fn factorize_batch<F: Factor>(
    engine: &mut dyn tenferro_tensor::Engine,
    inputs: &[DenseRead<'_>],
) -> Result<Vec<Vec<DenseTensor>>, DenseError> {
    with_cpu_engine(engine, |cpu| batch_on::<F>(cpu, inputs)).unwrap_or_else(|| {
        Err(tenferro_error(
            F::ENTRY,
            tenferro_tensor::Error::unsupported(F::ENTRY, NOT_A_CPU_ENGINE),
        ))
    })
}

/// [`factorize_batch`] of the factorization `op` names.
pub(crate) fn factorize(
    engine: &mut dyn tenferro_tensor::Engine,
    op: crate::DenseFactorization,
    inputs: &[DenseRead<'_>],
) -> Result<Vec<Vec<DenseTensor>>, DenseError> {
    match op {
        crate::DenseFactorization::Svd => factorize_batch::<Svd>(engine, inputs),
        crate::DenseFactorization::Qr => factorize_batch::<Qr>(engine, inputs),
    }
}

fn batch_on<F: Factor>(
    cpu: &mut CpuEngine,
    inputs: &[DenseRead<'_>],
) -> Result<Vec<Vec<DenseTensor>>, DenseError> {
    let mut jobs = Vec::with_capacity(inputs.len());
    let mut refused = None;
    for &input in inputs {
        match preflight::<F>(input) {
            Ok(strides) => jobs.push(Job {
                input,
                strides,
                factors: None,
            }),
            Err(error) => {
                refused = Some(error);
                break;
            }
        }
    }
    if jobs.is_empty() {
        return refused.map_or(Ok(Vec::new()), Err);
    }
    driver::drive(
        cpu,
        &Blocks::<F>(PhantomData),
        &mut jobs,
        &mut LaneScratch::default(),
    )?;
    if let Some(error) = refused {
        return Err(error);
    }
    Ok(jobs
        .into_iter()
        .map(|job| {
            job.factors
                .unwrap_or_else(|| unreachable!("every job ran without failing"))
        })
        .collect())
}

/// tlinalg's work estimate of `F` per matrix, four times that for a complex
/// scalar, over every batch item.
pub(crate) fn work<F: Factor>(input: DenseRead<'_>) -> f64 {
    let (shape, complex) = match input {
        DenseRead::F32(view) => (view.shape(), false),
        DenseRead::F64(view) => (view.shape(), false),
        DenseRead::C32(view) => (view.shape(), true),
        DenseRead::C64(view) => (view.shape(), true),
        _ => return 0.0,
    };
    let [m, n, batch @ ..] = shape else {
        return 0.0;
    };
    let items: f64 = batch.iter().map(|&b| b as f64).product();
    let weight = if complex { 4.0 } else { 1.0 };
    F::work(*m as f64, *n as f64) * items * weight
}

/// The checks the per-matrix entry makes before it enters the engine: the
/// tenferro view, then the provider's dtype set.
fn preflight<F: Factor>(input: DenseRead<'_>) -> Result<StrideVec, DenseError> {
    let view = tenferro_view(input)?;
    let dtype = view.dtype();
    if !matches!(
        input,
        DenseRead::F32(_) | DenseRead::F64(_) | DenseRead::C32(_) | DenseRead::C64(_)
    ) {
        let op = F::OP.as_str();
        return Err(tenferro_error(
            F::ENTRY,
            linalg_error(op, tenferro_linalg::Error::UnsupportedDType { op, dtype }),
        ));
    }
    strides_to_isize(view_strides(input))
}

fn view_strides(input: DenseRead<'_>) -> &[usize] {
    match input {
        DenseRead::F32(view) => view.strides(),
        DenseRead::F64(view) => view.strides(),
        DenseRead::I32(view) => view.strides(),
        DenseRead::I64(view) => view.strides(),
        DenseRead::Bool(view) => view.strides(),
        DenseRead::C32(view) => view.strides(),
        DenseRead::C64(view) => view.strides(),
    }
}

fn factor<F: Factor>(
    input: DenseRead<'_>,
    strides: &[isize],
    exec: &mut Exec<'_>,
) -> Result<Vec<DenseTensor>, DenseError> {
    match input {
        DenseRead::F32(view) => factor_typed::<F, _>(view, strides, exec),
        DenseRead::F64(view) => factor_typed::<F, _>(view, strides, exec),
        DenseRead::C32(view) => factor_typed::<F, _>(view, strides, exec),
        DenseRead::C64(view) => factor_typed::<F, _>(view, strides, exec),
        _ => unreachable!("preflight admits the provider's dtypes only"),
    }
}

/// tenferro's `svd_impl`/`qr_impl` on one input: rank check, the
/// empty-extent shortcut, then the provider's batched factorization.
fn factor_typed<F: Factor, T: Scalar>(
    view: DenseView<'_, T>,
    strides: &[isize],
    exec: &mut Exec<'_>,
) -> Result<Vec<DenseTensor>, DenseError> {
    let op = F::OP.as_str();
    let shape = view.shape();
    let [m, n, batch @ ..] = shape else {
        return Err(tenferro_error(
            F::ENTRY,
            tenferro_tensor::Error::rank_mismatch(op, 2, shape.len()),
        ));
    };
    let dims = [*m, *n];
    if shape.contains(&0) {
        return F::factor::<T>(None, dims, batch, exec);
    }
    // `DenseView` validated the offset against the storage, and the tenferro
    // view (preflight) proved it fits `isize`.
    let offset = isize::try_from(view.offset()).map_err(|_| DenseError::OffsetOverflow {
        value: view.offset(),
    })?;
    let raw = RawStridedRef::new(view.data(), shape, strides, offset).map_err(|err| {
        tenferro_error(
            F::ENTRY,
            tenferro_tensor::Error::invalid_argument(op, "layout", err.to_string()),
        )
    })?;
    // The provider's safe `Vec` entry: its workspace query and `*_into` on
    // one per-call region. Why not `*_into` with a lane-reused workspace:
    // its destinations are uninitialized and this crate forbids the unsafe
    // code that publishes them.
    F::factor(Some(raw), dims, batch, exec)
}

fn linalg_error(op: &'static str, error: tenferro_linalg::Error) -> tenferro_tensor::Error {
    tenferro_tensor::Error::extension(
        op,
        tenferro_linalg::LINALG_EXTENSION_FAMILY_ID,
        error.kind(),
        error,
    )
}

/// tenferro-linalg's `map_error` for the faer provider (`cpu/tlinalg_error.rs`),
/// so a provider failure is the same `tenferro_tensor::Error` value either
/// route reports.
fn provider_error(op: tlinalg::Op, error: tlinalg::Error) -> tenferro_tensor::Error {
    use tenferro_linalg::Error as Linalg;
    let op = op.as_str();
    match error {
        tlinalg::Error::NonConvergence { .. } => linalg_error(op, Linalg::NonConvergence { op }),
        tlinalg::Error::NonFinite { role, .. } => linalg_error(
            op,
            Linalg::NonFinite {
                op,
                role: role.as_str(),
            },
        ),
        tlinalg::Error::Singular { .. } => linalg_error(op, Linalg::Singular { op }),
        tlinalg::Error::InvalidArgument { role, detail, .. } => {
            tenferro_tensor::Error::invalid_argument(op, role, detail)
        }
        // tenferro wraps the same fields in its private `BackendError`, whose
        // message is this variant's; the faer provider never reports it.
        error @ tlinalg::Error::InvalidWorkspace { .. } => {
            tenferro_tensor::Error::backend_source(op, error)
        }
        tlinalg::Error::Inconsistent { detail, .. } => {
            tenferro_tensor::Error::Internal(format!("{op}: {detail}"))
        }
        error @ tlinalg::Error::InsufficientWorkspace { .. } => {
            tenferro_tensor::Error::runtime_state_source(op, error)
        }
        other => tenferro_tensor::Error::Internal(format!(
            "{op}: unmapped linalg provider error {other:?}"
        )),
    }
}
