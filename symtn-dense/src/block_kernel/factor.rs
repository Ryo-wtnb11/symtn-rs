//! The native provider's compact SVD of independent blocks: one tlinalg call
//! per block, the blocks spread over the shared scheduler's lanes on the
//! caller's engine.
//!
//! Each block reaches tlinalg exactly as tenferro's per-matrix `svd_read`
//! does (same provider, same validation, same error values), but on its
//! lane's child token instead of the whole engine, so a block's faer width is
//! the engine's decision within its lane's reserved share.

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

/// The `DenseError` op label: the per-matrix entry's.
const ENTRY: &str = "svd_read";
/// tenferro's provider-level op label inside the error payloads.
const OP: &str = tlinalg::Op::Svd.as_str();

/// A scalar the provider factors; its singular values are in
/// [`TensorScalar::Real`].
trait SvdScalar: tlinalg::FaerScalar + TensorScalar {
    /// tlinalg returns the singular values in `Self`; tenferro publishes
    /// their real part.
    fn real_values(values: Vec<Self>) -> Vec<<Self as TensorScalar>::Real>;
}

impl SvdScalar for f32 {
    fn real_values(values: Vec<f32>) -> Vec<f32> {
        values
    }
}

impl SvdScalar for f64 {
    fn real_values(values: Vec<f64>) -> Vec<f64> {
        values
    }
}

impl SvdScalar for Complex32 {
    fn real_values(values: Vec<Complex32>) -> Vec<f32> {
        values.into_iter().map(|value| value.re).collect()
    }
}

impl SvdScalar for Complex64 {
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

/// The compact SVD of each block on the lane driver.
struct SvdBlocks<'a>(PhantomData<fn() -> &'a ()>);

impl<'a> BlockOp for SvdBlocks<'a> {
    type Job = Job<'a>;
    const ENTRY: Entry = Entry::Linalg;
    const KIND: OpKind = OpKind::Factorization;
    const LABEL: &'static str = ENTRY;

    fn work(&self, _: usize, job: &Job<'a>) -> f64 {
        work(job.input)
    }

    fn run(&self, _: usize, job: &mut Job<'a>, mut exec: Exec<'_>) -> Result<(), DenseError> {
        job.factors = Some(factor(job.input, &job.strides, &mut exec)?);
        Ok(())
    }
}

/// Compact SVD of every input, returning `[U, S, Vh]` per input exactly as
/// the per-matrix [`crate::DenseExecutor::svd`] returns it, or the error of
/// the lowest-indexed failing input.
///
/// Checks run in the per-matrix entry's order: the engine kind, then per
/// input the view and the dtype (before the engine is entered), then rank,
/// layout and the factorization (inside it). Inputs above the first input
/// that fails a pre-entry check cannot change the result and are not run.
pub(crate) fn svd_batch(
    engine: &mut dyn tenferro_tensor::Engine,
    inputs: &[DenseRead<'_>],
) -> Result<Vec<Vec<DenseTensor>>, DenseError> {
    with_cpu_engine(engine, |cpu| svd_batch_on(cpu, inputs)).unwrap_or_else(|| {
        Err(tenferro_error(
            ENTRY,
            tenferro_tensor::Error::unsupported(ENTRY, NOT_A_CPU_ENGINE),
        ))
    })
}

fn svd_batch_on(
    cpu: &mut CpuEngine,
    inputs: &[DenseRead<'_>],
) -> Result<Vec<Vec<DenseTensor>>, DenseError> {
    let mut jobs = Vec::with_capacity(inputs.len());
    let mut refused = None;
    for &input in inputs {
        match preflight(input) {
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
        &SvdBlocks(PhantomData),
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

/// tlinalg's SVD work estimate: `m n min(m, n)` multiply-adds per matrix,
/// four times that for a complex scalar, over every batch item.
pub(crate) fn work(input: DenseRead<'_>) -> f64 {
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
    *m as f64 * *n as f64 * (*m).min(*n) as f64 * items * weight
}

/// The checks the per-matrix entry makes before it enters the engine: the
/// tenferro view, then the provider's dtype set.
fn preflight(input: DenseRead<'_>) -> Result<StrideVec, DenseError> {
    let view = tenferro_view(input)?;
    let dtype = view.dtype();
    if !matches!(
        input,
        DenseRead::F32(_) | DenseRead::F64(_) | DenseRead::C32(_) | DenseRead::C64(_)
    ) {
        return Err(tenferro_error(
            ENTRY,
            linalg_error(tenferro_linalg::Error::UnsupportedDType { op: OP, dtype }),
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

fn factor(
    input: DenseRead<'_>,
    strides: &[isize],
    exec: &mut Exec<'_>,
) -> Result<Vec<DenseTensor>, DenseError> {
    match input {
        DenseRead::F32(view) => factor_typed(view, strides, exec),
        DenseRead::F64(view) => factor_typed(view, strides, exec),
        DenseRead::C32(view) => factor_typed(view, strides, exec),
        DenseRead::C64(view) => factor_typed(view, strides, exec),
        _ => unreachable!("preflight admits the provider's dtypes only"),
    }
}

/// tenferro's `svd_impl` on one input: rank check, the empty-extent
/// shortcut, then the provider's batched SVD into compact column-major
/// `[m, k, b..]`, `[k, b..]` and `[k, n, b..]` factors.
fn factor_typed<T: SvdScalar>(
    view: DenseView<'_, T>,
    strides: &[isize],
    exec: &mut Exec<'_>,
) -> Result<Vec<DenseTensor>, DenseError> {
    let shape = view.shape();
    let [m, n, batch @ ..] = shape else {
        return Err(tenferro_error(
            ENTRY,
            tenferro_tensor::Error::rank_mismatch(OP, 2, shape.len()),
        ));
    };
    let (m, n) = (*m, *n);
    let k = m.min(n);
    let with_batch = |dims: &[usize]| [dims, batch].concat();
    let tensor = |dims: &[usize], data| {
        Tensor::from_vec_col_major(with_batch(dims), data)
            .map_err(|err| tenferro_error(ENTRY, err))
            .and_then(DenseTensor::from_tenferro)
    };
    let real = |dims: &[usize], data| {
        Tensor::from_vec_col_major(with_batch(dims), data)
            .map_err(|err| tenferro_error(ENTRY, err))
            .and_then(DenseTensor::from_tenferro)
    };
    if shape.contains(&0) {
        return Ok(vec![
            tensor(&[m, k], Vec::<T>::new())?,
            real(&[k], Vec::<<T as TensorScalar>::Real>::new())?,
            tensor(&[k, n], Vec::<T>::new())?,
        ]);
    }
    // `DenseView` validated the offset against the storage, and the tenferro
    // view (preflight) proved it fits `isize`.
    let offset = isize::try_from(view.offset()).map_err(|_| DenseError::OffsetOverflow {
        value: view.offset(),
    })?;
    let raw = RawStridedRef::new(view.data(), shape, strides, offset).map_err(|err| {
        tenferro_error(
            ENTRY,
            tenferro_tensor::Error::invalid_argument(OP, "layout", err.to_string()),
        )
    })?;
    let (mut u, mut s, mut vt) = (Vec::new(), Vec::new(), Vec::new());
    // The provider's safe `Vec` entry: its workspace query and `svd_into` on
    // one per-call region. Why not `svd_into` with a lane-reused workspace:
    // its destinations are uninitialized and this crate forbids the unsafe
    // code that publishes them.
    tlinalg::svd::svd(tlinalg::Op::Svd, raw, false, &mut u, &mut s, &mut vt, exec)
        .map_err(|err| tenferro_error(ENTRY, provider_error(err)))?;
    Ok(vec![
        tensor(&[m, k], u)?,
        real(&[k], T::real_values(s))?,
        tensor(&[k, n], vt)?,
    ])
}

fn linalg_error(error: tenferro_linalg::Error) -> tenferro_tensor::Error {
    tenferro_tensor::Error::extension(
        OP,
        tenferro_linalg::LINALG_EXTENSION_FAMILY_ID,
        error.kind(),
        error,
    )
}

/// tenferro-linalg's `map_error` for the faer provider (`cpu/tlinalg_error.rs`),
/// so a provider failure is the same `tenferro_tensor::Error` value either
/// route reports.
fn provider_error(error: tlinalg::Error) -> tenferro_tensor::Error {
    use tenferro_linalg::Error as Linalg;
    match error {
        tlinalg::Error::NonConvergence { .. } => linalg_error(Linalg::NonConvergence { op: OP }),
        tlinalg::Error::NonFinite { role, .. } => linalg_error(Linalg::NonFinite {
            op: OP,
            role: role.as_str(),
        }),
        tlinalg::Error::Singular { .. } => linalg_error(Linalg::Singular { op: OP }),
        tlinalg::Error::InvalidArgument { role, detail, .. } => {
            tenferro_tensor::Error::invalid_argument(OP, role, detail)
        }
        // tenferro wraps the same fields in its private `BackendError`, whose
        // message is this variant's; the faer provider never reports it.
        error @ tlinalg::Error::InvalidWorkspace { .. } => {
            tenferro_tensor::Error::backend_source(OP, error)
        }
        tlinalg::Error::Inconsistent { detail, .. } => {
            tenferro_tensor::Error::Internal(format!("{OP}: {detail}"))
        }
        error @ tlinalg::Error::InsufficientWorkspace { .. } => {
            tenferro_tensor::Error::runtime_state_source(OP, error)
        }
        other => tenferro_tensor::Error::Internal(format!(
            "{OP}: unmapped linalg provider error {other:?}"
        )),
    }
}
