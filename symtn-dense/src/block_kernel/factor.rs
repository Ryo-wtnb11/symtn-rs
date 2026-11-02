//! The native provider's compact factorizations of independent blocks (SVD,
//! QR, Hermitian eigendecomposition, their values-only forms and general
//! eigenvalues): one tlinalg call per block, the blocks spread over the lane
//! driver's lanes on the caller's engine.
//!
//! Each block reaches tlinalg exactly as tenferro's per-matrix entry
//! (`svd_read`, `qr_read`, `eigh_read`, `eigvalsh_read`, `svdvals_read`,
//! `eigvals`) does (same provider, same validation, same error values), but
//! on its lane's child token instead of the whole engine, so a block's faer
//! width is the engine's decision within its lane's reserved share. Each factor is a fresh `Vec`
//! from the provider's `Vec` entry: its length is the factor's logical size
//! and it carries no recycled-buffer history (its capacity is the provider's
//! reservation, not necessarily the length).

use std::marker::PhantomData;
use std::time::Duration;

use num_complex::{Complex32, Complex64};
use strided_view::RawStridedRef;
use tcontract_exec::Exec;
use tenferro_cpu::{with_cpu_engine, CpuEngine};
use tenferro_tensor::{StrideVec, Tensor, TensorScalar};

use super::driver::{self, BlockOp, Entry, LaneScratch, OpKind};
use super::schedule::{self, GrantKey};
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

    /// Whether the provider requires square matrices: tenferro checks this
    /// after the rank and before the zero-extent shortcut.
    const SQUARE: bool = false;

    /// tlinalg's work estimates for one `m x n` item of scalar `T`, in
    /// weighted multiply-adds: its total work and the part a team inside the
    /// item can pay back (`estimate_work`, `estimate_payback_work`).
    fn estimates<T: tlinalg::FaerScalar>(m: usize, n: usize) -> [f64; 2];

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

    fn estimates<T: tlinalg::FaerScalar>(m: usize, n: usize) -> [f64; 2] {
        use tlinalg::svd::{estimate_payback_work, estimate_work};
        [estimate_work::<T>(m, n), estimate_payback_work::<T>(m, n)]
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

    fn estimates<T: tlinalg::FaerScalar>(m: usize, n: usize) -> [f64; 2] {
        use tlinalg::qr::{estimate_payback_work, estimate_work};
        [estimate_work::<T>(m, n), estimate_payback_work::<T>(m, n)]
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

/// Hermitian eigendecomposition: `[values, vectors]` of shapes `[n, b..]`
/// (real) and `[n, n, b..]`.
pub(crate) struct Eigh;

impl Factor for Eigh {
    const ENTRY: &'static str = "eigh_read";
    const OP: tlinalg::Op = tlinalg::Op::Eigh;
    const SQUARE: bool = true;

    fn estimates<T: tlinalg::FaerScalar>(n: usize, _: usize) -> [f64; 2] {
        use tlinalg::eigh::{estimate_payback_work, estimate_work};
        [estimate_work::<T>(n), estimate_payback_work::<T>(n)]
    }

    fn factor<T: Scalar>(
        raw: Option<RawStridedRef<'_, T>>,
        [n, _]: [usize; 2],
        batch: &[usize],
        exec: &mut Exec<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        let (mut values, mut vectors) = (Vec::new(), Vec::new());
        if let Some(raw) = raw {
            tlinalg::eigh::eigh(Self::OP, raw, &mut values, &mut vectors, exec)
                .map_err(|err| tenferro_error(Self::ENTRY, provider_error(Self::OP, err)))?;
        }
        Ok(vec![
            output::<Self, <T as TensorScalar>::Real>(&[n], batch, T::real_values(values))?,
            output::<Self, T>(&[n, n], batch, vectors)?,
        ])
    }
}

/// Hermitian eigenvalues only: `[values]` of shape `[n, b..]`, real.
pub(crate) struct EighVals;

impl Factor for EighVals {
    const ENTRY: &'static str = "eigh_values";
    const OP: tlinalg::Op = tlinalg::Op::EighValues;
    const SQUARE: bool = true;

    fn estimates<T: tlinalg::FaerScalar>(n: usize, _: usize) -> [f64; 2] {
        use tlinalg::eigh::{estimate_payback_work, estimate_work};
        [estimate_work::<T>(n), estimate_payback_work::<T>(n)]
    }

    fn factor<T: Scalar>(
        raw: Option<RawStridedRef<'_, T>>,
        [n, _]: [usize; 2],
        batch: &[usize],
        exec: &mut Exec<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        let values = match raw {
            Some(raw) => T::eigh_values(raw, exec)
                .map_err(|err| tenferro_error(Self::ENTRY, provider_error(Self::OP, err)))?,
            None => Vec::new(),
        };
        Ok(vec![output::<Self, <T as TensorScalar>::Real>(
            &[n],
            batch,
            values,
        )?])
    }
}

/// Singular values only: `[S]` of shape `[k, b..]` with `k = min(m, n)`, real.
pub(crate) struct SvdVals;

impl Factor for SvdVals {
    const ENTRY: &'static str = "svd_values";
    const OP: tlinalg::Op = tlinalg::Op::SvdValues;

    fn estimates<T: tlinalg::FaerScalar>(m: usize, n: usize) -> [f64; 2] {
        use tlinalg::svd::{estimate_payback_work, estimate_work};
        [estimate_work::<T>(m, n), estimate_payback_work::<T>(m, n)]
    }

    fn factor<T: Scalar>(
        raw: Option<RawStridedRef<'_, T>>,
        [m, n]: [usize; 2],
        batch: &[usize],
        exec: &mut Exec<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        let s = match raw {
            Some(raw) => T::svd_values(raw, exec)
                .map_err(|err| tenferro_error(Self::ENTRY, provider_error(Self::OP, err)))?,
            None => Vec::new(),
        };
        Ok(vec![output::<Self, <T as TensorScalar>::Real>(
            &[m.min(n)],
            batch,
            s,
        )?])
    }
}

/// General eigenvalues only: `[values]` of shape `[n, b..]` in the input's
/// complex counterpart.
pub(crate) struct EigVals;

impl Factor for EigVals {
    const ENTRY: &'static str = "eig_values";
    const OP: tlinalg::Op = tlinalg::Op::EigValues;
    const SQUARE: bool = true;

    fn estimates<T: tlinalg::FaerScalar>(n: usize, _: usize) -> [f64; 2] {
        use tlinalg::eig::{estimate_payback_work, estimate_work};
        [estimate_work::<T>(n), estimate_payback_work::<T>(n)]
    }

    fn factor<T: Scalar>(
        raw: Option<RawStridedRef<'_, T>>,
        [n, _]: [usize; 2],
        batch: &[usize],
        exec: &mut Exec<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        let mut values = Vec::new();
        if let Some(raw) = raw {
            T::eig_values(raw, &mut values, exec)
                .map_err(|err| tenferro_error(Self::ENTRY, provider_error(Self::OP, err)))?;
        }
        Ok(vec![output::<Self, T::EigScalar>(&[n], batch, values)?])
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

/// A scalar the provider factors; its singular values and eigenvalues are in
/// [`TensorScalar::Real`].
pub(crate) trait Scalar: tlinalg::FaerScalar + TensorScalar {
    /// tlinalg returns the singular values and eigenvalues of a full
    /// factorization in `Self`; tenferro publishes their real part.
    fn real_values(values: Vec<Self>) -> Vec<<Self as TensorScalar>::Real>;

    /// tlinalg's Hermitian eigenvalues only, ascending.
    fn eigh_values(
        raw: RawStridedRef<'_, Self>,
        exec: &mut Exec<'_>,
    ) -> tlinalg::Result<Vec<<Self as TensorScalar>::Real>>;

    /// tlinalg's singular values only, descending.
    fn svd_values(
        raw: RawStridedRef<'_, Self>,
        exec: &mut Exec<'_>,
    ) -> tlinalg::Result<Vec<<Self as TensorScalar>::Real>>;

    /// The complex counterpart, in which the general eigenvalues are
    /// published.
    type EigScalar: TensorScalar;

    /// tlinalg's general eigenvalues only, in its order.
    fn eig_values(
        raw: RawStridedRef<'_, Self>,
        values: &mut Vec<Self::EigScalar>,
        exec: &mut Exec<'_>,
    ) -> tlinalg::Result<()>;
}

// Why one impl per concrete scalar: tlinalg's values-only real type is an
// associated type of a trait it does not export, so it equals
// `TensorScalar::Real` only once the scalar is concrete.
macro_rules! scalar {
    ($ty:ty, $complex:ty, |$values:ident| $real:expr) => {
        impl Scalar for $ty {
            type EigScalar = $complex;

            fn real_values($values: Vec<$ty>) -> Vec<<$ty as TensorScalar>::Real> {
                $real
            }

            fn eigh_values(
                raw: RawStridedRef<'_, $ty>,
                exec: &mut Exec<'_>,
            ) -> tlinalg::Result<Vec<<$ty as TensorScalar>::Real>> {
                let mut values = Vec::new();
                tlinalg::eigh::eigh_values(tlinalg::Op::EighValues, raw, &mut values, exec)?;
                Ok(values)
            }

            fn svd_values(
                raw: RawStridedRef<'_, $ty>,
                exec: &mut Exec<'_>,
            ) -> tlinalg::Result<Vec<<$ty as TensorScalar>::Real>> {
                let mut values = Vec::new();
                tlinalg::svd::svd_values(tlinalg::Op::SvdValues, raw, &mut values, exec)?;
                Ok(values)
            }

            fn eig_values(
                raw: RawStridedRef<'_, $ty>,
                values: &mut Vec<$complex>,
                exec: &mut Exec<'_>,
            ) -> tlinalg::Result<()> {
                tlinalg::eig::eig_values(tlinalg::Op::EigValues, raw, values, exec)
            }
        }
    };
}

scalar!(f32, Complex32, |values| values);
scalar!(f64, Complex64, |values| values);
scalar!(Complex32, Complex32, |values| values
    .into_iter()
    .map(|value| value.re)
    .collect());
scalar!(Complex64, Complex64, |values| values
    .into_iter()
    .map(|value| value.re)
    .collect());

/// One block: its validated input, its signed strides and, once run, its
/// own outcome (factors or that block's error).
pub(crate) struct Job<'a> {
    input: DenseRead<'a>,
    strides: StrideVec,
    outcome: Option<Result<Vec<DenseTensor>, DenseError>>,
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

    fn modeled(&self, _: usize, job: &Job<'a>) -> bool {
        single_item::<F>(job.input).is_some()
    }

    fn grant(&self, _: usize, job: &Job<'a>, exec: &Exec<'_>, budget: usize, _: Duration) -> usize {
        single_item::<F>(job.input).map_or(1, |item| grant(exec, budget, item))
    }

    fn grant_key(&self, _: usize, job: &Job<'a>) -> GrantKey {
        single_item::<F>(job.input).map_or([0; 2], |item| item.map(f64::to_bits))
    }

    // A block's error is its own outcome, not the driver's: the caller
    // delivers every block's outcome in input order.
    fn run(&self, _: usize, job: &mut Job<'a>, mut exec: Exec<'_>) -> Result<(), DenseError> {
        job.outcome = Some(factor::<F>(job.input, &job.strides, &mut exec));
        Ok(())
    }
}

/// The batch's outcome sink (see [`crate::DenseExecutor::factorize_batch`]).
pub(crate) type Sink<'s> =
    dyn FnMut(usize, Result<Vec<DenseTensor>, DenseError>) -> Result<(), DenseError> + 's;

/// Factorization `F` of every input, delivering each input's outcome to
/// `sink` in input order exactly as the per-matrix entry
/// ([`crate::DenseExecutor::svd`], `qr`, `eigh`, `eigh_vals`, `svd_vals`,
/// `eig_vals`)
/// returns it.
///
/// Checks run in the per-matrix entry's order: the engine kind, then per
/// input the view and the dtype (before the engine is entered), then rank,
/// layout and the factorization (inside it). The first input that fails a
/// pre-entry check is delivered after every input before it; inputs above it
/// are not run. A refused engine entry is returned with no outcome
/// delivered.
pub(crate) fn factorize_batch<F: Factor>(
    engine: &mut dyn tenferro_tensor::Engine,
    inputs: &[DenseRead<'_>],
    sink: &mut Sink<'_>,
) -> Result<(), DenseError> {
    with_cpu_engine(engine, |cpu| batch_on::<F>(cpu, inputs, sink)).unwrap_or_else(|| {
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
    sink: &mut Sink<'_>,
) -> Result<(), DenseError> {
    use crate::DenseFactorization as Op;
    match op {
        Op::Svd => factorize_batch::<Svd>(engine, inputs, sink),
        Op::Qr => factorize_batch::<Qr>(engine, inputs, sink),
        Op::Eigh => factorize_batch::<Eigh>(engine, inputs, sink),
        Op::EighVals => factorize_batch::<EighVals>(engine, inputs, sink),
        Op::SvdVals => factorize_batch::<SvdVals>(engine, inputs, sink),
        // Why not on the lanes: tenferro's eig publishes unit-norm
        // eigenvectors through a normalization it does not export
        // (tensor4all/tenferro-rs#2387), and symtn does not duplicate it, so
        // each input takes the per-matrix route on the whole engine and is
        // delivered before the next is solved.
        Op::Eig => inputs.iter().enumerate().try_for_each(|(index, &input)| {
            sink(index, crate::tenferro_adapter::eig_read(engine, input))
        }),
        Op::EigVals => factorize_batch::<EigVals>(engine, inputs, sink),
    }
}

fn batch_on<F: Factor>(
    cpu: &mut CpuEngine,
    inputs: &[DenseRead<'_>],
    sink: &mut Sink<'_>,
) -> Result<(), DenseError> {
    let mut jobs = Vec::with_capacity(inputs.len());
    let mut refused = None;
    for &input in inputs {
        match preflight::<F>(input) {
            Ok(strides) => jobs.push(Job {
                input,
                strides,
                outcome: None,
            }),
            Err(error) => {
                refused = Some(error);
                break;
            }
        }
    }
    if !jobs.is_empty() {
        driver::drive(
            cpu,
            &Blocks::<F>(PhantomData),
            &mut jobs,
            &mut LaneScratch::default(),
        )?;
    }
    let admitted = jobs.len();
    for (index, job) in jobs.into_iter().enumerate() {
        sink(
            index,
            job.outcome
                .unwrap_or_else(|| unreachable!("the driver runs every admitted job")),
        )?;
    }
    match refused {
        Some(error) => sink(admitted, Err(error)),
        None => Ok(()),
    }
}

/// tlinalg's per-item estimates of `F` for `input`'s scalar (see
/// [`Factor::estimates`]) and its batch product, or `None` for an input the
/// provider refuses (rank or dtype).
fn item_estimates<F: Factor>(input: DenseRead<'_>) -> Option<([f64; 2], f64)> {
    let (shape, estimates): (_, fn(usize, usize) -> [f64; 2]) = match input {
        DenseRead::F32(view) => (view.shape(), F::estimates::<f32>),
        DenseRead::F64(view) => (view.shape(), F::estimates::<f64>),
        DenseRead::C32(view) => (view.shape(), F::estimates::<Complex32>),
        DenseRead::C64(view) => (view.shape(), F::estimates::<Complex64>),
        _ => return None,
    };
    let [m, n, batch @ ..] = shape else {
        return None;
    };
    Some((estimates(*m, *n), batch.iter().map(|&b| b as f64).product()))
}

/// tlinalg's work estimate of `F` over every batch item of `input`, in
/// weighted multiply-adds (four per complex multiply-add).
pub(crate) fn work<F: Factor>(input: DenseRead<'_>) -> f64 {
    item_estimates::<F>(input).map_or(0.0, |([total, _], items)| total * items)
}

/// tlinalg's (total, payback) estimates of `F` for `input` when its batch
/// product is one, the only inputs whose provider width [`grant`] composes.
///
/// Why not a batched input: tlinalg's route for several items first gates the
/// whole batch, then splits it into lanes and caps each item at
/// `ceil(budget / lanes)` (`batch::decide`), a decision its public queries do
/// not expose; a batch holding such an input keeps the baseline placement.
pub(crate) fn single_item<F: Factor>(input: DenseRead<'_>) -> Option<[f64; 2]> {
    item_estimates::<F>(input).and_then(|(item, items)| (items == 1.0).then_some(item))
}

/// The width tlinalg gives one item of `estimates = [total, payback]`
/// weighted multiply-adds on a lane of `budget` slots of `exec` under Auto:
/// `batch::decide` for a batch of one takes the batch decision's item width
/// (`width_for` over the total's time, one section) and caps it at the width
/// the payback work's time pays for, both under the engine's work model; a
/// token without a team runs the item serially.
pub(crate) fn grant(exec: &Exec<'_>, budget: usize, [total, payback]: [f64; 2]) -> usize {
    if !exec.token_ref().carries_team() {
        return 1;
    }
    let model = &exec.batch_policy().model;
    let width = |work| exec.width_for_within(budget, schedule::work_model_ns(model, work), 1);
    width(total).min(width(payback))
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

/// tenferro's `svd_impl`/`qr_impl`/`eigh_impl`/`eig_impl` (values-only forms) on
/// one input: rank check, the square check where the provider requires one,
/// the empty-extent shortcut, then the provider's batched factorization.
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
    if F::SQUARE && m != n {
        return Err(tenferro_error(
            F::ENTRY,
            tenferro_tensor::Error::shape_mismatch(op, vec![*m], vec![*n]),
        ));
    }
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

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use tcontract_exec::{BatchPolicy, WorkModel};
    use tenferro_cpu::Pool;

    use super::*;

    /// The SVD kernel's own op with recording-only execution: cost, search
    /// admission and grant are `Blocks<Svd>`'s, and each block records the
    /// reserved width of the lane it runs on instead of factoring.
    struct RecordsWidth<'a>(Blocks<'a, Svd>, Vec<AtomicUsize>);

    impl<'a> BlockOp for RecordsWidth<'a> {
        type Job = Job<'a>;
        const ENTRY: Entry = Entry::Linalg;
        const KIND: OpKind = OpKind::Factorization;
        const LABEL: &'static str = Svd::ENTRY;

        fn work(&self, index: usize, job: &Job<'a>) -> f64 {
            self.0.work(index, job)
        }

        fn modeled(&self, index: usize, job: &Job<'a>) -> bool {
            self.0.modeled(index, job)
        }

        fn grant(
            &self,
            index: usize,
            job: &Job<'a>,
            exec: &Exec<'_>,
            budget: usize,
            serial: Duration,
        ) -> usize {
            self.0.grant(index, job, exec, budget, serial)
        }

        fn grant_key(&self, index: usize, job: &Job<'a>) -> GrantKey {
            self.0.grant_key(index, job)
        }

        fn run(&self, index: usize, _: &mut Job<'a>, exec: Exec<'_>) -> Result<(), DenseError> {
            self.1[index].store(exec.budget(), Ordering::Relaxed);
            Ok(())
        }
    }

    /// The reserved lane width each real SVD input of `shapes` runs on, on
    /// `threads` slots of flat 100 ns dispatch at one multiply-add per ns, no
    /// item overhead and a lane work minimum of `lane_ns`.
    fn svd_lane_widths(threads: usize, lane_ns: f64, shapes: &[&[usize]]) -> Vec<usize> {
        use tenferro_cpu::threadteam::{DispatchCosts, WidthCosts};
        let row = WidthCosts::flat(Duration::from_nanos(100));
        let pool = Pool::builder()
            .threads(threads)
            .dispatch_costs(DispatchCosts::new(row, row, row))
            .build()
            .unwrap();
        let model = WorkModel::default()
            .with_muladds_per_ns(1.0)
            .with_item_overhead_ns(0.0)
            .with_min_work_per_lane_ns(lane_ns);
        let mut engine =
            CpuEngine::new(pool).with_batch_policy(BatchPolicy::default().with_model(model));
        let data = vec![0.0_f64; 10000];
        // Column-major strides: each the product of the extents before it.
        let strides: Vec<Vec<usize>> = shapes
            .iter()
            .map(|shape| {
                (0..shape.len())
                    .map(|axis| shape[..axis].iter().product())
                    .collect()
            })
            .collect();
        let mut jobs: Vec<Job<'_>> = shapes
            .iter()
            .zip(&strides)
            .map(|(&shape, strides)| {
                let len = shape.iter().product();
                let input =
                    DenseRead::F64(DenseView::new(&data[..len], shape, strides, 0).unwrap());
                Job {
                    input,
                    strides: preflight::<Svd>(input).unwrap(),
                    outcome: None,
                }
            })
            .collect();
        let op = RecordsWidth(
            Blocks(PhantomData),
            shapes.iter().map(|_| 0.into()).collect(),
        );
        driver::drive(&mut engine, &op, &mut jobs, &mut LaneScratch::default()).unwrap();
        op.1.iter().map(|w| w.load(Ordering::Relaxed)).collect()
    }

    /// Purpose: a factorization batch whose every input is one matrix joins
    /// the lane search with tlinalg's own width for one SVD item, and a batch
    /// holding an input whose batch product is not one (including zero) keeps
    /// the baseline placement. Four slots, flat 100 ns dispatch (a width pays
    /// from 300 ns), one multiply-add per ns:
    /// - `80 x 80` and three `4 x 4`: the large SVD's payback work is
    ///   `80^3 * 6e-4 = 307.2` ns, so it is granted every width; one
    ///   four-slot lane scores `512000 / 4 + 100 + 3 * 64 = 128292` ns
    ///   against the baseline's 512000 ns on four one-slot lanes;
    /// - `79 x 79` instead: payback `79^3 * 6e-4 = 295.8` ns pays at no width
    ///   (its whole time, 493039 ns, would), so every grant is one, no
    ///   candidate scores below the baseline's 493039 ns, and the tie keeps
    ///   the baseline;
    /// - the first batch with one `4 x 4 x 2` input keeps the baseline.
    ///
    /// On six slots with a 10000 ns lane minimum, one `10000 x 1` and 120
    /// `100 x 1` (payback 6 and 0.06 ns: serial at every width) give `L0 = 2`
    /// lanes; the baseline and two equal lanes score 11000 ns, the
    /// reservation's six one-slot lanes 10000 ns but hold more than `L0`
    /// lanes and are discarded, so the baseline's two three-slot lanes run
    /// every block.
    #[test]
    fn single_matrix_factorizations_join_the_lane_search() {
        let small: &[usize] = &[4, 4];
        assert_eq!(
            svd_lane_widths(4, 0.0, &[&[80, 80], small, small, small]),
            [4; 4]
        );
        assert_eq!(
            svd_lane_widths(4, 0.0, &[&[79, 79], small, small, small]),
            [1; 4]
        );
        assert_eq!(
            svd_lane_widths(4, 0.0, &[&[80, 80], small, small, &[4, 4, 2]]),
            [1; 4]
        );
        let mut many: Vec<&[usize]> = vec![&[10000, 1]];
        many.extend([&[100, 1][..]; 120]);
        assert_eq!(svd_lane_widths(6, 10000.0, &many), [3; 121]);
    }
}
