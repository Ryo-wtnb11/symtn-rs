//! The native provider's compact factorizations of independent blocks (SVD,
//! QR, Hermitian eigendecomposition, their values-only forms and general
//! eigenvalues): one tlinalg call per block, the blocks spread over the lane
//! driver's lanes on the caller's engine.
//!
//! Each block reaches tlinalg exactly as tenferro's per-matrix entry
//! (`svd_read`, `qr_read`, `eigh_read`, `eigvalsh_read`, `svdvals_read`,
//! `eigvals`) does (same provider, same validation, same error values), but
//! on its lane's child token instead of the whole engine, so a block's faer
//! width is the engine's decision within its lane's reserved share.
//!
//! The owned batch ([`factorize`]) returns each factor as a fresh `Vec` from
//! the provider's `Vec` entry: its length is the factor's logical size and it
//! carries no recycled-buffer history (its capacity is the provider's
//! reservation, not necessarily the length). The destination batch
//! ([`factorize_into`], compact QR, compact SVD and Hermitian eigh) writes
//! each block's factors into its caller-owned destination through the
//! provider's initialized-destination entry (`*_into_init`), with one
//! workspace and one typed scratch per lane, sized before the lane's first
//! block from that lane's blocks and child context and dropped when the lane
//! finishes.

use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::time::Duration;

use num_complex::{Complex32, Complex64};
use strided_view::RawStridedRef;
use tcontract_exec::Exec;
use tenferro_cpu::{with_cpu_engine, CpuEngine};
use tenferro_tensor::{StrideVec, Tensor, TensorScalar};

use super::driver::{self, BlockOp, Entry, LaneScratch, OpKind};
use super::schedule::{self, GrantKey, LaneJobs};
use crate::factor_dest::{adjoint_items, compact_mut, Element, FactorSink};
use crate::layout::strides_to_isize;
use crate::tenferro_adapter::{tenferro_error, tenferro_view, NOT_A_CPU_ENGINE};
use crate::LaneAllocation;
use crate::{
    DenseError, DenseRead, DenseTensor, DenseView, DenseWrite, FactorDest, FactorFailure,
    FactorPlacement,
};

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

    /// Whether a batch of single-matrix inputs joins the lane search with
    /// [`grant`]; otherwise its batches keep the baseline placement.
    const SEARCHED: bool = true;

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
    // Why not searched: the item model credits a granted width with a linear
    // speedup, and QR's payback share lets the search hand few large QR blocks
    // wide lanes that faer's QR does not use (tlinalg's own measured `s(8)`:
    // about 1.03 at 192 x 192, 2.16 at 512 x 512). On lg the searched
    // placement ran 1.14-1.44x slower than the baseline on few-sector batches
    // of 192-1024 blocks at 16 and 48 threads. Searching QR needs the
    // provider's time-at-width estimates (tensor4all/tlinalg-rs#88).
    const SEARCHED: bool = false;

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
pub(crate) trait Scalar: tlinalg::FaerScalar + TensorScalar + Element {
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

    /// Whether tlinalg's `Self`-typed spectrum differs from the published
    /// real one, so a destination batch stages it in typed scratch.
    const COMPLEX: bool;

    /// The lane's typed scratch of this scalar.
    fn scratch(lane: &mut TypedScratch) -> &mut [Self];

    /// The compact payload of a destination of this scalar, `None` for
    /// another dtype.
    fn factor_slot<'b>(write: &'b mut DenseWrite<'_>) -> Option<&'b mut [Self]>;

    /// The compact payload of a spectrum destination (the real
    /// counterpart), `None` for another dtype.
    fn spectrum_slot<'b>(
        write: &'b mut DenseWrite<'_>,
    ) -> Option<&'b mut [<Self as TensorScalar>::Real]>;

    /// Runs `solve` with a `Self` spectrum slot of `dest.len()` elements:
    /// `dest` itself for a real scalar; otherwise the front of `scratch`,
    /// whose real parts are copied into `dest` after `solve` succeeds.
    fn with_spectrum<R>(
        dest: &mut [<Self as TensorScalar>::Real],
        scratch: &mut [Self],
        solve: impl FnOnce(&mut [Self]) -> tlinalg::Result<R>,
    ) -> tlinalg::Result<R>;

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
    ($ty:ty, $complex:ty, $variant:ident, $real_variant:ident, $slot:ident, $is_complex:expr,
     |$values:ident| $real:expr, |$dest:ident, $scratch:ident, $solve:ident| $spectrum:expr) => {
        impl Scalar for $ty {
            type EigScalar = $complex;
            const COMPLEX: bool = $is_complex;

            fn scratch(lane: &mut TypedScratch) -> &mut [Self] {
                &mut lane.$slot
            }

            fn factor_slot<'b>(write: &'b mut DenseWrite<'_>) -> Option<&'b mut [Self]> {
                match write {
                    DenseWrite::$variant(view) => Some(compact_mut(view)),
                    _ => None,
                }
            }

            fn spectrum_slot<'b>(
                write: &'b mut DenseWrite<'_>,
            ) -> Option<&'b mut [<Self as TensorScalar>::Real]> {
                match write {
                    DenseWrite::$real_variant(view) => Some(compact_mut(view)),
                    _ => None,
                }
            }

            fn with_spectrum<R>(
                $dest: &mut [<Self as TensorScalar>::Real],
                $scratch: &mut [Self],
                $solve: impl FnOnce(&mut [Self]) -> tlinalg::Result<R>,
            ) -> tlinalg::Result<R> {
                $spectrum
            }

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

scalar!(
    f32,
    Complex32,
    F32,
    F32,
    f32,
    false,
    |values| values,
    |dest, _scratch, solve| solve(dest)
);
scalar!(
    f64,
    Complex64,
    F64,
    F64,
    f64,
    false,
    |values| values,
    |dest, _scratch, solve| solve(dest)
);
scalar!(
    Complex32,
    Complex32,
    C32,
    F32,
    c32,
    true,
    |values| values.into_iter().map(|value| value.re).collect(),
    |dest, scratch, solve| complex_spectrum(dest, scratch, solve)
);
scalar!(
    Complex64,
    Complex64,
    C64,
    F64,
    c64,
    true,
    |values| values.into_iter().map(|value| value.re).collect(),
    |dest, scratch, solve| complex_spectrum(dest, scratch, solve)
);

/// [`Scalar::with_spectrum`] of a complex scalar: tlinalg returns the real
/// spectrum with a zero imaginary part.
fn complex_spectrum<T: Copy, Out>(
    dest: &mut [T],
    scratch: &mut [num_complex::Complex<T>],
    solve: impl FnOnce(&mut [num_complex::Complex<T>]) -> tlinalg::Result<Out>,
) -> tlinalg::Result<Out> {
    let slot = &mut scratch[..dest.len()];
    let out = solve(slot)?;
    for (dest, value) in dest.iter_mut().zip(slot.iter()) {
        *dest = value.re;
    }
    Ok(out)
}

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
    type Lane = ();

    fn lane(&self, _: &mut dyn LaneJobs<Job<'a>>, _: &Exec<'_>) {}

    fn work(&self, _: usize, job: &Job<'a>) -> f64 {
        work::<F>(job.input)
    }

    fn modeled(&self, _: usize, job: &Job<'a>) -> bool {
        F::SEARCHED && single_item::<F>(job.input).is_some()
    }

    fn grant(&self, _: usize, job: &Job<'a>, exec: &Exec<'_>, budget: usize, _: Duration) -> usize {
        single_item::<F>(job.input).map_or(1, |item| grant(exec, budget, item))
    }

    fn grant_key(&self, _: usize, job: &Job<'a>) -> GrantKey {
        single_item::<F>(job.input).map_or([0; 2], |item| item.map(f64::to_bits))
    }

    // A block's error is its own outcome, not the driver's: the caller
    // delivers every block's outcome in input order.
    fn run(
        &self,
        _: usize,
        job: &mut Job<'a>,
        (): &mut (),
        mut exec: Exec<'_>,
    ) -> Result<(), DenseError> {
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
    allocation: LaneAllocation,
    inputs: &[DenseRead<'_>],
    sink: &mut Sink<'_>,
) -> Result<(), DenseError> {
    with_cpu_engine(engine, |cpu| batch_on::<F>(cpu, allocation, inputs, sink)).unwrap_or_else(
        || {
            Err(tenferro_error(
                F::ENTRY,
                tenferro_tensor::Error::unsupported(F::ENTRY, NOT_A_CPU_ENGINE),
            ))
        },
    )
}

/// [`factorize_batch`] of the factorization `op` names.
pub(crate) fn factorize(
    engine: &mut dyn tenferro_tensor::Engine,
    op: crate::DenseFactorization,
    allocation: LaneAllocation,
    inputs: &[DenseRead<'_>],
    sink: &mut Sink<'_>,
) -> Result<(), DenseError> {
    use crate::DenseFactorization as Op;
    match op {
        Op::Svd => factorize_batch::<Svd>(engine, allocation, inputs, sink),
        Op::Qr => factorize_batch::<Qr>(engine, allocation, inputs, sink),
        Op::Eigh => factorize_batch::<Eigh>(engine, allocation, inputs, sink),
        Op::EighVals => factorize_batch::<EighVals>(engine, allocation, inputs, sink),
        Op::SvdVals => factorize_batch::<SvdVals>(engine, allocation, inputs, sink),
        // Why not on the lanes: tenferro's eig publishes unit-norm
        // eigenvectors through a normalization it does not export
        // (tensor4all/tenferro-rs#2387), and symtn does not duplicate it, so
        // each input takes the per-matrix route on the whole engine and is
        // delivered before the next is solved.
        Op::Eig => inputs.iter().enumerate().try_for_each(|(index, &input)| {
            sink(index, crate::tenferro_adapter::eig_read(engine, input))
        }),
        Op::EigVals => factorize_batch::<EigVals>(engine, allocation, inputs, sink),
    }
}

fn batch_on<F: Factor>(
    cpu: &mut CpuEngine,
    allocation: LaneAllocation,
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
            allocation,
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

/// A [`Factor`] with a destination form: its provider workspace query and
/// its initialized-destination entry.
pub(crate) trait FactorInto: Factor {
    /// tlinalg's workspace bytes for `batch` items of `dims` on `exec`.
    fn workspace<T: Scalar>(
        dims: [usize; 2],
        batch: usize,
        exec: &Exec<'_>,
    ) -> tlinalg::Result<usize>;

    /// The typed scratch elements one input needs: the staged complex
    /// spectrum and, under [`FactorPlacement::Adjoint`], the provider-shaped
    /// factors.
    fn scratch_len<T: Scalar>(dims: [usize; 2], batch: usize, placement: FactorPlacement) -> usize;

    /// Solves one input into `outputs` (validated destinations of `T`).
    #[allow(clippy::too_many_arguments)]
    fn factor_into<T: Scalar>(
        raw: RawStridedRef<'_, T>,
        dims: [usize; 2],
        batch: usize,
        outputs: &mut [Option<DenseWrite<'_>>; 3],
        placement: FactorPlacement,
        workspace: &mut [MaybeUninit<u8>],
        scratch: &mut [T],
        exec: &mut Exec<'_>,
    ) -> tlinalg::Result<()>;
}

fn slot<'b, T: Scalar>(output: &'b mut Option<DenseWrite<'_>>) -> &'b mut [T] {
    output
        .as_mut()
        .and_then(T::factor_slot)
        .expect("validation checked the destination dtype")
}

fn spectrum<'b, T: Scalar>(
    output: &'b mut Option<DenseWrite<'_>>,
) -> &'b mut [<T as TensorScalar>::Real] {
    output
        .as_mut()
        .and_then(T::spectrum_slot)
        .expect("validation checked the destination dtype")
}

impl FactorInto for Qr {
    fn workspace<T: Scalar>(
        [m, n]: [usize; 2],
        batch: usize,
        exec: &Exec<'_>,
    ) -> tlinalg::Result<usize> {
        tlinalg::qr::qr_workspace::<T>(m, n, batch, exec)
    }

    fn scratch_len<T: Scalar>(
        [m, n]: [usize; 2],
        batch: usize,
        placement: FactorPlacement,
    ) -> usize {
        match placement {
            FactorPlacement::AsComputed => 0,
            FactorPlacement::Adjoint => (m + n) * m.min(n) * batch,
        }
    }

    fn factor_into<T: Scalar>(
        raw: RawStridedRef<'_, T>,
        [m, n]: [usize; 2],
        batch: usize,
        outputs: &mut [Option<DenseWrite<'_>>; 3],
        placement: FactorPlacement,
        workspace: &mut [MaybeUninit<u8>],
        scratch: &mut [T],
        exec: &mut Exec<'_>,
    ) -> tlinalg::Result<()> {
        let [first, second, _] = outputs;
        match placement {
            FactorPlacement::AsComputed => tlinalg::qr::qr_into_init(
                Self::OP,
                raw,
                slot::<T>(first),
                slot::<T>(second),
                workspace,
                exec,
            ),
            FactorPlacement::Adjoint => {
                let k = m.min(n);
                let (q, rest) = scratch.split_at_mut(m * k * batch);
                let r = &mut rest[..k * n * batch];
                tlinalg::qr::qr_into_init(Self::OP, raw, q, r, workspace, exec)?;
                adjoint_items(r, k, n, slot::<T>(first));
                adjoint_items(q, m, k, slot::<T>(second));
                Ok(())
            }
        }
    }
}

impl FactorInto for Svd {
    fn workspace<T: Scalar>(
        [m, n]: [usize; 2],
        batch: usize,
        exec: &Exec<'_>,
    ) -> tlinalg::Result<usize> {
        tlinalg::svd::svd_workspace::<T>(m, n, false, batch, exec)
    }

    fn scratch_len<T: Scalar>(
        [m, n]: [usize; 2],
        batch: usize,
        placement: FactorPlacement,
    ) -> usize {
        let k = m.min(n);
        let spectrum = if T::COMPLEX { k * batch } else { 0 };
        match placement {
            FactorPlacement::AsComputed => spectrum,
            FactorPlacement::Adjoint => spectrum + (m + n) * k * batch,
        }
    }

    fn factor_into<T: Scalar>(
        raw: RawStridedRef<'_, T>,
        [m, n]: [usize; 2],
        batch: usize,
        outputs: &mut [Option<DenseWrite<'_>>; 3],
        placement: FactorPlacement,
        workspace: &mut [MaybeUninit<u8>],
        scratch: &mut [T],
        exec: &mut Exec<'_>,
    ) -> tlinalg::Result<()> {
        let [first, values, third] = outputs;
        let values = spectrum::<T>(values);
        match placement {
            FactorPlacement::AsComputed => {
                let (u, vt) = (slot::<T>(first), slot::<T>(third));
                T::with_spectrum(values, scratch, |s| {
                    tlinalg::svd::svd_into_init(Self::OP, raw, false, u, s, vt, workspace, exec)
                })
            }
            FactorPlacement::Adjoint => {
                let k = m.min(n);
                let (u, rest) = scratch.split_at_mut(m * k * batch);
                let (vt, s) = rest.split_at_mut(k * n * batch);
                T::with_spectrum(values, s, |s| {
                    tlinalg::svd::svd_into_init(Self::OP, raw, false, u, s, vt, workspace, exec)
                })?;
                adjoint_items(vt, k, n, slot::<T>(first));
                adjoint_items(u, m, k, slot::<T>(third));
                Ok(())
            }
        }
    }
}

impl FactorInto for Eigh {
    fn workspace<T: Scalar>(
        [n, _]: [usize; 2],
        batch: usize,
        exec: &Exec<'_>,
    ) -> tlinalg::Result<usize> {
        tlinalg::eigh::eigh_workspace::<T>(n, true, batch, exec)
    }

    fn scratch_len<T: Scalar>([n, _]: [usize; 2], batch: usize, _: FactorPlacement) -> usize {
        if T::COMPLEX {
            n * batch
        } else {
            0
        }
    }

    fn factor_into<T: Scalar>(
        raw: RawStridedRef<'_, T>,
        _: [usize; 2],
        _: usize,
        outputs: &mut [Option<DenseWrite<'_>>; 3],
        _: FactorPlacement,
        workspace: &mut [MaybeUninit<u8>],
        scratch: &mut [T],
        exec: &mut Exec<'_>,
    ) -> tlinalg::Result<()> {
        let [values, second, _] = outputs;
        let vectors = slot::<T>(second);
        T::with_spectrum(spectrum::<T>(values), scratch, |w| {
            tlinalg::eigh::eigh_into_init(Self::OP, raw, w, vectors, workspace, exec)
        })
    }
}

/// One lane's typed scratch per scalar, zero-filled once when the lane is
/// set up.
#[derive(Default)]
pub(crate) struct TypedScratch {
    f32: Vec<f32>,
    f64: Vec<f64>,
    c32: Vec<Complex32>,
    c64: Vec<Complex64>,
}

/// One lane's provider workspace and typed scratch: the largest need of the
/// lane's blocks, allocated once before its first block.
pub(crate) struct LaneBuffers {
    workspace: Vec<MaybeUninit<u8>>,
    typed: TypedScratch,
    /// A process-unique id of this lane setup.
    #[cfg(test)]
    id: usize,
}

/// The largest workspace bytes and typed scratch elements (per scalar, in
/// [`TypedScratch`] field order) of one lane's blocks.
#[derive(Default)]
struct Needs {
    bytes: usize,
    typed: [usize; 4],
}

/// One block of a destination batch: its validated input, its signed
/// strides, its destination, a workspace query that failed while its lane
/// was set up, and, once run, its outcome.
pub(crate) struct IntoJob<'a, 'd, 'w> {
    input: DenseRead<'a>,
    strides: StrideVec,
    dest: &'d mut FactorDest<'w>,
    sizing: Option<DenseError>,
    outcome: Option<Result<(), FactorFailure>>,
    /// The id and workspace length of the lane setup this block ran with.
    #[cfg(test)]
    lane_seen: (usize, usize),
}

/// Factorization `F` of each block into its destination on the lane driver.
struct IntoBlocks<'a, 'd, 'w, F>(PhantomData<IntoMarker<'a, 'd, 'w, F>>);

/// [`IntoBlocks`]' variance and auto-trait marker.
type IntoMarker<'a, 'd, 'w, F> = fn() -> (&'a (), &'d (), &'w (), F);

impl<'a, 'd, 'w: 'd, F: FactorInto> BlockOp for IntoBlocks<'a, 'd, 'w, F> {
    type Job = IntoJob<'a, 'd, 'w>;
    const ENTRY: Entry = Entry::Linalg;
    const KIND: OpKind = OpKind::Factorization;
    const LABEL: &'static str = F::ENTRY;
    type Lane = LaneBuffers;

    fn lane(&self, jobs: &mut dyn LaneJobs<Self::Job>, exec: &Exec<'_>) -> LaneBuffers {
        let mut needs = Needs::default();
        jobs.for_each(&mut |_, job| {
            if let Err(error) = needs_of::<F>(job.input, job.dest.placement(), exec, &mut needs) {
                job.sizing = Some(error);
            }
        });
        LaneBuffers {
            workspace: vec![MaybeUninit::uninit(); needs.bytes],
            typed: TypedScratch {
                f32: vec![0.0; needs.typed[0]],
                f64: vec![0.0; needs.typed[1]],
                c32: vec![Complex32::default(); needs.typed[2]],
                c64: vec![Complex64::default(); needs.typed[3]],
            },
            #[cfg(test)]
            id: {
                static SETUPS: std::sync::atomic::AtomicUsize =
                    std::sync::atomic::AtomicUsize::new(1);
                SETUPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            },
        }
    }

    fn work(&self, _: usize, job: &Self::Job) -> f64 {
        work::<F>(job.input)
    }

    fn modeled(&self, _: usize, job: &Self::Job) -> bool {
        F::SEARCHED && single_item::<F>(job.input).is_some()
    }

    fn grant(
        &self,
        _: usize,
        job: &Self::Job,
        exec: &Exec<'_>,
        budget: usize,
        _: Duration,
    ) -> usize {
        single_item::<F>(job.input).map_or(1, |item| grant(exec, budget, item))
    }

    fn grant_key(&self, _: usize, job: &Self::Job) -> GrantKey {
        single_item::<F>(job.input).map_or([0; 2], |item| item.map(f64::to_bits))
    }

    // A block's error is its own outcome, not the driver's: the caller
    // delivers every block's outcome in input order.
    fn run(
        &self,
        _: usize,
        job: &mut Self::Job,
        lane: &mut LaneBuffers,
        mut exec: Exec<'_>,
    ) -> Result<(), DenseError> {
        #[cfg(test)]
        {
            job.lane_seen = (lane.id, lane.workspace.len());
        }
        let sizing = job.sizing.take();
        let outcome = match job.input {
            DenseRead::F32(view) => {
                factor_into_typed::<F, _>(view, &job.strides, job.dest, sizing, lane, &mut exec)
            }
            DenseRead::F64(view) => {
                factor_into_typed::<F, _>(view, &job.strides, job.dest, sizing, lane, &mut exec)
            }
            DenseRead::C32(view) => {
                factor_into_typed::<F, _>(view, &job.strides, job.dest, sizing, lane, &mut exec)
            }
            DenseRead::C64(view) => {
                factor_into_typed::<F, _>(view, &job.strides, job.dest, sizing, lane, &mut exec)
            }
            _ => unreachable!("preflight admits the provider's dtypes only"),
        };
        job.outcome = Some(outcome.map_err(FactorFailure::Solver));
        Ok(())
    }
}

/// Folds `input`'s workspace and typed scratch into `needs` at the lane's
/// context `exec`. An input whose rank, squareness or extents the solve
/// refuses or skips needs nothing; a failed workspace query is that input's
/// own outcome.
fn needs_of<F: FactorInto>(
    input: DenseRead<'_>,
    placement: FactorPlacement,
    exec: &Exec<'_>,
    needs: &mut Needs,
) -> Result<(), DenseError> {
    fn of<F: FactorInto, T: Scalar>(
        shape: &[usize],
        placement: FactorPlacement,
        exec: &Exec<'_>,
        slot: usize,
        needs: &mut Needs,
    ) -> Result<(), DenseError> {
        let [m, n, batch @ ..] = shape else {
            return Ok(());
        };
        if F::SQUARE && m != n || shape.contains(&0) {
            return Ok(());
        }
        // The view's element count fits, so its batch product does.
        let items = batch.iter().product::<usize>();
        let bytes = F::workspace::<T>([*m, *n], items, exec)
            .map_err(|err| tenferro_error(F::ENTRY, provider_error(F::OP, err)))?;
        needs.bytes = needs.bytes.max(bytes);
        needs.typed[slot] = needs.typed[slot].max(F::scratch_len::<T>([*m, *n], items, placement));
        Ok(())
    }
    match input {
        DenseRead::F32(view) => of::<F, f32>(view.shape(), placement, exec, 0, needs),
        DenseRead::F64(view) => of::<F, f64>(view.shape(), placement, exec, 1, needs),
        DenseRead::C32(view) => of::<F, Complex32>(view.shape(), placement, exec, 2, needs),
        DenseRead::C64(view) => of::<F, Complex64>(view.shape(), placement, exec, 3, needs),
        _ => Ok(()),
    }
}

/// [`factor_typed`]'s checks in the same order (rank, square, the
/// empty-extent shortcut, layout), then the lane's failed workspace query or
/// the provider's destination entry on the lane's buffers.
fn factor_into_typed<F: FactorInto, T: Scalar>(
    view: DenseView<'_, T>,
    strides: &[isize],
    dest: &mut FactorDest<'_>,
    sizing: Option<DenseError>,
    lane: &mut LaneBuffers,
    exec: &mut Exec<'_>,
) -> Result<(), DenseError> {
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
    if shape.contains(&0) {
        return Ok(());
    }
    let offset = isize::try_from(view.offset()).map_err(|_| DenseError::OffsetOverflow {
        value: view.offset(),
    })?;
    let raw = RawStridedRef::new(view.data(), shape, strides, offset).map_err(|err| {
        tenferro_error(
            F::ENTRY,
            tenferro_tensor::Error::invalid_argument(op, "layout", err.to_string()),
        )
    })?;
    if let Some(error) = sizing {
        return Err(error);
    }
    let items = batch.iter().product::<usize>();
    let placement = dest.placement();
    F::factor_into::<T>(
        raw,
        [*m, *n],
        items,
        &mut dest.outputs,
        placement,
        &mut lane.workspace,
        T::scratch(&mut lane.typed),
        exec,
    )
    .map_err(|err| tenferro_error(F::ENTRY, provider_error(F::OP, err)))
}

/// The destination batch's dispatch over the ops with a destination form;
/// the caller validated `op` and `dests`.
pub(crate) fn factorize_into(
    engine: &mut dyn tenferro_tensor::Engine,
    op: crate::DenseFactorization,
    allocation: LaneAllocation,
    inputs: &[DenseRead<'_>],
    dests: &mut [FactorDest<'_>],
    sink: &mut FactorSink<'_>,
) -> Result<(), DenseError> {
    use crate::DenseFactorization as Op;
    match op {
        Op::Svd => factorize_into_batch::<Svd>(engine, allocation, inputs, dests, sink),
        Op::Qr => factorize_into_batch::<Qr>(engine, allocation, inputs, dests, sink),
        Op::Eigh => factorize_into_batch::<Eigh>(engine, allocation, inputs, dests, sink),
        _ => unreachable!("validation admits the ops with a destination form only"),
    }
}

fn factorize_into_batch<F: FactorInto>(
    engine: &mut dyn tenferro_tensor::Engine,
    allocation: LaneAllocation,
    inputs: &[DenseRead<'_>],
    dests: &mut [FactorDest<'_>],
    sink: &mut FactorSink<'_>,
) -> Result<(), DenseError> {
    with_cpu_engine(engine, |cpu| {
        batch_into_on::<F>(cpu, allocation, inputs, dests, sink)
    })
    .unwrap_or_else(|| {
        Err(tenferro_error(
            F::ENTRY,
            tenferro_tensor::Error::unsupported(F::ENTRY, NOT_A_CPU_ENGINE),
        ))
    })
}

/// [`batch_on`] into destinations: the same preflight, admission, lanes and
/// ordered post-join delivery, with each block's outcome kept as
/// `Result<(), FactorFailure>` instead of its factors.
fn batch_into_on<F: FactorInto>(
    cpu: &mut CpuEngine,
    allocation: LaneAllocation,
    inputs: &[DenseRead<'_>],
    dests: &mut [FactorDest<'_>],
    sink: &mut FactorSink<'_>,
) -> Result<(), DenseError> {
    let mut jobs = Vec::with_capacity(inputs.len());
    let mut slots = dests.iter_mut();
    let mut refused = None;
    for (&input, dest) in inputs.iter().zip(&mut slots) {
        match preflight::<F>(input) {
            Ok(strides) => jobs.push(IntoJob {
                input,
                strides,
                dest,
                sizing: None,
                outcome: None,
                #[cfg(test)]
                lane_seen: (0, 0),
            }),
            Err(error) => {
                refused = Some((error, dest));
                break;
            }
        }
    }
    if !jobs.is_empty() {
        driver::drive(
            cpu,
            &IntoBlocks::<F>(PhantomData),
            &mut jobs,
            &mut LaneScratch::default(),
            allocation,
        )?;
    }
    let admitted = jobs.len();
    for (index, job) in jobs.into_iter().enumerate() {
        let outcome = job
            .outcome
            .unwrap_or_else(|| unreachable!("the driver runs every admitted job"));
        sink(index, job.dest, outcome)?;
    }
    match refused {
        Some((error, dest)) => sink(admitted, dest, Err(FactorFailure::Solver(error))),
        None => Ok(()),
    }
}

/// tlinalg's per-item estimates of `F` for `input`'s scalar (see
/// [`Factor::estimates`]) and its batch product, or `None` for an input the
/// provider refuses (rank or dtype).
fn item_estimates<F: Factor>(input: DenseRead<'_>) -> Option<([f64; 2], f64)> {
    // Direct generic calls, not a function pointer: the estimates inline, and
    // `work`, which reads only the total, need not evaluate the payback.
    fn of<F: Factor, T: tlinalg::FaerScalar>(shape: &[usize]) -> Option<([f64; 2], f64)> {
        let [m, n, batch @ ..] = shape else {
            return None;
        };
        let items = batch.iter().map(|&b| b as f64).product();
        Some((F::estimates::<T>(*m, *n), items))
    }
    match input {
        DenseRead::F32(view) => of::<F, f32>(view.shape()),
        DenseRead::F64(view) => of::<F, f64>(view.shape()),
        DenseRead::C32(view) => of::<F, Complex32>(view.shape()),
        DenseRead::C64(view) => of::<F, Complex64>(view.shape()),
        _ => None,
    }
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
    // one per-call region. The owned outcome needs fresh factor storage per
    // block anyway; the destination batch (`factor_into_typed`) is the
    // lane-reused-workspace route.
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

    /// Factorization `F`'s own op with recording-only execution: cost,
    /// search admission and grant are `Blocks<F>`'s, and each block records
    /// the reserved width of the lane it runs on instead of factoring.
    struct RecordsWidth<'a, F>(Blocks<'a, F>, Vec<AtomicUsize>);

    impl<'a, F: Factor> BlockOp for RecordsWidth<'a, F> {
        type Job = Job<'a>;
        const ENTRY: Entry = Entry::Linalg;
        const KIND: OpKind = OpKind::Factorization;
        const LABEL: &'static str = F::ENTRY;
        type Lane = ();

        fn lane(&self, _: &mut dyn LaneJobs<Job<'a>>, _: &Exec<'_>) {}

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

        fn run(
            &self,
            index: usize,
            _: &mut Job<'a>,
            (): &mut (),
            exec: Exec<'_>,
        ) -> Result<(), DenseError> {
            self.1[index].store(exec.budget(), Ordering::Relaxed);
            Ok(())
        }
    }

    /// The reserved lane width each real input of `shapes` runs on under
    /// factorization `F`, on `threads` slots of flat 100 ns dispatch at one
    /// multiply-add per ns, no item overhead and a lane work minimum of
    /// `lane_ns`.
    fn lane_widths<F: Factor>(threads: usize, lane_ns: f64, shapes: &[&[usize]]) -> Vec<usize> {
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
                    strides: preflight::<F>(input).unwrap(),
                    outcome: None,
                }
            })
            .collect();
        let op = RecordsWidth::<F>(
            Blocks(PhantomData),
            shapes.iter().map(|_| 0.into()).collect(),
        );
        driver::drive(
            &mut engine,
            &op,
            &mut jobs,
            &mut LaneScratch::default(),
            LaneAllocation::Moldable,
        )
        .unwrap();
        op.1.iter().map(|w| w.load(Ordering::Relaxed)).collect()
    }

    /// Purpose: each lane of a destination batch is set up once, before its
    /// first block, with a workspace sufficient for every block it runs:
    /// every block of a lane runs with that one setup, whose workspace is
    /// never regrown, on uneven blocks over two forced lanes and on the
    /// single-lane path, including a two-item and an empty-batch input; every
    /// block succeeds, so no workspace was short.
    #[test]
    fn each_lane_is_set_up_once_for_all_its_blocks() {
        use crate::factor_dest::validate;
        use std::collections::BTreeMap;
        use tcontract_exec::BatchStrategy;
        let shapes: [&[usize]; 5] = [&[40, 30], &[3, 2], &[17, 9], &[6, 5, 2], &[4, 3, 0]];
        let data = vec![0.25_f64; 2400];
        let strides: Vec<Vec<usize>> = shapes
            .iter()
            .map(|shape| {
                (0..shape.len())
                    .map(|axis| shape[..axis].iter().product())
                    .collect()
            })
            .collect();
        let inputs: Vec<DenseRead<'_>> = shapes
            .iter()
            .zip(&strides)
            .map(|(&shape, strides)| {
                let len = shape.iter().product();
                DenseRead::F64(DenseView::new(&data[..len], shape, strides, 0).unwrap())
            })
            .collect();
        let engines = [
            CpuEngine::serial(),
            CpuEngine::new(Pool::builder().threads(4).build().unwrap()).with_batch_policy(
                BatchPolicy::new(BatchStrategy::Split {
                    lanes: std::num::NonZeroUsize::new(2).unwrap(),
                }),
            ),
        ];
        for (mut engine, lanes) in engines.into_iter().zip([1, 2]) {
            let geometry: Vec<_> = shapes
                .iter()
                .map(|shape| {
                    let (m, n) = (shape[0], shape[1]);
                    let k = m.min(n);
                    let batch = &shape[2..];
                    let q = [&[m, k][..], batch].concat();
                    let r = [&[k, n][..], batch].concat();
                    let q_strides: Vec<usize> =
                        (0..q.len()).map(|a| q[..a].iter().product()).collect();
                    let r_strides: Vec<usize> =
                        (0..r.len()).map(|a| r[..a].iter().product()).collect();
                    (q, q_strides, r, r_strides)
                })
                .collect();
            let mut storage: Vec<(Vec<f64>, Vec<f64>)> = geometry
                .iter()
                .map(|(q, _, r, _)| (vec![0.0; q.iter().product()], vec![0.0; r.iter().product()]))
                .collect();
            let mut dests: Vec<FactorDest<'_>> = storage
                .iter_mut()
                .zip(&geometry)
                .map(|((q, r), (qs, qst, rs, rst))| {
                    FactorDest::new(
                        [
                            Some(DenseWrite::F64(
                                crate::DenseViewMut::new(q, qs, qst, 0).unwrap(),
                            )),
                            Some(DenseWrite::F64(
                                crate::DenseViewMut::new(r, rs, rst, 0).unwrap(),
                            )),
                            None,
                        ],
                        FactorPlacement::AsComputed,
                    )
                })
                .collect();
            validate(crate::DenseFactorization::Qr, &inputs, &dests).unwrap();
            let mut jobs: Vec<IntoJob<'_, '_, '_>> = inputs
                .iter()
                .zip(dests.iter_mut())
                .map(|(&input, dest)| IntoJob {
                    input,
                    strides: preflight::<Qr>(input).unwrap(),
                    dest,
                    sizing: None,
                    outcome: None,
                    lane_seen: (0, 0),
                })
                .collect();
            driver::drive(
                &mut engine,
                &IntoBlocks::<Qr>(PhantomData),
                &mut jobs,
                &mut LaneScratch::default(),
                LaneAllocation::Moldable,
            )
            .unwrap();
            let mut setups = BTreeMap::new();
            for job in &jobs {
                assert_eq!(job.outcome, Some(Ok(())));
                let (id, len) = job.lane_seen;
                assert_eq!(
                    *setups.entry(id).or_insert(len),
                    len,
                    "one buffer per setup"
                );
            }
            assert_eq!(setups.len(), lanes, "{lanes} lane(s)");
        }
    }

    /// Purpose: a factorization batch whose every input is one matrix joins
    /// the lane search with tlinalg's own width for one SVD item, while a QR
    /// batch and a batch holding an input whose batch product is not one
    /// (including zero) keep the baseline placement. Four slots, flat 100 ns dispatch (a width pays
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
    ///
    /// QR keeps the baseline even where its grant would widen: the first
    /// batch's `80 x 80` QR has payback `(80^3 - 80^3 / 3) * 1e-2 = 3413` ns,
    /// enough for every width, yet runs on the baseline's four one-slot lanes.
    #[test]
    fn single_matrix_factorizations_join_the_lane_search() {
        let small: &[usize] = &[4, 4];
        assert_eq!(
            lane_widths::<Svd>(4, 0.0, &[&[80, 80], small, small, small]),
            [4; 4]
        );
        assert_eq!(
            lane_widths::<Svd>(4, 0.0, &[&[79, 79], small, small, small]),
            [1; 4]
        );
        assert_eq!(
            lane_widths::<Svd>(4, 0.0, &[&[80, 80], small, small, &[4, 4, 2]]),
            [1; 4]
        );
        let mut many: Vec<&[usize]> = vec![&[10000, 1]];
        many.extend([&[100, 1][..]; 120]);
        assert_eq!(lane_widths::<Svd>(6, 10000.0, &many), [3; 121]);
        assert_eq!(
            lane_widths::<Qr>(4, 0.0, &[&[80, 80], small, small, small]),
            [1; 4]
        );
    }
}
