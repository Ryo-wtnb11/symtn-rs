//! Caller-owned destinations of one batched factorization
//! ([`crate::DenseExecutor::factorize_batch_into`]): their validation, the
//! placement of a factorization's outputs into them, the compatibility
//! route through the owned batch, and the per-matrix composite of the
//! destination-only polar decompositions and Hermitian exponential.

use num_complex::{Complex32, Complex64};

use crate::executor::arity_mismatch;
use crate::{
    check_contiguous_output, DenseDType, DenseDotConfig, DenseError, DenseExecutor,
    DenseFactorization, DenseRead, DenseTensor, DenseView, DenseViewMut, DenseWrite, Engine,
};

/// How a [`FactorDest`] receives one input's factors.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum FactorPlacement {
    /// The factors of the input exactly as the per-matrix entry returns
    /// them.
    #[default]
    AsComputed,
    /// The factors of the input's adjoint problem: the outputs in reversed
    /// factor order, each conjugate-transposed (per batch item), the
    /// spectrum unchanged. QR of `A` is placed as the LQ of `A^H`
    /// (`l = R^H`, `q = Q^H`); the compact SVD of `A` as the compact SVD of
    /// `A^H` (`u = Vh^H`, `s`, `vt = U^H`), and the full SVD likewise.
    /// Hermitian eigendecompositions, LQ, full QR, the QR null basis, the polar decompositions and the
    /// Hermitian exponential refuse it.
    Adjoint,
}

/// Why one input's factorization failed, kept apart so a caller applies its
/// own precedence between solver failures and malformed outputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FactorFailure {
    /// The solver failed: the provider, or the executor's owned entry,
    /// reported this error for the input.
    Solver(DenseError),
    /// The solver returned, but its outputs do not fit the destination (a
    /// custom executor's owned outputs of the wrong count, dtype or shape),
    /// or fail the op's own output check (the Hermitian exponential's
    /// finite-eigenvalue check, [`EIGH_EIGENVALUE_CHECK`]).
    Output(DenseError),
}

/// The `op` of symtn's own Hermitian eigenvalue check, so a caller can tell
/// it from an executor's `NumericalFailure` by matching it exactly.
#[doc(hidden)]
pub const EIGH_EIGENVALUE_CHECK: &str = "eigh eigenvalue check";

/// The finite-eigenvalue check of an admitted finite input's Hermitian
/// eigendecomposition: a nonfinite eigenvalue is the eigensolver's numerical
/// failure, not a misuse ([`EIGH_EIGENVALUE_CHECK`]).
#[doc(hidden)]
pub fn finite_eigenvalues<R: Into<f64>>(
    values: impl IntoIterator<Item = R>,
) -> Result<(), DenseError> {
    if values.into_iter().all(|value| value.into().is_finite()) {
        return Ok(());
    }
    Err(DenseError::NumericalFailure {
        backend: crate::DenseBackend::Tenferro,
        op: EIGH_EIGENVALUE_CHECK,
        message: "eigenvalues must be finite".to_string(),
    })
}

impl FactorFailure {
    /// The error, whatever its origin.
    pub fn into_error(self) -> DenseError {
        match self {
            Self::Solver(error) | Self::Output(error) => error,
        }
    }
}

/// The destinations of one input of a batched factorization, in the op's
/// owned output order: `Qr` and `QrFull` `[q, r, -]`, `QrNull` `[n, -, -]`,
/// `Lq` `[l, q, -]`, `Svd` and `SvdFull` `[u, s, vt]`, `Eigh` `[values, vectors, -]`,
/// `LeftPolar` `[w, p, -]`, `RightPolar` `[p, wh, -]`, `ExpViaEigh`
/// `[exp, -, -]` (`-` is `None`).
///
/// Each factor view has the input's dtype and each spectrum (`s`, eigh
/// `values`) the input's real counterpart (`f32` for `f32`/`c32`, `f64` for
/// `f64`/`c64`), exactly what the per-matrix entry returns. Every view is
/// compact column-major with the output's shape, which [`FactorPlacement`]
/// sets: for an `m x n` input with trailing batch dimensions `b..` and
/// `k = min(m, n)`, `AsComputed` QR is `[m, k, b..]`, `[k, n, b..]` and
/// `Adjoint` QR `[n, k, b..]`, `[k, m, b..]`; LQ is `[m, k, b..]`,
/// `[k, n, b..]`; full QR is `[m, m, b..]`, `[m, n, b..]` and the QR null
/// basis `[m, m - k, b..]`; SVD adds `s` of shape
/// `[k, b..]` between them, and full SVD is `[m, m, b..]`, `[k, b..]`,
/// `[n, n, b..]` (`Adjoint`: `[n, n, b..]`, `[k, b..]`, `[m, m, b..]`); eigh of `n x n` is `[n, b..]`, `[n, n, b..]`;
/// left polar is `[m, n, b..]`, `[n, n, b..]`, right polar
/// `[m, m, b..]`, `[m, n, b..]` and the Hermitian exponential of `n x n`
/// `[n, n, b..]`.
#[derive(Debug)]
pub struct FactorDest<'a> {
    pub(crate) outputs: [Option<DenseWrite<'a>>; 3],
    placement: FactorPlacement,
}

impl<'a> FactorDest<'a> {
    pub fn new(outputs: [Option<DenseWrite<'a>>; 3], placement: FactorPlacement) -> Self {
        Self { outputs, placement }
    }

    pub fn placement(&self) -> FactorPlacement {
        self.placement
    }

    /// The outputs in the op's owned order, `None` past the op's arity.
    pub fn outputs_mut(&mut self) -> &mut [Option<DenseWrite<'a>>; 3] {
        &mut self.outputs
    }
}

const ENTRY: &str = "factorize_batch_into";

/// The outputs `op` delivers into a destination, or `Unsupported` for an op
/// without a destination form.
fn arity(op: DenseFactorization) -> Result<usize, DenseError> {
    match op {
        DenseFactorization::QrNull | DenseFactorization::ExpViaEigh => Ok(1),
        DenseFactorization::Qr
        | DenseFactorization::Lq
        | DenseFactorization::QrFull
        | DenseFactorization::Eigh
        | DenseFactorization::LeftPolar
        | DenseFactorization::RightPolar => Ok(2),
        DenseFactorization::Svd | DenseFactorization::SvdFull => Ok(3),
        _ => Err(DenseError::Unsupported {
            op: ENTRY,
            message: format!("{op:?} has no destination form"),
        }),
    }
}

/// The per-matrix entry label of `op`'s malformed-output errors.
fn output_label(op: DenseFactorization) -> &'static str {
    match op {
        DenseFactorization::Svd
        | DenseFactorization::SvdFull
        | DenseFactorization::LeftPolar
        | DenseFactorization::RightPolar => "svd_into",
        DenseFactorization::Eigh | DenseFactorization::ExpViaEigh => "eigh_into",
        DenseFactorization::Lq => "lq",
        _ => "qr_into",
    }
}

/// The real counterpart of a floating dtype.
fn real_of(dtype: DenseDType) -> Option<DenseDType> {
    match dtype {
        DenseDType::F32 | DenseDType::C32 => Some(DenseDType::F32),
        DenseDType::F64 | DenseDType::C64 => Some(DenseDType::F64),
        _ => None,
    }
}

/// Output `index`'s leading extents (before the batch dimensions) of `op`
/// on an `m x n` input, as the destination stores it, and how many of the
/// two are used (one for a spectrum).
fn head(
    op: DenseFactorization,
    placement: FactorPlacement,
    [m, n]: [usize; 2],
    index: usize,
) -> ([usize; 2], usize) {
    let k = m.min(n);
    let adjoint = placement == FactorPlacement::Adjoint;
    match (op, index, adjoint) {
        (DenseFactorization::Eigh, 0, _) => ([m, 0], 1),
        (DenseFactorization::Eigh, _, _) => ([m, m], 2),
        (DenseFactorization::Svd | DenseFactorization::SvdFull, 1, _) => ([k, 0], 1),
        // `Adjoint` places `U^H` last.
        (DenseFactorization::SvdFull, 0, false) | (DenseFactorization::SvdFull, 2, true) => {
            ([m, m], 2)
        }
        (DenseFactorization::SvdFull, _, _) => ([n, n], 2),
        (DenseFactorization::QrFull, 0, _) => ([m, m], 2),
        (DenseFactorization::QrFull, _, _) => ([m, n], 2),
        (DenseFactorization::QrNull, _, _) => ([m, m - k], 2),
        (DenseFactorization::LeftPolar, 0, _) => ([m, n], 2),
        (DenseFactorization::LeftPolar, _, _) => ([n, n], 2),
        (DenseFactorization::RightPolar, 0, _) => ([m, m], 2),
        (DenseFactorization::RightPolar, _, _) => ([m, n], 2),
        (DenseFactorization::ExpViaEigh, _, _) => ([m, m], 2),
        (_, 0, false) => ([m, k], 2),
        (_, 0, true) => ([n, k], 2),
        (_, _, false) => ([k, n], 2),
        (_, _, true) => ([k, m], 2),
    }
}

/// Whether output `index` of `op` is a spectrum (real counterpart dtype).
fn is_spectrum(op: DenseFactorization, index: usize) -> bool {
    matches!(
        (op, index),
        (DenseFactorization::Eigh, 0)
            | (DenseFactorization::Svd, 1)
            | (DenseFactorization::SvdFull, 1)
    )
}

fn write_strides<'b>(write: &'b DenseWrite<'_>) -> (&'b [usize], usize) {
    match write {
        DenseWrite::F32(view) => (view.strides(), view.offset()),
        DenseWrite::F64(view) => (view.strides(), view.offset()),
        DenseWrite::I32(view) => (view.strides(), view.offset()),
        DenseWrite::I64(view) => (view.strides(), view.offset()),
        DenseWrite::Bool(view) => (view.strides(), view.offset()),
        DenseWrite::C32(view) => (view.strides(), view.offset()),
        DenseWrite::C64(view) => (view.strides(), view.offset()),
    }
}

/// Checks `actual == head ++ batch` without allocating unless it fails.
fn check_shape(actual: &[usize], head: &[usize], batch: &[usize]) -> Result<(), DenseError> {
    if actual.len() == head.len() + batch.len()
        && actual[..head.len()] == *head
        && actual[head.len()..] == *batch
    {
        return Ok(());
    }
    Err(DenseError::ShapeMismatch {
        op: ENTRY,
        expected: [head, batch].concat(),
        actual: actual.to_vec(),
    })
}

/// Checks that a view of `shape` is compact column-major, returning its
/// element count.
fn check_compact(shape: &[usize], strides: &[usize]) -> Result<usize, DenseError> {
    let len = shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or(DenseError::ElementCountOverflow)?;
    if len == 0 {
        return Ok(0);
    }
    let mut expected = 1usize;
    for (&dim, &stride) in shape.iter().zip(strides) {
        if stride != expected {
            let mut compact = Vec::with_capacity(shape.len());
            let mut step = 1usize;
            for &dim in shape {
                compact.push(step);
                step = step.saturating_mul(dim);
            }
            return Err(DenseError::ShapeMismatch {
                op: ENTRY,
                expected: compact,
                actual: strides.to_vec(),
            });
        }
        expected *= dim;
    }
    Ok(len)
}

/// Validates every destination of the batch before any solve or write:
/// the destination count, each destination's output pattern and placement,
/// then for each input the provider can factor (a floating rank-2-or-more
/// input, square for eigh) each output's dtype, shape and compact layout.
/// An input the solver refuses keeps that refusal as its delivered outcome.
pub(crate) fn validate(
    op: DenseFactorization,
    inputs: &[DenseRead<'_>],
    dests: &[FactorDest<'_>],
) -> Result<(), DenseError> {
    let arity = arity(op)?;
    if dests.len() != inputs.len() {
        return Err(arity_mismatch(ENTRY, inputs.len(), dests.len()));
    }
    for (input, dest) in inputs.iter().zip(dests) {
        let present = dest
            .outputs
            .iter()
            .filter(|output| output.is_some())
            .count();
        if present != arity || dest.outputs[..arity].iter().any(Option::is_none) {
            return Err(arity_mismatch(ENTRY, arity, present));
        }
        if matches!(
            op,
            DenseFactorization::Eigh
                | DenseFactorization::Lq
                | DenseFactorization::QrFull
                | DenseFactorization::QrNull
                | DenseFactorization::LeftPolar
                | DenseFactorization::RightPolar
                | DenseFactorization::ExpViaEigh
        ) && dest.placement == FactorPlacement::Adjoint
        {
            return Err(DenseError::Unsupported {
                op: ENTRY,
                message: format!("{op:?} has no adjoint placement"),
            });
        }
        let dtype = input.dtype();
        let Some(real) = real_of(dtype) else {
            continue;
        };
        let [m, n, batch @ ..] = input.shape() else {
            continue;
        };
        if matches!(
            op,
            DenseFactorization::Eigh | DenseFactorization::ExpViaEigh
        ) && m != n
        {
            continue;
        }
        for (index, output) in dest.outputs[..arity].iter().enumerate() {
            let output = output.as_ref().expect("the pattern was checked");
            let expected = if is_spectrum(op, index) { real } else { dtype };
            if output.dtype() != expected {
                return Err(DenseError::DTypeMismatch {
                    op: ENTRY,
                    expected,
                    actual: output.dtype(),
                });
            }
            let (dims, rank) = head(op, dest.placement, [*m, *n], index);
            check_shape(output.shape(), &dims[..rank], batch)?;
            check_compact(output.shape(), write_strides(output).0)?;
        }
    }
    Ok(())
}

/// A scalar a destination holds: its conjugate, the arithmetic of the polar
/// and exponential post-processing, and its dtype-erased views and owned
/// outputs.
pub(crate) trait Element: Copy {
    /// The real counterpart, in which a spectrum is published.
    type Spectrum: tenferro_tensor::TensorScalar + Default + Into<f64>;
    fn conj(self) -> Self;
    /// `self * sqrt(sigma)`.
    fn scale_by_sqrt(self, sigma: Self::Spectrum) -> Self;
    /// `self * exp(lambda / 2)`.
    fn scale_by_half_exp(self, lambda: Self::Spectrum) -> Self;
    /// The real part, as this scalar.
    fn real_part(self) -> Self;
    /// `(self + other^H) / 2`.
    fn hermitian_mean(self, other: Self) -> Self;
    fn read(view: DenseView<'_, Self>) -> DenseRead<'_>;
    fn write(view: DenseViewMut<'_, Self>) -> DenseWrite<'_>;
    /// The compact payload of a validated destination of this scalar,
    /// `None` for another dtype.
    fn compact<'b>(write: &'b mut DenseWrite<'_>) -> Option<&'b mut [Self]>;
    fn slice(tensor: &DenseTensor) -> Result<&[Self], DenseError>;
    fn into_vec(tensor: DenseTensor) -> Result<Vec<Self>, DenseError>;
    fn spectrum(tensor: &DenseTensor) -> Result<&[Self::Spectrum], DenseError>;
}

macro_rules! element {
    ($ty:ty, $real:ty, $variant:ident, $slice:ident, $into_vec:ident, $spectrum:ident,
     |$c:ident| $conj:expr, |$p:ident| $part:expr,
     |$x:ident, $s:ident| $scale:expr) => {
        impl Element for $ty {
            type Spectrum = $real;
            fn conj(self) -> Self {
                let $c = self;
                $conj
            }
            fn scale_by_sqrt(self, sigma: $real) -> Self {
                let ($x, $s) = (self, sigma.sqrt());
                $scale
            }
            fn scale_by_half_exp(self, lambda: $real) -> Self {
                let ($x, $s) = (self, (lambda / 2.0).exp());
                $scale
            }
            fn real_part(self) -> Self {
                let $p = self;
                $part
            }
            fn hermitian_mean(self, other: Self) -> Self {
                let sum = self + Element::conj(other);
                let half: $real = 0.5;
                let $x = sum;
                let $s = half;
                $scale
            }
            fn read(view: DenseView<'_, Self>) -> DenseRead<'_> {
                DenseRead::$variant(view)
            }
            fn write(view: DenseViewMut<'_, Self>) -> DenseWrite<'_> {
                DenseWrite::$variant(view)
            }
            fn compact<'b>(write: &'b mut DenseWrite<'_>) -> Option<&'b mut [Self]> {
                match write {
                    DenseWrite::$variant(view) => Some(compact_mut(view)),
                    _ => None,
                }
            }
            fn slice(tensor: &DenseTensor) -> Result<&[Self], DenseError> {
                tensor.$slice()
            }
            fn into_vec(tensor: DenseTensor) -> Result<Vec<Self>, DenseError> {
                tensor.$into_vec()
            }
            fn spectrum(tensor: &DenseTensor) -> Result<&[$real], DenseError> {
                tensor.$spectrum()
            }
        }
    };
}

element!(
    f32,
    f32,
    F32,
    as_f32_slice,
    into_f32_vec,
    as_f32_slice,
    |v| v,
    |v| v,
    |x, s| x * s
);
element!(
    f64,
    f64,
    F64,
    as_f64_slice,
    into_f64_vec,
    as_f64_slice,
    |v| v,
    |v| v,
    |x, s| x * s
);
element!(
    Complex32,
    f32,
    C32,
    as_c32_slice,
    into_c32_vec,
    as_f32_slice,
    |v| Complex32::conj(&v),
    |v| Complex32::new(v.re, 0.0),
    |x, s| x * s
);
element!(
    Complex64,
    f64,
    C64,
    as_c64_slice,
    into_c64_vec,
    as_f64_slice,
    |v| Complex64::conj(&v),
    |v| Complex64::new(v.re, 0.0),
    |x, s| x * s
);

/// The compact payload of a validated destination view.
pub(crate) fn compact_mut<'b, T>(view: &'b mut DenseViewMut<'_, T>) -> &'b mut [T] {
    let len = view.shape().iter().product::<usize>();
    let offset = view.offset();
    &mut view.data_mut()[offset..offset + len]
}

/// Writes the conjugate transpose of each `rows x cols` column-major item of
/// `source` into the matching `cols x rows` item of `dest`.
pub(crate) fn adjoint_items<T: Element>(source: &[T], rows: usize, cols: usize, dest: &mut [T]) {
    map_transposed_items(source, rows, cols, dest, T::conj);
}

/// Writes `map` of each entry of each `rows x cols` column-major item of
/// `source` into the transposed position of the matching `cols x rows`
/// item of `dest`.
pub(crate) fn map_transposed_items<T: Copy>(
    source: &[T],
    rows: usize,
    cols: usize,
    dest: &mut [T],
    map: impl Fn(T) -> T,
) {
    let item = rows * cols;
    if item == 0 {
        return;
    }
    for (source, dest) in source.chunks_exact(item).zip(dest.chunks_exact_mut(item)) {
        for col in 0..cols {
            for row in 0..rows {
                dest[col + cols * row] = map(source[row + rows * col]);
            }
        }
    }
}

/// Places one owned output into its destination view: the shape check of
/// the per-matrix entry's caller against the computed (pre-placement)
/// shape, then a copy, or the adjoint copy under `Adjoint`.
fn place_one<T: Element>(
    source: &[T],
    source_shape: &[usize],
    view: &mut DenseViewMut<'_, T>,
    adjoint_matrix: bool,
    label: &'static str,
) -> Result<(), DenseError> {
    let shape = view.shape();
    let computed: &[usize] = shape;
    if adjoint_matrix {
        let (rows, cols) = (shape[1], shape[0]);
        if source_shape.len() != shape.len()
            || source_shape[0] != rows
            || source_shape[1] != cols
            || source_shape[2..] != shape[2..]
        {
            let mut expected = shape.to_vec();
            expected.swap(0, 1);
            return Err(DenseError::ShapeMismatch {
                op: label,
                expected,
                actual: source_shape.to_vec(),
            });
        }
        check_contiguous_output(label, source.len(), source_shape, source_shape)?;
        adjoint_items(source, rows, cols, compact_mut(view));
        return Ok(());
    }
    check_contiguous_output(label, source.len(), source_shape, computed)?;
    compact_mut(view).copy_from_slice(source);
    Ok(())
}

/// Places the owned outputs `tensors` of `op` into `dest`, checking them in
/// output order (count, then per output its dtype and shape) with the errors
/// the matrix-algebra caller reports for a malformed owned outcome.
pub(crate) fn place_owned(
    op: DenseFactorization,
    tensors: Vec<DenseTensor>,
    dest: &mut FactorDest<'_>,
) -> Result<(), DenseError> {
    let label = output_label(op);
    let arity = arity(op)?;
    if tensors.len() != arity {
        return Err(arity_mismatch(label, arity, tensors.len()));
    }
    let adjoint = dest.placement == FactorPlacement::Adjoint;
    if adjoint {
        // The adjoint problem's outputs come in reversed factor order.
        let [first, second, third] = &mut dest.outputs;
        match op {
            DenseFactorization::Svd | DenseFactorization::SvdFull => std::mem::swap(first, third),
            _ => std::mem::swap(first, second),
        }
    }
    let result = (|| {
        for (index, tensor) in tensors.iter().enumerate() {
            let output = dest.outputs[index]
                .as_mut()
                .expect("validation checked the pattern");
            let matrix = adjoint && !is_spectrum(op, index);
            let shape = tensor.shape();
            match output {
                DenseWrite::F32(view) => {
                    place_one(tensor.as_f32_slice()?, shape, view, matrix, label)?
                }
                DenseWrite::F64(view) => {
                    place_one(tensor.as_f64_slice()?, shape, view, matrix, label)?
                }
                DenseWrite::C32(view) => {
                    place_one(tensor.as_c32_slice()?, shape, view, matrix, label)?
                }
                DenseWrite::C64(view) => {
                    place_one(tensor.as_c64_slice()?, shape, view, matrix, label)?
                }
                // Validation skips the outputs of an input the solver refuses;
                // an executor that still returns factors for it does not fit.
                other => {
                    return Err(DenseError::Unsupported {
                        op: label,
                        message: format!("{:?} factorization outputs", other.dtype()),
                    })
                }
            }
        }
        Ok(())
    })();
    if adjoint {
        let [first, second, third] = &mut dest.outputs;
        match op {
            DenseFactorization::Svd | DenseFactorization::SvdFull => std::mem::swap(first, third),
            _ => std::mem::swap(first, second),
        }
    }
    result
}

/// The sink of [`DenseExecutor::factorize_batch_into`].
pub(crate) type FactorSink<'s> =
    dyn FnMut(usize, &mut FactorDest<'_>, Result<(), FactorFailure>) -> Result<(), DenseError> + 's;

/// [`DenseExecutor::factorize_batch_into`]'s default: [`validate`], then
/// [`factorize_batch_into_validated`].
pub(crate) fn factorize_batch_into_default<E: DenseExecutor + ?Sized>(
    executor: &mut E,
    engine: &mut dyn Engine,
    op: DenseFactorization,
    inputs: &[DenseRead<'_>],
    dests: &mut [FactorDest<'_>],
    sink: &mut FactorSink<'_>,
) -> Result<(), DenseError> {
    validate(op, inputs, dests)?;
    factorize_batch_into_validated(executor, engine, op, inputs, dests, sink)
}

/// The default destination batch over validated `dests`: a destination-only
/// op through [`factorize_batch_into_composite`], any other through the
/// executor's owned batch.
pub(crate) fn factorize_batch_into_validated<E: DenseExecutor + ?Sized>(
    executor: &mut E,
    engine: &mut dyn Engine,
    op: DenseFactorization,
    inputs: &[DenseRead<'_>],
    dests: &mut [FactorDest<'_>],
    sink: &mut FactorSink<'_>,
) -> Result<(), DenseError> {
    if op.destination_only() {
        factorize_batch_into_composite(executor, engine, op, inputs, dests, sink)
    } else {
        factorize_batch_into_owned(executor, engine, op, inputs, dests, sink)
    }
}

/// The executor's owned [`DenseExecutor::factorize_batch`] over validated
/// `dests`, each delivered outcome placed into its destination before `sink`
/// sees it.
fn factorize_batch_into_owned<E: DenseExecutor + ?Sized>(
    executor: &mut E,
    engine: &mut dyn Engine,
    op: DenseFactorization,
    inputs: &[DenseRead<'_>],
    dests: &mut [FactorDest<'_>],
    sink: &mut FactorSink<'_>,
) -> Result<(), DenseError> {
    let mut next = 0;
    let mut misdelivered = None;
    let batch = executor.factorize_batch(engine, op, inputs, &mut |index, outcome| {
        // The delivery guard runs before any destination is touched.
        if index != next || index >= dests.len() {
            let error = arity_mismatch("factorize_batch", next, index);
            misdelivered = Some(error.clone());
            return Err(error);
        }
        next += 1;
        let dest = &mut dests[index];
        let result = match outcome {
            Err(error) => Err(FactorFailure::Solver(error)),
            Ok(tensors) => place_owned(op, tensors, dest).map_err(FactorFailure::Output),
        };
        sink(index, dest, result)
    });
    match misdelivered {
        Some(error) => Err(error),
        None => batch,
    }
}

/// The destination-only ops over validated `dests`, one input after another
/// through the executor's per-matrix entries, every product written straight
/// into the destination; `sink(i, ..)` runs before input `i + 1` is solved.
///
/// - Polar: [`DenseExecutor::svd`], the `W` product, the `sqrt(S)` scaling
///   of the owned `Vh` (left) or `U` (right) in place, the `P` product
///   reading that scaled factor, and the Hermitian projection of `P`.
/// - Hermitian exponential: [`DenseExecutor::eigh`], the finite-eigenvalue
///   check, the `exp(lambda / 2)` column scaling of the owned `V` in place,
///   the product `exp = Y Y^H` of that scaled factor, and `copytri`.
///
/// A solver or product failure is [`FactorFailure::Solver`]; owned outputs
/// that do not fit (`svd_into`'s or `eigh_into`'s checks) and a nonfinite
/// eigenvalue are [`FactorFailure::Output`].
pub(crate) fn factorize_batch_into_composite<E: DenseExecutor + ?Sized>(
    executor: &mut E,
    engine: &mut dyn Engine,
    op: DenseFactorization,
    inputs: &[DenseRead<'_>],
    dests: &mut [FactorDest<'_>],
    sink: &mut FactorSink<'_>,
) -> Result<(), DenseError> {
    for (index, (&input, dest)) in inputs.iter().zip(dests.iter_mut()).enumerate() {
        let shape = input.shape();
        let outcome = match input {
            DenseRead::F32(_) if shape.len() >= 2 => {
                composite_into::<_, f32>(executor, engine, op, input, shape, dest)
            }
            DenseRead::F64(_) if shape.len() >= 2 => {
                composite_into::<_, f64>(executor, engine, op, input, shape, dest)
            }
            DenseRead::C32(_) if shape.len() >= 2 => {
                composite_into::<_, Complex32>(executor, engine, op, input, shape, dest)
            }
            DenseRead::C64(_) if shape.len() >= 2 => {
                composite_into::<_, Complex64>(executor, engine, op, input, shape, dest)
            }
            other => provider_refusal(executor, engine, op, other),
        };
        sink(index, dest, outcome)?;
    }
    Ok(())
}

/// The outcome of an input the composite cannot factor (a rank, dtype or,
/// for the exponential, non-square shape): the provider entry owns its
/// refusal, which is [`FactorFailure::Solver`]; an entry that accepts it
/// anyway leaves outputs the composite cannot place.
fn provider_refusal<E: DenseExecutor + ?Sized>(
    executor: &mut E,
    engine: &mut dyn Engine,
    op: DenseFactorization,
    input: DenseRead<'_>,
) -> Result<(), FactorFailure> {
    let (solved, label) = if op == DenseFactorization::ExpViaEigh {
        (executor.eigh(engine, input), "eigh_into")
    } else {
        (executor.svd(engine, input), "svd_into")
    };
    solved.map_err(FactorFailure::Solver).and_then(|_| {
        Err(FactorFailure::Output(DenseError::Unsupported {
            op: label,
            message: format!(
                "{op:?} input of dtype {:?} and shape {:?}",
                input.dtype(),
                input.shape()
            ),
        }))
    })
}

/// [`factorize_batch_into_composite`] of one floating input of rank two or
/// more and `shape`.
fn composite_into<E: DenseExecutor + ?Sized, T: Element>(
    executor: &mut E,
    engine: &mut dyn Engine,
    op: DenseFactorization,
    input: DenseRead<'_>,
    shape: &[usize],
    dest: &mut FactorDest<'_>,
) -> Result<(), FactorFailure> {
    if op != DenseFactorization::ExpViaEigh {
        let left = op == DenseFactorization::LeftPolar;
        return polar_into::<_, T>(executor, engine, left, input, shape, dest);
    }
    if shape[0] != shape[1] {
        return provider_refusal(executor, engine, op, input);
    }
    exp_into::<_, T>(executor, engine, input, shape, dest)
}

/// The refusal of a polar input in the wrong direction: left polar needs
/// `m >= n`, right polar `n >= m`. Checked after the rank and before the
/// zero-extent shortcut.
pub(crate) fn polar_direction(left: bool, [m, n]: [usize; 2]) -> Result<(), DenseError> {
    if left && m >= n || !left && n >= m {
        return Ok(());
    }
    let (long, short) = (m.max(n), m.min(n));
    Err(DenseError::ShapeMismatch {
        op: if left { "left_polar" } else { "right_polar" },
        expected: if left {
            vec![long, short]
        } else {
            vec![short, long]
        },
        actual: vec![m, n],
    })
}

/// One owned output of shape `expected`, checked as the per-matrix entry
/// `label` checks it: the borrowed dtype, then shape and length, and only
/// then the consuming transfer, so a malformed output carries the same error
/// as the per-matrix check.
fn owned_factor<T: Element>(
    tensor: DenseTensor,
    expected: &[usize],
    label: &'static str,
) -> Result<Vec<T>, DenseError> {
    let len = T::slice(&tensor)?.len();
    check_contiguous_output(label, len, tensor.shape(), expected)?;
    T::into_vec(tensor)
}

/// One owned spectrum of shape `expected`, checked as the per-matrix entry
/// `label` checks it (dtype, then shape and length).
fn owned_spectrum<'t, T: Element>(
    tensor: &'t DenseTensor,
    expected: &[usize],
    label: &'static str,
) -> Result<&'t [T::Spectrum], DenseError> {
    let values = T::spectrum(tensor)?;
    check_contiguous_output(label, values.len(), tensor.shape(), expected)?;
    Ok(values)
}

/// [`factorize_batch_into_composite`] of one floating input of rank two or
/// more and `shape`.
fn polar_into<E: DenseExecutor + ?Sized, T: Element>(
    executor: &mut E,
    engine: &mut dyn Engine,
    left: bool,
    input: DenseRead<'_>,
    shape: &[usize],
    dest: &mut FactorDest<'_>,
) -> Result<(), FactorFailure> {
    let (m, n, batch) = (shape[0], shape[1], &shape[2..]);
    polar_direction(left, [m, n]).map_err(FactorFailure::Solver)?;
    if shape.contains(&0) {
        return Ok(());
    }
    let outputs = executor.svd(engine, input).map_err(FactorFailure::Solver)?;
    let k = m.min(n);
    let dims = |a: usize, b: usize| [&[a, b][..], batch].concat();
    let [u, s, vh]: [DenseTensor; 3] = outputs
        .try_into()
        .map_err(|outputs: Vec<_>| arity_mismatch("svd_into", 3, outputs.len()))
        .map_err(FactorFailure::Output)?;
    let mut u = owned_factor::<T>(u, &dims(m, k), "svd_into").map_err(FactorFailure::Output)?;
    let sigma = owned_spectrum::<T>(&s, &[&[k][..], batch].concat(), "svd_into")
        .map_err(FactorFailure::Output)?;
    let mut vh = owned_factor::<T>(vh, &dims(k, n), "svd_into").map_err(FactorFailure::Output)?;
    // A nonempty view's element count fits, so its batch product does.
    let items = batch.iter().product::<usize>();
    let [first, second, _] = &mut dest.outputs;
    let (w, p) = if left {
        (first, second)
    } else {
        (second, first)
    };
    let (w, p) = (
        w.as_mut()
            .and_then(T::compact)
            .expect("validation checked the destination dtype"),
        p.as_mut()
            .and_then(T::compact)
            .expect("validation checked the destination dtype"),
    );
    let order = if left { n } else { m };
    let product = DenseDotConfig::matmul();
    let gram = if left {
        DenseDotConfig::matmul().with_conjugation(true, false)
    } else {
        DenseDotConfig::matmul().with_conjugation(false, true)
    };
    // Column-major layouts: `[shape, strides]` of the `W` product's output
    // and operands, then of the `P` product's (its adjoint operand is the
    // factor read transposed, conjugated by `gram`).
    let w_layout = [[m, n], [1, m]];
    let (u_layout, vh_layout) = ([[m, k], [1, m]], [[k, n], [1, k]]);
    let p_layout = [[order, order], [1, order]];
    let (p_lhs, p_rhs) = if left {
        ([[n, k], [k, 1]], vh_layout)
    } else {
        (u_layout, [[k, m], [m, 1]])
    };
    for item in 0..items {
        let (u, vh) = (&u[item * m * k..][..m * k], &vh[item * k * n..][..k * n]);
        let w = &mut w[item * m * n..][..m * n];
        composite_product(
            executor,
            engine,
            w,
            &w_layout,
            (u, &u_layout),
            (vh, &vh_layout),
            &product,
        )?;
    }
    scale_polar_factor(left, [m, n], items, |index| sigma[index], &mut u, &mut vh);
    for item in 0..items {
        let x = if left {
            &vh[item * k * n..][..k * n]
        } else {
            &u[item * m * k..][..m * k]
        };
        let p = &mut p[item * order * order..][..order * order];
        composite_product(
            executor,
            engine,
            p,
            &p_layout,
            (x, &p_lhs),
            (x, &p_rhs),
            &gram,
        )?;
    }
    project_hermitian_items(p, order);
    Ok(())
}

/// [`factorize_batch_into_composite`] of one floating square input of rank
/// two or more and `shape`.
fn exp_into<E: DenseExecutor + ?Sized, T: Element>(
    executor: &mut E,
    engine: &mut dyn Engine,
    input: DenseRead<'_>,
    shape: &[usize],
    dest: &mut FactorDest<'_>,
) -> Result<(), FactorFailure> {
    let (n, batch) = (shape[0], &shape[2..]);
    if shape.contains(&0) {
        return Ok(());
    }
    let outputs = executor
        .eigh(engine, input)
        .map_err(FactorFailure::Solver)?;
    let [values, vectors]: [DenseTensor; 2] = outputs
        .try_into()
        .map_err(|outputs: Vec<_>| arity_mismatch("eigh_into", 2, outputs.len()))
        .map_err(FactorFailure::Output)?;
    let lambda = owned_spectrum::<T>(&values, &[&[n][..], batch].concat(), "eigh_into")
        .map_err(FactorFailure::Output)?;
    let mut x = owned_factor::<T>(vectors, &[&[n, n][..], batch].concat(), "eigh_into")
        .map_err(FactorFailure::Output)?;
    finite_eigenvalues(lambda.iter().copied()).map_err(FactorFailure::Output)?;
    // A nonempty view's element count fits, so its batch product does.
    let items = batch.iter().product::<usize>();
    scale_exp_factor(n, items, |index| lambda[index], &mut x);
    let out = dest.outputs[0]
        .as_mut()
        .and_then(T::compact)
        .expect("validation checked the destination dtype");
    let gram = DenseDotConfig::matmul().with_conjugation(false, true);
    let (layout, adjoint) = ([[n, n], [1, n]], [[n, n], [n, 1]]);
    for item in 0..items {
        let x = &x[item * n * n..][..n * n];
        let out = &mut out[item * n * n..][..n * n];
        composite_product(
            executor,
            engine,
            out,
            &layout,
            (x, &layout),
            (x, &adjoint),
            &gram,
        )?;
    }
    copytri_items(out, n);
    Ok(())
}

/// One product of the composite into the column-major `out`, each operand
/// a whole slice with its `[shape, strides]`.
fn composite_product<E: DenseExecutor + ?Sized, T: Element>(
    executor: &mut E,
    engine: &mut dyn Engine,
    out: &mut [T],
    [out_shape, out_strides]: &[[usize; 2]; 2],
    (lhs, [lhs_shape, lhs_strides]): (&[T], &[[usize; 2]; 2]),
    (rhs, [rhs_shape, rhs_strides]): (&[T], &[[usize; 2]; 2]),
    config: &DenseDotConfig,
) -> Result<(), FactorFailure> {
    let output = FactorFailure::Output;
    let out = T::write(DenseViewMut::new(out, out_shape, out_strides, 0).map_err(output)?);
    let lhs = T::read(DenseView::new(lhs, lhs_shape, lhs_strides, 0).map_err(output)?);
    let rhs = T::read(DenseView::new(rhs, rhs_shape, rhs_strides, 0).map_err(output)?);
    executor
        .dot_general_into(engine, out, lhs, rhs, config)
        .map_err(FactorFailure::Solver)
}

/// Scales each item's `Vh` rows (left) or `U` columns (right) by
/// `sqrt(sigma(i))` in place, `i` the singular value's index over the whole
/// batch: `X = sqrt(S) Vh` or `U sqrt(S)`, the factor of `P = X^H X` or
/// `X X^H`.
pub(crate) fn scale_polar_factor<T: Element>(
    left: bool,
    [m, n]: [usize; 2],
    items: usize,
    sigma: impl Fn(usize) -> T::Spectrum,
    u: &mut [T],
    vh: &mut [T],
) {
    let k = m.min(n);
    for item in 0..items {
        if left {
            let vh = &mut vh[item * k * n..][..k * n];
            for row in 0..k {
                let sigma = sigma(item * k + row);
                for col in 0..n {
                    vh[row + k * col] = vh[row + k * col].scale_by_sqrt(sigma);
                }
            }
        } else {
            let u = &mut u[item * m * k..][..m * k];
            for col in 0..k {
                let sigma = sigma(item * k + col);
                for value in &mut u[m * col..m * (col + 1)] {
                    *value = value.scale_by_sqrt(sigma);
                }
            }
        }
    }
}

/// Scales each item's `n x n` eigenvectors `x` column by column by
/// `exp(lambda(i) / 2)` in place, `i` the eigenvalue's index over the whole
/// batch: `Y = V exp(D / 2)`, the factor of `exp(X) = Y Y^H`.
pub(crate) fn scale_exp_factor<T: Element>(
    n: usize,
    items: usize,
    lambda: impl Fn(usize) -> T::Spectrum,
    x: &mut [T],
) {
    for item in 0..items {
        let x = &mut x[item * n * n..][..n * n];
        for col in 0..n {
            let lambda = lambda(item * n + col);
            for value in &mut x[n * col..n * (col + 1)] {
                *value = value.scale_by_half_exp(lambda);
            }
        }
    }
}

/// Makes each column-major `n x n` item of `data` exactly Hermitian as
/// LAPACK `copytri` (`'U'`, conjugating) publishes it: the upper triangle
/// kept, the diagonal its real part, the lower triangle its adjoint.
pub(crate) fn copytri_items<T: Element>(data: &mut [T], n: usize) {
    if n == 0 {
        return;
    }
    for item in data.chunks_exact_mut(n * n) {
        for col in 0..n {
            item[col + n * col] = item[col + n * col].real_part();
            for row in 0..col {
                item[col + n * row] = item[row + n * col].conj();
            }
        }
    }
}

/// Replaces each column-major `n x n` item of `data` by its Hermitian part
/// `(A + A^H) / 2`, so `P` is exactly Hermitian whatever the product's
/// rounding.
pub(crate) fn project_hermitian_items<T: Element>(data: &mut [T], n: usize) {
    if n == 0 {
        return;
    }
    for item in data.chunks_exact_mut(n * n) {
        for col in 0..n {
            for row in 0..=col {
                let value = item[row + n * col].hermitian_mean(item[col + n * row]);
                item[row + n * col] = value;
                item[col + n * row] = value.conj();
            }
        }
    }
}
