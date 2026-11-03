//! Caller-owned destinations of one batched factorization
//! ([`crate::DenseExecutor::factorize_batch_into`]): their validation, the
//! placement of a factorization's outputs into them, and the compatibility
//! route through the owned batch.

use num_complex::{Complex32, Complex64};

use crate::executor::arity_mismatch;
use crate::{
    check_contiguous_output, DenseDType, DenseError, DenseExecutor, DenseFactorization, DenseRead,
    DenseTensor, DenseViewMut, DenseWrite, Engine,
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
    /// `A^H` (`u = Vh^H`, `s`, `vt = U^H`). Hermitian eigendecompositions
    /// refuse it.
    Adjoint,
}

/// Why one input's factorization failed, kept apart so a caller applies its
/// own precedence between solver failures and malformed outputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FactorFailure {
    /// The solver failed: the provider, or the executor's owned entry,
    /// reported this error for the input.
    Solver(DenseError),
    /// The solver returned, but its outputs do not fit the destination.
    Output(DenseError),
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
/// owned output order: `Qr` `[q, r, -]`, `Svd` `[u, s, vt]`, `Eigh`
/// `[values, vectors, -]` (`-` is `None`).
///
/// Each factor view has the input's dtype and each spectrum (`s`, eigh
/// `values`) the input's real counterpart (`f32` for `f32`/`c32`, `f64` for
/// `f64`/`c64`), exactly what the per-matrix entry returns. Every view is
/// compact column-major with the output's shape, which [`FactorPlacement`]
/// sets: for an `m x n` input with trailing batch dimensions `b..` and
/// `k = min(m, n)`, `AsComputed` QR is `[m, k, b..]`, `[k, n, b..]` and
/// `Adjoint` QR `[n, k, b..]`, `[k, m, b..]`; SVD adds `s` of shape
/// `[k, b..]` between them; eigh of `n x n` is `[n, b..]`, `[n, n, b..]`.
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
        DenseFactorization::Qr | DenseFactorization::Eigh => Ok(2),
        DenseFactorization::Svd => Ok(3),
        _ => Err(DenseError::Unsupported {
            op: ENTRY,
            message: format!("{op:?} has no destination form"),
        }),
    }
}

/// The per-matrix entry label of `op`'s malformed-output errors.
fn output_label(op: DenseFactorization) -> &'static str {
    match op {
        DenseFactorization::Svd => "svd_into",
        DenseFactorization::Eigh => "eigh_into",
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
        (DenseFactorization::Svd, 1, _) => ([k, 0], 1),
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
        (DenseFactorization::Eigh, 0) | (DenseFactorization::Svd, 1)
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
        if op == DenseFactorization::Eigh && dest.placement == FactorPlacement::Adjoint {
            return Err(DenseError::Unsupported {
                op: ENTRY,
                message: "a Hermitian eigendecomposition has no adjoint placement".to_string(),
            });
        }
        let dtype = input.dtype();
        let Some(real) = real_of(dtype) else {
            continue;
        };
        let [m, n, batch @ ..] = input.shape() else {
            continue;
        };
        if op == DenseFactorization::Eigh && m != n {
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

/// A scalar a destination holds.
pub(crate) trait Element: Copy {
    fn conj(self) -> Self;
}

impl Element for f32 {
    fn conj(self) -> Self {
        self
    }
}

impl Element for f64 {
    fn conj(self) -> Self {
        self
    }
}

impl Element for Complex32 {
    fn conj(self) -> Self {
        Complex32::conj(&self)
    }
}

impl Element for Complex64 {
    fn conj(self) -> Self {
        Complex64::conj(&self)
    }
}

/// The compact payload of a validated destination view.
pub(crate) fn compact_mut<'b, T>(view: &'b mut DenseViewMut<'_, T>) -> &'b mut [T] {
    let len = view.shape().iter().product::<usize>();
    let offset = view.offset();
    &mut view.data_mut()[offset..offset + len]
}

/// Writes the conjugate transpose of each `rows x cols` column-major item of
/// `source` into the matching `cols x rows` item of `dest`.
pub(crate) fn adjoint_items<T: Element>(source: &[T], rows: usize, cols: usize, dest: &mut [T]) {
    let item = rows * cols;
    if item == 0 {
        return;
    }
    for (source, dest) in source.chunks_exact(item).zip(dest.chunks_exact_mut(item)) {
        for col in 0..cols {
            for row in 0..rows {
                dest[col + cols * row] = source[row + rows * col].conj();
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
            DenseFactorization::Svd => std::mem::swap(first, third),
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
            DenseFactorization::Svd => std::mem::swap(first, third),
            _ => std::mem::swap(first, second),
        }
    }
    result
}

/// The sink of [`DenseExecutor::factorize_batch_into`].
pub(crate) type FactorSink<'s> =
    dyn FnMut(usize, &mut FactorDest<'_>, Result<(), FactorFailure>) -> Result<(), DenseError> + 's;

/// [`DenseExecutor::factorize_batch_into`]'s default: the executor's owned
/// [`DenseExecutor::factorize_batch`], each delivered outcome placed into its
/// destination before `sink` sees it.
pub(crate) fn factorize_batch_into_owned<E: DenseExecutor + ?Sized>(
    executor: &mut E,
    engine: &mut dyn Engine,
    op: DenseFactorization,
    inputs: &[DenseRead<'_>],
    dests: &mut [FactorDest<'_>],
    sink: &mut FactorSink<'_>,
) -> Result<(), DenseError> {
    validate(op, inputs, dests)?;
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
