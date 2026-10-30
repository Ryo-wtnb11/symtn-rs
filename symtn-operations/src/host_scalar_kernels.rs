use core::ops::{Add, Mul};

use num_traits::{One, Zero};

use strided_kernel::{
    axpy_conj_raw, axpy_raw, copy_scale_conj_raw, copy_scale_raw, CopyPlan, RawStridedMut,
    RawStridedRef, RAW_FUSED_RANK_LIMIT,
};

use smallvec::SmallVec;

use crate::kernel_adapter::{
    apply_fused_pair_slices, normalize_fused_layout, FusedLayoutScratch, FUSED_LAYOUT_INLINE_RANK,
};
use crate::scalar::scale_value;
use crate::strided::error as strided_error;
use crate::{ConjugateValue, OperationError};

/// Host scalar strided kernel boundary.
///
/// This module owns the current host-slice scalar kernels used by tensoradd,
/// pack, scatter, and scale replay. Higher-level tree/fusion algorithms should
/// call these primitives instead of embedding raw strided loops directly.
#[allow(clippy::too_many_arguments)]
pub fn tensoradd_raw_strided_kernel<T>(
    zero_strides: &mut Vec<isize>,
    dst_data: &mut [T],
    src_data: &[T],
    shape: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
    source_conjugate: bool,
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + Send
        + Sync,
{
    if source_conjugate {
        return tensoradd_raw_strided_conjugating_kernel(
            zero_strides,
            dst_data,
            src_data,
            shape,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            alpha,
            beta,
        );
    }
    zero_strides.clear();
    axpby_raw_strided_kernel(
        dst_data,
        src_data,
        shape,
        dst_strides,
        src_strides,
        dst_offset,
        src_offset,
        alpha,
        beta,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn tensoradd_raw_strided_kernel_trusted<T>(
    zero_strides: &mut Vec<isize>,
    dst_data: &mut [T],
    src_data: &[T],
    shape: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
    source_conjugate: bool,
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + Send
        + Sync,
{
    if source_conjugate {
        return tensoradd_raw_strided_conjugating_kernel_trusted(
            zero_strides,
            dst_data,
            src_data,
            shape,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            alpha,
            beta,
        );
    }
    zero_strides.clear();
    axpby_raw_strided_kernel_trusted(
        dst_data,
        src_data,
        shape,
        dst_strides,
        src_strides,
        dst_offset,
        src_offset,
        alpha,
        beta,
    )
}

/// Conjugated dot product over one strided block, accumulated in
/// [`WideScalar::Wide`].
///
/// The sum runs over the whole block, so its length is the payload's, not a
/// constant: a single-precision accumulator would lose digits in proportion to
/// the block and saturate near `1.8e19`. `Wide` is the payload type itself for
/// `f64`/`Complex64`, so this is the same arithmetic it always performed.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn bilinear_raw_strided_kernel_mapped<T, L, R>(
    lhs_data: &[T],
    rhs_data: &[T],
    shape: &[usize],
    lhs_stride: L,
    rhs_stride: R,
    lhs_offset: isize,
    rhs_offset: isize,
    lhs_conjugate: bool,
    rhs_conjugate: bool,
) -> Result<T::Wide, OperationError>
where
    T: crate::WideScalar,
    L: Copy + Fn(usize) -> Result<isize, OperationError>,
    R: Copy + Fn(usize) -> Result<isize, OperationError>,
{
    validate_raw_strided_bounds_mapped(lhs_data.len(), shape, lhs_stride, lhs_offset)?;
    validate_raw_strided_bounds_mapped(rhs_data.len(), shape, rhs_stride, rhs_offset)?;
    let len = crate::strided::element_count(shape)?;
    if len == 0 {
        return Ok(T::Wide::zero());
    }
    if shape.is_empty() {
        return Ok(lhs_data[checked_offset_to_index(lhs_offset)?]
            .widen()
            .maybe_conj(lhs_conjugate)
            * rhs_data[checked_offset_to_index(rhs_offset)?]
                .widen()
                .maybe_conj(rhs_conjugate));
    }
    bilinear_raw_strided_recurse_mapped(
        shape.len() - 1,
        lhs_data,
        rhs_data,
        shape,
        lhs_stride,
        rhs_stride,
        lhs_offset,
        rhs_offset,
        lhs_conjugate,
        rhs_conjugate,
    )
}

fn validate_raw_strided_bounds_mapped<F>(
    len: usize,
    shape: &[usize],
    stride: F,
    offset: isize,
) -> Result<(), OperationError>
where
    F: Fn(usize) -> Result<isize, OperationError>,
{
    if shape.contains(&0) {
        return Ok(());
    }
    let mut min_offset = offset;
    let mut max_offset = offset;
    for (axis, &dim) in shape.iter().enumerate() {
        if dim <= 1 {
            continue;
        }
        let dim = isize::try_from(dim - 1).map_err(|_| OperationError::ElementCountOverflow)?;
        let end = stride(axis)?
            .checked_mul(dim)
            .ok_or_else(|| OperationError::ElementCountOverflow)?;
        if end >= 0 {
            max_offset = max_offset
                .checked_add(end)
                .ok_or_else(|| OperationError::ElementCountOverflow)?;
        } else {
            min_offset = min_offset
                .checked_add(end)
                .ok_or_else(|| OperationError::ElementCountOverflow)?;
        }
    }
    if min_offset < 0 {
        return Err(OperationError::OffsetOverflow { value: usize::MAX });
    }
    let max_offset = checked_offset_to_index(max_offset)?;
    if max_offset >= len {
        return Err(OperationError::OffsetOverflow { value: max_offset });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn bilinear_raw_strided_recurse_mapped<T, L, R>(
    axis: usize,
    lhs_data: &[T],
    rhs_data: &[T],
    shape: &[usize],
    lhs_stride: L,
    rhs_stride: R,
    lhs_base: isize,
    rhs_base: isize,
    lhs_conjugate: bool,
    rhs_conjugate: bool,
) -> Result<T::Wide, OperationError>
where
    T: crate::WideScalar,
    L: Copy + Fn(usize) -> Result<isize, OperationError>,
    R: Copy + Fn(usize) -> Result<isize, OperationError>,
{
    let mut sum = T::Wide::zero();
    if axis == 0 {
        let lhs_stride = lhs_stride(0)?;
        let rhs_stride = rhs_stride(0)?;
        for index in 0..shape[0] {
            let lhs_index =
                checked_offset_to_index(checked_strided_offset(lhs_base, index, lhs_stride)?)?;
            let rhs_index =
                checked_offset_to_index(checked_strided_offset(rhs_base, index, rhs_stride)?)?;
            sum = sum
                + lhs_data[lhs_index].widen().maybe_conj(lhs_conjugate)
                    * rhs_data[rhs_index].widen().maybe_conj(rhs_conjugate);
        }
        return Ok(sum);
    }
    let lhs_axis_stride = lhs_stride(axis)?;
    let rhs_axis_stride = rhs_stride(axis)?;
    for index in 0..shape[axis] {
        sum = sum
            + bilinear_raw_strided_recurse_mapped(
                axis - 1,
                lhs_data,
                rhs_data,
                shape,
                lhs_stride,
                rhs_stride,
                checked_strided_offset(lhs_base, index, lhs_axis_stride)?,
                checked_strided_offset(rhs_base, index, rhs_axis_stride)?,
                lhs_conjugate,
                rhs_conjugate,
            )?;
    }
    Ok(sum)
}

#[allow(clippy::too_many_arguments)]
fn tensoradd_raw_strided_conjugating_kernel<T>(
    zero_strides: &mut Vec<isize>,
    dst_data: &mut [T],
    src_data: &[T],
    shape: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + Send
        + Sync,
{
    validate_raw_strided_views(
        dst_data,
        src_data,
        shape,
        dst_strides,
        src_strides,
        dst_offset,
        src_offset,
    )?;
    raw_strided_combine_loop(
        dst_data,
        src_data,
        shape,
        dst_strides,
        src_strides,
        dst_offset,
        src_offset,
        true,
        raw_strided_action(alpha, beta),
    )?;
    zero_strides.clear();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn tensoradd_raw_strided_conjugating_kernel_trusted<T>(
    zero_strides: &mut Vec<isize>,
    dst_data: &mut [T],
    src_data: &[T],
    shape: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + Send
        + Sync,
{
    #[cfg(debug_assertions)]
    validate_raw_strided_views(
        dst_data,
        src_data,
        shape,
        dst_strides,
        src_strides,
        dst_offset,
        src_offset,
    )?;
    raw_strided_combine_loop(
        dst_data,
        src_data,
        shape,
        dst_strides,
        src_strides,
        dst_offset,
        src_offset,
        true,
        raw_strided_action(alpha, beta),
    )?;
    zero_strides.clear();
    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum RawStridedAction<T> {
    /// `alpha == 1, beta == 0`: bit-exact copy (or conjugate). Why not
    /// `CopyScale { alpha: 1 }`: `1 * (inf + 0i)` is `inf + NaN i` for complex
    /// scalars, so a plain copy must not multiply (TensorKit skips the scale
    /// for `One()` as well).
    Copy,
    CopyScale {
        alpha: T,
    },
    Axpy {
        alpha: T,
    },
    Axpby {
        alpha: T,
        beta: T,
    },
}

pub(crate) fn raw_strided_action<T>(alpha: T, beta: T) -> RawStridedAction<T>
where
    T: Copy + PartialEq + Zero + One,
{
    if beta.is_zero() {
        if alpha.is_one() {
            RawStridedAction::Copy
        } else {
            RawStridedAction::CopyScale { alpha }
        }
    } else if beta.is_one() {
        RawStridedAction::Axpy { alpha }
    } else {
        RawStridedAction::Axpby { alpha, beta }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn axpby_raw_strided_kernel<T>(
    dst_data: &mut [T],
    src_data: &[T],
    shape: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + Send
        + Sync,
{
    validate_raw_strided_views(
        dst_data,
        src_data,
        shape,
        dst_strides,
        src_strides,
        dst_offset,
        src_offset,
    )?;
    raw_strided_combine_loop(
        dst_data,
        src_data,
        shape,
        dst_strides,
        src_strides,
        dst_offset,
        src_offset,
        false,
        raw_strided_action(alpha, beta),
    )
}

/// `dst = alpha * src + beta * dst` over raw strided views the caller has
/// already proved in bounds; only debug builds re-check them.
///
/// # Panics
///
/// If a reachable offset lies outside `dst_data` or `src_data`, a release
/// build panics on bounds-checked slice indexing (possibly after partial
/// writes); it is never undefined behavior.
#[allow(clippy::too_many_arguments)]
pub fn axpby_raw_strided_kernel_trusted<T>(
    dst_data: &mut [T],
    src_data: &[T],
    shape: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
    alpha: T,
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + PartialEq
        + Zero
        + One
        + ConjugateValue
        + Send
        + Sync,
{
    #[cfg(debug_assertions)]
    validate_raw_strided_views(
        dst_data,
        src_data,
        shape,
        dst_strides,
        src_strides,
        dst_offset,
        src_offset,
    )?;
    raw_strided_combine_loop(
        dst_data,
        src_data,
        shape,
        dst_strides,
        src_strides,
        dst_offset,
        src_offset,
        false,
        raw_strided_action(alpha, beta),
    )
}

/// `dst = beta * dst` over a raw strided view the caller has already proved
/// in bounds; only debug builds re-check it (returning an error before any
/// write).
///
/// # Panics
///
/// If a reachable offset lies outside `dst_data`, a release build either
/// returns [`OperationError::OffsetOverflow`] or panics on bounds-checked
/// slice indexing, in both cases possibly after partial writes; it is never
/// undefined behavior.
pub fn scale_raw_strided_kernel_trusted<T>(
    dst_data: &mut [T],
    shape: &[usize],
    dst_strides: &[isize],
    dst_offset: isize,
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy + Mul<T, Output = T> + Zero,
{
    #[cfg(debug_assertions)]
    validate_raw_strided_bounds(dst_data.len(), shape, dst_strides, dst_offset)?;
    raw_strided_scale_loop(dst_data, shape, dst_strides, dst_offset, beta)
}

/// Adds a coefficient-weighted raw trace after validating all ranks and
/// reachable ranges. Invalid layout metadata is returned before destination
/// storage is changed.
#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn tensortrace_raw_strided_kernel_add_with_coefficient<T, C>(
    dst_data: &mut [T],
    src_data: &[T],
    output_shape: &[usize],
    trace_shape: &[usize],
    dst_strides: &[isize],
    src_output_strides: &[isize],
    src_trace_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
    source_conjugate: bool,
    alpha: T,
    coefficient: C,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + Zero
        + One
        + PartialEq
        + ConjugateValue
        + crate::RecouplingCoefficientAction<C>,
    C: Copy,
{
    let (output_len, trace_len) = validate_tensortrace_raw_layout(
        dst_data.len(),
        src_data.len(),
        output_shape,
        trace_shape,
        dst_strides,
        src_output_strides,
        src_trace_strides,
        dst_offset,
        src_offset,
    )?;
    tensortrace_raw_strided_kernel_add_with_coefficient_loop(
        dst_data,
        src_data,
        output_shape,
        trace_shape,
        dst_strides,
        src_output_strides,
        src_trace_strides,
        dst_offset,
        src_offset,
        source_conjugate,
        alpha,
        coefficient,
        output_len,
        trace_len,
    )
}

/// Executes a coefficient-weighted raw trace layout already admitted by a
/// tensor trace descriptor.
///
/// The caller must provide matching shape/stride ranks and ranges reachable
/// within `dst_data` and `src_data`. Empty traces require representable,
/// nonnegative source output bases but do not require those bases to index
/// `src_data`.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn tensortrace_raw_strided_kernel_add_with_coefficient_trusted<T, C>(
    dst_data: &mut [T],
    src_data: &[T],
    output_shape: &[usize],
    trace_shape: &[usize],
    dst_strides: &[isize],
    src_output_strides: &[isize],
    src_trace_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
    source_conjugate: bool,
    alpha: T,
    coefficient: C,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + Zero
        + One
        + PartialEq
        + ConjugateValue
        + crate::RecouplingCoefficientAction<C>,
    C: Copy,
{
    #[cfg(debug_assertions)]
    let (output_len, trace_len) = validate_tensortrace_raw_layout(
        dst_data.len(),
        src_data.len(),
        output_shape,
        trace_shape,
        dst_strides,
        src_output_strides,
        src_trace_strides,
        dst_offset,
        src_offset,
    )?;
    #[cfg(not(debug_assertions))]
    let (output_len, trace_len) = (
        crate::strided::element_count(output_shape)?,
        crate::strided::element_count(trace_shape)?,
    );
    tensortrace_raw_strided_kernel_add_with_coefficient_loop(
        dst_data,
        src_data,
        output_shape,
        trace_shape,
        dst_strides,
        src_output_strides,
        src_trace_strides,
        dst_offset,
        src_offset,
        source_conjugate,
        alpha,
        coefficient,
        output_len,
        trace_len,
    )
}

#[allow(clippy::too_many_arguments)]
fn tensortrace_raw_strided_kernel_add_with_coefficient_loop<T, C>(
    dst_data: &mut [T],
    src_data: &[T],
    output_shape: &[usize],
    trace_shape: &[usize],
    dst_strides: &[isize],
    src_output_strides: &[isize],
    src_trace_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
    source_conjugate: bool,
    alpha: T,
    coefficient: C,
    output_len: usize,
    trace_len: usize,
) -> Result<(), OperationError>
where
    T: Copy
        + Add<T, Output = T>
        + Mul<T, Output = T>
        + Zero
        + One
        + PartialEq
        + ConjugateValue
        + crate::RecouplingCoefficientAction<C>,
    C: Copy,
{
    // TensorKit's `_trace_permute!` skips a zero coefficient and otherwise
    // folds `α′ = α * coeff`, keeping the coefficient in its own type; then
    // TensorOperations' `stridedtensortrace!` adds `scale(aᵢ, α′)` for every
    // traced element into `C` (`_mapreducedim!(Scaler(α′), Adder(), …)`).
    // Why not scale the traced sum once: `α′ * Σ aᵢ` overflows where
    // `Σ α′ aᵢ` does not, and `0 * inf` is NaN where `scale` gives zero.
    if T::coefficient_as_data(coefficient).is_zero() {
        return Ok(());
    }
    let scale = crate::TransformScale::new(alpha, coefficient);
    for output_linear in 0..output_len {
        let dst_index =
            strided_linear_offset(output_linear, output_shape, dst_strides, dst_offset)?;
        let src_base =
            strided_linear_offset(output_linear, output_shape, src_output_strides, src_offset)?;
        let src_base = isize::try_from(src_base)
            .map_err(|_| OperationError::OffsetOverflow { value: src_base })?;
        let mut accumulator = dst_data[dst_index];
        for trace_linear in 0..trace_len {
            let src_index =
                strided_linear_offset(trace_linear, trace_shape, src_trace_strides, src_base)?;
            accumulator =
                accumulator + scale.scale(src_data[src_index].maybe_conj(source_conjugate));
        }
        dst_data[dst_index] = accumulator;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_tensortrace_raw_layout(
    dst_len: usize,
    src_len: usize,
    output_shape: &[usize],
    trace_shape: &[usize],
    dst_strides: &[isize],
    src_output_strides: &[isize],
    src_trace_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
) -> Result<(usize, usize), OperationError> {
    validate_rank(output_shape, dst_strides)?;
    validate_rank(output_shape, src_output_strides)?;
    validate_rank(trace_shape, src_trace_strides)?;

    let output_len = crate::strided::element_count(output_shape)?;
    let trace_len = crate::strided::element_count(trace_shape)?;
    if output_len == 0 {
        return Ok((output_len, trace_len));
    }

    let (dst_min, dst_max) = checked_strided_extrema(dst_offset, output_shape, dst_strides)?;
    validate_reachable_bounds(dst_len, dst_min, dst_max)?;

    let (src_min, src_max) = checked_strided_extrema(src_offset, output_shape, src_output_strides)?;
    if src_min < 0 {
        return Err(OperationError::OffsetOverflow { value: usize::MAX });
    }
    if trace_len != 0 {
        let (src_min, src_max) =
            checked_strided_extrema_from(src_min, src_max, trace_shape, src_trace_strides)?;
        validate_reachable_bounds(src_len, src_min, src_max)?;
    }
    Ok((output_len, trace_len))
}

fn validate_rank(shape: &[usize], strides: &[isize]) -> Result<(), OperationError> {
    if shape.len() != strides.len() {
        return Err(OperationError::RankMismatch {
            expected: shape.len(),
            actual: strides.len(),
        });
    }
    Ok(())
}

fn checked_strided_extrema(
    offset: isize,
    shape: &[usize],
    strides: &[isize],
) -> Result<(isize, isize), OperationError> {
    checked_strided_extrema_from(offset, offset, shape, strides)
}

fn checked_strided_extrema_from(
    mut min_offset: isize,
    mut max_offset: isize,
    shape: &[usize],
    strides: &[isize],
) -> Result<(isize, isize), OperationError> {
    for (&dim, &stride) in shape.iter().zip(strides.iter()) {
        let coordinate =
            isize::try_from(dim - 1).map_err(|_| OperationError::ElementCountOverflow)?;
        let end = coordinate
            .checked_mul(stride)
            .ok_or_else(|| OperationError::ElementCountOverflow)?;
        if end >= 0 {
            max_offset = max_offset
                .checked_add(end)
                .ok_or_else(|| OperationError::ElementCountOverflow)?;
        } else {
            min_offset = min_offset
                .checked_add(end)
                .ok_or_else(|| OperationError::ElementCountOverflow)?;
        }
    }
    Ok((min_offset, max_offset))
}

fn validate_reachable_bounds(
    len: usize,
    min_offset: isize,
    max_offset: isize,
) -> Result<(), OperationError> {
    if min_offset < 0 {
        return Err(OperationError::OffsetOverflow { value: usize::MAX });
    }
    let max_offset = checked_offset_to_index(max_offset)?;
    if max_offset >= len {
        return Err(OperationError::OffsetOverflow { value: max_offset });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_raw_strided_views<T>(
    dst_data: &mut [T],
    src_data: &[T],
    shape: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
) -> Result<(), OperationError> {
    validate_raw_strided_bounds(dst_data.len(), shape, dst_strides, dst_offset)?;
    validate_raw_strided_bounds(src_data.len(), shape, src_strides, src_offset)?;
    Ok(())
}

pub(crate) fn validate_raw_strided_bounds(
    len: usize,
    shape: &[usize],
    strides: &[isize],
    offset: isize,
) -> Result<(), OperationError> {
    validate_rank(shape, strides)?;
    if shape.contains(&0) {
        return Ok(());
    }
    let (min_offset, max_offset) = checked_strided_extrema(offset, shape, strides)?;
    validate_reachable_bounds(len, min_offset, max_offset)
}

fn raw_strided_scale_loop<T>(
    dst_data: &mut [T],
    shape: &[usize],
    dst_strides: &[isize],
    dst_offset: isize,
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy + Mul<T, Output = T> + Zero,
{
    let len = crate::strided::element_count(shape)?;
    if len == 0 {
        return Ok(());
    }
    if shape.is_empty() {
        let dst_index = checked_offset_to_index(dst_offset)?;
        dst_data[dst_index] = scale_value(dst_data[dst_index], beta);
        return Ok(());
    }
    if is_column_major_contiguous(shape, dst_strides)? {
        let dst_start = checked_offset_to_index(dst_offset)?;
        let dst_end = dst_start
            .checked_add(len)
            .ok_or_else(|| OperationError::ElementCountOverflow)?;
        let dst = dst_data
            .get_mut(dst_start..dst_end)
            .ok_or_else(|| OperationError::OffsetOverflow { value: dst_end })?;
        for dst_value in dst.iter_mut() {
            *dst_value = scale_value(*dst_value, beta);
        }
        return Ok(());
    }

    raw_strided_scale_recurse(
        shape.len() - 1,
        dst_data,
        shape,
        dst_strides,
        dst_offset,
        beta,
    )
}

fn raw_strided_scale_recurse<T>(
    axis: usize,
    dst_data: &mut [T],
    shape: &[usize],
    dst_strides: &[isize],
    dst_base: isize,
    beta: T,
) -> Result<(), OperationError>
where
    T: Copy + Mul<T, Output = T> + Zero,
{
    if axis == 0 {
        for index in 0..shape[0] {
            let dst_index =
                checked_offset_to_index(checked_strided_offset(dst_base, index, dst_strides[0])?)?;
            dst_data[dst_index] = scale_value(dst_data[dst_index], beta);
        }
        return Ok(());
    }

    for index in 0..shape[axis] {
        raw_strided_scale_recurse(
            axis - 1,
            dst_data,
            shape,
            dst_strides,
            checked_strided_offset(dst_base, index, dst_strides[axis])?,
            beta,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn raw_strided_combine_loop<T>(
    dst_data: &mut [T],
    src_data: &[T],
    shape: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
    source_conjugate: bool,
    action: RawStridedAction<T>,
) -> Result<(), OperationError>
where
    T: Copy + Add<T, Output = T> + Mul<T, Output = T> + Zero + ConjugateValue + Send + Sync,
{
    if crate::strided::element_count(shape)? == 0 {
        return Ok(());
    }
    if raw_rank_within_strided_limit_routed(
        dst_data,
        src_data,
        shape,
        dst_strides,
        src_strides,
        dst_offset,
        src_offset,
        source_conjugate,
        action,
    )? {
        return Ok(());
    }
    // Why not normalize every layout: normalization orders axes by
    // destination stride, and on a destination whose axes are not separated
    // (possibly aliasing) that reorder would change the overwrite winner and
    // the accumulation order of symtn's column-major loop.
    if !destination_axes_separated(shape, dst_strides) {
        return raw_strided_combine_recurse_mapped(
            shape.len() - 1,
            dst_data,
            src_data,
            shape,
            |axis| Ok(dst_strides[axis]),
            |axis| Ok(src_strides[axis]),
            dst_offset,
            src_offset,
            source_conjugate,
            action,
        );
    }
    combine_fused_layout(
        dst_data,
        src_data,
        shape,
        dst_strides,
        src_strides,
        dst_offset,
        src_offset,
        source_conjugate,
        action,
        &mut FusedLayoutScratch::default(),
        &mut SmallVec::new(),
    )
}

/// Strided's raw kernels on the raw layout when its rank is within
/// [`RAW_FUSED_RANK_LIMIT`]: Strided then fuses the layout itself, so
/// symtn's normalization would only repeat that work. Callers follow a
/// `false` with [`combine_fused_layout`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn raw_rank_within_strided_limit_routed<T>(
    dst_data: &mut [T],
    src_data: &[T],
    shape: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
    source_conjugate: bool,
    action: RawStridedAction<T>,
) -> Result<bool, OperationError>
where
    T: Copy + Add<T, Output = T> + Mul<T, Output = T> + Zero + ConjugateValue + Send + Sync,
{
    if shape.len() > RAW_FUSED_RANK_LIMIT {
        return Ok(false);
    }
    strided_raw_action(
        dst_data,
        src_data,
        shape,
        dst_strides,
        src_strides,
        dst_offset,
        src_offset,
        source_conjugate,
        action,
    )
}

/// Normalizes the layout with [`normalize_fused_layout`] (unit axes
/// dropped, axes ordered by destination stride, axes contiguous in both
/// operands merged) and chooses the kernel by the fused rank: up to
/// [`RAW_FUSED_RANK_LIMIT`] Strided's raw kernels, otherwise one span walk
/// over the fused axes.
///
/// Strided gates on the rank it is given, before dropping unit axes, so a
/// raw rank above its limit must be fused here first. A raw rank within the
/// limit was already offered to Strided by
/// [`raw_rank_within_strided_limit_routed`], and fusion changes neither the
/// action nor destination separation, so it is not offered again.
#[allow(clippy::too_many_arguments)]
pub(crate) fn combine_fused_layout<T>(
    dst_data: &mut [T],
    src_data: &[T],
    shape: &[usize],
    raw_dst_strides: &[isize],
    raw_src_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
    source_conjugate: bool,
    action: RawStridedAction<T>,
    layout: &mut FusedLayoutScratch,
    index: &mut SmallVec<[usize; FUSED_LAYOUT_INLINE_RANK]>,
) -> Result<(), OperationError>
where
    T: Copy + Add<T, Output = T> + Mul<T, Output = T> + Zero + ConjugateValue + Send + Sync,
{
    normalize_fused_layout(shape, raw_dst_strides, raw_src_strides, layout)?;
    let (dims, dst_strides, src_strides) =
        (layout.dims(), layout.dst_strides(), layout.src_strides());
    if shape.len() > RAW_FUSED_RANK_LIMIT
        && strided_raw_action(
            dst_data,
            src_data,
            dims,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            source_conjugate,
            action,
        )?
    {
        return Ok(());
    }
    index.clear();
    index.resize(dims.len(), 0);
    let op = move |value: T| value.maybe_conj(source_conjugate);
    macro_rules! run {
        ($apply:expr) => {
            apply_fused_pair_slices(
                dst_data,
                src_data,
                dims,
                dst_strides,
                src_strides,
                dst_offset,
                src_offset,
                index,
                $apply,
                op,
            )
        };
    }
    match action {
        RawStridedAction::Copy => run!(|dst: &mut T, value| *dst = value),
        RawStridedAction::CopyScale { alpha } => {
            run!(move |dst: &mut T, value| *dst = scale_value(value, alpha))
        }
        RawStridedAction::Axpy { alpha } => {
            run!(move |dst: &mut T, value| *dst = *dst + scale_value(value, alpha))
        }
        RawStridedAction::Axpby { alpha, beta } => {
            run!(move |dst: &mut T, value| *dst = beta * *dst + scale_value(value, alpha))
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn raw_strided_combine_recurse_mapped<T, D, S>(
    axis: usize,
    dst_data: &mut [T],
    src_data: &[T],
    shape: &[usize],
    dst_stride: D,
    src_stride: S,
    dst_base: isize,
    src_base: isize,
    source_conjugate: bool,
    action: RawStridedAction<T>,
) -> Result<(), OperationError>
where
    T: Copy + Add<T, Output = T> + Mul<T, Output = T> + Zero + ConjugateValue,
    D: Copy + Fn(usize) -> Result<isize, OperationError>,
    S: Copy + Fn(usize) -> Result<isize, OperationError>,
{
    if axis == 0 {
        let dst_stride = dst_stride(0)?;
        let src_stride = src_stride(0)?;
        for index in 0..shape[0] {
            let dst_index =
                checked_offset_to_index(checked_strided_offset(dst_base, index, dst_stride)?)?;
            let src_index =
                checked_offset_to_index(checked_strided_offset(src_base, index, src_stride)?)?;
            apply_raw_strided_action(
                &mut dst_data[dst_index],
                src_data[src_index].maybe_conj(source_conjugate),
                action,
            );
        }
        return Ok(());
    }

    let dst_axis_stride = dst_stride(axis)?;
    let src_axis_stride = src_stride(axis)?;
    for index in 0..shape[axis] {
        raw_strided_combine_recurse_mapped(
            axis - 1,
            dst_data,
            src_data,
            shape,
            dst_stride,
            src_stride,
            checked_strided_offset(dst_base, index, dst_axis_stride)?,
            checked_strided_offset(src_base, index, src_axis_stride)?,
            source_conjugate,
            action,
        )?;
    }
    Ok(())
}

/// Runs `Copy`, a nonzero `CopyScale` and a nonzero `Axpy` through Strided's
/// raw kernels, which compute the same per-element expression as
/// [`apply_raw_strided_action`]. Returns `Ok(false)`, without touching `dst`,
/// when the call must stay on symtn's loop:
///
/// - `Axpby` (Strided has no allocation-free axpby) and a zero `alpha`, whose
///   TensorKit zero rule Strided's `alpha * x` would break (`0 * inf = NaN`);
/// - rank above [`RAW_FUSED_RANK_LIMIT`], where Strided falls back to
///   allocating view kernels. Strided gates on the rank it is given, before
///   dropping unit axes, so [`combine_fused_layout`] passes the fused layout
///   of a larger raw rank;
/// - a destination whose axes are not separated (see
///   [`destination_axes_separated`]). Strided replays axes in destination
///   stride order, so on an aliasing destination the overwrite winner and the
///   accumulation order would differ from symtn's column-major loop.
#[allow(clippy::too_many_arguments)]
pub(crate) fn strided_raw_action<T>(
    dst_data: &mut [T],
    src_data: &[T],
    shape: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
    source_conjugate: bool,
    action: RawStridedAction<T>,
) -> Result<bool, OperationError>
where
    T: Copy + Add<T, Output = T> + Mul<T, Output = T> + Zero + ConjugateValue + Send + Sync,
{
    match action {
        RawStridedAction::Copy => {}
        RawStridedAction::CopyScale { alpha } | RawStridedAction::Axpy { alpha }
            if !alpha.is_zero() => {}
        _ => return Ok(false),
    }
    if shape.len() > RAW_FUSED_RANK_LIMIT || !destination_axes_separated(shape, dst_strides) {
        return Ok(false);
    }
    let src =
        RawStridedRef::new(src_data, shape, src_strides, src_offset).map_err(strided_error)?;
    let mut dst =
        RawStridedMut::new(dst_data, shape, dst_strides, dst_offset).map_err(strided_error)?;
    let (dst, src) = (&mut dst, &src);
    crate::host_pool::strided(shape.iter().product(), |exec| {
        match (action, source_conjugate) {
            (RawStridedAction::Copy, conjugate) => {
                let plan = CopyPlan::compile(shape, dst_strides, src_strides)?;
                if conjugate {
                    plan.execute_conj(dst, src, exec)
                } else {
                    plan.execute(dst, src, exec)
                }
            }
            (RawStridedAction::CopyScale { alpha }, false) => copy_scale_raw(dst, src, alpha),
            (RawStridedAction::CopyScale { alpha }, true) => copy_scale_conj_raw(dst, src, alpha),
            (RawStridedAction::Axpy { alpha }, false) => axpy_raw(dst, src, alpha),
            (RawStridedAction::Axpy { alpha }, true) => axpy_conj_raw(dst, src, alpha),
            (RawStridedAction::Axpby { .. }, _) => return Ok(false),
        }
        .map(|()| true)
    })
    .map_err(strided_error)
}

/// Whether every non-unit destination axis, taken in ascending `|stride|`
/// order, has a stride beyond the offset span of all smaller-stride axes.
/// Such a layout is injective, so each destination slot is written once and
/// traversal order cannot change a value.
///
/// Why not Strided's `validate_destination_layout_without_alloc`: it also
/// accepts interleaved injective layouts through an `O(block²)` pairwise scan,
/// and `CopyPlan::compile` then re-proves them with an allocating exact check.
/// Separated layouts are the ones both decide in `O(rank²)` without
/// allocating; interleaved layouts keep symtn's `O(n)` loop.
fn destination_axes_separated(shape: &[usize], strides: &[isize]) -> bool {
    if shape.contains(&0) {
        return true;
    }
    let mut span = 0usize;
    let mut previous: Option<(usize, usize)> = None;
    loop {
        let mut next: Option<((usize, usize), usize)> = None;
        for (axis, (&dim, &stride)) in shape.iter().zip(strides).enumerate() {
            let key = (stride.unsigned_abs(), axis);
            if dim > 1
                && previous.is_none_or(|previous| key > previous)
                && next.is_none_or(|(best, _)| key < best)
            {
                next = Some((key, dim));
            }
        }
        let Some(((stride, axis), dim)) = next else {
            return true;
        };
        if stride <= span {
            return false;
        }
        match stride
            .checked_mul(dim - 1)
            .and_then(|extent| span.checked_add(extent))
        {
            Some(covered) => span = covered,
            None => return false,
        }
        previous = Some((stride, axis));
    }
}

fn apply_raw_strided_action<T>(dst: &mut T, src: T, action: RawStridedAction<T>)
where
    T: Copy + Add<T, Output = T> + Mul<T, Output = T> + Zero,
{
    *dst = match action {
        RawStridedAction::Copy => src,
        RawStridedAction::CopyScale { alpha } => scale_value(src, alpha),
        RawStridedAction::Axpy { alpha } => *dst + scale_value(src, alpha),
        RawStridedAction::Axpby { alpha, beta } => beta * *dst + scale_value(src, alpha),
    };
}

/// Overflow signal for the per-element strided offset helpers.
///
/// Fieldless on purpose: these helpers run once per element on the
/// non-contiguous combine/scale recurse path, where returning
/// `Result<_, OperationError>` (536 bytes) forced a per-element sret move +
/// drop that dominated the profile (issue #230). This ZST-sized error keeps
/// the hot `Result` pointer-small.
///
/// Why-not shrink `OperationError` itself: that is issue #231 (parked) and
/// would ripple through every operation call site; a local error confined to
/// this helper family fixes the hot path without touching the shared enum.
#[derive(Clone, Copy, Debug)]
enum OffsetError {
    /// index/stride `isize` arithmetic overflowed.
    ElementCount,
    /// signed offset did not fit back into `usize`.
    Offset,
}

impl From<OffsetError> for OperationError {
    fn from(err: OffsetError) -> Self {
        // Map to the exact OperationError variants and messages every `?` call
        // site reports.
        match err {
            OffsetError::ElementCount => OperationError::ElementCountOverflow,
            OffsetError::Offset => OperationError::OffsetOverflow { value: usize::MAX },
        }
    }
}

fn checked_strided_offset(base: isize, index: usize, stride: isize) -> Result<isize, OffsetError> {
    let index = isize::try_from(index).map_err(|_| OffsetError::ElementCount)?;
    base.checked_add(index.checked_mul(stride).ok_or(OffsetError::ElementCount)?)
        .ok_or(OffsetError::ElementCount)
}

fn checked_offset_to_index(offset: isize) -> Result<usize, OffsetError> {
    usize::try_from(offset).map_err(|_| OffsetError::Offset)
}

fn is_column_major_contiguous(shape: &[usize], strides: &[isize]) -> Result<bool, OperationError> {
    let mut expected = 1isize;
    for (&dim, &stride) in shape.iter().zip(strides.iter()) {
        if dim > 1 && stride != expected {
            return Ok(false);
        }
        let dim = isize::try_from(dim).map_err(|_| OperationError::ElementCountOverflow)?;
        expected = expected
            .checked_mul(dim)
            .ok_or_else(|| OperationError::ElementCountOverflow)?;
    }
    Ok(true)
}

fn strided_linear_offset(
    mut linear: usize,
    shape: &[usize],
    strides: &[isize],
    base: isize,
) -> Result<usize, OperationError> {
    let mut offset = base;
    for (&dim, &stride) in shape.iter().zip(strides.iter()) {
        let coord = if dim == 0 { 0 } else { linear % dim };
        if let Some(quotient) = linear.checked_div(dim) {
            linear = quotient;
        }
        let coord = isize::try_from(coord).map_err(|_| OperationError::ElementCountOverflow)?;
        offset = offset
            .checked_add(
                coord
                    .checked_mul(stride)
                    .ok_or_else(|| OperationError::ElementCountOverflow)?,
            )
            .ok_or_else(|| OperationError::ElementCountOverflow)?;
    }
    usize::try_from(offset).map_err(|_| OperationError::OffsetOverflow { value: usize::MAX })
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    // What: the offset helper family's Result must stay pointer-small so the
    // per-element non-contiguous combine/scale loop pays no 536-byte sret move
    // + drop (issue #230). Fails if OffsetError grows a field or OperationError
    // is routed back onto these hot helpers.
    #[test]
    fn offset_error_result_stays_small() {
        assert!(std::mem::size_of::<Result<usize, OffsetError>>() <= 16);
    }

    // What: the offset helpers emit the exact OperationError variants their
    // `?` call sites report.
    #[test]
    fn offset_helpers_map_to_original_operation_errors() {
        // isize::MAX * 2 overflows the checked_mul -> ElementCountOverflow.
        let err: OperationError = checked_strided_offset(0, 2, isize::MAX).unwrap_err().into();
        assert_eq!(err, OperationError::ElementCountOverflow);

        // A negative offset cannot be a usize index -> OffsetOverflow{usize::MAX}.
        let err: OperationError = checked_offset_to_index(-1).unwrap_err().into();
        assert_eq!(err, OperationError::OffsetOverflow { value: usize::MAX });
    }

    #[test]
    fn tensortrace_raw_rejects_short_destination_stride_rank() {
        let mut dst = [0.0; 4];
        let error = tensortrace_raw_strided_kernel_add_with_coefficient(
            &mut dst,
            &[1.0, 2.0, 3.0, 4.0],
            &[2, 2],
            &[],
            &[1],
            &[1, 2],
            &[],
            0,
            0,
            false,
            1.0,
            0.0,
        )
        .unwrap_err();

        assert_eq!(
            error,
            OperationError::RankMismatch {
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn tensortrace_raw_rejects_combined_source_range_before_writing() {
        let mut dst = [7.0, 8.0];
        let original = dst;
        let error = tensortrace_raw_strided_kernel_add_with_coefficient(
            &mut dst,
            &[1.0, 2.0, 3.0, 4.0],
            &[2],
            &[2],
            &[1],
            &[2],
            &[2],
            0,
            0,
            false,
            1.0,
            0.0,
        )
        .unwrap_err();

        assert_eq!(error, OperationError::OffsetOverflow { value: 4 });
        assert_eq!(dst, original);
    }

    #[test]
    fn tensortrace_raw_exports_reject_all_stride_rank_mismatches() {
        let cases = [
            (&[][..], &[1isize][..], &[1isize][..], 1, 0),
            (&[1, 1], &[1], &[1], 1, 2),
            (&[1], &[], &[1], 1, 0),
            (&[1], &[1, 1], &[1], 1, 2),
            (&[1], &[1], &[], 1, 0),
            (&[1], &[1], &[1, 1], 1, 2),
        ];
        for (dst_strides, src_output_strides, src_trace_strides, expected, actual) in cases {
            let mut dst = [7.0, 8.0];
            let original = dst;
            let error = tensortrace_raw_strided_kernel_add_with_coefficient(
                &mut dst,
                &[1.0, 2.0, 3.0],
                &[2],
                &[2],
                dst_strides,
                src_output_strides,
                src_trace_strides,
                0,
                0,
                false,
                1.0,
                1.0,
            )
            .unwrap_err();
            assert_eq!(error, OperationError::RankMismatch { expected, actual });
            assert_eq!(dst, original);
        }
    }

    #[test]
    fn coefficient_tensortrace_raw_rejects_bounds_before_writing() {
        for (dst_len, src_len, src_output_stride, src_trace_stride, expected) in
            [(1, 4, 1, 1, 1), (2, 2, 1, 1, 2), (2, 4, 2, 2, 4)]
        {
            let mut dst = vec![7.0; dst_len];
            let original = dst.clone();
            let error = tensortrace_raw_strided_kernel_add_with_coefficient(
                &mut dst,
                &vec![1.0; src_len],
                &[2],
                &[2],
                &[1],
                &[src_output_stride],
                &[src_trace_stride],
                0,
                0,
                false,
                1.0,
                1.0,
            )
            .unwrap_err();
            assert_eq!(error, OperationError::OffsetOverflow { value: expected });
            assert_eq!(dst, original);
        }
    }

    #[test]
    fn tensortrace_raw_rejects_coordinate_and_intermediate_overflow() {
        for (output_shape, dst_strides, src_output_strides, src_offset) in [
            (&[usize::MAX][..], &[0][..], &[0][..], 0),
            (&[2, 2][..], &[0, 0][..], &[isize::MAX, -isize::MAX][..], 1),
        ] {
            let mut dst = [9.0];
            let original = dst;
            let error = tensortrace_raw_strided_kernel_add_with_coefficient(
                &mut dst,
                &[1.0],
                output_shape,
                &[],
                dst_strides,
                src_output_strides,
                &[],
                0,
                src_offset,
                false,
                1.0,
                0.0,
            )
            .unwrap_err();
            assert_eq!(error, OperationError::ElementCountOverflow);
            assert_eq!(dst, original);
        }
    }

    #[test]
    fn tensortrace_raw_rejects_output_and_trace_element_count_overflow() {
        for (output_shape, trace_shape, dst_strides, src_output_strides, src_trace_strides) in [
            (
                &[usize::MAX, 2][..],
                &[][..],
                &[0, 0][..],
                &[0, 0][..],
                &[][..],
            ),
            (&[][..], &[usize::MAX, 2][..], &[][..], &[][..], &[0, 0][..]),
        ] {
            let mut dst = [9.0];
            let original = dst;
            let error = tensortrace_raw_strided_kernel_add_with_coefficient(
                &mut dst,
                &[1.0],
                output_shape,
                trace_shape,
                dst_strides,
                src_output_strides,
                src_trace_strides,
                0,
                0,
                false,
                1.0,
                0.0,
            )
            .unwrap_err();
            assert_eq!(error, OperationError::ElementCountOverflow);
            assert_eq!(dst, original);
        }
    }

    #[test]
    fn tensortrace_raw_rejects_negative_reachable_offsets_before_writing() {
        for (trace_shape, dst_offset, src_output_stride, src_offset) in
            [(&[1][..], -1, 1, 0), (&[0][..], 0, -2, 1)]
        {
            let mut dst = [7.0, 8.0];
            let original = dst;
            let error = tensortrace_raw_strided_kernel_add_with_coefficient(
                &mut dst,
                &[1.0, 2.0],
                &[2],
                trace_shape,
                &[1],
                &[src_output_stride],
                &[1],
                dst_offset,
                src_offset,
                false,
                1.0,
                0.0,
            )
            .unwrap_err();
            assert_eq!(error, OperationError::OffsetOverflow { value: usize::MAX });
            assert_eq!(dst, original);
        }
    }

    #[test]
    fn coefficient_tensortrace_raw_empty_trace_keeps_unread_base_and_scalar_action() {
        let mut dst = [1.0, 2.0];
        tensortrace_raw_strided_kernel_add_with_coefficient(
            &mut dst,
            &[],
            &[2],
            &[0],
            &[1],
            &[1],
            &[isize::MAX],
            0,
            100,
            false,
            f64::INFINITY,
            1.0,
        )
        .unwrap();

        // TensorOperations' `_mapreducedim!` over an empty trace applies only
        // `initop = Scaler(One())`, so `C` is left as it was: no `inf * 0`.
        assert_eq!(dst, [1.0, 2.0]);
    }

    /// Three `output [2] x trace [2]` source blocks whose traced sums carry
    /// `(inf, 1.5)`, `(-0, -0)`, NaN payloads and a subnormal.
    fn special_trace_source() -> Vec<Complex64> {
        vec![
            Complex64::new(f64::INFINITY, 1.0),
            Complex64::new(-0.0, -0.0),
            Complex64::new(-0.0, 0.5),
            Complex64::new(-0.0, -0.0),
            Complex64::new(f64::from_bits(0x7ff8_0000_0000_1407), 2.5),
            Complex64::new(1.5, f64::NEG_INFINITY),
            Complex64::new(0.0, -2.0),
            Complex64::new(-4.0, f64::from_bits(0xfff8_0000_0000_0042)),
            Complex64::new(2.0, f64::INFINITY),
            Complex64::new(-0.0, 3.0),
            Complex64::new(-1.25, 0.5),
            Complex64::new(f64::from_bits(1), -0.0),
        ]
    }

    /// `(src_offset, dst_offset)` of each term; the last one accumulates a
    /// second producer into the third destination block.
    const TRACE_TERMS: [(isize, isize); 4] = [(0, 0), (4, 2), (8, 4), (0, 4)];

    fn run_trace_terms<T, C: Copy>(dst: &mut [T], src: &[T], alpha: T, coefficients: [C; 4])
    where
        T: Copy
            + Add<T, Output = T>
            + Mul<T, Output = T>
            + Zero
            + One
            + PartialEq
            + ConjugateValue
            + crate::RecouplingCoefficientAction<C>,
    {
        for (&(src_offset, dst_offset), coefficient) in TRACE_TERMS.iter().zip(coefficients) {
            tensortrace_raw_strided_kernel_add_with_coefficient(
                dst,
                src,
                &[2],
                &[2],
                &[1],
                &[1],
                &[2],
                dst_offset,
                src_offset,
                false,
                alpha,
                coefficient,
            )
            .unwrap();
        }
    }

    /// What: the raw trace applies a real coefficient to a complex traced sum
    /// componentwise, i.e. exactly as it acts on the traced real and
    /// imaginary components, with `α = 1` and the coefficient applied to
    /// the traced sum.
    #[test]
    fn coefficient_tensortrace_raw_scales_real_coefficients_componentwise() {
        let src = special_trace_source();
        let re: Vec<f64> = src.iter().map(|v| v.re).collect();
        let im: Vec<f64> = src.iter().map(|v| v.im).collect();
        for coefficients in [[-0.5, 1.0, 2.0, 0.25], [0.0, -1.0, 0.1, 1.0]] {
            let mut got = vec![Complex64::new(-0.0, -0.0); 6];
            let mut want_re = vec![-0.0; 6];
            let mut want_im = vec![-0.0; 6];
            run_trace_terms(&mut got, &src, Complex64::one(), coefficients);
            run_trace_terms(&mut want_re, &re, 1.0, coefficients);
            run_trace_terms(&mut want_im, &im, 1.0, coefficients);
            for (position, value) in got.iter().enumerate() {
                assert_eq!(
                    (value.re.to_bits(), value.im.to_bits()),
                    (want_re[position].to_bits(), want_im[position].to_bits()),
                    "element {position} for {coefficients:?}"
                );
            }
        }
    }

    /// What: an anyonic complex coefficient keeps the complex multiply, an
    /// exact `1 + 0i` does not multiply, and a non-unit alpha is folded into
    /// the coefficient first (TensorKit's `α′ = α * coeff`) and then applied
    /// to every traced element before it is added (TensorOperations'
    /// `C + scale(a₁, α′) + scale(a₂, α′)`).
    #[test]
    fn coefficient_tensortrace_raw_complex_coefficients_and_folded_alpha() {
        let src = special_trace_source();
        let one = Complex64::new(1.0, 0.0);
        let mut got = vec![Complex64::zero(); 6];
        run_trace_terms(&mut got, &src, one, [one, one, one, one]);
        // `(inf, 1.5)` survives only because `1 + 0i` does not multiply.
        assert_eq!(
            (got[0].re, got[0].im.to_bits()),
            (f64::INFINITY, 1.5f64.to_bits())
        );

        let coefficient = Complex64::new(0.6, 0.8);
        let mut got = vec![Complex64::zero(); 2];
        tensortrace_raw_strided_kernel_add_with_coefficient(
            &mut got,
            &src,
            &[2],
            &[2],
            &[1],
            &[1],
            &[2],
            0,
            8,
            false,
            one,
            coefficient,
        )
        .unwrap();
        let want = [0, 1].map(|output| {
            Complex64::zero() + src[8 + output] * coefficient + src[10 + output] * coefficient
        });
        assert_eq!(got, want);

        let alpha = Complex64::new(0.5, -0.25);
        let finite: Vec<Complex64> = (0..12)
            .map(|k| Complex64::new(k as f64 * 0.37 - 2.0, 1.3 - k as f64 * 0.11))
            .collect();
        let mut got = vec![Complex64::zero(); 2];
        tensortrace_raw_strided_kernel_add_with_coefficient(
            &mut got,
            &finite,
            &[2],
            &[2],
            &[1],
            &[1],
            &[2],
            0,
            4,
            false,
            alpha,
            3.0,
        )
        .unwrap();
        let folded = Complex64::new(alpha.re * 3.0, alpha.im * 3.0);
        let want = [0, 1].map(|output| {
            Complex64::zero() + folded * finite[4 + output] + folded * finite[6 + output]
        });
        // Two traced terms reach each entry.
        crate::test_numerics::numerics::assert_slices_close("folded alpha", &got, &want, 2);
    }

    #[test]
    fn coefficient_tensortrace_raw_applies_conjugation_and_complex_coefficient() {
        let mut dst = [Complex64::new(1.0, 1.0)];
        tensortrace_raw_strided_kernel_add_with_coefficient(
            &mut dst,
            &[Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0)],
            &[],
            &[2],
            &[],
            &[],
            &[1],
            0,
            0,
            true,
            Complex64::new(2.0, -1.0),
            Complex64::new(0.0, 1.0),
        )
        .unwrap();

        assert_eq!(dst, [Complex64::new(1.0, 11.0)]);
    }

    #[test]
    fn bilinear_kernel_handles_independent_conjugation_and_padding() {
        let lhs = [
            Complex64::new(99.0, 0.0),
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, -1.0),
            Complex64::new(98.0, 0.0),
            Complex64::new(-2.0, 0.5),
            Complex64::new(4.0, 3.0),
        ];
        let rhs = [
            Complex64::new(5.0, -1.0),
            Complex64::new(97.0, 0.0),
            Complex64::new(2.0, 4.0),
            Complex64::new(-3.0, 2.0),
            Complex64::new(96.0, 0.0),
            Complex64::new(1.0, -2.0),
        ];
        let shape = [2, 2];
        let lhs_strides = [1, 3];
        let rhs_strides = [2, 3];
        let lhs_values = [lhs[1], lhs[2], lhs[4], lhs[5]];
        let rhs_values = [rhs[0], rhs[2], rhs[3], rhs[5]];
        for (lhs_conjugate, rhs_conjugate) in
            [(false, false), (true, false), (false, true), (true, true)]
        {
            let expected = lhs_values.iter().copied().zip(rhs_values).fold(
                Complex64::new(0.0, 0.0),
                |sum, (lhs, rhs)| {
                    sum + lhs.maybe_conj(lhs_conjugate) * rhs.maybe_conj(rhs_conjugate)
                },
            );
            assert_eq!(
                bilinear_raw_strided_kernel_mapped(
                    &lhs,
                    &rhs,
                    &shape,
                    |axis| Ok(lhs_strides[axis]),
                    |axis| Ok(rhs_strides[axis]),
                    1,
                    0,
                    lhs_conjugate,
                    rhs_conjugate,
                )
                .unwrap(),
                expected
            );
        }
    }

    #[test]
    fn bilinear_kernel_validates_each_view_before_reading() {
        assert_eq!(
            bilinear_raw_strided_kernel_mapped(
                &[1.0],
                &[2.0],
                &[2],
                |_| Ok(1),
                |_| Ok(1),
                0,
                0,
                false,
                false,
            )
            .unwrap_err(),
            OperationError::OffsetOverflow { value: 1 }
        );
    }

    #[test]
    fn bilinear_kernel_accepts_checked_negative_strides() {
        assert_eq!(
            bilinear_raw_strided_kernel_mapped(
                &[1.0, 2.0, 3.0],
                &[4.0, 5.0, 6.0],
                &[3],
                |_| Ok(-1),
                |_| Ok(1),
                2,
                0,
                false,
                false,
            )
            .unwrap(),
            3.0 * 4.0 + 2.0 * 5.0 + 1.0 * 6.0
        );
    }

    // TensorKit scaling parity for non-finite data (#1438). Every expected
    // value below is an output observed in Julia 1.11 with TensorKit main
    // f87ca7fe (version 0.17.1), TensorOperations 5.6.2 and VectorInterface
    // 0.6.0, quoted beside it.

    fn bit_pairs(values: &[Complex64]) -> Vec<(u64, u64)> {
        values
            .iter()
            .map(|v| (v.re.to_bits(), v.im.to_bits()))
            .collect()
    }

    fn bits(values: &[f64]) -> Vec<u64> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    /// What: a zero `α` adds VectorInterface's `scale(x, 0) = zero(x) * 0`,
    /// whatever the source holds, and `β` scales the destination the same way.
    /// `A = [Inf NaN; -Inf 1.0]`, `C = [1.0 -0.0; 2.0 NaN]` (column-major
    /// below) and `tensoradd!(C, A, ((2, 1), ()), false, α, β)` gives
    /// `α = 0, β = 1`: `[1.0 0.0; 2.0 NaN]`; `α = 0, β = 0`: all `0.0`;
    /// `α = 0, β = 2`: `[2.0 0.0; 4.0 NaN]`; `α = 2, β = 0`:
    /// `[Inf -Inf; NaN 2.0]`.
    #[test]
    fn tensoradd_raw_zero_alpha_matches_tensoroperations() {
        let src = [f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 1.0];
        let initial = [1.0, 2.0, -0.0, f64::NAN];
        for (alpha, beta, want) in [
            (0.0, 1.0, [1.0, 2.0, 0.0, f64::NAN]),
            (0.0, 0.0, [0.0; 4]),
            (0.0, 2.0, [2.0, 4.0, 0.0, f64::NAN]),
            (2.0, 0.0, [f64::INFINITY, f64::NAN, f64::NEG_INFINITY, 2.0]),
        ] {
            let mut dst = initial;
            tensoradd_raw_strided_kernel(
                &mut Vec::new(),
                &mut dst,
                &src,
                &[2, 2],
                &[1, 2],
                &[2, 1],
                0,
                0,
                false,
                alpha,
                beta,
            )
            .unwrap();
            // NaN bits are the kernel's own; compare NaN-ness, then bits.
            let canonical = |v: f64| if v.is_nan() { f64::NAN } else { v };
            assert_eq!(
                bits(&dst.map(canonical)),
                bits(&want.map(canonical)),
                "alpha {alpha}, beta {beta}"
            );
        }
    }

    /// What: a complex zero scale gives `0 + 0i` for a non-finite complex
    /// source: `scale(complex(Inf, NaN), complex(0.0, 0.0))` is `0.0 + 0.0im`,
    /// and `scale(complex(Inf, 1.0), 0.0)` is `0.0 + 0.0im`.
    #[test]
    fn complex_zero_scale_is_exact_zero() {
        let src = [
            Complex64::new(f64::INFINITY, f64::NAN),
            Complex64::new(f64::INFINITY, 1.0),
        ];
        let mut dst = [Complex64::new(f64::NAN, 1.0); 2];
        tensoradd_raw_strided_kernel(
            &mut Vec::new(),
            &mut dst,
            &src,
            &[2],
            &[1],
            &[1],
            0,
            0,
            false,
            Complex64::zero(),
            Complex64::zero(),
        )
        .unwrap();
        assert_eq!(bit_pairs(&dst), bit_pairs(&[Complex64::zero(); 2]));
    }

    /// What: `β = 0` wipes a NaN destination (`scale(NaN, 0.0)` is `0.0`) and
    /// a nonzero `β` still multiplies it.
    #[test]
    fn scale_raw_zero_beta_is_exact_zero() {
        let mut dst = [f64::NAN, f64::INFINITY, -1.0];
        scale_raw_strided_kernel_trusted(&mut dst, &[3], &[1], 0, 0.0).unwrap();
        assert_eq!(bits(&dst), bits(&[0.0; 3]));
        let mut dst = [f64::NAN, f64::INFINITY, -1.0];
        scale_raw_strided_kernel_trusted(&mut dst, &[3], &[1], 0, 2.0).unwrap();
        assert!(dst[0].is_nan());
        assert_eq!(dst[1..], [f64::INFINITY, -2.0]);
    }

    /// What: the coefficient trace skips a zero coefficient, as TensorKit's
    /// `_trace_permute!` does (`iszero(coeff) && continue`), so even a `-0.0`
    /// destination is untouched; a zero `α` with a nonzero coefficient adds
    /// `α′ = 0` scaled elements; and `α′ = α * coeff` scales each element
    /// before the sum, so `α = 1, coeff = 0.5` over `[1e308, 1e308]` gives
    /// `1.0e308` rather than `inf`.
    #[test]
    fn coefficient_trace_zero_scales_and_per_element_alpha() {
        let run = |dst: &mut [f64], src: &[f64], alpha: f64, coefficient: f64| {
            tensortrace_raw_strided_kernel_add_with_coefficient(
                dst,
                src,
                &[],
                &[2],
                &[],
                &[],
                &[3],
                0,
                0,
                false,
                alpha,
                coefficient,
            )
            .unwrap();
        };
        let non_finite = [f64::INFINITY, 0.0, 0.0, f64::NAN];
        let mut dst = [-0.0];
        run(&mut dst, &non_finite, 1.0, 0.0);
        assert_eq!(bits(&dst), bits(&[-0.0]));
        let mut dst = [-0.0];
        run(&mut dst, &non_finite, 3.0, 0.0);
        assert_eq!(bits(&dst), bits(&[-0.0]));

        let mut dst = [1.5];
        run(&mut dst, &non_finite, 0.0, 2.0);
        assert_eq!(dst, [1.5]);

        let mut dst = [0.0];
        run(&mut dst, &[1e308, 0.0, 0.0, 1e308], 1.0, 0.5);
        assert_eq!(dst, [1e308]);

        // Complex payload, real coefficient: `α = 0 + 0i` still gives zero
        // contributions for `(inf, NaN)` entries.
        let run = |dst: &mut [Complex64], src: &[Complex64], alpha: Complex64| {
            tensortrace_raw_strided_kernel_add_with_coefficient(
                dst,
                src,
                &[],
                &[2],
                &[],
                &[],
                &[3],
                0,
                0,
                false,
                alpha,
                -2.0,
            )
            .unwrap();
        };
        let non_finite = [
            Complex64::new(f64::INFINITY, f64::NAN),
            Complex64::zero(),
            Complex64::zero(),
            Complex64::new(1.0, f64::NEG_INFINITY),
        ];
        let mut dst = [Complex64::new(1.0, -2.0)];
        run(&mut dst, &non_finite, Complex64::zero());
        assert_eq!(bit_pairs(&dst), bit_pairs(&[Complex64::new(1.0, -2.0)]));
    }

    trait BitPattern: Copy {
        fn pattern(self) -> (u64, u64);
    }

    impl BitPattern for f64 {
        fn pattern(self) -> (u64, u64) {
            (self.to_bits(), 0)
        }
    }

    impl BitPattern for Complex64 {
        fn pattern(self) -> (u64, u64) {
            (self.re.to_bits(), self.im.to_bits())
        }
    }

    impl BitPattern for f32 {
        fn pattern(self) -> (u64, u64) {
            (f64::from(self).to_bits(), 0)
        }
    }

    impl BitPattern for num_complex::Complex32 {
        fn pattern(self) -> (u64, u64) {
            (f64::from(self.re).to_bits(), f64::from(self.im).to_bits())
        }
    }

    fn patterns<T: BitPattern>(values: &[T]) -> Vec<(u64, u64)> {
        values.iter().map(|&value| value.pattern()).collect()
    }

    /// Bit patterns with every NaN component collapsed to one pattern.
    ///
    /// Why not compare NaN payloads of arithmetic results: Rust leaves the
    /// payload and sign of a NaN produced by `+`/`*` unspecified, and LLVM
    /// commutes operands, so with two NaN inputs the surviving payload differs
    /// between optimization levels of the same kernel. Copies and
    /// conjugation (a sign flip) are exact and keep full bit comparison.
    fn arithmetic_patterns<T: BitPattern>(values: &[T]) -> Vec<(u64, u64)> {
        let canonical = |bits: u64| {
            if f64::from_bits(bits).is_nan() {
                f64::NAN.to_bits()
            } else {
                bits
            }
        };
        patterns(values)
            .into_iter()
            .map(|(re, im)| (canonical(re), canonical(im)))
            .collect()
    }

    const NAN_PAYLOAD: u64 = 0x7ff8_0000_0000_1234;

    fn special_reals() -> [f64; 12] {
        [
            -0.0,
            0.0,
            f64::from_bits(NAN_PAYLOAD),
            f64::from_bits(NAN_PAYLOAD | (1 << 63)),
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::from_bits(1),
            -f64::MIN_POSITIVE / 3.0,
            f64::MAX,
            1.5,
            -2.25,
            1.0e-300,
        ]
    }

    fn special_values<T>(len: usize, seed: usize, make: impl Fn(f64, f64) -> T) -> Vec<T> {
        let reals = special_reals();
        (0..len)
            .map(|i| {
                make(
                    reals[(i * 5 + seed) % reals.len()],
                    reals[(i * 7 + seed + 3) % reals.len()],
                )
            })
            .collect()
    }

    /// Layouts of the given shape: column-major, reversed axis order,
    /// negative strides, a stride-0 (broadcast) source axis, and a separated
    /// but padded destination (column-major over extents `dim + 1`). Returns
    /// `(dst_strides, dst_offset, src_strides, src_offset)` with offsets
    /// making every reachable index nonnegative.
    fn layouts(shape: &[usize]) -> Vec<(Vec<isize>, isize, Vec<isize>, isize)> {
        let column_major = |order: &[usize]| {
            let mut strides = vec![0isize; shape.len()];
            let mut stride = 1isize;
            for &axis in order {
                strides[axis] = stride;
                stride *= shape[axis] as isize;
            }
            strides
        };
        let forward: Vec<usize> = (0..shape.len()).collect();
        let reverse: Vec<usize> = (0..shape.len()).rev().collect();
        let negate_even = |strides: &[isize]| {
            let mut offset = 0isize;
            let strides: Vec<isize> = strides
                .iter()
                .enumerate()
                .map(|(axis, &stride)| {
                    if axis % 2 == 0 {
                        offset += stride * (shape[axis].max(1) as isize - 1);
                        -stride
                    } else {
                        stride
                    }
                })
                .collect();
            (strides, offset)
        };
        let padded = {
            let mut strides = vec![0isize; shape.len()];
            let mut stride = 1isize;
            for (axis, slot) in strides.iter_mut().enumerate() {
                *slot = stride;
                stride *= shape[axis] as isize + 1;
            }
            strides
        };
        let (negative, negative_offset) = negate_even(&column_major(&forward));
        let mut broadcast = column_major(&reverse);
        if let Some(first) = broadcast.first_mut() {
            *first = 0;
        }
        vec![
            (column_major(&forward), 0, column_major(&forward), 0),
            (column_major(&forward), 0, column_major(&reverse), 0),
            (column_major(&reverse), 0, negative.clone(), negative_offset),
            (negative, negative_offset, broadcast, 0),
            (padded, 0, column_major(&reverse), 0),
        ]
    }

    fn assert_strided_matches_column_major_loop<T>(make: impl Fn(f64, f64) -> T, scales: &[T])
    where
        T: BitPattern
            + Add<T, Output = T>
            + Mul<T, Output = T>
            + PartialEq
            + Zero
            + One
            + ConjugateValue
            + Send
            + Sync
            + core::fmt::Debug,
    {
        let shapes: Vec<Vec<usize>> = (0..=RAW_FUSED_RANK_LIMIT)
            .map(|rank| (0..rank).map(|axis| [2, 3, 1, 2][axis % 4]).collect())
            .chain([vec![2, 0, 3], vec![0]])
            .collect();
        let one = T::one();
        let mut actions = vec![RawStridedAction::Copy];
        for &alpha in scales {
            actions.push(RawStridedAction::CopyScale { alpha });
            actions.push(RawStridedAction::Axpy { alpha });
        }
        actions.push(RawStridedAction::Axpy { alpha: one });
        for shape in &shapes {
            // Covers the padded destination, whose extents are `dim + 1`.
            let len = shape.iter().map(|&dim| dim + 1).product::<usize>();
            for (dst_strides, dst_offset, src_strides, src_offset) in layouts(shape) {
                for &action in &actions {
                    for conjugate in [false, true] {
                        let src = special_values(len, 1, &make);
                        let initial = special_values(len, 4, &make);
                        let mut expected = initial.clone();
                        naive_combine(
                            &mut expected,
                            &src,
                            shape,
                            &dst_strides,
                            &src_strides,
                            dst_offset,
                            src_offset,
                            conjugate,
                            action,
                        );
                        let mut actual = initial.clone();
                        let routed = strided_raw_action(
                            &mut actual,
                            &src,
                            shape,
                            &dst_strides,
                            &src_strides,
                            dst_offset,
                            src_offset,
                            conjugate,
                            action,
                        )
                        .unwrap();
                        assert!(routed, "shape {shape:?} {action:?}");
                        let compare = if matches!(action, RawStridedAction::Copy) {
                            patterns::<T>
                        } else {
                            arithmetic_patterns::<T>
                        };
                        assert_eq!(
                            compare(&actual),
                            compare(&expected),
                            "shape {shape:?} dst {dst_strides:?} src {src_strides:?} \
                             {action:?} conj {conjugate}"
                        );
                    }
                }
            }
        }
    }

    /// What: `Copy`, nonzero `CopyScale` and nonzero `Axpy` through Strided
    /// are bit-identical to an independent column-major scalar loop at ranks 0 through 8 over
    /// permuted, negative, broadcast and zero-extent layouts, with and
    /// without conjugation, for signed zeros, NaN payloads, infinities and
    /// subnormals (NaN payloads compared exactly for copies, as NaN-ness for
    /// arithmetic). `Copy` of `inf + 0i` stays `inf + 0i` (no multiply).
    #[test]
    fn strided_raw_actions_match_column_major_loop_bitwise() {
        assert_strided_matches_column_major_loop(|re, _| re, &[-0.5, 3.0, f64::from_bits(1)]);
        assert_strided_matches_column_major_loop(
            Complex64::new,
            &[
                Complex64::new(-0.5, 0.25),
                Complex64::new(0.0, -1.0),
                Complex64::new(3.0, 0.0),
            ],
        );
        let mut dst = [Complex64::zero()];
        let src = [Complex64::new(f64::INFINITY, 0.0)];
        tensoradd_raw_strided_kernel(
            &mut Vec::new(),
            &mut dst,
            &src,
            &[1],
            &[1],
            &[1],
            0,
            0,
            false,
            Complex64::one(),
            Complex64::zero(),
        )
        .unwrap();
        assert_eq!(bit_pairs(&dst), bit_pairs(&src));
    }

    /// What: a zero `α` never reaches Strided, so a non-finite source still
    /// follows TensorKit's `scale(x, 0) = zero(x) * 0` through both public
    /// entries: `tensoradd!(C, A, (2, 1, 3), conj, 0, β)` with `A` all
    /// `Inf`/`NaN` gives `C = zero * 0` for `β = 0` and `C + zero * 0` for
    /// `β = 1` (so a `-0.0` component of `C` becomes `+0.0`).
    #[test]
    fn zero_alpha_keeps_tensorkit_rule_on_strided_route() {
        let shape = [2, 3, 2];
        let dst_strides = [3, 1, 6];
        let src_strides = [1, 2, 6];
        let src = [
            Complex64::new(f64::INFINITY, f64::NAN),
            Complex64::new(f64::NAN, f64::NEG_INFINITY),
        ]
        .repeat(6);
        let initial: Vec<Complex64> = (0..12)
            .map(|i| Complex64::new(i as f64 - 0.5, -(i as f64)))
            .collect();
        for conjugate in [false, true] {
            for (beta, want) in [
                (Complex64::zero(), vec![Complex64::zero(); 12]),
                (
                    Complex64::one(),
                    initial.iter().map(|&c| c + Complex64::zero()).collect(),
                ),
            ] {
                let alpha = Complex64::zero();
                let mut gated = initial.clone();
                assert!(!strided_raw_action(
                    &mut gated,
                    &src,
                    &shape,
                    &dst_strides,
                    &src_strides,
                    0,
                    0,
                    conjugate,
                    raw_strided_action(alpha, beta),
                )
                .unwrap());
                assert_eq!(bit_pairs(&gated), bit_pairs(&initial));

                let mut raw = initial.clone();
                tensoradd_raw_strided_kernel(
                    &mut Vec::new(),
                    &mut raw,
                    &src,
                    &shape,
                    &dst_strides,
                    &src_strides,
                    0,
                    0,
                    conjugate,
                    alpha,
                    beta,
                )
                .unwrap();
                assert_eq!(bit_pairs(&raw), bit_pairs(&want), "beta {beta}");

                let mut checked = initial.clone();
                crate::kernel_adapter::StridedHostKernelAdapter::default()
                    .tensoradd_strided_checked(
                        &mut checked,
                        &src,
                        &shape,
                        &dst_strides,
                        &src_strides,
                        0,
                        0,
                        conjugate,
                        alpha,
                        beta,
                    )
                    .unwrap();
                assert_eq!(bit_pairs(&checked), bit_pairs(&want), "beta {beta}");
            }
        }
    }

    /// What: an aliasing destination keeps symtn's column-major accumulation
    /// order. With `shape = [2, 3]` and destination strides `[2, 1]`, slot 2
    /// receives `(1, 0)` then `(0, 2)`: `(1e-16 + 1) - 1 = 0`. Strided's
    /// `axpy_raw` visits `(0, 2)` first and gives `(1e-16 - 1) + 1 ≠ 0`, which
    /// is why such layouts stay on symtn's loop.
    #[test]
    fn aliasing_destination_keeps_column_major_accumulation() {
        let shape = [2, 3];
        let dst_strides = [2, 1];
        let src_strides = [1, 2];
        let mut src = [0.0; 6];
        src[1] = 1.0;
        src[4] = -1.0;
        let mut initial = [0.0; 5];
        initial[2] = 1.0e-16;
        assert!(!destination_axes_separated(&shape, &dst_strides));

        let mut expected = initial;
        naive_combine(
            &mut expected,
            &src,
            &shape,
            &dst_strides,
            &src_strides,
            0,
            0,
            false,
            RawStridedAction::Axpy { alpha: 1.0 },
        );
        assert_eq!(expected[2], 0.0);
        for (alpha, beta) in [(1.0, 1.0), (1.0, 0.0), (2.0, 0.0)] {
            let mut want = initial;
            naive_combine(
                &mut want,
                &src,
                &shape,
                &dst_strides,
                &src_strides,
                0,
                0,
                false,
                raw_strided_action(alpha, beta),
            );
            let mut actual = initial;
            tensoradd_raw_strided_kernel(
                &mut Vec::new(),
                &mut actual,
                &src,
                &shape,
                &dst_strides,
                &src_strides,
                0,
                0,
                false,
                alpha,
                beta,
            )
            .unwrap();
            assert_eq!(bits(&actual), bits(&want), "alpha {alpha} beta {beta}");
        }

        let mut reordered = initial;
        axpy_raw(
            &mut RawStridedMut::new(&mut reordered, &shape, &dst_strides, 0).unwrap(),
            &RawStridedRef::new(&src, &shape, &src_strides, 0).unwrap(),
            1.0,
        )
        .unwrap();
        assert_ne!(reordered[2].to_bits(), expected[2].to_bits());
    }

    /// What: the separation test accepts exactly the layouts whose axes
    /// never share an offset range and rejects broadcast, overlapping and
    /// interleaved destinations.
    #[test]
    fn destination_separation_classifies_layouts() {
        assert!(destination_axes_separated(&[], &[]));
        assert!(destination_axes_separated(&[2, 3, 4], &[1, 2, 6]));
        assert!(destination_axes_separated(&[2, 3, 4], &[12, -4, 1]));
        assert!(destination_axes_separated(&[1, 5, 1], &[0, 1, 0]));
        assert!(!destination_axes_separated(&[2, 3], &[0, 1]));
        assert!(!destination_axes_separated(&[2, 2], &[1, 1]));
        // Injective but interleaved: kept on symtn's loop.
        assert!(!destination_axes_separated(&[3, 2], &[2, 3]));
        assert!(!destination_axes_separated(&[3, 3], &[1, isize::MAX]));
    }

    /// What: rank above `RAW_FUSED_RANK_LIMIT` stays on symtn's loop, which is
    /// allocation-free at any rank; the result is unchanged.
    #[test]
    fn rank_above_fused_limit_keeps_symtn_loop() {
        let shape = [2usize; RAW_FUSED_RANK_LIMIT + 1];
        let strides: Vec<isize> = (0..shape.len()).map(|axis| 1 << axis).collect();
        let reversed: Vec<isize> = strides.iter().rev().copied().collect();
        let src: Vec<f64> = (0..512).map(f64::from).collect();
        let mut dst = vec![0.0; 512];
        assert!(!strided_raw_action(
            &mut dst,
            &src,
            &shape,
            &strides,
            &reversed,
            0,
            0,
            false,
            RawStridedAction::Copy,
        )
        .unwrap());
        tensoradd_raw_strided_kernel(
            &mut Vec::new(),
            &mut dst,
            &src,
            &shape,
            &strides,
            &reversed,
            0,
            0,
            false,
            1.0,
            0.0,
        )
        .unwrap();
        let mut want = vec![0.0; 512];
        naive_combine(
            &mut want,
            &src,
            &shape,
            &strides,
            &reversed,
            0,
            0,
            false,
            RawStridedAction::Copy,
        );
        assert_eq!(bits(&dst), bits(&want));
    }

    /// Test-only oracle independent of every traversal under test: decode
    /// each column-major linear index into raw coordinates and apply the
    /// element action at the raw offsets.
    #[allow(clippy::too_many_arguments)]
    fn naive_combine<T>(
        dst: &mut [T],
        src: &[T],
        shape: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        dst_offset: isize,
        src_offset: isize,
        conjugate: bool,
        action: RawStridedAction<T>,
    ) where
        T: Copy + Add<T, Output = T> + Mul<T, Output = T> + Zero + ConjugateValue,
    {
        for linear in 0..shape.iter().product::<usize>() {
            let (mut rest, mut d, mut s) = (linear, dst_offset, src_offset);
            for axis in 0..shape.len() {
                let coord = (rest % shape[axis]) as isize;
                rest /= shape[axis];
                d += coord * dst_strides[axis];
                s += coord * src_strides[axis];
            }
            let value = src[s as usize].maybe_conj(conjugate);
            apply_raw_strided_action(&mut dst[d as usize], value, action);
        }
    }

    /// xorshift64: deterministic layouts without a test-only dependency.
    struct Rng(u64);

    impl Rng {
        fn below(&mut self, bound: usize) -> usize {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 % bound as u64) as usize
        }
    }

    /// Strides of a column-major layout over `order` with extent
    /// `dim + pad` per axis, negating some axes; returns the strides, the
    /// offset making every index nonnegative, and the buffer length.
    fn random_strides(
        rng: &mut Rng,
        shape: &[usize],
        order: &[usize],
        pad_odds: usize,
    ) -> (Vec<isize>, isize, usize) {
        let mut strides = vec![0isize; shape.len()];
        let mut stride = 1isize;
        for &axis in order {
            strides[axis] = stride;
            let pad = usize::from(rng.below(pad_odds) == 0);
            stride *= (shape[axis] + pad) as isize;
        }
        let mut offset = 0isize;
        for (axis, slot) in strides.iter_mut().enumerate() {
            if shape[axis] == 1 {
                // A unit axis contributes no offset: any stride is valid.
                *slot = rng.below(9) as isize;
            } else if rng.below(4) == 0 {
                offset += *slot * (shape[axis] as isize - 1);
                *slot = -*slot;
            }
        }
        (strides, offset, stride as usize)
    }

    fn assert_fused_rank_matches_naive<T>(make: impl Fn(f64, f64) -> T, alpha: T, beta: T)
    where
        T: BitPattern
            + Add<T, Output = T>
            + Mul<T, Output = T>
            + PartialEq
            + Zero
            + One
            + ConjugateValue
            + Send
            + Sync
            + core::fmt::Debug,
    {
        let mut rng = Rng(0x2230_9e37_79b9_7f4a);
        let mut fused_ranks = [0usize; 17];
        for case in 0..96 {
            let rank = 6 + case % 11;
            let mut shape = Vec::with_capacity(rank);
            let mut len = 1usize;
            for _ in 0..rank {
                let dim = [1, 1, 2, 2, 2, 3][rng.below(6)];
                let dim = if len * dim > 4096 { 1 } else { dim };
                len *= dim;
                shape.push(dim);
            }
            let mut dst_order: Vec<usize> = (0..rank).collect();
            for axis in (1..rank).rev() {
                dst_order.swap(axis, rng.below(axis + 1));
            }
            // Same order (fusable runs), reversed, or an independent shuffle.
            let src_order: Vec<usize> = match case % 3 {
                0 => dst_order.clone(),
                1 => dst_order.iter().rev().copied().collect(),
                _ => {
                    let mut order = dst_order.clone();
                    for axis in (1..rank).rev() {
                        order.swap(axis, rng.below(axis + 1));
                    }
                    order
                }
            };
            let (dst_strides, dst_offset, dst_len) =
                random_strides(&mut rng, &shape, &dst_order, 3);
            let (src_strides, src_offset, src_len) =
                random_strides(&mut rng, &shape, &src_order, 4);
            assert!(destination_axes_separated(&shape, &dst_strides));
            let mut layout = FusedLayoutScratch::default();
            normalize_fused_layout(&shape, &dst_strides, &src_strides, &mut layout).unwrap();
            fused_ranks[layout.dims().len()] += 1;

            let src = special_values(src_len, case, &make);
            let initial = special_values(dst_len, case + 2, &make);
            for (a, b) in [
                (T::one(), T::zero()),
                (alpha, T::zero()),
                (alpha, T::one()),
                (alpha, beta),
                (T::zero(), beta),
            ] {
                let action = raw_strided_action(a, b);
                let compare = if matches!(action, RawStridedAction::Copy) {
                    patterns::<T>
                } else {
                    arithmetic_patterns::<T>
                };
                for conjugate in [false, true] {
                    let mut expected = initial.clone();
                    naive_combine(
                        &mut expected,
                        &src,
                        &shape,
                        &dst_strides,
                        &src_strides,
                        dst_offset,
                        src_offset,
                        conjugate,
                        action,
                    );
                    let mut validated = initial.clone();
                    tensoradd_raw_strided_kernel(
                        &mut Vec::new(),
                        &mut validated,
                        &src,
                        &shape,
                        &dst_strides,
                        &src_strides,
                        dst_offset,
                        src_offset,
                        conjugate,
                        a,
                        b,
                    )
                    .unwrap();
                    assert_eq!(
                        compare(&validated),
                        compare(&expected),
                        "case {case} shape {shape:?} dst {dst_strides:?} src {src_strides:?} \
                         {action:?} conj {conjugate}"
                    );
                    if !conjugate {
                        let mut trusted = initial.clone();
                        axpby_raw_strided_kernel_trusted(
                            &mut trusted,
                            &src,
                            &shape,
                            &dst_strides,
                            &src_strides,
                            dst_offset,
                            src_offset,
                            a,
                            b,
                        )
                        .unwrap();
                        assert_eq!(compare(&trusted), compare(&expected), "case {case}");
                    }
                }
            }
        }
        // The sample reaches both kernels: fused rank at most Strided's limit
        // from raw rank above it, and fused rank above the limit.
        assert!(fused_ranks[..=RAW_FUSED_RANK_LIMIT].iter().sum::<usize>() > 0);
        assert!(
            fused_ranks[RAW_FUSED_RANK_LIMIT + 1..]
                .iter()
                .sum::<usize>()
                > 0
        );
    }

    /// What: at raw rank 6 through 16, with unit axes of arbitrary stride,
    /// padded, permuted, reversed and negative strides on separated
    /// destinations, every action (copy, scale, axpy, axpby, zero alpha), with
    /// and without conjugation, gives exactly the naive column-major result
    /// for f32, f64, Complex32 and Complex64, whether the fused rank routes
    /// the call to Strided or to the fused span walk.
    #[test]
    fn fused_rank_gate_matches_naive_oracle_bitwise() {
        assert_fused_rank_matches_naive(|re, _| re, -0.5, 0.25);
        assert_fused_rank_matches_naive(|re, _| re as f32, -0.5f32, 0.25);
        assert_fused_rank_matches_naive(
            Complex64::new,
            Complex64::new(-0.5, 0.25),
            Complex64::new(0.0, -1.0),
        );
        assert_fused_rank_matches_naive(
            |re, im| num_complex::Complex32::new(re as f32, im as f32),
            num_complex::Complex32::new(-0.5, 0.25),
            num_complex::Complex32::new(0.0, -1.0),
        );
    }

    /// Raw rank-12 layout `(shape, dst_strides, src_strides)` over logical
    /// axes `[4, 3, 2, 2, 3, 2, 2, 4]` (padded destination, reversed source,
    /// so no logical axes fuse): both extent-4 axes are split into
    /// contiguous `(2, 2)` pairs and two unit axes are inserted. It fuses
    /// back to rank 8.
    fn raw_rank_twelve_fusing_to_eight() -> (Vec<usize>, Vec<isize>, Vec<isize>) {
        let logical = [4usize, 3, 2, 2, 3, 2, 2, 4];
        let mut dst = Vec::new();
        let mut stride = 1isize;
        for &dim in &logical {
            dst.push(stride);
            stride *= dim as isize + 1;
        }
        let mut src = vec![0isize; logical.len()];
        let mut stride = 1isize;
        for axis in (0..logical.len()).rev() {
            src[axis] = stride;
            stride *= logical[axis] as isize;
        }
        let mut layout = (Vec::new(), Vec::new(), Vec::new());
        for axis in 0..logical.len() {
            if logical[axis] == 4 {
                layout.0.extend([2, 2]);
                layout.1.extend([dst[axis], 2 * dst[axis]]);
                layout.2.extend([src[axis], 2 * src[axis]]);
            } else {
                layout.0.push(logical[axis]);
                layout.1.push(dst[axis]);
                layout.2.push(src[axis]);
            }
            if axis == 2 || axis == 5 {
                layout.0.push(1);
                layout.1.push(11);
                layout.2.push(3);
            }
        }
        layout
    }

    /// What: a raw rank-12 layout whose unit axes and contiguous axis pairs
    /// fuse away normalizes to rank 8, Strided's limit, and every action
    /// over it (through Strided, or symtn's walk for axpby) gives exactly the
    /// naive column-major result.
    #[test]
    fn raw_rank_twelve_normalizes_to_strided_limit() {
        let (shape, dst_strides, src_strides) = raw_rank_twelve_fusing_to_eight();
        assert_eq!(shape.len(), 12);
        let mut layout = FusedLayoutScratch::default();
        normalize_fused_layout(&shape, &dst_strides, &src_strides, &mut layout).unwrap();
        assert_eq!(layout.dims().len(), RAW_FUSED_RANK_LIMIT);

        let len: usize = shape.iter().product();
        let dst_len = 5 * 4 * 3 * 3 * 4 * 3 * 3 * 5;
        let src = special_values(len, 3, Complex64::new);
        let initial = special_values(dst_len, 5, Complex64::new);
        let alpha = Complex64::new(-0.5, 0.25);
        for (a, b) in [
            (Complex64::one(), Complex64::zero()),
            (alpha, Complex64::one()),
            (alpha, alpha),
        ] {
            let action = raw_strided_action(a, b);
            let mut expected = initial.clone();
            naive_combine(
                &mut expected,
                &src,
                &shape,
                &dst_strides,
                &src_strides,
                0,
                0,
                true,
                action,
            );
            let mut actual = initial.clone();
            tensoradd_raw_strided_kernel(
                &mut Vec::new(),
                &mut actual,
                &src,
                &shape,
                &dst_strides,
                &src_strides,
                0,
                0,
                true,
                a,
                b,
            )
            .unwrap();
            assert_eq!(
                arithmetic_patterns(&actual),
                arithmetic_patterns(&expected),
                "{action:?}"
            );
        }
    }
}
