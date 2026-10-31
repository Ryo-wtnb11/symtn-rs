//! Bounds-checked strided updates of one block whose layout the caller has
//! not proved in bounds (lazy-adjoint add and materialization, degeneracy
//! restriction).
//!
//! The layout is normalized once, jointly over the destination and every
//! source, while the caller's strides are converted, and one span walk then
//! updates the block from all sources. This is the shape of Strided.jl's
//! `_mapreduce_fuse!`/`_mapreduce_order!` (`src/mapreduce.jl`), which
//! TensorOperations' strided backend runs: axes fuse only where every array's
//! strides agree, and the loop order is chosen once over all arrays.

use core::mem::MaybeUninit;
use core::ops::{Add, Mul};

use num_traits::{One, Zero};
use smallvec::SmallVec;

use crate::host_scalar_kernels::{
    raw_strided_action, validate_raw_strided_bounds, RawStridedAction,
};
use crate::kernel_adapter::apply_fused_pair_slices;
use crate::scalar::scale_value;
use crate::{ConjugateValue, OperationError};

/// Layout passes and span walks of the checked block entries on this thread.
///
/// Why test builds only (`cfg(test)` or the `testing` feature): the probe
/// would sit once per block on the eager path whose per-block fixed cost it
/// exists to bound.
fn convert(stride: usize) -> Result<isize, OperationError> {
    isize::try_from(stride).map_err(|_| OperationError::ElementCountOverflow)
}

/// One block's loop layout against a destination and one or two sources.
///
/// Inline capacity 16: one of these lives per op, not per block, so the
/// stack cost is fixed at about 0.7 KiB, and every block of rank at most 16
/// is updated without a heap allocation, as the per-element kernel this path
/// replaced was. Past 16 non-unit axes the four stride buffers spill once
/// per op and are reused across its blocks.
#[doc(hidden)]
#[derive(Debug, Default)]
pub struct CheckedBlockLayout {
    dims: SmallVec<[usize; 16]>,
    dst_strides: SmallVec<[isize; 16]>,
    lhs_strides: SmallVec<[isize; 16]>,
    /// Empty after [`Self::fill_one`].
    rhs_strides: SmallVec<[isize; 16]>,
    index: SmallVec<[usize; 16]>,
}

impl CheckedBlockLayout {
    /// Normalizes `shape` against the destination and one source, whose
    /// strides `strides(axis)` returns as `(destination, source)`.
    pub fn fill_one(
        &mut self,
        shape: &[usize],
        mut strides: impl FnMut(usize) -> Result<(usize, usize), OperationError>,
    ) -> Result<(), OperationError> {
        self.fill(shape, false, |axis| {
            let (destination, source) = strides(axis)?;
            Ok((convert(destination)?, convert(source)?, 0))
        })
    }

    /// Normalizes `shape` against the destination and two sources, whose
    /// strides `strides(axis)` returns as `(destination, lhs, rhs)`.
    ///
    /// [`Self::tensoradd`] then reads the `lhs` strides.
    pub fn fill_two(
        &mut self,
        shape: &[usize],
        mut strides: impl FnMut(usize) -> Result<(usize, usize, usize), OperationError>,
    ) -> Result<(), OperationError> {
        self.fill(shape, true, |axis| {
            let (destination, lhs, rhs) = strides(axis)?;
            Ok((convert(destination)?, convert(lhs)?, convert(rhs)?))
        })
    }

    /// Every axis' strides are converted, extent-one axes included, before
    /// any write, so an unrepresentable stride is an error for the block
    /// whatever its extents. Extent-one axes are then dropped, the rest are
    /// placed in destination-stride order as they are read, and adjacent axes
    /// fuse where every operand's strides agree. Reachable offsets are
    /// unchanged, so bounds checked on this layout are the original bounds.
    fn fill(
        &mut self,
        shape: &[usize],
        two: bool,
        mut strides: impl FnMut(usize) -> Result<(isize, isize, isize), OperationError>,
    ) -> Result<(), OperationError> {
        self.dims.clear();
        self.dst_strides.clear();
        self.lhs_strides.clear();
        self.rhs_strides.clear();
        let mut count = Some(1usize);
        let mut empty = false;
        for (axis, &extent) in shape.iter().enumerate() {
            let (dst, lhs, rhs) = strides(axis)?;
            count = count.and_then(|count| count.checked_mul(extent));
            empty |= extent == 0;
            if extent == 1 {
                continue;
            }
            let mut position = self.dims.len();
            while position > 0 && self.dst_strides[position - 1] > dst {
                position -= 1;
            }
            self.dims.insert(position, extent);
            self.dst_strides.insert(position, dst);
            self.lhs_strides.insert(position, lhs);
            if two {
                self.rhs_strides.insert(position, rhs);
            }
        }
        #[cfg(any(test, feature = "testing"))]
        crate::probe::add(crate::probe::Event::LayoutPass, 1);
        if empty || self.dims.is_empty() {
            self.dims.clear();
            self.dst_strides.clear();
            self.lhs_strides.clear();
            self.rhs_strides.clear();
            self.dims.push(usize::from(!empty));
            self.dst_strides.push(0);
            self.lhs_strides.push(0);
            if two {
                self.rhs_strides.push(0);
            }
            return Ok(());
        }
        count.ok_or(OperationError::ElementCountOverflow)?;
        let mut fused = 0usize;
        for axis in 1..self.dims.len() {
            let extent = self.dims[fused] as isize;
            let agrees =
                |strides: &[isize]| strides[fused].checked_mul(extent) == Some(strides[axis]);
            if agrees(&self.dst_strides)
                && agrees(&self.lhs_strides)
                && (!two || agrees(&self.rhs_strides))
            {
                // Cannot overflow: the product of every extent was checked.
                self.dims[fused] *= self.dims[axis];
            } else {
                fused += 1;
                self.dims[fused] = self.dims[axis];
                self.dst_strides[fused] = self.dst_strides[axis];
                self.lhs_strides[fused] = self.lhs_strides[axis];
                if two {
                    self.rhs_strides[fused] = self.rhs_strides[axis];
                }
            }
        }
        let rank = fused + 1;
        self.dims.truncate(rank);
        self.dst_strides.truncate(rank);
        self.lhs_strides.truncate(rank);
        self.rhs_strides.truncate(if two { rank } else { 0 });
        Ok(())
    }

    /// `dst = alpha * op(src) + beta * dst` over the filled block, reading the
    /// source through the `lhs` strides. Both reachable extents are checked,
    /// then one span walk runs the scalar kernels' action, so `alpha = 1,
    /// beta = 0` stays a bit-exact copy rather than `1 * src`.
    #[allow(clippy::too_many_arguments)]
    pub fn tensoradd<T>(
        &mut self,
        dst_data: &mut [T],
        src_data: &[T],
        dst_offset: isize,
        src_offset: isize,
        source_conjugate: bool,
        alpha: T,
        beta: T,
    ) -> Result<(), OperationError>
    where
        T: Copy + Add<T, Output = T> + Mul<T, Output = T> + PartialEq + Zero + One + ConjugateValue,
    {
        validate_raw_strided_bounds(dst_data.len(), &self.dims, &self.dst_strides, dst_offset)?;
        validate_raw_strided_bounds(src_data.len(), &self.dims, &self.lhs_strides, src_offset)?;
        self.walk_one(
            dst_data,
            src_data,
            false,
            dst_offset,
            src_offset,
            source_conjugate,
            alpha,
            beta,
        );
        Ok(())
    }

    /// `dst = act(alpha, op_l(lhs))`, then `dst = dst + beta * op_r(rhs)`,
    /// over a block filled by [`Self::fill_two`].
    ///
    /// Each destination element is computed as that exact two-step sequence,
    /// `op_l(l) + beta * op_r(r)` when `alpha` is one and
    /// `alpha * op_l(l) + beta * op_r(r)` otherwise, in one span walk. Why
    /// `beta` is multiplied even when it is one: the second step is the
    /// scalar kernels' `Axpy`, which always scales, and `1 * (inf + 0i)` is
    /// observable. When the destination layout is not proved to reach each
    /// element once, the two steps run as two walks, so repeated destination
    /// elements accumulate as they did.
    ///
    /// All three reachable extents are checked before any write. The two
    /// sources may alias each other.
    #[allow(clippy::too_many_arguments)]
    pub fn add_two_source<T>(
        &mut self,
        dst_data: &mut [T],
        lhs_data: &[T],
        rhs_data: &[T],
        dst_offset: isize,
        lhs_offset: isize,
        rhs_offset: isize,
        lhs_conjugate: bool,
        rhs_conjugate: bool,
        alpha: T,
        beta: T,
    ) -> Result<(), OperationError>
    where
        T: Copy + Add<T, Output = T> + Mul<T, Output = T> + PartialEq + Zero + One + ConjugateValue,
    {
        if self.rhs_strides.len() != self.dims.len() {
            return Err(OperationError::InvalidArgument {
                message: "two-source block add needs a layout filled for two sources",
            });
        }
        validate_raw_strided_bounds(dst_data.len(), &self.dims, &self.dst_strides, dst_offset)?;
        validate_raw_strided_bounds(lhs_data.len(), &self.dims, &self.lhs_strides, lhs_offset)?;
        validate_raw_strided_bounds(rhs_data.len(), &self.dims, &self.rhs_strides, rhs_offset)?;
        if !reaches_each_element_once(&self.dims, &self.dst_strides) {
            self.walk_one(
                dst_data,
                lhs_data,
                false,
                dst_offset,
                lhs_offset,
                lhs_conjugate,
                alpha,
                T::zero(),
            );
            self.walk_one(
                dst_data,
                rhs_data,
                true,
                dst_offset,
                rhs_offset,
                rhs_conjugate,
                beta,
                T::one(),
            );
            return Ok(());
        }
        #[cfg(any(test, feature = "testing"))]
        crate::probe::add(crate::probe::Event::SpanWalk, 1);
        let op_l = move |value: T| value.maybe_conj(lhs_conjugate);
        let op_r = move |value: T| value.maybe_conj(rhs_conjugate);
        let Self {
            dims,
            dst_strides,
            lhs_strides,
            rhs_strides,
            index,
        } = self;
        index.resize(dims.len(), 0);
        macro_rules! run {
            ($combine:expr) => {
                apply_fused_triple_slices(
                    dst_data,
                    lhs_data,
                    rhs_data,
                    dims,
                    [dst_strides, lhs_strides, rhs_strides],
                    [dst_offset, lhs_offset, rhs_offset],
                    index,
                    $combine,
                )
            };
        }
        if alpha.is_one() {
            run!(move |dst: &mut T, lhs: T, rhs: T| {
                *dst = op_l(lhs) + scale_value(op_r(rhs), beta);
            });
        } else {
            run!(move |dst: &mut T, lhs: T, rhs: T| {
                *dst = scale_value(op_l(lhs), alpha) + scale_value(op_r(rhs), beta);
            });
        }
        Ok(())
    }

    /// [`Self::tensoradd`] with `beta = 0` into uninitialized storage: every
    /// element the filled destination layout reaches is written, from the
    /// same `op(src)` (times `alpha` unless it is one) the initialized walk
    /// stores, and none is read.
    ///
    /// Crate-private: coverage of an owned buffer is proved by the caller
    /// (`owned_blocks`), not by this walk.
    pub(crate) fn copy_uninit<T>(
        &mut self,
        dst_data: &mut [MaybeUninit<T>],
        src_data: &[T],
        dst_offset: isize,
        src_offset: isize,
        source_conjugate: bool,
        alpha: T,
    ) -> Result<(), OperationError>
    where
        T: Copy + Mul<T, Output = T> + PartialEq + Zero + One + ConjugateValue,
    {
        validate_raw_strided_bounds(dst_data.len(), &self.dims, &self.dst_strides, dst_offset)?;
        validate_raw_strided_bounds(src_data.len(), &self.dims, &self.lhs_strides, src_offset)?;
        #[cfg(any(test, feature = "testing"))]
        crate::probe::add(crate::probe::Event::SpanWalk, 1);
        let op = move |value: T| value.maybe_conj(source_conjugate);
        let Self {
            dims,
            dst_strides,
            lhs_strides,
            index,
            ..
        } = self;
        index.resize(dims.len(), 0);
        macro_rules! run {
            ($store:expr) => {
                apply_fused_pair_slices(
                    dst_data,
                    src_data,
                    dims,
                    dst_strides,
                    lhs_strides,
                    dst_offset,
                    src_offset,
                    index,
                    $store,
                    op,
                )
            };
        }
        if alpha.is_one() {
            run!(|dst: &mut MaybeUninit<T>, value| {
                dst.write(value);
            });
        } else {
            run!(move |dst: &mut MaybeUninit<T>, value| {
                dst.write(scale_value(value, alpha));
            });
        }
        Ok(())
    }

    /// [`Self::add_two_source`] into uninitialized storage, in its single
    /// walk: the destination layout must reach each element at most once,
    /// which is rejected before any write otherwise, so no element is read.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn add_two_source_uninit<T>(
        &mut self,
        dst_data: &mut [MaybeUninit<T>],
        lhs_data: &[T],
        rhs_data: &[T],
        dst_offset: isize,
        lhs_offset: isize,
        rhs_offset: isize,
        lhs_conjugate: bool,
        rhs_conjugate: bool,
        alpha: T,
        beta: T,
    ) -> Result<(), OperationError>
    where
        T: Copy + Add<T, Output = T> + Mul<T, Output = T> + PartialEq + Zero + One + ConjugateValue,
    {
        if self.rhs_strides.len() != self.dims.len()
            || !reaches_each_element_once(&self.dims, &self.dst_strides)
        {
            return Err(OperationError::InvalidArgument {
                message: "uninitialized two-source block add needs a two-source, write-once layout",
            });
        }
        validate_raw_strided_bounds(dst_data.len(), &self.dims, &self.dst_strides, dst_offset)?;
        validate_raw_strided_bounds(lhs_data.len(), &self.dims, &self.lhs_strides, lhs_offset)?;
        validate_raw_strided_bounds(rhs_data.len(), &self.dims, &self.rhs_strides, rhs_offset)?;
        #[cfg(any(test, feature = "testing"))]
        crate::probe::add(crate::probe::Event::SpanWalk, 1);
        let op_l = move |value: T| value.maybe_conj(lhs_conjugate);
        let op_r = move |value: T| value.maybe_conj(rhs_conjugate);
        let Self {
            dims,
            dst_strides,
            lhs_strides,
            rhs_strides,
            index,
        } = self;
        index.resize(dims.len(), 0);
        macro_rules! run {
            ($store:expr) => {
                apply_fused_triple_slices(
                    dst_data,
                    lhs_data,
                    rhs_data,
                    dims,
                    [dst_strides, lhs_strides, rhs_strides],
                    [dst_offset, lhs_offset, rhs_offset],
                    index,
                    $store,
                )
            };
        }
        if alpha.is_one() {
            run!(move |dst: &mut MaybeUninit<T>, lhs: T, rhs: T| {
                dst.write(op_l(lhs) + scale_value(op_r(rhs), beta));
            });
        } else {
            run!(move |dst: &mut MaybeUninit<T>, lhs: T, rhs: T| {
                dst.write(scale_value(op_l(lhs), alpha) + scale_value(op_r(rhs), beta));
            });
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn walk_one<T>(
        &mut self,
        dst_data: &mut [T],
        src_data: &[T],
        rhs: bool,
        dst_offset: isize,
        src_offset: isize,
        source_conjugate: bool,
        alpha: T,
        beta: T,
    ) where
        T: Copy + Add<T, Output = T> + Mul<T, Output = T> + PartialEq + Zero + One + ConjugateValue,
    {
        #[cfg(any(test, feature = "testing"))]
        crate::probe::add(crate::probe::Event::SpanWalk, 1);
        let op = move |value: T| value.maybe_conj(source_conjugate);
        let Self {
            dims,
            dst_strides,
            lhs_strides,
            rhs_strides,
            index,
        } = self;
        let src_strides = if rhs { rhs_strides } else { lhs_strides };
        index.resize(dims.len(), 0);
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
        match raw_strided_action(alpha, beta) {
            RawStridedAction::Copy => run!(|dst: &mut T, value| *dst = value),
            RawStridedAction::CopyScale { alpha } => run!(move |dst: &mut T, value| {
                *dst = scale_value(value, alpha);
            }),
            RawStridedAction::Axpy { alpha } => run!(move |dst: &mut T, value| {
                *dst = *dst + scale_value(value, alpha);
            }),
            RawStridedAction::Axpby { alpha, beta } => run!(move |dst: &mut T, value| {
                *dst = beta * *dst + scale_value(value, alpha);
            }),
        }
    }
}

/// Whether a normalized layout reaches each element at most once: every
/// stride positive and at least the span of the faster axes. Sufficient, not
/// necessary; a `false` only costs the two-walk sequence.
fn reaches_each_element_once(dims: &[usize], strides: &[isize]) -> bool {
    let mut span = 1isize;
    for (&extent, &stride) in dims.iter().zip(strides) {
        if extent <= 1 {
            continue;
        }
        if stride < span {
            return false;
        }
        span = stride.saturating_mul(extent as isize);
    }
    true
}

/// The three-operand form of `apply_fused_pair_slices`: one odometer over the
/// normalized axes, the innermost axis as a contiguous run when all three
/// operands step by one.
#[allow(clippy::too_many_arguments)]
fn apply_fused_triple_slices<Dst, T, Combine>(
    dst_data: &mut [Dst],
    lhs_data: &[T],
    rhs_data: &[T],
    dims: &[usize],
    [dst_strides, lhs_strides, rhs_strides]: [&[isize]; 3],
    [dst_offset, lhs_offset, rhs_offset]: [isize; 3],
    index: &mut [usize],
    store: Combine,
) where
    T: Copy,
    Combine: Fn(&mut Dst, T, T),
{
    let rank = dims.len();
    if rank == 0 || dims.contains(&0) {
        return;
    }
    let inner_len = dims[0];
    let (inner_dst, inner_lhs, inner_rhs) = (dst_strides[0], lhs_strides[0], rhs_strides[0]);
    index.fill(0);
    let (mut dst_base, mut lhs_base, mut rhs_base) = (dst_offset, lhs_offset, rhs_offset);
    let contiguous = inner_dst == 1 && inner_lhs == 1 && inner_rhs == 1;
    loop {
        if contiguous {
            let dst = &mut dst_data[dst_base as usize..][..inner_len];
            let lhs = &lhs_data[lhs_base as usize..][..inner_len];
            let rhs = &rhs_data[rhs_base as usize..][..inner_len];
            for position in 0..inner_len {
                store(&mut dst[position], lhs[position], rhs[position]);
            }
        } else {
            for position in 0..inner_len as isize {
                store(
                    &mut dst_data[(dst_base + position * inner_dst) as usize],
                    lhs_data[(lhs_base + position * inner_lhs) as usize],
                    rhs_data[(rhs_base + position * inner_rhs) as usize],
                );
            }
        }
        let mut axis = 1;
        loop {
            if axis >= rank {
                return;
            }
            index[axis] += 1;
            dst_base += dst_strides[axis];
            lhs_base += lhs_strides[axis];
            rhs_base += rhs_strides[axis];
            if index[axis] < dims[axis] {
                break;
            }
            let extent = dims[axis] as isize;
            dst_base -= extent * dst_strides[axis];
            lhs_base -= extent * lhs_strides[axis];
            rhs_base -= extent * rhs_strides[axis];
            index[axis] = 0;
            axis += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe::{take_checked_block_passes, CheckedBlockPasses};
    use crate::tensoradd_raw_strided_kernel;
    use num_complex::Complex64;

    /// `(shape, [destination, lhs, rhs] strides, [destination, lhs, rhs]
    /// offsets)`: windows of 5-wide parents, as restriction and lazy-adjoint
    /// storage read them.
    type Case = (Vec<usize>, [Vec<isize>; 3], [isize; 3]);

    fn cases() -> Vec<Case> {
        vec![
            // Rank zero, and an empty block.
            (vec![], [vec![], vec![], vec![]], [2, 7, 3]),
            (
                vec![2, 0, 3],
                [vec![1, 2, 2], vec![1, 5, 5], vec![3, 1, 9]],
                [0, 0, 0],
            ),
            // Rank one, everything contiguous: the inner run fast path.
            (vec![6], [vec![1], vec![1], vec![1]], [1, 4, 0]),
            // Rank two: lhs transposed (adjoint storage), rhs a padded window.
            (vec![3, 4], [vec![1, 3], vec![5, 1], vec![1, 5]], [0, 2, 6]),
            // Negative strides: reversed reads and one reversed write axis.
            (
                vec![3, 4],
                [vec![-1, 3], vec![-1, -5], vec![5, 1]],
                [2, 40, 0],
            ),
            // Rank three, compact destination, windows of 5x7x2 parents.
            (
                vec![3, 4, 2],
                [vec![1, 3, 12], vec![1, 5, 35], vec![35, 1, 5]],
                [3, 11, 1],
            ),
            // Rank four with extent-one axes carrying arbitrary strides.
            (
                vec![1, 4, 1, 3],
                [vec![1, 1, 4, 4], vec![7, 5, 20, 20], vec![1, 1, 9, 4]],
                [1, 4, 0],
            ),
            // Rank five, permuted: nothing fuses across the three operands.
            (
                vec![2, 2, 1, 3, 2],
                [
                    vec![1, 2, 4, 4, 12],
                    vec![12, 6, 1, 2, 1],
                    vec![1, 2, 7, 8, 4],
                ],
                [0, 5, 1],
            ),
        ]
    }

    /// Signed strides, which block structures cannot state but the walk
    /// must handle, enter below the `usize` conversion.
    fn fill_two(layout: &mut CheckedBlockLayout, shape: &[usize], strides: &[Vec<isize>; 3]) {
        let [dst, lhs, rhs] = strides;
        layout
            .fill(shape, true, |axis| Ok((dst[axis], lhs[axis], rhs[axis])))
            .unwrap();
    }

    /// Every slot starts initialized, so the uninitialized walks can be
    /// compared on the slots they do not reach as well.
    fn as_uninit<T: Copy>(values: &[T]) -> Vec<MaybeUninit<T>> {
        values.iter().copied().map(MaybeUninit::new).collect()
    }

    #[allow(unsafe_code)]
    fn assume_all_init<T: Copy>(values: Vec<MaybeUninit<T>>) -> Vec<T> {
        // SAFETY: every element came from `as_uninit`, and the walks only
        // `write` initialized values over it.
        values
            .into_iter()
            .map(|value| unsafe { value.assume_init() })
            .collect()
    }

    fn check<T>(
        mut exec: strided_kernel::Exec<'_>,
        values: &[T],
        coefficients: &[(T, T)],
        bits: fn(T) -> u128,
    ) where
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
        let mut layout = CheckedBlockLayout::default();
        for (shape, strides, [dst_offset, lhs_offset, rhs_offset]) in cases() {
            for &(alpha, beta) in coefficients {
                for (lhs_conjugate, rhs_conjugate) in
                    [(false, false), (true, false), (false, true), (true, true)]
                {
                    let lhs: Vec<T> = (0..80).map(|i| values[i % values.len()]).collect();
                    let rhs: Vec<T> = (0..80)
                        .map(|i| values[(i * 5 + 1) % values.len()])
                        .collect();
                    let initial: Vec<T> = (0..40)
                        .map(|i| values[(i * 7 + 3) % values.len()])
                        .collect();
                    // Oracle: two independent per-element scalar-kernel
                    // passes.
                    let mut expected = initial.clone();
                    for (source, strides_of, offset, conjugate, a, b) in [
                        (
                            &lhs,
                            &strides[1],
                            lhs_offset,
                            lhs_conjugate,
                            alpha,
                            T::zero(),
                        ),
                        (&rhs, &strides[2], rhs_offset, rhs_conjugate, beta, T::one()),
                    ] {
                        tensoradd_raw_strided_kernel(
                            exec.reborrow(),
                            &mut Vec::new(),
                            &mut expected,
                            source,
                            &shape,
                            &strides[0],
                            strides_of,
                            dst_offset,
                            offset,
                            conjugate,
                            a,
                            b,
                        )
                        .unwrap();
                    }
                    fill_two(&mut layout, &shape, &strides);
                    let mut actual = initial.clone();
                    layout
                        .add_two_source(
                            &mut actual,
                            &lhs,
                            &rhs,
                            dst_offset,
                            lhs_offset,
                            rhs_offset,
                            lhs_conjugate,
                            rhs_conjugate,
                            alpha,
                            beta,
                        )
                        .unwrap();
                    let expected: Vec<u128> = expected.into_iter().map(bits).collect();
                    let actual: Vec<u128> = actual.into_iter().map(bits).collect();
                    assert_eq!(actual, expected, "shape {shape:?}");

                    // The uninitialized entry stores the same bits wherever
                    // the layout reaches, and leaves the rest untouched; a
                    // layout that could revisit an element is refused.
                    let mut uninit = as_uninit(&initial);
                    let result = layout.add_two_source_uninit(
                        &mut uninit,
                        &lhs,
                        &rhs,
                        dst_offset,
                        lhs_offset,
                        rhs_offset,
                        lhs_conjugate,
                        rhs_conjugate,
                        alpha,
                        beta,
                    );
                    if reaches_each_element_once(&layout.dims, &layout.dst_strides) {
                        result.unwrap();
                        let uninit: Vec<u128> =
                            assume_all_init(uninit).into_iter().map(bits).collect();
                        assert_eq!(uninit, expected, "uninit, shape {shape:?}");
                    } else {
                        assert!(matches!(
                            result,
                            Err(OperationError::InvalidArgument { message }) if message.contains("write-once")
                        ));
                    }

                    // The single-source entry against one scalar-kernel pass,
                    // for every action of the pair.
                    for b in [T::zero(), T::one(), beta] {
                        let mut expected = initial.clone();
                        tensoradd_raw_strided_kernel(
                            exec.reborrow(),
                            &mut Vec::new(),
                            &mut expected,
                            &lhs,
                            &shape,
                            &strides[0],
                            &strides[1],
                            dst_offset,
                            lhs_offset,
                            lhs_conjugate,
                            alpha,
                            b,
                        )
                        .unwrap();
                        let mut actual = initial.clone();
                        layout
                            .tensoradd(
                                &mut actual,
                                &lhs,
                                dst_offset,
                                lhs_offset,
                                lhs_conjugate,
                                alpha,
                                b,
                            )
                            .unwrap();
                        let expected: Vec<u128> = expected.into_iter().map(bits).collect();
                        let actual: Vec<u128> = actual.into_iter().map(bits).collect();
                        assert_eq!(actual, expected, "single source, shape {shape:?}");
                        if b.is_zero() {
                            let mut uninit = as_uninit(&initial);
                            layout
                                .copy_uninit(
                                    &mut uninit,
                                    &lhs,
                                    dst_offset,
                                    lhs_offset,
                                    lhs_conjugate,
                                    alpha,
                                )
                                .unwrap();
                            let uninit: Vec<u128> =
                                assume_all_init(uninit).into_iter().map(bits).collect();
                            assert_eq!(uninit, expected, "uninit copy, shape {shape:?}");
                        }
                    }

                    // Any source window past its end is rejected before the
                    // destination is written.
                    if shape.iter().all(|&extent| extent > 0) {
                        let mut untouched = initial.clone();
                        assert!(layout
                            .add_two_source(
                                &mut untouched,
                                &lhs,
                                &rhs[..rhs.len() - 60],
                                dst_offset,
                                lhs_offset,
                                rhs_offset + 60,
                                false,
                                false,
                                alpha,
                                beta,
                            )
                            .is_err());
                        assert!(layout
                            .add_two_source(
                                &mut untouched,
                                &lhs,
                                &rhs,
                                -1,
                                lhs_offset,
                                rhs_offset,
                                false,
                                false,
                                alpha,
                                beta,
                            )
                            .is_err());
                        let untouched: Vec<u128> = untouched.into_iter().map(bits).collect();
                        let initial: Vec<u128> = initial.into_iter().map(bits).collect();
                        assert_eq!(untouched, initial);
                    }
                }
            }
        }
    }

    /// The fused walk is bit-identical to the two scalar-kernel passes it
    /// replaces, including signed zeros, infinities, NaN and subnormals,
    /// where `1 * x`, `0 + x` or a skipped `beta = 1` scale would not be.
    #[test]
    fn two_source_walk_is_bit_identical_to_two_scalar_kernel_passes() {
        let reals = [
            1.5,
            -0.0,
            0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            -2.25,
            7.0,
            f64::MIN_POSITIVE / 4.0,
        ];
        check(
            strided_kernel::Exec::serial(),
            &reals,
            &[
                (1.0, 1.0),
                (1.0, -3.0),
                (-2.0, 1.0),
                (0.5, -3.0),
                (0.0, 2.0),
            ],
            |value: f64| u128::from(value.to_bits()),
        );
        let complexes = [
            Complex64::new(1.5, -0.0),
            Complex64::new(-0.0, -1.0),
            Complex64::new(f64::INFINITY, 0.0),
            Complex64::new(0.0, f64::NEG_INFINITY),
            Complex64::new(f64::NAN, 2.0),
            Complex64::new(-3.0, 4.5),
            Complex64::new(-0.0, 0.0),
        ];
        let one = Complex64::new(1.0, 0.0);
        check(
            strided_kernel::Exec::serial(),
            &complexes,
            &[
                (one, one),
                (one, Complex64::new(-1.0, 0.5)),
                (Complex64::new(0.5, -1.0), one),
                (Complex64::new(2.0, 0.0), Complex64::new(-1.0, 0.5)),
            ],
            |value: Complex64| {
                (u128::from(value.re.to_bits()) << 64) | u128::from(value.im.to_bits())
            },
        );
    }

    #[test]
    fn aliased_sources_and_a_repeating_destination_match_two_passes() {
        let mut exec = strided_kernel::Exec::serial();
        let source: Vec<f64> = (0..12).map(|i| i as f64 * 0.75 - 2.0).collect();
        // A stride-zero destination axis visits each element three times, so
        // only the two-walk sequence reproduces the accumulation.
        for dst_strides in [vec![1isize, 4], vec![1, 0]] {
            let shape = [4usize, 3];
            let strides = [dst_strides.clone(), vec![3, 1], vec![1, 4]];
            let mut expected = vec![0.5; 12];
            for (src_strides, beta) in [(&strides[1], 0.0), (&strides[2], 1.0)] {
                let alpha = if beta == 0.0 { 2.0 } else { -3.0 };
                tensoradd_raw_strided_kernel(
                    exec.reborrow(),
                    &mut Vec::new(),
                    &mut expected,
                    &source,
                    &shape,
                    &dst_strides,
                    src_strides,
                    0,
                    0,
                    false,
                    alpha,
                    beta,
                )
                .unwrap();
            }
            let mut layout = CheckedBlockLayout::default();
            fill_two(&mut layout, &shape, &strides);
            let mut actual = vec![0.5; 12];
            take_checked_block_passes();
            layout
                .add_two_source(
                    &mut actual,
                    &source,
                    &source,
                    0,
                    0,
                    0,
                    false,
                    false,
                    2.0,
                    -3.0,
                )
                .unwrap();
            let walks = if dst_strides[1] == 0 { 2 } else { 1 };
            assert_eq!(take_checked_block_passes().span_walks, walks);
            assert_eq!(actual, expected, "destination strides {dst_strides:?}");
        }
    }

    /// The structural gate: one layout pass per block per operand
    /// set, and one span walk for a two-source update.
    #[test]
    fn one_layout_pass_and_one_walk_per_two_source_block() {
        let mut layout = CheckedBlockLayout::default();
        let (shape, strides, _) = cases().swap_remove(5);
        let data = vec![1.0f64; 80];
        let mut dst = vec![0.0f64; 40];
        take_checked_block_passes();
        fill_two(&mut layout, &shape, &strides);
        layout
            .add_two_source(&mut dst, &data, &data, 3, 11, 1, false, false, 2.0, 0.5)
            .unwrap();
        assert_eq!(
            take_checked_block_passes(),
            CheckedBlockPasses {
                layout_passes: 1,
                span_walks: 1
            }
        );
    }

    #[test]
    fn fill_converts_every_stride_and_normalizes_once() {
        let mut layout = CheckedBlockLayout::default();
        let too_wide = usize::MAX;
        // An unrepresentable stride on an extent-one axis is still an error.
        for strides in [(too_wide, 1, 1), (1, too_wide, 1), (1, 1, too_wide)] {
            assert!(matches!(
                layout.fill_two(&[1, 2], |_| Ok(strides)),
                Err(OperationError::ElementCountOverflow)
            ));
        }
        assert!(matches!(
            layout.fill_one(&[2], |_| Ok((too_wide, 1))),
            Err(OperationError::ElementCountOverflow)
        ));
        layout.fill_one(&[2], |_| Ok((1, 1))).unwrap();
        assert!(matches!(
            layout.fill_two(&[usize::MAX, 2], |_| Ok((1, 1, 1))),
            Err(OperationError::ElementCountOverflow)
        ));
        // A zero extent empties the block whatever the other extents.
        layout
            .fill_two(&[usize::MAX, 0, 2], |_| Ok((1, 1, 1)))
            .unwrap();
        assert_eq!(&layout.dims[..], &[0]);
        // Extent-one axes drop, axes sort by destination stride, and runs
        // fuse only where all three operands agree.
        let strides = [[6, 6, 6], [1, 1, 1], [7, 7, 7], [2, 2, 3]];
        layout
            .fill_two(&[2, 2, 1, 3], |axis| {
                let [dst, lhs, rhs] = strides[axis];
                Ok((dst, lhs, rhs))
            })
            .unwrap();
        assert_eq!(&layout.dims[..], &[2, 3, 2]);
        assert_eq!(&layout.dst_strides[..], &[1, 2, 6]);
        assert_eq!(&layout.lhs_strides[..], &[1, 2, 6]);
        assert_eq!(&layout.rhs_strides[..], &[1, 3, 6]);
        layout
            .fill_one(&[2, 2, 1, 3], |axis| {
                Ok((strides[axis][0], strides[axis][1]))
            })
            .unwrap();
        assert_eq!(&layout.dims[..], &[12]);
        assert!(layout.rhs_strides.is_empty());
        // A one-source layout cannot feed the two-source entry.
        assert!(matches!(
            layout.add_two_source(
                &mut [0.0; 12],
                &[0.0; 12],
                &[0.0; 12],
                0,
                0,
                0,
                false,
                false,
                1.0,
                1.0
            ),
            Err(OperationError::InvalidArgument { message }) if message.contains("filled for two sources")
        ));
    }
}
