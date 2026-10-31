//! The fresh-destination carrier of [`DenseExecutor::matmul_batch_fresh`]
//! and the dtype-generic pieces every implementation of it shares.

use core::mem::MaybeUninit;

use num_complex::{Complex32, Complex64};

use crate::{
    DenseDType, DenseError, DenseExecutor, DenseGemmBatchJob, DenseRead, DenseScalar, DenseView,
    DenseViewMut, DenseWrite, Engine, MatrixOp,
};

/// The uninitialized destination of one [`DenseExecutor::matmul_batch_fresh`]
/// call, with the caller's typed callbacks, for one GEMM dtype.
///
/// Each variant holds:
/// - `segments`: `(start, memory)` pairs in ascending, non-overlapping
///   destination order; `memory` is the destination range
///   `start..start + memory.len()`, in the coordinates of
///   [`DenseGemmBatchJob::dst_offset`]. Every job with a nonempty output must
///   lie inside one segment, disjoint from every other job. The call takes
///   the memory out of the slice it borrows.
/// - `zeroed`: writes zero to every element of a segment piece and returns
///   that same piece initialized. The executor calls it only where a route
///   needs an initialized destination; a piece it returns that is not its
///   argument is refused.
/// - `written`: receives each finished job's output, by job index.
///
/// The memory is `'a`, so a reported slice keeps the borrow for the whole
/// lifetime of the caller's parts.
#[doc(hidden)]
pub enum DenseFreshSegments<'s, 'a> {
    F32 {
        segments: &'s mut [(usize, &'a mut [MaybeUninit<f32>])],
        zeroed: for<'x> fn(&'x mut [MaybeUninit<f32>]) -> &'x mut [f32],
        written: &'s mut dyn FnMut(usize, &'a mut [f32]),
    },
    F64 {
        segments: &'s mut [(usize, &'a mut [MaybeUninit<f64>])],
        zeroed: for<'x> fn(&'x mut [MaybeUninit<f64>]) -> &'x mut [f64],
        written: &'s mut dyn FnMut(usize, &'a mut [f64]),
    },
    C32 {
        segments: &'s mut [(usize, &'a mut [MaybeUninit<Complex32>])],
        zeroed: for<'x> fn(&'x mut [MaybeUninit<Complex32>]) -> &'x mut [Complex32],
        written: &'s mut dyn FnMut(usize, &'a mut [Complex32]),
    },
    C64 {
        segments: &'s mut [(usize, &'a mut [MaybeUninit<Complex64>])],
        zeroed: for<'x> fn(&'x mut [MaybeUninit<Complex64>]) -> &'x mut [Complex64],
        written: &'s mut dyn FnMut(usize, &'a mut [Complex64]),
    },
}

impl DenseFreshSegments<'_, '_> {
    pub(crate) fn dtype(&self) -> DenseDType {
        match self {
            Self::F32 { .. } => DenseDType::F32,
            Self::F64 { .. } => DenseDType::F64,
            Self::C32 { .. } => DenseDType::C32,
            Self::C64 { .. } => DenseDType::C64,
        }
    }
}

/// One variant of [`DenseFreshSegments`], typed.
pub(crate) struct FreshParts<'s, 'a, T> {
    pub(crate) segments: &'s mut [(usize, &'a mut [MaybeUninit<T>])],
    pub(crate) zeroed: for<'x> fn(&'x mut [MaybeUninit<T>]) -> &'x mut [T],
    pub(crate) written: &'s mut dyn FnMut(usize, &'a mut [T]),
}

/// The four GEMM dtypes' erasure wrappers.
pub(crate) trait FreshScalar: Copy + Default + 'static {
    fn write(view: DenseViewMut<'_, Self>) -> DenseWrite<'_>;
    fn read(view: DenseView<'_, Self>) -> DenseRead<'_>;
    fn scalar(value: Self) -> DenseScalar;
}

macro_rules! fresh_scalar {
    ($($ty:ty => $variant:ident);+) => {$(
        impl FreshScalar for $ty {
            fn write(view: DenseViewMut<'_, Self>) -> DenseWrite<'_> {
                DenseWrite::$variant(view)
            }
            fn read(view: DenseView<'_, Self>) -> DenseRead<'_> {
                DenseRead::$variant(view)
            }
            fn scalar(value: Self) -> DenseScalar {
                DenseScalar::$variant(value)
            }
        }
    )+};
}
fresh_scalar!(f32 => F32; f64 => F64; Complex32 => C32; Complex64 => C64);

/// Matches a [`DenseFreshSegments`] with two same-dtype operands and runs
/// `$body` with the typed parts and views; any other combination is the
/// batched-matmul operand error.
macro_rules! dispatch_fresh {
    ($op:expr, $segments:expr, $lhs:expr, $rhs:expr, |$parts:ident, $l:ident, $r:ident| $body:expr) => {{
        use $crate::fresh::{DenseFreshSegments as S, FreshParts};
        use $crate::DenseRead as R;
        let dtypes = ($segments.dtype(), $lhs.dtype(), $rhs.dtype());
        match ($segments, $lhs, $rhs) {
            (
                S::F32 {
                    segments,
                    zeroed,
                    written,
                },
                R::F32($l),
                R::F32($r),
            ) => {
                let $parts = FreshParts {
                    segments,
                    zeroed,
                    written,
                };
                $body
            }
            (
                S::F64 {
                    segments,
                    zeroed,
                    written,
                },
                R::F64($l),
                R::F64($r),
            ) => {
                let $parts = FreshParts {
                    segments,
                    zeroed,
                    written,
                };
                $body
            }
            (
                S::C32 {
                    segments,
                    zeroed,
                    written,
                },
                R::C32($l),
                R::C32($r),
            ) => {
                let $parts = FreshParts {
                    segments,
                    zeroed,
                    written,
                };
                $body
            }
            (
                S::C64 {
                    segments,
                    zeroed,
                    written,
                },
                R::C64($l),
                R::C64($r),
            ) => {
                let $parts = FreshParts {
                    segments,
                    zeroed,
                    written,
                };
                $body
            }
            _ => Err($crate::executor::batch_operand_error(
                $op, dtypes.0, dtypes.1, dtypes.2,
            )),
        }
    }};
}
pub(crate) use dispatch_fresh;

pub(crate) const FRESH_OP: &str = "matmul_batch_fresh";

/// Hands each job with a nonempty output its own piece of `segments`, as
/// `place(job, piece)`, once per such job. `order` is staging for the jobs'
/// destination order.
///
/// Refuses, before placing anything, segments out of ascending order or
/// overlapping, and a job whose output is not inside one segment or overlaps
/// another job's: [`DenseError::OutOfBounds`]. Segment memory outside every
/// job is dropped unplaced.
pub(crate) fn split_fresh_segments<'a, E>(
    segments: &mut [(usize, &'a mut [E])],
    jobs: &[DenseGemmBatchJob],
    order: &mut Vec<usize>,
    place: impl FnMut(usize, &'a mut [E]),
) -> Result<(), DenseError> {
    check_fresh_segments(segments, jobs, order)?;
    place_fresh_jobs(segments, jobs, order, place);
    Ok(())
}

/// The refusals of [`split_fresh_segments`], leaving `order` holding the
/// nonempty jobs in destination order.
pub(crate) fn check_fresh_segments<E>(
    segments: &[(usize, &mut [E])],
    jobs: &[DenseGemmBatchJob],
    order: &mut Vec<usize>,
) -> Result<(), DenseError> {
    let end = |start: usize, len: usize| {
        start
            .checked_add(len)
            .ok_or(DenseError::OffsetOverflow { value: start })
    };
    let mut previous_end = 0;
    for (start, memory) in segments.iter() {
        if *start < previous_end {
            return Err(DenseError::OutOfBounds);
        }
        previous_end = end(*start, memory.len())?;
    }
    order.clear();
    for (index, job) in jobs.iter().enumerate() {
        let len = job
            .rows
            .checked_mul(job.cols)
            .ok_or(DenseError::ElementCountOverflow)?;
        if len != 0 {
            end(job.dst_offset, len)?;
            order.push(index);
        }
    }
    order.sort_unstable_by_key(|&j| (jobs[j].dst_offset, j));
    // Check every job against the segments before taking any memory.
    let mut segment = 0;
    let mut covered = 0;
    for &j in order.iter() {
        let (start, len) = (jobs[j].dst_offset, jobs[j].rows * jobs[j].cols);
        if start < covered {
            return Err(DenseError::OutOfBounds);
        }
        while segment < segments.len() && segments[segment].0 + segments[segment].1.len() <= start {
            segment += 1;
        }
        let Some((first, memory)) = segments.get(segment) else {
            return Err(DenseError::OutOfBounds);
        };
        if start < *first || start + len > first + memory.len() {
            return Err(DenseError::OutOfBounds);
        }
        covered = start + len;
    }
    Ok(())
}

/// [`split_fresh_segments`]'s placement, for segments and an `order`
/// [`check_fresh_segments`] accepted.
pub(crate) fn place_fresh_jobs<'a, E>(
    segments: &mut [(usize, &'a mut [E])],
    jobs: &[DenseGemmBatchJob],
    order: &[usize],
    mut place: impl FnMut(usize, &'a mut [E]),
) {
    let mut segment = 0;
    let mut remaining: &'a mut [E] = &mut [];
    let mut cursor = 0;
    for &j in order.iter() {
        let (start, len) = (jobs[j].dst_offset, jobs[j].rows * jobs[j].cols);
        while cursor + remaining.len() < start + len {
            let (first, memory) = &mut segments[segment];
            remaining = core::mem::take(memory);
            cursor = *first;
            segment += 1;
        }
        let (_, rest) = core::mem::take(&mut remaining).split_at_mut(start - cursor);
        let (piece, tail) = rest.split_at_mut(len);
        place(j, piece);
        remaining = tail;
        cursor = start + len;
    }
}

/// `zeroed(piece)`, refused unless it returns `piece` itself.
pub(crate) fn zero_piece<T>(
    zeroed: for<'x> fn(&'x mut [MaybeUninit<T>]) -> &'x mut [T],
    piece: &mut [MaybeUninit<T>],
) -> Result<&mut [T], DenseError> {
    let (start, len) = (piece.as_ptr().cast::<T>(), piece.len());
    let initialized = zeroed(piece);
    if core::ptr::eq(initialized.as_ptr(), start) && initialized.len() == len {
        Ok(initialized)
    } else {
        Err(DenseError::Unsupported {
            op: FRESH_OP,
            message: "the zero-fill callback returned storage other than its piece".to_string(),
        })
    }
}

/// [`DenseExecutor::matmul_batch_fresh`]'s default: zero each job's piece,
/// run it as a one-job [`DenseExecutor::matmul_batch_axpby_with_ops_into`]
/// with `beta = 0`, and report every job once all succeed.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fresh_serial<'a, E, T>(
    executor: &mut E,
    engine: &mut dyn Engine,
    parts: FreshParts<'_, 'a, T>,
    lhs: DenseView<'_, T>,
    rhs: DenseView<'_, T>,
    jobs: &[DenseGemmBatchJob],
    ops: [MatrixOp; 2],
    alpha: DenseScalar,
) -> Result<(), DenseError>
where
    E: DenseExecutor + ?Sized,
    T: FreshScalar,
{
    let mut pieces: Vec<Option<&'a mut [MaybeUninit<T>]>> = jobs.iter().map(|_| None).collect();
    split_fresh_segments(parts.segments, jobs, &mut Vec::new(), |j, piece| {
        pieces[j] = Some(piece)
    })?;
    let mut finished: Vec<&'a mut [T]> = Vec::with_capacity(jobs.len());
    for (job, piece) in jobs.iter().zip(pieces) {
        let Some(piece) = piece else {
            finished.push(&mut []);
            continue;
        };
        let output = zero_piece(parts.zeroed, piece)?;
        let (shape, strides) = ([job.rows, job.cols], [1, job.rows]);
        executor.matmul_batch_axpby_with_ops_into(
            engine,
            T::write(DenseViewMut::new(&mut *output, &shape, &strides, 0)?),
            T::read(lhs),
            T::read(rhs),
            &[DenseGemmBatchJob {
                dst_offset: 0,
                ..*job
            }],
            &[1],
            ops[0],
            ops[1],
            alpha,
            T::scalar(T::default()),
        )?;
        finished.push(output);
    }
    for (j, output) in finished.into_iter().enumerate() {
        (parts.written)(j, output);
    }
    Ok(())
}
