use super::adapter::sealed::LeafGate;
use super::*;
use strided_kernel::{RawStridedMut, RawStridedRef, UpdatePlan};

/// Inline capacity of [`FusedLayoutScratch`] and the traversal index.
///
/// Normalization stores every non-unit raw axis before fusing, so the inline
/// capacity bounds the raw rank, not the fused rank. Twice Strided's fused
/// limit keeps raw layouts that fuse back under that limit (unit legs, split
/// degeneracies) allocation-free, as well as fused ranks just above it.
pub(crate) const FUSED_LAYOUT_INLINE_RANK: usize = 2 * strided_kernel::RAW_FUSED_RANK_LIMIT;

/// Inline up to [`FUSED_LAYOUT_INLINE_RANK`] so a fresh adapter per eager
/// call allocates nothing.
#[derive(Debug, Default)]
pub(crate) struct FusedLayoutScratch {
    dims: SmallVec<[usize; FUSED_LAYOUT_INLINE_RANK]>,
    dst_strides: SmallVec<[isize; FUSED_LAYOUT_INLINE_RANK]>,
    src_strides: SmallVec<[isize; FUSED_LAYOUT_INLINE_RANK]>,
}

impl FusedLayoutScratch {
    pub(crate) fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub(crate) fn dst_strides(&self) -> &[isize] {
        &self.dst_strides
    }

    pub(crate) fn src_strides(&self) -> &[isize] {
        &self.src_strides
    }
}

#[derive(Debug, Default)]
pub(super) struct StridedKernelScratch {
    pub(super) layout: FusedLayoutScratch,
    pub(super) index: SmallVec<[usize; FUSED_LAYOUT_INLINE_RANK]>,
}

/// Borrowed view of a prebaked fused loop layout (issue #232).
///
/// Holds the exact `(dims, dst_strides, src_strides)` that `normalize_fused_layout`
/// would return for one (block, role) stride pair, computed once at compile
/// time in the immutable `TreeTransformLayoutTable` and reused across every
/// replay call instead of recomputed. The slices live in that table's arena;
/// the movement leaf `span` consumes them directly. dtype-independent — one
/// baked layout serves f64 and c64 alike (the normalization never inspects
/// values).
///
/// Safe downstream adapters retain direct read access to the normalized slices:
///
/// ```
/// use symtn_operations::BakedFusedLayout;
///
/// fn inspect(layout: BakedFusedLayout<'_>) {
///     let _ = (layout.dims, layout.dst_strides, layout.src_strides);
/// }
/// ```
///
/// Construction remains sealed:
///
/// ```compile_fail
/// use symtn_operations::BakedFusedLayout;
///
/// let _ = BakedFusedLayout {
///     dims: &[2],
///     dst_strides: &[1],
///     src_strides: &[1],
/// };
/// ```
///
/// Exhaustive destructuring is sealed as well:
///
/// ```compile_fail
/// use symtn_operations::BakedFusedLayout;
///
/// fn inspect(layout: BakedFusedLayout<'_>) {
///     let BakedFusedLayout { dims, dst_strides, src_strides } = layout;
/// }
/// ```
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct BakedFusedLayout<'a> {
    pub dims: &'a [usize],
    pub dst_strides: &'a [isize],
    pub src_strides: &'a [isize],
    /// Strided's compiled replay of exactly this layout, retained by the
    /// layout table for the roles a threaded leaf may replay.
    pub(crate) plan: Option<&'a UpdatePlan>,
}

impl<'a> BakedFusedLayout<'a> {
    pub(crate) fn try_from_normalized_slices(
        dims: &'a [usize],
        dst_strides: &'a [isize],
        src_strides: &'a [isize],
    ) -> Result<Self, OperationError> {
        if dims.is_empty() {
            return Err(OperationError::RankMismatch {
                expected: 1,
                actual: 0,
            });
        }
        if dims.len() != dst_strides.len() {
            return Err(OperationError::RankMismatch {
                expected: dims.len(),
                actual: dst_strides.len(),
            });
        }
        if dims.len() != src_strides.len() {
            return Err(OperationError::RankMismatch {
                expected: dims.len(),
                actual: src_strides.len(),
            });
        }
        if dims == [0] && dst_strides == [0] && src_strides == [0] {
            return Ok(Self {
                dims,
                dst_strides,
                src_strides,
                plan: None,
            });
        }
        if dims == [1] && dst_strides == [0] && src_strides == [0] {
            return Ok(Self {
                dims,
                dst_strides,
                src_strides,
                plan: None,
            });
        }
        if dims.iter().any(|&dim| dim <= 1) {
            return Err(OperationError::InvalidArgument {
                message: "baked fused layout is not normalized",
            });
        }
        crate::strided::element_count(dims)?;
        Ok(Self {
            dims,
            dst_strides,
            src_strides,
            plan: None,
        })
    }

    pub(crate) fn from_compiled_normalized_slices(
        dims: &'a [usize],
        dst_strides: &'a [isize],
        src_strides: &'a [isize],
        plan: Option<&'a UpdatePlan>,
    ) -> Self {
        // Why not revalidate here: the compiler validates these exact arena
        // slices, and compiles `plan` from them, before publishing the
        // immutable layout table.
        Self {
            dims,
            dst_strides,
            src_strides,
            plan,
        }
    }

    pub fn dims(&self) -> &'a [usize] {
        self.dims
    }

    pub fn dst_strides(&self) -> &'a [isize] {
        self.dst_strides
    }

    pub fn src_strides(&self) -> &'a [isize] {
        self.src_strides
    }
}

pub(super) fn validate_strided_ranks(
    shape: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
) -> Result<(), OperationError> {
    if shape.len() != dst_strides.len() {
        return Err(OperationError::RankMismatch {
            expected: shape.len(),
            actual: dst_strides.len(),
        });
    }
    if shape.len() != src_strides.len() {
        return Err(OperationError::RankMismatch {
            expected: shape.len(),
            actual: src_strides.len(),
        });
    }
    Ok(())
}

/// The compiled plan a leaf replays `role` (of `elements` elements) with on
/// `exec`: present when the role carries a plan and `gate` (Strided's own
/// copy cost rule, the rule `UpdatePlan` itself applies) gives `exec` more
/// than one worker. Callers ask only for overwrites.
///
/// Why not replay the plan at width one: it then runs the same serial loop
/// as the baked walk. Why roles above `RAW_FUSED_RANK_LIMIT` carry no plan:
/// strided's spilled replay allocates a cursor per worker range, while the
/// baked walk reuses the replay workspace's index.
///
/// Why `#[inline]`: every role write asks this, from other modules' item
/// loops; out of line, the serial answer costs each role a call (measured
/// per Direct item on the owned permute replay). Inlined, a serial token
/// answers with two loads and a compare in the caller.
#[inline]
pub(crate) fn admitted_plan<'a>(
    role: BakedFusedLayout<'a>,
    elements: usize,
    exec: &crate::host_pool::Exec<'_>,
    gate: &mut LeafGate<'_>,
) -> Option<&'a UpdatePlan> {
    let plan = role.plan?;
    gate.widens(exec, elements).then_some(plan)
}

/// `dst = map(src)` over `baked` by its compiled `plan` on `exec`. The
/// plan's alpha is one and its beta zero, so the destination is never read
/// and `map` is the whole element action, exactly as in [`span`].
///
/// [`span`]: super::movement::span
#[allow(clippy::too_many_arguments)]
pub(crate) fn planned_overwrite<T, F>(
    plan: &UpdatePlan,
    baked: BakedFusedLayout<'_>,
    dst: &mut [T],
    src: &[T],
    dst_offset: isize,
    src_offset: isize,
    exec: crate::host_pool::Exec<'_>,
    map: F,
) -> Result<(), OperationError>
where
    T: Copy + Send + Sync + Add<Output = T> + Mul<Output = T> + Zero + One + PartialEq,
    F: Fn(T) -> T + Send + Sync,
{
    let source = RawStridedRef::new(src, baked.dims, baked.src_strides, src_offset)
        .map_err(crate::strided::error)?;
    let mut destination = RawStridedMut::new(dst, baked.dims, baked.dst_strides, dst_offset)
        .map_err(crate::strided::error)?;
    plan.execute(&mut destination, &source, T::one(), T::zero(), map, exec)
        .map_err(crate::strided::error)
}

/// [`planned_overwrite`] into uninitialized memory: every element `baked`
/// addresses is written once, with alpha one so `map` is the whole action.
#[allow(clippy::too_many_arguments)]
pub(crate) fn planned_write_uninit<T, F>(
    plan: &UpdatePlan,
    baked: BakedFusedLayout<'_>,
    dst: &mut [core::mem::MaybeUninit<T>],
    src: &[T],
    dst_offset: isize,
    src_offset: isize,
    exec: crate::host_pool::Exec<'_>,
    map: F,
) -> Result<(), OperationError>
where
    T: Copy + Send + Sync + Mul<Output = T> + One + PartialEq,
    F: Fn(T) -> T + Send + Sync,
{
    let source = RawStridedRef::new(src, baked.dims, baked.src_strides, src_offset)
        .map_err(crate::strided::error)?;
    let mut destination = RawStridedMut::new(dst, baked.dims, baked.dst_strides, dst_offset)
        .map_err(crate::strided::error)?;
    plan.execute_uninit(&mut destination, &source, T::one(), map, exec)
        .map_err(crate::strided::error)
}

pub(crate) fn normalize_fused_layout(
    shape: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    scratch: &mut FusedLayoutScratch,
) -> Result<(), OperationError> {
    validate_strided_ranks(shape, dst_strides, src_strides)?;
    scratch.dims.clear();
    scratch.dst_strides.clear();
    scratch.src_strides.clear();

    if shape.contains(&0) {
        scratch.dims.push(0);
        scratch.dst_strides.push(0);
        scratch.src_strides.push(0);
        return Ok(());
    }
    crate::strided::element_count(shape)?;

    for axis in 0..shape.len() {
        if shape[axis] == 1 {
            continue;
        }
        let mut position = scratch.dims.len();
        while position > 0 && scratch.dst_strides[position - 1] > dst_strides[axis] {
            position -= 1;
        }
        scratch.dims.insert(position, shape[axis]);
        scratch.dst_strides.insert(position, dst_strides[axis]);
        scratch.src_strides.insert(position, src_strides[axis]);
    }
    if scratch.dims.is_empty() {
        scratch.dims.push(1);
        scratch.dst_strides.push(0);
        scratch.src_strides.push(0);
    }
    let mut fused = 0usize;
    for axis in 1..scratch.dims.len() {
        let extent = scratch.dims[fused] as isize;
        if scratch.dst_strides[fused].checked_mul(extent) == Some(scratch.dst_strides[axis])
            && scratch.src_strides[fused].checked_mul(extent) == Some(scratch.src_strides[axis])
        {
            scratch.dims[fused] = scratch.dims[fused]
                .checked_mul(scratch.dims[axis])
                .ok_or_else(|| OperationError::ElementCountOverflow)?;
        } else {
            fused += 1;
            scratch.dims[fused] = scratch.dims[axis];
            scratch.dst_strides[fused] = scratch.dst_strides[axis];
            scratch.src_strides[fused] = scratch.src_strides[axis];
        }
    }
    let rank = fused + 1;
    scratch.dims.truncate(rank);
    scratch.dst_strides.truncate(rank);
    scratch.src_strides.truncate(rank);
    Ok(())
}
