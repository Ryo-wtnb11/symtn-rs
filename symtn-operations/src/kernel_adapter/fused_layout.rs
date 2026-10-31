use super::*;

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
/// `apply_fused_pair_slices` consumes them directly. dtype-independent — one
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
            });
        }
        if dims == [1] && dst_strides == [0] && src_strides == [0] {
            return Ok(Self {
                dims,
                dst_strides,
                src_strides,
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
        })
    }

    #[inline]
    pub(crate) fn from_compiled_normalized_slices(
        dims: &'a [usize],
        dst_strides: &'a [isize],
        src_strides: &'a [isize],
    ) -> Self {
        // Why not revalidate here: the compiler validates these exact arena
        // slices before publishing the immutable layout table.
        Self {
            dims,
            dst_strides,
            src_strides,
        }
    }

    #[inline]
    pub fn dims(&self) -> &'a [usize] {
        self.dims
    }

    #[inline]
    pub fn dst_strides(&self) -> &'a [isize] {
        self.dst_strides
    }

    #[inline]
    pub fn src_strides(&self) -> &'a [isize] {
        self.src_strides
    }
}

#[inline]
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

pub(crate) fn for_each_fused_span<F>(
    dims: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
    index: &mut [usize],
    mut visit: F,
) where
    F: FnMut(isize, isize, usize, isize, isize),
{
    let rank = dims.len();
    debug_assert_eq!(rank, dst_strides.len());
    debug_assert_eq!(rank, src_strides.len());
    if rank == 0 || dims.contains(&0) {
        return;
    }
    debug_assert_eq!(rank, index.len());
    let inner_len = dims[0];
    let inner_dst = dst_strides[0];
    let inner_src = src_strides[0];
    index.fill(0);
    let mut dst_base = dst_offset;
    let mut src_base = src_offset;
    loop {
        visit(dst_base, src_base, inner_len, inner_dst, inner_src);
        let mut axis = 1;
        loop {
            if axis >= rank {
                return;
            }
            index[axis] += 1;
            dst_base += dst_strides[axis];
            src_base += src_strides[axis];
            if index[axis] < dims[axis] {
                break;
            }
            dst_base -= dims[axis] as isize * dst_strides[axis];
            src_base -= dims[axis] as isize * src_strides[axis];
            index[axis] = 0;
            axis += 1;
        }
    }
}

/// Fused loop nest over a prebaked layout writing into uninitialized memory
/// (issue #232, condition 2), mirroring `apply_fused_pair_slices` but with
/// `MaybeUninit::write` for the destination. Each destination offset is visited
/// exactly once, identical to `layout_linear_offset`'s odometer, so the
/// write-once-then-`assume_init` invariants of `initialize_owned` (#226/#233)
/// and of the pack scratch column writer are preserved: the normalization only drops extent-1 axes, reorders, and fuses
/// contiguous runs — the *set* of visited (dst, src) offsets is unchanged, and
/// there is no read-after-write within a single writer (`src` is a disjoint,
/// fully-initialized slice). The caller supplies runtime-length traversal scratch.
pub(crate) fn write_fused_uninit<D, F>(
    baked: BakedFusedLayout<'_>,
    dst: &mut [core::mem::MaybeUninit<D>],
    src: &[D],
    dst_offset: isize,
    src_offset: isize,
    index: &mut [usize],
    map: F,
) -> Result<(), OperationError>
where
    D: Copy,
    F: Fn(D) -> D,
{
    let dims = baked.dims();
    let dst_strides = baked.dst_strides();
    let src_strides = baked.src_strides();
    let Some(index) = index.get_mut(..dims.len()) else {
        return Err(OperationError::InvalidArgument {
            message: "fused traversal scratch is shorter than the normalized rank",
        });
    };
    apply_fused_pair_slices(
        dst,
        src,
        dims,
        dst_strides,
        src_strides,
        dst_offset,
        src_offset,
        index,
        |slot: &mut core::mem::MaybeUninit<D>, value| {
            slot.write(value);
        },
        map,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fused_pair<T, Apply, ElementOp>(
    scratch: &mut StridedKernelScratch,
    dst_data: &mut [T],
    src_data: &[T],
    shape: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
    apply: Apply,
    op: ElementOp,
) -> Result<(), OperationError>
where
    T: Copy,
    Apply: Fn(&mut T, T),
    ElementOp: Fn(T) -> T,
{
    let StridedKernelScratch { layout, index } = scratch;
    normalize_fused_layout(shape, dst_strides, src_strides, layout)?;
    index.resize(layout.dims.len(), 0);
    apply_fused_pair_slices(
        dst_data,
        src_data,
        &layout.dims,
        &layout.dst_strides,
        &layout.src_strides,
        dst_offset,
        src_offset,
        index.as_mut_slice(),
        apply,
        op,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_fused_pair_slices<Dst, T, Apply, ElementOp>(
    dst_data: &mut [Dst],
    src_data: &[T],
    dims: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
    index: &mut [usize],
    apply: Apply,
    op: ElementOp,
) where
    T: Copy,
    Apply: Fn(&mut Dst, T),
    ElementOp: Fn(T) -> T,
{
    for_each_fused_span(
        dims,
        dst_strides,
        src_strides,
        dst_offset,
        src_offset,
        index,
        |dst_base, src_base, inner_len, inner_dst, inner_src| {
            if inner_dst == 1 && inner_src == 1 {
                let dst_start = dst_base as usize;
                let src_start = src_base as usize;
                let dst = &mut dst_data[dst_start..dst_start + inner_len];
                let src = &src_data[src_start..src_start + inner_len];
                for position in 0..inner_len {
                    apply(&mut dst[position], op(src[position]));
                }
            } else {
                for position in 0..inner_len {
                    let dst_position = (dst_base + position as isize * inner_dst) as usize;
                    let src_position = (src_base + position as isize * inner_src) as usize;
                    apply(&mut dst_data[dst_position], op(src_data[src_position]));
                }
            }
        },
    );
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fused_pair_baked<T, Apply, ElementOp>(
    scratch: &mut StridedKernelScratch,
    baked: Option<BakedFusedLayout<'_>>,
    dst_data: &mut [T],
    src_data: &[T],
    shape: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
    apply: Apply,
    op: ElementOp,
) -> Result<(), OperationError>
where
    T: Copy,
    Apply: Fn(&mut T, T),
    ElementOp: Fn(T) -> T,
{
    match baked {
        Some(baked) => {
            scratch.index.resize(baked.dims().len(), 0);
            apply_fused_pair_slices(
                dst_data,
                src_data,
                baked.dims(),
                baked.dst_strides(),
                baked.src_strides(),
                dst_offset,
                src_offset,
                scratch.index.as_mut_slice(),
                apply,
                op,
            );
            Ok(())
        }
        None => fused_pair(
            scratch,
            dst_data,
            src_data,
            shape,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            apply,
            op,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn fused_pair_baked_with_index<T, Apply, ElementOp>(
    layout: &mut FusedLayoutScratch,
    baked: Option<BakedFusedLayout<'_>>,
    index: &mut [usize],
    dst_data: &mut [T],
    src_data: &[T],
    shape: &[usize],
    dst_strides: &[isize],
    src_strides: &[isize],
    dst_offset: isize,
    src_offset: isize,
    apply: Apply,
    op: ElementOp,
) -> Result<(), OperationError>
where
    T: Copy,
    Apply: Fn(&mut T, T),
    ElementOp: Fn(T) -> T,
{
    match baked {
        Some(baked) => {
            let rank = baked.dims().len();
            let Some(index) = index.get_mut(..rank) else {
                return Err(OperationError::InvalidArgument {
                    message: "fused traversal scratch is shorter than the normalized rank",
                });
            };
            apply_fused_pair_slices(
                dst_data,
                src_data,
                baked.dims(),
                baked.dst_strides(),
                baked.src_strides(),
                dst_offset,
                src_offset,
                index,
                apply,
                op,
            );
        }
        None => {
            normalize_fused_layout(shape, dst_strides, src_strides, layout)?;
            let rank = layout.dims.len();
            let Some(index) = index.get_mut(..rank) else {
                return Err(OperationError::InvalidArgument {
                    message: "fused traversal scratch is shorter than the normalized rank",
                });
            };
            apply_fused_pair_slices(
                dst_data,
                src_data,
                &layout.dims,
                &layout.dst_strides,
                &layout.src_strides,
                dst_offset,
                src_offset,
                index,
                apply,
                op,
            );
        }
    }
    Ok(())
}
