//! Owned block outputs that every destination block overwrites exactly once
//! (#1290).
//!
//! The destination buffer is left uninitialized only when
//! [`blocks_tile_storage`] proves that the blocks' reachable offsets
//! partition `0..required_len`. Each block is then written through a
//! [`BlockOverwrite`] that derives the destination strides and offset from
//! the block itself, so a caller supplies only source geometry and cannot
//! aim a write elsewhere; [`initialize_owned`] publishes the length only after
//! every block reported a complete write. A structure without that proof is
//! written into a zeroed buffer through the initialized kernels, which is the
//! fill-then-overwrite sequence these outputs used before.

use core::mem::MaybeUninit;
use core::ops::{Add, Mul, Range};

use num_traits::{One, Zero};
use smallvec::SmallVec;
use std::borrow::Cow;

use symtn_core::{BlockRef, BlockStructure};

use crate::owned_overwrite_buffer::initialize_owned;
use crate::{CheckedBlockLayout, ConjugateValue, OperationError};

enum Storage<'a, D> {
    Uninit(&'a mut [MaybeUninit<D>]),
    Zeroed(&'a mut [D]),
}

/// A trailing member axis of a stacked output: `members` copies of the
/// destination structure at stride `destination_stride`, read from a source
/// whose members sit `source_stride` apart.
#[derive(Clone, Copy)]
struct MemberAxis {
    members: usize,
    destination_stride: usize,
    source_stride: usize,
}

/// The single write of one destination block of an owned output.
///
/// For a stacked output ([`overwrite_owned_member_blocks`]) the block is
/// written for every member at once: the member axis is one more strided
/// dimension of the same copy.
#[doc(hidden)]
pub struct BlockOverwrite<'a, D> {
    storage: Storage<'a, D>,
    layout: &'a mut CheckedBlockLayout,
    block: BlockRef<'a>,
    member_axis: Option<MemberAxis>,
    written: bool,
}

impl<D> BlockOverwrite<'_, D>
where
    D: Copy + Add<D, Output = D> + Mul<D, Output = D> + PartialEq + Zero + One + ConjugateValue,
{
    fn claim(&self) -> Result<(), OperationError> {
        if self.written {
            return Err(OperationError::InvalidArgument {
                message: "owned output block written twice",
            });
        }
        Ok(())
    }

    /// `block = alpha * op(source)`, where `source_stride(axis)` is the
    /// source stride of the block's logical `axis`, and `op` conjugates when
    /// `conjugate` is set. `alpha = 1` is a bit-exact copy.
    pub fn copy(
        &mut self,
        source_stride: impl FnMut(usize) -> Result<usize, OperationError>,
        source: &[D],
        source_offset: usize,
        conjugate: bool,
        alpha: D,
    ) -> Result<(), OperationError> {
        self.claim()?;
        let block = self.block;
        self.write_piece(
            &block_shape(block.shape(), self.member_axis),
            block.offset(),
            source_stride,
            source,
            source_offset,
            conjugate,
            alpha,
        )?;
        self.written = true;
        Ok(())
    }

    /// [`Self::copy`] of an order-preserving gather: on every logical axis
    /// the block's `extent` destination positions read the source positions
    /// `runs[axis]` (all of `0..extent` for `None`), in order. `source_offset`
    /// is the source block's origin; the run starts are added to it here.
    ///
    /// One strided copy per piece of the cartesian product of the per-axis
    /// runs, so a single run per axis is exactly one [`Self::copy`].
    ///
    /// # Errors
    ///
    /// Before any element is written: [`OperationError::InvalidArgument`]
    /// unless `runs` has one entry per axis and every `Some` entry is
    /// nonempty, sorted, disjoint runs whose lengths sum to the block's
    /// extent on that axis ([`check_axis_runs`]), and an offset error when
    /// the reachable source envelope leaves `source`.
    pub fn copy_runs(
        &mut self,
        mut source_stride: impl FnMut(usize) -> Result<usize, OperationError>,
        source: &[D],
        source_offset: usize,
        conjugate: bool,
        alpha: D,
        runs: &[AxisRuns<'_>],
    ) -> Result<(), OperationError> {
        self.claim()?;
        let block = self.block;
        let shape = block.shape();
        // SAFETY (of the uninitialized publish this write contributes to):
        // `initialize_owned` publishes the buffer once every block reported
        // `written`, relying on each block's write to reach every element of
        // its own destination layout. A gather reaches the destination only
        // through the pieces of `for_each_run_piece`; their packed
        // rectangles partition the block exactly when, per axis, the run
        // lengths sum to the block extent. This check is that invariant. It
        // is owned here, not by the caller, because `BlockOverwrite` is the
        // owner of the `MaybeUninit` contract: no runs a safe caller passes
        // can mark the block written with an element left unreached, since
        // any violation returns before the first write and `written` stays
        // false (an unwritten block aborts the output).
        check_axis_runs(shape, runs)?;
        let mut source_strides: SmallVec<[usize; 8]> = SmallVec::with_capacity(shape.len());
        for axis in 0..shape.len() {
            source_strides.push(source_stride(axis)?);
        }
        self.check_source_envelope(shape, runs, &source_strides, source.len(), source_offset)?;
        let member_axis = self.member_axis;
        let strides = block.strides();
        let source_strides = &source_strides;
        for_each_run_piece(shape, runs, |piece, packed, sparse| {
            let destination_offset = linear_offset(block.offset(), packed, strides)?;
            let piece_source_offset = linear_offset(source_offset, sparse, source_strides)?;
            let mut piece_shape: SmallVec<[usize; 9]> = SmallVec::from_slice(piece);
            if let Some(member) = member_axis {
                piece_shape.push(member.members);
            }
            self.write_piece(
                &piece_shape,
                destination_offset,
                |axis| Ok(source_strides[axis]),
                source,
                piece_source_offset,
                conjugate,
                alpha,
            )
        })?;
        self.written = true;
        Ok(())
    }

    /// Rejects, before the first piece is written, a gather whose farthest
    /// source element lies outside `source`, so a failing gather writes
    /// nothing. Strides are non-negative, so the farthest element is the
    /// last position of every axis' last run (and the last member).
    fn check_source_envelope(
        &self,
        shape: &[usize],
        runs: &[AxisRuns<'_>],
        source_strides: &[usize],
        source_len: usize,
        source_offset: usize,
    ) -> Result<(), OperationError> {
        if shape.contains(&0) || self.member_axis.is_some_and(|member| member.members == 0) {
            return Ok(());
        }
        let mut last = source_offset;
        for (axis, (&extent, &stride)) in shape.iter().zip(source_strides).enumerate() {
            let position = match runs[axis] {
                None => extent - 1,
                Some(runs) => runs.last().map_or(0, |run| run.end - 1),
            };
            last = position
                .checked_mul(stride)
                .and_then(|step| last.checked_add(step))
                .ok_or(OperationError::ElementCountOverflow)?;
        }
        if let Some(member) = self.member_axis {
            last = (member.members - 1)
                .checked_mul(member.source_stride)
                .and_then(|step| last.checked_add(step))
                .ok_or(OperationError::ElementCountOverflow)?;
        }
        if last >= source_len {
            return Err(OperationError::OffsetOverflow { value: last });
        }
        Ok(())
    }

    /// One strided copy of `shape` (the member count trailing for a stacked
    /// output) from `source_offset` to `destination_offset` along the block's
    /// own destination strides.
    #[allow(clippy::too_many_arguments)]
    fn write_piece(
        &mut self,
        shape: &[usize],
        destination_offset: usize,
        mut source_stride: impl FnMut(usize) -> Result<usize, OperationError>,
        source: &[D],
        source_offset: usize,
        conjugate: bool,
        alpha: D,
    ) -> Result<(), OperationError> {
        let strides = self.block.strides();
        let rank = strides.len();
        let member_axis = self.member_axis;
        self.layout.fill_one(shape, |axis| match member_axis {
            Some(member) if axis == rank => Ok((member.destination_stride, member.source_stride)),
            _ => Ok((strides[axis], source_stride(axis)?)),
        })?;
        let dst_offset = checked_offset(destination_offset)?;
        let src_offset = checked_offset(source_offset)?;
        match &mut self.storage {
            Storage::Uninit(dst) => self
                .layout
                .copy_uninit(dst, source, dst_offset, src_offset, conjugate, alpha)?,
            Storage::Zeroed(dst) => self.layout.tensoradd(
                dst,
                source,
                dst_offset,
                src_offset,
                conjugate,
                alpha,
                D::zero(),
            )?,
        }
        Ok(())
    }

    /// `block = alpha * op_l(lhs) + beta * op_r(rhs)` with the arithmetic of
    /// [`CheckedBlockLayout::add_two_source`]; a zero coefficient drops its
    /// source, and both zero leave the block zero. `source_strides(axis)`
    /// returns the `(lhs, rhs)` strides of logical `axis`, and both are
    /// converted before any write whichever coefficients are zero. Each
    /// source is `(data, offset, conjugate)`.
    pub fn add(
        &mut self,
        mut source_strides: impl FnMut(usize) -> Result<(usize, usize), OperationError>,
        (lhs, lhs_offset, lhs_conjugate): (&[D], usize, bool),
        (rhs, rhs_offset, rhs_conjugate): (&[D], usize, bool),
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError> {
        self.claim()?;
        if self.member_axis.is_some() {
            return Err(OperationError::InvalidArgument {
                message: "a stacked owned output supports block copies only",
            });
        }
        let strides = self.block.strides();
        // With `alpha = 0` the rhs is the layout's first source, which the
        // single-source walks read.
        let swap = alpha.is_zero();
        self.layout.fill_two(self.block.shape(), |axis| {
            let (l, r) = source_strides(axis)?;
            let (first, second) = if swap { (r, l) } else { (l, r) };
            Ok((strides[axis], first, second))
        })?;
        let dst_offset = checked_offset(self.block.offset())?;
        let lhs_offset = checked_offset(lhs_offset)?;
        let rhs_offset = checked_offset(rhs_offset)?;
        let single = match (alpha.is_zero(), beta.is_zero()) {
            (false, false) => None,
            (false, true) => Some((lhs, lhs_offset, lhs_conjugate, alpha)),
            (true, false) => Some((rhs, rhs_offset, rhs_conjugate, beta)),
            (true, true) => {
                self.write_zero()?;
                self.written = true;
                return Ok(());
            }
        };
        match (&mut self.storage, single) {
            (Storage::Uninit(dst), None) => self.layout.add_two_source_uninit(
                dst,
                lhs,
                rhs,
                dst_offset,
                lhs_offset,
                rhs_offset,
                lhs_conjugate,
                rhs_conjugate,
                alpha,
                beta,
            )?,
            (Storage::Zeroed(dst), None) => self.layout.add_two_source(
                dst,
                lhs,
                rhs,
                dst_offset,
                lhs_offset,
                rhs_offset,
                lhs_conjugate,
                rhs_conjugate,
                alpha,
                beta,
            )?,
            (Storage::Uninit(dst), Some((source, offset, conjugate, factor))) => self
                .layout
                .copy_uninit(dst, source, dst_offset, offset, conjugate, factor)?,
            (Storage::Zeroed(dst), Some((source, offset, conjugate, factor))) => {
                self.layout.tensoradd(
                    dst,
                    source,
                    dst_offset,
                    offset,
                    conjugate,
                    factor,
                    D::zero(),
                )?
            }
        }
        self.written = true;
        Ok(())
    }

    /// Leaves the block zero.
    pub fn zero(&mut self) -> Result<(), OperationError> {
        self.claim()?;
        self.write_zero()?;
        self.written = true;
        Ok(())
    }

    fn write_zero(&mut self) -> Result<(), OperationError> {
        match &mut self.storage {
            // Walks the block's own layout, as every other write does: a
            // tiled block need not be a contiguous range (coupled-sector
            // blocks are sub-matrices of their sector matrix). The source is
            // one zero read with stride zero on every axis.
            Storage::Uninit(dst) => {
                let strides = self.block.strides();
                let member_stride = self.member_axis.map(|member| member.destination_stride);
                let shape = block_shape(self.block.shape(), self.member_axis);
                self.layout.fill_one(&shape, |axis| {
                    Ok((strides.get(axis).copied().or(member_stride).unwrap_or(0), 0))
                })?;
                self.layout.copy_uninit(
                    dst,
                    &[D::zero()],
                    checked_offset(self.block.offset())?,
                    0,
                    false,
                    D::one(),
                )
            }
            // Why not write zeros: the buffer starts zeroed, and an unproved
            // layout may alias, where the fill-then-overwrite sequence this
            // path preserves left other blocks' writes in place.
            Storage::Zeroed(_) => Ok(()),
        }
    }
}

/// A block's extents, followed by the member count for a stacked output.
///
/// Why borrowed without a member axis: an unstacked block of rank above the
/// inline capacity would otherwise spill one more buffer per block.
fn block_shape(shape: &[usize], member_axis: Option<MemberAxis>) -> Cow<'_, [usize]> {
    match member_axis {
        None => Cow::Borrowed(shape),
        Some(member) => Cow::Owned([shape, &[member.members]].concat()),
    }
}

fn checked_offset(offset: usize) -> Result<isize, OperationError> {
    isize::try_from(offset).map_err(|_| OperationError::OffsetOverflow { value: offset })
}

/// One axis of an order-preserving gather or scatter: `None` is the whole
/// axis, `Some(runs)` the positions on the sparse side, as sorted, disjoint,
/// nonempty half-open runs. The packed side holds the same positions
/// consecutively, in order.
#[doc(hidden)]
pub type AxisRuns<'a> = Option<&'a [Range<usize>]>;

/// Checks that `runs` packs exactly `packed[axis]` positions on every axis:
/// one entry per axis, and each `Some` entry nonempty, sorted, disjoint runs
/// whose lengths sum to that extent. The pieces of [`for_each_run_piece`]
/// then partition the packed block.
///
/// Bounds on the sparse side (`last run end <= sparse extent`) are the
/// caller's, which knows that extent.
#[doc(hidden)]
pub fn check_axis_runs(packed: &[usize], runs: &[AxisRuns<'_>]) -> Result<(), OperationError> {
    if runs.len() != packed.len() {
        return Err(OperationError::InvalidArgument {
            message: "axis runs must name every block axis once",
        });
    }
    for (&extent, runs) in packed.iter().zip(runs) {
        let Some(runs) = runs else { continue };
        let mut covered = 0usize;
        let mut floor = 0usize;
        for run in *runs {
            if run.start < floor || run.start >= run.end {
                return Err(OperationError::InvalidArgument {
                    message: "axis runs must be nonempty, sorted and disjoint",
                });
            }
            floor = run.end;
            covered += run.end - run.start;
        }
        if covered != extent {
            return Err(OperationError::InvalidArgument {
                message: "axis runs must cover the packed block extent exactly",
            });
        }
    }
    Ok(())
}

/// Calls `piece(shape, packed_start, sparse_start)` once per piece of the
/// cartesian product of the per-axis runs, in column-major piece order.
/// `packed` gives each axis' packed extent (the run of a `None` axis).
/// Inline up to rank 8: no heap allocation per call or piece.
///
/// `runs` must have passed [`check_axis_runs`] against `packed`.
#[doc(hidden)]
pub fn for_each_run_piece<E>(
    packed: &[usize],
    runs: &[AxisRuns<'_>],
    mut piece: impl FnMut(&[usize], &[usize], &[usize]) -> Result<(), E>,
) -> Result<(), E> {
    let run = |axis: usize, index: usize| match runs[axis] {
        None => 0..packed[axis],
        Some(runs) => runs[index].clone(),
    };
    let count = |axis: usize| runs[axis].map_or(1, <[Range<usize>]>::len);
    let rank = packed.len();
    if (0..rank).any(|axis| count(axis) == 0) {
        return Ok(());
    }
    let mut index: SmallVec<[usize; 8]> = SmallVec::from_elem(0, rank);
    let mut shape: SmallVec<[usize; 8]> = (0..rank).map(|axis| run(axis, 0).len()).collect();
    let mut packed_start: SmallVec<[usize; 8]> = SmallVec::from_elem(0, rank);
    let mut sparse_start: SmallVec<[usize; 8]> = (0..rank).map(|axis| run(axis, 0).start).collect();
    loop {
        piece(&shape, &packed_start, &sparse_start)?;
        let mut axis = 0;
        loop {
            if axis == rank {
                return Ok(());
            }
            index[axis] += 1;
            if index[axis] < count(axis) {
                packed_start[axis] += shape[axis];
                let next = run(axis, index[axis]);
                shape[axis] = next.len();
                sparse_start[axis] = next.start;
                break;
            }
            let first = run(axis, 0);
            index[axis] = 0;
            packed_start[axis] = 0;
            shape[axis] = first.len();
            sparse_start[axis] = first.start;
            axis += 1;
        }
    }
}

/// `origin + sum_a position[a] * stride[a]`, checked.
fn linear_offset(
    origin: usize,
    position: &[usize],
    strides: &[usize],
) -> Result<usize, OperationError> {
    position
        .iter()
        .zip(strides)
        .try_fold(origin, |offset, (&position, &stride)| {
            position
                .checked_mul(stride)
                .and_then(|step| offset.checked_add(step))
        })
        .ok_or(OperationError::ElementCountOverflow)
}

/// Whether the blocks' reachable offsets partition `0..len`, so writing every
/// block once initializes every element exactly once. Sufficient, not
/// necessary; `false` only costs the zero fill. Two proofs are accepted, both
/// without allocation or hashing:
///
/// - the tiling its constructor recorded on the structure
///   ([`BlockStructure::storage_tiling_proven`]: the canonical coupled-sector
///   layout built from leg degeneracies, TensorKit's block layout);
/// - non-empty blocks that, in index order, are each compact (column-major
///   under some axis permutation) and contiguous from offset zero to `len`.
///
/// Why not `BlockStructure::coupled_sector_regions`: it compiles hashed region
/// metadata on the structure wrapper, which the interner rebuilds whenever the
/// last owner drops it, so a steady-state eager call would pay it every time.
fn blocks_tile_storage(structure: &BlockStructure, len: usize) -> Result<bool, OperationError> {
    Ok(structure.storage_tiling_proven() || compact_blocks_tile_storage(structure, len)?)
}

fn compact_blocks_tile_storage(
    structure: &BlockStructure,
    len: usize,
) -> Result<bool, OperationError> {
    let mut next = 0usize;
    for index in 0..structure.block_count() {
        let block = structure.block(index)?;
        let count = block.element_count()?;
        if count == 0 {
            continue;
        }
        if block.offset() != next || !is_compact(block.shape(), block.strides()) {
            return Ok(false);
        }
        next = next
            .checked_add(count)
            .ok_or(OperationError::ElementCountOverflow)?;
    }
    Ok(next == len)
}

/// For a block with no zero extent: whether its strides, ordered, are the
/// column-major strides of its extents, so it reaches `0..count` from its
/// offset, each once. Each step needs exactly one non-unit axis whose stride
/// is the running product; the product strictly grows, so no axis is used
/// twice. Why not sort the axes: that would spill a buffer past the inline
/// rank on the output path this module exists to keep allocation-free.
fn is_compact(shape: &[usize], strides: &[usize]) -> bool {
    let axes = || shape.iter().zip(strides).filter(|(&extent, _)| extent > 1);
    let mut expected = 1usize;
    for _ in axes() {
        let mut next = axes().filter(|(_, &stride)| stride == expected);
        let (Some((&extent, _)), None) = (next.next(), next.next()) else {
            return false;
        };
        // Cannot overflow: the block's element count was checked.
        expected *= extent;
    }
    true
}

/// Allocates `destination`'s payload and writes every block once through
/// `write_block`, in block order. Each call must complete exactly one
/// [`BlockOverwrite`] write or return an error; a block left unwritten is an
/// error, and on any error or panic no partially written buffer escapes.
#[doc(hidden)]
pub fn overwrite_owned_blocks<D>(
    destination: &BlockStructure,
    write_block: impl FnMut(BlockRef<'_>, &mut BlockOverwrite<'_, D>) -> Result<(), OperationError>,
) -> Result<Vec<D>, OperationError>
where
    D: Copy + Add<D, Output = D> + Mul<D, Output = D> + PartialEq + Zero + One + ConjugateValue,
{
    overwrite_owned_blocks_in(destination, true, None, write_block)
}

/// [`overwrite_owned_blocks`] for `members` outputs of one structure stacked
/// at stride `destination.required_len()`, read from a source whose members
/// sit `source_member_stride` apart: each [`BlockOverwrite::copy`] writes its
/// block, and each piece of a [`BlockOverwrite::copy_runs`] gather its piece,
/// for every member in one strided pass with a trailing member axis.
///
/// The buffer is left uninitialized under the same tiling proof as
/// [`overwrite_owned_blocks`], taken on the member structure: if its blocks
/// partition `0..L`, the (block, member) rectangles partition `0..members * L`,
/// member `m`'s image being the member-0 image shifted by `m * L`. Only
/// [`BlockOverwrite::copy`] and [`BlockOverwrite::copy_runs`] are supported.
#[doc(hidden)]
pub fn overwrite_owned_member_blocks<D>(
    destination: &BlockStructure,
    members: usize,
    source_member_stride: usize,
    write_block: impl FnMut(BlockRef<'_>, &mut BlockOverwrite<'_, D>) -> Result<(), OperationError>,
) -> Result<Vec<D>, OperationError>
where
    D: Copy + Add<D, Output = D> + Mul<D, Output = D> + PartialEq + Zero + One + ConjugateValue,
{
    let member_axis = MemberAxis {
        members,
        destination_stride: destination.required_len()?,
        source_stride: source_member_stride,
    };
    overwrite_owned_blocks_in(destination, true, Some(member_axis), write_block)
}

fn overwrite_owned_blocks_in<D>(
    destination: &BlockStructure,
    allow_uninit: bool,
    member_axis: Option<MemberAxis>,
    mut write_block: impl FnMut(BlockRef<'_>, &mut BlockOverwrite<'_, D>) -> Result<(), OperationError>,
) -> Result<Vec<D>, OperationError>
where
    D: Copy + Add<D, Output = D> + Mul<D, Output = D> + PartialEq + Zero + One + ConjugateValue,
{
    let member_len = destination.required_len()?;
    let len = match member_axis {
        Some(member) => member_len
            .checked_mul(member.members)
            .ok_or(OperationError::ElementCountOverflow)?,
        None => member_len,
    };
    let mut layout = CheckedBlockLayout::default();
    let mut write_all = |mut storage: Storage<'_, D>| -> Result<(), OperationError> {
        for index in 0..destination.block_count() {
            let block = destination.block(index)?;
            let mut writer = BlockOverwrite {
                storage: match &mut storage {
                    Storage::Uninit(dst) => Storage::Uninit(dst),
                    Storage::Zeroed(dst) => Storage::Zeroed(dst),
                },
                layout: &mut layout,
                block,
                member_axis,
                written: false,
            };
            write_block(block, &mut writer)?;
            if !writer.written {
                return Err(OperationError::InvalidArgument {
                    message: "owned output block left unwritten",
                });
            }
        }
        Ok(())
    };
    if allow_uninit && blocks_tile_storage(destination, member_len)? {
        return initialize_owned(len, |dst| write_all(Storage::Uninit(dst)));
    }
    #[cfg(any(test, feature = "testing"))]
    crate::probe::add(crate::probe::Event::OwnedPrefillElement, len);
    let mut data = vec![D::zero(); len];
    write_all(Storage::Zeroed(&mut data))?;
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::probe::take_owned_block_prefills;
    use num_complex::Complex64;
    use symtn_core::{BlockKey, BlockSpec};

    fn structure(blocks: &[(&[usize], &[usize], usize)]) -> BlockStructure {
        BlockStructure::from_blocks_with_rank(
            3,
            blocks
                .iter()
                .enumerate()
                .map(|(index, &(shape, strides, offset))| {
                    BlockSpec::with_key(
                        BlockKey::opaque([index as u64]),
                        shape.to_vec(),
                        strides.to_vec(),
                        offset,
                    )
                    .unwrap()
                })
                .collect(),
        )
        .unwrap()
    }

    /// Column-major, an empty block, an axis-permuted compact block, and
    /// extent-one axes with arbitrary strides: 23 elements, tiled in order.
    fn tiled() -> BlockStructure {
        structure(&[
            (&[2, 3, 1], &[1, 2, 6], 0),
            (&[0, 4, 2], &[1, 1, 4], 6),
            (&[3, 2, 2], &[4, 1, 2], 6),
            (&[1, 5, 1], &[7, 1, 9], 18),
        ])
    }

    fn values() -> Vec<Complex64> {
        let special = [
            Complex64::new(-0.0, 0.0),
            Complex64::new(f64::NAN, 1.0),
            Complex64::new(f64::INFINITY, -0.0),
            Complex64::new(f64::MIN_POSITIVE / 4.0, -2.5),
        ];
        (0..96)
            .map(|i| {
                special
                    .get(i % 11)
                    .copied()
                    .unwrap_or(Complex64::new(i as f64 * 0.75 - 9.0, 1.0 - i as f64 * 0.5))
            })
            .collect()
    }

    fn bits(values: &[Complex64]) -> Vec<(u64, u64)> {
        // Why canonicalize under Miri only: Miri picks arithmetic NaN sign
        // and payload nondeterministically; native runs compare them exactly.
        let bits = |value: f64| {
            if cfg!(miri) && value.is_nan() {
                f64::NAN.to_bits()
            } else {
                value.to_bits()
            }
        };
        values
            .iter()
            .map(|value| (bits(value.re), bits(value.im)))
            .collect()
    }

    /// Padded, transposed source strides (and a reversed axis order) per
    /// block, all in bounds of `values()`.
    fn source_stride(block: usize, axis: usize) -> usize {
        [[1, 5, 30], [2, 1, 9], [15, 1, 3], [40, 13, 1]][block][axis]
    }

    type Writer<'a> = BlockOverwrite<'a, Complex64>;

    fn write_copy(
        allow_uninit: bool,
        conjugate: bool,
        alpha: Complex64,
    ) -> Result<Vec<Complex64>, OperationError> {
        let source = values();
        let mut index = 0;
        overwrite_owned_blocks_in(
            &tiled(),
            allow_uninit,
            None,
            |_, writer: &mut Writer<'_>| {
                let block = index;
                index += 1;
                writer.copy(
                    |axis| Ok(source_stride(block, axis)),
                    &source,
                    2 + block,
                    conjugate,
                    alpha,
                )
            },
        )
    }

    fn write_add(allow_uninit: bool, alpha: Complex64, beta: Complex64) -> Vec<Complex64> {
        let lhs = values();
        let rhs: Vec<Complex64> = values().into_iter().rev().collect();
        let mut index = 0;
        overwrite_owned_blocks_in(
            &tiled(),
            allow_uninit,
            None,
            |_, writer: &mut Writer<'_>| {
                let block = index;
                index += 1;
                writer.add(
                    |axis| Ok((source_stride(block, axis), source_stride(3 - block, axis))),
                    (&lhs, block, block % 2 == 0),
                    (&rhs, 1, block % 2 == 1),
                    alpha,
                    beta,
                )
            },
        )
        .unwrap()
    }

    /// What: a stacked output equals the per-member outputs laid end to end,
    /// bit for bit, with and without the tiling proof; the member axis never
    /// triggers a fill of its own, and a two-source add is refused.
    #[test]
    fn stacked_copies_equal_the_per_member_outputs_bitwise() {
        const MEMBERS: usize = 3;
        let member_source = |member: usize| -> Vec<Complex64> {
            values()
                .into_iter()
                .map(|value| value * Complex64::new(1.0 + member as f64, -(member as f64)))
                .collect()
        };
        let stacked: Vec<Complex64> = (0..MEMBERS).flat_map(member_source).collect();
        let one = Complex64::new(1.0, 0.0);
        let run = |allow_uninit: bool| {
            let mut index = 0;
            let member_axis = MemberAxis {
                members: MEMBERS,
                destination_stride: 23,
                source_stride: 96,
            };
            overwrite_owned_blocks_in(
                &tiled(),
                allow_uninit,
                Some(member_axis),
                |_, writer: &mut Writer<'_>| {
                    let block = index;
                    index += 1;
                    writer.copy(
                        |axis| Ok(source_stride(block, axis)),
                        &stacked,
                        2 + block,
                        block % 2 == 1,
                        one,
                    )
                },
            )
            .unwrap()
        };
        let expected: Vec<Complex64> = (0..MEMBERS)
            .flat_map(|member| {
                let source = member_source(member);
                let mut index = 0;
                overwrite_owned_blocks_in(&tiled(), true, None, |_, writer: &mut Writer<'_>| {
                    let block = index;
                    index += 1;
                    writer.copy(
                        |axis| Ok(source_stride(block, axis)),
                        &source,
                        2 + block,
                        block % 2 == 1,
                        one,
                    )
                })
                .unwrap()
            })
            .collect();
        take_owned_block_prefills();
        let uninit = run(true);
        assert_eq!(take_owned_block_prefills(), 0);
        assert_eq!(bits(&uninit), bits(&expected));
        let zeroed = run(false);
        assert_eq!(take_owned_block_prefills(), 23 * MEMBERS);
        assert_eq!(bits(&zeroed), bits(&expected));

        let refused = overwrite_owned_member_blocks(&tiled(), MEMBERS, 96, |_, writer| {
            writer.add(
                |_| Ok((1, 1)),
                (&stacked, 0, false),
                (&stacked, 0, false),
                one,
                one,
            )
        });
        assert!(matches!(
            refused,
            Err(OperationError::InvalidArgument { .. })
        ));
    }

    /// What: a tiled layout skips the fill and stores the same bits as the
    /// base sequence (zero fill, then the initialized kernels), for copies,
    /// scaled and conjugated copies, and every coefficient case of an add.
    #[test]
    fn tiled_blocks_skip_the_fill_and_match_the_zero_fill_sequence_bitwise() {
        let one = Complex64::new(1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);
        let scaled = Complex64::new(0.5, -0.25);
        for conjugate in [false, true] {
            for alpha in [one, scaled, zero] {
                take_owned_block_prefills();
                let uninit = write_copy(true, conjugate, alpha).unwrap();
                assert_eq!(take_owned_block_prefills(), 0);
                let base = write_copy(false, conjugate, alpha).unwrap();
                assert_eq!(take_owned_block_prefills(), 23);
                assert_eq!(bits(&uninit), bits(&base));
            }
        }
        for (alpha, beta) in [
            (one, scaled),
            (scaled, -one),
            (zero, scaled),
            (scaled, zero),
            (zero, zero),
        ] {
            take_owned_block_prefills();
            let uninit = write_add(true, alpha, beta);
            assert_eq!(take_owned_block_prefills(), 0);
            assert_eq!(bits(&uninit), bits(&write_add(false, alpha, beta)));
        }
    }

    /// A U(1) `V ⊗ V <- V ⊗ V` coupled-sector layout: several trees per
    /// coupled sector, so blocks are strided sub-matrices that tile only
    /// jointly (the case the compact proof rejects).
    fn coupled() -> std::sync::Arc<BlockStructure> {
        use symtn_core::{
            FusionProductSpace, FusionTreeHomSpace, SectorLeg, U1FusionRule, U1Irrep,
        };
        let leg = || {
            SectorLeg::new(
                [
                    (U1Irrep::new(-1).sector_id(), 2),
                    (U1Irrep::new(0).sector_id(), 1),
                    (U1Irrep::new(1).sector_id(), 3),
                ],
                false,
            )
        };
        let product = || FusionProductSpace::new([leg(), leg()]);
        FusionTreeHomSpace::new(product(), product())
            .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
            .unwrap()
    }

    /// What: on a coupled-sector layout whose blocks are not contiguous
    /// ranges, the unfilled output stores the base sequence's bits for a
    /// transposing conjugated copy, every add coefficient case (both zero
    /// included), and an explicit zero block.
    #[test]
    fn coupled_sector_blocks_skip_the_fill_and_match_the_zero_fill_sequence_bitwise() {
        let structure = coupled();
        let len = structure.required_len().unwrap();
        assert!((0..structure.block_count()).any(|index| {
            let block = structure.block(index).unwrap();
            block.element_count().unwrap() > 0 && !is_compact(block.shape(), block.strides())
        }));
        assert!(!compact_blocks_tile_storage(&structure, len).unwrap());
        assert!(structure.storage_tiling_proven());
        let lhs = values().into_iter().cycle().take(len).collect::<Vec<_>>();
        let rhs = lhs.iter().rev().copied().collect::<Vec<_>>();
        let one = Complex64::new(1.0, 0.0);
        let zero = Complex64::new(0.0, 0.0);
        let scaled = Complex64::new(0.5, -0.25);
        let run = |allow_uninit: bool, alpha: Complex64, beta: Complex64, mode: u8| {
            overwrite_owned_blocks_in(
                &structure,
                allow_uninit,
                None,
                |block, writer: &mut Writer<'_>| {
                    let strides = block.strides();
                    // Reversed axis order reads a transposed, still in-bounds view.
                    let rank = strides.len();
                    match mode {
                        0 => writer.copy(
                            |axis| Ok(strides[rank - 1 - axis]),
                            &lhs,
                            block.offset(),
                            true,
                            alpha,
                        ),
                        1 => writer.add(
                            |axis| Ok((strides[axis], strides[axis])),
                            (&lhs, block.offset(), false),
                            (&rhs, block.offset(), true),
                            alpha,
                            beta,
                        ),
                        _ => writer.zero(),
                    }
                },
            )
            .unwrap()
        };
        for (alpha, beta, mode) in [
            (one, zero, 0),
            (scaled, zero, 0),
            (one, scaled, 1),
            (scaled, -one, 1),
            (zero, scaled, 1),
            (scaled, zero, 1),
            (zero, zero, 1),
            (zero, zero, 2),
        ] {
            take_owned_block_prefills();
            let uninit = run(true, alpha, beta, mode);
            assert_eq!(take_owned_block_prefills(), 0);
            assert_eq!(
                bits(&uninit),
                bits(&run(false, alpha, beta, mode)),
                "mode {mode}"
            );
        }
    }

    /// What: layouts without the tiling proof keep the zero fill and the
    /// unwritten storage reads as zero.
    #[test]
    fn untiled_layouts_keep_the_zero_fill() {
        let source = [2.0f64; 16];
        for (layout, len) in [
            // A hole between the blocks.
            (
                structure(&[(&[2, 1, 1], &[1, 1, 1], 0), (&[2, 1, 1], &[1, 1, 1], 3)]),
                5,
            ),
            // Contiguous, but not in block order.
            (
                structure(&[(&[2, 1, 1], &[1, 1, 1], 2), (&[2, 1, 1], &[1, 1, 1], 0)]),
                4,
            ),
            // A padded block.
            (structure(&[(&[2, 2, 1], &[1, 3, 1], 0)]), 5),
        ] {
            take_owned_block_prefills();
            let data = overwrite_owned_blocks(&layout, |_, writer| {
                writer.copy(|_| Ok(1), &source, 0, false, 1.0)
            })
            .unwrap();
            assert_eq!(take_owned_block_prefills(), len);
            assert_eq!(data.len(), len);
            assert_eq!(data.iter().filter(|&&value| value == 0.0).count(), len - 4);
        }
    }

    /// What: a writer that leaves a block unwritten, or writes one twice,
    /// is an error, and nothing is returned.
    #[test]
    fn every_block_is_written_exactly_once() {
        let source = values();
        let unwritten = overwrite_owned_blocks(&tiled(), |block, writer: &mut Writer<'_>| {
            if block.shape() == [3, 2, 2] {
                return Ok(());
            }
            writer.zero()
        });
        assert!(matches!(
            unwritten,
            Err(OperationError::InvalidArgument { .. })
        ));
        let twice = overwrite_owned_blocks(&tiled(), |_, writer: &mut Writer<'_>| {
            writer.zero()?;
            writer.copy(|_| Ok(1), &source, 0, false, Complex64::new(1.0, 0.0))
        });
        assert!(matches!(twice, Err(OperationError::InvalidArgument { .. })));
        // A failed write leaves the block claimable, but an error returned
        // by the writer still aborts the output.
        let failed = overwrite_owned_blocks(&tiled(), |_, writer: &mut Writer<'_>| {
            writer.copy(|_| Ok(1), &source[..1], 0, false, Complex64::new(1.0, 0.0))
        });
        assert!(failed.is_err());
        let zeros =
            overwrite_owned_blocks(&tiled(), |_, writer: &mut Writer<'_>| writer.zero()).unwrap();
        assert_eq!(bits(&zeros), bits(&[Complex64::new(0.0, 0.0); 23]));
    }

    /// One column-major `2 x 3 x 1` block, tiled `0..6`, gathered from a
    /// column-major `4 x 5 x 1` source (strides `1, 4, 20`) at rows `{0, 3}`
    /// and columns `{0, 2, 3}`: two runs on each of the first two axes.
    fn gather_block() -> BlockStructure {
        structure(&[(&[2, 3, 1], &[1, 2, 6], 0)])
    }

    fn gather_runs() -> [Range<usize>; 4] {
        [0..1, 3..4, 0..1, 2..4]
    }

    fn write_gather(
        allow_uninit: bool,
        member_axis: Option<MemberAxis>,
        source: &[Complex64],
        runs: &[AxisRuns<'_>],
    ) -> Result<Vec<Complex64>, OperationError> {
        overwrite_owned_blocks_in(
            &gather_block(),
            allow_uninit,
            member_axis,
            |_, writer: &mut Writer<'_>| {
                writer.copy_runs(
                    |axis| Ok([1, 4, 20][axis]),
                    source,
                    1,
                    true,
                    Complex64::new(1.0, 0.0),
                    runs,
                )
            },
        )
    }

    /// What: a multi-run gather stores, bit for bit, the conjugated source
    /// elements at the selected positions in order (an index-by-index
    /// oracle), into uninitialized and zeroed storage and for every member of
    /// a stacked output; one run per axis is exactly `copy`.
    #[test]
    fn copy_runs_gathers_the_selected_positions_in_order_bitwise() {
        let source = values();
        let all = gather_runs();
        let runs = [Some(&all[..2]), Some(&all[2..]), None];
        let rows = [0, 3];
        let columns = [0, 2, 3];
        let oracle = |member: usize| -> Vec<Complex64> {
            let mut expected = Vec::new();
            for &column in &columns {
                for &row in &rows {
                    expected.push(source[member * 40 + 1 + row + 4 * column].conj());
                }
            }
            expected
        };
        for allow_uninit in [true, false] {
            let gathered = write_gather(allow_uninit, None, &source, &runs).unwrap();
            assert_eq!(bits(&gathered), bits(&oracle(0)));
            let member_axis = MemberAxis {
                members: 2,
                destination_stride: 6,
                source_stride: 40,
            };
            let stacked = write_gather(allow_uninit, Some(member_axis), &source, &runs).unwrap();
            let expected: Vec<Complex64> = (0..2).flat_map(oracle).collect();
            assert_eq!(bits(&stacked), bits(&expected));
        }
        let contiguous = 1..3;
        let single = [Some(core::slice::from_ref(&contiguous)), None, None];
        let block = structure(&[(&[2, 3, 1], &[1, 2, 6], 0)]);
        let with_copy = overwrite_owned_blocks(&block, |_, writer: &mut Writer<'_>| {
            writer.copy(
                |axis| Ok([1, 4, 20][axis]),
                &source,
                2,
                true,
                Complex64::new(1.0, 0.0),
            )
        })
        .unwrap();
        assert_eq!(
            bits(&write_gather(true, None, &source, &single).unwrap()),
            bits(&with_copy)
        );
    }

    /// What (#2095): runs that do not cover the block exactly, or are
    /// empty, unsorted, overlapping or one short of the rank, and a source
    /// envelope past the end, are typed errors raised before the first
    /// write: nothing is published and no span walk ran.
    #[test]
    fn copy_runs_rejects_an_uncovered_block_before_any_write() {
        let source = values();
        let short = 0..1;
        let unsorted = [3..4, 0..1];
        let overlapping = [0..2, 1..2];
        let empty = [0..0, 0..2];
        let far = [0..1, 95..96];
        let invalid: [&[AxisRuns<'_>]; 5] = [
            // Covers one row of two: the extent - 1 under-coverage.
            &[Some(core::slice::from_ref(&short)), None, None],
            &[Some(&unsorted[..]), None, None],
            &[Some(&overlapping[..]), None, None],
            &[Some(&empty[..]), None, None],
            &[None, None],
        ];
        for runs in invalid {
            crate::take_checked_block_passes();
            for allow_uninit in [true, false] {
                assert!(matches!(
                    write_gather(allow_uninit, None, &source, runs),
                    Err(OperationError::InvalidArgument { .. })
                ));
            }
            assert_eq!(crate::take_checked_block_passes(), Default::default());
        }
        crate::take_checked_block_passes();
        assert!(matches!(
            write_gather(true, None, &source, &[Some(&far[..]), None, None]),
            Err(OperationError::OffsetOverflow { .. })
        ));
        assert_eq!(crate::take_checked_block_passes(), Default::default());
    }

    /// What: the piece walk visits the cartesian product of the runs in
    /// column-major order with packed and sparse starts, and nothing for an
    /// axis with no runs.
    #[test]
    fn run_pieces_walk_the_product_of_the_runs() {
        let axis0 = [0..1, 3..5];
        let axis2 = [1..2, 4..6];
        let mut pieces = Vec::new();
        for_each_run_piece::<()>(
            &[3, 2, 3],
            &[Some(&axis0), None, Some(&axis2)],
            |s, p, q| {
                pieces.push((s.to_vec(), p.to_vec(), q.to_vec()));
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(
            pieces,
            vec![
                (vec![1, 2, 1], vec![0, 0, 0], vec![0, 0, 1]),
                (vec![2, 2, 1], vec![1, 0, 0], vec![3, 0, 1]),
                (vec![1, 2, 2], vec![0, 0, 1], vec![0, 0, 4]),
                (vec![2, 2, 2], vec![1, 0, 1], vec![3, 0, 4]),
            ]
        );
        let none: [Range<usize>; 0] = [];
        let mut count = 0;
        for_each_run_piece::<()>(&[0, 2], &[Some(&none), None], |_, _, _| {
            count += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(count, 0);
    }

    /// What: unwinding from a block writer after earlier blocks were
    /// written drops the storage without exposing it (Miri checks the
    /// partially initialized buffer is never read or published).
    #[test]
    fn panicking_block_writer_publishes_nothing() {
        let source = values();
        let result = std::panic::catch_unwind(|| {
            let mut index = 0;
            let _ = overwrite_owned_blocks(&tiled(), |_, writer: &mut Writer<'_>| {
                index += 1;
                if index == 3 {
                    panic!("injected block writer panic");
                }
                writer.copy(|_| Ok(1), &source, 0, false, Complex64::new(1.0, 0.0))
            });
        });
        assert!(result.is_err());
    }
}
