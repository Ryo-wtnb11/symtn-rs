use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicUsize, Ordering};

use crate::kernel_adapter::{write_fused_uninit, BakedFusedLayout};
use crate::storage_scratch::SamePlacementScratchAllocator;
use crate::OperationError;
use symtn_core::{HostReadableStorage, HostWritableStorage, Placement, TensorStorage};

/// Host-owned scratch buffer used by current raw host-slice replay paths.
///
/// This keeps direct `Vec<T>` ownership out of categorical replay workspaces.
#[derive(Clone, Debug)]
pub struct HostScratchBuffer<T> {
    data: Vec<T>,
}

impl<T> Default for HostScratchBuffer<T> {
    fn default() -> Self {
        Self { data: Vec::new() }
    }
}

impl<T> HostScratchBuffer<T> {
    #[inline]
    pub fn filled(len: usize, value: T) -> Self
    where
        T: Clone,
    {
        let reference = Vec::<T>::new();
        let allocator = SamePlacementScratchAllocator;
        Self {
            data: allocator.filled_like(&reference, len, value),
        }
    }

    #[inline]
    pub fn resize_filled(&mut self, len: usize, value: T)
    where
        T: Clone,
    {
        self.data.resize(len, value);
    }

    #[inline]
    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        self.data.fill(value);
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Allocated element capacity retained by this buffer.
    #[doc(hidden)]
    #[inline]
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl<T: Copy> HostScratchBuffer<T> {
    /// Empties the buffer and makes room for an overwrite of `len` elements.
    pub(crate) fn reserve_overwrite(&mut self, len: usize) {
        self.data.clear();
        if len > self.data.capacity() {
            // Why not `reserve_exact`: growing would copy the old capacity,
            // which nothing reads.
            self.data = Vec::with_capacity(len);
        }
    }

    /// Replaces the contents with `len` elements that `write` produces as
    /// consecutive dense columns, without initialising them first.
    ///
    /// Why not zero-fill and then pack: every element of a pack scratch is
    /// overwritten before anything reads it, and an eager call's workspace is
    /// per call, so the fill was one more serial pass over the whole chunk.
    ///
    /// Initialisation invariant: the buffer's length becomes `len` only after
    /// every slot of `0..len` was written. This function upholds it alone:
    /// slots are reachable only through [`PackColumnWriter`], whose writers
    /// own disjoint ranges and write each range strictly front to back in
    /// whole dense columns, so the elements they report are distinct written
    /// slots; `len` reported elements are therefore all of them. Safe callers
    /// cannot break it: a writer that skips, stops early, errors, panics, or
    /// is leaked reports fewer than `len`, and the buffer stays empty.
    pub(crate) fn overwrite_with_columns(
        &mut self,
        len: usize,
        write: impl FnOnce(PackColumnWriter<'_, T>) -> Result<(), OperationError>,
    ) -> Result<(), OperationError> {
        self.reserve_overwrite(len);
        let written = AtomicUsize::new(0);
        write(PackColumnWriter {
            slots: &mut self.data.spare_capacity_mut()[..len],
            cursor: 0,
            written: &written,
        })?;
        if written.into_inner() != len {
            return Err(OperationError::InvalidArgument {
                message: "pack scratch left unwritten",
            });
        }
        // SAFETY: capacity is at least `len`, and by the invariant above all
        // `len` written reports are distinct slots of `0..len`, each holding
        // a value `MaybeUninit::write` stored. `T: Copy` has no drop to skip.
        #[allow(unsafe_code)]
        unsafe {
            self.data.set_len(len)
        };
        Ok(())
    }
}

/// Exclusive writer of one range of a pack scratch being overwritten by
/// [`HostScratchBuffer::overwrite_with_columns`].
pub(crate) struct PackColumnWriter<'a, T> {
    slots: &'a mut [MaybeUninit<T>],
    /// Slots before this are written; none at or after it are.
    cursor: usize,
    written: &'a AtomicUsize,
}

impl<T: Copy> PackColumnWriter<'_, T> {
    /// Splits off the slots from `mid` on as a second writer, for a parallel
    /// pack. `mid` must not precede a slot already written.
    pub(crate) fn split_at(self, mid: usize) -> Result<(Self, Self), OperationError> {
        if mid < self.cursor || mid > self.slots.len() {
            return Err(OperationError::InvalidArgument {
                message: "pack scratch split outside its unwritten range",
            });
        }
        let (left, right) = self.slots.split_at_mut(mid);
        Ok((
            PackColumnWriter {
                slots: left,
                cursor: self.cursor,
                written: self.written,
            },
            PackColumnWriter {
                slots: right,
                cursor: 0,
                written: self.written,
            },
        ))
    }

    /// Writes `map(src)` through `layout` as the column starting at `offset`
    /// (relative to this writer), which must be the next unwritten slot. The
    /// layout's destination side must address one dense column-major run.
    pub(crate) fn write_column(
        &mut self,
        offset: usize,
        layout: BakedFusedLayout<'_>,
        src: &[T],
        src_offset: isize,
        index: &mut [usize],
        map: impl Fn(T) -> T,
    ) -> Result<(), OperationError> {
        let end = dense_run_len(layout.dims(), layout.dst_strides())
            .and_then(|len| self.cursor.checked_add(len))
            .filter(|&end| offset == self.cursor && end <= self.slots.len())
            .ok_or(OperationError::InvalidArgument {
                message: "pack column is not the next dense run of its scratch",
            })?;
        write_fused_uninit(
            layout,
            &mut self.slots[self.cursor..end],
            src,
            0,
            src_offset,
            index,
            map,
        )?;
        self.written.fetch_add(end - self.cursor, Ordering::Relaxed);
        self.cursor = end;
        Ok(())
    }
}

/// Element count when `dims`/`strides` address exactly the column-major run
/// `0..count` (a bijection, extent-1 axes ignored), otherwise `None`.
fn dense_run_len(dims: &[usize], strides: &[isize]) -> Option<usize> {
    // Rank 0 is refused: the fused traversal visits nothing for it.
    if dims.is_empty() || dims.len() != strides.len() {
        return None;
    }
    if dims.contains(&0) {
        return Some(0);
    }
    // Each step needs the axis whose stride is the run length so far; that
    // length strictly grows, so no axis can be matched twice.
    let mut len = 1usize;
    for _ in dims.iter().filter(|&&extent| extent > 1) {
        let (extent, _) = dims
            .iter()
            .zip(strides)
            .find(|&(&extent, &stride)| extent > 1 && usize::try_from(stride) == Ok(len))?;
        len = len.checked_mul(*extent)?;
    }
    Some(len)
}

impl<T> TensorStorage<T> for HostScratchBuffer<T> {
    #[inline]
    fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    fn placement(&self) -> Placement {
        Placement::Host
    }
}

impl<T> HostReadableStorage<T> for HostScratchBuffer<T> {
    #[inline]
    fn as_slice(&self) -> &[T] {
        &self.data
    }
}

impl<T> HostWritableStorage<T> for HostScratchBuffer<T> {
    #[inline]
    fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn column<'a>(dims: &'a [usize], dst: &'a [isize], src: &'a [isize]) -> BakedFusedLayout<'a> {
        BakedFusedLayout::try_from_normalized_slices(dims, dst, src).unwrap()
    }

    #[test]
    fn overwrite_exposes_values_only_after_every_slot_is_written() {
        // What: split writers that write every slot hand out exactly the
        // packed values (Miri checks no slot is read uninitialised); a pass
        // that leaves a slot unwritten, writes out of order, or uses a
        // non-dense column is refused and the buffer stays empty.
        let src = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut index = [0usize; 2];
        let transposed = column(&[2, 3], &[3, 1], &[1, 2]);
        let mut buffer = HostScratchBuffer::default();
        buffer
            .overwrite_with_columns(8, |writer| {
                let (mut left, mut right) = writer.split_at(6)?;
                left.write_column(0, transposed, &src, 0, &mut index, |v| v)?;
                right.write_column(0, column(&[2], &[1], &[1]), &src, 4, &mut index, |v| -v)
            })
            .unwrap();
        assert_eq!(
            buffer.as_slice(),
            [1.0, 3.0, 5.0, 2.0, 4.0, 6.0, -5.0, -6.0]
        );

        let partial = buffer.overwrite_with_columns(8, |mut writer| {
            writer.write_column(0, transposed, &src, 0, &mut index, |v| v)
        });
        assert!(partial.is_err());
        assert!(buffer.is_empty());

        let skipped = buffer.overwrite_with_columns(8, |mut writer| {
            writer.write_column(2, transposed, &src, 0, &mut index, |v| v)
        });
        assert!(skipped.is_err());
        assert!(buffer.is_empty());

        let gapped = buffer.overwrite_with_columns(12, |mut writer| {
            writer.write_column(
                0,
                column(&[2, 3], &[1, 4], &[1, 2]),
                &src,
                0,
                &mut index,
                |v| v,
            )
        });
        assert!(gapped.is_err());
        assert!(buffer.is_empty());
    }
}
