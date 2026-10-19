//! Region descriptor and layout validation for the CUDA region primitive.
//!
//! Pure metadata: no device, no backend and no tenferro type appears here, so
//! this module compiles — and its tests run — in a CPU-only build. It is kept
//! out of `cuda_adapter` precisely so that the layout rules the device path
//! depends on are executed by CI, which only *checks* the `cuda` feature.

use crate::DenseError;

/// A strided, offset, rank-N region of a flat device buffer.
///
/// Strides are element counts and unsigned. Tenferro's CUDA dot-general
/// rejects a negative view stride outright ("requires nonnegative view
/// strides", `benchmarks/history/cuda-strided-region-probe-2026-09-20.md`
/// item 4), so the sign is a type boundary here rather than a runtime check:
/// a reversed axis must be canonicalized into the destination's stride
/// pattern by whoever bakes the layout.
///
/// Source and destination carry the *same* `dims` in the same axis order; an
/// axis permutation is expressed entirely by the destination's strides, which
/// is the `(dims, dst_strides, src_strides)` triple the host transform layout
/// already bakes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CudaRegion {
    dims: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
}

impl CudaRegion {
    /// Region of `dims` extents with `strides` (element units) from `offset`.
    ///
    /// Bounds, injectivity and overflow are checked against the buffer at the
    /// call site, because they depend on the storage this region is used with.
    pub fn new(dims: Vec<usize>, strides: Vec<usize>, offset: usize) -> Result<Self, DenseError> {
        if dims.len() != strides.len() {
            return Err(DenseError::RankMismatch {
                shape: dims.len(),
                strides: strides.len(),
            });
        }
        Ok(Self {
            dims,
            strides,
            offset,
        })
    }

    /// Rewrites this region in place, reusing its buffers, so a replay that
    /// describes the same regions on every call allocates only when a region
    /// outgrows the rank it had.
    pub fn assign(
        &mut self,
        dims: &[usize],
        strides: impl IntoIterator<Item = usize>,
        offset: usize,
    ) -> Result<(), DenseError> {
        self.dims.clear();
        self.dims.extend_from_slice(dims);
        self.strides.clear();
        self.strides.extend(strides);
        self.offset = offset;
        if self.dims.len() != self.strides.len() {
            return Err(DenseError::RankMismatch {
                shape: self.dims.len(),
                strides: self.strides.len(),
            });
        }
        Ok(())
    }

    /// Compact column-major region of `dims` extents starting at `offset`.
    pub fn packed(dims: &[usize], offset: usize) -> Result<Self, DenseError> {
        let mut strides = Vec::with_capacity(dims.len());
        let mut running = 1usize;
        for &dim in dims {
            strides.push(running);
            running = running
                .checked_mul(dim)
                .ok_or(DenseError::ElementCountOverflow)?;
        }
        Ok(Self {
            dims: dims.to_vec(),
            strides,
            offset,
        })
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Heap storage retained by this descriptor, excluding the inline value.
    #[doc(hidden)]
    pub fn retained_heap_bytes(&self) -> usize {
        self.dims
            .capacity()
            .saturating_add(self.strides.capacity())
            .saturating_mul(std::mem::size_of::<usize>())
    }

    /// Whether the region addresses no element at all. Checked before the
    /// element count, so a zero extent is a no-op even when the remaining
    /// extents would overflow — the host rule (`shape.contains(&0)`) exactly.
    pub fn is_empty(&self) -> bool {
        self.dims.contains(&0)
    }

    /// Number of elements the region addresses.
    pub fn element_count(&self) -> Result<usize, DenseError> {
        self.dims
            .iter()
            .try_fold(1usize, |count, dim| count.checked_mul(*dim))
            .ok_or(DenseError::ElementCountOverflow)
    }

    /// The `(dims ++ [1], strides ++ [1])` operand metadata `dot_general`
    /// wants: the trailing unit mode is the one contracted against the 1x1
    /// coefficient.
    pub(crate) fn contraction_view_metadata(&self) -> Result<(Vec<usize>, Vec<isize>), DenseError> {
        let mut dims = Vec::with_capacity(self.dims.len() + 1);
        dims.extend_from_slice(&self.dims);
        dims.push(1);
        let mut strides = Vec::with_capacity(self.strides.len() + 1);
        for &stride in &self.strides {
            strides.push(
                isize::try_from(stride)
                    .map_err(|_| DenseError::StrideOverflow { value: stride })?,
            );
        }
        strides.push(1);
        Ok((dims, strides))
    }

    pub(crate) fn offset_isize(&self) -> Result<isize, DenseError> {
        isize::try_from(self.offset).map_err(|_| DenseError::OffsetOverflow { value: self.offset })
    }

    /// Highest flat position the region reads or writes. Only meaningful for a
    /// non-empty region; an empty one addresses nothing.
    pub(crate) fn last_position(&self) -> Result<usize, DenseError> {
        self.dims
            .iter()
            .zip(&self.strides)
            .try_fold(self.offset, |position, (dim, stride)| {
                dim.checked_sub(1)
                    .and_then(|span| span.checked_mul(*stride))
                    .and_then(|span| position.checked_add(span))
            })
            .ok_or(DenseError::OffsetOverflow { value: self.offset })
    }

    /// Checks that this region can be written as a destination, so a caller
    /// that prepares regions ahead of submitting them can reject an
    /// inexpressible layout before it does any device work.
    ///
    /// This is exactly the check the region calls make on their destination,
    /// and it depends on nothing but the region itself, so passing here
    /// guarantees the submission will not fail on the layout. An empty region
    /// is accepted, because it addresses nothing and is never submitted.
    pub fn validate_as_destination(&self, op: &'static str) -> Result<(), DenseError> {
        if self.is_empty() {
            return Ok(());
        }
        validate_destination_layout(op, self)
    }

    /// Checks a prepared region against its flat storage before any device submission.
    #[doc(hidden)]
    pub fn validate_within(&self, len: usize) -> Result<(), DenseError> {
        if self.is_empty() {
            return Ok(());
        }
        self.offset_isize()?;
        validate_region(self, len)
    }

    /// Whether distinct index tuples map to distinct flat positions.
    ///
    /// This is the cumulative-span rule the host proves block layouts with
    /// (`symtn-core` `block_structure.rs::block_layout_is_proven_injective`):
    /// in ascending stride order over the axes of extent > 1, each stride must
    /// exceed the total span of the faster axes. Using the same rule keeps
    /// device admission equal to the host's *proven* class, and it is strictly
    /// stronger than the rule Tenferro applies at view construction, so a
    /// layout accepted here is never rejected after the operands have been
    /// prepared.
    ///
    /// Layouts the host admits only through its exact overlap fallback —
    /// interleaved-but-injective expert layouts — are `Unsupported` on device.
    pub(crate) fn is_injective(&self) -> bool {
        let mut axes: Vec<(usize, usize)> = self
            .dims
            .iter()
            .zip(&self.strides)
            .filter(|(dim, _)| **dim > 1)
            .map(|(dim, stride)| (*dim, *stride))
            .collect();
        axes.sort_unstable_by_key(|(_, stride)| *stride);
        let mut lower_span = 0usize;
        for (dim, stride) in axes {
            if stride == 0 || stride <= lower_span {
                return false;
            }
            match (dim - 1)
                .checked_mul(stride)
                .and_then(|span| lower_span.checked_add(span))
            {
                Some(span) => lower_span = span,
                None => return false,
            }
        }
        true
    }
}

/// Bounds and isize-representability of one region against its buffer.
pub(crate) fn validate_region(region: &CudaRegion, len: usize) -> Result<(), DenseError> {
    if region.last_position()? >= len {
        return Err(DenseError::OutOfBounds);
    }
    region.contraction_view_metadata().map(|_| ())
}

pub(crate) fn validate_destination_layout(
    op: &'static str,
    dst_region: &CudaRegion,
) -> Result<(), DenseError> {
    if dst_region.is_injective() {
        return Ok(());
    }
    Err(DenseError::Unsupported {
        op,
        message: format!(
            "destination region dims {:?} strides {:?} is not a proven-injective layout; \
             the backend requires an injective destination view",
            dst_region.dims, dst_region.strides
        ),
    })
}

/// The row entries of `cuda_gather_columns_batched_into`: each source
/// slice inside `n` rows, each destination slice inside `dst_ld` rows, and
/// no two destination slices sharing a row. Pure, so it runs before any
/// submission.
pub(crate) fn validate_gather_rows(
    n: usize,
    dst_ld: usize,
    rows: &[(usize, usize, usize)],
) -> Result<(), String> {
    let mut spans = Vec::with_capacity(rows.len());
    for (index, &(src_row, dst_row, count)) in rows.iter().enumerate() {
        let inside =
            |start: usize, bound: usize| start.checked_add(count).is_some_and(|end| end <= bound);
        if !inside(src_row, n) || !inside(dst_row, dst_ld) {
            return Err(format!(
                "row entry {index} ({src_row}, {dst_row}, {count}) exceeds {n} source or {dst_ld} destination rows"
            ));
        }
        if count != 0 {
            spans.push((dst_row, dst_row + count, index));
        }
    }
    spans.sort_unstable();
    for pair in spans.windows(2) {
        if pair[1].0 < pair[0].1 {
            return Err(format!(
                "row entries {} and {} write the same destination rows",
                pair[0].2, pair[1].2
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gather_rows_are_checked_for_bounds_and_overlap_before_any_work() {
        // What: every entry is inside the source and destination rows, and
        // no two destination slices share a row; a later bad entry rejects
        // the whole list.
        assert!(validate_gather_rows(4, 6, &[(0, 0, 2), (2, 3, 2), (0, 5, 1)]).is_ok());
        assert!(
            matches!(validate_gather_rows(4, 6, &[(0, 0, 2), (0, 5, 2)]), Err(message) if message.contains("row entry 1") && message.contains("exceeds"))
        );
        assert!(
            matches!(validate_gather_rows(4, 6, &[(0, 0, 2), (3, 2, 2)]), Err(message) if message.contains("row entry 1") && message.contains("exceeds"))
        );
        assert!(validate_gather_rows(4, 6, &[(0, 0, 3), (0, 2, 2)])
            .unwrap_err()
            .contains("write the same destination rows"));
        assert!(validate_gather_rows(4, 6, &[(0, 1, 0), (0, 1, 2)]).is_ok());
        assert!(
            matches!(validate_gather_rows(4, 6, &[(usize::MAX, 0, 2)]), Err(message) if message.contains("row entry 0") && message.contains("exceeds"))
        );
    }

    #[test]
    fn region_rank_disagreement_is_a_typed_rank_mismatch() {
        assert!(matches!(
            CudaRegion::new(vec![2, 3], vec![1], 0),
            Err(DenseError::RankMismatch {
                shape: 2,
                strides: 1
            })
        ));
    }

    #[test]
    fn an_assigned_region_equals_a_new_one_and_keeps_its_rank_check() {
        let mut region = CudaRegion::new(vec![2, 3, 4], vec![1, 2, 6], 5).unwrap();
        region.assign(&[3, 2], [2, 1], 9).unwrap();
        assert_eq!(region, CudaRegion::new(vec![3, 2], vec![2, 1], 9).unwrap());
        assert!(matches!(
            region.assign(&[2, 3], [1], 0),
            Err(DenseError::RankMismatch {
                shape: 2,
                strides: 1
            })
        ));
    }

    #[test]
    fn packed_region_strides_are_column_major() {
        let region = CudaRegion::packed(&[2, 3, 4], 7).unwrap();
        assert_eq!(region.strides(), &[1, 2, 6]);
        assert_eq!(region.dims(), &[2, 3, 4]);
        assert_eq!(region.offset(), 7);
        assert_eq!(region.element_count().unwrap(), 24);
        assert!(!region.is_empty());
    }

    #[test]
    fn the_injectivity_rule_is_the_one_the_host_proves_layouts_with() {
        // Every axis order of a dense block, which is what a baked tree
        // layout produces.
        for strides in [vec![1, 2, 6], vec![12, 1, 3], vec![6, 2, 1]] {
            let region = CudaRegion::new(vec![2, 3, 2], strides.clone(), 0).unwrap();
            assert!(region.is_injective(), "{strides:?} must be injective");
        }
        // A gapped sub-block of a wider parent: strides exceed the span of
        // the faster axes without being a multiple of them.
        let gapped = CudaRegion::new(vec![3, 2], vec![1, 8], 0).unwrap();
        assert!(gapped.is_injective());
        // Interleaved but injective: positions 0, 2, 3, 5. The host proves
        // this one, and so must the device — the strict "next stride is a
        // multiple of the previous span" rule would reject it.
        let interleaved = CudaRegion::new(vec![2, 2], vec![2, 3], 0).unwrap();
        assert!(interleaved.is_injective());
        // Genuinely non-injective: a repeated (stride 0) axis of extent > 1.
        let broadcast = CudaRegion::new(vec![4, 4], vec![1, 0], 0).unwrap();
        assert!(!broadcast.is_injective());
        // Genuinely non-injective: positions 0,1,2, 2,3,4, 4,5,6.
        let overlapping = CudaRegion::new(vec![3, 3], vec![1, 2], 0).unwrap();
        assert!(!overlapping.is_injective());
        // A stride 0 axis of extent 1 addresses one element, so it is fine.
        let unit = CudaRegion::new(vec![4, 1], vec![1, 0], 0).unwrap();
        assert!(unit.is_injective());
    }

    #[test]
    fn region_bounds_are_checked_against_the_buffer_length() {
        let region = CudaRegion::new(vec![2, 3], vec![1, 4], 2).unwrap();
        // Last position is 2 + 1*1 + 2*4 = 11.
        assert_eq!(region.last_position().unwrap(), 11);
        assert!(validate_region(&region, 12).is_ok());
        assert!(matches!(
            validate_region(&region, 11),
            Err(DenseError::OutOfBounds)
        ));
    }

    #[test]
    fn region_metadata_overflow_is_typed_rather_than_wrapping() {
        let huge = CudaRegion::new(vec![usize::MAX, 2], vec![1, 1], 0).unwrap();
        assert!(matches!(
            huge.element_count(),
            Err(DenseError::ElementCountOverflow)
        ));
        let far = CudaRegion::new(vec![3], vec![usize::MAX], 1).unwrap();
        assert!(matches!(
            far.last_position(),
            Err(DenseError::OffsetOverflow { value: 1 })
        ));
        let wide = CudaRegion::new(vec![2], vec![usize::MAX], 0).unwrap();
        assert!(matches!(
            wide.contraction_view_metadata(),
            Err(DenseError::StrideOverflow { value: usize::MAX })
        ));
        // A zero extent wins over an overflowing product, as it does on the
        // host: the region addresses nothing at all.
        let empty = CudaRegion::new(vec![usize::MAX, 2, 0], vec![1, 1, 1], 0).unwrap();
        assert!(empty.is_empty());
    }

    #[test]
    fn contraction_view_metadata_appends_the_contracted_unit_mode() {
        let region = CudaRegion::new(vec![2, 3], vec![3, 1], 5).unwrap();
        let (dims, strides) = region.contraction_view_metadata().unwrap();
        assert_eq!(dims, vec![2, 3, 1]);
        assert_eq!(strides, vec![3, 1, 1]);
        assert_eq!(region.offset_isize().unwrap(), 5);
    }

    #[test]
    fn retained_heap_bytes_uses_vector_capacity() {
        let mut dims = Vec::with_capacity(7);
        dims.extend([2, 3]);
        let mut strides = Vec::with_capacity(9);
        strides.extend([1, 2]);
        let expected = dims
            .capacity()
            .saturating_add(strides.capacity())
            .saturating_mul(std::mem::size_of::<usize>());
        let region = CudaRegion::new(dims, strides, 0).unwrap();
        assert_eq!(region.retained_heap_bytes(), expected);
    }

    #[test]
    fn the_destination_check_is_available_before_any_submission() {
        // What: a caller preparing regions ahead of time gets the same verdict
        // the submission would give, without a device.
        let interleaved = CudaRegion::new(vec![3, 2], vec![2, 3], 0).unwrap();
        assert!(interleaved
            .validate_as_destination("cuda_tree_transform")
            .is_err());
        let ok = CudaRegion::new(vec![3, 2], vec![1, 3], 0).unwrap();
        assert!(ok.validate_as_destination("cuda_tree_transform").is_ok());
        let empty = CudaRegion::new(vec![0, 2], vec![1, 0], 0).unwrap();
        assert!(empty.validate_as_destination("cuda_tree_transform").is_ok());
    }

    #[test]
    fn a_layout_that_is_not_proven_injective_is_reported_as_unsupported() {
        let region = CudaRegion::new(vec![4, 4], vec![1, 0], 0).unwrap();
        let err = validate_destination_layout("cuda_region_axpby", &region).unwrap_err();
        assert!(matches!(
            err,
            DenseError::Unsupported {
                op: "cuda_region_axpby",
                ..
            }
        ));
        assert!(err.to_string().contains("injective"), "{err}");
    }
}
