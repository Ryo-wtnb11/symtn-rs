use super::*;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TreeTransformRecouplingPlan {
    source_len: usize,
    destination_len: usize,
    coefficient_len: usize,
    pub(super) block_indices: Vec<usize>,
    pub(super) jobs: Vec<DenseGemmBatchJob>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TreeTransformPackReplay {
    pub src_layout: usize,
    pub packed_offset: usize,
    pub packed_hi: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TreeTransformScatterReplay {
    pub dst_layout: usize,
    pub packed_offset: usize,
    pub dst_lo: isize,
    pub dst_hi: isize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TreeTransformScatterGroupReplay {
    pub columns: Range<usize>,
    pub slice_disjoint: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct TreeTransformParallelSchedule {
    /// Indices of the non-empty Single blocks, ordered by destination start.
    /// Why indices rather than copied descriptors: the layout entries,
    /// coefficient and destination range are all read from the block and its
    /// compiled role, so a per-Single copy would duplicate retained state;
    /// threaded replay recomputes a destination start only at its split
    /// points.
    pub singles: Vec<u32>,
    pub pack_columns: Vec<TreeTransformPackReplay>,
    pub scatter_columns: Vec<TreeTransformScatterReplay>,
    pub scatter_groups: Vec<TreeTransformScatterGroupReplay>,
    pub single_block_count: usize,
    pub packed_column_count: usize,
    pub scattered_column_count: usize,
    pub singles_slice_disjoint: bool,
}

impl TreeTransformRecouplingPlan {
    #[inline]
    pub fn source_len(&self) -> usize {
        self.source_len
    }

    #[inline]
    pub fn destination_len(&self) -> usize {
        self.destination_len
    }

    #[inline]
    pub fn coefficient_len(&self) -> usize {
        self.coefficient_len
    }

    #[inline]
    pub fn jobs(&self) -> &[DenseGemmBatchJob] {
        &self.jobs
    }

    #[inline]
    pub fn block_indices(&self) -> &[usize] {
        &self.block_indices
    }

    #[inline]
    pub fn entries(&self) -> impl ExactSizeIterator<Item = (usize, &DenseGemmBatchJob)> + '_ {
        self.block_indices.iter().copied().zip(self.jobs.iter())
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }
}

pub(super) fn compile_parallel_schedule(
    blocks: &[TreeTransformBlock],
    layouts: &RawLayoutTable,
    recoupling_plan: &TreeTransformRecouplingPlan,
) -> Result<TreeTransformParallelSchedule, OperationError> {
    let single_block_count = blocks
        .iter()
        .filter(|block| matches!(block, TreeTransformBlock::Single { .. }))
        .count();
    let mut packed_column_count = 0usize;
    let mut scattered_column_count = 0usize;
    // Destination ranges are needed only to order the schedule and prove it
    // slice-disjoint, so they stay in this compile-time list.
    let mut ranges = Vec::with_capacity(single_block_count);
    for (block_index, block) in blocks.iter().enumerate() {
        let TreeTransformBlock::Single { dst_layout, .. } = *block else {
            continue;
        };
        let Some((dst_lo, dst_hi)) = layout_index_range(layouts, dst_layout)? else {
            continue;
        };
        let block_index =
            u32::try_from(block_index).map_err(|_| OperationError::ElementCountOverflow)?;
        ranges.push((dst_lo, dst_hi, block_index));
    }
    // Why the order is the base order: compilation rejects aliased
    // destinations and drops empty ones, so no two keys tie.
    ranges.sort_unstable_by_key(|&(dst_lo, _, _)| dst_lo);
    let singles_slice_disjoint =
        destination_ranges_are_slice_disjoint(ranges.iter().map(|&(lo, hi, _)| (lo, hi)));
    let mut singles = Vec::with_capacity(ranges.len());
    singles.extend(ranges.iter().map(|&(_, _, block_index)| block_index));
    drop(ranges);

    let mut pack_columns = Vec::new();
    let mut scatter_columns = Vec::new();
    let mut scatter_groups = Vec::new();
    for (block_index, job) in recoupling_plan.entries() {
        let TreeTransformBlock::Multi {
            dst_layout_start,
            dst_count,
            src_layout_start,
            src_count,
            element_count,
            ..
        } = blocks[block_index]
        else {
            return Err(OperationError::InvalidArgument {
                message: "tree transform recoupling plan references a single block",
            });
        };
        packed_column_count = packed_column_count
            .checked_add(src_count)
            .ok_or(OperationError::ElementCountOverflow)?;
        scattered_column_count = scattered_column_count
            .checked_add(dst_count)
            .ok_or(OperationError::ElementCountOverflow)?;
        let group_start = scatter_columns.len();
        if element_count == 0 {
            scatter_groups.push(TreeTransformScatterGroupReplay {
                columns: group_start..group_start,
                slice_disjoint: true,
            });
            continue;
        }
        for src_index in 0..src_count {
            let packed_offset = job
                .lhs_offset
                .checked_add(
                    src_index
                        .checked_mul(element_count)
                        .ok_or(OperationError::ElementCountOverflow)?,
                )
                .ok_or(OperationError::ElementCountOverflow)?;
            let packed_hi = packed_offset
                .checked_add(element_count - 1)
                .ok_or(OperationError::ElementCountOverflow)?;
            pack_columns.push(TreeTransformPackReplay {
                src_layout: src_layout_start + src_index,
                packed_offset,
                packed_hi,
            });
        }
        for dst_index in 0..dst_count {
            let dst_layout = dst_layout_start + dst_index;
            let Some((dst_lo, dst_hi)) = layout_index_range(layouts, dst_layout)? else {
                continue;
            };
            let packed_offset = job
                .dst_offset
                .checked_add(
                    dst_index
                        .checked_mul(element_count)
                        .ok_or(OperationError::ElementCountOverflow)?,
                )
                .ok_or(OperationError::ElementCountOverflow)?;
            scatter_columns.push(TreeTransformScatterReplay {
                dst_layout,
                packed_offset,
                dst_lo,
                dst_hi,
            });
        }
        scatter_columns[group_start..].sort_unstable_by_key(|item| item.dst_lo);
        scatter_groups.push(TreeTransformScatterGroupReplay {
            columns: group_start..scatter_columns.len(),
            slice_disjoint: destination_ranges_are_slice_disjoint(
                scatter_columns[group_start..]
                    .iter()
                    .map(|item| (item.dst_lo, item.dst_hi)),
            ),
        });
    }
    pack_columns.sort_unstable_by_key(|item| item.packed_offset);

    let pack_disjoint = pack_columns
        .windows(2)
        .all(|pair| pair[0].packed_hi < pair[1].packed_offset);
    if !pack_disjoint
        || pack_columns
            .last()
            .is_some_and(|item| item.packed_hi >= recoupling_plan.source_len())
    {
        return Err(OperationError::InvalidArgument {
            message: "tree transform packed source schedule is invalid",
        });
    }

    Ok(TreeTransformParallelSchedule {
        singles_slice_disjoint,
        single_block_count,
        packed_column_count,
        scattered_column_count,
        singles,
        pack_columns,
        scatter_columns,
        scatter_groups,
    })
}

fn destination_ranges_are_slice_disjoint(ranges: impl IntoIterator<Item = (isize, isize)>) -> bool {
    let mut previous_hi = None;
    for (lo, hi) in ranges {
        if lo < 0 || hi < lo || previous_hi.is_some_and(|previous| previous >= lo) {
            return false;
        }
        previous_hi = Some(hi);
    }
    true
}

pub(super) fn layout_index_range(
    layouts: &RawLayoutTable,
    layout_index: usize,
) -> Result<Option<(isize, isize)>, OperationError> {
    let layout = layouts.entry(layout_index);
    if layout.element_count == 0 {
        return Ok(None);
    }
    let mut lo = layout.offset;
    let mut hi = layout.offset;
    for (&extent, &stride) in layouts
        .shape(layout_index)
        .iter()
        .zip(layouts.strides(layout_index))
    {
        let extent = isize::try_from(extent.saturating_sub(1))
            .map_err(|_| OperationError::ElementCountOverflow)?;
        let span = extent
            .checked_mul(stride)
            .ok_or(OperationError::ElementCountOverflow)?;
        if span < 0 {
            lo = lo
                .checked_add(span)
                .ok_or(OperationError::ElementCountOverflow)?;
        } else {
            hi = hi
                .checked_add(span)
                .ok_or(OperationError::ElementCountOverflow)?;
        }
    }
    Ok(Some((lo, hi)))
}

pub(super) fn compile_recoupling_plan(
    blocks: &[TreeTransformBlock],
) -> Result<TreeTransformRecouplingPlan, OperationError> {
    #[derive(Clone, Copy)]
    struct MultiEntry {
        block_index: usize,
        element_count: usize,
        src_count: usize,
        dst_count: usize,
    }

    let mut entries = Vec::new();
    for (block_index, block) in blocks.iter().enumerate() {
        if let TreeTransformBlock::Multi {
            dst_count,
            src_count,
            element_count,
            ..
        } = *block
        {
            entries.push(MultiEntry {
                block_index,
                element_count,
                src_count,
                dst_count,
            });
        }
    }
    entries.sort_by_key(|entry| {
        (
            entry.element_count,
            entry.src_count,
            entry.dst_count,
            entry.block_index,
        )
    });

    let mut source_len = 0usize;
    let mut destination_len = 0usize;
    let mut coefficient_len = 0usize;
    let mut block_indices = Vec::with_capacity(entries.len());
    let mut jobs = Vec::with_capacity(entries.len());
    for entry in entries {
        let block_source_len = entry
            .element_count
            .checked_mul(entry.src_count)
            .ok_or(OperationError::ElementCountOverflow)?;
        let block_destination_len = entry
            .element_count
            .checked_mul(entry.dst_count)
            .ok_or(OperationError::ElementCountOverflow)?;
        let block_coefficient_len = entry
            .src_count
            .checked_mul(entry.dst_count)
            .ok_or(OperationError::ElementCountOverflow)?;
        block_indices.push(entry.block_index);
        jobs.push(DenseGemmBatchJob {
            dst_offset: destination_len,
            lhs_offset: source_len,
            rhs_offset: coefficient_len,
            rows: entry.element_count,
            contracted: entry.src_count,
            cols: entry.dst_count,
        });
        source_len = source_len
            .checked_add(block_source_len)
            .ok_or(OperationError::ElementCountOverflow)?;
        destination_len = destination_len
            .checked_add(block_destination_len)
            .ok_or(OperationError::ElementCountOverflow)?;
        coefficient_len = coefficient_len
            .checked_add(block_coefficient_len)
            .ok_or(OperationError::ElementCountOverflow)?;
    }
    Ok(TreeTransformRecouplingPlan {
        source_len,
        destination_len,
        coefficient_len,
        block_indices,
        jobs,
    })
}

pub(super) fn tree_transform_block_weight(
    block: &TreeTransformBlock,
    layouts: &[TreeTransformLayout],
) -> usize {
    match *block {
        TreeTransformBlock::Single { dst_layout, .. } => layouts[dst_layout].element_count,
        TreeTransformBlock::Multi {
            dst_count,
            src_count,
            element_count,
            ..
        } => dst_count
            .saturating_mul(src_count)
            .saturating_mul(element_count),
    }
}

/// `Some(required_len)` when writing every active destination and every
/// inactive zero writer once initializes `0..required_len` exactly; `None`
/// for padding and other holes. O(B·rank) in destination block metadata.
pub(super) fn compile_physical_overwrite_coverage(
    blocks: &[TreeTransformBlock],
    inactive_dst_layouts: &[usize],
    layouts: &RawLayoutTable,
    recoupling_plan: &TreeTransformRecouplingPlan,
    destination_block_count: usize,
    destination_tiling_proven: bool,
    required_len: usize,
) -> Result<Option<usize>, OperationError> {
    let multi_count = blocks
        .iter()
        .filter(|block| matches!(block, TreeTransformBlock::Multi { .. }))
        .count();
    let mut scheduled = vec![false; blocks.len()];
    if recoupling_plan.block_indices().len() != multi_count
        || recoupling_plan.block_indices().iter().any(|&index| {
            let Some(slot) = scheduled.get_mut(index) else {
                return true;
            };
            if *slot || !matches!(blocks[index], TreeTransformBlock::Multi { .. }) {
                return true;
            }
            *slot = true;
            false
        })
    {
        return Ok(None);
    }
    let active_layout_count = blocks.iter().try_fold(0usize, |count, block| {
        let destination_count = match *block {
            TreeTransformBlock::Single { .. } => 1,
            TreeTransformBlock::Multi { dst_count, .. } => dst_count,
        };
        count
            .checked_add(destination_count)
            .ok_or(OperationError::ElementCountOverflow)
    })?;
    let covered_layout_count = active_layout_count
        .checked_add(inactive_dst_layouts.len())
        .ok_or(OperationError::ElementCountOverflow)?;

    if covered_layout_count != destination_block_count {
        return Ok(None);
    }

    if destination_tiling_proven {
        return Ok(Some(required_len));
    }

    // Why not require each block to be one contiguous interval: canonical
    // blocks of a coupled sector with several codomain trees interleave
    // their columns. compile_shared_structures has already rejected aliased
    // destination layouts and duplicate block owners, so the written offsets
    // are distinct; with every one inside 0..required_len, a total equal to
    // required_len proves exact cover from block metadata alone.
    let mut covered_len = 0usize;
    let mut record_layout = |layout_index: usize| -> Result<bool, OperationError> {
        let Some((lo, hi)) = layout_index_range(layouts, layout_index)? else {
            return Ok(true);
        };
        let in_bounds = lo >= 0 && usize::try_from(hi).is_ok_and(|hi| hi < required_len);
        if in_bounds {
            covered_len = covered_len
                .checked_add(layouts.entry(layout_index).element_count)
                .ok_or(OperationError::ElementCountOverflow)?;
        }
        Ok(in_bounds)
    };
    let active_layouts = blocks.iter().flat_map(|block| match *block {
        TreeTransformBlock::Single { dst_layout, .. } => dst_layout..dst_layout + 1,
        TreeTransformBlock::Multi {
            dst_layout_start,
            dst_count,
            ..
        } => dst_layout_start..dst_layout_start + dst_count,
    });
    for layout in active_layouts.chain(inactive_dst_layouts.iter().copied()) {
        if !record_layout(layout)? {
            return Ok(None);
        }
    }

    // PhysicalOverwriteProof additionally requires canonical coupled-sector
    // regions to partition the same 0..required_len range before owned replay.
    Ok((covered_len == required_len).then_some(required_len))
}
