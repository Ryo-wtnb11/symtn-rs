use super::*;

/// Compiles the pack and scatter replay columns and main's replay dispatch
/// metadata from the raw layout table: the Direct lists in destination
/// address order with the Singles' slice-disjointness, each plan group's
/// scatter span, and the runnable clamp. O(I log I + B) time and O(I)
/// retained metadata for I intervals and B blocks: one sort of the Singles,
/// one of the inactive layouts and one per group's columns.
pub(super) fn compile_parallel_schedule(
    blocks: &[TreeTransformBlock],
    inactive_dst_layouts: &[usize],
    layouts: &RawLayoutTable,
    recoupling_plan: &TreeTransformRecouplingPlan,
) -> Result<TreeTransformParallelSchedule, OperationError> {
    let single_block_count = blocks
        .iter()
        .filter(|block| matches!(block, TreeTransformBlock::Single { .. }))
        .count();
    let mut packed_column_count = 0usize;
    let mut scattered_column_count = 0usize;
    let mut ranges = Vec::with_capacity(single_block_count);
    for (block_index, block) in blocks.iter().enumerate() {
        let TreeTransformBlock::Single { dst_layout, .. } = *block else {
            continue;
        };
        let Some((dst_lo, dst_hi)) = layout_index_range(layouts, dst_layout)? else {
            continue;
        };
        ranges.push((dst_lo, dst_hi, checked_u32(block_index)?));
    }
    // Why the order is the base order: compilation rejects aliased
    // destinations and drops empty ones, so no two keys tie.
    ranges.sort_unstable_by_key(|&(dst_lo, _, _)| dst_lo);

    // Every count is known from the plan, so no table grows while it is
    // built.
    let (packed_total, scattered_total) =
        recoupling_plan
            .jobs()
            .iter()
            .fold((0usize, 0usize), |(packed, scattered), job| {
                (
                    packed.saturating_add(job.contracted),
                    scattered.saturating_add(job.cols),
                )
            });
    let mut pack_columns = Vec::with_capacity(packed_total);
    // Each scatter column with its destination range `(lo, hi)`, which only
    // compilation reads.
    let mut scatter_spans = Vec::with_capacity(scattered_total);
    let mut scatter_groups = Vec::with_capacity(recoupling_plan.jobs().len());
    // The last packed column's end and whether every column starts after
    // the previous one ends: only the validation below reads them.
    let mut last_packed_hi: Option<usize> = None;
    let mut pack_disjoint = true;
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
        let group_start = scatter_spans.len();
        let pack_start = pack_columns.len();
        if element_count == 0 {
            scatter_groups.push(TreeTransformScatterGroupReplay {
                pack: pack_start..pack_start,
                columns: group_start..group_start,
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
            pack_disjoint &= last_packed_hi.is_none_or(|hi| hi < packed_offset);
            last_packed_hi = Some(packed_hi);
            pack_columns.push(TreeTransformPackReplay {
                src_layout: src_layout_start + src_index,
                packed_offset,
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
            scatter_spans.push((
                (dst_lo, dst_hi),
                TreeTransformScatterReplay {
                    dst_layout,
                    packed_offset,
                },
            ));
        }
        scatter_spans[group_start..].sort_unstable_by_key(|&((dst_lo, _), _)| dst_lo);
        scatter_groups.push(TreeTransformScatterGroupReplay {
            pack: pack_start..pack_columns.len(),
            columns: group_start..scatter_spans.len(),
        });
    }
    // Plan order lays the jobs' packed sources out consecutively, so the
    // columns are pushed in packed order; the check below proves it.
    if !pack_disjoint || last_packed_hi.is_some_and(|hi| hi >= recoupling_plan.source_len()) {
        return Err(OperationError::InvalidArgument {
            message: "tree transform packed source schedule is invalid",
        });
    }

    // Main's replay metadata: the Direct lists in address order, each
    // group's span, and the runnable clamp.
    let singles_slice_disjoint = slice_disjoint(ranges.iter().map(|&(lo, hi, _)| (lo, hi)));
    let mut singles = Vec::with_capacity(ranges.len());
    singles.extend(ranges.iter().map(|&(_, _, block)| block));
    let mut inactive = Vec::with_capacity(inactive_dst_layouts.len());
    for &layout in inactive_dst_layouts {
        if let Some((lo, _)) = destination_span(layouts, layout) {
            inactive.push((lo, checked_u32(layout)?));
        }
    }
    // Why the order is total: compilation rejects aliased destinations.
    inactive.sort_unstable_by_key(|&(lo, _)| lo);
    let inactive = inactive.into_iter().map(|(_, layout)| layout).collect();
    let group_spans = scatter_groups
        .iter()
        .map(|group| {
            let columns = &scatter_spans[group.columns.clone()];
            let ranges = columns.iter().map(|&(range, _)| range);
            TreeTransformGroupSpan {
                lo: columns.first().map_or(0, |&((lo, _), _)| lo),
                hi: ranges.clone().map(|(_, hi)| hi).max().unwrap_or(-1),
                slice_disjoint: slice_disjoint(ranges),
            }
        })
        .collect();
    let singles_runnable = if singles_slice_disjoint {
        singles.len()
    } else {
        usize::from(!singles.is_empty())
    };
    let replay_runnable = singles_runnable.max(pack_columns.len()).max(1);

    let scatter_columns = scatter_spans
        .into_iter()
        .map(|(_, column)| column)
        .collect();
    Ok(TreeTransformParallelSchedule {
        single_block_count,
        packed_column_count,
        scattered_column_count,
        pack_columns,
        scatter_columns,
        scatter_groups,
        inactive,
        singles,
        singles_slice_disjoint,
        group_spans,
        replay_runnable,
    })
}

fn checked_u32(value: usize) -> Result<u32, OperationError> {
    u32::try_from(value).map_err(|_| OperationError::ElementCountOverflow)
}

/// Whether the inclusive destination ranges, in their order, are
/// slice-disjoint: each nonnegative and starting after the previous one ends.
fn slice_disjoint(ranges: impl IntoIterator<Item = (isize, isize)>) -> bool {
    let mut previous_hi = None;
    for (lo, hi) in ranges {
        if lo < 0 || hi < lo || previous_hi.is_some_and(|previous| previous >= lo) {
            return false;
        }
        previous_hi = Some(hi);
    }
    true
}

/// The first and last element a destination `layout` writes, or `None` when
/// it is empty. Exact: destination strides are nonnegative and every address
/// lies below the destination's `usize` required length, so each partial sum
/// stays below `2^64`.
fn destination_span(layouts: &RawLayoutTable, layout_index: usize) -> Option<(i128, i128)> {
    let layout = layouts.entry(layout_index);
    if layout.element_count == 0 {
        return None;
    }
    let (mut lo, mut hi) = (layout.offset as i128, layout.offset as i128);
    for (&extent, &stride) in layouts
        .shape(layout_index)
        .iter()
        .zip(layouts.strides(layout_index))
    {
        let span = (extent as i128 - 1) * stride as i128;
        if span < 0 {
            lo += span;
        } else {
            hi += span;
        }
    }
    Some((lo, hi))
}
