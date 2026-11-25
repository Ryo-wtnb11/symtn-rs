use super::*;

/// Compiles the pack and scatter replay columns and main's replay dispatch
/// metadata from the raw layout table: the Direct lists in destination
/// address order with their hull runs, each plan group's scatter hull, and
/// the runnable clamp. O(I log I + B) time and O(I) retained metadata for I
/// intervals and B blocks: one sort of the Singles, one of the inactive
/// layouts and one per group's columns, and one linear sweep per list.
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
    let (packed_total, scattered_total) = recoupling_plan
        .jobs()
        .iter()
        .try_fold((0usize, 0usize), |(packed, scattered), job| {
            Some((
                packed.checked_add(job.contracted)?,
                scattered.checked_add(job.cols)?,
            ))
        })
        .ok_or(OperationError::ElementCountOverflow)?;
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

    // Main's replay metadata: the Direct lists in address order with their
    // hull runs, each group's hull, and the runnable clamp.
    let single_runs = hull_runs(ranges.iter().map(|&(lo, hi, _)| (lo, hi)));
    let singles_slice_disjoint = single_runs.len() == ranges.len();
    let mut singles = Vec::with_capacity(ranges.len());
    singles.extend(ranges.iter().map(|&(_, _, block)| block));
    let mut inactive = Vec::with_capacity(inactive_dst_layouts.len());
    for &layout in inactive_dst_layouts {
        if let Some(span) = destination_span(layouts, layout) {
            inactive.push((span, checked_u32(layout)?));
        }
    }
    // Why the order is total: compilation rejects aliased destinations.
    inactive.sort_unstable_by_key(|&((lo, _), _)| lo);
    let inactive_runs = hull_runs(inactive.iter().map(|&(span, _)| span));
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
                planned: 0,
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
        pack_columns,
        scatter_columns,
        scatter_groups,
        inactive,
        inactive_runs,
        singles,
        single_runs,
        singles_planned: 0,
        single_movement: MovementSummary::default(),
        inactive_movement: MovementSummary::default(),
        group_spans,
        replay_runnable,
    })
}

fn checked_u32(value: usize) -> Result<u32, OperationError> {
    u32::try_from(value).map_err(|_| OperationError::ElementCountOverflow)
}

/// Records, from the retained layout table, the largest planned Single and
/// each group's largest planned scatter column of `schedule`: a role is
/// planned when the table retains a Strided plan for it.
pub(super) fn record_planned(
    schedule: &mut TreeTransformParallelSchedule,
    blocks: &[TreeTransformBlock],
    layouts: &TreeTransformLayoutTable,
    source_tiled: bool,
) -> Result<(), OperationError> {
    schedule.single_movement =
        movement_summary(schedule, blocks, layouts, true, source_tiled, |_| false)?;
    schedule.inactive_movement =
        movement_summary(schedule, blocks, layouts, false, true, |_| false)?;
    let planned = |layout: usize| -> Result<usize, OperationError> {
        Ok(if layouts.role(layout)?.plan.is_some() {
            layouts.entry(layout).element_count
        } else {
            0
        })
    };
    let mut singles = 0;
    for &block in &schedule.singles {
        if let TreeTransformBlock::Single { dst_layout, .. } = blocks[block as usize] {
            singles = singles.max(planned(dst_layout)?);
        }
    }
    schedule.singles_planned = singles;
    for (group, span) in schedule.group_spans.iter_mut().enumerate() {
        let columns = schedule.scatter_groups[group].columns.clone();
        span.planned = schedule.scatter_columns[columns]
            .iter()
            .map(|column| planned(column.dst_layout))
            .try_fold(0, |max, planned| planned.map(|planned| max.max(planned)))?;
    }
    Ok(())
}

pub(super) fn movement_summary(
    schedule: &TreeTransformParallelSchedule,
    blocks: &[TreeTransformBlock],
    layouts: &TreeTransformLayoutTable,
    singles: bool,
    source_tiled: bool,
    scaled: impl Fn(usize) -> bool,
) -> Result<MovementSummary, OperationError> {
    let (list, jobs) = if singles {
        (&schedule.singles, &schedule.single_runs)
    } else {
        (&schedule.inactive, &schedule.inactive_runs)
    };
    let mut result = MovementSummary {
        exact_working_set: source_tiled,
        supported_span: true,
        ..Default::default()
    };
    let mut sources = std::collections::BTreeMap::new();
    for (job, &start) in jobs.iter().enumerate() {
        let end = jobs.get(job + 1).map_or(list.len(), |&v| v as usize);
        let mut counts = MovementCounts::default();
        for &item in &list[start as usize..end] {
            let (layout, source, scaled) = if singles {
                let TreeTransformBlock::Single {
                    dst_layout,
                    src_layout,
                    coefficient,
                } = blocks[item as usize]
                else {
                    unreachable!("compiled Single list")
                };
                (dst_layout, Some(src_layout), scaled(coefficient))
            } else {
                (item as usize, None, false)
            };
            let elements = layouts.entry(layout).element_count;
            if elements == 0 {
                continue;
            }
            let (inner, kind) = if singles {
                let role = layouts.role(layout)?;
                result.supported_span &= role.dst_strides()[0] != 0 || role.src_strides()[0] != 0;
                (
                    role.dims()[0],
                    usize::from(role.dst_strides()[0] != 1 || role.src_strides()[0] != 1),
                )
            } else {
                let (dims, strides) = layouts.inactive_role(layout)?;
                result.supported_span &= strides[0] != 0;
                (dims[0], 1)
            };
            let kind = kind + 2 * usize::from(scaled);
            let runs = elements
                .checked_div(inner)
                .ok_or(OperationError::ElementCountOverflow)?;
            let add =
                |a: usize, b: usize| a.checked_add(b).ok_or(OperationError::ElementCountOverflow);
            counts.roles[kind] = add(counts.roles[kind], 1)?;
            counts.runs[kind] = add(counts.runs[kind], runs)?;
            counts.elements[kind] = add(counts.elements[kind], elements)?;
            result.working_set_elements = add(result.working_set_elements, elements)?;
            if let Some(source) = source {
                sources
                    .entry(layouts.entry(source).offset)
                    .or_insert(elements);
            }
            if (elements, runs) > (result.largest.elements, result.largest.runs) {
                result.largest = MovementItem {
                    runs,
                    elements,
                    kind,
                    planned: singles && layouts.role(layout)?.plan.is_some(),
                };
            }
        }
        for kind in 0..4 {
            result.total.roles[kind] = result.total.roles[kind]
                .checked_add(counts.roles[kind])
                .ok_or(OperationError::ElementCountOverflow)?;
            result.max.roles[kind] = result.max.roles[kind].max(counts.roles[kind]);
            result.total.runs[kind] = result.total.runs[kind]
                .checked_add(counts.runs[kind])
                .ok_or(OperationError::ElementCountOverflow)?;
            result.total.elements[kind] = result.total.elements[kind]
                .checked_add(counts.elements[kind])
                .ok_or(OperationError::ElementCountOverflow)?;
            result.max.runs[kind] = result.max.runs[kind].max(counts.runs[kind]);
            result.max.elements[kind] = result.max.elements[kind].max(counts.elements[kind]);
        }
    }
    for elements in sources.into_values() {
        result.working_set_elements = result
            .working_set_elements
            .checked_add(elements)
            .ok_or(OperationError::ElementCountOverflow)?;
    }
    Ok(result)
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

/// The start index of each hull run of the inclusive destination ranges,
/// in their (ascending `lo`) order: a run ends where the next range starts
/// after the largest end so far.
///
/// Why the running maximum, not the previous end: a long range followed by
/// short ones inside it (`[0, 100], [1, 2], [3, 4]`) is one run.
fn hull_runs<N: Copy + Ord>(ranges: impl IntoIterator<Item = (N, N)>) -> Vec<u32> {
    let mut starts = Vec::new();
    let mut end: Option<N> = None;
    for (index, (lo, hi)) in ranges.into_iter().enumerate() {
        match end {
            Some(last) if lo <= last => end = Some(last.max(hi)),
            _ => {
                // Indices fit `u32`: the lists are indexed by `u32` already.
                starts.push(index as u32);
                end = Some(hi);
            }
        }
    }
    starts.shrink_to_fit();
    starts
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TreeTransformBlockSpec, TreeTransformStructure};
    use symtn_core::{BlockKey, BlockSpec, BlockStructure};

    /// What: a run ends only where a range starts after the largest end so
    /// far, so a long range followed by ranges inside it is one run, chained
    /// overlapping ranges are one run, and disjoint ranges are one run each.
    #[test]
    fn hull_runs_follow_the_running_maximum() {
        assert_eq!(hull_runs([(0, 100), (1, 2), (3, 4)]), [0]);
        assert_eq!(hull_runs([(0, 3), (3, 6), (6, 9), (10, 11)]), [0, 3]);
        assert_eq!(hull_runs([(0, 1), (2, 3), (4, 5)]), [0, 1, 2]);
        assert!(hull_runs(core::iter::empty::<(isize, isize)>()).is_empty());
    }

    /// What: two coupled matrices of 4 rows by 3 columns (column-major,
    /// stride 4), each with three one-tree specs on rows 0 to 2 (compiled
    /// as Singles; matrix 0's row 2 one column wide) and an inactive row 3,
    /// all interleaved in its columns: the Singles form one run per matrix,
    /// so they are not slice-disjoint, and the inactive rows one run each.
    #[test]
    fn interleaved_matrices_compile_one_run_each() {
        let mut shapes = Vec::new();
        let mut blocks = Vec::new();
        for matrix in 0..2 {
            for row in 0..4 {
                let cols = if matrix == 0 && row == 2 { 1 } else { 3 };
                shapes.push(vec![1, cols]);
                blocks.push(
                    BlockSpec::with_key(
                        BlockKey::ordinal(blocks.len()),
                        vec![1, cols],
                        vec![1, 4],
                        12 * matrix + row,
                    )
                    .unwrap(),
                );
            }
        }
        let dst = BlockStructure::from_blocks_with_rank(2, blocks).unwrap();
        let src = BlockStructure::packed_column_major(2, shapes).unwrap();
        let specs = (0..2)
            .flat_map(|matrix| {
                (0..3).map(move |row| {
                    let key = 4 * matrix + row;
                    TreeTransformBlockSpec::multi(vec![key], vec![key], vec![1.0])
                })
            })
            .collect::<Vec<_>>();
        let transform = TreeTransformStructure::compile_structures(&dst, &src, &specs).unwrap();
        let schedule = transform.parallel_schedule();
        assert_eq!(schedule.singles.len(), 6);
        assert_eq!(schedule.single_runs, [0, 3]);
        assert!(!schedule.singles_slice_disjoint());
        assert_eq!(schedule.inactive_runs, [0, 1]);
    }

    /// What: a group whose second column by start (three elements of stride
    /// 1 from element 2) ends before its first (stride 4 from element 1):
    /// the group's hull ends at the first column's end, 9, not at its last
    /// column's, 4.
    #[test]
    fn a_group_hull_ends_at_its_largest_column_end() {
        let dst = BlockStructure::from_blocks_with_rank(
            2,
            vec![
                BlockSpec::with_key(BlockKey::ordinal(0), vec![1, 3], vec![1, 4], 1).unwrap(),
                BlockSpec::with_key(BlockKey::ordinal(1), vec![1, 3], vec![1, 1], 2).unwrap(),
            ],
        )
        .unwrap();
        let src = BlockStructure::packed_column_major(2, [vec![1, 3], vec![1, 3]]).unwrap();
        let spec = TreeTransformBlockSpec::multi(vec![0, 1], vec![0, 1], vec![1.0, 0.5, 0.5, 1.0]);
        let transform = TreeTransformStructure::compile_structures(&dst, &src, &[spec]).unwrap();
        let span = transform.parallel_schedule().group_spans[0];
        assert_eq!((span.lo, span.hi), (1, 9));
    }
    /// Counts follow the baked traversal and inactive zero-source arm;
    /// removing the transpose is the control changing the copy class only.
    #[test]
    fn movement_summaries_count_actual_traversal() {
        let dst =
            BlockStructure::packed_column_major(2, [vec![2, 2], vec![2, 2], vec![2, 2]]).unwrap();
        let src = BlockStructure::packed_column_major(2, [vec![2, 2]]).unwrap();
        for transpose in [true, false] {
            let second = TreeTransformBlockSpec::single(1, 0, 1.0);
            let second = if transpose {
                second.with_source_axes([1, 0])
            } else {
                second
            };
            let transform = TreeTransformStructure::compile_structures(
                &dst,
                &src,
                &[TreeTransformBlockSpec::single(0, 0, 1.0), second],
            )
            .unwrap();
            let schedule = transform.parallel_schedule();
            let expected = if transpose {
                MovementCounts {
                    roles: [1, 1, 0, 0],
                    runs: [1, 2, 0, 0],
                    elements: [4, 4, 0, 0],
                }
            } else {
                MovementCounts {
                    roles: [2, 0, 0, 0],
                    runs: [2, 0, 0, 0],
                    elements: [8, 0, 0, 0],
                }
            };
            assert_eq!(schedule.single_movement.total, expected);
            assert_eq!(schedule.single_movement.working_set_elements, 12);
            let max = if transpose {
                expected
            } else {
                MovementCounts {
                    roles: [1, 0, 0, 0],
                    runs: [1, 0, 0, 0],
                    elements: [4, 0, 0, 0],
                }
            };
            assert_eq!(schedule.single_movement.max, max);
            assert_eq!(
                (
                    schedule.single_movement.largest.elements,
                    schedule.single_movement.largest.runs,
                    schedule.single_movement.largest.kind
                ),
                (4, if transpose { 2 } else { 1 }, usize::from(transpose))
            );
            assert_eq!(
                schedule.inactive_movement.total,
                MovementCounts {
                    roles: [0, 1, 0, 0],
                    runs: [0, 1, 0, 0],
                    elements: [0, 4, 0, 0]
                }
            );
            assert_eq!(schedule.inactive_movement.working_set_elements, 4);
        }
    }
    /// A Single retains its role's planned capability; an inactive role of
    /// the same geometry is the control that always remains serial.
    #[test]
    fn largest_role_marks_only_planned_singles() {
        let dst = BlockStructure::packed_column_major(1, [vec![24], vec![24]]).unwrap();
        let src = BlockStructure::packed_column_major(1, [vec![24]]).unwrap();
        let transform = TreeTransformStructure::compile_structures(
            &dst,
            &src,
            &[TreeTransformBlockSpec::single(0, 0, 1.0)],
        )
        .unwrap();
        let schedule = transform.parallel_schedule();
        assert!(schedule.single_movement.largest.planned);
        assert!(!schedule.inactive_movement.largest.planned);
    }

    /// Scalar normalization uses the uncalibrated broadcast arm for both copy
    /// and fill; a two-element role is the control admitted to the span model.
    #[test]
    fn scalar_roles_keep_the_uncalibrated_span_marker() {
        for elements in [1, 2] {
            let dst =
                BlockStructure::packed_column_major(1, [vec![elements], vec![elements]]).unwrap();
            let src = BlockStructure::packed_column_major(1, [vec![elements]]).unwrap();
            let transform = TreeTransformStructure::compile_structures(
                &dst,
                &src,
                &[TreeTransformBlockSpec::single(0, 0, 1.0)],
            )
            .unwrap();
            let schedule = transform.parallel_schedule();
            assert_eq!(schedule.single_movement.supported_span, elements > 1);
            assert_eq!(schedule.inactive_movement.supported_span, elements > 1);
        }
    }
}
