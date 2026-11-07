use super::*;

/// Compiles the pack and scatter replay columns and the joint ownership
/// schedule from the raw layout table; bands follow from the retained table
/// ([`compile_atom_bands`]).
///
/// Joint ownership merges the inclusive destination intervals of every
/// Single block, Multi scatter column and inactive layout into
/// address-ordered atoms (while `lo <= atom_hi`), unions the owners (Single,
/// Multi group, inactive layout) of each atom, and numbers the components by
/// their first atom. O(I log I + B) time and O(I) retained metadata for I
/// intervals and B blocks: one sort, one sweep, a union-find and counting
/// passes, no element enumeration.
///
/// Why components rather than one bounding slice per owner: canonical
/// coupled-sector blocks interleave, so an owner's intervals can lie in
/// disconnected atoms with other owners' atoms between them; only the atoms
/// themselves are exclusive slices, and one component writes all of its own.
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
                    slot: 0,
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

    let mut scatter_ranges = Vec::with_capacity(scatter_spans.len());
    let mut scatter_columns = Vec::with_capacity(scatter_spans.len());
    for (range, column) in scatter_spans {
        scatter_ranges.push(range);
        scatter_columns.push(column);
    }
    let mut schedule = TreeTransformParallelSchedule {
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
        ..Default::default()
    };
    compile_joint_ownership(
        &mut schedule,
        blocks,
        &scatter_ranges,
        &ranges,
        inactive_dst_layouts,
        layouts,
        recoupling_plan,
    )?;
    Ok(schedule)
}

/// The joint atoms, components, direct items and component groups of
/// [`compile_parallel_schedule`], and each scatter column's atom slot and
/// each group's component. `scatters` are the scatter columns' `(lo, hi)`
/// and `singles` the Singles' `(lo, hi, block)` destination ranges.
fn compile_joint_ownership(
    schedule: &mut TreeTransformParallelSchedule,
    blocks: &[TreeTransformBlock],
    scatters: &[(isize, isize)],
    singles: &[(isize, isize, u32)],
    inactive_dst_layouts: &[usize],
    layouts: &RawLayoutTable,
    recoupling_plan: &TreeTransformRecouplingPlan,
) -> Result<(), OperationError> {
    let groups = schedule.scatter_groups.len();
    // Every destination interval: `(lo, hi, owner, item)`. Owners are the
    // groups `0..G`, then the Singles, then the inactive layouts; `item`
    // indexes `scatter_columns` for a group's interval and is the block or
    // layout otherwise. Endpoints are `i128`: an inactive layout's last
    // element may lie beyond `isize::MAX` inside the `usize` storage.
    let mut intervals: Vec<(i128, i128, u32, usize)> = Vec::with_capacity(
        schedule
            .scatter_columns
            .len()
            .saturating_add(singles.len())
            .saturating_add(inactive_dst_layouts.len()),
    );
    for (group, replay) in schedule.scatter_groups.iter().enumerate() {
        for column in replay.columns.clone() {
            let (lo, hi) = scatters[column];
            intervals.push((lo as i128, hi as i128, checked_u32(group)?, column));
        }
    }
    let mut owners = groups;
    for &(lo, hi, block) in singles {
        intervals.push((lo as i128, hi as i128, checked_u32(owners)?, block as usize));
        owners += 1;
    }
    let single_owners = owners;
    for &layout in inactive_dst_layouts {
        if let Some((lo, hi)) = destination_span(layouts, layout) {
            intervals.push((lo, hi, checked_u32(owners)?, layout));
            owners += 1;
        }
    }
    if intervals.is_empty() {
        return Ok(());
    }
    // Why the order is total: compilation rejects aliased destinations and
    // drops empty ones, so no two intervals start at one element.
    intervals.sort_unstable_by_key(|&(lo, ..)| lo);

    // Sweep: each interval's atom, the atom bounds with one owner, and the
    // union of every owner writing into the same atom.
    let mut parent = (0..checked_u32(owners)?).collect::<Vec<u32>>();
    let mut interval_atom = Vec::with_capacity(intervals.len());
    let mut bounds: Vec<(i128, i128, u32)> = Vec::with_capacity(intervals.len());
    for &(lo, hi, owner, _) in &intervals {
        match bounds.last_mut() {
            Some((_, atom_hi, first)) if lo <= *atom_hi => {
                *atom_hi = (*atom_hi).max(hi);
                union(&mut parent, *first, owner);
            }
            _ => bounds.push((lo, hi, owner)),
        }
        interval_atom.push(checked_u32(bounds.len() - 1)?);
    }

    // Components numbered by first atom; atoms grouped by component, each
    // component's atoms kept in address order.
    let mut component_of_root = vec![u32::MAX; owners];
    let mut atom_component = Vec::with_capacity(bounds.len());
    let mut component_count = 0u32;
    for &(_, _, owner) in &bounds {
        let root = find(&mut parent, owner) as usize;
        if component_of_root[root] == u32::MAX {
            component_of_root[root] = component_count;
            component_count += 1;
        }
        atom_component.push(component_of_root[root]);
    }
    let components_len = component_count as usize;
    let mut atom_starts = vec![0u32; components_len + 1];
    for &component in &atom_component {
        atom_starts[component as usize + 1] += 1;
    }
    for component in 0..components_len {
        atom_starts[component + 1] += atom_starts[component];
    }
    let mut next_slot = atom_starts.clone();
    let mut atom_slot = Vec::with_capacity(bounds.len());
    let mut atoms = vec![
        TreeTransformJointAtom {
            lo: 0,
            len: 0,
            weight: 0,
        };
        bounds.len()
    ];
    for (&(lo, hi, _), &component) in bounds.iter().zip(&atom_component) {
        let slot = next_slot[component as usize];
        next_slot[component as usize] += 1;
        // Both fit: `lo` is a layout offset, already an `isize`, and the
        // atom lies inside the destination's `usize` storage.
        atoms[slot as usize].lo =
            isize::try_from(lo).map_err(|_| OperationError::ElementCountOverflow)?;
        atoms[slot as usize].len =
            usize::try_from(hi - lo + 1).map_err(|_| OperationError::ElementCountOverflow)?;
        atom_slot.push(slot);
    }
    drop(next_slot);
    // `bounds` is in address order, so the slots in its order are.
    schedule.address_order = atom_slot.clone();

    // Items: their atom slot and weight, the components' direct items (in
    // address order).
    let mut direct_weight = vec![0usize; components_len];
    let mut items: Vec<(u32, TreeTransformDirectItem)> = Vec::with_capacity(intervals.len());
    for (&(_, _, owner, item), &atom) in intervals.iter().zip(&interval_atom) {
        let slot = atom_slot[atom as usize];
        let component = atom_component[atom as usize];
        let (layout, direct) = if (owner as usize) < groups {
            let column = &mut schedule.scatter_columns[item];
            column.slot = slot;
            (column.dst_layout, None)
        } else if (owner as usize) < single_owners {
            let TreeTransformBlock::Single { dst_layout, .. } = blocks[item] else {
                unreachable!("Single owners name Single blocks");
            };
            (
                dst_layout,
                Some(TreeTransformDirectItem::Single {
                    block: item as u32,
                    slot,
                }),
            )
        } else {
            (
                item,
                Some(TreeTransformDirectItem::Inactive { layout: item, slot }),
            )
        };
        let weight = layouts.entry(layout).element_count;
        let atom = &mut atoms[slot as usize];
        atom.weight = atom
            .weight
            .checked_add(weight)
            .ok_or(OperationError::ElementCountOverflow)?;
        if let Some(direct) = direct {
            direct_weight[component as usize] = direct_weight[component as usize]
                .checked_add(weight)
                .ok_or(OperationError::ElementCountOverflow)?;
            items.push((component, direct));
        }
    }
    drop(intervals);
    // A stable bucket by component over the address-ordered items: within a
    // component, address order is slot order. O(I), no second sort.
    let mut direct_starts = vec![0u32; components_len + 1];
    for &(component, _) in &items {
        direct_starts[component as usize + 1] += 1;
    }
    for component in 0..components_len {
        direct_starts[component + 1] += direct_starts[component];
    }
    let mut next_direct = direct_starts.clone();
    let mut direct = vec![TreeTransformDirectItem::Inactive { layout: 0, slot: 0 }; items.len()];
    for &(component, item) in &items {
        let at = &mut next_direct[component as usize];
        direct[*at as usize] = item;
        *at += 1;
    }
    drop((items, next_direct));

    // Groups: their component and packed scratch, and each
    // component's groups in recoupling-plan order.
    let mut scratch = vec![(0usize, 0usize); components_len];
    let mut group_counts = vec![0u32; components_len + 1];
    for (group, replay) in schedule.scatter_groups.iter().enumerate() {
        if replay.columns.is_empty() {
            continue;
        }
        let component = component_of_root[find(&mut parent, checked_u32(group)?) as usize] as usize;
        group_counts[component + 1] += 1;
        let job = &recoupling_plan.jobs()[group];
        let source = job
            .rows
            .checked_mul(job.contracted)
            .ok_or(OperationError::ElementCountOverflow)?;
        let destination = job
            .rows
            .checked_mul(job.cols)
            .ok_or(OperationError::ElementCountOverflow)?;
        let (max_source, max_destination) = &mut scratch[component];
        *max_source = (*max_source).max(source);
        *max_destination = (*max_destination).max(destination);
    }
    for component in 0..components_len {
        group_counts[component + 1] += group_counts[component];
    }
    let mut component_groups = vec![0u32; group_counts[components_len] as usize];
    let mut next_group = group_counts.clone();
    for (group, replay) in schedule.scatter_groups.iter().enumerate() {
        if replay.columns.is_empty() {
            continue;
        }
        let component = component_of_root[find(&mut parent, checked_u32(group)?) as usize];
        let at = &mut next_group[component as usize];
        component_groups[*at as usize] = checked_u32(group)?;
        *at += 1;
    }

    schedule.components = (0..=components_len)
        .map(|component| {
            let (packed_source, packed_destination) =
                scratch.get(component).copied().unwrap_or((0, 0));
            TreeTransformJointComponent {
                atom_start: atom_starts[component],
                direct_start: direct_starts[component],
                group_start: group_counts[component],
                direct_weight: direct_weight.get(component).copied().unwrap_or(0),
                units: 0,
                packed_source,
                packed_destination,
            }
        })
        .collect();
    schedule.atoms = atoms;
    schedule.direct = direct;
    schedule.component_groups = component_groups;
    Ok(())
}

/// Fills the atoms' band certificates and the components' band units from
/// the retained layout table, at compile time with the rest of the
/// schedule. O(I r) for I items of fused rank at most r.
///
/// An atom is bandable when every item's baked layout has the same
/// outermost (largest destination stride) dimension, of stride `S > 0` and
/// extent `E >= 2`, nonnegative destination strides, a rank within the
/// inline traversal limit, and an inner reach that fits one slab:
/// `o - lo + Σ_{j < outer} (dim_j - 1) stride_j < S` for item offset `o` and
/// atom start `lo`. Then band `k` holds exactly the item elements of outer
/// index `k`, at `[lo + k S, lo + (k + 1) S)` cut at the atom's end, for
/// every item alike. Every destination role of the atom takes part: a
/// Single's and a scatter column's two-sided role, and an inactive layout's
/// one-sided role.
pub(super) fn compile_atom_bands(
    schedule: &mut TreeTransformParallelSchedule,
    blocks: &[TreeTransformBlock],
    layouts: &TreeTransformLayoutTable,
) -> Result<(), OperationError> {
    // Per atom `None` until its first item, then `Some(Some((S, E)))` while
    // every item agrees on one band, `Some(None)` once one does not.
    let mut states: Vec<Option<Option<(usize, usize)>>> = vec![None; schedule.atoms.len()];
    let atoms = &schedule.atoms;
    let mut record = |slot: u32, entry: usize, dims: &[usize], strides: &[isize]| {
        let offset = layouts.entry(entry).offset;
        let band = item_band(atoms[slot as usize].lo, offset, dims, strides);
        let state = &mut states[slot as usize];
        *state = Some(match *state {
            None => band,
            Some(previous) => previous.filter(|&previous| Some(previous) == band),
        });
    };
    for &item in &schedule.direct {
        match item {
            TreeTransformDirectItem::Single { block, slot } => {
                let TreeTransformBlock::Single { dst_layout, .. } = blocks[block as usize] else {
                    return Err(OperationError::InvalidArgument {
                        message: "tree transform direct item names a non-Single block",
                    });
                };
                let role = layouts.role(dst_layout)?;
                record(slot, dst_layout, role.dims, role.dst_strides);
            }
            TreeTransformDirectItem::Inactive { layout, slot } => {
                let (dims, strides) = layouts.inactive_role(layout)?;
                record(slot, layout, dims, strides);
            }
        }
    }
    for group in &schedule.scatter_groups {
        for column in &schedule.scatter_columns[group.columns.clone()] {
            let role = layouts.role(column.dst_layout)?;
            record(column.slot, column.dst_layout, role.dims, role.dst_strides);
        }
    }
    // Why `iter` rather than `into_iter`: an in-place collect would keep
    // the larger state allocation and exceed the charged size.
    schedule.bands = states
        .iter()
        .map(|&state| {
            let (stride, extent) = state.flatten().unwrap_or((0, 1));
            TreeTransformAtomBand { stride, extent }
        })
        .collect();
    for component in 0..schedule.component_count() {
        schedule.components[component].units = schedule.bands[schedule.component_atoms(component)]
            .iter()
            .try_fold(0usize, |units, band| units.checked_add(band.extent))
            .ok_or(OperationError::ElementCountOverflow)?;
    }
    Ok(())
}

/// The band `(S, E)` of one item with baked layout `(dims, strides)` at
/// offset `offset` in the atom starting at `lo`, or `None` when the item
/// cannot be banded (see [`compile_atom_bands`]).
fn item_band(
    lo: isize,
    offset: isize,
    dims: &[usize],
    strides: &[isize],
) -> Option<(usize, usize)> {
    let (&extent, inner_dims) = dims.split_last()?;
    let (&stride, inner_strides) = strides.split_last()?;
    if dims.len() > crate::kernel_adapter::FUSED_LAYOUT_INLINE_RANK
        || extent < 2
        || stride <= 0
        || strides.iter().any(|&stride| stride < 0)
    {
        return None;
    }
    let mut reach = offset.checked_sub(lo).filter(|&delta| delta >= 0)?;
    for (&dim, &inner) in inner_dims.iter().zip(inner_strides) {
        let span = isize::try_from(dim.checked_sub(1)?)
            .ok()?
            .checked_mul(inner)?;
        reach = reach.checked_add(span)?;
    }
    (reach < stride).then_some((stride as usize, extent))
}

fn checked_u32(value: usize) -> Result<u32, OperationError> {
    u32::try_from(value).map_err(|_| OperationError::ElementCountOverflow)
}

/// Union-find root with path halving.
fn find(parent: &mut [u32], mut node: u32) -> u32 {
    while parent[node as usize] != node {
        let grandparent = parent[parent[node as usize] as usize];
        parent[node as usize] = grandparent;
        node = grandparent;
    }
    node
}

/// Joins two union-find sets under the smaller root.
fn union(parent: &mut [u32], left: u32, right: u32) {
    let (left, right) = (find(parent, left), find(parent, right));
    if left != right {
        parent[left.max(right) as usize] = left.min(right);
    }
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
