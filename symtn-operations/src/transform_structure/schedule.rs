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
    /// Position of this column's ownership atom in the component-grouped
    /// atom order (`TreeTransformParallelSchedule::scatter_atoms[..].slot`).
    pub slot: u32,
    /// The destination layout offset relative to its atom's first element.
    pub local_base: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TreeTransformScatterGroupReplay {
    pub columns: Range<usize>,
    /// The ownership component of this Multi group's scatter columns.
    pub component: u32,
    /// Elements of the group's largest column with a compiled plan (zero
    /// without one): whether the column may widen on a token.
    pub widest: usize,
}

/// One maximal run of overlapping Single destination intervals, in address
/// order. A Single owns one interval, so each such atom is one ownership
/// component. The list ends with a sentinel whose `item` is the item count
/// and whose `weight_before` is the total.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TreeTransformSingleAtom {
    /// First index into `TreeTransformParallelSchedule::singles`.
    pub item: u32,
    /// First destination element of the atom.
    pub lo: isize,
    /// Destination elements written by every earlier atom.
    pub weight_before: usize,
    /// Elements of the atom's largest Single with a compiled plan (zero
    /// without one).
    pub widest: usize,
}

/// One maximal run of overlapping Multi scatter-column destination
/// intervals, in address order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TreeTransformScatterAtom {
    pub lo: isize,
    pub len: usize,
    /// Position in the component-grouped order: the atoms of component `c`
    /// occupy `component_slots[c]..component_slots[c + 1]`.
    pub slot: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct TreeTransformParallelSchedule {
    /// Indices of the non-empty Single blocks, ordered by destination start.
    /// Why indices rather than copied descriptors: the layout entries,
    /// coefficient and destination range are all read from the block and its
    /// compiled role, so a per-Single copy would duplicate retained state.
    pub singles: Vec<u32>,
    /// Ownership atoms of `singles`, plus the sentinel (empty when there are
    /// no Singles).
    pub single_atoms: Vec<TreeTransformSingleAtom>,
    pub pack_columns: Vec<TreeTransformPackReplay>,
    pub scatter_columns: Vec<TreeTransformScatterReplay>,
    pub scatter_groups: Vec<TreeTransformScatterGroupReplay>,
    /// Ownership atoms of the scatter columns, in address order.
    pub scatter_atoms: Vec<TreeTransformScatterAtom>,
    /// Slot range starts of each scatter component, plus the end (empty when
    /// there are no scatter columns).
    pub component_slots: Vec<u32>,
    pub single_block_count: usize,
    pub packed_column_count: usize,
    pub scattered_column_count: usize,
    /// The largest `widest` of any Single atom.
    pub widest_single: usize,
}

impl TreeTransformParallelSchedule {
    /// Ownership components of the Single blocks: one per atom.
    #[inline]
    pub fn single_component_count(&self) -> usize {
        self.single_atoms.len().saturating_sub(1)
    }

    /// Records the planned-leaf sizes (`widest`) from the finished layout
    /// table, whose roles carry the compiled plans. O(items).
    pub(super) fn record_planned_widths(
        &mut self,
        blocks: &[TreeTransformBlock],
        layouts: &TreeTransformLayoutTable,
    ) -> Result<(), OperationError> {
        let planned = |layout: usize| -> Result<usize, OperationError> {
            Ok(if layouts.role(layout)?.plan.is_some() {
                layouts.entry(layout).element_count
            } else {
                0
            })
        };
        for atom in 0..self.single_component_count() {
            let items =
                self.single_atoms[atom].item as usize..self.single_atoms[atom + 1].item as usize;
            let mut widest = 0;
            for &block in &self.singles[items] {
                if let TreeTransformBlock::Single { dst_layout, .. } = blocks[block as usize] {
                    widest = widest.max(planned(dst_layout)?);
                }
            }
            self.single_atoms[atom].widest = widest;
            self.widest_single = self.widest_single.max(widest);
        }
        for group in &mut self.scatter_groups {
            let mut widest = 0;
            for column in &self.scatter_columns[group.columns.clone()] {
                widest = widest.max(planned(column.dst_layout)?);
            }
            group.widest = widest;
        }
        Ok(())
    }

    /// Ownership components of the Multi scatter columns.
    #[inline]
    pub fn scatter_component_count(&self) -> usize {
        self.component_slots.len().saturating_sub(1)
    }
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
    // Destination ranges are needed only to order the schedule and form its
    // atoms, so they stay in this compile-time list.
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
        let weight = layouts.entry(dst_layout).element_count;
        ranges.push((dst_lo, dst_hi, block_index, weight));
    }
    // Why the order is the base order: compilation rejects aliased
    // destinations and drops empty ones, so no two keys tie.
    ranges.sort_unstable_by_key(|&(dst_lo, ..)| dst_lo);
    let mut singles = Vec::with_capacity(ranges.len());
    let mut single_atoms = Vec::new();
    let mut atom_hi: Option<isize> = None;
    let mut weight_before = 0usize;
    for &(dst_lo, dst_hi, block_index, weight) in &ranges {
        if atom_hi.is_none_or(|hi| dst_lo > hi) {
            single_atoms.push(TreeTransformSingleAtom {
                item: checked_u32(singles.len())?,
                lo: dst_lo,
                weight_before,
                widest: 0,
            });
        }
        atom_hi = Some(atom_hi.map_or(dst_hi, |hi| hi.max(dst_hi)));
        weight_before = weight_before
            .checked_add(weight)
            .ok_or(OperationError::ElementCountOverflow)?;
        singles.push(block_index);
    }
    if let Some(hi) = atom_hi {
        single_atoms.push(TreeTransformSingleAtom {
            item: checked_u32(singles.len())?,
            lo: hi
                .checked_add(1)
                .ok_or(OperationError::ElementCountOverflow)?,
            weight_before,
            widest: 0,
        });
    }
    single_atoms.shrink_to_fit();
    drop(ranges);

    let mut pack_columns = Vec::new();
    let mut scatter_columns = Vec::new();
    let mut scatter_groups = Vec::new();
    // `(dst_lo, dst_hi)` of each scatter column, for the atom sweep.
    let mut scatter_ranges = Vec::new();
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
                component: 0,
                widest: 0,
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
            let Some(range) = layout_index_range(layouts, dst_layout)? else {
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
                slot: 0,
                local_base: 0,
            });
            scatter_ranges.push(range);
        }
        scatter_groups.push(TreeTransformScatterGroupReplay {
            columns: group_start..scatter_columns.len(),
            component: 0,
            widest: 0,
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

    let (scatter_atoms, component_slots) = compile_scatter_components(
        layouts,
        &scatter_ranges,
        &mut scatter_columns,
        &mut scatter_groups,
    )?;

    Ok(TreeTransformParallelSchedule {
        singles,
        single_atoms,
        pack_columns,
        scatter_columns,
        scatter_groups,
        scatter_atoms,
        component_slots,
        single_block_count,
        packed_column_count,
        scattered_column_count,
        widest_single: 0,
    })
}

fn checked_u32(value: usize) -> Result<u32, OperationError> {
    u32::try_from(value).map_err(|_| OperationError::ElementCountOverflow)
}

/// Ownership components of the Multi scatter columns: merges overlapping
/// destination intervals (`ranges[i]` of `columns[i]`) into address-ordered
/// atoms, unions the Multi groups that write into one atom, and numbers the
/// components by their first atom. Records each column's atom slot and local
/// base and each group's component. O(I log I) for I columns: one sort, one
/// sweep and a union-find; no element enumeration and no pairwise tests.
///
/// Why components rather than one bounding slice per group: canonical
/// coupled-sector blocks interleave, so a group's columns can lie in
/// disconnected atoms with other groups' atoms in between; only the atoms
/// themselves are exclusive slices.
fn compile_scatter_components(
    layouts: &RawLayoutTable,
    ranges: &[(isize, isize)],
    columns: &mut [TreeTransformScatterReplay],
    groups: &mut [TreeTransformScatterGroupReplay],
) -> Result<(Vec<TreeTransformScatterAtom>, Vec<u32>), OperationError> {
    if columns.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    let mut column_group = vec![0u32; columns.len()];
    for (group, replay) in groups.iter().enumerate() {
        column_group[replay.columns.clone()].fill(checked_u32(group)?);
    }
    let mut order = (0..checked_u32(columns.len())?).collect::<Vec<u32>>();
    order.sort_unstable_by_key(|&column| ranges[column as usize].0);

    // Sweep: each column's atom, the atom bounds with one owning group, and
    // the union of every group writing into the same atom.
    let mut parent = (0..checked_u32(groups.len())?).collect::<Vec<u32>>();
    let mut column_atom = vec![0u32; columns.len()];
    let mut bounds: Vec<(isize, isize, u32)> = Vec::new();
    for &column in &order {
        let (lo, hi) = ranges[column as usize];
        let group = column_group[column as usize];
        match bounds.last_mut() {
            Some((_, atom_hi, owner)) if lo <= *atom_hi => {
                *atom_hi = (*atom_hi).max(hi);
                union(&mut parent, *owner, group);
            }
            _ => bounds.push((lo, hi, group)),
        }
        column_atom[column as usize] = checked_u32(bounds.len() - 1)?;
    }
    drop(order);

    // Components numbered by first atom; atoms grouped by component, each
    // component's atoms kept in address order.
    let mut component_of_root = vec![u32::MAX; groups.len()];
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
    let mut component_slots = vec![0u32; component_count as usize + 1];
    for &component in &atom_component {
        component_slots[component as usize + 1] += 1;
    }
    for component in 0..component_count as usize {
        component_slots[component + 1] += component_slots[component];
    }
    let mut next_slot = component_slots.clone();
    let mut atoms = Vec::with_capacity(bounds.len());
    for (&(lo, hi, _), &component) in bounds.iter().zip(&atom_component) {
        let slot = next_slot[component as usize];
        next_slot[component as usize] += 1;
        let len = usize::try_from(hi - lo)
            .ok()
            .and_then(|span| span.checked_add(1))
            .ok_or(OperationError::ElementCountOverflow)?;
        atoms.push(TreeTransformScatterAtom { lo, len, slot });
    }

    for (column, &atom) in columns.iter_mut().zip(&column_atom) {
        let atom = atoms[atom as usize];
        column.slot = atom.slot;
        column.local_base = layouts
            .entry(column.dst_layout)
            .offset
            .checked_sub(atom.lo)
            .and_then(|base| usize::try_from(base).ok())
            .ok_or(OperationError::ElementCountOverflow)?;
    }
    for (group, replay) in groups.iter_mut().enumerate() {
        if !replay.columns.is_empty() {
            replay.component = component_of_root[find(&mut parent, checked_u32(group)?) as usize];
        }
    }
    Ok((atoms, component_slots))
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
