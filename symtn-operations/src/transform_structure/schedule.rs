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
    /// Slot of this column's joint atom
    /// (`TreeTransformParallelSchedule::atoms`).
    pub slot: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct TreeTransformScatterGroupReplay {
    /// The group's pack columns in `pack_columns`, in packed order.
    pub pack: Range<usize>,
    /// The group's columns, ordered by destination start and so by joint
    /// atom slot: a component's atoms take their slots in address order.
    pub columns: Range<usize>,
    /// The joint component of this Multi group's scatter columns.
    pub component: u32,
}

/// A destination write that does not go through a packed GEMM column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TreeTransformDirectItem {
    /// An inactive destination layout: zeroed by an overwrite, beta-scaled
    /// by an accumulation.
    Inactive { layout: usize, slot: u32 },
    /// A Single block, by index into the structure's block list.
    Single { block: u32, slot: u32 },
}

impl TreeTransformDirectItem {
    /// The slot of the item's joint atom.
    pub fn slot(self) -> u32 {
        match self {
            Self::Inactive { slot, .. } | Self::Single { slot, .. } => slot,
        }
    }
}

/// One joint atom: a maximal run of overlapping destination intervals of
/// Singles, Multi scatter columns and inactive layouts. Only whole atoms (or
/// whole bands of a bandable atom) may be handed to different writers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TreeTransformJointAtom {
    /// First destination element.
    pub lo: isize,
    /// Elements from `lo` to the last one any item writes.
    pub len: usize,
    /// Destination elements the atom's items write.
    pub weight: usize,
}

/// One atom's band certificate: the common outermost baked destination
/// stride `S` and extent `E >= 2` of every item when the atom is bandable,
/// `(0, 1)` when it is indivisible (one band unit).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TreeTransformAtomBand {
    pub stride: usize,
    pub extent: usize,
}

/// One joint component: the atoms whose owners (Singles, Multi groups and
/// inactive layouts) are connected through shared atoms. The list ends with
/// a sentinel whose starts are the totals.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct TreeTransformJointComponent {
    /// First slot of the component's atoms in
    /// `TreeTransformParallelSchedule::atoms`.
    pub atom_start: u32,
    /// First of the component's entries in `direct`.
    pub direct_start: u32,
    /// First of the component's entries in `component_groups`.
    pub group_start: u32,
    /// Destination elements of its direct items.
    pub direct_weight: usize,
    /// Band units: the bands of its bandable atoms plus one per indivisible
    /// atom.
    pub units: usize,
    /// Largest packed source (`rows * k`) of its groups.
    pub packed_source: usize,
    /// Largest packed destination (`rows * n`) of its groups.
    pub packed_destination: usize,
}

/// The joint ownership schedule of a compiled structure, built eagerly with
/// it and immutable afterwards, so derived equality compares every field.
///
/// The atoms, components, direct items and bands are the
/// one-writer-per-element ownership the lane replay places by.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct TreeTransformParallelSchedule {
    pub pack_columns: Vec<TreeTransformPackReplay>,
    pub scatter_columns: Vec<TreeTransformScatterReplay>,
    pub scatter_groups: Vec<TreeTransformScatterGroupReplay>,
    pub single_block_count: usize,
    pub packed_column_count: usize,
    pub scattered_column_count: usize,
    /// Joint atoms in slot order: grouped by component, each component's
    /// atoms in address order.
    pub atoms: Vec<TreeTransformJointAtom>,
    /// Per atom slot.
    pub bands: Vec<TreeTransformAtomBand>,
    /// Joint components and the sentinel (empty without destination
    /// writes).
    pub components: Vec<TreeTransformJointComponent>,
    /// Direct items grouped by component; within one, inactive layouts and
    /// Singles ordered by atom slot.
    pub direct: Vec<TreeTransformDirectItem>,
    /// Multi group indices grouped by component, in recoupling-plan order.
    pub component_groups: Vec<u32>,
    /// Atom slots in destination address order, along which a replay
    /// splits its destination into atom slices.
    pub address_order: Vec<u32>,
}

impl TreeTransformParallelSchedule {
    /// Joint components.
    pub fn component_count(&self) -> usize {
        self.components.len().saturating_sub(1)
    }

    /// Atom slots of component `component`.
    pub fn component_atoms(&self, component: usize) -> Range<usize> {
        self.components[component].atom_start as usize
            ..self.components[component + 1].atom_start as usize
    }

    /// Direct items of component `component`, ordered by atom slot.
    pub fn component_direct(&self, component: usize) -> &[TreeTransformDirectItem] {
        &self.direct[self.components[component].direct_start as usize
            ..self.components[component + 1].direct_start as usize]
    }

    /// Multi groups of component `component`, in recoupling-plan order.
    pub fn component_group_list(&self, component: usize) -> &[u32] {
        &self.component_groups[self.components[component].group_start as usize
            ..self.components[component + 1].group_start as usize]
    }

    /// The bytes of the vectors this schedule retains, each allocated at its
    /// exact length.
    pub(crate) fn retained_bytes(&self) -> usize {
        let bytes = |capacity: usize, size: usize| capacity.saturating_mul(size);
        bytes(
            self.pack_columns.capacity(),
            core::mem::size_of::<TreeTransformPackReplay>(),
        )
        .saturating_add(bytes(
            self.scatter_columns.capacity(),
            core::mem::size_of::<TreeTransformScatterReplay>(),
        ))
        .saturating_add(bytes(
            self.scatter_groups.capacity(),
            core::mem::size_of::<TreeTransformScatterGroupReplay>(),
        ))
        .saturating_add(bytes(
            self.atoms.capacity(),
            core::mem::size_of::<TreeTransformJointAtom>(),
        ))
        .saturating_add(bytes(
            self.bands.capacity(),
            core::mem::size_of::<TreeTransformAtomBand>(),
        ))
        .saturating_add(bytes(
            self.components.capacity(),
            core::mem::size_of::<TreeTransformJointComponent>(),
        ))
        .saturating_add(bytes(
            self.direct.capacity(),
            core::mem::size_of::<TreeTransformDirectItem>(),
        ))
        .saturating_add(bytes(
            self.component_groups.capacity(),
            core::mem::size_of::<u32>(),
        ))
        .saturating_add(bytes(
            self.address_order.capacity(),
            core::mem::size_of::<u32>(),
        ))
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
                component: 0,
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
            component: 0,
        });
    }
    // Plan order lays the jobs' packed sources out consecutively, so the
    // columns are pushed in packed order; the check below proves it.
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
    for (group, replay) in schedule.scatter_groups.iter_mut().enumerate() {
        if replay.columns.is_empty() {
            continue;
        }
        let component = component_of_root[find(&mut parent, checked_u32(group)?) as usize] as usize;
        replay.component = checked_u32(component)?;
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
        let at = &mut next_group[replay.component as usize];
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
