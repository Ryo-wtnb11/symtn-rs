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
    /// The group's columns, ordered by joint atom slot.
    pub columns: Range<usize>,
    /// The joint component of this Multi group's scatter columns.
    pub component: u32,
}

/// A destination write that does not go through a packed GEMM column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TreeTransformDirectItem {
    /// An inactive destination layout: zeroed by an overwrite, beta-scaled
    /// by an accumulation.
    Inactive { layout: u32, slot: u32 },
    /// A Single block, by index into the structure's block list.
    Single { block: u32, slot: u32 },
}

impl TreeTransformDirectItem {
    /// The joint atom slot the item writes.
    #[inline]
    pub fn slot(self) -> u32 {
        match self {
            Self::Inactive { slot, .. } | Self::Single { slot, .. } => slot,
        }
    }
}

/// One joint atom: a maximal run of overlapping destination intervals of
/// Singles, Multi scatter columns and inactive layouts, which one lane slot
/// writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TreeTransformJointAtom {
    /// First destination element.
    pub lo: isize,
    /// Elements from `lo` to the last one any item writes.
    pub len: usize,
    /// Destination elements the atom's items write.
    pub weight: usize,
}

/// One atom's band: the common outermost baked destination stride `S` and
/// extent `E >= 2` of every item when the atom is bandable, `(0, 1)` when it
/// is indivisible (one band unit).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TreeTransformAtomBand {
    pub stride: usize,
    pub extent: usize,
}

impl TreeTransformAtomBand {
    /// Whether band `k` of the atom covers `[lo + k S, lo + (k + 1) S)` for
    /// every item, so a lane may write a range of its bands.
    #[inline]
    pub fn bandable(&self) -> bool {
        self.stride != 0
    }
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
    /// Elements the component moves: every destination element it writes
    /// and every packed source element its groups pack.
    pub moved: usize,
    /// Destination elements of its direct items.
    pub direct_weight: usize,
}

/// One joint component's multi-slot facts ([`TreeTransformLaneSchedule`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct TreeTransformLaneComponent {
    /// Elements of its largest Single with a compiled plan (zero without
    /// one).
    pub widest_direct: usize,
    /// Band units: the bands of its bandable atoms plus one per indivisible
    /// atom.
    pub units: usize,
    /// Whether the component is one Single with a compiled plan and no
    /// Multi group: an overwrite may run it whole on its plan at any width.
    pub planned_single: bool,
}

/// What only a replay on more than one slot reads: the atoms' bands, the
/// components' band units, widest planned Singles and whole-plan
/// eligibility, the groups' widest planned scatter columns and the cost
/// classes. Built from the retained layout table on the first such replay
/// of a compiled structure ([`TreeTransformParallelSchedule::lanes`]); a
/// one-slot replay never builds it.
///
/// Why not at compile: a one-slot replay reads none of it, and a caller
/// that rebuilds its structure per call (a disabled structure cache) would
/// pay it on every serial call.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct TreeTransformLaneSchedule {
    /// Per atom slot.
    pub bands: Vec<TreeTransformAtomBand>,
    /// Per component (no sentinel).
    pub components: Vec<TreeTransformLaneComponent>,
    /// Per Multi group: elements of its largest scatter column with a
    /// compiled plan (zero without one), whether the column may widen.
    pub group_widest: Vec<usize>,
    /// Band units of every component.
    pub band_units: usize,
    /// Each component's cost class: components of one class have equal
    /// moved elements, band units, eligibility to run whole
    /// ([`TreeTransformLaneComponent::planned_single`]) and multisets of
    /// Multi job shapes, so every width costs them alike.
    pub component_class: Vec<u32>,
    /// Number of distinct cost classes.
    pub class_count: usize,
}

/// One step of the whole-call one-slot iteration
/// ([`TreeTransformParallelSchedule::one_slot_order`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OneSlotStep {
    /// The direct item `direct[index]` of a component without Multi groups.
    Direct(u32),
    /// A whole component with Multi groups: its inactive writes, Singles,
    /// then each group.
    Component(u32),
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct TreeTransformParallelSchedule {
    /// Joint atoms in slot order: grouped by component, each component's
    /// atoms in address order.
    pub atoms: Vec<TreeTransformJointAtom>,
    /// Atom slots in address order: how a destination splits into atoms.
    pub address_order: Vec<u32>,
    /// Joint components and the sentinel (empty without destination
    /// writes).
    pub components: Vec<TreeTransformJointComponent>,
    /// Direct items grouped by component; within one, inactive layouts and
    /// Singles ordered by atom slot.
    ///
    /// Why indices rather than copied descriptors: the layout entries,
    /// coefficient and destination range are all read from the block and its
    /// compiled role, so a per-item copy would duplicate retained state.
    pub direct: Vec<TreeTransformDirectItem>,
    /// Multi group indices grouped by component, in recoupling-plan order.
    pub component_groups: Vec<u32>,
    pub pack_columns: Vec<TreeTransformPackReplay>,
    pub scatter_columns: Vec<TreeTransformScatterReplay>,
    pub scatter_groups: Vec<TreeTransformScatterGroupReplay>,
    pub single_block_count: usize,
    pub packed_column_count: usize,
    pub scattered_column_count: usize,
    /// The one-slot visit of a whole call, in source block order (the order
    /// of TensorKit's `add_transform_kernel!` and the per-block loop):
    /// inactive items first, then each Single of a group-free component at
    /// its block, and each component with Multi groups once, whole, at its
    /// first block. Valid because one slot writes everything and the items
    /// of group-free components write disjoint elements.
    pub one_slot_order: Vec<OneSlotStep>,
    /// The multi-slot part, built at most once ([`Self::lanes`]).
    pub lanes: std::sync::OnceLock<TreeTransformLaneSchedule>,
}

impl TreeTransformParallelSchedule {
    /// Joint components.
    #[inline]
    pub fn component_count(&self) -> usize {
        self.components.len().saturating_sub(1)
    }

    /// Atom slots of component `component`.
    #[inline]
    pub fn component_atoms(&self, component: usize) -> Range<usize> {
        self.components[component].atom_start as usize
            ..self.components[component + 1].atom_start as usize
    }

    /// Direct items of component `component`.
    #[inline]
    pub fn component_direct(&self, component: usize) -> &[TreeTransformDirectItem] {
        &self.direct[self.components[component].direct_start as usize
            ..self.components[component + 1].direct_start as usize]
    }

    /// Multi groups of component `component`, in recoupling-plan order.
    #[inline]
    pub fn component_group_list(&self, component: usize) -> &[u32] {
        &self.component_groups[self.components[component].group_start as usize
            ..self.components[component + 1].group_start as usize]
    }

    /// The multi-slot part ([`TreeTransformLaneSchedule`]), built from the
    /// retained layout table on the first call and kept.
    ///
    /// An atom is bandable when every item's baked layout has the same
    /// outermost (largest destination stride) dimension, of stride `S > 0`
    /// and extent `E >= 2`, nonnegative destination strides, a rank within
    /// the inline traversal limit, and an inner reach that fits one slab:
    /// `o - lo + Σ_{j < outer} (dim_j - 1) stride_j < S` for item offset `o`
    /// and atom start `lo`. Then band `k` holds exactly the item elements of
    /// outer index `k`, at `[lo + k S, lo + (k + 1) S)` cut at the atom's
    /// end, for every item alike. Every destination role of the atom takes
    /// part: a Single's and a scatter column's two-sided role, and an
    /// inactive layout's one-sided role (which plans nothing).
    ///
    /// Cost of the build: O(I r + K log K) for I items of fused rank at most
    /// r and the class keys' total length K = Σ_c (1 + g_c), with O(atoms)
    /// temporary band state; every retained vector is allocated at its
    /// exact length ([`Self::lane_bytes`]). Two threads that race to build
    /// it compute the same value and one is kept.
    pub(crate) fn lanes(
        &self,
        blocks: &[TreeTransformBlock],
        layouts: &TreeTransformLayoutTable,
        recoupling_plan: &TreeTransformRecouplingPlan,
    ) -> Result<&TreeTransformLaneSchedule, OperationError> {
        if let Some(lanes) = self.lanes.get() {
            return Ok(lanes);
        }
        let built = self.build_lanes(blocks, layouts, recoupling_plan)?;
        Ok(self.lanes.get_or_init(|| built))
    }

    /// The bytes [`Self::lanes`] retains once built: a function of the atom,
    /// component and group counts, so a retaining cache charges it at
    /// admission.
    pub(crate) fn lane_bytes(&self) -> usize {
        let components = self.component_count();
        self.atoms
            .len()
            .saturating_mul(core::mem::size_of::<TreeTransformAtomBand>())
            .saturating_add(
                components.saturating_mul(core::mem::size_of::<TreeTransformLaneComponent>()),
            )
            .saturating_add(
                self.scatter_groups
                    .len()
                    .saturating_mul(core::mem::size_of::<usize>()),
            )
            .saturating_add(components.saturating_mul(core::mem::size_of::<u32>()))
    }

    fn build_lanes(
        &self,
        blocks: &[TreeTransformBlock],
        layouts: &TreeTransformLayoutTable,
        recoupling_plan: &TreeTransformRecouplingPlan,
    ) -> Result<TreeTransformLaneSchedule, OperationError> {
        // Per atom `None` until its first item, then `Some(Some((S, E)))`
        // while every item agrees on one band, `Some(None)` once one does not.
        let mut states: Vec<Option<Option<(usize, usize)>>> = vec![None; self.atoms.len()];
        let atoms = &self.atoms;
        let mut record = |slot: u32, entry: usize, dims: &[usize], strides: &[isize]| {
            let offset = layouts.entry(entry).offset;
            let band = item_band(atoms[slot as usize].lo, offset, dims, strides);
            let state = &mut states[slot as usize];
            *state = Some(match *state {
                None => band,
                Some(previous) => previous.filter(|&previous| Some(previous) == band),
            });
        };
        let planned = |role: &BakedFusedLayout<'_>| {
            if role.plan.is_some() {
                role.dims.iter().product()
            } else {
                0
            }
        };
        let component_count = self.component_count();
        let mut components = Vec::with_capacity(component_count);
        for component in 0..component_count {
            let mut widest = 0usize;
            let mut planned_single = false;
            let direct = self.component_direct(component);
            for &item in direct {
                match item {
                    TreeTransformDirectItem::Single { block, slot } => {
                        let TreeTransformBlock::Single { dst_layout, .. } = blocks[block as usize]
                        else {
                            return Err(OperationError::InvalidArgument {
                                message: "tree transform direct item names a non-Single block",
                            });
                        };
                        let role = layouts.role(dst_layout)?;
                        record(slot, dst_layout, role.dims, role.dst_strides);
                        widest = widest.max(planned(&role));
                        planned_single = role.plan.is_some();
                    }
                    TreeTransformDirectItem::Inactive { layout, slot } => {
                        let (dims, strides) = layouts.inactive_role(layout as usize)?;
                        record(slot, layout as usize, dims, strides);
                    }
                }
            }
            components.push(TreeTransformLaneComponent {
                widest_direct: widest,
                units: 0,
                planned_single: planned_single
                    && direct.len() == 1
                    && self.component_group_list(component).is_empty(),
            });
        }
        let mut group_widest = Vec::with_capacity(self.scatter_groups.len());
        for group in &self.scatter_groups {
            let mut widest = 0usize;
            for column in &self.scatter_columns[group.columns.clone()] {
                let role = layouts.role(column.dst_layout)?;
                record(column.slot, column.dst_layout, role.dims, role.dst_strides);
                widest = widest.max(planned(&role));
            }
            group_widest.push(widest);
        }
        // Why `iter` rather than `into_iter`: an in-place collect would keep
        // the larger state allocation and exceed the charged size.
        let bands = states
            .iter()
            .map(|&state| {
                let (stride, extent) = state.flatten().unwrap_or((0, 1));
                TreeTransformAtomBand { stride, extent }
            })
            .collect::<Vec<_>>();
        let mut band_units = 0usize;
        for (component, lane) in components.iter_mut().enumerate() {
            lane.units = bands[self.component_atoms(component)]
                .iter()
                .try_fold(0usize, |units, band| units.checked_add(band.extent))
                .ok_or(OperationError::ElementCountOverflow)?;
            band_units = band_units
                .checked_add(lane.units)
                .ok_or(OperationError::ElementCountOverflow)?;
        }
        let (component_class, class_count) = self.cost_classes(&components, recoupling_plan);
        Ok(TreeTransformLaneSchedule {
            bands,
            components,
            group_widest,
            band_units,
            component_class,
            class_count,
        })
    }

    /// Numbers the components' cost classes: one class per distinct
    /// `(planned_single, moved, units, sorted Multi job shapes)`. These fix
    /// a component's modeled time at every width (see `lanes.rs`), so a
    /// placement evaluates each class once. O(K log K) for K = Σ_c (1 + g_c),
    /// each key built once.
    fn cost_classes(
        &self,
        lanes: &[TreeTransformLaneComponent],
        recoupling_plan: &TreeTransformRecouplingPlan,
    ) -> (Vec<u32>, usize) {
        let component_count = lanes.len();
        let jobs = recoupling_plan.jobs();
        let mut shapes = self
            .component_groups
            .iter()
            .map(|&group| {
                let job = &jobs[group as usize];
                (job.rows, job.contracted, job.cols)
            })
            .collect::<Vec<_>>();
        for component in 0..component_count {
            shapes[self.component_groups_range(component)].sort_unstable();
        }
        // Each key built once, then sorted with its component.
        let mut keys = (0..component_count)
            .map(|component| {
                let lane = &lanes[component];
                (
                    (
                        lane.planned_single,
                        self.components[component].moved,
                        lane.units,
                        &shapes[self.component_groups_range(component)],
                    ),
                    component,
                )
            })
            .collect::<Vec<_>>();
        keys.sort_unstable();
        let mut classes = vec![0u32; component_count];
        let mut count = 0usize;
        for (position, (key, component)) in keys.iter().enumerate() {
            if position > 0 && keys[position - 1].0 != *key {
                count += 1;
            }
            classes[*component] = count as u32;
        }
        (classes, if component_count == 0 { 0 } else { count + 1 })
    }

    /// The range of component `component`'s entries in `component_groups`.
    #[inline]
    fn component_groups_range(&self, component: usize) -> Range<usize> {
        self.components[component].group_start as usize
            ..self.components[component + 1].group_start as usize
    }
}

/// The band `(S, E)` of one item with baked layout `(dims, strides)` at
/// offset `offset` in the atom starting at `lo`, or `None` when the item
/// cannot be banded (see [`TreeTransformParallelSchedule::lanes`]).
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

/// The joint ownership schedule: merges the inclusive destination intervals
/// of every Single block, Multi scatter column and inactive layout into
/// address-ordered atoms (while `lo <= atom_hi`), unions the owners (Single,
/// Multi group, inactive layout) of each atom, and numbers the components by
/// their first atom. O(I log I + B) time and O(I) retained metadata for I
/// intervals and B blocks: one sort, one sweep, a union-find and counting
/// passes, no element enumeration. Bands, units, planned sizes and classes
/// follow from the retained layout table
/// ([`TreeTransformParallelSchedule::lanes`]).
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
    let groups = recoupling_plan.jobs().len();
    let mut packed_column_count = 0usize;
    let mut scattered_column_count = 0usize;

    // Every count is known from the plan and the block list, so no table
    // grows while it is built.
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
    let mut scatter_columns = Vec::with_capacity(scattered_total);
    let mut scatter_groups = Vec::with_capacity(groups);
    // Every destination interval: `(lo, hi, owner, item)`. Owners are the
    // groups `0..G`, then the Singles, then the inactive layouts; `item`
    // indexes `scatter_columns` for a group's interval and is the block or
    // layout otherwise.
    let mut intervals: Vec<(isize, isize, u32, u32)> = Vec::with_capacity(
        scattered_total
            .saturating_add(single_block_count)
            .saturating_add(inactive_dst_layouts.len()),
    );
    for (group, (block_index, job)) in recoupling_plan.entries().enumerate() {
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
            let Some((lo, hi)) = layout_index_range(layouts, dst_layout)? else {
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
            intervals.push((
                lo,
                hi,
                checked_u32(group)?,
                checked_u32(scatter_columns.len())?,
            ));
            scatter_columns.push(TreeTransformScatterReplay {
                dst_layout,
                packed_offset,
                slot: 0,
            });
        }
        scatter_groups.push(TreeTransformScatterGroupReplay {
            columns: group_start..scatter_columns.len(),
            component: 0,
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

    let mut owners = groups;
    for (block_index, block) in blocks.iter().enumerate() {
        let TreeTransformBlock::Single { dst_layout, .. } = *block else {
            continue;
        };
        if let Some((lo, hi)) = layout_index_range(layouts, dst_layout)? {
            intervals.push((lo, hi, checked_u32(owners)?, checked_u32(block_index)?));
            owners += 1;
        }
    }
    let single_owners = owners;
    for &layout in inactive_dst_layouts {
        if let Some((lo, hi)) = layout_index_range(layouts, layout)? {
            intervals.push((lo, hi, checked_u32(owners)?, checked_u32(layout)?));
            owners += 1;
        }
    }
    if intervals.is_empty() {
        return Ok(TreeTransformParallelSchedule {
            pack_columns,
            scatter_columns,
            scatter_groups,
            single_block_count,
            packed_column_count,
            scattered_column_count,
            ..Default::default()
        });
    }
    // Why the order is total: compilation rejects aliased destinations and
    // drops empty ones, so no two intervals start at one element.
    intervals.sort_unstable_by_key(|&(lo, ..)| lo);

    // Sweep: each interval's atom, the atom bounds with one owner, and the
    // union of every owner writing into the same atom.
    let mut parent = (0..checked_u32(owners)?).collect::<Vec<u32>>();
    let mut interval_atom = Vec::with_capacity(intervals.len());
    let mut bounds: Vec<(isize, isize, u32)> = Vec::with_capacity(intervals.len());
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
        let len = usize::try_from(hi - lo)
            .ok()
            .and_then(|span| span.checked_add(1))
            .ok_or(OperationError::ElementCountOverflow)?;
        atoms[slot as usize].lo = lo;
        atoms[slot as usize].len = len;
        atom_slot.push(slot);
    }
    drop(next_slot);
    let address_order = atom_slot.clone();

    // Items: their atom slot and weight, the components' direct items (in
    // address order) and moved elements.
    let mut moved = vec![0usize; components_len];
    let mut direct_weight = vec![0usize; components_len];
    let mut items: Vec<(u32, TreeTransformDirectItem)> = Vec::with_capacity(intervals.len());
    for (&(_, _, owner, item), &atom) in intervals.iter().zip(&interval_atom) {
        let slot = atom_slot[atom as usize];
        let component = atom_component[atom as usize];
        let (layout, direct) = if (owner as usize) < groups {
            let column = &mut scatter_columns[item as usize];
            column.slot = slot;
            (column.dst_layout, None)
        } else if (owner as usize) < single_owners {
            let TreeTransformBlock::Single { dst_layout, .. } = blocks[item as usize] else {
                unreachable!("Single owners name Single blocks");
            };
            (
                dst_layout,
                Some(TreeTransformDirectItem::Single { block: item, slot }),
            )
        } else {
            (
                item as usize,
                Some(TreeTransformDirectItem::Inactive { layout: item, slot }),
            )
        };
        let weight = layouts.entry(layout).element_count;
        let atom = &mut atoms[slot as usize];
        atom.weight = atom
            .weight
            .checked_add(weight)
            .ok_or(OperationError::ElementCountOverflow)?;
        moved[component as usize] = moved[component as usize]
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
    // component, address order is slot order, which replay's
    // `partition_point` by slot needs. O(I), no second sort.
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

    // Groups: their component, their columns in atom order, and each
    // component's groups in recoupling-plan order.
    let mut group_counts = vec![0u32; components_len + 1];
    for (group, replay) in scatter_groups.iter_mut().enumerate() {
        if replay.columns.is_empty() {
            continue;
        }
        let component = component_of_root[find(&mut parent, checked_u32(group)?) as usize] as usize;
        replay.component = checked_u32(component)?;
        group_counts[component + 1] += 1;
        scatter_columns[replay.columns.clone()].sort_by_key(|column| column.slot);
        let job = &recoupling_plan.jobs()[group];
        moved[component] = job
            .rows
            .checked_mul(job.contracted)
            .and_then(|packed| moved[component].checked_add(packed))
            .ok_or(OperationError::ElementCountOverflow)?;
    }
    for component in 0..components_len {
        group_counts[component + 1] += group_counts[component];
    }
    let mut component_groups = vec![0u32; group_counts[components_len] as usize];
    let mut next_group = group_counts.clone();
    for (group, replay) in scatter_groups.iter().enumerate() {
        if replay.columns.is_empty() {
            continue;
        }
        let at = &mut next_group[replay.component as usize];
        component_groups[*at as usize] = checked_u32(group)?;
        *at += 1;
    }

    let one_slot_order = one_slot_order(
        blocks.len(),
        &direct,
        &direct_starts,
        &group_counts,
        &component_groups,
        recoupling_plan.block_indices(),
    )?;
    let components = (0..=components_len)
        .map(|component| TreeTransformJointComponent {
            atom_start: atom_starts[component],
            direct_start: direct_starts[component],
            group_start: group_counts[component],
            moved: moved.get(component).copied().unwrap_or(0),
            direct_weight: direct_weight.get(component).copied().unwrap_or(0),
        })
        .collect();
    Ok(TreeTransformParallelSchedule {
        atoms,
        address_order,
        components,
        direct,
        component_groups,
        pack_columns,
        scatter_columns,
        scatter_groups,
        single_block_count,
        packed_column_count,
        scattered_column_count,
        one_slot_order,
        lanes: std::sync::OnceLock::new(),
    })
}

/// [`TreeTransformParallelSchedule::one_slot_order`] by one counting pass
/// over block positions: position 0 holds the inactive items of group-free
/// components, position `1 + b` the Single block `b` of a group-free
/// component or the component with Multi groups whose first block (Single
/// or Multi) is `b`. O(I + B) for I items and B blocks; at most I + C steps.
fn one_slot_order(
    block_count: usize,
    direct: &[TreeTransformDirectItem],
    direct_starts: &[u32],
    group_starts: &[u32],
    component_groups: &[u32],
    group_blocks: &[usize],
) -> Result<Vec<OneSlotStep>, OperationError> {
    let components = direct_starts.len().saturating_sub(1);
    let items =
        |component: usize| direct_starts[component] as usize..direct_starts[component + 1] as usize;
    let groups = |component: usize| {
        &component_groups[group_starts[component] as usize..group_starts[component + 1] as usize]
    };
    // Each step's position, then a stable placement by position.
    let mut steps: Vec<(usize, OneSlotStep)> = Vec::with_capacity(direct.len() + components);
    for component in 0..components {
        let groups = groups(component);
        if groups.is_empty() {
            for index in items(component) {
                let position = match direct[index] {
                    TreeTransformDirectItem::Inactive { .. } => 0,
                    TreeTransformDirectItem::Single { block, .. } => 1 + block as usize,
                };
                steps.push((position, OneSlotStep::Direct(checked_u32(index)?)));
            }
            continue;
        }
        let first_single = direct[items(component)]
            .iter()
            .filter_map(|item| match *item {
                TreeTransformDirectItem::Single { block, .. } => Some(block as usize),
                TreeTransformDirectItem::Inactive { .. } => None,
            });
        let first_group = groups.iter().map(|&group| group_blocks[group as usize]);
        let first = first_single.chain(first_group).min().unwrap_or(0);
        steps.push((1 + first, OneSlotStep::Component(checked_u32(component)?)));
    }
    let mut starts = vec![0usize; block_count + 2];
    for &(position, _) in &steps {
        starts[position + 1] += 1;
    }
    for position in 0..=block_count {
        starts[position + 1] += starts[position];
    }
    let mut order = vec![OneSlotStep::Component(0); steps.len()];
    for &(position, step) in &steps {
        order[starts[position]] = step;
        starts[position] += 1;
    }
    Ok(order)
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
