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
}

/// The destination span of one plan group's scatter columns, ordered by
/// destination start: its first and last written element and whether the
/// columns are slice-disjoint (each starts after the previous one ends). A
/// group without columns spans `(0, -1)`, disjoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TreeTransformGroupSpan {
    pub lo: isize,
    pub hi: isize,
    pub slice_disjoint: bool,
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
    /// The non-empty inactive destination layouts, in destination address
    /// order.
    pub inactive: Vec<u32>,
    /// The non-empty Single blocks, by index into the block list, in
    /// destination address order.
    pub singles: Vec<u32>,
    /// Whether the Singles' destinations are slice-disjoint in `singles`
    /// order, so a replay may split them between workers.
    pub singles_slice_disjoint: bool,
    /// Per plan group, its scatter columns' destination span.
    pub group_spans: Vec<TreeTransformGroupSpan>,
    /// The most workers a replay can keep busy on one phase: the Singles
    /// when they are slice-disjoint (else one, if any), or the pack columns,
    /// whichever is more, and at least one.
    pub replay_runnable: usize,
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
        .saturating_add(bytes(self.inactive.capacity(), core::mem::size_of::<u32>()))
        .saturating_add(bytes(self.singles.capacity(), core::mem::size_of::<u32>()))
        .saturating_add(bytes(
            self.group_spans.capacity(),
            core::mem::size_of::<TreeTransformGroupSpan>(),
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
