use smallvec::SmallVec;
use strided_kernel::UpdatePlan;

use super::*;

/// One layout entry's compiled replay role in the fused arena.
///
/// `rank == 0` marks an entry without a role of its own: a Single block's
/// source entry, which its destination twin's role already covers. Every
/// other role has rank >= 1, because normalization maps rank 0 to one unit
/// axis. The compact 32-bit fields match QSpace's runtime-rank metadata
/// without making a small-rank execution split.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct RoleSlot {
    start: u32,
    rank: u32,
    /// One plus the index of this role's compiled plan; zero for none.
    plan: u32,
}

/// The retained layout authority of a completed tree transform: per entry its
/// offset and element count, and per replayed `(entry, role)` the compiled
/// normalized loop layout.
///
/// Provenance: TensorKit `cfaa073e`
/// [`treetransformers.jl:AbelianTreeTransformer/GenericTreeTransformer`](https://github.com/Jutho/TensorKit.jl/blob/cfaa073e4d1e3eb2167edcbdc3be9872f41e7d91/src/tensors/treetransformers.jl#L10-L129)
/// retains, per block, the coefficient plus the destination and source
/// `StridedStructure` (size, strides, offset), and its replay
/// ([`indexmanipulations.jl:add_transform_kernel!`](https://github.com/Jutho/TensorKit.jl/blob/cfaa073e4d1e3eb2167edcbdc3be9872f41e7d91/src/tensors/indexmanipulations.jl#L648-L715))
/// re-normalizes them through `StridedView` + `tensoradd!(…, p, …)` on every
/// call. QSpace `d2d3d7da`
/// [`QSpace.hh:QSpace::Permute`](https://bitbucket.org/qspace4u/qspace-v4-pub/src/d2d3d7da6a59a2e8f2cb7dc8f33e7c345af59371/Source/QSpace.hh#lines-2837:2887) recomputes geometry per call and
/// retains no completed transformer, so it has no corresponding payload.
///
/// Rust deviations: the permuted, normalized pair TensorKit derives on every
/// replay is computed once here and is the *only* retained geometry. A Single
/// block keeps one shared `dims` plus a destination and a source stride column
/// (24f bytes for normalized rank f <= N, against TensorKit's two raw
/// structures of 16N each); its source entry keeps no geometry. A Multi
/// member keeps its pack or scatter role. An inactive destination keeps a
/// one-sided role (`dims` and its own strides, 16f), stored after every
/// two-sided role so the source stride column stops at the two-sided prefix.
/// The raw (pre-normalization, source-permuted) table lives only during
/// compilation, where every structural proof runs on it.
///
/// Every role a Single block or a Multi scatter writes, up to Strided's
/// inline fused rank (`RAW_FUSED_RANK_LIMIT`) and with a nonzero Strided copy
/// estimate, also retains Strided's compiled
/// replay of that normalized layout (a fixed inline size per role), so a leaf
/// that a wide token widens replays it without planning. TensorKit has no
/// counterpart: Strided.jl plans each `tensoradd!` on every call. Pack roles
/// keep none,
/// because pack columns stay on the baked walk; roles above the inline rank
/// keep none, because Strided's spilled replay allocates per worker range.
#[derive(Clone, Debug, Default)]
pub struct TreeTransformLayoutTable {
    pub(super) entries: Vec<TreeTransformLayout>,
    pub(super) fused_dims: Vec<usize>,
    pub(super) fused_dst_strides: Vec<isize>,
    /// Source strides of the two-sided roles only; a role starting at or past
    /// its length is one-sided.
    pub(super) fused_src_strides: Vec<isize>,
    /// Largest replayed role rank, one-sided (inactive) roles included: the
    /// traversal scratch compiled replay hands each worker.
    pub(super) max_fused_rank: usize,
    /// Zero source strides of the largest one-sided role: an inactive
    /// role's source is one zero element.
    pub(super) zero_src_strides: Vec<isize>,
    /// Compiled replays of the planned roles, in role order; inline plans
    /// own no heap storage, so the charge is the vector's capacity.
    pub(super) plans: Vec<UpdatePlan>,
    /// The largest element count of a planned role: the range a leaf gate
    /// searches for the smallest paying size.
    pub(super) max_planned_elements: usize,
}

/// Why not compare `plans`: each is a pure function of the role geometry the
/// other fields already compare, and Strided's plan type has no equality.
impl PartialEq for TreeTransformLayoutTable {
    fn eq(&self, other: &Self) -> bool {
        self.entries == other.entries
            && self.fused_dims == other.fused_dims
            && self.fused_dst_strides == other.fused_dst_strides
            && self.fused_src_strides == other.fused_src_strides
            && self.max_fused_rank == other.max_fused_rank
            && self.zero_src_strides == other.zero_src_strides
            && self.plans.len() == other.plans.len()
            && self.max_planned_elements == other.max_planned_elements
    }
}

impl TreeTransformLayoutTable {
    pub fn entry(&self, index: usize) -> &TreeTransformLayout {
        &self.entries[index]
    }

    pub(crate) fn max_fused_rank(&self) -> usize {
        self.max_fused_rank
    }

    pub(crate) fn max_planned_elements(&self) -> usize {
        self.max_planned_elements
    }

    /// `rank` zero source strides for a one-sided (inactive) role of at most
    /// that rank.
    pub(crate) fn zero_src_strides(&self, rank: usize) -> Result<&[isize], OperationError> {
        self.zero_src_strides.get(..rank).ok_or_else(missing_role)
    }

    /// Roles that retain a compiled plan; tests use it to prove a fixture
    /// reaches the planned leaf on every host class.
    #[cfg(test)]
    pub(crate) fn retained_plan_count(&self) -> usize {
        self.plans.len()
    }

    fn slot_range(&self, entry_index: usize) -> Option<Range<usize>> {
        let slot = self.entries.get(entry_index)?.role;
        if slot.rank == 0 {
            return None;
        }
        let start = slot.start as usize;
        Some(start..start + slot.rank as usize)
    }

    /// The compiled two-sided role of `entry_index`: a Single block's
    /// destination entry (dst = block, src = its source block, permuted), a
    /// Multi source entry's pack (dst = packed column, src = block) or a Multi
    /// destination entry's scatter (dst = block, src = packed column). These
    /// are exactly the normalized `(dims, dst_strides, src_strides)` the
    /// production normalizer produced at compile.
    #[inline]
    pub(crate) fn role(&self, entry_index: usize) -> Result<BakedFusedLayout<'_>, OperationError> {
        match self.slot_range(entry_index) {
            Some(range) if range.end <= self.fused_src_strides.len() => {
                let plan = (self.entries[entry_index].role.plan as usize)
                    .checked_sub(1)
                    .and_then(|plan| self.plans.get(plan));
                Ok(BakedFusedLayout::from_compiled_normalized_slices(
                    &self.fused_dims[range.clone()],
                    &self.fused_dst_strides[range.clone()],
                    &self.fused_src_strides[range],
                    plan,
                ))
            }
            _ => Err(missing_role()),
        }
    }

    /// Lowest physical index a Single block's destination writes, from its
    /// compiled role. Normalization only drops extent-1 axes, reorders, and
    /// fuses same-sign spans, so this equals the raw layout's low end, which
    /// compilation already computed without overflow. O(f), read only at a
    /// replay split's cut.
    pub(crate) fn destination_lo(&self, entry_index: usize) -> Result<isize, OperationError> {
        let role = self.role(entry_index)?;
        let mut lo = self.entries[entry_index].offset;
        for (&extent, &stride) in role.dims().iter().zip(role.dst_strides()) {
            if stride < 0 {
                let span = isize::try_from(extent.saturating_sub(1))
                    .ok()
                    .and_then(|extent| extent.checked_mul(stride))
                    .ok_or(OperationError::ElementCountOverflow)?;
                lo = lo
                    .checked_add(span)
                    .ok_or(OperationError::ElementCountOverflow)?;
            }
        }
        Ok(lo)
    }

    /// The compiled one-sided role of an inactive destination entry: the
    /// normalized `(dims, strides)` addressing exactly the block's elements.
    #[inline]
    pub(crate) fn inactive_role(
        &self,
        entry_index: usize,
    ) -> Result<(&[usize], &[isize]), OperationError> {
        match self.slot_range(entry_index) {
            Some(range) if range.start >= self.fused_src_strides.len() => Ok((
                &self.fused_dims[range.clone()],
                &self.fused_dst_strides[range],
            )),
            _ => Err(missing_role()),
        }
    }

    pub(super) fn diagnostics(
        &self,
        blocks: &[TreeTransformBlock],
    ) -> TreeTransformLayoutDiagnostics {
        let mut entries = self
            .entries
            .iter()
            .map(|layout| TreeTransformLayoutDiagnosticEntry {
                role: TreeTransformLayoutRole::InactiveDestination,
                offset: layout.offset,
                element_count: layout.element_count,
                start: 0,
                rank: 0,
                partner_start: None,
            })
            .collect::<Vec<_>>();
        for block in blocks {
            match *block {
                TreeTransformBlock::Single {
                    dst_layout,
                    src_layout,
                    ..
                } => {
                    entries[dst_layout].role = TreeTransformLayoutRole::Single {
                        source_entry: src_layout,
                    };
                    entries[src_layout].role = TreeTransformLayoutRole::SingleSource {
                        destination_entry: dst_layout,
                    };
                }
                TreeTransformBlock::Multi {
                    dst_layout_start,
                    dst_count,
                    src_layout_start,
                    src_count,
                    ..
                } => {
                    for entry in &mut entries[src_layout_start..src_layout_start + src_count] {
                        entry.role = TreeTransformLayoutRole::Pack;
                    }
                    for entry in &mut entries[dst_layout_start..dst_layout_start + dst_count] {
                        entry.role = TreeTransformLayoutRole::Scatter;
                    }
                }
            }
        }
        // A Single source entry reports its destination twin's role seen
        // from the source side.
        let owner = |index: usize, role: TreeTransformLayoutRole| match role {
            TreeTransformLayoutRole::SingleSource { destination_entry } => destination_entry,
            _ => index,
        };
        let (mut total, mut paired) = (0usize, 0usize);
        for (index, entry) in entries.iter().enumerate() {
            let rank = self
                .slot_range(owner(index, entry.role))
                .map_or(0, |range| range.len());
            total += rank;
            if entry.role != TreeTransformLayoutRole::InactiveDestination {
                paired += rank;
            }
        }
        let mut dims = Vec::with_capacity(total);
        let mut block_strides = Vec::with_capacity(total);
        let mut partner_strides = Vec::with_capacity(paired);
        for (index, entry) in entries.iter_mut().enumerate() {
            let range = self.slot_range(owner(index, entry.role)).unwrap_or(0..0);
            let (dst, src) = (
                &self.fused_dst_strides[range.clone()],
                self.fused_src_strides.get(range.clone()),
            );
            let (block_side, partner_side) = match entry.role {
                TreeTransformLayoutRole::InactiveDestination => (dst, None),
                TreeTransformLayoutRole::Pack | TreeTransformLayoutRole::SingleSource { .. } => {
                    (src.unwrap_or_default(), Some(dst))
                }
                TreeTransformLayoutRole::Single { .. } | TreeTransformLayoutRole::Scatter => {
                    (dst, Some(src.unwrap_or_default()))
                }
            };
            entry.start = dims.len();
            entry.rank = range.len();
            dims.extend_from_slice(&self.fused_dims[range]);
            block_strides.extend_from_slice(block_side);
            entry.partner_start = partner_side.map(|partner| {
                let start = partner_strides.len();
                partner_strides.extend_from_slice(partner);
                start
            });
        }
        TreeTransformLayoutDiagnostics {
            entries,
            dims,
            block_strides,
            partner_strides,
        }
    }
}

fn missing_role() -> OperationError {
    OperationError::InvalidArgument {
        message: "tree transform layout entry has no compiled role of this kind",
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TreeTransformLayout {
    pub(super) role: RoleSlot,
    pub offset: isize,
    pub element_count: usize,
}

/// What one layout entry does during replay.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum TreeTransformLayoutRole {
    /// A Single block's destination; `source_entry` is its source.
    Single { source_entry: usize },
    /// A Single block's source, replayed through its destination's role.
    SingleSource { destination_entry: usize },
    /// A Multi block's source member, packed into a workspace column.
    Pack,
    /// A Multi block's destination member, scattered from a workspace column.
    Scatter,
    /// A destination block no spec writes; only scaled or zeroed.
    InactiveDestination,
}

#[derive(Clone, Debug, PartialEq)]
struct TreeTransformLayoutDiagnosticEntry {
    role: TreeTransformLayoutRole,
    offset: isize,
    element_count: usize,
    start: usize,
    rank: usize,
    partner_start: Option<usize>,
}

/// Owned copy of a completed transform's compiled layouts, for diagnostics
/// and tests.
///
/// Geometry is the *compiled normalized* form replay executes: extent-1 axes
/// dropped, axes ordered by the role's destination stride, contiguous runs
/// fused, a zero extent as `[0]` and rank 0 as `[1]`. It addresses exactly the
/// raw block's element set but is not the raw (source-permuted) axis order,
/// which no completed transform retains.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct TreeTransformLayoutDiagnostics {
    entries: Vec<TreeTransformLayoutDiagnosticEntry>,
    dims: Vec<usize>,
    block_strides: Vec<isize>,
    partner_strides: Vec<isize>,
}

/// One entry of [`TreeTransformLayoutDiagnostics`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct TreeTransformLayoutView<'a> {
    pub role: TreeTransformLayoutRole,
    pub offset: isize,
    pub element_count: usize,
    /// Normalized extents of the entry's role.
    pub dims: &'a [usize],
    /// The entry's own block strides along `dims`.
    pub block_strides: &'a [isize],
    /// The paired strides along `dims`: the other block of a Single (its
    /// source, or for a source entry its destination), or the packed column
    /// of a pack or scatter. `None` for an inactive destination.
    pub partner_strides: Option<&'a [isize]>,
}

impl TreeTransformLayoutDiagnostics {
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Entry `index`, in the same indexing as
    /// [`TreeTransformLayoutTable::entry`] and the block descriptors.
    pub fn entry(&self, index: usize) -> Option<TreeTransformLayoutView<'_>> {
        let entry = self.entries.get(index)?;
        let range = entry.start..entry.start + entry.rank;
        Some(TreeTransformLayoutView {
            role: entry.role,
            offset: entry.offset,
            element_count: entry.element_count,
            dims: &self.dims[range.clone()],
            block_strides: &self.block_strides[range],
            partner_strides: entry
                .partner_start
                .map(|start| &self.partner_strides[start..start + entry.rank]),
        })
    }
}

/// Which normalized role compile bakes for one entry.
#[derive(Clone, Copy)]
enum RoleKind {
    /// Single block: dst = destination entry, src = source entry.
    Single { src_entry: usize },
    /// Multi source member: dst = packed column, src = block.
    Pack,
    /// Multi destination member: dst = block, src = packed column.
    Scatter,
    /// Inactive destination: its own strides on both sides.
    Inactive,
}

impl RoleKind {
    fn is_paired(self) -> bool {
        !matches!(self, Self::Inactive)
    }

    /// Whether the role retains a compiled plan: a destination write (Single
    /// or scatter) within Strided's inline fused rank whose copy Strided
    /// estimates above zero time.
    ///
    /// Why not plan a zero-estimate role: Strided's width rule can never
    /// widen it on any token, so its plan would only cost retained bytes and
    /// a width search per replay.
    fn is_planned(self, fused: &FusedLayoutScratch) -> bool {
        matches!(self, Self::Single { .. } | Self::Scatter)
            && fused.dims().len() <= strided_kernel::RAW_FUSED_RANK_LIMIT
            && !strided_kernel::execution::KernelClass::Copy
                .estimate(fused.dims().iter().product())
                .is_zero()
    }
}

/// Compile-time layout table: the retained entries plus the raw
/// (source-permuted) shapes and strides every structural proof reads. Every
/// entry has the structure rank, so entry `i`'s raw geometry starts at
/// `i * rank`. Dropped once [`Self::finish`] has baked the compiled roles.
pub(super) struct RawLayoutTable {
    pub(super) table: TreeTransformLayoutTable,
    rank: usize,
    pub(super) shapes: Vec<usize>,
    pub(super) strides: Vec<isize>,
}

impl RawLayoutTable {
    pub(super) fn new(rank: usize) -> Self {
        Self {
            table: TreeTransformLayoutTable::default(),
            rank,
            shapes: Vec::new(),
            strides: Vec::new(),
        }
    }

    /// Reserves exact capacity for `entries` layouts. Why exact: a retaining
    /// cache charges the entry vector's capacity, and amortized growth left
    /// up to half of it unused.
    pub(super) fn reserve_exact(&mut self, entries: usize) {
        self.table.entries.reserve_exact(entries);
        self.shapes.reserve_exact(entries.saturating_mul(self.rank));
        self.strides
            .reserve_exact(entries.saturating_mul(self.rank));
    }

    pub(super) fn entries(&self) -> &[TreeTransformLayout] {
        &self.table.entries
    }

    pub(super) fn entry(&self, index: usize) -> &TreeTransformLayout {
        &self.table.entries[index]
    }

    pub(super) fn entry_count(&self) -> usize {
        self.table.entries.len()
    }

    pub(super) fn shape(&self, index: usize) -> &[usize] {
        &self.shapes[index * self.rank..(index + 1) * self.rank]
    }

    pub(super) fn strides(&self, index: usize) -> &[isize] {
        &self.strides[index * self.rank..(index + 1) * self.rank]
    }

    /// Appends one layout entry.
    pub(super) fn push_block(
        &mut self,
        shape: &[usize],
        strides: &[usize],
        offset: usize,
    ) -> Result<usize, OperationError> {
        self.push_block_mapped(shape, strides, offset, None)
    }

    pub(super) fn push_block_with_axes(
        &mut self,
        shape: &[usize],
        strides: &[usize],
        offset: usize,
        axes: Option<&[usize]>,
    ) -> Result<usize, OperationError> {
        let Some(axes) = axes else {
            return self.push_block(shape, strides, offset);
        };
        crate::axis::validate_permutation(axes, self.rank)?;
        self.push_block_mapped(shape, strides, offset, Some(axes))
    }

    fn push_block_mapped(
        &mut self,
        shape: &[usize],
        strides: &[usize],
        offset: usize,
        axes: Option<&[usize]>,
    ) -> Result<usize, OperationError> {
        let rank = self.rank;
        if shape.len() != rank {
            return Err(OperationError::RankMismatch {
                expected: rank,
                actual: shape.len(),
            });
        }
        if strides.len() != rank {
            return Err(OperationError::RankMismatch {
                expected: rank,
                actual: strides.len(),
            });
        }
        let axis = |index: usize| axes.map_or(index, |axes| axes[index]);
        let element_count = (0..rank).try_fold(1usize, |count, index| {
            count
                .checked_mul(shape[axis(index)])
                .ok_or(OperationError::ElementCountOverflow)
        })?;
        // The packed column strides a pack or scatter role reads are the
        // column-major strides of this (permuted) shape; validate that they
        // are representable here so baking them cannot overflow.
        let mut packed_stride = 1usize;
        for index in 0..rank {
            isize::try_from(packed_stride).map_err(|_| OperationError::StrideOverflow {
                value: packed_stride,
            })?;
            packed_stride = packed_stride
                .checked_mul(shape[axis(index)])
                .ok_or(OperationError::ElementCountOverflow)?;
        }
        for index in 0..rank {
            let stride = strides[axis(index)];
            isize::try_from(stride)
                .map_err(|_| OperationError::StrideOverflow { value: stride })?;
        }
        let offset = offset_to_isize(offset)?;
        for index in 0..rank {
            let axis = axis(index);
            self.shapes.push(shape[axis]);
            self.strides.push(strides[axis] as isize);
        }
        self.table.entries.push(TreeTransformLayout {
            role: RoleSlot::default(),
            offset,
            element_count,
        });
        Ok(element_count)
    }

    /// Runs the production normalizer once per replayed `(entry, role)`:
    /// every Single block's role at its destination entry, every Multi source
    /// entry's pack and destination entry's scatter, then every inactive
    /// destination's one-sided role. Block order after the replay sort does
    /// not matter: roles are keyed by stable entry index.
    fn for_each_role(
        &self,
        blocks: &[TreeTransformBlock],
        inactive: &[usize],
        scratch: &mut FusedLayoutScratch,
        mut visit: impl FnMut(usize, RoleKind, &FusedLayoutScratch) -> Result<(), OperationError>,
    ) -> Result<(), OperationError> {
        let mut packed = SmallVec::<[isize; 8]>::new();
        let mut normalize = |entry: usize,
                             kind: RoleKind,
                             scratch: &mut FusedLayoutScratch|
         -> Result<(), OperationError> {
            let shape = self.shape(entry);
            let strides = self.strides(entry);
            if matches!(kind, RoleKind::Pack | RoleKind::Scatter) {
                packed.clear();
                let mut stride = 1isize;
                for &extent in shape {
                    packed.push(stride);
                    // Validated by `push_block_mapped`; a zero extent leaves
                    // later strides unused because normalization maps it to [0].
                    stride = stride.saturating_mul(extent as isize);
                }
            }
            match kind {
                RoleKind::Single { src_entry } => {
                    normalize_fused_layout(shape, strides, self.strides(src_entry), scratch)?
                }
                RoleKind::Pack => normalize_fused_layout(shape, &packed, strides, scratch)?,
                RoleKind::Scatter => normalize_fused_layout(shape, strides, &packed, scratch)?,
                RoleKind::Inactive => normalize_fused_layout(shape, strides, strides, scratch)?,
            }
            visit(entry, kind, scratch)
        };
        for block in blocks {
            match *block {
                TreeTransformBlock::Single {
                    dst_layout,
                    src_layout,
                    ..
                } => normalize(
                    dst_layout,
                    RoleKind::Single {
                        src_entry: src_layout,
                    },
                    scratch,
                )?,
                TreeTransformBlock::Multi {
                    dst_layout_start,
                    dst_count,
                    src_layout_start,
                    src_count,
                    ..
                } => {
                    for entry in src_layout_start..src_layout_start + src_count {
                        normalize(entry, RoleKind::Pack, scratch)?;
                    }
                    for entry in dst_layout_start..dst_layout_start + dst_count {
                        normalize(entry, RoleKind::Scatter, scratch)?;
                    }
                }
            }
        }
        for &entry in inactive {
            normalize(entry, RoleKind::Inactive, scratch)?;
        }
        Ok(())
    }

    /// Bakes every replayed role into an exactly sized arena and returns the
    /// retained table; the raw geometry is dropped with `self`.
    ///
    /// Why two normalization passes: the arena is charged by capacity, and
    /// reserving `max(rank, 1)` per role charged 24N bytes even when a role
    /// fuses to f < N axes. Counting first reserves exactly Σf without a
    /// growth reallocation; the extra pass costs O(N²) compares per role at
    /// compile and allocates nothing.
    pub(super) fn finish(
        mut self,
        blocks: &[TreeTransformBlock],
        inactive: &[usize],
    ) -> Result<TreeTransformLayoutTable, OperationError> {
        let mut scratch = FusedLayoutScratch::default();
        let (mut two_sided, mut one_sided, mut planned) = (0usize, 0usize, 0usize);
        let mut one_sided_rank = 0usize;
        self.for_each_role(blocks, inactive, &mut scratch, |_, kind, fused| {
            let total = if kind.is_paired() {
                &mut two_sided
            } else {
                one_sided_rank = one_sided_rank.max(fused.dims().len());
                &mut one_sided
            };
            *total = total.saturating_add(fused.dims().len());
            planned += usize::from(kind.is_planned(fused));
            Ok(())
        })?;
        let all = two_sided.saturating_add(one_sided);
        let mut table = core::mem::take(&mut self.table);
        table.fused_dims.reserve_exact(all);
        table.fused_dst_strides.reserve_exact(all);
        table.fused_src_strides.reserve_exact(two_sided);
        table.plans.reserve_exact(planned);
        table.zero_src_strides = vec![0; one_sided_rank];
        self.for_each_role(blocks, inactive, &mut scratch, |entry, kind, fused| {
            let paired = kind.is_paired();
            if paired {
                BakedFusedLayout::try_from_normalized_slices(
                    fused.dims(),
                    fused.dst_strides(),
                    fused.src_strides(),
                )?;
            }
            let mut plan = 0;
            if kind.is_planned(fused) {
                table.plans.push(
                    UpdatePlan::compile(fused.dims(), fused.dst_strides(), fused.src_strides())
                        .map_err(crate::strided::error)?,
                );
                plan = u32::try_from(table.plans.len())
                    .map_err(|_| OperationError::ElementCountOverflow)?;
                table.max_planned_elements = table
                    .max_planned_elements
                    .max(fused.dims().iter().product());
            }
            let start = u32::try_from(table.fused_dims.len())
                .map_err(|_| OperationError::ElementCountOverflow)?;
            let rank = u32::try_from(fused.dims().len())
                .map_err(|_| OperationError::ElementCountOverflow)?;
            start
                .checked_add(rank)
                .ok_or(OperationError::ElementCountOverflow)?;
            table.fused_dims.extend_from_slice(fused.dims());
            table
                .fused_dst_strides
                .extend_from_slice(fused.dst_strides());
            if paired {
                table
                    .fused_src_strides
                    .extend_from_slice(fused.src_strides());
            }
            table.max_fused_rank = table.max_fused_rank.max(rank as usize);
            table.entries[entry].role = RoleSlot { start, rank, plan };
            Ok(())
        })?;
        debug_assert_eq!(table.fused_src_strides.len(), two_sided);
        debug_assert_eq!(table.fused_dims.len(), all);
        debug_assert_eq!(table.plans.len(), planned);
        Ok(table)
    }
}
