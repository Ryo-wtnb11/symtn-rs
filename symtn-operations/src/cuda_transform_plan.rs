//! Host-side planning for device tree-transform replay: the region triples a
//! Single block lowers to, the pack → GEMM → scatter lowering of a Multi
//! (recoupling) block, the distinct cuTENSOR plan signatures a structure
//! submits, and the keyed cache that owns one uploaded coefficient vector per
//! structure.
//!
//! Pure metadata and bookkeeping: no device, no backend and no tenferro type
//! appears here, so this module compiles — and its tests run — in a CPU-only
//! build. It is kept out of `cuda_transform` precisely so the rules the device
//! executor depends on are executed by CI, which only *checks* the `cuda`
//! feature (the same split `symtn-dense`'s `cuda_region` makes).

use core::any::TypeId;
use std::sync::Weak;

use symtn_dense::DenseGemmBatchJob;

use crate::task_view::TreeTransformTaskView;
use crate::{OperationError, TreeTransformBlock};

/// A strided, offset region of a flat device buffer, in element units.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DeviceRegionSpec {
    pub(crate) dims: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    pub(crate) offset: usize,
}

/// One block, pack column or scatter column lowered to a device region move:
/// the same `dims` on both sides, the axis permutation carried by the
/// destination strides, and the coefficient the move is scaled by.
///
/// `coefficient` is the index of the block's scalar in the structure's
/// coefficient payload for a Single block, and `None` for a pack or scatter
/// column, which the host copies unscaled and the device therefore submits
/// against the context's shared `1`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DeviceMoveSpec {
    pub(crate) dims: Vec<usize>,
    pub(crate) dst_strides: Vec<usize>,
    pub(crate) src_strides: Vec<usize>,
    pub(crate) dst_offset: usize,
    pub(crate) src_offset: usize,
    pub(crate) coefficient: Option<usize>,
}

/// One Multi block lowered to the host's pack → `Uᵀ` GEMM → scatter sequence.
///
/// `packs` read the source storage into the workspace source column of `job`,
/// `job` multiplies that column block by the transposed recoupling matrix into
/// the workspace destination, and `scatters` write the result columns out.
/// Every offset in `job` addresses the structure-wide workspace the compile-time
/// recoupling plan sized, so the device reuses the plan's own arithmetic
/// instead of re-deriving per-block offsets.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct DeviceRecouplingSpec {
    pub(crate) packs: Vec<DeviceMoveSpec>,
    pub(crate) scatters: Vec<DeviceMoveSpec>,
    pub(crate) job: DenseGemmBatchJob,
    /// Where this block's recoupling matrix starts in the structure's own
    /// coefficient payload.
    ///
    /// The host re-packs the matrices into a dense scratch vector in plan order
    /// and addresses them through `job.rhs_offset`, because its scratch holds
    /// only the converted matrices. The device uploads the *whole* converted
    /// payload — it needs the Single blocks' scalars from it anyway — so the
    /// block's own `coefficient_start` already names its matrix, with its
    /// row-major interior untouched. Re-packing would upload `Σ_b S_b·D_b`
    /// elements a second time for no addressing gain.
    pub(crate) matrix_offset: usize,
}

/// The device lowering of one completed structure: its Single blocks, its Multi
/// blocks, and — for the Overwrite destination mode — its inactive destination
/// layouts.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct DeviceTransformPlan {
    pub(crate) moves: Vec<DeviceMoveSpec>,
    pub(crate) recouplings: Vec<DeviceRecouplingSpec>,
    pub(crate) zeros: Vec<DeviceRegionSpec>,
    /// Workspace elements the packs write and the GEMMs read.
    pub(crate) workspace_source_len: usize,
    /// Workspace elements the GEMMs write and the scatters read.
    pub(crate) workspace_destination_len: usize,
    /// Longest inactive layout, so the zero template is sized once.
    pub(crate) max_zero_len: usize,
}

fn unsupported(message: &'static str) -> OperationError {
    OperationError::UnsupportedDeviceTreeTransform { message }
}

/// Device regions are unsigned: Tenferro's CUDA dot-general rejects a negative
/// view stride, and a baked tree layout never produces one (block strides
/// originate as `usize`). A negative value is therefore a capability boundary,
/// not an expected input.
fn non_negative(values: &[isize], what: &'static str) -> Result<Vec<usize>, OperationError> {
    values
        .iter()
        .map(|value| usize::try_from(*value).map_err(|_| unsupported(what)))
        .collect()
}

fn offset_to_usize(offset: isize) -> Result<usize, OperationError> {
    usize::try_from(offset).map_err(|_| unsupported("device replay requires a non-negative offset"))
}

/// Lowers every Single block of `task` to a device region move and every Multi
/// block to the host's pack → `Uᵀ` → scatter sequence, skipping the
/// zero-extent ones, plus every inactive destination layout to a region the
/// Overwrite destination mode fills with zeros.
///
/// Both modes share one plan: the Accumulate mode simply does not execute the
/// zero fills, exactly as host replay returns before touching inactive layouts
/// when `beta == 1`.
///
/// Every move's `(dims, dst_strides, src_strides)` is the structure's own baked
/// fused layout, which is the canonical stride signature the host normalizer
/// produced at compile time: using it keeps the device's plan-key diversity at
/// the structural minimum instead of one plan per raw block rank. The host
/// bakes a Multi source entry with the packed column as its destination and a
/// Multi destination entry with the packed column as its source
/// (`RawLayoutTable::finish`), so pack and scatter read the same arena the host
/// kernels do rather than a device-local re-derivation.
pub(crate) fn compile_device_plan<C: Copy>(
    task: TreeTransformTaskView<'_, C>,
) -> Result<DeviceTransformPlan, OperationError> {
    let layouts = task.layouts();
    let mut moves = Vec::new();
    for block in task.blocks() {
        let TreeTransformBlock::Single {
            dst_layout,
            src_layout,
            coefficient,
        } = *block
        else {
            continue;
        };
        let baked = layouts.role(dst_layout).map_err(|_| {
            unsupported("device tree transform requires a baked fused layout per Single block")
        })?;
        if baked.dims.contains(&0) {
            continue;
        }
        moves.push(DeviceMoveSpec {
            dims: baked.dims.to_vec(),
            dst_strides: non_negative(
                baked.dst_strides,
                "device replay requires non-negative destination strides",
            )?,
            src_strides: non_negative(
                baked.src_strides,
                "device replay requires non-negative source strides",
            )?,
            dst_offset: offset_to_usize(layouts.entry(dst_layout).offset)?,
            src_offset: offset_to_usize(layouts.entry(src_layout).offset)?,
            coefficient: Some(coefficient),
        });
    }

    let recoupling_plan = task.recoupling_plan();
    let mut recouplings = Vec::with_capacity(recoupling_plan.jobs().len());
    for (block_index, job) in recoupling_plan.entries() {
        let block = task
            .blocks()
            .get(block_index)
            .ok_or_else(|| unsupported("device tree transform recoupling plan is out of range"))?;
        let TreeTransformBlock::Multi {
            dst_layout_start,
            dst_count,
            src_layout_start,
            src_count,
            coefficient_start,
            element_count,
            ..
        } = *block
        else {
            return Err(unsupported(
                "device tree transform recoupling plan names a Single block",
            ));
        };
        // The device reuses the plan's offsets, so a plan that disagrees with
        // the block it names would silently read another block's column.
        if job.rows != element_count || job.contracted != src_count || job.cols != dst_count {
            return Err(unsupported(
                "device tree transform recoupling job disagrees with its block",
            ));
        }
        let coefficient_end = src_count
            .checked_mul(dst_count)
            .and_then(|len| coefficient_start.checked_add(len))
            .ok_or(OperationError::ElementCountOverflow)?;
        if coefficient_end > task.coefficient_len() {
            return Err(OperationError::CoefficientCountMismatch {
                expected: coefficient_end,
                actual: task.coefficient_len(),
            });
        }
        let mut packs = Vec::with_capacity(src_count);
        for src_index in 0..src_count {
            let entry = src_layout_start + src_index;
            let column = element_count
                .checked_mul(src_index)
                .and_then(|start| job.lhs_offset.checked_add(start))
                .ok_or(OperationError::ElementCountOverflow)?;
            if let Some(pack) = column_move(task, entry, column, PackDirection::IntoColumn)? {
                packs.push(pack);
            }
        }
        let mut scatters = Vec::with_capacity(dst_count);
        for dst_index in 0..dst_count {
            let entry = dst_layout_start + dst_index;
            let column = element_count
                .checked_mul(dst_index)
                .and_then(|start| job.dst_offset.checked_add(start))
                .ok_or(OperationError::ElementCountOverflow)?;
            if let Some(scatter) = column_move(task, entry, column, PackDirection::OutOfColumn)? {
                scatters.push(scatter);
            }
        }
        recouplings.push(DeviceRecouplingSpec {
            packs,
            scatters,
            job: *job,
            matrix_offset: coefficient_start,
        });
    }

    let mut zeros = Vec::new();
    let mut max_zero_len = 0usize;
    // Each zero region is the inactive entry's compiled one-sided role: the
    // block's element set with extent-1 axes dropped and contiguous runs
    // fused, so equivalent blocks share one zero-fill plan signature.
    for &layout_index in task.inactive_destination_layouts() {
        let (dims, strides) = layouts.inactive_role(layout_index).map_err(|_| {
            unsupported("device tree transform requires a compiled role per inactive layout")
        })?;
        if dims.contains(&0) {
            continue;
        }
        let count = crate::strided::element_count(dims)?;
        max_zero_len = max_zero_len.max(count);
        zeros.push(DeviceRegionSpec {
            dims: dims.to_vec(),
            strides: non_negative(
                strides,
                "device replay requires non-negative destination strides",
            )?,
            offset: offset_to_usize(layouts.entry(layout_index).offset)?,
        });
    }

    Ok(DeviceTransformPlan {
        moves,
        recouplings,
        zeros,
        workspace_source_len: recoupling_plan.source_len(),
        workspace_destination_len: recoupling_plan.destination_len(),
        max_zero_len,
    })
}

/// Which side of a pack/scatter move the compact workspace column is on.
#[derive(Clone, Copy, Eq, PartialEq)]
enum PackDirection {
    /// Pack: block storage → workspace column.
    IntoColumn,
    /// Scatter: workspace column → block storage.
    OutOfColumn,
}

/// Lowers one Multi layout entry to the region move between its block storage
/// and its compact workspace column at `column`.
///
/// `None` for a zero-extent entry, which addresses nothing and is never
/// submitted — the host's `shape.contains(&0)` rule.
fn column_move<C: Copy>(
    task: TreeTransformTaskView<'_, C>,
    entry: usize,
    column: usize,
    direction: PackDirection,
) -> Result<Option<DeviceMoveSpec>, OperationError> {
    let layouts = task.layouts();
    let baked = layouts.role(entry).map_err(|_| {
        unsupported("device tree transform requires a baked fused layout per recoupling column")
    })?;
    if baked.dims.contains(&0) {
        return Ok(None);
    }
    let dims = baked.dims.to_vec();
    let dst_strides = non_negative(
        baked.dst_strides,
        "device replay requires non-negative destination strides",
    )?;
    let src_strides = non_negative(
        baked.src_strides,
        "device replay requires non-negative source strides",
    )?;
    let block_offset = offset_to_usize(layouts.entry(entry).offset)?;
    let (dst_offset, src_offset) = match direction {
        PackDirection::IntoColumn => (column, block_offset),
        PackDirection::OutOfColumn => (block_offset, column),
    };
    Ok(Some(DeviceMoveSpec {
        dims,
        dst_strides,
        src_strides,
        dst_offset,
        src_offset,
        // Host packs and scatters copy unscaled (`T::one()`); the block's own
        // coefficients are the recoupling matrix the GEMM applies.
        coefficient: None,
    }))
}

/// Identity of device state prepared for one completed structure.
///
/// The `Weak` is the replay core's identity marker, shared by every handle
/// bound to that core, so all Runtimes replaying one process-global
/// transformer reuse one preparation each. An entry whose core has been
/// evicted from the completed-transformer cache and dropped by every holder
/// is unreachable rather than stale: the key can never match a different
/// core, and the dead entry is purged on the next miss.
#[derive(Clone, Debug)]
pub(crate) struct StructureKey {
    pub(crate) structure: Weak<()>,
    pub(crate) scalar: TypeId,
    pub(crate) context: u64,
}

impl StructureKey {
    fn matches(&self, other: &Self) -> bool {
        self.scalar == other.scalar
            && self.context == other.context
            && Weak::ptr_eq(&self.structure, &other.structure)
    }
}

struct CacheEntry<V> {
    key: StructureKey,
    bytes: usize,
    value: V,
}

/// Entries a [`PreparedDeviceCache`] keeps by default, whatever the byte budget
/// allows.
///
/// A byte budget alone bounds the device memory but not the entry count, so a
/// replay alternating thousands of tiny structures would grow the cache without
/// limit. The host completed-transformer cache is byte-bounded only, so this
/// count is a device-side bound of its own. A workload alternating more than 256 transformers on one device
/// re-uploads the excess per replay; for large transformers the byte budget,
/// not this count, is the binding limit.
pub(crate) const DEFAULT_STRUCTURE_CACHE_ENTRIES: usize = 256;

/// A cached value's payload that other entries may hold too, charged once.
pub(crate) trait SharedPayload {
    /// The address of the payload this value holds, if any. Every holder keeps
    /// the payload alive, so two live values report equal addresses exactly
    /// when they hold one payload.
    fn shared_payload(&self) -> Option<*const ()> {
        None
    }
}

/// Small LRU cache of per-structure device state, bounded by retained bytes and
/// by an entry count (see [`DEFAULT_STRUCTURE_CACHE_ENTRIES`]).
///
/// Ownership is singular: the cached values are the only holders of that
/// device state, and dropping the last entry holding a payload frees it. A
/// payload held by several entries ([`SharedPayload`]) is charged to exactly
/// one of them, so the reported bytes are unique device bytes and the budget
/// binds on them; the charge moves to a remaining holder when its entry
/// leaves. Bytes are reported so the owner of the device memory budget can
/// charge them.
pub(crate) struct PreparedDeviceCache<V> {
    entries: Vec<CacheEntry<V>>,
    budget_bytes: usize,
    max_entries: usize,
    bytes: usize,
}

impl<V: SharedPayload> PreparedDeviceCache<V> {
    pub(crate) fn new(budget_bytes: usize, max_entries: usize) -> Self {
        Self {
            entries: Vec::new(),
            budget_bytes,
            max_entries,
            bytes: 0,
        }
    }

    pub(crate) fn retained_bytes(&self) -> usize {
        self.bytes
    }

    pub(crate) fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Where `key` sits, without changing the recency order.
    ///
    /// A replay resolves its entry once through this and then holds the index,
    /// so the key scan happens at most once per replay rather than once per
    /// thing the replay needs from the entry. The scan is bounded by the entry
    /// cap; a map keyed on `Weak::as_ptr` would be sound too — a live `Weak`
    /// pins its allocation, so the address cannot be reused while the entry
    /// exists — and is the shape to reach for if the cap ever grows enough for
    /// the scan to matter.
    pub(crate) fn position(&self, key: &StructureKey) -> Option<usize> {
        self.entries.iter().position(|entry| entry.key.matches(key))
    }

    /// The entry at `index`, promoted to most-recently-used.
    pub(crate) fn touch(&mut self, index: usize) -> Option<&V> {
        if index >= self.entries.len() {
            return None;
        }
        let entry = self.entries.remove(index);
        self.entries.push(entry);
        self.entries.last().map(|entry| &entry.value)
    }

    /// The entry for `key`, promoted to most-recently-used.
    #[cfg(test)]
    pub(crate) fn get(&mut self, key: &StructureKey) -> Option<&V> {
        let index = self.position(key)?;
        self.touch(index)
    }

    /// Drops every entry whose structure no longer exists.
    pub(crate) fn purge_dead(&mut self) {
        let mut index = 0;
        while index < self.entries.len() {
            if self.entries[index].key.structure.strong_count() > 0 {
                index += 1;
            } else {
                self.remove(index);
            }
        }
    }

    /// Removes the entry at `index`. Its charge moves to another holder of
    /// its payload if one remains, and is released otherwise.
    fn remove(&mut self, index: usize) {
        let removed = self.entries.remove(index);
        if removed.bytes == 0 {
            return;
        }
        let payload = removed.value.shared_payload();
        match self
            .entries
            .iter_mut()
            .find(|entry| payload.is_some() && entry.value.shared_payload() == payload)
        {
            Some(holder) => holder.bytes = removed.bytes,
            None => self.bytes = self.bytes.saturating_sub(removed.bytes),
        }
    }

    /// Inserts `value` as most-recently-used, evicting least-recently-used
    /// entries while the cache is over budget. `bytes` is the size of the
    /// value's device state, charged unless another entry already holds the
    /// same payload.
    ///
    /// The new entry is never evicted, even when it alone exceeds the budget:
    /// the budget bounds *reuse*, and refusing to keep the structure currently
    /// being replayed would only force an upload per replay.
    pub(crate) fn insert(&mut self, key: StructureKey, value: V, bytes: usize) {
        self.purge_dead();
        let payload = value.shared_payload();
        let held = payload.is_some()
            && self
                .entries
                .iter()
                .any(|entry| entry.value.shared_payload() == payload);
        let bytes = if held { 0 } else { bytes };
        self.entries.push(CacheEntry { key, bytes, value });
        self.bytes = self.bytes.saturating_add(bytes);
        while (self.bytes > self.budget_bytes || self.entries.len() > self.max_entries)
            && self.entries.len() > 1
        {
            self.remove(0);
        }
    }

    /// Every value prepared on context `context`.
    pub(crate) fn values_on(&self, context: u64) -> impl Iterator<Item = &V> {
        self.entries
            .iter()
            .filter(move |entry| entry.key.context == context)
            .map(|entry| &entry.value)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use symtn_core::{BlockKey, BlockSpec, BlockStructure};

    use super::*;
    use crate::{TreeTransformBlockSpec, TreeTransformStructure};

    impl SharedPayload for usize {}
    impl SharedPayload for &str {}

    fn structure(shapes: Vec<Vec<usize>>) -> Arc<BlockStructure> {
        Arc::new(BlockStructure::packed_column_major(shapes[0].len(), shapes).unwrap())
    }

    #[test]
    fn a_single_block_lowers_to_the_structures_baked_fused_layout() {
        // What: the device move carries the compiled fused triple, not the raw
        // block rank, and the permutation lives in the destination strides.
        let dst_space = structure(vec![vec![2, 3]]);
        let src_space = structure(vec![vec![3, 2]]);
        let compiled = TreeTransformStructure::compile_structures(
            &dst_space,
            &src_space,
            &[TreeTransformBlockSpec::single(0, 0, -1.0_f64).with_source_axes([1, 0])],
        )
        .unwrap();
        let plan = compile_device_plan(compiled.task_view().unwrap()).unwrap();

        assert_eq!(plan.moves.len(), 1);
        let entry = &plan.moves[0];
        assert_eq!(entry.dims.iter().product::<usize>(), 6);
        assert_eq!(entry.dims.len(), entry.dst_strides.len());
        assert_eq!(entry.dims.len(), entry.src_strides.len());
        assert_ne!(entry.dst_strides, entry.src_strides);
        assert_eq!(entry.coefficient, Some(0));
        assert!(plan.zeros.is_empty());
    }

    #[test]
    fn a_multi_block_lowers_to_pack_gemm_and_scatter_over_the_plans_offsets() {
        // What: a recoupling block becomes one pack per source layout, one GEMM
        // whose shape is (element_count, src_count, dst_count), and one scatter
        // per destination layout — with every workspace offset taken from the
        // compile-time recoupling plan, and with no per-column coefficient.
        let dst_space = structure(vec![vec![2, 3], vec![2, 3]]);
        let src_space = structure(vec![vec![3, 2], vec![3, 2], vec![3, 2]]);
        let compiled = TreeTransformStructure::compile_structures(
            &dst_space,
            &src_space,
            &[TreeTransformBlockSpec::multi(
                vec![0, 1],
                vec![0, 1, 2],
                vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
            )
            .with_source_axes([1, 0])],
        )
        .unwrap();
        let plan = compile_device_plan(compiled.task_view().unwrap()).unwrap();

        assert!(plan.moves.is_empty(), "no Single block");
        assert_eq!(plan.recouplings.len(), 1);
        let entry = &plan.recouplings[0];
        assert_eq!(entry.packs.len(), 3);
        assert_eq!(entry.scatters.len(), 2);
        assert_eq!(entry.job.rows, 6, "element count");
        assert_eq!(entry.job.contracted, 3, "source count");
        assert_eq!(entry.job.cols, 2, "destination count");
        for column in entry.packs.iter().chain(&entry.scatters) {
            assert_eq!(column.coefficient, None, "columns copy unscaled");
            assert_eq!(column.dims.iter().product::<usize>(), 6);
        }
        // Pack i writes column i of the job's source; scatter j reads column j
        // of its destination.
        let packed: Vec<usize> = entry.packs.iter().map(|pack| pack.dst_offset).collect();
        assert_eq!(packed, vec![0, 6, 12]);
        let scattered: Vec<usize> = entry
            .scatters
            .iter()
            .map(|scatter| scatter.src_offset)
            .collect();
        assert_eq!(scattered, vec![0, 6]);
        assert_eq!(plan.workspace_source_len, 18);
        assert_eq!(plan.workspace_destination_len, 12);
        // No device re-pack: the block's matrix is addressed at its own
        // `coefficient_start` in the structure's payload.
        assert_eq!(entry.matrix_offset, 0);
    }

    #[test]
    fn several_multi_blocks_keep_the_recoupling_plans_order_and_offsets() {
        // What: the plan sorts jobs by element count, so its order is not the
        // block order; the device must address each block's columns through the
        // plan's own offsets, and the coefficient ranges must follow the same
        // order so a job's `rhs_offset` names its own matrix.
        let space = structure(vec![
            vec![2, 3],
            vec![2, 3],
            vec![2, 2],
            vec![2, 2],
            vec![2, 2],
        ]);
        let compiled = TreeTransformStructure::compile_structures(
            &space,
            &space,
            &[
                // Declared first, but 6 elements wide, so the plan runs it second.
                TreeTransformBlockSpec::multi(vec![0, 1], vec![0, 1], vec![1.0_f64, 2.0, 3.0, 4.0]),
                TreeTransformBlockSpec::multi(vec![2, 3], vec![2, 3], vec![5.0_f64, 6.0, 7.0, 8.0]),
                TreeTransformBlockSpec::single(4, 4, -1.0_f64),
            ],
        )
        .unwrap();
        let plan = compile_device_plan(compiled.task_view().unwrap()).unwrap();

        assert_eq!(plan.moves.len(), 1, "the Single block is still lowered");
        let shapes: Vec<usize> = plan
            .recouplings
            .iter()
            .map(|entry| entry.job.rows)
            .collect();
        assert_eq!(shapes, vec![4, 6], "sorted by element count");
        // Disjoint, contiguous workspace columns in that same order.
        assert_eq!(plan.recouplings[0].job.lhs_offset, 0);
        assert_eq!(plan.recouplings[1].job.lhs_offset, 8);
        assert_eq!(plan.workspace_source_len, 20);
        assert_eq!(plan.workspace_destination_len, 20);
        // The 4-element block was declared second, so its matrix is the second
        // run of the payload although it is the first job of the plan: the
        // device addresses it where the structure put it, not where the host's
        // re-packed scratch would. The payload holds the one Single scalar
        // first, so the matrices start at 1.
        let matrices: Vec<usize> = plan
            .recouplings
            .iter()
            .map(|entry| entry.matrix_offset)
            .collect();
        assert_eq!(matrices, vec![5, 1]);
        assert_eq!(
            compiled.gathered_coefficients(),
            [-1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
        assert_eq!(plan.recouplings[0].job.rhs_offset, 0);
        assert_eq!(plan.recouplings[1].job.rhs_offset, 4);
    }

    #[test]
    fn the_cache_evicts_the_least_recently_used_beyond_its_entry_cap() {
        // What: the byte budget alone would let an unbounded number of tiny
        // structures accumulate, so the entry cap bounds the cache — and with
        // it the lookup scan — independently of size.
        let structures: Vec<Arc<()>> = (0..DEFAULT_STRUCTURE_CACHE_ENTRIES + 1)
            .map(|_| Arc::new(()))
            .collect();
        let key = |index: usize| StructureKey {
            structure: Arc::downgrade(&structures[index]),
            scalar: TypeId::of::<f64>(),
            context: 3,
        };
        let mut cache = PreparedDeviceCache::new(usize::MAX, DEFAULT_STRUCTURE_CACHE_ENTRIES);
        for index in 0..structures.len() {
            cache.insert(key(index), index, 1);
        }

        assert_eq!(cache.entry_count(), DEFAULT_STRUCTURE_CACHE_ENTRIES);
        assert_eq!(cache.get(&key(0)), None, "the oldest entry was evicted");
        assert_eq!(cache.get(&key(1)), Some(&1));
        assert_eq!(cache.retained_bytes(), DEFAULT_STRUCTURE_CACHE_ENTRIES);
    }

    #[test]
    fn zero_extent_blocks_are_skipped_and_inactive_layouts_are_planned() {
        // What: an untouched destination layout becomes a zero fill sized for
        // the template, and an empty block submits nothing at all.
        let blocks = vec![
            BlockSpec::with_key(BlockKey::ordinal(0), vec![2, 2], vec![1, 2], 0).unwrap(),
            BlockSpec::with_key(BlockKey::ordinal(1), vec![3, 2], vec![1, 3], 4).unwrap(),
            BlockSpec::with_key(BlockKey::ordinal(2), vec![0, 2], vec![1, 1], 10).unwrap(),
        ];
        let space = Arc::new(BlockStructure::from_blocks_with_rank(2, blocks).unwrap());
        let compiled = TreeTransformStructure::compile_structures(
            &space,
            &space,
            &[
                TreeTransformBlockSpec::single(0, 0, 1.0_f64),
                TreeTransformBlockSpec::single(2, 2, 1.0_f64),
            ],
        )
        .unwrap();

        let plan = compile_device_plan(compiled.task_view().unwrap()).unwrap();
        assert_eq!(plan.moves.len(), 1, "the zero-extent block is skipped");
        assert_eq!(plan.zeros.len(), 1, "block 1 is never written");
        // The zero region is the compiled one-sided role: the column-major
        // `[3, 2]` block with strides `[1, 3]` addresses offsets 4..10, the
        // same element set as one fused axis `[6]` of stride 1.
        assert_eq!(plan.zeros[0].dims, vec![6]);
        assert_eq!(plan.zeros[0].strides, vec![1]);
        assert_eq!(plan.zeros[0].offset, 4);
        assert_eq!(plan.max_zero_len, 6);
    }

    #[test]
    fn padded_extent_one_and_rank_zero_inactive_layouts_keep_their_element_sets() {
        // What: a zero region that cannot fuse to one axis keeps the gaps of
        // its padded block — `[2, 1, 3]` with strides `[1, 1, 4]` drops the
        // extent-1 axis and addresses 6, 7, 10, 11, 14, 15 as `[2, 3]` /
        // `[1, 4]` — and a rank-0 block is one element, `[1]` / `[0]`.
        let blocks = vec![
            BlockSpec::with_key(BlockKey::ordinal(0), vec![2, 3, 1], vec![1, 2, 6], 0).unwrap(),
            BlockSpec::with_key(BlockKey::ordinal(1), vec![2, 1, 3], vec![1, 1, 4], 6).unwrap(),
        ];
        let space = Arc::new(BlockStructure::from_blocks_with_rank(3, blocks).unwrap());
        let compiled = TreeTransformStructure::compile_structures(
            &space,
            &space,
            &[TreeTransformBlockSpec::single(0, 0, 1.0_f64)],
        )
        .unwrap();
        let plan = compile_device_plan(compiled.task_view().unwrap()).unwrap();
        assert_eq!(plan.zeros.len(), 1);
        assert_eq!(plan.zeros[0].dims, vec![2, 3]);
        assert_eq!(plan.zeros[0].strides, vec![1, 4]);
        assert_eq!(plan.zeros[0].offset, 6);
        assert_eq!(plan.max_zero_len, 6);

        let scalars = Arc::new(
            BlockStructure::packed_column_major(0, [Vec::<usize>::new(), Vec::new()]).unwrap(),
        );
        let compiled = TreeTransformStructure::compile_structures(
            &scalars,
            &scalars,
            &[TreeTransformBlockSpec::single(1, 0, 2.0_f64)],
        )
        .unwrap();
        let plan = compile_device_plan(compiled.task_view().unwrap()).unwrap();
        assert_eq!(plan.zeros.len(), 1);
        assert_eq!(plan.zeros[0].dims, vec![1]);
        assert_eq!(plan.zeros[0].strides, vec![0]);
        assert_eq!(plan.zeros[0].offset, 0);
        assert_eq!(plan.max_zero_len, 1);
    }

    #[test]
    fn the_cache_keys_on_structure_identity_dtype_and_context() {
        // What: nothing but the exact triple hits, so a second structure, a
        // second dtype or a second context prepares its own device state.
        let live = Arc::new(());
        let other = Arc::new(());
        let key = |structure: &Arc<()>, scalar, context| StructureKey {
            structure: Arc::downgrade(structure),
            scalar,
            context,
        };
        let mut cache = PreparedDeviceCache::new(1024, DEFAULT_STRUCTURE_CACHE_ENTRIES);
        cache.insert(key(&live, TypeId::of::<f64>(), 1), "f64@1", 8);

        assert_eq!(
            cache.get(&key(&live, TypeId::of::<f64>(), 1)),
            Some(&"f64@1")
        );
        assert_eq!(cache.get(&key(&live, TypeId::of::<u32>(), 1)), None);
        assert_eq!(cache.get(&key(&live, TypeId::of::<f64>(), 2)), None);
        assert_eq!(cache.get(&key(&other, TypeId::of::<f64>(), 1)), None);
        assert_eq!(cache.retained_bytes(), 8);
    }

    #[test]
    fn a_dropped_structure_is_purged_and_its_bytes_released() {
        let live = Arc::new(());
        let doomed = Arc::new(());
        let mut cache = PreparedDeviceCache::new(1024, DEFAULT_STRUCTURE_CACHE_ENTRIES);
        cache.insert(
            StructureKey {
                structure: Arc::downgrade(&doomed),
                scalar: TypeId::of::<f64>(),
                context: 1,
            },
            "doomed",
            64,
        );
        cache.insert(
            StructureKey {
                structure: Arc::downgrade(&live),
                scalar: TypeId::of::<f64>(),
                context: 1,
            },
            "live",
            16,
        );
        assert_eq!(cache.retained_bytes(), 80);

        drop(doomed);
        cache.purge_dead();

        assert_eq!(cache.entry_count(), 1);
        assert_eq!(cache.retained_bytes(), 16);
        assert_eq!(
            cache.values_on(1).copied().collect::<Vec<_>>(),
            vec!["live"]
        );
    }

    #[test]
    fn the_budget_evicts_least_recently_used_and_always_keeps_the_newest() {
        let structures: Vec<Arc<()>> = (0..3).map(|_| Arc::new(())).collect();
        let key = |index: usize| StructureKey {
            structure: Arc::downgrade(&structures[index]),
            scalar: TypeId::of::<f64>(),
            context: 7,
        };
        let mut cache = PreparedDeviceCache::new(100, DEFAULT_STRUCTURE_CACHE_ENTRIES);
        cache.insert(key(0), 0, 60);
        cache.insert(key(1), 1, 30);
        // Touching 0 makes 1 the least recently used.
        assert_eq!(cache.get(&key(0)), Some(&0));
        cache.insert(key(2), 2, 30);

        assert_eq!(cache.entry_count(), 2);
        assert_eq!(cache.get(&key(1)), None);
        assert_eq!(cache.get(&key(0)), Some(&0));
        assert_eq!(cache.get(&key(2)), Some(&2));

        // An entry larger than the whole budget still replaces everything and
        // stays, so the structure being replayed is never re-uploaded.
        cache.insert(key(1), 1, 4096);
        assert_eq!(cache.entry_count(), 1);
        assert_eq!(cache.get(&key(1)), Some(&1));
        assert_eq!(cache.retained_bytes(), 4096);
    }

    /// A value holding a shared payload, as a prepared structure holds its
    /// uploaded coefficients.
    struct Holder(Arc<()>);

    impl SharedPayload for Holder {
        fn shared_payload(&self) -> Option<*const ()> {
            Some(Arc::as_ptr(&self.0))
        }
    }

    #[test]
    fn a_shared_payload_is_charged_once_and_released_with_its_last_holder() {
        // What: holders of one payload are separate entries charged
        // once; the charge survives eviction and purge of any holder but the
        // last, whose removal releases it; another payload is charged apart.
        let structures: Vec<Arc<()>> = (0..4).map(|_| Arc::new(())).collect();
        let key = |index: usize| StructureKey {
            structure: Arc::downgrade(&structures[index]),
            scalar: TypeId::of::<f64>(),
            context: 7,
        };
        let shared = Arc::new(());
        let mut cache = PreparedDeviceCache::new(100, 3);
        cache.insert(key(0), Holder(Arc::clone(&shared)), 40);
        cache.insert(key(1), Holder(Arc::clone(&shared)), 40);
        assert_eq!((cache.entry_count(), cache.retained_bytes()), (2, 40));

        // A second 40-byte payload fits a 100-byte budget only because the
        // first is charged once.
        cache.insert(key(2), Holder(Arc::new(())), 40);
        assert_eq!((cache.entry_count(), cache.retained_bytes()), (3, 80));

        // The entry cap evicts holder 0, the charged one: the charge moves.
        cache.insert(key(3), Holder(Arc::new(())), 10);
        assert_eq!(cache.position(&key(0)), None);
        assert_eq!((cache.entry_count(), cache.retained_bytes()), (3, 90));

        // Purging the last holder releases the shared payload.
        let mut structures = structures;
        structures[1] = Arc::new(());
        cache.purge_dead();
        assert_eq!((cache.entry_count(), cache.retained_bytes()), (2, 50));
        assert_eq!(Arc::strong_count(&shared), 1, "no entry holds the payload");
    }
}
