use super::*;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockSpec {
    pub(crate) key: BlockKey,
    pub(crate) shape: DimVec,
    pub(crate) strides: DimVec,
    pub(crate) offset: usize,
}

impl BlockSpec {
    pub fn new(shape: Vec<usize>, strides: Vec<usize>, offset: usize) -> Result<Self, CoreError> {
        Self::with_key(BlockKey::trivial(), shape, strides, offset)
    }

    pub fn with_key(
        key: BlockKey,
        shape: Vec<usize>,
        strides: Vec<usize>,
        offset: usize,
    ) -> Result<Self, CoreError> {
        if shape.len() != strides.len() {
            return Err(CoreError::RankMismatch {
                shape: shape.len(),
                strides: strides.len(),
            });
        }
        storage_end_exclusive(&shape, &strides, offset)?;
        Ok(Self {
            key,
            shape: shape.into_iter().collect(),
            strides: strides.into_iter().collect(),
            offset,
        })
    }

    pub fn column_major(shape: Vec<usize>, offset: usize) -> Result<Self, CoreError> {
        Self::column_major_with_key(BlockKey::trivial(), shape, offset)
    }

    pub fn column_major_with_key(
        key: BlockKey,
        shape: Vec<usize>,
        offset: usize,
    ) -> Result<Self, CoreError> {
        let strides = column_major_strides(&shape)?;
        Self::with_key(key, shape, strides, offset)
    }

    #[inline]
    pub fn key(&self) -> &BlockKey {
        &self.key
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[inline]
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn element_count(&self) -> Result<usize, CoreError> {
        checked_product(&self.shape)
    }

    pub fn storage_end_exclusive(&self) -> Result<usize, CoreError> {
        storage_end_exclusive(&self.shape, &self.strides, self.offset)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SectorBlock {
    key: BlockKey,
}

impl SectorBlock {
    pub fn new(key: BlockKey) -> Self {
        Self { key }
    }

    #[inline]
    pub fn key(&self) -> &BlockKey {
        &self.key
    }
}

/// Indices of fusion-tree pairs sharing one external-sector group.
///
/// Unlike TensorKit's `FusionTreeBlock`, this value does not own a vector of
/// tree pairs; its indices refer back to the parent [`BlockStructure`].
/// `Group` is intentional because symtn keeps one canonical Rust block-storage
/// owner and uses this value only as a grouped execution view.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FusionTreeBlockGroup {
    pub(super) group_key: FusionTreeGroupKey,
    pub(crate) block_indices: DimVec,
}

impl FusionTreeBlockGroup {
    pub fn new(group_key: FusionTreeGroupKey, block_indices: Vec<usize>) -> Self {
        Self {
            group_key,
            block_indices: block_indices.into_iter().collect(),
        }
    }

    fn singleton(group_key: FusionTreeGroupKey, block_index: usize) -> Self {
        let mut block_indices = DimVec::new();
        block_indices.push(block_index);
        Self {
            group_key,
            block_indices,
        }
    }

    #[inline]
    pub fn group_key(&self) -> &FusionTreeGroupKey {
        &self.group_key
    }

    #[inline]
    pub fn block_indices(&self) -> &[usize] {
        &self.block_indices
    }
}

#[derive(Debug)]
pub struct SectorStructure {
    rank: usize,
    key_kind: Option<BlockKeyKind>,
    /// The canonical arena of every fusion-tree key in `blocks` and
    /// `fusion_tree_groups` (single-arena ownership, #2192); `None` for other kinds and when
    /// empty. Its per-side row order is the tree order, so a row index is the
    /// tree's rank on its side.
    pub(crate) trees: Option<Arc<TreeArena>>,
    pub(super) blocks: Vec<SectorBlock>,
    pub(super) fusion_tree_groups: Vec<FusionTreeBlockGroup>,
    pub(super) sorted_indices: DimVec,
    pub(super) compact_lookup: Option<CompactBlockLookup>,
    pub(super) charged_heap_bytes: OnceLock<usize>,
}

impl Clone for SectorStructure {
    fn clone(&self) -> Self {
        Self {
            rank: self.rank,
            key_kind: self.key_kind,
            trees: self.trees.clone(),
            blocks: self.blocks.clone(),
            fusion_tree_groups: self.fusion_tree_groups.clone(),
            sorted_indices: self.sorted_indices.clone(),
            compact_lookup: self.compact_lookup.clone(),
            // Vec cloning may change capacity, so the census cannot be
            // copied even though semantic equality is preserved.
            charged_heap_bytes: OnceLock::new(),
        }
    }
}

impl PartialEq for SectorStructure {
    fn eq(&self, other: &Self) -> bool {
        self.rank == other.rank
            && self.key_kind == other.key_kind
            && self.blocks == other.blocks
            && self.fusion_tree_groups == other.fusion_tree_groups
            && self.sorted_indices == other.sorted_indices
            && self.compact_lookup == other.compact_lookup
    }
}

impl Eq for SectorStructure {}

impl SectorStructure {
    pub fn dense(rank: usize) -> Self {
        Self::from_keys(rank, [BlockKey::trivial()]).expect("dense sector key is unique")
    }

    pub fn empty(rank: usize) -> Self {
        Self {
            rank,
            key_kind: None,
            trees: None,
            blocks: Vec::new(),
            fusion_tree_groups: Vec::new(),
            sorted_indices: DimVec::new(),
            compact_lookup: None,
            charged_heap_bytes: OnceLock::new(),
        }
    }

    pub fn from_keys<I, K>(rank: usize, keys: I) -> Result<Self, CoreError>
    where
        I: IntoIterator<Item = K>,
        K: Into<BlockKey>,
    {
        let mut blocks = Vec::new();
        let mut expected_kind = None;
        for key in keys {
            let key = key.into();
            let actual_kind = key.kind();
            if let Some(expected) = expected_kind {
                if expected != actual_kind {
                    return Err(CoreError::MixedBlockKeyKinds {
                        expected,
                        actual: actual_kind,
                    });
                }
            } else {
                expected_kind = Some(actual_kind);
            }
            blocks.push(SectorBlock::new(key));
        }
        Self::from_checked_blocks(rank, expected_kind, blocks)
    }

    /// [`Self::from_keys`] over the fusion-tree keys of an enumerated
    /// layout, borrowing them. Why a `Result`: the keys are distinct only if
    /// the provider's fusion channels are, which no fusion-rule trait
    /// promises, so a duplicate stays the typed error it is in `from_keys`.
    pub(crate) fn from_fusion_tree_keys(
        rank: usize,
        keys: &[crate::FusionTreePairKey],
    ) -> Result<Self, CoreError> {
        let blocks = keys
            .iter()
            .map(|key| SectorBlock::new(BlockKey::from(key.clone())))
            .collect::<Vec<_>>();
        let kind = blocks.first().map(|block| block.key().kind());
        Self::from_checked_blocks(rank, kind, blocks)
    }

    /// Re-home fusion-tree blocks into this structure's own canonical arena
    /// (single-arena ownership), sort, reject duplicates, and group.
    fn from_checked_blocks(
        rank: usize,
        expected_kind: Option<BlockKeyKind>,
        mut blocks: Vec<SectorBlock>,
    ) -> Result<Self, CoreError> {
        let trees = if expected_kind == Some(BlockKeyKind::FusionTree) {
            Some(rehome_fusion_tree_blocks(&mut blocks))
        } else {
            None
        };
        let sorted_indices = match &trees {
            Some(_) => {
                let mut sorted = (0..blocks.len()).collect::<DimVec>();
                sorted.sort_unstable_by_key(|&index| fusion_tree_rows(&blocks[index]));
                sorted
            }
            None => sorted_block_indices(&blocks),
        };
        for pair in sorted_indices.windows(2) {
            let left = blocks[pair[0]].key();
            let right = blocks[pair[1]].key();
            if left == right {
                return Err(CoreError::DuplicateBlockKey {
                    key: Box::new(left.clone()),
                });
            }
        }
        let mut fusion_tree_groups = Vec::<FusionTreeBlockGroup>::new();
        let mut fusion_tree_group_indices = FxHashMap::<FusionTreeGroupKey, usize>::default();
        for (index, block) in blocks.iter().enumerate() {
            let Some(group_key) = block.key().fusion_tree_group_key() else {
                continue;
            };
            if let Some(&group_index) = fusion_tree_group_indices.get(&group_key) {
                fusion_tree_groups[group_index].block_indices.push(index);
            } else {
                fusion_tree_group_indices.insert(group_key.clone(), fusion_tree_groups.len());
                fusion_tree_groups.push(FusionTreeBlockGroup::singleton(group_key, index));
            }
        }
        let compact_lookup = CompactBlockLookup::from_blocks(&blocks);
        Ok(Self {
            rank,
            key_kind: expected_kind,
            trees,
            blocks,
            fusion_tree_groups,
            sorted_indices,
            compact_lookup,
            charged_heap_bytes: OnceLock::new(),
        })
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.rank
    }

    #[inline]
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Return the homogeneous block-key namespace, or `None` when empty.
    #[inline]
    pub fn key_kind(&self) -> Option<BlockKeyKind> {
        self.key_kind
    }

    #[inline]
    pub fn blocks(&self) -> &[SectorBlock] {
        &self.blocks
    }

    /// Return owned fusion-tree groups in first-appearance storage order.
    ///
    /// Use [`Self::fusion_tree_group_slice`] when the groups do not need to
    /// outlive this structure.
    pub fn fusion_tree_groups(&self) -> Vec<FusionTreeBlockGroup> {
        self.fusion_tree_groups.clone()
    }

    /// Borrow construction-time fusion-tree groups without rebuilding them.
    #[inline]
    pub fn fusion_tree_group_slice(&self) -> &[FusionTreeBlockGroup] {
        &self.fusion_tree_groups
    }

    pub(crate) fn into_fusion_tree_groups(self) -> Vec<FusionTreeBlockGroup> {
        self.fusion_tree_groups
    }

    pub fn block(&self, index: usize) -> Result<&SectorBlock, CoreError> {
        self.blocks
            .get(index)
            .ok_or_else(|| CoreError::BlockIndexOutOfBounds {
                index,
                count: self.blocks.len(),
            })
    }

    pub fn key(&self, index: usize) -> Result<&BlockKey, CoreError> {
        Ok(self.block(index)?.key())
    }

    pub fn find_index(&self, key: &BlockKey) -> Option<usize> {
        if self.key_kind != Some(key.kind()) {
            return None;
        }
        if let BlockKey::FusionTree(pair) = key {
            return self.find_fusion_tree_pair_index(pair);
        }
        if let (Some(lookup), Some(id)) = (&self.compact_lookup, key.compact_id()) {
            if let Some(index) = lookup.get(id) {
                return Some(index);
            }
        }
        self.sorted_indices
            .binary_search_by(|&index| self.blocks[index].key().cmp(key))
            .ok()
            .map(|position| self.sorted_indices[position])
    }

    pub fn find_fusion_tree_pair_index(&self, key: &FusionTreePairKey) -> Option<usize> {
        self.find_tree_rows(key.codomain_tree(), key.domain_tree())
    }

    /// The block whose codomain tree equals `codomain` and whose domain tree
    /// equals `domain`.
    ///
    /// Same-arena fast path (#2192): a handle into this structure's arena
    /// with `codomain` on a codomain row and `domain` on a domain row already
    /// names its side ranks. Any other handle, a swapped-side one included,
    /// is searched per side by content; a miss on either side is a miss.
    fn find_tree_rows(&self, codomain: &FusionTreeKey, domain: &FusionTreeKey) -> Option<usize> {
        if self.key_kind != Some(BlockKeyKind::FusionTree) {
            return None;
        }
        let arena = self.trees.as_ref()?;
        let n_codomain = arena.codomain_rows()?;
        let rows = if Arc::ptr_eq(&codomain.arena, arena)
            && Arc::ptr_eq(&domain.arena, arena)
            && codomain.index < n_codomain
            && n_codomain <= domain.index
        {
            (codomain.index, domain.index)
        } else {
            (
                find_tree_row(arena, 0..n_codomain, codomain.as_view())?,
                find_tree_row(arena, n_codomain..arena.len(), domain.as_view())?,
            )
        };
        self.block_at_tree_rows(rows)
    }

    /// The block whose tree pair equals `pair`, searched per side by
    /// content: a borrowed tree never shares this structure's arena.
    #[doc(hidden)]
    pub fn find_fusion_tree_pair_view_index(&self, pair: FusionTreePairRef<'_>) -> Option<usize> {
        if self.key_kind != Some(BlockKeyKind::FusionTree) {
            return None;
        }
        let arena = self.trees.as_ref()?;
        let n_codomain = arena.codomain_rows()?;
        self.block_at_tree_rows((
            find_tree_row(arena, 0..n_codomain, pair.codomain_tree())?,
            find_tree_row(arena, n_codomain..arena.len(), pair.domain_tree())?,
        ))
    }

    fn block_at_tree_rows(&self, rows: (usize, usize)) -> Option<usize> {
        self.sorted_indices
            .binary_search_by_key(&rows, |&index| fusion_tree_rows(&self.blocks[index]))
            .ok()
            .map(|position| self.sorted_indices[position])
    }

    #[doc(hidden)]
    pub fn find_adjoint_fusion_tree_pair_index(
        &self,
        logical_key: &FusionTreePairKey,
    ) -> Option<usize> {
        self.find_tree_rows(logical_key.domain_tree(), logical_key.codomain_tree())
    }

    #[inline]
    pub fn has_compact_lookup(&self) -> bool {
        self.compact_lookup.is_some()
    }

    #[inline]
    pub fn sorted_indices(&self) -> &[usize] {
        &self.sorted_indices
    }

    pub fn pair_indices_from(&self, src: &Self) -> Result<Vec<usize>, CoreError> {
        if self.block_count() != src.block_count() {
            return Err(CoreError::BlockCountMismatch {
                expected: self.block_count(),
                actual: src.block_count(),
            });
        }
        if self.key_kind != src.key_kind {
            let (Some(expected), Some(actual)) = (self.key_kind, src.key_kind) else {
                unreachable!("equal nonzero block counts cannot mix empty and nonempty structures");
            };
            return Err(CoreError::MixedBlockKeyKinds { expected, actual });
        }
        if let Some(src_lookup) = &src.compact_lookup {
            if self
                .blocks
                .iter()
                .all(|block| block.key().compact_id().is_some())
            {
                return self
                    .blocks
                    .iter()
                    .map(|block| {
                        let id = block.key().compact_id().expect("checked above");
                        src_lookup
                            .get(id)
                            .ok_or_else(|| CoreError::MissingBlockKey {
                                key: Box::new(block.key().clone()),
                            })
                    })
                    .collect();
            }
        }
        self.pair_indices_from_sorted(src)
    }

    fn pair_indices_from_sorted(&self, src: &Self) -> Result<Vec<usize>, CoreError> {
        let mut src_for_dst = vec![usize::MAX; self.block_count()];
        let mut dst_pos = 0usize;
        let mut src_pos = 0usize;
        while dst_pos < self.sorted_indices.len() && src_pos < src.sorted_indices.len() {
            let dst_index = self.sorted_indices[dst_pos];
            let src_index = src.sorted_indices[src_pos];
            let dst_key = self.blocks[dst_index].key();
            let src_key = src.blocks[src_index].key();
            match dst_key.cmp(src_key) {
                std::cmp::Ordering::Less => {
                    return Err(CoreError::MissingBlockKey {
                        key: Box::new(dst_key.clone()),
                    });
                }
                std::cmp::Ordering::Greater => {
                    return Err(CoreError::MissingBlockKey {
                        key: Box::new(src_key.clone()),
                    });
                }
                std::cmp::Ordering::Equal => {
                    src_for_dst[dst_index] = src_index;
                    dst_pos += 1;
                    src_pos += 1;
                }
            }
        }
        if dst_pos < self.sorted_indices.len() {
            let dst_index = self.sorted_indices[dst_pos];
            return Err(CoreError::MissingBlockKey {
                key: Box::new(self.blocks[dst_index].key().clone()),
            });
        }
        if src_pos < src.sorted_indices.len() {
            let src_index = src.sorted_indices[src_pos];
            return Err(CoreError::MissingBlockKey {
                key: Box::new(src.blocks[src_index].key().clone()),
            });
        }
        Ok(src_for_dst)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct CompactBlockLookup {
    pub(super) indices: DimVec,
}

impl CompactBlockLookup {
    const MISSING: usize = usize::MAX;

    fn from_blocks(blocks: &[SectorBlock]) -> Option<Self> {
        if blocks.is_empty() {
            return None;
        }
        let mut max_id = 0usize;
        let mut ids = Vec::with_capacity(blocks.len());
        for block in blocks {
            let id = block.key().compact_id()?;
            max_id = max_id.max(id);
            ids.push(id);
        }
        let len = max_id.checked_add(1)?;
        if len > blocks.len().saturating_mul(4).max(1) {
            return None;
        }
        let mut indices = DimVec::from_elem(Self::MISSING, len);
        for (index, id) in ids.into_iter().enumerate() {
            if indices[id] != Self::MISSING {
                return None;
            }
            indices[id] = index;
        }
        Some(Self { indices })
    }

    fn get(&self, id: usize) -> Option<usize> {
        self.indices
            .get(id)
            .copied()
            .filter(|&index| index != Self::MISSING)
    }
}

/// The `(codomain row, domain row)` of a re-homed fusion-tree block.
#[inline]
fn fusion_tree_rows(block: &SectorBlock) -> (usize, usize) {
    match block.key() {
        BlockKey::FusionTree(pair) => (pair.codomain_tree.index, pair.domain_tree.index),
        BlockKey::Dense | BlockKey::Opaque(_) => (usize::MAX, usize::MAX),
    }
}

/// The row in `rows` of `arena` equal to `tree` (`Ord`, then stored hash:
/// the canonical row order), by binary search.
fn find_tree_row(
    arena: &TreeArena,
    rows: core::ops::Range<usize>,
    tree: FusionTreeRef<'_>,
) -> Option<usize> {
    let (mut low, mut high) = (rows.start, rows.end);
    while low < high {
        let middle = low + (high - low) / 2;
        let row = arena.view(middle);
        match row
            .cmp(&tree)
            .then_with(|| row.stored_hash().cmp(&tree.stored_hash()))
        {
            std::cmp::Ordering::Less => low = middle + 1,
            std::cmp::Ordering::Greater => high = middle,
            std::cmp::Ordering::Equal => return Some(middle),
        }
    }
    None
}

/// The tree pair of a block of a homogeneous fusion-tree structure.
#[inline]
fn fusion_tree_pair_of(block: &SectorBlock) -> &FusionTreePairKey {
    match block.key() {
        BlockKey::FusionTree(pair) => pair,
        BlockKey::Dense | BlockKey::Opaque(_) => {
            unreachable!("a fusion-tree structure holds only fusion-tree keys")
        }
    }
}

/// Point every block at one canonical arena the structure owns: the input
/// arena when the blocks already use all of it in canonical order (the
/// cache-built layouts, one refcount bump), else a canonical copy (5 arena
/// allocations and O(Σ words)).
fn rehome_fusion_tree_blocks(blocks: &mut [SectorBlock]) -> Arc<TreeArena> {
    if let Some(arena) = reusable_tree_arena(blocks) {
        return arena;
    }
    rehome_canonical_with(blocks, fusion_tree_pair_of, |block| match &mut block.key {
        BlockKey::FusionTree(pair) => pair,
        BlockKey::Dense | BlockKey::Opaque(_) => {
            unreachable!("a fusion-tree structure holds only fusion-tree keys")
        }
    })
}

/// The blocks' arena if it is canonical, every block's codomain tree is on
/// a codomain row and its domain tree on a domain row, and every row is
/// used (so a structure over a few keys of a large one never pins it).
fn reusable_tree_arena(blocks: &[SectorBlock]) -> Option<Arc<TreeArena>> {
    let arena = &fusion_tree_pair_of(blocks.first()?).codomain_tree.arena;
    let n_codomain = arena.codomain_rows()?;
    // Each block names one row per side, so fewer blocks than either side's
    // rows cannot use them all.
    if blocks.len() < n_codomain.max(arena.len() - n_codomain) {
        return None;
    }
    // One bit per row; inline up to 256 rows.
    let mut referenced = SmallVec::<[u64; 4]>::from_elem(0, arena.len().div_ceil(64));
    for block in blocks {
        let pair = fusion_tree_pair_of(block);
        let (codomain, domain) = (&pair.codomain_tree, &pair.domain_tree);
        if !Arc::ptr_eq(&codomain.arena, arena)
            || !Arc::ptr_eq(&domain.arena, arena)
            || codomain.index >= n_codomain
            || domain.index < n_codomain
        {
            return None;
        }
        for row in [codomain.index, domain.index] {
            referenced[row / 64] |= 1 << (row % 64);
        }
    }
    let used = referenced
        .iter()
        .map(|word| word.count_ones() as usize)
        .sum::<usize>();
    (used == arena.len()).then(|| Arc::clone(arena))
}

fn sorted_block_indices(blocks: &[SectorBlock]) -> DimVec {
    let mut sorted_indices = (0..blocks.len()).collect::<DimVec>();
    sorted_indices.sort_unstable_by(|&left, &right| blocks[left].key().cmp(blocks[right].key()));
    sorted_indices
}
