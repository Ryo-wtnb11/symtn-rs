use super::*;

#[derive(Default)]
pub(crate) struct BlockStructureRegionState {
    coupled_region_cache: OnceLock<CoupledRegionCache>,
    /// The degeneracy-cache entry that owns this wrapper, charged for each
    /// region memo as it materializes. Unset for wrappers no cache owns.
    /// Set once: admission links only a freshly built wrapper, so a warm
    /// reader takes no lock.
    owner: OnceLock<Weak<crate::fusion_space::DegeneracyStructureEntry>>,
    /// The checked Generic provider identity that admitted every fusion-tree
    /// key of this wrapper's content (#2233). Why it may outlive cache resets:
    /// admission is a pure function of the identity and the keys, whose
    /// answers [`crate::CheckedGenericFusion`] requires to be immutable;
    /// a reset drops only memoized provider state, never those answers.
    checked_generic_admitted: OnceLock<RuleIdentity>,
}

pub struct BlockStructure {
    pub(super) content: Arc<BlockStructureContent>,
    regions: Arc<BlockStructureRegionState>,
}

impl Clone for BlockStructure {
    fn clone(&self) -> Self {
        Self {
            content: Arc::clone(&self.content),
            regions: Arc::clone(&self.regions),
        }
    }
}

impl core::fmt::Debug for BlockStructure {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("BlockStructure")
            .field("sector", &self.content.sector)
            .field("degeneracy", &self.content.degeneracy)
            .field("content", &self.content)
            .field("required_len", &self.content.required_len)
            .finish()
    }
}

impl PartialEq for BlockStructure {
    fn eq(&self, other: &Self) -> bool {
        self.content == other.content
    }
}

impl Eq for BlockStructure {}

impl BlockStructure {
    pub(crate) fn from_content(content: Arc<BlockStructureContent>) -> Self {
        Self {
            content,
            regions: Arc::new(BlockStructureRegionState::default()),
        }
    }
}

/// Fully validated block metadata staged before its final owner is selected.
///
/// Structural ownership follows [TensorKit's sector/degeneracy structure](https://github.com/Jutho/TensorKit.jl/blob/cfaa073e4d1e3eb2167edcbdc3be9872f41e7d91/src/spaces/structure.jl#L114)
/// and [QSpace's grouped reduced-block trace](https://bitbucket.org/qspace4u/qspace-v4-pub/src/d2d3d7da6a59a2e8f2cb7dc8f33e7c345af59371/Source/QSpace.cc#lines-4501:4625).
/// Deferred checked publication is a Rust ownership adaptation: neither
/// reference supplies this reset-aware transaction.
///
/// The borrowed preview is suitable for fallible plan compilation. Call
/// [`Self::commit`] only after all later validation has succeeded.
#[doc(hidden)]
pub struct PreparedBlockStructure {
    state: PreparedBlockStructureState,
    required_len: usize,
    storage_tiling: bool,
}

enum PreparedBlockStructureState {
    Fresh(PreparedBlockStructureCandidate),
    CompleteHit {
        homspace: FusionTreeHomSpace,
        canonical_id: HomSpaceId,
        structure: Arc<BlockStructure>,
    },
    CompleteMiss {
        homspace: FusionTreeHomSpace,
        candidate: PreparedBlockStructureCandidate,
        key: CompleteHomSpaceStructureCacheKey,
        epoch: usize,
    },
}

struct PreparedBlockStructureCandidate {
    sector: Arc<SectorStructure>,
    degeneracy: std::sync::Mutex<Option<DegeneracyStructure>>,
    preview: OnceLock<BlockStructure>,
}

impl PreparedBlockStructure {
    fn from_blocks_with_rank(rank: usize, blocks: Vec<BlockSpec>) -> Result<Self, CoreError> {
        // Error order: each block's own layout, then the keys, then the rank.
        for block in &blocks {
            block.storage_end_exclusive()?;
        }
        let sector =
            SectorStructure::from_keys(rank, blocks.iter().map(|block| block.key().clone()))?;
        let mut degeneracy = DegeneracyStructure::with_block_capacity(rank, blocks.len());
        for block in &blocks {
            degeneracy.push_block(block.shape(), block.strides(), block.offset())?;
        }
        Self::from_parts(sector, degeneracy)
    }

    pub(crate) fn from_parts(
        sector: SectorStructure,
        degeneracy: DegeneracyStructure,
    ) -> Result<Self, CoreError> {
        Self::from_shared_parts(Arc::new(sector), degeneracy)
    }

    pub(crate) fn from_shared_parts(
        sector: Arc<SectorStructure>,
        degeneracy: DegeneracyStructure,
    ) -> Result<Self, CoreError> {
        if sector.rank() != degeneracy.rank() {
            return Err(CoreError::StructureRankMismatch {
                expected: sector.rank(),
                actual: degeneracy.rank(),
            });
        }
        if sector.block_count() != degeneracy.block_count() {
            return Err(CoreError::BlockCountMismatch {
                expected: sector.block_count(),
                actual: degeneracy.block_count(),
            });
        }
        let required_len = degeneracy.required_len()?;
        Ok(Self {
            state: PreparedBlockStructureState::Fresh(PreparedBlockStructureCandidate {
                sector,
                degeneracy: std::sync::Mutex::new(Some(degeneracy)),
                preview: OnceLock::new(),
            }),
            required_len,
            storage_tiling: false,
        })
    }

    pub(crate) fn complete_hit(
        homspace: FusionTreeHomSpace,
        canonical_id: HomSpaceId,
        structure: Arc<BlockStructure>,
    ) -> Self {
        Self {
            required_len: structure.content.required_len,
            storage_tiling: structure.storage_tiling_proven(),
            state: PreparedBlockStructureState::CompleteHit {
                homspace,
                canonical_id,
                structure,
            },
        }
    }

    pub(crate) fn complete_miss(
        homspace: FusionTreeHomSpace,
        prepared: Self,
        key: CompleteHomSpaceStructureCacheKey,
        epoch: usize,
    ) -> Self {
        let Self {
            state: PreparedBlockStructureState::Fresh(candidate),
            required_len,
            storage_tiling,
        } = prepared
        else {
            unreachable!("complete miss must wrap one fresh candidate")
        };
        Self {
            state: PreparedBlockStructureState::CompleteMiss {
                homspace,
                candidate,
                key,
                epoch,
            },
            required_len,
            storage_tiling,
        }
    }

    /// Marks this staged geometry as built by the canonical coupled-sector
    /// builder; the preview and the committed structure then record
    /// [`BlockStructure::storage_tiling_proven`].
    pub(crate) fn with_storage_tiling(mut self) -> Self {
        self.storage_tiling = true;
        self
    }

    /// Borrow the staged structure for validation and plan compilation.
    #[doc(hidden)]
    pub fn structure(&self) -> &BlockStructure {
        match &self.state {
            PreparedBlockStructureState::CompleteHit { structure, .. } => structure,
            PreparedBlockStructureState::Fresh(candidate)
            | PreparedBlockStructureState::CompleteMiss { candidate, .. } => candidate
                .preview
                .get_or_init(|| self.preview_candidate(candidate)),
        }
    }

    /// Share the staged structure with a plan compiled before commit.
    ///
    /// A complete hit returns its live canonical wrapper, so the plan reuses
    /// that wrapper's coupled-region memo. A fresh or missed candidate gets
    /// one operation-local wrapper around the preview, which shares the
    /// preview's region state with the structure published by [`Self::commit`].
    #[doc(hidden)]
    pub fn shared_structure(&self) -> Arc<BlockStructure> {
        match &self.state {
            PreparedBlockStructureState::CompleteHit { structure, .. } => Arc::clone(structure),
            PreparedBlockStructureState::Fresh(_)
            | PreparedBlockStructureState::CompleteMiss { .. } => {
                Arc::new(self.structure().clone())
            }
        }
    }

    #[doc(hidden)]
    pub fn required_len(&self) -> usize {
        self.required_len
    }

    /// Borrow the staged block keys alone. Why not `structure()`: key-only
    /// prevalidation must not allocate the preview's immutable content.
    #[doc(hidden)]
    pub fn sector_structure(&self) -> &SectorStructure {
        match &self.state {
            PreparedBlockStructureState::CompleteHit { structure, .. } => {
                structure.sector_structure()
            }
            PreparedBlockStructureState::Fresh(candidate)
            | PreparedBlockStructureState::CompleteMiss { candidate, .. } => &candidate.sector,
        }
    }

    /// Finish the validated structure, publishing through the complete-space
    /// cache when this preparation carries a complete-space transaction.
    #[doc(hidden)]
    pub fn commit(self) -> BlockStructure {
        let Self {
            state,
            required_len,
            storage_tiling,
        } = self;
        match state {
            PreparedBlockStructureState::Fresh(candidate) => {
                Self::finish_candidate(candidate, required_len, storage_tiling)
            }
            PreparedBlockStructureState::CompleteHit { structure, .. } => {
                structure.as_ref().clone()
            }
            PreparedBlockStructureState::CompleteMiss {
                candidate,
                key,
                epoch,
                ..
            } => {
                let structure =
                    Self::finish_candidate(candidate, required_len, storage_tiling).into_shared();
                admit_complete_hom_space_structure(key, structure, epoch)
                    .1
                    .as_ref()
                    .clone()
            }
        }
    }

    #[doc(hidden)]
    pub fn commit_with_complete_homspace(
        self,
    ) -> (Option<FusionTreeHomSpace>, Arc<BlockStructure>) {
        let Self {
            state,
            required_len,
            storage_tiling,
        } = self;
        match state {
            PreparedBlockStructureState::CompleteHit {
                homspace,
                canonical_id,
                structure,
            } => (Some(homspace.with_canonical_id(canonical_id)), structure),
            PreparedBlockStructureState::CompleteMiss {
                homspace,
                candidate,
                key,
                epoch,
            } => {
                let structure =
                    Self::finish_candidate(candidate, required_len, storage_tiling).into_shared();
                let (entry, structure) = admit_complete_hom_space_structure(key, structure, epoch);
                (
                    Some(homspace.with_canonical_id(entry.homspace_id())),
                    structure,
                )
            }
            PreparedBlockStructureState::Fresh(candidate) => (
                None,
                Self::finish_candidate(candidate, required_len, storage_tiling).into_shared(),
            ),
        }
    }

    fn preview_candidate(&self, candidate: &PreparedBlockStructureCandidate) -> BlockStructure {
        let degeneracy = candidate
            .degeneracy
            .lock()
            .expect("staged degeneracy ownership poisoned")
            .take()
            .expect("staged degeneracy already moved");
        let preview = BlockStructure::from_content(Arc::new(BlockStructureContent::new(
            Arc::clone(&candidate.sector),
            degeneracy,
            self.required_len,
        )));
        if self.storage_tiling {
            preview.record_storage_tiling();
        }
        preview
    }

    fn finish_candidate(
        candidate: PreparedBlockStructureCandidate,
        required_len: usize,
        storage_tiling: bool,
    ) -> BlockStructure {
        let structure = match candidate.preview.into_inner() {
            Some(preview) => preview,
            None => BlockStructure::from_content(Arc::new(BlockStructureContent::new(
                candidate.sector,
                candidate
                    .degeneracy
                    .into_inner()
                    .expect("staged degeneracy ownership poisoned")
                    .expect("staged degeneracy already moved"),
                required_len,
            ))),
        };
        if storage_tiling {
            structure.record_storage_tiling();
        }
        structure
    }
}

impl BlockStructure {
    /// Conservative retained bytes for this structure's immutable layout.
    ///
    /// The fixed region-state allocation is included, but its lazily derived
    /// coupled-region operation cache is not: that shared cache can be grown by
    /// non-workspace owners and is not required to retain or reactivate an idle
    /// destination. Excluding it also keeps this measurement allocation-free
    /// and stable for the lifetime of the structure.
    #[doc(hidden)]
    pub fn charged_retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            .saturating_add(self.content.charged_retained_bytes())
            .saturating_add(std::mem::size_of::<BlockStructureRegionState>())
            .saturating_add(2 * std::mem::size_of::<usize>())
    }

    pub fn trivial(shape: &[usize]) -> Result<Self, CoreError> {
        Self::from_parts(
            SectorStructure::dense(shape.len()),
            DegeneracyStructure::packed_column_major(shape.len(), [shape.to_vec()])?,
        )
    }

    pub fn empty(rank: usize) -> Self {
        let sector = Arc::new(SectorStructure::empty(rank));
        let degeneracy = DegeneracyStructure::empty(rank);
        Self::from_content(new_block_structure_content(sector, degeneracy, 0))
    }

    pub fn from_blocks(blocks: Vec<BlockSpec>) -> Result<Self, CoreError> {
        let rank = blocks.first().map(|block| block.shape().len()).unwrap_or(0);
        Self::from_blocks_with_rank(rank, blocks)
    }

    pub fn from_blocks_with_rank(rank: usize, blocks: Vec<BlockSpec>) -> Result<Self, CoreError> {
        PreparedBlockStructure::from_blocks_with_rank(rank, blocks)
            .map(|prepared| prepared.commit())
    }

    pub fn from_parts(
        sector: SectorStructure,
        degeneracy: DegeneracyStructure,
    ) -> Result<Self, CoreError> {
        PreparedBlockStructure::from_parts(sector, degeneracy).map(|prepared| prepared.commit())
    }

    /// [`Self::from_parts`] over a sector structure shared with other
    /// structures of the same sector layout.
    pub(crate) fn from_shared_parts(
        sector: Arc<SectorStructure>,
        degeneracy: DegeneracyStructure,
    ) -> Result<Self, CoreError> {
        PreparedBlockStructure::from_shared_parts(sector, degeneracy)
            .map(|prepared| prepared.commit())
    }

    pub fn into_shared(self) -> Arc<Self> {
        Arc::new(self)
    }

    pub fn canonicalize_shared(structure: Arc<Self>) -> Arc<Self> {
        structure
    }

    pub fn packed_column_major<I>(rank: usize, shapes: I) -> Result<Self, CoreError>
    where
        I: IntoIterator,
        I::Item: Into<Vec<usize>>,
    {
        let shapes = shapes
            .into_iter()
            .map(Into::into)
            .collect::<Vec<Vec<usize>>>();
        let sector = SectorStructure::from_keys(rank, (0..shapes.len()).map(BlockKey::ordinal))?;
        let degeneracy = DegeneracyStructure::packed_column_major(rank, shapes)?;
        Self::from_parts(sector, degeneracy)
    }

    /// Coupled-sector matrix layout over fusion-tree block keys.
    ///
    /// Every key is first validated against `rule` in caller order. Blocks are
    /// then stable-sorted by coupled sector, and each coupled sector is laid out
    /// as one contiguous column-major matrix with the fusion-tree subblocks as strided views (see
    /// [`FusionTreeHomSpace::coupled_subblock_structure`]). Fails when a
    /// coupled sector does not cover its full codomain-tree x domain-tree
    /// grid, because the sector matrix would contain uninitialized holes.
    ///
    /// Each key inherits [`FusionTreePairKey::validate_for_rule`]'s
    /// provider-domain precondition: arbitrary numeric sector IDs are not
    /// checked, and an infallible finite provider may panic on such IDs.
    pub fn coupled_sector_matrix_with_keys<R>(
        rule: &R,
        nout: usize,
        rank: usize,
        blocks: Vec<(FusionTreePairKey, Vec<usize>)>,
    ) -> Result<Self, CoreError>
    where
        R: FusionRule,
    {
        // Why not retain an otherwise-dead provider argument: this public
        // constructor is a categorical admission boundary, so the provider
        // validates every key in caller order before layout arithmetic begins.
        for (key, _) in &blocks {
            key.validate_for_rule(rule)?;
        }
        coupled_sector_matrix_from_validated_keys(nout, rank, blocks)
    }

    /// Checked finite-algebra sibling of [`Self::coupled_sector_matrix_with_keys`].
    pub fn coupled_sector_matrix_with_keys_checked<R>(
        rule: &R,
        nout: usize,
        rank: usize,
        blocks: Vec<(FusionTreePairKey, Vec<usize>)>,
    ) -> Result<Self, CheckedFusionSpaceError>
    where
        R: CheckedFusionAlgebra,
    {
        validate_coupled_sector_matrix_dimensions(
            nout,
            rank,
            blocks.iter().map(|(_, shape)| shape),
        )?;
        for (key, _) in &blocks {
            ShapeValidatedFusionTree::try_new(key.codomain_tree())?;
            ShapeValidatedFusionTree::try_new(key.domain_tree())?;
            let actual_nout = key.codomain_tree().uncoupled().len();
            let actual_nin = key.domain_tree().uncoupled().len();
            if actual_nout != nout || actual_nin != rank - nout {
                return Err(CoreError::FusionSpaceSplitMismatch {
                    expected_nout: nout,
                    expected_nin: rank - nout,
                    actual_nout,
                    actual_nin,
                }
                .into());
            }
            validate_fusion_tree_pair_coupled(key.codomain_tree(), key.domain_tree())?;
        }
        let prepared = {
            let mut order = (0..blocks.len()).collect::<Vec<_>>();
            order.sort_by_key(|&index| blocks[index].0.codomain_tree().coupled().id());
            let keys = order
                .iter()
                .map(|&index| &blocks[index].0)
                .collect::<Vec<_>>();
            let shapes = order
                .iter()
                .map(|&index| blocks[index].1.as_slice())
                .collect::<Vec<_>>();
            let specs = coupled_sector_matrix_block_specs_after_dimension_validation(
                nout, rank, &keys, &shapes,
            )?;
            PreparedBlockStructure::from_blocks_with_rank(rank, specs)?
        };
        for (key, _) in &blocks {
            validate_fusion_tree_for_rule_checked_after_shape(rule, key.codomain_tree())?;
            validate_fusion_tree_for_rule_checked_after_shape(rule, key.domain_tree())?;
        }
        Ok(prepared.commit())
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.content.sector.rank()
    }

    #[inline]
    pub fn block_count(&self) -> usize {
        self.content.sector.block_count()
    }

    #[inline]
    pub fn sector_structure(&self) -> &SectorStructure {
        &self.content.sector
    }

    #[inline]
    pub fn degeneracy_structure(&self) -> &DegeneracyStructure {
        &self.content.degeneracy
    }

    #[inline]
    pub fn content_id(&self) -> usize {
        self.content.id()
    }

    #[inline]
    pub fn content_key(&self) -> Arc<BlockStructureContent> {
        Arc::clone(&self.content)
    }

    /// Whether this structure's content is, or was, the resident value of a
    /// complete-HomSpace cache entry: one atomic load. The process-global
    /// completed-transformer cache publishes only keys whose structures all
    /// satisfy it, so call-local contents never take a bounded slot.
    #[doc(hidden)]
    #[inline]
    pub fn is_canonical(&self) -> bool {
        self.content.is_canonical()
    }

    pub fn fusion_tree_groups(&self) -> Vec<FusionTreeBlockGroup> {
        self.content.sector.fusion_tree_groups()
    }

    /// Borrow construction-time fusion-tree groups without rebuilding them.
    #[inline]
    pub fn fusion_tree_group_slice(&self) -> &[FusionTreeBlockGroup] {
        self.content.sector.fusion_tree_group_slice()
    }

    pub fn find_block_index_by_key(&self, key: &BlockKey) -> Option<usize> {
        self.content.sector.find_index(key)
    }

    pub fn find_block_index_by_fusion_tree_pair(&self, key: &FusionTreePairKey) -> Option<usize> {
        self.content.sector.find_fusion_tree_pair_index(key)
    }

    /// [`Self::find_block_index_by_fusion_tree_pair`] of a borrowed pair.
    #[doc(hidden)]
    pub fn find_block_index_by_fusion_tree_pair_view(
        &self,
        pair: FusionTreePairRef<'_>,
    ) -> Option<usize> {
        self.content.sector.find_fusion_tree_pair_view_index(pair)
    }

    pub fn pair_block_indices_from(&self, src: &BlockStructure) -> Result<Vec<usize>, CoreError> {
        self.content.sector.pair_indices_from(&src.content.sector)
    }

    pub fn only_block(&self) -> Result<BlockRef<'_>, CoreError> {
        if self.block_count() == 1 {
            self.block(0)
        } else {
            Err(CoreError::BlockCountMismatch {
                expected: 1,
                actual: self.block_count(),
            })
        }
    }

    pub fn block(&self, index: usize) -> Result<BlockRef<'_>, CoreError> {
        Ok(BlockRef {
            key: self.content.sector.key(index)?,
            degeneracy: self.content.degeneracy.block(index)?,
        })
    }

    pub fn block_by_key(&self, key: &BlockKey) -> Result<BlockRef<'_>, CoreError> {
        let index =
            self.find_block_index_by_key(key)
                .ok_or_else(|| CoreError::MissingBlockKey {
                    key: Box::new(key.clone()),
                })?;
        self.block(index)
    }

    pub fn fusion_tree_pair_block(
        &self,
        key: &FusionTreePairKey,
    ) -> Result<BlockRef<'_>, CoreError> {
        let index = self
            .find_block_index_by_fusion_tree_pair(key)
            .ok_or_else(|| CoreError::MissingBlockKey {
                key: Box::new(BlockKey::FusionTree(key.clone())),
            })?;
        self.block(index)
    }

    #[doc(hidden)]
    pub fn find_block_index_by_adjoint_fusion_tree_pair(
        &self,
        logical_key: &FusionTreePairKey,
    ) -> Option<usize> {
        self.content
            .sector
            .find_adjoint_fusion_tree_pair_index(logical_key)
    }

    pub fn required_len(&self) -> Result<usize, CoreError> {
        Ok(self.content.required_len)
    }

    /// Whether this structure's blocks are proved, at construction, to reach
    /// every offset of `0..required_len` exactly once, so writing each block
    /// once fully initializes an owned payload. O(1); `false` only means no
    /// constructor recorded the proof.
    #[doc(hidden)]
    #[inline]
    pub fn storage_tiling_proven(&self) -> bool {
        self.content.storage_tiling_proven()
    }

    /// See [`BlockStructureContent::record_storage_tiling`].
    pub(crate) fn record_storage_tiling(&self) {
        self.content.record_storage_tiling();
    }

    /// Compiles the canonical coupled-sector matrix layout of this structure.
    ///
    /// `Ok(Some(_))` contains immutable checked regions covering storage exactly once.
    /// `Ok(None)` means the structure is valid but is not the canonical contiguous
    /// coupled-sector layout. `Err` reports a structural lookup or size overflow.
    pub fn coupled_sector_regions(
        &self,
        nout: usize,
    ) -> Result<Option<Arc<[CoupledSectorRegion]>>, CoreError> {
        if nout > self.rank() {
            return Ok(None);
        }
        let mut materialized = None;
        let regions = self
            .regions
            .coupled_region_cache
            .get_or_init(|| new_coupled_region_cache(self.rank()))[nout]
            .get_or_init(|| {
                let regions = compile_coupled_sector_regions(self, nout)
                    .map(|regions| regions.map(Arc::<[_]>::from));
                materialized = Some(coupled_region_result_bytes(&regions));
                regions
            })
            .clone();
        // Charged after the slot is set, outside its initialization: the
        // charge takes a cache shard lock.
        if let Some(bytes) = materialized {
            if let Some(owner) = self.complete_entry() {
                owner.charge_regions(bytes as u64);
            }
        }
        regions
    }

    /// Makes `owner` the cache entry charged for this wrapper's region memo.
    pub(crate) fn link_region_owner(
        &self,
        owner: &Arc<crate::fusion_space::DegeneracyStructureEntry>,
    ) {
        let _ = self.regions.owner.set(Arc::downgrade(owner));
    }

    /// The live cache entry this wrapper was admitted under, if any: one
    /// atomic load and a `Weak` upgrade. Reached without a cache lookup.
    pub(crate) fn complete_entry(
        &self,
    ) -> Option<Arc<crate::fusion_space::DegeneracyStructureEntry>> {
        self.regions.owner.get()?.upgrade()
    }

    /// Whether a provider of `identity` already admitted every key.
    pub(crate) fn is_checked_generic_admitted(&self, identity: &RuleIdentity) -> bool {
        self.regions.checked_generic_admitted.get() == Some(identity)
    }

    /// Records that a provider of `identity` admitted every key. The first
    /// identity wins; another provider keeps validating per call.
    pub(crate) fn record_checked_generic_admitted(&self, identity: RuleIdentity) {
        let _ = self.regions.checked_generic_admitted.set(identity);
    }

    /// Heap bytes of the region memos materialized so far.
    pub(crate) fn materialized_region_bytes(&self) -> u64 {
        self.regions.coupled_region_cache.get().map_or(0, |slots| {
            slots
                .iter()
                .filter_map(OnceLock::get)
                .map(coupled_region_result_bytes)
                .sum::<usize>()
        }) as u64
    }

    /// Bytes a cache entry retains by owning a wrapper of rank `rank`: the
    /// wrapper and region-state allocations and the per-split memo slots.
    pub(crate) fn wrapper_retained_bytes(rank: usize) -> usize {
        let arc_header = 2 * std::mem::size_of::<usize>();
        (arc_header + std::mem::size_of::<Self>())
            .saturating_add(arc_header + std::mem::size_of::<BlockStructureRegionState>())
            .saturating_add(arc_header)
            .saturating_add(
                (rank + 1).saturating_mul(std::mem::size_of::<OnceLock<CoupledRegionResult>>()),
            )
    }

    #[cfg(test)]
    pub(crate) fn coupled_region_cache_is_initialized(&self) -> bool {
        self.regions.coupled_region_cache.get().is_some()
    }

    #[cfg(test)]
    pub(crate) fn weak_region_state(&self) -> Weak<BlockStructureRegionState> {
        Arc::downgrade(&self.regions)
    }
}

fn coupled_sector_matrix_from_validated_keys(
    nout: usize,
    rank: usize,
    mut blocks: Vec<(FusionTreePairKey, Vec<usize>)>,
) -> Result<BlockStructure, CoreError> {
    blocks.sort_by_key(|(key, _)| key.codomain_tree().coupled().id());
    let (keys, shapes): (Vec<_>, Vec<_>) = blocks.into_iter().unzip();
    let specs = coupled_sector_matrix_block_specs(nout, rank, &keys, &shapes)?;
    BlockStructure::from_blocks_with_rank(rank, specs)
}

/// Heap bytes one materialized memo slot retains: the region slice, each
/// region's tree list, and any spilled extent shapes. Tree keys share their
/// backings with the block content, already charged there.
fn coupled_region_result_bytes(result: &CoupledRegionResult) -> usize {
    let Ok(Some(regions)) = result else {
        return 0;
    };
    let arc_header = 2 * std::mem::size_of::<usize>();
    regions.iter().fold(
        arc_header + std::mem::size_of_val(regions.as_ref()),
        |bytes, region| {
            region.trees.iter().fold(
                bytes.saturating_add(std::mem::size_of_val(region.trees.as_ref())),
                |bytes, extent| {
                    if extent.shape.spilled() {
                        bytes.saturating_add(extent.shape.capacity() * std::mem::size_of::<usize>())
                    } else {
                        bytes
                    }
                },
            )
        },
    )
}
