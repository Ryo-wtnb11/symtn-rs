use super::*;

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct FusionTreeBlockLayoutEntry {
    pub(crate) row: usize,
    pub(crate) col: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct FusionTreeCoupledSectorLayout {
    pub(crate) start: usize,
    pub(crate) row_count: usize,
    pub(crate) col_count: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct FusionTreeHomSpaceLayoutData {
    pub(crate) keys: Arc<[FusionTreePairKey]>,
    pub(crate) sectors: Vec<FusionTreeCoupledSectorLayout>,
    /// The block keys as a sector structure, built once per sector layout
    /// and shared by every block structure over it, whatever its
    /// degeneracies (TensorKit `sectorstructure`, `structure.jl:41`). An
    /// error (duplicate keys from a provider with repeated fusion channels)
    /// is reported where a block structure is built, as before.
    pub(crate) sector: Result<Arc<SectorStructure>, CoreError>,
}

pub(super) fn generic_keys_for_coupled_from_groups(
    codomain: &[CoupledFusionTrees],
    codomain_fold: &CoupledSectorFold,
    domain: &[CoupledFusionTrees],
    domain_fold: &CoupledSectorFold,
    coupled: SectorId,
    outside_table_error: impl Fn(&str, &CoupledSectorFold, SectorId) -> CoreError,
) -> Result<Vec<FusionTreePairKey>, CoreError> {
    for (side, fold) in [("codomain", codomain_fold), ("domain", domain_fold)] {
        if fold.is_unknown() || fold.tainted().contains(&coupled) {
            return Err(outside_table_error(side, fold, coupled));
        }
    }
    let codomain = codomain
        .iter()
        .find(|g| g.coupled == coupled)
        .map_or(&[][..], |g| g.trees.as_slice());
    let domain = domain
        .iter()
        .find(|g| g.coupled == coupled)
        .map_or(&[][..], |g| g.trees.as_slice());
    let mut keys = Vec::with_capacity(codomain.len() * domain.len());
    for domain_tree in domain {
        for codomain_tree in codomain {
            keys.push(FusionTreePairKey::pair(
                codomain_tree.clone(),
                domain_tree.clone(),
            ));
        }
    }
    Ok(keys)
}

#[derive(Clone, Debug)]
pub(crate) struct FusionTreeHomSpaceLayout {
    data: FusionTreeHomSpaceLayoutData,
}

#[derive(Debug)]
pub(super) enum PreparedFusionTreeLayoutState {
    Cached {
        key: FusionTreeHomSpaceCacheKey,
        layout: Arc<FusionTreeHomSpaceLayout>,
    },
    Cold {
        key: FusionTreeHomSpaceCacheKey,
        data: FusionTreeHomSpaceLayoutData,
    },
}

/// Checked fusion-tree metadata staged without publishing process-local state.
///
/// This is an expert transaction boundary for downstream builders that still
/// have fallible shape or storage work. Call [`Self::commit`] only after that
/// work succeeds.
#[doc(hidden)]
#[derive(Debug)]
pub struct PreparedFusionTreeLayout {
    pub(super) state: PreparedFusionTreeLayoutState,
    pub(super) complete_epoch: usize,
}

impl PreparedFusionTreeLayout {
    fn cache_key(&self) -> &FusionTreeHomSpaceCacheKey {
        match &self.state {
            PreparedFusionTreeLayoutState::Cached { key, .. }
            | PreparedFusionTreeLayoutState::Cold { key, .. } => key,
        }
    }

    pub(super) fn layout_data(&self) -> &FusionTreeHomSpaceLayoutData {
        match &self.state {
            PreparedFusionTreeLayoutState::Cached { layout, .. } => layout,
            PreparedFusionTreeLayoutState::Cold { data, .. } => data,
        }
    }

    fn validate_homspace_signature(&self, homspace: &FusionTreeHomSpace) -> Result<(), CoreError> {
        let key = self.cache_key();
        if Arc::ptr_eq(&homspace.content, &key.homspace) {
            return Ok(());
        }
        if !product_space_signature_eq(homspace.codomain(), &key.homspace.codomain)
            || !product_space_signature_eq(homspace.domain(), &key.homspace.domain)
        {
            return Err(CoreError::MalformedFusionTree {
                message: "prepared layout does not match HomSpace sector signature",
            });
        }
        Ok(())
    }

    pub fn keys(&self) -> &[FusionTreePairKey] {
        match &self.state {
            PreparedFusionTreeLayoutState::Cached { layout, .. } => layout.keys.as_ref(),
            PreparedFusionTreeLayoutState::Cold { data, .. } => data.keys.as_ref(),
        }
    }

    pub fn keys_arc(&self) -> Arc<[FusionTreePairKey]> {
        match &self.state {
            PreparedFusionTreeLayoutState::Cached { layout, .. } => Arc::clone(&layout.keys),
            PreparedFusionTreeLayoutState::Cold { data, .. } => Arc::clone(&data.keys),
        }
    }

    /// Builds final coupled storage directly from authoritative leg
    /// degeneracies while keeping this prepared layout unpublished.
    pub fn build_from_leg_degeneracies(
        &self,
        homspace: &FusionTreeHomSpace,
    ) -> Result<Arc<BlockStructure>, CoreError> {
        self.validate_homspace_signature(homspace)?;
        // Why not call the cached public builder: downstream validation must
        // finish before this transaction publishes a layout ID or admission,
        // and the prepared data already owns the one checked enumeration.
        // Degeneracies are deliberately absent from the signature: the target
        // HomSpace is their authority, while sectors and duality select keys.
        let (sector, degeneracy) =
            coupled_subblock_parts_from_leg_degeneracies(homspace, self.layout_data())?;
        let structure = BlockStructure::from_shared_parts(sector, degeneracy)?;
        structure.record_storage_tiling();
        Ok(structure.into_shared())
    }

    /// Finalizes a checked complete multiplicity-free layout through the core
    /// structural cache after all fallible leg-derived storage work succeeds.
    #[doc(hidden)]
    pub fn build_complete_from_leg_degeneracies(
        &self,
        homspace: &FusionTreeHomSpace,
    ) -> Result<Arc<BlockStructure>, CoreError> {
        self.validate_homspace_signature(homspace)?;
        let epoch = self.complete_epoch;
        let key = CompleteHomSpaceStructureCacheKey {
            rule: self.cache_key().rule.clone(),
            mode: CompleteFusionMode::MultiplicityFree,
            homspace: Arc::clone(&homspace.content),
        };
        // Why no extent walk before the lookup: entries are admitted only
        // after the builder's walk succeeded over an equal (rule identity,
        // complete HomSpace content) key, and that walk is a pure function of
        // the key because `RuleIdentity` determines the fusion enumeration.
        // A hit therefore proves the walk would succeed; a miss walks once,
        // inside the builder, before any statistic changes.
        if let Some(entry) = complete_hom_space_structure_cached(&key) {
            return Ok(entry.structure());
        }

        let built = self.build_from_leg_degeneracies(homspace)?;
        Ok(admit_complete_hom_space_structure(key, built, epoch).1)
    }

    /// Completes the prepared MF layout and returns the HomSpace backing owned
    /// by the same complete-cache entry.
    #[doc(hidden)]
    pub fn build_complete_homspace_from_leg_degeneracies(
        &self,
        homspace: FusionTreeHomSpace,
    ) -> Result<(FusionTreeHomSpace, Arc<BlockStructure>), CoreError> {
        self.validate_homspace_signature(&homspace)?;
        let epoch = self.complete_epoch;
        let key = CompleteHomSpaceStructureCacheKey {
            rule: self.cache_key().rule.clone(),
            mode: CompleteFusionMode::MultiplicityFree,
            homspace: Arc::clone(&homspace.content),
        };
        let (entry, structure) = if let Some(entry) = complete_hom_space_structure_cached(&key) {
            let structure = entry.structure();
            (entry, structure)
        } else {
            let built = self.build_from_leg_degeneracies(&homspace)?;
            admit_complete_hom_space_structure(key, built, epoch)
        };
        Ok((homspace.with_canonical_id(entry.homspace_id()), structure))
    }

    /// Publishes the prepared layout and returns its shared key storage.
    ///
    /// Why no fallible return: checked enumeration and layout-data formation
    /// completed in `prepare`; what remains is the bounded admission of a
    /// cold layout, which is process-local publication only. A layout found
    /// warm in `prepare` is returned as is.
    pub fn commit(self) -> Arc<[FusionTreePairKey]> {
        Arc::clone(&self.commit_layout().keys)
    }

    pub(crate) fn commit_layout(self) -> Arc<FusionTreeHomSpaceLayout> {
        let cache = sector_structure_cache();
        let (key, layout) = match self.state {
            // Why no residency re-check: the layout is a pure function of its
            // key, so an entry evicted or reset since prepare is left out
            // rather than re-admitted, exactly as if it were evicted just
            // after this commit. Re-checking walked the sector signature a
            // second time on every warm call.
            PreparedFusionTreeLayoutState::Cached { layout, .. } => return layout,
            PreparedFusionTreeLayoutState::Cold { key, data } => {
                (key, Arc::new(FusionTreeHomSpaceLayout { data }))
            }
        };
        let charged_bytes = charged_fusion_tree_layout_bytes(&key, &layout);
        // A layout is pure data under a semantic key and carries no
        // identity, so any epoch outside a reset may publish it.
        let (published, _) =
            cache.publish(&key, Arc::clone(&layout), charged_bytes, core_reset_epoch());
        #[cfg(test)]
        if Arc::ptr_eq(&published, &layout) {
            crate::probe::hit(crate::probe::Event::LayoutBuild);
            crate::probe::hit(crate::probe::Event::LayoutAdmission);
        }
        published
    }
}

impl FusionTreeHomSpaceLayout {
    pub(crate) fn new(data: FusionTreeHomSpaceLayoutData) -> Self {
        Self { data }
    }
}

impl std::ops::Deref for FusionTreeHomSpaceLayout {
    type Target = FusionTreeHomSpaceLayoutData;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

#[cfg(test)]
pub(crate) struct ReconstructedFusionTreeCoupledSectorLayout {
    pub(crate) start: usize,
    pub(crate) row_count: usize,
    pub(crate) col_count: usize,
    pub(crate) row_key_offsets: Vec<usize>,
    pub(crate) col_key_offsets: Vec<usize>,
    pub(crate) entries: Vec<FusionTreeBlockLayoutEntry>,
}

#[cfg(test)]
pub(crate) struct ReconstructedFusionTreeHomSpaceLayoutData {
    pub(crate) keys: Arc<[FusionTreePairKey]>,
    pub(crate) sectors: Vec<ReconstructedFusionTreeCoupledSectorLayout>,
}

#[cfg(test)]
pub(crate) fn reconstructed_fusion_tree_layout_data_from_keys(
    keys: Vec<FusionTreePairKey>,
) -> ReconstructedFusionTreeHomSpaceLayoutData {
    let keys = Arc::<[FusionTreePairKey]>::from(keys);
    let mut sectors = Vec::new();
    let mut run_start = 0usize;
    while run_start < keys.len() {
        let coupled = keys[run_start].codomain_tree().coupled();
        let mut run_end = run_start;
        let mut row_indices = FxHashMap::<FusionTreeKey, usize>::default();
        let mut col_indices = FxHashMap::<FusionTreeKey, usize>::default();
        let mut row_key_offsets = Vec::new();
        let mut col_key_offsets = Vec::new();
        let mut entries = Vec::new();
        while run_end < keys.len() && keys[run_end].codomain_tree().coupled() == coupled {
            let row = match row_indices.get(keys[run_end].codomain_tree()) {
                Some(&index) => index,
                None => {
                    let index = row_indices.len();
                    row_indices.insert(keys[run_end].codomain_tree().clone(), index);
                    row_key_offsets.push(run_end - run_start);
                    crate::probe::hit(crate::probe::Event::CoupledGridReconstructionInsert);
                    index
                }
            };
            let col = match col_indices.get(keys[run_end].domain_tree()) {
                Some(&index) => index,
                None => {
                    let index = col_indices.len();
                    col_indices.insert(keys[run_end].domain_tree().clone(), index);
                    col_key_offsets.push(run_end - run_start);
                    crate::probe::hit(crate::probe::Event::CoupledGridReconstructionInsert);
                    index
                }
            };
            entries.push(FusionTreeBlockLayoutEntry { row, col });
            run_end += 1;
        }
        sectors.push(ReconstructedFusionTreeCoupledSectorLayout {
            start: run_start,
            row_count: row_indices.len(),
            col_count: col_indices.len(),
            row_key_offsets,
            col_key_offsets,
            entries,
        });
        run_start = run_end;
    }
    ReconstructedFusionTreeHomSpaceLayoutData { keys, sectors }
}

fn fusion_tree_layout_capacities(
    codomain: &[CoupledFusionTrees],
    domain: &[CoupledFusionTrees],
) -> Option<(usize, usize)> {
    let mut key_count = 0usize;
    let mut sector_count = 0usize;
    let mut codomain_index = 0usize;
    let mut domain_index = 0usize;
    while codomain_index < codomain.len() && domain_index < domain.len() {
        match codomain[codomain_index]
            .coupled
            .cmp(&domain[domain_index].coupled)
        {
            std::cmp::Ordering::Less => codomain_index += 1,
            std::cmp::Ordering::Greater => domain_index += 1,
            std::cmp::Ordering::Equal => {
                let row_count = codomain[codomain_index].trees.len();
                let col_count = domain[domain_index].trees.len();
                if row_count != 0 && col_count != 0 {
                    key_count = key_count.checked_add(row_count.checked_mul(col_count)?)?;
                    sector_count = sector_count.checked_add(1)?;
                }
                codomain_index += 1;
                domain_index += 1;
            }
        }
    }
    Some((key_count, sector_count))
}

pub(super) fn fusion_tree_layout_data_from_groups(
    rank: usize,
    codomain: &[CoupledFusionTrees],
    domain: &[CoupledFusionTrees],
) -> FusionTreeHomSpaceLayoutData {
    let (key_capacity, sector_capacity) =
        fusion_tree_layout_capacities(codomain, domain).unwrap_or((0, 0));
    let mut keys = Vec::with_capacity(key_capacity);
    let mut sectors = Vec::with_capacity(sector_capacity);
    let mut codomain_index = 0usize;
    let mut domain_index = 0usize;
    while codomain_index < codomain.len() && domain_index < domain.len() {
        match codomain[codomain_index]
            .coupled
            .cmp(&domain[domain_index].coupled)
        {
            std::cmp::Ordering::Less => codomain_index += 1,
            std::cmp::Ordering::Greater => domain_index += 1,
            std::cmp::Ordering::Equal => {
                let codomain_trees = &codomain[codomain_index].trees;
                let domain_trees = &domain[domain_index].trees;
                let start = keys.len();
                let row_count = codomain_trees.len();
                let col_count = domain_trees.len();
                if row_count == 0 || col_count == 0 {
                    codomain_index += 1;
                    domain_index += 1;
                    continue;
                }
                for domain_tree in domain_trees {
                    for codomain_tree in codomain_trees {
                        keys.push(FusionTreePairKey::pair(
                            codomain_tree.clone(),
                            domain_tree.clone(),
                        ));
                    }
                }
                sectors.push(FusionTreeCoupledSectorLayout {
                    start,
                    row_count,
                    col_count,
                });
                codomain_index += 1;
                domain_index += 1;
            }
        }
    }
    // Single-arena ownership: the layout keys and the sector structure share one canonical
    // arena; the structure then takes it without a copy.
    rehome_canonical(&mut keys);
    let sector = SectorStructure::from_fusion_tree_keys(rank, &keys).map(Arc::new);
    FusionTreeHomSpaceLayoutData {
        keys: Arc::from(keys),
        sectors,
        sector,
    }
}
