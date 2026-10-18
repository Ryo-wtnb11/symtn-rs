use super::*;

#[derive(Debug, Eq, PartialEq, Hash)]
pub(crate) struct FusionTreeHomSpaceContent {
    pub(super) codomain: FusionProductSpace,
    pub(super) domain: FusionProductSpace,
}

impl FusionTreeHomSpaceContent {
    pub(super) fn charged_retained_bytes(&self) -> usize {
        fn product_space_bytes(space: &FusionProductSpace) -> usize {
            std::mem::size_of::<FusionProductSpace>()
                .saturating_add(spilled_smallvec_heap_bytes(&space.legs))
                .saturating_add(space.legs.iter().fold(0usize, |bytes, leg| {
                    bytes.saturating_add(leg.charged_retained_bytes())
                }))
        }

        product_space_bytes(&self.codomain)
            .saturating_add(product_space_bytes(&self.domain))
            .saturating_add(2 * std::mem::size_of::<usize>())
    }
}

pub struct FusionTreeHomSpace {
    pub(crate) content: Arc<FusionTreeHomSpaceContent>,
    id: OnceLock<HomSpaceId>,
}

/// Describes the geometric placement of one canonical unit leg.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UnitLegInsertion {
    Left { position: usize, dual: bool },
    Right { position: usize, dual: bool },
}

impl UnitLegInsertion {
    pub(super) fn position(self) -> usize {
        match self {
            Self::Left { position, .. } | Self::Right { position, .. } => position,
        }
    }

    pub(super) fn dual(self) -> bool {
        match self {
            Self::Left { dual, .. } | Self::Right { dual, .. } => dual,
        }
    }
}

fn insert_product_space_leg(
    space: &FusionProductSpace,
    position: usize,
    leg: SectorLeg,
) -> FusionProductSpace {
    let mut legs = space.legs.clone();
    legs.insert(position, leg);
    FusionProductSpace { legs }
}

fn remove_product_space_leg(space: &FusionProductSpace, position: usize) -> FusionProductSpace {
    let mut legs = space.legs.clone();
    legs.remove(position);
    FusionProductSpace { legs }
}

impl std::fmt::Debug for FusionTreeHomSpace {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FusionTreeHomSpace")
            .field("codomain", &self.content.codomain)
            .field("domain", &self.content.domain)
            .field("id", &self.id)
            .finish()
    }
}

impl Clone for FusionTreeHomSpace {
    fn clone(&self) -> Self {
        Self {
            content: Arc::clone(&self.content),
            id: self.id.clone(),
        }
    }
}

impl PartialEq for FusionTreeHomSpace {
    fn eq(&self, other: &Self) -> bool {
        self.content.codomain == other.content.codomain
            && self.content.domain == other.content.domain
    }
}

impl Eq for FusionTreeHomSpace {}

impl std::hash::Hash for FusionTreeHomSpace {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.content.codomain.hash(state);
        self.content.domain.hash(state);
    }
}

impl FusionTreeHomSpace {
    /// Conservative retained bytes for this HomSpace and its Arc-backed
    /// provider-neutral leg metadata.
    #[doc(hidden)]
    pub fn charged_retained_bytes(&self) -> usize {
        std::mem::size_of::<Self>().saturating_add(self.content.charged_retained_bytes())
    }

    /// Builds a fusion-tree hom space from codomain and domain product spaces.
    ///
    /// # Examples
    ///
    /// ```
    /// use symtn_core::{FusionProductSpace, FusionTreeHomSpace, SectorLeg, Z2Irrep};
    ///
    /// let hom = FusionTreeHomSpace::new(
    ///     FusionProductSpace::new([SectorLeg::new([(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 1)], false)]),
    ///     FusionProductSpace::new([SectorLeg::new([(Z2Irrep::EVEN, 1)], false)]),
    /// );
    /// assert_eq!(hom.codomain().len(), 1);
    /// assert_eq!(hom.domain().len(), 1);
    /// ```
    pub fn new(codomain: FusionProductSpace, domain: FusionProductSpace) -> Self {
        Self {
            content: Arc::new(FusionTreeHomSpaceContent { codomain, domain }),
            id: OnceLock::new(),
        }
    }

    /// Builds a hom space when each external leg has exactly one sector,
    /// from `(sector, degeneracy)` pairs.
    ///
    /// # Examples
    ///
    /// ```
    /// use symtn_core::{FusionTreeHomSpace, Z2Irrep};
    ///
    /// let hom = FusionTreeHomSpace::from_sectors([(Z2Irrep::EVEN, 1)], [(Z2Irrep::ODD, 1)]);
    /// assert_eq!(hom.codomain().len(), 1);
    /// assert_eq!(hom.domain().len(), 1);
    /// ```
    pub fn from_sectors<Codomain, Domain, CodomainSector, DomainSector>(
        codomain: Codomain,
        domain: Domain,
    ) -> Self
    where
        Codomain: IntoIterator<Item = (CodomainSector, usize)>,
        Domain: IntoIterator<Item = (DomainSector, usize)>,
        CodomainSector: Into<SectorId>,
        DomainSector: Into<SectorId>,
    {
        Self::new(
            FusionProductSpace::new(
                codomain
                    .into_iter()
                    .map(|(sector, degeneracy)| SectorLeg::new([(sector, degeneracy)], false)),
            ),
            FusionProductSpace::new(
                domain
                    .into_iter()
                    .map(|(sector, degeneracy)| SectorLeg::new([(sector, degeneracy)], false)),
            ),
        )
    }

    /// Builds a hom space from raw `(sector id, degeneracy)` pairs.
    ///
    /// # Examples
    ///
    /// ```
    /// use symtn_core::FusionTreeHomSpace;
    ///
    /// let hom = FusionTreeHomSpace::from_sector_ids([(0, 1), (1, 1)], [(1, 1)]);
    /// assert_eq!(hom.codomain().len(), 2);
    /// assert_eq!(hom.domain().len(), 1);
    /// ```
    pub fn from_sector_ids<Codomain, Domain>(codomain: Codomain, domain: Domain) -> Self
    where
        Codomain: IntoIterator<Item = (usize, usize)>,
        Domain: IntoIterator<Item = (usize, usize)>,
    {
        Self::from_sectors(
            codomain
                .into_iter()
                .map(|(sector, degeneracy)| (SectorId::new(sector), degeneracy)),
            domain
                .into_iter()
                .map(|(sector, degeneracy)| (SectorId::new(sector), degeneracy)),
        )
    }

    #[inline]
    pub fn codomain(&self) -> &FusionProductSpace {
        &self.content.codomain
    }

    #[inline]
    pub fn domain(&self) -> &FusionProductSpace {
        &self.content.domain
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.content.codomain.len() + self.content.domain.len()
    }

    /// Inserts a canonical unit leg at a zero-based position using TensorKit's
    /// left-before seam convention.
    pub fn insert_left_unit<R>(
        &self,
        rule: &R,
        position: usize,
        dual: bool,
    ) -> Result<Self, CoreError>
    where
        R: CanonicalUnitFusionRule,
    {
        if position > self.rank() {
            return Err(CoreError::UnitLayoutCorrespondence);
        }
        let unit = SectorLeg::new([(rule.vacuum(), 1)], dual);
        if position < self.codomain().len() {
            Ok(Self::new(
                insert_product_space_leg(self.codomain(), position, unit),
                self.domain().clone(),
            ))
        } else {
            Ok(Self::new(
                self.codomain().clone(),
                insert_product_space_leg(self.domain(), position - self.codomain().len(), unit),
            ))
        }
    }

    /// Inserts a canonical unit leg at a zero-based position using TensorKit's
    /// right-after seam convention.
    pub fn insert_right_unit<R>(
        &self,
        rule: &R,
        position: usize,
        dual: bool,
    ) -> Result<Self, CoreError>
    where
        R: CanonicalUnitFusionRule,
    {
        if position > self.rank() {
            return Err(CoreError::UnitLayoutCorrespondence);
        }
        let unit = SectorLeg::new([(rule.vacuum(), 1)], dual);
        if position <= self.codomain().len() {
            Ok(Self::new(
                insert_product_space_leg(self.codomain(), position, unit),
                self.domain().clone(),
            ))
        } else {
            Ok(Self::new(
                self.codomain().clone(),
                insert_product_space_leg(self.domain(), position - self.codomain().len(), unit),
            ))
        }
    }

    /// Removes one canonical unit leg in flat external-axis order, matching
    /// TensorKit's unit-leg geometry.
    pub fn remove_unit<R>(&self, rule: &R, axis: usize) -> Result<Self, CoreError>
    where
        R: CanonicalUnitFusionRule,
    {
        if axis >= self.rank() {
            return Err(CoreError::UnitLayoutCorrespondence);
        }
        let leg = if axis < self.codomain().len() {
            &self.codomain().legs()[axis]
        } else {
            &self.domain().legs()[axis - self.codomain().len()]
        };
        if leg.sectors() != [rule.vacuum()] || leg.degeneracy(rule.vacuum()) != Some(1) {
            return Err(CoreError::UnitLayoutCorrespondence);
        }
        if axis < self.codomain().len() {
            Ok(Self::new(
                remove_product_space_leg(self.codomain(), axis),
                self.domain().clone(),
            ))
        } else {
            Ok(Self::new(
                self.codomain().clone(),
                remove_product_space_leg(self.domain(), axis - self.codomain().len()),
            ))
        }
    }

    /// Checked Generic counterpart of [`Self::insert_left_unit`].
    #[doc(hidden)]
    pub fn insert_left_unit_checked<R>(
        &self,
        rule: &R,
        position: usize,
        dual: bool,
    ) -> Result<Self, CoreError>
    where
        R: CheckedCanonicalUnitFusionRule,
    {
        if position > self.rank() {
            return Err(CoreError::UnitLayoutCorrespondence);
        }
        let unit = SectorLeg::new([(rule.vacuum(), 1)], dual);
        if position < self.codomain().len() {
            Ok(Self::new(
                insert_product_space_leg(self.codomain(), position, unit),
                self.domain().clone(),
            ))
        } else {
            Ok(Self::new(
                self.codomain().clone(),
                insert_product_space_leg(self.domain(), position - self.codomain().len(), unit),
            ))
        }
    }

    /// Checked Generic counterpart of [`Self::insert_right_unit`].
    #[doc(hidden)]
    pub fn insert_right_unit_checked<R>(
        &self,
        rule: &R,
        position: usize,
        dual: bool,
    ) -> Result<Self, CoreError>
    where
        R: CheckedCanonicalUnitFusionRule,
    {
        if position > self.rank() {
            return Err(CoreError::UnitLayoutCorrespondence);
        }
        let unit = SectorLeg::new([(rule.vacuum(), 1)], dual);
        if position <= self.codomain().len() {
            Ok(Self::new(
                insert_product_space_leg(self.codomain(), position, unit),
                self.domain().clone(),
            ))
        } else {
            Ok(Self::new(
                self.codomain().clone(),
                insert_product_space_leg(self.domain(), position - self.codomain().len(), unit),
            ))
        }
    }

    /// Checked Generic counterpart of [`Self::remove_unit`].
    #[doc(hidden)]
    pub fn remove_unit_checked<R>(&self, rule: &R, axis: usize) -> Result<Self, CoreError>
    where
        R: CheckedCanonicalUnitFusionRule,
    {
        if axis >= self.rank() {
            return Err(CoreError::UnitLayoutCorrespondence);
        }
        let vacuum = rule.vacuum();
        let leg = if axis < self.codomain().len() {
            &self.codomain().legs()[axis]
        } else {
            &self.domain().legs()[axis - self.codomain().len()]
        };
        if leg.sectors() != [vacuum] || leg.degeneracy(vacuum) != Some(1) {
            return Err(CoreError::UnitLayoutCorrespondence);
        }
        if axis < self.codomain().len() {
            Ok(Self::new(
                remove_product_space_leg(self.codomain(), axis),
                self.domain().clone(),
            ))
        } else {
            Ok(Self::new(
                self.codomain().clone(),
                remove_product_space_leg(self.domain(), axis - self.codomain().len()),
            ))
        }
    }

    /// Lazily assigned collision-safe process-local semantic identity.
    /// Hashing is O(1), and equal spaces compare equal across intern eviction.
    #[inline]
    pub fn id(&self) -> HomSpaceId {
        self.id
            .get_or_init(|| HomSpaceId::from_content(Arc::clone(&self.content)))
            .clone()
    }

    pub(crate) fn with_canonical_id(self, id: HomSpaceId) -> Self {
        debug_assert_eq!(self.content, id.content);
        Self {
            content: Arc::clone(&id.content),
            id: OnceLock::from(id),
        }
    }

    /// Returns the already-published semantic identity without initializing it.
    ///
    /// Why not call [`Self::id`]: fallible metadata builders must validate
    /// algebra before publishing process-local identity state.
    #[doc(hidden)]
    #[inline]
    pub fn existing_id(&self) -> Option<HomSpaceId> {
        self.id.get().cloned()
    }

    pub fn select<R>(
        &self,
        rule: &R,
        codomain_axes: &[usize],
        domain_axes: &[usize],
    ) -> Result<Self, CoreError>
    where
        R: FusionRule,
    {
        let descriptor = self.select_descriptor(codomain_axes, domain_axes)?;
        Ok(descriptor.materialize(rule))
    }

    /// Checked sibling of [`Self::select`] for finite or encoded fusion
    /// algebras whose dual operation may not be representable.
    pub fn try_select_checked<R>(
        &self,
        rule: &R,
        codomain_axes: &[usize],
        domain_axes: &[usize],
    ) -> Result<Self, CheckedFusionSpaceError>
    where
        R: CheckedFusionAlgebra,
    {
        let descriptor = self.select_descriptor(codomain_axes, domain_axes)?;
        descriptor.try_materialize(rule).map_err(Into::into)
    }

    pub fn permute<R>(
        &self,
        rule: &R,
        codomain_axes: &[usize],
        domain_axes: &[usize],
    ) -> Result<Self, CoreError>
    where
        R: FusionRule,
    {
        let mut axes = SmallVec::<[usize; 8]>::new();
        axes.extend_from_slice(codomain_axes);
        axes.extend_from_slice(domain_axes);
        validate_permutation_inline(&axes, self.rank())?;
        self.select(rule, codomain_axes, domain_axes)
    }

    /// Checked sibling of [`Self::permute`] for finite or encoded fusion
    /// algebras whose dual operation may not be representable.
    pub fn try_permute_checked<R>(
        &self,
        rule: &R,
        codomain_axes: &[usize],
        domain_axes: &[usize],
    ) -> Result<Self, CheckedFusionSpaceError>
    where
        R: CheckedFusionAlgebra,
    {
        let mut axes = SmallVec::<[usize; 8]>::new();
        axes.extend_from_slice(codomain_axes);
        axes.extend_from_slice(domain_axes);
        validate_permutation_inline(&axes, self.rank())?;
        let descriptor = self.select_descriptor(codomain_axes, domain_axes)?;
        descriptor.try_materialize(rule).map_err(Into::into)
    }

    /// Checked Generic sibling of [`Self::permute`] that uses the provider's
    /// fallible dual operation without requiring [`FusionRule`].
    #[doc(hidden)]
    pub fn try_permute_generic_checked<R>(
        &self,
        rule: &R,
        codomain_axes: &[usize],
        domain_axes: &[usize],
    ) -> Result<Self, CheckedGenericStructureError<R::Error>>
    where
        R: CheckedGenericFusion,
    {
        let mut axes = SmallVec::<[usize; 8]>::new();
        axes.extend_from_slice(codomain_axes);
        axes.extend_from_slice(domain_axes);
        validate_permutation_inline(&axes, self.rank())?;
        let descriptor = self.select_descriptor(codomain_axes, domain_axes)?;
        descriptor.try_materialize_generic(rule)
    }

    fn select_descriptor<'a>(
        &'a self,
        codomain_axes: &[usize],
        domain_axes: &[usize],
    ) -> Result<HomSpaceDescriptor<'a>, CoreError> {
        validate_axis_selection(codomain_axes, domain_axes, self.rank())?;
        Ok(HomSpaceDescriptor::new(
            codomain_axes
                .iter()
                .map(|&axis| self.external_axis_leg_view(axis)),
            domain_axes
                .iter()
                .map(|&axis| self.external_axis_leg_view(axis).toggled()),
        ))
    }

    pub fn compose<R>(rule: &R, lhs: &Self, rhs: &Self) -> Result<Self, CoreError>
    where
        R: FusionRule,
    {
        if lhs.domain().len() != rhs.codomain().len() {
            return Err(CoreError::DimensionMismatch {
                expected: lhs.domain().len(),
                actual: rhs.codomain().len(),
            });
        }
        let lhs_codomain_rank = lhs.codomain().len();
        for (index, (lhs_domain, rhs_codomain)) in lhs
            .domain()
            .legs()
            .iter()
            .zip(rhs.codomain().legs())
            .enumerate()
        {
            validate_composed_leg(lhs_domain, rhs_codomain, (lhs_codomain_rank + index, index))?;
        }
        let descriptor = HomSpaceDescriptor::new(
            lhs.codomain().legs().iter().map(OrientedLegView::borrowed),
            rhs.domain().legs().iter().map(OrientedLegView::borrowed),
        );
        Ok(descriptor.materialize(rule))
    }

    /// Structural lowering of a general axis contraction, the homspace-level
    /// analog of TensorOperations' `tensorcontract!`: reorders both operands
    /// to `(open, contracted)` x `(contracted, open)`, composes them (see
    /// [`Self::compose`]), and applies the requested output axis order.
    pub fn tensorcontract_homspace<R>(
        rule: &R,
        lhs: &Self,
        rhs: &Self,
        lhs_contracting_axes: &[usize],
        rhs_contracting_axes: &[usize],
        output_axes: &[usize],
        dst_codomain_rank: usize,
    ) -> Result<Self, CoreError>
    where
        R: FusionRule,
    {
        OrientedFusionTreeHomSpace::tensorcontract_homspace(
            rule,
            OrientedFusionTreeHomSpace::new(lhs, FusionTreePairOrientation::Direct),
            OrientedFusionTreeHomSpace::new(rhs, FusionTreePairOrientation::Direct),
            lhs_contracting_axes,
            rhs_contracting_axes,
            output_axes,
            dst_codomain_rank,
        )
    }

    /// Checked sibling of [`Self::tensorcontract_homspace`] for finite or
    /// encoded fusion algebras whose dual operation may not be representable.
    #[allow(clippy::too_many_arguments)]
    pub fn try_tensorcontract_homspace_checked<R>(
        rule: &R,
        lhs: &Self,
        rhs: &Self,
        lhs_contracting_axes: &[usize],
        rhs_contracting_axes: &[usize],
        output_axes: &[usize],
        dst_codomain_rank: usize,
    ) -> Result<Self, CheckedFusionSpaceError>
    where
        R: CheckedFusionAlgebra,
    {
        OrientedFusionTreeHomSpace::try_tensorcontract_homspace_checked(
            rule,
            OrientedFusionTreeHomSpace::new(lhs, FusionTreePairOrientation::Direct),
            OrientedFusionTreeHomSpace::new(rhs, FusionTreePairOrientation::Direct),
            lhs_contracting_axes,
            rhs_contracting_axes,
            output_axes,
            dst_codomain_rank,
        )
    }

    /// Whether [`Self::tensorcontract_homspace`] equals `expected`; see
    /// [`OrientedFusionTreeHomSpace::tensorcontract_homspace_matches`].
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn tensorcontract_homspace_matches<R>(
        rule: &R,
        lhs: &Self,
        rhs: &Self,
        lhs_contracting_axes: &[usize],
        rhs_contracting_axes: &[usize],
        output_axes: &[usize],
        dst_codomain_rank: usize,
        expected: &Self,
    ) -> Result<bool, CoreError>
    where
        R: FusionRule,
    {
        OrientedFusionTreeHomSpace::tensorcontract_homspace_matches(
            rule,
            OrientedFusionTreeHomSpace::new(lhs, FusionTreePairOrientation::Direct),
            OrientedFusionTreeHomSpace::new(rhs, FusionTreePairOrientation::Direct),
            lhs_contracting_axes,
            rhs_contracting_axes,
            output_axes,
            dst_codomain_rank,
            expected,
        )
    }

    /// Checked sibling of [`Self::tensorcontract_homspace_matches`].
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn try_tensorcontract_homspace_matches_checked<R>(
        rule: &R,
        lhs: &Self,
        rhs: &Self,
        lhs_contracting_axes: &[usize],
        rhs_contracting_axes: &[usize],
        output_axes: &[usize],
        dst_codomain_rank: usize,
        expected: &Self,
    ) -> Result<bool, CheckedFusionSpaceError>
    where
        R: CheckedFusionAlgebra,
    {
        OrientedFusionTreeHomSpace::try_tensorcontract_homspace_matches_checked(
            rule,
            OrientedFusionTreeHomSpace::new(lhs, FusionTreePairOrientation::Direct),
            OrientedFusionTreeHomSpace::new(rhs, FusionTreePairOrientation::Direct),
            lhs_contracting_axes,
            rhs_contracting_axes,
            output_axes,
            dst_codomain_rank,
            expected,
        )
    }

    /// Checked Generic sibling of [`Self::tensorcontract_homspace`].
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn try_tensorcontract_homspace_generic_checked<R>(
        rule: &R,
        lhs: &Self,
        rhs: &Self,
        lhs_contracting_axes: &[usize],
        rhs_contracting_axes: &[usize],
        output_axes: &[usize],
        dst_codomain_rank: usize,
    ) -> Result<Self, CheckedGenericStructureError<R::Error>>
    where
        R: CheckedGenericFusion,
    {
        OrientedFusionTreeHomSpace::try_tensorcontract_homspace_generic_checked(
            rule,
            OrientedFusionTreeHomSpace::new(lhs, FusionTreePairOrientation::Direct),
            OrientedFusionTreeHomSpace::new(rhs, FusionTreePairOrientation::Direct),
            lhs_contracting_axes,
            rhs_contracting_axes,
            output_axes,
            dst_codomain_rank,
        )
    }

    /// The cached fusion-tree block keys, shared in O(1) (`Arc::clone`): the
    /// layout already holds them as `Arc<[_]>`, so there is no need to deep-clone
    /// each key (two `FusionTreeKey`s, four `SectorVec`s each) into a fresh `Vec`
    /// on every call. Returns `Arc<[_]>`, which derefs to `[FusionTreePairKey]`,
    /// so iterate / index / `len` callers are unchanged; by-value consumers can
    /// `.to_vec()`. TensorKit's `fusiontrees(W)` likewise returns the cached
    /// index set by reference. See #53.
    pub fn fusion_tree_keys<R>(&self, rule: &R) -> Arc<[FusionTreePairKey]>
    where
        R: MultiplicityFreeFusionRule,
    {
        Arc::clone(&self.cached_fusion_tree_layout(rule).keys)
    }

    /// Stages multiplicity-free layout metadata without publishing it.
    #[doc(hidden)]
    pub fn prepare_fusion_tree_layout<R>(&self, rule: &R) -> PreparedFusionTreeLayout
    where
        R: MultiplicityFreeFusionRule,
    {
        self.prepare_fusion_tree_layout_with(rule, core_reset_epoch(), || {
            Ok::<_, std::convert::Infallible>(self.fusion_tree_layout_data_uncached(rule))
        })
        .unwrap_or_else(|error| match error {})
    }

    /// Stages checked layout metadata for an external multiplicity-free
    /// provider without issuing a layout ID or changing cache accounting.
    ///
    /// Every HomSpace leg sector is validated through [`CheckedFusionAlgebra::try_dual_sector`]
    /// before any cache lookup, because a rigid multiplicity-free provider
    /// requires a representable dual for every leg it admits. Enumeration then
    /// reuses the same [`Self::prepare_fusion_tree_layout_with`] staging and
    /// bounded layout cache as the built-in paths.
    #[doc(hidden)]
    pub fn prepare_fusion_tree_layout_checked<R>(
        &self,
        rule: &R,
    ) -> Result<PreparedFusionTreeLayout, FusionAlgebraError>
    where
        R: MultiplicityFreeFusionRule + CheckedFusionAlgebra,
    {
        self.prepare_fusion_tree_layout_checked_at(rule, core_reset_epoch())
    }

    /// Derives a HomSpace and prepares its checked MF layout in one publication epoch.
    #[doc(hidden)]
    pub fn prepare_fusion_tree_layout_checked_with<R, E>(
        rule: &R,
        build_homspace: impl FnOnce() -> Result<Self, E>,
    ) -> Result<(Self, PreparedFusionTreeLayout), E>
    where
        R: MultiplicityFreeFusionRule + CheckedFusionAlgebra,
        E: From<FusionAlgebraError>,
    {
        let epoch = core_reset_epoch();
        let homspace = build_homspace()?;
        let prepared = homspace.prepare_fusion_tree_layout_checked_at(rule, epoch)?;
        Ok((homspace, prepared))
    }

    fn prepare_fusion_tree_layout_checked_at<R>(
        &self,
        rule: &R,
        epoch: usize,
    ) -> Result<PreparedFusionTreeLayout, FusionAlgebraError>
    where
        R: MultiplicityFreeFusionRule + CheckedFusionAlgebra,
    {
        for leg in self.codomain().legs().iter().chain(self.domain().legs()) {
            for &sector in leg.sectors() {
                rule.try_dual_sector(sector)?;
            }
        }
        self.prepare_fusion_tree_layout_with(rule, epoch, || {
            self.try_fusion_tree_layout_data_uncached_checked(rule)
        })
    }

    fn prepare_fusion_tree_layout_with<R, E, F>(
        &self,
        rule: &R,
        complete_epoch: usize,
        build: F,
    ) -> Result<PreparedFusionTreeLayout, E>
    where
        R: MultiplicityFreeFusionRule,
        F: FnOnce() -> Result<FusionTreeHomSpaceLayoutData, E>,
    {
        let key = FusionTreeHomSpaceCacheKey::new(rule, self);
        if let Some(layout) = sector_structure_cache().get(&key) {
            return Ok(PreparedFusionTreeLayout {
                complete_epoch,
                state: PreparedFusionTreeLayoutState::Cached { key, layout },
            });
        }

        let data = build()?;
        Ok(PreparedFusionTreeLayout {
            complete_epoch,
            state: PreparedFusionTreeLayoutState::Cold { key, data },
        })
    }

    pub fn try_for_each_fusion_tree_key<R, F, E>(&self, rule: &R, mut f: F) -> Result<(), E>
    where
        R: MultiplicityFreeFusionRule,
        F: FnMut(&FusionTreePairKey) -> Result<(), E>,
    {
        let layout = self.cached_fusion_tree_layout(rule);
        for key in layout.keys.iter() {
            f(key)?;
        }
        Ok(())
    }

    pub fn try_with_fusion_tree_keys<R, F, T, E>(&self, rule: &R, f: F) -> Result<T, E>
    where
        R: MultiplicityFreeFusionRule,
        F: FnOnce(&[FusionTreePairKey]) -> Result<T, E>,
    {
        let layout = self.cached_fusion_tree_layout(rule);
        f(layout.keys.as_ref())
    }

    pub(crate) fn cached_fusion_tree_layout<R>(&self, rule: &R) -> Arc<FusionTreeHomSpaceLayout>
    where
        R: MultiplicityFreeFusionRule,
    {
        self.prepare_fusion_tree_layout(rule).commit_layout()
    }

    pub fn coupled_subblock_structure<R, Shapes>(
        &self,
        rule: &R,
        nout: usize,
        shapes: Shapes,
    ) -> Result<Arc<BlockStructure>, CoreError>
    where
        R: MultiplicityFreeFusionRule,
        Shapes: IntoIterator,
        Shapes::Item: Into<Vec<usize>>,
    {
        let layout = self.cached_fusion_tree_layout(rule);
        let rank = self.codomain().len() + self.domain().len();
        let shapes = shapes
            .into_iter()
            .map(|shape| shape.into().into_iter().collect::<DimVec>())
            .collect::<Vec<_>>();
        if layout.keys.len() != shapes.len() {
            return Err(CoreError::BlockCountMismatch {
                expected: layout.keys.len(),
                actual: shapes.len(),
            });
        }
        self.validate_degeneracy_shapes(layout.keys.as_ref(), &shapes)?;
        if nout == self.codomain().len() {
            // The shapes equal the leg degeneracies, so the canonical
            // leg-derived structure (and its bounded cache) is the answer.
            return self.coupled_subblock_structure_from_leg_degeneracies(rule);
        }
        // Why not reject a row/column split other than the codomain rank: it
        // is an accepted (uncached) layout of this public constructor.
        let specs = coupled_sector_matrix_block_specs(nout, rank, layout.keys.as_ref(), &shapes)?;
        Ok(BlockStructure::from_blocks_with_rank(rank, specs)?.into_shared())
    }

    /// Builds the canonical coupled-sector layout directly from this hom
    /// space's authoritative per-leg degeneracies.
    ///
    /// Unlike [`Self::coupled_subblock_structure`], callers do not first
    /// materialize one owned shape per tree pair. The miss builder derives each
    /// codomain row and domain column once before forming final subblocks.
    pub fn coupled_subblock_structure_from_leg_degeneracies<R>(
        &self,
        rule: &R,
    ) -> Result<Arc<BlockStructure>, CoreError>
    where
        R: MultiplicityFreeFusionRule,
    {
        let epoch = core_reset_epoch();
        let key = CompleteHomSpaceStructureCacheKey::new(rule, self);
        if let Some(entry) = complete_hom_space_structure_cached(&key) {
            return Ok(entry.structure());
        }

        let layout = self.cached_fusion_tree_layout(rule);
        let (sector, degeneracy) = coupled_subblock_parts_from_leg_degeneracies(self, &layout)?;
        let built = BlockStructure::from_shared_parts(sector, degeneracy)?;
        built.record_storage_tiling();
        Ok(admit_complete_hom_space_structure(key, built.into_shared(), epoch).1)
    }

    /// The canonical complete layout together with the HomSpace backing owned
    /// by the same cache entry.
    #[doc(hidden)]
    pub fn canonical_coupled_subblock_structure_from_leg_degeneracies<R>(
        self,
        rule: &R,
    ) -> Result<(Self, Arc<BlockStructure>), CoreError>
    where
        R: MultiplicityFreeFusionRule,
    {
        let epoch = core_reset_epoch();
        let key = CompleteHomSpaceStructureCacheKey::new(rule, &self);
        let (entry, structure) = if let Some(entry) = complete_hom_space_structure_cached(&key) {
            let structure = entry.structure();
            (entry, structure)
        } else {
            let layout = self.cached_fusion_tree_layout(rule);
            let (sector, degeneracy) =
                coupled_subblock_parts_from_leg_degeneracies(&self, &layout)?;
            let built = BlockStructure::from_shared_parts(sector, degeneracy)?;
            built.record_storage_tiling();
            admit_complete_hom_space_structure(key, built.into_shared(), epoch)
        };
        Ok((self.with_canonical_id(entry.homspace_id()), structure))
    }

    #[doc(hidden)]
    pub fn coupled_subblock_layout_probe_uncached<R>(
        &self,
        rule: &R,
        source: &BlockStructure,
    ) -> Result<(usize, bool), CoreError>
    where
        R: MultiplicityFreeFusionRule,
    {
        let layout = self.fusion_tree_layout_data_uncached(rule);
        let (sector, degeneracy) = coupled_subblock_parts_from_leg_degeneracies(self, &layout)?;
        let required_len = degeneracy.required_len()?;
        Ok((
            required_len,
            source.sector_structure() == &*sector && source.degeneracy_structure() == &degeneracy,
        ))
    }

    /// Multiplicity-aware sibling of
    /// [`Self::coupled_subblock_structure_from_leg_degeneracies`].
    ///
    /// Generic layouts share the sector-structure cache under their own key
    /// (vertex-resolved keys differ from multiplicity-free ones) and feed the
    /// same single-pass degeneracy builder.
    pub fn coupled_subblock_structure_from_leg_degeneracies_generic<R>(
        &self,
        rule: &R,
    ) -> Result<Arc<BlockStructure>, CoreError>
    where
        R: FusionRule,
    {
        self.clone()
            .canonical_coupled_subblock_structure_from_leg_degeneracies_generic(rule)
            .map(|(_, structure)| structure)
    }

    /// Canonical complete Generic layout and its cache-owned HomSpace backing.
    #[doc(hidden)]
    pub fn canonical_coupled_subblock_structure_from_leg_degeneracies_generic<R>(
        self,
        rule: &R,
    ) -> Result<(Self, Arc<BlockStructure>), CoreError>
    where
        R: FusionRule,
    {
        let epoch = core_reset_epoch();
        let rule_identity = rule.rule_identity();
        let layout = self.generic_sector_layout(rule_identity.clone(), || {
            self.fusion_tree_layout_data_generic(rule)
        })?;
        let key = CompleteHomSpaceStructureCacheKey::generic(rule_identity, &self);
        let (entry, structure) = if let Some(entry) = complete_hom_space_structure_cached(&key) {
            let structure = entry.structure();
            (entry, structure)
        } else {
            let (sector, degeneracy) =
                coupled_subblock_parts_from_leg_degeneracies(&self, &layout)?;
            let built = BlockStructure::from_shared_parts(sector, degeneracy)?;
            built.record_storage_tiling();
            admit_complete_hom_space_structure(key, built.into_shared(), epoch)
        };
        Ok((self.with_canonical_id(entry.homspace_id()), structure))
    }

    /// Checked Generic-fusion structural staging from this HomSpace's leg
    /// degeneracies. A failed provider walk is never cached; a successful one
    /// may be answered by the sector-structure cache, since the layout is a
    /// pure function of the rule identity and the sector signature.
    pub fn coupled_subblock_structure_from_leg_degeneracies_generic_checked<R>(
        &self,
        rule: &R,
    ) -> Result<Arc<BlockStructure>, CheckedGenericStructureError<R::Error>>
    where
        R: CheckedGenericFusion,
    {
        let prepared = self
            .clone()
            .prepare_complete_coupled_subblock_structure_generic_checked_after(rule, |_| Ok(()))?;
        Ok(prepared.commit_with_complete_homspace().1)
    }

    /// Stage checked Generic block metadata without entering the block
    /// interner. This is the transaction input used by owned checked
    /// transforms before their categorical plan is known to succeed.
    #[doc(hidden)]
    pub fn prepare_coupled_subblock_structure_from_leg_degeneracies_generic_checked<R>(
        &self,
        rule: &R,
    ) -> Result<PreparedBlockStructure, CheckedGenericStructureError<R::Error>>
    where
        R: CheckedGenericFusion,
    {
        let epoch = core_reset_epoch();
        let rule_identity = rule.rule_identity();
        let layout = self.generic_sector_layout(rule_identity.clone(), || {
            self.fusion_tree_layout_data_generic_checked(rule)
        })?;
        let (sector, degeneracy) = coupled_subblock_parts_from_leg_degeneracies(self, &layout)?;
        let candidate = PreparedBlockStructure::from_shared_parts(sector, degeneracy)
            .map(PreparedBlockStructure::with_storage_tiling)?;
        Ok(PreparedBlockStructure::complete_miss(
            self.clone(),
            candidate,
            CompleteHomSpaceStructureCacheKey::generic(rule_identity, self),
            epoch,
            true,
        ))
    }

    /// Begins one complete checked Generic layout transaction. The reset epoch
    /// is captured before the provider identity is read and handed to `admit`;
    /// checked sector admission then precedes the complete-cache probe. A hit reuses
    /// its canonical HomSpace/structure pair without building degeneracy
    /// metadata, while a miss carries one unpublished candidate to commit.
    #[doc(hidden)]
    pub fn prepare_complete_coupled_subblock_structure_generic_checked_after<R, A>(
        self,
        rule: &R,
        admit: A,
    ) -> Result<PreparedBlockStructure, CheckedGenericStructureError<R::Error>>
    where
        R: CheckedGenericFusion,
        A: FnOnce(&RuleIdentity) -> Result<(), CheckedGenericStructureError<R::Error>>,
    {
        Self::prepare_complete_coupled_subblock_structure_generic_checked_with(rule, |identity| {
            admit(identity)?;
            Ok(self)
        })
    }

    /// Begins one complete checked Generic layout transaction whose HomSpace
    /// is itself produced by checked operation admission. The reset epoch is
    /// captured before any provider query. `build_homspace` receives the
    /// provider identity that also keys the structure caches, so admission
    /// inside it compares against that one value instead of building another
    /// (#2046).
    #[doc(hidden)]
    pub fn prepare_complete_coupled_subblock_structure_generic_checked_with<R, E, B>(
        rule: &R,
        build_homspace: B,
    ) -> Result<PreparedBlockStructure, E>
    where
        R: CheckedGenericFusion,
        E: From<CheckedGenericStructureError<R::Error>>,
        B: FnOnce(&RuleIdentity) -> Result<Self, E>,
    {
        let epoch = core_reset_epoch();
        let rule_identity = rule.rule_identity();
        let homspace = build_homspace(&rule_identity)?;
        homspace
            .prepare_complete_coupled_subblock_structure_generic_checked_at(
                rule,
                rule_identity,
                epoch,
            )
            .map_err(E::from)
    }

    /// The complete checked Generic structure of `self` permuted by
    /// `(codomain_axes, domain_axes)`, where `source` is the structure stored
    /// over `self`. Order: reset epoch, one identity read, `admit`; then, if
    /// `source`'s resident cache entry memoizes this permutation and the
    /// destination's own entry is resident, that hit with no further
    /// provider query ([`Permuted::Memo`]); otherwise `before_derive`, the
    /// provider derivation and the usual sector-structure / degeneracy-structure cache tail
    /// ([`Permuted::Derived`], whose ticket publishes after the caller's
    /// commit).
    ///
    /// Why the memo hit skips the sector-structure cache: the destination's sector admission was
    /// proven when the memo was published, which required its complete entry.
    #[doc(hidden)]
    pub fn prepare_complete_permuted_generic_checked<R, E>(
        &self,
        source: &BlockStructure,
        rule: &R,
        codomain_axes: &[usize],
        domain_axes: &[usize],
        admit: impl FnOnce(&RuleIdentity) -> Result<(), E>,
        before_derive: impl FnOnce() -> Result<(), E>,
    ) -> Result<(PreparedBlockStructure, Self, Permuted), E>
    where
        R: CheckedGenericFusion,
        E: From<CheckedGenericStructureError<R::Error>>,
    {
        let epoch = core_reset_epoch();
        let rule_identity = rule.rule_identity();
        admit(&rule_identity)?;
        let mut axes = SmallVec::<[usize; 9]>::new();
        axes.push(codomain_axes.len());
        axes.extend_from_slice(codomain_axes);
        axes.extend_from_slice(domain_axes);
        let entry = source.complete_entry();
        if let Some(id) = entry
            .as_ref()
            .and_then(|entry| entry.permuted(&rule_identity, self, &axes))
        {
            let destination = Self {
                content: Arc::clone(&id.content),
                id: OnceLock::from(id),
            };
            let key =
                CompleteHomSpaceStructureCacheKey::generic(rule_identity.clone(), &destination);
            if let Some(resident) = complete_hom_space_structure_cached(&key) {
                #[cfg(test)]
                crate::probe::hit(crate::probe::Event::PermutedMemoHit);
                let prepared = PreparedBlockStructure::complete_hit(
                    destination.clone(),
                    resident.homspace_id(),
                    resident.structure(),
                );
                return Ok((prepared, destination, Permuted::Memo));
            }
        }
        before_derive()?;
        let destination = self.try_permute_generic_checked(rule, codomain_axes, domain_axes)?;
        let prepared = destination
            .clone()
            .prepare_complete_coupled_subblock_structure_generic_checked_at(
                rule,
                rule_identity,
                epoch,
            )?;
        let ticket = PermutedMemoTicket { entry, axes, epoch };
        Ok((prepared, destination, Permuted::Derived(ticket)))
    }

    fn prepare_complete_coupled_subblock_structure_generic_checked_at<R>(
        self,
        rule: &R,
        rule_identity: RuleIdentity,
        epoch: usize,
    ) -> Result<PreparedBlockStructure, CheckedGenericStructureError<R::Error>>
    where
        R: CheckedGenericFusion,
    {
        let layout = self.generic_sector_layout(rule_identity.clone(), || {
            self.fusion_tree_layout_data_generic_checked(rule)
        })?;
        let key = CompleteHomSpaceStructureCacheKey::generic(rule_identity, &self);
        if let Some(entry) = complete_hom_space_structure_cached(&key) {
            return Ok(PreparedBlockStructure::complete_hit(
                self,
                entry.homspace_id(),
                entry.structure(),
            ));
        }
        let (sector, degeneracy) = coupled_subblock_parts_from_leg_degeneracies(&self, &layout)?;
        let candidate = PreparedBlockStructure::from_shared_parts(sector, degeneracy)
            .map(PreparedBlockStructure::with_storage_tiling)?;
        Ok(PreparedBlockStructure::complete_miss(
            self, candidate, key, epoch, false,
        ))
    }

    #[cfg(test)]
    pub(crate) fn degeneracy_shape_for_key(
        &self,
        key: &FusionTreePairKey,
    ) -> Result<DimVec, CoreError> {
        let rank = self.rank();
        if key.codomain_uncoupled().len() != self.codomain().len()
            || key.domain_uncoupled().len() != self.domain().len()
        {
            return Err(CoreError::StructureRankMismatch {
                expected: rank,
                actual: key.codomain_uncoupled().len() + key.domain_uncoupled().len(),
            });
        }
        let mut shape = DimVec::new();
        for (leg, &sector) in self
            .codomain()
            .legs()
            .iter()
            .chain(self.domain().legs())
            .zip(
                key.codomain_uncoupled()
                    .iter()
                    .chain(key.domain_uncoupled()),
            )
        {
            shape.push(
                leg.degeneracy(sector)
                    .ok_or(CoreError::MalformedFusionTree {
                        message: "fusion tree uses a sector absent from its leg",
                    })?,
            );
        }
        Ok(shape)
    }

    #[cfg(test)]
    pub(crate) fn fusion_tree_keys_uncached<R>(&self, rule: &R) -> Vec<FusionTreePairKey>
    where
        R: MultiplicityFreeFusionRule,
    {
        let codomain = fusion_trees_by_coupled_for_space(rule, self.codomain());
        let domain = fusion_trees_by_coupled_for_space(rule, self.domain());
        let mut keys = Vec::new();
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
                    for domain_tree in &domain[domain_index].trees {
                        for codomain_tree in &codomain[codomain_index].trees {
                            keys.push(FusionTreePairKey::pair(
                                codomain_tree.clone(),
                                domain_tree.clone(),
                            ));
                        }
                    }
                    codomain_index += 1;
                    domain_index += 1;
                }
            }
        }
        keys
    }

    pub(crate) fn fusion_tree_layout_data_uncached<R>(
        &self,
        rule: &R,
    ) -> FusionTreeHomSpaceLayoutData
    where
        R: MultiplicityFreeFusionRule,
    {
        // Enumerate the side with fewer legs first and the other only for
        // the coupled sectors that side reaches: the layout pairs only
        // shared coupled sectors.
        let coupled_of = |groups: &[CoupledFusionTrees]| {
            groups
                .iter()
                .filter(|group| !group.trees.is_empty())
                .map(|group| group.coupled)
                .collect::<Vec<_>>()
        };
        let (codomain, domain) = if self.codomain().len() <= self.domain().len() {
            let codomain = fusion_trees_by_coupled_for_space(rule, self.codomain());
            let domain = fusion_trees_by_coupled_for_space_within(
                rule,
                self.domain(),
                Some(&coupled_of(&codomain)),
            );
            (codomain, domain)
        } else {
            let domain = fusion_trees_by_coupled_for_space(rule, self.domain());
            let codomain = fusion_trees_by_coupled_for_space_within(
                rule,
                self.codomain(),
                Some(&coupled_of(&domain)),
            );
            (codomain, domain)
        };
        fusion_tree_layout_data_from_groups(self.rank(), &codomain, &domain)
    }

    pub(crate) fn try_fusion_tree_layout_data_uncached_checked<R>(
        &self,
        rule: &R,
    ) -> Result<FusionTreeHomSpaceLayoutData, FusionAlgebraError>
    where
        R: MultiplicityFreeFusionRule + CheckedFusionAlgebra,
    {
        let codomain = try_fusion_trees_by_coupled_for_space_checked(rule, self.codomain())?;
        let domain = try_fusion_trees_by_coupled_for_space_checked(rule, self.domain())?;
        Ok(fusion_tree_layout_data_from_groups(
            self.rank(),
            &codomain,
            &domain,
        ))
    }

    /// Generic-fusion (outer-multiplicity) sibling of [`Self::fusion_tree_keys`]:
    /// enumerates multiplicity-aware block keys (codomain × domain tree pairs)
    /// for a `FusionRule` whose `nsymbol` can exceed 1. Not cached —
    /// the Generic path is not on any hot loop (mirrors the non-memoized
    /// add a cache only when a Generic workload measures a benefit.
    ///
    /// Bounded-table semantics (Option A, refute/b3b-verify): the result is
    /// either the exact provider-defined block key set or an `Err` — never a silently
    /// truncated one. Full-space enumeration errs as soon as either side's
    /// coupled fold reports escaped candidates, tainted sectors, or a poisoned
    /// (beyond one-hop) fold, even when the offending sectors could not survive
    /// the codomain∩domain merge — conservative on purpose; use
    /// [`Self::fusion_tree_keys_generic_for_coupled`] for a single provably
    /// clean sector. Unbounded Generic rules never err.
    pub fn fusion_tree_keys_generic<R>(&self, rule: &R) -> Result<Vec<FusionTreePairKey>, CoreError>
    where
        R: FusionRule,
    {
        Ok(self.fusion_tree_layout_data_generic(rule)?.keys.to_vec())
    }

    /// Checked Generic-fusion sibling of [`Self::fusion_tree_keys_generic`].
    ///
    /// All channel, dual, multiplicity, bounded-table, and fold queries finish
    /// before this method returns keys. Generic layouts are ephemeral, so a
    /// provider error cannot retain a layout-cache or complete-space entry.
    pub fn fusion_tree_keys_generic_checked<R>(
        &self,
        rule: &R,
    ) -> Result<Vec<FusionTreePairKey>, CheckedGenericStructureError<R::Error>>
    where
        R: CheckedGenericFusion,
    {
        Ok(self
            .fusion_tree_layout_data_generic_checked(rule)?
            .keys
            .to_vec())
    }

    /// The Generic layout through the sector-structure cache: a pure
    /// function of the rule identity and the sector signature, so a hit
    /// proves that an equal provider walk succeeded. A failed walk is not
    /// cached.
    ///
    /// Concurrent misses of one key build once (single-flight). That is safe
    /// only because `build` calls nothing but the provider's fusion
    /// enumeration (`try_coupled_sector_fold`, `try_fusion_channels`,
    /// `try_nsymbol`): no Clebsch-Gordan solve, no linear algebra and no
    /// Rayon region, so a thread holding the placeholder never waits inside
    /// a pool where work-stealing could run another caller of the same key on
    /// it. (fusion-symbols 0.3.0's SU(N) multiplicities are Gelfand-Tsetlin pattern
    /// counts, `sun.rs:shared_directproduct`, with no tenferro call.) Keep it
    /// so: a build that parallelizes must use `publish` instead, as the
    /// completed-transformer cache does.
    fn generic_sector_layout<E>(
        &self,
        rule: RuleIdentity,
        build: impl FnOnce() -> Result<FusionTreeHomSpaceLayoutData, E>,
    ) -> Result<Arc<FusionTreeHomSpaceLayout>, E> {
        let key = FusionTreeHomSpaceCacheKey::generic(rule, self);
        sector_structure_cache().get_or_try_build(&key, core_reset_epoch(), || {
            let data = build()?;
            let bytes = charged_fusion_tree_layout_bytes(&key, &data);
            Ok((Arc::new(FusionTreeHomSpaceLayout::new(data)), bytes))
        })
    }

    fn fusion_tree_layout_data_generic<R>(
        &self,
        rule: &R,
    ) -> Result<FusionTreeHomSpaceLayoutData, CoreError>
    where
        R: FusionRule,
    {
        let checked = InfallibleGeneric::new(rule);
        match self.fusion_tree_layout_data_generic_checked(&checked) {
            Ok(data) => Ok(data),
            Err(CheckedGenericStructureError::Provider(never)) => match never {},
            Err(CheckedGenericStructureError::Core(error)) => Err(error),
        }
    }

    fn fusion_tree_layout_data_generic_checked<R>(
        &self,
        rule: &R,
    ) -> Result<FusionTreeHomSpaceLayoutData, CheckedGenericStructureError<R::Error>>
    where
        R: CheckedGenericFusion,
    {
        let (codomain, codomain_fold) =
            fusion_trees_by_coupled_for_space_generic_checked(rule, self.codomain())?;
        let (domain, domain_fold) =
            fusion_trees_by_coupled_for_space_generic_checked(rule, self.domain())?;
        for (side, fold) in [("codomain", &codomain_fold), ("domain", &domain_fold)] {
            if !fold.is_fully_clean() {
                return Err(CoreError::FusionOutsideTable {
                    message: fusion_fold_error_message(side, fold),
                }
                .into());
            }
        }
        Ok(fusion_tree_layout_data_from_groups(
            self.rank(),
            &codomain,
            &domain,
        ))
    }

    /// Block keys of ONE coupled sector, for spaces whose full enumeration is
    /// an `Err` but whose requested sector is provably clean (its complete
    /// complete tree set stays inside the provider catalog). `Err` when the sector is
    /// tainted on either side (its trees would need out-of-table intermediates)
    /// or the fold is poisoned; `Ok(vec![])` when the sector is simply not a
    /// shared coupled candidate.
    pub fn fusion_tree_keys_generic_for_coupled<R>(
        &self,
        rule: &R,
        coupled: SectorId,
    ) -> Result<Vec<FusionTreePairKey>, CoreError>
    where
        R: FusionRule,
    {
        let (codomain, codomain_fold) =
            fusion_trees_by_coupled_for_space_generic(rule, self.codomain());
        let (domain, domain_fold) = fusion_trees_by_coupled_for_space_generic(rule, self.domain());
        generic_keys_for_coupled_from_groups(
            &codomain,
            &codomain_fold,
            &domain,
            &domain_fold,
            coupled,
            |side, fold, coupled| {
                CoreError::FusionOutsideTable {
                message: format!(
                    "coupled sector {coupled:?} on the {side} side requires out-of-catalog intermediates. {}",
                    fusion_fold_error_message(side, fold),
                ),
            }
            },
        )
    }

    /// Checked counterpart of [`Self::fusion_tree_keys_generic_for_coupled`].
    pub fn fusion_tree_keys_generic_for_coupled_checked<R>(
        &self,
        rule: &R,
        coupled: SectorId,
    ) -> Result<Vec<FusionTreePairKey>, CheckedGenericStructureError<R::Error>>
    where
        R: CheckedGenericFusion,
    {
        let (codomain, codomain_fold) =
            fusion_trees_by_coupled_for_space_generic_checked(rule, self.codomain())?;
        let (domain, domain_fold) =
            fusion_trees_by_coupled_for_space_generic_checked(rule, self.domain())?;
        generic_keys_for_coupled_from_groups(
            &codomain,
            &codomain_fold,
            &domain,
            &domain_fold,
            coupled,
            |side, fold, coupled| CoreError::FusionOutsideTable {
                message: format!(
                    "generic fusion provider cannot exactly represent coupled sector {coupled:?} on the {side} side ({fold:?}); block dimensions are either exact or an error, never truncated.",
                ),
            },
        )
        .map_err(Into::into)
    }

    pub fn sector_structure<R>(&self, rule: &R) -> Result<SectorStructure, CoreError>
    where
        R: MultiplicityFreeFusionRule,
    {
        let rank = self.codomain().len() + self.domain().len();
        // `from_keys` builds owned `BlockKey`s, so cloning out of the shared
        // slice is unavoidable here (a cold structure-build path, not a hot loop).
        SectorStructure::from_keys(rank, self.fusion_tree_keys(rule).iter().cloned())
    }

    pub fn unique_fusion_tree_key_from_external_sectors<R>(
        &self,
        rule: &R,
        sectors: &[SectorId],
    ) -> Result<FusionTreePairKey, CoreError>
    where
        R: MultiplicityFreeFusionRule,
    {
        let mut keys = self.fusion_tree_keys_from_external_sectors(rule, sectors)?;
        if keys.len() != 1 {
            return Err(CoreError::BlockCountMismatch {
                expected: 1,
                actual: keys.len(),
            });
        }
        Ok(keys.remove(0))
    }

    /// Lowers external per-leg sectors to fusion-tree keys. Domain-side
    /// external sectors are dualized into internal tree sectors here, the
    /// same convention TensorKit applies in `subblock(t, sectors)`.
    pub fn fusion_tree_keys_from_external_sectors<R>(
        &self,
        rule: &R,
        sectors: &[SectorId],
    ) -> Result<Vec<FusionTreePairKey>, CoreError>
    where
        R: MultiplicityFreeFusionRule,
    {
        let rank = self.codomain().len() + self.domain().len();
        if sectors.len() != rank {
            return Err(CoreError::DimensionMismatch {
                expected: rank,
                actual: sectors.len(),
            });
        }

        let codomain = fusion_trees_by_coupled_for_selected_space(
            rule,
            self.codomain(),
            &sectors[..self.codomain().len()],
        )?;
        let domain_sectors = sectors[self.codomain().len()..]
            .iter()
            .map(|&sector| rule.dual(sector))
            .collect::<Vec<_>>();
        let domain =
            fusion_trees_by_coupled_for_selected_space(rule, self.domain(), &domain_sectors)?;
        let mut keys = Vec::new();
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
                    for domain_tree in &domain[domain_index].trees {
                        for codomain_tree in &codomain[codomain_index].trees {
                            keys.push(FusionTreePairKey::pair(
                                codomain_tree.clone(),
                                domain_tree.clone(),
                            ));
                        }
                    }
                    codomain_index += 1;
                    domain_index += 1;
                }
            }
        }

        Ok(keys)
    }

    /// Validates per-tree degeneracy shapes against the leg-level
    /// degeneracies (the legs are authoritative): for every tree key,
    /// `shape[axis]` must equal the degeneracy the axis' leg stores for the
    /// tree's uncoupled sector on that axis.
    pub fn validate_degeneracy_shapes<S>(
        &self,
        keys: &[FusionTreePairKey],
        shapes: &[S],
    ) -> Result<(), CoreError>
    where
        S: AsRef<[usize]>,
    {
        let legs = self
            .codomain()
            .legs()
            .iter()
            .chain(self.domain().legs())
            .collect::<Vec<_>>();
        for (key, shape) in keys.iter().zip(shapes) {
            let shape = shape.as_ref();
            if shape.len() != legs.len() {
                return Err(CoreError::StructureRankMismatch {
                    expected: legs.len(),
                    actual: shape.len(),
                });
            }
            let uncoupled = key
                .codomain_uncoupled()
                .iter()
                .chain(key.domain_uncoupled());
            for ((leg, &sector), &dim) in legs.iter().zip(uncoupled).zip(shape) {
                let expected = leg
                    .degeneracy(sector)
                    .ok_or(CoreError::MalformedFusionTree {
                        message: "fusion tree uses a sector absent from its leg",
                    })?;
                if expected != dim {
                    return Err(CoreError::LegDegeneracyMismatch {
                        sector,
                        expected,
                        actual: dim,
                    });
                }
            }
        }
        Ok(())
    }

    /// Validates every present fusion-tree block against this HomSpace.
    ///
    /// Missing valid tree pairs are structural zeros. Block order, strides,
    /// offsets, and overlap are storage properties and are intentionally not
    /// part of this categorical subset proof.
    #[doc(hidden)]
    pub fn validate_subblock_structure_subset<R>(
        &self,
        rule: &R,
        structure: &BlockStructure,
    ) -> Result<(), CoreError>
    where
        R: FusionRule,
    {
        StructurallyValidatedFusionTreeSubset::try_new(self, structure)?.validate_for_rule(rule)
    }

    /// Checked finite-algebra sibling of [`Self::validate_subblock_structure_subset`].
    #[doc(hidden)]
    pub fn validate_subblock_structure_subset_checked<R>(
        &self,
        rule: &R,
        structure: &BlockStructure,
    ) -> Result<(), CheckedFusionSpaceError>
    where
        R: CheckedFusionAlgebra,
    {
        StructurallyValidatedFusionTreeSubset::try_new(self, structure)?
            .validate_for_rule_checked(rule)
    }

    pub fn fusion_tree_groups<R>(&self, rule: &R) -> Result<Vec<FusionTreeBlockGroup>, CoreError>
    where
        R: MultiplicityFreeFusionRule,
    {
        self.sector_structure(rule)
            .map(SectorStructure::into_fusion_tree_groups)
    }

    /// External leg view of flat axis `axis` (TensorKit's `space(t, i)`
    /// convention, homspace.jl:60-62): codomain legs verbatim, domain legs
    /// dualized. Degeneracies are carried along, keyed by the external
    /// (placement-invariant) sector labels.
    pub fn external_axis_leg<R>(&self, rule: &R, axis: usize) -> SectorLeg
    where
        R: FusionRule,
    {
        if axis < self.codomain().len() {
            self.codomain().legs()[axis].clone()
        } else {
            dual_sector_leg(rule, &self.domain().legs()[axis - self.codomain().len()])
        }
    }

    fn external_axis_leg_view(&self, axis: usize) -> OrientedLegView<'_> {
        if axis < self.codomain().len() {
            OrientedLegView::borrowed(&self.codomain().legs()[axis])
        } else {
            OrientedLegView::borrowed(&self.domain().legs()[axis - self.codomain().len()]).toggled()
        }
    }

    /// Returns the duality flag in the external-axis convention. Why not read
    /// a domain leg verbatim: external domain axes are duals of their stored
    /// hom-space legs, matching [`Self::external_axis_leg`].
    pub fn external_axis_is_dual(&self, axis: usize) -> Option<bool> {
        if axis < self.codomain().len() {
            Some(self.codomain().legs()[axis].is_dual())
        } else if axis < self.rank() {
            Some(!self.domain().legs()[axis - self.codomain().len()].is_dual())
        } else {
            None
        }
    }
}
