use super::*;

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct FusionProductSpace {
    pub(super) legs: SmallVec<[SectorLeg; 8]>,
}

impl FusionProductSpace {
    /// Builds a product of external fusion legs.
    ///
    /// # Examples
    ///
    /// ```
    /// use symtn_core::{FusionProductSpace, SectorLeg, FermionParityIrrep};
    ///
    /// let space = FusionProductSpace::new([
    ///     SectorLeg::new([(FermionParityIrrep::EVEN, 1), (FermionParityIrrep::ODD, 1)], false),
    ///     SectorLeg::new([(FermionParityIrrep::EVEN, 1)], true),
    /// ]);
    /// assert_eq!(space.len(), 2);
    /// ```
    pub fn new<Legs>(legs: Legs) -> Self
    where
        Legs: IntoIterator<Item = SectorLeg>,
    {
        Self {
            legs: legs.into_iter().collect(),
        }
    }

    /// Builds a product of single-sector legs from `(sector id, degeneracy)`
    /// pairs.
    pub fn from_sector_ids<Sectors>(sectors: Sectors) -> Self
    where
        Sectors: IntoIterator<Item = (usize, usize)>,
    {
        Self::new(
            sectors
                .into_iter()
                .map(|(sector, degeneracy)| SectorLeg::from_sector_id(sector, degeneracy)),
        )
    }

    #[inline]
    pub fn legs(&self) -> &[SectorLeg] {
        &self.legs
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.legs.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.legs.is_empty()
    }

    /// Computes every coupled sector's reduced dimension under `rule`.
    ///
    /// The eager fold includes external degeneracies and fusion multiplicities
    /// without constructing fusion trees or publishing cache state.
    pub fn coupled_sector_block_dimensions<R>(
        &self,
        rule: &R,
    ) -> Result<BTreeMap<SectorId, usize>, CoreError>
    where
        R: FusionRule,
    {
        // Why no catalog-completeness guard: an infallible `FusionRule` is
        // unbounded by construction, so every channel it names is
        // representable. A provider with a finite table implements
        // `CheckedGenericFusion` and reaches the checked sibling instead.
        self.fold_coupled_sector_block_dimensions(
            rule.vacuum(),
            |left, right| Ok(rule.fusion_channels(left, right)),
            |left, right, coupled| Ok(rule.nsymbol(left, right, coupled)),
        )
    }

    /// Checked sibling of [`Self::coupled_sector_block_dimensions`]: the same
    /// fold over a provider whose channel and multiplicity queries may fail.
    pub fn coupled_sector_block_dimensions_checked<R>(
        &self,
        rule: &R,
    ) -> Result<BTreeMap<SectorId, usize>, CheckedGenericStructureError<R::Error>>
    where
        R: CheckedGenericFusion,
    {
        self.fold_coupled_sector_block_dimensions(
            rule.vacuum(),
            |left, right| {
                rule.try_fusion_channels(left, right)
                    .map_err(CheckedGenericStructureError::Provider)
            },
            |left, right, coupled| {
                rule.try_nsymbol(left, right, coupled)
                    .map_err(CheckedGenericStructureError::Provider)
            },
        )
    }

    fn fold_coupled_sector_block_dimensions<E: From<CoreError>>(
        &self,
        vacuum: SectorId,
        mut channels: impl FnMut(SectorId, SectorId) -> Result<SectorVec, E>,
        mut nsymbol: impl FnMut(SectorId, SectorId, SectorId) -> Result<usize, E>,
    ) -> Result<BTreeMap<SectorId, usize>, E> {
        // Why not cache or enumerate fusion trees: this map is a small
        // structural preflight, and the dynamic program computes only the
        // dimensions that the caller needs.
        let mut dimensions = BTreeMap::from([(vacuum, 1usize)]);
        for leg in self.legs() {
            let mut next = BTreeMap::<SectorId, usize>::new();
            for (&left, &left_dimension) in &dimensions {
                for (right, right_degeneracy) in leg.iter() {
                    // Why not dualize `right` from `SectorLeg::is_dual`: stored
                    // sector IDs are already outward labels; the flag controls
                    // pivotal and braiding data, not ProductSpace block dimensions.
                    let mut channels = channels(left, right)?;
                    channels.sort_unstable();
                    channels.dedup();
                    for coupled in channels {
                        let multiplicity = nsymbol(left, right, coupled)?;
                        let contribution = left_dimension
                            .checked_mul(right_degeneracy)
                            .and_then(|value| value.checked_mul(multiplicity))
                            .ok_or(CoreError::ElementCountOverflow)?;
                        if contribution == 0 {
                            continue;
                        }
                        let coupled_dimension = next.entry(coupled).or_default();
                        *coupled_dimension = coupled_dimension
                            .checked_add(contribution)
                            .ok_or(CoreError::ElementCountOverflow)?;
                    }
                }
            }
            dimensions = next;
        }
        Ok(dimensions)
    }

    pub(crate) fn try_visit_selected_leg_tuples<E, F>(&self, emit: &mut F) -> Result<(), E>
    where
        F: FnMut(&[FusionTreeLeg]) -> Result<(), E>,
    {
        let mut current = vec![FusionTreeLeg::new(SectorId::new(0), false); self.legs.len()];
        try_visit_selected_leg_tuples(&self.legs, self.legs.len(), &mut current, emit)
    }
}
