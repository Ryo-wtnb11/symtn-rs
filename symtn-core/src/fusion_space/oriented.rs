use super::*;

#[derive(Clone, Copy)]
pub(crate) struct OrientedLegView<'a> {
    pub(super) source: &'a SectorLeg,
    pub(super) dualize: bool,
}

impl<'a> OrientedLegView<'a> {
    pub(crate) fn borrowed(source: &'a SectorLeg) -> Self {
        Self {
            source,
            dualize: false,
        }
    }

    pub(crate) fn toggled(self) -> Self {
        Self {
            source: self.source,
            dualize: !self.dualize,
        }
    }

    pub(super) fn is_dual(self) -> bool {
        self.source.is_dual() ^ self.dualize
    }

    pub(super) fn mapped_sector<R>(self, rule: &R, sector: SectorId) -> SectorId
    where
        R: FusionRule,
    {
        if self.dualize {
            rule.dual(sector)
        } else {
            sector
        }
    }

    pub(super) fn try_mapped_sector<R>(
        self,
        rule: &R,
        sector: SectorId,
    ) -> Result<SectorId, FusionAlgebraError>
    where
        R: CheckedFusionAlgebra,
    {
        if self.dualize {
            rule.try_dual_sector(sector)
        } else {
            Ok(sector)
        }
    }

    pub(crate) fn materialize<R>(self, rule: &R) -> SectorLeg
    where
        R: FusionRule,
    {
        if self.dualize {
            self.source.dual(rule)
        } else {
            self.source.clone()
        }
    }

    pub(super) fn try_materialize<R>(self, rule: &R) -> Result<SectorLeg, FusionAlgebraError>
    where
        R: CheckedFusionAlgebra,
    {
        if self.dualize {
            self.source.try_dual(rule)
        } else {
            Ok(self.source.clone())
        }
    }

    pub(super) fn try_materialize_generic<R>(
        self,
        rule: &R,
    ) -> Result<SectorLeg, CheckedGenericStructureError<R::Error>>
    where
        R: CheckedGenericFusion,
    {
        if !self.dualize {
            return Ok(self.source.clone());
        }
        self.source.try_dual_generic(rule)
    }
}

impl OrientedLegView<'_> {
    /// Whether materializing this view yields `leg`, decided without building
    /// a leg. `Ok(false)` means "not proven equal": the caller materializes
    /// and compares, which keeps every mismatch and non-injective-dual
    /// answer on the established path. `dual` is queried in materialization
    /// order and only until the first unproven sector, so its first error is
    /// the one materialization reports.
    fn matches_leg<E>(
        self,
        leg: &SectorLeg,
        dual: &mut impl FnMut(SectorId) -> Result<SectorId, E>,
    ) -> Result<bool, E> {
        if !self.dualize {
            return Ok(self.source == leg);
        }
        let len = self.source.sectors().len();
        // Why a 64-bit hit mask: it proves the dual injective on this leg
        // without a heap buffer; wider legs take the materializing path.
        if leg.is_dual() == self.source.is_dual() || leg.sectors().len() != len || len > 64 {
            return Ok(false);
        }
        let mut hits = 0u64;
        for (sector, degeneracy) in self.source.iter() {
            let Ok(index) = leg.sectors().binary_search(&dual(sector)?) else {
                return Ok(false);
            };
            let bit = 1u64 << index;
            if leg.degeneracies()[index] != degeneracy || hits & bit != 0 {
                return Ok(false);
            }
            hits |= bit;
        }
        // `len` distinct hits among `len` sectors: the dual is a bijection
        // onto `leg`'s sorted sectors with equal degeneracies.
        Ok(true)
    }
}

pub(super) struct HomSpaceDescriptor<'a> {
    // Why one vector instead of one per side: rank, not side rank, is the
    // natural inline bound. PEPS/MPS metadata up to rank 8 stays entirely on
    // the stack and `nout` splits the stored-orientation views.
    legs: SmallVec<[OrientedLegView<'a>; 8]>,
    nout: usize,
}

impl<'a> HomSpaceDescriptor<'a> {
    pub(super) fn new(
        codomain: impl IntoIterator<Item = OrientedLegView<'a>>,
        domain: impl IntoIterator<Item = OrientedLegView<'a>>,
    ) -> Self {
        let mut legs = SmallVec::new();
        legs.extend(codomain);
        let nout = legs.len();
        legs.extend(domain);
        Self { legs, nout }
    }

    fn codomain(&self) -> &[OrientedLegView<'a>] {
        &self.legs[..self.nout]
    }

    fn domain(&self) -> &[OrientedLegView<'a>] {
        &self.legs[self.nout..]
    }

    /// Allocation-free proof that [`Self::materialize`] equals `expected`;
    /// see [`OrientedLegView::matches_leg`] for the `Ok(false)` contract.
    fn matches<E>(
        &self,
        expected: &FusionTreeHomSpace,
        mut dual: impl FnMut(SectorId) -> Result<SectorId, E>,
    ) -> Result<bool, E> {
        if self.codomain().len() != expected.codomain().len()
            || self.domain().len() != expected.domain().len()
        {
            return Ok(false);
        }
        let expected_legs = expected
            .codomain()
            .legs()
            .iter()
            .chain(expected.domain().legs());
        for (view, leg) in self.legs.iter().copied().zip(expected_legs) {
            if !view.matches_leg(leg, &mut dual)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub(super) fn materialize<R>(&self, rule: &R) -> FusionTreeHomSpace
    where
        R: FusionRule,
    {
        #[cfg(test)]
        crate::probe::hit(crate::probe::Event::DescriptorMaterialization);
        FusionTreeHomSpace::new(
            FusionProductSpace::new(
                self.codomain()
                    .iter()
                    .copied()
                    .map(|view| view.materialize(rule)),
            ),
            FusionProductSpace::new(
                self.domain()
                    .iter()
                    .copied()
                    .map(|view| view.materialize(rule)),
            ),
        )
    }

    pub(super) fn try_materialize<R>(
        &self,
        rule: &R,
    ) -> Result<FusionTreeHomSpace, FusionAlgebraError>
    where
        R: CheckedFusionAlgebra,
    {
        #[cfg(test)]
        crate::probe::hit(crate::probe::Event::DescriptorMaterialization);
        let codomain = self
            .codomain()
            .iter()
            .copied()
            .map(|view| view.try_materialize(rule))
            .collect::<Result<SmallVec<[SectorLeg; 8]>, _>>()?;
        let domain = self
            .domain()
            .iter()
            .copied()
            .map(|view| view.try_materialize(rule))
            .collect::<Result<SmallVec<[SectorLeg; 8]>, _>>()?;
        Ok(FusionTreeHomSpace::new(
            FusionProductSpace::new(codomain),
            FusionProductSpace::new(domain),
        ))
    }

    pub(super) fn try_materialize_generic<R>(
        &self,
        rule: &R,
    ) -> Result<FusionTreeHomSpace, CheckedGenericStructureError<R::Error>>
    where
        R: CheckedGenericFusion,
    {
        let codomain = self
            .codomain()
            .iter()
            .copied()
            .map(|view| view.try_materialize_generic(rule))
            .collect::<Result<SmallVec<[SectorLeg; 8]>, _>>()?;
        let domain = self
            .domain()
            .iter()
            .copied()
            .map(|view| view.try_materialize_generic(rule))
            .collect::<Result<SmallVec<[SectorLeg; 8]>, _>>()?;
        Ok(FusionTreeHomSpace::new(
            FusionProductSpace::new(codomain),
            FusionProductSpace::new(domain),
        ))
    }
}

#[doc(hidden)]
#[derive(Clone, Copy)]
pub struct OrientedFusionTreeHomSpace<'a> {
    source: &'a FusionTreeHomSpace,
    orientation: FusionTreePairOrientation,
}

impl<'a> OrientedFusionTreeHomSpace<'a> {
    #[doc(hidden)]
    pub fn new(source: &'a FusionTreeHomSpace, orientation: FusionTreePairOrientation) -> Self {
        Self {
            source,
            orientation,
        }
    }

    #[doc(hidden)]
    pub fn rank(self) -> usize {
        self.source.rank()
    }

    #[doc(hidden)]
    pub fn nout(self) -> usize {
        match self.orientation {
            FusionTreePairOrientation::Direct => self.source.codomain().len(),
            FusionTreePairOrientation::Adjoint => self.source.domain().len(),
        }
    }

    #[doc(hidden)]
    pub fn nin(self) -> usize {
        self.rank() - self.nout()
    }

    #[doc(hidden)]
    pub fn orientation(self) -> FusionTreePairOrientation {
        self.orientation
    }

    #[doc(hidden)]
    pub fn materialize(self) -> FusionTreeHomSpace {
        match self.orientation {
            FusionTreePairOrientation::Direct => self.source.clone(),
            FusionTreePairOrientation::Adjoint => FusionTreeHomSpace::new(
                self.source.domain().clone(),
                self.source.codomain().clone(),
            ),
        }
    }

    #[doc(hidden)]
    pub fn select<R>(
        self,
        rule: &R,
        codomain_axes: &[usize],
        domain_axes: &[usize],
    ) -> Result<FusionTreeHomSpace, CoreError>
    where
        R: FusionRule,
    {
        let descriptor = self.select_descriptor(codomain_axes, domain_axes)?;
        Ok(descriptor.materialize(rule))
    }

    #[doc(hidden)]
    pub fn try_select_checked<R>(
        self,
        rule: &R,
        codomain_axes: &[usize],
        domain_axes: &[usize],
    ) -> Result<FusionTreeHomSpace, CheckedFusionSpaceError>
    where
        R: CheckedFusionAlgebra,
    {
        let descriptor = self.select_descriptor(codomain_axes, domain_axes)?;
        descriptor.try_materialize(rule).map_err(Into::into)
    }

    /// Checked Generic sibling of [`Self::select`] that preserves provider
    /// failures without requiring the `CheckedFusionAlgebra` contract.
    #[doc(hidden)]
    pub fn try_select_generic_checked<R>(
        self,
        rule: &R,
        codomain_axes: &[usize],
        domain_axes: &[usize],
    ) -> Result<FusionTreeHomSpace, CheckedGenericStructureError<R::Error>>
    where
        R: CheckedGenericFusion,
    {
        let descriptor = self.select_descriptor(codomain_axes, domain_axes)?;
        descriptor.try_materialize_generic(rule)
    }

    #[doc(hidden)]
    pub fn external_axis_leg<R>(self, rule: &R, axis: usize) -> Option<SectorLeg>
    where
        R: FusionRule,
    {
        self.external_axis_leg_view(axis)
            .map(|view| view.materialize(rule))
    }

    #[doc(hidden)]
    pub fn try_external_axis_leg<R>(
        self,
        rule: &R,
        axis: usize,
    ) -> Result<Option<SectorLeg>, FusionAlgebraError>
    where
        R: CheckedFusionAlgebra,
    {
        self.external_axis_leg_view(axis)
            .map(|view| view.try_materialize(rule))
            .transpose()
    }

    /// Checked Generic sibling of [`Self::try_external_axis_leg`] that keeps
    /// provider failures typed and does not require the infallible
    /// [`FusionRule`] contract.
    #[doc(hidden)]
    pub fn try_external_axis_leg_generic<R>(
        self,
        rule: &R,
        axis: usize,
    ) -> Result<Option<SectorLeg>, CheckedGenericStructureError<R::Error>>
    where
        R: CheckedGenericFusion,
    {
        self.external_axis_leg_view(axis)
            .map(|view| view.try_materialize_generic(rule))
            .transpose()
    }

    #[doc(hidden)]
    pub fn external_axis_is_dual(self, axis: usize) -> Option<bool> {
        self.external_axis_leg_view(axis)
            .map(OrientedLegView::is_dual)
    }

    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn tensorcontract_homspace<R>(
        rule: &R,
        lhs: Self,
        rhs: Self,
        lhs_contracting_axes: &[usize],
        rhs_contracting_axes: &[usize],
        output_axes: &[usize],
        dst_codomain_rank: usize,
    ) -> Result<FusionTreeHomSpace, CoreError>
    where
        R: FusionRule,
    {
        let descriptor = tensorcontract_descriptor(
            lhs,
            rhs,
            lhs_contracting_axes,
            rhs_contracting_axes,
            output_axes,
            dst_codomain_rank,
        )?;
        for (&lhs_axis, &rhs_axis) in lhs_contracting_axes.iter().zip(rhs_contracting_axes) {
            validate_oriented_composed_leg(
                rule,
                lhs.external_axis_leg_view(lhs_axis)
                    .expect("validated axis belongs to the lhs")
                    .toggled(),
                rhs.external_axis_leg_view(rhs_axis)
                    .expect("validated axis belongs to the rhs"),
                (lhs_axis, rhs_axis),
            )?;
        }
        Ok(descriptor.materialize(rule))
    }

    /// Whether [`Self::tensorcontract_homspace`] equals `expected`, with the
    /// same errors in the same order, without building the contracted
    /// HomSpace when the legs prove equality.
    ///
    /// Why not build and compare: this check validates a destination the
    /// caller already holds, on every eager contraction, and TensorKit's
    /// equivalent is one space comparison.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn tensorcontract_homspace_matches<R>(
        rule: &R,
        lhs: Self,
        rhs: Self,
        lhs_contracting_axes: &[usize],
        rhs_contracting_axes: &[usize],
        output_axes: &[usize],
        dst_codomain_rank: usize,
        expected: &FusionTreeHomSpace,
    ) -> Result<bool, CoreError>
    where
        R: FusionRule,
    {
        let descriptor = tensorcontract_descriptor(
            lhs,
            rhs,
            lhs_contracting_axes,
            rhs_contracting_axes,
            output_axes,
            dst_codomain_rank,
        )?;
        for (&lhs_axis, &rhs_axis) in lhs_contracting_axes.iter().zip(rhs_contracting_axes) {
            validate_oriented_composed_leg(
                rule,
                lhs.external_axis_leg_view(lhs_axis)
                    .expect("validated axis belongs to the lhs")
                    .toggled(),
                rhs.external_axis_leg_view(rhs_axis)
                    .expect("validated axis belongs to the rhs"),
                (lhs_axis, rhs_axis),
            )?;
        }
        let proven = descriptor
            .matches(expected, |sector| {
                Ok::<_, std::convert::Infallible>(rule.dual(sector))
            })
            .unwrap_or_else(|never| match never {});
        Ok(proven || descriptor.materialize(rule) == *expected)
    }

    /// Checked sibling of [`Self::tensorcontract_homspace_matches`].
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn try_tensorcontract_homspace_matches_checked<R>(
        rule: &R,
        lhs: Self,
        rhs: Self,
        lhs_contracting_axes: &[usize],
        rhs_contracting_axes: &[usize],
        output_axes: &[usize],
        dst_codomain_rank: usize,
        expected: &FusionTreeHomSpace,
    ) -> Result<bool, CheckedFusionSpaceError>
    where
        R: CheckedFusionAlgebra,
    {
        let descriptor = tensorcontract_descriptor(
            lhs,
            rhs,
            lhs_contracting_axes,
            rhs_contracting_axes,
            output_axes,
            dst_codomain_rank,
        )?;
        for (&lhs_axis, &rhs_axis) in lhs_contracting_axes.iter().zip(rhs_contracting_axes) {
            validate_oriented_composed_leg_checked(
                rule,
                lhs.external_axis_leg_view(lhs_axis)
                    .expect("validated axis belongs to the lhs")
                    .toggled(),
                rhs.external_axis_leg_view(rhs_axis)
                    .expect("validated axis belongs to the rhs"),
                (lhs_axis, rhs_axis),
            )?;
        }
        if descriptor.matches(expected, |sector| rule.try_dual_sector(sector))? {
            return Ok(true);
        }
        Ok(descriptor.try_materialize(rule)? == *expected)
    }

    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn try_tensorcontract_homspace_checked<R>(
        rule: &R,
        lhs: Self,
        rhs: Self,
        lhs_contracting_axes: &[usize],
        rhs_contracting_axes: &[usize],
        output_axes: &[usize],
        dst_codomain_rank: usize,
    ) -> Result<FusionTreeHomSpace, CheckedFusionSpaceError>
    where
        R: CheckedFusionAlgebra,
    {
        let descriptor = tensorcontract_descriptor(
            lhs,
            rhs,
            lhs_contracting_axes,
            rhs_contracting_axes,
            output_axes,
            dst_codomain_rank,
        )?;
        for (&lhs_axis, &rhs_axis) in lhs_contracting_axes.iter().zip(rhs_contracting_axes) {
            validate_oriented_composed_leg_checked(
                rule,
                lhs.external_axis_leg_view(lhs_axis)
                    .expect("validated axis belongs to the lhs")
                    .toggled(),
                rhs.external_axis_leg_view(rhs_axis)
                    .expect("validated axis belongs to the rhs"),
                (lhs_axis, rhs_axis),
            )?;
        }
        descriptor.try_materialize(rule).map_err(Into::into)
    }

    /// Checked Generic sibling of [`Self::tensorcontract_homspace`].
    ///
    /// Provider failures remain typed, including dualization required by
    /// oriented domain legs. Structural axis and leg-shape defects are
    /// rejected before the first provider query.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn try_tensorcontract_homspace_generic_checked<R>(
        rule: &R,
        lhs: Self,
        rhs: Self,
        lhs_contracting_axes: &[usize],
        rhs_contracting_axes: &[usize],
        output_axes: &[usize],
        dst_codomain_rank: usize,
    ) -> Result<FusionTreeHomSpace, CheckedGenericStructureError<R::Error>>
    where
        R: CheckedGenericFusion,
    {
        let descriptor = tensorcontract_descriptor(
            lhs,
            rhs,
            lhs_contracting_axes,
            rhs_contracting_axes,
            output_axes,
            dst_codomain_rank,
        )?;
        for (&lhs_axis, &rhs_axis) in lhs_contracting_axes.iter().zip(rhs_contracting_axes) {
            validate_oriented_composed_leg_generic_checked(
                rule,
                lhs.external_axis_leg_view(lhs_axis)
                    .expect("validated axis belongs to the lhs")
                    .toggled(),
                rhs.external_axis_leg_view(rhs_axis)
                    .expect("validated axis belongs to the rhs"),
                (lhs_axis, rhs_axis),
            )?;
        }
        descriptor.try_materialize_generic(rule)
    }

    fn select_descriptor(
        self,
        codomain_axes: &[usize],
        domain_axes: &[usize],
    ) -> Result<HomSpaceDescriptor<'a>, CoreError> {
        validate_axis_selection(codomain_axes, domain_axes, self.rank())?;
        Ok(HomSpaceDescriptor::new(
            codomain_axes.iter().map(|&axis| {
                self.external_axis_leg_view(axis)
                    .expect("validated axis belongs to the source")
            }),
            domain_axes.iter().map(|&axis| {
                self.external_axis_leg_view(axis)
                    .expect("validated axis belongs to the source")
                    .toggled()
            }),
        ))
    }

    pub(super) fn external_axis_leg_view(self, axis: usize) -> Option<OrientedLegView<'a>> {
        match self.orientation {
            FusionTreePairOrientation::Direct => {
                if axis < self.source.codomain().len() {
                    Some(OrientedLegView::borrowed(
                        &self.source.codomain().legs()[axis],
                    ))
                } else if axis < self.source.rank() {
                    Some(
                        OrientedLegView::borrowed(
                            &self.source.domain().legs()[axis - self.source.codomain().len()],
                        )
                        .toggled(),
                    )
                } else {
                    None
                }
            }
            FusionTreePairOrientation::Adjoint => {
                if axis < self.source.domain().len() {
                    Some(OrientedLegView::borrowed(
                        &self.source.domain().legs()[axis],
                    ))
                } else if axis < self.source.rank() {
                    Some(
                        OrientedLegView::borrowed(
                            &self.source.codomain().legs()[axis - self.source.domain().len()],
                        )
                        .toggled(),
                    )
                } else {
                    None
                }
            }
        }
    }
}
