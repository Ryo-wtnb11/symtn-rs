use super::*;

impl<R, D> TensorMap<R, D>
where
    D: TensorScalar,
{
    /// Restricts tensor-local logical degeneracy coordinates for network
    /// slicing without exposing provider allocation or storage orientation.
    ///
    /// `SectorId` is safe here only as an immediately validated, tensor-local
    /// argument: the authority id is decoded by this tensor's provider, dualized
    /// exactly once for a partner occurrence, and checked against the actual
    /// effective leg before any destination is allocated.
    pub(crate) fn network_restrict_degeneracies(
        &self,
        adjoint: bool,
        restrictions: &[NetworkDegeneracyRestriction],
    ) -> Result<Self, FacadeError<R>>
    where
        R: TypedSectorAdmission,
        R::Mode: TypedTensorRootDispatch<R>,
    {
        let _host_pool = self.runtime.enter_host_pool();
        if matches!(
            &self.repr,
            TypedTensorRepr::Owned(body) if matches!(body.data.as_ref(), TypedData::Diagonal(_))
        ) {
            return Err(Error::InvalidArgument(
                "network degeneracy restriction requires dense Host payloads".to_string(),
            )
            .into());
        }
        let rank = self.rank();
        let codomain_rank = self.codomain_rank();
        let domain_rank = self.domain_rank();
        let homspace = self.logical_space().space().homspace();
        let mut seen = vec![false; rank];
        let mut staged = Vec::with_capacity(restrictions.len());

        // Validate the complete request before deriving a layout or allocating
        // payload. In particular, callers never reconstruct dual-sector ids.
        for restriction in restrictions {
            let NetworkDegeneracyRestriction {
                effective_axis,
                authority_sector,
                range,
                partner,
            } = restriction;
            if *effective_axis >= rank || seen[*effective_axis] {
                return Err(Error::InvalidArgument(format!(
                    "invalid or duplicate effective restriction axis {effective_axis} for rank {rank}"
                ))
                .into());
            }
            seen[*effective_axis] = true;
            if range.start >= range.end {
                return Err(Error::InvalidArgument(format!(
                    "degeneracy restriction must be nonempty, got [{}, {})",
                    range.start, range.end
                ))
                .into());
            }
            TypedSectorAdmission::try_decode_label(self.provider(), *authority_sector)
                .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)?;
            let effective_sector = if *partner {
                TypedSectorAdmission::try_dual_id(self.provider(), *authority_sector)
                    .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)?
            } else {
                *authority_sector
            };

            let (logical_axis, dualized) = if adjoint {
                if *effective_axis < domain_rank {
                    (codomain_rank + *effective_axis, false)
                } else {
                    (*effective_axis - domain_rank, true)
                }
            } else if *effective_axis < codomain_rank {
                (*effective_axis, false)
            } else {
                (*effective_axis, true)
            };
            let logical_sector = if dualized {
                TypedSectorAdmission::try_dual_id(self.provider(), effective_sector)
                    .map_err(<R::Mode as TypedTensorModeDispatch<R>>::map_provider_error)?
            } else {
                effective_sector
            };
            let logical_leg = if logical_axis < codomain_rank {
                &homspace.codomain().legs()[logical_axis]
            } else {
                &homspace.domain().legs()[logical_axis - codomain_rank]
            };
            let degeneracy = logical_leg.degeneracy(logical_sector).ok_or_else(|| {
                FacadeError::<R>::from(Error::InvalidArgument(format!(
                    "sector {effective_sector:?} is absent from effective axis {effective_axis}"
                )))
            })?;
            if range.end > degeneracy {
                return Err(Error::InvalidArgument(format!(
                    "degeneracy restriction [{}, {}) exceeds axis {effective_axis} sector {effective_sector:?} degeneracy {degeneracy}",
                    range.start, range.end
                ))
                .into());
            }
            staged.push((
                logical_axis,
                logical_sector,
                range.start,
                range.end - range.start,
            ));
        }

        let restricted_product = |product: &FusionProductSpace, axis_base: usize| {
            product
                .legs()
                .iter()
                .enumerate()
                .map(|(axis, leg)| {
                    let logical_axis = axis_base + axis;
                    if let Some((_, sector, _, extent)) = staged
                        .iter()
                        .find(|(candidate, ..)| *candidate == logical_axis)
                    {
                        SectorLeg::try_new([(*sector, *extent)], leg.is_dual()).map_err(|error| {
                            FacadeError::<R>::from(Error::InvalidArgument(error.to_string()))
                        })
                    } else {
                        Ok(leg.clone())
                    }
                })
                .collect::<Result<Vec<_>, FacadeError<R>>>()
                .map(FusionProductSpace::new)
        };
        let restricted_homspace = FusionTreeHomSpace::new(
            restricted_product(homspace.codomain(), 0)?,
            restricted_product(homspace.domain(), codomain_rank)?,
        );
        let destination = <R::Mode as TypedTensorRootDispatch<R>>::build_root(
            Arc::clone(self.logical_space().provider_arc()),
            restricted_homspace,
        )?;
        // One single-entry table per restricted axis: this path always names
        // exactly one sector per axis, and the kernel reads the sector back
        // from each destination block's own key.
        let tables: Vec<[(SectorId, symtn_tensors::SelectedRuns); 1]> = staged
            .iter()
            .map(|&(_, sector, start, extent)| {
                [(
                    sector,
                    symtn_tensors::SelectedRuns::from_elem(start..start + extent, 1),
                )]
            })
            .collect();
        let mut runs: Vec<symtn_tensors::SectorRunTable<'_>> = vec![None; rank];
        for (table, &(axis, ..)) in tables.iter().zip(&staged) {
            runs[axis] = Some(table.as_slice());
        }
        let (source, source_data) = self.fusion_operand_and_data();
        let data = symtn_tensors::oriented_fusion_restrict_owned(
            destination.space().structure(),
            source,
            &source_data,
            &runs,
        )
        .map_err(Error::from)?;
        Ok(Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(destination, data)),
        })
    }

    pub(crate) fn network_has_compact_payload(&self) -> bool {
        matches!(self.storage_body().data.as_ref(), TypedData::Diagonal(_))
    }

    pub(crate) fn network_zeros_from_effective_legs(
        &self,
        codomain: &[GradedSpace<R>],
        domain: &[GradedSpace<R>],
    ) -> Result<Self, FacadeError<R>>
    where
        R: TypedSectorAdmission,
        R::Mode: TypedTensorRootDispatch<R>,
    {
        let identity = TypedSectorAdmission::typed_rule_identity(self.provider());
        if codomain
            .iter()
            .chain(domain)
            .any(|leg| TypedSectorAdmission::typed_rule_identity(leg.provider()) != identity)
        {
            return Err(Error::RuleMismatch.into());
        }
        let space = <R::Mode as TypedTensorRootDispatch<R>>::build_root_with(
            Arc::clone(self.logical_space().provider_arc()),
            || {
                let raw_domain = domain
                    .iter()
                    .map(GradedSpace::try_dual)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(FusionTreeHomSpace::new(
                    FusionProductSpace::new(
                        codomain.iter().map(|leg| leg.network_sector_leg().clone()),
                    ),
                    FusionProductSpace::new(
                        raw_domain
                            .iter()
                            .map(|leg| leg.network_sector_leg().clone()),
                    ),
                ))
            },
        )?;
        let len = space
            .space()
            .required_len()
            .map_err(Error::from)
            .map_err(FacadeError::<R>::from)?;
        Ok(Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(space, vec![D::from_real(0.0); len])),
        })
    }
}

/// One tensor-local restriction used by the internal network slice executor.
#[doc(hidden)]
pub struct NetworkDegeneracyRestriction {
    pub effective_axis: usize,
    pub authority_sector: SectorId,
    pub range: std::ops::Range<usize>,
    pub partner: bool,
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    D: TensorScalar,
{
    /// Adds every source block into the matching destination block rectangle.
    pub(crate) fn network_scatter_add_assign(
        &mut self,
        source: &Self,
        ranges: &[Option<std::ops::Range<usize>>],
    ) -> Result<(), Error> {
        if !self.runtime.same_runtime(&source.runtime) {
            return Err(Error::RuntimeMismatch);
        }
        let destination_space = self.logical_space().space();
        let _host_pool = self.runtime.enter_host_pool();
        let source_space = source.logical_space().space();
        if destination_space.nout() != source_space.nout()
            || destination_space.nin() != source_space.nin()
        {
            return Err(Error::InvalidArgument(
                format!("network slice partial has codomain/domain split ({}, {}) but the accumulator has ({}, {})", source_space.nout(), source_space.nin(), destination_space.nout(), destination_space.nin()),
            ));
        }
        if ranges.len() != destination_space.rank() {
            return Err(Error::InvalidArgument(format!(
                "network slice partial has {} ranges for a rank-{} accumulator",
                ranges.len(),
                destination_space.rank()
            )));
        }
        if !Arc::ptr_eq(
            self.logical_space().provider_arc(),
            source.logical_space().provider_arc(),
        ) {
            return Err(Error::InvalidArgument(
                "network slice partial comes from a different sector provider".to_string(),
            ));
        }
        let destination_legs = destination_space
            .homspace()
            .codomain()
            .legs()
            .iter()
            .chain(destination_space.homspace().domain().legs());
        let source_legs = source_space
            .homspace()
            .codomain()
            .legs()
            .iter()
            .chain(source_space.homspace().domain().legs());
        let mut sliced = Vec::new();
        for (axis, ((destination_leg, source_leg), range)) in
            destination_legs.zip(source_legs).zip(ranges).enumerate()
        {
            match range {
                None if source_leg == destination_leg => {}
                None => {
                    return Err(Error::InvalidArgument(
                        "network slice unsliced output leg differs from destination".to_string(),
                    ));
                }
                Some(range) => {
                    let Some((sector, degeneracy)) = source_leg.iter().next() else {
                        return Err(Error::InvalidArgument(
                            "network sliced output leg must select one sector".to_string(),
                        ));
                    };
                    if source_leg.iter().nth(1).is_some()
                        || source_leg.is_dual() != destination_leg.is_dual()
                        || range.end.checked_sub(range.start) != Some(degeneracy)
                        || destination_leg
                            .degeneracy(sector)
                            .is_none_or(|destination| range.end > destination)
                    {
                        return Err(Error::InvalidArgument(
                            "network sliced output leg does not match destination range"
                                .to_string(),
                        ));
                    }
                    sliced.push((
                        axis,
                        [(
                            sector,
                            symtn_tensors::SelectedRuns::from_elem(range.clone(), 1),
                        )],
                    ));
                }
            }
        }
        // The sliced source leg carries exactly one sector (checked above), so
        // each restricted axis needs a single-entry table; the kernel reads the
        // sector back from each source block's own key.
        let mut scatter_ranges: Vec<symtn_tensors::SectorRunTable<'_>> = vec![None; ranges.len()];
        for (axis, table) in &sliced {
            scatter_ranges[*axis] = Some(table.as_slice());
        }
        let source_is_dense = match &source.repr {
            TypedTensorRepr::Owned(body) => matches!(body.data.as_ref(), TypedData::Dense(_)),
            TypedTensorRepr::Adjoint(_) => true,
        };
        if !source_is_dense {
            return Err(Error::InvalidArgument(
                "network slice partial must have dense payload".to_string(),
            ));
        }
        let destination_structure = destination_space.structure().clone();
        let source_structure = source_space.structure();
        let (source_operand, source_data) = source.fusion_operand_and_data();
        let TypedTensorRepr::Owned(destination_body) = &mut self.repr else {
            return Err(Error::InvalidArgument(
                "network slice accumulator must be direct owned".to_string(),
            ));
        };
        let destination_body = Arc::get_mut(destination_body).ok_or_else(|| {
            Error::InvalidArgument("network slice accumulator payload is shared".to_string())
        })?;
        let TypedData::Dense(destination_data) = Arc::get_mut(&mut destination_body.data)
            .ok_or_else(|| {
                Error::InvalidArgument("network slice accumulator data is shared".to_string())
            })?
        else {
            return Err(Error::InvalidArgument(
                "network slice accumulator must have dense payload".to_string(),
            ));
        };
        symtn_tensors::fusion_scatter_add_assign(
            &destination_structure,
            destination_data,
            source_structure,
            source_operand,
            &source_data,
            &scatter_ranges,
        )
        .map_err(Error::from)
    }
}
