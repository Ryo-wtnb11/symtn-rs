use super::*;

impl<R, D> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    pub(in crate::typed) fn insert_unit_multiplicity_free(
        &self,
        insertion: UnitLegInsertion,
    ) -> Result<Self, Error>
    where
        R: CanonicalUnitFusionRule,
    {
        let (UnitLegInsertion::Left { position, .. } | UnitLegInsertion::Right { position, .. }) =
            insertion;
        if position > self.rank() {
            return Err(Error::InvalidArgument(format!(
                "TensorMap::insert_unit: position {position} exceeds rank {}",
                self.rank()
            )));
        }
        let provider = self.logical_space().provider();
        let source_hom = self.logical_space().space().homspace();
        let homspace = match insertion {
            UnitLegInsertion::Left { position, dual } => {
                source_hom.insert_left_unit(provider, position, dual)?
            }
            UnitLegInsertion::Right { position, dual } => {
                source_hom.insert_right_unit(provider, position, dual)?
            }
        };
        let destination = self.logical_space().derive_from_final_homspace(homspace)?;
        validate_unit_layout_correspondence_checked(
            provider,
            (source_hom, self.logical_space().space().structure()),
            (
                destination.space().homspace(),
                destination.space().structure(),
            ),
            insertion,
        )
        .map_err(map_checked_unit_layout_error)?;
        let data = self.shareable_dense_payload();
        Ok(Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::with_shared_payload(destination, data)),
        })
    }

    pub(in crate::typed) fn remove_unit_multiplicity_free(&self, axis: usize) -> Result<Self, Error>
    where
        R: CanonicalUnitFusionRule,
    {
        if axis >= self.rank() {
            return Err(Error::InvalidArgument(format!(
                "TensorMap::remove_unit: axis {axis} is out of range for rank {}",
                self.rank()
            )));
        }
        let provider = self.logical_space().provider();
        let source_hom = self.logical_space().space().homspace();
        let nout = source_hom.codomain().len();
        let leg = if axis < nout {
            &source_hom.codomain().legs()[axis]
        } else {
            &source_hom.domain().legs()[axis - nout]
        };
        if leg.sectors() != [provider.vacuum()] || leg.degeneracy(provider.vacuum()) != Some(1) {
            return Err(Error::InvalidArgument(format!(
                "TensorMap::remove_unit: axis {axis} is not a canonical unit leg"
            )));
        }
        // The insertion that this removal undoes, for the correspondence
        // validator: a codomain leg is the right seam's insertion, a domain
        // leg the left seam's.
        let insertion = if axis < nout {
            UnitLegInsertion::Right {
                position: axis,
                dual: leg.is_dual(),
            }
        } else {
            UnitLegInsertion::Left {
                position: axis,
                dual: leg.is_dual(),
            }
        };
        let homspace = source_hom.remove_unit(provider, axis)?;
        let destination = self.logical_space().derive_from_final_homspace(homspace)?;
        validate_unit_layout_correspondence_checked(
            provider,
            (
                destination.space().homspace(),
                destination.space().structure(),
            ),
            (source_hom, self.logical_space().space().structure()),
            insertion,
        )
        .map_err(map_checked_unit_layout_error)?;
        let data = self.shareable_dense_payload();
        Ok(Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::with_shared_payload(destination, data)),
        })
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorUnitDispatch<R, D>,
    D: TensorScalar,
{
    /// TensorKit `insertleftunit(t, i; dual)` (`seam = Side::Domain`) and
    /// `insertrightunit(t, i; dual)` (`seam = Side::Codomain`): inserts the
    /// canonical unit leg — the vacuum with degeneracy one, on the dual space
    /// for [`Duality::Dual`] — at zero-based flat external slot `position`.
    ///
    /// `seam` decides only the codomain/domain boundary slot
    /// `position == self.codomain_rank()`: the unit joins the side `seam`
    /// names. Every other position lies strictly inside one side, and both
    /// seams insert there identically. The trivial sector adds no block and
    /// reorders nothing, so the stored values are untouched.
    ///
    /// O(1) for a dense payload: the new body shares the payload allocation,
    /// exactly as TensorKit's `copy = false` default shares `t.data` for an
    /// ordinary `TensorMap`. A lazy dense adjoint is first converted to a
    /// fresh dense tensor. A compact spectrum factor materializes
    /// into a fresh dense payload first (one copy) — the unit-leg payload
    /// contract (#613); TensorKit routes its `DiagonalTensorMap` through the generic
    /// similar+block-copy branch for the same reason. No device arm.
    ///
    /// The provider must certify that its vacuum obeys the canonical unit
    /// laws (`CanonicalUnitFusionRule`, or `CheckedCanonicalUnitFusionRule`
    /// for checked Generic); the hom-space transform and the layout validator
    /// both demand it. Results retain the same provider instance.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when `position` exceeds the rank. Otherwise
    /// the layout derivation's and unit-correspondence validator's own
    /// classes; checked-Generic failures retain their typed
    /// [`GenericTensorError`] variants.
    pub fn insert_unit(
        &self,
        position: usize,
        seam: Side,
        dual: Duality,
    ) -> Result<Self, FacadeError<R>> {
        let dual = dual.is_dual();
        let insertion = match seam {
            Side::Domain => UnitLegInsertion::Left { position, dual },
            Side::Codomain => UnitLegInsertion::Right { position, dual },
        };
        <R::Mode as TypedTensorUnitDispatch<R, D>>::insert_unit(self, insertion)
    }

    /// TensorKit `removeunit(t, i)`: removes the canonical unit leg at flat
    /// external axis `axis`. The selected leg must contain exactly the vacuum
    /// sector with degeneracy one. This undoes [`Self::insert_unit`]; sharing
    /// and compact materialization exactly as there — a dense insert→remove
    /// round trip returns to the original spaces on the original payload
    /// allocation.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when `axis` is out of range or the leg is
    /// not a canonical unit leg. Otherwise the layout derivation's and
    /// validator's own classes; checked-Generic failures retain their typed
    /// [`GenericTensorError`] variants.
    pub fn remove_unit(&self, axis: usize) -> Result<Self, FacadeError<R>> {
        <R::Mode as TypedTensorUnitDispatch<R, D>>::remove_unit(self, axis)
    }
}

pub(in crate::typed) fn generic_insert_unit<R, D>(
    tensor: &TensorMap<R, D>,
    insertion: UnitLegInsertion,
) -> Result<TensorMap<R, D>, GenericTensorError<R::Error>>
where
    R: CheckedCanonicalUnitFusionRule,
    D: TensorScalar,
{
    let (UnitLegInsertion::Left { position, .. } | UnitLegInsertion::Right { position, .. }) =
        insertion;
    if position > tensor.rank() {
        return Err(GenericTensorError::Facade(Error::InvalidArgument(format!(
            "TensorMap::insert_unit: position {position} exceeds rank {}",
            tensor.rank()
        ))));
    }
    let provider = tensor.logical_space().provider();
    let source_hom = tensor.logical_space().space().homspace();
    let homspace = match insertion {
        UnitLegInsertion::Left { position, dual } => source_hom
            .insert_left_unit_checked(provider, position, dual)
            .map_err(|error| GenericTensorError::Facade(error.into()))?,
        UnitLegInsertion::Right { position, dual } => source_hom
            .insert_right_unit_checked(provider, position, dual)
            .map_err(|error| GenericTensorError::Facade(error.into()))?,
    };
    // TensorKit cfaa073e4d1e3eb2167edcbdc3be9872f41e7d91 derives the result
    // space before sharing `t.data` in `insertleftunit` / `insertrightunit`:
    // https://github.com/Jutho/TensorKit.jl/blob/cfaa073e4d1e3eb2167edcbdc3be9872f41e7d91/src/tensors/indexmanipulations.jl#L103-L172
    // Rust's checked provider can still reject the derived layout, so keep it
    // staged until the exact source/destination correspondence is proved.
    let validation_homspace = homspace.clone();
    let prepared = tensor
        .logical_space()
        .prepare_final_homspace_generic_with_checked(provider, homspace)
        .map_err(GenericTensorError::Structure)?;
    validate_unit_layout_correspondence_generic_checked(
        provider,
        (source_hom, tensor.logical_space().space().structure()),
        (&validation_homspace, prepared.structure()),
        insertion,
    )
    .map_err(GenericTensorError::Structure)?;
    let destination = tensor
        .logical_space()
        .commit_final_homspace_generic_bound_checked(prepared)
        .map_err(map_checked_unit_commit_error)?;
    let data = tensor.shareable_dense_payload();
    Ok(TensorMap {
        runtime: tensor.runtime.clone(),
        repr: owned_repr(TypedTensorBody::with_shared_payload(destination, data)),
    })
}

pub(in crate::typed) fn generic_remove_unit<R, D>(
    tensor: &TensorMap<R, D>,
    axis: usize,
) -> Result<TensorMap<R, D>, GenericTensorError<R::Error>>
where
    R: CheckedCanonicalUnitFusionRule,
    D: TensorScalar,
{
    if axis >= tensor.rank() {
        return Err(GenericTensorError::Facade(Error::InvalidArgument(format!(
            "TensorMap::remove_unit: axis {axis} is out of range for rank {}",
            tensor.rank()
        ))));
    }
    let provider = tensor.logical_space().provider();
    let source_hom = tensor.logical_space().space().homspace();
    let nout = source_hom.codomain().len();
    let leg = if axis < nout {
        &source_hom.codomain().legs()[axis]
    } else {
        &source_hom.domain().legs()[axis - nout]
    };
    let vacuum = CheckedGenericFusion::vacuum(provider);
    if leg.sectors() != [vacuum] || leg.degeneracy(vacuum) != Some(1) {
        return Err(GenericTensorError::Facade(Error::InvalidArgument(format!(
            "TensorMap::remove_unit: axis {axis} is not a canonical unit leg"
        ))));
    }
    let insertion = if axis < nout {
        UnitLegInsertion::Right {
            position: axis,
            dual: leg.is_dual(),
        }
    } else {
        UnitLegInsertion::Left {
            position: axis,
            dual: leg.is_dual(),
        }
    };
    let homspace = source_hom
        .remove_unit_checked(provider, axis)
        .map_err(|error| GenericTensorError::Facade(error.into()))?;
    // TensorKit cfaa073e4d1e3eb2167edcbdc3be9872f41e7d91 keeps the payload
    // only after deriving the smaller space in `removeunit`:
    // https://github.com/Jutho/TensorKit.jl/blob/cfaa073e4d1e3eb2167edcbdc3be9872f41e7d91/src/tensors/indexmanipulations.jl#L174-L197
    // The checked Rust path additionally delays publication until its layout
    // correspondence with the already-admitted larger source is proved.
    let validation_homspace = homspace.clone();
    let prepared = tensor
        .logical_space()
        .prepare_final_homspace_generic_with_checked(provider, homspace)
        .map_err(GenericTensorError::Structure)?;
    validate_unit_layout_correspondence_generic_checked(
        provider,
        (&validation_homspace, prepared.structure()),
        (source_hom, tensor.logical_space().space().structure()),
        insertion,
    )
    .map_err(GenericTensorError::Structure)?;
    let destination = tensor
        .logical_space()
        .commit_final_homspace_generic_bound_checked(prepared)
        .map_err(map_checked_unit_commit_error)?;
    let data = tensor.shareable_dense_payload();
    Ok(TensorMap {
        runtime: tensor.runtime.clone(),
        repr: owned_repr(TypedTensorBody::with_shared_payload(destination, data)),
    })
}

fn map_checked_unit_commit_error<E>(error: OperationError) -> GenericTensorError<E> {
    match error {
        OperationError::Core(error) => {
            GenericTensorError::Structure(CheckedGenericStructureError::Core(error))
        }
        error => GenericTensorError::from(error),
    }
}
