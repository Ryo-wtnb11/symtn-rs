use super::*;

impl<R> TypedSpaceModeDispatch<R> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols
        + CheckedFusionAlgebra
        + SectorCodec,
    MultiplicityFreeAdmissionMode: symtn_tensors::RigidCoefficientAlgebra<R>
        + symtn_tensors::CoefficientAlgebra<R, Error = symtn_tensors::OperationError>,
{
    fn vacuum(provider: &R) -> SectorId {
        provider.vacuum()
    }

    fn fusion_channels(
        provider: &R,
        left: SectorId,
        right: SectorId,
    ) -> Result<Vec<SectorId>, FacadeError<R>> {
        provider
            .try_fusion_channels(left, right)
            .map(|channels| channels.into_iter().collect())
            .map_err(Into::into)
    }

    fn nsymbol(
        provider: &R,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FacadeError<R>> {
        provider
            .try_nsymbol(left, right, coupled)
            .map_err(Into::into)
    }

    fn dim(provider: &R, sector: SectorId) -> Result<f64, FacadeError<R>> {
        <Self as symtn_tensors::RigidCoefficientAlgebra<R>>::dim(provider, sector)
            .map_err(Into::into)
    }
}

impl<R> TypedSpaceModeDispatch<R> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericRigidSymbols<Scalar = f64>,
{
    fn vacuum(provider: &R) -> SectorId {
        CheckedGenericFusion::vacuum(provider)
    }

    fn fusion_channels(
        provider: &R,
        left: SectorId,
        right: SectorId,
    ) -> Result<Vec<SectorId>, FacadeError<R>> {
        provider
            .try_fusion_channels(left, right)
            .map(|channels| channels.into_iter().collect())
            .map_err(<Self as TypedTensorModeDispatch<R>>::map_provider_error)
    }

    fn nsymbol(
        provider: &R,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FacadeError<R>> {
        provider
            .try_nsymbol(left, right, coupled)
            .map_err(<Self as TypedTensorModeDispatch<R>>::map_provider_error)
    }

    fn dim(provider: &R, sector: SectorId) -> Result<f64, FacadeError<R>> {
        <Self as symtn_tensors::RigidCoefficientAlgebra<R>>::dim(provider, sector).map_err(
            |error| match error {
                symtn_tensors::CheckedGenericPlanError::Provider(error) => {
                    <Self as TypedTensorModeDispatch<R>>::map_provider_error(error)
                }
                other => GenericTensorError::Plan(other),
            },
        )
    }
}

impl<R> TypedTruncationDispatch<R> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols
        + CheckedFusionAlgebra,
    MultiplicityFreeAdmissionMode: symtn_tensors::RigidCoefficientAlgebra<R>
        + symtn_tensors::CoefficientAlgebra<R, Error = symtn_tensors::OperationError>,
{
    fn decide_bond_truncation<V>(
        provider: &R,
        spectra: &[symtn_matrixalgebra::SectorSpectrum<V>],
        truncation: &Truncation,
    ) -> Result<symtn_matrixalgebra::TruncationDecision, Error>
    where
        V: SpectrumMagnitude,
    {
        // Magnitude-based, as MatrixAlgebraKit `findtruncated` is.
        symtn_matrixalgebra::seam::decide_bond_truncation::<Self, _, _>(
            provider,
            spectra,
            truncation,
            &symtn_core::FusionRule::rule_identity(provider),
            |sector| symtn_core::FusionRule::sector_order_key(provider, sector),
        )
        .map_err(Error::from)
    }
}

impl<R> TypedTruncationDispatch<R> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericRigidSymbols<Scalar = f64>,
{
    fn decide_bond_truncation<V>(
        provider: &R,
        spectra: &[symtn_matrixalgebra::SectorSpectrum<V>],
        truncation: &Truncation,
    ) -> Result<symtn_matrixalgebra::TruncationDecision, Self::FacadeError>
    where
        V: SpectrumMagnitude,
    {
        symtn_matrixalgebra::seam::decide_bond_truncation::<Self, _, _>(
            provider,
            spectra,
            truncation,
            &CheckedGenericFusion::rule_identity(provider),
            |sector| CheckedGenericFusion::sector_order_key(provider, sector),
        )
        .map_err(Into::into)
    }
}

impl<R> TypedTensorModeDispatch<R> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + CheckedFusionAlgebra,
{
    type FacadeError = Error;

    fn map_provider_error(error: <R as TypedSectorAdmission>::Error) -> Self::FacadeError {
        error.into()
    }

    fn fusion_style(provider: &R) -> symtn_core::FusionStyleKind {
        symtn_core::FusionRule::fusion_style(provider)
    }

    fn braiding_style(provider: &R) -> symtn_core::BraidingStyleKind {
        symtn_core::FusionRule::braiding_style(provider)
    }

    fn facade_error_into_error(error: Error) -> Error {
        error
    }

    fn sector_order_key(provider: &R, sector: SectorId) -> symtn_core::SectorOrderKey {
        symtn_core::FusionRule::sector_order_key(provider, sector)
    }

    fn coupled_block_dimension(
        provider: &R,
        product: &FusionProductSpace,
        coupled: SectorId,
    ) -> Result<usize, Self::FacadeError> {
        Ok(product
            .coupled_sector_block_dimensions(provider)?
            .get(&coupled)
            .copied()
            .unwrap_or(0))
    }

    fn fuse_sector_content(
        provider: &R,
        left: &[(SectorId, usize)],
        right: &[(SectorId, usize)],
    ) -> Result<Vec<(SectorId, usize)>, Self::FacadeError> {
        fuse_sector_content(
            left,
            right,
            |a, b| CheckedFusionAlgebra::try_fusion_channels(provider, a, b),
            |a, b, c| CheckedFusionAlgebra::try_nsymbol(provider, a, b, c),
        )
        .map_err(Into::into)
    }
}

impl<R> TypedTensorRootDispatch<R> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols
        + CheckedFusionAlgebra
        + SectorCodec,
{
    fn build_root(
        provider: Arc<R>,
        homspace: FusionTreeHomSpace,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::FacadeError> {
        BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(
            provider, homspace,
        )
        .map_err(Into::into)
    }

    fn build_root_with(
        provider: Arc<R>,
        build_homspace: impl FnOnce() -> Result<FusionTreeHomSpace, Self::FacadeError>,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::FacadeError> {
        BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked_with(
            provider,
            build_homspace,
        )
    }
}

impl<R, D> TypedTensorConstructionDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols
        + CheckedFusionAlgebra
        + SectorCodec,
    <R as MultiplicityFreeFusionSymbols>::Scalar:
        CategoricalScalar + symtn_tensors::DenseRecouplingScalar,
    D: TensorScalar
        + symtn_tensors::RecouplingCoefficientAction<<R as MultiplicityFreeFusionSymbols>::Scalar>,
{
    fn build_construction_root(
        provider: Arc<R>,
        homspace: FusionTreeHomSpace,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::FacadeError> {
        BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(
            provider, homspace,
        )
        .map_err(Into::into)
    }
}

impl<R> TypedTensorModeDispatch<R> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
{
    type FacadeError = GenericTensorError<<R as CheckedGenericFusion>::Error>;

    fn map_provider_error(error: <R as TypedSectorAdmission>::Error) -> Self::FacadeError {
        GenericTensorError::Structure(CheckedGenericStructureError::Provider(error))
    }

    // Why not ask the provider: admission already required the Generic
    // style (`validate_checked_generic_style`), so an op on admitted data
    // never asks again and cannot be swayed by a provider whose answer
    // changed since.
    fn fusion_style(_provider: &R) -> symtn_core::FusionStyleKind {
        symtn_core::FusionStyleKind::Generic
    }

    fn braiding_style(provider: &R) -> symtn_core::BraidingStyleKind {
        CheckedGenericFusion::braiding_style(provider)
    }

    // Why lossy: `TensorRef` is provider-neutral, so its adjoint constructor
    // returns `Error`. The only non-facade failure of a checked-Generic
    // adjoint is a broken invariant of an admitted layout.
    fn facade_error_into_error(error: Self::FacadeError) -> Error {
        match error {
            GenericTensorError::Facade(error) => error,
            other => Error::InvalidArgument(format!("adjoint view: {other}")),
        }
    }

    fn sector_order_key(provider: &R, sector: SectorId) -> symtn_core::SectorOrderKey {
        CheckedGenericFusion::sector_order_key(provider, sector)
    }

    fn coupled_block_dimension(
        provider: &R,
        product: &FusionProductSpace,
        coupled: SectorId,
    ) -> Result<usize, Self::FacadeError> {
        Ok(
            symtn_matrixalgebra::seam::coupled_sector_block_dimensions_generic_checked(
                product, provider,
            )?
            .get(&coupled)
            .copied()
            .unwrap_or(0),
        )
    }

    fn fuse_sector_content(
        provider: &R,
        left: &[(SectorId, usize)],
        right: &[(SectorId, usize)],
    ) -> Result<Vec<(SectorId, usize)>, Self::FacadeError> {
        fuse_sector_content(
            left,
            right,
            |a, b| CheckedGenericFusion::try_fusion_channels(provider, a, b),
            |a, b, c| CheckedGenericFusion::try_nsymbol(provider, a, b, c),
        )
        .map_err(<Self as TypedTensorModeDispatch<R>>::map_provider_error)
    }
}

impl<R> TypedTensorRootDispatch<R> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
{
    fn build_root(
        provider: Arc<R>,
        homspace: FusionTreeHomSpace,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::FacadeError> {
        BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(provider, homspace)
            .map_err(Into::into)
    }

    fn build_root_with(
        provider: Arc<R>,
        build_homspace: impl FnOnce() -> Result<FusionTreeHomSpace, Self::FacadeError>,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::FacadeError> {
        BoundDynamicFusionMapSpace::from_final_homspace_generic_checked_with(
            provider,
            build_homspace,
        )
    }
}

impl<R, D> TypedTensorConstructionDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericFusion,
    D: TensorScalar,
{
    fn build_construction_root(
        provider: Arc<R>,
        homspace: FusionTreeHomSpace,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::FacadeError> {
        BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(provider, homspace)
            .map_err(Into::into)
    }
}

impl<R, D> TypedTensorTransformDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols
        + CheckedFusionAlgebra
        + SectorCodec,
    <R as MultiplicityFreeFusionSymbols>::Scalar:
        CategoricalScalar + symtn_tensors::DenseRecouplingScalar,
    D: TensorScalar + MultiplicityFreeTransformExecution<R>,
{
    fn transform_bond_spectrum(
        tensor: &TensorMap<R, D>,
        spectrum: &[symtn_matrixalgebra::SectorSpectrum<D>],
        operation: &TreeTransformOperation,
        require_representable: bool,
    ) -> Result<BondTransform<R, D>, Error> {
        D::transform_bond_spectrum(tensor, spectrum, operation, require_representable)
    }

    fn try_lazy_adjoint_transform<S: TransformExec<R, D>>(
        tensor: &TensorMap<R, D, S>,
        operation: &TreeTransformOperation,
    ) -> Result<Option<TensorMap<R, D, S>>, Error> {
        D::try_lazy_adjoint(tensor, operation)
    }

    fn transform<S: TransformExec<R, D>>(
        tensor: &TensorMap<R, D, S>,
        operation: TreeTransformOperation,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, S), Error> {
        D::transform(tensor, operation)
    }

    fn transform_into<S: TransformExec<R, D>>(
        source: &TensorMap<R, D, S>,
        destination: &mut TensorMap<R, D, S>,
        alpha: D,
        beta: D,
        operation: impl FnOnce() -> Result<TreeTransformOperation, Error>,
        exact_layout_probe: impl FnOnce(&mut dyn FnMut(symtn_tensors::TreeTransformOperationView<'_>)),
    ) -> Result<(), Error> {
        D::transform_into(
            source,
            destination,
            alpha,
            beta,
            operation,
            exact_layout_probe,
        )
    }
}

impl<R, D> TypedTensorTransformDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericRigidSymbols<Scalar = f64>,
    D: TensorScalar,
{
    fn transform_bond_spectrum(
        tensor: &TensorMap<R, D>,
        spectrum: &[symtn_matrixalgebra::SectorSpectrum<D>],
        operation: &TreeTransformOperation,
        require_representable: bool,
    ) -> Result<BondTransform<R, D>, Self::FacadeError> {
        let source = tensor.logical_space();
        let mut lease = tensor.runtime.lease_context()?;
        let (destination, outcome) = lease
            .context()
            .generic_lane::<D>()?
            .tree_context_mut()
            .tree_transform_structure_checked_generic_in(
                source,
                operation,
                |structure, preview| {
                    crate::tensor_product::bond_spectrum(
                        source.space().structure(),
                        preview,
                        structure,
                        spectrum,
                        require_representable,
                    )
                },
                || symtn_matrixalgebra::seam::diagonal_bond_data(source.space(), spectrum, &|v| v),
            )?;
        Ok(BondTransform::from_outcome(destination, outcome))
    }

    fn try_lazy_adjoint_transform<S: TransformExec<R, D>>(
        tensor: &TensorMap<R, D, S>,
        operation: &TreeTransformOperation,
    ) -> Result<Option<TensorMap<R, D, S>>, Self::FacadeError> {
        lazy_adjoint_of_transformed_parent(tensor, operation)
    }

    fn transform<S: TransformExec<R, D>>(
        tensor: &TensorMap<R, D, S>,
        operation: TreeTransformOperation,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, S), Self::FacadeError> {
        // Why owned: the facade routes every lazy adjoint through
        // `try_lazy_adjoint_transform` above.
        let body = tensor
            .owned_body()
            .ok_or_else(|| internal_layout_error("checked Generic transform input is owned"))?;
        let src_len = transform_source_len(body)?;
        let mut lease = tensor.runtime.lease_context()?;
        let staged = lease
            .context()
            .generic_lane::<D>()?
            .tree_context_mut()
            .stage_tree_transform_checked_generic_in(&body.space, None, src_len, &operation)?;
        let data = S::replay_owned(
            EXEC,
            TreeReplay::staged::<CheckedGenericTreeLane, _, _>(&tensor.runtime, lease, &staged),
            tensor,
            D::from_real(1.0),
        )
        .map_err(checked_generic_exec_error)?;
        Ok((staged.commit()?, data))
    }

    /// No exact-layout probe: its key family is multiplicity-free direct
    /// only, so a checked entry can never hit it.
    fn transform_into<S: TransformExec<R, D>>(
        source: &TensorMap<R, D, S>,
        destination: &mut TensorMap<R, D, S>,
        alpha: D,
        beta: D,
        operation: impl FnOnce() -> Result<TreeTransformOperation, Error>,
        _exact_layout_probe: impl FnOnce(&mut dyn FnMut(symtn_tensors::TreeTransformOperationView<'_>)),
    ) -> Result<(), Self::FacadeError> {
        let mut lease = source.runtime.lease_context()?;
        let operation = operation()?;
        let adjoint_of = match &source.repr {
            TypedTensorRepr::Owned(_) => None,
            TypedTensorRepr::Adjoint(view) => Some(&view.parent.space),
        };
        let destination_space = destination.logical_space().space().clone();
        let staged = lease
            .context()
            .generic_lane::<D>()?
            .tree_context_mut()
            .stage_tree_transform_into_checked_generic_in::<R, Self::FacadeError>(
                source.logical_space(),
                adjoint_of,
                transform_source_len(source.storage_body())?,
                &operation,
                &destination_space,
            )?;
        let destination_data = destination_storage_mut(destination)?;
        with_transform_into_source(source, &operation, alpha, beta, |input| {
            S::replay_into(
                EXEC,
                TreeReplay::staged::<CheckedGenericTreeLane, _, _>(&source.runtime, lease, &staged),
                input,
                destination_data,
                alpha,
                beta,
            )
        })
        .map_err(checked_generic_exec_error)?;
        staged.commit()?;
        Ok(())
    }
}

impl<R, D> TypedTensorTwistDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols
        + CheckedFusionAlgebra
        + SectorCodec,
    R::Scalar: TwistCoefficient<D>,
    MultiplicityFreeAdmissionMode: symtn_tensors::RigidCoefficientAlgebra<R>
        + symtn_tensors::CoefficientAlgebra<R, Error = symtn_tensors::OperationError>,
    D: TensorScalar,
{
    type Coeff = R::Scalar;

    fn twist_values<'a>(
        provider: &'a R,
        structure: &symtn_core::BlockStructure,
        codomain_rank: usize,
        legs: &[usize],
    ) -> Result<Option<impl Fn(SectorId) -> R::Scalar + 'a>, Error> {
        if twist_is_identity_over_blocks(provider, structure, codomain_rank, legs)? {
            return Ok(None);
        }
        Ok(Some(move |sector| provider.twist_scalar(sector)))
    }
}

impl<R, D> TypedTensorTwistDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericPivotal<Scalar = f64>,
    D: TensorScalar,
{
    type Coeff = f64;

    fn twist_values<'a>(
        provider: &'a R,
        structure: &symtn_core::BlockStructure,
        codomain_rank: usize,
        legs: &[usize],
    ) -> Result<Option<impl Fn(SectorId) -> f64 + 'a>, Self::FacadeError> {
        // Why stage first: a late provider failure must leave no scaled
        // payload to publish, and replay over the staged table must never
        // call the provider.
        let mut staged = HashMap::<SectorId, f64>::new();
        for block_index in 0..structure.block_count() {
            let block = structure
                .block(block_index)
                .map_err(Error::from)
                .map_err(GenericTensorError::Facade)?;
            let BlockKey::FusionTree(key) = block.key() else {
                continue;
            };
            for &leg in legs {
                let sector = uncoupled_sector_of_leg(key, codomain_rank, leg);
                if let std::collections::hash_map::Entry::Vacant(entry) = staged.entry(sector) {
                    let value = provider.try_twist_scalar(sector).map_err(|error| {
                        GenericTensorError::Plan(CheckedGenericPlanError::Provider(error))
                    })?;
                    entry.insert(value);
                }
            }
        }
        if staged.values().all(|&value| value == 1.0) {
            return Ok(None);
        }
        Ok(Some(move |sector| staged[&sector]))
    }
}

impl<R, D> TypedTensorFlipDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols
        + CheckedFusionAlgebra
        + SectorCodec,
    R::Scalar: TwistCoefficient<D>,
    D: TensorScalar,
{
    type Coeff = R::Scalar;

    fn root(
        space: &BoundDynamicFusionMapSpace<R>,
        homspace: FusionTreeHomSpace,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Error> {
        Ok(space.derive_from_final_homspace(homspace)?)
    }

    fn pivotal_values<'a>(
        provider: &'a R,
        _structure: &symtn_core::BlockStructure,
        _codomain_rank: usize,
        _occurrences: &[(usize, bool)],
    ) -> Result<impl Fn(SectorId) -> (R::Scalar, R::Scalar) + 'a, Error> {
        Ok(move |sector| {
            (
                provider.frobenius_schur_phase_scalar(sector),
                provider.twist_scalar(sector),
            )
        })
    }
}

impl<R, D> TypedTensorFlipDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericPivotal<Scalar = f64>,
    D: TensorScalar,
{
    type Coeff = f64;

    fn root(
        space: &BoundDynamicFusionMapSpace<R>,
        homspace: FusionTreeHomSpace,
    ) -> Result<BoundDynamicFusionMapSpace<R>, Self::FacadeError> {
        <Self as TypedTensorRootDispatch<R>>::build_root(Arc::clone(space.provider_arc()), homspace)
    }

    fn pivotal_values<'a>(
        provider: &'a R,
        structure: &symtn_core::BlockStructure,
        codomain_rank: usize,
        occurrences: &[(usize, bool)],
    ) -> Result<impl Fn(SectorId) -> (f64, f64) + 'a, Self::FacadeError> {
        // Why stage first: any pivotal failure leaves the input and
        // destination unpublished, while replay over this table cannot query
        // the provider.
        let mut staged = HashMap::<SectorId, (f64, f64)>::new();
        for block_index in 0..structure.block_count() {
            let block = structure
                .block(block_index)
                .map_err(Error::from)
                .map_err(GenericTensorError::Facade)?;
            let BlockKey::FusionTree(key) = block.key() else {
                continue;
            };
            for &(leg, _) in occurrences {
                let sector = uncoupled_sector_of_leg(key, codomain_rank, leg);
                if let std::collections::hash_map::Entry::Vacant(entry) = staged.entry(sector) {
                    let chi =
                        provider
                            .try_frobenius_schur_phase_scalar(sector)
                            .map_err(|error| {
                                GenericTensorError::Plan(CheckedGenericPlanError::Provider(error))
                            })?;
                    let theta = provider.try_twist_scalar(sector).map_err(|error| {
                        GenericTensorError::Plan(CheckedGenericPlanError::Provider(error))
                    })?;
                    entry.insert((chi, theta));
                }
            }
        }
        Ok(move |sector| staged[&sector])
    }
}

impl<R, D> TypedTensorProductDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols
        + CheckedFusionAlgebra
        + SectorCodec
        + CanonicalUnitFusionRule,
    <R as MultiplicityFreeFusionSymbols>::Scalar:
        CategoricalScalar + symtn_tensors::DenseRecouplingScalar,
    D: TensorScalar
        + symtn_tensors::RecouplingCoefficientAction<<R as MultiplicityFreeFusionSymbols>::Scalar>,
{
    fn product(
        lhs_space: &BoundDynamicFusionMapSpace<R>,
        lhs_data: &[D],
        rhs_space: &BoundDynamicFusionMapSpace<R>,
        rhs_data: &[D],
    ) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), Error> {
        Ok(tensorproduct_owned_multiplicity_free(
            BoundDynamicTensorRef::try_new(lhs_space, lhs_data)?,
            BoundDynamicTensorRef::try_new(rhs_space, rhs_data)?,
        )?)
    }
}

impl<R, D> TypedTensorProductDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericRigidSymbols<Scalar = f64>,
    D: TensorScalar,
{
    fn product(
        lhs_space: &BoundDynamicFusionMapSpace<R>,
        lhs_data: &[D],
        rhs_space: &BoundDynamicFusionMapSpace<R>,
        rhs_data: &[D],
    ) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), Self::FacadeError> {
        Ok(tensorproduct_owned_checked_generic(
            lhs_space, lhs_data, rhs_space, rhs_data,
        )?)
    }
}

impl<R, D> TypedTensorContractDispatch<R, D> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols
        + CheckedFusionAlgebra
        + SectorCodec,
    <R as MultiplicityFreeFusionSymbols>::Scalar:
        CategoricalScalar + symtn_tensors::DenseRecouplingScalar,
    D: TensorScalar + MultiplicityFreeContractExecution<R> + MultiplicityFreeTransformExecution<R>,
{
    fn contract<S: ContractExec<R, D>>(
        lhs: &TensorMap<R, D, S>,
        rhs: &TensorMap<R, D, S>,
        spec: &ContractSpec<'_>,
        planar: Option<&symtn_tensors::PlanarContractOrders>,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, S), Error> {
        D::contract(lhs, rhs, spec, planar)
    }

    fn contract_into<S: ContractExec<R, D>>(
        lhs: &TensorMap<R, D, S>,
        rhs: &TensorMap<R, D, S>,
        spec: &ContractSpec<'_>,
        planar: Option<&symtn_tensors::PlanarContractOrders>,
        destination: &mut TensorMap<R, D, S>,
        alpha: D,
        beta: D,
    ) -> Result<(), Error> {
        D::contract_into(lhs, rhs, spec, planar, destination, alpha, beta)
    }

    fn compose<S: ContractExec<R, D>>(
        lhs: &TensorMap<R, D, S>,
        rhs: &TensorMap<R, D, S>,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, S), Error> {
        D::compose(lhs, rhs)
    }
}

impl<R, D> TypedTensorContractDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericRigidSymbols<Scalar = f64>,
    D: TensorScalar,
{
    fn contract<S: ContractExec<R, D>>(
        lhs: &TensorMap<R, D, S>,
        rhs: &TensorMap<R, D, S>,
        spec: &ContractSpec<'_>,
        planar: Option<&symtn_tensors::PlanarContractOrders>,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, S), Self::FacadeError> {
        let output_axes = spec.output_axes();
        let (lhs_operand, rhs_operand) = (lhs.fusion_operand(), rhs.fusion_operand());
        let request = contract_request(lhs_operand, rhs_operand, spec, &output_axes, planar);
        checked_generic_contract_owned(lhs, rhs, request)
    }

    fn contract_into<S: ContractExec<R, D>>(
        lhs: &TensorMap<R, D, S>,
        rhs: &TensorMap<R, D, S>,
        spec: &ContractSpec<'_>,
        planar: Option<&symtn_tensors::PlanarContractOrders>,
        destination: &mut TensorMap<R, D, S>,
        alpha: D,
        beta: D,
    ) -> Result<(), Self::FacadeError> {
        let output_axes = spec.output_axes();
        let mut lease = lhs.runtime.lease_context()?;
        lease.context().generic_lane::<D>()?;
        let (lhs_operand, rhs_operand) = (lhs.fusion_operand(), rhs.fusion_operand());
        let mut staged = symtn_tensors::StagedContract::checked_generic(
            (
                lhs.logical_space(),
                lhs_operand,
                transform_source_len(lhs.storage_body())?,
            ),
            (
                rhs.logical_space(),
                rhs_operand,
                transform_source_len(rhs.storage_body())?,
            ),
            contract_request(lhs_operand, rhs_operand, spec, &output_axes, planar),
        )?;
        staged.require_destination(destination.logical_space().space())?;
        let destination = destination_storage_mut(destination)?;
        let resolution = match staged.try_core::<S::Caps>()? {
            symtn_tensors::CoreRoute::Hit(core) => core,
            symtn_tensors::CoreRoute::Miss(miss) => staged.plan_beyond_core::<S::Caps>(miss)?,
        };
        S::contract_into(
            EXEC,
            ContractReplay::new::<CheckedGenericTreeLane>(
                &lhs.runtime,
                Some(lease),
                &resolution,
                (staged.destination(), staged.destination_len()),
                (lhs_operand, rhs_operand),
            ),
            lhs,
            rhs,
            destination,
            alpha,
            beta,
        )
        .map_err(checked_generic_exec_error)?;
        staged.commit(&resolution)?;
        Ok(())
    }

    fn compose<S: ContractExec<R, D>>(
        lhs: &TensorMap<R, D, S>,
        rhs: &TensorMap<R, D, S>,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, S), Self::FacadeError> {
        checked_generic_contract_owned(lhs, rhs, symtn_tensors::ContractRequest::Compose)
    }
}

/// The checked Generic owned contraction or composition on any storage:
/// one lease, the staging, the planner under the call's transaction, the
/// storage's kernel, then the commit and publication (#2063).
fn checked_generic_contract_owned<'a, R, D, S>(
    lhs: &'a TensorMap<R, D, S>,
    rhs: &'a TensorMap<R, D, S>,
    request: symtn_tensors::ContractRequest<'a>,
) -> Result<(BoundDynamicFusionMapSpace<R>, S), FacadeError<R>>
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericRigidSymbols<Scalar = f64>,
    D: TensorScalar,
    S: ContractExec<R, D>,
{
    // The kernel resolves the generic lane under this lease, once.
    let lease = lhs.runtime.lease_context()?;
    let (lhs_operand, rhs_operand) = (lhs.fusion_operand(), rhs.fusion_operand());
    let mut staged = symtn_tensors::StagedContract::checked_generic(
        (
            lhs.logical_space(),
            lhs_operand,
            transform_source_len(lhs.storage_body())?,
        ),
        (
            rhs.logical_space(),
            rhs_operand,
            transform_source_len(rhs.storage_body())?,
        ),
        request,
    )?;
    let resolution = match staged.try_core::<S::Caps>()? {
        symtn_tensors::CoreRoute::Hit(core) => core,
        symtn_tensors::CoreRoute::Miss(miss) => staged.plan_beyond_core::<S::Caps>(miss)?,
    };
    let data = S::contract_owned(
        EXEC,
        ContractReplay::new::<CheckedGenericTreeLane>(
            &lhs.runtime,
            Some(lease),
            &resolution,
            (staged.destination(), staged.destination_len()),
            (lhs_operand, rhs_operand),
        ),
        lhs,
        rhs,
    )
    .map_err(checked_generic_exec_error)?;
    Ok((staged.commit(&resolution)?, data))
}
