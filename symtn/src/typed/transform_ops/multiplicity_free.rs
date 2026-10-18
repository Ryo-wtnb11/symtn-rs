use super::*;

/// `data = alpha · diag(diagonal) + beta · data` on the bond blocks of
/// `structure`, where `diagonal[i]` is block `i`'s diagonal
/// ([`crate::tensor_product::bond_spectrum`] proved the bijection and the
/// shapes). `beta == 0` never reads `data` and `alpha == 0` never reads the
/// diagonal, as for the dense replay.
pub(super) fn axpby_bond_diagonal<D: TensorScalar>(
    structure: &symtn_core::BlockStructure,
    data: &mut [D],
    diagonal: &[symtn_matrixalgebra::SectorSpectrum<D>],
    alpha: D,
    beta: D,
) -> Result<(), Error> {
    scale_by_beta(data, beta);
    if alpha.is_zero() {
        return Ok(());
    }
    for (index, entry) in diagonal.iter().enumerate() {
        let block = structure.block(index)?;
        let strides = block.strides();
        let stride = strides[0] + strides[1];
        for (position, &value) in entry.values.iter().enumerate() {
            let slot = &mut data[block.offset() + position * stride];
            *slot = *slot + alpha * value;
        }
    }
    Ok(())
}

/// The multiplicity-free [`TypedTensorTransformDispatch::transform_into`] of
/// either coefficient lane: the exact-layout hit replays its proved
/// transformer; a miss stages the owned transform against the caller's
/// destination, the storage replays into it, and the proof is recorded for
/// the next call.
#[allow(private_bounds)]
fn tree_transform_into_multiplicity_free<R, D, S>(
    source: &TensorMap<R, D, S>,
    destination: &mut TensorMap<R, D, S>,
    alpha: D,
    beta: D,
    operation: impl FnOnce() -> Result<TreeTransformOperation, Error>,
    exact_layout_probe: impl FnOnce(&mut dyn FnMut(symtn_tensors::TreeTransformOperationView<'_>)),
) -> Result<(), Error>
where
    R: MultiplicityFreeRigidSymbols + CheckedFusionAlgebra + SectorCodec,
    <R as MultiplicityFreeFusionSymbols>::Scalar:
        CategoricalScalar + symtn_tensors::DenseRecouplingScalar,
    D: TensorScalar
        + crate::runtime::MultiplicityFreeCoefficientLane<
            <R as MultiplicityFreeFusionSymbols>::Scalar,
        >,
    S: TransformExec<R, D>,
{
    // Only an owned dense source keys the exact-layout proof: its key is a
    // direct, unconjugated source content, which neither a lazy adjoint's
    // parent nor a compact payload is (#1870).
    let direct = match &source.repr {
        TypedTensorRepr::Owned(body) => match body.data.as_ref() {
            TypedData::Dense(data) => body
                .space
                .space()
                .admission()
                .rule_identity()
                .map(|identity| (identity, body, data)),
            TypedData::Diagonal(_) => None,
        },
        TypedTensorRepr::Adjoint(_) => None,
    };
    let mut lease = source.runtime.lease_context()?;
    D::lane(lease.context())?;
    // A destination proved once to be this operation's result space of
    // this source hits the completed transformer directly: the owned
    // operation and the result space are not rebuilt.
    let mut exact_layout_hit = None;
    if let Some((identity, body, _)) = direct {
        exact_layout_probe(&mut |view| {
            exact_layout_hit = crate::runtime::Runtime::exact_layout_tree_pair_hit_of(
                identity,
                view,
                &body.space,
                destination.logical_space(),
            );
        });
    }
    if let (Some(structure), Some((_, body, data))) = (exact_layout_hit, direct) {
        let destination_structure = Arc::clone(destination.logical_space().space().structure());
        let nout = destination.codomain_rank();
        let replay = TreeReplay::new::<MultiplicityFreeTreeLane>(
            &source.runtime,
            lease,
            &structure,
            &destination_structure,
            body.space.space().structure(),
            nout,
        );
        return Ok(S::replay_into(
            EXEC,
            replay,
            TransformIntoSource::Stored(data),
            destination_storage_mut(destination)?,
            alpha,
            beta,
        )?);
    }
    let operation = operation()?;
    let adjoint_of = match &source.repr {
        TypedTensorRepr::Owned(_) => None,
        TypedTensorRepr::Adjoint(view) => Some(&view.parent.space),
    };
    let destination_space = destination.logical_space().space().clone();
    let staged = D::lane(lease.context())?
        .tree_context_mut()
        .stage_tree_transform_into_multiplicity_free_in(
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
            TreeReplay::staged::<MultiplicityFreeTreeLane, _, _>(&source.runtime, lease, &staged),
            input,
            destination_data,
            alpha,
            beta,
        )
    })?;
    staged.commit()?;
    if let Some((identity, body, _)) = direct {
        // Retention is an optimization: a lookup-only key keeps no proof.
        let _ = crate::runtime::Runtime::admit_exact_tree_pair_layout_of::<
            R,
            <R as MultiplicityFreeFusionSymbols>::Scalar,
        >(
            identity,
            &operation,
            &body.space,
            destination.logical_space(),
        );
    }
    Ok(())
}

/// The owned multiplicity-free transform of either coefficient lane: one
/// staging, the storage's replay into a fresh payload, and the commit.
#[allow(private_bounds)]
fn tree_transform_multiplicity_free_owned<R, D, S>(
    tensor: &TensorMap<R, D, S>,
    operation: TreeTransformOperation,
) -> Result<(BoundDynamicFusionMapSpace<R>, S), Error>
where
    R: MultiplicityFreeRigidSymbols + CheckedFusionAlgebra + SectorCodec,
    <R as MultiplicityFreeFusionSymbols>::Scalar:
        CategoricalScalar + symtn_tensors::DenseRecouplingScalar,
    D: TensorScalar
        + crate::runtime::MultiplicityFreeCoefficientLane<
            <R as MultiplicityFreeFusionSymbols>::Scalar,
        >,
    S: TransformExec<R, D>,
{
    // Leasing rather than locking: independent operations on one runtime
    // must not serialize behind each other.
    let mut lease = tensor.runtime.lease_context()?;
    // Why owned: a lazy adjoint input already took the mode's lazy-adjoint
    // arm (`try_lazy_adjoint`), for every coefficient scalar.
    let body = tensor
        .owned_body()
        .ok_or_else(|| internal_layout_error("multiplicity-free transform input is owned"))?;
    let src_len = transform_source_len(body)?;
    #[cfg(test)]
    super::probe::hit(super::probe::Event::TreeTransformSeamCall);
    let staged = D::lane(lease.context())?
        .tree_context_mut()
        .stage_tree_transform_multiplicity_free_in(&body.space, None, src_len, &operation)?;
    let data = S::replay_owned(
        EXEC,
        TreeReplay::staged::<MultiplicityFreeTreeLane, _, _>(&tensor.runtime, lease, &staged),
        tensor,
        D::from_real(1.0),
    )?;
    Ok((staged.commit()?, data))
}

/// The multiplicity-free [`TypedTensorTransformDispatch::transform_bond_spectrum`]
/// of either coefficient lane.
#[allow(private_bounds)]
fn transform_bond_spectrum_multiplicity_free<R, D>(
    tensor: &TensorMap<R, D>,
    spectrum: &[symtn_matrixalgebra::SectorSpectrum<D>],
    operation: &TreeTransformOperation,
    require_representable: bool,
) -> Result<BondTransform<R, D>, Error>
where
    R: MultiplicityFreeRigidSymbols + CheckedFusionAlgebra + SectorCodec,
    <R as MultiplicityFreeFusionSymbols>::Scalar:
        CategoricalScalar + symtn_tensors::DenseRecouplingScalar,
    D: TensorScalar
        + crate::runtime::MultiplicityFreeCoefficientLane<
            <R as MultiplicityFreeFusionSymbols>::Scalar,
        >,
{
    let source = tensor.logical_space();
    let mut lease = tensor.runtime.lease_context()?;
    let (destination, outcome) = D::lane(lease.context())?
        .tree_context_mut()
        .tree_transform_structure_multiplicity_free_in(
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

impl<R, D> BondTransform<R, D> {
    pub(in crate::typed) fn from_outcome(
        destination: BoundDynamicFusionMapSpace<R>,
        outcome: symtn_tensors::StructureOutcome<Vec<symtn_matrixalgebra::SectorSpectrum<D>>, D>,
    ) -> Self {
        let output = match outcome {
            symtn_tensors::StructureOutcome::Read(spectrum) => BondOutput::Spectrum(spectrum),
            symtn_tensors::StructureOutcome::Replayed(data) => BondOutput::Dense(data),
        };
        Self {
            destination,
            output,
        }
    }
}

/// TensorKit `permute(t') = adjoint(permute(parent, adjointtensorindices(..)))`
/// and its `braid`/`transpose`/`repartition` twins: the lazy adjoint of the
/// parent's owned transform, one body for every mode. `None` unless `tensor`
/// is a lazy adjoint.
pub(in crate::typed) fn lazy_adjoint_of_transformed_parent<R, D, S>(
    tensor: &TensorMap<R, D, S>,
    operation: &TreeTransformOperation,
) -> Result<Option<TensorMap<R, D, S>>, FacadeError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorTransformDispatch<R, D> + TypedAdjointSpace<R>,
    D: TensorScalar,
    S: TransformExec<R, D>,
{
    let TypedTensorRepr::Adjoint(view) = &tensor.repr else {
        return Ok(None);
    };
    let parent_space = view.parent.space.space();
    let lowered =
        lower_adjoint_tree_transform_operation(parent_space.nout(), parent_space.nin(), operation)?;
    TensorMap {
        runtime: tensor.runtime.clone(),
        repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
    }
    .tree_transform(lowered)?
    .adjoint()
    .map(Some)
}

impl<R, D> MultiplicityFreeTransformExecution<R> for D
where
    R: TypedSectorAdmission<Error = FusionAlgebraError, Mode = MultiplicityFreeAdmissionMode>
        + MultiplicityFreeRigidSymbols
        + CheckedFusionAlgebra
        + SectorCodec,
    <R as MultiplicityFreeFusionSymbols>::Scalar:
        CategoricalScalar + symtn_tensors::DenseRecouplingScalar,
    D: TensorScalar
        + crate::runtime::MultiplicityFreeCoefficientLane<
            <R as MultiplicityFreeFusionSymbols>::Scalar,
        >,
{
    fn transform_bond_spectrum(
        tensor: &TensorMap<R, Self>,
        spectrum: &[symtn_matrixalgebra::SectorSpectrum<Self>],
        operation: &TreeTransformOperation,
        require_representable: bool,
    ) -> Result<BondTransform<R, Self>, Error> {
        transform_bond_spectrum_multiplicity_free(
            tensor,
            spectrum,
            operation,
            require_representable,
        )
    }

    fn try_lazy_adjoint<S: TransformExec<R, Self>>(
        tensor: &TensorMap<R, Self, S>,
        operation: &TreeTransformOperation,
    ) -> Result<Option<TensorMap<R, Self, S>>, Error> {
        lazy_adjoint_of_transformed_parent(tensor, operation)
    }

    fn transform<S: TransformExec<R, Self>>(
        tensor: &TensorMap<R, Self, S>,
        operation: TreeTransformOperation,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, S), Error> {
        tree_transform_multiplicity_free_owned(tensor, operation)
    }

    fn transform_into<S: TransformExec<R, Self>>(
        source: &TensorMap<R, Self, S>,
        destination: &mut TensorMap<R, Self, S>,
        alpha: Self,
        beta: Self,
        operation: impl FnOnce() -> Result<TreeTransformOperation, Error>,
        exact_layout_probe: impl FnOnce(&mut dyn FnMut(symtn_tensors::TreeTransformOperationView<'_>)),
    ) -> Result<(), Error> {
        tree_transform_into_multiplicity_free(
            source,
            destination,
            alpha,
            beta,
            operation,
            exact_layout_probe,
        )
    }
}
