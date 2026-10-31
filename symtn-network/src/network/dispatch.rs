use super::*;
use symtn::typed::Engine;

mod mode_sealed {
    pub trait Sealed {}

    impl Sealed for symtn::sector::MultiplicityFreeAdmissionMode {}
    impl Sealed for symtn::sector::CheckedGenericAdmissionMode {}
}

/// Internal network policy selected by the provider's admission mode and the
/// operand payload storage.
///
/// `S` defaults to Host `Vec<D>`, so `NetworkModeDispatch<R, D>` keeps
/// naming the Host policy. A second instantiation per storage carries the
/// device policy: which primitives exist there, whether destinations may be
/// reused, and how an idle workspace is parked.
#[doc(hidden)]
pub trait NetworkModeDispatch<R, D, S = Vec<D>>:
    mode_sealed::Sealed
    + TypedTensorModeDispatch<R>
    + TypedSpaceModeDispatch<R>
    + TypedAdjointSpace<R>
    + TypedTensorContractDispatch<R, D>
    + TypedTensorRootDispatch<R>
    + TypedTensorTransformDispatch<R, D>
where
    R: TypedSectorAdmission<Mode = Self>,
    D: TensorScalar,
    S: NetworkPayloadStorage<D>,
{
    const REUSE_DESTINATIONS: bool;

    /// The provider's braiding style, which decides whether a network that
    /// contracts is admitted at all.
    fn braiding_style(provider: &R) -> symtn::sector::BraidingStyleKind;

    fn leg_dims<B: TensorStorage<D>>(
        tensor: &TensorMap<R, D, B>,
    ) -> Result<Vec<usize>, FacadeError<R>>;

    /// Lowers a `conj`-marked operand to its adjoint on this storage.
    fn adjoint_operand(
        engine: &mut dyn Engine,
        tensor: &TensorMap<R, D, S>,
    ) -> Result<TensorMap<R, D, S>, FacadeError<R>>;

    fn contract_step(
        engine: &mut dyn Engine,
        lhs: &TensorMap<R, D, S>,
        rhs: &TensorMap<R, D, S>,
        destination: &mut Option<TensorMap<R, D, S>>,
        spec: &ContractSpec<'_>,
    ) -> Result<StepOutput<TensorMap<R, D, S>>, FacadeError<R>>;

    /// Reorients the operand of a zero-step schedule (a relabel), the one
    /// result no contraction step writes in its output orientation.
    fn permute_final(
        engine: &mut dyn Engine,
        tensor: &TensorMap<R, D, S>,
        codomain: &[usize],
        domain: &[usize],
    ) -> Result<TensorMap<R, D, S>, FacadeError<R>>;

    /// Re-attaches the parked destinations to `runtime` and the step
    /// authorities, or discards them all on drift.
    fn activate_parked(
        workspace: &mut NetworkExecutionWorkspace<R, D, S>,
        runtime: &Runtime,
        tensors: &[&TensorMap<R, D, S>],
        steps: &[CompiledStep],
    ) -> Result<(), FacadeError<R>> {
        Ok(workspace.activate_parked(runtime, tensors, steps)?)
    }

    /// Parks every idle destination without its Runtime and provider.
    fn park_workspace(workspace: &mut NetworkExecutionWorkspace<R, D, S>) {
        workspace.park_runtime_owners();
    }
}

#[doc(hidden)]
pub enum StepOutput<T> {
    Returned(T),
    Overwritten,
}

impl<T> StepOutput<T> {
    pub(super) fn get<'a>(&'a self, destination: &'a Option<T>) -> &'a T {
        match self {
            Self::Returned(value) => value,
            Self::Overwritten => destination
                .as_ref()
                .expect("successful overwrite retains its destination"),
        }
    }

    pub(super) fn take(self, destination: &mut Option<T>) -> T {
        match self {
            Self::Returned(value) => value,
            Self::Overwritten => destination
                .take()
                .expect("successful overwrite retains its destination"),
        }
    }
}

/// The step of every mode and storage: a retained destination is overwritten
/// through `contract_into` (`beta = 0`), otherwise the step returns the eager
/// result.
fn host_contract_step<R, D, S>(
    engine: &mut dyn Engine,
    lhs: &TensorMap<R, D, S>,
    rhs: &TensorMap<R, D, S>,
    destination: &mut Option<TensorMap<R, D, S>>,
    spec: &ContractSpec<'_>,
) -> Result<StepOutput<TensorMap<R, D, S>>, FacadeError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorContractDispatch<R, D>,
    D: TensorScalar,
    S: ContractExec<R, D>,
{
    #[cfg(test)]
    crate::probe::hit(crate::probe::Event::ContractStep);
    if let Some(destination) = destination {
        lhs.contract_into(
            engine,
            rhs,
            spec,
            destination,
            D::from_real(1.0),
            D::from_real(0.0),
        )?;
        Ok(StepOutput::Overwritten)
    } else {
        Ok(StepOutput::Returned(lhs.contract(engine, rhs, spec)?))
    }
}

/// The multiplicity-free policy on every storage that has the transform and
/// contraction kernels: Host `Vec<D>` and, with the `cuda` feature, the
/// device storage for the multiplicity-free rules its kernels admit.
///
/// Each step is the typed general-axes `contract` / `contract_into` with
/// `beta = 0` (the Host-compiled DynamicTree or core route, fermionic twist
/// included), writing the step's own output orientation. Every step but the
/// last overwrites a destination the workspace kept from the previous call
/// (the device overwrite needs no reset of it); the final schedule slot
/// leaves the workspace and therefore always allocates a fresh returning
/// output. Slots, producers, input snapshots and parking are the same on
/// every storage, which is what lets one pool serve both placements. Device
/// admission is decided before the first step by
/// `PlannedNetwork::execute` (`cuda_operand_admission`).
impl<R, D, S> NetworkModeDispatch<R, D, S> for MultiplicityFreeAdmissionMode
where
    R: TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = FusionAlgebraError,
            Mode = MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: TensorScalar,
    S: NetworkPayloadStorage<D> + TransformExec<R, D> + ContractExec<R, D>,
{
    const REUSE_DESTINATIONS: bool = true;

    fn braiding_style(provider: &R) -> symtn::sector::BraidingStyleKind {
        symtn::sector::FusionRule::braiding_style(provider)
    }

    fn leg_dims<B: TensorStorage<D>>(tensor: &TensorMap<R, D, B>) -> Result<Vec<usize>, Error> {
        tensor.leg_dims()
    }

    fn adjoint_operand(
        engine: &mut dyn Engine,
        tensor: &TensorMap<R, D, S>,
    ) -> Result<TensorMap<R, D, S>, Error> {
        tensor.adjoint(engine)
    }

    fn contract_step(
        engine: &mut dyn Engine,
        lhs: &TensorMap<R, D, S>,
        rhs: &TensorMap<R, D, S>,
        destination: &mut Option<TensorMap<R, D, S>>,
        spec: &ContractSpec<'_>,
    ) -> Result<StepOutput<TensorMap<R, D, S>>, Error> {
        host_contract_step(engine, lhs, rhs, destination, spec)
    }

    fn permute_final(
        engine: &mut dyn Engine,
        tensor: &TensorMap<R, D, S>,
        codomain: &[usize],
        domain: &[usize],
    ) -> Result<TensorMap<R, D, S>, Error> {
        tensor.permute(engine, codomain, domain)
    }
}

impl<R, D> NetworkModeDispatch<R, D> for CheckedGenericAdmissionMode
where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericRigidSymbols<Scalar = f64>,
    D: TensorScalar,
{
    const REUSE_DESTINATIONS: bool = true;

    fn braiding_style(provider: &R) -> symtn::sector::BraidingStyleKind {
        CheckedGenericFusion::braiding_style(provider)
    }

    fn leg_dims<B: TensorStorage<D>>(
        tensor: &TensorMap<R, D, B>,
    ) -> Result<Vec<usize>, FacadeError<R>> {
        tensor
            .codomain()
            .into_iter()
            .chain(tensor.domain())
            .map(|space| space.dim().map(|dimension| dimension.round() as usize))
            .collect()
    }

    fn adjoint_operand(
        engine: &mut dyn Engine,
        tensor: &TensorMap<R, D>,
    ) -> Result<TensorMap<R, D>, FacadeError<R>> {
        tensor.adjoint(engine)
    }

    fn contract_step(
        engine: &mut dyn Engine,
        lhs: &TensorMap<R, D>,
        rhs: &TensorMap<R, D>,
        destination: &mut Option<TensorMap<R, D>>,
        spec: &ContractSpec<'_>,
    ) -> Result<StepOutput<TensorMap<R, D>>, FacadeError<R>> {
        host_contract_step(engine, lhs, rhs, destination, spec)
    }

    fn permute_final(
        engine: &mut dyn Engine,
        tensor: &TensorMap<R, D>,
        codomain: &[usize],
        domain: &[usize],
    ) -> Result<TensorMap<R, D>, FacadeError<R>> {
        tensor.permute(engine, codomain, domain)
    }
}
