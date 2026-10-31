use super::*;
use symtn::typed::Engine;

mod network_operand_sealed {
    pub trait Sealed {}
}

/// A tensor type [`Network::contract`] accepts: the typed Host tensor and,
/// with the `cuda` feature, the typed CUDA tensor. Every operand of one call
/// has the same provider, scalar and storage, which Rust checks.
pub trait NetworkOperand: network_operand_sealed::Sealed + Sized {
    /// The error [`Network::contract`] returns for this operand type.
    type Error;

    #[doc(hidden)]
    fn contract_network(
        engine: &mut dyn Engine,
        network: &Network,
        tensors: &[&Self],
    ) -> Result<Self, Self::Error>;

    /// [`contract!`](crate::contract)'s operand: method auto-deref takes
    /// both a tensor and a reference to one (`(w.get(0)?)[i; j]`) to
    /// `&Self`, which `&(expr)` cannot.
    #[doc(hidden)]
    fn __contract_operand(&self) -> &Self {
        self
    }
}

impl Network {
    /// Contracts `tensors` eagerly through the Runtime's plan cache.
    ///
    /// The plan is looked up by the network's topology — labels, `conj`
    /// markers, written `;` splits, output, the operands' codomain ranks —
    /// and the optimizer of the operands' Runtime
    /// ([`PlanCacheConfig::optimizer`](crate::PlanCacheConfig)), so a repeated
    /// call reuses the plan and a pooled workspace. Leg dimensions are not
    /// part of the key: [`ReplanPolicy`](crate::ReplanPolicy) decides when a
    /// dimension change replans. The cache's statistics, persistence and
    /// configuration are [`plan_cache_stats`](crate::plan_cache_stats),
    /// [`save_plan_cache`](crate::save_plan_cache) /
    /// [`load_plan_cache`](crate::load_plan_cache) and
    /// [`configure_plan_cache`](crate::configure_plan_cache).
    ///
    /// Use [`Network::plan`] and [`PlannedNetwork::execute`] instead to pick
    /// the optimizer per call or to own the plan and workspace.
    ///
    /// Every metadata rejection — operand count, labels against ranks and
    /// `;` splits, contracted spaces, Runtime and rule identity, braiding,
    /// and on a device the placement and dense payloads — is decided before
    /// the plan lookup, so a rejected call leaves the cache untouched.
    ///
    /// # Errors
    ///
    /// The typed error of the operands' mode, as [`PlannedNetwork::execute`]
    /// returns it, or [`Error`] for device operands.
    pub fn contract<T: NetworkOperand>(
        &self,
        engine: &mut dyn Engine,
        tensors: &[&T],
    ) -> Result<T, T::Error> {
        T::contract_network(engine, self, tensors)
    }

    fn validate_operand_count(&self, operands: usize) -> Result<(), Error> {
        if operands != self.inputs.len() {
            return Err(invalid(format!(
                "network has {} operands but {} tensors were given",
                self.inputs.len(),
                operands
            )));
        }
        Ok(())
    }

    fn contract_preflight<R, D, S>(
        &self,
        tensors: &[&TensorMap<R, D, S>],
    ) -> Result<(), FacadeError<R>>
    where
        R: TypedSectorAdmission,
        R::Mode: NetworkModeDispatch<R, D>,
        D: TensorScalar,
        S: TensorStorage<D>,
    {
        network_operand_preflight(
            tensors,
            &self.inputs,
            &self.conj,
            &self.codomain_splits,
            &self.contracted,
        )
    }
}

fn runtime_optimizer<R, D, S>(tensors: &[&TensorMap<R, D, S>]) -> crate::Optimizer
where
    S: TensorStorage<D>,
{
    tensors
        .first()
        .map(|tensor| crate::plan_cache_config(tensor.runtime()).optimizer)
        .unwrap_or_default()
}

/// The storage half of [`NetworkOperand`] and [`PlannedNetwork::execute`]:
/// the operand admission decided before any plan lookup or step. Host
/// checks braiding; the device adds the CUDA engine, placement and
/// dense-payload admission (`cuda_operand_admission`).
#[doc(hidden)]
pub trait NetworkOperandStorage<R, D>: NetworkPayloadStorage<D> + Send + Sync + Sized
where
    R: TypedSectorAdmission,
    R::Mode: NetworkModeDispatch<R, D, Self>,
    D: TensorScalar,
{
    fn admit_operands(
        engine: &mut dyn Engine,
        tensors: &[&TensorMap<R, D, Self>],
        contracts: bool,
    ) -> Result<(), FacadeError<R>>;
}

impl<R, D> NetworkOperandStorage<R, D> for Vec<D>
where
    R: TypedSectorAdmission,
    R::Mode: NetworkModeDispatch<R, D>,
    D: TensorScalar + Send + Sync + 'static,
{
    fn admit_operands(
        _engine: &mut dyn Engine,
        tensors: &[&TensorMap<R, D>],
        contracts: bool,
    ) -> Result<(), FacadeError<R>> {
        reject_non_symmetric_network(tensors, contracts)
    }
}

#[cfg(feature = "cuda")]
impl<R, D> NetworkOperandStorage<R, D> for CudaStorage<D>
where
    R: TypedSectorAdmission<
            Sector = <R as SectorCodec>::Sector,
            Error = FusionAlgebraError,
            Mode = MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: CudaPayload,
{
    fn admit_operands(
        engine: &mut dyn Engine,
        tensors: &[&TensorMap<R, D, Self>],
        contracts: bool,
    ) -> Result<(), Error> {
        cuda_operand_admission(engine, tensors, contracts)
    }
}

impl<R, D, S> network_operand_sealed::Sealed for TensorMap<R, D, S>
where
    R: TypedSectorAdmission + Send + Sync,
    // The metadata preflight and plan lookup read leg dimensions through the
    // Host policy, which exists whenever a storage's policy does.
    R::Mode: NetworkModeDispatch<R, D> + NetworkModeDispatch<R, D, S>,
    D: TensorScalar + Send + Sync + 'static,
    S: NetworkOperandStorage<R, D>,
{
}

impl<R, D, S> NetworkOperand for TensorMap<R, D, S>
where
    R: TypedSectorAdmission + Send + Sync,
    // The metadata preflight and plan lookup read leg dimensions through the
    // Host policy, which exists whenever a storage's policy does.
    R::Mode: NetworkModeDispatch<R, D> + NetworkModeDispatch<R, D, S>,
    D: TensorScalar + Send + Sync + 'static,
    S: NetworkOperandStorage<R, D>,
{
    type Error = FacadeError<R>;

    fn contract_network(
        engine: &mut dyn Engine,
        network: &Network,
        tensors: &[&Self],
    ) -> Result<Self, Self::Error> {
        network.validate_operand_count(tensors.len())?;
        S::admit_operands(engine, tensors, tensors.len() > 1)?;
        network.contract_preflight(tensors)?;
        crate::plancache::get_or_plan_network(network, tensors, &runtime_optimizer(tensors))?
            .execute(engine, tensors)
    }
}
