use super::*;

/// The device trace kernel: the terms compiled on the Host, replayed on the
/// device under one device lease, the placement checked under it first.
impl<R, D> TraceExec<R, D> for CudaStorage<D>
where
    R: symtn_core::TypedSectorAdmission<
            Sector = <R as symtn_core::SectorCodec>::Sector,
            Error = symtn_core::FusionAlgebraError,
            Mode = symtn_core::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: CudaPayload,
{
    /// The #740 output upload, then every term accumulates (`beta = 1`), so
    /// no `beta` pass is submitted.
    fn trace_owned(
        _: ExecToken,
        structure: &symtn_tensors::TensorTraceFusionStructure<f64>,
        _dst_nout: usize,
        compact: Option<D>,
        source: &TensorMap<R, D, Self>,
        alpha: D,
    ) -> Result<Self, TransformExecError> {
        let src = cuda_trace_source(source, compact)?;
        let required_len = structure
            .dst_structure()
            .required_len()
            .map_err(Error::from)?;
        let mut lease = source.runtime.lease_cuda()?;
        let (cuda, transforms) = lease.split();
        if src.placement() != Placement::Cuda(cuda.device()) {
            return Err(Error::PlacementMismatch.into());
        }
        // ponytail: #740 — the device seam initializes an output by uploading
        // zeros; replace only with a measured native allocation. The zeros are
        // also what the accumulating replay starts from.
        let mut output = CudaStorage::upload_owned(cuda, vec![D::from_real(0.0); required_len])
            .map_err(Error::from)?;
        symtn_tensors::tensortrace_fusion_structure_into_on_cuda(
            cuda,
            transforms,
            structure,
            structure.dst_structure(),
            &mut output,
            source.storage_body().space.space().structure(),
            src,
            alpha,
            D::from_real(1.0),
        )
        .map_err(TransformExecError::Replay)?;
        drop(lease);
        Ok(output)
    }

    /// The executor admits every region before its first submission, so a
    /// rejected replay leaves the caller's destination byte-identical.
    fn trace_into(
        _: ExecToken,
        structure: &symtn_tensors::TensorTraceFusionStructure<f64>,
        compact: Option<D>,
        source: &TensorMap<R, D, Self>,
        destination: &mut Self,
        alpha: D,
        beta: D,
    ) -> Result<(), TransformExecError> {
        let src = cuda_trace_source(source, compact)?;
        let mut lease = source.runtime.lease_cuda()?;
        let (cuda, transforms) = lease.split();
        let device = Placement::Cuda(cuda.device());
        if src.placement() != device || destination.placement() != device {
            return Err(Error::PlacementMismatch.into());
        }
        symtn_tensors::tensortrace_fusion_structure_into_on_cuda(
            cuda,
            transforms,
            structure,
            structure.dst_structure(),
            destination,
            source.storage_body().space.space().structure(),
            src,
            alpha,
            beta,
        )
        .map_err(TransformExecError::Replay)
    }
}

/// The dense device payload a trace reads. Device tensors hold no compact
/// data, so the compact arm never answers on the device.
fn cuda_trace_source<R, D: CudaPayload>(
    source: &TensorMap<R, D, CudaStorage<D>>,
    compact: Option<D>,
) -> Result<&CudaStorage<D>, Error> {
    match compact {
        None => cuda_contract_operand(source),
        Some(_) => Err(internal_layout_error("CUDA tensors hold no compact data")),
    }
}

/// The device lease a prepared stack replay holds from its output upload
/// through its replay. Nominally public only to name
/// [`PreparedExec::MembersLease`]; it is not reachable outside the crate.
#[doc(hidden)]
pub struct CudaMembersLease<'r>(crate::runtime::CudaLease<'r>);

/// The device prepared-batch kernels: the members' plan entries, lease,
/// upload and replay of the one prepared plan body.
impl<R, D> PreparedExec<R, D> for CudaStorage<D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: CudaPayload,
{
    type MembersLease<'r> = CudaMembersLease<'r>;

    fn reserve_members(
        _: ExecToken,
        plan: &ContractPlan<R, D, Self>,
        workspace: &mut ContractWorkspace<R, D, Self>,
    ) -> Result<(), Error> {
        let entries = plan.resolution.admit_cuda_members()?;
        workspace.reserved_plan_entries = plan
            .runtime
            .lease_cuda()?
            .reserve_plan_entries(entries)
            .map_err(symtn_operations::OperationError::Dense)?;
        Ok(())
    }

    fn lease_members(_: ExecToken, runtime: &Runtime) -> Result<CudaMembersLease<'_>, Error> {
        Ok(CudaMembersLease(runtime.lease_cuda()?))
    }

    fn zeroed_members(
        _: ExecToken,
        lease: &CudaMembersLease<'_>,
        total: usize,
        member_len: usize,
        members: usize,
    ) -> Result<Self, Error> {
        Ok(CudaStorage::upload_members(
            &lease.0,
            symtn_tensors::zeroed_payload(total),
            member_len,
            members,
        )?)
    }

    fn contract_members(
        _: ExecToken,
        plan: &ContractPlan<R, D, Self>,
        workspace: &mut ContractWorkspace<R, D, Self>,
        lease: &mut CudaMembersLease<'_>,
        dst: &mut Self,
        lhs: (&Arc<BlockStructure>, &Self),
        rhs: (&Arc<BlockStructure>, &Self),
        members: usize,
        init: symtn_tensors::ContractDestinationInit<D>,
    ) -> Result<(), Error> {
        symtn_tensors::execute_storage_contract_members_cuda(
            &mut lease.0,
            &plan.resolution,
            (plan.space.space().structure(), dst),
            lhs,
            rhs,
            &mut workspace.members_cuda,
            members,
            init,
        )?;
        Ok(())
    }

    fn retained_scratch_bytes(_: ExecToken, workspace: &ContractWorkspace<R, D, Self>) -> usize {
        workspace.members_cuda.retained_bytes()
    }

    fn retained_payload_bytes(_: ExecToken, _payload: &Self, len: usize) -> usize {
        len.saturating_mul(std::mem::size_of::<D>())
    }
}

/// The device contraction kernel: the route planned on the Host, replayed on
/// the device. Every transform structure, core plan, borrow decision and the
/// destination come from the one planner the Host uses, with the
/// fully-direct-core capabilities of the device executor.
impl<R, D> ContractExec<R, D> for CudaStorage<D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: CudaPayload,
{
    type Caps = symtn_tensors::DirectCoreExecutor;

    fn contract_owned<C: symtn_tensors::DenseBlockScalar>(
        _: ExecToken,
        replay: ContractReplay<'_, D, C>,
        lhs: &TensorMap<R, D, Self>,
        rhs: &TensorMap<R, D, Self>,
    ) -> Result<Self, TransformExecError>
    where
        D: RecouplingCoefficientAction<C>,
    {
        let (lhs, rhs) = (cuda_contract_operand(lhs)?, cuda_contract_operand(rhs)?);
        let ContractReplay {
            runtime,
            lease,
            resolution,
            destination,
            destination_len,
            lhs: lhs_structure,
            rhs: rhs_structure,
            ..
        } = replay;
        drop(lease);
        let mut lease = runtime.lease_cuda()?;
        let (cuda, transforms, scratch) = lease.split_contract();
        let device = Placement::Cuda(cuda.device());
        if lhs.placement() != device || rhs.placement() != device {
            return Err(Error::PlacementMismatch.into());
        }
        // ponytail: #740 — the device seam initializes an output by uploading
        // zeros; replace only with a measured native allocation.
        let mut dst = CudaStorage::upload_owned(cuda, vec![D::from_real(0.0); destination_len])
            .map_err(Error::from)?;
        symtn_tensors::execute_storage_contract_resolution_on_cuda(
            cuda,
            transforms,
            scratch,
            resolution,
            (destination, &mut dst),
            (lhs_structure, lhs),
            (rhs_structure, rhs),
            D::from_real(1.0),
            symtn_tensors::ContractDestinationInit::Zeroed,
        )
        .map_err(TransformExecError::Replay)?;
        drop(lease);
        Ok(dst)
    }

    /// The route executor admits every region before its first submission,
    /// so a rejected replay leaves the caller's destination byte-identical.
    fn contract_into<C: symtn_tensors::DenseBlockScalar>(
        _: ExecToken,
        replay: ContractReplay<'_, D, C>,
        lhs: &TensorMap<R, D, Self>,
        rhs: &TensorMap<R, D, Self>,
        destination: &mut Self,
        alpha: D,
        beta: D,
    ) -> Result<(), TransformExecError>
    where
        D: RecouplingCoefficientAction<C>,
    {
        let (lhs, rhs) = (cuda_contract_operand(lhs)?, cuda_contract_operand(rhs)?);
        let ContractReplay {
            runtime,
            lease,
            resolution,
            destination: destination_structure,
            lhs: lhs_structure,
            rhs: rhs_structure,
            ..
        } = replay;
        drop(lease);
        let mut lease = runtime.lease_cuda()?;
        let (cuda, transforms, scratch) = lease.split_contract();
        let device = Placement::Cuda(cuda.device());
        if lhs.placement() != device
            || rhs.placement() != device
            || destination.placement() != device
        {
            return Err(Error::PlacementMismatch.into());
        }
        symtn_tensors::execute_storage_contract_resolution_on_cuda(
            cuda,
            transforms,
            scratch,
            resolution,
            (destination_structure, destination),
            (lhs_structure, lhs),
            (rhs_structure, rhs),
            alpha,
            symtn_tensors::ContractDestinationInit::Axpby(beta),
        )
        .map_err(TransformExecError::Replay)
    }
}

/// The dense device payload a contraction operand stores: its own, or a
/// lazy adjoint's parent's. Device tensors hold no compact data — `to_cuda`
/// densifies and every device kernel outputs dense.
fn cuda_contract_operand<R, D: CudaPayload>(
    t: &TensorMap<R, D, CudaStorage<D>>,
) -> Result<&CudaStorage<D>, Error> {
    match t.storage_body().data.as_ref() {
        TypedData::Dense(storage) => Ok(storage),
        TypedData::Diagonal(_) => Err(internal_layout_error("CUDA tensors hold no compact data")),
    }
}
