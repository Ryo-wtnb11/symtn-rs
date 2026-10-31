use super::*;

/// The device replay of the one tree-transform body
/// ([`TensorMap::permute`], [`TensorMap::permute_into`] and their siblings).
///
/// The staging's context lease is dropped before the device lease is taken,
/// and nothing reached under the device lease leases again
/// (`std::sync::Mutex` is not re-entrant); the placement check is the first
/// thing under it, before any device write.
///
/// Checked-Generic, Generic and complex-coefficient providers are excluded
/// by this impl's bound, so they are a compile-time boundary:
///
/// ```compile_fail
/// use symtn::sector::{CheckedGenericAdmissionMode, CheckedGenericFusion, CheckedGenericRigidSymbols, TypedSectorAdmission};
/// use symtn::typed::{CudaStorage, TensorMap};
///
/// fn no_checked_generic_cuda_permute<R>(engine: &mut dyn symtn::typed::Engine, tensor: &TensorMap<R, f64, CudaStorage>)
/// where
///     R: TypedSectorAdmission<Mode = CheckedGenericAdmissionMode>
///         + CheckedGenericFusion
///         + CheckedGenericRigidSymbols<Scalar = f64>,
/// {
///     let _ = tensor.permute(engine, &[1], &[0]);
/// }
/// ```
///
/// ```compile_fail
/// use symtn::sector::TableRule;
/// use symtn::typed::{CudaStorage, TensorMap};
///
/// fn no_generic_cuda_permute(engine: &mut dyn symtn::typed::Engine, tensor: &TensorMap<TableRule, f64, CudaStorage>) {
///     let _ = tensor.permute(engine, &[1], &[0]);
/// }
/// ```
///
/// The multiplicity-free device methods compile, for either device payload:
///
/// ```
/// use symtn::sector::U1FusionRule;
/// use symtn::typed::{CudaPayload, CudaStorage, TensorMap};
///
/// fn device_permute<D: CudaPayload>(engine: &mut dyn symtn::typed::Engine, tensor: &TensorMap<U1FusionRule, D, CudaStorage<D>>) {
///     let _ = tensor.permute(engine, &[1], &[0]);
///     let _ = tensor.braid(engine, &[1], &[0], &[0, 1]);
///     let _ = tensor.transpose(engine, &[1], &[0]);
///     let _ = tensor.repartition(engine, 0);
/// }
/// ```
impl<R, D> TransformExec<R, D> for CudaStorage<D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: CudaPayload,
{
    fn replay_owned<C: Copy>(
        engine: &mut dyn Engine,
        _: ExecToken,
        replay: TreeReplay<'_, D, C>,
        t: &TensorMap<R, D, Self>,
        alpha: D,
    ) -> Result<Self, TransformExecError>
    where
        D: RecouplingCoefficientAction<C>,
    {
        let source = cuda_transform_source(t.owned_body().map(|body| body.data.as_ref()))?;
        let required_len = replay.destination.required_len().map_err(Error::from)?;
        let TreeReplay {
            lease,
            structure,
            destination,
            source: source_structure,
            ..
        } = replay;
        drop(lease);
        let mut lease = crate::runtime::cuda_call(engine)?;
        let (cuda, executor) = lease.split();
        ensure_cuda_owns(cuda, "transform", &[&source.0])?;
        // ponytail: #740 — the device seam still initializes an output by
        // uploading zeros; replace only with a measured native allocation.
        //
        // Why not report an inexpressible destination layout before spending
        // this upload: a returning transform cannot produce one. Its
        // destination is the staged one, i.e. the canonical final-homspace
        // layout, never derived from the source's strides, and block strides
        // are `usize`, so both arms of the executor's layout rejection are
        // unreachable here; the buffer is this kernel's own and is dropped on
        // any error, so no caller-visible state is touched either way. An
        // existing destination is the caller's, so `replay_into` has no such
        // argument and the executor validates before it writes.
        let mut output = CudaStorage::upload_owned(cuda, vec![D::from_real(0.0); required_len])
            .map_err(Error::from)?;
        executor
            .replay(
                cuda,
                structure,
                destination,
                source_structure,
                &mut output,
                source,
                alpha,
                CudaTreeTransformDestination::Overwrite,
            )
            .map_err(TransformExecError::Replay)?;
        drop(lease);
        Ok(output)
    }

    /// The executor decides Stage A, the prepared-structure layout
    /// expressibility and Stage C before its first submission, so a rejected
    /// replay leaves the caller's destination byte-identical.
    fn replay_into<C: Copy>(
        engine: &mut dyn Engine,
        _: ExecToken,
        replay: TreeReplay<'_, D, C>,
        source: TransformIntoSource<'_, D, Self, C>,
        destination: &mut Self,
        alpha: D,
        beta: D,
    ) -> Result<(), TransformExecError>
    where
        D: RecouplingCoefficientAction<C>,
    {
        let TransformIntoSource::Stored(source) = source else {
            return Err(internal_layout_error("CUDA tensors hold no compact data").into());
        };
        let TreeReplay {
            lease,
            structure,
            destination: destination_structure,
            source: source_structure,
            ..
        } = replay;
        drop(lease);
        let mut lease = crate::runtime::cuda_call(engine)?;
        let (cuda, executor) = lease.split();
        ensure_cuda_owns(cuda, "transform_into", &[&source.0, &destination.0])?;
        executor
            .replay(
                cuda,
                structure,
                destination_structure,
                source_structure,
                destination,
                source,
                alpha,
                CudaTreeTransformDestination::Axpby(beta),
            )
            .map_err(TransformExecError::Replay)
    }
}

/// The dense device payload of an owned transform input. Device tensors hold
/// no compact data — `to_cuda` densifies and every device kernel outputs
/// dense — and a lazy adjoint is lowered onto its parent before staging.
fn cuda_transform_source<D: CudaPayload>(
    data: Option<&TypedData<D, CudaStorage<D>>>,
) -> Result<&CudaStorage<D>, Error> {
    match data {
        Some(TypedData::Dense(storage)) => Ok(storage),
        _ => Err(internal_layout_error("owned dense CUDA transform input")),
    }
}

impl<R, D> TwistExec<R, D> for CudaStorage<D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: CudaPayload,
{
    fn scale_blocks<C: TwistCoefficient<D>>(
        engine: &mut dyn Engine,
        _: ExecToken,
        t: &TensorMap<R, D, Self>,
        factor_of: &dyn Fn(&BlockKey) -> C,
        operation: &'static str,
    ) -> Result<TensorMap<R, D, Self>, Error> {
        let src = t.direct_cuda_storage(operation)?;
        let space = t.logical_space().clone();
        let required_len = space.space().required_len()?;
        let blocks = cuda_twist_regions::<D, C>(space.space(), factor_of, operation)?;

        let mut lease = crate::runtime::cuda_call(engine)?;
        let cuda = &mut *lease;
        ensure_cuda_owns(cuda, operation, &[&src.0])?;
        TensorMap::<R, D, Self>::validate_cuda_owned_len(required_len, src.len())?;
        // ponytail: #740 — the device seam still initializes an output by
        // uploading zeros; replace only with a measured native allocation.
        let mut dst = CudaStorage::upload_owned(cuda, vec![D::from_real(0.0); required_len])?;
        for (region, factor) in &blocks {
            // A ribbon twist is never zero, so the factor rides the descriptor
            // scale: no coefficient buffer, and CUDA still has to read the
            // source rather than skipping it as it may for a zero alpha. That
            // keeps NaN and real infinities propagating as on Host; an infinite
            // complex entry is the disclosed #1301 difference, because Host
            // bit-copies a factor-1 block where this always multiplies.
            symtn_dense::cuda_region_axpby::<D>(
                cuda,
                &src.0,
                region,
                false,
                *factor,
                symtn_dense::CudaRegionCoefficient::One,
                symtn_dense::CudaRegionBeta::Overwrite,
                &mut dst.0,
                region,
            )
            .map_err(|err| Error::from(symtn_tensors::OperationError::Dense(err)))?;
        }
        drop(lease);
        Ok(TensorMap {
            runtime: t.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(space, dst)),
        })
    }
}

/// The device lowering of a twist, on the Host before the lease: one region
/// per block with that block's factor as the descriptor scale. A block whose
/// factor is 1 still has to be moved — the output starts as zeros, where the
/// Host started from a copy of the source. No provider is queried here.
pub(super) fn cuda_twist_regions<D: CudaPayload, C: TwistCoefficient<D>>(
    space: &DynamicFusionMapSpace,
    factor_of: &dyn Fn(&BlockKey) -> C,
    operation: &'static str,
) -> Result<Vec<(symtn_dense::CudaRegion, D)>, Error> {
    let structure = space.structure();
    (0..structure.block_count())
        .map(|index| {
            let block = structure.block(index)?;
            let region = symtn_dense::CudaRegion::new(
                block.shape().to_vec(),
                block.strides().to_vec(),
                block.offset(),
            )
            .and_then(|region| region.validate_as_destination(operation).map(|()| region))
            .map_err(|err| Error::from(symtn_tensors::OperationError::Dense(err)))?;
            Ok((region, factor_of(block.key()).as_data()))
        })
        .collect()
}
