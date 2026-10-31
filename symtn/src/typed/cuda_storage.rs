use super::storage::sealed;
use super::*;

impl<D: CudaPayload> sealed::Sealed for CudaStorage<D> {
    fn admit_shortcut_engine(engine: &mut dyn Engine, storage: &Self) -> Result<(), Error> {
        let lease = crate::runtime::cuda_call(engine)?;
        ensure_cuda_owns(&lease, "shortcut", &[&storage.0])
    }
}

impl<R, D> TypedStorage<R, D> for CudaStorage<D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: CudaPayload,
{
    const NOUN: &'static str = "CUDA";

    fn as_host(_: ExecToken, _: &TensorMap<R, D, Self>) -> Option<&TensorMap<R, D>> {
        None
    }

    fn as_host_mut(_: ExecToken, _: &mut TensorMap<R, D, Self>) -> Option<&mut TensorMap<R, D>> {
        None
    }

    fn from_host(_: ExecToken, _: TensorMap<R, D>) -> Result<TensorMap<R, D, Self>, Error> {
        Err(Error::UnsupportedOnDevice(
            "compact diagonal storage is host-only".to_string(),
        ))
    }
}
