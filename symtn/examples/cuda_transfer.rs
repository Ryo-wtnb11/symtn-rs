//! Upload a Host tensor to CUDA device 0 and download it without changing values.
//! Requires the cuda feature and a working CUDA device.

use std::sync::Arc;
use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{CpuEngine, CudaDeviceId, CudaEngine, GradedSpace, Runtime, TensorMap};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cpu = &mut CpuEngine::serial();
    let cuda = &mut CudaEngine::open(CudaDeviceId::from_ordinal(0))?;
    let rt = Runtime::builder().build()?;
    let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
    let host = TensorMap::<_, f64>::rand_with_seed(cpu, &rt, [&v], [&v], 7)?;
    let device = host.to_cuda(cuda)?;
    let restored = device.to_host(cuda)?;
    assert_eq!(host.dense_data()?, restored.dense_data()?);
    println!("cuda_transfer: assertions passed");
    Ok(())
}
