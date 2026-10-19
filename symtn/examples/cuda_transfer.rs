//! Upload a Host tensor to CUDA device 0 and download it without changing values.
//! Requires the cuda feature and a working CUDA device.

use std::sync::Arc;
use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{Error, GradedSpace, Runtime, TensorMap};

fn main() -> Result<(), Error> {
    let rt = Runtime::builder().cuda(0).build()?;
    let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
    let host = TensorMap::<_, f64>::rand_with_seed(&rt, [&v], [&v], 7)?;
    let device = host.to_cuda()?;
    let restored = device.to_host()?;
    assert_eq!(host.dense_data()?, restored.dense_data()?);
    println!("cuda_transfer: assertions passed");
    Ok(())
}
