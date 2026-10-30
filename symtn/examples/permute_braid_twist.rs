//! Exchange two odd fermionic legs and twist one leg: each gives a minus sign.

use std::sync::Arc;
use symtn::sector::{FermionParityFusionRule, Z2Irrep};
use symtn::typed::{Direction, Error, GradedSpace, Runtime, TensorMap};

fn main() -> Result<(), Error> {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rt = Runtime::builder().build()?;
    let v = GradedSpace::try_new(Arc::new(FermionParityFusionRule), [(Z2Irrep::ODD, 1)])?;
    let a = TensorMap::<_, f64>::from_subblock_fn(engine, &rt, [&v, &v], [], |_, _| 1.0)?;
    let permuted = a.permute(engine, &[1, 0], &[])?;
    let braided = a.braid(engine, &[1, 0], &[], &[0, 1])?;
    let twisted = a.twist(engine, &[0], Direction::Forward)?;
    for result in [permuted, braided, twisted] {
        assert!((result.dense_data()?[0] + 1.0).abs() < 1e-12);
    }
    println!("permute_braid_twist: assertions passed");
    Ok(())
}
