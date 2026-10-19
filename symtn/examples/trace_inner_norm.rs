//! Trace, inner product, and norm of the two-dimensional identity.

use std::sync::Arc;
use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{Error, GradedSpace, Runtime, TensorMap};

fn main() -> Result<(), Error> {
    let rt = Runtime::builder().build()?;
    let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
    let id = TensorMap::<_, f64>::isomorphism(&rt, [&v], [&v])?;
    assert!((id.trace_pairs(&[(0, 1)])?.scalar()? - 2.0).abs() < 1e-12);
    assert!((id.inner(&id)? - 2.0).abs() < 1e-12);
    assert!((id.norm(2.0)? - 2.0_f64.sqrt()).abs() < 1e-12);
    println!("trace_inner_norm: assertions passed");
    Ok(())
}
