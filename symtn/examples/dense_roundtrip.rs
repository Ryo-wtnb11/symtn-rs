//! Expand an SU(2) identity into the physical basis and project it back.

use std::sync::Arc;
use symtn::sector::{SU2FusionRule, SU2Irrep};
use symtn::typed::{GradedSpace, Runtime, TensorMap};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rt = Runtime::builder().build()?;
    let v = GradedSpace::try_new(Arc::new(SU2FusionRule), [(SU2Irrep::from_twice_spin(1), 1)])?;
    let id = TensorMap::<_, f64>::isomorphism(&rt, [&v], [&v])?;
    let dense = id.to_physical_dense()?;
    assert_eq!(dense.shape, [2, 2]);
    for (actual, expected) in dense.data.iter().zip([1.0, 0.0, 0.0, 1.0]) {
        assert!((actual - expected).abs() < 1e-12);
    }
    let restored = id.project_physical_dense(&dense)?;
    assert!(restored.axpby(1.0, &id, -1.0)?.norm(2.0)? < 1e-12);
    println!("dense_roundtrip: assertions passed");
    Ok(())
}
