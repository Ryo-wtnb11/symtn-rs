//! Expand an SU(2) identity into the physical basis and project it back.

use symtn::sector::{GroupId, LieGroupFusionRule, MultiplicityFreeAdmissionMode};

use std::sync::Arc;
use symtn::typed::{GradedSpace, Runtime, TensorMap};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rt = Runtime::builder().build()?;
    let rule = Arc::new(LieGroupFusionRule::<MultiplicityFreeAdmissionMode>::new(
        GroupId::su(2)?,
    )?);
    let half = rule.category().irrep(&[1])?;
    let v = GradedSpace::try_new(rule, [(half, 1)])?;
    let id = TensorMap::<_, f64>::isomorphism(engine, &rt, [&v], [&v])?;
    let dense = id.to_physical_dense(engine)?;
    assert_eq!(dense.shape, [2, 2]);
    for (actual, expected) in dense.data.iter().zip([1.0, 0.0, 0.0, 1.0]) {
        assert!((actual - expected).abs() < 1e-12);
    }
    let restored = id.project_physical_dense(engine, &dense)?;
    assert!(restored.axpby(engine, 1.0, &id, -1.0)?.norm(engine, 2.0)? < 1e-12);
    println!("dense_roundtrip: assertions passed");
    Ok(())
}
