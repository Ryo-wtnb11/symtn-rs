//! Construct U(1) and SU(2) spaces, dual legs, and reduced tensors.

use std::sync::Arc;
use symtn::sector::{SU2FusionRule, SU2Irrep};
use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{Error, GradedSpace, Runtime, TensorMap};

fn main() -> Result<(), Error> {
    let rt = Runtime::builder().build()?;
    let v = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )?;
    let dual = v.try_dual()?;
    assert!(dual.is_dual());
    assert_eq!(dual.degeneracy(&U1Irrep::new(-1))?, 1);
    assert_eq!(dual.try_dual()?, v);
    let zero = TensorMap::<_, f64>::zeros(&rt, [&v], [&v])?;
    assert!(zero.norm(2.0)? < 1e-12);
    let id = TensorMap::<_, f64>::from_subblock_fn(&rt, [&v], [&v], |_, ij| {
        if ij[0] == ij[1] {
            1.0
        } else {
            0.0
        }
    })?;
    assert_eq!(id.blocks()?.len(), 2);
    assert!((id.inner(&id)? - 3.0).abs() < 1e-12);
    let a = TensorMap::<_, f64>::rand_with_seed(&rt, [&v], [&v], 7)?;
    let b = TensorMap::<_, f64>::rand_with_seed(&rt, [&v], [&v], 7)?;
    assert_eq!(a.dense_data()?, b.dense_data()?);
    let spin = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 1),
            (SU2Irrep::from_twice_spin(1), 1),
        ],
    )?;
    let id = TensorMap::<_, f64>::isomorphism(&rt, [&spin], [&spin])?;
    assert_eq!(id.blocks()?.len(), 2);
    assert!((id.inner(&id)? - 3.0).abs() < 1e-12);
    println!("spaces_and_tensors: assertions passed");
    Ok(())
}
