//! Diagonalize a Hermitian map and compute its exponential.
//! A general matrix sqrt API is unavailable; map_diagonal handles its spectrum.

use std::sync::Arc;
use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{Eigh, HermitianTol};
use symtn::typed::{Error, GradedSpace, Runtime, TensorMap};

fn main() -> Result<(), Error> {
    let rt = Runtime::builder().build()?;
    let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
    let a = TensorMap::<_, f64>::from_subblock_fn(&rt, [&v], [&v], |_, ij| {
        if ij[0] == ij[1] {
            [1.0, 4.0][ij[0]]
        } else {
            0.0
        }
    })?;
    let Eigh { d, v: eigenvectors } = a.eigh_full(&[0], &[1], HermitianTol::DEFAULT)?;
    let reconstructed = eigenvectors
        .compose(&d)?
        .compose(&eigenvectors.adjoint()?)?;
    assert!(reconstructed.axpby(1.0, &a, -1.0)?.norm(2.0)? < 1e-12);
    let root = eigenvectors
        .compose(&d.map_diagonal(f64::sqrt)?)?
        .compose(&eigenvectors.adjoint()?)?;
    assert!(root.compose(&root)?.axpby(1.0, &a, -1.0)?.norm(2.0)? < 1e-12);
    let exponential = a.exp(&[0], &[1])?;
    let expected = TensorMap::<_, f64>::from_subblock_fn(&rt, [&v], [&v], |_, ij| {
        if ij[0] == ij[1] {
            [1.0_f64.exp(), 4.0_f64.exp()][ij[0]]
        } else {
            0.0
        }
    })?;
    assert!(exponential.axpby(1.0, &expected, -1.0)?.norm(2.0)? < 1e-10);
    println!("eigh_and_matrix_functions: assertions passed");
    Ok(())
}
