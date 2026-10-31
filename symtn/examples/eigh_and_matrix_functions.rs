//! Diagonalize a Hermitian map and compute its exponential.
//! A general matrix sqrt API is unavailable; map_diagonal handles its spectrum.

use std::sync::Arc;
use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{Eigh, HermitianTol};
use symtn::typed::{Error, GradedSpace, Runtime, TensorMap};

fn main() -> Result<(), Error> {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rt = Runtime::builder().build()?;
    let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
    let a = TensorMap::<_, f64>::from_subblock_fn(engine, &rt, [&v], [&v], |_, ij| {
        if ij[0] == ij[1] {
            [1.0, 4.0][ij[0]]
        } else {
            0.0
        }
    })?;
    let Eigh { d, v: eigenvectors } = a.eigh_full(engine, &[0], &[1], HermitianTol::DEFAULT)?;
    let eigenvectors_adjoint = &eigenvectors.adjoint(engine)?;
    let reconstructed = eigenvectors
        .compose(engine, &d)?
        .compose(engine, eigenvectors_adjoint)?;
    assert!(
        reconstructed
            .axpby(engine, 1.0, &a, -1.0)?
            .norm(engine, 2.0)?
            < 1e-12
    );
    let d_map_diagonal = &d.map_diagonal(engine, f64::sqrt)?;
    let eigenvectors_adjoint = &eigenvectors.adjoint(engine)?;
    let root = eigenvectors
        .compose(engine, d_map_diagonal)?
        .compose(engine, eigenvectors_adjoint)?;
    assert!(
        root.compose(engine, &root)?
            .axpby(engine, 1.0, &a, -1.0)?
            .norm(engine, 2.0)?
            < 1e-12
    );
    let exponential = a.exp(engine, &[0], &[1])?;
    let expected = TensorMap::<_, f64>::from_subblock_fn(engine, &rt, [&v], [&v], |_, ij| {
        if ij[0] == ij[1] {
            [1.0_f64.exp(), 4.0_f64.exp()][ij[0]]
        } else {
            0.0
        }
    })?;
    assert!(
        exponential
            .axpby(engine, 1.0, &expected, -1.0)?
            .norm(engine, 2.0)?
            < 1e-10
    );
    println!("eigh_and_matrix_functions: assertions passed");
    Ok(())
}
