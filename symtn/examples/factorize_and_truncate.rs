//! Reconstruct diag(3, 1) by SVD, then keep its largest singular value.

use std::sync::Arc;
use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{Error, GradedSpace, Runtime, TensorMap};
use symtn::typed::{Svd, Truncation};

fn main() -> Result<(), Error> {
    let rt = Runtime::builder().build()?;
    let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
    let a = TensorMap::<_, f64>::from_subblock_fn(&rt, [&v], [&v], |_, ij| {
        if ij[0] == ij[1] {
            [3.0, 1.0][ij[0]]
        } else {
            0.0
        }
    })?;
    let Svd { u, s, vh } = a.svd_compact(&[0], &[1])?;
    let full = u.compose(&s)?.compose(&vh)?;
    assert!(full.axpby(1.0, &a, -1.0)?.norm(2.0)? < 1e-12);
    let found = s.domain()[0].find_truncated(&s.diagview()?, &Truncation::rank(1))?;
    let u = u.restrict_leg(&[(1, &found.selection)])?;
    let s = s.restrict_leg(&[(0, &found.selection), (1, &found.selection)])?;
    let vh = vh.restrict_leg(&[(0, &found.selection)])?;
    let approximation = u.compose(&s)?.compose(&vh)?;
    let residual = a.axpby(1.0, &approximation, -1.0)?.norm(2.0)?;
    assert!((residual - 1.0).abs() < 1e-12);
    assert!((residual - found.error).abs() < 1e-12);
    println!("factorize_and_truncate: assertions passed");
    Ok(())
}
