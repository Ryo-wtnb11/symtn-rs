//! Disjoint U(1) zero-map null-space completion against TensorKit 0.17.1
//! (Julia 1.11.6).

include!("common/predicate_chains.rs");
include!("common/predicate_chain_coefficients.rs");

use std::sync::Arc;

use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{GradedSpace, Runtime, TensorMap};

#[test]
fn direct_and_lazy_adjoint_disjoint_null_spaces_match_tensorkit() {
    let runtime = Runtime::builder().build().unwrap();
    let rule = Arc::new(U1FusionRule);
    let codomain = GradedSpace::try_new(Arc::clone(&rule), [(U1Irrep::new(0), 2)]).unwrap();
    let domain = GradedSpace::try_new(rule, [(U1Irrep::new(1), 3)]).unwrap();
    let source: TensorMap<_, f64> = TensorMap::zeros(&runtime, [&codomain], [&domain]).unwrap();

    assert!(source.dense_data().unwrap().is_empty());
    for (tensor, left_dim, right_dim) in [(source.clone(), 2, 3), (source.adjoint().unwrap(), 3, 2)]
    {
        let left = tensor.left_null(&[0], &[1]).unwrap();
        let right = tensor.right_null(&[0], &[1]).unwrap();

        assert_eq!(left.dense_data().unwrap().len(), left_dim * left_dim);
        assert_eq!(right.dense_data().unwrap().len(), right_dim * right_dim);
        assert!((left.norm(2.0).unwrap() - (left_dim as f64).sqrt()).abs() <= 1e-12);
        assert!((right.norm(2.0).unwrap() - (right_dim as f64).sqrt()).abs() <= 1e-12);
        assert!(is_isometric!(left, 1e-12));
        assert!(is_isometric!(right.adjoint().unwrap(), 1e-12));
        assert!(left
            .adjoint()
            .unwrap()
            .compose(&tensor)
            .unwrap()
            .dense_data()
            .unwrap()
            .is_empty());
        assert!(tensor
            .compose(&right.adjoint().unwrap())
            .unwrap()
            .dense_data()
            .unwrap()
            .is_empty());
    }
}
