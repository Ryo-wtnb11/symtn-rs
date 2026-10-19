#![cfg(feature = "cgc-gen")]
//! A checked Generic trace pairs only mutually dual legs (#1856), as
//! TensorKit `trace_permute!` requires `space(tsrc, q₁) == dual(space(tsrc, q₂))`.
//! The fixtures share their first sector, which the former first-sector
//! comparison admitted: the trace then summed only the shared sectors.

#[path = "support/report.rs"]
mod report;

use std::sync::Arc;

use report::cause;
use symtn::sector::SUNFusionRule;
use symtn::typed::{GradedSpace, OperationError, Runtime, TensorMap};

#[test]
fn checked_generic_trace_rejects_legs_that_are_not_mutually_dual() {
    let rt = Runtime::builder().dense_threads(1).build().unwrap();
    let rule = Arc::new(SUNFusionRule::new(3).unwrap());
    let v = GradedSpace::try_new(Arc::clone(&rule), [(vec![0i64, 0], 1), (vec![1, 0], 2)]).unwrap();
    let w = GradedSpace::try_new(Arc::clone(&rule), [(vec![0i64, 0], 1), (vec![1, 1], 1)]).unwrap();
    let rejects = |result: Result<TensorMap<SUNFusionRule, f64>, _>, what: &str| {
        let error = result.expect_err(what);
        assert!(
            matches!(
                cause::<OperationError>(&error),
                Some(OperationError::StructureMismatch {
                    tensor: "trace axes"
                })
            ),
            "{what}: {error:?}"
        );
    };

    // Different sector sets with the same first sector.
    let mismatched = TensorMap::<_, f64>::rand_with_seed(&rt, [&v], [&w], 1).unwrap();
    rejects(mismatched.trace_pairs(&[(0, 1)]), "V <- W");
    // Two codomain copies of one leg: equal, not dual.
    let doubled = TensorMap::<_, f64>::rand_with_seed(&rt, [&v, &v], [&v, &v], 2).unwrap();
    rejects(doubled.trace_pairs(&[(0, 1)]), "codomain V with codomain V");

    // Control: the dual pair is traced.
    let square = TensorMap::<_, f64>::rand_with_seed(&rt, [&v], [&v], 3).unwrap();
    assert!(square.trace_pairs(&[(0, 1)]).is_ok());
}
