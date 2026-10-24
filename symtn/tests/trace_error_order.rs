//! One error per trace misuse, detected in TensorKit `trace_permute!`'s
//! order (#1872): braiding, then the axes, then the destination space (for a
//! trace into a destination), then the pair duality. The same input gives the
//! same error eagerly and into a destination, on Host and CUDA, in the
//! multiplicity-free and checked Generic modes, for `f64` and `Complex64`,
//! on legs with several sectors and degeneracies above one.
//!
//! The expected classes follow from TensorKit's checks, not from symtn: a
//! malformed pair list is an index error, `space(tsrc, q₁) ==
//! dual(space(tsrc, q₂))` is the pair duality, and `space(tdst) ==
//! select(space(tsrc), p)` comes before it. The device variants run with
//! `--features cuda -- --ignored`.

#[path = "support/report.rs"]
mod report;

use std::sync::Arc;

use report::cause;
use symtn::sector::{SU2FusionRule, SU2Irrep};
use symtn::typed::{Complex64, GradedSpace, OperationError, Runtime, TensorMap};

/// The misuse class an error reports, read from its typed operation cause
/// (both fusion modes' facade errors carry the same one).
fn class(error: impl std::error::Error + 'static) -> &'static str {
    match cause::<OperationError>(&error) {
        Some(OperationError::InvalidAxisSet {
            tensor: "trace pairs",
            ..
        }) => "pair list",
        Some(OperationError::StructureMismatch {
            tensor: "trace axes",
        }) => "non-dual pair",
        Some(OperationError::SpaceMismatch { .. }) => "destination space",
        _ => panic!("unclassified trace error: {error:?}"),
    }
}

fn su2_legs() -> (GradedSpace<SU2FusionRule>, GradedSpace<SU2FusionRule>) {
    let j = |twice: usize| SU2Irrep::from_twice_spin(twice);
    let rule = Arc::new(SU2FusionRule);
    let v = GradedSpace::try_new(Arc::clone(&rule), [(j(0), 2), (j(1), 1), (j(2), 2)]).unwrap();
    let w = GradedSpace::try_new(rule, [(j(0), 2), (j(3), 1)]).unwrap();
    (v, w)
}

/// Every multiplicity-free misuse, eager and into a destination. `$lift`
/// puts a Host tensor at the placement under test.
macro_rules! multiplicity_free_matrix {
    ($rt:expr, $d:ty, $lift:expr) => {{
        let rt = $rt;
        let lift = $lift;
        let (v, w) = su2_legs();
        let vd = v.try_dual().unwrap();
        let t = lift(&TensorMap::<_, $d>::rand_with_seed(rt, [&v, &vd, &v], [&v, &vd], 5).unwrap());
        let mismatched = lift(&TensorMap::<_, $d>::rand_with_seed(rt, [&v], [&w], 7).unwrap());
        // The result space of tracing (0, 2) on `t`: codomain [vd], domain [v, vd].
        let non_dual_target = || lift(&TensorMap::<_, $d>::zeros(rt, [&vd], [&v, &vd]).unwrap());
        // Every leg dualized: never the result space.
        let wrong_target = || lift(&TensorMap::<_, $d>::zeros(rt, [&v], [&vd, &v]).unwrap());
        let one = <$d>::from(1.0);
        let half = <$d>::from(0.5);
        assert!(t.trace_pairs(&[(0, 3), (1, 4)]).is_ok());

        for (pairs, expected) in [
            (vec![(0, 9)], "pair list"),
            (vec![(0, 0)], "pair list"),
            (vec![(0, 2)], "non-dual pair"),
        ] {
            assert_eq!(
                class(t.trace_pairs(&pairs).unwrap_err()),
                expected,
                "{pairs:?}"
            );
        }
        assert_eq!(
            class(mismatched.trace_pairs(&[(0, 1)]).unwrap_err()),
            "non-dual pair"
        );

        // Into a destination: a pair-list error needs no destination.
        for pairs in [vec![(0, 9)], vec![(0, 0)]] {
            let mut target = non_dual_target();
            assert_eq!(
                class(
                    t.trace_pairs_into(&pairs, &mut target, one, half)
                        .unwrap_err()
                ),
                "pair list"
            );
        }
        // The right destination space: the duality is the error.
        let mut target = non_dual_target();
        assert_eq!(
            class(
                t.trace_pairs_into(&[(0, 2)], &mut target, one, half)
                    .unwrap_err()
            ),
            "non-dual pair"
        );
        // A wrong destination space comes first, as `space(tdst)` does in
        // TensorKit.
        let mut target = wrong_target();
        assert_eq!(
            class(
                t.trace_pairs_into(&[(0, 2)], &mut target, one, half)
                    .unwrap_err()
            ),
            "destination space"
        );
        // A scalar destination, shared on purpose (`lift` clones on Host): the
        // duality error must come before the destination's storage checks
        // (`DestinationShared`), which TensorKit does not have.
        let scalar = TensorMap::<_, $d>::rand_with_seed(rt, [&v], [&v], 9)
            .unwrap()
            .trace_pairs(&[(0, 1)])
            .unwrap();
        let mut target = lift(&scalar);
        assert_eq!(
            class(
                mismatched
                    .trace_pairs_into(&[(0, 1)], &mut target, one, half)
                    .unwrap_err()
            ),
            "non-dual pair"
        );
    }};
}

#[test]
fn multiplicity_free_host_trace_reports_one_error_per_misuse() {
    let rt = Runtime::builder().dense_threads(1).build().unwrap();
    multiplicity_free_matrix!(&rt, f64, |x: &TensorMap<_, f64>| x.clone());
    multiplicity_free_matrix!(&rt, Complex64, |x: &TensorMap<_, Complex64>| x.clone());
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn multiplicity_free_device_trace_reports_the_host_error_per_misuse() {
    let rt = Runtime::builder().cuda(0).build().unwrap();
    multiplicity_free_matrix!(&rt, f64, |x: &TensorMap<_, f64>| x.to_cuda().unwrap());
    multiplicity_free_matrix!(&rt, Complex64, |x: &TensorMap<_, Complex64>| x
        .to_cuda()
        .unwrap());
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_generic_trace_reports_the_multiplicity_free_error_per_misuse() {
    use symtn::sector::LieGroupFusionRule;

    let rt = Runtime::builder().dense_threads(1).build().unwrap();
    let rule =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let v = GradedSpace::try_new(
        Arc::clone(&rule),
        [(vec![0i64, 0], 2), (vec![1, 0], 1), (vec![1, 1], 2)],
    )
    .unwrap();
    let w = GradedSpace::try_new(Arc::clone(&rule), [(vec![0i64, 0], 2), (vec![2, 1], 1)]).unwrap();
    let vd = v.try_dual().unwrap();
    macro_rules! generic {
        ($d:ty) => {{
            let t = TensorMap::<_, $d>::rand_with_seed(&rt, [&v, &vd, &v], [&v, &vd], 5).unwrap();
            let mismatched = TensorMap::<_, $d>::rand_with_seed(&rt, [&v], [&w], 7).unwrap();
            for (pairs, expected) in [
                (vec![(0, 9)], "pair list"),
                (vec![(0, 0)], "pair list"),
                (vec![(0, 2)], "non-dual pair"),
            ] {
                assert_eq!(
                    class(t.trace_pairs(&pairs).unwrap_err()),
                    expected,
                    "{pairs:?}"
                );
            }
            assert_eq!(
                class(mismatched.trace_pairs(&[(0, 1)]).unwrap_err()),
                "non-dual pair"
            );
            assert!(t.trace_pairs(&[(0, 3), (1, 4)]).is_ok());
        }};
    }
    generic!(f64);
    generic!(Complex64);
}
