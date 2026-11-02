//! Host eager `compose` against TensorKit `mul!` spelled per fusion-tree
//! block (`prepared::compose_oracle`), which reads the operands through the
//! public block API and never runs a compose plan.
//!
//! Over U(1), SU(2), fZ2xU(1) and the non-symmetric braiding probes, f64 and
//! c64, owned and lazy-adjoint operands compose with dyadic entries. A lazy
//! adjoint enters the oracle materialized (the exact block copy
//! `block(t^H, c) = block(t, c)^H`). Rejected compositions are matched by
//! their typed error.

use crate::common;

use crate::prepared;

use crate::braiding_probe;

use std::fmt::Debug;
use std::sync::Arc;

#[allow(unused_imports)]
use num_complex::{Complex32, Complex64};
use symtn::sector::{CheckedFusionAlgebra, MultiplicityFreeRigidSymbols, SectorCodec};
use symtn::typed::{CoreError, Error, GradedSpace, OperationError, Runtime, TensorMap};

use braiding_probe::{ProbeSector, RealBraidingProbe};
use common::Payload;
use prepared::{compose_oracle, fz2u1_legs, members, su2_legs, u1_legs};

fn assert_oracle<R, D>(
    what: &str,
    actual: &TensorMap<R, D>,
    a: &TensorMap<R, D>,
    b: &TensorMap<R, D>,
) where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    <R as SectorCodec>::Sector: Debug,
    D: Payload,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let actual = actual.materialize(engine).unwrap();
    let (expected, _) = compose_oracle(a, b, &actual);
    let found = actual.dense_data().unwrap();
    assert_eq!(found.len(), expected.len(), "{what}");
    assert!(
        expected.iter().any(|value| value.magnitude() > 0.0),
        "{what}"
    );
    for (index, (&value, &want)) in found.iter().zip(&expected).enumerate() {
        assert!(
            value.distance(want) <= 1e-12 * (1.0 + want.magnitude()),
            "{what} [{index}]: {value:?} vs {want:?}"
        );
    }
}

fn check<R, D>(label: &str, (v, w): (GradedSpace<R>, GradedSpace<R>), degeneracy_error: bool)
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    <R as SectorCodec>::Sector: Debug,
    D: Payload,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let what = |case: &str| format!("{label} {} {case}", D::NAME);
    // A: V ⊗ V ← W, B: W ← V, C: V ← V.
    let a = members::<R, D>(&runtime, &[&v, &v], &[&w], 1, 1).remove(0);
    let b = members::<R, D>(&runtime, &[&w], &[&v], 1, 2).remove(0);
    let c = members::<R, D>(&runtime, &[&v], &[&v], 1, 3).remove(0);
    let mut materialized =
        |t: &TensorMap<R, D>| t.adjoint(engine).unwrap().materialize(engine).unwrap();
    let (a_h, b_h, c_h) = (materialized(&a), materialized(&b), materialized(&c));

    let ab = a.compose(engine, &b).unwrap();
    assert_oracle(&what("A*B"), &ab, &a, &b);
    let a_adjoint = a.adjoint(engine).unwrap();
    for (case, product) in [
        (
            "B'*A' lazy",
            b.adjoint(engine)
                .unwrap()
                .compose(engine, &a_adjoint)
                .unwrap(),
        ),
        (
            "B'*A' view",
            b.adjoint(engine)
                .unwrap()
                .compose(engine, a.adjoint_view())
                .unwrap(),
        ),
        (
            "B'*A' mixed",
            b.adjoint(engine).unwrap().compose(engine, &a_h).unwrap(),
        ),
    ] {
        assert_oracle(&what(case), &product, &b_h, &a_h);
    }
    assert_oracle(
        &what("C*C'"),
        &c.compose(engine, c.adjoint_view()).unwrap(),
        &c,
        &c_h,
    );
    // `A*B` was checked above, so it can stand in for its own oracle value.
    assert_oracle(&what("(A*B)*C"), &ab.compose(engine, &c).unwrap(), &ab, &c);

    let other = Runtime::builder().build().unwrap();
    let foreign = members::<R, D>(&other, &[&w], &[&v], 1, 2).remove(0);
    assert!(
        matches!(a.compose(engine, &foreign), Err(Error::RuntimeMismatch)),
        "{}",
        what("runtime")
    );
    assert!(
        matches!(
            a.compose(engine, &a),
            Err(Error::Operation(error))
                if matches!(*error, OperationError::ContractAxisCountMismatch { lhs: 1, rhs: 2 })
        ),
        "{}",
        what("rank")
    );
    // `W` and `V` differ in total dimension (and, for the one-sector probes,
    // in the degeneracy of that sector); both orders are rejected. The total
    // dimension check reports the lhs domain first, the degeneracy check
    // the rhs codomain.
    let space_mismatch = |result: Result<TensorMap<R, D>, Error>, (expected, actual)| match result {
        Err(Error::Operation(error)) => match *error {
            OperationError::Core(CoreError::DimensionMismatch {
                expected: e,
                actual: a,
            }) => !degeneracy_error && (e, a) == (expected, actual),
            OperationError::Core(CoreError::LegDegeneracyMismatch {
                expected: e,
                actual: a,
                ..
            }) => degeneracy_error && (e, a) == (expected, actual),
            _ => false,
        },
        _ => false,
    };
    let (forward, backward) = if degeneracy_error {
        ((3, 2), (2, 3))
    } else {
        ((2, 3), (3, 2))
    };
    assert!(
        space_mismatch(a.compose(engine, &c), forward),
        "{}",
        what("space")
    );
    assert!(
        space_mismatch(b.compose(engine, a.adjoint_view()), backward),
        "{}",
        what("adjoint space")
    );
}

fn probe_legs<const ANYONIC: bool>() -> (
    GradedSpace<RealBraidingProbe<ANYONIC>>,
    GradedSpace<RealBraidingProbe<ANYONIC>>,
) {
    let rule = Arc::new(RealBraidingProbe::<ANYONIC>);
    (
        GradedSpace::try_new(Arc::clone(&rule), [(ProbeSector, 2)]).unwrap(),
        GradedSpace::try_new(rule, [(ProbeSector, 3)]).unwrap(),
    )
}

#[test]
fn host_eager_compose_matches_the_tree_block_mul_oracle() {
    check::<_, f64>("U1", u1_legs(), false);
    check::<_, Complex64>("U1", u1_legs(), false);
    check::<_, f64>("SU2", su2_legs(), false);
    check::<_, Complex64>("SU2", su2_legs(), false);
    check::<_, f64>("fZ2xU1", fz2u1_legs(), false);
    check::<_, Complex64>("fZ2xU1", fz2u1_legs(), false);
    check::<_, f64>("anyonic", probe_legs::<true>(), true);
    check::<_, f64>("unbraided", probe_legs::<false>(), true);
}
