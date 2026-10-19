//! TensorKit's `ishermitian`, `isantihermitian`, `isisometric`, `isunitary`,
//! `isposdef`, `project_hermitian` and `project_antihermitian` are short
//! chains of public operations in symtn (`common/predicate_chains.rs`). Each
//! predicate is checked on tensors constructed to have (or lack) the
//! property, and each projection against `(x ± x†) / 2` evaluated entrywise
//! from the fill.

#[path = "../../tests/support/numerics.rs"]
mod numerics;

include!("common/predicate_chains.rs");
include!("common/predicate_chain_coefficients.rs");

use num_complex::{Complex32, Complex64};
use numerics::Numeric;
use std::sync::Arc;
use symtn::sector::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use symtn::typed::HermitianTol;
use symtn::typed::{Eigh, LeftPolar, Qr};
use symtn::typed::{GradedSpace, Runtime, TensorMap};

/// Every coupled block of the projection `p` of the single-leg
/// endomorphism `x` (block entry `(i, j)` is `value([i, j])`) equals
/// `(x + sign · x†) / 2` entrywise: `(value(i, j) + sign · conj(value(j, i))) / 2`.
fn assert_projection<R, D>(
    what: &str,
    p: &TensorMap<R, D>,
    value: impl Fn(&[usize]) -> D,
    sign: f64,
    tol: f64,
) where
    R: symtn::sector::MultiplicityFreeRigidSymbols<Scalar = f64>
        + symtn::sector::CheckedFusionAlgebra
        + symtn::sector::SectorCodec,
    R::Sector: std::fmt::Debug,
    D: Numeric + symtn::typed::TensorScalar,
{
    let mut entries = 0;
    for (sector, block) in p.blocks().unwrap() {
        assert_eq!(block.rows(), block.cols(), "{what} {sector:?}");
        for i in 0..block.rows() {
            for j in 0..block.cols() {
                let want = (value(&[i, j]).wide() + sign * value(&[j, i]).wide().conj()) * 0.5;
                let found = block.get(i, j).unwrap().wide();
                assert!(
                    (found - want).norm() <= tol * (1.0 + want.norm()),
                    "{what} {sector:?} [{i}, {j}]: {found} vs {want}"
                );
                entries += 1;
            }
        }
    }
    assert_eq!(entries, p.dense_data().unwrap().len(), "{what}");
}

fn fill(ij: &[usize]) -> f64 {
    let key = ij
        .iter()
        .enumerate()
        .map(|(axis, &index)| (axis + 2) * (index + 1))
        .sum::<usize>();
    (key as f64 * 0.7 + 0.3).sin()
}

/// One multiplicity-free family: the truth table of every predicate chain on
/// tensors built to be (or not be) each property, and both projections
/// against their defining formula.
macro_rules! check_family {
    ($rt:expr, $leg:expr, $d:ty, $tol:expr, $value:expr) => {{
        let rt: &Runtime = $rt;
        let leg = $leg;
        let value = $value;
        let x: TensorMap<_, $d> =
            TensorMap::from_subblock_fn(rt, [&leg], [&leg], |_, ij: &[usize]| value(ij)).unwrap();
        let h = project_hermitian!(x).unwrap();
        let a = project_antihermitian!(x).unwrap();
        let identity = TensorMap::isomorphism(rt, [&leg], [&leg]).unwrap();
        let positive = x
            .adjoint()
            .unwrap()
            .compose(&x)
            .unwrap()
            .axpby(
                ChainCoefficient::real(1.0),
                &identity,
                ChainCoefficient::real(1.0),
            )
            .unwrap();
        let negative = positive.scale(ChainCoefficient::real(-1.0)).unwrap();
        // A polar factor, not a QR `q`: a 2x2 Householder `q` is a symmetric
        // reflector, which would also be Hermitian.
        let LeftPolar { w: unitary, .. } = x.left_polar(&[0], &[1]).unwrap();
        let tall: TensorMap<_, $d> =
            TensorMap::from_subblock_fn(rt, [&leg, &leg], [&leg], |_, ij: &[usize]| value(ij))
                .unwrap();
        let Qr { q: isometry, .. } = tall.qr_compact(&[0, 1], &[2]).unwrap();
        let Eigh {
            d: compact_positive,
            ..
        } = positive
            .eigh_full(&[0], &[1], HermitianTol::DEFAULT)
            .unwrap();
        let Eigh {
            d: compact_negative,
            ..
        } = negative
            .eigh_full(&[0], &[1], HermitianTol::DEFAULT)
            .unwrap();

        let tol: f64 = $tol;
        let dense = [
            ("x", &x, [false, false, false, false, false]),
            ("hermitian", &h, [true, false, false, false, false]),
            ("antihermitian", &a, [false, true, false, false, false]),
            ("positive", &positive, [true, false, false, false, true]),
            ("negative", &negative, [true, false, false, false, false]),
            ("unitary", &unitary, [false, false, true, true, false]),
            ("isometry", &isometry, [false, false, true, false, false]),
        ];
        for (name, t, expected) in dense {
            let chain = [
                is_hermitian!(t, tol),
                is_antihermitian!(t, tol),
                is_isometric!(t, tol),
                is_unitary!(t, tol),
                is_posdef!(t, tol),
            ];
            assert_eq!(chain, expected, "{name}");
        }
        let re = |value: $d| value.wide().re;
        for (name, t, expected) in [
            ("compact positive", &compact_positive, true),
            ("compact negative", &compact_negative, false),
        ] {
            assert_eq!(is_posdef_compact!(t, tol, re), expected, "{name}");
        }
        assert!(project_hermitian!(isometry).is_err());
        assert!(project_antihermitian!(isometry).is_err());

        assert_projection("hermitian", &h, &value, 1.0, tol);
        assert_projection("antihermitian", &a, &value, -1.0, tol);
    }};
}

fn u1_leg() -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 3)],
    )
    .unwrap()
}

fn su2_leg() -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 3),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap()
}

#[test]
fn u1_chains_decide_constructed_cases_and_project_by_formula() {
    let rt = Runtime::builder().dense_threads(1).build().unwrap();
    check_family!(&rt, u1_leg(), f64, 1e-10, fill);
    check_family!(&rt, u1_leg(), Complex64, 1e-10, |ij: &[usize]| {
        Complex64::new(fill(ij), fill(&[ij[0] + 1, ij[1]]))
    });
}

#[test]
fn su2_chains_decide_constructed_cases_and_project_by_formula() {
    let rt = Runtime::builder().dense_threads(1).build().unwrap();
    check_family!(&rt, su2_leg(), f64, 1e-10, fill);
    check_family!(&rt, su2_leg(), Complex32, 1e-4, |ij: &[usize]| {
        Complex32::new(fill(ij) as f32, fill(&[ij[0] + 1, ij[1]]) as f32)
    });
}

/// Checked-Generic providers never had the removed methods; the chains are
/// the only spelling there. SU(3) with an adjoint leg carries outer
/// multiplicity. The identity is `isomorphism(V, V)`, and checked-Generic
/// `compose` rejects a lazy adjoint operand, so `t†` enters the Gram map
/// materialized.
#[cfg(feature = "cgc-gen")]
#[test]
fn checked_generic_su3_chains_decide_true_and_false_cases() {
    use std::cell::Cell;

    use symtn::sector::SUNFusionRule;

    let rt = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 2)]).unwrap();
    let materialized_adjoint = |t: &TensorMap<SUNFusionRule, f64>| {
        let logical = t
            .adjoint()
            .unwrap()
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap()
            .to_vec();
        let position = Cell::new(0usize);
        TensorMap::from_subblock_fn(&rt, t.domain().iter(), t.codomain().iter(), |_, _| {
            let index = position.get();
            position.set(index + 1);
            logical[index]
        })
        .unwrap()
    };
    let identity_like = |t: &TensorMap<SUNFusionRule, f64>| {
        TensorMap::isomorphism(&rt, t.codomain().iter(), t.domain().iter()).unwrap()
    };
    let is_identity = |gram: TensorMap<SUNFusionRule, f64>, tol: f64| {
        let identity = identity_like(&gram);
        gram.axpby(1.0, &identity, -1.0).unwrap().norm(2.0).unwrap()
            <= tol * gram.norm(2.0).unwrap().max(1.0)
    };
    // `t† ∘ t == id`, and for unitarity also `t ∘ t† == id`.
    let is_isometric = |t: &TensorMap<SUNFusionRule, f64>, tol: f64| {
        is_identity(materialized_adjoint(t).compose(t).unwrap(), tol)
    };
    let is_unitary = |t: &TensorMap<SUNFusionRule, f64>, tol: f64| {
        is_isometric(t, tol) && is_identity(t.compose(&materialized_adjoint(t)).unwrap(), tol)
    };
    let x: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&rt, [&leg, &leg], [&leg, &leg], |_, ij: &[usize]| fill(ij))
            .unwrap();
    let h = project_hermitian!(x).unwrap();
    let a = project_antihermitian!(x).unwrap();
    let positive = h
        .compose(&h)
        .unwrap()
        .axpby(1.0, &identity_like(&x), 1.0)
        .unwrap();
    let negative = positive.scale(-1.0).unwrap();
    let Qr { q: unitary, .. } = x.qr_compact(&[0, 1], &[2, 3]).unwrap();
    let tall: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&rt, [&leg, &leg], [&leg], |_, ij: &[usize]| fill(ij)).unwrap();
    let Qr { q: isometry, .. } = tall.qr_compact(&[0, 1], &[2]).unwrap();
    assert!(project_hermitian!(isometry).is_err());

    let tol = 1e-10;
    for (name, t, expected) in [
        ("x", &x, [false, false, false, false, false]),
        ("hermitian", &h, [true, false, false, false, false]),
        ("antihermitian", &a, [false, true, false, false, false]),
        ("positive", &positive, [true, false, false, false, true]),
        ("negative", &negative, [true, false, false, false, false]),
        ("isometry", &isometry, [false, false, true, false, false]),
    ] {
        let chain = [
            is_hermitian!(t, tol),
            is_antihermitian!(t, tol),
            is_isometric(t, tol),
            is_unitary(t, tol),
            is_posdef!(t, tol),
        ];
        assert_eq!(chain, expected, "{name}");
    }
    // `q` of a square QR is unitary; whether it is also Hermitian depends on
    // the reflectors, so only unitarity is asserted.
    assert!(is_unitary(&unitary, tol));
}
