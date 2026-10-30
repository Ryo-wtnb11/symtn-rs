//! `contract` with a [`ContractSpec`] (#1549) against its definition: the
//! contraction with TensorOperations' default split (every open leg of the
//! left operand in the codomain), followed by `permute(codomain, domain)`.
//! That is TensorKit's `tensorcontract_structure`,
//! `permute(compose(sA, sB), pAB)` (tensoroperations.jl L174-182).
//!
//! The oracle only runs the default split and the public `permute`, so it never
//! builds a destination split elsewhere than after the left operand's open
//! legs. Every output order and every split size is checked over U(1) with
//! non-self-dual legs, SU(2) and fZ2 x U(1) with dual legs, for owned, lazy
//! adjoint and compact diagonal operands, in real and complex payloads.

use crate::contract_cases;
use crate::numerics;

use std::sync::Arc;

use contract_cases::{fermion_u1, fill, su2, u1_non_self_dual, Payload};
use num_complex::Complex64;
use symtn::sector::{CheckedFusionAlgebra, MultiplicityFreeRigidSymbols, SectorCodec};
use symtn::sector::{TableRule, TableSector};
use symtn::typed::{ContractSpec, GradedSpace, Runtime, TensorMap};

fn permutations(n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![Vec::new()];
    }
    let mut all = Vec::new();
    for tail in permutations(n - 1) {
        for position in 0..=tail.len() {
            let mut order = tail.clone();
            order.insert(position, n - 1);
            all.push(order);
        }
    }
    all
}

/// Checks every order and split of the open legs of `lhs · rhs` over the
/// contracted `lhs_axes` / `rhs_axes`, through both `contract` and
/// `contract_into`, and returns how many specs it checked. A
/// destination of another split is rejected and left bit-identical.
fn check_every_spec<R, D>(
    what: &str,
    lhs: &TensorMap<R, D>,
    rhs: &TensorMap<R, D>,
    lhs_axes: &[usize],
    rhs_axes: &[usize],
) -> usize
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: Payload,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let lhs_open = lhs.rank() - lhs_axes.len();
    let open = lhs_open + rhs.rank() - rhs_axes.len();
    let identity: Vec<usize> = (0..open).collect();
    let default = ContractSpec {
        lhs: lhs_axes,
        rhs: rhs_axes,
        codomain: &identity[..lhs_open],
        domain: &identity[lhs_open..],
    };
    let contracted = lhs.contract(engine, rhs, &default).unwrap();
    // An entry is bilinear in the operands, so their lengths bound its terms.
    let mut len = |t: &TensorMap<R, D>| t.materialize(engine).unwrap().dense_data().unwrap().len();
    let terms = len(lhs) * len(rhs);
    let mut checked = 0;
    for order in permutations(open) {
        for split in 0..=open {
            let (codomain, domain) = order.split_at(split);
            let label = format!("{what} {} {codomain:?} <- {domain:?}", D::NAME);
            let expected = contracted.permute(engine, codomain, domain).unwrap();
            let spec = ContractSpec {
                codomain,
                domain,
                ..default
            };
            let got = lhs.contract(engine, rhs, &spec).unwrap();
            assert_eq!(got.codomain(), expected.codomain(), "{label}: codomain");
            assert_eq!(got.domain(), expected.domain(), "{label}: domain");
            numerics::assert_nonzero_slices_close(
                &label,
                got.dense_data().unwrap(),
                expected.dense_data().unwrap(),
                terms,
            );
            let mut destination = expected.scale(engine, D::entry(7.5, 0.0)).unwrap();
            lhs.contract_into(
                engine,
                rhs,
                &spec,
                &mut destination,
                D::entry(1.0, 0.0),
                D::entry(0.0, 0.0),
            )
            .unwrap();
            assert_eq!(destination.codomain(), expected.codomain(), "{label}");
            numerics::assert_nonzero_slices_close(
                &format!("{label} overwrite"),
                destination.dense_data().unwrap(),
                expected.dense_data().unwrap(),
                terms,
            );
            checked += 1;
        }
    }

    // The default-split destination under a spec that moves the split.
    let mut mismatched = contracted.scale(engine, D::entry(7.5, 0.0)).unwrap();
    let before = mismatched.dense_data().unwrap().to_vec();
    let moved = ContractSpec {
        codomain: &identity[..lhs_open + 1],
        domain: &identity[lhs_open + 1..],
        ..default
    };
    assert!(
        lhs.contract_into(
            engine,
            rhs,
            &moved,
            &mut mismatched,
            D::entry(1.0, 0.0),
            D::entry(0.0, 0.0)
        )
        .is_err(),
        "{what}: a destination of another split must be rejected"
    );
    assert_eq!(
        mismatched.dense_data().unwrap(),
        before.as_slice(),
        "{what}"
    );
    checked
}

/// `a: v ⊗ v* ← v` and `b: v ← v* ⊗ v`, contracted on a compose-shaped pair
/// and on a pair that bends both legs; then a lazy-adjoint left and a
/// lazy-adjoint right operand.
fn check_symmetry<R, D>(symmetry: &str, v: &GradedSpace<R>)
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: Payload,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let dual = v.try_dual().unwrap();
    let a =
        TensorMap::<R, D>::from_subblock_fn(engine, &runtime, [v, &dual], [v], fill(11)).unwrap();
    let b =
        TensorMap::<R, D>::from_subblock_fn(engine, &runtime, [v], [&dual, v], fill(12)).unwrap();
    let lazy = TensorMap::<R, D>::from_subblock_fn(engine, &runtime, [v], [v, &dual], fill(13))
        .unwrap()
        .adjoint(engine)
        .unwrap();
    let lazy_rhs = TensorMap::<R, D>::from_subblock_fn(engine, &runtime, [&dual, v], [v], fill(14))
        .unwrap()
        .adjoint(engine)
        .unwrap();
    let checked = check_every_spec(&format!("{symmetry} a[2]·b[0]"), &a, &b, &[2], &[0])
        + check_every_spec(&format!("{symmetry} a[0]·b[2]"), &a, &b, &[0], &[2])
        + check_every_spec(&format!("{symmetry} a'[2]·b[0]"), &lazy, &b, &[2], &[0])
        + check_every_spec(&format!("{symmetry} a[2]·b'[0]"), &a, &lazy_rhs, &[2], &[0]);
    // 4! orders times 5 split sizes, for each of the four pairings.
    assert_eq!(checked, 4 * 24 * 5);
}

#[test]
fn every_order_and_split_matches_contract_then_permute() {
    check_symmetry::<_, f64>("U(1)", &u1_non_self_dual());
    check_symmetry::<_, Complex64>("U(1)", &u1_non_self_dual());
    check_symmetry::<_, f64>("SU(2)", &su2());
    check_symmetry::<_, Complex64>("SU(2)", &su2());
    check_symmetry::<_, f64>("fZ2xU(1)", &fermion_u1());
    check_symmetry::<_, Complex64>("fZ2xU(1)", &fermion_u1());
}

/// The compact diagonal arms of `contract` (#584): `t · s` and `s · t` with
/// `s` a compact SVD spectrum, against a dense copy of `s` through the default
/// split and `permute`.
fn check_compact<R>(symmetry: &str, v: &GradedSpace<R>)
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let t = TensorMap::<R, f64>::from_subblock_fn(engine, &runtime, [v, v], [v], fill(21)).unwrap();
    let s = t.svd_compact(engine, &[0, 1], &[2]).unwrap().s;
    // A mixed compact + dense sum yields dense storage (as `forced_dense` in
    // `typed_facade.rs` pins); the dense route then runs no compact arm.
    let zeros = TensorMap::<R, f64>::zeros(engine, &runtime, &s.codomain(), &s.domain()).unwrap();
    let dense_s = s.axpby(engine, 1.0, &zeros, 1.0).unwrap();
    assert!(symtn::expert::diagonal_spectrum(engine, &dense_s)
        .unwrap()
        .is_none());
    for (what, lhs, rhs, dense_lhs, dense_rhs, lhs_axes, rhs_axes) in [
        ("t·s", &t, &s, &t, &dense_s, [2], [0]),
        ("s·t", &s, &t, &dense_s, &t, [1], [0]),
    ] {
        let lhs_open = lhs.rank() - 1;
        let identity = [0, 1, 2];
        let default = ContractSpec {
            lhs: &lhs_axes,
            rhs: &rhs_axes,
            codomain: &identity[..lhs_open],
            domain: &identity[lhs_open..],
        };
        let contracted = dense_lhs.contract(engine, dense_rhs, &default).unwrap();
        for order in permutations(3) {
            for split in 0..=3 {
                let (codomain, domain) = order.split_at(split);
                let label = format!("{symmetry} {what} {codomain:?} <- {domain:?}");
                let expected = contracted.permute(engine, codomain, domain).unwrap();
                let spec = ContractSpec {
                    codomain,
                    domain,
                    ..default
                };
                let got = lhs.contract(engine, rhs, &spec).unwrap();
                assert_eq!(got.codomain(), expected.codomain(), "{label}");
                assert_eq!(got.domain(), expected.domain(), "{label}");
                numerics::assert_nonzero_slices_close(
                    &label,
                    got.dense_data().unwrap(),
                    expected.dense_data().unwrap(),
                    64,
                );
            }
        }
    }
}

#[test]
fn compact_diagonal_arms_match_contract_then_permute() {
    check_compact("U(1)", &u1_non_self_dual());
    check_compact("SU(2)", &su2());
    check_compact("fZ2xU(1)", &fermion_u1());
}

/// Anyonic braiding stays outside `contract` whatever the spec: moving a leg
/// across the split needs a braid direction the spec does not carry.
#[test]
fn non_symmetric_braiding_is_unsupported_for_every_split() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(TableRule::fibonacci()),
        [(TableSector::new(1), 1), (TableSector::new(2), 1)],
    )
    .unwrap();
    // Fibonacci F-symbols are complex, so the payload is too.
    let a =
        TensorMap::<_, Complex64>::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, _| {
            Complex64::new(1.0, 0.5)
        })
        .unwrap();
    for (codomain, domain) in [
        (&[0][..], &[1][..]),
        (&[1][..], &[0][..]),
        (&[0, 1][..], &[][..]),
        (&[][..], &[1, 0][..]),
    ] {
        let spec = ContractSpec {
            lhs: &[1],
            rhs: &[0],
            codomain,
            domain,
        };
        let error = a.contract(engine, &a, &spec).unwrap_err();
        assert!(
            matches!(
                &error,
                symtn::typed::Error::Operation(operation)
                    if matches!(
                        **operation,
                        symtn::typed::OperationError::UnsupportedTensorContractScope {
                            message: symtn::typed::NON_SYMMETRIC_CONTRACTION_UNSUPPORTED
                        }
                    )
            ),
            "{codomain:?} <- {domain:?}: {error:?}"
        );
    }
}

/// Checked Generic providers take the same split through their own
/// destination derivation: SU(3) adjoint legs carry fusion multiplicity.
#[cfg(feature = "cgc-gen")]
#[test]
fn checked_generic_split_matches_contract_then_permute() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    use symtn::sector::LieGroupFusionRule;

    let runtime = Runtime::builder().build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 2)]).unwrap();
    let mut next = 0.0;
    let mut value = |_: &_, _: &[usize]| {
        next += 0.25;
        next
    };
    let a =
        TensorMap::<_, f64>::from_subblock_fn(engine, &runtime, [&leg, &leg], [&leg], &mut value)
            .unwrap();
    let b =
        TensorMap::<_, f64>::from_subblock_fn(engine, &runtime, [&leg], [&leg, &leg], &mut value)
            .unwrap();
    let identity = [0, 1, 2, 3];
    let default = ContractSpec {
        lhs: &[2],
        rhs: &[0],
        codomain: &identity[..2],
        domain: &identity[2..],
    };
    let contracted = a.contract(engine, &b, &default).unwrap();
    for order in permutations(4) {
        for split in 0..=4 {
            let (codomain, domain) = order.split_at(split);
            let expected = contracted.permute(engine, codomain, domain).unwrap();
            let spec = ContractSpec {
                codomain,
                domain,
                ..default
            };
            let got = a.contract(engine, &b, &spec).unwrap();
            assert_eq!(got.codomain(), expected.codomain());
            assert_eq!(got.domain(), expected.domain());
            let scale = expected
                .dense_data()
                .unwrap()
                .iter()
                .fold(0.0_f64, |max, x| max.max(x.abs()));
            for (x, y) in got
                .dense_data()
                .unwrap()
                .iter()
                .zip(expected.dense_data().unwrap())
            {
                assert!(
                    (x - y).abs() <= 1e-12 * (1.0 + scale),
                    "{codomain:?} <- {domain:?}: {x} vs {y}"
                );
            }
        }
    }
}
