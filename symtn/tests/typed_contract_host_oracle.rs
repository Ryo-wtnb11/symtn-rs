//! Host pin of the general-contraction oracles the device gate
//! (`typed_cuda_contract.rs`, #1345) compares against.
//!
//! An oracle is only independent evidence for the device once it is shown to
//! agree with the Host on the same fixtures; this file does that without a
//! device, so it runs in ordinary CI:
//!
//! * TensorKit's `blas_contract!` step sequence on Host typed ops, for every
//!   fixture and payload dtype;
//! * the physical-basis dense contraction, for the U(1) and SU(2) fixtures
//!   whose contracted legs do not bend;
//! * for fermionic providers (#1347), the same sequence with
//!   TensorKit's `twist!` on the B role and on the A role — which must agree
//!   with each other and with the Host — and a twist-free control the Host
//!   must *not* match, so every fixture really exercises the twist;
//! * the TensorKit-valued FZ2 closed loops as explicit `contract` calls;
//! * `contract_into` over a NaN-poisoned destination equals the
//!   returning contraction on every fixture (#1346), so the Host
//!   overwrite is the same oracle the device overwrite is gated against.

use crate::contract_cases;
use crate::numerics;
use symtn::typed::Engine;

use contract_cases::{
    blas_contract_oracle, dense_oracle, fermionic_blas_contract_oracle, fz2_tensorkit_loops,
    lazy_cases, poisoned_destination, product_general, su2_bent, su2_core_form_lazy_cases,
    su2_reordered, u1_inactive_cases, u1_lhs_identity, u1_rank_five, u1_reordered, u1_rhs_identity,
    Case, Payload, TwistRole,
};
use num_complex::{Complex32, Complex64};
use symtn::sector::{CheckedFusionAlgebra, MultiplicityFreeRigidSymbols, SectorCodec};
use symtn::typed::{Runtime, TensorMap};

fn check_blas<R, D>(engine: &mut dyn Engine, case: Case<R, D>)
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
    let host = case.host(engine);
    let oracle = blas_contract_oracle(engine, &case);
    assert_eq!(
        host.codomain_rank(),
        oracle.codomain_rank(),
        "{}",
        case.name
    );
    numerics::assert_nonzero_slices_close(
        case.name,
        host.dense_data().unwrap(),
        oracle.dense_data().unwrap(),
        case.terms(),
    );
}

fn every_fixture<D: Payload>(engine: &mut dyn Engine) {
    let runtime = Runtime::builder().build().unwrap();
    let operand = u1_rank_five::<D>(engine, &runtime);
    check_blas(engine, operand);
    let operand = u1_reordered::<D>(engine, &runtime);
    check_blas(engine, operand);
    let operand = su2_reordered::<D>(engine, &runtime);
    check_blas(engine, operand);
    let operand = su2_bent::<D>(engine, &runtime);
    check_blas(engine, operand);
    let operand = product_general::<D>(engine, &runtime);
    check_blas(engine, operand);
    let operand = u1_lhs_identity::<D>(engine, &runtime);
    check_blas(engine, operand);
    let operand = u1_rhs_identity::<D>(engine, &runtime);
    check_blas(engine, operand);
    let operand = &u1_rank_five::<D>(engine, &runtime).lhs;
    for case in lazy_cases(engine, operand, "U(1) lazy") {
        check_blas(engine, case);
    }
    let operand = &su2_reordered::<D>(engine, &runtime).lhs;
    for case in lazy_cases(engine, operand, "SU(2) lazy") {
        check_blas(engine, case);
    }
    let operand = &product_general::<D>(engine, &runtime).lhs;
    for case in lazy_cases(engine, operand, "U(1) x SU(2) lazy") {
        check_blas(engine, case);
    }
    for case in su2_core_form_lazy_cases::<D>(engine, &runtime) {
        check_blas(engine, case);
    }
}

#[test]
fn host_contract_matches_the_tensorkit_blas_contract_sequence_at_every_dtype() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    every_fixture::<f64>(engine);
    every_fixture::<Complex64>(engine);
    every_fixture::<f32>(engine);
    every_fixture::<Complex32>(engine);
}

fn check_dense<R, D>(case: Case<R, D>)
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + symtn::sector::PhysicalFusionBasis<Scalar = f64>,
    D: Payload,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    assert!(case.dense, "{}", case.name);
    let (shape, expected) = dense_oracle(engine, &case);
    let actual = case.host(engine).to_physical_dense(engine).unwrap();
    assert_eq!(actual.shape, shape, "{}", case.name);
    numerics::assert_nonzero_slices_close(case.name, &actual.data, &expected, case.terms());
}

#[test]
fn host_contract_matches_the_physical_basis_contraction() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    check_dense(u1_reordered::<f64>(engine, &runtime));
    check_dense(u1_reordered::<Complex64>(engine, &runtime));
    check_dense(su2_reordered::<f64>(engine, &runtime));
    check_dense(su2_reordered::<Complex64>(engine, &runtime));
}

fn check_fermionic<R, D>(
    engine: &mut dyn Engine,
    case: Case<R, D>,
    twist: impl Fn(&mut dyn Engine, &TensorMap<R, D>, &[usize]) -> TensorMap<R, D> + Copy,
) where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: Payload,
{
    let host = case.host(engine);
    let b_role = fermionic_blas_contract_oracle(engine, &case, TwistRole::B, twist);
    let a_role = fermionic_blas_contract_oracle(engine, &case, TwistRole::A, twist);
    numerics::assert_nonzero_slices_close(
        case.name,
        a_role.dense_data().unwrap(),
        b_role.dense_data().unwrap(),
        case.terms(),
    );
    numerics::assert_nonzero_slices_close(
        case.name,
        host.dense_data().unwrap(),
        b_role.dense_data().unwrap(),
        case.terms(),
    );
    let untwisted = fermionic_blas_contract_oracle(engine, &case, TwistRole::None, twist);
    let scale = host
        .dense_data()
        .unwrap()
        .iter()
        .map(|value| value.magnitude())
        .fold(0.0_f64, f64::max);
    let differs = host
        .dense_data()
        .unwrap()
        .iter()
        .zip(untwisted.dense_data().unwrap())
        .any(|(&left, &right)| left.distance(right) > 1e-3 * scale);
    assert!(differs, "{}: the twist changes nothing here", case.name);
}

#[test]
fn host_fermionic_contract_matches_both_tensorkit_twist_roles_at_every_dtype() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    for_each_fermionic_fixture!(engine, &runtime, f64, check_fermionic);
    for_each_fermionic_fixture!(engine, &runtime, Complex64, check_fermionic);
    for_each_fermionic_fixture!(engine, &runtime, f32, check_fermionic);
    for_each_fermionic_fixture!(engine, &runtime, Complex32, check_fermionic);
}

#[test]
fn host_fz2_loops_as_explicit_contracts_match_tensorkit() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let loops = fz2_tensorkit_loops(
        engine,
        &runtime,
        |tensor| tensor,
        |engine, x, y, spec| x.contract(engine, y, spec).unwrap(),
        |engine, tensor| tensor.scalar(engine).unwrap(),
    );
    for (name, value, expected) in loops {
        assert!(
            (value - expected).abs() < 1e-12,
            "{name}: {value} vs {expected}"
        );
    }
}

fn check_overwrite<R, D>(case: Case<R, D>)
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
    let mut destination = poisoned_destination(engine, &case);
    case.lhs
        .contract_into(
            engine,
            &case.rhs,
            &case.spec(),
            &mut destination,
            D::entry(1.0, 0.0),
            D::entry(0.0, 0.0),
        )
        .unwrap();
    numerics::assert_nonzero_slices_close(
        case.name,
        destination.dense_data().unwrap(),
        case.host(engine).dense_data().unwrap(),
        case.terms(),
    );
}

fn check_overwrite_fermionic<R, D>(
    _engine: &mut dyn Engine,
    case: Case<R, D>,
    _twist: impl Fn(&mut dyn Engine, &TensorMap<R, D>, &[usize]) -> TensorMap<R, D> + Copy,
) where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: Payload,
{
    check_overwrite(case);
}

fn every_overwrite_fixture<D: Payload>(engine: &mut dyn Engine) {
    let runtime = Runtime::builder().build().unwrap();
    check_overwrite(u1_rank_five::<D>(engine, &runtime));
    check_overwrite(u1_reordered::<D>(engine, &runtime));
    check_overwrite(su2_reordered::<D>(engine, &runtime));
    check_overwrite(su2_bent::<D>(engine, &runtime));
    check_overwrite(product_general::<D>(engine, &runtime));
    check_overwrite(u1_lhs_identity::<D>(engine, &runtime));
    check_overwrite(u1_rhs_identity::<D>(engine, &runtime));
    for case in u1_inactive_cases::<D>(engine, &runtime) {
        check_overwrite(case);
    }
    let operand = &u1_rank_five::<D>(engine, &runtime).lhs;
    for case in lazy_cases(engine, operand, "U(1) lazy") {
        check_overwrite(case);
    }
    let operand = &su2_reordered::<D>(engine, &runtime).lhs;
    for case in lazy_cases(engine, operand, "SU(2) lazy") {
        check_overwrite(case);
    }
    for case in su2_core_form_lazy_cases::<D>(engine, &runtime) {
        check_overwrite(case);
    }
    for_each_fermionic_fixture!(engine, &runtime, D, check_overwrite_fermionic);
}

#[test]
fn host_into_a_poisoned_destination_matches_the_returning_contraction() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    every_overwrite_fixture::<f64>(engine);
    every_overwrite_fixture::<Complex64>(engine);
    every_overwrite_fixture::<f32>(engine);
    every_overwrite_fixture::<Complex32>(engine);
}
