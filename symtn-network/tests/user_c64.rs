//! Complex user-level contracts not already owned by the typed operation and
//! decomposition suites.

use std::sync::Arc;
use symtn::typed::Engine;

use symtn::sector::{
    CheckedFusionAlgebra, MultiplicityFreeAdmissionMode, MultiplicityFreeRigidSymbols, SectorCodec,
    TypedSectorAdmission,
};
use symtn::sector::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use symtn::typed::FusionAlgebraError;
use symtn::typed::{Complex64, Runtime};
use symtn::typed::{GradedSpace, TensorMap};

use crate::network_support;
use network_support::{conj, net, op};

fn one() -> Complex64 {
    Complex64::new(1.0, 0.0)
}

fn assert_typed_network_conj<R>(engine: &mut dyn Engine, runtime: &Runtime, p: &GradedSpace<R>)
where
    R: TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = FusionAlgebraError,
            Mode = MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send,
{
    let l = p.clone();
    let r = p.try_dual().unwrap();
    let psi =
        TensorMap::<R, Complex64>::rand_with_seed(engine, runtime, [p], [&l, &r], 71).unwrap();
    let h0 = TensorMap::<R, Complex64>::rand_with_seed(engine, runtime, [p], [p], 72).unwrap();
    let h0_adjoint = &h0.adjoint(engine).unwrap();
    let h = h0
        .axpby(engine, one() * 0.5, h0_adjoint, one() * 0.5)
        .unwrap();

    let expectation = net(
        &[
            conj(op(&["p"], &["l", "r"])),
            op(&["p"], &["q"]),
            op(&["q"], &["l", "r"]),
        ],
        &[],
        &[],
    )
    .contract(engine, &[&psi, &h, &psi])
    .unwrap()
    .scalar(engine)
    .unwrap();
    let norm = net(
        &[conj(op(&["p"], &["l", "r"])), op(&["p"], &["l", "r"])],
        &[],
        &[],
    )
    .contract(engine, &[&psi, &psi])
    .unwrap()
    .scalar(engine)
    .unwrap();
    assert!(norm.re > 0.0);
    assert!(norm.im.abs() <= 1.0e-12 * (1.0 + norm.re));
    assert!(expectation.im.abs() <= 1.0e-10 * (1.0 + expectation.norm()));
    let h_compose = &h.compose(engine, &psi).unwrap();

    let via_inner = psi.inner(engine, h_compose).unwrap();
    assert!((via_inner - expectation).norm() <= 1.0e-10 * (1.0 + expectation.norm()));
}

/// The network layer lowers `conj` to typed adjoint, including complex
/// conjugation: a Hermitian expectation value is real.
#[test]
fn network_conj_expectation_value_is_real() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let u1 = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 1),
            (U1Irrep::new(0), 2),
            (U1Irrep::new(1), 1),
        ],
    )
    .unwrap();
    assert_typed_network_conj(engine, &runtime, &u1);

    let su2 = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 1),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap();
    assert_typed_network_conj(engine, &runtime, &su2);
}
