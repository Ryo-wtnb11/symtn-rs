//! Complex user-level contracts not already owned by the typed operation and
//! decomposition suites.

use std::sync::Arc;

use symtn::sector::{
    CheckedFusionAlgebra, MultiplicityFreeAdmissionMode, MultiplicityFreeRigidSymbols, SectorCodec,
    TypedSectorAdmission,
};
use symtn::sector::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use symtn::typed::FusionAlgebraError;
use symtn::typed::{Complex64, Runtime};
use symtn::typed::{GradedSpace, TensorMap};

#[path = "../../tests/support/network.rs"]
mod network_support;
use network_support::{conj, net, op};

fn one() -> Complex64 {
    Complex64::new(1.0, 0.0)
}

fn assert_typed_network_conj<R>(runtime: &Runtime, p: &GradedSpace<R>)
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
    let psi = TensorMap::<R, Complex64>::rand_with_seed(runtime, [p], [&l, &r], 71).unwrap();
    let h0 = TensorMap::<R, Complex64>::rand_with_seed(runtime, [p], [p], 72).unwrap();
    let h = h0
        .axpby(one() * 0.5, &h0.adjoint().unwrap(), one() * 0.5)
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
    .contract(&[&psi, &h, &psi])
    .unwrap()
    .scalar()
    .unwrap();
    let norm = net(
        &[conj(op(&["p"], &["l", "r"])), op(&["p"], &["l", "r"])],
        &[],
        &[],
    )
    .contract(&[&psi, &psi])
    .unwrap()
    .scalar()
    .unwrap();
    assert!(norm.re > 0.0);
    assert!(norm.im.abs() <= 1.0e-12 * (1.0 + norm.re));
    assert!(expectation.im.abs() <= 1.0e-10 * (1.0 + expectation.norm()));

    let via_inner = psi.inner(&h.compose(&psi).unwrap()).unwrap();
    assert!((via_inner - expectation).norm() <= 1.0e-10 * (1.0 + expectation.norm()));
}

/// The network layer lowers `conj` to typed adjoint, including complex
/// conjugation: a Hermitian expectation value is real.
#[test]
fn network_conj_expectation_value_is_real() {
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
    assert_typed_network_conj(&runtime, &u1);

    let su2 = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 1),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap();
    assert_typed_network_conj(&runtime, &su2);
}
