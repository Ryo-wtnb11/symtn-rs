//! Complex composition and contraction agree with their real and imaginary
//! parts.

use std::sync::Arc;
use symtn::typed::ContractSpec;

use symtn::sector::{
    CheckedFusionAlgebra, MultiplicityFreeAdmissionMode, MultiplicityFreeRigidSymbols, SectorCodec,
    TypedSectorAdmission,
};
use symtn::sector::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use symtn::typed::FusionAlgebraError;
use symtn::typed::{Complex64, Runtime};
use symtn::typed::{GradedSpace, TensorMap};

fn one() -> Complex64 {
    Complex64::new(1.0, 0.0)
}

fn i() -> Complex64 {
    Complex64::new(0.0, 1.0)
}

fn complexify<R>(re: &TensorMap<R, f64>, im: &TensorMap<R, f64>) -> TensorMap<R, Complex64>
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
{
    re.convert::<Complex64>()
        .axpby(one(), &im.convert::<Complex64>(), i())
        .unwrap()
}

fn assert_close(actual: &[Complex64], expected: &[Complex64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).norm() <= tolerance,
            "element {index}: {actual} vs {expected}"
        );
    }
}

fn assert_complex_contract_identity<R>(runtime: &Runtime, space: &GradedSpace<R>)
where
    R: TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = FusionAlgebraError,
            Mode = MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
{
    let a =
        TensorMap::<R, f64>::rand_with_seed(runtime, [space, space], [space, space], 11).unwrap();
    let b =
        TensorMap::<R, f64>::rand_with_seed(runtime, [space, space], [space, space], 12).unwrap();
    let c =
        TensorMap::<R, f64>::rand_with_seed(runtime, [space, space], [space, space], 13).unwrap();
    let d =
        TensorMap::<R, f64>::rand_with_seed(runtime, [space, space], [space, space], 14).unwrap();

    let x = complexify(&a, &b);
    let y = complexify(&c, &d);
    let real = a
        .compose(&c)
        .unwrap()
        .axpby(1.0, &b.compose(&d).unwrap(), -1.0)
        .unwrap();
    let imaginary = a
        .compose(&d)
        .unwrap()
        .axpby(1.0, &b.compose(&c).unwrap(), 1.0)
        .unwrap();
    let expected = complexify(&real, &imaginary);

    assert_close(
        x.compose(&y).unwrap().dense_data().unwrap(),
        expected.dense_data().unwrap(),
        1.0e-12,
    );
    assert_close(
        x.contract(
            &y,
            &ContractSpec {
                lhs: &[2, 3],
                rhs: &[0, 1],
                codomain: &[0, 1],
                domain: &[2, 3],
            },
        )
        .unwrap()
        .dense_data()
        .unwrap(),
        expected.dense_data().unwrap(),
        1.0e-12,
    );
}

/// `(A + iB)(C + iD) = (AC - BD) + i(AD + BC)` through both typed
/// composition and arbitrary-axis contraction.
#[test]
fn c64_contract_matches_real_imag_decomposition() {
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
    assert_complex_contract_identity(&runtime, &u1);

    let su2 = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 1),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap();
    assert_complex_contract_identity(&runtime, &su2);
}
