//! Host pin of the partial-trace oracles the device gate
//! (`typed_cuda_trace.rs`, #1349) compares against: the physical-basis
//! diagonal sum (U(1), SU(2)) and the identity contraction (twist-free
//! providers) agree with Host `trace_pairs` on every fixture and payload
//! dtype, without a device, so ordinary CI holds them.

use crate::contract_cases;
use crate::numerics;
use crate::trace_cases;

use contract_cases::Payload;
use num_complex::{Complex32, Complex64};
use symtn::sector::{CheckedFusionAlgebra, MultiplicityFreeRigidSymbols, SectorCodec};
use symtn::typed::Runtime;
use trace_cases::{
    dense_trace, fermion_su2_cases, fermion_u1_cases, su2_cases, u1_cases, u1_su2_cases, TraceCase,
};

fn check_identity<R, D>(case: &TraceCase<R, D>)
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
    if let Some(identity) = &case.identity {
        let host = case.host();
        assert_eq!(
            host.codomain_rank(),
            identity.codomain_rank(),
            "{}",
            case.name
        );
        numerics::assert_nonzero_slices_close(
            case.name,
            host.dense_data().unwrap(),
            identity.dense_data().unwrap(),
            case.terms(),
        );
    }
}

fn check_dense<R, D>(case: &TraceCase<R, D>)
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
    check_identity(case);
    if case.dense {
        let host = case.host().to_physical_dense().unwrap().data;
        numerics::assert_nonzero_slices_close(case.name, &host, &dense_trace(case), case.terms());
    }
}

fn every_fixture<D: Payload>() {
    let runtime = Runtime::builder().build().unwrap();
    let mut oracles = 0;
    for case in u1_cases::<D>(&runtime) {
        oracles += usize::from(case.dense) + usize::from(case.identity.is_some());
        check_dense(&case);
    }
    for case in su2_cases::<D>(&runtime) {
        oracles += usize::from(case.dense) + usize::from(case.identity.is_some());
        check_dense(&case);
    }
    for case in u1_su2_cases::<D>(&runtime) {
        oracles += usize::from(case.identity.is_some());
        check_identity(&case);
    }
    assert_eq!(
        oracles, 12,
        "every bosonic fixture has an independent oracle"
    );
    // Fermionic fixtures only need to be admissible here: their oracle is the
    // Host itself plus the hand-valued fZ2 supertrace on the device side.
    for case in fermion_u1_cases::<D>(&runtime)
        .iter()
        .chain(&trace_cases::lazy(fermion_u1_cases::<D>(&runtime)))
    {
        case.host();
    }
    for case in fermion_su2_cases::<D>(&runtime) {
        case.host();
    }
}

#[test]
fn the_trace_oracles_agree_with_the_host_at_every_payload_dtype() {
    every_fixture::<f64>();
    every_fixture::<Complex64>();
    every_fixture::<f32>();
    every_fixture::<Complex32>();
}
