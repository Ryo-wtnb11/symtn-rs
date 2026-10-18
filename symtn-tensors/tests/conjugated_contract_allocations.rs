//! #1368: a warm contraction with a conjugated source reuses its space's
//! adjoint structure instead of rebuilding it on every call.

use std::sync::Arc;

use symtn_core::{
    CheckedFusionAlgebra, FermionParityFusionRule, FusionProductSpace, FusionTreeHomSpace,
    MultiplicityFreeRigidSymbols, ProductFusionRule, ProductSectorCodec, SU2FusionRule, SU2Irrep,
    SectorId, SectorLeg, TensorKitProductCodec, U1FusionRule, U1Irrep,
};
use symtn_tensors::{
    BoundDynamicFusionMapSpace, OutputAxisOrder, TensorContractFusionExecutionContext,
    TensorContractSpec, TreeTransformRuleCacheKey,
};

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

/// Allocation calls and bytes of one warm conjugated-source contraction
/// `conj(A) · B` over `A, B: V ← V`, plus the retained bytes of the memoized
/// adjoint structure.
fn warm_conjugated_contract<R>(rule: R, legs: &[(SectorId, usize)]) -> (usize, usize, usize)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + TreeTransformRuleCacheKey,
    R::Key: 'static + Clone + Eq + std::hash::Hash + Send + Sync,
{
    let rule = Arc::new(rule);
    let space = || {
        let leg = || SectorLeg::new(legs.iter().copied(), false);
        BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(
            Arc::clone(&rule),
            FusionTreeHomSpace::new(
                FusionProductSpace::new([leg()]),
                FusionProductSpace::new([leg()]),
            ),
        )
        .unwrap()
    };
    let (lhs, rhs) = (space(), space());
    let dst = BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
        &space().adjoint_view().unwrap(),
        &rhs,
        &[1],
        &[0],
        OutputAxisOrder::identity(),
    )
    .unwrap();
    let len = lhs.space().required_len().unwrap();
    let lhs_data: Vec<f64> = (0..len).map(|i| i as f64 * 0.25 - 1.0).collect();
    let rhs_data: Vec<f64> = (0..len).map(|i| 2.0 - i as f64 * 0.5).collect();
    let mut out = vec![0.0; dst.space().required_len().unwrap()];
    let mut context = TensorContractFusionExecutionContext::<f64, R::Key>::default();
    let mut call = || {
        context
            .tensorcontract_fusion_dyn_into(
                &dst,
                &mut out,
                &lhs,
                &lhs_data,
                &rhs,
                &rhs_data,
                TensorContractSpec::new_with_conjugation(
                    &[0],
                    &[0],
                    OutputAxisOrder::identity(),
                    true,
                    false,
                ),
                1.0,
                0.0,
            )
            .unwrap();
    };
    call();
    call();
    counting_alloc::start();
    call();
    let allocs = counting_alloc::stop();
    let retained = lhs
        .adjoint_view()
        .unwrap()
        .space()
        .structure()
        .charged_retained_bytes();
    (allocs.calls as usize, allocs.bytes as usize, retained)
}

fn u1(deg: usize) -> Vec<(SectorId, usize)> {
    (-1..=1)
        .map(|charge| (U1Irrep::new(charge).sector_id(), deg))
        .collect()
}

fn su2(deg: usize) -> Vec<(SectorId, usize)> {
    (0..3)
        .map(|twice| (SU2Irrep::from_twice_spin(twice).sector_id(), deg))
        .collect()
}

fn fz2_u1(deg: usize) -> Vec<(SectorId, usize)> {
    (-1i32..=1)
        .map(|charge| {
            let parity = SectorId::new(charge.rem_euclid(2) as _);
            let sector =
                TensorKitProductCodec::try_encode(parity, U1Irrep::new(charge).sector_id())
                    .unwrap();
            (sector, deg)
        })
        .collect()
}

#[test]
fn warm_conjugated_contract_allocations_are_pinned() {
    let _serial = counting_alloc::serial();
    let fz2_u1_rule = || {
        ProductFusionRule::<FermionParityFusionRule, U1FusionRule>::new(
            FermionParityFusionRule,
            U1FusionRule,
        )
    };
    for deg in [2, 16] {
        let rows = [
            ("U1", warm_conjugated_contract(U1FusionRule, &u1(deg))),
            ("SU2", warm_conjugated_contract(SU2FusionRule, &su2(deg))),
            (
                "fZ2xU1",
                warm_conjugated_contract(fz2_u1_rule(), &fz2_u1(deg)),
            ),
        ];
        for (name, (calls, bytes, retained)) in rows {
            eprintln!("{name} d{deg}: {calls} calls, {bytes} bytes; adjoint memo {retained} bytes");
        }
        // What: the warm count does not grow with the degeneracy and holds no
        // per-call adjoint rebuild. The conjugation flag is the lazy adjoint
        // of the owned space, which the core reads in place: 5 calls,
        // 624 bytes for every row.
        for ((name, (calls, bytes, retained)), (expected_calls, max_bytes)) in
            rows.into_iter().zip([(5, 1024), (5, 1024), (5, 1024)])
        {
            assert_eq!(calls, expected_calls, "{name} d{deg}");
            assert!(bytes <= max_bytes, "{name} d{deg}: {bytes} warm bytes");
            assert!(retained <= 4096, "{name} d{deg}: {retained} memo bytes");
        }
    }
}
