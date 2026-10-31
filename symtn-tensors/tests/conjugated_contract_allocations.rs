//! A warm contraction with a lazy-adjoint source reuses its space's adjoint
//! structure instead of rebuilding it on every call.

use std::sync::Arc;
use symtn_dense::Engine;

use symtn_core::{
    CheckedFusionAlgebra, FermionParityFusionRule, FusionProductSpace, FusionTreeHomSpace,
    MultiplicityFreeRigidSymbols, ProductFusionRule, ProductSectorCodec, SU2FusionRule, SU2Irrep,
    SectorId, SectorLeg, TensorKitProductCodec, U1FusionRule, U1Irrep,
};
use symtn_tensors::{
    BoundDynamicFusionMapSpace, ContractDestinationInit, FusionOperand, OutputAxisOrder,
    TensorContractFusionExecutionContext, TreeTransformRuleCacheKey,
};

use crate::counting_alloc;

/// Allocation calls and bytes of one warm lazy-adjoint-source contraction
/// `A† · B` over `A, B: V ← V`, plus the retained bytes of the memoized
/// adjoint structure.
fn warm_conjugated_contract<R>(
    engine: &mut dyn Engine,
    rule: R,
    legs: &[(SectorId, usize)],
) -> (usize, usize, usize)
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
            .tensorcontract_planned_into(
                engine,
                &dst,
                &mut out,
                (FusionOperand::adjoint(lhs.space()), &lhs_data),
                (FusionOperand::direct(rhs.space()), &rhs_data),
                &[1],
                &[0],
                &[0, 1],
                1.0,
                ContractDestinationInit::Axpby(0.0),
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

/// Purpose: a warm lazy-adjoint-source contraction rebuilds no adjoint
/// structure: its calls, bytes and the adjoint memo's retained bytes are the
/// same at degeneracy 2 and 16.
#[test]
fn warm_conjugated_contract_allocations_do_not_grow_with_degeneracy() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    symtn_core::clear_structure_caches();
    let fz2_u1_rule = || {
        ProductFusionRule::<FermionParityFusionRule, U1FusionRule>::try_new(
            FermionParityFusionRule,
            U1FusionRule,
        )
        .unwrap()
    };
    let mut rows = |deg| {
        [
            (
                "U1",
                warm_conjugated_contract(engine, U1FusionRule, &u1(deg)),
            ),
            (
                "SU2",
                warm_conjugated_contract(engine, SU2FusionRule, &su2(deg)),
            ),
            (
                "fZ2xU1",
                warm_conjugated_contract(engine, fz2_u1_rule(), &fz2_u1(deg)),
            ),
        ]
    };
    let (small, large) = (rows(2), rows(16));
    // What: the core reads the owned space's lazy adjoint in place, so
    // nothing the warm call allocates or the memo retains depends on the
    // degeneracy.
    for ((name, small), (_, large)) in small.into_iter().zip(large) {
        assert_eq!(
            small, large,
            "{name}: (calls, bytes, memo bytes) d2 against d16"
        );
    }
}
