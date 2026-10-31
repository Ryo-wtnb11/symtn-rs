use std::sync::Arc;
use symtn_dense::Engine;

use symtn_core::{
    FermionParityFusionRule, FusionProductSpace, FusionTreeHomSpace, MultiplicityFreeFusionRule,
    MultiplicityFreeRigidSymbols, ProductFusionRule, ProductSectorCodec, SU2Irrep, SectorId,
    SectorLeg, TensorKitProductCodec, U1FusionRule, U1Irrep,
};
use symtn_operations::ContractDestinationInit;
use symtn_tensors::{
    BoundDynamicFusionMapSpace, DynamicFusionMapSpace, FusionOperand, HostEagerExecutor,
    TensorContractFusionExecutionContext, TreeTransformRuleCacheKey,
};

use crate::counting_alloc;

const WORKLOADS: [([usize; 2], [usize; 2], [usize; 4]); 3] = [
    ([2, 3], [0, 1], [0, 1, 2, 3]),
    ([3, 2], [0, 1], [0, 1, 2, 3]),
    ([3, 2], [0, 1], [1, 0, 2, 3]),
];

fn u1_sectors() -> Vec<SectorId> {
    [-1, 0, 1]
        .into_iter()
        .map(|charge| U1Irrep::new(charge).sector_id())
        .collect()
}

fn product_sectors() -> Vec<SectorId> {
    [(-1, 1), (0, 0), (1, 1)]
        .into_iter()
        .map(|(charge, parity)| {
            TensorKitProductCodec::try_encode(
                U1Irrep::new(charge).sector_id(),
                SectorId::new(parity),
            )
            .unwrap()
        })
        .collect()
}

/// A context replay of a structurally prepared resolution equals the planned
/// eager contraction. Structural preparation does not include numerical
/// preparation: this path prepares the dense backend's grouped GEMM plans per
/// call, so its allocations are reported, not gated. The numerical no-rebuild
/// contract belongs to the caller-owned member workspace
/// (`HostContractMembersWorkspace`, gated through `ContractPlan` and
/// `ComposePlan` in symtn's `prepared_member1_core_reuse`).
fn assert_replay_matches_planned_eager<R>(engine: &mut dyn Engine, rule: &R, sectors: &[SectorId])
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>
        + MultiplicityFreeFusionRule
        + TreeTransformRuleCacheKey
        + Clone,
    R::Key: 'static + Clone + Eq + std::hash::Hash + Send + Sync,
{
    let provider = Arc::new(rule.clone());
    let bind = |space: &DynamicFusionMapSpace| {
        BoundDynamicFusionMapSpace::bind_multiplicity_free((*space).clone(), Arc::clone(&provider))
            .unwrap()
    };
    let leg = || SectorLeg::new(sectors.iter().map(|&sector| (sector, 1)), false);
    let homspace = || {
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg()]),
            FusionProductSpace::new([leg(), leg()]),
        )
    };
    let space =
        |hom: FusionTreeHomSpace| DynamicFusionMapSpace::from_final_homspace(rule, hom).unwrap();
    let lhs_space = space(homspace());
    let rhs_space = space(homspace());
    let lhs_data: Vec<f64> = (0..lhs_space.required_len().unwrap())
        .map(|index| index as f64 * 0.25 - 2.0)
        .collect();
    let rhs_data: Vec<f64> = (0..rhs_space.required_len().unwrap())
        .map(|index| index as f64 * 0.5 - 3.0)
        .collect();
    let (lhs_bound, rhs_bound) = (bind(&lhs_space), bind(&rhs_space));

    for (lhs_axes, rhs_axes, output_axes) in WORKLOADS {
        let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
            rule,
            lhs_space.homspace(),
            rhs_space.homspace(),
            &lhs_axes,
            &rhs_axes,
            &output_axes,
            2,
        )
        .unwrap();
        let dst_space = space(dst_hom);
        let mut expected = vec![0.0; dst_space.required_len().unwrap()];
        let mut context = TensorContractFusionExecutionContext::<f64, R::Key>::default();
        let dst_bound = bind(&dst_space);
        let dst_structure = Arc::clone(dst_space.structure());
        context
            .tensorcontract_planned_into(
                engine,
                &dst_bound,
                &mut expected,
                (FusionOperand::direct(lhs_bound.space()), &lhs_data),
                (FusionOperand::direct(rhs_bound.space()), &rhs_data),
                &lhs_axes,
                &rhs_axes,
                &output_axes,
                1.0,
                ContractDestinationInit::Axpby(0.0),
            )
            .unwrap();
        let mut actual = expected.clone();
        let prepared = context
            .plan_contract::<HostEagerExecutor, R>(
                engine,
                &dst_bound,
                FusionOperand::direct(lhs_bound.space()),
                FusionOperand::direct(rhs_bound.space()),
                &lhs_axes,
                &rhs_axes,
                &output_axes,
            )
            .unwrap();
        let mut replay = |context: &mut TensorContractFusionExecutionContext<f64, R::Key>,
                          dst: &mut [f64]| {
            context.execute_contract_resolution_host(
                engine,
                &prepared,
                (&dst_structure, dst),
                (lhs_space.structure(), &lhs_data),
                (rhs_space.structure(), &rhs_data),
                1.0,
                ContractDestinationInit::Axpby(0.0),
            )
        };
        for _ in 0..2 {
            replay(&mut context, &mut actual).unwrap();
        }
        counting_alloc::start();
        let result = replay(&mut context, &mut actual);
        let allocs = counting_alloc::stop();
        result.unwrap();

        assert_eq!(actual, expected);
        eprintln!(
            "axes={lhs_axes:?}/{output_axes:?}: warm replay {} calls, {} B",
            allocs.calls, allocs.bytes
        );
    }
}

#[test]
fn warmed_prepared_rank4_fusion_replay_matches_planned_eager() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    assert_replay_matches_planned_eager(engine, &U1FusionRule, &u1_sectors());
    assert_replay_matches_planned_eager(
        engine,
        &FermionParityFusionRule,
        &[SectorId::new(0), SectorId::new(1)],
    );
    assert_replay_matches_planned_eager(
        engine,
        &symtn_core::SU2FusionRule,
        &[
            SU2Irrep::from_twice_spin(0).sector_id(),
            SU2Irrep::from_twice_spin(1).sector_id(),
            SU2Irrep::from_twice_spin(2).sector_id(),
        ],
    );
    assert_replay_matches_planned_eager(
        engine,
        &ProductFusionRule::<U1FusionRule, FermionParityFusionRule>::try_new(
            U1FusionRule,
            FermionParityFusionRule,
        )
        .unwrap(),
        &product_sectors(),
    );
}
