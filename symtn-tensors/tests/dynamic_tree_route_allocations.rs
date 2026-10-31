//! Allocation contracts of the Host DynamicTree route, eager and member
//! (B = 1, 17), on one thread (#1859):
//!
//! - a warm member replay allocates nothing, at B = 1 and B = 17;
//! - a warm eager call and a warm member replay build no GEMM plan (the
//!   geometry-plan cache's misses do not move);
//! - a warm eager call and a member workspace's cold call make as many
//!   allocation calls on a larger structure of the same fixture (more
//!   sectors, so more blocks and block geometries) as on the small one:
//!   their cost is per call, not per block;
//! - dropping a member workspace frees no more than its `retained_bytes`.
//!
//! Exact counts and bytes are printed, not pinned: they move with platform
//! and dependency internals.

use std::sync::Arc;
use symtn_dense::Engine;

use symtn_core::{
    FusionProductSpace, FusionTreeHomSpace, MultiplicityFreeRigidSymbols, SU2FusionRule, SU2Irrep,
    SectorLeg, StructureCacheKind, U1FusionRule, U1Irrep,
};
use symtn_tensors::{
    BoundDynamicFusionMapSpace, DirectCoreExecutor, FusionOperand, HostContractMembersWorkspace,
    OutputAxisOrder, RuleIdentity, TensorContractFusionExecutionContext, TreeTransformRuleCacheKey,
};

use crate::counting_alloc;

type Context = TensorContractFusionExecutionContext<f64, RuleIdentity>;

struct Case<R> {
    lhs: BoundDynamicFusionMapSpace<R>,
    rhs: BoundDynamicFusionMapSpace<R>,
    lhs_axes: Vec<usize>,
    rhs_axes: Vec<usize>,
    output_axes: Vec<usize>,
}

fn space<R: MultiplicityFreeRigidSymbols<Scalar = f64>>(
    provider: &Arc<R>,
    codomain: Vec<SectorLeg>,
    domain: Vec<SectorLeg>,
) -> BoundDynamicFusionMapSpace<R> {
    BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::clone(provider),
        FusionTreeHomSpace::new(
            FusionProductSpace::new(codomain),
            FusionProductSpace::new(domain),
        ),
    )
    .unwrap()
}

/// U(1) charges `-reach..=reach`, degeneracy two for charge zero, else one.
fn u1_leg(reach: i32) -> SectorLeg {
    SectorLeg::new(
        (-reach..=reach).map(|charge| {
            (
                U1Irrep::new(charge).sector_id(),
                if charge == 0 { 2 } else { 1 },
            )
        }),
        false,
    )
}

/// SU(2) spins `0..=reach/2`, degeneracy two for spin zero, else one.
fn su2_leg(reach: i32) -> SectorLeg {
    SectorLeg::new(
        (0..=reach).map(|twice| {
            (
                SU2Irrep::from_twice_spin(twice as _).sector_id(),
                if twice == 0 { 2 } else { 1 },
            )
        }),
        false,
    )
}

/// Source transforms on both sides and an output transform.
fn transformed<R: MultiplicityFreeRigidSymbols<Scalar = f64>>(
    provider: R,
    leg: impl Fn() -> SectorLeg,
) -> Case<R> {
    let provider = Arc::new(provider);
    Case {
        lhs: space(&provider, vec![leg(), leg(), leg()], vec![leg(), leg()]),
        rhs: space(&provider, vec![leg(), leg()], vec![leg(), leg()]),
        lhs_axes: vec![3, 1],
        rhs_axes: vec![0, 3],
        output_axes: vec![2, 0, 4, 1, 3],
    }
}

/// A transformed lhs written by the core straight into the destination.
fn identity_output<R: MultiplicityFreeRigidSymbols<Scalar = f64>>(
    provider: R,
    leg: impl Fn() -> SectorLeg,
) -> Case<R> {
    let provider = Arc::new(provider);
    Case {
        lhs: space(&provider, vec![leg()], vec![leg(), leg()]),
        rhs: space(&provider, vec![leg()], vec![leg()]),
        lhs_axes: vec![1],
        rhs_axes: vec![0],
        output_axes: vec![0, 1, 2],
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Row {
    /// Blocks of the destination, the size the scaling check compares.
    blocks: usize,
    eager_warm: u64,
    member1_cold: u64,
    member1_warm: u64,
    member17_cold: u64,
    member17_warm: u64,
    /// GEMM plans built by the warm eager and warm member calls.
    warm_plan_builds: u64,
    retained: usize,
    freed: u64,
}

fn plan_builds() -> u64 {
    symtn_core::structure_cache_info(StructureCacheKind::GemmPlans).misses()
}

fn row<R>(engine: &mut dyn Engine, case: &Case<R>) -> Row
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey<Key = RuleIdentity>,
{
    let dst = BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
        &case.lhs,
        &case.rhs,
        &case.lhs_axes,
        &case.rhs_axes,
        OutputAxisOrder::from_axes(&case.output_axes),
    )
    .unwrap();
    let lhs_len = case.lhs.space().required_len().unwrap();
    let rhs_len = case.rhs.space().required_len().unwrap();
    let dst_len = dst.space().required_len().unwrap();
    let data = |len: usize, salt: f64| -> Vec<f64> {
        (0..len).map(|i| (i as f64 * 0.37 + salt).sin()).collect()
    };
    let (lhs, rhs) = (data(17 * lhs_len, 0.1), data(17 * rhs_len, 0.2));

    let mut context = Context::default();
    let mut out = vec![0.0; dst_len];
    let mut eager = |context: &mut Context, out: &mut [f64]| {
        counting_alloc::measure(|| {
            context
                .tensorcontract_planned_into(
                    engine,
                    &dst,
                    out,
                    (FusionOperand::direct(case.lhs.space()), &lhs[..lhs_len]),
                    (FusionOperand::direct(case.rhs.space()), &rhs[..rhs_len]),
                    &case.lhs_axes,
                    &case.rhs_axes,
                    &case.output_axes,
                    1.0,
                    symtn_tensors::ContractDestinationInit::Axpby(0.0),
                )
                .unwrap()
        })
        .1
        .calls
    };
    eager(&mut context, &mut out);
    let builds = plan_builds();
    let eager_warm = eager(&mut context, &mut out);
    let mut warm_plan_builds = plan_builds() - builds;

    let mut context = Context::default();
    let resolution = context
        .plan_contract::<DirectCoreExecutor, _>(
            engine,
            &dst,
            FusionOperand::direct(case.lhs.space()),
            FusionOperand::direct(case.rhs.space()),
            &case.lhs_axes,
            &case.rhs_axes,
            &case.output_axes,
        )
        .unwrap();
    assert!(resolution.is_dynamic_tree());
    let mut out = vec![0.0; 17 * dst_len];
    let mut member = |members: usize, workspace: &mut HostContractMembersWorkspace<f64>| {
        let out = &mut out[..members * dst_len];
        counting_alloc::measure(|| {
            context
                .execute_storage_contract_members_host(
                    engine,
                    &resolution,
                    (dst.space().structure(), out),
                    (case.lhs.space().structure(), &lhs[..members * lhs_len]),
                    (case.rhs.space().structure(), &rhs[..members * rhs_len]),
                    workspace,
                    members,
                    symtn_tensors::ContractDestinationInit::Axpby(0.0),
                )
                .unwrap()
        })
        .1
        .calls
    };
    // A throwaway workspace per member count admits that count's plans and
    // sizes the context's GEMM staging, so a cold call below measures only
    // what a new workspace allocates for itself.
    let cold_warm =
        |members: usize,
         member: &mut dyn FnMut(usize, &mut HostContractMembersWorkspace<f64>) -> u64| {
            member(members, &mut HostContractMembersWorkspace::default());
            let mut workspace = HostContractMembersWorkspace::<f64>::default();
            let cold = member(members, &mut workspace);
            let builds = plan_builds();
            let warm = member(members, &mut workspace);
            (cold, warm, plan_builds() - builds, workspace)
        };
    let (member1_cold, member1_warm, builds1, workspace) = cold_warm(1, &mut member);
    let retained = workspace.retained_bytes();
    let ((), dropped) = counting_alloc::measure(|| drop(workspace));
    let (member17_cold, member17_warm, builds17, _) = cold_warm(17, &mut member);
    warm_plan_builds += builds1 + builds17;
    Row {
        blocks: dst.space().structure().block_count(),
        eager_warm,
        member1_cold,
        member1_warm,
        member17_cold,
        member17_warm,
        warm_plan_builds,
        retained,
        freed: dropped.freed_bytes,
    }
}

#[test]
fn dynamic_tree_route_allocations_follow_their_contracts() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    type Fixture = fn(&mut dyn Engine, i32) -> Row;
    let fixtures: [(&str, Fixture); 4] = [
        ("U1 transformed", |engine, reach| {
            row(engine, &transformed(U1FusionRule, || u1_leg(reach)))
        }),
        ("U1 identity output", |engine, reach| {
            row(engine, &identity_output(U1FusionRule, || u1_leg(reach)))
        }),
        ("SU2 transformed", |engine, reach| {
            row(engine, &transformed(SU2FusionRule, || su2_leg(reach)))
        }),
        ("SU2 identity output", |engine, reach| {
            row(engine, &identity_output(SU2FusionRule, || su2_leg(reach)))
        }),
    ];
    for (name, row) in fixtures {
        let (small, large) = (row(engine, 1), row(engine, 2));
        eprintln!("{name}: small {small:?}\n{name}: large {large:?}");
        assert!(large.blocks > small.blocks, "{name}: premise");
        for row in [small, large] {
            // What: warm member replays allocate nothing; no warm call
            // builds a GEMM plan; a dropped workspace frees what it charges.
            assert_eq!(
                (row.member1_warm, row.member17_warm),
                (0, 0),
                "{name}: {row:?}"
            );
            assert_eq!(row.warm_plan_builds, 0, "{name}: {row:?}");
            assert!(row.freed as usize <= row.retained, "{name}: {row:?}");
        }
        // What: the warm eager call and a workspace's cold calls cost per
        // call, not per block: the larger structure adds fewer calls than
        // it adds blocks (amortized growth of reused buffers may add a few).
        let added = large.blocks - small.blocks;
        for (what, small, large) in [
            ("eager warm", small.eager_warm, large.eager_warm),
            ("member1 cold", small.member1_cold, large.member1_cold),
            ("member17 cold", small.member17_cold, large.member17_cold),
        ] {
            assert!(
                large.saturating_sub(small) < added as u64,
                "{name}: {what} grows with the block count: {small} -> {large} calls for {added} added blocks"
            );
        }
    }
}
