//! Allocation calls and bytes of the Host DynamicTree route, eager and member
//! (B = 1, 17), cold and warm, on one thread (#1859).
//!
//! Call-count upper bounds are those recorded on `d416b1a2`, a revision with
//! separate eager and member replays (the `B = 17` rows re-recorded, see
//! `BASE`); bytes are reported, not asserted.

use std::sync::Arc;
use symtn_dense::Engine;

use symtn_core::{
    FusionProductSpace, FusionTreeHomSpace, MultiplicityFreeRigidSymbols, SU2FusionRule, SU2Irrep,
    SectorLeg, U1FusionRule, U1Irrep,
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

fn u1_leg() -> SectorLeg {
    SectorLeg::new(
        [-1, 0, 1].map(|charge| {
            (
                U1Irrep::new(charge).sector_id(),
                2 - charge.unsigned_abs() as usize,
            )
        }),
        false,
    )
}

fn su2_leg() -> SectorLeg {
    SectorLeg::new(
        [(0, 2), (1, 1), (2, 1)]
            .map(|(twice, deg)| (SU2Irrep::from_twice_spin(twice).sector_id(), deg)),
        false,
    )
}

/// Source transforms on both sides and an output transform.
fn transformed<R: MultiplicityFreeRigidSymbols<Scalar = f64>>(
    provider: R,
    leg: fn() -> SectorLeg,
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
    leg: fn() -> SectorLeg,
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
    eager_cold: (u64, u64),
    eager_warm: (u64, u64),
    member1_cold: (u64, u64),
    member1_warm: (u64, u64),
    member17_cold: (u64, u64),
    member17_warm: (u64, u64),
    member1_retained: usize,
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
    let calls = |allocs: counting_alloc::Allocs| (allocs.calls, allocs.bytes);

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
    };
    let eager_cold = calls(eager(&mut context, &mut out));
    let eager_warm = calls(eager(&mut context, &mut out));

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
    let mut workspace = HostContractMembersWorkspace::<f64>::default();
    let mut out = vec![0.0; 17 * dst_len];
    let mut member = |members: usize, workspace: &mut HostContractMembersWorkspace<f64>| {
        let out = &mut out[..members * dst_len];
        calls(
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
            .1,
        )
    };
    let member1_cold = member(1, &mut workspace);
    let member1_warm = member(1, &mut workspace);
    let member1_retained = workspace.retained_bytes();
    let mut workspace = HostContractMembersWorkspace::<f64>::default();
    let member17_cold = member(17, &mut workspace);
    let member17_warm = member(17, &mut workspace);
    Row {
        eager_cold,
        eager_warm,
        member1_cold,
        member1_warm,
        member17_cold,
        member17_warm,
        member1_retained,
    }
}

#[test]
fn dynamic_tree_route_allocations_are_bounded_by_base() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let rows = [
        (
            "U1 transformed",
            row(engine, &transformed(U1FusionRule, u1_leg)),
        ),
        (
            "U1 identity output",
            row(engine, &identity_output(U1FusionRule, u1_leg)),
        ),
        (
            "SU2 transformed",
            row(engine, &transformed(SU2FusionRule, su2_leg)),
        ),
        (
            "SU2 identity output",
            row(engine, &identity_output(SU2FusionRule, su2_leg)),
        ),
    ];
    for ((name, row), (base_name, base)) in rows.iter().zip(BASE) {
        eprintln!("{name}: {row:?}");
        assert_eq!(*name, base_name);
        // What: the executor merge adds no allocation call to a warm eager
        // replay, and a member replay makes no more calls than its base (no
        // replay build at B = 1). Bytes and the cold eager row (planning)
        // are reported only: byte counts move with platform and dependency
        // internals.
        for (head, base) in [
            (row.eager_warm, base.eager_warm),
            (row.member1_cold, base.member1_cold),
            (row.member1_warm, base.member1_warm),
            (row.member17_cold, base.member17_cold),
            (row.member17_warm, base.member17_warm),
        ] {
            assert!(head.0 <= base.0, "{name}: {row:?}");
        }
    }
}

/// `d416b1a2`, debug test build, one thread, except the `eager_cold`,
/// `eager_warm` and member rows, re-recorded with the block GEMM kernel, its
/// process-global geometry-plan cache and its executor-kept staging. Every
/// grouped submission takes its plans from the cache, so a warm call builds
/// none, and its staging (job records, geometry index, plan-handle slots,
/// costs, lane tables) from the executor that runs it, so a warm call on a
/// kept executor (this test's eager context, and every member workspace)
/// stages without allocating: `member1_warm` and `member17_warm` are zero
/// (before: 0/0/2/0 and 7/7/31/7) and `eager_warm` drops by the kernel's
/// seven calls. Cold calls pay the cache admission of new geometries (the
/// shared `Arc` and the cache's placeholder) and size the staging, so
/// `member1_cold` moves 10/6/30/6 -> 11/8/22/8 and SU2 transformed
/// `member17_cold` 104 -> 116 calls. The plan bytes moved from
/// `member1_retained` to the shared cache.
const BASE: [(&str, Row); 4] = [
    (
        "U1 transformed",
        Row {
            eager_cold: (547, 201537),
            eager_warm: (16, 2032),
            member1_cold: (11, 5484),
            member1_warm: (0, 0),
            member17_cold: (31, 89248),
            member17_warm: (0, 0),
            member1_retained: 4592,
        },
    ),
    (
        "U1 identity output",
        Row {
            eager_cold: (114, 36016),
            eager_warm: (13, 1312),
            member1_cold: (8, 940),
            member1_warm: (0, 0),
            member17_cold: (13, 8704),
            member17_warm: (0, 0),
            member1_retained: 160,
        },
    ),
    (
        "SU2 transformed",
        Row {
            eager_cold: (22532, 6846533),
            eager_warm: (17, 2096),
            member1_cold: (22, 18732),
            member1_warm: (0, 0),
            member17_cold: (116, 634996),
            member17_warm: (0, 0),
            member1_retained: 16992,
        },
    ),
    (
        "SU2 identity output",
        Row {
            eager_cold: (476, 66500),
            eager_warm: (13, 1312),
            member1_cold: (8, 972),
            member1_warm: (0, 0),
            member17_cold: (13, 9248),
            member17_warm: (0, 0),
            member1_retained: 192,
        },
    ),
];
