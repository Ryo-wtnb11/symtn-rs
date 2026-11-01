//! Allocation contracts of the Host Core and CopyC routes,
//! eager (`contract_into`) and `ContractPlan` members (B = 1, 17), on one
//! thread:
//!
//! - a warm member call allocates nothing, at B = 1 and B = 17;
//! - a warm eager call and a warm member call build no GEMM plan (the
//!   geometry-plan cache's misses do not move);
//! - a warm eager call and a new workspace's cold call cost per call, not per
//!   block: a larger structure of the same family (more sectors, so more
//!   blocks) adds fewer allocation calls than it adds blocks;
//! - dropping a warmed workspace frees no more than its `retained_bytes`;
//! - on a four-thread engine, a second identical eager contraction (a fresh
//!   dense executor) takes its lane allocation from the process-global
//!   lane-allocation cache instead of searching again.
//!
//! Exact counts are printed, not pinned: they move with platform and
//! dependency internals. The blocks stay small enough that the dense
//! provider's GEMM needs no packing buffer of its own (faer allocates one
//! per large GEMM, which no workspace of symtn can own).

use std::sync::Arc;

use symtn::cache::StructureCacheKind;
use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{
    ContractPlan, ContractSpec, CpuEngine, GradedSpace, Pool, Runtime, StackedTensorMap, TensorMap,
};

use crate::counting_alloc;

fn calls(f: impl FnOnce()) -> u64 {
    counting_alloc::measure(f).1.calls
}

fn plan_builds() -> u64 {
    symtn::cache::stats()
        .into_iter()
        .find(|info| info.kind() == StructureCacheKind::GemmPlans)
        .unwrap()
        .misses()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Row {
    /// Coupled-sector blocks of the output (one core GEMM each), the size
    /// the scaling check compares.
    blocks: usize,
    eager_warm: u64,
    member1_cold: u64,
    member1_warm: u64,
    member17_cold: u64,
    member17_warm: u64,
    /// GEMM plans built by the warm eager and warm member calls.
    warm_plan_builds: u64,
    /// `retained_bytes` and the bytes its drop frees, at B = 1 and 17.
    retained: [usize; 2],
    freed: [u64; 2],
}

/// `v` has U(1) charges `-reach..=reach` (degeneracies 4, 3, 4 on the
/// charges up to one, 1 beyond) and the contracted `w` charges
/// `1 - reach..=reach` (degeneracy 4 on charge 0, 2 on 1, 1 beyond), so the
/// core's GEMM count grows with `reach`.
fn row(codomain: &[usize], reach: i32) -> Row {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let rule = Arc::new(U1FusionRule);
    let v = GradedSpace::try_new(
        Arc::clone(&rule),
        (-reach..=reach).map(|charge| {
            let degeneracy = match charge.abs() {
                0 => 3,
                1 => 4,
                _ => 1,
            };
            (U1Irrep::new(charge), degeneracy)
        }),
    )
    .unwrap();
    let w = GradedSpace::try_new(
        rule,
        (1 - reach..=reach).map(|charge| {
            let degeneracy = match charge {
                0 => 4,
                1 => 2,
                _ => 1,
            };
            (U1Irrep::new(charge), degeneracy)
        }),
    )
    .unwrap();
    let a = TensorMap::<_, f64>::rand_with_seed(engine, &runtime, [&v, &v], [&w], 5).unwrap();
    let b = TensorMap::<_, f64>::rand_with_seed(engine, &runtime, [&w], [&v], 6).unwrap();
    let spec = ContractSpec {
        lhs: &[2],
        rhs: &[0],
        codomain,
        domain: &[2],
    };
    let mut out = a.contract(engine, &b, &spec).unwrap();
    a.contract_into(engine, &b, &spec, &mut out, 1.0, 0.0)
        .unwrap();
    let mut builds = plan_builds();
    let eager_warm = calls(|| {
        a.contract_into(engine, &b, &spec, &mut out, 1.0, 0.0)
            .unwrap()
    });
    let mut warm_plan_builds = plan_builds() - builds;

    let mut member = |members: usize| {
        let lhs = StackedTensorMap::pack(&vec![&a; members]).unwrap();
        let rhs = StackedTensorMap::pack(&vec![&b; members]).unwrap();
        let plan = ContractPlan::new(engine, &lhs, &rhs, &spec).unwrap();
        // A first workspace admits the plans, so the cold call below
        // measures only what a new workspace allocates for itself.
        let mut workspace = plan.workspace().unwrap();
        plan.execute(engine, &lhs, &rhs, &mut workspace).unwrap();
        let mut dst = workspace.take_output().unwrap();
        let mut workspace = plan.workspace().unwrap();
        let cold = calls(|| {
            plan.execute_into(engine, &lhs, &rhs, &mut dst, &mut workspace)
                .unwrap()
        });
        builds = plan_builds();
        let warm = calls(|| {
            plan.execute_into(engine, &lhs, &rhs, &mut dst, &mut workspace)
                .unwrap()
        });
        warm_plan_builds += plan_builds() - builds;
        // Dropped while the plan, which shares its replay with the
        // workspace, is alive: the plan's own data is not the workspace's.
        let retained = workspace.retained_bytes();
        let ((), dropped) = counting_alloc::measure(|| drop(workspace));
        (cold, warm, retained, dropped.freed_bytes)
    };
    let (member1_cold, member1_warm, retained, freed) = member(1);
    let (member17_cold, member17_warm, retained17, freed17) = member(17);
    Row {
        blocks: out.blocks().unwrap().len(),
        eager_warm,
        member1_cold,
        member1_warm,
        member17_cold,
        member17_warm,
        warm_plan_builds,
        retained: [retained, retained17],
        freed: [freed, freed17],
    }
}

#[test]
#[cfg_attr(
    any(
        feature = "cpu-blas",
        feature = "blas-accelerate",
        feature = "blas-openblas",
        feature = "blas-mkl",
        feature = "provider-inject"
    ),
    ignore = "the warm-zero and no-plan-build contracts are the native grouped GEMM kernel's; a BLAS provider allocates its own per-call scratch, so warm prepared replays allocate"
)]
fn core_and_copy_c_route_allocations_follow_their_contracts() {
    let _serial = crate::counting_alloc::serial();
    for (name, codomain) in [("core", [0, 1]), ("copyC", [1, 0])] {
        let (small, large) = (row(&codomain, 1), row(&codomain, 6));
        eprintln!("{name}: small {small:?}\n{name}: large {large:?}");
        assert!(large.blocks > small.blocks, "{name}: premise");
        for row in [small, large] {
            // What: warm member calls allocate nothing; no warm call builds
            // a GEMM plan; a dropped workspace frees what it charges.
            assert_eq!(
                (row.member1_warm, row.member17_warm),
                (0, 0),
                "{name}: {row:?}"
            );
            assert_eq!(row.warm_plan_builds, 0, "{name}: {row:?}");
            for (freed, retained) in row.freed.into_iter().zip(row.retained) {
                assert!(freed as usize <= retained, "{name}: {row:?}");
            }
        }
        // What: the warm eager call and a workspace's cold calls cost per
        // call, not per block (amortized growth of reused buffers may add a
        // few calls).
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

fn lane_allocations() -> (u64, u64) {
    let info = symtn::cache::stats()
        .into_iter()
        .find(|info| info.kind() == StructureCacheKind::LaneAllocations)
        .unwrap();
    (info.hits(), info.misses())
}

#[test]
#[cfg_attr(
    any(
        feature = "cpu-blas",
        feature = "blas-accelerate",
        feature = "blas-openblas",
        feature = "blas-mkl",
        feature = "provider-inject"
    ),
    ignore = "the lane search is the native grouped GEMM kernel's; a BLAS provider runs its own batch"
)]
fn warm_eager_contract_takes_its_lane_allocation_from_the_cache() {
    let _serial = crate::counting_alloc::serial();
    let engine = &mut CpuEngine::new(Pool::builder().threads(4).build().unwrap());
    let runtime = Runtime::builder().build().unwrap();
    let rule = Arc::new(U1FusionRule);
    // Unequal blocks large enough that the engine grants the batch several
    // threads, so the kernel searches its lanes.
    let v = GradedSpace::try_new(
        rule,
        (-3..=3).map(|charge: i32| {
            (
                U1Irrep::new(charge),
                48 - 8 * charge.unsigned_abs() as usize,
            )
        }),
    )
    .unwrap();
    let a = TensorMap::<_, f64>::rand_with_seed(engine, &runtime, [&v], [&v], 5).unwrap();
    let b = TensorMap::<_, f64>::rand_with_seed(engine, &runtime, [&v], [&v], 6).unwrap();
    let spec = ContractSpec {
        lhs: &[1],
        rhs: &[0],
        codomain: &[0],
        domain: &[1],
    };
    let cold = a.contract(engine, &b, &spec).unwrap();
    let (hits, misses) = lane_allocations();
    let warm = a.contract(engine, &b, &spec).unwrap();
    let (warm_hits, warm_misses) = lane_allocations();
    // What: the warm call is a hit and searches nothing new; the result is
    // the cold call's.
    assert!(warm_hits > hits, "{hits} -> {warm_hits}");
    assert_eq!(warm_misses, misses);
    assert_eq!(warm.dense_data().unwrap(), cold.dense_data().unwrap());
}
