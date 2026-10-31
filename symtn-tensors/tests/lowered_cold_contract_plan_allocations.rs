//! Cold lowered enumeration + cold contraction plan-build allocation,
//! measured relatively (no absolute byte pins: platform allocators differ).
//!
//! Two properties are pinned:
//! 1. The genuine cold mechanism (`prepare_fusion_tree_layout_lowered`)
//!    stays streaming — enumeration bytes track the admitted key count, not
//!    the candidate sector product — and builds the layout exactly once
//!    (a committed layout makes the second prepare a cache hit: no
//!    structure cache misses).
//! 2. The tensor-wide fusion route's cold plan build: the plain entry is the
//!    production route and this test pins its cold/warm plan-build
//!    behavior.

use std::sync::Arc;
use symtn_dense::Engine;

use symtn_core::{FusionProductSpace, FusionTreeHomSpace, SectorLeg, U1FusionRule, U1Irrep};
use symtn_tensors::{
    BoundDynamicFusionMapSpace, ContractDestinationInit, FusionOperand, OutputAxisOrder,
    RuleIdentity, TensorContractFusionExecutionContext, TensorContractSpec,
};

use crate::counting_alloc;

fn measured<T>(operation: impl FnOnce() -> T) -> (T, usize, usize) {
    counting_alloc::start();
    let value = operation();
    let allocs = counting_alloc::stop();
    (value, allocs.bytes as usize, allocs.calls as usize)
}

fn reset_all_caches() {
    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();
}

/// Structure-cache misses so far, over every kind.
fn structure_misses() -> u64 {
    symtn_core::StructureCacheKind::ALL
        .into_iter()
        .map(|kind| symtn_core::structure_cache_info(kind).misses())
        .sum()
}

fn chain_homspace(sector_count: i32) -> FusionTreeHomSpace {
    let sectors = (0..sector_count)
        .map(|charge| (U1Irrep::new(charge).sector_id(), 1))
        .collect::<Vec<_>>();
    FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new(sectors.clone(), false)]),
        FusionProductSpace::new([SectorLeg::new(sectors, false)]),
    )
}

/// Purpose: the cold lowered enumeration streams its keys, and a second
/// prepare of a committed layout rebuilds nothing.
#[test]
fn cold_lowered_enumeration_streams_and_builds_once() {
    let _serial = crate::counting_alloc::serial();

    // Streaming: admitted keys grow linearly with the sector count while the
    // candidate space (codomain x domain sector pairs) grows quadratically.
    // Doubling the sectors at most doubles the working set; an eager
    // whole-table materialization would quadruple it.
    let peak_live_bytes = |sector_count: i32| {
        reset_all_caches();
        let homspace = chain_homspace(sector_count);
        let (prepared, allocs) = counting_alloc::measure(|| {
            homspace
                .prepare_fusion_tree_layout_checked(&U1FusionRule)
                .unwrap()
        });
        let keys = prepared.commit();
        assert_eq!(keys.len(), sector_count as usize);
        allocs.peak_live_bytes
    };
    let (small, large) = (peak_live_bytes(32), peak_live_bytes(64));
    assert!(
        large <= 2 * small,
        "cold lowered enumeration grows faster than its keys: peak {small} B \
         at N=32, {large} B at N=64"
    );

    // Builds-once: with the cold layout committed, the second prepare is a
    // cache lookup, not a rebuild.
    reset_all_caches();
    let homspace = chain_homspace(16);
    let misses = structure_misses();
    let cold_keys = homspace
        .prepare_fusion_tree_layout_checked(&U1FusionRule)
        .unwrap()
        .commit();
    assert!(
        structure_misses() > misses,
        "premise: the cold prepare misses"
    );
    let misses = structure_misses();
    let second = homspace
        .prepare_fusion_tree_layout_checked(&U1FusionRule)
        .unwrap();
    assert_eq!(
        structure_misses(),
        misses,
        "second lowered prepare rebuilt the layout"
    );
    let warm_keys = second.commit();
    assert!(Arc::ptr_eq(&cold_keys, &warm_keys));
}

struct ContractFixture {
    dst: BoundDynamicFusionMapSpace<U1FusionRule>,
    lhs: BoundDynamicFusionMapSpace<U1FusionRule>,
    rhs: BoundDynamicFusionMapSpace<U1FusionRule>,
    lhs_data: Vec<f64>,
    rhs_data: Vec<f64>,
}

/// Rank-3 x rank-3 U(1) contraction with a non-identity output order: the
/// tensor-wide fusion route (tree-transform plan build), not the direct-core
/// fast path.
fn contract_fixture() -> ContractFixture {
    let provider = Arc::new(U1FusionRule);
    let leg = |dual| {
        SectorLeg::new(
            [
                (U1Irrep::new(0).sector_id(), 2),
                (U1Irrep::new(1).sector_id(), 2),
            ],
            dual,
        )
    };
    let lhs_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(false), leg(true)]),
        FusionProductSpace::new([leg(false)]),
    );
    let rhs_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(true)]),
        FusionProductSpace::new([leg(true), leg(false)]),
    );
    let lhs = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(
        Arc::clone(&provider),
        lhs_hom,
    )
    .unwrap();
    let rhs = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(
        Arc::clone(&provider),
        rhs_hom,
    )
    .unwrap();
    let dst = BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
        &lhs,
        &rhs,
        &[0],
        &[2],
        OutputAxisOrder::from_axes(&[2, 0, 3, 1]),
    )
    .unwrap();
    let lhs_data = (0..lhs.space().required_len().unwrap())
        .map(|index| index as f64 + 1.0)
        .collect::<Vec<_>>();
    let rhs_data = (0..rhs.space().required_len().unwrap())
        .map(|index| 0.5 * index as f64 - 2.0)
        .collect::<Vec<_>>();
    ContractFixture {
        dst,
        lhs,
        rhs,
        lhs_data,
        rhs_data,
    }
}

struct RouteRun {
    cold_bytes: usize,
    cold_allocations: usize,
    warm_bytes: usize,
    cold_misses: usize,
    cold_hits: usize,
    warm_misses: usize,
    warm_hits: usize,
}

fn run_route(engine: &mut dyn Engine) -> RouteRun {
    reset_all_caches();
    let fixture = contract_fixture();
    let axes = || TensorContractSpec::new(&[0], &[2], OutputAxisOrder::from_axes(&[2, 0, 3, 1]));
    let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
    let mut output = vec![0.0; fixture.dst.space().required_len().unwrap()];
    // The plain entry is the production route and the one this guard pins.
    let mut run =
        |output: &mut [f64],
         context: &mut TensorContractFusionExecutionContext<f64, RuleIdentity>| {
            context
                .tensorcontract_planned_into(
                    engine,
                    &fixture.dst,
                    output,
                    (
                        FusionOperand::direct(fixture.lhs.space()),
                        &fixture.lhs_data,
                    ),
                    (
                        FusionOperand::direct(fixture.rhs.space()),
                        &fixture.rhs_data,
                    ),
                    axes().lhs_contracting_axes(),
                    axes().rhs_contracting_axes(),
                    &[2, 0, 3, 1],
                    1.0,
                    ContractDestinationInit::Axpby(0.0),
                )
                .unwrap();
        };
    let completed = || {
        let info = symtn_core::structure_cache_info(
            symtn_core::StructureCacheKind::CompletedTreeTransformer,
        );
        (info.misses() as usize, info.hits() as usize)
    };
    let before = completed();
    let ((), cold_bytes, cold_allocations) = measured(|| run(&mut output, &mut context));
    let (cold_misses, cold_hits) = completed();
    let mut warm_output = vec![0.0; output.len()];
    let ((), warm_bytes, _) = measured(|| run(&mut warm_output, &mut context));
    assert_eq!(warm_output, output);
    let (warm_misses, warm_hits) = completed();
    RouteRun {
        cold_bytes,
        cold_allocations,
        warm_bytes,
        cold_misses: cold_misses - before.0,
        cold_hits: cold_hits - before.1,
        warm_misses: warm_misses - before.0,
        warm_hits: warm_hits - before.1,
    }
}

#[test]
fn cold_contract_plan_build_stays_cached_after_first_run() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();

    // Discard one run: the first contraction in the process pays a one-time
    // warmup (lazy statics, thread-local init) that would otherwise skew the
    // repeat-stability comparison below.
    let _ = run_route(engine);
    let first = run_route(engine);
    let second = run_route(engine);

    // The fixture must exercise the fusion plan-build route, not the
    // direct-core fast path: a cold run builds and publishes its completed
    // transformers, a warm run hits them without new builds.
    assert!(first.cold_misses >= 3, "fixture took the core fast path");
    assert_eq!(first.warm_misses, first.cold_misses);
    assert!(first.warm_hits > first.cold_hits);
    assert!(
        first.warm_bytes < first.cold_bytes,
        "warm run rebuilt the cold plan: warm {} bytes vs cold {} bytes",
        first.warm_bytes,
        first.cold_bytes
    );

    // Cold plan-build allocation is deterministic run to run: the relative
    // baseline any entry-point change must reproduce.
    assert_eq!(
        (second.cold_bytes, second.cold_allocations),
        (first.cold_bytes, first.cold_allocations),
        "cold plan-build allocation became nondeterministic"
    );
    assert_eq!(second.warm_bytes, first.warm_bytes);
    assert_eq!(second.cold_misses, first.cold_misses);
    assert_eq!(second.warm_hits, first.warm_hits);
}
