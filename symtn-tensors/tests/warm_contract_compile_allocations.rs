//! A warm eager contraction compiles its resolution without heap
//! allocations for rank-sized data.

use std::sync::Arc;
use symtn_dense::Engine;

use symtn_core::{FusionProductSpace, FusionTreeHomSpace, SectorLeg, U1FusionRule, U1Irrep};
use symtn_tensors::{
    prepare_tensorcontract_fusion_plan_dyn, try_compile_storage_contract_core_route,
    BoundDynamicFusionMapSpace, DirectCoreExecutor, FusionOperand, OutputAxisOrder, RuleIdentity,
    StorageContractResolution, TensorContractFusionExecutionContext, TensorContractSpec,
};

use crate::counting_alloc;

/// Allocations of one warm call, including the drop of its result.
fn warm_allocations<T>(mut call: impl FnMut() -> T) -> usize {
    drop(call());
    drop(call());
    counting_alloc::start();
    drop(call());
    let allocs = counting_alloc::stop();
    allocs.calls as usize
}

type Space = BoundDynamicFusionMapSpace<U1FusionRule>;

fn leg(dual: bool) -> SectorLeg {
    SectorLeg::new(
        [
            (U1Irrep::new(-1).sector_id(), 2),
            (U1Irrep::new(0).sector_id(), 3),
            (U1Irrep::new(1).sector_id(), 2),
        ],
        dual,
    )
}

/// Same sectors with a degeneracy profile the `q -> -q` dual does not fix, so
/// the dual leg's map differs from the source leg's.
fn asymmetric_leg(dual: bool) -> SectorLeg {
    SectorLeg::new(
        [
            (U1Irrep::new(-1).sector_id(), 2),
            (U1Irrep::new(0).sector_id(), 3),
            (U1Irrep::new(1).sector_id(), 4),
        ],
        dual,
    )
}

fn space(provider: &Arc<U1FusionRule>, codomain: usize, domain: usize) -> Space {
    space_of(leg, provider, codomain, domain)
}

fn space_of(
    leg: fn(bool) -> SectorLeg,
    provider: &Arc<U1FusionRule>,
    codomain: usize,
    domain: usize,
) -> Space {
    Space::from_final_homspace_multiplicity_free_checked(
        Arc::clone(provider),
        FusionTreeHomSpace::new(
            FusionProductSpace::new((0..codomain).map(|_| leg(false))),
            FusionProductSpace::new((0..domain).map(|_| leg(false))),
        ),
    )
    .unwrap()
}

/// Completed transformers and composed coefficients both come from the
/// process-global caches, so every context is configured alike.
fn runtime_like_context() -> TensorContractFusionExecutionContext<f64, RuleIdentity> {
    TensorContractFusionExecutionContext::<f64, RuleIdentity>::default()
}

fn completed_transformers() -> symtn_core::StructureCacheInfo {
    symtn_core::structure_cache_info(symtn_core::StructureCacheKind::CompletedTreeTransformer)
}

fn coefficient_groups() -> symtn_core::StructureCacheInfo {
    symtn_core::structure_cache_info(symtn_core::StructureCacheKind::TreeTransformCoefficients)
}

/// The storage ladder these rows pin: the lock-free canonical core, then the
/// `DynamicTree` artifact (the planner's `CopyC` rung is not measured here).
fn storage_ladder(
    engine: &mut dyn Engine,
    context: &mut TensorContractFusionExecutionContext<f64, RuleIdentity>,
    dst: &Space,
    lhs: FusionOperand<'_>,
    rhs: FusionOperand<'_>,
    axes: TensorContractSpec<'_>,
) -> StorageContractResolution<f64> {
    match try_compile_storage_contract_core_route::<DirectCoreExecutor, _>(dst, lhs, rhs, axes)
        .unwrap()
        .hit()
    {
        Some(core) => core,
        None => context
            .compile_storage_contract_dynamic_tree(engine, dst, lhs, rhs, axes)
            .unwrap(),
    }
}

/// `A(V^codomain ← V^domain)` composed with `S(V^domain ← V^domain)` in
/// three ways: the canonical Core route, the same contraction with its two
/// leading open axes swapped (a DynamicTree route whose source and output
/// transforms keep every leg on its side), and the plan alone.
fn warm_compile_allocations(engine: &mut dyn Engine, codomain: usize, domain: usize) -> [usize; 3] {
    let provider = Arc::new(U1FusionRule);
    let lhs = space(&provider, codomain, domain);
    let square = space(&provider, domain, domain);
    let lhs_domain = (codomain..codomain + domain).collect::<Vec<_>>();
    let square_codomain = (0..domain).collect::<Vec<_>>();
    let composed = Space::contracted_multiplicity_free_ordered(
        &lhs,
        &square,
        &lhs_domain,
        &square_codomain,
        OutputAxisOrder::identity(),
    )
    .unwrap();
    let core = warm_allocations(|| {
        try_compile_storage_contract_core_route::<DirectCoreExecutor, _>(
            &composed,
            FusionOperand::direct(lhs.space()),
            FusionOperand::direct(square.space()),
            TensorContractSpec::with_default_output_order(&lhs_domain, &square_codomain),
        )
        .unwrap()
        .hit()
        .expect("canonical composition takes the core route")
    });

    let mut swapped = (0..codomain + domain).collect::<Vec<_>>();
    swapped.swap(0, 1);
    let axes = || {
        TensorContractSpec::new(
            &lhs_domain,
            &square_codomain,
            OutputAxisOrder::from_axes(&swapped),
        )
    };
    let dst = Space::contracted_multiplicity_free_ordered(
        &lhs,
        &square,
        &lhs_domain,
        &square_codomain,
        OutputAxisOrder::from_axes(&swapped),
    )
    .unwrap();
    let mut context = runtime_like_context();
    let dynamic_tree = warm_allocations(|| {
        let resolution = storage_ladder(
            engine,
            &mut context,
            &dst,
            FusionOperand::direct(lhs.space()),
            FusionOperand::direct(square.space()),
            axes(),
        );
        assert!(resolution.is_dynamic_tree());
        resolution
    });
    let plan = warm_allocations(|| {
        prepare_tensorcontract_fusion_plan_dyn(
            &dst,
            FusionOperand::direct(lhs.space()),
            FusionOperand::direct(square.space()),
            &lhs_domain,
            &square_codomain,
            &swapped,
        )
        .unwrap()
    });
    [core, dynamic_tree, plan]
}

#[test]
fn warm_contract_compile_allocations_do_not_scale_with_rank() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let shapes = [(2, 1), (2, 2), (3, 2), (3, 3), (4, 3)];
    let counts =
        shapes.map(|(codomain, domain)| warm_compile_allocations(engine, codomain, domain));
    // What: the same allocation count of each route at every rank from 3 to
    // 7: per-call objects (plans, `Arc`s, permuted HomSpaces, the artifact),
    // never rank-sized buffers. An empty twist list is an empty scale list
    // and allocates nothing; the layout lookup key borrows the
    // HomSpace content.
    assert!(
        counts.iter().all(|count| *count == counts[0]),
        "[core, dynamic tree, plan] per rank 3..=7: {counts:?}"
    );
}

/// Allocations of one warm DynamicTree compile of
/// `A(V^codomain ← V^domain)` contracted on its codomain axis 0 with
/// `M(V ← V)` on its domain axis, identity output:
/// A's source transform sends leg 0 to the domain and its `domain` legs to
/// the codomain, and M's transform swaps its two legs.
fn crossing_compile_allocations(
    engine: &mut dyn Engine,
    leg: fn(bool) -> SectorLeg,
    codomain: usize,
    domain: usize,
) -> usize {
    let provider = Arc::new(U1FusionRule);
    let lhs = space_of(leg, &provider, codomain, domain);
    let matrix = space_of(leg, &provider, 1, 1);
    let open = (0..codomain + domain).collect::<Vec<_>>();
    let axes = || TensorContractSpec::new(&[0], &[1], OutputAxisOrder::from_axes(&open));
    let dst = Space::contracted_multiplicity_free_ordered(
        &lhs,
        &matrix,
        &[0],
        &[1],
        OutputAxisOrder::from_axes(&open),
    )
    .unwrap();
    let mut context = runtime_like_context();
    warm_allocations(|| {
        storage_ladder(
            engine,
            &mut context,
            &dst,
            FusionOperand::direct(lhs.space()),
            FusionOperand::direct(matrix.space()),
            axes(),
        )
    })
}

/// Shapes whose source transforms move `(1 + domain) + 2` legs across sides
/// while forming permuted HomSpaces, from rank 3 to 6. Rank 2 is left out:
/// there the planner takes the reversed orientation, whose source transforms
/// keep every leg on its side; shape (1, 2) forms no permuted HomSpace with a
/// crossing leg.
const CROSSING_SHAPES: [(usize, usize); 5] = [(2, 1), (2, 2), (3, 2), (2, 3), (3, 3)];

#[test]
fn warm_contract_compile_allocates_nothing_per_leg_that_changes_side() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let counts = CROSSING_SHAPES
        .map(|(codomain, domain)| crossing_compile_allocations(engine, leg, codomain, domain));
    // What: the rank-independent DynamicTree compile, and nothing per leg
    // that changes side in a source transform. There is no per-leg
    // allocation: `SectorLeg::dual` shares the source leg's sector data, as
    // TensorKit's `dual(V)` shares `V.dims`.
    assert!(
        counts.iter().all(|count| *count == counts[0]),
        "{CROSSING_SHAPES:?}: {counts:?}"
    );
}

/// A leg whose sector -> degeneracy map the dual moves shares its
/// storage too, so it allocates nothing per crossing leg either.
#[test]
fn warm_contract_compile_allocates_nothing_per_crossing_leg_the_dual_moves() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    for (codomain, domain) in [(1, 2)].into_iter().chain(CROSSING_SHAPES) {
        let moved = crossing_compile_allocations(engine, asymmetric_leg, codomain, domain);
        let fixed = crossing_compile_allocations(engine, leg, codomain, domain);
        // What: the same count as legs the dual fixes.
        assert_eq!(moved, fixed, "rank {}", codomain + domain);
    }
}

/// Allocations of one warm `plan_contract` of `A(V^rank ← V^rank)` with
/// `B(V^rank ← V^rank)` on A's domain and B's codomain, both reversed, into
/// each operand's open legs reversed: TensorKit's `copyC` shape (C1p at rank
/// 2), so the planner builds the temporary space, its core and one permute.
/// Paired with the same contraction through the core-then-`DynamicTree`
/// ladder.
fn copy_c_plan_allocations(engine: &mut dyn Engine, rank: usize) -> [usize; 2] {
    let provider = Arc::new(U1FusionRule);
    let lhs = space(&provider, rank, rank);
    let rhs = space(&provider, rank, rank);
    let lhs_axes = (rank..2 * rank).rev().collect::<Vec<_>>();
    let rhs_axes = (0..rank).rev().collect::<Vec<_>>();
    let output = (0..rank)
        .rev()
        .chain((rank..2 * rank).rev())
        .collect::<Vec<_>>();
    let dst = Space::contracted_multiplicity_free_partitioned(
        &lhs,
        &rhs,
        &lhs_axes,
        &rhs_axes,
        OutputAxisOrder::from_axes(&output),
        rank,
    )
    .unwrap();
    let mut context = runtime_like_context();
    let rejections = completed_transformers().rejections();
    let group_rejections = coefficient_groups().rejections();
    let copy_c = warm_allocations(|| {
        let resolution = context
            .plan_contract::<DirectCoreExecutor, _>(
                engine,
                &dst,
                FusionOperand::direct(lhs.space()),
                FusionOperand::direct(rhs.space()),
                &lhs_axes,
                &rhs_axes,
                &output,
            )
            .unwrap();
        assert!(resolution.copy_c().is_some(), "rank {rank} takes CopyC");
        resolution
    });
    let dynamic_tree = warm_allocations(|| {
        let resolution = storage_ladder(
            engine,
            &mut context,
            &dst,
            FusionOperand::direct(lhs.space()),
            FusionOperand::direct(rhs.space()),
            TensorContractSpec::new(&lhs_axes, &rhs_axes, OutputAxisOrder::from_axes(&output)),
        );
        assert!(resolution.is_dynamic_tree(), "rank {rank}");
        resolution
    });
    // Every structure this fixture builds is retained; the old 8 MiB
    // per-entry limit bypassed the rank-5 transform and recompiled it per call.
    assert_eq!(
        coefficient_groups().rejections(),
        group_rejections,
        "rank {rank}: a coefficient group was rejected"
    );
    assert_eq!(
        completed_transformers().rejections(),
        rejections,
        "rank {rank}: a completed transformer was rejected"
    );
    [copy_c, dynamic_tree]
}

#[test]
fn warm_copy_c_planning_is_bounded_and_no_costlier_than_the_dynamic_tree() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // Rank 6 is the largest case: its complete HomSpace, transform
    // and fusion-tree layouts must fit the default budgets and their
    // largest-entry limits, or each warm call would rebuild them.
    for rank in [2, 3, 4, 5, 6] {
        let complete_bypasses =
            symtn_core::structure_cache_info(symtn_core::StructureCacheKind::DegeneracyStructure)
                .rejections();
        let layout_bypasses =
            symtn_core::structure_cache_info(symtn_core::StructureCacheKind::SectorStructure)
                .rejections();
        let [copy_c, dynamic_tree] = copy_c_plan_allocations(engine, rank);
        eprintln!("rank {rank}: warm CopyC plan {copy_c}, DynamicTree ladder {dynamic_tree}");
        // What: device-eager CopyC planning stays a small bounded count once
        // its structures are warm (9-10 through rank 4, 18 at ranks 5 and 6
        // where rank-sized inline axis vectors spill), and never above the
        // DynamicTree compile of the same contraction.
        assert!(copy_c <= 20, "rank {rank}: {copy_c}");
        assert!(
            copy_c <= dynamic_tree,
            "rank {rank}: {copy_c} > {dynamic_tree}"
        );
        assert_eq!(
            symtn_core::structure_cache_info(symtn_core::StructureCacheKind::DegeneracyStructure)
                .rejections(),
            complete_bypasses,
            "rank {rank}: a complete HomSpace structure bypassed the cache"
        );
        assert_eq!(
            symtn_core::structure_cache_info(symtn_core::StructureCacheKind::SectorStructure)
                .rejections(),
            layout_bypasses,
            "rank {rank}: a fusion-tree layout bypassed the cache"
        );
    }
}

#[test]
fn rank_six_complete_owner_reuses_content_across_derivations() {
    let _serial = crate::counting_alloc::serial();
    // What: the U(1) `V^6 <- V^6` content (73,789 blocks) fits the
    // complete owner, so independent ordinary derivations share its content.
    // Expert from_parts construction is deliberately fresh.
    let provider = Arc::new(U1FusionRule);
    let first = space(&provider, 6, 6);
    let cache = symtn_core::StructureCacheKind::DegeneracyStructure;
    let bypasses = symtn_core::structure_cache_info(cache).rejections();
    let second = space(&provider, 6, 6);
    assert!(Arc::ptr_eq(
        first.space().structure(),
        second.space().structure()
    ));
    assert_eq!(
        symtn_core::structure_cache_info(cache).rejections(),
        bypasses
    );
}

/// Purpose: the rank-6 CopyC and DynamicTree working set stays resident in
/// the completed-transformer cache when the two routes alternate.
#[test]
fn rank_six_copy_c_and_dynamic_tree_transforms_stay_co_resident() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: the rank-6 C1p working set — the
    // CopyC output transform and the DynamicTree route's transforms — fits
    // the default process-global completed-transformer cache, so
    // alternating the two routes evicts nothing and every warm call reuses
    // its retained transformers instead of recompiling the DynamicTree route
    // (8.26 M allocations) after each 65 MB transform evicts the other.
    let rank = 6;
    let provider = Arc::new(U1FusionRule);
    let lhs = space(&provider, rank, rank);
    let rhs = space(&provider, rank, rank);
    let lhs_axes = (rank..2 * rank).rev().collect::<Vec<_>>();
    let rhs_axes = (0..rank).rev().collect::<Vec<_>>();
    let output = (0..rank)
        .rev()
        .chain((rank..2 * rank).rev())
        .collect::<Vec<_>>();
    let dst = Space::contracted_multiplicity_free_partitioned(
        &lhs,
        &rhs,
        &lhs_axes,
        &rhs_axes,
        OutputAxisOrder::from_axes(&output),
        rank,
    )
    .unwrap();
    let mut context = runtime_like_context();
    let mut run = |copy_c: bool| -> u64 {
        counting_alloc::start();
        let resolution = if copy_c {
            context
                .plan_contract::<DirectCoreExecutor, _>(
                    engine,
                    &dst,
                    FusionOperand::direct(lhs.space()),
                    FusionOperand::direct(rhs.space()),
                    &lhs_axes,
                    &rhs_axes,
                    &output,
                )
                .unwrap()
        } else {
            storage_ladder(
                engine,
                &mut context,
                &dst,
                FusionOperand::direct(lhs.space()),
                FusionOperand::direct(rhs.space()),
                TensorContractSpec::new(&lhs_axes, &rhs_axes, OutputAxisOrder::from_axes(&output)),
            )
        };
        assert_eq!(resolution.copy_c().is_some(), copy_c);
        assert_eq!(resolution.is_dynamic_tree(), !copy_c);
        drop(resolution);
        counting_alloc::stop().calls
    };
    // A clean cache: the working set alone decides residency.
    symtn_core::clear_structure_caches();
    run(true);
    run(false);
    let warm = completed_transformers();
    let warm_calls = (0..4).map(|step| run(step % 2 == 0)).collect::<Vec<_>>();
    let after = completed_transformers();
    eprintln!("rank-6 working set: {warm:?}");

    assert_eq!(warm.rejections(), 0, "{warm:?}");
    assert_eq!(warm.evictions(), 0, "{warm:?}");
    assert_eq!(after.entries(), warm.entries(), "{after:?}");
    assert_eq!(after.evictions(), 0, "{after:?}");
    assert_eq!(after.misses(), warm.misses(), "no warm call recompiles");
    // No eviction thrash: each route's second warm call costs what its first
    // did, so neither recompiled after the other ran.
    counting_alloc::assert_same_calls("CopyC warm calls", warm_calls[0], warm_calls[2]);
    counting_alloc::assert_same_calls("DynamicTree warm calls", warm_calls[1], warm_calls[3]);
}
