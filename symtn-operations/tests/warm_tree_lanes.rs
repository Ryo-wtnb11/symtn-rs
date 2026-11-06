//! A reused workspace's warm tree replays on four lanes, counted on every
//! thread of the process: on a target whose std mutex locks without
//! allocating they allocate nothing; on Apple targets, where each runner
//! lane's fresh std mutex allocates its pthread mutex on first lock (one per
//! lane per call, the disclosed runner residual), repeated warm calls
//! allocate the same and no more, a large call does not make a later small
//! call allocate more, and an overwrite of many high-rank inactive blocks
//! allocates no more than one of few, as does a general-`beta` update of
//! many such blocks in Multi groups.
//!
//! Its own test binary with one test, so the shared counting allocator's
//! process-wide measurement sees only that test. That the fixture really runs on four
//! lanes is asserted in the crate's
//! `the_warm_allocation_fixture_runs_on_four_lanes`, which can read the
//! placement.

#![cfg(all(
    feature = "cpu-faer",
    not(any(
        feature = "cpu-blas",
        feature = "blas-accelerate",
        feature = "blas-openblas",
        feature = "blas-mkl",
        feature = "provider-inject"
    ))
))]

use std::sync::Arc;

use symtn_core::{BlockKey, BlockSpec, BlockStructure};
use symtn_dense::{CpuEngine, Pool};
use symtn_operations::{
    DenseTreeTransformOperations, TreeTransformBackend, TreeTransformBlockSpec,
    TreeTransformStructure, TreeTransformWorkspace,
};
use tenferro_cpu::{CpuBatchPolicy, CpuBatchStrategy};

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

/// `groups` Multi groups of two `[rows, 3]` blocks each (24 elements at
/// eight rows keep every stage's copy estimate nonzero), one joint component
/// per group, plus one Single, recoupled by a non-identity matrix.
fn fixture(
    groups: usize,
    rows: usize,
) -> (Arc<BlockStructure>, TreeTransformStructure<f64>, Vec<f64>) {
    let structure = Arc::new(
        BlockStructure::packed_column_major(2, vec![vec![rows, 3]; 2 * groups + 1]).unwrap(),
    );
    let mut specs = (0..groups)
        .map(|g| {
            TreeTransformBlockSpec::multi(
                vec![2 * g, 2 * g + 1],
                vec![2 * g, 2 * g + 1],
                vec![1.0, 2.0, -0.5, 1.5],
            )
        })
        .collect::<Vec<_>>();
    specs.push(TreeTransformBlockSpec::single(2 * groups, 2 * groups, -2.0));
    let transform =
        TreeTransformStructure::compile_structures(&structure, &structure, &specs).unwrap();
    let source = (0..structure.required_len().unwrap())
        .map(|value| value as f64 * 0.25)
        .collect();
    (structure, transform, source)
}

/// Rank of the high-rank fixtures.
const RANK: usize = 17;
/// Elements of one high-rank block.
const RANK_ELEMENTS: usize = 1 << RANK;

/// Strides `2^(i+1) - 1` of a high-rank block.
fn high_rank_strides() -> Vec<usize> {
    (0..RANK).map(|i| (1usize << (i + 1)) - 1).collect()
}

/// Storage span of one high-rank block.
fn high_rank_span() -> usize {
    1 + high_rank_strides().iter().sum::<usize>()
}

/// `blocks` rank-17 `[2; 17]` blocks with strides `2^(i+1) - 1`: injective,
/// no two axes fuse (nor with a packed column's), so each role keeps rank
/// 17, above the inline traversal rank, and is one indivisible band unit;
/// the blocks fill consecutive disjoint spans. Inactive (spec-free) without
/// `groups`, else paired into Multi groups `(2g, 2g + 1)` with the symmetric
/// recoupling `[[1, 2], [2, -0.5]]`.
fn high_rank_fixture(
    blocks: usize,
    groups: bool,
) -> (Arc<BlockStructure>, TreeTransformStructure<f64>) {
    let strides = high_rank_strides();
    let span = high_rank_span();
    let structure = Arc::new(
        BlockStructure::from_blocks_with_rank(
            RANK,
            (0..blocks)
                .map(|b| {
                    BlockSpec::with_key(
                        BlockKey::ordinal(b),
                        vec![2; RANK],
                        strides.clone(),
                        b * span,
                    )
                    .unwrap()
                })
                .collect(),
        )
        .unwrap(),
    );
    let specs = if groups {
        (0..blocks / 2)
            .map(|g| {
                TreeTransformBlockSpec::multi(
                    vec![2 * g, 2 * g + 1],
                    vec![2 * g, 2 * g + 1],
                    vec![1.0, 2.0, 2.0, -0.5],
                )
            })
            .collect()
    } else {
        Vec::new()
    };
    let transform =
        TreeTransformStructure::compile_structures(&structure, &structure, &specs).unwrap();
    (structure, transform)
}

#[test]
fn warm_multi_lane_tree_replay_allocates_nothing_but_the_lane_locks() {
    let pool = Pool::builder().threads(4).build().unwrap();
    // `Outer` places the components on four one-slot lanes.
    let engine =
        &mut CpuEngine::new(pool).with_batch_policy(CpuBatchPolicy::new(CpuBatchStrategy::Outer));
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();
    let small = fixture(8, 8);
    let large = fixture(16, 16);
    let mut destinations = [
        vec![0.0; small.0.required_len().unwrap()],
        vec![0.0; large.0.required_len().unwrap()],
    ];
    let mut call = |which: usize| {
        let (structure, transform, source) = if which == 0 { &small } else { &large };
        let destination = &mut destinations[which];
        counting_alloc::measure_process(|| {
            backend
                .tree_transform_structure_into_raw(
                    engine,
                    &mut workspace,
                    transform,
                    structure,
                    structure,
                    destination,
                    source,
                    1.0,
                    0.0,
                )
                .unwrap()
        })
        .1
    };
    call(0);
    let warm = call(0);
    call(1);
    let after_large = call(0);
    if cfg!(target_vendor = "apple") {
        assert_eq!(call(0), warm, "warm calls grow");
        assert_eq!(after_large, warm, "a large call grows the small one's");
    } else {
        assert_eq!(warm, 0);
        assert_eq!(after_large, 0);
    }

    // High-rank inactive overwrites: four and eight components on the same
    // four lanes.
    let inactive = [high_rank_fixture(4, false), high_rank_fixture(8, false)];
    let mut buffers = inactive
        .iter()
        .map(|(structure, _)| {
            let len = structure.required_len().unwrap();
            (vec![1.0; len], vec![0.0; len])
        })
        .collect::<Vec<_>>();
    let mut overwrite = |which: usize| {
        let (structure, transform) = &inactive[which];
        let (destination, source) = &mut buffers[which];
        counting_alloc::measure_process(|| {
            backend
                .tree_transform_structure_overwrite_into_raw(
                    engine,
                    &mut workspace,
                    transform,
                    structure,
                    structure,
                    destination,
                    source,
                    1.0,
                    &[],
                )
                .unwrap()
        })
        .1
    };
    overwrite(0);
    overwrite(1);
    let few = overwrite(0);
    let many = overwrite(1);
    if cfg!(target_vendor = "apple") {
        assert_eq!(many, few, "inactive overwrites allocate per component");
    } else {
        assert_eq!(few, 0);
        assert_eq!(many, 0);
    }

    // High-rank Multi groups at `alpha = 1, beta = 2`: four and eight
    // groups (components) on the same four lanes.
    let multi = [high_rank_fixture(8, true), high_rank_fixture(16, true)];
    let span = high_rank_span();
    // Block `b`'s source span holds `b + 1`.
    let sources = multi
        .iter()
        .map(|(structure, _)| {
            (0..structure.required_len().unwrap())
                .map(|at| (at / span + 1) as f64)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut destinations = sources
        .iter()
        .map(|source| vec![1.0; source.len()])
        .collect::<Vec<_>>();
    let mut update = |which: usize, destination: &mut [f64]| {
        let (structure, transform) = &multi[which];
        counting_alloc::measure_process(|| {
            backend
                .tree_transform_structure_into_raw(
                    engine,
                    &mut workspace,
                    transform,
                    structure,
                    structure,
                    destination,
                    &sources[which],
                    1.0,
                    2.0,
                )
                .unwrap()
        })
        .1
    };
    // The first call's values: group blocks `a = 2g`, `b = 2g + 1` become
    // `2 * 1 + (va + 2 vb)` and `2 * 1 + (2 va - 0.5 vb)`; the gaps between
    // blocks keep their one.
    update(0, &mut destinations[0]);
    for (block, chunk) in destinations[0].chunks(span).enumerate() {
        let va = (block - block % 2 + 1) as f64;
        let vb = va + 1.0;
        let expected = if block % 2 == 0 {
            2.0 + va + 2.0 * vb
        } else {
            2.0 + 2.0 * va - 0.5 * vb
        };
        let hits = chunk.iter().filter(|&&value| value == expected).count();
        let gaps = chunk.iter().filter(|&&value| value == 1.0).count();
        assert_eq!(
            (hits, gaps),
            (RANK_ELEMENTS, span - RANK_ELEMENTS),
            "block {block}"
        );
    }
    let (small, large) = destinations.split_at_mut(1);
    update(1, &mut large[0]);
    let few = update(0, &mut small[0]);
    let many = update(1, &mut large[0]);
    if cfg!(target_vendor = "apple") {
        assert_eq!(many, few, "general-beta scatters allocate per column");
    } else {
        assert_eq!(few, 0);
        assert_eq!(many, 0);
    }
}
