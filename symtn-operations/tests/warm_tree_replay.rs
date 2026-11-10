//! A reused workspace's warm tree replays on four workers, counted on every
//! thread of the process: on a target whose std mutex locks without
//! allocating they allocate nothing; on Apple targets, where a fresh std
//! mutex allocates its pthread mutex on first lock (the native provider's
//! grouped GEMM lanes take such locks), repeated warm calls allocate the
//! same and no more, a large call does not make a later small call allocate
//! more, and an overwrite of many high-rank inactive blocks allocates no
//! more than one of few, as does a general-`beta` update of many such
//! blocks in Multi groups. A custom executor, under `Auto` since it honors
//! no forced policy, keeps the same contract, as do member batches of
//! several members, whose moves split into contiguous member ranges on the
//! four workers around one dense submission (including a five-member batch
//! over four ranges, one of them two members wide).
//!
//! Its own test binary with one test, so the shared counting allocator's
//! process-wide measurement sees only that test.

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
use symtn_dense::{
    CpuEngine, DefaultDenseExecutor, DenseDotConfig, DenseError, DenseExecutor, DenseRead,
    DenseScalar, DenseTensor, DenseWrite, Engine,
};
use symtn_operations::{
    tree_transform_members_overwrite_raw, DenseTreeTransformOperations, StridedHostKernelAdapter,
    TreeTransformBackend, TreeTransformBlockSpec, TreeTransformStructure, TreeTransformWorkspace,
};
use tenferro_cpu::{CpuBatchPolicy, CpuBatchStrategy};

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;
#[path = "../../tests/support/started_pool.rs"]
mod started_pool;
#[path = "support/warm_tree_replay.rs"]
mod warm_tree_replay;

use warm_tree_replay::fixture;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

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
/// 17, above the inline traversal rank;
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

/// A custom executor forwarding every call to the built-in executor, on
/// the default grouped GEMM entry.
#[derive(Default)]
struct Forwarding(DefaultDenseExecutor);

impl DenseExecutor for Forwarding {
    fn svd(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        self.0.svd(engine, input)
    }

    fn qr(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        self.0.qr(engine, input)
    }

    fn eigh(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        self.0.eigh(engine, input)
    }

    fn dot_general_into(
        &mut self,
        engine: &mut dyn Engine,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        config: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        self.0.dot_general_into(engine, output, lhs, rhs, config)
    }

    fn matmul_axpby_into(
        &mut self,
        engine: &mut dyn Engine,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        alpha: DenseScalar,
        beta: DenseScalar,
    ) -> Result<(), DenseError> {
        self.0
            .matmul_axpby_into(engine, output, lhs, rhs, alpha, beta)
    }
}

#[test]
fn warm_parallel_tree_replay_allocates_nothing_but_provider_locks() {
    // Every worker has started, so no worker start-up lands in a window.
    let pool = started_pool::started_pool(warm_tree_replay::THREADS);
    // `Outer` admits the replay on all four workers.
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

    // The custom executor honors no forced policy, so it replays under
    // `Auto` on the same four workers: small, large, small.
    let auto = &mut CpuEngine::new(started_pool::started_pool(warm_tree_replay::THREADS));
    let mut custom = DenseTreeTransformOperations::new(Forwarding::default());
    let mut workspace = TreeTransformWorkspace::default();
    let mut destinations = [
        vec![0.0; small.0.required_len().unwrap()],
        vec![0.0; large.0.required_len().unwrap()],
    ];
    let mut call = |which: usize| {
        let (structure, transform, source) = if which == 0 { &small } else { &large };
        let destination = &mut destinations[which];
        counting_alloc::measure_process(|| {
            custom
                .tree_transform_structure_into_raw(
                    auto,
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
        assert_eq!(call(0), warm, "warm custom-executor calls grow");
        assert_eq!(
            after_large, warm,
            "a large custom-executor call grows the small one's"
        );
    } else {
        assert_eq!(warm, 0);
        assert_eq!(after_large, 0);
    }

    // Member batches on the same four workers: three members of the small
    // fixture, two of the large, three of the small; then five of the small,
    // which split into four member ranges.
    let mut workspace = TreeTransformWorkspace::default();
    let mut dense = DefaultDenseExecutor::new();
    let mut kernels = StridedHostKernelAdapter::default();
    let batches = [(&small, 3usize), (&large, 2), (&small, 5)];
    let mut buffers = batches
        .iter()
        .map(|&((structure, _, source), members)| {
            let source = source.repeat(members);
            (
                vec![0.0; members * structure.required_len().unwrap()],
                source,
            )
        })
        .collect::<Vec<_>>();
    let mut members = |which: usize| {
        let ((structure, transform, _), members) = batches[which];
        let (destination, source) = &mut buffers[which];
        counting_alloc::measure_process(|| {
            tree_transform_members_overwrite_raw(
                engine,
                &mut kernels,
                &mut dense,
                &mut workspace,
                transform,
                structure,
                structure,
                destination,
                source,
                members,
                &[],
            )
            .unwrap()
        })
        .1
    };
    members(0);
    let warm = members(0);
    members(1);
    let after_large = members(0);
    if cfg!(target_vendor = "apple") {
        assert_eq!(members(0), warm, "warm member calls grow");
        assert_eq!(
            after_large, warm,
            "a large member call grows the small one's"
        );
    } else {
        assert_eq!(warm, 0);
        assert_eq!(after_large, 0);
    }
    members(2);
    let warm = members(2);
    if cfg!(target_vendor = "apple") {
        assert_eq!(members(2), warm, "warm uneven member-range calls grow");
    } else {
        assert_eq!(warm, 0);
    }

    // High-rank inactive overwrites: four and eight blocks on the same four
    // workers.
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
        assert_eq!(many, few, "inactive overwrites allocate per block");
    } else {
        assert_eq!(few, 0);
        assert_eq!(many, 0);
    }

    // High-rank Multi groups at `alpha = 1, beta = 2`: four and eight
    // groups on the same four workers.
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
