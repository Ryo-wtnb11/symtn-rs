//! Warm-replay microbench: a U(1) many-charge (21-sector),
//! degeneracy-2 rank-4 permute is the regime where per-call `fuse_pair_layout`
//! recomputation dominated. Baking the fused layout
//! per (block, role) at compile time removes that recompute from replay.
//!
//! Each iteration replays 21 Single-block permuted copies through the strided
//! host adapter with a reused workspace — the warm hot path. Distinct one-leg
//! sector ids stand in for U(1) charges; the fused-layout normalization is
//! group-agnostic (it sees only shapes and strides), so this exercises the
//! exact baked vs. recomputed dispatch the SU(2)/U(1) tensor paths hit.
//!
//! No timing assertion here: shared CI runners are too noisy for a pass/fail
//! latency gate (same rationale as benches/strided_combine.rs). The gate is
//! this bench compiling and running; the numbers are read off a quiet machine.

use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use symtn_core::{BlockKey, BlockSpec, BlockStructure};
use symtn_operations::{
    tree_transform_structure_overwrite_with_strided_kernel_raw, StridedHostKernelAdapter,
    TreeTransformBlockSpec, TreeTransformStructure, TreeTransformWorkspace,
};

const CHARGES: usize = 21;
const RANK: usize = 4;
// Degeneracy-2 rank-4 block: 16 elements each.
const BLOCK_DIMS: [usize; RANK] = [2, 2, 2, 2];
const BLOCK_LEN: usize = 16;

fn build_structure() -> Arc<BlockStructure> {
    // Column-major strides within each block; blocks packed back-to-back.
    let strides = [1usize, 2, 4, 8];
    let blocks = (0..CHARGES)
        .map(|charge| {
            BlockSpec::with_key(
                BlockKey::ordinal(charge),
                BLOCK_DIMS.to_vec(),
                strides.to_vec(),
                charge * BLOCK_LEN,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    Arc::new(BlockStructure::from_blocks_with_rank(RANK, blocks).unwrap())
}

fn bench_permute_u1_many(c: &mut Criterion) {
    let structure = build_structure();
    // Non-fully-fusable transpose so the fused layout stays rank>1 (the case
    // where recompute cost is real), matching the deg2 c21 permute regime.
    let specs = (0..CHARGES)
        .map(|block| {
            TreeTransformBlockSpec::single(block, block, 1.0_f64).with_source_axes([1, 0, 3, 2])
        })
        .collect::<Vec<_>>();
    let transform =
        TreeTransformStructure::compile_structures(&structure, &structure, &specs).unwrap();

    let len = CHARGES * BLOCK_LEN;
    let src = (0..len).map(|i| i as f64).collect::<Vec<_>>();
    let mut dst = vec![0.0_f64; len];
    let mut kernels = StridedHostKernelAdapter::default();
    let mut workspace = TreeTransformWorkspace::default();

    c.bench_function("permute_u1_many_deg2_c21", |bencher| {
        bencher.iter(|| {
            tree_transform_structure_overwrite_with_strided_kernel_raw(
                strided_kernel::Exec::serial(),
                &mut kernels,
                &mut workspace,
                &transform,
                &structure,
                &structure,
                black_box(&mut dst),
                black_box(&src),
                1.0,
            )
            .unwrap();
        });
    });
}

criterion_group!(benches, bench_permute_u1_many);
criterion_main!(benches);
