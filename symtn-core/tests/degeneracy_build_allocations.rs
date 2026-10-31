//! Cold degeneracy-structure build allocations are independent of the block
//! count at every rank: a block's shape and strides are
//! written straight into the packed `DegeneracyStructure` arena, so no
//! rank-sized per-block (or per-tree) buffer can spill above rank 8.

use std::hint::black_box;

use symtn_core::{FusionProductSpace, FusionTreeHomSpace, SectorLeg, Z2FusionRule, Z2Irrep};

use crate::counting_alloc;

/// A Z2 HomSpace of `rank` legs, split as evenly as possible. The first
/// `even_only` legs carry only the even sector, which divides the block count
/// by `2^even_only` without changing the rank. `degeneracy` scales every
/// sector.
fn homspace(rank: usize, even_only: usize, degeneracy: usize) -> FusionTreeHomSpace {
    let leg = |axis: usize| {
        let odd = (axis >= even_only).then_some((Z2Irrep::ODD, degeneracy + 1));
        SectorLeg::new(
            std::iter::once((Z2Irrep::EVEN, degeneracy)).chain(odd),
            false,
        )
    };
    let nout = rank / 2;
    FusionTreeHomSpace::new(
        FusionProductSpace::new((0..nout).map(leg).collect::<Vec<_>>()),
        FusionProductSpace::new((nout..rank).map(leg).collect::<Vec<_>>()),
    )
}

/// Allocation calls of one cold degeneracy-structure build over a warm
/// sector layout. Why the prepared builder rather than the cached public one:
/// it builds the same structure without publishing it, so the measurement
/// excludes cache-internal growth that depends on what was admitted before.
fn cold_degeneracy_build_calls(rank: usize, even_only: usize) -> (usize, u64) {
    let fresh = homspace(rank, even_only, 2);
    let layout = fresh.prepare_fusion_tree_layout(&Z2FusionRule);
    let (structure, allocs) = counting_alloc::measure(|| {
        layout
            .build_from_leg_degeneracies(black_box(&fresh))
            .unwrap()
    });
    (structure.block_count(), allocs.calls)
}

#[test]
fn cold_degeneracy_build_calls_do_not_grow_with_block_count() {
    let _serial = crate::counting_alloc::serial();
    let rows = [6usize, 8, 10, 12].map(|rank| {
        let small = cold_degeneracy_build_calls(rank, 2);
        let large = cold_degeneracy_build_calls(rank, 0);
        assert_eq!(large.0, 4 * small.0);
        (rank, small, large)
    });
    println!("(rank, (blocks, calls) small, large): {rows:?}");
    for (rank, (small_blocks, small_calls), (large_blocks, large_calls)) in rows {
        assert_eq!(
            large_calls, small_calls,
            "rank {rank}: {large_calls} calls for {large_blocks} blocks, \
             {small_calls} for {small_blocks}; all ranks: {rows:?}"
        );
    }
}
