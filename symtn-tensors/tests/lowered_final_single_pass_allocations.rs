use std::sync::Arc;

use symtn_core::{FusionProductSpace, FusionTreeHomSpace, SectorLeg, U1FusionRule, U1Irrep};
use symtn_tensors::BoundDynamicFusionMapSpace;

use crate::counting_alloc;

fn homspace(sector_count: i32) -> FusionTreeHomSpace {
    let sectors = (0..sector_count)
        .map(|charge| (U1Irrep::new(charge).sector_id(), 2))
        .collect::<Vec<_>>();
    FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new(sectors.clone(), false)]),
        FusionProductSpace::new([SectorLeg::new(sectors, false)]),
    )
}

fn cold_lowered_allocations(sector_count: i32) -> usize {
    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();
    counting_alloc::start();
    let result = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(
        Arc::new(U1FusionRule),
        homspace(sector_count),
    )
    .unwrap();
    let allocs = counting_alloc::stop();
    assert_eq!(
        result.space().structure().block_count(),
        sector_count as usize
    );
    allocs.calls as usize
}

fn cold_encoded_allocations(sector_count: i32) -> usize {
    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();
    counting_alloc::start();
    let result = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::new(U1FusionRule),
        homspace(sector_count),
    )
    .unwrap();
    let allocs = counting_alloc::stop();
    assert_eq!(
        result.space().structure().block_count(),
        sector_count as usize
    );
    allocs.calls as usize
}

#[test]
fn lowered_final_cold_build_has_no_per_tree_shape_allocation_slope() {
    let _serial = crate::counting_alloc::serial();
    // What: growing one-sector final storage to sixteen U1 sectors does not
    // restore one heap-allocated shape vector per fusion-tree key.
    let one = cold_lowered_allocations(1);
    let sixteen = cold_lowered_allocations(16);
    let encoded_one = cold_encoded_allocations(1);
    let encoded_sixteen = cold_encoded_allocations(16);

    // The encoded builder is the single-pass baseline. The
    // checked enumeration may change the constant term, but must not add a
    // steeper allocation slope as the tree count grows.
    assert!(sixteen - one <= encoded_sixteen - encoded_one);
}
