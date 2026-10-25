use std::sync::Arc;

use symtn_core::{FusionProductSpace, FusionTreeHomSpace, U1FusionRule};
use symtn_tensors::BoundDynamicFusionMapSpace;

use crate::counting_alloc;

#[test]
fn lowered_scratch_hit_matches_encoded_hit_allocation_and_identity() {
    let _serial = crate::counting_alloc::serial();
    // What: under the shared serial guard, the transactional lowered warm hit
    // keeps encoded allocation cost and exact retained structure identity.
    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();
    let provider = Arc::new(U1FusionRule);
    let homspace =
        FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([]));
    let cold = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(
        Arc::clone(&provider),
        homspace,
    )
    .unwrap();

    let lowered_homspace = cold.space().homspace().clone();
    counting_alloc::start();
    let lowered = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(
        Arc::clone(&provider),
        lowered_homspace,
    )
    .unwrap();
    let allocs = counting_alloc::stop();
    let lowered_allocations = allocs.calls as usize;

    let encoded_homspace = cold.space().homspace().clone();
    counting_alloc::start();
    let encoded = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        provider,
        encoded_homspace,
    )
    .unwrap();
    let allocs = counting_alloc::stop();
    let encoded_allocations = allocs.calls as usize;

    assert_eq!(lowered_allocations, encoded_allocations);
    assert!(Arc::ptr_eq(
        cold.space().structure(),
        lowered.space().structure()
    ));
    assert!(Arc::ptr_eq(
        cold.space().structure(),
        encoded.space().structure()
    ));
    assert_eq!(lowered.space(), encoded.space());
}
