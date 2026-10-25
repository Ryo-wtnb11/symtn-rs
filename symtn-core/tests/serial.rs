//! Global-state and allocation tests each take the shared serial guard first.

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

mod degeneracy_build_allocations;
mod identity_transform_allocations;
mod structure_cache_hit_allocations;

mod checked_generic_structure;
