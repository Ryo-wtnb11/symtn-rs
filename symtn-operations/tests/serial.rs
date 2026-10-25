//! Global-state and allocation tests each take the shared serial guard first.

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

mod fusion_group_replay_allocations;
mod owned_cat_allocations;
mod owned_trace_allocations;
mod owned_transform_allocations;
mod raw_combine_allocations;
mod task_view_admission_allocations;
mod tree_transform_compile_allocations;
