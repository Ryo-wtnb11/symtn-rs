//! Global-state and allocation tests each take the shared serial guard first.

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

mod conjugated_contract_allocations;
mod dynamic_tree_route_allocations;
mod lowered_cold_contract_plan_allocations;
mod lowered_final_single_pass_allocations;
mod lowered_transaction_allocations;
mod prepared_fusion_replay_allocations;
mod tree_transform_compile_allocations;
mod warm_contract_compile_allocations;
mod warm_trace_compile_allocations;

mod lowered_complete_cache_preflight;
