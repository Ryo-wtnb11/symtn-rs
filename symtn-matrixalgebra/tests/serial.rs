//! Global-state and allocation tests each take the shared serial guard first.

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;
#[path = "../../tests/support/generic_toy_fusion.rs"]
mod generic_toy_fusion;

#[path = "../src/tests/scripted_executor.rs"]
mod scripted_executor;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

mod factorization_session_scope;
