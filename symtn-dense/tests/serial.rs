//! Global-state and allocation tests each take the shared serial guard first.

use num_complex::{Complex32, Complex64};

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

#[path = "../../tests/support/numerics.rs"]
mod numerics;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

mod cuda_transfer_allocations;
mod fresh_allocations;
mod gemm_plan_cache;
