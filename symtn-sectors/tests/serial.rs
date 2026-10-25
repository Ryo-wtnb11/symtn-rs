//! Global-state and allocation tests each take the shared serial guard first.

use num_complex::{Complex32, Complex64};

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;
#[path = "../../tests/support/numerics.rs"]
mod numerics;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

mod abelian_cu1_symbol_allocations;
mod lie_group_adapter_allocations;
mod lie_group_identity_allocations;
mod lie_label_allocations;
mod product_query_allocations;
