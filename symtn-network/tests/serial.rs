//! Global-state and allocation tests each take the shared serial guard first.

use symtn::typed::{Complex32, Complex64};

#[path = "../../symtn/tests/braiding_probe/mod.rs"]
mod braiding_probe;
#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;
#[path = "../../tests/support/network.rs"]
mod network_support;
#[path = "../../tests/support/numerics.rs"]
mod numerics;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

mod intermediate_orientation_cost;
mod output_partition_passes;
mod preflight_allocations;

mod checked_generic_network;
