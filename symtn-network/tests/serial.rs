//! Global-state and allocation tests each take the shared serial guard first.

#[path = "../../tests/support/su2.rs"]
mod su2;
use symtn::typed::{Complex32, Complex64};
use symtn_core as sector_types;

#[path = "../../symtn/tests/braiding_probe/mod.rs"]
mod braiding_probe;
#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;
#[path = "../../tests/support/errors.rs"]
mod errors;
#[cfg(feature = "cgc-gen")]
#[path = "support/injected_generic.rs"]
mod injected_generic;
#[path = "../../tests/support/network.rs"]
mod network_support;
#[path = "../../tests/support/numerics.rs"]
mod numerics;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

mod intermediate_orientation_cost;
mod network_engine_scratch;
mod output_partition_passes;
mod preflight_allocations;

mod checked_generic_network;
