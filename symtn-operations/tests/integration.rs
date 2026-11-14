//! Behavior tests neither observe nor reset process-global state and run concurrently.

mod common;
#[path = "../../tests/support/su2.rs"]
mod su2;
use symtn_core as sector_types;

mod cuda_tree_transform;
mod dense_backend_oracle;
mod tree_transform_device_oracle;
