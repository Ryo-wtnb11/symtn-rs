//! Behavior tests neither observe nor reset process-global state and run concurrently.

mod common;

mod cuda_tree_transform;
mod dense_backend_oracle;
mod tree_transform_device_oracle;
