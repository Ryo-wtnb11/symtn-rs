//! Behavior tests neither observe nor reset process-global state and run concurrently.

mod canonical_destination;
mod categorical_recoupling;
#[path = "../../tests/support/su2.rs"]
mod su2;
use symtn_core as sector_types;

mod cuda_tree_transform_categorical;
#[path = "../../tests/support/export_scan.rs"]
mod export_scan;
mod rule_identity_boundary;
mod tensors_exports;
mod tree_transform_categorical_recoupling;
