//! Behavior tests neither observe nor reset process-global state and run concurrently.

mod core_exports;
#[path = "../../tests/support/export_scan.rs"]
mod export_scan;
mod key_identity;
mod rule_identity;
mod sector_vocabulary_compat;
#[path = "../../tests/support/su2.rs"]
mod su2;
use symtn_core as sector_types;
mod unique_scratch_reuse;
