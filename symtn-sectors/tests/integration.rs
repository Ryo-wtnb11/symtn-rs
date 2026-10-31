//! Behavior tests neither observe nor reset process-global state and run concurrently.

use num_complex::{Complex32, Complex64};

#[path = "../../tests/support/numerics.rs"]
mod numerics;
#[path = "../../tests/support/sector_ids.rs"]
mod sector_ids;

mod abelian_cu1_values;
mod builtin_rules;
mod category_rule;
mod downstream_contracts;
mod fibonacci_providers;
mod infallible_matches_checked_domain;
mod infallible_tier_gate;
mod lie_group_adapter_contracts;
mod lie_group_cross_process_determinism;
mod lie_group_oracles;
mod product_oracles;
mod product_rule_client;
#[path = "provider_contracts/main.rs"]
mod provider_contracts;
mod rule_identity;
mod sector_labels;
mod semantic_axioms;
mod su2_bend_oracle;
mod table_rule;
