//! Global-state and allocation tests each take the shared serial guard first.

use num_complex::{Complex32, Complex64};

#[path = "../../tests/support/backend.rs"]
mod backend;
mod braiding_probe;
#[macro_use]
mod common;
#[macro_use]
#[allow(unused_macros)]
mod contract_cases;
#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;
#[path = "../../tests/support/fixtures.rs"]
mod fixtures;
#[path = "../../tests/support/numerics.rs"]
mod numerics;
#[macro_use]
#[path = "support/nonfinite_factor_input.rs"]
mod nonfinite_factor_input;
mod prepared;
#[path = "support/report.rs"]
mod report;
#[macro_use]
mod single_precision_oracle;
#[path = "../../tests/support/toy_rules.rs"]
mod toy_rules;
mod ulp;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

mod adjoint_view_allocations;
mod adjoint_view_operand_allocations;
mod batched_failure_allocations;
mod checked_staged_structure_allocations;
mod compact_factor_plan_allocations;
mod compact_mode_parity;
mod completed_transformer_group_eviction_charge;
mod contract_candidate_core;
mod contract_output_axes_allocations;
mod contract_route_host_allocations;
mod contraction_output_allocations;
mod direct_factor_publication;
mod eager_copy_c_pool;
mod exp_pade_allocations;
mod factorization_leg_roles;
mod identity_adjoint_borrow_allocations;
mod inner_norm_allocations;
mod into_admission_matrix;
mod krylov_vector;
mod lazy_adjoint_contract_allocations;
mod lazy_adjoint_factorization_allocations;
mod permute_overwrite_allocations;
mod planar_contract_allocations;
mod pooled_dense_allocations;
mod precision_conversions;
mod prepared_compose_allocations;
mod prepared_compose_pins;
mod prepared_eigh_allocations;
mod repartition_allocations;
mod restrict_leg_allocations;
mod single_precision_allocations;
mod structure_signature;
mod trace_warm_allocations;
mod transform_owned_parallel;
mod typed_add_assign;
mod typed_cat_absorb_allocations;
mod typed_cuda_contract;
mod typed_cuda_contract_batch;
mod typed_diagonal_allocations;
mod typed_otimes_allocations;

mod checked_generic_braiding;

#[path = "checked_generic_facade/main.rs"]
mod checked_generic_facade;

mod coefficient_group_cache;

mod complete_structure_warm_hits;

mod complete_structure_working_set;

mod composed_coefficient_cache;

mod factorization_session_scope;

mod solve_leg_roles;

mod typed_cuda_transform_contracts;

#[path = "typed_facade/main.rs"]
mod typed_facade;

mod typed_transform_host_side;
