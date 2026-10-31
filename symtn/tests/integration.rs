//! Behavior tests neither observe nor reset process-global state and run concurrently.

use num_complex::{Complex32, Complex64};

#[path = "../../tests/support/backend.rs"]
mod backend;
mod braiding_probe;
#[macro_use]
mod common;
#[macro_use]
mod contract_cases;
#[path = "../../tests/support/fixtures.rs"]
mod fixtures;
#[path = "../../tests/support/legs.rs"]
mod legs;
#[cfg(feature = "cgc-gen")]
#[path = "../../tests/support/su3.rs"]
mod su3;
#[macro_use]
#[path = "support/nonfinite_factor_input.rs"]
mod nonfinite_factor_input;
#[path = "../../tests/support/numerics.rs"]
mod numerics;
mod prepared;
#[path = "support/report.rs"]
mod report;
#[macro_use]
mod single_precision_oracle;
#[macro_use]
#[path = "stacked/fixtures.rs"]
mod stacked_fixtures;
#[macro_use]
#[path = "support/table_fixture.rs"]
mod table_fixture;
mod engine_ownership;
#[path = "../../tests/support/toy_rules.rs"]
mod toy_rules;
mod trace_cases;
#[macro_use]
mod truncation_oracle;

mod axis_error_matrix;
mod axpby_dense_oracle;
mod batched_failure_contract;
mod c64_compose_contract_identity;
mod capability_markers;
mod checked_core_region_oracle;
mod checked_generic_absorb;
mod checked_generic_trace_duality;
mod checked_generic_trace_oracle;
mod compact_reductions_both_modes;
mod compact_spectrum_axpby_lazy_adjoint;
mod complex_symbol_lazy_adjoint;
mod contract_into_reordered_overwrite;
mod contract_spec_split;
mod contraction_output_init;
mod coupled_blocks;
mod cuda_engine_ownership;
mod cuda_warm_up;
mod current_public_api_smoke;
mod destination_on_failure;
mod diagonal_api;
mod eager_compose_oracle;
mod eig_su2_equation;
mod eig_value_order;
mod eigh_hermitian_tolerance;
mod error_chain;
mod exp_spectral_route;
#[path = "../../tests/support/export_scan.rs"]
mod export_scan;
mod facade_exports;
mod factorization_nonfinite_input;
mod into_coefficient_pairs;
mod lazy_adjoint_contract_both_modes;
mod lazy_adjoint_exact;
mod lazy_adjoint_factorizations;
mod lazy_adjoint_lq;
mod lazy_adjoint_non_self_dual_contract;
mod lazy_adjoint_shared_owner;
mod lazy_axpby_both_modes;
mod lie_group_tensors;
mod mf_linalg_conformance;
mod mf_structural_conformance;
mod named_factorization_reconstruction;
mod non_finite_spectra;
mod norm_order;
mod null_space_nonabelian;
mod path_form_additive;
mod physical_dense;
mod predicate_chains;
mod prepared_compose;
mod prepared_compose_cuda;
mod prepared_eigh;
mod prepared_eigh_cuda;
mod product_generic;
mod qspace_su2_correspondence;
mod reduction_accumulator_precision;
mod restrict_embed_leg;
mod restrict_leg_checked_generic;
mod scaled_complex64_reciprocal;
mod semantic_suite;
mod shared_runtime_parallel;
mod side_only_factor_publication;
mod single_precision_advanced_linalg;
mod single_precision_base;
mod single_precision_factorizations;
mod stacked_restrict_leg;
mod stacked_restrict_leg_cuda;
mod stacked_select_cuda;
mod stacked_tensor_map;
mod stacked_tensor_map_cuda;
mod storage_parity_matrix;
mod structural_block_identity;
mod svd_null_composition;
mod table_factorizations;
mod table_fusion_only;
mod table_planar_contract;
mod table_planar_workflow;
mod table_reductions;
mod table_twist_flip;
mod tk_complex_su2;
mod tk_disjoint_null;
mod tk_exp_general;
mod tk_norm_p;
mod tk_norm_range;
mod trace_error_order;
mod transform_coefficient_exact;
mod truncation_composition;
mod truncation_composition_checked_generic;
mod truncation_composition_eig;
mod truncation_dim_weight;
mod truncation_range;
mod truncation_tie_order;
mod truncspace;
mod typed_contract_batch;
mod typed_contract_host_oracle;
mod typed_cuda_contract_stages;
mod typed_cuda_eigh_assembly;
mod typed_cuda_leg_roles;
mod typed_cuda_materialize;
mod typed_cuda_single_precision;
mod typed_cuda_single_precision_factorizations;
mod typed_cuda_spectrum_download;
mod typed_cuda_trace;
#[path = "typed_cuda_transfer/main.rs"]
mod typed_cuda_transfer;
mod typed_cuda_transform;
mod typed_cuda_twist;
mod typed_materialize;
mod typed_serialization;
mod typed_trace_host_oracle;
mod typed_zero_scale;
mod user_decomp;
mod zeros_like_all_modes;
