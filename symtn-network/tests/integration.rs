//! Behavior tests neither observe nor reset process-global state and run concurrently.

use symtn::typed::{Complex32, Complex64};

#[path = "../../symtn/tests/braiding_probe/mod.rs"]
mod braiding_probe;
#[path = "../../tests/support/errors.rs"]
mod errors;
#[path = "../../tests/support/legs.rs"]
mod legs;
#[path = "../../tests/support/network.rs"]
mod network_support;
#[path = "../../tests/support/numerics.rs"]
mod numerics;

mod contract_macro;
mod error_chain;
#[path = "../../tests/support/export_scan.rs"]
mod export_scan;
mod itebd_smoke;
mod mf_provider_conformance;
mod network_construction_errors;
mod network_contract;
mod network_exports;
mod optimizer_fallback;
mod plan_api_equivalence;
mod plan_cache;
mod plan_text_input;
mod single_precision_network;
mod symmetric_slice;
mod symmetric_slice_lowering;
mod tk_fermionic_correspondence;
mod typed_cuda_network;
mod typed_host_network;
mod typed_network;
mod user_c64;
