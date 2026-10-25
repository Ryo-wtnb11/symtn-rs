//! Real-device gates for #1278: building a CUDA `Runtime` pays the tenferro
//! backend library initialization, so no user operation does.
//!
//! Run with `cargo test -p symtn --features cuda,cpu-faer --test integration cuda_warm_up:: \
//! -- --ignored` on a CUDA host. Transfer counter snapshots are thread-local.

#![cfg(feature = "cuda")]

use crate::report;
use report::Report;
use std::sync::Arc;

use symtn::expert::cuda_transfer_stats;
use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{ContractSpec, GradedSpace, Runtime, TensorMap};

#[test]
#[ignore = "requires a real CUDA device"]
fn a_first_contract_on_a_fresh_runtime_does_no_backend_initialization_work() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0), 2)]).unwrap();
    let lhs = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        indices.iter().sum::<usize>() as f64 + 1.0
    })
    .unwrap();
    let rhs = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        indices.iter().sum::<usize>() as f64 - 1.0
    })
    .unwrap();
    let expected = lhs
        .contract(
            &rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    let lhs_device = lhs.to_cuda().unwrap();
    let rhs_device = rhs.to_cuda().unwrap();

    // The build above already ran the warm-up's GEMM and solver call, so this
    // first device contract contributes only its own work: one GEMM for the
    // single coupled block, and no solver call at all.
    let before = cuda_transfer_stats();
    let product = lhs_device
        .contract(
            &rhs_device,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    let after = cuda_transfer_stats();
    assert_eq!(after.gemm_calls - before.gemm_calls, 1);
    assert_eq!(after.solver_calls, before.solver_calls);

    assert_eq!(
        product.to_host().unwrap().dense_data().unwrap(),
        expected.dense_data().unwrap()
    );
}

// Not `#[ignore]`d: the ordinal is rejected before anything touches CUDA, so
// this runs wherever the `cuda` feature compiles.
#[test]
fn building_on_an_absent_device_fails_with_the_typed_context_error() {
    // The warm-up runs only after the context exists, so an unusable ordinal
    // still surfaces as the context construction error and no Runtime is built.
    let error = Runtime::builder().cuda(usize::MAX).build().unwrap_err();
    let text = error.report();
    assert!(text.contains("Cuda"), "names the CUDA backend: {text}");
    assert!(
        text.contains("cuda_context"),
        "names the failing operation: {text}"
    );
}
