//! Real-device gate: a first device contraction submits only its
//! own work (one GEMM per coupled block, no solver call).
//!
//! Run with `cargo test -p symtn --features cuda,cpu-faer --test integration cuda_warm_up:: \
//! -- --ignored` on a CUDA host. Transfer counter snapshots are thread-local.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use symtn::expert::cuda_transfer_stats;
use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{ContractSpec, GradedSpace, Runtime, TensorMap};

#[test]
#[ignore = "requires a real CUDA device"]
fn a_first_contract_on_a_fresh_runtime_does_no_backend_initialization_work() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut crate::common::cuda_engine();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0), 2)]).unwrap();
    let lhs = TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, indices| {
        indices.iter().sum::<usize>() as f64 + 1.0
    })
    .unwrap();
    let rhs = TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, indices| {
        indices.iter().sum::<usize>() as f64 - 1.0
    })
    .unwrap();
    let expected = lhs
        .contract(
            engine,
            &rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    let lhs_device = lhs.to_cuda(cuda).unwrap();
    let rhs_device = rhs.to_cuda(cuda).unwrap();

    // The first device contract contributes only its own work: one GEMM for
    // the single coupled block, and no solver call at all.
    let before = cuda_transfer_stats();
    let product = lhs_device
        .contract(
            cuda,
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
        product.to_host(cuda).unwrap().dense_data().unwrap(),
        expected.dense_data().unwrap()
    );
}
