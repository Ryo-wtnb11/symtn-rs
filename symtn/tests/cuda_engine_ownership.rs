//! A device operation runs on the caller's CUDA engine and on nothing else.
//!
//! - A CPU engine handed to a device operation is refused with a typed error
//!   before any transfer, and the device destination keeps its bits.
//! - A tensor owned by another CUDA engine, even one on the same device
//!   ordinal, is refused the same way: engines are separate allocation
//!   domains, and nothing moves a buffer between them implicitly.
//! - A CUDA engine handed to a Host contraction, permutation or value
//!   operation is refused; there is no device fallback for Host payloads,
//!   and the CUDA engine's serial Host token covers structural metadata
//!   only.
//!
//! Run with `cargo test -p symtn --no-default-features --features cuda,cpu-faer
//! --test integration cuda_engine_ownership:: -- --ignored` on a CUDA host.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use symtn::expert::cuda_transfer_stats;
use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{ContractSpec, CpuEngine, CudaStorage, Engine, GradedSpace, Runtime, TensorMap};

type Host = TensorMap<U1FusionRule, f64>;
type Device = TensorMap<U1FusionRule, f64, CudaStorage<f64>>;

const SPEC: ContractSpec<'static> = ContractSpec {
    lhs: &[1],
    rhs: &[0],
    codomain: &[0],
    domain: &[1],
};

fn fixture(engine: &mut dyn Engine, runtime: &Runtime, salt: f64) -> Host {
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 3)],
    )
    .unwrap();
    TensorMap::from_subblock_fn(engine, runtime, [&leg], [&leg], |_, index| {
        salt + (index[0] + 2 * index[1]) as f64
    })
    .unwrap()
}

fn bits(cuda: &mut dyn Engine, tensor: &Device) -> Vec<u64> {
    let host = tensor.to_host(cuda).unwrap();
    host.dense_data()
        .unwrap()
        .iter()
        .map(|x| x.to_bits())
        .collect()
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_cpu_engine_is_refused_on_device_tensors_before_any_transfer() {
    let engine = &mut CpuEngine::serial();
    let cuda = &mut crate::common::cuda_engine();
    let runtime = Runtime::builder().build().unwrap();
    let lhs = fixture(engine, &runtime, 1.0).to_cuda(cuda).unwrap();
    let rhs = fixture(engine, &runtime, 2.0).to_cuda(cuda).unwrap();
    let mut destination = fixture(engine, &runtime, 3.0).to_cuda(cuda).unwrap();
    let before = bits(cuda, &destination);

    let stats = cuda_transfer_stats();
    assert!(lhs.contract(engine, &rhs, &SPEC).is_err());
    assert!(lhs
        .contract_into(engine, &rhs, &SPEC, &mut destination, 1.0, 0.0)
        .is_err());
    assert!(lhs.norm(engine, 2.0).is_err());
    assert!(lhs.to_host(engine).is_err());
    assert_eq!(cuda_transfer_stats(), stats, "nothing reached the device");
    assert_eq!(bits(cuda, &destination), before);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_tensor_of_another_engine_on_the_same_device_is_refused_before_any_transfer() {
    let engine = &mut CpuEngine::serial();
    let first = &mut crate::common::cuda_engine();
    let second = &mut crate::common::cuda_engine();
    let runtime = Runtime::builder().build().unwrap();
    let lhs = fixture(engine, &runtime, 1.0).to_cuda(first).unwrap();
    let rhs = fixture(engine, &runtime, 2.0).to_cuda(first).unwrap();
    let mut destination = fixture(engine, &runtime, 3.0).to_cuda(first).unwrap();
    let before = bits(first, &destination);

    let stats = cuda_transfer_stats();
    assert!(lhs.contract(second, &rhs, &SPEC).is_err());
    assert!(lhs
        .contract_into(second, &rhs, &SPEC, &mut destination, 1.0, 0.0)
        .is_err());
    assert!(lhs.to_host(second).is_err());
    assert_eq!(
        cuda_transfer_stats(),
        stats,
        "nothing reached either engine"
    );
    assert_eq!(bits(first, &destination), before);
    // The owning engine still runs the same call.
    assert!(lhs.contract(first, &rhs, &SPEC).is_ok());
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_cuda_engine_is_refused_on_host_contractions() {
    let engine = &mut CpuEngine::serial();
    let cuda = &mut crate::common::cuda_engine();
    let runtime = Runtime::builder().build().unwrap();
    let lhs = fixture(engine, &runtime, 1.0);
    let rhs = fixture(engine, &runtime, 2.0);
    let mut destination = fixture(engine, &runtime, 3.0);
    let before = destination.dense_data().unwrap().to_vec();
    assert!(lhs.contract(cuda, &rhs, &SPEC).is_err());
    assert!(lhs
        .contract_into(cuda, &rhs, &SPEC, &mut destination, 1.0, 0.0)
        .is_err());
    assert_eq!(destination.dense_data().unwrap(), before.as_slice());
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_cuda_engine_is_refused_on_host_permutations_and_value_operations() {
    let engine = &mut CpuEngine::serial();
    let cuda = &mut crate::common::cuda_engine();
    let runtime = Runtime::builder().build().unwrap();
    let host = fixture(engine, &runtime, 1.0);
    let mut destination = fixture(engine, &runtime, 3.0);
    let before = destination.dense_data().unwrap().to_vec();
    // A nonidentity U(1) permutation is a pure Host payload replay.
    assert!(host.permute(cuda, &[1], &[0]).is_err());
    assert!(host.scale(cuda, 2.0).is_err());
    assert!(host.norm(cuda, 2.0).is_err());
    assert!(host.axpby_into(cuda, &mut destination, 1.0, 1.0).is_err());
    assert_eq!(destination.dense_data().unwrap(), before.as_slice());
}
