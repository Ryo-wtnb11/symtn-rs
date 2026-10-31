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
//! - A prepared workspace follows the engine of each call: an output it
//!   retained for one engine is dropped, not reused, on another.
//!
//! Run with `cargo test -p symtn --no-default-features --features cuda,cpu-faer
//! --test integration cuda_engine_ownership:: -- --ignored` on a CUDA host.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use symtn::expert::cuda_transfer_stats;
use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{
    ComposePlan, ContractPlan, ContractSpec, CpuEngine, CudaStorage, EighFullPlan, Engine,
    GradedSpace, HermitianTol, Runtime, StackedTensorMap, TensorMap,
};

use crate::prepared::eigh::hermitian_members;
use crate::prepared::{members, u1_legs};

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
    // Every returning entry is refused before it uploads its output.
    assert!(lhs.contract(second, &rhs, &SPEC).is_err());
    assert!(lhs.compose(second, &rhs).is_err());
    assert!(lhs.zeros_like(second).is_err());
    assert!(lhs.scale(second, 2.0).is_err());
    assert!(lhs.axpby(second, 2.0, &rhs, -3.0).is_err());
    assert!(lhs.permute(second, &[1], &[0]).is_err());
    assert!(lhs.norm(second, 2.0).is_err());
    assert!(lhs.inner(second, &rhs).is_err());
    assert!(lhs
        .contract_into(second, &rhs, &SPEC, &mut destination, 1.0, 0.0)
        .is_err());
    assert!(lhs.axpby_into(second, &mut destination, 1.0, 1.0).is_err());
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

const MEMBERS: usize = 2;

fn stack_values(engine: &mut dyn Engine, stack: &StackedTensorMap<U1FusionRule, f64>) -> Vec<f64> {
    (0..MEMBERS)
        .flat_map(|member| {
            stack
                .member(engine, member)
                .unwrap()
                .dense_data()
                .unwrap()
                .to_vec()
        })
        .collect()
}

fn assert_close(what: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (got, want) in got.iter().zip(want) {
        assert!(
            (got - want).abs() <= 1e-12 * (1.0 + want.abs()),
            "{what}: {got} != {want}"
        );
    }
}

/// What: one compose and one contract workspace driven A -> B -> A at a fixed
/// member count, with matching operands uploaded on the engine of each call,
/// succeed every time with the Host values: the output retained for one
/// engine is dropped on the other instead of being written or kept.
#[test]
#[ignore = "requires a real CUDA device"]
fn a_prepared_contraction_follows_the_engine_of_each_call() {
    let engine = &mut CpuEngine::serial();
    let first = &mut crate::common::cuda_engine();
    let second = &mut crate::common::cuda_engine();
    let runtime = Runtime::builder().build().unwrap();
    let (v, w) = u1_legs();
    let lhs =
        StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&v, &v], &[&w], MEMBERS, 1)).unwrap();
    let rhs =
        StackedTensorMap::pack(&members::<_, f64>(&runtime, &[&w], &[&v], MEMBERS, 2)).unwrap();
    let spec = ContractSpec {
        lhs: &[2],
        rhs: &[0],
        codomain: &[0, 1],
        domain: &[2],
    };
    let expected = {
        let plan = ComposePlan::new(engine, &lhs, &rhs).unwrap();
        let mut workspace = plan.workspace().unwrap();
        let output = plan.execute(engine, &lhs, &rhs, &mut workspace).unwrap();
        stack_values(engine, output)
    };
    let (lhs_a, rhs_a) = (lhs.to_cuda(first).unwrap(), rhs.to_cuda(first).unwrap());
    let (lhs_b, rhs_b) = (lhs.to_cuda(second).unwrap(), rhs.to_cuda(second).unwrap());

    let compose = ComposePlan::new(first, &lhs_a, &rhs_a).unwrap();
    let mut workspace = compose.workspace().unwrap();
    for (what, on_b) in [
        ("compose A", false),
        ("compose B", true),
        ("compose A again", false),
    ] {
        let (cuda, lhs, rhs): (&mut dyn Engine, _, _) = if on_b {
            (&mut *second, &lhs_b, &rhs_b)
        } else {
            (&mut *first, &lhs_a, &rhs_a)
        };
        let output = compose.execute(cuda, lhs, rhs, &mut workspace).unwrap();
        let host = output.to_host(cuda).unwrap();
        assert_close(what, &stack_values(engine, &host), &expected);
    }

    let contract = ContractPlan::new(first, &lhs_a, &rhs_a, &spec).unwrap();
    let mut workspace = contract.workspace().unwrap();
    for (what, on_b) in [
        ("contract A", false),
        ("contract B", true),
        ("contract A again", false),
    ] {
        let (cuda, lhs, rhs): (&mut dyn Engine, _, _) = if on_b {
            (&mut *second, &lhs_b, &rhs_b)
        } else {
            (&mut *first, &lhs_a, &rhs_a)
        };
        let output = contract.execute(cuda, lhs, rhs, &mut workspace).unwrap();
        let host = output.to_host(cuda).unwrap();
        assert_close(what, &stack_values(engine, &host), &expected);
    }

    // A's operands on B are refused before the call uploads anything.
    let stats = cuda_transfer_stats();
    assert!(contract
        .execute(second, &lhs_a, &rhs_a, &mut workspace)
        .is_err());
    assert_eq!(
        cuda_transfer_stats(),
        stats,
        "nothing reached either engine"
    );
}

/// What: one EIGH workspace driven A -> B -> A at a fixed member count
/// returns the same spectra and factors every time: `(d, v)` retained for
/// one engine are dropped on the other before any solver work.
#[test]
#[ignore = "requires a real CUDA device"]
fn a_prepared_eigh_follows_the_engine_of_each_call() {
    let engine = &mut CpuEngine::serial();
    let first = &mut crate::common::cuda_engine();
    let second = &mut crate::common::cuda_engine();
    let runtime = Runtime::builder().build().unwrap();
    let (v, _) = u1_legs();
    let source = StackedTensorMap::pack(&hermitian_members(&runtime, &[&v], MEMBERS, 7)).unwrap();
    let (source_a, source_b) = (
        source.to_cuda(first).unwrap(),
        source.to_cuda(second).unwrap(),
    );
    let plan = EighFullPlan::new(engine, &source_a, &[0], &[1], HermitianTol::DEFAULT).unwrap();
    let mut workspace = plan.workspace().unwrap();
    let mut runs = Vec::new();
    for (what, on_b) in [("A", false), ("B", true), ("A again", false)] {
        let (cuda, source): (&mut dyn Engine, _) = if on_b {
            (&mut *second, &source_b)
        } else {
            (&mut *first, &source_a)
        };
        let output = plan.execute(cuda, source, &mut workspace).unwrap();
        let spectra: Vec<f64> = output
            .spectra
            .iter()
            .flatten()
            .flat_map(|entry| entry.values.clone())
            .collect();
        let d = output.d.to_host(cuda).unwrap();
        runs.push((what, spectra, stack_values(engine, &d)));
    }
    for (what, spectra, d) in &runs[1..] {
        assert_close(what, spectra, &runs[0].1);
        assert_close(what, d, &runs[0].2);
    }
}
