//! Device-only gates for `trace_pairs` on device tensors: device == Host of
//! the same public call for every fixture and payload dtype, and rejections
//! that match the Host's and submit no device work. The region primitive is
//! gated in `symtn-dense/tests/cuda_region_trace.rs`; the independent oracles
//! run on both engines in `engine_suites/trace.rs`.
//!
//! CUDA transfer counters are thread-local, so these tests run concurrently.
//! Run with `cargo test -p symtn --no-default-features
//! --features cuda,cpu-faer --test integration typed_cuda_trace:: -- --ignored
//! --test-threads=1` on a CUDA host.

#![cfg(feature = "cuda")]

use crate::common;

use crate::numerics;
use crate::report;
use crate::trace_cases;
use report::Report;

use common::{DevicePayload, DeviceRule};
use num_complex::{Complex32, Complex64};
use symtn::expert::{cuda_transfer_stats, CudaTransferStats};
use symtn::typed::{Runtime, TensorMap};
use trace_cases::{
    fermion_su2_cases, fermion_u1_cases, su2_cases, u1_cases, u1_su2_cases, TraceCase,
};

fn delta<T>(body: impl FnOnce() -> T) -> (T, CudaTransferStats) {
    let before = cuda_transfer_stats();
    let value = body();
    let after = cuda_transfer_stats();
    (
        value,
        CudaTransferStats {
            h2d_calls: after.h2d_calls - before.h2d_calls,
            h2d_bytes: after.h2d_bytes - before.h2d_bytes,
            d2h_calls: after.d2h_calls - before.d2h_calls,
            d2h_bytes: after.d2h_bytes - before.d2h_bytes,
            device_allocs: after.device_allocs - before.device_allocs,
            gemm_calls: after.gemm_calls - before.gemm_calls,
            solver_calls: after.solver_calls - before.solver_calls,
            copy_calls: after.copy_calls - before.copy_calls,
            gauge_ops: after.gauge_ops - before.gauge_ops,
        },
    )
}

/// Device `trace_pairs` of `case`, downloaded. A lazy-adjoint fixture is
/// rebuilt as a device lazy adjoint of the device parent, so the device path
/// under test is the parent read, not a materialized copy.
fn device<R: DeviceRule, D: DevicePayload>(
    cuda: &mut dyn symtn::typed::Engine,
    tensor: &TensorMap<R, D, symtn::typed::CudaStorage<D>>,
    pairs: &[(usize, usize)],
) -> TensorMap<R, D> {
    tensor
        .trace_pairs(cuda, pairs)
        .unwrap()
        .to_host(cuda)
        .unwrap()
}

/// Device == Host of the same public call. The independent oracles (identity
/// contraction, physical-basis sum, materialized adjoint, fZ2 supertrace,
/// accumulation) run on both engines in `engine_suites/trace.rs`; the
/// fermionic fixtures have no other oracle than the Host.
fn check<R: DeviceRule, D: DevicePayload>(
    cuda: &mut dyn symtn::typed::Engine,
    case: &TraceCase<R, D>,
) {
    let host = case.host();
    let device_result = case.tensor.to_cuda(cuda).unwrap();
    let actual = device(cuda, &device_result, &case.pairs);
    assert_eq!(
        actual.codomain_rank(),
        host.codomain_rank(),
        "{}",
        case.name
    );
    assert_eq!(actual.rank(), host.rank(), "{}", case.name);
    numerics::assert_nonzero_slices_close(
        case.name,
        actual.dense_data().unwrap(),
        host.dense_data().unwrap(),
        case.terms(),
    );
}

fn every_fixture<D: DevicePayload>(cuda: &mut dyn symtn::typed::Engine, runtime: &Runtime) {
    for case in u1_cases::<D>(runtime) {
        check(cuda, &case);
    }
    for case in su2_cases::<D>(runtime) {
        check(cuda, &case);
    }
    for case in u1_su2_cases::<D>(runtime) {
        check(cuda, &case);
    }
    for case in fermion_u1_cases::<D>(runtime) {
        check(cuda, &case);
    }
    for case in fermion_su2_cases::<D>(runtime) {
        check(cuda, &case);
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_traces_equal_the_host_at_every_dtype() {
    let cuda = &mut crate::common::cuda_engine();
    let runtime = Runtime::builder().build().unwrap();
    every_fixture::<f64>(cuda, &runtime);
    every_fixture::<Complex64>(cuda, &runtime);
    every_fixture::<f32>(cuda, &runtime);
    every_fixture::<Complex32>(cuda, &runtime);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn rejections_match_the_host_in_order_and_do_no_device_work() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut crate::common::cuda_engine();
    let runtime = Runtime::builder().build().unwrap();
    let case = &u1_cases::<f64>(&runtime)[0];
    let source = case.tensor.to_cuda(cuda).unwrap();
    let _ = source.trace_pairs(cuda, &case.pairs).unwrap();
    // Out of range, repeated, and a pair of two equal (not dual) codomain legs.
    let malformed: [&[(usize, usize)]; 4] = [&[(0, 5)], &[(0, 3), (3, 4)], &[(0, 0)], &[(0, 2)]];
    let (errors, counters) = delta(|| {
        malformed
            .iter()
            .map(|pairs| {
                let expected = case.tensor.trace_pairs(engine, pairs).unwrap_err().report();
                let actual = source.trace_pairs(cuda, pairs).unwrap_err().report();
                assert_eq!(actual, expected, "device error text must be the Host's");
                actual
            })
            .collect::<Vec<_>>()
    });
    assert_eq!(errors.len(), malformed.len());
    assert_eq!(
        counters,
        CudaTransferStats::default(),
        "a rejected trace must submit nothing: {counters:?}"
    );
    // An empty pair list is the identity, with no device work either.
    let (clone, counters) = delta(|| source.trace_pairs(cuda, &[]).unwrap());
    assert_eq!(counters, CudaTransferStats::default());
    assert_eq!(
        clone.to_host(cuda).unwrap().dense_data().unwrap(),
        case.tensor.dense_data().unwrap()
    );
}
