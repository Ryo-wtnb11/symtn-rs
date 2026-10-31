//! Real-device gates for `trace_pairs` on device tensors (issue
//! #1349): the Host trace structure replayed as one diagonal-view x ones
//! contraction per term.
//!
//! Evidence chain:
//!
//! 1. the region primitive is gated against an independent host strided loop
//!    in `symtn-dense/tests/cuda_region_trace.rs` (several pairs, conj, zero
//!    and complex scales, NaN propagation, rejections);
//! 2. the typed lowering is gated here: device == Host of the same public
//!    call, at every device payload dtype, owned and lazy-adjoint, for U(1),
//!    SU(2), U(1)xSU(2), fZ2xU(1) and fZ2(x)SU(2);
//! 3. independence: device == the physical-basis diagonal sum (U(1), SU(2))
//!    and device == the identity contraction (twist-free providers), both
//!    pinned against the Host by the ungated `typed_trace_host_oracle.rs`;
//!    device == the hand-valued fZ2 supertrace;
//! 4. accumulation: a destination with several producer blocks equals the
//!    sum of the traces of each producer alone;
//! 5. warm cost and rejections.
//!
//! CUDA transfer counters are thread-local, so these tests run concurrently.
//! Run with `cargo test -p symtn --no-default-features
//! --features cuda,cpu-faer --test integration typed_cuda_trace:: -- --ignored
//! --test-threads=1` on a CUDA host.

#![cfg(feature = "cuda")]

use crate::common;

use crate::contract_cases;
use crate::numerics;
use crate::report;
use crate::trace_cases;
use report::Report;

use std::sync::Arc;

use common::{DevicePayload, DeviceRule};
use contract_cases::{fill, u1};
use num_complex::{Complex32, Complex64};
use symtn::expert::{cuda_transfer_stats, CudaTransferStats};
use symtn::sector::PhysicalFusionBasis;
use symtn::sector::{FermionParityFusionRule, U1Irrep, Z2Irrep};
use symtn::typed::{GradedSpace, Runtime, TensorMap};
use trace_cases::{
    adjoint_pairs, dense_trace, fermion_su2_cases, fermion_u1_cases, su2_cases, u1_cases,
    u1_su2_cases, TraceCase,
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

/// Device == Host, and device == the identity contraction where it applies.
fn check<R: DeviceRule, D: DevicePayload>(
    engine: &mut dyn symtn::typed::Engine,
    cuda: &mut dyn symtn::typed::Engine,
    case: &TraceCase<R, D>,
) -> TensorMap<R, D> {
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
    if let Some(identity) = &case.identity {
        numerics::assert_nonzero_slices_close(
            case.name,
            actual.dense_data().unwrap(),
            identity.dense_data().unwrap(),
            case.terms(),
        );
    }
    // The same legs through a device lazy adjoint of the device parent.
    let pairs = adjoint_pairs(&case.tensor, &case.pairs);
    let host_lazy = case
        .tensor
        .adjoint(engine)
        .unwrap()
        .trace_pairs(engine, &pairs)
        .unwrap();
    let device_result = case.tensor.to_cuda(cuda).unwrap().adjoint(cuda).unwrap();
    let lazy = device(cuda, &device_result, &pairs);
    assert_eq!(
        lazy.codomain_rank(),
        host_lazy.codomain_rank(),
        "{} lazy",
        case.name
    );
    numerics::assert_nonzero_slices_close(
        case.name,
        lazy.dense_data().unwrap(),
        host_lazy.dense_data().unwrap(),
        case.terms(),
    );
    actual
}

fn check_dense<R, D>(
    cuda: &mut dyn symtn::typed::Engine,
    engine: &mut dyn symtn::typed::Engine,
    case: &TraceCase<R, D>,
) where
    R: DeviceRule + PhysicalFusionBasis<Scalar = f64>,
    D: DevicePayload,
{
    let actual = check(engine, cuda, case);
    if case.dense {
        let physical = actual.to_physical_dense(engine).unwrap().data;
        numerics::assert_nonzero_slices_close(
            case.name,
            &physical,
            &dense_trace(case),
            case.terms(),
        );
    }
}

fn every_fixture<D: DevicePayload>(
    engine: &mut dyn symtn::typed::Engine,
    cuda: &mut dyn symtn::typed::Engine,
    runtime: &Runtime,
) {
    for case in u1_cases::<D>(runtime) {
        check_dense(cuda, engine, &case);
    }
    for case in su2_cases::<D>(runtime) {
        check_dense(cuda, engine, &case);
    }
    for case in u1_su2_cases::<D>(runtime) {
        check(engine, cuda, &case);
    }
    for case in fermion_u1_cases::<D>(runtime) {
        check(engine, cuda, &case);
    }
    for case in fermion_su2_cases::<D>(runtime) {
        check(engine, cuda, &case);
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_traces_match_the_host_and_the_independent_oracles_at_every_dtype() {
    let cuda = &mut crate::common::cuda_engine();
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    every_fixture::<f64>(engine, cuda, &runtime);
    every_fixture::<Complex64>(engine, cuda, &runtime);
    every_fixture::<f32>(engine, cuda, &runtime);
    every_fixture::<Complex32>(engine, cuda, &runtime);
}

/// The fZ2 supertrace of `diag(2, 3 | 5, 6, 7)` with off-diagonal noise is
/// `(2 + 3) - (5 + 6 + 7) = -13` (the value `trace_macro.rs` pins for the
/// Host), while the ordinary sum would be `23`.
#[test]
#[ignore = "requires a real CUDA device"]
fn device_fz2_trace_is_the_hand_valued_supertrace() {
    let cuda = &mut crate::common::cuda_engine();
    let engine = &mut symtn::typed::CpuEngine::serial();
    fn at<D: DevicePayload>(
        engine: &mut dyn symtn::typed::Engine,
        cuda: &mut dyn symtn::typed::Engine,
        runtime: &Runtime,
    ) {
        let space = GradedSpace::try_new(
            Arc::new(FermionParityFusionRule),
            [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 3)],
        )
        .unwrap();
        let tensor: TensorMap<_, D> =
            TensorMap::from_subblock_fn(engine, runtime, [&space], [&space], |trees, index| {
                let value = if index[0] != index[1] {
                    9.0
                } else if *trees.coupled() == Z2Irrep::EVEN {
                    2.0 + index[0] as f64
                } else {
                    5.0 + index[0] as f64
                };
                D::entry(value, 0.0)
            })
            .unwrap();
        let device_result = tensor.to_cuda(cuda).unwrap();
        let traced = device(cuda, &device_result, &[(0, 1)]);
        numerics::assert_nonzero_slices_close(
            "fZ2 supertrace",
            traced.dense_data().unwrap(),
            &[D::entry(-13.0, 0.0)],
            8,
        );
    }
    let runtime = Runtime::builder().build().unwrap();
    at::<f64>(engine, cuda, &runtime);
    at::<Complex64>(engine, cuda, &runtime);
    at::<f32>(engine, cuda, &runtime);
    at::<Complex32>(engine, cuda, &runtime);
}

/// Every destination block of `trace(T, (0, 3))` over `v = (-1:2, 0:1, 1:2)`
/// has one producer per traced sector. The device result must be the *sum* of
/// the traces of `T` restricted to each traced sector, and at least two of
/// those overlap on one output entry — so a replay that overwrote instead of
/// accumulating would fail.
#[test]
#[ignore = "requires a real CUDA device"]
fn repeated_destinations_accumulate_every_producer() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut crate::common::cuda_engine();
    let runtime = Runtime::builder().build().unwrap();
    let v = u1(&[(-1, 2), (0, 1), (1, 2)]);
    let w = u1(&[(0, 1), (1, 2)]);
    let restricted = |engine: &mut dyn symtn::typed::Engine,
                      only: Option<i32>|
     -> TensorMap<_, f64> {
        let mut next = fill::<_, f64>(61);
        TensorMap::from_subblock_fn(
            engine,
            &runtime,
            [&v, &w, &v],
            [&v, &w],
            move |trees, index| {
                let value = next(trees, index);
                match only {
                    Some(charge) if trees.codomain_uncoupled()[0] != U1Irrep::new(charge) => 0.0,
                    _ => value,
                }
            },
        )
        .unwrap()
    };
    let pairs = [(0, 3)];
    let device_result = restricted(engine, None).to_cuda(cuda).unwrap();
    let full = device(cuda, &device_result, &pairs);
    let parts: Vec<Vec<f64>> = [-1, 0, 1]
        .into_iter()
        .map(|charge| {
            restricted(engine, Some(charge))
                .trace_pairs(engine, &pairs)
                .unwrap()
                .dense_data()
                .unwrap()
                .to_vec()
        })
        .collect();
    let sum: Vec<f64> = (0..full.dense_data().unwrap().len())
        .map(|index| parts.iter().map(|part| part[index]).sum())
        .collect();
    numerics::assert_nonzero_slices_close("sum of producers", full.dense_data().unwrap(), &sum, 64);
    let overlapping =
        (0..sum.len()).any(|index| parts.iter().filter(|part| part[index] != 0.0).count() >= 2);
    assert!(
        overlapping,
        "no output entry has two producers; the fixture proves nothing"
    );
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
