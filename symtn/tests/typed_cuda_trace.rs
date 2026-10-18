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
//! The contract tests read the process-wide transfer counters, hence
//! `--test-threads=1`. Run with `cargo test -p symtn --no-default-features
//! --features cuda,cpu-faer --test typed_cuda_trace -- --ignored
//! --test-threads=1` on a CUDA host.

#![cfg(feature = "cuda")]

mod common;
#[path = "../../tests/support"]
mod support {
    use num_complex::{Complex32, Complex64};
    pub mod numerics;
}
#[path = "support/report.rs"]
mod report;
use report::Report;
use support::numerics;
#[allow(unused_macros)] // the fermionic contraction fixture macro
mod contract_cases;
mod trace_cases;

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
    tensor: &TensorMap<R, D, symtn::typed::CudaStorage<D>>,
    pairs: &[(usize, usize)],
) -> TensorMap<R, D> {
    tensor.trace_pairs(pairs).unwrap().to_host().unwrap()
}

/// Device == Host, and device == the identity contraction where it applies.
fn check<R: DeviceRule, D: DevicePayload>(case: &TraceCase<R, D>) -> TensorMap<R, D> {
    let host = case.host();
    let actual = device(&case.tensor.to_cuda().unwrap(), &case.pairs);
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
    let host_lazy = case.tensor.adjoint().unwrap().trace_pairs(&pairs).unwrap();
    let lazy = device(&case.tensor.to_cuda().unwrap().adjoint().unwrap(), &pairs);
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

fn check_dense<R, D>(case: &TraceCase<R, D>)
where
    R: DeviceRule + PhysicalFusionBasis<Scalar = f64>,
    D: DevicePayload,
{
    let actual = check(case);
    if case.dense {
        let physical = actual.to_physical_dense().unwrap().data;
        numerics::assert_nonzero_slices_close(
            case.name,
            &physical,
            &dense_trace(case),
            case.terms(),
        );
    }
}

fn every_fixture<D: DevicePayload>(runtime: &Runtime) {
    for case in u1_cases::<D>(runtime) {
        check_dense(&case);
    }
    for case in su2_cases::<D>(runtime) {
        check_dense(&case);
    }
    for case in u1_su2_cases::<D>(runtime) {
        check(&case);
    }
    for case in fermion_u1_cases::<D>(runtime) {
        check(&case);
    }
    for case in fermion_su2_cases::<D>(runtime) {
        check(&case);
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_traces_match_the_host_and_the_independent_oracles_at_every_dtype() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    every_fixture::<f64>(&runtime);
    every_fixture::<Complex64>(&runtime);
    every_fixture::<f32>(&runtime);
    every_fixture::<Complex32>(&runtime);
}

/// The fZ2 supertrace of `diag(2, 3 | 5, 6, 7)` with off-diagonal noise is
/// `(2 + 3) - (5 + 6 + 7) = -13` (the value `trace_macro.rs` pins for the
/// Host), while the ordinary sum would be `23`.
#[test]
#[ignore = "requires a real CUDA device"]
fn device_fz2_trace_is_the_hand_valued_supertrace() {
    fn at<D: DevicePayload>(runtime: &Runtime) {
        let space = GradedSpace::try_new(
            Arc::new(FermionParityFusionRule),
            [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 3)],
        )
        .unwrap();
        let tensor: TensorMap<_, D> =
            TensorMap::from_subblock_fn(runtime, [&space], [&space], |trees, index| {
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
        let traced = device(&tensor.to_cuda().unwrap(), &[(0, 1)]);
        numerics::assert_nonzero_slices_close(
            "fZ2 supertrace",
            traced.dense_data().unwrap(),
            &[D::entry(-13.0, 0.0)],
            8,
        );
    }
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    at::<f64>(&runtime);
    at::<Complex64>(&runtime);
    at::<f32>(&runtime);
    at::<Complex32>(&runtime);
}

/// Every destination block of `trace(T, (0, 3))` over `v = (-1:2, 0:1, 1:2)`
/// has one producer per traced sector. The device result must be the *sum* of
/// the traces of `T` restricted to each traced sector, and at least two of
/// those overlap on one output entry — so a replay that overwrote instead of
/// accumulating would fail.
#[test]
#[ignore = "requires a real CUDA device"]
fn repeated_destinations_accumulate_every_producer() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let v = u1(&[(-1, 2), (0, 1), (1, 2)]);
    let w = u1(&[(0, 1), (1, 2)]);
    let restricted = |only: Option<i32>| -> TensorMap<_, f64> {
        let mut next = fill::<_, f64>(61);
        TensorMap::from_subblock_fn(&runtime, [&v, &w, &v], [&v, &w], move |trees, index| {
            let value = next(trees, index);
            match only {
                Some(charge) if trees.codomain_uncoupled()[0] != U1Irrep::new(charge) => 0.0,
                _ => value,
            }
        })
        .unwrap()
    };
    let pairs = [(0, 3)];
    let full = device(&restricted(None).to_cuda().unwrap(), &pairs);
    let parts: Vec<Vec<f64>> = [-1, 0, 1]
        .into_iter()
        .map(|charge| {
            restricted(Some(charge))
                .trace_pairs(&pairs)
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
fn a_warm_trace_uploads_only_its_output() {
    fn warm<R: DeviceRule>(runtime: &Runtime, case: &TraceCase<R, f64>) {
        let source = case.tensor.to_cuda().unwrap();
        let output_bytes = std::mem::size_of_val(case.host().dense_data().unwrap()) as u64;
        let call = || source.trace_pairs(&case.pairs).unwrap();
        let before = runtime.cuda_tree_transform_stats().unwrap();
        let (_, cold) = delta(call);
        let resident = runtime.cuda_tree_transform_stats().unwrap();
        assert!(
            resident.context_scalar_operand_bytes >= before.context_scalar_operand_bytes,
            "the ones template never shrinks"
        );
        let plans = runtime.cuda_plan_cache_stats().unwrap().unwrap();
        let (_, warm) = delta(call);
        let after_plans = runtime.cuda_plan_cache_stats().unwrap().unwrap();
        assert_eq!(warm.h2d_calls, 1, "{}: {warm:?}", case.name);
        assert_eq!(warm.h2d_bytes, output_bytes, "{}: {warm:?}", case.name);
        assert_eq!(warm.d2h_calls, 0, "{}: {warm:?}", case.name);
        assert_eq!(warm.device_allocs, 1, "{}: {warm:?}", case.name);
        assert!(warm.gemm_calls > 0, "{}: {warm:?}", case.name);
        assert_eq!(
            after_plans.misses, plans.misses,
            "{}: no new plan",
            case.name
        );
        assert_eq!(after_plans.evictions, plans.evictions, "{}", case.name);
        assert_eq!(
            runtime.cuda_tree_transform_stats().unwrap(),
            resident,
            "{}: a warm call grows no context operand",
            case.name
        );
        println!(
            "{}: output {output_bytes} B; cold {cold:?}; warm {warm:?}; \
             context operands {} B; plans {} -> {} misses",
            case.name, resident.context_scalar_operand_bytes, plans.misses, after_plans.misses
        );
    }
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    for case in u1_cases::<f64>(&runtime) {
        warm(&runtime, &case);
    }
    for case in su2_cases::<f64>(&runtime).iter().take(2) {
        warm(&runtime, case);
    }
    for case in fermion_su2_cases::<f64>(&runtime).iter().take(3) {
        warm(&runtime, case);
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn rejections_match_the_host_in_order_and_do_no_device_work() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let case = &u1_cases::<f64>(&runtime)[0];
    let source = case.tensor.to_cuda().unwrap();
    let _ = source.trace_pairs(&case.pairs).unwrap();
    let transforms = runtime.cuda_tree_transform_stats().unwrap();
    let plans = runtime.cuda_plan_cache_stats().unwrap().unwrap();
    // Out of range, repeated, and a pair of two equal (not dual) codomain legs.
    let malformed: [&[(usize, usize)]; 4] = [&[(0, 5)], &[(0, 3), (3, 4)], &[(0, 0)], &[(0, 2)]];
    let (errors, counters) = delta(|| {
        malformed
            .iter()
            .map(|pairs| {
                let expected = case.tensor.trace_pairs(pairs).unwrap_err().report();
                let actual = source.trace_pairs(pairs).unwrap_err().report();
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
    assert_eq!(runtime.cuda_tree_transform_stats().unwrap(), transforms);
    assert_eq!(runtime.cuda_plan_cache_stats().unwrap().unwrap(), plans);
    // An empty pair list is the identity, with no device work either.
    let (clone, counters) = delta(|| source.trace_pairs(&[]).unwrap());
    assert_eq!(counters, CudaTransferStats::default());
    assert_eq!(
        clone.to_host().unwrap().dense_data().unwrap(),
        case.tensor.dense_data().unwrap()
    );
}

/// A trace with more distinct term signatures than Tenferro's default bound
/// of 64 plans: `V` has nine charges of degeneracies 1..=9 and `W` nine of
/// its own, charges spaced so every coupled sector has one `(a, w)` pair, so
/// `trace(T, (0, 2))` of `T: V ⊗ W ← V ⊗ W` has 81 terms of distinct block
/// extents. The executor raises the plan bound by that count, so the second
/// call misses no plan.
#[test]
#[ignore = "requires a real CUDA device"]
fn a_warm_trace_past_the_default_plan_bound_rebuilds_no_plan() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let v = u1(&(0..9).map(|a| (a, a as usize + 1)).collect::<Vec<_>>());
    let w = u1(&(0..9)
        .map(|j| (100 * j, j as usize + 1))
        .collect::<Vec<_>>());
    let host: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&v, &w], [&v, &w], fill(71)).unwrap();
    assert_eq!(host.subblock_count(), 81);
    let source = host.to_cuda().unwrap();
    let cold = source.trace_pairs(&[(0, 2)]).unwrap();
    numerics::assert_nonzero_slices_close(
        "81-signature trace",
        cold.to_host().unwrap().dense_data().unwrap(),
        host.trace_pairs(&[(0, 2)]).unwrap().dense_data().unwrap(),
        64,
    );
    let plans = runtime.cuda_plan_cache_stats().unwrap().unwrap();
    let _ = source.trace_pairs(&[(0, 2)]).unwrap();
    let after = runtime.cuda_plan_cache_stats().unwrap().unwrap();
    assert!(plans.entries > 64, "{plans:?}");
    assert_eq!(after.misses, plans.misses, "{plans:?} -> {after:?}");
    assert_eq!(after.evictions, plans.evictions, "{plans:?} -> {after:?}");
}

/// The trace and the transform executor reserve plan entries independently,
/// and their reservations add: interleaving the 81-signature trace with a
/// device permute of the same tensor evicts no plan once both are warm.
#[test]
#[ignore = "requires a real CUDA device"]
fn an_interleaved_trace_and_transform_evict_no_plan() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let v = u1(&(0..9).map(|a| (a, a as usize + 1)).collect::<Vec<_>>());
    let w = u1(&(0..9)
        .map(|j| (100 * j, j as usize + 1))
        .collect::<Vec<_>>());
    let host: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&v, &w], [&v, &w], fill(71)).unwrap();
    let source = host.to_cuda().unwrap();
    let round = || {
        let _ = source.trace_pairs(&[(0, 2)]).unwrap();
        let _ = source.permute(&[1, 0], &[3, 2]).unwrap();
    };
    round();
    let plans = runtime.cuda_plan_cache_stats().unwrap().unwrap();
    let executor = runtime
        .cuda_tree_transform_stats()
        .unwrap()
        .required_plan_entries;
    assert!(
        plans.reserved_entries > executor,
        "the trace reserves beside the executor: {plans:?}, executor {executor}"
    );
    for _ in 0..3 {
        round();
    }
    let after = runtime.cuda_plan_cache_stats().unwrap().unwrap();
    assert_eq!(after.evictions, plans.evictions, "{plans:?} -> {after:?}");
    assert_eq!(after.misses, plans.misses, "{plans:?} -> {after:?}");
    assert_eq!(after.reserved_entries, plans.reserved_entries);
}
