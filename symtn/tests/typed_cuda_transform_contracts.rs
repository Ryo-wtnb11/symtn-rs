//! Transfer, allocation, statistics and rejection contracts of the typed
//! device structural transforms (issue #1322).
//!
//! These read the calling thread's `symtn::expert::cuda_transfer_stats`
//! counters, so device work another test submits concurrently does not
//! perturb their deltas.
//!
//! Every test here needs a real device. The Host half of the same contract —
//! a device-less Runtime reporting no device state and still clearing its
//! store — is in the ungated `typed_transform_host_side.rs`.
//!
//! Run with `cargo test -p symtn --features cuda,cpu-faer --test \
//! serial typed_cuda_transform_contracts:: -- --ignored`.

#![cfg(feature = "cuda")]

/// TensorKit's argument-free `transpose(t)`: the full planar rotation, which
/// carries every codomain leg across the boundary and every domain leg back.
macro_rules! full_rotation {
    ($tensor:expr) => {{
        let tensor = &$tensor;
        let codomain_rank = tensor.codomain_rank();
        let codomain_axes: Vec<usize> = (codomain_rank..tensor.rank()).rev().collect();
        let domain_axes: Vec<usize> = (0..codomain_rank).rev().collect();
        (codomain_axes, domain_axes)
    }};
}

macro_rules! full_transpose_into {
    ($tensor:expr, $destination:expr, $alpha:expr) => {{
        let tensor = &$tensor;
        let (codomain_axes, domain_axes) = full_rotation!(tensor);
        tensor.transpose_into(
            &codomain_axes,
            &domain_axes,
            $destination,
            $alpha,
            Default::default(),
        )
    }};
}

macro_rules! full_transpose {
    ($tensor:expr) => {{
        let tensor = &$tensor;
        let (codomain_axes, domain_axes) = full_rotation!(tensor);
        tensor.transpose(&codomain_axes, &domain_axes)
    }};
}

use crate::report;
use report::Report;
use std::sync::Arc;
use symtn::typed::Direction;

use symtn::expert::{cuda_transfer_stats, CudaTransferStats};
use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::Error;
use symtn::typed::{GradedSpace, Runtime, TensorMap};

/// Orders the local process-global cache probes; Runtime-local probes run concurrently.
static CACHE_TEST_LOCK: std::sync::RwLock<()> = std::sync::RwLock::new(());

fn cache_exclusive() -> std::sync::RwLockWriteGuard<'static, ()> {
    CACHE_TEST_LOCK.write().unwrap_or_else(|e| e.into_inner())
}

fn leg() -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 1),
            (U1Irrep::new(1), 2),
        ],
    )
    .unwrap()
}

fn fixture(runtime: &Runtime) -> TensorMap<U1FusionRule, f64> {
    let v = leg();
    TensorMap::from_subblock_fn(runtime, [&v, &v], [&v, &v], |_, indices| {
        indices.iter().map(|&i| i as f64 + 1.0).sum::<f64>()
    })
    .unwrap()
}

/// Counter deltas across one closure.
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

// ---------------------------------------------------------------------------
// Device contracts
// ---------------------------------------------------------------------------

#[test]
#[ignore = "requires a real CUDA device"]
fn a_warm_device_transform_uploads_only_its_output_and_downloads_nothing() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let device = fixture(&runtime).to_cuda().unwrap();
    // Sized on the Host, so the device's first call below really is its cold
    // one: measuring the output on device would have prepared the structure.
    let output_bytes = std::mem::size_of_val(
        fixture(&runtime)
            .permute(&[2, 0], &[1, 3])
            .unwrap()
            .dense_data()
            .unwrap(),
    ) as u64;

    // Cold: the output plus exactly one coefficient payload for the structure
    // — a Single-block permute needs no pack/scatter workspace, so those two
    // uploads are the whole of it.
    let cold_stats_before = runtime.cuda_tree_transform_stats().unwrap();
    let (_, cold) = delta(|| device.permute(&[2, 0], &[1, 3]).unwrap());
    let cold_stats_after = runtime.cuda_tree_transform_stats().unwrap();
    assert_eq!(cold.h2d_calls, 2, "{cold:?}");
    assert_eq!(cold.device_allocs, 2, "{cold:?}");
    assert_eq!(cold_stats_after.workspace_bytes, 0, "{cold_stats_after:?}");
    assert_eq!(cold.d2h_calls, 0, "no device transform downloads: {cold:?}");
    assert_eq!(
        cold_stats_after.prepared_structures,
        cold_stats_before.prepared_structures + 1,
        "exactly one device entry is prepared per structure"
    );

    // Warm: exactly one H2D of the output bytes, one device allocation, no
    // download. This is the #740 constant and nothing else.
    let (_, warm) = delta(|| device.permute(&[2, 0], &[1, 3]).unwrap());
    assert_eq!(warm.h2d_calls, 1, "{warm:?}");
    assert_eq!(warm.h2d_bytes, output_bytes, "{warm:?}");
    assert_eq!(warm.d2h_calls, 0, "{warm:?}");
    assert_eq!(warm.d2h_bytes, 0, "{warm:?}");
    assert_eq!(warm.device_allocs, 1, "{warm:?}");
    let warm_stats = runtime.cuda_tree_transform_stats().unwrap();
    assert_eq!(
        warm_stats.prepared_structures, cold_stats_after.prepared_structures,
        "a warm replay prepares nothing new"
    );
    assert_eq!(
        warm_stats.executor_bytes, cold_stats_after.executor_bytes,
        "a warm replay grows no device state"
    );
    assert!(
        warm_stats.required_plan_entries > 0,
        "the prepared structure declares its plan signatures"
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn clearing_the_cuda_caches_releases_only_the_device_executor_state() {
    let _serial = crate::counting_alloc::serial();
    let _cache = cache_exclusive();
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let device = fixture(&runtime).to_cuda().unwrap();
    let _ = device.permute(&[2, 0], &[1, 3]).unwrap();
    let _ = full_transpose!(device).unwrap();
    let before = runtime.cuda_tree_transform_stats().unwrap();
    assert!(before.prepared_structures >= 2, "{before:?}");
    assert!(before.executor_bytes > 0, "{before:?}");
    let semantic_before = completed_transformers().entries();
    assert!(semantic_before > 0);

    runtime.clear_cuda_caches();

    let after = runtime.cuda_tree_transform_stats().unwrap();
    assert_eq!(after.prepared_structures, 0, "{after:?}");
    assert_eq!(after.executor_bytes, 0, "{after:?}");
    assert_eq!(after.workspace_bytes, 0, "{after:?}");
    assert_eq!(after.required_plan_entries, 0, "{after:?}");
    // The shared context operands belong to the context, not the executor, and
    // are reported separately — clearing the executor does not release them.
    assert_eq!(
        after.context_scalar_operand_bytes, before.context_scalar_operand_bytes,
        "context operands are counted once and are not the executor's to drop"
    );
    // Device state only: the process-global semantic caches are untouched.
    assert_eq!(completed_transformers().entries(), semantic_before);

    // Re-preparing after the clear still produces the same answer.
    let expected = fixture(&runtime).permute(&[2, 0], &[1, 3]).unwrap();
    let again = device.permute(&[2, 0], &[1, 3]).unwrap().to_host().unwrap();
    assert_eq!(again.dense_data().unwrap(), expected.dense_data().unwrap());
}

#[test]
#[ignore = "requires a real CUDA device"]
fn clearing_and_re_preparing_does_not_creep_the_plan_reservation() {
    let _serial = crate::counting_alloc::serial();
    // What: `clear_cuda_caches` returns the executor's plan-entry
    // reservation, so repeated clear/re-prepare cycles of the same structures
    // leave the ledger and the cap where the first preparation put them.
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let device = fixture(&runtime).to_cuda().unwrap();
    let prepare = || {
        let _ = device.permute(&[2, 0], &[1, 3]).unwrap();
        let _ = full_transpose!(device).unwrap();
    };
    prepare();
    let reserved = runtime
        .cuda_plan_cache_stats()
        .unwrap()
        .unwrap()
        .reserved_entries;
    let required = runtime
        .cuda_tree_transform_stats()
        .unwrap()
        .required_plan_entries;
    assert!(reserved > 0);
    assert_eq!(reserved, required);

    for cycle in 0..3 {
        runtime.clear_cuda_caches();
        let cleared = runtime.cuda_plan_cache_stats().unwrap().unwrap();
        assert_eq!(cleared.reserved_entries, 0, "cycle {cycle}");
        prepare();
        let again = runtime.cuda_plan_cache_stats().unwrap().unwrap();
        assert_eq!(again.reserved_entries, reserved, "cycle {cycle}");
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_transform_short_circuits_do_no_device_work() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let device = fixture(&runtime).to_cuda().unwrap();
    let expected = fixture(&runtime).dense_data().unwrap().to_vec();

    let (results, counters) = delta(|| {
        [
            // Identity axis lists, the same split, and a rank-0 transpose all
            // return a clone of the receiver.
            device.permute(&[0, 1], &[2, 3]).unwrap(),
            device.transpose(&[0, 1], &[2, 3]).unwrap(),
            device.braid(&[0, 1], &[2, 3], &[1, 2, 3, 4]).unwrap(),
            device.repartition(2).unwrap(),
        ]
    });
    assert_eq!(
        counters,
        CudaTransferStats::default(),
        "a short-circuited transform must submit nothing: {counters:?}"
    );
    for result in results {
        assert_eq!(result.to_host().unwrap().dense_data().unwrap(), expected);
    }

    // A rank-0 tensor has no legs to build from, so it comes from a full Host
    // trace (this file does not test device `trace`) and is then uploaded.
    let v = leg();
    let endomorphism: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&v], [&v], |_, indices| indices[0] as f64 + 1.0)
            .unwrap();
    let rank_zero = endomorphism.trace_pairs(&[(0, 1)]).unwrap();
    assert_eq!(rank_zero.rank(), 0);
    let device_zero = rank_zero.to_cuda().unwrap();
    let (transposed, counters) = delta(|| full_transpose!(device_zero).unwrap());
    assert_eq!(counters, CudaTransferStats::default(), "{counters:?}");
    assert_eq!(
        transposed.to_host().unwrap().dense_data().unwrap(),
        rank_zero.dense_data().unwrap()
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_transform_rejections_happen_before_any_device_work() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let device = fixture(&runtime).to_cuda().unwrap();
    // Warm the structures first, so a rejection cannot be confused with the
    // cold-path uploads of a neighbouring call.
    let _ = device.permute(&[2, 0], &[1, 3]).unwrap();
    let before = runtime.cuda_tree_transform_stats().unwrap();

    let host = fixture(&runtime);
    // Each rejection is written once and expanded against both receivers, so
    // the device error *mapping* is gated, not merely "an error came back".
    macro_rules! rejects_like_host {
        (|$t:ident| $call:expr) => {{
            let expected = {
                let $t = &host;
                $call
            }
            .unwrap_err()
            .report();
            let actual = {
                let $t = &device;
                $call
            }
            .unwrap_err()
            .report();
            assert_eq!(actual, expected, "device error text must be the Host's");
            actual
        }};
    }

    let (errors, counters) = delta(|| {
        [
            // The braid levels-length check precedes the identity detection,
            // exactly as on Host: identity axes with a short levels list is an
            // error, not a clone.
            rejects_like_host!(|t| t.braid(&[0, 1], &[2, 3], &[1, 2, 3])),
            rejects_like_host!(|t| t.braid(&[2, 0], &[1, 3], &[1, 2, 3, 4, 5])),
            // Malformed axes come back from the expert layer.
            rejects_like_host!(|t| t.permute(&[0, 0], &[2, 3])),
            // A non-planar re-arrangement is refused rather than braided.
            rejects_like_host!(|t| t.transpose(&[1, 2], &[3, 0])),
            // A split beyond the rank has no planar reading.
            rejects_like_host!(|t| t.repartition(5)),
        ]
    });
    assert!(
        errors[0].contains("one level per source axis"),
        "{}",
        errors[0]
    );
    assert!(errors[0].contains("expected 4"), "{}", errors[0]);
    assert!(errors[4].contains("exceeds rank 4"), "{}", errors[4]);
    assert_eq!(
        counters,
        CudaTransferStats::default(),
        "a rejected transform must submit nothing: {counters:?}"
    );
    assert_eq!(runtime.cuda_tree_transform_stats().unwrap(), before);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_cold_device_transform_uploads_one_coefficient_payload_per_structure() {
    let _serial = crate::counting_alloc::serial();
    // SU(2) recouples, so the structure has a real coefficient payload and a
    // pack/scatter workspace. Both are uploaded once, on the first replay of
    // that structure, and never again while the Host store admits it.
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let v = GradedSpace::try_new(
        Arc::new(symtn::sector::SU2FusionRule),
        [
            (symtn::sector::SU2Irrep::from_twice_spin(0), 2),
            (symtn::sector::SU2Irrep::from_twice_spin(1), 2),
            (symtn::sector::SU2Irrep::from_twice_spin(2), 1),
        ],
    )
    .unwrap();
    let host: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&v, &v], [&v, &v], |_, indices| {
            indices.iter().map(|&i| i as f64 + 1.0).sum::<f64>()
        })
        .unwrap();
    let output_bytes = std::mem::size_of_val(
        host.permute(&[1, 2], &[3, 0])
            .unwrap()
            .dense_data()
            .unwrap(),
    ) as u64;
    let device = host.to_cuda().unwrap();

    let before = runtime.cuda_tree_transform_stats().unwrap();
    let (_, cold) = delta(|| device.permute(&[1, 2], &[3, 0]).unwrap());
    let after = runtime.cuda_tree_transform_stats().unwrap();
    assert!(
        cold.h2d_calls >= 2,
        "a cold recoupling replay uploads the output and its coefficient payload: {cold:?}"
    );
    assert_eq!(cold.d2h_calls, 0, "{cold:?}");
    assert_eq!(after.prepared_structures, before.prepared_structures + 1);
    assert!(
        after.executor_bytes > before.executor_bytes,
        "the coefficient payload and workspace are retained: {before:?} -> {after:?}"
    );
    assert!(after.workspace_bytes > 0, "{after:?}");

    // Warm: the #740 output initialisation and nothing else. No second payload,
    // no workspace growth.
    let (_, warm) = delta(|| device.permute(&[1, 2], &[3, 0]).unwrap());
    assert_eq!(warm.h2d_calls, 1, "{warm:?}");
    assert_eq!(warm.h2d_bytes, output_bytes, "{warm:?}");
    assert_eq!(warm.d2h_calls, 0, "{warm:?}");
    assert_eq!(warm.device_allocs, 1, "{warm:?}");
    assert_eq!(runtime.cuda_tree_transform_stats().unwrap(), after);
}

// ---------------------------------------------------------------------------
// `*_into` contracts (issue #1329)
// ---------------------------------------------------------------------------

/// The device wording of a Host message. The device mirrors Host's error
/// variants, messages and order; only the storage noun names the placement,
/// as in `contract_into`. That these are the exact
/// Host strings is pinned without a device in `typed_transform_host_side.rs`.
fn as_device_message(host: &str) -> String {
    host.replace("ordinary dense host storage", "ordinary dense CUDA storage")
}

/// The warm contract: no transfer in either direction and no device
/// allocation. Kernel submission counts (`gemm_calls`, `copy_calls`) are not
/// transfers and are expected to be non-zero.
fn assert_transfer_free(stats: &CudaTransferStats) {
    assert_eq!(stats.h2d_calls, 0, "{stats:?}");
    assert_eq!(stats.h2d_bytes, 0, "{stats:?}");
    assert_eq!(stats.d2h_calls, 0, "{stats:?}");
    assert_eq!(stats.d2h_bytes, 0, "{stats:?}");
    assert_eq!(stats.device_allocs, 0, "{stats:?}");
}

/// A device payload as raw bits, so a NaN-poisoned destination compares equal
/// to itself.
fn payload_bits<R>(tensor: &TensorMap<R, f64, symtn::typed::CudaStorage>) -> Vec<u64>
where
    R: symtn::sector::MultiplicityFreeRigidSymbols<Scalar = f64>
        + symtn::sector::CheckedFusionAlgebra
        + symtn::sector::SectorCodec,
{
    tensor
        .to_host()
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .map(|value| value.to_bits())
        .collect()
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_warm_device_into_transfers_nothing_and_allocates_nothing() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let source = fixture(&runtime).to_cuda().unwrap();
    let mut destination = fixture(&runtime)
        .permute(&[2, 0], &[1, 3])
        .unwrap()
        .to_cuda()
        .unwrap();

    // Cold: prepares the structure for this pair.
    source
        .permute_into(&[2, 0], &[1, 3], &mut destination, 1.0, 0.0)
        .unwrap();
    let cold = runtime.cuda_tree_transform_stats().unwrap();

    // Warm: the destination is the caller's, so unlike the returning
    // `permute` there is no #740 output initialisation left to pay.
    let (_, warm) = delta(|| {
        source
            .permute_into(&[2, 0], &[1, 3], &mut destination, -2.5, 0.0)
            .unwrap()
    });
    assert_eq!(warm.h2d_calls, 0, "{warm:?}");
    assert_eq!(warm.h2d_bytes, 0, "{warm:?}");
    assert_eq!(warm.d2h_calls, 0, "{warm:?}");
    assert_eq!(warm.d2h_bytes, 0, "{warm:?}");
    assert_eq!(warm.device_allocs, 0, "{warm:?}");
    assert_eq!(
        runtime.cuda_tree_transform_stats().unwrap(),
        cold,
        "a warm overwrite grows no device state"
    );

    // A zero caller scale writes zero fills over the written layouts
    // (#1438), reading the context's zero template, so the *first* zero-scale
    // replay may size that template to the largest written layout and create
    // the `1` operand — at most two uploads bounded by the destination, once
    // per context, not per call. Warm, it transfers nothing.
    let (_, first_zero) = delta(|| {
        source
            .permute_into(&[2, 0], &[1, 3], &mut destination, 0.0, 0.0)
            .unwrap()
    });
    let destination_len = fixture(&runtime)
        .permute(&[2, 0], &[1, 3])
        .unwrap()
        .dense_data()
        .unwrap()
        .len();
    let bound = ((destination_len + 1) * std::mem::size_of::<f64>()) as u64;
    assert!(
        first_zero.h2d_calls <= 2 && first_zero.h2d_bytes <= bound && first_zero.device_allocs <= 2,
        "the zero template is bounded by the destination and uploaded once: {first_zero:?}"
    );
    let (_, zero) = delta(|| {
        source
            .permute_into(&[2, 0], &[1, 3], &mut destination, -0.0, 0.0)
            .unwrap()
    });
    assert_eq!(zero.h2d_calls, 0, "{zero:?}");
    assert_eq!(zero.d2h_calls, 0, "{zero:?}");
    assert_eq!(zero.device_allocs, 0, "{zero:?}");

    // And the other three methods are warm on their own structures.
    let mut transposed = full_transpose!(fixture(&runtime))
        .unwrap()
        .to_cuda()
        .unwrap();
    full_transpose_into!(source, &mut transposed, 1.0).unwrap();
    let (_, warm) = delta(|| full_transpose_into!(source, &mut transposed, 1.0).unwrap());
    assert_transfer_free(&warm);

    let mut bent = fixture(&runtime).repartition(1).unwrap().to_cuda().unwrap();
    source.repartition_into(&mut bent, 1.0, 0.0).unwrap();
    let (_, warm) = delta(|| source.repartition_into(&mut bent, 1.0, 0.0).unwrap());
    assert_transfer_free(&warm);

    let mut cyclic = fixture(&runtime)
        .transpose(&[1, 3], &[0, 2])
        .unwrap()
        .to_cuda()
        .unwrap();
    source
        .transpose_into(&[1, 3], &[0, 2], &mut cyclic, 1.0, 0.0)
        .unwrap();
    let (_, warm) = delta(|| {
        source
            .transpose_into(&[1, 3], &[0, 2], &mut cyclic, 1.0, 0.0)
            .unwrap()
    });
    assert_transfer_free(&warm);

    // A lazy-adjoint source (#1870) is read through its parent's
    // device payload: warm, it materializes nothing and transfers nothing.
    let lazy = source.adjoint().unwrap();
    let mut from_lazy = fixture(&runtime)
        .adjoint()
        .unwrap()
        .permute(&[2, 0], &[1, 3])
        .unwrap()
        .materialize()
        .unwrap()
        .to_cuda()
        .unwrap();
    lazy.permute_into(&[2, 0], &[1, 3], &mut from_lazy, 1.0, 0.0)
        .unwrap();
    let (_, warm) = delta(|| {
        lazy.permute_into(&[2, 0], &[1, 3], &mut from_lazy, 1.0, 0.5)
            .unwrap()
    });
    assert_transfer_free(&warm);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_into_admits_the_exact_layout_on_the_shared_runtime_store() {
    let _serial = crate::counting_alloc::serial();
    let _cache = cache_exclusive();
    // Host `overwrite_tree_transform` admits the source/destination layout
    // pair after a successful replay, so a second call resolves the operation
    // out of the store instead of deriving it. The store is the Runtime's own
    // Host store — the device path must feed and reuse exactly that one.
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let host = fixture(&runtime);
    let expected = host.permute(&[2, 0], &[1, 3]).unwrap();
    let source = host.to_cuda().unwrap();

    symtn::cache::clear();
    let mut first = expected.to_cuda().unwrap();
    source
        .permute_into(&[2, 0], &[1, 3], &mut first, 1.0, 0.0)
        .unwrap();
    let cold = completed_transformers();

    let mut second = expected.to_cuda().unwrap();
    source
        .permute_into(&[2, 0], &[1, 3], &mut second, 1.0, 0.0)
        .unwrap();
    let warm = completed_transformers();
    assert_eq!(
        warm.entries(),
        cold.entries(),
        "the admitted pair adds no entry"
    );
    assert!(
        warm.hits() > cold.hits(),
        "the second call must take the admitted path: {cold:?} -> {warm:?}"
    );
    assert_eq!(
        first.to_host().unwrap().dense_data().unwrap(),
        second.to_host().unwrap().dense_data().unwrap()
    );

    // Admission is keyed on the layout pair, so a destination that is *not*
    // this operation's result still misses the lookup and is still rejected.
    let mut wrong = full_transpose!(host).unwrap().to_cuda().unwrap();
    let error = source
        .permute_into(&[2, 0], &[1, 3], &mut wrong, 1.0, 0.0)
        .unwrap_err();
    assert!(
        matches!(&error, symtn::typed::Error::Operation(operation)
            if matches!(**operation, symtn::typed::OperationError::SpaceMismatch { .. })),
        "{error}"
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_into_rejections_happen_before_any_device_work() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let host = fixture(&runtime);
    let expected = host.permute(&[2, 0], &[1, 3]).unwrap();
    let source = host.to_cuda().unwrap();
    // Fresh, uniquely owned destinations on demand: a `clone()` would share
    // the payload `Arc` and trip the unique-ownership check instead.
    let host_destination = || host.permute(&[2, 0], &[1, 3]).unwrap();
    let device_destination = || expected.to_cuda().unwrap();

    // Warm the structure first, so a rejection cannot be confused with the
    // cold-path uploads of a neighbouring call.
    {
        let mut warm = device_destination();
        source
            .permute_into(&[2, 0], &[1, 3], &mut warm, 1.0, 0.0)
            .unwrap();
    }
    let stats_before = runtime.cuda_tree_transform_stats().unwrap();

    // Asserts one device rejection: the Host's error text (with the placement
    // noun), no transfer, allocation or executor-state change, and a
    // byte-identical witness — the owned device tensor whose payload the
    // destination is (itself, or, for a lazy destination, its parent).
    macro_rules! rejects_like_host {
        ($what:expr, $host_call:expr, $witness:ident, $device_call:expr) => {{
            let expected_message = as_device_message(&$host_call.unwrap_err().report());
            // Bitwise: the poisoned destinations are NaN, which is not equal
            // to itself.
            let before = payload_bits(&$witness);
            let (error, counters) = delta(|| $device_call.unwrap_err());
            assert_eq!(error.report(), expected_message, "{}", $what);
            assert_eq!(
                counters,
                CudaTransferStats::default(),
                "{}: a rejected overwrite must submit nothing: {counters:?}",
                $what
            );
            assert_eq!(
                payload_bits(&$witness),
                before,
                "{}: the caller's destination must be untouched",
                $what
            );
            error
        }};
    }

    // `permute_into` takes `&mut Self`, so a rule mismatch is only
    // reachable between two *instances* of the same rule type — Z2 against
    // Z3, exactly as the Host unit test builds it.
    let z2 = Arc::new(symtn::sector::ZNFusionRule::new(2).unwrap());
    let z3 = Arc::new(symtn::sector::ZNFusionRule::new(3).unwrap());
    let z2_leg = GradedSpace::try_new(Arc::clone(&z2), [(z2.irrep(0), 2)]).unwrap();
    let z3_leg = GradedSpace::try_new(Arc::clone(&z3), [(z3.irrep(0), 2)]).unwrap();
    let zn_fill = |_: &_, indices: &[usize]| indices.iter().map(|&i| i as f64 + 1.0).sum::<f64>();
    let z2_host = TensorMap::from_subblock_fn(&runtime, [&z2_leg], [&z2_leg], zn_fill).unwrap();
    let z2_source = z2_host.to_cuda().unwrap();

    // 1. Runtime mismatch wins over a rule mismatch: a Z3 destination on
    //    another Runtime.
    let other = Runtime::builder().cuda(0).build().unwrap();
    let mut host_foreign =
        TensorMap::from_subblock_fn(&other, [&z3_leg], [&z3_leg], |_, _| f64::NAN).unwrap();
    let mut foreign = host_foreign.to_cuda().unwrap();
    {
        let witness = foreign.clone();
        let error = rejects_like_host!(
            "runtime mismatch precedes rule mismatch",
            z2_host.permute_into(&[0], &[1], &mut host_foreign, 1.0, 0.0),
            witness,
            z2_source.permute_into(&[0], &[1], &mut foreign, 1.0, 0.0)
        );
        assert_eq!(error, Error::RuntimeMismatch);
    }

    // 2. Rule mismatch wins over a lazy-adjoint source.
    let mut host_z3 =
        TensorMap::from_subblock_fn(&runtime, [&z3_leg], [&z3_leg], |_, _| f64::NAN).unwrap();
    let mut z3_destination = host_z3.to_cuda().unwrap();
    let lazy_source = source.adjoint().unwrap();
    let host_lazy_source = host.adjoint().unwrap();
    let z2_lazy_source = z2_source.adjoint().unwrap();
    let z2_host_lazy_source = z2_host.adjoint().unwrap();
    {
        let witness = z3_destination.clone();
        let error = rejects_like_host!(
            "rule mismatch precedes the lazy-adjoint source",
            z2_host_lazy_source.permute_into(&[0], &[1], &mut host_z3, 1.0, 0.0),
            witness,
            z2_lazy_source.permute_into(&[0], &[1], &mut z3_destination, 1.0, 0.0)
        );
        assert_eq!(error, Error::RuleMismatch);
    }

    // 3. A lazy-adjoint source is admitted (read through its parent, #1870),
    //    so with a lazy-adjoint destination the destination is reported.
    let lazy_parent = device_destination();
    let mut lazy_destination = lazy_parent.adjoint().unwrap();
    let mut host_lazy_destination = host_destination().adjoint().unwrap();
    let error = rejects_like_host!(
        "a lazy-adjoint source with a lazy-adjoint destination",
        host_lazy_source.permute_into(&[2, 0], &[1, 3], &mut host_lazy_destination, 1.0, 0.0),
        lazy_parent,
        lazy_source.permute_into(&[2, 0], &[1, 3], &mut lazy_destination, 1.0, 0.0)
    );
    assert!(
        error.report().contains("ordinary dense CUDA storage"),
        "{error}"
    );

    // 4. A lazy-adjoint destination alone.
    let mut host_lazy_destination = host_destination().adjoint().unwrap();
    let error = rejects_like_host!(
        "lazy-adjoint destination",
        host.permute_into(&[2, 0], &[1, 3], &mut host_lazy_destination, 1.0, 0.0),
        lazy_parent,
        source.permute_into(&[2, 0], &[1, 3], &mut lazy_destination, 1.0, 0.0)
    );
    assert!(
        error.report().contains("ordinary dense CUDA storage"),
        "{error}"
    );

    // 5. Alias through a clone: `TensorMap` clones share the payload `Arc`, so
    //    source and destination can be the same device allocation. The alias
    //    check precedes the operation build, so malformed axes do not mask it.
    let mut alias = source.clone();
    let mut host_alias = host.clone();
    {
        let witness = source.clone();
        let error = rejects_like_host!(
            "the alias check precedes the operation build",
            host.permute_into(&[0, 0], &[1, 3], &mut host_alias, 1.0, 0.0),
            witness,
            source.permute_into(&[0, 0], &[1, 3], &mut alias, 1.0, 0.0)
        );
        assert!(error.report().contains("must not alias"), "{error}");
    }

    // 5b. A lazy-adjoint source aliasing its parent's payload: the alias
    //     check runs against the source's stored body (#1870).
    let mut parent_alias = source.clone();
    let mut host_parent_alias = host.clone();
    {
        let witness = source.clone();
        let error = rejects_like_host!(
            "a lazy source aliasing its parent",
            host_lazy_source.permute_into(&[2, 0], &[1, 3], &mut host_parent_alias, 1.0, 0.0),
            witness,
            lazy_source.permute_into(&[2, 0], &[1, 3], &mut parent_alias, 1.0, 0.0)
        );
        assert!(error.report().contains("must not alias"), "{error}");
    }

    // 6. A destination on the wrong space, held shared: the space check
    //    precedes the unique-ownership check.
    let mut wrong_space = full_transpose!(host).unwrap().to_cuda().unwrap();
    let wrong_space_handle = wrong_space.clone();
    let mut host_wrong_space = full_transpose!(host).unwrap();
    let host_wrong_space_handle = host_wrong_space.clone();
    {
        let witness = wrong_space.clone();
        let error = rejects_like_host!(
            "the space check precedes unique ownership",
            host.permute_into(&[2, 0], &[1, 3], &mut host_wrong_space, 1.0, 0.0),
            witness,
            source.permute_into(&[2, 0], &[1, 3], &mut wrong_space, 1.0, 0.0)
        );
        assert!(
            matches!(&error, symtn::typed::Error::Operation(operation)
                if matches!(**operation, symtn::typed::OperationError::SpaceMismatch { .. })),
            "{error}"
        );
    }
    drop(wrong_space_handle);
    drop(host_wrong_space_handle);

    // 7. A shared destination on the right space.
    let mut shared = device_destination();
    let shared_handle = shared.clone();
    let mut host_shared = host_destination();
    let host_shared_handle = host_shared.clone();
    {
        let witness = shared.clone();
        let error = rejects_like_host!(
            "shared destination",
            host.permute_into(&[2, 0], &[1, 3], &mut host_shared, 1.0, 0.0),
            witness,
            source.permute_into(&[2, 0], &[1, 3], &mut shared, 1.0, 0.0)
        );
        assert!(error.report().contains("shares its storage"), "{error}");
    }
    drop(shared_handle);
    drop(host_shared_handle);

    // 8. Malformed axes, a non-planar cyclic request and a `repartition`
    //    destination of another rank all come back from the operation build,
    //    before the lease.
    let mut destination = device_destination();
    let mut host_dst = host_destination();
    let witness = destination.clone();
    rejects_like_host!(
        "malformed axes",
        host.permute_into(&[0, 0], &[1, 3], &mut host_dst, 1.0, 0.0),
        witness,
        source.permute_into(&[0, 0], &[1, 3], &mut destination, 1.0, 0.0)
    );
    rejects_like_host!(
        "non-planar transpose",
        host.transpose_into(&[1, 2], &[3, 0], &mut host_dst, 1.0, 0.0),
        witness,
        source.transpose_into(&[1, 2], &[3, 0], &mut destination, 1.0, 0.0)
    );

    let v = leg();
    let mut host_rank_three =
        TensorMap::from_subblock_fn(&runtime, [&v, &v], [&v], |_, _| f64::NAN).unwrap();
    let mut rank_three = host_rank_three.to_cuda().unwrap();
    {
        let witness = rank_three.clone();
        let error = rejects_like_host!(
            "repartition destination of another rank",
            host.repartition_into(&mut host_rank_three, 1.0, 0.0),
            witness,
            source.repartition_into(&mut rank_three, 1.0, 0.0)
        );
        assert!(
            error.report().contains("does not match source rank"),
            "{error}"
        );
    }

    assert_eq!(
        runtime.cuda_tree_transform_stats().unwrap(),
        stats_before,
        "no rejection may change the device executor state"
    );
}

// ---------------------------------------------------------------------------
// `twist` in both directions contracts (issue #1330)
// ---------------------------------------------------------------------------

/// fZ2 with both parities on every leg, so a twist really scales blocks.
fn fermionic_fixture(runtime: &Runtime) -> TensorMap<symtn::sector::FermionParityFusionRule, f64> {
    let leg = GradedSpace::try_new(
        Arc::new(symtn::sector::FermionParityFusionRule),
        [
            (symtn::sector::Z2Irrep::EVEN, 2),
            (symtn::sector::Z2Irrep::ODD, 1),
        ],
    )
    .unwrap();
    TensorMap::from_subblock_fn(runtime, [&leg, &leg], [&leg, &leg], |_, indices| {
        indices.iter().map(|&i| i as f64 + 1.0).sum::<f64>()
    })
    .unwrap()
}

#[test]
#[ignore = "requires a real CUDA device"]
fn a_warm_device_twist_uploads_only_its_output_and_downloads_nothing() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let host = fermionic_fixture(&runtime);
    let device = host.to_cuda().unwrap();
    let output_bytes = std::mem::size_of_val(host.dense_data().unwrap()) as u64;
    let blocks = host.subblock_count() as u64;
    assert!(blocks > 1, "multi-block fixture");

    // A twist compiles no structure, so its only cold cost is the context's
    // shared `1` coefficient operand, created once per dtype per context.
    let before = runtime.cuda_tree_transform_stats().unwrap();
    let _ = device.twist(&[0, 3], Direction::Forward).unwrap();
    let after = runtime.cuda_tree_transform_stats().unwrap();
    assert_eq!(
        after.prepared_structures, before.prepared_structures,
        "a twist is not a tree transform and prepares no structure"
    );
    assert_eq!(after.executor_bytes, before.executor_bytes, "{after:?}");
    assert_eq!(after.workspace_bytes, 0, "{after:?}");

    // Warm: the #740 output initialisation and nothing else, with exactly one
    // submission per block — no coefficient table is ever uploaded, because
    // the twist factor rides the contraction descriptor's own scale.
    let (twisted, warm) = delta(|| device.twist(&[0, 3], Direction::Forward).unwrap());
    assert_eq!(warm.h2d_calls, 1, "{warm:?}");
    assert_eq!(warm.h2d_bytes, output_bytes, "{warm:?}");
    assert_eq!(warm.d2h_calls, 0, "{warm:?}");
    assert_eq!(warm.d2h_bytes, 0, "{warm:?}");
    assert_eq!(warm.device_allocs, 1, "{warm:?}");
    assert_eq!(
        warm.gemm_calls, blocks,
        "one submission per block: {warm:?}"
    );
    assert_eq!(
        twisted.to_host().unwrap().dense_data().unwrap(),
        host.twist(&[0, 3], Direction::Forward)
            .unwrap()
            .dense_data()
            .unwrap(),
        "the measured call must still be the Host's answer"
    );

    let (_, warm_inverse) = delta(|| device.twist(&[0, 3], Direction::Inverse).unwrap());
    assert_eq!(
        warm_inverse, warm,
        "inverse twist has the same cost profile"
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_twist_short_circuits_do_no_device_work() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let bosonic = fixture(&runtime).to_cuda().unwrap();
    let fermionic_host = fermionic_fixture(&runtime);
    let fermionic = fermionic_host.to_cuda().unwrap();
    let expected = fixture(&runtime).dense_data().unwrap().to_vec();

    let (results, counters) = delta(|| {
        [
            // An empty leg list, and a bosonic provider whose every twist is
            // one, both return a clone of the receiver.
            bosonic.twist(&[], Direction::Forward).unwrap(),
            bosonic.twist(&[0, 2], Direction::Forward).unwrap(),
            bosonic.twist(&[1], Direction::Inverse).unwrap(),
        ]
    });
    assert_eq!(
        counters,
        CudaTransferStats::default(),
        "a short-circuited twist must submit nothing: {counters:?}"
    );
    for result in results {
        assert_eq!(result.to_host().unwrap().dense_data().unwrap(), expected);
    }

    // Fermionic, and the identity in *value* — the factors of all legs of a
    // block multiply to its total parity, which is even — but not a short
    // circuit: Host's detection tests each leg's own factor, not the product,
    // so it publishes a fresh unscaled copy. The device does exactly the same
    // work rather than inventing a cheaper answer.
    let _ = fermionic.twist(&[0, 3], Direction::Forward).unwrap();
    let (all_legs, counters) =
        delta(|| fermionic.twist(&[0, 1, 2, 3], Direction::Forward).unwrap());
    assert_eq!(counters.h2d_calls, 1, "{counters:?}");
    assert_eq!(counters.device_allocs, 1, "{counters:?}");
    assert_eq!(
        counters.gemm_calls,
        fermionic_host.subblock_count() as u64,
        "one submission per block, as for any other twist: {counters:?}"
    );
    assert_eq!(
        all_legs.to_host().unwrap().dense_data().unwrap(),
        fermionic_host.dense_data().unwrap()
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_twist_rejections_happen_before_any_device_work() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let host = fermionic_fixture(&runtime);
    let device = host.to_cuda().unwrap();
    // Warm the shared coefficient operand first, so a rejection cannot be
    // confused with a cold-path upload.
    let _ = device.twist(&[0, 3], Direction::Forward).unwrap();
    let before = runtime.cuda_tree_transform_stats().unwrap();

    let (errors, counters) = delta(|| {
        [
            // The range check precedes the empty-list short circuit and every
            // categorical step, exactly as on Host.
            (
                device.twist(&[4], Direction::Forward).unwrap_err().report(),
                host.twist(&[4], Direction::Forward).unwrap_err().report(),
            ),
            (
                device
                    .twist(&[1, 9], Direction::Inverse)
                    .unwrap_err()
                    .report(),
                host.twist(&[1, 9], Direction::Inverse)
                    .unwrap_err()
                    .report(),
            ),
        ]
    });
    for (actual, expected) in &errors {
        assert_eq!(actual, expected, "device error text must be the Host's");
    }
    assert!(
        errors[0].0.contains("out of range for rank 4"),
        "{errors:?}"
    );
    assert_eq!(
        counters,
        CudaTransferStats::default(),
        "a rejected twist must submit nothing: {counters:?}"
    );
    assert_eq!(runtime.cuda_tree_transform_stats().unwrap(), before);
}

/// The process-global completed-transformer cache (`symtn::cache`).
fn completed_transformers() -> symtn::cache::StructureCacheInfo {
    symtn::cache::stats()
        .into_iter()
        .find(|info| info.kind() == symtn::cache::StructureCacheKind::CompletedTreeTransformer)
        .expect("every structure cache kind reports")
}
