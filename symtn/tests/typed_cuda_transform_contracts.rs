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

macro_rules! full_transpose {
    ($engine:ident, $tensor:expr) => {{
        let tensor = &$tensor;
        let (codomain_axes, domain_axes) = full_rotation!(tensor);
        tensor.transpose($engine, &codomain_axes, &domain_axes)
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

fn fixture(
    engine: &mut dyn symtn::typed::Engine,
    runtime: &Runtime,
) -> TensorMap<U1FusionRule, f64> {
    let v = leg();
    TensorMap::from_subblock_fn(engine, runtime, [&v, &v], [&v, &v], |_, indices| {
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
fn device_transform_short_circuits_do_no_device_work() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut crate::common::cuda_engine();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let device = fixture(engine, &runtime).to_cuda(cuda).unwrap();
    let expected = fixture(engine, &runtime).dense_data().unwrap().to_vec();

    let (results, counters) = delta(|| {
        [
            // Identity axis lists, the same split, and a rank-0 transpose all
            // return a clone of the receiver.
            device.permute(cuda, &[0, 1], &[2, 3]).unwrap(),
            device.transpose(cuda, &[0, 1], &[2, 3]).unwrap(),
            device.braid(cuda, &[0, 1], &[2, 3], &[1, 2, 3, 4]).unwrap(),
            device.repartition(cuda, 2).unwrap(),
        ]
    });
    assert_eq!(
        counters,
        CudaTransferStats::default(),
        "a short-circuited transform must submit nothing: {counters:?}"
    );
    for result in results {
        assert_eq!(
            result.to_host(cuda).unwrap().dense_data().unwrap(),
            expected
        );
    }

    // A rank-0 tensor has no legs to build from, so it comes from a full Host
    // trace (this file does not test device `trace`) and is then uploaded.
    let v = leg();
    let endomorphism: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&v], [&v], |_, indices| {
            indices[0] as f64 + 1.0
        })
        .unwrap();
    let rank_zero = endomorphism.trace_pairs(engine, &[(0, 1)]).unwrap();
    assert_eq!(rank_zero.rank(), 0);
    let device_zero = rank_zero.to_cuda(cuda).unwrap();
    let (transposed, counters) = delta(|| full_transpose!(cuda, device_zero).unwrap());
    assert_eq!(counters, CudaTransferStats::default(), "{counters:?}");
    assert_eq!(
        transposed.to_host(cuda).unwrap().dense_data().unwrap(),
        rank_zero.dense_data().unwrap()
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_transform_rejections_happen_before_any_device_work() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut crate::common::cuda_engine();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let device = fixture(engine, &runtime).to_cuda(cuda).unwrap();
    // Warm the structures first, so a rejection cannot be confused with the
    // cold-path uploads of a neighbouring call.
    let _ = device.permute(cuda, &[2, 0], &[1, 3]).unwrap();

    let host = fixture(engine, &runtime);
    // Each rejection is written once and expanded against both receivers, so
    // the device error *mapping* is gated, not merely "an error came back".
    macro_rules! rejects_like_host {
        (|$e:ident, $t:ident| $call:expr) => {{
            let expected = {
                let $e: &mut dyn symtn::typed::Engine = &mut *engine;
                let $t = &host;
                $call
            }
            .unwrap_err()
            .report();
            let actual = {
                let $e: &mut dyn symtn::typed::Engine = &mut *cuda;
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
            rejects_like_host!(|e, t| t.braid(e, &[0, 1], &[2, 3], &[1, 2, 3])),
            rejects_like_host!(|e, t| t.braid(e, &[2, 0], &[1, 3], &[1, 2, 3, 4, 5])),
            // Malformed axes come back from the expert layer.
            rejects_like_host!(|e, t| t.permute(e, &[0, 0], &[2, 3])),
            // A non-planar re-arrangement is refused rather than braided.
            rejects_like_host!(|e, t| t.transpose(e, &[1, 2], &[3, 0])),
            // A split beyond the rank has no planar reading.
            rejects_like_host!(|e, t| t.repartition(e, 5)),
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

/// A device payload as raw bits, so a NaN-poisoned destination compares equal
/// to itself.
fn payload_bits<R>(
    cuda: &mut dyn symtn::typed::Engine,
    tensor: &TensorMap<R, f64, symtn::typed::CudaStorage>,
) -> Vec<u64>
where
    R: symtn::sector::MultiplicityFreeRigidSymbols<Scalar = f64>
        + symtn::sector::CheckedFusionAlgebra
        + symtn::sector::SectorCodec,
{
    tensor
        .to_host(cuda)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .map(|value| value.to_bits())
        .collect()
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_into_admits_the_exact_layout_on_the_shared_runtime_store() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut crate::common::cuda_engine();
    let _serial = crate::counting_alloc::serial();
    let _cache = cache_exclusive();
    // Host `overwrite_tree_transform` admits the source/destination layout
    // pair after a successful replay, so a second call resolves the operation
    // out of the store instead of deriving it. The store is the Runtime's own
    // Host store — the device path must feed and reuse exactly that one.
    let runtime = Runtime::builder().build().unwrap();
    let host = fixture(engine, &runtime);
    let expected = host.permute(engine, &[2, 0], &[1, 3]).unwrap();
    let source = host.to_cuda(cuda).unwrap();

    symtn::cache::clear();
    let mut first = expected.to_cuda(cuda).unwrap();
    source
        .permute_into(cuda, &[2, 0], &[1, 3], &mut first, 1.0, 0.0)
        .unwrap();
    let cold = completed_transformers();

    let mut second = expected.to_cuda(cuda).unwrap();
    source
        .permute_into(cuda, &[2, 0], &[1, 3], &mut second, 1.0, 0.0)
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
        first.to_host(cuda).unwrap().dense_data().unwrap(),
        second.to_host(cuda).unwrap().dense_data().unwrap()
    );

    // Admission is keyed on the layout pair, so a destination that is *not*
    // this operation's result still misses the lookup and is still rejected.
    let mut wrong = full_transpose!(engine, host)
        .unwrap()
        .to_cuda(cuda)
        .unwrap();
    let error = source
        .permute_into(cuda, &[2, 0], &[1, 3], &mut wrong, 1.0, 0.0)
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
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut crate::common::cuda_engine();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let host = fixture(engine, &runtime);
    let expected = host.permute(engine, &[2, 0], &[1, 3]).unwrap();
    let source = host.to_cuda(cuda).unwrap();
    // Fresh, uniquely owned destinations on demand: a `clone()` would share
    // the payload `Arc` and trip the unique-ownership check instead.
    let host_destination =
        |engine: &mut dyn symtn::typed::Engine| host.permute(engine, &[2, 0], &[1, 3]).unwrap();
    let device_destination = |cuda: &mut dyn symtn::typed::Engine| expected.to_cuda(cuda).unwrap();

    // Warm the structure first, so a rejection cannot be confused with the
    // cold-path uploads of a neighbouring call.
    {
        let mut warm = device_destination(cuda);
        source
            .permute_into(cuda, &[2, 0], &[1, 3], &mut warm, 1.0, 0.0)
            .unwrap();
    }

    // Asserts one device rejection: the Host's error text (with the placement
    // noun), no transfer, allocation or executor-state change, and a
    // byte-identical witness — the owned device tensor whose payload the
    // destination is (itself, or, for a lazy destination, its parent).
    macro_rules! rejects_like_host {
        ($what:expr, $host_call:expr, $witness:ident, $device_call:expr) => {{
            let expected_message = as_device_message(&$host_call.unwrap_err().report());
            // Bitwise: the poisoned destinations are NaN, which is not equal
            // to itself.
            let before = payload_bits(cuda, &$witness);
            let (error, counters) = delta(|| $device_call.unwrap_err());
            assert_eq!(error.report(), expected_message, "{}", $what);
            assert_eq!(
                counters,
                CudaTransferStats::default(),
                "{}: a rejected overwrite must submit nothing: {counters:?}",
                $what
            );
            assert_eq!(
                payload_bits(cuda, &$witness),
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
    let z2_host =
        TensorMap::from_subblock_fn(engine, &runtime, [&z2_leg], [&z2_leg], zn_fill).unwrap();
    let z2_source = z2_host.to_cuda(cuda).unwrap();

    // 1. Runtime mismatch wins over a rule mismatch: a Z3 destination on
    //    another Runtime.
    let other = Runtime::builder().build().unwrap();
    let mut host_foreign =
        TensorMap::from_subblock_fn(engine, &other, [&z3_leg], [&z3_leg], |_, _| f64::NAN).unwrap();
    let mut foreign = host_foreign.to_cuda(cuda).unwrap();
    {
        let witness = foreign.clone();
        let error = rejects_like_host!(
            "runtime mismatch precedes rule mismatch",
            z2_host.permute_into(engine, &[0], &[1], &mut host_foreign, 1.0, 0.0),
            witness,
            z2_source.permute_into(cuda, &[0], &[1], &mut foreign, 1.0, 0.0)
        );
        assert_eq!(error, Error::RuntimeMismatch);
    }

    // 2. Rule mismatch wins over a lazy-adjoint source.
    let mut host_z3 =
        TensorMap::from_subblock_fn(engine, &runtime, [&z3_leg], [&z3_leg], |_, _| f64::NAN)
            .unwrap();
    let mut z3_destination = host_z3.to_cuda(cuda).unwrap();
    let lazy_source = source.adjoint(cuda).unwrap();
    let host_lazy_source = host.adjoint(engine).unwrap();
    let z2_lazy_source = z2_source.adjoint(cuda).unwrap();
    let z2_host_lazy_source = z2_host.adjoint(engine).unwrap();
    {
        let witness = z3_destination.clone();
        let error = rejects_like_host!(
            "rule mismatch precedes the lazy-adjoint source",
            z2_host_lazy_source.permute_into(engine, &[0], &[1], &mut host_z3, 1.0, 0.0),
            witness,
            z2_lazy_source.permute_into(cuda, &[0], &[1], &mut z3_destination, 1.0, 0.0)
        );
        assert_eq!(error, Error::RuleMismatch);
    }

    // 3. A lazy-adjoint source is admitted (read through its parent, #1870),
    //    so with a lazy-adjoint destination the destination is reported.
    let lazy_parent = device_destination(cuda);
    let mut lazy_destination = lazy_parent.adjoint(cuda).unwrap();
    let mut host_lazy_destination = host_destination(engine).adjoint(engine).unwrap();
    let error = rejects_like_host!(
        "a lazy-adjoint source with a lazy-adjoint destination",
        host_lazy_source.permute_into(
            engine,
            &[2, 0],
            &[1, 3],
            &mut host_lazy_destination,
            1.0,
            0.0
        ),
        lazy_parent,
        lazy_source.permute_into(cuda, &[2, 0], &[1, 3], &mut lazy_destination, 1.0, 0.0)
    );
    assert!(
        error.report().contains("ordinary dense CUDA storage"),
        "{error}"
    );

    // 4. A lazy-adjoint destination alone.
    let mut host_lazy_destination = host_destination(engine).adjoint(engine).unwrap();
    let error = rejects_like_host!(
        "lazy-adjoint destination",
        host.permute_into(
            engine,
            &[2, 0],
            &[1, 3],
            &mut host_lazy_destination,
            1.0,
            0.0
        ),
        lazy_parent,
        source.permute_into(cuda, &[2, 0], &[1, 3], &mut lazy_destination, 1.0, 0.0)
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
            host.permute_into(engine, &[0, 0], &[1, 3], &mut host_alias, 1.0, 0.0),
            witness,
            source.permute_into(cuda, &[0, 0], &[1, 3], &mut alias, 1.0, 0.0)
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
            host_lazy_source.permute_into(
                engine,
                &[2, 0],
                &[1, 3],
                &mut host_parent_alias,
                1.0,
                0.0
            ),
            witness,
            lazy_source.permute_into(cuda, &[2, 0], &[1, 3], &mut parent_alias, 1.0, 0.0)
        );
        assert!(error.report().contains("must not alias"), "{error}");
    }

    // 6. A destination on the wrong space, held shared: the space check
    //    precedes the unique-ownership check.
    let mut wrong_space = full_transpose!(engine, host)
        .unwrap()
        .to_cuda(cuda)
        .unwrap();
    let wrong_space_handle = wrong_space.clone();
    let mut host_wrong_space = full_transpose!(engine, host).unwrap();
    let host_wrong_space_handle = host_wrong_space.clone();
    {
        let witness = wrong_space.clone();
        let error = rejects_like_host!(
            "the space check precedes unique ownership",
            host.permute_into(engine, &[2, 0], &[1, 3], &mut host_wrong_space, 1.0, 0.0),
            witness,
            source.permute_into(cuda, &[2, 0], &[1, 3], &mut wrong_space, 1.0, 0.0)
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
    let mut shared = device_destination(cuda);
    let shared_handle = shared.clone();
    let mut host_shared = host_destination(engine);
    let host_shared_handle = host_shared.clone();
    {
        let witness = shared.clone();
        let error = rejects_like_host!(
            "shared destination",
            host.permute_into(engine, &[2, 0], &[1, 3], &mut host_shared, 1.0, 0.0),
            witness,
            source.permute_into(cuda, &[2, 0], &[1, 3], &mut shared, 1.0, 0.0)
        );
        assert!(error.report().contains("shares its storage"), "{error}");
    }
    drop(shared_handle);
    drop(host_shared_handle);

    // 8. Malformed axes, a non-planar cyclic request and a `repartition`
    //    destination of another rank all come back from the operation build,
    //    before the lease.
    let mut destination = device_destination(cuda);
    let mut host_dst = host_destination(engine);
    let witness = destination.clone();
    rejects_like_host!(
        "malformed axes",
        host.permute_into(engine, &[0, 0], &[1, 3], &mut host_dst, 1.0, 0.0),
        witness,
        source.permute_into(cuda, &[0, 0], &[1, 3], &mut destination, 1.0, 0.0)
    );
    rejects_like_host!(
        "non-planar transpose",
        host.transpose_into(engine, &[1, 2], &[3, 0], &mut host_dst, 1.0, 0.0),
        witness,
        source.transpose_into(cuda, &[1, 2], &[3, 0], &mut destination, 1.0, 0.0)
    );

    let v = leg();
    let mut host_rank_three =
        TensorMap::from_subblock_fn(engine, &runtime, [&v, &v], [&v], |_, _| f64::NAN).unwrap();
    let mut rank_three = host_rank_three.to_cuda(cuda).unwrap();
    {
        let witness = rank_three.clone();
        let error = rejects_like_host!(
            "repartition destination of another rank",
            host.repartition_into(engine, &mut host_rank_three, 1.0, 0.0),
            witness,
            source.repartition_into(cuda, &mut rank_three, 1.0, 0.0)
        );
        assert!(
            error.report().contains("does not match source rank"),
            "{error}"
        );
    }
}

// ---------------------------------------------------------------------------
// `twist` in both directions contracts (issue #1330)
// ---------------------------------------------------------------------------

/// fZ2 with both parities on every leg, so a twist really scales blocks.
fn fermionic_fixture(
    engine: &mut dyn symtn::typed::Engine,
    runtime: &Runtime,
) -> TensorMap<symtn::sector::FermionParityFusionRule, f64> {
    let leg = GradedSpace::try_new(
        Arc::new(symtn::sector::FermionParityFusionRule),
        [
            (symtn::sector::Z2Irrep::EVEN, 2),
            (symtn::sector::Z2Irrep::ODD, 1),
        ],
    )
    .unwrap();
    TensorMap::from_subblock_fn(engine, runtime, [&leg, &leg], [&leg, &leg], |_, indices| {
        indices.iter().map(|&i| i as f64 + 1.0).sum::<f64>()
    })
    .unwrap()
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_twist_short_circuits_do_no_device_work() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut crate::common::cuda_engine();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let bosonic = fixture(engine, &runtime).to_cuda(cuda).unwrap();
    let fermionic_host = fermionic_fixture(engine, &runtime);
    let fermionic = fermionic_host.to_cuda(cuda).unwrap();
    let expected = fixture(engine, &runtime).dense_data().unwrap().to_vec();

    let (results, counters) = delta(|| {
        [
            // An empty leg list, and a bosonic provider whose every twist is
            // one, both return a clone of the receiver.
            bosonic.twist(cuda, &[], Direction::Forward).unwrap(),
            bosonic.twist(cuda, &[0, 2], Direction::Forward).unwrap(),
            bosonic.twist(cuda, &[1], Direction::Inverse).unwrap(),
        ]
    });
    assert_eq!(
        counters,
        CudaTransferStats::default(),
        "a short-circuited twist must submit nothing: {counters:?}"
    );
    for result in results {
        assert_eq!(
            result.to_host(cuda).unwrap().dense_data().unwrap(),
            expected
        );
    }

    // Fermionic, and the identity in *value* — the factors of all legs of a
    // block multiply to its total parity, which is even — but not a short
    // circuit: Host's detection tests each leg's own factor, not the product,
    // so it publishes a fresh unscaled copy. The device does exactly the same
    // work rather than inventing a cheaper answer.
    let _ = fermionic.twist(cuda, &[0, 3], Direction::Forward).unwrap();
    let (all_legs, counters) = delta(|| {
        fermionic
            .twist(cuda, &[0, 1, 2, 3], Direction::Forward)
            .unwrap()
    });
    // The zeroed output (#740) and, since an eager call keeps no device
    // state between calls, this call's one-element `1` template.
    let output_len = fermionic_host.dense_data().unwrap().len() as u64;
    assert_eq!(counters.h2d_calls, 2, "{counters:?}");
    assert_eq!(counters.h2d_bytes, (output_len + 1) * 8, "{counters:?}");
    assert_eq!(counters.device_allocs, 2, "{counters:?}");
    assert_eq!(
        counters.gemm_calls,
        fermionic_host.subblock_count() as u64,
        "one submission per block, as for any other twist: {counters:?}"
    );
    assert_eq!(
        all_legs.to_host(cuda).unwrap().dense_data().unwrap(),
        fermionic_host.dense_data().unwrap()
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn device_twist_rejections_happen_before_any_device_work() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut crate::common::cuda_engine();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let host = fermionic_fixture(engine, &runtime);
    let device = host.to_cuda(cuda).unwrap();
    // Warm the shared coefficient operand first, so a rejection cannot be
    // confused with a cold-path upload.
    let _ = device.twist(cuda, &[0, 3], Direction::Forward).unwrap();

    let (errors, counters) = delta(|| {
        [
            // The range check precedes the empty-list short circuit and every
            // categorical step, exactly as on Host.
            (
                device
                    .twist(cuda, &[4], Direction::Forward)
                    .unwrap_err()
                    .report(),
                host.twist(engine, &[4], Direction::Forward)
                    .unwrap_err()
                    .report(),
            ),
            (
                device
                    .twist(cuda, &[1, 9], Direction::Inverse)
                    .unwrap_err()
                    .report(),
                host.twist(engine, &[1, 9], Direction::Inverse)
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
}

/// The process-global completed-transformer cache (`symtn::cache`).
fn completed_transformers() -> symtn::cache::StructureCacheInfo {
    symtn::cache::stats()
        .into_iter()
        .find(|info| info.kind() == symtn::cache::StructureCacheKind::CompletedTreeTransformer)
        .expect("every structure cache kind reports")
}
