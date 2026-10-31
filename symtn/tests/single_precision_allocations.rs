//! Allocation contracts of the single-precision payloads (#1315).
//!
//! Two contracts:
//!
//! * admitting a payload dtype to `TensorScalar` adds an execution lane per
//!   `Ctxs`, and those lanes are built on first use — a program that never
//!   names a dtype pays nothing for it at `Runtime::build()`;
//! * a single-precision tensor performs the *same number* of allocations as
//!   its double-precision twin and allocates exactly half the payload bytes.
//!   Narrowing the payload must not change the algorithm, only the bytes it
//!   moves.
//!
//! Every assertion here is *relative*: two measurements taken in this same
//! process compared against each other. Absolute allocation counts depend on
//! the platform, the allocator and the core count — the CPU context sizes its
//! pool from the available parallelism — so a number measured on one machine
//! is evidence, not a contract. The measured figures live in
//! [historical audit](https://github.com/Ryo-wtnb11/symtn-rs/blob/8c54ce960d0dd5ea000c75ff1893624e3da42420/docs/audit/issue-1315-single-precision-base.md).

use std::hint::black_box;
use symtn::typed::Engine;

use std::sync::Arc;

use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::FactorScalar;
use symtn::typed::{Complex32, Complex64, GradedSpace, Runtime, TensorMap, Truncation};
use symtn::typed::{Qr, Svd};

use crate::counting_alloc;

// Why `counting_alloc::serial()`: the three tests here can run
// concurrently under cargo's default in-binary test threading. The
// factorization and the plain-pipeline tests both build `u1_space()` — the
// same sectors and degeneracies, hence the same key into symtn-core's
// process-global fusion-tree-layout and complete-HomSpace-structure caches
// (`symtn-core/src/fusion_space/layout_cache.rs`). One test's concurrent
// admission or (weakly held, #1610) eviction of that shared entry between
// another test's warm-up and measured call changes the measured call's own
// allocation count without changing what it computes: reported as 218 f32
// allocations against 202 f64 under a full parallel workspace run. Every test in
// this file holds `counting_alloc::serial()` for its whole body, the same
// technique `transform_owned_parallel.rs` uses.

fn measured<T>(operation: impl FnOnce() -> T) -> (T, usize, usize) {
    let (value, allocs) = counting_alloc::measure(operation);
    (value, allocs.calls as usize, allocs.bytes as usize)
}

fn build_runtime() -> Runtime {
    Runtime::builder().build().unwrap()
}

/// A permute at one payload dtype, which is the cheapest public operation
/// that builds an execution lane for that dtype.
macro_rules! permuted {
    ($engine:ident, $runtime:expr, $dtype:ty, $space:expr, $seed:expr) => {{
        let tensor: TensorMap<U1FusionRule, $dtype> =
            TensorMap::rand_with_seed($engine, $runtime, [$space, $space], [$space], $seed)
                .unwrap();
        black_box(tensor.permute($engine, &[1], &[2, 0]).unwrap());
    }};
}

#[test]
fn runtime_construction_does_not_build_unused_dtype_lanes() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // Warm anything the process initialises once (thread pools, env parsing)
    // so every build measured below is steady state.
    black_box(build_runtime());
    let space = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();

    let (_, baseline, baseline_bytes) = measured(build_runtime);
    eprintln!("Runtime::build baseline: {baseline} allocation calls, {baseline_bytes} bytes");

    // Using single precision must not make the *next* runtime more expensive
    // to build: the lanes belong to the runtime that needed them, and nothing
    // about them is global or sticky.
    let user = build_runtime();
    permuted!(engine, &user, Complex32, &space, 1);
    let (_, after_use, _) = measured(build_runtime);
    assert_eq!(
        after_use, baseline,
        "constructing a Runtime cost {after_use} allocation calls after single precision \
         had been used in this process, against {baseline} before"
    );

    // The lanes are per call, not per runtime: on a fresh runtime, after the
    // double-precision operation warmed everything cached by structure (the
    // layout admission and the shared tree-transform plan store), the first
    // single-precision operation costs what the second does. A Complex32
    // lane built into the runtime on first use would make the first dearer.
    for round in 0..2 {
        let runtime = build_runtime();
        permuted!(engine, &runtime, f64, &space, 10 + round);
        permuted!(engine, &runtime, f64, &space, 20 + round);

        let (_, cold, _) = measured(|| permuted!(engine, &runtime, Complex32, &space, 30 + round));
        let (_, warm, _) = measured(|| permuted!(engine, &runtime, Complex32, &space, 40 + round));
        eprintln!("round {round}: first Complex32 permute {cold} calls, second {warm}");
        assert_eq!(
            cold, warm,
            "round {round}: the first Complex32 call on a runtime built per-runtime state"
        );
    }
}

fn u1_space() -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 4),
            (U1Irrep::new(0), 5),
            (U1Irrep::new(1), 4),
        ],
    )
    .unwrap()
}

/// Builds a tensor, permutes it and reduces it, at one payload dtype.
///
/// A macro because the payload dtype is the only thing that varies and the
/// two instantiations must run the identical sequence of public calls.
macro_rules! measure_pipeline {
    ($engine:ident, $runtime:expr, $dtype:ty, $space:expr) => {{
        let space = $space;
        // Warm the lane, the layout admission and the transform plan, so the
        // measured pass is steady state at both precisions.
        let warm: TensorMap<U1FusionRule, $dtype> =
            TensorMap::rand_with_seed($engine, $runtime, [space, space], [space], 5_501).unwrap();
        black_box(warm.permute($engine, &[1], &[2, 0]).unwrap());
        black_box(warm.norm($engine, 2.0).unwrap());

        measured(|| {
            let tensor: TensorMap<U1FusionRule, $dtype> =
                TensorMap::rand_with_seed($engine, $runtime, [space, space], [space], 5_502)
                    .unwrap();
            let permuted = tensor.permute($engine, &[1], &[2, 0]).unwrap();
            // `FactorScalar::from_real`, not `<$dtype>::from(1.0)`: an
            // unsuffixed float literal has no type to fall back to that every
            // instantiation of this macro accepts.
            let one = <$dtype as FactorScalar>::from_real(1.0);
            let sum = tensor.axpby($engine, one, &tensor, one).unwrap();
            black_box((
                permuted,
                sum,
                tensor.norm($engine, 2.0).unwrap(),
                tensor.dense_data().unwrap().len(),
            ))
        })
    }};
}

#[test]
fn single_precision_allocates_as_often_as_double_and_half_the_payload_bytes() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let space = u1_space();

    let (_, f64_calls, f64_bytes) = measure_pipeline!(engine, &runtime, f64, &space);
    let (single, f32_calls, f32_bytes) = measure_pipeline!(engine, &runtime, f32, &space);
    let (_, c64_calls, c64_bytes) = measure_pipeline!(engine, &runtime, Complex64, &space);
    let (_, c32_calls, c32_bytes) = measure_pipeline!(engine, &runtime, Complex32, &space);
    let payload = single.3;

    eprintln!(
        "payload {payload}: f64 {f64_calls} calls/{f64_bytes} B, f32 {f32_calls}/{f32_bytes}, \
         c64 {c64_calls}/{c64_bytes}, c32 {c32_calls}/{c32_bytes}"
    );

    assert_eq!(
        f32_calls, f64_calls,
        "f32 must run the same algorithm as f64, not a different number of allocations"
    );
    assert_eq!(c32_calls, c64_calls, "Complex32 against Complex64");

    // Three payload-sized buffers are produced above (the tensor, the permuted
    // result and the sum), so narrowing the payload saves exactly half of each.
    assert_eq!(
        f64_bytes - f32_bytes,
        3 * payload * (size_of::<f64>() - size_of::<f32>()),
        "f32 must allocate exactly half the payload bytes of f64"
    );
    assert_eq!(
        c64_bytes - c32_bytes,
        3 * payload * (size_of::<Complex64>() - size_of::<Complex32>()),
        "Complex32 must allocate exactly half the payload bytes of Complex64"
    );
}

/// Builds a tensor and runs one factorization on it, at one payload dtype.
///
/// The same public sequence at every dtype: `qr` is a compact QR, `svd` a
/// compact SVD followed by its truncation (a composition). The spectra are
/// `f64` at every payload dtype by contract, so only the *factor payloads*
/// narrow — which is exactly what the byte assertion counts.
///
/// Returns `(produced, spectrum_sectors, calls, bytes)` for one **steady-state**
/// iteration: `produced` is the total payload length of every tensor the
/// sequence returns, `spectrum_sectors` the number of coupled sectors across
/// the spectrum factors it builds.
///
/// Steady state, not "one warm pass", is the whole point. A single iteration
/// measured cold includes one-time growth of the dense backend's buffer pool,
/// which is amortized across dtypes through process-global state: measured
/// that way the counts depend on *which dtype ran first*. After four identical
/// iterations the per-iteration count is reproducible and independent of the
/// dtype order and of the runtime instance, which is what makes it a property
/// of the algorithm rather than of the process.
macro_rules! measure_factorization {
    ($engine:ident, $runtime:expr, $dtype:ty, $space:expr, qr) => {{
        let space = $space;
        let sequence = |$engine: &mut dyn Engine| {
            let tensor: TensorMap<U1FusionRule, $dtype> =
                TensorMap::rand_with_seed($engine, $runtime, [space, space], [space], 7_502)
                    .unwrap();
            let Qr { q, r } = tensor.qr_compact($engine, &[0, 1], &[2]).unwrap();
            (tensor, q, r)
        };
        for _ in 0..4 {
            black_box(sequence($engine));
        }
        let (tensor, q, r) = sequence($engine);
        let produced = tensor.dense_data().unwrap().len()
            + q.dense_data().unwrap().len()
            + r.dense_data().unwrap().len();
        black_box((tensor, q, r));
        let (_, calls, bytes) = measured(|| black_box(sequence($engine)));
        (produced, 0, calls, bytes)
    }};
    ($engine:ident, $runtime:expr, $dtype:ty, $space:expr, svd) => {{
        let space = $space;
        let sequence = |$engine: &mut dyn Engine| {
            let tensor: TensorMap<U1FusionRule, $dtype> =
                TensorMap::rand_with_seed($engine, $runtime, [space, space], [space], 7_502)
                    .unwrap();
            let Svd { u, s, vh } = tensor.svd_compact($engine, &[0, 1], &[2]).unwrap();
            // The truncated SVD is a composition (#1534).
            let found = s.domain()[0]
                .find_truncated(&s.diagview($engine).unwrap(), &Truncation::rank(4))
                .unwrap();
            let truncated = (
                u.restrict_leg($engine, &[(u.codomain_rank(), &found.selection)])
                    .unwrap(),
                s.restrict_leg($engine, &[(0, &found.selection), (1, &found.selection)])
                    .unwrap(),
                vh.restrict_leg($engine, &[(0, &found.selection)]).unwrap(),
            );
            (tensor, u, s, vh, truncated)
        };
        for _ in 0..4 {
            black_box(sequence($engine));
        }
        // Shape observation, outside the counter: `diagview` allocates.
        let (tensor, u, s, vh, truncated) = sequence($engine);
        let produced = tensor.dense_data().unwrap().len()
            + u.dense_data().unwrap().len()
            + s.materialize($engine).unwrap().dense_data().unwrap().len()
            + vh.dense_data().unwrap().len()
            + truncated.0.dense_data().unwrap().len()
            + truncated
                .1
                .materialize($engine)
                .unwrap()
                .dense_data()
                .unwrap()
                .len()
            + truncated.2.dense_data().unwrap().len();
        let spectrum_sectors =
            s.diagview($engine).unwrap().len() + truncated.1.diagview($engine).unwrap().len();
        black_box((tensor, u, s, vh, truncated));
        let (_, calls, bytes) = measured(|| black_box(sequence($engine)));
        (produced, spectrum_sectors, calls, bytes)
    }};
}

/// Measures one factorization at all four payload dtypes and asserts that
/// each single-precision run takes the double-precision allocation route.
macro_rules! assert_factorization_allocates_like_double {
    ($op:tt) => {{
        let engine = &mut symtn::typed::CpuEngine::serial();
        let _serial = crate::counting_alloc::serial();
        let runtime = Runtime::builder().build().unwrap();
        let space = u1_space();

        let (f64_payload, f64_sectors, f64_calls, f64_bytes) =
            measure_factorization!(engine, &runtime, f64, &space, $op);
        let (f32_payload, f32_sectors, f32_calls, f32_bytes) =
            measure_factorization!(engine, &runtime, f32, &space, $op);
        let (c64_payload, c64_sectors, c64_calls, c64_bytes) =
            measure_factorization!(engine, &runtime, Complex64, &space, $op);
        let (c32_payload, c32_sectors, c32_calls, c32_bytes) =
            measure_factorization!(engine, &runtime, Complex32, &space, $op);

        eprintln!(
            "{}: f64 {f64_calls} calls/{f64_bytes} B ({f64_payload} entries, \
             {f64_sectors} spectrum sectors), f32 {f32_calls}/{f32_bytes} ({f32_payload}, \
             {f32_sectors}), c64 {c64_calls}/{c64_bytes} ({c64_payload}, {c64_sectors}), \
             c32 {c32_calls}/{c32_bytes} ({c32_payload}, {c32_sectors})",
            stringify!($op)
        );

        assert_eq!(
            (f32_payload, f32_sectors),
            (f64_payload, f64_sectors),
            "the twins must produce factors of the same shape"
        );
        assert_eq!(
            (c32_payload, c32_sectors),
            (c64_payload, c64_sectors),
            "Complex32 against Complex64"
        );

        // Equality, not a bound (#1337): the buffer count is a property of
        // the algorithm rather than of the payload layout. The spectrum
        // factor is filled from a borrowed slice; a
        // `into_iter().map(to_scalar).collect()` from the `Vec<f64>` spectrum
        // would reuse the source buffer only when `D` matches `f64` in size
        // and alignment, costing the narrow dtypes one allocation per sector.
        for (single, double, what) in [
            (f32_calls, f64_calls, "f32 against f64"),
            (c32_calls, c64_calls, "Complex32 against Complex64"),
        ] {
            assert_eq!(
                single, double,
                "{what}: narrowing the payload must not change how often the \
                 factorization allocates ({single} against {double})"
            );
        }

        // Every entry of every tensor the sequence returns is half as wide;
        // the scratch the factorization allocates on top is not necessarily
        // payload typed, so this is the lower bound the returned tensors
        // alone guarantee.
        for (wide_bytes, narrow_bytes, payload, step, what) in [
            (
                f64_bytes,
                f32_bytes,
                f64_payload,
                size_of::<f64>() - size_of::<f32>(),
                "f32 against f64",
            ),
            (
                c64_bytes,
                c32_bytes,
                c64_payload,
                size_of::<Complex64>() - size_of::<Complex32>(),
                "Complex32 against Complex64",
            ),
        ] {
            // Addition, not subtraction: a regression must fail the
            // assertion, not panic on a `usize` underflow before reaching it.
            assert!(
                narrow_bytes + payload * step <= wide_bytes,
                "{what}: the narrow run allocated {narrow_bytes} B against {wide_bytes} B, \
                 less than the {} B its factors alone are narrower by",
                payload * step
            );
        }
    }};
}

/// Purpose: single-precision compact SVD (and its truncation) takes the
/// double-precision allocation route — no hidden widening or copy.
#[test]
fn single_precision_svd_allocates_like_double() {
    assert_factorization_allocates_like_double!(svd);
}

/// Purpose: single-precision compact QR takes the double-precision
/// allocation route — no hidden widening or copy.
#[test]
#[ignore = "compact QR grows a recycled backend buffer in place, so its allocation count follows buffer reuse history, not the dtype"]
fn single_precision_qr_allocates_like_double() {
    assert_factorization_allocates_like_double!(qr);
}
