//! The pooled dense lease adds no allocations to a warm factorization (#2116).
//!
//! A default runtime leases a pooled executor per call; a runtime with an
//! injected executor takes the `state` lock instead. With the same provider
//! thread count both run the identical dense route, so the pooled arm must not
//! allocate more calls than the locked arm, at either block count: pool
//! pop/push, the lease guard and the Host-pool entry cost nothing per call.
//!
//! Why the locked executor is built exactly as the pool mints one (on a
//! one-thread `SharedCpuContext`): the counter is thread-local, and any other
//! constructor may run provider calls off the calling thread, so the locked
//! arm looks 3-6x cheaper than it is. `DefaultDenseExecutor::default()` sizes
//! a Rayon pool from the environment (the #1996 observation), and on Linux
//! even `with_threads(1)` builds a managed engine with its own workers.

use std::hint::black_box;
use std::sync::Arc;

use symtn::expert::{DefaultDenseExecutor, SharedCpuContext};
use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{GradedSpace, HermitianTol, Runtime, TensorMap};

use crate::counting_alloc;

fn pooled() -> Runtime {
    Runtime::builder().dense_threads(1).build().unwrap()
}

fn locked() -> Runtime {
    Runtime::builder()
        .dense_threads(1)
        .with_dense_executor(Box::new(
            DefaultDenseExecutor::with_shared_context(
                &SharedCpuContext::with_threads(1).unwrap(),
                None,
            )
            .unwrap(),
        ))
        .build()
        .unwrap()
}

/// One warm measured call (after two warm-ups): allocation calls and output.
fn warm(op: impl Fn() -> Vec<f64>) -> (u64, Vec<f64>) {
    black_box(op());
    black_box(op());
    let (out, allocs) = counting_alloc::measure(|| black_box(op()));
    (allocs.calls, out)
}

/// `a` diagonally dominant (svd/qr/pinv/solve divisor), `h` symmetric
/// (eigh), `b` small (exp argument, solve rhs).
macro_rules! fixture {
    ($runtime:expr, $space:expr) => {{
        let (rt, v) = ($runtime, $space);
        [
            TensorMap::from_subblock_fn(rt, [v], [v], |_, i| {
                if i[0] == i[1] {
                    4.0 + i[0] as f64
                } else {
                    0.3 * i[0] as f64 - 0.2 * i[1] as f64 + 0.1
                }
            })
            .unwrap(),
            TensorMap::from_subblock_fn(rt, [v], [v], |_, i| {
                1.0 / (1.0 + i[0] as f64 + i[1] as f64)
            })
            .unwrap(),
            TensorMap::from_subblock_fn(rt, [v], [v], |_, i| {
                0.01 * (1.0 + i[0] as f64 - 0.5 * i[1] as f64)
            })
            .unwrap(),
        ]
    }};
}

/// Runs the six dense routes on one fixture: `(op, calls, output)`.
macro_rules! six {
    ($f:expr) => {{
        let [a, h, b] = $f;
        let d = |t: &TensorMap<_, f64>| t.dense_data().unwrap().to_vec();
        vec![
            (
                "svd_compact",
                warm(|| {
                    let s = a.svd_compact(&[0], &[1]).unwrap();
                    [d(&s.u), d(&s.vh)].concat()
                }),
            ),
            (
                "qr_compact",
                warm(|| {
                    let q = a.qr_compact(&[0], &[1]).unwrap();
                    [d(&q.q), d(&q.r)].concat()
                }),
            ),
            (
                "eigh_full",
                warm(|| d(&h.eigh_full(&[0], &[1], HermitianTol::DEFAULT).unwrap().v)),
            ),
            ("pinv", warm(|| d(&a.pinv(&[0], &[1], 1e-12).unwrap()))),
            ("exp", warm(|| d(&b.exp(&[0], &[1]).unwrap()))),
            (
                "solve",
                warm(|| d(&a.solve(&[0], &[1], &b, &[0], &[1]).unwrap())),
            ),
        ]
    }};
}

fn assert_pooled_within_locked(
    label: &str,
    pooled: Vec<(&str, (u64, Vec<f64>))>,
    locked: Vec<(&str, (u64, Vec<f64>))>,
) {
    for ((op, (pooled_calls, pooled_out)), (_, (locked_calls, locked_out))) in
        pooled.into_iter().zip(locked)
    {
        assert!(
            pooled_calls <= locked_calls,
            "{label} {op}: pooled {pooled_calls} calls > locked {locked_calls}"
        );
        // Same provider and route; a checked-Generic recoupling may sum its
        // coefficients in another hash order per runtime, hence a tolerance.
        assert_eq!(pooled_out.len(), locked_out.len(), "{label} {op}");
        for (p, l) in pooled_out.iter().zip(&locked_out) {
            assert!((p - l).abs() <= 1e-12 * (1.0 + l.abs()), "{label} {op}");
        }
    }
}

#[test]
fn pooled_lease_adds_no_allocation_calls_at_any_block_count() {
    let _serial = crate::counting_alloc::serial();
    for sectors in [2, 6] {
        let space = GradedSpace::try_new(
            Arc::new(U1FusionRule),
            (0..sectors).map(|q| (U1Irrep::new(q), 4)),
        )
        .unwrap();
        let (p, l) = (pooled(), locked());
        assert_pooled_within_locked(
            &format!("U(1) {sectors} sectors"),
            six!(fixture!(&p, &space)),
            six!(fixture!(&l, &space)),
        );
    }
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_su3_pooled_lease_adds_no_allocation_calls_at_any_block_count() {
    let _serial = crate::counting_alloc::serial();
    use symtn::sector::LieGroupFusionRule;
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let sets: [&[([i64; 2], usize)]; 2] = [
        &[([0, 0], 3), ([1, 0], 2)],
        &[
            ([0, 0], 3),
            ([1, 0], 2),
            ([1, 1], 3),
            ([2, 0], 2),
            ([2, 1], 2),
        ],
    ];
    for set in sets {
        let space = GradedSpace::try_new(
            Arc::clone(&provider),
            set.iter().map(|&(labels, d)| (labels.to_vec(), d)),
        )
        .unwrap();
        let (p, l) = (pooled(), locked());
        assert_pooled_within_locked(
            &format!("SU(3) {} sectors", set.len()),
            six!(fixture!(&p, &space)),
            six!(fixture!(&l, &space)),
        );
    }
}
