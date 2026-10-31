//! Standalone ops on a SHARED `Runtime` must be correct and deterministic when
//! driven from many threads at once, each thread on its own engine:
//! each op leases a per-rule execution context from the Runtime's pool
//! instead of holding a coarse runtime lock, so this exercises that the
//! pooled machinery never leaks state across concurrent ops.
//!
//! Determinism: every thread runs the SAME deterministic computation on its own
//! seeded operands and its own serial engine, so every result must be
//! BIT-identical to the single-threaded reference (a leased context that
//! carried torn state from another thread would change the value or panic).

use std::sync::Arc;
use std::thread;
use symtn::typed::ContractSpec;

use symtn::sector::{SU2FusionRule, SU2Irrep};
use symtn::typed::{GradedSpace, Runtime, TensorMap};

fn space() -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 3),
            (SU2Irrep::from_twice_spin(1), 2),
            (SU2Irrep::from_twice_spin(2), 1),
        ],
    )
    .unwrap()
}

/// A rank-4 contract, a permute, and a factorization (each hits a different
/// pool) on freshly seeded operands; returns (contract norm, first singular
/// value) as a deterministic fingerprint of the results.
fn work(rt: &Runtime, seed: u64) -> (f64, f64) {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let v = space();
    let a: TensorMap<_, f64> =
        TensorMap::rand_with_seed(engine, rt, [&v, &v], [&v, &v], seed).unwrap();
    let b: TensorMap<_, f64> =
        TensorMap::rand_with_seed(engine, rt, [&v, &v], [&v, &v], seed + 1).unwrap();
    let c = a
        .contract(
            engine,
            &b,
            &ContractSpec {
                lhs: &[2, 3],
                rhs: &[0, 1],
                codomain: &[0, 1],
                domain: &[2, 3],
            },
        )
        .unwrap(); // ContextPool
    let p = c.permute(engine, &[1, 0], &[3, 2]).unwrap(); // ContextPool
    let s = p.svd_vals(engine, &[0, 1], &[2, 3]).unwrap(); // per-call dense executor
    let first = s
        .first()
        .and_then(|spec| spec.values.first().copied())
        .unwrap();
    (c.norm(engine, 2.0).unwrap(), first)
}

#[test]
fn shared_runtime_concurrent_ops_match_serial_bit_for_bit() {
    let rt = Runtime::builder().build().unwrap();

    // Single-threaded references, one per distinct seed.
    let seeds: Vec<u64> = (0..8).map(|t| 100 + t * 2).collect();
    let refs: Vec<(f64, f64)> = seeds.iter().map(|&s| work(&rt, s)).collect();

    // Same ops, from 8 threads sharing the one runtime, each looping so the
    // pools churn (lease/return under contention). Every result must equal its
    // serial reference exactly.
    let rt = Arc::new(rt);
    let refs = Arc::new(refs);
    let seeds = Arc::new(seeds);
    let handles: Vec<_> = (0..8usize)
        .map(|t| {
            let rt = Arc::clone(&rt);
            let refs = Arc::clone(&refs);
            let seeds = Arc::clone(&seeds);
            thread::spawn(move || {
                for _ in 0..64 {
                    let got = work(&rt, seeds[t]);
                    assert_eq!(
                        got, refs[t],
                        "thread {t}: concurrent op result diverged from the serial reference"
                    );
                }
            })
        })
        .collect();
    for h in handles {
        h.join()
            .expect("worker thread panicked (pool state corrupted?)");
    }
}
