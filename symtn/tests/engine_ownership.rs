//! The caller's engine owns every resource an operation runs on.
//!
//! - One configuration (engine width, inputs) gives identical bits on every
//!   repetition, at widths 1, 4 and 8; across widths the results agree to
//!   the numerical tolerance only.
//! - Two engines on two threads, sharing one Runtime and its tensors, run at
//!   the same time and both return the bits of the same width run alone.
//! - A broken engine is refused with a typed error; symtn does not fall back
//!   to a serial engine of its own.

use std::sync::{Arc, Barrier};

use symtn::sector::{product_sector, SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use symtn::typed::{
    ContractSpec, CpuEngine, Direction, Engine, GradedSpace, Pool, Runtime, TensorMap,
};

use crate::numerics;

type U1Su2 = symtn::sector::ProductFusionRule<U1FusionRule, SU2FusionRule>;

/// U(1) x SU(2) with degeneracies large enough that a threaded engine has
/// real work to split in the recoupling replay, the contraction and the
/// factorization.
fn space() -> GradedSpace<U1Su2> {
    let provider = Arc::new(U1Su2::try_new(U1FusionRule, SU2FusionRule).unwrap());
    let sector =
        |q, twice_spin| product_sector(U1Irrep::new(q), SU2Irrep::from_twice_spin(twice_spin));
    GradedSpace::try_new(
        provider,
        [
            (sector(0, 0), 18),
            (sector(1, 1), 14),
            (sector(-1, 1), 14),
            (sector(0, 2), 12),
        ],
    )
    .unwrap()
}

fn engine(threads: usize) -> CpuEngine {
    if threads == 1 {
        CpuEngine::serial()
    } else {
        CpuEngine::new(Pool::builder().threads(threads).build().unwrap())
    }
}

/// A permute, a contraction and a compact SVD of `t`, as one payload.
fn workload(engine: &mut dyn Engine, t: &TensorMap<U1Su2, f64>) -> Vec<f64> {
    let permuted = t.permute(engine, &[2, 0], &[1]).unwrap();
    let adjoint = t.adjoint(engine).unwrap();
    let gram = t.compose(engine, &adjoint).unwrap();
    let svd = t.svd_compact(engine, &[0, 1], &[2]).unwrap();
    let mut out = permuted.dense_data().unwrap().to_vec();
    out.extend_from_slice(gram.dense_data().unwrap());
    for entry in svd.s.diagview(engine).unwrap() {
        out.extend_from_slice(&entry.values);
    }
    out
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|x| x.to_bits()).collect()
}

fn fixture(runtime: &Runtime) -> TensorMap<U1Su2, f64> {
    let v = space();
    TensorMap::rand_with_seed(&mut CpuEngine::serial(), runtime, [&v, &v], [&v], 1531).unwrap()
}

#[test]
fn one_configuration_repeats_its_bits_at_every_width() {
    let runtime = Runtime::builder().build().unwrap();
    let t = fixture(&runtime);
    let serial = workload(&mut CpuEngine::serial(), &t);
    for threads in [1, 4, 8] {
        let mut engine = engine(threads);
        let first = workload(&mut engine, &t);
        let second = workload(&mut engine, &t);
        let fresh = workload(&mut self::engine(threads), &t);
        assert_eq!(
            bits(&first),
            bits(&second),
            "width {threads}: repeated call"
        );
        assert_eq!(bits(&first), bits(&fresh), "width {threads}: fresh engine");
        numerics::assert_slices_close(&format!("width {threads} vs 1"), &first, &serial, 64);
    }
}

#[test]
fn two_engines_on_two_threads_share_one_runtime() {
    let runtime = Runtime::builder().build().unwrap();
    let t = fixture(&runtime);
    let alone = workload(&mut engine(4), &t);
    let start = Barrier::new(2);
    let results: Vec<Vec<f64>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..2)
            .map(|_| {
                scope.spawn(|| {
                    let mut engine = engine(4);
                    start.wait();
                    workload(&mut engine, &t)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for result in &results {
        assert_eq!(bits(result), bits(&alone));
    }
}

#[test]
fn a_broken_engine_is_refused_with_a_typed_error() {
    let runtime = Runtime::builder().build().unwrap();
    let t = fixture(&runtime);
    for threads in [1, 4] {
        let mut engine = engine(threads);
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            engine.with_linalg_pool(|_, _| -> Result<(), _> { panic!("break the engine") })
        }));
        assert!(panicked.is_err());
        assert!(engine.is_broken());
        assert!(
            t.svd_compact(&mut engine, &[0, 1], &[2]).is_err(),
            "width {threads}: svd"
        );
        assert!(
            t.permute(&mut engine, &[2, 0], &[1]).is_err(),
            "width {threads}: permute"
        );
        let adjoint = t.adjoint(&mut CpuEngine::serial()).unwrap();
        assert!(
            t.compose(&mut engine, &adjoint).is_err(),
            "width {threads}: compose"
        );
        // Host value operations whose arithmetic needs no token admit the
        // engine too: a value result, a destination write, an in-place
        // update, and the compact-diagonal arms that return before any
        // storage kernel.
        let refused = |result: Result<(), String>, what: &str| {
            let message = result.expect_err(&format!("width {threads}: {what}"));
            assert!(
                message.contains("host_region"),
                "width {threads}: {what}: {message}"
            );
        };
        let err = |e: &dyn std::fmt::Debug| format!("{e:?}");
        refused(
            t.scale(&mut engine, 2.0).map(drop).map_err(|e| err(&e)),
            "scale",
        );
        refused(
            t.norm(&mut engine, 2.0).map(drop).map_err(|e| err(&e)),
            "norm",
        );
        refused(
            t.inner(&mut engine, &t).map(drop).map_err(|e| err(&e)),
            "inner",
        );
        refused(
            t.zeros_like(&mut engine).map(drop).map_err(|e| err(&e)),
            "zeros_like",
        );
        // Short circuits that return a clone without a kernel admit it the
        // same way: identity permute, transpose, braid and repartition, and
        // an empty twist.
        refused(
            t.permute(&mut engine, &[0, 1], &[2])
                .map(drop)
                .map_err(|e| err(&e)),
            "identity permute",
        );
        refused(
            t.transpose(&mut engine, &[0, 1], &[2])
                .map(drop)
                .map_err(|e| err(&e)),
            "identity transpose",
        );
        refused(
            t.braid(&mut engine, &[0, 1], &[2], &[0, 1, 2])
                .map(drop)
                .map_err(|e| err(&e)),
            "identity braid",
        );
        refused(
            t.repartition(&mut engine, 2).map(drop).map_err(|e| err(&e)),
            "identity repartition",
        );
        refused(
            t.twist(&mut engine, &[], symtn::typed::Direction::Forward)
                .map(drop)
                .map_err(|e| err(&e)),
            "empty twist",
        );
        let mut destination = t.scale(&mut CpuEngine::serial(), 1.0).unwrap();
        let before = destination.dense_data().unwrap().to_vec();
        refused(
            t.axpby_into(&mut engine, &mut destination, 1.0, 1.0)
                .map_err(|e| err(&e)),
            "axpby_into",
        );
        refused(
            destination
                .scale_assign(&mut engine, 2.0)
                .map_err(|e| err(&e)),
            "scale_assign",
        );
        assert_eq!(destination.dense_data().unwrap(), before.as_slice());
        let s = t
            .svd_compact(&mut CpuEngine::serial(), &[0, 1], &[2])
            .unwrap()
            .s;
        refused(
            s.norm(&mut engine, 2.0).map(drop).map_err(|e| err(&e)),
            "compact norm",
        );
        refused(
            s.scale(&mut engine, 2.0).map(drop).map_err(|e| err(&e)),
            "compact scale",
        );
        refused(
            s.axpby(&mut engine, 1.0, &s, 1.0)
                .map(drop)
                .map_err(|e| err(&e)),
            "compact axpby",
        );
    }
}

/// What: every compact (diagonal) numerical arm admits the caller's engine
/// before its spectrum arithmetic, so a broken engine is refused whatever
/// the representation: clone shortcuts, a swap read straight off the
/// spectrum, the compact adjoint, compose and contract, and a compact-source
/// `permute_into`, whose destination keeps its bits. (Basis: the review probe
/// `/private/tmp/symtn-l1-rereview-probe`.)
#[test]
fn compact_arms_refuse_a_broken_engine_before_any_write() {
    let runtime = Runtime::builder().build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let good = &mut CpuEngine::serial();
    let a = TensorMap::<_, f64>::from_subblock_fn(good, &runtime, [&leg], [&leg], |_, i| {
        (i[0] + 2 * i[1] + 1) as f64
    })
    .unwrap();
    let s = a.svd_compact(good, &[0], &[1]).unwrap().s;
    assert!(
        s.dense_data().is_err(),
        "a compact diagonal stores no dense payload"
    );
    let mut destination = s.materialize(good).unwrap().zeros_like(good).unwrap();
    let before = destination.dense_data().unwrap().to_vec();

    let mut engine = CpuEngine::serial();
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        engine.with_linalg_pool(|_, _| -> Result<(), _> { panic!("break the engine") })
    }));
    assert!(panicked.is_err() && engine.is_broken());
    let engine = &mut engine;
    let spec = ContractSpec {
        lhs: &[1],
        rhs: &[0],
        codomain: &[0],
        domain: &[1],
    };
    assert!(s.permute(engine, &[0], &[1]).is_err(), "identity permute");
    assert!(s.permute(engine, &[1], &[0]).is_err(), "swap");
    assert!(
        s.twist(engine, &[], Direction::Forward).is_err(),
        "empty twist"
    );
    assert!(s.adjoint(engine).is_err(), "compact adjoint");
    assert!(s.compose(engine, &s).is_err(), "compact compose");
    assert!(s.contract(engine, &s, &spec).is_err(), "compact contract");
    assert!(s
        .permute_into(engine, &[0], &[1], &mut destination, 1.0, 0.0)
        .is_err());
    assert_eq!(destination.dense_data().unwrap(), before.as_slice());
    // A dense lazy adjoint moves no payload and needs nothing of the engine.
    assert!(a.adjoint(engine).is_ok());
}
