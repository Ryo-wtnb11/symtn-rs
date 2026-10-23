//! #1361: a compact QR or SVD enters one CPU linear-algebra session per call,
//! however many coupled sectors it factorizes, under both the default and the
//! one-thread CPU layout.
//!
//! Routes: the multiplicity-free fixtures are canonical coupled-sector matrix
//! layouts, so they take the direct-region QR and SVD. The matricization,
//! Generic and checked-Generic QR routes are gated in
//! `symtn-matrixalgebra/tests/factorization_session_scope.rs`.
//!
//! Its own test binary: `cpu_session_stats` is a process-wide counter, so a
//! delta means something only when no unrelated test opens a session in the
//! same process. The mutex serializes the readers inside this binary.

#![cfg(all(feature = "cpu-faer", not(feature = "provider-inject")))]

use std::sync::Mutex;
use symtn::typed::HermitianTol;

use symtn::expert::cpu_session_stats;
use symtn::sector::{
    FermionParityFusionRule, ProductFusionRule, ProductSector, SU2FusionRule, SU2Irrep,
    U1FusionRule, U1Irrep, Z2Irrep,
};
use symtn::typed::{Complex64, Eigh, GradedSpace, Lq, Runtime, TensorMap};

#[path = "../../tests/support/fixtures.rs"]
mod fixtures;

use fixtures::{codomain_axes, domain_axes};

static COUNTER_LOCK: Mutex<()> = Mutex::new(());

fn sessions_during<T>(call: impl FnOnce() -> T) -> (u64, T) {
    let before = cpu_session_stats().sessions_opened;
    let value = call();
    (cpu_session_stats().sessions_opened - before, value)
}

macro_rules! assert_one_session_per_factorization {
    ($runtime:expr, $provider:expr, $dtype:ty, $sectors:expr, $nc:expr, $nd:expr) => {{
        let _guard = COUNTER_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let leg = GradedSpace::try_new(
            std::sync::Arc::new($provider),
            $sectors.into_iter().map(|s| (s, 2)),
        )
        .unwrap();
        let tensor =
            TensorMap::<_, $dtype>::rand_with_seed($runtime, vec![&leg; $nc], vec![&leg; $nd], 7)
                .unwrap();
        let mut coupled = (0..tensor.subblock_count())
            .map(|i| tensor.subblock_fusion_trees(i).unwrap().coupled().clone())
            .collect::<Vec<_>>();
        coupled.sort();
        coupled.dedup();
        assert!(coupled.len() > 2, "fixture must span many sectors");

        let (sessions, _) = sessions_during(|| {
            tensor
                .qr_compact(&codomain_axes(&tensor), &domain_axes(&tensor))
                .unwrap()
        });
        assert_eq!(sessions, 1, "qr_compact");

        let (sessions, _) = sessions_during(|| {
            tensor
                .svd_compact(&codomain_axes(&tensor), &domain_axes(&tensor))
                .unwrap()
        });
        assert_eq!(sessions, 1, "svd_compact");
    }};
}

fn runtimes() -> [Runtime; 2] {
    [
        Runtime::builder().build().unwrap(),
        Runtime::builder().dense_threads(1).build().unwrap(),
    ]
}

fn centered(count: i32) -> impl Iterator<Item = i32> {
    let low = -((count - 1) / 2);
    (0..count).map(move |i| low + i)
}

#[test]
fn u1_compact_factorizations_open_one_session() {
    let sectors = || centered(8).map(U1Irrep::new).collect::<Vec<_>>();
    for runtime in runtimes() {
        assert_one_session_per_factorization!(&runtime, U1FusionRule, f64, sectors(), 1, 1);
        assert_one_session_per_factorization!(&runtime, U1FusionRule, Complex64, sectors(), 2, 2);
    }
}

#[test]
fn su2_compact_factorizations_open_one_session() {
    let sectors = || (0..4).map(SU2Irrep::from_twice_spin).collect::<Vec<_>>();
    for runtime in runtimes() {
        assert_one_session_per_factorization!(&runtime, SU2FusionRule, f64, sectors(), 2, 1);
        assert_one_session_per_factorization!(&runtime, SU2FusionRule, Complex64, sectors(), 2, 2);
    }
}

#[test]
fn fz2_u1_compact_factorizations_open_one_session() {
    type Fz2U1 = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
    let sectors = || {
        centered(6)
            .map(|q| {
                let parity = if q.rem_euclid(2) == 0 {
                    Z2Irrep::EVEN
                } else {
                    Z2Irrep::ODD
                };
                ProductSector::new(parity, U1Irrep::new(q))
            })
            .collect::<Vec<_>>()
    };
    let rule = || Fz2U1::try_new(FermionParityFusionRule, U1FusionRule).unwrap();
    for runtime in runtimes() {
        assert_one_session_per_factorization!(&runtime, rule(), f64, sectors(), 1, 1);
        assert_one_session_per_factorization!(&runtime, rule(), Complex64, sectors(), 2, 1);
    }
}

/// #1389: the streaming factorizations enter a Tenferro execution scope that
/// holds the process permit for the whole call. Calls from a worker of a
/// foreign Rayon pool, inside a `rayon::join`, and concurrent calls from
/// plain threads must serialize on that permit without deadlock or panic and
/// return the serial results.
///
/// Not covered: concurrent calls from several workers of one foreign pool.
/// A worker blocked in Tenferro's pool install steals a sibling job that
/// re-enters Tenferro, which panics in Tenferro's re-entry guard with or
/// without the scope.
#[test]
fn streaming_factorizations_from_rayon_workers_and_threads_match_serial_results() {
    let _guard = COUNTER_LOCK.lock().unwrap_or_else(|p| p.into_inner());
    let runtime = Runtime::builder().build().unwrap();
    let leg = GradedSpace::try_new(
        std::sync::Arc::new(U1FusionRule),
        centered(5).map(|q| (U1Irrep::new(q), 2)),
    )
    .unwrap();
    let tensors = (0..6)
        .map(|seed| TensorMap::<_, f64>::rand_with_seed(&runtime, [&leg, &leg], [&leg], seed))
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let run = |tensor: &TensorMap<U1FusionRule, f64>| {
        let Lq { l, q } = tensor.lq_compact(&[0, 1], &[2]).unwrap();
        let null = tensor.left_null(&[0, 1], &[2]).unwrap();
        let Eigh { d: w, v } = tensor
            .compose(&tensor.adjoint().unwrap())
            .unwrap()
            .eigh_full(&[0, 1], &[2, 3], HermitianTol::DEFAULT)
            .unwrap();
        [l, q, null, w, v]
            .iter()
            .flat_map(|factor| {
                factor
                    .materialize()
                    .unwrap()
                    .dense_data()
                    .unwrap()
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<u64>>()
    };
    let serial = tensors.iter().map(run).collect::<Vec<_>>();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    let pooled = pool.install(|| run(&tensors[1]));
    assert_eq!(pooled, serial[1]);
    let joined = pool.install(|| rayon::join(|| run(&tensors[2]), || (0..1000u64).sum::<u64>()));
    assert_eq!(joined.0, serial[2]);
    let threaded = std::thread::scope(|scope| {
        let handles = tensors
            .iter()
            .map(|tensor| scope.spawn(|| run(tensor)))
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert_eq!(threaded, serial);
}
