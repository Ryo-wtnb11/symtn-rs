//! Streaming factorizations called from a worker of a foreign Rayon
//! pool, inside a `rayon::join`, and concurrently from plain threads, each
//! caller on its own engine over one shared Runtime, return the results of
//! the same calls run serially on one thread.

use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::HermitianTol;
use symtn::typed::{CpuEngine, Eigh, Engine, GradedSpace, Lq, Runtime, TensorMap};

fn centered(count: i32) -> impl Iterator<Item = i32> {
    let low = -((count - 1) / 2);
    (0..count).map(move |i| low + i)
}

/// The bits of every factor of `tensor`'s LQ, left null space and the
/// eigendecomposition of `tensor tensor†`.
fn run(engine: &mut dyn Engine, tensor: &TensorMap<U1FusionRule, f64>) -> Vec<u64> {
    let Lq { l, q } = tensor.lq_compact(engine, &[0, 1], &[2]).unwrap();
    let null = tensor.left_null(engine, &[0, 1], &[2]).unwrap();
    let adjoint = tensor.adjoint(engine).unwrap();
    let Eigh { d: w, v } = tensor
        .compose(engine, &adjoint)
        .unwrap()
        .eigh_full(engine, &[0, 1], &[2, 3], HermitianTol::DEFAULT)
        .unwrap();
    [l, q, null, w, v]
        .iter()
        .flat_map(|factor| {
            factor
                .materialize(engine)
                .unwrap()
                .dense_data()
                .unwrap()
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        })
        .collect()
}

#[test]
fn streaming_factorizations_from_rayon_workers_and_threads_match_serial_results() {
    let engine = &mut CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let leg = GradedSpace::try_new(
        std::sync::Arc::new(U1FusionRule),
        centered(5).map(|q| (U1Irrep::new(q), 2)),
    )
    .unwrap();
    let tensors = (0..6)
        .map(|seed| {
            TensorMap::<_, f64>::rand_with_seed(engine, &runtime, [&leg, &leg], [&leg], seed)
        })
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    let serial = tensors
        .iter()
        .map(|tensor| run(engine, tensor))
        .collect::<Vec<_>>();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    let pooled = pool.install(|| run(&mut CpuEngine::serial(), &tensors[1]));
    assert_eq!(pooled, serial[1]);
    let joined = pool.install(|| {
        rayon::join(
            || run(&mut CpuEngine::serial(), &tensors[2]),
            || (0..1000u64).sum::<u64>(),
        )
    });
    assert_eq!(joined.0, serial[2]);
    let threaded = std::thread::scope(|scope| {
        let handles = tensors
            .iter()
            .map(|tensor| scope.spawn(|| run(&mut CpuEngine::serial(), tensor)))
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert_eq!(threaded, serial);
}
