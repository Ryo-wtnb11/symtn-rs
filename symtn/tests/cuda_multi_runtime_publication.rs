//! Public-API gate for #1384: a fresh device output returned by one `Runtime`
//! must be fully written before a thread that also uses a second `Runtime` on
//! the same device reads it. It covers fresh outputs only; an overwrite
//! destination or reused buffer is covered by `cuda_overwrite_publication.rs`
//! (#1391).
//!
//! `symtn-dense/tests/cuda_multi_context_publication.rs` forces the exact
//! interleaving at the adapter seam. Here the same steps run through
//! `Runtime`, where step 2 cannot be placed inside `compose` directly, so the
//! thread-B delay is swept across the late part of the call instead:
//!
//! 1. thread A uploads a tiny `T` through R2, then runs `compose` through R1
//!    (zero upload binds `O`, then one GEMM per sector block writes it);
//! 2. thread B, after the swept delay, reads `T` through R2; if that lands
//!    between `O`'s bind and a later GEMM, B's stream records itself synced to
//!    A's past `O`'s bind cursor (tensor4all/cubecl#16);
//! 3. B reads `O` through R1 and compares with the hand oracle: all-ones
//!    operands, so every element equals the sector degeneracy.
//!
//! Without the per-device lock (`runtime.rs`, `cuda_device_lock`) this reads a
//! stale `O` (13/120 iterations on an A100). The lock makes step 2 wait for
//! `compose`'s whole lease, so B's sync cannot land between `O`'s bind and write.
//! The one-Runtime control reads `T` through R1, whose lease orders it after
//! `compose`, and must always pass. Both assert no stale read. Sensitivity is
//! timing-based: the sweep is placed from one measured `compose`, so on other
//! hardware a pre-fix build may miss the window; a pass there is not proof.
//!
//! The ordering test runs many threads over two Runtimes on one device,
//! interleaving uploads, `compose`, cross-Runtime reads and device
//! maintenance, and checks it neither deadlocks (bounded by a measured serial
//! run) nor departs from Host.
//!
//! Run with `cargo test --release -p symtn --no-default-features --features \
//! cuda,cpu-faer --test integration cuda_multi_runtime_publication:: -- --ignored \
//! --nocapture --test-threads=1` on a CUDA host (release: the gate needs the
//! write to still be in flight). A debug build also trips an
//! unrelated `u32` overflow in CubeCL's staged-upload counter
//! (`drop_queue/policy.rs:36`) once 4 GiB have been uploaded.

#![cfg(feature = "cuda")]

use std::sync::{mpsc, Arc, Barrier};
use std::time::{Duration, Instant};

use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{CudaStorage, GradedSpace, Runtime, TensorMap};

const SECTORS: i32 = 2;
const DEG: usize = 4096;
const ITERS: usize = 120;

type Host = TensorMap<U1FusionRule, f64>;
type Device = TensorMap<U1FusionRule, f64, CudaStorage<f64>>;

fn ones(runtime: &Runtime, sectors: i32, deg: usize) -> Host {
    let space = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        (0..sectors).map(|q| (U1Irrep::new(q), deg)),
    )
    .unwrap();
    TensorMap::from_subblock_fn(runtime, [&space], [&space], |_, _| 1.0).unwrap()
}

/// Host wall time of one warm `compose`, which B's delay sweep spans.
fn compose_time(runtime: &Runtime) -> Duration {
    let lhs = ones(runtime, SECTORS, DEG).to_cuda().unwrap();
    lhs.compose(&lhs).unwrap().to_host().unwrap();
    let start = Instant::now();
    let o = lhs.compose(&lhs).unwrap();
    let elapsed = start.elapsed();
    o.to_host().unwrap();
    elapsed
}

/// B's delay for `iter`: the window sits late in `compose` (after the zero
/// upload), so the sweep covers its last 30%.
fn delay(iter: usize, compose: Duration) -> Duration {
    compose.mul_f64(0.7 + 0.3 * iter as f64 / ITERS as f64)
}

fn run(r1: Runtime, r2: Runtime) -> usize {
    let compose = compose_time(&r1);
    println!("compose takes {compose:?}");
    let (t_tx, t_rx) = mpsc::channel::<Device>();
    let (o_tx, o_rx) = mpsc::channel::<Device>();
    let (done_tx, done_rx) = mpsc::channel::<()>();
    let a = std::thread::spawn(move || {
        let lhs = ones(&r1, SECTORS, DEG).to_cuda().unwrap();
        let tiny = ones(&r2, 1, 1);
        for _ in 0..ITERS {
            t_tx.send(tiny.to_cuda().unwrap()).unwrap();
            o_tx.send(lhs.compose(&lhs).unwrap()).unwrap();
            done_rx.recv().unwrap();
        }
    });
    let mut stale = 0;
    for iter in 0..ITERS {
        let t = t_rx.recv().unwrap();
        std::thread::sleep(delay(iter, compose));
        t.to_host().unwrap();
        let host = o_rx.recv().unwrap().to_host().unwrap();
        let data = host.dense_data().unwrap();
        let wrong = data.iter().filter(|&&x| x != DEG as f64).count();
        if wrong > 0 {
            stale += 1;
            let first = data.iter().position(|&x| x != DEG as f64).unwrap();
            println!(
                "iter {iter} (delay {:?}): {wrong}/{} stale, first [{first}] = {}",
                delay(iter, compose),
                data.len(),
                data[first]
            );
        }
        done_tx.send(()).unwrap();
    }
    a.join().unwrap();
    stale
}

#[test]
#[ignore = "requires a CUDA device"]
fn second_runtime_on_one_device_reads_only_completed_outputs() {
    let r1 = Runtime::builder().cuda(0).build().unwrap();
    let r2 = Runtime::builder().cuda(0).build().unwrap();
    let stale = run(r1, r2);
    println!("two runtimes: {stale}/{ITERS} iterations read a stale output");
    assert_eq!(stale, 0, "#1384 reproduced: {stale}/{ITERS} stale reads");
}

#[test]
#[ignore = "requires a CUDA device"]
fn single_runtime_control_reads_only_completed_outputs() {
    let r = Runtime::builder().cuda(0).build().unwrap();
    let stale = run(r.clone(), r);
    println!("one runtime: {stale}/{ITERS} iterations read a stale output");
    assert_eq!(stale, 0);
}

const THREADS: usize = 8;
const ROUNDS: usize = 20;
const SMALL: usize = 48;

fn graded(runtime: &Runtime, seed: usize) -> Host {
    let space = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        (0..3).map(|q| (U1Irrep::new(q), SMALL)),
    )
    .unwrap();
    TensorMap::from_subblock_fn(runtime, [&space], [&space], |_, index| {
        ((index[0] * 7 + index[1] * 3 + seed) % 11) as f64 - 5.0
    })
    .unwrap()
}

/// One thread's work: upload through one Runtime, compose, read the output
/// back through the same Runtime after touching the other one, and run the
/// other Runtime's maintenance paths in between.
fn round(own: &Runtime, other: &Runtime, seed: usize) {
    let host = graded(own, seed);
    let expected = host.compose(&host).unwrap();
    let device = host.to_cuda().unwrap();
    let out = device.compose(&device).unwrap();
    graded(other, seed).to_cuda().unwrap().to_host().unwrap();
    other.cuda_tree_transform_stats().unwrap();
    if seed.is_multiple_of(5) {
        other.clear_cuda_caches();
    }
    assert_eq!(
        out.to_host().unwrap().dense_data().unwrap(),
        expected.dense_data().unwrap()
    );
}

fn interleave(r1: &Runtime, r2: &Runtime, threads: usize) {
    let barrier = Arc::new(Barrier::new(threads));
    let handles: Vec<_> = (0..threads)
        .map(|t| {
            let (own, other) = if t.is_multiple_of(2) {
                (r1.clone(), r2.clone())
            } else {
                (r2.clone(), r1.clone())
            };
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                for r in 0..ROUNDS {
                    round(&own, &other, t * ROUNDS + r);
                }
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn two_runtimes_interleaved_on_one_device_finish_and_match_host() {
    let r1 = Runtime::builder().cuda(0).build().unwrap();
    let r2 = Runtime::builder().cuda(0).build().unwrap();
    interleave(&r1, &r2, 1);
    let start = Instant::now();
    interleave(&r1, &r2, 1);
    // The device lock serializes enqueue, so THREADS threads do at most
    // THREADS times the serial work; the factor 10 over that is slack for
    // scheduling, not a platform constant.
    let bound = start.elapsed() * (10 * THREADS as u32);
    let (tx, rx) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        interleave(&r1, &r2, THREADS);
        tx.send(()).unwrap();
    });
    match rx.recv_timeout(bound) {
        // Joined so both Runtimes are torn down before the process exits;
        // a detached CUDA teardown racing process exit crashes the driver.
        Ok(()) => worker.join().unwrap(),
        Err(mpsc::RecvTimeoutError::Timeout) => panic!("no progress within {bound:?}: deadlock"),
        Err(mpsc::RecvTimeoutError::Disconnected) => panic!("a worker failed"),
    }
}
