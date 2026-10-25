//! Gate for #1384/#1391 at the adapter seam: a device output written by one
//! `CudaDenseContext` is never read early by a thread that also uses a second
//! context on the same device, even with nothing serializing the two contexts
//! (no symtn `Runtime` device lock).
//!
//! Both contexts share CubeCL's process-wide per-device client. CubeCL records
//! a binding's cursor only at bind, so a later write into that binding is not
//! published (tensor4all/cubecl#16). The test forces the interleaving from the
//! issue, step by step, with the same seam `TypedTensor::contract` uses
//! (`upload_owned` zeros, then GEMM, both under one context lock):
//!
//! 1. thread A, holding context 1, uploads output `O` (bind at cursor `k`);
//! 2. thread B, holding context 2, reads a fresh A-origin tensor `T`, which
//!    records `B.last_synced[A] >= k`;
//! 3. A enqueues a long accumulating GEMM into `O` and releases context 1;
//! 4. B takes context 1 and downloads `O`: `k <= last_synced[A]`, so on
//!    separate streams B would skip the event and race the GEMM.
//!
//! On separate streams this read a stale `O` (40/40 iterations on an A100).
//! `CudaDenseContext::new` pins CubeCL to one stream per device
//! (`streaming.max_streams = 1`, #1391), so A's GEMM and B's download run in
//! enqueue order on that stream and the skipped event does not matter. The
//! forced case asserts no stale read; it does not detect a cubecl#16 fix,
//! which only matters if the single stream is lifted.
//!
//! The oracle is a hand calculation: all-ones operands, so every element of
//! `O` is exactly `REPS * N`. The single-context control runs the same steps
//! with one shared lock, which forces step 2 after step 3. The no-sync control
//! keeps two contexts but skips step 2.
//!
//! Run with `cargo test -p symtn-dense --features cuda,cpu-faer --test \
//! integration cuda_multi_context_publication:: -- --ignored --nocapture --test-threads=1`.

#![cfg(feature = "cuda")]

use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use symtn_dense::{cuda_gemm_region_into, CudaDenseContext, CudaDenseStorage};

const N: usize = 2048;
const REPS: usize = 8;
const ITERS: usize = 40;

type Shared = Arc<Mutex<CudaDenseContext>>;

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    /// Two contexts, step 2 forced between bind and write.
    Forced,
    /// Two contexts, step 2 skipped.
    NoSync,
    /// One context: the shared lock orders step 2 after the write.
    Single,
}

fn accumulate(ctx: &mut CudaDenseContext, dst: &mut CudaDenseStorage, ones: &CudaDenseStorage) {
    for _ in 0..REPS {
        cuda_gemm_region_into::<f64>(ctx, dst, 0, N, ones, 0, N, ones, 0, N, N, N, N, 1.0, 1.0)
            .expect("gemm");
    }
}

/// Runs the forced interleaving and returns the number of iterations whose
/// download of `O` differed from the oracle.
fn run(ctx1: Shared, ctx2: Shared, mode: Mode) -> usize {
    let forced = mode == Mode::Forced;
    let expected = (REPS * N) as f64;
    let (t_tx, t_rx) = mpsc::channel::<CudaDenseStorage>();
    let (ack_tx, ack_rx) = mpsc::channel::<()>();
    let (o_tx, o_rx) = mpsc::channel::<CudaDenseStorage>();
    let (done_tx, done_rx) = mpsc::channel::<()>();

    let a_ctx1 = ctx1.clone();
    let a_ctx2 = ctx2.clone();
    let a = std::thread::spawn(move || {
        let ones = {
            let ctx = a_ctx1.lock().unwrap();
            CudaDenseStorage::upload_owned(&ctx, vec![1.0f64; N * N]).expect("upload ones")
        };
        for _ in 0..ITERS {
            let t = {
                let ctx = a_ctx2.lock().unwrap();
                CudaDenseStorage::upload_owned(&ctx, vec![0.0f64; 1]).expect("upload t")
            };
            let mut lease = a_ctx1.lock().unwrap();
            let mut o =
                CudaDenseStorage::upload_owned(&lease, vec![0.0f64; N * N]).expect("upload o");
            t_tx.send(t).unwrap();
            if forced {
                // The controls cannot wait here: B's step 2 needs this lock.
                ack_rx.recv().unwrap();
            }
            accumulate(&mut lease, &mut o, &ones);
            drop(lease);
            o_tx.send(o).unwrap();
            done_rx.recv().unwrap();
        }
    });

    let mut stale = 0;
    for iter in 0..ITERS {
        let t = t_rx.recv().unwrap();
        if mode != Mode::NoSync {
            let ctx = ctx2.lock().unwrap();
            t.download::<f64>(&ctx).expect("download t");
        }
        if forced {
            ack_tx.send(()).unwrap();
        }
        let o = o_rx.recv().unwrap();
        let start = Instant::now();
        let host = {
            let ctx = ctx1.lock().unwrap();
            o.download::<f64>(&ctx).expect("download o")
        };
        let elapsed = start.elapsed();
        let wrong = host.iter().filter(|&&x| x != expected).count();
        if wrong > 0 {
            stale += 1;
            let first = host.iter().position(|&x| x != expected).unwrap();
            println!(
                "iter {iter}: {wrong}/{} stale elements, first [{first}] = {} (expected {expected}), download {elapsed:?}",
                host.len(),
                host[first]
            );
        }
        done_tx.send(()).unwrap();
    }
    a.join().unwrap();
    stale
}

/// Host-observed GPU time of one `accumulate`, i.e. how long the write into
/// `O` stays in flight after its enqueue returns.
fn window(ctx: &Shared) {
    let mut ctx = ctx.lock().unwrap();
    let ones = CudaDenseStorage::upload_owned(&ctx, vec![1.0f64; N * N]).unwrap();
    let mut o = CudaDenseStorage::upload_owned(&ctx, vec![0.0f64; N * N]).unwrap();
    o.download::<f64>(&ctx).unwrap();
    let start = Instant::now();
    accumulate(&mut ctx, &mut o, &ones);
    let enqueued = start.elapsed();
    o.download::<f64>(&ctx).unwrap();
    println!(
        "window: enqueue returned after {enqueued:?}, write + download done after {:?}",
        start.elapsed()
    );
}

fn context() -> Shared {
    Arc::new(Mutex::new(CudaDenseContext::new(0).expect("cuda context")))
}

#[test]
#[ignore = "requires a CUDA device"]
fn second_context_forced_interleaving_reads_only_completed_outputs() {
    let ctx1 = context();
    let ctx2 = context();
    window(&ctx1);
    let stale = run(ctx1, ctx2, Mode::Forced);
    println!("two contexts: {stale}/{ITERS} iterations read a stale output");
    assert_eq!(
        stale, 0,
        "cross-stream stale read: is CubeCL still on one stream?"
    );
}

#[test]
#[ignore = "requires a CUDA device"]
fn single_context_control_reads_only_completed_outputs() {
    let ctx = context();
    let stale = run(ctx.clone(), ctx, Mode::Single);
    println!("single context: {stale}/{ITERS} iterations read a stale output");
    assert_eq!(stale, 0);
}

#[test]
#[ignore = "requires a CUDA device"]
fn two_context_control_without_cursor_sync_reads_only_completed_outputs() {
    let stale = run(context(), context(), Mode::NoSync);
    println!("two contexts, no sync: {stale}/{ITERS} iterations read a stale output");
    assert_eq!(stale, 0);
}
