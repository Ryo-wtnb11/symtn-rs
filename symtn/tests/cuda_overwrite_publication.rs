//! Probes for #1391: a device buffer bound in one lease and written in a later
//! one is never read stale by another thread of the same `Runtime`
//! (tensor4all/cubecl#16 publishes a binding's stream cursor only at bind).
//!
//! F1, overwrite destination. Thread A runs `D = a.contract(a)` and hands `D`
//! to thread B; B downloads it, which syncs B's stream past `D`'s bind; A then
//! runs a long `b.contract_into(b, &mut D)` and hands `D` back; B
//! downloads `D` again. Without publication B's stream skips the event and
//! copies `D` while the overwrite GEMM is still running. The control skips
//! B's first download, so B's only sync to A's stream is the final one.
//!
//! F2, reused scratch, write-after-read. A general contraction packs its
//! permuted operand into the Runtime's `CudaContractScratch` and GEMMs from
//! it. Thread 1 grows the scratch; thread 2 syncs past its bind once. Then,
//! each iteration, thread 1 enqueues a long contraction of `x1` and thread 2
//! immediately one of `x2`: without publication, thread 2's pack overwrites
//! the scratch while thread 1's GEMM still reads it. Both outputs are checked
//! against hand values. The control runs both contractions on one thread.
//!
//! F1 oracle: a hand calculation (constant operands, every element is the
//! product of the constants times the contracted dimension). F2 oracle: the
//! Host contraction of the same operands.
//!
//! On an A100 before the fix, F1 read a stale destination in 40/40 forced
//! iterations (control 0/40). F2 never showed a wrong value (0/40), although
//! CubeCL's streaming log confirmed thread 2 never waited on thread 1's
//! stream after its first sync: whether thread 2's pack lands between
//! thread 1's GEMMs depends on GPU block scheduling, so F2 is a regression
//! gate here, not a demonstrated reproduction. The fix is structural for
//! both: `CudaDenseContext::new` pins CubeCL to one stream per device, so
//! every write and read runs in enqueue order (0/40 for both after it).
//!
//! Run with `cargo test --release -p symtn --no-default-features --features
//! cuda,cpu-faer --test cuda_overwrite_publication -- --ignored --nocapture
//! --test-threads=1` on a CUDA host (release: the writes must still be in
//! flight when the reader enqueues).

#![cfg(feature = "cuda")]

use std::sync::{mpsc, Arc};

use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{ContractSpec, CudaStorage, GradedSpace, Runtime, TensorMap};

type Host = TensorMap<U1FusionRule, f64>;
type Device = TensorMap<U1FusionRule, f64, CudaStorage<f64>>;

const ITERS: usize = 40;

fn space(sectors: i32, deg: usize) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        (0..sectors).map(|q| (U1Irrep::new(q), deg)),
    )
    .unwrap()
}

fn constant(runtime: &Runtime, codomain: &[&GradedSpace<U1FusionRule>], value: f64) -> Device {
    Host::from_subblock_fn(
        runtime,
        codomain.iter().copied(),
        codomain.iter().copied(),
        |_, _| value,
    )
    .unwrap()
    .to_cuda()
    .unwrap()
}

/// Iterations whose elements differ from `expected`, with a line per miss.
fn check(label: &str, iter: usize, data: &[f64], expected: f64) -> usize {
    let wrong = data.iter().filter(|&&x| x != expected).count();
    if wrong == 0 {
        return 0;
    }
    let first = data.iter().position(|&x| x != expected).unwrap();
    println!(
        "{label} iter {iter}: {wrong}/{} stale, first [{first}] = {} (expected {expected})",
        data.len(),
        data[first]
    );
    1
}

const F1_SECTORS: i32 = 2;
const F1_DEG: usize = 4096;

fn overwrite_destination(forced: bool) -> usize {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let expected = 4.0 * F1_DEG as f64;
    let (to_b, from_a) = mpsc::channel::<Device>();
    let (to_a, from_b) = mpsc::channel::<Device>();
    let (read_tx, read_rx) = mpsc::channel::<()>();
    let writer = std::thread::spawn(move || {
        // Operands bound on the writer's own stream: a foreign-stream operand
        // would make Tenferro host-sync the writer after every GEMM
        // (`finish_vendor_enqueue`), completing the write before B looks.
        let v = space(F1_SECTORS, F1_DEG);
        let a = constant(&runtime, &[&v], 1.0);
        let b = constant(&runtime, &[&v], 2.0);
        for _ in 0..ITERS {
            // B's last read has finished, so in the control B's stream was
            // last synced to A's before `D`'s bind.
            read_rx.recv().unwrap();
            to_b.send(
                a.contract(
                    &a,
                    &ContractSpec {
                        lhs: &[1],
                        rhs: &[0],
                        codomain: &[0],
                        domain: &[1],
                    },
                )
                .unwrap(),
            )
            .unwrap();
            let mut d = from_b.recv().unwrap();
            b.contract_into(
                &b,
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &[0],
                    domain: &[1],
                },
                &mut d,
                1.0,
                0.0,
            )
            .unwrap();
            to_b.send(d).unwrap();
        }
    });
    let mut stale = 0;
    read_tx.send(()).unwrap();
    for iter in 0..ITERS {
        let d = from_a.recv().unwrap();
        if forced {
            d.to_host().unwrap();
        }
        to_a.send(d).unwrap();
        let d = from_a.recv().unwrap();
        stale += check(
            "F1",
            iter,
            d.to_host().unwrap().dense_data().unwrap(),
            expected,
        );
        let _ = read_tx.send(());
    }
    writer.join().unwrap();
    stale
}

#[test]
#[ignore = "requires a CUDA device"]
fn overwrite_destination_read_by_another_thread_is_complete() {
    let stale = overwrite_destination(true);
    println!("F1 forced: {stale}/{ITERS} iterations read a stale destination");
    assert_eq!(
        stale, 0,
        "stale overwrite destination (F1): {stale}/{ITERS}"
    );
}

#[test]
#[ignore = "requires a CUDA device"]
fn overwrite_destination_control_without_early_sync_is_complete() {
    let stale = overwrite_destination(false);
    println!("F1 control: {stale}/{ITERS} iterations read a stale destination");
    assert_eq!(stale, 0);
}

/// `[V, V] <- [V, V]` operands contracted over their domain in reversed
/// order, so an operand is permuted into scratch; one GEMM per coupled sector
/// then reads it, the later ones well after the pack has run.
const F2_SECTORS: i32 = 4;
const F2_DEG: usize = 32;
const F2_SPEC: ContractSpec<'static> = ContractSpec {
    lhs: &[3, 2],
    rhs: &[0, 1],
    codomain: &[0, 1],
    domain: &[2, 3],
};

/// One thread's contraction: operands and destination bound on that thread's
/// stream, and the Host value.
struct F2 {
    x: Device,
    y: Device,
    d: Device,
    expected: Vec<f64>,
}

impl F2 {
    fn new(runtime: &Runtime, value: f64) -> Self {
        let v = space(F2_SECTORS, F2_DEG);
        let host =
            |value| Host::from_subblock_fn(runtime, [&v, &v], [&v, &v], |_, _| value).unwrap();
        let (x, y) = (host(value), host(1.0));
        let expected = x.contract(&y, &F2_SPEC).unwrap();
        let (x, y) = (x.to_cuda().unwrap(), y.to_cuda().unwrap());
        let d = x.contract(&y, &F2_SPEC).unwrap();
        d.to_host().unwrap();
        F2 {
            x,
            y,
            d,
            expected: expected.dense_data().unwrap().to_vec(),
        }
    }

    /// Enqueues the contraction into the retained destination: no zero
    /// upload, so the enqueue returns while the GEMMs still run.
    fn enqueue(&mut self) {
        self.x
            .contract_into(&self.y, &F2_SPEC, &mut self.d, 1.0, 0.0)
            .unwrap();
    }

    /// Reads the destination on the calling (writing) thread's own stream.
    fn wrong(&self, label: &str, iter: usize) -> usize {
        let host = self.d.to_host().unwrap();
        let data = host.dense_data().unwrap();
        let miss = data
            .iter()
            .zip(&self.expected)
            .filter(|(a, b)| a != b)
            .count();
        if miss > 0 {
            println!("F2 {label} iter {iter}: {miss}/{} wrong", data.len());
        }
        usize::from(miss > 0)
    }
}

fn reused_scratch(threaded: bool) -> usize {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    // Thread 1 (this thread) binds the scratch.
    let mut first = F2::new(&runtime, 1.0);
    assert!(runtime.cuda_contract_scratch_bytes().unwrap() > 0);
    // Bound on thread 1's stream after the scratch: thread 2's download of it
    // syncs thread 2's stream past the scratch's bind, once.
    let marker = constant(&runtime, &[&space(1, 1)], 0.0);
    let mut stale = 0;
    if !threaded {
        let mut second = F2::new(&runtime, 2.0);
        for iter in 0..ITERS {
            first.enqueue();
            second.enqueue();
            stale += first.wrong("first", iter).max(second.wrong("second", iter));
        }
        return stale;
    }
    let (go_tx, go_rx) = mpsc::channel::<()>();
    let (out_tx, out_rx) = mpsc::channel::<usize>();
    let second = std::thread::spawn(move || {
        marker.to_host().unwrap();
        // Thread 2's own operands and destination: only the scratch is shared.
        let mut second = F2::new(&runtime, 2.0);
        out_tx.send(0).unwrap();
        let mut iter = 0;
        while go_rx.recv().is_ok() {
            second.enqueue();
            out_tx.send(second.wrong("second", iter)).unwrap();
            iter += 1;
        }
    });
    out_rx.recv().unwrap();
    for iter in 0..ITERS {
        first.enqueue();
        go_tx.send(()).unwrap();
        let wrong_second = out_rx.recv().unwrap();
        stale += first.wrong("first", iter).max(wrong_second);
    }
    drop(go_tx);
    second.join().unwrap();
    stale
}

#[test]
#[ignore = "requires a CUDA device"]
fn reused_scratch_is_not_overwritten_while_another_thread_reads_it() {
    let stale = reused_scratch(true);
    println!("F2 two threads: {stale}/{ITERS} iterations read a wrong output");
    assert_eq!(stale, 0, "stale reused scratch (F2): {stale}/{ITERS}");
}

#[test]
#[ignore = "requires a CUDA device"]
fn reused_scratch_control_on_one_thread_is_correct() {
    let stale = reused_scratch(false);
    println!("F2 one thread: {stale}/{ITERS} iterations read a wrong output");
    assert_eq!(stale, 0);
}
