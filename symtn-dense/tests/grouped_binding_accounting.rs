//! `DenseGemmBatchBinding::retained_bytes` against the live allocations a
//! bound grouped batch keeps: the charge must cover everything the binding
//! owns (it may overcharge), the warm call allocates nothing, and dropping
//! the binding releases exactly what the cold call kept.
//!
//! Scope: the native grouped route (no BLAS provider), one dense thread, so
//! every allocation is on the measured thread. The executor is warmed with
//! the same batch unbound first, so engine scratch is not in the window.

#![cfg(not(feature = "cpu-blas-core"))]

use num_complex::Complex64;
use symtn_dense::{
    DefaultDenseExecutor, DenseExecutor, DenseGemmBatchBinding, DenseGemmBatchJob, DenseRead,
    DenseScalar, DenseView, DenseViewMut, DenseWrite,
};

use crate::counting_alloc;

fn job(dst: usize, lhs: usize, rhs: usize, m: usize, k: usize, n: usize) -> DenseGemmBatchJob {
    DenseGemmBatchJob {
        dst_offset: dst,
        lhs_offset: lhs,
        rhs_offset: rhs,
        rows: m,
        contracted: k,
        cols: n,
    }
}

/// Lengths `[out, lhs, rhs]` the jobs reach.
fn lens(jobs: &[DenseGemmBatchJob]) -> [usize; 3] {
    jobs.iter().fold([0; 3], |[o, l, r], j| {
        [
            o.max(j.dst_offset + j.rows * j.cols),
            l.max(j.lhs_offset + j.rows * j.contracted),
            r.max(j.rhs_offset + j.contracted * j.cols),
        ]
    })
}

fn check<T: Copy>(
    what: &str,
    [half, quarter, nought]: [T; 3],
    jobs: &[DenseGemmBatchJob],
    write: fn(DenseViewMut<'_, T>) -> DenseWrite<'_>,
    read: fn(DenseView<'_, T>) -> DenseRead<'_>,
    one: DenseScalar,
    zero: DenseScalar,
) {
    let [out_len, lhs_len, rhs_len] = lens(jobs);
    let lhs = vec![half; lhs_len];
    let rhs = vec![quarter; rhs_len];
    let mut out = vec![nought; out_len];
    let runs = vec![1; jobs.len()];
    let (o, l, r) = ([out_len], [lhs_len], [rhs_len]);
    let mut executor = DefaultDenseExecutor::new();
    let mut engine = symtn_dense::CpuEngine::serial();
    let mut call = |executor: &mut DefaultDenseExecutor,
                    binding: Option<&mut DenseGemmBatchBinding>| {
        let output = write(DenseViewMut::new(&mut out, &o, &[1], 0).unwrap());
        let lhs = read(DenseView::new(&lhs, &l, &[1], 0).unwrap());
        let rhs = read(DenseView::new(&rhs, &r, &[1], 0).unwrap());
        match binding {
            Some(binding) => executor.matmul_batch_axpby_into_bound(
                &mut engine,
                output,
                lhs,
                rhs,
                jobs,
                &runs,
                one,
                zero,
                binding,
            ),
            None => executor
                .matmul_batch_axpby_into(&mut engine, output, lhs, rhs, jobs, &runs, one, zero),
        }
        .unwrap()
    };
    call(&mut executor, None);
    call(&mut executor, None);
    let mut binding = DenseGemmBatchBinding::default();
    let ((), cold) = counting_alloc::measure(|| call(&mut executor, Some(&mut binding)));
    let ((), warm) = counting_alloc::measure(|| call(&mut executor, Some(&mut binding)));
    let charge = binding.retained_bytes();
    let ((), dropped) = counting_alloc::measure(|| drop(binding));
    eprintln!(
        "{what}: jobs={} cold calls={} live={} B, charge={charge} B, warm calls={}, freed on drop={} B",
        jobs.len(),
        cold.calls,
        cold.live_bytes,
        warm.calls,
        dropped.freed_bytes
    );
    assert!(cold.live_bytes > 0, "{what}: the binding keeps the plan");
    assert!(
        cold.live_bytes as usize <= charge,
        "{what}: live {} B exceeds the charge {charge} B",
        cold.live_bytes
    );
    assert_eq!(warm.calls, 0, "{what}: a warm bound call allocates nothing");
    assert_eq!(
        dropped.freed_bytes as i64, cold.live_bytes,
        "{what}: dropping the binding releases what it kept"
    );
}

#[test]
fn binding_charge_covers_its_live_allocations() {
    let _serial = counting_alloc::serial();
    // Tiny f64: compact Faer children without child heap.
    check::<f64>(
        "tiny f64",
        [0.5, -0.25, 0.0],
        &[
            job(10, 0, 0, 2, 2, 2),
            job(0, 4, 4, 2, 2, 2),
            job(4, 8, 8, 3, 1, 2),
        ],
        |v: DenseViewMut<'_, f64>| DenseWrite::F64(v),
        |v: DenseView<'_, f64>| DenseRead::F64(v),
        DenseScalar::F64(1.0),
        DenseScalar::F64(0.0),
    );
    // c64 above tcontract's default Faer limit (m n k > 2^17): Packed
    // children with boxed scatters and axis vectors.
    let a = 64 * 64;
    check::<Complex64>(
        "packed c64",
        [0.5, -0.25, 0.0].map(|re| Complex64::new(re, 0.5)),
        &[job(72 * 56, 0, 0, 64, 64, 64), job(0, a, a, 72, 48, 56)],
        |v: DenseViewMut<'_, Complex64>| DenseWrite::C64(v),
        |v: DenseView<'_, Complex64>| DenseRead::C64(v),
        DenseScalar::C64(Complex64::new(1.0, 0.0)),
        DenseScalar::C64(Complex64::new(0.0, 0.0)),
    );
}
