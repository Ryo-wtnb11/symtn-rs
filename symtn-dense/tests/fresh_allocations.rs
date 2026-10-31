//! A warm fresh batch costs no more allocations than the same warm
//! initialized batch: the fresh entry reuses the initialized route's plans
//! and the executor's staging instead of preparing per call.
//!
//! Scope: one serial engine, so every allocation is on the measured thread.

#![cfg(all(feature = "tenferro", not(feature = "provider-inject")))]

use std::mem::MaybeUninit;

use symtn_dense::{
    strided_batch_runs, CpuEngine, DefaultDenseExecutor, DenseExecutor, DenseFreshSegments,
    DenseGemmBatchJob, DenseRead, DenseScalar, DenseView, DenseViewMut, DenseWrite, MatrixOp,
};

use crate::counting_alloc;

/// A zero fill that allocates nothing, so the measurement sees only the
/// executor.
fn zeroed(piece: &mut [MaybeUninit<f64>]) -> &mut [f64] {
    static ZEROS: [f64; 16] = [0.0; 16];
    piece.write_copy_of_slice(&ZEROS[..piece.len()])
}

fn job(shape: (usize, usize, usize), dst: usize, lhs: usize, rhs: usize) -> DenseGemmBatchJob {
    DenseGemmBatchJob {
        rows: shape.0,
        contracted: shape.1,
        cols: shape.2,
        dst_offset: dst,
        lhs_offset: lhs,
        rhs_offset: rhs,
    }
}

#[test]
fn warm_fresh_op_bearing_batch_allocates_no_more_than_the_initialized_one() {
    let _serial = counting_alloc::serial();
    let engine = &mut CpuEngine::serial();
    // Two heterogeneous jobs: the grouped (or BLAS op-bearing serial) route.
    let jobs = [job((2, 3, 2), 0, 0, 0), job((3, 2, 1), 4, 6, 6)];
    let runs = strided_batch_runs(&jobs);
    let (lhs, rhs) = ([1.0; 12], [2.0; 8]);
    let ops = [MatrixOp::Transpose, MatrixOp::Identity];

    let mut initialized = DefaultDenseExecutor::new();
    let mut output = [0.0; 7];
    let mut initialized_call = |executor: &mut DefaultDenseExecutor, output: &mut [f64; 7]| {
        executor
            .matmul_batch_axpby_with_ops_into(
                engine,
                DenseWrite::F64(DenseViewMut::new(output, &[7], &[1], 0).unwrap()),
                DenseRead::F64(DenseView::new(&lhs, &[12], &[1], 0).unwrap()),
                DenseRead::F64(DenseView::new(&rhs, &[8], &[1], 0).unwrap()),
                &jobs,
                &runs,
                ops[0],
                ops[1],
                DenseScalar::F64(1.0),
                DenseScalar::F64(0.0),
            )
            .unwrap()
    };
    for _ in 0..3 {
        initialized_call(&mut initialized, &mut output);
    }
    let ((), initialized_allocs) =
        counting_alloc::measure(|| initialized_call(&mut initialized, &mut output));

    let engine = &mut CpuEngine::serial();
    let mut fresh = DefaultDenseExecutor::new();
    let mut buffer = [MaybeUninit::<f64>::uninit(); 7];
    let mut reported = 0usize;
    let mut fresh_call = |executor: &mut DefaultDenseExecutor,
                          buffer: &mut [MaybeUninit<f64>; 7],
                          reported: &mut usize| {
        let mut parts = [(0, &mut buffer[..])];
        let mut written = |_: usize, _: &mut [f64]| *reported += 1;
        executor
            .matmul_batch_fresh(
                engine,
                DenseFreshSegments::F64 {
                    segments: &mut parts,
                    zeroed,
                    written: &mut written,
                },
                DenseRead::F64(DenseView::new(&lhs, &[12], &[1], 0).unwrap()),
                DenseRead::F64(DenseView::new(&rhs, &[8], &[1], 0).unwrap()),
                &jobs,
                ops,
                DenseScalar::F64(1.0),
            )
            .unwrap()
    };
    for _ in 0..3 {
        fresh_call(&mut fresh, &mut buffer, &mut reported);
    }
    let ((), fresh_allocs) =
        counting_alloc::measure(|| fresh_call(&mut fresh, &mut buffer, &mut reported));

    assert_eq!(reported, 4 * jobs.len());
    assert!(
        fresh_allocs.calls <= initialized_allocs.calls
            && fresh_allocs.bytes <= initialized_allocs.bytes,
        "fresh {fresh_allocs:?} against initialized {initialized_allocs:?}"
    );
}
