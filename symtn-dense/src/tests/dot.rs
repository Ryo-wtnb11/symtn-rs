use super::*;

/// The BLAS provider keeps its op-bearing run partition; the native provider
/// sends every batch that is not one strided run to the block kernel.
const BLAS: bool = cfg!(feature = "cpu-blas-core");

#[test]
fn op_bearing_batch_reuses_executor_across_shapes_offsets_and_alpha_beta() {
    let engine = &mut CpuEngine::serial();
    // Adjoint gives both rectangular operands noncontiguous matrix strides.
    let alpha = Complex64::new(0.75, -0.5);
    let beta = Complex64::new(-0.25, 0.125);
    let mut executor = DefaultDenseExecutor::new();
    for (rows, contracted, cols, seed) in [(2, 3, 4, 1.0), (3, 2, 1, 9.0)] {
        let (view_offset, job_offset) = (1, 2);
        let lhs_values = (0..contracted * rows)
            .map(|i| Complex64::new(seed + i as f64, 0.25 * i as f64 - 0.5))
            .collect::<Vec<_>>();
        let rhs_values = (0..cols * contracted)
            .map(|i| Complex64::new(seed - 0.5 * i as f64, 0.75 - 0.1 * i as f64))
            .collect::<Vec<_>>();
        let mut lhs = vec![Complex64::new(-99.0, 1.0); view_offset + job_offset];
        lhs.extend_from_slice(&lhs_values);
        let mut rhs = vec![Complex64::new(-98.0, 2.0); view_offset + job_offset];
        rhs.extend_from_slice(&rhs_values);
        let initial = Complex64::new(0.5, -0.25);
        let mut output = vec![initial; view_offset + job_offset + rows * cols + 1];
        let jobs = [DenseGemmBatchJob {
            dst_offset: job_offset,
            lhs_offset: job_offset,
            rhs_offset: job_offset,
            rows,
            contracted,
            cols,
        }];
        let flat_strides = [1];
        let (lhs_shape, rhs_shape, output_shape) = (
            [lhs.len() - view_offset],
            [rhs.len() - view_offset],
            [output.len() - view_offset],
        );
        executor
            .matmul_batch_axpby_with_ops_into(
                engine,
                DenseWrite::C64(
                    DenseViewMut::new(&mut output, &output_shape, &flat_strides, view_offset)
                        .unwrap(),
                ),
                DenseRead::C64(
                    DenseView::new(&lhs, &lhs_shape, &flat_strides, view_offset).unwrap(),
                ),
                DenseRead::C64(
                    DenseView::new(&rhs, &rhs_shape, &flat_strides, view_offset).unwrap(),
                ),
                &jobs,
                &[1],
                MatrixOp::Adjoint,
                MatrixOp::Adjoint,
                DenseScalar::C64(alpha),
                DenseScalar::C64(beta),
            )
            .unwrap();

        for col in 0..cols {
            for row in 0..rows {
                let mut sum = Complex64::new(0.0, 0.0);
                for inner in 0..contracted {
                    let left = lhs_values[inner + contracted * row].conj();
                    let right = rhs_values[col + cols * inner].conj();
                    sum += left * right;
                }
                let index = view_offset + job_offset + row + rows * col;
                assert_c64_close(output[index], alpha * sum + beta * initial, 1.0e-12);
            }
        }
    }
}
#[test]
fn op_bearing_batch_rejects_offset_overflow_before_view_construction() {
    let engine = &mut CpuEngine::serial();
    // What: malformed public batch jobs return a typed offset error instead of
    // wrapping an operand base offset before the transposed view is validated.
    let lhs = vec![1.0, 2.0];
    let rhs = vec![3.0];
    let mut output = vec![0.0];
    let jobs = [DenseGemmBatchJob {
        dst_offset: 0,
        lhs_offset: usize::MAX,
        rhs_offset: 0,
        rows: 1,
        contracted: 1,
        cols: 1,
    }];
    let shape = [1];
    let strides = [1];
    let mut executor = DefaultDenseExecutor::new();

    let error = executor
        .matmul_batch_axpby_with_ops_into(
            engine,
            DenseWrite::F64(DenseViewMut::new(&mut output, &shape, &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&lhs, &shape, &strides, 1).unwrap()),
            DenseRead::F64(DenseView::new(&rhs, &shape, &strides, 0).unwrap()),
            &jobs,
            &[1],
            MatrixOp::Adjoint,
            MatrixOp::Identity,
            DenseScalar::F64(1.0),
            DenseScalar::F64(0.0),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        DenseError::OffsetOverflow { value: usize::MAX }
    ));
}
#[test]
fn op_bearing_uniform_runs_batch_literal_complex_all_matrix_ops() {
    let mut executor = DefaultDenseExecutor::new();
    let ops = [MatrixOp::Identity, MatrixOp::Transpose, MatrixOp::Adjoint];
    for (shape, broadcast_lhs, run_len) in [
        ((2, 3, 4), false, 4),
        ((3, 2, 2), true, 3),
        ((2, 4, 3), false, 2),
    ] {
        for lhs_op in ops {
            for rhs_op in ops {
                if lhs_op != MatrixOp::Identity || rhs_op != MatrixOp::Identity {
                    check_complex_oriented_run(
                        &mut executor,
                        shape,
                        lhs_op,
                        rhs_op,
                        broadcast_lhs,
                        run_len,
                    );
                }
            }
        }
    }
}
#[test]
fn op_bearing_rectangular_run_uses_strided_seam_for_other_float_dtypes() {
    let engine = &mut CpuEngine::serial();
    let jobs = [
        batch_job((2, 3, 2), (1, 1, 2)),
        batch_job((2, 3, 2), (7, 8, 10)),
    ];
    assert_eq!(strided_batch_runs(&jobs), [2]);
    let lhs_f32 = (0..14).map(|i| 0.5 + i as f32).collect::<Vec<_>>();
    let rhs_f32 = (0..16).map(|i| 1.25 - 0.125 * i as f32).collect::<Vec<_>>();
    let mut output_f32 = vec![3.0_f32; 12];
    let strides = [1];
    let mut executor = DefaultDenseExecutor::new();
    executor.reset_seam_dispatches();
    executor
        .matmul_batch_axpby_with_ops_into(
            engine,
            DenseWrite::F32(DenseViewMut::new(&mut output_f32, &[12], &strides, 0).unwrap()),
            DenseRead::F32(DenseView::new(&lhs_f32, &[14], &strides, 0).unwrap()),
            DenseRead::F32(DenseView::new(&rhs_f32, &[16], &strides, 0).unwrap()),
            &jobs,
            &[2],
            MatrixOp::Transpose,
            MatrixOp::Identity,
            DenseScalar::F32(1.25),
            DenseScalar::F32(-0.5),
        )
        .unwrap();
    assert_eq!(executor.seam_dispatches(), 1);
    for job in &jobs {
        for col in 0..2 {
            for row in 0..2 {
                let sum = (0..3)
                    .map(|inner| {
                        lhs_f32[job.lhs_offset + inner + 3 * row]
                            * rhs_f32[job.rhs_offset + inner + 3 * col]
                    })
                    .sum::<f32>();
                assert_f32_close(
                    output_f32[job.dst_offset + row + 2 * col],
                    1.25 * sum - 1.5,
                    1.0e-4,
                );
            }
        }
    }

    let lhs_f64 = lhs_f32
        .iter()
        .map(|&value| value as f64)
        .collect::<Vec<_>>();
    let rhs_f64 = rhs_f32
        .iter()
        .map(|&value| value as f64)
        .collect::<Vec<_>>();
    let mut output_f64 = vec![3.0_f64; 12];
    executor.reset_seam_dispatches();
    executor
        .matmul_batch_axpby_with_ops_into(
            engine,
            DenseWrite::F64(DenseViewMut::new(&mut output_f64, &[12], &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&lhs_f64, &[14], &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&rhs_f64, &[16], &strides, 0).unwrap()),
            &jobs,
            &[2],
            MatrixOp::Transpose,
            MatrixOp::Identity,
            DenseScalar::F64(1.25),
            DenseScalar::F64(-0.5),
        )
        .unwrap();
    assert_eq!(executor.seam_dispatches(), 1);
    for job in &jobs {
        for col in 0..2 {
            for row in 0..2 {
                let sum = (0..3)
                    .map(|inner| {
                        lhs_f64[job.lhs_offset + inner + 3 * row]
                            * rhs_f64[job.rhs_offset + inner + 3 * col]
                    })
                    .sum::<f64>();
                assert_f64_close(
                    output_f64[job.dst_offset + row + 2 * col],
                    1.25 * sum - 1.5,
                    1.0e-12,
                );
            }
        }
    }

    let lhs_c32 = lhs_f32
        .iter()
        .enumerate()
        .map(|(i, &value)| Complex32::new(value, 0.25 * i as f32 - 0.5))
        .collect::<Vec<_>>();
    let rhs_c32 = rhs_f32
        .iter()
        .enumerate()
        .map(|(i, &value)| Complex32::new(value, 0.75 - 0.1 * i as f32))
        .collect::<Vec<_>>();
    let initial = Complex32::new(3.0, -2.0);
    let alpha = Complex32::new(0.75, -0.25);
    let beta = Complex32::new(-0.5, 0.125);
    let mut output_c32 = vec![initial; 12];
    executor.reset_seam_dispatches();
    executor
        .matmul_batch_axpby_with_ops_into(
            engine,
            DenseWrite::C32(DenseViewMut::new(&mut output_c32, &[12], &strides, 0).unwrap()),
            DenseRead::C32(DenseView::new(&lhs_c32, &[14], &strides, 0).unwrap()),
            DenseRead::C32(DenseView::new(&rhs_c32, &[16], &strides, 0).unwrap()),
            &jobs,
            &[2],
            MatrixOp::Adjoint,
            MatrixOp::Adjoint,
            DenseScalar::C32(alpha),
            DenseScalar::C32(beta),
        )
        .unwrap();
    assert_eq!(executor.seam_dispatches(), 1);
    for job in &jobs {
        for col in 0..2 {
            for row in 0..2 {
                let sum = (0..3)
                    .map(|inner| {
                        lhs_c32[job.lhs_offset + inner + 3 * row].conj()
                            * rhs_c32[job.rhs_offset + col + 2 * inner].conj()
                    })
                    .sum::<Complex32>();
                assert_c32_close(
                    output_c32[job.dst_offset + row + 2 * col],
                    alpha * sum + beta * initial,
                    1.0e-3,
                );
            }
        }
    }
}
#[test]
fn op_bearing_mixed_runs_and_safe_subpartitions_use_absolute_jobs() {
    let engine = &mut CpuEngine::serial();
    let jobs = vec![
        batch_job((1, 1, 1), (0, 0, 0)),
        batch_job((1, 1, 1), (1, 1, 1)),
        batch_job((1, 2, 1), (2, 2, 2)),
        batch_job((1, 1, 1), (3, 4, 4)),
        batch_job((1, 1, 1), (4, 5, 5)),
        batch_job((1, 1, 1), (5, 6, 6)),
    ];
    let runs = strided_batch_runs(&jobs);
    assert_eq!(runs, [2, 1, 3]);
    let lhs = [2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0];
    let rhs = [19.0, 23.0, 29.0, 31.0, 37.0, 41.0, 43.0];
    let mut output = [1.0; 8];
    let shape = [output.len()];
    let strides = [1];
    let mut executor = DefaultDenseExecutor::new();
    executor.reset_seam_dispatches();
    executor
        .matmul_batch_axpby_with_ops_into(
            engine,
            DenseWrite::F64(DenseViewMut::new(&mut output, &shape, &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&lhs, &[lhs.len()], &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&rhs, &[rhs.len()], &strides, 0).unwrap()),
            &jobs,
            &runs,
            MatrixOp::Transpose,
            MatrixOp::Identity,
            DenseScalar::F64(2.0),
            DenseScalar::F64(-0.5),
        )
        .unwrap();
    // Native: one block-kernel submission; BLAS: run partition.
    assert_eq!(executor.seam_dispatches(), if BLAS { 3 } else { 1 });
    let sums = [
        lhs[0] * rhs[0],
        lhs[1] * rhs[1],
        lhs[2] * rhs[2] + lhs[3] * rhs[3],
        lhs[4] * rhs[4],
        lhs[5] * rhs[5],
        lhs[6] * rhs[6],
    ];
    for (actual, sum) in output[..6].iter().zip(sums) {
        assert_eq!(*actual, 2.0 * sum - 0.5);
    }
    assert_eq!(&output[6..], &[1.0, 1.0]);

    let mut singleton_output = [1.0; 8];
    executor.reset_seam_dispatches();
    executor
        .matmul_batch_axpby_with_ops_into(
            engine,
            DenseWrite::F64(DenseViewMut::new(&mut singleton_output, &shape, &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&lhs, &[lhs.len()], &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&rhs, &[rhs.len()], &strides, 0).unwrap()),
            &jobs,
            &[1; 6],
            MatrixOp::Transpose,
            MatrixOp::Identity,
            DenseScalar::F64(2.0),
            DenseScalar::F64(-0.5),
        )
        .unwrap();
    assert_eq!(
        executor.seam_dispatches(),
        if BLAS { jobs.len() } else { 1 }
    );
    for (actual, sum) in singleton_output[..6].iter().zip(sums) {
        assert_eq!(*actual, 2.0 * sum - 0.5);
    }
    assert_eq!(&singleton_output[6..], &[1.0, 1.0]);

    let affine = (0..4)
        .map(|offset| batch_job((1, 1, 1), (offset, offset, offset)))
        .collect::<Vec<_>>();
    let data = [2.0, 3.0, 5.0, 7.0];
    let mut split_output = [0.0; 4];
    executor.reset_seam_dispatches();
    executor
        .matmul_batch_axpby_with_ops_into(
            engine,
            DenseWrite::F64(DenseViewMut::new(&mut split_output, &[4], &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&data, &[4], &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&data, &[4], &strides, 0).unwrap()),
            &affine,
            &[2, 2],
            MatrixOp::Transpose,
            MatrixOp::Identity,
            DenseScalar::F64(1.0),
            DenseScalar::F64(0.0),
        )
        .unwrap();
    assert_eq!(executor.seam_dispatches(), if BLAS { 2 } else { 1 });
    assert_eq!(split_output, [4.0, 9.0, 25.0, 49.0]);
}
#[test]
fn op_bearing_malformed_runs_fall_back_without_skipping_late_errors() {
    let engine = &mut CpuEngine::serial();
    let strides = [1];
    let affine = (0..4)
        .map(|offset| batch_job((1, 1, 1), (offset, offset, offset)))
        .collect::<Vec<_>>();
    let data = [2.0, 3.0, 5.0, 7.0];
    let mut executor = DefaultDenseExecutor::new();
    for malformed_runs in [&[3][..], &[usize::MAX][..], &[4, 0][..]] {
        let mut output = [-1.0; 4];
        executor.reset_seam_dispatches();
        executor
            .matmul_batch_axpby_with_ops_into(
                engine,
                DenseWrite::F64(DenseViewMut::new(&mut output, &[4], &strides, 0).unwrap()),
                DenseRead::F64(DenseView::new(&data, &[4], &strides, 0).unwrap()),
                DenseRead::F64(DenseView::new(&data, &[4], &strides, 0).unwrap()),
                &affine,
                malformed_runs,
                MatrixOp::Transpose,
                MatrixOp::Identity,
                DenseScalar::F64(1.0),
                DenseScalar::F64(0.0),
            )
            .unwrap();
        assert_eq!(executor.seam_dispatches(), if BLAS { 4 } else { 1 });
        assert_eq!(output, [4.0, 9.0, 25.0, 49.0]);
    }

    let mut nonaffine = affine.clone();
    nonaffine[2].lhs_offset = 3;
    let mut nonaffine_output = [-1.0; 4];
    executor.reset_seam_dispatches();
    executor
        .matmul_batch_axpby_with_ops_into(
            engine,
            DenseWrite::F64(DenseViewMut::new(&mut nonaffine_output, &[4], &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&data, &[4], &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&data, &[4], &strides, 0).unwrap()),
            &nonaffine,
            &[4],
            MatrixOp::Transpose,
            MatrixOp::Identity,
            DenseScalar::F64(1.0),
            DenseScalar::F64(0.0),
        )
        .unwrap();
    assert_eq!(executor.seam_dispatches(), if BLAS { 4 } else { 1 });
    assert_eq!(nonaffine_output, [4.0, 9.0, 35.0, 49.0]);

    let lhs = [2.0, 3.0, 5.0];
    let rhs = [7.0, 11.0, 13.0, 17.0];
    for runs in [&[4][..], &[1, 1, 0, 2][..]] {
        let mut output = [-1.0; 4];
        executor.reset_seam_dispatches();
        let error = executor
            .matmul_batch_axpby_with_ops_into(
                engine,
                DenseWrite::F64(DenseViewMut::new(&mut output, &[4], &strides, 0).unwrap()),
                DenseRead::F64(DenseView::new(&lhs, &[3], &strides, 0).unwrap()),
                DenseRead::F64(DenseView::new(&rhs, &[4], &strides, 0).unwrap()),
                &affine,
                runs,
                MatrixOp::Transpose,
                MatrixOp::Identity,
                DenseScalar::F64(1.0),
                DenseScalar::F64(0.0),
            )
            .unwrap_err();
        if BLAS {
            // The serial remainder writes every job below the failing one.
            assert_eq!(error, DenseError::OutOfBounds);
            assert_eq!(executor.seam_dispatches(), 3);
            assert_eq!(output, [14.0, 33.0, 65.0, -1.0]);
        } else {
            // The block kernel bounds every job before the first write.
            assert!(
                matches!(&error, DenseError::Backend { op: "grouped_gemm", message, .. } if message.contains("job 3 lhs")),
                "{error:?}"
            );
            assert_eq!(executor.seam_dispatches(), 1);
            assert_eq!(output, [-1.0; 4]);
        }
    }
}
#[test]
fn op_bearing_empty_zero_and_backend_failure_keep_serial_boundaries() {
    let engine = &mut CpuEngine::serial();
    let strides = [1];
    let mut executor = DefaultDenseExecutor::new();
    let mut output = [3.0];
    executor.reset_seam_dispatches();
    executor
        .matmul_batch_axpby_with_ops_into(
            engine,
            DenseWrite::F64(DenseViewMut::new(&mut output, &[1], &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&[1.0], &[1], &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&[2.0], &[1], &strides, 0).unwrap()),
            &[],
            &[],
            MatrixOp::Transpose,
            MatrixOp::Identity,
            DenseScalar::F64(1.0),
            DenseScalar::F64(0.0),
        )
        .unwrap();
    assert_eq!(executor.seam_dispatches(), 0);
    assert_eq!(output, [3.0]);

    let zero_jobs = [
        batch_job((0, 1, 1), (0, 0, 0)),
        batch_job((0, 1, 1), (1, 0, 1)),
    ];
    let rhs = [2.0, 3.0];
    executor.reset_seam_dispatches();
    executor
        .matmul_batch_axpby_with_ops_into(
            engine,
            DenseWrite::F64(DenseViewMut::new(&mut output, &[1], &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&[], &[0], &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&rhs, &[2], &strides, 0).unwrap()),
            &zero_jobs,
            &[2],
            MatrixOp::Transpose,
            MatrixOp::Identity,
            DenseScalar::F64(1.0),
            DenseScalar::F64(0.0),
        )
        .unwrap();
    assert_eq!(executor.seam_dispatches(), if BLAS { 2 } else { 1 });
    assert_eq!(output, [3.0]);

    let jobs = [
        batch_job((1, 1, 1), (0, 0, 0)),
        batch_job((1, 1, 1), (1, 1, 1)),
    ];
    let values = [2.0, 3.0];
    let mut failed_output = [5.0, 7.0];
    executor.reset_seam_dispatches();
    let error = executor
        .matmul_batch_axpby_with_ops_into(
            engine,
            DenseWrite::F64(DenseViewMut::new(&mut failed_output, &[2], &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&values, &[2], &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&values, &[2], &strides, 0).unwrap()),
            &jobs,
            &[2],
            MatrixOp::Transpose,
            MatrixOp::Identity,
            DenseScalar::C64(Complex64::new(1.0, 0.0)),
            DenseScalar::F64(0.0),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        DenseError::Backend {
            op: "matmul_batch_axpby_with_ops_into",
            ..
        }
    ));
    assert_eq!(executor.seam_dispatches(), 1);
    assert_eq!(failed_output, [5.0, 7.0]);
}
#[test]
fn identity_strided_run_keeps_lhs_bounds_before_rhs_offset_error() {
    let engine = &mut CpuEngine::serial();
    let jobs = (0..4)
        .map(|batch| DenseGemmBatchJob {
            dst_offset: 2 * batch,
            lhs_offset: 0,
            rhs_offset: usize::MAX,
            rows: 2,
            contracted: 1,
            cols: 1,
        })
        .collect::<Vec<_>>();
    assert_eq!(strided_batch_runs(&jobs), [4]);
    let lhs = [1.0];
    let rhs = [2.0, 3.0];
    let mut output = [5.0; 8];
    let strides = [1];
    let mut executor = DefaultDenseExecutor::new();
    executor.reset_seam_dispatches();
    let error = executor
        .matmul_batch_axpby_into(
            engine,
            DenseWrite::F64(DenseViewMut::new(&mut output, &[8], &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&lhs, &[1], &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&rhs, &[1], &strides, 1).unwrap()),
            &jobs,
            &[4],
            DenseScalar::F64(1.0),
            DenseScalar::F64(0.0),
        )
        .unwrap_err();
    assert_eq!(error, DenseError::OutOfBounds);
    assert_eq!(executor.seam_dispatches(), 0);
    assert_eq!(output, [5.0; 8]);
}
// Regression guard for the conjugated-contraction fast path: a conj flag on
// `dot_general_into` must fold conjugation into the kernel and produce exactly
// what contracting an elementwise-conjugated operand would — and it must
// actually change the result (so the flag can't be silently dropped back to a
// no-op or a bypassed scalar loop).
#[test]
fn dot_general_conjugation_flag_matches_materialized_conjugate() {
    let engine = &mut CpuEngine::serial();
    let c = |re: f64, im: f64| Complex64::new(re, im);
    let shape = [2usize, 2];
    let strides = [1usize, 2]; // column-major
    let lhs = vec![c(1.0, 1.0), c(3.0, 2.0), c(2.0, -1.0), c(4.0, -3.0)];
    let rhs = vec![c(5.0, -2.0), c(7.0, -4.0), c(6.0, 1.0), c(8.0, 2.0)];

    let mut run = |lhs_data: &[Complex64], lhs_conj: bool, rhs_conj: bool| -> Vec<Complex64> {
        let mut out = vec![c(0.0, 0.0); 4];
        let mut executor = DefaultDenseExecutor::new();
        executor
            .dot_general_into(
                engine,
                DenseWrite::C64(DenseViewMut::new(&mut out, &shape, &strides, 0).unwrap()),
                DenseRead::C64(DenseView::new(lhs_data, &shape, &strides, 0).unwrap()),
                DenseRead::C64(DenseView::new(&rhs, &shape, &strides, 0).unwrap()),
                &DenseDotConfig::matmul().with_conjugation(lhs_conj, rhs_conj),
            )
            .unwrap();
        out
    };

    let via_flag = run(&lhs, true, false);
    let lhs_conjugated: Vec<Complex64> = lhs.iter().map(|z| z.conj()).collect();
    let via_materialized = run(&lhs_conjugated, false, false);
    for (actual, expected) in via_flag.iter().zip(&via_materialized) {
        assert_c64_close(*actual, *expected, 1.0e-12);
    }

    let plain = run(&lhs, false, false);
    assert!(
        via_flag
            .iter()
            .zip(&plain)
            .any(|(a, b)| (a - b).norm() > 1.0e-9),
        "conjugation flag had no effect on the result"
    );
}
