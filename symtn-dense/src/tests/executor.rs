use super::*;

#[cfg(feature = "provider-inject")]
#[test]
fn provider_inject_rejects_linalg_before_backend_work() {
    let mut executor = DefaultDenseExecutor::new();
    let data = [1.0, 0.0, 0.0, 1.0];
    let error = executor
        .svd(DenseRead::F64(
            DenseView::new(&data, &[2, 2], &[1, 2], 0).unwrap(),
        ))
        .unwrap_err();
    let DenseError::Unsupported { op, message } = error else {
        panic!("provider-inject SVD must be unsupported")
    };
    assert_eq!(op, "svd");
    assert_eq!(
        message,
        "provider-inject does not route factorizations or solves; they are unsupported in this configuration"
    );
}
#[cfg(feature = "provider-inject")]
#[test]
fn provider_inject_rejects_values_only_before_backend_work() {
    let mut executor = DefaultDenseExecutor::new();
    let data = [1.0, 0.0, 0.0, 1.0];

    let error = executor
        .svd_vals(DenseRead::F64(
            DenseView::new(&data, &[2, 2], &[1, 2], 0).unwrap(),
        ))
        .unwrap_err();
    let DenseError::Unsupported { op, message } = error else {
        panic!("provider-inject SVD values must be unsupported")
    };
    assert_eq!(op, "svd_vals");
    assert_eq!(
        message,
        "provider-inject does not route factorizations or solves; they are unsupported in this configuration"
    );

    let error = executor
        .eigh_vals(DenseRead::F64(
            DenseView::new(&data, &[2, 2], &[1, 2], 0).unwrap(),
        ))
        .unwrap_err();
    let DenseError::Unsupported { op, message } = error else {
        panic!("provider-inject EIGH values must be unsupported")
    };
    assert_eq!(op, "eigh_vals");
    assert_eq!(
        message,
        "provider-inject does not route factorizations or solves; they are unsupported in this configuration"
    );
}
#[cfg(feature = "tenferro")]
#[test]
fn default_executor_matmul_into_matches_tensorkit_recoupling_view_for_all_gemm_dtypes() {
    let lhs_shape = [2, 3];
    let lhs_strides = [1, 2];
    let rhs_shape = [3, 2];
    let rhs_strides = [1, 3];
    let out_shape = [2, 2];
    let out_strides = [1, 4];
    let out_offset = 1;

    let mut executor = DefaultDenseExecutor::new();

    let lhs_f32 = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let u_f32 = vec![10.0_f32, 100.0, 1000.0, 20.0, 200.0, 2000.0];
    let mut out_f32 = vec![-1.0_f32; 8];
    executor
        .matmul_into(
            DenseWrite::F32(
                DenseViewMut::new(&mut out_f32, &out_shape, &out_strides, out_offset).unwrap(),
            ),
            DenseRead::F32(DenseView::new(&lhs_f32, &lhs_shape, &lhs_strides, 0).unwrap()),
            DenseRead::F32(DenseView::new(&u_f32, &rhs_shape, &rhs_strides, 0).unwrap()),
        )
        .unwrap();
    assert_eq!(
        out_f32,
        vec![-1.0, 5310.0, 6420.0, -1.0, -1.0, 10620.0, 12840.0, -1.0]
    );

    let lhs_f64 = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let u_f64 = vec![10.0_f64, 100.0, 1000.0, 20.0, 200.0, 2000.0];
    let mut out_f64 = vec![-1.0_f64; 8];
    executor
        .matmul_into(
            DenseWrite::F64(
                DenseViewMut::new(&mut out_f64, &out_shape, &out_strides, out_offset).unwrap(),
            ),
            DenseRead::F64(DenseView::new(&lhs_f64, &lhs_shape, &lhs_strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&u_f64, &rhs_shape, &rhs_strides, 0).unwrap()),
        )
        .unwrap();
    assert_eq!(
        out_f64,
        vec![-1.0, 5310.0, 6420.0, -1.0, -1.0, 10620.0, 12840.0, -1.0]
    );

    let lhs_c32 = lhs_f32
        .iter()
        .map(|&value| Complex32::new(value, 0.0))
        .collect::<Vec<_>>();
    let u_c32 = u_f32
        .iter()
        .map(|&value| Complex32::new(value, 0.0))
        .collect::<Vec<_>>();
    let mut out_c32 = vec![Complex32::new(-1.0, -2.0); 8];
    executor
        .matmul_into(
            DenseWrite::C32(
                DenseViewMut::new(&mut out_c32, &out_shape, &out_strides, out_offset).unwrap(),
            ),
            DenseRead::C32(DenseView::new(&lhs_c32, &lhs_shape, &lhs_strides, 0).unwrap()),
            DenseRead::C32(DenseView::new(&u_c32, &rhs_shape, &rhs_strides, 0).unwrap()),
        )
        .unwrap();
    assert_eq!(
        out_c32,
        vec![
            Complex32::new(-1.0, -2.0),
            Complex32::new(5310.0, 0.0),
            Complex32::new(6420.0, 0.0),
            Complex32::new(-1.0, -2.0),
            Complex32::new(-1.0, -2.0),
            Complex32::new(10620.0, 0.0),
            Complex32::new(12840.0, 0.0),
            Complex32::new(-1.0, -2.0),
        ]
    );

    let lhs_c64 = lhs_f64
        .iter()
        .map(|&value| Complex64::new(value, 0.0))
        .collect::<Vec<_>>();
    let u_c64 = u_f64
        .iter()
        .map(|&value| Complex64::new(value, 0.0))
        .collect::<Vec<_>>();
    let mut out_c64 = vec![Complex64::new(-1.0, -2.0); 8];
    executor
        .matmul_into(
            DenseWrite::C64(
                DenseViewMut::new(&mut out_c64, &out_shape, &out_strides, out_offset).unwrap(),
            ),
            DenseRead::C64(DenseView::new(&lhs_c64, &lhs_shape, &lhs_strides, 0).unwrap()),
            DenseRead::C64(DenseView::new(&u_c64, &rhs_shape, &rhs_strides, 0).unwrap()),
        )
        .unwrap();
    assert_eq!(
        out_c64,
        vec![
            Complex64::new(-1.0, -2.0),
            Complex64::new(5310.0, 0.0),
            Complex64::new(6420.0, 0.0),
            Complex64::new(-1.0, -2.0),
            Complex64::new(-1.0, -2.0),
            Complex64::new(10620.0, 0.0),
            Complex64::new(12840.0, 0.0),
            Complex64::new(-1.0, -2.0),
        ]
    );
}
#[cfg(feature = "tenferro")]
#[test]
fn default_executor_fuses_same_shape_strided_batch_jobs_for_all_gemm_dtypes() {
    // Four same-shape constant-stride jobs form one affine run, so the batch
    // routes through the strided-batch seam as a single dispatch rather than
    // one call per job.
    let mut lhs = Vec::new();
    let mut rhs = Vec::new();
    for block in 0..4 {
        let base = block as f64;
        lhs.extend_from_slice(&[1.0 + base, 2.0 + base, 3.0 + base, 4.0 + base]);
        rhs.extend_from_slice(&[5.0 + base, 6.0 + base, 7.0 + base, 8.0 + base]);
    }
    let mut output = vec![-99.0; 4 * 4];
    let jobs = [0usize, 1, 2, 3]
        .into_iter()
        .map(|block| DenseGemmBatchJob {
            dst_offset: block * 4,
            lhs_offset: block * 4,
            rhs_offset: block * 4,
            rows: 2,
            contracted: 2,
            cols: 2,
        })
        .collect::<Vec<_>>();
    let runs = strided_batch_runs(&jobs);
    assert_eq!(runs, vec![4]);
    let flat_shape = [4 * 4];
    let flat_strides = [1usize];

    let mut executor = DefaultDenseExecutor::new();
    executor.reset_seam_dispatches();
    executor
        .matmul_batch_axpby_into(
            DenseWrite::F64(DenseViewMut::new(&mut output, &flat_shape, &flat_strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&lhs, &flat_shape, &flat_strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&rhs, &flat_shape, &flat_strides, 0).unwrap()),
            &jobs,
            &runs,
            DenseScalar::F64(1.0),
            DenseScalar::F64(0.0),
        )
        .unwrap();

    assert_eq!(
        executor.staged_grouped_jobs(),
        0,
        "valid run must stay strided"
    );
    assert_eq!(
        executor.seam_dispatches(),
        1,
        "same-shape strided batch made {} seam dispatches for {} jobs",
        executor.seam_dispatches(),
        jobs.len()
    );
    assert!(
        executor.seam_dispatches() < jobs.len(),
        "batched GEMM seam dispatch count must not scale with same-shape job count"
    );
    for block in 0..4 {
        let start = block * 4;
        let expected = matmul_f64(&lhs[start..start + 4], &rhs[start..start + 4], 2, 2, 2);
        for (actual, expected) in output[start..start + 4].iter().zip(expected) {
            assert_f64_close(*actual, expected, 1.0e-12);
        }
    }

    let lhs_f32 = lhs.iter().map(|&value| value as f32).collect::<Vec<_>>();
    let rhs_f32 = rhs.iter().map(|&value| value as f32).collect::<Vec<_>>();
    let mut output_f32 = vec![-99.0_f32; 4 * 4];
    let mut executor = DefaultDenseExecutor::new();
    executor
        .matmul_batch_axpby_into(
            DenseWrite::F32(
                DenseViewMut::new(&mut output_f32, &flat_shape, &flat_strides, 0).unwrap(),
            ),
            DenseRead::F32(DenseView::new(&lhs_f32, &flat_shape, &flat_strides, 0).unwrap()),
            DenseRead::F32(DenseView::new(&rhs_f32, &flat_shape, &flat_strides, 0).unwrap()),
            &jobs,
            &runs,
            DenseScalar::F32(1.0),
            DenseScalar::F32(0.0),
        )
        .unwrap();
    assert_eq!(executor.seam_dispatches(), 1);
    for block in 0..4 {
        let start = block * 4;
        let expected = matmul_f32(
            &lhs_f32[start..start + 4],
            &rhs_f32[start..start + 4],
            2,
            2,
            2,
        );
        for (actual, expected) in output_f32[start..start + 4].iter().zip(expected) {
            assert_f32_close(*actual, expected, 1.0e-4);
        }
    }

    let lhs_c32 = lhs_f32
        .iter()
        .map(|&value| Complex32::new(value, 0.25 * value))
        .collect::<Vec<_>>();
    let rhs_c32 = rhs_f32
        .iter()
        .map(|&value| Complex32::new(value, -0.125 * value))
        .collect::<Vec<_>>();
    let mut output_c32 = vec![Complex32::new(-99.0, -99.0); 4 * 4];
    let mut executor = DefaultDenseExecutor::new();
    executor
        .matmul_batch_axpby_into(
            DenseWrite::C32(
                DenseViewMut::new(&mut output_c32, &flat_shape, &flat_strides, 0).unwrap(),
            ),
            DenseRead::C32(DenseView::new(&lhs_c32, &flat_shape, &flat_strides, 0).unwrap()),
            DenseRead::C32(DenseView::new(&rhs_c32, &flat_shape, &flat_strides, 0).unwrap()),
            &jobs,
            &runs,
            DenseScalar::C32(Complex32::new(1.0, 0.0)),
            DenseScalar::C32(Complex32::new(0.0, 0.0)),
        )
        .unwrap();
    assert_eq!(executor.seam_dispatches(), 1);
    for block in 0..4 {
        let start = block * 4;
        let expected = matmul_c32(
            &lhs_c32[start..start + 4],
            &rhs_c32[start..start + 4],
            2,
            2,
            2,
        );
        for (actual, expected) in output_c32[start..start + 4].iter().zip(expected) {
            assert_c32_close(*actual, expected, 1.0e-3);
        }
    }

    let lhs_c64 = lhs
        .iter()
        .map(|&value| Complex64::new(value, 0.25 * value))
        .collect::<Vec<_>>();
    let rhs_c64 = rhs
        .iter()
        .map(|&value| Complex64::new(value, -0.125 * value))
        .collect::<Vec<_>>();
    let mut output_c64 = vec![Complex64::new(-99.0, -99.0); 4 * 4];
    let mut executor = DefaultDenseExecutor::new();
    executor
        .matmul_batch_axpby_into(
            DenseWrite::C64(
                DenseViewMut::new(&mut output_c64, &flat_shape, &flat_strides, 0).unwrap(),
            ),
            DenseRead::C64(DenseView::new(&lhs_c64, &flat_shape, &flat_strides, 0).unwrap()),
            DenseRead::C64(DenseView::new(&rhs_c64, &flat_shape, &flat_strides, 0).unwrap()),
            &jobs,
            &runs,
            DenseScalar::C64(Complex64::new(1.0, 0.0)),
            DenseScalar::C64(Complex64::new(0.0, 0.0)),
        )
        .unwrap();
    assert_eq!(executor.seam_dispatches(), 1);
    for block in 0..4 {
        let start = block * 4;
        let expected = matmul_c64(
            &lhs_c64[start..start + 4],
            &rhs_c64[start..start + 4],
            2,
            2,
            2,
        );
        for (actual, expected) in output_c64[start..start + 4].iter().zip(expected) {
            assert_c64_close(*actual, expected, 1.0e-12);
        }
    }
}
#[cfg(feature = "tenferro")]
#[test]
fn default_executor_bundles_short_runs_into_one_seam_dispatch() {
    // Structural guard for issue #103: a fragmented batch (three runs — two
    // length-2 same-shape runs plus a singleton) is not one affine run, so it
    // must dispatch ONE grouped seam call, not one per run. Seam-call count
    // stays flat as a batch fragments into more runs.
    //
    // Layout: shape A = 2x2x2 (blocks at storage offsets 0,4), shape B = 1x3x1
    // (blocks at 8,10), shape C = 2x1x2 singleton (at 12). All dst ranges
    // disjoint. lhs/rhs share the same flat buffer regions via the offsets.
    let jobs = vec![
        DenseGemmBatchJob {
            dst_offset: 0,
            lhs_offset: 0,
            rhs_offset: 0,
            rows: 2,
            contracted: 2,
            cols: 2,
        },
        DenseGemmBatchJob {
            dst_offset: 4,
            lhs_offset: 4,
            rhs_offset: 4,
            rows: 2,
            contracted: 2,
            cols: 2,
        },
        DenseGemmBatchJob {
            dst_offset: 8,
            lhs_offset: 8,
            rhs_offset: 8,
            rows: 1,
            contracted: 3,
            cols: 1,
        },
        DenseGemmBatchJob {
            dst_offset: 9,
            lhs_offset: 11,
            rhs_offset: 11,
            rows: 1,
            contracted: 3,
            cols: 1,
        },
        DenseGemmBatchJob {
            dst_offset: 10,
            lhs_offset: 14,
            rhs_offset: 14,
            rows: 2,
            contracted: 1,
            cols: 2,
        },
    ];
    let runs = strided_batch_runs(&jobs);
    assert_eq!(runs, vec![2, 2, 1], "batch must present three runs");

    // Storage large enough for every lhs/rhs/dst range referenced above.
    let buf_len = 16usize;
    let lhs: Vec<f64> = (0..buf_len).map(|i| 1.0 + i as f64).collect();
    let rhs: Vec<f64> = (0..buf_len).map(|i| 2.0 + 0.5 * i as f64).collect();
    let mut output = vec![-99.0; buf_len];
    let flat_shape = [buf_len];
    let flat_strides = [1usize];

    let mut executor = DefaultDenseExecutor::new();
    executor.reset_seam_dispatches();
    executor
        .matmul_batch_axpby_into(
            DenseWrite::F64(DenseViewMut::new(&mut output, &flat_shape, &flat_strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&lhs, &flat_shape, &flat_strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&rhs, &flat_shape, &flat_strides, 0).unwrap()),
            &jobs,
            &runs,
            DenseScalar::F64(1.0),
            DenseScalar::F64(0.0),
        )
        .unwrap();

    assert_eq!(
        executor.seam_dispatches(),
        1,
        "three sub-cutoff runs must bundle into one grouped seam dispatch, got {}",
        executor.seam_dispatches()
    );
    assert!(
        executor.seam_dispatches() < runs.len(),
        "seam dispatch count must not scale with the number of runs"
    );

    // Byte-for-byte correctness of every bundled job.
    for job in &jobs {
        let lhs_block = &lhs[job.lhs_offset..job.lhs_offset + job.rows * job.contracted];
        let rhs_block = &rhs[job.rhs_offset..job.rhs_offset + job.contracted * job.cols];
        let expected = matmul_f64(lhs_block, rhs_block, job.rows, job.contracted, job.cols);
        let got = &output[job.dst_offset..job.dst_offset + job.rows * job.cols];
        for (actual, expected) in got.iter().zip(expected) {
            assert_f64_close(*actual, expected, 1.0e-12);
        }
    }
}
#[cfg(feature = "tenferro")]
#[test]
fn default_executor_qr_reads_transposed_views_for_all_linalg_dtypes() {
    let f32_data = vec![1.0_f32, -2.0, 3.0, 0.5, -1.0, 4.0];
    let f64_data = vec![1.0_f64, -2.0, 3.0, 0.5, -1.0, 4.0];
    let c32_data = vec![
        Complex32::new(1.0, 0.5),
        Complex32::new(-2.0, 1.0),
        Complex32::new(3.0, -0.25),
        Complex32::new(0.5, -1.0),
        Complex32::new(-1.0, 0.75),
        Complex32::new(4.0, 1.5),
    ];
    let c64_data = vec![
        Complex64::new(1.0, 0.5),
        Complex64::new(-2.0, 1.0),
        Complex64::new(3.0, -0.25),
        Complex64::new(0.5, -1.0),
        Complex64::new(-1.0, 0.75),
        Complex64::new(4.0, 1.5),
    ];
    let shape = [3, 2];
    let strides = [2, 1];
    let mut executor = DefaultDenseExecutor::new();

    let outputs = executor
        .qr(DenseRead::F32(
            DenseView::new(&f32_data, &shape, &strides, 0).unwrap(),
        ))
        .unwrap();
    assert_eq!(outputs[0].dtype(), DenseDType::F32);
    let recon = matmul_f32(
        outputs[0].as_f32_slice().unwrap(),
        outputs[1].as_f32_slice().unwrap(),
        3,
        2,
        2,
    );
    let expected = transpose_f32(&f32_data, 2, 3);
    for (actual, expected) in recon.iter().zip(expected.iter()) {
        assert_f32_close(*actual, *expected, 1.0e-5);
    }

    let outputs = executor
        .qr(DenseRead::F64(
            DenseView::new(&f64_data, &shape, &strides, 0).unwrap(),
        ))
        .unwrap();
    assert_eq!(outputs[0].dtype(), DenseDType::F64);
    let recon = matmul_f64(
        outputs[0].as_f64_slice().unwrap(),
        outputs[1].as_f64_slice().unwrap(),
        3,
        2,
        2,
    );
    let expected = transpose_f64(&f64_data, 2, 3);
    for (actual, expected) in recon.iter().zip(expected.iter()) {
        assert_f64_close(*actual, *expected, 1.0e-9);
    }

    let outputs = executor
        .qr(DenseRead::C32(
            DenseView::new(&c32_data, &shape, &strides, 0).unwrap(),
        ))
        .unwrap();
    assert_eq!(outputs[0].dtype(), DenseDType::C32);
    let recon = matmul_c32(
        outputs[0].as_c32_slice().unwrap(),
        outputs[1].as_c32_slice().unwrap(),
        3,
        2,
        2,
    );
    let expected = transpose_c32(&c32_data, 2, 3);
    for (actual, expected) in recon.iter().zip(expected.iter()) {
        assert_c32_close(*actual, *expected, 1.0e-5);
    }

    let outputs = executor
        .qr(DenseRead::C64(
            DenseView::new(&c64_data, &shape, &strides, 0).unwrap(),
        ))
        .unwrap();
    assert_eq!(outputs[0].dtype(), DenseDType::C64);
    let recon = matmul_c64(
        outputs[0].as_c64_slice().unwrap(),
        outputs[1].as_c64_slice().unwrap(),
        3,
        2,
        2,
    );
    let expected = transpose_c64(&c64_data, 2, 3);
    for (actual, expected) in recon.iter().zip(expected.iter()) {
        assert_c64_close(*actual, *expected, 1.0e-9);
    }
}
#[cfg(feature = "tenferro")]
#[test]
fn default_executor_eigh_reads_transposed_views_for_all_linalg_dtypes() {
    let f32_data = vec![4.0_f32, 1.0, 1.0, 3.0];
    let f64_data = vec![4.0_f64, 1.0, 1.0, 3.0];
    let c32_data = vec![
        Complex32::new(4.0, 0.0),
        Complex32::new(1.0, -0.5),
        Complex32::new(1.0, 0.5),
        Complex32::new(3.0, 0.0),
    ];
    let c64_data = vec![
        Complex64::new(4.0, 0.0),
        Complex64::new(1.0, -0.5),
        Complex64::new(1.0, 0.5),
        Complex64::new(3.0, 0.0),
    ];
    let shape = [2, 2];
    let strides = [2, 1];
    let mut executor = DefaultDenseExecutor::new();

    let outputs = executor
        .eigh(DenseRead::F32(
            DenseView::new(&f32_data, &shape, &strides, 0).unwrap(),
        ))
        .unwrap();
    assert_eq!(outputs[0].dtype(), DenseDType::F32);
    assert_eq!(outputs[1].dtype(), DenseDType::F32);
    let values = outputs[0].as_f32_slice().unwrap();
    let vectors = outputs[1].as_f32_slice().unwrap();
    let recon = matmul_f32(
        &matmul_f32(vectors, &diag_f32(values), 2, 2, 2),
        &transpose_f32(vectors, 2, 2),
        2,
        2,
        2,
    );
    let expected = transpose_f32(&f32_data, 2, 2);
    for (actual, expected) in recon.iter().zip(expected.iter()) {
        assert_f32_close(*actual, *expected, 1.0e-5);
    }

    let outputs = executor
        .eigh(DenseRead::F64(
            DenseView::new(&f64_data, &shape, &strides, 0).unwrap(),
        ))
        .unwrap();
    assert_eq!(outputs[0].dtype(), DenseDType::F64);
    assert_eq!(outputs[1].dtype(), DenseDType::F64);
    let values = outputs[0].as_f64_slice().unwrap();
    let vectors = outputs[1].as_f64_slice().unwrap();
    let recon = matmul_f64(
        &matmul_f64(vectors, &diag_f64(values), 2, 2, 2),
        &transpose_f64(vectors, 2, 2),
        2,
        2,
        2,
    );
    let expected = transpose_f64(&f64_data, 2, 2);
    for (actual, expected) in recon.iter().zip(expected.iter()) {
        assert_f64_close(*actual, *expected, 1.0e-10);
    }

    let outputs = executor
        .eigh(DenseRead::C32(
            DenseView::new(&c32_data, &shape, &strides, 0).unwrap(),
        ))
        .unwrap();
    assert_eq!(outputs[0].dtype(), DenseDType::F32);
    assert_eq!(outputs[1].dtype(), DenseDType::C32);
    let values = outputs[0].as_f32_slice().unwrap();
    let vectors = outputs[1].as_c32_slice().unwrap();
    let recon = matmul_c32(
        &matmul_c32(vectors, &diag_c32_from_real(values), 2, 2, 2),
        &conjugate_transpose_c32(vectors, 2, 2),
        2,
        2,
        2,
    );
    let expected = transpose_c32(&c32_data, 2, 2);
    for (actual, expected) in recon.iter().zip(expected.iter()) {
        assert_c32_close(*actual, *expected, 1.0e-5);
    }

    let outputs = executor
        .eigh(DenseRead::C64(
            DenseView::new(&c64_data, &shape, &strides, 0).unwrap(),
        ))
        .unwrap();
    assert_eq!(outputs[0].dtype(), DenseDType::F64);
    assert_eq!(outputs[1].dtype(), DenseDType::C64);
    let values = outputs[0].as_f64_slice().unwrap();
    let vectors = outputs[1].as_c64_slice().unwrap();
    let recon = matmul_c64(
        &matmul_c64(vectors, &diag_c64_from_real(values), 2, 2, 2),
        &conjugate_transpose_c64(vectors, 2, 2),
        2,
        2,
        2,
    );
    let expected = transpose_c64(&c64_data, 2, 2);
    for (actual, expected) in recon.iter().zip(expected.iter()) {
        assert_c64_close(*actual, *expected, 1.0e-10);
    }
}
#[cfg(feature = "tenferro")]
#[test]
fn default_executor_svd_accepts_all_supported_linalg_dtypes() {
    let f32_data = [1.0_f32, -2.0, 0.5, 4.0];
    let f64_data = [1.0_f64, -2.0, 0.5, 4.0];
    let c32_data = [
        Complex32::new(1.0, 0.5),
        Complex32::new(-2.0, 1.0),
        Complex32::new(0.5, -0.25),
        Complex32::new(4.0, 1.5),
    ];
    let c64_data = [
        Complex64::new(1.0, 0.5),
        Complex64::new(-2.0, 1.0),
        Complex64::new(0.5, -0.25),
        Complex64::new(4.0, 1.5),
    ];
    let shape = [2, 2];
    let strides = [2, 1];

    let mut executor = DefaultDenseExecutor::new();
    for (input, dtype) in [
        (
            DenseRead::F32(DenseView::new(&f32_data, &shape, &strides, 0).unwrap()),
            DenseDType::F32,
        ),
        (
            DenseRead::F64(DenseView::new(&f64_data, &shape, &strides, 0).unwrap()),
            DenseDType::F64,
        ),
        (
            DenseRead::C32(DenseView::new(&c32_data, &shape, &strides, 0).unwrap()),
            DenseDType::C32,
        ),
        (
            DenseRead::C64(DenseView::new(&c64_data, &shape, &strides, 0).unwrap()),
            DenseDType::C64,
        ),
    ] {
        let outputs = executor.svd(input).unwrap();
        assert_eq!(outputs[0].dtype(), dtype);
        assert!(matches!(
            (dtype, outputs[1].dtype()),
            (DenseDType::F32, DenseDType::F32)
                | (DenseDType::F64, DenseDType::F64)
                | (DenseDType::C32, DenseDType::F32)
                | (DenseDType::C64, DenseDType::F64)
        ));
        assert_eq!(outputs[2].dtype(), dtype);
    }
}
#[cfg(feature = "tenferro")]
#[test]
fn default_executor_svd_into_writes_strided_destination_views() {
    let data = [1.0_f64, -2.0, 0.5, 4.0];
    let input_shape = [2, 2];
    let input_strides = [1, 2];
    let input = DenseRead::F64(DenseView::new(&data, &input_shape, &input_strides, 0).unwrap());

    let mut executor = DefaultDenseExecutor::new();
    let expected = executor.svd(input).unwrap();

    let mut u = vec![-99.0; 8];
    let mut s = vec![-99.0; 4];
    let mut vt = vec![-99.0; 8];
    let matrix_shape = [2, 2];
    let matrix_strides = [1, 3];
    let s_shape = [2];
    let s_strides = [2];
    executor
        .svd_into(
            input,
            DenseWrite::F64(DenseViewMut::new(&mut u, &matrix_shape, &matrix_strides, 1).unwrap()),
            DenseWrite::F64(DenseViewMut::new(&mut s, &s_shape, &s_strides, 0).unwrap()),
            DenseWrite::F64(DenseViewMut::new(&mut vt, &matrix_shape, &matrix_strides, 1).unwrap()),
        )
        .unwrap();

    let expected_u = expected[0].as_f64_slice().unwrap();
    let expected_s = expected[1].as_f64_slice().unwrap();
    let expected_vt = expected[2].as_f64_slice().unwrap();
    for col in 0..2 {
        for row in 0..2 {
            assert_f64_close(u[1 + row + 3 * col], expected_u[row + 2 * col], 1e-12);
            assert_f64_close(vt[1 + row + 3 * col], expected_vt[row + 2 * col], 1e-12);
        }
    }
    for index in 0..2 {
        assert_f64_close(s[2 * index], expected_s[index], 1e-12);
    }
}
#[cfg(feature = "tenferro")]
#[test]
fn default_executor_qr_into_writes_strided_destination_views() {
    let data = [
        Complex64::new(1.0, 0.5),
        Complex64::new(-2.0, 1.0),
        Complex64::new(0.5, -0.25),
        Complex64::new(4.0, 1.5),
    ];
    let input_shape = [2, 2];
    let input_strides = [1, 2];
    let input = DenseRead::C64(DenseView::new(&data, &input_shape, &input_strides, 0).unwrap());

    let mut executor = DefaultDenseExecutor::new();
    let expected = executor.qr(input).unwrap();

    let sentinel = Complex64::new(-99.0, 0.0);
    let mut q = vec![sentinel; 8];
    let mut r = vec![sentinel; 8];
    let matrix_shape = [2, 2];
    let matrix_strides = [1, 3];
    executor
        .qr_into(
            input,
            DenseWrite::C64(DenseViewMut::new(&mut q, &matrix_shape, &matrix_strides, 1).unwrap()),
            DenseWrite::C64(DenseViewMut::new(&mut r, &matrix_shape, &matrix_strides, 1).unwrap()),
        )
        .unwrap();

    let expected_q = expected[0].as_c64_slice().unwrap();
    let expected_r = expected[1].as_c64_slice().unwrap();
    for col in 0..2 {
        for row in 0..2 {
            assert_c64_close(q[1 + row + 3 * col], expected_q[row + 2 * col], 1e-12);
            assert_c64_close(r[1 + row + 3 * col], expected_r[row + 2 * col], 1e-12);
        }
    }
}
#[cfg(feature = "tenferro")]
#[test]
fn default_executor_eigh_into_writes_strided_destination_views() {
    let data = [
        Complex64::new(4.0, 0.0),
        Complex64::new(1.0, 0.5),
        Complex64::new(1.0, -0.5),
        Complex64::new(3.0, 0.0),
    ];
    let input_shape = [2, 2];
    let input_strides = [1, 2];
    let input = DenseRead::C64(DenseView::new(&data, &input_shape, &input_strides, 0).unwrap());

    let mut executor = DefaultDenseExecutor::new();
    let expected = executor.eigh(input).unwrap();

    let mut values = vec![-99.0; 4];
    let sentinel = Complex64::new(-99.0, 0.0);
    let mut vectors = vec![sentinel; 8];
    let values_shape = [2];
    let values_strides = [2];
    let matrix_shape = [2, 2];
    let matrix_strides = [1, 3];
    executor
        .eigh_into(
            input,
            DenseWrite::F64(
                DenseViewMut::new(&mut values, &values_shape, &values_strides, 1).unwrap(),
            ),
            DenseWrite::C64(
                DenseViewMut::new(&mut vectors, &matrix_shape, &matrix_strides, 1).unwrap(),
            ),
        )
        .unwrap();

    let expected_values = expected[0].as_f64_slice().unwrap();
    let expected_vectors = expected[1].as_c64_slice().unwrap();
    for index in 0..2 {
        assert_f64_close(values[1 + 2 * index], expected_values[index], 1e-12);
    }
    for col in 0..2 {
        for row in 0..2 {
            assert_c64_close(
                vectors[1 + row + 3 * col],
                expected_vectors[row + 2 * col],
                1e-12,
            );
        }
    }
}
#[cfg(feature = "tenferro")]
#[test]
fn default_executor_rejects_integer_linalg_view() {
    let data = [1_i32, 0, 0, 1];
    let shape = [2, 2];
    let strides = [1, 2];
    let view = DenseView::new(&data, &shape, &strides, 0).unwrap();

    let mut executor = DefaultDenseExecutor::new();
    let err = executor.qr(DenseRead::I32(view)).unwrap_err();

    assert!(matches!(
        err,
        DenseError::Unsupported {
            op: "qr_read",
            ref message,
        } if message.contains("does not support dtype I32")
    ));
}
struct FullOnly(DefaultDenseExecutor);
impl DenseExecutor for FullOnly {
    fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.0.svd(input)
    }

    fn qr(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.0.qr(input)
    }

    fn eigh(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.0.eigh(input)
    }

    fn eig(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.0.eig(input)
    }

    fn dot_general_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        config: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        self.0.dot_general_into(output, lhs, rhs, config)
    }
}
#[test]
fn owned_full_svd_default_is_explicitly_unsupported() {
    let mut executor = FullOnly(DefaultDenseExecutor::new());

    assert!(!executor.supports_svd_full());
    let error = executor
        .svd_full_owned(DenseOwned::F64(vec![1.0]), 1, 1)
        .unwrap_err();

    assert!(matches!(
        error,
        DenseError::Unsupported {
            op: "svd_full_owned",
            ..
        }
    ));
}
/// Implements only the required trait methods, so the accumulate-form matmul
/// falls through to the [`DenseExecutor`] default.
#[derive(Default)]
struct NoAxpby {
    dot_calls: usize,
}
impl DenseExecutor for NoAxpby {
    fn svd(&mut self, _input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        unreachable!("the accumulate-form matmul default never factorizes")
    }

    fn qr(&mut self, _input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        unreachable!("the accumulate-form matmul default never factorizes")
    }

    fn eigh(&mut self, _input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        unreachable!("the accumulate-form matmul default never factorizes")
    }

    fn dot_general_into(
        &mut self,
        _output: DenseWrite<'_>,
        _lhs: DenseRead<'_>,
        _rhs: DenseRead<'_>,
        _config: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        self.dot_calls += 1;
        Ok(())
    }
}
#[test]
fn accumulate_form_matmul_default_overwrites_then_reports_unsupported() {
    let mut executor = NoAxpby::default();
    let lhs = [1.0_f64];
    let rhs = [1.0_f64];
    let mut output = [0.0_f64];
    let shape = [1, 1];
    let strides = [1, 1];

    executor
        .matmul_axpby_into(
            DenseWrite::F64(DenseViewMut::new(&mut output, &shape, &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&lhs, &shape, &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&rhs, &shape, &strides, 0).unwrap()),
            DenseScalar::F64(1.0),
            DenseScalar::F64(0.0),
        )
        .unwrap();
    assert_eq!(
        executor.dot_calls, 1,
        "alpha = 1, beta = 0 must delegate to matmul_into"
    );

    let error = executor
        .matmul_axpby_into(
            DenseWrite::F64(DenseViewMut::new(&mut output, &shape, &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&lhs, &shape, &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&rhs, &shape, &strides, 0).unwrap()),
            DenseScalar::F64(1.0),
            DenseScalar::F64(1.0),
        )
        .unwrap_err();
    assert!(
        matches!(
            error,
            DenseError::Unsupported {
                op: "matmul_axpby_into",
                ..
            }
        ),
        "a missing accumulate-form capability must be Unsupported, got {error:?}"
    );
    assert_eq!(
        executor.dot_calls, 1,
        "the unsupported arm must not drive a kernel"
    );
}
#[cfg(all(
    any(feature = "cpu-faer", feature = "cpu-blas-core"),
    not(feature = "provider-inject")
))]
#[test]
fn default_executor_advertises_native_owned_full_svd() {
    assert!(DefaultDenseExecutor::new().supports_svd_full());
}
#[cfg(all(
    any(feature = "cpu-faer", feature = "cpu-blas-core"),
    not(feature = "provider-inject")
))]
#[test]
fn native_owned_full_svd_validates_overflow_length_then_zero_extent() {
    let mut executor = DefaultDenseExecutor::new();
    let inputs = crate::probe::record();

    assert!(matches!(
        executor.svd_full_owned(DenseOwned::F64(Vec::new()), usize::MAX, 2),
        Err(DenseError::ElementCountOverflow)
    ));
    assert!(matches!(
        executor.svd_full_owned(DenseOwned::F64(vec![1.0]), 2, 2),
        Err(DenseError::ShapeMismatch {
            op: "svd_full_owned",
            ..
        })
    ));
    assert!(matches!(
        executor.svd_full_owned(DenseOwned::F64(Vec::new()), 0, 2),
        Err(DenseError::Unsupported {
            op: "svd_full_owned",
            ..
        })
    ));
    assert!(matches!(
        executor.svd_full_owned(DenseOwned::F64(Vec::new()), 2, 0),
        Err(DenseError::Unsupported {
            op: "svd_full_owned",
            ..
        })
    ));
    assert!(matches!(
        executor.svd_full_owned(DenseOwned::F64(Vec::new()), 0, 0),
        Err(DenseError::Unsupported {
            op: "svd_full_owned",
            ..
        })
    ));
    assert!(matches!(
        executor.svd_full_owned(DenseOwned::F64(vec![1.0]), 0, 2),
        Err(DenseError::ShapeMismatch {
            op: "svd_full_owned",
            ..
        })
    ));
    assert!(inputs.owned_full_svd_inputs().is_empty());
}
#[cfg(all(
    any(feature = "cpu-faer", feature = "cpu-blas-core"),
    not(feature = "provider-inject")
))]
#[test]
fn native_owned_full_svd_moves_each_dtype_input_buffer() {
    let mut executor = DefaultDenseExecutor::new();
    let inputs = crate::probe::record();
    let mut expected = Vec::new();

    let f32 = vec![1.0_f32, 3.0, 2.0, 4.0];
    expected.push(f32.as_ptr() as usize);
    assert_eq!(
        executor
            .svd_full_owned(DenseOwned::F32(f32), 2, 2)
            .unwrap()
            .len(),
        3
    );
    let f64 = vec![1.0_f64, 3.0, 2.0, 4.0];
    expected.push(f64.as_ptr() as usize);
    assert_eq!(
        executor
            .svd_full_owned(DenseOwned::F64(f64), 2, 2)
            .unwrap()
            .len(),
        3
    );
    let c32 = vec![Complex32::new(1.0, 1.0); 4];
    expected.push(c32.as_ptr() as usize);
    assert_eq!(
        executor
            .svd_full_owned(DenseOwned::C32(c32), 2, 2)
            .unwrap()
            .len(),
        3
    );
    let c64 = vec![Complex64::new(1.0, 1.0); 4];
    expected.push(c64.as_ptr() as usize);
    assert_eq!(
        executor
            .svd_full_owned(DenseOwned::C64(c64), 2, 2)
            .unwrap()
            .len(),
        3
    );

    assert_eq!(inputs.owned_full_svd_inputs(), expected);
}
#[test]
fn solve_default_is_explicitly_unsupported_without_writing() {
    // What: executors without solve capability reject the operation before
    // publishing anything into the caller's destination.
    let a = [1.0_f64];
    let b = [2.0_f64];
    let mut x = [37.0_f64];
    let shape = [1, 1];
    let strides = [1, 1];
    let mut executor = FullOnly(DefaultDenseExecutor::new());

    let error = executor
        .solve_into(
            DenseRead::F64(DenseView::new(&a, &shape, &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&b, &shape, &strides, 0).unwrap()),
            DenseWrite::F64(DenseViewMut::new(&mut x, &shape, &strides, 0).unwrap()),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        DenseError::Unsupported {
            op: "solve_into",
            ..
        }
    ));
    assert_eq!(x, [37.0]);
}
#[test]
fn default_executor_solves_f64_from_and_into_strided_views() {
    // What: solve reads legal noncontiguous coefficients and writes only the
    // logical elements of a noncontiguous caller-owned destination.
    let mut a = vec![-99.0; 10];
    a[1] = 3.0;
    a[3] = 1.0;
    a[6] = 1.0;
    a[8] = 2.0;
    let mut b = vec![-88.0; 12];
    b[2] = 2.0;
    b[5] = -1.0;
    b[8] = 9.0;
    b[11] = 8.0;
    let mut x = vec![-77.0; 9];
    let shape = [2, 2];
    let a_strides = [2, 5];
    let b_strides = [3, 6];
    let x_strides = [2, 5];
    let mut executor = DefaultDenseExecutor::with_threads(1).unwrap();

    executor
        .solve_into(
            DenseRead::F64(DenseView::new(&a, &shape, &a_strides, 1).unwrap()),
            DenseRead::F64(DenseView::new(&b, &shape, &b_strides, 2).unwrap()),
            DenseWrite::F64(DenseViewMut::new(&mut x, &shape, &x_strides, 0).unwrap()),
        )
        .unwrap();

    for (actual, expected) in [x[0], x[2], x[5], x[7]]
        .into_iter()
        .zip([1.0, -1.0, 2.0, 3.0])
    {
        assert_f64_close(actual, expected, 1.0e-12);
    }
    assert_eq!([x[1], x[3], x[4], x[6], x[8]], [-77.0; 5]);
}
#[test]
fn default_executor_solves_c64_system() {
    // What: complex solve preserves both real and imaginary components for a
    // rank-2 right-hand side.
    let c = |re, im| Complex64::new(re, im);
    let a = [c(2.0, 1.0), c(0.0, 0.0), c(1.0, 0.0), c(1.0, -1.0)];
    let b = [c(-1.0, 6.0), c(0.0, 2.0)];
    let mut x = [c(0.0, 0.0); 2];
    let a_shape = [2, 2];
    let b_shape = [2, 1];
    let strides = [1, 2];
    let mut executor = DefaultDenseExecutor::with_threads(1).unwrap();

    executor
        .solve_into(
            DenseRead::C64(DenseView::new(&a, &a_shape, &strides, 0).unwrap()),
            DenseRead::C64(DenseView::new(&b, &b_shape, &strides, 0).unwrap()),
            DenseWrite::C64(DenseViewMut::new(&mut x, &b_shape, &strides, 0).unwrap()),
        )
        .unwrap();

    assert_c64_close(x[0], c(1.0, 2.0), 1.0e-12);
    assert_c64_close(x[1], c(-1.0, 1.0), 1.0e-12);
}
#[test]
fn solve_validates_destination_before_singular_factorization() {
    // What: destination dtype and shape errors take precedence over a singular
    // factorization, while a valid destination exposes the numerical failure;
    // no failed call publishes output.
    let a = [1.0_f64, 2.0, 2.0, 4.0];
    let b = [1.0_f64, 1.0];
    let a_shape = [2, 2];
    let b_shape = [2, 1];
    let strides = [1, 2];
    let mut executor = DefaultDenseExecutor::with_threads(1).unwrap();

    let mut wrong_dtype = [Complex64::new(19.0, 23.0); 2];
    let error = executor
        .solve_into(
            DenseRead::F64(DenseView::new(&a, &a_shape, &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&b, &b_shape, &strides, 0).unwrap()),
            DenseWrite::C64(DenseViewMut::new(&mut wrong_dtype, &b_shape, &strides, 0).unwrap()),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        DenseError::DTypeMismatch {
            op: "solve_into",
            expected: DenseDType::F64,
            actual: DenseDType::C64,
        }
    ));
    assert_eq!(wrong_dtype, [Complex64::new(19.0, 23.0); 2]);

    let mut wrong_shape = [29.0_f64];
    let wrong_shape_dims = [1, 1];
    let error = executor
        .solve_into(
            DenseRead::F64(DenseView::new(&a, &a_shape, &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&b, &b_shape, &strides, 0).unwrap()),
            DenseWrite::F64(
                DenseViewMut::new(&mut wrong_shape, &wrong_shape_dims, &strides, 0).unwrap(),
            ),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        DenseError::ShapeMismatch {
            op: "solve_into",
            ref expected,
            ref actual,
        } if expected == &[2, 1] && actual == &[1, 1]
    ));
    assert_eq!(wrong_shape, [29.0]);

    let mut x = [13.0_f64, 17.0];
    let error = executor
        .solve_into(
            DenseRead::F64(DenseView::new(&a, &a_shape, &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&b, &b_shape, &strides, 0).unwrap()),
            DenseWrite::F64(DenseViewMut::new(&mut x, &b_shape, &strides, 0).unwrap()),
        )
        .unwrap_err();

    assert!(matches!(
        error,
        DenseError::NumericalFailure {
            backend: DenseBackend::Tenferro,
            op: "solve_into",
            ..
        }
    ));
    assert_eq!(x, [13.0, 17.0]);
}
// Exercises the values-only trait *defaults* (full decomposition minus the
// vectors). `DefaultDenseExecutor` overrides them, so this wraps it in an
// executor that implements svd/eigh/eig but leaves svd_vals/eigh_vals/eig_vals
// at their trait default, then checks the fallback spectra agree with the
// backend's no-vector override to LAPACK precision.
#[test]
fn values_only_defaults_fall_back_to_the_full_decomposition_spectrum() {
    let shape = [2usize, 2];
    let strides = [1usize, 2]; // column-major
    let m = vec![2.0f64, 1.0, 1.0, 3.0]; // symmetric, so eigh applies too
    fn view<'a>(data: &'a [f64], shape: &'a [usize], strides: &'a [usize]) -> DenseRead<'a> {
        DenseRead::F64(DenseView::new(data, shape, strides, 0).unwrap())
    }

    let mut fallback = FullOnly(DefaultDenseExecutor::new());
    let mut direct = DefaultDenseExecutor::new();
    let tol = 1e-10;

    let f = fallback.svd_vals(view(&m, &shape, &strides)).unwrap();
    let d = direct.svd_vals(view(&m, &shape, &strides)).unwrap();
    let (f, d) = (f.as_f64_slice().unwrap(), d.as_f64_slice().unwrap());
    assert_eq!(f.len(), 2);
    for (a, b) in f.iter().zip(d) {
        assert_f64_close(*a, *b, tol);
    }

    let f = fallback.eigh_vals(view(&m, &shape, &strides)).unwrap();
    let d = direct.eigh_vals(view(&m, &shape, &strides)).unwrap();
    let (f, d) = (f.as_f64_slice().unwrap(), d.as_f64_slice().unwrap());
    assert_eq!(f.len(), 2);
    for (a, b) in f.iter().zip(d) {
        assert_f64_close(*a, *b, tol);
    }

    let f = fallback.eig_vals(view(&m, &shape, &strides)).unwrap();
    let d = direct.eig_vals(view(&m, &shape, &strides)).unwrap();
    let (f, d) = (f.as_c64_slice().unwrap(), d.as_c64_slice().unwrap());
    assert_eq!(f.len(), 2);
    for (a, b) in f.iter().zip(d) {
        assert_c64_close(*a, *b, tol);
    }
}
#[cfg(not(feature = "provider-inject"))]
#[test]
fn default_executor_svd_vals_returns_literal_spectra_for_every_dense_dtype() {
    let mut executor = DefaultDenseExecutor::new();
    let c32 = |re, im| Complex32::new(re, im);
    let c64 = |re, im| Complex64::new(re, im);

    let f32_shape = [2, 3];
    let f32_data = [0.0_f32, 0.0, 3.0, 0.0, 4.0, 0.0];
    let f32_before = f32_data;
    let values = executor
        .svd_vals(DenseRead::F32(
            DenseView::new(&f32_data, &f32_shape, &[1, 2], 0).unwrap(),
        ))
        .unwrap();
    assert_eq!(values.dtype(), DenseDType::F32);
    let values = values.as_f32_slice().unwrap();
    assert_eq!(values.len(), 2);
    for (actual, expected) in values.iter().zip([5.0, 0.0]) {
        assert_f32_close(*actual, expected, 1.0e-5);
    }
    assert_eq!(f32_data, f32_before);

    let f64_shape = [3, 2];
    let f64_data = [3.0_f64, 4.0, 0.0, 0.0, 0.0, 0.0];
    let f64_before = f64_data;
    let values = executor
        .svd_vals(DenseRead::F64(
            DenseView::new(&f64_data, &f64_shape, &[1, 3], 0).unwrap(),
        ))
        .unwrap();
    assert_eq!(values.dtype(), DenseDType::F64);
    let values = values.as_f64_slice().unwrap();
    assert_eq!(values.len(), 2);
    for (actual, expected) in values.iter().zip([5.0, 0.0]) {
        assert_f64_close(*actual, expected, 1.0e-12);
    }
    assert_eq!(f64_data, f64_before);

    let c32_shape = [2, 2];
    let c32_data = [c32(1.0, 0.0), c32(0.0, 1.0), c32(0.0, 1.0), c32(1.0, 0.0)];
    let c32_before = c32_data;
    let values = executor
        .svd_vals(DenseRead::C32(
            DenseView::new(&c32_data, &c32_shape, &[1, 2], 0).unwrap(),
        ))
        .unwrap();
    assert_eq!(values.dtype(), DenseDType::F32);
    let values = values.as_f32_slice().unwrap();
    assert_eq!(values.len(), 2);
    for (actual, expected) in values.iter().zip([2.0_f32.sqrt(), 2.0_f32.sqrt()]) {
        assert_f32_close(*actual, expected, 1.0e-5);
    }
    assert_eq!(c32_data, c32_before);

    let c64_shape = [2, 2];
    let c64_data = [c64(1.0, 0.0), c64(0.0, 1.0), c64(0.0, 1.0), c64(1.0, 0.0)];
    let c64_before = c64_data;
    let values = executor
        .svd_vals(DenseRead::C64(
            DenseView::new(&c64_data, &c64_shape, &[1, 2], 0).unwrap(),
        ))
        .unwrap();
    assert_eq!(values.dtype(), DenseDType::F64);
    let values = values.as_f64_slice().unwrap();
    assert_eq!(values.len(), 2);
    for (actual, expected) in values.iter().zip([2.0_f64.sqrt(), 2.0_f64.sqrt()]) {
        assert_f64_close(*actual, expected, 1.0e-12);
    }
    assert_eq!(c64_data, c64_before);
}
#[cfg(not(feature = "provider-inject"))]
#[test]
fn default_executor_eigh_vals_returns_literal_spectra_for_every_dense_dtype() {
    let mut executor = DefaultDenseExecutor::new();
    let c32 = |re, im| Complex32::new(re, im);
    let c64 = |re, im| Complex64::new(re, im);
    let shape = [2, 2];
    let strides = [1, 2];

    let f32_data = [-2.0_f32, 0.0, 0.0, 3.0];
    let f32_before = f32_data;
    let values = executor
        .eigh_vals(DenseRead::F32(
            DenseView::new(&f32_data, &shape, &strides, 0).unwrap(),
        ))
        .unwrap();
    assert_eq!(values.dtype(), DenseDType::F32);
    let values = values.as_f32_slice().unwrap();
    assert_eq!(values.len(), 2);
    for (actual, expected) in values.iter().zip([-2.0, 3.0]) {
        assert_f32_close(*actual, expected, 1.0e-5);
    }
    assert_eq!(f32_data, f32_before);

    let f64_data = [1.0_f64, 2.0, 2.0, 1.0];
    let f64_before = f64_data;
    let values = executor
        .eigh_vals(DenseRead::F64(
            DenseView::new(&f64_data, &shape, &strides, 0).unwrap(),
        ))
        .unwrap();
    assert_eq!(values.dtype(), DenseDType::F64);
    let values = values.as_f64_slice().unwrap();
    assert_eq!(values.len(), 2);
    for (actual, expected) in values.iter().zip([-1.0, 3.0]) {
        assert_f64_close(*actual, expected, 1.0e-12);
    }
    assert_eq!(f64_data, f64_before);

    let c32_data = [c32(2.0, 0.0), c32(0.0, -1.0), c32(0.0, 1.0), c32(2.0, 0.0)];
    let c32_before = c32_data;
    let values = executor
        .eigh_vals(DenseRead::C32(
            DenseView::new(&c32_data, &shape, &strides, 0).unwrap(),
        ))
        .unwrap();
    assert_eq!(values.dtype(), DenseDType::F32);
    let values = values.as_f32_slice().unwrap();
    assert_eq!(values.len(), 2);
    for (actual, expected) in values.iter().zip([1.0, 3.0]) {
        assert_f32_close(*actual, expected, 1.0e-5);
    }
    assert_eq!(c32_data, c32_before);

    let c64_data = [c64(2.0, 0.0), c64(0.0, -1.0), c64(0.0, 1.0), c64(2.0, 0.0)];
    let c64_before = c64_data;
    let values = executor
        .eigh_vals(DenseRead::C64(
            DenseView::new(&c64_data, &shape, &strides, 0).unwrap(),
        ))
        .unwrap();
    assert_eq!(values.dtype(), DenseDType::F64);
    let values = values.as_f64_slice().unwrap();
    assert_eq!(values.len(), 2);
    for (actual, expected) in values.iter().zip([1.0, 3.0]) {
        assert_f64_close(*actual, expected, 1.0e-12);
    }
    assert_eq!(c64_data, c64_before);
}
#[cfg(not(feature = "provider-inject"))]
#[test]
fn default_executor_values_only_reads_offset_padded_transposed_views() {
    let shape = [2, 2];
    let strides = [7, 3];
    let mut executor = DefaultDenseExecutor::new();

    let mut svd_data = vec![-17.0_f64; 12];
    svd_data[1] = 0.0;
    svd_data[4] = 3.0;
    svd_data[8] = 4.0;
    svd_data[11] = 0.0;
    let svd_before = svd_data.clone();
    let values = executor
        .svd_vals(DenseRead::F64(
            DenseView::new(&svd_data, &shape, &strides, 1).unwrap(),
        ))
        .unwrap();
    let values = values.as_f64_slice().unwrap();
    assert_eq!(values.len(), 2);
    for (actual, expected) in values.iter().zip([4.0, 3.0]) {
        assert_f64_close(*actual, expected, 1.0e-12);
    }
    assert_eq!(svd_data, svd_before);

    let c = |re, im| Complex64::new(re, im);
    let mut eigh_data = vec![c(-17.0, 9.0); 12];
    eigh_data[1] = c(2.0, 0.0);
    eigh_data[4] = c(0.0, 1.0);
    eigh_data[8] = c(0.0, -1.0);
    eigh_data[11] = c(2.0, 0.0);
    let eigh_before = eigh_data.clone();
    let values = executor
        .eigh_vals(DenseRead::C64(
            DenseView::new(&eigh_data, &shape, &strides, 1).unwrap(),
        ))
        .unwrap();
    let values = values.as_f64_slice().unwrap();
    assert_eq!(values.len(), 2);
    for (actual, expected) in values.iter().zip([1.0, 3.0]) {
        assert_f64_close(*actual, expected, 1.0e-12);
    }
    assert_eq!(eigh_data, eigh_before);

    let overlap_data = [2.0_f64];
    let values = executor
        .svd_vals(DenseRead::F64(
            DenseView::new(&overlap_data, &shape, &[0, 0], 0).unwrap(),
        ))
        .unwrap();
    let values = values.as_f64_slice().unwrap();
    assert_eq!(values.len(), 2);
    for (actual, expected) in values.iter().zip([4.0, 0.0]) {
        assert_f64_close(*actual, expected, 1.0e-12);
    }
    assert_eq!(overlap_data, [2.0]);

    let values = executor
        .eigh_vals(DenseRead::F64(
            DenseView::new(&overlap_data, &shape, &[0, 0], 0).unwrap(),
        ))
        .unwrap();
    let values = values.as_f64_slice().unwrap();
    assert_eq!(values.len(), 2);
    for (actual, expected) in values.iter().zip([0.0, 4.0]) {
        assert_f64_close(*actual, expected, 1.0e-12);
    }
    assert_eq!(overlap_data, [2.0]);
}
#[cfg(not(feature = "provider-inject"))]
#[test]
fn default_executor_values_only_admits_batches_and_zero_extents() {
    let mut executor = DefaultDenseExecutor::new();

    let batch_shape = [2, 2, 2];
    let batch_data = [3.0_f64, 0.0, 0.0, 1.0, 4.0, 0.0, 0.0, 2.0];
    let batch_before = batch_data;
    let values = executor
        .svd_vals(DenseRead::F64(
            DenseView::new(&batch_data, &batch_shape, &[1, 2, 4], 0).unwrap(),
        ))
        .unwrap();
    assert_eq!(values.shape(), &[2, 2]);
    for (actual, expected) in values
        .as_f64_slice()
        .unwrap()
        .iter()
        .zip([3.0, 1.0, 4.0, 2.0])
    {
        assert_f64_close(*actual, expected, 1.0e-12);
    }
    assert_eq!(batch_data, batch_before);

    let values = executor
        .eigh_vals(DenseRead::F64(
            DenseView::new(&batch_data, &batch_shape, &[1, 2, 4], 0).unwrap(),
        ))
        .unwrap();
    assert_eq!(values.shape(), &[2, 2]);
    for (actual, expected) in values
        .as_f64_slice()
        .unwrap()
        .iter()
        .zip([1.0, 3.0, 2.0, 4.0])
    {
        assert_f64_close(*actual, expected, 1.0e-12);
    }
    assert_eq!(batch_data, batch_before);

    for (shape, strides) in [([0, 3], [1, 0]), ([3, 0], [1, 3]), ([0, 0], [1, 0])] {
        let values = executor
            .svd_vals(DenseRead::F64(
                DenseView::new(&[] as &[f64], &shape, &strides, 0).unwrap(),
            ))
            .unwrap();
        assert_eq!(values.shape(), &[0]);
        assert!(values.as_f64_slice().unwrap().is_empty());
    }

    let values = executor
        .eigh_vals(DenseRead::F64(
            DenseView::new(&[] as &[f64], &[0, 0], &[1, 0], 0).unwrap(),
        ))
        .unwrap();
    assert_eq!(values.shape(), &[0]);
    assert!(values.as_f64_slice().unwrap().is_empty());

    let values = executor
        .svd_vals(DenseRead::F64(
            DenseView::new(&[] as &[f64], &[2, 2, 0], &[1, 2, 4], 0).unwrap(),
        ))
        .unwrap();
    assert_eq!(values.shape(), &[2, 0]);
    assert!(values.as_f64_slice().unwrap().is_empty());
}
#[cfg(not(feature = "provider-inject"))]
#[test]
fn default_executor_values_only_preserves_rank_and_dtype_rejections() {
    let mut executor = DefaultDenseExecutor::new();

    let svd_rank_one = [1.0_f64, 2.0];
    let error = executor
        .svd_vals(DenseRead::F64(
            DenseView::new(&svd_rank_one, &[2], &[1], 0).unwrap(),
        ))
        .unwrap_err();
    assert!(matches!(
        error,
        DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op: "svd_values",
            ..
        }
    ));

    let rank_zero = [1.0_f64];
    let error = executor
        .eigh_vals(DenseRead::F64(
            DenseView::new(&rank_zero, &[], &[], 0).unwrap(),
        ))
        .unwrap_err();
    assert!(matches!(
        error,
        DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op: "eigh_values",
            ..
        }
    ));

    let nonsquare = [1.0_f64; 6];
    let error = executor
        .eigh_vals(DenseRead::F64(
            DenseView::new(&nonsquare, &[2, 3], &[1, 2], 0).unwrap(),
        ))
        .unwrap_err();
    assert!(matches!(
        error,
        DenseError::Backend {
            backend: DenseBackend::Tenferro,
            op: "eigh_values",
            ..
        }
    ));

    let integers = [1_i32, 0, 0, 1];
    let error = executor
        .svd_vals(DenseRead::I32(
            DenseView::new(&integers, &[2, 2], &[1, 2], 0).unwrap(),
        ))
        .unwrap_err();
    assert!(matches!(
        error,
        DenseError::Unsupported {
            op: "svd_values",
            ref message,
        } if message.contains("does not support dtype I32")
    ));

    let booleans = [true, false, false, true];
    let error = executor
        .eigh_vals(DenseRead::Bool(
            DenseView::new(&booleans, &[2, 2], &[1, 2], 0).unwrap(),
        ))
        .unwrap_err();
    assert!(matches!(
        error,
        DenseError::Unsupported {
            op: "eigh_values",
            ref message,
        } if message.contains("does not support dtype Bool")
    ));
}
#[cfg(all(
    not(feature = "provider-inject"),
    any(feature = "cpu-faer", feature = "cpu-blas-core")
))]
#[test]
fn default_executor_values_only_runs_the_compiled_provider() {
    assert_values_only_for_compiled_cpu_provider();
}
#[test]
fn strided_batch_runs_partitions_same_shape_constant_stride_runs() {
    // Two length-2 constant-stride runs (shapes A, B) followed by a singleton
    // (shape C): the plan-time partition the executor routes over.
    let jobs = vec![
        batch_job((2, 2, 2), (0, 0, 0)),
        batch_job((2, 2, 2), (4, 4, 4)),
        batch_job((3, 1, 2), (8, 8, 8)),
        batch_job((3, 1, 2), (14, 11, 10)),
        batch_job((1, 5, 1), (20, 14, 12)),
    ];
    assert_eq!(strided_batch_runs(&jobs), vec![2, 2, 1]);
    let mut reused = vec![usize::MAX; jobs.len()];
    let capacity = reused.capacity();
    strided_batch_runs_into(&jobs, &mut reused);
    // What: caller-owned run storage receives the same partition and retains
    // its warm capacity for subsequent batches.
    assert_eq!(reused, [2, 2, 1]);
    assert_eq!(reused.capacity(), capacity);
    strided_batch_runs_into(&[], &mut reused);
    assert!(reused.is_empty());
    assert_eq!(reused.capacity(), capacity);
    // Empty batch => empty partition; the lengths always cover every job.
    assert_eq!(strided_batch_runs(&[]), Vec::<usize>::new());
    assert_eq!(
        strided_batch_runs(&jobs).iter().sum::<usize>(),
        jobs.len(),
        "run partition must cover all jobs"
    );
}
#[test]
fn strided_batch_runs_breaks_on_shape_and_stride_changes() {
    // A shape change ends a run; a non-constant stride within one shape also
    // ends it (the second/third jobs share a shape but not a common stride).
    let jobs = vec![
        batch_job((2, 2, 2), (0, 0, 0)),
        batch_job((2, 2, 2), (4, 4, 4)),
        batch_job((2, 2, 2), (100, 4, 4)),
    ];
    assert_eq!(strided_batch_runs(&jobs), vec![2, 1]);
}
#[cfg(feature = "tenferro")]
#[test]
fn identity_single_affine_run_is_one_strided_dispatch_for_two_and_four_jobs() {
    let mut executor = DefaultDenseExecutor::with_threads(1).unwrap();
    for len in [2usize, 4] {
        let shapes = vec![(2, 3, 2); len];
        check_identity_batch::<f64>(&mut executor, &shapes, 0, (0, 0, 0), None, 1.25, -0.5, 1);
        check_identity_batch::<f32>(&mut executor, &shapes, 0, (1, 2, 3), None, 1.0, 0.0, 1);
        check_identity_batch::<Complex64>(
            &mut executor,
            &shapes,
            1,
            (2, 1, 3),
            None,
            c64(0.75, -0.5),
            c64(-0.25, 0.125),
            1,
        );
        check_identity_batch::<Complex32>(
            &mut executor,
            &shapes,
            0,
            (0, 0, 0),
            None,
            Complex32::new(0.5, 0.5),
            Complex32::new(0.0, 0.0),
            1,
        );
    }
}
#[cfg(feature = "tenferro")]
#[test]
fn identity_long_run_with_residual_jobs_is_one_grouped_dispatch() {
    let mut executor = DefaultDenseExecutor::with_threads(1).unwrap();
    // 7 + singleton.
    let mut shapes = vec![(2, 2, 3); 7];
    shapes.push((3, 1, 1));
    assert_eq!(
        strided_batch_runs(&identity_fixture::<f64>(&shapes, 0, (0, 0, 0), 0.0).jobs),
        [7, 1]
    );
    check_identity_batch::<f64>(&mut executor, &shapes, 0, (0, 0, 0), None, 1.0, 0.0, 1);
    check_identity_batch::<Complex64>(
        &mut executor,
        &shapes,
        0,
        (1, 1, 1),
        None,
        c64(0.5, 1.5),
        c64(-1.0, 0.25),
        1,
    );
    // 7 + short run of 2.
    shapes.push((3, 1, 1));
    assert_eq!(
        strided_batch_runs(&identity_fixture::<f64>(&shapes, 0, (0, 0, 0), 0.0).jobs),
        [7, 2]
    );
    check_identity_batch::<f64>(&mut executor, &shapes, 0, (0, 0, 0), None, -2.0, 0.5, 1);
    check_identity_batch::<Complex64>(
        &mut executor,
        &shapes,
        0,
        (0, 0, 0),
        None,
        c64(0.0, 1.0),
        c64(0.0, 0.0),
        1,
    );
}
#[cfg(feature = "tenferro")]
#[test]
fn identity_heterogeneous_batch_with_gaps_and_base_offsets_is_one_dispatch() {
    let mut executor = DefaultDenseExecutor::with_threads(1).unwrap();
    let shapes = [(2, 3, 1), (1, 1, 4), (3, 2, 2), (2, 2, 2), (1, 5, 1)];
    let jobs = identity_fixture::<f64>(&shapes, 2, (0, 0, 0), 0.0).jobs;
    assert_eq!(strided_batch_runs(&jobs), [1; 5]);
    check_identity_batch::<f64>(&mut executor, &shapes, 2, (3, 5, 7), None, 1.5, -1.0, 1);
    check_identity_batch::<Complex64>(
        &mut executor,
        &shapes,
        3,
        (4, 0, 2),
        None,
        c64(-0.5, 0.75),
        c64(0.25, -0.5),
        1,
    );
    check_identity_batch::<f32>(&mut executor, &shapes, 1, (0, 1, 0), None, 2.0, 0.0, 1);
}
#[cfg(feature = "tenferro")]
#[test]
fn identity_beta_zero_overwrites_sentinel_on_both_paths() {
    let mut executor = DefaultDenseExecutor::with_threads(1).unwrap();
    let sentinel = c64(-99.0, 99.0);
    let alpha = c64(0.5, -1.25);
    for (shapes, gap) in [
        (vec![(2, 2, 2); 3], 0usize),
        (vec![(2, 2, 2), (1, 3, 2)], 0),
    ] {
        let mut fixture = identity_fixture::<Complex64>(&shapes, gap, (1, 0, 2), sentinel);
        let runs = strided_batch_runs(&fixture.jobs);
        let expected = identity_batch_oracle(
            &fixture.jobs,
            &fixture.lhs,
            &fixture.rhs,
            &fixture.output,
            fixture.base,
            alpha,
            c64(0.0, 0.0),
        );
        run_identity_batch(&mut executor, &mut fixture, &runs, alpha, c64(0.0, 0.0)).unwrap();
        assert_eq!(executor.seam_dispatches(), 1);
        for job in &fixture.jobs {
            let start = fixture.base.2 + job.dst_offset;
            for index in start..start + job.rows * job.cols {
                assert_ne!(fixture.output[index], sentinel);
            }
        }
        for (actual, expected) in fixture.output.iter().zip(&expected) {
            assert_c64_close(*actual, *expected, 1.0e-10);
        }
    }
}
#[cfg(feature = "tenferro")]
#[test]
fn identity_overlapping_destinations_are_rejected_by_the_grouped_validator_without_writes() {
    let mut executor = DefaultDenseExecutor::with_threads(1).unwrap();
    // One affine run whose destination step (2) is below rows * cols (4): the
    // strided view would alias itself, so admission fails and the grouped
    // validator reports the overlap before writing anything.
    let jobs = (0..3)
        .map(|batch| batch_job((2, 1, 2), (2 * batch, 2 * batch, 2 * batch)))
        .collect::<Vec<_>>();
    assert_eq!(strided_batch_runs(&jobs), [3]);
    let lhs = (0..8).map(|i| 1.0 + i as f64).collect::<Vec<_>>();
    let rhs = lhs.clone();
    let mut output = vec![-7.0; 10];
    let strides = [1];
    executor.reset_seam_dispatches();
    let error = executor
        .matmul_batch_axpby_into(
            DenseWrite::F64(DenseViewMut::new(&mut output, &[10], &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&lhs, &[8], &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&rhs, &[8], &strides, 0).unwrap()),
            &jobs,
            &[3],
            DenseScalar::F64(1.0),
            DenseScalar::F64(0.0),
        )
        .unwrap_err();
    assert!(
        matches!(&error, DenseError::Backend { op: "grouped_gemm", message, .. } if message.contains("overlap")),
        "{error:?}"
    );
    assert_eq!(executor.seam_dispatches(), 1);
    assert_eq!(output, vec![-7.0; 10]);

    // Two heterogeneous jobs whose destination ranges 0..4 and 2..5 overlap.
    let jobs = [
        batch_job((2, 1, 2), (0, 0, 0)),
        batch_job((1, 1, 3), (2, 2, 2)),
    ];
    assert_eq!(strided_batch_runs(&jobs), [1, 1]);
    let lhs = (0..6).map(|i| c64(i as f64, 1.0)).collect::<Vec<_>>();
    let mut output = vec![c64(3.0, -3.0); 6];
    executor.reset_seam_dispatches();
    let error = executor
        .matmul_batch_axpby_into(
            DenseWrite::C64(DenseViewMut::new(&mut output, &[6], &strides, 0).unwrap()),
            DenseRead::C64(DenseView::new(&lhs, &[6], &strides, 0).unwrap()),
            DenseRead::C64(DenseView::new(&lhs, &[6], &strides, 0).unwrap()),
            &jobs,
            &[1, 1],
            DenseScalar::C64(c64(1.0, 0.0)),
            DenseScalar::C64(c64(0.0, 0.0)),
        )
        .unwrap_err();
    assert!(
        matches!(&error, DenseError::Backend { op: "grouped_gemm", message, .. } if message.contains("overlap")),
        "{error:?}"
    );
    assert_eq!(executor.seam_dispatches(), 1);
    assert_eq!(output, vec![c64(3.0, -3.0); 6]);
}
#[cfg(feature = "tenferro")]
#[test]
fn identity_empty_and_zero_dimension_batches_on_both_paths() {
    let mut executor = DefaultDenseExecutor::with_threads(1).unwrap();
    let strides = [1];
    let mut output = [3.0, 4.0];
    executor.reset_seam_dispatches();
    executor
        .matmul_batch_axpby_into(
            DenseWrite::F64(DenseViewMut::new(&mut output, &[2], &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&[1.0], &[1], &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&[2.0], &[1], &strides, 0).unwrap()),
            &[],
            &[],
            DenseScalar::F64(1.0),
            DenseScalar::F64(0.0),
        )
        .unwrap();
    assert_eq!(executor.seam_dispatches(), 0);
    assert_eq!(output, [3.0, 4.0]);

    // rows = 0 and cols = 0 own no destination elements; contracted = 0 scales
    // the destination by beta. Gap 1 keeps the zero-size blocks at distinct
    // offsets so the dispatch decision is exercised, not degenerate.
    let alpha = c64(2.0, -1.0);
    let beta = c64(-0.5, 0.25);
    for shapes in [
        vec![(0, 2, 3); 2],
        vec![(2, 3, 0); 3],
        vec![(2, 0, 2); 2],
        vec![(2, 0, 2), (1, 0, 3)],
        vec![(0, 1, 1), (2, 2, 2)],
    ] {
        check_identity_batch::<Complex64>(
            &mut executor,
            &shapes,
            1,
            (1, 1, 1),
            None,
            alpha,
            beta,
            1,
        );
        check_identity_batch::<f64>(&mut executor, &shapes, 1, (0, 0, 0), None, 1.5, 0.5, 1);
    }
}
#[cfg(feature = "tenferro")]
#[test]
fn identity_executor_replays_strided_grouped_strided_across_shapes() {
    let mut executor = DefaultDenseExecutor::with_threads(1).unwrap();
    let alpha = c64(1.0, 0.5);
    let beta = c64(0.25, 0.0);
    // strided (one affine run) -> grouped (heterogeneous) -> strided with a
    // different shape and step, all through cache slot 0 of one executor.
    check_identity_batch::<Complex64>(
        &mut executor,
        &[(2, 2, 2); 3],
        0,
        (0, 0, 0),
        None,
        alpha,
        beta,
        1,
    );
    check_identity_batch::<Complex64>(
        &mut executor,
        &[(2, 2, 2), (3, 1, 2)],
        0,
        (0, 0, 0),
        None,
        alpha,
        beta,
        1,
    );
    check_identity_batch::<Complex64>(
        &mut executor,
        &[(3, 1, 4); 2],
        2,
        (1, 0, 0),
        None,
        alpha,
        beta,
        1,
    );
    check_identity_batch::<f64>(
        &mut executor,
        &[(1, 3, 1); 5],
        0,
        (0, 0, 0),
        None,
        1.0,
        0.0,
        1,
    );
    check_identity_batch::<f64>(
        &mut executor,
        &[(1, 3, 1), (2, 1, 1)],
        0,
        (0, 0, 0),
        None,
        1.0,
        0.0,
        1,
    );
    check_identity_batch::<f64>(
        &mut executor,
        &[(2, 2, 1); 4],
        0,
        (0, 0, 0),
        None,
        1.0,
        0.0,
        1,
    );
}
#[cfg(feature = "tenferro")]
#[test]
fn identity_malformed_runs_route_grouped_over_all_jobs() {
    // `runs` is consulted only for the O(1) single-run test (#1182), so a
    // short, long, or zero-entry partition can neither skip trailing jobs (a
    // short sum) nor over-index the job slice (a long sum). Every job runs,
    // grouped, in one submission.
    let mut executor = DefaultDenseExecutor::with_threads(1).unwrap();
    let shapes = [(2, 2, 2); 3];
    for runs in [&[2usize][..], &[5], &[0, 3], &[1, 1, 1], &[]] {
        check_identity_batch::<f64>(
            &mut executor,
            &shapes,
            0,
            (0, 0, 0),
            Some(runs),
            1.0,
            0.0,
            1,
        );
        check_identity_batch::<Complex64>(
            &mut executor,
            &shapes,
            0,
            (2, 0, 1),
            Some(runs),
            c64(0.5, 0.5),
            c64(1.0, -1.0),
            1,
        );
    }
}
#[cfg(feature = "tenferro")]
#[test]
fn identity_grouped_out_of_range_span_fails_before_any_write() {
    let mut executor = DefaultDenseExecutor::with_threads(1).unwrap();
    let strides = [1];
    let values = (0..8).map(|i| 1.0 + i as f64).collect::<Vec<_>>();
    // Heterogeneous batch: the second job's lhs span 6..10 exceeds the 8-element
    // buffer; tenferro's grouped validator rejects the whole submission before
    // the first (valid) job writes.
    let jobs = [
        batch_job((2, 1, 2), (0, 0, 0)),
        batch_job((2, 2, 1), (4, 6, 4)),
    ];
    assert_eq!(strided_batch_runs(&jobs), [1, 1]);
    let mut output = vec![-1.0; 8];
    executor.reset_seam_dispatches();
    let error = executor
        .matmul_batch_axpby_into(
            DenseWrite::F64(DenseViewMut::new(&mut output, &[8], &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&values, &[8], &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&values, &[8], &strides, 0).unwrap()),
            &jobs,
            &[1, 1],
            DenseScalar::F64(1.0),
            DenseScalar::F64(0.0),
        )
        .unwrap_err();
    assert!(
        matches!(
            error,
            DenseError::Backend {
                op: "grouped_gemm",
                ..
            }
        ),
        "{error:?}"
    );
    assert_eq!(executor.seam_dispatches(), 1);
    assert_eq!(output, vec![-1.0; 8]);

    // Same for an rhs span past the end of a base-offset view.
    let jobs = [
        batch_job((1, 2, 1), (0, 0, 0)),
        batch_job((1, 1, 2), (1, 2, 7)),
    ];
    let mut output = vec![c64(-1.0, 1.0); 4];
    let complex = values.iter().map(|&v| c64(v, -v)).collect::<Vec<_>>();
    executor.reset_seam_dispatches();
    let error = executor
        .matmul_batch_axpby_into(
            DenseWrite::C64(DenseViewMut::new(&mut output, &[3], &strides, 1).unwrap()),
            DenseRead::C64(DenseView::new(&complex, &[7], &strides, 1).unwrap()),
            DenseRead::C64(DenseView::new(&complex, &[7], &strides, 1).unwrap()),
            &jobs,
            &[1, 1],
            DenseScalar::C64(c64(1.0, 0.0)),
            DenseScalar::C64(c64(0.0, 0.0)),
        )
        .unwrap_err();
    assert!(
        matches!(
            error,
            DenseError::Backend {
                op: "grouped_gemm",
                ..
            }
        ),
        "{error:?}"
    );
    assert_eq!(executor.seam_dispatches(), 1);
    assert_eq!(output, vec![c64(-1.0, 1.0); 4]);
}
#[cfg(feature = "tenferro")]
#[test]
fn identity_two_job_affine_run_is_admitted_to_the_strided_view() {
    // `seam_dispatches()` is 1 on both routes for a two-job run, and the old
    // cutoff also produced one (grouped) dispatch there, so the route is pinned
    // through the bounds-check contract instead: an lhs span past the buffer
    // fails symtn-side as `OutOfBounds` with 0 dispatches only on the strided
    // route; the grouped route reaches tenferro (`Backend{op:"grouped_gemm"}`,
    // 1 dispatch). The mirror with dst step 2 < rows * cols = 4 must therefore
    // take the grouped route (the three-job variant is in
    // `identity_overlapping_destinations_are_rejected_by_the_grouped_validator_without_writes`).
    fn jobs(dst_step: usize) -> [DenseGemmBatchJob; 2] {
        [
            batch_job((2, 1, 2), (0, 0, 0)),
            batch_job((2, 1, 2), (dst_step, 2, 2)),
        ]
    }
    let strides = [1];
    let mut executor = DefaultDenseExecutor::with_threads(1).unwrap();

    // lhs holds 3 elements; the run's rank-3 lhs view needs index 3.
    let lhs_f64 = [1.0, 2.0, 3.0];
    let rhs_f64 = [4.0, 5.0, 6.0, 7.0];
    let strided = jobs(4);
    assert_eq!(strided_batch_runs(&strided), [2]);
    let mut output = [-3.0; 8];
    executor.reset_seam_dispatches();
    let error = executor
        .matmul_batch_axpby_into(
            DenseWrite::F64(DenseViewMut::new(&mut output, &[8], &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&lhs_f64, &[3], &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&rhs_f64, &[4], &strides, 0).unwrap()),
            &strided,
            &[2],
            DenseScalar::F64(1.0),
            DenseScalar::F64(0.0),
        )
        .unwrap_err();
    assert_eq!(error, DenseError::OutOfBounds);
    assert_eq!(executor.seam_dispatches(), 0);
    assert_eq!(output, [-3.0; 8]);

    let grouped = jobs(2);
    assert_eq!(strided_batch_runs(&grouped), [2]);
    let mut output = [-3.0; 8];
    executor.reset_seam_dispatches();
    let error = executor
        .matmul_batch_axpby_into(
            DenseWrite::F64(DenseViewMut::new(&mut output, &[8], &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&lhs_f64, &[3], &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&rhs_f64, &[4], &strides, 0).unwrap()),
            &grouped,
            &[2],
            DenseScalar::F64(1.0),
            DenseScalar::F64(0.0),
        )
        .unwrap_err();
    assert!(
        matches!(
            error,
            DenseError::Backend {
                op: "grouped_gemm",
                ..
            }
        ),
        "{error:?}"
    );
    assert_eq!(executor.seam_dispatches(), 1);
    assert_eq!(output, [-3.0; 8]);

    let lhs_c64 = lhs_f64.map(|v| c64(v, -v));
    let rhs_c64 = rhs_f64.map(|v| c64(-v, 0.5 * v));
    let sentinel = c64(-3.0, 3.0);
    let mut output = [sentinel; 8];
    executor.reset_seam_dispatches();
    let error = executor
        .matmul_batch_axpby_into(
            DenseWrite::C64(DenseViewMut::new(&mut output, &[8], &strides, 0).unwrap()),
            DenseRead::C64(DenseView::new(&lhs_c64, &[3], &strides, 0).unwrap()),
            DenseRead::C64(DenseView::new(&rhs_c64, &[4], &strides, 0).unwrap()),
            &strided,
            &[2],
            DenseScalar::C64(c64(0.5, -0.5)),
            DenseScalar::C64(c64(1.0, 1.0)),
        )
        .unwrap_err();
    assert_eq!(error, DenseError::OutOfBounds);
    assert_eq!(executor.seam_dispatches(), 0);
    assert_eq!(output, [sentinel; 8]);

    let mut output = [sentinel; 8];
    executor.reset_seam_dispatches();
    let error = executor
        .matmul_batch_axpby_into(
            DenseWrite::C64(DenseViewMut::new(&mut output, &[8], &strides, 0).unwrap()),
            DenseRead::C64(DenseView::new(&lhs_c64, &[3], &strides, 0).unwrap()),
            DenseRead::C64(DenseView::new(&rhs_c64, &[4], &strides, 0).unwrap()),
            &grouped,
            &[2],
            DenseScalar::C64(c64(0.5, -0.5)),
            DenseScalar::C64(c64(1.0, 1.0)),
        )
        .unwrap_err();
    assert!(
        matches!(
            error,
            DenseError::Backend {
                op: "grouped_gemm",
                ..
            }
        ),
        "{error:?}"
    );
    assert_eq!(executor.seam_dispatches(), 1);
    assert_eq!(output, [sentinel; 8]);
}
#[cfg(feature = "tenferro")]
#[test]
fn identity_batch_with_malformed_single_run_falls_back_to_grouped() {
    // `runs == [2]` claims one run, but job 1 has a different row count, so the
    // two jobs are not one affine run (issue #1733). Job 1's destination step
    // (8) still clears job 0's block (2*4), which must not select the strided
    // route and compute job 1 with job 0's shape.
    let jobs = [
        DenseGemmBatchJob {
            dst_offset: 0,
            lhs_offset: 0,
            rhs_offset: 0,
            rows: 2,
            contracted: 3,
            cols: 4,
        },
        DenseGemmBatchJob {
            dst_offset: 8,
            lhs_offset: 6,
            rhs_offset: 12,
            rows: 3,
            contracted: 3,
            cols: 4,
        },
    ];
    let runs = vec![2];
    let lhs = (0..15).map(|i| 1.0 + i as f64).collect::<Vec<_>>();
    let rhs = (0..24).map(|i| 0.5 - 0.25 * i as f64).collect::<Vec<_>>();
    let mut output = vec![-99.0; 20];

    let mut executor = DefaultDenseExecutor::new();
    executor
        .matmul_batch_axpby_into(
            DenseWrite::F64(DenseViewMut::new(&mut output, &[20], &[1], 0).unwrap()),
            DenseRead::F64(DenseView::new(&lhs, &[15], &[1], 0).unwrap()),
            DenseRead::F64(DenseView::new(&rhs, &[24], &[1], 0).unwrap()),
            &jobs,
            &runs,
            DenseScalar::F64(1.0),
            DenseScalar::F64(0.0),
        )
        .unwrap();

    for job in &jobs {
        let (m, k, n) = (job.rows, job.contracted, job.cols);
        let expected = matmul_f64(
            &lhs[job.lhs_offset..job.lhs_offset + m * k],
            &rhs[job.rhs_offset..job.rhs_offset + k * n],
            m,
            k,
            n,
        );
        for (actual, expected) in output[job.dst_offset..job.dst_offset + m * n]
            .iter()
            .zip(expected)
        {
            assert_f64_close(*actual, expected, 1.0e-12);
        }
    }
    assert_eq!(executor.staged_grouped_jobs(), jobs.len());
}

#[cfg(feature = "tenferro")]
#[test]
fn identity_single_job_batch_takes_the_strided_entry() {
    // One job is one GEMM. It is computed at its offsets with the caller's
    // alpha/beta, through the strided entry rather than the grouped one, whose
    // validator allocates on every call (#2184). The `runs` hint is ignored.
    let job = DenseGemmBatchJob {
        dst_offset: 3,
        lhs_offset: 2,
        rhs_offset: 1,
        rows: 3,
        contracted: 2,
        cols: 2,
    };
    let lhs = (0..8).map(|i| 1.0 + i as f64).collect::<Vec<_>>();
    let rhs = (0..5).map(|i| 0.5 - 0.25 * i as f64).collect::<Vec<_>>();
    let initial = (0..10).map(|i| -1.0 + 0.5 * i as f64).collect::<Vec<_>>();
    let (alpha, beta) = (2.0, -0.5);
    for runs in [&[1usize][..], &[2]] {
        let mut output = initial.clone();
        let mut executor = DefaultDenseExecutor::new();
        executor.reset_seam_dispatches();
        executor
            .matmul_batch_axpby_into(
                DenseWrite::F64(DenseViewMut::new(&mut output, &[10], &[1], 0).unwrap()),
                DenseRead::F64(DenseView::new(&lhs, &[8], &[1], 0).unwrap()),
                DenseRead::F64(DenseView::new(&rhs, &[5], &[1], 0).unwrap()),
                &[job],
                runs,
                DenseScalar::F64(alpha),
                DenseScalar::F64(beta),
            )
            .unwrap();

        let product = matmul_f64(&lhs[2..8], &rhs[1..5], 3, 2, 2);
        for (index, actual) in output.iter().enumerate() {
            let expected = match index.checked_sub(3) {
                Some(local) if local < 6 => alpha * product[local] + beta * initial[index],
                _ => initial[index],
            };
            assert_f64_close(*actual, expected, 1.0e-12);
        }
        assert_eq!(executor.staged_grouped_jobs(), 0);
        assert_eq!(executor.seam_dispatches(), 1);
    }
}

/// Decomposition outputs one tensor short, to reach the arity checks of the
/// `*_into` / values-only trait defaults.
struct DropsLastOutput(DefaultDenseExecutor);
impl DenseExecutor for DropsLastOutput {
    fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        let mut out = self.0.svd(input)?;
        out.pop();
        Ok(out)
    }

    fn qr(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        let mut out = self.0.qr(input)?;
        out.pop();
        Ok(out)
    }

    fn eigh(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        let mut out = self.0.eigh(input)?;
        out.pop();
        Ok(out)
    }

    fn eig(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        let mut out = self.0.eig(input)?;
        out.pop();
        Ok(out)
    }

    fn dot_general_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        config: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        self.0.dot_general_into(output, lhs, rhs, config)
    }
}

#[test]
fn default_decomposition_arity_disagreement_is_shape_mismatch() {
    let data = [2.0_f64, 1.0, 1.0, 3.0];
    let (shape, strides) = ([2, 2], [1, 2]);
    let read = || DenseRead::F64(DenseView::new(&data, &shape, &strides, 0).unwrap());
    let mut out = [[0.0_f64; 4]; 3];
    let [o0, o1, o2] = &mut out;
    fn write<'a>(buf: &'a mut [f64; 4]) -> DenseWrite<'a> {
        DenseWrite::F64(DenseViewMut::new(buf, &[2, 2], &[1, 2], 0).unwrap())
    }
    let mut executor = DropsLastOutput(DefaultDenseExecutor::new());

    let cases: [(&str, usize, usize, Result<(), DenseError>); 6] = [
        ("svd_into", 3, 2, {
            let (a, b, c) = (write(o0), write(o1), write(o2));
            executor.svd_into(read(), a, b, c)
        }),
        ("qr_into", 2, 1, {
            let (a, b) = (write(o0), write(o1));
            executor.qr_into(read(), a, b)
        }),
        ("eigh_into", 2, 1, {
            let (a, b) = (write(o0), write(o1));
            executor.eigh_into(read(), a, b)
        }),
        ("svd_vals", 3, 2, executor.svd_vals(read()).map(drop)),
        ("eigh_vals", 2, 1, executor.eigh_vals(read()).map(drop)),
        ("eig_vals", 2, 1, executor.eig_vals(read()).map(drop)),
    ];
    for (op, expected, actual, result) in cases {
        assert_eq!(
            result.unwrap_err(),
            DenseError::ShapeMismatch {
                op,
                expected: vec![expected],
                actual: vec![actual],
            }
        );
    }
}

#[test]
fn default_copy_into_reports_dtype_and_shape_mismatch() {
    let data = [2.0_f64, 1.0, 1.0, 3.0];
    let (shape, strides) = ([2, 2], [1, 2]);
    let read = || DenseRead::F64(DenseView::new(&data, &shape, &strides, 0).unwrap());
    let mut executor = DefaultDenseExecutor::new();

    // QR through the trait default: integer destination for Q.
    let mut qi = [0_i32; 4];
    let mut r = [0.0_f64; 4];
    let error = executor
        .qr_into(
            read(),
            DenseWrite::I32(DenseViewMut::new(&mut qi, &shape, &strides, 0).unwrap()),
            DenseWrite::F64(DenseViewMut::new(&mut r, &shape, &strides, 0).unwrap()),
        )
        .unwrap_err();
    assert_eq!(
        error,
        DenseError::DTypeMismatch {
            op: "qr_into",
            expected: DenseDType::F64,
            actual: DenseDType::I32,
        }
    );

    // A float destination of another precision is the same own dtype check,
    // ahead of the Tenferro slice accessor (#1765), never a backend error.
    let mut qf = [0.0_f32; 4];
    let error = executor
        .qr_into(
            read(),
            DenseWrite::F32(DenseViewMut::new(&mut qf, &shape, &strides, 0).unwrap()),
            DenseWrite::F64(DenseViewMut::new(&mut r, &shape, &strides, 0).unwrap()),
        )
        .unwrap_err();
    assert_eq!(
        error,
        DenseError::DTypeMismatch {
            op: "qr_into",
            expected: DenseDType::F64,
            actual: DenseDType::F32,
        }
    );

    // Wrong destination shape for the same tensor.
    let mut q = [0.0_f64; 2];
    let error = executor
        .qr_into(
            read(),
            DenseWrite::F64(DenseViewMut::new(&mut q, &[2, 1], &[1, 2], 0).unwrap()),
            DenseWrite::F64(DenseViewMut::new(&mut r, &shape, &strides, 0).unwrap()),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        DenseError::ShapeMismatch { op: "qr_into", ref expected, ref actual }
            if expected == &[2, 1] && actual == &[2, 2]
    ));
}

#[test]
fn batched_matmul_mismatch_and_ops_gap_are_typed() {
    let (a, b) = ([1.0_f64], [1.0_f32]);
    let mut out = [0.0_f64];
    let (shape, strides) = ([1, 1], [1, 1]);
    fn batch_args<'a>(out: &'a mut [f64; 1], a: &'a [f64; 1]) -> (DenseWrite<'a>, DenseRead<'a>) {
        (
            DenseWrite::F64(DenseViewMut::new(out, &[1, 1], &[1, 1], 0).unwrap()),
            DenseRead::F64(DenseView::new(a, &[1, 1], &[1, 1], 0).unwrap()),
        )
    }
    let (alpha, beta) = (DenseScalar::F64(1.0), DenseScalar::F64(0.0));
    let rhs_f32 = || DenseRead::F32(DenseView::new(&b, &shape, &strides, 0).unwrap());
    let expected = DenseError::DTypeMismatch {
        op: "matmul_batch_axpby_into",
        expected: DenseDType::F64,
        actual: DenseDType::F32,
    };

    // Trait default and the Tenferro adapter share one taxonomy.
    let mut no_axpby = NoAxpby::default();
    let (o, l) = batch_args(&mut out, &a);
    let error = no_axpby
        .matmul_batch_axpby_into(o, l, rhs_f32(), &[], &[], alpha, beta)
        .unwrap_err();
    assert_eq!(error, expected);

    let mut default = DefaultDenseExecutor::new();
    let (o, l) = batch_args(&mut out, &a);
    let error = default
        .matmul_batch_axpby_into(o, l, rhs_f32(), &[], &[], alpha, beta)
        .unwrap_err();
    assert_eq!(error, expected);

    let (o, l) = batch_args(&mut out, &a);
    let error = no_axpby
        .matmul_batch_axpby_with_ops_into(
            o,
            l,
            DenseRead::F64(DenseView::new(&a, &shape, &strides, 0).unwrap()),
            &[],
            &[],
            MatrixOp::Adjoint,
            MatrixOp::Identity,
            alpha,
            beta,
        )
        .unwrap_err();
    assert!(matches!(
        error,
        DenseError::Unsupported {
            op: "matmul_batch_axpby_with_ops_into",
            ..
        }
    ));
}

#[test]
fn copy_into_storage_length_disagreement_is_shape_mismatch() {
    let mut dst = [0.0_f64; 2];
    let view = DenseViewMut::new(&mut dst, &[2], &[1], 0).unwrap();
    let error =
        crate::executor::copy_contiguous_tensor_into_view(&[1.0_f64], &[2], view, "copy_probe")
            .unwrap_err();
    assert_eq!(
        error,
        DenseError::ShapeMismatch {
            op: "copy_probe",
            expected: vec![2],
            actual: vec![1],
        }
    );
}

#[test]
fn batched_matmul_with_ops_dtype_mismatch_and_integer_operands_are_typed() {
    let (a, b, i) = ([1.0_f64], [1.0_f32], [1_i32]);
    let mut out = [0.0_f64];
    let mut out_i = [0_i32];
    let (shape, strides) = ([1, 1], [1, 1]);
    let (alpha, beta) = (DenseScalar::F64(1.0), DenseScalar::F64(0.0));

    let mut default = DefaultDenseExecutor::new();
    let error = default
        .matmul_batch_axpby_with_ops_into(
            DenseWrite::F64(DenseViewMut::new(&mut out, &shape, &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&a, &shape, &strides, 0).unwrap()),
            DenseRead::F32(DenseView::new(&b, &shape, &strides, 0).unwrap()),
            &[],
            &[],
            MatrixOp::Adjoint,
            MatrixOp::Identity,
            alpha,
            beta,
        )
        .unwrap_err();
    assert_eq!(
        error,
        DenseError::DTypeMismatch {
            op: "matmul_batch_axpby_with_ops_into",
            expected: DenseDType::F64,
            actual: DenseDType::F32,
        }
    );

    // All operands agree on a dtype the batched route does not cover.
    let error = NoAxpby::default()
        .matmul_batch_axpby_into(
            DenseWrite::I32(DenseViewMut::new(&mut out_i, &shape, &strides, 0).unwrap()),
            DenseRead::I32(DenseView::new(&i, &shape, &strides, 0).unwrap()),
            DenseRead::I32(DenseView::new(&i, &shape, &strides, 0).unwrap()),
            &[],
            &[],
            alpha,
            beta,
        )
        .unwrap_err();
    assert!(matches!(
        error,
        DenseError::Unsupported {
            op: "matmul_batch_axpby_into",
            ..
        }
    ));
}
#[cfg(feature = "tenferro")]
#[test]
fn identity_grouped_batch_on_a_multi_thread_engine_matches_serial_bitwise() {
    // Large enough heterogeneous jobs that the grouped plan's Auto width can
    // fan them out over lanes. Every destination is written by one job on a
    // serial item context, so the result is bitwise the serial engine's.
    let shapes = (0..12)
        .map(|i| (24 + i % 3, 20 + i % 5, 22 + i % 4))
        .collect::<Vec<_>>();
    let (alpha, beta) = (c64(0.5, -1.0), c64(0.25, 0.5));
    let run = |threads: usize| {
        let mut executor = DefaultDenseExecutor::with_threads(threads).unwrap();
        let mut fixture = identity_fixture::<Complex64>(&shapes, 1, (2, 1, 3), c64(3.0, -4.0));
        let runs = strided_batch_runs(&fixture.jobs);
        run_identity_batch(&mut executor, &mut fixture, &runs, alpha, beta).unwrap();
        assert_eq!(executor.staged_grouped_jobs(), shapes.len());
        fixture.output
    };
    let serial = run(1);
    assert_eq!(run(4), serial);
    assert_eq!(run(8), serial);
}
#[cfg(feature = "tenferro")]
#[test]
fn identity_grouped_first_error_is_the_lowest_job_index() {
    // Job 0 writes after job 1 in output order; both are out of range, and the
    // reported job is 0 (job-index order), before any write.
    let jobs = [
        batch_job((1, 2, 1), (4, 7, 0)),
        batch_job((1, 1, 2), (0, 0, 7)),
    ];
    let values = [1.0; 8];
    let mut output = [-1.0; 8];
    let mut executor = DefaultDenseExecutor::with_threads(1).unwrap();
    let error = executor
        .matmul_batch_axpby_into(
            DenseWrite::F64(DenseViewMut::new(&mut output, &[8], &[1], 0).unwrap()),
            DenseRead::F64(DenseView::new(&values, &[8], &[1], 0).unwrap()),
            DenseRead::F64(DenseView::new(&values, &[8], &[1], 0).unwrap()),
            &jobs,
            &[1, 1],
            DenseScalar::F64(1.0),
            DenseScalar::F64(0.0),
        )
        .unwrap_err();
    assert!(
        matches!(&error, DenseError::Backend { op: "grouped_gemm", message, .. } if message.starts_with("job 0 lhs")),
        "{error:?}"
    );
    assert_eq!(output, [-1.0; 8]);
}
