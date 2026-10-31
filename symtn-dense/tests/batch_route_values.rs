//! Value contracts of the dense executor's batched routes, independent of
//! how the engine schedules them: a serial op-bearing batch lands bitwise
//! what one call per job lands, a failed batch leaves the executor usable,
//! `factorize_batch` returns the per-matrix factors, and a mixed run
//! partition lands bitwise what one call per run lands (on the same engine).

#![cfg(all(feature = "tenferro", not(feature = "provider-inject")))]

use num_complex::Complex64;
use symtn_dense::{
    strided_batch_runs, CpuEngine, DefaultDenseExecutor, DenseError, DenseExecutor,
    DenseFactorization, DenseGemmBatchJob, DenseRead, DenseScalar, DenseView, DenseViewMut,
    DenseWrite, Engine, MatrixOp, Pool,
};

use crate::numerics;
// Heterogeneous jobs with a non-Identity operand op take the op-bearing serial
// route (`runs.len() == jobs.len()`). It must land bitwise what the same jobs
// produce when issued one at a time through the public batch API. Bit
// equality is a determinism contract here, not an arithmetic one: the fixture
// pins the singleton runs, so both sides issue the identical single-threaded
// kernel per job.
#[allow(clippy::too_many_arguments)]
fn serial_route_is_bitwise_equal_to_per_job_calls<T, W, R>(
    values: impl Fn(usize) -> T,
    accumulator: impl Fn(usize) -> T,
    alpha: DenseScalar,
    beta: DenseScalar,
    lhs_op: MatrixOp,
    rhs_op: MatrixOp,
    wrap_write: W,
    wrap_read: R,
) where
    T: Copy + PartialEq + std::fmt::Debug + 'static,
    W: for<'x> Fn(DenseViewMut<'x, T>) -> DenseWrite<'x>,
    R: for<'x> Fn(DenseView<'x, T>) -> DenseRead<'x>,
{
    // Alternating shapes: no constant-stride run, so `strided_batch_runs`
    // yields singleton runs and the serial route is forced.
    let shapes = [(4usize, 3usize, 5usize), (5, 4, 3)];
    let mut jobs = Vec::new();
    let (mut lhs_offset, mut rhs_offset, mut dst_offset) = (0usize, 0usize, 0usize);
    for index in 0..8usize {
        let (rows, contracted, cols) = shapes[index % 2];
        jobs.push(DenseGemmBatchJob {
            rows,
            contracted,
            cols,
            lhs_offset,
            rhs_offset,
            dst_offset,
        });
        lhs_offset += rows * contracted + 3;
        rhs_offset += contracted * cols + 5;
        dst_offset += rows * cols + 7;
    }
    let runs = strided_batch_runs(&jobs);
    assert_eq!(
        runs,
        vec![1; jobs.len()],
        "fixture must force the serial route"
    );

    let lhs = (0..lhs_offset).map(&values).collect::<Vec<_>>();
    let rhs = (0..rhs_offset).map(|i| values(i + 11)).collect::<Vec<_>>();
    let initial = (0..dst_offset).map(&accumulator).collect::<Vec<_>>();
    let strides = [1usize];

    let mut batched = initial.clone();
    let engine = &mut CpuEngine::serial();
    let mut executor = DefaultDenseExecutor::new();
    executor
        .matmul_batch_axpby_with_ops_into(
            engine,
            wrap_write(DenseViewMut::new(&mut batched, &[initial.len()], &strides, 0).unwrap()),
            wrap_read(DenseView::new(&lhs, &[lhs.len()], &strides, 0).unwrap()),
            wrap_read(DenseView::new(&rhs, &[rhs.len()], &strides, 0).unwrap()),
            &jobs,
            &runs,
            lhs_op,
            rhs_op,
            alpha,
            beta,
        )
        .unwrap();

    // Reference: the identical kernels issued one job per call.
    let mut per_job = initial.clone();
    let mut reference = DefaultDenseExecutor::new();
    for job in &jobs {
        reference
            .matmul_batch_axpby_with_ops_into(
                engine,
                wrap_write(DenseViewMut::new(&mut per_job, &[initial.len()], &strides, 0).unwrap()),
                wrap_read(DenseView::new(&lhs, &[lhs.len()], &strides, 0).unwrap()),
                wrap_read(DenseView::new(&rhs, &[rhs.len()], &strides, 0).unwrap()),
                std::slice::from_ref(job),
                &[1],
                lhs_op,
                rhs_op,
                alpha,
                beta,
            )
            .unwrap();
    }
    assert_eq!(batched, per_job, "batching must not change any value");
    assert_ne!(batched, initial, "fixture must actually write");
}

#[test]
#[allow(clippy::redundant_closure)]
fn op_bearing_serial_batch_is_per_job_bitwise_f64() {
    serial_route_is_bitwise_equal_to_per_job_calls(
        |i| 0.5 + 0.25 * i as f64,
        |i| 2.0 + 0.05 * i as f64,
        DenseScalar::F64(0.75),
        DenseScalar::F64(-0.25),
        MatrixOp::Transpose,
        MatrixOp::Identity,
        // Constructor functions fix one lifetime; the closures stay generic.
        |view| DenseWrite::F64(view),
        |view| DenseRead::F64(view),
    );
}

// The three operand-op pairs the tensor layer actually emits
// (`fusion_block.rs:1718-1719`, `:1742-1743`, `:1848-1852`); `Transpose` is
// never produced there.
#[test]
#[allow(clippy::redundant_closure)]
fn op_bearing_serial_batch_is_per_job_bitwise_c64() {
    for (lhs_op, rhs_op) in [
        (MatrixOp::Adjoint, MatrixOp::Identity),
        (MatrixOp::Identity, MatrixOp::Adjoint),
        (MatrixOp::Adjoint, MatrixOp::Adjoint),
    ] {
        serial_route_is_bitwise_equal_to_per_job_calls(
            |i| Complex64::new(0.5 + 0.25 * i as f64, -0.75 + 0.125 * i as f64),
            |i| Complex64::new(2.0 + 0.05 * i as f64, -1.0 - 0.025 * i as f64),
            DenseScalar::C64(Complex64::new(0.75, -0.5)),
            DenseScalar::C64(Complex64::new(-0.25, 0.125)),
            lhs_op,
            rhs_op,
            // Constructor functions fix one lifetime; the closures stay generic.
            |view| DenseWrite::C64(view),
            |view| DenseRead::C64(view),
        );
    }
}

// A failure raised inside the batch must surface the same typed error the
// per-job path produced, and the executor and engine must stay usable.
#[test]
fn op_bearing_serial_batch_error_keeps_executor_usable() {
    let good = DenseGemmBatchJob {
        rows: 2,
        contracted: 2,
        cols: 2,
        lhs_offset: 0,
        rhs_offset: 0,
        dst_offset: 0,
    };
    // Second job reads past the end of the operand buffers.
    let bad = DenseGemmBatchJob {
        rhs_offset: usize::MAX,
        ..good
    };
    let data = [1.0f64, 2.0, 3.0, 4.0];
    let mut output = [0.0f64; 4];
    let strides = [1usize];
    let engine = &mut CpuEngine::serial();
    let mut executor = DefaultDenseExecutor::new();

    let error = {
        executor
            .matmul_batch_axpby_with_ops_into(
                engine,
                DenseWrite::F64(DenseViewMut::new(&mut output, &[4], &strides, 0).unwrap()),
                DenseRead::F64(DenseView::new(&data, &[4], &strides, 0).unwrap()),
                DenseRead::F64(DenseView::new(&data, &[4], &strides, 0).unwrap()),
                &[good, bad],
                &[1, 1],
                MatrixOp::Transpose,
                MatrixOp::Identity,
                DenseScalar::F64(1.0),
                DenseScalar::F64(0.0),
            )
            .unwrap_err()
    };
    // The BLAS serial route writes the good job, then builds the bad job's
    // view and reports it out of bounds; the native block kernel bounds
    // every job first, where the bad offset overflows, and writes nothing.
    let (expected, written) = if cfg!(feature = "cpu-blas-core") {
        (DenseError::OutOfBounds, [5.0, 11.0, 11.0, 25.0])
    } else {
        (DenseError::OffsetOverflow { value: usize::MAX }, [0.0; 4])
    };
    assert_eq!(error, expected);
    assert_eq!(output, written);

    // The executor still works after the failed batch.
    let mut again = [0.0f64; 4];
    executor
        .matmul_batch_axpby_with_ops_into(
            engine,
            DenseWrite::F64(DenseViewMut::new(&mut again, &[4], &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&data, &[4], &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&data, &[4], &strides, 0).unwrap()),
            &[good],
            &[1],
            MatrixOp::Transpose,
            MatrixOp::Identity,
            DenseScalar::F64(1.0),
            DenseScalar::F64(0.0),
        )
        .unwrap();
    // A^T A for the column-major [[1,3],[2,4]].
    assert_eq!(again, [5.0, 11.0, 11.0, 25.0]);
}

/// Asserts equal shapes and factor entries within the workspace tolerance.
///
/// Path agreement is the contract: `factorize_batch` must return the factors
/// (including their gauge) the per-matrix entry returns for the same input.
/// It is compared under the rule rather than bit for bit because the batch
/// route is free to order the same factorization's floating work differently;
/// `terms` is the longest inner product, the larger matrix dimension.
fn assert_factor_agrees(
    what: &str,
    got: &symtn_dense::DenseTensor,
    want: &symtn_dense::DenseTensor,
    terms: usize,
) {
    assert_eq!(got.shape(), want.shape(), "{what}: factor shapes differ");
    match (got.as_f64_slice(), want.as_f64_slice()) {
        (Ok(got), Ok(want)) => numerics::assert_slices_close(what, got, want, terms),
        _ => numerics::assert_slices_close(
            what,
            got.as_c64_slice().expect("fixture outputs are f64 or c64"),
            want.as_c64_slice().expect("fixture outputs are f64 or c64"),
            terms,
        ),
    }
}

// A coupled-sector factorization loop is one `factorize_batch`: it must
// return what the per-matrix entries return (path agreement under the
// tolerance rule, see `assert_factor_agrees`).
fn factorize_batch_matches_per_call<T: Copy>(
    value: impl Fn(usize) -> T,
    wrap: impl for<'x> Fn(DenseView<'x, T>) -> DenseRead<'x>,
) {
    let shapes = [(3usize, 2usize), (2, 4), (4, 4), (1, 3), (5, 2)];
    let data = shapes
        .iter()
        .enumerate()
        .map(|(block, &(rows, cols))| {
            (0..rows * cols)
                .map(|i| value(31 * block + i))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let layouts = shapes
        .iter()
        .map(|&(rows, cols)| ([rows, cols], [1usize, rows]))
        .collect::<Vec<_>>();
    let inputs = data
        .iter()
        .zip(&layouts)
        .map(|(block, (shape, strides))| wrap(DenseView::new(block, shape, strides, 0).unwrap()))
        .collect::<Vec<_>>();

    let engine = &mut CpuEngine::serial();
    for op in [DenseFactorization::Qr, DenseFactorization::Svd] {
        let mut executor = DefaultDenseExecutor::new();
        let batched = executor.factorize_batch(engine, op, &inputs).unwrap();

        let mut reference = DefaultDenseExecutor::new();
        let per_call = inputs
            .iter()
            .map(|&input| match op {
                DenseFactorization::Qr => reference.qr(engine, input),
                DenseFactorization::Svd => reference.svd(engine, input),
                _ => unreachable!("fixture exercises QR and SVD only"),
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(batched.len(), per_call.len());
        for ((got, want), &(rows, cols)) in batched.iter().zip(&per_call).zip(&shapes) {
            assert_eq!(got.len(), want.len(), "{op:?}: factor counts differ");
            for (got, want) in got.iter().zip(want) {
                assert_factor_agrees(&format!("{op:?}"), got, want, rows.max(cols));
            }
        }
    }
}

#[test]
#[allow(clippy::redundant_closure)]
fn factorize_batch_matches_per_call_f64() {
    factorize_batch_matches_per_call(
        |i| ((i * 7 + 3) % 11) as f64 - 4.5,
        // Constructor functions fix one lifetime; the closure stays generic.
        |view| DenseRead::F64(view),
    );
}

#[test]
#[allow(clippy::redundant_closure)]
fn factorize_batch_matches_per_call_c64() {
    factorize_batch_matches_per_call(
        |i| {
            Complex64::new(
                ((i * 7 + 3) % 11) as f64 - 4.5,
                ((i * 5 + 1) % 7) as f64 - 3.0,
            )
        },
        |view| DenseRead::C64(view),
    );
}

/// Runs `jobs` with `runs` as one op-bearing c64 call on `dst`.
fn partition_call(
    engine: &mut dyn Engine,
    executor: &mut DefaultDenseExecutor,
    (lhs, rhs): (&[Complex64], &[Complex64]),
    dst: &mut [Complex64],
    jobs: &[DenseGemmBatchJob],
    runs: &[usize],
) {
    let strides = [1usize];
    let dst_len = dst.len();
    executor
        .matmul_batch_axpby_with_ops_into(
            engine,
            DenseWrite::C64(DenseViewMut::new(dst, &[dst_len], &strides, 0).unwrap()),
            DenseRead::C64(DenseView::new(lhs, &[lhs.len()], &strides, 0).unwrap()),
            DenseRead::C64(DenseView::new(rhs, &[rhs.len()], &strides, 0).unwrap()),
            jobs,
            runs,
            MatrixOp::Adjoint,
            MatrixOp::Identity,
            DenseScalar::C64(Complex64::new(0.75, -0.5)),
            DenseScalar::C64(Complex64::new(-0.25, 0.125)),
        )
        .unwrap();
}

// #1291: a mixed run partition is one dense phase. Each GEMM keeps its
// kernel, order, alpha/beta and conjugation, so on one engine the output is
// bitwise what one call per run produces, serial or threaded.
#[test]
fn mixed_partition_is_bitwise_equal_to_per_run_calls() {
    let patterns: [&[usize]; 4] = [&[4, 1, 1, 1], &[1, 4, 1, 4], &[1, 1, 3, 1, 2], &[4]];
    for threads in [1usize, 4] {
        let mut engine = if threads == 1 {
            CpuEngine::serial()
        } else {
            CpuEngine::new(Pool::builder().threads(threads).build().unwrap())
        };
        for pattern in patterns {
            let mut jobs = Vec::new();
            let (mut l, mut r, mut d) = (0usize, 0usize, 0usize);
            for (index, &len) in pattern.iter().enumerate() {
                let (rows, contracted, cols) = if len >= 2 {
                    (3, 4, 2)
                } else {
                    (2 + index % 2, 3, 4)
                };
                for _ in 0..len {
                    jobs.push(DenseGemmBatchJob {
                        dst_offset: d,
                        lhs_offset: l,
                        rhs_offset: r,
                        rows,
                        contracted,
                        cols,
                    });
                    l += rows * contracted;
                    r += contracted * cols;
                    d += rows * cols;
                }
            }
            let value = |i: usize| {
                Complex64::new(
                    ((i * 7 + 3) % 11) as f64 * 0.25 - 1.0,
                    ((i * 5) % 7) as f64 * 0.125,
                )
            };
            let lhs = (0..l).map(value).collect::<Vec<_>>();
            let rhs = (0..r).map(|i| value(i + 5)).collect::<Vec<_>>();
            let initial = (0..d).map(|i| value(i + 9)).collect::<Vec<_>>();
            let label = format!("{pattern:?} threads={threads}");

            let mut whole = initial.clone();
            partition_call(
                &mut engine,
                &mut DefaultDenseExecutor::new(),
                (&lhs, &rhs),
                &mut whole,
                &jobs,
                pattern,
            );

            let mut per_run = initial.clone();
            let mut reference = DefaultDenseExecutor::new();
            let mut start = 0;
            for &len in pattern {
                partition_call(
                    &mut engine,
                    &mut reference,
                    (&lhs, &rhs),
                    &mut per_run,
                    &jobs[start..start + len],
                    &[len],
                );
                start += len;
            }
            let bits = |v: &[Complex64]| {
                v.iter()
                    .flat_map(|z| [z.re.to_bits(), z.im.to_bits()])
                    .collect::<Vec<_>>()
            };
            assert_eq!(bits(&whole), bits(&per_run), "{label}");
            assert_ne!(whole, initial, "{label}: fixture must write");
        }
    }
}

// A replay issues one single-job GEMM per block, block shapes differing. On a
// warm engine every such GEMM finds its plan: none is rebuilt because the
// previous block had another shape (tenferro-rs#2349).
#[test]
fn single_job_gemms_of_alternating_shapes_reuse_their_plans_when_warm() {
    let shapes = [(4usize, 3usize, 5usize), (5, 4, 3), (2, 6, 7)];
    let lhs = (0..64).map(|i| i as f64 * 0.5 - 3.0).collect::<Vec<_>>();
    let rhs = (0..64).map(|i| 1.0 - i as f64 * 0.25).collect::<Vec<_>>();
    let mut dst = vec![0.0; 64];
    let strides = [1usize];
    let engine = &mut CpuEngine::serial();
    let mut executor = DefaultDenseExecutor::new();
    let mut round = |engine: &mut CpuEngine, dst: &mut [f64]| {
        for (rows, contracted, cols) in shapes {
            let job = DenseGemmBatchJob {
                rows,
                contracted,
                cols,
                lhs_offset: 0,
                rhs_offset: 0,
                dst_offset: 0,
            };
            executor
                .matmul_batch_axpby_into(
                    engine,
                    DenseWrite::F64(DenseViewMut::new(dst, &[64], &strides, 0).unwrap()),
                    DenseRead::F64(DenseView::new(&lhs, &[64], &strides, 0).unwrap()),
                    DenseRead::F64(DenseView::new(&rhs, &[64], &strides, 0).unwrap()),
                    &[job],
                    &[1],
                    DenseScalar::F64(1.0),
                    DenseScalar::F64(0.0),
                )
                .unwrap();
        }
    };
    round(engine, &mut dst);
    let warm = engine.plan_cache().contraction_stats();
    round(engine, &mut dst);
    let after = engine.plan_cache().contraction_stats();
    assert_eq!(after.misses, warm.misses, "a warm round rebuilt a plan");
    assert_eq!(after.hits - warm.hits, shapes.len() as u64);
}
