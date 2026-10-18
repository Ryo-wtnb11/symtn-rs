//! S1a evidence: `DefaultDenseExecutor` opens one Tenferro CPU session per
//! dense phase, and the phase's results are unchanged.
//!
//! These live in their own test binary on purpose: `cpu_session_stats` is a
//! process-wide counter, so a delta is only meaningful when no unrelated test
//! opens a session in the same process. Inside this binary the mutex below
//! serializes the readers.

#![cfg(all(feature = "tenferro", not(feature = "provider-inject")))]

use num_complex::{Complex32, Complex64};
use symtn_dense::{
    cpu_session_stats, reset_cpu_session_stats, strided_batch_runs, CpuSessionStats,
    DefaultDenseExecutor, DenseError, DenseExecutor, DenseFactorization, DenseGemmBatchJob,
    DenseRead, DenseScalar, DenseView, DenseViewMut, DenseWrite, MatrixOp,
};

#[path = "../../tests/support/numerics.rs"]
mod numerics;

fn assert_close(got: f64, want: f64, tol: f64) {
    assert!((got - want).abs() <= tol, "{got} != {want} (tol {tol})");
}

// Session-scope evidence. `cpu_session_stats` is a process-wide counter,
// so every test that reads a delta takes this lock; they must not run
// concurrently with one another (other tests never touch the counter).
static CPU_SESSION_COUNTER_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Held for the whole of each test below, not just around the measured call:
/// the reference loops open sessions too, and a concurrent test would show up
/// in another test's delta.
fn counter_lock() -> std::sync::MutexGuard<'static, ()> {
    CPU_SESSION_COUNTER_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn sessions_opened() -> u64 {
    cpu_session_stats().sessions_opened
}

// Heterogeneous jobs with a non-Identity operand op take the op-bearing serial
// route (`runs.len() == jobs.len()`). It must enter exactly one Tenferro
// session for the whole loop and land bitwise what the same jobs produce when
// issued one at a time through the public batch API. Bit equality is a
// determinism contract here, not an arithmetic one: the fixture pins the
// singleton runs, so both sides issue the identical single-threaded kernel
// per job and only the session boundary differs.
#[allow(clippy::too_many_arguments)]
fn serial_route_is_one_session_and_bitwise_equal<T, W, R>(
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
    let _guard = counter_lock();
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
    let mut executor = DefaultDenseExecutor::with_threads(1).unwrap();
    let before = sessions_opened();
    let result = {
        executor.matmul_batch_axpby_with_ops_into(
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
    };
    let sessions = sessions_opened() - before;
    result.unwrap();
    assert_eq!(sessions, 1, "the whole job loop must share one session");

    // Reference: the identical kernels issued one job per call.
    let mut per_job = initial.clone();
    let mut reference = DefaultDenseExecutor::with_threads(1).unwrap();
    for job in &jobs {
        reference
            .matmul_batch_axpby_with_ops_into(
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
    assert_eq!(batched, per_job, "session scope must not change any value");
    assert_ne!(batched, initial, "fixture must actually write");
}

#[test]
#[allow(clippy::redundant_closure)]
fn op_bearing_serial_batch_uses_one_session_f64() {
    serial_route_is_one_session_and_bitwise_equal(
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
fn op_bearing_serial_batch_uses_one_session_c64() {
    for (lhs_op, rhs_op) in [
        (MatrixOp::Adjoint, MatrixOp::Identity),
        (MatrixOp::Identity, MatrixOp::Adjoint),
        (MatrixOp::Adjoint, MatrixOp::Adjoint),
    ] {
        serial_route_is_one_session_and_bitwise_equal(
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

// An empty job list is a supported call on the op-bearing route
// (`runs.len() == jobs.len()` with both empty). A session is a process-wide
// critical section, so it must not be taken for an empty loop.
#[test]
fn op_bearing_empty_batch_opens_no_session() {
    let _guard = counter_lock();
    let strides = [1usize];
    let mut output = [3.0f64];
    let mut executor = DefaultDenseExecutor::with_threads(1).unwrap();

    let before = sessions_opened();
    executor
        .matmul_batch_axpby_with_ops_into(
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
    assert_eq!(sessions_opened() - before, 0);
    assert_eq!(output, [3.0]);
}

// A failure raised inside the session must surface the same typed error the
// per-job path produced, and the executor must stay usable: the session is
// released on the error path, so a following call can enter a new one.
#[test]
fn op_bearing_serial_batch_error_inside_session_keeps_executor_usable() {
    let _guard = counter_lock();
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
    let mut executor = DefaultDenseExecutor::with_threads(1).unwrap();

    let before = sessions_opened();
    let error = {
        executor
            .matmul_batch_axpby_with_ops_into(
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
    let sessions = sessions_opened() - before;
    assert_eq!(error, DenseError::OutOfBounds);
    assert_eq!(sessions, 1);

    // The executor still works after the failed scope.
    let mut again = [0.0f64; 4];
    executor
        .matmul_batch_axpby_with_ops_into(
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

// `eig_vals` runs its contiguity pre-pass and the values call in one session
// (it opened two before), with the spectrum unchanged.
#[test]
fn eig_vals_uses_one_session_for_prepass_and_values() {
    let _guard = counter_lock();
    let shape = [2usize, 2];
    // Non-contiguous source, so the pre-pass actually copies.
    let padded = [2.0f64, 1.0, 9.0, 1.0, 3.0, 9.0];
    let strides = [1usize, 3];
    let mut executor = DefaultDenseExecutor::new();

    let before = sessions_opened();
    let values = {
        executor
            .eig_vals(DenseRead::F64(
                DenseView::new(&padded, &shape, &strides, 0).unwrap(),
            ))
            .unwrap()
    };
    let sessions = sessions_opened() - before;
    assert_eq!(sessions, 1);

    // [[2,1],[1,3]] is symmetric: the eigenvalues are real and known.
    let values = values.as_c64_slice().unwrap();
    assert_eq!(values.len(), 2);
    let mut real = values.iter().map(|v| v.re).collect::<Vec<_>>();
    real.sort_by(f64::total_cmp);
    let expected = [(5.0 - 5.0f64.sqrt()) / 2.0, (5.0 + 5.0f64.sqrt()) / 2.0];
    for (got, want) in real.iter().zip(expected) {
        assert_close(*got, want, 1e-12);
    }
    for value in values {
        assert_close(value.im, 0.0, 1e-12);
    }
}

#[test]
fn eig_vals_uses_one_session_for_c64() {
    let _guard = counter_lock();
    let c = Complex64::new;
    let shape = [2usize, 2];
    let strides = [1usize, 2];
    // [[0,-i],[i,0]] — Hermitian, eigenvalues -1 and +1.
    let m = [c(0.0, 0.0), c(0.0, 1.0), c(0.0, -1.0), c(0.0, 0.0)];
    let mut executor = DefaultDenseExecutor::new();

    let before = sessions_opened();
    let values = {
        executor
            .eig_vals(DenseRead::C64(
                DenseView::new(&m, &shape, &strides, 0).unwrap(),
            ))
            .unwrap()
    };
    let sessions = sessions_opened() - before;
    assert_eq!(sessions, 1);

    let values = values.as_c64_slice().unwrap();
    assert_eq!(values.len(), 2);
    let mut real = values.iter().map(|v| v.re).collect::<Vec<_>>();
    real.sort_by(f64::total_cmp);
    assert_close(real[0], -1.0, 1e-12);
    assert_close(real[1], 1.0, 1e-12);
}

// The counter is documented as a lower bound: `reset` zeroes it, and the
// sessions Tenferro opens internally for a plain backend-level dot are not
// visible here.
#[test]
fn cpu_session_stats_reset_and_scope() {
    let _guard = counter_lock();
    reset_cpu_session_stats();
    assert_eq!(cpu_session_stats(), CpuSessionStats::default());

    let data = [1.0f64, 2.0, 3.0, 4.0];
    let mut output = [0.0f64; 4];
    let strides = [1usize, 2];
    let mut executor = DefaultDenseExecutor::with_threads(1).unwrap();
    executor
        .matmul_into(
            DenseWrite::F64(DenseViewMut::new(&mut output, &[2, 2], &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&data, &[2, 2], &strides, 0).unwrap()),
            DenseRead::F64(DenseView::new(&data, &[2, 2], &strides, 0).unwrap()),
        )
        .unwrap();
    assert_eq!(cpu_session_stats().sessions_opened, 0);
    reset_cpu_session_stats();
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

// A coupled-sector factorization loop is one `factorize_batch`: it must enter
// one session for every matrix, and return what the per-matrix entries return
// (path agreement under the tolerance rule, see `assert_factor_agrees`).
fn factorize_batch_is_one_session_and_matches_per_call<T: Copy>(
    value: impl Fn(usize) -> T,
    wrap: impl for<'x> Fn(DenseView<'x, T>) -> DenseRead<'x>,
) {
    let _guard = counter_lock();
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

    for op in [DenseFactorization::Qr, DenseFactorization::Svd] {
        let mut executor = DefaultDenseExecutor::new();
        let before = sessions_opened();
        let batched = executor.factorize_batch(op, &inputs).unwrap();
        assert_eq!(
            sessions_opened() - before,
            1,
            "{op:?}: one session per batch"
        );

        let mut reference = DefaultDenseExecutor::new();
        let before = sessions_opened();
        let per_call = inputs
            .iter()
            .map(|&input| match op {
                DenseFactorization::Qr => reference.qr(input),
                DenseFactorization::Svd => reference.svd(input),
                _ => unreachable!("fixture exercises QR and SVD only"),
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(
            sessions_opened() - before,
            inputs.len() as u64,
            "{op:?}: the per-call entry opens one session per matrix"
        );
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
fn factorize_batch_is_one_session_and_matches_per_call_f64() {
    factorize_batch_is_one_session_and_matches_per_call(
        |i| ((i * 7 + 3) % 11) as f64 - 4.5,
        // Constructor functions fix one lifetime; the closure stays generic.
        |view| DenseRead::F64(view),
    );
}

#[test]
#[allow(clippy::redundant_closure)]
fn factorize_batch_is_one_session_and_matches_per_call_c64() {
    factorize_batch_is_one_session_and_matches_per_call(
        |i| {
            Complex64::new(
                ((i * 7 + 3) % 11) as f64 - 4.5,
                ((i * 5 + 1) % 7) as f64 - 3.0,
            )
        },
        |view| DenseRead::C64(view),
    );
}

// An empty batch is a supported call and must not take the process-wide
// execution permit for an empty loop.
#[test]
fn factorize_batch_empty_opens_no_session() {
    let _guard = counter_lock();
    let mut executor = DefaultDenseExecutor::new();
    let before = sessions_opened();
    let outputs = executor
        .factorize_batch(DenseFactorization::Qr, &[])
        .unwrap();
    assert!(outputs.is_empty());
    assert_eq!(sessions_opened() - before, 0);
}

/// Runs `jobs` with `runs` as one op-bearing c64 call on `dst` and returns the
/// (sessions, admissions) it took.
fn partition_call(
    executor: &mut DefaultDenseExecutor,
    (lhs, rhs): (&[Complex64], &[Complex64]),
    dst: &mut [Complex64],
    jobs: &[DenseGemmBatchJob],
    runs: &[usize],
) -> (u64, u64) {
    let strides = [1usize];
    let dst_len = dst.len();
    let before = cpu_session_stats();
    executor
        .matmul_batch_axpby_with_ops_into(
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
    let after = cpu_session_stats();
    (
        after.sessions_opened - before.sessions_opened,
        after.admissions - before.admissions,
    )
}

// #1291: a mixed run partition is one dense phase. Adjacent non-batchable runs
// share one serial session, and the whole phase — strided runs included —
// pays one execution admission. Each GEMM keeps its kernel, order, alpha/beta
// and conjugation, so the output is bitwise what one call per run produces.
#[test]
fn mixed_partition_is_one_admission_and_bitwise_equal_to_per_run_calls() {
    let _guard = counter_lock();
    // (partition, serial stretches, runs issued one call each: admissions)
    let cases: [(&[usize], u64, u64); 4] = [
        (&[4, 1, 1, 1], 1, 4),
        (&[1, 4, 1, 4], 2, 4),
        (&[1, 1, 3, 1, 2], 2, 5),
        // One run: no scope, its single strided call admits itself.
        (&[4], 0, 1),
    ];
    for threads in [Some(1), None] {
        for (pattern, stretches, per_run_admissions) in cases {
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
            let executor = || match threads {
                Some(threads) => DefaultDenseExecutor::with_threads(threads).unwrap(),
                None => DefaultDenseExecutor::new(),
            };
            let label = format!("{pattern:?} threads={threads:?}");

            let mut whole = initial.clone();
            let counts = partition_call(&mut executor(), (&lhs, &rhs), &mut whole, &jobs, pattern);
            assert_eq!(counts, (stretches, 1), "{label}: (sessions, admissions)");

            let mut per_run = initial.clone();
            let mut reference = executor();
            let mut admissions = 0;
            let mut start = 0;
            for &len in pattern {
                admissions += partition_call(
                    &mut reference,
                    (&lhs, &rhs),
                    &mut per_run,
                    &jobs[start..start + len],
                    &[len],
                )
                .1;
                start += len;
            }
            assert_eq!(
                admissions, per_run_admissions,
                "{label}: per-run admissions"
            );
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
