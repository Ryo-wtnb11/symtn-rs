//! `DenseExecutor::matmul_batch_fresh`: every job's output is
//! `alpha * op(A) op(B)` in its own piece, reported once, without reading
//! the destination; the default executor keeps its initialized route choice.

use core::mem::MaybeUninit;
use core::ops::{Add, Mul};

use super::*;
use crate::fresh::FreshScalar;

use super::numerics;

/// A GEMM dtype with what the scalar oracle and the carrier need.
trait Val:
    FreshScalar + PartialEq + numerics::Numeric + Add<Output = Self> + Mul<Output = Self>
{
    fn new(re: f64, im: f64) -> Self;
    fn conj(self) -> Self;
    fn carrier<'s, 'a>(
        segments: &'s mut [(usize, &'a mut [MaybeUninit<Self>])],
        written: &'s mut dyn FnMut(usize, &'a mut [Self]),
    ) -> DenseFreshSegments<'s, 'a>;
}

/// The test callers' zero fill: safe, from a zero source of the piece's length.
fn zeroed<T: Copy + Default>(piece: &mut [MaybeUninit<T>]) -> &mut [T] {
    piece.write_copy_of_slice(&vec![T::default(); piece.len()])
}

macro_rules! val {
    ($ty:ty, $variant:ident, |$re:ident, $im:ident| $new:expr, |$v:ident| $conj:expr) => {
        impl Val for $ty {
            fn new($re: f64, $im: f64) -> Self {
                $new
            }
            fn conj(self) -> Self {
                let $v = self;
                $conj
            }
            fn carrier<'s, 'a>(
                segments: &'s mut [(usize, &'a mut [MaybeUninit<Self>])],
                written: &'s mut dyn FnMut(usize, &'a mut [Self]),
            ) -> DenseFreshSegments<'s, 'a> {
                DenseFreshSegments::$variant {
                    segments,
                    zeroed: zeroed::<$ty>,
                    written,
                }
            }
        }
    };
}
val!(f32, F32, |re, _im| re as f32, |v| v);
val!(f64, F64, |re, _im| re, |v| v);
val!(
    Complex32,
    C32,
    |re, im| Complex32::new(re as f32, im as f32),
    |v| num_complex::Complex::conj(&v)
);
val!(Complex64, C64, |re, im| Complex64::new(re, im), |v| {
    num_complex::Complex::conj(&v)
});

/// Distinct, exactly representable operand values.
fn values<T: Val>(len: usize, seed: usize) -> Vec<T> {
    (0..len)
        .map(|i| {
            let k = (i * 7 + seed * 3) % 13;
            T::new(k as f64 * 0.25 - 1.5, (k % 5) as f64 * 0.5 - 1.0)
        })
        .collect()
}

/// The scalar GEMM of one job: `alpha * sum_p op(A)[i, p] op(B)[p, j]`,
/// column-major; `alpha == 0` is exactly zero whatever the operands hold.
fn oracle<T: Val>(
    lhs: &[T],
    rhs: &[T],
    job: &DenseGemmBatchJob,
    ops: [MatrixOp; 2],
    alpha: T,
) -> Vec<T> {
    let (m, k, n) = (job.rows, job.contracted, job.cols);
    let entry = |data: &[T], offset: usize, rows: usize, cols: usize, r: usize, c: usize, op| {
        let value = match op {
            MatrixOp::Identity => data[offset + r + rows * c],
            MatrixOp::Transpose | MatrixOp::Adjoint => data[offset + c + cols * r],
        };
        if op == MatrixOp::Adjoint {
            value.conj()
        } else {
            value
        }
    };
    let mut out = vec![T::default(); m * n];
    if alpha == T::default() {
        return out;
    }
    for j in 0..n {
        for i in 0..m {
            let mut sum = T::default();
            for p in 0..k {
                sum = sum
                    + entry(lhs, job.lhs_offset, m, k, i, p, ops[0])
                        * entry(rhs, job.rhs_offset, k, n, p, j, ops[1]);
            }
            out[i + m * j] = alpha * sum;
        }
    }
    out
}

/// The maximal runs of adjacent job outputs: the segments an owner of the
/// destination hands one call.
fn segments_of(jobs: &[DenseGemmBatchJob]) -> Vec<(usize, usize)> {
    let mut ranges: Vec<(usize, usize)> = jobs
        .iter()
        .filter(|job| job.rows * job.cols != 0)
        .map(|job| (job.dst_offset, job.dst_offset + job.rows * job.cols))
        .collect();
    ranges.sort_unstable();
    let mut segments: Vec<(usize, usize)> = Vec::new();
    for (start, end) in ranges {
        match segments.last_mut() {
            Some(last) if last.1 == start => last.1 = end,
            _ => segments.push((start, end)),
        }
    }
    segments
}

/// Runs one fresh call over `buffer` cut into `segments` and returns each
/// job's reports. Every report is checked to be the job's own piece of
/// `buffer`.
#[allow(clippy::too_many_arguments)]
fn run_fresh<T: Val>(
    executor: &mut dyn DenseExecutor,
    engine: &mut dyn Engine,
    buffer: &mut [MaybeUninit<T>],
    segments: &[(usize, usize)],
    lhs: &[T],
    rhs: &[T],
    jobs: &[DenseGemmBatchJob],
    ops: [MatrixOp; 2],
    alpha: T,
) -> Result<Vec<Vec<T>>, DenseError> {
    let base = buffer.as_ptr().cast::<T>();
    let mut parts = Vec::new();
    let mut rest = buffer;
    let mut cursor = 0;
    for &(start, end) in segments {
        let (_, tail) = rest.split_at_mut(start - cursor);
        let (piece, tail) = tail.split_at_mut(end - start);
        parts.push((start, piece));
        rest = tail;
        cursor = end;
    }
    let mut reports: Vec<Vec<Vec<T>>> = vec![Vec::new(); jobs.len()];
    let mut written = |job: usize, output: &mut [T]| {
        let expected_len = jobs[job].rows * jobs[job].cols;
        assert_eq!(output.len(), expected_len, "job {job} report length");
        if expected_len != 0 {
            assert!(
                core::ptr::eq(output.as_ptr(), base.wrapping_add(jobs[job].dst_offset)),
                "job {job} reported memory other than its own piece"
            );
        }
        reports[job].push(output.to_vec());
    };
    let shape = |len: usize| [len];
    let (lhs_shape, rhs_shape, strides) = (shape(lhs.len()), shape(rhs.len()), [1]);
    let result = executor.matmul_batch_fresh(
        engine,
        T::carrier(&mut parts, &mut written),
        T::read(DenseView::new(lhs, &lhs_shape, &strides, 0)?),
        T::read(DenseView::new(rhs, &rhs_shape, &strides, 0)?),
        jobs,
        ops,
        T::scalar(alpha),
    );
    if let Err(error) = result {
        assert!(
            reports.iter().all(Vec::is_empty),
            "a failed call reported a job: {error}"
        );
        return Err(error);
    }
    Ok(reports
        .into_iter()
        .enumerate()
        .map(|(job, mut once)| {
            assert_eq!(once.len(), 1, "job {job} must be reported exactly once");
            once.pop().unwrap()
        })
        .collect())
}

/// The destination length a batch addresses.
fn destination_len(jobs: &[DenseGemmBatchJob]) -> usize {
    jobs.iter()
        .map(|job| job.dst_offset + job.rows * job.cols)
        .max()
        .unwrap_or(0)
}

fn check_values<T: Val>(
    reports: &[Vec<T>],
    lhs: &[T],
    rhs: &[T],
    jobs: &[DenseGemmBatchJob],
    ops: [MatrixOp; 2],
    alpha: T,
) {
    for (job, got) in jobs.iter().zip(reports) {
        let want = oracle(lhs, rhs, job, ops, alpha);
        assert_eq!(got.len(), want.len());
        for (got, want) in got.iter().zip(&want) {
            numerics::assert_close("fresh GEMM entry", *got, *want, job.contracted);
        }
    }
}

/// A pure-Rust executor whose only GEMM is a scalar loop, so the trait's
/// default fresh entry runs without any vendor or SIMD kernel.
struct ScalarLoop {
    /// Calls made so far, and the call that fails (by error or panic).
    calls: usize,
    fail_at: Option<(usize, Failure)>,
}

#[derive(Clone, Copy)]
enum Failure {
    Error,
    Panic,
}

impl ScalarLoop {
    fn new() -> Self {
        Self {
            calls: 0,
            fail_at: None,
        }
    }
}

impl DenseExecutor for ScalarLoop {
    fn svd(
        &mut self,
        _: &mut dyn Engine,
        _: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        unreachable!("not a factorization test")
    }
    fn qr(&mut self, _: &mut dyn Engine, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        unreachable!("not a factorization test")
    }
    fn eigh(
        &mut self,
        _: &mut dyn Engine,
        _: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        unreachable!("not a factorization test")
    }
    fn dot_general_into(
        &mut self,
        _: &mut dyn Engine,
        _: DenseWrite<'_>,
        _: DenseRead<'_>,
        _: DenseRead<'_>,
        _: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        unreachable!("the batch entry is overridden")
    }
    fn matmul_batch_axpby_with_ops_into(
        &mut self,
        _: &mut dyn Engine,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        jobs: &[DenseGemmBatchJob],
        _: &[usize],
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
        alpha: DenseScalar,
        beta: DenseScalar,
    ) -> Result<(), DenseError> {
        assert!(beta.is_zero(), "the default fresh entry overwrites");
        let call = self.calls;
        self.calls += 1;
        match self.fail_at {
            Some((at, Failure::Error)) if at == call => {
                return Err(DenseError::Unsupported {
                    op: "scalar_loop",
                    message: "injected failure".to_string(),
                })
            }
            Some((at, Failure::Panic)) if at == call => panic!("injected failure"),
            _ => {}
        }
        let (
            DenseWrite::F64(mut out),
            DenseRead::F64(lhs),
            DenseRead::F64(rhs),
            DenseScalar::F64(alpha),
        ) = (output, lhs, rhs, alpha)
        else {
            unreachable!("f64 fixture")
        };
        for job in jobs {
            let block = oracle(lhs.data(), rhs.data(), job, [lhs_op, rhs_op], alpha);
            let start = out.offset() + job.dst_offset;
            out.data_mut()[start..start + block.len()].copy_from_slice(&block);
        }
        Ok(())
    }
}

#[cfg(feature = "tenferro")]
#[test]
fn default_fresh_entry_writes_every_job_of_an_uninitialized_destination() {
    // Genuinely uninitialized memory: under Miri, a read of any element the
    // entry did not write before reporting it is undefined behavior.
    let engine = &mut CpuEngine::serial();
    let jobs = [
        batch_job((2, 3, 3), (0, 0, 0)),
        batch_job((3, 2, 2), (6, 6, 9)),
        batch_job((0, 2, 4), (12, 0, 0)),
        batch_job((2, 0, 2), (12, 0, 0)),
    ];
    let lhs = values::<f64>(12, 1);
    let rhs = values::<f64>(13, 2);
    for ops in [
        [MatrixOp::Identity; 2],
        [MatrixOp::Transpose, MatrixOp::Adjoint],
    ] {
        let mut buffer = vec![MaybeUninit::<f64>::uninit(); destination_len(&jobs)];
        let reports = run_fresh(
            &mut ScalarLoop::new(),
            engine,
            &mut buffer,
            &segments_of(&jobs),
            &lhs,
            &rhs,
            &jobs,
            ops,
            1.5,
        )
        .unwrap();
        check_values(&reports, &lhs, &rhs, &jobs, ops, 1.5);
    }
}

#[cfg(feature = "tenferro")]
#[test]
fn fresh_entry_refuses_a_job_outside_its_segments_and_reports_nothing() {
    let engine = &mut CpuEngine::serial();
    let jobs = [
        batch_job((2, 1, 2), (0, 0, 0)),
        batch_job((2, 1, 2), (4, 0, 0)),
    ];
    let (lhs, rhs) = (values::<f64>(2, 0), values::<f64>(2, 1));
    let mut buffer = vec![MaybeUninit::<f64>::uninit(); 8];
    // Only the first job's output is handed over.
    let error = run_fresh(
        &mut ScalarLoop::new(),
        engine,
        &mut buffer,
        &[(0, 4)],
        &lhs,
        &rhs,
        &jobs,
        [MatrixOp::Identity; 2],
        1.0,
    )
    .unwrap_err();
    assert_eq!(error, DenseError::OutOfBounds);
}

/// The default entry runs one call per job; job 0 succeeds and writes its
/// piece before job 1's call fails.
#[cfg(feature = "tenferro")]
fn fail_on_second_job(failure: Failure) -> std::thread::Result<Result<(), DenseError>> {
    let engine = &mut CpuEngine::serial();
    let jobs = [
        batch_job((2, 1, 2), (0, 0, 0)),
        batch_job((2, 1, 2), (4, 2, 2)),
    ];
    let (lhs, rhs) = (values::<f64>(4, 0), values::<f64>(4, 1));
    let mut buffer = [MaybeUninit::<f64>::uninit(); 8];
    let mut parts = vec![(0, &mut buffer[..])];
    let mut executor = ScalarLoop {
        calls: 0,
        fail_at: Some((1, failure)),
    };
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut written = |job: usize, _: &mut [f64]| panic!("job {job} reported after a failure");
        executor.matmul_batch_fresh(
            engine,
            DenseFreshSegments::F64 {
                segments: &mut parts,
                zeroed: zeroed::<f64>,
                written: &mut written,
            },
            DenseRead::F64(DenseView::new(&lhs, &[4], &[1], 0).unwrap()),
            DenseRead::F64(DenseView::new(&rhs, &[4], &[1], 0).unwrap()),
            &jobs,
            [MatrixOp::Identity; 2],
            DenseScalar::F64(1.0),
        )
    }))
}

#[cfg(feature = "tenferro")]
#[test]
fn an_error_after_a_finished_job_reports_no_job() {
    let error = fail_on_second_job(Failure::Error).unwrap().unwrap_err();
    assert!(matches!(
        error,
        DenseError::Unsupported {
            op: "scalar_loop",
            ..
        }
    ));
}

#[cfg(feature = "tenferro")]
#[test]
fn a_panic_after_a_finished_job_reports_no_job() {
    let payload = fail_on_second_job(Failure::Panic).unwrap_err();
    assert_eq!(payload.downcast_ref::<&str>(), Some(&"injected failure"));
}

#[cfg(feature = "tenferro")]
mod default_executor {
    use super::*;

    const IDENTITY: [MatrixOp; 2] = [MatrixOp::Identity; 2];
    const OPS: [MatrixOp; 2] = [MatrixOp::Adjoint, MatrixOp::Transpose];

    /// One job; a gapless affine run; an affine run with gaps; heterogeneous
    /// jobs with gaps, an empty output and `k = 0`.
    fn fixtures() -> Vec<(&'static str, Vec<DenseGemmBatchJob>)> {
        let run = |count: usize, dst_step: usize| -> Vec<DenseGemmBatchJob> {
            (0..count)
                .map(|i| batch_job((3, 4, 2), (1 + i * dst_step, 2 + i * 12, 1 + i * 8)))
                .collect()
        };
        vec![
            ("one job", vec![batch_job((3, 4, 2), (2, 1, 3))]),
            ("gapless run", run(4, 6)),
            ("gapped run", run(3, 9)),
            (
                "heterogeneous",
                vec![
                    batch_job((2, 3, 4), (20, 0, 0)),
                    batch_job((3, 2, 1), (0, 6, 12)),
                    batch_job((0, 3, 2), (8, 0, 0)),
                    batch_job((2, 0, 3), (4, 0, 0)),
                    batch_job((1, 5, 3), (10, 12, 2)),
                ],
            ),
        ]
    }

    fn operands<T: Val>(jobs: &[DenseGemmBatchJob]) -> (Vec<T>, Vec<T>) {
        let len = |f: fn(&DenseGemmBatchJob) -> usize| jobs.iter().map(f).max().unwrap_or(0);
        (
            values(len(|j| j.lhs_offset + j.rows * j.contracted), 1),
            values(len(|j| j.rhs_offset + j.contracted * j.cols), 2),
        )
    }

    /// Every element of the destination starts as NaN, so a route that read
    /// the prior content (`beta * C`, or an accumulation) shows up as NaN.
    fn poisoned<T: Val>(len: usize) -> Vec<MaybeUninit<T>> {
        vec![MaybeUninit::new(T::new(f64::NAN, f64::NAN)); len]
    }

    fn values_on<T: Val>(engine: &mut dyn Engine) {
        for (name, jobs) in fixtures() {
            let (lhs, rhs) = operands::<T>(&jobs);
            for ops in [IDENTITY, OPS] {
                let alpha = T::new(0.75, -0.5);
                let mut buffer = poisoned::<T>(destination_len(&jobs));
                let reports = run_fresh(
                    &mut DefaultDenseExecutor::new(),
                    engine,
                    &mut buffer,
                    &segments_of(&jobs),
                    &lhs,
                    &rhs,
                    &jobs,
                    ops,
                    alpha,
                )
                .unwrap_or_else(|err| panic!("{name} {ops:?}: {err}"));
                check_values(&reports, &lhs, &rhs, &jobs, ops, alpha);
            }
        }
    }

    #[test]
    fn fresh_batches_match_the_scalar_gemm_for_every_dtype_route_and_op() {
        let serial = &mut CpuEngine::serial();
        values_on::<f32>(serial);
        values_on::<f64>(serial);
        values_on::<Complex32>(serial);
        values_on::<Complex64>(serial);
    }

    #[test]
    fn fresh_batches_match_the_scalar_gemm_under_a_forced_lane_policy() {
        // A forced policy takes the BLAS build off its vendor loop onto the
        // lanes; the native build runs its lanes either way.
        use tcontract_exec::{BatchPolicy, BatchStrategy};
        let engine = &mut CpuEngine::new(Pool::builder().threads(3).build().unwrap())
            .with_batch_policy(BatchPolicy::new(BatchStrategy::Outer));
        values_on::<f64>(engine);
        values_on::<Complex64>(engine);
    }

    #[test]
    fn zero_alpha_fresh_batches_are_zero_with_nonfinite_operands() {
        let engine = &mut CpuEngine::serial();
        for (name, jobs) in fixtures() {
            let (mut lhs, mut rhs) = operands::<f64>(&jobs);
            lhs.iter_mut().step_by(2).for_each(|v| *v = f64::NAN);
            rhs.iter_mut().step_by(3).for_each(|v| *v = f64::INFINITY);
            for ops in [IDENTITY, OPS] {
                let mut buffer = poisoned::<f64>(destination_len(&jobs));
                let reports = run_fresh(
                    &mut DefaultDenseExecutor::new(),
                    engine,
                    &mut buffer,
                    &segments_of(&jobs),
                    &lhs,
                    &rhs,
                    &jobs,
                    ops,
                    0.0,
                )
                .unwrap();
                for report in reports {
                    assert!(
                        report.iter().all(|v| *v == 0.0),
                        "{name} {ops:?}: {report:?}"
                    );
                }
            }
        }
    }

    /// A grouped batch whose `k = 0` job has no vendor plan, so the
    /// initialized BLAS group goes native as a whole.
    fn mixed_zero_k() -> Vec<DenseGemmBatchJob> {
        vec![
            batch_job((2, 3, 2), (0, 0, 0)),
            batch_job((2, 0, 2), (5, 0, 0)),
        ]
    }

    #[test]
    fn fresh_batches_keep_the_initialized_route_except_gapped_runs() {
        let engine = &mut CpuEngine::serial();
        let mut cases = fixtures();
        cases.retain(|(name, _)| *name != "heterogeneous");
        cases.push(("mixed zero k", mixed_zero_k()));
        for (name, jobs) in cases {
            let (lhs, rhs) = operands::<f64>(&jobs);
            let runs = strided_batch_runs(&jobs);
            let len = destination_len(&jobs);
            // The mixed batch is handed over as the one destination its
            // initialized call writes; the others as their maximal runs.
            let segments = if name == "mixed zero k" {
                vec![(0, len)]
            } else {
                segments_of(&jobs)
            };
            for ops in [IDENTITY, OPS] {
                let mut initialized = DefaultDenseExecutor::new();
                let mut output = vec![0.0; len];
                let (shape, lhs_shape, rhs_shape, strides) = ([len], [lhs.len()], [rhs.len()], [1]);
                initialized
                    .matmul_batch_axpby_with_ops_into(
                        engine,
                        DenseWrite::F64(
                            DenseViewMut::new(&mut output, &shape, &strides, 0).unwrap(),
                        ),
                        DenseRead::F64(DenseView::new(&lhs, &lhs_shape, &strides, 0).unwrap()),
                        DenseRead::F64(DenseView::new(&rhs, &rhs_shape, &strides, 0).unwrap()),
                        &jobs,
                        &runs,
                        ops[0],
                        ops[1],
                        DenseScalar::F64(1.0),
                        DenseScalar::F64(0.0),
                    )
                    .unwrap();
                let mut fresh = DefaultDenseExecutor::new();
                let mut buffer = poisoned::<f64>(len);
                run_fresh(
                    &mut fresh,
                    engine,
                    &mut buffer,
                    &segments,
                    &lhs,
                    &rhs,
                    &jobs,
                    ops,
                    1.0,
                )
                .unwrap();
                let what = format!("{name} {ops:?}");
                if name == "gapped run" {
                    assert_eq!(initialized.staged_grouped_jobs(), 0, "{what}");
                    if cfg!(feature = "cpu-blas-core") {
                        // One initialized call per segment, each one job.
                        assert_eq!(fresh.seam_dispatches(), segments.len(), "{what}");
                        assert_eq!(fresh.staged_grouped_jobs(), 0, "{what}");
                    } else {
                        assert_eq!(fresh.staged_grouped_jobs(), jobs.len(), "{what}");
                    }
                    continue;
                }
                assert_eq!(
                    fresh.seam_dispatches(),
                    initialized.seam_dispatches(),
                    "{what}"
                );
                assert_eq!(
                    fresh.staged_grouped_jobs(),
                    initialized.staged_grouped_jobs(),
                    "{what}"
                );
                #[cfg(feature = "cpu-blas-core")]
                assert_eq!(fresh.grouped_route(), initialized.grouped_route(), "{what}");
            }
        }
    }

    #[cfg(feature = "cpu-blas-core")]
    #[test]
    fn mixed_zero_k_blas_group_goes_native_on_both_routes() {
        // The decision the route test compares is a real one: this group is
        // native, not the vendor loop.
        let engine = &mut CpuEngine::serial();
        let jobs = mixed_zero_k();
        let (lhs, rhs) = operands::<f64>(&jobs);
        let len = destination_len(&jobs);
        let mut fresh = DefaultDenseExecutor::new();
        let mut buffer = poisoned::<f64>(len);
        run_fresh(
            &mut fresh,
            engine,
            &mut buffer,
            &[(0, len)],
            &lhs,
            &rhs,
            &jobs,
            IDENTITY,
            1.0,
        )
        .unwrap();
        assert!(
            matches!(
                fresh.grouped_route(),
                Some(cpueinsum_blas::GroupedRoute::Native(_))
            ),
            "{:?}",
            fresh.grouped_route()
        );
    }

    fn many_segments_on<T: Val>(engine: &mut dyn Engine) {
        // Every job in its own one-element segment, in a job order unrelated
        // to destination order, so each job's segment must be found by
        // destination, not by position.
        let count = 64usize;
        let k = 3usize;
        let jobs: Vec<DenseGemmBatchJob> = (0..count)
            .map(|i| batch_job((1, k, 1), (2 * ((i * 37) % count), i * k, i * k)))
            .collect();
        let segments = segments_of(&jobs);
        assert_eq!(segments.len(), count);
        let (lhs, rhs) = operands::<T>(&jobs);
        for ops in [IDENTITY, OPS] {
            let alpha = T::new(-1.25, 0.5);
            let mut buffer = poisoned::<T>(destination_len(&jobs));
            let reports = run_fresh(
                &mut DefaultDenseExecutor::new(),
                engine,
                &mut buffer,
                &segments,
                &lhs,
                &rhs,
                &jobs,
                ops,
                alpha,
            )
            .unwrap();
            check_values(&reports, &lhs, &rhs, &jobs, ops, alpha);
        }
    }

    #[test]
    fn fresh_batches_over_many_one_element_segments_match_the_scalar_gemm() {
        let engine = &mut CpuEngine::serial();
        many_segments_on::<f64>(engine);
        many_segments_on::<Complex64>(engine);
    }

    #[test]
    fn failed_fresh_batches_report_no_job() {
        let engine = &mut CpuEngine::serial();
        for (name, jobs) in fixtures() {
            let (lhs, rhs) = operands::<f64>(&jobs);
            // One element short: the last job's lhs range is out of bounds.
            let short = &lhs[..lhs.len() - 1];
            let mut buffer = poisoned::<f64>(destination_len(&jobs));
            let result = run_fresh(
                &mut DefaultDenseExecutor::new(),
                engine,
                &mut buffer,
                &segments_of(&jobs),
                short,
                &rhs,
                &jobs,
                IDENTITY,
                1.0,
            );
            assert!(result.is_err(), "{name}");
        }
    }

    #[test]
    fn fresh_entry_refuses_a_zero_fill_that_returns_other_storage() {
        fn foreign(piece: &mut [MaybeUninit<f64>]) -> &mut [f64] {
            let _ = piece;
            vec![0.0; 6].leak()
        }
        let engine = &mut CpuEngine::serial();
        let jobs = [batch_job((3, 4, 2), (0, 0, 0))];
        let (lhs, rhs) = operands::<f64>(&jobs);
        let mut buffer = poisoned::<f64>(6);
        let mut parts = vec![(0, &mut buffer[..])];
        let mut written = |_: usize, _: &mut [f64]| panic!("nothing is reported on error");
        let error = DefaultDenseExecutor::new()
            .matmul_batch_fresh(
                engine,
                DenseFreshSegments::F64 {
                    segments: &mut parts,
                    zeroed: foreign,
                    written: &mut written,
                },
                DenseRead::F64(DenseView::new(&lhs, &[12], &[1], 0).unwrap()),
                DenseRead::F64(DenseView::new(&rhs, &[8], &[1], 0).unwrap()),
                &jobs,
                IDENTITY,
                DenseScalar::F64(1.0),
            )
            .unwrap_err();
        assert!(matches!(
            error,
            DenseError::Unsupported {
                op: "matmul_batch_fresh",
                ..
            }
        ));
    }
}
