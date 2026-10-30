#![allow(dead_code)]

use super::*;
use num_complex::{Complex32, Complex64};

mod dot;
mod executor;
#[cfg(all(
    any(feature = "cpu-faer", feature = "cpu-blas-core"),
    not(feature = "provider-inject")
))]
mod factor_properties;
mod tensor;
mod view;

fn assert_f64_close(actual: f64, expected: f64, tol: f64) {
    assert!(
        (actual - expected).abs() <= tol,
        "expected {expected}, got {actual}, tol={tol}"
    );
}
fn assert_f32_close(actual: f32, expected: f32, tol: f32) {
    assert!(
        (actual - expected).abs() <= tol,
        "expected {expected}, got {actual}, tol={tol}"
    );
}
fn assert_c32_close(actual: Complex32, expected: Complex32, tol: f32) {
    assert_f32_close(actual.re, expected.re, tol);
    assert_f32_close(actual.im, expected.im, tol);
}
fn assert_c64_close(actual: Complex64, expected: Complex64, tol: f64) {
    assert_f64_close(actual.re, expected.re, tol);
    assert_f64_close(actual.im, expected.im, tol);
}
fn oriented_c64_value(
    data: &[Complex64],
    offset: usize,
    rows: usize,
    cols: usize,
    row: usize,
    col: usize,
    op: MatrixOp,
) -> Complex64 {
    let value = match op {
        MatrixOp::Identity => data[offset + row + rows * col],
        MatrixOp::Transpose | MatrixOp::Adjoint => data[offset + col + cols * row],
    };
    if op == MatrixOp::Adjoint {
        value.conj()
    } else {
        value
    }
}
fn check_complex_oriented_run(
    executor: &mut DefaultDenseExecutor,
    shape: (usize, usize, usize),
    lhs_op: MatrixOp,
    rhs_op: MatrixOp,
    broadcast_lhs: bool,
    run_len: usize,
) {
    let engine = &mut CpuEngine::serial();
    let (rows, contracted, cols) = shape;
    let view_offset = 2usize;
    let lhs_first = 3usize;
    let rhs_first = 4usize;
    let dst_first = 5usize;
    let lhs_step = if broadcast_lhs {
        0
    } else {
        rows * contracted + 2
    };
    let rhs_step = contracted * cols + 3;
    let dst_step = rows * cols + 3;
    let jobs = (0..run_len)
        .map(|batch| DenseGemmBatchJob {
            lhs_offset: lhs_first + batch * lhs_step,
            rhs_offset: rhs_first + batch * rhs_step,
            dst_offset: dst_first + batch * dst_step,
            rows,
            contracted,
            cols,
        })
        .collect::<Vec<_>>();
    let runs = strided_batch_runs(&jobs);
    assert_eq!(runs, [run_len]);

    let last = &jobs[run_len - 1];
    let lhs_len = view_offset + last.lhs_offset + rows * contracted + 2;
    let rhs_len = view_offset + last.rhs_offset + contracted * cols + 2;
    let out_len = view_offset + last.dst_offset + rows * cols + 2;
    let lhs = (0..lhs_len)
        .map(|i| Complex64::new(0.5 + 0.25 * i as f64, -0.75 + 0.125 * i as f64))
        .collect::<Vec<_>>();
    let rhs = (0..rhs_len)
        .map(|i| Complex64::new(-1.0 + 0.2 * i as f64, 0.625 - 0.1 * i as f64))
        .collect::<Vec<_>>();
    let mut output = (0..out_len)
        .map(|i| Complex64::new(2.0 + 0.05 * i as f64, -1.0 - 0.025 * i as f64))
        .collect::<Vec<_>>();
    let lhs_before = lhs.clone();
    let rhs_before = rhs.clone();
    let output_before = output.clone();
    let alpha = Complex64::new(0.75, -0.5);
    let beta = Complex64::new(-0.25, 0.125);
    let strides = [1usize];
    let lhs_shape = [lhs.len() - view_offset];
    let rhs_shape = [rhs.len() - view_offset];
    let out_shape = [output.len() - view_offset];

    executor.reset_seam_dispatches();
    executor
        .matmul_batch_axpby_with_ops_into(
            engine,
            DenseWrite::C64(
                DenseViewMut::new(&mut output, &out_shape, &strides, view_offset).unwrap(),
            ),
            DenseRead::C64(DenseView::new(&lhs, &lhs_shape, &strides, view_offset).unwrap()),
            DenseRead::C64(DenseView::new(&rhs, &rhs_shape, &strides, view_offset).unwrap()),
            &jobs,
            &runs,
            lhs_op,
            rhs_op,
            DenseScalar::C64(alpha),
            DenseScalar::C64(beta),
        )
        .unwrap();
    assert_eq!(executor.seam_dispatches(), 1);

    let mut touched = vec![false; output.len()];
    for job in &jobs {
        for col in 0..cols {
            for row in 0..rows {
                let mut sum = Complex64::new(0.0, 0.0);
                for inner in 0..contracted {
                    let lhs_value = oriented_c64_value(
                        &lhs,
                        view_offset + job.lhs_offset,
                        rows,
                        contracted,
                        row,
                        inner,
                        lhs_op,
                    );
                    let rhs_value = oriented_c64_value(
                        &rhs,
                        view_offset + job.rhs_offset,
                        contracted,
                        cols,
                        inner,
                        col,
                        rhs_op,
                    );
                    sum += lhs_value * rhs_value;
                }
                let index = view_offset + job.dst_offset + row + rows * col;
                touched[index] = true;
                assert_c64_close(
                    output[index],
                    alpha * sum + beta * output_before[index],
                    1.0e-11,
                );
            }
        }
    }
    assert_eq!(lhs, lhs_before);
    assert_eq!(rhs, rhs_before);
    for (index, value) in output.iter().enumerate() {
        if !touched[index] {
            assert_eq!(*value, output_before[index]);
        }
    }
}
fn col_major_index(rows: usize, row: usize, col: usize) -> usize {
    row + col * rows
}
fn transpose_f32(mat: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0; rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            out[col_major_index(cols, j, i)] = mat[col_major_index(rows, i, j)];
        }
    }
    out
}
fn transpose_f64(mat: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            out[col_major_index(cols, j, i)] = mat[col_major_index(rows, i, j)];
        }
    }
    out
}
fn transpose_c32(mat: &[Complex32], rows: usize, cols: usize) -> Vec<Complex32> {
    let mut out = vec![Complex32::new(0.0, 0.0); rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            out[col_major_index(cols, j, i)] = mat[col_major_index(rows, i, j)];
        }
    }
    out
}
fn transpose_c64(mat: &[Complex64], rows: usize, cols: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            out[col_major_index(cols, j, i)] = mat[col_major_index(rows, i, j)];
        }
    }
    out
}
fn conjugate_transpose_c32(mat: &[Complex32], rows: usize, cols: usize) -> Vec<Complex32> {
    let mut out = vec![Complex32::new(0.0, 0.0); rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            out[col_major_index(cols, j, i)] = mat[col_major_index(rows, i, j)].conj();
        }
    }
    out
}
fn conjugate_transpose_c64(mat: &[Complex64], rows: usize, cols: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            out[col_major_index(cols, j, i)] = mat[col_major_index(rows, i, j)].conj();
        }
    }
    out
}
fn matmul_f32(lhs: &[f32], rhs: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0; m * n];
    for j in 0..n {
        for p in 0..k {
            let rhs_pj = rhs[col_major_index(k, p, j)];
            for i in 0..m {
                out[col_major_index(m, i, j)] += lhs[col_major_index(m, i, p)] * rhs_pj;
            }
        }
    }
    out
}
fn matmul_f64(lhs: &[f64], rhs: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; m * n];
    for j in 0..n {
        for p in 0..k {
            let rhs_pj = rhs[col_major_index(k, p, j)];
            for i in 0..m {
                out[col_major_index(m, i, j)] += lhs[col_major_index(m, i, p)] * rhs_pj;
            }
        }
    }
    out
}
fn matmul_c32(
    lhs: &[Complex32],
    rhs: &[Complex32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<Complex32> {
    let mut out = vec![Complex32::new(0.0, 0.0); m * n];
    for j in 0..n {
        for p in 0..k {
            let rhs_pj = rhs[col_major_index(k, p, j)];
            for i in 0..m {
                out[col_major_index(m, i, j)] += lhs[col_major_index(m, i, p)] * rhs_pj;
            }
        }
    }
    out
}
fn matmul_c64(
    lhs: &[Complex64],
    rhs: &[Complex64],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); m * n];
    for j in 0..n {
        for p in 0..k {
            let rhs_pj = rhs[col_major_index(k, p, j)];
            for i in 0..m {
                out[col_major_index(m, i, j)] += lhs[col_major_index(m, i, p)] * rhs_pj;
            }
        }
    }
    out
}
fn diag_f32(values: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; values.len() * values.len()];
    for (i, value) in values.iter().enumerate() {
        out[col_major_index(values.len(), i, i)] = *value;
    }
    out
}
fn diag_f64(values: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; values.len() * values.len()];
    for (i, value) in values.iter().enumerate() {
        out[col_major_index(values.len(), i, i)] = *value;
    }
    out
}
fn diag_c32_from_real(values: &[f32]) -> Vec<Complex32> {
    let mut out = vec![Complex32::new(0.0, 0.0); values.len() * values.len()];
    for (i, value) in values.iter().enumerate() {
        out[col_major_index(values.len(), i, i)] = Complex32::new(*value, 0.0);
    }
    out
}
fn diag_c64_from_real(values: &[f64]) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); values.len() * values.len()];
    for (i, value) in values.iter().enumerate() {
        out[col_major_index(values.len(), i, i)] = Complex64::new(*value, 0.0);
    }
    out
}
#[cfg(all(
    not(feature = "provider-inject"),
    any(feature = "cpu-faer", feature = "cpu-blas-core")
))]
fn assert_values_only_for_compiled_cpu_provider() {
    let engine = &mut CpuEngine::serial();
    let mut executor = DefaultDenseExecutor::new();
    let data = [2.0_f64, 0.0, 0.0, -1.0];
    let shape = [2, 2];
    let strides = [1, 2];

    let values = executor
        .svd_vals(
            engine,
            DenseRead::F64(DenseView::new(&data, &shape, &strides, 0).unwrap()),
        )
        .unwrap();
    let values = values.as_f64_slice().unwrap();
    assert_eq!(values.len(), 2);
    for (actual, expected) in values.iter().zip([2.0, 1.0]) {
        assert_f64_close(*actual, expected, 1.0e-12);
    }

    let values = executor
        .eigh_vals(
            engine,
            DenseRead::F64(DenseView::new(&data, &shape, &strides, 0).unwrap()),
        )
        .unwrap();
    let values = values.as_f64_slice().unwrap();
    assert_eq!(values.len(), 2);
    for (actual, expected) in values.iter().zip([-1.0, 2.0]) {
        assert_f64_close(*actual, expected, 1.0e-12);
    }
}
fn batch_job(shape: (usize, usize, usize), offsets: (usize, usize, usize)) -> DenseGemmBatchJob {
    DenseGemmBatchJob {
        rows: shape.0,
        contracted: shape.1,
        cols: shape.2,
        dst_offset: offsets.0,
        lhs_offset: offsets.1,
        rhs_offset: offsets.2,
    }
}
// ---------------------------------------------------------------------------
// Identity batch dispatch (#1182): one strided rank-3 dot iff the batch is one
// affine run of >= 2 jobs with destination step >= rows * cols; otherwise one
// grouped submission over every job. Oracles below are scalar loops over the
// job list, independent of the adapter's routing.
// ---------------------------------------------------------------------------
trait IdentityBatchScalar:
    Copy + Default + std::ops::Add<Output = Self> + std::ops::Mul<Output = Self> + PartialEq
{
    fn read(view: DenseView<'_, Self>) -> DenseRead<'_>;
    fn write(view: DenseViewMut<'_, Self>) -> DenseWrite<'_>;
    fn scalar(value: Self) -> DenseScalar;
    fn sample(index: usize, salt: f64) -> Self;
    fn assert_close(actual: Self, expected: Self);
}
impl IdentityBatchScalar for f64 {
    fn read(view: DenseView<'_, Self>) -> DenseRead<'_> {
        DenseRead::F64(view)
    }
    fn write(view: DenseViewMut<'_, Self>) -> DenseWrite<'_> {
        DenseWrite::F64(view)
    }
    fn scalar(value: Self) -> DenseScalar {
        DenseScalar::F64(value)
    }
    fn sample(index: usize, salt: f64) -> Self {
        0.5 + salt + 0.25 * index as f64 - 0.01 * (index * index % 7) as f64
    }
    fn assert_close(actual: Self, expected: Self) {
        assert_f64_close(actual, expected, 1.0e-10);
    }
}
impl IdentityBatchScalar for f32 {
    fn read(view: DenseView<'_, Self>) -> DenseRead<'_> {
        DenseRead::F32(view)
    }
    fn write(view: DenseViewMut<'_, Self>) -> DenseWrite<'_> {
        DenseWrite::F32(view)
    }
    fn scalar(value: Self) -> DenseScalar {
        DenseScalar::F32(value)
    }
    fn sample(index: usize, salt: f64) -> Self {
        <f64 as IdentityBatchScalar>::sample(index, salt) as f32
    }
    fn assert_close(actual: Self, expected: Self) {
        assert_f32_close(actual, expected, 1.0e-3);
    }
}
impl IdentityBatchScalar for Complex64 {
    fn read(view: DenseView<'_, Self>) -> DenseRead<'_> {
        DenseRead::C64(view)
    }
    fn write(view: DenseViewMut<'_, Self>) -> DenseWrite<'_> {
        DenseWrite::C64(view)
    }
    fn scalar(value: Self) -> DenseScalar {
        DenseScalar::C64(value)
    }
    fn sample(index: usize, salt: f64) -> Self {
        Complex64::new(
            <f64 as IdentityBatchScalar>::sample(index, salt),
            -0.75 + 0.125 * index as f64 + salt,
        )
    }
    fn assert_close(actual: Self, expected: Self) {
        assert_c64_close(actual, expected, 1.0e-10);
    }
}
impl IdentityBatchScalar for Complex32 {
    fn read(view: DenseView<'_, Self>) -> DenseRead<'_> {
        DenseRead::C32(view)
    }
    fn write(view: DenseViewMut<'_, Self>) -> DenseWrite<'_> {
        DenseWrite::C32(view)
    }
    fn scalar(value: Self) -> DenseScalar {
        DenseScalar::C32(value)
    }
    fn sample(index: usize, salt: f64) -> Self {
        let value = <Complex64 as IdentityBatchScalar>::sample(index, salt);
        Complex32::new(value.re as f32, value.im as f32)
    }
    fn assert_close(actual: Self, expected: Self) {
        assert_c32_close(actual, expected, 1.0e-3);
    }
}
/// Column-major scalar oracle: `alpha * lhs * rhs + beta * dst` per job over
/// flat buffers with view base offsets; elements no job owns are unchanged.
fn identity_batch_oracle<T: IdentityBatchScalar>(
    jobs: &[DenseGemmBatchJob],
    lhs: &[T],
    rhs: &[T],
    output_before: &[T],
    base: (usize, usize, usize),
    alpha: T,
    beta: T,
) -> Vec<T> {
    let mut expected = output_before.to_vec();
    for job in jobs {
        let lhs_base = base.0 + job.lhs_offset;
        let rhs_base = base.1 + job.rhs_offset;
        let dst_base = base.2 + job.dst_offset;
        for col in 0..job.cols {
            for row in 0..job.rows {
                let mut sum = T::default();
                for inner in 0..job.contracted {
                    sum = sum
                        + lhs[lhs_base + row + job.rows * inner]
                            * rhs[rhs_base + inner + job.contracted * col];
                }
                let index = dst_base + row + job.rows * col;
                expected[index] = alpha * sum + beta * output_before[index];
            }
        }
    }
    expected
}
struct IdentityBatchFixture<T> {
    jobs: Vec<DenseGemmBatchJob>,
    lhs: Vec<T>,
    rhs: Vec<T>,
    output: Vec<T>,
    base: (usize, usize, usize),
}
/// Lays `shapes` out sequentially in flat lhs/rhs/dst buffers, leaving `gap`
/// unused elements between consecutive blocks of every operand and `base`
/// elements before the view offset. `gap == 0` yields a batch whose same-shape
/// neighbours are one affine run.
fn identity_fixture<T: IdentityBatchScalar>(
    shapes: &[(usize, usize, usize)],
    gap: usize,
    base: (usize, usize, usize),
    sentinel: T,
) -> IdentityBatchFixture<T> {
    let (mut lhs_off, mut rhs_off, mut dst_off) = (0usize, 0usize, 0usize);
    let jobs = shapes
        .iter()
        .map(|&(rows, contracted, cols)| {
            let job = DenseGemmBatchJob {
                dst_offset: dst_off,
                lhs_offset: lhs_off,
                rhs_offset: rhs_off,
                rows,
                contracted,
                cols,
            };
            lhs_off += rows * contracted + gap;
            rhs_off += contracted * cols + gap;
            dst_off += rows * cols + gap;
            job
        })
        .collect::<Vec<_>>();
    let lhs = (0..base.0 + lhs_off + 1)
        .map(|i| T::sample(i, 0.0))
        .collect();
    let rhs = (0..base.1 + rhs_off + 1)
        .map(|i| T::sample(i, 1.5))
        .collect();
    let output = vec![sentinel; base.2 + dst_off + 1];
    IdentityBatchFixture {
        jobs,
        lhs,
        rhs,
        output,
        base,
    }
}
fn run_identity_batch<T: IdentityBatchScalar>(
    executor: &mut DefaultDenseExecutor,
    fixture: &mut IdentityBatchFixture<T>,
    runs: &[usize],
    alpha: T,
    beta: T,
) -> Result<(), DenseError> {
    let engine = &mut CpuEngine::serial();
    let strides = [1usize];
    let (lhs_base, rhs_base, dst_base) = fixture.base;
    let lhs_shape = [fixture.lhs.len() - lhs_base];
    let rhs_shape = [fixture.rhs.len() - rhs_base];
    let out_shape = [fixture.output.len() - dst_base];
    executor.reset_seam_dispatches();
    executor.matmul_batch_axpby_into(
        engine,
        T::write(DenseViewMut::new(&mut fixture.output, &out_shape, &strides, dst_base).unwrap()),
        T::read(DenseView::new(&fixture.lhs, &lhs_shape, &strides, lhs_base).unwrap()),
        T::read(DenseView::new(&fixture.rhs, &rhs_shape, &strides, rhs_base).unwrap()),
        &fixture.jobs,
        runs,
        T::scalar(alpha),
        T::scalar(beta),
    )
}
/// Executes the batch with the plan-time partition (unless `runs` overrides
/// it), asserts exactly `dispatches` seam submissions and oracle-exact values.
#[allow(clippy::too_many_arguments)]
fn check_identity_batch<T: IdentityBatchScalar>(
    executor: &mut DefaultDenseExecutor,
    shapes: &[(usize, usize, usize)],
    gap: usize,
    base: (usize, usize, usize),
    runs: Option<&[usize]>,
    alpha: T,
    beta: T,
    dispatches: usize,
) {
    let mut fixture = identity_fixture::<T>(shapes, gap, base, T::sample(3, -4.0));
    let plan_runs = strided_batch_runs(&fixture.jobs);
    let runs = runs.unwrap_or(&plan_runs);
    let expected = identity_batch_oracle(
        &fixture.jobs,
        &fixture.lhs,
        &fixture.rhs,
        &fixture.output,
        base,
        alpha,
        beta,
    );
    run_identity_batch(executor, &mut fixture, runs, alpha, beta).unwrap();
    assert_eq!(
        executor.seam_dispatches(),
        dispatches,
        "shapes {shapes:?} gap {gap} runs {runs:?}"
    );
    for (actual, expected) in fixture.output.iter().zip(&expected) {
        T::assert_close(*actual, *expected);
    }
}
fn c64(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}
