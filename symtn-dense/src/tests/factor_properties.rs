//! Host dense QR and SVD return valid factors for every supported dtype:
//! the defining identities are checked directly, so no reference values are
//! needed.

use super::*;

const DTYPES: [DenseDType; 4] = [
    DenseDType::F32,
    DenseDType::F64,
    DenseDType::C32,
    DenseDType::C64,
];

/// `(rows, cols, rank_deficient)`: square, tall and wide, each at full rank
/// and with rank `min(rows, cols) - 1`.
const CASES: [(usize, usize, bool); 6] = [
    (3, 3, false),
    (4, 3, false),
    (3, 4, false),
    (3, 3, true),
    (4, 3, true),
    (3, 4, true),
];

fn is_single(dtype: DenseDType) -> bool {
    matches!(dtype, DenseDType::F32 | DenseDType::C32)
}

fn tolerance(dtype: DenseDType) -> f64 {
    if is_single(dtype) {
        1.0e-4
    } else {
        1.0e-10
    }
}

/// Column-major `rows x cols` input, exactly representable in `dtype`. A
/// rank-deficient case makes its last column (tall) or last row (otherwise)
/// the sum of the first two.
fn matrix(dtype: DenseDType, rows: usize, cols: usize, rank_deficient: bool) -> Vec<Complex64> {
    let complex = matches!(dtype, DenseDType::C32 | DenseDType::C64);
    let entry = |i: usize, j: usize| {
        let re = ((3 * i + 5 * j) % 7) as f64 - 3.0 + 0.5 * i as f64;
        let im = if complex {
            ((2 * i + j) % 5) as f64 - 2.0 + 0.25 * j as f64
        } else {
            0.0
        };
        Complex64::new(re, im)
    };
    let mut a = vec![Complex64::new(0.0, 0.0); rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            a[i + rows * j] = entry(i, j);
        }
    }
    if rank_deficient {
        if rows > cols {
            for i in 0..rows {
                a[i + rows * (cols - 1)] = a[i] + a[i + rows];
            }
        } else {
            for j in 0..cols {
                a[rows - 1 + rows * j] = a[rows * j] + a[1 + rows * j];
            }
        }
    }
    a
}

fn owned(dtype: DenseDType, a: &[Complex64]) -> DenseOwned {
    match dtype {
        DenseDType::F32 => DenseOwned::F32(a.iter().map(|z| z.re as f32).collect()),
        DenseDType::F64 => DenseOwned::F64(a.iter().map(|z| z.re).collect()),
        DenseDType::C32 => DenseOwned::C32(
            a.iter()
                .map(|z| Complex32::new(z.re as f32, z.im as f32))
                .collect(),
        ),
        DenseDType::C64 => DenseOwned::C64(a.to_vec()),
        other => panic!("no linalg input for {other:?}"),
    }
}

fn read<'a>(input: &'a DenseOwned, shape: &'a [usize], strides: &'a [usize]) -> DenseRead<'a> {
    match input {
        DenseOwned::F32(data) => DenseRead::F32(DenseView::new(data, shape, strides, 0).unwrap()),
        DenseOwned::F64(data) => DenseRead::F64(DenseView::new(data, shape, strides, 0).unwrap()),
        DenseOwned::C32(data) => DenseRead::C32(DenseView::new(data, shape, strides, 0).unwrap()),
        DenseOwned::C64(data) => DenseRead::C64(DenseView::new(data, shape, strides, 0).unwrap()),
    }
}

fn widen(tensor: &DenseTensor) -> Vec<Complex64> {
    match tensor.dtype() {
        DenseDType::F32 => tensor
            .as_f32_slice()
            .unwrap()
            .iter()
            .map(|&x| Complex64::new(x.into(), 0.0))
            .collect(),
        DenseDType::F64 => tensor
            .as_f64_slice()
            .unwrap()
            .iter()
            .map(|&x| Complex64::new(x, 0.0))
            .collect(),
        DenseDType::C32 => tensor
            .as_c32_slice()
            .unwrap()
            .iter()
            .map(|z| Complex64::new(z.re.into(), z.im.into()))
            .collect(),
        DenseDType::C64 => tensor.as_c64_slice().unwrap().to_vec(),
        other => panic!("no linalg output for {other:?}"),
    }
}

fn real_counterpart(dtype: DenseDType) -> DenseDType {
    if is_single(dtype) {
        DenseDType::F32
    } else {
        DenseDType::F64
    }
}

/// Column-major `lhs (m x k) * rhs (k x n)`.
fn product(lhs: &[Complex64], rhs: &[Complex64], m: usize, k: usize, n: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); m * n];
    for j in 0..n {
        for p in 0..k {
            for i in 0..m {
                out[i + m * j] += lhs[i + m * p] * rhs[p + k * j];
            }
        }
    }
    out
}

fn adjoint(x: &[Complex64], rows: usize, cols: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); rows * cols];
    for j in 0..cols {
        for i in 0..rows {
            out[j + cols * i] = x[i + rows * j].conj();
        }
    }
    out
}

fn assert_all_close(actual: &[Complex64], expected: &[Complex64], tol: f64, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).norm() <= tol,
            "{what}: element {index} is {a}, expected {e} (tol {tol})"
        );
    }
}

fn identity(n: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); n * n];
    for i in 0..n {
        out[i + n * i] = Complex64::new(1.0, 0.0);
    }
    out
}

/// `xᴴ x = I` for a column-major `rows x cols` `x`.
fn assert_orthonormal_columns(x: &[Complex64], rows: usize, cols: usize, tol: f64, what: &str) {
    let gram = product(&adjoint(x, rows, cols), x, cols, rows, cols);
    assert_all_close(&gram, &identity(cols), tol, what);
}

#[test]
fn host_qr_returns_orthonormal_q_and_upper_triangular_r_for_every_dtype_and_shape() {
    let engine = &mut CpuEngine::serial();
    let mut executor = DefaultDenseExecutor::new();
    for dtype in DTYPES {
        let tol = tolerance(dtype);
        for (m, n, rank_deficient) in CASES {
            let what = format!("{dtype:?} {m}x{n} rank_deficient={rank_deficient}");
            let k = m.min(n);
            let a = matrix(dtype, m, n, rank_deficient);
            let input = owned(dtype, &a);
            let (shape, strides) = ([m, n], [1, m]);
            let outputs = executor.qr(engine, read(&input, &shape, &strides)).unwrap();
            assert_eq!(outputs.len(), 2, "{what}");
            assert_eq!(outputs[0].shape(), [m, k], "{what}: Q shape");
            assert_eq!(outputs[1].shape(), [k, n], "{what}: R shape");
            assert_eq!(outputs[0].dtype(), dtype, "{what}: Q dtype");
            assert_eq!(outputs[1].dtype(), dtype, "{what}: R dtype");
            let q = widen(&outputs[0]);
            let r = widen(&outputs[1]);

            for j in 0..n {
                for i in (j + 1)..k {
                    assert_eq!(
                        r[i + k * j],
                        Complex64::new(0.0, 0.0),
                        "{what}: R[{i}, {j}] below the diagonal"
                    );
                }
            }
            assert_orthonormal_columns(&q, m, k, tol, &format!("{what}: QᴴQ"));
            assert_all_close(&product(&q, &r, m, k, n), &a, tol, &format!("{what}: QR"));
        }
    }
}

/// Checks the compact factors `u (m x k)`, `s (k)`, `vh (k x n)` of `a`.
fn check_compact_svd(
    a: &[Complex64],
    m: usize,
    n: usize,
    rank_deficient: bool,
    (u, s, vh): (&[Complex64], &[Complex64], &[Complex64]),
    tol: f64,
    what: &str,
) {
    let k = m.min(n);
    let s: Vec<f64> = s.iter().map(|z| z.re).collect();
    assert!(
        s.iter().all(|&x| x >= 0.0),
        "{what}: negative singular value in {s:?}"
    );
    assert!(
        s.windows(2).all(|pair| pair[0] >= pair[1]),
        "{what}: singular values not descending: {s:?}"
    );
    if rank_deficient {
        assert!(
            s[k - 1] <= tol * s[0],
            "{what}: rank-deficient input has smallest singular value {} (largest {})",
            s[k - 1],
            s[0]
        );
    }
    let frobenius_sq: f64 = a.iter().map(|z| z.norm_sqr()).sum();
    let spectrum_sq: f64 = s.iter().map(|x| x * x).sum();
    assert!(
        (frobenius_sq - spectrum_sq).abs() <= tol * frobenius_sq,
        "{what}: sum of squared singular values {spectrum_sq} != ||A||_F^2 {frobenius_sq}"
    );
    assert_orthonormal_columns(u, m, k, tol, &format!("{what}: UᴴU"));
    assert_orthonormal_columns(&adjoint(vh, k, n), n, k, tol, &format!("{what}: VhVhᴴ"));
    let mut us = u.to_vec();
    for (col, sigma) in s.iter().enumerate() {
        for row in 0..m {
            us[row + m * col] *= sigma;
        }
    }
    assert_all_close(&product(&us, vh, m, k, n), a, tol, &format!("{what}: USVh"));
}

#[test]
fn host_svd_returns_unitary_factors_and_ordered_spectrum_for_every_dtype_and_shape() {
    let engine = &mut CpuEngine::serial();
    let mut executor = DefaultDenseExecutor::new();
    for dtype in DTYPES {
        let tol = tolerance(dtype);
        for (m, n, rank_deficient) in CASES {
            let what = format!("{dtype:?} {m}x{n} rank_deficient={rank_deficient}");
            let k = m.min(n);
            let a = matrix(dtype, m, n, rank_deficient);
            let input = owned(dtype, &a);
            let (shape, strides) = ([m, n], [1, m]);
            let outputs = executor
                .svd(engine, read(&input, &shape, &strides))
                .unwrap();
            assert_eq!(outputs.len(), 3, "{what}");
            assert_eq!(outputs[0].shape(), [m, k], "{what}: U shape");
            assert_eq!(outputs[1].shape(), [k], "{what}: S shape");
            assert_eq!(outputs[2].shape(), [k, n], "{what}: Vh shape");
            assert_eq!(outputs[0].dtype(), dtype, "{what}: U dtype");
            assert_eq!(
                outputs[1].dtype(),
                real_counterpart(dtype),
                "{what}: S dtype"
            );
            assert_eq!(outputs[2].dtype(), dtype, "{what}: Vh dtype");
            let (u, s, vh) = (widen(&outputs[0]), widen(&outputs[1]), widen(&outputs[2]));
            check_compact_svd(&a, m, n, rank_deficient, (&u, &s, &vh), tol, &what);
        }
    }
}

#[test]
fn host_owned_full_svd_returns_square_unitary_factors_for_every_dtype_and_shape() {
    let engine = &mut CpuEngine::serial();
    let mut executor = DefaultDenseExecutor::new();
    for dtype in DTYPES {
        let tol = tolerance(dtype);
        for (m, n, rank_deficient) in CASES {
            let what = format!("{dtype:?} {m}x{n} rank_deficient={rank_deficient}");
            let k = m.min(n);
            let a = matrix(dtype, m, n, rank_deficient);
            let outputs = executor
                .svd_full_owned(engine, owned(dtype, &a), m, n)
                .unwrap();
            assert_eq!(outputs.len(), 3, "{what}");
            assert_eq!(outputs[0].shape(), [m, m], "{what}: U shape");
            assert_eq!(outputs[1].shape(), [k], "{what}: S shape");
            assert_eq!(outputs[2].shape(), [n, n], "{what}: Vh shape");
            assert_eq!(outputs[0].dtype(), dtype, "{what}: U dtype");
            assert_eq!(
                outputs[1].dtype(),
                real_counterpart(dtype),
                "{what}: S dtype"
            );
            assert_eq!(outputs[2].dtype(), dtype, "{what}: Vh dtype");
            let (u, s, vh) = (widen(&outputs[0]), widen(&outputs[1]), widen(&outputs[2]));

            assert_orthonormal_columns(&u, m, m, tol, &format!("{what}: full UᴴU"));
            assert_orthonormal_columns(
                &adjoint(&vh, n, n),
                n,
                n,
                tol,
                &format!("{what}: full VhVhᴴ"),
            );
            let leading_vh: Vec<Complex64> = (0..n)
                .flat_map(|col| (0..k).map(move |row| (row, col)))
                .map(|(row, col)| vh[row + n * col])
                .collect();
            check_compact_svd(
                &a,
                m,
                n,
                rank_deficient,
                (&u[..m * k], &s, &leading_vh),
                tol,
                &what,
            );
        }
    }
}
