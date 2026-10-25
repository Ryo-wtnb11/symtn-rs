//! `DenseExecutor::eig` returns a general eigendecomposition: `A V = V
//! diag(w)` with nonzero eigenvector columns, and `w` is the spectrum of `A`.
//!
//! Inputs are non-normal with a spectrum known by construction: a real
//! rotation-scaling block (a conjugate pair) above a real eigenvalue, and the
//! complex companion matrix of `(x - r0)(x - r1)(x - r2)`. Both the input and
//! the outputs are read column-major, the executor's layout convention.

#![cfg(all(feature = "tenferro", not(feature = "provider-inject")))]

use num_complex::Complex64;
use symtn_dense::{DefaultDenseExecutor, DenseDType, DenseExecutor, DenseRead, DenseView};

use crate::numerics;

const N: usize = 3;

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

/// `A V` and `V diag(w)` for column-major `n × n` `a` and `v`.
fn eigen_sides(a: &[Complex64], v: &[Complex64], w: &[Complex64]) -> [Vec<Complex64>; 2] {
    let n = w.len();
    let mut av = vec![c(0.0, 0.0); n * n];
    let mut vd = vec![c(0.0, 0.0); n * n];
    for col in 0..n {
        for row in 0..n {
            av[row + n * col] = (0..n).map(|k| a[row + n * k] * v[k + n * col]).sum();
            vd[row + n * col] = v[row + n * col] * w[col];
        }
    }
    [av, vd]
}

fn assert_eig(a: &[Complex64], read: DenseRead<'_>, spectrum: &[Complex64]) {
    let outputs = DefaultDenseExecutor::new().eig(read).unwrap();
    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].dtype(), DenseDType::C64);
    assert_eq!(outputs[1].dtype(), DenseDType::C64);
    assert_eq!(outputs[0].shape(), [N]);
    assert_eq!(outputs[1].shape(), [N, N]);
    let w = outputs[0].as_c64_slice().unwrap();
    let v = outputs[1].as_c64_slice().unwrap();

    for col in 0..N {
        let norm = (0..N).map(|row| v[row + N * col].norm_sqr()).sum::<f64>();
        assert!(norm > 0.5, "eigenvector column {col} is (nearly) zero");
    }
    let [av, vd] = eigen_sides(a, v, w);
    numerics::assert_nonzero_slices_close("A V = V diag(w)", &av, &vd, N);

    // Each hand eigenvalue is matched by one computed value (all distinct).
    let mut unmatched = w.to_vec();
    for &want in spectrum {
        let (index, &got) = unmatched
            .iter()
            .enumerate()
            .min_by(|x, y| (x.1 - want).norm().total_cmp(&(y.1 - want).norm()))
            .unwrap();
        numerics::assert_close("eigenvalue", got, want, N);
        unmatched.swap_remove(index);
    }
}

#[test]
fn real_eig_returns_the_conjugate_pair_and_satisfies_the_eigen_equation() {
    // [[1, -2, 0.3], [2, 1, -0.4], [0, 0, -1.5]] (rows): 1 ± 2i and -1.5.
    let a = [1.0, 2.0, 0.0, -2.0, 1.0, 0.0, 0.3, -0.4, -1.5];
    let wide = a.map(|x| c(x, 0.0));
    assert_eig(
        &wide,
        DenseRead::F64(DenseView::new(&a, &[N, N], &[1, N], 0).unwrap()),
        &[c(1.0, 2.0), c(1.0, -2.0), c(-1.5, 0.0)],
    );
}

#[test]
fn complex_eig_returns_the_companion_roots_and_satisfies_the_eigen_equation() {
    // Companion matrix of x^3 + c2 x^2 + c1 x + c0 = (x - r0)(x - r1)(x - r2):
    // ones on the subdiagonal, last column -(c0, c1, c2).
    let roots = [c(1.0, 1.0), c(-2.0, 0.0), c(0.0, 0.5)];
    let [r0, r1, r2] = roots;
    let c2 = -(r0 + r1 + r2);
    let c1 = r0 * r1 + r0 * r2 + r1 * r2;
    let c0 = -(r0 * r1 * r2);
    let zero = c(0.0, 0.0);
    let one = c(1.0, 0.0);
    let a = [zero, one, zero, zero, zero, one, -c0, -c1, -c2];
    assert_eig(
        &a,
        DenseRead::C64(DenseView::new(&a, &[N, N], &[1, N], 0).unwrap()),
        &roots,
    );
}
