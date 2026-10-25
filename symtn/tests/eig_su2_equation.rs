//! `TensorMap::eig_full` on a non-Abelian, non-Hermitian endomorphism: in
//! every coupled sector `c`, `block(t, c) block(v, c) = block(v, c)
//! diag(d_c)` with nonzero eigenvector columns, and `d_c` is the whole
//! spectrum of `block(t, c)`, checked through the power sums
//! `tr(block(t, c)^k) = Σ d_c^k` for `k = 1..=n_c`, which fix the multiset.
//!
//! The SU(2) legs carry several irreps with degeneracy, so each coupled
//! block of the rank-(2, 2) map stacks several fusion trees. The payload is
//! an arbitrary non-symmetric fill; the oracle reads the coupled blocks back
//! and multiplies them densely here, with no symtn contraction.

use std::sync::Arc;

use num_complex::Complex64;
use symtn::sector::{SU2FusionRule, SU2Irrep};
use symtn::typed::{Eig, GradedSpace, TensorMap};

use crate::fixtures;
use crate::numerics;

use fixtures::host_runtime;

type Map<D> = TensorMap<SU2FusionRule, D>;

fn leg() -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [(0, 2), (1, 1), (2, 1)]
            .map(|(twice, degeneracy)| (SU2Irrep::from_twice_spin(twice), degeneracy)),
    )
    .unwrap()
}

/// The `k`-th stored element of a deterministic, non-Hermitian fill.
fn entry(k: usize) -> Complex64 {
    let k = k as f64;
    Complex64::new((0.7 * k + 0.1).sin(), (1.3 * k).cos())
}

fn general<D>(lift: impl Fn(Complex64) -> D) -> Map<D>
where
    D: symtn::typed::TensorScalar,
{
    let space = leg();
    let mut k = 0;
    TensorMap::from_subblock_fn(
        &host_runtime(),
        [&space, &space],
        [&space, &space],
        |_, _| {
            k += 1;
            lift(entry(k))
        },
    )
    .unwrap()
}

/// Column-major `n × n` product.
fn matmul(a: &[Complex64], b: &[Complex64], n: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); n * n];
    for col in 0..n {
        for row in 0..n {
            out[row + n * col] = (0..n).map(|k| a[row + n * k] * b[k + n * col]).sum();
        }
    }
    out
}

fn dense_block(t: &Map<Complex64>, sector: &SU2Irrep) -> (usize, Vec<Complex64>) {
    let block = t.block(sector).unwrap();
    let n = block.rows();
    assert_eq!(block.cols(), n, "an endomorphism block is square");
    let data = (0..n * n)
        .map(|index| block.get(index % n, index / n).unwrap())
        .collect();
    (n, data)
}

fn assert_eig_full(t: &Map<Complex64>, Eig { d, v }: Eig<Map<Complex64>>) {
    let spectra = d.diagview().unwrap();
    let sectors: Vec<SU2Irrep> = t.blocks().unwrap().map(|(sector, _)| sector).collect();
    assert_eq!(
        spectra.iter().map(|s| s.sector).collect::<Vec<_>>(),
        sectors,
        "one spectrum per coupled sector"
    );
    let mut multi_tree = false;
    for spectrum in &spectra {
        let (n, a) = dense_block(t, &spectrum.sector);
        let (vn, vectors) = dense_block(&v, &spectrum.sector);
        let w = &spectrum.values;
        assert_eq!((vn, w.len()), (n, n));
        multi_tree |= t
            .block(&spectrum.sector)
            .unwrap()
            .row_trees()
            .unwrap()
            .len()
            > 1;

        for col in 0..n {
            let norm: f64 = vectors[n * col..n * (col + 1)]
                .iter()
                .map(|x| x.norm_sqr())
                .sum();
            assert!(
                norm > 0.5,
                "{:?}: eigenvector {col} is (nearly) zero",
                spectrum.sector
            );
        }
        let av = matmul(&a, &vectors, n);
        let vd: Vec<Complex64> = (0..n * n).map(|i| vectors[i] * w[i / n]).collect();
        numerics::assert_nonzero_slices_close("block(t) block(v) = block(v) diag(d)", &av, &vd, n);

        let frobenius = a.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        let mut power = a.clone();
        for k in 1..=n {
            let trace: Complex64 = (0..n).map(|i| power[i + n * i]).sum();
            let sum: Complex64 = w.iter().map(|x| x.powu(k as u32)).sum();
            let bound = numerics::tolerance::<Complex64>(n.pow(k as u32), frobenius.powi(k as i32));
            assert!(
                (trace - sum).norm() <= bound,
                "{:?}: tr(A^{k}) = {trace} but the eigenvalue power sum is {sum}",
                spectrum.sector
            );
            power = matmul(&power, &a, n);
        }
    }
    assert!(
        multi_tree,
        "the fixture must stack several fusion trees in a block"
    );
}

#[test]
fn complex_su2_eig_full_diagonalizes_every_multi_tree_coupled_block() {
    let t = general::<Complex64>(|x| x);
    let eig = t.eig_full(&[0, 1], &[2, 3]).unwrap();
    assert_eig_full(&t, eig);
}

#[test]
fn real_su2_eig_full_diagonalizes_every_multi_tree_coupled_block() {
    let t = general::<f64>(|x| x.re);
    let eig = t.eig_full(&[0, 1], &[2, 3]).unwrap();
    assert_eig_full(&t.convert::<Complex64>(), eig);
}
