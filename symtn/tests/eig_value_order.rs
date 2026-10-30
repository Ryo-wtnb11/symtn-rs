//! The published eigenvalue order (#1985), against an independent oracle.
//!
//! Every fixture is built from hand-chosen eigenvalues: Hermitian blocks as
//! `U diag(λ) U^H` with a fixed unitary `U`, general blocks (quasi-)upper
//! triangular, so each spectrum is known without an eigensolver. The expected
//! order is constructed here, with no symtn code: `eigh` ascending, `eig`
//! ascending lexicographic `(re, im)`. The fixtures make this order differ
//! from the former descending `|λ|` order, with a `±λ` pair, equal real parts
//! and conjugate pairs. Eigenvector columns are checked to follow the values
//! through the eigen relation `t v = v d`, on the dense and the compact
//! diagonal route, values-only and full, for every factorization dtype.

use std::sync::Arc;

use num_complex::{Complex32, Complex64};
use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{Eig, Eigh, GradedSpace, HermitianTol, SectorSpectrum, TensorMap};

use crate::fixtures;

use fixtures::host_runtime;

trait Payload: Copy {
    type Eig: Copy;
    const COMPLEX: bool;
    const TOL: f64;
    fn lift(value: Complex64) -> Self;
    fn widen(self) -> Complex64;
    fn widen_eig(value: Self::Eig) -> Complex64;
    fn to_eig(t: &TensorMap<U1FusionRule, Self>) -> TensorMap<U1FusionRule, Self::Eig>;
}

impl Payload for f32 {
    type Eig = Complex32;
    const COMPLEX: bool = false;
    const TOL: f64 = 1e-4;
    fn lift(value: Complex64) -> Self {
        value.re as f32
    }
    fn widen(self) -> Complex64 {
        Complex64::new(self.into(), 0.0)
    }
    fn widen_eig(value: Complex32) -> Complex64 {
        Complex64::new(value.re.into(), value.im.into())
    }
    fn to_eig(t: &TensorMap<U1FusionRule, Self>) -> TensorMap<U1FusionRule, Complex32> {
        t.convert(&mut symtn_dense::CpuEngine::serial()).unwrap()
    }
}

impl Payload for f64 {
    type Eig = Complex64;
    const COMPLEX: bool = false;
    const TOL: f64 = 1e-12;
    fn lift(value: Complex64) -> Self {
        value.re
    }
    fn widen(self) -> Complex64 {
        Complex64::new(self, 0.0)
    }
    fn widen_eig(value: Complex64) -> Complex64 {
        value
    }
    fn to_eig(t: &TensorMap<U1FusionRule, Self>) -> TensorMap<U1FusionRule, Complex64> {
        t.convert(&mut symtn_dense::CpuEngine::serial()).unwrap()
    }
}

impl Payload for Complex32 {
    type Eig = Complex32;
    const COMPLEX: bool = true;
    const TOL: f64 = 1e-4;
    fn lift(value: Complex64) -> Self {
        Complex32::new(value.re as f32, value.im as f32)
    }
    fn widen(self) -> Complex64 {
        Complex64::new(self.re.into(), self.im.into())
    }
    fn widen_eig(value: Complex32) -> Complex64 {
        value.widen()
    }
    fn to_eig(t: &TensorMap<U1FusionRule, Self>) -> TensorMap<U1FusionRule, Complex32> {
        t.clone()
    }
}

impl Payload for Complex64 {
    type Eig = Complex64;
    const COMPLEX: bool = true;
    const TOL: f64 = 1e-12;
    fn lift(value: Complex64) -> Self {
        value
    }
    fn widen(self) -> Complex64 {
        self
    }
    fn widen_eig(value: Complex64) -> Complex64 {
        value
    }
    fn to_eig(t: &TensorMap<U1FusionRule, Self>) -> TensorMap<U1FusionRule, Complex64> {
        t.clone()
    }
}

fn charge(sector: &U1Irrep) -> i32 {
    i32::from(*sector != U1Irrep::new(0))
}

fn leg(dims: [usize; 2]) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), dims[0]), (U1Irrep::new(1), dims[1])],
    )
    .unwrap()
}

/// Hand Hermitian eigenvalues per sector, in no particular order.
fn eigh_values(charge: i32) -> Vec<f64> {
    if charge == 0 {
        // Magnitude order would be [-3, 2, 0.5].
        vec![2.0, -3.0, 0.5]
    } else {
        vec![1.0, -1.0]
    }
}

/// Hand general eigenvalues per sector of [`general_entry`].
fn eig_values(charge: i32, complex: bool) -> Vec<Complex64> {
    let c = Complex64::new;
    match (charge, complex) {
        // A conjugate pair with equal real parts; magnitude order would be
        // [1+2i, 1-2i, -1.5, 0.5i].
        (0, true) => vec![c(1.0, 2.0), c(-1.5, 0.0), c(1.0, -2.0), c(0.0, 0.5)],
        (_, true) => vec![c(2.0, 0.0), c(0.0, -2.0)],
        (0, false) => vec![c(1.0, 2.0), c(1.0, -2.0), c(-1.5, 0.0)],
        (_, false) => vec![c(2.0, 0.0), c(-3.0, 0.0)],
    }
}

fn ascending(mut values: Vec<f64>) -> Vec<f64> {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values
}

fn lexicographic(mut values: Vec<Complex64>) -> Vec<Complex64> {
    values.sort_by(|a, b| (a.re, a.im).partial_cmp(&(b.re, b.im)).unwrap());
    values
}

/// `diag(e^{i 0.3 k}) (I - 2 u u^T / u^T u)` with `u = (1, 2, ..)`; real
/// payloads drop the phases.
fn unitary(n: usize, complex: bool) -> Vec<Vec<Complex64>> {
    let u: Vec<f64> = (1..=n).map(|k| k as f64).collect();
    let norm: f64 = u.iter().map(|x| x * x).sum();
    (0..n)
        .map(|i| {
            let phase = if complex {
                Complex64::from_polar(1.0, 0.3 * i as f64)
            } else {
                Complex64::new(1.0, 0.0)
            };
            (0..n)
                .map(|j| phase * (f64::from(u8::from(i == j)) - 2.0 * u[i] * u[j] / norm))
                .collect()
        })
        .collect()
}

fn hermitian_entry(charge: i32, i: usize, j: usize, complex: bool) -> Complex64 {
    let values = eigh_values(charge);
    let q = unitary(values.len(), complex);
    (0..values.len())
        .map(|k| q[i][k] * values[k] * q[j][k].conj())
        .sum()
}

/// Complex payloads: upper triangular with the [`eig_values`] diagonal.
/// Real payloads: the rotation-scaling block `[[1, -2], [2, 1]]` (`1 ± 2i`)
/// above `-1.5` in sector 0, triangular `[[2, 0.7], [0, -3]]` in sector 1.
fn general_entry(charge: i32, i: usize, j: usize, complex: bool) -> Complex64 {
    if complex {
        return match i.cmp(&j) {
            std::cmp::Ordering::Equal => eig_values(charge, true)[i],
            std::cmp::Ordering::Less => Complex64::new(0.25 * (i + 2 * j) as f64, -0.1 * j as f64),
            std::cmp::Ordering::Greater => Complex64::new(0.0, 0.0),
        };
    }
    let rows: [[f64; 3]; 3] = if charge == 0 {
        [[1.0, -2.0, 0.3], [2.0, 1.0, -0.4], [0.0, 0.0, -1.5]]
    } else {
        [[2.0, 0.7, 0.0], [0.0, -3.0, 0.0], [0.0; 3]]
    };
    Complex64::new(rows[i][j], 0.0)
}

/// `left` and `right` share one space, so their payloads align.
fn assert_relation(left: &[Complex64], right: &[Complex64], tol: f64, what: &str) {
    assert_eq!(left.len(), right.len(), "{what}");
    for (a, b) in left.iter().zip(right) {
        assert!((a - b).norm() <= tol, "{what}: {a} vs {b}");
    }
}

macro_rules! check_dtype {
    ($engine:ident, $dtype:ty) => {{
        type D = $dtype;
        let runtime = host_runtime();
        let complex = <D as Payload>::COMPLEX;
        let tol = <D as Payload>::TOL;
        let widen = |data: &[D]| data.iter().map(|&x| x.widen()).collect::<Vec<_>>();
        let widen_eig = |data: &[<D as Payload>::Eig]| {
            data.iter()
                .map(|&x| <D as Payload>::widen_eig(x))
                .collect::<Vec<_>>()
        };

        // Hermitian, dense route.
        let space = leg([3, 2]);
        let h: TensorMap<_, D> =
            TensorMap::from_subblock_fn($engine, &runtime, [&space], [&space], |trees, index| {
                D::lift(hermitian_entry(
                    charge(trees.coupled()),
                    index[0],
                    index[1],
                    complex,
                ))
            })
            .unwrap();
        let Eigh { d, v } = h
            .eigh_full($engine, &[0], &[1], HermitianTol::DEFAULT)
            .unwrap();
        let spectra = d.diagview($engine).unwrap();
        let values = h
            .eigh_vals($engine, &[0], &[1], HermitianTol::DEFAULT)
            .unwrap();
        assert_eq!(spectra.len(), 2);
        for (entry, vals) in spectra.iter().zip(&values) {
            assert_eq!(entry.sector, vals.sector);
            let want = ascending(eigh_values(charge(&entry.sector)));
            assert_eq!(entry.values.len(), want.len());
            for ((&got, &val), &want) in entry.values.iter().zip(&vals.values).zip(&want) {
                assert!((got.widen() - want).norm() <= tol, "eigh {got:?} vs {want}");
                assert!((val - want).abs() <= tol, "eigh_vals {val} vs {want}");
            }
        }
        assert_relation(
            &widen(h.compose($engine, &v).unwrap().dense_data().unwrap()),
            &widen(v.compose($engine, &d).unwrap().dense_data().unwrap()),
            tol,
            "eigh t v = v d",
        );

        // General, dense route.
        let space = leg([eig_values(0, complex).len(), 2]);
        let g: TensorMap<_, D> =
            TensorMap::from_subblock_fn($engine, &runtime, [&space], [&space], |trees, index| {
                D::lift(general_entry(
                    charge(trees.coupled()),
                    index[0],
                    index[1],
                    complex,
                ))
            })
            .unwrap();
        let Eig { d, v } = g.eig_full($engine, &[0], &[1]).unwrap();
        let spectra = d.diagview($engine).unwrap();
        let values = g.eig_vals($engine, &[0], &[1]).unwrap();
        assert_eq!(spectra.len(), 2);
        for (entry, vals) in spectra.iter().zip(&values) {
            let want = lexicographic(eig_values(charge(&entry.sector), complex));
            assert_eq!(entry.values.len(), want.len());
            for ((&got, &val), &want) in entry.values.iter().zip(&vals.values).zip(&want) {
                let got = <D as Payload>::widen_eig(got);
                assert!((got - want).norm() <= tol, "eig {got} vs {want}");
                assert!((val - want).norm() <= tol, "eig_vals {val} vs {want}");
            }
        }
        let lifted = <D as Payload>::to_eig(&g);
        assert_relation(
            &widen_eig(lifted.compose($engine, &v).unwrap().dense_data().unwrap()),
            &widen_eig(v.compose($engine, &d).unwrap().dense_data().unwrap()),
            tol,
            "eig t v = v d",
        );

        // Compact diagonal, EIGH: the published order sorts the stored one,
        // and `v` is that permutation.
        let space = leg([3, 2]);
        let compact: TensorMap<_, D> = TensorMap::diagonal(
            $engine,
            &runtime,
            &space,
            [0, 1].map(|charge| SectorSpectrum {
                sector: U1Irrep::new(charge),
                values: eigh_values(charge)
                    .into_iter()
                    .map(|value| D::lift(Complex64::new(value, 0.0)))
                    .collect(),
            }),
        )
        .unwrap();
        let Eigh { d, v } = compact
            .eigh_full($engine, &[0], &[1], HermitianTol::DEFAULT)
            .unwrap();
        let values = compact
            .eigh_vals($engine, &[0], &[1], HermitianTol::DEFAULT)
            .unwrap();
        for (entry, vals) in d.diagview($engine).unwrap().iter().zip(&values) {
            let want = ascending(eigh_values(charge(&entry.sector)));
            let got: Vec<f64> = entry.values.iter().map(|&x| x.widen().re).collect();
            assert_eq!(got, want);
            assert_eq!(vals.values, want);
        }
        assert_relation(
            &widen(compact.compose($engine, &v).unwrap().dense_data().unwrap()),
            &widen(v.compose($engine, &d).unwrap().dense_data().unwrap()),
            0.0,
            "compact eigh t v = v d",
        );

        // Compact diagonal, EIG.
        let stored = |charge: i32| -> Vec<Complex64> {
            if complex {
                eig_values(charge, true)
            } else {
                eigh_values(charge)
                    .into_iter()
                    .map(|value| Complex64::new(value, 0.0))
                    .collect()
            }
        };
        let space = leg([stored(0).len(), 2]);
        let compact: TensorMap<_, D> = TensorMap::diagonal(
            $engine,
            &runtime,
            &space,
            [0, 1].map(|charge| SectorSpectrum {
                sector: U1Irrep::new(charge),
                values: stored(charge).into_iter().map(D::lift).collect(),
            }),
        )
        .unwrap();
        let Eig { d, v } = compact.eig_full($engine, &[0], &[1]).unwrap();
        let values = compact.eig_vals($engine, &[0], &[1]).unwrap();
        for (entry, vals) in d.diagview($engine).unwrap().iter().zip(&values) {
            let want = lexicographic(stored(charge(&entry.sector)));
            let got: Vec<Complex64> = entry
                .values
                .iter()
                .map(|&x| <D as Payload>::widen_eig(x))
                .collect();
            assert_eq!(got, want);
            assert_eq!(vals.values, want);
        }
        let lifted = <D as Payload>::to_eig(&compact);
        assert_relation(
            &widen_eig(lifted.compose($engine, &v).unwrap().dense_data().unwrap()),
            &widen_eig(v.compose($engine, &d).unwrap().dense_data().unwrap()),
            0.0,
            "compact eig t v = v d",
        );
    }};
}

#[test]
fn f32_eigenvalues_follow_the_published_order() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    check_dtype!(engine, f32);
}

#[test]
fn f64_eigenvalues_follow_the_published_order() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    check_dtype!(engine, f64);
}

#[test]
fn complex32_eigenvalues_follow_the_published_order() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    check_dtype!(engine, Complex32);
}

#[test]
fn complex64_eigenvalues_follow_the_published_order() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    check_dtype!(engine, Complex64);
}
