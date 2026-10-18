//! `exp` takes the Hermitian spectral route `V exp(D) Vᴴ` exactly when every
//! coupled-sector block passes the `64 eps(D)` relative Hermitian predicate,
//! in both fusion modes, and Padé [13/13] otherwise (#1799:
//! TensorKit `exp!` per block, whose `LinearAlgebra.exp!` takes
//! `exp(Hermitian(A))` and `copytri!`s it exactly Hermitian).
//!
//! Oracles: a Taylor series of each block evaluated here in `Complex64`
//! (independent of both symtn routes), and the spectral route's exact
//! Hermiticity, which Padé of a non-Hermitian block does not have.

use std::sync::Arc;

use num_complex::Complex64;
use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{GradedSpace, Runtime, TensorMap};

const N: usize = 3;

/// Symmetric part of every test block (diagonally dominant, norm ~ 0.5).
fn s(i: usize, j: usize) -> f64 {
    0.1 * (1.0 + ((i + j) % 3) as f64) * if i == j { 2.0 } else { 0.5 }
}

/// Antisymmetric part of every test block.
fn k(i: usize, j: usize) -> f64 {
    0.1 * (i as f64 - j as f64)
}

fn frobenius(f: fn(usize, usize) -> f64) -> f64 {
    (0..N)
        .flat_map(|i| (0..N).map(move |j| f(i, j).powi(2)))
        .sum::<f64>()
        .sqrt()
}

trait Scalar: Copy {
    fn to_c64(self) -> Complex64;
    fn from_parts(re: f64, im: f64) -> Self;
}

impl Scalar for f64 {
    fn to_c64(self) -> Complex64 {
        Complex64::new(self, 0.0)
    }
    fn from_parts(re: f64, _im: f64) -> Self {
        re
    }
}

impl Scalar for Complex64 {
    fn to_c64(self) -> Complex64 {
        self
    }
    fn from_parts(re: f64, im: f64) -> Self {
        Complex64::new(re, im)
    }
}

/// `exp` of a column-major `N x N` matrix by its Taylor series, accurate to
/// rounding for the norms used here (`||A|| < 1`, 40 terms).
fn taylor_exp(a: &[Complex64]) -> Vec<Complex64> {
    let mut result = vec![Complex64::new(0.0, 0.0); N * N];
    let mut term = vec![Complex64::new(0.0, 0.0); N * N];
    for i in 0..N {
        result[i + N * i] = Complex64::new(1.0, 0.0);
        term[i + N * i] = Complex64::new(1.0, 0.0);
    }
    for order in 1..40 {
        let mut next = vec![Complex64::new(0.0, 0.0); N * N];
        for col in 0..N {
            for row in 0..N {
                for inner in 0..N {
                    next[row + N * col] += term[row + N * inner] * a[inner + N * col];
                }
                next[row + N * col] /= order as f64;
            }
        }
        for (r, t) in result.iter_mut().zip(&next) {
            *r += t;
        }
        term = next;
    }
    result
}

/// Every stored block of `image`, column-major, widened.
fn blocks<R, D>(image: &TensorMap<R, D>) -> Vec<Vec<Complex64>>
where
    R: symtn::sector::TypedSectorAdmission,
    R::Mode: symtn::typed::TypedTensorModeDispatch<R>,
    D: symtn::typed::TensorScalar + Scalar,
{
    image
        .blocks()
        .unwrap()
        .map(|(_, block)| {
            assert_eq!((block.rows(), block.cols()), (N, N));
            (0..N * N)
                .map(|index| block.get(index % N, index / N).unwrap().to_c64())
                .collect()
        })
        .collect()
}

fn exactly_hermitian(block: &[Complex64]) -> bool {
    (0..N).all(|col| (0..N).all(|row| block[row + N * col] == block[col + N * row].conj()))
}

/// Each block of `image` equals the Taylor `exp` of `fill`'s block.
fn assert_matches_taylor(
    what: &str,
    image: &[Vec<Complex64>],
    fill: &dyn Fn(usize, usize) -> Complex64,
) {
    let source = (0..N * N)
        .map(|index| fill(index % N, index / N))
        .collect::<Vec<_>>();
    let expected = taylor_exp(&source);
    for block in image {
        for (actual, expected) in block.iter().zip(&expected) {
            assert!(
                (actual - expected).norm() <= 1e-13,
                "{what}: {actual} vs {expected}"
            );
        }
    }
}

fn u1_leg() -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [-1, 0, 1].map(|charge| (U1Irrep::new(charge), N)),
    )
    .unwrap()
}

macro_rules! exp_of {
    ($runtime:expr, $leg:expr, $dtype:ty, $fill:expr) => {{
        let fill = $fill;
        let tensor: TensorMap<_, $dtype> =
            TensorMap::from_subblock_fn($runtime, [$leg], [$leg], |_, index| {
                let value: Complex64 = fill(index[0], index[1]);
                <$dtype as Scalar>::from_parts(value.re, value.im)
            })
            .unwrap();
        blocks(&tensor.exp(&[0], &[1]).unwrap())
    }};
}

fn hermitian_real(i: usize, j: usize) -> Complex64 {
    Complex64::new(s(i, j), 0.0)
}
fn hermitian_complex(i: usize, j: usize) -> Complex64 {
    Complex64::new(s(i, j), k(i, j))
}
fn general_real(i: usize, j: usize) -> Complex64 {
    Complex64::new(s(i, j) + 0.3 * k(i, j), 0.0)
}
fn general_complex(i: usize, j: usize) -> Complex64 {
    Complex64::new(s(i, j) + 0.3 * k(i, j), 0.2 * s(i, j))
}

#[test]
fn multiplicity_free_exp_routes_by_the_hermitian_predicate() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = u1_leg();
    for (what, image, fill, hermitian) in [
        (
            "f64 Hermitian",
            exp_of!(&runtime, &leg, f64, hermitian_real),
            hermitian_real as fn(_, _) -> _,
            true,
        ),
        (
            "c64 Hermitian",
            exp_of!(&runtime, &leg, Complex64, hermitian_complex),
            hermitian_complex,
            true,
        ),
        (
            "f64 general",
            exp_of!(&runtime, &leg, f64, general_real),
            general_real,
            false,
        ),
        (
            "c64 general",
            exp_of!(&runtime, &leg, Complex64, general_complex),
            general_complex,
            false,
        ),
    ] {
        assert_matches_taylor(what, &image, &fill);
        if hermitian {
            assert!(image.iter().all(|block| exactly_hermitian(block)), "{what}");
        }
    }
}

/// `S + t K` with `||t K||_F = scale * 64 eps ||S||_F`: below the predicate
/// at `scale = 0.5` (spectral route, exactly Hermitian), above it at `2`
/// (Padé, which keeps the input's small anti-Hermitian part).
fn boundary_fill(scale: f64) -> impl Fn(usize, usize) -> Complex64 {
    let t = scale * 64.0 * f64::EPSILON * frobenius(s) / frobenius(k);
    move |i, j| Complex64::new(s(i, j) + t * k(i, j), 0.0)
}

#[test]
fn multiplicity_free_exp_route_switches_at_the_predicate_boundary() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = u1_leg();
    let below = exp_of!(&runtime, &leg, f64, boundary_fill(0.5));
    assert!(
        below.iter().all(|block| exactly_hermitian(block)),
        "below: spectral"
    );
    assert_matches_taylor("below", &below, &boundary_fill(0.5));
    let above = exp_of!(&runtime, &leg, f64, boundary_fill(2.0));
    assert!(
        above.iter().all(|block| !exactly_hermitian(block)),
        "above: Padé"
    );
    assert_matches_taylor("above", &above, &boundary_fill(2.0));
}

/// The same blocks through multiplicity-free SU(2) and checked SU(2) (fusion-symbols
/// `SU(N=2)`): both modes take the same route and agree to rounding.
#[cfg(feature = "cgc-gen")]
#[test]
fn checked_and_multiplicity_free_exp_take_the_same_route() {
    use symtn::sector::{SU2FusionRule, SU2Irrep, SUNFusionRule};

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let mf = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [0, 1, 2].map(|twice| (SU2Irrep::from_twice_spin(twice), N)),
    )
    .unwrap();
    let checked = GradedSpace::try_new(
        Arc::new(SUNFusionRule::new(2).unwrap()),
        [0i64, 1, 2].map(|twice| (vec![twice], N)),
    )
    .unwrap();
    macro_rules! both {
        ($dtype:ty, $fill:expr, $hermitian:expr, $what:expr) => {{
            let mf = exp_of!(&runtime, &mf, $dtype, $fill);
            let checked = exp_of!(&runtime, &checked, $dtype, $fill);
            assert_matches_taylor($what, &mf, &$fill);
            assert_matches_taylor($what, &checked, &$fill);
            for (a, b) in mf.iter().zip(&checked) {
                for (a, b) in a.iter().zip(b) {
                    assert!((a - b).norm() <= 1e-14, "{}: {a} vs {b}", $what);
                }
            }
            for image in [&mf, &checked] {
                assert_eq!(
                    image.iter().all(|block| exactly_hermitian(block)),
                    $hermitian,
                    "{}",
                    $what
                );
            }
        }};
    }
    both!(f64, hermitian_real, true, "f64 Hermitian");
    both!(Complex64, hermitian_complex, true, "c64 Hermitian");
    both!(f64, general_real, false, "f64 general");
    both!(Complex64, general_complex, false, "c64 general");
    both!(f64, boundary_fill(0.5), true, "below the predicate");
    both!(f64, boundary_fill(2.0), false, "above the predicate");
}
