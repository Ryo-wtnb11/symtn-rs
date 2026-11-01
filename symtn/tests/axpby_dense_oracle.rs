//! `x.axpby(alpha, &y, beta) == alpha * x + beta * y` against an oracle that
//! never calls `axpby`: the physical dense expansion for U(1) and SU(2), and
//! the per-entry block values for the Abelian fZ2×U(1), where every block
//! entry is a dense entry. The coefficients are asymmetric (and complex for
//! `Complex64`), so a swapped binding such as VectorInterface's
//! `add(y, x, α, β) == β·y + α·x` fails every case.
//!
//! The CUDA gate runs with `cargo test -p symtn --features cuda,cpu-faer
//! --test integration axpby_dense_oracle:: -- --ignored` on a CUDA host.

use std::collections::HashMap;

use num_complex::Complex64;
use symtn::sector::{
    FermionParityFusionRule, ProductFusionRule, ProductSector, SU2FusionRule, SU2Irrep,
    U1FusionRule, U1Irrep, Z2Irrep,
};
use symtn::typed::{CpuEngine, Engine, GradedSpace, Runtime, TensorMap};

use crate::numerics;

const F64_COEFFS: (f64, f64) = (0.75, -2.5);
const C64_ALPHA: Complex64 = Complex64::new(0.5, -2.0);
const C64_BETA: Complex64 = Complex64::new(-1.5, 0.25);

trait Sample: numerics::Numeric + std::ops::Mul<Output = Self> + std::ops::Add<Output = Self> {
    fn sample(k: usize, salt: usize) -> Self;
}

impl Sample for f64 {
    fn sample(k: usize, salt: usize) -> Self {
        ((k * 7 + salt * 5) % 13) as f64 / 3.0 - 2.0
    }
}

impl Sample for Complex64 {
    fn sample(k: usize, salt: usize) -> Self {
        Complex64::new(f64::sample(k, salt), f64::sample(k + 3, salt + 1))
    }
}

/// Fills a tensor from a counter so `x` and `y` carry unrelated values.
macro_rules! filled {
    ($engine:ident, $rt:expr, $codomain:expr, $domain:expr, $d:ty, $salt:expr) => {{
        let mut k = 0usize;
        TensorMap::<_, $d>::from_subblock_fn($engine, $rt, $codomain, $domain, |_, _| {
            k += 1;
            <$d as Sample>::sample(k, $salt)
        })
        .unwrap()
    }};
}

fn assert_linear_combination<D: Sample>(what: &str, z: &[D], x: &[D], y: &[D], a: D, b: D) {
    let want: Vec<D> = x.iter().zip(y).map(|(&x, &y)| a * x + b * y).collect();
    numerics::assert_slices_close(what, z, &want, 2);
}

/// The physical dense expansion of `z` against `alpha * X + beta * Y`,
/// computed on the expansions of `x` and `y`.
macro_rules! physical_case {
    ($engine:ident, $what:expr, $x:expr, $y:expr, $z:expr, $alpha:expr, $beta:expr) => {{
        let x = $x.to_physical_dense($engine).unwrap();
        let y = $y.to_physical_dense($engine).unwrap();
        let z = $z.to_physical_dense($engine).unwrap();
        assert_eq!(z.shape, x.shape);
        assert_linear_combination($what, &z.data, &x.data, &y.data, $alpha, $beta);
    }};
}

/// Every entry of `z`, addressed by its fusion trees and block index,
/// against the same entries of `x` and `y`.
macro_rules! block_case {
    ($engine:ident, $what:expr, $x:expr, $y:expr, $z:expr, $alpha:expr, $beta:expr) => {{
        let entries = |t: &TensorMap<_, _>| {
            let mut out = HashMap::new();
            for (trees, block) in t.subblocks().unwrap() {
                let shape = block.shape().to_vec();
                let len: usize = shape.iter().product();
                for linear in 0..len {
                    let mut index = Vec::with_capacity(shape.len());
                    let mut rest = linear;
                    for &n in &shape {
                        index.push(rest % n);
                        rest /= n;
                    }
                    let value = *block.get(&index).unwrap();
                    out.insert((format!("{trees:?}"), index), value);
                }
            }
            out
        };
        let (x, y, z) = (entries(&$x), entries(&$y), entries(&$z));
        assert_eq!(z.len(), x.len());
        let mut keys: Vec<_> = z.keys().cloned().collect();
        keys.sort();
        let pick = |m: &HashMap<_, _>| keys.iter().map(|k| m[k]).collect::<Vec<_>>();
        assert_linear_combination($what, &pick(&z), &pick(&x), &pick(&y), $alpha, $beta);
    }};
}

fn u1_legs() -> (GradedSpace<U1FusionRule>, GradedSpace<U1FusionRule>) {
    let v = GradedSpace::try_new(
        std::sync::Arc::new(U1FusionRule),
        [(-1, 2), (0, 1), (1, 3)].map(|(q, n)| (U1Irrep::new(q), n)),
    )
    .unwrap();
    let w = v.try_dual().unwrap();
    (v, w)
}

fn su2_legs() -> (GradedSpace<SU2FusionRule>, GradedSpace<SU2FusionRule>) {
    let v = GradedSpace::try_new(
        std::sync::Arc::new(SU2FusionRule),
        [(0, 2), (1, 2), (2, 1)].map(|(s, n)| (SU2Irrep::from_twice_spin(s), n)),
    )
    .unwrap();
    let w = v.try_dual().unwrap();
    (v, w)
}

type Fz2U1 = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;

fn fz2u1_legs() -> (GradedSpace<Fz2U1>, GradedSpace<Fz2U1>) {
    let sector = |q: i32| {
        let parity = if q.rem_euclid(2) == 0 {
            Z2Irrep::EVEN
        } else {
            Z2Irrep::ODD
        };
        ProductSector::new(parity, U1Irrep::new(q))
    };
    let v = GradedSpace::try_new(
        std::sync::Arc::new(Fz2U1::try_new(FermionParityFusionRule, U1FusionRule).unwrap()),
        [(-1, 2), (0, 1), (1, 2), (2, 1)].map(|(q, n)| (sector(q), n)),
    )
    .unwrap();
    let w = v.try_dual().unwrap();
    (v, w)
}

/// Runs one symmetry for both payload dtypes. `$check` is the oracle macro,
/// `$run` maps `(x, y, alpha, beta)` to a Host `z` (directly or via CUDA).
macro_rules! symmetry {
    ($engine:ident, $rt:expr, $legs:expr, $check:ident, $run:expr) => {{
        let (v, w) = $legs;
        {
            let (a, b) = F64_COEFFS;
            let x = filled!($engine, $rt, [&v, &w], [&v], f64, 1);
            let y = filled!($engine, $rt, [&v, &w], [&v], f64, 2);
            let z = $run($engine, &x, &y, a, b);
            $check!($engine, "f64", x, y, z, a, b);
        }
        {
            let (a, b) = (C64_ALPHA, C64_BETA);
            let x = filled!($engine, $rt, [&v, &w], [&v], Complex64, 1);
            let y = filled!($engine, $rt, [&v, &w], [&v], Complex64, 2);
            let z = $run($engine, &x, &y, a, b);
            $check!($engine, "c64", x, y, z, a, b);
        }
    }};
}

macro_rules! host {
    ($engine:ident) => {
        |engine: &mut dyn Engine, x: &TensorMap<_, _>, y: &TensorMap<_, _>, a, b| {
            x.axpby(engine, a, y, b).unwrap()
        }
    };
}

#[test]
fn axpby_matches_dense_expansion_on_host() {
    let engine = &mut CpuEngine::serial();
    let rt = Runtime::builder().build().unwrap();
    symmetry!(engine, &rt, u1_legs(), physical_case, host!(engine));
    symmetry!(engine, &rt, su2_legs(), physical_case, host!(engine));
    symmetry!(engine, &rt, fz2u1_legs(), block_case, host!(engine));
}

#[cfg(feature = "cuda")]
macro_rules! device {
    ($cuda:ident) => {
        |_engine: &mut dyn Engine, x: &TensorMap<_, _>, y: &TensorMap<_, _>, a, b| {
            let (xd, yd) = (x.to_cuda($cuda).unwrap(), y.to_cuda($cuda).unwrap());
            xd.axpby($cuda, a, &yd, b).unwrap().to_host($cuda).unwrap()
        }
    };
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA device"]
fn axpby_matches_dense_expansion_on_cuda() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut crate::common::cuda_engine();
    let rt = Runtime::builder().build().unwrap();
    symmetry!(engine, &rt, u1_legs(), physical_case, device!(cuda));
    symmetry!(engine, &rt, su2_legs(), physical_case, device!(cuda));
    symmetry!(engine, &rt, fz2u1_legs(), block_case, device!(cuda));
}
