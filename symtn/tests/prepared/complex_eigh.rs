//! Complex fixtures and the spectral oracle shared by the Host and device
//! `EighFullPlan` gates.
//!
//! Each coupled sector is a hand-made Hermitian `H = U diag(λ) Uᴴ`. A factor
//! pair is checked through quantities that hold in every gauge: the
//! eigenvalues, the reconstruction `v d vᴴ = t`, `vᴴ v = 1`, and the spectral
//! projector of every eigenvalue cluster. The placement-specific gauge
//! checks stay with each suite.

use std::fmt::Debug;

use num_complex::{Complex32, Complex64};
use symtn::sector::{CheckedFusionAlgebra, MultiplicityFreeRigidSymbols, SectorCodec};
use symtn::typed::{GradedSpace, Runtime, TensorMap};

use crate::truncation_oracle::Matrix;

use crate::prepared::members;

/// Complex members: the bounds of a payload the Host plan factors.
pub trait ComplexPayload: crate::common::Payload + symtn::typed::FactorizationScalar {}
impl ComplexPayload for Complex32 {}
impl ComplexPayload for Complex64 {}

/// One coupled sector's spectral reference: `values` ascending and the
/// unitary `vectors` whose column `k` is `values[k]`'s eigenvector.
pub struct SectorOracle<S> {
    pub sector: S,
    pub values: Vec<f64>,
    pub vectors: Matrix,
}

/// An `n x n` unitary with complex phases and unequal entry magnitudes: two
/// sweeps of complex Givens rotations `[[c, -s̄], [s, c]]` on adjacent
/// columns of the identity, angles drawn from `seed`.
fn givens_unitary(n: usize, seed: usize) -> Matrix {
    let mut u = Matrix::zeros(n, n);
    for i in 0..n {
        u.data[i + i * n] = Complex64::ONE;
    }
    let frac = |x: f64| x - x.floor();
    for sweep in 0..2 {
        for p in 0..n.saturating_sub(1) {
            let k = (seed * 31 + sweep * 17 + p * 7 + 1) as f64;
            let theta = 0.3 + 0.9 * frac(k * 0.618_033_988_75);
            let phi = std::f64::consts::TAU * frac(k * 0.414_213_562_37);
            let (c, s) = (theta.cos(), Complex64::from_polar(theta.sin(), phi));
            for r in 0..n {
                let (a, b) = (u.data[r + p * n], u.data[r + (p + 1) * n]);
                u.data[r + p * n] = a * c + b * s;
                u.data[r + (p + 1) * n] = b * c - a * s.conj();
            }
        }
    }
    u
}

/// The spectrum of coupled sector order `n` for `member`: exactly repeated
/// triples (even members) or triples clustered `2^-30` apart (odd members)
/// in the lower half, distinct values above, scaled by `2^-40`, `1` or
/// `2^40` so members of one batch differ by up to `2^80`.
fn spectrum(member: usize, salt: usize, n: usize) -> Vec<f64> {
    let scale = 2f64.powi(40 * ((member + salt) % 3) as i32 - 40);
    let cluster = if member % 2 == 1 { 2f64.powi(-30) } else { 0.0 };
    (0..n)
        .map(|k| {
            let group = if k < n / 2 { k / 3 } else { k };
            scale * (-1.0 + 0.4 * group as f64 + (k % 3) as f64 * cluster + 0.1 * salt as f64)
        })
        .collect()
}

/// `count` endomorphisms `leg ← leg` whose coupled sector of order `n` is
/// `U diag(spectrum) Uᴴ`, Hermitian by construction (the lower triangle is
/// the conjugate of the upper, the diagonal real), with the oracle of every
/// member.
#[allow(clippy::type_complexity)]
pub fn spectral_members<R, D>(
    runtime: &Runtime,
    leg: &GradedSpace<R>,
    count: usize,
    salt: usize,
) -> (
    Vec<TensorMap<R, D>>,
    Vec<Vec<SectorOracle<<R as SectorCodec>::Sector>>>,
)
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    <R as SectorCodec>::Sector: Clone + PartialEq,
    D: ComplexPayload,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    // Each coupled sector's order, read from the block indices.
    let mut orders: Vec<(<R as SectorCodec>::Sector, usize)> = Vec::new();
    TensorMap::<R, D>::from_subblock_fn(engine, runtime, [leg], [leg], |trees, index| {
        match orders.iter_mut().find(|(s, _)| s == trees.coupled()) {
            Some((_, n)) => *n = (*n).max(index[0] + 1),
            None => orders.push((trees.coupled().clone(), index[0] + 1)),
        }
        D::entry(0.0, 0.0)
    })
    .unwrap();
    let mut tensors = Vec::new();
    let mut oracles = Vec::new();
    for member in 0..count {
        let oracle: Vec<_> = orders
            .iter()
            .enumerate()
            .map(|(position, (sector, n))| SectorOracle {
                sector: sector.clone(),
                values: spectrum(member, salt, *n),
                vectors: givens_unitary(*n, 97 * member + 13 * position + 5 * salt),
            })
            .collect();
        let h = |o: &SectorOracle<_>, i: usize, j: usize| -> Complex64 {
            let u = &o.vectors;
            (0..u.rows)
                .map(|k| u.at(i, k) * o.values[k] * u.at(j, k).conj())
                .sum()
        };
        let tensor = TensorMap::from_subblock_fn(engine, runtime, [leg], [leg], |trees, index| {
            let o = oracle
                .iter()
                .find(|o| &o.sector == trees.coupled())
                .unwrap();
            let (i, j) = (index[0], index[1]);
            let z = match i.cmp(&j) {
                std::cmp::Ordering::Less => h(o, i, j),
                std::cmp::Ordering::Equal => Complex64::new(h(o, i, i).re, 0.0),
                std::cmp::Ordering::Greater => h(o, j, i).conj(),
            };
            D::entry(z.re, z.im)
        })
        .unwrap();
        tensors.push(tensor);
        oracles.push(oracle);
    }
    (tensors, oracles)
}

/// `count` complex Hermitian `X + Xᴴ` over `codomain ← codomain`: several
/// fusion trees per coupled sector.
pub fn complex_hermitian_members<R, D>(
    runtime: &Runtime,
    codomain: &[&GradedSpace<R>],
    count: usize,
    salt: usize,
) -> Vec<TensorMap<R, D>>
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: ComplexPayload,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let one = D::entry(1.0, 0.0);
    members::<R, D>(runtime, codomain, codomain, count, salt)
        .into_iter()
        .map(|x| {
            let adjoint = x.adjoint(engine).unwrap();
            x.axpby(engine, one, &adjoint, one).unwrap()
        })
        .collect()
}

/// A computed factor pair `(d, v)` as a spectral reference, per coupled
/// sector: another solver's output compared through [`check_spectral`].
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub fn spectral_reference<R, D>(
    d: &TensorMap<R, D>,
    v: &TensorMap<R, D>,
) -> Vec<SectorOracle<<R as SectorCodec>::Sector>>
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    <R as SectorCodec>::Sector: Clone + PartialEq,
    D: ComplexPayload,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let dm = crate::sector_matrices!(engine, d);
    crate::sector_matrices!(engine, v)
        .into_iter()
        .map(|(sector, vectors)| {
            let dd = &dm.iter().find(|(s, _)| s == &sector).unwrap().1;
            SectorOracle {
                values: (0..dd.rows).map(|k| dd.at(k, k).re).collect(),
                sector,
                vectors,
            }
        })
        .collect()
}

/// The gauge-free checks of one complex member's factors `(d, v)` of
/// `input`, per coupled sector:
///
/// - `d` ascending, `v d vᴴ = t` and `vᴴ v = 1`;
/// - with a reference: the eigenvalues equal its values, and the spectral
///   projector `Σ v_k v_kᴴ` of every eigenvalue cluster equals the
///   reference's.
///
/// Tolerances are `64 n eps` of the sector's largest `|λ|`, the projectors
/// further divided by the relative gap to the next cluster (Davis–Kahan).
/// Every sector of `reference` is checked, so it must list them all.
pub fn check_spectral<R, D>(
    what: &str,
    input: &TensorMap<R, D>,
    d: &TensorMap<R, D>,
    v: &TensorMap<R, D>,
    reference: Option<&[SectorOracle<<R as SectorCodec>::Sector>]>,
) where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    <R as SectorCodec>::Sector: Clone + PartialEq + Debug,
    D: ComplexPayload,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let eps = <D as crate::numerics::Numeric>::EPS;
    let t = crate::sector_matrices!(engine, input);
    let dm = crate::sector_matrices!(engine, d);
    let vm = crate::sector_matrices!(engine, v);
    assert_eq!(t.len(), vm.len(), "{what}: sectors");
    if let Some(reference) = reference {
        assert_eq!(reference.len(), t.len(), "{what}: reference sectors");
    }
    for (sector, a) in &t {
        let what = format!("{what} sector {sector:?}");
        let n = a.rows;
        let find = |list: &[(_, Matrix)]| {
            list.iter()
                .find(|(s, _)| s == sector)
                .map(|(_, m)| m.clone())
                .unwrap()
        };
        let (dd, vv) = (find(&dm), find(&vm));
        assert_eq!((vv.rows, vv.cols), (n, n), "{what}: v shape");
        let values: Vec<f64> = (0..n).map(|k| dd.at(k, k).re).collect();
        assert!(values.is_sorted(), "{what}: ascending {values:?}");
        let scale = values.iter().fold(f64::MIN_POSITIVE, |m, x| m.max(x.abs()));
        let bound = 64.0 * n as f64 * eps;
        for i in 0..n {
            for j in 0..n {
                let rebuilt: Complex64 = (0..n)
                    .map(|k| vv.at(i, k) * values[k] * vv.at(j, k).conj())
                    .sum();
                assert!(
                    (rebuilt - a.at(i, j)).norm() <= bound * scale,
                    "{what}: (v d vᴴ)[{i},{j}] = {rebuilt} vs {}",
                    a.at(i, j)
                );
                let gram: Complex64 = (0..n).map(|k| vv.at(k, i).conj() * vv.at(k, j)).sum();
                let unit = Complex64::new(f64::from(u8::from(i == j)), 0.0);
                assert!(
                    (gram - unit).norm() <= bound,
                    "{what}: (vᴴv)[{i},{j}] = {gram}"
                );
            }
        }
        let Some(reference) = reference else { continue };
        let o = reference.iter().find(|o| &o.sector == sector).unwrap();
        for (got, want) in values.iter().zip(&o.values) {
            assert!(
                (got - want).abs() <= bound * scale,
                "{what}: λ {got} vs reference {want}"
            );
        }
        // Clusters: maximal runs of reference values within `1e-6` relative.
        let mut clusters: Vec<std::ops::Range<usize>> = Vec::new();
        for k in 0..n {
            match clusters.last_mut() {
                Some(c) if o.values[k] - o.values[k - 1] <= 1e-6 * scale => c.end = k + 1,
                _ => clusters.push(k..k + 1),
            }
        }
        let gap = clusters
            .windows(2)
            .map(|pair| o.values[pair[1].start] - o.values[pair[0].end - 1])
            .fold(scale, f64::min);
        for cluster in &clusters {
            for i in 0..n {
                for j in 0..n {
                    let got: Complex64 = cluster
                        .clone()
                        .map(|k| vv.at(i, k) * vv.at(j, k).conj())
                        .sum();
                    let want: Complex64 = cluster
                        .clone()
                        .map(|k| o.vectors.at(i, k) * o.vectors.at(j, k).conj())
                        .sum();
                    assert!(
                        (got - want).norm() <= bound * scale / gap,
                        "{what}: projector {cluster:?} [{i},{j}] {got} vs {want}"
                    );
                }
            }
        }
    }
}
