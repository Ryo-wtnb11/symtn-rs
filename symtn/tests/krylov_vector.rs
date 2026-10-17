//! Host `TensorMap` as a `krylov::Vector` (feature `krylov`).
//!
//! Every oracle is independent of the impl under test:
//! - the vector operations against the physical dense expansion (U(1) and
//!   SU(2), with dual and non-self-dual legs) and, for the complex-symbol
//!   Fibonacci rule, against `Σ_c d_c ⟨x_c, y_c⟩` summed by hand over the
//!   coupled blocks;
//! - `eigsolve` on a two-site Heisenberg `S·S` against the hand eigenvalues
//!   `[S(S+1) - 2s(s+1)]/2` of each total spin `S`;
//! - `exponentiate` against `exp(t·H_c)·x_c` per coupled block, the small
//!   block exponential formed by Lagrange interpolation on those hand
//!   eigenvalues;
//! - an allocation upper bound for a warm `eigsolve`.
//!
//! Run with `--features krylov,blas-openblas` (or another LAPACK provider).
#![cfg(feature = "krylov")]

use std::sync::Arc;

use num_complex::{Complex32, Complex64};
use symtn::krylov::{self, Lanczos, Vector, Which, Workspace};
use symtn::sector::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use symtn::typed::{
    BlockFusionTrees, ContractSpec, Error, GradedSpace, OperationError, Runtime, SectorSpectrum,
    TensorMap, TensorScalar,
};
use symtn_sectors::{TableRule, TableSector};

#[path = "../../tests/support/numerics.rs"]
mod numerics;

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

use numerics::Numeric;

/// Deterministic fill values, unrelated between salts.
trait Sample: Numeric + TensorScalar {
    fn sample(k: usize, salt: usize) -> Self;
    fn real(v: f64) -> Self;
}

impl Sample for f64 {
    fn sample(k: usize, salt: usize) -> Self {
        ((k * 7 + salt * 5) % 13) as f64 / 3.0 - 2.0
    }
    fn real(v: f64) -> Self {
        v
    }
}

impl Sample for f32 {
    fn sample(k: usize, salt: usize) -> Self {
        f64::sample(k, salt) as f32
    }
    fn real(v: f64) -> Self {
        v as f32
    }
}

impl Sample for Complex32 {
    fn sample(k: usize, salt: usize) -> Self {
        Complex32::new(f32::sample(k, salt), f32::sample(k + 3, salt + 1))
    }
    fn real(v: f64) -> Self {
        Complex32::new(v as f32, 0.0)
    }
}

impl Sample for Complex64 {
    fn sample(k: usize, salt: usize) -> Self {
        Complex64::new(f64::sample(k, salt), f64::sample(k + 3, salt + 1))
    }
    fn real(v: f64) -> Self {
        Complex64::new(v, 0.0)
    }
}

macro_rules! filled {
    ($rt:expr, $codomain:expr, $domain:expr, $d:ty, $salt:expr) => {{
        let mut k = 0usize;
        TensorMap::<_, $d>::from_subblock_fn($rt, $codomain, $domain, |_, _| {
            k += 1;
            <$d as Sample>::sample(k, $salt)
        })
        .unwrap()
    }};
}

fn u1_legs() -> (GradedSpace<U1FusionRule>, GradedSpace<U1FusionRule>) {
    let v = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(-1, 2), (0, 1), (1, 3)].map(|(q, n)| (U1Irrep::new(q), n)),
    )
    .unwrap();
    let w = v.try_dual().unwrap();
    (v, w)
}

fn su2_legs() -> (GradedSpace<SU2FusionRule>, GradedSpace<SU2FusionRule>) {
    let v = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [(0, 2), (1, 2), (2, 1)].map(|(s, n)| (SU2Irrep::from_twice_spin(s), n)),
    )
    .unwrap();
    let w = v.try_dual().unwrap();
    (v, w)
}

fn is_space_mismatch(error: &Error) -> bool {
    matches!(error, Error::Operation(op) if matches!(**op, OperationError::SpaceMismatch { .. }))
}

/// `|got - want| <= tolerance` in the precision of the payload `D`, for
/// values widened to `Complex64`.
#[track_caller]
fn close<D: Numeric>(what: &str, got: Complex64, want: Complex64, terms: usize) {
    let bound = numerics::tolerance::<D>(terms, want.norm());
    assert!(
        (got - want).norm() <= bound,
        "{what}: {got} vs {want} (bound {bound:e})"
    );
}

#[track_caller]
fn slices_close<D: Numeric>(what: &str, got: &[Complex64], want: &[Complex64], terms: usize) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close::<D>(&format!("{what}[{i}]"), g, w, terms);
    }
}

fn wide<D: Numeric>(v: &[D]) -> Vec<Complex64> {
    v.iter().map(|x| x.wide()).collect()
}

/// The five vector operations on `x`, `y` (one space `[v, w] <- [v]`)
/// against their physical dense expansions.
macro_rules! vector_contract {
    ($what:expr, $rt:expr, $legs:expr, $d:ty, $a:expr, $b:expr) => {{
        let (v, w) = $legs;
        let what = $what;
        let x = filled!($rt, [&v, &w], [&v], $d, 1);
        let y = filled!($rt, [&v, &w], [&v], $d, 2);
        let xd = wide(&x.to_physical_dense().unwrap().data);
        let yd = wide(&y.to_physical_dense().unwrap().data);
        let n = xd.len();
        let inner: Complex64 = xd.iter().zip(&yd).map(|(p, q)| p.conj() * q).sum();
        let got = Vector::inner(&x, &y).unwrap().wide();
        close::<$d>(&format!("{what} inner"), got, inner, n);
        let norm = xd.iter().map(|p| p.norm_sqr()).sum::<f64>().sqrt();
        let got = Vector::norm(&x).unwrap().wide();
        close::<$d>(&format!("{what} norm"), got, Complex64::new(norm, 0.0), n);

        // `similar` is a fresh exact zero that accepts writes: no payload
        // shared with `x` (a clone would be `DestinationShared`).
        let mut z = x.similar().unwrap();
        assert!(wide(&z.to_physical_dense().unwrap().data)
            .iter()
            .all(|v| *v == Complex64::new(0.0, 0.0)));
        z.copy_from(&y).unwrap();
        assert_eq!(
            wide(&z.to_physical_dense().unwrap().data),
            yd,
            "{what} copy_from"
        );
        assert_eq!(
            wide(&x.to_physical_dense().unwrap().data),
            xd,
            "{what} source kept"
        );

        // `b = 0` overwrites a NaN-filled destination without reading it.
        let mut nan = TensorMap::<_, $d>::from_subblock_fn($rt, [&v, &w], [&v], |_, _| {
            <$d as Sample>::real(f64::NAN)
        })
        .unwrap();
        let (a, b): ($d, $d) = ($a, $b);
        Vector::axpby(&mut nan, a, &x, <$d as Sample>::real(0.0)).unwrap();
        let want: Vec<Complex64> = xd.iter().map(|p| a.wide() * p).collect();
        let got = wide(&nan.to_physical_dense().unwrap().data);
        assert!(
            got.iter().all(|v| v.re.is_finite() && v.im.is_finite()),
            "{what}: NaN leaked"
        );
        slices_close::<$d>(&format!("{what} axpby b=0"), &got, &want, 1);

        // General `a·x + b·self`.
        let mut d = x.similar().unwrap();
        d.copy_from(&y).unwrap();
        Vector::axpby(&mut d, a, &x, b).unwrap();
        let want: Vec<Complex64> = xd
            .iter()
            .zip(&yd)
            .map(|(p, q)| a.wide() * p + b.wide() * q)
            .collect();
        let got = wide(&d.to_physical_dense().unwrap().data);
        slices_close::<$d>(&format!("{what} axpby"), &got, &want, 2);

        // A tensor on another space is refused before any write.
        let other = filled!($rt, [&v], [&v], $d, 3);
        let mut e = other.similar().unwrap();
        let err = Vector::axpby(&mut e, a, &x, b).unwrap_err();
        assert!(is_space_mismatch(&err), "{what}: {err:?}");
    }};
}

#[test]
fn vector_operations_match_the_dense_expansion() {
    let rt = Runtime::builder().build().unwrap();
    let (a, b) = (0.75, -2.5);
    let (ca, cb) = (Complex64::new(0.5, -2.0), Complex64::new(-1.5, 0.25));
    vector_contract!("u1 f64", &rt, u1_legs(), f64, a, b);
    vector_contract!("u1 c64", &rt, u1_legs(), Complex64, ca, cb);
    vector_contract!("su2 f64", &rt, su2_legs(), f64, a, b);
    vector_contract!("su2 c64", &rt, su2_legs(), Complex64, ca, cb);
    let (a32, b32) = (0.75f32, -2.5f32);
    let (ca32, cb32) = (Complex32::new(0.5, -2.0), Complex32::new(-1.5, 0.25));
    vector_contract!("u1 f32", &rt, u1_legs(), f32, a32, b32);
    vector_contract!("su2 f32", &rt, su2_legs(), f32, a32, b32);
    vector_contract!("su2 c32", &rt, su2_legs(), Complex32, ca32, cb32);
}

#[test]
fn lazy_and_compact_receivers_are_refused_by_similar() {
    let rt = Runtime::builder().build().unwrap();
    let (v, _) = su2_legs();
    let x = filled!(&rt, [&v], [&v], Complex64, 4);
    let lazy = x.adjoint().unwrap();
    let names_materialize =
        |e: &Error| matches!(e, Error::InvalidArgument(m) if m.contains("materialize()"));
    assert!(names_materialize(&lazy.similar().unwrap_err()));
    let compact = TensorMap::<_, Complex64>::diagonal(
        &rt,
        &v,
        [(0, 2), (1, 2), (2, 1)].map(|(s, n)| SectorSpectrum {
            sector: SU2Irrep::from_twice_spin(s),
            values: (0..n)
                .map(|i| Complex64::new(1.0 + i as f64, 0.5))
                .collect(),
        }),
    )
    .unwrap();
    assert!(names_materialize(&compact.similar().unwrap_err()));
    // Both still serve as sources, read in their logical form.
    let mut z = lazy.materialize().unwrap().similar().unwrap();
    z.copy_from(&lazy).unwrap();
    assert_eq!(
        z.dense_data().unwrap(),
        lazy.materialize().unwrap().dense_data().unwrap()
    );
    let mut z = compact.materialize().unwrap().similar().unwrap();
    z.copy_from(&compact).unwrap();
    assert_eq!(
        z.dense_data().unwrap(),
        compact.materialize().unwrap().dense_data().unwrap()
    );
}

#[test]
fn krylov_reports_operation_errors_as_vector_errors() {
    let rt = Runtime::builder().build().unwrap();
    let (v, w) = u1_legs();
    let b = filled!(&rt, [&v, &w], [&v], f64, 5);
    let x0 = filled!(&rt, [&v], [&v], f64, 6);
    let mut ws = Workspace::with_capacity(&b, 5).unwrap();
    let mut op = |x: &TensorMap<U1FusionRule, f64>, y: &mut TensorMap<U1FusionRule, f64>| {
        x.axpby_into(y, 2.0, 0.0)
    };
    let result = krylov::linsolve(&mut op, &b, &x0, &krylov::Cg::new(), 0.0, 1.0, &mut ws);
    match result {
        Err(krylov::Error::Vector(e)) => assert!(is_space_mismatch(&e), "{e:?}"),
        other => panic!("{:?}", other.map(|_| ())),
    }
}

/// Two-site `S·S` for spin `s` (`twice = 2s`) with leg degeneracy `deg`:
/// eigenvalue `[S(S+1) - 2s(s+1)]/2` on total spin `S`, identity on the
/// degeneracy indices. Distinct eigenvalues ascending.
fn heisenberg_eigenvalues(twice: usize) -> Vec<f64> {
    let s = twice as f64 / 2.0;
    (0..=twice)
        .map(|big| {
            let big = big as f64;
            (big * (big + 1.0) - 2.0 * s * (s + 1.0)) / 2.0
        })
        .collect()
}

/// `S·S` on `V ⊗ V <- V ⊗ V`, SU(2): `λ_c` on coupled spin `c`.
fn su2_heisenberg<D: Sample>(
    rt: &Runtime,
    v: &GradedSpace<SU2FusionRule>,
    twice: usize,
) -> TensorMap<SU2FusionRule, D> {
    let lambda = heisenberg_eigenvalues(twice);
    TensorMap::from_subblock_fn(
        rt,
        [v, v],
        [v, v],
        |trees: &BlockFusionTrees<SU2Irrep>, idx| {
            let diag = idx[0] == idx[2] && idx[1] == idx[3];
            D::real(if diag {
                lambda[trees.coupled().twice_spin() / 2]
            } else {
                0.0
            })
        },
    )
    .unwrap()
}

/// `S·S` on `V ⊗ V <- V ⊗ V`, U(1) with charge `2m`: `Sz⊗Sz + (S+⊗S- + S-⊗S+)/2`.
fn u1_heisenberg<D: Sample>(
    rt: &Runtime,
    v: &GradedSpace<U1FusionRule>,
    twice: usize,
) -> TensorMap<U1FusionRule, D> {
    let s = twice as f64 / 2.0;
    let raise = |m: f64| (s * (s + 1.0) - m * (m + 1.0)).sqrt();
    let lower = |m: f64| (s * (s + 1.0) - m * (m - 1.0)).sqrt();
    TensorMap::from_subblock_fn(
        rt,
        [v, v],
        [v, v],
        |trees: &BlockFusionTrees<U1Irrep>, idx| {
            if !(idx[0] == idx[2] && idx[1] == idx[3]) {
                return D::real(0.0);
            }
            let m = |q: &U1Irrep| f64::from(q.charge()) / 2.0;
            let (c, d) = (trees.codomain_uncoupled(), trees.domain_uncoupled());
            let (p1, p2, m1, m2) = (m(&c[0]), m(&c[1]), m(&d[0]), m(&d[1]));
            let value = if (p1, p2) == (m1, m2) {
                m1 * m2
            } else if (p1, p2) == (m1 + 1.0, m2 - 1.0) {
                0.5 * raise(m1) * lower(m2)
            } else if (p1, p2) == (m1 - 1.0, m2 + 1.0) {
                0.5 * lower(m1) * raise(m2)
            } else {
                0.0
            };
            D::real(value)
        },
    )
    .unwrap()
}

/// `y ← H·x` for `H: V⊗V <- V⊗V`, `x: V⊗V <- W`, written in place.
const APPLY: ContractSpec<'static> = ContractSpec {
    lhs: &[2, 3],
    rhs: &[0, 1],
    codomain: &[0, 1],
    domain: &[2],
};

/// Distinct eigenvalues of the operator found by Lanczos on `x0`, plus the
/// eigen-residuals and the eigenvector norms in the dense expansion.
macro_rules! eigsolve_case {
    ($what:expr, $rt:expr, $h:expr, $x0:expr, $want:expr, $alg:expr, $tol:expr) => {{
        let (h, x0, want): (_, _, Vec<f64>) = (&$h, &$x0, $want);
        let mut op =
            |x: &_, y: &mut _| h.contract_into(x, &APPLY, y, Sample::real(1.0), Sample::real(0.0));
        let (alg, tol): (Lanczos, f64) = ($alg, $tol);
        let mut ws = Workspace::with_capacity(x0, 10).unwrap();
        let (vals, vecs, info) =
            krylov::eigsolve(&mut op, x0, want.len(), Which::SmallestReal, &alg, &mut ws).unwrap();
        assert_eq!(info.converged, want.len(), "{}: {info:?}", $what);
        for ((&got, &want), x) in vals.iter().zip(&want).zip(&vecs) {
            let got: f64 = got.into();
            assert!((got - want).abs() < tol, "{}: {got} vs {want}", $what);
            let dense = wide(&x.to_physical_dense().unwrap().data);
            let n = dense.iter().map(|p| p.norm_sqr()).sum::<f64>().sqrt();
            assert!((n - 1.0).abs() < tol, "{}: |x| = {n}", $what);
            let hx = h.contract(x, &APPLY).unwrap();
            let r = hx.axpby(Sample::real(1.0), x, Sample::real(-got)).unwrap();
            assert!(r.norm(2.0).unwrap() < 10.0 * tol, "{}: residual", $what);
        }
    }};
}

fn f64_alg() -> Lanczos {
    Lanczos::new().krylovdim(10)
}

#[test]
fn eigsolve_in_single_precision() {
    // SU(2) spin 1: [-2, -1, 1], to single precision (Lanczos tolerance
    // above the f32 rounding of the residual).
    let rt = Runtime::builder().build().unwrap();
    let v =
        GradedSpace::try_new(Arc::new(SU2FusionRule), [(SU2Irrep::from_twice_spin(2), 2)]).unwrap();
    let w = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        (0..=2).map(|big| (SU2Irrep::from_twice_spin(2 * big), 1 + big % 2)),
    )
    .unwrap();
    let want = heisenberg_eigenvalues(2);
    let alg = || Lanczos::new().krylovdim(10).tol(1e-5);
    let h = su2_heisenberg::<f32>(&rt, &v, 2);
    let x0 = filled!(&rt, [&v, &v], [&w], f32, 7);
    eigsolve_case!("su2 2s=2 f32", &rt, h, x0, want.clone(), alg(), 1e-5);
    let h = su2_heisenberg::<Complex32>(&rt, &v, 2);
    let x0 = filled!(&rt, [&v, &v], [&w], Complex32, 7);
    eigsolve_case!("su2 2s=2 c32", &rt, h, x0, want, alg(), 1e-5);
}

#[test]
fn eigsolve_finds_the_two_site_heisenberg_spectrum() {
    let rt = Runtime::builder().build().unwrap();
    for twice in [1usize, 2] {
        let want = heisenberg_eigenvalues(twice);
        // SU(2): spin `s` with degeneracy 2, and W carrying every total spin.
        let v = GradedSpace::try_new(
            Arc::new(SU2FusionRule),
            [(SU2Irrep::from_twice_spin(twice), 2)],
        )
        .unwrap();
        let w = GradedSpace::try_new(
            Arc::new(SU2FusionRule),
            (0..=twice).map(|big| (SU2Irrep::from_twice_spin(2 * big), 1 + big % 2)),
        )
        .unwrap();
        let h = su2_heisenberg::<f64>(&rt, &v, twice);
        let x0 = filled!(&rt, [&v, &v], [&w], f64, 7);
        eigsolve_case!(
            format!("su2 2s={twice} f64"),
            &rt,
            h,
            x0,
            want.clone(),
            f64_alg(),
            1e-10
        );
        let h = su2_heisenberg::<Complex64>(&rt, &v, twice);
        let x0 = filled!(&rt, [&v, &v], [&w], Complex64, 7);
        eigsolve_case!(
            format!("su2 2s={twice} c64"),
            &rt,
            h,
            x0,
            want.clone(),
            f64_alg(),
            1e-10
        );

        // U(1): the same model in the Sz basis, W carrying every charge.
        let v = GradedSpace::try_new(
            Arc::new(U1FusionRule),
            (0..=twice).map(|k| (U1Irrep::new(2 * k as i32 - twice as i32), 2)),
        )
        .unwrap();
        let w = GradedSpace::try_new(
            Arc::new(U1FusionRule),
            (0..=2 * twice).map(|k| (U1Irrep::new(2 * k as i32 - 2 * twice as i32), 2)),
        )
        .unwrap();
        let h = u1_heisenberg::<f64>(&rt, &v, twice);
        let x0 = filled!(&rt, [&v, &v], [&w], f64, 8);
        eigsolve_case!(
            format!("u1 2s={twice} f64"),
            &rt,
            h,
            x0,
            want.clone(),
            f64_alg(),
            1e-10
        );
        let h = u1_heisenberg::<Complex64>(&rt, &v, twice);
        let x0 = filled!(&rt, [&v, &v], [&w], Complex64, 8);
        eigsolve_case!(
            format!("u1 2s={twice} c64"),
            &rt,
            h,
            x0,
            want,
            f64_alg(),
            1e-10
        );
    }
}

/// `exp(t·H_c)·x_c` per coupled block, with `exp(t·H_c)` the Lagrange
/// interpolant `Σ_k e^{tλ_k} Π_{j≠k} (H_c - λ_j)/(λ_k - λ_j)` over the
/// distinct eigenvalues `λ` of `H`, compared with `exponentiate`.
macro_rules! exponentiate_case {
    ($what:expr, $h:expr, $x:expr, $t:expr, $lambda:expr) => {{
        let (h, x, lambda): (_, _, Vec<f64>) = (&$h, &$x, $lambda);
        let t = $t;
        let mut op =
            |x: &_, y: &mut _| h.contract_into(x, &APPLY, y, Sample::real(1.0), Sample::real(0.0));
        let alg = Lanczos::new().krylovdim(10).tol(1e-13);
        let mut ws = Workspace::with_capacity(x, 10).unwrap();
        let (y, info) = krylov::exponentiate(&mut op, t, x, &alg, &mut ws).unwrap();
        assert_eq!(info.converged, 1, "{}", $what);
        let tw = t.wide();
        for (sector, xb) in x.blocks().unwrap() {
            let hb = h.block(&sector).unwrap();
            let yb = y.block(&sector).unwrap();
            let (r, c) = (xb.rows(), xb.cols());
            let hm: Vec<Complex64> = (0..r * r).map(|k| symtn_get(&hb, k % r, k / r)).collect();
            let xm: Vec<Complex64> = (0..r * c).map(|k| symtn_get(&xb, k % r, k / r)).collect();
            let mut want = vec![Complex64::new(0.0, 0.0); r * c];
            for (k, &lk) in lambda.iter().enumerate() {
                // Π_{j≠k} (H_c - λ_j)/(λ_k - λ_j) applied to x_c.
                let mut term = xm.clone();
                for (j, &lj) in lambda.iter().enumerate() {
                    if j == k {
                        continue;
                    }
                    let mut next = vec![Complex64::new(0.0, 0.0); r * c];
                    for col in 0..c {
                        for row in 0..r {
                            let mut acc = -lj * term[row + col * r];
                            for m in 0..r {
                                acc += hm[row + m * r] * term[m + col * r];
                            }
                            next[row + col * r] = acc / (lk - lj);
                        }
                    }
                    term = next;
                }
                let e = (tw * lk).exp();
                for (wv, tv) in want.iter_mut().zip(&term) {
                    *wv += e * tv;
                }
            }
            let got: Vec<Complex64> = (0..r * c).map(|k| symtn_get(&yb, k % r, k / r)).collect();
            let scale = want.iter().map(|v| v.norm()).fold(1.0, f64::max);
            for (g, w) in got.iter().zip(&want) {
                assert!((g - w).norm() < 1e-10 * scale, "{}: {g} vs {w}", $what);
            }
        }
    }};
}

fn symtn_get<R, D: Numeric + TensorScalar>(
    b: &symtn::typed::CoupledBlock<'_, R, D>,
    i: usize,
    j: usize,
) -> Complex64 {
    b.get(i, j).unwrap().wide()
}

#[test]
fn exponentiate_matches_the_block_exponential() {
    let rt = Runtime::builder().build().unwrap();
    for twice in [1usize, 2] {
        let lambda = heisenberg_eigenvalues(twice);
        let v = GradedSpace::try_new(
            Arc::new(SU2FusionRule),
            [(SU2Irrep::from_twice_spin(twice), 2)],
        )
        .unwrap();
        let w = GradedSpace::try_new(
            Arc::new(SU2FusionRule),
            (0..=twice).map(|big| (SU2Irrep::from_twice_spin(2 * big), 2)),
        )
        .unwrap();
        let h = su2_heisenberg::<f64>(&rt, &v, twice);
        let x = filled!(&rt, [&v, &v], [&w], f64, 9);
        exponentiate_case!(format!("su2 2s={twice} f64"), h, x, -0.7, lambda.clone());
        let h = su2_heisenberg::<Complex64>(&rt, &v, twice);
        let x = filled!(&rt, [&v, &v], [&w], Complex64, 9);
        exponentiate_case!(
            format!("su2 2s={twice} c64"),
            h,
            x,
            Complex64::new(0.3, -1.1),
            lambda.clone()
        );

        let v = GradedSpace::try_new(
            Arc::new(U1FusionRule),
            (0..=twice).map(|k| (U1Irrep::new(2 * k as i32 - twice as i32), 2)),
        )
        .unwrap();
        let w = GradedSpace::try_new(
            Arc::new(U1FusionRule),
            (0..=2 * twice).map(|k| (U1Irrep::new(2 * k as i32 - 2 * twice as i32), 1)),
        )
        .unwrap();
        let h = u1_heisenberg::<f64>(&rt, &v, twice);
        let x = filled!(&rt, [&v, &v], [&w], f64, 10);
        exponentiate_case!(format!("u1 2s={twice} f64"), h, x, -0.7, lambda.clone());
        let h = u1_heisenberg::<Complex64>(&rt, &v, twice);
        let x = filled!(&rt, [&v, &v], [&w], Complex64, 10);
        exponentiate_case!(
            format!("u1 2s={twice} c64"),
            h,
            x,
            Complex64::new(0.0, -2.0),
            lambda
        );
    }
}

#[test]
fn fibonacci_vectors_carry_the_golden_ratio_weight() {
    let rt = Runtime::builder().build().unwrap();
    let rule = Arc::new(TableRule::fibonacci());
    let (one, tau) = (TableSector::new(1), TableSector::new(2));
    let t = GradedSpace::try_new(Arc::clone(&rule), [(tau, 1)]).unwrap();
    let w = GradedSpace::try_new(Arc::clone(&rule), [(one, 2), (tau, 3)]).unwrap();
    let x = filled!(&rt, [&t, &t], [&w], Complex64, 11);
    let y = filled!(&rt, [&t, &t], [&w], Complex64, 12);
    let phi = (1.0 + 5f64.sqrt()) / 2.0;
    let weighted = |a: &TensorMap<TableRule, Complex64>, b: &TensorMap<TableRule, Complex64>| {
        let mut sum = Complex64::new(0.0, 0.0);
        for (sector, ab) in a.blocks().unwrap() {
            let bb = b.block(&sector).unwrap();
            let d = if sector == one { 1.0 } else { phi };
            for i in 0..ab.rows() {
                for j in 0..ab.cols() {
                    sum += d * ab.get(i, j).unwrap().conj() * bb.get(i, j).unwrap();
                }
            }
        }
        sum
    };
    numerics::assert_close(
        "fib inner",
        Vector::inner(&x, &y).unwrap(),
        weighted(&x, &y),
        32,
    );
    numerics::assert_close(
        "fib norm",
        Vector::norm(&x).unwrap(),
        weighted(&x, &x).re.sqrt(),
        32,
    );

    // H = -P_𝟙 on τ ⊗ τ: eigenvalue -1 on channel 𝟙, 0 on channel τ.
    let h = TensorMap::<_, Complex64>::from_subblock_fn(&rt, [&t, &t], [&t, &t], |trees, _| {
        Complex64::new(if *trees.coupled() == one { -1.0 } else { 0.0 }, 0.0)
    })
    .unwrap();
    let mut op = |x: &TensorMap<TableRule, Complex64>, y: &mut TensorMap<TableRule, Complex64>| {
        h.compose(x)?
            .axpby_into(y, Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0))
    };
    let mut ws = Workspace::with_capacity(&x, 6).unwrap();
    let alg = Lanczos::new().krylovdim(6);
    let (vals, _, info) =
        krylov::eigsolve(&mut op, &x, 2, Which::SmallestReal, &alg, &mut ws).unwrap();
    assert_eq!(info.converged, 2);
    assert!(
        (vals[0] + 1.0).abs() < 1e-12 && vals[1].abs() < 1e-12,
        "{vals:?}"
    );
}

#[test]
fn warm_eigsolve_allocates_only_its_results() {
    // A random Hermitian H with many distinct eigenvalues, so Lanczos
    // restarts; the map writes in place (`contract_into`).
    let rt = Runtime::builder().dense_threads(1).build().unwrap();
    let v = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(-1, 4), (0, 3), (1, 4)].map(|(q, n)| (U1Irrep::new(q), n)),
    )
    .unwrap();
    let w = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(-2, 12), (-1, 16), (0, 20), (1, 16), (2, 12)].map(|(q, n)| (U1Irrep::new(q), n)),
    )
    .unwrap();
    let a = TensorMap::<_, f64>::rand_with_seed(&rt, [&v, &v], [&v, &v], 3).unwrap();
    let h = a.axpby(1.0, &a.adjoint().unwrap(), 1.0).unwrap();
    let x0 = filled!(&rt, [&v, &v], [&w], f64, 13);
    let payload = std::mem::size_of_val(x0.dense_data().unwrap());
    let howmany = 2;
    let alg = Lanczos::new().krylovdim(12).maxiter(20);
    let mut ws = Workspace::with_capacity(&x0, 12).unwrap();
    let mut op = |x: &TensorMap<U1FusionRule, f64>, y: &mut TensorMap<U1FusionRule, f64>| {
        h.contract_into(x, &APPLY, y, 1.0, 0.0)
    };
    // Warm the contraction plan and the dense backend.
    krylov::eigsolve(&mut op, &x0, howmany, Which::SmallestReal, &alg, &mut ws).unwrap();
    let ((vals, vecs, info), allocs) = counting_alloc::measure(|| {
        krylov::eigsolve(&mut op, &x0, howmany, Which::SmallestReal, &alg, &mut ws).unwrap()
    });
    // The map's own allocations (contraction metadata), per application.
    let mut y = x0.similar().unwrap();
    let ((), map) = counting_alloc::measure(|| op(&x0, &mut y).unwrap());
    // Many map applications, but payload allocations only for the returned
    // eigenvectors and residuals, plus O(k²) dense scratch and metadata: a
    // payload-sized allocation per step would exceed the bound several times.
    assert!(info.numops > 4 * alg.get_krylovdim(), "{info:?}");
    let results = vecs.len() + info.residual.len();
    let bound =
        info.numops as u64 * map.bytes + (results + 1) as u64 * (payload as u64 + 4096) + 64 * 1024;
    assert!(
        info.numops as u64 * payload as u64 > 2 * bound,
        "fixture too small"
    );
    assert!(
        allocs.bytes <= bound,
        "{} bytes > {bound} (payload {payload}, {} ops, {vals:?})",
        allocs.bytes,
        info.numops
    );
}
