//! Cross-storage matrix for the typed methods unified over storage (#1756).
//!
//! Host rows always run; CUDA rows compare the device result with the Host
//! result on the same fixture and need a real device. Every unified family
//! has rows here.
//!
//! Oracles are hand sums over the entries the fixtures write
//! (`norm(t, p) = (Σ_c dim(c) Σ_ij |t_c[i,j]|^p)^(1/p)`, `norm(t, Inf) = max`,
//! `inner(a, b) = Σ_c dim(c) Σ_ij conj(a_c[i,j]) b_c[i,j]`), never read back
//! from a symtn result. Elementwise results (`scale`, `zeros_like`, `axpby`,
//! `axpby_into`) are checked through `norm` and the linear functional
//! `inner(c, ·)` against the entrywise combination of the written entries;
//! fixtures on one space are written in one order, so their entries align.
//! Compact factors (`svd_compact`, `qr_compact`) are checked the same way
//! through their rebuilt product, and through the norm of `s`/`r`, which an
//! isometric outer factor preserves. `eigh_full` runs on the Hermitian
//! fixture `h_c[i,j] = e_c[i,j] + conj(e_c[j,i])` and is checked through
//! `v d v^H` and the norm of `d`. `twist` runs on fermion parity (the U(1)
//! and SU(2) fixtures are bosonic, where it is the identity) and is checked
//! through `inner(t, twist(t))` against `Σ θ |t|²`, with θ = −1 for each
//! listed leg whose uncoupled sector is odd. `materialize` of a lazy
//! adjoint is checked through `inner(a^H, b^H) = conj(inner(a, b))` over the
//! written entries and the exact round trip `(a^H)^H = a`. The transforms
//! (`permute`, `braid`, `transpose`, `repartition` and their `_into`) are
//! unitary, so they are checked through `norm`, `inner(T a, T b) =
//! inner(a, b)` and their inverse motion, and `_into` through
//! `inner(T a, alpha T a + beta T b)`. `contract`, `compose` and
//! `contract_into` are checked through `tr(a^H ∘ b) = inner(a, b)` and the
//! documented `compose`/`contract` twist relation (see `contract_rows`).
//!
//! Unsupported cells:
//! - CUDA `norm(p)` for `p != 2`, and `inner` with a lazy adjoint CUDA
//!   operand: `UnsupportedOnDevice`, asserted below.
//! - CUDA `svd_compact`/`qr_compact`/`eigh_full` of a lazy adjoint operand:
//!   `UnsupportedOnDevice`, asserted below.
//! - Fibonacci and SU(2)_k have complex twists (`Scalar = Complex64`), so
//!   they have no CUDA methods (compile-time absent); every device-admitted
//!   twist is ±1.
//! - CUDA compact storage does not exist (`to_cuda` densifies), so the
//!   compact column is Host-only.
//! - Checked Generic and complex-coefficient rules have no CUDA methods
//!   (compile-time absent).

use std::sync::Arc;

use num_complex::Complex64;
use symtn::sector::{
    FermionParityFusionRule, SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep, Z2Irrep,
};
use symtn::typed::{
    ContractSpec, Direction, Eigh, Error, GradedSpace, HermitianTol, OperationError, Qr, Runtime,
    SectorSpectrum, Svd, TensorMap,
};

const POWERS: [f64; 4] = [1.0, 2.0, 3.0, f64::INFINITY];

fn close(actual: f64, expected: f64) -> bool {
    (actual - expected).abs() <= 1e-12 * (1.0 + expected.abs())
}

fn close_c(actual: Complex64, expected: Complex64) -> bool {
    (actual - expected).norm() <= 1e-12 * (1.0 + expected.norm())
}

/// `norm(t, p)` over `(dim(c), entry)` pairs.
fn norm_oracle(entries: &[(f64, Complex64)], p: f64) -> f64 {
    if p.is_infinite() {
        return entries.iter().map(|(_, x)| x.norm()).fold(0.0, f64::max);
    }
    entries
        .iter()
        .map(|(d, x)| d * x.norm().powf(p))
        .sum::<f64>()
        .powf(p.recip())
}

/// `inner(a, b)` over entries written in the same order.
fn inner_oracle(a: &[(f64, Complex64)], b: &[(f64, Complex64)]) -> Complex64 {
    a.iter()
        .zip(b)
        .map(|((d, x), (_, y))| *d * x.conj() * y)
        .sum()
}

fn entry(seed: f64, coupled: usize, i: usize, j: usize) -> Complex64 {
    Complex64::new(
        seed + 0.37 * i as f64 - 0.21 * j as f64 + 0.11 * coupled as f64,
        0.13 * seed - 0.17 * (i + 2 * j) as f64 + 0.05,
    )
}

/// The Hermitian entry `e[i,j] + conj(e[j,i])`.
fn hermitian_entry(seed: f64, coupled: usize, i: usize, j: usize) -> Complex64 {
    entry(seed, coupled, i, j) + entry(seed, coupled, j, i).conj()
}

/// `alpha * x + beta * y` entrywise.
fn combo(
    alpha: Complex64,
    x: &[(f64, Complex64)],
    beta: Complex64,
    y: &[(f64, Complex64)],
) -> Vec<(f64, Complex64)> {
    x.iter()
        .zip(y)
        .map(|((d, x), (_, y))| (*d, alpha * x + beta * y))
        .collect()
}

/// Host rows for one rule and payload: dense, lazy adjoint and compact
/// operands of every reduction and elementwise operation against the hand
/// oracle, plus the admission errors. The CUDA rows reuse the same fixtures
/// through `$cuda`.
macro_rules! storage_rows {
    ($runtime:expr, $leg:expr, $coupled:expr, $dim:expr, $D:ty, $conv:expr, $cuda:expr) => {{
        let runtime: &Runtime = $runtime;
        let leg = $leg;
        let coupled = $coupled;
        let dim = $dim;
        let conv = $conv;
        let build_with = |seed: f64, entry: fn(f64, usize, usize, usize) -> Complex64| {
            let mut entries = Vec::new();
            let t = TensorMap::<_, $D>::from_subblock_fn(runtime, [&leg], [&leg], |trees, ix| {
                let c = coupled(trees.coupled());
                let x = conv(entry(seed, c, ix[0], ix[1]));
                entries.push((dim(c), x));
                x
            })
            .unwrap();
            let wide: Vec<(f64, Complex64)> = entries
                .into_iter()
                .map(|(d, x): (f64, $D)| (d, Complex64::from(x)))
                .collect();
            (t, wide)
        };
        let build = |seed: f64| build_with(seed, entry);
        let (a, a_entries) = build(0.9);
        let (b, b_entries) = build(-0.4);

        // Dense and lazy adjoint: every entrywise norm is adjoint invariant.
        let lazy_a = a.adjoint().unwrap();
        let lazy_b = b.adjoint().unwrap();
        for p in POWERS {
            let expected = norm_oracle(&a_entries, p);
            assert!(close(a.norm(p).unwrap(), expected), "dense norm({p})");
            assert!(close(lazy_a.norm(p).unwrap(), expected), "lazy norm({p})");
        }
        let ab = inner_oracle(&a_entries, &b_entries);
        assert!(close_c(Complex64::from(a.inner(&b).unwrap()), ab));
        // `<a^H, b^H> = conj(<a, b>)`, with dim(c) = dim(conj(c)).
        assert!(close_c(Complex64::from(lazy_a.inner(&lazy_b).unwrap()), ab.conj()));
        let materialized = lazy_a.materialize().unwrap();
        assert!(close_c(
            Complex64::from(lazy_a.inner(&materialized).unwrap()),
            Complex64::from(materialized.inner(&materialized).unwrap()),
        ));

        // Compact diagonal: the stored diagonal only.
        let mut diag_entries = Vec::new();
        let spectra: Vec<_> = leg
            .sectors()
            .unwrap()
            .into_iter()
            .map(|sector| {
                let k = leg.degeneracy(&sector).unwrap();
                let c = coupled(&sector);
                let values: Vec<$D> = (0..k).map(|i| conv(entry(0.6, c, i, i))).collect();
                diag_entries.extend(values.iter().map(|&x| (dim(c), Complex64::from(x))));
                SectorSpectrum { sector, values }
            })
            .collect();
        let compact = TensorMap::<_, $D>::diagonal(runtime, &leg, spectra).unwrap();
        for p in POWERS {
            let expected = norm_oracle(&diag_entries, p);
            assert!(close(compact.norm(p).unwrap(), expected), "compact norm({p})");
        }
        assert!(close_c(
            Complex64::from(compact.inner(&compact).unwrap()),
            inner_oracle(&diag_entries, &diag_entries),
        ));

        // Admission: invalid `p` and a foreign space.
        assert!(matches!(a.norm(0.0), Err(Error::InvalidArgument(_))));
        let wider = TensorMap::<_, $D>::from_subblock_fn(runtime, [&leg, &leg], [&leg], |_, _| {
            conv(Complex64::new(1.0, 0.0))
        })
        .unwrap();
        assert!(matches!(
            a.inner(&wider),
            Err(Error::Operation(op)) if matches!(*op, OperationError::SpaceMismatch { .. })
        ));

        // Elementwise. `c` is on `a`'s space, `lazy_c` on its adjoint space;
        // for `t = e^H`, `<c^H, e^H> = conj(<c, e>)` and `norm(t) = norm(e)`.
        let (c, c_entries) = build(0.3);
        let lazy_c = c.adjoint().unwrap();
        let (alpha, beta) = (
            conv(Complex64::new(0.7, -0.3)),
            conv(Complex64::new(-1.1, 0.4)),
        );
        let (wa, wb) = (Complex64::from(alpha), Complex64::from(beta));
        let zero = Complex64::new(0.0, 0.0);
        let check = |what: &str, t: &TensorMap<_, $D>, e: &[(f64, Complex64)]| {
            let functional = Complex64::from(c.inner(t).unwrap());
            assert!(close_c(functional, inner_oracle(&c_entries, e)), "{what}");
            assert!(close(t.norm(2.0).unwrap(), norm_oracle(e, 2.0)), "{what} norm");
        };
        let check_lazy = |what: &str, t: &TensorMap<_, $D>, e: &[(f64, Complex64)]| {
            let functional = Complex64::from(lazy_c.inner(t).unwrap());
            assert!(close_c(functional, inner_oracle(&c_entries, e).conj()), "{what}");
            assert!(close(t.norm(2.0).unwrap(), norm_oracle(e, 2.0)), "{what} norm");
        };
        let ab = combo(wa, &a_entries, wb, &b_entries);
        let zeros = combo(zero, &a_entries, zero, &b_entries);
        check("scale", &a.scale(alpha).unwrap(), &combo(wa, &a_entries, zero, &b_entries));
        check("axpby", &a.axpby(alpha, &b, beta).unwrap(), &ab);
        check("zeros_like", &a.zeros_like().unwrap(), &zeros);
        let (mut destination, _) = build(-0.4);
        a.axpby_into(&mut destination, alpha, beta).unwrap();
        check("axpby_into", &destination, &ab);

        // Lazy adjoints: `alpha A^H + beta B^H = (conj(alpha) A + conj(beta) B)^H`.
        let lazy_ab = combo(wa.conj(), &a_entries, wb.conj(), &b_entries);
        let lazy_scaled = lazy_a.scale(alpha).unwrap();
        check_lazy("lazy scale", &lazy_scaled, &combo(wa.conj(), &a_entries, zero, &b_entries));
        let lazy_sum = lazy_a.axpby(alpha, &lazy_b, beta).unwrap();
        check_lazy("lazy + lazy", &lazy_sum, &lazy_ab);
        // Both stay lazy adjoints (A10): the payload is the parents'.
        for lazy in [&lazy_scaled, &lazy_sum] {
            assert!(matches!(lazy.dense_data(), Err(Error::Unsupported { .. })));
        }
        let owned_b = lazy_b.materialize().unwrap();
        check_lazy("lazy + owned", &lazy_a.axpby(alpha, &owned_b, beta).unwrap(), &lazy_ab);
        let zero_lazy = lazy_a.zeros_like().unwrap();
        check_lazy("lazy zeros_like", &zero_lazy, &zeros);
        assert!(matches!(zero_lazy.dense_data(), Err(Error::Unsupported { .. })));
        let mut destination = lazy_b.materialize().unwrap();
        lazy_a.axpby_into(&mut destination, alpha, beta).unwrap();
        check_lazy("lazy axpby_into", &destination, &lazy_ab);

        // Compact: scale and compact + compact stay compact.
        let (wide_alpha, wide_beta) = (Complex64::from(alpha), Complex64::from(beta));
        let diag_scaled: Vec<_> = diag_entries.iter().map(|(d, x)| (*d, wide_alpha * x)).collect();
        assert!(close(compact.scale(alpha).unwrap().norm(2.0).unwrap(), norm_oracle(&diag_scaled, 2.0)));
        let diag_sum = combo(wide_alpha, &diag_entries, wide_beta, &diag_entries);
        let compact_sum = compact.axpby(alpha, &compact, beta).unwrap();
        assert!(close(compact_sum.norm(2.0).unwrap(), norm_oracle(&diag_sum, 2.0)));
        assert!(close(compact.zeros_like().unwrap().norm(2.0).unwrap(), 0.0));

        // Admission: a foreign space, before any write.
        assert!(matches!(
            a.axpby(alpha, &wider, beta),
            Err(Error::Operation(op)) if matches!(*op, OperationError::SpaceMismatch { .. })
        ));
        let mut foreign = wider.clone();
        assert!(matches!(
            a.axpby_into(&mut foreign, alpha, beta),
            Err(Error::Operation(op)) if matches!(*op, OperationError::SpaceMismatch { .. })
        ));

        // Compact factors: `u s vh` and `q r` rebuild the operand, and `s`
        // and `r` carry its dim-weighted Frobenius norm (the outer factors are
        // isometries).
        let norm_a = norm_oracle(&a_entries, 2.0);
        let svd_rebuild = |Svd { u, s, vh }: Svd<TensorMap<_, $D>>| {
            assert!(close(s.norm(2.0).unwrap(), u.compose(&s).unwrap().norm(2.0).unwrap()));
            u.compose(&s).unwrap().compose(&vh).unwrap()
        };
        let qr_rebuild = |Qr { q, r }: Qr<TensorMap<_, $D>>| {
            assert!(close(r.norm(2.0).unwrap(), q.compose(&r).unwrap().norm(2.0).unwrap()));
            q.compose(&r).unwrap()
        };
        check("svd_compact", &svd_rebuild(a.svd_compact(&[0], &[1]).unwrap()), &a_entries);
        check("qr_compact", &qr_rebuild(a.qr_compact(&[0], &[1]).unwrap()), &a_entries);
        assert!(close(a.svd_compact(&[0], &[1]).unwrap().s.norm(2.0).unwrap(), norm_a));
        check_lazy(
            "lazy svd_compact",
            &svd_rebuild(lazy_a.svd_compact(&[0], &[1]).unwrap()),
            &a_entries,
        );
        check_lazy(
            "lazy qr_compact",
            &qr_rebuild(lazy_a.qr_compact(&[0], &[1]).unwrap()),
            &a_entries,
        );
        // A moved split factors `permute(a, [1], [0])`, of the same norm.
        let swapped = svd_rebuild(a.svd_compact(&[1], &[0]).unwrap());
        assert!(close(swapped.norm(2.0).unwrap(), norm_a));
        let swapped = qr_rebuild(a.qr_compact(&[1], &[0]).unwrap());
        assert!(close(swapped.norm(2.0).unwrap(), norm_a));
        let norm_diag = norm_oracle(&diag_entries, 2.0);
        let compact_svd = compact.svd_compact(&[0], &[1]).unwrap();
        assert!(close(compact_svd.s.norm(2.0).unwrap(), norm_diag));
        assert!(close(svd_rebuild(compact_svd).norm(2.0).unwrap(), norm_diag));
        let compact_qr = compact.qr_compact(&[0], &[1]).unwrap();
        assert!(close(qr_rebuild(compact_qr).norm(2.0).unwrap(), norm_diag));
        for malformed in [a.svd_compact(&[0], &[0]).map(|_| ()), a.qr_compact(&[0], &[2]).map(|_| ())] {
            assert!(matches!(
                malformed,
                Err(Error::Operation(op)) if matches!(*op, OperationError::InvalidPermutation { .. })
            ));
        }

        // Hermitian eigendecomposition: `v d v^H` rebuilds `h`, `d` carries
        // its norm (`v` is unitary), and `h^H = h` so the lazy adjoint
        // rebuilds `h` too.
        let tol = HermitianTol::DEFAULT;
        let (h, h_entries) = build_with(0.9, hermitian_entry);
        let lazy_h = h.adjoint().unwrap();
        let norm_h = norm_oracle(&h_entries, 2.0);
        let eigh_rebuild = |Eigh { d, v }: Eigh<TensorMap<_, $D>>| {
            assert!(close(d.norm(2.0).unwrap(), norm_h));
            v.compose(&d).unwrap().compose(&v.adjoint().unwrap()).unwrap()
        };
        check("eigh_full", &eigh_rebuild(h.eigh_full(&[0], &[1], tol).unwrap()), &h_entries);
        check("lazy eigh_full", &eigh_rebuild(lazy_h.eigh_full(&[0], &[1], tol).unwrap()), &h_entries);
        let swapped = eigh_rebuild(h.eigh_full(&[1], &[0], tol).unwrap());
        assert!(close(swapped.norm(2.0).unwrap(), norm_h));
        // A compact diagonal is Hermitian when its entries are real.
        let mut real_diag = Vec::new();
        let spectra: Vec<_> = leg
            .sectors()
            .unwrap()
            .into_iter()
            .map(|sector| {
                let k = leg.degeneracy(&sector).unwrap();
                let c = coupled(&sector);
                let values: Vec<$D> =
                    (0..k).map(|i| conv(Complex64::from(entry(0.6, c, i, i).re))).collect();
                real_diag.extend(values.iter().map(|&x| (dim(c), Complex64::from(x))));
                SectorSpectrum { sector, values }
            })
            .collect();
        let real_compact = TensorMap::<_, $D>::diagonal(runtime, &leg, spectra).unwrap();
        let Eigh { d, v } = real_compact.eigh_full(&[0], &[1], tol).unwrap();
        let norm_real = norm_oracle(&real_diag, 2.0);
        assert!(close(d.norm(2.0).unwrap(), norm_real));
        let rebuilt = v.compose(&d).unwrap().compose(&v.adjoint().unwrap()).unwrap();
        assert!(close(rebuilt.norm(2.0).unwrap(), norm_real));
        assert!(close_c(
            Complex64::from(real_compact.inner(&rebuilt).unwrap()),
            inner_oracle(&real_diag, &real_diag),
        ));
        // `a` is not Hermitian; malformed axes are the permute's error.
        assert!(matches!(a.eigh_full(&[0], &[1], tol), Err(Error::Operation(_))));
        assert!(matches!(
            h.eigh_full(&[0], &[0], tol),
            Err(Error::Operation(op)) if matches!(*op, OperationError::InvalidPermutation { .. })
        ));

        $cuda(&a, &b, &lazy_a, &h, alpha, beta);
    }};
}

fn no_cuda<T, D>(_: &T, _: &T, _: &T, _: &T, _: D, _: D) {}

fn u1_leg() -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(0), 2),
            (U1Irrep::new(1), 3),
            (U1Irrep::new(-1), 1),
        ],
    )
    .unwrap()
}

fn su2_leg() -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 3),
            (SU2Irrep::from_twice_spin(2), 1),
        ],
    )
    .unwrap()
}

fn u1_coupled(c: &U1Irrep) -> usize {
    c.charge().unsigned_abs() as usize
}

fn su2_coupled(c: &SU2Irrep) -> usize {
    c.twice_spin()
}

#[test]
fn host_rows_match_the_hand_oracle() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let real = |x: Complex64| x.re;
    let complex = |x: Complex64| x;
    let one = |_: usize| 1.0;
    let su2_dim = |twice: usize| (twice + 1) as f64;
    storage_rows!(&runtime, u1_leg(), u1_coupled, one, f64, real, no_cuda);
    storage_rows!(
        &runtime,
        u1_leg(),
        u1_coupled,
        one,
        Complex64,
        complex,
        no_cuda
    );
    storage_rows!(
        &runtime,
        su2_leg(),
        su2_coupled,
        su2_dim,
        f64,
        real,
        no_cuda
    );
    storage_rows!(
        &runtime,
        su2_leg(),
        su2_coupled,
        su2_dim,
        Complex64,
        complex,
        no_cuda
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn cuda_rows_match_host() {
    use symtn::sector::{CheckedFusionAlgebra, MultiplicityFreeRigidSymbols, SectorCodec};
    use symtn::typed::{CudaFactorizationPayload, CudaStorage};

    fn device_rows<R, D>(
        a: &TensorMap<R, D>,
        b: &TensorMap<R, D>,
        lazy_a: &TensorMap<R, D>,
        h: &TensorMap<R, D>,
        alpha: D,
        beta: D,
    ) where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
        D: CudaFactorizationPayload + Into<Complex64> + std::fmt::Debug,
    {
        let (a_dev, b_dev) = (a.to_cuda().unwrap(), b.to_cuda().unwrap());
        let host = a.norm(2.0).unwrap();
        assert!(close(a_dev.norm(2.0).unwrap(), host));
        assert!(close(lazy_a.to_cuda().unwrap().norm(2.0).unwrap(), host));
        assert!(close(a_dev.adjoint().unwrap().norm(2.0).unwrap(), host));
        assert!(close_c(
            a_dev.inner(&b_dev).unwrap().into(),
            a.inner(b).unwrap().into(),
        ));
        assert_eq!(
            a_dev.norm(3.0).unwrap_err(),
            Error::UnsupportedOnDevice("device norm supports only p = 2, got 3".into())
        );
        assert_eq!(
            a_dev.norm(f64::INFINITY).unwrap_err(),
            Error::UnsupportedOnDevice("device norm supports only p = 2, got inf".into())
        );
        assert_eq!(
            a_dev.inner(&a_dev.adjoint().unwrap()).unwrap_err(),
            Error::UnsupportedOnDevice("inner does not support lazy adjoint CUDA operands".into())
        );
        assert!(matches!(a_dev.norm(-1.0), Err(Error::InvalidArgument(_))));

        // Elementwise: the device result equals the Host result.
        let same = |what: &str, device: TensorMap<R, D, CudaStorage<D>>, host: TensorMap<R, D>| {
            let device = device.to_host().unwrap().materialize().unwrap();
            let host = host.materialize().unwrap();
            assert_eq!(device.codomain(), host.codomain(), "{what}");
            assert_eq!(device.domain(), host.domain(), "{what}");
            let (device, host) = (device.dense_data().unwrap(), host.dense_data().unwrap());
            assert_eq!(device.len(), host.len(), "{what}");
            for (&x, &y) in device.iter().zip(host) {
                assert!(close_c(x.into(), y.into()), "{what}");
            }
        };
        let lazy_dev = a_dev.adjoint().unwrap();
        let lazy_b = b.adjoint().unwrap();
        let lazy_b_dev = b_dev.adjoint().unwrap();
        same(
            "scale",
            a_dev.scale(alpha).unwrap(),
            a.scale(alpha).unwrap(),
        );
        same(
            "axpby",
            a_dev.axpby(alpha, &b_dev, beta).unwrap(),
            a.axpby(alpha, b, beta).unwrap(),
        );
        same(
            "zeros_like",
            a_dev.zeros_like().unwrap(),
            a.zeros_like().unwrap(),
        );
        let mut destination = b_dev.materialize().unwrap();
        a_dev.axpby_into(&mut destination, alpha, beta).unwrap();
        let mut expected = b.materialize().unwrap();
        a.axpby_into(&mut expected, alpha, beta).unwrap();
        same("axpby_into", destination, expected);
        same(
            "lazy scale",
            lazy_dev.scale(alpha).unwrap(),
            lazy_a.scale(alpha).unwrap(),
        );
        same(
            "lazy + lazy",
            lazy_dev.axpby(alpha, &lazy_b_dev, beta).unwrap(),
            lazy_a.axpby(alpha, &lazy_b, beta).unwrap(),
        );
        same(
            "lazy zeros_like",
            lazy_dev.zeros_like().unwrap(),
            lazy_a.zeros_like().unwrap(),
        );
        // One lazy and one owned operand, both orders, and a lazy
        // `axpby_into` source (#1874): the Host rule on the device.
        let owned_b_host = lazy_b.materialize().unwrap();
        let owned_b = owned_b_host.to_cuda().unwrap();
        same(
            "lazy + owned",
            lazy_dev.axpby(alpha, &owned_b, beta).unwrap(),
            lazy_a.axpby(alpha, &owned_b_host, beta).unwrap(),
        );
        same(
            "owned + lazy",
            owned_b.axpby(alpha, &lazy_dev, beta).unwrap(),
            owned_b_host.axpby(alpha, lazy_a, beta).unwrap(),
        );
        let mut destination = owned_b_host.to_cuda().unwrap();
        lazy_dev.axpby_into(&mut destination, alpha, beta).unwrap();
        let mut expected = owned_b_host.materialize().unwrap();
        lazy_a.axpby_into(&mut expected, alpha, beta).unwrap();
        same("lazy axpby_into", destination.clone(), expected);
        // The destination checks still run first (#1756).
        let shared = destination.clone();
        assert_eq!(
            lazy_dev
                .axpby_into(&mut destination, alpha, beta)
                .unwrap_err(),
            Error::DestinationShared
        );
        drop(shared);

        // Compact factors: the device rebuilds, `s` and `r` equal the Host's
        // (both storages fix the Host gauge), and a lazy device operand is
        // the storage's capability error.
        for (rows, cols) in [([0], [1]), ([1], [0])] {
            let (device, host) = (
                a_dev.svd_compact(&rows, &cols).unwrap(),
                a.svd_compact(&rows, &cols).unwrap(),
            );
            same("svd_compact s", device.s.clone(), host.s.clone());
            same(
                "svd_compact",
                device
                    .u
                    .compose(&device.s)
                    .unwrap()
                    .compose(&device.vh)
                    .unwrap(),
                host.u.compose(&host.s).unwrap().compose(&host.vh).unwrap(),
            );
            let (device, host) = (
                a_dev.qr_compact(&rows, &cols).unwrap(),
                a.qr_compact(&rows, &cols).unwrap(),
            );
            same("qr_compact r", device.r.clone(), host.r.clone());
            same(
                "qr_compact",
                device.q.compose(&device.r).unwrap(),
                host.q.compose(&host.r).unwrap(),
            );
        }
        assert_eq!(
            lazy_dev.svd_compact(&[0], &[1]).unwrap_err(),
            Error::UnsupportedOnDevice(
                "svd_compact does not support lazy adjoint CUDA operands".into()
            )
        );
        assert_eq!(
            lazy_dev.qr_compact(&[0], &[1]).unwrap_err(),
            Error::UnsupportedOnDevice(
                "qr_compact does not support lazy adjoint CUDA operands".into()
            )
        );

        // Hermitian eigendecomposition: the device `d` (dense) and `v d v^H`
        // equal the Host's.
        let tol = HermitianTol::DEFAULT;
        let h_dev = h.to_cuda().unwrap();
        for (rows, cols) in [([0], [1]), ([1], [0])] {
            let (device, host) = (
                h_dev.eigh_full(&rows, &cols, tol).unwrap(),
                h.eigh_full(&rows, &cols, tol).unwrap(),
            );
            same("eigh_full d", device.d.clone(), host.d.clone());
            same(
                "eigh_full",
                device
                    .v
                    .compose(&device.d)
                    .unwrap()
                    .compose(&device.v.adjoint().unwrap())
                    .unwrap(),
                host.v
                    .compose(&host.d)
                    .unwrap()
                    .compose(&host.v.adjoint().unwrap())
                    .unwrap(),
            );
        }
        assert_eq!(
            h_dev
                .adjoint()
                .unwrap()
                .eigh_full(&[0], &[1], tol)
                .unwrap_err(),
            Error::UnsupportedOnDevice(
                "eigh_full does not support lazy adjoint CUDA operands".into()
            )
        );
    }

    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let real = |x: Complex64| x.re;
    let complex = |x: Complex64| x;
    let one = |_: usize| 1.0;
    let su2_dim = |twice: usize| (twice + 1) as f64;
    storage_rows!(&runtime, u1_leg(), u1_coupled, one, f64, real, device_rows);
    storage_rows!(
        &runtime,
        u1_leg(),
        u1_coupled,
        one,
        Complex64,
        complex,
        device_rows
    );
    storage_rows!(
        &runtime,
        su2_leg(),
        su2_coupled,
        su2_dim,
        f64,
        real,
        device_rows
    );
    storage_rows!(
        &runtime,
        su2_leg(),
        su2_coupled,
        su2_dim,
        Complex64,
        complex,
        device_rows
    );
}

fn fz2_leg() -> GradedSpace<FermionParityFusionRule> {
    GradedSpace::try_new(
        Arc::new(FermionParityFusionRule),
        [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 3)],
    )
    .unwrap()
}

const TWIST_LEGS: [&[usize]; 5] = [&[0], &[1], &[2], &[0, 2], &[0, 1, 1]];

/// `twist` Host rows on fZ2 `[V, V*] <- [V]`: dense, lazy adjoint and compact
/// operands in both directions against `Σ θ |t|²`, plus the admission rows.
fn twist_rows<D>(runtime: &Runtime, conv: fn(Complex64) -> D)
where
    D: symtn::typed::TensorScalar + Into<Complex64>,
{
    let leg = fz2_leg();
    let dual = leg.try_dual().unwrap();
    // Per written entry: |x|² and the oddness of each leg's uncoupled sector.
    let mut entries: Vec<(f64, [bool; 3])> = Vec::new();
    let a = TensorMap::<_, D>::from_subblock_fn(runtime, [&leg, &dual], [&leg], |trees, ix| {
        let odd = |s: &Z2Irrep| *s == Z2Irrep::ODD;
        let (co, dom) = (trees.codomain_uncoupled(), trees.domain_uncoupled());
        let x = conv(entry(0.7, ix[0] + 2 * ix[1], ix[1], ix[2]));
        let weight = Into::<Complex64>::into(x).norm_sqr();
        entries.push((weight, [odd(&co[0]), odd(&co[1]), odd(&dom[0])]));
        x
    })
    .unwrap();
    let oracle = |legs: &[usize], parity: &dyn Fn(usize) -> usize| -> f64 {
        entries
            .iter()
            .map(|(weight, odd)| {
                let flips = legs.iter().filter(|&&l| odd[parity(l)]).count();
                if flips % 2 == 0 {
                    *weight
                } else {
                    -weight
                }
            })
            .sum()
    };
    let lazy = a.adjoint().unwrap();
    // `a^H` is `[V] <- [V, V*]`: logical leg 0 is parent leg 2, 1 and 2 are 0, 1.
    let to_parent = |l: usize| [2, 0, 1][l];
    for legs in TWIST_LEGS {
        let expected = oracle(legs, &|l| l);
        let expected_lazy = oracle(legs, &to_parent);
        assert!(
            expected.abs() < oracle(&[], &|l| l) - 1e-9,
            "{legs:?}: a block is negated"
        );
        for direction in [Direction::Forward, Direction::Inverse] {
            let twisted = a.twist(legs, direction).unwrap();
            assert!(close_c(a.inner(&twisted).unwrap().into(), expected.into()));
            let twisted = lazy.twist(legs, direction).unwrap();
            assert!(close_c(
                lazy.inner(&twisted).unwrap().into(),
                expected_lazy.into()
            ));
        }
    }
    // Compact: θ(c)^|legs| per sector, so an even leg count is the identity.
    let mut weights = Vec::new();
    let spectra: Vec<_> = leg
        .sectors()
        .unwrap()
        .into_iter()
        .map(|sector| {
            let sign = if sector == Z2Irrep::ODD { -1.0 } else { 1.0 };
            let values: Vec<D> = (0..leg.degeneracy(&sector).unwrap())
                .map(|i| conv(entry(0.3, i, i, i)))
                .collect();
            weights.extend(
                values
                    .iter()
                    .map(|&x| (sign, Into::<Complex64>::into(x).norm_sqr())),
            );
            symtn::typed::SectorSpectrum { sector, values }
        })
        .collect();
    let compact = TensorMap::<_, D>::diagonal(runtime, &leg, spectra).unwrap();
    for (legs, odd) in [(&[0usize][..], true), (&[1], true), (&[0, 1], false)] {
        let expected: f64 = weights
            .iter()
            .map(|&(sign, w)| if odd { sign * w } else { w })
            .sum();
        let twisted = compact.twist(legs, Direction::Forward).unwrap();
        assert!(twisted.dense_data().is_err(), "compact stays compact");
        assert!(close_c(
            compact.inner(&twisted).unwrap().into(),
            expected.into()
        ));
    }
    // Admission: the range check precedes the empty-list short circuit.
    assert!(matches!(
        a.twist(&[3], Direction::Forward),
        Err(Error::InvalidArgument(message)) if message == "twist leg 3 out of range for rank 3"
    ));
    assert!(matches!(
        lazy.twist(&[5], Direction::Inverse),
        Err(Error::InvalidArgument(message)) if message == "inverse twist leg 5 out of range for rank 3"
    ));
    let same = a.twist(&[], Direction::Forward).unwrap();
    assert!(close_c(
        a.inner(&same).unwrap().into(),
        oracle(&[], &|l| l).into()
    ));
}

#[test]
fn twist_host_rows_match_the_hand_oracle() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    twist_rows::<f64>(&runtime, |x| x.re);
    twist_rows::<Complex64>(&runtime, |x| x);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn twist_cuda_rows_match_host() {
    fn rows<D>(runtime: &Runtime, conv: fn(Complex64) -> D)
    where
        D: symtn::typed::CudaPayload + Into<Complex64> + std::fmt::Debug,
    {
        let leg = fz2_leg();
        let dual = leg.try_dual().unwrap();
        let a = TensorMap::<_, D>::from_subblock_fn(runtime, [&leg, &dual], [&leg], |_, ix| {
            conv(entry(0.7, ix[0] + 2 * ix[1], ix[1], ix[2]))
        })
        .unwrap();
        let a_dev = a.to_cuda().unwrap();
        for legs in TWIST_LEGS {
            for direction in [Direction::Forward, Direction::Inverse] {
                for (host, device) in [
                    (a.clone(), a_dev.clone()),
                    (a.adjoint().unwrap(), a_dev.adjoint().unwrap()),
                ] {
                    let device = device.twist(legs, direction).unwrap().to_host().unwrap();
                    let host = host.twist(legs, direction).unwrap();
                    let (device, host) =
                        (device.materialize().unwrap(), host.materialize().unwrap());
                    assert_eq!(device.codomain(), host.codomain(), "{legs:?}");
                    assert_eq!(device.domain(), host.domain(), "{legs:?}");
                    assert_eq!(
                        device.dense_data().unwrap(),
                        host.dense_data().unwrap(),
                        "{legs:?} {direction:?}: every device twist is ±1, so exact"
                    );
                }
            }
        }
        assert!(matches!(
            a_dev.twist(&[3], Direction::Forward),
            Err(Error::InvalidArgument(_))
        ));
    }
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    rows::<f64>(&runtime, |x| x.re);
    rows::<Complex64>(&runtime, |x| x);
}

/// `adjoint`/`adjoint_view`/`materialize` Host rows on `[V, V*] <- [V]`:
/// dense, lazy adjoint and compact operands. The lazy materialization is
/// checked through `inner(a^H, b^H) = conj(inner(a, b))`, which pins the
/// conjugation, and through the exact round trip `(a^H)^H = a`.
macro_rules! materialize_rows {
    ($runtime:expr, $leg:expr, $dim:expr, $D:ty, $conv:expr) => {{
        let runtime: &Runtime = $runtime;
        let leg = $leg;
        let dual = leg.try_dual().unwrap();
        let dim = $dim;
        let conv = $conv;
        let build = |seed: f64| {
            let mut entries = Vec::new();
            let t = TensorMap::<_, $D>::from_subblock_fn(
                runtime,
                [&leg, &dual],
                [&leg],
                |trees, ix| {
                    let x: $D = conv(entry(seed, ix[0] + 2 * ix[1], ix[1], ix[2]));
                    entries.push((dim(trees), Complex64::from(x)));
                    x
                },
            )
            .unwrap();
            (t, entries)
        };
        let (a, a_entries) = build(0.9);
        let (b, b_entries) = build(-0.4);

        let owned = a.materialize().unwrap();
        assert_eq!(owned.dense_data().unwrap(), a.dense_data().unwrap());
        assert_ne!(
            owned.dense_data().unwrap().as_ptr(),
            a.dense_data().unwrap().as_ptr(),
            "a fresh payload"
        );

        let lazy_a = a.adjoint().unwrap();
        let lazy_b = b.adjoint().unwrap();
        assert_eq!(lazy_a.codomain(), a.domain());
        assert_eq!(lazy_a.domain(), a.codomain());
        assert!(lazy_a.dense_data().is_err(), "dense adjoint is lazy");
        assert_eq!(
            lazy_a.adjoint().unwrap().dense_data().unwrap().as_ptr(),
            a.dense_data().unwrap().as_ptr(),
            "adjoint twice is the parent"
        );
        let (ma, mb) = (lazy_a.materialize().unwrap(), lazy_b.materialize().unwrap());
        assert!(close(ma.norm(2.0).unwrap(), norm_oracle(&a_entries, 2.0)));
        let expected = inner_oracle(&a_entries, &b_entries).conj();
        assert!(close_c(ma.inner(&mb).unwrap().into(), expected));
        assert_eq!(
            ma.adjoint()
                .unwrap()
                .materialize()
                .unwrap()
                .dense_data()
                .unwrap(),
            a.dense_data().unwrap(),
            "exact round trip"
        );

        // The borrowed view is the operand `adjoint` gives.
        assert_eq!(
            a.compose(a.adjoint_view()).unwrap().dense_data().unwrap(),
            a.compose(&lazy_a).unwrap().dense_data().unwrap()
        );

        // Compact: the adjoint stays compact; materialize densifies.
        let spectra: Vec<_> = leg
            .sectors()
            .unwrap()
            .into_iter()
            .enumerate()
            .map(|(s, sector)| SectorSpectrum {
                sector,
                values: (0..leg.degeneracy(&sector).unwrap())
                    .map(|i| conv(entry(0.3, s, i, i)))
                    .collect(),
            })
            .collect();
        let compact = TensorMap::<_, $D>::diagonal(runtime, &leg, spectra).unwrap();
        let adjoint = compact.adjoint().unwrap();
        assert!(
            adjoint.dense_data().is_err(),
            "compact adjoint stays compact"
        );
        for t in [&compact, &adjoint] {
            let dense = t.materialize().unwrap();
            assert!(dense.dense_data().is_ok());
            assert!(close(dense.norm(2.0).unwrap(), t.norm(2.0).unwrap()));
        }
        // `inner(d^H, d) = Σ x²` and `inner(d, d^H) = conj(Σ x²)`.
        let forward: Complex64 = adjoint.inner(&compact).unwrap().into();
        let backward: Complex64 = compact.inner(&adjoint).unwrap().into();
        assert!(close_c(forward, backward.conj()));
    }};
}

/// The quantum dimension of an Abelian coupled sector.
fn one<S>(_: &symtn::typed::BlockFusionTrees<S>) -> f64 {
    1.0
}

#[test]
fn materialize_host_rows_match_the_hand_oracle() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let su2_dim = |trees: &symtn::typed::BlockFusionTrees<SU2Irrep>| {
        (trees.coupled().twice_spin() + 1) as f64
    };
    let (real, complex) = (|x: Complex64| x.re, |x: Complex64| x);
    materialize_rows!(&runtime, u1_leg(), one, f64, real);
    materialize_rows!(&runtime, u1_leg(), one, Complex64, complex);
    materialize_rows!(&runtime, su2_leg(), su2_dim, f64, real);
    materialize_rows!(&runtime, su2_leg(), su2_dim, Complex64, complex);
    materialize_rows!(&runtime, fz2_leg(), one, f64, real);
    materialize_rows!(&runtime, fz2_leg(), one, Complex64, complex);
}

/// The leg motions of the transform rows on a `[V, V*] <- [V]` fixture:
/// `(permute codomain, permute domain)` and its inverse.
const PERMUTE: (&[usize], &[usize]) = (&[2, 0], &[1]);
const PERMUTE_BACK: (&[usize], &[usize]) = (&[1, 2], &[0]);

/// The planar order (codomain, then the domain reversed) of a
/// `(codomain_rank, rank)` split rotated by `by`, as `transpose` axes on the
/// same split.
fn rotation(codomain_rank: usize, rank: usize, by: usize) -> (Vec<usize>, Vec<usize>) {
    let planar: Vec<usize> = (0..codomain_rank)
        .chain((codomain_rank..rank).rev())
        .collect();
    let rotated: Vec<usize> = (0..rank).map(|i| planar[(i + by) % rank]).collect();
    let codomain = rotated[..codomain_rank].to_vec();
    let domain = rotated[codomain_rank..].iter().rev().copied().collect();
    (codomain, domain)
}

/// `permute`/`braid`/`transpose`/`repartition` and their `_into` Host rows
/// on `[V, V*] <- [V]`: dense, lazy adjoint and compact operands. Every
/// motion is unitary, so it is checked through `norm` and
/// `inner(T a, T b) = inner(a, b)` against the hand oracle and through its
/// inverse; `_into` through `inner(T a, alpha T a + beta T b)`. Braiding the
/// leg-swap with levels is the permute for the symmetric braidings here.
macro_rules! transform_rows {
    ($runtime:expr, $leg:expr, $dim:expr, $D:ty, $conv:expr, $cuda:expr) => {{
        let runtime: &Runtime = $runtime;
        let leg = $leg;
        let dual = leg.try_dual().unwrap();
        let dim = $dim;
        let conv = $conv;
        let build = |seed: f64| {
            let mut entries = Vec::new();
            let t = TensorMap::<_, $D>::from_subblock_fn(
                runtime,
                [&leg, &dual],
                [&leg],
                |trees, ix| {
                    let x: $D = conv(entry(seed, ix[0] + 2 * ix[1], ix[1], ix[2]));
                    entries.push((dim(trees), Complex64::from(x)));
                    x
                },
            )
            .unwrap();
            (t, entries)
        };
        let (a, a_entries) = build(0.9);
        let (b, b_entries) = build(-0.4);
        let (alpha, beta) = (conv(Complex64::new(2.0, 0.5)), conv(Complex64::new(-0.5, 0.25)));
        let (alpha_c, beta_c) = (Complex64::from(alpha), Complex64::from(beta));
        let a_norm = norm_oracle(&a_entries, 2.0);
        let ab = inner_oracle(&a_entries, &b_entries);
        let aa = inner_oracle(&a_entries, &a_entries);

        let motions: [(&str, fn(&TensorMap<_, $D>) -> TensorMap<_, $D>, fn(&TensorMap<_, $D>) -> TensorMap<_, $D>); 4] = [
            (
                "permute",
                |t| t.permute(PERMUTE.0, PERMUTE.1).unwrap(),
                |t| t.permute(PERMUTE_BACK.0, PERMUTE_BACK.1).unwrap(),
            ),
            (
                "braid",
                |t| t.braid(PERMUTE.0, PERMUTE.1, &[0, 1, 2]).unwrap(),
                |t| t.braid(PERMUTE_BACK.0, PERMUTE_BACK.1, &[0, 1, 2]).unwrap(),
            ),
            (
                "transpose",
                |t| {
                    let (codomain, domain) = rotation(t.codomain_rank(), t.rank(), 1);
                    t.transpose(&codomain, &domain).unwrap()
                },
                |t| {
                    let (codomain, domain) = rotation(t.codomain_rank(), t.rank(), t.rank() - 1);
                    t.transpose(&codomain, &domain).unwrap()
                },
            ),
            (
                "repartition",
                |t| t.repartition(t.rank() - t.codomain_rank()).unwrap(),
                |t| t.repartition(t.rank() - t.codomain_rank()).unwrap(),
            ),
        ];
        let lazy_a = a.adjoint().unwrap();
        let lazy_b = b.adjoint().unwrap();
        for (name, forward, back) in motions {
            let (ta, tb) = (forward(&a), forward(&b));
            assert!(close(ta.norm(2.0).unwrap(), a_norm), "{name}: norm");
            assert!(close_c(ta.inner(&tb).unwrap().into(), ab), "{name}: inner");
            let round = back(&ta);
            assert_eq!(round.codomain(), a.codomain(), "{name}: round trip");
            assert!(close(round.norm(2.0).unwrap(), a_norm), "{name}: round trip");
            assert!(close_c(round.inner(&a).unwrap().into(), aa), "{name}: round trip");

            // Lazy adjoint: the transformed parent stays lazy.
            let (la, lb) = (forward(&lazy_a), forward(&lazy_b));
            assert!(la.dense_data().is_err(), "{name}: lazy stays lazy");
            assert!(close(la.norm(2.0).unwrap(), a_norm), "{name}: lazy norm");
            let (la, lb) = (la.materialize().unwrap(), lb.materialize().unwrap());
            assert!(close_c(la.inner(&lb).unwrap().into(), ab.conj()), "{name}: lazy inner");
        }

        // `_into`: dst = alpha T a + beta T b, from an owned and from a lazy source.
        let into = |source: &TensorMap<_, $D>, name: &str, destination: &mut TensorMap<_, $D>| match name {
            "permute" => source.permute_into(PERMUTE.0, PERMUTE.1, destination, alpha, beta),
            "braid" => source.braid_into(PERMUTE.0, PERMUTE.1, &[0, 1, 2], destination, alpha, beta),
            "transpose" => {
                let (codomain, domain) = rotation(source.codomain_rank(), source.rank(), 1);
                source.transpose_into(&codomain, &domain, destination, alpha, beta)
            }
            _ => source.repartition_into(destination, alpha, beta),
        };
        for (name, forward, _) in motions {
            let mut destination = forward(&b);
            into(&a, name, &mut destination).unwrap();
            let expected = alpha_c * aa + beta_c * ab;
            assert!(close_c(forward(&a).inner(&destination).unwrap().into(), expected), "{name}_into");

            let mut destination = forward(&lazy_b).materialize().unwrap();
            into(&lazy_a, name, &mut destination).unwrap();
            let expected = alpha_c * aa + beta_c * ab.conj();
            assert!(
                close_c(forward(&lazy_a).materialize().unwrap().inner(&destination).unwrap().into(), expected),
                "{name}_into lazy"
            );
        }

        // Compact rank-(1,1): the leg swap stays compact, and `_into` writes
        // its transformed diagonal; both preserve the norm.
        let spectra: Vec<_> = leg
            .sectors()
            .unwrap()
            .into_iter()
            .enumerate()
            .map(|(s, sector)| SectorSpectrum {
                sector,
                values: (0..leg.degeneracy(&sector).unwrap())
                    .map(|i| conv(entry(0.3, s, i, i)))
                    .collect(),
            })
            .collect();
        let compact = TensorMap::<_, $D>::diagonal(runtime, &leg, spectra).unwrap();
        let swapped = compact.permute(&[1], &[0]).unwrap();
        assert!(swapped.dense_data().is_err(), "compact swap stays compact");
        let norm = compact.norm(2.0).unwrap();
        assert!(close(swapped.norm(2.0).unwrap(), norm));
        let mut destination = swapped.materialize().unwrap();
        compact
            .permute_into(&[1], &[0], &mut destination, conv(Complex64::new(1.0, 0.0)), conv(Complex64::new(1.0, 0.0)))
            .unwrap();
        assert!(close(destination.norm(2.0).unwrap(), 2.0 * norm));

        // Admission: malformed axes, mis-lengthed levels, an aliasing, a
        // mismatched and a shared destination.
        assert!(matches!(
            a.permute(&[0, 0], &[1]),
            Err(Error::Operation(op)) if matches!(*op, OperationError::InvalidPermutation { .. })
        ));
        assert!(matches!(
            a.braid(PERMUTE.0, PERMUTE.1, &[0]),
            Err(Error::InvalidArgument(message))
                if message == "braid levels must list one level per source axis (expected 3, got 1)"
        ));
        let mut alias = a.clone();
        assert!(matches!(
            a.permute_into(PERMUTE.0, PERMUTE.1, &mut alias, alpha, beta),
            Err(Error::InvalidArgument(_))
        ));
        let mut mismatched = b.clone().materialize().unwrap();
        assert!(matches!(
            a.permute_into(PERMUTE.0, PERMUTE.1, &mut mismatched, alpha, beta),
            Err(Error::Operation(op)) if matches!(*op, OperationError::SpaceMismatch { .. })
        ));
        let mut shared = b.permute(PERMUTE.0, PERMUTE.1).unwrap();
        let _keep = shared.clone();
        assert!(matches!(
            a.permute_into(PERMUTE.0, PERMUTE.1, &mut shared, alpha, beta),
            Err(Error::DestinationShared)
        ));

        $cuda(&a, &b, alpha, beta);
    }};
}

fn no_cuda_transform<T, D>(_: &T, _: &T, _: D, _: D) {}

#[test]
fn transform_host_rows_match_the_hand_oracle() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let su2_dim = |trees: &symtn::typed::BlockFusionTrees<SU2Irrep>| {
        (trees.coupled().twice_spin() + 1) as f64
    };
    let (real, complex) = (|x: Complex64| x.re, |x: Complex64| x);
    transform_rows!(&runtime, u1_leg(), one, f64, real, no_cuda_transform);
    transform_rows!(
        &runtime,
        u1_leg(),
        one,
        Complex64,
        complex,
        no_cuda_transform
    );
    transform_rows!(&runtime, su2_leg(), su2_dim, f64, real, no_cuda_transform);
    transform_rows!(
        &runtime,
        su2_leg(),
        su2_dim,
        Complex64,
        complex,
        no_cuda_transform
    );
    transform_rows!(&runtime, fz2_leg(), one, f64, real, no_cuda_transform);
    transform_rows!(
        &runtime,
        fz2_leg(),
        one,
        Complex64,
        complex,
        no_cuda_transform
    );
}

/// Device results of the transform rows against Host: owned and lazy
/// sources, eager and `_into`, within the executor's dtype tolerance (the
/// recoupling GEMM sums in cuTENSOR's order), and the admission errors with
/// the storage noun naming the placement.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn transform_cuda_rows_match_host() {
    use symtn::sector::{CheckedFusionAlgebra, MultiplicityFreeRigidSymbols, SectorCodec};

    fn device_rows<R, D>(a: &TensorMap<R, D>, b: &TensorMap<R, D>, alpha: D, beta: D)
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
        D: symtn::typed::CudaPayload + Into<Complex64> + std::fmt::Debug,
    {
        let agree = |device: &[D], host: &[D], what: &str| {
            assert_eq!(device.len(), host.len(), "{what}");
            for (&x, &y) in device.iter().zip(host) {
                let (x, y): (Complex64, Complex64) = (x.into(), y.into());
                assert!(
                    (x - y).norm() <= 1e-12 * (1.0 + y.norm()),
                    "{what}: {x} != {y}"
                );
            }
        };
        let (a_dev, b_dev) = (a.to_cuda().unwrap(), b.to_cuda().unwrap());
        for (host, device, host_b, device_b) in [
            (a.clone(), a_dev.clone(), b.clone(), b_dev.clone()),
            (
                a.adjoint().unwrap(),
                a_dev.adjoint().unwrap(),
                b.adjoint().unwrap(),
                b_dev.adjoint().unwrap(),
            ),
        ] {
            let rotated = rotation(host.codomain_rank(), host.rank(), 1);
            let flipped = host.rank() - host.codomain_rank();
            let pairs = [
                (
                    "permute",
                    host.permute(PERMUTE.0, PERMUTE.1).unwrap(),
                    device.permute(PERMUTE.0, PERMUTE.1).unwrap(),
                ),
                (
                    "braid",
                    host.braid(PERMUTE.0, PERMUTE.1, &[0, 1, 2]).unwrap(),
                    device.braid(PERMUTE.0, PERMUTE.1, &[0, 1, 2]).unwrap(),
                ),
                (
                    "transpose",
                    host.transpose(&rotated.0, &rotated.1).unwrap(),
                    device.transpose(&rotated.0, &rotated.1).unwrap(),
                ),
                (
                    "repartition",
                    host.repartition(flipped).unwrap(),
                    device.repartition(flipped).unwrap(),
                ),
            ];
            for (name, host, device) in pairs {
                let (host, device) = (
                    host.materialize().unwrap(),
                    device.to_host().unwrap().materialize().unwrap(),
                );
                assert_eq!(device.codomain(), host.codomain(), "{name}");
                assert_eq!(device.domain(), host.domain(), "{name}");
                agree(
                    device.dense_data().unwrap(),
                    host.dense_data().unwrap(),
                    name,
                );
            }
            // `_into` from this source into `T b`.
            let mut host_destination = host_b
                .permute(PERMUTE.0, PERMUTE.1)
                .unwrap()
                .materialize()
                .unwrap();
            let mut device_destination = device_b
                .permute(PERMUTE.0, PERMUTE.1)
                .unwrap()
                .materialize()
                .unwrap();
            host.permute_into(PERMUTE.0, PERMUTE.1, &mut host_destination, alpha, beta)
                .unwrap();
            device
                .permute_into(PERMUTE.0, PERMUTE.1, &mut device_destination, alpha, beta)
                .unwrap();
            agree(
                device_destination.to_host().unwrap().dense_data().unwrap(),
                host_destination.dense_data().unwrap(),
                "permute_into",
            );
        }
        // Admission, as on Host with the storage noun.
        assert!(matches!(
            a_dev.permute(&[0, 0], &[1]),
            Err(Error::Operation(op)) if matches!(*op, OperationError::InvalidPermutation { .. })
        ));
        let mut alias = a_dev.clone();
        assert!(matches!(
            a_dev.permute_into(PERMUTE.0, PERMUTE.1, &mut alias, alpha, beta),
            Err(Error::InvalidArgument(message)) if message == "destination storage must not alias an input"
        ));
        let mut lazy_destination = b_dev
            .permute(PERMUTE.0, PERMUTE.1)
            .unwrap()
            .adjoint()
            .unwrap();
        assert!(matches!(
            a_dev.permute_into(PERMUTE.0, PERMUTE.1, &mut lazy_destination, alpha, beta),
            Err(Error::InvalidArgument(message)) if message == "destination must use ordinary dense CUDA storage"
        ));
        let mut mismatched = b_dev.clone();
        assert!(matches!(
            a_dev.permute_into(PERMUTE.0, PERMUTE.1, &mut mismatched, alpha, beta),
            Err(Error::Operation(op)) if matches!(*op, OperationError::SpaceMismatch { .. })
        ));
    }
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let su2_dim = |trees: &symtn::typed::BlockFusionTrees<SU2Irrep>| {
        (trees.coupled().twice_spin() + 1) as f64
    };
    let (real, complex) = (|x: Complex64| x.re, |x: Complex64| x);
    transform_rows!(&runtime, u1_leg(), one, f64, real, device_rows);
    transform_rows!(&runtime, u1_leg(), one, Complex64, complex, device_rows);
    transform_rows!(&runtime, su2_leg(), su2_dim, f64, real, device_rows);
    transform_rows!(
        &runtime,
        su2_leg(),
        su2_dim,
        Complex64,
        complex,
        device_rows
    );
    transform_rows!(&runtime, fz2_leg(), one, f64, real, device_rows);
    transform_rows!(&runtime, fz2_leg(), one, Complex64, complex, device_rows);
}

/// Mixed owned/lazy `axpby` (both orders) and a lazy `axpby_into` source on
/// rank-3 `[V, V*] <- [V]` (U(1), fermion parity) and `[V, V] <- [V]` (SU(2),
/// several fusion trees per coupled sector): the lazy operand's blocks are
/// strided, permuted views of its parent. The device equals the Host rule at
/// every coefficient pattern, including a zero `beta` over a NaN destination
/// (the NaN must not survive) and a zero `alpha` (#1874).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn mixed_lazy_axpby_cuda_rows_match_host() {
    use symtn::sector::{CheckedFusionAlgebra, MultiplicityFreeRigidSymbols, SectorCodec};
    use symtn::typed::{CudaPayload, CudaStorage};

    fn same<R, D>(what: &str, device: TensorMap<R, D, CudaStorage<D>>, host: TensorMap<R, D>)
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
        D: CudaPayload + Into<Complex64>,
    {
        let device = device.to_host().unwrap();
        assert!(device.dense_data().is_ok(), "{what}: owned dense result");
        let host = host.materialize().unwrap();
        assert_eq!(device.codomain(), host.codomain(), "{what}");
        assert_eq!(device.domain(), host.domain(), "{what}");
        let (device, host) = (device.dense_data().unwrap(), host.dense_data().unwrap());
        assert_eq!(device.len(), host.len(), "{what}");
        for (&x, &y) in device.iter().zip(host) {
            let (x, y): (Complex64, Complex64) = (x.into(), y.into());
            assert!(x.re.is_finite() && x.im.is_finite(), "{what}: {x}");
            assert!(close_c(x, y), "{what}: {x} vs {y}");
        }
    }

    fn rows<R, D>(a: &TensorMap<R, D>, b: &TensorMap<R, D>, coefficients: &[(D, D)], nan: D)
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
        D: CudaPayload + Into<Complex64> + std::fmt::Debug,
    {
        let lazy = a.adjoint().unwrap();
        let owned = b.adjoint().unwrap().materialize().unwrap();
        let lazy_dev = a.to_cuda().unwrap().adjoint().unwrap();
        let owned_dev = owned.to_cuda().unwrap();
        let poisoned = owned.scale(nan).unwrap();
        for &(alpha, beta) in coefficients {
            let what = format!("alpha {alpha:?} beta {beta:?}");
            same(
                &format!("lazy + owned, {what}"),
                lazy_dev.axpby(alpha, &owned_dev, beta).unwrap(),
                lazy.axpby(alpha, &owned, beta).unwrap(),
            );
            same(
                &format!("owned + lazy, {what}"),
                owned_dev.axpby(alpha, &lazy_dev, beta).unwrap(),
                owned.axpby(alpha, &lazy, beta).unwrap(),
            );
            // A zero `beta` overwrites a NaN destination on both storages.
            let zero_beta = Into::<Complex64>::into(beta) == Complex64::new(0.0, 0.0);
            let start = if zero_beta { &poisoned } else { &owned };
            let mut device = start.to_cuda().unwrap();
            lazy_dev.axpby_into(&mut device, alpha, beta).unwrap();
            let mut host = start.materialize().unwrap();
            lazy.axpby_into(&mut host, alpha, beta).unwrap();
            same(&format!("lazy axpby_into, {what}"), device, host);
        }
    }

    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    /// A value fixed by the block's trees and the local index, distinct
    /// across blocks, so a wrong parent block cannot pass.
    fn fill<S: std::fmt::Debug>(
        trees: &symtn::typed::BlockFusionTrees<S>,
        ix: &[usize],
    ) -> Complex64 {
        let key = format!(
            "{:?}{:?}{:?}{:?}{:?}{ix:?}",
            trees.codomain_uncoupled(),
            trees.codomain_innerlines(),
            trees.domain_uncoupled(),
            trees.domain_innerlines(),
            trees.coupled(),
        );
        let hash = key.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
        });
        Complex64::new(
            ((hash % 2001) as f64 - 1000.0) / 257.0,
            (((hash >> 20) % 2001) as f64 - 1000.0) / 311.0,
        )
    }
    fn fill_b<S: std::fmt::Debug>(
        trees: &symtn::typed::BlockFusionTrees<S>,
        ix: &[usize],
    ) -> Complex64 {
        fill(trees, ix).conj() * Complex64::new(-0.6, 0.9) + Complex64::new(0.25, 0.0)
    }
    macro_rules! both_dtypes {
        ($codomain:expr, $domain:expr) => {{
            let real: Vec<(f64, f64)> = vec![(0.7, -1.1), (0.0, -1.1), (0.7, 0.0), (1.0, 1.0)];
            let a: TensorMap<_, f64> =
                TensorMap::from_subblock_fn(&runtime, $codomain, $domain, |t, ix| fill(t, ix).re)
                    .unwrap();
            let b: TensorMap<_, f64> =
                TensorMap::from_subblock_fn(&runtime, $codomain, $domain, |t, ix| fill_b(t, ix).re)
                    .unwrap();
            rows(&a, &b, &real, f64::NAN);
            let complex: Vec<(Complex64, Complex64)> = vec![
                (Complex64::new(0.7, -0.3), Complex64::new(-1.1, 0.4)),
                (Complex64::new(0.0, 0.0), Complex64::new(-1.1, 0.4)),
                (Complex64::new(0.7, -0.3), Complex64::new(0.0, 0.0)),
                (Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)),
            ];
            let a: TensorMap<_, Complex64> =
                TensorMap::from_subblock_fn(&runtime, $codomain, $domain, fill).unwrap();
            let b: TensorMap<_, Complex64> =
                TensorMap::from_subblock_fn(&runtime, $codomain, $domain, fill_b).unwrap();
            rows(&a, &b, &complex, Complex64::new(f64::NAN, f64::NAN));
        }};
    }
    let u1 = u1_leg();
    let u1_dual = u1.try_dual().unwrap();
    both_dtypes!([&u1, &u1_dual], [&u1]);
    let su2 = su2_leg();
    both_dtypes!([&su2, &su2], [&su2]);
    let fz2 = fz2_leg();
    let fz2_dual = fz2.try_dual().unwrap();
    both_dtypes!([&fz2, &fz2_dual], [&fz2]);
}

/// `contract`/`compose`/`contract_into` Host rows on `a, b: [V, V*] <- [V]`
/// with `x = a^H` owned and lazy. Composition is checked against the hand
/// oracle through `tr(a^H ∘ b) = inner(a, b)` (TensorKit `tr`/`dot`:
/// `Σ_c dim(c) tr(block(a, c)^H block(b, c))`); `contract` through its
/// documented relation `compose(x, y) = contract(x, twist(y, dual contracted
/// legs))`, so `tr(contract(a^H, twist(b, [1]))) = inner(a, b)` on every
/// braiding; a general output order through `contract` followed by
/// `permute`; `contract_into` through `tr(alpha a^H b + beta a^H a) =
/// alpha inner(a, b) + beta inner(a, a)`. A compact operand's arms are
/// compared with the dense route of its materialized image.
macro_rules! contract_rows {
    ($runtime:expr, $leg:expr, $dim:expr, $D:ty, $conv:expr, $cuda:expr) => {{
        let runtime: &Runtime = $runtime;
        let leg = $leg;
        let dual = leg.try_dual().unwrap();
        let dim = $dim;
        let conv = $conv;
        let build = |seed: f64| {
            let mut entries = Vec::new();
            let t = TensorMap::<_, $D>::from_subblock_fn(
                runtime,
                [&leg, &dual],
                [&leg],
                |trees, ix| {
                    let x: $D = conv(entry(seed, ix[0] + 2 * ix[1], ix[1], ix[2]));
                    entries.push((dim(trees), Complex64::from(x)));
                    x
                },
            )
            .unwrap();
            (t, entries)
        };
        let (a, a_entries) = build(0.9);
        let (b, b_entries) = build(-0.4);
        let (one, minus_one) = (conv(Complex64::new(1.0, 0.0)), conv(Complex64::new(-1.0, 0.0)));
        let (alpha, beta) = (conv(Complex64::new(2.0, 0.5)), conv(Complex64::new(-0.5, 0.25)));
        let (alpha_c, beta_c) = (Complex64::from(alpha), Complex64::from(beta));
        let ab = inner_oracle(&a_entries, &b_entries);
        let aa = inner_oracle(&a_entries, &a_entries);
        let agree = |x: &TensorMap<_, $D>, y: &TensorMap<_, $D>| {
            x.axpby(one, y, minus_one).unwrap().norm(2.0).unwrap()
                <= 1e-12 * (1.0 + y.norm(2.0).unwrap())
        };

        let twisted_b = b.twist(&[1], Direction::Forward).unwrap();
        for (name, x) in [
            ("owned", a.adjoint().unwrap().materialize().unwrap()),
            ("lazy", a.adjoint().unwrap()),
        ] {
            let composed = x.compose(&b).unwrap();
            assert!(close_c(composed.tr().unwrap().into(), ab), "{name}: compose");
            let contracted = x.contract(&twisted_b, &CANONICAL).unwrap();
            assert!(close_c(contracted.tr().unwrap().into(), ab), "{name}: contract");
            let flat = x.contract(&twisted_b, &FLAT).unwrap();
            let permuted = contracted.permute(FLAT.codomain, FLAT.domain).unwrap();
            assert_eq!(flat.codomain(), permuted.codomain(), "{name}: flat");
            assert!(agree(&flat, &permuted), "{name}: flat = contract then permute");

            let mut destination = x.compose(&a).unwrap();
            x.contract_into(&twisted_b, &CANONICAL, &mut destination, alpha, beta)
                .unwrap();
            let expected = alpha_c * ab + beta_c * aa;
            assert!(close_c(destination.tr().unwrap().into(), expected), "{name}: contract_into");
        }

        // Compact rank-(1,1) operand: the bond-scaling arms against the dense
        // route of its materialized image.
        let spectra: Vec<_> = leg
            .sectors()
            .unwrap()
            .into_iter()
            .enumerate()
            .map(|(s, sector)| SectorSpectrum {
                sector,
                values: (0..leg.degeneracy(&sector).unwrap())
                    .map(|i| conv(entry(0.3, s, i, i)))
                    .collect(),
            })
            .collect();
        let compact = TensorMap::<_, $D>::diagonal(runtime, &leg, spectra).unwrap();
        let dense = compact.materialize().unwrap();
        let scaled = b.contract(&dense, &SCALE).unwrap();
        assert!(agree(&b.contract(&compact, &SCALE).unwrap(), &scaled), "compact contract");
        assert!(agree(&b.compose(&compact).unwrap(), &b.compose(&dense).unwrap()), "compact compose");
        let mut destination = b.contract(&dense, &SCALE).unwrap();
        b.contract_into(&compact, &SCALE, &mut destination, alpha, beta)
            .unwrap();
        let expected = scaled.scale(conv(alpha_c + beta_c)).unwrap();
        assert!(agree(&destination, &expected), "compact contract_into");

        // Admission: malformed axes, an aliasing, a lazy, a mismatched and a
        // shared destination, each leaving the destination untouched.
        let x = a.adjoint().unwrap().materialize().unwrap();
        let malformed = ContractSpec { lhs: &[1, 1], ..CANONICAL };
        assert!(matches!(
            x.contract(&b, &malformed),
            Err(Error::Operation(op)) if matches!(*op, OperationError::InvalidAxisSet { .. })
        ));
        let mut alias = b.clone();
        assert!(matches!(
            x.contract_into(&b, &CANONICAL, &mut alias, alpha, beta),
            Err(Error::InvalidArgument(message)) if message == "destination storage must not alias an input"
        ));
        let mut lazy = x.compose(&a).unwrap().adjoint().unwrap();
        assert!(matches!(
            x.contract_into(&b, &CANONICAL, &mut lazy, alpha, beta),
            Err(Error::InvalidArgument(message)) if message == "destination must use ordinary dense host storage"
        ));
        let mut mismatched = a.scale(one).unwrap();
        assert!(matches!(
            x.contract_into(&b, &CANONICAL, &mut mismatched, alpha, beta),
            Err(Error::Operation(op)) if matches!(*op, OperationError::SpaceMismatch { .. })
        ));
        assert!(agree(&mismatched, &a), "a rejected destination is untouched");
        let mut shared = x.compose(&a).unwrap();
        let keep = shared.clone();
        assert!(matches!(
            x.contract_into(&b, &CANONICAL, &mut shared, alpha, beta),
            Err(Error::DestinationShared)
        ));
        assert!(agree(&keep, &x.compose(&a).unwrap()), "a shared destination is untouched");

        $cuda(&a, &b, alpha, beta);
    }};
}

/// `x[1, 2]` against `y[0, 1]`: the canonical composition geometry.
const CANONICAL: ContractSpec<'static> = ContractSpec {
    lhs: &[1, 2],
    rhs: &[0, 1],
    codomain: &[0],
    domain: &[1],
};

/// [`CANONICAL`] with both open legs in the codomain, reversed.
const FLAT: ContractSpec<'static> = ContractSpec {
    lhs: &[1, 2],
    rhs: &[0, 1],
    codomain: &[1, 0],
    domain: &[],
};

/// `t[2]` against a rank-(1,1) bond operand's codomain leg.
const SCALE: ContractSpec<'static> = ContractSpec {
    lhs: &[2],
    rhs: &[0],
    codomain: &[0, 1],
    domain: &[2],
};

fn no_cuda_contract<T, D>(_: &T, _: &T, _: D, _: D) {}

#[test]
fn contract_host_rows_match_the_hand_oracle() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let su2_dim = |trees: &symtn::typed::BlockFusionTrees<SU2Irrep>| {
        (trees.coupled().twice_spin() + 1) as f64
    };
    let (real, complex) = (|x: Complex64| x.re, |x: Complex64| x);
    contract_rows!(&runtime, u1_leg(), one, f64, real, no_cuda_contract);
    contract_rows!(
        &runtime,
        u1_leg(),
        one,
        Complex64,
        complex,
        no_cuda_contract
    );
    contract_rows!(&runtime, su2_leg(), su2_dim, f64, real, no_cuda_contract);
    contract_rows!(
        &runtime,
        su2_leg(),
        su2_dim,
        Complex64,
        complex,
        no_cuda_contract
    );
    contract_rows!(&runtime, fz2_leg(), one, f64, real, no_cuda_contract);
    contract_rows!(
        &runtime,
        fz2_leg(),
        one,
        Complex64,
        complex,
        no_cuda_contract
    );
}

/// Device results of the contract rows against Host: owned and lazy `a^H`,
/// `compose`, `contract` in the canonical and a general output order,
/// `contract_into`, within the executor's dtype tolerance (GEMMs sum in
/// cuTENSOR's order), and the admission errors with the storage noun naming
/// the placement.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn contract_cuda_rows_match_host() {
    use symtn::sector::{CheckedFusionAlgebra, MultiplicityFreeRigidSymbols, SectorCodec};

    fn device_rows<R, D>(a: &TensorMap<R, D>, b: &TensorMap<R, D>, alpha: D, beta: D)
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
        D: symtn::typed::CudaPayload + Into<Complex64> + std::fmt::Debug,
    {
        let agree = |device: &TensorMap<R, D, symtn::typed::CudaStorage<D>>,
                     host: &TensorMap<R, D>,
                     what: &str| {
            let (device, host) = (
                device.to_host().unwrap().materialize().unwrap(),
                host.materialize().unwrap(),
            );
            assert_eq!(device.codomain(), host.codomain(), "{what}");
            assert_eq!(device.domain(), host.domain(), "{what}");
            let (device, host) = (device.dense_data().unwrap(), host.dense_data().unwrap());
            assert_eq!(device.len(), host.len(), "{what}");
            for (&x, &y) in device.iter().zip(host) {
                let (x, y): (Complex64, Complex64) = (x.into(), y.into());
                assert!(
                    (x - y).norm() <= 1e-12 * (1.0 + y.norm()),
                    "{what}: {x} != {y}"
                );
            }
        };
        let (a_dev, b_dev) = (a.to_cuda().unwrap(), b.to_cuda().unwrap());
        let (tb, tb_dev) = (
            b.twist(&[1], Direction::Forward).unwrap(),
            b_dev.twist(&[1], Direction::Forward).unwrap(),
        );
        for (host, device) in [
            (
                a.adjoint().unwrap().materialize().unwrap(),
                a_dev.adjoint().unwrap().materialize().unwrap(),
            ),
            (a.adjoint().unwrap(), a_dev.adjoint().unwrap()),
        ] {
            agree(
                &device.compose(&b_dev).unwrap(),
                &host.compose(b).unwrap(),
                "compose",
            );
            for spec in [CANONICAL, FLAT] {
                agree(
                    &device.contract(&tb_dev, &spec).unwrap(),
                    &host.contract(&tb, &spec).unwrap(),
                    "contract",
                );
            }
            let mut host_destination = host.compose(a).unwrap();
            let mut device_destination = device.compose(&a_dev).unwrap();
            host.contract_into(&tb, &CANONICAL, &mut host_destination, alpha, beta)
                .unwrap();
            device
                .contract_into(&tb_dev, &CANONICAL, &mut device_destination, alpha, beta)
                .unwrap();
            agree(&device_destination, &host_destination, "contract_into");
        }
        // Admission, as on Host with the storage noun.
        let x = a_dev.adjoint().unwrap().materialize().unwrap();
        let mut alias = b_dev.clone();
        assert!(matches!(
            x.contract_into(&b_dev, &CANONICAL, &mut alias, alpha, beta),
            Err(Error::InvalidArgument(message)) if message == "destination storage must not alias an input"
        ));
        let mut lazy = x.compose(&a_dev).unwrap().adjoint().unwrap();
        assert!(matches!(
            x.contract_into(&b_dev, &CANONICAL, &mut lazy, alpha, beta),
            Err(Error::InvalidArgument(message)) if message == "destination must use ordinary dense CUDA storage"
        ));
        let mut mismatched = a_dev.clone().to_host().unwrap().to_cuda().unwrap();
        assert!(matches!(
            x.contract_into(&b_dev, &CANONICAL, &mut mismatched, alpha, beta),
            Err(Error::Operation(op)) if matches!(*op, OperationError::SpaceMismatch { .. })
        ));
    }
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let su2_dim = |trees: &symtn::typed::BlockFusionTrees<SU2Irrep>| {
        (trees.coupled().twice_spin() + 1) as f64
    };
    let (real, complex) = (|x: Complex64| x.re, |x: Complex64| x);
    contract_rows!(&runtime, u1_leg(), one, f64, real, device_rows);
    contract_rows!(&runtime, u1_leg(), one, Complex64, complex, device_rows);
    contract_rows!(&runtime, su2_leg(), su2_dim, f64, real, device_rows);
    contract_rows!(
        &runtime,
        su2_leg(),
        su2_dim,
        Complex64,
        complex,
        device_rows
    );
    contract_rows!(&runtime, fz2_leg(), one, f64, real, device_rows);
    contract_rows!(&runtime, fz2_leg(), one, Complex64, complex, device_rows);
}

/// `trace_pairs`/`trace_pairs_into` rows on `a, b: [V, V] <- [V, V]`, owned
/// and lazy `a^H`. The hand oracle sums the written entries whose codomain
/// and domain basis states coincide, `Σ_c dim(c) Σ_k θ_k block(a, c)[k, k]`,
/// where every traced pair contributes the twist `θ` of its codomain leg's
/// sector (the supertrace: −1 for an odd fermion parity, else 1). The full
/// trace over both pairs is that sum with both twists; the partial trace
/// over `(0, 2)` has the sum with the first twist as its own `tr`; a lazy
/// adjoint gives the conjugate; and `trace_pairs_into` gives
/// `tr(alpha · trace(x) + beta · dst₀) = alpha tr(x) + beta tr(b)`.
/// A compact source's arms are compared with the dense route of its
/// materialized image. Admission pins the Host order: destination kind and
/// alias before the destination space, so a destination that is both of the
/// wrong kind (or aliases the source) and of the wrong space reports the
/// kind (alias) error.
macro_rules! trace_rows {
    ($runtime:expr, $leg:expr, $dim:expr, $theta:expr, $D:ty, $conv:expr, $cuda:expr) => {{
        let runtime: &Runtime = $runtime;
        let leg = $leg;
        let (dim, theta) = ($dim, $theta);
        let conv = $conv;
        let build = |seed: f64| {
            let (mut partial, mut full) = (Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0));
            let t = TensorMap::<_, $D>::from_subblock_fn(
                runtime,
                [&leg, &leg],
                [&leg, &leg],
                |trees, ix| {
                    let x: $D = conv(entry(seed, ix[0] + 2 * ix[1], ix[2], ix[3]));
                    if trees.codomain_uncoupled() == trees.domain_uncoupled()
                        && trees.codomain_innerlines() == trees.domain_innerlines()
                        && ix[0] == ix[2]
                        && ix[1] == ix[3]
                    {
                        let co = trees.codomain_uncoupled();
                        let x = dim(trees) * theta(&co[0]) * Complex64::from(x);
                        partial += x;
                        full += theta(&co[1]) * x;
                    }
                    x
                },
            )
            .unwrap();
            (t, partial, full)
        };
        let (a, a_partial, a_full) = build(0.9);
        let (b, b_partial, _) = build(-0.4);
        let (one, minus_one) = (conv(Complex64::new(1.0, 0.0)), conv(Complex64::new(-1.0, 0.0)));
        let (alpha, beta) = (conv(Complex64::new(2.0, 0.5)), conv(Complex64::new(-0.5, 0.25)));
        let (alpha_c, beta_c) = (Complex64::from(alpha), Complex64::from(beta));
        let agree = |x: &TensorMap<_, $D>, y: &TensorMap<_, $D>| {
            x.axpby(one, y, minus_one).unwrap().norm(2.0).unwrap()
                <= 1e-12 * (1.0 + y.norm(2.0).unwrap())
        };
        let tr = |t: &TensorMap<_, $D>| -> Complex64 { t.tr().unwrap().into() };

        for (name, x, (expected_partial, expected_full)) in [
            ("owned", a.clone(), (a_partial, a_full)),
            ("lazy", a.adjoint().unwrap(), (a_partial.conj(), a_full.conj())),
        ] {
            let partial = x.trace_pairs(&[(0, 2)]).unwrap();
            assert_eq!(partial.codomain().len() + partial.domain().len(), 2, "{name}");
            assert!(close_c(tr(&partial), expected_partial), "{name}: partial trace");
            let full = x.trace_pairs(&[(0, 2), (1, 3)]).unwrap();
            assert!(close_c(full.scalar().unwrap().into(), expected_full), "{name}: full trace");
            let mut destination = b.trace_pairs(&[(0, 2)]).unwrap();
            x.trace_pairs_into(&[(0, 2)], &mut destination, alpha, beta)
                .unwrap();
            let expected = alpha_c * expected_partial + beta_c * b_partial;
            assert!(close_c(tr(&destination), expected), "{name}: trace_pairs_into");
        }

        // Compact rank-(1,1) source: the spectrum arm (full trace) and the
        // materializing arm against the dense route of its materialized image.
        let spectra: Vec<_> = leg
            .sectors()
            .unwrap()
            .into_iter()
            .enumerate()
            .map(|(s, sector)| SectorSpectrum {
                sector,
                values: (0..leg.degeneracy(&sector).unwrap())
                    .map(|i| conv(entry(0.3, s, i, i)))
                    .collect(),
            })
            .collect();
        let compact = TensorMap::<_, $D>::diagonal(runtime, &leg, spectra).unwrap();
        let dense = compact.materialize().unwrap();
        let value: Complex64 = dense.trace_pairs(&[(0, 1)]).unwrap().scalar().unwrap().into();
        let compact_value: Complex64 =
            compact.trace_pairs(&[(0, 1)]).unwrap().scalar().unwrap().into();
        assert!(close_c(compact_value, value), "compact trace_pairs");
        let mut destination = dense.trace_pairs(&[(0, 1)]).unwrap();
        compact
            .trace_pairs_into(&[(0, 1)], &mut destination, alpha, beta)
            .unwrap();
        let scalar: Complex64 = destination.scalar().unwrap().into();
        assert!(close_c(scalar, (alpha_c + beta_c) * value), "compact trace_pairs_into");

        // Admission, each rejected destination left untouched.
        assert!(matches!(
            a.trace_pairs(&[(0, 0)]),
            Err(Error::Operation(op)) if matches!(*op, OperationError::InvalidAxisSet { .. })
        ));
        // Doubly invalid: the source's own storage, of the wrong space.
        let mut alias = a.clone();
        assert!(matches!(
            a.trace_pairs_into(&[(0, 2)], &mut alias, alpha, beta),
            Err(Error::InvalidArgument(message)) if message == "destination storage must not alias an input"
        ));
        // Doubly invalid: a lazy adjoint, of the wrong space.
        let mut lazy = b.adjoint().unwrap();
        assert!(matches!(
            a.trace_pairs_into(&[(0, 2)], &mut lazy, alpha, beta),
            Err(Error::InvalidArgument(message)) if message == "destination must use ordinary dense host storage"
        ));
        let mut mismatched = b.scale(one).unwrap();
        assert!(matches!(
            a.trace_pairs_into(&[(0, 2)], &mut mismatched, alpha, beta),
            Err(Error::Operation(op)) if matches!(*op, OperationError::SpaceMismatch { .. })
        ));
        assert!(agree(&mismatched, &b), "a rejected destination is untouched");
        let mut shared = b.trace_pairs(&[(0, 2)]).unwrap();
        let keep = shared.clone();
        assert!(matches!(
            a.trace_pairs_into(&[(0, 2)], &mut shared, alpha, beta),
            Err(Error::DestinationShared)
        ));
        assert!(agree(&keep, &b.trace_pairs(&[(0, 2)]).unwrap()), "a shared destination is untouched");

        $cuda(&a, &b, alpha, beta);
    }};
}

/// The twist of a fermion parity sector: −1 when odd.
fn odd_twist(s: &Z2Irrep) -> f64 {
    if *s == Z2Irrep::ODD {
        -1.0
    } else {
        1.0
    }
}

#[test]
fn trace_host_rows_match_the_hand_oracle() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let su2_dim = |trees: &symtn::typed::BlockFusionTrees<SU2Irrep>| {
        (trees.coupled().twice_spin() + 1) as f64
    };
    let (real, complex) = (|x: Complex64| x.re, |x: Complex64| x);
    trace_rows!(
        &runtime,
        u1_leg(),
        one,
        |_: &U1Irrep| 1.0,
        f64,
        real,
        no_cuda_contract
    );
    trace_rows!(
        &runtime,
        u1_leg(),
        one,
        |_: &U1Irrep| 1.0,
        Complex64,
        complex,
        no_cuda_contract
    );
    trace_rows!(
        &runtime,
        su2_leg(),
        su2_dim,
        |_: &SU2Irrep| 1.0,
        f64,
        real,
        no_cuda_contract
    );
    trace_rows!(
        &runtime,
        su2_leg(),
        su2_dim,
        |_: &SU2Irrep| 1.0,
        Complex64,
        complex,
        no_cuda_contract
    );
    trace_rows!(
        &runtime,
        fz2_leg(),
        one,
        odd_twist,
        f64,
        real,
        no_cuda_contract
    );
    trace_rows!(
        &runtime,
        fz2_leg(),
        one,
        odd_twist,
        Complex64,
        complex,
        no_cuda_contract
    );
}

/// Device results of the trace rows against Host: owned and lazy `a^H`,
/// partial and full `trace_pairs`, `trace_pairs_into`, within the
/// executor's dtype tolerance (cuTENSOR's sum order), and the admission
/// order with the storage noun naming the placement: kind and alias before
/// the destination space (#1870 C).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn trace_cuda_rows_match_host() {
    use symtn::sector::{CheckedFusionAlgebra, MultiplicityFreeRigidSymbols, SectorCodec};

    fn device_rows<R, D>(a: &TensorMap<R, D>, b: &TensorMap<R, D>, alpha: D, beta: D)
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
        D: symtn::typed::CudaPayload + Into<Complex64> + std::fmt::Debug,
    {
        let agree = |device: &TensorMap<R, D, symtn::typed::CudaStorage<D>>,
                     host: &TensorMap<R, D>,
                     what: &str| {
            let (device, host) = (
                device.to_host().unwrap().materialize().unwrap(),
                host.materialize().unwrap(),
            );
            assert_eq!(device.codomain(), host.codomain(), "{what}");
            assert_eq!(device.domain(), host.domain(), "{what}");
            let (device, host) = (device.dense_data().unwrap(), host.dense_data().unwrap());
            assert_eq!(device.len(), host.len(), "{what}");
            for (&x, &y) in device.iter().zip(host) {
                let (x, y): (Complex64, Complex64) = (x.into(), y.into());
                assert!(
                    (x - y).norm() <= 1e-12 * (1.0 + y.norm()),
                    "{what}: {x} != {y}"
                );
            }
        };
        let (a_dev, b_dev) = (a.to_cuda().unwrap(), b.to_cuda().unwrap());
        for (host, device) in [
            (a.clone(), a_dev.clone()),
            (a.adjoint().unwrap(), a_dev.adjoint().unwrap()),
        ] {
            for pairs in [&[(0, 2)][..], &[(0, 2), (1, 3)]] {
                agree(
                    &device.trace_pairs(pairs).unwrap(),
                    &host.trace_pairs(pairs).unwrap(),
                    "trace_pairs",
                );
            }
            let mut host_destination = b.trace_pairs(&[(0, 2)]).unwrap();
            let mut device_destination = b_dev.trace_pairs(&[(0, 2)]).unwrap();
            host.trace_pairs_into(&[(0, 2)], &mut host_destination, alpha, beta)
                .unwrap();
            device
                .trace_pairs_into(&[(0, 2)], &mut device_destination, alpha, beta)
                .unwrap();
            agree(&device_destination, &host_destination, "trace_pairs_into");
        }
        // Admission, as on Host with the storage noun.
        let mut alias = a_dev.clone();
        assert!(matches!(
            a_dev.trace_pairs_into(&[(0, 2)], &mut alias, alpha, beta),
            Err(Error::InvalidArgument(message)) if message == "destination storage must not alias an input"
        ));
        let mut lazy = b_dev.adjoint().unwrap();
        assert!(matches!(
            a_dev.trace_pairs_into(&[(0, 2)], &mut lazy, alpha, beta),
            Err(Error::InvalidArgument(message)) if message == "destination must use ordinary dense CUDA storage"
        ));
        let mut mismatched = b_dev.clone().to_host().unwrap().to_cuda().unwrap();
        assert!(matches!(
            a_dev.trace_pairs_into(&[(0, 2)], &mut mismatched, alpha, beta),
            Err(Error::Operation(op)) if matches!(*op, OperationError::SpaceMismatch { .. })
        ));
        agree(&mismatched, b, "a rejected destination is untouched");
    }
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let su2_dim = |trees: &symtn::typed::BlockFusionTrees<SU2Irrep>| {
        (trees.coupled().twice_spin() + 1) as f64
    };
    let (real, complex) = (|x: Complex64| x.re, |x: Complex64| x);
    trace_rows!(
        &runtime,
        u1_leg(),
        one,
        |_: &U1Irrep| 1.0,
        f64,
        real,
        device_rows
    );
    trace_rows!(
        &runtime,
        u1_leg(),
        one,
        |_: &U1Irrep| 1.0,
        Complex64,
        complex,
        device_rows
    );
    trace_rows!(
        &runtime,
        su2_leg(),
        su2_dim,
        |_: &SU2Irrep| 1.0,
        f64,
        real,
        device_rows
    );
    trace_rows!(
        &runtime,
        su2_leg(),
        su2_dim,
        |_: &SU2Irrep| 1.0,
        Complex64,
        complex,
        device_rows
    );
    trace_rows!(&runtime, fz2_leg(), one, odd_twist, f64, real, device_rows);
    trace_rows!(
        &runtime,
        fz2_leg(),
        one,
        odd_twist,
        Complex64,
        complex,
        device_rows
    );
}
