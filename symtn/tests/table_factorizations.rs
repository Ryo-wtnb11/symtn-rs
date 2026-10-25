//! Factorizations and truncation of complex-symbol rules (#2281): SVD, QR/LQ,
//! polar, eigh/eig, null spaces, matrix functions and the
//! `svd_compact` → `find_truncated` → `restrict_leg` chain on Fibonacci and
//! Ising through `TableRule`, plus a planar contraction, the golden chain and
//! the Ising σ-chain.
//!
//! The oracles are readable in this file: reconstruction and isometry
//! identities, the `dim(c)`-weighted trace identities of a blockwise
//! spectrum (`Σ_c d_c Σ σ² = ‖t‖²`, `Σ_c d_c Σ λ = tr t`), TensorKit's
//! `SectorVector` truncation rule written out over that spectrum, rank–nullity
//! for null spaces, the golden chain diagonalised from a Hamiltonian written
//! directly in the fusion-path basis with the closed-form `F^{τττ}_τ`
//! (Feiguin et al., PRL 98, 160409), and the Ising σ-chain as the critical
//! transverse-field Ising chain, whose open-chain spectrum is the free-fermion
//! closed form. A few values checked against TensorKit 0.17.1 anchor the
//! weighted conventions.
//!
//! Operands are filled with the tree-hash `entry` of `support/table_fixture.rs`,
//! so a value checked against TensorKit is reproducible from the same
//! coordinates (Ising entries there carry AnyonWiki's vertex sign
//! `u^{σσ}_ψ = −1`, #2277, so both sides are the same morphism).

use std::f64::consts::SQRT_2;
use std::sync::Arc;

use num_complex::Complex64;
use symtn::sector::{SU2FusionRule, SU2Irrep};
use symtn::typed::{
    ContractSpec, EighFullPlan, GradedSpace, HermitianTol, LeftPolar, Lq, OperationError, Qr,
    RightPolar, Runtime, SectorSpectrum, StackedTensorMap, Svd, TensorMap, Truncation,
};
use symtn_sectors::{TableRule, TableSector};

use crate::table_fixture;
use table_fixture::*;

type Table = TensorMap<TableRule, Complex64>;

/// Spectral values are backward stable to a few dozen ulps of the largest
/// one; `64 · 64 · ε` leaves room for a few dozen accumulated terms.
fn close(got: f64, want: f64, scale: f64) -> bool {
    (got - want).abs() <= 4096.0 * f64::EPSILON * scale.max(1.0)
}

/// `‖x − y‖₂`, the dimension-weighted Frobenius norm.
fn distance(x: &Table, y: &Table) -> f64 {
    x.axpby(Complex64::ONE, y, -Complex64::ONE)
        .unwrap()
        .norm(2.0)
        .unwrap()
}

fn assert_near(what: &str, x: &Table, y: &Table) {
    let scale = y.norm(2.0).unwrap().max(1.0);
    let residual = distance(x, y);
    assert!(
        residual <= 1.0e-12 * scale,
        "{what}: residual {residual:e} (scale {scale})"
    );
}

fn identity(runtime: &Runtime, legs: &[GradedSpace<TableRule>]) -> Table {
    TensorMap::isomorphism(runtime, legs, legs).unwrap()
}

/// `w^H w = id` on the domain of `w`.
fn assert_isometry(what: &str, runtime: &Runtime, w: &Table) {
    let gram = w
        .adjoint()
        .unwrap()
        .materialize()
        .unwrap()
        .compose(w)
        .unwrap();
    assert_near(what, &gram, &identity(runtime, &w.domain()));
}

/// Nonzero degeneracies of a leg, by AnyonWiki label.
fn degeneracies(leg: &GradedSpace<TableRule>) -> Vec<(u16, usize)> {
    leg.sectors()
        .unwrap()
        .into_iter()
        .map(|sector| (sector.label(), leg.degeneracy(&sector).unwrap()))
        .filter(|&(_, n)| n > 0)
        .collect()
}

struct Case {
    runtime: Runtime,
    v: GradedSpace<TableRule>,
    /// `d_c` by AnyonWiki label.
    dim: fn(u16) -> f64,
    /// `dim_c(V ⊗ V)` by coupled sector: the block sizes of `V⊗V ← V⊗V`.
    pair_blocks: &'static [(u16, usize)],
}

impl Case {
    fn new(
        rule: TableRule,
        sectors: &[(u16, usize)],
        dim: fn(u16) -> f64,
        pair_blocks: &'static [(u16, usize)],
    ) -> Self {
        let v = GradedSpace::try_new(
            Arc::new(rule),
            sectors
                .iter()
                .map(|&(label, degeneracy)| (TableSector::new(label), degeneracy)),
        )
        .unwrap();
        Self {
            runtime: Runtime::builder().build().unwrap(),
            v,
            dim,
            pair_blocks,
        }
    }

    /// `V = 𝟙 ⊕ τ²`, `d_τ = φ`; `V ⊗ V` has `1 + 2·2 = 5` states in `𝟙` and
    /// `2 + 2 + 2·2 = 8` in `τ`.
    fn fibonacci() -> Self {
        Self::new(
            TableRule::fibonacci(),
            &[(1, 1), (2, 2)],
            |label| match label {
                1 => 1.0,
                _ => (1.0 + 5.0_f64.sqrt()) / 2.0,
            },
            &[(1, 5), (2, 8)],
        )
    }

    /// `V = 𝟙 ⊕ ψ ⊕ σ²`, `d_ψ = 1`, `d_σ = √2`; `V ⊗ V` has `1 + 1 + 2·2 = 6`
    /// states in `𝟙` and in `ψ`, and `4 · 2 = 8` in `σ`.
    fn ising() -> Self {
        Self::new(
            TableRule::ising(),
            &[(1, 1), (2, 1), (3, 2)],
            |label| match label {
                3 => SQRT_2,
                _ => 1.0,
            },
            &[(1, 6), (2, 6), (3, 8)],
        )
    }

    fn filled(&self, tag: &'static str, codomain: usize, domain: usize) -> Table {
        let v = &self.v;
        TensorMap::from_subblock_fn(
            &self.runtime,
            vec![v; codomain],
            vec![v; domain],
            |trees, index| entry(tag, trees, index),
        )
        .unwrap()
    }

    /// `Σ_c d_c Σ_i f(x_{c,i})` over a blockwise spectrum.
    fn weighted_sum<V: Copy, S: std::iter::Sum<S> + std::ops::Mul<f64, Output = S>>(
        &self,
        spectra: &[SectorSpectrum<TableSector, V>],
        f: impl Fn(V) -> S,
    ) -> S {
        spectra
            .iter()
            .map(|entry| {
                entry.values.iter().map(|&x| f(x)).sum::<S>() * (self.dim)(entry.sector.label())
            })
            .sum()
    }

    /// One spectrum per coupled sector of `V ⊗ V`, each the block's size.
    fn assert_pair_blocks<V>(&self, what: &str, spectra: &[SectorSpectrum<TableSector, V>]) {
        let sizes: Vec<(u16, usize)> = spectra
            .iter()
            .map(|entry| (entry.sector.label(), entry.values.len()))
            .collect();
        assert_eq!(sizes, self.pair_blocks, "{what}: block sizes");
    }
}

/// The weighted trace identities of a blockwise spectrum:
/// `Σ_c d_c Σ σ² = ‖t‖²` (SVD, which also holds for the lazy adjoint),
/// `Σ_c d_c Σ λ = tr h` and `Σ_c d_c Σ λ² = ‖h‖²` (eigh of `h = t + t^H`),
/// and `Σ_c d_c Σ λ = tr t` (eig).
fn spectra_satisfy_weighted_trace_identities(case: &Case) {
    let t = case.filled("t", 2, 2);
    let norm2 = t.norm(2.0).unwrap().powi(2);
    for (what, svd) in [
        ("svd_vals", t.svd_vals(&[0, 1], &[2, 3]).unwrap()),
        (
            "svd_vals of the adjoint",
            t.adjoint().unwrap().svd_vals(&[0, 1], &[2, 3]).unwrap(),
        ),
    ] {
        case.assert_pair_blocks(what, &svd);
        for entry in &svd {
            assert!(
                entry.values.windows(2).all(|w| w[0] >= w[1]) && entry.values[0] >= 0.0,
                "{what}: descending and non-negative"
            );
        }
        let got = case.weighted_sum(&svd, |s| s * s);
        assert!(close(got, norm2, norm2), "{what}: {got} != ‖t‖² {norm2}");
    }

    let hermitian = t
        .axpby(Complex64::ONE, t.adjoint_view(), Complex64::ONE)
        .unwrap();
    let eigh = hermitian
        .eigh_vals(&[0, 1], &[2, 3], HermitianTol::DEFAULT)
        .unwrap();
    case.assert_pair_blocks("eigh_vals", &eigh);
    let (trace, h2) = (
        hermitian.tr().unwrap(),
        hermitian.norm(2.0).unwrap().powi(2),
    );
    let got = case.weighted_sum(&eigh, |x| x);
    assert!(
        close(got, trace.re, h2),
        "eigh: Σ d λ {got} != tr h {trace}"
    );
    let got = case.weighted_sum(&eigh, |x| x * x);
    assert!(close(got, h2, h2), "eigh: Σ d λ² {got} != ‖h‖² {h2}");

    let eig = t.eig_vals(&[0, 1], &[2, 3]).unwrap();
    case.assert_pair_blocks("eig_vals", &eig);
    let got = case.weighted_sum(&eig, |z: Complex64| z);
    let trace = t.tr().unwrap();
    assert!(
        (got - trace).norm() <= 4096.0 * f64::EPSILON * norm2,
        "eig: Σ d λ {got} != tr t {trace}"
    );
}

#[test]
fn fibonacci_spectra_satisfy_weighted_trace_identities() {
    spectra_satisfy_weighted_trace_identities(&Case::fibonacci());
}

#[test]
fn ising_spectra_satisfy_weighted_trace_identities() {
    spectra_satisfy_weighted_trace_identities(&Case::ising());
}

/// Equal sector labels and, per sector, equal values in order.
fn assert_same_spectra<U: Copy + Into<Complex64>, W: Copy + Into<Complex64>>(
    what: &str,
    got: &[SectorSpectrum<TableSector, U>],
    want: &[SectorSpectrum<TableSector, W>],
) {
    assert_eq!(got.len(), want.len(), "{what}: sectors");
    for (g, w) in got.iter().zip(want) {
        assert_eq!(g.sector, w.sector, "{what}: sector labels");
        assert_eq!(g.values.len(), w.values.len(), "{what}: {:?}", g.sector);
        let scale = w
            .values
            .iter()
            .fold(0.0_f64, |m, &v| m.max(v.into().norm()));
        for (&x, &y) in g.values.iter().zip(&w.values) {
            assert!(
                (x.into() - y.into()).norm() <= 4096.0 * f64::EPSILON * scale.max(1.0),
                "{what}: sector {:?}",
                g.sector
            );
        }
    }
}

/// Rank–nullity per coupled sector: the null space of a map whose `V ⊗ V`
/// side has `dim_c(V⊗V)` states has `dim_c(V⊗V) − rank_c` of them, `rank_c`
/// the number of nonzero singular values of the block.
fn assert_rank_nullity(
    what: &str,
    case: &Case,
    spectra: &[SectorSpectrum<TableSector>],
    null_leg: &GradedSpace<TableRule>,
) {
    let largest = spectra
        .iter()
        .flat_map(|entry| entry.values.iter().copied())
        .fold(0.0, f64::max);
    let want: Vec<(u16, usize)> = case
        .pair_blocks
        .iter()
        .map(|&(label, states)| {
            let rank = spectra
                .iter()
                .filter(|entry| entry.sector.label() == label)
                .flat_map(|entry| entry.values.iter())
                .filter(|&&s| s > 1.0e-12 * largest)
                .count();
            (label, states - rank)
        })
        .filter(|&(_, n)| n > 0)
        .collect();
    assert_eq!(degeneracies(null_leg), want, "{what}: rank–nullity");
}
fn decompositions_reconstruct(case: &Case) {
    let runtime = &case.runtime;
    let t = case.filled("t", 2, 2);
    let tall = case.filled("x", 2, 1);
    let wide = case.filled("y", 1, 2);

    let Svd { u, s, vh } = t.svd_compact(&[0, 1], &[2, 3]).unwrap();
    assert_near("u s vh", &u.compose(&s).unwrap().compose(&vh).unwrap(), &t);
    assert_isometry("svd u", runtime, &u);
    assert_isometry(
        "svd vh^H",
        runtime,
        &vh.adjoint().unwrap().materialize().unwrap(),
    );
    let Svd { u, s, vh } = t.svd_full(&[0, 1], &[2, 3]).unwrap();
    assert_near(
        "full u s vh",
        &u.compose(&s).unwrap().compose(&vh).unwrap(),
        &t,
    );
    assert_isometry("full svd u", runtime, &u);

    for (what, Qr { q, r }) in [
        ("qr_compact", tall.qr_compact(&[0, 1], &[2]).unwrap()),
        ("qr_full", tall.qr_full(&[0, 1], &[2]).unwrap()),
    ] {
        assert_near(what, &q.compose(&r).unwrap(), &tall);
        assert_isometry(what, runtime, &q);
    }
    for (what, Lq { l, q }) in [
        ("lq_compact", wide.lq_compact(&[0], &[1, 2]).unwrap()),
        ("lq_full", wide.lq_full(&[0], &[1, 2]).unwrap()),
    ] {
        assert_near(what, &l.compose(&q).unwrap(), &wide);
        assert_isometry(what, runtime, &q.adjoint().unwrap().materialize().unwrap());
    }
    let LeftPolar { w, p } = tall.left_polar(&[0, 1], &[2]).unwrap();
    assert_near("left polar", &w.compose(&p).unwrap(), &tall);
    assert_isometry("left polar w", runtime, &w);
    let RightPolar { p, wh } = wide.right_polar(&[0], &[1, 2]).unwrap();
    assert_near("right polar", &p.compose(&wh).unwrap(), &wide);
    assert_isometry(
        "right polar wh^H",
        runtime,
        &wh.adjoint().unwrap().materialize().unwrap(),
    );

    let hermitian = t
        .axpby(Complex64::ONE, t.adjoint_view(), Complex64::ONE)
        .unwrap();
    let eigh = hermitian
        .eigh_full(&[0, 1], &[2, 3], HermitianTol::DEFAULT)
        .unwrap();
    assert_near(
        "v d v^H",
        &eigh
            .v
            .compose(&eigh.d)
            .unwrap()
            .compose(eigh.v.adjoint_view())
            .unwrap(),
        &hermitian,
    );
    assert_isometry("eigh v", runtime, &eigh.v);
    // The values-only spectra are the factor's diagonal, sector by sector.
    assert_same_spectra(
        "eigh_vals vs diag(eigh d)",
        &hermitian
            .eigh_vals(&[0, 1], &[2, 3], HermitianTol::DEFAULT)
            .unwrap(),
        &eigh.d.diagview().unwrap(),
    );
    let eig = t.eig_full(&[0, 1], &[2, 3]).unwrap();
    assert_near(
        "t v = v d",
        &t.compose(&eig.v).unwrap(),
        &eig.v.compose(&eig.d).unwrap(),
    );
    assert_same_spectra(
        "eig_vals vs diag(eig d)",
        &t.eig_vals(&[0, 1], &[2, 3]).unwrap(),
        &eig.d.diagview().unwrap(),
    );
    // `t v = v d` alone holds for `v = 0`: `v` must be invertible.
    assert_near(
        "inv(eig v) eig v",
        &eig.v.inv(&[0, 1], &[2]).unwrap().compose(&eig.v).unwrap(),
        &identity(runtime, &eig.v.domain()),
    );

    // exp(h) = v exp(d) v^H; inv(t) t = id; t solve(t, y) = y.
    let small = hermitian.scale(Complex64::new(0.25, 0.0)).unwrap();
    let small_eigh = small
        .eigh_full(&[0, 1], &[2, 3], HermitianTol::DEFAULT)
        .unwrap();
    let spectral_exp = small_eigh
        .v
        .compose(&small_eigh.d.map_diagonal(|x| x.exp()).unwrap())
        .unwrap()
        .compose(small_eigh.v.adjoint_view())
        .unwrap();
    assert_near("exp", &small.exp(&[0, 1], &[2, 3]).unwrap(), &spectral_exp);
    let id = identity(runtime, &t.domain());
    assert_near(
        "inv t",
        &t.inv(&[0, 1], &[2, 3]).unwrap().compose(&t).unwrap(),
        &id,
    );
    let rhs = case.filled("b", 2, 1);
    let solved = t.solve(&[0, 1], &[2, 3], &rhs, &[0, 1], &[2]).unwrap();
    assert_near("t solve", &t.compose(&solved).unwrap(), &rhs);

    // Null spaces: orthonormal, annihilated by the input, of the size
    // rank–nullity gives.
    let n = tall.left_null(&[0, 1], &[2]).unwrap();
    assert_isometry("left null", runtime, &n);
    assert!(
        n.adjoint()
            .unwrap()
            .materialize()
            .unwrap()
            .compose(&tall)
            .unwrap()
            .norm(2.0)
            .unwrap()
            < 1.0e-12
    );
    assert_rank_nullity(
        "left null",
        case,
        &tall.svd_vals(&[0, 1], &[2]).unwrap(),
        &n.domain()[0],
    );
    let n = wide.right_null(&[0], &[1, 2]).unwrap();
    assert_isometry(
        "right null",
        runtime,
        &n.adjoint().unwrap().materialize().unwrap(),
    );
    assert!(wide.compose(n.adjoint_view()).unwrap().norm(2.0).unwrap() < 1.0e-12);
    assert_rank_nullity(
        "right null",
        case,
        &wide.svd_vals(&[0], &[1, 2]).unwrap(),
        &n.codomain()[0],
    );
}

/// Moore–Penrose: `t · pinv(t) · t = t`, for a wide map whose `pinv` is
/// not an inverse.
fn pseudo_inverse_reproduces_the_input(case: &Case) {
    let wide = case.filled("y", 1, 2);
    let pinv = wide.pinv(&[0], &[1, 2], 1.0e-12).unwrap();
    let roundtrip = wide.compose(&pinv).unwrap().compose(&wide).unwrap();
    assert_near("t pinv(t) t", &roundtrip, &wide);
}

#[test]
fn fibonacci_pseudo_inverse_reproduces_the_input() {
    pseudo_inverse_reproduces_the_input(&Case::fibonacci());
}

#[test]
fn ising_pseudo_inverse_reproduces_the_input() {
    pseudo_inverse_reproduces_the_input(&Case::ising());
}

#[test]
fn fibonacci_decompositions_reconstruct() {
    decompositions_reconstruct(&Case::fibonacci());
}

#[test]
fn ising_decompositions_reconstruct() {
    decompositions_reconstruct(&Case::ising());
}

/// TensorKit's `SectorVector` truncation (`findtruncated`), written out over
/// the `d_c`-weighted spectrum: every policy keeps a prefix of the values in
/// descending order. `rank(n)` keeps values while the running `Σ d_c` stays
/// `≤ n`; `absolute_cutoff(a)` keeps `σ ≥ a`
/// (MatrixAlgebraKit `findtruncated_svd`, `searchsortedlast` with `rev = true`); an error budget `ε` discards
/// from the smallest value while the discarded `Σ d_c σ²` stays `≤ ε²`. The
/// reported error is `√(Σ_discarded d_c σ²)`.
enum ByHand {
    Rank(f64),
    Cutoff(f64),
    Error(f64),
}

fn truncate_by_hand(
    case: &Case,
    spectra: &[SectorSpectrum<TableSector>],
    policy: &ByHand,
) -> (Vec<(u16, usize)>, f64) {
    let mut flat: Vec<(f64, u16)> = spectra
        .iter()
        .flat_map(|entry| entry.values.iter().map(|&s| (s, entry.sector.label())))
        .collect();
    flat.sort_by(|x, y| y.0.total_cmp(&x.0));
    let d = |label: u16| (case.dim)(label);
    let kept = match *policy {
        ByHand::Rank(n) => {
            let mut total = 0.0;
            flat.iter()
                .take_while(|&&(_, c)| {
                    total += d(c);
                    total <= n
                })
                .count()
        }
        ByHand::Cutoff(a) => flat.iter().take_while(|&&(s, _)| s >= a).count(),
        ByHand::Error(eps) => {
            let mut discarded = 0.0;
            let dropped = flat
                .iter()
                .rev()
                .take_while(|&&(s, c)| {
                    discarded += d(c) * s * s;
                    discarded <= eps * eps
                })
                .count();
            flat.len() - dropped
        }
    };
    let counts = case
        .pair_blocks
        .iter()
        .map(|&(label, _)| (label, flat[..kept].iter().filter(|e| e.1 == label).count()))
        .filter(|&(_, n)| n > 0)
        .collect();
    let error = flat[kept..]
        .iter()
        .map(|&(s, c)| d(c) * s * s)
        .sum::<f64>()
        .sqrt();
    (counts, error)
}

/// The golden-chain DMRG step: `svd_compact` → `find_truncated` →
/// `restrict_leg`, weighted by `dim(c)`, against the rule written out above.
/// `kept` are TensorKit's `svd_trunc` dimensions for `truncrank(7)`,
/// `trunctol(; atol = 1)` and `truncerror(; atol = 1.5)` on the same operand
/// (checked against TensorKit 0.17.1).
fn truncation_follows_the_weighted_rule(case: &Case, kept: [&[(u16, usize)]; 3]) {
    let t = case.filled("t", 2, 2);
    let norm = t.norm(2.0).unwrap();
    let spectrum = t.svd_vals(&[0, 1], &[2, 3]).unwrap();
    let Svd { u, s, vh } = t.svd_compact(&[0, 1], &[2, 3]).unwrap();
    for ((what, truncation, by_hand), tensorkit) in [
        ("rank 7", Truncation::rank(7), ByHand::Rank(7.0)),
        (
            "cutoff 1",
            Truncation::absolute_cutoff(1.0).unwrap(),
            ByHand::Cutoff(1.0),
        ),
        // symtn's budget is `(rtol · norm)²`, `norm` the weighted norm of the
        // spectrum (= ‖t‖), so `rtol = 1.5 / ‖t‖` is the absolute budget 1.5.
        (
            "error 1.5",
            Truncation::relative_error(1.5 / norm).unwrap(),
            ByHand::Error(1.5),
        ),
    ]
    .into_iter()
    .zip(kept)
    {
        let (want_dims, want_error) = truncate_by_hand(case, &spectrum, &by_hand);
        assert_eq!(
            want_dims, tensorkit,
            "{what}: the rule by hand vs TensorKit"
        );
        let found = s.domain()[0]
            .find_truncated(&s.diagview().unwrap(), &truncation)
            .unwrap();
        let u = u
            .restrict_leg(&[(u.codomain_rank(), &found.selection)])
            .unwrap();
        let s = s
            .restrict_leg(&[(0, &found.selection), (1, &found.selection)])
            .unwrap();
        let vh = vh.restrict_leg(&[(0, &found.selection)]).unwrap();
        assert_eq!(degeneracies(&s.domain()[0]), want_dims, "{what}: kept");
        assert!(
            close(found.error, want_error, norm),
            "{what}: error {} != {want_error}",
            found.error
        );
        // The truncated map is orthogonal to what was discarded.
        let truncated = u.compose(&s).unwrap().compose(&vh).unwrap();
        let want_norm = (norm * norm - want_error * want_error).sqrt();
        assert!(
            close(truncated.norm(2.0).unwrap(), want_norm, norm),
            "{what}: norm"
        );
        assert!(
            close(distance(&t, &truncated), want_error, norm),
            "{what}: distance"
        );
    }

    // A cutoff exactly at a singular value keeps that value.
    let at = spectrum[0].values[1];
    let (want_dims, _) = truncate_by_hand(case, &spectrum, &ByHand::Cutoff(at));
    let found = s.domain()[0]
        .find_truncated(
            &s.diagview().unwrap(),
            &Truncation::absolute_cutoff(at).unwrap(),
        )
        .unwrap();
    let s = s
        .restrict_leg(&[(0, &found.selection), (1, &found.selection)])
        .unwrap();
    assert_eq!(degeneracies(&s.domain()[0]), want_dims, "cutoff at σ");
}

#[test]
fn fibonacci_truncation_follows_the_weighted_rule() {
    truncation_follows_the_weighted_rule(
        &Case::fibonacci(),
        [&[(1, 2), (2, 3)], &[(1, 2), (2, 4)], &[(1, 2), (2, 5)]],
    );
}

#[test]
fn ising_truncation_follows_the_weighted_rule() {
    truncation_follows_the_weighted_rule(
        &Case::ising(),
        [
            &[(1, 2), (2, 1), (3, 2)],
            &[(1, 3), (2, 2), (3, 3)],
            &[(1, 4), (2, 2), (3, 5)],
        ],
    );
}

/// `@planar C[b d; h f] := A[a b; c d] * B[c f; a h]` against
/// `planarcontract!`'s steps: two crossing-free transposes, then compose.
fn planar_contraction(case: &Case) -> Table {
    let a = case.filled("a", 2, 2);
    let b = case.filled("b", 2, 2);
    let spec = ContractSpec {
        lhs: &[0, 2],
        rhs: &[2, 0],
        codomain: &[0, 1],
        domain: &[3, 2],
    };
    let c = a.planar_contract(&b, &spec).unwrap();
    let steps = a
        .transpose(&[1, 3], &[0, 2])
        .unwrap()
        .compose(&b.transpose(&[2, 0], &[3, 1]).unwrap())
        .unwrap();
    for (got, want) in c
        .dense_data()
        .unwrap()
        .iter()
        .zip(steps.dense_data().unwrap().iter())
    {
        assert_close("planar_contract vs transpose ∘ compose", *got, *want);
    }
    c
}

/// The planar result's `‖C‖` and `tr C` (checked against TensorKit 0.17.1
/// `@planar` on the same operands), and its spectra by the weighted trace
/// identities.
fn planar_contraction_then_factorization(case: &Case, norm: f64, trace: Complex64) {
    let c = planar_contraction(case);
    assert!(close(c.norm(2.0).unwrap(), norm, norm), "planar norm");
    let got = c.tr().unwrap();
    assert!(
        close(got.re, trace.re, 1.0) && close(got.im, trace.im, 1.0),
        "planar tr {got}"
    );
    let svd = c.svd_vals(&[0, 1], &[2, 3]).unwrap();
    let got = case.weighted_sum(&svd, |s| s * s);
    assert!(close(got, norm * norm, norm * norm), "planar svd: Σ d σ²");
    let hermitian = c
        .axpby(Complex64::ONE, c.adjoint_view(), Complex64::ONE)
        .unwrap();
    let eigh = hermitian
        .eigh_vals(&[0, 1], &[2, 3], HermitianTol::DEFAULT)
        .unwrap();
    let got = case.weighted_sum(&eigh, |x| x);
    assert!(
        close(got, 2.0 * trace.re, norm * norm),
        "planar eigh: Σ d λ = 2 Re tr C"
    );
}

#[test]
fn fibonacci_planar_contraction_then_factorization() {
    planar_contraction_then_factorization(
        &Case::fibonacci(),
        4.937180638781755,
        Complex64::new(-0.0014360199429993945, -0.5709659851802477),
    );
}

#[test]
fn ising_planar_contraction_then_factorization() {
    planar_contraction_then_factorization(
        &Case::ising(),
        6.108913657070855,
        Complex64::new(0.00024355996664443742, -3.7216496922109203),
    );
}

/// `H = −Σ_i P^{(1)}_{i,i+1}` on `w^{⊗L}`, built as TensorKit builds it: the
/// two-site vacuum-channel projector, `otimes` with identities, and sums.
fn projector_chain(runtime: &Runtime, w: &GradedSpace<TableRule>, length: usize) -> Table {
    let vacuum = TableSector::new(1);
    let projector: Table = TensorMap::from_subblock_fn(runtime, [w, w], [w, w], |trees, _| {
        Complex64::new((*trees.coupled() == vacuum) as u8 as f64, 0.0)
    })
    .unwrap();
    let identity =
        |n: usize| -> Table { TensorMap::isomorphism(runtime, vec![w; n], vec![w; n]).unwrap() };
    let mut hamiltonian: Option<Table> = None;
    for i in 1..length {
        let mut term = projector.clone();
        if i > 1 {
            term = identity(i - 1).otimes(&term).unwrap();
        }
        if i + 1 < length {
            term = term.otimes(&identity(length - i - 1)).unwrap();
        }
        hamiltonian = Some(match hamiltonian {
            None => term.scale(-Complex64::ONE).unwrap(),
            Some(h) => h.axpby(Complex64::ONE, &term, -Complex64::ONE).unwrap(),
        });
    }
    hamiltonian.unwrap()
}

fn chain_spectra(h: &Table, length: usize) -> Vec<SectorSpectrum<TableSector>> {
    let rows: Vec<usize> = (0..length).collect();
    let cols: Vec<usize> = (length..2 * length).collect();
    h.eigh_vals(&rows, &cols, HermitianTol::DEFAULT).unwrap()
}

/// The open golden chain in the fusion-path basis `x₀ = 𝟙, x₁ = τ, …,
/// x_L = c` (`x_k` the inner line after site `k`): `P_{i,i+1}` acts on `x_i`
/// through `F^{x_{i−1} τ τ}_{x_{i+1}}[x_i, 𝟙]`, which is `1` for
/// `x_{i−1} = x_{i+1} = 𝟙`, the first column `(φ^{−1}, φ^{−1/2})` of
/// `F^{τττ}_τ` for `x_{i−1} = x_{i+1} = τ`, and zero otherwise.
fn golden_chain_by_hand(runtime: &Runtime, w: &GradedSpace<TableRule>, length: usize) -> Table {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let (unit, tau) = (TableSector::new(1), TableSector::new(2));
    let f_column = |x: TableSector| {
        if x == unit {
            1.0 / phi
        } else {
            1.0 / phi.sqrt()
        }
    };
    let path = |inner: &[TableSector], coupled: TableSector| -> Vec<TableSector> {
        let mut path = vec![unit, tau];
        path.extend_from_slice(inner);
        path.push(coupled);
        path
    };
    TensorMap::from_subblock_fn(runtime, vec![w; length], vec![w; length], |trees, _| {
        let x = path(trees.codomain_innerlines(), *trees.coupled());
        let y = path(trees.domain_innerlines(), *trees.coupled());
        let mut value = 0.0;
        for i in 1..length {
            let others_agree = (0..=length).all(|k| k == i || x[k] == y[k]);
            if !others_agree || x[i - 1] != x[i + 1] {
                continue;
            }
            value -= if x[i - 1] == unit {
                1.0
            } else {
                f_column(x[i]) * f_column(y[i])
            };
        }
        Complex64::new(value, 0.0)
    })
    .unwrap()
}

/// Eigenvalues by AnyonWiki sector label.
type Spectrum = &'static [(u16, &'static [f64])];

/// The golden chain against the fusion-path Hamiltonian for `L = 4 … 8`, and
/// for `L = 4, 5` against eigenvalues checked against TensorKit 0.17.1
/// (`eigh_vals` of the same `−Σ P^{(𝟙)}`).
#[test]
fn golden_chain_spectra_match_the_fusion_path_hamiltonian() {
    let tensorkit: [(usize, Spectrum); 2] = [
        (
            4,
            &[
                (1, &[-2.506941916150187, -0.49305808384981287]),
                (
                    2,
                    &[
                        -1.874032048897642,
                        -0.9999999999999996,
                        -0.12596795110235737,
                    ],
                ),
            ],
        ),
        (
            5,
            &[
                (
                    1,
                    &[-1.9999999999999996, -1.381966011250105, -0.6180339887498942],
                ),
                (
                    2,
                    &[
                        -3.1761168197374268,
                        -2.294962899291598,
                        -1.642922825273373,
                        -0.705037100708401,
                        -0.18096035498919852,
                    ],
                ),
            ],
        ),
    ];
    let runtime = Runtime::builder().build().unwrap();
    let tau =
        GradedSpace::try_new(Arc::new(TableRule::fibonacci()), [(TableSector::new(2), 1)]).unwrap();
    for length in 4..=8 {
        let built = chain_spectra(&projector_chain(&runtime, &tau, length), length);
        let by_hand = chain_spectra(&golden_chain_by_hand(&runtime, &tau, length), length);
        assert_eq!(built.len(), by_hand.len());
        for (b, h) in built.iter().zip(&by_hand) {
            assert_eq!(b.sector, h.sector);
            assert_eq!(b.values.len(), h.values.len(), "L = {length}");
            assert!(
                b.values
                    .iter()
                    .zip(&h.values)
                    .all(|(x, y)| close(*x, *y, 8.0)),
                "L = {length}"
            );
        }
        if let Some((_, want)) = tensorkit.iter().find(|(l, _)| *l == length) {
            let got: Vec<(u16, &[f64])> = built
                .iter()
                .map(|entry| (entry.sector.label(), entry.values.as_slice()))
                .collect();
            assert_eq!(got.len(), want.len(), "L = {length}");
            for ((label, values), (want_label, want)) in got.iter().zip(want.iter()) {
                assert_eq!(label, want_label, "L = {length}");
                assert_eq!(values.len(), want.len(), "L = {length}");
                assert!(
                    values.iter().zip(*want).all(|(x, y)| close(*x, *y, 8.0)),
                    "L = {length} sector {label}: {values:?} != {want:?}"
                );
            }
        }
    }
}

/// `−P^{(𝟙)}` on two σ's is `−(1 + iγ_iγ_{i+1})/2`, so the open σ-chain is
/// the free Majorana chain: `E = −(L−1)/2 − Σ_k cos(πk/(L+1)) s_k`,
/// `s_k = ±1` for `k = 1 … ⌊L/2⌋`. Even `L` splits by fermion parity
/// relative to the ground state (`s ≡ +1`): an even number of `s_k = −1` is
/// the vacuum sector, an odd number ψ; odd `L` is all σ.
fn free_fermion_spectrum(length: usize) -> Vec<(u16, Vec<f64>)> {
    let modes = length / 2;
    let even = length.is_multiple_of(2);
    let mut sectors: Vec<(u16, Vec<f64>)> = if even {
        vec![(1, vec![]), (2, vec![])]
    } else {
        vec![(3, vec![])]
    };
    for signs in 0..(1u32 << modes) {
        let mut energy = -((length - 1) as f64) / 2.0;
        for k in 1..=modes {
            let s = if signs >> (k - 1) & 1 == 1 { -1.0 } else { 1.0 };
            energy -= s * (std::f64::consts::PI * k as f64 / (length + 1) as f64).cos();
        }
        let sector = if even {
            (signs.count_ones() % 2) as usize
        } else {
            0
        };
        sectors[sector].1.push(energy);
    }
    for (_, values) in &mut sectors {
        values.sort_by(f64::total_cmp);
    }
    sectors
}

#[test]
fn ising_sigma_chain_matches_free_fermions() {
    let runtime = Runtime::builder().build().unwrap();
    let sigma =
        GradedSpace::try_new(Arc::new(TableRule::ising()), [(TableSector::new(3), 1)]).unwrap();
    for length in 4..=8 {
        let quantity = format!("chain{length}");
        let built = chain_spectra(&projector_chain(&runtime, &sigma, length), length);
        let closed_form = free_fermion_spectrum(length);
        let got: Vec<(u16, &[f64])> = built
            .iter()
            .map(|entry| (entry.sector.label(), entry.values.as_slice()))
            .collect();
        assert_eq!(got.len(), closed_form.len(), "{quantity}");
        for ((label, values), (want_label, want)) in got.iter().zip(&closed_form) {
            assert_eq!(label, want_label, "{quantity}");
            assert_eq!(values.len(), want.len(), "{quantity}");
            assert!(
                values.iter().zip(want).all(|(x, y)| close(*x, *y, 8.0)),
                "{quantity} sector {label}"
            );
        }
    }
}

#[test]
fn a_real_complex_embedding_factorizes_exactly_as_the_real_symbol_rule() {
    // One path: the complex-symbol rule runs the real-symbol rule's
    // blockwise factorizations and dimension weights bit for bit.
    let runtime = Runtime::builder().build().unwrap();
    let (_, complex) = operands!(&runtime, Embedded::<_, false>(SU2FusionRule), su2_sectors());
    let (_, real) = operands!(&runtime, SU2FusionRule, su2_sectors());
    let bits = |spectra: Vec<SectorSpectrum<SU2Irrep>>| -> Vec<(SU2Irrep, Vec<u64>)> {
        spectra
            .into_iter()
            .map(|entry| {
                (
                    entry.sector,
                    entry.values.iter().map(|v| v.to_bits()).collect(),
                )
            })
            .collect()
    };
    assert_eq!(
        bits(complex.svd_vals(&[0, 1], &[2, 3]).unwrap()),
        bits(real.svd_vals(&[0, 1], &[2, 3]).unwrap())
    );
    let (cs, rs) = (
        complex.svd_compact(&[0, 1], &[2, 3]).unwrap(),
        real.svd_compact(&[0, 1], &[2, 3]).unwrap(),
    );
    assert_eq!(cs.u.dense_data().unwrap(), rs.u.dense_data().unwrap());
    assert_eq!(cs.vh.dense_data().unwrap(), rs.vh.dense_data().unwrap());
    let truncation = Truncation::rank(5);
    let cf = cs.s.domain()[0]
        .find_truncated(&cs.s.diagview().unwrap(), &truncation)
        .unwrap();
    let rf = rs.s.domain()[0]
        .find_truncated(&rs.s.diagview().unwrap(), &truncation)
        .unwrap();
    assert_eq!(cf.error.to_bits(), rf.error.to_bits());
    assert_eq!(
        cs.u.restrict_leg(&[(2, &cf.selection)])
            .unwrap()
            .dense_data()
            .unwrap(),
        rs.u.restrict_leg(&[(2, &rf.selection)])
            .unwrap()
            .dense_data()
            .unwrap()
    );
    let (cq, rq) = (
        complex.qr_compact(&[0, 1], &[2, 3]).unwrap(),
        real.qr_compact(&[0, 1], &[2, 3]).unwrap(),
    );
    assert_eq!(cq.r.dense_data().unwrap(), rq.r.dense_data().unwrap());
    macro_rules! hermitian_spectrum {
        ($t:expr) => {
            bits(
                $t.axpby(Complex64::ONE, $t.adjoint_view(), Complex64::ONE)
                    .unwrap()
                    .eigh_vals(&[0, 1], &[2, 3], HermitianTol::DEFAULT)
                    .unwrap(),
            )
        };
    }
    assert_eq!(hermitian_spectrum!(complex), hermitian_spectrum!(real));
}

#[test]
fn a_non_real_quantum_dimension_is_refused_by_truncation() {
    // SU(2) is `Simple`: a rank budget reads `dim(c)`. The SVD itself reads
    // no dimension and succeeds.
    let runtime = Runtime::builder().build().unwrap();
    let (_, t) = operands!(&runtime, Embedded::<_, true>(SU2FusionRule), su2_sectors());
    let Svd { s, .. } = t.svd_compact(&[0, 1], &[2, 3]).unwrap();
    let error = s.domain()[0]
        .find_truncated(&s.diagview().unwrap(), &Truncation::rank(5))
        .unwrap_err();
    assert!(is_non_real_dimension(&error), "{error:?}");
}

#[test]
fn a_prepared_batched_eigh_refuses_table_rule_payloads_with_a_typed_error() {
    // TableRule tensors are `Complex64`, and the prepared Host eigh supports
    // real payloads only (tenferro-rs#1923): the refusal is the same typed
    // capability error as for complex data on a real-symbol rule.
    let case = Case::ising();
    let t = case.filled("t", 1, 1);
    let h = t
        .axpby(Complex64::ONE, t.adjoint_view(), Complex64::ONE)
        .unwrap();
    let stack = StackedTensorMap::pack(&[h.clone(), h]).unwrap();
    let Err(error) = EighFullPlan::new(&stack, &[0], &[1], HermitianTol::DEFAULT) else {
        panic!("a Complex64 prepared eigh must be refused");
    };
    assert!(
        matches!(
            &error,
            symtn::typed::Error::Operation(operation)
                if matches!(**operation, OperationError::UnsupportedTensorContractScope { .. })
        ),
        "{error:?}"
    );
}
