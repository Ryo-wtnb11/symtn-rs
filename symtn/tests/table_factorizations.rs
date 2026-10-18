//! Factorizations and truncation of complex-symbol rules (#2281): SVD, QR/LQ,
//! polar, eigh/eig, null spaces, matrix functions and the
//! `svd_compact` → `find_truncated` → `restrict_leg` chain on Fibonacci and
//! Ising through `TableRule`, against TensorKit 0.17.1
//! (`fixtures/tensorkit_table_linalg.jl`), plus a planar contraction, the
//! golden chain and the Ising σ-chain.
//!
//! Entries are the tree-hash function the fixture also evaluates, so the
//! oracle builds the same tensors without reading anything symtn computed.
//! Blockwise spectra and kept dimensions are invariant under AnyonWiki's
//! Ising vertex gauge; the planar rows are compared through the recorded
//! witness (the fixture signs its TensorKit entries), with the unsigned
//! TensorKit result as the negative control.
//!
//! Independent of TensorKit: the golden chain is also diagonalised from a
//! Hamiltonian written directly in the fusion-path basis with the closed-form
//! `F^{τττ}_τ` (Feiguin et al., PRL 98, 160409), and the Ising σ-chain is the
//! critical transverse-field Ising chain, whose open-chain spectrum is the
//! free-fermion closed form.

use std::sync::Arc;

use num_complex::Complex64;
use symtn::sector::{SU2FusionRule, SU2Irrep};
use symtn::typed::{
    EighFullPlan, GradedSpace, HermitianTol, LeftPolar, Lq, OperationError, Qr, RightPolar,
    Runtime, SectorSpectrum, StackedTensorMap, Svd, TensorMap, Truncation,
};
use symtn_sectors::{TableRule, TableSector};

#[macro_use]
#[path = "support/table_fixture.rs"]
mod table_fixture;
use table_fixture::*;

type Table = TensorMap<TableRule, Complex64>;

const TENSORKIT: &str = include_str!("fixtures/tensorkit-0.17.1-table-linalg.txt");

/// The fixture's values of `quantity` in `sector` (`-` for a scalar row).
fn tensorkit(category: &str, quantity: &str, sector: &str) -> Vec<f64> {
    TENSORKIT
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| line.split_whitespace().collect::<Vec<_>>())
        .find(|fields| fields[0] == category && fields[1] == quantity && fields[2] == sector)
        .unwrap_or_else(|| panic!("fixture is missing {category} {quantity} {sector}"))[3..]
        .iter()
        .map(|value| value.parse().unwrap())
        .collect()
}

/// The fixture's sectors of `quantity`, by AnyonWiki label.
fn tensorkit_sectors(category: &str, quantity: &str) -> Vec<u16> {
    TENSORKIT
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| line.split_whitespace().collect::<Vec<_>>())
        .filter(|fields| fields[0] == category && fields[1] == quantity)
        .map(|fields| fields[2].parse().unwrap())
        .collect()
}

/// Spectral values are backward stable to a few dozen ulps of the largest
/// one; `64 · 64 · ε` leaves room for a few dozen accumulated terms.
fn close(got: f64, want: f64, scale: f64) -> bool {
    (got - want).abs() <= 4096.0 * f64::EPSILON * scale.max(1.0)
}

fn assert_spectra<V: Copy>(
    what: &str,
    category: &str,
    quantity: &str,
    got: &[SectorSpectrum<TableSector, V>],
    flatten: impl Fn(V) -> Vec<f64>,
) {
    let labels: Vec<u16> = got.iter().map(|entry| entry.sector.label()).collect();
    assert_eq!(
        labels,
        tensorkit_sectors(category, quantity),
        "{what}: sectors"
    );
    for entry in got {
        let got: Vec<f64> = entry.values.iter().flat_map(|&v| flatten(v)).collect();
        let want = tensorkit(category, quantity, &entry.sector.label().to_string());
        let scale = want.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert_eq!(got.len(), want.len(), "{what} sector {:?}", entry.sector);
        for (g, w) in got.iter().zip(&want) {
            assert!(
                close(*g, *w, scale),
                "{what} sector {:?}: {got:?} != {want:?}",
                entry.sector
            );
        }
    }
}

fn real_spectra(what: &str, category: &str, quantity: &str, got: &[SectorSpectrum<TableSector>]) {
    assert_spectra(what, category, quantity, got, |v| vec![v]);
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

fn assert_tensorkit_dims(what: &str, category: &str, quantity: &str, leg: &GradedSpace<TableRule>) {
    let want: Vec<(u16, usize)> = tensorkit_sectors(category, quantity)
        .into_iter()
        .map(|label| {
            let n = tensorkit(category, quantity, &label.to_string())[0];
            (label, n as usize)
        })
        .collect();
    assert_eq!(degeneracies(leg), want, "{what}");
}

fn tensorkit_scalar(category: &str, quantity: &str) -> f64 {
    tensorkit(category, quantity, "-")[0]
}

struct Case {
    category: &'static str,
    runtime: Runtime,
    v: GradedSpace<TableRule>,
}

impl Case {
    fn new(category: &'static str, rule: TableRule, sectors: &[(u16, usize)]) -> Self {
        let v = GradedSpace::try_new(
            Arc::new(rule),
            sectors
                .iter()
                .map(|&(label, degeneracy)| (TableSector::new(label), degeneracy)),
        )
        .unwrap();
        Self {
            category,
            runtime: Runtime::builder().build().unwrap(),
            v,
        }
    }

    fn fibonacci() -> Self {
        Self::new("fibonacci", TableRule::fibonacci(), &[(1, 1), (2, 2)])
    }

    fn ising() -> Self {
        Self::new("ising", TableRule::ising(), &[(1, 1), (2, 1), (3, 2)])
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
}

fn spectra_match_tensorkit(case: &Case) {
    let category = case.category;
    let t = case.filled("t", 2, 2);
    real_spectra(
        "svd_vals",
        category,
        "svd",
        &t.svd_vals(&[0, 1], &[2, 3]).unwrap(),
    );
    let hermitian = t
        .axpby(Complex64::ONE, t.adjoint_view(), Complex64::ONE)
        .unwrap();
    real_spectra(
        "eigh_vals",
        category,
        "eigh",
        &hermitian
            .eigh_vals(&[0, 1], &[2, 3], HermitianTol::DEFAULT)
            .unwrap(),
    );
    assert_spectra(
        "eig_vals",
        category,
        "eig",
        &t.eig_vals(&[0, 1], &[2, 3]).unwrap(),
        |z: Complex64| vec![z.re, z.im],
    );

    // A lazy adjoint is factorized through its parent: the same spectrum.
    real_spectra(
        "svd_vals of the adjoint",
        category,
        "svd",
        &t.adjoint().unwrap().svd_vals(&[0, 1], &[2, 3]).unwrap(),
    );
}

#[test]
fn fibonacci_spectra_match_tensorkit() {
    spectra_match_tensorkit(&Case::fibonacci());
}

#[test]
fn ising_spectra_match_tensorkit() {
    spectra_match_tensorkit(&Case::ising());
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
    let eig = t.eig_full(&[0, 1], &[2, 3]).unwrap();
    assert_near(
        "t v = v d",
        &t.compose(&eig.v).unwrap(),
        &eig.v.compose(&eig.d).unwrap(),
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

    // Null spaces: orthonormal, annihilated by the input, of TensorKit's size.
    let category = case.category;
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
    assert_tensorkit_dims("left null", category, "left_null", &n.domain()[0]);
    let n = wide.right_null(&[0], &[1, 2]).unwrap();
    assert_isometry(
        "right null",
        runtime,
        &n.adjoint().unwrap().materialize().unwrap(),
    );
    assert!(wide.compose(n.adjoint_view()).unwrap().norm(2.0).unwrap() < 1.0e-12);
    assert_tensorkit_dims("right null", category, "right_null", &n.codomain()[0]);
}

#[test]
fn fibonacci_decompositions_reconstruct() {
    decompositions_reconstruct(&Case::fibonacci());
}

#[test]
fn ising_decompositions_reconstruct() {
    decompositions_reconstruct(&Case::ising());
}

/// The golden-chain DMRG step: `svd_compact` → `find_truncated` →
/// `restrict_leg`, weighted by `dim(c)`, against TensorKit `svd_trunc`.
fn truncation_matches_tensorkit(case: &Case) {
    let category = case.category;
    let t = case.filled("t", 2, 2);
    let Svd { u, s, vh } = t.svd_compact(&[0, 1], &[2, 3]).unwrap();
    for (tag, truncation) in [
        ("rank7", Truncation::rank(7)),
        ("tol", Truncation::absolute_cutoff(1.0).unwrap()),
        // TensorKit `truncerror(; atol = 1.5)`: symtn's budget is
        // `(rtol · norm)²`, `norm` the weighted norm of the spectrum (= ‖t‖).
        (
            "discard",
            Truncation::relative_error(1.5 / t.norm(2.0).unwrap()).unwrap(),
        ),
    ] {
        let quantity = format!("trunc_{tag}");
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
        assert_tensorkit_dims(&quantity, category, &quantity, &s.domain()[0]);
        let error = tensorkit_scalar(category, &format!("{quantity}_error"));
        assert!(
            close(found.error, error, error),
            "{quantity}: {} != {error}",
            found.error
        );
        let truncated = u.compose(&s).unwrap().compose(&vh).unwrap();
        let norm = tensorkit_scalar(category, &format!("{quantity}_norm"));
        assert!(
            close(truncated.norm(2.0).unwrap(), norm, norm),
            "{quantity} norm"
        );
        // The reported error is the weighted norm of what was discarded.
        assert!(
            close(distance(&t, &truncated), found.error, error),
            "{quantity} distance"
        );
    }
}

#[test]
fn fibonacci_truncation_matches_tensorkit() {
    truncation_matches_tensorkit(&Case::fibonacci());
}

#[test]
fn ising_truncation_matches_tensorkit() {
    truncation_matches_tensorkit(&Case::ising());
}

/// TensorKit `@planar C[b d; h f] := A[a b; c d] * B[c f; a h]` as
/// `planarcontract!` runs it: two crossing-free transposes, then compose.
fn planar_contraction(case: &Case) -> Table {
    let a = case.filled("a", 2, 2);
    let b = case.filled("b", 2, 2);
    a.transpose(&[1, 3], &[0, 2])
        .unwrap()
        .compose(&b.transpose(&[2, 0], &[3, 1]).unwrap())
        .unwrap()
}

fn planar_contraction_then_factorization(case: &Case) {
    let category = case.category;
    let c = planar_contraction(case);
    real_spectra(
        "planar svd",
        category,
        "planar_svd",
        &c.svd_vals(&[0, 1], &[2, 3]).unwrap(),
    );
    let hermitian = c
        .axpby(Complex64::ONE, c.adjoint_view(), Complex64::ONE)
        .unwrap();
    real_spectra(
        "planar eigh",
        category,
        "planar_eigh",
        &hermitian
            .eigh_vals(&[0, 1], &[2, 3], HermitianTol::DEFAULT)
            .unwrap(),
    );
    let tr = tensorkit(category, "planar_tr", "-");
    let got = c.tr().unwrap();
    assert!(
        close(got.re, tr[0], 1.0) && close(got.im, tr[1], 1.0),
        "planar tr {got}"
    );
    let norm = tensorkit_scalar(category, "planar_norm");
    assert!(close(c.norm(2.0).unwrap(), norm, norm), "planar norm");
}

#[test]
fn fibonacci_planar_contraction_then_factorization_matches_tensorkit() {
    planar_contraction_then_factorization(&Case::fibonacci());
}

#[test]
fn ising_planar_contraction_then_factorization_matches_tensorkit() {
    planar_contraction_then_factorization(&Case::ising());
}

#[test]
fn ising_planar_comparison_detects_the_vertex_gauge() {
    // Negative control: the TensorKit contraction of the same coordinates in
    // TensorKit's own gauge is a different morphism, and its σ spectrum
    // differs from symtn's; the ψ-free unit and ψ blocks agree either way.
    let case = Case::ising();
    let c = planar_contraction(&case);
    let spectra = c.svd_vals(&[0, 1], &[2, 3]).unwrap();
    let sigma = spectra
        .iter()
        .find(|entry| entry.sector == TableSector::new(3))
        .unwrap();
    let ungauged = tensorkit("ising", "planar_ungauged_svd", "3");
    assert!(
        sigma
            .values
            .iter()
            .zip(&ungauged)
            .any(|(g, w)| !close(*g, *w, 4.0)),
        "the ungauged comparison must fail"
    );
    let unit = tensorkit("ising", "planar_ungauged_svd", "1");
    assert!(spectra[0]
        .values
        .iter()
        .zip(&unit)
        .all(|(g, w)| close(*g, *w, 4.0)));
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

#[test]
fn golden_chain_spectra_match_tensorkit_and_the_fusion_path_hamiltonian() {
    let runtime = Runtime::builder().build().unwrap();
    let tau =
        GradedSpace::try_new(Arc::new(TableRule::fibonacci()), [(TableSector::new(2), 1)]).unwrap();
    for length in 4..=8 {
        let quantity = format!("chain{length}");
        let built = chain_spectra(&projector_chain(&runtime, &tau, length), length);
        real_spectra(&quantity, "fibonacci", &quantity, &built);
        let by_hand = chain_spectra(&golden_chain_by_hand(&runtime, &tau, length), length);
        assert_eq!(built.len(), by_hand.len());
        for (b, h) in built.iter().zip(&by_hand) {
            assert_eq!(b.sector, h.sector);
            assert!(
                b.values
                    .iter()
                    .zip(&h.values)
                    .all(|(x, y)| close(*x, *y, 8.0)),
                "{quantity}"
            );
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
fn ising_sigma_chain_matches_free_fermions_and_tensorkit() {
    let runtime = Runtime::builder().build().unwrap();
    let sigma =
        GradedSpace::try_new(Arc::new(TableRule::ising()), [(TableSector::new(3), 1)]).unwrap();
    for length in 4..=8 {
        let quantity = format!("chain{length}");
        let built = chain_spectra(&projector_chain(&runtime, &sigma, length), length);
        real_spectra(&quantity, "ising", &quantity, &built);
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
