//! iTEBD ground-state search for the spin-1/2 antiferromagnetic Heisenberg
//! chain, U(1)-symmetric (Sz conservation), through the typed `TensorMap` API
//! and the explicit `Network` API.
//!
//! Charge convention: the physical leg has U(1) sectors `[(+1, 1), (-1, 1)]`
//! with charge `q = 2 Sz` (so `+1` = up, `-1` = down). Total charge is
//! conserved, which makes the two-site Hamiltonian and gate block-diagonal
//! in `q1 + q2 ∈ {-2, 0, +2}`: the `±2` blocks are the 1x1 `Sz Sz = +1/4`
//! diagonals and the `0` block is the 2x2 `[[-1/4, 1/2], [1/2, -1/4]]`
//! flip-flop matrix.
//!
//! Vidal form with a two-site unit cell A-B: state = `... λb Γa λa Γb λb ...`,
//! Γ tensors shaped `[left_bond, phys] <- [right_bond]`, λ diagonal bond
//! endomorphisms. One bond update contracts
//! `θ = λ_out Γ1 λ_mid Γ2 λ_out · gate`, truncates it (`svd_compact`, then
//! `find_truncated` on the spectrum and `restrict_leg` on the factors),
//! and
//! restores Vidal form by multiplying the outer `λ_out^{-1}` back in
//! (diagonal inverse via `TensorMap::pinv`).
//!
//! Reference: ground-state energy per bond `e = 1/4 - ln 2 ≈ -0.4431471`.
//!
//! Run with:
//! `cargo run --release -p symtn-network --example itebd_heisenberg --no-default-features --features cpu-faer`

use std::time::Instant;

use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::Svd;
use symtn::typed::{Error, GradedSpace, Runtime, TensorMap, Truncation};
use symtn_network::Network;

const E_EXACT: f64 = 0.25 - std::f64::consts::LN_2;
/// Relative cutoff on kept singular values: values this small get inverted
/// by the Vidal `λ^{-1}` step, so don't keep anything numerically dangerous.
const BOND_RTOL: f64 = 1e-8;
/// `pinv` cutoff (relative to the largest singular value of λ).
const PINV_RCOND: f64 = 1e-12;
type Map = TensorMap<U1FusionRule, f64>;

fn space<const N: usize>(sectors: [(i32, usize); N]) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        std::sync::Arc::new(U1FusionRule),
        sectors.map(|(charge, degeneracy)| (U1Irrep::new(charge), degeneracy)),
    )
    .expect("valid U(1) fixture")
}

/// Two-site Heisenberg Hamiltonian `h = S·S` on `[p, p] <- [p, p]`, built
/// block-by-block in the U(1) fusion-tree basis. Blocks are keyed by
/// (codomain uncoupled charges, domain uncoupled charges) with total charge
/// conserved, so only same-charge tree pairs appear:
/// `<s1 s2| h |s1 s2> = ±1/4` (Sz Sz), `<up dn| h |dn up> = 1/2` (flip-flop).
fn heisenberg_two_site(rt: &Runtime, p: &GradedSpace<U1FusionRule>) -> Result<Map, Error> {
    TensorMap::from_subblock_fn(rt, [p, p], [p, p], |trees, _| {
        let cod = trees.codomain_uncoupled();
        let dom = trees.domain_uncoupled();
        if cod == dom {
            if cod[0] == cod[1] {
                0.25 // up-up / dn-dn
            } else {
                -0.25 // up-dn / dn-up diagonal
            }
        } else {
            0.5 // flip-flop, only present in the charge-0 block
        }
    })
}

/// One iTEBD bond update in Vidal form: applies `gate` to the bond between
/// `g1` and `g2` (inner weight `l_mid`, outer weights `l_out` on both open
/// bonds), truncates, and returns `(g1', l_mid', g2', truncation_error)`.
fn bond_update(
    gate: &Map,
    l_out: &Map,
    g1: &Map,
    l_mid: &Map,
    g2: &Map,
    trunc: &Truncation,
) -> Result<(Map, Map, Map, f64), Error> {
    let theta = Network::new(
        vec![
            vec!["l", "x"],
            vec!["x", "qa", "y"],
            vec!["y", "z"],
            vec!["z", "qb", "w"],
            vec!["w", "r"],
            vec!["pa", "pb", "qa", "qb"],
        ],
        vec![false, false, false, false, false, false],
        vec![1, 2, 1, 2, 1, 2],
        vec!["l", "pa", "pb", "r"],
        Some(2),
    )
    .expect("fixed network labels are valid")
    .contract(&[l_out, g1, l_mid, g2, l_out, gate])?;
    // Truncated SVD: factorize, decide the kept bond from the spectrum, then
    // restrict every factor to it.
    let Svd { u, s, vh } = theta.svd_compact(&[0, 1], &[2, 3])?;
    let found = s.domain()[0].find_truncated(&s.diagview()?, trunc)?;
    let u = u.restrict_leg(&[(u.codomain_rank(), &found.selection)])?;
    let s = s.restrict_leg(&[(0, &found.selection), (1, &found.selection)])?;
    let vh = vh.restrict_leg(&[(0, &found.selection)])?;
    let l_new = s.scale(1.0 / s.norm(2.0)?)?;
    // Divide the outer λ back out: diagonal inverse via pinv.
    let l_out_inv = l_out.pinv(&[0], &[1], PINV_RCOND)?;
    let g1_new = Network::new(
        vec![vec!["l", "x"], vec!["x", "pa", "m"]],
        vec![false, false],
        vec![1, 2],
        vec!["l", "pa", "m"],
        Some(2),
    )
    .expect("fixed network labels are valid")
    .contract(&[&l_out_inv, &u])?;
    let g2_new = Network::new(
        vec![vec!["m", "pb", "x"], vec!["x", "r"]],
        vec![false, false],
        vec![1, 1],
        vec!["m", "pb", "r"],
        Some(2),
    )
    .expect("fixed network labels are valid")
    .contract(&[&vh, &l_out_inv])?;
    Ok((g1_new, l_new, g2_new, found.error))
}

/// Energy of one bond, `<θ|h|θ> / <θ|θ>`, on the two-site wavefunction
/// `θ = λ_out Γ1 λ_mid Γ2 λ_out` (no gate).
fn bond_energy(h: &Map, l_out: &Map, g1: &Map, l_mid: &Map, g2: &Map) -> Result<f64, Error> {
    let theta = Network::new(
        vec![
            vec!["l", "x"],
            vec!["x", "pa", "y"],
            vec!["y", "z"],
            vec!["z", "pb", "w"],
            vec!["w", "r"],
        ],
        vec![false, false, false, false, false],
        vec![1, 2, 1, 2, 1],
        vec!["l", "pa", "pb", "r"],
        Some(3),
    )
    .expect("fixed network labels are valid")
    .contract(&[l_out, g1, l_mid, g2, l_out])?;
    let num = Network::new(
        vec![
            vec!["l", "pa", "pb", "r"],
            vec!["pa", "pb", "qa", "qb"],
            vec!["l", "qa", "qb", "r"],
        ],
        vec![true, false, false],
        vec![3, 2, 3],
        vec![],
        Some(0),
    )
    .expect("fixed network labels are valid")
    .contract(&[&theta, h, &theta])?
    .scalar()?;
    Ok(num / theta.inner(&theta)?)
}

/// Weighted bond dimension of a diagonal λ (its single codomain leg).
fn bond_dim(lambda: &Map) -> Result<usize, Error> {
    Ok(lambda.leg_dims()?[0])
}

struct State {
    ga: Map,
    la: Map, // bond A-B
    gb: Map,
    lb: Map, // bond B-A
}

impl State {
    /// Charge-balanced entangled start, `∝ ... (|up dn> + |dn up>) ...`
    /// per unit cell: bond B-A carries charge 0, bond A-B charges ±1.
    ///
    /// (A strict Neel product state — phys leg populated only in `+1` on A /
    /// `-1` on B — also works, because legs carry per-sector degeneracies;
    /// see the `neel_product_state_contracts_with_the_full_gate` test. The
    /// entangled start is kept because it converges faster.)
    fn init(rt: &Runtime, p: &GradedSpace<U1FusionRule>) -> Result<Self, Error> {
        let vb = space([(0, 1)]);
        let va = space([(1, 1), (-1, 1)]);
        Ok(Self {
            ga: TensorMap::from_subblock_fn(rt, [&vb, p], [&va], |_, _| 1.0)?,
            la: TensorMap::from_subblock_fn(rt, [&va], [&va], |_, _| 1.0)?,
            gb: TensorMap::from_subblock_fn(rt, [&va, p], [&vb], |_, _| 1.0)?,
            lb: TensorMap::from_subblock_fn(rt, [&vb], [&vb], |_, _| 1.0)?,
        })
    }

    /// One full iTEBD step: update bond A-B, then bond B-A.
    fn step(&mut self, gate: &Map, trunc: &Truncation) -> Result<f64, Error> {
        let (ga, la, gb, err_a) = bond_update(gate, &self.lb, &self.ga, &self.la, &self.gb, trunc)?;
        self.ga = ga;
        self.la = la;
        self.gb = gb;
        let (gb, lb, ga, err_b) = bond_update(gate, &self.la, &self.gb, &self.lb, &self.ga, trunc)?;
        self.gb = gb;
        self.lb = lb;
        self.ga = ga;
        Ok(err_a.max(err_b))
    }

    /// Mean energy per bond, averaged over the A-B and B-A bonds.
    fn energy(&self, h: &Map) -> Result<f64, Error> {
        let e_ab = bond_energy(h, &self.lb, &self.ga, &self.la, &self.gb)?;
        let e_ba = bond_energy(h, &self.la, &self.gb, &self.lb, &self.ga)?;
        Ok(0.5 * (e_ab + e_ba))
    }
}

/// Runs the full schedule; returns the final energy per bond.
pub fn run(chi: usize, schedule: &[(f64, usize)], verbose: bool) -> Result<f64, Error> {
    let rt = Runtime::builder().build()?;
    let p = space([(1, 1), (-1, 1)]);
    let h = heisenberg_two_site(&rt, &p)?;
    let trunc = Truncation::rank(chi).and(Truncation::relative_cutoff(BOND_RTOL)?);

    let mut state = State::init(&rt, &p)?;
    let mut energy = f64::NAN;
    for &(dt, steps) in schedule {
        let gate = h.scale(-dt)?.exp(&[0, 1], &[2, 3])?;
        let start = Instant::now();
        let mut max_err = 0.0f64;
        for _ in 0..steps {
            max_err = max_err.max(state.step(&gate, &trunc)?);
        }
        let elapsed = start.elapsed().as_secs_f64();
        energy = state.energy(&h)?;
        if verbose {
            println!(
                "dt = {dt:<7} steps = {steps:<5} chi = {:<3} E/bond = {energy:.8} \
                 (err vs exact {:+.2e}, max trunc err {max_err:.1e}, \
                 {:.1} steps/s, {elapsed:.2} s)",
                bond_dim(&state.la)?.max(bond_dim(&state.lb)?),
                energy - E_EXACT,
                steps as f64 / elapsed,
            );
        }
    }
    Ok(energy)
}

fn main() -> Result<(), Error> {
    let chi = 32;
    let schedule = [
        (0.1, 500),
        (0.05, 500),
        (0.01, 500),
        (0.005, 500),
        (0.001, 500),
    ];
    let start = Instant::now();
    let energy = run(chi, &schedule, true)?;
    let total = start.elapsed().as_secs_f64();
    println!();
    println!("final E/bond   = {energy:.8}");
    println!("exact  1/4-ln2 = {E_EXACT:.8}");
    println!("error          = {:+.3e}", energy - E_EXACT);
    println!("total wall time: {total:.1} s");
    assert!(
        (energy - E_EXACT).abs() < 5e-3,
        "iTEBD did not reach the exact ground-state energy: {energy} vs {E_EXACT}"
    );
    Ok(())
}
