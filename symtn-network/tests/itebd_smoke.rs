//! Physics smoke test: iTEBD ground-state search for the spin-1/2 AFM
//! Heisenberg chain with U(1) (Sz) symmetry, user-layer API only.
//! Charges label twice Sz: -1 and +1. Exact energy per bond: `1/4 - ln 2`.

use std::sync::Arc;

use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::Truncation;
use symtn::typed::{GradedSpace, Runtime, Svd, TensorMap};

use crate::network_support;
use network_support::{conj, net, op};

const E_EXACT: f64 = 0.25 - std::f64::consts::LN_2;
type Map = TensorMap<U1FusionRule, f64>;

fn space<const N: usize>(
    rule: &Arc<U1FusionRule>,
    sectors: [(i32, usize); N],
) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::clone(rule),
        sectors.map(|(charge, degeneracy)| (U1Irrep::new(charge), degeneracy)),
    )
    .unwrap()
}

fn heisenberg_two_site(rt: &Runtime, p: &GradedSpace<U1FusionRule>) -> Map {
    TensorMap::from_subblock_fn(rt, [p, p], [p, p], |trees, _| {
        let cod = trees.codomain_uncoupled();
        let dom = trees.domain_uncoupled();
        if cod == dom {
            if cod[0] == cod[1] {
                0.25
            } else {
                -0.25
            }
        } else {
            0.5
        }
    })
    .unwrap()
}

/// One Vidal bond update; returns `(g1', l_mid', g2')`.
fn bond_update(
    gate: &Map,
    l_out: &Map,
    g1: &Map,
    l_mid: &Map,
    g2: &Map,
    trunc: &Truncation,
) -> (Map, Map, Map) {
    let theta = net(
        &[
            op(&["l"], &["x"]),
            op(&["x", "qa"], &["y"]),
            op(&["y"], &["z"]),
            op(&["z", "qb"], &["w"]),
            op(&["w"], &["r"]),
            op(&["pa", "pb"], &["qa", "qb"]),
        ],
        &["l", "pa"],
        &["pb", "r"],
    )
    .contract(&[l_out, g1, l_mid, g2, l_out, gate])
    .unwrap();
    let Svd { u, s, vh } = theta.svd_compact(&[0, 1], &[2, 3]).unwrap();
    let found = s.domain()[0]
        .find_truncated(&s.diagview().unwrap(), trunc)
        .unwrap();
    let u = u
        .restrict_leg(&[(u.codomain_rank(), &found.selection)])
        .unwrap();
    let s = s
        .restrict_leg(&[(0, &found.selection), (1, &found.selection)])
        .unwrap();
    let vh = vh.restrict_leg(&[(0, &found.selection)]).unwrap();
    let l_new = s.scale(1.0 / s.norm(2.0).unwrap()).unwrap();
    let l_out_inv = l_out.pinv(&[0], &[1], 1e-12).unwrap();
    let g1_new = net(
        &[op(&["l"], &["x"]), op(&["x", "pa"], &["m"])],
        &["l", "pa"],
        &["m"],
    )
    .contract(&[&l_out_inv, &u])
    .unwrap();
    let g2_new = net(
        &[op(&["m"], &["pb", "x"]), op(&["x"], &["r"])],
        &["m", "pb"],
        &["r"],
    )
    .contract(&[&vh, &l_out_inv])
    .unwrap();
    (g1_new, l_new, g2_new)
}

fn bond_energy(h: &Map, l_out: &Map, g1: &Map, l_mid: &Map, g2: &Map) -> f64 {
    let theta = net(
        &[
            op(&["l"], &["x"]),
            op(&["x", "pa"], &["y"]),
            op(&["y"], &["z"]),
            op(&["z", "pb"], &["w"]),
            op(&["w"], &["r"]),
        ],
        &["l", "pa", "pb"],
        &["r"],
    )
    .contract(&[l_out, g1, l_mid, g2, l_out])
    .unwrap();
    let num = net(
        &[
            conj(op(&["l", "pa", "pb"], &["r"])),
            op(&["pa", "pb"], &["qa", "qb"]),
            op(&["l", "qa", "qb"], &["r"]),
        ],
        &[],
        &[],
    )
    .contract(&[&theta, h, &theta])
    .unwrap()
    .scalar()
    .unwrap();
    num / theta.inner(&theta).unwrap()
}

/// Runs the schedule from a Neel state; returns the energy per bond.
fn run_itebd(chi: usize, schedule: &[(f64, usize)]) -> f64 {
    let rt = Runtime::builder().build().unwrap();
    let rule = Arc::new(U1FusionRule);
    let p = space(&rule, [(1, 1), (-1, 1)]);
    let h = heisenberg_two_site(&rt, &p);
    let trunc = Truncation::rank(chi).and(Truncation::relative_cutoff(1e-8).unwrap());

    // Charge-balanced entangled start (converges faster than a product
    // state; a strict Neel start also works, see the regression test below).
    let vb = space(&rule, [(0, 1)]);
    let va = space(&rule, [(1, 1), (-1, 1)]);
    let mut ga = TensorMap::from_subblock_fn(&rt, [&vb, &p], [&va], |_, _| 1.0).unwrap();
    let mut la = TensorMap::from_subblock_fn(&rt, [&va], [&va], |_, _| 1.0).unwrap();
    let mut gb = TensorMap::from_subblock_fn(&rt, [&va, &p], [&vb], |_, _| 1.0).unwrap();
    let mut lb = TensorMap::from_subblock_fn(&rt, [&vb], [&vb], |_, _| 1.0).unwrap();

    for &(dt, steps) in schedule {
        let gate = h.scale(-dt).unwrap().exp(&[0, 1], &[2, 3]).unwrap();
        for _ in 0..steps {
            (ga, la, gb) = bond_update(&gate, &lb, &ga, &la, &gb, &trunc);
            (gb, lb, ga) = bond_update(&gate, &la, &gb, &lb, &ga, &trunc);
        }
    }
    0.5 * (bond_energy(&h, &lb, &ga, &la, &gb) + bond_energy(&h, &la, &gb, &lb, &ga))
}

/// Regression: a strict Neel product state built on the FULL physical space.
///
/// Each site tensor has a physical leg whose spaces contain both charges but
/// where one charge participates in NO fusion tree on that tensor (the
/// singleton bond legs fix the total charge): site A populates only `up`,
/// site B only `dn`. Deriving leg dimensions and result-block shapes from
/// populated blocks only would reject the contraction with the gate's full
/// physical leg; with graded legs (sector -> degeneracy on the leg itself) it
/// must work.
#[test]
fn neel_product_state_contracts_with_the_full_gate() {
    let rt = Runtime::builder().build().unwrap();
    let rule = Arc::new(U1FusionRule);
    let p = space(&rule, [(1, 1), (-1, 1)]);
    let h = heisenberg_two_site(&rt, &p);

    // |up dn>: bonds {0} -> {+1} -> {0}; a has no tree with phys charge -1,
    // b none with +1.
    let vl = space(&rule, [(0, 1)]);
    let vm = space(&rule, [(1, 1)]);
    let vr = space(&rule, [(0, 1)]);
    let a = TensorMap::from_subblock_fn(&rt, [&vl, &p], [&vm], |_, _| 1.0).unwrap();
    let b = TensorMap::from_subblock_fn(&rt, [&vm, &p], [&vr], |_, _| 1.0).unwrap();

    // The legs report the full graded space, not just populated sectors.
    assert_eq!(a.leg_dims().unwrap(), vec![1, 2, 1]);
    assert_eq!(a.codomain()[1], p);

    let psi = net(
        &[op(&["l", "pa"], &["m"]), op(&["m", "pb"], &["r"])],
        &["l", "pa", "pb"],
        &["r"],
    )
    .contract(&[&a, &b])
    .unwrap();
    assert!((psi.norm(2.0).unwrap() - 1.0).abs() < 1e-12);

    // theta = h |psi>: this contraction must not be rejected with a leg
    // dimension mismatch against the gate's full physical leg.
    let theta = net(
        &[
            op(&["l", "qa"], &["m"]),
            op(&["m", "qb"], &["r"]),
            op(&["pa", "pb"], &["qa", "qb"]),
        ],
        &["l", "pa", "pb"],
        &["r"],
    )
    .contract(&[&a, &b, &h])
    .unwrap();

    // h |up dn> = -1/4 |up dn> + 1/2 |dn up>, so <psi|h|psi> = -1/4 and
    // |h psi|^2 = 1/16 + 1/4 = 5/16.
    let energy = net(
        &[
            conj(op(&["l", "pa", "pb"], &["r"])),
            op(&["pa", "pb"], &["qa", "qb"]),
            op(&["l", "qa", "qb"], &["r"]),
        ],
        &[],
        &[],
    )
    .contract(&[&psi, &h, &psi])
    .unwrap()
    .scalar()
    .unwrap();
    assert!((energy - (-0.25)).abs() < 1e-12, "energy = {energy}");
    let theta_norm = theta.norm(2.0).unwrap();
    assert!(
        (theta_norm - (5.0f64 / 16.0).sqrt()).abs() < 1e-12,
        "|h psi| = {theta_norm}"
    );
}

/// Tiny chi and few steps, loose tolerance.
#[test]
fn ground_state_energy_converges_coarsely() {
    let energy = run_itebd(8, &[(0.1, 100), (0.05, 100)]);
    assert!(
        (energy - E_EXACT).abs() < 5e-2,
        "E/bond = {energy} vs exact {E_EXACT}"
    );
}
