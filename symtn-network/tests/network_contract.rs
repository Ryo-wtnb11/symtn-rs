//! `Network::contract`: eager contraction through the Runtime's plan cache
//! (#2022), checked against an independent pairwise oracle, explicit
//! `plan` + `execute`, and an equal network built anew, which shares its
//! cached plan.

use std::sync::Arc;

use symtn::sector::{
    FermionParityFusionRule, SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep, Z2Irrep,
};
use symtn::typed::{ContractSpec, GradedSpace, Runtime, TensorMap};
use symtn_network::{
    clear_plan_cache, load_plan_cache, plan_cache_stats, save_plan_cache, GreedyDenseOptimizer,
    Network, NetworkExecutionWorkspace,
};

use crate::network_support;
use network_support::{conj, net, op};

/// `[i; m] = a[i; j] * b[j; k] * c[k; m]`.
fn chain() -> Network {
    Network::new(
        vec![vec!["i", "j"], vec!["j", "k"], vec!["k", "m"]],
        vec![false; 3],
        vec![1; 3],
        vec!["i", "m"],
        Some(1),
    )
    .unwrap()
}

/// `[] = conj(psi)[p; l, r] * h[p; q] * psi[q; l, r]`.
fn expectation() -> Network {
    Network::new(
        vec![vec!["p", "l", "r"], vec!["p", "q"], vec!["q", "l", "r"]],
        vec![true, false, false],
        vec![1; 3],
        vec![],
        Some(0),
    )
    .unwrap()
}

fn matrix() -> ContractSpec<'static> {
    ContractSpec {
        lhs: &[1],
        rhs: &[0],
        codomain: &[0],
        domain: &[1],
    }
}

fn close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    let scale = expected.iter().fold(1.0_f64, |m, x| m.max(x.abs()));
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() <= 1e-12 * scale, "{a} vs {e}");
    }
}

macro_rules! paths {
    ($name:ident, $space:expr) => {
        #[test]
        fn $name() {
            let runtime = Runtime::builder().dense_threads(1).build().unwrap();
            let v = $space;
            let a = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v], [&v], 1).unwrap();
            let b = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v], [&v], 2).unwrap();
            let c = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v], [&v], 3).unwrap();

            // The chain.
            let oracle = a
                .contract(&b, &matrix())
                .unwrap()
                .contract(&c, &matrix())
                .unwrap();
            let network = chain();
            let planned = network
                .plan(&[&a, &b, &c], &GreedyDenseOptimizer)
                .unwrap()
                .execute(&[&a, &b, &c], &mut NetworkExecutionWorkspace::default())
                .unwrap();
            let eager = network.contract(&[&a, &b, &c]).unwrap();
            let cached_result = net(
                &[op(&["i"], &["j"]), op(&["j"], &["k"]), op(&["k"], &["m"])],
                &["i"],
                &["m"],
            )
            .contract(&[&a, &b, &c])
            .unwrap();
            assert_eq!(eager.codomain(), oracle.codomain());
            assert_eq!(eager.domain(), oracle.domain());
            close(&eager.dense_data().unwrap(), &oracle.dense_data().unwrap());
            close(
                &planned.dense_data().unwrap(),
                &oracle.dense_data().unwrap(),
            );
            assert_eq!(
                eager.dense_data().unwrap(),
                cached_result.dense_data().unwrap()
            );

            // The expectation value, with a conjugated operand: the oracle
            // contracts `psi.adjoint()` (legs `[l, r; p]`) with `h psi` (legs
            // `[p; l, r]`) by the typed pairwise contraction. Why not
            // `psi.inner(h psi)`: for fermions the closed network is the
            // supertrace, which differs from the inner product by the odd
            // sectors' signs.
            let psi = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v], [&v, &v], 4).unwrap();
            let h = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v], [&v], 5).unwrap();
            let h_psi = h
                .contract(
                    &psi,
                    &ContractSpec {
                        lhs: &[1],
                        rhs: &[0],
                        codomain: &[0],
                        domain: &[1, 2],
                    },
                )
                .unwrap();
            let oracle = psi
                .adjoint()
                .unwrap()
                .contract(
                    &h_psi,
                    &ContractSpec {
                        lhs: &[2, 0, 1],
                        rhs: &[0, 1, 2],
                        codomain: &[],
                        domain: &[],
                    },
                )
                .unwrap()
                .scalar()
                .unwrap();
            let network = expectation();
            let eager = network
                .contract(&[&psi, &h, &psi])
                .unwrap()
                .scalar()
                .unwrap();
            let planned = network
                .plan(&[&psi, &h, &psi], &GreedyDenseOptimizer)
                .unwrap()
                .execute(&[&psi, &h, &psi], &mut NetworkExecutionWorkspace::default())
                .unwrap()
                .scalar()
                .unwrap();
            let cached_result = net(
                &[
                    conj(op(&["p"], &["l", "r"])),
                    op(&["p"], &["q"]),
                    op(&["q"], &["l", "r"]),
                ],
                &[],
                &[],
            )
            .contract(&[&psi, &h, &psi])
            .unwrap()
            .scalar()
            .unwrap();
            close(&[eager], &[oracle]);
            close(&[planned], &[oracle]);
            assert_eq!(eager.to_bits(), cached_result.to_bits());
        }
    };
}

paths!(
    u1_paths_agree,
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 1),
            (U1Irrep::new(1), 2)
        ],
    )
    .unwrap()
);

paths!(
    su2_paths_agree,
    GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 1),
            (SU2Irrep::from_twice_spin(2), 1),
        ],
    )
    .unwrap()
);

paths!(
    fermionic_paths_agree,
    GradedSpace::try_new(
        Arc::new(FermionParityFusionRule),
        [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 1)],
    )
    .unwrap()
);

fn u1_space(dim: usize) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), dim), (U1Irrep::new(1), 1)],
    )
    .unwrap()
}

#[test]
fn contract_shares_the_topology_cache_and_counts_hits() {
    let runtime = Runtime::builder().build().unwrap();
    let v = u1_space(2);
    let a = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v], [&v], 10).unwrap();
    let b = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v], [&v], 11).unwrap();
    let c = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v], [&v], 12).unwrap();
    let network = chain();

    let first = network.contract(&[&a, &b, &c]).unwrap();
    let cold = plan_cache_stats(&runtime);
    assert_eq!((cold.misses, cold.hits, cold.entries), (1, 0, 1));
    assert_eq!(cold.workspaces_created, 1);

    // The network's own alias.
    let second = network.contract(&[&a, &b, &c]).unwrap();
    let warm = plan_cache_stats(&runtime);
    assert_eq!((warm.misses, warm.hits, warm.entries), (1, 1, 1));
    assert_eq!(warm.workspace_reuses, 1);
    assert_eq!(first.dense_data().unwrap(), second.dense_data().unwrap());

    // Equal networks built anew over the same topology
    // reach the same entry.
    chain().contract(&[&a, &b, &c]).unwrap();
    net(
        &[op(&["i"], &["j"]), op(&["j"], &["k"]), op(&["k"], &["m"])],
        &["i"],
        &["m"],
    )
    .contract(&[&a, &b, &c])
    .unwrap();
    let shared = plan_cache_stats(&runtime);
    assert_eq!((shared.misses, shared.hits, shared.entries), (1, 3, 1));

    // Another topology is another entry.
    let other = Network::new(
        vec![vec!["i", "j"], vec!["j", "k"], vec!["k", "m"]],
        vec![false; 3],
        vec![1; 3],
        vec!["m", "i"],
        Some(1),
    )
    .unwrap();
    other.contract(&[&a, &b, &c]).unwrap();
    let after = plan_cache_stats(&runtime);
    assert_eq!((after.misses, after.entries), (2, 2));

    clear_plan_cache(&runtime);
    assert_eq!(plan_cache_stats(&runtime), Default::default());
    network.contract(&[&a, &b, &c]).unwrap();
    assert_eq!(plan_cache_stats(&runtime).misses, 1);
}

#[test]
fn contract_rejects_before_the_lookup_and_leaves_the_cache_unchanged() {
    let runtime = Runtime::builder().build().unwrap();
    let v = u1_space(2);
    let w = u1_space(3);
    let a = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v], [&v], 20).unwrap();
    let b = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v], [&v], 21).unwrap();
    let c = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v], [&v], 22).unwrap();
    let wide = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v], [&w], 23).unwrap();
    let rank3 = TensorMap::<_, f64>::rand_with_seed(&runtime, [&v], [&v, &v], 24).unwrap();
    let network = chain();
    network.contract(&[&a, &b, &c]).unwrap();
    let before = plan_cache_stats(&runtime);

    for (what, tensors) in [
        ("operand count", vec![&a, &b]),
        ("contracted space", vec![&a, &wide, &c]),
        ("rank", vec![&a, &rank3, &c]),
    ] {
        assert!(network.contract(&tensors).is_err(), "{what}");
        assert_eq!(plan_cache_stats(&runtime), before, "{what}");
    }

    let foreign = Runtime::builder().build().unwrap();
    let elsewhere = TensorMap::<_, f64>::rand_with_seed(&foreign, [&v], [&v], 25).unwrap();
    assert!(network.contract(&[&a, &b, &elsewhere]).is_err());
    assert_eq!(plan_cache_stats(&runtime), before);
}

#[test]
fn contract_orders_round_trip_through_save_and_load() {
    let first = Runtime::builder().build().unwrap();
    assert_eq!(load_plan_cache(&first, ""), 0);
    let v = u1_space(2);
    let a = TensorMap::<_, f64>::rand_with_seed(&first, [&v], [&v], 40).unwrap();
    let b = TensorMap::<_, f64>::rand_with_seed(&first, [&v], [&v], 41).unwrap();
    let c = TensorMap::<_, f64>::rand_with_seed(&first, [&v], [&v], 42).unwrap();
    let expected = chain().contract(&[&a, &b, &c]).unwrap();
    let saved = save_plan_cache(&first);
    assert!(saved.contains("TOPO "));
    assert_eq!(plan_cache_stats(&first).persisted_orders, 1);

    let second = Runtime::builder().build().unwrap();
    assert_eq!(load_plan_cache(&second, &saved), 1);
    let a = TensorMap::<_, f64>::rand_with_seed(&second, [&v], [&v], 40).unwrap();
    let b = TensorMap::<_, f64>::rand_with_seed(&second, [&v], [&v], 41).unwrap();
    let c = TensorMap::<_, f64>::rand_with_seed(&second, [&v], [&v], 42).unwrap();
    let actual = chain().contract(&[&a, &b, &c]).unwrap();
    assert_eq!(actual.dense_data().unwrap(), expected.dense_data().unwrap());
    // The second runtime keeps exactly the loaded order under the same
    // topology text. Greedy search is deterministic, so replay of a loaded
    // order over a fresh search is shown by
    // `plan_text_input.rs::loaded_non_greedy_order_is_replayed_instead_of_searched`.
    assert_eq!(save_plan_cache(&second), saved);
}
