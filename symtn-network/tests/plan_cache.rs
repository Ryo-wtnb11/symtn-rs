use std::sync::{Arc, Barrier};
use symtn::typed::ContractSpec;

use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{Complex32, Complex64};
use symtn::typed::{GradedSpace, Runtime, TensorMap};
use symtn_network::{
    clear_plan_cache, configure_plan_cache, load_plan_cache, plan_cache_config, plan_cache_stats,
    save_plan_cache, PlanCacheConfig, ReplanPolicy,
};

#[path = "../../tests/support/network.rs"]
mod network_support;
use network_support::{net, op};

#[path = "../../tests/support/numerics.rs"]
mod numerics;

fn space(provider: Arc<U1FusionRule>, dim: usize) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(provider, [(U1Irrep::new(0), dim)]).unwrap()
}

fn pair(
    runtime: &Runtime,
    space: &GradedSpace<U1FusionRule>,
    seed: u64,
) -> (TensorMap<U1FusionRule, f64>, TensorMap<U1FusionRule, f64>) {
    (
        TensorMap::rand_with_seed(runtime, [space], [space], seed).unwrap(),
        TensorMap::rand_with_seed(runtime, [space], [space], seed + 1).unwrap(),
    )
}

#[test]
fn typed_static_cache_preserves_hit_clear_and_workspace_stats() {
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let space = space(provider, 3);
    let (a, b) = pair(&runtime, &space, 10);

    let first = net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i"], &["k"])
        .contract(&[&a, &b])
        .unwrap();
    let cold = plan_cache_stats(&runtime);
    assert_eq!((cold.misses, cold.hits, cold.entries), (1, 0, 1));
    assert_eq!(cold.topology_materializations, 1);
    assert_eq!(cold.workspaces_created, 1);
    assert_eq!(cold.idle_workspaces, 1);
    assert!(cold.retained_workspace_bytes > 0);
    assert_eq!(
        cold.peak_retained_workspace_bytes,
        cold.retained_workspace_bytes
    );
    assert_eq!(cold.workspace_byte_admissions, 1);
    assert_eq!(cold.workspace_byte_rejections, 0);

    let second = net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i"], &["k"])
        .contract(&[&a, &b])
        .unwrap();
    let warm = plan_cache_stats(&runtime);
    assert_eq!(first.dense_data().unwrap(), second.dense_data().unwrap());
    assert_eq!((warm.misses, warm.hits, warm.entries), (1, 1, 1));
    assert_eq!(warm.topology_materializations, 1);
    assert_eq!(warm.workspace_reuses, 1);

    clear_plan_cache(&runtime);
    assert_eq!(plan_cache_stats(&runtime), Default::default());
}

#[test]
fn parked_host_workspace_releases_provider_and_rebinds_exact_current_arc() {
    let runtime = Runtime::builder().build().unwrap();
    let first_provider = Arc::new(U1FusionRule);
    let first_weak = Arc::downgrade(&first_provider);
    let first_space = space(Arc::clone(&first_provider), 2);
    let (a, b) = pair(&runtime, &first_space, 20);
    drop(
        net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i"], &["k"])
            .contract(&[&a, &b])
            .unwrap(),
    );
    drop((a, b, first_space, first_provider));
    assert!(first_weak.upgrade().is_none());

    let current_provider = Arc::new(U1FusionRule);
    let current_space = space(Arc::clone(&current_provider), 2);
    let (c, d) = pair(&runtime, &current_space, 30);
    let output = net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i"], &["k"])
        .contract(&[&c, &d])
        .unwrap();
    assert!(std::ptr::eq(output.provider(), current_provider.as_ref()));
    assert_eq!(plan_cache_stats(&runtime).workspace_reuses, 1);
}

#[test]
fn dimension_drift_replans_without_put_before_residency_recheck() {
    let runtime = Runtime::builder().build().unwrap();
    configure_plan_cache(
        &runtime,
        PlanCacheConfig {
            replan: ReplanPolicy::DriftFactor(1.5),
            ..Default::default()
        },
    );
    let provider = Arc::new(U1FusionRule);
    let small = space(Arc::clone(&provider), 2);
    let large = space(provider, 5);
    let (a, b) = pair(&runtime, &small, 40);
    drop(
        net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i"], &["k"])
            .contract(&[&a, &b])
            .unwrap(),
    );
    let (c, d) = pair(&runtime, &large, 50);
    drop(
        net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i"], &["k"])
            .contract(&[&c, &d])
            .unwrap(),
    );
    let stats = plan_cache_stats(&runtime);
    assert_eq!((stats.misses, stats.replans, stats.entries), (1, 1, 1));
}

#[test]
fn disabled_cache_keeps_entries_and_counters_empty() {
    let runtime = Runtime::builder().build().unwrap();
    configure_plan_cache(
        &runtime,
        PlanCacheConfig {
            enabled: false,
            ..Default::default()
        },
    );
    let space = space(Arc::new(U1FusionRule), 2);
    let (a, b) = pair(&runtime, &space, 60);
    drop(
        net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i"], &["k"])
            .contract(&[&a, &b])
            .unwrap(),
    );
    drop(
        net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i"], &["k"])
            .contract(&[&a, &b])
            .unwrap(),
    );
    assert_eq!(plan_cache_stats(&runtime), Default::default());
}

#[test]
fn lru_hit_touches_entry_before_capacity_eviction() {
    let runtime = Runtime::builder().build().unwrap();
    configure_plan_cache(
        &runtime,
        PlanCacheConfig {
            capacity: 2,
            ..Default::default()
        },
    );
    let space = space(Arc::new(U1FusionRule), 2);
    let (a, b) = pair(&runtime, &space, 70);
    drop(
        net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i"], &["k"])
            .contract(&[&a, &b])
            .unwrap(),
    );
    drop(
        net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["k"], &["i"])
            .contract(&[&a, &b])
            .unwrap(),
    );
    drop(
        net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i"], &["k"])
            .contract(&[&a, &b])
            .unwrap(),
    );
    drop(
        net(&[op(&["j"], &["k"]), op(&["i"], &["j"])], &["k"], &["i"])
            .contract(&[&b, &a])
            .unwrap(),
    );
    let before = plan_cache_stats(&runtime);
    drop(
        net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i"], &["k"])
            .contract(&[&a, &b])
            .unwrap(),
    );
    assert_eq!(plan_cache_stats(&runtime).hits, before.hits + 1);
    drop(
        net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["k"], &["i"])
            .contract(&[&a, &b])
            .unwrap(),
    );
    let after = plan_cache_stats(&runtime);
    assert_eq!(after.misses, before.misses + 1);
    assert_eq!(after.entries, 2);
}

#[test]
fn persisted_typed_order_roundtrips_into_a_fresh_runtime() {
    let first = Runtime::builder().build().unwrap();
    assert_eq!(load_plan_cache(&first, ""), 0);
    let first_space = space(Arc::new(U1FusionRule), 2);
    let (a, b) = pair(&first, &first_space, 80);
    let expected = net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i"], &["k"])
        .contract(&[&a, &b])
        .unwrap();
    let saved = save_plan_cache(&first);
    assert!(saved.contains("TOPO "));

    let second = Runtime::builder().build().unwrap();
    assert_eq!(load_plan_cache(&second, &saved), 1);
    let second_space = space(Arc::new(U1FusionRule), 2);
    let (c, d) = pair(&second, &second_space, 80);
    let actual = net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i"], &["k"])
        .contract(&[&c, &d])
        .unwrap();
    assert_eq!(actual.dense_data().unwrap(), expected.dense_data().unwrap());
}

#[test]
fn clear_plan_cache_drops_persisted_orders_and_turns_persistence_off() {
    let first = Runtime::builder().build().unwrap();
    assert_eq!(load_plan_cache(&first, ""), 0);
    let first_space = space(Arc::new(U1FusionRule), 2);
    let (a, b) = pair(&first, &first_space, 82);
    drop(
        net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i"], &["k"])
            .contract(&[&a, &b])
            .unwrap(),
    );
    let saved = save_plan_cache(&first);

    let runtime = Runtime::builder().build().unwrap();
    assert_eq!(load_plan_cache(&runtime, &saved), 1);
    assert_eq!(plan_cache_stats(&runtime).persisted_orders, 1);
    clear_plan_cache(&runtime);
    assert_eq!(plan_cache_stats(&runtime), Default::default());

    // The miss searches afresh: nothing is replayed, and with persistence off
    // the fresh order is not recorded either.
    let space = space(Arc::new(U1FusionRule), 2);
    let (c, d) = pair(&runtime, &space, 84);
    drop(
        net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i"], &["k"])
            .contract(&[&c, &d])
            .unwrap(),
    );
    let stats = plan_cache_stats(&runtime);
    assert_eq!((stats.misses, stats.persisted_orders), (1, 0));
    assert!(!save_plan_cache(&runtime).contains("TOPO "));
}

#[test]
fn persisted_orders_are_bounded_by_the_plan_cache_capacity() {
    let runtime = Runtime::builder().build().unwrap();
    configure_plan_cache(
        &runtime,
        PlanCacheConfig {
            capacity: 2,
            ..Default::default()
        },
    );
    assert_eq!(load_plan_cache(&runtime, ""), 0);
    let space = space(Arc::new(U1FusionRule), 2);
    let (a, b) = pair(&runtime, &space, 86);
    drop(
        net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i"], &["k"])
            .contract(&[&a, &b])
            .unwrap(),
    );
    drop(
        net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["k"], &["i"])
            .contract(&[&a, &b])
            .unwrap(),
    );
    drop(
        net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i", "k"], &[])
            .contract(&[&a, &b])
            .unwrap(),
    );
    assert_eq!(plan_cache_stats(&runtime).persisted_orders, 2);
    let saved = save_plan_cache(&runtime);
    assert_eq!(saved.matches("TOPO ").count(), 2);

    // A loaded file larger than the bound keeps only the bound.
    let fresh = Runtime::builder().build().unwrap();
    configure_plan_cache(
        &fresh,
        PlanCacheConfig {
            capacity: 1,
            ..Default::default()
        },
    );
    assert_eq!(load_plan_cache(&fresh, &saved), 2);
    assert_eq!(plan_cache_stats(&fresh).persisted_orders, 1);
}

#[test]
fn concurrent_cached_calls_share_one_plan_and_bound_idle_pool() {
    let runtime = Runtime::builder().build().unwrap();
    let space = space(Arc::new(U1FusionRule), 8);
    let (a, b) = pair(&runtime, &space, 90);
    let expected = a
        .contract(
            &b,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    let barrier = Arc::new(Barrier::new(8));
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..8 {
            let barrier = Arc::clone(&barrier);
            let a = &a;
            let b = &b;
            handles.push(scope.spawn(move || {
                barrier.wait();
                net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i"], &["k"])
                    .contract(&[a, b])
                    .unwrap()
            }));
        }
        // The cached plan may group the length-8 contraction differently from the
        // direct call, so each thread is compared to it under the tolerance
        // rule.
        for handle in handles {
            numerics::assert_slices_close(
                "cached vs direct contract",
                handle.join().unwrap().dense_data().unwrap(),
                expected.dense_data().unwrap(),
                8,
            );
        }
    });
    let stats = plan_cache_stats(&runtime);
    assert_eq!(stats.entries, 1);
    assert!(stats.idle_workspaces <= 2);
    assert!(stats.workspaces_created >= stats.idle_workspaces as u64);
}

/// #1371: an operand from another Runtime is a metadata rejection, decided
/// before the plan lookup, so it leases (and quarantines) no workspace and the
/// next valid call reuses the idle one. Quarantine of a lease whose execution
/// fails is the lease-level `panic_quarantines_typed_workspace_lease` unit
/// test; after the preflight only a runtime fault can fail an execution.
#[test]
fn a_metadata_rejection_leases_no_workspace_and_the_next_call_reuses_the_idle_one() {
    let runtime = Runtime::builder().build().unwrap();
    let other = Runtime::builder().build().unwrap();
    let local_space = space(Arc::new(U1FusionRule), 2);
    let other_space = space(Arc::new(U1FusionRule), 2);
    let (a, b) = pair(&runtime, &local_space, 100);
    let (_, foreign) = pair(&other, &other_space, 110);
    drop(
        net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i"], &["k"])
            .contract(&[&a, &b])
            .unwrap(),
    );
    let warm = plan_cache_stats(&runtime);
    assert!(
        net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i"], &["k"])
            .contract(&[&a, &foreign])
            .is_err()
    );
    assert_eq!(plan_cache_stats(&runtime), warm);
    drop(
        net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i"], &["k"])
            .contract(&[&a, &b])
            .unwrap(),
    );
    let stats = plan_cache_stats(&runtime);
    assert_eq!(stats.workspaces_created, 1);
    assert_eq!(stats.workspace_reuses, 1);
    assert_eq!(stats.idle_workspaces, 1);
    assert_eq!(stats.workspace_byte_rejections, 0);
}

#[test]
fn workspace_budget_admits_or_rejects_the_complete_idle_workspace() {
    let runtime = Runtime::builder().build().unwrap();
    let space = space(Arc::new(U1FusionRule), 8);
    let (a, b) = pair(&runtime, &space, 130);
    let (c, _) = pair(&runtime, &space, 132);

    drop(
        net(
            &[op(&["i"], &["j"]), op(&["j"], &["k"]), op(&["k"], &["l"])],
            &["i"],
            &["l"],
        )
        .contract(&[&a, &b, &c])
        .unwrap(),
    );
    let charge = plan_cache_stats(&runtime).retained_workspace_bytes;
    assert!(charge > 8 * 8 * std::mem::size_of::<f64>());

    clear_plan_cache(&runtime);
    configure_plan_cache(
        &runtime,
        PlanCacheConfig {
            workspace_budget_bytes: charge - 1,
            ..Default::default()
        },
    );
    drop(
        net(
            &[op(&["i"], &["j"]), op(&["j"], &["k"]), op(&["k"], &["l"])],
            &["i"],
            &["l"],
        )
        .contract(&[&a, &b, &c])
        .unwrap(),
    );
    let rejected = plan_cache_stats(&runtime);
    assert_eq!(rejected.retained_workspace_bytes, 0);
    assert_eq!(rejected.peak_retained_workspace_bytes, 0);
    assert_eq!(rejected.idle_workspaces, 0);
    assert_eq!(rejected.workspace_byte_admissions, 0);
    assert_eq!(rejected.workspace_byte_rejections, 1);

    configure_plan_cache(
        &runtime,
        PlanCacheConfig {
            workspace_budget_bytes: charge,
            ..Default::default()
        },
    );
    drop(
        net(
            &[op(&["i"], &["j"]), op(&["j"], &["k"]), op(&["k"], &["l"])],
            &["i"],
            &["l"],
        )
        .contract(&[&a, &b, &c])
        .unwrap(),
    );
    let admitted = plan_cache_stats(&runtime);
    assert_eq!(admitted.retained_workspace_bytes, charge);
    assert_eq!(admitted.peak_retained_workspace_bytes, charge);
    assert_eq!(admitted.idle_workspaces, 1);
    assert_eq!(admitted.workspace_byte_admissions, 1);
    assert_eq!(admitted.workspace_byte_rejections, 1);

    configure_plan_cache(
        &runtime,
        PlanCacheConfig {
            workspace_budget_bytes: 0,
            ..Default::default()
        },
    );
    let zeroed = plan_cache_stats(&runtime);
    assert_eq!(zeroed.retained_workspace_bytes, 0);
    assert_eq!(zeroed.idle_workspaces, 0);
    assert_eq!(zeroed.workspace_byte_evictions, 1);
    drop(
        net(
            &[op(&["i"], &["j"]), op(&["j"], &["k"]), op(&["k"], &["l"])],
            &["i"],
            &["l"],
        )
        .contract(&[&a, &b, &c])
        .unwrap(),
    );
    assert_eq!(plan_cache_stats(&runtime).workspace_byte_rejections, 2);
}

#[test]
fn configuration_flushes_and_capacity_shrink_release_synchronously() {
    let runtime = Runtime::builder().build().unwrap();
    configure_plan_cache(
        &runtime,
        PlanCacheConfig {
            capacity: 2,
            ..Default::default()
        },
    );
    let space = space(Arc::new(U1FusionRule), 8);
    let (a, b) = pair(&runtime, &space, 140);
    let (c, _) = pair(&runtime, &space, 142);
    drop(
        net(
            &[op(&["i"], &["j"]), op(&["j"], &["k"]), op(&["k"], &["l"])],
            &["i"],
            &["l"],
        )
        .contract(&[&a, &b, &c])
        .unwrap(),
    );
    drop(
        net(
            &[op(&["i"], &["j"]), op(&["j"], &["k"]), op(&["k"], &["l"])],
            &["l"],
            &["i"],
        )
        .contract(&[&a, &b, &c])
        .unwrap(),
    );
    let initial = plan_cache_stats(&runtime);
    assert_eq!((initial.entries, initial.idle_workspaces), (2, 2));
    let two_workspace_budget = initial.retained_workspace_bytes;

    configure_plan_cache(
        &runtime,
        PlanCacheConfig {
            capacity: 2,
            workspace_budget_bytes: two_workspace_budget,
            ..Default::default()
        },
    );
    let flushed = plan_cache_stats(&runtime);
    assert_eq!(flushed.entries, 2);
    assert_eq!(flushed.idle_workspaces, 0);
    assert_eq!(flushed.retained_workspace_bytes, 0);
    assert_eq!(flushed.workspace_byte_evictions, 2);

    drop(
        net(
            &[op(&["i"], &["j"]), op(&["j"], &["k"]), op(&["k"], &["l"])],
            &["i"],
            &["l"],
        )
        .contract(&[&a, &b, &c])
        .unwrap(),
    );
    drop(
        net(
            &[op(&["i"], &["j"]), op(&["j"], &["k"]), op(&["k"], &["l"])],
            &["l"],
            &["i"],
        )
        .contract(&[&a, &b, &c])
        .unwrap(),
    );
    let repopulated = plan_cache_stats(&runtime);
    assert_eq!(repopulated.idle_workspaces, 2);
    assert_eq!(repopulated.retained_workspace_bytes, two_workspace_budget);

    configure_plan_cache(
        &runtime,
        PlanCacheConfig {
            capacity: 1,
            workspace_budget_bytes: two_workspace_budget,
            ..Default::default()
        },
    );
    let shrunk = plan_cache_stats(&runtime);
    assert_eq!(shrunk.entries, 1);
    assert_eq!(shrunk.idle_workspaces, 1);
    assert!(shrunk.retained_workspace_bytes < repopulated.retained_workspace_bytes);
    assert_eq!(shrunk.workspace_byte_evictions, 3);

    configure_plan_cache(
        &runtime,
        PlanCacheConfig {
            enabled: false,
            capacity: 1,
            workspace_budget_bytes: two_workspace_budget,
            ..Default::default()
        },
    );
    let disabled = plan_cache_stats(&runtime);
    assert_eq!((disabled.entries, disabled.idle_workspaces), (0, 0));
    assert_eq!(disabled.retained_workspace_bytes, 0);
    assert_eq!(disabled.workspace_byte_evictions, 4);
}

/// `configure_plan_cache` is the only post-build setter; a transition that
/// keeps the cache enabled and does not shrink it must keep every compiled
/// plan and idle workspace, so the next call is a hit, not a re-plan.
#[test]
fn enabled_reconfiguration_retains_plans_and_workspaces() {
    let runtime = Runtime::builder().build().unwrap();
    let space = space(Arc::new(U1FusionRule), 3);
    let (a, b) = pair(&runtime, &space, 150);
    let cold = net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i"], &["k"])
        .contract(&[&a, &b])
        .unwrap();
    let before = plan_cache_stats(&runtime);
    assert_eq!(
        (before.misses, before.entries, before.idle_workspaces),
        (1, 1, 1)
    );

    let grown = PlanCacheConfig {
        capacity: 2 * plan_cache_config(&runtime).capacity,
        workspace_budget_bytes: 2 * plan_cache_config(&runtime).workspace_budget_bytes,
        replan: ReplanPolicy::AlwaysReuse,
        ..Default::default()
    };
    configure_plan_cache(&runtime, grown.clone());
    let config = plan_cache_config(&runtime);
    assert_eq!(
        (
            config.capacity,
            config.workspace_budget_bytes,
            config.replan
        ),
        (grown.capacity, grown.workspace_budget_bytes, grown.replan)
    );
    let retained = plan_cache_stats(&runtime);
    assert_eq!(
        (
            retained.entries,
            retained.idle_workspaces,
            retained.retained_workspace_bytes
        ),
        (1, 1, before.retained_workspace_bytes)
    );

    let warm = net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i"], &["k"])
        .contract(&[&a, &b])
        .unwrap();
    let after = plan_cache_stats(&runtime);
    assert_eq!((after.misses, after.hits, after.replans), (1, 1, 0));
    assert_eq!(after.workspace_reuses, before.workspace_reuses + 1);
    assert_eq!(cold.dense_data().unwrap(), warm.dense_data().unwrap());
}

#[test]
fn a_foreign_extension_value_coexists_with_the_plan_cache() {
    // Another crate's value in the runtime's extension slot must not make the
    // first contraction panic ("runtime plan-cache slot claimed by another
    // type"). Each type has its own entry: the plan cache works and the
    // foreign value is neither read as a cache nor dropped.
    use symtn::typed::__network::with_extension_slot;
    let runtime = Runtime::builder().build().unwrap();
    with_extension_slot(&runtime, |slot| {
        slot.get_or_insert_with(|| 7u8);
    });
    let space = space(Arc::new(U1FusionRule), 2);
    let (a, b) = pair(&runtime, &space, 130);
    let first = net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i"], &["k"])
        .contract(&[&a, &b])
        .unwrap();
    let second = net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i"], &["k"])
        .contract(&[&a, &b])
        .unwrap();
    let stats = plan_cache_stats(&runtime);
    assert_eq!((stats.misses, stats.hits, stats.entries), (1, 1, 1));
    assert_eq!(first.dense_data().unwrap(), second.dense_data().unwrap());
    assert_eq!(
        with_extension_slot(&runtime, |slot| slot.get_mut::<u8>().copied()),
        Some(7)
    );
}

#[test]
fn dropping_warm_runtime_breaks_the_cache_workspace_cycle() {
    let runtime = Runtime::builder().build().unwrap();
    let identity = symtn::typed::__network::runtime_identity(&runtime);
    let space = space(Arc::new(U1FusionRule), 2);
    let (a, b) = pair(&runtime, &space, 120);
    let output = net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i"], &["k"])
        .contract(&[&a, &b])
        .unwrap();
    drop((output, a, b, space, runtime));
    assert!(!identity.is_alive());
}

#[test]
fn output_order_and_split_key_separate_cached_step_specs() {
    // The last step writes the requested output orientation in its own
    // ContractSpec, so that spec is part of the cached plan. Networks that
    // differ only in output order or split must neither share an entry nor
    // replay each other's spec.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let (p, q, c, r) = (
        space(Arc::clone(&provider), 2),
        space(Arc::clone(&provider), 3),
        space(Arc::clone(&provider), 4),
        space(Arc::clone(&provider), 5),
    );
    let a = TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&p, &q], [&c], 71).unwrap();
    let b = TensorMap::rand_with_seed(&runtime, [&c], [&r], 72).unwrap();
    let chain = a
        .contract(
            &b,
            &ContractSpec {
                lhs: &[2],
                rhs: &[0],
                codomain: &[0, 1],
                domain: &[2],
            },
        )
        .unwrap();
    let run = |which: usize| match which {
        0 => net(
            &[op(&["p", "q"], &["c"]), op(&["c"], &["r"])],
            &["p", "q"],
            &["r"],
        )
        .contract(&[&a, &b])
        .unwrap(),
        1 => net(
            &[op(&["p", "q"], &["c"]), op(&["c"], &["r"])],
            &["p", "r"],
            &["q"],
        )
        .contract(&[&a, &b])
        .unwrap(),
        _ => net(
            &[op(&["p", "q"], &["c"]), op(&["c"], &["r"])],
            &["q", "p"],
            &["r"],
        )
        .contract(&[&a, &b])
        .unwrap(),
    };
    let expected = [
        chain.clone(),
        chain.permute(&[0, 2], &[1]).unwrap(),
        chain.permute(&[1, 0], &[2]).unwrap(),
    ];
    for round in 0..2 {
        for (which, expected) in expected.iter().enumerate() {
            let actual = run(which);
            assert_eq!(actual.codomain(), expected.codomain(), "{round}/{which}");
            assert_eq!(actual.domain(), expected.domain(), "{round}/{which}");
            assert_eq!(
                actual.dense_data().unwrap(),
                expected.dense_data().unwrap(),
                "{round}/{which}"
            );
        }
    }
    let stats = plan_cache_stats(&runtime);
    assert_eq!((stats.misses, stats.hits, stats.entries), (3, 3, 3));
}
