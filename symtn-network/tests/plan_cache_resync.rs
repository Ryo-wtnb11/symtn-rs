//! `load_plan_cache` skips a corrupt entry and keeps loading the valid
//! entries after it.

use std::sync::Arc;

use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{GradedSpace, Runtime, TensorMap};
use symtn_network::{load_plan_cache, plan_cache_stats, save_plan_cache};

#[path = "../../tests/support/network.rs"]
mod network_support;
use network_support::{net, op};

/// A saved blob with two persisted orders (a two- and a three-tensor chain).
fn two_entry_blob() -> String {
    let runtime = Runtime::builder().build().unwrap();
    assert_eq!(load_plan_cache(&runtime, ""), 0);
    let space = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let t: Vec<TensorMap<U1FusionRule, f64>> = (0..3)
        .map(|seed| TensorMap::rand_with_seed(&runtime, [&space], [&space], seed).unwrap())
        .collect();
    drop(
        net(&[op(&["i"], &["j"]), op(&["j"], &["k"])], &["i"], &["k"])
            .contract(&[&t[0], &t[1]])
            .unwrap(),
    );
    drop(
        net(
            &[op(&["i"], &["j"]), op(&["j"], &["k"]), op(&["k"], &["l"])],
            &["i"],
            &["l"],
        )
        .contract(&[&t[0], &t[1], &t[2]])
        .unwrap(),
    );
    let saved = save_plan_cache(&runtime);
    assert_eq!(saved.matches("\nTOPO ").count(), 2);
    saved
}

#[test]
fn corrupt_entry_between_valid_entries_does_not_drop_the_later_one() {
    let saved = two_entry_blob();
    let second = saved.rfind("\nTOPO ").unwrap() + 1;
    let (before, after) = saved.split_at(second);
    for corrupt in [
        // No numeric count.
        "TOPO corrupt\nPLAN ?\ntensor_count 2\n",
        // No PLAN line at all: the next line is already the next entry.
        "TOPO corrupt\n",
        // A count larger than the plan: must not swallow the next entry.
        "TOPO corrupt\nPLAN 999\ntensor_count 2\n",
    ] {
        let blob = format!("{before}{corrupt}{after}");
        let runtime = Runtime::builder().build().unwrap();
        assert_eq!(load_plan_cache(&runtime, &blob), 2, "{corrupt:?}");
        assert_eq!(plan_cache_stats(&runtime).persisted_orders, 2);
        assert_eq!(save_plan_cache(&runtime), saved, "{corrupt:?}");
    }
}
