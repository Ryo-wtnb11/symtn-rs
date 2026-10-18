//! `DynamicProgramming` and `AutoHq` must survive networks the upstream `opt-einsum-path`
//! drivers reject (all-dim-1 gram topology from finite-torus CTRG) by
//! falling back down the auto-hq -> auto -> dp -> greedy chain.
#![cfg(feature = "opt-path")]

use std::sync::Arc;

use symtn::sector::{SU2FusionRule, SU2Irrep};
use symtn::typed::{GradedSpace, Runtime, TensorMap};
use symtn_network::{configure_plan_cache, Optimizer, PlanCacheConfig};

#[path = "../../tests/support/network.rs"]
mod network_support;
use network_support::{conj, net, op};

#[test]
fn mf_optimizers_fall_back_on_all_dim1_gram_topology() {
    // Upstream repro: dp/auto-hq error with "No contraction found for given
    // memory_limit" on this 6-operand topology when every dimension is 1.
    for (driver, optimizer) in [
        ("dp", Optimizer::DynamicProgramming),
        ("auto-hq", Optimizer::AutoHq),
    ] {
        let upstream = opt_einsum_path::contract_path(
            "abc,dce,bfgh,ghdi,jkf,kil->ajel",
            &[
                vec![1usize, 1, 1],
                vec![1, 1, 1],
                vec![1, 1, 1, 1],
                vec![1, 1, 1, 1],
                vec![1, 1, 1],
                vec![1, 1, 1],
            ],
            driver,
            opt_einsum_path::typing::SizeLimitType::None,
        );
        assert!(
            upstream.is_err(),
            "upstream fixed the dim-1 {driver} bug; drop the fallback?"
        );

        let rt = Runtime::builder().build().unwrap();
        configure_plan_cache(
            &rt,
            PlanCacheConfig {
                optimizer,
                ..PlanCacheConfig::default()
            },
        );
        let v = GradedSpace::try_new(Arc::new(SU2FusionRule), [(SU2Irrep::from_twice_spin(0), 1)])
            .unwrap();
        let cne = TensorMap::<_, f64>::rand_with_seed(&rt, [&v, &v], [&v], 1).unwrap();
        let sne = TensorMap::<_, f64>::rand_with_seed(&rt, [&v, &v], [&v], 2).unwrap();
        let ev = TensorMap::<_, f64>::rand_with_seed(&rt, [&v, &v], [&v, &v], 3).unwrap();
        let out = net(
            &[
                op(&["n3", "n4"], &["o3"]),
                conj(op(&["n3", "n5"], &["o1"])),
                op(&["n1", "n2"], &["n7", "n4"]),
                conj(op(&["n1", "n2"], &["n8", "n5"])),
                op(&["n7", "n6"], &["o4"]),
                conj(op(&["n8", "n6"], &["o2"])),
            ],
            &["o1", "o2"],
            &["o3", "o4"],
        )
        .contract(&[&cne, &cne, &ev, &ev, &sne, &sne]);
        assert!(out.is_ok(), "{driver}: {:?}", out.err());
    }
}
