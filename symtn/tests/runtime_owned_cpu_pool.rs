//! #1531: every Host eager operation runs its replay, plan-compile, strided
//! and dense work on the CPU pool of the operand's own `Runtime`, whatever
//! order runtimes with different thread counts were built in. Its own test
//! binary: the process-global Rayon pool is sized here, to a count no runtime
//! below uses, so a region that escaped to it would be visible.

use std::sync::Arc;

use symtn::sector::{
    product_sector, ProductFusionRuleExt, SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep,
};
use symtn::typed::{GradedSpace, TensorMap};
use symtn::typed::{LegSelection, Runtime};
use symtn_tensors::host_pool::{
    observe_host_pools, take_host_pool_observations, HostPoolObservation, HostPoolSite,
};

const GLOBAL_THREADS: usize = 5;

type U1Su2 = symtn::sector::ProductFusionRule<U1FusionRule, SU2FusionRule>;

/// U(1) x SU(2) with degeneracies that put the payload past the replay
/// parallel gate, so every permutation recouples in parallel. The extra
/// `salt` sector gives each call a structure no earlier call compiled.
fn space(provider: &Arc<U1Su2>, salt: usize) -> GradedSpace<U1Su2> {
    let sector =
        |q, twice_spin| product_sector(U1Irrep::new(q), SU2Irrep::from_twice_spin(twice_spin));
    GradedSpace::try_new(
        Arc::clone(provider),
        [
            (sector(0, 0), 18),
            (sector(1, 1), 14),
            (sector(-1, 1), 14),
            (sector(0, 2), 12),
            (sector(2, salt), 1),
        ],
    )
    .unwrap()
}

fn runtime(threads: usize) -> Runtime {
    Runtime::builder()
        .dense_threads(threads)
        .recoupling_threads(8)
        .build()
        .unwrap()
}

/// Runs one of each Host work kind on `rt` and returns what the sites saw.
fn observe(rt: &Runtime, salt: usize) -> Vec<HostPoolObservation> {
    let provider = Arc::new(U1FusionRule.product(SU2FusionRule));
    let v = space(&provider, salt);
    let t: TensorMap<_, f64> = TensorMap::rand_with_seed(rt, [&v, &v], [&v], 1531).unwrap();
    assert!(
        t.dense_data().unwrap().len() > 1 << 15,
        "fixture must pass the parallel gates"
    );
    take_host_pool_observations();
    observe_host_pools(true);
    // Cold transform: plan compile, then parallel replay with strided blocks.
    let p = t.permute(&[1], &[2, 0]).unwrap();
    let _ = p.braid(&[1, 0], &[2], &[0, 2, 1]).unwrap();
    // One block past strided's threading gate, embedded by a strided copy.
    let w = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 40)]).unwrap();
    let block: TensorMap<_, f64> = TensorMap::rand_with_seed(rt, [&w, &w], [&w], 1532).unwrap();
    let selection = LegSelection::try_new(&w, [(U1Irrep::new(0), 0..39)]).unwrap();
    let _ = block
        .restrict_leg(&[(0, &selection)])
        .unwrap()
        .embed_leg(0, &selection)
        .unwrap();
    let _ = t.axpby(2.0, &t, 1.0).unwrap();
    let _ = t.svd_compact(&[0, 1], &[2]).unwrap();
    let _ = block.svd_compact(&[0, 1], &[2]).unwrap();
    let _ = block.lq_compact(&[0, 1], &[2]).unwrap();
    observe_host_pools(false);
    take_host_pool_observations()
}

/// Returns the identity of the pool `rt`'s work entered, so the caller can
/// check that each runtime has its own and keeps it.
fn assert_on_own_pool(rt: &Runtime, threads: usize, salt: usize) -> usize {
    let observations = observe(rt, salt);
    let id = observations
        .iter()
        .filter(|o| o.site != HostPoolSite::Dense)
        .find_map(|o| o.entered)
        .expect("no host region entered a pool");
    for site in [
        HostPoolSite::Replay,
        HostPoolSite::PlanCompile,
        HostPoolSite::Strided,
        HostPoolSite::Dense,
    ] {
        let seen: Vec<_> = observations.iter().filter(|o| o.site == site).collect();
        if threads == 1 {
            // One thread: no pool. Nothing may run on any Rayon worker, and
            // every region that is entered at all is entered in this runtime.
            assert!(
                seen.iter().all(|o| o.worker_pool_threads.is_none()),
                "{site:?}: {seen:?}"
            );
            assert!(
                seen.iter()
                    .filter(|o| o.site != HostPoolSite::Dense)
                    .all(|o| o.entered == Some(id)),
                "{site:?}: {seen:?}"
            );
            continue;
        }
        assert!(
            !seen.is_empty(),
            "{site:?} was not exercised: {observations:?}"
        );
        for o in &seen {
            // The pool size identifies the pool: the global one has a count
            // no runtime here uses.
            assert_eq!(
                o.worker_pool_threads,
                Some(threads),
                "{site:?} ran off the runtime pool"
            );
            // A task stolen by another worker of the pool, or the dense body
            // Tenferro installs there, carries no symtn entry of its own and
            // runs on the ambient pool: the one it is a worker of.
            assert!(
                o.entered.is_none_or(|e| e == id),
                "{site:?} entered another pool"
            );
        }
        if site != HostPoolSite::Dense {
            assert!(
                seen.iter().any(|o| o.entered == Some(id)),
                "{site:?}: {seen:?}"
            );
        }
    }
    id
}

#[test]
fn each_runtime_runs_host_work_on_its_own_pool_in_either_build_order() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(GLOBAL_THREADS)
        .build_global()
        .unwrap();

    // A serial runtime built first does not cap a later one.
    let (serial, wide) = (runtime(1), runtime(4));
    let wide_pool = assert_on_own_pool(&wide, 4, 0);
    let serial_pool = assert_on_own_pool(&serial, 1, 1);

    // Reverse order, two parallel counts, interleaved use.
    let (three, two) = (runtime(3), runtime(2));
    let two_pool = assert_on_own_pool(&two, 2, 2);
    let three_pool = assert_on_own_pool(&three, 3, 3);
    assert_eq!(
        assert_on_own_pool(&wide, 4, 4),
        wide_pool,
        "wide changed pools"
    );
    let pools = std::collections::BTreeSet::from([wide_pool, serial_pool, two_pool, three_pool]);
    assert_eq!(pools.len(), 4, "runtimes share a pool: {pools:?}");
}
