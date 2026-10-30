//! #1614: an intermediate network step whose next-use orientation moves legs
//! across the codomain/domain split is one contraction into one retained
//! buffer, not a contraction followed by a separate permute call into a
//! second one. On the copyC route the data moved is the same (GEMM into the
//! pooled temporary, then one transform); the saving is the separate call and
//! its second buffer.
//!
//! This test, with the step loop's structure (no permute path per step), is
//! the evidence for #1614's gate: symtn-network has no transform-seam
//! counter, so the warm transform-lookup count and the retained workspace
//! bytes stand in for it.

use std::sync::Arc;
use symtn::typed::Engine;

use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{ContractSpec, GradedSpace, Runtime, TensorMap};
use symtn_network::plan_cache_stats;

use crate::network_support;
use network_support::{net, op};

use crate::numerics;

use crate::counting_alloc;

type Map = TensorMap<U1FusionRule, f64>;

/// Lookups of the process-global completed-transformer cache; this
/// target serializes all tests, so only this test moves them.
fn transform_lookups(_: &Runtime) -> usize {
    let info = completed_transformers();
    (info.hits() + info.misses()) as usize
}

#[test]
fn split_moving_intermediate_retains_one_buffer_and_one_transform() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // a[x; c] · b[c; y, z, u] runs first (x < w), giving (x | y, z, u). Its
    // consumer contracts y and z with t[y, z; w], so the next-use orientation
    // is (x, u | y, z) as lhs or (y, z | x, u) as rhs: two codomain legs where
    // the default split has one, a split move either way. Distinct prime
    // degeneracies keep the intermediate payload size unique in the run.
    let runtime = Runtime::builder().build().unwrap();
    let space = |n| GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), n)]).unwrap();
    let (x, c, y, z, u, w) = (
        space(11),
        space(13),
        space(17),
        space(19),
        space(23),
        space(29),
    );
    let a = Map::rand_with_seed(engine, &runtime, [&x], [&c], 1).unwrap();
    let b = Map::rand_with_seed(engine, &runtime, [&c], [&y, &z, &u], 2).unwrap();
    let t = Map::rand_with_seed(engine, &runtime, [&y, &z], [&w], 3).unwrap();
    let intermediate_bytes = 11 * 17 * 19 * 23 * size_of::<f64>();

    let run = |engine: &mut dyn Engine| {
        net(
            &[
                op(&["x"], &["c"]),
                op(&["c"], &["y", "z", "u"]),
                op(&["y", "z"], &["w"]),
            ],
            &["x", "u"],
            &["w"],
        )
        .contract(engine, &[&a, &b, &t])
        .unwrap()
    };
    let oracle = a
        .contract(
            engine,
            &b,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1, 2, 3],
            },
        )
        .unwrap()
        .contract(
            engine,
            &t,
            &ContractSpec {
                lhs: &[1, 2],
                rhs: &[0, 1],
                codomain: &[0, 1],
                domain: &[2],
            },
        )
        .unwrap();
    for _ in 0..2 {
        let output = run(engine);
        numerics::assert_slices_close(
            "network vs chained contract",
            output.dense_data().unwrap(),
            oracle.dense_data().unwrap(),
            13 * 17 * 19,
        );
    }

    let lookups_before = transform_lookups(&runtime);
    let ((), allocs) =
        counting_alloc::measure_matching(intermediate_bytes..=intermediate_bytes, || {
            drop(run(engine))
        });
    let warm_allocs = allocs.matched_calls;
    let warm_lookups = transform_lookups(&runtime) - lookups_before;
    let retained = plan_cache_stats(&runtime).retained_workspace_bytes;

    // What: the intermediate's buffer is reused, not reallocated.
    assert_eq!(warm_allocs, 0, "warm intermediate-sized allocations");
    // What: one transform for the one split-moving intermediate — the
    // step's own output permute (copyC). An operand-rebuilding route would
    // look up three.
    assert!(warm_lookups <= 1, "warm transform lookups: {warm_lookups}");
    // What: the idle workspace keeps one intermediate buffer plus metadata.
    // A separate orientation permute retains a second intermediate-sized
    // buffer (1,319,096 B on the pre-#1614 lowering against 660,988 B).
    assert!(
        retained >= intermediate_bytes && retained < intermediate_bytes + intermediate_bytes / 4,
        "retained workspace bytes {retained} vs one intermediate buffer {intermediate_bytes}"
    );
}

/// The process-global completed-transformer cache (`symtn::cache`).
fn completed_transformers() -> symtn::cache::StructureCacheInfo {
    symtn::cache::stats()
        .into_iter()
        .find(|info| info.kind() == symtn::cache::StructureCacheKind::CompletedTreeTransformer)
        .expect("every structure cache kind reports")
}
