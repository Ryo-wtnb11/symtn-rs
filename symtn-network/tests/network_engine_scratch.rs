//! What: a warm network contraction allocates no route scratch on the
//! caller's engine. `a[x; c] · b[c; y, z, u]` runs first and its result
//! leaves in the orientation its consumer `t[y, z; w]` reads, a split move
//! (TensorKit `copyC`): that step's temporary is the one intermediate-sized
//! scratch. Cached `Network::contract` and planned `PlannedNetwork::execute`
//! both reach the eager contraction with the caller's engine, so the engine
//! keeps it; the intermediate itself is reused by the plan cache's pool or
//! the caller's workspace.

use std::sync::Arc;

use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{CpuEngine, GradedSpace, Runtime, TensorMap};
use symtn_network::{GreedyDenseOptimizer, NetworkExecutionWorkspace};

use crate::counting_alloc;
use crate::network_support::{net, op};

#[test]
fn warm_network_contract_allocates_no_route_scratch() {
    let _serial = counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let engine = &mut CpuEngine::serial();
    let space =
        |n| GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0).unwrap(), n)]).unwrap();
    let (x, c, y, z, u, w) = (
        space(7),
        space(11),
        space(13),
        space(17),
        space(19),
        space(23),
    );
    let a = TensorMap::<_, f64>::rand_with_seed(engine, &runtime, [&x], [&c], 1).unwrap();
    let b = TensorMap::<_, f64>::rand_with_seed(engine, &runtime, [&c], [&y, &z, &u], 2).unwrap();
    let t = TensorMap::<_, f64>::rand_with_seed(engine, &runtime, [&y, &z], [&w], 3).unwrap();
    let intermediate = 7 * 13 * 17 * 19 * size_of::<f64>();
    let network = || {
        net(
            &[
                op(&["x"], &["c"]),
                op(&["c"], &["y", "z", "u"]),
                op(&["y", "z"], &["w"]),
            ],
            &["x", "u"],
            &["w"],
        )
    };
    let tensors = [&a, &b, &t];
    let intermediate_allocs = |f: &mut dyn FnMut()| {
        counting_alloc::measure_matching(intermediate..=intermediate, f)
            .1
            .matched_calls
    };

    let cached = |engine: &mut CpuEngine| drop(network().contract(engine, &tensors).unwrap());
    for _ in 0..2 {
        cached(engine);
    }
    symtn::scratch::trim(engine);
    assert!(
        intermediate_allocs(&mut || cached(engine)) >= 1,
        "cached: cold"
    );
    assert_eq!(
        intermediate_allocs(&mut || cached(engine)),
        0,
        "cached: warm"
    );

    let planned = network().plan(&tensors, &GreedyDenseOptimizer).unwrap();
    let mut workspace = NetworkExecutionWorkspace::default();
    let mut execute =
        |engine: &mut CpuEngine| drop(planned.execute(engine, &tensors, &mut workspace).unwrap());
    for _ in 0..2 {
        execute(engine);
    }
    symtn::scratch::trim(engine);
    assert!(
        intermediate_allocs(&mut || execute(engine)) >= 1,
        "planned: cold"
    );
    assert_eq!(
        intermediate_allocs(&mut || execute(engine)),
        0,
        "planned: warm"
    );
}
