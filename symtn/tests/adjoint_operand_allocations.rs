//! Performance contract: allocation budget of a lazy adjoint operand.

use std::hint::black_box;
use std::sync::Arc;

use num_complex::Complex64;
use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::Runtime;
use symtn::typed::{GradedSpace, TensorMap};

use crate::counting_alloc;

fn measure(f: impl FnOnce()) -> (u64, u64) {
    let ((), allocs) = counting_alloc::measure(f);
    (allocs.calls, allocs.bytes)
}

fn tensor(runtime: &Runtime, degeneracy: usize, seed: u64) -> TensorMap<U1FusionRule, Complex64> {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let space = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        (-2..=2).map(|charge| (U1Irrep::new(charge).unwrap(), degeneracy)),
    )
    .unwrap();
    TensorMap::rand_with_seed(engine, runtime, [&space, &space], [&space], seed).unwrap()
}

/// What: building the lazy adjoint `&t.adjoint()?` and passing it to `inner`
/// costs one adjoint header and its logical layout, which does not grow with
/// degeneracy: no payload is copied.
#[test]
fn lazy_adjoint_operand_cost_does_not_grow_with_the_payload() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::new();
    let mut costs = Vec::new();
    for degeneracy in [1, 8] {
        let b = tensor(&runtime, degeneracy, 1);
        let a = tensor(&runtime, degeneracy, 2)
            .adjoint(engine)
            .unwrap()
            .materialize(engine)
            .unwrap();
        let inner = |engine: &mut symtn_dense::CpuEngine| {
            let b_adjoint = &b.adjoint(engine).unwrap();
            black_box(a.inner(engine, b_adjoint).unwrap());
        };
        inner(engine);
        costs.push(measure(|| inner(engine)));
    }
    assert_eq!(
        costs[0], costs[1],
        "lazy adjoint cost grew with the payload"
    );
}
