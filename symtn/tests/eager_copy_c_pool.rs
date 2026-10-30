//! #1626: an eager `contract(spec)` that takes TensorKit `blas_contract!`'s
//! `copyC` route (a zero-copy contraction into a temporary, then one permute
//! into the result) keeps that temporary in the Runtime's pooled execution
//! scratch, as `contract_into` keeps its own. A warm call therefore
//! allocates exactly one output-sized buffer: the returned tensor, and a warm
//! `contract_into` (#1631), which takes the same route, allocates none.

use std::sync::Arc;

use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{ContractSpec, GradedSpace, Runtime, TensorMap};

use crate::counting_alloc;

/// Allocations of exactly `bytes` made while `f` runs; its result is
/// dropped after counting stops.
fn sized_allocs<T>(bytes: usize, f: impl FnOnce() -> T) -> usize {
    let (result, allocs) = counting_alloc::measure_matching(bytes..=bytes, f);
    drop(result);
    allocs.matched_calls as usize
}

type Map = TensorMap<U1FusionRule, f64>;

/// `a: p ⊗ q ← c`, `b: c ← r`, distinct prime degeneracies so the output
/// payload size is unique among the call's allocations.
fn operands(runtime: &Runtime) -> (Map, Map) {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let space = |n| GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), n)]).unwrap();
    let (p, q, c, r) = (space(13), space(17), space(19), space(23));
    (
        Map::rand_with_seed(engine, runtime, [&p, &q], [&c], 1).unwrap(),
        Map::rand_with_seed(engine, runtime, [&c], [&r], 2).unwrap(),
    )
}

#[test]
fn warm_split_moving_eager_contract_allocates_only_its_result() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let (a, b) = operands(&runtime);
    let output_bytes = 13 * 17 * 23 * size_of::<f64>();
    // `r` joins the codomain and `q` moves to the domain: a split move, which
    // no GEMM writes directly (copyC).
    let spec = ContractSpec {
        lhs: &[2],
        rhs: &[0],
        codomain: &[0, 2],
        domain: &[1],
    };
    // Cold: the result and the pooled temporary.
    assert_eq!(
        sized_allocs(output_bytes, || a.contract(engine, &b, &spec).unwrap()),
        2
    );
    for _ in 0..3 {
        assert_eq!(
            sized_allocs(output_bytes, || a.contract(engine, &b, &spec).unwrap()),
            1
        );
    }

    // What: `contract_into` takes the same route (#1631) with the same pooled
    // temporary, so a warm call allocates nothing output-sized at all.
    let mut destination = a.contract(engine, &b, &spec).unwrap();
    for _ in 0..2 {
        a.contract_into(engine, &b, &spec, &mut destination, 1.0, 0.0)
            .unwrap();
    }
    for _ in 0..3 {
        assert_eq!(
            sized_allocs(output_bytes, || a
                .contract_into(engine, &b, &spec, &mut destination, 1.0, 0.0)
                .unwrap()),
            0
        );
    }

    // What: bit-identical to the literal copyC sequence spelled with public
    // operations — the default-split contraction, then the permute.
    let two_step = a
        .contract(
            engine,
            &b,
            &ContractSpec {
                lhs: &[2],
                rhs: &[0],
                codomain: &[0, 1],
                domain: &[2],
            },
        )
        .unwrap()
        .permute(engine, &[0, 2], &[1])
        .unwrap();
    let fused = a.contract(engine, &b, &spec).unwrap();
    assert_eq!(fused.codomain(), two_step.codomain());
    assert_eq!(fused.domain(), two_step.domain());
    assert_eq!(fused.dense_data().unwrap(), two_step.dense_data().unwrap());
}
