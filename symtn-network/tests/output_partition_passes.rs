//! What: a warm contraction whose output codomain takes legs from both
//! operands costs one output-sized allocation, by the eager `contract(spec)`
//! and by `Network::contract` alike: the network step passes the output split as its
//! `ContractSpec` (TensorOperations `pAB`), so no separate permute pass and no
//! second output-sized buffer follow the contraction. Spelling the same result
//! as contract-then-permute still costs two.

use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{ContractSpec, GradedSpace, Runtime, TensorMap};

use crate::network_support;
use network_support::{net, op};

use crate::counting_alloc;

/// The output payload of `operands()`'s contraction, in bytes; the distinct
/// prime degeneracies keep it unique in the run.
const PAYLOAD_BYTES: usize = 13 * 17 * 23 * size_of::<f64>();

/// Output-payload-sized allocations made while `f` runs (result dropped after).
fn output_sized_allocs<T>(f: impl FnOnce() -> T) -> usize {
    let (result, allocs) = counting_alloc::measure_matching(PAYLOAD_BYTES..=PAYLOAD_BYTES, f);
    drop(result);
    allocs.matched_calls as usize
}

type Map = TensorMap<U1FusionRule, f64>;

/// `a: p ⊗ q ← c`, `b: c ← r` on a fresh runtime (cold plan cache and pool).
/// Distinct prime degeneracies keep the output payload size unique.
fn operands() -> (Map, Map) {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let space = |n| {
        GradedSpace::try_new(std::sync::Arc::new(U1FusionRule), [(U1Irrep::new(0), n)]).unwrap()
    };
    let (p, q, c, r) = (space(13), space(17), space(19), space(23));
    (
        Map::rand_with_seed(&runtime, [&p, &q], [&c], 1).unwrap(),
        Map::rand_with_seed(&runtime, [&c], [&r], 2).unwrap(),
    )
}

#[test]
fn mixed_output_partition_costs_one_output_sized_allocation() {
    let _serial = crate::counting_alloc::serial();
    let (a, b) = operands();
    assert_eq!(
        output_sized_allocs(|| a
            .contract(
                &b,
                &ContractSpec {
                    lhs: &[2],
                    rhs: &[0],
                    codomain: &[0, 1],
                    domain: &[2]
                }
            )
            .unwrap()),
        1
    );
    let (a, b) = operands();
    assert_eq!(
        output_sized_allocs(|| {
            a.contract(
                &b,
                &ContractSpec {
                    lhs: &[2],
                    rhs: &[0],
                    codomain: &[0, 1],
                    domain: &[2],
                },
            )
            .unwrap()
            .permute(&[0, 2], &[1])
            .unwrap()
        }),
        2
    );

    let (a, b) = operands();
    let moved = || {
        a.contract(
            &b,
            &ContractSpec {
                lhs: &[2],
                rhs: &[0],
                codomain: &[0, 2],
                domain: &[1],
            },
        )
        .unwrap()
    };
    // Cold: the result and the Runtime-pooled copyC temporary (#1626).
    assert_eq!(output_sized_allocs(moved), 2);
    assert_eq!(output_sized_allocs(moved), 1);

    let (a, b) = operands();
    assert_eq!(
        output_sized_allocs(|| net(
            &[op(&["p", "q"], &["c"]), op(&["c"], &["r"])],
            &["p", "q"],
            &["r"]
        )
        .contract(&[&a, &b])
        .unwrap()),
        1
    );
    let (a, b) = operands();
    let mixed = || {
        net(
            &[op(&["p", "q"], &["c"]), op(&["c"], &["r"])],
            &["p", "r"],
            &["q"],
        )
        .contract(&[&a, &b])
        .unwrap()
    };
    // The step's spec is that same contract: cold it pools its temporary,
    // warm only the result is output-sized — no permute pass follows.
    assert_eq!(output_sized_allocs(mixed), 2);
    assert_eq!(output_sized_allocs(mixed), 1);
    assert_eq!(output_sized_allocs(mixed), 1);
}
