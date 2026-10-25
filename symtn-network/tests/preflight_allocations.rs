//! Allocation contracts of the network metadata preflight (#1371).
//!
//! The counter is per thread, so the tests do not disturb each other; each
//! call under test runs on the test's own thread.

use std::sync::Arc;
use symtn::typed::ContractSpec;

use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::Runtime;
use symtn::typed::{GradedSpace, TensorMap};
use symtn_network::{
    slice_plan_for, DenseCostModel, DenseTensorInfo, GreedyDenseOptimizer, Network,
    NetworkExecutionWorkspace, NetworkIR, SlicedPlan, TemporaryLabel,
};

use crate::braiding_probe;
use braiding_probe::{ProbeSector, RealBraidingProbe};

use crate::counting_alloc;

/// Allocation calls (alloc and realloc) made by `run` on this thread.
fn allocations<T>(run: impl FnOnce() -> T) -> (u64, T) {
    let (value, allocs) = counting_alloc::measure(run);
    (allocs.calls, value)
}

/// `a: [v, v; w]`, `b: [w; v, v]` and `c: [w*; v, v]`, 8 U(1) sectors of
/// degeneracy 2 per leg.
fn u1_operands(runtime: &Runtime) -> [TensorMap<U1FusionRule, f64>; 3] {
    let provider = Arc::new(U1FusionRule);
    let space = |shift: i32| {
        GradedSpace::try_new(
            Arc::clone(&provider),
            (0..8).map(|charge| (U1Irrep::new(charge - 4 + shift), 2)),
        )
        .unwrap()
    };
    let (v, w) = (space(0), space(1));
    [
        TensorMap::rand_with_seed(runtime, [&v, &v], [&w], 1_371_100).unwrap(),
        TensorMap::rand_with_seed(runtime, [&w], [&v, &v], 1_371_101).unwrap(),
        TensorMap::rand_with_seed(runtime, [&w.try_dual().unwrap()], [&v, &v], 1_371_102).unwrap(),
    ]
}

fn pair_network() -> Network {
    Network::new(
        [vec!["i", "j", "m"], vec!["m", "k", "l"]],
        [false, false],
        vec![2, 1],
        ["i", "j", "k", "l"],
        Some(2),
    )
    .unwrap()
}

/// #1371 / #2022: a warm `Network::contract` hit adds no allocation to the
/// replay it runs. Its alias lookup hashes the topology hash `Network::new`
/// computed and compares the cached topology in place, and its metadata
/// preflight reads the pairing `Network::new` resolved: it would allocate if
/// it built dualised spaces, label lists or a label map. The same replay with
/// a caller-owned workspace is the baseline.
#[test]
fn a_warm_network_contract_allocates_no_more_than_its_replay() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let [a, b, c] = u1_operands(&runtime);
    let network = pair_network();
    let planned = network.plan(&[&a, &b], &GreedyDenseOptimizer).unwrap();
    let mut workspace = NetworkExecutionWorkspace::default();
    for _ in 0..3 {
        drop(network.contract(&[&a, &b]).unwrap());
        drop(planned.execute(&[&a, &b], &mut workspace).unwrap());
    }
    let (by_replay, output) = allocations(|| planned.execute(&[&a, &b], &mut workspace));
    drop(output.unwrap());
    let (by_contract, output) = allocations(|| network.contract(&[&a, &b]));
    drop(output.unwrap());
    eprintln!("warm call: replay {by_replay}, Network::contract {by_contract} allocation calls");
    assert!(by_contract <= by_replay, "{by_contract} > {by_replay}");

    // A `conj` operand: the preflight reads its legs through the reversed
    // written/lowered axis map, still without allocating. `conj(b)` is
    // `[v, v; w]`, so it contracts with `b` over all three legs.
    let conj_pair = Network::new(
        [["m", "i", "j"], ["m", "i", "j"]],
        [true, false],
        [1, 1],
        [],
        Some(0),
    )
    .unwrap();
    let planned = conj_pair.plan(&[&b, &b], &GreedyDenseOptimizer).unwrap();
    let mut workspace = NetworkExecutionWorkspace::default();
    for _ in 0..3 {
        drop(conj_pair.contract(&[&b, &b]).unwrap());
        drop(planned.execute(&[&b, &b], &mut workspace).unwrap());
    }
    let (by_replay, output) = allocations(|| planned.execute(&[&b, &b], &mut workspace));
    drop(output.unwrap());
    let (by_contract, output) = allocations(|| conj_pair.contract(&[&b, &b]));
    drop(output.unwrap());
    eprintln!("warm conj call: replay {by_replay}, Network::contract {by_contract}");
    assert!(
        by_contract <= by_replay,
        "conj: {by_contract} > {by_replay}"
    );

    // Non-vacuity: the same preflight rejects `c`, whose `m` leg has the
    // wrong duality.
    let error = network.contract(&[&a, &c]).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("space mismatch for contracted label `m`"),
        "{error}"
    );
}

/// #1371, #1372: `PlannedNetwork::execute` and the sliced path
/// reject a non-symmetric braiding before their first step or accumulator:
/// with the typed `contract`'s error, allocating no more than the typed
/// `contract` does to reject the same operands (the error itself).
#[test]
fn planned_and_sliced_executions_reject_non_symmetric_braiding_first() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg =
        GradedSpace::try_new(Arc::new(RealBraidingProbe::<true>), [(ProbeSector, 2)]).unwrap();
    let lhs = TensorMap::<_, f64>::rand_with_seed(&runtime, [&leg], [&leg], 1_371_200).unwrap();
    let rhs = TensorMap::<_, f64>::rand_with_seed(&runtime, [&leg], [&leg], 1_371_201).unwrap();
    let tensors = [&lhs, &rhs];
    let labels = |names: &[&str]| {
        names
            .iter()
            .copied()
            .map(TemporaryLabel::from)
            .collect::<Vec<_>>()
    };
    let inputs = vec![labels(&["a", "x"]), labels(&["x", "b"])];
    let output = labels(&["a", "b"]);
    let network = Network::new(
        inputs.clone(),
        vec![false; 2],
        vec![1, 1],
        output.clone(),
        Some(1),
    )
    .unwrap();
    let planned = network.plan(&tensors, &GreedyDenseOptimizer).unwrap();
    let ir = NetworkIR::from_labels(inputs, output).unwrap();
    let cost = DenseCostModel::from_network(
        &ir,
        &[
            DenseTensorInfo::new(vec![2, 2]),
            DenseTensorInfo::new(vec![2, 2]),
        ],
    )
    .unwrap();
    let dense = SlicedPlan::new(
        planned.plan().clone(),
        slice_plan_for(&ir, planned.plan(), &cost, &labels(&["a"])),
    );
    let sliced = network
        .lower_symmetric_sliced_plan(&tensors, dense)
        .unwrap();

    let (contract_count, contract) = allocations(|| {
        lhs.contract(
            &rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .map(drop)
    });
    let contract = contract.unwrap_err().to_string();

    let (count, error) =
        allocations(|| planned.execute(&tensors, &mut Default::default()).map(drop));
    assert_eq!(error.unwrap_err().to_string(), contract, "execute");
    assert!(
        count <= contract_count,
        "execute: {count} > {contract_count}"
    );

    let (count, error) = allocations(|| {
        network
            .execute_symmetric_sliced(&tensors, sliced, usize::MAX)
            .map(drop)
    });
    let error = error.unwrap_err();
    assert!(
        matches!(
            &error,
            symtn_network::SymmetricSliceExecutionError::Tensor(inner)
                if inner.to_string() == contract
        ),
        "{error:?}"
    );
    assert!(
        count <= contract_count,
        "sliced: {count} > {contract_count}"
    );
}
