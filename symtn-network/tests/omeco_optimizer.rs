//! `OmecoOptimizer` returns valid pairwise orders: the exhaustive search
//! reaches the brute-force minimum of total pair cost when no outer product
//! pays off, and stays above it when one does; TreeSA's order passes plan
//! validation, and both execute to the greedy plan's values.

use std::collections::HashMap;
use std::sync::Arc;

use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{GradedSpace, Runtime, TensorMap};
use symtn_network::{
    ContractionPlan, DenseContractionOptimizer, DenseCostModel, DenseTensorInfo,
    GreedyDenseOptimizer, NetworkExecutionWorkspace, NetworkIR, OmecoOptimizer,
};

use symtn_network::omeco::TreeSA;

use crate::network_support::{net, op};
use crate::numerics;

/// A network as per-tensor `(label, dim)` legs plus open labels.
struct Case {
    tensors: Vec<Vec<(&'static str, usize)>>,
    output: Vec<&'static str>,
}

impl Case {
    fn ir_and_cost(&self) -> (NetworkIR, DenseCostModel) {
        let labels = self
            .tensors
            .iter()
            .map(|legs| legs.iter().map(|&(label, _)| label.into()).collect())
            .collect();
        let output = self.output.iter().map(|&label| label.into()).collect();
        let ir = NetworkIR::from_labels(labels, output).unwrap();
        let infos = self
            .tensors
            .iter()
            .map(|legs| DenseTensorInfo::new(legs.iter().map(|&(_, dim)| dim).collect()))
            .collect::<Vec<_>>();
        let cost = DenseCostModel::from_network(&ir, &infos).unwrap();
        (ir, cost)
    }

    fn dims(&self) -> HashMap<&'static str, usize> {
        self.tensors.iter().flatten().copied().collect()
    }
}

fn cases() -> Vec<Case> {
    vec![
        // Matrix chain whose best order is neither left- nor right-first.
        Case {
            tensors: vec![
                vec![("a", 2), ("b", 64)],
                vec![("b", 64), ("c", 3)],
                vec![("c", 3), ("d", 64)],
                vec![("d", 64), ("e", 2)],
            ],
            output: vec!["a", "e"],
        },
        // Ring with two open legs.
        Case {
            tensors: vec![
                vec![("a", 8), ("b", 2), ("x", 3)],
                vec![("b", 2), ("c", 8)],
                vec![("c", 8), ("d", 4), ("y", 5)],
                vec![("d", 4), ("a", 8)],
            ],
            output: vec!["x", "y"],
        },
        // MPS environment update: L A W conj(A).
        Case {
            tensors: vec![
                vec![("a", 16), ("b", 4), ("c", 16)],
                vec![("a", 16), ("s", 2), ("d", 16)],
                vec![("b", 4), ("s", 2), ("t", 2), ("e", 4)],
                vec![("c", 16), ("t", 2), ("f", 16)],
            ],
            output: vec!["d", "e", "f"],
        },
    ]
}

/// Product of the dims of every label on either operand.
fn pair_cost<S: AsRef<str> + PartialEq>(
    dims: &HashMap<&str, usize>,
    lhs: &[S],
    rhs: &[S],
) -> usize {
    let mut union = lhs.iter().collect::<Vec<_>>();
    union.extend(rhs.iter().filter(|label| !lhs.contains(label)));
    union.iter().map(|label| dims[label.as_ref()]).product()
}

/// Minimum total pair cost over every pairwise tree, by exhaustive recursion.
fn brute_force(
    dims: &HashMap<&str, usize>,
    output: &[&str],
    live: Vec<Vec<&'static str>>,
) -> usize {
    if live.len() == 1 {
        return 0;
    }
    let mut best = usize::MAX;
    for i in 0..live.len() {
        for j in i + 1..live.len() {
            let mut rest = live.clone();
            let rhs = rest.remove(j);
            let lhs = rest.remove(i);
            let merged = lhs
                .iter()
                .chain(&rhs)
                .filter(|label| {
                    output.contains(label) || rest.iter().any(|legs| legs.contains(label))
                })
                .copied()
                .collect();
            let step = pair_cost(dims, &lhs, &rhs);
            rest.push(merged);
            best = best.min(step + brute_force(dims, output, rest));
        }
    }
    best
}

/// Total pair cost of a plan, recomputed from its labels (the stored step
/// costs also charge operand orientation).
fn plan_pair_cost(case: &Case, plan: &ContractionPlan) -> usize {
    let dims = case.dims();
    let mut labels = case
        .tensors
        .iter()
        .map(|legs| {
            legs.iter()
                .map(|&(label, _)| label.to_string())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut total = 0;
    for step in plan.steps() {
        let lhs = labels[step.lhs().index()].clone();
        let rhs = labels[step.rhs().index()].clone();
        total += pair_cost(&dims, &lhs, &rhs);
        let result = step
            .result_labels()
            .iter()
            .map(ToString::to_string)
            .collect();
        labels.push(result);
    }
    total
}

/// Total pair cost of omeco's exhaustive order and of the brute-force minimum.
fn exhaustive_and_minimum(case: &Case) -> (usize, usize) {
    let (ir, cost) = case.ir_and_cost();
    let steps = OmecoOptimizer::Exhaustive.optimize(&ir, &cost).unwrap();
    let plan = ContractionPlan::from_steps(&ir, steps).unwrap();
    let live = case
        .tensors
        .iter()
        .map(|legs| legs.iter().map(|&(label, _)| label).collect())
        .collect();
    (
        plan_pair_cost(case, &plan),
        brute_force(&case.dims(), &case.output, live),
    )
}

#[test]
fn exhaustive_reaches_the_brute_force_minimum() {
    for case in cases() {
        let (exhaustive, minimum) = exhaustive_and_minimum(&case);
        assert_eq!(exhaustive, minimum);
    }
}

/// `i,j,ijz->z`, i = j = 2, z = 10: `(i ijz) j` costs 2*2*10 + 2*10 = 60;
/// the outer product first, `(i j) ijz`, costs 2*2 + 2*2*10 = 44. omeco's
/// search never takes an outer product inside a connected component.
#[test]
fn exhaustive_skips_an_outer_product_that_would_pay_off() {
    let case = Case {
        tensors: vec![
            vec![("i", 2)],
            vec![("j", 2)],
            vec![("i", 2), ("j", 2), ("z", 10)],
        ],
        output: vec!["z"],
    };
    assert_eq!(exhaustive_and_minimum(&case), (60, 44));
}

#[test]
fn tree_sa_returns_a_valid_plan() {
    for case in cases() {
        let (ir, cost) = case.ir_and_cost();
        let steps = OmecoOptimizer::TreeSa(TreeSA::fast())
            .optimize(&ir, &cost)
            .unwrap();
        let plan = ContractionPlan::from_steps(&ir, steps).unwrap();
        assert_eq!(plan.steps().len(), case.tensors.len() - 1);
    }
}

#[test]
fn omeco_plans_execute_to_the_greedy_values() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let runtime = Runtime::new();
    let provider = Arc::new(U1FusionRule);
    let space = |dim| {
        GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0).unwrap(), dim)]).unwrap()
    };
    let dims = [2, 16, 3, 16, 2];
    let tensors = (0..4)
        .map(|i| {
            TensorMap::<_, f64>::rand_with_seed(
                engine,
                &runtime,
                [&space(dims[i])],
                [&space(dims[i + 1])],
                i as u64 + 1,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let tensors = tensors.iter().collect::<Vec<_>>();
    let network = net(
        &[
            op(&["a"], &["b"]),
            op(&["b"], &["c"]),
            op(&["c"], &["d"]),
            op(&["d"], &["e"]),
        ],
        &["a"],
        &["e"],
    );
    let mut run = |optimizer: &dyn DenseContractionOptimizer| {
        network
            .plan(&tensors, optimizer)
            .unwrap()
            .execute(engine, &tensors, &mut NetworkExecutionWorkspace::default())
            .unwrap()
    };
    let expected = run(&GreedyDenseOptimizer);
    for optimizer in [
        OmecoOptimizer::Exhaustive,
        OmecoOptimizer::TreeSa(TreeSA::fast()),
    ] {
        let actual = run(&optimizer);
        // Each entry sums 16 * 3 * 16 products.
        numerics::assert_slices_close(
            "omeco order against greedy",
            actual.dense_data().unwrap(),
            expected.dense_data().unwrap(),
            768,
        );
    }
}
