//! Contraction-order search by the `omeco` crate (MIT).

use std::collections::{BTreeMap, HashMap};

use omeco::{CodeOptimizer, EinCode, NestedEinsum};

use crate::cost::DenseCostModel;
use crate::error::{ContractError, Result};
use crate::ir::NetworkIR;
use crate::labels::TemporaryLabel;
use crate::optimizer::{ContractionStep, DenseContractionOptimizer};
use crate::plan::{dense_steps_from_active_pair_path, orient_unordered_active_pairs, ActivePair};

/// Order search by [omeco](https://crates.io/crates/omeco): exact search or
/// tree simulated annealing, over the same per-label dimensions as
/// [`crate::GreedyDenseOptimizer`]. Pass it to [`crate::Network::plan`]; the
/// returned order is validated and executed like any other plan.
///
/// - **Not deterministic.** omeco breaks cost ties by iterating a
///   `std::collections::HashMap`, whose order is seeded per process, so two
///   processes can return different orders of equal estimated cost (and,
///   through a different summation order, results that differ in rounding).
///   [`crate::GreedyDenseOptimizer`] is the deterministic default.
/// - **Threads.** [`TreeSa`](Self::TreeSa) runs its trials on the global
///   rayon pool, outside the engine's thread budget.
/// - **Not on a default path.** [`crate::Network::contract`] and the plan
///   cache plan with [`crate::Optimizer`]; this type is used only where a
///   caller passes it.
/// - **Slicing** stays with [`crate::greedy_slice`]: omeco's slicer always
///   treats open legs as candidates (no [`crate::SliceLabels::InternalOnly`])
///   and re-anneals the tree while slicing, so its result is not a
///   [`crate::SlicedPlan`] over a fixed, validated order.
#[derive(Debug, Clone)]
pub enum OmecoOptimizer {
    /// The order of least total pair cost (omeco `ExhaustiveSearch`, an
    /// exact dynamic program exponential in the network size; at most 128
    /// tensors and 128 labels).
    Optimal,
    /// omeco `TreeSA` with the given configuration: a cold-path search for
    /// networks where greedy goes wrong.
    TreeSa(omeco::TreeSA),
}

impl DenseContractionOptimizer for OmecoOptimizer {
    fn optimize(
        &self,
        ir: &NetworkIR,
        cost_model: &DenseCostModel,
    ) -> Result<Vec<ContractionStep>> {
        let tensor_count = ir.tensors().len();
        if tensor_count < 2 {
            return Err(ContractError::NotEnoughTensors);
        }
        let mut ids = BTreeMap::<&TemporaryLabel, usize>::new();
        let mut id = |label| {
            let next = ids.len();
            *ids.entry(label).or_insert(next)
        };
        let ixs = ir
            .tensors()
            .iter()
            .map(|tensor| tensor.labels().iter().map(&mut id).collect())
            .collect();
        let iy = ir.output_labels().iter().map(&mut id).collect();
        let sizes = ids
            .iter()
            .map(|(label, &id)| (id, cost_model.dim(label).unwrap_or(1)))
            .collect::<HashMap<_, _>>();
        let code = EinCode::new(ixs, iy);
        let tree = match self {
            Self::Optimal => {
                omeco::optimize_exhaustive(&code, &sizes, &omeco::ExhaustiveSearch::default())
                    .map_err(|error| omeco_error(error.to_string()))?
            }
            Self::TreeSa(config) => config
                .optimize(&code, &sizes)
                .ok_or_else(|| omeco_error("TreeSA found no contraction tree".to_string()))?,
        };
        let mut active = (0..tensor_count).collect::<Vec<_>>();
        let mut pairs = Vec::with_capacity(tensor_count - 1);
        push_pairs(&tree, tensor_count, &mut active, &mut pairs)?;
        let pairs = orient_unordered_active_pairs(&pairs, tensor_count)?;
        dense_steps_from_active_pair_path(ir, &pairs, cost_model)
    }
}

fn omeco_error(message: String) -> ContractError {
    ContractError::InvalidContractionPlan(format!("omeco: {message}"))
}

/// Appends the post-order active pairs of `node` and returns its id: an input
/// index for a leaf, `tensor_count + k` for the k-th contraction.
fn push_pairs(
    node: &NestedEinsum<usize>,
    tensor_count: usize,
    active: &mut Vec<usize>,
    pairs: &mut Vec<ActivePair>,
) -> Result<usize> {
    let args = match node {
        NestedEinsum::Leaf { tensor_index } => return Ok(*tensor_index),
        NestedEinsum::Node { args, .. } => args,
    };
    let [lhs, rhs] = args.as_slice() else {
        return Err(omeco_error(format!(
            "a tree node has {} operands; symtn plans are pairwise",
            args.len()
        )));
    };
    let lhs = push_pairs(lhs, tensor_count, active, pairs)?;
    let rhs = push_pairs(rhs, tensor_count, active, pairs)?;
    let position = |id| {
        active
            .iter()
            .position(|&live| live == id)
            .ok_or_else(|| omeco_error(format!("tree uses operand {id} twice")))
    };
    pairs.push(ActivePair::new(position(lhs)?, position(rhs)?));
    let result = tensor_count + pairs.len() - 1;
    active.retain(|&live| live != lhs && live != rhs);
    active.push(result);
    Ok(result)
}
