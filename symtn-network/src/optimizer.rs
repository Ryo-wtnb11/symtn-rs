use std::collections::{BTreeSet, HashMap};

use crate::cost::DenseCostModel;
use crate::error::{ContractError, Result};
use crate::ir::NetworkIR;
use crate::labels::{TemporaryLabel, TensorId};
use crate::stepflow::{consumers, declared_step_labels, pair_result_labels, planned_label_orders};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContractionStep {
    lhs: TensorId,
    rhs: TensorId,
    result: TensorId,
    cost: usize,
    result_labels: Vec<TemporaryLabel>,
}

impl ContractionStep {
    pub fn new(
        lhs: TensorId,
        rhs: TensorId,
        result: TensorId,
        cost: usize,
        result_labels: Vec<TemporaryLabel>,
    ) -> Self {
        Self {
            lhs,
            rhs,
            result,
            cost,
            result_labels,
        }
    }

    pub fn lhs(&self) -> TensorId {
        self.lhs
    }

    pub fn rhs(&self) -> TensorId {
        self.rhs
    }

    pub fn result(&self) -> TensorId {
        self.result
    }

    pub fn cost(&self) -> usize {
        self.cost
    }

    pub fn result_labels(&self) -> &[TemporaryLabel] {
        &self.result_labels
    }
}

#[derive(Debug, Clone)]
struct ActiveTensor {
    id: TensorId,
    labels: Vec<TemporaryLabel>,
    /// Earliest written operand represented by this subtree. Pair selection
    /// uses active-list positions, but contraction orientation must preserve
    /// the expression's left-to-right factor order after results are appended.
    first_input: usize,
}

fn greedy_order(ir: &NetworkIR, cost_model: &DenseCostModel) -> Result<Vec<ContractionStep>> {
    if ir.tensors().len() < 2 {
        return Err(ContractError::NotEnoughTensors);
    }

    let mut active = ir
        .tensors()
        .iter()
        .enumerate()
        .map(|(first_input, tensor)| ActiveTensor {
            id: tensor.id(),
            labels: tensor.labels().to_vec(),
            first_input,
        })
        .collect::<Vec<_>>();
    let mut steps = Vec::new();

    while active.len() > 1 {
        let mut best = None::<(usize, usize, usize)>;
        for lhs_index in 0..active.len() {
            for rhs_index in (lhs_index + 1)..active.len() {
                let cost =
                    cost_model.pair_cost(&active[lhs_index].labels, &active[rhs_index].labels);
                match best {
                    Some((_, _, best_cost)) if best_cost <= cost => {}
                    _ => best = Some((lhs_index, rhs_index, cost)),
                }
            }
        }

        let (lhs_index, rhs_index, cost) = best.ok_or(ContractError::NotEnoughTensors)?;
        let mut rhs = active.remove(rhs_index);
        let mut lhs = active.remove(lhs_index);
        if rhs.first_input < lhs.first_input {
            std::mem::swap(&mut lhs, &mut rhs);
        }
        let result_labels = declared_step_labels(
            &lhs.labels,
            &rhs.labels,
            active.is_empty(),
            ir.output_labels(),
        );
        let result_id = TensorId::new(ir.tensors().len() + steps.len());
        steps.push(ContractionStep::new(
            lhs.id,
            rhs.id,
            result_id,
            cost,
            result_labels.clone(),
        ));
        active.push(ActiveTensor {
            id: result_id,
            labels: result_labels,
            first_input: lhs.first_input.min(rhs.first_input),
        });
    }

    charge_dense_orientation_costs(ir, cost_model, &mut steps)?;
    Ok(steps)
}

/// Optimizer interface for dense-shape contraction plans.
///
/// Implement this trait to plug in a custom order search while reusing symtn's
/// parser, plan validation, and executors.
///
/// ```text
/// use symtn_network::*;
///
/// let ir = NetworkIR::from_labels(vec![vec!["a".into(), "b".into()], vec!["b".into(), "c".into()]], vec!["a".into(), "c".into()])?;
/// let infos = vec![
///     DenseTensorInfo::new(vec![2, 3]),
///     DenseTensorInfo::new(vec![3, 4]),
/// ];
/// let cost = DenseCostModel::from_network(&ir, &infos)?;
/// let steps = GreedyDenseOptimizer.optimize(&ir, &cost)?;
/// let plan = ContractionPlan::from_steps(&ir, steps)?;
/// assert_eq!(plan.active_pair_path()?, vec![ActivePair::new(0, 1)]);
/// ```
pub trait DenseContractionOptimizer {
    fn optimize(&self, ir: &NetworkIR, cost_model: &DenseCostModel)
        -> Result<Vec<ContractionStep>>;
}

/// Default dense greedy contraction optimizer.
///
/// This is cheap to run and deterministic. For hard networks, import an
/// external active-pair path (e.g. from the optional cotengra backend) and
/// build a [`crate::ContractionPlan`] from that path.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GreedyDenseOptimizer;

impl DenseContractionOptimizer for GreedyDenseOptimizer {
    fn optimize(
        &self,
        ir: &NetworkIR,
        cost_model: &DenseCostModel,
    ) -> Result<Vec<ContractionStep>> {
        greedy_order(ir, cost_model)
    }
}

/// Optimizer that contracts labels in a caller-provided order.
///
/// This implements explicit contraction-label ordering over [`NetworkIR`].
///
/// The order should contain only internal labels that appear exactly twice.
/// See crate-level design references for the index-notation and path-ordering
/// APIs that motivated this boundary.
///
/// ```text
/// use symtn_network::*;
///
/// let ir = NetworkIR::from_labels(vec![vec!["a".into(), "b".into()], vec!["b".into(), "c".into()], vec!["c".into(), "d".into()]], vec!["a".into(), "d".into()])?;
/// let infos = vec![
///     DenseTensorInfo::new(vec![2, 3]),
///     DenseTensorInfo::new(vec![3, 4]),
///     DenseTensorInfo::new(vec![4, 5]),
/// ];
/// let cost = DenseCostModel::from_network(&ir, &infos)?;
/// let optimizer = LabelOrderDenseOptimizer::new(vec![
///     TemporaryLabel::new("b"),
///     TemporaryLabel::new("c"),
/// ]);
/// let steps = DenseContractionOptimizer::optimize(&optimizer, &ir, &cost)?;
/// let plan = ContractionPlan::from_steps(&ir, steps)?;
/// assert_eq!(plan.active_pair_path()?.len(), 2);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabelOrderDenseOptimizer {
    order: Vec<TemporaryLabel>,
}

impl LabelOrderDenseOptimizer {
    /// Create an optimizer from an explicit label order.
    pub fn new(order: Vec<TemporaryLabel>) -> Self {
        Self { order }
    }

    /// Return the explicit label order.
    pub fn order(&self) -> &[TemporaryLabel] {
        &self.order
    }
}

impl DenseContractionOptimizer for LabelOrderDenseOptimizer {
    fn optimize(
        &self,
        ir: &NetworkIR,
        cost_model: &DenseCostModel,
    ) -> Result<Vec<ContractionStep>> {
        dense_order_from_labels(ir, cost_model, &self.order)
    }
}

/// Cost comparison between a dense plan and symtn's dense greedy baseline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DensePlanCostReport {
    plan_cost: usize,
    greedy_cost: usize,
}

impl DensePlanCostReport {
    /// Construct a report from precomputed cost totals.
    pub fn new(plan_cost: usize, greedy_cost: usize) -> Self {
        Self {
            plan_cost,
            greedy_cost,
        }
    }

    /// Total estimated cost of the inspected plan.
    pub fn plan_cost(&self) -> usize {
        self.plan_cost
    }

    /// Total estimated cost of the greedy baseline.
    pub fn greedy_cost(&self) -> usize {
        self.greedy_cost
    }

    /// Return true when the inspected plan is more expensive than greedy.
    pub fn is_suboptimal(&self) -> bool {
        self.plan_cost > self.greedy_cost
    }
}

/// Compare a supplied dense plan with the greedy baseline using saturating totals.
pub fn dense_plan_cost_report(
    steps: &[ContractionStep],
    ir: &NetworkIR,
    cost_model: &DenseCostModel,
) -> Result<DensePlanCostReport> {
    let plan_cost = steps
        .iter()
        .map(ContractionStep::cost)
        .fold(0, usize::saturating_add);
    let greedy_cost = greedy_order(ir, cost_model)?
        .iter()
        .map(ContractionStep::cost)
        .fold(0, usize::saturating_add);
    Ok(DensePlanCostReport::new(plan_cost, greedy_cost))
}

pub fn dense_order_from_labels(
    ir: &NetworkIR,
    cost_model: &DenseCostModel,
    order: &[TemporaryLabel],
) -> Result<Vec<ContractionStep>> {
    if ir.tensors().len() < 2 {
        return Err(ContractError::NotEnoughTensors);
    }

    validate_explicit_order_labels(ir, order)?;

    let mut active = ir
        .tensors()
        .iter()
        .enumerate()
        .map(|(first_input, tensor)| ActiveTensor {
            id: tensor.id(),
            labels: tensor.labels().to_vec(),
            first_input,
        })
        .collect::<Vec<_>>();
    let mut steps = Vec::new();

    let mut remaining_order = order.to_vec();
    while let Some(label) = remaining_order.first().cloned() {
        let Some(lhs_index) = active
            .iter()
            .position(|tensor| tensor.labels.contains(&label))
        else {
            remaining_order.remove(0);
            continue;
        };
        let Some(rhs_index) = active
            .iter()
            .enumerate()
            .skip(lhs_index + 1)
            .find_map(|(position, tensor)| tensor.labels.contains(&label).then_some(position))
        else {
            remaining_order.remove(0);
            continue;
        };

        let shared_order_labels = active[lhs_index]
            .labels
            .iter()
            .filter(|candidate| {
                active[rhs_index].labels.contains(candidate) && remaining_order.contains(candidate)
            })
            .cloned()
            .collect::<BTreeSet<_>>();
        remaining_order.retain(|candidate| !shared_order_labels.contains(candidate));
        push_dense_active_contraction_step(
            ir,
            cost_model,
            &mut active,
            &mut steps,
            lhs_index,
            rhs_index,
        )?;
    }

    while active.len() > 1 {
        let rhs_index = active.len() - 1;
        let lhs_index = active.len() - 2;
        push_dense_active_contraction_step(
            ir,
            cost_model,
            &mut active,
            &mut steps,
            lhs_index,
            rhs_index,
        )?;
    }

    charge_dense_orientation_costs(ir, cost_model, &mut steps)?;
    Ok(steps)
}

pub(crate) fn charge_dense_orientation_costs(
    ir: &NetworkIR,
    cost_model: &DenseCostModel,
    steps: &mut [ContractionStep],
) -> Result<()> {
    let planned_labels = planned_label_orders(ir, steps)?;
    // Resolve each result's single later consumer once (O(steps)); the topology
    // (lhs/rhs/result) is fixed even though this loop mutates step costs.
    let consumers = consumers(steps);
    let mut active = ir
        .tensors()
        .iter()
        .map(|tensor| (tensor.id(), tensor.labels().to_vec()))
        .collect::<HashMap<_, _>>();

    for step_index in 0..steps.len() {
        let lhs_id = steps[step_index].lhs;
        let rhs_id = steps[step_index].rhs;
        let result_id = steps[step_index].result;
        let lhs = active.remove(&lhs_id).ok_or_else(|| {
            ContractError::InvalidContractionPlan(format!(
                "step {step_index} lhs {} is not active",
                lhs_id.index()
            ))
        })?;
        let rhs = active.remove(&rhs_id).ok_or_else(|| {
            ContractError::InvalidContractionPlan(format!(
                "step {step_index} rhs {} is not active",
                rhs_id.index()
            ))
        })?;

        let raw_codomain_rank = lhs.iter().filter(|label| !rhs.contains(label)).count();
        let mut labels = pair_result_labels(&lhs, &rhs);

        if let Some(&(future_index, result_is_lhs)) = consumers.get(&result_id) {
            let sibling_id = if result_is_lhs {
                steps[future_index].rhs
            } else {
                steps[future_index].lhs
            };
            let sibling_labels = planned_labels.get(&sibling_id).ok_or_else(|| {
                ContractError::InvalidContractionPlan(format!(
                    "step {step_index} future sibling {} has no planned labels",
                    sibling_id.index()
                ))
            })?;
            let (needs_orientation, oriented_labels) = dense_orientation_for_next_use(
                &labels,
                raw_codomain_rank,
                result_is_lhs,
                sibling_labels,
            );
            if needs_orientation {
                steps[step_index].cost = steps[step_index]
                    .cost
                    .saturating_add(cost_model.tensor_size(&labels));
                labels = oriented_labels;
            }
        }

        active.insert(result_id, labels);
    }

    if let Some(final_labels) = active.values().next() {
        if !steps.is_empty() && final_labels.as_slice() != ir.output_labels() {
            let final_permute_cost = cost_model.tensor_size(final_labels);
            let last_step = steps.len() - 1;
            steps[last_step].cost = steps[last_step].cost.saturating_add(final_permute_cost);
        }
    }

    Ok(())
}

pub(crate) fn dense_orientation_for_next_use(
    labels: &[TemporaryLabel],
    raw_codomain_rank: usize,
    result_is_lhs: bool,
    sibling_labels: &[TemporaryLabel],
) -> (bool, Vec<TemporaryLabel>) {
    let (cod_axes, dom_axes) = next_use_axes(labels, result_is_lhs, sibling_labels);
    let rank = labels.len();
    let identity = cod_axes.iter().copied().eq(0..raw_codomain_rank)
        && dom_axes.iter().copied().eq(raw_codomain_rank..rank);
    if identity {
        return (false, labels.to_vec());
    }

    let mut oriented_labels = Vec::with_capacity(labels.len());
    oriented_labels.extend(cod_axes.iter().map(|&axis| labels[axis].clone()));
    oriented_labels.extend(dom_axes.iter().map(|&axis| labels[axis].clone()));
    (true, oriented_labels)
}

/// Codomain/domain axes of an intermediate for its consuming step: `(open |
/// contracted)` as lhs, `(contracted | open)` as rhs, with the contracted legs
/// in the lhs's order on both sides. TensorOperations 5.3.1
/// `instantiators.jl:instantiate_contraction` takes `cind = intersect(indA,
/// indB)` in A's order (L209) for both an lhs temporary `(oindA | cind)` (L217)
/// and an rhs temporary `(cind | oindB)` (L233), so the consuming step borrows
/// both operands. Ordering the rhs by its own labels instead would force that
/// step to copy one operand. `sibling_labels` is the planned order; an lhs
/// layout never reorders legs within a group, so the planned relative order of
/// its contracted legs is the order the consuming step enumerates.
pub(crate) fn next_use_axes(
    labels: &[TemporaryLabel],
    result_is_lhs: bool,
    sibling_labels: &[TemporaryLabel],
) -> (Vec<usize>, Vec<usize>) {
    let open_axes = (0..labels.len())
        .filter(|&axis| !sibling_labels.contains(&labels[axis]))
        .collect();
    if result_is_lhs {
        let contracted_axes = (0..labels.len())
            .filter(|&axis| sibling_labels.contains(&labels[axis]))
            .collect();
        (open_axes, contracted_axes)
    } else {
        let contracted_axes = sibling_labels
            .iter()
            .filter_map(|label| labels.iter().position(|own| own == label))
            .collect();
        (contracted_axes, open_axes)
    }
}

fn push_dense_active_contraction_step(
    ir: &NetworkIR,
    cost_model: &DenseCostModel,
    active: &mut Vec<ActiveTensor>,
    steps: &mut Vec<ContractionStep>,
    lhs_index: usize,
    rhs_index: usize,
) -> Result<()> {
    if lhs_index >= rhs_index || rhs_index >= active.len() {
        return Err(ContractError::InvalidContractionPlan(format!(
            "invalid active pair ({lhs_index}, {rhs_index}) for {} active tensors",
            active.len()
        )));
    }

    let rhs = active.remove(rhs_index);
    let lhs = active.remove(lhs_index);
    let result_labels = declared_step_labels(
        &lhs.labels,
        &rhs.labels,
        active.is_empty(),
        ir.output_labels(),
    );
    let cost = cost_model.pair_cost(&lhs.labels, &rhs.labels);
    let result_id = TensorId::new(ir.tensors().len() + steps.len());
    steps.push(ContractionStep::new(
        lhs.id,
        rhs.id,
        result_id,
        cost,
        result_labels.clone(),
    ));
    active.push(ActiveTensor {
        id: result_id,
        labels: result_labels,
        first_input: lhs.first_input.min(rhs.first_input),
    });
    Ok(())
}

fn validate_explicit_order_labels(ir: &NetworkIR, order: &[TemporaryLabel]) -> Result<()> {
    let mut order_set = BTreeSet::new();
    for label in order {
        if !order_set.insert(label.clone()) {
            return Err(ContractError::InvalidContractionPlan(format!(
                "explicit contraction order contains duplicate label `{label}`"
            )));
        }
    }

    for edge in ir.edges() {
        let label = edge.label();
        let occurrence_count = edge.occurrences().len();
        if order_set.contains(label) {
            if edge.is_output() {
                return Err(ContractError::InvalidContractionPlan(format!(
                    "explicit contraction order label `{label}` is an output label"
                )));
            }
            if occurrence_count != 2 {
                return Err(ContractError::InvalidContractionPlan(format!(
                    "explicit contraction order label `{label}` must occur exactly twice, got {occurrence_count}"
                )));
            }
        } else if edge.is_output() {
            if occurrence_count != 1 {
                return Err(ContractError::InvalidContractionPlan(format!(
                    "output label `{label}` must occur exactly once for explicit contraction order, got {occurrence_count}"
                )));
            }
        } else {
            return Err(ContractError::InvalidContractionPlan(format!(
                "contracted label `{label}` is absent from explicit contraction order"
            )));
        }
    }

    for label in order {
        if !ir.edges().iter().any(|edge| edge.label() == label) {
            return Err(ContractError::InvalidContractionPlan(format!(
                "explicit contraction order label `{label}` does not occur in inputs"
            )));
        }
    }
    Ok(())
}
