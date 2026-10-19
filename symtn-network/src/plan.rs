use std::collections::HashMap;

use crate::cost::DenseCostModel;
use crate::error::{ContractError, Result};
use crate::ir::NetworkIR;
use crate::labels::{TemporaryLabel, TensorId};
use crate::optimizer::{
    charge_dense_orientation_costs, dense_plan_cost_report, ContractionStep, DensePlanCostReport,
};
use crate::stepflow::{declared_step_labels, pair_result_labels};
use crate::tree::ContractionTree;

/// v2 writes each label as a quoted, escaped string (v1 wrote it bare).
const PLAN_HEADER: &str = "symtn-contract-plan-v2";

/// Append `label` as a quoted, escaped string (`{:?}`). `Network` labels are
/// arbitrary strings, so a bare label holding whitespace or a line break would
/// change the line structure of the plan text and of the plan-cache key.
pub(crate) fn push_label(text: &mut String, label: &str) {
    use std::fmt::Write;
    let _ = write!(text, "{label:?}");
}

/// Parse a sequence of ` <label>` written by [`push_label`].
pub(crate) fn parse_labels(mut rest: &str) -> Result<Vec<TemporaryLabel>> {
    let mut labels = Vec::new();
    while !rest.is_empty() {
        let quoted = rest
            .strip_prefix(" \"")
            .ok_or_else(|| invalid_serialized_plan("label is not a quoted string"))?;
        let (label, after) = parse_escaped_label(quoted)
            .ok_or_else(|| invalid_serialized_plan("invalid quoted label"))?;
        labels.push(TemporaryLabel::from(label.as_str()));
        rest = after;
    }
    Ok(labels)
}

/// Inverse of `str`'s `Debug` escaping, from just after the opening quote up
/// to and including the closing quote; returns the label and the remainder.
fn parse_escaped_label(text: &str) -> Option<(String, &str)> {
    let mut label = String::new();
    let mut chars = text.char_indices();
    while let Some((i, c)) = chars.next() {
        match c {
            '"' => return Some((label, &text[i + 1..])),
            '\\' => label.push(match chars.next()?.1 {
                't' => '\t',
                'r' => '\r',
                'n' => '\n',
                '0' => '\0',
                c @ ('\\' | '"' | '\'') => c,
                'u' => {
                    let start = chars.next().filter(|&(_, c)| c == '{')?.0 + 1;
                    let end = chars.find(|&(_, c)| c == '}')?.0;
                    char::from_u32(u32::from_str_radix(&text[start..end], 16).ok()?)?
                }
                _ => return None,
            }),
            c => label.push(c),
        }
    }
    None
}

/// A contraction step expressed as positions in the current active tensor list.
///
/// This is the external optimizer boundary used by active-list contraction
/// paths. After each step the two selected active tensors are removed and the
/// contraction result is appended to the active list.
///
/// Design references:
/// - opt_einsum path format:
///   <https://optimized-einsum.readthedocs.io/en/stable/path_finding.html#format-of-the-path>
/// - cotengra path provider:
///   <https://github.com/jcmgray/cotengra>
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
/// let path = vec![ActivePair::new(0, 1), ActivePair::new(0, 1)];
/// let steps = dense_steps_from_active_pair_path(&ir, &path, &cost)?;
/// let plan = ContractionPlan::from_steps(&ir, steps)?;
/// assert_eq!(plan.active_pair_path()?, path);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ActivePair {
    lhs_position: usize,
    rhs_position: usize,
}

/// Orient an external planner's unordered active pairs by written expression
/// order while preserving its contraction tree. External einsum planners use
/// pair positions only to select a tree edge; symtn additionally assigns
/// fermionic and orientation semantics to lhs/rhs. The public explicit path
/// constructor intentionally does not call this helper.
#[cfg(any(feature = "opt-path", feature = "cotengra-python"))]
pub(crate) fn orient_unordered_active_pairs(
    path: &[ActivePair],
    tensor_count: usize,
) -> Result<Vec<ActivePair>> {
    let mut active = (0..tensor_count).collect::<Vec<_>>();
    let mut oriented = Vec::with_capacity(path.len());
    for pair in path {
        let (mut lhs, mut rhs) = (pair.lhs_position(), pair.rhs_position());
        validate_active_pair(lhs, rhs, active.len())?;
        if active[rhs] < active[lhs] {
            std::mem::swap(&mut lhs, &mut rhs);
        }
        let first_input = active[lhs].min(active[rhs]);
        remove_pair(&mut active, lhs, rhs);
        active.push(first_input);
        oriented.push(ActivePair::new(lhs, rhs));
    }
    Ok(oriented)
}

impl ActivePair {
    /// Construct a pair of active tensor positions.
    pub const fn new(lhs_position: usize, rhs_position: usize) -> Self {
        Self {
            lhs_position,
            rhs_position,
        }
    }

    /// Position of the left operand in the active tensor list before this step.
    pub fn lhs_position(self) -> usize {
        self.lhs_position
    }

    /// Position of the right operand in the active tensor list before this step.
    pub fn rhs_position(self) -> usize {
        self.rhs_position
    }
}

/// Reusable pairwise contraction plan.
///
/// A plan stores tensor count, output labels, and a validated list of pairwise
/// contraction steps. It is independent of tensor values, but it assumes the same
/// expression structure and compatible dimensions when executed.
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
/// let path = vec![ActivePair::new(0, 1)];
/// let steps = dense_steps_from_active_pair_path(&ir, &path, &cost)?;
/// let plan = ContractionPlan::from_steps(&ir, steps)?;
/// let text = plan.to_text();
/// let restored = ContractionPlan::from_text(&text)?;
/// assert_eq!(plan.active_pair_path()?, restored.active_pair_path()?);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContractionPlan {
    tensor_count: usize,
    output_labels: Vec<TemporaryLabel>,
    steps: Vec<ContractionStep>,
}

impl ContractionPlan {
    /// Construct a plan for an already parsed network and its contraction steps.
    ///
    /// In addition to the topology checks in `validate`, this
    /// path knows each input tensor's leg labels (from `ir`) and therefore
    /// additionally verifies that every step's declared `result_labels` equals
    /// what contracting its two operands actually produces. See
    /// [`validate_step_result_labels`](Self::validate_step_result_labels).
    pub fn from_steps(ir: &NetworkIR, steps: Vec<ContractionStep>) -> Result<Self> {
        let plan = Self {
            tensor_count: ir.tensors().len(),
            output_labels: ir.output_labels().to_vec(),
            steps,
        };
        plan.validate()?;
        let input_labels: Vec<Vec<TemporaryLabel>> =
            ir.tensors().iter().map(|t| t.labels().to_vec()).collect();
        plan.validate_step_result_labels(&input_labels)?;
        Ok(plan)
    }

    /// Number of original input tensors expected by this plan.
    pub fn tensor_count(&self) -> usize {
        self.tensor_count
    }

    /// Labels of the final output tensor.
    pub fn output_labels(&self) -> &[TemporaryLabel] {
        &self.output_labels
    }

    /// Pairwise contraction steps in execution order.
    pub fn steps(&self) -> &[ContractionStep] {
        &self.steps
    }

    /// Convert this plan into a contraction tree.
    pub fn tree(&self) -> Result<ContractionTree> {
        ContractionTree::from_steps(self.tensor_count, &self.steps)
    }

    /// Return this plan as active-pair positions.
    ///
    /// This is the inverse representation consumed by
    /// [`dense_steps_from_active_pair_path`].
    pub fn active_pair_path(&self) -> Result<Vec<ActivePair>> {
        active_pair_path_from_steps(self.tensor_count, &self.steps)
    }

    /// Saturating sum of the optimizer cost estimates stored on all steps.
    pub fn total_cost(&self) -> usize {
        self.steps
            .iter()
            .map(ContractionStep::cost)
            .fold(0, usize::saturating_add)
    }

    /// Compare this dense plan against symtn's greedy dense baseline.
    pub fn dense_cost_report(
        &self,
        ir: &NetworkIR,
        cost_model: &DenseCostModel,
    ) -> Result<DensePlanCostReport> {
        if self.tensor_count != ir.tensors().len() {
            return Err(ContractError::InvalidContractionPlan(format!(
                "plan tensor count {} does not match network tensor count {}",
                self.tensor_count,
                ir.tensors().len()
            )));
        }
        if self.output_labels != ir.output_labels() {
            return Err(ContractError::InvalidContractionPlan(
                "plan output labels do not match network output labels".to_string(),
            ));
        }
        dense_plan_cost_report(&self.steps, ir, cost_model)
    }

    /// Serialize the plan to a compact text format.
    ///
    /// The text is intended for cache files controlled by the application. It is
    /// versioned with a header and validated on restore.
    pub fn to_text(&self) -> String {
        let mut text = String::new();
        text.push_str(PLAN_HEADER);
        text.push('\n');
        text.push_str("tensor_count ");
        text.push_str(&self.tensor_count.to_string());
        text.push('\n');
        text.push_str("output");
        for label in &self.output_labels {
            text.push(' ');
            push_label(&mut text, label.as_str());
        }
        text.push('\n');
        for step in &self.steps {
            text.push_str("step ");
            text.push_str(&step.lhs().index().to_string());
            text.push(' ');
            text.push_str(&step.rhs().index().to_string());
            text.push(' ');
            text.push_str(&step.result().index().to_string());
            text.push(' ');
            text.push_str(&step.cost().to_string());
            for label in step.result_labels() {
                text.push(' ');
                push_label(&mut text, label.as_str());
            }
            text.push('\n');
        }
        text
    }

    /// Restore a plan serialized by [`to_text`](Self::to_text).
    pub fn from_text(text: &str) -> Result<Self> {
        let mut lines = text.lines();
        let header = lines
            .next()
            .ok_or_else(|| invalid_serialized_plan("missing header"))?;
        if header != PLAN_HEADER {
            return Err(invalid_serialized_plan("unsupported plan header"));
        }

        let tensor_count_line = lines
            .next()
            .ok_or_else(|| invalid_serialized_plan("missing tensor_count line"))?;
        let tensor_count = parse_tensor_count(tensor_count_line)?;

        let output_line = lines
            .next()
            .ok_or_else(|| invalid_serialized_plan("missing output line"))?;
        let output_labels = parse_output_labels(output_line)?;

        let mut steps = Vec::new();
        for line in lines {
            if line.trim().is_empty() {
                continue;
            }
            steps.push(parse_step(line)?);
        }

        let plan = Self {
            tensor_count,
            output_labels,
            steps,
        };
        plan.validate()?;
        Ok(plan)
    }

    fn validate(&self) -> Result<()> {
        if self.tensor_count == 0 {
            return Err(ContractError::NotEnoughTensors);
        }
        if self.tensor_count > 1 && self.steps.len() + 1 != self.tensor_count {
            return Err(ContractError::InvalidContractionPlan(format!(
                "complete pairwise plan for {} tensors needs {} steps, got {}",
                self.tensor_count,
                self.tensor_count - 1,
                self.steps.len()
            )));
        }
        ContractionTree::from_steps(self.tensor_count, &self.steps)?;
        if let Some(last) = self.steps.last() {
            if last.result_labels() != self.output_labels() {
                return Err(ContractError::InvalidContractionPlan(
                    "final step labels do not match plan output labels".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// Verify that every step's declared `result_labels` is *mathematically*
    /// consistent with contracting its two operands.
    ///
    /// `validate` only checks topology (operand ids in range,
    /// single final result, …); it trusts the labels each step claims to emit.
    /// The schedule compiler (`network/schedule.rs`, `compile_schedule`),
    /// however, derives the actual result leg order purely from the two
    /// operands: a label shared by *both*
    /// operands is contracted away; every other label (an *open* leg) survives,
    /// in lhs-then-rhs order. A hand-built plan that declares a different
    /// `result_labels` would silently compute the wrong network.
    ///
    /// This mirrors that derivation (the `result_labels` block there) and
    /// rejects any step whose declared labels disagree. `input_labels` supplies
    /// the leg labels of each original input tensor (id `0..tensor_count`);
    /// intermediate operands' labels are looked up from earlier steps.
    pub fn validate_step_result_labels(&self, input_labels: &[Vec<TemporaryLabel>]) -> Result<()> {
        if input_labels.len() != self.tensor_count {
            return Err(ContractError::InvalidContractionPlan(format!(
                "expected {} input label sets, got {}",
                self.tensor_count,
                input_labels.len()
            )));
        }

        // Labels currently carried by each live tensor id (inputs + intermediates).
        let mut labels_of: HashMap<TensorId, Vec<TemporaryLabel>> = HashMap::new();
        for (i, labels) in input_labels.iter().enumerate() {
            labels_of.insert(TensorId::new(i), labels.clone());
        }

        for (step_index, step) in self.steps.iter().enumerate() {
            let ll = labels_of.get(&step.lhs()).ok_or_else(|| {
                ContractError::InvalidContractionPlan(format!(
                    "ContractionStep {step_index}: lhs operand {} has no known labels",
                    step.lhs().index()
                ))
            })?;
            let rl = labels_of.get(&step.rhs()).ok_or_else(|| {
                ContractError::InvalidContractionPlan(format!(
                    "ContractionStep {step_index}: rhs operand {} has no known labels",
                    step.rhs().index()
                ))
            })?;

            // Mirror execute_plan: a label shared by both operands is contracted
            // away; every other (open) leg survives, in lhs-then-rhs flat order.
            // This `expected` order is what the executor itself tracks for this
            // result, so it (not the declared order) feeds later steps.
            let expected = pair_result_labels(ll, rl);

            // The declared `result_labels` must carry exactly the open legs (as a
            // multiset). The executor recomputes the leg ORDER from the operands
            // and ends with a final `permute` to `output_labels`, so a permuted
            // *intermediate* order is harmless and the optimizers legitimately
            // emit the final step in `output_labels` order — comparing as
            // multisets accepts those yet still rejects any step that keeps a
            // contracted label, drops an open leg, or invents a spurious one.
            if !same_label_multiset(step.result_labels(), &expected) {
                return Err(ContractError::InvalidContractionPlan(format!(
                    "ContractionStep {step_index}: declared result_labels {:?} \u{2260} the open legs {:?} of operands {:?} and {:?}",
                    labels_as_strings(step.result_labels()),
                    labels_as_strings(&expected),
                    labels_as_strings(ll),
                    labels_as_strings(rl),
                )));
            }

            labels_of.insert(step.result(), expected);
        }

        Ok(())
    }
}

fn labels_as_strings(labels: &[TemporaryLabel]) -> Vec<&str> {
    labels.iter().map(TemporaryLabel::as_str).collect()
}

/// Whether two label lists are equal as multisets (same labels with the same
/// multiplicities, independent of order).
fn same_label_multiset(a: &[TemporaryLabel], b: &[TemporaryLabel]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut counts: HashMap<&TemporaryLabel, isize> = HashMap::new();
    for label in a {
        *counts.entry(label).or_insert(0) += 1;
    }
    for label in b {
        match counts.get_mut(label) {
            Some(count) => *count -= 1,
            None => return false,
        }
    }
    counts.values().all(|&c| c == 0)
}

pub fn active_pair_path_from_steps(
    tensor_count: usize,
    steps: &[ContractionStep],
) -> Result<Vec<ActivePair>> {
    let mut active = (0..tensor_count).map(TensorId::new).collect::<Vec<_>>();
    let mut path = Vec::with_capacity(steps.len());

    for step in steps {
        let lhs_position = position_of(&active, step.lhs())?;
        let rhs_position = position_of(&active, step.rhs())?;
        validate_active_pair(lhs_position, rhs_position, active.len())?;
        path.push(ActivePair::new(lhs_position, rhs_position));
        remove_pair_and_push_result(&mut active, lhs_position, rhs_position, step.result());
    }

    Ok(path)
}

/// Convert a contraction tree into active-list pair positions.
pub fn active_pair_path_from_tree(
    tensor_count: usize,
    tree: &ContractionTree,
) -> Result<Vec<ActivePair>> {
    let mut active = (0..tensor_count).map(TensorId::new).collect::<Vec<_>>();
    let mut path = Vec::with_capacity(tensor_count.saturating_sub(1));
    emit_tree_active_pairs(tree, tensor_count, &mut active, &mut path)?;
    if active.len() != 1 {
        return Err(ContractError::InvalidContractionPlan(format!(
            "tree leaves {} active tensors",
            active.len()
        )));
    }
    Ok(path)
}

fn emit_tree_active_pairs(
    tree: &ContractionTree,
    tensor_count: usize,
    active: &mut Vec<TensorId>,
    path: &mut Vec<ActivePair>,
) -> Result<TensorId> {
    match tree {
        ContractionTree::Leaf { tensor } => {
            if tensor.index() >= tensor_count {
                return Err(ContractError::InvalidTensorId {
                    tensor: tensor.index(),
                    tensor_count,
                });
            }
            if !active.contains(tensor) {
                return Err(ContractError::InvalidContractionPlan(format!(
                    "tree leaf tensor {} is not active",
                    tensor.index()
                )));
            }
            Ok(*tensor)
        }
        ContractionTree::Pair { lhs, rhs, .. } => {
            let lhs_id = emit_tree_active_pairs(lhs, tensor_count, active, path)?;
            let rhs_id = emit_tree_active_pairs(rhs, tensor_count, active, path)?;
            let lhs_position = position_of(active, lhs_id)?;
            let rhs_position = position_of(active, rhs_id)?;
            validate_active_pair(lhs_position, rhs_position, active.len())?;
            path.push(ActivePair::new(lhs_position, rhs_position));
            let result = TensorId::new(tensor_count + path.len() - 1);
            remove_pair_and_push_result(active, lhs_position, rhs_position, result);
            Ok(result)
        }
    }
}

pub fn dense_steps_from_active_pair_path(
    ir: &NetworkIR,
    path: &[ActivePair],
    cost_model: &DenseCostModel,
) -> Result<Vec<ContractionStep>> {
    let mut active = ir
        .tensors()
        .iter()
        .map(|tensor| ActiveTensorForPath {
            id: tensor.id(),
            labels: tensor.labels().to_vec(),
        })
        .collect::<Vec<_>>();
    let mut steps = Vec::with_capacity(path.len());

    for pair in path {
        validate_active_pair(pair.lhs_position(), pair.rhs_position(), active.len())?;
        let lhs = active[pair.lhs_position()].clone();
        let rhs = active[pair.rhs_position()].clone();
        remove_pair(&mut active, pair.lhs_position(), pair.rhs_position());

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
        active.push(ActiveTensorForPath {
            id: result_id,
            labels: result_labels,
        });
    }

    if active.len() != 1 {
        return Err(ContractError::InvalidContractionPlan(format!(
            "active-pair path leaves {} active tensors",
            active.len()
        )));
    }

    charge_dense_orientation_costs(ir, cost_model, &mut steps)?;
    Ok(steps)
}

#[derive(Debug, Clone)]
struct ActiveTensorForPath {
    id: TensorId,
    labels: Vec<TemporaryLabel>,
}

fn position_of(active: &[TensorId], id: TensorId) -> Result<usize> {
    active
        .iter()
        .position(|&candidate| candidate == id)
        .ok_or_else(|| {
            ContractError::InvalidContractionPlan(format!(
                "tensor {} is not active in this step",
                id.index()
            ))
        })
}

fn validate_active_pair(lhs_position: usize, rhs_position: usize, active_len: usize) -> Result<()> {
    if lhs_position == rhs_position {
        return Err(ContractError::InvalidContractionPlan(format!(
            "active pair uses the same position {lhs_position} twice"
        )));
    }
    if lhs_position >= active_len || rhs_position >= active_len {
        return Err(ContractError::InvalidContractionPlan(format!(
            "active pair ({lhs_position}, {rhs_position}) is out of range for {active_len} active tensors"
        )));
    }
    Ok(())
}

fn remove_pair_and_push_result(
    active: &mut Vec<TensorId>,
    lhs_position: usize,
    rhs_position: usize,
    result: TensorId,
) {
    remove_pair(active, lhs_position, rhs_position);
    active.push(result);
}

fn remove_pair<T>(active: &mut Vec<T>, lhs_position: usize, rhs_position: usize) {
    let high = lhs_position.max(rhs_position);
    let low = lhs_position.min(rhs_position);
    active.remove(high);
    active.remove(low);
}

fn parse_tensor_count(line: &str) -> Result<usize> {
    let parts = line.split_whitespace().collect::<Vec<_>>();
    if parts.len() != 2 || parts[0] != "tensor_count" {
        return Err(invalid_serialized_plan("invalid tensor_count line"));
    }
    parts[1]
        .parse::<usize>()
        .map_err(|_| invalid_serialized_plan("invalid tensor_count value"))
}

fn parse_output_labels(line: &str) -> Result<Vec<TemporaryLabel>> {
    match line.strip_prefix("output") {
        Some(rest) if rest.is_empty() || rest.starts_with(' ') => parse_labels(rest),
        _ => Err(invalid_serialized_plan("invalid output line")),
    }
}

fn parse_step(line: &str) -> Result<ContractionStep> {
    // Labels may contain spaces, so only the fixed fields are split off.
    let parts = line.splitn(5, ' ').collect::<Vec<_>>();
    if parts.len() < 5 || parts[0] != "step" {
        return Err(invalid_serialized_plan("invalid step line"));
    }
    let lhs = parse_tensor_id(parts[1], "lhs")?;
    let rhs = parse_tensor_id(parts[2], "rhs")?;
    let result = parse_tensor_id(parts[3], "result")?;
    let (cost, labels) = parts[4].split_at(parts[4].find(' ').unwrap_or(parts[4].len()));
    let cost = cost
        .parse::<usize>()
        .map_err(|_| invalid_serialized_plan("invalid step cost"))?;
    let result_labels = parse_labels(labels)?;
    Ok(ContractionStep::new(lhs, rhs, result, cost, result_labels))
}

fn parse_tensor_id(text: &str, field: &str) -> Result<TensorId> {
    text.parse::<usize>()
        .map(TensorId::new)
        .map_err(|_| invalid_serialized_plan(format!("invalid {field} tensor id")))
}

fn invalid_serialized_plan(message: impl Into<String>) -> ContractError {
    ContractError::InvalidContractionPlan(format!("serialized plan: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::DenseTensorInfo;
    use crate::ir::test_ir;
    use crate::optimizer::{DenseContractionOptimizer, GreedyDenseOptimizer};

    fn label(s: &str) -> TemporaryLabel {
        TemporaryLabel::new(s)
    }

    #[test]
    fn dense_active_pair_path_charges_intermediate_orientation_cost() {
        let ir = test_ir(&[&["x", "y"], &["x", "z"], &["y", "w"]], &["z", "w"]);
        let infos = vec![
            DenseTensorInfo::new(vec![2, 3]),
            DenseTensorInfo::new(vec![2, 5]),
            DenseTensorInfo::new(vec![3, 7]),
        ];
        let cost = DenseCostModel::from_network(&ir, &infos).unwrap();
        let steps = dense_steps_from_active_pair_path(
            &ir,
            &[ActivePair::new(0, 1), ActivePair::new(1, 0)],
            &cost,
        )
        .unwrap();
        let plan = ContractionPlan::from_steps(&ir, steps).unwrap();

        // Step 0 contracts xy with xz: contraction work 2*3*5 = 30.
        // Its raw result is y,z but the next parent needs it as lhs with y
        // contracted against yw, so execution materializes z <- y; add 3*5 = 15.
        assert_eq!(plan.steps()[0].cost(), 45);
        assert_eq!(plan.steps()[1].cost(), 105);
        assert_eq!(plan.total_cost(), 150);
    }

    #[test]
    fn dense_plan_charges_final_output_permute_cost() {
        let ir = test_ir(&[&["a", "b"], &["b", "c"]], &["c", "a"]);
        let infos = vec![
            DenseTensorInfo::new(vec![2, 3]),
            DenseTensorInfo::new(vec![3, 5]),
        ];
        let cost = DenseCostModel::from_network(&ir, &infos).unwrap();
        let plan =
            ContractionPlan::from_steps(&ir, GreedyDenseOptimizer.optimize(&ir, &cost).unwrap())
                .unwrap();

        // Pair contraction work is a*b*c = 30. The raw tensor order is a,c, but
        // the requested output is c,a; charge one final 2*5 element permute.
        assert_eq!(plan.steps()[0].cost(), 40);
        assert_eq!(plan.total_cost(), 40);
    }

    fn chain_steps(first_cost: usize, second_cost: usize) -> Vec<ContractionStep> {
        vec![
            ContractionStep::new(
                TensorId::new(0),
                TensorId::new(1),
                TensorId::new(3),
                first_cost,
                vec![label("a"), label("c")],
            ),
            ContractionStep::new(
                TensorId::new(3),
                TensorId::new(2),
                TensorId::new(4),
                second_cost,
                vec![label("a"), label("d")],
            ),
        ]
    }

    #[test]
    fn aggregate_costs_saturate_and_preserve_exact_totals() {
        let ir = test_ir(&[&["a", "b"], &["b", "c"], &["c", "d"]], &["a", "d"]);
        let saturated = ContractionPlan::from_steps(&ir, chain_steps(usize::MAX, 1)).unwrap();
        assert_eq!(saturated.total_cost(), usize::MAX);
        assert_eq!(saturated.tree().unwrap().total_cost(), usize::MAX);

        let boundary = ContractionPlan::from_steps(&ir, chain_steps(usize::MAX - 1, 1)).unwrap();
        assert_eq!(boundary.total_cost(), usize::MAX);
        assert_eq!(boundary.tree().unwrap().total_cost(), usize::MAX);

        let ordinary = ContractionPlan::from_steps(&ir, chain_steps(20, 22)).unwrap();
        assert_eq!(ordinary.total_cost(), 42);
        assert_eq!(ordinary.tree().unwrap().total_cost(), 42);

        let leaf = ContractionPlan::from_steps(&test_ir(&[&["a"]], &["a"]), Vec::new()).unwrap();
        assert_eq!(leaf.total_cost(), 0);
        assert_eq!(leaf.tree().unwrap().total_cost(), 0);
    }

    #[test]
    fn dense_cost_report_saturates_supplied_and_greedy_totals() {
        let ir = test_ir(&[&["a", "b"], &["b", "c"], &["c", "d"]], &["a", "d"]);
        let small = DenseCostModel::from_network(
            &ir,
            &[
                DenseTensorInfo::new(vec![2, 2]),
                DenseTensorInfo::new(vec![2, 2]),
                DenseTensorInfo::new(vec![2, 2]),
            ],
        )
        .unwrap();
        let supplied = ContractionPlan::from_steps(&ir, chain_steps(usize::MAX, 1)).unwrap();
        let report = supplied.dense_cost_report(&ir, &small).unwrap();
        assert_eq!(report.plan_cost(), usize::MAX);
        assert!(report.greedy_cost() < usize::MAX);
        assert!(report.is_suboptimal());

        let overflow = DenseCostModel::from_network(
            &ir,
            &[
                DenseTensorInfo::new(vec![usize::MAX, 2]),
                DenseTensorInfo::new(vec![2, 2]),
                DenseTensorInfo::new(vec![2, usize::MAX]),
            ],
        )
        .unwrap();
        let cheap = ContractionPlan::from_steps(&ir, chain_steps(1, 2)).unwrap();
        let report = cheap.dense_cost_report(&ir, &overflow).unwrap();
        assert_eq!(report.plan_cost(), 3);
        assert_eq!(report.greedy_cost(), usize::MAX);
        assert!(!report.is_suboptimal());
    }

    /// A permuted *intermediate* order is harmless (executor re-derives order);
    /// the multiset of legs is what matters.
    #[test]
    fn hand_built_intermediate_permuted_order_validates() {
        // ab,bc,cd->ad : first contract tensors 1,2 (b? no) -> contract pair
        // (0,1) sharing b gives open {a,c}; declare it permuted as [c, a].
        let ir = test_ir(&[&["a", "b"], &["b", "c"], &["c", "d"]], &["a", "d"]);
        let steps = vec![
            ContractionStep::new(
                TensorId::new(0),
                TensorId::new(1),
                TensorId::new(3),
                0,
                vec![label("c"), label("a")], // permuted open legs {a,c}
            ),
            ContractionStep::new(
                TensorId::new(3),
                TensorId::new(2),
                TensorId::new(4),
                0,
                vec![label("a"), label("d")], // final step in output order
            ),
        ];
        let plan = ContractionPlan::from_steps(&ir, steps);
        assert!(
            plan.is_ok(),
            "permuted intermediate order must validate: {plan:?}"
        );
    }

    /// A hand-built plan whose `result_labels` are mathematically wrong on an
    /// *intermediate* (non-final) step — here it keeps the contracted label `b`
    /// and drops the open leg `c` — must be rejected by the new label check.
    /// (The existing final-step `== output_labels` check would not catch this,
    /// since it only constrains the last step, so this exercises the new logic.)
    #[test]
    fn hand_built_plan_with_wrong_result_labels_is_rejected() {
        // ab,bc,cd->ad : first step contracts (0,1) sharing `b`; open legs {a,c}.
        let ir = test_ir(&[&["a", "b"], &["b", "c"], &["c", "d"]], &["a", "d"]);
        let steps = vec![
            ContractionStep::new(
                TensorId::new(0),
                TensorId::new(1),
                TensorId::new(3),
                0,
                // WRONG: `b` is contracted away and `c` is open; declaring
                // [a, b] keeps the contracted leg and omits an open one.
                vec![label("a"), label("b")],
            ),
            ContractionStep::new(
                TensorId::new(3),
                TensorId::new(2),
                TensorId::new(4),
                0,
                vec![label("a"), label("d")],
            ),
        ];
        let err = ContractionPlan::from_steps(&ir, steps)
            .expect_err("wrong result_labels must be rejected");
        match err {
            ContractError::InvalidContractionPlan(msg) => {
                assert!(
                    msg.contains("ContractionStep 0") && msg.contains("result_labels"),
                    "unexpected message: {msg}"
                );
            }
            other => panic!("expected InvalidContractionPlan, got {other:?}"),
        }
    }

    /// A spurious extra label that is on neither operand is rejected too.
    #[test]
    fn hand_built_plan_with_extra_label_is_rejected() {
        let ir = test_ir(&[&["a", "b"], &["b", "c"], &["c", "d"]], &["a", "d"]);
        let steps = vec![
            ContractionStep::new(
                TensorId::new(0),
                TensorId::new(1),
                TensorId::new(3),
                0,
                vec![label("a"), label("c"), label("z")], // `z` is invented
            ),
            ContractionStep::new(
                TensorId::new(3),
                TensorId::new(2),
                TensorId::new(4),
                0,
                vec![label("a"), label("d")],
            ),
        ];
        assert!(matches!(
            ContractionPlan::from_steps(&ir, steps),
            Err(ContractError::InvalidContractionPlan(_))
        ));
    }
}
