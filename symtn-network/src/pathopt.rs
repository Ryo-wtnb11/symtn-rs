//! Contraction-path optimization via the `opt-einsum-path` crate (Apache-2.0).
//!
//! [`OptEinsumPathOptimizer`] wraps `opt_einsum_path::contract_path` as a
//! [`DenseContractionOptimizer`], so symtn's plan validation and the slicer
//! ([`crate::greedy_slice`]) reuse it unchanged. It gives the optimal /
//! dynamic-programming / branch-and-bound /
//! random-greedy / auto drivers on top of symtn's built-in greedy.
//!
//! `opt_einsum_path` takes a single-`&str` einsum equation, so symtn's arbitrary
//! string labels ([`TemporaryLabel`]) are first remapped to unique single
//! characters (in deterministic first-seen order). For symmetric tensors pass a
//! [`DenseCostModel`] built from each bond's effective (total) dimension.
//!
//! The optional cotengra backend (the `cotengra-python` feature) implements the
//! same [`DenseContractionOptimizer`] trait and drops into the same pipeline;
//! there is no separate Rust `symtn-cotengrust` port — cotengrust is AGPL, and
//! the Python backend already covers the hyper / reconfigure / slicing search.

use std::borrow::Cow;
use std::collections::BTreeMap;

use crate::cost::DenseCostModel;
use crate::error::{ContractError, Result};
use crate::ir::NetworkIR;
use crate::labels::TemporaryLabel;
use crate::optimizer::{ContractionStep, DenseContractionOptimizer};
use crate::plan::{dense_steps_from_active_pair_path, orient_unordered_active_pairs, ActivePair};

/// einsum equation + per-tensor shapes in `opt_einsum_path` form, plus the
/// label→symbol map used to build them (kept for debugging/round-tripping).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OptEinsumInputs {
    /// einsum equation, e.g. `"ĀāÁ,āÂ->ĀÂ"` after remapping.
    pub subscripts: String,
    /// Shapes per input tensor, in tensor order.
    pub shapes: Vec<Vec<usize>>,
    /// Distinct symtn label -> assigned single-char symbol.
    pub symbols: BTreeMap<TemporaryLabel, char>,
}

/// Map distinct labels to unique single chars starting at U+0100 (Latin
/// Extended-A): all letters, no collision with einsum separators (`,`/`-`/`>`)
/// or whitespace, and well clear of the math operators opt_einsum reserves.
///
/// The UTF-16 surrogate block (U+D800..=U+DFFF) is skipped so every index maps
/// to a valid Unicode scalar value; the mapping stays injective up to ~1.1M
/// distinct labels (the single-`char` einsum ceiling), far beyond any realistic
/// network.
fn symbol_for(index: usize) -> Option<char> {
    let mut cp = 0x100u32.checked_add(u32::try_from(index).ok()?)?;
    if cp >= 0xD800 {
        cp = cp.checked_add(0x800)?; // jump over the surrogate gap
    }
    char::from_u32(cp)
}

/// Build the `opt_einsum_path` equation and shapes from a network + cost model.
///
/// Distinct labels are numbered in first-seen order (inputs in tensor order,
/// then any output-only labels). Returns an error only via the dimension lookup
/// being absent, which is treated as dimension 1 by [`DenseCostModel::dim`].
pub(crate) fn build_opt_einsum_inputs(
    ir: &NetworkIR,
    cost_model: &DenseCostModel,
) -> Result<OptEinsumInputs> {
    let mut symbols: BTreeMap<TemporaryLabel, char> = BTreeMap::new();
    let mut next_index = 0usize;
    let mut assign = |label: &TemporaryLabel,
                      symbols: &mut BTreeMap<TemporaryLabel, char>|
     -> Result<()> {
        if !symbols.contains_key(label) {
            let symbol = symbol_for(next_index).ok_or_else(|| {
                ContractError::InvalidContractionPlan(
                    "too many distinct indices to map to single-char einsum symbols".to_string(),
                )
            })?;
            symbols.insert(label.clone(), symbol);
            next_index += 1;
        }
        Ok(())
    };

    for tensor in ir.tensors() {
        for label in tensor.labels() {
            assign(label, &mut symbols)?;
        }
    }
    for label in ir.output_labels() {
        assign(label, &mut symbols)?;
    }

    let input_groups = ir
        .tensors()
        .iter()
        .map(|tensor| {
            tensor
                .labels()
                .iter()
                .map(|label| symbols[label])
                .collect::<String>()
        })
        .collect::<Vec<_>>();
    let output_group = ir
        .output_labels()
        .iter()
        .map(|label| symbols[label])
        .collect::<String>();
    let subscripts = format!("{}->{}", input_groups.join(","), output_group);

    let shapes = ir
        .tensors()
        .iter()
        .map(|tensor| {
            tensor
                .labels()
                .iter()
                .map(|label| cost_model.dim(label).unwrap_or(1))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    Ok(OptEinsumInputs {
        subscripts,
        shapes,
        symbols,
    })
}

/// Branch-search level for `opt_einsum_path`'s branch-and-bound driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BranchLevel {
    /// Try only the best first branch at each level (`"branch-1"`).
    One,
    /// Try the best two branches at each level (`"branch-2"`).
    Two,
    /// Full branch-and-bound search (`"branch-all"`).
    All,
}

impl BranchLevel {
    fn as_opt_einsum_str(self) -> &'static str {
        match self {
            BranchLevel::One => "branch-1",
            BranchLevel::Two => "branch-2",
            BranchLevel::All => "branch-all",
        }
    }
}

/// Dynamic-programming objective for `opt_einsum_path`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DpObjective {
    /// Minimize FLOPs (`"dp-flops"`).
    Flops,
    /// Minimize largest intermediate size (`"dp-size"`).
    Size,
    /// Minimize total write size (`"dp-write"`).
    Write,
    /// Combined FLOPs / size objective (`"dp-combo"`).
    Combo,
    /// Memory-limit-oriented combined objective (`"dp-limit"`).
    Limit,
    /// Combined objective with an explicit combo factor (`"dp-combo-N"`).
    ComboFactor(usize),
    /// Limit objective with an explicit combo factor (`"dp-limit-N"`).
    LimitFactor(usize),
}

impl DpObjective {
    fn as_opt_einsum_str(self) -> Cow<'static, str> {
        match self {
            DpObjective::Flops => Cow::Borrowed("dp-flops"),
            DpObjective::Size => Cow::Borrowed("dp-size"),
            DpObjective::Write => Cow::Borrowed("dp-write"),
            DpObjective::Combo => Cow::Borrowed("dp-combo"),
            DpObjective::Limit => Cow::Borrowed("dp-limit"),
            DpObjective::ComboFactor(factor) => Cow::Owned(format!("dp-combo-{}", factor.max(1))),
            DpObjective::LimitFactor(factor) => Cow::Owned(format!("dp-limit-{}", factor.max(1))),
        }
    }
}

/// Memory limit passed through to `opt_einsum_path::contract_path`.
///
/// The value is in elements, matching `opt_einsum_path`'s own `SizeLimitType`.
/// symtn's slicers remain the preferred way to force executable peak memory,
/// while this option lets the upstream path search avoid paths with large
/// intermediates in the first place.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PathMemoryLimit {
    /// No memory limit.
    #[default]
    None,
    /// Limit to the largest input/output tensor size (`"max-input"`).
    MaxInput,
    /// Limit intermediates to this many elements.
    Size(usize),
}

impl PathMemoryLimit {
    fn as_size_limit_type(self) -> opt_einsum_path::typing::SizeLimitType {
        match self {
            PathMemoryLimit::None => opt_einsum_path::typing::SizeLimitType::None,
            PathMemoryLimit::MaxInput => opt_einsum_path::typing::SizeLimitType::MaxInput,
            PathMemoryLimit::Size(size) => {
                opt_einsum_path::typing::SizeLimitType::Size(size as f64)
            }
        }
    }
}

/// Which `opt_einsum_path` driver to run.
///
/// Each maps to a known-good `opt_einsum_path` strategy string. (Passing an
/// unknown string to `opt_einsum_path` panics, so the public API is a typed enum
/// rather than a raw string.)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PathStrategy {
    /// Pick a driver automatically by network size (opt_einsum `"auto"`).
    Auto,
    /// Higher-quality automatic search (opt_einsum `"auto-hq"`). This is symtn's default:
    /// small networks use exhaustive optimal search, medium networks use dynamic programming,
    /// and larger networks use randomized greedy.
    #[default]
    AutoHq,
    /// Greedy heuristic — fast, occasionally suboptimal.
    Greedy,
    /// Exhaustive optimal search (small networks only).
    Optimal,
    /// Dynamic programming.
    DynamicProgramming,
    /// Dynamic programming with a specific objective (`"dp-flops"`, etc.).
    DynamicProgrammingObjective(DpObjective),
    /// Branch-and-bound (full search level).
    BranchBound,
    /// Branch-and-bound with a specific search level.
    Branch(BranchLevel),
    /// Randomized greedy search; `0` is clamped to one repeat.
    RandomGreedy(usize),
}

impl PathStrategy {
    fn as_opt_einsum_str(self) -> Cow<'static, str> {
        match self {
            PathStrategy::Auto => Cow::Borrowed("auto"),
            PathStrategy::AutoHq => Cow::Borrowed("auto-hq"),
            PathStrategy::Greedy => Cow::Borrowed("greedy"),
            PathStrategy::Optimal => Cow::Borrowed("optimal"),
            PathStrategy::DynamicProgramming => Cow::Borrowed("dp"),
            PathStrategy::DynamicProgrammingObjective(objective) => objective.as_opt_einsum_str(),
            PathStrategy::BranchBound => Cow::Borrowed("branch-all"),
            PathStrategy::Branch(level) => Cow::Borrowed(level.as_opt_einsum_str()),
            PathStrategy::RandomGreedy(repeats) => {
                Cow::Owned(format!("random-greedy-{}", repeats.max(1)))
            }
        }
    }
}

/// Dense contraction-order optimizer backed by the `opt-einsum-path` crate.
///
/// Implements [`DenseContractionOptimizer`]; its steps feed
/// [`crate::ContractionPlan::from_steps`].
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
/// let steps = OptEinsumPathOptimizer::default().optimize(&ir, &cost)?;
/// let plan = ContractionPlan::from_steps(&ir, steps)?;
/// assert_eq!(plan.active_pair_path()?.len(), 2);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OptEinsumPathOptimizer {
    strategy: PathStrategy,
    memory_limit: PathMemoryLimit,
}

impl OptEinsumPathOptimizer {
    /// Optimizer using the given driver.
    pub fn new(strategy: PathStrategy) -> Self {
        Self {
            strategy,
            memory_limit: PathMemoryLimit::None,
        }
    }

    /// Set the upstream path search memory limit.
    pub fn with_memory_limit(mut self, memory_limit: PathMemoryLimit) -> Self {
        self.memory_limit = memory_limit;
        self
    }

    /// Driver this optimizer runs.
    pub fn strategy(&self) -> PathStrategy {
        self.strategy
    }

    /// Upstream path-search memory limit.
    pub fn memory_limit(&self) -> PathMemoryLimit {
        self.memory_limit
    }
}

impl DenseContractionOptimizer for OptEinsumPathOptimizer {
    fn optimize(
        &self,
        ir: &NetworkIR,
        cost_model: &DenseCostModel,
    ) -> Result<Vec<ContractionStep>> {
        if ir.tensors().len() < 2 {
            return Err(ContractError::NotEnoughTensors);
        }

        let built = build_opt_einsum_inputs(ir, cost_model)?;
        let strategy = self.strategy.as_opt_einsum_str();
        let (path, _info) = opt_einsum_path::contract_path(
            &built.subscripts,
            &built.shapes,
            strategy.as_ref(),
            self.memory_limit.as_size_limit_type(),
        )
        .map_err(|e| {
            ContractError::InvalidContractionPlan(format!(
                "opt-einsum-path: {e} (subscripts={} shapes={:?})",
                built.subscripts, built.shapes
            ))
        })?;

        let pairs = path_to_active_pairs(&path, ir.tensors().len())?;
        dense_steps_from_active_pair_path(ir, &pairs, cost_model)
    }
}

/// Convert an opt_einsum linear path (positions into the shrinking active list,
/// result appended at the end — the same convention as [`ActivePair`]) into
/// symtn active pairs. opt_einsum treats each pair as unordered; symtn does
/// not, because lhs/rhs orientation controls fermionic order and orientation
/// costs. Preserve the upstream contraction tree while orienting each pair by
/// the earliest written operand in its subtree. The public explicit
/// [`dense_steps_from_active_pair_path`] preserves caller-supplied lhs/rhs exactly.
fn path_to_active_pairs(path: &[Vec<usize>], tensor_count: usize) -> Result<Vec<ActivePair>> {
    let pairs = path
        .iter()
        .map(|step| match step.as_slice() {
            [lhs, rhs] => Ok(ActivePair::new(*lhs, *rhs)),
            other => Err(ContractError::InvalidContractionPlan(format!(
                "opt-einsum-path returned a non-pairwise step with {} operands; \
                 symtn plans are strictly pairwise",
                other.len()
            ))),
        })
        .collect::<Result<Vec<_>>>()?;
    orient_unordered_active_pairs(&pairs, tensor_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::test_ir;
    use crate::{ContractionPlan, DenseTensorInfo};

    #[test]
    fn build_inputs_maps_labels_to_unique_symbols() {
        let ir = test_ir(&[&["a", "b"], &["b", "c"]], &["a", "c"]);
        let infos = vec![
            DenseTensorInfo::new(vec![2, 3]),
            DenseTensorInfo::new(vec![3, 4]),
        ];
        let cost = DenseCostModel::from_network(&ir, &infos).unwrap();

        let built = build_opt_einsum_inputs(&ir, &cost).unwrap();

        // 3 distinct labels a,b,c -> 3 distinct symbols.
        assert_eq!(built.symbols.len(), 3);
        let distinct: std::collections::BTreeSet<_> = built.symbols.values().copied().collect();
        assert_eq!(distinct.len(), 3, "symbols must be unique");

        // Equation shape: two inputs of width 2, output of width 2, with the
        // shared (middle) symbol appearing in both inputs but not the output.
        let (inputs, output) = built.subscripts.split_once("->").unwrap();
        let groups: Vec<&str> = inputs.split(',').collect();
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].chars().count(), 2);
        assert_eq!(groups[1].chars().count(), 2);
        assert_eq!(output.chars().count(), 2);

        let a = built.symbols[&TemporaryLabel::new("a")];
        let b = built.symbols[&TemporaryLabel::new("b")];
        let c = built.symbols[&TemporaryLabel::new("c")];
        assert_eq!(groups[0], format!("{a}{b}"));
        assert_eq!(groups[1], format!("{b}{c}"));
        assert_eq!(output, format!("{a}{c}"));

        // Shapes mirror the dense tensor infos.
        assert_eq!(built.shapes, vec![vec![2, 3], vec![3, 4]]);
    }

    #[test]
    fn optimizer_produces_valid_pairwise_plan() {
        // Chain of 3 tensors -> 2 pairwise steps; opt-einsum must be no worse
        // than symtn's greedy baseline.
        let ir = test_ir(&[&["a", "b"], &["b", "c"], &["c", "d"]], &["a", "d"]);
        let infos = vec![
            DenseTensorInfo::new(vec![2, 3]),
            DenseTensorInfo::new(vec![3, 4]),
            DenseTensorInfo::new(vec![4, 5]),
        ];
        let cost = DenseCostModel::from_network(&ir, &infos).unwrap();

        let plan = ContractionPlan::from_steps(
            &ir,
            OptEinsumPathOptimizer::default()
                .optimize(&ir, &cost)
                .unwrap(),
        )
        .unwrap();

        assert_eq!(
            plan.active_pair_path().unwrap(),
            vec![ActivePair::new(0, 1), ActivePair::new(1, 0)]
        );
        assert_eq!(plan.output_labels(), ir.output_labels());

        let explicit_path = vec![ActivePair::new(0, 1), ActivePair::new(0, 1)];
        let explicit = ContractionPlan::from_steps(
            &ir,
            dense_steps_from_active_pair_path(&ir, &explicit_path, &cost).unwrap(),
        )
        .unwrap();
        assert_eq!(explicit.active_pair_path().unwrap(), explicit_path);

        let report = plan.dense_cost_report(&ir, &cost).unwrap();
        assert!(
            !report.is_suboptimal(),
            "opt-einsum plan cost {} should be <= greedy {}",
            report.plan_cost(),
            report.greedy_cost()
        );
    }

    #[test]
    fn strategy_strings_cover_extended_optimizers() {
        assert_eq!(
            PathStrategy::Branch(BranchLevel::One).as_opt_einsum_str(),
            "branch-1"
        );
        assert_eq!(
            PathStrategy::Branch(BranchLevel::Two).as_opt_einsum_str(),
            "branch-2"
        );
        assert_eq!(
            PathStrategy::DynamicProgrammingObjective(DpObjective::Size).as_opt_einsum_str(),
            "dp-size"
        );
        assert_eq!(
            PathStrategy::DynamicProgrammingObjective(DpObjective::ComboFactor(256))
                .as_opt_einsum_str(),
            "dp-combo-256"
        );
        assert_eq!(
            PathStrategy::RandomGreedy(4).as_opt_einsum_str(),
            "random-greedy-4"
        );
        assert_eq!(
            PathStrategy::RandomGreedy(0).as_opt_einsum_str(),
            "random-greedy-1"
        );
    }

    #[test]
    fn extended_optimizer_options_produce_valid_plans() {
        let ir = test_ir(&[&["a", "b"], &["b", "c"], &["c", "d"]], &["a", "d"]);
        let infos = vec![
            DenseTensorInfo::new(vec![2, 3]),
            DenseTensorInfo::new(vec![3, 4]),
            DenseTensorInfo::new(vec![4, 5]),
        ];
        let cost = DenseCostModel::from_network(&ir, &infos).unwrap();

        for optimizer in [
            OptEinsumPathOptimizer::new(PathStrategy::Branch(BranchLevel::One))
                .with_memory_limit(PathMemoryLimit::MaxInput),
            OptEinsumPathOptimizer::new(PathStrategy::DynamicProgrammingObjective(
                DpObjective::Size,
            ))
            .with_memory_limit(PathMemoryLimit::Size(20)),
            OptEinsumPathOptimizer::new(PathStrategy::RandomGreedy(4)),
        ] {
            let plan =
                ContractionPlan::from_steps(&ir, optimizer.optimize(&ir, &cost).unwrap()).unwrap();
            assert_eq!(plan.active_pair_path().unwrap().len(), 2);
        }
    }

    #[test]
    fn non_pairwise_step_is_rejected() {
        let err = path_to_active_pairs(&[vec![0, 1, 2]], 3).unwrap_err();
        assert!(matches!(err, ContractError::InvalidContractionPlan(_)));
    }
}
