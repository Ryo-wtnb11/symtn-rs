#![forbid(unsafe_code)]

//! # symtn-network — contraction planning and execution for the symtn user layer
//!
//! The planner half (labels, [`NetworkIR`], cost models, optimizer trait,
//! [`ContractionPlan`], slicing) is **pure structure** over labels and leg
//! dimensions and never touches tensor data. Explicit execution uses homogeneous
//! typed Host [`symtn::typed::TensorMap`] operands and a caller-owned typed
//! workspace; device operands run the same schedule on CUDA when enabled.
//!
//! ## Pipeline
//!
//! ```text
//! Network::new(labels) ->  Network (label lists + conj markers + output)
//!   -> NetworkIR + DenseCostModel        (per-label dimension map)
//!   -> DenseContractionOptimizer         (greedy by default)
//!   -> ContractionPlan                   (reusable, serializable)
//!   -> PlannedNetwork::execute(&[&TensorMap<R, D>], &mut workspace) -> TensorMap<R, D>
//! ```
//!
//! [`Network::contract`] runs the same pipeline eagerly through the
//! Runtime's topology-keyed plan cache.
//!
//! There is **no einsum-string parser**: labels are per-operand
//! identifier lists given to [`Network::new`] and lower directly to
//! [`NetworkIR`]. [`contract!`] writes the same network in index notation
//! and checks its label structure at compile time.
//!
//! ## Path search and slicing
//!
//! - **cotengra external path search**: the optional `cotengra-python`
//!   feature calls the installed Python `cotengra` package for path search
//!   while keeping execution in Rust. The optional `opt-path` feature wraps
//!   the `opt-einsum-path` crate for optimal / dp / branch-and-bound searches.
//! - **Sliced execution**: the slicing *decision* types ([`SlicePlan`],
//!   [`greedy_slice`]) choose the sliced labels, and [`Network::lower_symmetric_sliced_plan`]
//!   with [`Network::execute_symmetric_sliced`] runs a lowered plan under a
//!   measured payload ceiling.

#[cfg(test)]
extern crate self as symtn_network;

#[cfg(test)]
#[path = "../../symtn/tests/braiding_probe/mod.rs"]
mod braiding_probe;

mod cost;
#[cfg(feature = "cotengra-python")]
mod cotengra_python;
mod error;
mod ir;
mod labels;
mod network;
mod optimizer;
#[cfg(feature = "opt-path")]
mod pathopt;
mod plan;
mod plancache;
#[cfg(test)]
mod probe;
mod slice;
mod stepflow;
mod tree;

pub use cost::{DenseCostModel, DenseTensorInfo};
#[cfg(feature = "cotengra-python")]
pub use cotengra_python::CotengraPythonOptimizer;
pub use error::{
    ContractError, Result, SliceError, SliceResult, SymmetricSliceExecutionError,
    SymmetricSliceLowerError,
};
pub use ir::{HyperEdge, NetworkIR, TensorNode};
pub use labels::{LabelOccurrence, TemporaryLabel, TensorAxis, TensorId};
pub use network::{
    Network, NetworkExecutionWorkspace, NetworkOperand, PlannedNetwork, SymmetricSliceStats,
};
pub use optimizer::{
    ContractionStep, DenseContractionOptimizer, DensePlanCostReport, GreedyDenseOptimizer,
    LabelOrderDenseOptimizer,
};
#[cfg(feature = "opt-path")]
pub use pathopt::{
    BranchLevel, DpObjective, OptEinsumPathOptimizer, PathMemoryLimit, PathStrategy,
};
pub use plan::{
    active_pair_path_from_steps, active_pair_path_from_tree, dense_steps_from_active_pair_path,
    ActivePair, ContractionPlan,
};
pub use plancache::{
    clear_plan_cache, configure_plan_cache, load_plan_cache, plan_cache_config, plan_cache_stats,
    save_plan_cache, Optimizer, PlanCacheConfig, PlanCacheStats, ReplanPolicy,
    DEFAULT_PLAN_CACHE_CAPACITY, DEFAULT_REPLAN_DRIFT_FACTOR, DEFAULT_WORKSPACE_BUDGET_BYTES,
};
#[cfg(feature = "cotengra-python")]
pub use plancache::{
    CotengraMinimize, CotengraPythonConfig, CotengraPythonMethod, CotengraSlicingConfig,
};
pub use slice::{
    best_next_internal_index, best_next_slice_index, contraction_width, greedy_slice,
    slice_plan_for, DegeneracyRange, SectorSlice, SliceKind, SliceLabels, SlicePlan, SlicedPlan,
    SymmetricIndexSlice, SymmetricSlicePlan, SymmetricSliceSpec, SymmetricSlicedPlan,
};
pub use tree::ContractionTree;

/// Contracts a network written in index notation, like TensorOperations'
/// `@tensor`: `contract!([i; k] := a[i; j] * b[j; k])?`.
///
/// It is [`Network::new`] plus [`Network::contract`], nothing more: each call
/// site builds its `Network` once (in a `static`), and every call contracts
/// the operands through the Runtime's plan cache, on Host and CUDA operands
/// alike. Call [`Network::plan`] to choose the optimizer.
///
/// | Written | Meaning |
/// |---|---|
/// | `[i; k] := …` | a new tensor, codomain `i`, domain `k` (the output's `;` position) |
/// | `[i, k] := …`, `[i k] := …` | a new tensor with every leg in the codomain |
/// | `[] := …` | a rank-0 tensor |
/// | no left-hand side | every label contracted; returns the scalar `D` through [`symtn::typed::TensorMap::scalar`], so it needs a Host tensor over multiplicity-free sectors (as does `[] :=` followed by `.scalar()`); checked-generic operands (e.g. SU(N)) and CUDA operands write `[] :=` and keep the rank-0 tensor |
/// | `a[i, j; k]` | operand `a`, labels in its leg order; the `;` must sit at its codomain rank, checked at run time |
/// | `a[i, j]` (no `;`) | every leg of `a` in the codomain: `a` must have domain rank 0 |
/// | `conj(a)[i; j]` | the adjoint of `a`, labels in `a`'s own leg order (the [`Network::new`] `conj` convention) |
/// | `svd.u[i; s]`, `t.0.0[i; s]` | an identifier or a field chain, as written |
/// | `(expr)[i; s]` | any other operand expression, in parentheses (`?` inside is the caller's) |
/// | `*` | joins the operands |
/// | a label twice | contracted; once plus once on the left-hand side: open |
///
/// Labels are identifiers, separated by `,` or spaces. Compile errors, each
/// with a hint: a label repeated within one operand (an intra-operand trace,
/// use `TensorMap::trace_pairs` first), a label on three legs (a
/// hyperedge), a sum (`+`, `-`), a scalar factor, `=` and `+=` (a `Network`
/// has no destination API), an unpaired or unknown label, an indexed
/// operand such as `w[0][i; j]` (write `(w[0])[i; j]`), and a named output
/// `c[i; k] := …` (write `let c = contract!([i; k] := …)?`).
///
/// The result is `Result<T, T::Error>` of [`Network::contract`] (or
/// `Result<D, T::Error>` without a left-hand side): an operand whose tensor
/// does not match its written labels or `;` split is a run-time error,
/// decided before the plan lookup.
///
/// The expansion names `::symtn_network`, so the calling crate must depend
/// on `symtn-network` under that name.
///
/// ```
/// use std::sync::Arc;
/// use symtn::sector::{U1FusionRule, U1Irrep};
/// use symtn::typed::{Error, GradedSpace, Runtime, TensorMap};
/// use symtn_network::contract;
///
/// # fn main() -> Result<(), Error> {
/// let runtime = Runtime::builder().build()?;
/// let v = GradedSpace::try_new(
///     Arc::new(U1FusionRule),
///     [(U1Irrep::new(-1), 1), (U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
/// )?;
/// let a = TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&v], [&v], 1)?;
/// let b = TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&v], [&v], 2)?;
/// let c = contract!([i; k] := a[i; j] * b[j; k])?;
/// assert_eq!((c.codomain_rank(), c.domain_rank()), (1, 1));
/// let trace: f64 = contract!(a[i; j] * b[j; i])?;
/// let rank0 = contract!([] := a[i; j] * b[j; i])?;
/// assert!((trace - rank0.scalar()?).abs() <= 1e-12 * trace.abs().max(1.0));
/// # Ok(())
/// # }
/// ```
pub use symtn_macros::contract;

/// The workspace tolerance rule for arithmetic test comparisons
/// (`docs/testing_numerics.md`).
#[cfg(test)]
#[path = "../../tests/support"]
mod test_numerics {
    use symtn::typed::{Complex32, Complex64};
    pub(crate) mod numerics;
}
