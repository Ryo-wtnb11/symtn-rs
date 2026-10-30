//! Typed Host contraction of a labeled tensor network.
//!
//! This is the execution half rewritten for the current user layer: the
//! planner ([`NetworkIR`], [`DenseCostModel`], [`ContractionPlan`]) is pure
//! structure, and each planned pairwise step lowers to
//! typed contraction plus orientation/final permutation calls.
//! [`Network::contract`] runs the same typed schedule through the Runtime's
//! plan cache.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "cuda")]
use symtn::expert::Placement;
use symtn::expert::{SectorLeg, TensorStorage};
use symtn::sector::{
    CheckedFusionAlgebra, CheckedGenericAdmissionMode, CheckedGenericFusion,
    CheckedGenericRigidSymbols, MultiplicityFreeAdmissionMode, MultiplicityFreeRigidSymbols,
    RuleIdentity, SectorCodec, TypedSectorAdmission,
};
use symtn::typed::__network::{
    self, NetworkDegeneracyRestriction, NetworkPayloadStorage, NetworkReuseClass,
    RuntimeDetachedTensorMap, RuntimeIdentity,
};
use symtn::typed::FusionAlgebraError;
#[cfg(test)]
use symtn::typed::OperationError;
use symtn::typed::{
    ContractExec, ContractSpec, GradedSpace, TensorMap, TransformExec, TypedAdjointSpace,
    TypedSpaceModeDispatch, TypedTensorContractDispatch, TypedTensorModeDispatch,
    TypedTensorRootDispatch, TypedTensorTransformDispatch,
};
#[cfg(feature = "cuda")]
use symtn::typed::{CudaPayload, CudaStorage};
use symtn::typed::{Error, Runtime, TensorScalar};

use crate::cost::{DenseCostModel, DenseTensorInfo};
use crate::error::{SliceError, SymmetricSliceExecutionError, SymmetricSliceLowerError};
use crate::ir::NetworkIR;
use crate::labels::{TemporaryLabel, TensorId};
use crate::optimizer::{next_use_axes, DenseContractionOptimizer};
use crate::plan::ContractionPlan;
use crate::slice::{
    lower_symmetric_sliced_plan, validate_contraction_plan_for_ir, SlicedPlan, SymmetricSlicePlan,
    SymmetricSliceSpec, SymmetricSlicedPlan,
};

mod construct;
mod dispatch;
mod eager;
mod execute;
#[cfg(test)]
mod leg_contract_tests;
mod preflight;
mod schedule;
#[cfg(test)]
mod typed_replay_tests;

// Every split-out file is a sibling submodule of `network`, not a
// standalone crate: re-export each child's items here (at their own,
// unwidened visibility) so `use super::*;` in any sibling, including the
// two test modules, resolves them exactly as when they were all one file.
// The crate's actual public surface is unchanged: only the names in the
// explicit `pub use` block below were reachable outside this crate before
// the split (via `symtn_network::network::{...}` in lib.rs), and they keep
// that exact path.
use execute::*;
use preflight::*;
use schedule::*;

// `NetworkModeDispatch` was directly in this module before the split and
// `crate::plancache` names it at `crate::network::NetworkModeDispatch`;
// plancache is not a descendant of `network`, so it needs a crate-visible
// (not just intra-`network`) path back to it.
pub(crate) use dispatch::NetworkModeDispatch;

pub use eager::NetworkOperand;
pub(crate) use eager::NetworkOperandStorage;
pub use execute::{NetworkExecutionWorkspace, PlannedNetwork};

/// A labeled tensor network: per-operand label lists (+ conj markers) and
/// the requested output labels with their codomain/domain split.
///
/// Labels are caller-supplied strings, one list per operand; there is no
/// public einsum-string parser. Build with
/// [`Network::new`], then either contract eagerly through the Runtime's plan
/// cache with [`Network::contract`], or plan explicitly with
/// [`Network::plan`] / [`Network::plan_with`] and run
/// [`PlannedNetwork::execute`] (Host or device) with a caller-owned
/// [`NetworkExecutionWorkspace`].
///
/// # Intra-operand traces
///
/// A label written twice on one operand is rejected by [`Network::new`]:
/// a network only contracts legs of different operands. Trace first, then
/// build the network over the traced tensor — `t.trace_pairs(&[(i, j)])`
/// (read through `t.adjoint()?` first when the operand is conjugated, with
/// the pairs in the adjoint's leg order) — and label only its remaining
/// legs:
///
/// ```
/// use std::sync::Arc;
/// use symtn::sector::{U1FusionRule, U1Irrep};
/// use symtn::typed::{CpuEngine, GradedSpace, Runtime, TensorMap};
/// use symtn_network::Network;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut engine = CpuEngine::serial();
/// let runtime = Runtime::builder().build()?;
/// let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)])?;
/// // a[i, k; i, l] traced over its first codomain and domain legs, then
/// // contracted with b[l; m] over `l`.
/// let a = TensorMap::<_, f64>::rand_with_seed(&mut engine, &runtime, [&v, &v], [&v, &v], 1)?;
/// let b = TensorMap::<_, f64>::rand_with_seed(&mut engine, &runtime, [&v], [&v], 2)?;
/// let traced = a.trace_pairs(&mut engine, &[(0, 2)])?; // legs [k; l]
/// let network = Network::new([["k", "l"], ["l", "m"]], [false, false], [1, 1], ["k", "m"], Some(1))?;
/// let result = network.contract(&mut engine, &[&traced, &b])?;
/// assert_eq!(result.rank(), 2);
/// # Ok(())
/// # }
/// ```
pub struct Network {
    pub(crate) inputs: Vec<Vec<TemporaryLabel>>,
    pub(crate) conj: Vec<bool>,
    pub(crate) codomain_splits: Vec<usize>,
    pub(crate) output: Vec<TemporaryLabel>,
    /// Number of output labels on the codomain side (`;` position);
    /// `None` = all-codomain output.
    pub(crate) output_codomain_rank: Option<usize>,
    /// The contracted leg pairing, resolved once here: for each operand and
    /// each written axis, the earlier `(operand, written axis)` carrying the
    /// same label, or `None` for the label's first occurrence. It keeps
    /// [`Network::contract`]'s per-call metadata preflight linear in the legs.
    pub(crate) contracted: Vec<Vec<Option<(usize, usize)>>>,
    /// Hash of every field above except `contracted`, which they determine:
    /// the [`Network::contract`] plan-cache lookup hashes it with the
    /// optimizer instead of rehashing the labels on every call.
    pub(crate) topology_hash: u64,
}

static NEXT_PLAN_OWNER_TOKEN: AtomicU64 = AtomicU64::new(1);

fn invalid(message: impl std::fmt::Display) -> Error {
    Error::InvalidArgument(message.to_string())
}

pub(crate) use symtn::typed::FacadeError;

/// Network-owned payload accounting for one symmetric sliced execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SymmetricSliceStats {
    destination_bytes: usize,
    peak_workspace_bytes: usize,
    peak_total_bytes: usize,
}

impl SymmetricSliceStats {
    /// Full private unsliced destination payload capacity.
    pub fn destination_bytes(self) -> usize {
        self.destination_bytes
    }

    /// Peak non-destination network-owned payload observed during execution.
    pub fn peak_workspace_bytes(self) -> usize {
        self.peak_workspace_bytes
    }

    /// Peak total payload, checked as destination plus workspace at each observation.
    pub fn peak_total_bytes(self) -> usize {
        self.peak_total_bytes
    }
}
