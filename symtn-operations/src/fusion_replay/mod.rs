//! Symmetry-free replay half of the core fusion-block contraction:
//! plan data (offsets, strides, coefficients), group-local host execution,
//! workspaces, and the storage-direct device seam. The symmetric compile layer
//! builds these plans; nothing here consumes fusion rules.

use std::collections::HashSet;
use std::mem::MaybeUninit;
use std::ops::Mul;
use std::sync::Arc;

use num_traits::{One, Zero};
use symtn_core::{BlockStructure, CoupledSectorRegion, Placement, SectorId, TensorStorage};
pub use symtn_dense::MatrixOp;
use symtn_dense::{strided_batch_runs, strided_batch_runs_into, DenseGemmBatchJob, Engine};

use crate::host_scalar_kernels::validate_raw_strided_bounds;
use crate::placement::ReportsPlacement;
use crate::strided::{offset_to_isize, strides_to_isize};
use crate::structure_identity::validate_structure_identity;
use crate::transform_structure::validate_destination_layouts_injective;
use crate::{DenseBlockScalar, HostKernelAdapter, OperationError, RecouplingCoefficientAction};

/// Every replay reads and writes each group's coupled-sector matrix in place.
/// Why no pack/scatter for other group layouts: every public tensor has the
/// canonical coupled-sector tiling, whose matrices are in place by
/// construction, so such a group never reaches a plan.
const NON_COUPLED_OPERAND_MESSAGE: &str =
    "core fusion-block replay requires every group matrix in place in storage";

mod execute;
mod gemm;
mod owned;
#[cfg(test)]
mod owned_tests;
mod plan;
#[cfg(test)]
mod tests;
mod validate;

pub use execute::*;
pub use gemm::*;
pub use plan::*;
use validate::*;

#[derive(Default)]
pub struct HostFusionBlockContractWorkspace {
    // Zero-stride source layout of the `Axpby(0)` inactive-block assignment,
    // kept here so a warmed replay allocates nothing.
    zero_strides: Vec<isize>,
    // Which jobs of the current fresh GEMM call have reported their output
    // (owned replay); reset at the start of every call.
    fresh_reported: Vec<bool>,
}

pub type FusionBlockContractWorkspace = HostFusionBlockContractWorkspace;

impl HostFusionBlockContractWorkspace {
    /// Bytes of every buffer's capacity: the zero-stride and fresh-report
    /// bookkeeping.
    #[doc(hidden)]
    pub fn retained_bytes(&self) -> usize {
        self.zero_strides
            .capacity()
            .saturating_mul(std::mem::size_of::<isize>())
            .saturating_add(
                self.fresh_reported
                    .capacity()
                    .saturating_mul(std::mem::size_of::<bool>()),
            )
    }
}

impl ReportsPlacement for HostFusionBlockContractWorkspace {
    #[inline]
    fn placement(&self) -> Placement {
        Placement::Host
    }
}

/// How a replay initialises the destination blocks that no GEMM
/// job writes (the inactive blocks); the active blocks always receive the
/// corresponding `beta`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ContractDestinationInit<D> {
    /// `dst = alpha * lhs * rhs + beta * dst`. `Axpby(0)` is a strong zero:
    /// inactive blocks are assigned `0` without reading them (TensorKit
    /// `β = false`, BLAS `beta == 0`), so NaN or Inf destination values do
    /// not leak into the output.
    Axpby(D),
    /// The caller proves every destination element is already `D::zero()`
    /// (an `alloc_zeroed` payload or a zero-filled member output; an owned
    /// eager output uses `execute_owned` instead). Inactive blocks are not
    /// touched at all: plan validation proves each destination block is
    /// owned by exactly one GEMM job or the inactive
    /// list, so `Axpby(0)` would only rewrite zeros that are already there.
    Zeroed,
}

impl<D: Zero> ContractDestinationInit<D> {
    /// The `beta` handed to the GEMM jobs of the active blocks.
    #[inline]
    pub fn active_beta(self) -> D {
        match self {
            Self::Axpby(beta) => beta,
            Self::Zeroed => D::zero(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct FusionBlockContractPlan<C = f64> {
    dst_structure: Arc<BlockStructure>,
    lhs_structure: Arc<BlockStructure>,
    rhs_structure: Arc<BlockStructure>,
    inactive_dst_scale_blocks: Vec<FusionScaleBlockLayout>,
    direct_batch: Vec<Rank2GemmBatchJob>,
    direct_batch_alpha: Vec<C>,
    // Plan-time run partition of `direct_batch` (see issue #103): the backend
    // reads it to route each run without recomputing the partition per replay.
    // A backend-agnostic shape fact, so it lives in this operations-layer plan
    // while the route choice (strided seam vs grouped) stays at the
    // dense-executor boundary.
    //
    // Jobs of one `direct_batch_alpha` coefficient are contiguous, and runs
    // never cross a coefficient change, so a Host replay submits one batch
    // per coefficient (a fermionic twist uniform per coupled sector) with no
    // per-replay partition and no extra plan field.
    direct_batch_runs: Vec<usize>,
    // `direct_batch` split into its coefficient batches: one GEMM call each.
    direct_calls: Vec<owned::DirectCall>,
    // Which call or zero fill initializes each element of an owned output.
    fresh: owned::FreshLayout,
    lhs_op: MatrixOp,
    rhs_op: MatrixOp,
}

#[derive(Clone, Debug)]
pub struct FusionBlockContractGroupPlan<C = f64> {
    pub lhs: FusionBlockMatrixGroup<C>,
    pub rhs: FusionBlockMatrixGroup<C>,
    pub dst: FusionBlockMatrixGroup<C>,
}

impl<C> FusionBlockContractGroupPlan<C> {
    /// Validates and packages a group triple; called by the compile layer.
    pub fn new(
        lhs: FusionBlockMatrixGroup<C>,
        rhs: FusionBlockMatrixGroup<C>,
        dst: FusionBlockMatrixGroup<C>,
    ) -> Result<Self, OperationError> {
        if lhs.cols != rhs.rows {
            return Err(OperationError::ShapeMismatch {
                dst: vec![lhs.cols],
                src: vec![rhs.rows],
            });
        }
        if dst.rows != lhs.rows || dst.cols != rhs.cols {
            return Err(OperationError::ShapeMismatch {
                dst: vec![dst.rows, dst.cols],
                src: vec![lhs.rows, rhs.cols],
            });
        }

        Ok(Self { lhs, rhs, dst })
    }
}

#[derive(Clone, Debug)]
pub struct FusionBlockMatrixGroup<C = f64> {
    pub coupled: SectorId,
    pub rows: usize,
    pub cols: usize,
    // False only when the group's subblocks cover the packed matrix exactly.
    // Sparse fusion layouts keep this true so stale workspace cannot leak into GEMM.
    pub needs_clear: bool,
    // Storage offset of the group matrix when the operand's subblocks already
    // form it in place (coupled-sector matrix layout, unit coefficients):
    // packing is the identity copy and replay can hand storage to GEMM
    // directly.
    pub direct_offset: Option<usize>,
    pub block_indices: Vec<usize>,
    pub subblocks: Vec<FusionSubblockMatrixLayout<C>>,
}

#[derive(Clone, Debug)]
pub struct FusionSubblockMatrixLayout<C = f64> {
    pub block: FusionStridedBlockLayout,
    pub matrix_offset: isize,
    pub matrix_strides: Vec<isize>,
    pub coefficient: C,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FusionStridedBlockLayout {
    pub shape: Vec<usize>,
    pub strides: Vec<isize>,
    pub offset: isize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FusionScaleBlockLayout {
    pub block: FusionStridedBlockLayout,
}
