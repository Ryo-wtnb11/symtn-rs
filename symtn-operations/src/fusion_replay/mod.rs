//! Symmetry-free replay half of the core fusion-block contraction:
//! plan data (offsets, strides, coefficients), group-local host execution,
//! workspaces, and the storage-direct device seam. The symmetric compile layer
//! builds these plans; nothing here consumes fusion rules.

use std::collections::HashSet;
use std::ops::Mul;
use std::sync::Arc;

use num_traits::{One, Zero};
use symtn_core::{BlockStructure, CoupledSectorRegion, Placement, SectorId, TensorStorage};
pub use symtn_dense::MatrixOp;
use symtn_dense::{strided_batch_runs, strided_batch_runs_into, DenseGemmBatchJob, Engine};

use crate::host_scalar_kernels::validate_raw_strided_bounds;
use crate::host_scratch::HostScratchBuffer;
use crate::placement::ReportsPlacement;
use crate::strided::{offset_to_isize, strides_to_isize};
use crate::structure_identity::validate_structure_identity;
use crate::transform_structure::validate_destination_layouts_injective;
use crate::{
    DenseBlockScalar, HostKernelAdapter, OperationError, RecouplingCoefficientAction,
    TransformScale,
};

/// Storage-handle replay accepts only operands whose coupled-sector matrices
/// sit directly in storage. Host replay also supports group-local pack/scatter,
/// but storage/device kernels do not expose that capability.
const NON_COUPLED_OPERAND_MESSAGE: &str =
    "storage-handle core fusion-block replay requires the coupled sector matrix layout";

mod execute;
mod gemm;
mod plan;
#[cfg(test)]
mod tests;
mod validate;

pub use execute::*;
pub use gemm::*;
pub use plan::*;
use validate::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FusionGroupExecutionClass(u8);

impl FusionGroupExecutionClass {
    const PACK_LHS: u8 = 1;
    const PACK_RHS: u8 = 2;
    const SCATTER_DST: u8 = 4;

    fn compile<C>(group: &FusionBlockContractGroupPlan<C>) -> Self {
        let mut bits = 0;
        if group.lhs.direct_offset.is_none() {
            bits |= Self::PACK_LHS;
        }
        if group.rhs.direct_offset.is_none() {
            bits |= Self::PACK_RHS;
        }
        if group.dst.direct_offset.is_none() {
            bits |= Self::SCATTER_DST;
        }
        Self(bits)
    }

    #[inline]
    fn is_direct(self) -> bool {
        self.0 == 0
    }

    #[inline]
    fn packs_lhs(self) -> bool {
        self.0 & Self::PACK_LHS != 0
    }

    #[inline]
    fn packs_rhs(self) -> bool {
        self.0 & Self::PACK_RHS != 0
    }

    #[inline]
    fn scatters_dst(self) -> bool {
        self.0 & Self::SCATTER_DST != 0
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct FusionGroupScratchLayout {
    lhs_len: usize,
    rhs_offset: usize,
    rhs_len: usize,
    dst_offset: usize,
    dst_len: usize,
    total_len: usize,
}

#[derive(Clone, Debug)]
struct FusionIrregularGroupExecution {
    group_index: usize,
    class: FusionGroupExecutionClass,
    job: Rank2GemmBatchJob,
    scratch: FusionGroupScratchLayout,
}

type CompiledGroupExecution = (
    Vec<Rank2GemmBatchJob>,
    Vec<FusionIrregularGroupExecution>,
    usize,
);

pub struct HostFusionBlockContractWorkspace<T> {
    scratch: HostScratchBuffer<T>,
    // Zero-stride source layout of the `Axpby(0)` inactive-block assignment,
    // kept here so a warmed replay allocates nothing.
    zero_strides: Vec<isize>,
}

pub type FusionBlockContractWorkspace<T> = HostFusionBlockContractWorkspace<T>;

impl<T> Default for HostFusionBlockContractWorkspace<T> {
    fn default() -> Self {
        Self {
            scratch: HostScratchBuffer::default(),
            zero_strides: Vec::new(),
        }
    }
}

impl<T> ReportsPlacement for HostFusionBlockContractWorkspace<T> {
    #[inline]
    fn placement(&self) -> Placement {
        Placement::Host
    }
}

/// How a replay initialises the destination blocks that no GEMM or scatter
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
    /// (an owned output born from `alloc_zeroed`). Inactive blocks are not
    /// touched at all: plan validation proves each destination block is
    /// owned by exactly one GEMM job, one scatter group, or the inactive
    /// list, so `Axpby(0)` would only rewrite zeros that are already there.
    Zeroed,
}

impl<D: Zero> ContractDestinationInit<D> {
    /// The `beta` handed to the GEMM and scatter jobs of the active blocks.
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
    groups: Vec<FusionBlockContractGroupPlan<C>>,
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
    irregular: Vec<FusionIrregularGroupExecution>,
    max_irregular_scratch_len: usize,
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
