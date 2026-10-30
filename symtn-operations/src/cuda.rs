//! CUDA storage and GEMM seams for the symmetry-free replay layer.
//!
//! `CudaStorage<D>` is a flat device buffer of [`CudaScalar`] payload `D`
//! implementing [`TensorStorage`] (never host-readable: no silent transfers),
//! and [`CudaStorageGemm`] implements the [`StorageGemm`] device replay seam by
//! delegating each coupled-sector matrix GEMM to the symtn-dense CUDA boundary.
//!
//! The payload dtype is the only thing that varies: structural (fusion-tree)
//! coefficients stay real, and operand conjugation stays a GEMM flag rather
//! than a materialized buffer.

use std::marker::PhantomData;

use symtn_core::{Placement, TensorStorage};
use symtn_dense::{
    cuda_conj, cuda_gather_elements, cuda_gather_member_elements, cuda_gather_members,
    cuda_gemm_region_batched_into, cuda_gemm_region_batched_scaled_into,
    cuda_gemm_region_with_ops_into, cuda_matmul_region_into, cuda_widen, CudaDenseContext,
    CudaDenseStorage, CudaScalar, MatrixOp,
};

use crate::fusion_replay::StorageGemm;
use crate::stacked::{StackedStorageView, StackedStorageViewMut};
use crate::OperationError;

/// Flat device buffer of payload `D` usable as replay storage.
///
/// The default `D = f64` keeps the historical `CudaStorage` spelling valid.
pub struct CudaStorage<D: CudaScalar = f64>(pub CudaDenseStorage, PhantomData<D>);

impl<D: CudaScalar> std::fmt::Debug for CudaStorage<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaStorage")
            .field("len", &self.0.len())
            .field("dtype", &self.0.dtype())
            .field("device", &self.0.device())
            .finish()
    }
}

impl<D: CudaScalar> CudaStorage<D> {
    /// Uploads borrowed host data. Costs the one host copy Tenferro's
    /// owned-host-tensor upload requires; see [`CudaDenseStorage::upload`].
    pub fn upload(ctx: &CudaDenseContext, data: &[D]) -> Result<Self, OperationError> {
        Self::upload_owned(ctx, data.to_vec())
    }

    /// Uploads owned host data by moving it into the uploaded host tensor.
    pub fn upload_owned(ctx: &CudaDenseContext, data: Vec<D>) -> Result<Self, OperationError> {
        CudaDenseStorage::upload_owned(ctx, data)
            .map(|storage| Self(storage, PhantomData))
            .map_err(OperationError::Dense)
    }

    /// Uploads a stack of `members` members of `member_len` elements as a
    /// `[member_len, members]` buffer; see [`CudaDenseStorage::upload_members`].
    #[doc(hidden)]
    pub fn upload_members(
        ctx: &CudaDenseContext,
        data: Vec<D>,
        member_len: usize,
        members: usize,
    ) -> Result<Self, OperationError> {
        CudaDenseStorage::upload_members(ctx, data, member_len, members)
            .map(|storage| Self(storage, PhantomData))
            .map_err(OperationError::Dense)
    }

    pub fn download(&self, ctx: &CudaDenseContext) -> Result<Vec<D>, OperationError> {
        self.0.download(ctx).map_err(OperationError::Dense)
    }

    /// This buffer widened to the double-precision lane `W` on the device;
    /// see [`cuda_widen`].
    pub fn widened<W: CudaScalar>(
        &self,
        ctx: &mut CudaDenseContext,
    ) -> Result<CudaStorage<W>, OperationError> {
        cuda_widen::<W>(ctx, &self.0)
            .map(|storage| CudaStorage(storage, PhantomData))
            .map_err(OperationError::Dense)
    }

    /// Elements `elements` of this buffer, by linear position, as a new flat
    /// buffer; see [`cuda_gather_elements`].
    #[doc(hidden)]
    pub fn gather_elements(
        &self,
        ctx: &mut CudaDenseContext,
        elements: &[usize],
    ) -> Result<Self, OperationError> {
        cuda_gather_elements::<D>(ctx, &self.0, elements)
            .map(|storage| Self(storage, PhantomData))
            .map_err(OperationError::Dense)
    }

    /// The elementwise conjugate of this buffer in a new buffer; see
    /// [`cuda_conj`].
    #[doc(hidden)]
    pub fn conj(&self, ctx: &mut CudaDenseContext) -> Result<Self, OperationError> {
        cuda_conj::<D>(ctx, &self.0)
            .map(|storage| Self(storage, PhantomData))
            .map_err(OperationError::Dense)
    }

    /// Members `selection` of a stack of `members` members of `member_len`
    /// elements held in this buffer, as a new buffer; see
    /// [`cuda_gather_members`].
    #[doc(hidden)]
    pub fn gather_members(
        &self,
        ctx: &mut CudaDenseContext,
        member_len: usize,
        members: usize,
        selection: &[usize],
    ) -> Result<Self, OperationError> {
        cuda_gather_members::<D>(ctx, &self.0, member_len, members, selection)
            .map(|storage| Self(storage, PhantomData))
            .map_err(OperationError::Dense)
    }

    /// Elements `elements` of every member of a `[member_len, members]`
    /// stack held in this buffer, as a new `[elements.len(), members]`
    /// buffer; see [`cuda_gather_member_elements`].
    #[doc(hidden)]
    pub fn gather_member_elements(
        &self,
        ctx: &mut CudaDenseContext,
        member_len: usize,
        members: usize,
        elements: Vec<i64>,
    ) -> Result<Self, OperationError> {
        cuda_gather_member_elements::<D>(ctx, &self.0, member_len, members, elements)
            .map(|storage| Self(storage, PhantomData))
            .map_err(OperationError::Dense)
    }
}

impl<D: CudaScalar> TensorStorage<D> for CudaStorage<D> {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn placement(&self) -> Placement {
        Placement::Cuda(self.0.device())
    }
}

/// Operand orientations the device GEMM seam accepts. `Transpose` has no
/// conjugation-free device analogue here and is rejected before device work.
fn cuda_operand_is_supported(op: MatrixOp) -> bool {
    matches!(op, MatrixOp::Identity | MatrixOp::Adjoint)
}

/// [`StorageGemm`] over CUDA storage: one tenferro dot-general per
/// coupled-sector matrix, executed in place on device buffer regions.
pub struct CudaStorageGemm<'a, 'e> {
    ctx: &'a mut CudaDenseContext<'e>,
}

impl<'a, 'e> CudaStorageGemm<'a, 'e> {
    pub fn new(ctx: &'a mut CudaDenseContext<'e>) -> Self {
        Self { ctx }
    }
}

impl<D: CudaScalar> StorageGemm<D, CudaStorage<D>, CudaStorage<D>, CudaStorage<D>>
    for CudaStorageGemm<'_, '_>
{
    fn supports_matmul_with_ops_scaled(&self, lhs_op: MatrixOp, rhs_op: MatrixOp) -> bool {
        cuda_operand_is_supported(lhs_op) && cuda_operand_is_supported(rhs_op)
    }

    fn matmul_range_into(
        &mut self,
        dst: &mut CudaStorage<D>,
        dst_offset: usize,
        lhs: &CudaStorage<D>,
        lhs_offset: usize,
        rhs: &CudaStorage<D>,
        rhs_offset: usize,
        rows: usize,
        contracted: usize,
        cols: usize,
    ) -> Result<(), OperationError> {
        cuda_matmul_region_into::<D>(
            self.ctx, &mut dst.0, dst_offset, &lhs.0, lhs_offset, &rhs.0, rhs_offset, rows,
            contracted, cols,
        )
        .map_err(OperationError::Dense)
    }

    fn matmul_range_with_ops_scaled_into(
        &mut self,
        dst: &mut CudaStorage<D>,
        dst_offset: usize,
        lhs: &CudaStorage<D>,
        lhs_offset: usize,
        rhs: &CudaStorage<D>,
        rhs_offset: usize,
        rows: usize,
        contracted: usize,
        cols: usize,
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
        alpha: D,
    ) -> Result<(), OperationError> {
        if !(cuda_operand_is_supported(lhs_op) && cuda_operand_is_supported(rhs_op)) {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "CUDA storage GEMM supports only identity and adjoint operands",
            });
        }
        cuda_gemm_region_with_ops_into::<D>(
            self.ctx,
            &mut dst.0,
            dst_offset,
            &lhs.0,
            lhs_offset,
            &rhs.0,
            rhs_offset,
            rows,
            contracted,
            cols,
            lhs_op,
            rhs_op,
            alpha,
            D::ZERO,
        )
        .map_err(OperationError::Dense)
    }

    fn matmul_range_axpby_with_ops_into(
        &mut self,
        dst: &mut CudaStorage<D>,
        dst_offset: usize,
        lhs: &CudaStorage<D>,
        lhs_offset: usize,
        rhs: &CudaStorage<D>,
        rhs_offset: usize,
        rows: usize,
        contracted: usize,
        cols: usize,
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError> {
        if !(cuda_operand_is_supported(lhs_op) && cuda_operand_is_supported(rhs_op)) {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "CUDA storage GEMM supports only identity and adjoint operands",
            });
        }
        cuda_gemm_region_with_ops_into::<D>(
            self.ctx, &mut dst.0, dst_offset, &lhs.0, lhs_offset, &rhs.0, rhs_offset, rows,
            contracted, cols, lhs_op, rhs_op, alpha, beta,
        )
        .map_err(OperationError::Dense)
    }
}

/// [`StorageGemm`] over member-strided device stacks: each plan job is one
/// batched `dot_general` over all `B` members (see
/// [`cuda_gemm_region_batched_into`]), so a replay submits one GEMM per job
/// whatever `B` is.
///
/// Identity orientation with exact-sign coefficients. It is the only seam that
/// accepts the stacked views, so a stack never reaches
/// [`CudaStorageGemm`], which would compute member 0 alone.
#[doc(hidden)]
pub struct CudaStackedStorageGemm<'a, 'e> {
    ctx: &'a mut CudaDenseContext<'e>,
}

impl<'a, 'e> CudaStackedStorageGemm<'a, 'e> {
    pub fn new(ctx: &'a mut CudaDenseContext<'e>) -> Self {
        Self { ctx }
    }
}

impl<D: CudaScalar>
    StorageGemm<
        D,
        StackedStorageViewMut<'_, CudaStorage<D>>,
        StackedStorageView<'_, CudaStorage<D>>,
        StackedStorageView<'_, CudaStorage<D>>,
    > for CudaStackedStorageGemm<'_, '_>
{
    fn supports_matmul_with_ops_scaled(&self, lhs_op: MatrixOp, rhs_op: MatrixOp) -> bool {
        lhs_op == MatrixOp::Identity && rhs_op == MatrixOp::Identity
    }

    fn matmul_range_into(
        &mut self,
        dst: &mut StackedStorageViewMut<'_, CudaStorage<D>>,
        dst_offset: usize,
        lhs: &StackedStorageView<'_, CudaStorage<D>>,
        lhs_offset: usize,
        rhs: &StackedStorageView<'_, CudaStorage<D>>,
        rhs_offset: usize,
        rows: usize,
        contracted: usize,
        cols: usize,
    ) -> Result<(), OperationError> {
        let members = dst.members();
        if lhs.members() != members || rhs.members() != members {
            return Err(OperationError::InvalidArgument {
                message: "stacked GEMM operands have different member counts",
            });
        }
        let dst_stride = dst.member_stride();
        cuda_gemm_region_batched_into::<D>(
            self.ctx,
            &mut dst.storage_mut().0,
            dst_offset,
            dst_stride,
            &lhs.storage().0,
            lhs_offset,
            lhs.member_stride(),
            &rhs.storage().0,
            rhs_offset,
            rhs.member_stride(),
            rows,
            contracted,
            cols,
            members,
        )
        .map_err(OperationError::Dense)
    }

    fn matmul_range_with_ops_scaled_into(
        &mut self,
        dst: &mut StackedStorageViewMut<'_, CudaStorage<D>>,
        dst_offset: usize,
        lhs: &StackedStorageView<'_, CudaStorage<D>>,
        lhs_offset: usize,
        rhs: &StackedStorageView<'_, CudaStorage<D>>,
        rhs_offset: usize,
        rows: usize,
        contracted: usize,
        cols: usize,
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
        alpha: D,
    ) -> Result<(), OperationError> {
        if lhs_op != MatrixOp::Identity || rhs_op != MatrixOp::Identity {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "stacked CUDA GEMM requires identity operand orientations",
            });
        }
        let members = dst.members();
        if lhs.members() != members || rhs.members() != members {
            return Err(OperationError::InvalidArgument {
                message: "stacked GEMM operands have different member counts",
            });
        }
        let dst_stride = dst.member_stride();
        cuda_gemm_region_batched_scaled_into::<D>(
            self.ctx,
            &mut dst.storage_mut().0,
            dst_offset,
            dst_stride,
            &lhs.storage().0,
            lhs_offset,
            lhs.member_stride(),
            &rhs.storage().0,
            rhs_offset,
            rhs.member_stride(),
            rows,
            contracted,
            cols,
            members,
            alpha,
        )
        .map_err(OperationError::Dense)
    }
}
