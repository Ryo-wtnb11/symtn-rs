use super::*;

/// Column-major matrix GEMM over device buffer regions:
/// `dst[dst_offset..][rows x cols] = lhs_part * rhs_part` (overwrite).
#[allow(clippy::too_many_arguments)]
pub fn cuda_matmul_region_into<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    dst: &mut CudaDenseStorage,
    dst_offset: usize,
    lhs: &CudaDenseStorage,
    lhs_offset: usize,
    rhs: &CudaDenseStorage,
    rhs_offset: usize,
    rows: usize,
    contracted: usize,
    cols: usize,
) -> Result<(), DenseError> {
    cuda_gemm_region_into::<D>(
        ctx,
        dst,
        dst_offset,
        rows,
        lhs,
        lhs_offset,
        rows,
        rhs,
        rhs_offset,
        contracted,
        rows,
        contracted,
        cols,
        D::ONE,
        D::ZERO,
    )
}
/// General column-major GEMM over device buffer regions with explicit
/// per-operand offsets and leading dimensions, plus scaling:
/// `dst_region[m x n] = alpha * lhs_region[m x k] * rhs_region[k x n]
///  + beta * dst_region`.
///
/// This is the single device seam the user layer builds everything
/// non-cuSOLVER on: sector inner products (`m = n = 1`), axpby via a `[1,1]`
/// ones operand (`k = n = 1`), and factor assembly through small selector
/// matrices (identity / prefix / sign / permutation).
#[allow(clippy::too_many_arguments)]
pub fn cuda_gemm_region_into<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    dst: &mut CudaDenseStorage,
    dst_offset: usize,
    dst_ld: usize,
    lhs: &CudaDenseStorage,
    lhs_offset: usize,
    lhs_ld: usize,
    rhs: &CudaDenseStorage,
    rhs_offset: usize,
    rhs_ld: usize,
    m: usize,
    k: usize,
    n: usize,
    alpha: D,
    beta: D,
) -> Result<(), DenseError> {
    cuda_gemm_region_strided_into::<D>(
        ctx,
        dst,
        dst_offset,
        dst_ld,
        lhs,
        lhs_offset,
        [1, lhs_ld],
        false,
        rhs,
        rhs_offset,
        [1, rhs_ld],
        false,
        m,
        k,
        n,
        alpha,
        beta,
    )
}
/// GEMM over logical matrix views of packed parent regions. `Adjoint` changes
/// the two strides and carries conjugation metadata without creating a
/// transposed or conjugated payload: for a complex payload the conjugation is
/// a backend operand flag, and for `f64` it is a no-op.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn cuda_gemm_region_with_ops_into<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    dst: &mut CudaDenseStorage,
    dst_offset: usize,
    lhs: &CudaDenseStorage,
    lhs_offset: usize,
    rhs: &CudaDenseStorage,
    rhs_offset: usize,
    m: usize,
    k: usize,
    n: usize,
    lhs_op: MatrixOp,
    rhs_op: MatrixOp,
    alpha: D,
    beta: D,
) -> Result<(), DenseError> {
    let (lhs_strides, lhs_conj) = cuda_operand_view(lhs_op, m, k);
    let (rhs_strides, rhs_conj) = cuda_operand_view(rhs_op, k, n);
    cuda_gemm_region_strided_into::<D>(
        ctx,
        dst,
        dst_offset,
        m,
        lhs,
        lhs_offset,
        lhs_strides,
        lhs_conj,
        rhs,
        rhs_offset,
        rhs_strides,
        rhs_conj,
        m,
        k,
        n,
        alpha,
        beta,
    )
}
#[allow(clippy::too_many_arguments)]
fn cuda_gemm_region_strided_into<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    dst: &mut CudaDenseStorage,
    dst_offset: usize,
    dst_ld: usize,
    lhs: &CudaDenseStorage,
    lhs_offset: usize,
    lhs_strides: [usize; 2],
    lhs_conj: bool,
    rhs: &CudaDenseStorage,
    rhs_offset: usize,
    rhs_strides: [usize; 2],
    rhs_conj: bool,
    m: usize,
    k: usize,
    n: usize,
    alpha: D,
    beta: D,
) -> Result<(), DenseError> {
    ensure_cuda_device(
        ctx.device,
        "cuda_matmul",
        &[
            ("dst", dst.device),
            ("lhs", lhs.device),
            ("rhs", rhs.device),
        ],
    )?;
    let lhs_view = lhs.region_view_strided::<D>([m, k], lhs_strides, lhs_offset)?;
    let rhs_view = rhs.region_view_strided::<D>([k, n], rhs_strides, rhs_offset)?;
    let dst_view = dst.region_view_mut::<D>(m, n, dst_ld, dst_offset)?;
    let config = DotGeneralConfig {
        lhs_contracting_dims: [1].into_iter().collect(),
        rhs_contracting_dims: [0].into_iter().collect(),
        lhs_batch_dims: Default::default(),
        rhs_batch_dims: Default::default(),
    };
    let accumulation = DotGeneralAccumulation {
        lhs_conj,
        rhs_conj,
        alpha: alpha.contraction_scalar(),
        beta: beta.contraction_scalar(),
    };
    record(|stats| stats.gemm_calls += 1);
    ctx.backend
        .dot_general_read_into_accum(
            TensorRead::from_view(lhs_view),
            TensorRead::from_view(rhs_view),
            &config,
            accumulation,
            TensorWrite::from_view(dst_view),
        )
        .map_err(|err| cuda_error("cuda_matmul", err))
}
/// `members` independent column-major GEMMs at a fixed member stride per
/// operand, as ONE `dot_general` with a trailing batch mode:
/// `dst[dst_offset + i·dst_member_stride][m x n] = lhs[lhs_offset +
/// i·lhs_member_stride][m x k] · rhs[rhs_offset + i·rhs_member_stride][k x n]`
/// for `i < members` (overwrite, `beta = 0`).
///
/// The views are `[m, k, B]`, `[k, n, B]` and `[m, n, B]` with the member
/// stride on the last axis; Tenferro orders the output modes `[lhs_free,
/// rhs_free, batch]`, which is the destination view. One submission, and one
/// `gemm_calls` count, whatever `members` is. The destination must be
/// injective (members may not overlap), and every region must lie inside its
/// buffer's active length; both are checked before any device work.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn cuda_gemm_region_batched_into<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    dst: &mut CudaDenseStorage,
    dst_offset: usize,
    dst_member_stride: usize,
    lhs: &CudaDenseStorage,
    lhs_offset: usize,
    lhs_member_stride: usize,
    rhs: &CudaDenseStorage,
    rhs_offset: usize,
    rhs_member_stride: usize,
    m: usize,
    k: usize,
    n: usize,
    members: usize,
) -> Result<(), DenseError> {
    cuda_gemm_region_batched_scaled_into::<D>(
        ctx,
        dst,
        dst_offset,
        dst_member_stride,
        lhs,
        lhs_offset,
        lhs_member_stride,
        rhs,
        rhs_offset,
        rhs_member_stride,
        m,
        k,
        n,
        members,
        D::ONE,
    )
}

/// Batched region GEMM with one common alpha and overwrite semantics.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn cuda_gemm_region_batched_scaled_into<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    dst: &mut CudaDenseStorage,
    dst_offset: usize,
    dst_member_stride: usize,
    lhs: &CudaDenseStorage,
    lhs_offset: usize,
    lhs_member_stride: usize,
    rhs: &CudaDenseStorage,
    rhs_offset: usize,
    rhs_member_stride: usize,
    m: usize,
    k: usize,
    n: usize,
    members: usize,
    alpha: D,
) -> Result<(), DenseError> {
    const OP: &str = "cuda_gemm_batched";
    ensure_cuda_device(
        ctx.device,
        OP,
        &[
            ("dst", dst.device),
            ("lhs", lhs.device),
            ("rhs", rhs.device),
        ],
    )?;
    let dst_region = CudaRegion::new(
        vec![m, n, members],
        vec![1, m, dst_member_stride],
        dst_offset,
    )?;
    let lhs_region = CudaRegion::new(
        vec![m, k, members],
        vec![1, m, lhs_member_stride],
        lhs_offset,
    )?;
    let rhs_region = CudaRegion::new(
        vec![k, n, members],
        vec![1, k, rhs_member_stride],
        rhs_offset,
    )?;
    if !dst_region.is_empty() {
        validate_destination_layout(OP, &dst_region)?;
        validate_region(&dst_region, dst.len)?;
    }
    for (region, len) in [(&lhs_region, lhs.len), (&rhs_region, rhs.len)] {
        if !region.is_empty() {
            validate_region(region, len)?;
        }
    }
    let lhs_view = lhs.region_view_nd::<D>(
        lhs_region.dims(),
        &isize_strides(lhs_region.strides())?,
        lhs_region.offset_isize()?,
    )?;
    let rhs_view = rhs.region_view_nd::<D>(
        rhs_region.dims(),
        &isize_strides(rhs_region.strides())?,
        rhs_region.offset_isize()?,
    )?;
    let dst_view = dst.region_view_nd_mut::<D>(
        dst_region.dims(),
        &isize_strides(dst_region.strides())?,
        dst_region.offset_isize()?,
    )?;
    let config = DotGeneralConfig {
        lhs_contracting_dims: [1].into_iter().collect(),
        rhs_contracting_dims: [0].into_iter().collect(),
        lhs_batch_dims: [2].into_iter().collect(),
        rhs_batch_dims: [2].into_iter().collect(),
    };
    let accumulation = DotGeneralAccumulation {
        lhs_conj: false,
        rhs_conj: false,
        alpha: alpha.contraction_scalar(),
        beta: D::ZERO.contraction_scalar(),
    };
    record(|stats| stats.gemm_calls += 1);
    ctx.backend
        .dot_general_read_into_accum(
            TensorRead::from_view(lhs_view),
            TensorRead::from_view(rhs_view),
            &config,
            accumulation,
            TensorWrite::from_view(dst_view),
        )
        .map_err(|err| cuda_error(OP, err))
}
