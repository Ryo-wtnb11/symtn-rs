use super::*;

/// Uploads a rank-0 operand in the payload's **real** lane.
///
/// The lane is not a symmetry: Tenferro rejects an `F64` rank-0 operand
/// against an `F32` or `C32` tensor as a dtype mismatch, and rejects a
/// *payload*-typed complex operand as a shape mismatch, so "complex tensor
/// with a real rank-0 scalar" is the one production shape (probe
/// `cuda-single-precision-probe-2026-09-20.md`, finding 3).
fn upload_scalar<R: CudaRealScalar>(
    ctx: &CudaDenseContext,
    op: &'static str,
    value: f64,
) -> Result<Tensor, DenseError> {
    let host = R::into_tensor(vec![], vec![R::narrow(value)]).map_err(|err| cuda_error(op, err))?;
    let tensor = upload_tensor(&ctx.runtime, &host).map_err(|err| cuda_error(op, err))?;
    record_h2d(std::mem::size_of::<R>());
    Ok(tensor)
}
/// The rank-0 lane operand through which [`scale_by_power_of_two`] multiplies
/// a payload by the exact power of two `normalizer`.
fn power_of_two_operand<D: CudaScalar>(
    ctx: &CudaDenseContext,
    op: &'static str,
    normalizer: f64,
) -> Result<Tensor, DenseError> {
    let operand = if D::IS_COMPLEX {
        normalizer
    } else {
        // Exact: the reciprocal of a power of two inside the lane's range.
        normalizer.recip()
    };
    upload_scalar::<D::Real>(ctx, op, operand)
}
/// Multiplies a payload tensor by the power of two that `operand` encodes.
fn scale_by_power_of_two<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    op: &'static str,
    tensor: &Tensor,
    operand: &Tensor,
) -> Result<Tensor, DenseError> {
    if D::IS_COMPLEX {
        ctx.backend.mul_read(
            TensorRead::from_tensor(tensor),
            TensorRead::from_tensor(operand),
        )
    } else {
        // Why not `mul`: Tenferro 0.7.1's real `mul` does not broadcast a
        // rank-0 operand, while its real `div` does and, unlike the
        // complex-by-real one, never squares the divisor.
        ctx.backend.div_read(
            TensorRead::from_tensor(tensor),
            TensorRead::from_tensor(operand),
        )
    }
    .map_err(|err| cuda_error(op, err))
}
/// Real magnitudes of a device tensor, so the real-only sum-of-squares
/// reduction sees `|z|^2` for a complex payload. `abs` leaves an `f64` tensor
/// numerically unchanged under that reduction, so the extra pass is skipped
/// there on the dtype invariant.
fn magnitudes_for_sum_squares<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    op: &'static str,
    tensor: &Tensor,
) -> Result<Option<Tensor>, DenseError> {
    if !D::IS_COMPLEX {
        return Ok(None);
    }
    ctx.backend
        .abs_read(TensorRead::from_tensor(tensor))
        .map(Some)
        .map_err(|err| cuda_error(op, err))
}
/// Tests packed square CUDA regions `(offset, n)` of `src` with the host EIGH
/// rule `||(A - A^H)/2||_F <= relative_tolerance * ||A||_F`, returning one
/// decision per region. The residual uses the *conjugate* transpose, so a
/// complex-symmetric non-Hermitian block is rejected.
///
/// `relative_tolerance` is the caller's resolved eigh admission tolerance
/// (symtn's `HermitianTol` at the payload's real-lane epsilon), the same one
/// the host twin `normwise_hermitian` applies.
/// The caller must supply a finite, non-negative value.
///
/// A complex entry whose modulus overflows the lane is rejected here (the
/// `hypot`-based `abs` gives infinity) but admitted by the component-scaled
/// host twin. That disagreement is on the safe side and predates the power
/// of two normalizer; it is why the device tests' scale window stops at
/// `2^(MAX_EXP - 4)`.
///
/// This is the one-member case of the pipeline [`admit_regions`] runs for
/// [`cuda_hermitian_regions_batched`] too; see there for the stages, their
/// download bound and their working set.
///
/// Only scalar norm metadata is downloaded; no region is copied to the host.
#[doc(hidden)]
pub fn cuda_hermitian_regions<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
    regions: &[(usize, usize)],
    relative_tolerance: f64,
) -> Result<Vec<bool>, DenseError> {
    let members = Members::One;
    ensure_cuda_device(ctx.device, members.op(), &[("src", src.device)])?;
    admit_regions::<D>(ctx, src, regions, members, relative_tolerance)
}
/// [`cuda_hermitian_regions`] over `members` stacked copies of the same
/// regions, member `b` of region `(offset, n)` at `offset + b *
/// member_stride`. Returns one decision per (member, region), member-major:
/// `decisions[b * regions.len() + r]`. Zero members is an empty stack: no
/// decisions, no transfers (after the complex check below).
/// The caller must supply a finite, non-negative `relative_tolerance`.
///
/// Each (member, region) pair is decided by exactly the rule and the three
/// scalar stages of [`cuda_hermitian_regions`] — the same [`admit_regions`]
/// pipeline — with its own power-of-two normalizers: members of one batch may
/// differ by many orders of magnitude, and a normalizer shared across members
/// would change the acceptance of the small ones.
///
/// Real payloads only: a complex payload returns [`DenseError::Unsupported`].
/// Why: Tenferro 0.7.1 scales a complex tensor by a real operand only when the
/// operand is rank 0, so a per-member real normalizer has no complex form.
#[doc(hidden)]
pub fn cuda_hermitian_regions_batched<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
    regions: &[(usize, usize)],
    members: usize,
    member_stride: usize,
    relative_tolerance: f64,
) -> Result<Vec<bool>, DenseError> {
    let members = Members::Stack {
        count: members,
        stride: member_stride,
    };
    ensure_cuda_device(ctx.device, members.op(), &[("src", src.device)])?;
    if D::IS_COMPLEX {
        return Err(DenseError::Unsupported {
            op: members.op(),
            message: "per-member normalizers of a complex payload".into(),
        });
    }
    if members.count() == 0 {
        // An empty stack has no (member, region) pair to decide, as an empty
        // `regions` list has none: the empty result, with no device work.
        return Ok(Vec::new());
    }
    admit_regions::<D>(ctx, src, regions, members, relative_tolerance)
}
/// How the admitted regions are stored: one copy each, or `count` copies
/// `stride` elements apart.
///
/// Why two materializations rather than a one-member stack: each entry keeps
/// its own bound validation (`check_matrix_bound` for a single region,
/// `validate_region` for a stack) and so its own error variants.
#[derive(Clone, Copy)]
enum Members {
    One,
    Stack { count: usize, stride: usize },
}
impl Members {
    fn op(self) -> &'static str {
        match self {
            Members::One => "cuda_hermitian",
            Members::Stack { .. } => "cuda_hermitian_batched",
        }
    }
    fn count(self) -> usize {
        match self {
            Members::One => 1,
            Members::Stack { count, .. } => count,
        }
    }
    /// The `n x n` region at `offset` with element strides `strides`, every
    /// member of it, as one compact `[n, n, count]` device tensor.
    fn materialize<D: CudaScalar>(
        self,
        ctx: &mut CudaDenseContext,
        src: &CudaDenseStorage,
        n: usize,
        strides: [usize; 2],
        offset: usize,
    ) -> Result<Tensor, DenseError> {
        match self {
            Members::One => contiguous_square::<D>(ctx, src, n, strides, offset),
            Members::Stack { count, stride } => {
                contiguous_stack::<D>(ctx, src, n, strides, offset, count, stride)
            }
        }
    }
}
/// The admission pipeline of both entries: one decision per (member,
/// region), member-major.
///
/// Each pair's rule needs three dependent scalar stages: its maximum picks
/// the input normalizer, the residual maximum picks the residual normalizer,
/// and the residual sum of squares decides ([`input_stage`],
/// [`residual_stage`], [`scaled_hermitian_residual_accepts`]). Every stage
/// runs over all still-undecided regions before its scalars are gathered with
/// one `concatenate` and downloaded together, so a call makes at most three
/// downloads whatever the region or member count. The member axis is a
/// trailing mode of every device tensor, so the submissions per region do not
/// depend on the member count. Why not per region: each download blocks the
/// host (a D2H plus a CubeCL flush), which made admission the largest
/// symtn-owned cost of `eigh` (#1483). The price is that the materialized
/// input, then residual, of every undecided region is live across a stage
/// boundary: one extra copy of the regions' payload, the same order as the
/// eigenvector factor `eigh` allocates anyway.
///
/// Reference: the rule is MatrixAlgebraKit 0.6.8 (`33d77fdf`)
/// `src/implementations/eigh.jl:check_hermitian` ->
/// `src/common/matrixproperties.jl:strided_ishermitian_approx`, reached per
/// block from TensorKit `cfaa073e` `src/factorizations/matrixalgebrakit.jl`
/// (`eigh_full!` over `foreachblock`). symtn keeps its approved relative
/// tolerance instead of MAK's absolute `default_hermitian_tol` (#1987). QSpace
/// `dd2cc7e1` `Source/wbarray_blas.cc:wbEigenS` ->
/// `Source/wbarray.hh:isHConj` -> `Source/wbarray.cc:wbarray::isSym_aux`
/// checks entrywise and is not ported.
/// Neither reference has a device or member-stacked admission path: the
/// stages, normalizers and member axis are symtn's.
fn admit_regions<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
    regions: &[(usize, usize)],
    members: Members,
    relative_tolerance: f64,
) -> Result<Vec<bool>, DenseError> {
    let op = members.op();
    let count = members.count();
    let max_exp = <D::Real as CudaRealScalar>::MAX_EXP;
    let mut accepted = vec![true; regions.len() * count];
    let slot = |member: usize, region: usize| member * regions.len() + region;

    // Stage 1: max |A| per member.
    let mut inputs = Vec::new();
    let mut maxima = Vec::new();
    for (index, &(offset, n)) in regions.iter().enumerate() {
        if n == 0 {
            continue;
        }
        let normal = members.materialize::<D>(ctx, src, n, [1, n], offset)?;
        let input_abs = ctx
            .backend
            .abs_read(TensorRead::from_tensor(&normal))
            .map_err(|err| cuda_error(op, err))?;
        maxima.push(
            ctx.backend
                .reduce_max_read(TensorRead::from_tensor(&input_abs), &[0, 1])
                .map_err(|err| cuda_error(op, err))?,
        );
        inputs.push((index, normal));
    }
    let input_scales = download_concatenated::<D::Real>(ctx, &maxima, count, op)?;
    drop(maxima);

    // Stage 2: sum |A/s|^2 and max |(A - A^H)/s| per member.
    let mut residuals = Vec::new();
    let mut input_sums = Vec::new();
    let mut residual_maxima = Vec::new();
    for ((index, normal), scales) in inputs.into_iter().zip(input_scales.chunks(count)) {
        let normalizers: Vec<Option<f64>> = scales
            .iter()
            .enumerate()
            .map(|(member, &scale)| match input_stage(scale, max_exp) {
                StageOutcome::Decided(decision) => {
                    accepted[slot(member, index)] = decision;
                    None
                }
                StageOutcome::Normalize(normalizer) => Some(normalizer),
            })
            .collect();
        if normalizers.iter().all(Option::is_none) {
            continue;
        }
        let (offset, n) = regions[index];
        let transpose = members.materialize::<D>(ctx, src, n, [n, 1], offset)?;
        // Conjugation is a backend op on the transposed copy, never a symtn loop.
        let transpose = if D::IS_COMPLEX {
            ctx.backend
                .conj_read(TensorRead::from_tensor(&transpose))
                .map_err(|err| cuda_error(op, err))?
        } else {
            transpose
        };
        let scale = MemberScale::upload::<D>(ctx, op, &normalizers, n)?;
        let normal_scaled = scale.apply::<D>(ctx, op, &normal)?;
        drop(normal);
        let transpose_scaled = scale.apply::<D>(ctx, op, &transpose)?;
        input_sums.push(sum_of_squares::<D>(ctx, op, &normal_scaled)?);
        let residual = ctx
            .backend
            .sub_read(
                TensorRead::from_tensor(&normal_scaled),
                TensorRead::from_tensor(&transpose_scaled),
            )
            .map_err(|err| cuda_error(op, err))?;
        let residual_abs = ctx
            .backend
            .abs_read(TensorRead::from_tensor(&residual))
            .map_err(|err| cuda_error(op, err))?;
        residual_maxima.push(
            ctx.backend
                .reduce_max_read(TensorRead::from_tensor(&residual_abs), &[0, 1])
                .map_err(|err| cuda_error(op, err))?,
        );
        residuals.push((index, residual, normalizers));
    }
    let stage2_regions = residuals.len();
    input_sums.append(&mut residual_maxima);
    let stage2 = download_concatenated::<D::Real>(ctx, &input_sums, count, op)?;
    drop(input_sums);
    let (input_ss, residual_scales) = stage2.split_at(stage2_regions * count);

    // Stage 3: sum |R/r|^2 per member.
    let mut undecided = Vec::new();
    let mut residual_sums = Vec::new();
    for (((index, residual, input_normalizers), input_ss), residual_scales) in residuals
        .into_iter()
        .zip(input_ss.chunks(count))
        .zip(residual_scales.chunks(count))
    {
        let mut pending = Vec::new();
        let normalizers: Vec<Option<f64>> = input_normalizers
            .iter()
            .enumerate()
            .map(|(member, input_normalizer)| {
                // A member decided by stage 1 has no residual to read.
                (*input_normalizer)?;
                match residual_stage(input_ss[member], residual_scales[member], max_exp) {
                    StageOutcome::Decided(decision) => {
                        accepted[slot(member, index)] = decision;
                        None
                    }
                    StageOutcome::Normalize(normalizer) => {
                        pending.push((member, input_ss[member], normalizer));
                        Some(normalizer)
                    }
                }
            })
            .collect();
        if pending.is_empty() {
            continue;
        }
        let (_, n) = regions[index];
        let scale = MemberScale::upload::<D>(ctx, op, &normalizers, n)?;
        let residual_normalized = scale.apply::<D>(ctx, op, &residual)?;
        residual_sums.push(sum_of_squares::<D>(ctx, op, &residual_normalized)?);
        undecided.push((index, pending));
    }
    let residual_ss = download_concatenated::<D::Real>(ctx, &residual_sums, count, op)?;
    for ((index, pending), residual_ss) in undecided.into_iter().zip(residual_ss.chunks(count)) {
        for (member, input_ss, residual_normalizer) in pending {
            accepted[slot(member, index)] = scaled_hermitian_residual_accepts(
                input_ss,
                // Exact: the reciprocal of a normal power of two.
                residual_normalizer.recip(),
                residual_ss[member],
                relative_tolerance,
            );
        }
    }
    Ok(accepted)
}
/// The device operand that multiplies member `b` of an `[n, n, members]`
/// payload by its exact power-of-two normalizer; a member the stage already
/// decided (`None`) is multiplied by one and its reductions are ignored.
enum MemberScale {
    /// One member: a rank-0 lane operand. Also the only form a complex
    /// payload has (see [`cuda_hermitian_regions_batched`]), and one
    /// broadcast kernel fewer than the per-member form.
    Scalar(Tensor),
    /// One real divisor per member, broadcast over the matrix modes.
    PerMember(Tensor),
}
impl MemberScale {
    fn upload<D: CudaScalar>(
        ctx: &mut CudaDenseContext,
        op: &'static str,
        normalizers: &[Option<f64>],
        n: usize,
    ) -> Result<Self, DenseError> {
        let normalizer = |value: &Option<f64>| value.unwrap_or(1.0);
        match normalizers {
            [single] => power_of_two_operand::<D>(ctx, op, normalizer(single)).map(Self::Scalar),
            _ => {
                // Exact: the reciprocal of a power of two inside the lane.
                let divisors: Vec<f64> = normalizers
                    .iter()
                    .map(|value| normalizer(value).recip())
                    .collect();
                broadcast_member_divisors::<D>(ctx, &divisors, n, op).map(Self::PerMember)
            }
        }
    }
    fn apply<D: CudaScalar>(
        &self,
        ctx: &mut CudaDenseContext,
        op: &'static str,
        tensor: &Tensor,
    ) -> Result<Tensor, DenseError> {
        match self {
            Self::Scalar(operand) => scale_by_power_of_two::<D>(ctx, op, tensor, operand),
            Self::PerMember(divisor) => ctx
                .backend
                .div_read(
                    TensorRead::from_tensor(tensor),
                    TensorRead::from_tensor(divisor),
                )
                .map_err(|err| cuda_error(op, err)),
        }
    }
}
/// Materializes `members` stacked `n x n` regions (element strides `strides`
/// within a member, `member_stride` between members) as one compact
/// `[n, n, members]` device tensor.
fn contiguous_stack<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
    n: usize,
    strides: [usize; 2],
    offset: usize,
    members: usize,
    member_stride: usize,
) -> Result<Tensor, DenseError> {
    const OP: &str = "cuda_hermitian_batched";
    let region = CudaRegion::new(
        vec![n, n, members],
        vec![strides[0], strides[1], member_stride],
        offset,
    )?;
    validate_region(&region, src.len)?;
    let view = src.region_view_nd::<D>(
        region.dims(),
        &isize_strides(region.strides())?,
        region.offset_isize()?,
    )?;
    ctx.backend
        .to_contiguous_read(TensorRead::from_view(view))
        .map_err(|err| cuda_error(OP, err))
}
/// Uploads one real divisor per member and broadcasts it over the matrix
/// modes of an `[n, n, members]` payload, so one elementwise `div` scales
/// every member by its own power of two.
///
/// Why broadcast rather than a rank-0 operand per member: Tenferro 0.7.1's
/// elementwise ops take equal shapes or a rank-0 operand, so a per-member
/// divisor needs the full shape; the broadcast is one device kernel, and the
/// upload is `members` real values.
fn broadcast_member_divisors<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    divisors: &[f64],
    n: usize,
    op: &'static str,
) -> Result<Tensor, DenseError> {
    let values: Vec<D::Real> = divisors
        .iter()
        .map(|&value| <D::Real as CudaRealScalar>::narrow(value))
        .collect();
    let host =
        D::Real::into_tensor(vec![divisors.len()], values).map_err(|err| cuda_error(op, err))?;
    let tensor = upload_tensor(&ctx.runtime, &host).map_err(|err| cuda_error(op, err))?;
    record_h2d(divisors.len() * std::mem::size_of::<D::Real>());
    ctx.backend
        .broadcast_in_dim_read(
            TensorRead::from_tensor(&tensor),
            &[n, n, divisors.len()],
            &[2],
        )
        .map_err(|err| cuda_error(op, err))
}
/// Downloads rank-1 real reductions of `members` values each with one
/// transfer (concatenated on device first). No parts means no transfer.
fn download_concatenated<R: CudaRealScalar>(
    ctx: &mut CudaDenseContext,
    parts: &[Tensor],
    members: usize,
    op: &'static str,
) -> Result<Vec<f64>, DenseError> {
    if parts.is_empty() {
        return Ok(Vec::new());
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    let gathered = ctx
        .backend
        .concatenate(&refs, 0)
        .map_err(|err| cuda_error(op, err))?;
    let values = download_values::<R>(ctx, &gathered)?;
    if values.len() != parts.len() * members {
        return Err(cuda_error(
            op,
            format!(
                "device reductions returned {} values; expected {}",
                values.len(),
                parts.len() * members
            ),
        ));
    }
    Ok(values)
}
/// Materializes the `n x n` region at `offset` with element strides
/// `strides` as a compact `[n, n, 1]` device tensor.
///
/// Why the trailing unit axis: a reduction over axes `[0, 1]` then yields a
/// rank-1 `[1]` tensor, which Tenferro's `concatenate` accepts, whereas the
/// rank-0 result of a rank-2 input would need a reshape copy per scalar.
fn contiguous_square<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
    n: usize,
    strides: [usize; 2],
    offset: usize,
) -> Result<Tensor, DenseError> {
    const OP: &str = "cuda_hermitian";
    src.check_matrix_bound([n, n], strides, offset)?;
    let to_isize =
        |value: usize| isize::try_from(value).map_err(|_| cuda_error(OP, "index exceeds isize"));
    let view = src.region_view_nd::<D>(
        &[n, n, 1],
        &[to_isize(strides[0])?, to_isize(strides[1])?, 1],
        to_isize(offset)?,
    )?;
    ctx.backend
        .to_contiguous_read(TensorRead::from_view(view))
        .map_err(|err| cuda_error(OP, err))
}
/// `sum |x|^2` over axes `[0, 1]` of a `[n, n, 1]` payload tensor, in its
/// real lane.
fn sum_of_squares<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    op: &'static str,
    tensor: &Tensor,
) -> Result<Tensor, DenseError> {
    let magnitudes = magnitudes_for_sum_squares::<D>(ctx, op, tensor)?;
    ctx.backend
        .reduce_sum_squares_read(
            TensorRead::from_tensor(magnitudes.as_ref().unwrap_or(tensor)),
            &[0, 1],
        )
        .map_err(|err| cuda_error(op, err))
}
