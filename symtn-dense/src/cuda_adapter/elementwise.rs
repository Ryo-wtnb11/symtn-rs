use super::*;

/// How a region call treats the destination it writes: the `beta` of
/// `dst_region = coefficient * src_region + beta * dst_region`, carried in the
/// same `dot_general` epilogue as the move itself.
///
/// [`Self::Overwrite`] (`beta = 0`) never reads the destination, so a NaN there
/// does not survive. [`Self::Scale`] is a general `beta`; `Scale(0)` and
/// `Scale(1)` are accepted and behave as the two named variants.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CudaRegionBeta<D> {
    /// `dst_region = coefficient * src_region` (`beta = 0`).
    Overwrite,
    /// `dst_region += coefficient * src_region` (`beta = 1`).
    Accumulate,
    /// `dst_region = coefficient * src_region + beta * dst_region`.
    Scale(D),
}
impl<D: CudaScalar> CudaRegionBeta<D> {
    fn scalar(self) -> ContractionScalar {
        match self {
            Self::Overwrite => D::ZERO.contraction_scalar(),
            Self::Accumulate => D::ONE.contraction_scalar(),
            Self::Scale(beta) => beta.contraction_scalar(),
        }
    }
}
/// The payload dtype of an operand, checked before any device work rather
/// than at view construction, so a mismatched call cannot have uploaded a
/// lazily created scalar operand first.
pub(super) fn ensure_payload_dtype<D: CudaScalar>(
    op: &'static str,
    storage: &CudaDenseStorage,
) -> Result<(), DenseError> {
    let actual = storage.dtype();
    if actual != D::DTYPE {
        return Err(DenseError::DTypeMismatch {
            op,
            expected: D::DTYPE,
            actual,
        });
    }
    Ok(())
}
/// Rejects a descriptor scale of zero, which the backend is free to answer by
/// skipping the source read.
///
/// Every other scale is a plain multiplication, so the caller's NaN and
/// infinities survive it; a zero one is the single value whose backend
/// behaviour does not match `alpha * src`, and silently answering it with a
/// cleared destination is a wrong answer rather than a slow one. The zero
/// scale is expressible — as an exact zero *data* operand — so this is a
/// misuse of the argument, not a missing capability.
pub(super) fn reject_zero_alpha<D: CudaScalar>(
    op: &'static str,
    alpha: D,
) -> Result<(), DenseError> {
    // IEEE comparison, so `-0.0` is rejected too: it skips the read just as
    // `0.0` does.
    if alpha != D::ZERO {
        return Ok(());
    }
    Err(DenseError::Unsupported {
        op,
        message: "a descriptor alpha of zero would let the backend skip the source read and                   erase NaN/Inf; pass alpha = 1 with CudaRegionCoefficient::Zero to multiply                   by an exact zero instead"
            .to_string(),
    })
}
/// Submits one validated region move. Takes the backend rather than the whole
/// context so a caller can pass a coefficient the context itself owns.
#[allow(clippy::too_many_arguments)]
pub(super) fn submit_region_axpby<D: CudaScalar>(
    backend: &mut dyn Engine,
    op: &'static str,
    src: &CudaDenseStorage,
    src_region: &CudaRegion,
    conj: bool,
    alpha: D,
    coeff: &CudaDenseStorage,
    coeff_offset: usize,
    beta: CudaRegionBeta<D>,
    dst: &mut CudaDenseStorage,
    dst_region: &CudaRegion,
) -> Result<(), DenseError> {
    let rank = src_region.dims().len();
    let (view_dims, src_strides) = src_region.contraction_view_metadata()?;
    let (_, dst_strides) = dst_region.contraction_view_metadata()?;
    let coeff_offset = isize::try_from(coeff_offset).map_err(|_| DenseError::OffsetOverflow {
        value: coeff_offset,
    })?;

    let lhs = src.region_view_nd::<D>(&view_dims, &src_strides, src_region.offset_isize()?)?;
    let rhs = coeff.region_view_nd::<D>(&[1, 1], &[1, 1], coeff_offset)?;
    let out = dst.region_view_nd_mut::<D>(&view_dims, &dst_strides, dst_region.offset_isize()?)?;

    let config = DotGeneralConfig {
        lhs_contracting_dims: [rank].into_iter().collect(),
        rhs_contracting_dims: [0].into_iter().collect(),
        lhs_batch_dims: Default::default(),
        rhs_batch_dims: Default::default(),
    };
    // The structural coefficient is the 1x1 *operand*, never the descriptor
    // alpha: a descriptor alpha of 0 lets CUDA skip the source read and erase
    // NaN/Inf, where the host multiplies and propagates it. The caller's own
    // scale rides the descriptor, and a caller whose scale is zero passes it
    // as the zero *operand* instead ([`CudaRegionCoefficient::Zero`]) for the
    // same reason.
    let accumulation = DotGeneralAccumulation {
        lhs_conj: conj,
        rhs_conj: false,
        alpha: alpha.contraction_scalar(),
        beta: beta.scalar(),
    };
    record(|stats| stats.gemm_calls += 1);
    backend
        .dot_general_read_into_accum(
            TensorRead::from_view(lhs),
            TensorRead::from_view(rhs),
            &config,
            accumulation,
            TensorWrite::from_view(out),
        )
        .map_err(|err| cuda_error(op, err))
}
/// `dst_region = alpha * c * [conj] src_region + beta * dst_region`, where `c`
/// is a 1x1 **data** operand chosen by `coeff` and `alpha` is the caller's own
/// scale, carried by the contraction descriptor.
///
/// This is the single strided data-movement primitive the device structural
/// operations are built on: one strided, offset, rank-N source region of one
/// buffer moved into a strided, offset, rank-N destination region of another,
/// with the axis permutation carried by the destination strides. It submits
/// one `dot_general` against the 1x1 coefficient and allocates no device
/// buffer of its own.
///
/// Transfer contract: a call with [`CudaRegionCoefficient::Buffer`] moves
/// nothing across the host boundary, ever. A call with
/// [`CudaRegionCoefficient::One`] uploads this context's one-element `1` the
/// first time that dtype is used, and a call with
/// [`CudaRegionCoefficient::Zero`] its one-element zero template, and nothing
/// afterwards (see [`CudaDenseContext::scalar_operand_bytes`] and
/// [`CudaDenseContext::reserve_zero_template`]).
///
/// The structural coefficient is a data operand and never the descriptor
/// alpha, because a descriptor alpha lets CUDA skip the source read when it is
/// 0 and so erases NaN/Inf that the host propagates. `alpha` is the caller's
/// *own* scale, which the host applies as one more multiplication. A zero
/// `alpha` is therefore **rejected** (`Unsupported`, IEEE comparison so `-0.0`
/// too), before any device work: pass `alpha = D::ONE` with
/// [`CudaRegionCoefficient::Zero`] to multiply by an exact zero and keep the
/// host's NaN/Inf propagation. Passing `alpha = D::ONE` is the unscaled default
/// this primitive had before it carried a caller scale.
///
/// Disclosed differences from a host `alpha * (c * x)` chain: this path rounds
/// as `alpha * (c * x)` where a host that folds the two scales first rounds as
/// `(alpha * c) * x` (one rounding position, values equal to dtype tolerance,
/// and either order overflows where the other need not); and with
/// [`CudaRegionCoefficient::Zero`] the written zeros carry the sign of
/// `0 * src` alone, not the sign of the caller's `-0.0` or of `c`.
///
/// One numerical deviation from the host is disclosed and pinned by the device
/// tests: the host copies bit-exactly when the coefficient is 1, while this
/// path always multiplies, and an infinite complex payload is observed to come
/// back as `NaN` in both components: `inf * 0` in the complex product already
/// yields a `NaN` component, which the remaining multiply spreads across both.
/// `f64` infinities and every finite payload are unaffected. See
/// `benchmarks/history/cuda-region-axpby-2026-09-20.md`.
///
/// Validation order, all of it before any device work:
///
/// 0. the descriptor scale is not zero (`Unsupported`);
/// 1. every operand is on the context's device;
/// 2. every operand has payload dtype `D` (`DTypeMismatch`);
/// 3. source and destination `dims` are equal (`ShapeMismatch`);
/// 4. the coefficient offset is inside `coeff` (`OutOfBounds`);
/// 5. the element count does not overflow (`ElementCountOverflow`) — a
///    zero-extent region returns `Ok` here, with no submission;
/// 6. the destination layout is injective (`Unsupported`);
/// 7. both regions stay inside their buffers and fit `isize`
///    (`OutOfBounds` / `OffsetOverflow` / `StrideOverflow`).
///
/// Two further constraints are type boundaries rather than checks: strides
/// cannot be negative ([`CudaRegion`]), and `src`, `coeff` and `dst` cannot be
/// the same buffer, because [`CudaDenseStorage`] is neither `Clone` nor
/// refcounted, so a shared and an exclusive borrow of one buffer cannot
/// coexist. An in-place strided transform is therefore inexpressible here.
///
/// No rank limit is imposed: the probe accepted every mode count up to 72,
/// far above any rank a fusion tree reaches.
///
/// Context-owned operands (a context `1` or zero-template coefficient, or any
/// [`cuda_region_zero`]) belong to one context, which borrows one engine for
/// one operation, so they are created and read on that engine's stream.
#[allow(clippy::too_many_arguments)]
pub fn cuda_region_axpby<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
    src_region: &CudaRegion,
    conj: bool,
    alpha: D,
    coeff: CudaRegionCoefficient<'_>,
    beta: CudaRegionBeta<D>,
    dst: &mut CudaDenseStorage,
    dst_region: &CudaRegion,
) -> Result<(), DenseError> {
    const OP: &str = "cuda_region_axpby";
    reject_zero_alpha::<D>(OP, alpha)?;
    match coeff {
        CudaRegionCoefficient::Buffer(coeff, _) => ensure_cuda_device(
            ctx.device,
            OP,
            &[
                ("src", src.device),
                ("coefficient", coeff.device),
                ("dst", dst.device),
            ],
        )?,
        CudaRegionCoefficient::One | CudaRegionCoefficient::Zero => {
            ensure_cuda_device(ctx.device, OP, &[("src", src.device), ("dst", dst.device)])?;
        }
    }
    ensure_payload_dtype::<D>(OP, src)?;
    ensure_payload_dtype::<D>(OP, dst)?;
    if let CudaRegionCoefficient::Buffer(coeff, _) = coeff {
        ensure_payload_dtype::<D>(OP, coeff)?;
    }
    if src_region.dims() != dst_region.dims() {
        return Err(DenseError::ShapeMismatch {
            op: OP,
            expected: dst_region.dims().to_vec(),
            actual: src_region.dims().to_vec(),
        });
    }
    if let CudaRegionCoefficient::Buffer(coeff, offset) = coeff {
        if offset >= coeff.len {
            return Err(DenseError::OutOfBounds);
        }
    }
    if src_region.is_empty() {
        return Ok(());
    }
    src_region.element_count()?;
    validate_destination_layout(OP, dst_region)?;
    validate_region(src_region, src.len)?;
    validate_region(dst_region, dst.len)?;

    match coeff {
        CudaRegionCoefficient::Buffer(coeff, offset) => submit_region_axpby::<D>(
            &mut *ctx.backend,
            OP,
            src,
            src_region,
            conj,
            alpha,
            coeff,
            offset,
            beta,
            dst,
            dst_region,
        ),
        CudaRegionCoefficient::One => {
            ctx.ensure_ones::<D>(1)?;
            let (backend, operands) = ctx.split_operands::<D>();
            let Some(ones) = operands.ones.as_ref() else {
                return Err(cuda_error(OP, "context scalar operand is missing"));
            };
            submit_region_axpby::<D>(
                backend, OP, src, src_region, conj, alpha, ones, 0, beta, dst, dst_region,
            )
        }
        CudaRegionCoefficient::Zero => {
            // Idempotent once the caller has reserved the template, which is
            // what keeps a warm replay upload-free; one element is all this
            // operand reads.
            ctx.ensure_zeros::<D>(1)?;
            let (backend, operands) = ctx.split_operands::<D>();
            let Some(zeros) = operands.zeros.as_ref() else {
                return Err(cuda_error(OP, "context scalar operand is missing"));
            };
            submit_region_axpby::<D>(
                backend, OP, src, src_region, conj, alpha, zeros, 0, beta, dst, dst_region,
            )
        }
    }
}
/// Where [`cuda_region_axpby`] reads its 1x1 coefficient operand.
///
/// The variants exist because the operand is data, not a descriptor scalar: a
/// caller that needs an exact `0` there — a caller scale of zero, which the
/// host still multiplies by — cannot express it as a descriptor alpha without
/// letting CUDA skip the source read.
#[derive(Clone, Copy)]
pub enum CudaRegionCoefficient<'a> {
    /// The context's shared `1`: an unscaled move, the default for a pure
    /// relayout.
    One,
    /// The first element of the context's zero template. Reserve it with
    /// [`CudaDenseContext::reserve_zero_template`] to keep the submission
    /// upload-free.
    Zero,
    /// A caller-owned device vector at an element offset — block `b` of a
    /// structure's uploaded coefficients is the 1x1 view at offset `b`.
    Buffer(&'a CudaDenseStorage, usize),
}
/// Writes zeros over `dst_region`, reading the context's zero template as a
/// packed source of the same extents.
///
/// This is the overwrite-mode destination rule expressed with the same
/// primitive: an inactive destination layout is zeroed rather than left
/// alone, so the result is independent of what the caller's buffer held —
/// including a NaN. It is a region move, not a fill kernel, so it inherits
/// exactly the validation of [`cuda_region_axpby`].
///
/// Transfer contract: it uploads the context's `1` and its zero template on
/// first use of a dtype, and re-uploads the template whenever a fill is longer
/// than the resident one. Size the template once with
/// [`CudaDenseContext::reserve_zero_template`] to make every later fill of a
/// replay transfer-free; otherwise a sequence of ascending fills pays one
/// upload each.
pub fn cuda_region_zero<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    dst: &mut CudaDenseStorage,
    dst_region: &CudaRegion,
) -> Result<(), DenseError> {
    zero_source_move::<D>(
        ctx,
        "cuda_region_zero",
        dst,
        dst_region,
        CudaRegionBeta::Overwrite,
    )
}
/// `dst_region = beta * dst_region`: the destination rule of a
/// beta-accumulating write for a region no source move writes.
///
/// Why not a scale kernel: Tenferro has no in-place strided scale, so this is
/// the [`cuda_region_zero`] move — the context zero template times the
/// context `1` — with `beta` in its `dot_general` epilogue, `0 + beta * dst`.
/// A zero `beta` is exactly [`cuda_region_zero`] (the destination is not
/// read); a unit `beta` submits nothing. Transfer contract and validation are
/// [`cuda_region_zero`]'s.
pub fn cuda_region_scale<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    dst: &mut CudaDenseStorage,
    dst_region: &CudaRegion,
    beta: D,
) -> Result<(), DenseError> {
    const OP: &str = "cuda_region_scale";
    if beta == D::ONE {
        ensure_cuda_device(ctx.device, OP, &[("dst", dst.device)])?;
        ensure_payload_dtype::<D>(OP, dst)?;
        return Ok(());
    }
    let beta = if beta == D::ZERO {
        CudaRegionBeta::Overwrite
    } else {
        CudaRegionBeta::Scale(beta)
    };
    zero_source_move::<D>(ctx, OP, dst, dst_region, beta)
}
fn zero_source_move<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    op: &'static str,
    dst: &mut CudaDenseStorage,
    dst_region: &CudaRegion,
    beta: CudaRegionBeta<D>,
) -> Result<(), DenseError> {
    ensure_cuda_device(ctx.device, op, &[("dst", dst.device)])?;
    ensure_payload_dtype::<D>(op, dst)?;
    if dst_region.is_empty() {
        return Ok(());
    }
    let count = dst_region.element_count()?;
    validate_destination_layout(op, dst_region)?;
    validate_region(dst_region, dst.len)?;
    let src_region = CudaRegion::packed(dst_region.dims(), 0)?;

    ctx.ensure_ones::<D>(1)?;
    ctx.ensure_zeros::<D>(count)?;
    let (backend, operands) = ctx.split_operands::<D>();
    let (Some(ones), Some(zeros)) = (operands.ones.as_ref(), operands.zeros.as_ref()) else {
        return Err(cuda_error(op, "context scalar operands are missing"));
    };
    submit_region_axpby::<D>(
        backend,
        op,
        zeros,
        &src_region,
        false,
        D::ONE,
        ones,
        0,
        beta,
        dst,
        dst_region,
    )
}
/// `dst_region += alpha * [conj] trace(src_region)`: the partial trace of one
/// strided source region, accumulated into a strided destination region.
///
/// `src_region` carries the destination's `dims` followed by one axis per
/// traced pair, each already *merged*: extent `t_k` and stride `s_lhs + s_rhs`
/// of the pair's two source axes, so its index walks the diagonal. The
/// trailing axes are contracted jointly against the packed first `prod t_k`
/// elements of the context's ones template, one `dot_general` in all
/// (TensorOperations' cuTENSOR trace builds the same diagonal-stride
/// descriptor and hands it to a reduction, which Tenferro did not expose on
/// views when this route was written against 0.5.0). Pairs are never merged
/// with one another, so no pair needs a stride commensurate with another's.
///
/// `alpha` scales every traced element before the sum, as TensorOperations'
/// `_mapreducedim!(Scaler(α), Adder(), …)` does and the host does: the
/// diagonal is contracted against `alpha` repeated (the scaled template) with
/// a unit descriptor scale, `dst += Σ aᵢ α`. Why not the descriptor: it
/// scales the finished sum, which overflows where `Σ α aᵢ` does not. At
/// `alpha = 1` the ones template is read and nothing is filled. A zero
/// `alpha` adds VectorInterface's `scale(x, 0) = 0` whatever the source
/// holds, so it submits nothing.
///
/// Transfer contract: the ones and scaled templates grow to `prod t_k` on
/// first need and nothing is uploaded afterwards; a change of `alpha` refills
/// the scaled template on the device (one more submission). Reserve them with
/// [`CudaDenseContext::reserve_ones_template`] and
/// [`CudaDenseContext::reserve_scaled_template`] to pay one upload each for a
/// whole replay.
///
/// Validation, all before any device work: devices, payload dtypes, the
/// source rank and leading extents against the destination (`ShapeMismatch`),
/// element counts, an injective destination, and both regions inside their
/// buffers. An empty destination or an empty traced extent adds nothing and
/// returns `Ok` with no submission.
#[allow(clippy::too_many_arguments)]
pub fn cuda_region_trace_accumulate<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
    src_region: &CudaRegion,
    conj: bool,
    alpha: D,
    dst: &mut CudaDenseStorage,
    dst_region: &CudaRegion,
) -> Result<(), DenseError> {
    const OP: &str = "cuda_region_trace_accumulate";
    ensure_cuda_device(ctx.device, OP, &[("src", src.device), ("dst", dst.device)])?;
    ensure_payload_dtype::<D>(OP, src)?;
    ensure_payload_dtype::<D>(OP, dst)?;
    let rank = dst_region.dims().len();
    if src_region.dims().len() < rank || src_region.dims()[..rank] != *dst_region.dims() {
        return Err(DenseError::ShapeMismatch {
            op: OP,
            expected: dst_region.dims().to_vec(),
            actual: src_region.dims().to_vec(),
        });
    }
    if src_region.is_empty() {
        return Ok(());
    }
    src_region.element_count()?;
    let trace_dims = &src_region.dims()[rank..];
    let trace_len = trace_dims
        .iter()
        .try_fold(1usize, |count, dim| count.checked_mul(*dim))
        .ok_or(DenseError::ElementCountOverflow)?;
    validate_destination_layout(OP, dst_region)?;
    validate_region(src_region, src.len)?;
    validate_region(dst_region, dst.len)?;

    if alpha == D::ZERO {
        return Ok(());
    }
    let unit = alpha == D::ONE;
    if unit {
        ctx.ensure_ones::<D>(trace_len)?;
    } else {
        ctx.ensure_scaled::<D>(alpha, trace_len)?;
    }
    let (backend, operands) = ctx.split_operands::<D>();
    let template = if unit {
        operands.ones.as_ref()
    } else {
        operands.scaled.as_ref()
    };
    let Some(template) = template else {
        return Err(cuda_error(OP, "context scalar operand is missing"));
    };

    let src_strides = src_region
        .strides()
        .iter()
        .map(|&stride| {
            isize::try_from(stride).map_err(|_| DenseError::StrideOverflow { value: stride })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut ones_dims = trace_dims.to_vec();
    ones_dims.push(1);
    let mut ones_strides = Vec::with_capacity(ones_dims.len());
    let mut running = 1isize;
    for &dim in &ones_dims {
        ones_strides.push(running);
        running *= dim as isize;
    }
    let (out_dims, out_strides) = dst_region.contraction_view_metadata()?;

    let lhs =
        src.region_view_nd::<D>(src_region.dims(), &src_strides, src_region.offset_isize()?)?;
    let rhs = template.region_view_nd::<D>(&ones_dims, &ones_strides, 0)?;
    let out = dst.region_view_nd_mut::<D>(&out_dims, &out_strides, dst_region.offset_isize()?)?;
    let traced = trace_dims.len();
    let config = DotGeneralConfig {
        lhs_contracting_dims: (rank..rank + traced).collect(),
        rhs_contracting_dims: (0..traced).collect(),
        lhs_batch_dims: Default::default(),
        rhs_batch_dims: Default::default(),
    };
    let accumulation = DotGeneralAccumulation {
        lhs_conj: conj,
        rhs_conj: false,
        alpha: D::ONE.contraction_scalar(),
        beta: D::ONE.contraction_scalar(),
    };
    record(|stats| stats.gemm_calls += 1);
    backend
        .dot_general_read_into_accum(
            TensorRead::from_view(lhs),
            TensorRead::from_view(rhs),
            &config,
            accumulation,
            TensorWrite::from_view(out),
        )
        .map_err(|err| cuda_error(OP, err))
}
/// Copies the leading compact `rows x cols` block of a device buffer into a
/// packed sub-region of `dst`.
///
/// The source is passed to tenferro's `copy_read_into` as a compact view at
/// offset 0 (strided sources, tenferro-rs#1836, are not used here), so it is
/// read from its start; the caller owns the proof that the destination region's tree layout
/// is identical to what it reads (see `CompactFactorPlan::left_preserves_trees`). A source longer
/// than the region is accepted because contiguity is a layout predicate: the
/// leading `rows * cols` elements of a compact buffer are themselves compact,
/// so one maximum-length buffer can serve every shorter region.
///
/// One `cutensorPermute` through `copy_read_into` at every destination
/// offset: Tenferro 0.6.0 advertises the shifted operand address's true
/// alignment to cuTENSOR (`device_address_alignment`, tenferro-gpu
/// `cubecl/permutation.rs:771`, tensor4all/tenferro-rs#1836), so an offset view
/// never selects a vectorized kernel its pointer cannot satisfy (#1320). It
/// counts one `copy_calls` and transfers nothing.
///
/// Values pass through cuTENSOR's `alpha = 1` scaling: non-NaN real values
/// (signed zeros, infinities and subnormals included) and finite complex
/// values with no negative-zero component arrive bit-exact. NaN payloads may
/// be canonicalized, and a complex value with a negative-zero or non-finite
/// component follows IEEE multiplication by `(1, 0)`.
pub fn cuda_copy_region_into<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    dst: &mut CudaDenseStorage,
    dst_offset: usize,
    dst_ld: usize,
    src: &CudaDenseStorage,
    rows: usize,
    cols: usize,
) -> Result<(), DenseError> {
    const OP: &str = "cuda_copy_region";
    ensure_cuda_device(ctx.device, OP, &[("dst", dst.device), ("src", src.device)])?;
    if rows == 0 || cols == 0 {
        return Ok(());
    }
    record(|stats| stats.copy_calls += 1);
    if src.len < rows * cols {
        return Err(cuda_error(
            OP,
            format!(
                "source factor holds {} elements; expected at least {rows} x {cols}",
                src.len
            ),
        ));
    }
    let src_view = src.region_view::<D>(rows, cols, rows, 0)?;
    let dst_view = dst.region_view_mut::<D>(rows, cols, dst_ld, dst_offset)?;
    ctx.backend
        .copy_read_into(
            TensorRead::from_view(src_view),
            TensorWrite::from_view(dst_view),
        )
        .map_err(|err| cuda_error(OP, err))
}
/// Widens a single-precision device buffer to its double-precision lane,
/// `f32 -> f64` or `Complex32 -> Complex64`, with one device cast
/// (tenferro-gpu 0.7.1 `TensorStructural::cast`, `cubecl/mod.rs:6006`).
///
/// Why: Tenferro 0.7.1 has no reduction that accumulates wider than its
/// operands — the GEMM, `vdot_read` and `norm_squared_read` (cuBLAS
/// `dot`/`dotc`) and `reduce_sum_squares` all sum in the payload dtype — so a
/// reduction that must accumulate as wide as the Host does widens its
/// operands once and then runs the ordinary double-precision reduction. Exact: every `f32` is an
/// `f64`. Costs one device allocation of `capacity * size_of::<W>()` bytes
/// (the whole allocation is cast; [`CudaDenseStorage::len`] carries over) and
/// one elementwise pass; no host transfer.
pub fn cuda_widen<W: CudaScalar>(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
) -> Result<CudaDenseStorage, DenseError> {
    const OP: &str = "cuda_widen";
    ensure_cuda_device(ctx.device, OP, &[("src", src.device)])?;
    if !matches!(
        (src.dtype, W::DTYPE),
        (DenseDType::F32, DenseDType::F64) | (DenseDType::C32, DenseDType::C64)
    ) {
        return Err(cuda_error(
            OP,
            format!("{:?} does not widen to {:?}", src.dtype, W::DTYPE),
        ));
    }
    let tensor = ctx
        .backend
        .cast(&src.tensor, W::dtype())
        .map_err(|err| cuda_error(OP, err))?;
    let mut wide = CudaDenseStorage::from_tensor::<W>(OP, tensor, ctx.device)?;
    wide.len = src.len;
    Ok(wide)
}
/// Gathers whole members of a member-major stack into a new buffer: member
/// `j` of the result is member `selection[j]` of `src`, where member `i`
/// occupies elements `[i * member_len, (i + 1) * member_len)`.
///
/// `src` is flat (`[n]`, `n >= members * member_len`) or `[member_len,
/// members]` (the shape this function returns). Indices may repeat and
/// appear in any order; each must be below `members`. Everything is checked
/// before any device work.
///
/// Why one gather rather than a copy per member: a member is one contiguous
/// slice, but an arbitrary index list has no constant source stride, so no
/// single strided copy expresses it. The gather is one launch for every
/// selection and allocates the result itself, where copies would need
/// `selection.len()` submissions into a destination that must first be
/// uploaded. The values are moved, not recomputed: the result is the source
/// bits.
///
/// Counts one `h2d_calls` for the index table and one `copy_calls` for the
/// gather. A zero `member_len` or an empty selection launches nothing and
/// uploads an empty `[member_len, selection.len()]` buffer.
#[doc(hidden)]
pub fn cuda_gather_members<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
    member_len: usize,
    members: usize,
    selection: &[usize],
) -> Result<CudaDenseStorage, DenseError> {
    const OP: &str = "cuda_gather_members";
    ensure_cuda_device(ctx.device, OP, &[("src", src.device)])?;
    ensure_payload_dtype::<D>(OP, src)?;
    // Why checked here: the gather kernel clamps an out-of-range start
    // (tenferro-gpu `indexing.rs:clamp_window_start`) instead of faulting, so
    // an unchecked index would silently return another member.
    if members
        .checked_mul(member_len)
        .is_none_or(|len| len > src.len)
        || selection.iter().any(|&member| member >= members)
    {
        return Err(cuda_error(
            OP,
            "the stack must hold `members` members and every selected member must be below it",
        ));
    }
    if member_len == 0 || selection.is_empty() {
        return CudaDenseStorage::upload_members::<D>(ctx, Vec::new(), member_len, selection.len());
    }
    // Why a second, `[L, B]` configuration rather than reshaping the gather
    // output to flat: Tenferro 0.7.1 has no metadata-only owned reshape
    // (`CudaBackend::reshape` materializes a copy, `TypedTensor::into_parts`
    // refuses backend storage), so normalizing would add a full copy.
    let (start_stride, config) = match src.tensor.shape() {
        [_] => (
            member_len,
            tenferro_tensor::GatherConfig {
                offset_dims: [0].into_iter().collect(),
                collapsed_slice_dims: Default::default(),
                start_index_map: vec![0],
                index_vector_dim: 1,
                slice_sizes: vec![member_len],
            },
        ),
        &[rows, _] if rows == member_len => (
            1,
            tenferro_tensor::GatherConfig {
                offset_dims: [0].into_iter().collect(),
                collapsed_slice_dims: [1].into_iter().collect(),
                start_index_map: vec![1],
                index_vector_dim: 1,
                slice_sizes: vec![member_len, 1],
            },
        ),
        _ => {
            return Err(cuda_error(
                OP,
                "a stack buffer must be flat or [member_len, members]",
            ))
        }
    };
    let starts = selection
        .iter()
        .map(|&member| {
            i64::try_from(member * start_stride).map_err(|_| cuda_error(OP, "index exceeds i64"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let host =
        i64::into_tensor(vec![selection.len(), 1], starts).map_err(|err| cuda_error(OP, err))?;
    let indices = upload_tensor(&ctx.runtime, &host).map_err(|err| cuda_error(OP, err))?;
    record_h2d(selection.len() * std::mem::size_of::<i64>());
    record(|stats| stats.copy_calls += 1);
    let gathered = ctx
        .backend
        .gather(&src.tensor, &indices, &config)
        .map_err(|err| cuda_error(OP, err))?;
    CudaDenseStorage::from_tensor::<D>(OP, gathered, ctx.device)
}
/// Gathers the same elements of every member of a `[member_len, members]`
/// stack into a new `[elements.len(), members]` buffer: element `p` of result
/// member `m` is element `elements[p]` of source member `m`.
///
/// One Tenferro gather whose window is the member axis (`slice_sizes =
/// [1, members]`), so the launch count is one whatever the member count and
/// however many blocks the element table spans. The gather allocates the
/// result, so no zero-filled destination is uploaded; the only transfer is
/// the `elements` table. The values are moved, not recomputed.
///
/// `src` must have the `[member_len, members]` shape the stack uploads
/// produce ([`CudaDenseStorage::upload_members`]); every element index must be
/// below `member_len`. Both are checked before any device work. Counts one
/// `h2d_calls` and one `copy_calls`. An empty table launches nothing and
/// uploads an empty buffer.
#[doc(hidden)]
pub fn cuda_gather_member_elements<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
    member_len: usize,
    members: usize,
    elements: Vec<i64>,
) -> Result<CudaDenseStorage, DenseError> {
    const OP: &str = "cuda_gather_member_elements";
    ensure_cuda_device(ctx.device, OP, &[("src", src.device)])?;
    ensure_payload_dtype::<D>(OP, src)?;
    if src.tensor.shape() != [member_len, members] {
        return Err(cuda_error(
            OP,
            "a stack buffer must be [member_len, members]",
        ));
    }
    // Why checked here: the gather kernel clamps an out-of-range start
    // (tenferro-gpu `indexing.rs:clamp_window_start`) instead of faulting, so
    // an unchecked index would silently read another element.
    if elements
        .iter()
        .any(|&element| usize::try_from(element).map_or(true, |element| element >= member_len))
    {
        return Err(cuda_error(
            OP,
            "every element index must be below `member_len`",
        ));
    }
    if elements.is_empty() || members == 0 {
        return CudaDenseStorage::upload_members::<D>(ctx, Vec::new(), elements.len(), members);
    }
    let count = elements.len();
    let host = i64::into_tensor(vec![count, 1], elements).map_err(|err| cuda_error(OP, err))?;
    let indices = upload_tensor(&ctx.runtime, &host).map_err(|err| cuda_error(OP, err))?;
    record_h2d(count * std::mem::size_of::<i64>());
    record(|stats| stats.copy_calls += 1);
    let config = tenferro_tensor::GatherConfig {
        offset_dims: [1].into_iter().collect(),
        collapsed_slice_dims: [0].into_iter().collect(),
        start_index_map: vec![0],
        index_vector_dim: 1,
        slice_sizes: vec![1, members],
    };
    let gathered = ctx
        .backend
        .gather(&src.tensor, &indices, &config)
        .map_err(|err| cuda_error(OP, err))?;
    CudaDenseStorage::from_tensor::<D>(OP, gathered, ctx.device)
}
/// Moves `src_region` of `src` into `dst_region` of `dst` (equal `dims`,
/// any strides and offsets, an injective destination) with one cuTENSOR
/// permutation: the source bits arrive unchanged (a NaN payload may be
/// canonicalized). Unlike [`cuda_region_axpby`] it submits no contraction,
/// so it needs no plan-entry reservation. Counts one `copy_calls`.
#[doc(hidden)]
pub fn cuda_copy_strided_into<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
    src_region: &CudaRegion,
    dst: &mut CudaDenseStorage,
    dst_region: &CudaRegion,
) -> Result<(), DenseError> {
    const OP: &str = "cuda_copy_strided";
    ensure_cuda_device(ctx.device, OP, &[("dst", dst.device), ("src", src.device)])?;
    if src_region.dims() != dst_region.dims() {
        return Err(DenseError::ShapeMismatch {
            op: OP,
            expected: dst_region.dims().to_vec(),
            actual: src_region.dims().to_vec(),
        });
    }
    if src_region.is_empty() {
        return Ok(());
    }
    validate_region(src_region, src.len)?;
    validate_destination_layout(OP, dst_region)?;
    validate_region(dst_region, dst.len)?;
    let src_view = src.region_view_nd::<D>(
        src_region.dims(),
        &isize_strides(src_region.strides())?,
        src_region.offset_isize()?,
    )?;
    let dst_view = dst.region_view_nd_mut::<D>(
        dst_region.dims(),
        &isize_strides(dst_region.strides())?,
        dst_region.offset_isize()?,
    )?;
    record(|stats| stats.copy_calls += 1);
    ctx.backend
        .copy_read_into(
            TensorRead::from_view(src_view),
            TensorWrite::from_view(dst_view),
        )
        .map_err(|err| cuda_error(OP, err))
}
/// Gathers single elements of `src` into a new flat buffer: element `p` of
/// the result is element `elements[p]` of `src`, by linear (first axis
/// fastest) position, whatever `src`'s shape.
///
/// One Tenferro gather with a `[elements.len(), rank]` coordinate table, so
/// the only transfer is that table (`8 * rank` bytes per element). The values
/// are moved, not recomputed: unlike a strided copy or a contraction, which
/// scale by a complex `1`, a complex infinity or signed zero arrives bit for
/// bit. Every index is checked before any device work. Counts one
/// `h2d_calls` and one `copy_calls`.
#[doc(hidden)]
pub fn cuda_gather_elements<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
    elements: &[usize],
) -> Result<CudaDenseStorage, DenseError> {
    const OP: &str = "cuda_gather_elements";
    ensure_cuda_device(ctx.device, OP, &[("src", src.device)])?;
    ensure_payload_dtype::<D>(OP, src)?;
    // Why checked here: the gather kernel clamps an out-of-range start
    // instead of faulting (see `cuda_gather_members`).
    if elements.iter().any(|&element| element >= src.len) {
        return Err(cuda_error(
            OP,
            "every element index must be below the buffer length",
        ));
    }
    if elements.is_empty() {
        return CudaDenseStorage::upload_members::<D>(ctx, Vec::new(), 0, 1);
    }
    let shape = src.tensor.shape().to_vec();
    let rank = shape.len();
    let mut coordinates = Vec::with_capacity(elements.len() * rank);
    // Column-major `[count, rank]`: coordinate `axis` of element `p` sits at
    // `p + axis * count`.
    let mut strides = Vec::with_capacity(rank);
    let mut running = 1usize;
    for &extent in &shape {
        strides.push(running);
        running *= extent;
    }
    for axis in 0..rank {
        for &element in elements {
            let coordinate = (element / strides[axis]) % shape[axis];
            coordinates
                .push(i64::try_from(coordinate).map_err(|_| cuda_error(OP, "index exceeds i64"))?);
        }
    }
    let count = elements.len();
    let host =
        i64::into_tensor(vec![count, rank], coordinates).map_err(|err| cuda_error(OP, err))?;
    let indices = upload_tensor(&ctx.runtime, &host).map_err(|err| cuda_error(OP, err))?;
    record_h2d(count * rank * std::mem::size_of::<i64>());
    record(|stats| stats.copy_calls += 1);
    let config = tenferro_tensor::GatherConfig {
        offset_dims: Default::default(),
        collapsed_slice_dims: (0..rank).collect(),
        start_index_map: (0..rank).collect(),
        index_vector_dim: 1,
        slice_sizes: vec![1; rank],
    };
    let gathered = ctx
        .backend
        .gather(&src.tensor, &indices, &config)
        .map_err(|err| cuda_error(OP, err))?;
    CudaDenseStorage::from_tensor::<D>(OP, gathered, ctx.device)
}
/// The elementwise complex conjugate of `src` in a new buffer of the same
/// shape: one Tenferro `conj` launch, `(re, im) -> (re, -im)` by a float
/// negate of the imaginary part, so finite values, infinities and signed
/// zeros are exact; a NaN stays NaN, but its payload and sign bits are not
/// guaranteed. Why not [`cuda_region_axpby`] with its conjugation flag:
/// that is a contraction against a unit operand, and complex multiplication
/// by `(1, 0)` turns an infinite part into NaN and does not preserve `-0`.
/// A real payload is returned as a copy of its bits. Counts one device
/// allocation and no transfer.
#[doc(hidden)]
pub fn cuda_conj<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
) -> Result<CudaDenseStorage, DenseError> {
    const OP: &str = "cuda_conj";
    ensure_cuda_device(ctx.device, OP, &[("src", src.device)])?;
    ensure_payload_dtype::<D>(OP, src)?;
    let conjugated = ctx
        .backend
        .conj_read(TensorRead::from_tensor(&src.tensor))
        .map_err(|err| cuda_error(OP, err))?;
    CudaDenseStorage::from_tensor::<D>(OP, conjugated, ctx.device)
}
