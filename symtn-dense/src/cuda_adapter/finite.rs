use super::*;

/// Whether every real and imaginary component of the active payload
/// (`src.len()` elements) is finite: the device form of the Host
/// finite-input stage of the eager factorizations (#1986, #2038).
///
/// One device pass computes `sum(x - x)`. `x - x` is `0` for a finite
/// component and `NaN` for `NaN` or `±Inf` (`Inf - Inf`), with no overflow,
/// so the sum is `0` exactly when every component is finite. The sum is
/// downloaded as one real-lane scalar: one D2H, which synchronizes the
/// stream. An empty payload is finite with no device work.
///
/// Why `abs` before the complex sum: Tenferro 0.7.1's full `reduce_sum` of a
/// complex tensor fails NVRTC compilation (tensor4all/tenferro-rs#2133).
/// `|x - x|` keeps the `{0, NaN}` set componentwise (`hypot(0, NaN)` is
/// `NaN`) and lands in the real lane, where the reduction compiles.
///
/// Cost: one `sub`, (complex) one `abs`, one `reduce_sum` over the payload,
/// one transient payload-sized buffer, and the D2H sync. Tenferro 0.7.1's
/// full reduction runs on a single warp (tensor4all/tenferro-rs#2134), so the
/// reduction, not the elementwise pass, dominates on large payloads.
#[doc(hidden)]
pub fn cuda_all_finite<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
) -> Result<bool, DenseError> {
    const OP: &str = "cuda_all_finite";
    ensure_cuda_device(ctx.device, OP, &[("src", src.device)])?;
    if D::typed(&src.tensor).is_none() {
        return Err(dtype_mismatch::<D>(OP, &src.tensor));
    }
    if src.len == 0 {
        return Ok(true);
    }
    // A grow-only scratch may hold stale elements past its active length.
    let prefix = if src.capacity() == src.len {
        None
    } else {
        let view = src.region_view_nd::<D>(&[src.len], &[1], 0)?;
        Some(
            ctx.backend
                .to_contiguous_read(TensorRead::from_view(view))
                .map_err(|err| cuda_error(OP, err))?,
        )
    };
    let payload = prefix.as_ref().unwrap_or(&src.tensor);
    let axes: Vec<usize> = (0..payload.shape().len()).collect();
    let diff = ctx
        .backend
        .sub_read(
            TensorRead::from_tensor(payload),
            TensorRead::from_tensor(payload),
        )
        .map_err(|err| cuda_error(OP, err))?;
    let diff = if D::IS_COMPLEX {
        ctx.backend
            .abs_read(TensorRead::from_tensor(&diff))
            .map_err(|err| cuda_error(OP, err))?
    } else {
        diff
    };
    let sum = ctx
        .backend
        .reduce_sum_read(TensorRead::from_tensor(&diff), &axes)
        .map_err(|err| cuda_error(OP, err))?;
    let sum = if sum.shape().is_empty() {
        ctx.backend
            .reshape_read(TensorRead::from_tensor(&sum), &[1])
            .map_err(|err| cuda_error(OP, err))?
    } else {
        sum
    };
    Ok(download_values::<D::Real>(ctx, &sum)?
        .iter()
        .all(|value| *value == 0.0))
}
