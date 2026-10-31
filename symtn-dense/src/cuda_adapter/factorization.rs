use super::*;

/// A factorization's real spectrum (singular values or eigenvalues), still
/// device-resident in the payload's real lane.
///
/// Why not downloaded by the factorization itself: every download blocks the
/// host (a D2H plus a CubeCL flush), so a per-block download made a
/// block-sparse factorization pay one host sync per coupled sector (#1484).
/// Callers collect the spectra of all their blocks and read them with one
/// [`cuda_download_spectra`].
///
/// It records the device of the context whose solver produced it, as
/// [`CudaDenseStorage`] does, and every consumer rejects a context on another
/// device before any submission (#1766). Reference provenance — TensorKit,
/// TensorOperations and QSpace have no explicit device-admission path to
/// port — is recorded in `docs/backend_policy.md#device-validation`.
pub struct CudaSpectrum {
    pub(super) tensor: Tensor,
    pub(super) device: CudaPlacement,
}
impl CudaSpectrum {
    /// Number of values.
    pub fn len(&self) -> usize {
        self.tensor.shape().iter().product()
    }

    /// Whether the spectrum holds no values.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
/// Rejects a context on another device than any of `spectra`, empty ones
/// included, before the caller submits or allocates anything.
pub(super) fn ensure_spectra_device(
    ctx_device: CudaPlacement,
    op: &'static str,
    spectra: &[CudaSpectrum],
) -> Result<(), DenseError> {
    // The operand name is formatted only for the error: a matching download,
    // one spectrum per coupled sector, allocates nothing on the host here.
    match spectra
        .iter()
        .enumerate()
        .find(|(_, spectrum)| spectrum.device != ctx_device)
    {
        Some((index, spectrum)) => ensure_cuda_device(
            ctx_device,
            op,
            &[(&format!("spectra[{index}]"), spectrum.device)],
        ),
        None => Ok(()),
    }
}
/// Downloads `spectra` (all from factorizations of payload `D`, so in its real
/// lane) with at most one transfer: two or more nonempty spectra are first concatenated on
/// device. Returns each spectrum's values, widened to `f64`, in input order;
/// the values are the solver's own, since concatenation only moves them.
pub fn cuda_download_spectra<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    spectra: &[CudaSpectrum],
) -> Result<Vec<Vec<f64>>, DenseError> {
    const OP: &str = "cuda_download_spectra";
    ensure_spectra_device(ctx.device, OP, spectra)?;
    let parts: Vec<&Tensor> = spectra
        .iter()
        .filter(|spectrum| !spectrum.is_empty())
        .map(|spectrum| &spectrum.tensor)
        .collect();
    let values = match parts.as_slice() {
        [] => Vec::new(),
        [single] => download_values::<D::Real>(ctx, single)?,
        _ => {
            let gathered = ctx
                .backend
                .concatenate(&parts, 0)
                .map_err(|err| cuda_error(OP, err))?;
            download_values::<D::Real>(ctx, &gathered)?
        }
    };
    let total: usize = spectra.iter().map(CudaSpectrum::len).sum();
    if values.len() != total {
        return Err(cuda_error(
            OP,
            format!(
                "downloaded {} spectrum values; expected {total}",
                values.len()
            ),
        ));
    }
    let mut rest = values.as_slice();
    Ok(spectra
        .iter()
        .map(|spectrum| {
            let (head, tail) = rest.split_at(spectrum.len());
            rest = tail;
            head.to_vec()
        })
        .collect())
}
/// Writes `spectrum` (from a factorization of payload `D`) into `dst` as
/// `dst[dst_offset + j * dst_stride] = spectrum[j]`, entirely on the device:
/// with `dst_stride = k + 1` it is the diagonal of a packed `k x k` block.
///
/// A real payload copies the spectrum's own buffer. A complex payload first
/// casts it to `D` (imaginary part `+0`): one device allocation of
/// `len * size_of::<D>()` bytes. Then one [`cuda_copy_region_into`] of a
/// `1 x len` region with leading dimension `dst_stride`, with its value
/// contract. Nothing crosses the host boundary. An empty spectrum is a no-op
/// once `spectrum` and `dst` are proven to be on the context's device.
pub fn cuda_copy_spectrum_into<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    spectrum: CudaSpectrum,
    dst: &mut CudaDenseStorage,
    dst_offset: usize,
    dst_stride: usize,
) -> Result<(), DenseError> {
    const OP: &str = "cuda_copy_spectrum";
    ensure_cuda_device(
        ctx.device,
        OP,
        &[("spectrum", spectrum.device), ("dst", dst.device)],
    )?;
    let len = spectrum.len();
    if len == 0 {
        return Ok(());
    }
    let src = if D::IS_COMPLEX {
        let cast = ctx
            .backend
            .cast(&spectrum.tensor, D::dtype())
            .map_err(|err| cuda_error(OP, err))?;
        CudaDenseStorage::from_tensor::<D>(OP, cast, spectrum.device)?
    } else {
        // The real lane is the payload itself: wrap the solver's buffer
        // without an allocation, so none is counted.
        if D::typed(&spectrum.tensor).is_none() {
            return Err(dtype_mismatch::<D>(OP, &spectrum.tensor));
        }
        CudaDenseStorage {
            tensor: spectrum.tensor,
            dtype: D::DTYPE,
            len,
            device: spectrum.device,
        }
    };
    cuda_copy_region_into::<D>(ctx, dst, dst_offset, dst_stride, &src, 1, len)
}
/// cuSOLVER SVD of one packed column-major `rows x cols` region:
/// `region = U * diag(s) * Vt` with `k = min(rows, cols)`. `U` (`rows x k`),
/// the singular values `s` (descending) and `Vt` (`k x cols`) all stay
/// device-resident; read `s` with [`cuda_download_spectra`].
pub fn cuda_svd_region<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
    offset: usize,
    rows: usize,
    cols: usize,
) -> Result<(CudaDenseStorage, CudaSpectrum, CudaDenseStorage), DenseError> {
    ensure_cuda_device(ctx.device, "cuda_svd", &[("src", src.device)])?;
    let view = src.region_view::<D>(rows, cols, rows, offset)?;
    record(|stats| stats.solver_calls += 1);
    let (u, s, vt) = TensorRead::from_view(view)
        .svd_read(&mut *ctx.backend)
        .map_err(|err| cuda_error("cuda_svd", err))?;
    let vt = CudaDenseStorage::from_tensor::<D>("cuda_svd", vt, ctx.device)?;
    let s = CudaSpectrum {
        tensor: s,
        device: ctx.device,
    };
    let u = CudaDenseStorage::from_tensor::<D>("cuda_svd", u, ctx.device)?;
    validate_svd_factor_shapes(u.tensor.shape(), s.len(), vt.tensor.shape(), rows, cols)?;
    Ok((u, s, vt))
}
/// Row weights `N - i` (`i < N`, `i64`) that make "first row among equal
/// maxima" a device reduction for [`cuda_svd_gauge_phases`]. One upload of
/// `N` `i64`s per `svd_compact` call, shared by every route with at most `N`
/// rows. Integer weights are exact at any row count and independent of the
/// payload lane; every op on them is native for `i64` in Tenferro 0.7.1.
#[doc(hidden)]
pub struct CudaSvdGaugeWeights {
    tensor: Tensor,
    len: usize,
    device: CudaPlacement,
}
impl CudaSvdGaugeWeights {
    /// Uploads the `max_rows` weights: one H2D of `8 * max_rows` bytes.
    pub fn upload(ctx: &CudaDenseContext, max_rows: usize) -> Result<Self, DenseError> {
        const OP: &str = "cuda_svd_gauge";
        let data = (0..max_rows)
            .map(|row| i64::try_from(max_rows - row))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| cuda_error(OP, "row count does not fit in i64"))?;
        let host =
            Tensor::from_vec_col_major(vec![max_rows], data).map_err(|err| cuda_error(OP, err))?;
        let tensor = upload_tensor(&ctx.runtime, &host).map_err(|err| cuda_error(OP, err))?;
        record_h2d(max_rows * std::mem::size_of::<i64>());
        Ok(Self {
            tensor,
            len: max_rows,
            device: ctx.device,
        })
    }
}
/// Per-column phases of a compact SVD's `U`, from [`cuda_svd_gauge_phases`].
///
/// Applying them (`U diag(conj(phase))`, `diag(phase) Vh`) leaves
/// `U diag(s) Vh` unchanged and makes the first largest-magnitude entry of
/// every `U` column real and non-negative: the Host `svd_compact_gauge`
/// (symtn-matrixalgebra) and MatrixAlgebraKit's `gaugefix!(svd_compact!)`.
pub struct CudaSvdPhases {
    phase: Tensor,
    k: usize,
    device: CudaPlacement,
}
const SVD_GAUGE_OP: &str = "cuda_svd_gauge";
/// One counted Tenferro op of the SVD gauge.
fn gauge_op<T>(result: tenferro_tensor::Result<T>) -> Result<T, DenseError> {
    record(|stats| stats.gauge_ops += 1);
    result.map_err(|err| cuda_error(SVD_GAUGE_OP, err))
}
/// The Host SVD gauge phases of the `rows x k` compact left factor `u`
/// (`rows, k > 0`), computed on the device: nothing is downloaded and no host
/// barrier is taken.
///
/// Tenferro 0.7.1 has no arg-max reduction, so the pivot is composed from 13
/// ops (each counted in `gauge_ops`): `abs`, column `reduce_max`, broadcast,
/// `compare` (every maximum), `cast` to `i64`, weight broadcast, `mul`, column
/// `reduce_max`, broadcast, `compare` (the first maximum: the weights are
/// distinct and decreasing), `cast` to `D`, a column-batched `dot_general` of
/// `u` with that one-hot mask (the pivot) and `sign`. Why not `gather` of an
/// arg-max index: it needs the index as an owned tensor per route, and a
/// float-to-int `cast` validates its range with a download.
///
/// Working set: each intermediate is dropped after its last use. Besides
/// `u`, the live `rows x k` intermediates peak at 24 bytes per element, at
/// the weighted `mul` (two `i64` operands and its `i64` output): 3x `u` for
/// `f64`, 1.5x for `Complex64`. Every other step holds less: `abs` output,
/// broadcast maximum and bool mask (at most 17 B), or the bool mask and the
/// `D`-typed one-hot mask (at most 17 B for `Complex64`).
///
/// Magnitudes are compared by `abs` (`hypot`), the Host by `norm_sqr`; they
/// can disagree only when two complex entries tie within an ulp in one
/// metric, where the pivot row is ill-conditioned anyway.
///
/// Non-finite input differs from the Host: a NaN anywhere in a column makes
/// its maximum NaN, no entry compares equal to it, the pivot mask is then all
/// rows, and the column's `u` and the matching `vh` row become NaN. The Host
/// skips NaN entries when it picks the pivot.
pub fn cuda_svd_gauge_phases<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    u: &CudaDenseStorage,
    rows: usize,
    k: usize,
    weights: &CudaSvdGaugeWeights,
) -> Result<CudaSvdPhases, DenseError> {
    const OP: &str = SVD_GAUGE_OP;
    ensure_cuda_device(
        ctx.device,
        OP,
        &[("u", u.device), ("weights", weights.device)],
    )?;
    ensure_payload_dtype::<D>(OP, u)?;
    if rows == 0 || k == 0 || u.tensor.shape() != [rows, k] || rows > weights.len {
        return Err(cuda_error(
            OP,
            format!(
                "gauge of U={:?} as a nonempty [{rows}, {k}] factor with {} weights",
                u.tensor.shape(),
                weights.len
            ),
        ));
    }
    let shape = [rows, k];
    let backend = &mut ctx.backend;
    let maxima = {
        let hits = {
            let magnitude = gauge_op(backend.abs_read(TensorRead::from_tensor(&u.tensor)))?;
            let largest =
                gauge_op(backend.reduce_max_read(TensorRead::from_tensor(&magnitude), &[0]))?;
            let largest = gauge_op(backend.broadcast_in_dim_read(
                TensorRead::from_tensor(&largest),
                &shape,
                &[1],
            ))?;
            gauge_op(backend.compare_read(
                TensorRead::from_tensor(&magnitude),
                TensorRead::from_tensor(&largest),
                &CompareDir::Eq,
            ))?
        };
        gauge_op(backend.cast(&hits, tenferro_tensor::DType::I64))?
    };
    let score = {
        let row_weights = if rows == weights.len {
            gauge_op(backend.broadcast_in_dim_read(
                TensorRead::from_tensor(&weights.tensor),
                &shape,
                &[0],
            ))?
        } else {
            let prefix = weights
                .tensor
                .as_typed::<i64>()
                .ok_or_else(|| cuda_error(OP, "gauge weights are not i64"))?
                .backend_region_view(vec![rows], vec![1], 0)
                .map(<i64 as TenferroScalar>::tensor_view)
                .map_err(|err| cuda_error(OP, err))?;
            gauge_op(backend.broadcast_in_dim_read(TensorRead::from_view(prefix), &shape, &[0]))?
        };
        gauge_op(backend.mul_read(
            TensorRead::from_tensor(&maxima),
            TensorRead::from_tensor(&row_weights),
        ))?
    };
    drop(maxima);
    let first_hit = {
        let best = gauge_op(backend.reduce_max_read(TensorRead::from_tensor(&score), &[0]))?;
        let best =
            gauge_op(backend.broadcast_in_dim_read(TensorRead::from_tensor(&best), &shape, &[1]))?;
        gauge_op(backend.compare_read(
            TensorRead::from_tensor(&score),
            TensorRead::from_tensor(&best),
            &CompareDir::Eq,
        ))?
    };
    drop(score);
    let first = gauge_op(backend.cast(&first_hit, D::dtype()))?;
    drop(first_hit);
    // Why a batched dot rather than `mul` + `reduce_sum`: Tenferro 0.7.1's
    // complex warp-plane `reduce_sum` kernel fails NVRTC compilation
    // (`__shfl_xor_sync` has no `cuDoubleComplex` overload).
    let column_dot = DotGeneralConfig {
        lhs_contracting_dims: [0].into_iter().collect(),
        rhs_contracting_dims: [0].into_iter().collect(),
        lhs_batch_dims: [1].into_iter().collect(),
        rhs_batch_dims: [1].into_iter().collect(),
    };
    let pivot = gauge_op(backend.dot_general_read(
        TensorRead::from_tensor(&u.tensor),
        TensorRead::from_tensor(&first),
        &column_dot,
    ))?;
    drop(first);
    let phase = gauge_op(backend.sign_read(TensorRead::from_tensor(&pivot)))?;
    Ok(CudaSvdPhases {
        phase,
        k,
        device: ctx.device,
    })
}
impl CudaSvdPhases {
    /// Runs `body` on `conj(phase)`: one `conj` for a complex payload, the
    /// phase itself for a real one (`±1`).
    fn with_left_phase<D: CudaScalar, T>(
        &self,
        ctx: &mut CudaDenseContext,
        body: impl FnOnce(&mut CudaDenseContext, &Tensor) -> Result<T, DenseError>,
    ) -> Result<T, DenseError> {
        ensure_cuda_device(ctx.device, SVD_GAUGE_OP, &[("phases", self.device)])?;
        if D::IS_COMPLEX {
            let conjugated = gauge_op(ctx.backend.conj_read(TensorRead::from_tensor(&self.phase)))?;
            body(ctx, &conjugated)
        } else {
            body(ctx, &self.phase)
        }
    }

    fn checked_factor<D: CudaScalar>(
        &self,
        factor: &CudaDenseStorage,
        shape: [usize; 2],
    ) -> Result<(), DenseError> {
        ensure_payload_dtype::<D>(SVD_GAUGE_OP, factor)?;
        if factor.tensor.shape() != shape {
            return Err(DenseError::ShapeMismatch {
                op: SVD_GAUGE_OP,
                expected: shape.to_vec(),
                actual: factor.tensor.shape().to_vec(),
            });
        }
        Ok(())
    }

    /// `u diag(conj(phase))` for a `rows x k` factor: broadcast and `mul`,
    /// plus `conj` for a complex payload.
    pub fn scale_left<D: CudaScalar>(
        &self,
        ctx: &mut CudaDenseContext,
        u: &CudaDenseStorage,
        rows: usize,
    ) -> Result<CudaDenseStorage, DenseError> {
        ensure_cuda_device(ctx.device, SVD_GAUGE_OP, &[("u", u.device)])?;
        self.checked_factor::<D>(u, [rows, self.k])?;
        let k = self.k;
        self.with_left_phase::<D, _>(ctx, |ctx, phase| {
            let phase = gauge_op(ctx.backend.broadcast_in_dim_read(
                TensorRead::from_tensor(phase),
                &[rows, k],
                &[1],
            ))?;
            let scaled = gauge_op(ctx.backend.mul_read(
                TensorRead::from_tensor(&u.tensor),
                TensorRead::from_tensor(&phase),
            ))?;
            CudaDenseStorage::from_tensor::<D>(SVD_GAUGE_OP, scaled, ctx.device)
        })
    }

    /// `diag(phase) vh` for a `k x cols` factor: broadcast and `mul`.
    pub fn scale_right<D: CudaScalar>(
        &self,
        ctx: &mut CudaDenseContext,
        vh: &CudaDenseStorage,
        cols: usize,
    ) -> Result<CudaDenseStorage, DenseError> {
        ensure_cuda_device(
            ctx.device,
            SVD_GAUGE_OP,
            &[("phases", self.device), ("vh", vh.device)],
        )?;
        self.checked_factor::<D>(vh, [self.k, cols])?;
        let phase = gauge_op(ctx.backend.broadcast_in_dim_read(
            TensorRead::from_tensor(&self.phase),
            &[self.k, cols],
            &[0],
        ))?;
        let scaled = gauge_op(ctx.backend.mul_read(
            TensorRead::from_tensor(&phase),
            TensorRead::from_tensor(&vh.tensor),
        ))?;
        CudaDenseStorage::from_tensor::<D>(SVD_GAUGE_OP, scaled, ctx.device)
    }

    /// `diag(conj(phase))` as a `k x k` right operand for an assembly GEMM
    /// that already multiplies by a selector: `embed_diagonal`, plus `conj`
    /// for a complex payload.
    pub fn left_selector<D: CudaScalar>(
        &self,
        ctx: &mut CudaDenseContext,
    ) -> Result<CudaDenseStorage, DenseError> {
        self.with_left_phase::<D, _>(ctx, |ctx, phase| {
            let selector = gauge_op(ctx.backend.embed_diagonal(phase, 0, 1))?;
            CudaDenseStorage::from_tensor::<D>(SVD_GAUGE_OP, selector, ctx.device)
        })
    }

    /// `diag(phase)` as a `k x k` left operand for an assembly GEMM:
    /// one `embed_diagonal`.
    pub fn right_selector<D: CudaScalar>(
        &self,
        ctx: &mut CudaDenseContext,
    ) -> Result<CudaDenseStorage, DenseError> {
        ensure_cuda_device(ctx.device, SVD_GAUGE_OP, &[("phases", self.device)])?;
        let selector = gauge_op(ctx.backend.embed_diagonal(&self.phase, 0, 1))?;
        CudaDenseStorage::from_tensor::<D>(SVD_GAUGE_OP, selector, ctx.device)
    }
}
pub(super) fn validate_svd_factor_shapes(
    u_shape: &[usize],
    s_len: usize,
    vt_shape: &[usize],
    rows: usize,
    cols: usize,
) -> Result<(), DenseError> {
    let k = rows.min(cols);
    if u_shape != [rows, k] || s_len != k || vt_shape != [k, cols] {
        return Err(cuda_error(
            "cuda_svd",
            format!(
                "device SVD returned U={u_shape:?}, len(S)={s_len}, Vt={vt_shape:?}; expected U=[{rows}, {k}], len(S)={k}, Vt=[{k}, {cols}]"
            ),
        ));
    }
    Ok(())
}
/// cuSOLVER QR of one packed column-major `rows x cols` region:
/// `region = Q * R` with `k = min(rows, cols)`, `Q` (`rows x k`) and `R`
/// (`k x cols`) device-resident, in the positive-diagonal gauge
/// (`R_jj` real and non-negative, phase 1 kept when `R_jj == 0`).
///
/// The gauge is Tenferro's own [`QrGauge::PositiveDiagonal`]: it is applied on
/// device inside the QR primitive, so no diagonal crosses to the host and no
/// symtn-side re-gauging selector exists.
///
/// Every [`CudaScalar`] payload is admitted. Up to Tenferro 0.5.0 the complex
/// payloads could not be: the gauge's `triu` fill materialized its complex zero
/// as `E::cast_from(0u32)`, which NVRTC rejected (tenferro-rs#1833, #1271).
pub fn cuda_qr_region<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
    offset: usize,
    rows: usize,
    cols: usize,
) -> Result<(CudaDenseStorage, CudaDenseStorage), DenseError> {
    ensure_cuda_device(ctx.device, "cuda_qr", &[("src", src.device)])?;
    let view = src.region_view::<D>(rows, cols, rows, offset)?;
    let options = QrOptions::default().gauge(QrGauge::PositiveDiagonal);
    record(|stats| stats.solver_calls += 1);
    let (q, r) = TensorRead::from_view(view)
        .qr_with_options_read(options, &mut *ctx.backend)
        .map_err(|err| cuda_error("cuda_qr", err))?;
    let r = CudaDenseStorage::from_tensor::<D>("cuda_qr", r, ctx.device)?;
    let q = CudaDenseStorage::from_tensor::<D>("cuda_qr", q, ctx.device)?;
    validate_qr_factor_shapes(q.tensor.shape(), r.tensor.shape(), rows, cols)?;
    Ok((q, r))
}
pub(super) fn validate_qr_factor_shapes(
    q_shape: &[usize],
    r_shape: &[usize],
    rows: usize,
    cols: usize,
) -> Result<(), DenseError> {
    let k = rows.min(cols);
    if q_shape != [rows, k] || r_shape != [k, cols] {
        return Err(cuda_error(
            "cuda_qr",
            format!(
                "device QR returned shapes Q={q_shape:?}, R={r_shape:?}; expected Q=[{rows}, {k}], R=[{k}, {cols}]"
            ),
        ));
    }
    Ok(())
}
/// cuSOLVER Hermitian eigendecomposition of one packed column-major
/// `n x n` region: eigenvalues (ascending) and eigenvectors (`n x n`, one
/// eigenvector per column, in the same order) stay device-resident; read the
/// eigenvalues with [`cuda_download_spectra`] for host ordering and
/// truncation decisions.
pub fn cuda_eigh_region<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
    offset: usize,
    n: usize,
) -> Result<(CudaSpectrum, CudaDenseStorage), DenseError> {
    ensure_cuda_device(ctx.device, "cuda_eigh", &[("src", src.device)])?;
    let view = src.region_view::<D>(n, n, n, offset)?;
    record(|stats| stats.solver_calls += 1);
    let (values, vectors) = TensorRead::from_view(view)
        .eigh_read(&mut *ctx.backend)
        .map_err(|err| cuda_error("cuda_eigh", err))?;
    let vectors = CudaDenseStorage::from_tensor::<D>("cuda_eigh", vectors, ctx.device)?;
    let values = CudaSpectrum {
        tensor: values,
        device: ctx.device,
    };
    validate_eigh_factor_shapes(values.len(), vectors.tensor.shape(), n)?;
    Ok((values, vectors))
}
/// [`cuda_eigh_region`] of `members` stacked `n x n` regions, member `b` at
/// `offset + b * member_stride`, as one backend call on the `[n, n, members]`
/// view. Tenferro's `Auto` driver solves a batch of more than one matrix with
/// one `XsyevBatched` launch and a single matrix with `syevd`, exactly as
/// [`cuda_eigh_region`] does; the batched routine may differ from the
/// per-matrix one in the last ULPs.
///
/// Returns the eigenvalues as one `[n, members]` spectrum (ascending per
/// member) and the eigenvectors as a compact `[n, n, members]` buffer. One
/// `solver_calls` count whatever `members` is. Tenferro reads the solver
/// status of the whole batch once and reports the first failure, without its
/// member: a positive status is [`DenseError::NumericalFailure`]; argument,
/// bounds and backend failures keep their own variants.
#[doc(hidden)]
pub fn cuda_eigh_region_batched<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    src: &CudaDenseStorage,
    offset: usize,
    n: usize,
    members: usize,
    member_stride: usize,
) -> Result<(CudaSpectrum, CudaDenseStorage), DenseError> {
    const OP: &str = "cuda_eigh_batched";
    ensure_cuda_device(ctx.device, OP, &[("src", src.device)])?;
    let region = CudaRegion::new(vec![n, n, members], vec![1, n, member_stride], offset)?;
    validate_region(&region, src.len)?;
    let view = src.region_view_nd::<D>(
        region.dims(),
        &isize_strides(region.strides())?,
        region.offset_isize()?,
    )?;
    record(|stats| stats.solver_calls += 1);
    let (values, vectors) = TensorRead::from_view(view)
        .eigh_read(&mut *ctx.backend)
        .map_err(|err| {
            // The solver's own status (cuSOLVER `info > 0`) is a numerical
            // failure of some member; everything else is the call's.
            if err.kind() == tenferro_tensor::ErrorKind::NumericalFailure {
                DenseError::NumericalFailure {
                    backend: DenseBackend::Cuda,
                    op: OP,
                    message: err.to_string(),
                }
            } else {
                cuda_error(OP, err)
            }
        })?;
    if values.shape() != [n, members] || vectors.shape() != [n, n, members] {
        return Err(cuda_error(
            OP,
            format!(
                "device EIGH returned values={:?}, vectors={:?}; expected [{n}, {members}] and [{n}, {n}, {members}]",
                values.shape(),
                vectors.shape()
            ),
        ));
    }
    let vectors = CudaDenseStorage::from_tensor::<D>(OP, vectors, ctx.device)?;
    Ok((
        CudaSpectrum {
            tensor: values,
            device: ctx.device,
        },
        vectors,
    ))
}
/// Downloads the `[n_r, members]` spectra of [`cuda_eigh_region_batched`]
/// with at most one transfer, as one column-major `[N, members]` table,
/// `N = Σ_r n_r`: value `i` of spectrum `r` for member `b` is at
/// `(Σ_{r' < r} n_r') + i + N * b`.
#[doc(hidden)]
pub fn cuda_download_batched_spectra<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    spectra: &[CudaSpectrum],
    members: usize,
) -> Result<Vec<f64>, DenseError> {
    const OP: &str = "cuda_download_spectra";
    ensure_spectra_device(ctx.device, OP, spectra)?;
    let parts: Vec<&Tensor> = spectra
        .iter()
        .filter(|spectrum| !spectrum.is_empty())
        .map(|spectrum| &spectrum.tensor)
        .collect();
    if parts
        .iter()
        .any(|tensor| tensor.shape().get(1) != Some(&members))
    {
        return Err(cuda_error(OP, "a batched spectrum is not [n, members]"));
    }
    let values = match parts.as_slice() {
        [] => Vec::new(),
        [single] => download_values::<D::Real>(ctx, single)?,
        _ => {
            let gathered = ctx
                .backend
                .concatenate(&parts, 0)
                .map_err(|err| cuda_error(OP, err))?;
            download_values::<D::Real>(ctx, &gathered)?
        }
    };
    let total: usize = spectra.iter().map(CudaSpectrum::len).sum();
    if values.len() != total {
        return Err(cuda_error(
            OP,
            format!(
                "downloaded {} spectrum values; expected {total}",
                values.len()
            ),
        ));
    }
    Ok(values)
}
/// Writes selected eigenvector columns of every member into a stacked
/// destination: for member `b` and output column `c`, column
/// `columns[b * kept + c]` of member `b` of `src`, a compact `[n, n,
/// members]` buffer (as [`cuda_eigh_region_batched`] returns it).
///
/// The destination is the `dst_ld x kept` region at `dst_offset` of each
/// member (leading dimension `dst_ld`, member stride `dst_member_stride`).
/// One Tenferro `gather` over all members, then one strided copy per entry
/// of `rows`: `(src_row, dst_row, rows)` copies source rows `src_row..src_row
/// + rows` of every selected column to rows `dst_row..dst_row + rows` of the
/// destination region. So a layout-aligned target is one entry and a
/// tree-wise target one entry per codomain tree.
///
/// Every entry is validated before any submission: source rows inside `n`,
/// destination rows inside `dst_ld` and pairwise disjoint, and the whole
/// destination region inside `dst` and injective. A rejected call writes
/// nothing.
///
/// Why not a selector GEMM, as eager's rank-2 assembly uses: over a member
/// axis the selector is `members · n · kept` values uploaded per call and
/// `2 n² kept` FLOPs per member, against `2 · members · kept` indices and
/// `n · kept` moves here; the A100 measurement of #1499 found the GEMM up to
/// 53x slower at 1024 members and at most 0.3 ms faster at 16. The values
/// are moved, not recomputed, so the result is the source bits (a NaN
/// payload may be canonicalized by the copy).
///
/// Counts one `h2d_calls` for the index table and one `copy_calls` per
/// submission (the gather and each copy).
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn cuda_gather_columns_batched_into<D: CudaScalar>(
    ctx: &mut CudaDenseContext,
    dst: &mut CudaDenseStorage,
    dst_offset: usize,
    dst_ld: usize,
    dst_member_stride: usize,
    src: &CudaDenseStorage,
    n: usize,
    members: usize,
    columns: &[usize],
    rows: &[(usize, usize, usize)],
) -> Result<(), DenseError> {
    const OP: &str = "cuda_gather_columns";
    ensure_cuda_device(ctx.device, OP, &[("dst", dst.device), ("src", src.device)])?;
    if members == 0 || columns.is_empty() {
        return Ok(());
    }
    let kept = columns.len() / members;
    if kept * members != columns.len()
        || columns.iter().any(|&column| column >= n)
        || src.tensor.shape() != [n, n, members]
    {
        return Err(cuda_error(
            OP,
            "columns must hold `kept` source columns (< n) per member of an [n, n, members] source",
        ));
    }
    validate_gather_rows(n, dst_ld, rows).map_err(|message| cuda_error(OP, message))?;
    let target = CudaRegion::new(
        vec![dst_ld, kept, members],
        vec![1, dst_ld, dst_member_stride],
        dst_offset,
    )?;
    validate_destination_layout(OP, &target)?;
    validate_region(&target, dst.len)?;

    let to_i64 =
        |value: usize| i64::try_from(value).map_err(|_| cuda_error(OP, "index exceeds i64"));
    let mut indices = Vec::with_capacity(2 * columns.len());
    for &column in columns {
        indices.push(to_i64(column)?);
    }
    for member in 0..members {
        for _ in 0..kept {
            indices.push(to_i64(member)?);
        }
    }
    let host =
        i64::into_tensor(vec![columns.len(), 2], indices).map_err(|err| cuda_error(OP, err))?;
    let indices = upload_tensor(&ctx.runtime, &host).map_err(|err| cuda_error(OP, err))?;
    record_h2d(2 * columns.len() * std::mem::size_of::<i64>());
    let config = tenferro_tensor::GatherConfig {
        offset_dims: [0].into_iter().collect(),
        collapsed_slice_dims: [1, 2].into_iter().collect(),
        start_index_map: vec![1, 2],
        index_vector_dim: 1,
        slice_sizes: vec![n, 1, 1],
    };
    record(|stats| stats.copy_calls += 1);
    let gathered = ctx
        .backend
        .gather(&src.tensor, &indices, &config)
        .map_err(|err| cuda_error(OP, err))?;
    // `[n, kept * members]`, member-major: the compact `[n, kept, members]`.
    let gathered = CudaDenseStorage::from_tensor::<D>(OP, gathered, ctx.device)?;
    for &(src_row, dst_row, count) in rows {
        if count == 0 {
            continue;
        }
        cuda_copy_strided_into::<D>(
            ctx,
            &gathered,
            &CudaRegion::new(vec![count, kept, members], vec![1, n, n * kept], src_row)?,
            dst,
            &CudaRegion::new(
                vec![count, kept, members],
                vec![1, dst_ld, dst_member_stride],
                dst_offset + dst_row,
            )?,
        )?;
    }
    Ok(())
}
pub(super) fn validate_eigh_factor_shapes(
    values_len: usize,
    vectors_shape: &[usize],
    n: usize,
) -> Result<(), DenseError> {
    if values_len != n || vectors_shape != [n, n] {
        return Err(cuda_error(
            "cuda_eigh",
            format!(
                "device EIGH returned len(values)={values_len}, vectors={vectors_shape:?}; expected len(values)={n}, vectors=[{n}, {n}]"
            ),
        ));
    }
    Ok(())
}
