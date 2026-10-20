use super::*;

impl<R, D> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols + CheckedFusionAlgebra + SectorCodec,
    D: TensorScalar,
{
    /// The single element of a rank-0 (scalar) tensor, e.g. the result of
    /// contracting every leg — TensorKit `scalar` (an empty payload reads
    /// as zero there too).
    ///
    /// Returns `D` directly: the value is the sum of the coupled payload. A
    /// lazy adjoint is materialized operation-locally; a rank-0 payload holds
    /// at most one value per coupled sector.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] on a tensor with legs.
    pub fn scalar(&self) -> Result<D, Error> {
        if self.rank() != 0 {
            return Err(Error::InvalidArgument(format!(
                "scalar() requires a rank-0 tensor, got rank {}",
                self.rank()
            )));
        }
        // A rank-0 payload holds at most one element; summing gives the empty
        // payload its zero for free.
        let materialized = self.materialized_tensor_uncached()?;
        Ok(materialized
            .owned_body()
            .expect("uncached materialization is owned")
            .materialized_dense_data()
            .as_ref()
            .iter()
            .fold(D::from_real(0.0), |acc, &value| acc + value))
    }
}

/// The quantum-dimension-weighted inner product of two compact spectra,
/// `Σ_c dim(c) * Σ_i conj(a_i) b_i` — [`coupled_region_inner`]'s reduction
/// with the zeros left out, since a bond space's dense form is zero off the
/// per-sector diagonal.
fn compact_inner<D, E>(
    lhs: &[symtn_matrixalgebra::SectorSpectrum<D>],
    rhs: &[symtn_matrixalgebra::SectorSpectrum<D>],
    mut weight_of: impl FnMut(SectorId) -> Result<f64, E>,
) -> Result<num_complex::Complex64, E>
where
    D: TensorScalar,
    E: From<Error>,
{
    if lhs.len() != rhs.len() {
        return Err(spectra_disagree().into());
    }
    let mut total = num_complex::Complex64::new(0.0, 0.0);
    for (left, right) in lhs.iter().zip(rhs) {
        if left.sector != right.sector || left.values.len() != right.values.len() {
            return Err(spectra_disagree().into());
        }
        // `D::Wide` for the same reason as `coupled_region_inner`: the
        // identity for the double pair, double precision for the single
        // one, so a compact `norm`/`inner` answers like its dense twin.
        let mut partial = D::Wide::from_real(0.0);
        for (&a, &b) in left.values.iter().zip(&right.values) {
            partial = partial + FactorScalar::adjoint(a.widen()) * b.widen();
        }
        total += partial.widen_complex() * weight_of(left.sector)?;
    }
    Ok(total)
}

/// The largest stored magnitude of a compact spectrum, as [`max_abs`].
fn spectrum_max_abs<D: TensorScalar>(spectrum: &[symtn_matrixalgebra::SectorSpectrum<D>]) -> f64 {
    max_abs(
        spectrum
            .iter()
            .flat_map(|entry| entry.values.iter().copied()),
    )
}

/// `Σ_c dim(c) · Σ_k term(s_{c,k})` over a compact spectrum: the stored
/// diagonal of each coupled block, whose off-diagonal zeros add nothing to
/// any power sum with `p > 0`.
fn spectrum_weighted_sum<D, E>(
    spectrum: &[symtn_matrixalgebra::SectorSpectrum<D>],
    mut weight_of: impl FnMut(SectorId) -> Result<f64, E>,
    term: impl Fn(D) -> f64,
) -> Result<f64, E>
where
    D: TensorScalar,
{
    spectrum.iter().try_fold(0.0, |total, entry| {
        Ok(total
            + weight_of(entry.sector)? * entry.values.iter().map(|&value| term(value)).sum::<f64>())
    })
}

/// The dimension-weighted dense inner product `Σ_c dim(c) * <a_c, b_c>`, with
/// TensorKit's `UniqueFusion` specialization: every `dim(c) == 1` and the
/// coupled buffer is a padding-free concatenation of the per-sector blocks, so
/// the weighted per-block sum collapses to one whole-buffer conjugated dot
/// (`vectorinterface.jl:124`, `linalg.jl:277`).
fn dense_weighted_inner<D, E>(
    unique: bool,
    structure: &BlockStructure,
    nout: usize,
    a: &[D],
    b: &[D],
    weight_of: impl FnMut(SectorId) -> Result<f64, E>,
) -> Result<num_complex::Complex64, E>
where
    D: TensorScalar,
    E: From<Error>,
{
    if unique {
        return Ok(super::numeric::wide_conj_dot(a, b).widen_complex());
    }
    coupled_region_inner(structure, nout, a, b, weight_of)
}

/// `dim(c)` through the mode, evaluated in place with a one-entry run cache:
/// a canonical structure lists each coupled sector as one contiguous run of
/// blocks, so this asks once per coupled sector with no allocation and no
/// hashing.
fn run_cached_dim<R>(provider: &R) -> impl FnMut(SectorId) -> Result<f64, FacadeError<R>> + '_
where
    R: TypedSectorAdmission,
    R::Mode: TypedSpaceModeDispatch<R>,
{
    let mut last: Option<(SectorId, f64)> = None;
    move |sector| match last {
        Some((cached, weight)) if cached == sector => Ok(weight),
        _ => {
            let weight = <R::Mode as TypedSpaceModeDispatch<R>>::dim(provider, sector)?;
            last = Some((sector, weight));
            Ok(weight)
        }
    }
}

impl<R, D, S> TensorMap<R, D, S>
where
    R: TypedSectorAdmission,
    R::Mode: TypedSpaceModeDispatch<R>,
    D: TensorScalar,
    S: ReduceExec<R, D>,
{
    /// TensorKit `norm(t, p)`: the entrywise `p`-norm of the reduced blocks,
    ///
    /// ```text
    /// p == 2       -> sqrt(sum_c dim(c) * sum_ij |self_c[i,j]|^2)
    /// p == Inf     -> max_c max_ij |self_c[i,j]|          (not dim-weighted)
    /// finite p > 0 -> (sum_c dim(c) * sum_ij |self_c[i,j]|^p)^(1/p)
    /// ```
    ///
    /// `p == 2.0` is the quantum-dimension-weighted Frobenius norm and is the
    /// only exponent the CUDA storage supports; `p` is an `f64` exactly as in
    /// TensorKit, so each norm has one spelling. The entrywise norm is never
    /// an operator norm, matrices included.
    ///
    /// For `p == 2`, abelian providers have `dim(c) = 1`, giving the ordinary Frobenius
    /// norm. Compact diagonal input is reduced directly in `O(sum_c k_c)`;
    /// dense input is one pass over the payload. Lazy adjoints read their
    /// parent orientation without materializing. Both admission modes share
    /// this one body; they differ only in how `dim(c)` is asked for.
    ///
    /// The Host norm does not overflow or underflow while the norm itself is
    /// representable: when the unscaled sum of squares leaves the `f64` range,
    /// two more passes rescale every entry by the largest magnitude, as Julia's
    /// `LinearAlgebra.generic_norm2` does. An entry of NaN magnitude `|x|` gives
    /// NaN; otherwise an infinite magnitude gives `inf`.
    ///
    /// `p == Inf` follows Julia's NaN-propagating `max`: a payload holding
    /// any NaN, including a complex entry whose real or imaginary part alone
    /// is NaN, returns NaN; an infinite entry returns `+inf`; a tensor with no
    /// stored entries returns `+0.0`. Every exponent is one pass over the
    /// payload, `O(sum_c k_c)` on compact diagonal storage.
    ///
    /// # CUDA storage
    ///
    /// The device norm accumulates in `f64` at every payload dtype, within
    /// and across coupled sectors, like the Host norm: an `f32`/`Complex32`
    /// payload is widened on the device by one cast (one device allocation
    /// of twice the payload bytes and one elementwise pass), because
    /// Tenferro 0.7.1 offers no widening reduction (#1344). Unlike the Host
    /// norm, an `f64`/`Complex64` device sum is not rescaled: it returns
    /// `inf` once the squared norm exceeds `f64::MAX` and loses accuracy
    /// below `f64::MIN_POSITIVE`.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] when `p` is NaN, zero, negative, or
    ///   `-inf`; TensorKit throws `ArgumentError` over the same domain.
    /// - [`Error::UnsupportedOnDevice`] for any other `p != 2` on CUDA
    ///   storage, rather than a hidden Host transfer.
    /// - If a checked provider cannot supply a quantum dimension, its original
    ///   error is available as the source. An invalid coupled-sector layout
    ///   returns [`Error::Core`].
    /// - A multiplicity-free provider whose `Complex64` quantum dimension has
    ///   a nonzero imaginary part (no unitary category has one) is refused
    ///   with [`Error::Operation`] wrapping
    ///   [`OperationError::InvalidArgument`](crate::typed::OperationError::InvalidArgument).
    ///   Unique-fusion providers never read the dimension here, as TensorKit's
    ///   `norm` and `inner` do not.
    ///   [`Self::inner`] and [`Self::tr`] report the same refusal (`tr` also
    ///   for Unique fusion).
    ///
    /// ```
    /// use std::sync::Arc;
    /// use symtn::sector::{U1FusionRule, U1Irrep};
    /// use symtn::typed::{GradedSpace, Runtime, TensorMap};
    ///
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
    /// let id: TensorMap<_, f64> = TensorMap::isomorphism(&runtime, [&v], [&v])?;
    /// let twice = id.axpby(1.0, &id, 1.0)?;
    /// assert!((twice.norm(2.0)? - 2.0_f64.sqrt() * 2.0).abs() < 1e-12);
    /// assert_eq!(twice.norm(f64::INFINITY)?, 2.0);
    /// assert_eq!(id.inner(&id)?, 2.0);
    /// assert_eq!(id.tr()?, 2.0);
    /// # Ok::<(), symtn::typed::Error>(())
    /// ```
    pub fn norm(&self, p: f64) -> Result<f64, FacadeError<R>> {
        // Checked before any dispatch so an invalid `p` is rejected the same
        // way on every representation and storage.
        validate_norm_p(p)?;
        if let Some(spectrum) = S::as_host(EXEC, self).and_then(TensorMap::spectrum) {
            return compact_norm(spectrum, self.logical_space().provider(), p);
        }
        // Every entrywise norm is adjoint invariant: reduce the parent's
        // payload without materializing the logical adjoint.
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            return S::dense_norm(
                EXEC,
                &Self {
                    runtime: self.runtime.clone(),
                    repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
                },
                p,
            );
        }
        S::dense_norm(EXEC, self, p)
    }

    /// Returns the quantum-dimension-weighted Frobenius inner product
    /// `sum_c dim(c) * sum_ij conj(self_c[i,j]) * other_c[i,j]`.
    ///
    /// The product is conjugate-linear in `self`, and `self.inner(&self)` is
    /// `self.norm(2.0)^2` up to floating-point error. Both tensors must share the
    /// same runtime, hom space, and block layout. A compact
    /// diagonal operand is reduced directly from its stored spectrum in
    /// `O(sum_c k_c)` payload reads, including against a dense lazy adjoint;
    /// it is never densified. Its off-diagonal entries are structural zeros,
    /// so matching dense off-diagonal values are not read even when they are
    /// `NaN` or infinite. This deliberately differs from TensorKit 0.17's
    /// current generic mixed-block reduction, which visits those stored dense
    /// positions and therefore propagates their non-finite values. See
    /// [`Self::norm`] for the weighting, lazy behavior, and example.
    ///
    /// On Host storage owned dense sums use a fixed multi-accumulator order, widening
    /// single-precision inputs to double precision before arithmetic. If an
    /// intermediate partial sum overflows, the result is unspecified among
    /// ±Inf/NaN.
    ///
    /// On CUDA storage the sum accumulates in `f64` within and across coupled
    /// sectors at every payload dtype and narrows to `D` once, as on the
    /// Host; a single-precision call widens each distinct operand on the
    /// device first (#1383). A lazy adjoint CUDA operand returns
    /// [`Error::UnsupportedOnDevice`].
    #[doc(alias = "dot")]
    pub fn inner<'a>(&self, other: impl Into<TensorRef<'a, R, D, S>>) -> Result<D, FacadeError<R>> {
        let other = other.into().operand()?;
        let other = &*other;
        super::linear_ops::require_same_runtime_and_space(self, other)?;
        if let (Some(lhs), Some(rhs)) = (S::as_host(EXEC, self), S::as_host(EXEC, other)) {
            if let Some(value) = lhs.compact_inner_arms(rhs)? {
                return Ok(value);
            }
        }
        S::dense_inner(EXEC, self, other)
    }
}

/// The compact arm of [`TensorMap::norm`] over a validated `p`: the stored
/// diagonal of each coupled block, in `O(sum_c k_c)`.
fn compact_norm<R, D>(
    spectrum: &[symtn_matrixalgebra::SectorSpectrum<D>],
    provider: &R,
    p: f64,
) -> Result<f64, FacadeError<R>>
where
    R: TypedSectorAdmission,
    R::Mode: TypedSpaceModeDispatch<R>,
    D: TensorScalar,
{
    if p == 2.0 {
        return rescaled_power_norm(
            compact_inner(spectrum, spectrum, run_cached_dim(provider))?.re,
            2.0,
            || spectrum_max_abs(spectrum),
            |max| {
                spectrum_weighted_sum(spectrum, run_cached_dim(provider), |value| {
                    scaled_power(value, max, 2.0)
                })
            },
        );
    }
    if p.is_infinite() {
        return Ok(spectrum_max_abs(spectrum));
    }
    let power = |value: D| value.widen_complex().norm().powf(p);
    rescaled_power_norm(
        spectrum_weighted_sum(spectrum, run_cached_dim(provider), power)?,
        p,
        || spectrum_max_abs(spectrum),
        |max| {
            spectrum_weighted_sum(spectrum, run_cached_dim(provider), |value| {
                scaled_power(value, max, p)
            })
        },
    )
}

/// The dense kernels of `norm` and `inner` for one storage (#1756).
///
/// [`TensorMap::norm`] and [`TensorMap::inner`] own admission, the compact
/// arms and the lazy-adjoint redirect of `norm`; each calls exactly one of
/// these kernels last. A storage without a kernel for an argument returns
/// [`Error::UnsupportedOnDevice`] there.
#[doc(hidden)]
pub trait ReduceExec<R, D>: TypedStorage<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedSpaceModeDispatch<R>,
{
    /// `norm(t, p)` of an owned dense tensor, `p` already validated.
    fn dense_norm(_: ExecToken, t: &TensorMap<R, D, Self>, p: f64) -> Result<f64, FacadeError<R>>;

    /// `inner(x, y)` of two dense operands, owned or lazy adjoint, on one
    /// runtime and one space.
    fn dense_inner(
        _: ExecToken,
        x: &TensorMap<R, D, Self>,
        y: &TensorMap<R, D, Self>,
    ) -> Result<D, FacadeError<R>>;
}

impl<R, D> ReduceExec<R, D> for Vec<D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedSpaceModeDispatch<R>,
    D: TensorScalar,
{
    fn dense_norm(_: ExecToken, t: &TensorMap<R, D>, p: f64) -> Result<f64, FacadeError<R>> {
        let provider = t.logical_space().provider();
        let payload = t
            .owned_body()
            .expect("owned norm input")
            .materialized_dense_data();
        let data: &[D] = &payload;
        if p.is_infinite() {
            return Ok(max_abs(data.iter().copied()));
        }
        let structure = t.logical_space().space().structure();
        let nout = t.logical_space().space().nout();
        let sum = if p == 2.0 {
            let unique = <R::Mode as TypedTensorModeDispatch<R>>::fusion_style(provider)
                == symtn_core::FusionStyleKind::Unique;
            // Why not `t.inner(t)`: it narrows the wide sum to `D`, which
            // for `f32`/`Complex32` rounds `|t|²` to single precision and can
            // leave it subnormal, above the rescaling threshold.
            dense_weighted_inner(
                unique,
                structure,
                nout,
                data,
                data,
                run_cached_dim(provider),
            )?
            .re
        } else {
            coupled_region_weighted_sum(structure, nout, data, run_cached_dim(provider), |value| {
                value.widen_complex().norm().powf(p)
            })?
        };
        rescaled_power_norm(
            sum,
            p,
            || max_abs(data.iter().copied()),
            |max| {
                coupled_region_weighted_sum(
                    structure,
                    nout,
                    data,
                    run_cached_dim(provider),
                    |value| scaled_power(value, max, p),
                )
            },
        )
    }

    fn dense_inner(
        _: ExecToken,
        x: &TensorMap<R, D>,
        y: &TensorMap<R, D>,
    ) -> Result<D, FacadeError<R>> {
        let _host_pool = x.runtime.enter_host_pool();
        let provider = x.logical_space().provider();
        if matches!(&x.repr, TypedTensorRepr::Adjoint(_))
            || matches!(&y.repr, TypedTensorRepr::Adjoint(_))
        {
            let (lhs_operand, lhs_data) = x.fusion_operand_and_data();
            let (rhs_operand, rhs_data) = y.fusion_operand_and_data();
            return match (&x.repr, &y.repr) {
                (TypedTensorRepr::Adjoint(lhs), TypedTensorRepr::Adjoint(rhs)) => {
                    symtn_tensors::oriented_fusion_inner_with(
                        lhs.parent.space.space().structure(),
                        symtn_tensors::FusionOperand::direct(rhs.parent.space.space()),
                        rhs.parent_data(),
                        symtn_tensors::FusionOperand::direct(lhs.parent.space.space()),
                        lhs.parent_data(),
                        run_cached_dim(provider),
                    )
                }
                _ => symtn_tensors::oriented_fusion_inner_with(
                    x.logical_space().space().structure(),
                    lhs_operand,
                    &lhs_data,
                    rhs_operand,
                    &rhs_data,
                    run_cached_dim(provider),
                ),
            };
        }
        let unique = <R::Mode as TypedTensorModeDispatch<R>>::fusion_style(provider)
            == symtn_core::FusionStyleKind::Unique;
        // `D::from_complex64` is `.re` for the real scalar and the identity for
        // the complex one, so one static conversion covers both scalar types.
        Ok(D::from_complex64(dense_weighted_inner(
            unique,
            x.logical_space().space().structure(),
            x.logical_space().space().nout(),
            x.owned_body()
                .expect("owned inner input")
                .materialized_dense_data()
                .as_ref(),
            y.owned_body()
                .expect("owned inner input")
                .materialized_dense_data()
                .as_ref(),
            run_cached_dim(provider),
        )?))
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedSpaceModeDispatch<R>,
    D: TensorScalar,
{
    /// The compact arms of [`Self::inner`]: `None` when neither operand is
    /// compact. Compact operands reduce without materializing their
    /// structural zeros.
    fn compact_inner_arms(&self, other: &Self) -> Result<Option<D>, FacadeError<R>> {
        let value = match (self.spectrum(), other.spectrum()) {
            (Some(lhs), Some(rhs)) => {
                compact_inner(lhs, rhs, run_cached_dim(self.logical_space().provider()))?
            }
            (Some(lhs), None) => Self::compact_dense_inner(lhs, other, true)?,
            (None, Some(rhs)) => Self::compact_dense_inner(rhs, self, false)?,
            (None, None) => return Ok(None),
        };
        Ok(Some(D::from_complex64(value)))
    }

    /// Mixed compact/dense inner over the stored diagonal only. A compact
    /// operand defines structural zeros off diagonal, so those dense entries
    /// are not part of this reduction (including non-finite values).
    fn compact_dense_inner(
        spectrum: &[symtn_matrixalgebra::SectorSpectrum<D>],
        dense: &Self,
        compact_is_lhs: bool,
    ) -> Result<num_complex::Complex64, FacadeError<R>> {
        let logical_structure = dense.logical_space().space().structure();
        let (data_structure, data, dense_is_adjoint) = match &dense.repr {
            TypedTensorRepr::Owned(body) => {
                let TypedData::Dense(data) = body.data.as_ref() else {
                    unreachable!("mixed compact/dense dispatch requires a dense operand")
                };
                (body.space.space().structure(), data.as_slice(), false)
            }
            TypedTensorRepr::Adjoint(view) => (
                view.parent.space.space().structure(),
                view.parent_data(),
                true,
            ),
        };
        if logical_structure.block_count() != spectrum.len()
            || data_structure.required_len().map_err(Error::from)? != data.len()
        {
            return Err(spectra_disagree().into());
        }
        // A compact operand proves a rank-(1,1) bond endomorphism. Its equal
        // codomain and domain make the adjoint parent's admitted structure
        // content-identical to the logical structure, including block order.
        // Compare that contract once in O(G), then traverse both in lockstep;
        // per-block indexed lookup would make this O(G log G).
        if dense_is_adjoint && logical_structure != data_structure {
            return Err(internal_layout_error(
                "compact/dense inner adjoint parent has a different bond layout",
            )
            .into());
        }

        let provider = dense.logical_space().provider();
        let mut total = num_complex::Complex64::new(0.0, 0.0);
        for (index, entry) in spectrum.iter().enumerate() {
            let logical_block = logical_structure.block(index).map_err(Error::from)?;
            let Some(pair) = logical_block.key().as_fusion_tree_pair() else {
                return Err(spectra_disagree().into());
            };
            if pair.codomain_tree().coupled() != entry.sector
                || logical_block.shape().len() != 2
                || logical_block.shape()[0] != logical_block.shape()[1]
                || logical_block.shape()[0] != entry.values.len()
            {
                return Err(spectra_disagree().into());
            }
            let data_block = data_structure.block(index).map_err(Error::from)?;
            if data_block.shape().len() != 2
                || data_block.shape()[0] != data_block.shape()[1]
                || data_block.shape()[0] != entry.values.len()
            {
                return Err(spectra_disagree().into());
            }
            let stride = data_block.strides()[0] + data_block.strides()[1];
            let mut partial = D::Wide::from_real(0.0);
            for (i, &compact_value) in entry.values.iter().enumerate() {
                let dense_value = *data.get(data_block.offset() + i * stride).ok_or_else(|| {
                    internal_layout_error("compact/dense inner diagonal exceeds scalar storage")
                })?;
                let compact_value = compact_value.widen();
                let dense_value = dense_value.widen();
                partial = partial
                    + match (compact_is_lhs, dense_is_adjoint) {
                        (true, false) => FactorScalar::adjoint(compact_value) * dense_value,
                        (true, true) => {
                            FactorScalar::adjoint(compact_value)
                                * FactorScalar::adjoint(dense_value)
                        }
                        (false, false) => FactorScalar::adjoint(dense_value) * compact_value,
                        (false, true) => dense_value * compact_value,
                    };
            }
            total += partial.widen_complex()
                * <R::Mode as TypedSpaceModeDispatch<R>>::dim(provider, entry.sector)?;
        }
        Ok(total)
    }

    /// The parent of a lazy adjoint, as an owned tensor.
    fn adjoint_parent(view: &Arc<TypedAdjointView<R, D>>, runtime: &Runtime) -> Self {
        Self {
            runtime: runtime.clone(),
            repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
        }
    }

    /// Returns the quantum-dimension-weighted block trace
    /// `sum_c dim(c) * Tr(self_c)` of an endomorphism.
    ///
    /// The codomain and domain legs must be exactly equal. This is TensorKit's
    /// `tr`; it is not a positivity claim, and the result may be negative or
    /// complex. It also is not the fermionic supertrace: no twist factor is
    /// applied. Use [`Self::trace_pairs`] for the categorical contraction
    /// trace, which includes the provider's pivotal/twist data.
    ///
    /// Compact diagonal input is summed directly in `O(sum_c k_c)`. A
    /// lazy adjoint returns the conjugate of its parent's trace without
    /// materializing. See [`Self::norm`] for a runnable example.
    ///
    /// The weight is the real quantum dimension also for a rule with
    /// complex categorical coefficients (TensorKit's real `dimscalartype`):
    ///
    /// ```
    /// use std::sync::Arc;
    /// use symtn::sector::{TableRule, TableSector};
    /// use symtn::typed::{Complex64, GradedSpace, Runtime, TensorMap};
    ///
    /// let runtime = Runtime::builder().build()?;
    /// let tau = GradedSpace::try_new(
    ///     Arc::new(TableRule::fibonacci()),
    ///     [(TableSector::new(2), 1)],
    /// )?;
    /// let id: TensorMap<TableRule, Complex64> =
    ///     TensorMap::isomorphism(&runtime, [&tau], [&tau])?;
    /// let golden_ratio = (1.0 + 5.0_f64.sqrt()) / 2.0;
    /// assert!((id.tr()? - Complex64::new(golden_ratio, 0.0)).norm() < 1e-15);
    /// # Ok::<(), symtn::typed::Error>(())
    /// ```
    pub fn tr(&self) -> Result<D, FacadeError<R>> {
        let hom = self.logical_space().space().homspace();
        // The weighted trace below indexes codomain axis `i` together with
        // domain axis `nout + i` and would be meaningless without this check.
        if hom.codomain().legs() != hom.domain().legs() {
            return Err(Error::from(symtn_tensors::OperationError::SpaceMismatch {
                message: "tr() requires an endomorphism (domain == codomain)",
            })
            .into());
        }
        let provider = self.logical_space().provider();
        if let Some(spectrum) = self.spectrum() {
            // `Σ_c dim(c) * Σ_i d_i`: TensorKit's block trace on a
            // `DiagonalTensorMap`, read straight off the stored values.
            let mut weight_of = run_cached_dim(provider);
            let mut total = num_complex::Complex64::new(0.0, 0.0);
            for entry in spectrum {
                let mut partial = D::Wide::from_real(0.0);
                for &value in &entry.values {
                    partial = partial + value.widen();
                }
                total += partial.widen_complex() * weight_of(entry.sector)?;
            }
            return Ok(D::from_complex64(total));
        }
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            return Ok(FactorScalar::adjoint(
                Self::adjoint_parent(view, &self.runtime).tr()?,
            ));
        }
        Ok(D::from_complex64(weighted_trace(
            self.logical_space().space().structure(),
            self.logical_space().space().nout(),
            self.owned_body()
                .expect("owned trace input")
                .materialized_dense_data()
                .as_ref(),
            run_cached_dim(provider),
        )?))
    }
}
