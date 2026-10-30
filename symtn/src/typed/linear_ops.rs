use super::*;

// The `re`/`im` gates are law checks (`re(t) + i·im(t)` rebuilds `t`).
// TensorKit's real-input
// branches (`real(t) = t`, `imag(t) = zerovector(t)` for a real scalartype)
// are statically unrepresentable here: these methods exist on the `Complex64`
// impl only, and `to_c64().re()` covers the round trip.
impl<R> TensorMap<R, num_complex::Complex64> {
    /// The element-wise real part, as an f64 tensor map on the same spaces
    /// (TensorKit `Base.real`: blockwise element-wise, result scalartype
    /// real).
    ///
    /// A compact spectrum maps spectrum-to-spectrum and stays compact. A cold
    /// lazy adjoint uses an operation-local payload for the `O(stored_len)`
    /// output; the logical space is shared.
    ///
    /// # Errors
    ///
    /// The materialization of a lazy adjoint input can fail on the engine.
    pub fn re(&self, engine: &mut dyn Engine) -> Result<TensorMap<R, f64>, Error> {
        self.map_parts(engine, |value| value.re)
    }

    /// The element-wise imaginary part, as an f64 tensor map on the same
    /// spaces (TensorKit `Base.imag`).
    ///
    /// A compact spectrum maps spectrum-to-spectrum and stays compact. A cold
    /// lazy adjoint uses an operation-local payload for the `O(stored_len)`
    /// output; the logical space is shared.
    ///
    /// # Errors
    ///
    /// As [`Self::re`].
    pub fn im(&self, engine: &mut dyn Engine) -> Result<TensorMap<R, f64>, Error> {
        self.map_parts(engine, |value| value.im)
    }

    /// The shared owned-input route of [`Self::re`] / [`Self::im`].
    fn map_parts(
        &self,
        engine: &mut dyn Engine,
        part: impl Fn(num_complex::Complex64) -> f64,
    ) -> Result<TensorMap<R, f64>, Error> {
        let materialized = self.materialized_tensor_uncached(engine)?;
        let source = materialized
            .owned_body()
            .expect("uncached materialization is owned");
        let body = match source.data.as_ref() {
            TypedData::Dense(data) => TypedTensorBody::dense(
                source.space.clone(),
                data.iter().map(|&value| part(value)).collect(),
            ),
            TypedData::Diagonal(spectrum) => {
                TypedTensorBody::diagonal(source.space.clone(), map_spectrum_dtype(spectrum, part))
            }
        };
        Ok(TensorMap {
            runtime: self.runtime.clone(),
            repr: owned_repr(body),
        })
    }
}

impl<R, D, S> TensorMap<R, D, S>
where
    R: TypedSectorAdmission,
    R::Mode: TypedAdjointSpace<R>,
    D: TensorScalar,
    S: ElementwiseExec<R, D>,
{
    /// Returns the linear combination
    /// `x.axpby(alpha, &y, beta) == alpha * x + beta * y` on the operands'
    /// common tensor space, where `x` is the receiver.
    ///
    /// The name is BLAS `axpby`: each coefficient sits next to the operand it
    /// multiplies. symtn has no `add` with coefficients because
    /// VectorInterface's `add(y, x, α, β)` computes `β·y + α·x`, binding the
    /// coefficients the other way round; a ported `add` call would compile and
    /// silently swap them.
    ///
    /// Both operands must share a runtime and exactly the same tensor space and
    /// block layout. Two compact diagonal inputs stay compact. A mixed
    /// compact/dense pair allocates only the dense result. Two lazy adjoints
    /// give the lazy adjoint `adjoint(conj(alpha) * x' + conj(beta) * y')` of
    /// their parents' combination, so only the parent-sized sum is allocated;
    /// one lazy operand is read in its logical orientation without filling its
    /// cache. A zero coefficient drops its operand, NaN and Inf included
    /// (TensorKit's `add`); signed-zero bits are storage-local.
    ///
    /// Returns [`Error::RuntimeMismatch`] or `SpaceMismatch` (unequal spaces or block layouts) before
    /// producing a result. See [`Self::norm`] for a runnable example.
    ///
    /// # CUDA storage
    ///
    /// The same rule, through one body for every representation (#2187):
    /// each nonzero-coefficient operand is read by one strided `dot_general`
    /// per stored block, an owned operand as one packed block and a lazy one
    /// from its parent's blocks, conjugating (#1874), with its coefficient as
    /// the descriptor scale. Nothing is materialized and nothing crosses the
    /// host boundary except the output's zero initialization (#740).
    ///
    /// ```compile_fail
    /// use symtn::sector::U1FusionRule;
    /// use symtn::typed::TensorMap;
    ///
    /// fn removed(x: &TensorMap<U1FusionRule, f64>) {
    ///     let _ = x.add(x, 1.0, 2.0);
    /// }
    /// ```
    pub fn axpby<'a>(
        &self,
        engine: &mut dyn Engine,
        alpha: D,
        y: impl Into<TensorRef<'a, R, D, S>>,
        beta: D,
    ) -> Result<Self, FacadeError<R>> {
        let y = y.into().operand()?;
        Ok(self.axpby_checked(engine, alpha, &y, beta)?)
    }

    /// The one body of [`Self::axpby`].
    fn axpby_checked(
        &self,
        engine: &mut dyn Engine,
        alpha: D,
        other: &Self,
        beta: D,
    ) -> Result<Self, Error> {
        // Runtime first, exactly as `contract` does: crossing runtimes is a
        // trust-boundary violation rather than an algebra error.
        require_same_runtime_and_space(self, other)?;
        if let (Some(x), Some(y)) = (S::as_host(EXEC, self), S::as_host(EXEC, other)) {
            if let Some(sum) = x.compact_axpby(alpha, y, beta)? {
                return S::from_host(EXEC, sum);
            }
        }
        if let (TypedTensorRepr::Adjoint(x), TypedTensorRepr::Adjoint(y)) =
            (&self.repr, &other.repr)
        {
            // `alpha A^H + beta B^H == (conj(alpha) A + conj(beta) B)^H`.
            let sum = self.parent_of(x).axpby_checked(
                engine,
                FactorScalar::adjoint(alpha),
                &other.parent_of(y),
                FactorScalar::adjoint(beta),
            )?;
            return sum.lazy_adjoint_on(&x.logical_space);
        }
        S::combine(engine, EXEC, self, alpha, Some((other, beta)))
    }

    /// Returns `factor * self`.
    ///
    /// Compact diagonal storage stays compact. A dense lazy adjoint remains
    /// lazy by scaling its parent with the conjugated factor, so no receiver
    /// materialization is cached. A zero factor gives exact zeros whatever
    /// `self` holds (TensorKit's `scale(x, 0)`); signed-zero bits are
    /// storage-local.
    pub fn scale(&self, engine: &mut dyn Engine, factor: D) -> Result<Self, FacadeError<R>> {
        if let Some(spectrum) = self.spectrum() {
            return Ok(self.with_spectrum(
                spectrum
                    .iter()
                    .map(|entry| symtn_matrixalgebra::SectorSpectrum {
                        sector: entry.sector,
                        values: entry
                            .values
                            .iter()
                            .map(|&value| scale_value(value, factor))
                            .collect(),
                    })
                    .collect(),
            ));
        }
        if let TypedTensorRepr::Adjoint(view) = &self.repr {
            let parent = self
                .parent_of(view)
                .scale(engine, FactorScalar::adjoint(factor))?;
            return Ok(parent.lazy_adjoint_on(&view.logical_space)?);
        }
        Ok(S::combine(engine, EXEC, self, factor, None)?)
    }

    /// `destination = alpha * self + beta * destination`: BLAS `axpby`,
    /// TensorKit `add!(ty, tx, α, β)`, in place on the destination's dense
    /// payload.
    ///
    /// Each element is `scale(destination, beta) + scale(self, alpha)` with
    /// VectorInterface's strong zero, in one pass: `beta == 0` does not let a
    /// NaN in `destination` through, and `alpha == 0` does not read `self`.
    /// A lazy-adjoint `self` is read in its logical orientation without
    /// filling its cache; a compact diagonal `self` is added onto the
    /// diagonal after `destination` is scaled.
    ///
    /// # CUDA storage
    ///
    /// One `dot_general` over the whole payload with `alpha` as its
    /// descriptor scale and `beta` in its epilogue; nothing is uploaded or
    /// allocated on a warm context. A lazy-adjoint `self` is read from its
    /// parent with one conjugating strided `dot_general` per block, likewise
    /// upload-free (#1874). A payload on another device returns
    /// [`Error::PlacementMismatch`].
    ///
    /// # Errors
    ///
    /// [`Error::RuntimeMismatch`]; `SpaceMismatch` when the two tensors live
    /// on different spaces or block layouts; [`Error::InvalidArgument`] when `destination`
    /// is not owned dense storage, or when it aliases `self`;
    /// [`Error::DestinationShared`] when `destination` shares its storage with
    /// a clone. TensorKit checks the destination space before the kernel, so
    /// every destination error precedes a storage capability error.
    ///
    /// # Failure
    ///
    /// A validation or capability error leaves `destination` bit-identical:
    /// every check completes before the first write. On failure after
    /// validation (a backend or other execution error) the destination's
    /// contents are unspecified, while its space and block structure stay
    /// intact.
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use symtn::sector::{U1FusionRule, U1Irrep};
    /// use symtn::typed::{Error, GradedSpace, Runtime, TensorMap};
    ///
    /// let engine = &mut symtn::typed::CpuEngine::serial();
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
    /// let x: TensorMap<_, f64> = TensorMap::rand_with_seed(engine, &runtime, [&v], [&v], 1)?;
    /// let mut y = x.zeros_like(engine)?;
    /// x.axpby_into(engine, &mut y, 2.0, 0.0)?;
    /// assert_eq!(y.dense_data()?, x.scale(engine, 2.0)?.dense_data()?);
    ///
    /// // A clone shares the payload, so it is not a valid destination.
    /// let shared = y.clone();
    /// assert_eq!(x.axpby_into(engine, &mut y, 1.0, 1.0), Err(Error::DestinationShared));
    /// drop(shared);
    /// # Ok::<(), symtn::typed::Error>(())
    /// ```
    pub fn axpby_into(
        &self,
        engine: &mut dyn Engine,
        destination: &mut Self,
        alpha: D,
        beta: D,
    ) -> Result<(), FacadeError<R>> {
        Ok(axpby_into_checked(engine, self, destination, alpha, beta)?)
    }
}

impl<R, D, S> TensorMap<R, D, S> {
    /// The owned tensor on a lazy adjoint's parent payload.
    pub(super) fn parent_of(&self, view: &TypedAdjointView<R, D, S>) -> Self {
        Self {
            runtime: self.runtime.clone(),
            repr: TypedTensorRepr::Owned(Arc::clone(&view.parent)),
        }
    }

    /// `adjoint(self)` for an owned dense result whose adjoint space is
    /// already known: the elementwise redirects through a lazy parent keep
    /// the parent's space, so the operand's logical space is the result's,
    /// with no adjoint-space query.
    pub(super) fn lazy_adjoint_on(
        self,
        logical_space: &BoundDynamicFusionMapSpace<R>,
    ) -> Result<Self, Error> {
        let TypedTensorRepr::Owned(parent) = self.repr else {
            return Err(internal_layout_error(
                "an elementwise parent result is owned",
            ));
        };
        Ok(Self {
            runtime: self.runtime,
            repr: TypedTensorRepr::Adjoint(Arc::new(TypedAdjointView::new(
                parent,
                logical_space.clone(),
            ))),
        })
    }
}

impl<R, D> TensorMap<R, D>
where
    D: TensorScalar,
{
    /// The compact arms of [`TensorMap::axpby`]: `None` unless an operand is
    /// a compact diagonal. Equal spaces are checked by the caller.
    fn compact_axpby(&self, alpha: D, other: &Self, beta: D) -> Result<Option<Self>, Error> {
        let (lhs, rhs) = (self.spectrum(), other.spectrum());
        if lhs.is_none() && rhs.is_none() {
            return Ok(None);
        }
        let sum = match (lhs, rhs) {
            (None, None) => return Ok(None),
            // A lazy adjoint against a spectrum: the oriented sum of the
            // lazy operand, then the spectrum added onto its diagonal.
            (Some(spectrum), None) if matches!(&other.repr, TypedTensorRepr::Adjoint(_)) => {
                self.with_data(lazy_plus_spectrum(other, beta, spectrum, alpha)?)
            }
            (None, Some(spectrum)) if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) => {
                self.with_data(lazy_plus_spectrum(self, alpha, spectrum, beta)?)
            }
            // Two spectra on one bond space: the sum is diagonal too.
            (Some(lhs), Some(rhs)) => {
                if lhs.len() != rhs.len() {
                    return Err(spectra_disagree());
                }
                let sum = lhs
                    .iter()
                    .zip(rhs)
                    .map(|(left, right)| {
                        if left.sector != right.sector || left.values.len() != right.values.len() {
                            return Err(spectra_disagree());
                        }
                        Ok(symtn_matrixalgebra::SectorSpectrum {
                            sector: left.sector,
                            values: left
                                .values
                                .iter()
                                .zip(&right.values)
                                .map(|(&x, &y)| scale_value(x, alpha) + scale_value(y, beta))
                                .collect(),
                        })
                    })
                    .collect::<Result<Vec<_>, Error>>()?;
                self.with_spectrum(sum)
            }
            // Mixed: the result is dense, but the *diagonal operand* is never
            // materialized to get there — the one O(n²) buffer this needs is
            // the owned result, which scatters the spectrum onto its own
            // diagonal. Materializing first would allocate a second.
            (Some(diagonal), None) => self.with_data(scatter_spectrum(
                self.logical_space().space(),
                other
                    .owned_body()
                    .expect("owned add input")
                    .materialized_dense_data()
                    .as_ref(),
                beta,
                diagonal,
                alpha,
            )?),
            (None, Some(diagonal)) => self.with_data(scatter_spectrum(
                self.logical_space().space(),
                self.owned_body()
                    .expect("owned add input")
                    .materialized_dense_data()
                    .as_ref(),
                alpha,
                diagonal,
                beta,
            )?),
        };
        Ok(Some(sum))
    }
}

/// `factor * lazy + diagonal_factor * spectrum` as a dense payload on the
/// lazy operand's logical space.
fn lazy_plus_spectrum<R, D: TensorScalar>(
    lazy: &TensorMap<R, D>,
    factor: D,
    spectrum: &[symtn_matrixalgebra::SectorSpectrum<D>],
    diagonal_factor: D,
) -> Result<Vec<D>, Error> {
    let (operand, dense) = lazy.fusion_operand_and_data();
    let mut data = symtn_tensors::oriented_fusion_add_owned(
        lazy.logical_space().space().structure(),
        operand,
        &dense,
        operand,
        &dense,
        factor,
        D::from_real(0.0),
    )?;
    add_spectrum_into(
        lazy.logical_space().space(),
        &mut data,
        spectrum,
        diagonal_factor,
    )?;
    Ok(data)
}

/// The elementwise kernels for one storage (#1756).
///
/// [`TensorMap::scale`], [`TensorMap::zeros_like`], [`TensorMap::axpby`]
/// and [`TensorMap::axpby_into`] own admission, the compact arms and the
/// lazy-adjoint redirects; each calls exactly one of these kernels last,
/// after every no-write check. A storage without a kernel for an operand
/// representation returns [`Error::UnsupportedOnDevice`] there.
#[doc(hidden)]
pub trait ElementwiseExec<R, D>: TypedStorage<R, D> {
    /// A fresh exact-zero dense payload on an owned dense `t`'s space.
    fn zeros(
        engine: &mut dyn Engine,
        _: ExecToken,
        t: &TensorMap<R, D, Self>,
    ) -> Result<TensorMap<R, D, Self>, Error>;

    /// `alpha * x + beta * y`, or `alpha * x` without `y`, as a fresh owned
    /// dense result on `x`'s logical space. The operands are dense, share
    /// one runtime and space, and at most one is a lazy adjoint; `x` alone
    /// is owned.
    fn combine(
        engine: &mut dyn Engine,
        _: ExecToken,
        x: &TensorMap<R, D, Self>,
        alpha: D,
        y: Option<(&TensorMap<R, D, Self>, D)>,
    ) -> Result<TensorMap<R, D, Self>, Error>;

    /// `destination = alpha * x + beta * destination` into an admitted
    /// destination: same runtime and space, owned dense, unaliased, unique.
    fn axpby_into(
        engine: &mut dyn Engine,
        _: ExecToken,
        x: &TensorMap<R, D, Self>,
        destination: &mut TensorMap<R, D, Self>,
        alpha: D,
        beta: D,
    ) -> Result<(), Error>;
}

impl<R, D: TensorScalar> ElementwiseExec<R, D> for Vec<D> {
    fn zeros(
        _engine: &mut dyn Engine,
        _: ExecToken,
        t: &TensorMap<R, D>,
    ) -> Result<TensorMap<R, D>, Error> {
        Ok(t.with_data(vec![
            D::from_real(0.0);
            t.owned_body()
                .expect("owned zero input")
                .materialized_dense_data()
                .as_ref()
                .len()
        ]))
    }

    fn combine(
        _engine: &mut dyn Engine,
        _: ExecToken,
        x: &TensorMap<R, D>,
        alpha: D,
        y: Option<(&TensorMap<R, D>, D)>,
    ) -> Result<TensorMap<R, D>, Error> {
        let Some((y, beta)) = y else {
            let data = x
                .owned_body()
                .expect("owned scale input")
                .materialized_dense_data();
            let scaled = if alpha.is_zero() {
                vec![D::zero() * alpha; data.len()]
            } else {
                data.iter().map(|&value| alpha * value).collect()
            };
            return Ok(x.with_data(scaled));
        };
        if matches!(&x.repr, TypedTensorRepr::Adjoint(_))
            || matches!(&y.repr, TypedTensorRepr::Adjoint(_))
        {
            let (lhs, lhs_data) = x.fusion_operand_and_data();
            let (rhs, rhs_data) = y.fusion_operand_and_data();
            return Ok(x.with_data(symtn_tensors::oriented_fusion_add_owned(
                x.logical_space().space().structure(),
                lhs,
                &lhs_data,
                rhs,
                &rhs_data,
                alpha,
                beta,
            )?));
        }
        Ok(x.with_data(
            x.owned_body()
                .expect("owned add input")
                .materialized_dense_data()
                .as_ref()
                .iter()
                .zip(
                    y.owned_body()
                        .expect("owned add input")
                        .materialized_dense_data()
                        .as_ref(),
                )
                .map(|(&x, &y)| scale_value(x, alpha) + scale_value(y, beta))
                .collect(),
        ))
    }

    fn axpby_into(
        engine: &mut dyn Engine,
        _: ExecToken,
        x: &TensorMap<R, D>,
        destination: &mut TensorMap<R, D>,
        alpha: D,
        beta: D,
    ) -> Result<(), Error> {
        let TypedTensorRepr::Owned(destination_body) = &mut destination.repr else {
            return Err(internal_layout_error("ordinary destination checked above"));
        };
        let destination_data = Arc::get_mut(destination_body)
            .and_then(|body| Arc::get_mut(&mut body.data))
            .ok_or_else(|| internal_layout_error("unique destination checked above"))?;
        let TypedData::Dense(destination_data) = destination_data else {
            return Err(internal_layout_error("dense destination checked above"));
        };
        match &x.repr {
            TypedTensorRepr::Owned(body) => match body.data.as_ref() {
                TypedData::Dense(source) => {
                    if source.len() != destination_data.len() {
                        return Err(Error::InvalidArgument(
                            "tensors have different dense payload lengths".to_string(),
                        ));
                    }
                    for (value, &source) in destination_data.iter_mut().zip(source) {
                        *value = scale_value(*value, beta) + scale_value(source, alpha);
                    }
                }
                TypedData::Diagonal(spectrum) => {
                    for value in destination_data.iter_mut() {
                        *value = scale_value(*value, beta);
                    }
                    add_spectrum_into(
                        x.logical_space().space(),
                        destination_data,
                        spectrum,
                        alpha,
                    )?;
                }
            },
            TypedTensorRepr::Adjoint(_) => {
                let (operand, source) = x.fusion_operand_and_data();
                symtn_operations::host_pool::region(
                    engine,
                    symtn_operations::host_pool::HostPoolSite::Strided,
                    |exec| {
                        symtn_tensors::oriented_fusion_axpby_into(
                            exec,
                            x.logical_space().space().structure(),
                            destination_data,
                            operand,
                            &source,
                            alpha,
                            beta,
                        )
                    },
                )??;
            }
        }
        Ok(())
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedAdjointSpace<R>,
    D: TensorScalar,
{
    /// `self = factor * self` in place, VectorInterface `scale!(t, α)`, on
    /// the receiver's own storage: a dense payload element by element, a
    /// compact diagonal on its stored values (it stays compact, TensorKit's
    /// `scale!` over a `DiagonalTensorMap`'s blocks), and a lazy adjoint on
    /// its parent with the conjugated factor (it stays lazy, as
    /// [`Self::scale`] does). Nothing is allocated.
    ///
    /// # Errors
    ///
    /// [`Error::DestinationShared`] when the receiver shares its storage
    /// with another handle (a shallow `Clone`, or the parent of a lazy
    /// adjoint): writing it would change that handle too, and replacing it
    /// would silently allocate, so the call does neither and leaves the
    /// receiver unchanged. The scaling itself cannot fail.
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use symtn::sector::{U1FusionRule, U1Irrep};
    /// use symtn::typed::{Error, GradedSpace, Runtime, TensorMap};
    ///
    /// let engine = &mut symtn::typed::CpuEngine::serial();
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
    /// let mut t: TensorMap<_, f64> = TensorMap::rand_with_seed(engine, &runtime, [&v], [&v], 1)?;
    /// let expected = t.scale(engine, 2.0)?;
    /// t.scale_assign(engine, 2.0)?;
    /// assert_eq!(t.dense_data()?, expected.dense_data()?);
    ///
    /// let shared = t.clone();
    /// assert_eq!(t.scale_assign(engine, 2.0), Err(Error::DestinationShared));
    /// drop(shared);
    /// # Ok::<(), symtn::typed::Error>(())
    /// ```
    pub fn scale_assign(
        &mut self,
        engine: &mut dyn Engine,
        factor: D,
    ) -> Result<(), FacadeError<R>> {
        // Host-local work: the engine is the uniform call shape.
        let _ = engine;
        let (body, factor) = match &mut self.repr {
            TypedTensorRepr::Owned(body) => (Arc::get_mut(body), factor),
            TypedTensorRepr::Adjoint(view) => (
                Arc::get_mut(view).and_then(|view| Arc::get_mut(&mut view.parent)),
                FactorScalar::adjoint(factor),
            ),
        };
        let data = body
            .and_then(|body| Arc::get_mut(&mut body.data))
            .ok_or(Error::DestinationShared)?;
        if factor.is_zero() {
            let zero = D::zero() * factor;
            match data {
                TypedData::Dense(data) => data.fill(zero),
                TypedData::Diagonal(spectrum) => {
                    spectrum
                        .iter_mut()
                        .for_each(|entry| entry.values.fill(zero));
                }
            }
        } else {
            let scale = |value: &mut D| *value = factor * *value;
            match data {
                TypedData::Dense(data) => data.iter_mut().for_each(scale),
                TypedData::Diagonal(spectrum) => spectrum
                    .iter_mut()
                    .flat_map(|entry| entry.values.iter_mut())
                    .for_each(scale),
            }
        }
        Ok(())
    }
}

/// The destination's space against the operation result's, as TensorKit
/// compares `space(tdst)`; the same error as [`unique_dense_destination`]'s
/// space check, for routes that compare the space before their other checks.
pub(super) fn require_destination_space(
    destination: &DynamicFusionMapSpace,
    expected: &DynamicFusionMapSpace,
) -> Result<(), Error> {
    if destination != expected {
        return Err(Error::from(symtn_tensors::OperationError::SpaceMismatch {
            message: "destination fusion space or block layout does not match the operation result",
        }));
    }
    Ok(())
}

/// The destination checks a `*_into` shares on every placement, in one
/// order: [`destination_precheck`], the result's space and block layout,
/// then [`destination_postcheck`].
/// `storage` names the placement in the messages ("host", "CUDA").
pub(super) fn unique_dense_destination<'d, R, D, S>(
    destination: &'d TensorMap<R, D, S>,
    input: &Arc<TypedData<D, S>>,
    expected: &DynamicFusionMapSpace,
    storage: &str,
) -> Result<&'d S, Error>
where
    S: TensorStorage<D>,
{
    destination_precheck(destination, input, storage)?;
    require_destination_space(destination.logical_space().space(), expected)?;
    destination_postcheck(destination, expected.required_len()?)
}

/// The destination checks before its space: owned dense storage, then no
/// alias of the input payload.
pub(super) fn destination_precheck<R, D, S>(
    destination: &TensorMap<R, D, S>,
    input: &Arc<TypedData<D, S>>,
    storage: &str,
) -> Result<(), Error> {
    let body = match &destination.repr {
        TypedTensorRepr::Owned(body) if matches!(body.data.as_ref(), TypedData::Dense(_)) => body,
        _ => {
            return Err(Error::InvalidArgument(format!(
                "destination must use ordinary dense {storage} storage"
            )))
        }
    };
    if Arc::ptr_eq(&body.data, input) {
        return Err(Error::InvalidArgument(
            "destination storage must not alias an input".to_string(),
        ));
    }
    Ok(())
}

/// The destination checks after its space: the exact `required` length, then
/// unique ownership. The caller ran [`destination_precheck`].
pub(super) fn destination_postcheck<R, D, S>(
    destination: &TensorMap<R, D, S>,
    required: usize,
) -> Result<&S, Error>
where
    S: TensorStorage<D>,
{
    let TypedTensorRepr::Owned(body) = &destination.repr else {
        return Err(internal_layout_error("ordinary destination checked above"));
    };
    let TypedData::Dense(data) = body.data.as_ref() else {
        return Err(internal_layout_error("dense destination checked above"));
    };
    let actual = data.len();
    if actual != required {
        return Err(Error::InvalidArgument(format!(
            "destination storage length {actual} does not match required length {required}"
        )));
    }
    if Arc::strong_count(body) != 1 || Arc::strong_count(&body.data) != 1 {
        return Err(Error::DestinationShared);
    }
    Ok(data)
}

/// [`destination_postcheck`] against the destination's own space, then the
/// stored payload to write.
pub(super) fn destination_storage_mut<R, D, S>(
    destination: &mut TensorMap<R, D, S>,
) -> Result<&mut S, Error>
where
    S: TensorStorage<D>,
{
    destination_postcheck(
        destination,
        destination.logical_space().space().required_len()?,
    )?;
    let TypedTensorRepr::Owned(body) = &mut destination.repr else {
        return Err(internal_layout_error("ordinary destination checked above"));
    };
    match Arc::get_mut(body).and_then(|body| Arc::get_mut(&mut body.data)) {
        Some(TypedData::Dense(data)) => Ok(data),
        _ => Err(internal_layout_error(
            "unique dense destination checked above",
        )),
    }
}

/// [`destination_storage_mut`] of a host destination, as a slice.
pub(super) fn destination_slice<R, D>(
    destination: &mut TensorMap<R, D>,
) -> Result<&mut [D], Error> {
    destination_storage_mut(destination).map(Vec::as_mut_slice)
}

/// Runtime first, then space: crossing runtimes is a trust-boundary violation
/// rather than an algebra error. `DynamicFusionMapSpace: PartialEq` covers the
/// hom space, the codomain/domain split and the block structure, which is
/// exactly what makes an element-wise combination meaningful.
pub(super) fn require_same_runtime_and_space<R, D, S>(
    a: &TensorMap<R, D, S>,
    b: &TensorMap<R, D, S>,
) -> Result<(), Error> {
    if !a.runtime.same_runtime(&b.runtime) {
        return Err(Error::RuntimeMismatch);
    }
    if a.logical_space().space() != b.logical_space().space() {
        return Err(Error::from(symtn_tensors::OperationError::SpaceMismatch {
            message: "tensors live on different spaces or block layouts",
        }));
    }
    Ok(())
}

/// The one body of [`TensorMap::axpby_into`], shared with the empty-pair
/// `trace_pairs_into`: the TensorKit order (runtime, space, then the
/// destination checks) before the one storage kernel.
pub(super) fn axpby_into_checked<R, D, S>(
    engine: &mut dyn Engine,
    x: &TensorMap<R, D, S>,
    destination: &mut TensorMap<R, D, S>,
    alpha: D,
    beta: D,
) -> Result<(), Error>
where
    S: ElementwiseExec<R, D>,
{
    require_same_runtime_and_space(x, destination)?;
    unique_dense_destination(
        destination,
        &x.storage_body().data,
        x.logical_space().space(),
        S::NOUN,
    )?;
    S::axpby_into(engine, EXEC, x, destination, alpha, beta)
}
