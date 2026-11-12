use super::*;
use symtn_matrixalgebra::seam::FactorSpaceAuthority;

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R> + TypedAdjointSpace<R>,
    D: AdvancedLinalgScalar,
{
    /// The one body of the inverse. A lazy adjoint is redirected,
    /// `(A^H)^-1 = (A^-1)^H`; a compact diagonal inverts its entries
    /// (TensorKit `inv(::DiagonalTensorMap)`).
    pub(super) fn factor_inv(&self, engine: &mut dyn Engine) -> Result<Self, FacadeError<R>> {
        self.require_isomorphic("inv requires isomorphic codomain and domain")?;
        self.factor_inv_admitted(engine)
    }

    /// The representation stage of [`Self::factor_inv`]. The redirect
    /// re-enters here rather than at the preflight: `A^H` and `A` are
    /// isomorphic together, so the parent needs no second check.
    fn factor_inv_admitted(&self, engine: &mut dyn Engine) -> Result<Self, FacadeError<R>> {
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_)) {
            const { assert!(redirects(FactorOp::Inv)) };
            return self.adjoint_of_parent(engine, |t, engine| t.factor_inv_admitted(engine));
        }
        if let Some(spectrum) = self.spectrum() {
            self.admit_compact(spectrum)?;
            return Ok(self.with_spectrum(inv_spectrum(spectrum)?));
        }
        let output = self.swapped_output_space()?;
        let (space, data) = self.dense_operand()?;
        let input = BoundDynamicTensorRef::try_new(space, &data).map_err(Error::from)?;
        let factor = symtn_matrixalgebra::seam::matrix_function(
            engine,
            RuntimeDense(&self.runtime),
            symtn_matrixalgebra::seam::op::Inv,
            symtn_matrixalgebra::seam::MatrixFunctionSource::Dense {
                input,
                output_space: output,
            },
        )
        .map_err(Error::from)?;
        Ok(wrap_factor_on(&self.runtime, factor))
    }

    /// The one body of the pseudo-inverse: a lazy adjoint `A^H` is read
    /// through its parent's SVD, `(A^H)^+ = U S^+ Vh`; a compact diagonal
    /// inverts its retained entries.
    pub(super) fn factor_pinv(
        &self,
        engine: &mut dyn Engine,
        rcond: f64,
    ) -> Result<Self, FacadeError<R>> {
        // Ahead of every arm, so all of them answer alike: the dense arms of
        // the seam repeat this check for their own expert callers, but its
        // compact arm relies on this one, as on the compact admission below.
        // TensorKit's `pinv` has no space check.
        if !rcond.is_finite() || rcond < 0.0 {
            return Err(Error::InvalidArgument(
                "pinv rcond must be finite and non-negative".to_string(),
            )
            .into());
        }
        // A lazy adjoint is never compact; the dense stage reads its parent.
        const {
            assert!(matches!(
                FactorOp::Pinv.adjoint_rule(),
                AdjointRule::AdjointSeam
            ))
        };
        let op = symtn_matrixalgebra::seam::op::Pinv {
            rcond,
            reciprocal: |value: D| value.recip_value(),
        };
        let operand;
        let source = if let Some(spectrum) = self.spectrum() {
            self.admit_compact(spectrum)?;
            symtn_matrixalgebra::seam::MatrixFunctionSource::Diagonal {
                space: self.logical_space(),
                spectrum,
            }
        } else {
            let output_space = self.swapped_output_space()?;
            match &self.repr {
                TypedTensorRepr::Adjoint(view) => {
                    symtn_matrixalgebra::seam::MatrixFunctionSource::AdjointOf {
                        parent: BoundDynamicTensorRef::try_new(
                            &view.parent.space,
                            view.parent_data(),
                        )
                        .map_err(Error::from)?,
                        output_space,
                    }
                }
                TypedTensorRepr::Owned(_) => {
                    operand = self.dense_operand()?;
                    symtn_matrixalgebra::seam::MatrixFunctionSource::Dense {
                        input: BoundDynamicTensorRef::try_new(operand.0, &operand.1)
                            .map_err(Error::from)?,
                        output_space,
                    }
                }
            }
        };
        let out = symtn_matrixalgebra::seam::matrix_function(
            engine,
            RuntimeDense(&self.runtime),
            op,
            source,
        )
        .map_err(Error::from)?;
        self.factor_output(engine, out)
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R>,
    D: TensorScalar,
{
    /// TensorKit `cod ≅ dom` (`inv`, `\`), refused with `message`. Equal
    /// sides are isomorphic without a provider query.
    pub(super) fn require_isomorphic(&self, message: &'static str) -> Result<(), FacadeError<R>> {
        let space = self.logical_space();
        let homspace = space.space().homspace();
        if homspace.codomain() == homspace.domain()
            || <R::Mode as symtn_matrixalgebra::seam::FactorMode<R>>::authority(space)
                .isomorphic(space)
                .map_err(R::Mode::map_factor_error)?
        {
            return Ok(());
        }
        Err(Error::from(symtn_tensors::OperationError::SpaceMismatch { message }).into())
    }

    /// The mode's output space of `homspace` on this tensor's provider: the
    /// space TensorKit's `similar(t, T, homspace)` allocates.
    pub(super) fn factor_output_space(
        &self,
        homspace: FusionTreeHomSpace,
    ) -> Result<BoundDynamicFusionMapSpace<R>, FacadeError<R>> {
        <R::Mode as symtn_matrixalgebra::seam::FactorMode<R>>::authority(self.logical_space())
            .output_space(homspace)
            .map_err(R::Mode::map_root_error)
    }

    /// `domain <- codomain`: the output of the inverse and pseudo-inverse.
    fn swapped_output_space(&self) -> Result<BoundDynamicFusionMapSpace<R>, FacadeError<R>> {
        let homspace = self.logical_space().space().homspace();
        self.factor_output_space(FusionTreeHomSpace::new(
            homspace.domain().clone(),
            homspace.codomain().clone(),
        ))
    }

    /// The one admission of a compact diagonal's elementwise matrix
    /// functions (inv, pinv, exp, solve) in every fusion mode (#1994): the
    /// matrix-algebra bond admission without a value check.
    pub(super) fn admit_compact(
        &self,
        spectrum: &[symtn_matrixalgebra::SectorSpectrum<D>],
    ) -> Result<(), FacadeError<R>> {
        symtn_matrixalgebra::seam::admit_compact_diagonal::<R::Mode, _, _>(
            self.logical_space(),
            spectrum,
        )
        .map_err(R::Mode::map_factor_error)
    }
}
