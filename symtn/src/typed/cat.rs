use super::*;

impl<R, D> TensorMap<R, D>
where
    D: TensorScalar,
{
    fn cat_operand(&self) -> Result<(CatOperandLayout<'_>, CatOperandData<'_, D>), Error> {
        match &self.repr {
            TypedTensorRepr::Owned(body) => {
                let layout = CatOperandLayout::owned(
                    body.space.space().structure(),
                    body.space.space().nout(),
                    body.space.space().nin(),
                )?;
                let data = match body.data.as_ref() {
                    TypedData::Dense(data) => CatOperandData::Dense(data),
                    TypedData::Diagonal(spectrum) => CatOperandData::Diagonal {
                        structure: body.space.space().structure(),
                        spectrum,
                    },
                };
                Ok((layout, data))
            }
            TypedTensorRepr::Adjoint(view) => Ok((
                CatOperandLayout::adjoint(
                    view.parent.space.space().structure(),
                    view.parent.space.space().nout(),
                    view.parent.space.space().nin(),
                )?,
                CatOperandData::Dense(view.parent_data()),
            )),
        }
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorRootDispatch<R>,
    D: TensorScalar,
{
    /// TensorKit `catdomain(t1, t2)` (`side = Side::Domain`) and
    /// `catcodomain(t1, t2)` (`side = Side::Codomain`): concatenate two tensor
    /// maps along their sole leg on `side`. The product spaces on the other
    /// side must match exactly; the two concatenated legs must share duality
    /// and are combined by direct sum `V = V1 ⊕ V2`; reduced data is copied
    /// into adjacent slabs per coupled sector (column slabs for the domain,
    /// row slabs for the codomain), `self` first.
    ///
    /// Rust uses one method (`t1.cat(&t2, side)`) because binary tensor
    /// operations in this API are methods and the two TensorKit functions are
    /// one operation that differs only in the side; the operand order matches
    /// TensorKit's free functions.
    ///
    /// Both operands share one `D`, so mixed-dtype widening is statically
    /// unrepresentable — widen with [`Self::convert`] first.
    /// A lazy adjoint is read from parent storage through the oriented copy
    /// plan without publishing a receiver-sized materialization. A compact
    /// diagonal operand is read directly from its spectrum into the output;
    /// nothing is retained.
    ///
    /// # Complexity
    ///
    /// One output admission and allocation plus
    /// `O(len(self) + len(other))` work over the compiled per-sector slab plan.
    /// The compact path also initializes structural zeros in the output.
    ///
    /// # Errors
    ///
    /// [`Error::RuleMismatch`] on differing admitted rule identities and
    /// [`Error::RuntimeMismatch`] on differing runtimes, in that order; then
    /// [`Error::InvalidArgument`] for more than one leg on `side`;
    /// `SpaceMismatch` for mismatched product spaces on the other side or
    /// concatenated legs of opposite duality. Checked-Generic output-admission failures retain their typed
    /// provider error.
    pub fn cat<'a>(
        &self,
        engine: &mut dyn Engine,
        other: impl Into<TensorRef<'a, R, D>>,
        side: Side,
    ) -> Result<Self, FacadeError<R>> {
        let other = other.into().operand()?;
        let other = &*other;
        let lhs_space = self.logical_space().space();
        let rhs_space = other.logical_space().space();
        if lhs_space.admission().rule_identity() != rhs_space.admission().rule_identity() {
            return Err(FacadeError::<R>::from(Error::RuleMismatch));
        }
        if !self.runtime.same_runtime(&other.runtime) {
            return Err(FacadeError::<R>::from(Error::RuntimeMismatch));
        }
        let lhs = lhs_space.homspace();
        let rhs = rhs_space.homspace();
        let (axis, homspace) = cat_homspace(
            lhs.codomain(),
            lhs.domain(),
            rhs.codomain(),
            rhs.domain(),
            side,
        )
        .map_err(FacadeError::<R>::from)?;
        let space = <R::Mode as TypedTensorRootDispatch<R>>::build_root(
            Arc::clone(self.logical_space().provider_arc()),
            homspace,
        )?;
        let (lhs_layout, lhs_data) = self.cat_operand().map_err(FacadeError::<R>::from)?;
        let (rhs_layout, rhs_data) = other.cat_operand().map_err(FacadeError::<R>::from)?;
        let data = compile_cat_plan(
            space.space().structure(),
            space.space().nout(),
            [lhs_layout, rhs_layout],
            axis,
            side,
        )
        .and_then(|plan| plan.execute(engine, [lhs_data, rhs_data]))
        .map_err(FacadeError::<R>::from)?;
        Ok(Self {
            runtime: self.runtime.clone(),
            repr: owned_repr(TypedTensorBody::dense(space, data)),
        })
    }
}
