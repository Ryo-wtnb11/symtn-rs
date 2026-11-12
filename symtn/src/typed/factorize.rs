use super::*;

mod api;
mod exp_solve;
mod inverse;
mod mode;

pub use mode::FusionMode;
use mode::{AdjointRule, FactorOp};

/// The runtime as the seam's executor lease, the one lease rule of every
/// factorization in both fusion modes (#1996): a dense executor is leased
/// only for the dense stage, after preflight, materialization and payload
/// binding, so a compact diagonal or an earlier error never builds an
/// executor.
struct RuntimeDense<'a>(&'a Runtime);

impl symtn_matrixalgebra::seam::ExecutorLease for RuntimeDense<'_> {
    type Executor = dyn symtn_dense::DenseExecutor + Send;

    // Why not `DenseLease::dense`: its signature ties the trait object's
    // lifetime to the borrow, while both arms hold `'static` executors.
    fn run<T>(self, stage: impl FnOnce(&mut Self::Executor) -> T) -> T {
        let mut lease = self.0.lease_dense();
        let executor: &mut Self::Executor = &mut **lease
            .executor
            .as_mut()
            .expect("dense lease always owns an executor");
        stage(executor)
    }
}

/// Whether `op` redirects a lazy adjoint: a body whose only lazy-adjoint arm
/// is the redirect asserts it at compile time, so the rule table and the
/// body cannot disagree.
const fn redirects(op: FactorOp) -> bool {
    matches!(op.adjoint_rule(), AdjointRule::Redirect)
}

/// The seam source of an owned checked body: its compact diagonal, or its
/// dense payload bound to its space.
fn owned_factor_source<'a, R, D>(
    body: &'a TypedTensorBody<R, D>,
) -> Result<symtn_matrixalgebra::seam::FactorSource<'a, R, D>, Error> {
    Ok(match body.data.as_ref() {
        TypedData::Diagonal(spectrum) => symtn_matrixalgebra::seam::FactorSource::Diagonal {
            space: &body.space,
            spectrum,
        },
        TypedData::Dense(data) => symtn_matrixalgebra::seam::FactorSource::Dense(
            BoundDynamicTensorRef::try_new(&body.space, data)?,
        ),
    })
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R>,
    D: TensorScalar,
{
    /// The bound space and dense payload of an owned operand on a dense
    /// route; a compact diagonal is densified operation-locally.
    #[allow(clippy::type_complexity)]
    fn dense_operand(
        &self,
    ) -> Result<(&BoundDynamicFusionMapSpace<R>, std::borrow::Cow<'_, [D]>), Error> {
        let body = self.owned_body().ok_or_else(|| {
            internal_layout_error("a dense-route operand must be owned after adjoint dispatch")
        })?;
        Ok((&body.space, body.materialized_dense_data()))
    }

    /// The storage `op` reads: the receiver's own, or for a lazy adjoint
    /// whatever [`FactorOp::adjoint_rule`] selects. `local` holds an
    /// operation-local materialization.
    fn factor_input<'a>(
        &'a self,
        engine: &mut dyn Engine,
        op: FactorOp,
        local: &'a mut Option<Self>,
    ) -> Result<symtn_matrixalgebra::seam::FactorSource<'a, R, D>, FacadeError<R>> {
        let view = match &self.repr {
            TypedTensorRepr::Owned(body) => return Ok(owned_factor_source(body)?),
            TypedTensorRepr::Adjoint(view) => view,
        };
        match op.adjoint_rule() {
            AdjointRule::Parent => Ok(symtn_matrixalgebra::seam::FactorSource::Dense(
                BoundDynamicTensorRef::try_new(&view.parent.space, view.parent_data())
                    .map_err(Error::from)?,
            )),
            AdjointRule::Materialize => {
                let local = local.insert(self.materialized_tensor_uncached(engine)?);
                Ok(owned_factor_source(
                    local
                        .owned_body()
                        .expect("materialize returns an owned body"),
                )?)
            }
            AdjointRule::Redirect | AdjointRule::AdjointSeam => Err(internal_layout_error(
                "a redirected factorization must not read its own lazy input",
            )
            .into()),
        }
    }

    /// `f(A)^H` for the lazy adjoint `self = A^H`, detached: the redirect
    /// of an operation with `f(A^H) = f(A)^H`. The parent is owned, so `f`
    /// never redirects again.
    fn adjoint_of_parent(
        &self,
        engine: &mut dyn Engine,
        f: impl FnOnce(&Self, &mut dyn Engine) -> Result<Self, FacadeError<R>>,
    ) -> Result<Self, FacadeError<R>> {
        Ok(f(&self.adjoint(&mut *engine)?, engine)?
            .adjoint(&mut *engine)?
            .materialized_tensor_uncached(engine)?)
    }

    /// A seam factor in the storage its route produced; the adjoint of a
    /// dense factor is materialized detached.
    fn factor_output(
        &self,
        engine: &mut dyn Engine,
        output: symtn_matrixalgebra::seam::FactorOutput<R, D>,
    ) -> Result<Self, FacadeError<R>> {
        Ok(match output {
            symtn_matrixalgebra::seam::FactorOutput::Dense(factor) => {
                wrap_factor_on(&self.runtime, factor)
            }
            symtn_matrixalgebra::seam::FactorOutput::Diagonal { space, values } => {
                self.with_spectrum_on(space, values)
            }
            symtn_matrixalgebra::seam::FactorOutput::DenseAdjoint(factor) => {
                wrap_factor_on(&self.runtime, factor)
                    .adjoint(&mut *engine)?
                    .materialized_tensor_uncached(engine)?
            }
        })
    }

    /// The one body of the values-only factorizations: `seam_op` computes
    /// the raw spectra, and the coupled sectors are decoded into public
    /// labels and sorted by label.
    fn factor_values<V, O>(
        &self,
        engine: &mut dyn Engine,
        op: FactorOp,
        seam_op: O,
    ) -> Result<Vec<SectorSpectrum<R::Sector, V>>, FacadeError<R>>
    where
        D: FactorScalar,
        O: symtn_matrixalgebra::seam::FactorOp<
            R,
            D,
            Output = Vec<symtn_matrixalgebra::SectorSpectrum<V>>,
        >,
    {
        let mut local = None;
        let source = self.factor_input(engine, op, &mut local)?;
        let raw = symtn_matrixalgebra::seam::factor::<R::Mode, _, _, _, _>(
            engine,
            RuntimeDense(&self.runtime),
            seam_op,
            source,
        )
        .map_err(R::Mode::map_factor_error)?;
        let provider = self.logical_space().provider();
        let mut decoded = raw
            .into_iter()
            .map(|entry| {
                Ok(SectorSpectrum {
                    sector: R::Mode::decode_label(provider, entry.sector)?,
                    values: entry.values,
                })
            })
            .collect::<Result<Vec<_>, FacadeError<R>>>()?;
        // Public label order, not the engine's opaque sector-id order.
        decoded.sort_by(|left, right| left.sector.cmp(&right.sector));
        Ok(decoded)
    }

    /// The one body of QR: each factor keeps the storage its route produced.
    fn factor_qr<O>(
        &self,
        engine: &mut dyn Engine,
        op: FactorOp,
        seam_op: O,
    ) -> Result<Qr<Self>, FacadeError<R>>
    where
        D: FactorScalar,
        O: symtn_matrixalgebra::seam::FactorOp<
            R,
            D,
            Output = Qr<symtn_matrixalgebra::seam::FactorOutput<R, D>>,
        >,
    {
        let mut local = None;
        let source = self.factor_input(engine, op, &mut local)?;
        let Qr { q, r } = symtn_matrixalgebra::seam::factor::<R::Mode, _, _, _, _>(
            engine,
            RuntimeDense(&self.runtime),
            seam_op,
            source,
        )
        .map_err(R::Mode::map_factor_error)?;
        Ok(Qr {
            q: self.factor_output(engine, q)?,
            r: self.factor_output(engine, r)?,
        })
    }

    /// The one body of LQ: each factor keeps the storage its route produced.
    /// Under [`AdjointRule::AdjointSeam`] a lazy adjoint's dense parent is
    /// read in place.
    fn factor_lq<O>(
        &self,
        engine: &mut dyn Engine,
        op: FactorOp,
        seam_op: O,
    ) -> Result<Lq<Self>, FacadeError<R>>
    where
        D: FactorScalar,
        O: symtn_matrixalgebra::seam::FactorOp<
            R,
            D,
            Output = Lq<symtn_matrixalgebra::seam::FactorOutput<R, D>>,
        >,
    {
        let Lq { l, q } = self.seam_factors(engine, op, seam_op)?;
        Ok(Lq {
            l: self.factor_output(engine, l)?,
            q: self.factor_output(engine, q)?,
        })
    }

    /// The one body of the null spaces. Under [`AdjointRule::Redirect`] a
    /// lazy adjoint's null space is the adjoint of `partner` (the opposite
    /// null space) of its parent, detached.
    fn factor_null<O>(
        &self,
        engine: &mut dyn Engine,
        op: FactorOp,
        partner: impl FnOnce(&Self, &mut dyn Engine) -> Result<Self, FacadeError<R>>,
        seam_op: O,
    ) -> Result<Self, FacadeError<R>>
    where
        R::Mode: TypedAdjointSpace<R>,
        D: FactorScalar,
        O: symtn_matrixalgebra::seam::FactorOp<
            R,
            D,
            Output = symtn_matrixalgebra::BoundDynFactor<R, D>,
        >,
    {
        if matches!(&self.repr, TypedTensorRepr::Adjoint(_))
            && op.adjoint_rule() == AdjointRule::Redirect
        {
            return self.adjoint_of_parent(engine, partner);
        }
        let mut local = None;
        let source = self.factor_input(engine, op, &mut local)?;
        let factor = symtn_matrixalgebra::seam::factor::<R::Mode, _, _, _, _>(
            engine,
            RuntimeDense(&self.runtime),
            seam_op,
            source,
        )
        .map_err(R::Mode::map_factor_error)?;
        Ok(wrap_factor_on(&self.runtime, factor))
    }

    /// The one body of the left polar decomposition. A lazy adjoint `A^H` is
    /// read in place as the right polar of its parent `A = p wh`: `A^H =
    /// wh^H p`, with `wh^H` detached. A compact diagonal factors directly on
    /// its bond.
    fn factor_left_polar(&self, engine: &mut dyn Engine) -> Result<LeftPolar<Self>, FacadeError<R>>
    where
        D: FactorizationScalar,
    {
        let LeftPolar { w, p } = self.seam_factors(
            engine,
            FactorOp::LeftPolar,
            symtn_matrixalgebra::seam::op::LeftPolar,
        )?;
        Ok(LeftPolar {
            w: self.factor_output(engine, w)?,
            p: self.factor_output(engine, p)?,
        })
    }

    /// The one body of the right polar decomposition; see
    /// [`Self::factor_left_polar`].
    fn factor_right_polar(
        &self,
        engine: &mut dyn Engine,
    ) -> Result<RightPolar<Self>, FacadeError<R>>
    where
        D: FactorizationScalar,
    {
        let RightPolar { p, wh } = self.seam_factors(
            engine,
            FactorOp::RightPolar,
            symtn_matrixalgebra::seam::op::RightPolar,
        )?;
        Ok(RightPolar {
            p: self.factor_output(engine, p)?,
            wh: self.factor_output(engine, wh)?,
        })
    }

    /// The one body of the Hermitian eigendecomposition: `d` is a compact
    /// diagonal on the spectrum bond `fuse(V)` (`seam::spectrum_bond`).
    fn factor_eigh_full(
        &self,
        engine: &mut dyn Engine,
        hermitian_tol: HermitianTol,
    ) -> Result<Eigh<Self>, FacadeError<R>>
    where
        D: FactorizationScalar,
    {
        let mut local = None;
        let source = self.factor_input(engine, FactorOp::EighFull, &mut local)?;
        let out = symtn_matrixalgebra::seam::factor::<R::Mode, _, _, _, _>(
            engine,
            RuntimeDense(&self.runtime),
            symtn_matrixalgebra::seam::op::EighFull { hermitian_tol },
            source,
        )
        .map_err(R::Mode::map_factor_error)?;
        let (v, mut eigenvalues) = out.into_parts();
        let space = symtn_matrixalgebra::seam::spectrum_bond::<R::Mode, _, _>(
            self.logical_space(),
            &eigenvalues,
        )
        .map_err(R::Mode::map_factor_error)?;
        Ok(Eigh {
            d: diagonal_factor_on_bound(&self.runtime, space, &mut eigenvalues, D::from_real),
            v: wrap_factor_on(&self.runtime, v),
        })
    }

    /// The one body of the general eigendecomposition; complex factors at the
    /// payload's precision, `d` on the spectrum bond as for
    /// [`Self::factor_eigh_full`].
    #[allow(clippy::type_complexity)]
    fn factor_eig_full(
        &self,
        engine: &mut dyn Engine,
    ) -> Result<Eig<TensorMap<R, <D as FactorScalar>::Eig>>, FacadeError<R>>
    where
        D: AdvancedLinalgScalar,
        <D as FactorScalar>::Eig: TensorScalar,
    {
        let mut local = None;
        let source = self.factor_input(engine, FactorOp::EigFull, &mut local)?;
        let out = symtn_matrixalgebra::seam::factor::<R::Mode, _, _, _, _>(
            engine,
            RuntimeDense(&self.runtime),
            symtn_matrixalgebra::seam::op::EigFull,
            source,
        )
        .map_err(R::Mode::map_factor_error)?;
        let (v, mut eigenvalues) = out.into_parts();
        let space = symtn_matrixalgebra::seam::spectrum_bond::<R::Mode, _, _>(
            self.logical_space(),
            &eigenvalues,
        )
        .map_err(R::Mode::map_factor_error)?;
        Ok(Eig {
            d: diagonal_factor_on_bound(
                &self.runtime,
                space,
                &mut eigenvalues,
                <<D as FactorScalar>::Eig as FactorScalar>::from_complex64,
            ),
            v: wrap_factor_on(&self.runtime, v),
        })
    }

    /// The factors `seam_op` computes. Under [`AdjointRule::AdjointSeam`] a
    /// lazy adjoint's dense parent is read in place, with the adjoint's own
    /// layout.
    fn seam_factors<O>(
        &self,
        engine: &mut dyn Engine,
        op: FactorOp,
        seam_op: O,
    ) -> Result<O::Output, FacadeError<R>>
    where
        D: FactorScalar,
        O: symtn_matrixalgebra::seam::FactorOp<R, D>,
    {
        let mut local = None;
        let source = match &self.repr {
            TypedTensorRepr::Adjoint(view) if op.adjoint_rule() == AdjointRule::AdjointSeam => {
                symtn_matrixalgebra::seam::FactorSource::AdjointOf {
                    parent: BoundDynamicTensorRef::try_new(&view.parent.space, view.parent_data())
                        .map_err(Error::from)?,
                    adjoint_space: &view.logical_space,
                }
            }
            _ => self.factor_input(engine, op, &mut local)?,
        };
        symtn_matrixalgebra::seam::factor::<R::Mode, _, _, _, _>(
            engine,
            RuntimeDense(&self.runtime),
            seam_op,
            source,
        )
        .map_err(R::Mode::map_factor_error)
    }

    /// The one body of compact SVD: `s` is a compact diagonal on the fresh
    /// spectrum bond `fuse(V)` (`seam::spectrum_bond`).
    fn factor_svd_compact(&self, engine: &mut dyn Engine) -> Result<Svd<Self>, FacadeError<R>>
    where
        D: FactorizationScalar,
    {
        let (u, vh, mut spectrum) = self.seam_factors(
            engine,
            FactorOp::SvdCompact,
            symtn_matrixalgebra::seam::op::SvdCompact,
        )?;
        let space = symtn_matrixalgebra::seam::spectrum_bond::<R::Mode, _, _>(
            self.logical_space(),
            &spectrum,
        )
        .map_err(R::Mode::map_factor_error)?;
        Ok(Svd {
            u: wrap_factor_on(&self.runtime, u),
            s: diagonal_factor_on_bound(&self.runtime, space, &mut spectrum, D::from_real),
            vh: wrap_factor_on(&self.runtime, vh),
        })
    }

    /// The one body of full SVD. A compact diagonal takes the compact SVD,
    /// with which its full SVD coincides (TensorKit
    /// `svd_compact!(::DiagonalAlgorithm)` is `svd_full!`). Otherwise `s` is
    /// compact on the spectrum bond when every sector is square, and
    /// dense (possibly rectangular) when not.
    fn factor_svd_full(&self, engine: &mut dyn Engine) -> Result<Svd<Self>, FacadeError<R>>
    where
        D: FactorizationScalar,
    {
        if self.spectrum().is_some() {
            return self.factor_svd_compact(engine);
        }
        let factors = self.seam_factors(
            engine,
            FactorOp::SvdFull,
            symtn_matrixalgebra::seam::op::SvdFull,
        )?;
        let (u, vh, mut spectrum, row_dimensions, col_dimensions) = factors.into_parts();
        if full_svd_compact_bond(&u, &vh, &spectrum) {
            let space = symtn_matrixalgebra::seam::spectrum_bond::<R::Mode, _, _>(
                self.logical_space(),
                &spectrum,
            )
            .map_err(R::Mode::map_factor_error)?;
            return Ok(Svd {
                u: wrap_factor_on(&self.runtime, u),
                s: diagonal_factor_on_bound(&self.runtime, space, &mut spectrum, D::from_real),
                vh: wrap_factor_on(&self.runtime, vh),
            });
        }
        let s = symtn_matrixalgebra::seam::rectangular_diagonal_bond_tensor(
            &<R::Mode as symtn_matrixalgebra::seam::FactorMode<R>>::authority(self.logical_space()),
            &spectrum,
            &row_dimensions,
            &col_dimensions,
        )
        .map_err(R::Mode::map_factor_error)?;
        Ok(Svd {
            u: wrap_factor_on(&self.runtime, u),
            s: wrap_factor_on(&self.runtime, s),
            vh: wrap_factor_on(&self.runtime, vh),
        })
    }
}

/// The storage kernels of the factorizations with a device route
/// (`qr_compact`, `svd_compact`, `eigh_full`): one typed body per operation
/// (#1756), with each storage supplying its admission *sequence* and its
/// execution.
///
/// The admission rules and their refusals are not the storage's: the Host
/// kernel admits inside the matrix-algebra stage, and the device kernel asks
/// the same matrix-algebra seam (endomorphism and stacking admission, the
/// finite-input and Hermitian refusals) at the points its own placement
/// preflight and device scans dictate. Only measuring a rule on the device
/// (finiteness, hermiticity) is storage code.
///
/// Each kernel factors the matrix view the body's leg roles selected, in
/// whatever representation it holds; a representation the storage cannot
/// factor is its `UnsupportedOnDevice`, raised before any output exists.
#[doc(hidden)]
pub trait FactorExec<R, D>: TransformExec<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R>,
{
    /// Compact QR of `t`'s current split.
    fn qr_compact(
        engine: &mut dyn Engine,
        _: ExecToken,
        t: &TensorMap<R, D, Self>,
    ) -> Result<Qr<TensorMap<R, D, Self>>, FacadeError<R>>;

    /// Compact SVD of `t`'s current split.
    fn svd_compact(
        engine: &mut dyn Engine,
        _: ExecToken,
        t: &TensorMap<R, D, Self>,
    ) -> Result<Svd<TensorMap<R, D, Self>>, FacadeError<R>>;

    /// Hermitian eigendecomposition of `t`'s current split.
    fn eigh_full(
        engine: &mut dyn Engine,
        _: ExecToken,
        t: &TensorMap<R, D, Self>,
        hermitian_tol: HermitianTol,
    ) -> Result<Eigh<TensorMap<R, D, Self>>, FacadeError<R>>;

    /// [`EighFullPlan::execute`]'s solve of every member of `source` into
    /// `buffers` (allocated when `None`) and `workspace`'s sorted spectra,
    /// after the plan checked the workspace and the source signature.
    fn eigh_full_stack(
        engine: &mut dyn Engine,
        _: ExecToken,
        plan: &EighFullPlan<R, D, Self>,
        source: &StackedTensorMap<R, D, Self>,
        buffers: &mut Option<super::batched::StackPair<R, D, Self>>,
        workspace: &mut EighFullWorkspace<R, D, Self>,
    ) -> Result<(), BatchError>
    where
        R: MultiplicityFreeRigidSymbols + CheckedFusionAlgebra + SectorCodec;
}

impl<R, D> FactorExec<R, D> for Vec<D>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R> + TypedTensorTransformDispatch<R, D>,
    D: FactorizationScalar,
{
    fn qr_compact(
        engine: &mut dyn Engine,
        _: ExecToken,
        t: &TensorMap<R, D>,
    ) -> Result<Qr<TensorMap<R, D>>, FacadeError<R>> {
        t.factor_qr(
            engine,
            FactorOp::QrCompact,
            symtn_matrixalgebra::seam::op::QrCompact,
        )
    }

    fn svd_compact(
        engine: &mut dyn Engine,
        _: ExecToken,
        t: &TensorMap<R, D>,
    ) -> Result<Svd<TensorMap<R, D>>, FacadeError<R>> {
        t.factor_svd_compact(engine)
    }

    fn eigh_full(
        engine: &mut dyn Engine,
        _: ExecToken,
        t: &TensorMap<R, D>,
        hermitian_tol: HermitianTol,
    ) -> Result<Eigh<TensorMap<R, D>>, FacadeError<R>> {
        t.factor_eigh_full(engine, hermitian_tol)
    }

    fn eigh_full_stack(
        engine: &mut dyn Engine,
        _: ExecToken,
        plan: &EighFullPlan<R, D>,
        source: &StackedTensorMap<R, D>,
        buffers: &mut Option<super::batched::StackPair<R, D, Self>>,
        workspace: &mut EighFullWorkspace<R, D>,
    ) -> Result<(), BatchError>
    where
        R: MultiplicityFreeRigidSymbols + CheckedFusionAlgebra + SectorCodec,
    {
        plan.run_host(engine, source, buffers, &mut workspace.sorted)
    }
}
