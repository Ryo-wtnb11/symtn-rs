use super::*;
use symtn_dense::Engine;

/// The storage a checked factorization reads: dense coupled-sector storage,
/// a compact diagonal on its bond space, or the lazy adjoint of dense
/// storage. A diagonal is factorized directly (MAK `DiagonalAlgorithm`), so
/// it never needs a dense executor.
pub enum FactorSource<'a, R, D> {
    Dense(BoundDynamicTensorRef<'a, R, D>),
    Diagonal {
        space: &'a BoundDynamicFusionMapSpace<R>,
        spectrum: &'a [SectorSpectrum<D>],
    },
    /// The adjoint of the dense `parent`, read in place; `adjoint_space` is
    /// the adjoint's own layout. Only the operations with a lazy-adjoint
    /// route (compact and full SVD and LQ, left and right polar) accept it;
    /// the others refuse it before any lease or value scan.
    AdjointOf {
        parent: BoundDynamicTensorRef<'a, R, D>,
        adjoint_space: &'a BoundDynamicFusionMapSpace<R>,
    },
}

/// A factor in the storage its route produced: a compact diagonal on
/// `space`, dense storage, or the adjoint of dense storage.
pub enum FactorOutput<R, D> {
    Dense(BoundDynFactor<R, D>),
    Diagonal {
        space: BoundDynamicFusionMapSpace<R>,
        values: Vec<SectorSpectrum<D>>,
    },
    /// The adjoint of this dense factor: the polar factor of a lazy adjoint
    /// that is the adjoint of its parent's opposite polar factor.
    DenseAdjoint(BoundDynFactor<R, D>),
}

impl<R, D> FactorOutput<R, D> {
    /// The dense factor, which every factor of a dense source is; `None` for
    /// a compact diagonal and for the adjoint of a dense factor.
    pub fn into_dense(self) -> Option<BoundDynFactor<R, D>> {
        match self {
            Self::Dense(factor) => Some(factor),
            Self::Diagonal { .. } | Self::DenseAdjoint(_) => None,
        }
    }
}

/// The dense executor of a factorization's dense stage, obtained only when
/// that stage runs: a compact diagonal never needs one. A plain `&mut E` is
/// one; the facade's runtime lease is another.
pub trait ExecutorLease {
    type Executor: DenseExecutor + ?Sized;

    fn run<T>(self, stage: impl FnOnce(&mut Self::Executor) -> T) -> T;
}

impl<E: DenseExecutor + ?Sized> ExecutorLease for &mut E {
    type Executor = E;

    fn run<T>(self, stage: impl FnOnce(&mut E) -> T) -> T {
        stage(self)
    }
}

/// One factorization of the svd, qr/lq/null, eigh/eig and polar families,
/// chosen by an [`op`](super::op) value; its result type is fixed per
/// operation.
pub trait FactorOp<R, D: FactorScalar>: Run<R, D> {
    type Output;
}

/// The one entry of the factorizations: `op` of `source` in fusion mode
/// `M`. Only a dense stage leases an executor from `lease`.
///
/// The execution hooks behind it take a token only this crate constructs,
/// so a generic bound on the public traits does not reach them:
///
/// ```compile_fail,E0308
/// use symtn_matrixalgebra::seam::{ExecutorLease, FactorMode, FactorOp, FactorSource};
/// use symtn_matrixalgebra::FactorScalar;
///
/// fn bypass<M: FactorMode<R>, O: FactorOp<R, D>, L: ExecutorLease, R, D: FactorScalar>(
///     engine: &mut dyn symtn_dense::Engine,
///     lease: L,
///     op: O,
///     source: FactorSource<'_, R, D>,
/// ) {
///     let _ = O::run::<M, L>(op, (), engine, lease, source);
/// }
/// ```
///
/// ```compile_fail,E0308
/// use symtn_matrixalgebra::seam::{BoundDynamicTensorRef, FactorMode};
/// use symtn_matrixalgebra::FactorScalar;
///
/// fn bypass<M: FactorMode<R>, E: symtn_dense::DenseExecutor, R, D: FactorScalar>(
///     engine: &mut dyn symtn_dense::Engine,
///     dense: &mut E,
///     input: &BoundDynamicTensorRef<'_, R, D>,
/// ) {
///     let _ = M::svd_compact_dense((), engine, dense, input);
/// }
/// ```
pub fn factor<M, O, L, R, D>(
    engine: &mut dyn Engine,
    lease: L,
    op: O,
    source: FactorSource<'_, R, D>,
) -> Result<O::Output, M::Error>
where
    M: FactorMode<R>,
    O: FactorOp<R, D>,
    L: ExecutorLease,
    D: FactorScalar,
{
    op.run::<M, L>(TOKEN, engine, lease, source)
}

/// The refusal of a [`FactorSource::AdjointOf`] by an operation without a
/// lazy-adjoint route: its caller materializes or redirects the adjoint.
fn no_adjoint_route() -> OperationError {
    OperationError::UnsupportedTensorContractScope {
        message:
            "this factorization has no lazy-adjoint route; materialize or redirect the adjoint",
    }
}

/// The `adjoint` argument of [`factor_from_source`] for an operation without
/// a lazy-adjoint route.
#[allow(clippy::type_complexity)]
pub(super) fn no_adjoint<E: ?Sized, R, D, T, X>() -> Option<
    fn(
        &mut dyn Engine,
        &mut E,
        &BoundDynamicTensorRef<'_, R, D>,
        &BoundDynamicFusionMapSpace<R>,
    ) -> Result<T, X>,
> {
    None
}

/// Runs the diagonal, the dense or the lazy-adjoint stage of `source`; only
/// the dense and adjoint stages lease an executor.
///
/// `dense_finite` names the family whose shared finite-input stage
/// ([`require_finite_factor_input`], #1986) runs on dense storage here,
/// before the lease, because that family's dense route has no structural
/// admission to run first; an adjoint's parent is scanned the same way (the
/// adjoint is finite exactly when its parent is). `None` leaves the check to
/// a dense route that admits structure first (eig/eigh: endomorphism and
/// stacking; polar: its block shape), so every route reports structure
/// before values, as the diagonal route does.
///
/// `adjoint` is `None` for an operation without a lazy-adjoint route, which
/// refuses [`FactorSource::AdjointOf`] before any scan or lease.
pub(super) fn factor_from_source<L, E, R, D, T, X, A>(
    engine: &mut dyn Engine,
    lease: L,
    source: FactorSource<'_, R, D>,
    dense_finite: Option<FactorFamily>,
    diagonal: impl FnOnce(&BoundDynamicFusionMapSpace<R>, &[SectorSpectrum<D>]) -> Result<T, X>,
    dense: impl FnOnce(&mut dyn Engine, &mut E, &BoundDynamicTensorRef<'_, R, D>) -> Result<T, X>,
    adjoint: Option<A>,
) -> Result<T, X>
where
    A: FnOnce(
        &mut dyn Engine,
        &mut E,
        &BoundDynamicTensorRef<'_, R, D>,
        &BoundDynamicFusionMapSpace<R>,
    ) -> Result<T, X>,
    L: ExecutorLease<Executor = E>,
    E: DenseExecutor + ?Sized,
    D: FactorScalar,
    X: From<OperationError>,
{
    match source {
        FactorSource::Dense(input) => {
            if let Some(family) = dense_finite {
                require_finite_factor_input(input.data().iter().copied(), family)?;
            }
            lease.run(|executor| dense(engine, executor, &input))
        }
        FactorSource::Diagonal { space, spectrum } => diagonal(space, spectrum),
        FactorSource::AdjointOf {
            parent,
            adjoint_space,
        } => {
            let Some(adjoint) = adjoint else {
                return Err(no_adjoint_route().into());
            };
            if let Some(family) = dense_finite {
                require_finite_factor_input(parent.data().iter().copied(), family)?;
            }
            lease.run(|executor| adjoint(engine, executor, &parent, adjoint_space))
        }
    }
}
