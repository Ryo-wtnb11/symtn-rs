use super::*;

/// The tree-transform lane of payload `D` and coefficient `C`.
type TreeLane<D, C> = symtn_tensors::TreeTransformExecutionContext<
    D,
    symtn_core::RuleIdentity,
    C,
    symtn_tensors::DenseTreeTransformOperations,
>;

/// Which execution-context lane a mode stages and replays its transforms and
/// contractions on.
pub(in crate::typed) trait HostTreeLane<D, C>
where
    D: symtn_tensors::DenseRecouplingScalar
        + RecouplingCoefficientAction<C>
        + symtn_tensors::ConjugateValue,
    C: symtn_tensors::DenseBlockScalar,
{
    fn contract_lane(
        context: &mut crate::runtime::TensorExecutionContext,
    ) -> Result<&mut crate::runtime::CoefficientCtx<D, symtn_core::RuleIdentity, C>, Error>;

    fn lane(
        context: &mut crate::runtime::TensorExecutionContext,
    ) -> Result<&mut TreeLane<D, C>, Error> {
        Ok(Self::contract_lane(context)?.tree_context_mut())
    }
}

/// The multiplicity-free namespace's coefficient lane.
pub(in crate::typed) struct MultiplicityFreeTreeLane;

impl<D, C> HostTreeLane<D, C> for MultiplicityFreeTreeLane
where
    D: crate::runtime::MultiplicityFreeCoefficientLane<C>
        + symtn_tensors::DenseRecouplingScalar
        + symtn_tensors::ConjugateValue,
    C: symtn_tensors::DenseBlockScalar,
{
    fn contract_lane(
        context: &mut crate::runtime::TensorExecutionContext,
    ) -> Result<&mut crate::runtime::CoefficientCtx<D, symtn_core::RuleIdentity, C>, Error> {
        D::lane(context)
    }
}

/// The checked Generic namespace's lane.
pub(in crate::typed) struct CheckedGenericTreeLane;

impl<D> HostTreeLane<D, f64> for CheckedGenericTreeLane
where
    D: ScalarOps + symtn_tensors::DenseRecouplingScalar + symtn_tensors::ConjugateValue,
{
    fn contract_lane(
        context: &mut crate::runtime::TensorExecutionContext,
    ) -> Result<&mut crate::runtime::CoefficientCtx<D, symtn_core::RuleIdentity, f64>, Error> {
        context.generic_lane::<D>()
    }
}

type OwnedReplay<D, C> = fn(
    &mut dyn Engine,
    &mut crate::runtime::TensorExecutionContext,
    &symtn_tensors::TreeTransformStructure<C>,
    &Arc<BlockStructure>,
    &Arc<BlockStructure>,
    usize,
    &[D],
    D,
) -> Result<Vec<D>, TransformExecError>;

type IntoReplay<D, C> = fn(
    &mut dyn Engine,
    &mut crate::runtime::TensorExecutionContext,
    &symtn_tensors::TreeTransformStructure<C>,
    &Arc<BlockStructure>,
    &Arc<BlockStructure>,
    &mut [D],
    &[D],
    D,
    D,
) -> Result<(), TransformExecError>;

#[allow(clippy::too_many_arguments)]
fn host_owned_replay<L, D, C>(
    engine: &mut dyn Engine,
    context: &mut crate::runtime::TensorExecutionContext,
    structure: &symtn_tensors::TreeTransformStructure<C>,
    destination: &Arc<BlockStructure>,
    source: &Arc<BlockStructure>,
    nout: usize,
    data: &[D],
    alpha: D,
) -> Result<Vec<D>, TransformExecError>
where
    L: HostTreeLane<D, C>,
    D: symtn_tensors::DenseRecouplingScalar
        + RecouplingCoefficientAction<C>
        + symtn_tensors::ConjugateValue,
    C: symtn_tensors::DenseBlockScalar,
{
    L::lane(context)?
        .replay_tree_transform_dyn_owned(engine, structure, destination, source, nout, data, alpha)
        .map_err(TransformExecError::Replay)
}

#[allow(clippy::too_many_arguments)]
fn host_into_replay<L, D, C>(
    engine: &mut dyn Engine,
    context: &mut crate::runtime::TensorExecutionContext,
    structure: &symtn_tensors::TreeTransformStructure<C>,
    destination: &Arc<BlockStructure>,
    source: &Arc<BlockStructure>,
    destination_data: &mut [D],
    data: &[D],
    alpha: D,
    beta: D,
) -> Result<(), TransformExecError>
where
    L: HostTreeLane<D, C>,
    D: symtn_tensors::DenseRecouplingScalar
        + RecouplingCoefficientAction<C>
        + symtn_tensors::ConjugateValue,
    C: symtn_tensors::DenseBlockScalar,
{
    L::lane(context)?
        .replay_tree_transform_dyn_into(
            engine,
            structure,
            destination,
            source,
            destination_data,
            data,
            alpha,
            beta,
        )
        .map_err(TransformExecError::Replay)
}

/// One staged tree-transform replay, handed to the storage kernel by value:
/// the execution context the transform was staged in, the completed
/// transformer, and the stored block structures it maps between.
///
/// Host replays in that context, so a Host transform builds one context;
/// CUDA drops it before taking the device lease. The Host replay of the
/// staging mode's lane is bound here, where that lane's scalar bounds are
/// proved.
///
/// Internal and unstable despite being public for [`TransformExec`].
#[doc(hidden)]
pub struct TreeReplay<'a, D, C> {
    pub(in crate::typed) lease: crate::runtime::ContextLease,
    owned: OwnedReplay<D, C>,
    into: IntoReplay<D, C>,
    pub(in crate::typed) structure: &'a symtn_tensors::TreeTransformStructure<C>,
    pub(in crate::typed) destination: &'a Arc<BlockStructure>,
    pub(in crate::typed) source: &'a Arc<BlockStructure>,
    pub(in crate::typed) nout: usize,
}

impl<'a, D, C> TreeReplay<'a, D, C>
where
    D: symtn_tensors::DenseRecouplingScalar
        + RecouplingCoefficientAction<C>
        + symtn_tensors::ConjugateValue,
    C: symtn_tensors::DenseBlockScalar,
{
    /// A replay of `structure` on lane `L`, `destination <- source`.
    pub(in crate::typed) fn new<L: HostTreeLane<D, C>>(
        lease: crate::runtime::ContextLease,
        structure: &'a symtn_tensors::TreeTransformStructure<C>,
        destination: &'a Arc<BlockStructure>,
        source: &'a Arc<BlockStructure>,
        nout: usize,
    ) -> Self {
        Self {
            lease,
            owned: host_owned_replay::<L, D, C>,
            into: host_into_replay::<L, D, C>,
            structure,
            destination,
            source,
            nout,
        }
    }

    /// The replay of a staged transform on lane `L`.
    pub(in crate::typed) fn staged<L: HostTreeLane<D, C>, R, T>(
        lease: crate::runtime::ContextLease,
        staged: &'a symtn_tensors::StagedTreeTransform<'_, R, T, C>,
    ) -> Self {
        Self::new::<L>(
            lease,
            staged.structure(),
            staged.destination_structure(),
            staged.source_structure(),
            staged.nout(),
        )
    }
}

/// A storage kernel's failure. The replay's own error is kept apart so each
/// mode classifies it exactly as its staging's errors.
///
/// Internal and unstable despite being public for [`TransformExec`].
#[doc(hidden)]
pub enum TransformExecError {
    /// The replay failed.
    Replay(OperationError),
    /// A storage, placement or capability error.
    Storage(Error),
}

impl From<OperationError> for TransformExecError {
    fn from(error: OperationError) -> Self {
        Self::Replay(error)
    }
}

impl From<Error> for TransformExecError {
    fn from(error: Error) -> Self {
        Self::Storage(error)
    }
}

impl From<TransformExecError> for Error {
    fn from(error: TransformExecError) -> Self {
        match error {
            TransformExecError::Replay(error) => error.into(),
            TransformExecError::Storage(error) => error,
        }
    }
}

/// [`TransformExecError`] classified as the checked Generic staging
/// classifies its own replay: a plan failure.
pub(in crate::typed) fn checked_generic_exec_error<E>(
    error: TransformExecError,
) -> GenericTensorError<E> {
    match error {
        TransformExecError::Replay(error) => CheckedGenericPlanError::<E>::from(error).into(),
        TransformExecError::Storage(error) => error.into(),
    }
}

/// The source a transform `_into` replays: the stored payload (its own, or a
/// lazy adjoint's parent's, TensorKit `add_transform_kernel!` reading
/// `subblock(::AdjointTensorMap)` through the parent), or a compact bond.
/// `read` either writes `alpha · operation(src) + beta · dst` from the
/// resolved structure and the destination structure it was resolved against
/// and returns `true`, or returns `false` with `dst` untouched;
/// `dense_source` then builds the dense image that structure replays.
///
/// Internal and unstable despite being public for [`TransformExec`].
#[doc(hidden)]
#[allow(clippy::type_complexity)]
pub enum TransformIntoSource<'s, D, S, C> {
    Stored(&'s S),
    Bond {
        read: &'s mut dyn FnMut(
            &symtn_tensors::TreeTransformStructure<C>,
            &BlockStructure,
            &mut [D],
        ) -> Result<bool, Error>,
        dense_source: &'s dyn Fn() -> Result<Vec<D>, OperationError>,
    },
}

/// The storage kernels of the one tree-transform body: everything before
/// them — admission, staging, the destination checks — and the commit after
/// them are shared.
#[doc(hidden)]
pub trait TransformExec<R, D>: TypedStorage<R, D> {
    /// `alpha · structure(t)` into a fresh payload of `replay`'s destination
    /// structure, for an owned `t` (Host densifies a compact diagonal).
    fn replay_owned<C: Copy>(
        engine: &mut dyn Engine,
        _: ExecToken,
        replay: TreeReplay<'_, D, C>,
        t: &TensorMap<R, D, Self>,
        alpha: D,
    ) -> Result<Self, TransformExecError>
    where
        D: RecouplingCoefficientAction<C>;

    /// `destination = alpha · structure(source) + beta · destination` into a
    /// destination that passed every admission check.
    fn replay_into<C: Copy>(
        engine: &mut dyn Engine,
        _: ExecToken,
        replay: TreeReplay<'_, D, C>,
        source: TransformIntoSource<'_, D, Self, C>,
        destination: &mut Self,
        alpha: D,
        beta: D,
    ) -> Result<(), TransformExecError>
    where
        D: RecouplingCoefficientAction<C>;
}

impl<R, D: TensorScalar> TransformExec<R, D> for Vec<D> {
    fn replay_owned<C: Copy>(
        engine: &mut dyn Engine,
        _: ExecToken,
        replay: TreeReplay<'_, D, C>,
        t: &TensorMap<R, D>,
        alpha: D,
    ) -> Result<Self, TransformExecError> {
        let data = t
            .owned_body()
            .ok_or_else(|| internal_layout_error("owned transform input"))?
            .materialized_dense_data();
        let TreeReplay {
            mut lease,
            owned,
            structure,
            destination,
            source,
            nout,
            ..
        } = replay;
        owned(
            engine,
            lease.context(),
            structure,
            destination,
            source,
            nout,
            &data,
            alpha,
        )
    }

    fn replay_into<C: Copy>(
        engine: &mut dyn Engine,
        _: ExecToken,
        replay: TreeReplay<'_, D, C>,
        source: TransformIntoSource<'_, D, Self, C>,
        destination: &mut Self,
        alpha: D,
        beta: D,
    ) -> Result<(), TransformExecError> {
        let TreeReplay {
            mut lease,
            into,
            structure,
            destination: destination_structure,
            source: source_structure,
            ..
        } = replay;
        let densified;
        let source = match source {
            TransformIntoSource::Stored(data) => data.as_slice(),
            TransformIntoSource::Bond { read, dense_source } => {
                // The bond read writes the destination itself, ahead of the
                // replay's own engine admission.
                admit_host_engine(engine)?;
                if read(structure, destination_structure, destination.as_mut_slice())? {
                    return Ok(());
                }
                densified = dense_source().map_err(Error::from)?;
                &densified[..]
            }
        };
        into(
            engine,
            lease.context(),
            structure,
            destination_structure,
            source_structure,
            destination.as_mut_slice(),
            source,
            alpha,
            beta,
        )
    }
}

/// The stored element count a transform stages its source with; a compact
/// diagonal stages its dense image.
pub(in crate::typed) fn transform_source_len<R, D, S: TensorStorage<D>>(
    body: &TypedTensorBody<R, D, S>,
) -> Result<usize, Error> {
    Ok(match body.data.as_ref() {
        TypedData::Dense(data) => TensorStorage::len(data),
        TypedData::Diagonal(_) => body.space.space().required_len()?,
    })
}

/// The source of a transform `_into` as its storage kernel replays it,
/// handed to `run`. A compact bond is read from the resolved structure when
/// that structure is a single-term square bijection, as the owned compact arm
/// reads it, and is otherwise densified once for the dense replay.
pub(in crate::typed) fn with_transform_into_source<R, D, S, C, T>(
    source: &TensorMap<R, D, S>,
    operation: &TreeTransformOperation,
    alpha: D,
    beta: D,
    run: impl FnOnce(TransformIntoSource<'_, D, S, C>) -> T,
) -> T
where
    D: TensorScalar + symtn_tensors::RecouplingCoefficientAction<C>,
    C: Copy,
{
    let body = source.storage_body();
    let spectrum = match body.data.as_ref() {
        TypedData::Dense(data) => return run(TransformIntoSource::Stored(data)),
        TypedData::Diagonal(spectrum) => spectrum,
    };
    let logical = &body.space;
    // As the owned arm: a swap keeps whatever coefficient it carries, and any
    // other read keeps the dense replay's off-diagonal `0 · c` products.
    let require_representable = !crate::tensor_product::is_rank_one_diagonal_swap(
        source.codomain_rank(),
        source.domain_rank(),
        operation,
    );
    let mut read = |structure: &symtn_tensors::TreeTransformStructure<C>,
                    destination: &symtn_core::BlockStructure,
                    data: &mut [D]|
     -> Result<bool, Error> {
        let Some(diagonal) = crate::tensor_product::bond_spectrum(
            logical.space().structure(),
            destination,
            structure,
            spectrum,
            require_representable,
        ) else {
            return Ok(false);
        };
        axpby_bond_diagonal(destination, data, &diagonal, alpha, beta)?;
        Ok(true)
    };
    let dense_source =
        || symtn_matrixalgebra::seam::diagonal_bond_data(logical.space(), spectrum, &|v| v);
    run(TransformIntoSource::Bond {
        read: &mut read,
        dense_source: &dense_source,
    })
}

/// `data = beta · data`: a strong zero for `beta == 0`, which never reads
/// `data`, and nothing for `beta == 1`.
pub(in crate::typed) fn scale_by_beta<D: TensorScalar>(data: &mut [D], beta: D) {
    if beta.is_zero() {
        data.fill(D::zero());
    } else if beta != D::one() {
        for value in data.iter_mut() {
            *value = beta * *value;
        }
    }
}
