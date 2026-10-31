use core::ops::{Add, Mul};
use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::Arc;
use symtn_dense::Engine;

use num_traits::Zero;
use symtn_core::{
    BlockStructure, CheckedGenericAdmissionMode, CheckedGenericRigidSymbols,
    MultiplicityFreeAdmissionMode, MultiplicityFreeRigidSymbols, Placement, RuleIdentity,
};

use crate::contract::{BoundDynamicFusionMapSpace, DynamicFusionMapSpace, FusionOperand};
use crate::mode::{PlanningAlgebra, StagedTransform, TreeStructureSource};
use crate::tree_transform::{
    CheckedGenericPlanError, TreeTransformOperation, TreeTransformPlanning,
    TreeTransformRuleCacheKey,
};
use crate::{
    RecouplingCoefficientAction, ReportsPlacement, TreeTransformReplayProfile,
    TreeTransformStructure,
};
use symtn_dense::DefaultDenseExecutor;
use symtn_operations::OperationError;
use symtn_operations::TreeTransformScalar;
use symtn_operations::{DenseTreeTransformOperations, TreeTransformBackend};

#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn replay_structure_overwrite<D, C, B>(
    engine: &mut dyn Engine,
    backend: &mut B,
    workspace: &mut B::Workspace,
    structure: &TreeTransformStructure<C>,
    dst_structure: &Arc<BlockStructure>,
    src_structure: &Arc<BlockStructure>,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: D,
    destination_scales: &[(usize, C)],
    profile: Option<&mut TreeTransformReplayProfile>,
) -> Result<(), OperationError>
where
    D: TreeTransformScalar + RecouplingCoefficientAction<C>,
    C: Copy,
    B: TreeTransformBackend<D, C>,
{
    match profile {
        Some(profile) => backend.tree_transform_structure_overwrite_into_raw_profiled(
            engine,
            workspace,
            structure,
            dst_structure,
            src_structure,
            dst_data,
            src_data,
            alpha,
            destination_scales,
            profile,
        ),
        None => backend.tree_transform_structure_overwrite_into_raw(
            engine,
            workspace,
            structure,
            dst_structure,
            src_structure,
            dst_data,
            src_data,
            alpha,
            destination_scales,
        ),
    }
}

#[derive(Debug)]
pub struct TreeTransformExecutionContext<D, RuleKey, C = D, B = DenseTreeTransformOperations>
where
    D: TreeTransformScalar,
    C: Copy,
    B: TreeTransformBackend<D, C>,
{
    backend: B,
    workspace: B::Workspace,
    planning: TreeTransformPlanning,
    rule_key: PhantomData<fn() -> RuleKey>,
}

impl<D, RuleKey, C, B> TreeTransformExecutionContext<D, RuleKey, C, B>
where
    D: TreeTransformScalar,
    C: Copy,
    B: TreeTransformBackend<D, C>,
{
    /// Completed tree transformers are resolved through the process-global
    /// cache (`symtn::cache`); a context owns only its backend and workspace.
    pub fn with_parts(backend: B, workspace: B::Workspace) -> Self {
        Self {
            backend,
            workspace,
            planning: TreeTransformPlanning::default(),
            rule_key: PhantomData,
        }
    }

    #[inline]
    pub fn backend(&self) -> &B {
        &self.backend
    }

    #[inline]
    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    #[inline]
    pub fn workspace(&self) -> &B::Workspace {
        &self.workspace
    }

    #[inline]
    pub(crate) fn backend_workspace_mut(&mut self) -> (&mut B, &mut B::Workspace) {
        (&mut self.backend, &mut self.workspace)
    }

    pub fn into_parts(self) -> (B, B::Workspace) {
        (self.backend, self.workspace)
    }
}

impl<D, RuleKey, C>
    TreeTransformExecutionContext<D, RuleKey, C, DenseTreeTransformOperations<DefaultDenseExecutor>>
where
    D: crate::DenseRecouplingScalar + RecouplingCoefficientAction<C> + crate::ConjugateValue,
    C: 'static + Copy + Clone + Add<Output = C> + Mul<Output = C> + Zero + Send + Sync,
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
{
    /// Attempts the built-in uninitialised writer used by owned tensor
    /// transforms. `Ok(None)` means the proof was unavailable and no output
    /// was allocated. The writer runs under the backend's replay worker
    /// count: the parallel schedule splits the destination on its compiled
    /// slice-disjoint boundaries, so thread count never forces the
    /// zero-then-replay fallback.
    ///
    /// This concrete cross-crate entrypoint is internal and unstable despite
    /// being public for `symtn`; downstream callers must not rely on it.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn try_tree_transform_dyn_overwrite_owned<R>(
        &mut self,
        engine: &mut dyn Engine,
        rule: &R,
        operation: &TreeTransformOperation,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        nout: usize,
        src_data: &[D],
        alpha: D,
    ) -> Result<Option<Vec<D>>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        let structure = self.tree_structure(
            engine,
            rule,
            operation,
            dst_structure,
            TreeStructureSource::Stored {
                structure: src_structure,
                storage_conjugate: false,
            },
        )?;
        self.backend
            .try_tree_transform_structure_overwrite_owned_raw(
                engine,
                &mut self.workspace,
                &structure,
                dst_structure,
                src_structure,
                nout,
                src_data,
                alpha,
            )
    }
}

impl<D, C>
    TreeTransformExecutionContext<
        D,
        RuleIdentity,
        C,
        DenseTreeTransformOperations<DefaultDenseExecutor>,
    >
where
    D: crate::DenseRecouplingScalar + RecouplingCoefficientAction<C> + crate::ConjugateValue,
    C: 'static + Copy + Clone + Add<Output = C> + Mul<Output = C> + Zero + Send + Sync,
{
    /// The owned multiplicity-free permute, braid, transpose or repartition
    /// `alpha · operation(src)` of a direct source, into a fresh payload.
    ///
    /// This concrete cross-crate entrypoint is internal and unstable despite
    /// being public for `symtn`; downstream callers must not rely on it.
    #[doc(hidden)]
    pub fn tree_transform_owned_multiplicity_free_in<R>(
        &mut self,
        engine: &mut dyn Engine,
        src: &BoundDynamicFusionMapSpace<R>,
        src_data: &[D],
        operation: &TreeTransformOperation,
        alpha: D,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleIdentity>,
    {
        self.tree_transform_owned_in::<MultiplicityFreeAdmissionMode, R>(
            engine, src, None, src_data, operation, alpha,
        )
    }

    /// Stages and resolves the multiplicity-free permute, braid, transpose or
    /// repartition of `logical` (stored as the parent `adjoint_of` when it is
    /// that parent's lazy adjoint, `src_len` stored elements): the owned
    /// entry's staging, with the replay left to the caller's storage and the
    /// commit to [`StagedTreeTransform::commit`].
    ///
    /// This concrete cross-crate entrypoint is internal and unstable despite
    /// being public for `symtn`; downstream callers must not rely on it.
    #[doc(hidden)]
    pub fn stage_tree_transform_multiplicity_free_in<'a, R>(
        &mut self,
        engine: &mut dyn Engine,
        logical: &'a BoundDynamicFusionMapSpace<R>,
        adjoint_of: Option<&'a BoundDynamicFusionMapSpace<R>>,
        src_len: usize,
        operation: &TreeTransformOperation,
    ) -> Result<StagedTreeTransform<'a, R, BoundDynamicFusionMapSpace<R>, C>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleIdentity>,
    {
        let resolved = self.stage_resolve::<MultiplicityFreeAdmissionMode, R>(
            engine, logical, adjoint_of, src_len, operation,
        )?;
        Ok(StagedTreeTransform { logical, resolved })
    }

    /// [`Self::stage_tree_transform_multiplicity_free_in`] for an existing
    /// destination `dst_space` (TensorKit `spacecheck_transform`): the
    /// structure is resolved against `dst_space`'s own block structure, and a
    /// `dst_space` that is not the staged destination is
    /// [`OperationError::SpaceMismatch`].
    ///
    /// This concrete cross-crate entrypoint is internal and unstable despite
    /// being public for `symtn`; downstream callers must not rely on it.
    #[doc(hidden)]
    pub fn stage_tree_transform_into_multiplicity_free_in<'a, R>(
        &mut self,
        engine: &mut dyn Engine,
        logical: &'a BoundDynamicFusionMapSpace<R>,
        adjoint_of: Option<&'a BoundDynamicFusionMapSpace<R>>,
        src_len: usize,
        operation: &TreeTransformOperation,
        dst_space: &'a DynamicFusionMapSpace,
    ) -> Result<StagedTreeTransform<'a, R, BoundDynamicFusionMapSpace<R>, C>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleIdentity>,
    {
        let resolved = self.stage_into_in::<MultiplicityFreeAdmissionMode, R, OperationError>(
            engine, logical, adjoint_of, src_len, operation, dst_space,
        )?;
        Ok(StagedTreeTransform { logical, resolved })
    }

    /// The owned multiplicity-free `operation` of a direct source in one
    /// staging: `read` answers from the resolved structure and the
    /// destination preview, or else `dense_source` is replayed through that
    /// same structure; the destination is committed once either way.
    ///
    /// This concrete cross-crate entrypoint is internal and unstable despite
    /// being public for `symtn`; downstream callers must not rely on it.
    #[doc(hidden)]
    #[allow(clippy::type_complexity)]
    pub fn tree_transform_structure_multiplicity_free_in<R, T>(
        &mut self,
        engine: &mut dyn Engine,
        logical: &BoundDynamicFusionMapSpace<R>,
        operation: &TreeTransformOperation,
        read: impl FnOnce(&TreeTransformStructure<C>, &BlockStructure) -> Option<T>,
        dense_source: impl FnOnce() -> Result<Vec<D>, OperationError>,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, StructureOutcome<T, D>), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleIdentity>,
    {
        self.tree_transform_structure_in::<MultiplicityFreeAdmissionMode, R, T>(
            engine,
            logical,
            operation,
            read,
            dense_source,
        )
    }

    /// The one owned eager tree transform of both modes: stage and resolve,
    /// replay, then commit and publish.
    fn tree_transform_owned_in<M, R>(
        &mut self,
        engine: &mut dyn Engine,
        logical: &BoundDynamicFusionMapSpace<R>,
        adjoint_of: Option<&BoundDynamicFusionMapSpace<R>>,
        src_data: &[D],
        operation: &TreeTransformOperation,
        alpha: D,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), M::Error>
    where
        M: PlanningAlgebra<R, Scalar = C>,
    {
        let resolved =
            self.stage_resolve::<M, R>(engine, logical, adjoint_of, src_data.len(), operation)?;
        let data = self.replay_owned(engine, &resolved, src_data, alpha)?;
        let destination = M::commit_transform(logical, resolved.stage, resolved.preview)?;
        Ok((destination, data))
    }

    /// [`Self::stage_resolve`] for an existing destination `dst_space`, in
    /// the mode's match/resolve order. A mismatch is the caller's argument
    /// error in every mode, never a checked plan failure.
    fn stage_into_in<'a, M, R, E>(
        &mut self,
        engine: &mut dyn Engine,
        logical: &'a BoundDynamicFusionMapSpace<R>,
        adjoint_of: Option<&'a BoundDynamicFusionMapSpace<R>>,
        src_len: usize,
        operation: &TreeTransformOperation,
        dst_space: &'a DynamicFusionMapSpace,
    ) -> Result<ResolvedTransform<'a, M::TransformStage, C>, E>
    where
        M: PlanningAlgebra<R, Scalar = C>,
        E: From<M::Error> + From<OperationError>,
    {
        let staged = M::stage_transform(logical, adjoint_of, src_len, operation)?;
        M::match_and_resolve(
            staged,
            dst_space,
            |staged, dst| {
                self.resolve_staged::<M, R>(engine, logical, adjoint_of, operation, staged, dst)
            },
            |resolved| &resolved.stage,
        )?
        .ok_or_else(|| {
            E::from(OperationError::SpaceMismatch {
                message:
                    "destination fusion space or block layout does not match the operation result",
            })
        })
    }

    /// [`Self::tree_transform_owned_in`] with the replay replaced by `read`
    /// when it answers. A decline replays `dense_source` with the same
    /// structure, so every path stages, resolves and commits exactly once
    /// and a decline costs the dense route's provider calls and errors.
    #[allow(clippy::type_complexity)]
    fn tree_transform_structure_in<M, R, T>(
        &mut self,
        engine: &mut dyn Engine,
        logical: &BoundDynamicFusionMapSpace<R>,
        operation: &TreeTransformOperation,
        read: impl FnOnce(&TreeTransformStructure<C>, &BlockStructure) -> Option<T>,
        dense_source: impl FnOnce() -> Result<Vec<D>, OperationError>,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, StructureOutcome<T, D>), M::Error>
    where
        M: PlanningAlgebra<R, Scalar = C>,
    {
        // Why the logical length: the caller holds a compact payload whose
        // dense image has exactly this length, and replays only that image.
        let src_len = logical
            .space()
            .required_len()
            .map_err(OperationError::from)?;
        let resolved = self.stage_resolve::<M, R>(engine, logical, None, src_len, operation)?;
        let outcome = match read(&resolved.structure, &resolved.preview) {
            Some(value) => StructureOutcome::Read(value),
            None => {
                let src_data = dense_source()?;
                StructureOutcome::Replayed(self.replay_owned(
                    engine,
                    &resolved,
                    &src_data,
                    D::one(),
                )?)
            }
        };
        let destination = M::commit_transform(logical, resolved.stage, resolved.preview)?;
        Ok((destination, outcome))
    }

    /// Stages the destination (checked: admit and preflight the source) and
    /// resolves the structure with the staging's proof.
    fn stage_resolve<'a, M, R>(
        &mut self,
        engine: &mut dyn Engine,
        logical: &'a BoundDynamicFusionMapSpace<R>,
        adjoint_of: Option<&'a BoundDynamicFusionMapSpace<R>>,
        src_len: usize,
        operation: &TreeTransformOperation,
    ) -> Result<ResolvedTransform<'a, M::TransformStage, C>, M::Error>
    where
        M: PlanningAlgebra<R, Scalar = C>,
    {
        let staged = M::stage_transform(logical, adjoint_of, src_len, operation)?;
        self.resolve_staged::<M, R>(engine, logical, adjoint_of, operation, staged, None)
    }

    /// Resolves the structure of a staged transform with the staging's
    /// proof, against `dst` or else the staged preview.
    fn resolve_staged<'a, M, R>(
        &mut self,
        engine: &mut dyn Engine,
        logical: &'a BoundDynamicFusionMapSpace<R>,
        adjoint_of: Option<&'a BoundDynamicFusionMapSpace<R>>,
        operation: &TreeTransformOperation,
        staged: StagedTransform<M::TransformStage, M::SourceProof<'a>>,
        dst: Option<&'a Arc<BlockStructure>>,
    ) -> Result<ResolvedTransform<'a, M::TransformStage, C>, M::Error>
    where
        M: PlanningAlgebra<R, Scalar = C>,
    {
        let StagedTransform {
            preview,
            nout,
            mut stage,
            proof,
        } = staged;
        let (source, src_structure) = match adjoint_of {
            None => {
                let structure = logical.space().structure();
                (
                    TreeStructureSource::Stored {
                        structure,
                        storage_conjugate: false,
                    },
                    structure,
                )
            }
            Some(parent) => (
                TreeStructureSource::StorageMapped {
                    logical: logical.space().structure(),
                    operand: FusionOperand::adjoint(parent.space()),
                },
                parent.space().structure(),
            ),
        };
        let structure = M::tree_structure_with(
            engine,
            M::transform_cache(self.planning(), &mut stage),
            logical.provider(),
            operation,
            dst.unwrap_or(&preview),
            source,
            Some(&proof),
        )?;
        Ok(ResolvedTransform {
            preview,
            replay_destination: dst,
            nout,
            stage,
            structure,
            src_structure,
        })
    }

    /// Writes `alpha · structure(src_data)` into a fresh payload through the
    /// built-in overwrite writer, or else zero-fills and replays.
    fn replay_owned<S>(
        &mut self,
        engine: &mut dyn Engine,
        resolved: &ResolvedTransform<'_, S, C>,
        src_data: &[D],
        alpha: D,
    ) -> Result<Vec<D>, OperationError> {
        self.replay_tree_transform_dyn_owned(
            engine,
            &resolved.structure,
            &resolved.preview,
            resolved.src_structure,
            resolved.nout,
            src_data,
            alpha,
        )
    }

    /// The owned replay of a staged transform:
    /// `alpha · structure(src_data)` into a fresh `dst_structure` payload,
    /// through the built-in overwrite writer, or else zero-filled and
    /// replayed.
    ///
    /// This concrete cross-crate entrypoint is internal and unstable despite
    /// being public for `symtn`; downstream callers must not rely on it.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn replay_tree_transform_dyn_owned(
        &mut self,
        engine: &mut dyn Engine,
        structure: &TreeTransformStructure<C>,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        nout: usize,
        src_data: &[D],
        alpha: D,
    ) -> Result<Vec<D>, OperationError> {
        if let Some(data) = self
            .backend
            .try_tree_transform_structure_overwrite_owned_raw(
                engine,
                &mut self.workspace,
                structure,
                dst_structure,
                src_structure,
                nout,
                src_data,
                alpha,
            )?
        {
            return Ok(data);
        }
        let mut data = vec![D::zero(); dst_structure.required_len().map_err(OperationError::from)?];
        self.tree_transform_structure_into_raw(
            engine,
            structure,
            dst_structure,
            src_structure,
            &mut data,
            src_data,
            alpha,
            D::zero(),
        )?;
        Ok(data)
    }
}

/// A staged destination with its resolved structure, between staging and
/// commit.
struct ResolvedTransform<'a, S, C> {
    preview: Arc<BlockStructure>,
    /// The structure resolved against when not `preview`: an existing
    /// destination's, which the replay then writes.
    replay_destination: Option<&'a Arc<BlockStructure>>,
    nout: usize,
    stage: S,
    structure: TreeTransformStructure<C>,
    src_structure: &'a Arc<BlockStructure>,
}

/// What a structure-reading transform produced: the reader's answer, or the
/// dense replay of the declined source.
///
/// Internal and unstable despite being public for `symtn`.
#[doc(hidden)]
pub enum StructureOutcome<T, D> {
    Read(T),
    Replayed(Vec<D>),
}

/// One staged eager tree transform between its staging and its commit: the
/// resolved structure and the block structures its replay maps between.
/// It borrows only the source and destination spaces, never the context it
/// was staged on, so a storage may release that context before replaying.
///
/// Internal and unstable despite being public for `symtn`.
#[doc(hidden)]
pub struct StagedTreeTransform<'a, R, S, C> {
    logical: &'a BoundDynamicFusionMapSpace<R>,
    resolved: ResolvedTransform<'a, S, C>,
}

impl<R, S, C> StagedTreeTransform<'_, R, S, C> {
    /// The completed transformer the replay runs.
    #[inline]
    pub fn structure(&self) -> &TreeTransformStructure<C> {
        &self.resolved.structure
    }

    /// The destination block structure the replay writes: the existing
    /// destination's, else the staged one.
    #[inline]
    pub fn destination_structure(&self) -> &Arc<BlockStructure> {
        self.resolved
            .replay_destination
            .unwrap_or(&self.resolved.preview)
    }

    /// The stored source block structure the replay reads: the source's own,
    /// or its lazy-adjoint parent's.
    #[inline]
    pub fn source_structure(&self) -> &Arc<BlockStructure> {
        self.resolved.src_structure
    }

    /// The destination codomain rank.
    #[inline]
    pub fn nout(&self) -> usize {
        self.resolved.nout
    }
}

impl<R> StagedTreeTransform<'_, R, BoundDynamicFusionMapSpace<R>, R::Scalar>
where
    R: MultiplicityFreeRigidSymbols + TreeTransformRuleCacheKey,
    R::Scalar:
        'static + Copy + Add<Output = R::Scalar> + Mul<Output = R::Scalar> + Zero + Send + Sync,
{
    /// Commits the staged multiplicity-free destination after the replay.
    pub fn commit(self) -> Result<BoundDynamicFusionMapSpace<R>, OperationError> {
        let ResolvedTransform { stage, preview, .. } = self.resolved;
        <MultiplicityFreeAdmissionMode as PlanningAlgebra<R>>::commit_transform(
            self.logical,
            stage,
            preview,
        )
    }
}

/// A checked Generic transform's staged destination, its call transaction
/// and its memo publication, held opaque until [`StagedTreeTransform::commit`].
///
/// Internal and unstable despite being public for `symtn`.
#[doc(hidden)]
pub struct CheckedTreeTransformStage(
    (
        crate::contract::PreparedCheckedGenericDynamicSpace,
        crate::contract::CheckedContractTxn,
        Option<symtn_core::PermutedMemoTicket>,
    ),
);

impl<'a, R> StagedTreeTransform<'a, R, CheckedTreeTransformStage, f64>
where
    R: CheckedGenericRigidSymbols<Scalar = f64>,
{
    fn checked(
        logical: &'a BoundDynamicFusionMapSpace<R>,
        resolved: ResolvedTransform<
            'a,
            <CheckedGenericAdmissionMode as PlanningAlgebra<R>>::TransformStage,
            f64,
        >,
    ) -> Self {
        let ResolvedTransform {
            preview,
            replay_destination,
            nout,
            stage,
            structure,
            src_structure,
        } = resolved;
        Self {
            logical,
            resolved: ResolvedTransform {
                preview,
                replay_destination,
                nout,
                stage: CheckedTreeTransformStage(stage),
                structure,
                src_structure,
            },
        }
    }

    /// Commits the staged checked Generic destination after the replay, then
    /// publishes what the transform staged.
    pub fn commit(
        self,
    ) -> Result<BoundDynamicFusionMapSpace<R>, CheckedGenericPlanError<R::Error>> {
        let ResolvedTransform { stage, preview, .. } = self.resolved;
        <CheckedGenericAdmissionMode as PlanningAlgebra<R>>::commit_transform(
            self.logical,
            stage.0,
            preview,
        )
    }
}

impl<D>
    TreeTransformExecutionContext<
        D,
        RuleIdentity,
        f64,
        DenseTreeTransformOperations<DefaultDenseExecutor>,
    >
where
    D: crate::DenseRecouplingScalar + RecouplingCoefficientAction<f64> + crate::ConjugateValue,
{
    /// The owned checked Generic permute, braid, transpose or repartition
    /// `alpha · operation(logical)`, into a fresh payload. `adjoint_of` is
    /// `None` for a direct source stored as `logical`, or the parent whose
    /// storage `src_data` is when `logical` is that parent's lazy adjoint.
    ///
    /// Provider queries and replay compilation finish against an uninterned
    /// destination preview; the destination structure, composed coefficients
    /// and completed transformer become visible only after those fallible
    /// stages succeed.
    ///
    /// This concrete cross-crate entrypoint is internal and unstable despite
    /// being public for `symtn`; downstream callers must not rely on it.
    #[doc(hidden)]
    #[allow(clippy::type_complexity)]
    pub fn tree_transform_owned_checked_generic_in<R>(
        &mut self,
        engine: &mut dyn Engine,
        logical: &BoundDynamicFusionMapSpace<R>,
        adjoint_of: Option<&BoundDynamicFusionMapSpace<R>>,
        src_data: &[D],
        operation: &TreeTransformOperation,
        alpha: D,
    ) -> Result<(BoundDynamicFusionMapSpace<R>, Vec<D>), CheckedGenericPlanError<R::Error>>
    where
        R: CheckedGenericRigidSymbols<Scalar = f64>,
    {
        self.tree_transform_owned_in::<CheckedGenericAdmissionMode, R>(
            engine, logical, adjoint_of, src_data, operation, alpha,
        )
    }

    /// The checked Generic twin of
    /// [`Self::stage_tree_transform_multiplicity_free_in`]: admits and
    /// preflights the source and resolves against an uninterned destination
    /// preview; nothing publishes before [`StagedTreeTransform::commit`].
    ///
    /// This concrete cross-crate entrypoint is internal and unstable despite
    /// being public for `symtn`; downstream callers must not rely on it.
    #[doc(hidden)]
    #[allow(clippy::type_complexity)]
    pub fn stage_tree_transform_checked_generic_in<'a, R>(
        &mut self,
        engine: &mut dyn Engine,
        logical: &'a BoundDynamicFusionMapSpace<R>,
        adjoint_of: Option<&'a BoundDynamicFusionMapSpace<R>>,
        src_len: usize,
        operation: &TreeTransformOperation,
    ) -> Result<
        StagedTreeTransform<'a, R, CheckedTreeTransformStage, f64>,
        CheckedGenericPlanError<R::Error>,
    >
    where
        R: CheckedGenericRigidSymbols<Scalar = f64>,
    {
        let resolved = self.stage_resolve::<CheckedGenericAdmissionMode, R>(
            engine, logical, adjoint_of, src_len, operation,
        )?;
        Ok(StagedTreeTransform::checked(logical, resolved))
    }

    /// The checked Generic twin of
    /// [`Self::stage_tree_transform_into_multiplicity_free_in`]: resolved
    /// against the staged preview, then matched with `dst_space`.
    ///
    /// This concrete cross-crate entrypoint is internal and unstable despite
    /// being public for `symtn`; downstream callers must not rely on it.
    #[doc(hidden)]
    pub fn stage_tree_transform_into_checked_generic_in<'a, R, E>(
        &mut self,
        engine: &mut dyn Engine,
        logical: &'a BoundDynamicFusionMapSpace<R>,
        adjoint_of: Option<&'a BoundDynamicFusionMapSpace<R>>,
        src_len: usize,
        operation: &TreeTransformOperation,
        dst_space: &'a DynamicFusionMapSpace,
    ) -> Result<StagedTreeTransform<'a, R, CheckedTreeTransformStage, f64>, E>
    where
        R: CheckedGenericRigidSymbols<Scalar = f64>,
        E: From<CheckedGenericPlanError<R::Error>> + From<OperationError>,
    {
        let resolved = self.stage_into_in::<CheckedGenericAdmissionMode, R, E>(
            engine, logical, adjoint_of, src_len, operation, dst_space,
        )?;
        Ok(StagedTreeTransform::checked(logical, resolved))
    }

    /// The checked Generic twin of
    /// [`Self::tree_transform_structure_multiplicity_free_in`]: one admission,
    /// staging, structure resolution and commit, whether `read` answers or
    /// `dense_source` is replayed.
    ///
    /// This concrete cross-crate entrypoint is internal and unstable despite
    /// being public for `symtn`; downstream callers must not rely on it.
    #[doc(hidden)]
    #[allow(clippy::type_complexity)]
    pub fn tree_transform_structure_checked_generic_in<R, T>(
        &mut self,
        engine: &mut dyn Engine,
        logical: &BoundDynamicFusionMapSpace<R>,
        operation: &TreeTransformOperation,
        read: impl FnOnce(&TreeTransformStructure<f64>, &BlockStructure) -> Option<T>,
        dense_source: impl FnOnce() -> Result<Vec<D>, OperationError>,
    ) -> Result<
        (BoundDynamicFusionMapSpace<R>, StructureOutcome<T, D>),
        CheckedGenericPlanError<R::Error>,
    >
    where
        R: CheckedGenericRigidSymbols<Scalar = f64>,
    {
        self.tree_transform_structure_in::<CheckedGenericAdmissionMode, R, T>(
            engine,
            logical,
            operation,
            read,
            dense_source,
        )
    }
}

impl<D, RuleKey, C, B> TreeTransformExecutionContext<D, RuleKey, C, B>
where
    D: TreeTransformScalar,
    C: Copy,
    B: TreeTransformBackend<D, C> + ReportsPlacement,
    B::Workspace: ReportsPlacement,
{
    #[inline]
    pub fn backend_placement(&self) -> Placement {
        self.backend.placement()
    }

    #[inline]
    pub fn workspace_placement(&self) -> Placement {
        self.workspace.placement()
    }

    #[inline]
    pub fn is_host_context(&self) -> bool {
        self.backend.is_host_placement() && self.workspace.is_host_placement()
    }
}

impl<D, RuleKey, C, B> TreeTransformExecutionContext<D, RuleKey, C, B>
where
    D: TreeTransformScalar,
    C: Copy,
    RuleKey: Clone + Eq + Hash,
    B: TreeTransformBackend<D, C>,
    B::Workspace: Default,
{
    pub fn new(backend: B) -> Self {
        Self::with_parts(backend, B::Workspace::default())
    }
}

impl<D, RuleKey, C, B> Default for TreeTransformExecutionContext<D, RuleKey, C, B>
where
    D: TreeTransformScalar,
    C: Copy,
    RuleKey: Clone + Eq + Hash,
    B: TreeTransformBackend<D, C> + Default,
    B::Workspace: Default,
{
    fn default() -> Self {
        Self::new(B::default())
    }
}

impl<D, RuleKey, C, B> TreeTransformExecutionContext<D, RuleKey, C, B>
where
    D: TreeTransformScalar,
    C: 'static + Copy + Clone + Add<Output = C> + Mul<Output = C> + Zero + Send + Sync,
    RuleKey: 'static + Clone + Eq + Hash + Send + Sync,
    B: TreeTransformBackend<D, C>,
{
    /// Dynamic-rank tree transform (permute / braid / transpose): operates
    /// on raw slices plus their block structures, through this context's
    /// structure-compile cache. `dst_data` must be
    /// zero-filled (or carry the `beta`-scaled accumuland) and sized for
    /// `dst_structure.required_len()`.
    #[allow(clippy::too_many_arguments)]
    pub fn tree_transform_dyn_into<R>(
        &mut self,
        engine: &mut dyn Engine,
        rule: &R,
        operation: TreeTransformOperation,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        src_data: &[D],
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        self.tree_transform_into_raw_with_storage_conjugation(
            engine,
            rule,
            operation,
            dst_structure,
            src_structure,
            dst_data,
            src_data,
            false,
            alpha,
            beta,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn tree_transform_dyn_into_ref<R>(
        &mut self,
        engine: &mut dyn Engine,
        rule: &R,
        operation: &TreeTransformOperation,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        src_data: &[D],
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        let structure = self.tree_structure(
            engine,
            rule,
            operation,
            dst_structure,
            TreeStructureSource::Stored {
                structure: src_structure,
                storage_conjugate: false,
            },
        )?;
        self.backend.tree_transform_structure_into_raw(
            engine,
            &mut self.workspace,
            &structure,
            dst_structure,
            src_structure,
            dst_data,
            src_data,
            alpha,
            beta,
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[doc(hidden)]
    pub fn tree_transform_dyn_overwrite_into_ref<R>(
        &mut self,
        engine: &mut dyn Engine,
        rule: &R,
        operation: &TreeTransformOperation,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        src_data: &[D],
        alpha: D,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
        D: RecouplingCoefficientAction<C>,
    {
        self.compile_and_replay_overwrite(
            engine,
            |context, engine| {
                context.tree_structure(
                    engine,
                    rule,
                    operation,
                    dst_structure,
                    TreeStructureSource::Stored {
                        structure: src_structure,
                        storage_conjugate: false,
                    },
                )
            },
            dst_structure,
            src_structure,
            dst_data,
            src_data,
            alpha,
        )
    }

    /// The one tree-structure resolution of the planner and the eager
    /// transform entries: the multiplicity-free
    /// [`PlanningAlgebra::tree_structure`] with this context's planning state.
    pub(crate) fn tree_structure<R>(
        &mut self,
        engine: &mut dyn Engine,
        rule: &R,
        operation: &TreeTransformOperation,
        dst_structure: &Arc<BlockStructure>,
        src: TreeStructureSource<'_>,
    ) -> Result<TreeTransformStructure<C>, OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        <MultiplicityFreeAdmissionMode as PlanningAlgebra<R>>::tree_structure(
            engine,
            self.planning(),
            rule,
            operation,
            dst_structure,
            src,
        )
    }

    /// This context's multiplicity-free planning state: the
    /// [`PlanningAlgebra::StructureCache`] a multiplicity-free resolution
    /// scope plans with.
    pub(crate) fn planning(&mut self) -> &mut TreeTransformPlanning {
        &mut self.planning
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tree_transform_into_raw_with_storage_conjugation<R>(
        &mut self,
        engine: &mut dyn Engine,
        rule: &R,
        operation: TreeTransformOperation,
        dst_structure: &std::sync::Arc<BlockStructure>,
        src_structure: &std::sync::Arc<BlockStructure>,
        dst_data: &mut [D],
        src_data: &[D],
        storage_conjugate: bool,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols<Scalar = C> + TreeTransformRuleCacheKey<Key = RuleKey>,
    {
        let structure = self.tree_structure(
            engine,
            rule,
            &operation,
            dst_structure,
            TreeStructureSource::Stored {
                structure: src_structure,
                storage_conjugate,
            },
        )?;
        self.backend.tree_transform_structure_into_raw(
            engine,
            &mut self.workspace,
            &structure,
            dst_structure,
            src_structure,
            dst_data,
            src_data,
            alpha,
            beta,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn compile_and_replay_overwrite<F>(
        &mut self,
        engine: &mut dyn Engine,
        compile: F,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        src_data: &[D],
        alpha: D,
    ) -> Result<(), OperationError>
    where
        F: FnOnce(&mut Self, &mut dyn Engine) -> Result<TreeTransformStructure<C>, OperationError>,
        D: RecouplingCoefficientAction<C>,
    {
        let structure = compile(self, engine)?;
        let Self {
            backend, workspace, ..
        } = self;
        replay_structure_overwrite(
            engine,
            backend,
            workspace,
            &structure,
            dst_structure,
            src_structure,
            dst_data,
            src_data,
            alpha,
            &[],
            None,
        )
    }

    /// Replays an already resolved transformer, beta-accumulating: the
    /// typed exact-layout hit's entry.
    ///
    /// This concrete cross-crate entrypoint is internal and unstable despite
    /// being public for `symtn`; downstream callers must not rely on it.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn replay_tree_transform_dyn_into(
        &mut self,
        engine: &mut dyn Engine,
        structure: &TreeTransformStructure<C>,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        src_data: &[D],
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError> {
        self.tree_transform_structure_into_raw(
            engine,
            structure,
            dst_structure,
            src_structure,
            dst_data,
            src_data,
            alpha,
            beta,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tree_transform_structure_into_raw(
        &mut self,
        engine: &mut dyn Engine,
        structure: &TreeTransformStructure<C>,
        dst_structure: &Arc<BlockStructure>,
        src_structure: &Arc<BlockStructure>,
        dst_data: &mut [D],
        src_data: &[D],
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError> {
        let Self {
            backend, workspace, ..
        } = self;
        backend.tree_transform_structure_into_raw(
            engine,
            workspace,
            structure,
            dst_structure,
            src_structure,
            dst_data,
            src_data,
            alpha,
            beta,
        )
    }
}
