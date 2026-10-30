use super::*;
use symtn_dense::Engine;

#[derive(Clone, Debug)]
pub(crate) struct DynamicTreeExecutionArtifact<C = f64> {
    pub(in crate::contract) orientation: FusionContractOrientation,
    pub(in crate::contract) lhs_transform: DynamicFusionTransformedSourceEntry<C>,
    pub(in crate::contract) rhs_transform: DynamicFusionTransformedSourceEntry<C>,
    pub(in crate::contract) lhs_borrowed: bool,
    pub(in crate::contract) rhs_borrowed: bool,
    /// The fermionic contraction twist scales the physical lhs's
    /// materialized core source (else the physical rhs's); meaningful only
    /// when `source_twist_destination_scales` is non-empty.
    pub(in crate::contract) twist_lhs: bool,
    /// The twist as `(twisted destination block offset, θ_b ≠ 1)`, sorted by
    /// offset and without zero-element blocks: the single representation
    /// every executor folds into the move writing block b of the twisted
    /// source transform (Host `tree_transform_structure_overwrite_into_raw`
    /// and member replay, device `replay_with_destination_scales`).
    source_twist_destination_scales: Box<[(usize, C)]>,
    pub(in crate::contract) core_dst: Option<DynamicFusionCoreDstEntry<C>>,
    pub(in crate::contract) block_plan: Arc<FusionBlockContractPlan<C>>,
}

impl<C: DenseBlockScalar> DynamicTreeExecutionArtifact<C> {
    #[cfg(test)]
    pub(crate) fn test_has_core_dst(&self) -> bool {
        self.core_dst.is_some()
    }

    #[cfg(test)]
    pub(crate) fn test_lhs_transform(
        &self,
    ) -> (
        &TreeTransformStructure<C>,
        &Arc<BlockStructure>,
        &Arc<BlockStructure>,
        &DynamicFusionMapSpace,
    ) {
        (
            self.lhs_transform
                .transform_structure
                .as_ref()
                .expect("test artifacts transform their lhs"),
            self.lhs_transform.space.structure(),
            &self.lhs_transform.replay_structure,
            &self.lhs_transform.space,
        )
    }

    #[cfg(test)]
    pub(crate) fn orientation(&self) -> FusionContractOrientation {
        self.orientation
    }

    /// The structure the core GEMMs read for the physical lhs (`true`) or
    /// rhs: a borrowed storage adjoint is read in its parent's storage,
    /// every other source in its core layout.
    pub(crate) fn core_source_structure(&self, lhs: bool) -> &Arc<BlockStructure> {
        let (entry, borrowed) = if lhs {
            (&self.lhs_transform, self.lhs_borrowed)
        } else {
            (&self.rhs_transform, self.rhs_borrowed)
        };
        if borrowed && entry.core_is_storage_adjoint {
            &entry.replay_structure
        } else {
            entry.space.structure()
        }
    }

    pub(crate) fn requires_source_twist(&self) -> bool {
        !self.source_twist_destination_scales.is_empty()
    }

    /// `(lhs, rhs)` in the core GEMM's operand order: swapped for an
    /// `RhsLhs` orientation.
    #[inline]
    pub(crate) fn core_order<T>(&self, lhs: T, rhs: T) -> (T, T) {
        if self.orientation == FusionContractOrientation::RhsLhs {
            (rhs, lhs)
        } else {
            (lhs, rhs)
        }
    }

    /// The physical lhs or rhs value, whichever source the contraction twist
    /// scales; meaningful only when [`Self::requires_source_twist`].
    #[inline]
    pub(crate) fn on_twisted<T>(&self, lhs: T, rhs: T) -> T {
        if self.twist_lhs {
            lhs
        } else {
            rhs
        }
    }

    /// The twist folds of the lhs and rhs source stages: the twisted stage
    /// carries [`Self::source_twist_destination_scales`], the other none. A
    /// twisted source is never borrowed (checked at construction).
    pub(crate) fn stage_scales(&self) -> [&[(usize, C)]; 2] {
        let scales = &self.source_twist_destination_scales[..];
        let (lhs, rhs) = self.on_twisted((scales, &[][..]), (&[][..], scales));
        [lhs, rhs]
    }

    /// Whether the twist, if any, scales the physical lhs's core source.
    #[cfg(test)]
    pub(crate) fn twists_lhs(&self) -> bool {
        self.twist_lhs
    }

    #[cfg(test)]
    pub(crate) fn source_twist_destination_scales(&self) -> &[(usize, C)] {
        &self.source_twist_destination_scales
    }

    pub(crate) fn block_plan(&self) -> &FusionBlockContractPlan<C> {
        &self.block_plan
    }

    pub(crate) fn block_plan_is_fully_direct(&self) -> bool {
        self.block_plan.is_fully_direct()
    }

    /// The core plan's inactive destination blocks when the core GEMMs write
    /// the caller's destination directly (identity output); `None` when an
    /// output transform writes it.
    #[cfg(test)]
    pub(crate) fn direct_destination_inactive_blocks(&self) -> Option<usize> {
        self.core_dst
            .is_none()
            .then(|| self.block_plan.inactive_destination_regions().len())
    }

    #[cfg(test)]
    pub(crate) fn borrowed_sources(&self) -> (bool, bool) {
        (self.lhs_borrowed, self.rhs_borrowed)
    }

    /// The destination space of the physical lhs (`true`) or rhs source
    /// transform, whichever orientation the artifact has.
    #[cfg(test)]
    pub(crate) fn physical_core_space(&self, lhs: bool) -> &DynamicFusionMapSpace {
        if lhs {
            &self.lhs_transform.space
        } else {
            &self.rhs_transform.space
        }
    }

    #[cfg(test)]
    pub(crate) fn twisted_transform_structure(&self) -> &TreeTransformStructure<C> {
        let entry = if self.twist_lhs {
            &self.lhs_transform
        } else {
            &self.rhs_transform
        };
        entry
            .transform_structure
            .as_ref()
            .expect("a twisted source is transformed")
    }
}

/// The `DynamicTree` artifact of two stored operands in admission mode `M`,
/// derived under `cache`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compile_dynamic_tree_execution_artifact<M, R>(
    engine: &mut dyn Engine,
    cache: &mut M::StructureCache,
    rule: &R,
    authority: M::SpaceAuthority<'_>,
    plan: &FusionContractPlan,
    dst_space: &DynamicFusionMapSpace,
    lhs_space: &DynamicFusionMapSpace,
    lhs_structure: &Arc<BlockStructure>,
    rhs_space: &DynamicFusionMapSpace,
    rhs_structure: &Arc<BlockStructure>,
) -> Result<DynamicTreeExecutionArtifact<M::Scalar>, M::Error>
where
    M: PlanningAlgebra<R>,
    M::Scalar: DenseBlockScalar,
{
    let lhs_transform = M::transformed_source(
        engine,
        cache,
        authority,
        rule,
        lhs_space,
        lhs_structure,
        plan.lhs_transform(),
        plan.lhs_source_conjugate(),
    )?;
    let lhs_layout_borrowable = lhs_transform.core_is_storage_adjoint
        || source_is_borrowable_core_layout(
            lhs_space,
            lhs_structure,
            &lhs_transform.space,
            plan.lhs_transform(),
            plan.lhs_source_conjugate(),
        );
    let rhs_transform = M::transformed_source(
        engine,
        cache,
        authority,
        rule,
        rhs_space,
        rhs_structure,
        plan.rhs_transform(),
        plan.rhs_source_conjugate(),
    )?;
    let rhs_layout_borrowable = rhs_transform.core_is_storage_adjoint
        || source_is_borrowable_core_layout(
            rhs_space,
            rhs_structure,
            &rhs_transform.space,
            plan.rhs_transform(),
            plan.rhs_source_conjugate(),
        );
    let borrowing = resolve_source_borrowing_in::<M, R>(
        rule,
        authority,
        plan,
        &lhs_transform.space,
        &rhs_transform.space,
        lhs_layout_borrowable,
        rhs_layout_borrowable,
    )?;
    finish_dynamic_tree_execution_artifact::<M, R>(
        engine,
        cache,
        rule,
        authority,
        plan,
        dst_space,
        lhs_transform,
        rhs_transform,
        borrowing,
    )
}

/// The multiplicity-free `DynamicTree` artifact over prelowered operands,
/// a lazy adjoint read through its parent's storage.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compile_prelowered_dynamic_tree_execution_artifact<R>(
    engine: &mut dyn Engine,
    planning: &mut TreeTransformPlanning,
    rule: &R,
    layout_primer: LayoutKeyBuilder<R>,
    plan: &FusionContractPlan,
    dst_space: &DynamicFusionMapSpace,
    lhs: &FusionOperandLayout<'_>,
    rhs: &FusionOperandLayout<'_>,
) -> Result<DynamicTreeExecutionArtifact<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols + TreeTransformRuleCacheKey,
    R::Scalar: DenseBlockScalar + MultiplicityFreePlanningScalar,
{
    debug_assert_eq!(
        plan.lhs_source_conjugate(),
        lhs.orientation() == FusionTreePairOrientation::Adjoint
    );
    debug_assert_eq!(
        plan.rhs_source_conjugate(),
        rhs.orientation() == FusionTreePairOrientation::Adjoint
    );
    let lhs_direct = lhs.is_direct();
    let lhs_transform = compile_prelowered_source_transform(
        engine,
        planning,
        rule,
        lhs,
        plan.lhs_transform(),
        layout_primer,
    )?;
    let lhs_layout_borrowable = lhs_transform.core_is_storage_adjoint
        || (lhs_direct
            && source_is_borrowable_core_layout(
                lhs.storage_space(),
                lhs.storage_space().structure(),
                &lhs_transform.space,
                plan.lhs_transform(),
                false,
            ));
    let rhs_direct = rhs.is_direct();
    let rhs_transform = compile_prelowered_source_transform(
        engine,
        planning,
        rule,
        rhs,
        plan.rhs_transform(),
        layout_primer,
    )?;
    let rhs_layout_borrowable = rhs_transform.core_is_storage_adjoint
        || (rhs_direct
            && source_is_borrowable_core_layout(
                rhs.storage_space(),
                rhs.storage_space().structure(),
                &rhs_transform.space,
                plan.rhs_transform(),
                false,
            ));
    let borrowing = resolve_source_borrowing(
        rule,
        plan,
        &lhs_transform.space,
        &rhs_transform.space,
        lhs_layout_borrowable,
        rhs_layout_borrowable,
    )?;
    finish_dynamic_tree_execution_artifact::<MultiplicityFreeAdmissionMode, R>(
        engine,
        planning,
        rule,
        layout_primer,
        plan,
        dst_space,
        lhs_transform,
        rhs_transform,
        borrowing,
    )
}

pub(super) fn compile_prelowered_source_transform<R>(
    engine: &mut dyn Engine,
    planning: &mut TreeTransformPlanning,
    rule: &R,
    source: &FusionOperandLayout<'_>,
    operation: &TreeTransformOperation,
    layout_primer: LayoutKeyBuilder<R>,
) -> Result<DynamicFusionTransformedSourceEntry<R::Scalar>, OperationError>
where
    R: MultiplicityFreeRigidSymbols + TreeTransformRuleCacheKey,
    R::Scalar: MultiplicityFreePlanningScalar,
{
    if source.is_direct() {
        compile_transformed_source(
            engine,
            planning,
            rule,
            source.storage_space(),
            source.storage_space().structure(),
            operation,
            false,
            layout_primer,
        )
    } else {
        compile_transformed_source_oriented(
            engine,
            planning,
            rule,
            source,
            operation,
            layout_primer,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn finish_dynamic_tree_execution_artifact<M, R>(
    engine: &mut dyn Engine,
    cache: &mut M::StructureCache,
    rule: &R,
    authority: M::SpaceAuthority<'_>,
    plan: &FusionContractPlan,
    dst_space: &DynamicFusionMapSpace,
    lhs_transform: DynamicFusionTransformedSourceEntry<M::Scalar>,
    rhs_transform: DynamicFusionTransformedSourceEntry<M::Scalar>,
    mut borrowing: SourceBorrowing,
) -> Result<DynamicTreeExecutionArtifact<M::Scalar>, M::Error>
where
    M: PlanningAlgebra<R>,
    M::Scalar: DenseBlockScalar,
{
    let physical_lhs_core_space = lhs_transform.space.clone();
    let physical_rhs_core_space = rhs_transform.space.clone();
    let reverse = plan.orientation() == FusionContractOrientation::RhsLhs;
    let (core_left_space, core_right_space) = if reverse {
        (&physical_rhs_core_space, &physical_lhs_core_space)
    } else {
        (&physical_lhs_core_space, &physical_rhs_core_space)
    };
    let twisted_transform = if borrowing.twist_lhs {
        &lhs_transform
    } else {
        &rhs_transform
    };
    let source_twist_destination_scales = if plan.supertrace_twist() {
        M::contract_twist_scales(
            rule,
            authority,
            &twisted_transform.space,
            core_right_space.homspace(),
            borrowing.twist_lhs != reverse,
            plan.core_axes().as_spec().rhs_contracting_axes(),
        )?
    } else {
        Vec::new()
    };
    if !source_twist_destination_scales.is_empty() {
        let Some(structure) = &twisted_transform.transform_structure else {
            return Err(OperationError::InvalidArgument {
                message: "contraction twist must scale an owned source",
            }
            .into());
        };
        validate_uniform_multi_scales(structure, &source_twist_destination_scales)?;
    }
    let core_dst = if plan.output_transform_is_identity() {
        None
    } else {
        Some(M::core_destination(
            engine,
            cache,
            authority,
            rule,
            core_left_space,
            core_right_space,
            plan,
            dst_space,
        )?)
    };
    let block_dst_space = core_dst
        .as_ref()
        .map_or(dst_space, |entry| entry.space.as_ref());
    let block_plan = match storage_adjoint_core_plan(
        block_dst_space,
        plan.orientation(),
        (&lhs_transform, borrowing.lhs_borrowed),
        (&rhs_transform, borrowing.rhs_borrowed),
    )? {
        Some(block_plan) => Arc::new(block_plan),
        None => {
            borrowing.lhs_borrowed &= !lhs_transform.core_is_storage_adjoint;
            borrowing.rhs_borrowed &= !rhs_transform.core_is_storage_adjoint;
            M::derived_core_plan(
                rule,
                block_dst_space,
                core_left_space,
                core_right_space,
                plan.core_axes().as_spec(),
            )?
        }
    };
    // Why here, once: every executor reads a source's transformer only when
    // that source is not borrowed (the route view, the CUDA admission).
    for (entry, borrowed) in [
        (&lhs_transform, borrowing.lhs_borrowed),
        (&rhs_transform, borrowing.rhs_borrowed),
    ] {
        if !borrowed && entry.transform_structure.is_none() {
            return Err(OperationError::InvalidArgument {
                message: "a copied contraction source has no transformer",
            }
            .into());
        }
    }
    // Why not check this in every executor: Host, member and device replays
    // all fold the twist into the transform writing the twisted source's
    // owned scratch, and borrowing never picks the twisted side
    // (`resolve_source_borrowing`).
    let twisted_borrowed = if borrowing.twist_lhs {
        borrowing.lhs_borrowed
    } else {
        borrowing.rhs_borrowed
    };
    if !source_twist_destination_scales.is_empty() && twisted_borrowed {
        return Err(OperationError::InvalidArgument {
            message: "contraction twist must scale an owned source",
        }
        .into());
    }
    Ok(DynamicTreeExecutionArtifact {
        orientation: plan.orientation(),
        lhs_transform,
        rhs_transform,
        lhs_borrowed: borrowing.lhs_borrowed,
        rhs_borrowed: borrowing.rhs_borrowed,
        twist_lhs: borrowing.twist_lhs,
        source_twist_destination_scales: source_twist_destination_scales.into_boxed_slice(),
        core_dst,
        block_plan,
    })
}

/// The core plan reading each borrowed storage-adjoint source in its parent's
/// storage with [`MatrixOp::Adjoint`], when every core operand and the core
/// destination are canonical coupled-sector matrices whose trees agree: then
/// `block(A', c) = block(A, c)'` is one strided GEMM operand per coupled
/// sector (TensorKit `blas_contract!` `mul!` over `A'`; QSpace
/// `wbarray<T>::contract` folding `conj` into the `'C'` op). `None` when no
/// side is such a source or the layouts differ; the caller then copies it.
///
/// Why not the irregular packed plan for other layouts: it would pack the
/// parent per group, the same element traffic as the copy it replaces.
fn storage_adjoint_core_plan<C: DenseBlockScalar>(
    dst: &DynamicFusionMapSpace,
    orientation: FusionContractOrientation,
    lhs: (&DynamicFusionTransformedSourceEntry<C>, bool),
    rhs: (&DynamicFusionTransformedSourceEntry<C>, bool),
) -> Result<Option<FusionBlockContractPlan<C>>, OperationError> {
    fn side<C>(
        (entry, borrowed): (&DynamicFusionTransformedSourceEntry<C>, bool),
    ) -> (&Arc<BlockStructure>, usize, MatrixOp) {
        if borrowed && entry.core_is_storage_adjoint {
            // The parent's codomain is the core operand's domain.
            (
                &entry.replay_structure,
                entry.space.nin(),
                MatrixOp::Adjoint,
            )
        } else {
            (
                entry.space.structure(),
                entry.space.nout(),
                MatrixOp::Identity,
            )
        }
    }
    let (lhs, rhs) = (side(lhs), side(rhs));
    if lhs.2 == MatrixOp::Identity && rhs.2 == MatrixOp::Identity {
        return Ok(None);
    }
    let (left, right) = match orientation {
        FusionContractOrientation::LhsRhs => (lhs, rhs),
        FusionContractOrientation::RhsLhs => (rhs, lhs),
    };
    FusionBlockContractPlan::try_from_canonical_coupled_regions_with_ops_generic(
        dst.structure(),
        dst.nout(),
        left.0,
        left.1,
        right.0,
        right.1,
        left.2,
        right.2,
    )
}
