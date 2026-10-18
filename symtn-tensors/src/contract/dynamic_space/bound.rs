use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock};

#[cfg(all(test, feature = "cgc-gen"))]
use symtn_core::BlockStructure;
use symtn_core::{
    CheckedFusionAlgebra, CheckedGenericFusion, CheckedGenericStructureError, CoreError,
    FusionRule, FusionSpaceAdmission, FusionStyleKind, FusionTreeHomSpace, FusionTreePairKey,
    MultiplicityFreeFusionRule, MultiplicityFreeRigidSymbols, PreparedBlockStructure, RuleIdentity,
    StructurallyValidatedFusionTreeSubset,
};

#[cfg(test)]
use super::metadata::PreparedLayoutKeys;
use super::metadata::{checked_metadata_operation_error, LayoutBuildCapability, LayoutKeyBuilder};
use super::{
    validate_generic_provider_style, BoundDynamicFusionMapSpace, DynamicFusionMapSpace,
    PreparedCheckedGenericDynamicSpace, ValidatedDynamicFusionLayout,
};
use crate::{OperationError, TreeTransformOperation};
use symtn_operations::{OutputAxisOrder, TensorContractSpec};

fn validate_bound_space_invariants(space: &DynamicFusionMapSpace) -> Result<(), OperationError> {
    let expected_nout = space.homspace().codomain().len();
    let expected_nin = space.homspace().domain().len();
    if space.nout() != expected_nout || space.nin() != expected_nin {
        return Err(OperationError::from_core_preserving_context(
            CoreError::FusionSpaceSplitMismatch {
                expected_nout,
                expected_nin,
                actual_nout: space.nout(),
                actual_nin: space.nin(),
            },
        ));
    }
    let expected_rank = expected_nout + expected_nin;
    if space.structure().rank() != expected_rank {
        return Err(OperationError::from_core_preserving_context(
            CoreError::StructureRankMismatch {
                expected: expected_rank,
                actual: space.structure().rank(),
            },
        ));
    }
    Ok(())
}

fn validate_complete_admission(space: &DynamicFusionMapSpace) -> Result<(), OperationError> {
    if matches!(&space.admission, FusionSpaceAdmission::Complete(_)) {
        Ok(())
    } else {
        Err(OperationError::StructureMismatch {
            tensor: "complete fusion layout",
        })
    }
}

impl ValidatedDynamicFusionLayout {
    /// Conservative retained bytes for the complete provider-neutral layout.
    /// Shared Arc descendants are charged per retained owner so a workspace
    /// budget never relies on an external owner keeping them alive.
    #[doc(hidden)]
    pub fn charged_retained_bytes(&self) -> usize {
        let identity_bytes = self
            .0
            .admission
            .rule_identity()
            .map_or(0, RuleIdentity::charged_retained_bytes);
        std::mem::size_of::<Self>()
            .saturating_add(self.0.homspace().charged_retained_bytes())
            .saturating_add(self.0.structure().charged_retained_bytes())
            .saturating_add(identity_bytes)
    }

    /// Flat storage length required by this validated layout.
    ///
    /// Why not expose the raw space: executors only need allocation length;
    /// structural access would let consumers rebuild a second authority.
    pub fn required_len(&self) -> Result<usize, CoreError> {
        self.0.required_len()
    }
}

impl PartialEq for ValidatedDynamicFusionLayout {
    fn eq(&self, other: &Self) -> bool {
        self.0.admission.rule_identity() == other.0.admission.rule_identity()
            && self.0.homspace().id() == other.0.homspace().id()
            && self.0.structure() == other.0.structure()
            && self.0.nout() == other.0.nout()
            && self.0.nin() == other.0.nin()
    }
}

impl Eq for ValidatedDynamicFusionLayout {}

impl Hash for ValidatedDynamicFusionLayout {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.admission.rule_identity().hash(state);
        self.0.homspace().id().hash(state);
        self.0.structure().content_key().as_ref().hash(state);
        self.0.nout().hash(state);
        self.0.nin().hash(state);
    }
}

impl<R> Clone for BoundDynamicFusionMapSpace<R> {
    fn clone(&self) -> Self {
        Self {
            space: self.space.clone(),
            provider: Arc::clone(&self.provider),
            layout_build: self.layout_build,
        }
    }
}

impl<R> fmt::Debug for BoundDynamicFusionMapSpace<R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BoundDynamicFusionMapSpace")
            .field("space", &self.space)
            .field("provider_type", &std::any::type_name::<R>())
            .finish_non_exhaustive()
    }
}

#[cfg(all(test, feature = "cgc-gen"))]
impl<R> BoundDynamicFusionMapSpace<R> {
    /// The same binding over a caller-restacked layout of the same trees:
    /// the only way a test obtains a non-canonical checked Generic operand,
    /// since checked admission always derives the canonical layout.
    pub(crate) fn with_test_structure(&self, structure: BlockStructure) -> Self {
        Self {
            space: DynamicFusionMapSpace {
                subblock_structure: structure.into_shared(),
                adjoint: OnceLock::new(),
                ..self.space.clone()
            },
            provider: Arc::clone(&self.provider),
            layout_build: self.layout_build,
        }
    }
}

impl<R> BoundDynamicFusionMapSpace<R> {
    /// Identity this binding admitted from its provider.
    ///
    /// Why not `self.provider.rule_identity()`: `FusionRule::rule_identity`
    /// (and `CheckedGenericFusion`'s, under the same contract) is stable, so
    /// the recorded identity is the provider's; rebuilding a content identity
    /// would only repeat its allocation (#2046).
    fn held_rule_identity(&self) -> &RuleIdentity {
        self.space
            .admission()
            .rule_identity()
            .expect("every bound space carries a Complete admission")
    }

    /// Erases only the provider allocation after preserving the checked layout proof.
    ///
    /// Why not return [`DynamicFusionMapSpace`]: a raw value does not carry the
    /// complete-tree-grid proof established by the bound constructor.
    pub fn validated_layout(&self) -> ValidatedDynamicFusionLayout {
        let mut space = self.space.clone();
        // Why a fresh slot rather than charging the memo: a clone shares the
        // filled `Arc<AdjointMemo>`, whose block structure can still fill after
        // the layout was charged, so the charge would go stale. The parked
        // layout never adjoints itself and a rebound space refills on demand.
        space.adjoint = OnceLock::new();
        ValidatedDynamicFusionLayout(space)
    }

    /// Rebinds a validated cached layout to this space's exact provider allocation.
    ///
    /// Why not retain the provider that first populated a process-global cache:
    /// semantically equal callers may carry distinct provider allocations.
    pub fn rebind_validated(
        &self,
        layout: &ValidatedDynamicFusionLayout,
    ) -> Result<Self, OperationError> {
        let expected = self.held_rule_identity();
        let actual = layout.0.admission.rule_identity().ok_or_else(|| {
            OperationError::from_core_preserving_context(CoreError::MissingFusionRuleIdentity)
        })?;
        if expected != actual {
            return Err(OperationError::from_core_preserving_context(
                CoreError::FusionRuleMismatch {
                    expected: expected.clone(),
                    actual: actual.clone(),
                },
            ));
        }
        validate_complete_admission(&layout.0)?;
        Ok(Self {
            space: layout.0.clone(),
            provider: Arc::clone(&self.provider),
            layout_build: self.layout_build,
        })
    }

    #[inline]
    /// Read-only access to the validated dynamic layout for expert planning
    /// and diagnostics. The provider remains attached to this binding.
    pub fn space(&self) -> &DynamicFusionMapSpace {
        &self.space
    }

    #[inline]
    pub fn provider(&self) -> &R {
        self.provider.as_ref()
    }

    #[inline]
    pub fn provider_arc(&self) -> &Arc<R> {
        &self.provider
    }
}

impl<R> BoundDynamicFusionMapSpace<R>
where
    R: CheckedGenericFusion,
{
    fn validate_checked_generic_style(
        provider: &R,
    ) -> Result<(), CheckedGenericStructureError<R::Error>> {
        let actual = provider.fusion_style();
        if actual == FusionStyleKind::Generic {
            Ok(())
        } else {
            Err(CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Generic,
                actual,
            }
            .into())
        }
    }

    /// Builds a complete checked Generic root and retains its exact provider allocation.
    ///
    /// Equal [`RuleIdentity`] values denote one logically immutable
    /// catalog/gauge authority; internal warm-cache state may differ without
    /// changing any structural or symbol answer.
    pub fn from_final_homspace_generic_checked(
        provider: Arc<R>,
        homspace: FusionTreeHomSpace,
    ) -> Result<Self, CheckedGenericStructureError<R::Error>> {
        Self::from_final_homspace_generic_checked_with(provider, || Ok(homspace))
    }

    /// Builds a checked root after deriving its HomSpace within the same reset epoch.
    #[doc(hidden)]
    pub fn from_final_homspace_generic_checked_with<E>(
        provider: Arc<R>,
        build_homspace: impl FnOnce() -> Result<FusionTreeHomSpace, E>,
    ) -> Result<Self, E>
    where
        E: From<CheckedGenericStructureError<R::Error>>,
    {
        let root_identity = std::cell::OnceCell::new();
        let structure =
            FusionTreeHomSpace::prepare_complete_coupled_subblock_structure_generic_checked_with::<
                R,
                E,
                _,
            >(provider.as_ref(), |identity| {
                let homspace = build_homspace()?;
                Self::validate_checked_generic_style(provider.as_ref())?;
                root_identity
                    .set(identity.clone())
                    .expect("checked root builder runs once");
                Ok(homspace)
            })?;
        let identity = root_identity
            .into_inner()
            .expect("successful checked root builder records identity");
        let (homspace, subblock_structure) = structure.commit_with_complete_homspace();
        let homspace = homspace.expect("complete checked builder retains its HomSpace");
        let nout = homspace.codomain().len();
        let nin = homspace.domain().len();
        Ok(Self {
            space: DynamicFusionMapSpace {
                nout,
                nin,
                homspace: Arc::new(homspace),
                subblock_structure,
                admission: FusionSpaceAdmission::Complete(identity),
                adjoint: OnceLock::new(),
            },
            provider,
            layout_build: LayoutBuildCapability::CheckedGeneric,
        })
    }

    /// Commits a structure that `homspace` already enumerated under `provider`
    /// as a complete checked Generic root, without enumerating it again. The
    /// style check runs here because the checked enumeration does not perform
    /// it; the root constructor above checks before enumerating instead.
    ///
    /// Why not the existing prepare/commit pairs: they either require a
    /// `CheckedGeneric`-bound input space or hand the result the input's
    /// capability, so a caller that only holds the provider could not obtain
    /// the `CheckedGeneric` binding `from_final_homspace_generic_checked` gives.
    #[doc(hidden)]
    pub fn from_prepared_final_homspace_generic_checked(
        provider: Arc<R>,
        homspace: FusionTreeHomSpace,
        structure: PreparedBlockStructure,
    ) -> Result<Self, CheckedGenericStructureError<R::Error>> {
        Self::validate_checked_generic_style(provider.as_ref())?;
        Ok(Self::commit_prepared_generic_checked(
            provider, homspace, structure,
        ))
    }

    fn commit_prepared_generic_checked(
        provider: Arc<R>,
        homspace: FusionTreeHomSpace,
        structure: PreparedBlockStructure,
    ) -> Self {
        let identity = provider.rule_identity();
        let nout = homspace.codomain().len();
        let nin = homspace.domain().len();
        Self {
            space: PreparedCheckedGenericDynamicSpace {
                nout,
                nin,
                homspace,
                structure,
                identity,
            }
            .commit(),
            provider,
            layout_build: LayoutBuildCapability::CheckedGeneric,
        }
    }

    /// Stages one checked Generic final HomSpace without publishing its layout.
    #[doc(hidden)]
    pub fn prepare_final_homspace_generic_with_checked<P>(
        &self,
        provider: &P,
        homspace: FusionTreeHomSpace,
    ) -> Result<PreparedCheckedGenericDynamicSpace, CheckedGenericStructureError<P::Error>>
    where
        P: CheckedGenericFusion,
    {
        self.prepare_final_homspace_generic_from_checked(provider, || Ok(homspace))
    }

    pub(crate) fn prepare_final_homspace_generic_from_checked<P, E>(
        &self,
        provider: &P,
        build_homspace: impl FnOnce() -> Result<FusionTreeHomSpace, E>,
    ) -> Result<PreparedCheckedGenericDynamicSpace, E>
    where
        P: CheckedGenericFusion,
        E: From<CheckedGenericStructureError<P::Error>>,
    {
        let actual = std::cell::OnceCell::new();
        let destination = std::cell::OnceCell::new();
        let structure =
            FusionTreeHomSpace::prepare_complete_coupled_subblock_structure_generic_checked_with::<
                P,
                E,
                _,
            >(provider, |provider_identity| {
                let homspace = build_homspace()?;
                self.admit_checked_generic_derivation(provider, provider_identity)?;
                actual
                    .set(provider_identity.clone())
                    .expect("checked admission runs exactly once");
                destination
                    .set(homspace.clone())
                    .expect("destination is derived once");
                Ok(homspace)
            })?;
        let homspace = destination
            .into_inner()
            .expect("successful derivation records destination");
        Ok(PreparedCheckedGenericDynamicSpace {
            nout: homspace.codomain().len(),
            nin: homspace.domain().len(),
            homspace,
            structure,
            identity: actual
                .into_inner()
                .expect("successful checked admission records provider identity"),
        })
    }

    /// Admits a checked Generic space derived under this binding: a checked
    /// binding, `provider_identity` its held identity, `provider` Generic.
    pub(crate) fn admit_checked_generic_derivation<P>(
        &self,
        provider: &P,
        provider_identity: &RuleIdentity,
    ) -> Result<(), CheckedGenericStructureError<P::Error>>
    where
        P: CheckedGenericFusion,
    {
        if !matches!(self.layout_build, LayoutBuildCapability::CheckedGeneric) {
            return Err(CoreError::MalformedFusionTree {
                message: "checked Generic preparation requires a checked provider binding",
            }
            .into());
        }
        crate::admission::admit_checked_generic_providers(
            self.held_rule_identity(),
            provider_identity,
            [provider.fusion_style()],
        )?;
        Ok(())
    }

    pub(crate) fn validate_prepared_final_homspace_generic_checked(
        &self,
        prepared: &PreparedCheckedGenericDynamicSpace,
    ) -> Result<(), OperationError> {
        if !matches!(self.layout_build, LayoutBuildCapability::CheckedGeneric) {
            return Err(OperationError::StructureMismatch {
                tensor: "checked Generic provider binding",
            });
        }
        crate::admission::admit_checked_generic_providers(
            &prepared.identity,
            self.held_rule_identity(),
            [self.provider.fusion_style()],
        )
        .map_err(OperationError::from_core_preserving_context)?;
        Ok(())
    }

    /// Commits a checked Generic final HomSpace under the source provider allocation.
    #[doc(hidden)]
    pub fn commit_final_homspace_generic_bound_checked(
        &self,
        prepared: PreparedCheckedGenericDynamicSpace,
    ) -> Result<Self, OperationError> {
        self.validate_prepared_final_homspace_generic_checked(&prepared)?;
        Ok(Self {
            space: prepared.commit(),
            provider: Arc::clone(&self.provider),
            layout_build: LayoutBuildCapability::CheckedGeneric,
        })
    }
}

impl<R> BoundDynamicFusionMapSpace<R>
where
    R: FusionRule,
{
    fn from_derived_with_capability(
        provider: Arc<R>,
        space: DynamicFusionMapSpace,
        layout_build: LayoutBuildCapability<R>,
    ) -> Result<Self, OperationError> {
        // Why not enumerate the tree grid again: callers in this crate create
        // `space` through checked structural operations from an already-bound
        // source. Re-enumeration would duplicate that work without adding a
        // new trust boundary; the rule identity remains cheap to verify.
        validate_bound_space_invariants(&space)?;
        space.validate_rule(provider.as_ref())?;
        validate_complete_admission(&space)?;
        Ok(Self {
            space,
            provider,
            layout_build,
        })
    }

    pub(crate) fn from_derived(
        provider: Arc<R>,
        space: DynamicFusionMapSpace,
    ) -> Result<Self, OperationError> {
        Self::from_derived_with_capability(provider, space, LayoutBuildCapability::encoded())
    }

    pub(crate) fn from_derived_like(
        source: &Self,
        space: DynamicFusionMapSpace,
    ) -> Result<Self, OperationError> {
        Self::from_derived_with_capability(Arc::clone(&source.provider), space, source.layout_build)
    }

    /// Expert compatibility root from caller-supplied fusion-tree subblock
    /// shapes, bound to a multiplicity-free provider in one checked pass.
    pub fn from_degeneracy_shapes<Shapes>(
        provider: Arc<R>,
        homspace: FusionTreeHomSpace,
        shapes: Shapes,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols,
        Shapes: IntoIterator,
        Shapes::Item: Into<Vec<usize>>,
    {
        let space =
            DynamicFusionMapSpace::from_degeneracy_shapes(provider.as_ref(), homspace, shapes)?;
        Self::from_derived(provider, space)
    }

    /// Test-only shape-admission bridge using lowered metadata (#586
    /// demotion: external callers use the public final-homspace installer).
    #[cfg(test)]
    pub(crate) fn from_degeneracy_shapes_lowered<Shapes>(
        provider: Arc<R>,
        homspace: FusionTreeHomSpace,
        shapes: Shapes,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols + CheckedFusionAlgebra,
        Shapes: IntoIterator,
        Shapes::Item: Into<Vec<usize>>,
    {
        let layout_build = LayoutBuildCapability::checked();
        let space = DynamicFusionMapSpace::from_degeneracy_shapes_with_key_builder(
            provider.as_ref(),
            homspace,
            shapes,
            move |rule, homspace| layout_build.prepare(rule, homspace),
        )?;
        Self::from_derived_with_capability(provider, space, layout_build)
    }

    /// Expert compatibility root from caller-supplied fusion-tree subblock
    /// shapes, bound to a multiplicity-aware provider in one checked pass.
    pub fn from_degeneracy_shapes_generic<Shapes>(
        provider: Arc<R>,
        homspace: FusionTreeHomSpace,
        shapes: Shapes,
    ) -> Result<Self, OperationError>
    where
        Shapes: IntoIterator,
        Shapes::Item: Into<Vec<usize>>,
    {
        validate_generic_provider_style(provider.as_ref())?;
        let space = DynamicFusionMapSpace::from_degeneracy_shapes_generic(
            provider.as_ref(),
            homspace,
            shapes,
        )?;
        Self::from_derived(provider, space)
    }

    fn bind_subset_with_keys(
        mut space: DynamicFusionMapSpace,
        provider: Arc<R>,
        keys: Vec<FusionTreePairKey>,
    ) -> Result<Self, OperationError> {
        debug_assert!(matches!(&space.admission, FusionSpaceAdmission::Subset(_)));
        validate_bound_space_invariants(&space)?;
        space.validate_complete_tree_grid(&keys)?;
        space.admission = FusionSpaceAdmission::Complete(provider.rule_identity());
        Ok(Self {
            space,
            provider,
            layout_build: LayoutBuildCapability::encoded(),
        })
    }

    /// Expert admission boundary for an already-constructed multiplicity-free
    /// dynamic layout.
    pub fn bind_multiplicity_free(
        space: DynamicFusionMapSpace,
        provider: Arc<R>,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeFusionRule,
    {
        space.validate_rule(provider.as_ref())?;
        if matches!(&space.admission, FusionSpaceAdmission::Complete(_)) {
            return Self::from_derived(provider, space);
        }
        let keys = space
            .homspace()
            .fusion_tree_keys(provider.as_ref())
            .to_vec();
        Self::bind_subset_with_keys(space, provider, keys)
    }

    /// Checked built-in admission using transactional lowered tree metadata.
    ///
    /// Matching unchecked Subset and Complete stamps are revalidated; publication
    /// occurs only after structural, algebraic, and complete-grid proofs pass.
    #[cfg(test)]
    pub(crate) fn bind_multiplicity_free_lowered(
        mut space: DynamicFusionMapSpace,
        provider: Arc<R>,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeFusionRule + CheckedFusionAlgebra,
    {
        space.validate_rule(provider.as_ref())?;
        validate_bound_space_invariants(&space)?;
        let proof =
            StructurallyValidatedFusionTreeSubset::try_new(space.homspace(), space.structure())
                .map_err(OperationError::from_core_preserving_context)?;
        proof
            .validate_for_rule_checked(provider.as_ref())
            .map_err(checked_metadata_operation_error)?;
        let prepared = proof
            .homspace()
            .prepare_fusion_tree_layout_checked(provider.as_ref())
            .map_err(|error| OperationError::FusionAlgebra(Box::new(error)))?;
        space.validate_complete_tree_grid(prepared.keys())?;
        PreparedLayoutKeys::Checked(prepared).commit();
        space.admission = FusionSpaceAdmission::Complete(provider.rule_identity());
        Ok(Self {
            space,
            provider,
            layout_build: LayoutBuildCapability::checked(),
        })
    }

    /// Checked external-provider admission for an already-constructed
    /// multiplicity-free dynamic layout.
    ///
    /// Structural, algebraic, and complete-grid proofs run before any layout
    /// or admission state is published, and the provider only needs to certify
    /// [`CheckedFusionAlgebra`] rather than the sealed built-in lowered codec.
    /// Matching unchecked Subset and Complete stamps are revalidated.
    pub fn bind_multiplicity_free_checked(
        mut space: DynamicFusionMapSpace,
        provider: Arc<R>,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols + CheckedFusionAlgebra,
    {
        space.validate_rule(provider.as_ref())?;
        validate_bound_space_invariants(&space)?;
        // Step 1: structural proof before any provider algebra runs.
        let proof =
            StructurallyValidatedFusionTreeSubset::try_new(space.homspace(), space.structure())
                .map_err(OperationError::from_core_preserving_context)?;
        proof
            .validate_for_rule_checked(provider.as_ref())
            .map_err(checked_metadata_operation_error)?;
        // Step 2: checked sector/tree/layout preparation (validates leg duals).
        let prepared = proof
            .homspace()
            .prepare_fusion_tree_layout_checked(provider.as_ref())
            .map_err(|error| OperationError::FusionAlgebra(Box::new(error)))?;
        // Step 3: complete-grid validation against the caller's storage.
        space.validate_complete_tree_grid(prepared.keys())?;
        // Steps 5-6: publish only after every fallible check has passed.
        prepared.commit();
        space.admission = FusionSpaceAdmission::Complete(provider.rule_identity());
        Ok(Self {
            space,
            provider,
            layout_build: LayoutBuildCapability::checked(),
        })
    }

    /// Builds a checked contraction result while retaining the exact provider
    /// allocation shared by both operands.
    pub fn contracted_multiplicity_free(
        lhs: &Self,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols,
    {
        if lhs.provider.rule_identity() != rhs.provider.rule_identity() {
            return Err(OperationError::from_core_preserving_context(
                CoreError::FusionRuleMismatch {
                    expected: lhs.provider.rule_identity(),
                    actual: rhs.provider.rule_identity(),
                },
            ));
        }
        // The lhs is the authority for a result of two independently-built but
        // semantically identical tensors. Why not require Arc::ptr_eq: public
        // tensors may own distinct provider allocations with one RuleIdentity.
        let axes = TensorContractSpec::with_default_output_order(lhs_axes, rhs_axes);
        let space = DynamicFusionMapSpace::contracted_with_spec_and_primer(
            lhs.provider.as_ref(),
            &lhs.space,
            &rhs.space,
            axes,
            None,
            lhs.layout_build.legacy_dispatch(),
        )?;
        Self::from_derived_like(lhs, space)
    }

    /// Builds a checked contraction result directly in the requested output
    /// order while retaining the exact lhs provider allocation.
    ///
    /// Why not accept [`TensorContractSpec`]: conjugation flags belong to the
    /// numerical execution plan after categorical adjoints have been lowered.
    /// Destination metadata is derived from the already-visible bound spaces.
    pub fn contracted_multiplicity_free_ordered(
        lhs: &Self,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_order: OutputAxisOrder<'_>,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols,
    {
        Self::contracted_multiplicity_free_space(lhs, rhs, lhs_axes, rhs_axes, output_order, None)
    }

    /// [`Self::contracted_multiplicity_free_ordered`] with the result split
    /// after its first `codomain_rank` output axes: TensorOperations
    /// `pAB = (output[..codomain_rank], output[codomain_rank..])`. The space is
    /// the one `permute` gives the default-split result, so a leg moved across
    /// the split is dualized as there.
    pub fn contracted_multiplicity_free_partitioned(
        lhs: &Self,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_order: OutputAxisOrder<'_>,
        codomain_rank: usize,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols,
    {
        Self::contracted_multiplicity_free_space(
            lhs,
            rhs,
            lhs_axes,
            rhs_axes,
            output_order,
            Some(codomain_rank),
        )
    }

    fn contracted_multiplicity_free_space(
        lhs: &Self,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_order: OutputAxisOrder<'_>,
        codomain_rank: Option<usize>,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols,
    {
        Self::validate_shared_provider(lhs, rhs)?;
        let axes = TensorContractSpec::new(lhs_axes, rhs_axes, output_order);
        let space = DynamicFusionMapSpace::contracted_with_spec_and_primer(
            lhs.provider.as_ref(),
            &lhs.space,
            &rhs.space,
            axes,
            codomain_rank,
            lhs.layout_build.legacy_dispatch(),
        )?;
        Self::from_derived_like(lhs, space)
    }

    /// The destination space of a lazy-adjoint-aware contraction, with the
    /// Host's validation order: rule identity, axis counts, axis sets, output
    /// permutation, then the oriented homspace. The one derivation the Host
    /// contraction, the device contraction and the contraction planner's
    /// `CopyC` temporary share. The result is split after `codomain_rank`
    /// output axes, or after every open lhs axis for `None`.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn contracted_multiplicity_free_oriented(
        lhs_authority: &Self,
        lhs: super::FusionOperand<'_>,
        rhs_authority: &Self,
        rhs: super::FusionOperand<'_>,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        output_order: OutputAxisOrder<'_>,
        codomain_rank: Option<usize>,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols + CheckedFusionAlgebra,
    {
        use symtn_core::{FusionTreePairOrientation, OrientedFusionTreeHomSpace};
        Self::validate_shared_provider(lhs_authority, rhs_authority)?;
        if lhs_axes.len() != rhs_axes.len() {
            return Err(OperationError::ContractAxisCountMismatch {
                lhs: lhs_axes.len(),
                rhs: rhs_axes.len(),
            });
        }
        let lhs_rank = lhs.storage_space().rank();
        let rhs_rank = rhs.storage_space().rank();
        // Keep `TensorContractAxisPlan::compile`'s public error order before any
        // oriented homspace/provider work.
        for (tensor, axes, rank) in [("lhs", lhs_axes, lhs_rank), ("rhs", rhs_axes, rhs_rank)] {
            if symtn_core::axes::validate_axis_subset(axes, rank).is_err() {
                return Err(OperationError::InvalidAxisSet {
                    tensor,
                    axes: axes.to_vec(),
                    rank,
                });
            }
        }
        let lhs_open_rank = lhs_rank - lhs_axes.len();
        let rhs_open_rank = rhs_rank - rhs_axes.len();
        let identity_axes;
        let output_axes = match output_order {
            OutputAxisOrder::Identity => {
                identity_axes = (0..lhs_open_rank + rhs_open_rank).collect::<Vec<_>>();
                identity_axes.as_slice()
            }
            OutputAxisOrder::Axes(axes) => axes,
        };
        if symtn_core::axes::validate_permutation(output_axes, lhs_open_rank + rhs_open_rank)
            .is_err()
        {
            return Err(OperationError::InvalidPermutation {
                axes: output_axes.to_vec(),
                rank: lhs_open_rank + rhs_open_rank,
            });
        }
        let orientation = |operand: super::FusionOperand<'_>| {
            if operand.storage_conjugate() {
                FusionTreePairOrientation::Adjoint
            } else {
                FusionTreePairOrientation::Direct
            }
        };
        let homspace = OrientedFusionTreeHomSpace::try_tensorcontract_homspace_checked(
            lhs_authority.provider(),
            OrientedFusionTreeHomSpace::new(lhs.storage_space().homspace(), orientation(lhs)),
            OrientedFusionTreeHomSpace::new(rhs.storage_space().homspace(), orientation(rhs)),
            lhs_axes,
            rhs_axes,
            output_axes,
            codomain_rank.unwrap_or(lhs_open_rank),
        )
        .map_err(|error| match error {
            symtn_core::CheckedFusionSpaceError::Core(error) => {
                OperationError::from_core_preserving_context(*error)
            }
            symtn_core::CheckedFusionSpaceError::FusionAlgebra(error) => {
                OperationError::FusionAlgebra(error)
            }
            _ => OperationError::InvalidArgument {
                message: "unknown checked fusion metadata error",
            },
        })?;
        lhs_authority.derive_from_final_homspace(homspace)
    }

    fn validate_shared_provider(lhs: &Self, rhs: &Self) -> Result<(), OperationError> {
        if lhs.provider.rule_identity() != rhs.provider.rule_identity() {
            return Err(OperationError::from_core_preserving_context(
                CoreError::FusionRuleMismatch {
                    expected: lhs.provider.rule_identity(),
                    actual: rhs.provider.rule_identity(),
                },
            ));
        }
        Ok(())
    }

    /// Ordinary multiplicity-free root/output layout derived from the final
    /// hom space's stored leg degeneracies, without a caller-supplied per-tree
    /// shape list.
    pub fn from_final_homspace_multiplicity_free(
        provider: Arc<R>,
        homspace: FusionTreeHomSpace,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols,
    {
        let space = DynamicFusionMapSpace::from_final_homspace(provider.as_ref(), homspace)?;
        Self::from_derived(provider, space)
    }

    #[doc(hidden)]
    pub fn from_final_homspace_multiplicity_free_lowered(
        provider: Arc<R>,
        homspace: FusionTreeHomSpace,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols + CheckedFusionAlgebra,
    {
        let layout_build = LayoutBuildCapability::checked();
        let prepared = layout_build.prepare(provider.as_ref(), &homspace)?;
        // Why not route this ordinary constructor through the caller-shape
        // expert API: the final HomSpace already owns every leg degeneracy,
        // and the prepared lowered layout owns the matching tree enumeration.
        let space = DynamicFusionMapSpace::from_final_homspace_with_prepared(
            provider.as_ref(),
            homspace,
            prepared,
        )?;
        Self::from_derived_with_capability(provider, space, layout_build)
    }

    /// Ordinary multiplicity-free root/output layout for an external provider,
    /// derived from the final hom space's stored leg degeneracies.
    ///
    /// The provider certifies only [`CheckedFusionAlgebra`]; the layout is
    /// enumerated, its complete storage grid is built, and the prepared layout
    /// is committed before the [`FusionSpaceAdmission::Complete`] stamp, so a
    /// failure at any checked stage leaves no published layout or admission.
    pub fn from_final_homspace_multiplicity_free_checked(
        provider: Arc<R>,
        homspace: FusionTreeHomSpace,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols + CheckedFusionAlgebra,
    {
        let layout_build = LayoutBuildCapability::checked();
        let prepared = layout_build.prepare(provider.as_ref(), &homspace)?;
        let space = DynamicFusionMapSpace::from_final_homspace_with_prepared(
            provider.as_ref(),
            homspace,
            prepared,
        )?;
        Self::from_derived_with_capability(provider, space, layout_build)
    }

    /// Derives a checked MF root inside its complete-layout publication epoch.
    #[doc(hidden)]
    pub fn from_final_homspace_multiplicity_free_checked_with<E>(
        provider: Arc<R>,
        build_homspace: impl FnOnce() -> Result<FusionTreeHomSpace, E>,
    ) -> Result<Self, E>
    where
        R: MultiplicityFreeRigidSymbols + CheckedFusionAlgebra,
        E: From<symtn_core::FusionAlgebraError> + From<OperationError>,
    {
        let (homspace, prepared) = FusionTreeHomSpace::prepare_fusion_tree_layout_checked_with(
            provider.as_ref(),
            build_homspace,
        )?;
        let space = DynamicFusionMapSpace::from_final_homspace_with_prepared(
            provider.as_ref(),
            homspace,
            super::PreparedLayoutKeys::Checked(prepared),
        )?;
        Self::from_derived_with_capability(provider, space, LayoutBuildCapability::checked())
            .map_err(E::from)
    }

    /// Builds a multiplicity-aware contraction result and normalizes its
    /// authority to the lhs provider allocation.
    pub fn contracted_generic(
        lhs: &Self,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
    ) -> Result<Self, OperationError> {
        validate_generic_provider_style(lhs.provider.as_ref())?;
        if lhs.provider.rule_identity() != rhs.provider.rule_identity() {
            return Err(OperationError::from_core_preserving_context(
                CoreError::FusionRuleMismatch {
                    expected: lhs.provider.rule_identity(),
                    actual: rhs.provider.rule_identity(),
                },
            ));
        }
        let space = DynamicFusionMapSpace::contracted_generic(
            lhs.provider.as_ref(),
            &lhs.space,
            &rhs.space,
            lhs_axes,
            rhs_axes,
        )?;
        Self::from_derived_like(lhs, space)
    }

    /// Ordinary Generic root/output layout derived from the final hom space's
    /// stored leg degeneracies.
    pub fn from_final_homspace_generic(
        provider: Arc<R>,
        homspace: FusionTreeHomSpace,
    ) -> Result<Self, OperationError> {
        let space =
            DynamicFusionMapSpace::from_final_homspace_generic(provider.as_ref(), homspace)?;
        Self::from_derived(provider, space)
    }

    /// Stages one checked Generic final HomSpace without publishing its layout.
    #[doc(hidden)]
    pub fn prepare_final_homspace_generic_checked<P>(
        &self,
        provider: &P,
        homspace: FusionTreeHomSpace,
    ) -> Result<PreparedCheckedGenericDynamicSpace, CheckedGenericStructureError<P::Error>>
    where
        P: CheckedGenericFusion,
    {
        let actual = std::cell::OnceCell::new();
        let nout = homspace.codomain().len();
        let nin = homspace.domain().len();
        let structure = homspace
            .clone()
            .prepare_complete_coupled_subblock_structure_generic_checked_after(
                provider,
                |provider_identity| {
                    crate::admission::admit_checked_generic_providers(
                        self.held_rule_identity(),
                        provider_identity,
                        [self.provider.fusion_style(), provider.fusion_style()],
                    )?;
                    actual
                        .set(provider_identity.clone())
                        .expect("checked admission runs exactly once");
                    Ok(())
                },
            )?;
        Ok(PreparedCheckedGenericDynamicSpace {
            nout,
            nin,
            homspace,
            structure,
            identity: actual
                .into_inner()
                .expect("successful checked admission records provider identity"),
        })
    }

    /// Commits a checked Generic final HomSpace under this exact provider.
    #[doc(hidden)]
    pub fn commit_final_homspace_generic_checked(
        &self,
        prepared: PreparedCheckedGenericDynamicSpace,
    ) -> Result<Self, OperationError> {
        crate::admission::admit_checked_generic_providers(
            &prepared.identity,
            self.held_rule_identity(),
            [self.provider.fusion_style()],
        )
        .map_err(OperationError::from_core_preserving_context)?;
        let space = prepared.commit();
        Self::from_derived_with_capability(Arc::clone(&self.provider), space, self.layout_build)
    }

    /// Tree-transform result retaining the source provider proof.
    pub fn transformed_multiplicity_free(
        &self,
        operation: &TreeTransformOperation,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols,
    {
        let space = self.space.transformed_with_primer(
            self.provider.as_ref(),
            operation,
            self.layout_build.legacy_dispatch(),
        )?;
        Self::from_derived_like(self, space)
    }

    /// Generic tree-transform result retaining the source provider proof.
    pub fn transformed_generic(
        &self,
        operation: &TreeTransformOperation,
    ) -> Result<Self, OperationError> {
        validate_generic_provider_style(self.provider.as_ref())?;
        let space = self
            .space
            .transformed_generic(self.provider.as_ref(), operation)?;
        Self::from_derived_like(self, space)
    }

    /// Creates the zero-copy adjoint lowering/replay view while retaining the
    /// exact provider allocation of the source binding.
    ///
    /// This preserves the source blocks' offsets and custom strides while
    /// swapping their logical codomain/domain axes. It is an expert view
    /// capability used by lazy lowering, not an owned ordinary adjoint output
    /// layout.
    pub fn adjoint_view(&self) -> Result<Self, OperationError> {
        let space = self.space.adjoint_view()?;
        Self::from_derived_like(self, space)
    }

    /// Expert admission boundary for an already-constructed
    /// multiplicity-aware dynamic layout.
    pub fn bind_generic(
        space: DynamicFusionMapSpace,
        provider: Arc<R>,
    ) -> Result<Self, OperationError> {
        space.validate_rule(provider.as_ref())?;
        validate_generic_provider_style(provider.as_ref())?;
        if matches!(&space.admission, FusionSpaceAdmission::Complete(_)) {
            return Self::from_derived(provider, space);
        }
        let keys = space
            .homspace()
            .fusion_tree_keys_generic(provider.as_ref())
            .map_err(OperationError::from_core_preserving_context)?;
        Self::bind_subset_with_keys(space, provider, keys)
    }

    /// Whether codomain and domain have the same reduced dimension in every
    /// coupled sector under this bound provider.
    pub fn codomain_isomorphic_to_domain(&self) -> Result<bool, OperationError> {
        let homspace = self.space.homspace();
        if homspace.codomain() == homspace.domain() {
            return Ok(true);
        }
        // Why not compare stored regions or total dimension: storage contains
        // only the shared coupled sectors, while isomorphism concerns both
        // complete sector-dimension maps.
        Ok(homspace
            .codomain()
            .coupled_sector_block_dimensions(self.provider.as_ref())
            .map_err(OperationError::from_core_preserving_context)?
            == homspace
                .domain()
                .coupled_sector_block_dimensions(self.provider.as_ref())
                .map_err(OperationError::from_core_preserving_context)?)
    }

    pub(crate) fn layout_primer(&self) -> LayoutKeyBuilder<R> {
        self.layout_build.legacy_dispatch()
    }

    #[cfg(test)]
    pub(crate) fn with_test_layout_primer(mut self, primer: LayoutKeyBuilder<R>) -> Self {
        self.layout_build = LayoutBuildCapability::Legacy(primer);
        self
    }

    /// Primes a derived HomSpace with this binding's opaque build strategy.
    #[doc(hidden)]
    pub fn prime_derived_homspace(
        &self,
        homspace: &FusionTreeHomSpace,
    ) -> Result<(), OperationError> {
        self.layout_build.prime(self.provider.as_ref(), homspace)
    }

    /// Builds caller-defined per-tree shapes from this binding's authoritative
    /// key order, then consumes those same keys for the storage layout.
    #[doc(hidden)]
    pub fn derive_from_fusion_tree_shapes<BuildShapes, Shapes>(
        &self,
        homspace: FusionTreeHomSpace,
        build_shapes: BuildShapes,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols,
        BuildShapes: FnOnce(&[FusionTreePairKey]) -> Result<Shapes, OperationError>,
        Shapes: IntoIterator,
        Shapes::Item: Into<Vec<usize>>,
    {
        let prepared = self
            .layout_build
            .prepare(self.provider.as_ref(), &homspace)?;
        let keys = prepared.keys(self.provider.as_ref(), &homspace);
        let shapes = build_shapes(keys.as_ref())?;
        let space = DynamicFusionMapSpace::from_degeneracy_shapes_with_key_builder(
            self.provider.as_ref(),
            homspace,
            shapes,
            move |_, _| Ok(prepared),
        )?;
        Self::from_derived_like(self, space)
    }

    /// Internal expert shape bridge preserving this binding's build strategy.
    #[doc(hidden)]
    pub fn derive_from_degeneracy_shapes<Shapes>(
        &self,
        homspace: FusionTreeHomSpace,
        shapes: Shapes,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols,
        Shapes: IntoIterator,
        Shapes::Item: Into<Vec<usize>>,
    {
        let layout_build = self.layout_build;
        let space = DynamicFusionMapSpace::from_degeneracy_shapes_with_key_builder(
            self.provider.as_ref(),
            homspace,
            shapes,
            move |rule, homspace| layout_build.prepare(rule, homspace),
        )?;
        Self::from_derived_like(self, space)
    }

    /// Builds an ordinary derived layout from the final hom space's leg
    /// degeneracies.
    #[doc(hidden)]
    pub fn derive_from_final_homspace(
        &self,
        homspace: FusionTreeHomSpace,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols,
    {
        let prepared = self
            .layout_build
            .prepare(self.provider.as_ref(), &homspace)?;
        let space = DynamicFusionMapSpace::from_final_homspace_with_prepared(
            self.provider.as_ref(),
            homspace,
            prepared,
        )?;
        Self::from_derived_like(self, space)
    }
}
