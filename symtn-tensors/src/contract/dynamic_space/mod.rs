use std::sync::{Arc, OnceLock};

use symtn_core::{
    validate_block_storage_injective, BlockKey, BlockStructure, CoreError, FusionRule,
    FusionSpaceAdmission, FusionStyleKind, FusionTreeHomSpace, FusionTreePairKey,
    FusionTreePairOrientation, MultiplicityFreeRigidSymbols, PreparedBlockStructure, RuleIdentity,
};

use crate::{OperationError, TreeTransformOperation};
#[cfg(test)]
use std::fmt;
#[cfg(test)]
use symtn_core::{CheckedFusionAlgebra, MultiplicityFreeFusionRule};
#[cfg(test)]
use symtn_core::{CheckedGenericFusion, CheckedGenericStructureError};
#[cfg(test)]
use symtn_operations::OutputAxisOrder;
use symtn_operations::TensorContractSpec;

mod bound;
mod metadata;
mod operand;

use metadata::LayoutBuildCapability;
#[cfg(test)]
pub(crate) use metadata::{checked_layout_primer, checked_metadata_dispatcher};
pub(crate) use metadata::{
    dispatch_final, dispatch_prepare, encoded_layout_primer, LayoutKeyBuilder, MetadataOutput,
    MetadataRequest, PreparedLayoutKeys,
};
pub use operand::FusionOperand;
pub(crate) use operand::FusionOperandLayout;

use super::fusion::FusionContractPlan;
use super::structure::TensorContractAxisPlan;

pub(crate) struct TransformedLayoutProbe {
    pub(crate) nout: usize,
    pub(crate) homspace: FusionTreeHomSpace,
    pub(crate) required_len: usize,
    pub(crate) source_structure_matches: bool,
}

/// Dynamic-rank fusion space: the expert-layer space handle whose
/// codomain/domain split is a runtime property.
///
/// The dynamic expert entry points (`*_dyn_into`) take it directly together with
/// raw `f64` slices in the coupled-sector matrix layout.
#[derive(Clone, Debug)]
pub struct DynamicFusionMapSpace {
    nout: usize,
    nin: usize,
    homspace: Arc<FusionTreeHomSpace>,
    subblock_structure: Arc<BlockStructure>,
    admission: FusionSpaceAdmission,
    /// The adjoint view's hom space and block structure, each derived on
    /// first use.
    ///
    /// Why not the bounded complete-structure cache or a context cache: this
    /// is immutable data derived only from `nout`, `nin`, `homspace`, and
    /// `subblock_structure`, none of which is reassigned after construction,
    /// so it lives and dies with the space and needs no key, byte bound, or
    /// invalidation. Its cost is one adjoint hom space and block structure per
    /// space that is ever adjointed; a lazy-adjoint contraction fills only
    /// the hom space. It stays out of `PartialEq` and the
    /// layout hash because it is a function of the compared fields; clones
    /// share the filled value. It sits behind an `Arc` so an unused slot adds
    /// only one pointer and the once-state to every space.
    ///
    /// Hazard: a struct update (`Self { subblock_structure, ..other }`) that
    /// replaces `nout`, `nin`, `homspace`, or `subblock_structure` must also
    /// reset this slot with `OnceLock::new()`, or it inherits a stale adjoint.
    adjoint: OnceLock<Arc<AdjointMemo>>,
}

#[derive(Debug)]
struct AdjointMemo {
    homspace: Arc<FusionTreeHomSpace>,
    structure: OnceLock<Result<Arc<BlockStructure>, OperationError>>,
}

impl PartialEq for DynamicFusionMapSpace {
    fn eq(&self, other: &Self) -> bool {
        self.nout == other.nout
            && self.nin == other.nin
            && self.homspace == other.homspace
            && self.subblock_structure == other.subblock_structure
            && self.admission.rule_identity() == other.admission.rule_identity()
    }
}

impl Eq for DynamicFusionMapSpace {}

fn validate_generic_provider_style<R>(rule: &R) -> Result<(), OperationError>
where
    R: FusionRule,
{
    if rule.fusion_style().has_multiplicity() {
        Ok(())
    } else {
        Err(OperationError::from_core_preserving_context(
            CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Generic,
                actual: rule.fusion_style(),
            },
        ))
    }
}

/// A complete dynamic fusion space bound to the provider that defines it.
///
/// Ordinary operation-output layouts are built from their final hom space with
/// [`Self::from_final_homspace_multiplicity_free`] or
/// [`Self::from_final_homspace_generic`]. The `bind_*` methods are expert
/// admission boundaries for an already-constructed dynamic layout. Generic
/// roots require the provider-owned [`FusionStyleKind::Generic`] capability;
/// tree keys do not carry a duplicate binding-mode flag. A missing rule
/// identity is rejected rather than inferred.
pub struct BoundDynamicFusionMapSpace<R> {
    space: DynamicFusionMapSpace,
    provider: Arc<R>,
    /// Opaque layout-key strategy selected at admission and propagated to
    /// derived outputs; product/rule authority remains in `provider`.
    layout_build: LayoutBuildCapability<R>,
}

#[derive(Clone, Debug)]
/// Provider-neutral dynamic layout that has passed a bound space's full validation.
///
/// Why not expose the raw space, its metadata fields, or the first provider:
/// cached consumers must preserve one validation proof without reconstructing
/// identity or retaining an arbitrary provider allocation.
pub struct ValidatedDynamicFusionLayout(DynamicFusionMapSpace);

#[doc(hidden)]
pub struct PreparedCheckedGenericDynamicSpace {
    nout: usize,
    nin: usize,
    homspace: FusionTreeHomSpace,
    structure: PreparedBlockStructure,
    identity: RuleIdentity,
}

impl PreparedCheckedGenericDynamicSpace {
    pub(crate) fn from_complete_parts(
        nout: usize,
        nin: usize,
        homspace: FusionTreeHomSpace,
        structure: PreparedBlockStructure,
        identity: RuleIdentity,
    ) -> Self {
        Self {
            nout,
            nin,
            homspace,
            structure,
            identity,
        }
    }

    #[doc(hidden)]
    pub fn structure(&self) -> &BlockStructure {
        self.structure.structure()
    }

    #[doc(hidden)]
    pub fn required_len(&self) -> usize {
        self.structure.required_len()
    }

    pub(crate) fn shared_structure(&self) -> Arc<BlockStructure> {
        self.structure.shared_structure()
    }

    pub(crate) fn homspace(&self) -> &FusionTreeHomSpace {
        &self.homspace
    }

    pub(crate) fn nout(&self) -> usize {
        self.nout
    }

    /// `space == self.commit()` by value, without committing or allocating:
    /// the fields `DynamicFusionMapSpace`'s `PartialEq` compares.
    pub(crate) fn matches(&self, space: &DynamicFusionMapSpace) -> bool {
        self.nout == space.nout
            && self.nin == space.nin
            && *space.homspace == self.homspace
            && *space.subblock_structure == *self.structure()
            && space.admission.rule_identity() == Some(&self.identity)
    }

    /// The staged space as a planner reads it: its HomSpace, the preview
    /// structure and the admitted identity. Publishes nothing; the planned
    /// route replays over the preview structure, and only a later
    /// [`Self::commit`] publishes the layout (#2063).
    pub(crate) fn preview(&self) -> DynamicFusionMapSpace {
        DynamicFusionMapSpace {
            nout: self.nout,
            nin: self.nin,
            homspace: Arc::new(self.homspace.clone()),
            subblock_structure: self.shared_structure(),
            admission: FusionSpaceAdmission::Complete(self.identity.clone()),
            adjoint: OnceLock::new(),
        }
    }

    /// Commits an intermediate's structure through the cache-2 owner and
    /// returns the committed structure (the resident winner after a race)
    /// with its canonical HomSpace, which keys a destination memo (#2157).
    /// Why no `DynamicFusionMapSpace`: no caller binds an intermediate.
    pub(crate) fn commit_structure(self) -> (Option<FusionTreeHomSpace>, Arc<BlockStructure>) {
        self.structure.commit_with_complete_homspace()
    }

    pub(crate) fn commit(self) -> DynamicFusionMapSpace {
        let (canonical_homspace, subblock_structure) =
            self.structure.commit_with_complete_homspace();
        let homspace = canonical_homspace.unwrap_or(self.homspace);
        DynamicFusionMapSpace {
            nout: self.nout,
            nin: self.nin,
            homspace: Arc::new(homspace),
            subblock_structure,
            admission: FusionSpaceAdmission::Complete(self.identity),
            adjoint: OnceLock::new(),
        }
    }
}

impl DynamicFusionMapSpace {
    /// Root layout of a multiplicity-free hom space: the TensorKit coupled-sector
    /// matrix layout derived from the hom space's stored leg degeneracies,
    /// identical to [`FusionTreeHomSpace::coupled_subblock_structure`].
    pub fn from_final_homspace<R>(
        rule: &R,
        homspace: FusionTreeHomSpace,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols,
    {
        Self::from_final_homspace_with_primer(rule, homspace, encoded_layout_primer::<R>)
    }

    pub(crate) fn from_final_homspace_with_primer<R>(
        rule: &R,
        homspace: FusionTreeHomSpace,
        primer: LayoutKeyBuilder<R>,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols,
    {
        let prepared = dispatch_final(primer, rule, &homspace)?;
        Self::from_final_homspace_with_prepared(rule, homspace, prepared)
    }

    pub(crate) fn from_final_homspace_with_prepared<R>(
        rule: &R,
        homspace: FusionTreeHomSpace,
        prepared: PreparedLayoutKeys,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols,
    {
        #[cfg(test)]
        crate::probe::hit(crate::probe::Event::FinalResultLayoutBuild);
        let staged = match prepared {
            PreparedLayoutKeys::Checked(prepared) => Some(prepared),
            PreparedLayoutKeys::Encoded => None,
            PreparedLayoutKeys::Final(prepared) => {
                let (canonical, subblock_structure) = prepared.commit_with_complete_homspace();
                let homspace = canonical.unwrap_or(homspace);
                return Ok(Self {
                    nout: homspace.codomain().len(),
                    nin: homspace.domain().len(),
                    homspace: Arc::new(homspace),
                    subblock_structure,
                    admission: FusionSpaceAdmission::Complete(rule.rule_identity()),
                    adjoint: OnceLock::new(),
                });
            }
        };
        if let Some(prepared) = staged {
            let nout = homspace.codomain().len();
            let nin = homspace.domain().len();
            let (homspace, subblock_structure) = prepared
                .build_complete_homspace_from_leg_degeneracies(homspace)
                .map_err(OperationError::from_core_preserving_context)?;
            // All fallible final-storage work is complete. Why not commit
            // before building: malformed leg degeneracies or extent overflow
            // must leave process-local layout identity untouched. The commit
            prepared.commit();
            return Ok(Self {
                nout,
                nin,
                homspace: Arc::new(homspace),
                subblock_structure,
                admission: FusionSpaceAdmission::Complete(rule.rule_identity()),
                adjoint: OnceLock::new(),
            });
        }
        let nout = homspace.codomain().len();
        let nin = homspace.domain().len();
        let (homspace, subblock_structure) = homspace
            .canonical_coupled_subblock_structure_from_leg_degeneracies(rule)
            .map_err(OperationError::from_core_preserving_context)?;
        Ok(Self {
            nout,
            nin,
            homspace: Arc::new(homspace),
            subblock_structure,
            admission: FusionSpaceAdmission::Complete(rule.rule_identity()),
            adjoint: OnceLock::new(),
        })
    }

    /// Generic-fusion sibling of [`Self::from_final_homspace`]: the provider
    /// must report [`FusionStyleKind::Generic`].
    pub fn from_final_homspace_generic<R>(
        rule: &R,
        homspace: FusionTreeHomSpace,
    ) -> Result<Self, OperationError>
    where
        R: FusionRule,
    {
        validate_generic_provider_style(rule)?;
        #[cfg(test)]
        crate::probe::hit(crate::probe::Event::FinalResultLayoutBuild);
        let nout = homspace.codomain().len();
        let nin = homspace.domain().len();
        let (homspace, subblock_structure) = homspace
            .canonical_coupled_subblock_structure_from_leg_degeneracies_generic(rule)
            .map_err(OperationError::from_core_preserving_context)?;
        Ok(Self {
            nout,
            nin,
            homspace: Arc::new(homspace),
            subblock_structure,
            admission: FusionSpaceAdmission::Complete(rule.rule_identity()),
            adjoint: OnceLock::new(),
        })
    }

    fn validate_complete_tree_grid(
        &self,
        keys: &[FusionTreePairKey],
    ) -> Result<(), OperationError> {
        let structure = self.structure();
        if structure.block_count() != keys.len() {
            return Err(OperationError::from_core_preserving_context(
                CoreError::BlockCountMismatch {
                    expected: keys.len(),
                    actual: structure.block_count(),
                },
            ));
        }
        for key in keys {
            structure
                .find_block_index_by_key(&BlockKey::FusionTree(key.clone()))
                .ok_or_else(|| {
                    OperationError::from_core_preserving_context(CoreError::MissingBlockKey {
                        key: Box::new(BlockKey::FusionTree(key.clone())),
                    })
                })?;
        }
        Ok(())
    }

    /// Expert constructor for a caller-selected block layout over `homspace`.
    ///
    /// Use this at an import or custom-layout boundary where the
    /// [`BlockStructure`] (gapped, reordered, or a tree subset) has already
    /// been prepared; ordinary operation outputs derive their layout from the
    /// final hom space instead. This checks only that the structure rank is
    /// the hom-space rank and that logical block footprints do not overlap.
    /// The result is unbound: key, sector, duality, and logical-shape
    /// admission is [`Self::try_bind_rule`] or a `BoundDynamicFusionMapSpace`
    /// `bind_*` constructor.
    #[doc(hidden)]
    pub fn try_from_layout(
        homspace: FusionTreeHomSpace,
        structure: BlockStructure,
    ) -> Result<Self, CoreError> {
        let nout = homspace.codomain().len();
        let nin = homspace.domain().len();
        if structure.rank() != nout + nin {
            return Err(CoreError::StructureRankMismatch {
                expected: nout + nin,
                actual: structure.rank(),
            });
        }
        // Why not always run the pairwise footprint check: canonical
        // coupled-sector regions already cover storage exactly once.
        if structure.coupled_sector_regions(nout)?.is_none() {
            validate_block_storage_injective(&structure)?;
        }
        Ok(Self {
            nout,
            nin,
            homspace: Arc::new(homspace),
            subblock_structure: BlockStructure::canonicalize_shared(structure.into_shared()),
            admission: FusionSpaceAdmission::Unbound,
            adjoint: OnceLock::new(),
        })
    }

    /// Expert subset admission of a [`Self::try_from_layout`] layout.
    ///
    /// This certifies every present block against the rule and hom-space split,
    /// sector membership, duality, and logical degeneracy shape. Missing valid
    /// pairs remain structural zeros. It does not rebuild, repack, or otherwise
    /// select the caller's storage order, strides, or offsets. A space already
    /// admitted under the same rule is returned unchanged.
    #[doc(hidden)]
    pub fn try_bind_rule<R: FusionRule>(mut self, rule: &R) -> Result<Self, CoreError> {
        let actual = rule.rule_identity();
        if let Some(expected) = self.admission.rule_identity() {
            if expected != &actual {
                return Err(CoreError::FusionRuleMismatch {
                    expected: expected.clone(),
                    actual,
                });
            }
            return Ok(self);
        }
        self.homspace
            .validate_subblock_structure_subset(rule, &self.subblock_structure)?;
        self.admission = FusionSpaceAdmission::Subset(actual);
        Ok(self)
    }

    /// Space of the tree-transformed (permute / braid / transpose) tensor:
    /// the hom space is permuted and the full tree set of the result is
    /// enumerated (trees the transform coefficients never reach stay as
    /// structural zeros, keeping every coupled sector grid complete).
    #[cfg(test)]
    pub(crate) fn transformed<R>(
        &self,
        rule: &R,
        operation: &TreeTransformOperation,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols,
    {
        self.transformed_with_primer(rule, operation, encoded_layout_primer::<R>)
    }

    pub(crate) fn transformed_with_primer<R>(
        &self,
        rule: &R,
        operation: &TreeTransformOperation,
        primer: LayoutKeyBuilder<R>,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols,
    {
        self.validate_rule(rule)?;
        Self::permuted_memoized(
            rule,
            self,
            FusionTreePairOrientation::Direct,
            self.homspace(),
            operation,
            primer,
        )
    }

    /// The multiplicity-free permutation of `logical`, the `orientation` of
    /// `storage`'s HomSpace: from the permuted memo of `storage`'s entry, or
    /// derived through `primer` and then memoized there.
    ///
    /// Why the memo may skip the primer: the destination is a pure function
    /// of (identity, storage HomSpace, orientation, axes), and only a
    /// derivation that succeeded under this identity published it.
    pub(super) fn permuted_memoized<R>(
        rule: &R,
        storage: &Self,
        orientation: FusionTreePairOrientation,
        logical: &FusionTreeHomSpace,
        operation: &TreeTransformOperation,
        primer: LayoutKeyBuilder<R>,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols,
    {
        let (codomain_axes, domain_axes) = tree_transform_operation_axes(operation);
        let nout = codomain_axes.len();
        let nin = domain_axes.len();
        let identity = storage.admission.rule_identity().ok_or_else(|| {
            OperationError::from_core_preserving_context(CoreError::MissingFusionRuleIdentity)
        })?;
        let ticket = match storage.homspace().memoized_permuted_multiplicity_free(
            storage.structure(),
            identity,
            orientation,
            codomain_axes,
            domain_axes,
        ) {
            Ok((homspace, subblock_structure)) => {
                return Ok(Self {
                    nout,
                    nin,
                    homspace: Arc::new(homspace),
                    subblock_structure,
                    admission: FusionSpaceAdmission::Complete(identity.clone()),
                    adjoint: OnceLock::new(),
                });
            }
            Err(ticket) => ticket,
        };
        let capability = LayoutBuildCapability::KeyBuilder(primer);
        let (homspace, prepared) = capability.permute(rule, logical, codomain_axes, domain_axes)?;
        debug_assert_eq!(nout, homspace.codomain().len());
        debug_assert_eq!(nin, homspace.domain().len());
        // Why not rebuild external source legs and per-tree shape vectors:
        // #256 already carried the authoritative degeneracies into the final
        // HomSpace. The final grouped builder consumes that value directly.
        let destination = Self::from_final_homspace_with_prepared(rule, homspace, prepared)?;
        ticket.publish(destination.homspace(), destination.structure());
        Ok(destination)
    }

    pub(crate) fn transformed_layout_probe<R>(
        &self,
        rule: &R,
        operation: &TreeTransformOperation,
    ) -> Result<TransformedLayoutProbe, OperationError>
    where
        R: MultiplicityFreeRigidSymbols,
    {
        self.validate_rule(rule)?;
        let (codomain_axes, domain_axes) = tree_transform_operation_axes(operation);
        let homspace = self
            .homspace()
            .permute(rule, codomain_axes, domain_axes)
            .map_err(OperationError::from_core_preserving_context)?;
        let (required_len, source_structure_matches) = homspace
            .coupled_subblock_layout_probe_uncached(rule, self.structure())
            .map_err(OperationError::from_core_preserving_context)?;
        Ok(TransformedLayoutProbe {
            nout: codomain_axes.len(),
            homspace,
            required_len,
            source_structure_matches,
        })
    }

    /// Generic-fusion sibling of [`Self::transformed`]: the permuted /
    /// braided / transposed result space, enumerated with multiplicity-aware
    /// keys. Not cached: the Generic path is not on a hot loop.
    pub(crate) fn transformed_generic<R>(
        &self,
        rule: &R,
        operation: &TreeTransformOperation,
    ) -> Result<Self, OperationError>
    where
        R: FusionRule,
    {
        self.validate_rule(rule)?;
        let source = self;
        let (codomain_axes, domain_axes) = tree_transform_operation_axes(operation);
        let nout = codomain_axes.len();
        let nin = domain_axes.len();
        let homspace = source
            .homspace()
            .permute(rule, codomain_axes, domain_axes)
            .map_err(OperationError::from_core_preserving_context)?;
        debug_assert_eq!(nout, homspace.codomain().len());
        debug_assert_eq!(nin, homspace.domain().len());
        Self::from_final_homspace_generic(rule, homspace)
    }

    pub(crate) fn validate_transformed_generic_checked_identity(
        &self,
        actual: &RuleIdentity,
    ) -> Result<RuleIdentity, CoreError> {
        let expected = match &self.admission {
            FusionSpaceAdmission::Complete(identity) => identity.clone(),
            _ => {
                return Err(CoreError::MalformedFusionTree {
                    message: "checked Generic owned transform requires a Complete source layout",
                })
            }
        };
        if &expected != actual {
            return Err(CoreError::FusionRuleMismatch {
                expected,
                actual: actual.clone(),
            });
        }
        Ok(expected)
    }

    #[cfg(all(test, feature = "cgc-gen"))]
    pub(crate) fn prepare_transformed_generic_checked<R>(
        &self,
        rule: &R,
        operation: &TreeTransformOperation,
    ) -> Result<PreparedCheckedGenericDynamicSpace, CheckedGenericStructureError<R::Error>>
    where
        R: CheckedGenericFusion,
    {
        let (codomain_axes, domain_axes) = tree_transform_operation_axes(operation);
        let homspace = std::cell::OnceCell::new();
        let actual = std::cell::OnceCell::new();
        let structure =
            FusionTreeHomSpace::prepare_complete_coupled_subblock_structure_generic_checked_with::<
                R,
                CheckedGenericStructureError<R::Error>,
                _,
            >(rule, |identity| {
                let actual_identity =
                    self.validate_transformed_generic_checked_identity(identity)?;
                if rule.fusion_style() != FusionStyleKind::Generic {
                    return Err(CoreError::UnsupportedFusionStyle {
                        expected: FusionStyleKind::Generic,
                        actual: rule.fusion_style(),
                    }
                    .into());
                }
                let destination = self.homspace().try_permute_generic_checked(
                    rule,
                    codomain_axes,
                    domain_axes,
                )?;
                actual.set(actual_identity).expect("producer runs once");
                homspace
                    .set(destination.clone())
                    .expect("producer runs once");
                Ok(destination)
            })?;
        Ok(PreparedCheckedGenericDynamicSpace {
            nout: codomain_axes.len(),
            nin: domain_axes.len(),
            homspace: homspace
                .into_inner()
                .expect("successful producer records HomSpace"),
            structure,
            identity: actual
                .into_inner()
                .expect("successful producer records identity"),
        })
    }

    /// Space of the contraction result in the default output order (`lhs`
    /// open axes ascending on the codomain side, `rhs` open axes ascending on
    /// the domain side). Mirrors the destination TensorKit's
    /// `tensorcontract!` with default `pAB` writes into.
    #[cfg(test)]
    pub(crate) fn contracted<R>(
        rule: &R,
        lhs: &Self,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols,
    {
        let axes = TensorContractSpec::with_default_output_order(lhs_axes, rhs_axes);
        Self::contracted_with_spec(rule, lhs, rhs, axes)
    }

    #[cfg(test)]
    fn contracted_with_spec<R>(
        rule: &R,
        lhs: &Self,
        rhs: &Self,
        axes: TensorContractSpec<'_>,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols,
    {
        Self::contracted_with_spec_and_primer(
            rule,
            lhs,
            rhs,
            axes,
            None,
            encoded_layout_primer::<R>,
        )
    }

    /// `codomain_rank` is the result's codomain rank, TensorOperations
    /// `length(pAB[1])`; `None` keeps every open lhs axis in the codomain.
    fn contracted_with_spec_and_primer<R>(
        rule: &R,
        lhs: &Self,
        rhs: &Self,
        axes: TensorContractSpec<'_>,
        codomain_rank: Option<usize>,
        primer: LayoutKeyBuilder<R>,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols,
    {
        lhs.validate_rule(rule)?;
        rhs.validate_rule(rule)?;
        if axes.lhs_contracting_axes().len() != axes.rhs_contracting_axes().len() {
            return Err(OperationError::ContractAxisCountMismatch {
                lhs: axes.lhs_contracting_axes().len(),
                rhs: axes.rhs_contracting_axes().len(),
            });
        }
        let nout = lhs
            .rank()
            .checked_sub(axes.lhs_contracting_axes().len())
            .ok_or_else(|| OperationError::RankMismatch {
                expected: axes.lhs_contracting_axes().len(),
                actual: lhs.rank(),
            })?;
        let nin = rhs
            .rank()
            .checked_sub(axes.rhs_contracting_axes().len())
            .ok_or_else(|| OperationError::RankMismatch {
                expected: axes.rhs_contracting_axes().len(),
                actual: rhs.rank(),
            })?;
        let open = nout + nin;
        let axis_plan = TensorContractAxisPlan::compile(lhs.rank(), rhs.rank(), open, axes)?;
        let nout = codomain_rank.unwrap_or(nout);
        let nin = open
            .checked_sub(nout)
            .ok_or(OperationError::InvalidArgument {
                message: "contraction codomain rank exceeds the open rank",
            })?;
        Self::contracted_space_from_plan(rule, lhs, rhs, axes, &axis_plan, nout, nin, primer)
    }

    #[cfg(test)]
    fn validate_contracted_homspace<R>(
        rule: &R,
        lhs: &Self,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols,
    {
        Self::validate_contracted_homspace_with_primer(
            rule,
            lhs,
            rhs,
            lhs_axes,
            rhs_axes,
            encoded_layout_primer::<R>,
        )
    }

    #[cfg(test)]
    fn validate_contracted_homspace_with_primer<R>(
        rule: &R,
        lhs: &Self,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
        primer: LayoutKeyBuilder<R>,
    ) -> Result<(), OperationError>
    where
        R: MultiplicityFreeRigidSymbols,
    {
        lhs.validate_rule(rule)?;
        rhs.validate_rule(rule)?;
        let nout =
            lhs.rank()
                .checked_sub(lhs_axes.len())
                .ok_or_else(|| OperationError::RankMismatch {
                    expected: lhs_axes.len(),
                    actual: lhs.rank(),
                })?;
        let nin =
            rhs.rank()
                .checked_sub(rhs_axes.len())
                .ok_or_else(|| OperationError::RankMismatch {
                    expected: rhs_axes.len(),
                    actual: rhs.rank(),
                })?;
        let axes = TensorContractSpec::with_default_output_order(lhs_axes, rhs_axes);
        let axis_plan = TensorContractAxisPlan::compile(lhs.rank(), rhs.rank(), nout + nin, axes)?;
        Self::contracted_homspace_from_plan(rule, lhs, rhs, axes, &axis_plan, nout, primer)?;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn core_dst<R>(
        rule: &R,
        lhs: &Self,
        rhs: &Self,
        plan: &FusionContractPlan,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols,
    {
        Self::core_dst_with_primer(rule, lhs, rhs, plan, encoded_layout_primer::<R>)
    }

    pub(crate) fn core_dst_with_primer<R>(
        rule: &R,
        lhs: &Self,
        rhs: &Self,
        plan: &FusionContractPlan,
        primer: LayoutKeyBuilder<R>,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols,
    {
        let nout = plan.core_dst_open_lhs_rank();
        let nin = plan.core_dst_open_rhs_rank();
        Self::contracted_space(
            rule,
            lhs,
            rhs,
            plan.core_axes().as_spec(),
            nout,
            nin,
            primer,
        )
    }

    fn contracted_space<R>(
        rule: &R,
        lhs: &Self,
        rhs: &Self,
        axes: TensorContractSpec<'_>,
        nout: usize,
        nin: usize,
        primer: LayoutKeyBuilder<R>,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols,
    {
        let axis_plan = TensorContractAxisPlan::compile(lhs.rank(), rhs.rank(), nout + nin, axes)?;
        Self::contracted_space_from_plan(rule, lhs, rhs, axes, &axis_plan, nout, nin, primer)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the private builder keeps source spaces, compiled axis plan, output split, and layout primer explicit"
    )]
    fn contracted_space_from_plan<R>(
        rule: &R,
        lhs: &Self,
        rhs: &Self,
        axes: TensorContractSpec<'_>,
        axis_plan: &TensorContractAxisPlan,
        nout: usize,
        nin: usize,
        primer: LayoutKeyBuilder<R>,
    ) -> Result<Self, OperationError>
    where
        R: MultiplicityFreeRigidSymbols,
    {
        let (homspace, prepared) =
            Self::contracted_homspace_from_plan(rule, lhs, rhs, axes, axis_plan, nout, primer)?;
        debug_assert_eq!(nout, homspace.codomain().len());
        debug_assert_eq!(nin, homspace.domain().len());
        Self::from_final_homspace_with_prepared(rule, homspace, prepared)
    }

    fn contracted_homspace_from_plan<R>(
        rule: &R,
        lhs: &Self,
        rhs: &Self,
        axes: TensorContractSpec<'_>,
        axis_plan: &TensorContractAxisPlan,
        nout: usize,
        primer: LayoutKeyBuilder<R>,
    ) -> Result<(FusionTreeHomSpace, PreparedLayoutKeys), OperationError>
    where
        R: MultiplicityFreeRigidSymbols,
    {
        LayoutBuildCapability::KeyBuilder(primer).contract(
            rule,
            lhs.homspace(),
            rhs.homspace(),
            axes,
            &axis_plan.output_axes,
            nout,
        )
    }

    /// Generic-fusion sibling of [`Self::contracted`]: the
    /// contraction result space for an outer-multiplicity rule, enumerated with
    /// multiplicity-aware fusion-tree keys (`fusion_tree_keys_generic`). Not
    /// cached: the Generic path is not on a hot loop. The final HomSpace is consumed
    /// directly by the multiplicity-aware single-pass layout builder.
    pub(crate) fn contracted_generic<R>(
        rule: &R,
        lhs: &Self,
        rhs: &Self,
        lhs_axes: &[usize],
        rhs_axes: &[usize],
    ) -> Result<Self, OperationError>
    where
        R: FusionRule,
    {
        lhs.validate_rule(rule)?;
        rhs.validate_rule(rule)?;
        if lhs_axes.len() != rhs_axes.len() {
            return Err(OperationError::ContractAxisCountMismatch {
                lhs: lhs_axes.len(),
                rhs: rhs_axes.len(),
            });
        }
        let nout =
            lhs.rank()
                .checked_sub(lhs_axes.len())
                .ok_or_else(|| OperationError::RankMismatch {
                    expected: lhs_axes.len(),
                    actual: lhs.rank(),
                })?;
        let nin =
            rhs.rank()
                .checked_sub(rhs_axes.len())
                .ok_or_else(|| OperationError::RankMismatch {
                    expected: rhs_axes.len(),
                    actual: rhs.rank(),
                })?;
        let axes = TensorContractSpec::with_default_output_order(lhs_axes, rhs_axes);
        let axis_plan = TensorContractAxisPlan::compile(lhs.rank(), rhs.rank(), nout + nin, axes)?;
        let homspace = FusionTreeHomSpace::tensorcontract_homspace(
            rule,
            lhs.homspace(),
            rhs.homspace(),
            axes.lhs_contracting_axes(),
            axes.rhs_contracting_axes(),
            &axis_plan.output_axes,
            nout,
        )
        .map_err(OperationError::from_core_preserving_context)?;
        debug_assert_eq!(nout, homspace.codomain().len());
        debug_assert_eq!(nin, homspace.domain().len());
        Self::from_final_homspace_generic(rule, homspace)
    }

    /// Adjoint view: codomain and domain swap (spaces and per-block shapes),
    /// no data movement implied. The block layout is a strided view into the
    /// source layout, so this space is for replay bookkeeping, not for
    /// allocating fresh coupled storage.
    pub(crate) fn adjoint_view(&self) -> Result<Self, OperationError> {
        let memo = self.adjoint_memo();
        let homspace = Arc::clone(&memo.homspace);
        let structure = memo
            .structure
            .get_or_init(|| {
                crate::lowering::adjoint_block_structure_view(
                    self.nout,
                    self.nin,
                    &self.subblock_structure,
                )
                .map(Arc::new)
            })
            .clone()?;
        debug_assert_eq!(structure.rank(), self.subblock_structure.rank());
        debug_assert_eq!(
            structure.block_count(),
            self.subblock_structure.block_count()
        );
        Ok(Self {
            nout: self.nin,
            nin: self.nout,
            homspace,
            subblock_structure: structure,
            admission: self.admission.clone(),
            adjoint: OnceLock::new(),
        })
    }

    /// The memoized adjoint hom space; the block structure fills on demand.
    fn adjoint_memo(&self) -> &AdjointMemo {
        self.adjoint.get_or_init(|| {
            Arc::new(AdjointMemo {
                homspace: Arc::new(FusionTreeHomSpace::new(
                    self.homspace.domain().clone(),
                    self.homspace.codomain().clone(),
                )),
                structure: OnceLock::new(),
            })
        })
    }

    /// Number of codomain legs.
    #[inline]
    pub fn nout(&self) -> usize {
        self.nout
    }

    /// Number of domain legs.
    #[inline]
    pub fn nin(&self) -> usize {
        self.nin
    }

    /// Total number of legs.
    #[inline]
    pub fn rank(&self) -> usize {
        self.nout + self.nin
    }

    pub(crate) fn validate_rule<R: FusionRule>(&self, rule: &R) -> Result<(), OperationError> {
        match self.admission.rule_identity() {
            Some(expected) if expected != &rule.rule_identity() => Err(
                OperationError::from_core_preserving_context(CoreError::FusionRuleMismatch {
                    expected: expected.clone(),
                    actual: rule.rule_identity(),
                }),
            ),
            Some(_) => Ok(()),
            None => Err(OperationError::from_core_preserving_context(
                CoreError::MissingFusionRuleIdentity,
            )),
        }
    }

    #[doc(hidden)]
    #[inline]
    pub fn admission(&self) -> &FusionSpaceAdmission {
        &self.admission
    }

    #[inline]
    pub fn homspace(&self) -> &FusionTreeHomSpace {
        &self.homspace
    }

    /// Shared hom-space handle for pointer-identity fast paths in replay
    /// caches.
    pub fn homspace_arc(&self) -> &Arc<FusionTreeHomSpace> {
        &self.homspace
    }

    /// Subblock (fusion-tree) block structure of the coupled storage layout.
    #[inline]
    pub fn structure(&self) -> &Arc<BlockStructure> {
        &self.subblock_structure
    }

    /// Flat storage length this space requires.
    pub fn required_len(&self) -> Result<usize, CoreError> {
        self.subblock_structure.required_len()
    }
}

pub(crate) fn tree_transform_operation_axes(
    operation: &TreeTransformOperation,
) -> (&[usize], &[usize]) {
    (
        operation.codomain_permutation(),
        operation.domain_permutation(),
    )
}

#[cfg(test)]
mod bound_invariant_tests;

#[cfg(test)]
mod checked_metadata_tests;

#[cfg(test)]
mod scratch_cache_tests;

#[cfg(test)]
mod checked_admission_tests;
