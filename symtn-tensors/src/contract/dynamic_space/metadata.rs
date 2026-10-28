use std::sync::Arc;

use symtn_core::{
    CheckedFusionAlgebra, CheckedFusionSpaceError, FusionRule, FusionTreeHomSpace,
    FusionTreePairKey, MultiplicityFreeFusionRule, PreparedBlockStructure,
    PreparedFusionTreeLayout,
};

use crate::OperationError;
use symtn_operations::TensorContractSpec;

// Why not box the checked layout: the value lives on the stack for one
// constructor call and is consumed by value, so boxing only adds an allocation.
#[allow(clippy::large_enum_variant)]
pub(crate) enum PreparedLayoutKeys {
    Encoded,
    /// Checked staging: the one enumeration path, used by every
    /// multiplicity-free provider.
    Checked(PreparedFusionTreeLayout),
    /// Checked staging of a final layout's complete structure, answered by
    /// the resident complete structure when there is one. Only
    /// `DynamicFusionMapSpace::from_final_homspace_with_prepared` consumes it;
    /// key consumers stage through `MetadataRequest::Prepare`.
    Final(PreparedBlockStructure),
}

impl std::fmt::Debug for PreparedLayoutKeys {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Encoded => formatter.write_str("Encoded"),
            Self::Checked(prepared) => formatter.debug_tuple("Checked").field(prepared).finish(),
            Self::Final(_) => formatter.write_str("Final(..)"),
        }
    }
}

impl PreparedLayoutKeys {
    pub(crate) fn keys<R>(
        &self,
        rule: &R,
        homspace: &FusionTreeHomSpace,
    ) -> Arc<[FusionTreePairKey]>
    where
        R: MultiplicityFreeFusionRule,
    {
        match self {
            Self::Checked(prepared) => prepared.keys_arc(),
            Self::Encoded => homspace.fusion_tree_keys(rule),
            // Why not fall back to `fusion_tree_keys`: that is the unchecked
            // enumeration, and a checked provider must not reach it silently.
            Self::Final(_) => unreachable!("a final staging is consumed only by its final build"),
        }
    }

    pub(super) fn commit(self) {
        let prepared = match self {
            Self::Checked(prepared) => prepared,
            Self::Final(_) => unreachable!("a final staging is consumed only by its final build"),
            Self::Encoded => return,
        };
        prepared.commit();
    }
}

pub(crate) enum MetadataRequest<'a> {
    Prepare {
        homspace: &'a FusionTreeHomSpace,
    },
    /// Stages the layout of a final HomSpace; the response is `Prepared`. A
    /// checked dispatcher answers `Final`, which consults the sector
    /// structure only when the complete structure is not resident.
    FinalLayout {
        homspace: &'a FusionTreeHomSpace,
    },
    Permute {
        homspace: &'a FusionTreeHomSpace,
        codomain_axes: &'a [usize],
        domain_axes: &'a [usize],
    },
    Contract {
        lhs: &'a FusionTreeHomSpace,
        rhs: &'a FusionTreeHomSpace,
        lhs_axes: &'a [usize],
        rhs_axes: &'a [usize],
        output_axes: &'a [usize],
        dst_codomain_rank: usize,
    },
    /// Whether the contracted HomSpace equals `expected`, answered without
    /// building it when the legs prove equality.
    ContractHomSpaceMatches {
        lhs: &'a FusionTreeHomSpace,
        rhs: &'a FusionTreeHomSpace,
        lhs_axes: &'a [usize],
        rhs_axes: &'a [usize],
        output_axes: &'a [usize],
        dst_codomain_rank: usize,
        expected: &'a FusionTreeHomSpace,
    },
}

pub(crate) enum MetadataOutput {
    Prepared(PreparedLayoutKeys),
    HomSpace {
        homspace: FusionTreeHomSpace,
        prepared: PreparedLayoutKeys,
    },
    Matches(bool),
}

pub(crate) type LayoutKeyBuilder<R> =
    for<'a> fn(&R, MetadataRequest<'a>) -> Result<MetadataOutput, OperationError>;

pub(super) enum LayoutBuildCapability<R> {
    KeyBuilder(LayoutKeyBuilder<R>),
    CheckedGeneric,
}

impl<R> Copy for LayoutBuildCapability<R> {}

impl<R> Clone for LayoutBuildCapability<R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R> LayoutBuildCapability<R>
where
    R: FusionRule,
{
    pub(super) const fn encoded() -> Self {
        Self::KeyBuilder(encoded_layout_primer::<R>)
    }

    pub(super) fn key_builder(self) -> LayoutKeyBuilder<R> {
        match self {
            Self::KeyBuilder(dispatch) => dispatch,
            Self::CheckedGeneric => {
                unreachable!("checked Generic bindings do not use a layout key builder")
            }
        }
    }

    pub(super) fn prime(
        self,
        rule: &R,
        homspace: &FusionTreeHomSpace,
    ) -> Result<(), OperationError> {
        let prepared = self.prepare(rule, homspace)?;
        prepared.commit();
        Ok(())
    }

    pub(super) fn prepare(
        self,
        rule: &R,
        homspace: &FusionTreeHomSpace,
    ) -> Result<PreparedLayoutKeys, OperationError> {
        match (self.key_builder())(rule, MetadataRequest::Prepare { homspace })? {
            MetadataOutput::Prepared(prepared) => Ok(prepared),
            _ => unreachable!("metadata dispatcher returned a non-prepare response"),
        }
    }

    pub(super) fn permute(
        self,
        rule: &R,
        homspace: &FusionTreeHomSpace,
        codomain_axes: &[usize],
        domain_axes: &[usize],
    ) -> Result<(FusionTreeHomSpace, PreparedLayoutKeys), OperationError> {
        #[cfg(test)]
        crate::probe::hit(crate::probe::Event::DerivedHomspaceBuild);
        match (self.key_builder())(
            rule,
            MetadataRequest::Permute {
                homspace,
                codomain_axes,
                domain_axes,
            },
        )? {
            MetadataOutput::HomSpace { homspace, prepared } => Ok((homspace, prepared)),
            _ => unreachable!("metadata dispatcher returned a non-HomSpace response"),
        }
    }

    pub(super) fn contract(
        self,
        rule: &R,
        lhs: &FusionTreeHomSpace,
        rhs: &FusionTreeHomSpace,
        axes: TensorContractSpec<'_>,
        output_axes: &[usize],
        dst_codomain_rank: usize,
    ) -> Result<(FusionTreeHomSpace, PreparedLayoutKeys), OperationError> {
        #[cfg(test)]
        crate::probe::hit(crate::probe::Event::DerivedHomspaceBuild);
        match (self.key_builder())(
            rule,
            MetadataRequest::Contract {
                lhs,
                rhs,
                lhs_axes: axes.lhs_contracting_axes(),
                rhs_axes: axes.rhs_contracting_axes(),
                output_axes,
                dst_codomain_rank,
            },
        )? {
            MetadataOutput::HomSpace { homspace, prepared } => Ok((homspace, prepared)),
            _ => unreachable!("metadata dispatcher returned a non-HomSpace response"),
        }
    }
}

impl<R> LayoutBuildCapability<R>
where
    R: MultiplicityFreeFusionRule + CheckedFusionAlgebra,
{
    pub(super) const fn checked() -> Self {
        Self::KeyBuilder(checked_metadata_dispatcher::<R>)
    }
}

pub(crate) fn dispatch_prepare<R>(
    dispatcher: LayoutKeyBuilder<R>,
    rule: &R,
    homspace: &FusionTreeHomSpace,
) -> Result<PreparedLayoutKeys, OperationError> {
    match dispatcher(rule, MetadataRequest::Prepare { homspace })? {
        MetadataOutput::Prepared(prepared) => Ok(prepared),
        _ => unreachable!("metadata dispatcher returned a non-prepare response"),
    }
}

pub(crate) fn dispatch_final<R>(
    dispatcher: LayoutKeyBuilder<R>,
    rule: &R,
    homspace: &FusionTreeHomSpace,
) -> Result<PreparedLayoutKeys, OperationError> {
    match dispatcher(rule, MetadataRequest::FinalLayout { homspace })? {
        MetadataOutput::Prepared(prepared) => Ok(prepared),
        _ => unreachable!("metadata dispatcher returned a non-prepare response"),
    }
}

pub(super) fn checked_metadata_operation_error(error: CheckedFusionSpaceError) -> OperationError {
    match error {
        CheckedFusionSpaceError::Core(error) => {
            OperationError::from_core_preserving_context(*error)
        }
        CheckedFusionSpaceError::FusionAlgebra(error) => OperationError::FusionAlgebra(error),
        _ => OperationError::InvalidArgument {
            message: "unknown checked fusion metadata error",
        },
    }
}

pub(crate) fn encoded_layout_primer<R>(
    rule: &R,
    request: MetadataRequest<'_>,
) -> Result<MetadataOutput, OperationError>
where
    R: FusionRule,
{
    let derived = |homspace| MetadataOutput::HomSpace {
        homspace,
        prepared: PreparedLayoutKeys::Encoded,
    };
    match request {
        MetadataRequest::Prepare { .. } | MetadataRequest::FinalLayout { .. } => {
            Ok(MetadataOutput::Prepared(PreparedLayoutKeys::Encoded))
        }
        MetadataRequest::Permute {
            homspace,
            codomain_axes,
            domain_axes,
        } => homspace
            .permute(rule, codomain_axes, domain_axes)
            .map(derived)
            .map_err(OperationError::from_core_preserving_context),
        MetadataRequest::Contract {
            lhs,
            rhs,
            lhs_axes,
            rhs_axes,
            output_axes,
            dst_codomain_rank,
        } => FusionTreeHomSpace::tensorcontract_homspace(
            rule,
            lhs,
            rhs,
            lhs_axes,
            rhs_axes,
            output_axes,
            dst_codomain_rank,
        )
        .map(derived)
        .map_err(OperationError::from_core_preserving_context),
        MetadataRequest::ContractHomSpaceMatches {
            lhs,
            rhs,
            lhs_axes,
            rhs_axes,
            output_axes,
            dst_codomain_rank,
            expected,
        } => FusionTreeHomSpace::tensorcontract_homspace_matches(
            rule,
            lhs,
            rhs,
            lhs_axes,
            rhs_axes,
            output_axes,
            dst_codomain_rank,
            expected,
        )
        .map(MetadataOutput::Matches)
        .map_err(OperationError::from_core_preserving_context),
    }
}

/// Checked metadata dispatch for a multiplicity-free provider that certifies
/// [`CheckedFusionAlgebra`].
pub(crate) fn checked_metadata_dispatcher<R>(
    rule: &R,
    request: MetadataRequest<'_>,
) -> Result<MetadataOutput, OperationError>
where
    R: MultiplicityFreeFusionRule + CheckedFusionAlgebra,
{
    // Why a permute keeps the sector-first staging: its warm derivation is
    // answered by the source's permuted-destination memo, and a layout probe
    // discards this staging, where a final staging would build the degeneracy
    // part for nothing on a miss.
    let permuted = |homspace: FusionTreeHomSpace| {
        let prepared = checked_layout_primer(rule, &homspace)?;
        Ok(MetadataOutput::HomSpace { homspace, prepared })
    };
    let contracted = |homspace: FusionTreeHomSpace| {
        let prepared = checked_final_layout_primer(rule, &homspace)?;
        Ok(MetadataOutput::HomSpace { homspace, prepared })
    };
    match request {
        MetadataRequest::Prepare { homspace } => {
            checked_layout_primer(rule, homspace).map(MetadataOutput::Prepared)
        }
        MetadataRequest::FinalLayout { homspace } => {
            checked_final_layout_primer(rule, homspace).map(MetadataOutput::Prepared)
        }
        MetadataRequest::Permute {
            homspace,
            codomain_axes,
            domain_axes,
        } => homspace
            .try_permute_checked(rule, codomain_axes, domain_axes)
            .map_err(checked_metadata_operation_error)
            .and_then(permuted),
        MetadataRequest::Contract {
            lhs,
            rhs,
            lhs_axes,
            rhs_axes,
            output_axes,
            dst_codomain_rank,
        } => FusionTreeHomSpace::try_tensorcontract_homspace_checked(
            rule,
            lhs,
            rhs,
            lhs_axes,
            rhs_axes,
            output_axes,
            dst_codomain_rank,
        )
        .map_err(checked_metadata_operation_error)
        .and_then(contracted),
        MetadataRequest::ContractHomSpaceMatches {
            lhs,
            rhs,
            lhs_axes,
            rhs_axes,
            output_axes,
            dst_codomain_rank,
            expected,
        } => FusionTreeHomSpace::try_tensorcontract_homspace_matches_checked(
            rule,
            lhs,
            rhs,
            lhs_axes,
            rhs_axes,
            output_axes,
            dst_codomain_rank,
            expected,
        )
        .map(MetadataOutput::Matches)
        .map_err(checked_metadata_operation_error),
    }
}

/// Stages the checked layout of `homspace` for a multiplicity-free rule that
/// certifies [`CheckedFusionAlgebra`].
pub(crate) fn checked_layout_primer<R>(
    rule: &R,
    homspace: &FusionTreeHomSpace,
) -> Result<PreparedLayoutKeys, OperationError>
where
    R: MultiplicityFreeFusionRule + CheckedFusionAlgebra,
{
    homspace
        .prepare_fusion_tree_layout_checked(rule)
        .map(PreparedLayoutKeys::Checked)
        .map_err(|error| OperationError::FusionAlgebra(Box::new(error)))
}

fn checked_final_layout_primer<R>(
    rule: &R,
    homspace: &FusionTreeHomSpace,
) -> Result<PreparedLayoutKeys, OperationError>
where
    R: MultiplicityFreeFusionRule + CheckedFusionAlgebra,
{
    homspace
        .prepare_final_coupled_subblock_structure_checked(rule)
        .map(PreparedLayoutKeys::Final)
        .map_err(checked_metadata_operation_error)
}
