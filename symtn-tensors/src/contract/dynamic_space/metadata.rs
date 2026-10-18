use std::sync::Arc;

use symtn_core::{
    CheckedFusionAlgebra, CheckedFusionSpaceError, FusionRule, FusionTreeHomSpace,
    FusionTreePairKey, MultiplicityFreeFusionRule, PreparedFusionTreeLayout, SectorLeg,
};

use crate::OperationError;
use symtn_operations::TensorContractSpec;

#[derive(Debug)]
pub(crate) enum PreparedLayoutKeys {
    Encoded,
    Staged(PreparedFusionTreeLayout),
    /// Checked staging: the one enumeration path, used by every
    /// multiplicity-free provider.
    Checked(PreparedFusionTreeLayout),
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
            Self::Staged(prepared) | Self::Checked(prepared) => prepared.keys_arc(),
            Self::Encoded => homspace.fusion_tree_keys(rule),
        }
    }

    pub(super) fn commit(self) {
        let prepared = match self {
            Self::Staged(prepared) | Self::Checked(prepared) => prepared,
            Self::Encoded => return,
        };
        prepared.commit();
    }
}

#[allow(dead_code)]
pub(crate) enum MetadataRequest<'a> {
    Prepare {
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
    Select {
        homspace: &'a FusionTreeHomSpace,
        codomain_axes: &'a [usize],
        domain_axes: &'a [usize],
    },
    OutwardLeg {
        homspace: &'a FusionTreeHomSpace,
        axis: usize,
        dualize: bool,
        tensor: &'static str,
    },
}

pub(crate) enum MetadataOutput {
    Prepared(PreparedLayoutKeys),
    HomSpace {
        homspace: FusionTreeHomSpace,
        prepared: PreparedLayoutKeys,
    },
    Matches(bool),
    Leg(SectorLeg),
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

#[allow(dead_code)]
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

    pub(crate) fn select(
        self,
        rule: &R,
        homspace: &FusionTreeHomSpace,
        codomain_axes: &[usize],
        domain_axes: &[usize],
    ) -> Result<(FusionTreeHomSpace, PreparedLayoutKeys), OperationError> {
        match (self.key_builder())(
            rule,
            MetadataRequest::Select {
                homspace,
                codomain_axes,
                domain_axes,
            },
        )? {
            MetadataOutput::HomSpace { homspace, prepared } => Ok((homspace, prepared)),
            _ => unreachable!("metadata dispatcher returned a non-HomSpace response"),
        }
    }

    pub(crate) fn outward_leg(
        self,
        rule: &R,
        homspace: &FusionTreeHomSpace,
        axis: usize,
        dualize: bool,
        tensor: &'static str,
    ) -> Result<SectorLeg, OperationError> {
        match (self.key_builder())(
            rule,
            MetadataRequest::OutwardLeg {
                homspace,
                axis,
                dualize,
                tensor,
            },
        )? {
            MetadataOutput::Leg(leg) => Ok(leg),
            _ => unreachable!("metadata dispatcher returned a non-leg response"),
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
        MetadataRequest::Prepare { .. } => {
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
        MetadataRequest::Select {
            homspace,
            codomain_axes,
            domain_axes,
        } => homspace
            .select(rule, codomain_axes, domain_axes)
            .map(derived)
            .map_err(OperationError::from_core_preserving_context),
        MetadataRequest::OutwardLeg {
            homspace,
            axis,
            dualize,
            tensor,
        } => encoded_outward_leg(rule, homspace, axis, dualize, tensor).map(MetadataOutput::Leg),
    }
}

/// Shared checked-metadata dispatch, parameterized by the layout primer.
///
/// Why not collapse the lowered dispatcher onto the checked one directly:
/// primer provenance differs (the sealed built-in lowered codec vs an
/// external provider that certifies only `CheckedFusionAlgebra`) and so does
/// its error mapping; everything else — checked permute/contract/select
/// homspace derivation, checked outward legs — is common and lives here.
fn checked_metadata_dispatch_with_primer<R>(
    rule: &R,
    request: MetadataRequest<'_>,
    primer: impl Fn(&R, &FusionTreeHomSpace) -> Result<PreparedLayoutKeys, OperationError>,
) -> Result<MetadataOutput, OperationError>
where
    R: CheckedFusionAlgebra,
{
    let prepare = |homspace: FusionTreeHomSpace| {
        let prepared = primer(rule, &homspace)?;
        Ok(MetadataOutput::HomSpace { homspace, prepared })
    };
    match request {
        MetadataRequest::Prepare { homspace } => {
            primer(rule, homspace).map(MetadataOutput::Prepared)
        }
        MetadataRequest::Permute {
            homspace,
            codomain_axes,
            domain_axes,
        } => homspace
            .try_permute_checked(rule, codomain_axes, domain_axes)
            .map_err(checked_metadata_operation_error)
            .and_then(prepare),
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
        .and_then(prepare),
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
        MetadataRequest::Select {
            homspace,
            codomain_axes,
            domain_axes,
        } => homspace
            .try_select_checked(rule, codomain_axes, domain_axes)
            .map_err(checked_metadata_operation_error)
            .and_then(prepare),
        MetadataRequest::OutwardLeg {
            homspace,
            axis,
            dualize,
            tensor,
        } => checked_outward_leg(rule, homspace, axis, dualize, tensor).map(MetadataOutput::Leg),
    }
}

/// Checked external-provider sibling of [`checked_layout_primer`]: stages a
/// complete layout for a rule that certifies only [`CheckedFusionAlgebra`],
/// without requiring the sealed built-in lowered codec.
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

pub(crate) fn checked_metadata_dispatcher<R>(
    rule: &R,
    request: MetadataRequest<'_>,
) -> Result<MetadataOutput, OperationError>
where
    R: MultiplicityFreeFusionRule + CheckedFusionAlgebra,
{
    checked_metadata_dispatch_with_primer(rule, request, checked_layout_primer)
}

fn encoded_outward_leg<R>(
    rule: &R,
    homspace: &FusionTreeHomSpace,
    axis: usize,
    dualize: bool,
    tensor: &'static str,
) -> Result<SectorLeg, OperationError>
where
    R: FusionRule,
{
    let mut leg = if axis < homspace.codomain().len() {
        homspace.codomain().legs()[axis].clone()
    } else if axis < homspace.rank() {
        homspace.domain().legs()[axis - homspace.codomain().len()].dual(rule)
    } else {
        return Err(OperationError::InvalidAxisSet {
            tensor,
            axes: vec![axis],
            rank: homspace.rank(),
        });
    };
    if dualize {
        leg = leg.dual(rule);
    }
    Ok(leg)
}

fn checked_outward_leg<R>(
    rule: &R,
    homspace: &FusionTreeHomSpace,
    axis: usize,
    dualize: bool,
    tensor: &'static str,
) -> Result<SectorLeg, OperationError>
where
    R: CheckedFusionAlgebra,
{
    let mut leg = if axis < homspace.codomain().len() {
        homspace.codomain().legs()[axis].clone()
    } else if axis < homspace.rank() {
        homspace.domain().legs()[axis - homspace.codomain().len()]
            .try_dual(rule)
            .map_err(|error| OperationError::FusionAlgebra(Box::new(error)))?
    } else {
        return Err(OperationError::InvalidAxisSet {
            tensor,
            axes: vec![axis],
            rank: homspace.rank(),
        });
    };
    if dualize {
        leg = leg
            .try_dual(rule)
            .map_err(|error| OperationError::FusionAlgebra(Box::new(error)))?;
    }
    Ok(leg)
}
