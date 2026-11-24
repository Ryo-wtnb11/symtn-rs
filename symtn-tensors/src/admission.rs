//! Provider and braiding admission shared by every entry that accepts a
//! checked-Generic provider or whose admission depends on the braiding style
//! (#1840).
//!
//! The facade, the network preflight and the tensor engines all route through
//! this one owner, which keeps the checks, their order and the public
//! messages identical everywhere.

use symtn_core::{
    BraidingStyleKind, CheckedGenericFusion, CoreError, FusionStyleKind, RuleIdentity,
};

use crate::{BoundDynamicFusionMapSpace, BraidingOp, OperationError};

pub(crate) const CROSSING_FREE_CONTRACTION_TWIST: OperationError = OperationError::Internal {
    message: "contraction twist asked of a crossing-free request",
};

/// The one braiding admission of contraction and trace, independent of mode
/// and storage. TensorKit requires symmetric braiding for both operations;
/// planar contraction has no braiding admission.
#[doc(hidden)]
pub fn admit_braiding(op: BraidingOp, style: BraidingStyleKind) -> Result<(), OperationError> {
    let admitted = match op {
        BraidingOp::Contraction | BraidingOp::Trace => style.is_symmetric(),
    };
    if admitted {
        return Ok(());
    }
    Err(OperationError::UnsupportedContractionBraiding {
        operation: op,
        style,
    })
}

/// Admits checked-Generic providers by value: the rule identities agree,
/// then every provider style is `GenericFusion`, in that order.
#[doc(hidden)]
pub fn admit_checked_generic_providers(
    expected: &RuleIdentity,
    actual: &RuleIdentity,
    styles: impl IntoIterator<Item = FusionStyleKind>,
) -> Result<(), CoreError> {
    if expected != actual {
        return Err(CoreError::FusionRuleMismatch {
            expected: expected.clone(),
            actual: actual.clone(),
        });
    }
    for actual in styles {
        if actual != FusionStyleKind::Generic {
            return Err(CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Generic,
                actual,
            });
        }
    }
    Ok(())
}

/// Admits a checked-Generic binary operation on two spaces: one rule
/// identity, both providers `GenericFusion`. Returns the left provider.
#[doc(hidden)]
pub fn admit_checked_generic_pair<'a, P>(
    lhs_space: &'a BoundDynamicFusionMapSpace<P>,
    rhs_space: &BoundDynamicFusionMapSpace<P>,
) -> Result<&'a P, CoreError>
where
    P: CheckedGenericFusion,
{
    let (lhs, rhs) = (lhs_space.provider(), rhs_space.provider());
    admit_checked_generic_providers(
        &lhs.rule_identity(),
        &rhs.rule_identity(),
        [lhs.fusion_style(), rhs.fusion_style()],
    )?;
    Ok(lhs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn braiding_admission_table() {
        use BraidingOp::{Contraction, Trace};
        use BraidingStyleKind::{Anyonic, Bosonic, Fermionic, NoBraiding};
        // Purpose: ordinary contraction and trace share TensorKit's admission table.
        for op in [Contraction, Trace] {
            for (style, admitted) in [
                (Bosonic, true),
                (Fermionic, true),
                (Anyonic, false),
                (NoBraiding, false),
            ] {
                let expected = if admitted {
                    Ok(())
                } else {
                    Err(OperationError::UnsupportedContractionBraiding {
                        operation: op,
                        style,
                    })
                };
                assert_eq!(admit_braiding(op, style), expected, "{op:?} {style:?}");
            }
        }
    }

    #[test]
    fn checked_generic_admission_checks_identity_before_style() {
        let a = RuleIdentity::of_type::<u8>();
        let b = RuleIdentity::of_type::<u16>();
        assert_eq!(
            admit_checked_generic_providers(&a, &b, [FusionStyleKind::Unique]),
            Err(CoreError::FusionRuleMismatch {
                expected: a.clone(),
                actual: b
            })
        );
        assert_eq!(
            admit_checked_generic_providers(
                &a,
                &a,
                [FusionStyleKind::Generic, FusionStyleKind::Simple]
            ),
            Err(CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Generic,
                actual: FusionStyleKind::Simple
            })
        );
        assert_eq!(
            admit_checked_generic_providers(&a, &a, [FusionStyleKind::Generic; 2]),
            Ok(())
        );
    }
}
