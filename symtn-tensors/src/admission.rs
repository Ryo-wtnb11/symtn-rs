//! Provider and braiding admission shared by every entry that accepts a
//! checked-Generic provider or whose admission depends on the braiding style
//! (#1840).
//!
//! The facade, the network preflight and the tensor engines all route through
//! this one owner, which keeps the checks, their order and the public
//! messages identical everywhere.

use symtn_core::{
    BraidingStyleKind, CheckedGenericAdmissionMode, CheckedGenericFusion, CoreError,
    FusionStyleKind, MultiplicityFreeAdmissionMode, RuleIdentity,
};

use crate::{BoundDynamicFusionMapSpace, BraidingOp, OperationError};

/// The checked engine's internal fail-closed refusal (twist and core-alpha
/// invariants after admission).
pub(crate) const CHECKED_CONTRACTION_REQUIRES_BOSONIC: OperationError =
    OperationError::UnsupportedTensorContractScope {
        message: "checked Generic contraction requires Bosonic braiding",
    };

mod sealed {
    pub trait Sealed {}
    impl Sealed for symtn_core::MultiplicityFreeAdmissionMode {}
    impl Sealed for symtn_core::CheckedGenericAdmissionMode {}
}

/// The braiding styles an admission mode admits for each [`BraidingOp`].
#[doc(hidden)]
pub trait BraidingAdmission: sealed::Sealed {
    /// Whether `style` is admitted for `op`.
    fn admits(op: BraidingOp, style: BraidingStyleKind) -> bool;
}

/// TensorKit: ordinary contraction and trace need a symmetric braiding;
/// planar contraction admits every style.
impl BraidingAdmission for MultiplicityFreeAdmissionMode {
    fn admits(op: BraidingOp, style: BraidingStyleKind) -> bool {
        match op {
            BraidingOp::Contraction | BraidingOp::Trace => style.is_symmetric(),
            BraidingOp::PlanarContraction => true,
        }
    }
}

/// The checked engine implements neither TensorKit `blas_contract!`'s
/// Fermionic twist nor a planar route, so ordinary and planar contraction
/// admit Bosonic braiding only; trace follows TensorKit.
impl BraidingAdmission for CheckedGenericAdmissionMode {
    fn admits(op: BraidingOp, style: BraidingStyleKind) -> bool {
        match op {
            BraidingOp::Trace => style.is_symmetric(),
            BraidingOp::Contraction | BraidingOp::PlanarContraction => {
                style == BraidingStyleKind::Bosonic
            }
        }
    }
}

/// The one braiding admission of contraction and trace: the outcome depends
/// only on the mode `M`, `op` and `style`, never on storage form, and a
/// refusal is [`OperationError::UnsupportedContractionBraiding`] carrying
/// both.
#[doc(hidden)]
pub fn admit_braiding<M: BraidingAdmission>(
    op: BraidingOp,
    style: BraidingStyleKind,
) -> Result<(), OperationError> {
    if M::admits(op, style) {
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
        use BraidingOp::{Contraction, PlanarContraction, Trace};
        use BraidingStyleKind::{Anyonic, Bosonic, Fermionic, NoBraiding};
        // (op, [B, F, A, N]): whether the style is admitted.
        let mf: [(BraidingOp, [bool; 4]); 3] = [
            (Contraction, [true, true, false, false]),
            (PlanarContraction, [true; 4]),
            (Trace, [true, true, false, false]),
        ];
        let checked: [(BraidingOp, [bool; 4]); 3] = [
            (Contraction, [true, false, false, false]),
            (PlanarContraction, [true, false, false, false]),
            (Trace, [true, true, false, false]),
        ];
        let styles = [Bosonic, Fermionic, Anyonic, NoBraiding];
        let expect = |operation, style, admitted: bool| {
            if admitted {
                Ok(())
            } else {
                Err(OperationError::UnsupportedContractionBraiding { operation, style })
            }
        };
        for (op, row) in mf {
            for (style, admitted) in styles.into_iter().zip(row) {
                assert_eq!(
                    admit_braiding::<MultiplicityFreeAdmissionMode>(op, style),
                    expect(op, style, admitted),
                    "MF {op:?} {style:?}"
                );
            }
        }
        for (op, row) in checked {
            for (style, admitted) in styles.into_iter().zip(row) {
                assert_eq!(
                    admit_braiding::<CheckedGenericAdmissionMode>(op, style),
                    expect(op, style, admitted),
                    "checked {op:?} {style:?}"
                );
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
