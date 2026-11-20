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

use crate::{BoundDynamicFusionMapSpace, OperationError};

/// The refusal of ordinary contraction over a non-symmetric braiding, in
/// every mode.
#[doc(hidden)]
pub const NON_SYMMETRIC_CONTRACTION_UNSUPPORTED: &str =
    "ordinary contraction requires symmetric (bosonic or fermionic) braiding; \
     use compose for the composition of morphisms, or planar_contract for a planar contraction";

/// The refusal of checked-Generic ordinary contraction over a symmetric but
/// non-Bosonic (Fermionic) braiding: the checked engine has no contraction
/// twist.
#[doc(hidden)]
pub const CHECKED_CONTRACTION_REQUIRES_BOSONIC_BRAIDING: &str =
    "checked Generic contraction requires Bosonic braiding";

/// The refusal of a checked-Generic planar contraction: the checked engine
/// has no planar route, so only a Bosonic planar contraction, which the
/// facade runs as the ordinary one, is admitted.
#[doc(hidden)]
pub const CHECKED_PLANAR_CONTRACTION_UNSUPPORTED: &str =
    "checked Generic planar contraction requires Bosonic braiding";

/// The refusal of a fusion trace over a non-symmetric braiding.
#[doc(hidden)]
pub const FUSION_TENSORTRACE_REQUIRES_SYMMETRIC_BRAIDING: &str =
    "fusion tensortrace requires symmetric braiding";

/// The checked engine's internal fail-closed refusal (twist and core-alpha
/// invariants after admission).
pub(crate) const CHECKED_CONTRACTION_REQUIRES_BOSONIC: OperationError =
    OperationError::UnsupportedTensorContractScope {
        message: CHECKED_CONTRACTION_REQUIRES_BOSONIC_BRAIDING,
    };

/// The operations whose admission depends on the braiding style.
///
/// Composition (TensorKit `mul!`) crosses no strands and is admitted for
/// every style in every mode, so it has no entry.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BraidingOp {
    /// General-axis contraction (TensorKit `blas_contract!`).
    Contraction,
    /// Planar contraction (TensorKit `planarcontract!`).
    PlanarContraction,
    /// Fusion trace (TensorKit `trace_permute!`).
    Trace,
}

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
/// refusal is `UnsupportedTensorContractScope` with the operation's message.
#[doc(hidden)]
pub fn admit_braiding<M: BraidingAdmission>(
    op: BraidingOp,
    style: BraidingStyleKind,
) -> Result<(), OperationError> {
    if M::admits(op, style) {
        return Ok(());
    }
    Err(OperationError::UnsupportedTensorContractScope {
        message: match (op, style.is_symmetric()) {
            (BraidingOp::Contraction, false) => NON_SYMMETRIC_CONTRACTION_UNSUPPORTED,
            (BraidingOp::Contraction, true) => CHECKED_CONTRACTION_REQUIRES_BOSONIC_BRAIDING,
            (BraidingOp::PlanarContraction, _) => CHECKED_PLANAR_CONTRACTION_UNSUPPORTED,
            (BraidingOp::Trace, _) => FUSION_TENSORTRACE_REQUIRES_SYMMETRIC_BRAIDING,
        },
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
        const NS: &str = NON_SYMMETRIC_CONTRACTION_UNSUPPORTED;
        const CB: &str = CHECKED_CONTRACTION_REQUIRES_BOSONIC_BRAIDING;
        const CP: &str = CHECKED_PLANAR_CONTRACTION_UNSUPPORTED;
        const TR: &str = FUSION_TENSORTRACE_REQUIRES_SYMMETRIC_BRAIDING;
        // (op, [B, F, A, N]); `None` is admitted.
        let mf: [(BraidingOp, [Option<&str>; 4]); 3] = [
            (Contraction, [None, None, Some(NS), Some(NS)]),
            (PlanarContraction, [None; 4]),
            (Trace, [None, None, Some(TR), Some(TR)]),
        ];
        let checked: [(BraidingOp, [Option<&str>; 4]); 3] = [
            (Contraction, [None, Some(CB), Some(NS), Some(NS)]),
            (PlanarContraction, [None, Some(CP), Some(CP), Some(CP)]),
            (Trace, [None, None, Some(TR), Some(TR)]),
        ];
        let styles = [Bosonic, Fermionic, Anyonic, NoBraiding];
        let expect = |message: Option<&'static str>| match message {
            None => Ok(()),
            Some(message) => Err(OperationError::UnsupportedTensorContractScope { message }),
        };
        for (op, row) in mf {
            for (style, message) in styles.into_iter().zip(row) {
                assert_eq!(
                    admit_braiding::<MultiplicityFreeAdmissionMode>(op, style),
                    expect(message),
                    "MF {op:?} {style:?}"
                );
            }
        }
        for (op, row) in checked {
            for (style, message) in styles.into_iter().zip(row) {
                assert_eq!(
                    admit_braiding::<CheckedGenericAdmissionMode>(op, style),
                    expect(message),
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
