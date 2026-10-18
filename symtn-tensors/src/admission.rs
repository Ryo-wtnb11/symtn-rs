//! Provider and braiding admission shared by every entry that accepts a
//! checked-Generic provider or needs a symmetric braiding (#1840).
//!
//! The facade, the network preflight and the tensor engines all route through
//! this one owner, which keeps the checks, their order and the public
//! messages identical everywhere.

use symtn_core::{
    BraidingStyleKind, CheckedGenericFusion, CoreError, FusionStyleKind, RuleIdentity,
};

use crate::{BoundDynamicFusionMapSpace, OperationError};

/// The error message of [`reject_non_symmetric_contraction`], shared with the
/// device network preflight so a network rejection and a typed `contract`
/// rejection are indistinguishable.
#[doc(hidden)]
pub const NON_SYMMETRIC_CONTRACTION_UNSUPPORTED: &str =
    "ordinary contraction requires symmetric (bosonic or fermionic) braiding; \
     use compose for the composition of morphisms, or planar_contract for a planar contraction";

/// The error message of a fusion trace over a non-symmetric braiding.
#[doc(hidden)]
pub const FUSION_TENSORTRACE_REQUIRES_SYMMETRIC_BRAIDING: &str =
    "fusion tensortrace requires symmetric braiding";

/// The operations that admit only a symmetric braiding; each keeps its own
/// public message.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SymmetricBraidingOp {
    /// General-axis contraction (TensorKit `blas_contract!`).
    Contraction,
    /// Fusion trace (TensorKit `trace_permute!`).
    Trace,
}

/// Admits `braiding` for `op`: only a symmetric braiding (TensorKit
/// `SymmetricBraiding`: Bosonic, Fermionic) is accepted.
#[doc(hidden)]
pub fn require_symmetric_braiding(
    braiding: BraidingStyleKind,
    op: SymmetricBraidingOp,
) -> Result<(), OperationError> {
    if braiding.is_symmetric() {
        return Ok(());
    }
    Err(OperationError::UnsupportedTensorContractScope {
        message: match op {
            SymmetricBraidingOp::Contraction => NON_SYMMETRIC_CONTRACTION_UNSUPPORTED,
            SymmetricBraidingOp::Trace => FUSION_TENSORTRACE_REQUIRES_SYMMETRIC_BRAIDING,
        },
    })
}

/// The ordinary-contraction boundary of every `contract` entry, returning or
/// overwriting, Host or device, typed or network, as by TensorKit
/// `blas_contract!` before any layout test.
///
/// Why not admit the canonical axes of a `NoBraiding` or `Anyonic` rule:
/// general-axes contraction is defined by braiding legs into place, which is
/// well defined for arbitrary permutations only under a symmetric braiding.
/// The one axis pattern that needs no braid is composition (TensorKit
/// `mul!`), which stays admitted for every braiding style; planar contraction
/// is the separate `planar_contract` (TensorKit `@planar`, #2284).
#[doc(hidden)]
pub fn reject_non_symmetric_contraction(braiding: BraidingStyleKind) -> Result<(), OperationError> {
    require_symmetric_braiding(braiding, SymmetricBraidingOp::Contraction)
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
    fn symmetric_braiding_keeps_each_operations_message() {
        for braiding in [BraidingStyleKind::Bosonic, BraidingStyleKind::Fermionic] {
            assert_eq!(
                require_symmetric_braiding(braiding, SymmetricBraidingOp::Trace),
                Ok(())
            );
        }
        assert_eq!(
            reject_non_symmetric_contraction(BraidingStyleKind::Anyonic),
            Err(OperationError::UnsupportedTensorContractScope {
                message: NON_SYMMETRIC_CONTRACTION_UNSUPPORTED
            })
        );
        assert_eq!(
            require_symmetric_braiding(BraidingStyleKind::NoBraiding, SymmetricBraidingOp::Trace),
            Err(OperationError::UnsupportedTensorContractScope {
                message: FUSION_TENSORTRACE_REQUIRES_SYMMETRIC_BRAIDING
            })
        );
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
