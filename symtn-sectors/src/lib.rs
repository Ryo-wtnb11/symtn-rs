#![forbid(unsafe_code)]

//! Category vocabulary shared by symtn's sector and tensor layers.
//!
//! The process-global fusion-symbols coefficient caches are application
//! policy: only the application calls `fusion_symbols::cache::{reset, trim_to, configure_cache_budgets}`,
//! never symtn.

use smallvec::SmallVec;

mod rule_identity;
pub use rule_identity::RuleIdentity;

mod algebra;
pub use algebra::*;

mod category_rule;
pub use category_rule::{
    CanonicalUnitAdmission, CategoryAdapterError, CategoryAdmissionError, CategoryCodec,
    CategoryFailure, CategoryOperation, CategoryProvider, CategoryRule, EnumeratedCodec,
    UncertifiedUnitAdmission,
};

/// The seal of the provider traits.
///
/// Why sealed: a new symmetry is a `fusion_symbols` category in a
/// [`CategoryRule`], so symtn holds no second category implementation.
mod seal {
    /// The seal of [`FusionRule`](crate::FusionRule),
    /// [`CheckedGenericFusion`](crate::CheckedGenericFusion),
    /// [`TypedSectorAdmission`](crate::TypedSectorAdmission), and
    /// [`PhysicalFusionBasis`](crate::PhysicalFusionBasis).
    ///
    /// A native test provider (instrumentation, failure injection, a
    /// malformed-provider probe that no `fusion_symbols` category can
    /// express) implements it explicitly; there is no blanket implementation.
    pub trait Provider {}
}

/// Test-only access to native provider implementations.
#[cfg(any(test, feature = "testing"))]
pub mod testing {
    pub use crate::seal::Provider;
}

mod abelian;
pub use abelian::{
    FermionParityFusionRule, U1FusionRule, U1Irrep, Z2FusionRule, Z2Irrep, ZNFusionRule, ZNIrrep,
};

mod cu1;
pub use cu1::{CU1FusionRule, CU1Irrep, CU1_MAX_TWICE_CHARGE};

mod table_rule;
pub use table_rule::{TableRule, TableRuleError, TableSector};

mod su2;
pub use su2::{SU2CoefficientError, SU2FusionRule, SU2Irrep, SU2_MAX_DOUBLED_SPIN};

#[cfg(feature = "cgc-gen")]
mod sun;
#[cfg(feature = "cgc-gen")]
pub use sun::{SUNFusionRule, SUNFusionRuleError, SUNSymbolError};

mod product_rule;
pub use product_rule::{ProductAdmission, ProductFusionRule, ProductFusionRuleError};

/// Opaque identifier used by one fusion-rule implementation.
///
/// The numeric value is an internal representation, not a stable wire format
/// or a cross-version sector label. Persist and compare the corresponding
/// semantic irrep/product labels instead. Codec or layout changes may alter
/// numeric IDs, block order, and storage offsets without changing the
/// represented tensor.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct SectorId(usize);

impl SectorId {
    /// Constructs an expert-layer identifier for a specific fusion rule.
    ///
    /// Callers are responsible for using the rule's matching codec.
    pub const fn new(id: usize) -> Self {
        Self(id)
    }

    /// Returns the rule-local opaque numeric representation.
    ///
    /// Do not serialize this value as a semantic sector label.
    #[inline]
    pub const fn id(self) -> usize {
        self.0
    }
}

impl From<usize> for SectorId {
    fn from(value: usize) -> Self {
        Self::new(value)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum FusionStyleKind {
    Unique,
    Simple,
    Generic,
}

impl FusionStyleKind {
    #[inline]
    pub const fn is_multiplicity_free(self) -> bool {
        matches!(self, Self::Unique | Self::Simple)
    }

    #[inline]
    pub const fn has_multiple_outputs(self) -> bool {
        matches!(self, Self::Simple | Self::Generic)
    }

    #[inline]
    pub const fn has_multiplicity(self) -> bool {
        matches!(self, Self::Generic)
    }

    pub const fn combined_with(self, other: Self) -> Self {
        match (self, other) {
            (Self::Generic, _) | (_, Self::Generic) => Self::Generic,
            (Self::Simple, _) | (_, Self::Simple) => Self::Simple,
            (Self::Unique, Self::Unique) => Self::Unique,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum BraidingStyleKind {
    NoBraiding,
    Bosonic,
    Fermionic,
    Anyonic,
}

impl BraidingStyleKind {
    #[inline]
    pub const fn has_braiding(self) -> bool {
        !matches!(self, Self::NoBraiding)
    }

    #[inline]
    pub const fn is_symmetric(self) -> bool {
        matches!(self, Self::Bosonic | Self::Fermionic)
    }

    #[inline]
    pub const fn is_bosonic(self) -> bool {
        matches!(self, Self::Bosonic)
    }

    pub const fn combined_with(self, other: Self) -> Self {
        match (self, other) {
            (Self::NoBraiding, _) | (_, Self::NoBraiding) => Self::NoBraiding,
            (Self::Anyonic, _) | (_, Self::Anyonic) => Self::Anyonic,
            (Self::Fermionic, _) | (_, Self::Fermionic) => Self::Fermionic,
            (Self::Bosonic, Self::Bosonic) => Self::Bosonic,
        }
    }
}

/// Inline storage for the low layer's small per-rank / per-leg / per-block
/// metadata — the Rust analog of TensorKit's `NTuple` stack fields on
/// `FusionTree`. Structural keys and layouts (sector lists, dims, duals,
/// block indices, strides) stay allocation-free for the common small ranks,
/// so hashing/cloning/comparing them in the cold structure/plan/recoupling
/// caches touches no heap. Inline capacity 8 covers typical tensor ranks and
/// per-leg sector counts; larger cases spill to heap exactly like `Vec`.
pub type SectorVec = SmallVec<[SectorId; 8]>;

/// The workspace tolerance rule for arithmetic test comparisons
/// (`docs/testing_numerics.md`).
#[cfg(test)]
#[path = "../../tests/support"]
mod test_numerics {
    use num_complex::{Complex32, Complex64};
    pub(crate) mod numerics;
}
