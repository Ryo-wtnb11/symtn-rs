use std::sync::Arc;

use fusion_symbols::tables::{TableCategory, TableError, TableObject};
use num_complex::Complex64;

use crate::{
    BraidingStyleKind, CanonicalUnitFusionRule, CheckedFusionAlgebra, FusionAlgebraError,
    FusionRule, FusionStyleKind, MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols,
    MultiplicityFreeRigidSymbols, RuleIdentity, SectorCodec, SectorId, SectorVec,
};

/// A multiplicity-free unitary fusion category from the pinned AnyonWiki
/// tables of `fusion-symbols` (database commit `badbfe2`), as a symtn fusion
/// rule.
///
/// Every value is the database's, in the database's gauge, including for
/// [`fibonacci`](Self::fibonacci) and [`ising`](Self::ising); nothing is
/// converted to TensorKitSectors' gauge. TensorKit's `FibonacciAnyon` is the
/// same gauge exactly; `IsingAnyon` differs by the vertex gauge
/// `u^{σσ}_ψ = −1` and by its object order.
///
/// # Labels
///
/// Sectors are [`TableSector`]s: the database's one-based object labels in
/// its fusion-ring order, with `TableSector::new(1)` the unit. `SectorId(i)`
/// is `TableSector::new(i + 1)`, and the sector order is that label order.
///
/// # Styles
///
/// [`FusionRule::fusion_style`] is `Unique` when every fusion has exactly
/// one channel (read from the exact `N`), otherwise `Simple`.
/// [`FusionRule::braiding_style`] is `NoBraiding` for a fusion-only entry
/// and `Anyonic` for every braided one.
///
/// Why not report a symmetric table braiding as `Bosonic`/`Fermionic`, as
/// TensorKit would for a symmetric category: the tables carry no symmetry
/// flag, and deciding `R^{ab}_c R^{ba}_c = 1` from rounded floats is a
/// tolerance heuristic, not a proved invariant. `Anyonic` is stricter than
/// TensorKit there (general-axes `permute`, `contract` and `tr` are
/// rejected; `braid` with levels works); the symmetric group categories
/// have exact closed-form rules (`ZNFusionRule`, `Z2FusionRule`, ...).
///
/// On a `NoBraiding` rule every braiding operation is rejected by symtn
/// before it asks for an R-symbol or a twist, as in TensorKit. The
/// infallible [`r_symbol_scalar`](MultiplicityFreeFusionSymbols::r_symbol_scalar)
/// and [`twist_scalar`](MultiplicityFreeRigidSymbols::twist_scalar) panic
/// there, as for an out-of-domain label.
///
/// # Identity
///
/// The [`RuleIdentity`] is built from the table's authority fingerprint
/// (provenance record plus source bytes): each entry is its own category,
/// two instances of one entry are equal, and no table entry equals a
/// closed-form rule, even for the same group.
///
/// # Examples
///
/// ```
/// use symtn_sectors::{BraidingStyleKind, FusionRule, SectorCodec, TableRule, TableSector};
///
/// let fib = TableRule::fibonacci();
/// let tau = fib.encode_sector(&TableSector::new(2))?;
/// assert_eq!(fib.fusion_channels(tau, tau).len(), 2);
///
/// // Rep(S₃): a fusion category without braiding.
/// let rep_s3 = TableRule::by_uuid("37fa4e9c-97c6-11f1-8ce9-19309dfc597c")?;
/// assert_eq!(rep_s3.braiding_style(), BraidingStyleKind::NoBraiding);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug)]
pub struct TableRule {
    category: TableCategory,
    identity: RuleIdentity,
    fusion_style: FusionStyleKind,
}

/// A simple object of a [`TableRule`]: the database's one-based object
/// label, `1` being the unit. Which labels exist depends on the rule.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct TableSector(u16);

impl TableSector {
    /// The object with one-based label `label`.
    pub const fn new(label: u16) -> Self {
        Self(label)
    }

    /// The one-based label.
    pub const fn label(self) -> u16 {
        self.0
    }
}

/// A [`TableRule`] that cannot be constructed: the uuid names no entry of
/// the pinned database, or one symtn does not ship.
///
/// Opaque: the table's own error is reachable only through
/// [`std::error::Error::source`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableRuleError(TableError);

impl std::fmt::Display for TableRuleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("table category is unavailable")
    }
}

impl std::error::Error for TableRuleError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}

impl TableRule {
    /// The category with AnyonWiki uuid `uuid`, the database's stable
    /// identity of an entry.
    ///
    /// # Errors
    ///
    /// [`TableRuleError`] if the pinned database has no such entry or it is
    /// not shipped (non-unitary, or above rank 4).
    pub fn by_uuid(uuid: &str) -> Result<Self, TableRuleError> {
        TableCategory::by_uuid(uuid)
            .map(Self::from_category)
            .map_err(TableRuleError)
    }

    /// Fibonacci, `τ ⊗ τ = 𝟙 ⊕ τ`: `TableSector::new(1)` = `𝟙`,
    /// `TableSector::new(2)` = `τ`. TensorKit's `FibonacciAnyon`, gauge and
    /// braiding included (`R^{ττ}_𝟙 = e^{4πi/5}`).
    pub fn fibonacci() -> Self {
        Self::from_category(TableCategory::fibonacci())
    }

    /// Ising: `TableSector::new(1)` = `𝟙`, `2` = `ψ`, `3` = `σ` (TensorKit's
    /// `IsingAnyon` orders `I, σ, ψ`). Its R-symbols are `IsingAnyon`'s; its
    /// F-symbols differ by the vertex gauge `u^{σσ}_ψ = −1`, e.g.
    /// `F^{σσσ}_σ[𝟙, ψ] = −1/√2` here.
    pub fn ising() -> Self {
        Self::from_category(TableCategory::ising())
    }

    fn from_category(category: TableCategory) -> Self {
        let bytes = category.authority_fingerprint();
        let identity = RuleIdentity::from_canonical_bytes::<Self>(fnv1a64(bytes), Arc::from(bytes));
        let pointed = category.objects().all(|a| {
            category
                .objects()
                .all(|b| category.fusion(a, b).count() == 1)
        });
        let fusion_style = if pointed {
            FusionStyleKind::Unique
        } else {
            FusionStyleKind::Simple
        };
        Self {
            category,
            identity,
            fusion_style,
        }
    }

    /// The AnyonWiki uuid of this category.
    pub fn uuid(&self) -> &'static str {
        self.category.uuid()
    }

    /// Number of simple objects.
    pub fn rank(&self) -> usize {
        self.category.rank()
    }

    fn index(&self, sector: SectorId) -> Result<TableObject, FusionAlgebraError> {
        (sector.id() < self.rank())
            .then(|| TableObject(sector.id() as u16 + 1))
            .ok_or(FusionAlgebraError::InvalidSector { sector })
    }

    /// The infallible symbols' domain check (#1737): an out-of-domain id
    /// panics with the typed error instead of reaching the table.
    fn object(&self, method: &str, sector: SectorId) -> TableObject {
        self.index(sector)
            .unwrap_or_else(|error| panic!("{method} precondition violated: {error}"))
    }

    fn sector_id(object: TableObject) -> SectorId {
        SectorId::new(usize::from(object.label()) - 1)
    }
}

/// FNV-1a over 64 bits, the prehash `RuleIdentity` carries beside the bytes.
///
/// Why not `DefaultHasher`: its output is not stable across Rust releases.
/// The prehash is only a bucket key; equality compares the full bytes.
fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

impl FusionRule for TableRule {
    fn rule_identity(&self) -> RuleIdentity {
        self.identity.clone()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        self.fusion_style
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        if self.category.is_braided() {
            BraidingStyleKind::Anyonic
        } else {
            BraidingStyleKind::NoBraiding
        }
    }

    fn vacuum(&self) -> SectorId {
        Self::sector_id(self.category.unit())
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        self.dual_or_panic(sector)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        self.fusion_channels_or_panic(left, right)
    }

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        self.nsymbol_or_panic(left, right, coupled)
    }
}

impl CheckedFusionAlgebra for TableRule {
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        Ok(Self::sector_id(self.category.dual(self.index(sector)?)))
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        let (left, right) = (self.index(left)?, self.index(right)?);
        Ok(self
            .category
            .fusion(left, right)
            .map(Self::sector_id)
            .collect())
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        let (left, right, coupled) = (self.index(left)?, self.index(right)?, self.index(coupled)?);
        Ok(self.category.nsymbol(left, right, coupled) as usize)
    }
}

impl SectorCodec for TableRule {
    type Sector = TableSector;

    fn encode_sector(&self, value: &Self::Sector) -> Result<SectorId, FusionAlgebraError> {
        let label = usize::from(value.label());
        if (1..=self.rank()).contains(&label) {
            Ok(SectorId::new(label - 1))
        } else {
            Err(FusionAlgebraError::UnrepresentableSectorLabel {
                rule: self.identity.clone(),
                label: format!("table object {label} (rank {})", self.rank()),
            })
        }
    }

    fn decode_sector(&self, id: SectorId) -> Result<Self::Sector, FusionAlgebraError> {
        Ok(TableSector(self.index(id)?.label()))
    }
}

impl MultiplicityFreeFusionRule for TableRule {}

// The unit fusion laws are checked exactly at construction; the unitor
// associators are asserted bitwise `1 + 0i` for every shipped entry by the
// tables' own tests.
impl CanonicalUnitFusionRule for TableRule {}

impl MultiplicityFreeFusionSymbols for TableRule {
    type Scalar = Complex64;

    fn f_symbol_scalar(
        &self,
        left: SectorId,
        middle: SectorId,
        right: SectorId,
        coupled: SectorId,
        left_coupled: SectorId,
        right_coupled: SectorId,
    ) -> Self::Scalar {
        let o = |sector| self.object("f_symbol_scalar", sector);
        self.category.f_symbol(
            o(left),
            o(middle),
            o(right),
            o(coupled),
            o(left_coupled),
            o(right_coupled),
        )
    }

    fn r_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar {
        let o = |sector| self.object("r_symbol_scalar", sector);
        // Panics on a fusion-only entry (`TableError::NoBraiding`): every
        // braiding operation is rejected on `braiding_style()` first.
        self.category.r_symbol(o(left), o(right), o(coupled))
    }
}

impl MultiplicityFreeRigidSymbols for TableRule {
    fn dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        Complex64::new(self.category.dim(self.object("dim_scalar", sector)), 0.0)
    }

    fn inv_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        Complex64::new(
            1.0 / self.category.dim(self.object("inv_dim_scalar", sector)),
            0.0,
        )
    }

    fn sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        Complex64::new(
            self.category
                .dim(self.object("sqrt_dim_scalar", sector))
                .sqrt(),
            0.0,
        )
    }

    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> Self::Scalar {
        Complex64::new(
            1.0 / self
                .category
                .dim(self.object("inv_sqrt_dim_scalar", sector))
                .sqrt(),
            0.0,
        )
    }

    fn twist_scalar(&self, sector: SectorId) -> Self::Scalar {
        // Panics on a fusion-only entry, as `r_symbol_scalar`.
        self.category.twist(self.object("twist_scalar", sector))
    }

    fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> Self::Scalar {
        self.category
            .frobenius_schur(self.object("frobenius_schur_phase_scalar", sector))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv1a64_matches_the_published_reference_vectors() {
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a64(b"foobar"), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn identity_is_bound_to_the_table_fingerprint() {
        let fib = TableRule::fibonacci();
        let bytes = fib.category.authority_fingerprint();
        assert_eq!(
            fib.rule_identity(),
            RuleIdentity::from_canonical_bytes::<TableRule>(fnv1a64(bytes), Arc::from(bytes))
        );
    }
}
