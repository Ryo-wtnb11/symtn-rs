use std::sync::Arc;

use fusion_symbols::category::FusionCategory;
use fusion_symbols::table::{TableCategory, TableError, TableObject};

use crate::category_rule::{
    fusion_algebra_error, AdapterAccess, CanonicalUnitAdmission, CategoryAdapterError,
    CategoryCodec,
};
use crate::{FusionAlgebraError, MultiplicityFreeAdmissionMode, RuleIdentity, SectorId};

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
/// [`FusionRule::fusion_style`](crate::FusionRule::fusion_style) is `Unique` when every fusion has exactly
/// one channel (read from the exact `N`), otherwise `Simple`.
/// [`FusionRule::braiding_style`](crate::FusionRule::braiding_style) and
/// [`has_trivial_associator_gauge`](crate::MultiplicityFreeFusionSymbols::has_trivial_associator_gauge)
/// report the table category's style and certificate directly. The category
/// authority owns their interpretation and the symbol gauge.
///
/// On a `NoBraiding` rule every braiding operation is rejected by symtn
/// before it asks for an R-symbol or a twist, as in TensorKit. The
/// infallible [`r_symbol_scalar`](crate::MultiplicityFreeFusionSymbols::r_symbol_scalar)
/// and [`twist_scalar`](crate::MultiplicityFreeRigidSymbols::twist_scalar) panic
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

/// Failure to load a [`TableRule`] through
/// [`fusion_symbols::table::TableCategory::by_uuid`].
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
    /// Returns [`TableRuleError`] when fusion-symbols rejects the UUID or
    /// cannot load the category. Its concrete [`TableError`] is preserved
    /// through [`std::error::Error::source`].
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
        Self { category, identity }
    }

    /// The AnyonWiki uuid of this category.
    pub fn uuid(&self) -> &'static str {
        self.category.uuid()
    }

    /// Number of simple objects.
    pub fn rank(&self) -> usize {
        table_rank(&self.category)
    }
}

fn table_rank(category: &TableCategory) -> usize {
    match category.object_count() {
        Ok(Some(count)) => usize::try_from(count).expect("table ranks fit u16 labels"),
        _ => unreachable!("table categories enumerate a finite object list"),
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

/// One-based `TableSector` labels and zero-based ids over a table.
#[derive(Debug)]
pub struct TableCodec;

impl CategoryCodec<TableCategory> for TableCodec {
    type Sector = TableSector;
    type Error = FusionAlgebraError;
    type Mode = MultiplicityFreeAdmissionMode;
    // Every shipped entry carries the table's canonical-unit certificate
    // (`tests/table_rule.rs`), which reads the stored unitor associators.
    type Unit = CanonicalUnitAdmission;

    fn encode_label<'a>(
        &self,
        category: impl Fn() -> &'a TableCategory + Copy,
        rule: &RuleIdentity,
        label: &TableSector,
    ) -> Result<SectorId, FusionAlgebraError> {
        let rank = table_rank(category());
        let label = usize::from(label.label());
        if (1..=rank).contains(&label) {
            Ok(SectorId::new(label - 1))
        } else {
            Err(FusionAlgebraError::UnrepresentableSectorLabel {
                rule: rule.clone(),
                label: format!("table object {label} (rank {rank})"),
            })
        }
    }

    fn decode_label<'a>(
        &self,
        category: impl Fn() -> &'a TableCategory + Copy,
        id: SectorId,
    ) -> Result<TableSector, FusionAlgebraError> {
        self.object(category, id)
            .map(|object| TableSector(object.index()))
    }

    fn object<'a>(
        &self,
        category: impl Fn() -> &'a TableCategory + Copy,
        id: SectorId,
    ) -> Result<TableObject, FusionAlgebraError> {
        u16::try_from(id.id())
            .ok()
            .and_then(|index| index.checked_add(1))
            .and_then(|label| category().object(label).ok())
            .ok_or(FusionAlgebraError::InvalidSector { sector: id })
    }

    fn sector_id<'a>(
        &self,
        _category: impl Fn() -> &'a TableCategory + Copy,
        _rule: &RuleIdentity,
        object: &TableObject,
    ) -> Result<SectorId, FusionAlgebraError> {
        Ok(SectorId::new(usize::from(object.index()) - 1))
    }

    fn map_error(
        &self,
        _inputs: &[SectorId],
        error: CategoryAdapterError<TableError>,
    ) -> FusionAlgebraError {
        fusion_algebra_error(error)
    }
}

impl crate::seal::Provider for TableRule {}

crate::category_rule::multiplicity_free_adapter_impls!(
    TableRule,
    TableSector,
    num_complex::Complex64
);

impl AdapterAccess for TableRule {
    type Category = TableCategory;
    type Codec = TableCodec;

    fn category(&self) -> &TableCategory {
        &self.category
    }
    fn codec(&self) -> &TableCodec {
        &TableCodec
    }
    fn identity(&self) -> &RuleIdentity {
        &self.identity
    }
    fn vacuum_id(&self) -> SectorId {
        SectorId::new(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FusionRule;

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
