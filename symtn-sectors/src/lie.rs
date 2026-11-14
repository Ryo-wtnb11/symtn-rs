use fusion_symbols::{
    category::{FusionCategory, FusionStyle},
    lie::{GroupId, HalfInt, LieError, LieGroup, LieIrrep, RootSystem},
};
use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;
use std::sync::{Arc, PoisonError, RwLock};

use crate::category_rule::{forward, fusion_algebra_error, AdapterAccess};
use crate::{
    BraidingStyleKind, CanonicalUnitAdmission, CategoryAdapterError, CategoryCodec,
    CategoryFailure, CheckedCanonicalUnitFusionRule, CheckedFusionAlgebra,
    CheckedGenericAdmissionMode, CheckedGenericFusion, CheckedGenericPivotal,
    CheckedGenericRigidSymbols, FusionAlgebraError, FusionStyleKind, GenericFArray, GenericRMatrix,
    MultiplicityFreeAdmissionMode, PhysicalBasisError, PhysicalFusionBasis, RuleIdentity,
    SectorCodec, SectorId, SectorOrderKey, SectorVec, SymbolShapeError, TypedSectorAdmission,
};

/// Compact Lie-group rule over the fusion-symbols category adapter, admitted
/// in mode `M`.
///
/// `M` is [`CheckedGenericAdmissionMode`] (the default) or
/// [`MultiplicityFreeAdmissionMode`], as TensorKit fixes a sector type's
/// `FusionStyle`:
///
/// - checked Generic admits every group fusion-symbols executes and runs
///   every rank, rank one included, on the multiplicity path, with
///   [`LieGroupFusionRuleError`];
/// - multiplicity-free admits a group whose fusion-symbols fusion style is
///   `Unique` or `Simple` (tori, rank-one groups, D(2), U(2), and their
///   products), with [`FusionAlgebraError`](crate::FusionAlgebraError) and the
///   multiplicity-free provider traits. [`Self::new`] refuses a `Generic`
///   group such as SU(3) with
///   [`LieGroupFusionRuleError::IncompatibleFusionStyle`].
///
/// ```
/// use fusion_symbols::lie::GroupId;
/// use symtn_sectors::{FusionRule, LieGroupFusionRule, MultiplicityFreeAdmissionMode};
///
/// let su2 = LieGroupFusionRule::<MultiplicityFreeAdmissionMode>::new(GroupId::su(2)?)?;
/// let half = su2.encode_irrep(&[1])?;
/// assert_eq!(su2.nsymbol(half, half, su2.vacuum()), 1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// The typed label is fusion-symbols' [`LieIrrep`], built by
/// [`LieGroup::irrep`], [`LieGroup::irrep_with_charges`] or
/// [`LieGroup::irrep_spin`] on the rule's category. The private codec ranks
/// Dynkin labels then zigzag-mapped torus charges by graded total then
/// lexicographic order (see [`Self::encode_irrep_with_charges`]), orders
/// channels by encoded id, and constructs the category eagerly. The id map is
/// the same in both modes.
///
/// [`CheckedGenericFusion::sector_order_key`] is not the id: it is the
/// TensorKit `ProductSector` key of the group's factors in [`GroupId`] order.
/// A simple factor contributes its own graded-lexicographic Dynkin rank
/// (SUNRepresentations' `findindex - 1` for SU(N), the Dynkin label for
/// SU(2)), D(2) its two A1 labels, and each torus charge its TensorKit
/// `U1Irrep` `findindex - 1` (`4q - 1` for `q > 0`, else `4|q|`). A single
/// simple factor's key is therefore its id.
///
/// Multiplicity-free mode implements
/// [`PhysicalFusionBasis`](crate::PhysicalFusionBasis) (see that impl); the
/// checked Generic mode provides no physical expansion.
///
/// The rule's [`RuleIdentity`] is its mode and [`GroupId`] value: rules over
/// equal groups in one mode have equal identities whatever constructor or
/// lattice basis built them, and share structure-cache entries. Like a
/// [`CategoryRule`](crate::CategoryRule) identity, it holds the group weakly,
/// so that sharing ends once every clone of the rule is dropped.
pub struct LieGroupFusionRule<M = CheckedGenericAdmissionMode> {
    codec: LieCodec<M>,
    identity: RuleIdentity,
    category: LieGroup,
}

impl<M> Clone for LieGroupFusionRule<M> {
    fn clone(&self) -> Self {
        Self {
            codec: self.codec.clone(),
            identity: self.identity.clone(),
            category: self.category.clone(),
        }
    }
}

impl<M> fmt::Debug for LieGroupFusionRule<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LieGroupFusionRule")
            .field("mode", &std::any::type_name::<M>())
            .field("category", &self.category)
            .finish_non_exhaustive()
    }
}

pub(crate) struct LieCodec<M> {
    // Why an `Arc`: the value identity refers to the group weakly.
    group: Arc<GroupId>,
    // Why no table for a group with one coordinate (a rank-one simple factor
    // or a rank-one torus): fusion-symbols stores its irrep inline, so
    // rebuilding it from the id's coordinate costs one admission check,
    // while a table lookup hashes the whole `LieIrrep`, `GroupId` included,
    // under a lock. Every other group keeps the table: its irreps carry
    // root-data payloads (or a composite of factors) that are costly to
    // rebuild per query.
    irreps: Option<Arc<IrrepTable>>,
    // Why cached: `GroupId` sums its root-system ranks per call, and every
    // id conversion needs the split between Dynkin labels and charges.
    semisimple_rank: usize,
    coordinate_count: usize,
    mode: PhantomData<fn() -> M>,
}

impl<M> Clone for LieCodec<M> {
    fn clone(&self) -> Self {
        Self {
            group: self.group.clone(),
            irreps: self.irreps.clone(),
            semisimple_rank: self.semisimple_rank,
            coordinate_count: self.coordinate_count,
            mode: PhantomData,
        }
    }
}

impl<M> fmt::Debug for LieCodec<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LieCodec")
            .field("group", &self.group)
            .finish_non_exhaustive()
    }
}

mod sealed {
    use super::*;

    /// The admission modes of [`LieGroupFusionRule`], with the error each
    /// reports. The codec computes in [`LieGroupFusionRuleError`]; a mode
    /// translates it at the provider boundary.
    pub trait LieAdmissionMode: 'static {
        type Error: std::error::Error + Send + Sync + 'static;
        const MULTIPLICITY_FREE: bool;

        /// An id that names no admitted irrep.
        fn sector_error(sector: SectorId, error: LieGroupFusionRuleError) -> Self::Error;
        /// An irrep that has no id.
        fn label_error(
            rule: &RuleIdentity,
            label: &LieIrrep,
            error: LieGroupFusionRuleError,
        ) -> Self::Error;
        /// A fusion outcome of `left x right` that has no id.
        fn outcome_error(
            left: SectorId,
            right: SectorId,
            error: LieGroupFusionRuleError,
        ) -> Self::Error;
        /// A category or adapter failure; `generic` is the checked Generic
        /// translation.
        fn adapter_error(
            inputs: &[SectorId],
            error: CategoryAdapterError<LieError>,
            generic: impl FnOnce(CategoryAdapterError<LieError>) -> LieGroupFusionRuleError,
        ) -> Self::Error;
    }
}
use sealed::LieAdmissionMode;

impl LieAdmissionMode for CheckedGenericAdmissionMode {
    type Error = LieGroupFusionRuleError;
    const MULTIPLICITY_FREE: bool = false;

    fn sector_error(_: SectorId, error: LieGroupFusionRuleError) -> Self::Error {
        error
    }
    fn label_error(_: &RuleIdentity, _: &LieIrrep, error: LieGroupFusionRuleError) -> Self::Error {
        error
    }
    fn outcome_error(_: SectorId, _: SectorId, error: LieGroupFusionRuleError) -> Self::Error {
        error
    }
    fn adapter_error(
        _: &[SectorId],
        error: CategoryAdapterError<LieError>,
        generic: impl FnOnce(CategoryAdapterError<LieError>) -> LieGroupFusionRuleError,
    ) -> Self::Error {
        generic(error)
    }
}

/// Whether a codec failure means that a label or id lies outside the
/// codec's or fusion-symbols' domain, rather than a provider failure.
fn outside_domain(error: &LieGroupFusionRuleError) -> bool {
    matches!(
        error,
        LieGroupFusionRuleError::DecodeOverflow { .. }
            | LieGroupFusionRuleError::UnrepresentableLabel { .. }
            | LieGroupFusionRuleError::Provider(
                LieError::InvalidLabel | LieError::NotAdmissible | LieError::LabelOverflow
            )
    )
}

fn category_failure(error: LieGroupFusionRuleError) -> FusionAlgebraError {
    FusionAlgebraError::CategoryFailure(CategoryFailure::new(error))
}

/// The multiplicity-free vocabulary, as [`EnumeratedCodec`](crate::EnumeratedCodec)
/// reports it: an id outside the domain is `InvalidSector`, a label without an
/// id `UnrepresentableSectorLabel`, an unencodable or overflowing fusion
/// outcome `FusionNotRepresentable`, and every other failure a
/// `CategoryFailure` keeping its source.
impl LieAdmissionMode for MultiplicityFreeAdmissionMode {
    type Error = FusionAlgebraError;
    const MULTIPLICITY_FREE: bool = true;

    fn sector_error(sector: SectorId, error: LieGroupFusionRuleError) -> Self::Error {
        if outside_domain(&error) {
            FusionAlgebraError::InvalidSector { sector }
        } else {
            category_failure(error)
        }
    }
    fn label_error(
        rule: &RuleIdentity,
        label: &LieIrrep,
        error: LieGroupFusionRuleError,
    ) -> Self::Error {
        if outside_domain(&error) {
            FusionAlgebraError::UnrepresentableSectorLabel {
                rule: rule.clone(),
                label: format!("{label:?}"),
            }
        } else {
            category_failure(error)
        }
    }
    fn outcome_error(
        left: SectorId,
        right: SectorId,
        error: LieGroupFusionRuleError,
    ) -> Self::Error {
        if outside_domain(&error) {
            FusionAlgebraError::FusionNotRepresentable { left, right }
        } else {
            category_failure(error)
        }
    }
    fn adapter_error(
        inputs: &[SectorId],
        error: CategoryAdapterError<LieError>,
        _: impl FnOnce(CategoryAdapterError<LieError>) -> LieGroupFusionRuleError,
    ) -> Self::Error {
        match (&error, inputs) {
            (CategoryAdapterError::Provider(LieError::LabelOverflow), &[left, right]) => {
                FusionAlgebraError::FusionNotRepresentable { left, right }
            }
            _ => fusion_algebra_error(error),
        }
    }
}

/// Most sectors one rule interns. A full table still answers every query:
/// misses are converted per call.
const IRREP_TABLE_CAPACITY: usize = 1 << 16;

/// Interned `SectorId <-> fusion-symbols Irrep` pairs and duals of one Lie-group rule.
///
/// The codec is a bijection between admitted ids and group irreps, fixed by `GroupId`, so
/// either side is a complete key for the other, and the dual is a function of
/// the id. Entries are immutable once inserted and only successful
/// conversions are stored, so errors are recomputed per call. At most
/// [`IRREP_TABLE_CAPACITY`] pairs, each one provider `Arc<LieIrrep>` shared
/// by both maps, and as many dual entries. Clones of a rule share the table;
/// readers take a shared lock.
#[derive(Default)]
struct IrrepTable(RwLock<IrrepMaps>);

#[derive(Default)]
struct IrrepMaps {
    by_id: HashMap<SectorId, Arc<LieIrrep>>,
    by_irrep: HashMap<Arc<LieIrrep>, SectorId>,
    duals: HashMap<SectorId, SectorId>,
}

impl fmt::Debug for IrrepTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IrrepTable").finish_non_exhaustive()
    }
}

impl IrrepTable {
    fn by_id(&self, sector: SectorId) -> Option<LieIrrep> {
        let maps = self.0.read().unwrap_or_else(PoisonError::into_inner);
        maps.by_id.get(&sector).map(|irrep| LieIrrep::clone(irrep))
    }

    fn by_irrep(&self, irrep: &LieIrrep) -> Option<SectorId> {
        let maps = self.0.read().unwrap_or_else(PoisonError::into_inner);
        maps.by_irrep.get(irrep).copied()
    }

    fn contains(&self, sector: SectorId) -> bool {
        let maps = self.0.read().unwrap_or_else(PoisonError::into_inner);
        maps.by_id.contains_key(&sector)
    }

    fn dual(&self, sector: SectorId) -> Option<SectorId> {
        let maps = self.0.read().unwrap_or_else(PoisonError::into_inner);
        maps.duals.get(&sector).copied()
    }

    /// Records `sector <-> dual`; duality is an involution.
    fn insert_dual(&self, sector: SectorId, dual: SectorId) {
        let mut maps = self.0.write().unwrap_or_else(PoisonError::into_inner);
        if maps.duals.len() < IRREP_TABLE_CAPACITY - usize::from(sector != dual) {
            maps.duals.insert(sector, dual);
            maps.duals.insert(dual, sector);
        }
    }

    /// Records a pair and returns the interned irrep (the first one inserted
    /// when threads race on the same sector).
    ///
    /// Why callers insert only fusion-symbols-admitted irreps (built by
    /// `LieGroup::irrep` or returned by the category): encode and decode skip
    /// lattice admission on a hit, so an excluded label in this table would
    /// silently become a sector. Every path's hole rejection rests on it.
    fn insert(&self, sector: SectorId, irrep: LieIrrep) -> LieIrrep {
        let mut maps = self.0.write().unwrap_or_else(PoisonError::into_inner);
        if let Some(existing) = maps.by_id.get(&sector) {
            return LieIrrep::clone(existing);
        }
        if maps.by_id.len() < IRREP_TABLE_CAPACITY {
            let shared = Arc::new(irrep.clone());
            maps.by_id.insert(sector, shared.clone());
            maps.by_irrep.insert(shared, sector);
        }
        irrep
    }
}

// The identity and the irrep table are pure functions of the mode and
// `GroupId`, so equality and hashing ignore them.
impl<M> PartialEq for LieGroupFusionRule<M> {
    fn eq(&self, other: &Self) -> bool {
        self.codec.group == other.codec.group
    }
}
impl<M> Eq for LieGroupFusionRule<M> {}
impl<M> std::hash::Hash for LieGroupFusionRule<M> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.codec.group.hash(state);
    }
}

/// Failure at the Lie-group label/provider boundary.
#[derive(Debug)]
pub enum LieGroupFusionRuleError {
    /// Multiplicity-free admission of a group whose fusion has multiplicity.
    IncompatibleFusionStyle {
        actual: FusionStyle,
    },
    /// Semisimple plus torus rank exceeds this platform's label buffers.
    RankNotRepresentable {
        rank: usize,
    },
    LabelAllocation {
        len: usize,
        source: std::collections::TryReserveError,
    },
    /// A label slice has the wrong length: the Dynkin labels are checked
    /// first, then the charges, and the first mismatch is reported.
    LabelLength {
        expected: usize,
        found: usize,
    },
    NegativeLabel {
        index: usize,
        value: i64,
    },
    /// The label (Dynkin labels, then charges) is representable neither by
    /// this codec's `usize` ids nor by fusion-symbols.
    UnrepresentableLabel {
        labels: Vec<i64>,
    },
    DecodeOverflow {
        sector: SectorId,
    },
    DimensionNotRepresentable {
        sector: SectorId,
    },
    /// Returned symbol storage length disagrees with the logical N axes.
    MalformedSymbolData(SymbolShapeError),
    /// Returned symbol axes disagree with the fusion multiplicities.
    MalformedSymbolAxes {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    Provider(LieError),
}

impl fmt::Display for LieGroupFusionRuleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IncompatibleFusionStyle { actual } => write!(
                f,
                "multiplicity-free admission requires a multiplicity-free group, got {actual:?} fusion"
            ),
            Self::RankNotRepresentable { rank } => {
                write!(
                    f,
                    "{rank} label coordinates exceed this platform's Vec limit"
                )
            }
            Self::LabelAllocation { len, .. } => {
                write!(f, "could not allocate {len} label coordinates")
            }
            Self::LabelLength { expected, found } => {
                write!(
                    f,
                    "expected {expected} entries in the first mismatched label slice \
                     (Dynkin labels, then charges), got {found}"
                )
            }
            Self::NegativeLabel { index, value } => {
                write!(f, "Dynkin label {index} is negative ({value})")
            }
            Self::UnrepresentableLabel { labels } => {
                write!(
                    f,
                    "label {labels:?} (Dynkin labels, then charges) is not representable \
                     by this codec or fusion-symbols"
                )
            }
            Self::DecodeOverflow { sector } => {
                write!(
                    f,
                    "SectorId {} cannot be decoded by this Lie-group codec",
                    sector.id()
                )
            }
            Self::DimensionNotRepresentable { sector } => write!(
                f,
                "exact dimension for SectorId {} is not a finite positive f64",
                sector.id()
            ),
            Self::MalformedSymbolData(_) => f.write_str("malformed provider symbol data"),
            Self::MalformedSymbolAxes { expected, actual } => write!(
                f,
                "provider symbol axes {actual:?} disagree with the fusion multiplicities {expected:?}"
            ),
            Self::Provider(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for LieGroupFusionRuleError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::LabelAllocation { source, .. } => Some(source),
            Self::MalformedSymbolData(error) => Some(error),
            Self::Provider(error) => Some(error),
            _ => None,
        }
    }
}

fn provider_error(error: LieError) -> LieGroupFusionRuleError {
    LieGroupFusionRuleError::Provider(error)
}

impl<M: LieAdmissionMode> LieGroupFusionRule<M> {
    /// Constructs a rule in mode `M` and eagerly checks provider support for
    /// the full group.
    /// Every group that fusion-symbols' [`LieGroup::new`] executes is admitted
    /// in checked Generic mode: simple groups of every type, A–G (D2
    /// included), semisimple groups, tori, U(n) and semisimple×torus central
    /// quotients. A group it rejects returns
    /// [`LieGroupFusionRuleError::Provider`] with its `LieError`. Without the
    /// `cgc-gen` feature fusion-symbols executes only A1, B1, C1 and D2
    /// factors (with tori), and every other group returns
    /// `Provider(LieError::Unsupported { requires: "lie-generated" })`.
    ///
    /// Multiplicity-free mode admits only a group whose fusion-symbols
    /// fusion style is `Unique` or `Simple`; any other style returns
    /// [`LieGroupFusionRuleError::IncompatibleFusionStyle`].
    ///
    /// The `RankNotRepresentable` bound covers this rule's label buffers only.
    /// The rule's identity hashes the canonical [`GroupId`] once and compares
    /// groups by value, without materializing the character lattice; the
    /// fusion-symbols category's own construction cost for a large rank is
    /// fusion-symbols' (fusion-symbols#187).
    pub fn new(group: GroupId) -> Result<Self, LieGroupFusionRuleError> {
        let rank = group.semisimple_rank().saturating_add(group.torus_rank());
        if rank > isize::MAX as usize / size_of::<i64>() {
            return Err(LieGroupFusionRuleError::RankNotRepresentable { rank });
        }
        let category = LieGroup::new(group.clone()).map_err(provider_error)?;
        match category.fusion_style() {
            FusionStyle::Unique | FusionStyle::Simple => {}
            actual if M::MULTIPLICITY_FREE => {
                return Err(LieGroupFusionRuleError::IncompatibleFusionStyle { actual })
            }
            _ => {}
        }
        // Why the mode and group are a complete key: the codec and its
        // conventions are fixed by the rule type, and the provider data by
        // the group and this build.
        let group = Arc::new(group);
        let identity = RuleIdentity::value_of::<Self, GroupId>().resolve_value(&group);
        let irreps = (rank != 1).then(Arc::default);
        let semisimple_rank = group.semisimple_rank();
        Ok(Self {
            codec: LieCodec {
                group,
                irreps,
                semisimple_rank,
                coordinate_count: rank,
                mode: PhantomData,
            },
            identity,
            category,
        })
    }

    /// The fusion-symbols category, whose [`LieGroup::irrep`],
    /// [`LieGroup::irrep_with_charges`] and [`LieGroup::irrep_spin`] build
    /// the typed labels.
    pub fn category(&self) -> &LieGroup {
        &self.category
    }

    /// Returns the complete root datum and character lattice.
    pub fn group_id(&self) -> &GroupId {
        &self.codec.group
    }

    /// Encodes an admitted irrep of a group without torus charges, given as
    /// fusion-symbols' [`LieGroup::irrep`] takes it: nonnegative Dynkin labels
    /// in root-system order. A group with a torus returns
    /// [`LieGroupFusionRuleError::Provider`] with [`LieError::InvalidLabel`];
    /// use [`Self::encode_irrep_with_charges`] there.
    pub fn encode_irrep(&self, dynkin: &[i64]) -> Result<SectorId, LieGroupFusionRuleError> {
        if self.codec.group.torus_rank() != 0 {
            return Err(provider_error(LieError::InvalidLabel));
        }
        self.encode_irrep_with_charges(dynkin, &[])
    }

    /// Encodes an admitted irrep of any group, given as fusion-symbols'
    /// [`LieGroup::irrep_with_charges`] takes it: nonnegative Dynkin labels in
    /// root-system order and signed torus charges.
    ///
    /// Ids rank all coordinates by total grade, then lexicographically: the
    /// Dynkin labels as they are, then each charge zigzag-mapped
    /// (`0, 1, -1, 2, -2, ...` to `0, 1, 2, 3, 4, ...`). Labels the group's
    /// character lattice excludes are holes in the id space. Labels whose id
    /// overflows `usize`, or that fusion-symbols cannot represent, return
    /// [`LieGroupFusionRuleError::UnrepresentableLabel`].
    pub fn encode_irrep_with_charges(
        &self,
        dynkin: &[i64],
        charges: &[i64],
    ) -> Result<SectorId, LieGroupFusionRuleError> {
        self.codec.encode_irrep(self.category(), dynkin, charges)
    }

    /// Encodes the spin-`j` irrep of SU(2) or SO(3), in any rank-one
    /// presentation, as fusion-symbols' [`LieGroup::irrep_spin`] admits it: the
    /// irrep with Dynkin label `2j`, so SO(3) rejects half-integer `j` with
    /// [`LieError::NotAdmissible`]. Every other group returns
    /// [`LieGroupFusionRuleError::Provider`] with [`LieError::InvalidLabel`].
    pub fn encode_spin(&self, j: HalfInt) -> Result<SectorId, LieGroupFusionRuleError> {
        let spin_group = matches!(
            self.codec.group.root_systems(),
            [RootSystem::A(1) | RootSystem::B(1) | RootSystem::C(1)]
        ) && self.codec.group.torus_rank() == 0;
        if !spin_group {
            return Err(provider_error(LieError::InvalidLabel));
        }
        self.encode_irrep_with_charges(&[i64::from(j.doubled())], &[])
    }

    /// Decodes an admitted id to its Dynkin labels and torus charges, the
    /// inverse of [`Self::encode_irrep_with_charges`].
    pub fn decode_irrep(
        &self,
        sector: SectorId,
    ) -> Result<(Vec<i64>, Vec<i64>), LieGroupFusionRuleError> {
        self.codec.decode_irrep(self.category(), sector)
    }
}

/// Graded-lexicographic coordinate of a torus charge: `0, 1, -1, 2, -2, ...`
/// map to `0, 1, 2, 3, 4, ...`, the index of fusion-symbols' U(1) enumeration
/// (TensorKitSectors' `U1Irrep` order, not its `findindex`).
fn zigzag(charge: i64) -> Option<usize> {
    let doubled = usize::try_from(charge.unsigned_abs())
        .ok()?
        .checked_mul(2)?;
    Some(if charge > 0 { doubled - 1 } else { doubled })
}

fn unzigzag(coordinate: usize) -> Option<i64> {
    let magnitude = i64::try_from(coordinate.div_ceil(2)).ok()?;
    Some(if coordinate % 2 == 1 {
        magnitude
    } else {
        -magnitude
    })
}

fn joined(dynkin: &[i64], charges: &[i64]) -> Vec<i64> {
    dynkin.iter().chain(charges).copied().collect()
}

/// fusion-symbols' `LabelOverflow` on a label is this codec's unrepresentable label.
fn label_error(error: LieError, dynkin: &[i64], charges: &[i64]) -> LieGroupFusionRuleError {
    match error {
        LieError::LabelOverflow => LieGroupFusionRuleError::UnrepresentableLabel {
            labels: joined(dynkin, charges),
        },
        error => provider_error(error),
    }
}

fn try_collect_labels(
    labels: impl ExactSizeIterator<Item = i64>,
) -> Result<Vec<i64>, LieGroupFusionRuleError> {
    let len = labels.len();
    let mut out = Vec::new();
    out.try_reserve_exact(len)
        .map_err(|source| LieGroupFusionRuleError::LabelAllocation { len, source })?;
    out.extend(labels);
    Ok(out)
}

impl<M> LieCodec<M> {
    fn semisimple_rank(&self) -> usize {
        self.semisimple_rank
    }

    fn coordinate_count(&self) -> usize {
        self.coordinate_count
    }

    // Arithmetic gives the id. Lattice admission and representability are
    // fusion-symbols' `irrep`, run once per interned sector, or per call for
    // a table-free group: the rank is a bijection on all coordinate vectors,
    // so an interned id names this very, already admitted, label, and a warm
    // encode allocates nothing.
    fn encode_irrep(
        &self,
        category: &LieGroup,
        dynkin: &[i64],
        charges: &[i64],
    ) -> Result<SectorId, LieGroupFusionRuleError> {
        self.check_labels(dynkin, charges)?;
        let sector = self.encode_coordinates(
            || {
                let dynkin = dynkin.iter().map(|&x| usize::try_from(x).ok());
                dynkin.chain(charges.iter().map(|&q| zigzag(q)))
            },
            || joined(dynkin, charges),
        )?;
        if !self
            .irreps
            .as_ref()
            .is_some_and(|table| table.contains(sector))
        {
            self.admitted(category, sector, dynkin, charges)?;
        }
        Ok(sector)
    }

    /// The fusion-symbols irrep of decoded coordinates, interned when the
    /// group has a table.
    fn admitted(
        &self,
        category: &LieGroup,
        sector: SectorId,
        dynkin: &[i64],
        charges: &[i64],
    ) -> Result<LieIrrep, LieGroupFusionRuleError> {
        let irrep = category
            .irrep_with_charges(dynkin, charges)
            .map_err(|error| label_error(error, dynkin, charges))?;
        Ok(match &self.irreps {
            Some(table) => table.insert(sector, irrep),
            None => irrep,
        })
    }

    // Compute the grade and lexicographic rank without materializing labels.
    // `None` coordinates are unrepresentable.
    fn encode_coordinates<I: Iterator<Item = Option<usize>>>(
        &self,
        coordinates: impl Fn() -> I,
        labels: impl Fn() -> Vec<i64>,
    ) -> Result<SectorId, LieGroupFusionRuleError> {
        graded_lex_rank(self.coordinate_count(), coordinates)
            .map(SectorId::new)
            .ok_or_else(|| LieGroupFusionRuleError::UnrepresentableLabel { labels: labels() })
    }

    /// TensorKit `ProductSector` positions of the factors, in [`GroupId`]
    /// order: each simple factor's graded-lexicographic Dynkin rank (D(2) as
    /// its two A1 labels), then each charge's `U1Irrep` `findindex - 1`.
    /// `None` when a position does not fit.
    fn order_key<D: Iterator<Item = i64>, Q: Iterator<Item = i64>>(
        &self,
        dynkin: impl Fn() -> D,
        charges: Q,
    ) -> Option<SectorOrderKey> {
        let mut key: Option<SectorOrderKey> = None;
        let mut push = |position: u64| {
            let factor = SectorOrderKey::position(position);
            key = Some(match key.take() {
                Some(key) => SectorOrderKey::product(key, factor),
                None => factor,
            });
        };
        let mut offset = 0;
        for &root in self.group.root_systems() {
            let rank = root.rank();
            let block = || {
                dynkin()
                    .skip(offset)
                    .take(rank)
                    .map(|x| usize::try_from(x).ok())
            };
            if root == RootSystem::D(2) {
                for label in block() {
                    push(u64::try_from(label?).ok()?);
                }
            } else {
                push(u64::try_from(graded_lex_rank(rank, block)?).ok()?);
            }
            offset += rank;
        }
        for charge in charges {
            push(crate::abelian::u1_order_position(charge)?);
        }
        // The coordinate-free group's only sector.
        Some(key.unwrap_or_else(|| SectorOrderKey::position(0)))
    }

    fn decode_irrep(
        &self,
        category: &LieGroup,
        sector: SectorId,
    ) -> Result<(Vec<i64>, Vec<i64>), LieGroupFusionRuleError> {
        if let Some(irrep) = self.irreps.as_ref().and_then(|table| table.by_id(sector)) {
            return Ok((
                try_collect_labels(irrep.dynkin())?,
                try_collect_labels(irrep.charges())?,
            ));
        }
        let (dynkin, charges) = self.decode_coordinates(sector)?;
        self.admitted(category, sector, &dynkin, &charges)?;
        Ok((dynkin, charges))
    }

    /// The single coordinate of a one-coordinate group's id: the Dynkin
    /// label itself, or the unzigzagged charge.
    fn single_coordinate(&self, sector: SectorId) -> Result<i64, LieGroupFusionRuleError> {
        let coordinate = if self.semisimple_rank() == 1 {
            i64::try_from(sector.id()).ok()
        } else {
            unzigzag(sector.id())
        };
        coordinate.ok_or(LieGroupFusionRuleError::DecodeOverflow { sector })
    }

    /// Decodes a `SectorId` with bounded binary searches over the grade and
    /// each coordinate. Lattice admission and representability are the
    /// caller's, through fusion-symbols' `irrep`.
    fn decode_coordinates(
        &self,
        sector: SectorId,
    ) -> Result<(Vec<i64>, Vec<i64>), LieGroupFusionRuleError> {
        let id = sector.id();
        let r = self.semisimple_rank();
        let n = self.coordinate_count();
        let mut low = 0usize;
        let mut high = id;
        while low < high {
            let mid = low + (high - low) / 2;
            if count_before_grade(mid + 1, n, id).is_exceeded() {
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        let grade = low;
        let before = if grade == 0 {
            0
        } else {
            count_before_grade(grade, n, id)
                .exact()
                .ok_or(LieGroupFusionRuleError::DecodeOverflow { sector })?
        };
        let mut residual = id - before;
        let mut remaining = grade;
        let mut dynkin = Vec::new();
        dynkin
            .try_reserve_exact(r)
            .map_err(|source| LieGroupFusionRuleError::LabelAllocation { len: r, source })?;
        let mut charges = Vec::new();
        charges
            .try_reserve_exact(n - r)
            .map_err(|source| LieGroupFusionRuleError::LabelAllocation { len: n - r, source })?;
        for index in 0..n {
            let tail = n - index - 1;
            let mut low = 0usize;
            let mut high = remaining;
            while low < high {
                let mid = low + (high - low).div_ceil(2);
                match prefix_count(remaining, tail, mid, residual) {
                    Some(Capped::Exact(count)) if count <= residual => low = mid,
                    _ => high = mid - 1,
                }
            }
            let skipped = prefix_count(remaining, tail, low, residual)
                .and_then(Capped::exact)
                .ok_or(LieGroupFusionRuleError::DecodeOverflow { sector })?;
            residual -= skipped;
            let label = if index < r {
                i64::try_from(low).ok()
            } else {
                unzigzag(low)
            };
            let label = label.ok_or(LieGroupFusionRuleError::DecodeOverflow { sector })?;
            if index < r {
                dynkin.push(label);
            } else {
                charges.push(label);
            }
            remaining -= low;
        }
        // A coordinate-free (trivial) group has the single id 0.
        if residual != 0 || remaining != 0 {
            return Err(LieGroupFusionRuleError::DecodeOverflow { sector });
        }
        Ok((dynkin, charges))
    }

    fn check_labels(&self, dynkin: &[i64], charges: &[i64]) -> Result<(), LieGroupFusionRuleError> {
        for (labels, expected) in [
            (dynkin, self.semisimple_rank()),
            (charges, self.group.torus_rank()),
        ] {
            if labels.len() != expected {
                return Err(LieGroupFusionRuleError::LabelLength {
                    expected,
                    found: labels.len(),
                });
            }
        }
        if let Some((index, &value)) = dynkin.iter().enumerate().find(|(_, &x)| x < 0) {
            return Err(LieGroupFusionRuleError::NegativeLabel { index, value });
        }
        Ok(())
    }

    fn irrep(
        &self,
        category: &LieGroup,
        sector: SectorId,
    ) -> Result<LieIrrep, LieGroupFusionRuleError> {
        let Some(table) = &self.irreps else {
            let coordinate = [self.single_coordinate(sector)?];
            let (dynkin, charges) = coordinate.split_at(self.semisimple_rank());
            return self.admitted(category, sector, dynkin, charges);
        };
        if let Some(irrep) = table.by_id(sector) {
            return Ok(irrep);
        }
        let (dynkin, charges) = self.decode_coordinates(sector)?;
        self.admitted(category, sector, &dynkin, &charges)
    }

    fn sector(
        &self,
        category: &LieGroup,
        irrep: Cow<'_, LieIrrep>,
    ) -> Result<SectorId, LieGroupFusionRuleError> {
        if !category.contains(&irrep) {
            return Err(provider_error(LieError::InvalidLabel));
        }
        if let Some(sector) = self
            .irreps
            .as_ref()
            .and_then(|table| table.by_irrep(&irrep))
        {
            return Ok(sector);
        }
        let sector = self.encode_coordinates(
            || {
                let dynkin = irrep.dynkin().map(|x| usize::try_from(x).ok());
                dynkin.chain(irrep.charges().map(zigzag))
            },
            || irrep.dynkin().chain(irrep.charges()).collect(),
        )?;
        if let Some(table) = &self.irreps {
            table.insert(sector, irrep.into_owned());
        }
        Ok(sector)
    }

    /// Coordinate-wise sum of two decoded labels: the highest weight of their
    /// tensor product, which carries any coordinate a fusion overflowed.
    fn highest_weight_sum(&self, left: SectorId, right: SectorId) -> Option<Vec<i64>> {
        let (left_dynkin, left_charges) = self.decode_coordinates(left).ok()?;
        let (right_dynkin, right_charges) = self.decode_coordinates(right).ok()?;
        left_dynkin
            .iter()
            .chain(&left_charges)
            .zip(right_dynkin.iter().chain(&right_charges))
            .map(|(a, b)| a.checked_add(*b))
            .collect()
    }

    /// The order key of `id`; `None` when it decodes to no label with a key.
    fn try_sector_order_key(&self, id: SectorId) -> Option<SectorOrderKey> {
        let Some(table) = &self.irreps else {
            let coordinate = [self.single_coordinate(id).ok()?];
            let (dynkin, charges) = coordinate.split_at(self.semisimple_rank());
            return self.order_key(|| dynkin.iter().copied(), charges.iter().copied());
        };
        match table.by_id(id) {
            Some(irrep) => self.order_key(|| irrep.dynkin(), irrep.charges()),
            None => self
                .decode_coordinates(id)
                .ok()
                .and_then(|(dynkin, charges)| {
                    self.order_key(|| dynkin.iter().copied(), charges.into_iter())
                }),
        }
    }

    fn generic_map_error(
        &self,
        inputs: &[SectorId],
        error: CategoryAdapterError<LieError>,
    ) -> LieGroupFusionRuleError {
        match error {
            CategoryAdapterError::Provider(LieError::LabelOverflow) => match inputs {
                &[left, right] => self
                    .highest_weight_sum(left, right)
                    .map_or(provider_error(LieError::LabelOverflow), |labels| {
                        LieGroupFusionRuleError::UnrepresentableLabel { labels }
                    }),
                _ => provider_error(LieError::LabelOverflow),
            },
            CategoryAdapterError::Provider(error) => provider_error(error),
            CategoryAdapterError::DimensionNotRepresentable { sector } => {
                LieGroupFusionRuleError::DimensionNotRepresentable { sector }
            }
            CategoryAdapterError::MalformedSymbolData(error) => {
                LieGroupFusionRuleError::MalformedSymbolData(error)
            }
            CategoryAdapterError::MalformedSymbolAxes { expected, actual } => {
                LieGroupFusionRuleError::MalformedSymbolAxes { expected, actual }
            }
            CategoryAdapterError::ForeignObject { .. } => {
                provider_error(fusion_symbols::lie::LieError::InvalidLabel)
            }
            CategoryAdapterError::NoBraiding => {
                provider_error(fusion_symbols::lie::LieError::Unsupported {
                    requires: "braiding",
                })
            }
        }
    }
}

impl<M: LieAdmissionMode> LieCodec<M> {
    fn dual<'a>(
        &self,
        category: impl Fn() -> &'a LieGroup + Copy,
        sector: SectorId,
    ) -> Result<SectorId, M::Error> {
        if let Some(dual) = self.irreps.as_ref().and_then(|table| table.dual(sector)) {
            return Ok(dual);
        }
        let object = crate::category_rule::member_object(self, category, sector)?;
        let dual = self
            .sector(category(), Cow::Owned(category().dual(&object)))
            .map_err(|error| M::sector_error(sector, error))?;
        if let Some(table) = &self.irreps {
            table.insert_dual(sector, dual);
        }
        Ok(dual)
    }
}

impl<M: LieAdmissionMode> CategoryCodec<LieGroup> for LieCodec<M> {
    type Sector = LieIrrep;
    type Error = M::Error;
    type Mode = M;
    type Unit = CanonicalUnitAdmission;

    fn encode_label<'a>(
        &self,
        category: impl Fn() -> &'a LieGroup + Copy,
        rule: &RuleIdentity,
        label: &LieIrrep,
    ) -> Result<SectorId, M::Error> {
        self.sector_id(category, rule, label)
    }

    fn decode_label<'a>(
        &self,
        category: impl Fn() -> &'a LieGroup + Copy,
        id: SectorId,
    ) -> Result<LieIrrep, M::Error> {
        self.object(category, id)
    }

    fn object<'a>(
        &self,
        category: impl Fn() -> &'a LieGroup + Copy,
        id: SectorId,
    ) -> Result<LieIrrep, M::Error> {
        self.irrep(category(), id)
            .map_err(|error| M::sector_error(id, error))
    }

    fn sector_id<'a>(
        &self,
        category: impl Fn() -> &'a LieGroup + Copy,
        rule: &RuleIdentity,
        object: &LieIrrep,
    ) -> Result<SectorId, M::Error> {
        self.sector(category(), Cow::Borrowed(object))
            .map_err(|error| M::label_error(rule, object, error))
    }

    fn admit<'a, const N: usize>(
        &self,
        category: impl Fn() -> &'a LieGroup + Copy,
        _rule: &RuleIdentity,
        operation: crate::CategoryOperation,
        ids: [SectorId; N],
    ) -> Result<[LieIrrep; N], M::Error> {
        // Why not `try_map_array`: its `array::map` with a captured failure
        // stays out of line per element (`core::array::try_map` frames in the
        // profile) and made every multiplicity query about 1.4x slower than
        // `SU2FusionRule`'s; this loop inlines the conversions.
        let mut slots: [Option<LieIrrep>; N] = core::array::from_fn(|_| None);
        for (slot, &id) in slots.iter_mut().zip(&ids) {
            *slot = Some(crate::category_rule::member_object(self, category, id)?);
        }
        let objects = slots.map(|object| object.expect("every input was decoded"));
        let dualized = match operation {
            crate::CategoryOperation::A => ids.first(),
            crate::CategoryOperation::B => ids.get(1),
            _ => None,
        };
        if let Some(&id) = dualized {
            self.dual(category, id)?;
        }
        Ok(objects)
    }

    /// Why the key, unlike the id, is per factor: TensorKit orders a
    /// `ProductSector` by its factors' `findindex` sum, which the id's
    /// flattened graded rank does not reproduce once a factor has rank ≥ 2 or
    /// a torus charge is present. A warm call reads the interned irrep; a miss
    /// decodes the id.
    fn sector_order_key(&self, id: SectorId) -> SectorOrderKey {
        let key = self.try_sector_order_key(id);
        // Why an admitted sector always has a key: a factor's rank is at most
        // the id (its lower-ranked sub-vectors embed injectively below the
        // full vector) and admitted charges lie within `±i32::MAX`. A
        // congruence-hole id still decodes and gets its excluded label's key;
        // only an undecodable id, or a decoded charge whose position passes
        // `u64`, has none.
        key.unwrap_or_else(|| {
            panic!("Lie-group rule cannot order id {id:?}: it decodes to no label with a key")
        })
    }

    fn encode_channels(
        &self,
        category: &LieGroup,
        _rule: &RuleIdentity,
        [left, right]: [SectorId; 2],
        outcomes: impl Iterator<Item = LieIrrep>,
    ) -> Result<SectorVec, M::Error> {
        let mut channels: SectorVec = outcomes
            .map(|irrep| {
                self.sector(category, Cow::Owned(irrep))
                    .map_err(|error| M::outcome_error(left, right, error))
            })
            .collect::<Result<_, _>>()?;
        channels.sort_unstable();
        Ok(channels)
    }

    fn map_error(&self, inputs: &[SectorId], error: CategoryAdapterError<LieError>) -> M::Error {
        M::adapter_error(inputs, error, |error| self.generic_map_error(inputs, error))
    }
}

impl<M: LieAdmissionMode> AdapterAccess for LieGroupFusionRule<M> {
    type Category = LieGroup;
    type Codec = LieCodec<M>;

    fn category(&self) -> &LieGroup {
        self.category()
    }
    fn codec(&self) -> &LieCodec<M> {
        &self.codec
    }
    fn admit_vertex_move(
        &self,
        operation: crate::CategoryOperation,
        ids: [SectorId; 3],
    ) -> Result<crate::category_rule::AdmittedObjects<LieIrrep, 3>, M::Error> {
        let objects = self
            .codec
            .admit(|| self.category(), &self.identity, operation, ids)?;
        let dualized = match operation {
            crate::CategoryOperation::A => ids.first(),
            crate::CategoryOperation::B => ids.get(1),
            _ => None,
        };
        let dual = dualized
            .map(|&id| {
                let dual_id = self.codec.dual(|| self.category(), id)?;
                self.codec.object(|| self.category(), dual_id)
            })
            .transpose()?;
        Ok((objects, dual))
    }
    fn identity(&self) -> &RuleIdentity {
        &self.identity
    }
    fn vacuum_id(&self) -> SectorId {
        SectorId::new(0)
    }
}

impl<M: 'static> crate::seal::Provider for LieGroupFusionRule<M> {}

crate::category_rule::multiplicity_free_adapter_impls!(
    LieGroupFusionRule<MultiplicityFreeAdmissionMode>,
    LieIrrep,
    f64
);

/// Physical carrier basis of a multiplicity-free group: the Kronecker product
/// of its rank-one factors' SU(2) carriers in Dynkin-label order, the first
/// factor most significant, as TensorKitSectors' `ProductSector`
/// `fusiontensor` composes factors. Torus factors are one-dimensional and
/// contribute no index. Each rank-one carrier follows TensorKitSectors'
/// `SU2Irrep`: zero-based index `k` has doubled projection `dm = dj - 2k`,
/// descending in `m`. That is the index reversal of fusion-symbols'
/// ascending `GroupRepresentations::cgc`; reversing every factor reverses the
/// whole Kronecker index.
///
/// Each element is a product of fusion-symbols' scalar `clebsch_gordan`
/// values, one per rank-one factor, after symtn checks admission,
/// multiplicity and carrier indices; no CGC block is built. A sector with a
/// doubled spin beyond `i32` (fusion-symbols' projection range) or a carrier
/// dimension beyond `usize` returns [`PhysicalBasisError::Unsupported`].
impl PhysicalFusionBasis for LieGroupFusionRule<MultiplicityFreeAdmissionMode> {
    type Scalar = f64;
    type Error = PhysicalBasisError<std::convert::Infallible>;

    fn try_carrier_dimension(&self, sector: SectorId) -> Result<usize, Self::Error> {
        carrier_dimension(sector, &self.decode_sector(sector)?)
    }

    fn try_fusion_tensor_element(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
        left_basis: usize,
        right_basis: usize,
        coupled_basis: usize,
        multiplicity: usize,
    ) -> Result<Self::Scalar, Self::Error> {
        let fusion_multiplicity = self.try_nsymbol(left, right, coupled)?;
        if multiplicity >= fusion_multiplicity {
            return Err(PhysicalBasisError::FusionMultiplicityOutOfBounds {
                left,
                right,
                coupled,
                multiplicity,
                dimension: fusion_multiplicity,
            });
        }
        let [(a, da), (b, db), (c, dc)] = [
            self.checked_carrier(left, left_basis)?,
            self.checked_carrier(right, right_basis)?,
            self.checked_carrier(coupled, coupled_basis)?,
        ];
        let mut remaining = [da, db, dc];
        let mut index = [left_basis, right_basis, coupled_basis];
        let mut element = 1.0;
        for ((ja, jb), jc) in a.dynkin().zip(b.dynkin()).zip(c.dynkin()) {
            let dj = [ja, jb, jc];
            let dm: [i32; 3] = std::array::from_fn(|axis| {
                remaining[axis] /= dj[axis] as usize + 1;
                let k = index[axis] / remaining[axis];
                index[axis] %= remaining[axis];
                // `checked_carrier` bounds `dj` by `i32::MAX`, and `|dj - 2k| <= dj`.
                (dj[axis] - 2 * k as i64) as i32
            });
            // Admitted N and carrier bounds leave only a projection-sum
            // mismatch, where fusion-symbols returns exact zero.
            element *= fusion_symbols::lie::su2::clebsch_gordan(
                ja as u32, dm[0], jb as u32, dm[1], jc as u32, dm[2],
            )
            .to_f64();
        }
        Ok(element)
    }
}

impl LieGroupFusionRule<MultiplicityFreeAdmissionMode> {
    /// The decoded irrep and carrier dimension of `sector`, after checking
    /// that `index` lies in the carrier.
    fn checked_carrier(
        &self,
        sector: SectorId,
        index: usize,
    ) -> Result<(LieIrrep, usize), PhysicalBasisError<std::convert::Infallible>> {
        let irrep = self.decode_sector(sector)?;
        let dimension = carrier_dimension(sector, &irrep)?;
        if index >= dimension {
            return Err(PhysicalBasisError::CarrierIndexOutOfBounds {
                sector,
                index,
                dimension,
            });
        }
        Ok((irrep, dimension))
    }
}

/// Product of the rank-one carrier dimensions `dj + 1`. Why per Dynkin label:
/// multiplicity-free admission leaves only rank-one simple factors (D2 as two
/// A1) and tori, so every Dynkin label is one SU(2) factor.
fn carrier_dimension(
    sector: SectorId,
    irrep: &LieIrrep,
) -> Result<usize, PhysicalBasisError<std::convert::Infallible>> {
    irrep
        .dynkin()
        .try_fold(1usize, |dimension, dj| {
            let dj = i32::try_from(dj).ok()?;
            dimension.checked_mul(usize::try_from(dj).ok()?.checked_add(1)?)
        })
        .ok_or(PhysicalBasisError::Unsupported { sector })
}

impl CheckedGenericFusion for LieGroupFusionRule {
    type Error = LieGroupFusionRuleError;

    fn rule_identity(&self) -> RuleIdentity {
        self.identity.clone()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Generic
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }
    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        self.codec.dual(|| self.category(), sector)
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        forward::fusion_channels(self, left, right)
    }

    fn try_fusion_channels_in_table(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        self.try_fusion_channels(left, right)
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, Self::Error> {
        forward::nsymbol(self, left, right, coupled)
    }

    fn sector_order_key(&self, sector: SectorId) -> SectorOrderKey {
        self.codec.sector_order_key(sector)
    }
}

/// Lie groups use the canonical trivial representation as their tensor unit.
impl CheckedCanonicalUnitFusionRule for LieGroupFusionRule {}

/// The typed label is fusion-symbols' irrep of the rule's group, encoded as by
/// [`LieGroupFusionRule::encode_irrep_with_charges`].
impl TypedSectorAdmission for LieGroupFusionRule {
    type Sector = LieIrrep;
    type Error = LieGroupFusionRuleError;
    type Mode = CheckedGenericAdmissionMode;

    fn typed_rule_identity(&self) -> RuleIdentity {
        CheckedGenericFusion::rule_identity(self)
    }

    fn try_encode_label(&self, sector: &Self::Sector) -> Result<SectorId, Self::Error> {
        self.codec
            .encode_label(|| self.category(), &self.identity, sector)
    }

    fn try_decode_label(&self, sector: SectorId) -> Result<Self::Sector, Self::Error> {
        self.codec.decode_label(|| self.category(), sector)
    }

    fn try_dual_id(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        CheckedGenericFusion::try_dual(self, sector)
    }
}

impl CheckedGenericRigidSymbols for LieGroupFusionRule {
    type Scalar = f64;

    /// The provider's Weyl dimension, exact below `2^53`, rounded once above
    /// it, and [`LieGroupFusionRuleError::DimensionNotRepresentable`] past `f64::MAX`.
    fn try_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        forward::dimension(self, sector, FusionCategory::dim)
    }

    fn try_sqrt_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        forward::dimension(self, sector, FusionCategory::sqrt_dim)
    }

    fn try_inv_sqrt_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        forward::dimension(self, sector, FusionCategory::inv_sqrt_dim)
    }

    fn try_frobenius_schur_phase_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        forward::frobenius_schur_phase(self, sector)
    }

    fn try_f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Result<GenericFArray<f64>, Self::Error> {
        forward::f_block(self, [a, b, c, d, e, f])
    }

    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<f64>, Self::Error> {
        forward::r_block(self, [a, b, c])
    }

    // The existing boundary accepts empty moves before the provider needs
    // dimensions or a dual object; neither must fit an additional symtn id.
    // Validate logical axes without strengthening that admission.
    fn try_a_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<f64>, Self::Error> {
        let ids = [a, b, c];
        let [a, b, c] = crate::category_rule::try_map_array(ids, |id| {
            self.codec.object(|| self.category(), id)
        })?;
        let block = self
            .category()
            .a_symbol(&a, &b, &c)
            .map_err(provider_error)?;
        let rows = self
            .category()
            .nsymbol(&a, &b, &c)
            .map_err(provider_error)? as usize;
        let cols = if rows == 0 {
            0
        } else {
            let dual_id = CheckedGenericFusion::try_dual(self, ids[0])?;
            let dual = self.codec.object(|| self.category(), dual_id)?;
            self.category()
                .nsymbol(&dual, &c, &b)
                .map_err(provider_error)? as usize
        };
        let data = forward::shaped(self, &ids, &block, [rows, cols])?.to_vec();
        Ok(GenericRMatrix::new(data, rows, cols))
    }

    fn try_b_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<f64>, Self::Error> {
        let ids = [a, b, c];
        let [a, b, c] = crate::category_rule::try_map_array(ids, |id| {
            self.codec.object(|| self.category(), id)
        })?;
        let block = self
            .category()
            .b_symbol(&a, &b, &c)
            .map_err(provider_error)?;
        let rows = self
            .category()
            .nsymbol(&a, &b, &c)
            .map_err(provider_error)? as usize;
        let cols = if rows == 0 {
            0
        } else {
            let dual_id = CheckedGenericFusion::try_dual(self, ids[1])?;
            let dual = self.codec.object(|| self.category(), dual_id)?;
            self.category()
                .nsymbol(&c, &dual, &a)
                .map_err(provider_error)? as usize
        };
        let data = forward::shaped(self, &ids, &block, [rows, cols])?.to_vec();
        Ok(GenericRMatrix::new(data, rows, cols))
    }
}

impl CheckedGenericPivotal for LieGroupFusionRule {
    fn try_twist_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        forward::twist(self, sector)
    }
}

/// Rank of `parts` coordinates by total grade, then lexicographically.
/// `None` when a coordinate is unrepresentable or the rank overflows `usize`.
fn graded_lex_rank<I: Iterator<Item = Option<usize>>>(
    parts: usize,
    coordinates: impl Fn() -> I,
) -> Option<usize> {
    // One coordinate is its own rank; this keeps the binomial arithmetic off
    // every one-coordinate fusion channel.
    if parts == 1 {
        return coordinates().next()?;
    }
    let grade = coordinates().try_fold(0usize, |sum, x| sum.checked_add(x?))?;
    let mut rank = count_before_grade(grade, parts, usize::MAX).exact()?;
    let mut remaining = grade;
    for (index, coordinate) in coordinates().enumerate() {
        let coordinate = coordinate?;
        let tail = parts - 1 - index;
        rank = rank.checked_add(
            prefix_count(remaining, tail, coordinate, usize::MAX).and_then(Capped::exact)?,
        )?;
        remaining -= coordinate;
    }
    Some(rank)
}

enum Capped {
    Exact(usize),
    Exceeded,
}
impl Capped {
    fn exact(self) -> Option<usize> {
        if let Self::Exact(value) = self {
            Some(value)
        } else {
            None
        }
    }
    fn is_exceeded(&self) -> bool {
        matches!(self, Self::Exceeded)
    }
}

fn count_before_grade(grade: usize, parts: usize, cap: usize) -> Capped {
    if parts == 0 {
        // The coordinate-free label has grade zero.
        return Capped::Exact(usize::from(grade > 0));
    }
    match grade.checked_add(parts.saturating_sub(1)) {
        Some(n) => binomial(n, parts, cap),
        None => Capped::Exceeded,
    }
}

fn prefix_count(remaining: usize, tail: usize, labels_before: usize, cap: usize) -> Option<Capped> {
    if labels_before == 0 || tail == 0 {
        return Some(Capped::Exact(usize::from(labels_before > remaining)));
    }
    let upper = binomial_u128(remaining.checked_add(tail)?, tail)?;
    let lower = binomial_u128(
        remaining.checked_sub(labels_before)?.checked_add(tail)?,
        tail,
    )?;
    let difference = upper.checked_sub(lower)?;
    Some(if difference > cap as u128 {
        Capped::Exceeded
    } else {
        Capped::Exact(difference as usize)
    })
}

fn binomial(n: usize, k: usize, cap: usize) -> Capped {
    match binomial_u128(n, k) {
        Some(value) if value <= cap as u128 => Capped::Exact(value as usize),
        _ => Capped::Exceeded,
    }
}

const _: () = assert!(usize::BITS <= 64);

// Why not arbitrary precision: for a valid grade g > 0 with r parts,
// B=C(g+r-1,r)<=M. The largest prefix upper is
// U=C(g+r-1,r-1)=B*r/g<=M²<2¹²⁸ for M=usize::MAX; later states decrease.
// A count-before value overflowing u128 is therefore already greater than M.
fn binomial_u128(n: usize, k: usize) -> Option<u128> {
    if k > n {
        return Some(0);
    }
    let k = k.min(n - k);
    let mut result = 1u128;
    for i in 1..=k {
        let mut numerator = (n - k + i) as u128;
        let mut denominator = i as u128;
        let divisor = gcd_u128(numerator, denominator);
        numerator /= divisor;
        denominator /= divisor;
        let divisor = gcd_u128(result, denominator);
        result /= divisor;
        denominator /= divisor;
        if denominator != 1 {
            return None;
        }
        result = result.checked_mul(numerator)?;
    }
    Some(result)
}

const fn gcd_u128(mut left: u128, mut right: u128) -> u128 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

// Why per-test gates: a default build executes rank-one groups, U(2) and tori;
// the SU(3)+ fixtures need fusion-symbols' `lie-generated`.
#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "cgc-gen")]
    use fusion_symbols::lie::RootSystem;

    #[test]
    fn past_capacity_misses_still_admit_through_fusion_symbols() {
        let rule =
            LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(GroupId::u(2).unwrap()).unwrap();
        let mut id = 0;
        while rule
            .codec
            .irreps
            .as_ref()
            .unwrap()
            .0
            .read()
            .unwrap()
            .by_id
            .len()
            < IRREP_TABLE_CAPACITY
        {
            let _ = rule.decode_irrep(SectorId::new(id));
            id += 1;
        }
        // Full table: these labels are never interned and run fusion-symbols per call.
        for _ in 0..2 {
            let sector = rule.encode_irrep_with_charges(&[1], &[1001]).unwrap();
            assert_eq!(rule.codec.irreps.as_ref().unwrap().by_id(sector), None);
            assert_eq!(rule.decode_irrep(sector).unwrap(), (vec![1], vec![1001]));
            assert!(matches!(
                rule.encode_irrep_with_charges(&[1], &[1000]),
                Err(LieGroupFusionRuleError::Provider(LieError::NotAdmissible))
            ));
        }
    }

    #[test]
    fn dual_interner_keeps_both_directions_within_capacity() {
        let table = IrrepTable::default();
        for id in 0..IRREP_TABLE_CAPACITY - 1 {
            table.insert_dual(SectorId::new(id), SectorId::new(id));
        }
        let last = SectorId::new(IRREP_TABLE_CAPACITY - 1);
        let extra = SectorId::new(IRREP_TABLE_CAPACITY);
        table.insert_dual(last, extra);
        assert_eq!(table.dual(last), None);
        assert_eq!(table.dual(extra), None);
        table.insert_dual(last, last);
        assert_eq!(table.dual(last), Some(last));
        assert_eq!(table.0.read().unwrap().duals.len(), IRREP_TABLE_CAPACITY);
    }

    #[cfg(feature = "cgc-gen")]
    #[test]
    fn vertex_admission_rejects_an_unencodable_dual() {
        if usize::BITS != 64 {
            return;
        }
        let rule = LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(GroupId::su(3).unwrap())
            .unwrap();
        let unit = rule.vacuum();
        let sector = SectorId::new(usize::MAX);
        for (operation, ids) in [
            (crate::CategoryOperation::A, [sector, unit, unit]),
            (crate::CategoryOperation::B, [unit, sector, unit]),
        ] {
            for _ in 0..2 {
                assert!(matches!(
                    crate::CategoryProvider::admit(&rule, operation, ids),
                    Err(LieGroupFusionRuleError::UnrepresentableLabel { labels })
                        if labels == [3_327_948_884, 2_746_052_115]
                ));
            }
        }
    }

    #[cfg(feature = "cgc-gen")]
    #[test]
    fn malformed_symbol_axes_map_without_a_source() {
        use std::error::Error as _;
        let rule = LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(GroupId::su(3).unwrap())
            .unwrap();
        let error = rule.codec.map_error(
            &[],
            CategoryAdapterError::MalformedSymbolAxes {
                expected: vec![2, 2],
                actual: vec![1, 4],
            },
        );
        assert!(
            matches!(&error, LieGroupFusionRuleError::MalformedSymbolAxes { expected, actual } if expected == &[2, 2] && actual == &[1, 4])
        );
        assert_eq!(
            error.to_string(),
            "provider symbol axes [1, 4] disagree with the fusion multiplicities [2, 2]"
        );
        assert!(error.source().is_none());
    }

    /// The codec's dual is Dynkin reversal and the twist is exactly one; a
    /// complex SU(4) irrep keeps its non-trivial FS phase.
    #[cfg(feature = "cgc-gen")]
    #[test]
    fn dual_is_dynkin_reversal_and_twist_is_unit() {
        for (n, max_total) in [(2, 6), (3, 3), (4, 2)] {
            let rule =
                LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(GroupId::su(n).unwrap())
                    .unwrap();
            let mut swept = 0;
            for id in 0.. {
                let sector = SectorId::new(id);
                let mut labels = rule.decode_irrep(sector).unwrap().0;
                if labels.iter().sum::<i64>() > max_total {
                    break;
                }
                labels.reverse();
                assert_eq!(
                    rule.try_dual(sector).unwrap(),
                    rule.encode_irrep(&labels).unwrap()
                );
                assert_eq!(
                    rule.try_twist_scalar(sector).unwrap().to_bits(),
                    1.0f64.to_bits()
                );
                swept += 1;
            }
            assert!(swept > n);
        }
        // A complex irrep whose phase is not +1 in this gauge.
        let su4 = LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(GroupId::su(4).unwrap())
            .unwrap();
        let fundamental = su4.encode_irrep(&[1, 0, 0]).unwrap();
        assert_eq!(
            su4.try_frobenius_schur_phase_scalar(fundamental).unwrap(),
            -1.0
        );
    }

    #[cfg(feature = "cgc-gen")]
    #[test]
    fn codec_is_graded_lexicographic_and_roundtrips() {
        let rule = LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(GroupId::su(3).unwrap())
            .unwrap();
        let labels = [[0, 0], [0, 1], [1, 0], [0, 2], [1, 1], [2, 0]];
        for (id, label) in labels.into_iter().enumerate() {
            assert_eq!(rule.encode_irrep(&label).unwrap(), SectorId::new(id));
            assert_eq!(rule.decode_irrep(SectorId::new(id)).unwrap().0, label);
        }
        let su2 = LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(GroupId::su(2).unwrap())
            .unwrap();
        let edge = [i64::from(u32::MAX)];
        let id = su2.encode_irrep(&edge).unwrap();
        assert_eq!(id, SectorId::new(u32::MAX as usize));
        assert_eq!(su2.decode_irrep(id).unwrap().0, edge);
    }

    #[test]
    fn checked_rigid_dimension_accepts_the_largest_su2_dynkin_label() {
        let rule = LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(GroupId::su(2).unwrap())
            .unwrap();
        let sector = rule.encode_irrep(&[i64::from(u32::MAX)]).unwrap();
        let expected_dimension = 2.0_f64.powi(32);
        let sqrt_dimension = rule.try_sqrt_dim_scalar(sector).unwrap();
        assert_eq!(sqrt_dimension, expected_dimension.sqrt());
        assert_eq!(
            rule.try_inv_sqrt_dim_scalar(sector).unwrap(),
            sqrt_dimension.recip()
        );
        assert_eq!(rule.try_dim_scalar(sector).unwrap(), expected_dimension);
        for label in [i64::from(u32::MAX) + 1, i64::MAX] {
            assert!(matches!(
                rule.encode_irrep(&[label]),
                Err(LieGroupFusionRuleError::UnrepresentableLabel { .. })
            ));
            assert!(matches!(
                rule.decode_irrep(SectorId::new(label as usize)),
                Err(LieGroupFusionRuleError::UnrepresentableLabel { .. })
            ));
            assert!(matches!(
                rule.try_dim_scalar(SectorId::new(label as usize)),
                Err(LieGroupFusionRuleError::UnrepresentableLabel { .. })
            ));
        }
    }

    #[cfg(feature = "cgc-gen")]
    #[test]
    fn checked_rigid_dimension_is_the_exact_weyl_dimension() {
        // What: `dim` is not `sqrt(dim)^2`, which is off by one rounding for
        // 3, 6 and 8 (`sqrt(8)^2 = 8.000000000000002`).
        let rule = LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(GroupId::su(3).unwrap())
            .unwrap();
        for p in 0..6_i64 {
            for q in 0..6_i64 {
                let sector = rule.encode_irrep(&[p, q]).unwrap();
                let weyl = (p + 1) * (q + 1) * (p + q + 2) / 2;
                assert_eq!(rule.try_dim_scalar(sector).unwrap(), weyl as f64);
            }
        }
    }

    #[cfg(feature = "cgc-gen")]
    #[test]
    fn codec_handles_bounded_binomial_cancellation() {
        if usize::BITS != 64 {
            return;
        }
        let rule = LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(GroupId::su(39).unwrap())
            .unwrap();
        let sector = SectorId::new(usize::try_from(17_876_288_714_431_443_296u64).unwrap());
        let mut expected = vec![0; 38];
        expected[37] = 31;
        assert_eq!(rule.decode_irrep(sector).unwrap().0, expected);
        assert_eq!(rule.encode_irrep(&expected).unwrap(), sector);
    }

    #[cfg(feature = "cgc-gen")]
    #[test]
    fn validation_and_identity_are_typed_and_stable() {
        assert!(GroupId::su(1).is_err());
        let rule = LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(GroupId::su(3).unwrap())
            .unwrap();
        assert!(matches!(
            rule.encode_irrep(&[1]),
            Err(LieGroupFusionRuleError::LabelLength { .. })
        ));
        assert!(matches!(
            rule.encode_irrep(&[-1, 0]),
            Err(LieGroupFusionRuleError::NegativeLabel { .. })
        ));
        assert!(matches!(
            rule.encode_irrep(&[i64::MAX, i64::MAX]),
            Err(LieGroupFusionRuleError::UnrepresentableLabel { .. })
        ));
        let too_large = GroupId::cover(
            &[RootSystem::A(isize::MAX as usize / size_of::<i64>() + 1)],
            0,
        )
        .unwrap();
        assert!(matches!(
            LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(too_large.clone()),
            Err(LieGroupFusionRuleError::RankNotRepresentable { .. })
        ));
        // Exercise the codec's fallible allocation independently: production
        // rejects this rank before constructing eager provider state.
        let semisimple_rank = too_large.semisimple_rank();
        let codec = LieCodec::<CheckedGenericAdmissionMode> {
            group: Arc::new(too_large),
            irreps: Some(Arc::default()),
            semisimple_rank,
            coordinate_count: semisimple_rank,
            mode: PhantomData,
        };
        assert!(matches!(
            codec.decode_coordinates(SectorId::new(0)),
            Err(LieGroupFusionRuleError::LabelAllocation { .. })
        ));
        for n in [3, 4] {
            let rule =
                LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(GroupId::su(n).unwrap())
                    .unwrap();
            let labels = rule.decode_irrep(SectorId::new(usize::MAX)).unwrap().0;
            assert_eq!(
                rule.encode_irrep(&labels).unwrap(),
                SectorId::new(usize::MAX)
            );
        }
        assert!(matches!(
            LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(GroupId::su(2).unwrap())
                .unwrap()
                .decode_irrep(SectorId::new(usize::MAX)),
            Err(LieGroupFusionRuleError::DecodeOverflow { .. })
        ));
        // Identity follows `GroupId` equality, not the constructor. The rules
        // stay alive: an identity whose rules are all dropped equals nothing
        // but its clones.
        let quotient = LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(
            GroupId::su_quotient(3, 1).unwrap(),
        )
        .unwrap();
        assert_eq!(rule.rule_identity(), quotient.rule_identity());
        let psu3 = LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(GroupId::psu(3).unwrap())
            .unwrap();
        assert_ne!(psu3.rule_identity(), rule.rule_identity());
    }

    #[cfg(feature = "cgc-gen")]
    #[test]
    fn stored_identity_is_shared_and_distinguishes_rank() {
        let su3 = LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(GroupId::su(3).unwrap())
            .unwrap();
        let su4 = LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(GroupId::su(4).unwrap())
            .unwrap();
        assert_eq!(su3.rule_identity(), su3.clone().rule_identity());
        assert_ne!(su3.rule_identity(), su4.rule_identity());
        let hash = |rule: &LieGroupFusionRule| {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            rule.rule_identity().hash(&mut hasher);
            hasher.finish()
        };
        assert_ne!(hash(&su3), hash(&su4));
    }

    #[cfg(feature = "cgc-gen")]
    #[test]
    fn irreps_and_duals_are_interned_once_per_sector() {
        let rule = LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(GroupId::su(3).unwrap())
            .unwrap();
        let three = rule.encode_irrep(&[1, 0]).unwrap();
        let eight = rule.encode_irrep(&[1, 1]).unwrap();
        let table = rule.codec.irreps.as_ref().unwrap();
        let interned = || table.0.read().unwrap().by_id.len();
        let first = rule.codec.irrep(rule.category(), three).unwrap();
        let before = interned();
        assert_eq!(rule.codec.irrep(rule.category(), three).unwrap(), first);
        let cloned_rule = rule.clone();
        assert!(Arc::ptr_eq(
            table,
            cloned_rule.codec.irreps.as_ref().unwrap()
        ));
        assert_eq!(
            cloned_rule
                .codec
                .irrep(cloned_rule.category(), three)
                .unwrap(),
            first
        );
        assert_eq!(interned(), before);
        assert_eq!(first, rule.category().irrep(&[1, 0]).unwrap());
        assert_ne!(rule.codec.irrep(rule.category(), eight).unwrap(), first);
        assert_eq!(
            rule.codec
                .sector(rule.category(), Cow::Borrowed(&first))
                .unwrap(),
            three
        );

        let anti_three = rule.encode_irrep(&[0, 1]).unwrap();
        assert_eq!(table.dual(three), None);
        assert_eq!(rule.try_dual(three).unwrap(), anti_three);
        assert_eq!(table.dual(three), Some(anti_three));
        assert_eq!(table.dual(anti_three), Some(three));

        // Failed conversions are not interned.
        let su2 = LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(GroupId::su(2).unwrap())
            .unwrap();
        let su4 = LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(GroupId::su(4).unwrap())
            .unwrap();
        let wide = su4.codec.irrep(su4.category(), three).unwrap();
        assert!(su2
            .codec
            .sector(su2.category(), Cow::Borrowed(&wide))
            .is_err());
        assert!(su2
            .codec
            .sector(su2.category(), Cow::Borrowed(&wide))
            .is_err());
    }

    #[cfg(feature = "cgc-gen")]
    #[test]
    fn su3_octet_self_fusion_has_dynkin_ordered_channels_and_multiplicities() {
        let rule = LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(GroupId::su(3).unwrap())
            .unwrap();
        let eight = rule.encode_irrep(&[1, 1]).unwrap();
        assert_eq!(rule.fusion_style(), FusionStyleKind::Generic);
        assert_eq!(rule.braiding_style(), BraidingStyleKind::Bosonic);
        assert_eq!(rule.try_nsymbol(eight, eight, eight).unwrap(), 2);
        assert_eq!(
            rule.try_dual(rule.encode_irrep(&[2, 1]).unwrap()).unwrap(),
            rule.encode_irrep(&[1, 2]).unwrap()
        );
        let channels = rule.try_fusion_channels(eight, eight).unwrap();
        assert_eq!(
            channels
                .iter()
                .map(|sector| sector.id())
                .collect::<Vec<_>>(),
            [0, 4, 6, 9, 12]
        );
        assert_eq!(
            channels
                .iter()
                .map(|&sector| rule.decode_irrep(sector).unwrap().0)
                .collect::<Vec<_>>(),
            [[0, 0], [1, 1], [0, 3], [3, 0], [2, 2]]
        );
        assert_eq!(
            rule.try_fusion_channels_in_table(eight, eight).unwrap(),
            channels
        );
        for (labels, multiplicity) in [
            ([0, 0], 1),
            ([1, 1], 2),
            ([0, 3], 1),
            ([3, 0], 1),
            ([2, 2], 1),
        ] {
            assert_eq!(
                rule.try_nsymbol(eight, eight, rule.encode_irrep(&labels).unwrap())
                    .unwrap(),
                multiplicity
            );
        }
        assert_eq!(
            rule.try_nsymbol(eight, eight, rule.encode_irrep(&[1, 0]).unwrap())
                .unwrap(),
            0
        );
    }

    #[cfg(feature = "cgc-gen")]
    #[test]
    fn su3_rigid_symbols_match_sunrepresentations_fixtures() {
        let rule = LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(GroupId::su(3).unwrap())
            .unwrap();
        let three = rule.encode_irrep(&[1, 0]).unwrap();
        let anti_three = rule.encode_irrep(&[0, 1]).unwrap();
        let eight = rule.encode_irrep(&[1, 1]).unwrap();

        for (sector, dimension) in [(three, 3.0), (anti_three, 3.0), (eight, 8.0)] {
            let sqrt = rule.try_sqrt_dim_scalar(sector).unwrap();
            assert!((sqrt * sqrt - dimension).abs() < 1e-12);
            assert!((sqrt * rule.try_inv_sqrt_dim_scalar(sector).unwrap() - 1.0).abs() < 1e-12);
            assert_ne!(rule.try_frobenius_schur_phase_scalar(sector).unwrap(), 0.0);
            assert_eq!(rule.try_twist_scalar(sector).unwrap(), 1.0);
        }

        let f = rule
            .try_f_symbol_generic(eight, eight, eight, eight, eight, eight)
            .unwrap();
        assert_eq!(f.shape(), (2, 2, 2, 2));
        for (index, expected) in [
            ((0, 0, 0, 0), 0.857_142_857_142_856),
            ((0, 0, 1, 1), -0.142_857_142_857_142_63),
            ((0, 1, 1, 1), -0.383_325_938_999_963_37),
            ((1, 1, 1, 1), 0.628_571_428_571_427_4),
        ] {
            let (mu, nu, kappa, lambda) = index;
            assert!((*f.get(mu, nu, kappa, lambda) - expected).abs() < 1e-12);
        }

        let twenty_seven = rule.encode_irrep(&[2, 2]).unwrap();
        let asymmetric = rule
            .try_f_symbol_generic(
                twenty_seven,
                twenty_seven,
                anti_three,
                anti_three,
                eight,
                rule.encode_irrep(&[3, 1]).unwrap(),
            )
            .unwrap();
        assert_eq!(asymmetric.shape(), (2, 1, 1, 1));
        assert!((*asymmetric.get(1, 0, 0, 0) - 0.797_724_035_217_465_4).abs() < 1e-12);

        let r = rule.try_r_symbol_generic(eight, eight, eight).unwrap();
        assert_eq!(r.shape(), (2, 2));
        for (index, expected) in [
            ((0, 0), -0.285_714_285_714_285_3),
            ((0, 1), 0.958_314_847_499_908_8),
            ((1, 0), 0.958_314_847_499_908_9),
            ((1, 1), 0.285_714_285_714_285_53),
        ] {
            assert!((*r.get(index.0, index.1) - expected).abs() < 1e-12);
        }

        let left = rule.encode_irrep(&[2, 2]).unwrap();
        let right = rule.encode_irrep(&[2, 3]).unwrap();
        let asymmetric_r = rule.try_r_symbol_generic(left, right, right).unwrap();
        assert_eq!(asymmetric_r.shape(), (3, 3));
        for (index, expected) in [
            ((0, 1), -0.439_764_978_446_187_45),
            ((1, 0), -0.374_954_863_181_616_1),
            ((0, 2), 0.887_615_402_640_999_9),
            ((2, 0), 0.916_876_867_332_670_7),
        ] {
            assert!((*asymmetric_r.get(index.0, index.1) - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn su2_fundamental_pivotal_phase_matches_f_gauge() {
        let rule = LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(GroupId::su(2).unwrap())
            .unwrap();
        let fundamental = rule.encode_irrep(&[1]).unwrap();
        assert_eq!(
            rule.try_frobenius_schur_phase_scalar(fundamental).unwrap(),
            -1.0
        );
    }

    #[cfg(feature = "cgc-gen")]
    #[test]
    fn su4_adjoint_rigid_shape_smoke() {
        let rule = LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(GroupId::su(4).unwrap())
            .unwrap();
        let adjoint = rule.encode_irrep(&[1, 0, 1]).unwrap();
        assert_eq!(rule.try_dim_scalar(adjoint).unwrap(), 15.0);
        assert_eq!(
            rule.try_f_symbol_generic(adjoint, adjoint, adjoint, adjoint, adjoint, adjoint)
                .unwrap()
                .shape(),
            (2, 2, 2, 2)
        );
        assert_eq!(
            rule.try_r_symbol_generic(adjoint, adjoint, adjoint)
                .unwrap()
                .shape(),
            (2, 2)
        );
    }

    #[cfg(feature = "cgc-gen")]
    #[test]
    fn rigid_failures_remain_typed() {
        let rule = LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(GroupId::su(3).unwrap())
            .unwrap();
        let three = rule.encode_irrep(&[1, 0]).unwrap();
        let eight = rule.encode_irrep(&[1, 1]).unwrap();
        let r = rule.try_r_symbol_generic(three, three, eight).unwrap();
        assert_eq!(r.shape(), (0, 0));
        assert!(r.data().is_empty());
        let f = rule
            .try_f_symbol_generic(three, three, three, three, eight, three)
            .unwrap();
        assert_eq!(f.shape(), (0, 1, 0, 0));
        assert!(f.data().is_empty());

        let rule = LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(GroupId::su(78).unwrap())
            .unwrap();
        let mut labels = vec![0; 77];
        labels[36] = 19;
        let sector = rule.encode_irrep(&labels).unwrap();
        assert!(matches!(
            rule.try_sqrt_dim_scalar(sector),
            Err(LieGroupFusionRuleError::DimensionNotRepresentable { .. })
        ));
    }
}
