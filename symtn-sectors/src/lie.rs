use fusion_symbols::{
    category::FusionCategory,
    lie::{CenterSubgroup, GlobalForm, GroupId, LieError, LieGroup, LieIrrep, RootSystem},
};
use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, PoisonError, RwLock};

use crate::category_rule::{forward, AdapterAccess};
use crate::{
    BraidingStyleKind, CanonicalUnitAdmission, CategoryAdapterError, CategoryCodec,
    CheckedCanonicalUnitFusionRule, CheckedGenericAdmissionMode, CheckedGenericFusion,
    CheckedGenericPivotal, CheckedGenericRigidSymbols, FusionStyleKind, GenericFArray,
    GenericRMatrix, RuleIdentity, SectorId, SectorVec, SymbolShapeError, TypedSectorAdmission,
};

const CODEC_VERSION: &[u8] = b"symtn:lie:dynkin:graded-total-then-lex:v1";
const IDENTITY_SCHEMA: u64 = 0x4c49_455f_434f_4445;

/// Checked compact Lie-group rule over the fusion-symbols category adapter.
///
/// The private Dynkin codec retains graded-total-then-lex ids, encoded-id
/// channel ordering, bounded interning and eager category construction.
/// Every rank, including rank one, uses checked Generic admission; no physical
/// expansion is provided.
#[derive(Clone, Debug)]
pub struct LieGroupFusionRule {
    codec: LieCodec,
    identity: RuleIdentity,
    category: LieGroup,
}

#[derive(Clone, Debug)]
pub(crate) struct LieCodec {
    group: GroupId,
    irreps: Arc<IrrepTable>,
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
    fn by_id(&self, sector: SectorId) -> Option<Arc<LieIrrep>> {
        let maps = self.0.read().unwrap_or_else(PoisonError::into_inner);
        maps.by_id.get(&sector).cloned()
    }

    fn by_irrep(&self, irrep: &LieIrrep) -> Option<SectorId> {
        let maps = self.0.read().unwrap_or_else(PoisonError::into_inner);
        maps.by_irrep.get(irrep).copied()
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
    fn insert(&self, sector: SectorId, irrep: Arc<LieIrrep>) -> Arc<LieIrrep> {
        let mut maps = self.0.write().unwrap_or_else(PoisonError::into_inner);
        if let Some(existing) = maps.by_id.get(&sector) {
            return existing.clone();
        }
        if maps.by_id.len() < IRREP_TABLE_CAPACITY {
            maps.by_id.insert(sector, irrep.clone());
            maps.by_irrep.insert(irrep.clone(), sector);
        }
        irrep
    }
}

// The identity and the irrep table are pure functions of `GroupId`, so equality and
// hashing ignore them.
impl PartialEq for LieGroupFusionRule {
    fn eq(&self, other: &Self) -> bool {
        self.group_id() == other.group_id()
    }
}
impl Eq for LieGroupFusionRule {}
impl std::hash::Hash for LieGroupFusionRule {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.group_id().hash(state);
    }
}

/// Failure at the Lie-group label/provider boundary.
#[derive(Debug)]
pub enum LieGroupFusionRuleError {
    RankNotRepresentable {
        rank: usize,
    },
    LabelAllocation {
        len: usize,
        source: std::collections::TryReserveError,
    },
    LabelLength {
        expected: usize,
        found: usize,
    },
    NegativeLabel {
        index: usize,
        value: i64,
    },
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
            Self::RankNotRepresentable { rank } => {
                write!(f, "rank {rank} Dynkin labels exceed this platform's Vec limit")
            }
            Self::LabelAllocation { len, .. } => {
                write!(f, "could not allocate {len} Dynkin labels")
            }
            Self::LabelLength { expected, found } => {
                write!(f, "expected {expected} Dynkin labels, got {found}")
            }
            Self::NegativeLabel { index, value } => {
                write!(f, "Dynkin label {index} is negative ({value})")
            }
            Self::UnrepresentableLabel { labels } => {
                write!(
                    f,
                    "Dynkin label is not representable as a SectorId: {labels:?}"
                )
            }
            Self::DecodeOverflow { sector } => {
                write!(
                    f,
                    "SectorId {} cannot be decoded by this Dynkin codec",
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

impl LieGroupFusionRule {
    /// Constructs a rule and eagerly checks provider support for the full group.
    /// Rank-one labels span `0..=u32::MAX`; larger labels return
    /// [`LieGroupFusionRuleError::UnrepresentableLabel`]. Other ranks use the
    /// target's `usize` ID budget. Excluded global-form labels leave ID holes.
    pub fn new(group: GroupId) -> Result<Self, LieGroupFusionRuleError> {
        let rank = group.root_system.rank();
        if rank > isize::MAX as usize / size_of::<i64>() {
            return Err(LieGroupFusionRuleError::RankNotRepresentable { rank });
        }
        let category = LieGroup::new(group).map_err(provider_error)?;
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&(IDENTITY_SCHEMA, group), &mut hasher);
        let identity = RuleIdentity::from_canonical_bytes::<Self>(
            std::hash::Hasher::finish(&hasher),
            Self::identity_bytes(group, &category)?,
        );
        Ok(Self {
            codec: LieCodec {
                group,
                irreps: Arc::default(),
            },
            identity,
            category,
        })
    }

    fn category(&self) -> &LieGroup {
        &self.category
    }

    /// Returns the complete root datum, including the global form.
    pub const fn group_id(&self) -> GroupId {
        self.codec.group
    }

    /// Encodes admitted nonnegative Dynkin labels by total grade then lexicographic order.
    pub fn encode_dynkin(&self, labels: &[i64]) -> Result<SectorId, LieGroupFusionRuleError> {
        self.codec.encode_dynkin(labels)
    }

    /// Decodes an admitted ID with bounded binary searches over the grade and each label.
    pub fn decode_dynkin(&self, sector: SectorId) -> Result<Vec<i64>, LieGroupFusionRuleError> {
        self.codec.decode_dynkin(sector)
    }

    fn identity_bytes(
        group: GroupId,
        category: &LieGroup,
    ) -> Result<Arc<[u8]>, LieGroupFusionRuleError> {
        let (root_tag, rank) = match group.root_system {
            RootSystem::A(r) => (0, r),
            RootSystem::B(r) => (1, r),
            RootSystem::C(r) => (2, r),
            RootSystem::D(r) => (3, r),
            RootSystem::G2 => (4, 2),
            RootSystem::F4 => (5, 4),
            RootSystem::E6 => (6, 6),
            RootSystem::E7 => (7, 7),
            RootSystem::E8 => (8, 8),
            _ => {
                return Err(provider_error(LieError::Unsupported {
                    requires: "root-system identity encoding",
                }))
            }
        };
        let (form_tag, parameter) = match group.form {
            GlobalForm::SimplyConnected => (0, 0),
            GlobalForm::Quotient(subgroup) => match subgroup {
                CenterSubgroup::Zk(k) => (1, k),
                CenterSubgroup::Z2 => (2, 0),
                CenterSubgroup::Z4 => (3, 0),
                CenterSubgroup::DVector => (4, 0),
                CenterSubgroup::DHalfSpinPlus => (5, 0),
                CenterSubgroup::DHalfSpinMinus => (6, 0),
                CenterSubgroup::DFull => (7, 0),
                _ => {
                    return Err(provider_error(LieError::Unsupported {
                        requires: "global-form identity encoding",
                    }))
                }
            },
        };
        let mut bytes = CODEC_VERSION.to_vec();
        bytes.extend_from_slice(&usize::BITS.to_le_bytes());
        bytes.push(root_tag);
        bytes.extend_from_slice(&rank.to_le_bytes());
        bytes.push(form_tag);
        bytes.extend_from_slice(&parameter.to_le_bytes());
        bytes.extend_from_slice(b":generic:bosonic:rank-one-max=u32::MAX");
        bytes.extend_from_slice(
            b":rigid=f64-finite-exact-dim:f-axes=mu-nu-kappa-lambda:r-axes=mu-nu:pivotal=f-sign-0000",
        );
        bytes.extend_from_slice(category.authority_fingerprint());
        Ok(Arc::from(bytes))
    }
}

impl LieCodec {
    fn encode_dynkin(&self, labels: &[i64]) -> Result<SectorId, LieGroupFusionRuleError> {
        self.check_labels(labels.iter().copied())?;
        if !self.group.admits(labels) {
            return Err(provider_error(LieError::NotAdmissible));
        }
        self.encode_dynkin_iter(|| labels.iter().copied())
    }

    // Validate, then compute the grade and lexicographic rank without materializing labels.
    fn encode_dynkin_iter<I: ExactSizeIterator<Item = i64>>(
        &self,
        labels: impl Fn() -> I,
    ) -> Result<SectorId, LieGroupFusionRuleError> {
        self.check_labels(labels())?;
        let unrepresentable = || LieGroupFusionRuleError::UnrepresentableLabel {
            labels: labels().collect(),
        };
        let grade = labels()
            .try_fold(0usize, |sum, x| sum.checked_add(usize::try_from(x).ok()?))
            .ok_or_else(unrepresentable)?;
        let mut rank = count_before_grade(grade, self.group.root_system.rank(), usize::MAX)
            .exact()
            .ok_or_else(unrepresentable)?;
        let mut remaining = grade;
        for (index, label) in labels().enumerate() {
            let label = label as usize;
            let tail = self.group.root_system.rank() - 1 - index;
            rank = rank
                .checked_add(
                    prefix_count(remaining, tail, label, usize::MAX)
                        .and_then(Capped::exact)
                        .ok_or_else(unrepresentable)?,
                )
                .ok_or_else(unrepresentable)?;
            remaining -= label;
        }
        Ok(SectorId::new(rank))
    }

    /// Decodes a `SectorId` with bounded binary searches over the grade and each label.
    /// Rank-one ids above `u32::MAX` are unrepresentable.
    fn decode_dynkin(&self, sector: SectorId) -> Result<Vec<i64>, LieGroupFusionRuleError> {
        let id = sector.id();
        let r = self.group.root_system.rank();
        let mut low = 0usize;
        let mut high = id;
        while low < high {
            let mid = low + (high - low) / 2;
            if count_before_grade(mid + 1, r, id).is_exceeded() {
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        let grade = low;
        let before = if grade == 0 {
            0
        } else {
            count_before_grade(grade, r, id)
                .exact()
                .ok_or(LieGroupFusionRuleError::DecodeOverflow { sector })?
        };
        let mut residual = id - before;
        let mut remaining = grade;
        let mut labels = Vec::new();
        labels
            .try_reserve_exact(r)
            .map_err(|source| LieGroupFusionRuleError::LabelAllocation { len: r, source })?;
        for index in 0..r {
            let tail = r - index - 1;
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
            labels.push(
                i64::try_from(low)
                    .map_err(|_| LieGroupFusionRuleError::DecodeOverflow { sector })?,
            );
            remaining -= low;
        }
        if residual != 0 {
            return Err(LieGroupFusionRuleError::DecodeOverflow { sector });
        }
        self.check_labels(labels.iter().copied())?;
        if !self.group.admits(&labels) {
            return Err(provider_error(LieError::NotAdmissible));
        }
        Ok(labels)
    }

    fn check_labels(
        &self,
        labels: impl ExactSizeIterator<Item = i64>,
    ) -> Result<(), LieGroupFusionRuleError> {
        if labels.len() != self.group.root_system.rank() {
            return Err(LieGroupFusionRuleError::LabelLength {
                expected: self.group.root_system.rank(),
                found: labels.len(),
            });
        }
        for (index, value) in labels.enumerate() {
            if value < 0 {
                return Err(LieGroupFusionRuleError::NegativeLabel { index, value });
            }
            if self.group.root_system.rank() == 1 && value > i64::from(u32::MAX) {
                return Err(LieGroupFusionRuleError::UnrepresentableLabel {
                    labels: vec![value],
                });
            }
        }
        Ok(())
    }

    fn irrep<'a>(
        &self,
        category: impl Fn() -> &'a LieGroup,
        sector: SectorId,
    ) -> Result<Arc<LieIrrep>, LieGroupFusionRuleError> {
        if let Some(irrep) = self.irreps.by_id(sector) {
            return Ok(irrep);
        }
        let labels = self.decode_dynkin(sector)?;
        let irrep = category().irrep(&labels).map_err(provider_error)?;
        Ok(self.irreps.insert(sector, Arc::new(irrep)))
    }

    fn sector(
        &self,
        category: &LieGroup,
        irrep: Cow<'_, LieIrrep>,
    ) -> Result<SectorId, LieGroupFusionRuleError> {
        if !category.contains(&irrep) {
            return Err(provider_error(LieError::InvalidLabel));
        }
        if let Some(sector) = self.irreps.by_irrep(&irrep) {
            return Ok(sector);
        }
        let sector = self.encode_dynkin_iter(|| irrep.dynkin_labels())?;
        self.irreps.insert(sector, Arc::new(irrep.into_owned()));
        Ok(sector)
    }

    fn dual<'a>(
        &self,
        category: impl Fn() -> &'a LieGroup + Copy,
        sector: SectorId,
    ) -> Result<SectorId, LieGroupFusionRuleError> {
        if let Some(dual) = self.irreps.dual(sector) {
            return Ok(dual);
        }
        let object = crate::category_rule::member_object(self, category, sector)?;
        let dual = self.sector(category(), Cow::Owned(category().dual(&object)))?;
        self.irreps.insert_dual(sector, dual);
        Ok(dual)
    }
}

impl CategoryCodec<LieGroup> for LieCodec {
    type Sector = Vec<i64>;
    type Error = LieGroupFusionRuleError;
    type Mode = CheckedGenericAdmissionMode;
    type Unit = CanonicalUnitAdmission;

    fn encode_label<'a>(
        &self,
        _category: impl Fn() -> &'a LieGroup + Copy,
        _rule: &RuleIdentity,
        label: &Vec<i64>,
    ) -> Result<SectorId, LieGroupFusionRuleError> {
        self.encode_dynkin(label)
    }

    fn decode_label<'a>(
        &self,
        _category: impl Fn() -> &'a LieGroup + Copy,
        id: SectorId,
    ) -> Result<Vec<i64>, LieGroupFusionRuleError> {
        self.decode_dynkin(id)
    }

    fn object<'a>(
        &self,
        category: impl Fn() -> &'a LieGroup + Copy,
        id: SectorId,
    ) -> Result<LieIrrep, LieGroupFusionRuleError> {
        self.irrep(category, id).map(|irrep| (*irrep).clone())
    }

    fn sector_id<'a>(
        &self,
        category: impl Fn() -> &'a LieGroup + Copy,
        _rule: &RuleIdentity,
        object: &LieIrrep,
    ) -> Result<SectorId, LieGroupFusionRuleError> {
        self.sector(category(), Cow::Borrowed(object))
    }

    fn admit<'a, const N: usize>(
        &self,
        category: impl Fn() -> &'a LieGroup + Copy,
        _rule: &RuleIdentity,
        operation: crate::CategoryOperation,
        ids: [SectorId; N],
    ) -> Result<[LieIrrep; N], LieGroupFusionRuleError> {
        let objects = crate::category_rule::try_map_array(ids, |id| {
            crate::category_rule::member_object(self, category, id)
        })?;
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

    fn encode_channels(
        &self,
        category: &LieGroup,
        _rule: &RuleIdentity,
        _inputs: [SectorId; 2],
        outcomes: impl Iterator<Item = LieIrrep>,
    ) -> Result<SectorVec, LieGroupFusionRuleError> {
        let mut channels: SectorVec = outcomes
            .map(|irrep| self.sector(category, Cow::Owned(irrep)))
            .collect::<Result<_, _>>()?;
        channels.sort_unstable();
        Ok(channels)
    }

    fn map_error(
        &self,
        inputs: &[SectorId],
        error: CategoryAdapterError<fusion_symbols::lie::LieError>,
    ) -> LieGroupFusionRuleError {
        match error {
            CategoryAdapterError::Provider(fusion_symbols::lie::LieError::LabelOverflow)
                if self.group.root_system.rank() == 1
                    && matches!(inputs, [left, right] if left.id() <= u32::MAX as usize && right.id() <= u32::MAX as usize) =>
            {
                LieGroupFusionRuleError::UnrepresentableLabel {
                    labels: vec![inputs[0].id() as i64 + inputs[1].id() as i64],
                }
            }
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

impl AdapterAccess for LieGroupFusionRule {
    type Category = LieGroup;
    type Codec = LieCodec;

    fn category(&self) -> &LieGroup {
        self.category()
    }
    fn codec(&self) -> &LieCodec {
        &self.codec
    }
    fn admit_vertex_move(
        &self,
        operation: crate::CategoryOperation,
        ids: [SectorId; 3],
    ) -> Result<crate::category_rule::AdmittedObjects<LieIrrep, 3>, LieGroupFusionRuleError> {
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

impl crate::seal::Provider for LieGroupFusionRule {}

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
}

/// Lie groups use the canonical trivial representation as their tensor unit.
impl CheckedCanonicalUnitFusionRule for LieGroupFusionRule {}

impl TypedSectorAdmission for LieGroupFusionRule {
    type Sector = Vec<i64>;
    type Error = LieGroupFusionRuleError;
    type Mode = CheckedGenericAdmissionMode;

    fn typed_rule_identity(&self) -> RuleIdentity {
        CheckedGenericFusion::rule_identity(self)
    }

    fn try_encode_label(&self, sector: &Self::Sector) -> Result<SectorId, Self::Error> {
        self.encode_dynkin(sector)
    }

    fn try_decode_label(&self, sector: SectorId) -> Result<Self::Sector, Self::Error> {
        self.decode_dynkin(sector)
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
            .try_nsymbol(&a, &b, &c)
            .map_err(provider_error)? as usize;
        let cols = if rows == 0 {
            0
        } else {
            let dual_id = CheckedGenericFusion::try_dual(self, ids[0])?;
            let dual = self.codec.object(|| self.category(), dual_id)?;
            self.category()
                .try_nsymbol(&dual, &c, &b)
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
            .try_nsymbol(&a, &b, &c)
            .map_err(provider_error)? as usize;
        let cols = if rows == 0 {
            0
        } else {
            let dual_id = CheckedGenericFusion::try_dual(self, ids[1])?;
            let dual = self.codec.object(|| self.category(), dual_id)?;
            self.category()
                .try_nsymbol(&c, &dual, &a)
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

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn vertex_admission_rejects_an_unencodable_dual() {
        if usize::BITS != 64 {
            return;
        }
        let rule = LieGroupFusionRule::new(GroupId::su(3).unwrap()).unwrap();
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

    #[test]
    fn malformed_symbol_axes_map_without_a_source() {
        use std::error::Error as _;
        let rule = LieGroupFusionRule::new(GroupId::su(3).unwrap()).unwrap();
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

    /// The provider's FS phase, dual and twist must match sign F[a,ā,a,a;1,1],
    /// Dynkin reversal and constant unit twist, respectively, bit for bit.
    #[test]
    fn rigid_data_matches_f_symbol_and_dynkin_derivations() {
        for (n, max_total) in [(2, 6), (3, 3), (4, 2)] {
            let rule = LieGroupFusionRule::new(GroupId::su(n).unwrap()).unwrap();
            let unit = rule.category().unit();
            let mut swept = 0;
            for id in 0.. {
                let sector = SectorId::new(id);
                let mut labels = rule.decode_dynkin(sector).unwrap();
                if labels.iter().sum::<i64>() > max_total {
                    break;
                }
                let a = rule.category().irrep(&labels).unwrap();
                let dual = rule.category().dual(&a);
                let block = rule
                    .category()
                    .f_symbol(&a, &dual, &a, &a, &unit, &unit)
                    .unwrap();
                assert_eq!(block.dims(), [1, 1, 1, 1]);
                let f_sign = block.at(0, 0, 0, 0).signum();
                assert_eq!(
                    rule.try_frobenius_schur_phase_scalar(sector)
                        .unwrap()
                        .to_bits(),
                    f_sign.to_bits(),
                    "SU({n}) {labels:?}"
                );
                labels.reverse();
                assert_eq!(
                    rule.try_dual(sector).unwrap(),
                    rule.encode_dynkin(&labels).unwrap()
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
        let su4 = LieGroupFusionRule::new(GroupId::su(4).unwrap()).unwrap();
        let fundamental = su4.encode_dynkin(&[1, 0, 0]).unwrap();
        assert_eq!(
            su4.try_frobenius_schur_phase_scalar(fundamental).unwrap(),
            -1.0
        );
    }

    #[test]
    fn codec_is_graded_lexicographic_and_roundtrips() {
        let rule = LieGroupFusionRule::new(GroupId::su(3).unwrap()).unwrap();
        let labels = [[0, 0], [0, 1], [1, 0], [0, 2], [1, 1], [2, 0]];
        for (id, label) in labels.into_iter().enumerate() {
            assert_eq!(rule.encode_dynkin(&label).unwrap(), SectorId::new(id));
            assert_eq!(rule.decode_dynkin(SectorId::new(id)).unwrap(), label);
        }
        let su2 = LieGroupFusionRule::new(GroupId::su(2).unwrap()).unwrap();
        let edge = [i64::from(u32::MAX)];
        let id = su2.encode_dynkin(&edge).unwrap();
        assert_eq!(id, SectorId::new(u32::MAX as usize));
        assert_eq!(su2.decode_dynkin(id).unwrap(), edge);
    }

    #[test]
    fn checked_rigid_dimension_accepts_the_largest_su2_dynkin_label() {
        let rule = LieGroupFusionRule::new(GroupId::su(2).unwrap()).unwrap();
        let sector = rule.encode_dynkin(&[i64::from(u32::MAX)]).unwrap();
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
                rule.encode_dynkin(&[label]),
                Err(LieGroupFusionRuleError::UnrepresentableLabel { .. })
            ));
            assert!(matches!(
                rule.decode_dynkin(SectorId::new(label as usize)),
                Err(LieGroupFusionRuleError::UnrepresentableLabel { .. })
            ));
            assert!(matches!(
                rule.try_dim_scalar(SectorId::new(label as usize)),
                Err(LieGroupFusionRuleError::UnrepresentableLabel { .. })
            ));
        }
    }

    #[test]
    fn checked_rigid_dimension_is_the_exact_weyl_dimension() {
        // What: `dim` is not `sqrt(dim)^2`, which is off by one rounding for
        // 3, 6 and 8 (`sqrt(8)^2 = 8.000000000000002`).
        let rule = LieGroupFusionRule::new(GroupId::su(3).unwrap()).unwrap();
        for p in 0..6_i64 {
            for q in 0..6_i64 {
                let sector = rule.encode_dynkin(&[p, q]).unwrap();
                let weyl = (p + 1) * (q + 1) * (p + q + 2) / 2;
                assert_eq!(rule.try_dim_scalar(sector).unwrap(), weyl as f64);
            }
        }
    }

    #[test]
    fn codec_handles_bounded_binomial_cancellation() {
        if usize::BITS != 64 {
            return;
        }
        let rule = LieGroupFusionRule::new(GroupId::su(39).unwrap()).unwrap();
        let sector = SectorId::new(usize::try_from(17_876_288_714_431_443_296u64).unwrap());
        let mut expected = vec![0; 38];
        expected[37] = 31;
        assert_eq!(rule.decode_dynkin(sector).unwrap(), expected);
        assert_eq!(rule.encode_dynkin(&expected).unwrap(), sector);
    }

    #[test]
    fn validation_and_identity_are_typed_and_stable() {
        assert!(GroupId::su(1).is_err());
        let rule = LieGroupFusionRule::new(GroupId::su(3).unwrap()).unwrap();
        assert!(matches!(
            rule.encode_dynkin(&[1]),
            Err(LieGroupFusionRuleError::LabelLength { .. })
        ));
        assert!(matches!(
            rule.encode_dynkin(&[-1, 0]),
            Err(LieGroupFusionRuleError::NegativeLabel { .. })
        ));
        assert!(matches!(
            rule.encode_dynkin(&[i64::MAX, i64::MAX]),
            Err(LieGroupFusionRuleError::UnrepresentableLabel { .. })
        ));
        let too_large = GroupId {
            root_system: RootSystem::A(isize::MAX as usize / size_of::<i64>() + 1),
            form: GlobalForm::SimplyConnected,
        };
        assert!(matches!(
            LieGroupFusionRule::new(too_large),
            Err(LieGroupFusionRuleError::RankNotRepresentable { .. })
        ));
        // Exercise the codec's fallible allocation independently: production
        // rejects this rank before constructing eager provider state.
        let codec = LieCodec {
            group: too_large,
            irreps: Arc::default(),
        };
        assert!(matches!(
            codec.decode_dynkin(SectorId::new(0)),
            Err(LieGroupFusionRuleError::LabelAllocation { .. })
        ));
        for n in [3, 4] {
            let rule = LieGroupFusionRule::new(GroupId::su(n).unwrap()).unwrap();
            let labels = rule.decode_dynkin(SectorId::new(usize::MAX)).unwrap();
            assert_eq!(
                rule.encode_dynkin(&labels).unwrap(),
                SectorId::new(usize::MAX)
            );
        }
        assert!(matches!(
            LieGroupFusionRule::new(GroupId::su(2).unwrap())
                .unwrap()
                .decode_dynkin(SectorId::new(usize::MAX)),
            Err(LieGroupFusionRuleError::DecodeOverflow { .. })
        ));
        assert_eq!(
            rule.rule_identity(),
            LieGroupFusionRule::new(GroupId::su(3).unwrap())
                .unwrap()
                .rule_identity()
        );
        assert_ne!(
            rule.rule_identity(),
            LieGroupFusionRule::new(GroupId::su(4).unwrap())
                .unwrap()
                .rule_identity()
        );
        let original =
            LieGroupFusionRule::identity_bytes(rule.group_id(), rule.category()).unwrap();
        let mut expected = b"symtn:lie:dynkin:graded-total-then-lex:v1".to_vec();
        expected.extend_from_slice(&usize::BITS.to_le_bytes());
        expected.push(0); // A-series
        expected.extend_from_slice(&2usize.to_le_bytes());
        expected.push(0); // simply connected
        expected.extend_from_slice(&0usize.to_le_bytes());
        expected.extend_from_slice(b":generic:bosonic:rank-one-max=u32::MAX");
        expected.extend_from_slice(
            b":rigid=f64-finite-exact-dim:f-axes=mu-nu-kappa-lambda:r-axes=mu-nu:pivotal=f-sign-0000",
        );
        expected.extend_from_slice(b"racah:sun-gt:ref=sunrep-0.4:basis=gt-order:gauge=qrpos-cref:descent=ladder-lstsq:tol=sunrep-tol-tier:epoch=1");
        assert_eq!(original.as_ref(), expected);
        let mut changed = original.to_vec();
        changed[0] ^= 1;
        assert_ne!(
            RuleIdentity::from_canonical_bytes::<LieGroupFusionRule>(IDENTITY_SCHEMA, original),
            RuleIdentity::from_canonical_bytes::<LieGroupFusionRule>(
                IDENTITY_SCHEMA,
                Arc::from(changed)
            )
        );
    }

    #[test]
    fn stored_identity_is_shared_and_distinguishes_rank() {
        let su3 = LieGroupFusionRule::new(GroupId::su(3).unwrap()).unwrap();
        let su4 = LieGroupFusionRule::new(GroupId::su(4).unwrap()).unwrap();
        let (a, b) = (su3.rule_identity(), su3.rule_identity());
        assert!(a.same_content_allocation(&b));
        assert_ne!(a, su4.rule_identity());
        let hash = |rule: &LieGroupFusionRule| {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            rule.rule_identity().hash(&mut hasher);
            hasher.finish()
        };
        assert_ne!(hash(&su3), hash(&su4));
    }

    #[test]
    fn irreps_and_duals_are_interned_once_per_sector() {
        let rule = LieGroupFusionRule::new(GroupId::su(3).unwrap()).unwrap();
        let three = rule.encode_dynkin(&[1, 0]).unwrap();
        let eight = rule.encode_dynkin(&[1, 1]).unwrap();
        let first = rule.codec.irrep(|| rule.category(), three).unwrap();
        assert!(Arc::ptr_eq(
            &first,
            &rule.codec.irrep(|| rule.category(), three).unwrap()
        ));
        let cloned_rule = rule.clone();
        assert!(Arc::ptr_eq(
            &first,
            &cloned_rule
                .codec
                .irrep(|| cloned_rule.category(), three)
                .unwrap()
        ));
        assert_eq!(*first, rule.category().irrep(&[1, 0]).unwrap());
        assert_ne!(
            *rule.codec.irrep(|| rule.category(), eight).unwrap(),
            *first
        );
        assert_eq!(
            rule.codec
                .sector(rule.category(), Cow::Borrowed(&first))
                .unwrap(),
            three
        );

        let anti_three = rule.encode_dynkin(&[0, 1]).unwrap();
        assert_eq!(rule.codec.irreps.dual(three), None);
        assert_eq!(rule.try_dual(three).unwrap(), anti_three);
        assert_eq!(rule.codec.irreps.dual(three), Some(anti_three));
        assert_eq!(rule.codec.irreps.dual(anti_three), Some(three));

        // Failed conversions are not interned.
        let su2 = LieGroupFusionRule::new(GroupId::su(2).unwrap()).unwrap();
        let su4 = LieGroupFusionRule::new(GroupId::su(4).unwrap()).unwrap();
        let wide = su4.codec.irrep(|| su4.category(), three).unwrap();
        assert!(su2
            .codec
            .sector(su2.category(), Cow::Borrowed(&wide))
            .is_err());
        assert!(su2
            .codec
            .sector(su2.category(), Cow::Borrowed(&wide))
            .is_err());
    }

    #[test]
    fn su3_octet_self_fusion_has_dynkin_ordered_channels_and_multiplicities() {
        let rule = LieGroupFusionRule::new(GroupId::su(3).unwrap()).unwrap();
        let eight = rule.encode_dynkin(&[1, 1]).unwrap();
        assert_eq!(rule.fusion_style(), FusionStyleKind::Generic);
        assert_eq!(rule.braiding_style(), BraidingStyleKind::Bosonic);
        assert_eq!(rule.try_nsymbol(eight, eight, eight).unwrap(), 2);
        assert_eq!(
            rule.try_dual(rule.encode_dynkin(&[2, 1]).unwrap()).unwrap(),
            rule.encode_dynkin(&[1, 2]).unwrap()
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
                .map(|&sector| rule.decode_dynkin(sector).unwrap())
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
                rule.try_nsymbol(eight, eight, rule.encode_dynkin(&labels).unwrap())
                    .unwrap(),
                multiplicity
            );
        }
        assert_eq!(
            rule.try_nsymbol(eight, eight, rule.encode_dynkin(&[1, 0]).unwrap())
                .unwrap(),
            0
        );
    }

    #[test]
    fn su3_rigid_symbols_match_sunrepresentations_fixtures() {
        let rule = LieGroupFusionRule::new(GroupId::su(3).unwrap()).unwrap();
        let three = rule.encode_dynkin(&[1, 0]).unwrap();
        let anti_three = rule.encode_dynkin(&[0, 1]).unwrap();
        let eight = rule.encode_dynkin(&[1, 1]).unwrap();

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

        let twenty_seven = rule.encode_dynkin(&[2, 2]).unwrap();
        let asymmetric = rule
            .try_f_symbol_generic(
                twenty_seven,
                twenty_seven,
                anti_three,
                anti_three,
                eight,
                rule.encode_dynkin(&[3, 1]).unwrap(),
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

        let left = rule.encode_dynkin(&[2, 2]).unwrap();
        let right = rule.encode_dynkin(&[2, 3]).unwrap();
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
        let rule = LieGroupFusionRule::new(GroupId::su(2).unwrap()).unwrap();
        let fundamental = rule.encode_dynkin(&[1]).unwrap();
        assert_eq!(
            rule.try_frobenius_schur_phase_scalar(fundamental).unwrap(),
            -1.0
        );
    }

    #[test]
    fn su4_adjoint_rigid_shape_smoke() {
        let rule = LieGroupFusionRule::new(GroupId::su(4).unwrap()).unwrap();
        let adjoint = rule.encode_dynkin(&[1, 0, 1]).unwrap();
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

    #[test]
    fn rigid_failures_remain_typed() {
        let rule = LieGroupFusionRule::new(GroupId::su(3).unwrap()).unwrap();
        let three = rule.encode_dynkin(&[1, 0]).unwrap();
        let eight = rule.encode_dynkin(&[1, 1]).unwrap();
        let r = rule.try_r_symbol_generic(three, three, eight).unwrap();
        assert_eq!(r.shape(), (0, 0));
        assert!(r.data().is_empty());
        let f = rule
            .try_f_symbol_generic(three, three, three, three, eight, three)
            .unwrap();
        assert_eq!(f.shape(), (0, 1, 0, 0));
        assert!(f.data().is_empty());

        let rule = LieGroupFusionRule::new(GroupId::su(78).unwrap()).unwrap();
        let mut labels = vec![0; 77];
        labels[36] = 19;
        let sector = rule.encode_dynkin(&labels).unwrap();
        assert!(matches!(
            rule.try_sqrt_dim_scalar(sector),
            Err(LieGroupFusionRuleError::DimensionNotRepresentable { .. })
        ));
    }
}
