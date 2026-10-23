use std::sync::LazyLock;

use fusion_symbols::{category::FusionCategory, lie as fs};

use crate::category_rule::{
    admit_coupled_first, admit_in_order, fusion_algebra_error, AdapterAccess,
    CanonicalUnitAdmission, CategoryAdapterError, CategoryCodec, CategoryOperation,
};
use crate::{FusionAlgebraError, MultiplicityFreeAdmissionMode, RuleIdentity, SectorId, SectorVec};

/// Largest `twice_charge` accepted by symtn's CU(1) sector codec.
pub const CU1_MAX_TWICE_CHARGE: u32 = u32::MAX - 1;

/// TensorKit's `CU1Irrep`: `0+`, `0-`, or a positive half-integral charge.
///
/// The representation is private so `Charged(0)` and the forbidden
/// charge-conjugation labels cannot be constructed.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct CU1Irrep(CU1Kind);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
enum CU1Kind {
    Vacuum,
    Pseudoscalar,
    Charged(u32),
}

impl CU1Irrep {
    pub const VACUUM: Self = Self(CU1Kind::Vacuum);
    pub const PSEUDOSCALAR: Self = Self(CU1Kind::Pseudoscalar);

    pub const fn try_from_twice_charge(twice_charge: u32) -> Option<Self> {
        if twice_charge > 0 && twice_charge <= CU1_MAX_TWICE_CHARGE {
            Some(Self(CU1Kind::Charged(twice_charge)))
        } else {
            None
        }
    }

    pub const fn from_twice_charge(twice_charge: u32) -> Self {
        match Self::try_from_twice_charge(twice_charge) {
            Some(irrep) => irrep,
            None => panic!("CU(1) twice charge must be in 1..=u32::MAX-1"),
        }
    }

    pub const fn twice_charge(self) -> u32 {
        match self.0 {
            CU1Kind::Charged(charge) => charge,
            CU1Kind::Vacuum | CU1Kind::Pseudoscalar => 0,
        }
    }

    /// TensorKit's `s`: 0 for `0+`, 1 for `0-`, and 2 for charged sectors.
    pub const fn conjugation_sector(self) -> u8 {
        match self.0 {
            CU1Kind::Vacuum => 0,
            CU1Kind::Pseudoscalar => 1,
            CU1Kind::Charged(_) => 2,
        }
    }

    pub const fn sector_id(self) -> SectorId {
        match self.0 {
            CU1Kind::Vacuum => SectorId::new(0),
            CU1Kind::Pseudoscalar => SectorId::new(1),
            CU1Kind::Charged(charge) => SectorId::new(charge as usize + 1),
        }
    }

    pub fn try_from_sector_id(sector: SectorId) -> Option<Self> {
        match sector.id() {
            0 => Some(Self::VACUUM),
            1 => Some(Self::PSEUDOSCALAR),
            id => match u32::try_from(id - 1) {
                Ok(charge) => Self::try_from_twice_charge(charge),
                Err(_) => None,
            },
        }
    }

    const fn fs(self) -> fs::CU1Irrep {
        match self.0 {
            CU1Kind::Vacuum => fs::CU1Irrep::VACUUM,
            CU1Kind::Pseudoscalar => fs::CU1Irrep::PSEUDOSCALAR,
            CU1Kind::Charged(charge) => fs::CU1Irrep::from_twice_charge(charge),
        }
    }

    /// `None` above symtn's codec bound [`CU1_MAX_TWICE_CHARGE`].
    const fn from_fs(label: fs::CU1Irrep) -> Option<Self> {
        match (label.twice_charge(), label.s()) {
            (0, 0) => Some(Self::VACUUM),
            (0, _) => Some(Self::PSEUDOSCALAR),
            (charge, _) => Self::try_from_twice_charge(charge),
        }
    }
}

impl From<CU1Irrep> for SectorId {
    fn from(value: CU1Irrep) -> Self {
        value.sector_id()
    }
}

/// TensorKit's `CU1Irrep` fusion rule: [`fusion_symbols::lie::CU1`] behind
/// the shared category adapter, which supplies every symbol (fusion, N, F,
/// R, A, B, dims, twist, Frobenius–Schur phase) in TensorKitSectors 0.3.9's
/// gauge; this type owns symtn's label codec, its [`CU1_MAX_TWICE_CHARGE`]
/// bound and the checked error contract.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct CU1FusionRule;

static CU1_IDENTITY: LazyLock<RuleIdentity> = LazyLock::new(RuleIdentity::of_type::<CU1FusionRule>);

fn checked_irrep(sector: SectorId) -> Result<CU1Irrep, FusionAlgebraError> {
    CU1Irrep::try_from_sector_id(sector).ok_or(FusionAlgebraError::InvalidSector { sector })
}

/// `CU1Irrep` labels and ids over `fs::CU1`, bounded by
/// [`CU1_MAX_TWICE_CHARGE`].
///
/// A fusion with a channel above that bound is `FusionNotRepresentable`, for
/// the channels and for `N` alike, as fusion-symbols takes the whole `u32`
/// twice charge; `N` checks the coupled sector before the other two.
#[derive(Debug)]
pub struct Cu1Codec;

impl Cu1Codec {
    /// The ids of the complete channels in TensorKitSectors order.
    fn channels(
        inputs: [SectorId; 2],
        outcomes: impl Iterator<Item = fs::CU1Irrep>,
    ) -> Result<SectorVec, FusionAlgebraError> {
        let [left, right] = inputs;
        outcomes
            .map(|channel| {
                CU1Irrep::from_fs(channel)
                    .map(SectorId::from)
                    .ok_or(FusionAlgebraError::FusionNotRepresentable { left, right })
            })
            .collect()
    }
}

impl CategoryCodec<fs::CU1> for Cu1Codec {
    type Sector = CU1Irrep;
    type Error = FusionAlgebraError;
    type Mode = MultiplicityFreeAdmissionMode;
    type Unit = CanonicalUnitAdmission;

    fn encode_label<'a>(
        &self,
        _category: impl Fn() -> &'a fs::CU1 + Copy,
        _rule: &RuleIdentity,
        label: &CU1Irrep,
    ) -> Result<SectorId, FusionAlgebraError> {
        Ok((*label).into())
    }

    fn decode_label<'a>(
        &self,
        _category: impl Fn() -> &'a fs::CU1 + Copy,
        id: SectorId,
    ) -> Result<CU1Irrep, FusionAlgebraError> {
        checked_irrep(id)
    }

    fn object<'a>(
        &self,
        _category: impl Fn() -> &'a fs::CU1 + Copy,
        id: SectorId,
    ) -> Result<fs::CU1Irrep, FusionAlgebraError> {
        checked_irrep(id).map(CU1Irrep::fs)
    }

    fn sector_id<'a>(
        &self,
        _category: impl Fn() -> &'a fs::CU1 + Copy,
        rule: &RuleIdentity,
        object: &fs::CU1Irrep,
    ) -> Result<SectorId, FusionAlgebraError> {
        CU1Irrep::from_fs(*object)
            .map(SectorId::from)
            .ok_or_else(|| FusionAlgebraError::UnrepresentableSectorLabel {
                rule: rule.clone(),
                label: format!("{object:?}"),
            })
    }

    fn admit<'a, const N: usize>(
        &self,
        category: impl Fn() -> &'a fs::CU1 + Copy,
        rule: &RuleIdentity,
        operation: CategoryOperation,
        ids: [SectorId; N],
    ) -> Result<[fs::CU1Irrep; N], FusionAlgebraError> {
        if let (CategoryOperation::N, &[left, right, _]) = (operation, ids.as_slice()) {
            let objects = admit_coupled_first(self, category, ids)?;
            let outcomes = category()
                .fusion(&objects[0], &objects[1])
                .map_err(|_| FusionAlgebraError::FusionNotRepresentable { left, right })?;
            Self::channels([left, right], outcomes)?;
            return Ok(objects);
        }
        admit_in_order(self, category, rule, operation, ids)
    }

    fn encode_channels(
        &self,
        _category: &fs::CU1,
        _rule: &RuleIdentity,
        inputs: [SectorId; 2],
        outcomes: impl Iterator<Item = fs::CU1Irrep>,
    ) -> Result<SectorVec, FusionAlgebraError> {
        Self::channels(inputs, outcomes)
    }

    fn map_error(
        &self,
        inputs: &[SectorId],
        error: CategoryAdapterError<fs::CU1Error>,
    ) -> FusionAlgebraError {
        // The only CU(1) failure is a fusion leaving the twice-charge range.
        match (error, inputs) {
            (CategoryAdapterError::Provider(_), &[left, right]) => {
                FusionAlgebraError::FusionNotRepresentable { left, right }
            }
            (error, _) => fusion_algebra_error(error),
        }
    }
}

impl crate::seal::Provider for CU1FusionRule {}

crate::category_rule::multiplicity_free_adapter_impls!(CU1FusionRule, CU1Irrep, f64);

impl AdapterAccess for CU1FusionRule {
    type Category = fs::CU1;
    type Codec = Cu1Codec;

    fn category(&self) -> &fs::CU1 {
        &fs::CU1
    }
    fn codec(&self) -> &Cu1Codec {
        &Cu1Codec
    }
    fn identity(&self) -> &RuleIdentity {
        &CU1_IDENTITY
    }
    fn vacuum_id(&self) -> SectorId {
        CU1Irrep::VACUUM.sector_id()
    }
}
