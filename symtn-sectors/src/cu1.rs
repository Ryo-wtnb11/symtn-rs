use fusion_symbols::{category::FusionCategory, lie as fs};

use crate::{
    BraidingStyleKind, CanonicalUnitFusionRule, CheckedFusionAlgebra, FusionAlgebraError,
    FusionRule, FusionStyleKind, MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols,
    MultiplicityFreeRigidSymbols, RuleIdentity, SectorCodec, SectorId, SectorVec,
};

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

/// TensorKit's `CU1Irrep` fusion rule. Every symbol (fusion, N, F, R, dim,
/// twist, Frobenius–Schur phase) comes from [`fusion_symbols::lie::CU1`], in
/// TensorKitSectors 0.3.9's gauge; this type owns symtn's label codec, its
/// [`CU1_MAX_TWICE_CHARGE`] bound and the checked error contract.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct CU1FusionRule;

fn checked_irrep(sector: SectorId) -> Result<CU1Irrep, FusionAlgebraError> {
    CU1Irrep::try_from_sector_id(sector).ok_or(FusionAlgebraError::InvalidSector { sector })
}

fn checked_irreps(
    left: SectorId,
    right: SectorId,
) -> Result<(CU1Irrep, CU1Irrep), FusionAlgebraError> {
    Ok((checked_irrep(left)?, checked_irrep(right)?))
}

/// fusion-symbols' channels in TensorKitSectors order; a channel above
/// symtn's codec bound is `FusionNotRepresentable`, as fusion-symbols takes
/// the whole `u32` twice charge.
fn channels(left: CU1Irrep, right: CU1Irrep) -> Result<SectorVec, FusionAlgebraError> {
    let not_representable = || FusionAlgebraError::FusionNotRepresentable {
        left: left.into(),
        right: right.into(),
    };
    let fusion = fs::CU1
        .fusion(&left.fs(), &right.fs())
        .map_err(|_| not_representable())?;
    fusion
        .map(|channel| {
            CU1Irrep::from_fs(channel)
                .map(SectorId::from)
                .ok_or_else(not_representable)
        })
        .collect()
}

impl CU1FusionRule {
    fn label(&self, sector: SectorId) -> fs::CU1Irrep {
        checked_irrep(sector)
            .unwrap_or_else(|error| panic!("{error}"))
            .fs()
    }
}

impl crate::seal::Provider for CU1FusionRule {}

crate::algebra::multiplicity_free_typed_admission!(CU1FusionRule);

impl FusionRule for CU1FusionRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Simple
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }
    fn vacuum(&self) -> SectorId {
        CU1Irrep::VACUUM.into()
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

impl CheckedFusionAlgebra for CU1FusionRule {
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        let dual = fs::CU1.dual(&checked_irrep(sector)?.fs());
        Ok(CU1Irrep::from_fs(dual)
            .expect("the dual keeps the twice charge")
            .into())
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        let (left, right) = checked_irreps(left, right)?;
        channels(left, right)
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        let coupled = checked_irrep(coupled)?;
        let (left, right) = checked_irreps(left, right)?;
        Ok(usize::from(
            channels(left, right)?
                .iter()
                .any(|&sector| sector == coupled.into()),
        ))
    }
}

impl SectorCodec for CU1FusionRule {
    type Sector = CU1Irrep;
    fn encode_sector(&self, value: &Self::Sector) -> Result<SectorId, FusionAlgebraError> {
        Ok((*value).into())
    }
    fn decode_sector(&self, id: SectorId) -> Result<Self::Sector, FusionAlgebraError> {
        checked_irrep(id)
    }
}

impl MultiplicityFreeFusionRule for CU1FusionRule {}
impl CanonicalUnitFusionRule for CU1FusionRule {}

impl MultiplicityFreeFusionSymbols for CU1FusionRule {
    type Scalar = f64;

    fn f_symbol_scalar(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> f64 {
        let [a, b, c, d, e, f] = [a, b, c, d, e, f].map(|sector| self.label(sector));
        fs::CU1.f_symbol(&a, &b, &c, &d, &e, &f).unwrap()
    }

    fn r_symbol_scalar(&self, a: SectorId, b: SectorId, c: SectorId) -> f64 {
        fs::CU1
            .r_symbol(&self.label(a), &self.label(b), &self.label(c))
            .unwrap()
    }
}

impl MultiplicityFreeRigidSymbols for CU1FusionRule {
    fn dim_scalar(&self, sector: SectorId) -> f64 {
        fs::CU1.dim(&self.label(sector))
    }
    fn inv_dim_scalar(&self, sector: SectorId) -> f64 {
        1.0 / self.dim_scalar(sector)
    }
    fn sqrt_dim_scalar(&self, sector: SectorId) -> f64 {
        self.dim_scalar(sector).sqrt()
    }
    fn inv_sqrt_dim_scalar(&self, sector: SectorId) -> f64 {
        1.0 / self.sqrt_dim_scalar(sector)
    }
    fn twist_scalar(&self, sector: SectorId) -> f64 {
        fs::CU1.twist(&self.label(sector))
    }
    fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> f64 {
        fs::CU1.frobenius_schur_phase(&self.label(sector))
    }
}
