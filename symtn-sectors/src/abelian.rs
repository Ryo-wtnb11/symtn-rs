//! Z2, Z_N, U(1) and fermion parity as symtn fusion rules.
//!
//! Every symbol (dual, fusion, N, F, R, dim, twist, Frobenius–Schur phase)
//! comes from fusion-symbols categories, which follow TensorKitSectors
//! 0.3.9; this module owns only symtn's labels, `SectorId` codecs, styles,
//! identities and checked error contracts. An inadmissible F or R is `0.0`,
//! as in TensorKitSectors. A `SectorId` outside the rule's domain panics in
//! the infallible symbols, as in the other providers.

use fusion_symbols::{
    category::FusionCategory,
    fermion::{FermionParity, FermionParityIrrep},
    finite::{ZNIrrep as FsZNIrrep, ZN},
    lie::{U1Irrep as FsU1Irrep, U1},
};

use crate::{
    BraidingStyleKind, CanonicalUnitFusionRule, CheckedFusionAlgebra, FusionAlgebraError,
    FusionRule, FusionStyleKind, MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols,
    MultiplicityFreeRigidSymbols, PhysicalBasisError, PhysicalFusionBasis, RuleIdentity,
    SectorCodec, SectorId, SectorOrderKey, SectorVec,
};

/// Decodes a label for an infallible symbol; out of domain is a caller bug.
fn or_panic<T>(label: Result<T, FusionAlgebraError>) -> T {
    label.unwrap_or_else(|error| panic!("{error}"))
}

/// The rigid symbols of a rule whose `fn label(&self, SectorId)` returns a
/// fusion-symbols label. The inverse and square-root forms are derived from
/// `dim` exactly as the fusion-symbols golden records them.
macro_rules! rigid_symbols_from_labels {
    ($rule:ty, $this:ident, $category:expr) => {
        impl MultiplicityFreeRigidSymbols for $rule {
            fn dim_scalar(&$this, sector: SectorId) -> f64 {
                $category.dim(&$this.label(sector))
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
            fn twist_scalar(&$this, sector: SectorId) -> f64 {
                $category.twist(&$this.label(sector))
            }
            fn frobenius_schur_phase_scalar(&$this, sector: SectorId) -> f64 {
                $category.frobenius_schur_phase(&$this.label(sector))
            }
        }
    };
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Z2Irrep {
    parity: u8,
}

/// Irreducible sector of the finite cyclic group Z_N.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ZNIrrep {
    charge: u32,
}

impl ZNIrrep {
    pub const fn charge(self) -> u32 {
        self.charge
    }

    pub const fn sector_id(self) -> SectorId {
        SectorId::new(self.charge as usize)
    }
}

impl From<ZNIrrep> for SectorId {
    fn from(value: ZNIrrep) -> Self {
        value.sector_id()
    }
}

#[derive(Clone, Debug)]
pub struct ZNFusionRule {
    modulus: std::num::NonZeroU32,
    identity: RuleIdentity,
}

impl PartialEq for ZNFusionRule {
    fn eq(&self, other: &Self) -> bool {
        self.modulus == other.modulus
    }
}
impl Eq for ZNFusionRule {}
impl std::hash::Hash for ZNFusionRule {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.modulus.hash(state);
    }
}

impl ZNFusionRule {
    pub fn new(modulus: u32) -> Result<Self, FusionAlgebraError> {
        let Some(modulus) = std::num::NonZeroU32::new(modulus) else {
            return Err(FusionAlgebraError::UnrepresentableSectorLabel {
                rule: RuleIdentity::of_type::<Self>(),
                label: "Z_N modulus must be nonzero".into(),
            });
        };
        let bytes: std::sync::Arc<[u8]> = modulus.get().to_le_bytes().to_vec().into();
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&modulus.get(), &mut hasher);
        let identity =
            RuleIdentity::from_canonical_bytes::<Self>(std::hash::Hasher::finish(&hasher), bytes);
        Ok(Self { modulus, identity })
    }

    pub const fn modulus(&self) -> u32 {
        self.modulus.get()
    }

    pub fn irrep(&self, charge: i64) -> ZNIrrep {
        ZNIrrep {
            charge: self.fs_irrep(charge).charge(),
        }
    }

    fn fs_irrep(&self, charge: i64) -> FsZNIrrep {
        FsZNIrrep::new(charge, self.modulus())
            .expect("the modulus is a NonZeroU32, so ZeroModulus cannot occur")
    }

    fn checked(&self, sector: SectorId) -> Result<FsZNIrrep, FusionAlgebraError> {
        let charge = u32::try_from(sector.id()).ok();
        match charge.filter(|&charge| charge < self.modulus()) {
            Some(charge) => Ok(self.fs_irrep(i64::from(charge))),
            None => Err(FusionAlgebraError::InvalidSector { sector }),
        }
    }

    fn label(&self, sector: SectorId) -> FsZNIrrep {
        or_panic(self.checked(sector))
    }

    fn sector(label: FsZNIrrep) -> SectorId {
        SectorId::new(label.charge() as usize)
    }
}

impl crate::seal::Provider for ZNFusionRule {}

crate::algebra::multiplicity_free_typed_admission!(ZNFusionRule);

impl FusionRule for ZNFusionRule {
    fn rule_identity(&self) -> RuleIdentity {
        self.identity.clone()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }
    fn vacuum(&self) -> SectorId {
        self.irrep(0).into()
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

impl CheckedFusionAlgebra for ZNFusionRule {
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        Ok(Self::sector(
            ZN::new(self.modulus())
                .unwrap()
                .dual(&self.checked(sector)?),
        ))
    }
    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        let left = self.checked(left)?;
        let right = self.checked(right)?;
        Ok(ZN::new(self.modulus())
            .unwrap()
            .fusion(&left, &right)
            .unwrap()
            .map(Self::sector)
            .collect())
    }
    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        let coupled = self.checked(coupled)?;
        let left = self.checked(left)?;
        let right = self.checked(right)?;
        Ok(ZN::new(self.modulus())
            .unwrap()
            .nsymbol(&left, &right, &coupled) as usize)
    }
}

impl SectorCodec for ZNFusionRule {
    type Sector = ZNIrrep;
    fn encode_sector(&self, value: &Self::Sector) -> Result<SectorId, FusionAlgebraError> {
        if value.charge < self.modulus() {
            Ok((*value).into())
        } else {
            Err(FusionAlgebraError::InvalidSector {
                sector: value.sector_id(),
            })
        }
    }
    fn decode_sector(&self, id: SectorId) -> Result<Self::Sector, FusionAlgebraError> {
        self.checked(id).map(|label| ZNIrrep {
            charge: label.charge(),
        })
    }
}

impl MultiplicityFreeFusionRule for ZNFusionRule {}
impl CanonicalUnitFusionRule for ZNFusionRule {}
impl MultiplicityFreeFusionSymbols for ZNFusionRule {
    type Scalar = f64;
    fn has_trivial_associator_gauge(&self) -> bool {
        true
    }
    fn f_symbol_scalar(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Self::Scalar {
        let [a, b, c, d, e, f] = [a, b, c, d, e, f].map(|sector| self.label(sector));
        ZN::new(self.modulus())
            .unwrap()
            .f_symbol(&a, &b, &c, &d, &e, &f)
            .unwrap()
    }
    fn r_symbol_scalar(&self, a: SectorId, b: SectorId, c: SectorId) -> Self::Scalar {
        ZN::new(self.modulus())
            .unwrap()
            .r_symbol(&self.label(a), &self.label(b), &self.label(c))
            .unwrap()
    }
}
rigid_symbols_from_labels!(ZNFusionRule, self, ZN::new(self.modulus()).unwrap());

impl Z2Irrep {
    pub const EVEN: Self = Self { parity: 0 };
    pub const ODD: Self = Self { parity: 1 };

    pub const fn new(parity: u8) -> Self {
        Self { parity: parity & 1 }
    }

    #[inline]
    pub const fn parity(self) -> u8 {
        self.parity
    }

    #[inline]
    pub const fn sector_id(self) -> SectorId {
        SectorId::new(self.parity as usize)
    }

    pub const fn from_sector_id(sector: SectorId) -> Option<Self> {
        match sector.id() {
            0 => Some(Self::EVEN),
            1 => Some(Self::ODD),
            _ => None,
        }
    }

    const fn is_odd(self) -> bool {
        self.parity == 1
    }
}

impl From<Z2Irrep> for SectorId {
    fn from(value: Z2Irrep) -> Self {
        value.sector_id()
    }
}

fn checked_z2_irrep(sector: SectorId) -> Result<Z2Irrep, FusionAlgebraError> {
    match Z2Irrep::from_sector_id(sector) {
        Some(irrep) => Ok(irrep),
        None => Err(FusionAlgebraError::InvalidSector { sector }),
    }
}

fn checked_z2(sector: SectorId) -> Result<FsZNIrrep, FusionAlgebraError> {
    checked_z2_irrep(sector)
        .map(|irrep| FsZNIrrep::from_canonical(u32::from(irrep.is_odd()), 2).unwrap())
}

fn checked_fermion(sector: SectorId) -> Result<FermionParityIrrep, FusionAlgebraError> {
    checked_z2_irrep(sector).map(|irrep| FermionParityIrrep::new(irrep.is_odd()))
}

fn parity_sector(odd: bool) -> SectorId {
    Z2Irrep::new(u8::from(odd)).sector_id()
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct Z2FusionRule;

impl Z2FusionRule {
    fn label(&self, sector: SectorId) -> FsZNIrrep {
        or_panic(checked_z2(sector))
    }
}

impl crate::seal::Provider for Z2FusionRule {}

crate::algebra::multiplicity_free_typed_admission!(Z2FusionRule);

impl FusionRule for Z2FusionRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        Z2Irrep::EVEN.into()
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

impl CheckedFusionAlgebra for Z2FusionRule {
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        Ok(parity_sector(
            ZN::new(2).unwrap().dual(&checked_z2(sector)?).charge() == 1,
        ))
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        Ok(ZN::new(2)
            .unwrap()
            .fusion(&checked_z2(left)?, &checked_z2(right)?)
            .unwrap()
            .map(|label| parity_sector(label.charge() == 1))
            .collect())
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        let (left, right) = (checked_z2(left)?, checked_z2(right)?);
        Ok(ZN::new(2)
            .unwrap()
            .nsymbol(&left, &right, &checked_z2(coupled)?) as usize)
    }
}

impl SectorCodec for Z2FusionRule {
    type Sector = Z2Irrep;

    fn encode_sector(&self, value: &Self::Sector) -> Result<SectorId, FusionAlgebraError> {
        Ok(value.sector_id())
    }

    fn decode_sector(&self, id: SectorId) -> Result<Self::Sector, FusionAlgebraError> {
        checked_z2_irrep(id)
    }
}

impl MultiplicityFreeFusionRule for Z2FusionRule {}

impl CanonicalUnitFusionRule for Z2FusionRule {}

impl MultiplicityFreeFusionSymbols for Z2FusionRule {
    type Scalar = f64;

    fn has_trivial_associator_gauge(&self) -> bool {
        true
    }

    fn f_symbol_scalar(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Self::Scalar {
        let [a, b, c, d, e, f] = [a, b, c, d, e, f].map(|sector| self.label(sector));
        ZN::new(2)
            .unwrap()
            .f_symbol(&a, &b, &c, &d, &e, &f)
            .unwrap()
    }

    fn r_symbol_scalar(&self, a: SectorId, b: SectorId, c: SectorId) -> Self::Scalar {
        ZN::new(2)
            .unwrap()
            .r_symbol(&self.label(a), &self.label(b), &self.label(c))
            .unwrap()
    }
}

rigid_symbols_from_labels!(Z2FusionRule, self, ZN::new(2).unwrap());

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct FermionParityFusionRule;

impl FermionParityFusionRule {
    fn label(&self, sector: SectorId) -> FermionParityIrrep {
        or_panic(checked_fermion(sector))
    }
}

impl crate::seal::Provider for FermionParityFusionRule {}

crate::algebra::multiplicity_free_typed_admission!(FermionParityFusionRule);

impl FusionRule for FermionParityFusionRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Fermionic
    }

    fn vacuum(&self) -> SectorId {
        Z2Irrep::EVEN.into()
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

impl CheckedFusionAlgebra for FermionParityFusionRule {
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        Ok(parity_sector(
            FermionParity.dual(&checked_fermion(sector)?).is_odd(),
        ))
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        Ok(FermionParity
            .fusion(&checked_fermion(left)?, &checked_fermion(right)?)
            .unwrap()
            .map(|label| parity_sector(label.is_odd()))
            .collect())
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        let (left, right) = (checked_fermion(left)?, checked_fermion(right)?);
        Ok(FermionParity.nsymbol(&left, &right, &checked_fermion(coupled)?) as usize)
    }
}

impl SectorCodec for FermionParityFusionRule {
    type Sector = Z2Irrep;

    fn encode_sector(&self, value: &Self::Sector) -> Result<SectorId, FusionAlgebraError> {
        Ok(value.sector_id())
    }

    fn decode_sector(&self, id: SectorId) -> Result<Self::Sector, FusionAlgebraError> {
        checked_z2_irrep(id)
    }
}

impl MultiplicityFreeFusionRule for FermionParityFusionRule {}

impl CanonicalUnitFusionRule for FermionParityFusionRule {}

impl MultiplicityFreeFusionSymbols for FermionParityFusionRule {
    type Scalar = f64;

    fn has_trivial_associator_gauge(&self) -> bool {
        true
    }

    fn f_symbol_scalar(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Self::Scalar {
        let [a, b, c, d, e, f] = [a, b, c, d, e, f].map(|sector| self.label(sector));
        FermionParity.f_symbol(&a, &b, &c, &d, &e, &f).unwrap()
    }

    fn r_symbol_scalar(&self, a: SectorId, b: SectorId, c: SectorId) -> Self::Scalar {
        FermionParity
            .r_symbol(&self.label(a), &self.label(b), &self.label(c))
            .unwrap()
    }
}

rigid_symbols_from_labels!(FermionParityFusionRule, self, FermionParity);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct U1Irrep {
    charge: i32,
}

impl U1Irrep {
    /// Returns the U(1) label when its dual is representable by this encoding.
    pub const fn try_new(charge: i32) -> Option<Self> {
        if charge == i32::MIN {
            None
        } else {
            Some(Self { charge })
        }
    }

    /// Creates a U(1) label.
    ///
    /// # Panics
    ///
    /// Panics for `i32::MIN`, whose dual charge cannot be represented by `i32`.
    pub const fn new(charge: i32) -> Self {
        match Self::try_new(charge) {
            Some(irrep) => irrep,
            None => panic!("U(1) charge i32::MIN has no representable dual label"),
        }
    }

    #[inline]
    pub const fn charge(self) -> i32 {
        self.charge
    }

    pub const fn sector_id(self) -> SectorId {
        SectorId::new(u1_charge_to_zigzag_u32(self.charge) as usize)
    }

    pub fn from_sector_id(sector: SectorId) -> Option<Self> {
        u32::try_from(sector.id())
            .ok()
            .map(u1_charge_from_zigzag_u32)
            .and_then(Self::try_new)
    }

    pub fn checked_fuse(self, other: Self) -> Result<Self, FusionAlgebraError> {
        match U1.fusion(&self.fs(), &other.fs()) {
            Ok(mut fused) => Ok(Self::from_fs(fused.next().unwrap())),
            Err(_) => Err(FusionAlgebraError::U1FusionOverflow {
                left: self.charge,
                right: other.charge,
            }),
        }
    }

    fn fs(self) -> FsU1Irrep {
        FsU1Irrep::new(self.charge).expect("both encodings exclude only i32::MIN")
    }

    const fn from_fs(label: FsU1Irrep) -> Self {
        Self {
            charge: label.charge(),
        }
    }
}

const fn u1_charge_to_zigzag_u32(charge: i32) -> u32 {
    ((charge as u32) << 1) ^ ((charge >> 31) as u32)
}

const fn u1_charge_from_zigzag_u32(encoded: u32) -> i32 {
    ((encoded >> 1) as i32) ^ -((encoded & 1) as i32)
}

fn checked_u1_irrep(sector: SectorId) -> Result<U1Irrep, FusionAlgebraError> {
    match U1Irrep::from_sector_id(sector) {
        Some(irrep) => Ok(irrep),
        None => Err(FusionAlgebraError::InvalidSector { sector }),
    }
}

impl From<U1Irrep> for SectorId {
    fn from(value: U1Irrep) -> Self {
        value.sector_id()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct U1FusionRule;

impl U1FusionRule {
    fn label(&self, sector: SectorId) -> FsU1Irrep {
        or_panic(checked_u1_irrep(sector)).fs()
    }
}

impl crate::seal::Provider for U1FusionRule {}

crate::algebra::multiplicity_free_typed_admission!(U1FusionRule);

impl FusionRule for U1FusionRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        U1Irrep::new(0).into()
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

    /// TensorKitSectors 0.3.9 `findindex(::SectorValues{U1Irrep}, c)`
    /// (`src/irreps/u1irrep.jl`) is `2|2q| + (2q <= 0)`: TensorKit also
    /// enumerates half-integer charges (0, 1/2, -1/2, 1, -1, ...), so an
    /// integer charge `q` sits at position `4q - 1` for `q > 0` and `4|q|`
    /// otherwise. The spacing matters inside a product's degree, not only the
    /// order.
    fn sector_order_key(&self, sector: SectorId) -> SectorOrderKey {
        let charge = match checked_u1_irrep(sector) {
            Ok(irrep) => irrep.charge(),
            Err(error) => panic!("{error}"),
        };
        SectorOrderKey::position(4 * u64::from(charge.unsigned_abs()) - u64::from(charge > 0))
    }
}

impl CheckedFusionAlgebra for U1FusionRule {
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        let irrep = checked_u1_irrep(sector)?;
        Ok(U1Irrep::from_fs(U1.dual(&irrep.fs())).into())
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        let left = checked_u1_irrep(left)?;
        let right = checked_u1_irrep(right)?;
        Ok(core::iter::once(left.checked_fuse(right)?.into()).collect())
    }

    /// Why not fusion-symbols' exact `U1.nsymbol` alone: symtn's checked
    /// algebra reports a sum outside the label range as
    /// `U1FusionOverflow` (stricter than TensorKitSectors' `0`), so a caller
    /// that enumerates channels and one that asks for `N` fail alike.
    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        let left = checked_u1_irrep(left)?;
        let right = checked_u1_irrep(right)?;
        let coupled = checked_u1_irrep(coupled)?;
        left.checked_fuse(right)?;
        Ok(U1.nsymbol(&left.fs(), &right.fs(), &coupled.fs()) as usize)
    }
}

impl SectorCodec for U1FusionRule {
    type Sector = U1Irrep;

    fn encode_sector(&self, value: &Self::Sector) -> Result<SectorId, FusionAlgebraError> {
        if Self::Sector::try_new(value.charge()).is_some() {
            Ok(value.sector_id())
        } else {
            Err(FusionAlgebraError::UnrepresentableSectorLabel {
                rule: self.rule_identity(),
                label: value.charge().to_string(),
            })
        }
    }

    fn decode_sector(&self, id: SectorId) -> Result<Self::Sector, FusionAlgebraError> {
        checked_u1_irrep(id)
    }
}

impl MultiplicityFreeFusionRule for U1FusionRule {}

impl CanonicalUnitFusionRule for U1FusionRule {}

impl PhysicalFusionBasis for U1FusionRule {
    type Scalar = f64;
    type Error = PhysicalBasisError<std::convert::Infallible>;

    fn try_carrier_dimension(&self, sector: SectorId) -> Result<usize, Self::Error> {
        checked_u1_irrep(sector)?;
        Ok(1)
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
        for (sector, index) in [
            (left, left_basis),
            (right, right_basis),
            (coupled, coupled_basis),
        ] {
            if index != 0 {
                return Err(PhysicalBasisError::CarrierIndexOutOfBounds {
                    sector,
                    index,
                    dimension: 1,
                });
            }
        }
        Ok(1.0)
    }
}

impl MultiplicityFreeFusionSymbols for U1FusionRule {
    type Scalar = f64;

    fn has_trivial_associator_gauge(&self) -> bool {
        true
    }

    fn f_symbol_scalar(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Self::Scalar {
        let [a, b, c, d, e, f] = [a, b, c, d, e, f].map(|sector| self.label(sector));
        U1.f_symbol(&a, &b, &c, &d, &e, &f).unwrap()
    }

    fn r_symbol_scalar(&self, a: SectorId, b: SectorId, c: SectorId) -> Self::Scalar {
        U1.r_symbol(&self.label(a), &self.label(b), &self.label(c))
            .unwrap()
    }
}

rigid_symbols_from_labels!(U1FusionRule, self, U1);

#[cfg(test)]
mod tests {
    use super::{
        u1_charge_from_zigzag_u32, u1_charge_to_zigzag_u32, FermionParityFusionRule, U1FusionRule,
        U1Irrep, Z2FusionRule, Z2Irrep, ZNFusionRule,
    };
    use crate::{
        CanonicalUnitFusionRule, CheckedFusionAlgebra, FusionAlgebraError, FusionRule,
        MultiplicityFreeFusionSymbols, PhysicalBasisError, PhysicalFusionBasis, SectorCodec,
        SectorId,
    };

    fn assert_canonical_unit<R>(rule: &R, sector: SectorId)
    where
        R: CanonicalUnitFusionRule + MultiplicityFreeFusionSymbols<Scalar = f64>,
    {
        let vacuum = rule.vacuum();
        let fused = rule.fusion_channels(sector, sector)[0];
        assert_eq!(rule.dual(vacuum), vacuum);
        assert_eq!(rule.fusion_channels(vacuum, sector).as_slice(), &[sector]);
        assert_eq!(rule.fusion_channels(sector, vacuum).as_slice(), &[sector]);
        assert_eq!(rule.nsymbol(vacuum, sector, sector), 1);
        assert_eq!(rule.nsymbol(sector, vacuum, sector), 1);
        assert_eq!(rule.nsymbol(vacuum, sector, vacuum), 0);
        assert_eq!(
            rule.f_symbol_scalar(vacuum, sector, sector, fused, sector, fused),
            1.0
        );
        assert_eq!(
            rule.f_symbol_scalar(sector, vacuum, sector, fused, sector, sector),
            1.0
        );
        assert_eq!(
            rule.f_symbol_scalar(sector, sector, vacuum, fused, fused, sector),
            1.0
        );
    }

    #[test]
    fn abelian_providers_have_canonical_unit_channels_and_associators() {
        // What: certified abelian providers keep the vacuum as a unique,
        // self-dual identity with an exact unit F-symbol.
        assert_canonical_unit(&Z2FusionRule, Z2Irrep::ODD.into());
        assert_canonical_unit(&FermionParityFusionRule, Z2Irrep::ODD.into());
        assert_canonical_unit(&U1FusionRule, U1Irrep::new(-17).into());
        let u1_boundary: SectorId = U1Irrep::new(i32::MAX).into();
        let u1_vacuum = U1FusionRule.vacuum();
        assert_eq!(
            U1FusionRule
                .fusion_channels(u1_vacuum, u1_boundary)
                .as_slice(),
            &[u1_boundary]
        );
        assert_eq!(
            U1FusionRule
                .fusion_channels(u1_boundary, u1_vacuum)
                .as_slice(),
            &[u1_boundary]
        );
    }

    #[test]
    fn u1_zigzag_encoding_is_unchanged_at_32_bit_extremes() {
        // What: rejecting one label does not change the packed zigzag layout.
        for (charge, encoded) in [
            (i32::MIN, u32::MAX),
            (-1, 1),
            (0, 0),
            (1, 2),
            (i32::MAX, u32::MAX - 1),
        ] {
            assert_eq!(u1_charge_to_zigzag_u32(charge), encoded);
            assert_eq!(u1_charge_from_zigzag_u32(encoded), charge);
            if let Some(irrep) = U1Irrep::try_new(charge) {
                let sector = irrep.sector_id();
                assert_eq!(sector.id(), encoded as usize);
                assert_eq!(U1Irrep::from_sector_id(sector), Some(irrep));
            }
        }
    }

    #[test]
    fn u1_label_boundary_rejects_only_i32_min() {
        // What: i32::MIN has no label, while both adjacent representable
        // extremes encode, decode, and dualize normally.
        let invalid = SectorId::new(u32::MAX as usize);
        assert_eq!(U1Irrep::try_new(i32::MIN), None);
        assert_eq!(U1Irrep::from_sector_id(invalid), None);
        assert_eq!(
            U1FusionRule.decode_sector(invalid),
            Err(FusionAlgebraError::InvalidSector { sector: invalid })
        );

        for charge in [i32::MIN + 1, i32::MAX] {
            let irrep = U1Irrep::new(charge);
            assert_eq!(U1FusionRule.encode_sector(&irrep), Ok(irrep.sector_id()));
            assert_eq!(U1Irrep::from_sector_id(irrep.sector_id()), Some(irrep));
            assert_eq!(
                U1FusionRule
                    .try_dual_sector(irrep.sector_id())
                    .and_then(|dual| U1FusionRule.try_dual_sector(dual)),
                Ok(irrep.sector_id())
            );
        }
    }

    #[test]
    #[should_panic(expected = "U(1) charge i32::MIN has no representable dual label")]
    fn u1_new_rejects_i32_min() {
        let _ = U1Irrep::new(i32::MIN);
    }

    #[test]
    fn u1_codec_defensively_rejects_an_invalid_label() {
        // What: the codec keeps its own trust-boundary check even though safe
        // public constructors cannot create this value.
        let invalid = U1Irrep { charge: i32::MIN };
        assert_eq!(
            U1FusionRule.encode_sector(&invalid),
            Err(FusionAlgebraError::UnrepresentableSectorLabel {
                rule: U1FusionRule.rule_identity(),
                label: i32::MIN.to_string(),
            })
        );
    }

    #[test]
    fn u1_physical_basis_is_the_checked_one_dimensional_embedding() {
        // What: U(1) exposes exactly one carrier state and one unit CGC for an
        // allowed charge sum, while invalid physical indices remain typed.
        let rule = U1FusionRule;
        let left: SectorId = U1Irrep::new(2).into();
        let right: SectorId = U1Irrep::new(-1).into();
        let coupled: SectorId = U1Irrep::new(1).into();
        assert_eq!(rule.try_carrier_dimension(left), Ok(1));
        assert_eq!(
            rule.try_fusion_tensor_element(left, right, coupled, 0, 0, 0, 0),
            Ok(1.0)
        );
        assert_eq!(
            rule.try_fusion_tensor_element(left, right, coupled, 1, 0, 0, 0),
            Err(PhysicalBasisError::CarrierIndexOutOfBounds {
                sector: left,
                index: 1,
                dimension: 1,
            })
        );
        let absent: SectorId = U1Irrep::new(0).into();
        assert_eq!(
            rule.try_fusion_tensor_element(left, right, absent, 0, 0, 0, 0),
            Err(PhysicalBasisError::FusionMultiplicityOutOfBounds {
                left,
                right,
                coupled: absent,
                multiplicity: 0,
                dimension: 0,
            })
        );
    }

    #[test]
    fn zn_fuses_wraps_and_identity_contains_modulus() {
        let a = ZNFusionRule::new(3).unwrap();
        let b = ZNFusionRule::new(4).unwrap();
        assert_eq!(a.irrep(-1).charge(), 2);
        assert_eq!(
            a.try_fusion_channels(a.irrep(2).into(), a.irrep(2).into())
                .unwrap()[0],
            a.irrep(1).into()
        );
        assert_ne!(a.rule_identity(), b.rule_identity());
        assert!(ZNFusionRule::new(0).is_err());
    }

    #[test]
    fn zn_checked_rejects_out_of_domain_sector() {
        let rule = ZNFusionRule::new(3).unwrap();
        assert!(matches!(
            rule.try_dual_sector(SectorId::new(3)),
            Err(crate::FusionAlgebraError::InvalidSector { .. })
        ));
    }

    #[test]
    fn zn_boundary_labels_fuse_and_symbols_are_canonical() {
        for modulus in [1, 2, 3, 17, u32::MAX] {
            let rule = ZNFusionRule::new(modulus).unwrap();
            for charge in [i64::MIN, -1, 0, 1, i64::MAX] {
                let sector = rule.irrep(charge);
                assert_eq!(
                    rule.try_dual_sector(rule.try_dual_sector(sector.into()).unwrap())
                        .unwrap(),
                    sector.into()
                );
                assert_eq!(rule.decode_sector(sector.into()).unwrap(), sector);
            }
            let a = rule.irrep(u32::MAX as i64);
            let b = rule.irrep(1);
            let fused = rule.try_fusion_channels(a.into(), b.into()).unwrap()[0];
            assert_eq!(fused, rule.irrep((u32::MAX as u64 + 1) as i64).into());
            if modulus == u32::MAX {
                let edge = rule.irrep((u32::MAX - 1) as i64);
                assert_eq!(
                    rule.try_fusion_channels(edge.into(), edge.into()).unwrap()[0],
                    rule.irrep((u32::MAX - 2) as i64).into()
                );
            }
            assert_eq!(
                rule.f_symbol_scalar(a.into(), b.into(), rule.vacuum(), fused, fused, b.into()),
                1.0
            );
            assert_eq!(rule.r_symbol_scalar(a.into(), b.into(), fused), 1.0);
        }
        assert_ne!(
            ZNFusionRule::new(3).unwrap().rule_identity(),
            ZNFusionRule::new(4).unwrap().rule_identity()
        );
        assert_eq!(
            ZNFusionRule::new(3).unwrap().rule_identity(),
            ZNFusionRule::new(3).unwrap().rule_identity()
        );
    }
}
