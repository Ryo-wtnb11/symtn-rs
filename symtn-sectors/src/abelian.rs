//! Z2, Z_N, U(1) and fermion parity as symtn fusion rules.
//!
//! Each rule is a fusion-symbols category (`ZN`, `FermionParity`, `U1`)
//! behind the shared category adapter, which supplies every symbol (dual,
//! fusion, N, F, R, A, B, dims, twist, Frobenius–Schur phase) in
//! TensorKitSectors 0.3.9's conventions. This module owns only symtn's
//! labels, `SectorId` codecs, identities, and checked error contracts. An
//! inadmissible F or R is `0.0`, as in TensorKitSectors. A `SectorId`
//! outside the rule's domain panics in the infallible symbols, as in the
//! other providers.

use std::sync::LazyLock;

use fusion_symbols::{
    category::{FusionCategory, GroupRepresentations},
    fermion::{FermionParity, FermionParityIrrep},
    finite::{ZNIrrep as FsZNIrrep, ZN},
    lie::{U1Error, U1Irrep as FsU1Irrep, U1},
};

use crate::category_rule::{
    admit_coupled_first, admit_in_order, fusion_algebra_error, AdapterAccess,
    CanonicalUnitAdmission, CategoryAdapterError, CategoryCodec, CategoryFailure,
    CategoryOperation,
};
use crate::{
    CheckedFusionAlgebra, FusionAlgebraError, MultiplicityFreeAdmissionMode, PhysicalBasisError,
    PhysicalFusionBasis, RuleIdentity, SectorId, SectorOrderKey,
};

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
    category: ZN,
    identity: RuleIdentity,
}

impl PartialEq for ZNFusionRule {
    fn eq(&self, other: &Self) -> bool {
        self.category == other.category
    }
}
impl Eq for ZNFusionRule {}
impl std::hash::Hash for ZNFusionRule {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.modulus().hash(state);
    }
}

impl ZNFusionRule {
    pub fn new(modulus: u32) -> Result<Self, FusionAlgebraError> {
        let Ok(category) = ZN::new(modulus) else {
            return Err(FusionAlgebraError::UnrepresentableSectorLabel {
                rule: RuleIdentity::of_type::<Self>(),
                label: "Z_N modulus must be nonzero".into(),
            });
        };
        let bytes: std::sync::Arc<[u8]> = modulus.to_le_bytes().to_vec().into();
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&modulus, &mut hasher);
        let identity =
            RuleIdentity::from_canonical_bytes::<Self>(std::hash::Hasher::finish(&hasher), bytes);
        Ok(Self { category, identity })
    }

    pub const fn modulus(&self) -> u32 {
        self.category.modulus()
    }

    pub fn irrep(&self, charge: i64) -> ZNIrrep {
        let label = FsZNIrrep::new(charge, self.modulus())
            .expect("the category's modulus is nonzero, so ZeroModulus cannot occur");
        ZNIrrep {
            charge: label.charge(),
        }
    }
}

/// `ZNIrrep` labels and charge ids over `ZN`; `N` checks the coupled sector
/// before the other two.
#[derive(Debug)]
pub struct ZnCodec;

impl CategoryCodec<ZN> for ZnCodec {
    type Sector = ZNIrrep;
    type Error = FusionAlgebraError;
    type Mode = MultiplicityFreeAdmissionMode;
    type Unit = CanonicalUnitAdmission;

    fn encode_label<'a>(
        &self,
        category: impl Fn() -> &'a ZN + Copy,
        _rule: &RuleIdentity,
        label: &ZNIrrep,
    ) -> Result<SectorId, FusionAlgebraError> {
        if label.charge < category().modulus() {
            Ok(label.sector_id())
        } else {
            Err(FusionAlgebraError::InvalidSector {
                sector: label.sector_id(),
            })
        }
    }

    fn decode_label<'a>(
        &self,
        category: impl Fn() -> &'a ZN + Copy,
        id: SectorId,
    ) -> Result<ZNIrrep, FusionAlgebraError> {
        self.object(category, id).map(|label| ZNIrrep {
            charge: label.charge(),
        })
    }

    fn object<'a>(
        &self,
        category: impl Fn() -> &'a ZN + Copy,
        id: SectorId,
    ) -> Result<FsZNIrrep, FusionAlgebraError> {
        u32::try_from(id.id())
            .ok()
            .and_then(|charge| FsZNIrrep::from_canonical(charge, category().modulus()).ok())
            .ok_or(FusionAlgebraError::InvalidSector { sector: id })
    }

    fn sector_id<'a>(
        &self,
        _category: impl Fn() -> &'a ZN + Copy,
        _rule: &RuleIdentity,
        object: &FsZNIrrep,
    ) -> Result<SectorId, FusionAlgebraError> {
        Ok(SectorId::new(object.charge() as usize))
    }

    fn admit<'a, const N: usize>(
        &self,
        category: impl Fn() -> &'a ZN + Copy,
        rule: &RuleIdentity,
        operation: CategoryOperation,
        ids: [SectorId; N],
    ) -> Result<[FsZNIrrep; N], FusionAlgebraError> {
        if operation == CategoryOperation::N {
            return admit_coupled_first(self, category, ids);
        }
        admit_in_order(self, category, rule, operation, ids)
    }

    fn map_error(
        &self,
        _inputs: &[SectorId],
        error: CategoryAdapterError<std::convert::Infallible>,
    ) -> FusionAlgebraError {
        fusion_algebra_error(error)
    }
}

impl crate::seal::Provider for ZNFusionRule {}

crate::category_rule::multiplicity_free_adapter_impls!(ZNFusionRule, ZNIrrep, f64);

impl AdapterAccess for ZNFusionRule {
    type Category = ZN;
    type Codec = ZnCodec;

    fn category(&self) -> &ZN {
        &self.category
    }
    fn codec(&self) -> &ZnCodec {
        &ZnCodec
    }
    fn identity(&self) -> &RuleIdentity {
        &self.identity
    }
    fn vacuum_id(&self) -> SectorId {
        SectorId::new(0)
    }
}

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

/// Parity labels and ids over a two-object category: `Z2Codec` over `ZN`
/// with modulus 2, `Fz2Codec` over `FermionParity`.
macro_rules! parity_codec {
    ($codec:ident, $category:ty, $object:ty, $to_object:expr, $is_odd:expr) => {
        #[derive(Debug)]
        pub struct $codec;

        impl CategoryCodec<$category> for $codec {
            type Sector = Z2Irrep;
            type Error = FusionAlgebraError;
            type Mode = MultiplicityFreeAdmissionMode;
            type Unit = CanonicalUnitAdmission;

            fn encode_label<'a>(
                &self,
                _category: impl Fn() -> &'a $category + Copy,
                _rule: &RuleIdentity,
                label: &Z2Irrep,
            ) -> Result<SectorId, FusionAlgebraError> {
                Ok(label.sector_id())
            }

            fn decode_label<'a>(
                &self,
                _category: impl Fn() -> &'a $category + Copy,
                id: SectorId,
            ) -> Result<Z2Irrep, FusionAlgebraError> {
                checked_z2_irrep(id)
            }

            fn object<'a>(
                &self,
                _category: impl Fn() -> &'a $category + Copy,
                id: SectorId,
            ) -> Result<$object, FusionAlgebraError> {
                checked_z2_irrep(id).map(|irrep| $to_object(irrep.is_odd()))
            }

            fn sector_id<'a>(
                &self,
                _category: impl Fn() -> &'a $category + Copy,
                _rule: &RuleIdentity,
                object: &$object,
            ) -> Result<SectorId, FusionAlgebraError> {
                Ok(Z2Irrep::new(u8::from($is_odd(object))).sector_id())
            }

            fn map_error(
                &self,
                _inputs: &[SectorId],
                error: CategoryAdapterError<std::convert::Infallible>,
            ) -> FusionAlgebraError {
                fusion_algebra_error(error)
            }
        }
    };
}

parity_codec!(
    Z2Codec,
    ZN,
    FsZNIrrep,
    |odd: bool| FsZNIrrep::from_canonical(u32::from(odd), 2)
        .expect("a parity is a canonical charge modulo 2"),
    |object: &FsZNIrrep| object.charge() == 1
);
parity_codec!(
    Fz2Codec,
    FermionParity,
    FermionParityIrrep,
    FermionParityIrrep::new,
    |object: &FermionParityIrrep| object.is_odd()
);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct Z2FusionRule;

static Z2: LazyLock<ZN> = LazyLock::new(|| ZN::new(2).expect("2 is a nonzero modulus"));
static Z2_IDENTITY: LazyLock<RuleIdentity> = LazyLock::new(RuleIdentity::of_type::<Z2FusionRule>);

impl crate::seal::Provider for Z2FusionRule {}

crate::category_rule::multiplicity_free_adapter_impls!(Z2FusionRule, Z2Irrep, f64);

impl AdapterAccess for Z2FusionRule {
    type Category = ZN;
    type Codec = Z2Codec;

    fn category(&self) -> &ZN {
        &Z2
    }
    fn codec(&self) -> &Z2Codec {
        &Z2Codec
    }
    fn identity(&self) -> &RuleIdentity {
        &Z2_IDENTITY
    }
    fn vacuum_id(&self) -> SectorId {
        Z2Irrep::EVEN.sector_id()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct FermionParityFusionRule;

static FERMION_PARITY_IDENTITY: LazyLock<RuleIdentity> =
    LazyLock::new(RuleIdentity::of_type::<FermionParityFusionRule>);

impl crate::seal::Provider for FermionParityFusionRule {}

crate::category_rule::multiplicity_free_adapter_impls!(FermionParityFusionRule, Z2Irrep, f64);

impl AdapterAccess for FermionParityFusionRule {
    type Category = FermionParity;
    type Codec = Fz2Codec;

    fn category(&self) -> &FermionParity {
        &FermionParity
    }
    fn codec(&self) -> &Fz2Codec {
        &Fz2Codec
    }
    fn identity(&self) -> &RuleIdentity {
        &FERMION_PARITY_IDENTITY
    }
    fn vacuum_id(&self) -> SectorId {
        Z2Irrep::EVEN.sector_id()
    }
}

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
            Ok(mut fused) => {
                Ok(Self::from_fs(fused.next().expect(
                    "a successful U(1) fusion has exactly one channel",
                )))
            }
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

static U1_IDENTITY: LazyLock<RuleIdentity> = LazyLock::new(RuleIdentity::of_type::<U1FusionRule>);

/// `U1Irrep` labels and zigzag ids over `U1`.
///
/// Why `N` also checks the fusion: symtn's checked algebra reports a sum
/// outside the label range as `U1FusionOverflow` (stricter than
/// TensorKitSectors' `0`), so a caller that enumerates channels and one that
/// asks for `N` fail alike.
#[derive(Debug)]
pub struct U1Codec;

impl U1Codec {
    fn overflow(left: &FsU1Irrep, right: &FsU1Irrep) -> FusionAlgebraError {
        FusionAlgebraError::U1FusionOverflow {
            left: left.charge(),
            right: right.charge(),
        }
    }
}

/// TensorKit `findindex - 1` of an integer U(1) charge, shared by
/// [`U1FusionRule`] and the Lie-group rule's torus charges; `None` past `u64`.
pub(crate) fn u1_order_position(charge: i64) -> Option<u64> {
    4u64.checked_mul(charge.unsigned_abs())
        .map(|position| position - u64::from(charge > 0))
}

impl CategoryCodec<U1> for U1Codec {
    type Sector = U1Irrep;
    type Error = FusionAlgebraError;
    type Mode = MultiplicityFreeAdmissionMode;
    type Unit = CanonicalUnitAdmission;

    fn encode_label<'a>(
        &self,
        _category: impl Fn() -> &'a U1 + Copy,
        rule: &RuleIdentity,
        label: &U1Irrep,
    ) -> Result<SectorId, FusionAlgebraError> {
        if U1Irrep::try_new(label.charge()).is_some() {
            Ok(label.sector_id())
        } else {
            Err(FusionAlgebraError::UnrepresentableSectorLabel {
                rule: rule.clone(),
                label: label.charge().to_string(),
            })
        }
    }

    fn decode_label<'a>(
        &self,
        _category: impl Fn() -> &'a U1 + Copy,
        id: SectorId,
    ) -> Result<U1Irrep, FusionAlgebraError> {
        checked_u1_irrep(id)
    }

    fn object<'a>(
        &self,
        _category: impl Fn() -> &'a U1 + Copy,
        id: SectorId,
    ) -> Result<FsU1Irrep, FusionAlgebraError> {
        checked_u1_irrep(id).map(U1Irrep::fs)
    }

    fn sector_id<'a>(
        &self,
        _category: impl Fn() -> &'a U1 + Copy,
        _rule: &RuleIdentity,
        object: &FsU1Irrep,
    ) -> Result<SectorId, FusionAlgebraError> {
        Ok(U1Irrep::from_fs(*object).sector_id())
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
        SectorOrderKey::position(
            u1_order_position(i64::from(charge)).expect("an i32 charge's position fits u64"),
        )
    }

    fn admit<'a, const N: usize>(
        &self,
        category: impl Fn() -> &'a U1 + Copy,
        rule: &RuleIdentity,
        operation: CategoryOperation,
        ids: [SectorId; N],
    ) -> Result<[FsU1Irrep; N], FusionAlgebraError> {
        let objects = admit_in_order(self, category, rule, operation, ids)?;
        if let (CategoryOperation::N, [left, right, ..]) = (operation, objects.as_slice()) {
            if category().fusion(left, right).is_err() {
                return Err(Self::overflow(left, right));
            }
        }
        Ok(objects)
    }

    fn map_error(
        &self,
        inputs: &[SectorId],
        error: CategoryAdapterError<U1Error>,
    ) -> FusionAlgebraError {
        // The only U(1) failure is a fusion leaving the charge range.
        if let (CategoryAdapterError::Provider(_), &[left, right]) = (&error, inputs) {
            if let (Some(left), Some(right)) = (
                U1Irrep::from_sector_id(left),
                U1Irrep::from_sector_id(right),
            ) {
                return Self::overflow(&left.fs(), &right.fs());
            }
        }
        fusion_algebra_error(error)
    }
}

impl crate::seal::Provider for U1FusionRule {}

crate::category_rule::multiplicity_free_adapter_impls!(U1FusionRule, U1Irrep, f64);

impl AdapterAccess for U1FusionRule {
    type Category = U1;
    type Codec = U1Codec;

    fn category(&self) -> &U1 {
        &U1
    }
    fn codec(&self) -> &U1Codec {
        &U1Codec
    }
    fn identity(&self) -> &RuleIdentity {
        &U1_IDENTITY
    }
    fn vacuum_id(&self) -> SectorId {
        U1Irrep::new(0).sector_id()
    }
}

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
        // `try_nsymbol` admitted all three labels.
        let [left, right, coupled] = [left, right, coupled].map(|sector| {
            U1Irrep::from_sector_id(sector)
                .expect("admitted by try_nsymbol")
                .fs()
        });
        U1.cgc(&left, &right, &coupled).map_err(|error| {
            PhysicalBasisError::FusionAlgebra(FusionAlgebraError::CategoryFailure(
                CategoryFailure::new(error),
            ))
        })
    }
}

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
    fn u1_physical_coefficient_is_the_fusion_symbols_cgc() {
        // What: the one-dimensional U(1) coefficient comes from
        // fusion-symbols' group representations, after symtn's validation.
        use fusion_symbols::{category::GroupRepresentations, lie};
        let fs = |charge| lie::U1Irrep::new(charge).unwrap();
        for (left, right) in [
            (0, 0),
            (2, -1),
            (-7, 3),
            (i32::MAX, 0),
            (i32::MIN + 1, i32::MAX),
        ] {
            let coupled = left + right;
            let expected = lie::U1.cgc(&fs(left), &fs(right), &fs(coupled)).unwrap();
            assert_eq!(expected, 1.0);
            assert_eq!(
                U1FusionRule.try_fusion_tensor_element(
                    U1Irrep::new(left).into(),
                    U1Irrep::new(right).into(),
                    U1Irrep::new(coupled).into(),
                    0,
                    0,
                    0,
                    0,
                ),
                Ok(expected)
            );
        }
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
