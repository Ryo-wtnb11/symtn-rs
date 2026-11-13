//! U(1) as a symtn fusion rule.
//!
//! The rule is the fusion-symbols category `U1` behind the shared category
//! adapter, which supplies every symbol (dual, fusion, N, F, R, A, B, dims,
//! twist, Frobenius–Schur phase) in TensorKitSectors 0.3.9's conventions.
//! This module owns only symtn's label, `SectorId` codec, TensorKit order
//! keys, identity, and checked error contract. An
//! inadmissible F or R is `0.0`, as in TensorKitSectors. A `SectorId`
//! outside the rule's domain panics in the infallible symbols, as in the
//! other providers.

use std::sync::LazyLock;

use fusion_symbols::{
    category::{FusionCategory, GroupRepresentations},
    lie::{U1Error, U1Irrep as FsU1Irrep, U1},
};

use crate::category_rule::{
    admit_in_order, fusion_algebra_error, AdapterAccess, CanonicalUnitAdmission,
    CategoryAdapterError, CategoryCodec, CategoryFailure, CategoryOperation,
};
use crate::{
    CheckedFusionAlgebra, FusionAlgebraError, MultiplicityFreeAdmissionMode, PhysicalBasisError,
    PhysicalFusionBasis, RuleIdentity, SectorId, SectorOrderKey,
};

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
    use super::{u1_charge_from_zigzag_u32, u1_charge_to_zigzag_u32, U1FusionRule, U1Irrep};
    use crate::{
        CanonicalUnitFusionRule, CheckedFusionAlgebra, FusionAlgebraError, FusionRule,
        MultiplicityFreeFusionSymbols, PhysicalBasisError, PhysicalFusionBasis, SectorCodec,
        SectorId,
    };
    use crate::{FermionParityFusionRule, FermionParityIrrep, ZNFusionRule};

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
        assert_canonical_unit(&ZNFusionRule::new(2).unwrap(), SectorId::new(1));
        assert_canonical_unit(
            &FermionParityFusionRule::new(),
            FermionParityIrrep::ODD.into(),
        );
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
}
