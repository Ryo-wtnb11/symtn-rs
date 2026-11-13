//! U(1) as a symtn fusion rule.
//!
//! The rule is the fusion-symbols category `U1` behind the shared category
//! adapter, which supplies every symbol (dual, fusion, N, F, R, A, B, dims,
//! twist, Frobenius–Schur phase) in TensorKitSectors 0.3.9's conventions.
//! The label is fusion-symbols' `U1Irrep`; this module owns only its
//! `SectorId` codec, TensorKit order keys, identity, and checked error
//! contract. An inadmissible F or R is `0.0`, as in TensorKitSectors. A `SectorId`
//! outside the rule's domain panics in the infallible symbols, as in the
//! other providers.

use std::sync::LazyLock;

use fusion_symbols::{
    category::{FusionCategory, GroupRepresentations},
    lie::{U1Error, U1Irrep, U1},
};

use crate::category_rule::{
    admit_in_order, fusion_algebra_error, AdapterAccess, CanonicalUnitAdmission,
    CategoryAdapterError, CategoryCodec, CategoryFailure, CategoryOperation,
};
use crate::{
    CheckedFusionAlgebra, FusionAlgebraError, MultiplicityFreeAdmissionMode, PhysicalBasisError,
    PhysicalFusionBasis, RuleIdentity, SectorId, SectorOrderKey,
};

const fn u1_charge_to_zigzag_u32(charge: i32) -> u32 {
    ((charge as u32) << 1) ^ ((charge >> 31) as u32)
}

const fn u1_charge_from_zigzag_u32(encoded: u32) -> i32 {
    ((encoded >> 1) as i32) ^ -((encoded & 1) as i32)
}

/// The label of a U(1) zigzag id, `None` outside the label domain.
fn u1_irrep(sector: SectorId) -> Option<U1Irrep> {
    let encoded = u32::try_from(sector.id()).ok()?;
    U1Irrep::new(u1_charge_from_zigzag_u32(encoded)).ok()
}

fn checked_u1_irrep(sector: SectorId) -> Result<U1Irrep, FusionAlgebraError> {
    u1_irrep(sector).ok_or(FusionAlgebraError::InvalidSector { sector })
}

impl From<U1Irrep> for SectorId {
    fn from(value: U1Irrep) -> Self {
        Self::new(u1_charge_to_zigzag_u32(value.charge()) as usize)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct U1FusionRule;

static U1_IDENTITY: LazyLock<RuleIdentity> = LazyLock::new(RuleIdentity::of_type::<U1FusionRule>);

/// `U1Irrep` labels and zigzag ids over `U1`.
///
/// Why `N` also checks the fusion: symtn's checked algebra reports a sum
/// outside the label range as `FusionNotRepresentable` (stricter than
/// TensorKitSectors' `0`), so a caller that enumerates channels and one that
/// asks for `N` fail alike.
#[derive(Debug)]
pub struct U1Codec;

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
        _rule: &RuleIdentity,
        label: &U1Irrep,
    ) -> Result<SectorId, FusionAlgebraError> {
        Ok(SectorId::from(*label))
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
    ) -> Result<U1Irrep, FusionAlgebraError> {
        checked_u1_irrep(id)
    }

    fn sector_id<'a>(
        &self,
        _category: impl Fn() -> &'a U1 + Copy,
        _rule: &RuleIdentity,
        object: &U1Irrep,
    ) -> Result<SectorId, FusionAlgebraError> {
        Ok(SectorId::from(*object))
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
    ) -> Result<[U1Irrep; N], FusionAlgebraError> {
        let objects = admit_in_order(self, category, rule, operation, ids)?;
        if let (CategoryOperation::N, [left, right, ..]) = (operation, objects.as_slice()) {
            if category().fusion(left, right).is_err() {
                return Err(FusionAlgebraError::FusionNotRepresentable {
                    left: ids[0],
                    right: ids[1],
                });
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
            if u1_irrep(left).is_some() && u1_irrep(right).is_some() {
                return FusionAlgebraError::FusionNotRepresentable { left, right };
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
        SectorId::from(U1Irrep::UNIT)
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
        let [left, right, coupled] =
            [left, right, coupled].map(|sector| u1_irrep(sector).expect("admitted by try_nsymbol"));
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
        u1_charge_from_zigzag_u32, u1_charge_to_zigzag_u32, u1_irrep, U1FusionRule, U1Irrep,
    };
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
        assert_canonical_unit(&U1FusionRule, U1Irrep::new(-17).unwrap().into());
        let u1_boundary: SectorId = U1Irrep::new(i32::MAX).unwrap().into();
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
            if let Ok(irrep) = U1Irrep::new(charge) {
                let sector = SectorId::from(irrep);
                assert_eq!(sector.id(), encoded as usize);
                assert_eq!(u1_irrep(sector), Some(irrep));
            }
        }
    }

    #[test]
    fn u1_label_boundary_rejects_only_i32_min() {
        // What: i32::MIN has no label, while both adjacent representable
        // extremes encode, decode, and dualize normally.
        let invalid = SectorId::new(u32::MAX as usize);
        assert!(U1Irrep::new(i32::MIN).is_err());
        assert_eq!(u1_irrep(invalid), None);
        assert_eq!(
            U1FusionRule.decode_sector(invalid),
            Err(FusionAlgebraError::InvalidSector { sector: invalid })
        );

        for charge in [i32::MIN + 1, i32::MAX] {
            let irrep = U1Irrep::new(charge).unwrap();
            let sector = SectorId::from(irrep);
            assert_eq!(U1FusionRule.encode_sector(&irrep), Ok(sector));
            assert_eq!(u1_irrep(sector), Some(irrep));
            assert_eq!(
                U1FusionRule
                    .try_dual_sector(sector)
                    .and_then(|dual| U1FusionRule.try_dual_sector(dual)),
                Ok(sector)
            );
        }
    }

    #[test]
    fn u1_physical_basis_is_the_checked_one_dimensional_embedding() {
        // What: U(1) exposes exactly one carrier state and one unit CGC for an
        // allowed charge sum, while invalid physical indices remain typed.
        let rule = U1FusionRule;
        let left: SectorId = U1Irrep::new(2).unwrap().into();
        let right: SectorId = U1Irrep::new(-1).unwrap().into();
        let coupled: SectorId = U1Irrep::new(1).unwrap().into();
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
        let absent: SectorId = U1Irrep::new(0).unwrap().into();
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
                    U1Irrep::new(left).unwrap().into(),
                    U1Irrep::new(right).unwrap().into(),
                    U1Irrep::new(coupled).unwrap().into(),
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
