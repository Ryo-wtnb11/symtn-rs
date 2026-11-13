//! Z_N, fermion parity and CU(1): fusion-symbols categories under the
//! enumerated codec.
//!
//! Each rule is a [`CategoryRule`] over the fusion-symbols category with
//! multiplicity-free [`EnumeratedCodec`] admission. The category supplies
//! every symbol in TensorKitSectors 0.3.9's conventions, the public labels
//! are the category's own objects, and a `SectorId` is the object's frozen
//! enumeration index: the charge for Z_N, `0` even and `1` odd for fermion
//! parity, and `0+`, `0-`, then `2j + 1` for CU(1).

use fusion_symbols::{
    category::FusionCategory,
    fermion::{FermionParity, FermionParityIrrep},
    finite::{ZNIrrep, ZN},
    lie::CU1,
};

use crate::{
    CategoryRule, EnumeratedCodec, FusionAlgebraError, MultiplicityFreeAdmissionMode, RuleIdentity,
    SectorId,
};

type Enumerated = EnumeratedCodec<MultiplicityFreeAdmissionMode>;

/// TensorKit's `ZNIrrep{N}` fusion rule; `Z2Irrep` is `N = 2`.
///
/// Labels are fusion-symbols' [`ZNIrrep`], built with
/// `ZNIrrep::new(charge, modulus)`; the modulus is the category's.
///
/// ```
/// use symtn_sectors::{SectorCodec, SectorId, ZNFusionRule, ZNIrrep};
///
/// let z3 = ZNFusionRule::new(3).unwrap();
/// let modulus = z3.category().modulus();
/// let two = ZNIrrep::new(-1, modulus).unwrap();
/// assert_eq!(z3.encode_sector(&two), Ok(SectorId::new(2)));
/// ```
pub type ZNFusionRule = CategoryRule<ZN, Enumerated>;

/// TensorKit's `FermionParity` fusion rule.
pub type FermionParityFusionRule = CategoryRule<FermionParity, Enumerated>;

/// TensorKit's `CU1Irrep` fusion rule.
///
/// A fusion whose channel leaves fusion-symbols' `u32` twice-charge range is
/// `FusionNotRepresentable`. `N` is the category's exact value even there,
/// as TensorKit's `Nsymbol` is: it compares charges and never builds the
/// unrepresentable channel.
pub type CU1FusionRule = CategoryRule<CU1, Enumerated>;

/// The rule of a category whose admission is known to succeed.
fn admitted<C>(category: C) -> CategoryRule<C, Enumerated>
where
    C: FusionCategory + Send + Sync + 'static,
    C::Error: Send + Sync + 'static,
{
    // Z_N, fermion parity and CU(1) are multiplicity-free, have a known
    // braiding style and a canonical unit at enumeration index 0, so the
    // codec admits each of them.
    CategoryRule::try_new(category, EnumeratedCodec::new())
        .expect("Z_N, fermion parity and CU(1) are admitted by the enumerated codec")
}

impl ZNFusionRule {
    /// The Z_N rule of `modulus`.
    ///
    /// # Errors
    ///
    /// `UnrepresentableSectorLabel` for a zero modulus.
    pub fn new(modulus: u32) -> Result<Self, FusionAlgebraError> {
        let category =
            ZN::new(modulus).map_err(|_| FusionAlgebraError::UnrepresentableSectorLabel {
                rule: RuleIdentity::of_type::<Self>(),
                label: "Z_N modulus must be nonzero".into(),
            })?;
        Ok(admitted(category))
    }
}

impl FermionParityFusionRule {
    pub fn new() -> Self {
        admitted(FermionParity)
    }
}

impl Default for FermionParityFusionRule {
    fn default() -> Self {
        Self::new()
    }
}

impl CU1FusionRule {
    pub fn new() -> Self {
        admitted(CU1)
    }
}

impl Default for CU1FusionRule {
    fn default() -> Self {
        Self::new()
    }
}

// Why no `From<CU1Irrep>`: the index `2j + 1` of the largest charges does
// not fit a 32-bit `usize`; encode CU(1) labels through the rule instead.

impl From<ZNIrrep> for SectorId {
    fn from(value: ZNIrrep) -> Self {
        Self::new(value.charge() as usize)
    }
}

impl From<FermionParityIrrep> for SectorId {
    fn from(value: FermionParityIrrep) -> Self {
        Self::new(usize::from(value.is_odd()))
    }
}

#[cfg(test)]
mod tests {
    use fusion_symbols::finite::ZNIrrep;

    use super::ZNFusionRule;
    use crate::{
        CheckedFusionAlgebra, FusionAlgebraError, FusionRule, MultiplicityFreeFusionSymbols,
        SectorCodec, SectorId,
    };

    #[test]
    fn zn_identity_contains_the_modulus_and_zero_is_rejected() {
        let three = || ZNFusionRule::new(3).unwrap();
        assert_ne!(
            three().rule_identity(),
            ZNFusionRule::new(4).unwrap().rule_identity()
        );
        assert_eq!(three().rule_identity(), three().rule_identity());
        assert!(matches!(
            ZNFusionRule::new(0),
            Err(FusionAlgebraError::UnrepresentableSectorLabel { .. })
        ));
    }

    #[test]
    fn zn_rejects_an_id_past_the_modulus_and_a_foreign_label() {
        let rule = ZNFusionRule::new(3).unwrap();
        assert_eq!(
            rule.try_dual_sector(SectorId::new(3)),
            Err(FusionAlgebraError::InvalidSector {
                sector: SectorId::new(3)
            })
        );
        assert!(matches!(
            rule.encode_sector(&ZNIrrep::new(1, 4).unwrap()),
            Err(FusionAlgebraError::UnrepresentableSectorLabel { .. })
        ));
    }

    #[test]
    fn zn_ids_round_trip_and_fuse_at_the_u32_boundary() {
        for modulus in [1, 2, 3, 17, u32::MAX] {
            let rule = ZNFusionRule::new(modulus).unwrap();
            let label = |charge: i64| ZNIrrep::new(charge, modulus).unwrap();
            for charge in [i64::MIN, -1, 0, 1, i64::MAX] {
                let sector = SectorId::from(label(charge));
                assert_eq!(rule.encode_sector(&label(charge)), Ok(sector));
                assert_eq!(rule.decode_sector(sector), Ok(label(charge)));
                assert_eq!(
                    rule.try_dual_sector(rule.try_dual_sector(sector).unwrap()),
                    Ok(sector)
                );
            }
            let a = SectorId::from(label(u32::MAX.into()));
            let b = SectorId::from(label(1));
            let fused = rule.try_fusion_channels(a, b).unwrap()[0];
            assert_eq!(fused, label(i64::from(u32::MAX) + 1).into());
            assert_eq!(
                rule.f_symbol_scalar(a, b, rule.vacuum(), fused, fused, b),
                1.0
            );
            assert_eq!(rule.r_symbol_scalar(a, b, fused), 1.0);
        }
    }
}
