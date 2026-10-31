use std::sync::Arc;

use symtn::expert::SectorLeg;
use symtn::sector::SectorId;
use symtn::sector::{
    BraidingStyleKind, CheckedFusionAlgebra, FusionRule, FusionStyleKind, RuleIdentity,
    SectorCodec, SectorVec,
};
use symtn::typed::FusionAlgebraError;
use symtn::typed::GradedSpace;

/// A faulty rule whose dual is not an involution: `a* = b* = c`,
/// `c* = a`, `d* = b`. `{a, b}` and `{c, d}` then pass a one-way sector
/// match in both directions although `{a, b}` has no dual leg.
struct CollapsingDual;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct Label(usize);

const DUAL: [usize; 4] = [2, 2, 0, 1];

impl symtn_core::testing::Provider for CollapsingDual {}

impl symtn::sector::TypedSectorAdmission for CollapsingDual {
    type Sector = <Self as symtn::sector::SectorCodec>::Sector;
    type Error = symtn::typed::FusionAlgebraError;
    type Mode = symtn::sector::MultiplicityFreeAdmissionMode;
    fn typed_rule_identity(&self) -> symtn::sector::RuleIdentity {
        symtn::sector::FusionRule::rule_identity(self)
    }
    fn try_encode_label(
        &self,
        sector: &Self::Sector,
    ) -> Result<symtn::sector::SectorId, Self::Error> {
        symtn::sector::SectorCodec::encode_sector(self, sector)
    }
    fn try_decode_label(
        &self,
        sector: symtn::sector::SectorId,
    ) -> Result<Self::Sector, Self::Error> {
        symtn::sector::SectorCodec::decode_sector(self, sector)
    }
    fn try_dual_id(
        &self,
        sector: symtn::sector::SectorId,
    ) -> Result<symtn::sector::SectorId, Self::Error> {
        symtn::sector::CheckedFusionAlgebra::try_dual_sector(self, sector)
    }
}

impl FusionRule for CollapsingDual {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::from_canonical_bytes::<Self>(0x1371_d0a1, Arc::<[u8]>::from([]))
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }
    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }
    fn fusion_channels(&self, _: SectorId, _: SectorId) -> SectorVec {
        core::iter::once(SectorId::new(0)).collect()
    }
}

impl CheckedFusionAlgebra for CollapsingDual {
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        Ok(SectorId::new(DUAL[sector.id()]))
    }
    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        Ok(self.fusion_channels(left, right))
    }
    fn try_nsymbol(
        &self,
        _: SectorId,
        _: SectorId,
        _: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        Ok(1)
    }
}

impl SectorCodec for CollapsingDual {
    type Sector = Label;
    fn encode_sector(&self, label: &Label) -> Result<SectorId, FusionAlgebraError> {
        Ok(SectorId::new(label.0))
    }
    fn decode_sector(&self, sector: SectorId) -> Result<Label, FusionAlgebraError> {
        Ok(Label(sector.id()))
    }
}

/// The allocation-free leg comparison enforces the dual
/// round trip, so a non-involutive provider is rejected with
/// `GradedSpace::try_dual`'s own error rather than accepted.
#[test]
fn a_non_involutive_dual_is_rejected_like_try_dual() {
    let leg = |sectors: [usize; 2], dual: bool| {
        SectorLeg::try_new(sectors.map(|sector| (SectorId::new(sector), 2)), dual).unwrap()
    };
    // `{a, b}` against `{c, d}` of the opposite flag: every sector of either
    // side has its dual on the other, so only the round trip rejects.
    let (from, into) = (leg([0, 1], false), leg([2, 3], true));
    let oracle = GradedSpace::try_new(Arc::new(CollapsingDual), [(Label(0), 2), (Label(1), 2)])
        .unwrap()
        .try_dual()
        .unwrap_err()
        .to_string();
    let error = super::legs_contract(&CollapsingDual, (&from, false), (&into, false))
        .unwrap_err()
        .to_string();
    assert_eq!(error, oracle);
    assert!(error.contains("dual map is not injective"), "{error}");
}
