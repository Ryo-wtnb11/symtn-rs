use super::*;

// Finite provider over sectors {0, 1}; any other sector is outside its table.
#[derive(Clone, Copy, Debug)]
struct TwoSectorTable;

impl TwoSectorTable {
    fn check(sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        (sector.id() < 2)
            .then_some(sector)
            .ok_or(FusionAlgebraError::InvalidSector { sector })
    }
}

impl symtn_sectors::testing::Provider for TwoSectorTable {}

impl FusionRule for TwoSectorTable {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
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
    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        smallvec![SectorId::new(left.id() ^ right.id())]
    }
}

impl CheckedFusionAlgebra for TwoSectorTable {
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        Self::check(sector)
    }
    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        Self::check(left)?;
        Self::check(right)?;
        Ok(self.fusion_channels(left, right))
    }
    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        Self::check(left)?;
        Self::check(right)?;
        Self::check(coupled)?;
        Ok(self.nsymbol(left, right, coupled))
    }
}

impl CheckedGenericFusion for TwoSectorTable {
    type Error = FusionAlgebraError;
    fn rule_identity(&self) -> RuleIdentity {
        FusionRule::rule_identity(self)
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionRule::fusion_style(self)
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        FusionRule::braiding_style(self)
    }
    fn vacuum(&self) -> SectorId {
        FusionRule::vacuum(self)
    }
    fn try_dual(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        Self::check(sector)
    }
    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        CheckedFusionAlgebra::try_fusion_channels(self, left, right)
    }
    fn try_fusion_channels_in_table(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        CheckedFusionAlgebra::try_fusion_channels(self, left, right)
    }
    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        CheckedFusionAlgebra::try_nsymbol(self, left, right, coupled)
    }
}

#[test]
fn checked_generic_split_rejects_out_of_table_rank1_tree_like_multiplicity_free() {
    // What: both checked validators query the provider for a rank-1 tree's
    // sector, so an out-of-table sector is the provider's typed error on
    // the Generic path, exactly as on the multiplicity-free path.
    let sector = SectorId::new(1 << 40);
    let tree = FusionTreeKey::new([sector], sector, [false], [], []);
    let invalid = FusionAlgebraError::InvalidSector { sector };

    let generic = split_fusion_tree_generic_checked(&TwoSectorTable, &tree, 1).unwrap_err();
    assert!(matches!(
        generic,
        CheckedGenericStructureError::Provider(ref error) if *error == invalid
    ));

    assert!(matches!(
        validate_fusion_tree_for_rule_checked(&TwoSectorTable, &tree),
        Err(CheckedFusionSpaceError::FusionAlgebra(error)) if *error == invalid
    ));
}

#[test]
fn checked_generic_split_admits_in_table_rank1_tree() {
    let sector = SectorId::new(1);
    let tree = FusionTreeKey::new([sector], sector, [false], [], []);
    let (front, tail) = split_fusion_tree_generic_checked(&TwoSectorTable, &tree, 1).unwrap();
    assert_eq!(front, tree);
    assert_eq!(tail, tree);
}
