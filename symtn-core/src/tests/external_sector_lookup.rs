//! Unique fusion-tree block lookup from external sector tuples.

use super::*;

#[derive(Clone, Copy, Debug)]
struct Z2xZ3PointedRule;

impl Z2xZ3PointedRule {
    const fn encode(z2: usize, z3: usize) -> SectorId {
        SectorId::new((z2 % 2) + 2 * (z3 % 3))
    }

    const fn decode(sector: SectorId) -> (usize, usize) {
        (sector.id() % 2, (sector.id() / 2) % 3)
    }
}

impl FusionRule for Z2xZ3PointedRule {
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
        Self::encode(0, 0)
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        let (z2, z3) = Self::decode(sector);
        Self::encode((2 - z2) % 2, (3 - z3) % 3)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        let (left_z2, left_z3) = Self::decode(left);
        let (right_z2, right_z3) = Self::decode(right);
        smallvec![Self::encode(
            (left_z2 + right_z2) % 2,
            (left_z3 + right_z3) % 3,
        )]
    }
}

impl MultiplicityFreeFusionRule for Z2xZ3PointedRule {}

/// The storage offset of the unique fusion-tree block that `sectors` (every
/// codomain then every domain external sector) select.
fn sector_block_offset<R: MultiplicityFreeFusionRule>(
    homspace: &FusionTreeHomSpace,
    structure: &BlockStructure,
    rule: &R,
    sectors: &[SectorId],
) -> usize {
    let key = homspace
        .unique_fusion_tree_key_from_external_sectors(rule, sectors)
        .unwrap();
    structure.fusion_tree_pair_block(&key).unwrap().offset()
}

#[test]
fn product_sector_lookup_handles_simple_fusion_channels_without_manual_tree_keys() {
    type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
    type FpU1Su2Rule = ProductFusionRule<FpU1Rule, SU2FusionRule>;
    let left_rule = FpU1Rule::default();
    let rule = FpU1Su2Rule::default();
    let left_sector = |parity, charge| left_rule.encode_component_ids(parity, u1(charge));
    let sector = |parity, charge, twice_spin| {
        rule.encode_component_ids(left_sector(parity, charge), su2(twice_spin))
    };

    let a = sector(z2_odd(), 1, 1);
    let b = sector(z2_odd(), -1, 1);
    let c0 = sector(z2_even(), 0, 0);
    let c1 = sector(z2_even(), 0, 2);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(a, 1)], false),
            SectorLeg::new([(b, 1)], false),
        ]),
        FusionProductSpace::new([SectorLeg::new([(c0, 1), (c1, 1)], false)]),
    );
    let structure = hom
        .coupled_subblock_structure(&rule, 2, [vec![1, 1, 1], vec![1, 1, 1]])
        .unwrap();

    assert_eq!(sector_block_offset(&hom, &structure, &rule, &[a, b, c0]), 0);
    assert_eq!(sector_block_offset(&hom, &structure, &rule, &[a, b, c1]), 1);
    let all_c0_keys = hom
        .fusion_tree_keys_from_external_sectors(&rule, &[a, b, c0])
        .unwrap();
    assert_eq!(all_c0_keys.len(), 1);
}

#[test]
fn sector_lookup_matches_z2_unique() {
    let rule = Z2FusionRule;
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new(
            [(SectorId::new(0), 1), (SectorId::new(1), 1)],
            false,
        )]),
        FusionProductSpace::new([SectorLeg::new(
            [(SectorId::new(0), 1), (SectorId::new(1), 1)],
            false,
        )]),
    );
    let structure = hom
        .coupled_subblock_structure(&rule, 1, [vec![1, 1], vec![1, 1]])
        .unwrap();

    assert_eq!(
        sector_block_offset(
            &hom,
            &structure,
            &rule,
            &[SectorId::new(1), SectorId::new(1)]
        ),
        1
    );
}

#[test]
fn sector_lookup_dualizes_z4_domain_sector() {
    let rule = Z4PointedRule;
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(SectorId::new(1), 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(SectorId::new(1), 1)], false)]),
    );
    let structure = hom
        .coupled_subblock_structure(&rule, 1, [vec![1, 1]])
        .unwrap();

    assert_eq!(
        sector_block_offset(
            &hom,
            &structure,
            &rule,
            &[SectorId::new(1), SectorId::new(3)]
        ),
        0
    );
}

#[test]
fn sector_lookup_handles_fermionic_z2_key() {
    let rule = FermionParityFusionRule;
    let hom = FusionTreeHomSpace::from_sector_ids([(1, 1)], [(1, 1)]);
    let structure = hom
        .coupled_subblock_structure(&rule, 1, [vec![1, 1]])
        .unwrap();

    assert_eq!(
        sector_block_offset(
            &hom,
            &structure,
            &rule,
            &[SectorId::new(1), SectorId::new(1)]
        ),
        0
    );
}

#[test]
fn sector_lookup_handles_product_pointed_rule() {
    let rule = Z2xZ3PointedRule;
    let codomain_sector = Z2xZ3PointedRule::encode(1, 2);
    let domain_tree_sector = rule.dual(Z2xZ3PointedRule::encode(1, 1));
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(codomain_sector, 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(domain_tree_sector, 1)], false)]),
    );
    let structure = hom
        .coupled_subblock_structure(&rule, 1, [vec![1, 1]])
        .unwrap();

    assert_eq!(
        sector_block_offset(
            &hom,
            &structure,
            &rule,
            &[codomain_sector, Z2xZ3PointedRule::encode(1, 1)]
        ),
        0
    );
}
