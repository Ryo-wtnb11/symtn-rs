use symtn_core::{
    BlockKey, BlockSpec, BlockStructure, BraidingStyleKind, CoreError, FusionProductSpace,
    FusionRule, FusionStyleKind, FusionTreeHomSpace, FusionTreeKey, FusionTreePairKey,
    MultiplicityFreeFusionRule, MultiplicityIndex, RuleIdentity, SectorId, SectorLeg, SectorVec,
    Z2FusionRule,
};

#[derive(Clone, Debug)]
struct StatefulPointedRule {
    identity: RuleIdentity,
    one_times_one: SectorId,
}

impl StatefulPointedRule {
    fn new(one_times_one: usize) -> Self {
        Self {
            identity: RuleIdentity::new_unique::<Self>(),
            one_times_one: SectorId::new(one_times_one),
        }
    }
}

impl symtn_core::testing::Provider for StatefulPointedRule {}

impl FusionRule for StatefulPointedRule {
    fn rule_identity(&self) -> RuleIdentity {
        self.identity.clone()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Simple
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        let output = if left.id() == 1 && right.id() == 1 {
            self.one_times_one
        } else {
            SectorId::new(left.id().max(right.id()))
        };
        [output].into_iter().collect()
    }
}

impl MultiplicityFreeFusionRule for StatefulPointedRule {}

#[test]
fn core_rule_identity_reexports_the_symtn_sectors_type() {
    let _: symtn_sectors::RuleIdentity = RuleIdentity::of_type::<StatefulPointedRule>();
}

fn two_ones_to_vacuum_homspace() -> FusionTreeHomSpace {
    FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::from_sector_id(1, 1),
            SectorLeg::from_sector_id(1, 1),
        ]),
        FusionProductSpace::new([]),
    )
}

#[test]
fn same_rule_type_with_distinct_semantics_does_not_share_fusion_tree_layout() {
    let fuses_to_vacuum = StatefulPointedRule::new(0);
    let fuses_to_one = StatefulPointedRule::new(1);
    let homspace = two_ones_to_vacuum_homspace();

    assert_eq!(homspace.fusion_tree_keys(&fuses_to_vacuum).len(), 1);
    assert_eq!(homspace.fusion_tree_keys(&fuses_to_one).len(), 0);
}

fn raw_scalar_layout_with_key(key: BlockKey) -> (FusionTreeHomSpace, BlockStructure) {
    (
        FusionTreeHomSpace::from_sector_ids([], []),
        BlockStructure::from_blocks(vec![
            BlockSpec::column_major_with_key(key, vec![], 0).unwrap()
        ])
        .unwrap(),
    )
}

fn z2_rank_one_pair(sector: SectorId, is_dual: bool) -> FusionTreePairKey {
    let tree = FusionTreeKey::try_new_for_rule(&Z2FusionRule, [sector], sector, [is_dual], [], [])
        .unwrap();
    FusionTreePairKey::pair(tree.clone(), tree)
}

fn raw_z2_matrix_layout(
    leg_sector: SectorId,
    degeneracy: usize,
    key: FusionTreePairKey,
    shape: Vec<usize>,
) -> (FusionTreeHomSpace, BlockStructure) {
    let leg = || FusionProductSpace::new([SectorLeg::new([(leg_sector, degeneracy)], false)]);
    (
        FusionTreeHomSpace::new(leg(), leg()),
        BlockStructure::from_blocks_with_rank(
            2,
            vec![BlockSpec::column_major_with_key(key.into(), shape, 0).unwrap()],
        )
        .unwrap(),
    )
}

#[test]
fn binding_a_rule_rejects_non_categorical_block_namespaces() {
    // What: neither the anonymous dense key nor application routing metadata
    // can acquire a categorical rule identity.
    for key in [BlockKey::Dense, BlockKey::opaque([0])] {
        let (homspace, structure) = raw_scalar_layout_with_key(key);
        let error = homspace
            .validate_subblock_structure_subset(&Z2FusionRule, &structure)
            .unwrap_err();
        assert!(matches!(error, CoreError::ExpectedFusionTreePairKey { .. }));
    }
}

#[test]
fn rule_binding_requires_every_present_block_to_be_a_homspace_subset() {
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let split_codomain = FusionTreeKey::try_new_for_rule(
        &Z2FusionRule,
        [even, even],
        even,
        [false, false],
        [],
        [MultiplicityIndex::ONE],
    )
    .unwrap();
    let split_domain =
        FusionTreeKey::try_new_for_rule(&Z2FusionRule, [], even, [], [], []).unwrap();
    let cases = [
        (
            "codomain/domain split",
            raw_z2_matrix_layout(
                even,
                1,
                FusionTreePairKey::pair(split_codomain, split_domain),
                vec![1, 1],
            ),
            CoreError::FusionSpaceSplitMismatch {
                expected_nout: 1,
                expected_nin: 1,
                actual_nout: 2,
                actual_nin: 0,
            },
        ),
        (
            "leg membership",
            raw_z2_matrix_layout(even, 1, z2_rank_one_pair(odd, false), vec![1, 1]),
            CoreError::MalformedFusionTree {
                message: "fusion tree uses a sector absent from its HomSpace leg",
            },
        ),
        (
            "leg duality",
            raw_z2_matrix_layout(even, 1, z2_rank_one_pair(even, true), vec![1, 1]),
            CoreError::MalformedFusionTree {
                message: "fusion tree duality disagrees with its HomSpace leg",
            },
        ),
        (
            "leg degeneracy",
            raw_z2_matrix_layout(even, 2, z2_rank_one_pair(even, false), vec![1, 2]),
            CoreError::LegDegeneracyMismatch {
                sector: even,
                expected: 2,
                actual: 1,
            },
        ),
    ];

    for (name, (homspace, structure), expected) in cases {
        // What: a locally admissible tree cannot acquire a rule identity when
        // its external sectors, orientations, or logical shape disagree with
        // the declared HomSpace.
        assert_eq!(
            homspace.validate_subblock_structure_subset(&Z2FusionRule, &structure),
            Err(expected),
            "{name}"
        );
    }
}
