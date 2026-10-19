use std::hash::{Hash, Hasher};
use std::sync::Arc;
use symtn_core::{
    product_fusion_rule, BlockKey, BlockSpec, BlockStructure, BraidingStyleKind, CoreError,
    FusionProductSpace, FusionRule, FusionStyleKind, FusionTreeHomSpace, FusionTreeKey,
    FusionTreePairKey, MultiplicityFreeFusionRule, MultiplicityIndex, RuleIdentity, SectorId,
    SectorLeg, SectorVec, Z2FusionRule,
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

struct AlternateCanonicalRule;
struct FirstIdentityCodec;
struct SecondIdentityCodec;

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

#[test]
fn cloned_stateful_rule_preserves_identity() {
    let rule = StatefulPointedRule::new(0);

    assert_eq!(rule.rule_identity(), rule.clone().rule_identity());
}

#[derive(Default)]
struct CountingHasher(usize);

impl Hasher for CountingHasher {
    fn finish(&self) -> u64 {
        self.0 as u64
    }
    fn write(&mut self, bytes: &[u8]) {
        self.0 += bytes.len();
    }
}

#[test]
fn content_identity_hash_cost_does_not_scale_with_table_bytes() {
    let short = RuleIdentity::from_canonical_bytes::<StatefulPointedRule>(7, Arc::from([1u8]));
    let long = RuleIdentity::from_canonical_bytes::<StatefulPointedRule>(
        7,
        Arc::from(vec![2u8; 1_000_000]),
    );
    let mut short_hasher = CountingHasher::default();
    let mut long_hasher = CountingHasher::default();

    short.hash(&mut short_hasher);
    long.hash(&mut long_hasher);

    assert_eq!(short_hasher.0, long_hasher.0);
    assert_ne!(short, long);
}

#[test]
fn canonical_identity_keeps_prehash_and_provider_type_as_determinants() {
    let bytes = Arc::<[u8]>::from([1u8, 2, 3]);
    let same_type_other_prehash =
        RuleIdentity::from_canonical_bytes::<StatefulPointedRule>(8, Arc::clone(&bytes));
    let other_type = RuleIdentity::from_canonical_bytes::<AlternateCanonicalRule>(7, bytes);
    let baseline =
        RuleIdentity::from_canonical_bytes::<StatefulPointedRule>(7, Arc::from([1u8, 2, 3]));

    assert_ne!(baseline, same_type_other_prehash);
    assert_ne!(baseline, other_type);
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

#[test]
fn product_rule_identity_includes_stateful_child_identity() {
    let first = product_fusion_rule(StatefulPointedRule::new(0), Z2FusionRule);
    let second = product_fusion_rule(StatefulPointedRule::new(1), Z2FusionRule);

    assert_ne!(first.rule_identity(), second.rule_identity());
}

#[test]
fn product_identity_keeps_child_order_codec_and_retained_charge() {
    let left = RuleIdentity::from_canonical_bytes::<StatefulPointedRule>(3, Arc::from([1u8, 2, 3]));
    let right =
        RuleIdentity::from_canonical_bytes::<AlternateCanonicalRule>(5, Arc::from([4u8, 5]));
    let forward =
        RuleIdentity::compose_with_codec::<FirstIdentityCodec>(left.clone(), right.clone());
    let reverse =
        RuleIdentity::compose_with_codec::<FirstIdentityCodec>(right.clone(), left.clone());
    let alternate_codec =
        RuleIdentity::compose_with_codec::<SecondIdentityCodec>(left.clone(), right.clone());
    let repeat = RuleIdentity::compose_with_codec::<FirstIdentityCodec>(left, right);

    assert_ne!(forward, reverse);
    assert_ne!(forward, alternate_codec);
    assert_eq!(forward, repeat);
    assert_eq!(
        forward.charged_retained_bytes(),
        repeat.charged_retained_bytes()
    );
    assert!(forward.charged_retained_bytes() > 5);
}
