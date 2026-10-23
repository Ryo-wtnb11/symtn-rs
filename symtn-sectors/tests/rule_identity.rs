use std::hash::{Hash, Hasher};
use std::sync::Arc;
use symtn_sectors::{
    BraidingStyleKind, FusionRule, FusionStyleKind, MultiplicityFreeFusionRule, RuleIdentity,
    SectorId, SectorVec, Z2FusionRule,
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

impl symtn_sectors::testing::Provider for StatefulPointedRule {}

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

#[test]
fn product_rule_identity_includes_stateful_child_identity() {
    let first = symtn_sectors::ProductFusionRule::<_, _>::try_new(
        symtn_sectors::ZNFusionRule::new(3).unwrap(),
        Z2FusionRule,
    )
    .unwrap();
    let second = symtn_sectors::ProductFusionRule::<_, _>::try_new(
        symtn_sectors::ZNFusionRule::new(4).unwrap(),
        Z2FusionRule,
    )
    .unwrap();

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
