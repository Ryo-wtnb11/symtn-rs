use super::*;
use std::cell::Cell;
use std::sync::Arc;

std::thread_local! {
    static DUAL_QUERIES: Cell<usize> = const { Cell::new(0) };
}

/// Provider `dual` queries made on this thread since the last call.
fn take_dual_queries() -> usize {
    DUAL_QUERIES.replace(0)
}

/// U(1) charges whose `dual` is `q -> -q` (`negate`) or the identity, under
/// a chosen identity, counting every provider `dual` query. `fail` makes
/// those sectors unrepresentable.
#[derive(Clone, Debug)]
struct CountingDualRule {
    identity: RuleIdentity,
    negate: bool,
    fail: &'static [i32],
}

impl CountingDualRule {
    fn new(identity: RuleIdentity, negate: bool) -> Self {
        Self {
            identity,
            negate,
            fail: &[],
        }
    }

    fn query(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        DUAL_QUERIES.set(DUAL_QUERIES.get() + 1);
        if self.fail.iter().any(|&charge| u1(charge) == sector) {
            return Err(FusionAlgebraError::InvalidSector { sector });
        }
        Ok(if self.negate {
            U1FusionRule.dual(sector)
        } else {
            sector
        })
    }
}

impl symtn_sectors::testing::Provider for CountingDualRule {}

impl FusionRule for CountingDualRule {
    fn rule_identity(&self) -> RuleIdentity {
        self.identity.clone()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }
    fn vacuum(&self) -> SectorId {
        U1FusionRule.vacuum()
    }
    fn dual(&self, sector: SectorId) -> SectorId {
        self.query(sector).expect("representable sector")
    }
    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        U1FusionRule.fusion_channels(left, right)
    }
}

impl CheckedFusionAlgebra for CountingDualRule {
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        self.query(sector)
    }
    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        Ok(U1FusionRule.fusion_channels(left, right))
    }
    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        Ok(U1FusionRule.nsymbol(left, right, coupled))
    }
}

impl CheckedGenericFusion for CountingDualRule {
    type Error = FusionAlgebraError;
    fn rule_identity(&self) -> RuleIdentity {
        self.identity.clone()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }
    fn vacuum(&self) -> SectorId {
        U1FusionRule.vacuum()
    }
    fn try_dual(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        self.query(sector)
    }
    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        Ok(U1FusionRule.fusion_channels(left, right))
    }
    fn try_fusion_channels_in_table(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        Ok(U1FusionRule.fusion_channels(left, right))
    }
    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        Ok(U1FusionRule.nsymbol(left, right, coupled))
    }
}

fn negating() -> CountingDualRule {
    CountingDualRule::new(RuleIdentity::of_type::<CountingDualRule>(), true)
}

fn leg_hash(leg: &SectorLeg) -> u64 {
    let mut state = std::collections::hash_map::DefaultHasher::new();
    leg.hash(&mut state);
    state.finish()
}

fn homspace_hash(space: &FusionTreeHomSpace) -> u64 {
    let mut state = std::collections::hash_map::DefaultHasher::new();
    space.hash(&mut state);
    state.finish()
}

/// A leg the negating dual moves, and its hand-computed dual.
fn moved_leg() -> (SectorLeg, SectorLeg) {
    (
        SectorLeg::new([(u1(-2), 1), (u1(0), 3), (u1(1), 5)], false),
        SectorLeg::new([(u1(2), 1), (u1(0), 3), (u1(-1), 5)], true),
    )
}

#[test]
fn a_dual_under_the_filling_identity_queries_no_sector() {
    // What: the first dual query fills the shared map with one query per
    // sector; every later dual of that storage under the same identity,
    // through each entry point and from either orientation, asks the
    // provider nothing and returns the hand-computed dual.
    let rule = negating();
    let (leg, expected) = moved_leg();
    take_dual_queries();
    assert_eq!(leg.dual(&rule), expected);
    assert_eq!(take_dual_queries(), 3);

    let dual = leg.dual(&rule);
    assert_eq!(dual, expected);
    assert_eq!(leg_hash(&dual), leg_hash(&expected));
    assert!(dual.shares_sector_data_with(&leg));
    assert_eq!(leg.try_dual(&rule), Ok(expected.clone()));
    assert_eq!(leg.try_dual_generic(&rule).unwrap(), expected);
    assert_eq!(dual.dual(&rule), leg);
    assert_eq!(dual.try_dual(&rule), Ok(leg.clone()));
    assert_eq!(take_dual_queries(), 0);

    // An unmoved map: the dual is the same map with the flag flipped.
    let symmetric = SectorLeg::new([(u1(-1), 2), (u1(0), 3), (u1(1), 2)], true);
    let flipped = SectorLeg::new([(u1(-1), 2), (u1(0), 3), (u1(1), 2)], false);
    assert_eq!(symmetric.dual(&rule), flipped);
    assert_eq!(take_dual_queries(), 3);
    assert_eq!(symmetric.dual(&rule), flipped);
    assert_eq!(symmetric.try_dual_generic(&rule).unwrap(), flipped);
    assert_eq!(take_dual_queries(), 0);
}

#[test]
fn a_warm_homspace_derivation_queries_no_dual() {
    // What: the operation-count contract of a derived HomSpace. Permuting
    // `(a b <- c d)` to `(b c <- d a)` dualizes `c` and `a`; once their
    // maps are filled, select, permute and contraction outputs ask the
    // provider for no dual, and the result is the hand-built space.
    let rule = negating();
    let a = SectorLeg::new([(u1(-1), 2), (u1(2), 1)], false);
    let b = SectorLeg::new([(u1(0), 2), (u1(1), 3)], true);
    let c = SectorLeg::new([(u1(-2), 4), (u1(1), 1)], false);
    let d = SectorLeg::new([(u1(0), 1), (u1(3), 2)], false);
    let source = FusionTreeHomSpace::new(
        FusionProductSpace::new([a.clone(), b.clone()]),
        FusionProductSpace::new([c.clone(), d.clone()]),
    );
    let expected = FusionTreeHomSpace::new(
        FusionProductSpace::new([b, SectorLeg::new([(u1(2), 4), (u1(-1), 1)], true)]),
        FusionProductSpace::new([d, SectorLeg::new([(u1(1), 2), (u1(-2), 1)], true)]),
    );
    take_dual_queries();
    assert_eq!(source.permute(&rule, &[1, 2], &[3, 0]).unwrap(), expected);
    assert_eq!(take_dual_queries(), 4);

    assert_eq!(source.permute(&rule, &[1, 2], &[3, 0]).unwrap(), expected);
    assert_eq!(
        source.try_permute_checked(&rule, &[1, 2], &[3, 0]).unwrap(),
        expected
    );
    assert_eq!(
        source
            .try_permute_generic_checked(&rule, &[1, 2], &[3, 0])
            .unwrap(),
        expected
    );
    assert_eq!(take_dual_queries(), 0);
}

#[test]
fn a_different_identity_of_the_same_type_requeries_and_keeps_its_own_storage() {
    // What: two Generic providers of one Rust type with different
    // identities dualize differently. The second is never trusted with the
    // first's map: it queries every sector on every call, gets its own
    // dual, and leaves the map serving the first unchanged.
    let first = CountingDualRule::new(RuleIdentity::new_unique::<CountingDualRule>(), true);
    let second = CountingDualRule::new(RuleIdentity::new_unique::<CountingDualRule>(), false);
    let (leg, first_dual) = moved_leg();
    let second_dual = SectorLeg::new([(u1(-2), 1), (u1(0), 3), (u1(1), 5)], true);
    assert_eq!(leg.try_dual_generic(&first).unwrap(), first_dual);
    take_dual_queries();

    for _ in 0..2 {
        assert_eq!(leg.try_dual_generic(&second).unwrap(), second_dual);
        assert_eq!(take_dual_queries(), 3);
    }
    assert_eq!(leg.dual(&second), second_dual);
    assert_eq!(take_dual_queries(), 3);

    let again = leg.try_dual_generic(&first).unwrap();
    assert_eq!(again, first_dual);
    assert!(again.shares_sector_data_with(&leg));
    assert_eq!(take_dual_queries(), 0);
}

#[test]
fn a_different_identity_reports_its_first_failing_sector() {
    // What: an identity that did not fill the map takes the per-sector
    // path, so its error is the first failing sector in sector order
    // (zigzag ids order the leg as 0, 1, -2; both 1 and -2 fail), for
    // the checked and the checked Generic entry points alike.
    let (leg, _) = moved_leg();
    leg.dual(&negating());
    let failing = CountingDualRule {
        identity: RuleIdentity::new_unique::<CountingDualRule>(),
        negate: true,
        fail: &[-2, 1],
    };
    let expected = FusionAlgebraError::InvalidSector { sector: u1(1) };
    take_dual_queries();
    assert_eq!(leg.try_dual(&failing), Err(expected.clone()));
    assert_eq!(take_dual_queries(), 2);
    match leg.try_dual_generic(&failing) {
        Err(CheckedGenericStructureError::Provider(error)) => assert_eq!(error, expected),
        other => panic!("expected the provider error for u1(1), got {other:?}"),
    }
    assert_eq!(take_dual_queries(), 2);
}

#[test]
fn trusted_and_validated_derivations_are_the_same_homspace() {
    // What: a derivation through filled maps and one that fills them yield
    // equal spaces with equal hashes and ids, and both reach the one
    // complete-structure cache entry. A structure-cache reset does not
    // touch leg-owned dual maps.
    let _guard = crate::tests::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = U1FusionRule;
    let source = || {
        FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(u1(-1), 2), (u1(1), 3)], false),
                SectorLeg::new([(u1(0), 2), (u1(2), 1)], false),
            ]),
            FusionProductSpace::new([
                SectorLeg::new([(u1(-1), 1), (u1(1), 2)], false),
                SectorLeg::new([(u1(0), 1), (u1(-2), 3)], false),
            ]),
        )
    };
    let expected = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(u1(0), 2), (u1(2), 1)], false),
            SectorLeg::new([(u1(1), 1), (u1(-1), 2)], true),
        ]),
        FusionProductSpace::new([
            SectorLeg::new([(u1(0), 1), (u1(-2), 3)], false),
            SectorLeg::new([(u1(1), 2), (u1(-1), 3)], true),
        ]),
    );
    let warm = source();
    let validated = warm.permute(&rule, &[1, 2], &[3, 0]).unwrap();
    let trusted = warm.permute(&rule, &[1, 2], &[3, 0]).unwrap();
    let fresh = source().permute(&rule, &[1, 2], &[3, 0]).unwrap();
    for space in [&validated, &trusted, &fresh] {
        assert_eq!(space, &expected);
        assert_eq!(homspace_hash(space), homspace_hash(&expected));
        assert_eq!(space.id(), expected.id());
    }

    let structure = validated
        .coupled_subblock_structure_from_leg_degeneracies(&rule)
        .unwrap();
    assert!(Arc::ptr_eq(
        &structure,
        &trusted
            .coupled_subblock_structure_from_leg_degeneracies(&rule)
            .unwrap()
    ));

    let counting = negating();
    let (leg, dual) = moved_leg();
    leg.dual(&counting);
    crate::clear_structure_caches();
    take_dual_queries();
    assert_eq!(leg.dual(&counting), dual);
    assert_eq!(take_dual_queries(), 0);
}

#[test]
fn filling_the_dual_map_under_any_identity_kind_keeps_the_charge() {
    // What: the charge a cache admitted a leg with covers the dual map and
    // the inline identity of whichever rule fills it; heap behind the
    // identity is the rule's own and shared, not newly retained.
    let content =
        || RuleIdentity::from_canonical_bytes::<CountingDualRule>(7, Arc::from(&[1u8, 2, 3][..]));
    let identities = [
        RuleIdentity::of_type::<CountingDualRule>(),
        RuleIdentity::new_unique::<CountingDualRule>(),
        content(),
        RuleIdentity::compose_with_codec::<CountingDualRule>(
            RuleIdentity::of_type::<U1FusionRule>(),
            content(),
        ),
    ];
    for identity in identities {
        let rule = CountingDualRule::new(identity, true);
        let leg = SectorLeg::new(
            (0..12).map(|charge| (u1(charge), 1 + charge as usize)),
            false,
        );
        let charge = leg.charged_retained_bytes();
        let dual = leg.dual(&rule);
        assert_eq!(leg.charged_retained_bytes(), charge);
        assert_eq!(dual.charged_retained_bytes(), charge);
        assert_eq!(
            dual,
            SectorLeg::new(
                (0..12).map(|charge| (u1(-charge), 1 + charge as usize)),
                true
            )
        );
    }
}

#[cfg(feature = "cgc-gen")]
#[test]
fn filling_the_dual_map_under_a_value_identity_keeps_the_charge() {
    // What: the value-identity kind of the charge contract above, through a
    // real Lie rule; SU(2) irreps are self-dual.
    let rule = symtn_sectors::testing::su_lie_rule(2);
    let leg = SectorLeg::new([(SectorId::new(0), 2), (SectorId::new(1), 3)], false);
    let charge = leg.charged_retained_bytes();
    let dual = leg.try_dual_generic(&rule).unwrap();
    assert_eq!(leg.charged_retained_bytes(), charge);
    assert_eq!(dual.charged_retained_bytes(), charge);
    assert_eq!(
        dual,
        SectorLeg::new([(SectorId::new(0), 2), (SectorId::new(1), 3)], true)
    );
}
