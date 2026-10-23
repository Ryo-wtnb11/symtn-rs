use super::*;

fn materialized_multiplicity_free_group_oracle<R>(
    rule: &R,
    space: &FusionProductSpace,
) -> Vec<CoupledFusionTrees>
where
    R: MultiplicityFreeFusionRule,
{
    let mut grouped = Vec::<CoupledFusionTrees>::new();
    let mut index: FxHashMap<SectorId, usize> = FxHashMap::default();
    for tuple in materialized_leg_tuple_oracle(space) {
        let uncoupled = tuple.iter().map(|leg| leg.sector()).collect::<Vec<_>>();
        let is_dual = tuple.iter().map(|leg| leg.is_dual()).collect::<Vec<_>>();
        let effective = uncoupled.clone();
        for coupled in reachable_coupled_sectors(rule, &effective) {
            let trees =
                collect_fusion_trees_for_coupled(rule, &uncoupled, &is_dual, &effective, coupled);
            match index.get(&coupled) {
                Some(&i) => grouped[i].trees.extend(trees),
                None => {
                    index.insert(coupled, grouped.len());
                    grouped.push(CoupledFusionTrees { coupled, trees });
                }
            }
        }
    }
    grouped.sort_by_key(|group| group.coupled);
    grouped
}

fn materialized_multiplicity_free_key_oracle<R>(
    rule: &R,
    codomain: &FusionProductSpace,
    domain: &FusionProductSpace,
) -> Vec<FusionTreePairKey>
where
    R: MultiplicityFreeFusionRule,
{
    let codomain = materialized_multiplicity_free_group_oracle(rule, codomain);
    let domain = materialized_multiplicity_free_group_oracle(rule, domain);
    merge_generic_tree_groups(&codomain, &domain)
}

#[test]
fn fusion_tree_keys_match_materialized_cartesian_oracle_across_ranks_and_duals() {
    // What: public key enumeration keeps the Cartesian tuple order without
    // reusing the production visitor in the test oracle.
    let rule = Z4PointedRule;
    let leg = |sectors: &[usize], is_dual| {
        SectorLeg::new(
            sectors
                .iter()
                .copied()
                .map(|sector| (SectorId::new(sector), 1usize)),
            is_dual,
        )
    };

    let cases = [
        (
            "empty",
            FusionProductSpace::new([]),
            FusionProductSpace::new([]),
        ),
        (
            "rank1",
            FusionProductSpace::new([leg(&[1, 3], true)]),
            FusionProductSpace::new([leg(&[1, 3], false)]),
        ),
        (
            "rank2",
            FusionProductSpace::new([leg(&[0, 1], false), leg(&[2, 3], true)]),
            FusionProductSpace::new([leg(&[1, 2, 3], true)]),
        ),
        (
            "rank8",
            FusionProductSpace::new(
                (0..8).map(|axis| leg(&[axis % 4, (axis + 1) % 4], axis % 2 == 1)),
            ),
            FusionProductSpace::new([leg(&[0, 2], true)]),
        ),
    ];

    for (name, codomain, domain) in cases {
        let expected = materialized_multiplicity_free_key_oracle(&rule, &codomain, &domain);
        let hom = FusionTreeHomSpace::new(codomain, domain);
        assert_eq!(
            hom.fusion_tree_keys(&rule).as_ref(),
            expected.as_slice(),
            "{name}"
        );
    }
}

#[test]
fn fusion_tree_homspace_generates_canonical_coupled_sector_order() {
    let rule = BranchingMultiplicityFreeRule;
    let hom = FusionTreeHomSpace::from_sector_ids([(1, 1), (1, 1)], [(1, 1), (1, 1)]);

    let keys = hom.fusion_tree_keys(&rule);

    assert_eq!(keys.len(), 2);
    assert_eq!(keys[0].coupled(), SectorId::new(0));
    assert_eq!(keys[1].coupled(), SectorId::new(2));
    assert_eq!(
        keys[0].codomain_uncoupled(),
        &[SectorId::new(1), SectorId::new(1)]
    );
    assert_eq!(
        keys[0].domain_uncoupled(),
        &[SectorId::new(1), SectorId::new(1)]
    );
    assert!(keys[0].codomain_innerlines().is_empty());
    assert!(keys[0].domain_innerlines().is_empty());
    assert_eq!(keys[0].codomain_vertices(), &[MultiplicityIndex::ONE]);
    assert_eq!(keys[0].domain_vertices(), &[MultiplicityIndex::ONE]);

    let sector = hom.sector_structure(&rule).unwrap();
    let groups = sector.fusion_tree_groups();
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].block_indices(), &[0, 1]);
    assert_eq!(
        groups[0].group_key(),
        &FusionTreeGroupKey::from_sector_ids([1, 1], [1, 1], [false, false], [false, false])
    );
}

#[test]
fn uncached_coupled_layout_probe_does_not_publish_identity_or_cache_state() {
    // What: the cold cost probe computes layout size/equality without
    // interning BlockStructure content or publishing a fusion-layout entry.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = Z2FusionRule;
    let leg = || {
        SectorLeg::new(
            [
                (Z2Irrep::EVEN.sector_id(), 2),
                (Z2Irrep::ODD.sector_id(), 3),
            ],
            false,
        )
    };
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let source = homspace
        .coupled_subblock_structure_from_leg_degeneracies(&rule)
        .unwrap();

    let recording = crate::probe::record();
    let (required_len, source_matches) = homspace
        .coupled_subblock_layout_probe_uncached(&rule, source.as_ref())
        .unwrap();

    assert_eq!(required_len, source.required_len().unwrap());
    assert!(source_matches);
    assert_eq!(
        (
            recording.get(crate::probe::Event::LayoutBuild),
            recording.get(crate::probe::Event::LayoutAdmission)
        ),
        (0, 0)
    );
    let cached_again = homspace
        .coupled_subblock_structure_from_leg_degeneracies(&rule)
        .unwrap();
    assert!(std::sync::Arc::ptr_eq(&source, &cached_again));
}

#[test]
fn fusion_tree_homspace_compose_matches_nonselfdual_domain_convention() {
    // TensorKit: `A * B` needs `domain(A) == codomain(B)` as spaces, so
    // the stored legs pair verbatim even for non-self-dual sectors
    // (Julia check: `rand(U1Space(0=>1,1=>1) ← same) * itself` works).
    let rule = U1FusionRule;
    let physical = u1(1);
    let lhs = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(u1(2), 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(physical, 1)], false)]),
    );
    let rhs = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(physical, 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(u1(3), 1)], false)]),
    );

    let composed = FusionTreeHomSpace::compose(&rule, &lhs, &rhs).unwrap();

    assert_eq!(composed.codomain().legs()[0].sectors(), &[u1(2)]);
    assert_eq!(composed.domain().legs()[0].sectors(), &[u1(3)]);
}

#[test]
fn fusion_tree_homspace_select_dualizes_axes_like_tensorkit() {
    let rule = U1FusionRule;
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(u1(1), 1)], false),
            SectorLeg::new([(u1(2), 1)], true),
        ]),
        FusionProductSpace::new([
            SectorLeg::new([(u1(3), 1)], false),
            SectorLeg::new([(u1(-5), 1)], true),
        ]),
    );

    let selected = hom.select(&rule, &[2, 0], &[1, 3]).unwrap();

    assert_eq!(selected.codomain().legs()[0].sectors(), &[u1(-3)]);
    assert!(selected.codomain().legs()[0].is_dual());
    assert_eq!(selected.codomain().legs()[1].sectors(), &[u1(1)]);
    assert!(!selected.codomain().legs()[1].is_dual());
    assert_eq!(selected.domain().legs()[0].sectors(), &[u1(-2)]);
    assert!(!selected.domain().legs()[0].is_dual());
    assert_eq!(selected.domain().legs()[1].sectors(), &[u1(-5)]);
    assert!(selected.domain().legs()[1].is_dual());
}

#[test]
fn fusion_tree_homspace_permute_requires_full_axis_permutation() {
    let rule = U1FusionRule;
    let hom = FusionTreeHomSpace::from_sectors([(u1(0), 1), (u1(1), 1)], [(u1(2), 1)]);

    let err = hom.permute(&rule, &[0], &[2]).unwrap_err();
    assert_eq!(
        err,
        CoreError::InvalidPermutation {
            permutation: vec![0, 2],
            rank: 3,
        }
    );

    let err = hom.permute(&rule, &[0, 0], &[2]).unwrap_err();
    assert_eq!(
        err,
        CoreError::InvalidPermutation {
            permutation: vec![0, 0, 2],
            rank: 3,
        }
    );
}

#[test]
fn fusion_tree_homspace_tensorcontract_preserves_canonical_compose() {
    let rule = U1FusionRule;
    let lhs = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(u1(2), 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(u1(1), 1)], false)]),
    );
    let rhs = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(u1(1), 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(u1(3), 1)], false)]),
    );

    let expected = FusionTreeHomSpace::compose(&rule, &lhs, &rhs).unwrap();
    let actual =
        FusionTreeHomSpace::tensorcontract_homspace(&rule, &lhs, &rhs, &[1], &[0], &[0, 1], 1)
            .unwrap();

    assert_eq!(actual, expected);
}

#[test]
fn fusion_tree_homspace_tensorcontract_matches_tensorkit_structural_formula() {
    let rule = U1FusionRule;
    let lhs = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(u1(1), 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(u1(5), 1)], false)]),
    );
    let rhs = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(u1(7), 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(u1(1), 1)], false)]),
    );

    let lhs_permuted = lhs.permute(&rule, &[1], &[0]).unwrap();
    let rhs_permuted = rhs.permute(&rule, &[1], &[0]).unwrap();
    let expected = FusionTreeHomSpace::compose(&rule, &lhs_permuted, &rhs_permuted)
        .unwrap()
        .permute(&rule, &[0], &[1])
        .unwrap();
    let actual =
        FusionTreeHomSpace::tensorcontract_homspace(&rule, &lhs, &rhs, &[0], &[1], &[0, 1], 1)
            .unwrap();

    assert_eq!(actual, expected);
    assert_eq!(actual.codomain().legs()[0].sectors(), &[u1(-5)]);
    assert!(actual.codomain().legs()[0].is_dual());
    assert_eq!(actual.domain().legs()[0].sectors(), &[u1(-7)]);
    assert!(actual.domain().legs()[0].is_dual());
}

#[test]
fn fusion_tree_homspace_tensorcontract_accepts_output_permutation_structurally() {
    let rule = U1FusionRule;
    let lhs = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(u1(2), 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(u1(1), 1)], false)]),
    );
    let rhs = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(u1(1), 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(u1(3), 1)], false)]),
    );

    let composed =
        FusionTreeHomSpace::tensorcontract_homspace(&rule, &lhs, &rhs, &[1], &[0], &[1, 0], 1)
            .unwrap();
    assert_eq!(composed.codomain().len(), 1);
    assert_eq!(composed.domain().len(), 1);
    assert_eq!(composed.codomain().legs()[0].sectors(), &[u1(-3)]);
    assert!(composed.codomain().legs()[0].is_dual());
    assert_eq!(composed.domain().legs()[0].sectors(), &[u1(-2)]);
    assert!(composed.domain().legs()[0].is_dual());
}

#[test]
fn direct_homspace_derivation_matches_reference_for_supported_rules() {
    let mixed_leg =
        |sectors: &[(SectorId, usize)], dual| SectorLeg::new(sectors.iter().copied(), dual);

    let u1_sectors = [(u1(-2), 1), (u1(1), 2)];
    let u1_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([mixed_leg(&u1_sectors, false), mixed_leg(&u1_sectors, true)]),
        FusionProductSpace::new([mixed_leg(&u1_sectors, true), mixed_leg(&u1_sectors, false)]),
    );
    assert_direct_contract_matches_reference(
        &U1FusionRule,
        &u1_hom,
        &u1_hom,
        &[3, 2],
        &[0, 1],
        &[2, 0, 3, 1],
        2,
    );

    let parity = [(SectorId::new(0), 2), (SectorId::new(1), 1)];
    let fz2_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([mixed_leg(&parity, false), mixed_leg(&parity, true)]),
        FusionProductSpace::new([mixed_leg(&parity, true), mixed_leg(&parity, false)]),
    );
    assert_direct_contract_matches_reference(
        &FermionParityFusionRule,
        &fz2_hom,
        &fz2_hom,
        &[3, 2],
        &[0, 1],
        &[1, 3, 0, 2],
        2,
    );

    let su2 = [
        (SU2Irrep::from_twice_spin(0).sector_id(), 2),
        (SU2Irrep::from_twice_spin(1).sector_id(), 1),
    ];
    let su2_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([mixed_leg(&su2, false), mixed_leg(&su2, true)]),
        FusionProductSpace::new([mixed_leg(&su2, true), mixed_leg(&su2, false)]),
    );
    assert_direct_contract_matches_reference(
        &SU2FusionRule,
        &su2_hom,
        &su2_hom,
        &[3, 2],
        &[0, 1],
        &[3, 1, 2, 0],
        2,
    );

    type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
    type ProductRule = ProductFusionRule<FpU1Rule, SU2FusionRule>;
    let product_rule = ProductRule::try_new(
        FpU1Rule::try_new(FermionParityFusionRule, U1FusionRule).unwrap(),
        SU2FusionRule,
    )
    .unwrap();
    let encode = |parity, charge, spin| {
        let inner =
            TensorKitProductCodec::encode(SectorId::new(parity), U1Irrep::new(charge).sector_id());
        TensorKitProductCodec::encode(inner, SU2Irrep::from_twice_spin(spin).sector_id())
    };
    let product = [(encode(0, 0, 0), 2), (encode(1, 1, 1), 1)];
    let product_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([mixed_leg(&product, false), mixed_leg(&product, true)]),
        FusionProductSpace::new([mixed_leg(&product, true), mixed_leg(&product, false)]),
    );
    assert_direct_contract_matches_reference(
        &product_rule,
        &product_hom,
        &product_hom,
        &[3, 2],
        &[0, 1],
        &[2, 1, 3, 0],
        2,
    );
}

#[derive(Clone)]
pub(super) struct DualCountingRule<R> {
    inner: R,
    dual_calls: Arc<AtomicUsize>,
    reset_on_dual: Arc<std::sync::atomic::AtomicBool>,
}

impl<R> DualCountingRule<R>
where
    R: FusionRule,
{
    pub(super) fn new(inner: R) -> Self {
        Self {
            inner,
            dual_calls: Arc::new(AtomicUsize::new(0)),
            reset_on_dual: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    fn reset_dual_calls(&self) {
        self.dual_calls.store(0, Ordering::Relaxed);
    }

    pub(super) fn dual_calls(&self) -> usize {
        self.dual_calls.load(Ordering::Relaxed)
    }
}

impl<R> symtn_sectors::testing::Provider for DualCountingRule<R> {}

impl<R> FusionRule for DualCountingRule<R>
where
    R: FusionRule,
{
    fn rule_identity(&self) -> RuleIdentity {
        self.inner.rule_identity()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        self.inner.fusion_style()
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        self.inner.braiding_style()
    }

    fn vacuum(&self) -> SectorId {
        self.inner.vacuum()
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        self.dual_calls.fetch_add(1, Ordering::Relaxed);
        self.inner.dual(sector)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        self.inner.fusion_channels(left, right)
    }
}

impl<R> CheckedFusionAlgebra for DualCountingRule<R>
where
    R: CheckedFusionAlgebra,
{
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        if self.reset_on_dual.swap(false, Ordering::SeqCst) {
            clear_structure_caches();
        }
        self.dual_calls.fetch_add(1, Ordering::Relaxed);
        self.inner.try_dual_sector(sector)
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        self.inner.try_fusion_channels(left, right)
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        self.inner.try_nsymbol(left, right, coupled)
    }
}

/// U(1) with a deliberately broken dual: charges in `collide` all map
/// to the dual of the first of them (not injective), and `fail` has no
/// representable dual. Everything else is U(1).
#[derive(Clone)]
struct FaultyDualRule {
    collide: Vec<SectorId>,
    fail: Option<SectorId>,
}

impl FaultyDualRule {
    fn mapped(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        if Some(sector) == self.fail {
            return Err(FusionAlgebraError::InvalidSector { sector });
        }
        let sector = if self.collide.contains(&sector) {
            self.collide[0]
        } else {
            sector
        };
        U1FusionRule.try_dual_sector(sector)
    }
}

impl symtn_sectors::testing::Provider for FaultyDualRule {}

impl FusionRule for FaultyDualRule {
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
        U1FusionRule.vacuum()
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        self.mapped(sector).unwrap()
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        U1FusionRule.fusion_channels(left, right)
    }
}

impl CheckedFusionAlgebra for FaultyDualRule {
    fn try_dual_sector(&self, sector: SectorId) -> Result<SectorId, FusionAlgebraError> {
        self.mapped(sector)
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, FusionAlgebraError> {
        U1FusionRule.try_fusion_channels(left, right)
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, FusionAlgebraError> {
        U1FusionRule.try_nsymbol(left, right, coupled)
    }
}

/// `A(v ← w)` contracted with `B(w ← x)` over `w`, output `(x ← v)`:
/// both open legs change side, so the matcher sees two dualized views.
fn crossing_contraction(v: SectorLeg, x: SectorLeg) -> (FusionTreeHomSpace, FusionTreeHomSpace) {
    let w = || SectorLeg::new([(u1(0), 2)], false);
    (
        FusionTreeHomSpace::new(FusionProductSpace::new([v]), FusionProductSpace::new([w()])),
        FusionTreeHomSpace::new(FusionProductSpace::new([w()]), FusionProductSpace::new([x])),
    )
}

/// The matcher's answer and the build path's answer for `expected`, both
/// unchecked and checked, plus whether the matcher materialized.
fn match_against_build<R: CheckedFusionAlgebra>(
    rule: &R,
    lhs: &FusionTreeHomSpace,
    rhs: &FusionTreeHomSpace,
    expected: &FusionTreeHomSpace,
) -> (
    Result<bool, CheckedFusionSpaceError>,
    Result<bool, CheckedFusionSpaceError>,
    usize,
) {
    let args = (&[1usize][..], &[0usize][..], &[1usize, 0][..], 1usize);
    let built = FusionTreeHomSpace::try_tensorcontract_homspace_checked(
        rule, lhs, rhs, args.0, args.1, args.2, args.3,
    )
    .map(|built| built == *expected);
    let recording = crate::probe::record();
    let matched = FusionTreeHomSpace::try_tensorcontract_homspace_matches_checked(
        rule, lhs, rhs, args.0, args.1, args.2, args.3, expected,
    );
    let materialized = recording.get(crate::probe::Event::DescriptorMaterialization);
    (matched, built, materialized)
}

#[test]
fn homspace_matcher_falls_back_to_the_build_answer_on_every_unproven_leg() {
    let rule = U1FusionRule;
    let leg = |charges: std::ops::Range<i32>, dual| {
        SectorLeg::new(
            charges.map(|charge| (u1(charge), (charge.unsigned_abs() % 3) as usize + 1)),
            dual,
        )
    };
    let build = |lhs: &FusionTreeHomSpace, rhs: &FusionTreeHomSpace| {
        FusionTreeHomSpace::tensorcontract_homspace(&rule, lhs, rhs, &[1], &[0], &[1, 0], 1)
            .unwrap()
    };
    let unchecked =
        |lhs: &FusionTreeHomSpace, rhs: &FusionTreeHomSpace, expected: &FusionTreeHomSpace| {
            FusionTreeHomSpace::tensorcontract_homspace_matches(
                &rule,
                lhs,
                rhs,
                &[1],
                &[0],
                &[1, 0],
                1,
                expected,
            )
            .unwrap()
        };

    // What: 64 sectors on a dualized leg are proven without building;
    // 65 exceed the injectivity mask and fall back, with the same answer.
    for (sectors, materializations) in [(64, 0), (65, 1)] {
        let (lhs, rhs) = crossing_contraction(leg(0..3, false), leg(-32..sectors - 32, false));
        let expected = build(&lhs, &rhs);
        let (matched, built, materialized) = match_against_build(&rule, &lhs, &rhs, &expected);
        assert_eq!((matched, built), (Ok(true), Ok(true)), "{sectors} sectors");
        assert_eq!(materialized, materializations, "{sectors} sectors");
        assert!(unchecked(&lhs, &rhs, &expected));
    }

    // What: a degeneracy mismatch and a sector mismatch on a dualized
    // leg are rejected, after the same fallback build.
    let (lhs, rhs) = crossing_contraction(leg(0..3, false), leg(-2..3, false));
    let expected = build(&lhs, &rhs);
    let replaced = |edit: fn(&mut [(SectorId, usize)])| {
        let x = &expected.codomain().legs()[0];
        FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new(
                {
                    let mut pairs = x.iter().collect::<Vec<_>>();
                    edit(&mut pairs);
                    pairs
                },
                x.is_dual(),
            )]),
            expected.domain().clone(),
        )
    };
    let degeneracy = replaced(|pairs| pairs[1].1 += 1);
    let sector = replaced(|pairs| pairs[4].0 = u1(7));
    for wrong in [&degeneracy, &sector] {
        let (matched, built, materialized) = match_against_build(&rule, &lhs, &rhs, wrong);
        assert_eq!((matched, built), (Ok(false), Ok(false)));
        assert_eq!(materialized, 1);
        assert!(!unchecked(&lhs, &rhs, wrong));
    }

    // What: a dual that is not injective on one leg is never taken as
    // proof: the checked matcher reports the build's DualNotInjective,
    // and the unchecked one panics exactly where the build panics.
    let collide = FaultyDualRule {
        collide: vec![u1(1), u1(2)],
        fail: None,
    };
    let (lhs, rhs) = crossing_contraction(leg(0..1, false), leg(0..4, false));
    let candidate = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(u1(0), 1), (u1(-1), 2), (u1(-3), 1)], true)]),
        FusionProductSpace::new([SectorLeg::new([(u1(0), 1)], true)]),
    );
    let (matched, built, _) = match_against_build(&collide, &lhs, &rhs, &candidate);
    assert!(matches!(
        built,
        Err(CheckedFusionSpaceError::FusionAlgebra(ref error))
            if matches!(**error, FusionAlgebraError::DualNotInjective { .. })
    ));
    assert_eq!(matched, built);
    let panics =
        |run: &dyn Fn()| std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)).is_err();
    assert!(panics(&|| {
        let _ = FusionTreeHomSpace::tensorcontract_homspace(
            &collide,
            &lhs,
            &rhs,
            &[1],
            &[0],
            &[1, 0],
            1,
        );
    }));
    assert!(panics(&|| {
        let _ = FusionTreeHomSpace::tensorcontract_homspace_matches(
            &collide,
            &lhs,
            &rhs,
            &[1],
            &[0],
            &[1, 0],
            1,
            &candidate,
        );
    }));

    // What: a dual error is the build path's first error, whether the
    // matcher meets it before or after an unproven leg.
    for fail in [u1(0), u1(1), u1(3)] {
        let failing = FaultyDualRule {
            collide: Vec::new(),
            fail: Some(fail),
        };
        let (lhs, rhs) = crossing_contraction(leg(0..2, false), leg(0..4, false));
        let proven = build(&lhs, &rhs);
        for expected in [&proven, &degeneracy] {
            let (matched, built, _) = match_against_build(&failing, &lhs, &rhs, expected);
            assert!(built.is_err(), "{fail:?}");
            assert_eq!(matched, built, "{fail:?}");
        }
    }
}

fn assert_oriented_validation_dual_calls_are_linear<R>(
    rule: &DualCountingRule<R>,
    sectors: impl IntoIterator<Item = (SectorId, usize)>,
    sector_count: usize,
) where
    R: FusionRule,
{
    let leg = SectorLeg::new(sectors, false);
    let lhs = OrientedLegView::borrowed(&leg).toggled();
    let rhs = OrientedLegView::borrowed(&leg).toggled();
    rule.reset_dual_calls();
    validate_oriented_composed_leg(rule, lhs, rhs, (0, 0)).unwrap();
    assert_eq!(rule.dual_calls(), 2 * sector_count);
}

#[test]
fn oriented_composed_leg_membership_is_linear_in_dual_operations() {
    const SECTORS: usize = 256;
    let u1_rule = DualCountingRule::new(U1FusionRule);
    assert_oriented_validation_dual_calls_are_linear(
        &u1_rule,
        (0..SECTORS).map(|index| (u1(index as i32 - 97), index % 5 + 1)),
        SECTORS,
    );

    let product_rule = DualCountingRule::new(
        ProductFusionRule::<U1FusionRule, FermionParityFusionRule, TensorKitProductCodec>::try_new(
            U1FusionRule,
            FermionParityFusionRule,
        )
        .unwrap(),
    );
    assert_oriented_validation_dual_calls_are_linear(
        &product_rule,
        (0..SECTORS).map(|index| {
            (
                TensorKitProductCodec::encode(u1(index as i32 - 113), SectorId::new(index % 2)),
                index % 7 + 1,
            )
        }),
        SECTORS,
    );
}

#[test]
fn oriented_composed_leg_invalid_path_matches_materialized_error_order() {
    let rule = U1FusionRule;
    let lhs = SectorLeg::new([(u1(-3), 1), (u1(2), 2)], false);
    let rhs = SectorLeg::new([(u1(-3), 4), (u1(2), 2)], false);
    let lhs_view = OrientedLegView::borrowed(&lhs).toggled();
    let rhs_view = OrientedLegView::borrowed(&rhs).toggled();
    let expected = validate_composed_leg(
        &lhs_view.materialize(&rule),
        &rhs_view.materialize(&rule),
        (0, 0),
    );
    assert_eq!(
        validate_oriented_composed_leg(&rule, lhs_view, rhs_view, (0, 0)),
        expected
    );
}

#[test]
fn generic_select_matches_reference() {
    let rule = UnitaryToyOmRule;
    let leg = SectorLeg::new(
        [
            (SectorId::new(UnitaryToyOmRule::VACUUM), 2),
            (SectorId::new(UnitaryToyOmRule::A), 1),
        ],
        false,
    );
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone(), leg.clone()]),
        FusionProductSpace::new([leg.clone(), leg]),
    );
    let expected = reference_select(&rule, &hom, &[3, 0], &[2, 1]);
    let actual = hom.select(&rule, &[3, 0], &[2, 1]).unwrap();
    assert_eq!(actual, expected);
}

impl<R: MultiplicityFreeFusionRule> MultiplicityFreeFusionRule for DualCountingRule<R> {}

#[test]
fn checked_mf_complete_epoch_precedes_provider_and_homspace_derivation() {
    if test_support::run_isolated_or_return(
        "SYMTN_MF_PRODUCER_RESET", "tests::fusion_space::homspace_derivation::checked_mf_complete_epoch_precedes_provider_and_homspace_derivation",
    ) { return; }
    let rule = DualCountingRule::new(U1FusionRule);
    let hom = FusionTreeHomSpace::from_sectors([(u1(1), 2)], [(u1(1), 3)]);
    for derive in [false, true] {
        clear_structure_caches();
        rule.reset_on_dual.store(true, Ordering::SeqCst);
        let (hom, prepared) = if derive {
            FusionTreeHomSpace::prepare_fusion_tree_layout_checked_with(&rule, || {
                rule.try_dual_sector(u1(1))?;
                Ok::<_, FusionAlgebraError>(hom.clone())
            })
            .unwrap()
        } else {
            (
                hom.clone(),
                hom.prepare_fusion_tree_layout_checked(&rule).unwrap(),
            )
        };
        assert!(!rule.reset_on_dual.load(Ordering::SeqCst));
        let (_, stale) = prepared
            .build_complete_homspace_from_leg_degeneracies(hom.clone())
            .unwrap();
        assert_eq!(
            structure_cache_info(StructureCacheKind::DegeneracyStructure).entries(),
            0
        );
        let fresh = hom.prepare_fusion_tree_layout_checked(&rule).unwrap();
        let (_, fresh) = fresh
            .build_complete_homspace_from_leg_degeneracies(hom)
            .unwrap();
        assert_eq!(*stale, *fresh);
        assert_ne!(stale.content_id(), fresh.content_id());
        assert_eq!(
            structure_cache_info(StructureCacheKind::DegeneracyStructure).entries(),
            1
        );
    }
}
