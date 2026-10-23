use super::*;

#[test]
fn coupled_sector_dimensions_cover_empty_and_multiplicity_free_products() {
    // What: product-space dimensions use the tensor unit for rank zero,
    // annihilate on an empty leg, and reproduce U1/SU2 fusion dimensions.
    let empty = FusionProductSpace::new(std::iter::empty::<SectorLeg>());
    assert_eq!(
        empty
            .coupled_sector_block_dimensions(&U1FusionRule)
            .unwrap(),
        BTreeMap::from([(u1(0), 1)])
    );

    let empty_leg = FusionProductSpace::new([SectorLeg::new(
        std::iter::empty::<(SectorId, usize)>(),
        false,
    )]);
    assert!(empty_leg
        .coupled_sector_block_dimensions(&U1FusionRule)
        .unwrap()
        .is_empty());

    let u1_leg = SectorLeg::new([(u1(0), 1), (u1(1), 1)], false);
    let u1_product = FusionProductSpace::new([u1_leg.clone(), u1_leg]);
    assert_eq!(
        u1_product
            .coupled_sector_block_dimensions(&U1FusionRule)
            .unwrap(),
        BTreeMap::from([(u1(0), 1), (u1(1), 2), (u1(2), 1)])
    );

    let half = SectorLeg::new([(su2(1), 1)], false);
    let su2_product = FusionProductSpace::new([half.clone(), half]);
    assert_eq!(
        su2_product
            .coupled_sector_block_dimensions(&SU2FusionRule)
            .unwrap(),
        BTreeMap::from([(su2(0), 1), (su2(2), 1)])
    );
}

#[test]
fn coupled_sector_dimensions_keep_outward_labels_for_dual_legs() {
    // What: the pivotal dual flag does not dualize an already-outward U1
    // sector label a second time.
    let product = FusionProductSpace::new([SectorLeg::new([(u1(-3), 2)], true)]);
    assert_eq!(
        product
            .coupled_sector_block_dimensions(&U1FusionRule)
            .unwrap(),
        BTreeMap::from([(u1(-3), 2)])
    );
}

#[derive(Clone, Copy, Debug)]
pub(super) struct BranchingMultiplicityFreeRule;

impl symtn_sectors::testing::Provider for BranchingMultiplicityFreeRule {}

impl FusionRule for BranchingMultiplicityFreeRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
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

    fn dual(&self, sector: SectorId) -> SectorId {
        match sector.id() {
            3 => SectorId::new(1),
            other => SectorId::new(other),
        }
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        match (left.id(), right.id()) {
            (0, x) | (x, 0) => smallvec![SectorId::new(x)],
            (1, 1) => smallvec![SectorId::new(0), SectorId::new(2)],
            (1, 2) | (2, 1) => smallvec![SectorId::new(1), SectorId::new(3)],
            (2, 2) => smallvec![SectorId::new(0)],
            _ => SmallVec::new(),
        }
    }
}

impl MultiplicityFreeFusionRule for BranchingMultiplicityFreeRule {}

#[derive(Clone, Copy, Debug)]
pub(super) struct UnsortedFusionIteratorOrderRule;

impl symtn_sectors::testing::Provider for UnsortedFusionIteratorOrderRule {}

impl FusionRule for UnsortedFusionIteratorOrderRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
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

    fn dual(&self, sector: SectorId) -> SectorId {
        sector
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        match (left.id(), right.id()) {
            (0, x) | (x, 0) => smallvec![SectorId::new(x)],
            (1, 1) => smallvec![SectorId::new(2), SectorId::new(0)],
            (1, 2) | (2, 1) => smallvec![SectorId::new(1)],
            (2, 2) => smallvec![SectorId::new(0)],
            _ => SmallVec::new(),
        }
    }
}

impl MultiplicityFreeFusionRule for UnsortedFusionIteratorOrderRule {}

pub(super) fn fusion_tree_pair_order(
    keys: &[FusionTreePairKey],
) -> Vec<(Vec<usize>, Vec<usize>, usize)> {
    keys.iter()
        .map(|key| {
            (
                sector_ids(key.codomain_uncoupled()),
                sector_ids(key.domain_uncoupled()),
                key.coupled().id(),
            )
        })
        .collect()
}

#[test]
fn canonical_coupled_layouts_pass_checked_subset_proof() {
    // What: canonical built-in coupled layouts pass the finite-algebra
    // subset proof.
    fn assert_rule<R>(rule: &R, sector: SectorId)
    where
        R: MultiplicityFreeFusionRule + CheckedFusionAlgebra,
    {
        let homspace = FusionTreeHomSpace::from_sectors([(sector, 1)], [(sector, 1)]);
        let structure = homspace
            .coupled_subblock_structure(rule, 1, [vec![1, 1]])
            .unwrap();
        homspace
            .validate_subblock_structure_subset_checked(rule, &structure)
            .unwrap();
    }

    assert_rule(&Z2FusionRule, z2_even());
    assert_rule(&FermionParityFusionRule, z2_odd());
    assert_rule(&U1FusionRule, u1(7));
    assert_rule(&SU2FusionRule, su2(3));
    assert_rule(&TableRule::fibonacci(), SectorId::new(1));

    type Fz2U1 = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
    type Triple = ProductFusionRule<Fz2U1, SU2FusionRule>;
    let pair = Fz2U1::try_new(FermionParityFusionRule, U1FusionRule).unwrap();
    let pair_sector = pair.encode_component_ids(z2_odd(), u1(2));
    let triple = Triple::try_new(pair, SU2FusionRule).unwrap();
    let triple_sector = triple.encode_component_ids(pair_sector, su2(1));
    assert_rule(&triple, triple_sector);
}

#[test]
fn tree_pair_block_apis_reject_mixed_fusion_tree_groups_before_symbols() {
    let base = tree_pair_group_fixture(&[1, 2], &[3], 3, &[false, false], &[false]);
    let mixed = [
        tree_pair_group_fixture(&[1, 4], &[5], 5, &[false, false], &[false]),
        tree_pair_group_fixture(&[1, 2], &[3], 3, &[false, true], &[false]),
        tree_pair_group_fixture(&[1, 2], &[3], 3, &[false, false], &[true]),
    ];

    for other in mixed {
        let keys = [base.clone(), other];
        let snapshot = keys.clone();
        assert_mixed_tree_pair_block_group_is_rejected(
            &IdentitySymbolPanicRule,
            &keys,
            CoreError::MalformedFusionTree {
                message: TREE_PAIR_BLOCK_GROUP_ERROR,
            },
        );
        // What: validation errors do not alter caller-owned source keys.
        assert_eq!(keys, snapshot);
    }
}

#[test]
fn simple_prepared_braid_has_runtime_rank_schedule() {
    // What: a nontrivial SimpleFusion braid keeps the complete high-rank
    // Artin schedule through compilation without a rank-specialized plan.
    let rank = 19;
    let permutation = (0..rank).rev().collect::<Vec<_>>();
    let levels = (0..rank).collect::<Vec<_>>();
    let prepared = PreparedTreePairOperation::prepare_braid(
        &SU2FusionRule,
        rank,
        0,
        &permutation,
        &[],
        &levels,
        &[],
    )
    .unwrap();

    assert!(matches!(prepared.plan, PreparedTreePairPlan::Braid(_)));
    let prepared = prepared.into_compiler_owned_simple_braid();
    assert!(matches!(
        prepared.plan,
        PreparedTreePairPlan::SimpleBraid(_)
    ));
    assert_eq!(
        prepared.plan.artin_steps().unwrap().count(),
        rank * (rank - 1) / 2
    );
}

#[test]
fn borrowed_unique_artin_lowering_matches_materialized_replay() {
    // What: operation-local inverse positions reproduce the
    // materialized Artin word and inverse flag for every small-rank
    // permutation, not only a selected reverse ordering.
    for rank in 0..=8 {
        let mut permutation = (0..rank).collect::<Vec<_>>();
        let levels = (0..rank).map(|index| (index * 7) % 11).collect::<Vec<_>>();
        loop {
            let mut raw_axis_positions = vec![usize::MAX; rank];
            for (position, &axis) in permutation.iter().enumerate() {
                raw_axis_positions[axis] = position;
            }
            let expected = PreparedTreeBraid::new(&permutation, &levels, rank)
                .unwrap()
                .artin_steps
                .into_vec();
            let prepared = PreparedTreePairOperation::prepare_braid_with_raw_axis_positions(
                &Z2FusionRule,
                rank,
                0,
                &permutation,
                &[],
                &levels,
                &[],
                &raw_axis_positions,
            )
            .unwrap();
            assert_eq!(
                prepared
                    .plan
                    .artin_steps()
                    .map(|steps| steps.collect::<Vec<_>>())
                    .unwrap_or_default(),
                expected,
                "rank-{rank} permutation {permutation:?} changed"
            );
            if !next_lexicographic_permutation(&mut permutation) {
                break;
            }
        }
    }
}

fn next_lexicographic_permutation(values: &mut [usize]) -> bool {
    let Some(pivot) = (0..values.len().saturating_sub(1))
        .rev()
        .find(|&index| values[index] < values[index + 1])
    else {
        return false;
    };
    let successor = (pivot + 1..values.len())
        .rev()
        .find(|&index| values[pivot] < values[index])
        .expect("increasing pivot has a successor");
    values.swap(pivot, successor);
    values[pivot + 1..].reverse();
    true
}

struct PermissiveOneBitLayout;

impl PackedSectorLayout for PermissiveOneBitLayout {
    const BITS: u32 = 1;

    fn validate(_sector: SectorId) -> Result<(), ProductSectorCodecError> {
        Ok(())
    }
}

struct ZeroBitLayout;

impl PackedSectorLayout for ZeroBitLayout {
    const BITS: u32 = 0;

    fn validate(sector: SectorId) -> Result<(), ProductSectorCodecError> {
        (sector.id() == 0)
            .then_some(())
            .ok_or(ProductSectorCodecError::InvalidHighBits {
                sector,
                total_bits: 0,
            })
    }
}

struct FullWidthLayout;

impl PackedSectorLayout for FullWidthLayout {
    const BITS: u32 = usize::BITS;

    fn validate(_sector: SectorId) -> Result<(), ProductSectorCodecError> {
        Ok(())
    }
}

#[test]
fn packed_product_codec_reports_invalid_components_and_widths() {
    // What: malformed packed IDs and layouts wider than usize fail with a
    // typed reason instead of panicking, wrapping, or silently truncating.
    type FpU1Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
    let invalid_parity = FpU1Codec::encode_checked(SectorId::new(2), u1(0));
    assert!(matches!(
        invalid_parity,
        Err(ProductSectorCodecError::ComponentOutOfRange {
            component: ProductSectorComponent::Left,
            ..
        })
    ));

    type MaliciousCodec = PackedProductCodec<PermissiveOneBitLayout, Fz2SectorLayout>;
    assert!(matches!(
        MaliciousCodec::encode_checked(SectorId::new(2), SectorId::new(0)),
        Err(ProductSectorCodecError::ComponentOutOfRange {
            component: ProductSectorComponent::Left,
            ..
        })
    ));

    type TripleLayout =
        ProductSectorLayout<ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>, Su2SectorLayout>;
    assert_eq!(TripleLayout::BITS, 41);
    #[cfg(target_pointer_width = "64")]
    {
        let invalid_high_bits = SectorId::new(1usize << TripleLayout::BITS);
        assert!(matches!(
            TripleLayout::validate(invalid_high_bits),
            Err(ProductSectorCodecError::InvalidHighBits { .. })
        ));
    }

    type TooWide =
        PackedProductCodec<ProductSectorLayout<U1SectorLayout, U1SectorLayout>, U1SectorLayout>;
    assert!(matches!(
        TooWide::encode_checked(SectorId::new(0), SectorId::new(0)),
        Err(ProductSectorCodecError::WidthOverflow { .. })
    ));

    type FullThenZero = PackedProductCodec<FullWidthLayout, ZeroBitLayout>;
    let full = FullThenZero::encode_checked(SectorId::new(usize::MAX), SectorId::new(0)).unwrap();
    assert_eq!(full, SectorId::new(usize::MAX));
    assert_eq!(
        FullThenZero::decode_checked(full).unwrap(),
        (SectorId::new(usize::MAX), SectorId::new(0))
    );

    type ZeroThenParity = PackedProductCodec<ZeroBitLayout, Fz2SectorLayout>;
    assert_eq!(
        ZeroThenParity::encode_checked(SectorId::new(0), SectorId::new(1)).unwrap(),
        SectorId::new(1)
    );
}

#[test]
fn packed_and_tensorkit_codecs_remain_distinct_compatible_options() {
    // What: the expert Cantor codec keeps its Cantor-pairing IDs while the
    // packed codec has a distinct rule identity and round-trips the same
    // semantic components.
    type PackedCodec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
    type PackedRule = ProductFusionRule<FermionParityFusionRule, U1FusionRule, PackedCodec>;
    type CantorRule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;

    let parity = z2_odd();
    let charge = u1(2);
    let packed = PackedCodec::encode(parity, charge);
    let cantor = TensorKitProductCodec::encode(parity, charge);
    assert_ne!(packed, cantor);
    assert_eq!(PackedCodec::decode(packed), Some((parity, charge)));
    assert_eq!(
        TensorKitProductCodec::decode(cantor),
        Some((parity, charge))
    );
    assert_ne!(
        PackedRule::try_new(FermionParityFusionRule, U1FusionRule)
            .unwrap()
            .rule_identity(),
        CantorRule::try_new(FermionParityFusionRule, U1FusionRule)
            .unwrap()
            .rule_identity()
    );
}

#[test]
fn product_external_domain_sector_is_dualized_componentwise() {
    type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
    let rule = FpU1Rule::try_new(FermionParityFusionRule, U1FusionRule).unwrap();
    let a = rule.encode_component_ids(z2_odd(), u1(2));
    let external_domain = rule.dual(a);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(a, 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(a, 1)], false)]),
    );

    let keys = hom
        .fusion_tree_keys_from_external_sectors(&rule, &[a, external_domain])
        .unwrap();
    assert_eq!(keys.len(), 1);
    assert_eq!(keys[0].codomain_uncoupled(), &[a]);
    assert_eq!(keys[0].domain_uncoupled(), &[a]);
    assert_eq!(keys[0].coupled(), a);

    let err = hom
        .fusion_tree_keys_from_external_sectors(&rule, &[a, a])
        .unwrap_err();
    assert_eq!(
        err,
        CoreError::InvalidSector {
            sector: external_domain
        }
    );
}
