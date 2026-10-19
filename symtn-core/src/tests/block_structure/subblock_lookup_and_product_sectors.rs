use super::*;

#[test]
fn block_structure_finds_fusion_tree_subblock_by_key() {
    let first =
        FusionTreePairKey::try_pair_from_sector_ids([1], [1], 1, [false], [true], [], [], [], [])
            .unwrap();
    let second =
        FusionTreePairKey::try_pair_from_sector_ids([0], [0], 0, [false], [true], [], [], [], [])
            .unwrap();
    let structure = packed_fixture_structure(
        2,
        [
            (BlockKey::from(second.clone()), vec![1, 4]),
            (BlockKey::from(first.clone()), vec![2, 3]),
        ],
    )
    .unwrap();

    let first_block = structure.fusion_tree_pair_block(&first).unwrap();
    let second_block = structure
        .block_by_key(&BlockKey::from(second.clone()))
        .unwrap();

    assert_eq!(first_block.key(), &BlockKey::from(first));
    assert_eq!(first_block.shape(), &[2, 3]);
    assert_eq!(first_block.offset(), 4);
    assert_eq!(second_block.key(), &BlockKey::from(second));
    assert_eq!(second_block.shape(), &[1, 4]);
    assert_eq!(second_block.offset(), 0);
}

#[test]
fn fusion_tree_pair_block_reports_missing_key() {
    let existing =
        FusionTreePairKey::try_pair_from_sector_ids([0], [0], 0, [false], [true], [], [], [], [])
            .unwrap();
    let missing =
        FusionTreePairKey::try_pair_from_sector_ids([1], [1], 1, [false], [true], [], [], [], [])
            .unwrap();
    let structure = packed_fixture_structure(2, [(BlockKey::from(existing), vec![1, 1])]).unwrap();

    let err = structure.fusion_tree_pair_block(&missing).unwrap_err();

    assert_eq!(
        err,
        CoreError::MissingBlockKey {
            key: Box::new(BlockKey::from(missing)),
        }
    );
}

#[test]
fn public_u1_irrep_roundtrips_compact_ids_and_fuses() {
    let rule = U1FusionRule;
    let charges = [
        U1Irrep::new(-2),
        U1Irrep::new(-1),
        U1Irrep::new(0),
        U1Irrep::new(1),
        U1Irrep::new(2),
    ];
    let ids = charges.map(SectorId::from);

    assert_eq!(
        ids,
        [
            SectorId::new(3),
            SectorId::new(1),
            SectorId::new(0),
            SectorId::new(2),
            SectorId::new(4),
        ]
    );
    for charge in charges {
        assert_eq!(U1Irrep::from_sector_id(charge.sector_id()), Some(charge));
    }
    assert_eq!(rule.vacuum(), U1Irrep::new(0).sector_id());
    assert_eq!(
        rule.dual(U1Irrep::new(3).sector_id()),
        U1Irrep::new(-3).sector_id()
    );
    assert_eq!(
        rule.fusion_channels(U1Irrep::new(-2).sector_id(), U1Irrep::new(5).sector_id())
            .to_vec(),
        vec![U1Irrep::new(3).sector_id()]
    );
}

#[cfg(target_pointer_width = "64")]
#[test]
fn packed_product_codec_covers_the_builtin_leaf_domains() {
    // What: the packed codec preserves the full 32-bit U1 component range,
    // including the raw ID reserved by the semantic label constructor,
    // together with every currently supported SU2 label.
    type FpU1Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
    type FpU1Layout = ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>;
    type TripleCodec = PackedProductCodec<FpU1Layout, Su2SectorLayout>;

    for charge in [excluded_u1_id(), u1(-1), u1(0), u1(1), u1(i32::MAX)] {
        for twice_spin in [0, 1, 127, 254] {
            let inner = FpU1Codec::encode(z2_odd(), charge);
            let encoded = TripleCodec::encode(inner, su2(twice_spin));
            let (decoded_inner, decoded_spin) = TripleCodec::decode(encoded).unwrap();
            let (decoded_parity, decoded_charge) = FpU1Codec::decode(decoded_inner).unwrap();
            assert_eq!(decoded_parity, z2_odd());
            assert_eq!(decoded_charge, charge);
            assert_eq!(decoded_spin, su2(twice_spin));
        }
    }
}

#[test]
fn product_sector_api_exposes_only_generic_composition() {
    let pair = product_sector(z2_odd(), u1(2));
    let encoded = pair.sector_id_with::<TensorKitProductCodec>();
    assert_eq!(encoded, TensorKitProductCodec::encode(z2_odd(), u1(2)));
    assert_eq!(pair.left(), &z2_odd());
    assert_eq!(pair.right(), &u1(2));

    let left_rule = product_fusion_rule(FermionParityFusionRule, U1FusionRule);
    let chained_rule = FermionParityFusionRule
        .product(U1FusionRule)
        .product(SU2FusionRule);
    let left_sector = |parity, charge| left_rule.encode_component_ids(parity, u1(charge));
    let chained_sector = |parity, charge, twice_spin| {
        chained_rule.encode_component_ids(left_sector(parity, charge), su2(twice_spin))
    };

    let a = chained_sector(z2_odd(), 1, 1);
    let b = chained_sector(z2_odd(), -1, 1);
    let c0 = chained_sector(z2_even(), 0, 0);
    let c2 = chained_sector(z2_even(), 0, 2);

    assert_eq!(chained_rule.fusion_style(), FusionStyleKind::Simple);
    assert_eq!(chained_rule.braiding_style(), BraidingStyleKind::Fermionic);
    assert_eq!(chained_rule.fusion_channels(a, b).to_vec(), vec![c0, c2]);
}

#[test]
fn product_fusion_rule_combines_fermion_parity_and_u1_componentwise() {
    type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
    let rule = FpU1Rule::default();
    let sector = |parity, charge| rule.encode_component_ids(parity, u1(charge));
    let odd_two = sector(z2_odd(), 2);
    let odd_minus_five = sector(z2_odd(), -5);
    let even_minus_three = sector(z2_even(), -3);

    assert_eq!(rule.fusion_style(), FusionStyleKind::Unique);
    assert_eq!(rule.braiding_style(), BraidingStyleKind::Fermionic);
    assert_eq!(rule.vacuum(), sector(z2_even(), 0));
    assert_eq!(rule.dual(odd_two), sector(z2_odd(), -2));
    assert_eq!(
        rule.fusion_channels(odd_two, odd_minus_five).to_vec(),
        vec![even_minus_three]
    );
    assert_eq!(rule.nsymbol(odd_two, odd_minus_five, even_minus_three), 1);
    assert_eq!(
        rule.r_symbol_scalar(odd_two, odd_minus_five, even_minus_three),
        -1.0
    );
    assert_eq!(rule.sqrt_dim_scalar(odd_two), 1.0);
}

#[test]
fn product_fusion_rule_nested_fz2_u1_su2_channels_and_symbols_match_tensorkit() {
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
    let c2 = sector(z2_even(), 0, 2);

    assert_eq!(rule.fusion_style(), FusionStyleKind::Simple);
    assert_eq!(rule.braiding_style(), BraidingStyleKind::Fermionic);
    assert_eq!(rule.dual(a), sector(z2_odd(), -1, 1));
    assert_eq!(rule.fusion_channels(a, b).to_vec(), vec![c0, c2]);
    assert_eq!(rule.r_symbol_scalar(a, b, c0), 1.0);
    assert_eq!(rule.r_symbol_scalar(a, b, c2), -1.0);
    assert!((rule.sqrt_dim_scalar(c2) - 3.0_f64.sqrt()).abs() < 1.0e-12);

    let vacuum_left = left_sector(z2_even(), 0);
    let spin_half = rule.encode_component_ids(vacuum_left, su2(1));
    let spin_zero = rule.encode_component_ids(vacuum_left, su2(0));
    assert!(
        (rule.f_symbol_scalar(spin_half, spin_half, spin_half, spin_half, spin_zero, spin_zero,)
            + 0.5)
            .abs()
            < 1.0e-12
    );
}

#[test]
fn product_fusion_tree_homspace_matches_tensorkit_fz2_u1_su2_fixture() {
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
    assert_eq!(a.id(), 43);
    assert_eq!(b.id(), 19);
    assert_eq!(c0.id(), 0);
    assert_eq!(c1.id(), 3);

    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(a, 1)], false),
            SectorLeg::new([(b, 1)], false),
        ]),
        FusionProductSpace::new([SectorLeg::new([(c0, 1), (c1, 1)], false)]),
    );
    let keys = hom.fusion_tree_keys(&rule);

    assert_eq!(keys.len(), 2);
    for (key, coupled) in keys.iter().zip([c0, c1]) {
        assert_eq!(key.coupled(), coupled);
        assert_eq!(key.codomain_uncoupled(), &[a, b]);
        assert_eq!(key.domain_uncoupled(), &[coupled]);
        assert_eq!(key.codomain_is_dual(), &[false, false]);
        assert_eq!(key.domain_is_dual(), &[false]);
        assert_eq!(key.codomain_innerlines(), &[]);
        assert_eq!(key.domain_innerlines(), &[]);
        assert_eq!(key.codomain_vertices(), &[MultiplicityIndex::ONE]);
        assert_eq!(key.domain_vertices(), &[]);
    }
}

#[test]
// Message updated for #971: the component's FusionRule::fusion_channels
// panic is now derived from CheckedFusionAlgebra::try_fusion_channels
// rather than restated separately; the invalid sector it names is
// unchanged.
#[should_panic(expected = "invalid fusion sector SectorId(2)")]
fn product_fusion_rule_panics_on_component_invalid_sector_like_existing_rules() {
    type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
    let rule = FpU1Rule::default();
    let invalid_left_component = rule.encode_component_ids(SectorId::new(2), u1(0));
    let valid = rule.encode_component_ids(z2_even(), u1(0));

    let _ = rule.fusion_channels(invalid_left_component, valid);
}

#[test]
fn public_su2_irrep_fusion_channels_match_doubled_spin_order() {
    let rule = SU2FusionRule;

    assert_eq!(
        rule.fusion_channels(
            SU2Irrep::from_twice_spin(1).sector_id(),
            SU2Irrep::from_twice_spin(2).sector_id(),
        )
        .to_vec(),
        vec![
            SU2Irrep::from_twice_spin(1).sector_id(),
            SU2Irrep::from_twice_spin(3).sector_id(),
        ]
    );
}

#[test]
fn multiplicity_free_su2_repartition_matches_tensorkit_bend_factor() {
    let rule = SU2FusionRule;
    let source =
        FusionTreePairKey::try_pair_from_sector_ids([1], [1], 1, [false], [false], [], [], [], [])
            .unwrap();

    let all_codomain = multiplicity_free_repartition_tree_pair(&rule, &source, 2).unwrap();
    assert_eq!(all_codomain.len(), 1);
    assert_eq!(
        all_codomain[0].0.codomain_uncoupled(),
        &[SectorId::new(1); 2]
    );
    assert_eq!(all_codomain[0].0.codomain_is_dual(), &[false, true]);
    assert_eq!(all_codomain[0].0.codomain_innerlines(), &[]);
    assert_eq!(
        all_codomain[0].0.codomain_vertices(),
        &[MultiplicityIndex::ONE]
    );
    assert_eq!(
        all_codomain[0].0.codomain_tree().coupled(),
        SectorId::new(0)
    );
    assert_eq!(all_codomain[0].0.domain_uncoupled(), &[]);
    assert_eq!(all_codomain[0].0.domain_tree().coupled(), SectorId::new(0));
    assert!((all_codomain[0].1 - 2.0_f64.sqrt()).abs() < 1.0e-12);

    let all_domain = multiplicity_free_repartition_tree_pair(&rule, &source, 0).unwrap();
    assert_eq!(all_domain.len(), 1);
    assert_eq!(all_domain[0].0.codomain_uncoupled(), &[]);
    assert_eq!(all_domain[0].0.codomain_tree().coupled(), SectorId::new(0));
    assert_eq!(all_domain[0].0.domain_uncoupled(), &[SectorId::new(1); 2]);
    assert_eq!(all_domain[0].0.domain_is_dual(), &[false, true]);
    assert!((all_domain[0].1 - 2.0_f64.sqrt()).abs() < 1.0e-12);
}

#[test]
fn multiplicity_free_su2_permute_tree_pair_matches_tensorkit_swap() {
    let rule = SU2FusionRule;
    let source =
        FusionTreePairKey::try_pair_from_sector_ids([1], [1], 1, [false], [false], [], [], [], [])
            .unwrap();

    let permuted = multiplicity_free_permute_tree_pair(&rule, &source, &[1], &[0]).unwrap();

    assert_eq!(permuted.len(), 1);
    assert_eq!(permuted[0].0.codomain_uncoupled(), &[SectorId::new(1)]);
    assert_eq!(permuted[0].0.domain_uncoupled(), &[SectorId::new(1)]);
    assert_eq!(permuted[0].0.codomain_is_dual(), &[true]);
    assert_eq!(permuted[0].0.domain_is_dual(), &[true]);
    assert_eq!(permuted[0].0.codomain_tree().coupled(), SectorId::new(1));
    assert_eq!(permuted[0].0.domain_tree().coupled(), SectorId::new(1));
    assert!((permuted[0].1 - 1.0).abs() < 1.0e-12);
}

#[test]
fn prepared_simple_pair_matches_explicit_generic_composition() {
    // What: a prepared Simple-fusion pair operation equals the explicit
    // repartition -> all-codomain Artin -> repartition composition.
    let rule = SU2FusionRule;
    let source =
        FusionTreePairKey::try_pair_from_sector_ids([1], [1], 1, [false], [false], [], [], [], [])
            .unwrap();
    let prepared = PreparedTreePairOperation::prepare_permute(&rule, 1, 1, &[1], &[0]).unwrap();
    let actual = prepared.execute_multiplicity_free(&rule, &source).unwrap();

    let all_codomain = multiplicity_free_repartition_tree_pair(&rule, &source, 2).unwrap();
    let braided = compose_terms(all_codomain, |key| {
        multiplicity_free_braid_tree(&rule, key.codomain_tree(), &[1, 0], &[0, 1]).map(|terms| {
            terms
                .into_iter()
                .map(|(tree, coefficient)| {
                    (
                        FusionTreePairKey::pair(tree, key.domain_tree().clone()),
                        coefficient,
                    )
                })
                .collect::<Vec<_>>()
        })
    })
    .unwrap();
    let expected = multiplicity_free_repartition_terms(&rule, braided, 1).unwrap();

    assert_eq!(
        actual.iter().map(|(key, _)| key).collect::<Vec<_>>(),
        expected.iter().map(|(key, _)| key).collect::<Vec<_>>()
    );
    for ((_, actual), (_, expected)) in actual.iter().zip(expected) {
        assert!((*actual - expected).abs() < 1.0e-12);
    }
}

pub(super) fn u1_nonselfdual_tree_pair_fixture() -> FusionTreePairKey {
    FusionTreePairKey::pair(
        FusionTreeKey::new(
            [u1(1), u1(2)],
            u1(3),
            [false, false],
            Vec::<SectorId>::new(),
            [MultiplicityIndex::ONE],
        ),
        FusionTreeKey::new(
            [u1(3)],
            u1(3),
            [false],
            Vec::<SectorId>::new(),
            Vec::<MultiplicityIndex>::new(),
        ),
    )
}

#[test]
fn u1_bendright_dualizes_visible_sector_and_flips_isdual_like_tensorkit() {
    let out =
        multiplicity_free_bendright_tree_pair(&U1FusionRule, &u1_nonselfdual_tree_pair_fixture())
            .unwrap();

    assert_eq!(out.len(), 1);
    assert_eq!(out[0].1, 1.0);
    assert_eq!(out[0].0.codomain_uncoupled(), &[u1(1)]);
    assert_eq!(out[0].0.codomain_tree().coupled(), u1(1));
    assert_eq!(out[0].0.codomain_is_dual(), &[false]);
    assert_eq!(out[0].0.codomain_innerlines(), &[]);
    assert_eq!(out[0].0.codomain_vertices(), &[]);
    assert_eq!(out[0].0.domain_uncoupled(), &[u1(3), u1(-2)]);
    assert_eq!(out[0].0.domain_tree().coupled(), u1(1));
    assert_eq!(out[0].0.domain_is_dual(), &[false, true]);
    assert_eq!(out[0].0.domain_innerlines(), &[]);
    assert_eq!(out[0].0.domain_vertices(), &[MultiplicityIndex::ONE]);
}

#[test]
fn u1_foldright_dualizes_first_visible_sector_and_flips_isdual_like_tensorkit() {
    let out =
        multiplicity_free_foldright_tree_pair(&U1FusionRule, &u1_nonselfdual_tree_pair_fixture())
            .unwrap();

    assert_eq!(out.len(), 1);
    assert_eq!(out[0].1, 1.0);
    assert_eq!(out[0].0.codomain_uncoupled(), &[u1(2)]);
    assert_eq!(out[0].0.codomain_tree().coupled(), u1(2));
    assert_eq!(out[0].0.codomain_is_dual(), &[false]);
    assert_eq!(out[0].0.codomain_innerlines(), &[]);
    assert_eq!(out[0].0.codomain_vertices(), &[]);
    assert_eq!(out[0].0.domain_uncoupled(), &[u1(-1), u1(3)]);
    assert_eq!(out[0].0.domain_tree().coupled(), u1(2));
    assert_eq!(out[0].0.domain_is_dual(), &[true, false]);
    assert_eq!(out[0].0.domain_innerlines(), &[]);
    assert_eq!(out[0].0.domain_vertices(), &[MultiplicityIndex::ONE]);
}

#[test]
fn u1_repartition_to_all_domain_matches_tensorkit_nonselfdual_fixture() {
    let out = multiplicity_free_repartition_tree_pair(
        &U1FusionRule,
        &u1_nonselfdual_tree_pair_fixture(),
        0,
    )
    .unwrap();

    assert_eq!(out.len(), 1);
    assert_eq!(out[0].1, 1.0);
    assert_eq!(out[0].0.codomain_uncoupled(), &[]);
    assert_eq!(out[0].0.codomain_tree().coupled(), u1(0));
    assert_eq!(out[0].0.codomain_is_dual(), &[] as &[bool]);
    assert_eq!(out[0].0.codomain_innerlines(), &[]);
    assert_eq!(out[0].0.codomain_vertices(), &[]);
    assert_eq!(out[0].0.domain_uncoupled(), &[u1(3), u1(-2), u1(-1)]);
    assert_eq!(out[0].0.domain_tree().coupled(), u1(0));
    assert_eq!(out[0].0.domain_is_dual(), &[false, true, true]);
    assert_eq!(out[0].0.domain_innerlines(), &[u1(1)]);
    assert_eq!(
        out[0].0.domain_vertices(),
        &[MultiplicityIndex::ONE, MultiplicityIndex::ONE]
    );
}

#[test]
fn u1_repartition_to_all_codomain_matches_tensorkit_nonselfdual_fixture() {
    let out = multiplicity_free_repartition_tree_pair(
        &U1FusionRule,
        &u1_nonselfdual_tree_pair_fixture(),
        3,
    )
    .unwrap();

    assert_eq!(out.len(), 1);
    assert_eq!(out[0].1, 1.0);
    assert_eq!(out[0].0.codomain_uncoupled(), &[u1(1), u1(2), u1(-3)]);
    assert_eq!(out[0].0.codomain_tree().coupled(), u1(0));
    assert_eq!(out[0].0.codomain_is_dual(), &[false, false, true]);
    assert_eq!(out[0].0.codomain_innerlines(), &[u1(3)]);
    assert_eq!(
        out[0].0.codomain_vertices(),
        &[MultiplicityIndex::ONE, MultiplicityIndex::ONE]
    );
    assert_eq!(out[0].0.domain_uncoupled(), &[]);
    assert_eq!(out[0].0.domain_tree().coupled(), u1(0));
    assert_eq!(out[0].0.domain_is_dual(), &[] as &[bool]);
    assert_eq!(out[0].0.domain_innerlines(), &[]);
    assert_eq!(out[0].0.domain_vertices(), &[]);
}

#[test]
fn u1_transpose_cyclic_23_1_matches_tensorkit_nonselfdual_fixture() {
    let out = multiplicity_free_transpose_tree_pair(
        &U1FusionRule,
        &u1_nonselfdual_tree_pair_fixture(),
        &[1, 2],
        &[0],
    )
    .unwrap();

    assert_eq!(out.len(), 1);
    assert_eq!(out[0].1, 1.0);
    assert_eq!(out[0].0.codomain_uncoupled(), &[u1(2), u1(-3)]);
    assert_eq!(out[0].0.codomain_tree().coupled(), u1(-1));
    assert_eq!(out[0].0.codomain_is_dual(), &[false, true]);
    assert_eq!(out[0].0.codomain_innerlines(), &[]);
    assert_eq!(out[0].0.codomain_vertices(), &[MultiplicityIndex::ONE]);
    assert_eq!(out[0].0.domain_uncoupled(), &[u1(-1)]);
    assert_eq!(out[0].0.domain_tree().coupled(), u1(-1));
    assert_eq!(out[0].0.domain_is_dual(), &[true]);
    assert_eq!(out[0].0.domain_innerlines(), &[]);
    assert_eq!(out[0].0.domain_vertices(), &[]);
}

#[test]
fn nested_product_elementary_bend_keeps_the_fermionic_phase() {
    // What: the elementary bend of an odd fZ2 pair retains the negative
    // product-category phase independently of the block/per-source runners.
    type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
    type ProductRule = ProductFusionRule<FpU1Rule, SU2FusionRule>;
    let left_rule = FpU1Rule::default();
    let rule = ProductRule::default();
    let coupled =
        rule.encode_component_ids(left_rule.encode_component_ids(z2_even(), u1(0)), su2(1));
    let odd_half =
        rule.encode_component_ids(left_rule.encode_component_ids(z2_odd(), u1(1)), su2(1));
    let odd_one =
        rule.encode_component_ids(left_rule.encode_component_ids(z2_odd(), u1(-1)), su2(2));
    let source = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(&rule, [coupled], coupled, [false], [], []).unwrap(),
        FusionTreeKey::try_new_for_rule(
            &rule,
            [odd_half, odd_one],
            coupled,
            [false, true],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
    );

    let bent = multiplicity_free_bendleft_tree_pair(&rule, &source).unwrap();
    assert_eq!(bent.len(), 1);
    assert!(bent[0].1 < 0.0);
    let restored = multiplicity_free_bendright_tree_pair(&rule, &bent[0].0).unwrap();
    assert_eq!(restored.len(), 1);
    assert_eq!(restored[0].0, source);
    assert!((bent[0].1 * restored[0].1 - 1.0).abs() < 1.0e-12);
}

#[test]
fn typed_sector_homspace_builds_u1_tree_key() {
    let rule = U1FusionRule;
    let hom = FusionTreeHomSpace::from_sectors([(U1Irrep::new(2), 1)], [(U1Irrep::new(2), 1)]);

    let key = hom
        .unique_fusion_tree_key_from_external_sectors(
            &rule,
            &[U1Irrep::new(2).sector_id(), U1Irrep::new(-2).sector_id()],
        )
        .unwrap();

    assert_eq!(key.codomain_uncoupled(), &[U1Irrep::new(2).sector_id()]);
    assert_eq!(key.domain_uncoupled(), &[U1Irrep::new(2).sector_id()]);
    assert_eq!(key.coupled(), U1Irrep::new(2).sector_id());
}

#[test]
fn fusion_tensor_space_builds_subblockstructure_from_homspace() {
    let rule = Z2FusionRule;
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new(
            [(SectorId::new(0), 1), (SectorId::new(1), 3)],
            false,
        )]),
        FusionProductSpace::new([SectorLeg::new(
            [(SectorId::new(0), 2), (SectorId::new(1), 1)],
            false,
        )]),
    );

    let structure = hom
        .coupled_subblock_structure(&rule, 1, [vec![1, 2], vec![3, 1]])
        .unwrap();

    assert_eq!(structure.block_count(), 2);
    assert_eq!(structure.required_len().unwrap(), 5);
    assert_eq!(
        structure.block(0).unwrap().key(),
        &BlockKey::from(
            FusionTreePairKey::try_pair_from_sector_ids(
                [0],
                [0],
                0,
                [false],
                [false],
                [],
                [],
                [],
                [],
            )
            .unwrap()
        )
    );
    assert_eq!(structure.block(1).unwrap().shape(), &[3, 1]);
}

#[test]
fn packed_block_structure_records_rank_offsets_and_required_len() {
    let structure = BlockStructure::packed_column_major(2, [vec![2, 3], vec![1, 4]]).unwrap();

    assert_eq!(structure.rank(), 2);
    assert_eq!(structure.block_count(), 2);
    assert_eq!(structure.sector_structure().block_count(), 2);
    assert_eq!(structure.degeneracy_structure().block_count(), 2);
    let first = structure.block(0).unwrap();
    assert_eq!(first.key(), &BlockKey::ordinal(0));
    assert_eq!(first.shape(), &[2, 3]);
    assert_eq!(first.strides(), &[1, 2]);
    assert_eq!(first.offset(), 0);
    let second = structure.block(1).unwrap();
    assert_eq!(second.key(), &BlockKey::ordinal(1));
    assert_eq!(second.shape(), &[1, 4]);
    assert_eq!(second.strides(), &[1, 1]);
    assert_eq!(second.offset(), 6);
    assert_eq!(structure.required_len().unwrap(), 10);
}

#[test]
fn block_structure_rejects_duplicate_keys() {
    let first = BlockSpec::column_major_with_key(BlockKey::opaque([7]), vec![2, 2], 0).unwrap();
    let second = BlockSpec::column_major_with_key(BlockKey::opaque([7]), vec![1, 3], 4).unwrap();

    let err = BlockStructure::from_blocks_with_rank(2, vec![first, second]).unwrap_err();

    assert_eq!(
        err,
        CoreError::DuplicateBlockKey {
            key: Box::new(BlockKey::opaque([7]))
        }
    );
}

#[test]
fn block_structure_validates_degeneracy_before_sector_keys() {
    // What: preparing an owned block structure reports degeneracy errors
    // first when both degeneracy metadata and sector keys are bad.
    let key = BlockKey::opaque([7]);
    let first = BlockSpec::column_major_with_key(key.clone(), vec![2, 2], 0).unwrap();
    let malformed = BlockSpec {
        key,
        shape: smallvec![1, 3],
        strides: smallvec![1],
        offset: 4,
    };

    assert_eq!(
        BlockStructure::from_blocks_with_rank(2, vec![first, malformed]),
        Err(CoreError::RankMismatch {
            shape: 2,
            strides: 1,
        })
    );
}

#[test]
fn fusion_tree_pair_key_records_tensorkit_subblock_pair_fields() {
    let key = FusionTreePairKey::try_pair_from_sector_ids(
        [2, 3],
        [5, 7],
        11,
        [false, true],
        [true, false],
        [13],
        [17],
        [19, 23],
        [29, 31],
    )
    .unwrap();

    assert_eq!(
        key.codomain_uncoupled(),
        &[SectorId::new(2), SectorId::new(3)]
    );
    assert_eq!(
        key.domain_uncoupled(),
        &[SectorId::new(5), SectorId::new(7)]
    );
    assert_eq!(key.coupled(), SectorId::new(11));
    assert_eq!(key.codomain_is_dual(), &[false, true]);
    assert_eq!(key.domain_is_dual(), &[true, false]);
    assert_eq!(key.codomain_innerlines(), &[SectorId::new(13)]);
    assert_eq!(key.domain_innerlines(), &[SectorId::new(17)]);
    assert_eq!(
        key.codomain_vertices(),
        &[
            MultiplicityIndex::new(19).unwrap(),
            MultiplicityIndex::new(23).unwrap(),
        ]
    );
    assert_eq!(
        key.domain_vertices(),
        &[
            MultiplicityIndex::new(29).unwrap(),
            MultiplicityIndex::new(31).unwrap(),
        ]
    );

    let group = key.group_key();
    assert_eq!(group.codomain_uncoupled(), key.codomain_uncoupled());
    assert_eq!(group.domain_uncoupled(), key.domain_uncoupled());
    assert_eq!(group.codomain_is_dual(), key.codomain_is_dual());
    assert_eq!(group.domain_is_dual(), key.domain_is_dual());
}

#[test]
fn sorted_lookup_distinguishes_rank1_tree_duality() {
    let nondual =
        FusionTreePairKey::try_pair_from_sector_ids([0], [], 0, [false], [], [], [], [], [])
            .unwrap();
    let dual = FusionTreePairKey::try_pair_from_sector_ids([0], [], 0, [true], [], [], [], [], [])
        .unwrap();
    let structure = SectorStructure::from_keys(1, [BlockKey::from(nondual)]).unwrap();

    assert!(!structure.has_compact_lookup());
    assert_eq!(structure.find_index(&BlockKey::from(dual.clone())), None);
    assert_eq!(structure.find_fusion_tree_pair_index(&dual), None);
}

#[test]
fn unique_homspace_builds_subblock_key_from_external_sectors() {
    let rule = Z2FusionRule;
    let hom = FusionTreeHomSpace::from_sector_ids([(1, 1)], [(1, 1)]);

    let key = hom
        .unique_fusion_tree_key_from_external_sectors(&rule, &[SectorId::new(1), SectorId::new(1)])
        .unwrap();

    assert_eq!(key.codomain_uncoupled(), &[SectorId::new(1)]);
    assert_eq!(key.domain_uncoupled(), &[SectorId::new(1)]);
    assert_eq!(key.coupled(), SectorId::new(1));
    assert_eq!(key.codomain_is_dual(), &[false]);
    assert_eq!(key.domain_is_dual(), &[false]);
}

#[test]
fn unique_homspace_dualizes_domain_external_sectors_like_tensorkit() {
    let rule = Z4PointedRule;
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(SectorId::new(1), 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(SectorId::new(1), 1)], false)]),
    );

    let key = hom
        .unique_fusion_tree_key_from_external_sectors(&rule, &[SectorId::new(1), SectorId::new(3)])
        .unwrap();

    assert_eq!(key.codomain_uncoupled(), &[SectorId::new(1)]);
    assert_eq!(key.domain_uncoupled(), &[SectorId::new(1)]);
    assert_eq!(key.coupled(), SectorId::new(1));
}

#[test]
fn fusion_tree_pair_key_external_sectors_restore_visible_domain_sector() {
    let rule = Z4PointedRule;
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(SectorId::new(1), 1)], true)]),
        FusionProductSpace::new([SectorLeg::new([(SectorId::new(1), 1)], false)]),
    );
    let key = hom
        .unique_fusion_tree_key_from_external_sectors(&rule, &[SectorId::new(1), SectorId::new(3)])
        .unwrap();

    assert_eq!(key.codomain_uncoupled(), &[SectorId::new(1)]);
    assert_eq!(key.domain_uncoupled(), &[SectorId::new(1)]);
    assert_eq!(
        key.external_sectors(&rule),
        vec![SectorId::new(1), SectorId::new(3)]
    );
    assert_eq!(key.external_is_dual(), vec![true, false]);
}

// A unique, non-self-dual rule with complex symbols, so every factor of the
// bend and fold coefficients (dims, B, F-derived A and multi-F-moves, κ, and
// the left moves' conj) is visible.
#[derive(Clone, Copy, Debug)]
struct ComplexZ4BendRule;

impl FusionRule for ComplexZ4BendRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Anyonic
    }
    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }
    fn dual(&self, s: SectorId) -> SectorId {
        SectorId::new((4 - s.id() % 4) % 4)
    }
    fn fusion_channels(&self, l: SectorId, r: SectorId) -> SectorVec {
        smallvec::smallvec![SectorId::new((l.id() + r.id()) % 4)]
    }
}

impl MultiplicityFreeFusionRule for ComplexZ4BendRule {}

impl MultiplicityFreeFusionSymbols for ComplexZ4BendRule {
    type Scalar = Complex64;
    fn f_symbol_scalar(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Complex64 {
        let phase = a.id() + 2 * b.id() + 3 * c.id() + 5 * d.id() + 7 * e.id() + 11 * f.id();
        Complex64::from_polar(1.0, 0.1 * phase as f64)
    }
    fn r_symbol_scalar(&self, _: SectorId, _: SectorId, _: SectorId) -> Complex64 {
        Complex64::new(1.0, 0.0)
    }
}

impl MultiplicityFreeRigidSymbols for ComplexZ4BendRule {
    fn dim_scalar(&self, _: SectorId) -> Complex64 {
        Complex64::new(1.0, 0.0)
    }
    fn inv_dim_scalar(&self, _: SectorId) -> Complex64 {
        Complex64::new(1.0, 0.0)
    }
    fn sqrt_dim_scalar(&self, s: SectorId) -> Complex64 {
        Complex64::new(1.0 + s.id() as f64, 0.0)
    }
    fn inv_sqrt_dim_scalar(&self, s: SectorId) -> Complex64 {
        Complex64::new(1.0 / (1.0 + s.id() as f64), 0.0)
    }
    fn twist_scalar(&self, _: SectorId) -> Complex64 {
        Complex64::new(1.0, 0.0)
    }
    fn frobenius_schur_phase_scalar(&self, s: SectorId) -> Complex64 {
        Complex64::from_polar(1.0, 0.7 * s.id() as f64)
    }
    fn b_symbol_scalar(&self, a: SectorId, b: SectorId, c: SectorId) -> Complex64 {
        Complex64::from_polar(2.0, 0.2 * (a.id() + 4 * b.id() + 9 * c.id()) as f64)
    }
}

fn z4(id: usize) -> SectorId {
    SectorId::new(id)
}

#[test]
fn complex_non_self_dual_bends_match_hand_computed_tensorkit_coefficients() {
    let rule = ComplexZ4BendRule;
    let assert_close = |got: Complex64, want: Complex64| {
        assert!((got - want).norm() < 1e-12, "{got} != {want}");
    };

    // What: bendright of the dual leg b = 1 off (2 ⊗ 1̄ ← 3) gives
    // √d₃/√d₂ · B(2, 1, 3) · conj(κ_{1̄ = 3}) and appends dual(b) = 3,
    // undualized, to the domain (duality_manipulations.jl:33-65).
    let source = FusionTreePairKey::pair(
        FusionTreeKey::new(
            [z4(2), z4(1)],
            z4(3),
            [false, true],
            Vec::<SectorId>::new(),
            [MultiplicityIndex::ONE],
        ),
        FusionTreeKey::new(
            [z4(3)],
            z4(3),
            [false],
            Vec::<SectorId>::new(),
            Vec::<MultiplicityIndex>::new(),
        ),
    );
    let out = multiplicity_free_bendright_tree_pair(&rule, &source).unwrap();
    assert_eq!(out.len(), 1);
    let want = Complex64::new(4.0 / 3.0, 0.0)
        * Complex64::from_polar(2.0, 0.2 * 33.0)
        * Complex64::from_polar(1.0, 0.7 * 3.0).conj();
    assert_close(out[0].1, want);
    let expected = FusionTreePairKey::pair(
        FusionTreeKey::new(
            [z4(2)],
            z4(2),
            [false],
            Vec::<SectorId>::new(),
            Vec::<MultiplicityIndex>::new(),
        ),
        FusionTreeKey::new(
            [z4(3), z4(3)],
            z4(2),
            [false, false],
            Vec::<SectorId>::new(),
            [MultiplicityIndex::ONE],
        ),
    );
    assert_eq!(out[0].0, expected);

    // What: bendleft is conj(bendright) through the swapped pair
    // (duality_manipulations.jl:140-148): the domain leg 1̄ off (1 ⊗ 1̄ ← 2)
    // moves to the codomain as dual(1) = 3 with
    // conj(√d₂/√d₁ · B(1, 1, 2) · conj(κ_{1̄ = 3})).
    let source = FusionTreePairKey::pair(
        FusionTreeKey::new(
            [z4(2)],
            z4(2),
            [false],
            Vec::<SectorId>::new(),
            Vec::<MultiplicityIndex>::new(),
        ),
        FusionTreeKey::new(
            [z4(1), z4(1)],
            z4(2),
            [false, true],
            Vec::<SectorId>::new(),
            [MultiplicityIndex::ONE],
        ),
    );
    let out = multiplicity_free_bendleft_tree_pair(&rule, &source).unwrap();
    assert_eq!(out.len(), 1);
    let want = (Complex64::new(3.0 / 2.0, 0.0)
        * Complex64::from_polar(2.0, 0.2 * 23.0)
        * Complex64::from_polar(1.0, 0.7 * 3.0).conj())
    .conj();
    assert_close(out[0].1, want);
    let expected = FusionTreePairKey::pair(
        FusionTreeKey::new(
            [z4(2), z4(3)],
            z4(1),
            [false, false],
            Vec::<SectorId>::new(),
            [MultiplicityIndex::ONE],
        ),
        FusionTreeKey::new(
            [z4(1)],
            z4(1),
            [false],
            Vec::<SectorId>::new(),
            Vec::<MultiplicityIndex>::new(),
        ),
    );
    assert_eq!(out[0].0, expected);
}

#[test]
fn unique_cycles_match_keyed_multiplicity_free_cycles_on_complex_non_self_dual_z4() {
    // What: the unique fold's unrolled step sequence equals the keyed
    // multiplicity-free fold surgery on a rule whose dims, κ and B are all
    // nontrivial and complex, including dual first codomain legs (κ_a).
    let rule = ComplexZ4BendRule;
    let leg =
        |is_dual: bool| SectorLeg::new((0..4).map(|s| (z4(s), 1)).collect::<Vec<_>>(), is_dual);
    let mut checked = 0;
    for codomain_rank in 0..=3usize {
        for domain_rank in 0..=(4 - codomain_rank).min(2) {
            if codomain_rank + domain_rank == 0 {
                continue;
            }
            for dual_mask in 0..(1u32 << (codomain_rank + domain_rank)) {
                let is_dual = |i: usize| dual_mask & (1 << i) != 0;
                let hom = FusionTreeHomSpace::new(
                    FusionProductSpace::new(
                        (0..codomain_rank)
                            .map(|i| leg(is_dual(i)))
                            .collect::<Vec<_>>(),
                    ),
                    FusionProductSpace::new(
                        (0..domain_rank)
                            .map(|i| leg(is_dual(codomain_rank + i)))
                            .collect::<Vec<_>>(),
                    ),
                );
                for source in hom.fusion_tree_keys_uncached(&rule) {
                    for (unique, keyed) in [
                        (
                            unique_rigid_cycle_tree_pair(
                                &rule,
                                &source,
                                PreparedCycleDirection::Clockwise,
                            )
                            .unwrap(),
                            multiplicity_free_cycle_tree_pair(
                                &rule,
                                &source,
                                PreparedCycleDirection::Clockwise,
                            )
                            .unwrap(),
                        ),
                        (
                            unique_rigid_cycle_tree_pair(
                                &rule,
                                &source,
                                PreparedCycleDirection::Anticlockwise,
                            )
                            .unwrap(),
                            multiplicity_free_cycle_tree_pair(
                                &rule,
                                &source,
                                PreparedCycleDirection::Anticlockwise,
                            )
                            .unwrap(),
                        ),
                    ] {
                        assert_eq!(keyed.len(), 1, "{source:?}");
                        assert_eq!(unique.0, keyed[0].0, "{source:?}");
                        assert!((unique.1 - keyed[0].1).norm() < 1e-12, "{source:?}");
                        checked += 1;
                    }
                }
            }
        }
    }
    assert!(checked > 1000, "{checked}");
}
