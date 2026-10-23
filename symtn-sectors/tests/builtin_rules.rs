//! Public label, codec, style and symbol contracts of the built-in fusion
//! rules and their products.

use symtn_sectors::*;

fn u1(charge: i32) -> SectorId {
    U1Irrep::new(charge).sector_id()
}

fn excluded_u1_id() -> SectorId {
    SectorId::new(u32::MAX as usize)
}

fn z2_even() -> SectorId {
    Z2Irrep::EVEN.sector_id()
}

fn z2_odd() -> SectorId {
    Z2Irrep::ODD.sector_id()
}

fn su2(twice_spin: usize) -> SectorId {
    SU2Irrep::from_twice_spin(twice_spin).sector_id()
}

#[test]
fn fusion_style_kind_matches_tensorkit_multiplicity_free_split() {
    assert!(FusionStyleKind::Unique.is_multiplicity_free());
    assert!(FusionStyleKind::Simple.is_multiplicity_free());
    assert!(!FusionStyleKind::Generic.is_multiplicity_free());
    assert!(!FusionStyleKind::Unique.has_multiple_outputs());
    assert!(FusionStyleKind::Simple.has_multiple_outputs());
    assert!(FusionStyleKind::Generic.has_multiple_outputs());
    assert!(!FusionStyleKind::Unique.has_multiplicity());
    assert!(!FusionStyleKind::Simple.has_multiplicity());
    assert!(FusionStyleKind::Generic.has_multiplicity());
}

#[test]
fn braiding_style_kind_matches_tensorkit_hierarchy() {
    assert!(!BraidingStyleKind::NoBraiding.has_braiding());
    assert!(BraidingStyleKind::Bosonic.has_braiding());
    assert!(BraidingStyleKind::Fermionic.has_braiding());
    assert!(BraidingStyleKind::Anyonic.has_braiding());

    assert!(!BraidingStyleKind::NoBraiding.is_symmetric());
    assert!(BraidingStyleKind::Bosonic.is_symmetric());
    assert!(BraidingStyleKind::Fermionic.is_symmetric());
    assert!(!BraidingStyleKind::Anyonic.is_symmetric());

    assert!(BraidingStyleKind::Bosonic.is_bosonic());
    assert!(!BraidingStyleKind::Fermionic.is_bosonic());
    assert_eq!(
        BraidingStyleKind::Bosonic.combined_with(BraidingStyleKind::Fermionic),
        BraidingStyleKind::Fermionic
    );
    assert_eq!(
        BraidingStyleKind::Fermionic.combined_with(BraidingStyleKind::Anyonic),
        BraidingStyleKind::Anyonic
    );
    assert_eq!(
        BraidingStyleKind::Anyonic.combined_with(BraidingStyleKind::NoBraiding),
        BraidingStyleKind::NoBraiding
    );
}

#[test]
fn product_sector_codec_uses_tensorkit_diagonal_component_order() {
    let expected = [
        (0, 0),
        (0, 1),
        (1, 0),
        (0, 2),
        (1, 1),
        (2, 0),
        (0, 3),
        (1, 2),
        (2, 1),
        (3, 0),
    ];

    for (id, &(left, right)) in expected.iter().enumerate() {
        let encoded = TensorKitProductCodec::encode(SectorId::new(left), SectorId::new(right));
        assert_eq!(encoded, SectorId::new(id));
        assert_eq!(
            TensorKitProductCodec::decode(encoded),
            Some((SectorId::new(left), SectorId::new(right)))
        );
    }
}

#[cfg(target_pointer_width = "64")]
#[test]
fn packed_product_codec_is_association_independent() {
    // What: fixed-width product IDs flatten numerically regardless of the
    // source-level association used to build the same ordered leaves.
    type FpU1Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
    type FpU1Layout = ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>;
    type LeftAssociated = PackedProductCodec<FpU1Layout, Su2SectorLayout>;
    type U1Su2Codec = PackedProductCodec<U1SectorLayout, Su2SectorLayout>;
    type U1Su2Layout = ProductSectorLayout<U1SectorLayout, Su2SectorLayout>;
    type RightAssociated = PackedProductCodec<Fz2SectorLayout, U1Su2Layout>;

    for (parity, charge, twice_spin) in [
        (z2_even(), excluded_u1_id(), 0),
        (z2_odd(), u1(-1), 1),
        (z2_even(), u1(0), 2),
        (z2_odd(), u1(i32::MAX), 254),
    ] {
        let left = FpU1Codec::encode(parity, charge);
        let left_associated = LeftAssociated::encode(left, su2(twice_spin));
        let right = U1Su2Codec::encode(charge, su2(twice_spin));
        let right_associated = RightAssociated::encode(parity, right);
        assert_eq!(left_associated, right_associated);
    }
}

#[test]
fn rigid_symbols_separate_twist_from_frobenius_schur_phase() {
    let fermion = FermionParityFusionRule;
    let odd = SectorId::new(1);
    assert_eq!(fermion.dim_scalar(odd), 1.0);
    assert_eq!(fermion.twist_scalar(odd), -1.0);
    assert_eq!(fermion.frobenius_schur_phase_scalar(odd), 1.0);

    let su2 = SU2FusionRule;
    let half = SU2Irrep::from_twice_spin(1).sector_id();
    assert_eq!(su2.dim_scalar(half), 2.0);
    assert_eq!(su2.twist_scalar(half), 1.0);
    assert_eq!(su2.frobenius_schur_phase_scalar(half), -1.0);
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

    let left_rule =
        symtn_sectors::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule)
            .unwrap();
    let chained_rule = symtn_sectors::ProductFusionRule::<_, _>::try_new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule)
            .unwrap(),
        SU2FusionRule,
    )
    .unwrap();
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
    let rule = FpU1Rule::try_new(FermionParityFusionRule, U1FusionRule).unwrap();
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
    let left_rule = FpU1Rule::try_new(FermionParityFusionRule, U1FusionRule).unwrap();
    let rule = FpU1Su2Rule::try_new(
        FpU1Rule::try_new(FermionParityFusionRule, U1FusionRule).unwrap(),
        SU2FusionRule,
    )
    .unwrap();
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
#[should_panic(expected = "invalid fusion sector SectorId(2)")]
fn product_fusion_rule_panics_on_an_invalid_component_sector() {
    type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
    let rule = FpU1Rule::try_new(FermionParityFusionRule, U1FusionRule).unwrap();
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
