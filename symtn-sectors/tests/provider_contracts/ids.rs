//! `SectorId` encodings of representative and boundary labels.
//!
//! Already pinned elsewhere:
//! - U(1) zigzag at the 32-bit extremes and the `i32::MIN` exclusion:
//!   `src/abelian.rs` `u1_zigzag_encoding_is_unchanged_at_32_bit_extremes`,
//!   `u1_label_boundary_rejects_only_i32_min`; `tests/sector_labels.rs`
//!   `u1_codec_rejects_ids_outside_the_zigzag_domain`.
//! - Z2/fZ2 id 2 rejection: `tests/sector_labels.rs`
//!   `parity_codecs_round_trip_and_reject_non_parity_ids`.
//! - SU(2) id 255 rejection: `tests/sector_labels.rs`
//!   `su2_codec_round_trips_up_to_the_supported_doubled_spin`.
//! - CU(1) id `u32::MAX + 1` rejection: `tests/sector_labels.rs`
//!   `cu1_codec_and_checked_overflow_preserve_the_tensor_kit_boundary`.
//! - SU(3) graded-lex ids 0..6 and the SU(2) `u32::MAX` Dynkin edge:
//!   `src/sun.rs` `codec_is_graded_lexicographic_and_roundtrips`,
//!   `checked_rigid_dimension_accepts_the_largest_su2_dynkin_label`.
//! - Table one-based labels and out-of-rank rejection: `tests/table_rule.rs`
//!   `codec_round_trips_and_rejects_labels_outside_the_rank`.
//! - Cantor pairing order and packed association independence:
//!   `tests/builtin_rules.rs` `product_sector_codec_uses_tensorkit_diagonal_component_order`,
//!   `packed_product_codec_is_association_independent`.
//! - Packed U(1) x fZ2 high-bit rejection: `tests/sector_labels.rs`
//!   `product_codec_reports_the_packing_failure_it_hit`.

use symtn_sectors::*;

use crate::{s, table_identity};

#[test]
fn parity_ids_are_the_parity() {
    for (label, id) in [(Z2Irrep::EVEN, 0), (Z2Irrep::ODD, 1)] {
        assert_eq!(Z2FusionRule.encode_sector(&label), Ok(s(id)));
        assert_eq!(FermionParityFusionRule.encode_sector(&label), Ok(s(id)));
        assert_eq!(Z2FusionRule.decode_sector(s(id)), Ok(label));
        assert_eq!(FermionParityFusionRule.decode_sector(s(id)), Ok(label));
    }
}

#[test]
fn zn_ids_are_canonical_charges_below_the_modulus() {
    let z5 = ZNFusionRule::new(5).unwrap();
    for (charge, id) in [(0, 0), (4, 4), (-1, 4), (5, 0), (7, 2), (-6, 4)] {
        assert_eq!(z5.encode_sector(&z5.irrep(charge)), Ok(s(id)), "{charge}");
    }
    for id in 0..5 {
        assert_eq!(z5.decode_sector(s(id)).unwrap().charge(), id as u32);
    }
    assert_eq!(
        z5.decode_sector(s(5)),
        Err(FusionAlgebraError::InvalidSector { sector: s(5) })
    );

    let widest = ZNFusionRule::new(u32::MAX).unwrap();
    let last = s(u32::MAX as usize - 1);
    assert_eq!(widest.decode_sector(last).unwrap().charge(), u32::MAX - 1);
    assert_eq!(
        widest.decode_sector(s(u32::MAX as usize)),
        Err(FusionAlgebraError::InvalidSector {
            sector: s(u32::MAX as usize)
        })
    );

    assert_eq!(
        ZNFusionRule::new(0).unwrap_err(),
        FusionAlgebraError::UnrepresentableSectorLabel {
            rule: RuleIdentity::of_type::<ZNFusionRule>(),
            label: "Z_N modulus must be nonzero".into(),
        }
    );
}

#[test]
fn u1_small_charges_interleave_by_sign() {
    for (charge, id) in [(0, 0), (-1, 1), (1, 2), (-2, 3), (2, 4), (-3, 5)] {
        assert_eq!(U1FusionRule.encode_sector(&U1Irrep::new(charge)), Ok(s(id)));
        assert_eq!(U1FusionRule.decode_sector(s(id)), Ok(U1Irrep::new(charge)));
    }
}

#[test]
fn cu1_ids_place_the_two_zero_charge_sectors_first() {
    let rule = CU1FusionRule;
    for (label, id) in [
        (CU1Irrep::VACUUM, 0),
        (CU1Irrep::PSEUDOSCALAR, 1),
        (CU1Irrep::from_twice_charge(1), 2),
        (CU1Irrep::from_twice_charge(2), 3),
        (
            CU1Irrep::from_twice_charge(CU1_MAX_TWICE_CHARGE),
            u32::MAX as usize,
        ),
    ] {
        assert_eq!(rule.encode_sector(&label), Ok(s(id)));
        assert_eq!(rule.decode_sector(s(id)), Ok(label));
    }
}

#[test]
fn su2_ids_are_doubled_spins() {
    for twice_spin in [0, 1, 2, 3, SU2_MAX_DOUBLED_SPIN] {
        assert_eq!(
            SU2FusionRule.encode_sector(&SU2Irrep::from_twice_spin(twice_spin)),
            Ok(s(twice_spin))
        );
    }
    assert_eq!(SU2_MAX_DOUBLED_SPIN, 254);
    assert_eq!(SU2Irrep::try_from_twice_spin(255), None);
}

#[cfg(feature = "cgc-gen")]
#[test]
fn sun_ids_are_graded_then_lexicographic_through_the_typed_path() {
    let su4 = SUNFusionRule::new(4).unwrap();
    for (labels, id) in [
        (vec![0, 0, 0], 0),
        (vec![0, 0, 1], 1),
        (vec![0, 1, 0], 2),
        (vec![1, 0, 0], 3),
        (vec![0, 0, 2], 4),
        (vec![0, 1, 1], 5),
        (vec![0, 2, 0], 6),
        (vec![1, 0, 1], 7),
        (vec![1, 1, 0], 8),
        (vec![2, 0, 0], 9),
    ] {
        assert_eq!(su4.try_encode_label(&labels).unwrap(), s(id), "{labels:?}");
        assert_eq!(su4.try_decode_label(s(id)).unwrap(), labels);
    }
    let su2 = SUNFusionRule::new(2).unwrap();
    for twice_spin in [0, 1, 7] {
        assert_eq!(
            su2.try_encode_label(&vec![twice_spin]).unwrap(),
            s(twice_spin as usize)
        );
    }
}

#[test]
fn table_encode_rejection_names_the_label_and_rank() {
    let fib = TableRule::fibonacci();
    assert_eq!(fib.encode_sector(&TableSector::new(1)), Ok(s(0)));
    assert_eq!(fib.encode_sector(&TableSector::new(2)), Ok(s(1)));
    assert_eq!(
        fib.encode_sector(&TableSector::new(3)),
        Err(FusionAlgebraError::UnrepresentableSectorLabel {
            rule: table_identity(&fusion_symbols::table::TableCategory::fibonacci()),
            label: "table object 3 (rank 2)".into(),
        })
    );
    assert_eq!(
        fib.decode_sector(s(2)),
        Err(FusionAlgebraError::InvalidSector { sector: s(2) })
    );
}

#[test]
fn cantor_product_ids_nest_through_both_factors() {
    // Cantor(l, r) = (l + r)(l + r + 1) / 2 + l.
    let fz2_u1 = FermionParityFusionRule.product(U1FusionRule);
    let odd_one = product_sector(Z2Irrep::ODD, U1Irrep::new(1));
    // (1, 2) -> 3 * 4 / 2 + 1.
    assert_eq!(fz2_u1.encode_sector(&odd_one), Ok(s(7)));
    assert_eq!(fz2_u1.decode_sector(s(7)), Ok(odd_one));

    let nested = FermionParityFusionRule
        .product(U1FusionRule)
        .product(SU2FusionRule);
    let label = product_sector(
        product_sector(Z2Irrep::ODD, U1Irrep::new(-1)),
        SU2Irrep::from_twice_spin(1),
    );
    // inner (1, 1) -> 4; outer (4, 1) -> 5 * 6 / 2 + 4.
    assert_eq!(nested.encode_sector(&label), Ok(s(19)));
    assert_eq!(nested.decode_sector(s(19)), Ok(label));
}

#[cfg(target_pointer_width = "64")]
#[test]
fn cantor_product_rejects_the_first_unpairable_ids() {
    let u1_u1 = U1FusionRule.product(U1FusionRule);
    let widest = product_sector(U1Irrep::new(i32::MAX), U1Irrep::new(i32::MAX));
    assert_eq!(
        u1_u1.encode_sector(&widest),
        Err(FusionAlgebraError::ProductCodec(
            ProductSectorCodecError::CodecRejected
        ))
    );
    // Zigzag ids (2^32 - 2, 3) still pair: 9223372047592194047.
    let pairable = product_sector(U1Irrep::new(i32::MAX), U1Irrep::new(-2));
    assert_eq!(
        u1_u1.encode_sector(&pairable),
        Ok(s(9_223_372_047_592_194_047))
    );
    // The exact edge for left id l = zigzag(i32::MAX) = 2^32 - 2: Cantor needs
    // T(s) + l <= 2^64 - 1 with s = l + r, T(s) = s(s + 1) / 2.
    // s = 6074000998: T = 18446744064889498501, + l = 18446744069184465795 fits;
    // s = 6074000999: T = 18446744070963499500, + l = 18446744075258466794 > 2^64 - 1.
    // So r = 1779033704 (zigzag of charge 889516852) is the last right id and
    // r = 1779033705 (zigzag of charge -889516853) the first unpairable one.
    let last = product_sector(U1Irrep::new(i32::MAX), U1Irrep::new(889_516_852));
    assert_eq!(U1Irrep::new(889_516_852).sector_id(), s(1_779_033_704));
    assert_eq!(
        u1_u1.encode_sector(&last),
        Ok(s(18_446_744_069_184_465_795))
    );
    let first = product_sector(U1Irrep::new(i32::MAX), U1Irrep::new(-889_516_853));
    assert_eq!(U1Irrep::new(-889_516_853).sector_id(), s(1_779_033_705));
    assert_eq!(
        u1_u1.encode_sector(&first),
        Err(FusionAlgebraError::ProductCodec(
            ProductSectorCodecError::CodecRejected
        ))
    );
}

#[cfg(target_pointer_width = "64")]
#[test]
fn packed_product_ids_place_the_left_factor_in_the_low_bits() {
    type U1Fz2 = ProductFusionRule<
        U1FusionRule,
        FermionParityFusionRule,
        PackedProductCodec<U1SectorLayout, Fz2SectorLayout>,
    >;
    let rule = U1Fz2::default();
    let label = product_sector(U1Irrep::new(-1), Z2Irrep::ODD);
    assert_eq!(rule.encode_sector(&label), Ok(s(0x1_0000_0001)));
    assert_eq!(rule.decode_sector(s(0x1_0000_0001)), Ok(label));

    type Fz2Su2 = ProductFusionRule<
        FermionParityFusionRule,
        SU2FusionRule,
        PackedProductCodec<Fz2SectorLayout, Su2SectorLayout>,
    >;
    let rule = Fz2Su2::default();
    let label = product_sector(Z2Irrep::ODD, SU2Irrep::from_twice_spin(254));
    assert_eq!(rule.encode_sector(&label), Ok(s(509)));
    // 510 = spin component 255: inside the 8-bit width, outside SU(2)'s range.
    assert_eq!(
        rule.decode_sector(s(510)),
        Err(FusionAlgebraError::ProductCodec(
            ProductSectorCodecError::ComponentOutOfRange {
                component: ProductSectorComponent::Right,
                sector: s(255),
                bits: 8,
            }
        ))
    );
    assert_eq!(
        rule.decode_sector(s(1 << 9)),
        Err(FusionAlgebraError::ProductCodec(
            ProductSectorCodecError::InvalidHighBits {
                sector: s(1 << 9),
                total_bits: 9,
            }
        ))
    );

    type TooWide =
        PackedProductCodec<U1SectorLayout, ProductSectorLayout<U1SectorLayout, U1SectorLayout>>;
    assert_eq!(
        TooWide::encode_checked(s(0), s(0)),
        Err(ProductSectorCodecError::WidthOverflow {
            left_bits: 32,
            right_bits: 64,
            available_bits: 64,
        })
    );
}
