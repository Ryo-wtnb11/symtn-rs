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
//!   `src/lie.rs` `codec_is_graded_lexicographic_and_roundtrips`,
//!   `checked_rigid_dimension_accepts_the_largest_su2_dynkin_label`.
//! - Table one-based labels and out-of-rank rejection: `tests/table_rule.rs`
//!   `codec_round_trips_and_rejects_labels_outside_the_rank`.
//! - Cantor pairing order and packed association independence:
//!   `tests/builtin_rules.rs` `product_sector_codec_uses_tensorkit_diagonal_component_order`,
//!   `packed_product_codec_is_association_independent`.
//! - Packed U(1) x fZ2 high-bit rejection: `tests/sector_labels.rs`
//!   `product_codec_reports_the_packing_failure_it_hit`.

use symtn_sectors::*;

use super::{s, table_identity};

#[test]
fn parity_ids_are_the_parity() {
    for (label, id) in [(FermionParityIrrep::EVEN, 0), (FermionParityIrrep::ODD, 1)] {
        assert_eq!(
            FermionParityFusionRule::new().encode_sector(&label),
            Ok(s(id))
        );
        assert_eq!(
            FermionParityFusionRule::new().decode_sector(s(id)),
            Ok(label)
        );
    }
}

#[test]
fn zn_ids_are_canonical_charges_below_the_modulus() {
    let z5 = ZNFusionRule::new(5).unwrap();
    for (charge, id) in [(0, 0), (4, 4), (-1, 4), (5, 0), (7, 2), (-6, 4)] {
        assert_eq!(
            z5.encode_sector(&ZNIrrep::new(charge, 5).unwrap()),
            Ok(s(id)),
            "{charge}"
        );
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
        assert_eq!(
            U1FusionRule.encode_sector(&U1Irrep::new(charge).unwrap()),
            Ok(s(id))
        );
        assert_eq!(
            U1FusionRule.decode_sector(s(id)),
            Ok(U1Irrep::new(charge).unwrap())
        );
    }
}

#[test]
fn cu1_ids_place_the_two_zero_charge_sectors_first() {
    let rule = CU1FusionRule::new();
    for (label, id) in [
        (CU1Irrep::VACUUM, 0),
        (CU1Irrep::PSEUDOSCALAR, 1),
        (CU1Irrep::from_twice_charge(1), 2),
        (CU1Irrep::from_twice_charge(2), 3),
        (CU1Irrep::from_twice_charge(u32::MAX), u32::MAX as usize + 1),
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
fn lie_group_ids_are_graded_then_lexicographic_through_the_typed_path() {
    let su4 = LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(
        fusion_symbols::lie::GroupId::su(4).unwrap(),
    )
    .unwrap();
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
        let label = su4.category().irrep(&labels).unwrap();
        assert_eq!(su4.try_encode_label(&label).unwrap(), s(id), "{labels:?}");
        assert_eq!(su4.try_decode_label(s(id)).unwrap(), label);
    }
    let su2 = LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(
        fusion_symbols::lie::GroupId::su(2).unwrap(),
    )
    .unwrap();
    for twice_spin in [0, 1, 7] {
        assert_eq!(
            su2.try_encode_label(&su2.category().irrep(&[twice_spin]).unwrap())
                .unwrap(),
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
    let fz2_u1 = symtn_sectors::ProductFusionRule::<_, _>::try_new(
        FermionParityFusionRule::new(),
        U1FusionRule,
    )
    .unwrap();
    let odd_one = product_sector(FermionParityIrrep::ODD, U1Irrep::new(1).unwrap());
    // (1, 2) -> 3 * 4 / 2 + 1.
    assert_eq!(fz2_u1.encode_sector(&odd_one), Ok(s(7)));
    assert_eq!(fz2_u1.decode_sector(s(7)), Ok(odd_one));

    let nested = symtn_sectors::ProductFusionRule::<_, _>::try_new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(
            FermionParityFusionRule::new(),
            U1FusionRule,
        )
        .unwrap(),
        SU2FusionRule,
    )
    .unwrap();
    let label = product_sector(
        product_sector(FermionParityIrrep::ODD, U1Irrep::new(-1).unwrap()),
        SU2Irrep::from_twice_spin(1),
    );
    // inner (1, 1) -> 4; outer (4, 1) -> 5 * 6 / 2 + 4.
    assert_eq!(nested.encode_sector(&label), Ok(s(19)));
    assert_eq!(nested.decode_sector(s(19)), Ok(label));
}

#[cfg(target_pointer_width = "64")]
#[test]
fn cantor_product_rejects_the_first_unpairable_ids() {
    let u1_u1 =
        symtn_sectors::ProductFusionRule::<_, _>::try_new(U1FusionRule, U1FusionRule).unwrap();
    let widest = product_sector(
        U1Irrep::new(i32::MAX).unwrap(),
        U1Irrep::new(i32::MAX).unwrap(),
    );
    assert_eq!(
        u1_u1.encode_sector(&widest),
        Err(FusionAlgebraError::ProductCodec(
            ProductSectorCodecError::CodecRejected
        ))
    );
    // Zigzag ids (2^32 - 2, 3) still pair: 9223372047592194047.
    let pairable = product_sector(U1Irrep::new(i32::MAX).unwrap(), U1Irrep::new(-2).unwrap());
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
    let last = product_sector(
        U1Irrep::new(i32::MAX).unwrap(),
        U1Irrep::new(889_516_852).unwrap(),
    );
    assert_eq!(
        SectorId::from(U1Irrep::new(889_516_852).unwrap()),
        s(1_779_033_704)
    );
    assert_eq!(
        u1_u1.encode_sector(&last),
        Ok(s(18_446_744_069_184_465_795))
    );
    let first = product_sector(
        U1Irrep::new(i32::MAX).unwrap(),
        U1Irrep::new(-889_516_853).unwrap(),
    );
    assert_eq!(
        SectorId::from(U1Irrep::new(-889_516_853).unwrap()),
        s(1_779_033_705)
    );
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
    let rule = U1Fz2::try_new(U1FusionRule, FermionParityFusionRule::new()).unwrap();
    let label = product_sector(U1Irrep::new(-1).unwrap(), FermionParityIrrep::ODD);
    assert_eq!(rule.encode_sector(&label), Ok(s(0x1_0000_0001)));
    assert_eq!(rule.decode_sector(s(0x1_0000_0001)), Ok(label));

    type Fz2Su2 = ProductFusionRule<
        FermionParityFusionRule,
        SU2FusionRule,
        PackedProductCodec<Fz2SectorLayout, Su2SectorLayout>,
    >;
    let rule = Fz2Su2::try_new(FermionParityFusionRule::new(), SU2FusionRule).unwrap();
    let label = product_sector(FermionParityIrrep::ODD, SU2Irrep::from_twice_spin(254));
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

/// The three multiplicity-free Lie codec routes: a rank-one factor (id =
/// `2j`), a rank-one torus (id = zigzag charge `0, 1, -1, 2, -2`), and a
/// multi-coordinate group through the interner. U(2) = (SU(2) x U(1))/Z2
/// admits `2j = q (mod 2)` and ranks `(2j, zigzag(q))` by total then
/// lexicographically, so excluded labels are holes in the id space.
#[test]
fn multiplicity_free_lie_ids_cover_every_codec_route() {
    use fusion_symbols::lie::GroupId;
    let mf = |group| LieGroupFusionRule::<MultiplicityFreeAdmissionMode>::new(group).unwrap();
    let su2 = mf(GroupId::su(2).unwrap());
    for twice_spin in [0, 1, 2, 3, 254, 255, i64::from(u32::MAX)] {
        let label = su2.category().irrep(&[twice_spin]).unwrap();
        let id = s(twice_spin as usize);
        assert_eq!(su2.encode_sector(&label), Ok(id));
        assert_eq!(su2.decode_sector(id), Ok(label));
        assert_eq!(su2.dual(id), id);
    }
    let u1 = mf(GroupId::u(1).unwrap());
    let charge = |q: i64| u1.category().irrep_with_charges(&[], &[q]).unwrap();
    for (q, id) in [(0, 0), (1, 1), (-1, 2), (2, 3), (-2, 4)] {
        assert_eq!(u1.encode_sector(&charge(q)), Ok(s(id)));
        assert_eq!(u1.decode_sector(s(id)), Ok(charge(q)));
        assert_eq!(u1.encode_sector(&charge(-q)), Ok(u1.dual(s(id))));
    }
    let u2 = mf(GroupId::u(2).unwrap());
    let label = |dj: i64, q: i64| u2.category().irrep_with_charges(&[dj], &[q]).unwrap();
    // (2j, q) -> id, and the dual (2j, -q).
    for (dj, q, id, dual) in [
        (0, 0, 0, 0),
        (1, 1, 4, 7),
        (2, 0, 5, 5),
        (0, 2, 6, 10),
        (1, -1, 7, 4),
        (2, 2, 17, 23),
    ] {
        assert_eq!(u2.encode_sector(&label(dj, q)), Ok(s(id)), "({dj}, {q})");
        assert_eq!(u2.decode_sector(s(id)), Ok(label(dj, q)));
        assert_eq!(u2.dual(s(id)), s(dual), "dual of ({dj}, {q})");
    }
    for hole in [1, 2, 3] {
        assert_eq!(
            u2.decode_sector(s(hole)),
            Err(FusionAlgebraError::InvalidSector { sector: s(hole) })
        );
    }
}
