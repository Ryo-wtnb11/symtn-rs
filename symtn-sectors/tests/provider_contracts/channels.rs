//! Observable fusion-channel order.
//!
//! Already pinned elsewhere:
//! - U(1) and SU(2) single pairs: `tests/builtin_rules.rs`
//!   `public_u1_irrep_roundtrips_compact_ids_and_fuses`,
//!   `public_su2_irrep_fusion_channels_match_doubled_spin_order`; `src/su2.rs`
//!   `frozen_su2_symbol_and_channel_oracle`.
//! - CU(1) complete small-label order: `tests/abelian_cu1_values.rs`
//!   `cu1_unary_data_and_fusion_order_follow_tensorkitsectors`.
//! - SU(3) octet channels sorted by id: `src/lie.rs`
//!   `su3_octet_self_fusion_has_dynkin_ordered_channels_and_multiplicities`.

use symtn_sectors::*;

use super::s;

fn ids(channels: SectorVec) -> Vec<usize> {
    channels.iter().map(|sector| sector.id()).collect()
}

#[test]
fn abelian_rules_have_one_channel() {
    let z5 = ZNFusionRule::new(5).unwrap();
    assert_eq!(ids(z5.fusion_channels(s(3), s(4))), [2]);
    assert_eq!(
        ids(ZNFusionRule::new(2).unwrap().fusion_channels(s(1), s(1))),
        [0]
    );
    assert_eq!(
        ids(FermionParityFusionRule::new().fusion_channels(s(1), s(1))),
        [0]
    );
    // -3 + 5 = 2, zigzag id 4.
    assert_eq!(ids(U1FusionRule.fusion_channels(s(5), s(10))), [4]);
}

#[test]
fn su2_channels_ascend_in_doubled_spin() {
    assert_eq!(ids(SU2FusionRule.fusion_channels(s(2), s(3))), [1, 3, 5]);
    assert_eq!(ids(SU2FusionRule.fusion_channels(s(3), s(2))), [1, 3, 5]);
}

#[cfg(feature = "cgc-gen")]
#[test]
fn lie_group_channels_are_sorted_by_encoded_id() {
    let su3 = LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(
        fusion_symbols::lie::GroupId::su(3).unwrap(),
    )
    .unwrap();
    // 3 = [1,0] (id 2), 3bar = [0,1] (id 1), 6 = [2,0] (id 5), 8 = [1,1] (id 4).
    assert_eq!(ids(su3.try_fusion_channels(s(2), s(2)).unwrap()), [1, 5]);
    assert_eq!(ids(su3.try_fusion_channels(s(2), s(1)).unwrap()), [0, 4]);
    assert_eq!(
        ids(su3.try_fusion_channels_in_table(s(2), s(2)).unwrap()),
        [1, 5]
    );
    let su2 = LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(
        fusion_symbols::lie::GroupId::su(2).unwrap(),
    )
    .unwrap();
    assert_eq!(ids(su2.try_fusion_channels(s(1), s(1)).unwrap()), [0, 2]);
    assert_eq!(ids(su2.try_fusion_channels(s(2), s(3)).unwrap()), [1, 3, 5]);
}

#[test]
fn table_channels_follow_the_object_order() {
    let fib = TableRule::fibonacci();
    assert_eq!(ids(fib.fusion_channels(s(1), s(1))), [0, 1]);
    // Ising: 1 = unit, 2 = psi, 3 = sigma.
    let ising = TableRule::ising();
    assert_eq!(ids(ising.fusion_channels(s(2), s(2))), [0, 1]);
    assert_eq!(ids(ising.fusion_channels(s(1), s(2))), [2]);
    assert_eq!(ids(ising.fusion_channels(s(1), s(1))), [0]);
    // Rep(S3): the two-dimensional object fuses into all three objects.
    let rep_s3 = TableRule::by_uuid("37fa4e9c-97c6-11f1-8ce9-19309dfc597c").unwrap();
    assert_eq!(ids(rep_s3.fusion_channels(s(2), s(2))), [0, 1, 2]);
    assert_eq!(ids(rep_s3.fusion_channels(s(1), s(2))), [2]);
}

#[test]
fn product_channels_enumerate_the_left_factor_fastest() {
    // x = (1/2, 1): left SU(2) channels [0, 2], right [0, 2, 4] (doubled).
    let su2_su2 =
        symtn_sectors::ProductFusionRule::<_, _>::try_new(SU2FusionRule, SU2FusionRule).unwrap();
    let x = su2_su2
        .encode_sector(&product_sector(
            SU2Irrep::from_twice_spin(1),
            SU2Irrep::from_twice_spin(2),
        ))
        .unwrap();
    let channels = su2_su2.try_fusion_channels(x, x).unwrap();
    // Cantor ids of (0,0), (2,0), (0,2), (2,2), (0,4), (2,4).
    assert_eq!(ids(channels.clone()), [0, 5, 3, 12, 10, 23]);
    let labels: Vec<_> = channels
        .iter()
        .map(|&sector| {
            let label = su2_su2.decode_sector(sector).unwrap();
            (label.left().twice_spin(), label.right().twice_spin())
        })
        .collect();
    assert_eq!(labels, [(0, 0), (2, 0), (0, 2), (2, 2), (0, 4), (2, 4)]);
    assert_eq!(ids(su2_su2.fusion_channels(x, x)), [0, 5, 3, 12, 10, 23]);

    // The same order under a packed codec, whose ids are not monotone in it.
    let packed = ProductFusionRule::<
        SU2FusionRule,
        SU2FusionRule,
        PackedProductCodec<Su2SectorLayout, Su2SectorLayout>,
    >::try_new(SU2FusionRule, SU2FusionRule)
    .unwrap();
    let x = s(1 | (2 << 8));
    assert_eq!(
        ids(packed.try_fusion_channels(x, x).unwrap()),
        [0, 2, 512, 514, 1024, 1026]
    );
}

#[test]
fn nested_product_channels_enumerate_the_inner_product_fastest() {
    // ((1/2, odd), 1/2): inner channels (0, even), (2, even) = Cantor ids 0, 5;
    // outer right channels 0, 2.
    let rule = symtn_sectors::ProductFusionRule::<_, _>::try_new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(
            SU2FusionRule,
            ZNFusionRule::new(2).unwrap(),
        )
        .unwrap(),
        SU2FusionRule,
    )
    .unwrap();
    let x = rule
        .encode_sector(&product_sector(
            product_sector(SU2Irrep::from_twice_spin(1), ZNIrrep::new(1, 2).unwrap()),
            SU2Irrep::from_twice_spin(1),
        ))
        .unwrap();
    // Cantor ids of (0,0), (5,0), (0,2), (5,2).
    assert_eq!(ids(rule.try_fusion_channels(x, x).unwrap()), [0, 20, 3, 33]);
}

/// Multiplicity-free Lie channels are sorted by encoded id on every codec
/// route, and N is their indicator.
#[test]
fn multiplicity_free_lie_channels_follow_the_codec_order() {
    use fusion_symbols::lie::GroupId;
    let mf = |group| LieGroupFusionRule::<MultiplicityFreeAdmissionMode>::new(group).unwrap();
    let su2 = mf(GroupId::su(2).unwrap());
    assert_eq!(ids(su2.fusion_channels(s(2), s(3))), [1, 3, 5]);
    assert_eq!(su2.nsymbol(s(2), s(3), s(3)), 1);
    assert_eq!(su2.nsymbol(s(2), s(3), s(2)), 0);
    // U(1) ids 0, 1, -1, 2, -2: 1 + (-2) = -1.
    let u1 = mf(GroupId::u(1).unwrap());
    assert_eq!(ids(u1.fusion_channels(s(1), s(4))), [2]);
    assert_eq!(u1.nsymbol(s(1), s(4), s(2)), 1);
    assert_eq!(u1.nsymbol(s(1), s(4), s(1)), 0);
    // U(2): fundamental (1, 1) = 4 and its dual (1, -1) = 7.
    // 4 x 4 = det (0, 2) = 6 + (2, 2) = 17; 4 x 7 = (0, 0) = 0 + (2, 0) = 5.
    let u2 = mf(GroupId::u(2).unwrap());
    assert_eq!(ids(u2.fusion_channels(s(4), s(4))), [6, 17]);
    assert_eq!(ids(u2.fusion_channels(s(4), s(7))), [0, 5]);
    assert_eq!(u2.nsymbol(s(4), s(7), s(5)), 1);
    assert_eq!(u2.nsymbol(s(4), s(4), s(5)), 0);
}
