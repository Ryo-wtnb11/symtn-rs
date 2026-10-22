//! Checked errors and which failure wins when several inputs are invalid.
//!
//! Already pinned elsewhere:
//! - U(1) overflow for fusion and N (left charge): `tests/abelian_cu1_values.rs`
//!   `u1_fusion_leaving_the_charge_range_is_an_error`.
//! - CU(1) codec bound and its precedence (coupled first for N):
//!   `tests/abelian_cu1_values.rs` `cu1_fusion_leaving_the_charge_range_is_an_error`;
//!   `tests/sector_labels.rs` `cu1_checked_validation_preserves_the_shared_error_precedence`.
//! - SU(2) fusion closure beyond 254 and its validation order:
//!   `src/su2.rs` `checked_su2_preserves_validation_order_and_domain`.
//! - SU(N) label limits, lazy allocation failure, and absent-channel
//!   shaped-empty F/R blocks: `src/sun.rs` `validation_and_identity_are_typed_and_stable`,
//!   `checked_rigid_dimension_accepts_the_largest_su2_dynkin_label`,
//!   `rigid_failures_remain_typed`.
//! - Table R/twist rejection message on a fusion-only entry:
//!   `tests/table_rule.rs` `fusion_only_entry_has_f_data_and_panics_on_r_and_twist`.

use std::panic::{self, AssertUnwindSafe};

use symtn_sectors::*;

use crate::s;

fn invalid(id: usize) -> Result<usize, FusionAlgebraError> {
    Err(FusionAlgebraError::InvalidSector { sector: s(id) })
}

#[test]
fn zn_checks_the_coupled_sector_first() {
    let z5 = ZNFusionRule::new(5).unwrap();
    assert_eq!(z5.try_nsymbol(s(5), s(6), s(7)), invalid(7));
    assert_eq!(z5.try_nsymbol(s(5), s(6), s(0)), invalid(5));
    assert_eq!(z5.try_nsymbol(s(0), s(6), s(0)), invalid(6));
    assert_eq!(
        z5.try_fusion_channels(s(5), s(6)),
        Err(FusionAlgebraError::InvalidSector { sector: s(5) })
    );
    assert_eq!(
        z5.try_dual_sector(s(5)),
        Err(FusionAlgebraError::InvalidSector { sector: s(5) })
    );
}

#[test]
fn parity_rules_check_left_right_then_coupled() {
    assert_eq!(Z2FusionRule.try_nsymbol(s(2), s(3), s(4)), invalid(2));
    assert_eq!(
        FermionParityFusionRule.try_nsymbol(s(2), s(3), s(4)),
        invalid(2)
    );
    assert_eq!(Z2FusionRule.try_nsymbol(s(0), s(3), s(4)), invalid(3));
    assert_eq!(
        FermionParityFusionRule.try_nsymbol(s(0), s(0), s(4)),
        invalid(4)
    );
}

#[test]
fn table_checks_left_right_then_coupled() {
    let fib = TableRule::fibonacci();
    assert_eq!(fib.try_nsymbol(s(2), s(3), s(4)), invalid(2));
    assert_eq!(fib.try_nsymbol(s(0), s(3), s(4)), invalid(3));
    assert_eq!(fib.try_nsymbol(s(0), s(0), s(4)), invalid(4));
}

#[test]
fn u1_validates_every_label_before_reporting_overflow() {
    let u1 = |q: i32| U1Irrep::new(q).sector_id();
    let excluded = u32::MAX as usize;
    assert_eq!(
        U1FusionRule.try_nsymbol(u1(i32::MAX), u1(1), s(excluded)),
        invalid(excluded)
    );
    // Stricter than TensorKit's N = 0: the sum itself is unrepresentable.
    assert_eq!(
        U1FusionRule.try_nsymbol(u1(i32::MAX), u1(1), u1(0)),
        Err(FusionAlgebraError::U1FusionOverflow {
            left: i32::MAX,
            right: 1,
        })
    );
    assert_eq!(
        U1FusionRule.try_fusion_channels(u1(i32::MIN + 1), u1(-1)),
        Err(FusionAlgebraError::U1FusionOverflow {
            left: i32::MIN + 1,
            right: -1,
        })
    );
    assert_eq!(
        U1Irrep::new(i32::MAX).checked_fuse(U1Irrep::new(1)),
        Err(FusionAlgebraError::U1FusionOverflow {
            left: i32::MAX,
            right: 1,
        })
    );
}

#[test]
fn su2_n_rejects_unrepresentable_closure_after_validating_labels() {
    let closure = Err(FusionAlgebraError::FusionNotRepresentable {
        left: s(128),
        right: s(127),
    });
    // Even for a coupled sector that is not in the closure at all.
    assert_eq!(SU2FusionRule.try_nsymbol(s(128), s(127), s(1)), closure);
    assert_eq!(SU2FusionRule.try_nsymbol(s(128), s(127), s(254)), closure);
    assert_eq!(
        SU2FusionRule.try_nsymbol(s(128), s(127), s(255)),
        invalid(255)
    );
    assert_eq!(SU2FusionRule.try_nsymbol(s(127), s(127), s(254)), Ok(1));
    assert_eq!(
        SU2FusionRule.try_dual_sector(s(255)),
        Err(FusionAlgebraError::InvalidSector { sector: s(255) })
    );
}

#[cfg(feature = "cgc-gen")]
#[test]
fn sun_errors_are_typed_and_checked_left_right_coupled() {
    assert!(matches!(
        SUNFusionRule::new(1),
        Err(SUNFusionRuleError::InvalidRank { n: 1 })
    ));
    assert!(matches!(
        SUNFusionRule::new(0),
        Err(SUNFusionRuleError::InvalidRank { n: 0 })
    ));
    let su3 = SUNFusionRule::new(3).unwrap();
    assert!(matches!(
        su3.encode_dynkin(&[1, 0, 0]),
        Err(SUNFusionRuleError::LabelLength {
            expected: 2,
            found: 3
        })
    ));
    assert!(matches!(
        su3.encode_dynkin(&[0, -1]),
        Err(SUNFusionRuleError::NegativeLabel {
            index: 1,
            value: -1
        })
    ));
    assert!(matches!(
        su3.try_encode_label(&vec![-2, 0]),
        Err(SUNFusionRuleError::NegativeLabel {
            index: 0,
            value: -2
        })
    ));

    #[cfg(target_pointer_width = "64")]
    {
        let su2 = SUNFusionRule::new(2).unwrap();
        let first = u32::MAX as usize + 1;
        let unrepresentable = |result: Result<SectorVec, SUNFusionRuleError>, id: usize| matches!(result, Err(SUNFusionRuleError::UnrepresentableLabel { labels }) if labels == [id as i64]);
        assert!(unrepresentable(
            su2.try_fusion_channels(s(first), s(first + 1)),
            first
        ));
        assert!(unrepresentable(
            su2.try_fusion_channels(s(0), s(first + 1)),
            first + 1
        ));
        assert!(matches!(
            su2.try_nsymbol(s(first), s(first + 1), s(first + 2)),
            Err(SUNFusionRuleError::UnrepresentableLabel { labels }) if labels == [first as i64]
        ));
        assert!(matches!(
            su2.try_nsymbol(s(0), s(0), s(first + 2)),
            Err(SUNFusionRuleError::UnrepresentableLabel { labels }) if labels == [first as i64 + 2]
        ));
        assert!(matches!(
            su2.try_dual(s(first)),
            Err(SUNFusionRuleError::UnrepresentableLabel { labels }) if labels == [first as i64]
        ));
    }
}

#[test]
fn product_errors_follow_outer_decode_then_left_factor_then_right_factor() {
    // Cantor (2, 3) = 17: both components invalid for Z2; the left factor wins.
    let z2_z2 = Z2FusionRule.product(Z2FusionRule);
    assert_eq!(z2_z2.try_nsymbol(s(17), s(0), s(0)), invalid(2));
    assert_eq!(
        z2_z2.try_fusion_channels(s(17), s(0)),
        Err(FusionAlgebraError::InvalidSector { sector: s(2) })
    );
    assert_eq!(
        z2_z2.try_dual_sector(s(17)),
        Err(FusionAlgebraError::InvalidSector { sector: s(2) })
    );
    // Cantor (0, 3) = 6: only the right component is invalid.
    assert_eq!(z2_z2.try_nsymbol(s(6), s(0), s(0)), invalid(3));
    // Within one factor its own order applies: Z2 checks left, right, coupled
    // per factor; here the left factor sees (0, 0, 2) and the right (0, 0, 0).
    // Cantor (2, 0) = 5.
    assert_eq!(z2_z2.try_nsymbol(s(0), s(0), s(5)), invalid(2));
}

#[cfg(target_pointer_width = "64")]
#[test]
fn product_left_factor_overflow_beats_right_factor_domain_error() {
    let u1_z2 = U1FusionRule.product(Z2FusionRule);
    // Cantor (zigzag(i32::MAX), 3) = 9223372047592194047; Cantor (zigzag(1), 0) = 5.
    assert_eq!(
        u1_z2.try_nsymbol(s(9_223_372_047_592_194_047), s(5), s(0)),
        Err(FusionAlgebraError::U1FusionOverflow {
            left: i32::MAX,
            right: 1,
        })
    );
}

#[cfg(target_pointer_width = "64")]
#[test]
fn product_codec_decode_beats_factor_errors() {
    type U1Fz2 = ProductFusionRule<
        U1FusionRule,
        FermionParityFusionRule,
        PackedProductCodec<U1SectorLayout, Fz2SectorLayout>,
    >;
    let rule = U1Fz2::default();
    let bad_u1 = s(u32::MAX as usize);
    assert_eq!(
        rule.try_nsymbol(bad_u1, s(0), s(1 << 33)),
        Err(FusionAlgebraError::ProductCodec(
            ProductSectorCodecError::InvalidHighBits {
                sector: s(1 << 33),
                total_bits: 33,
            }
        ))
    );
    assert_eq!(
        rule.try_nsymbol(bad_u1, s(0), s(0)),
        invalid(u32::MAX as usize)
    );
}

#[cfg(target_pointer_width = "64")]
#[test]
fn product_fusion_reports_an_unpairable_channel_after_both_factors() {
    let u1_u1 = U1FusionRule.product(U1FusionRule);
    let q = 3 << 28;
    let x = u1_u1
        .encode_sector(&product_sector(U1Irrep::new(q), U1Irrep::new(q)))
        .unwrap();
    assert_eq!(x, s(5_188_146_773_952_036_864));
    // Channel (2q, 2q) has zigzag ids (3 * 2^30, 3 * 2^30): no Cantor id.
    assert_eq!(
        u1_u1.try_fusion_channels(x, x),
        Err(FusionAlgebraError::ProductCodec(
            ProductSectorCodecError::CodecRejected
        ))
    );
    assert_eq!(u1_u1.try_nsymbol(x, x, s(0)), Ok(0));
}

#[test]
fn product_with_a_fusion_only_factor_panics_with_the_table_message() {
    let rule = TableRule::by_uuid("37fa4e9c-97c6-11f1-8ce9-19309dfc597c")
        .unwrap()
        .product(Z2FusionRule);
    assert_eq!(rule.braiding_style(), BraidingStyleKind::NoBraiding);
    let message = panic::catch_unwind(AssertUnwindSafe(|| {
        rule.r_symbol_scalar(s(0), s(0), s(0));
    }))
    .unwrap_err();
    assert_eq!(
        message.downcast_ref::<String>().map(String::as_str),
        Some("r_symbol precondition violated: the category has no braiding")
    );
}
