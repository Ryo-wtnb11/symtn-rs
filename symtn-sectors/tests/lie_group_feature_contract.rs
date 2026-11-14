//! `LieGroupFusionRule` exists in every build; `cgc-gen` decides which groups
//! fusion-symbols can execute.

use fusion_symbols::lie::GroupId;
use symtn_sectors::{CheckedGenericAdmissionMode, CheckedGenericFusion, LieGroupFusionRule};

/// A rule that serves `dynkin`/`charges` round-trips the label and finds the
/// vacuum in the label fused with its dual.
fn assert_serves(rule: &LieGroupFusionRule, dynkin: &[i64], charges: &[i64]) {
    let a = rule.encode_irrep_with_charges(dynkin, charges).unwrap();
    assert_eq!(
        rule.decode_irrep(a).unwrap(),
        (dynkin.to_vec(), charges.to_vec())
    );
    let channels = rule
        .try_fusion_channels(a, rule.try_dual(a).unwrap())
        .unwrap();
    assert!(channels.contains(&rule.vacuum()));
}

#[test]
fn featureless_groups_build_and_fuse_in_every_build() {
    for (group, dynkin, charges) in [
        (GroupId::su(2), &[1][..], &[][..]),
        (GroupId::sp(2), &[1], &[]),
        (GroupId::so(3), &[2], &[]),
        (GroupId::spin(4), &[1, 1], &[]),
        (GroupId::u(2), &[1], &[1]),
        (GroupId::torus(2), &[], &[1, -2]),
    ] {
        assert_serves(
            &LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(group.unwrap()).unwrap(),
            dynkin,
            charges,
        );
    }
}

#[test]
fn rank_two_simple_factors_follow_the_cgc_gen_feature() {
    for (group, dynkin, charges) in [
        (GroupId::su(3), &[1, 0][..], &[][..]),
        (GroupId::u(3), &[1, 0], &[1]),
        (GroupId::sp(4), &[1, 0], &[]),
    ] {
        let rule = LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(group.unwrap());
        #[cfg(not(feature = "cgc-gen"))]
        {
            let _ = (dynkin, charges);
            assert!(matches!(
                rule,
                Err(symtn_sectors::LieGroupFusionRuleError::Provider(
                    fusion_symbols::lie::LieError::Unsupported {
                        requires: "lie-generated"
                    }
                ))
            ));
        }
        #[cfg(feature = "cgc-gen")]
        assert_serves(&rule.unwrap(), dynkin, charges);
    }
}
