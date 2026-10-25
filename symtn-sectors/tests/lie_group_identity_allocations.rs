//! `LieGroupFusionRule::rule_identity` returns the identity stored at construction:
//! no allocation per call (#1750).

#![cfg(feature = "cgc-gen")]

use crate::counting_alloc;

use symtn_sectors::{CheckedGenericFusion, LieGroupFusionRule};

#[test]
fn rule_identity_does_not_allocate() {
    let _serial = crate::counting_alloc::serial();
    let rule = LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap();
    let (identity, allocs) = counting_alloc::measure(|| rule.rule_identity());
    assert_eq!(allocs.calls, 0);
    assert_eq!(identity, rule.rule_identity());
}
