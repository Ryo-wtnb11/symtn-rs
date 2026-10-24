//! `LieGroupFusionRule::rule_identity` returns the identity stored at construction:
//! no allocation per call (#1750).

#![cfg(feature = "cgc-gen")]

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

use symtn_sectors::{CheckedGenericFusion, LieGroupFusionRule};

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

#[test]
fn rule_identity_does_not_allocate() {
    let rule = LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap();
    let (identity, allocs) = counting_alloc::measure(|| rule.rule_identity());
    assert_eq!(allocs.calls, 0);
    assert_eq!(identity, rule.rule_identity());
}
