//! `LieGroupFusionRule::rule_identity` returns the identity stored at construction:
//! no allocation per call.

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

#[test]
fn large_rank_identity_does_not_materialize_the_lattice() {
    let _serial = crate::counting_alloc::serial();
    use fusion_symbols::lie::GroupId;
    // The identity hashes the sparse canonical GroupId; a dense A_10000
    // character lattice would hold 10^8 entries.
    let rank = 10_000;
    let (rule, allocs) = counting_alloc::measure(|| {
        LieGroupFusionRule::new(GroupId::su(rank + 1).unwrap()).unwrap()
    });
    // At most O(rank); the sparse GroupId and value identity are rank-independent.
    let bound = 4 * rank * size_of::<i64>() + 16 * 1024;
    assert!(allocs.bytes <= bound as u64, "{allocs:?} > {bound}");
    assert_eq!(rule.group_id().semisimple_rank(), rank);
}
