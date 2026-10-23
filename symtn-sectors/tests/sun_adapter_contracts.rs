//! Absent SUN vertices do not require representable derived data.

#![cfg(feature = "cgc-gen")]

use symtn_sectors::{CheckedGenericFusion, CheckedGenericRigidSymbols, SUNFusionRule, SectorId};

#[test]
fn absent_sun_moves_do_not_require_an_encoded_dual_or_finite_dimension() {
    if usize::BITS != 64 {
        return;
    }
    let su3 = SUNFusionRule::new(3).unwrap();
    // This id's Dynkin reversal lies beyond usize::MAX in graded-lex order.
    let unencodable_dual = SectorId::new(usize::MAX);
    assert_eq!(
        su3.decode_dynkin(unencodable_dual).unwrap(),
        [2_746_052_115, 3_327_948_884]
    );

    let su78 = SUNFusionRule::new(78).unwrap();
    let mut labels = vec![0; 77];
    labels[36] = 19;
    let overflowing_dimension = su78.encode_dynkin(&labels).unwrap();
    for (rule, sector) in [(&su3, unencodable_dual), (&su78, overflowing_dimension)] {
        let unit = rule.vacuum();
        // For nontrivial a, N(a,1;1) = N(1,a;1) = 0 without dimensions or duals.
        for block in [
            rule.try_a_symbol_generic(sector, unit, unit).unwrap(),
            rule.try_b_symbol_generic(unit, sector, unit).unwrap(),
        ] {
            assert_eq!(block.shape(), (0, 0));
            assert!(block.data().is_empty());
        }
    }
}
