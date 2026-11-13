//! The abelian and CU(1) symbols decode their `SectorId`s into fusion-symbols
//! labels on every call; that decode, and the symbol itself, must not
//! allocate, on admissible and inadmissible labels alike.

use crate::counting_alloc;

use symtn_sectors::{
    CU1FusionRule, CU1Irrep, FermionParityFusionRule, FermionParityIrrep,
    MultiplicityFreeRigidSymbols, SectorCodec, SectorId, U1FusionRule, U1Irrep, ZNFusionRule,
    ZNIrrep,
};

/// Every symbol over every tuple of `ids`, summed so nothing is optimized out.
fn sweep<R: MultiplicityFreeRigidSymbols<Scalar = f64>>(rule: &R, ids: &[SectorId]) -> f64 {
    let mut sum = 0.0;
    for &a in ids {
        sum += rule.dim_scalar(a)
            + rule.inv_dim_scalar(a)
            + rule.sqrt_dim_scalar(a)
            + rule.inv_sqrt_dim_scalar(a)
            + rule.twist_scalar(a)
            + rule.frobenius_schur_phase_scalar(a);
        for &b in ids {
            for &c in ids {
                sum += rule.r_symbol_scalar(a, b, c);
                sum += rule.a_symbol_scalar(a, b, c) + rule.b_symbol_scalar(a, b, c);
                for &d in ids {
                    for &e in ids {
                        for &f in ids {
                            sum += rule.f_symbol_scalar(a, b, c, d, e, f);
                        }
                    }
                }
            }
        }
    }
    sum
}

fn assert_no_allocation<R: MultiplicityFreeRigidSymbols<Scalar = f64>>(rule: &R, ids: &[SectorId]) {
    let _ = sweep(rule, ids);
    let (sum, allocs) = counting_alloc::measure(|| sweep(rule, ids));
    assert!(sum.is_finite());
    assert_eq!(allocs.calls, 0, "{allocs:?}");
}

#[test]
fn symbols_do_not_allocate() {
    let _serial = crate::counting_alloc::serial();
    let parity = [FermionParityIrrep::EVEN, FermionParityIrrep::ODD].map(SectorId::from);
    assert_no_allocation(&FermionParityFusionRule::new(), &parity);
    let zn = ZNFusionRule::new(3).unwrap();
    let zn_ids: Vec<SectorId> = (0..3)
        .map(|q| SectorId::from(ZNIrrep::new(q, 3).unwrap()))
        .collect();
    assert_no_allocation(&zn, &zn_ids);
    let u1: Vec<SectorId> = (-1..=1).map(|q| U1Irrep::new(q).sector_id()).collect();
    assert_no_allocation(&U1FusionRule, &u1);
    let cu1_rule = CU1FusionRule::new();
    let cu1: Vec<SectorId> = [CU1Irrep::VACUUM, CU1Irrep::PSEUDOSCALAR]
        .into_iter()
        .chain((1..=3).map(CU1Irrep::from_twice_charge))
        .map(|label| cu1_rule.encode_sector(&label).unwrap())
        .collect();
    assert_no_allocation(&cu1_rule, &cu1);
}
