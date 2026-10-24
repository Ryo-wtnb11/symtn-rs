//! The abelian and CU(1) symbols decode their `SectorId`s into fusion-symbols
//! labels on every call (#2278); that decode, and the symbol itself, must not
//! allocate, on admissible and inadmissible labels alike.

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

use symtn_sectors::{
    CU1FusionRule, CU1Irrep, FermionParityFusionRule, MultiplicityFreeRigidSymbols, SectorId,
    U1FusionRule, U1Irrep, Z2FusionRule, Z2Irrep, ZNFusionRule,
};

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

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
    let parity = [Z2Irrep::EVEN.sector_id(), Z2Irrep::ODD.sector_id()];
    assert_no_allocation(&Z2FusionRule, &parity);
    assert_no_allocation(&FermionParityFusionRule, &parity);
    let zn = ZNFusionRule::new(3).unwrap();
    let zn_ids: Vec<SectorId> = (0..3).map(|q| zn.irrep(q).into()).collect();
    assert_no_allocation(&zn, &zn_ids);
    let u1: Vec<SectorId> = (-1..=1).map(|q| U1Irrep::new(q).sector_id()).collect();
    assert_no_allocation(&U1FusionRule, &u1);
    let cu1: Vec<SectorId> = [CU1Irrep::VACUUM, CU1Irrep::PSEUDOSCALAR]
        .into_iter()
        .chain((1..=3).map(CU1Irrep::from_twice_charge))
        .map(Into::into)
        .collect();
    assert_no_allocation(&CU1FusionRule, &cu1);
}
