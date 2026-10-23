//! `sector_order_key`: TensorKit `findindex - 1` positions per factor.

use symtn_sectors::{
    product_sector, CU1FusionRule, FermionParityFusionRule, FusionRule, Fz2SectorLayout,
    PackedProductCodec, ProductFusionRule, SU2FusionRule, SU2Irrep, SectorCodec, SectorOrderKey,
    TableRule, U1FusionRule, U1Irrep, U1SectorLayout, Z2FusionRule, Z2Irrep, ZNFusionRule,
};
#[cfg(feature = "cgc-gen")]
use symtn_sectors::{CheckedGenericFusion, SUNFusionRule};

use crate::s;

fn positions(key: SectorOrderKey) -> Vec<u64> {
    key.positions().to_vec()
}

#[test]
fn u1_positions_leave_room_for_half_integer_charges() {
    // TensorKit position of integer q: 4q - 1 for q > 0, 4|q| otherwise.
    for (charge, position) in [(0, 0), (1, 3), (-1, 4), (2, 7), (-2, 8), (5, 19)] {
        let sector = U1Irrep::new(charge).sector_id();
        assert_eq!(positions(U1FusionRule.sector_order_key(sector)), [position]);
    }
    let mut charges = [-2, -1, 0, 1, 2];
    charges.sort_by_key(|&q| U1FusionRule.sector_order_key(U1Irrep::new(q).sector_id()));
    assert_eq!(charges, [0, 1, -1, 2, -2]);
}

#[test]
fn single_factor_rules_order_by_id() {
    let z5 = ZNFusionRule::new(5).unwrap();
    let fib = TableRule::fibonacci();
    for id in [0, 1] {
        assert_eq!(positions(Z2FusionRule.sector_order_key(s(id))), [id as u64]);
        assert_eq!(
            positions(FermionParityFusionRule.sector_order_key(s(id))),
            [id as u64]
        );
        assert_eq!(positions(fib.sector_order_key(s(id))), [id as u64]);
    }
    for id in [0, 3, 4] {
        assert_eq!(positions(z5.sector_order_key(s(id))), [id as u64]);
    }
    // CU(1): 0+, 0-, 1/2, 1, ... is TensorKit's `values(CU1Irrep)` order.
    for id in [0, 1, 2, 7] {
        assert_eq!(
            positions(CU1FusionRule.sector_order_key(s(id))),
            [id as u64]
        );
    }
    for id in [0, 1, 254] {
        assert_eq!(
            positions(SU2FusionRule.sector_order_key(s(id))),
            [id as u64]
        );
    }
}

#[cfg(feature = "cgc-gen")]
#[test]
fn sun_orders_by_encoded_id() {
    let su3 = SUNFusionRule::new(3).unwrap();
    for id in [0, 1, 2, 4, 9] {
        assert_eq!(
            positions(CheckedGenericFusion::sector_order_key(&su3, s(id))),
            [id as u64]
        );
    }
}

#[test]
fn product_keys_concatenate_factors_and_order_by_degree_then_lex() {
    let cantor =
        symtn_sectors::ProductFusionRule::<_, _>::try_new(U1FusionRule, Z2FusionRule).unwrap();
    let packed = ProductFusionRule::<
        U1FusionRule,
        Z2FusionRule,
        PackedProductCodec<U1SectorLayout, Fz2SectorLayout>,
    >::try_new(U1FusionRule, Z2FusionRule)
    .unwrap();
    let labels = [
        (-1, Z2Irrep::EVEN),
        (1, Z2Irrep::ODD),
        (1, Z2Irrep::EVEN),
        (0, Z2Irrep::ODD),
        (0, Z2Irrep::EVEN),
    ];
    let expected_keys = [[4, 0], [3, 1], [3, 0], [0, 1], [0, 0]];
    for ((charge, parity), key) in labels.into_iter().zip(expected_keys) {
        let label = product_sector(U1Irrep::new(charge), parity);
        for sector_key in [
            cantor.sector_order_key(cantor.encode_sector(&label).unwrap()),
            packed.sector_order_key(packed.encode_sector(&label).unwrap()),
        ] {
            assert_eq!(positions(sector_key), key);
        }
    }
    let mut sorted = labels;
    sorted.sort_by_key(|&(charge, parity)| {
        cantor.sector_order_key(
            cantor
                .encode_sector(&product_sector(U1Irrep::new(charge), parity))
                .unwrap(),
        )
    });
    // Degree 4 ties break lexicographically: [3, 1] < [4, 0].
    assert_eq!(
        sorted,
        [
            (0, Z2Irrep::EVEN),
            (0, Z2Irrep::ODD),
            (1, Z2Irrep::EVEN),
            (1, Z2Irrep::ODD),
            (-1, Z2Irrep::EVEN),
        ]
    );

    let nested = symtn_sectors::ProductFusionRule::<_, _>::try_new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(U1FusionRule, Z2FusionRule).unwrap(),
        SU2FusionRule,
    )
    .unwrap();
    let label = product_sector(
        product_sector(U1Irrep::new(-1), Z2Irrep::ODD),
        SU2Irrep::from_twice_spin(3),
    );
    assert_eq!(
        positions(nested.sector_order_key(nested.encode_sector(&label).unwrap())),
        [4, 1, 3]
    );
}
