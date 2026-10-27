//! `sector_order_key`: TensorKit `findindex - 1` positions per factor.

use symtn_sectors::{
    product_sector, CU1FusionRule, FermionParityFusionRule, FusionRule, Fz2SectorLayout,
    PackedProductCodec, ProductFusionRule, SU2FusionRule, SU2Irrep, SectorCodec, SectorOrderKey,
    TableRule, U1FusionRule, U1Irrep, U1SectorLayout, Z2FusionRule, Z2Irrep, ZNFusionRule,
};

use super::s;

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
mod lie {
    use super::{positions, s};
    use fusion_symbols::lie::{GroupId, RootSystem};
    use symtn_sectors::{
        CheckedGenericFusion, LieGroupFusionRule, ProductFusionRule, U1FusionRule, U1Irrep,
    };

    fn rule(group: GroupId) -> LieGroupFusionRule {
        LieGroupFusionRule::new(group).unwrap()
    }

    fn key(rule: &LieGroupFusionRule, dynkin: &[i64], charges: &[i64]) -> Vec<u64> {
        positions(CheckedGenericFusion::sector_order_key(
            rule,
            rule.encode_irrep(dynkin, charges).unwrap(),
        ))
    }

    /// Asserts each label's hand key, then that sorting the labels (given in
    /// a scrambled order) by key yields `sorted`.
    fn assert_order(
        rule: &LieGroupFusionRule,
        cases: &[(&[i64], &[i64], &[u64])],
        sorted: &[(&[i64], &[i64])],
    ) {
        for &(dynkin, charges, expected) in cases {
            assert_eq!(
                key(rule, dynkin, charges),
                expected,
                "{dynkin:?} {charges:?}"
            );
        }
        let mut labels: Vec<_> = cases.iter().rev().map(|&(d, q, _)| (d, q)).collect();
        labels.sort_by_key(|&(d, q)| {
            CheckedGenericFusion::sector_order_key(rule, rule.encode_irrep(d, q).unwrap())
        });
        assert_eq!(labels, sorted);
    }

    #[test]
    fn single_simple_factor_keys_are_their_ids() {
        // SU(2): Dynkin label p = 2j = TensorKit `findindex - 1`.
        // SU(3): SUNRepresentations `findindex` is the Manhattan rank of the
        // Dynkin labels, i.e. the graded-lexicographic id.
        // B(2) has no TensorKit sector; its key is still its id.
        for group in [
            GroupId::su(2).unwrap(),
            GroupId::su(3).unwrap(),
            GroupId::spin(5).unwrap(),
        ] {
            let rule = rule(group);
            for id in [0, 1, 2, 4, 9] {
                assert_eq!(
                    positions(CheckedGenericFusion::sector_order_key(&rule, s(id))),
                    [id as u64]
                );
            }
        }
    }

    #[test]
    fn su2_su3_orders_by_factor_index_sum_then_lex() {
        // Key [p, r(b1, b2)]: SU(2) Dynkin p; SU(3) graded-lex rank r of
        // (0,0)=0, (0,1)=1, (1,0)=2, (0,2)=3, (1,1)=4.
        // Degrees: [0,0]=0; [0,1],[1,0]=1; [0,2],[1,1],[2,0]=2; [3,0]=3;
        // [0,4]=4, so the SU(3) octet (0;1,1) follows (3;0,0). The flattened
        // id would put it first: its coordinate grade is 2 < 3.
        let rule = rule(GroupId::cover(&[RootSystem::A(1), RootSystem::A(2)], 0).unwrap());
        assert_order(
            &rule,
            &[
                (&[0, 0, 0], &[], &[0, 0]),
                (&[1, 0, 0], &[], &[1, 0]),
                (&[0, 0, 1], &[], &[0, 1]),
                (&[0, 1, 0], &[], &[0, 2]),
                (&[2, 0, 0], &[], &[2, 0]),
                (&[1, 0, 1], &[], &[1, 1]),
                (&[0, 1, 1], &[], &[0, 4]),
                (&[3, 0, 0], &[], &[3, 0]),
            ],
            &[
                (&[0, 0, 0], &[]),
                (&[0, 0, 1], &[]),
                (&[1, 0, 0], &[]),
                (&[0, 1, 0], &[]),
                (&[1, 0, 1], &[]),
                (&[2, 0, 0], &[]),
                (&[3, 0, 0], &[]),
                (&[0, 1, 1], &[]),
            ],
        );
    }

    #[test]
    fn su2_su3_keys_equal_the_product_rule_keys() {
        let su2 = rule(GroupId::su(2).unwrap());
        let su3 = rule(GroupId::su(3).unwrap());
        let product = ProductFusionRule::<_, _>::try_new(su2.clone(), su3.clone()).unwrap();
        let lie = rule(GroupId::cover(&[RootSystem::A(1), RootSystem::A(2)], 0).unwrap());
        for p in 0..4 {
            for b in [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [0, 3]] {
                let id = product.encode_component_ids(
                    su2.encode_irrep(&[p], &[]).unwrap(),
                    su3.encode_irrep(&b, &[]).unwrap(),
                );
                assert_eq!(
                    positions(CheckedGenericFusion::sector_order_key(&product, id)),
                    key(&lie, &[p, b[0], b[1]], &[])
                );
            }
        }
    }

    #[test]
    fn spin5_su2_uses_the_b2_graded_lex_rank() {
        // B(2) has no TensorKit sector type: its position is its own
        // graded-lex Dynkin rank r: (0,0)=0, (0,1)=1, (1,0)=2, (0,2)=3,
        // (1,1)=4. Key [r(b1, b2), a]. Degrees: [0,0]=0; [0,1],[1,0]=1;
        // [0,2],[1,1],[2,0]=2; [0,3]=3; [4,0]=4. The flattened id would put
        // (1,1;0) (coordinate grade 2) before (0,0;3) (grade 3).
        let rule = rule(GroupId::cover(&[RootSystem::B(2), RootSystem::A(1)], 0).unwrap());
        assert_order(
            &rule,
            &[
                (&[0, 0, 0], &[], &[0, 0]),
                (&[0, 0, 1], &[], &[0, 1]),
                (&[0, 1, 0], &[], &[1, 0]),
                (&[1, 0, 0], &[], &[2, 0]),
                (&[0, 0, 2], &[], &[0, 2]),
                (&[0, 1, 1], &[], &[1, 1]),
                (&[1, 1, 0], &[], &[4, 0]),
                (&[0, 0, 3], &[], &[0, 3]),
            ],
            &[
                (&[0, 0, 0], &[]),
                (&[0, 0, 1], &[]),
                (&[0, 1, 0], &[]),
                (&[0, 0, 2], &[]),
                (&[0, 1, 1], &[]),
                (&[1, 0, 0], &[]),
                (&[0, 0, 3], &[]),
                (&[1, 1, 0], &[]),
            ],
        );
    }

    #[test]
    fn su2_u1_orders_as_su2_times_u1irrep() {
        // Key [p, u(q)], u(1)=3, u(-1)=4. Degrees: (0;0)=0, (1;0)=1,
        // (2;0)=2, (0;1)=3, (3;0)=3, (0;-1)=4, (1;1)=4, (1;-1)=5; ties break
        // lexicographically ([0,3] < [3,0], [0,4] < [1,3]). A zigzag charge
        // index would instead tie (0;1) with (1;0) at degree 1.
        let rule = rule(GroupId::cover(&[RootSystem::A(1)], 1).unwrap());
        assert_order(
            &rule,
            &[
                (&[0], &[0], &[0, 0]),
                (&[1], &[0], &[1, 0]),
                (&[2], &[0], &[2, 0]),
                (&[0], &[1], &[0, 3]),
                (&[0], &[-1], &[0, 4]),
                (&[1], &[1], &[1, 3]),
                (&[3], &[0], &[3, 0]),
                (&[1], &[-1], &[1, 4]),
            ],
            &[
                (&[0], &[0]),
                (&[1], &[0]),
                (&[2], &[0]),
                (&[0], &[1]),
                (&[3], &[0]),
                (&[0], &[-1]),
                (&[1], &[1]),
                (&[1], &[-1]),
            ],
        );
        let su2 = self::rule(GroupId::su(2).unwrap());
        let product = ProductFusionRule::<_, _>::try_new(su2.clone(), U1FusionRule).unwrap();
        for (p, q) in [(0, 0), (1, -1), (2, 1), (3, -2)] {
            let id = product.encode_component_ids(
                su2.encode_irrep(&[p], &[]).unwrap(),
                U1Irrep::new(q).sector_id(),
            );
            assert_eq!(
                positions(CheckedGenericFusion::sector_order_key(&product, id)),
                key(&rule, &[p], &[q.into()])
            );
        }
    }

    #[test]
    fn u2_orders_su2_label_then_u1_findindex() {
        // Key [p, u(q)] with TensorKit `U1Irrep` `findindex - 1`
        // u(q) = 4q - 1 (q > 0), 4|q| (q <= 0): u(1)=3, u(-1)=4, u(2)=7,
        // u(-2)=8. U(2) admits q ≡ p (mod 2). Degrees: (0;0)=0, (2;0)=2,
        // (1;1)=4, (1;-1)=5, (3;1)=6, (0;2)=7, (0;-2)=8, (2;2)=9.
        let rule = rule(GroupId::u(2).unwrap());
        assert_order(
            &rule,
            &[
                (&[0], &[0], &[0, 0]),
                (&[1], &[1], &[1, 3]),
                (&[1], &[-1], &[1, 4]),
                (&[0], &[2], &[0, 7]),
                (&[0], &[-2], &[0, 8]),
                (&[2], &[0], &[2, 0]),
                (&[2], &[2], &[2, 7]),
                (&[3], &[1], &[3, 3]),
            ],
            &[
                (&[0], &[0]),
                (&[2], &[0]),
                (&[1], &[1]),
                (&[1], &[-1]),
                (&[3], &[1]),
                (&[0], &[2]),
                (&[0], &[-2]),
                (&[2], &[2]),
            ],
        );
    }

    #[test]
    fn spin4_keys_are_two_su2_positions() {
        // D(2) = SU(2)⊠SU(2): key [a1, a2], degree a1 + a2 then lex.
        let rule = rule(GroupId::spin(4).unwrap());
        assert_order(
            &rule,
            &[
                (&[0, 0], &[], &[0, 0]),
                (&[1, 0], &[], &[1, 0]),
                (&[0, 1], &[], &[0, 1]),
                (&[1, 1], &[], &[1, 1]),
                (&[2, 0], &[], &[2, 0]),
                (&[0, 2], &[], &[0, 2]),
                (&[2, 1], &[], &[2, 1]),
                (&[0, 3], &[], &[0, 3]),
            ],
            &[
                (&[0, 0], &[]),
                (&[0, 1], &[]),
                (&[1, 0], &[]),
                (&[0, 2], &[]),
                (&[1, 1], &[]),
                (&[2, 0], &[]),
                (&[0, 3], &[]),
                (&[2, 1], &[]),
            ],
        );
    }

    #[test]
    fn torus2_orders_as_u1_u1() {
        // Key [u(q1), u(q2)] with u as above. Degrees: (0,0)=0, (0,1)=3,
        // (1,0)=3, (0,-1)=4, (-1,0)=4, (1,1)=6, (1,-1)=7, (2,0)=7,
        // (-1,-1)=8; ties break lexicographically ([3,4] < [7,0]).
        let rule = rule(GroupId::torus(2).unwrap());
        assert_order(
            &rule,
            &[
                (&[], &[0, 0], &[0, 0]),
                (&[], &[1, 0], &[3, 0]),
                (&[], &[0, 1], &[0, 3]),
                (&[], &[-1, 0], &[4, 0]),
                (&[], &[0, -1], &[0, 4]),
                (&[], &[1, 1], &[3, 3]),
                (&[], &[1, -1], &[3, 4]),
                (&[], &[2, 0], &[7, 0]),
                (&[], &[-1, -1], &[4, 4]),
            ],
            &[
                (&[], &[0, 0]),
                (&[], &[0, 1]),
                (&[], &[1, 0]),
                (&[], &[0, -1]),
                (&[], &[-1, 0]),
                (&[], &[1, 1]),
                (&[], &[1, -1]),
                (&[], &[2, 0]),
                (&[], &[-1, -1]),
            ],
        );
        let u1u1 = ProductFusionRule::<_, _>::try_new(U1FusionRule, U1FusionRule).unwrap();
        for (q1, q2) in [(0, 0), (1, -1), (-2, 1), (2, 2)] {
            let id = u1u1
                .encode_component_ids(U1Irrep::new(q1).sector_id(), U1Irrep::new(q2).sector_id());
            assert_eq!(
                positions(symtn_sectors::FusionRule::sector_order_key(&u1u1, id)),
                key(&rule, &[], &[q1.into(), q2.into()])
            );
        }
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
