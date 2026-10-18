use super::*;

#[test]
fn prepared_tensorcontract_fusion_route_replays_like_the_eager_entry() {
    let rule = Z2FusionRule;
    let leg = || SectorLeg::new([(SectorId::new(0), 1), (SectorId::new(1), 1)], false);
    let fusion_space = || {
        DynamicFusionMapSpace::from_final_homspace(
            &rule,
            FusionTreeHomSpace::new(
                FusionProductSpace::new([leg()]),
                FusionProductSpace::new([leg()]),
            ),
        )
        .unwrap()
    };
    // What: a route planned once replays to the eager entry's values.
    let space = fusion_space();
    let lhs = TestTensor::from_vec_with_fusion_space(vec![2.0_f64, 3.0], space.clone()).unwrap();
    let rhs = TestTensor::from_vec_with_fusion_space(vec![5.0_f64, 7.0], space.clone()).unwrap();
    let mut dst_facade =
        TestTensor::from_vec_with_fusion_space(vec![10.0_f64, 20.0], space.clone()).unwrap();
    let mut dst_prepared =
        TestTensor::from_vec_with_fusion_space(vec![10.0_f64, 20.0], space.clone()).unwrap();
    let axes = TensorContractSpec::with_default_output_order(&[1], &[0]);
    let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();

    let prepared =
        fixture_prepare_contract(&mut context, &rule, &dst_prepared, &lhs, &rhs, axes).unwrap();
    for _ in 0..2 {
        context_contract_into(
            &mut context,
            &rule,
            &mut dst_facade,
            &lhs,
            &rhs,
            axes,
            2.0,
            3.0,
        )
        .unwrap();
        fixture_execute_prepared(
            &mut context,
            &prepared,
            &mut dst_prepared,
            &lhs,
            &rhs,
            2.0,
            3.0,
        )
        .unwrap();
    }
    assert_eq!(dst_prepared.data(), dst_facade.data());
}
