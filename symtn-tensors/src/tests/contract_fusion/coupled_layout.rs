use super::*;

#[test]
fn coupled_layout_complex_scalar_contraction_matches_closed_form() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rule = Z2FusionRule;
    let space = DynamicFusionMapSpace::from_final_homspace(
        &rule,
        FusionTreeHomSpace::from_sector_ids([], []),
    )
    .unwrap();
    let lhs_value = Complex64::new(2.0, 1.0);
    let rhs_value = Complex64::new(3.0, -2.0);
    let initial = Complex64::new(5.0, 0.5);
    let alpha = Complex64::new(0.75, -0.25);
    let beta = Complex64::new(-0.5, 0.125);
    let lhs = TestTensor::<Complex64>::from_vec_with_fusion_space(vec![lhs_value], space.clone())
        .unwrap();
    let rhs = TestTensor::<Complex64>::from_vec_with_fusion_space(vec![rhs_value], space.clone())
        .unwrap();
    let mut dst =
        TestTensor::<Complex64>::from_vec_with_fusion_space(vec![initial], space).unwrap();
    fusion_contract_into(
        engine,
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        TensorContractSpec::with_default_output_order(&[], &[]),
        alpha,
        beta,
    )
    .unwrap();

    // What: scalar direct replay matches an implementation-independent
    // closed form rather than another compiled contraction route.
    assert_eq!(
        dst.data(),
        &[alpha * lhs_value * rhs_value + beta * initial]
    );
}

#[test]
fn coupled_layout_compose_uses_direct_gemm_groups() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rule = Z2FusionRule;
    let degeneracy = 2usize;
    let leg = || {
        SectorLeg::new(
            [
                (SectorId::new(0), degeneracy),
                (SectorId::new(1), degeneracy),
            ],
            false,
        )
    };
    let homspace = || {
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg()]),
            FusionProductSpace::new([leg(), leg()]),
        )
    };
    let space =
        |hom: FusionTreeHomSpace| DynamicFusionMapSpace::from_final_homspace(&rule, hom).unwrap();
    let lhs_space = space(homspace());
    let lhs_len = lhs_space.required_len().unwrap();
    let lhs = TestTensor::<f64>::from_vec_with_fusion_space(
        (0..lhs_len).map(|i| i as f64 * 0.5).collect(),
        lhs_space,
    )
    .unwrap();
    let rhs_space = space(homspace());
    let rhs = TestTensor::<f64>::from_vec_with_fusion_space(
        (0..lhs_len).map(|i| 1.0 - i as f64 * 0.25).collect(),
        rhs_space,
    )
    .unwrap();
    let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        lhs.fusion_space().unwrap().homspace(),
        rhs.fusion_space().unwrap().homspace(),
        &[2, 3],
        &[0, 1],
        &[0, 1, 2, 3],
        2,
    )
    .unwrap();
    let dst_space = space(dst_hom);
    let dst_len = dst_space.required_len().unwrap();
    let dst = TestTensor::<f64>::from_vec_with_fusion_space(vec![0.0; dst_len], dst_space).unwrap();

    let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
    let dst_dyn = dst.fusion_space().unwrap().as_ref().clone();
    let lhs_dyn = lhs.fusion_space().unwrap().as_ref().clone();
    let rhs_dyn = rhs.fusion_space().unwrap().as_ref().clone();
    let resolution = context
        .plan_tensorcontract_fusion_dyn_raw(
            engine,
            &rule,
            &dst_dyn,
            &lhs_dyn,
            &rhs_dyn,
            TensorContractSpec::new(&[2, 3], &[0, 1], OutputAxisOrder::from_axes(&[0, 1, 2, 3])),
        )
        .unwrap();

    // What: coupled-layout compose GEMMs directly into destination blocks
    // (no pack or scatter).
    assert!(resolution.direct_core().unwrap().0.is_fully_direct());
}
