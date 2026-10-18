use super::*;
use crate::probe::{self, Event};

#[test]
fn tensorcontract_fusion_lowers_lhs_categorical_adjoint_lazily() {
    let rule = Z2FusionRule;
    let leg = || SectorLeg::new([(SectorId::new(0), 1), (SectorId::new(1), 1)], false);
    let space = || {
        DynamicFusionMapSpace::from_final_homspace(
            &rule,
            FusionTreeHomSpace::new(
                FusionProductSpace::new([leg()]),
                FusionProductSpace::new([leg()]),
            ),
        )
        .unwrap()
    };
    let lhs = TestTensor::<Complex64>::from_vec_with_fusion_space(
        vec![Complex64::new(2.0, 1.0), Complex64::new(3.0, -1.0)],
        space(),
    )
    .unwrap();
    let rhs = TestTensor::<Complex64>::from_vec_with_fusion_space(
        vec![Complex64::new(5.0, 2.0), Complex64::new(7.0, -2.0)],
        space(),
    )
    .unwrap();
    let mut dst = TestTensor::<Complex64>::from_vec_with_fusion_space(
        vec![Complex64::new(10.0, 0.0), Complex64::new(20.0, 0.0)],
        space(),
    )
    .unwrap();
    let axes =
        TensorContractSpec::with_default_output_order_and_conjugation(&[1], &[0], true, false);

    fusion_contract_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        axes,
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[Complex64::new(12.0, -1.0), Complex64::new(23.0, 1.0)]
    );

    dst.data_mut()
        .copy_from_slice(&[Complex64::new(10.0, 0.0), Complex64::new(20.0, 0.0)]);
    let mut context = TensorContractFusionExecutionContext::<Complex64, _>::default();
    context_contract_into(
        &mut context,
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        axes,
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();
    assert_eq!(
        dst.data(),
        &[Complex64::new(12.0, -1.0), Complex64::new(23.0, 1.0)]
    );
}

#[test]
fn tensorcontract_fusion_lowers_rhs_categorical_adjoint_lazily() {
    let rule = Z2FusionRule;
    let leg = || SectorLeg::new([(SectorId::new(0), 1), (SectorId::new(1), 1)], false);
    let space = || {
        DynamicFusionMapSpace::from_final_homspace(
            &rule,
            FusionTreeHomSpace::new(
                FusionProductSpace::new([leg()]),
                FusionProductSpace::new([leg()]),
            ),
        )
        .unwrap()
    };
    let lhs = TestTensor::<Complex64>::from_vec_with_fusion_space(
        vec![Complex64::new(2.0, 1.0), Complex64::new(3.0, -1.0)],
        space(),
    )
    .unwrap();
    let rhs = TestTensor::<Complex64>::from_vec_with_fusion_space(
        vec![Complex64::new(5.0, 2.0), Complex64::new(7.0, -2.0)],
        space(),
    )
    .unwrap();
    let mut dst = TestTensor::<Complex64>::from_vec_with_fusion_space(
        vec![Complex64::zero(), Complex64::zero()],
        space(),
    )
    .unwrap();

    fusion_contract_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        TensorContractSpec::with_default_output_order_and_conjugation(&[1], &[0], false, true),
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[Complex64::new(12.0, 1.0), Complex64::new(23.0, -1.0)]
    );
}

#[test]
fn tensorcontract_fusion_lowers_both_categorical_adjoint_inputs_lazily() {
    let rule = Z2FusionRule;
    let leg = || SectorLeg::new([(SectorId::new(0), 1), (SectorId::new(1), 1)], false);
    let space = || {
        DynamicFusionMapSpace::from_final_homspace(
            &rule,
            FusionTreeHomSpace::new(
                FusionProductSpace::new([leg()]),
                FusionProductSpace::new([leg()]),
            ),
        )
        .unwrap()
    };
    let lhs_space = space();
    let rhs_space = space();
    let lhs_adjoint_space = lhs_space
        .validate_rule(&rule)
        .and_then(|()| lhs_space.adjoint_view())
        .unwrap();
    let rhs_adjoint_space = rhs_space
        .validate_rule(&rule)
        .and_then(|()| rhs_space.adjoint_view())
        .unwrap();
    let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        lhs_adjoint_space.homspace(),
        rhs_adjoint_space.homspace(),
        &[1],
        &[0],
        &[0, 1],
        1,
    )
    .unwrap();
    let dst_space = DynamicFusionMapSpace::from_final_homspace(&rule, dst_hom).unwrap();
    let lhs = TestTensor::<Complex64>::from_vec_with_fusion_space(
        vec![Complex64::new(2.0, 1.0), Complex64::new(3.0, -1.0)],
        lhs_space,
    )
    .unwrap();
    let rhs = TestTensor::<Complex64>::from_vec_with_fusion_space(
        vec![Complex64::new(5.0, 2.0), Complex64::new(7.0, -2.0)],
        rhs_space,
    )
    .unwrap();
    let mut dst = TestTensor::<Complex64>::from_vec_with_fusion_space(
        vec![Complex64::zero(), Complex64::zero()],
        dst_space,
    )
    .unwrap();
    let axes =
        TensorContractSpec::with_default_output_order_and_conjugation(&[1], &[0], true, true);

    fusion_contract_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        axes,
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[Complex64::new(8.0, -9.0), Complex64::new(19.0, 13.0)]
    );
}

#[test]
fn tensorcontract_fusion_lhs_adjoint_uses_degeneracy_matrix_contract() {
    let rule = Z2FusionRule;
    let lhs_space = z2_matrix_space_with_homspace(z2_matrix_homspace());
    let rhs_space = z2_matrix_space_with_homspace(z2_matrix_homspace());
    let lhs_adjoint_space = lhs_space
        .validate_rule(&rule)
        .and_then(|()| lhs_space.adjoint_view())
        .unwrap();
    let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        lhs_adjoint_space.homspace(),
        rhs_space.homspace(),
        &[1],
        &[0],
        &[0, 1],
        1,
    )
    .unwrap();
    let dst_space = z2_matrix_space_with_homspace(dst_hom);
    let lhs = TestTensor::<Complex64>::from_vec_with_fusion_space(
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(3.0, 2.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(4.0, -1.0),
            Complex64::new(-1.0, 2.0),
            Complex64::new(2.0, -2.0),
            Complex64::new(0.0, 3.0),
            Complex64::new(-3.0, 1.0),
        ],
        lhs_space,
    )
    .unwrap();
    let rhs = TestTensor::<Complex64>::from_vec_with_fusion_space(
        vec![
            Complex64::new(5.0, -2.0),
            Complex64::new(7.0, -4.0),
            Complex64::new(6.0, 1.0),
            Complex64::new(8.0, 2.0),
            Complex64::new(4.0, 1.0),
            Complex64::new(1.0, -3.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(5.0, -1.0),
        ],
        rhs_space,
    )
    .unwrap();
    let mut dst =
        TestTensor::<Complex64>::from_vec_with_fusion_space(vec![Complex64::zero(); 8], dst_space)
            .unwrap();

    fusion_contract_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        // TensorKit 1-based pA=((2,), (1,)), pB=((1,), (2,)).
        TensorContractSpec::with_default_output_order_and_conjugation(&[1], &[0], true, false),
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[
            Complex64::new(16.0, -33.0),
            Complex64::new(44.0, -8.0),
            Complex64::new(35.0, -15.0),
            Complex64::new(41.0, 24.0),
            Complex64::new(6.0, -13.0),
            Complex64::new(-3.0, -4.0),
            Complex64::new(18.0, 10.0),
            Complex64::new(-10.0, 4.0),
        ]
    );
}

#[test]
fn tensorcontract_fusion_fermion_lhs_adjoint_uses_degeneracy_matrix_contract() {
    let rule = FermionParityFusionRule;
    let lhs_space = fermion_parity_matrix_space_with_homspace(fermion_parity_matrix_homspace());
    let rhs_space = fermion_parity_matrix_space_with_homspace(fermion_parity_matrix_homspace());
    let lhs_adjoint_space = lhs_space
        .validate_rule(&rule)
        .and_then(|()| lhs_space.adjoint_view())
        .unwrap();
    let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        lhs_adjoint_space.homspace(),
        rhs_space.homspace(),
        &[1],
        &[0],
        &[0, 1],
        1,
    )
    .unwrap();
    let dst_space = fermion_parity_matrix_space_with_homspace(dst_hom);
    let lhs = TestTensor::<Complex64>::from_vec_with_fusion_space(
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(3.0, 2.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(4.0, -1.0),
            Complex64::new(-1.0, 2.0),
            Complex64::new(2.0, -2.0),
            Complex64::new(0.0, 3.0),
            Complex64::new(-3.0, 1.0),
        ],
        lhs_space,
    )
    .unwrap();
    let rhs = TestTensor::<Complex64>::from_vec_with_fusion_space(
        vec![
            Complex64::new(5.0, -2.0),
            Complex64::new(7.0, -4.0),
            Complex64::new(6.0, 1.0),
            Complex64::new(8.0, 2.0),
            Complex64::new(4.0, 1.0),
            Complex64::new(1.0, -3.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(5.0, -1.0),
        ],
        rhs_space,
    )
    .unwrap();
    let mut dst =
        TestTensor::<Complex64>::from_vec_with_fusion_space(vec![Complex64::zero(); 8], dst_space)
            .unwrap();

    fusion_contract_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        // TensorKit 1-based pA=((2,), (1,)), pB=((1,), (2,)).
        TensorContractSpec::with_default_output_order_and_conjugation(&[1], &[0], true, false),
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[
            Complex64::new(16.0, -33.0),
            Complex64::new(44.0, -8.0),
            Complex64::new(35.0, -15.0),
            Complex64::new(41.0, 24.0),
            Complex64::new(6.0, -13.0),
            Complex64::new(-3.0, -4.0),
            Complex64::new(18.0, 10.0),
            Complex64::new(-10.0, 4.0),
        ]
    );
}

#[test]
fn tensorcontract_fusion_rhs_adjoint_uses_degeneracy_matrix_contract() {
    let rule = Z2FusionRule;
    let lhs_space = z2_matrix_space_with_homspace(z2_matrix_homspace());
    let rhs_space = z2_matrix_space_with_homspace(z2_matrix_homspace());
    let rhs_adjoint_space = rhs_space
        .validate_rule(&rule)
        .and_then(|()| rhs_space.adjoint_view())
        .unwrap();
    let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        lhs_space.homspace(),
        rhs_adjoint_space.homspace(),
        &[1],
        &[0],
        &[0, 1],
        1,
    )
    .unwrap();
    let dst_space = z2_matrix_space_with_homspace(dst_hom);
    let lhs = TestTensor::<Complex64>::from_vec_with_fusion_space(
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(3.0, 2.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(4.0, -1.0),
            Complex64::new(-1.0, 2.0),
            Complex64::new(2.0, -2.0),
            Complex64::new(0.0, 3.0),
            Complex64::new(-3.0, 1.0),
        ],
        lhs_space,
    )
    .unwrap();
    let rhs = TestTensor::<Complex64>::from_vec_with_fusion_space(
        vec![
            Complex64::new(5.0, -2.0),
            Complex64::new(7.0, -4.0),
            Complex64::new(6.0, 1.0),
            Complex64::new(8.0, 2.0),
            Complex64::new(4.0, 1.0),
            Complex64::new(1.0, -3.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(5.0, -1.0),
        ],
        rhs_space,
    )
    .unwrap();
    let mut dst =
        TestTensor::<Complex64>::from_vec_with_fusion_space(vec![Complex64::zero(); 8], dst_space)
            .unwrap();

    fusion_contract_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        TensorContractSpec::with_default_output_order_and_conjugation(&[1], &[0], false, true),
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[
            Complex64::new(14.0, -1.0),
            Complex64::new(34.0, 6.0),
            Complex64::new(17.0, -1.0),
            Complex64::new(43.0, 10.0),
            Complex64::new(4.0, 3.0),
            Complex64::new(14.0, -6.0),
            Complex64::new(-10.0, 14.0),
            Complex64::new(-8.0, 6.0),
        ]
    );
}

#[test]
fn tensorcontract_fusion_fermion_rhs_adjoint_uses_degeneracy_matrix_contract() {
    let rule = FermionParityFusionRule;
    let lhs_space = fermion_parity_matrix_space_with_homspace(fermion_parity_matrix_homspace());
    let rhs_space = fermion_parity_matrix_space_with_homspace(fermion_parity_matrix_homspace());
    let rhs_adjoint_space = rhs_space
        .validate_rule(&rule)
        .and_then(|()| rhs_space.adjoint_view())
        .unwrap();
    let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        lhs_space.homspace(),
        rhs_adjoint_space.homspace(),
        &[1],
        &[0],
        &[0, 1],
        1,
    )
    .unwrap();
    let dst_space = fermion_parity_matrix_space_with_homspace(dst_hom);
    let lhs = TestTensor::<Complex64>::from_vec_with_fusion_space(
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(3.0, 2.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(4.0, -1.0),
            Complex64::new(-1.0, 2.0),
            Complex64::new(2.0, -2.0),
            Complex64::new(0.0, 3.0),
            Complex64::new(-3.0, 1.0),
        ],
        lhs_space,
    )
    .unwrap();
    let rhs = TestTensor::<Complex64>::from_vec_with_fusion_space(
        vec![
            Complex64::new(5.0, -2.0),
            Complex64::new(7.0, -4.0),
            Complex64::new(6.0, 1.0),
            Complex64::new(8.0, 2.0),
            Complex64::new(4.0, 1.0),
            Complex64::new(1.0, -3.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(5.0, -1.0),
        ],
        rhs_space,
    )
    .unwrap();
    let mut dst =
        TestTensor::<Complex64>::from_vec_with_fusion_space(vec![Complex64::zero(); 8], dst_space)
            .unwrap();

    fusion_contract_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        // TensorKit 1-based pA=((1,), (2,)), pB=((2,), (1,)).
        TensorContractSpec::with_default_output_order_and_conjugation(&[1], &[0], false, true),
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[
            Complex64::new(14.0, -1.0),
            Complex64::new(34.0, 6.0),
            Complex64::new(17.0, -1.0),
            Complex64::new(43.0, 10.0),
            Complex64::new(4.0, 3.0),
            Complex64::new(14.0, -6.0),
            Complex64::new(-10.0, 14.0),
            Complex64::new(-8.0, 6.0),
        ]
    );
}

#[test]
fn tensorcontract_fusion_both_adjoint_uses_degeneracy_matrix_contract() {
    let rule = Z2FusionRule;
    let lhs_space = z2_matrix_space_with_homspace(z2_matrix_homspace());
    let rhs_space = z2_matrix_space_with_homspace(z2_matrix_homspace());
    let lhs_adjoint_space = lhs_space
        .validate_rule(&rule)
        .and_then(|()| lhs_space.adjoint_view())
        .unwrap();
    let rhs_adjoint_space = rhs_space
        .validate_rule(&rule)
        .and_then(|()| rhs_space.adjoint_view())
        .unwrap();
    let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        lhs_adjoint_space.homspace(),
        rhs_adjoint_space.homspace(),
        &[1],
        &[0],
        &[0, 1],
        1,
    )
    .unwrap();
    let dst_space = z2_matrix_space_with_homspace(dst_hom);
    let lhs = TestTensor::<Complex64>::from_vec_with_fusion_space(
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(3.0, 2.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(4.0, -1.0),
            Complex64::new(-1.0, 2.0),
            Complex64::new(2.0, -2.0),
            Complex64::new(0.0, 3.0),
            Complex64::new(-3.0, 1.0),
        ],
        lhs_space,
    )
    .unwrap();
    let rhs = TestTensor::<Complex64>::from_vec_with_fusion_space(
        vec![
            Complex64::new(5.0, -2.0),
            Complex64::new(7.0, -4.0),
            Complex64::new(6.0, 1.0),
            Complex64::new(8.0, 2.0),
            Complex64::new(4.0, 1.0),
            Complex64::new(1.0, -3.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(5.0, -1.0),
        ],
        rhs_space,
    )
    .unwrap();
    let mut dst =
        TestTensor::<Complex64>::from_vec_with_fusion_space(vec![Complex64::zero(); 8], dst_space)
            .unwrap();

    fusion_contract_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        TensorContractSpec::with_default_output_order_and_conjugation(&[1], &[0], true, true),
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[
            Complex64::new(23.0, -18.0),
            Complex64::new(33.0, 11.0),
            Complex64::new(31.0, -25.0),
            Complex64::new(44.0, 15.0),
            Complex64::new(-6.0, -15.0),
            Complex64::new(1.0, -4.0),
            Complex64::new(13.0, 7.0),
            Complex64::new(-5.0, -11.0),
        ]
    );
}

#[test]
fn tensorcontract_fusion_fermion_both_adjoint_uses_degeneracy_matrix_contract() {
    let rule = FermionParityFusionRule;
    let lhs_space = fermion_parity_matrix_space_with_homspace(fermion_parity_matrix_homspace());
    let rhs_space = fermion_parity_matrix_space_with_homspace(fermion_parity_matrix_homspace());
    let lhs_adjoint_space = lhs_space
        .validate_rule(&rule)
        .and_then(|()| lhs_space.adjoint_view())
        .unwrap();
    let rhs_adjoint_space = rhs_space
        .validate_rule(&rule)
        .and_then(|()| rhs_space.adjoint_view())
        .unwrap();
    let dst_hom = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        lhs_adjoint_space.homspace(),
        rhs_adjoint_space.homspace(),
        &[1],
        &[0],
        &[0, 1],
        1,
    )
    .unwrap();
    let dst_space = fermion_parity_matrix_space_with_homspace(dst_hom);
    let lhs = TestTensor::<Complex64>::from_vec_with_fusion_space(
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(3.0, 2.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(4.0, -1.0),
            Complex64::new(-1.0, 2.0),
            Complex64::new(2.0, -2.0),
            Complex64::new(0.0, 3.0),
            Complex64::new(-3.0, 1.0),
        ],
        lhs_space,
    )
    .unwrap();
    let rhs = TestTensor::<Complex64>::from_vec_with_fusion_space(
        vec![
            Complex64::new(5.0, -2.0),
            Complex64::new(7.0, -4.0),
            Complex64::new(6.0, 1.0),
            Complex64::new(8.0, 2.0),
            Complex64::new(4.0, 1.0),
            Complex64::new(1.0, -3.0),
            Complex64::new(-2.0, 2.0),
            Complex64::new(5.0, -1.0),
        ],
        rhs_space,
    )
    .unwrap();
    let mut dst =
        TestTensor::<Complex64>::from_vec_with_fusion_space(vec![Complex64::zero(); 8], dst_space)
            .unwrap();

    fusion_contract_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        // TensorKit 1-based pA=((2,), (1,)), pB=((2,), (1,)).
        TensorContractSpec::with_default_output_order_and_conjugation(&[1], &[0], true, true),
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[
            Complex64::new(23.0, -18.0),
            Complex64::new(33.0, 11.0),
            Complex64::new(31.0, -25.0),
            Complex64::new(44.0, 15.0),
            Complex64::new(-6.0, -15.0),
            Complex64::new(1.0, -4.0),
            Complex64::new(13.0, 7.0),
            Complex64::new(-5.0, -11.0),
        ]
    );
}

#[test]
fn tensorproduct_fusion_lowers_lhs_adjoint_through_source_transform() {
    let rule = Z2FusionRule;
    let sector = SectorId::new(0);
    let lhs_space = DynamicFusionMapSpace::from_final_homspace(
        &rule,
        FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(sector, 1)], false)]),
            FusionProductSpace::new(Vec::<SectorLeg>::new()),
        ),
    )
    .unwrap();
    let rhs_space = DynamicFusionMapSpace::from_final_homspace(
        &rule,
        FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(sector, 1)], false)]),
            FusionProductSpace::new(Vec::<SectorLeg>::new()),
        ),
    )
    .unwrap();
    let dst_space = DynamicFusionMapSpace::from_final_homspace(
        &rule,
        FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(sector, 1)], true)]),
            FusionProductSpace::new([SectorLeg::new([(sector, 1)], true)]),
        ),
    )
    .unwrap();
    let lhs: TestTensor<Complex64> =
        TestTensor::from_vec_with_fusion_space(vec![Complex64::new(2.0, 1.0)], lhs_space).unwrap();
    let rhs: TestTensor<Complex64> =
        TestTensor::from_vec_with_fusion_space(vec![Complex64::new(3.0, -1.0)], rhs_space).unwrap();
    let mut dst: TestTensor<Complex64> =
        TestTensor::from_vec_with_fusion_space(vec![Complex64::new(0.0, 0.0)], dst_space).unwrap();

    fusion_contract_into(
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        TensorContractSpec::new_with_conjugation(
            &[],
            &[],
            OutputAxisOrder::identity(),
            true,
            false,
        ),
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();

    assert_eq!(dst.data(), &[Complex64::new(5.0, -5.0)]);

    dst.data_mut().copy_from_slice(&[Complex64::new(0.0, 0.0)]);
    let mut context = TensorContractFusionExecutionContext::<Complex64, _>::default();
    context_contract_into(
        &mut context,
        &rule,
        &mut dst,
        &lhs,
        &rhs,
        TensorContractSpec::new_with_conjugation(
            &[],
            &[],
            OutputAxisOrder::identity(),
            true,
            false,
        ),
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();
    assert_eq!(dst.data(), &[Complex64::new(5.0, -5.0)]);
}

// A lhs-conjugate (categorical-adjoint) contraction over a NON-self-dual (U(1))
// symmetry must equal the eager conjugate-transpose composed plainly. The other
// conjugate recipe tests are all self-dual (Z2 / fermion parity), where a charge
// equals its dual, so a sector-dualization mislabel is invisible; here charge +1
// and -1 are distinct duals. This rank-(1,1) compose is core form, so the
// parent-storage matrix is consumed through the op-bearing Core batch.
#[test]
fn tensorcontract_fusion_u1_lhs_adjoint_matches_eager_conjugate_transpose() {
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    use num_complex::Complex64;
    let rule = U1FusionRule;
    let c = |q: i32| U1Irrep::new(q).sector_id();
    // Non-self-dual charges: +1 and -1 are distinct duals of each other.
    let leg = || SectorLeg::new([(c(-1), 2), (c(1), 2)], false);
    let hom = || {
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg()]),
            FusionProductSpace::new([leg()]),
        )
    };
    let fusion = || {
        let h = hom();
        DynamicFusionMapSpace::from_final_homspace(&rule, h).unwrap()
    };
    let lhs_fusion = fusion();
    let rhs_fusion = fusion();
    let lhs_space = lhs_fusion.clone();
    let rhs_space = rhs_fusion.clone();
    let len = lhs_space.required_len().unwrap();
    let mk = |seed: f64| {
        (0..len)
            .map(|i| Complex64::new(seed + i as f64, 1.0 + (i % 3) as f64 - seed))
            .collect::<Vec<_>>()
    };
    let lhs_data = mk(1.0);
    let rhs_data = mk(2.0);

    // Oracle: eager conjugate-transpose, then a plain (no-conjugate) contraction.
    let (adj_space, adj_data) = crate::adjoint::adjoint_dyn(&rule, &lhs_space, &lhs_data).unwrap();
    // compose(a†, b): contract a†'s domain axis (1) with b's codomain axis (0).
    let dst_space =
        crate::DynamicFusionMapSpace::contracted(&rule, &adj_space, &rhs_space, &[1], &[0])
            .unwrap();
    let provider = Arc::new(rule);
    let dst_bound = crate::BoundDynamicFusionMapSpace::bind_multiplicity_free(
        dst_space.clone(),
        Arc::clone(&provider),
    )
    .unwrap();
    let adj_bound =
        crate::BoundDynamicFusionMapSpace::bind_multiplicity_free(adj_space, Arc::clone(&provider))
            .unwrap();
    let lhs_bound =
        crate::BoundDynamicFusionMapSpace::bind_multiplicity_free(lhs_space, Arc::clone(&provider))
            .unwrap();
    let rhs_bound =
        crate::BoundDynamicFusionMapSpace::bind_multiplicity_free(rhs_space, Arc::clone(&provider))
            .unwrap();
    let mut oracle = vec![Complex64::new(0.0, 0.0); dst_space.required_len().unwrap()];
    let mut ctx = crate::TensorContractFusionExecutionContext::<Complex64, _>::default();
    ctx.tensorcontract_prelowered_into(
        &dst_bound,
        &mut oracle,
        crate::FusionOperand::direct(adj_bound.space()),
        &adj_data,
        crate::FusionOperand::direct(rhs_bound.space()),
        &rhs_data,
        TensorContractSpec::with_default_output_order(&[1], &[0]),
        Complex64::new(1.0, 0.0),
        crate::ContractDestinationInit::Axpby(Complex64::new(0.0, 0.0)),
    )
    .unwrap();

    let lhs_operand = crate::FusionOperand::adjoint(lhs_bound.space());
    let rhs_operand = crate::FusionOperand::direct(rhs_bound.space());
    let axes = || {
        TensorContractSpec::new_with_conjugation(
            &[1],
            &[0],
            crate::OutputAxisOrder::identity(),
            true,
            false,
        )
    };
    for pass in 0..2 {
        let mut ctx = crate::TensorContractFusionExecutionContext::<Complex64, _>::default();
        let adjoint_view_build_count_rec = probe::record();
        let fusion_operand_projection_prepares_rec = probe::record();
        for _ in 0..2 {
            let mut fold = vec![Complex64::new(0.0, 0.0); dst_space.required_len().unwrap()];
            ctx.tensorcontract_prelowered_into(
                &dst_bound,
                &mut fold,
                lhs_operand,
                &lhs_data,
                rhs_operand,
                &rhs_data,
                axes(),
                Complex64::new(1.0, 0.0),
                crate::ContractDestinationInit::Axpby(Complex64::new(0.0, 0.0)),
            )
            .unwrap();
            assert!(
                ctx.last_resolution_is_core(),
                "core-form prelowered adjoint must resolve to Core"
            );
            let max = oracle
                .iter()
                .zip(&fold)
                .map(|(o, f)| (o - f).norm())
                .fold(0.0f64, f64::max);
            assert!(max < 1e-10, "fold vs eager oracle: max diff {max}");
        }
        if pass == 0 {
            let mut folded_again =
                vec![Complex64::new(0.0, 0.0); dst_space.required_len().unwrap()];
            ctx.tensorcontract_prelowered_into(
                &dst_bound,
                &mut folded_again,
                lhs_operand,
                &lhs_data,
                rhs_operand,
                &rhs_data,
                axes(),
                Complex64::new(1.0, 0.0),
                crate::ContractDestinationInit::Axpby(Complex64::new(0.0, 0.0)),
            )
            .unwrap();
            assert_eq!(folded_again, oracle);
            assert!(ctx.last_resolution_is_core());
        }
        // What: neither cold/reset execution nor NoCache may reconstruct the
        // full categorical adjoint view or prepare a logical block projection
        // for a canonical parent-owned coupled-region contraction.
        assert_eq!(adjoint_view_build_count_rec.get(Event::AdjointViewBuild), 0);
        assert_eq!(
            fusion_operand_projection_prepares_rec.get(Event::FusionOperandProjectionPrepare),
            0
        );
    }

    let typed_dst_hom = dst_space.homspace().clone();
    let typed_dst_space =
        DynamicFusionMapSpace::from_final_homspace(provider.as_ref(), typed_dst_hom.clone())
            .unwrap();
    let lhs_typed =
        TestTensor::<Complex64>::from_vec_with_fusion_space(lhs_data.clone(), lhs_fusion).unwrap();
    let rhs_typed =
        TestTensor::<Complex64>::from_vec_with_fusion_space(rhs_data.clone(), rhs_fusion).unwrap();
    let candidate = crate::contract::contracted_axis_order_candidates(&[0], &[0]).remove(0);
    let reverse_plan = crate::contract::prepare_tensorcontract_fusion_plan_dyn_raw_with_axis_order_and_orientation(
        provider.as_ref(),
        &dst_space,
        &lhs_typed.fusion_space().unwrap().as_ref().clone(),
        &rhs_typed.fusion_space().unwrap().as_ref().clone(),
        TensorContractSpec::new_with_conjugation(
            &[0],
            &[0],
            OutputAxisOrder::identity(),
            true,
            false,
        ),
        &candidate,
        crate::contract::FusionContractOrientation::RhsLhs,
    )
    .unwrap();
    let mut reverse_dst = TestTensor::<Complex64>::from_vec_with_fusion_space(
        vec![Complex64::zero(); oracle.len()],
        typed_dst_space,
    )
    .unwrap();
    let mut tree_backend = DenseTreeTransformOperations::default_executor();
    let mut tree_workspace = TreeTransformWorkspace::default();
    let mut contract_backend = DenseTreeTransformOperations::default_executor();
    let mut contract_workspace = TensorContractWorkspace::default();
    crate::contract::tensorcontract_fusion_dynamic_plan_into_with(
        &mut tree_backend,
        &mut tree_workspace,
        &mut contract_backend,
        &mut contract_workspace,
        provider.as_ref(),
        &reverse_plan,
        &mut reverse_dst,
        &lhs_typed,
        &rhs_typed,
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();
    for (&actual, &expected) in reverse_dst.data().iter().zip(&oracle) {
        assert!((actual - expected).norm() < 1.0e-10);
    }
    let mut artifact_dst = TestTensor::<Complex64>::from_vec_with_dynamic_space(
        vec![Complex64::zero(); oracle.len()],
        reverse_dst.dims().to_vec(),
        reverse_dst.fusion_space().unwrap().as_ref().clone(),
    )
    .unwrap();
    crate::contract::execute_dynamic_tree_execution_artifact_for_test(
        provider.as_ref(),
        &reverse_plan,
        &mut artifact_dst,
        &lhs_typed,
        &rhs_typed,
        Complex64::one(),
        Complex64::zero(),
    )
    .unwrap();
    for (&actual, &expected) in artifact_dst.data().iter().zip(&oracle) {
        assert!((actual - expected).norm() < 1.0e-10);
    }
}
