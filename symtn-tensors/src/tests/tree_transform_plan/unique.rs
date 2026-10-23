use super::*;

#[test]
fn unique_tree_transform_plan_builder_rejects_generic_fusion() {
    let src_key = fusion_tree_test_key([1, 1], [1], 1, [false, false], [false]);
    let src_structure = packed_fixture_structure(3, [(src_key, vec![1, 1, 1])]).unwrap();
    let operation = TreeTransformOperation::braid([1, 0], [0], [1, 0], [0]);

    let err = build_unique_tree_transform_group_plan(
        &GenericMultiplicityRule,
        operation.clone(),
        &src_structure,
        |_| -> Result<(FusionTreePairKey, f64), OperationError> {
            unreachable!("GenericFusion must be rejected before transforming keys")
        },
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::UnsupportedFusionStyle {
            operation: Box::new(operation),
            style: FusionStyleKind::Generic,
        }
    );
}

#[test]
fn tree_transform_operation_key_distinguishes_permute_from_explicit_braid() {
    assert!(TreeTransformOperation::permute([1, 0], [0]).requires_symmetric_braiding());
    assert!(!TreeTransformOperation::transpose([1, 0], [0]).requires_symmetric_braiding());
    assert!(!TreeTransformOperation::braid([1, 0], [0], [1, 0], [0]).requires_symmetric_braiding());
}

#[test]
fn unique_tree_transform_plan_builder_rejects_permute_without_symmetric_braiding() {
    let src_key = fusion_tree_test_key([1, 0], [1], 1, [false, false], [false]);
    let src_structure = packed_fixture_structure(3, [(src_key, vec![1, 1, 1])]).unwrap();
    let operation = TreeTransformOperation::permute([1, 0], [0]);

    let err = build_unique_tree_transform_group_plan(
        &UniqueAnyonicRule,
        operation.clone(),
        &src_structure,
        |_| -> Result<(FusionTreePairKey, f64), OperationError> {
            unreachable!("permutation must reject non-symmetric braiding before key transform")
        },
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::UnsupportedBraidingStyle {
            operation: Box::new(operation),
            style: BraidingStyleKind::Anyonic,
        }
    );
}

#[test]
fn unique_tree_transform_plan_builder_defers_explicit_no_braiding_to_crossing_logic() {
    let src_key = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [1, 0],
            [1],
            1,
            [false, false],
            [false],
            [],
            [],
            [1],
            [],
        )
        .unwrap(),
    );
    let src_tree = expect_tree_key(&src_key);
    let src_structure = packed_fixture_structure(3, [(src_key.clone(), vec![1, 1, 1])]).unwrap();

    let plan = build_unique_tree_transform_group_plan(
        &UniquePlanarRule,
        TreeTransformOperation::braid([1, 0], [2], [1, 0], [0]),
        &src_structure,
        |src| Ok((src.clone(), 1.0_f64)),
    )
    .unwrap();

    assert_eq!(plan.specs().len(), 1);
    assert_eq!(&plan.specs()[0].group_key(), &src_tree.group_key());
    assert_eq!(plan.specs()[0].src_keys(), &[expect_tree_key(&src_key)]);
    assert_eq!(plan.specs()[0].dst_keys(), &[expect_tree_key(&src_key)]);
    assert_eq!(plan.specs()[0].recoupling_coefficients_dst_src(), &[1.0]);
}

#[test]
fn unique_all_codomain_braid_plan_builder_lowers_codomain_single_tree() {
    let src_key = all_codomain_fusion_tree_test_key([1, 1], 0, [false, true], [], [1]);
    let expected_dst_key = all_codomain_fusion_tree_test_key([1, 1], 0, [true, false], [], [1]);
    let src_tree = expect_tree_key(&src_key);
    let src_structure = packed_fixture_structure(2, [(src_key.clone(), vec![1, 1])]).unwrap();

    let plan = build_unique_all_codomain_tree_transform_group_plan(
        &UniqueAnyonicRule,
        TreeTransformOperation::braid([1, 0], Vec::<usize>::new(), [0, 1], Vec::<usize>::new()),
        &src_structure,
    )
    .unwrap();

    assert_eq!(plan.specs().len(), 1);
    assert_eq!(&plan.specs()[0].group_key(), &src_tree.group_key());
    assert_eq!(plan.specs()[0].src_keys(), &[expect_tree_key(&src_key)]);
    assert_eq!(
        plan.specs()[0].dst_keys(),
        &[expect_tree_key(&expected_dst_key)]
    );
    assert_eq!(plan.specs()[0].recoupling_coefficients_dst_src(), &[-2.0]);
}

#[test]
fn unique_all_codomain_permute_plan_builder_lowers_symmetric_permutation() {
    let src_key = all_codomain_fusion_tree_test_key([1, 1], 0, [false, true], [], [1]);
    let expected_dst_key = all_codomain_fusion_tree_test_key([1, 1], 0, [true, false], [], [1]);
    let src_structure = packed_fixture_structure(2, [(src_key.clone(), vec![1, 1])]).unwrap();

    let plan = build_unique_all_codomain_tree_transform_group_plan(
        &UniqueZ2Rule,
        TreeTransformOperation::permute([1, 0], Vec::<usize>::new()),
        &src_structure,
    )
    .unwrap();

    assert_eq!(plan.specs().len(), 1);
    assert_eq!(plan.specs()[0].src_keys(), &[expect_tree_key(&src_key)]);
    assert_eq!(
        plan.specs()[0].dst_keys(),
        &[expect_tree_key(&expected_dst_key)]
    );
    assert_eq!(plan.specs()[0].recoupling_coefficients_dst_src(), &[1.0]);
}

#[test]
fn unique_all_codomain_plan_builder_rejects_domain_operation_scope() {
    let src_key = all_codomain_fusion_tree_test_key([1, 1], 0, [false, false], [], [1]);
    let src_structure = packed_fixture_structure(2, [(src_key, vec![1, 1])]).unwrap();
    let operation = TreeTransformOperation::braid([1, 0], [0], [0, 1], [0]);

    let err = build_unique_all_codomain_tree_transform_group_plan(
        &UniqueZ2Rule,
        operation.clone(),
        &src_structure,
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::UnsupportedTreeTransformScope {
            operation: Box::new(operation),
            message: "all-codomain UniqueFusion lowering requires an empty domain operation",
        }
    );
}

#[test]
fn unique_all_codomain_plan_builder_accepts_explicit_vacuum_empty_domain() {
    let src_key = BlockKey::from(FusionTreePairKey::pair(
        FusionTreeKey::try_from_sector_ids_for_rule(
            &UniqueZ2Rule,
            [1, 1],
            0,
            [false, false],
            [],
            [1],
        )
        .unwrap(),
        empty_fusion_tree_with_coupled(0),
    ));
    let expected_dst_key = BlockKey::from(FusionTreePairKey::pair(
        FusionTreeKey::try_from_sector_ids_for_rule(
            &UniqueZ2Rule,
            [1, 1],
            0,
            [false, false],
            [],
            [1],
        )
        .unwrap(),
        empty_fusion_tree_with_coupled(0),
    ));
    let src_structure = packed_fixture_structure(2, [(src_key.clone(), vec![1, 1])]).unwrap();

    let plan = build_unique_all_codomain_tree_transform_group_plan(
        &UniqueZ2Rule,
        TreeTransformOperation::permute([1, 0], Vec::<usize>::new()),
        &src_structure,
    )
    .unwrap();

    assert_eq!(plan.specs().len(), 1);
    assert_eq!(plan.specs()[0].src_keys(), &[expect_tree_key(&src_key)]);
    assert_eq!(
        plan.specs()[0].dst_keys(),
        &[expect_tree_key(&expected_dst_key)]
    );
    assert_eq!(plan.specs()[0].recoupling_coefficients_dst_src(), &[1.0]);
}

#[test]
fn unique_all_codomain_plan_builder_rejects_explicit_nonvacuum_empty_domain() {
    let src_key = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [1, 0],
            [],
            1,
            [false, false],
            [],
            [],
            [],
            [1],
            [],
        )
        .unwrap(),
    );
    let src_structure = packed_fixture_structure(2, [(src_key, vec![1, 1])]).unwrap();

    let err = build_unique_all_codomain_tree_transform_group_plan(
        &UniqueZ2Rule,
        TreeTransformOperation::permute([1, 0], Vec::<usize>::new()),
        &src_structure,
    )
    .unwrap_err();

    // What: the current all-codomain facade diagnoses its operation scope
    // before the core categorical boundary admits the raw source.
    assert_eq!(
        err,
        OperationError::ExpectedAllCodomainFusionTree { index: 0 }
    );
}

#[test]
fn unique_all_codomain_plan_builder_rejects_nonempty_domain_tree() {
    let src_key = fusion_tree_test_key([1, 0], [1], 1, [false, false], [false]);
    let src_structure = packed_fixture_structure(3, [(src_key, vec![1, 1, 1])]).unwrap();

    let err = build_unique_all_codomain_tree_transform_group_plan(
        &UniqueZ2Rule,
        TreeTransformOperation::permute([1, 0], Vec::<usize>::new()),
        &src_structure,
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::ExpectedAllCodomainFusionTree { index: 0 }
    );
}

#[test]
fn unique_all_codomain_permute_plan_builder_rejects_nonsymmetric_braiding() {
    let src_key = all_codomain_fusion_tree_test_key([1, 1], 0, [false, false], [], [1]);
    let src_structure = packed_fixture_structure(2, [(src_key, vec![1, 1])]).unwrap();
    let operation = TreeTransformOperation::permute([1, 0], Vec::<usize>::new());

    let err = build_unique_all_codomain_tree_transform_group_plan(
        &UniqueAnyonicRule,
        operation.clone(),
        &src_structure,
    )
    .unwrap_err();

    assert_eq!(
        err,
        OperationError::UnsupportedBraidingStyle {
            operation: Box::new(operation),
            style: BraidingStyleKind::Anyonic,
        }
    );
}

#[test]
fn unique_tree_pair_plan_builder_lowers_domain_only_permutation() {
    let src_key = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [1],
            [0, 1],
            1,
            [false],
            [false, true],
            [],
            [],
            [],
            [1],
        )
        .unwrap(),
    );
    let expected_dst_key = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [1],
            [1, 0],
            1,
            [false],
            [true, false],
            [],
            [],
            [],
            [1],
        )
        .unwrap(),
    );
    let src_tree = expect_tree_key(&src_key);
    let src_structure = packed_fixture_structure(3, [(src_key.clone(), vec![1, 1, 1])]).unwrap();

    let plan = build_unique_tree_pair_transform_group_plan(
        &UniqueZ2Rule,
        TreeTransformOperation::permute([0], [2, 1]),
        &src_structure,
    )
    .unwrap();

    assert_eq!(plan.specs().len(), 1);
    assert_eq!(&plan.specs()[0].group_key(), &src_tree.group_key());
    assert_eq!(plan.specs()[0].src_keys(), &[expect_tree_key(&src_key)]);
    assert_eq!(
        plan.specs()[0].dst_keys(),
        &[expect_tree_key(&expected_dst_key)]
    );
    assert_eq!(plan.specs()[0].recoupling_coefficients_dst_src(), &[1.0]);
}

#[test]
fn unique_tree_pair_plan_builder_lowers_codomain_domain_crossing_braid() {
    let src_key = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids([1], [1], 1, [false], [true], [], [], [], [])
            .unwrap(),
    );
    let expected_dst_key = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids([1], [1], 1, [false], [true], [], [], [], [])
            .unwrap(),
    );
    let src_structure = packed_fixture_structure(2, [(src_key.clone(), vec![1, 1])]).unwrap();

    let plan = build_unique_tree_pair_transform_group_plan(
        &UniqueAnyonicRule,
        TreeTransformOperation::braid([1], [0], [0], [1]),
        &src_structure,
    )
    .unwrap();

    assert_eq!(plan.specs().len(), 1);
    assert_eq!(plan.specs()[0].src_keys(), &[expect_tree_key(&src_key)]);
    assert_eq!(
        plan.specs()[0].dst_keys(),
        &[expect_tree_key(&expected_dst_key)]
    );
    assert_eq!(plan.specs()[0].recoupling_coefficients_dst_src(), &[-2.0]);
}

#[test]
fn unique_tree_pair_plan_builder_lowers_cyclic_transpose() {
    let src_key = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids([1], [1], 1, [false], [true], [], [], [], [])
            .unwrap(),
    );
    let expected_dst_key = src_key.clone();
    let src_structure = packed_fixture_structure(2, [(src_key.clone(), vec![1, 1])]).unwrap();
    let operation = TreeTransformOperation::transpose([1], [0]);

    let plan =
        build_unique_tree_pair_transform_group_plan(&UniqueZ2Rule, operation, &src_structure)
            .unwrap();

    assert_eq!(plan.specs().len(), 1);
    assert_eq!(plan.specs()[0].src_keys(), &[expect_tree_key(&src_key)]);
    assert_eq!(
        plan.specs()[0].dst_keys(),
        &[expect_tree_key(&expected_dst_key)]
    );
    assert_eq!(plan.specs()[0].recoupling_coefficients_dst_src(), &[1.0]);
}

#[test]
fn unique_tree_pair_plan_builder_lowers_rank_four_cyclic_transpose() {
    let src_key = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [1, 0],
            [1, 0],
            1,
            [false, false],
            [false, false],
            [],
            [],
            [1],
            [1],
        )
        .unwrap(),
    );
    let expected_dst_key = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [1, 1],
            [0, 0],
            0,
            [true, false],
            [false, true],
            [],
            [],
            [1],
            [1],
        )
        .unwrap(),
    );
    let src_structure = packed_fixture_structure(4, [(src_key.clone(), vec![1, 1, 1, 1])]).unwrap();
    let operation = TreeTransformOperation::transpose([2, 0], [3, 1]);

    let plan =
        build_unique_tree_pair_transform_group_plan(&UniqueZ2Rule, operation, &src_structure)
            .unwrap();

    assert_eq!(plan.specs().len(), 1);
    assert_eq!(plan.specs()[0].src_keys(), &[expect_tree_key(&src_key)]);
    assert_eq!(
        plan.specs()[0].dst_keys(),
        &[expect_tree_key(&expected_dst_key)]
    );
    assert_eq!(plan.specs()[0].recoupling_coefficients_dst_src(), &[1.0]);
}

#[derive(Clone, Copy, Debug)]
struct TensorKitZ4Element2Rule;

impl TensorKitZ4Element2Rule {
    fn label(sector: SectorId) -> usize {
        assert!(sector.id() < 4, "Z4 fixture sector must be in 0..4");
        sector.id()
    }

    fn cispi(exponent: f64) -> Complex64 {
        Complex64::from_polar(1.0, std::f64::consts::PI * exponent)
    }

    fn cocycle(left: SectorId, middle: SectorId, right: SectorId) -> Complex64 {
        let left = Self::label(left);
        let middle = Self::label(middle);
        let right = Self::label(right);
        let wrapped_sum = (middle + right) % 4;
        let carry = middle + right - wrapped_sum;
        Self::cispi((4 * left * carry) as f64 / 16.0)
    }
}

impl symtn_core::testing::Provider for TensorKitZ4Element2Rule {}

impl FusionRule for TensorKitZ4Element2Rule {
    fn rule_identity(&self) -> symtn_core::RuleIdentity {
        symtn_core::RuleIdentity::of_type::<Self>()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Anyonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        SectorId::new((4 - Self::label(sector)) % 4)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        vec![SectorId::new((Self::label(left) + Self::label(right)) % 4)].into()
    }
}

impl MultiplicityFreeFusionRule for TensorKitZ4Element2Rule {}

impl MultiplicityFreeFusionSymbols for TensorKitZ4Element2Rule {
    type Scalar = Complex64;

    fn f_symbol_scalar(
        &self,
        left: SectorId,
        middle: SectorId,
        right: SectorId,
        coupled: SectorId,
        left_coupled: SectorId,
        right_coupled: SectorId,
    ) -> Self::Scalar {
        let expected_left = SectorId::new((Self::label(left) + Self::label(middle)) % 4);
        let expected_right = SectorId::new((Self::label(middle) + Self::label(right)) % 4);
        let expected_coupled = SectorId::new((Self::label(expected_left) + Self::label(right)) % 4);
        if left_coupled == expected_left
            && right_coupled == expected_right
            && coupled == expected_coupled
        {
            Self::cocycle(left, middle, right)
        } else {
            Complex64::new(0.0, 0.0)
        }
    }

    fn r_symbol_scalar(&self, left: SectorId, right: SectorId, coupled: SectorId) -> Self::Scalar {
        let expected = SectorId::new((Self::label(left) + Self::label(right)) % 4);
        if coupled == expected {
            Self::cispi((Self::label(left) * Self::label(right)) as f64 / 4.0)
        } else {
            Complex64::new(0.0, 0.0)
        }
    }
}

impl MultiplicityFreeRigidSymbols for TensorKitZ4Element2Rule {
    fn dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        Complex64::new(1.0, 0.0)
    }

    fn inv_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        Complex64::new(1.0, 0.0)
    }

    fn sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        Complex64::new(1.0, 0.0)
    }

    fn inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        Complex64::new(1.0, 0.0)
    }

    fn twist_scalar(&self, sector: SectorId) -> Self::Scalar {
        let label = Self::label(sector);
        Self::cispi((label * label) as f64 / 4.0)
    }

    fn frobenius_schur_phase_scalar(&self, sector: SectorId) -> Self::Scalar {
        Self::cocycle(sector, self.dual(sector), sector)
    }
}

fn tensor_kit_z4_rank_three_pair() -> FusionTreePairKey {
    let rule = TensorKitZ4Element2Rule;
    FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &rule,
            [SectorId::new(1), SectorId::new(2), SectorId::new(3)],
            SectorId::new(2),
            [false, false, false],
            [SectorId::new(3)],
            [MultiplicityIndex::ONE, MultiplicityIndex::ONE],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(
            &rule,
            [SectorId::new(2)],
            SectorId::new(2),
            [false],
            [],
            [],
        )
        .unwrap(),
    )
}

fn assert_complex_oracle(actual: Complex64, expected: Complex64) {
    assert!(
        (actual - expected).norm() < 1.0e-14,
        "actual={actual:?}, expected={expected:?}"
    );
}

#[test]
fn unique_production_domain_fermion_crossing_matches_tensorkit_oracle() {
    let rule = FermionParityFusionRule;
    let odd = SectorId::new(1);
    let source = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(&rule, [odd], odd, [false], [], []).unwrap(),
        FusionTreeKey::try_new_for_rule(&rule, [odd], odd, [true], [], []).unwrap(),
    );
    let source_structure =
        packed_fixture_structure(2, [(BlockKey::from(source.clone()), vec![1, 1])]).unwrap();

    let plan = build_unique_tree_pair_transform_group_plan(
        &rule,
        TreeTransformOperation::braid([1], [0], [0], [1]),
        &source_structure,
    )
    .unwrap();

    // What: moving one odd domain leg across one odd codomain leg carries the
    // exact TensorKit fermionic sign, rather than a self-consistency result.
    // Closed form: the odd-odd exchange sign -1; checked against TensorKit
    // 0.17.0 (cfaa073) and 0.17.1.
    assert_eq!(plan.specs().len(), 1);
    assert_eq!(plan.specs()[0].dst_keys(), &[source]);
    assert_eq!(plan.specs()[0].recoupling_coefficients_dst_src(), &[-1.0]);
}

#[test]
fn unique_production_complex_artin_and_inverse_match_tensorkit_oracle() {
    let rule = TensorKitZ4Element2Rule;
    let source = tensor_kit_z4_rank_three_pair();
    let source_structure =
        packed_fixture_structure(4, [(BlockKey::from(source.clone()), vec![1; 4])]).unwrap();
    let expected = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &rule,
            [SectorId::new(1), SectorId::new(3), SectorId::new(2)],
            SectorId::new(2),
            [false, false, false],
            [SectorId::new(0)],
            [MultiplicityIndex::ONE, MultiplicityIndex::ONE],
        )
        .unwrap(),
        source.domain_tree().clone(),
    );

    for (codomain_levels, expected_coefficient) in [
        ([0, 1, 2], Complex64::new(0.0, -1.0)),
        ([0, 2, 1], Complex64::new(0.0, 1.0)),
    ] {
        let plan = build_unique_tree_pair_transform_group_plan(
            &rule,
            TreeTransformOperation::braid([0, 2, 1], [3], codomain_levels, [3]),
            &source_structure,
        )
        .unwrap();

        // What: Z4Element{2}'s later Artin crossing preserves the recoupled
        // innerline and conjugates the complex phase when the levels reverse.
        // Closed form: R(2,3) = exp(2πi p·2·3/N²) = exp(3πi/2) = -i for
        // N = 4, p = 2 (the F ratio is 1), and +i for the inverse; checked
        // against TensorKit 0.17.0 (cfaa073) and 0.17.1.
        assert_eq!(plan.specs().len(), 1);
        assert_eq!(plan.specs()[0].dst_keys(), std::slice::from_ref(&expected));
        assert_complex_oracle(
            plan.specs()[0].recoupling_coefficients_dst_src()[0],
            expected_coefficient,
        );
    }
}

#[test]
fn unique_production_pivotal_transpose_matches_tensorkit_oracle() {
    let rule = TensorKitZ4Element2Rule;
    let source = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &rule,
            [SectorId::new(1), SectorId::new(2)],
            SectorId::new(3),
            [false, true],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(
            &rule,
            [SectorId::new(3)],
            SectorId::new(3),
            [true],
            [],
            [],
        )
        .unwrap(),
    );
    let expected = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &rule,
            [SectorId::new(2), SectorId::new(1)],
            SectorId::new(3),
            [true, false],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(
            &rule,
            [SectorId::new(3)],
            SectorId::new(3),
            [true],
            [],
            [],
        )
        .unwrap(),
    );
    let source_structure =
        packed_fixture_structure(3, [(BlockKey::from(source), vec![1; 3])]).unwrap();

    let plan = build_unique_tree_pair_transform_group_plan(
        &rule,
        TreeTransformOperation::transpose([1, 2], [0]),
        &source_structure,
    )
    .unwrap();

    // What: the nontrivial Z4 Frobenius-Schur/A-symbol phase survives the
    // cyclic transpose with TensorKit's exact destination dual flags.
    // Closed form: fs(1) = ω(1, 3, 1) = exp(iπ·2p·1·4/N²) = -1 for N = 4,
    // p = 2; checked against TensorKit 0.17.0 (cfaa073) and 0.17.1.
    assert_eq!(plan.specs().len(), 1);
    assert_eq!(plan.specs()[0].dst_keys(), &[expected]);
    assert_complex_oracle(
        plan.specs()[0].recoupling_coefficients_dst_src()[0],
        Complex64::new(-1.0, 0.0),
    );
}

fn unique_rank_three_tree_pair<R>(rule: &R, left: SectorId, right: SectorId) -> FusionTreePairKey
where
    R: FusionRule,
{
    let coupled = rule.fusion_channels(left, right)[0];
    FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            rule,
            [left, right],
            coupled,
            [false, true],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(rule, [coupled], coupled, [true], [], []).unwrap(),
    )
}

fn assert_unique_and_generic_plan_are_identical<R>(
    rule: &R,
    source: FusionTreePairKey,
    operation: TreeTransformOperation,
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let rank = source.codomain_tree().uncoupled().len() + source.domain_tree().uncoupled().len();
    let source_structure =
        packed_fixture_structure(rank, [(BlockKey::from(source), vec![1; rank])]).unwrap();
    let specialized =
        build_unique_tree_pair_transform_group_plan(rule, operation.clone(), &source_structure)
            .unwrap();
    let generic = build_multiplicity_free_tree_pair_transform_group_plan(
        rule,
        operation.clone(),
        &source_structure,
    )
    .unwrap();
    let standard =
        build_tree_pair_transform_group_plan(rule, operation, &source_structure).unwrap();

    // What: the production Unique lowering is the exact one-term form of the
    // explicit multiplicity-free algorithm, including key and coefficient
    // ordering, and is the branch selected by the standard builder.
    assert_eq!(specialized, generic);
    assert_eq!(standard, specialized);
    assert_eq!(specialized.specs().len(), 1);
    assert_eq!(specialized.specs()[0].src_keys().len(), 1);
    assert_eq!(specialized.specs()[0].dst_keys().len(), 1);

    let destination_structure = packed_fixture_structure(
        rank,
        [(specialized.specs()[0].dst_keys()[0].clone(), vec![1; rank])],
    )
    .unwrap();
    let specialized_replay = specialized
        .compile_structures(&destination_structure, &source_structure)
        .unwrap();
    let generic_replay = generic
        .compile_structures(&destination_structure, &source_structure)
        .unwrap();

    // What: specialized and explicit generic plans compile to identical raw
    // replay blocks, layouts, coefficients, schedules, and structure guards.
    assert_eq!(specialized_replay, generic_replay);
}

#[test]
fn unique_production_lowering_matches_generic_across_pointed_rules_and_operations() {
    let u1_source = unique_rank_three_tree_pair(
        &U1FusionRule,
        U1Irrep::new(1).sector_id(),
        U1Irrep::new(-2).sector_id(),
    );
    assert_unique_and_generic_plan_are_identical(
        &U1FusionRule,
        u1_source,
        TreeTransformOperation::permute([1, 0], [2]),
    );

    let z2_source = unique_rank_three_tree_pair(&Z2FusionRule, SectorId::new(1), SectorId::new(0));
    assert_unique_and_generic_plan_are_identical(
        &Z2FusionRule,
        z2_source,
        TreeTransformOperation::braid([2, 0], [1], [0, 2], [1]),
    );

    let fz2_source =
        unique_rank_three_tree_pair(&FermionParityFusionRule, SectorId::new(1), SectorId::new(1));
    assert_unique_and_generic_plan_are_identical(
        &FermionParityFusionRule,
        fz2_source,
        TreeTransformOperation::transpose([2, 0], [1]),
    );

    let product = FpU1Rule::try_new(FermionParityFusionRule, U1FusionRule).unwrap();
    let odd_charge = product.encode_component_ids(SectorId::new(1), U1Irrep::new(2).sector_id());
    let even_charge = product.encode_component_ids(SectorId::new(0), U1Irrep::new(-1).sector_id());
    let coupled = product.fusion_channels(odd_charge, even_charge)[0];
    let product_source = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &product,
            [odd_charge, even_charge],
            coupled,
            [false, true],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(
            &product,
            [product.vacuum(), coupled],
            coupled,
            [true, false],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
    );
    assert_unique_and_generic_plan_are_identical(
        &product,
        product_source,
        TreeTransformOperation::braid([0, 1, 3], [2], [0, 1], [2, 3]),
    );

    let anyonic_source = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &UniqueAnyonicRule,
            [SectorId::new(1)],
            SectorId::new(1),
            [false],
            [],
            [],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(
            &UniqueAnyonicRule,
            [SectorId::new(1)],
            SectorId::new(1),
            [true],
            [],
            [],
        )
        .unwrap(),
    );
    assert_unique_and_generic_plan_are_identical(
        &UniqueAnyonicRule,
        anyonic_source,
        TreeTransformOperation::braid([1], [0], [0], [1]),
    );

    let asymmetric_source = unique_rank_three_tree_pair(
        &AsymmetricAnyonicPointedRule,
        SectorId::new(1),
        SectorId::new(2),
    );
    assert_unique_and_generic_plan_are_identical(
        &AsymmetricAnyonicPointedRule,
        asymmetric_source.clone(),
        TreeTransformOperation::braid([1, 0], [2], [0, 1], [2]),
    );
    assert_unique_and_generic_plan_are_identical(
        &AsymmetricAnyonicPointedRule,
        asymmetric_source,
        TreeTransformOperation::transpose([2, 0], [1]),
    );
}

#[test]
fn unique_production_lowering_matches_generic_at_rank_zero_and_one() {
    let scalar = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(&U1FusionRule, [], U1FusionRule.vacuum(), [], [], [])
            .unwrap(),
        FusionTreeKey::try_new_for_rule(&U1FusionRule, [], U1FusionRule.vacuum(), [], [], [])
            .unwrap(),
    );
    assert_unique_and_generic_plan_are_identical(
        &U1FusionRule,
        scalar,
        TreeTransformOperation::permute([], []),
    );

    let rank_one = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &Z2FusionRule,
            [Z2FusionRule.vacuum()],
            Z2FusionRule.vacuum(),
            [true],
            [],
            [],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(&Z2FusionRule, [], Z2FusionRule.vacuum(), [], [], [])
            .unwrap(),
    );
    assert_unique_and_generic_plan_are_identical(
        &Z2FusionRule,
        rank_one,
        TreeTransformOperation::transpose([0], []),
    );
}

#[test]
fn unique_all_codomain_production_lowering_matches_generic_replay_exactly() {
    let source = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &Z2FusionRule,
            [SectorId::new(1), SectorId::new(1)],
            Z2FusionRule.vacuum(),
            [false, true],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(&Z2FusionRule, [], Z2FusionRule.vacuum(), [], [], [])
            .unwrap(),
    );
    let source_structure =
        packed_fixture_structure(2, [(BlockKey::from(source), vec![1; 2])]).unwrap();
    let operation = TreeTransformOperation::permute([1, 0], []);
    let specialized = build_unique_all_codomain_tree_transform_group_plan(
        &Z2FusionRule,
        operation.clone(),
        &source_structure,
    )
    .unwrap();
    let generic = build_multiplicity_free_all_codomain_tree_transform_group_plan(
        &Z2FusionRule,
        operation.clone(),
        &source_structure,
    )
    .unwrap();
    let standard =
        build_all_codomain_tree_transform_group_plan(&Z2FusionRule, operation, &source_structure)
            .unwrap();

    // What: all-codomain dispatch lowers the same key, phase, and raw replay
    // descriptor as the explicit multiplicity-free algorithm.
    assert_eq!(specialized, generic);
    assert_eq!(standard, specialized);
    let destination_structure = packed_fixture_structure(
        2,
        [(specialized.specs()[0].dst_keys()[0].clone(), vec![1; 2])],
    )
    .unwrap();
    assert_eq!(
        specialized
            .compile_structures(&destination_structure, &source_structure)
            .unwrap(),
        generic
            .compile_structures(&destination_structure, &source_structure)
            .unwrap()
    );
}

fn assert_unique_and_generic_error_are_identical<R>(
    rule: &R,
    source_structure: &BlockStructure,
    operation: TreeTransformOperation,
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let specialized =
        build_unique_tree_pair_transform_group_plan(rule, operation.clone(), source_structure)
            .unwrap_err();
    let generic =
        build_multiplicity_free_tree_pair_transform_group_plan(rule, operation, source_structure)
            .unwrap_err();
    assert_eq!(specialized, generic);
}

#[test]
fn unique_production_lowering_preserves_generic_error_precedence() {
    let source = unique_rank_three_tree_pair(&Z2FusionRule, SectorId::new(1), SectorId::new(0));
    let source_structure =
        packed_fixture_structure(3, [(BlockKey::from(source), vec![1; 3])]).unwrap();

    // What: malformed permutations, braid levels, and noncyclic transposes
    // fail with the same error and precedence as the explicit generic path.
    assert_unique_and_generic_error_are_identical(
        &Z2FusionRule,
        &source_structure,
        TreeTransformOperation::permute([0, 0], [2]),
    );
    assert_unique_and_generic_error_are_identical(
        &Z2FusionRule,
        &source_structure,
        TreeTransformOperation::braid([1, 0], [2], [0], [2]),
    );
    assert_unique_and_generic_error_are_identical(
        &Z2FusionRule,
        &source_structure,
        TreeTransformOperation::transpose([1, 0], [2]),
    );

    let planar_source =
        unique_rank_three_tree_pair(&UniquePlanarRule, SectorId::new(1), SectorId::new(0));
    let planar_structure =
        packed_fixture_structure(3, [(BlockKey::from(planar_source), vec![1; 3])]).unwrap();
    assert_unique_and_generic_error_are_identical(
        &UniquePlanarRule,
        &planar_structure,
        TreeTransformOperation::permute([1, 0], [2]),
    );
}

fn z2_two_leg_pair_with_empty_domain(uncoupled: [SectorId; 2]) -> FusionTreePairKey {
    let rule = Z2FusionRule;
    let coupled = rule.fusion_channels(uncoupled[0], uncoupled[1])[0];
    FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &rule,
            uncoupled,
            coupled,
            [false, false],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(&rule, [], rule.vacuum(), [], [], []).unwrap(),
    )
}

#[test]
fn unique_production_valid_groups_are_singleton_and_preserve_source_order() {
    let rule = Z2FusionRule;
    let odd = SectorId::new(1);
    let group_a = z2_two_leg_pair_with_empty_domain([odd, odd]);
    let group_b = z2_two_leg_pair_with_empty_domain([SectorId::new(0), SectorId::new(0)]);
    let split_group = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(&rule, [odd], odd, [false], [], []).unwrap(),
        FusionTreeKey::try_new_for_rule(&rule, [odd], odd, [false], [], []).unwrap(),
    );
    let source_order = [group_a, group_b, split_group];
    let source_structure =
        packed_fixture_structure(2, source_order.iter().cloned().map(|key| (key, vec![1, 1])))
            .unwrap();
    // What: canonical UniqueFusion admits exactly one tree for each external
    // group, so an interleaved same-group fixture is not constructible.
    assert!(source_structure
        .fusion_tree_groups()
        .iter()
        .all(|group| group.block_indices().len() == 1));

    let plan = build_tree_pair_transform_group_plan(
        &rule,
        TreeTransformOperation::permute([0], [1]),
        &source_structure,
    )
    .unwrap();

    // What: the Unique/Abelian transformer emits one spec per canonical source
    // block in raw source order.
    assert_eq!(
        plan.specs()
            .iter()
            .map(|spec| spec.src_keys()[0].clone())
            .collect::<Vec<_>>(),
        source_order
    );
}

#[test]
fn unique_production_prepares_each_distinct_source_split() {
    let rule = Z2FusionRule;
    let vacuum = rule.vacuum();
    let odd = SectorId::new(1);
    let split_one_one = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(&rule, [odd], odd, [false], [], []).unwrap(),
        FusionTreeKey::try_new_for_rule(&rule, [odd], odd, [false], [], []).unwrap(),
    );
    let split_two_zero = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &rule,
            [odd, odd],
            vacuum,
            [false, false],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(&rule, [], vacuum, [], [], []).unwrap(),
    );
    let source_order = [split_one_one, split_two_zero];
    let source_structure =
        packed_fixture_structure(2, source_order.iter().cloned().map(|key| (key, vec![1, 1])))
            .unwrap();
    let operation = TreeTransformOperation::permute([1], [0]);

    let specialized =
        build_tree_pair_transform_group_plan(&rule, operation.clone(), &source_structure).unwrap();
    let generic =
        build_multiplicity_free_tree_pair_transform_group_plan(&rule, operation, &source_structure)
            .unwrap();

    // What: one nontrivial operation-local inverse table is interpreted
    // against both source splits, preserving the generic destination,
    // coefficient, and raw source order under the prepared Unique path.
    assert_eq!(specialized, generic);
    assert_eq!(
        specialized
            .specs()
            .iter()
            .map(|spec| spec.src_keys()[0].clone())
            .collect::<Vec<_>>(),
        source_order
    );
}

#[test]
fn unique_production_preserves_generic_noncategorical_error_precedence() {
    let dense = BlockStructure::trivial(&[1, 1]).unwrap();
    let opaque = packed_fixture_structure(2, [(BlockKey::opaque([7]), vec![1, 1])]).unwrap();
    let malformed = TreeTransformOperation::permute([0, 0], []);

    for structure in [&dense, &opaque] {
        // What: Unique and general multiplicity-free builders both report
        // malformed syntax before either noncategorical namespace.
        assert_eq!(
            build_tree_pair_transform_group_plan(&Z2FusionRule, malformed.clone(), structure,),
            build_multiplicity_free_tree_pair_transform_group_plan(
                &Z2FusionRule,
                malformed.clone(),
                structure,
            ),
        );
    }
}
