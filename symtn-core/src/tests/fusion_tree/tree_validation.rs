use super::*;

#[test]
fn rule_aware_tree_validation_covers_rank_shape_style_and_vertices() {
    // What: categorical interpretation requires the canonical empty-tree
    // vacuum and rejects every representable malformed tree field.
    let empty_vacuum = FusionTreeKey::new([], z2_even(), [], [], []);
    empty_vacuum.validate_for_rule(&Z2FusionRule).unwrap();
    assert_eq!(
        FusionTreeKey::new([], z2_odd(), [], [], [])
            .validate_for_rule(&Z2FusionRule)
            .unwrap_err(),
        CoreError::MalformedFusionTree {
            message: "rank-0 fusion tree coupled sector must equal the vacuum",
        }
    );

    let rank_one = FusionTreeKey::new([z2_odd()], z2_odd(), [true], [], []);
    rank_one.validate_for_rule(&Z2FusionRule).unwrap();
    assert_eq!(
        FusionTreeKey::new([z2_odd()], z2_even(), [true], [], [])
            .validate_for_rule(&Z2FusionRule)
            .unwrap_err(),
        CoreError::MalformedFusionTree {
            message: "rank-1 fusion tree coupled sector must equal its uncoupled sector",
        }
    );

    let bad_shapes = [
        (
            FusionTreeKey::new(
                [z2_odd(), z2_odd()],
                z2_even(),
                [false],
                [],
                [MultiplicityIndex::ONE],
            ),
            "fusion tree sectors and duality flags must have matching length",
        ),
        (
            FusionTreeKey::new(
                [z2_odd(), z2_odd(), z2_odd()],
                z2_odd(),
                [false; 3],
                [],
                [MultiplicityIndex::ONE; 2],
            ),
            "fusion tree has an invalid number of innerlines",
        ),
        (
            FusionTreeKey::new([z2_odd(), z2_odd()], z2_even(), [false; 2], [], []),
            "fusion tree has an invalid number of vertices",
        ),
    ];
    for (tree, message) in bad_shapes {
        assert_eq!(
            tree.validate_for_rule(&Z2FusionRule).unwrap_err(),
            CoreError::MalformedFusionTree { message }
        );
    }

    for tree in [
        FusionTreeKey::new(
            [z2_odd(); 4],
            z2_even(),
            [false; 4],
            [z2_even(), z2_even()],
            [MultiplicityIndex::ONE; 3],
        ),
        FusionTreeKey::new(
            [z2_odd(); 4],
            z2_odd(),
            [false; 4],
            [z2_even(), z2_odd()],
            [MultiplicityIndex::ONE; 3],
        ),
    ] {
        assert_eq!(
            tree.validate_for_rule(&Z2FusionRule).unwrap_err(),
            CoreError::MalformedFusionTree {
                message: "fusion tree contains an inadmissible fusion vertex",
            }
        );
    }
}

#[test]
fn rule_aware_constructor_enforces_one_based_vertex_bounds() {
    // What: vertex labels are checked against the actual N-symbol for
    // both multiplicity-free and Generic rules.
    assert_eq!(
        FusionTreeKey::try_new_for_rule(
            &Z2FusionRule,
            [z2_odd(), z2_odd()],
            z2_even(),
            [false; 2],
            [],
            [MultiplicityIndex::new(2).expect("test multiplicity label is one-based")],
        )
        .unwrap_err(),
        CoreError::MalformedFusionTree {
            message: "fusion tree vertex label exceeds its fusion multiplicity",
        }
    );
    assert_eq!(
        MultiplicityIndex::try_from(0).unwrap_err(),
        CoreError::InvalidMultiplicityIndex { value: 0 }
    );
    assert_eq!(
        FusionTreeKey::try_new_for_rule(
            &ToyOmRule,
            [SectorId::new(ToyOmRule::A), SectorId::new(ToyOmRule::A)],
            SectorId::new(ToyOmRule::C),
            [false; 2],
            [],
            [MultiplicityIndex::new(3).unwrap()],
        )
        .unwrap_err(),
        CoreError::MalformedFusionTree {
            message: "fusion tree vertex label exceeds its fusion multiplicity",
        }
    );
}

#[test]
fn checked_tree_constructors_report_builtin_closure_without_unwinding() {
    // What: raw checked construction preserves exact finite-algebra
    // failures instead of entering the infallible provider path.
    assert_eq!(
        FusionTreeKey::try_new_for_rule_checked(
            &U1FusionRule,
            [u1(i32::MAX), u1(1)],
            u1(0),
            [false; 2],
            [],
            [MultiplicityIndex::ONE],
        ),
        Err(CheckedFusionSpaceError::FusionAlgebra(Box::new(
            FusionAlgebraError::U1FusionOverflow {
                left: i32::MAX,
                right: 1,
            }
        )))
    );
    assert_eq!(
        FusionTreeKey::try_from_sector_ids_for_rule_checked(
            &SU2FusionRule,
            [128, 127],
            0,
            [false; 2],
            [],
            [1],
        ),
        Err(CheckedFusionSpaceError::FusionAlgebra(Box::new(
            FusionAlgebraError::FusionNotRepresentable {
                left: su2(128),
                right: su2(127),
            }
        )))
    );
}

#[test]
fn checked_tree_distinguishes_absent_channels_from_algebra_failure() {
    // What: a representable but mathematically absent stored channel is a
    // malformed tree, while invalid stored IDs and recursive product
    // closure keep their exact algebra causes.
    assert_eq!(
        FusionTreeKey::new(
            [u1(1), u1(2)],
            u1(4),
            [false; 2],
            [],
            [MultiplicityIndex::ONE],
        )
        .validate_for_rule_checked(&U1FusionRule),
        Err(CheckedFusionSpaceError::Core(Box::new(
            CoreError::MalformedFusionTree {
                message: "fusion tree contains an inadmissible fusion vertex",
            }
        )))
    );
    let invalid = SectorId::new(2);
    assert_eq!(
        FusionTreeKey::new(
            [z2_odd(), z2_odd()],
            invalid,
            [false; 2],
            [],
            [MultiplicityIndex::ONE],
        )
        .validate_for_rule_checked(&Z2FusionRule),
        Err(CheckedFusionSpaceError::FusionAlgebra(Box::new(
            FusionAlgebraError::InvalidSector { sector: invalid }
        )))
    );

    #[cfg(target_pointer_width = "64")]
    {
        type Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
        type Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule, Codec>;
        let rule = Rule::try_new(FermionParityFusionRule, U1FusionRule).unwrap();
        let left = Codec::encode(z2_even(), u1(i32::MAX));
        let right = Codec::encode(z2_odd(), u1(1));
        let coupled = Codec::encode(z2_odd(), u1(0));
        assert_eq!(
            FusionTreeKey::new(
                [left, right],
                coupled,
                [false; 2],
                [],
                [MultiplicityIndex::ONE],
            )
            .validate_for_rule_checked(&rule),
            Err(CheckedFusionSpaceError::FusionAlgebra(Box::new(
                FusionAlgebraError::U1FusionOverflow {
                    left: i32::MAX,
                    right: 1,
                }
            )))
        );
    }
}

fn assert_checked_tree_matches_infallible<R>(rule: &R, tree: FusionTreeKey)
where
    R: CheckedFusionAlgebra,
{
    tree.validate_for_rule(rule).unwrap();
    tree.validate_for_rule_checked(rule).unwrap();
}

#[test]
fn checked_tree_matches_closed_builtin_and_nested_product_rules() {
    // What: checked validation accepts the same valid raw trees as the
    // established validator across built-in and recursive product rules.
    assert_checked_tree_matches_infallible(
        &Z2FusionRule,
        FusionTreeKey::new(
            [z2_odd(); 2],
            z2_even(),
            [false; 2],
            [],
            [MultiplicityIndex::ONE],
        ),
    );
    assert_checked_tree_matches_infallible(
        &FermionParityFusionRule,
        FusionTreeKey::new(
            [z2_odd(); 2],
            z2_even(),
            [false; 2],
            [],
            [MultiplicityIndex::ONE],
        ),
    );
    assert_checked_tree_matches_infallible(
        &U1FusionRule,
        FusionTreeKey::new(
            [u1(2), u1(-1)],
            u1(1),
            [false; 2],
            [],
            [MultiplicityIndex::ONE],
        ),
    );
    assert_checked_tree_matches_infallible(
        &SU2FusionRule,
        FusionTreeKey::new(
            [su2(1); 2],
            su2(0),
            [false; 2],
            [],
            [MultiplicityIndex::ONE],
        ),
    );
    assert_checked_tree_matches_infallible(
        &TableRule::fibonacci(),
        FusionTreeKey::new(
            [SectorId::new(1); 2],
            SectorId::new(1),
            [false; 2],
            [],
            [MultiplicityIndex::ONE],
        ),
    );

    #[cfg(target_pointer_width = "64")]
    {
        type InnerCodec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
        type InnerRule = ProductFusionRule<FermionParityFusionRule, U1FusionRule, InnerCodec>;
        type InnerLayout = ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>;
        type Codec = PackedProductCodec<InnerLayout, Su2SectorLayout>;
        type Rule = ProductFusionRule<InnerRule, SU2FusionRule, Codec>;
        let rule = Rule::try_new(
            InnerRule::try_new(FermionParityFusionRule, U1FusionRule).unwrap(),
            SU2FusionRule,
        )
        .unwrap();
        let sector = |parity, charge, spin| Codec::encode(InnerCodec::encode(parity, charge), spin);
        assert_checked_tree_matches_infallible(
            &rule,
            FusionTreeKey::new(
                [
                    sector(z2_odd(), u1(2), su2(1)),
                    sector(z2_odd(), u1(-1), su2(1)),
                ],
                sector(z2_even(), u1(1), su2(0)),
                [false; 2],
                [],
                [MultiplicityIndex::ONE],
            ),
        );
    }

    assert_checked_tree_matches_infallible(
        &CheckedTreeProbe::default(),
        FusionTreeKey::new(
            [SectorId::new(0); 2],
            SectorId::new(0),
            [false; 2],
            [],
            [MultiplicityIndex::ONE],
        ),
    );
}

#[test]
fn tree_pair_validation_requires_exact_coupled_sector() {
    // What: pair compatibility compares the same coupled sector on both
    // sides, including the canonical rank-zero vacuum.
    let empty_pair = FusionTreePairKey::pair(
        FusionTreeKey::new([], u1(0), [], [], []),
        FusionTreeKey::new([], u1(0), [], [], []),
    );
    empty_pair.validate_for_rule(&U1FusionRule).unwrap();
    FusionTreePairKey::pair(
        FusionTreeKey::new([], u1(0), [], [], []),
        FusionTreeKey::new([u1(0)], u1(0), [false], [], []),
    )
    .validate_for_rule(&U1FusionRule)
    .unwrap();
    assert_eq!(
        FusionTreePairKey::pair(
            FusionTreeKey::new([], u1(0), [], [], []),
            FusionTreeKey::new([u1(1)], u1(1), [false], [], []),
        )
        .validate_for_rule(&U1FusionRule)
        .unwrap_err(),
        CoreError::MalformedFusionTree {
            message: "fusion tree pair requires matching coupled sectors",
        }
    );

    let codomain = FusionTreeKey::try_new_for_rule(
        &U1FusionRule,
        [u1(1), u1(2)],
        u1(3),
        [false; 2],
        [],
        [MultiplicityIndex::ONE],
    )
    .unwrap();
    let same_c_domain = FusionTreeKey::try_new_for_rule(
        &U1FusionRule,
        [u1(4), u1(-1)],
        u1(3),
        [true, false],
        [],
        [MultiplicityIndex::ONE],
    )
    .unwrap();
    FusionTreePairKey::pair(codomain.clone(), same_c_domain)
        .validate_for_rule(&U1FusionRule)
        .unwrap();
    let dual_c_domain = FusionTreeKey::try_new_for_rule(
        &U1FusionRule,
        [u1(-4), u1(1)],
        u1(-3),
        [false; 2],
        [],
        [MultiplicityIndex::ONE],
    )
    .unwrap();
    assert_eq!(
        FusionTreePairKey::pair(codomain, dual_c_domain)
            .validate_for_rule(&U1FusionRule)
            .unwrap_err(),
        CoreError::MalformedFusionTree {
            message: "fusion tree pair requires matching coupled sectors",
        }
    );
}

#[test]
fn unique_proof_hook_rechecks_style_and_prepared_source_rank() {
    // What: downstream callers cannot use a general categorical proof to
    // bypass the Unique-style or prepared source-split contract.
    let vacuum = su2(0);
    let half = su2(1);
    let simple_pair = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &SU2FusionRule,
            [half, half],
            vacuum,
            [false; 2],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(&SU2FusionRule, [], vacuum, [], [], []).unwrap(),
    );
    let simple_structure = packed_fixture_structure(2, [(simple_pair, vec![1, 1])]).unwrap();
    let simple_proof =
        LocallyValidatedFusionTreeBlockStructure::try_new(&SU2FusionRule, &simple_structure)
            .unwrap();
    let rank_two_identity =
        PreparedTreePairOperation::prepare_transpose(2, 0, &[0, 1], &[]).unwrap();
    assert_eq!(
        simple_proof
            .execute_unique_rigid_for_block_index(0, &rank_two_identity, &mut Default::default())
            .unwrap_err(),
        CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Unique,
            actual: FusionStyleKind::Simple,
        }
    );

    let unique_pair = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &U1FusionRule,
            [u1(1), u1(-1)],
            u1(0),
            [false; 2],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
        FusionTreeKey::try_new_for_rule(&U1FusionRule, [], u1(0), [], [], []).unwrap(),
    );
    let unique_structure = packed_fixture_structure(2, [(unique_pair, vec![1, 1])]).unwrap();
    let unique_proof =
        LocallyValidatedFusionTreeBlockStructure::try_new(&U1FusionRule, &unique_structure)
            .unwrap();
    let rank_one_identity = PreparedTreePairOperation::prepare_transpose(1, 0, &[0], &[]).unwrap();
    assert_eq!(
        unique_proof
            .execute_unique_rigid_for_block_index(0, &rank_one_identity, &mut Default::default())
            .unwrap_err(),
        CoreError::DimensionMismatch {
            expected: 1,
            actual: 2,
        }
    );
}

#[test]
fn rule_validation_preserves_builtin_multiplicity_free_keys() {
    // What: validation is observational for representative abelian,
    // non-abelian, anyonic, fermionic, and product fusion trees.
    fn assert_preserved<R: FusionRule>(rule: &R, tree: FusionTreeKey) {
        let before = tree.clone();
        tree.validate_for_rule(rule).unwrap();
        assert_eq!(tree, before);
    }

    assert_preserved(
        &U1FusionRule,
        FusionTreeKey::try_new_for_rule(
            &U1FusionRule,
            [u1(1), u1(-2), u1(3)],
            u1(2),
            [false; 3],
            [u1(-1)],
            [MultiplicityIndex::ONE; 2],
        )
        .unwrap(),
    );
    let half = su2(1);
    let one = su2(2);
    assert_preserved(
        &SU2FusionRule,
        FusionTreeKey::try_new_for_rule(
            &SU2FusionRule,
            [half, half, one],
            one,
            [false, true, false],
            [su2(0)],
            [MultiplicityIndex::ONE; 2],
        )
        .unwrap(),
    );
    assert_preserved(
        &TableRule::fibonacci(),
        FusionTreeKey::try_new_for_rule(
            &TableRule::fibonacci(),
            [SectorId::new(1); 3],
            SectorId::new(1),
            [false; 3],
            [SectorId::new(0)],
            [MultiplicityIndex::ONE; 2],
        )
        .unwrap(),
    );
    assert_preserved(
        &FermionParityFusionRule,
        FusionTreeKey::try_new_for_rule(
            &FermionParityFusionRule,
            [z2_odd(); 3],
            z2_odd(),
            [false, true, false],
            [z2_even()],
            [MultiplicityIndex::ONE; 2],
        )
        .unwrap(),
    );

    type ProductRule = ProductFusionRule<U1FusionRule, SU2FusionRule>;
    let product = ProductRule::try_new(U1FusionRule, SU2FusionRule).unwrap();
    let left = product.encode_component_ids(u1(1), half);
    let right = product.encode_component_ids(u1(-1), half);
    let coupled = product.encode_component_ids(u1(0), su2(0));
    assert_preserved(
        &product,
        FusionTreeKey::try_new_for_rule(
            &product,
            [left, right],
            coupled,
            [false; 2],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
    );
}

#[test]
fn public_operation_errors_precede_categorical_source_errors() {
    // What: deterministic argument and capability failures retain their
    // public precedence over a malformed source.
    let tau = SectorId::new(1);
    let vacuum = SectorId::new(0);
    let invalid = FusionTreeKey::new(
        [tau; 3],
        vacuum,
        [false; 3],
        [vacuum],
        [MultiplicityIndex::ONE; 2],
    );
    assert_eq!(
        multiplicity_free_braid_tree(&TableRule::fibonacci(), &invalid, &[0, 1, 2], &[],)
            .unwrap_err(),
        CoreError::DimensionMismatch {
            expected: 3,
            actual: 0,
        }
    );
    assert!(matches!(
        multiplicity_free_braid_tree(&TableRule::fibonacci(), &invalid, &[0, 0, 2], &[0, 1, 2],),
        Err(CoreError::InvalidPermutation { .. })
    ));
    // Not asserted here: `unique_braid_tree`'s own rejection of a Simple
    // provider. That entry point is gone and
    // `multiplicity_free_braid_tree` accepts Simple providers by design.
    // The Unique-only style guard itself still exists on the surviving
    // route and is asserted elsewhere — `execute_unique_rigid` at
    // `fusion_tree.rs`, exercised with a Simple rule below in this file.
    assert!(matches!(
        multiplicity_free_permute_tree(&TableRule::fibonacci(), &invalid, &[0, 1, 2],),
        Err(CoreError::UnsupportedBraidingStyle { .. })
    ));
    assert_eq!(
        multiplicity_free_braid_tree_block(
            &TableRule::fibonacci(),
            std::slice::from_ref(&invalid),
            &[0, 1, 2],
            &[],
        )
        .unwrap_err(),
        CoreError::DimensionMismatch {
            expected: 3,
            actual: 0,
        }
    );

    let pair = FusionTreePairKey::pair(invalid, FusionTreeKey::new([], vacuum, [], [], []));
    let wrong_rank = PreparedTreePairOperation::prepare_braid(
        &TableRule::fibonacci(),
        2,
        0,
        &[0, 1],
        &[],
        &[0, 1],
        &[],
    )
    .unwrap();
    assert_eq!(
        wrong_rank
            .execute_multiplicity_free(&TableRule::fibonacci(), &pair)
            .unwrap_err(),
        CoreError::DimensionMismatch {
            expected: 2,
            actual: 3,
        }
    );
}

#[test]
fn unique_direct_braid_eligibility_excludes_rank_zero() {
    // What: the canonical empty tree carries the vacuum, but no rank-zero
    // operation is eligible for a direct braid rebuild.
    let empty =
        FusionTreeKey::try_new_for_rule(&Z2FusionRule, [], Z2FusionRule.vacuum(), [], [], [])
            .unwrap();

    assert_eq!(empty.coupled(), Z2FusionRule.vacuum());
    assert!(!is_unique_direct_braid_source(
        &Z2FusionRule,
        empty.as_view()
    ));
}

#[test]
fn uncertified_custom_symbols_stay_on_general_artin_path() {
    // What: the public provider trait does not runtime-check coherence, so
    // arbitrary custom F/R data must remain on the default-false Artin
    // boundary instead of claiming TensorKit's certified F=1 shortcut.
    let rule = UncertifiedCustomSymbolsRule;
    assert!(!rule.has_trivial_associator_gauge());
    let tree = FusionTreeKey::try_from_sector_ids([1, 1, 1], 1, [false, false, false], [0], [1, 1])
        .unwrap();

    let terms = multiplicity_free_braid_tree(&rule, &tree, &[0, 2, 1], &[0, 1, 2]).unwrap();
    assert_eq!(terms.len(), 1);
    let (destination, coefficient) = terms.into_iter().next().unwrap();

    let mut expected_tree = UnhashedFusionTree::from(tree.clone());
    let expected_coefficient =
        apply_unique_artin_braid_at_with_inverse(&rule, &mut expected_tree, 1, false).unwrap();

    assert_eq!(
        (destination, coefficient),
        (expected_tree.freeze(), expected_coefficient)
    );
    assert_eq!(coefficient, 30.0);
}

#[test]
fn complex_unique_prepared_steps_keep_asymmetric_inverse_orientation() {
    // What: prepared Artin steps retain Complex64 conjugation and reverse
    // R-symbol argument order for an inverse anyonic crossing.
    let rule = ComplexAsymmetricUniqueRule;
    let tree = FusionTreeKey::try_from_sector_ids([1, 2], 3, [false, true], [], [1]).unwrap();

    let forward_terms = multiplicity_free_braid_tree(&rule, &tree, &[1, 0], &[0, 1]).unwrap();
    assert_eq!(forward_terms.len(), 1);
    let forward = forward_terms.into_iter().next().unwrap();
    let inverse_terms = multiplicity_free_braid_tree(&rule, &tree, &[1, 0], &[1, 0]).unwrap();
    assert_eq!(inverse_terms.len(), 1);
    let inverse = inverse_terms.into_iter().next().unwrap();
    let expected_forward = Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_3);
    let expected_inverse = Complex64::from_polar(1.0, -std::f64::consts::FRAC_PI_6);

    assert_eq!(forward.0, inverse.0);
    assert!((forward.1 - expected_forward).norm() < 1.0e-12);
    assert!((inverse.1 - expected_inverse).norm() < 1.0e-12);
}

#[test]
fn prepared_tree_pair_operation_size_excludes_a_second_step_arena() {
    // What: the expert plan stores one inline Artin lowering, not parallel
    // Artin and inversion arrays for mutually exclusive execution paths.
    assert!(std::mem::size_of::<PreparedTreePairOperation<'static>>() <= 600);
}

#[test]
fn unique_braid_tree_rejects_invalid_permutation_and_level_count() {
    let tree = FusionTreeKey::try_from_sector_ids([1, 2], 3, [false, false], [], [1]).unwrap();

    let err =
        multiplicity_free_braid_tree(&AsymmetricAnyonicRule, &tree, &[1, 1], &[0, 1]).unwrap_err();
    assert_eq!(
        err,
        CoreError::InvalidPermutation {
            permutation: vec![1, 1],
            rank: 2,
        }
    );

    let err =
        multiplicity_free_braid_tree(&AsymmetricAnyonicRule, &tree, &[1, 0], &[0]).unwrap_err();
    assert_eq!(
        err,
        CoreError::DimensionMismatch {
            expected: 2,
            actual: 1,
        }
    );
}

#[test]
fn unique_permute_tree_requires_symmetric_braiding() {
    let tree = FusionTreeKey::try_from_sector_ids([1, 2], 3, [false, false], [], [1]).unwrap();

    let err = unique_permute_tree(&AsymmetricAnyonicRule, &tree, &[1, 0]).unwrap_err();

    assert_eq!(
        err,
        CoreError::UnsupportedBraidingStyle {
            expected: "symmetric braiding",
            actual: BraidingStyleKind::Anyonic,
        }
    );
}
