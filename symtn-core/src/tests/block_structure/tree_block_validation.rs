use super::*;

#[test]
fn coupled_layout_embeds_subblocks_into_sector_matrices() {
    let rule = Z2FusionRule;
    let leg =
        |degeneracy, dual| SectorLeg::new([(z2_even(), degeneracy), (z2_odd(), degeneracy)], dual);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(2, false), leg(3, false)]),
        FusionProductSpace::new([leg(2, false), leg(3, false)]),
    );
    let keys = homspace.fusion_tree_keys(&rule);
    let shapes = keys
        .iter()
        .map(|_| vec![2usize, 3, 2, 3])
        .collect::<Vec<_>>();
    let packed = homspace
        .clone()
        .coupled_subblock_structure(&rule, 2, shapes.clone())
        .unwrap();
    let coupled = homspace
        .coupled_subblock_structure(&rule, 2, shapes)
        .unwrap();

    let packed_structure = &packed;
    let coupled_structure = &coupled;
    assert_eq!(
        packed_structure.block_count(),
        coupled_structure.block_count()
    );
    assert_eq!(
        packed_structure.required_len().unwrap(),
        coupled_structure.required_len().unwrap()
    );

    // Two coupled sectors (even/odd), each with two codomain and two
    // domain trees of subblock row dim 6 and column dim 6: sector matrices
    // are 12 x 12.
    let matrix_rows = 12usize;
    let mut covered = vec![false; coupled_structure.required_len().unwrap()];
    for index in 0..coupled_structure.block_count() {
        let packed_block = packed_structure.block(index).unwrap();
        let coupled_block = coupled_structure.block(index).unwrap();
        assert_eq!(packed_block.key(), coupled_block.key());
        assert_eq!(packed_block.shape(), coupled_block.shape());
        // Codomain legs stay column-major inside the row block; domain
        // legs step whole matrix columns.
        assert_eq!(coupled_block.strides()[0], 1);
        assert_eq!(coupled_block.strides()[1], 2);
        assert_eq!(coupled_block.strides()[2], matrix_rows);
        assert_eq!(coupled_block.strides()[3], matrix_rows * 2);
        for i3 in 0..3 {
            for i2 in 0..2 {
                for i1 in 0..3 {
                    for i0 in 0..2 {
                        let strides = coupled_block.strides();
                        let position = coupled_block.offset()
                            + i0 * strides[0]
                            + i1 * strides[1]
                            + i2 * strides[2]
                            + i3 * strides[3];
                        assert!(
                            !covered[position],
                            "coupled layout must not overlap between subblocks"
                        );
                        covered[position] = true;
                    }
                }
            }
        }
    }
    assert!(
        covered.iter().all(|&flag| flag),
        "coupled layout must cover the sector matrices without holes"
    );
}

#[derive(Clone, Copy, Debug)]
struct MisreportedGenericMultiplicityFreeRule;

impl FusionRule for MisreportedGenericMultiplicityFreeRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Generic
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        IdentitySymbolPanicRule.vacuum()
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        IdentitySymbolPanicRule.fusion_channels(left, right)
    }
}

impl MultiplicityFreeFusionRule for MisreportedGenericMultiplicityFreeRule {}

impl MultiplicityFreeFusionSymbols for MisreportedGenericMultiplicityFreeRule {
    type Scalar = f64;

    fn f_symbol_scalar(
        &self,
        _left: SectorId,
        _middle: SectorId,
        _right: SectorId,
        _coupled: SectorId,
        _left_coupled: SectorId,
        _right_coupled: SectorId,
    ) -> Self::Scalar {
        panic!("misreported Generic provider evaluated an F symbol")
    }

    fn r_symbol_scalar(
        &self,
        _left: SectorId,
        _right: SectorId,
        _coupled: SectorId,
    ) -> Self::Scalar {
        panic!("misreported Generic provider evaluated an R symbol")
    }
}

impl MultiplicityFreeRigidSymbols for MisreportedGenericMultiplicityFreeRule {
    fn dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn inv_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn inv_sqrt_dim_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn twist_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }

    fn frobenius_schur_phase_scalar(&self, _sector: SectorId) -> Self::Scalar {
        1.0
    }
}

#[test]
fn unique_artin_braid_at_rejects_out_of_range_index() {
    let tree = FusionTreeKey::try_from_sector_ids([1, 1], 0, [false, false], [], [1]).unwrap();

    // What: index 1 is out of range for a rank-2 tree; a swap starting at
    // that index has no adjacent partner, so a permutation cannot express
    // it either. Repoint at the live single-step primitive that the
    // surviving multiplicity-free braid decomposes each Artin swap into.
    let err = apply_unique_artin_braid_at_with_inverse(
        &FermionParityFusionRule,
        &mut UnhashedFusionTree::from(tree.clone()),
        1,
        false,
    )
    .unwrap_err();

    assert_eq!(err, CoreError::InvalidBraidIndex { index: 1, rank: 2 });
}

#[test]
fn rule_aware_constructor_rejects_inadmissible_unique_innerline() {
    // What: a rule-aware constructor rejects a stored vertex that is
    // outside the Unique rule's fusion graph.
    assert_eq!(
        FusionTreeKey::try_new_for_rule(
            &FermionParityFusionRule,
            [SectorId::new(1), SectorId::new(1), SectorId::new(1)],
            SectorId::new(1),
            [false, false, false],
            [SectorId::new(1)],
            [MultiplicityIndex::ONE, MultiplicityIndex::ONE],
        )
        .unwrap_err(),
        CoreError::MalformedFusionTree {
            message: "fusion tree contains an inadmissible fusion vertex",
        }
    );
}

fn assert_invalid_rank_one_checked<R>(rule: &R, invalid: SectorId, expected: FusionAlgebraError)
where
    R: CheckedFusionAlgebra,
{
    let tree = FusionTreeKey::new([invalid], invalid, [false], [], []);
    assert_eq!(
        tree.validate_for_rule_checked(rule),
        Err(CheckedFusionSpaceError::FusionAlgebra(Box::new(expected)))
    );
}

#[test]
fn checked_rank_one_validates_ids_by_unit_fusion() {
    // What: rank-one raw imports reject every unrepresentable built-in ID,
    // including the zigzag ID reserved for the excluded U1 charge.
    let invalid = SectorId::new(2);
    assert_invalid_rank_one_checked(
        &Z2FusionRule,
        invalid,
        FusionAlgebraError::InvalidSector { sector: invalid },
    );
    assert_invalid_rank_one_checked(
        &FermionParityFusionRule,
        invalid,
        FusionAlgebraError::InvalidSector { sector: invalid },
    );
    assert_invalid_rank_one_checked(
        &TableRule::fibonacci(),
        invalid,
        FusionAlgebraError::InvalidSector { sector: invalid },
    );
    let invalid_su2 = SectorId::new(255);
    assert_invalid_rank_one_checked(
        &SU2FusionRule,
        invalid_su2,
        FusionAlgebraError::InvalidSector {
            sector: invalid_su2,
        },
    );
    #[cfg(target_pointer_width = "64")]
    {
        let invalid_u1 = SectorId::new(u32::MAX as usize + 1);
        assert_invalid_rank_one_checked(
            &U1FusionRule,
            invalid_u1,
            FusionAlgebraError::InvalidSector { sector: invalid_u1 },
        );

        type Layout = ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>;
        type Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
        type Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule, Codec>;
        let rule = Rule::new(FermionParityFusionRule, U1FusionRule);
        let invalid_product = SectorId::new(1usize << Layout::BITS);
        assert_invalid_rank_one_checked(
            &rule,
            invalid_product,
            FusionAlgebraError::ProductCodec(ProductSectorCodecError::InvalidHighBits {
                sector: invalid_product,
                total_bits: Layout::BITS,
            }),
        );
    }

    let invalid_u1 = excluded_u1_id();
    assert_invalid_rank_one_checked(
        &U1FusionRule,
        invalid_u1,
        FusionAlgebraError::InvalidSector { sector: invalid_u1 },
    );
}

#[test]
fn checked_tree_normalizes_structure_before_provider_calls() {
    // What: shape and rank errors touch no checked provider operation, and
    // generated-channel failure precedes stored multiplicity validation.
    let rule = CheckedTreeProbe::default();
    for tree in [
        FusionTreeKey::new(
            [SectorId::new(0); 2],
            SectorId::new(0),
            [false],
            [],
            [MultiplicityIndex::ONE],
        ),
        FusionTreeKey::new([SectorId::new(0)], SectorId::new(1), [false], [], []),
    ] {
        assert!(matches!(
            tree.validate_for_rule_checked(&rule),
            Err(CheckedFusionSpaceError::Core(_))
        ));
    }
    assert_eq!(rule.channel_calls.load(Ordering::Relaxed), 0);
    assert_eq!(rule.nsymbol_calls.load(Ordering::Relaxed), 0);

    let failure = FusionTreeKey::new(
        [SectorId::new(9), SectorId::new(0)],
        SectorId::new(0),
        [false; 2],
        [],
        [MultiplicityIndex::new(2).unwrap()],
    )
    .validate_for_rule_checked(&rule);
    assert_eq!(
        failure,
        Err(CheckedFusionSpaceError::FusionAlgebra(Box::new(
            FusionAlgebraError::FusionNotRepresentable {
                left: SectorId::new(9),
                right: SectorId::new(0),
            }
        )))
    );
    assert_eq!(rule.channel_calls.load(Ordering::Relaxed), 1);
    assert_eq!(rule.nsymbol_calls.load(Ordering::Relaxed), 0);

    rule.channel_calls.store(0, Ordering::Relaxed);
    rule.nsymbol_calls.store(0, Ordering::Relaxed);
    assert_eq!(
        FusionTreeKey::new(
            [SectorId::new(0); 2],
            SectorId::new(0),
            [false; 2],
            [],
            [MultiplicityIndex::new(2).unwrap()],
        )
        .validate_for_rule_checked(&rule),
        Err(CheckedFusionSpaceError::Core(Box::new(
            CoreError::MalformedFusionTree {
                message: "fusion tree vertex label exceeds its fusion multiplicity",
            }
        )))
    );
    assert_eq!(rule.channel_calls.load(Ordering::Relaxed), 1);
    assert_eq!(rule.nsymbol_calls.load(Ordering::Relaxed), 1);
}

#[test]
fn checked_tree_pair_preserves_validation_precedence() {
    // What: pair validation reports codomain, then domain, then coupled
    // mismatch without reordering finite-algebra checks.
    let bad_shape = FusionTreeKey::new([u1(0); 2], u1(0), [false], [], [MultiplicityIndex::ONE]);
    let overflow = FusionTreeKey::new(
        [u1(i32::MAX), u1(1)],
        u1(0),
        [false; 2],
        [],
        [MultiplicityIndex::ONE],
    );
    assert!(matches!(
        FusionTreePairKey::pair(bad_shape, overflow.clone())
            .validate_for_rule_checked(&U1FusionRule),
        Err(CheckedFusionSpaceError::Core(_))
    ));
    assert_eq!(
        FusionTreePairKey::pair(FusionTreeKey::new([], u1(0), [], [], []), overflow,)
            .validate_for_rule_checked(&U1FusionRule),
        Err(CheckedFusionSpaceError::FusionAlgebra(Box::new(
            FusionAlgebraError::U1FusionOverflow {
                left: i32::MAX,
                right: 1,
            }
        )))
    );
    assert_eq!(
        FusionTreePairKey::pair(
            FusionTreeKey::new([u1(1)], u1(1), [false], [], []),
            FusionTreeKey::new([u1(2)], u1(2), [false], [], []),
        )
        .validate_for_rule_checked(&U1FusionRule),
        Err(CheckedFusionSpaceError::Core(Box::new(
            CoreError::MalformedFusionTree {
                message: "fusion tree pair requires matching coupled sectors",
            }
        )))
    );
}

#[test]
fn fusion_subset_structural_proof_precedes_algebra_and_preserves_unchecked_work() {
    // What: HomSpace metadata is proved without provider calls, while the
    // categorical phase makes exactly the local validator's provider calls.
    let scalar = FusionTreeKey::new([], SectorId::new(0), [], [], []);
    let malformed = FusionTreeKey::new(
        [SectorId::new(0); 2],
        SectorId::new(0),
        [false],
        [],
        [MultiplicityIndex::ONE],
    );
    let malformed_structure = packed_fixture_structure(
        2,
        [(
            FusionTreePairKey::pair(malformed, scalar.clone()),
            vec![1, 1],
        )],
    )
    .unwrap();
    let homspace = FusionTreeHomSpace::from_sector_ids([(0, 1), (0, 1)], []);
    let checked = CheckedTreeProbe::default();
    assert!(matches!(
        homspace.validate_subblock_structure_subset_checked(&checked, &malformed_structure,),
        Err(CheckedFusionSpaceError::Core(_))
    ));
    assert_eq!(checked.channel_calls.load(Ordering::Relaxed), 0);
    assert_eq!(checked.nsymbol_calls.load(Ordering::Relaxed), 0);

    let valid = FusionTreeKey::new(
        [SectorId::new(0); 2],
        SectorId::new(0),
        [false; 2],
        [],
        [MultiplicityIndex::ONE],
    );
    let coupling_probe = CheckedTreeProbe::default();
    let mismatched_structure = packed_fixture_structure(
        2,
        [
            (
                FusionTreePairKey::pair(valid.clone(), scalar.clone()),
                vec![1, 1],
            ),
            (
                FusionTreePairKey::pair(
                    valid.clone(),
                    FusionTreeKey::new([], SectorId::new(1), [], [], []),
                ),
                vec![1, 1],
            ),
        ],
    )
    .unwrap();
    assert!(matches!(
        homspace
            .validate_subblock_structure_subset_checked(&coupling_probe, &mismatched_structure,),
        Err(CheckedFusionSpaceError::Core(_))
    ));
    assert_eq!(coupling_probe.channel_calls.load(Ordering::Relaxed), 0);
    assert_eq!(coupling_probe.nsymbol_calls.load(Ordering::Relaxed), 0);

    let structure =
        packed_fixture_structure(2, [(FusionTreePairKey::pair(valid, scalar), vec![1, 1])])
            .unwrap();
    let direct = CheckedTreeProbe::default();
    LocallyValidatedFusionTreeBlockStructure::try_new(&direct, &structure).unwrap();
    let expected_calls = direct.unchecked_nsymbol_calls.load(Ordering::Relaxed);
    let admitted = CheckedTreeProbe::default();
    homspace
        .validate_subblock_structure_subset(&admitted, &structure)
        .unwrap();
    assert!(expected_calls > 0);
    assert_eq!(
        admitted.unchecked_nsymbol_calls.load(Ordering::Relaxed),
        expected_calls
    );
}

#[test]
fn checked_layout_and_space_admission_reject_finite_nonclosure_transactionally() {
    // What: caller-order layout validation and same-rule unchecked admission
    // surface the exact finite-algebra error without publishing a new stamp.
    let scalar_zero = FusionTreeKey::new([], SectorId::new(0), [], [], []);
    let valid = FusionTreeKey::new(
        [SectorId::new(0); 2],
        SectorId::new(0),
        [false; 2],
        [],
        [MultiplicityIndex::ONE],
    );
    let malformed = FusionTreeKey::new(
        [SectorId::new(0); 2],
        SectorId::new(0),
        [false],
        [],
        [MultiplicityIndex::ONE],
    );
    let structural_probe = CheckedTreeProbe::default();
    assert!(matches!(
        BlockStructure::coupled_sector_matrix_with_keys_checked(
            &structural_probe,
            2,
            2,
            vec![
                (
                    FusionTreePairKey::pair(valid.clone(), scalar_zero.clone()),
                    vec![1, 1],
                ),
                (
                    FusionTreePairKey::pair(malformed, scalar_zero.clone()),
                    vec![1, 1],
                ),
            ],
        ),
        Err(CheckedFusionSpaceError::Core(_))
    ));
    assert_eq!(structural_probe.channel_calls.load(Ordering::Relaxed), 0);
    assert_eq!(structural_probe.nsymbol_calls.load(Ordering::Relaxed), 0);

    let split_probe = CheckedTreeProbe::default();
    assert_eq!(
        BlockStructure::coupled_sector_matrix_with_keys_checked(
            &split_probe,
            3,
            2,
            vec![(
                FusionTreePairKey::pair(valid.clone(), scalar_zero.clone()),
                vec![1, 1],
            )],
        ),
        Err(CheckedFusionSpaceError::Core(Box::new(
            CoreError::StructureRankMismatch {
                expected: 2,
                actual: 3,
            },
        )))
    );
    assert_eq!(split_probe.channel_calls.load(Ordering::Relaxed), 0);
    assert_eq!(split_probe.nsymbol_calls.load(Ordering::Relaxed), 0);
    assert_eq!(
        BlockStructure::coupled_sector_matrix_with_keys(
            &CheckedTreeProbe::default(),
            3,
            2,
            vec![(FusionTreePairKey::pair(valid, scalar_zero), vec![1, 1])],
        ),
        Err(CoreError::StructureRankMismatch {
            expected: 2,
            actual: 3,
        })
    );

    let overflow = FusionTreeKey::new(
        [u1(i32::MAX), u1(1)],
        u1(0),
        [false; 2],
        [],
        [MultiplicityIndex::ONE],
    );
    let scalar = FusionTreeKey::new([], u1(0), [], [], []);
    let pair = FusionTreePairKey::pair(overflow, scalar);
    let expected =
        CheckedFusionSpaceError::FusionAlgebra(Box::new(FusionAlgebraError::U1FusionOverflow {
            left: i32::MAX,
            right: 1,
        }));
    assert_eq!(
        BlockStructure::coupled_sector_matrix_with_keys_checked(
            &U1FusionRule,
            2,
            2,
            vec![(pair.clone(), vec![1, 1])],
        ),
        Err(expected.clone())
    );

    let probe = CheckedTreeProbe::default();
    let failing_pair = FusionTreePairKey::pair(
        FusionTreeKey::new(
            [SectorId::new(9), SectorId::new(0)],
            SectorId::new(0),
            [false; 2],
            [],
            [MultiplicityIndex::ONE],
        ),
        FusionTreeKey::new([], SectorId::new(0), [], [], []),
    );
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(SectorId::new(9), 1)], false),
            SectorLeg::new([(SectorId::new(0), 1)], false),
        ]),
        FusionProductSpace::new([]),
    );
    let structure = packed_fixture_structure(2, [(failing_pair, vec![1, 1])]).unwrap();
    homspace
        .validate_subblock_structure_subset(&probe, &structure)
        .unwrap();
    assert_eq!(
        homspace.validate_subblock_structure_subset_checked(&probe, &structure),
        Err(CheckedFusionSpaceError::FusionAlgebra(Box::new(
            FusionAlgebraError::FusionNotRepresentable {
                left: SectorId::new(9),
                right: SectorId::new(0),
            },
        )))
    );
}

#[test]
fn checked_coupled_layout_finishes_structural_preflight_before_algebra() {
    // What: incomplete grids, conflicting row extents, and overflowing
    // dimensions fail before checked algebra and publish no structure.
    let tree = |left| {
        FusionTreeKey::new(
            [SectorId::new(left), SectorId::new(0)],
            SectorId::new(0),
            [false; 2],
            [],
            [MultiplicityIndex::ONE],
        )
    };
    let row_zero = tree(0);
    let row_one = tree(1);
    let col_zero = tree(2);
    let col_one = tree(3);

    let wrong_split = CheckedTreeProbe::default();
    let rank_one = FusionTreeKey::new([SectorId::new(0)], SectorId::new(0), [false], [], []);
    assert_eq!(
        BlockStructure::coupled_sector_matrix_with_keys_checked(
            &wrong_split,
            2,
            2,
            vec![(
                FusionTreePairKey::pair(rank_one.clone(), rank_one),
                vec![1; 2],
            )],
        ),
        Err(CheckedFusionSpaceError::Core(Box::new(
            CoreError::FusionSpaceSplitMismatch {
                expected_nout: 2,
                expected_nin: 0,
                actual_nout: 1,
                actual_nin: 1,
            },
        )))
    );
    assert_eq!(wrong_split.channel_calls.load(Ordering::Relaxed), 0);
    assert_eq!(wrong_split.nsymbol_calls.load(Ordering::Relaxed), 0);

    let missing_grid = CheckedTreeProbe::default();
    assert_eq!(
        BlockStructure::coupled_sector_matrix_with_keys_checked(
            &missing_grid,
            2,
            4,
            vec![
                (
                    FusionTreePairKey::pair(row_zero.clone(), col_zero.clone()),
                    vec![1; 4],
                ),
                (
                    FusionTreePairKey::pair(row_one, col_one.clone()),
                    vec![1; 4],
                ),
            ],
        ),
        Err(CheckedFusionSpaceError::Core(Box::new(
            CoreError::BlockCountMismatch {
                expected: 4,
                actual: 2,
            },
        )))
    );
    assert_eq!(missing_grid.channel_calls.load(Ordering::Relaxed), 0);
    assert_eq!(missing_grid.nsymbol_calls.load(Ordering::Relaxed), 0);

    let conflicting_extent = CheckedTreeProbe::default();
    assert_eq!(
        BlockStructure::coupled_sector_matrix_with_keys_checked(
            &conflicting_extent,
            2,
            4,
            vec![
                (
                    FusionTreePairKey::pair(row_zero.clone(), col_zero.clone()),
                    vec![1; 4],
                ),
                (
                    FusionTreePairKey::pair(row_zero.clone(), col_one.clone()),
                    vec![2, 1, 1, 1],
                ),
            ],
        ),
        Err(CheckedFusionSpaceError::Core(Box::new(
            CoreError::DimensionMismatch {
                expected: 1,
                actual: 2,
            },
        )))
    );
    assert_eq!(conflicting_extent.channel_calls.load(Ordering::Relaxed), 0);
    assert_eq!(conflicting_extent.nsymbol_calls.load(Ordering::Relaxed), 0);

    let overflowing_extent = CheckedTreeProbe::default();
    assert_eq!(
        BlockStructure::coupled_sector_matrix_with_keys_checked(
            &overflowing_extent,
            2,
            4,
            vec![(
                FusionTreePairKey::pair(row_zero, col_zero),
                vec![usize::MAX, 2, 1, 1],
            )],
        ),
        Err(CheckedFusionSpaceError::Core(Box::new(
            CoreError::ElementCountOverflow,
        )))
    );
    assert_eq!(overflowing_extent.channel_calls.load(Ordering::Relaxed), 0);
    assert_eq!(overflowing_extent.nsymbol_calls.load(Ordering::Relaxed), 0);
}

#[test]
fn raw_tree_pair_constructor_rejects_zero_vertices_in_source_order() {
    let domain_vertices = std::iter::once_with(|| {
        panic!("domain vertices must not be consumed after a codomain error")
    });
    let error = FusionTreePairKey::try_pair_from_sector_ids(
        [1, 1],
        [1, 1],
        0,
        [false; 2],
        [false; 2],
        [],
        [],
        [0],
        domain_vertices,
    )
    .unwrap_err();
    // What: the public numeric import rejects zero while constructing the
    // codomain and does not continue into the domain after that error.
    assert_eq!(error, CoreError::InvalidMultiplicityIndex { value: 0 });

    assert_eq!(
        FusionTreePairKey::try_pair_from_sector_ids(
            [1, 1],
            [1, 1],
            0,
            [false; 2],
            [false; 2],
            [],
            [],
            [1],
            [0],
        )
        .unwrap_err(),
        CoreError::InvalidMultiplicityIndex { value: 0 },
    );
}

#[test]
fn fusion_tree_block_structure_proof_reports_the_first_physical_invalid_pair() {
    // What: structure admission reports the earliest malformed
    // fusion-tree pair in physical block order.
    let valid = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &U1FusionRule,
            [u1(1), u1(-1)],
            u1(0),
            [false; 2],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
        FusionTreeKey::new([], u1(0), [], [], []),
    );
    let mismatched_pair = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(&U1FusionRule, [u1(1)], u1(1), [false], [], []).unwrap(),
        FusionTreeKey::try_new_for_rule(&U1FusionRule, [u1(2)], u1(2), [false], [], []).unwrap(),
    );
    let later_bad_shape = FusionTreePairKey::pair(
        FusionTreeKey::new(
            [u1(1), u1(-1)],
            u1(0),
            [false],
            [],
            [MultiplicityIndex::ONE],
        ),
        FusionTreeKey::new([], u1(0), [], [], []),
    );
    let structure = BlockStructure::from_blocks(vec![
        BlockSpec::column_major_with_key(valid.into(), vec![1, 1], 0).unwrap(),
        BlockSpec::column_major_with_key(mismatched_pair.into(), vec![1, 1], 1).unwrap(),
        BlockSpec::column_major_with_key(later_bad_shape.into(), vec![1, 1], 2).unwrap(),
    ])
    .unwrap();

    let error = match LocallyValidatedFusionTreeBlockStructure::try_new(&U1FusionRule, &structure) {
        Ok(_) => panic!("malformed structure unexpectedly admitted"),
        Err(error) => error,
    };
    assert_eq!(
        error,
        CoreError::MalformedFusionTree {
            message: "fusion tree pair requires matching coupled sectors",
        }
    );
}

#[test]
fn fusion_tree_block_structure_proof_rejects_nontrivial_vertex_for_unique_provider() {
    let invalid = FusionTreePairKey::pair(
        FusionTreeKey::new(
            [z2_odd(), z2_odd()],
            z2_even(),
            [false; 2],
            [],
            [MultiplicityIndex::new(2).unwrap()],
        ),
        FusionTreeKey::new([z2_even()], z2_even(), [false], [], []),
    );
    let structure = BlockStructure::from_blocks(vec![BlockSpec::column_major_with_key(
        invalid.into(),
        vec![1, 1, 1],
        0,
    )
    .unwrap()])
    .unwrap();

    // What: a raw label-two key cannot acquire the local proof required
    // by compact multiplicity-free batch execution.
    let error = match LocallyValidatedFusionTreeBlockStructure::try_new(&Z2FusionRule, &structure) {
        Ok(_) => panic!("nontrivial multiplicity label unexpectedly admitted"),
        Err(error) => error,
    };
    assert_eq!(
        error,
        CoreError::MalformedFusionTree {
            message: "fusion tree vertex label exceeds its fusion multiplicity",
        }
    );
}

#[test]
fn coupled_sector_constructor_validates_in_caller_order_before_sorting() {
    let first = FusionTreePairKey::pair(
        FusionTreeKey::new(
            [SectorId::new(1), SectorId::new(2)],
            SectorId::new(3),
            [false; 2],
            [],
            [MultiplicityIndex::new(2).unwrap()],
        ),
        FusionTreeKey::new([SectorId::new(3)], SectorId::new(3), [false], [], []),
    );
    let later_lower_coupled = FusionTreePairKey::pair(
        FusionTreeKey::new(
            [SectorId::new(0), SectorId::new(1)],
            SectorId::new(1),
            [false],
            [],
            [MultiplicityIndex::ONE],
        ),
        FusionTreeKey::new([SectorId::new(1)], SectorId::new(1), [false], [], []),
    );

    // What: the first caller-supplied categorical error wins even though
    // coupled-sector layout order would move the later key before it.
    assert_eq!(
        BlockStructure::coupled_sector_matrix_with_keys(
            &IdentitySymbolPanicRule,
            2,
            3,
            vec![(first, vec![1, 1, 1]), (later_lower_coupled, vec![1, 1, 1]),],
        )
        .unwrap_err(),
        CoreError::MalformedFusionTree {
            message: "fusion tree vertex label exceeds its fusion multiplicity",
        }
    );
}

#[test]
fn fusion_tree_block_structure_proof_rejects_non_categorical_namespaces() {
    // What: LOCAL categorical admission rejects both anonymous dense
    // storage and arbitrary application routing keys.
    for key in [BlockKey::Dense, BlockKey::opaque([7, 11])] {
        let structure = BlockStructure::from_blocks(vec![BlockSpec::column_major_with_key(
            key.clone(),
            vec![1],
            0,
        )
        .unwrap()])
        .unwrap();
        let error =
            match LocallyValidatedFusionTreeBlockStructure::try_new(&Z2FusionRule, &structure) {
                Ok(_) => panic!("non-categorical structure unexpectedly admitted"),
                Err(error) => error,
            };
        assert_eq!(
            error,
            CoreError::ExpectedFusionTreePairKey { actual: key.kind() }
        );
    }
}

#[test]
fn fusion_tree_block_structure_proof_rejects_local_rank_mismatch() {
    // What: categorical admission requires every tree pair to describe the
    // same number of external legs as its containing block structure.
    let key = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(&Z2FusionRule, [z2_even()], z2_even(), [false], [], [])
            .unwrap(),
        FusionTreeKey::new([], z2_even(), [], [], []),
    );
    let structure = BlockStructure::from_blocks(vec![BlockSpec::column_major_with_key(
        key.into(),
        vec![1, 1],
        0,
    )
    .unwrap()])
    .unwrap();

    let error = match LocallyValidatedFusionTreeBlockStructure::try_new(&Z2FusionRule, &structure) {
        Ok(_) => panic!("rank-mismatched structure unexpectedly admitted"),
        Err(error) => error,
    };
    assert_eq!(
        error,
        CoreError::StructureRankMismatch {
            expected: 2,
            actual: 1,
        }
    );
}

#[test]
fn malformed_compact_tree_key_is_panic_free_for_structure_lookup() {
    // What: a raw one-leg key with missing dual metadata remains a fallible
    // categorical input rather than panicking during structure indexing.
    let malformed = FusionTreePairKey::pair(
        FusionTreeKey::new([z2_even()], z2_even(), [], [], []),
        FusionTreeKey::new([], z2_even(), [], [], []),
    );
    let structure = BlockStructure::from_blocks(vec![BlockSpec::column_major_with_key(
        malformed.clone().into(),
        vec![1],
        0,
    )
    .unwrap()])
    .unwrap();

    assert_eq!(
        structure.find_block_index_by_fusion_tree_pair(&malformed),
        Some(0)
    );
    assert_eq!(
        structure.fusion_tree_pair_block(&malformed).unwrap().key(),
        &BlockKey::from(malformed)
    );
}

#[test]
fn fusion_tree_block_structure_proof_indexes_only_its_bound_structure() {
    // What: a successful proof exposes borrowed keys by physical index from
    // its exact categorical structure, including out-of-bounds errors.
    let valid = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(
            &U1FusionRule,
            [u1(1), u1(-1)],
            u1(0),
            [false; 2],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
        FusionTreeKey::new([], u1(0), [], [], []),
    );
    let structure = BlockStructure::from_blocks(vec![BlockSpec::column_major_with_key(
        valid.clone().into(),
        vec![1, 1],
        0,
    )
    .unwrap()])
    .unwrap();

    let proof =
        LocallyValidatedFusionTreeBlockStructure::try_new(&U1FusionRule, &structure).unwrap();
    assert!(std::ptr::eq(proof.rule(), &U1FusionRule));
    assert!(std::ptr::eq(proof.structure(), &structure));
    let canonical = proof.fusion_tree_pair_key(0);
    assert_eq!(canonical.unwrap(), Some(&valid));
    let canonical_missing = proof.fusion_tree_pair_key(1);
    assert_eq!(
        canonical_missing.unwrap_err(),
        CoreError::BlockIndexOutOfBounds { index: 1, count: 1 }
    );
}

#[test]
fn validated_per_index_permute_preserves_planar_braiding_rejection_before_index_lookup() {
    let structure = BlockStructure::empty(0);
    let proof =
        LocallyValidatedFusionTreeBlockStructure::try_new(&PlanarZ2Rule, &structure).unwrap();

    // What: a proof-consuming per-index call observes the symmetric-
    // braiding boundary before attempting to read its requested block.
    assert_eq!(
        proof.permute_codomain_rows_for_block_index(0, &[]),
        Err(CoreError::UnsupportedBraidingStyle {
            expected: "symmetric braiding",
            actual: BraidingStyleKind::NoBraiding,
        })
    );
}

#[test]
fn multiplicity_free_per_block_seam_rejects_misreported_style_under_identity_plan() {
    let rule = MisreportedGenericMultiplicityFreeRule;
    let vacuum = rule.vacuum();
    let key = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(&rule, [], vacuum, [], [], []).unwrap(),
        FusionTreeKey::try_new_for_rule(&rule, [], vacuum, [], [], []).unwrap(),
    );
    let structure = packed_fixture_structure(0, [(key, vec![])]).unwrap();
    let proof = LocallyValidatedFusionTreeBlockStructure::try_new(&rule, &structure).unwrap();
    let identity = PreparedTreePairOperation::prepare_transpose(0, 0, &[], &[]).unwrap();

    // What: identity rows never consult the rule, so the per-block seam's
    // own style check is the only guard; it rejects in release builds too
    // (#1950).
    assert_eq!(
        proof
            .execute_multiplicity_free_for_block_index(0, &identity)
            .unwrap_err(),
        CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Simple,
            actual: FusionStyleKind::Generic,
        }
    );
}

#[test]
fn multiplicity_free_block_boundaries_reject_misreported_style_before_empty_identity() {
    let rule = MisreportedGenericMultiplicityFreeRule;
    let structure = BlockStructure::empty(0);
    let proof = LocallyValidatedFusionTreeBlockStructure::try_new(&rule, &structure).unwrap();
    let identity = PreparedTreePairOperation::prepare_transpose(0, 0, &[], &[]).unwrap();
    let expected = CoreError::UnsupportedFusionStyle {
        expected: FusionStyleKind::Simple,
        actual: FusionStyleKind::Generic,
    };

    // What: neither proof-bound nor raw block entry points let empty or
    // identity work bypass the runtime fusion-style capability.
    assert_eq!(
            proof
                .execute_multiplicity_free_braid_for_block_indices(
                    std::iter::empty(),
                    identity.clone(),
                )
                .unwrap_err(),
            expected
        );
    assert_eq!(
        proof
            .execute_multiplicity_free_braid_ordered_for_block_indices(
                std::iter::empty(),
                identity.clone(),
            )
            .unwrap_err(),
        expected
    );
    assert_eq!(
        proof
            .execute_multiplicity_free_transpose_ordered_for_block_indices(
                std::iter::empty(),
                identity.clone(),
            )
            .unwrap_err(),
        expected
    );
    assert_eq!(
        proof
            .execute_multiplicity_free_transpose_for_block_indices(std::iter::empty(), identity,)
            .unwrap_err(),
        expected
    );
    assert_eq!(
        multiplicity_free_braid_tree_block(&rule, &[], &[], &[]).unwrap_err(),
        expected
    );
    assert_eq!(
        multiplicity_free_permute_tree_block(&rule, &[], &[]).unwrap_err(),
        expected
    );
    assert_eq!(
        multiplicity_free_braid_tree_pair_block(&rule, &[], &[], &[], &[], &[]).unwrap_err(),
        expected
    );
    assert_eq!(
        multiplicity_free_permute_tree_pair_block(&rule, &[], &[], &[]).unwrap_err(),
        expected
    );
    assert_eq!(
        multiplicity_free_transpose_tree_pair_block(&rule, &[], &[], &[]).unwrap_err(),
        expected
    );
}

#[test]
fn borrowed_block_execution_rejects_prepared_family_and_source_split_mismatch() {
    let vacuum = su2(0);
    let half = su2(1);
    let pair = FusionTreePairKey::pair(
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
    let structure = packed_fixture_structure(2, [(pair, vec![1, 1])]).unwrap();
    let proof =
        LocallyValidatedFusionTreeBlockStructure::try_new(&SU2FusionRule, &structure).unwrap();
    let transpose = PreparedTreePairOperation::prepare_transpose(2, 0, &[1], &[0]).unwrap();
    let transpose_identity =
        PreparedTreePairOperation::prepare_transpose(2, 0, &[0, 1], &[]).unwrap();
    let braid =
        PreparedTreePairOperation::prepare_permute(&SU2FusionRule, 2, 0, &[1, 0], &[]).unwrap();
    let wrong_split =
        PreparedTreePairOperation::prepare_permute(&SU2FusionRule, 1, 1, &[1], &[0]).unwrap();

    // What: safe borrowed block executors reject a prepared operation from
    // the other family before its private plan variant reaches execution.
    assert_eq!(
        proof
            .execute_multiplicity_free_braid_ordered_for_block_indices_borrowed([0], &transpose,)
            .unwrap_err(),
        CoreError::MalformedFusionTree {
            message: "prepared tree-pair operation is incompatible with braid block execution",
        }
    );
    assert_eq!(
        proof
            .execute_multiplicity_free_braid_ordered_for_block_indices_borrowed(
                [0],
                &transpose_identity,
            )
            .unwrap_err(),
        CoreError::MalformedFusionTree {
            message: "prepared tree-pair operation is incompatible with braid block execution",
        }
    );
    assert_eq!(
        proof
            .execute_multiplicity_free_transpose_ordered_for_block_indices_borrowed([0], &braid,)
            .unwrap_err(),
        CoreError::MalformedFusionTree {
            message: "prepared tree-pair operation is incompatible with transpose block execution",
        }
    );

    // What: operation family is a source-independent preflight. Empty and
    // invalid-index calls cannot bypass it, while a valid empty operation
    // remains the empty linear map.
    assert_eq!(
        proof
            .execute_multiplicity_free_braid_ordered_for_block_indices_borrowed(
                std::iter::empty(),
                &transpose,
            )
            .unwrap_err(),
        CoreError::MalformedFusionTree {
            message: "prepared tree-pair operation is incompatible with braid block execution",
        }
    );
    assert_eq!(
        proof
            .execute_multiplicity_free_transpose_ordered_for_block_indices_borrowed(
                std::iter::empty(),
                &braid,
            )
            .unwrap_err(),
        CoreError::MalformedFusionTree {
            message: "prepared tree-pair operation is incompatible with transpose block execution",
        }
    );
    assert_eq!(
        proof
            .execute_multiplicity_free_braid_ordered_for_block_indices(
                [usize::MAX],
                transpose.clone(),
            )
            .unwrap_err(),
        CoreError::MalformedFusionTree {
            message: "prepared tree-pair operation is incompatible with braid block execution",
        }
    );
    assert_eq!(
        proof
            .execute_multiplicity_free_transpose_ordered_for_block_indices(
                [usize::MAX],
                braid.clone(),
            )
            .unwrap_err(),
        CoreError::MalformedFusionTree {
            message: "prepared tree-pair operation is incompatible with transpose block execution",
        }
    );
    for empty in [
        proof
            .execute_multiplicity_free_braid_ordered_for_block_indices_borrowed(
                std::iter::empty(),
                &braid,
            )
            .unwrap(),
        proof
            .execute_multiplicity_free_transpose_ordered_for_block_indices_borrowed(
                std::iter::empty(),
                &transpose,
            )
            .unwrap(),
    ] {
        assert!(empty.destinations().is_empty());
        assert_eq!(empty.source_count(), 0);
    }

    // What: preparation for the right operation family is still bound to
    // its exact source codomain/domain split.
    assert_eq!(
        proof
            .execute_multiplicity_free_braid_ordered_for_block_indices_borrowed([0], &wrong_split,)
            .unwrap_err(),
        CoreError::DimensionMismatch {
            expected: 1,
            actual: 2,
        }
    );
}

#[test]
fn block_preflight_rejects_prepared_capability_before_empty_or_invalid_source() {
    let structure = BlockStructure::empty(0);
    let proof =
        LocallyValidatedFusionTreeBlockStructure::try_new(&PlanarZ2Rule, &structure).unwrap();
    let symmetric_identity =
        PreparedTreePairOperation::prepare_permute(&Z2FusionRule, 0, 0, &[], &[]).unwrap();
    let expected = CoreError::UnsupportedBraidingStyle {
        expected: "symmetric braiding",
        actual: BraidingStyleKind::NoBraiding,
    };

    // What: rule capability belongs to operation preflight, so neither an
    // empty iterator nor an invalid source index can bypass it.
    assert_eq!(
        proof
            .execute_multiplicity_free_braid_ordered_for_block_indices_borrowed(
                std::iter::empty(),
                &symmetric_identity,
            )
            .unwrap_err(),
        expected
    );
    assert_eq!(
        proof
            .execute_multiplicity_free_braid_ordered_for_block_indices(
                [usize::MAX],
                symmetric_identity,
            )
            .unwrap_err(),
        expected
    );
}

#[test]
fn inadmissible_tree_is_rejected_before_every_identity_path() {
    // What: a shape-correct but inadmissible tree cannot pass through
    // scalar, prepared, or whole-block identity shortcuts.
    fn assert_inadmissible<T: std::fmt::Debug>(result: Result<T, CoreError>) {
        assert_eq!(
            result.unwrap_err(),
            CoreError::MalformedFusionTree {
                message: "fusion tree contains an inadmissible fusion vertex",
            }
        );
    }

    let rule = SplitOnlyCountingRule::default();
    let odd = SectorId::new(1);
    let vacuum = SectorId::new(0);
    let source = FusionTreeKey::new(
        [odd; 3],
        vacuum,
        [false; 3],
        [vacuum],
        [MultiplicityIndex::ONE; 2],
    );
    let identity = [0, 1, 2];
    let levels = [0, 1, 2];
    assert_inadmissible(multiplicity_free_braid_tree(
        &rule, &source, &identity, &levels,
    ));
    assert_inadmissible(multiplicity_free_permute_tree(&rule, &source, &identity));
    assert_inadmissible(multiplicity_free_braid_tree_block(
        &rule,
        std::slice::from_ref(&source),
        &identity,
        &levels,
    ));
    assert_inadmissible(multiplicity_free_permute_tree_block(
        &rule,
        std::slice::from_ref(&source),
        &identity,
    ));

    let pair = FusionTreePairKey::pair(source, FusionTreeKey::new([], vacuum, [], [], []));
    let prepared =
        PreparedTreePairOperation::prepare_braid(&rule, 3, 0, &identity, &[], &levels, &[])
            .unwrap();
    assert_inadmissible(prepared.execute_multiplicity_free(&rule, &pair));
    assert_inadmissible(multiplicity_free_braid_tree_pair(
        &rule,
        &pair,
        &identity,
        &[],
        &levels,
        &[],
    ));
    assert_inadmissible(multiplicity_free_permute_tree_pair(
        &rule,
        &pair,
        &identity,
        &[],
    ));
    assert_inadmissible(multiplicity_free_transpose_tree_pair(
        &rule,
        &pair,
        &identity,
        &[],
    ));
    assert_inadmissible(multiplicity_free_braid_tree_pair_block(
        &rule,
        std::slice::from_ref(&pair),
        &identity,
        &[],
        &levels,
        &[],
    ));
    assert_inadmissible(multiplicity_free_permute_tree_pair_block(
        &rule,
        std::slice::from_ref(&pair),
        &identity,
        &[],
    ));
    assert_inadmissible(multiplicity_free_transpose_tree_pair_block(
        &rule,
        std::slice::from_ref(&pair),
        &identity,
        &[],
    ));
    assert_eq!(rule.f_calls.load(std::sync::atomic::Ordering::Relaxed), 0);
    assert_eq!(rule.r_calls.load(std::sync::atomic::Ordering::Relaxed), 0);
}

#[test]
fn malformed_sources_do_not_evaluate_symbols() {
    // What: categorical rejection occurs before an F or R provider can
    // observe a malformed source.
    //
    // Bends are not counted here: the coefficient comes from
    // b_symbol_scalar / sqrt_dim_scalar on MultiplicityFreeRigidSymbols,
    // which have no separate counter, so a bend-specific assertion would
    // only be able to observe zero and would prove nothing.
    let rule = SplitOnlyCountingRule::default();
    let invalid = FusionTreeKey::new(
        [SectorId::new(1); 2],
        SectorId::new(1),
        [false; 2],
        [],
        [MultiplicityIndex::ONE],
    );
    assert!(multiplicity_free_braid_tree(&rule, &invalid, &[0, 1], &[0, 1]).is_err());
    let pair = FusionTreePairKey::pair(
        invalid,
        FusionTreeKey::new([SectorId::new(1)], SectorId::new(1), [false], [], []),
    );
    assert!(multiplicity_free_repartition_tree_pair(&rule, &pair, 1).is_err());
    assert_eq!(rule.f_calls.load(std::sync::atomic::Ordering::Relaxed), 0);
    assert_eq!(rule.r_calls.load(std::sync::atomic::Ordering::Relaxed), 0);
}

#[test]
fn block_validation_is_source_major_and_runs_once_per_source() {
    // What: block proofs report the first source in slice order and do one
    // N-symbol validation pass before identity execution.
    let tau = SectorId::new(1);
    let vacuum = SectorId::new(0);
    let invalid_first = FusionTreeKey::new(
        [tau; 3],
        vacuum,
        [false; 3],
        [vacuum],
        [MultiplicityIndex::ONE; 2],
    );
    let valid_first = FusionTreeKey::new(
        [tau; 3],
        tau,
        [false; 3],
        [vacuum],
        [MultiplicityIndex::ONE; 2],
    );
    let different_group =
        FusionTreeKey::new([tau; 2], vacuum, [false; 2], [], [MultiplicityIndex::ONE]);
    assert_eq!(
        multiplicity_free_braid_tree_block(
            &TableRule::fibonacci(),
            &[invalid_first, different_group.clone()],
            &[0, 1, 2],
            &[0, 1, 2],
        )
        .unwrap_err(),
        CoreError::MalformedFusionTree {
            message: "fusion tree contains an inadmissible fusion vertex",
        }
    );
    assert_eq!(
        multiplicity_free_braid_tree_block(
            &TableRule::fibonacci(),
            &[valid_first, different_group],
            &[0, 1, 2],
            &[0, 1, 2],
        )
        .unwrap_err(),
        CoreError::MalformedFusionTree {
            message: "fusion-tree keys must share one group",
        }
    );

    let rule = SplitOnlyCountingRule::default();
    let valid = FusionTreeKey::new(
        [SectorId::new(1); 2],
        SectorId::new(0),
        [false; 2],
        [],
        [MultiplicityIndex::ONE],
    );
    let rows = multiplicity_free_braid_tree_block(
        &rule,
        &[valid.clone(), valid.clone()],
        &[0, 1],
        &[0, 1],
    )
    .unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rule.n_calls.load(std::sync::atomic::Ordering::Relaxed), 2);

    rule.n_calls.store(0, std::sync::atomic::Ordering::Relaxed);
    let pair = FusionTreePairKey::pair(
        valid,
        FusionTreeKey::new([SectorId::new(0)], SectorId::new(0), [false], [], []),
    );
    let rows =
        multiplicity_free_permute_tree_pair_block(&rule, &[pair.clone(), pair], &[0, 1], &[2])
            .unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rule.n_calls.load(std::sync::atomic::Ordering::Relaxed), 2);
}

#[test]
fn keyed_coupled_layout_reports_extent_overflow_when_the_tree_is_placed() {
    // One coupled sector with two row trees whose extents together exceed
    // `usize` and two column trees.
    let tree = |a: usize, b: usize| {
        FusionTreeKey::try_from_sector_ids([a, b], 1, [false, false], [], [1]).unwrap()
    };
    let huge = usize::MAX / 2 + 1;
    let (row_a, row_b, col_c, col_d) = (tree(1, 100), tree(2, 101), tree(10, 200), tree(11, 201));
    let grid = [
        (row_a.clone(), col_c.clone()),
        (row_b.clone(), col_c.clone()),
        (row_a, col_d.clone()),
        (row_b, col_d),
    ];
    let keys = grid
        .iter()
        .map(|(row, col)| FusionTreePairKey::pair(row.clone(), col.clone()))
        .collect::<Vec<_>>();
    let shape = |dim: usize| vec![dim, 1, 1, 1];

    // What: the placement rule reports the overflow as soon as the second
    // row tree is placed, ahead of a later grid hole or extent change.
    let holed = coupled_sector_matrix_block_specs(
        2,
        4,
        &keys[..3],
        &[shape(huge), shape(huge), shape(huge)],
    );
    assert_eq!(holed.unwrap_err(), CoreError::ElementCountOverflow);
    let changed = coupled_sector_matrix_block_specs(
        2,
        4,
        &keys,
        &[shape(huge), shape(huge), shape(huge), shape(1)],
    );
    assert_eq!(changed.unwrap_err(), CoreError::ElementCountOverflow);
}
