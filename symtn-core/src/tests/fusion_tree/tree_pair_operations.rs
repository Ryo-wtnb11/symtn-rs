use super::*;

#[test]
fn linearize_tree_pair_permutation_matches_tensorkit_zero_based_formula() {
    assert_eq!(
        linearize_tree_pair_permutation(&[0, 1], &[2, 3], 2, 2).unwrap(),
        vec![0, 1, 2, 3]
    );
    assert_eq!(
        linearize_tree_pair_permutation(&[3, 0], &[1, 2], 2, 2).unwrap(),
        vec![2, 0, 3, 1]
    );

    let err = linearize_tree_pair_permutation(&[0, 0], &[1, 2], 2, 2).unwrap_err();
    assert_eq!(
        err,
        CoreError::InvalidPermutation {
            permutation: vec![0, 0, 1, 2],
            rank: 4,
        }
    );
}

#[test]
fn identity_braid_tree_pair_skips_symbols_and_repartition() {
    // What: an exact same-split braid is source => one for both Unique and
    // multiplicity-free entry points, without consulting F/R/bend data.
    let source =
        FusionTreePairKey::try_pair_from_sector_ids([1], [1], 1, [false], [false], [], [], [], [])
            .unwrap();

    // Not covered here: the deleted execute_unique_pivotal route's own
    // identity-braid shortcut (`unique_braid_tree_pair`). Its no-op
    // symbol-free contract is already covered below by
    // multiplicity_free_braid_tree_pair and its block variant, which are
    // the live entry points for this behavior.
    let multiplicity_free = multiplicity_free_braid_tree_pair(
        &IdentitySymbolPanicRule,
        &source,
        &[0],
        &[1],
        &[19],
        &[3],
    )
    .unwrap();
    assert_eq!(multiplicity_free, vec![(source.clone(), 1.0)]);

    let block = multiplicity_free_braid_tree_pair_block(
        &IdentitySymbolPanicRule,
        std::slice::from_ref(&source),
        &[0],
        &[1],
        &[19],
        &[3],
    )
    .unwrap();
    assert_eq!(block, vec![vec![(source, 1.0)]]);
}

#[test]
fn indexed_ordered_identity_preserves_duplicate_sources_and_skips_symbols() {
    let source =
        FusionTreePairKey::try_pair_from_sector_ids([1], [1], 1, [false], [false], [], [], [], [])
            .unwrap();
    let structure = BlockStructure::from_blocks(vec![BlockSpec::column_major_with_key(
        source.clone().into(),
        vec![2, 3],
        0,
    )
    .unwrap()])
    .unwrap();
    let braid = PreparedTreePairOperation::prepare_braid(
        &IdentitySymbolPanicRule,
        1,
        1,
        &[0],
        &[1],
        &[19],
        &[3],
    )
    .unwrap();
    let map = multiplicity_free_braid_tree_pair_block_ordered_indexed(
        &IdentitySymbolPanicRule,
        &structure,
        &[0, 0],
        FusionTreePairOrientation::Adjoint,
        &braid,
    )
    .unwrap();
    assert_eq!(map.destinations(), &[source]);
    assert_eq!(map.source_count(), 2);
    assert_eq!(
        map.storage(),
        &OrderedBlockLinearStorage::SingletonColumns {
            destination_rows: vec![0, 0],
            coefficients: vec![1.0, 1.0]
        }
    );
    let scalar =
        FusionTreePairKey::try_pair_from_sector_ids([], [], 0, [], [], [], [], [], []).unwrap();
    let scalar_structure = BlockStructure::from_blocks(vec![BlockSpec::column_major_with_key(
        scalar.clone().into(),
        vec![],
        0,
    )
    .unwrap()])
    .unwrap();
    let transpose = PreparedTreePairOperation::prepare_transpose(0, 0, &[], &[]).unwrap();
    let map = multiplicity_free_transpose_tree_pair_block_ordered_indexed(
        &IdentitySymbolPanicRule,
        &scalar_structure,
        &[0, 0],
        FusionTreePairOrientation::Direct,
        &transpose,
    )
    .unwrap();
    assert_eq!(map.destinations(), &[scalar]);
    assert_eq!(
        map.storage(),
        &OrderedBlockLinearStorage::SingletonColumns {
            destination_rows: vec![0, 0],
            coefficients: vec![1.0, 1.0]
        }
    );
}

#[test]
fn indexed_ordered_identity_preserves_distinct_and_reordered_columns() {
    let keys: Vec<_> = [0, 2]
        .into_iter()
        .map(|coupled| {
            FusionTreePairKey::try_pair_from_sector_ids(
                [1, 1],
                [1, 1],
                coupled,
                [false, false],
                [false, false],
                [],
                [],
                [1],
                [1],
            )
            .unwrap()
        })
        .collect();
    let structure = BlockStructure::from_blocks(
        keys.iter()
            .enumerate()
            .map(|(index, key)| {
                BlockSpec::column_major_with_key(key.clone().into(), vec![1; 4], index).unwrap()
            })
            .collect(),
    )
    .unwrap();
    let prepared =
        PreparedTreePairOperation::prepare_permute(&SU2FusionRule, 2, 2, &[0, 1], &[2, 3]).unwrap();
    for (indices, destinations, rows) in [
        (
            vec![0, 1],
            vec![keys[0].clone(), keys[1].clone()],
            vec![0, 1],
        ),
        (
            vec![1, 0],
            vec![keys[1].clone(), keys[0].clone()],
            vec![0, 1],
        ),
        (
            vec![1, 0, 1],
            vec![keys[1].clone(), keys[0].clone()],
            vec![0, 1, 0],
        ),
    ] {
        let map = multiplicity_free_braid_tree_pair_block_ordered_indexed(
            &SU2FusionRule,
            &structure,
            &indices,
            FusionTreePairOrientation::Adjoint,
            &prepared,
        )
        .unwrap();
        assert_eq!(map.destinations(), destinations);
        assert_eq!(map.source_count(), indices.len());
        assert_eq!(
            map.storage(),
            &OrderedBlockLinearStorage::SingletonColumns {
                destination_rows: rows,
                coefficients: vec![1.0; indices.len()],
            }
        );
    }
}

#[test]
fn braid_tree_block_matches_per_source_su2_rows() {
    use std::collections::BTreeMap;

    let rule = SU2FusionRule;
    let sources = [
        FusionTreeKey::try_from_sector_ids([1, 1, 1], 1, [false, false, false], [0], [1, 1])
            .unwrap(),
        FusionTreeKey::try_from_sector_ids([1, 1, 1], 1, [false, false, false], [2], [1, 1])
            .unwrap(),
    ];

    let block =
        multiplicity_free_braid_tree_block(&rule, &sources, &[0, 2, 1], &[0, 1, 2]).unwrap();
    assert_eq!(block.len(), sources.len());
    for (source, block_rows) in sources.iter().zip(&block) {
        let per_source =
            multiplicity_free_braid_tree(&rule, source, &[0, 2, 1], &[0, 1, 2]).unwrap();
        let collect = |rows: &[(FusionTreeKey, f64)]| {
            let mut coefficients = BTreeMap::<FusionTreeKey, f64>::new();
            for (key, coefficient) in rows {
                *coefficients.entry(key.clone()).or_default() += coefficient;
            }
            coefficients
        };
        let expected = collect(&per_source);
        let actual = collect(block_rows);

        // What: the whole-block walk preserves every destination tree and
        // coefficient produced by independent SU(2) source transforms.
        assert_eq!(
            expected.keys().collect::<Vec<_>>(),
            actual.keys().collect::<Vec<_>>()
        );
        for (key, expected_coefficient) in expected {
            let actual_coefficient = actual[&key];
            assert!(
                (expected_coefficient - actual_coefficient).abs()
                    <= 1.0e-12 * (1.0 + expected_coefficient.abs())
            );
        }
    }
}

#[test]
fn tree_block_admits_every_source_before_group_check_and_symbols() {
    let base = FusionTreeKey::try_from_sector_ids([1, 2], 3, [false, false], [], [1]).unwrap();
    let valid_mixed = [
        FusionTreeKey::try_from_sector_ids([1, 4], 5, [false, false], [], [1]).unwrap(),
        FusionTreeKey::try_from_sector_ids([1, 2], 3, [false, true], [], [1]).unwrap(),
    ];
    let expected = CoreError::MalformedFusionTree {
        message: "fusion-tree keys must share one group",
    };

    for other in valid_mixed {
        let sources = [base.clone(), other];
        // What: admitted mixed groups fail before the panic-on-symbol
        // fixture can evaluate F or R data.
        assert_eq!(
            multiplicity_free_braid_tree_block(
                &IdentitySymbolPanicRule,
                &sources,
                &[0, 1],
                &[0, 1],
            )
            .unwrap_err(),
            expected
        );
        assert_eq!(
            multiplicity_free_permute_tree_block(&IdentitySymbolPanicRule, &sources, &[1, 0],)
                .unwrap_err(),
            expected
        );
    }

    let invalid_vertex = FusionTreeKey::new(
        base.uncoupled().iter().copied(),
        base.coupled(),
        base.is_dual().iter().copied(),
        base.innerlines().iter().copied(),
        [MultiplicityIndex::new(2).unwrap()],
    );
    let invalid_sources = [base, invalid_vertex];
    let expected_vertex = CoreError::MalformedFusionTree {
        message: "fusion tree vertex label exceeds its fusion multiplicity",
    };
    // What: provider-owned multiplicity admission checks every source
    // before a block shortcut can discard explicit vertex identity.
    assert_eq!(
        multiplicity_free_braid_tree_block(
            &IdentitySymbolPanicRule,
            &invalid_sources,
            &[0, 1],
            &[0, 1],
        )
        .unwrap_err(),
        expected_vertex
    );
}

#[test]
fn tree_block_empty_and_identity_contracts_preserve_source_order() {
    let empty: &[FusionTreeKey] = &[];
    assert_eq!(
        multiplicity_free_braid_tree_block(&IdentitySymbolPanicRule, empty, &[], &[],).unwrap(),
        Vec::<Vec<(FusionTreeKey, f64)>>::new()
    );
    assert_eq!(
        multiplicity_free_permute_tree_block(&IdentitySymbolPanicRule, empty, &[]).unwrap(),
        Vec::<Vec<(FusionTreeKey, f64)>>::new()
    );

    let half = su2(1);
    let sources = [
        FusionTreeKey::try_from_sector_ids(
            [half.id(), half.id()],
            su2(0).id(),
            [false; 2],
            [],
            [1],
        )
        .unwrap(),
        FusionTreeKey::try_from_sector_ids(
            [half.id(), half.id()],
            su2(2).id(),
            [false; 2],
            [],
            [1],
        )
        .unwrap(),
    ];
    let expected = sources
        .iter()
        .cloned()
        .map(|source| vec![(source, 1.0)])
        .collect::<Vec<_>>();
    // What: distinct coupled labels remain one external-sector group and
    // the symbol-free identity path returns exact rows in source order.
    assert_eq!(
        multiplicity_free_braid_tree_block(&SU2FusionRule, &sources, &[0, 1], &[13, 5],).unwrap(),
        expected
    );
    assert_eq!(
        multiplicity_free_permute_tree_block(&SU2FusionRule, &sources, &[0, 1],).unwrap(),
        expected
    );
}

#[test]
fn tree_pair_block_apis_reject_invalid_later_vertex_before_symbols() {
    let codomain = FusionTreeKey::try_from_sector_ids([1, 2], 3, [false, false], [], [1]).unwrap();
    let domain = FusionTreeKey::try_from_sector_ids([3], 3, [false], [], []).unwrap();
    let base = FusionTreePairKey::pair(codomain.clone(), domain.clone());
    let invalid_codomain = FusionTreeKey::new(
        codomain.uncoupled().iter().copied(),
        codomain.coupled(),
        codomain.is_dual().iter().copied(),
        codomain.innerlines().iter().copied(),
        [MultiplicityIndex::new(2).unwrap()],
    );
    let invalid = FusionTreePairKey::pair(invalid_codomain, domain);

    // What: every source pair is categorically admitted before block
    // identity shortcuts or symbol evaluation.
    assert_mixed_tree_pair_block_group_is_rejected(
        &IdentitySymbolPanicRule,
        &[base, invalid],
        CoreError::MalformedFusionTree {
            message: "fusion tree vertex label exceeds its fusion multiplicity",
        },
    );
}

#[test]
fn split_only_su2_braid_matches_composition_oracle_in_both_directions() {
    // What: SU(2) 2|2 -> 3|1 and 2|2 -> 1|3 retain the exact tree keys,
    // dual flags, and bend coefficients of the all-codomain reference route.
    let source = FusionTreePairKey::try_pair_from_sector_ids(
        [1, 2],
        [2, 1],
        1,
        [false, true],
        [true, false],
        [],
        [],
        [1],
        [1],
    )
    .unwrap();

    for (codomain_axes, domain_axes, target_rank) in
        [(&[0, 1, 3][..], &[2][..], 3), (&[0][..], &[2, 3, 1][..], 1)]
    {
        let actual = multiplicity_free_braid_tree_pair(
            &SU2FusionRule,
            &source,
            codomain_axes,
            domain_axes,
            &[0, 1],
            &[2, 3],
        )
        .unwrap();
        let expected =
            split_only_tree_pair_route_composition_oracle(&SU2FusionRule, &source, target_rank)
                .unwrap();
        assert_eq!(actual.len(), expected.len());
        for ((actual_key, actual_coefficient), (expected_key, expected_coefficient)) in
            actual.iter().zip(&expected)
        {
            assert_eq!(actual_key, expected_key);
            assert!((actual_coefficient - expected_coefficient).abs() < 1.0e-12);
        }
    }
}

#[test]
fn split_only_tree_pair_braid_handles_empty_split_boundaries() {
    // What: the two extreme repartitions 0|2 -> 2|0 and 2|0 -> 0|2
    // preserve the same keys, dual flags, and coefficients as the direct
    // primitive, including an empty codomain or domain tree.
    let all_domain = FusionTreePairKey::try_pair_from_sector_ids(
        [],
        [1, 1],
        0,
        [],
        [false, true],
        [],
        [],
        [],
        [1],
    )
    .unwrap();
    let to_codomain =
        multiplicity_free_braid_tree_pair(&SU2FusionRule, &all_domain, &[1, 0], &[], &[], &[0, 1])
            .unwrap();
    let expected_codomain =
        multiplicity_free_repartition_tree_pair(&SU2FusionRule, &all_domain, 2).unwrap();
    assert_eq!(to_codomain, expected_codomain);

    let all_codomain = &to_codomain[0].0;
    let to_domain =
        multiplicity_free_braid_tree_pair(&SU2FusionRule, all_codomain, &[], &[1, 0], &[0, 1], &[])
            .unwrap();
    let expected_domain =
        multiplicity_free_repartition_tree_pair(&SU2FusionRule, all_codomain, 0).unwrap();
    assert_eq!(to_domain, expected_domain);
}

#[test]
fn nonidentity_tree_pair_braid_does_not_enter_split_only_path() {
    // What: changing the split does not suppress a real external-leg
    // permutation, and malformed axis maps still fail validation.
    let source = FusionTreePairKey::try_pair_from_sector_ids(
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
    .unwrap();

    let rule = SplitOnlyCountingRule::default();
    multiplicity_free_braid_tree_pair(&rule, &source, &[2, 0], &[1], &[0], &[1, 2]).unwrap();
    assert!(rule.r_calls.load(std::sync::atomic::Ordering::Relaxed) > 0);

    for (codomain_axes, domain_axes) in [
        (&[0, 0][..], &[1][..]),
        (&[0, 3][..], &[1][..]),
        (&[0][..], &[1][..]),
    ] {
        assert!(matches!(
            multiplicity_free_braid_tree_pair(
                &rule,
                &source,
                codomain_axes,
                domain_axes,
                &[0],
                &[1, 2],
            ),
            Err(CoreError::InvalidPermutation { rank: 3, .. })
        ));
    }
}

#[test]
fn identity_braid_tree_pair_validates_levels_before_symbol_free_return() {
    // What: malformed levels remain errors even when the axis map is the
    // exact current split.
    let source =
        FusionTreePairKey::try_pair_from_sector_ids([1], [1], 1, [false], [false], [], [], [], [])
            .unwrap();

    assert!(matches!(
        multiplicity_free_braid_tree_pair(&IdentitySymbolPanicRule, &source, &[0], &[1], &[], &[3],),
        Err(CoreError::DimensionMismatch { actual: 0, .. })
    ));
    assert!(matches!(
        multiplicity_free_braid_tree_pair(
            &IdentitySymbolPanicRule,
            &source,
            &[0],
            &[1],
            &[19],
            &[],
        ),
        Err(CoreError::DimensionMismatch { actual: 0, .. })
    ));
    assert!(matches!(
        multiplicity_free_braid_tree_pair(
            &IdentitySymbolPanicRule,
            &source,
            &[0],
            &[0],
            &[19],
            &[3],
        ),
        Err(CoreError::InvalidPermutation { rank: 2, .. })
    ));
}

#[test]
fn unique_identity_tree_operations_reject_simple_fusion_rules() {
    // What: identity axes do not let a Simple rule enter an API whose
    // contract requires Unique fusion.
    //
    // Not covered here: the deleted execute_unique_pivotal route's own
    // identity-axis rejection of Simple providers. That contract no
    // longer exists — the surviving multiplicity_free_braid_tree accepts
    // Simple providers by design, so there is nothing left to assert for
    // the braid side of this test.
    let tree = FusionTreeKey::try_from_sector_ids([1], 1, [false], [], []).unwrap();
    let expected = CoreError::UnsupportedFusionStyle {
        expected: FusionStyleKind::Unique,
        actual: FusionStyleKind::Simple,
    };

    assert_eq!(
        unique_permute_tree(&SU2FusionRule, &tree, &[0]).unwrap_err(),
        expected
    );
}

#[test]
fn same_split_transpose_of_dual_tree_pair_skips_bend_symbols() {
    // What: a real codomain/domain tree pair at its current 2|1 split is
    // source => one without consulting bend/fold data.
    let source = FusionTreePairKey::try_pair_from_sector_ids(
        [1, 0],
        [1],
        1,
        [false, true],
        [true],
        [],
        [],
        [1],
        [],
    )
    .unwrap();

    assert_eq!(
        multiplicity_free_transpose_tree_pair(&IdentitySymbolPanicRule, &source, &[0, 1], &[2])
            .unwrap(),
        vec![(source, 1.0)]
    );
}

#[test]
fn unique_repartition_tree_pair_moves_domain_to_reversed_dual_codomain() {
    let source = FusionTreePairKey::try_pair_from_sector_ids(
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
    .unwrap();

    let terms = multiplicity_free_repartition_tree_pair(&Z2FusionRule, &source, 3).unwrap();
    assert_eq!(terms.len(), 1);
    let (all_out, coefficient) = terms.into_iter().next().unwrap();

    assert_eq!(coefficient, 1.0);
    assert_eq!(
        all_out.codomain_uncoupled(),
        &[SectorId::new(1), SectorId::new(1), SectorId::new(0)]
    );
    assert_eq!(all_out.codomain_is_dual(), &[false, false, true]);
    assert_eq!(all_out.codomain_innerlines(), &[SectorId::new(0)]);
    assert_eq!(
        all_out.codomain_vertices(),
        &[MultiplicityIndex::ONE, MultiplicityIndex::ONE]
    );
    assert!(all_out.domain_uncoupled().is_empty());
    assert_eq!(all_out.domain_tree().coupled(), SectorId::new(0));
}

#[test]
fn unique_braid_tree_pair_matches_single_tree_when_domain_is_empty() {
    let source = FusionTreePairKey::pair(
        FusionTreeKey::try_from_sector_ids([1, 1], 0, [false, true], [], [1]).unwrap(),
        FusionTreeKey::new(
            Vec::<SectorId>::new(),
            FermionParityFusionRule.vacuum(),
            Vec::<bool>::new(),
            Vec::<SectorId>::new(),
            Vec::<MultiplicityIndex>::new(),
        ),
    );

    let terms = multiplicity_free_braid_tree_pair(
        &FermionParityFusionRule,
        &source,
        &[1, 0],
        &[],
        &[0, 1],
        &[],
    )
    .unwrap();
    assert_eq!(terms.len(), 1);
    let (braided, coefficient) = terms.into_iter().next().unwrap();

    assert_eq!(coefficient, -1.0);
    assert_eq!(
        braided.codomain_uncoupled(),
        &[SectorId::new(1), SectorId::new(1)]
    );
    assert_eq!(braided.codomain_is_dual(), &[true, false]);
    assert!(braided.domain_uncoupled().is_empty());
    assert_eq!(
        braided.domain_tree().coupled(),
        FermionParityFusionRule.vacuum()
    );
}

#[test]
fn unique_permute_tree_pair_handles_domain_only_swap() {
    let source = FusionTreePairKey::try_pair_from_sector_ids(
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
    .unwrap();

    let terms = multiplicity_free_permute_tree_pair(&Z2FusionRule, &source, &[0], &[2, 1]).unwrap();
    assert_eq!(terms.len(), 1);
    let (permuted, coefficient) = terms.into_iter().next().unwrap();

    assert_eq!(coefficient, 1.0);
    assert_eq!(permuted.codomain_uncoupled(), &[SectorId::new(1)]);
    assert_eq!(
        permuted.domain_uncoupled(),
        &[SectorId::new(1), SectorId::new(0)]
    );
    assert_eq!(permuted.domain_is_dual(), &[true, false]);
    assert_eq!(permuted.domain_vertices(), &[MultiplicityIndex::ONE]);
}

#[test]
fn unique_permute_tree_pair_includes_codomain_domain_crossing() {
    let source =
        FusionTreePairKey::try_pair_from_sector_ids([1], [1], 1, [false], [true], [], [], [], [])
            .unwrap();

    let terms =
        multiplicity_free_permute_tree_pair(&FermionParityFusionRule, &source, &[1], &[0]).unwrap();
    assert_eq!(terms.len(), 1);
    let (permuted, coefficient) = terms.into_iter().next().unwrap();

    assert_eq!(coefficient, -1.0);
    assert_eq!(permuted.codomain_uncoupled(), &[SectorId::new(1)]);
    assert_eq!(permuted.codomain_is_dual(), &[false]);
    assert_eq!(permuted.domain_uncoupled(), &[SectorId::new(1)]);
    assert_eq!(permuted.domain_is_dual(), &[true]);
}

#[test]
fn prepared_fermionic_domain_crossing_is_exactly_negative_one() {
    // What: an actual odd fZ2 domain leg crossing an odd codomain leg
    // retains the exact TensorKit fermionic phase through prepared replay.
    let source =
        FusionTreePairKey::try_pair_from_sector_ids([1], [1], 1, [false], [true], [], [], [], [])
            .unwrap();
    let prepared =
        PreparedTreePairOperation::prepare_permute(&FermionParityFusionRule, 1, 1, &[1], &[0])
            .unwrap();

    let (destination, coefficient) = prepared
        .execute_unique_rigid(&FermionParityFusionRule, &source)
        .unwrap();

    assert_eq!(coefficient, -1.0);
    assert_eq!(destination.codomain_uncoupled(), &[SectorId::new(1)]);
    assert_eq!(destination.domain_uncoupled(), &[SectorId::new(1)]);
    assert_eq!(destination.codomain_is_dual(), &[false]);
    assert_eq!(destination.domain_is_dual(), &[true]);
}

#[test]
fn prepared_permute_reverses_domain_levels_before_artin_lowering() {
    // What: TensorKit linearizes incoming legs in reverse order, so a
    // domain-only swap is prepared as the inverse Artin generator.
    let prepared =
        PreparedTreePairOperation::prepare_permute(&Z2FusionRule, 1, 2, &[0], &[2, 1]).unwrap();
    assert_eq!(
        prepared
            .plan
            .artin_steps()
            .expect("domain-only swap must prepare a braid")
            .collect::<Vec<_>>(),
        &[PreparedArtinStep {
            index: 1,
            inverse: true,
        }]
    );
}

#[test]
fn unique_multistep_braid_mutates_one_operation_local_tree() {
    // What: a prepared UniqueFusion schedule has the exact frozen-tree
    // fields and coefficient bits of one-step replay without publishing
    // a frozen intermediate key for every Artin crossing.
    let rule = AsymmetricAnyonicRule;
    let source = FusionTreeKey::try_from_sector_ids(
        [1, 1, 2, 0],
        0,
        [false, true, false, true],
        [2, 0],
        [1, 1, 1],
    )
    .unwrap();
    let permutation = [3, 2, 1, 0];
    let levels = [0, 3, 1, 2];
    let owned = PreparedTreeBraid::new(&permutation, &levels, source.uncoupled().len()).unwrap();

    let mut expected_tree = source.clone();
    let mut expected_coefficient =
        <AsymmetricAnyonicRule as MultiplicityFreeFusionSymbols>::Scalar::one();
    for step in &owned.artin_steps {
        let (next, coefficient) = immutable_unique_artin_braid_at_with_inverse_oracle(
            &rule,
            &expected_tree,
            step.index,
            step.inverse,
        )
        .unwrap();
        expected_tree = next;
        expected_coefficient *= coefficient;
    }
    let assert_replay = |actual: &(FusionTreeKey, f64)| {
        assert_eq!(actual.0.uncoupled(), expected_tree.uncoupled());
        assert_eq!(actual.0.is_dual(), expected_tree.is_dual());
        assert_eq!(actual.0.coupled(), expected_tree.coupled());
        assert_eq!(actual.0.innerlines(), expected_tree.innerlines());
        assert_eq!(actual.0.vertices(), expected_tree.vertices());
        assert_eq!(actual.1.to_bits(), expected_coefficient.to_bits());
    };

    let actual =
        execute_unique_tree_braid(&rule, &source, &owned.permutation, &owned.artin_steps).unwrap();
    assert_replay(&actual);

    let generic = execute_multiplicity_free_tree_braid(
        &rule,
        &source,
        &owned.permutation,
        &owned.artin_steps,
    )
    .unwrap();
    assert_eq!(generic.len(), 1);
    assert_replay(&generic[0]);

    let raw_axis_positions = [3, 2, 1, 0];
    let prepared = PreparedTreePairOperation::prepare_braid_with_raw_axis_positions(
        &rule,
        4,
        0,
        &permutation,
        &[],
        &levels,
        &[],
        &raw_axis_positions,
    )
    .unwrap();
    let PreparedTreePairPlan::UniqueBraid(borrowed) = &prepared.plan else {
        panic!("UniqueFusion braid must retain borrowed operation metadata");
    };
    let actual = execute_unique_tree_braid_borrowed(&rule, &source, borrowed).unwrap();
    assert_replay(&actual);

    let domain = FusionTreeKey::try_new_for_rule(&rule, [], rule.vacuum(), [], [], []).unwrap();
    let second = FusionTreeKey::try_from_sector_ids(
        [1, 1, 2, 0],
        0,
        [true, false, true, false],
        [2, 0],
        [1, 1, 1],
    )
    .unwrap();
    let sources = [
        FusionTreePairKey::pair(source.clone(), domain.clone()),
        FusionTreePairKey::pair(second, domain),
    ];
    let source_snapshots = sources.clone();
    for source in &sources {
        let mut expected_tree = source.codomain_tree().clone();
        let mut expected_coefficient =
            <AsymmetricAnyonicRule as MultiplicityFreeFusionSymbols>::Scalar::one();
        for step in &owned.artin_steps {
            let (next, coefficient) = immutable_unique_artin_braid_at_with_inverse_oracle(
                &rule,
                &expected_tree,
                step.index,
                step.inverse,
            )
            .unwrap();
            expected_tree = next;
            expected_coefficient *= coefficient;
        }
        let actual = prepared.execute_multiplicity_free(&rule, source).unwrap();
        assert_eq!(actual.len(), 1);
        assert_eq!(
            actual[0].0,
            FusionTreePairKey::pair(expected_tree, source.domain_tree().clone())
        );
        assert_eq!(actual[0].1.to_bits(), expected_coefficient.to_bits());
    }
    assert_eq!(sources, source_snapshots);
}

#[test]
fn borrowed_unique_metadata_rejects_inconsistent_inverse_positions() {
    // What: the doc-hidden safe lowering entry points reject a same-length
    // inverse table that would otherwise erase a nontrivial swap.
    let expected = CoreError::InconsistentAxisPosition {
        logical_axis: 1,
        expected_position: 0,
        actual_position: 1,
    };
    assert_eq!(
        PreparedTreePairOperation::prepare_permute_with_raw_axis_positions(
            &Z2FusionRule,
            2,
            0,
            &[1, 0],
            &[],
            &[0, 1],
        ),
        Err(expected.clone())
    );
    assert_eq!(
        PreparedTreePairOperation::prepare_braid_with_raw_axis_positions(
            &Z2FusionRule,
            2,
            0,
            &[1, 0],
            &[],
            &[0, 1],
            &[],
            &[0, 1],
        ),
        Err(expected)
    );
    assert_eq!(
        PreparedTreePairOperation::prepare_permute_with_raw_axis_positions(
            &Z2FusionRule,
            2,
            0,
            &[0, 0],
            &[],
            &[0, 1],
        ),
        Err(CoreError::InvalidPermutation {
            permutation: vec![0, 0],
            rank: 2,
        })
    );
}

#[test]
fn prepared_permute_revalidates_symmetric_capability_for_reused_rule() {
    // What: a prepared permutation cannot become an identity, repartition,
    // or general braid when executed with a non-symmetric provider.
    let source =
        FusionTreePairKey::try_pair_from_sector_ids([1], [1], 1, [false], [false], [], [], [], [])
            .unwrap();
    let plans = [
        PreparedTreePairOperation::prepare_permute(&Z2FusionRule, 1, 1, &[0], &[1]).unwrap(),
        PreparedTreePairOperation::prepare_permute(&Z2FusionRule, 1, 1, &[0, 1], &[]).unwrap(),
        PreparedTreePairOperation::prepare_permute(&Z2FusionRule, 1, 1, &[1], &[0]).unwrap(),
    ];
    assert!(matches!(plans[0].plan, PreparedTreePairPlan::Identity));
    assert!(matches!(plans[1].plan, PreparedTreePairPlan::Repartition));
    assert!(matches!(
        plans[2].plan,
        PreparedTreePairPlan::Braid(_) | PreparedTreePairPlan::UniqueBraid(_)
    ));

    for prepared in plans {
        assert_eq!(
            prepared.execute_unique_rigid(&AsymmetricAnyonicRule, &source),
            Err(CoreError::UnsupportedBraidingStyle {
                expected: "symmetric braiding",
                actual: BraidingStyleKind::Anyonic,
            })
        );
    }
}

#[test]
fn prepared_transpose_fixes_both_cycle_directions_once() {
    // What: clockwise and anticlockwise TensorKit cyclic permutations are
    // lowered to one fixed direction/count before any source execution.
    let source = FusionTreePairKey::try_pair_from_sector_ids(
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
    .unwrap();
    let cases = [
        (&[1, 3][..], &[0, 2][..], PreparedCycleDirection::Clockwise),
        (
            &[2, 0][..],
            &[3, 1][..],
            PreparedCycleDirection::Anticlockwise,
        ),
    ];
    for (codomain, domain, expected_direction) in cases {
        let prepared =
            PreparedTreePairOperation::prepare_transpose(2, 2, codomain, domain).unwrap();
        assert_eq!(
            prepared.plan,
            PreparedTreePairPlan::Transpose {
                direction: expected_direction,
                count: 1,
            }
        );
        let actual = prepared
            .execute_unique_rigid(&Z2FusionRule, &source)
            .unwrap();
        let repartitioned =
            unique_rigid_repartition_tree_pair_unchecked(&Z2FusionRule, &source, codomain.len())
                .unwrap();
        let (oracle_tree, cycle_coefficient) = match expected_direction {
            PreparedCycleDirection::Clockwise => unique_rigid_cycle_tree_pair(
                &Z2FusionRule,
                &repartitioned.0,
                PreparedCycleDirection::Clockwise,
            )
            .unwrap(),
            PreparedCycleDirection::Anticlockwise => unique_rigid_cycle_tree_pair(
                &Z2FusionRule,
                &repartitioned.0,
                PreparedCycleDirection::Anticlockwise,
            )
            .unwrap(),
        };
        let oracle = (oracle_tree, repartitioned.1 * cycle_coefficient);
        assert_eq!(actual, oracle);
    }
}

#[test]
fn unique_prepared_executor_rejects_simple_for_every_plan_variant() {
    // What: a general multiplicity-free plan never becomes a Unique plan
    // merely because its operation variant is an identity or repartition.
    let source =
        FusionTreePairKey::try_pair_from_sector_ids([1], [1], 1, [false], [false], [], [], [], [])
            .unwrap();
    let plans = [
        PreparedTreePairOperation::prepare_braid(&SU2FusionRule, 1, 1, &[0], &[1], &[0], &[1])
            .unwrap(),
        PreparedTreePairOperation::prepare_braid(&SU2FusionRule, 1, 1, &[0, 1], &[], &[0], &[1])
            .unwrap(),
        PreparedTreePairOperation::prepare_braid(&SU2FusionRule, 1, 1, &[1], &[0], &[0], &[1])
            .unwrap(),
        PreparedTreePairOperation::prepare_transpose(1, 1, &[1], &[0]).unwrap(),
    ];
    assert!(matches!(plans[0].plan, PreparedTreePairPlan::Identity));
    assert!(matches!(plans[1].plan, PreparedTreePairPlan::Repartition));
    assert!(matches!(plans[2].plan, PreparedTreePairPlan::Braid(_)));
    assert!(matches!(
        plans[3].plan,
        PreparedTreePairPlan::Transpose { .. }
    ));

    for prepared in plans {
        assert_eq!(
            prepared.execute_unique_rigid(&SU2FusionRule, &source),
            Err(CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Unique,
                actual: FusionStyleKind::Simple,
            })
        );
    }
}

#[test]
fn prepared_operation_preserves_validation_error_precedence() {
    // What: level dimensions still fail before identity short-circuiting,
    // while transpose still reports the original unlinearized permutation.
    assert_eq!(
        PreparedTreePairOperation::prepare_braid(&SU2FusionRule, 1, 1, &[0], &[1], &[], &[1],),
        Err(CoreError::DimensionMismatch {
            expected: 1,
            actual: 0,
        })
    );
    assert_eq!(
        PreparedTreePairOperation::prepare_transpose(2, 1, &[0, 2], &[2]),
        Err(CoreError::InvalidPermutation {
            permutation: vec![0, 2, 2],
            rank: 3,
        })
    );
}

#[test]
fn validation_only_tree_pair_syntax_handles_large_and_rank_zero_maps() {
    let codomain_rank = 10;
    let domain_rank = 9;
    let codomain = (0..codomain_rank).collect::<Vec<_>>();
    let domain = (codomain_rank..codomain_rank + domain_rank).collect::<Vec<_>>();
    let codomain_levels = (0..codomain_rank).collect::<Vec<_>>();
    let domain_levels = (codomain_rank..codomain_rank + domain_rank).collect::<Vec<_>>();

    // What: the validation-only API handles ranks beyond SmallVec's inline
    // permutation capacity without requiring a prepared Artin plan.
    PreparedTreePairOperation::validate_permute_syntax(
        codomain_rank,
        domain_rank,
        &codomain,
        &domain,
    )
    .unwrap();
    PreparedTreePairOperation::validate_braid_syntax(
        codomain_rank,
        domain_rank,
        &codomain,
        &domain,
        &codomain_levels,
        &domain_levels,
    )
    .unwrap();
    PreparedTreePairOperation::validate_transpose_syntax(
        codomain_rank,
        domain_rank,
        &codomain,
        &domain,
    )
    .unwrap();

    // What: the empty tensor map remains a valid identity operation.
    PreparedTreePairOperation::validate_permute_syntax(0, 0, &[], &[]).unwrap();
    PreparedTreePairOperation::validate_braid_syntax(0, 0, &[], &[], &[], &[]).unwrap();
    PreparedTreePairOperation::validate_transpose_syntax(0, 0, &[], &[]).unwrap();
}

#[test]
fn unique_transpose_tree_pair_is_cyclic_and_reversible() {
    let source =
        FusionTreePairKey::try_pair_from_sector_ids([1], [1], 1, [false], [true], [], [], [], [])
            .unwrap();

    let forward_terms =
        multiplicity_free_transpose_tree_pair(&Z2FusionRule, &source, &[1], &[0]).unwrap();
    assert_eq!(forward_terms.len(), 1);
    let (transposed, coefficient) = forward_terms.into_iter().next().unwrap();
    let roundtrip_terms =
        multiplicity_free_transpose_tree_pair(&Z2FusionRule, &transposed, &[1], &[0]).unwrap();
    assert_eq!(roundtrip_terms.len(), 1);
    let (roundtrip, inverse_coefficient) = roundtrip_terms.into_iter().next().unwrap();

    assert_eq!(coefficient, 1.0);
    assert_eq!(inverse_coefficient, 1.0);
    assert_eq!(roundtrip, source);
}

#[test]
fn unique_transpose_tree_pair_matches_tensorkit_clockwise_cycle() {
    let source = FusionTreePairKey::try_pair_from_sector_ids(
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
    .unwrap();
    let expected = FusionTreePairKey::try_pair_from_sector_ids(
        [0, 0],
        [1, 1],
        0,
        [false, true],
        [true, false],
        [],
        [],
        [1],
        [1],
    )
    .unwrap();

    let terms =
        multiplicity_free_transpose_tree_pair(&Z2FusionRule, &source, &[1, 3], &[0, 2]).unwrap();
    assert_eq!(terms.len(), 1);
    let (transposed, coefficient) = terms.into_iter().next().unwrap();

    assert_eq!(coefficient, 1.0);
    assert_eq!(transposed, expected);
}

#[test]
fn unique_transpose_tree_pair_matches_tensorkit_anticlockwise_cycle() {
    let source = FusionTreePairKey::try_pair_from_sector_ids(
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
    .unwrap();
    let expected = FusionTreePairKey::try_pair_from_sector_ids(
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
    .unwrap();

    let terms =
        multiplicity_free_transpose_tree_pair(&Z2FusionRule, &source, &[2, 0], &[3, 1]).unwrap();
    assert_eq!(terms.len(), 1);
    let (transposed, coefficient) = terms.into_iter().next().unwrap();

    assert_eq!(coefficient, 1.0);
    assert_eq!(transposed, expected);
}

#[test]
fn unique_transpose_tree_pair_rejects_noncyclic_permutation() {
    let source = FusionTreePairKey::try_pair_from_sector_ids(
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
    .unwrap();

    let err =
        multiplicity_free_transpose_tree_pair(&Z2FusionRule, &source, &[0, 2], &[1]).unwrap_err();

    assert_eq!(
        err,
        CoreError::InvalidPermutation {
            permutation: vec![0, 2, 1],
            rank: 3,
        }
    );
}
