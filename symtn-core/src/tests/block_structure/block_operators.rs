use super::*;

fn fibonacci_multi_associator_counterexample() -> (FusionTreeKey, FusionTreeKey) {
    let long =
        FusionTreeKey::try_from_sector_ids([1; 4], 1, [false; 4], [1, 0], [1, 1, 1]).unwrap();
    let short = FusionTreeKey::try_from_sector_ids([1; 3], 1, [false; 3], [0], [1, 1]).unwrap();
    let tau = SectorId::new(1);
    assert!(collect_fusion_trees_for_coupled(
        &TableRule::fibonacci(),
        &[tau; 4],
        &[false; 4],
        &[tau; 4],
        tau,
    )
    .contains(&long));
    assert!(collect_fusion_trees_for_coupled(
        &TableRule::fibonacci(),
        &[tau; 3],
        &[false; 3],
        &[tau; 3],
        tau,
    )
    .contains(&short));
    (long, short)
}

#[test]
fn fibonacci_multi_associator_filters_cross_inadmissible_candidates() {
    let rule = FibonacciFAdmissibilityProbe::new();
    let (long, short) = fibonacci_multi_associator_counterexample();
    let first = long.uncoupled()[0];
    let right = long.uncoupled()[2];
    let (middle_left, middle_right) = fusion_tree_vertex_neighbors(&long, 2).unwrap();
    let (short_left, short_right) = fusion_tree_vertex_neighbors(&short, 1).unwrap();

    // What: both stored trees are valid, but their staged cross vertex is
    // absent and therefore does not name an F-symbol coefficient.
    assert_ne!(rule.nsymbol(middle_left, right, middle_right), 0);
    assert_ne!(rule.nsymbol(short_left, right, short_right), 0);
    assert_ne!(rule.nsymbol(first, short_left, middle_left), 0);
    assert_eq!(rule.nsymbol(first, short_right, middle_right), 0);
    assert_eq!(
        rule.f_symbol_scalar(
            first,
            short_left,
            right,
            middle_right,
            middle_left,
            short_right,
        ),
        FibonacciFAdmissibilityProbe::SENTINEL
    );
    assert_eq!(rule.take_calls().len(), 1);

    assert_eq!(
        multiplicity_free_multi_associator_scalar(&rule, &long, &short).unwrap(),
        None
    );
    assert!(rule.take_calls().is_empty());
}

#[test]
fn fibonacci_multi_fmove_forward_and_inverse_call_only_admissible_f() {
    let rule = FibonacciFAdmissibilityProbe::new();
    let (long, short) = fibonacci_multi_associator_counterexample();
    let actual_forward = SimpleK(&rule).multi_fmove(&long).unwrap();
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    // What: TensorKit's Stage-1 candidate intersection retains these two
    // tails in this order; Stage 2 gives the listed Fibonacci F products.
    assert_eq!(
        actual_forward
            .iter()
            .map(|(tree, _)| (tree.coupled(), tree.innerlines().to_vec()))
            .collect::<Vec<_>>(),
        vec![
            (SectorId::new(0), vec![SectorId::new(1)]),
            (SectorId::new(1), vec![SectorId::new(1)]),
        ]
    );
    assert!((actual_forward[0].1 - Complex64::new(1.0 / phi, 0.0)).norm() < 1.0e-12);
    assert!((actual_forward[1].1 - Complex64::new(1.0 / phi.sqrt(), 0.0)).norm() < 1.0e-12);
    let calls = rule.take_calls();
    assert!(!calls.is_empty());
    assert_fibonacci_f_calls_are_admissible(&calls);

    let actual_inverse = SimpleK(&rule)
        .multi_fmove_inv(SectorId::new(1), SectorId::new(1), &short, false)
        .unwrap();
    // What: TensorKit's right-to-left inverse construction retains these
    // two rank-4 trees in canonical order and conjugates the same real
    // Fibonacci coefficients.
    assert_eq!(
        actual_inverse
            .iter()
            .map(|(tree, _)| (tree.coupled(), tree.innerlines().to_vec()))
            .collect::<Vec<_>>(),
        vec![
            (SectorId::new(1), vec![SectorId::new(0), SectorId::new(1)]),
            (SectorId::new(1), vec![SectorId::new(1), SectorId::new(1)]),
        ]
    );
    assert!((actual_inverse[0].1 - Complex64::new(1.0 / phi, 0.0)).norm() < 1.0e-12);
    assert!((actual_inverse[1].1 - Complex64::new(1.0 / phi.sqrt(), 0.0)).norm() < 1.0e-12);
    let calls = rule.take_calls();
    assert!(!calls.is_empty());
    assert_fibonacci_f_calls_are_admissible(&calls);
}

#[test]
fn grouped_multi_fmove_matches_legacy_order_and_reuses_stage_symbols() {
    let rule = FibonacciFAdmissibilityProbe::with_complex_f_phase();
    let tau = SectorId::new(1);
    let trees = collect_fusion_trees_for_coupled(&rule, &[tau; 6], &[false; 6], &[tau; 6], tau);
    let mut fixture = None;
    for tree in trees {
        let grouped = SimpleK(&rule).multi_fmove(&tree).unwrap();
        let grouped_calls = rule.take_calls();
        let legacy = multiplicity_free_multi_fmove_tree_legacy_oracle(&rule, &tree).unwrap();
        let legacy_calls = rule.take_calls();
        if grouped_calls.len() < legacy_calls.len() {
            fixture = Some((tree, grouped, grouped_calls, legacy, legacy_calls));
            break;
        }
    }
    let (tree, grouped, grouped_calls, legacy, legacy_calls) =
        fixture.expect("rank-six Fibonacci must repeat stage-local F arguments");

    // What: grouped forward execution preserves the ungrouped oracle's candidate order
    // and coefficients while evaluating one complete F sextuple per stage.
    assert_eq!(grouped, legacy);
    assert_eq!(grouped_calls.len(), 7);
    assert_eq!(legacy_calls.len(), 18);
    assert_fibonacci_f_calls_are_admissible(&grouped_calls);
    assert!(grouped
        .iter()
        .any(|(_, coefficient)| coefficient.im.abs() > 1.0e-12));

    let tail = grouped
        .first()
        .expect("rank-six Fibonacci forward move has a tail")
        .0
        .clone();
    let grouped_inverse = SimpleK(&rule)
        .multi_fmove_inv(tau, tree.coupled(), &tail, false)
        .unwrap();
    let grouped_inverse_calls = rule.take_calls();
    let legacy_inverse = multiplicity_free_multi_fmove_inv_tree_legacy_oracle(
        &rule,
        tau,
        tree.coupled(),
        &tail,
        false,
    )
    .unwrap();
    let legacy_inverse_calls = rule.take_calls();

    // What: inverse execution uses the same canonical candidates and applies
    // conjugation after the same grouped associator products.
    assert_eq!(grouped_inverse, legacy_inverse);
    assert!(grouped_inverse_calls.len() <= legacy_inverse_calls.len());
    assert_fibonacci_f_calls_are_admissible(&grouped_inverse_calls);
    assert!(grouped_inverse
        .iter()
        .any(|(_, coefficient)| coefficient.im.abs() > 1.0e-12));
}

#[test]
fn fibonacci_braid_then_inverse_braid_is_identity() {
    // Self-consistency (a): braiding a crossing and then undoing it
    // (reflected levels select the inverse-artin branch) must return the
    // exact original tree with total coefficient 1 — this only holds
    // because R^{ττ}_* is a genuine unit-modulus phase.
    let rule = TableRule::fibonacci();
    for coupled in [0usize, 1usize] {
        let tree =
            FusionTreeKey::try_from_sector_ids([1, 1], coupled, [false, false], [], [1]).unwrap();

        let forward = multiplicity_free_braid_tree(&rule, &tree, &[1, 0], &[0, 1]).unwrap();
        assert_eq!(forward.len(), 1);
        let backward =
            multiplicity_free_braid_tree(&rule, &forward[0].0, &[1, 0], &[1, 0]).unwrap();
        assert_eq!(backward.len(), 1);

        assert_eq!(backward[0].0.uncoupled(), tree.uncoupled());
        assert_eq!(backward[0].0.coupled(), tree.coupled());
        let total = forward[0].1 * backward[0].1;
        assert!((total - Complex64::new(1.0, 0.0)).norm() < 1.0e-12);
    }

    // Same check through the rank > 2 loop branch, where the round trip
    // additionally exercises the F-symbol: this only returns to the
    // identity because TensorKitSectors' F^{τττ}_τ block is a genuine
    // (real, orthogonal) unitary matrix.
    let tree = FusionTreeKey::try_from_sector_ids([1, 1, 1], 1, [false, false, false], [1], [1, 1])
        .unwrap();
    let forward = multiplicity_free_braid_tree(&rule, &tree, &[0, 2, 1], &[0, 1, 2]).unwrap();
    assert_eq!(forward.len(), 2);
    let mut total = Complex64::new(0.0, 0.0);
    for (intermediate, coeff) in &forward {
        let backward =
            multiplicity_free_braid_tree(&rule, intermediate, &[0, 2, 1], &[0, 2, 1]).unwrap();
        for (roundtrip, back_coeff) in &backward {
            if roundtrip.innerlines() == tree.innerlines() {
                total += *coeff * *back_coeff;
            }
        }
    }
    assert!((total - Complex64::new(1.0, 0.0)).norm() < 1.0e-12);
}

#[test]
fn tree_pair_block_apis_reject_mixed_product_sector_components() {
    type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
    let rule = FpU1Rule::default();
    let sector_a = rule.encode_component_ids(z2_even(), u1(2)).id();
    let sector_b = rule.encode_component_ids(z2_even(), u1(3)).id();
    let coupled_a = rule.encode_component_ids(z2_even(), u1(4)).id();
    let coupled_b = rule.encode_component_ids(z2_even(), u1(5)).id();
    let keys = [
        tree_pair_group_fixture(
            &[sector_a, sector_a],
            &[coupled_a],
            coupled_a,
            &[false; 2],
            &[false],
        ),
        tree_pair_group_fixture(
            &[sector_a, sector_b],
            &[coupled_b],
            coupled_b,
            &[false; 2],
            &[false],
        ),
    ];

    // What: a changed component of an interned product-sector label is a
    // different shared basis group, even when ranks and duality match.
    assert_mixed_tree_pair_block_group_is_rejected(
        &rule,
        &keys,
        CoreError::MalformedFusionTree {
            message: TREE_PAIR_BLOCK_GROUP_ERROR,
        },
    );
}

#[test]
fn tree_pair_block_empty_and_valid_group_identity_policies_are_stable() {
    let rule = IdentitySymbolPanicRule;
    let empty: &[FusionTreePairKey] = &[];
    // What: an empty block remains a valid empty coefficient transform.
    assert_eq!(
        multiplicity_free_braid_tree_pair_block(&rule, empty, &[], &[], &[], &[]).unwrap(),
        Vec::<Vec<(FusionTreePairKey, f64)>>::new()
    );
    assert_eq!(
        multiplicity_free_permute_tree_pair_block(&rule, empty, &[], &[]).unwrap(),
        Vec::<Vec<(FusionTreePairKey, f64)>>::new()
    );
    assert_eq!(
        multiplicity_free_transpose_tree_pair_block(&rule, empty, &[], &[]).unwrap(),
        Vec::<Vec<(FusionTreePairKey, f64)>>::new()
    );

    let half = su2(1).id();
    let keys = vec![
        tree_pair_group_fixture(
            &[half, half],
            &[half, half],
            su2(0).id(),
            &[false; 2],
            &[false; 2],
        ),
        tree_pair_group_fixture(
            &[half, half],
            &[half, half],
            su2(2).id(),
            &[false; 2],
            &[false; 2],
        ),
    ];
    let expected = keys
        .iter()
        .cloned()
        .map(|key| vec![(key, 1.0)])
        .collect::<Vec<_>>();
    // What: distinct coupled labels remain valid basis states in one
    // external-sector group, preserving source order, destinations, and
    // coefficients on the symbol-free identity path.
    assert_eq!(
        multiplicity_free_braid_tree_pair_block(
            &SU2FusionRule,
            &keys,
            &[0, 1],
            &[2, 3],
            &[0, 1],
            &[2, 3],
        )
        .unwrap(),
        expected
    );
}

#[test]
fn indexed_adjoint_tree_pair_block_reuses_parent_group_order() {
    let half = su2(1).id();
    let keys = [
        tree_pair_group_fixture(
            &[half, half],
            &[half, half],
            su2(0).id(),
            &[false; 2],
            &[false; 2],
        ),
        tree_pair_group_fixture(
            &[half, half],
            &[half, half],
            su2(0).id(),
            &[true, false],
            &[false; 2],
        ),
        tree_pair_group_fixture(
            &[half, half],
            &[half, half],
            su2(2).id(),
            &[false; 2],
            &[false; 2],
        ),
        tree_pair_group_fixture(
            &[half, half],
            &[half, half],
            su2(2).id(),
            &[true, false],
            &[false; 2],
        ),
    ];
    let structure =
        packed_fixture_structure(4, keys.iter().cloned().map(|key| (key, vec![1; 4]))).unwrap();
    let eager_adjoint_keys = keys
        .iter()
        .map(|key| FusionTreePairKey::pair(key.domain_tree().clone(), key.codomain_tree().clone()))
        .collect::<Vec<_>>();
    let eager_adjoint = packed_fixture_structure(
        4,
        eager_adjoint_keys
            .iter()
            .cloned()
            .map(|key| (key, vec![1; 4])),
    )
    .unwrap();

    // What: side swapping is a bijection on group labels, so an adjoint
    // projection preserves the parent group partition and storage order.
    assert_eq!(
        structure
            .fusion_tree_group_slice()
            .iter()
            .map(FusionTreeBlockGroup::block_indices)
            .collect::<Vec<_>>(),
        eager_adjoint
            .fusion_tree_group_slice()
            .iter()
            .map(FusionTreeBlockGroup::block_indices)
            .collect::<Vec<_>>(),
    );

    for group in structure.fusion_tree_group_slice() {
        let columns = multiplicity_free_permute_tree_pair_block_indexed(
            &SU2FusionRule,
            &structure,
            group.block_indices(),
            FusionTreePairOrientation::Adjoint,
            &[0, 1],
            &[2, 3],
        )
        .unwrap();
        let rows = (0..columns.source_count())
            .map(|source| {
                columns
                    .column(source)
                    .iter()
                    .map(|(row, coefficient)| (columns.destinations()[*row].clone(), *coefficient))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        // What: indexed adjoint identity emits logical swapped keys once,
        // in parent source order, without changing compact transform rows.
        assert_eq!(
            rows,
            group
                .block_indices()
                .iter()
                .map(|&index| vec![(eager_adjoint_keys[index].clone(), 1.0)])
                .collect::<Vec<_>>()
        );

        let prepared =
            PreparedTreePairOperation::prepare_permute(&SU2FusionRule, 2, 2, &[0, 1], &[2, 3])
                .unwrap();
        let ordered = multiplicity_free_braid_tree_pair_block_ordered_indexed(
            &SU2FusionRule,
            &structure,
            group.block_indices(),
            FusionTreePairOrientation::Adjoint,
            &prepared,
        )
        .unwrap();
        // What: the ordered indexed kernel exposes the same logical
        // adjoint columns while retaining parent block-index order.
        assert_eq!(ordered.source_count(), group.block_indices().len());
        assert_eq!(
            ordered.destinations(),
            group
                .block_indices()
                .iter()
                .map(|&index| eager_adjoint_keys[index].clone())
                .collect::<Vec<_>>()
        );

        let prepared =
            PreparedTreePairOperation::prepare_transpose(2, 2, &[0, 1], &[2, 3]).unwrap();
        let transposed = multiplicity_free_transpose_tree_pair_block_ordered_indexed(
            &SU2FusionRule,
            &structure,
            group.block_indices(),
            FusionTreePairOrientation::Adjoint,
            &prepared,
        )
        .unwrap();
        // What: transpose uses the same oriented parent projection and
        // preserves the canonical logical source-column count.
        assert_eq!(transposed.source_count(), group.block_indices().len());
        assert_eq!(transposed.destinations(), ordered.destinations());
    }
}

#[test]
fn empty_block_permute_preserves_braiding_style_error_precedence() {
    // What: an empty source block does not bypass the public permutation
    // API's symmetric-braiding capability check.
    assert_eq!(
        multiplicity_free_permute_tree_pair_block(&TableRule::fibonacci(), &[], &[], &[],)
            .unwrap_err(),
        CoreError::UnsupportedBraidingStyle {
            expected: "symmetric braiding",
            actual: BraidingStyleKind::Anyonic,
        }
    );
}

#[test]
fn split_only_tree_pair_braid_uses_only_the_required_bend() {
    // What: moving the split from 1|2 to 2|1 with unchanged linearized
    // external-leg order evaluates one bend and no braid symbols, for both
    // the single-source and block entry points.
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
    let single =
        multiplicity_free_braid_tree_pair(&rule, &source, &[0, 2], &[1], &[0], &[1, 2]).unwrap();
    assert_eq!(rule.f_calls.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(rule.r_calls.load(std::sync::atomic::Ordering::Relaxed), 0);

    rule.f_calls.store(0, std::sync::atomic::Ordering::Relaxed);
    let block = multiplicity_free_braid_tree_pair_block(
        &rule,
        std::slice::from_ref(&source),
        &[0, 2],
        &[1],
        &[0],
        &[1, 2],
    )
    .unwrap();
    assert_eq!(rule.f_calls.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(rule.r_calls.load(std::sync::atomic::Ordering::Relaxed), 0);
    assert_eq!(block, vec![single]);
}

#[test]
fn split_only_nested_product_braid_matches_legacy_composition() {
    // What: a non-Abelian fZ2 x U(1) x SU(2) tree preserves the product
    // bend sign and duality bookkeeping of the all-codomain reference route in
    // both split directions.
    type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
    type ProductRule = ProductFusionRule<FpU1Rule, SU2FusionRule>;
    let left_rule = FpU1Rule::default();
    let rule = ProductRule::default();
    let coupled =
        rule.encode_component_ids(left_rule.encode_component_ids(z2_even(), u1(0)), su2(1));
    let domain_left =
        rule.encode_component_ids(left_rule.encode_component_ids(z2_odd(), u1(1)), su2(1));
    let domain_right =
        rule.encode_component_ids(left_rule.encode_component_ids(z2_odd(), u1(-1)), su2(2));
    let source = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(&rule, [coupled], coupled, [false], [], []).unwrap(),
        FusionTreeKey::try_new_for_rule(
            &rule,
            [domain_left, domain_right],
            coupled,
            [false, true],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
    );

    let forward =
        multiplicity_free_braid_tree_pair(&rule, &source, &[0, 2], &[1], &[0], &[1, 2]).unwrap();
    let expected = legacy_split_only_tree_pair_route(&rule, &source, 2).unwrap();
    assert_eq!(forward.len(), expected.len());
    assert!(forward[0].1 < 0.0);
    for ((actual_key, actual_coefficient), (expected_key, expected_coefficient)) in
        forward.iter().zip(&expected)
    {
        assert_eq!(actual_key, expected_key);
        assert!((actual_coefficient - expected_coefficient).abs() < 1.0e-12);
    }

    let reverse_source = &forward[0].0;
    let reverse =
        multiplicity_free_braid_tree_pair(&rule, reverse_source, &[0], &[2, 1], &[0, 1], &[2])
            .unwrap();
    let reverse_expected = legacy_split_only_tree_pair_route(&rule, reverse_source, 1).unwrap();
    assert_eq!(reverse.len(), reverse_expected.len());
    assert!(reverse[0].1 < 0.0);
    for ((actual_key, actual_coefficient), (expected_key, expected_coefficient)) in
        reverse.iter().zip(&reverse_expected)
    {
        assert_eq!(actual_key, expected_key);
        assert!((actual_coefficient - expected_coefficient).abs() < 1.0e-12);
    }
}

#[test]
fn identity_braid_rows_are_exact_for_supported_symmetry_families_and_rank_zero() {
    // What: identity rows preserve their exact source key and unit
    // coefficient for fermionic, non-abelian, product, and scalar spaces.
    let pair = |sector: SectorId| {
        FusionTreePairKey::try_pair_from_sector_ids(
            [sector.id()],
            [sector.id()],
            sector.id(),
            [false],
            [false],
            [],
            [],
            [],
            [],
        )
        .unwrap()
    };

    let fz2_source = pair(z2_odd());
    assert_eq!(
        multiplicity_free_braid_tree_pair(
            &FermionParityFusionRule,
            &fz2_source,
            &[0],
            &[1],
            &[41],
            &[2],
        )
        .unwrap(),
        vec![(fz2_source, 1.0)]
    );

    let su2_source = pair(su2(1));
    assert_eq!(
        multiplicity_free_braid_tree_pair(&SU2FusionRule, &su2_source, &[0], &[1], &[13], &[5],)
            .unwrap(),
        vec![(su2_source, 1.0)]
    );

    type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
    type FpU1Su2Rule = ProductFusionRule<FpU1Rule, SU2FusionRule>;
    let left_rule = FpU1Rule::default();
    let product_rule = FpU1Su2Rule::default();
    let product_sector =
        product_rule.encode_component_ids(left_rule.encode_component_ids(z2_odd(), u1(2)), su2(1));
    let product_source = pair(product_sector);
    assert_eq!(
        multiplicity_free_braid_tree_pair(&product_rule, &product_source, &[0], &[1], &[8], &[3],)
            .unwrap(),
        vec![(product_source, 1.0)]
    );

    let scalar_source = FusionTreePairKey::try_pair_from_sector_ids(
        Vec::<usize>::new(),
        Vec::<usize>::new(),
        z2_even().id(),
        Vec::<bool>::new(),
        Vec::<bool>::new(),
        Vec::<usize>::new(),
        Vec::<usize>::new(),
        Vec::<usize>::new(),
        Vec::<usize>::new(),
    )
    .unwrap();
    assert_eq!(
        multiplicity_free_braid_tree_pair(&Z2FusionRule, &scalar_source, &[], &[], &[], &[],)
            .unwrap(),
        vec![(scalar_source, 1.0)]
    );
}

#[test]
fn borrowed_unique_schedule_derivation_uses_quadratic_position_queries() {
    // What: operation-local inverse positions keep lazy Artin schedule
    // derivation quadratic in runtime rank; this deliberately measures
    // schedule lowering only, not the subsequent tree execution.
    let rank = 129;
    let permutation = (0..rank).rev().collect::<Vec<_>>();
    let raw_axis_positions = (0..rank).rev().collect::<Vec<_>>();
    let prepared = PreparedTreePairOperation::prepare_permute_with_raw_axis_positions(
        &Z2FusionRule,
        rank,
        0,
        &permutation,
        &[],
        &raw_axis_positions,
    )
    .unwrap();

    let recording = crate::probe::record();
    let steps = prepared
        .plan
        .artin_steps()
        .expect("reverse permutation must prepare a braid")
        .count();
    assert_eq!(steps, rank * (rank - 1) / 2);
    assert_eq!(
        recording.get(crate::probe::Event::UniqueBorrowedPositionQuery),
        rank * (rank - 1)
    );
}
