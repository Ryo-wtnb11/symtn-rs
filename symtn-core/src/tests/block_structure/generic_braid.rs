use super::*;

#[test]
fn toy_om_rule_nsymbol_reports_outer_multiplicity() {
    let rule = ToyOmRule;
    let a = SectorId::new(ToyOmRule::A);
    let c = SectorId::new(ToyOmRule::C);
    let vacuum = SectorId::new(ToyOmRule::VACUUM);

    assert_eq!(rule.fusion_style(), FusionStyleKind::Generic);
    assert_eq!(rule.nsymbol(a, a, c), 2);
    // Everything else in this toy rule stays multiplicity-free (0 or 1).
    assert_eq!(rule.nsymbol(a, vacuum, a), 1);
    assert_eq!(rule.nsymbol(a, a, vacuum), 0);
}

#[test]
fn toy_om_rule_f_symbol_generic_has_nsymbol_shaped_block() {
    let rule = ToyOmRule;
    let a = SectorId::new(ToyOmRule::A);
    let c = SectorId::new(ToyOmRule::C);
    let vacuum = SectorId::new(ToyOmRule::VACUUM);

    // F(a,a,0,c,c,a): shape (N(a,a,c), N(c,0,c), N(a,0,a), N(a,a,c))
    // = (2, 1, 1, 2).
    let f = rule.f_symbol_generic(a, a, vacuum, c, c, a);
    assert_eq!(f.shape(), (2, 1, 1, 2));
    assert_eq!(f.data().len(), 4);

    // The (mu, lambda) 2x2 block (nu = kappa = 0) is an orthogonal
    // rotation: R^T R == I.
    let m = [
        [*f.get(0, 0, 0, 0), *f.get(0, 0, 0, 1)],
        [*f.get(1, 0, 0, 0), *f.get(1, 0, 0, 1)],
    ];
    for col in 0..2 {
        for other in 0..2 {
            let dot: f64 = (0..2).map(|row| m[row][col] * m[row][other]).sum();
            let expected = if col == other { 1.0 } else { 0.0 };
            assert!(
                (dot - expected).abs() < 1e-12,
                "F (mu,lambda) block is not orthogonal at columns {col},{other}: {dot}"
            );
        }
    }
}

#[test]
fn toy_om_rule_r_symbol_generic_has_nsymbol_shaped_matrix() {
    let rule = ToyOmRule;
    let a = SectorId::new(ToyOmRule::A);
    let c = SectorId::new(ToyOmRule::C);

    let r = rule.r_symbol_generic(a, a, c);
    assert_eq!(r.shape(), (2, 2));
    assert_eq!(r.data().len(), 4);
}

#[test]
fn frozen_fusion_tree_identity_and_group_projection_are_content_based() {
    // What: independently allocated equal keys compare and hash by content,
    // while group projection reuses the pair's tree handles.
    let a = SectorId::new(ToyOmRule::A);
    let c = SectorId::new(ToyOmRule::C);
    let first = FusionTreeKey::new([a, a], c, [false, true], [], [MultiplicityIndex::ONE]);
    let second = FusionTreeKey::new([a, a], c, [false, true], [], [MultiplicityIndex::ONE]);
    assert!(!Arc::ptr_eq(&first.arena, &second.arena));
    assert_eq!(first, second);
    assert_eq!(first.cmp(&second), std::cmp::Ordering::Equal);

    let mut cache = std::collections::HashMap::new();
    cache.insert(first.clone(), 7usize);
    assert_eq!(cache.get(&second), Some(&7));

    let pair = FusionTreePairKey::pair(first, second);
    let group = pair.group_key();
    // The group holds the pair's own tree handles: no copy, no allocation.
    assert!(Arc::ptr_eq(
        &group.codomain.arena,
        &pair.codomain_tree.arena
    ));
    assert_eq!(group.codomain.index, pair.codomain_tree.index);
    assert!(Arc::ptr_eq(&group.domain.arena, &pair.domain_tree.arena));
    assert_eq!(group.domain.index, pair.domain_tree.index);
}

#[test]
fn tree_pair_axis_validation_remains_linear_above_inline_bitset_capacity() {
    let rule = Z2FusionRule;
    let rank = 129;
    let codomain = FusionTreeKey::try_new_for_rule(
        &rule,
        vec![SectorId::new(0); rank],
        SectorId::new(0),
        vec![false; rank],
        vec![SectorId::new(0); rank - 2],
        vec![MultiplicityIndex::ONE; rank - 1],
    )
    .unwrap();
    let domain = FusionTreeKey::try_new_for_rule(&rule, [], SectorId::new(0), [], [], []).unwrap();
    let pair = FusionTreePairKey::pair(codomain, domain);
    let identity = (0..rank).collect::<Vec<_>>();

    let rows = multiplicity_free_permute_tree_pair(&rule, &pair, &identity, &[]).unwrap();
    assert_eq!(rows, vec![(pair.clone(), 1.0)]);

    let mut duplicate = identity;
    duplicate[rank - 1] = rank - 2;
    assert!(matches!(
        multiplicity_free_permute_tree_pair(&rule, &pair, &duplicate, &[]),
        Err(CoreError::InvalidPermutation { .. })
    ));
}

#[test]
fn checked_generic_tree_split_is_structural_and_provider_typed() {
    let rule = UnitaryToyOmRule;
    let a = SectorId::new(UnitaryToyOmRule::A);
    let c = SectorId::new(UnitaryToyOmRule::C);
    let tree = FusionTreeKey::new(
        [a, a, a],
        a,
        [false, false, false],
        [c],
        [MultiplicityIndex::ONE, MultiplicityIndex::ONE],
    );

    let (front, tail) = split_fusion_tree_generic_checked(&rule, &tree, 1).unwrap();
    assert_eq!(front.uncoupled(), &[a]);
    assert_eq!(front.coupled(), a);
    assert_eq!(tail.uncoupled(), &[a, a, a]);
    assert_eq!(tail.coupled(), a);
    assert_eq!(tail.is_dual(), &[false, false, false]);
}

#[test]
fn checked_generic_artin_matches_unchecked_outer_and_inner_rows() {
    let rule = UnitaryToyOmRule;
    for (tree, index) in [(unitary_rank2_tree(1), 0), (unitary_rank3_tree(1), 1)] {
        let unchecked = generic_artin_braid_at_with_inverse(&rule, &tree, index, false).unwrap();
        let checked =
            generic_artin_braid_at_with_inverse_checked(&rule, &tree, index, false).unwrap();
        assert_eq!(checked, unchecked);
    }
}

#[test]
fn checked_generic_merge_rejects_a_malformed_f_before_emitting_terms() {
    let rule = ArtinSpy {
        bad_f: true,
        ..ArtinSpy::new()
    };
    let lhs = unitary_rank3_tree(1);
    let rhs = unitary_rank2_tree(1);
    let result = merge_fusion_trees_generic_checked(
        &rule,
        &lhs,
        &rhs,
        SectorId::new(UnitaryToyOmRule::A),
        MultiplicityIndex::ONE,
    );
    assert!(
        matches!(
            result,
            Err(CheckedGenericSymbolError::Shape { symbol: "F", .. })
        ),
        "{result:?}"
    );
    assert!(rule.f_calls.get() > 0);
}

// Braid at `index` (inv=false) then braid every output at `index`
// (inv=true), summing coefficients per final tree. TensorKit's inverse
// identity: this must equal the input tree with coefficient 1.
fn braid_then_inverse_braid(
    rule: &UnitaryToyOmRule,
    tree: &FusionTreeKey,
    index: usize,
) -> std::collections::HashMap<FusionTreeKey, f64> {
    let forward = generic_artin_braid_at_with_inverse(rule, tree, index, false).unwrap();
    let mut totals: std::collections::HashMap<FusionTreeKey, f64> =
        std::collections::HashMap::new();
    for (mid, c_forward) in forward {
        let inverse = generic_artin_braid_at_with_inverse(rule, &mid, index, true).unwrap();
        for (final_tree, c_inverse) in inverse {
            *totals.entry(final_tree).or_insert(0.0) += c_forward * c_inverse;
        }
    }
    totals
}

fn assert_is_identity_on(
    totals: &std::collections::HashMap<FusionTreeKey, f64>,
    original: &FusionTreeKey,
) {
    let on_original = totals.get(original).copied().unwrap_or(0.0);
    assert!(
        (on_original - 1.0).abs() < 1e-12,
        "braid × inverse must be 1 on the original tree, got {on_original}"
    );
    for (tree, coeff) in totals {
        if tree != original {
            assert!(
                coeff.abs() < 1e-12,
                "braid × inverse must be 0 off the original tree, got {coeff}"
            );
        }
    }
}

#[test]
fn unitary_toy_om_rule_r_and_f_blocks_are_unitary() {
    // Precondition of the round-trip tests: the F/R blocks the braid uses
    // are genuinely unitary. Assert it here so the identity tests below
    // rest on a checked assumption, not an asserted-by-fiat one.
    let rule = UnitaryToyOmRule;
    let a = SectorId::new(UnitaryToyOmRule::A);
    let c = SectorId::new(UnitaryToyOmRule::C);
    let vacuum = SectorId::new(UnitaryToyOmRule::VACUUM);

    // R(a,a,c): 2×2, must satisfy Rᵀ R = I.
    let r = rule.r_symbol_generic(a, a, c);
    assert_eq!(r.shape(), (2, 2));
    for col in 0..2 {
        for other in 0..2 {
            let dot: f64 = (0..2).map(|row| r.get(row, col) * r.get(row, other)).sum();
            let expected = if col == other { 1.0 } else { 0.0 };
            assert!(
                (dot - expected).abs() < 1e-12,
                "R(a,a,c) is not unitary at columns {col},{other}: {dot}"
            );
        }
    }

    // F(a,a,a,a,c,c): the braid's F block, shape (2,1,2,1) => 2×2 in
    // (mu,kappa). Must be unitary (it is the identity here).
    let f_braid = rule.f_symbol_generic(a, a, a, a, c, c);
    assert_eq!(f_braid.shape(), (2, 1, 2, 1));
    for mu in 0..2 {
        for other in 0..2 {
            let dot: f64 = (0..2)
                .map(|kappa| f_braid.get(mu, 0, kappa, 0) * f_braid.get(other, 0, kappa, 0))
                .sum();
            let expected = if mu == other { 1.0 } else { 0.0 };
            assert!(
                (dot - expected).abs() < 1e-12,
                "F(a,a,a,a,c,c) is not unitary at rows {mu},{other}: {dot}"
            );
        }
    }

    // F(a,a,0,c,c,a): the non-identity unitary F block, (mu,lambda) 2×2.
    let f_rot = rule.f_symbol_generic(a, a, vacuum, c, c, a);
    assert_eq!(f_rot.shape(), (2, 1, 1, 2));
    for col in 0..2 {
        for other in 0..2 {
            let dot: f64 = (0..2)
                .map(|row| f_rot.get(row, 0, 0, col) * f_rot.get(row, 0, 0, other))
                .sum();
            let expected = if col == other { 1.0 } else { 0.0 };
            assert!(
                (dot - expected).abs() < 1e-12,
                "F(a,a,0,c,c,a) is not unitary at columns {col},{other}: {dot}"
            );
        }
    }
}

#[test]
fn generic_braid_index0_output_count_matches_nsymbol() {
    // Test (2): the forward braid of [a,a]->c at index 0 produces exactly
    // N(a,a,c) = 2 output trees, labelled with vertices 1 and 2 (both
    // nonzero because Rθ is a full rotation).
    let rule = UnitaryToyOmRule;
    let tree = unitary_rank2_tree(1);
    let outputs = generic_artin_braid_at_with_inverse(&rule, &tree, 0, false).unwrap();
    assert_eq!(outputs.len(), 2, "expected N(a,a,c)=2 output trees");
    let mut labels: Vec<usize> = outputs.iter().map(|(t, _)| t.vertices()[0].get()).collect();
    labels.sort_unstable();
    assert_eq!(labels, vec![1, 2], "output vertex labels must be {{1,2}}");
}

#[test]
fn generic_braid_index1_output_count_matches_nsymbol() {
    // Test (2), index>0 branch: braid of [a,a,a]->a at index 1 produces the
    // single c'=c channel, σ ∈ {1,2} (N(a,a,c)=2), λ = 1 (N(c,a,a)=1).
    let rule = UnitaryToyOmRule;
    let tree = unitary_rank3_tree(1);
    let outputs = generic_artin_braid_at_with_inverse(&rule, &tree, 1, false).unwrap();
    assert_eq!(outputs.len(), 2, "expected 2 (σ) output trees at index 1");
    let mut labels: Vec<(usize, usize)> = outputs
        .iter()
        .map(|(t, _)| {
            assert_eq!(t.innerlines(), &[SectorId::new(UnitaryToyOmRule::C)]);
            (t.vertices()[0].get(), t.vertices()[1].get())
        })
        .collect();
    labels.sort_unstable();
    assert_eq!(labels, vec![(1, 1), (2, 1)]);
}

#[test]
fn generic_braid_inverse_is_identity_rank2_index0() {
    // Test (1), index==0 branch, over every enumerated vertex assignment.
    let rule = UnitaryToyOmRule;
    for vertex in 1..=2 {
        let tree = unitary_rank2_tree(vertex);
        let totals = braid_then_inverse_braid(&rule, &tree, 0);
        assert_is_identity_on(&totals, &tree);
    }
}

#[test]
fn generic_braid_inverse_is_identity_rank3_index0() {
    // Test (1) + (4): index==0 on a rank>2 tree (uses the innerline as the
    // coupled sector for R), over every OM vertex assignment.
    let rule = UnitaryToyOmRule;
    for vertex1 in 1..=2 {
        let tree = unitary_rank3_tree(vertex1);
        let totals = braid_then_inverse_braid(&rule, &tree, 0);
        assert_is_identity_on(&totals, &tree);
    }
}

#[test]
fn generic_braid_inverse_is_identity_rank3_index1() {
    // Test (1) + (4): the index>1 branch (F-move + R·F̄·R̄ contraction),
    // over every OM vertex assignment.
    let rule = UnitaryToyOmRule;
    for vertex1 in 1..=2 {
        let tree = unitary_rank3_tree(vertex1);
        let totals = braid_then_inverse_braid(&rule, &tree, 1);
        assert_is_identity_on(&totals, &tree);
    }
}

#[test]
fn generic_braid_tree_roundtrip_is_identity() {
    // Exercises `generic_braid_tree`: braid [a,a]->c with the swap
    // permutation under levels [0,1] (=> inv=false), then braid the outputs
    // back under levels [1,0] (=> inv=true). The composite must be identity.
    let rule = UnitaryToyOmRule;
    for vertex in 1..=2 {
        let tree = unitary_rank2_tree(vertex);
        let forward = generic_braid_tree(&rule, &tree, &[1, 0], &[0, 1]).unwrap();
        let mut totals: std::collections::HashMap<FusionTreeKey, f64> =
            std::collections::HashMap::new();
        for (mid, c_forward) in forward {
            let back = generic_braid_tree(&rule, &mid, &[1, 0], &[1, 0]).unwrap();
            for (final_tree, c_back) in back {
                *totals.entry(final_tree).or_insert(0.0) += c_forward * c_back;
            }
        }
        assert_is_identity_on(&totals, &tree);
    }
}

#[test]
fn generic_braid_tree_identity_permutation_is_noop() {
    // The identity permutation decomposes to zero swaps: the tree returns
    // unchanged with coefficient 1.
    let rule = UnitaryToyOmRule;
    let tree = unitary_rank3_tree(2);
    let out = generic_braid_tree(&rule, &tree, &[0, 1, 2], &[0, 1, 2]).unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].0, tree);
    assert!((out[0].1 - 1.0).abs() < 1e-12);
}
