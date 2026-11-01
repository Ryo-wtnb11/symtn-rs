use super::*;

#[test]
fn fusion_tree_homspace_matches_tensorkit_su2_simple_order() {
    let rule = SU2FusionRule;
    let leg = || {
        SectorLeg::new(
            [
                (SectorId::new(0), 1),
                (SectorId::new(1), 1),
                (SectorId::new(2), 1),
            ],
            false,
        )
    };
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg()]),
    );

    let keys = hom.fusion_tree_keys(&rule);

    // TensorKit.jl 6Camk with sector id = twice spin:
    // V=Vect[SU2Irrep](0=>1,1//2=>1,1=>1); W=(V⊗V)←V;
    // [(2f1.uncoupled, 2f2.uncoupled, 2f1.coupled) for (f1,f2) in fusiontrees(W)]
    assert_eq!(
        fusion_tree_pair_order(&keys),
        vec![
            (vec![0, 0], vec![0], 0),
            (vec![1, 1], vec![0], 0),
            (vec![2, 2], vec![0], 0),
            (vec![1, 0], vec![1], 1),
            (vec![0, 1], vec![1], 1),
            (vec![2, 1], vec![1], 1),
            (vec![1, 2], vec![1], 1),
            (vec![2, 0], vec![2], 2),
            (vec![1, 1], vec![2], 2),
            (vec![0, 2], vec![2], 2),
            (vec![2, 2], vec![2], 2),
        ]
    );
    assert!(keys
        .iter()
        .all(|key| key.codomain_vertices() == [MultiplicityIndex::ONE]));
    assert!(keys.iter().all(|key| key.domain_vertices().is_empty()));
}

#[test]
fn ordered_transpose_preserves_custom_source_cohort_order() {
    let rule = FibonacciFAdmissibilityProbe::with_complex_f_phase();
    let tau = || SectorLeg::new([(SectorId::new(1), 1)], false);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([tau(), tau()]),
        FusionProductSpace::new([tau(), tau()]),
    );
    let mut sources = hom.fusion_tree_keys(&rule).as_ref().to_vec();
    assert!(sources.len() > 1);
    sources.reverse();

    // What: a caller-selected source subset/order determines destination
    // first appearance; canonical HomSpace basis order is not substituted.
    assert_compact_transpose_matches_full_key_oracle(&rule, &sources, &[1, 3], &[0, 2], true);
    assert_compact_transpose_matches_full_key_oracle(&rule, &sources[..2], &[1, 3], &[0, 2], true);
}

#[test]
fn transpose_tree_pair_block_matches_fz2_odd_pivotal_oracle() {
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(z2_odd(), 1)], true),
            SectorLeg::new([(z2_odd(), 1)], false),
        ]),
        FusionProductSpace::new([SectorLeg::new([(z2_even(), 1)], false)]),
    );
    let sources = hom.fusion_tree_keys(&FermionParityFusionRule);
    assert!(!sources.is_empty());

    // What: cycling a dual odd leg retains the Frobenius-Schur/pivotal sign.
    assert_compact_transpose_matches_full_key_oracle(
        &FermionParityFusionRule,
        &sources,
        &[1, 2],
        &[0],
        true,
    );
}

#[test]
fn fusion_tree_homspace_matches_tensorkit_su2_innerline_order() {
    let rule = SU2FusionRule;
    let hom = FusionTreeHomSpace::from_sector_ids([(1, 1), (1, 1), (1, 1)], [(1, 1)]);

    let keys = hom.fusion_tree_keys(&rule);

    // TensorKit.jl 6Camk with sector id = twice spin:
    // V=Vect[SU2Irrep](1//2=>1); W=(V⊗V⊗V)←V;
    // codomain innerlines for fusiontrees(W) are [0], then [2].
    assert_eq!(keys.len(), 2);
    assert_eq!(keys[0].codomain_innerlines(), &[SectorId::new(0)]);
    assert_eq!(keys[1].codomain_innerlines(), &[SectorId::new(2)]);
    assert_eq!(
        fusion_tree_pair_order(&keys),
        vec![(vec![1, 1, 1], vec![1], 1), (vec![1, 1, 1], vec![1], 1),]
    );

    let groups = hom.fusion_tree_groups(&rule).unwrap();
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].block_indices(), &[0, 1]);
}

#[test]
fn fusion_tree_homspace_external_sectors_preserve_su2_simple_innerline_order() {
    let rule = SU2FusionRule;
    let half = SectorId::new(1);
    let hom = FusionTreeHomSpace::from_sector_ids([(1, 1), (1, 1), (1, 1)], [(1, 1)]);

    let keys = hom
        .fusion_tree_keys_from_external_sectors(&rule, &[half, half, half, half])
        .unwrap();

    assert_eq!(keys.len(), 2);
    assert_eq!(keys[0].codomain_uncoupled(), &[half, half, half]);
    assert_eq!(keys[0].domain_uncoupled(), &[half]);
    assert_eq!(keys[0].codomain_innerlines(), &[SectorId::new(0)]);
    assert_eq!(keys[1].codomain_innerlines(), &[SectorId::new(2)]);
    assert_eq!(
        fusion_tree_pair_order(&keys),
        vec![(vec![1, 1, 1], vec![1], 1), (vec![1, 1, 1], vec![1], 1),]
    );
}

#[test]
fn fusion_tree_homspace_uses_tensorkit_parent_iterator_order_not_ord_sort() {
    let rule = UnsortedFusionIteratorOrderRule;
    let hom = FusionTreeHomSpace::from_sector_ids([(1, 1), (1, 1), (1, 1)], [(1, 1)]);

    let keys = hom.fusion_tree_keys(&rule);

    // TensorKit rank >= 3 iterator picks the parent line from
    // `coupled ⊗ dual(last)` order. This toy rule returns 1 ⊗ 1 as [2, 0],
    // deliberately opposite to `SectorId` Ord, so an Ord-based replay would
    // produce [0], [2].
    assert_eq!(keys.len(), 2);
    assert_eq!(keys[0].codomain_innerlines(), &[SectorId::new(2)]);
    assert_eq!(keys[1].codomain_innerlines(), &[SectorId::new(0)]);
}

#[test]
fn fusion_tree_homspace_uses_visible_dual_space_sector_label_like_tensorkit() {
    let rule = U1FusionRule;
    let minus_one = U1Irrep::new(-1);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(minus_one, 1)], true)]),
        FusionProductSpace::new([SectorLeg::new([(minus_one, 1)], false)]),
    );

    let keys = hom.fusion_tree_keys(&rule);

    // TensorKit:
    // collect(sectors(Vect[U1Irrep](1=>1)')) == [U1Irrep(-1)]
    // fusiontrees((U1Irrep(-1),), U1Irrep(-1), (true,)) keeps uncoupled = -1.
    assert_eq!(keys.len(), 1);
    assert_eq!(keys[0].coupled(), minus_one.into());
    assert_eq!(keys[0].codomain_uncoupled(), &[minus_one.into()]);
    assert_eq!(keys[0].codomain_is_dual(), &[true]);
    assert_eq!(keys[0].domain_uncoupled(), &[minus_one.into()]);
    assert_eq!(keys[0].domain_is_dual(), &[false]);
}

#[test]
fn fusion_tree_homspace_does_not_dualize_selected_dual_leg_again() {
    let rule = BranchingMultiplicityFreeRule;
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(SectorId::new(1), 1)], true)]),
        FusionProductSpace::from_sector_ids([(1, 1)]),
    );

    let keys = hom.fusion_tree_keys(&rule);

    assert_eq!(keys.len(), 1);
    assert_eq!(keys[0].coupled(), SectorId::new(1));
    assert_eq!(keys[0].codomain_uncoupled(), &[SectorId::new(1)]);
    assert_eq!(keys[0].codomain_is_dual(), &[true]);
    assert_eq!(keys[0].domain_uncoupled(), &[SectorId::new(1)]);
    assert_eq!(keys[0].domain_is_dual(), &[false]);
}

#[test]
fn fusion_tree_homspace_fusionblocks_follow_domain_outer_codomain_inner_order() {
    let rule = BranchingMultiplicityFreeRule;
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(SectorId::new(1), 1), (SectorId::new(2), 1)], false),
            SectorLeg::new([(SectorId::new(1), 1)], false),
        ]),
        FusionProductSpace::new([SectorLeg::new(
            [(SectorId::new(1), 1), (SectorId::new(2), 1)],
            false,
        )]),
    );

    let groups = hom.fusion_tree_groups(&rule).unwrap();

    assert_eq!(groups.len(), 2);
    assert_eq!(
        groups[0].group_key(),
        &FusionTreeGroupKey::from_sector_ids([2, 1], [1], [false, false], [false])
    );
    assert_eq!(
        groups[1].group_key(),
        &FusionTreeGroupKey::from_sector_ids([1, 1], [2], [false, false], [false])
    );
}

#[test]
fn checked_generic_external_axis_leg_keeps_provider_boundary() {
    let rule = UnitaryToyOmRule;
    let a = SectorId::new(UnitaryToyOmRule::A);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(a, 1)], false)]),
        FusionProductSpace::new([]),
    );
    let oriented = OrientedFusionTreeHomSpace::new(&homspace, FusionTreePairOrientation::Direct);
    let leg = oriented
        .try_external_axis_leg_generic(&rule, 0)
        .unwrap()
        .unwrap();
    assert_eq!(leg.sectors(), &[a]);
    assert_eq!(leg.degeneracies(), &[1]);
    assert!(!leg.is_dual());
    let dual = leg.try_dual_generic(&rule).unwrap();
    assert_eq!(dual.sectors(), &[a]);
    assert!(dual.is_dual());
}

/// `(uncoupled, inner lines, vertices)` ids of each tree of `coupled`.
fn tree_rows(groups: &[CoupledFusionTrees], coupled: usize) -> Vec<TreeRow> {
    let ids = |sectors: &[SectorId]| sectors.iter().map(|s| s.id()).collect::<Vec<_>>();
    groups
        .iter()
        .find(|group| group.coupled == SectorId::new(coupled))
        .expect("coupled sector present")
        .trees
        .iter()
        .map(|tree| {
            (
                ids(tree.uncoupled()),
                ids(tree.innerlines()),
                tree.vertices().iter().map(|v| v.get()).collect(),
            )
        })
        .collect()
}

type TreeRow = (Vec<usize>, Vec<usize>, Vec<usize>);

fn product_space(sectors: &[usize], is_dual: &[bool]) -> FusionProductSpace {
    FusionProductSpace::new(
        is_dual
            .iter()
            .map(|&dual| SectorLeg::new(sectors.iter().map(|&s| (SectorId::new(s), 1)), dual)),
    )
}

/// Every producer of a space's trees: infallible, checked and Generic.
fn all_producer_rows<R>(rule: &R, space: &FusionProductSpace, coupled: usize) -> Vec<TreeRow>
where
    R: MultiplicityFreeFusionRule + CheckedFusionAlgebra,
{
    let infallible = tree_rows(&fusion_trees_by_coupled_for_space(rule, space), coupled);
    let checked = tree_rows(
        &try_fusion_trees_by_coupled_for_space_checked(rule, space).unwrap(),
        coupled,
    );
    let (generic, _) = fusion_trees_by_coupled_for_space_generic(rule, space);
    assert_eq!(checked, infallible);
    assert_eq!(tree_rows(&generic, coupled), infallible);
    infallible
}

#[test]
fn fibonacci_three_leg_trees_follow_tensorkit_fusiontrees_order() {
    // What: within a coupled sector, trees of different uncoupled tuples
    // interleave by TensorKit's nesting (last leg, then the inner line, then
    // the front), not grouped by tuple. TensorKit 0.17.1:
    // V = Vect[FibonacciAnyon](:I=>1, :τ=>1); fusiontrees(V ⊗ V ⊗ V', c),
    // with ids I = 0, τ = 1 (both orders list I first).
    let rule = TableRule::fibonacci();
    let space = product_space(&[0, 1], &[false, false, true]);
    let expected_vacuum: Vec<TreeRow> = vec![
        ([0, 0, 0], [0], [1, 1]),
        ([1, 1, 0], [0], [1, 1]),
        ([1, 0, 1], [1], [1, 1]),
        ([0, 1, 1], [1], [1, 1]),
        ([1, 1, 1], [1], [1, 1]),
    ]
    .into_iter()
    .map(|(u, a, v)| (u.to_vec(), a.to_vec(), v.to_vec()))
    .collect();
    let expected_tau: Vec<TreeRow> = vec![
        ([1, 0, 0], [1], [1, 1]),
        ([0, 1, 0], [1], [1, 1]),
        ([1, 1, 0], [1], [1, 1]),
        ([0, 0, 1], [0], [1, 1]),
        ([1, 1, 1], [0], [1, 1]),
        ([1, 0, 1], [1], [1, 1]),
        ([0, 1, 1], [1], [1, 1]),
        ([1, 1, 1], [1], [1, 1]),
    ]
    .into_iter()
    .map(|(u, a, v)| (u.to_vec(), a.to_vec(), v.to_vec()))
    .collect();
    assert_eq!(all_producer_rows(&rule, &space, 0), expected_vacuum);
    assert_eq!(all_producer_rows(&rule, &space, 1), expected_tau);
}

#[test]
fn su2_three_leg_trees_follow_tensorkit_fusiontrees_order() {
    // TensorKit 0.17.1, ids = twice spin (both orders ascend in spin):
    // V = Vect[SU2Irrep](0=>1, 1//2=>1, 1=>1); fusiontrees(V ⊗ V' ⊗ V, c).
    let rule = SU2FusionRule;
    let space = product_space(&[0, 1, 2], &[false, true, false]);
    let expected_half: Vec<TreeRow> = vec![
        ([1, 0, 0], [1], [1, 1]),
        ([0, 1, 0], [1], [1, 1]),
        ([2, 1, 0], [1], [1, 1]),
        ([1, 2, 0], [1], [1, 1]),
        ([0, 0, 1], [0], [1, 1]),
        ([1, 1, 1], [0], [1, 1]),
        ([2, 2, 1], [0], [1, 1]),
        ([2, 0, 1], [2], [1, 1]),
        ([1, 1, 1], [2], [1, 1]),
        ([0, 2, 1], [2], [1, 1]),
        ([2, 2, 1], [2], [1, 1]),
        ([1, 0, 2], [1], [1, 1]),
        ([0, 1, 2], [1], [1, 1]),
        ([2, 1, 2], [1], [1, 1]),
        ([1, 2, 2], [1], [1, 1]),
        ([2, 1, 2], [3], [1, 1]),
        ([1, 2, 2], [3], [1, 1]),
    ]
    .into_iter()
    .map(|(u, a, v)| (u.to_vec(), a.to_vec(), v.to_vec()))
    .collect();
    let expected_one: Vec<TreeRow> = vec![
        ([2, 0, 0], [2], [1, 1]),
        ([1, 1, 0], [2], [1, 1]),
        ([0, 2, 0], [2], [1, 1]),
        ([2, 2, 0], [2], [1, 1]),
        ([1, 0, 1], [1], [1, 1]),
        ([0, 1, 1], [1], [1, 1]),
        ([2, 1, 1], [1], [1, 1]),
        ([1, 2, 1], [1], [1, 1]),
        ([2, 1, 1], [3], [1, 1]),
        ([1, 2, 1], [3], [1, 1]),
        ([0, 0, 2], [0], [1, 1]),
        ([1, 1, 2], [0], [1, 1]),
        ([2, 2, 2], [0], [1, 1]),
        ([2, 0, 2], [2], [1, 1]),
        ([1, 1, 2], [2], [1, 1]),
        ([0, 2, 2], [2], [1, 1]),
        ([2, 2, 2], [2], [1, 1]),
        ([2, 2, 2], [4], [1, 1]),
    ]
    .into_iter()
    .map(|(u, a, v)| (u.to_vec(), a.to_vec(), v.to_vec()))
    .collect();
    assert_eq!(all_producer_rows(&rule, &space, 1), expected_half);
    assert_eq!(all_producer_rows(&rule, &space, 2), expected_one);
}

#[test]
fn su2_four_leg_trees_follow_tensorkit_fusiontrees_order() {
    // What: two inner lines, each slower than the legs in front of it.
    // TensorKit 0.17.1, ids = twice spin:
    // W = Vect[SU2Irrep](1//2=>1, 1=>1); fusiontrees(W ⊗ W ⊗ W' ⊗ W, SU2Irrep(0)).
    let rule = SU2FusionRule;
    let space = product_space(&[1, 2], &[false, false, true, false]);
    let expected: Vec<TreeRow> = vec![
        ([1, 1, 1, 1], [0, 1], [1, 1, 1]),
        ([2, 2, 1, 1], [0, 1], [1, 1, 1]),
        ([1, 1, 1, 1], [2, 1], [1, 1, 1]),
        ([2, 2, 1, 1], [2, 1], [1, 1, 1]),
        ([2, 1, 2, 1], [1, 1], [1, 1, 1]),
        ([1, 2, 2, 1], [1, 1], [1, 1, 1]),
        ([2, 1, 2, 1], [3, 1], [1, 1, 1]),
        ([1, 2, 2, 1], [3, 1], [1, 1, 1]),
        ([2, 1, 1, 2], [1, 2], [1, 1, 1]),
        ([1, 2, 1, 2], [1, 2], [1, 1, 1]),
        ([2, 1, 1, 2], [3, 2], [1, 1, 1]),
        ([1, 2, 1, 2], [3, 2], [1, 1, 1]),
        ([1, 1, 2, 2], [0, 2], [1, 1, 1]),
        ([2, 2, 2, 2], [0, 2], [1, 1, 1]),
        ([1, 1, 2, 2], [2, 2], [1, 1, 1]),
        ([2, 2, 2, 2], [2, 2], [1, 1, 1]),
        ([2, 2, 2, 2], [4, 2], [1, 1, 1]),
    ]
    .into_iter()
    .map(|(u, a, v)| (u.to_vec(), a.to_vec(), v.to_vec()))
    .collect();
    assert_eq!(all_producer_rows(&rule, &space, 0), expected);
}
