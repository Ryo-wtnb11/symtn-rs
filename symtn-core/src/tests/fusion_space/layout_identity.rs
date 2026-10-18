use super::*;

#[test]
fn explicit_shape_coupled_grid_reports_extent_overflow() {
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(u1(0), usize::MAX)], false),
            SectorLeg::new([(u1(0), 2)], false),
        ]),
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
    );

    let error = homspace
        .coupled_subblock_structure(&U1FusionRule, 2, [[usize::MAX, 2]])
        .unwrap_err();
    assert_eq!(error, CoreError::ElementCountOverflow);
}

fn assert_direct_generic_leg_degeneracy_structure_matches_reference<R>(
    rule: &R,
    homspace: &FusionTreeHomSpace,
) where
    R: FusionRule,
{
    let keys = homspace.fusion_tree_keys_generic(rule).unwrap();
    let blocks = keys
        .iter()
        .map(|key| {
            (
                key.clone(),
                homspace.degeneracy_shape_for_key(key).unwrap().to_vec(),
            )
        })
        .collect();
    let expected = BlockStructure::coupled_sector_matrix_with_keys(
        rule,
        homspace.codomain().len(),
        homspace.rank(),
        blocks,
    )
    .unwrap()
    .into_shared();
    let actual = homspace
        .coupled_subblock_structure_from_leg_degeneracies_generic(rule)
        .unwrap();
    assert_eq!(actual, expected);
    assert_ne!(actual.content_id(), expected.content_id());
}

#[test]
fn direct_generic_leg_degeneracy_layout_matches_reference() {
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = UnitaryToyOmRule;
    let a = SectorId::new(UnitaryToyOmRule::A);
    let c = SectorId::new(UnitaryToyOmRule::C);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(a, 2)], false),
            SectorLeg::new([(a, 2)], false),
        ]),
        FusionProductSpace::new([SectorLeg::new([(c, 3)], false)]),
    );
    assert_direct_generic_leg_degeneracy_structure_matches_reference(&rule, &homspace);
}

#[test]
fn canonical_coupled_grid_derives_each_row_and_column_once() {
    let rule = U1FusionRule;
    let leg = SectorLeg::new([(u1(-1), 2), (u1(0), 3), (u1(2), 1)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone(), leg.clone()]),
        FusionProductSpace::new([leg.clone(), leg]),
    );
    let recording = crate::probe::record();
    let layout = homspace.fusion_tree_layout_data_uncached(&rule);
    let expected_derivations = layout
        .sectors
        .iter()
        .map(|sector| sector.row_count + sector.col_count)
        .sum::<usize>();
    let actual = coupled_subblock_parts_from_leg_degeneracies(&homspace, &layout).unwrap();
    let expected = reference_leg_degeneracy_structure(&rule, &homspace);
    assert_eq!(*actual.0, *expected.sector_structure());
    assert_eq!(actual.1, *expected.degeneracy_structure());
    assert_eq!(
        (
            recording.get(crate::probe::Event::CoupledGridReconstructionInsert),
            recording.get(crate::probe::Event::CoupledGridSideDerivation)
        ),
        (0, expected_derivations)
    );
}

#[test]
fn fusion_tree_homspace_compose_rejects_unmatched_contracted_sector() {
    // Pairing a domain leg with the *dual* codomain leg is a
    // SpaceMismatch in TensorKit (`(X ← V) * (V' ← Y)` fails).
    let rule = U1FusionRule;
    let lhs = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(u1(0), 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(u1(1), 1)], false)]),
    );
    let rhs = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(rule.dual(u1(1)), 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(u1(0), 1)], false)]),
    );

    let err = FusionTreeHomSpace::compose(&rule, &lhs, &rhs).unwrap_err();

    assert_eq!(
        err,
        CoreError::SectorMismatch {
            expected: u1(1),
            actual: rule.dual(u1(1)),
        }
    );
}

#[test]
fn unique_homspace_rejects_invalid_external_sector_tuple() {
    let rule = Z4PointedRule;
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(SectorId::new(1), 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(SectorId::new(1), 1)], false)]),
    );

    let err = hom
        .unique_fusion_tree_key_from_external_sectors(&rule, &[SectorId::new(1), SectorId::new(2)])
        .unwrap_err();

    assert_eq!(
        err,
        CoreError::InvalidSector {
            sector: SectorId::new(2),
        }
    );
}

#[test]
fn fusion_tree_homspace_generates_innerline_paths_for_simple_fusion() {
    let rule = BranchingMultiplicityFreeRule;
    let hom = FusionTreeHomSpace::from_sector_ids([(1, 1), (1, 1), (1, 1)], [(1, 1)]);

    let keys = hom.fusion_tree_keys(&rule);

    assert_eq!(keys.len(), 2);
    assert_eq!(keys[0].coupled(), SectorId::new(1));
    assert_eq!(keys[1].coupled(), SectorId::new(1));
    assert_eq!(keys[0].codomain_innerlines(), &[SectorId::new(0)]);
    assert_eq!(keys[1].codomain_innerlines(), &[SectorId::new(2)]);
    assert_eq!(
        keys[0].codomain_vertices(),
        &[MultiplicityIndex::ONE, MultiplicityIndex::ONE]
    );
    assert!(keys[0].domain_innerlines().is_empty());
    assert!(keys[0].domain_vertices().is_empty());
    assert_eq!(keys[0].domain_uncoupled(), &[SectorId::new(1)]);

    let groups = hom.fusion_tree_groups(&rule).unwrap();
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].block_indices(), &[0, 1]);
}

#[test]
fn fusion_tree_homspace_matches_tensorkit_z2_fusiontreelist_order() {
    let rule = Z2FusionRule;
    let leg = || SectorLeg::new([(SectorId::new(0), 1), (SectorId::new(1), 1)], false);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );

    let keys = hom.fusion_tree_keys(&rule);

    // TensorKit.jl 6Camk:
    // V=Vect[Z2Irrep](0=>1,1=>1); W=(V⊗V)←(V⊗V);
    // [(f1.uncoupled, f2.uncoupled, f1.coupled) for (f1,f2) in fusiontrees(W)]
    assert_eq!(
        fusion_tree_pair_order(&keys),
        vec![
            (vec![0, 0], vec![0, 0], 0),
            (vec![1, 1], vec![0, 0], 0),
            (vec![0, 0], vec![1, 1], 0),
            (vec![1, 1], vec![1, 1], 0),
            (vec![1, 0], vec![1, 0], 1),
            (vec![0, 1], vec![1, 0], 1),
            (vec![1, 0], vec![0, 1], 1),
            (vec![0, 1], vec![0, 1], 1),
        ]
    );

    let groups = hom.fusion_tree_groups(&rule).unwrap();
    assert_eq!(groups.len(), keys.len());
    for (index, group) in groups.iter().enumerate() {
        assert_eq!(group.block_indices(), &[index]);
    }
}

#[test]
fn fusion_tree_key_cache_hits_across_degeneracy_and_keeps_dual_signature() {
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = SU2FusionRule;
    let mk_leg = |degeneracy| {
        SectorLeg::new(
            [
                (SU2Irrep::from_twice_spin(0).sector_id(), degeneracy),
                (SU2Irrep::from_twice_spin(1).sector_id(), degeneracy + 1),
            ],
            false,
        )
    };
    let hom_small = FusionTreeHomSpace::new(
        FusionProductSpace::new([mk_leg(1), mk_leg(1)]),
        FusionProductSpace::new([mk_leg(1)]),
    );
    let hom_large = FusionTreeHomSpace::new(
        FusionProductSpace::new([mk_leg(4), mk_leg(4)]),
        FusionProductSpace::new([mk_leg(4)]),
    );

    let small_layout = hom_small.cached_fusion_tree_layout(&rule);
    let large_layout = hom_large.cached_fusion_tree_layout(&rule);
    assert!(Arc::ptr_eq(&small_layout, &large_layout));
    assert_eq!(small_layout.keys.as_ref(), large_layout.keys.as_ref());

    let dual_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([mk_leg(1).dual(&rule), mk_leg(1)]),
        FusionProductSpace::new([mk_leg(1)]),
    );
    let dual_layout = dual_hom.cached_fusion_tree_layout(&rule);
    assert!(!Arc::ptr_eq(&small_layout, &dual_layout));
    assert_ne!(small_layout.keys.as_ref(), dual_layout.keys.as_ref());
}

#[test]
fn fusion_layout_identity_hashes_inner_semantics_not_arc_address() {
    // What: independently allocated identity Arcs compare and hash by their
    // complete rule/sector/duality value, while distinct rules and splits do not alias.
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(z2_even(), 1), (z2_odd(), 1)], false),
            SectorLeg::new([(z2_even(), 1)], true),
        ]),
        FusionProductSpace::new([SectorLeg::new([(z2_odd(), 1)], false)]),
    );
    let first = Arc::new(FusionTreeHomSpaceCacheKey::new(&Z2FusionRule, &hom));
    let second = Arc::new(FusionTreeHomSpaceCacheKey::new(&Z2FusionRule, &hom));
    assert!(!Arc::ptr_eq(&first, &second));
    assert_eq!(first, second);

    let mut first_hash = rustc_hash::FxHasher::default();
    first.hash(&mut first_hash);
    let mut second_hash = rustc_hash::FxHasher::default();
    second.hash(&mut second_hash);
    assert_eq!(first_hash.finish(), second_hash.finish());

    let fermionic = Arc::new(FusionTreeHomSpaceCacheKey::new(
        &FermionParityFusionRule,
        &hom,
    ));
    assert_ne!(first, fermionic);

    let repartitioned = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(z2_even(), 1), (z2_odd(), 1)], false)]),
        FusionProductSpace::new([
            SectorLeg::new([(z2_even(), 1)], true),
            SectorLeg::new([(z2_odd(), 1)], false),
        ]),
    );
    let repartitioned = Arc::new(FusionTreeHomSpaceCacheKey::new(
        &Z2FusionRule,
        &repartitioned,
    ));
    assert_ne!(first, repartitioned);
}

#[test]
fn fusion_layout_reset_preserves_coupled_structure() {
    // What: a reset rebuilds the layout; coupled content remains equal and a
    // live old structure is never aliased by the rebuilt one. (Budget-bound
    // eviction is covered by the cache's own tests.)
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    clear_structure_caches();
    let rule = U1FusionRule;
    let hom = FusionTreeHomSpace::from_sectors([(U1Irrep::new(0), 2)], [(U1Irrep::new(0), 3)]);
    let old_layout = hom.cached_fusion_tree_layout(&rule);
    let old_structure = hom
        .coupled_subblock_structure(&rule, 1, [vec![2, 3]])
        .unwrap();

    let rebuilt_layout = hom.cached_fusion_tree_layout(&rule);
    assert!(Arc::ptr_eq(&rebuilt_layout, &old_layout));
    let reused_structure = hom
        .coupled_subblock_structure(&rule, 1, [vec![2, 3]])
        .unwrap();
    assert_eq!(old_structure.as_ref(), reused_structure.as_ref());

    clear_structure_caches();
    let after_reset_layout = hom.cached_fusion_tree_layout(&rule);
    let after_reset_structure = hom
        .coupled_subblock_structure(&rule, 1, [vec![2, 3]])
        .unwrap();
    assert!(!Arc::ptr_eq(&rebuilt_layout, &after_reset_layout));
    assert!(!Arc::ptr_eq(&old_structure, &after_reset_structure));
    assert_eq!(old_structure.as_ref(), after_reset_structure.as_ref());
}
