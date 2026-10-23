use super::*;

fn assert_checked_keys_match_encoded_oracle<R>(rule: &R, hom: &FusionTreeHomSpace)
where
    R: MultiplicityFreeFusionRule + CheckedFusionAlgebra,
{
    let encoded_layout = hom.fusion_tree_layout_data_uncached(rule);
    let checked_layout = hom
        .try_fusion_tree_layout_data_uncached_checked(rule)
        .unwrap();
    assert_eq!(checked_layout.keys, encoded_layout.keys);
    assert_eq!(
        checked_layout.keys.as_ref(),
        hom.fusion_tree_keys_uncached(rule).as_slice()
    );
    assert_eq!(checked_layout.sectors.len(), encoded_layout.sectors.len());
    for (actual, expected) in checked_layout.sectors.iter().zip(&encoded_layout.sectors) {
        assert_eq!(actual.start, expected.start);
        assert_eq!(actual.row_count, expected.row_count);
        assert_eq!(actual.col_count, expected.col_count);
    }
}

#[test]
fn checked_builder_matches_encoded_oracle_for_builtin_ranks_and_products() {
    // What: every persistent key field and key order stays identical for
    // ranks 0 through 6 across all built-in multiplicity-free algebras.
    for rank in 0..=6 {
        assert_checked_keys_match_encoded_oracle(&U1FusionRule, &singleton_rank_hom(u1(1), rank));
        assert_checked_keys_match_encoded_oracle(
            &Z2FusionRule,
            &singleton_rank_hom(z2_odd(), rank),
        );
        assert_checked_keys_match_encoded_oracle(
            &FermionParityFusionRule,
            &singleton_rank_hom(z2_odd(), rank),
        );
        assert_checked_keys_match_encoded_oracle(&SU2FusionRule, &singleton_rank_hom(su2(1), rank));
    }
    // Dual legs change the stored key duality flags on a non-product rule too.
    let su2_dual_leg = FusionProductSpace::new([
        SectorLeg::new([(su2(1), 2)], true),
        SectorLeg::new([(su2(1), 2)], false),
    ]);
    assert_checked_keys_match_encoded_oracle(
        &SU2FusionRule,
        &FusionTreeHomSpace::new(su2_dual_leg.clone(), su2_dual_leg),
    );

    type U1Fz2Codec = PackedProductCodec<U1SectorLayout, Fz2SectorLayout>;
    type U1Fz2Rule = ProductFusionRule<U1FusionRule, FermionParityFusionRule, U1Fz2Codec>;
    let pair_rule = U1Fz2Rule::try_new(U1FusionRule, FermionParityFusionRule).unwrap();
    let pair_sector = U1Fz2Codec::encode(u1(1), z2_odd());

    type Fz2U1Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
    type Fz2U1Layout = ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>;
    type TripleCodec = PackedProductCodec<Fz2U1Layout, Su2SectorLayout>;
    type Fz2U1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule, Fz2U1Codec>;
    type TripleRule = ProductFusionRule<Fz2U1Rule, SU2FusionRule, TripleCodec>;
    let triple_rule = TripleRule::try_new(
        Fz2U1Rule::try_new(FermionParityFusionRule, U1FusionRule).unwrap(),
        SU2FusionRule,
    )
    .unwrap();
    let triple_sector = TripleCodec::encode(Fz2U1Codec::encode(z2_odd(), u1(1)), su2(1));
    let triple_pair_coupled = TripleCodec::encode(Fz2U1Codec::encode(z2_even(), u1(2)), su2(0));

    for rank in 0..=6 {
        assert_checked_keys_match_encoded_oracle(
            &pair_rule,
            &singleton_rank_hom(pair_sector, rank),
        );
        assert_checked_keys_match_encoded_oracle(
            &triple_rule,
            &singleton_rank_hom(triple_sector, rank),
        );
    }

    let triple_vacuum = TripleCodec::encode(Fz2U1Codec::encode(z2_even(), u1(0)), su2(0));
    let multi_tuple_side = |invert_dual| {
        FusionProductSpace::new((0..4).map(|axis| {
            SectorLeg::new(
                [(triple_vacuum, 1), (triple_sector, 2)],
                (axis % 2 == 0) ^ invert_dual,
            )
        }))
    };
    let multi_tuple_rank_eight =
        FusionTreeHomSpace::new(multi_tuple_side(false), multi_tuple_side(true));
    assert_checked_keys_match_encoded_oracle(&triple_rule, &multi_tuple_rank_eight);

    for dual_mask in 0usize..(1 << 3) {
        let all_dual_masks = FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(triple_sector, 2)], dual_mask & 1 != 0),
                SectorLeg::new([(triple_sector, 3)], dual_mask & 2 != 0),
            ]),
            FusionProductSpace::new([SectorLeg::new(
                [(triple_pair_coupled, 4)],
                dual_mask & 4 != 0,
            )]),
        );
        assert_checked_keys_match_encoded_oracle(&triple_rule, &all_dual_masks);
    }

    let asymmetric = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(triple_sector, 2)], true),
            SectorLeg::new([(triple_sector, 3)], false),
        ]),
        FusionProductSpace::new([
            SectorLeg::new(
                [(
                    TripleCodec::encode(Fz2U1Codec::encode(z2_even(), u1(2)), su2(0)),
                    4,
                )],
                true,
            ),
            SectorLeg::new(
                [(
                    TripleCodec::encode(Fz2U1Codec::encode(z2_even(), u1(0)), su2(0)),
                    5,
                )],
                false,
            ),
        ]),
    );
    assert_checked_keys_match_encoded_oracle(&triple_rule, &asymmetric);
}

#[test]
fn checked_and_encoded_entries_share_the_same_layout_cache() {
    // What: encoded-first and lowered-first construction converge on the same
    // Arc rather than publishing parallel layouts for one semantic key.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let hom = singleton_rank_hom(su2(1), 4);

    clear_structure_caches();
    let encoded_first = hom.cached_fusion_tree_layout(&SU2FusionRule);
    let checked_second = hom
        .prepare_fusion_tree_layout_checked(&SU2FusionRule)
        .unwrap()
        .commit_layout();
    assert!(Arc::ptr_eq(&encoded_first, &checked_second));

    clear_structure_caches();
    let checked_first = hom
        .prepare_fusion_tree_layout_checked(&SU2FusionRule)
        .unwrap()
        .commit_layout();
    let encoded_second = hom.cached_fusion_tree_layout(&SU2FusionRule);
    assert!(Arc::ptr_eq(&checked_first, &encoded_second));
}

#[test]
fn warm_layout_commit_publishes_nothing() {
    // What: publishing a layout admits it once; a warm call finds it in the
    // prepare lookup and publishes nothing.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    clear_structure_caches();
    let hom = singleton_rank_hom(su2(1), 5);

    let recording = crate::probe::record();
    let cold = hom.cached_fusion_tree_layout(&SU2FusionRule);
    assert_eq!(
        (
            recording.get(crate::probe::Event::LayoutBuild),
            recording.get(crate::probe::Event::LayoutAdmission)
        ),
        (1, 1)
    );

    let warm = hom.cached_fusion_tree_layout(&SU2FusionRule);
    assert_eq!(
        (
            recording.get(crate::probe::Event::LayoutBuild),
            recording.get(crate::probe::Event::LayoutAdmission)
        ),
        (1, 1)
    );
    assert!(Arc::ptr_eq(&cold, &warm));
}

#[test]
fn prepared_layout_publishes_only_at_commit() {
    // What: cold preparation enumerates exactly once but does not consume
    // identity or cache admission until its explicit commit point.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    clear_structure_caches();
    let recording = crate::probe::record();
    let hom = singleton_rank_hom(su2(1), 5);

    let prepared = hom
        .prepare_fusion_tree_layout_checked(&SU2FusionRule)
        .unwrap();
    // Why not inspect global cache totals: unrelated parallel tests may
    // populate the same process cache. These thread-local probes attribute
    // publication exactly to this transaction.
    assert_eq!(
        (
            recording.get(crate::probe::Event::LayoutBuild),
            recording.get(crate::probe::Event::LayoutAdmission)
        ),
        (0, 0)
    );

    let keys = prepared.commit();
    assert!(!keys.is_empty());
    assert_eq!(
        (
            recording.get(crate::probe::Event::LayoutBuild),
            recording.get(crate::probe::Event::LayoutAdmission)
        ),
        (1, 1)
    );
}

#[test]
fn prepared_final_structure_reuses_one_checked_enumeration() {
    // What: cold checked preparation and the direct leg-degeneracy builder
    // match the established single-pass structure without a second
    // decode/channel enumeration or early cache publication.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let hom = singleton_rank_hom(su2(1), 5);
    clear_structure_caches();
    let expected = hom
        .coupled_subblock_structure_from_leg_degeneracies(&SU2FusionRule)
        .unwrap();

    clear_structure_caches();
    let recording = crate::probe::record();
    let prepared = hom
        .prepare_fusion_tree_layout_checked(&SU2FusionRule)
        .unwrap();
    assert_eq!(
        (
            recording.get(crate::probe::Event::LayoutBuild),
            recording.get(crate::probe::Event::LayoutAdmission)
        ),
        (0, 0)
    );

    let actual = prepared.build_from_leg_degeneracies(&hom).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(
        (
            recording.get(crate::probe::Event::LayoutBuild),
            recording.get(crate::probe::Event::LayoutAdmission)
        ),
        (0, 0)
    );

    prepared.commit();
    assert_eq!(
        (
            recording.get(crate::probe::Event::LayoutBuild),
            recording.get(crate::probe::Event::LayoutAdmission)
        ),
        (1, 1)
    );
}

#[test]
fn prepared_complete_structure_hits_without_rebuilding_layout() {
    // Isolated like symtn-tensors #649/#650's checked_bind_failure
    // test: this asserts an absolute process-global cache admission
    // count right after a reset, which CACHE_TEST_LOCK does not
    // protect from the crate's many ordinary, unlocked complete-
    // structure builds landing in the same narrow window.
    if test_support::run_isolated_or_return(
        "SYMTN_CORE_PREPARED_COMPLETE_STRUCTURE_HITS_ISOLATED",
        "tests::fusion_space::checked_layout_builder::prepared_complete_structure_hits_without_rebuilding_layout",
    ) {
        return;
    }
    // What: a repeated valid finalization validates its target
    // locally, then reuses the retained complete content without another
    // tree enumeration or cache admission.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    clear_structure_caches();
    let hom = singleton_rank_hom(su2(1), 5);

    let first_prepared = hom
        .prepare_fusion_tree_layout_checked(&SU2FusionRule)
        .unwrap();
    let first = first_prepared
        .build_complete_from_leg_degeneracies(&hom)
        .unwrap();
    first_prepared.commit();
    let after_first = structure_cache_info(StructureCacheKind::DegeneracyStructure);
    assert_eq!(after_first.admissions(), 1);

    let second_prepared = hom
        .prepare_fusion_tree_layout_checked(&SU2FusionRule)
        .unwrap();
    let second = second_prepared
        .build_complete_from_leg_degeneracies(&hom)
        .unwrap();
    second_prepared.commit();
    let after_second = structure_cache_info(StructureCacheKind::DegeneracyStructure);

    assert_eq!(first.content_id(), second.content_id());
    assert_eq!(after_second.hits(), after_first.hits() + 1);
    assert_eq!(after_second.admissions(), after_first.admissions());
}

fn finalize_complete<R>(
    rule: &R,
    hom: &FusionTreeHomSpace,
) -> Result<Arc<BlockStructure>, CoreError>
where
    R: MultiplicityFreeFusionRule + CheckedFusionAlgebra,
{
    let prepared = hom.prepare_fusion_tree_layout_checked(rule).unwrap();
    let structure = prepared.build_complete_from_leg_degeneracies(hom)?;
    prepared.commit();
    Ok(structure)
}

/// Side derivations of one full extent walk over `hom`.
fn one_walk_side_derivations<R>(rule: &R, hom: &FusionTreeHomSpace) -> usize
where
    R: MultiplicityFreeFusionRule,
{
    hom.fusion_tree_layout_data_uncached(rule)
        .sectors
        .iter()
        .map(|sector| sector.row_count + sector.col_count)
        .sum()
}

#[test]
fn complete_structure_hit_skips_extent_walk_until_evicted() {
    // Isolated like prepared_complete_structure_hits_without_rebuilding_layout:
    // this asserts process-global complete-structure cache counters, which
    // CACHE_TEST_LOCK does not protect from the crate's many ordinary,
    // unlocked complete-structure builds landing between two reads (#1903).
    if test_support::run_isolated_or_return(
        "SYMTN_CORE_COMPLETE_STRUCTURE_HIT_SKIPS_WALK_ISOLATED",
        "tests::fusion_space::checked_layout_builder::complete_structure_hit_skips_extent_walk_until_evicted",
    ) {
        return;
    }
    // What: the miss walks the per-block extents exactly once (inside the
    // builder); a hit walks none; after an eviction the next call is a miss
    // that walks exactly once again.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    clear_structure_caches();
    let leg = SectorLeg::new([(u1(-1), 2), (u1(0), 3), (u1(2), 1)], false);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone(), leg.clone()]),
        FusionProductSpace::new([leg.clone(), leg]),
    );
    let walk = one_walk_side_derivations(&U1FusionRule, &hom);
    assert!(walk > 2);

    let recording = crate::probe::record();
    let first = finalize_complete(&U1FusionRule, &hom).unwrap();
    assert_eq!(
        recording.get(crate::probe::Event::CoupledGridSideDerivation),
        walk
    );
    let after_miss = structure_cache_info(StructureCacheKind::DegeneracyStructure);
    assert_eq!((after_miss.misses(), after_miss.admissions()), (1, 1));

    let recording = crate::probe::record();
    let second = finalize_complete(&U1FusionRule, &hom).unwrap();
    assert_eq!(
        recording.get(crate::probe::Event::CoupledGridSideDerivation),
        0
    );
    assert!(Arc::ptr_eq(&first, &second));
    let after_hit = structure_cache_info(StructureCacheKind::DegeneracyStructure);
    assert_eq!(after_hit.hits(), after_miss.hits() + 1);
    assert_eq!(after_hit.misses(), after_miss.misses());

    // An eviction (a zero budget evicts everything) makes the next call a
    // miss that walks once again and is admitted again.
    drop((first, second));
    let budget = after_hit.byte_budget();
    crate::set_structure_cache_byte_budget(StructureCacheKind::DegeneracyStructure, 0);
    crate::set_structure_cache_byte_budget(StructureCacheKind::DegeneracyStructure, budget);
    let evicted = structure_cache_info(StructureCacheKind::DegeneracyStructure);
    assert_eq!(evicted.entries(), 0);
    let recording = crate::probe::record();
    finalize_complete(&U1FusionRule, &hom).unwrap();
    assert_eq!(
        recording.get(crate::probe::Event::CoupledGridSideDerivation),
        walk
    );
    let rewalked = structure_cache_info(StructureCacheKind::DegeneracyStructure);
    assert_eq!(rewalked.misses(), evicted.misses() + 1);
    assert_eq!(rewalked.admissions(), evicted.admissions() + 1);
    assert_eq!(rewalked.hits(), evicted.hits());
}

#[test]
fn complete_structure_split_and_fermionic_rule_force_misses() {
    // Isolated like prepared_complete_structure_hits_without_rebuilding_layout:
    // this asserts process-global complete-structure cache counters, which
    // CACHE_TEST_LOCK does not protect from the crate's many ordinary,
    // unlocked complete-structure builds landing between two reads (#1903).
    if test_support::run_isolated_or_return(
        "SYMTN_CORE_COMPLETE_STRUCTURE_SPLIT_FERMIONIC_ISOLATED",
        "tests::fusion_space::checked_layout_builder::complete_structure_split_and_fermionic_rule_force_misses",
    ) {
        return;
    }
    // What: equal legs under another codomain/domain split, and equal
    // sectors under Z2 versus fermion parity, are misses that admit their
    // own entries rather than hits on a same-content neighbour.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    clear_structure_caches();
    let leg = SectorLeg::new([(u1(0), 2), (u1(1), 3)], false);
    let split = |nout: usize| {
        let legs = [leg.clone(), leg.clone(), leg.clone()];
        FusionTreeHomSpace::new(
            FusionProductSpace::new(legs[..nout].iter().cloned()),
            FusionProductSpace::new(legs[nout..].iter().cloned()),
        )
    };
    let two_one = finalize_complete(&U1FusionRule, &split(2)).unwrap();
    let before = structure_cache_info(StructureCacheKind::DegeneracyStructure);
    let one_two = finalize_complete(&U1FusionRule, &split(1)).unwrap();
    let after = structure_cache_info(StructureCacheKind::DegeneracyStructure);
    assert_ne!(two_one.content_id(), one_two.content_id());
    assert_eq!(after.misses(), before.misses() + 1);
    assert_eq!(after.admissions(), before.admissions() + 1);
    assert_eq!(after.hits(), before.hits());

    let parity_leg = SectorLeg::new([(z2_even(), 2), (z2_odd(), 3)], false);
    let parity_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([parity_leg.clone(), parity_leg.clone()]),
        FusionProductSpace::new([parity_leg]),
    );
    let bosonic = finalize_complete(&Z2FusionRule, &parity_hom).unwrap();
    let before = structure_cache_info(StructureCacheKind::DegeneracyStructure);
    let recording = crate::probe::record();
    let fermionic = finalize_complete(&FermionParityFusionRule, &parity_hom).unwrap();
    let after = structure_cache_info(StructureCacheKind::DegeneracyStructure);
    // The geometry is equal, but distinct rule keys own distinct complete
    // entries and therefore distinct monotonic operation ids.
    assert_eq!(bosonic, fermionic);
    assert_ne!(bosonic.content_id(), fermionic.content_id());
    assert!(recording.get(crate::probe::Event::CoupledGridSideDerivation) > 0);
    assert_eq!(after.misses(), before.misses() + 1);
    assert_eq!(after.admissions(), before.admissions() + 1);
    assert_eq!(after.entries(), before.entries() + 1);
    assert_eq!(after.hits(), before.hits());
}

#[test]
fn complete_structure_overflow_is_rejected_beside_cached_neighbour() {
    // Isolated like prepared_complete_structure_hits_without_rebuilding_layout:
    // this asserts process-global complete-structure cache counters, which
    // CACHE_TEST_LOCK does not protect from the crate's many ordinary,
    // unlocked complete-structure builds landing between two reads (#1903).
    if test_support::run_isolated_or_return(
        "SYMTN_CORE_COMPLETE_STRUCTURE_OVERFLOW_ISOLATED",
        "tests::fusion_space::checked_layout_builder::complete_structure_overflow_is_rejected_beside_cached_neighbour",
    ) {
        return;
    }
    // What: an extent overflow whose sectors and duals equal a cached
    // valid neighbour is still walked and rejected without touching the
    // statistics, and the neighbour keeps hitting without a walk.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    clear_structure_caches();
    let hom = |first_degeneracy: usize| {
        FusionTreeHomSpace::new(
            FusionProductSpace::new([
                SectorLeg::new([(u1(0), first_degeneracy)], false),
                SectorLeg::new([(u1(0), 2)], false),
            ]),
            FusionProductSpace::new([SectorLeg::new([(u1(0), 2)], false)]),
        )
    };
    let neighbour = hom(3);
    let cached = finalize_complete(&U1FusionRule, &neighbour).unwrap();
    let overflow = hom(usize::MAX);
    let before = structure_cache_info(StructureCacheKind::DegeneracyStructure);

    let error = finalize_complete(&U1FusionRule, &overflow).unwrap_err();
    assert_eq!(error, CoreError::ElementCountOverflow);
    assert_eq!(
        structure_cache_info(StructureCacheKind::DegeneracyStructure),
        before
    );

    let recording = crate::probe::record();
    let hit = finalize_complete(&U1FusionRule, &neighbour).unwrap();
    assert!(Arc::ptr_eq(&cached, &hit));
    assert_eq!(
        recording.get(crate::probe::Event::CoupledGridSideDerivation),
        0
    );
    assert_eq!(
        structure_cache_info(StructureCacheKind::DegeneracyStructure).hits(),
        before.hits() + 1
    );
}

#[test]
fn prepared_lowered_final_structure_checks_signature_but_reads_target_degeneracies() {
    // What: a prepared layout rejects another same-rank sector signature
    // without publication, while the same sectors/duality with different
    // degeneracies are accepted as the target structure authority.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    clear_structure_caches();
    let source = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(u1(1), 2)], false)]),
        FusionProductSpace::new([SectorLeg::new([(u1(1), 3)], true)]),
    );
    let prepared = source
        .prepare_fusion_tree_layout_checked(&U1FusionRule)
        .unwrap();
    let recording = crate::probe::record();
    let mismatched = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(u1(2), 2)], false)]),
        FusionProductSpace::new([SectorLeg::new([(u1(2), 3)], true)]),
    );

    let error = prepared
        .build_from_leg_degeneracies(&mismatched)
        .unwrap_err();
    assert_eq!(
        error,
        CoreError::MalformedFusionTree {
            message: "prepared layout does not match HomSpace sector signature",
        }
    );
    assert_eq!(
        (
            recording.get(crate::probe::Event::LayoutBuild),
            recording.get(crate::probe::Event::LayoutAdmission)
        ),
        (0, 0)
    );
    let duality_mismatched = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(u1(1), 2)], true)]),
        FusionProductSpace::new([SectorLeg::new([(u1(1), 3)], true)]),
    );
    assert_eq!(
        prepared
            .build_from_leg_degeneracies(&duality_mismatched)
            .unwrap_err(),
        CoreError::MalformedFusionTree {
            message: "prepared layout does not match HomSpace sector signature",
        }
    );
    assert_eq!(
        (
            recording.get(crate::probe::Event::LayoutBuild),
            recording.get(crate::probe::Event::LayoutAdmission)
        ),
        (0, 0)
    );

    let target = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(u1(1), 5)], false)]),
        FusionProductSpace::new([SectorLeg::new([(u1(1), 7)], true)]),
    );
    let structure = prepared.build_from_leg_degeneracies(&target).unwrap();
    assert_eq!(
        structure.degeneracy_structure().block(0).unwrap().shape(),
        &[5, 7]
    );
    assert_eq!(
        (
            recording.get(crate::probe::Event::LayoutBuild),
            recording.get(crate::probe::Event::LayoutAdmission)
        ),
        (0, 0)
    );
}

#[test]
fn cached_commit_readmits_after_core_reset() {
    // What: a cached preparation that survives reset republishes its exact
    // retained keys without consuming a fresh layout identity.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    clear_structure_caches();
    let hom = singleton_rank_hom(su2(1), 5);
    hom.prepare_fusion_tree_layout_checked(&SU2FusionRule)
        .unwrap()
        .commit();
    let prepared = hom
        .prepare_fusion_tree_layout_checked(&SU2FusionRule)
        .unwrap();
    let retained = prepared.keys_arc();
    clear_structure_caches();
    let recording = crate::probe::record();

    let committed = prepared.commit();

    assert!(Arc::ptr_eq(&retained, &committed));
    assert_eq!(
        (
            recording.get(crate::probe::Event::LayoutBuild),
            recording.get(crate::probe::Event::LayoutAdmission)
        ),
        (0, 1)
    );
}

#[test]
fn concurrent_commits_share_one_layout_admission() {
    // What: two cold preparations racing to commit converge on one Arc
    // and one cache miss without the losing transaction issuing an ID.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    clear_structure_caches();
    let hom = singleton_rank_hom(su2(1), 5);
    let barrier = Arc::new(std::sync::Barrier::new(2));
    let handles = (0..2)
        .map(|_| {
            let hom = hom.clone();
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                let recording = crate::probe::record();
                let prepared = hom
                    .prepare_fusion_tree_layout_checked(&SU2FusionRule)
                    .unwrap();
                barrier.wait();
                let keys = prepared.commit();
                (
                    keys,
                    (
                        recording.get(crate::probe::Event::LayoutBuild),
                        recording.get(crate::probe::Event::LayoutAdmission),
                    ),
                )
            })
        })
        .collect::<Vec<_>>();
    let mut results = handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .collect::<Vec<_>>();
    let (second, second_calls) = results.pop().unwrap();
    let (first, first_calls) = results.pop().unwrap();

    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(first_calls.0 + second_calls.0, 1);
    assert_eq!(first_calls.1 + second_calls.1, 1);
}

#[test]
fn checked_builder_reports_malformed_ids_and_algebra_closure_without_panicking() {
    // What: packed decode stays a codec error, while U(1), SU(2), and
    // recursive product closure failures retain exact causes.
    type Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
    type Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule, Codec>;
    let rule = Rule::try_new(FermionParityFusionRule, U1FusionRule).unwrap();
    let malformed = singleton_rank_hom(SectorId::new(usize::MAX), 1);
    let error = malformed
        .prepare_fusion_tree_layout_checked(&rule)
        .unwrap_err();
    assert_eq!(
        error,
        FusionAlgebraError::ProductCodec(
            Codec::decode_checked(SectorId::new(usize::MAX)).unwrap_err(),
        )
    );

    let invalid_z2 = singleton_rank_hom(SectorId::new(2), 1)
        .prepare_fusion_tree_layout_checked(&Z2FusionRule)
        .unwrap_err();
    assert_eq!(
        invalid_z2,
        FusionAlgebraError::InvalidSector {
            sector: SectorId::new(2),
        }
    );

    let u1_overflow = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(u1(i32::MAX), 1)], false),
            SectorLeg::new([(u1(1), 1)], false),
        ]),
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
    );
    let error = u1_overflow
        .try_fusion_tree_layout_data_uncached_checked(&U1FusionRule)
        .unwrap_err();
    assert_eq!(
        error,
        FusionAlgebraError::U1FusionOverflow {
            left: i32::MAX,
            right: 1,
        }
    );

    let invalid_u1_label = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(u1(0), 1)], false),
            SectorLeg::new([(u1(0), 1)], false),
            SectorLeg::new([(excluded_u1_id(), 1)], false),
        ]),
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
    );
    let error = invalid_u1_label
        .try_fusion_tree_layout_data_uncached_checked(&U1FusionRule)
        .unwrap_err();
    assert_eq!(
        error,
        FusionAlgebraError::InvalidSector {
            sector: excluded_u1_id()
        }
    );

    let su2_overflow = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(su2(128), 1)], false),
            SectorLeg::new([(su2(127), 1)], false),
        ]),
        FusionProductSpace::new([]),
    );
    let error = su2_overflow
        .try_fusion_tree_layout_data_uncached_checked(&SU2FusionRule)
        .unwrap_err();
    assert_eq!(
        error,
        FusionAlgebraError::FusionNotRepresentable {
            left: su2(128),
            right: su2(127),
        }
    );

    let pair_overflow = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(Codec::encode(z2_even(), u1(i32::MAX)), 1)], false),
            SectorLeg::new([(Codec::encode(z2_odd(), u1(1)), 1)], false),
        ]),
        FusionProductSpace::new([]),
    );
    let error = pair_overflow
        .try_fusion_tree_layout_data_uncached_checked(&rule)
        .unwrap_err();
    assert_eq!(
        error,
        FusionAlgebraError::U1FusionOverflow {
            left: i32::MAX,
            right: 1,
        }
    );

    type PairLayout = ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>;
    type TripleCodec = PackedProductCodec<PairLayout, Su2SectorLayout>;
    type TripleRule = ProductFusionRule<Rule, SU2FusionRule, TripleCodec>;
    let triple_rule = TripleRule::try_new(rule, SU2FusionRule).unwrap();
    let triple = |parity, charge, spin| TripleCodec::encode(Codec::encode(parity, charge), spin);
    let triple_overflow = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(triple(z2_even(), u1(i32::MAX), su2(0)), 1)], false),
            SectorLeg::new([(triple(z2_odd(), u1(1), su2(1)), 1)], false),
        ]),
        FusionProductSpace::new([]),
    );
    let error = triple_overflow
        .try_fusion_tree_layout_data_uncached_checked(&triple_rule)
        .unwrap_err();
    assert_eq!(
        error,
        FusionAlgebraError::U1FusionOverflow {
            left: i32::MAX,
            right: 1,
        }
    );
}

fn assert_failed_checked_build_is_transactional<R>(
    rule: &R,
    hom: &FusionTreeHomSpace,
    expected: FusionAlgebraError,
) where
    R: MultiplicityFreeFusionRule + CheckedFusionAlgebra,
{
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    clear_structure_caches();
    let recording = crate::probe::record();
    let error = hom.prepare_fusion_tree_layout_checked(rule).unwrap_err();
    assert_eq!(error, expected);
    assert_eq!(
        (
            recording.get(crate::probe::Event::LayoutBuild),
            recording.get(crate::probe::Event::LayoutAdmission)
        ),
        (0, 0)
    );
}

#[test]
fn failed_checked_algebra_builds_publish_no_identity_or_intern_state() {
    // What: invalid built-in U1, SU2, and product closure leaves layout
    // identity/admission, HomSpace, and BlockStructure state untouched.
    let u1_overflow = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(u1(i32::MAX), 1)], false),
            SectorLeg::new([(u1(1), 1)], false),
        ]),
        FusionProductSpace::new([]),
    );
    assert_failed_checked_build_is_transactional(
        &U1FusionRule,
        &u1_overflow,
        FusionAlgebraError::U1FusionOverflow {
            left: i32::MAX,
            right: 1,
        },
    );

    let su2_overflow = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(su2(128), 1)], false),
            SectorLeg::new([(su2(127), 1)], false),
        ]),
        FusionProductSpace::new([]),
    );
    assert_failed_checked_build_is_transactional(
        &SU2FusionRule,
        &su2_overflow,
        FusionAlgebraError::FusionNotRepresentable {
            left: su2(128),
            right: su2(127),
        },
    );

    type Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
    type Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule, Codec>;
    let product_rule = Rule::try_new(FermionParityFusionRule, U1FusionRule).unwrap();
    let product_overflow = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(Codec::encode(z2_even(), u1(i32::MAX)), 1)], false),
            SectorLeg::new([(Codec::encode(z2_odd(), u1(1)), 1)], false),
        ]),
        FusionProductSpace::new([]),
    );
    assert_failed_checked_build_is_transactional(
        &product_rule,
        &product_overflow,
        FusionAlgebraError::U1FusionOverflow {
            left: i32::MAX,
            right: 1,
        },
    );
}

#[test]
fn empty_lowered_leg_short_circuits_before_other_leg_decode() {
    // What: an empty product has no tuples and returns empty even when a
    // different leg carries an ID that would fail lowered decoding.
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(SectorId::new(usize::MAX), 1)], false),
            SectorLeg::new(Vec::<(SectorId, usize)>::new(), false),
        ]),
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
    );
    let layout = hom
        .try_fusion_tree_layout_data_uncached_checked(&U1FusionRule)
        .unwrap();
    assert!(layout.keys.is_empty());
}
