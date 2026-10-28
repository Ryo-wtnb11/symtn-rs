use super::*;
use std::convert::Infallible;

fn degeneracy_cache_info() -> StructureCacheInfo {
    structure_cache_info(StructureCacheKind::DegeneracyStructure)
}

#[test]
fn complete_homspace_layout_cache_owns_its_wrapper_and_charges_its_regions() {
    // Isolated like prepared_complete_structure_hits_without_rebuilding_layout:
    // this asserts process-global complete-structure cache counters, which
    // CACHE_TEST_LOCK does not protect from the crate's many ordinary,
    // unlocked complete-structure builds landing between two reads (#1903).
    if test_support::run_isolated_or_return(
        "SYMTN_CORE_COMPLETE_HOMSPACE_REUSE_ISOLATED",
        "tests::fusion_space::layout_caches::complete_homspace_layout_cache_owns_its_wrapper_and_charges_its_regions",
    ) {
        return;
    }
    // What: independently constructed complete multiplicity-free U1,
    // SU2, and product HomSpaces share one canonical wrapper, which the
    // cache entry owns with its region memo; the memo joins the entry's
    // charge when it materializes.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    clear_structure_caches();

    fn assert_reused<R>(rule: &R, first_hom: FusionTreeHomSpace, second_hom: FusionTreeHomSpace)
    where
        R: MultiplicityFreeFusionRule,
    {
        let first = first_hom
            .coupled_subblock_structure_from_leg_degeneracies(rule)
            .unwrap();
        let info = degeneracy_cache_info();
        // Live wrapper: the hit returns the canonical Arc itself, without
        // re-interning content.
        let second = second_hom
            .coupled_subblock_structure_from_leg_degeneracies(rule)
            .unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        let after = degeneracy_cache_info();
        assert_eq!(after.hits(), info.hits() + 1);
        assert_eq!(after.misses(), info.misses());
        assert_eq!(after.admissions(), info.admissions());

        // No caller holds the wrapper: the entry still does, so the next
        // hit returns it with its region memo.
        let region = first.weak_region_state();
        let first = Arc::downgrade(&first);
        drop(second);
        assert!(region.upgrade().is_some());
        let charged = degeneracy_cache_info().charged_bytes();
        let third = second_hom
            .coupled_subblock_structure_from_leg_degeneracies(rule)
            .unwrap();
        assert!(Arc::ptr_eq(&third, &first.upgrade().unwrap()));
        let regions = third.coupled_sector_regions(1).unwrap().unwrap();
        let grown = degeneracy_cache_info().charged_bytes() - charged;
        assert!(
            grown >= std::mem::size_of_val(regions.as_ref()) as u64,
            "{grown}"
        );
        // A repeated query reuses the memo and charges nothing more.
        let again = third.coupled_sector_regions(1).unwrap().unwrap();
        assert!(Arc::ptr_eq(&regions, &again));
        assert_eq!(degeneracy_cache_info().charged_bytes() - charged, grown);
        let after = degeneracy_cache_info();
        assert_eq!(after.hits(), info.hits() + 2);
        assert_eq!(after.misses(), info.misses());
        assert_eq!(after.admissions(), info.admissions());
    }

    let u1_hom = || FusionTreeHomSpace::from_sectors([(u1(1), 2)], [(u1(1), 3)]);
    assert_reused(&U1FusionRule, u1_hom(), u1_hom());
    let su2_hom = || FusionTreeHomSpace::from_sectors([(su2(1), 2)], [(su2(1), 3)]);
    assert_reused(&SU2FusionRule, su2_hom(), su2_hom());

    type Fz2U1 = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
    type Product = ProductFusionRule<Fz2U1, SU2FusionRule>;
    let pair = Fz2U1::try_new(FermionParityFusionRule, U1FusionRule).unwrap();
    let sector = pair.encode_component_ids(z2_odd(), u1(1));
    let rule = Product::try_new(pair, SU2FusionRule).unwrap();
    let product_hom = || {
        FusionTreeHomSpace::from_sectors(
            [(rule.encode_component_ids(sector, su2(1)), 2)],
            [(rule.encode_component_ids(sector, su2(1)), 3)],
        )
    };
    assert_reused(&rule, product_hom(), product_hom());

    let info = degeneracy_cache_info();
    assert_eq!(info.entries(), 3);
    assert_eq!(info.admissions(), 3);
    assert!(info.hits() >= 3);
}

#[test]
fn complete_homspace_layout_cache_concurrent_hits_share_one_canonical_arc() {
    // Isolated like symtn-tensors #649/#650's checked_bind_failure
    // test: this asserts absolute process-global admissions()/
    // misses()/hits() counts, which CACHE_TEST_LOCK does not protect
    // from the crate's many ordinary, unlocked complete-structure
    // builds landing in the same narrow window. Found flaking
    // (left: 3/2, right: 1 at line 2898-ish) during #1598/#1606
    // verification; same class as that pair's four named instances.
    if test_support::run_isolated_or_return(
            "SYMTN_CORE_COMPLETE_HOMSPACE_CONCURRENT_HITS_ISOLATED",
            "tests::fusion_space::layout_caches::complete_homspace_layout_cache_concurrent_hits_share_one_canonical_arc",
        ) {
            return;
        }
    // What: threads looking up one key concurrently, with and without a
    // live wrapper between rounds, all receive the same canonical Arc from
    // a single admission.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    clear_structure_caches();
    let hom = || FusionTreeHomSpace::from_sectors([(su2(1), 2), (su2(3), 1)], [(su2(1), 3)]);
    let held = hom()
        .coupled_subblock_structure_from_leg_degeneracies(&SU2FusionRule)
        .unwrap();

    let round = || {
        let barrier = std::sync::Barrier::new(4);
        let results = std::thread::scope(|scope| {
            let handles = (0..4)
                .map(|_| {
                    scope.spawn(|| {
                        barrier.wait();
                        hom()
                            .coupled_subblock_structure_from_leg_degeneracies(&SU2FusionRule)
                            .unwrap()
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(results.iter().all(|r| Arc::ptr_eq(r, &results[0])));
        results
    };

    let live = round();
    assert!(Arc::ptr_eq(&live[0], &held));
    drop(live);
    drop(held);
    let rebuilt = round();
    // The entry's wrapper is the rebuilt Arc.
    let key = CompleteHomSpaceStructureCacheKey::new(&SU2FusionRule, &hom());
    let before = degeneracy_cache_info().hits();
    let entry = degeneracy_structure_cache().get(&key).unwrap();
    assert!(Arc::ptr_eq(&entry.structure(), &rebuilt[0]));
    assert_eq!(degeneracy_cache_info().hits(), before + 1);
    assert_eq!(degeneracy_cache_info().admissions(), 1);
    assert_eq!(degeneracy_cache_info().misses(), 1);
    assert_eq!(degeneracy_cache_info().hits(), 9);
    drop(rebuilt);
}

#[test]
fn complete_homspace_layout_cache_keys_semantics_and_preserves_direct_layout() {
    // Isolated like symtn-tensors #649/#650's checked_bind_failure
    // test: this asserts an absolute process-global entries() count,
    // which CACHE_TEST_LOCK does not protect from the crate's many
    // ordinary, unlocked complete-structure builds landing in the
    // same narrow window. Found flaking (left: 5, right: 4 at line
    // 2960-ish) during #1598/#1606 verification; same class as that
    // pair's four named instances.
    if test_support::run_isolated_or_return(
            "SYMTN_CORE_COMPLETE_HOMSPACE_KEYS_SEMANTICS_ISOLATED",
            "tests::fusion_space::layout_caches::complete_homspace_layout_cache_keys_semantics_and_preserves_direct_layout",
        ) {
            return;
        }
    // What: rule identity, degeneracies, and dual flags are distinct
    // complete-layout keys, while cache admission preserves the direct
    // builder's ordered block tuples and required storage length.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    clear_structure_caches();

    let base = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(u1(0), 2), (u1(1), 3)], false)]),
        FusionProductSpace::new([SectorLeg::new([(u1(0), 2), (u1(1), 3)], false)]),
    );
    let cached = base
        .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
        .unwrap();
    let layout = base.fusion_tree_layout_data_uncached(&U1FusionRule);
    let (sector, degeneracy) =
        coupled_subblock_parts_from_leg_degeneracies(&base, &layout).unwrap();
    let direct = BlockStructure::from_shared_parts(sector, degeneracy).unwrap();
    let signature = |structure: &BlockStructure| {
        (0..structure.block_count())
            .map(|index| {
                let block = structure.block(index).unwrap();
                (
                    block.key().clone(),
                    block.shape().to_vec(),
                    block.strides().to_vec(),
                    block.offset(),
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(signature(&cached), signature(&direct));
    assert_eq!(cached.required_len(), direct.required_len());

    let different_degeneracy = FusionTreeHomSpace::from_sectors([(u1(0), 4)], [(u1(0), 2)]);
    let dual = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(u1(0), 2), (u1(1), 3)], true)]),
        FusionProductSpace::new([SectorLeg::new([(u1(0), 2), (u1(1), 3)], false)]),
    );
    let rule_changed = FusionTreeHomSpace::from_sectors([(z2_even(), 2)], [(z2_even(), 2)]);
    different_degeneracy
        .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
        .unwrap();
    dual.coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
        .unwrap();
    rule_changed
        .coupled_subblock_structure_from_leg_degeneracies(&Z2FusionRule)
        .unwrap();
    assert_eq!(degeneracy_cache_info().entries(), 4);

    let live = Arc::clone(&cached);
    clear_structure_caches();
    assert_eq!(live.required_len(), cached.required_len());
    assert_eq!(degeneracy_cache_info().entries(), 0);
}

#[test]
fn fusion_layout_shape_and_fermionic_rule_provenance_do_not_alias() {
    // What: one sector layout may be shared across degeneracies, but concrete
    // shapes and bosonic/fermionic rule provenance select distinct structures/layouts.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    clear_structure_caches();
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(z2_even(), 1), (z2_odd(), 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(z2_even(), 1), (z2_odd(), 1)], false)]),
    );
    let bosonic_layout = hom.cached_fusion_tree_layout(&Z2FusionRule);
    let fermionic_layout = hom.cached_fusion_tree_layout(&FermionParityFusionRule);
    assert!(!Arc::ptr_eq(&bosonic_layout, &fermionic_layout));

    let small = hom
        .coupled_subblock_structure(&FermionParityFusionRule, 1, [vec![1, 1], vec![1, 1]])
        .unwrap();
    let large_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(z2_even(), 2), (z2_odd(), 3)], false)]),
        FusionProductSpace::new([SectorLeg::new([(z2_even(), 2), (z2_odd(), 3)], false)]),
    );
    let large = large_hom
        .coupled_subblock_structure(&FermionParityFusionRule, 1, [vec![2, 2], vec![3, 3]])
        .unwrap();
    assert!(!Arc::ptr_eq(&small, &large));
    assert_ne!(small.as_ref(), large.as_ref());

    let transient_hom =
        FusionTreeHomSpace::from_sectors([(U1Irrep::new(17), 4)], [(U1Irrep::new(17), 5)]);
    let transient = transient_hom
        .coupled_subblock_structure(&U1FusionRule, 1, [vec![4, 5]])
        .unwrap();
    // The cache entry owns the canonical wrapper: dropping the caller's
    // handle keeps it, and the next build returns the same one.
    let owned = Arc::downgrade(&transient);
    drop(transient);
    let rebuilt = transient_hom
        .coupled_subblock_structure(&U1FusionRule, 1, [vec![4, 5]])
        .unwrap();
    assert!(Arc::ptr_eq(&rebuilt, &owned.upgrade().unwrap()));
    assert_eq!(rebuilt.block(0).unwrap().shape(), &[4, 5]);
    drop(rebuilt);
    clear_structure_caches();
    assert!(owned.upgrade().is_none(), "a clear releases the wrapper");
}

#[test]
fn explicit_shape_coupled_structure_reuses_the_bounded_leg_cache() {
    // What: with shapes equal to the leg degeneracies, the explicit-shape
    // constructor is answered by the byte-bounded complete-HomSpace cache: one
    // leg-grid build on the miss, none on the hit, and the same shared value
    // as the leg-derived constructor.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    clear_structure_caches();
    let leg = SectorLeg::new([(u1(-1), 2), (u1(0), 3), (u1(2), 1)], false);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone(), leg.clone()]),
        FusionProductSpace::new([leg.clone(), leg]),
    );
    let layout = hom.fusion_tree_layout_data_uncached(&U1FusionRule);
    let grid_sides = layout
        .sectors
        .iter()
        .map(|sector| sector.row_count + sector.col_count)
        .sum::<usize>();
    let shapes = hom
        .fusion_tree_keys(&U1FusionRule)
        .iter()
        .map(|key| {
            key.codomain_uncoupled()
                .iter()
                .chain(key.domain_uncoupled())
                .map(|&sector| hom.codomain().legs()[0].degeneracy(sector).unwrap())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let misses = crate::probe::record();
    let recording = crate::probe::record();
    let first = hom
        .coupled_subblock_structure(&U1FusionRule, 2, shapes.clone())
        .unwrap();
    assert_eq!(
        recording.get(crate::probe::Event::CoupledGridSideDerivation),
        grid_sides
    );
    assert_eq!(misses.get(crate::probe::Event::CompleteHomSpaceMiss), 1);

    let recording = crate::probe::record();
    let second = hom
        .coupled_subblock_structure(&U1FusionRule, 2, shapes)
        .unwrap();
    let from_legs = hom
        .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
        .unwrap();
    assert_eq!(
        (
            recording.get(crate::probe::Event::CoupledGridReconstructionInsert),
            recording.get(crate::probe::Event::CoupledGridSideDerivation)
        ),
        (0, 0)
    );
    assert_eq!(misses.get(crate::probe::Event::CompleteHomSpaceMiss), 1);
    assert!(Arc::ptr_eq(&first, &second));
    assert!(Arc::ptr_eq(&first, &from_legs));
    assert!(first.storage_tiling_proven());
}

fn reset_test_homspace() -> FusionTreeHomSpace {
    FusionTreeHomSpace::from_sectors([(U1Irrep::new(3), 2)], [(U1Irrep::new(3), 5)])
}

/// One build of the complete structure, as `coupled_subblock_structure_from_leg_degeneracies`
/// performs it before admission.
fn complete_build(
    homspace: &FusionTreeHomSpace,
) -> (CompleteHomSpaceStructureCacheKey, Arc<BlockStructure>) {
    let key = CompleteHomSpaceStructureCacheKey::new(&U1FusionRule, homspace);
    let layout = homspace.cached_fusion_tree_layout(&U1FusionRule);
    let (sector, degeneracy) =
        coupled_subblock_parts_from_leg_degeneracies(homspace, &layout).unwrap();
    let built = BlockStructure::from_shared_parts(sector, degeneracy)
        .unwrap()
        .into_shared();
    (key, built)
}

#[test]
fn complete_cache_publishes_a_build_that_saw_no_reset() {
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    clear_structure_caches();
    let homspace = reset_test_homspace();
    let epoch = core_reset_epoch();
    let (key, built) = complete_build(&homspace);

    let (published, canonical) = admit_complete_hom_space_structure(key, Arc::clone(&built), epoch);

    // What: the epoch check does not block an ordinary miss.
    assert!(Arc::ptr_eq(&canonical, &built));
    assert!(Arc::ptr_eq(&published.structure(), &built));
    let hit = homspace
        .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
        .unwrap();
    assert!(Arc::ptr_eq(&hit, &built));
}

#[test]
fn complete_admission_preserves_the_winner_wrapper_and_region_memo() {
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    clear_structure_caches();
    let homspace = reset_test_homspace();
    let epoch = core_reset_epoch();
    let (key, winner) = complete_build(&homspace);
    let regions = winner.coupled_sector_regions(1).unwrap().unwrap();

    let (_, canonical) = admit_complete_hom_space_structure(key, Arc::clone(&winner), epoch);
    assert!(Arc::ptr_eq(&canonical, &winner));
    assert!(Arc::ptr_eq(
        &canonical.coupled_sector_regions(1).unwrap().unwrap(),
        &regions
    ));

    let (same_key, loser) = complete_build(&homspace);
    assert_ne!(loser.content_id(), winner.content_id());
    let (_, raced) = admit_complete_hom_space_structure(same_key, loser, epoch);
    assert!(Arc::ptr_eq(&raced, &winner));
    assert!(Arc::ptr_eq(
        &raced.coupled_sector_regions(1).unwrap().unwrap(),
        &regions
    ));
}

#[test]
fn complete_cache_separates_multiplicity_free_and_generic_modes() {
    if test_support::run_isolated_or_return(
        "SYMTN_COMPLETE_OWNER_MODE_ISOLATED",
        "tests::fusion_space::layout_caches::complete_cache_separates_multiplicity_free_and_generic_modes",
    ) { return; }
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    clear_structure_caches();
    let hom = FusionTreeHomSpace::from_sectors([(su2(1), 2)], [(su2(1), 3)]);

    let multiplicity_free = hom
        .coupled_subblock_structure_from_leg_degeneracies(&SU2FusionRule)
        .unwrap();
    let generic = hom
        .coupled_subblock_structure_from_leg_degeneracies_generic(&SU2FusionRule)
        .unwrap();

    assert_eq!(multiplicity_free.as_ref(), generic.as_ref());
    assert_ne!(multiplicity_free.content_id(), generic.content_id());
    assert_eq!(degeneracy_cache_info().entries(), 2);
    for _ in 0..2 {
        assert!(Arc::ptr_eq(
            &generic,
            &hom.coupled_subblock_structure_from_leg_degeneracies_generic(&SU2FusionRule)
                .unwrap()
        ));
        assert!(Arc::ptr_eq(
            &multiplicity_free,
            &hom.coupled_subblock_structure_from_leg_degeneracies(&SU2FusionRule)
                .unwrap()
        ));
    }
    let budget = degeneracy_cache_info().byte_budget();
    set_structure_cache_byte_budget(StructureCacheKind::DegeneracyStructure, 0);
    let rejected_mf = hom
        .coupled_subblock_structure_from_leg_degeneracies(&SU2FusionRule)
        .unwrap();
    let rejected_generic = hom
        .coupled_subblock_structure_from_leg_degeneracies_generic(&SU2FusionRule)
        .unwrap();
    assert_eq!(rejected_mf.as_ref(), multiplicity_free.as_ref());
    assert_eq!(rejected_generic.as_ref(), generic.as_ref());
    assert_ne!(rejected_mf.content_id(), multiplicity_free.content_id());
    assert_ne!(rejected_generic.content_id(), generic.content_id());
    // What: residency alone makes content canonical (#2014): admitted
    // winners are, zero-budget rejections are not.
    assert!(multiplicity_free.is_canonical() && generic.is_canonical());
    assert!(!rejected_mf.is_canonical() && !rejected_generic.is_canonical());
    assert_eq!(degeneracy_cache_info().entries(), 0);
    assert_eq!(degeneracy_cache_info().charged_bytes(), 0);
    assert!(degeneracy_cache_info().rejections() >= 2);
    set_structure_cache_byte_budget(StructureCacheKind::DegeneracyStructure, budget);
    clear_structure_caches();
    let fresh_generic = hom
        .coupled_subblock_structure_from_leg_degeneracies_generic(&SU2FusionRule)
        .unwrap();
    let fresh_mf = hom
        .coupled_subblock_structure_from_leg_degeneracies(&SU2FusionRule)
        .unwrap();
    assert_ne!(fresh_mf.content_id(), rejected_mf.content_id());
    assert_ne!(fresh_generic.content_id(), rejected_generic.content_id());
    assert!(fresh_mf.is_canonical() && fresh_generic.is_canonical());
    // The flag outlives the clear that evicted these contents.
    assert!(multiplicity_free.is_canonical());
    assert_eq!(degeneracy_cache_info().entries(), 2);
}

#[test]
fn checked_homspace_factory_captures_reset_epoch_before_the_producer() {
    if test_support::run_isolated_or_return(
        "SYMTN_COMPLETE_OWNER_PRODUCER_EPOCH_ISOLATED",
        "tests::fusion_space::layout_caches::checked_homspace_factory_captures_reset_epoch_before_the_producer",
    ) { return; }
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    clear_structure_caches();
    let rule = InfallibleGeneric::new(&SU2FusionRule);
    let hom = || FusionTreeHomSpace::from_sectors([(su2(1), 2)], [(su2(1), 3)]);

    let prepared =
        FusionTreeHomSpace::prepare_complete_coupled_subblock_structure_generic_checked_with(
            &rule,
            |_| {
                clear_structure_caches();
                Ok::<_, CheckedGenericStructureError<Infallible>>(hom())
            },
        )
        .unwrap();
    let stale = prepared.commit_with_complete_homspace().1;
    assert_eq!(degeneracy_cache_info().entries(), 0);

    let fresh = hom()
        .coupled_subblock_structure_from_leg_degeneracies_generic_checked(&rule)
        .unwrap();
    assert_eq!(stale.as_ref(), fresh.as_ref());
    assert_ne!(stale.content_id(), fresh.content_id());
    assert_eq!(degeneracy_cache_info().entries(), 1);
}

#[test]
fn complete_cache_drops_a_build_that_started_before_a_reset() {
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    clear_structure_caches();
    let homspace = reset_test_homspace();
    // Order 1: a build records its epoch and interns content, then a reset runs,
    // then the build admits.
    let epoch = core_reset_epoch();
    let (key, stale) = complete_build(&homspace);
    clear_structure_caches();

    let (returned, canonical) = admit_complete_hom_space_structure(key, Arc::clone(&stale), epoch);

    // What: the straddling build keeps its correct result, but the cache does
    // not publish it, so a post-reset build mints a fresh content identity.
    assert!(Arc::ptr_eq(&canonical, &stale));
    assert!(Arc::ptr_eq(&returned.structure(), &stale));
    let fresh = homspace
        .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
        .unwrap();
    assert_ne!(fresh.content_id(), stale.content_id());
    assert_eq!(fresh.as_ref(), stale.as_ref());
}

#[test]
fn complete_cache_drops_a_build_that_ran_inside_a_reset() {
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    clear_structure_caches();
    let homspace = reset_test_homspace();
    let source = homspace
        .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
        .unwrap();
    // Order 2: a build runs after the reset cleared the complete cache but before
    // it cleared the intern tables, so it interns the pre-reset content.
    let during = std::rc::Rc::new(std::cell::RefCell::new(None));
    let slot = std::rc::Rc::clone(&during);
    let hom = homspace.clone();
    crate::probe::set_mid_reset_hook(Box::new(move || {
        let epoch = core_reset_epoch();
        *slot.borrow_mut() = Some((epoch, complete_build(&hom)));
    }));
    clear_structure_caches();
    let (epoch, (key, stale)) = during.borrow_mut().take().unwrap();
    assert_eq!(stale.as_ref(), source.as_ref());
    assert_ne!(stale.content_id(), source.content_id());

    admit_complete_hom_space_structure(key, stale, epoch);

    // What: no pre-reset content identity is published after the reset.
    let fresh = homspace
        .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
        .unwrap();
    assert_ne!(fresh.content_id(), source.content_id());
}

#[test]
fn overlapping_resets_wait_so_the_epoch_stays_odd_until_the_first_finishes() {
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    clear_structure_caches();
    let homspace = reset_test_homspace();
    // Kept alive so the intern table still resolves its content mid-reset.
    let source = homspace
        .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
        .unwrap();
    let source_id = source.content_id();
    let start = core_reset_epoch();
    let second_done = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let second = std::rc::Rc::new(std::cell::RefCell::new(None));
    let (slot, done, hom) = (
        std::rc::Rc::clone(&second),
        Arc::clone(&second_done),
        homspace.clone(),
    );
    crate::probe::set_mid_reset_hook(Box::new(move || {
        // Reset B starts on another thread while reset A is mid-way.
        let finished = Arc::clone(&done);
        *slot.borrow_mut() = Some(std::thread::spawn(move || {
            clear_structure_caches();
            finished.store(true, std::sync::atomic::Ordering::SeqCst);
        }));
        std::thread::sleep(std::time::Duration::from_millis(50));
        // What: B cannot run inside A, so the epoch stays odd and a build
        // here is not published.
        assert!(!done.load(std::sync::atomic::Ordering::SeqCst));
        assert_eq!(core_reset_epoch(), start + 1);
        let inside = hom
            .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
            .unwrap();
        assert_eq!(inside.as_ref(), source.as_ref());
        assert_ne!(inside.content_id(), source_id);
    }));
    clear_structure_caches();
    second.borrow_mut().take().unwrap().join().unwrap();

    assert!(second_done.load(std::sync::atomic::Ordering::SeqCst));
    assert_eq!(core_reset_epoch(), start + 4);
    let after = homspace
        .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
        .unwrap();
    assert_ne!(after.content_id(), source_id);
}

#[test]
fn a_degeneracy_only_change_reuses_the_sector_structure() {
    // What (#2014, variable shape): two HomSpaces with the same sectors and
    // different degeneracies enumerate their fusion trees once and share one
    // sector structure; only the degeneracy structure differs. TensorKit
    // keys `sectorstructure` on the sectors alone (`structure.jl:41`).
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    clear_structure_caches();
    let hom = |small: usize, large: usize| {
        let leg = SectorLeg::new([(u1(-1), small), (u1(0), large), (u1(1), small)], false);
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg.clone(), leg.clone()]),
            FusionProductSpace::new([leg]),
        )
    };
    let recording = crate::probe::record();
    let first = hom(1, 2)
        .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
        .unwrap();
    let second = hom(3, 5)
        .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
        .unwrap();
    assert_eq!(
        (
            recording.get(crate::probe::Event::LayoutBuild),
            recording.get(crate::probe::Event::LayoutAdmission)
        ),
        (1, 1)
    );
    assert!(Arc::ptr_eq(
        &first.content_key().sector,
        &second.content_key().sector
    ));
    assert_ne!(first.required_len(), second.required_len());

    // The Generic builder shares through the same cache, under its own key.
    let su2_hom = |deg: usize| {
        let leg = SectorLeg::new([(su2(1), deg), (su2(2), deg + 1)], false);
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg.clone(), leg.clone()]),
            FusionProductSpace::new([leg]),
        )
    };
    let small = su2_hom(1)
        .coupled_subblock_structure_from_leg_degeneracies_generic(&SU2FusionRule)
        .unwrap();
    let large = su2_hom(4)
        .coupled_subblock_structure_from_leg_degeneracies_generic(&SU2FusionRule)
        .unwrap();
    assert!(Arc::ptr_eq(
        &small.content_key().sector,
        &large.content_key().sector
    ));
    assert_ne!(small.required_len(), large.required_len());

    // So does the checked Generic builder (#2030), once a walk succeeded.
    let checked = InfallibleGeneric::new(&SU2FusionRule);
    let checked_small = su2_hom(2)
        .coupled_subblock_structure_from_leg_degeneracies_generic_checked(&checked)
        .unwrap();
    let checked_large = su2_hom(5)
        .coupled_subblock_structure_from_leg_degeneracies_generic_checked(&checked)
        .unwrap();
    assert!(Arc::ptr_eq(
        &checked_small.content_key().sector,
        &checked_large.content_key().sector
    ));

    #[cfg(feature = "cgc-gen")]
    {
        let checked = symtn_sectors::testing::su_lie_rule(3);
        let vacuum = checked.encode_irrep(&[0, 0], &[]).unwrap();
        let fundamental = checked.encode_irrep(&[1, 0], &[]).unwrap();
        let hom = |deg: usize| {
            let leg = SectorLeg::new([(vacuum, deg), (fundamental, deg + 1)], false);
            FusionTreeHomSpace::new(
                FusionProductSpace::new([leg.clone(), leg.clone()]),
                FusionProductSpace::new([leg]),
            )
        };
        let small = hom(1)
            .coupled_subblock_structure_from_leg_degeneracies_generic_checked(&checked)
            .unwrap();
        let large = hom(4)
            .coupled_subblock_structure_from_leg_degeneracies_generic_checked(&checked)
            .unwrap();
        assert!(Arc::ptr_eq(
            &small.content_key().sector,
            &large.content_key().sector
        ));
        assert_ne!(small.required_len(), large.required_len());
    }
}

/// A provider whose fusion channels repeat: `1 ⊗ 1` lists the vacuum twice.
#[derive(Clone, Copy, Debug)]
struct RepeatedChannelRule;

impl symtn_sectors::testing::Provider for RepeatedChannelRule {}

impl FusionRule for RepeatedChannelRule {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }
    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }
    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        let fused = SectorId::new((left.id() + right.id()) % 2);
        if left.id() == 1 && right.id() == 1 {
            smallvec![fused, fused]
        } else {
            smallvec![fused]
        }
    }
}

impl MultiplicityFreeFusionRule for RepeatedChannelRule {}

#[test]
fn repeated_fusion_channels_are_a_typed_error_not_a_duplicate_block() {
    // What (#2014): duplicate tree keys from a provider with repeated
    // channels are `DuplicateBlockKey`, not a structure with two equal keys.
    let leg = || SectorLeg::new([(SectorId::new(0), 1), (SectorId::new(1), 1)], false);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg(), leg()]),
        FusionProductSpace::new([leg()]),
    );
    let error = hom
        .coupled_subblock_structure_from_leg_degeneracies(&RepeatedChannelRule)
        .unwrap_err();
    assert!(
        matches!(error, CoreError::DuplicateBlockKey { .. }),
        "{error:?}"
    );
}

#[cfg(feature = "cgc-gen")]
#[test]
fn generic_complete_owner_keeps_canonical_identity_regions_and_budget() {
    if test_support::run_isolated_or_return(
        "SYMTN_GENERIC_OWNER_CANONICAL_BUDGET",
        "tests::fusion_space::layout_caches::generic_complete_owner_keeps_canonical_identity_regions_and_budget",
    ) { return; }
    clear_structure_caches();
    let rule = symtn_sectors::testing::su_lie_rule(3);
    let adjoint = rule.encode_irrep(&[1, 1], &[]).unwrap();
    let vacuum = rule.encode_irrep(&[0, 0], &[]).unwrap();
    let hom = || {
        let leg = || SectorLeg::new([(adjoint, 2), (vacuum, 3)], false);
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg()]),
            FusionProductSpace::new([leg(), leg()]),
        )
    };
    let first_hom = hom();
    let first_id = first_hom.id();
    let other_hom = hom();
    let other_id = other_hom.id();
    assert_eq!(first_id, other_id);
    assert!(!first_id.downgrade().matches(&other_id));
    let prepare = |hom: FusionTreeHomSpace| {
        hom.prepare_complete_coupled_subblock_structure_generic_checked_after(&rule, |_| Ok(()))
            .unwrap()
    };
    let first = prepare(first_hom);
    let raced = prepare(other_hom);
    let preview_regions = first
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .unwrap();
    assert!(!preview_regions.is_empty());
    assert_eq!(degeneracy_cache_info().entries(), 0);
    let (canonical_hom, winner) = first.commit_with_complete_homspace();
    let canonical_hom = canonical_hom.unwrap();
    let (raced_hom, second) = raced.commit_with_complete_homspace();
    assert!(Arc::ptr_eq(&winner, &second));
    assert!(canonical_hom
        .id()
        .downgrade()
        .matches(&raced_hom.unwrap().id()));
    assert!(Arc::ptr_eq(
        &preview_regions,
        &winner.coupled_sector_regions(2).unwrap().unwrap()
    ));
    let key = CompleteHomSpaceStructureCacheKey::generic(
        CheckedGenericFusion::rule_identity(&rule),
        &canonical_hom,
    );
    // The owned wrapper's region memo, built on the preview, is charged too.
    let charge = charged_complete_hom_space_structure_bytes(&key, &winner.content_key())
        + winner.materialized_region_bytes();
    assert!(winner.materialized_region_bytes() > 0);
    let info = degeneracy_cache_info();
    assert_eq!(info.entries(), 1);
    assert_eq!(info.charged_bytes(), charge);
    assert!(
        charge
            >= std::mem::size_of_val(key.homspace.as_ref()) as u64
                + winner.content_key().charged_retained_bytes() as u64
    );
    // Independently count the dynamic arrays reachable from the one retained
    // HomSpace backing; the entry's id and semantic key share those arrays.
    let leg_backing_floor = canonical_hom
        .codomain()
        .legs()
        .iter()
        .chain(canonical_hom.domain().legs())
        .map(|leg| std::mem::size_of_val(leg.sectors()) + std::mem::size_of_val(leg.degeneracies()))
        .sum::<usize>()
        + canonical_hom.rank() * std::mem::size_of::<SectorLeg>();
    assert!(
        charge
            >= winner.content_key().charged_retained_bytes() as u64
                + std::mem::size_of_val(key.homspace.as_ref()) as u64
                + leg_backing_floor as u64
    );
    let content_id = winner.content_id();
    drop((winner, second, preview_regions));
    let (revived_hom, revived) = prepare(hom()).commit_with_complete_homspace();
    assert_eq!(revived.content_id(), content_id);
    // A committed staged candidate became resident, hence canonical.
    assert!(revived.is_canonical());
    assert!(canonical_hom
        .id()
        .downgrade()
        .matches(&revived_hom.unwrap().id()));
    let (_, again) = prepare(hom()).commit_with_complete_homspace();
    assert!(Arc::ptr_eq(&revived, &again));
    assert_eq!(degeneracy_cache_info().charged_bytes(), charge);
    // A staged hit lends the live canonical wrapper to pre-commit plans.
    let hit = prepare(hom());
    assert!(Arc::ptr_eq(&hit.shared_structure(), &revived));
    assert!(Arc::ptr_eq(
        &hit.commit_with_complete_homspace().1,
        &revived
    ));

    let budget = info.byte_budget();
    set_structure_cache_byte_budget(StructureCacheKind::DegeneracyStructure, 0);
    // A miss lends a local wrapper whose region memo the published
    // structure keeps.
    let miss = prepare(hom());
    let lent_regions = miss
        .shared_structure()
        .coupled_sector_regions(2)
        .unwrap()
        .unwrap();
    let (_, uncached) = miss.commit_with_complete_homspace();
    // A staged commit refused by a zero budget publishes nothing canonical.
    assert!(!uncached.is_canonical());
    assert!(Arc::ptr_eq(
        &lent_regions,
        &uncached.coupled_sector_regions(2).unwrap().unwrap()
    ));
    assert_eq!(*uncached, *revived);
    assert_ne!(uncached.content_id(), content_id);
    assert_eq!(degeneracy_cache_info().entries(), 0);
    assert_eq!(degeneracy_cache_info().charged_bytes(), 0);
    assert!(degeneracy_cache_info().rejections() > 0);
    set_structure_cache_byte_budget(StructureCacheKind::DegeneracyStructure, budget);
}

/// What (#2154): a committed checked permutation memoizes its destination
/// in the source's complete entry, charged once to that entry; the next
/// staging of the same axes takes the memo without deriving, while an
/// unpublished, reset-straddling or non-resident staging derives.
#[test]
fn a_committed_permutation_memoizes_its_destination_in_the_source_entry() {
    if test_support::run_isolated_or_return(
        "SYMTN_PERMUTED_MEMO_ISOLATED",
        "tests::fusion_space::layout_caches::a_committed_permutation_memoizes_its_destination_in_the_source_entry",
    ) {
        return;
    }
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    clear_structure_caches();
    let rule = InfallibleGeneric::new(&SU2FusionRule);
    let admit = || {
        let (hom, structure) =
            FusionTreeHomSpace::from_sectors([(su2(1), 2), (su2(1), 1)], [(su2(2), 3)])
                .prepare_complete_coupled_subblock_structure_generic_checked_after(
                    &rule,
                    |_| Ok(()),
                )
                .unwrap()
                .commit_with_complete_homspace();
        (hom.unwrap(), structure)
    };
    let stage = |(hom, source): &(FusionTreeHomSpace, Arc<BlockStructure>),
                 codomain: &[usize],
                 domain: &[usize]| {
        let derived = std::cell::Cell::new(false);
        let (prepared, destination, permuted) = hom
            .prepare_complete_permuted_generic_checked::<_, CheckedGenericStructureError<Infallible>>(
                source,
                &rule,
                codomain,
                domain,
                |_| Ok(()),
                || {
                    derived.set(true);
                    Ok(())
                },
            )
            .unwrap();
        (prepared, destination, permuted, derived.get())
    };
    let commit = |prepared: PreparedBlockStructure| {
        let (hom, structure) = prepared.commit_with_complete_homspace();
        (hom.unwrap(), structure)
    };
    let source = admit();
    let swap = |source| stage(source, &[1, 0], &[2]);

    // Unpublished stagings derive; the first publication charges the entry,
    // a second one of the same key charges nothing.
    let (prepared, _, Permuted::Derived(first), true) = swap(&source) else {
        panic!("a cold staging derives");
    };
    let (raced, _, Permuted::Derived(second), true) = swap(&source) else {
        panic!("an unpublished memo is not read");
    };
    let (destination, structure) = commit(prepared);
    drop(commit(raced));
    let before = degeneracy_cache_info();
    let recording = crate::probe::record();
    first.publish(&destination, &structure);
    let charged = degeneracy_cache_info().charged_bytes() - before.charged_bytes();
    assert_eq!(degeneracy_cache_info().entries(), before.entries());
    assert_eq!(
        recording.get(crate::probe::Event::PermutedMemoPublication),
        1
    );
    // At least the key words and the destination content; the inline slot
    // it fills is charged with the entry.
    let floor = 3 * std::mem::size_of::<usize>() + destination.charged_retained_bytes()
        - std::mem::size_of::<FusionTreeHomSpace>();
    assert!(charged >= floor as u64, "{charged} < {floor}");
    second.publish(&destination, &structure);
    assert_eq!(
        degeneracy_cache_info().charged_bytes() - before.charged_bytes(),
        charged
    );
    assert_eq!(
        recording.get(crate::probe::Event::PermutedMemoPublication),
        1
    );

    // The memo answers the same axes with the canonical destination.
    let hits = crate::probe::record();
    let (prepared, memoized, Permuted::Memo, false) = swap(&source) else {
        panic!("a published memo answers without deriving");
    };
    assert_eq!(hits.get(crate::probe::Event::PermutedMemoHit), 1);
    assert_eq!(memoized, destination);
    assert!(Arc::ptr_eq(&prepared.shared_structure(), &structure));
    // Another split of the same concatenation is another key.
    let (_, _, Permuted::Derived(_), true) = stage(&source, &[1], &[0, 2]) else {
        panic!("a different split derives");
    };

    // A ticket staged before a reset publishes nothing, and the source's
    // cleared entry no longer answers.
    let (prepared, _, Permuted::Derived(stale), true) = stage(&source, &[0], &[2, 1]) else {
        panic!("a cold staging derives");
    };
    let (destination, structure) = commit(prepared);
    clear_structure_caches();
    stale.publish(&destination, &structure);
    assert_eq!(
        recording.get(crate::probe::Event::PermutedMemoPublication),
        1
    );
    assert_eq!(degeneracy_cache_info().charged_bytes(), 0);
    let (_, _, Permuted::Derived(_), true) = swap(&source) else {
        panic!("a reset drops every memo");
    };

    // A source no entry retains never hits, even after a publication.
    set_structure_cache_byte_budget(StructureCacheKind::DegeneracyStructure, 0);
    let transient = admit();
    assert!(!transient.1.is_canonical());
    for _ in 0..2 {
        let (prepared, _, Permuted::Derived(ticket), true) = swap(&transient) else {
            panic!("a non-resident source derives");
        };
        let (destination, structure) = commit(prepared);
        ticket.publish(&destination, &structure);
    }
    assert_eq!(
        recording.get(crate::probe::Event::PermutedMemoPublication),
        1
    );
    set_structure_cache_byte_budget(
        StructureCacheKind::DegeneracyStructure,
        DEGENERACY_STRUCTURE_CACHE_BYTE_BUDGET,
    );
}

/// What: the multiplicity-free memo reuses the Generic memo's storage
/// under its own mode: a published destination is charged to the source's
/// entry and answers the same axes with the canonical destination and no
/// sector-structure lookup; a source structure owned by another HomSpace's
/// entry neither answers nor stores, and a reset drops the memo.
#[test]
fn a_multiplicity_free_permutation_memoizes_its_destination_in_the_source_entry() {
    if test_support::run_isolated_or_return(
        "SYMTN_MF_PERMUTED_MEMO_ISOLATED",
        "tests::fusion_space::layout_caches::a_multiplicity_free_permutation_memoizes_its_destination_in_the_source_entry",
    ) {
        return;
    }
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    clear_structure_caches();
    let rule = U1FusionRule;
    let identity = rule.rule_identity();
    let leg = |dual| SectorLeg::new([(u1(-1), 1), (u1(0), 2), (u1(1), 1)], dual);
    let (hom, source) = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(false), leg(true)]),
        FusionProductSpace::new([leg(false)]),
    )
    .canonical_coupled_subblock_structure_from_leg_degeneracies(&rule)
    .unwrap();
    let (codomain, domain) = (&[2][..], &[1, 0][..]);
    let derive = |hom: &FusionTreeHomSpace| {
        hom.permute(&rule, codomain, domain)
            .unwrap()
            .canonical_coupled_subblock_structure_from_leg_degeneracies(&rule)
            .unwrap()
    };

    let Err(ticket) = hom.memoized_permuted_multiplicity_free(
        &source,
        &identity,
        FusionTreePairOrientation::Direct,
        codomain,
        domain,
    ) else {
        panic!("a cold permutation derives");
    };
    let (destination, structure) = derive(&hom);
    let before = degeneracy_cache_info();
    let recording = crate::probe::record();
    ticket.publish(&destination, &structure);
    assert_eq!(
        recording.get(crate::probe::Event::PermutedMemoPublication),
        1
    );
    assert_eq!(degeneracy_cache_info().entries(), before.entries());
    assert!(degeneracy_cache_info().charged_bytes() > before.charged_bytes());

    let hits = crate::probe::record();
    let Ok((memoized, memoized_structure)) = hom.memoized_permuted_multiplicity_free(
        &source,
        &identity,
        FusionTreePairOrientation::Direct,
        codomain,
        domain,
    ) else {
        panic!("a published memo answers");
    };
    assert_eq!(hits.get(crate::probe::Event::PermutedMemoHit), 1);
    assert_eq!(hits.get(crate::probe::Event::SectorStructureLookup), 0);
    assert_eq!(hits.get(crate::probe::Event::CompleteHomSpaceMiss), 0);
    assert_eq!(memoized, destination);
    assert_eq!(memoized.existing_id(), destination.existing_id());
    assert!(Arc::ptr_eq(&memoized_structure, &structure));
    // The adjoint orientation of the same axes is another key.
    assert!(hom
        .memoized_permuted_multiplicity_free(
            &source,
            &identity,
            FusionTreePairOrientation::Adjoint,
            codomain,
            domain
        )
        .is_err());

    // Another HomSpace reaching `source`'s entry neither reads nor writes it.
    let other = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(true), leg(true)]),
        FusionProductSpace::new([leg(false)]),
    );
    let (other_codomain, other_domain) = (&[0][..], &[2, 1][..]);
    let Err(foreign) = other.memoized_permuted_multiplicity_free(
        &source,
        &identity,
        FusionTreePairOrientation::Direct,
        other_codomain,
        other_domain,
    ) else {
        panic!("a foreign entry answers nothing");
    };
    let (other_destination, other_structure) = other
        .permute(&rule, other_codomain, other_domain)
        .unwrap()
        .canonical_coupled_subblock_structure_from_leg_degeneracies(&rule)
        .unwrap();
    foreign.publish(&other_destination, &other_structure);
    assert_eq!(
        recording.get(crate::probe::Event::PermutedMemoPublication),
        1
    );
    assert!(hom
        .memoized_permuted_multiplicity_free(
            &source,
            &identity,
            FusionTreePairOrientation::Direct,
            other_codomain,
            other_domain
        )
        .is_err());

    clear_structure_caches();
    assert!(hom
        .memoized_permuted_multiplicity_free(
            &source,
            &identity,
            FusionTreePairOrientation::Direct,
            codomain,
            domain
        )
        .is_err());
}

/// What: the first memo keys of an entry fill its lock-free slots and later
/// keys overflow to the map; each publication is charged either way, and a
/// hit answers the same destination with the same probe counts from a slot
/// or from the map.
#[test]
fn permuted_memo_keys_fill_the_slots_then_overflow_to_the_map() {
    if test_support::run_isolated_or_return(
        "SYMTN_PERMUTED_MEMO_SLOTS_ISOLATED",
        "tests::fusion_space::layout_caches::permuted_memo_keys_fill_the_slots_then_overflow_to_the_map",
    ) {
        return;
    }
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    clear_structure_caches();
    let rule = U1FusionRule;
    let identity = rule.rule_identity();
    let leg = |dual| SectorLeg::new([(u1(-1), 1), (u1(0), 2), (u1(1), 1)], dual);
    let (hom, source) = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(false), leg(true)]),
        FusionProductSpace::new([leg(false)]),
    )
    .canonical_coupled_subblock_structure_from_leg_degeneracies(&rule)
    .unwrap();
    let keys: [(&[usize], &[usize]); PERMUTED_SLOTS + 2] = [
        (&[2], &[1, 0]),
        (&[1], &[0, 2]),
        (&[0], &[2, 1]),
        (&[2, 0], &[1]),
        (&[1, 2], &[0]),
        (&[0, 1, 2], &[]),
    ];
    let memo = |(codomain, domain): (&[usize], &[usize])| {
        hom.memoized_permuted_multiplicity_free(
            &source,
            &identity,
            FusionTreePairOrientation::Direct,
            codomain,
            domain,
        )
    };
    let mut destinations = Vec::new();
    for (index, &(codomain, domain)) in keys.iter().enumerate() {
        let Err(ticket) = memo((codomain, domain)) else {
            panic!("an unpublished key derives");
        };
        let (destination, structure) = hom
            .permute(&rule, codomain, domain)
            .unwrap()
            .canonical_coupled_subblock_structure_from_leg_degeneracies(&rule)
            .unwrap();
        let before = degeneracy_cache_info().charged_bytes();
        let recording = crate::probe::record();
        ticket.publish(&destination, &structure);
        assert_eq!(
            recording.get(crate::probe::Event::PermutedMemoPublication),
            1,
            "key {index}"
        );
        assert!(
            degeneracy_cache_info().charged_bytes() > before,
            "key {index}"
        );
        destinations.push(destination);
    }
    for (index, &key) in keys.iter().enumerate() {
        let hits = crate::probe::record();
        let Ok((memoized, _)) = memo(key) else {
            panic!("key {index} is memoized");
        };
        assert_eq!(memoized, destinations[index]);
        assert_eq!(hits.get(crate::probe::Event::PermutedMemoHit), 1);
        assert_eq!(hits.get(crate::probe::Event::SectorStructureLookup), 0);
        assert_eq!(hits.get(crate::probe::Event::CompleteHomSpaceMiss), 0);
        assert_eq!(
            hits.get(crate::probe::Event::PermutedMemoMapRead),
            usize::from(index >= PERMUTED_SLOTS),
            "key {index}"
        );
    }
}

#[test]
fn a_sector_layout_entry_charges_its_tree_arena_once() {
    // What (single-arena ownership, #2192): the layout keys and the sector
    // structure built from them share one canonical arena, and the
    // layout-cache entry charges it once, through the structure, not again
    // for the keys.
    let leg = SectorLeg::new([(u1(-1), 1), (u1(0), 2), (u1(1), 1)], false);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone(), leg.clone(), leg.clone()]),
        FusionProductSpace::new([leg.clone(), leg]),
    );
    let key = FusionTreeHomSpaceCacheKey::new(&U1FusionRule, &hom);
    let layout = hom.fusion_tree_layout_data_uncached(&U1FusionRule);
    let sector = Arc::clone(layout.sector.as_ref().unwrap());
    let arena = Arc::clone(sector.trees.as_ref().unwrap());
    assert!(layout.keys.len() > 1);
    assert!(layout.keys.iter().all(|pair| {
        Arc::ptr_eq(&arena, &pair.codomain_tree().arena)
            && Arc::ptr_eq(&arena, &pair.domain_tree().arena)
    }));
    let owned = charged_fusion_tree_layout_bytes(&key, &layout);

    // The same entry without its structure must charge the arena itself.
    let FusionTreeHomSpaceLayoutData { keys, sectors, .. } = layout;
    let unowned = FusionTreeHomSpaceLayoutData {
        keys,
        sectors,
        sector: Err(CoreError::DuplicateBlockKey {
            key: Box::new(BlockKey::trivial()),
        }),
    };
    let alone = charged_fusion_tree_layout_bytes(&key, &unowned);
    let census = sector.charged_heap_bytes();
    // No undercount: the census covers the arena on top of the block and
    // group tables, each sized here from its own length, not the census.
    let tables = std::mem::size_of_val(sector.blocks())
        + sector.fusion_tree_groups().len() * std::mem::size_of::<FusionTreeBlockGroup>()
        + std::mem::size_of_val(sector.sorted_indices());
    assert!(
        census >= tables + arena.heap_bytes(),
        "{census} < {tables} + arena {}",
        arena.heap_bytes()
    );
    // Ledger relation, not a byte budget: dropping the structure removes its
    // census and adds back exactly the one arena it charged.
    assert_eq!(
        owned as usize + arena.heap_bytes(),
        alone as usize + census,
        "the layout keys' arena is charged twice"
    );
}
