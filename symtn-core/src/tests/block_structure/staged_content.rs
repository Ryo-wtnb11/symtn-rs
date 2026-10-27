use super::*;

fn prepared_with_backing() -> (PreparedBlockStructure, usize) {
    let degeneracy =
        DegeneracyStructure::packed_column_major(3, (0..4).map(|_| vec![2, 3, 4])).unwrap();
    let backing = degeneracy.blocks().next().unwrap().shape().as_ptr() as usize;
    let prepared = PreparedBlockStructure::from_parts(
        SectorStructure::from_keys(3, (0..4).map(BlockKey::ordinal)).unwrap(),
        degeneracy,
    )
    .unwrap();
    (prepared, backing)
}

fn prepared() -> PreparedBlockStructure {
    prepared_with_backing().0
}

#[test]
fn preview_commit_reuses_one_unpublished_content() {
    let (staged, backing) = prepared_with_backing();
    assert_eq!(
        staged.structure().block(0).unwrap().shape().as_ptr() as usize,
        backing
    );
    let preview = staged.structure().content_key();
    let committed = staged.commit();
    assert!(Arc::ptr_eq(&preview, &committed.content_key()));
}

#[test]
fn concurrent_preview_initialization_moves_backing_once() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<PreparedBlockStructure>();
    let (staged, backing) = prepared_with_backing();
    let staged = Arc::new(staged);
    let threads = (0..4)
        .map(|_| {
            let staged = Arc::clone(&staged);
            std::thread::spawn(move || {
                assert_eq!(
                    staged.structure().block(0).unwrap().shape().as_ptr() as usize,
                    backing
                );
                staged.structure().content_key()
            })
        })
        .collect::<Vec<_>>();
    let contents = threads
        .into_iter()
        .map(|thread| thread.join().unwrap())
        .collect::<Vec<_>>();
    assert!(contents
        .iter()
        .all(|content| Arc::ptr_eq(content, &contents[0])));
}

#[test]
fn equal_uncached_structures_are_semantic_but_keep_monotonic_ids() {
    // The clear below must not land inside a sibling's locked intern-table window (#2084).
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let first = prepared().commit();
    clear_structure_caches();
    let second = prepared().commit();
    assert_eq!(first, second);
    assert_ne!(first.content_id(), second.content_id());
    let mut first_hash = rustc_hash::FxHasher::default();
    first.content_key().hash(&mut first_hash);
    let mut second_hash = rustc_hash::FxHasher::default();
    second.content_key().hash(&mut second_hash);
    assert_eq!(
        std::hash::Hasher::finish(&first_hash),
        std::hash::Hasher::finish(&second_hash)
    );
}

#[cfg(feature = "cgc-gen")]
#[test]
fn uncached_generic_preview_keeps_its_coupled_region_geometry() {
    if test_support::run_isolated_or_return(
        "SYMTN_GENERIC_PREVIEW_COLD", "tests::block_structure::staged_content::uncached_generic_preview_keeps_its_coupled_region_geometry",
    ) { return; }
    clear_structure_caches();
    let rule = symtn_sectors::testing::su_lie_rule(3);
    let adjoint = rule.encode_irrep(&[1, 1], &[]).unwrap();
    let trivial = rule.encode_irrep(&[0, 0], &[]).unwrap();
    let leg = |dual| SectorLeg::new([(adjoint, 2), (trivial, 3)], dual);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(false), leg(true)]),
        FusionProductSpace::new([leg(false), leg(true)]),
    );
    let staged = hom
        .prepare_coupled_subblock_structure_from_leg_degeneracies_generic_checked(&rule)
        .unwrap();
    let preview = staged.structure().clone();
    let regions = preview.coupled_sector_regions(2).unwrap().unwrap();
    let committed = staged.commit();
    assert!(Arc::ptr_eq(
        &preview.content_key(),
        &committed.content_key()
    ));
    assert!(Arc::ptr_eq(
        &regions,
        &committed.coupled_sector_regions(2).unwrap().unwrap()
    ));
    assert!(committed.storage_tiling_proven());
}
