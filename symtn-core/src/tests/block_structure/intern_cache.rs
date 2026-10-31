use super::*;

fn local_parts(index: usize) -> (SectorStructure, DegeneracyStructure) {
    (
        SectorStructure::from_keys(1, vec![BlockKey::ordinal(index)]).unwrap(),
        DegeneracyStructure::packed_column_major(1, [vec![1]]).unwrap(),
    )
}

#[derive(Clone)]
struct LocalKey {
    index: usize,
    key: BlockStructureInternKey,
    rank: usize,
}

fn local_block_structure_intern_key(index: usize) -> LocalKey {
    let (sector, degeneracy) = local_parts(index);
    LocalKey {
        index,
        key: BlockStructureInternKey::of(&sector, &degeneracy),
        rank: 1,
    }
}

fn local_block_structure_intern(
    table: &mut BlockStructureInternTable,
    key: LocalKey,
    charged_key_bytes: usize,
) -> Arc<BlockStructureContent> {
    let (sector, degeneracy) = local_parts(key.index);
    if let Some(content) = table.lookup(&key.key, |content| {
        *content.sector == sector && content.degeneracy == degeneracy
    }) {
        return content;
    }
    let required_len = degeneracy.required_len().unwrap();
    table.intern_with(
        key.key,
        |_| charged_key_bytes,
        || {
            Arc::new(BlockStructureContent {
                id: BLOCK_STRUCTURE_CONTENT_ID.fetch_add(1, Ordering::Relaxed),
                sector: Arc::new(sector),
                degeneracy,
                required_len,
                storage_tiling: Default::default(),
            })
        },
    )
}

#[test]
fn block_structure_intern_hash_collision_never_aliases() {
    // What: a live content of different blocks under the same hash key
    // is neither returned for the colliding content nor replaced by it; the
    // colliding content comes back uninterned with its own id.
    let key0 = local_block_structure_intern_key(50);
    let charge = charged_block_structure_intern_key_bytes(&key0.key);
    let mut table = BlockStructureInternTable::new(4, charge.saturating_mul(4), charge);
    let original = local_block_structure_intern(&mut table, key0.clone(), charge);
    let before = table.info();

    let (sector, degeneracy) = local_parts(51);
    let matches_colliding = |content: &BlockStructureContent| {
        *content.sector == sector && content.degeneracy == degeneracy
    };
    assert!(table.lookup(&key0.key, matches_colliding).is_none());
    let required_len = degeneracy.required_len().unwrap();
    let colliding = table.intern_with(
        key0.key.clone(),
        |_| charge,
        || {
            Arc::new(BlockStructureContent {
                id: BLOCK_STRUCTURE_CONTENT_ID.fetch_add(1, Ordering::Relaxed),
                sector: Arc::new(sector.clone()),
                degeneracy: degeneracy.clone(),
                required_len,
                storage_tiling: Default::default(),
            })
        },
    );

    assert!(!Arc::ptr_eq(&colliding, &original));
    assert_ne!(colliding.id(), original.id());
    assert_eq!(colliding.sector_structure(), &sector);
    let still = table.lookup(&key0.key, |_| true).unwrap();
    assert!(Arc::ptr_eq(&still, &original));
    assert_eq!(table.info(), before);
}

#[test]
fn block_structure_intern_charge_counts_only_spilled_smallvec_storage() {
    // What: inline SmallVec storage adds no heap charge, while spilled
    // storage contributes its full heap capacity in item bytes.
    let inline: SmallVec<[u64; 2]> = smallvec::smallvec![1_u64, 2];
    let spilled: SmallVec<[u64; 2]> = smallvec::smallvec![1_u64, 2, 3];
    assert!(!inline.spilled());
    assert!(spilled.spilled());
    assert_eq!(spilled_smallvec_heap_bytes(&inline), 0);
    assert_eq!(
        spilled_smallvec_heap_bytes(&spilled),
        spilled.capacity() * std::mem::size_of::<u64>()
    );
}

#[test]
fn block_structure_intern_entry_pressure_evicts_oldest() {
    // What: entry pressure removes the oldest admitted key, and a read hit
    // does not promote it in the FIFO order.
    let key0 = local_block_structure_intern_key(0);
    let key1 = local_block_structure_intern_key(1);
    let key2 = local_block_structure_intern_key(2);
    let charge = charged_block_structure_intern_key_bytes(&key0.key);
    assert_eq!(charged_block_structure_intern_key_bytes(&key1.key), charge);
    assert_eq!(charged_block_structure_intern_key_bytes(&key2.key), charge);
    let mut table = BlockStructureInternTable::new(2, charge.saturating_mul(3), charge);

    let _content0 = local_block_structure_intern(&mut table, key0.clone(), charge);
    let _content1 = local_block_structure_intern(&mut table, key1.clone(), charge);
    assert!(table.lookup(&key0.key, |_| true).is_some());
    let _content2 = local_block_structure_intern(&mut table, key2.clone(), charge);

    assert!(table.lookup(&key0.key, |_| true).is_none());
    assert!(table.lookup(&key1.key, |_| true).is_some());
    assert!(table.lookup(&key2.key, |_| true).is_some());
    let info = table.info();
    assert_eq!(info.entries(), 2);
    assert_eq!(info.pressure_evictions(), 1);
}

#[test]
fn block_structure_intern_byte_pressure_subtracts_exact_charge() {
    // What: byte pressure subtracts the evicted entry's unequal stored
    // charge exactly before admitting the incoming key.
    let key0 = local_block_structure_intern_key(10);
    let key1 = local_block_structure_intern_key(11);
    let key2 = local_block_structure_intern_key(12);
    let base_charge = charged_block_structure_intern_key_bytes(&key0.key);
    let charges = [base_charge, base_charge + 1, base_charge + 2];
    let budget = charges[1].saturating_add(charges[2]);
    let mut table = BlockStructureInternTable::new(3, budget, charges[2]);

    let _content0 = local_block_structure_intern(&mut table, key0.clone(), charges[0]);
    let _content1 = local_block_structure_intern(&mut table, key1.clone(), charges[1]);
    assert_eq!(
        table.info().charged_key_bytes(),
        charges[0].saturating_add(charges[1])
    );
    assert_eq!(table.info().pressure_evictions(), 0);
    assert!(table.lookup(&key0.key, |_| true).is_some());

    let _content2 = local_block_structure_intern(&mut table, key2.clone(), charges[2]);
    assert!(table.lookup(&key0.key, |_| true).is_none());
    assert!(table.lookup(&key1.key, |_| true).is_some());
    assert!(table.lookup(&key2.key, |_| true).is_some());
    assert_eq!(table.info().charged_key_bytes(), budget);
    assert_eq!(table.info().pressure_evictions(), 1);
}

#[test]
fn block_structure_intern_bypasses_oversized_and_saturated_charges() {
    // What: oversized and saturated charges return complete content but
    // never consume an entry or charged-byte budget.
    let oversized_key = local_block_structure_intern_key(20);
    let saturated_key = local_block_structure_intern_key(21);
    let charge = charged_block_structure_intern_key_bytes(&oversized_key.key);
    let mut oversized_table = BlockStructureInternTable::new(2, charge, charge - 1);

    let oversized =
        local_block_structure_intern(&mut oversized_table, oversized_key.clone(), charge);
    assert_eq!(oversized.rank(), oversized_key.rank);
    assert_eq!(
        oversized.sector_structure(),
        &local_parts(oversized_key.index).0
    );
    assert!(oversized_table
        .lookup(&oversized_key.key, |_| true)
        .is_none());
    assert_eq!(oversized_table.info().oversized_admission_bypasses(), 1);

    let mut saturated_table = BlockStructureInternTable::new(2, usize::MAX, usize::MAX);
    let saturated =
        local_block_structure_intern(&mut saturated_table, saturated_key.clone(), usize::MAX);
    assert_eq!(saturated.rank(), saturated_key.rank);
    assert_eq!(
        saturated.sector_structure(),
        &local_parts(saturated_key.index).0
    );
    assert!(saturated_table
        .lookup(&saturated_key.key, |_| true)
        .is_none());

    let info = saturated_table.info();
    assert_eq!(info.entries(), 0);
    assert_eq!(info.charged_key_bytes(), 0);
    assert_eq!(info.oversized_admission_bypasses(), 1);
}

#[test]
fn block_structure_intern_dead_replacement_preserves_fifo_accounting() {
    // What: replacing a dead Weak changes only its content epoch; entry
    // count, charge, counters, and oldest-first eviction order stay fixed.
    let key0 = local_block_structure_intern_key(30);
    let key1 = local_block_structure_intern_key(31);
    let key2 = local_block_structure_intern_key(32);
    let charge = charged_block_structure_intern_key_bytes(&key0.key);
    let mut table = BlockStructureInternTable::new(2, charge.saturating_mul(2), charge);

    let content0 = local_block_structure_intern(&mut table, key0.clone(), charge);
    let id0 = content0.id();
    let _content1 = local_block_structure_intern(&mut table, key1.clone(), charge);
    let before = table.info();
    drop(content0);
    assert!(table.lookup(&key0.key, |_| true).is_none());

    let replacement = local_block_structure_intern(&mut table, key0.clone(), charge);
    assert!(replacement.id() > id0);
    assert_eq!(table.info(), before);

    let _content2 = local_block_structure_intern(&mut table, key2.clone(), charge);
    assert!(table.lookup(&key0.key, |_| true).is_none());
    assert!(table.lookup(&key1.key, |_| true).is_some());
    assert!(table.lookup(&key2.key, |_| true).is_some());
    assert_eq!(table.info().pressure_evictions(), 1);
}

#[test]
fn block_structure_intern_clear_resets_resources_and_counters() {
    // What: clear releases every admitted key and resets byte, eviction,
    // and bypass accounting while preserving configured limits.
    let key0 = local_block_structure_intern_key(40);
    let key1 = local_block_structure_intern_key(41);
    let key2 = local_block_structure_intern_key(42);
    let charge = charged_block_structure_intern_key_bytes(&key0.key);
    let mut table = BlockStructureInternTable::new(1, charge, charge);
    let _content0 = local_block_structure_intern(&mut table, key0, charge);
    let _content1 = local_block_structure_intern(&mut table, key1, charge);
    let _content2 = local_block_structure_intern(&mut table, key2, usize::MAX);
    assert_eq!(table.info().pressure_evictions(), 1);
    assert_eq!(table.info().oversized_admission_bypasses(), 1);

    table.clear();

    let info = table.info();
    assert_eq!(info.entries(), 0);
    assert_eq!(info.entry_capacity(), 1);
    assert_eq!(info.charged_key_bytes(), 0);
    assert_eq!(info.byte_budget(), charge);
    assert_eq!(info.max_admitted_entry_bytes(), charge);
    assert_eq!(info.pressure_evictions(), 0);
    assert_eq!(info.oversized_admission_bypasses(), 0);
}

fn coupled_z2_matrix_structure() -> BlockStructure {
    let rule = Z2FusionRule;
    let leg = || SectorLeg::new([(z2_even(), 2), (z2_odd(), 3)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg()]),
        FusionProductSpace::new([leg()]),
    );
    let blocks = homspace
        .fusion_tree_keys(&rule)
        .iter()
        .cloned()
        .map(|key| (key, vec![2, 3]))
        .collect();
    BlockStructure::coupled_sector_matrix_with_keys(&rule, 1, 2, blocks).unwrap()
}

#[test]
fn frozen_content_outlives_wrapper_local_coupled_regions() {
    // What: equal live wrappers canonicalized through the weak table share
    // coupled-region state, while retained immutable content cannot retain it.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_core_intern_tables();

    let first = coupled_z2_matrix_structure().into_shared();
    let second = coupled_z2_matrix_structure().into_shared();
    assert!(Arc::ptr_eq(&first, &second));
    let first_content = first.content_key();
    let second_content = second.content_key();
    assert!(Arc::ptr_eq(&first_content, &second_content));

    let first_regions = first.coupled_sector_regions(1).unwrap().unwrap();
    let second_regions = second.coupled_sector_regions(1).unwrap().unwrap();
    assert!(Arc::ptr_eq(&first_regions, &second_regions));

    let expected_sector = first.sector_structure().clone();
    let expected_degeneracy = first.degeneracy_structure().clone();
    let expected_len = first.required_len().unwrap();
    let expected_regions = first_regions.as_ref().to_vec();
    let expected_blocks = (0..first.block_count())
        .map(|index| {
            let block = first.block(index).unwrap();
            (
                block.key().clone(),
                block.shape().to_vec(),
                block.strides().to_vec(),
                block.offset(),
            )
        })
        .collect::<Vec<_>>();
    let weak_regions = first.weak_region_state();

    drop(first_regions);
    drop(second_regions);
    drop(second_content);
    drop(first);
    drop(second);
    assert!(weak_regions.upgrade().is_none());

    let rebuilt = coupled_z2_matrix_structure();
    assert_eq!(rebuilt.sector_structure(), &expected_sector);
    assert_eq!(rebuilt.degeneracy_structure(), &expected_degeneracy);
    assert_eq!(rebuilt.required_len().unwrap(), expected_len);
    assert_eq!(
        (0..rebuilt.block_count())
            .map(|index| {
                let block = rebuilt.block(index).unwrap();
                (
                    block.key().clone(),
                    block.shape().to_vec(),
                    block.strides().to_vec(),
                    block.offset(),
                )
            })
            .collect::<Vec<_>>(),
        expected_blocks
    );
    assert_eq!(
        rebuilt.coupled_sector_regions(1).unwrap().unwrap().as_ref(),
        expected_regions
    );
    drop(first_content);
}

#[test]
fn concurrent_equal_live_content_canonicalizes_once() {
    // What: concurrent equal construction shares one content Arc and id
    // while every returned structure remains live.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    reset_core_intern_tables();
    let barrier = std::sync::Barrier::new(8);
    let structures = std::thread::scope(|scope| {
        let barrier = &barrier;
        let threads = (0..8)
            .map(|_| {
                scope.spawn(move || {
                    barrier.wait();
                    BlockStructure::trivial(&[17, 19]).unwrap()
                })
            })
            .collect::<Vec<_>>();
        threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect::<Vec<_>>()
    });
    let content = structures[0].content_key();
    let id = content.id();
    for structure in &structures[1..] {
        let candidate = structure.content_key();
        assert!(Arc::ptr_eq(&content, &candidate));
        assert_eq!(candidate.id(), id);
    }
}

#[test]
fn reset_core_intern_tables_clears_without_reusing_ids() {
    // What: reset preserves a surviving structure and its published region
    // while equal content rebuilt afterward receives a fresh monotonic id.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let before = coupled_z2_matrix_structure();
    let id_before = before.content_id();
    let regions_before = before.coupled_sector_regions(1).unwrap().unwrap();

    reset_core_intern_tables();

    let surviving_regions = before.coupled_sector_regions(1).unwrap().unwrap();
    assert!(Arc::ptr_eq(&regions_before, &surviving_regions));
    assert_eq!(before.content_id(), id_before);

    let after = coupled_z2_matrix_structure();
    let id_after = after.content_id();
    assert!(
        id_after > id_before,
        "reset must not reuse content ids, got before={id_before} after={id_after}"
    );
}
