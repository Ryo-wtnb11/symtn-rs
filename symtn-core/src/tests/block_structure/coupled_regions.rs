use super::*;

#[test]
fn concurrent_equal_hom_spaces_share_semantics_without_global_interning() {
    let ids = std::thread::scope(|scope| {
        (0..8)
            .map(|_| {
                scope.spawn(|| {
                    FusionTreeHomSpace::new(
                        FusionProductSpace::new([u1_leg(41, 7, false)]),
                        FusionProductSpace::new([u1_leg(41, 9, true)]),
                    )
                    .id()
                })
            })
            .map(|thread| thread.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert!(ids.windows(2).all(|pair| pair[0] == pair[1]));
    assert!(ids
        .windows(2)
        .all(|pair| !Arc::ptr_eq(&pair[0].content, &pair[1].content)));
}

#[test]
fn coupled_sector_regions_describe_canonical_matrix_spans() {
    // What: canonical coupled storage compiles to exact sector ranges and tree extents.
    let rule = ZNFusionRule::new(2).unwrap();
    let leg = || SectorLeg::new([(z2_even(), 2), (z2_odd(), 2)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let keys = homspace.fusion_tree_keys(&rule);
    let structure = homspace
        .coupled_subblock_structure(&rule, 2, vec![vec![2; 4]; keys.len()])
        .unwrap();

    let cloned_before_query = structure.as_ref().clone();
    assert!(!structure.coupled_region_cache_is_initialized());
    let cold_charge = structure.charged_retained_bytes();
    assert!(!structure.coupled_region_cache_is_initialized());
    let regions = cloned_before_query
        .coupled_sector_regions(2)
        .unwrap()
        .unwrap();
    assert!(structure.coupled_region_cache_is_initialized());
    let original_regions = structure.coupled_sector_regions(2).unwrap().unwrap();
    assert!(Arc::ptr_eq(&regions, &original_regions));

    assert_eq!(regions.len(), 2);
    assert_eq!(regions[0].range().start, 0);
    assert_eq!(
        regions[0].range().len(),
        regions[0].rows() * regions[0].cols()
    );
    assert_eq!(regions[1].range().start, regions[0].range().end);
    assert_eq!(regions[1].range().end, structure.required_len().unwrap());
    assert!(regions.iter().all(|region| {
        !region.row_trees().is_empty()
            && !region.col_trees().is_empty()
            && region.has_aligned_diagonal()
            && region
                .row_trees()
                .iter()
                .all(|tree| tree.extent().unwrap() > 0)
    }));

    assert_eq!(structure.charged_retained_bytes(), cold_charge);
    for nout in 0..=structure.rank() {
        let _ = structure.coupled_sector_regions(nout);
    }
    assert_eq!(structure.charged_retained_bytes(), cold_charge);
}

#[test]
fn coupled_sector_regions_preserve_literal_expert_tree_order_and_extents() {
    // What: rank-five expert metadata preserves independent first-seen row and
    // column order while compiling a rectangular, nonuniform coupled matrix.
    let row_a = FusionTreeKey::try_from_sector_ids([9, 1, 4], 7, [false, true, false], [5], [1, 1])
        .unwrap();
    let row_b = FusionTreeKey::try_from_sector_ids([2, 8, 3], 7, [true, false, false], [6], [1, 1])
        .unwrap();
    let row_c = FusionTreeKey::try_from_sector_ids([6, 0, 5], 7, [false, false, true], [4], [1, 1])
        .unwrap();
    let col_y = FusionTreeKey::try_from_sector_ids([8, 1], 7, [true, false], [], [1]).unwrap();
    let col_x = FusionTreeKey::try_from_sector_ids([1, 3], 7, [false, true], [], [1]).unwrap();

    let block = |row: &FusionTreeKey,
                 col: &FusionTreeKey,
                 shape: [usize; 5],
                 strides: [usize; 5],
                 offset| {
        BlockSpec::with_key(
            BlockKey::FusionTree(FusionTreePairKey::pair(row.clone(), col.clone())),
            shape.to_vec(),
            strides.to_vec(),
            offset,
        )
        .unwrap()
    };
    let structure = BlockStructure::from_blocks(vec![
        block(&row_a, &col_y, [2, 1, 1, 2, 1], [1, 2, 2, 9, 18], 0),
        block(&row_b, &col_y, [1, 3, 1, 2, 1], [1, 1, 3, 9, 18], 2),
        block(&row_c, &col_y, [1, 1, 4, 2, 1], [1, 1, 1, 9, 18], 5),
        block(&row_a, &col_x, [2, 1, 1, 1, 3], [1, 2, 2, 9, 9], 18),
        block(&row_b, &col_x, [1, 3, 1, 1, 3], [1, 1, 3, 9, 9], 20),
        block(&row_c, &col_x, [1, 1, 4, 1, 3], [1, 1, 1, 9, 9], 23),
    ])
    .unwrap();

    let regions = structure.coupled_sector_regions(3).unwrap().unwrap();
    let warm_regions = structure.coupled_sector_regions(3).unwrap().unwrap();
    assert!(Arc::ptr_eq(&regions, &warm_regions));
    assert_eq!(regions.len(), 1);
    assert_eq!(regions[0].coupled(), SectorId::new(7));
    assert_eq!(regions[0].rows(), 9);
    assert_eq!(regions[0].cols(), 5);
    assert_eq!(regions[0].range(), 0..45);
    assert!(!regions[0].has_aligned_diagonal());
    assert_eq!(
        regions[0]
            .row_trees()
            .iter()
            .map(|extent| (extent.tree(), extent.offset(), extent.shape()))
            .collect::<Vec<_>>(),
        vec![
            (&row_a, 0, [2, 1, 1].as_slice()),
            (&row_b, 2, [1, 3, 1].as_slice()),
            (&row_c, 5, [1, 1, 4].as_slice()),
        ]
    );
    assert_eq!(
        regions[0]
            .col_trees()
            .iter()
            .map(|extent| (extent.tree(), extent.offset(), extent.shape()))
            .collect::<Vec<_>>(),
        vec![
            (&col_y, 0, [2, 1].as_slice()),
            (&col_x, 2, [1, 3].as_slice()),
        ]
    );
}

#[test]
fn coupled_sector_regions_require_source_ordered_tree_diagonals() {
    // What: matching row/column extents alone do not prove that walking
    // them preserves the source diagonal-block encounter order.
    let tree = |label| FusionTreeKey::try_from_sector_ids([label], 7, [false], [], []).unwrap();
    let trees = [tree(0), tree(1), tree(2)];
    let block = |row: usize, col: usize| {
        BlockSpec::with_key(
            BlockKey::FusionTree(FusionTreePairKey::pair(
                trees[row].clone(),
                trees[col].clone(),
            )),
            vec![1, 1],
            vec![1, 3],
            row + 3 * col,
        )
        .unwrap()
    };
    let ordered = BlockStructure::from_blocks(
        [
            (0, 0),
            (1, 0),
            (0, 1),
            (1, 1),
            (2, 0),
            (0, 2),
            (2, 1),
            (1, 2),
            (2, 2),
        ]
        .into_iter()
        .map(|(row, col)| block(row, col))
        .collect(),
    )
    .unwrap();
    assert!(ordered.coupled_sector_regions(1).unwrap().unwrap()[0].has_aligned_diagonal());

    let reordered = BlockStructure::from_blocks(
        [
            (0, 0),
            (1, 0),
            (0, 1),
            (2, 0),
            (0, 2),
            (2, 2),
            (1, 1),
            (1, 2),
            (2, 1),
        ]
        .into_iter()
        .map(|(row, col)| block(row, col))
        .collect(),
    )
    .unwrap();
    let regions = reordered.coupled_sector_regions(1).unwrap().unwrap();
    assert!(!regions[0].has_aligned_diagonal());
}

#[test]
fn coupled_sector_regions_preserve_empty_scalar_and_zero_extents() {
    let empty = BlockStructure::from_blocks_with_rank(3, vec![]).unwrap();
    assert_eq!(empty.coupled_sector_regions(2).unwrap().unwrap().len(), 0);

    let scalar_tree = FusionTreeKey::try_from_sector_ids([], 0, [], [], []).unwrap();
    let scalar = BlockStructure::from_blocks(vec![BlockSpec::with_key(
        BlockKey::FusionTree(FusionTreePairKey::pair(scalar_tree.clone(), scalar_tree)),
        vec![],
        vec![],
        0,
    )
    .unwrap()])
    .unwrap();
    let scalar_regions = scalar.coupled_sector_regions(0).unwrap().unwrap();
    assert_eq!(
        (
            scalar_regions[0].rows(),
            scalar_regions[0].cols(),
            scalar_regions[0].range(),
            scalar_regions[0].row_trees()[0].shape(),
            scalar_regions[0].col_trees()[0].shape(),
        ),
        (1, 1, 0..1, [].as_slice(), [].as_slice())
    );

    let row = FusionTreeKey::try_from_sector_ids([3, 4], 2, [false; 2], [], [1]).unwrap();
    let col = FusionTreeKey::try_from_sector_ids([5], 2, [true], [], []).unwrap();
    let zero = BlockStructure::from_blocks(vec![BlockSpec::with_key(
        BlockKey::FusionTree(FusionTreePairKey::pair(row, col)),
        vec![0, usize::MAX, usize::MAX],
        vec![1, 0, 0],
        0,
    )
    .unwrap()])
    .unwrap();
    let zero_regions = zero.coupled_sector_regions(2).unwrap().unwrap();
    assert_eq!(
        (
            zero_regions[0].rows(),
            zero_regions[0].cols(),
            zero_regions[0].range(),
            zero_regions[0].row_trees()[0].shape(),
            zero_regions[0].col_trees()[0].shape(),
        ),
        (
            0,
            usize::MAX,
            0..0,
            [0, usize::MAX].as_slice(),
            [usize::MAX].as_slice()
        )
    );
}

#[test]
fn coupled_sector_regions_reject_shape_mismatch_before_irrelevant_overflow() {
    // What: a repeated row tree must keep its exact shape, and that `None`
    // decision precedes computing the new column tree's overflowing extent.
    let row = FusionTreeKey::try_from_sector_ids([3, 4], 2, [false; 2], [], [1]).unwrap();
    let first_col = FusionTreeKey::try_from_sector_ids([5, 6], 2, [false; 2], [], [1]).unwrap();
    let overflowing_col =
        FusionTreeKey::try_from_sector_ids([7, 8], 2, [false; 2], [], [1]).unwrap();
    let structure = BlockStructure::from_blocks(vec![
        BlockSpec::with_key(
            BlockKey::FusionTree(FusionTreePairKey::pair(row.clone(), first_col)),
            vec![2, 3, 1, 1],
            vec![0; 4],
            0,
        )
        .unwrap(),
        BlockSpec::with_key(
            BlockKey::FusionTree(FusionTreePairKey::pair(row, overflowing_col)),
            vec![1, 6, usize::MAX, 2],
            vec![0; 4],
            0,
        )
        .unwrap(),
    ])
    .unwrap();

    assert_eq!(structure.coupled_sector_regions(2), Ok(None));
}

#[test]
fn coupled_sector_regions_reject_noncanonical_and_incomplete_grids() {
    // What: independently packed subblocks and a missing tree pair cannot claim direct spans.
    let rule = ZNFusionRule::new(2).unwrap();
    let leg = || SectorLeg::new([(z2_even(), 1), (z2_odd(), 1)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let keys = homspace.fusion_tree_keys(&rule);
    let mut offset = 0usize;
    let independently_packed = keys
        .iter()
        .map(|key| {
            let block = BlockSpec::column_major_with_key(
                BlockKey::FusionTree(key.clone()),
                vec![1; 4],
                offset,
            )
            .unwrap();
            offset += 1;
            block
        })
        .collect();
    let independently_packed = BlockStructure::from_blocks(independently_packed).unwrap();
    assert_eq!(
        independently_packed.coupled_sector_regions(2).unwrap(),
        None
    );

    let coupled = BlockStructure::coupled_sector_matrix_with_keys(
        &rule,
        2,
        4,
        keys.iter().cloned().map(|key| (key, vec![1; 4])).collect(),
    )
    .unwrap();
    let incomplete = BlockStructure::from_blocks(
        (0..coupled.block_count() - 1)
            .map(|index| {
                let block = coupled.block(index).unwrap();
                BlockSpec::with_key(
                    block.key().clone(),
                    block.shape().to_vec(),
                    block.strides().to_vec(),
                    block.offset(),
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap();
    assert_eq!(incomplete.coupled_sector_regions(2).unwrap(), None);
}
