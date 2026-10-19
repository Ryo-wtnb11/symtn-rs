use super::*;

#[test]
fn block_view_validates_column_major_layout() {
    let data = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0];
    let shape = [2, 3];
    let strides = [1, 2];
    let view = BlockView::new(&data, &shape, &strides, 1).unwrap();
    assert_eq!(view.shape(), &[2, 3]);
    assert_eq!(view.strides(), &[1, 2]);
    assert_eq!(view.get(&[0, 0]), Some(&11.0));
    assert_eq!(view.get(&[1, 2]), Some(&16.0));
    assert_eq!(view.get(&[0]), None);
    assert_eq!(view.get(&[2, 0]), None);
    assert_eq!(view.get(&[0, 3]), None);
}

#[test]
fn block_view_rejects_out_of_bounds_layout() {
    let data = [0.0; 6];
    let shape = [2, 3];
    let strides = [1, 4];
    let err = BlockView::new(&data, &shape, &strides, 0).unwrap_err();
    assert_eq!(err, CoreError::OutOfBounds);
}

#[test]
fn trivial_structure_is_one_column_major_block() {
    let structure = BlockStructure::trivial(&[2, 3, 4]).unwrap();

    assert_eq!(structure.required_len().unwrap(), 24);
    assert_eq!(structure.block_count(), 1);
    let block = structure.only_block().unwrap();
    assert_eq!(block.key(), &BlockKey::trivial());
    assert_eq!(block.shape(), &[2, 3, 4]);
    assert_eq!(block.strides(), &[1, 2, 6]);
    assert_eq!(block.offset(), 0);
}

#[test]
fn fusion_tree_block_view_mut_updates_selected_storage() {
    let key =
        FusionTreePairKey::try_pair_from_sector_ids([1], [1], 1, [false], [true], [], [], [], [])
            .unwrap();
    let other =
        FusionTreePairKey::try_pair_from_sector_ids([0], [0], 0, [false], [true], [], [], [], [])
            .unwrap();
    let structure = packed_fixture_structure(
        2,
        [
            (BlockKey::from(other), vec![1, 2]),
            (BlockKey::from(key.clone()), vec![2, 1]),
        ],
    )
    .unwrap();
    let mut data = vec![1, 2, 3, 4];

    {
        let block = structure.fusion_tree_pair_block(&key).unwrap();
        let mut view =
            BlockViewMut::new(&mut data, block.shape(), block.strides(), block.offset()).unwrap();
        let offset = view.offset();
        view.data_mut()[offset] = 30;
        view.data_mut()[offset + 1] = 40;
    }

    assert_eq!(data, [1, 2, 30, 40]);
}

#[test]
fn packed_block_structure_places_blocks_contiguously() {
    let structure = BlockStructure::packed_column_major(2, [vec![2, 3], vec![1, 4]]).unwrap();

    assert_eq!(structure.required_len().unwrap(), 10);
    assert_eq!(structure.rank(), 2);
    let first = structure.block(0).unwrap();
    assert_eq!(first.shape(), &[2, 3]);
    assert_eq!(first.offset(), 0);
    let second = structure.block(1).unwrap();
    assert_eq!(second.shape(), &[1, 4]);
    assert_eq!(second.offset(), 6);
}
