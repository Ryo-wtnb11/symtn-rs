use super::*;

#[test]
fn compile_rejects_inactive_destination_aliases() {
    let src_structure = BlockStructure::packed_column_major(1, [vec![1]]).unwrap();
    let dst_structure = custom_structure(vec![block(0, 1, 1, 0), block(1, 1, 1, 0)]);
    assert_eq!(
        TreeTransformStructure::<f64>::compile_structures(&dst_structure, &src_structure, &[],)
            .unwrap_err(),
        OperationError::InvalidArgument {
            message: "tree transform destination layouts overlap"
        }
    );
}

#[test]
fn compile_rejects_active_inactive_destination_aliases() {
    let src_structure = BlockStructure::packed_column_major(1, [vec![1]]).unwrap();
    let dst_structure = custom_structure(vec![block(0, 1, 1, 0), block(1, 1, 1, 0)]);
    assert_eq!(
        TreeTransformStructure::compile_structures(
            &dst_structure,
            &src_structure,
            &[TreeTransformBlockSpec::single(0, 0, 1.0)],
        )
        .unwrap_err(),
        OperationError::InvalidArgument {
            message: "tree transform destination layouts overlap"
        }
    );
}

#[test]
fn compile_rejects_active_destination_aliases() {
    let src_structure = BlockStructure::packed_column_major(1, [vec![1], vec![1]]).unwrap();
    let dst_structure = custom_structure(vec![block(0, 1, 1, 0), block(1, 1, 1, 0)]);
    assert_eq!(
        TreeTransformStructure::compile_structures(
            &dst_structure,
            &src_structure,
            &[
                TreeTransformBlockSpec::single(0, 0, 1.0),
                TreeTransformBlockSpec::single(1, 1, 1.0),
            ],
        )
        .unwrap_err(),
        OperationError::InvalidArgument {
            message: "tree transform destination layouts overlap"
        }
    );
}

#[test]
fn compile_rejects_self_overlapping_destination_layout() {
    let src_structure = BlockStructure::packed_column_major(1, [vec![2]]).unwrap();
    let dst_structure = custom_structure(vec![block(0, 2, 0, 0)]);
    assert_eq!(
        TreeTransformStructure::compile_structures(
            &dst_structure,
            &src_structure,
            &[TreeTransformBlockSpec::single(0, 0, 1.0)],
        )
        .unwrap_err(),
        OperationError::InvalidArgument {
            message: "tree transform destination layouts overlap"
        }
    );
}

#[test]
fn compile_rejects_nonzero_stride_self_overlap() {
    let src_structure = BlockStructure::packed_column_major(2, [vec![2, 2]]).unwrap();
    let dst_structure = BlockStructure::from_blocks_with_rank(
        2,
        vec![BlockSpec::with_key(BlockKey::opaque([0]), vec![2, 2], vec![1, 1], 0).unwrap()],
    )
    .unwrap();
    assert_eq!(
        TreeTransformStructure::compile_structures(
            &dst_structure,
            &src_structure,
            &[TreeTransformBlockSpec::single(0, 0, 1.0)],
        )
        .unwrap_err(),
        OperationError::InvalidArgument {
            message: "tree transform destination layouts overlap",
        }
    );
}

#[test]
fn interleaved_disjoint_destinations_with_overlapping_ranges_are_valid() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let src_structure = BlockStructure::packed_column_major(1, [vec![1]]).unwrap();
    let dst_structure = custom_structure(vec![block(0, 2, 2, 0), block(1, 2, 2, 1)]);
    let structure =
        TreeTransformStructure::<f64>::compile_structures(&dst_structure, &src_structure, &[])
            .unwrap();
    let mut dst = vec![10.0, 20.0, 30.0, 40.0];
    tree_transform_structure_with_structural_recoupling_raw(
        engine,
        &mut StridedHostKernelAdapter::default(),
        &mut DefaultDenseExecutor::new(),
        &mut TreeTransformWorkspace::default(),
        &structure,
        &Arc::new(dst_structure),
        &Arc::new(src_structure),
        &mut dst,
        &[3.0],
        1.0,
        0.5,
        4,
    )
    .unwrap();
    assert_eq!(dst, [5.0, 10.0, 15.0, 20.0]);
}

#[test]
fn many_range_connected_interleaved_destinations_are_valid() {
    let src_structure = BlockStructure::packed_column_major(1, [vec![1]]).unwrap();
    let dst_structure =
        custom_structure((0..64).map(|offset| block(offset, 2, 64, offset)).collect());
    TreeTransformStructure::<f64>::compile_structures(&dst_structure, &src_structure, &[]).unwrap();
}
