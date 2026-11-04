use std::sync::Arc;

use symtn_core::BlockStructure;
use symtn_dense::DefaultDenseExecutor;
use symtn_operations::{
    tree_transform_structure_overwrite_with_structural_recoupling_raw, StridedHostKernelAdapter,
    TreeTransformBlockSpec, TreeTransformStructure, TreeTransformWorkspace,
};

use crate::counting_alloc;

#[test]
fn a_reduced_grant_workspace_releases_no_more_than_it_reports() {
    let _serial = counting_alloc::serial();
    // What: four transposing Singles on eight free-dispatch workers each
    // take part of the token, so the lane placement weighs reduced split
    // widths; once warm, dropping the workspace frees no more bytes than
    // its `retained_bytes` charges, and the values are the transpose.
    let (rows, cols) = (4096, 4);
    let dst = Arc::new(BlockStructure::packed_column_major(2, vec![vec![rows, cols]; 4]).unwrap());
    let src = Arc::new(BlockStructure::packed_column_major(2, vec![vec![cols, rows]; 4]).unwrap());
    let transform = TreeTransformStructure::compile_structures(
        &dst,
        &src,
        &(0..4)
            .map(|block| TreeTransformBlockSpec::single(block, block, 1.0).with_source_axes([1, 0]))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let block = rows * cols;
    let source = (0..4 * block).map(|i| (i % 251) as f64).collect::<Vec<_>>();
    let mut expected = vec![0.0; 4 * block];
    for b in 0..4 {
        for c in 0..cols {
            for r in 0..rows {
                expected[b * block + r + rows * c] = source[b * block + c + cols * r];
            }
        }
    }
    let mut engine = symtn_operations::host_pool::eager_test_engine(8);
    let mut dense = DefaultDenseExecutor::new();
    let mut workspace = TreeTransformWorkspace::default();
    for _ in 0..2 {
        let mut destination = vec![f64::NAN; 4 * block];
        tree_transform_structure_overwrite_with_structural_recoupling_raw(
            &mut engine,
            &mut StridedHostKernelAdapter::default(),
            &mut dense,
            &mut workspace,
            &transform,
            &dst,
            &src,
            &mut destination,
            &source,
            1.0,
            &[],
            8,
        )
        .unwrap();
        assert!(destination == expected);
    }
    let charged = workspace.retained_bytes() as u64;
    let ((), released) = counting_alloc::measure(|| drop(workspace));
    assert!(
        released.freed_bytes <= charged,
        "released {} bytes, charged {charged}",
        released.freed_bytes
    );
}
