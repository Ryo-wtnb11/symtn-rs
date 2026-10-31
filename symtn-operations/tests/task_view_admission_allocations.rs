use std::sync::Arc;

use symtn_core::BlockStructure;
use symtn_operations::{
    DenseTreeTransformOperations, TreeTransformBackend, TreeTransformReplayProfile,
    TreeTransformStructure, TreeTransformWorkspace,
};

use crate::counting_alloc;

#[test]
fn warmed_public_host_admission_without_tasks_is_allocation_free() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // An empty completed transform takes the normal profiled Host admission
    // seam but has no scale, block, coefficient, or dense numerical work.
    let structure =
        Arc::new(BlockStructure::packed_column_major(1, std::iter::empty::<Vec<usize>>()).unwrap());
    let transform =
        TreeTransformStructure::<f64>::compile_structures(&structure, &structure, &[]).unwrap();
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();
    let mut destination = [];
    let source = [];

    let mut replay = |profile: &mut TreeTransformReplayProfile| {
        backend
            .tree_transform_structure_overwrite_into_raw_profiled(
                engine,
                &mut workspace,
                &transform,
                &structure,
                &structure,
                &mut destination,
                &source,
                1.0,
                &[],
                profile,
            )
            .unwrap();
    };
    replay(&mut TreeTransformReplayProfile::default());
    counting_alloc::start();
    let mut profile = TreeTransformReplayProfile::default();
    for _ in 0..100 {
        replay(&mut profile);
    }
    let allocs = counting_alloc::stop();

    assert_eq!(allocs.calls, 0);
    assert_eq!(profile.single_blocks, 0);
    assert_eq!(profile.multi_blocks, 0);
    assert_eq!(profile.multi_coefficient_prepare, std::time::Duration::ZERO);
}
