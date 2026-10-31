use super::*;
use crate::kernel_adapter::observed::{KernelCall, ObservedKernels};

#[test]
fn zero_extent_profile_counts_match_serial_replay() {
    let structure = Arc::new(BlockStructure::packed_column_major(1, [vec![0]]).unwrap());
    let transform = TreeTransformStructure::compile_structures(
        &structure,
        &structure,
        &[TreeTransformBlockSpec::single(0, 0, 1.0)],
    )
    .unwrap();
    let mut serial = TreeTransformReplayProfile::default();
    let mut threaded = TreeTransformReplayProfile::default();
    for (profile, threads) in [(&mut serial, 1), (&mut threaded, 4)] {
        crate::host_pool::test_pool(threads)
            .install(|engine| {
                tree_transform_structure_with_structural_recoupling_raw_profiled(
                    engine,
                    &mut StridedHostKernelAdapter::default(),
                    &mut DefaultDenseExecutor::new(),
                    &mut TreeTransformWorkspace::default(),
                    &transform,
                    &structure,
                    &structure,
                    &mut [],
                    &[],
                    1.0,
                    0.0,
                    threads,
                    profile,
                )
            })
            .unwrap();
    }

    assert_eq!(threaded.single_blocks, serial.single_blocks);
    assert_eq!(threaded.multi_blocks, serial.multi_blocks);
    assert_eq!(threaded.packed_columns, serial.packed_columns);
    assert_eq!(threaded.scattered_columns, serial.scattered_columns);
}

#[test]
fn zero_extent_multi_profile_counts_match_serial_replay() {
    let structure = Arc::new(BlockStructure::packed_column_major(1, [vec![0], vec![0]]).unwrap());
    let transform = TreeTransformStructure::compile_structures(
        &structure,
        &structure,
        &[TreeTransformBlockSpec::multi(
            vec![0, 1],
            vec![0, 1],
            vec![1.0, 0.0, 0.0, 1.0],
        )],
    )
    .unwrap();
    let mut serial = TreeTransformReplayProfile::default();
    let mut threaded = TreeTransformReplayProfile::default();
    for (profile, threads) in [(&mut serial, 1), (&mut threaded, 4)] {
        crate::host_pool::test_pool(threads)
            .install(|engine| {
                tree_transform_structure_with_structural_recoupling_raw_profiled(
                    engine,
                    &mut StridedHostKernelAdapter::default(),
                    &mut DefaultDenseExecutor::new(),
                    &mut TreeTransformWorkspace::default(),
                    &transform,
                    &structure,
                    &structure,
                    &mut [],
                    &[],
                    1.0,
                    0.0,
                    threads,
                    profile,
                )
            })
            .unwrap();
    }

    assert_eq!(threaded.multi_blocks, serial.multi_blocks);
    assert_eq!(threaded.packed_columns, serial.packed_columns);
    assert_eq!(threaded.scattered_columns, serial.scattered_columns);
    assert_eq!(threaded.packed_columns, 2);
    assert_eq!(threaded.scattered_columns, 2);
}

#[test]
fn profiled_multi_attributes_recoupling_to_dense_gemm() {
    let mut pool = crate::host_pool::test_pool(2);
    pool.install(|engine| {
        let (dst_structure, src_structure, structure) = identity_multi_fixture();
        for threads in [1, 2] {
            let mut dst = vec![f64::NAN; 3];
            let mut profile = TreeTransformReplayProfile::default();
            tree_transform_structure_overwrite_with_structural_recoupling_raw_profiled(
                engine,
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &structure,
                &dst_structure,
                &src_structure,
                &mut dst,
                &[3.0, 4.0],
                2.0,
                &[],
                threads,
                &mut profile,
            )
            .unwrap();

            assert_eq!(dst, [6.0, 8.0, 0.0]);
            assert_eq!(profile.multi_blocks, 1);
            assert!(profile.multi_dense_matmul_call > Duration::ZERO);
            assert_eq!(profile.multi_scalar_recoupling, Duration::ZERO);
            assert_eq!(
                profile.multi_matmul_total,
                profile.multi_dense_view_setup
                    + profile.multi_dense_matmul_call
                    + profile.multi_scalar_recoupling
            );
        }

        let (mut dst, src, structure) = fixture();
        let mut profile = TreeTransformReplayProfile::default();
        tree_transform_structure_with_structural_recoupling_raw_profiled(
            engine,
            &mut StridedHostKernelAdapter::default(),
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            &structure,
            &Arc::clone(dst.structure()),
            &Arc::clone(src.structure()),
            dst.data_mut(),
            src.data(),
            1.0,
            0.0,
            2,
            &mut profile,
        )
        .unwrap();
        assert_eq!(profile.multi_blocks, 0);
        assert_eq!(profile.multi_matmul_total, Duration::ZERO);
        assert_eq!(profile.multi_dense_view_setup, Duration::ZERO);
        assert_eq!(profile.multi_dense_matmul_call, Duration::ZERO);
        assert_eq!(profile.multi_scalar_recoupling, Duration::ZERO);
    });
}

#[test]
fn profiled_overwrite_zeros_inactive_layout_without_touching_padding() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let src_structure = Arc::new(BlockStructure::packed_column_major(1, [vec![1]]).unwrap());
    let dst_structure = Arc::new(custom_structure(vec![block(0, 1, 1, 0), block(1, 1, 1, 2)]));
    let structure = TreeTransformStructure::compile_structures(
        &dst_structure,
        &src_structure,
        &[TreeTransformBlockSpec::single(0, 0, 2.0)],
    )
    .unwrap();
    let mut dst = [f64::NAN, 99.0, f64::NAN];
    let mut profile = TreeTransformReplayProfile::default();

    tree_transform_structure_overwrite_with_structural_recoupling_raw_profiled(
        engine,
        &mut StridedHostKernelAdapter::default(),
        &mut DefaultDenseExecutor::new(),
        &mut TreeTransformWorkspace::default(),
        &structure,
        &dst_structure,
        &src_structure,
        &mut dst,
        &[3.0],
        1.0,
        &[],
        2,
        &mut profile,
    )
    .unwrap();

    assert_eq!(dst, [6.0, 99.0, 0.0]);
    assert_eq!(profile.single_blocks, 1);
}

#[test]
fn profiled_replay_attributes_inactive_destination_scaling() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let (mut dst, src, structure) = fixture();
    let mut profile = TreeTransformReplayProfile::default();
    tree_transform_structure_with_structural_recoupling_raw_profiled(
        engine,
        &mut ObservedKernels::new(|call| {
            if matches!(call, KernelCall::Scale { .. }) {
                std::thread::sleep(Duration::from_millis(40));
            }
        }),
        &mut DefaultDenseExecutor::new(),
        &mut TreeTransformWorkspace::default(),
        &structure,
        &Arc::clone(dst.structure()),
        &Arc::clone(src.structure()),
        dst.data_mut(),
        src.data(),
        1.0,
        0.5,
        1,
        &mut profile,
    )
    .unwrap();
    let attributed = profile.validate
        + profile.single_total
        + profile.strided_kernel.saturating_sub(profile.single_total)
        + profile.multi_workspace_prepare
        + profile.multi_pack
        + profile.multi_coefficient_prepare
        + profile.multi_matmul_total
        + profile.multi_scatter;
    assert!(profile.total.saturating_sub(attributed) < Duration::from_millis(20));
}
