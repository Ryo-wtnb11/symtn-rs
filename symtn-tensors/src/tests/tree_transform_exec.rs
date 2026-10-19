use super::*;

#[test]
fn tree_transform_structure_replay_applies_alpha_and_beta() {
    let space = vec![2, 2];
    let structure = BlockStructure::packed_column_major(2, [vec![2, 2]]).unwrap();
    let src = TestTensor::from_vec_with_structure(
        vec![1.0_f64, 2.0, 3.0, 4.0],
        space.clone(),
        structure.clone(),
    )
    .unwrap();
    let mut dst = TestTensor::from_vec_with_structure(vec![10.0_f64; 4], space, structure).unwrap();
    let transform = TreeTransformStructure::compile_structures(
        dst.structure(),
        src.structure(),
        &[TreeTransformBlockSpec::single(0, 0, 3.0)],
    )
    .unwrap();
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();

    replay_into(
        &mut backend,
        &mut workspace,
        &transform,
        &mut dst,
        &src,
        2.0,
        4.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[46.0, 52.0, 58.0, 64.0]);
}

#[test]
fn tree_transform_overwrite_replay_ignores_nan_destination_bits() {
    let space = vec![2, 2];
    let structure = BlockStructure::packed_column_major(2, [vec![2, 2]]).unwrap();
    let src = TestTensor::from_vec_with_structure(
        vec![1.0_f64, 2.0, 3.0, 4.0],
        space.clone(),
        structure.clone(),
    )
    .unwrap();
    let mut dst = TestTensor::from_vec_with_structure(vec![f64::NAN; 4], space, structure).unwrap();
    let transform = TreeTransformStructure::compile_structures(
        dst.structure(),
        src.structure(),
        &[TreeTransformBlockSpec::single(0, 0, 3.0)],
    )
    .unwrap();
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();

    replay_overwrite_into(
        &mut backend,
        &mut workspace,
        &transform,
        &mut dst,
        &src,
        2.0,
    )
    .unwrap();

    assert_eq!(dst.data(), &[6.0, 12.0, 18.0, 24.0]);
}

#[test]
fn tree_transform_single_replay_supports_all_numeric_dtypes() {
    assert_tree_single_dtype(
        vec![1.0_f32, 2.0, 3.0, 4.0],
        10.0,
        3.0,
        2.0,
        4.0,
        vec![46.0, 52.0, 58.0, 64.0],
    );
    assert_tree_single_dtype(
        vec![1.0_f64, 2.0, 3.0, 4.0],
        10.0,
        3.0,
        2.0,
        4.0,
        vec![46.0, 52.0, 58.0, 64.0],
    );
    assert_tree_single_dtype(vec![1_i32, 2, 3, 4], 10, 3, 2, 4, vec![46, 52, 58, 64]);
    assert_tree_single_dtype(vec![1_i64, 2, 3, 4], 10, 3, 2, 4, vec![46, 52, 58, 64]);
    assert_tree_single_dtype(
        vec![
            Complex32::new(1.0, 1.0),
            Complex32::new(2.0, -1.0),
            Complex32::new(3.0, 0.5),
            Complex32::new(4.0, -0.5),
        ],
        Complex32::new(10.0, 1.0),
        Complex32::new(3.0, 0.0),
        Complex32::new(2.0, 0.0),
        Complex32::new(4.0, 0.0),
        vec![
            Complex32::new(46.0, 10.0),
            Complex32::new(52.0, -2.0),
            Complex32::new(58.0, 7.0),
            Complex32::new(64.0, 1.0),
        ],
    );
    assert_tree_single_dtype(
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(3.0, 0.5),
            Complex64::new(4.0, -0.5),
        ],
        Complex64::new(10.0, 1.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(4.0, 0.0),
        vec![
            Complex64::new(46.0, 10.0),
            Complex64::new(52.0, -2.0),
            Complex64::new(58.0, 7.0),
            Complex64::new(64.0, 1.0),
        ],
    );
}

#[test]
fn tree_transform_single_replay_supports_complex_data_with_real_coefficients() {
    assert_tree_single_mixed_dtype(
        vec![
            Complex32::new(1.0, 1.0),
            Complex32::new(2.0, -1.0),
            Complex32::new(3.0, 0.5),
            Complex32::new(4.0, -0.5),
        ],
        Complex32::new(10.0, 1.0),
        3.0_f64,
        Complex32::new(2.0, 0.0),
        Complex32::new(4.0, 0.0),
        vec![
            Complex32::new(46.0, 10.0),
            Complex32::new(52.0, -2.0),
            Complex32::new(58.0, 7.0),
            Complex32::new(64.0, 1.0),
        ],
    );
    assert_tree_single_mixed_dtype(
        vec![
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(3.0, 0.5),
            Complex64::new(4.0, -0.5),
        ],
        Complex64::new(10.0, 1.0),
        3.0_f64,
        Complex64::new(2.0, 0.0),
        Complex64::new(4.0, 0.0),
        vec![
            Complex64::new(46.0, 10.0),
            Complex64::new(52.0, -2.0),
            Complex64::new(58.0, 7.0),
            Complex64::new(64.0, 1.0),
        ],
    );
}

#[test]
fn tree_transform_multi_pack_gemm_scatter_supports_all_numeric_dtypes() {
    assert_tree_multi_dtype(
        vec![2.0_f32, 3.0, 5.0, 7.0],
        2.0,
        10.0,
        1.0,
        vec![44.0, 54.0, 64.0, 74.0, 90.0, 114.0, 138.0, 162.0],
    );
    assert_tree_multi_dtype(
        vec![2.0_f64, 3.0, 5.0, 7.0],
        2.0,
        10.0,
        1.0,
        vec![44.0, 54.0, 64.0, 74.0, 90.0, 114.0, 138.0, 162.0],
    );
    assert_tree_multi_dtype(
        vec![2_i32, 3, 5, 7],
        2,
        10,
        1,
        vec![44, 54, 64, 74, 90, 114, 138, 162],
    );
    assert_tree_multi_dtype(
        vec![2_i64, 3, 5, 7],
        2,
        10,
        1,
        vec![44, 54, 64, 74, 90, 114, 138, 162],
    );
    assert_tree_multi_dtype(
        vec![
            Complex32::new(2.0, 0.0),
            Complex32::new(3.0, 0.0),
            Complex32::new(5.0, 0.0),
            Complex32::new(7.0, 0.0),
        ],
        Complex32::new(2.0, 0.0),
        Complex32::new(10.0, 0.0),
        Complex32::new(1.0, 1.0),
        vec![
            Complex32::new(44.0, 10.0),
            Complex32::new(54.0, 10.0),
            Complex32::new(64.0, 10.0),
            Complex32::new(74.0, 10.0),
            Complex32::new(90.0, 10.0),
            Complex32::new(114.0, 10.0),
            Complex32::new(138.0, 10.0),
            Complex32::new(162.0, 10.0),
        ],
    );
    assert_tree_multi_dtype(
        vec![
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(5.0, 0.0),
            Complex64::new(7.0, 0.0),
        ],
        Complex64::new(2.0, 0.0),
        Complex64::new(10.0, 0.0),
        Complex64::new(1.0, 1.0),
        vec![
            Complex64::new(44.0, 10.0),
            Complex64::new(54.0, 10.0),
            Complex64::new(64.0, 10.0),
            Complex64::new(74.0, 10.0),
            Complex64::new(90.0, 10.0),
            Complex64::new(114.0, 10.0),
            Complex64::new(138.0, 10.0),
            Complex64::new(162.0, 10.0),
        ],
    );
}

#[test]
fn tree_transform_multi_pack_gemm_scatter_supports_complex_data_with_real_coefficients() {
    assert_tree_multi_mixed_dtype(
        vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(3.0, 0.0),
            Complex32::new(4.0, 0.0),
            Complex32::new(5.0, 0.0),
            Complex32::new(6.0, 0.0),
            Complex32::new(7.0, 0.0),
            Complex32::new(8.0, 0.0),
        ],
        vec![2.0_f64, 3.0, 5.0, 7.0],
        Complex32::new(2.0, 0.0),
        Complex32::new(10.0, 0.0),
        Complex32::new(1.0, 1.0),
        vec![
            Complex32::new(44.0, 10.0),
            Complex32::new(54.0, 10.0),
            Complex32::new(64.0, 10.0),
            Complex32::new(74.0, 10.0),
            Complex32::new(90.0, 10.0),
            Complex32::new(114.0, 10.0),
            Complex32::new(138.0, 10.0),
            Complex32::new(162.0, 10.0),
        ],
    );
    assert_tree_multi_mixed_dtype(
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(5.0, 0.0),
            Complex64::new(6.0, 0.0),
            Complex64::new(7.0, 0.0),
            Complex64::new(8.0, 0.0),
        ],
        vec![2.0_f64, 3.0, 5.0, 7.0],
        Complex64::new(2.0, 0.0),
        Complex64::new(10.0, 0.0),
        Complex64::new(1.0, 1.0),
        vec![
            Complex64::new(44.0, 10.0),
            Complex64::new(54.0, 10.0),
            Complex64::new(64.0, 10.0),
            Complex64::new(74.0, 10.0),
            Complex64::new(90.0, 10.0),
            Complex64::new(114.0, 10.0),
            Complex64::new(138.0, 10.0),
            Complex64::new(162.0, 10.0),
        ],
    );
}

#[test]
fn tree_transform_multi_uses_tensorkit_recoupling_orientation_for_all_numeric_dtypes() {
    assert_tree_multi_tensorkit_orientation_dtype(
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0],
        2.0,
        3.0,
        1.0,
        vec![10623.0, 12843.0, 21243.0, 25683.0],
    );
    assert_tree_multi_tensorkit_orientation_dtype(
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0],
        2.0,
        3.0,
        1.0,
        vec![10623.0, 12843.0, 21243.0, 25683.0],
    );
    assert_tree_multi_tensorkit_orientation_dtype(
        vec![1_i32, 2, 3, 4, 5, 6],
        vec![10, 100, 1000, 20, 200, 2000],
        2,
        3,
        1,
        vec![10623, 12843, 21243, 25683],
    );
    assert_tree_multi_tensorkit_orientation_dtype(
        vec![1_i64, 2, 3, 4, 5, 6],
        vec![10, 100, 1000, 20, 200, 2000],
        2,
        3,
        1,
        vec![10623, 12843, 21243, 25683],
    );
    assert_tree_multi_tensorkit_orientation_dtype(
        vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(3.0, 0.0),
            Complex32::new(4.0, 0.0),
            Complex32::new(5.0, 0.0),
            Complex32::new(6.0, 0.0),
        ],
        vec![
            Complex32::new(10.0, 0.0),
            Complex32::new(100.0, 0.0),
            Complex32::new(1000.0, 0.0),
            Complex32::new(20.0, 0.0),
            Complex32::new(200.0, 0.0),
            Complex32::new(2000.0, 0.0),
        ],
        Complex32::new(2.0, 0.0),
        Complex32::new(3.0, 0.0),
        Complex32::new(1.0, 1.0),
        vec![
            Complex32::new(10623.0, 3.0),
            Complex32::new(12843.0, 3.0),
            Complex32::new(21243.0, 3.0),
            Complex32::new(25683.0, 3.0),
        ],
    );
    assert_tree_multi_tensorkit_orientation_dtype(
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(5.0, 0.0),
            Complex64::new(6.0, 0.0),
        ],
        vec![
            Complex64::new(10.0, 0.0),
            Complex64::new(100.0, 0.0),
            Complex64::new(1000.0, 0.0),
            Complex64::new(20.0, 0.0),
            Complex64::new(200.0, 0.0),
            Complex64::new(2000.0, 0.0),
        ],
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(1.0, 1.0),
        vec![
            Complex64::new(10623.0, 3.0),
            Complex64::new(12843.0, 3.0),
            Complex64::new(21243.0, 3.0),
            Complex64::new(25683.0, 3.0),
        ],
    );
}

#[test]
fn tree_transform_dense_backend_matches_tensorkit_recoupling_orientation_for_gemm_dtypes() {
    assert_tree_multi_tensorkit_orientation_dense_dtype(
        vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0],
        2.0,
        3.0,
        1.0,
        vec![10623.0, 12843.0, 21243.0, 25683.0],
    );
    assert_tree_multi_tensorkit_orientation_dense_dtype(
        vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0],
        2.0,
        3.0,
        1.0,
        vec![10623.0, 12843.0, 21243.0, 25683.0],
    );
    assert_tree_multi_tensorkit_orientation_dense_dtype(
        vec![
            Complex32::new(1.0, 0.0),
            Complex32::new(2.0, 0.0),
            Complex32::new(3.0, 0.0),
            Complex32::new(4.0, 0.0),
            Complex32::new(5.0, 0.0),
            Complex32::new(6.0, 0.0),
        ],
        vec![
            Complex32::new(10.0, 0.0),
            Complex32::new(100.0, 0.0),
            Complex32::new(1000.0, 0.0),
            Complex32::new(20.0, 0.0),
            Complex32::new(200.0, 0.0),
            Complex32::new(2000.0, 0.0),
        ],
        Complex32::new(2.0, 0.0),
        Complex32::new(3.0, 0.0),
        Complex32::new(1.0, 1.0),
        vec![
            Complex32::new(10623.0, 3.0),
            Complex32::new(12843.0, 3.0),
            Complex32::new(21243.0, 3.0),
            Complex32::new(25683.0, 3.0),
        ],
    );
    assert_tree_multi_tensorkit_orientation_dense_dtype(
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
            Complex64::new(5.0, 0.0),
            Complex64::new(6.0, 0.0),
        ],
        vec![
            Complex64::new(10.0, 0.0),
            Complex64::new(100.0, 0.0),
            Complex64::new(1000.0, 0.0),
            Complex64::new(20.0, 0.0),
            Complex64::new(200.0, 0.0),
            Complex64::new(2000.0, 0.0),
        ],
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(1.0, 1.0),
        vec![
            Complex64::new(10623.0, 3.0),
            Complex64::new(12843.0, 3.0),
            Complex64::new(21243.0, 3.0),
            Complex64::new(25683.0, 3.0),
        ],
    );
}

#[test]
fn tree_transform_dense_backend_routes_multi_tree_recoupling_through_one_gemm() {
    let src_space = vec![6, 1];
    let dst_space = vec![4, 1];
    let src_structure =
        BlockStructure::packed_column_major(2, [vec![2, 1], vec![2, 1], vec![2, 1]]).unwrap();
    let dst_structure = BlockStructure::packed_column_major(2, [vec![2, 1], vec![2, 1]]).unwrap();
    let src = TestTensor::<f64>::from_vec_with_structure(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        src_space,
        src_structure,
    )
    .unwrap();
    let mut dst =
        TestTensor::<f64>::from_vec_with_structure(vec![1.0; 4], dst_space, dst_structure).unwrap();
    let structure = TreeTransformStructure::compile_structures(
        dst.structure(),
        src.structure(),
        &[TreeTransformBlockSpec::multi(
            vec![0, 1],
            vec![0, 1, 2],
            vec![10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0],
        )],
    )
    .unwrap();
    let mut backend = DenseTreeTransformOperations::new(CountingDenseExecutor::default());
    let mut workspace = TreeTransformWorkspace::default();

    replay_into(
        &mut backend,
        &mut workspace,
        &structure,
        &mut dst,
        &src,
        2.0,
        3.0,
    )
    .unwrap();

    // T20 (TensorKit _add_transform_multi! parity): the multi-tree
    // recoupling matrix is applied as ONE dense GEMM through the executor,
    // not as per-element scalar loops in the replay kernel.
    assert_eq!(backend.dense().dot_general_into_calls, 1);
    assert_eq!(dst.data(), &[10623.0, 12843.0, 21243.0, 25683.0]);
}

#[test]
fn tree_transform_recoupling_replay_uses_sorted_plan_offsets() {
    let src_space = vec![8, 1];
    let dst_space = vec![8, 1];
    let structure = BlockStructure::packed_column_major(
        2,
        [
            vec![2, 1],
            vec![2, 1],
            vec![1, 1],
            vec![1, 1],
            vec![1, 1],
            vec![1, 1],
        ],
    )
    .unwrap();
    let src = TestTensor::<f64>::from_vec_with_structure(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        src_space,
        structure.clone(),
    )
    .unwrap();
    let mut dst =
        TestTensor::<f64>::from_vec_with_structure(vec![0.0; 8], dst_space, structure).unwrap();
    let transform = TreeTransformStructure::compile_structures(
        dst.structure(),
        src.structure(),
        &[
            TreeTransformBlockSpec::multi(vec![2, 3], vec![2, 3], vec![11.0, 13.0, 17.0, 19.0]),
            TreeTransformBlockSpec::multi(vec![0, 1], vec![0, 1], vec![2.0, 3.0, 5.0, 7.0]),
            TreeTransformBlockSpec::multi(vec![4, 5], vec![4, 5], vec![23.0, 29.0, 31.0, 37.0]),
        ],
    )
    .unwrap();
    let mut backend = DenseTreeTransformOperations::default();
    let mut workspace = TreeTransformWorkspace::default();

    replay_into(
        &mut backend,
        &mut workspace,
        &transform,
        &mut dst,
        &src,
        1.0,
        0.0,
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[11.0, 16.0, 26.0, 38.0, 133.0, 199.0, 393.0, 513.0]
    );

    dst.data_mut().fill(0.0);
    let mut backend = DenseTreeTransformOperations::default();
    // What: two jobs enter the first chunk and the distinct third coefficient
    // matrix enters a later chunk, pinning global coefficient offsets.
    backend.set_recoupling_threads(std::num::NonZeroUsize::new(2).unwrap());
    backend.set_transform_parallel_min_len(0);
    let mut workspace = TreeTransformWorkspace::default();
    replay_into(
        &mut backend,
        &mut workspace,
        &transform,
        &mut dst,
        &src,
        1.0,
        0.0,
    )
    .unwrap();

    assert_eq!(
        dst.data(),
        &[11.0, 16.0, 26.0, 38.0, 133.0, 199.0, 393.0, 513.0]
    );
}
