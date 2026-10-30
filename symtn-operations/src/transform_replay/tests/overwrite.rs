use super::*;

fn expected(beta: f64) -> [f64; 2] {
    [6.0 + beta * 10.0, beta * 20.0]
}

struct FailFirstBatchExecutor {
    inner: DefaultDenseExecutor,
    fail_next: bool,
}

impl FailFirstBatchExecutor {
    fn new() -> Self {
        Self {
            inner: DefaultDenseExecutor::new(),
            fail_next: true,
        }
    }
}

impl DenseExecutor for FailFirstBatchExecutor {
    fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.svd(input)
    }

    fn qr(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.qr(input)
    }

    fn eigh(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.eigh(input)
    }

    fn dot_general_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        config: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        self.inner.dot_general_into(output, lhs, rhs, config)
    }

    fn matmul_batch_axpby_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        jobs: &[DenseGemmBatchJob],
        runs: &[usize],
        alpha: DenseScalar,
        beta: DenseScalar,
    ) -> Result<(), DenseError> {
        if self.fail_next {
            self.fail_next = false;
            match output {
                DenseWrite::F64(mut output) => output.data_mut().fill(f64::NAN),
                _ => unreachable!("retry oracle uses f64"),
            }
            return Err(DenseError::Backend {
                backend: DenseBackend::Tenferro,
                op: "matmul_batch_axpby_into",
                message: "injected first-call failure".to_string(),
            });
        }
        self.inner
            .matmul_batch_axpby_into(output, lhs, rhs, jobs, runs, alpha, beta)
    }
}

#[test]
fn inactive_destination_scaling_preserves_storage_padding() {
    let src_structure = BlockStructure::packed_column_major(1, [vec![1]]).unwrap();
    let dst_structure = custom_structure(vec![block(0, 2, 2, 0)]);
    let structure =
        TreeTransformStructure::<f64>::compile_structures(&dst_structure, &src_structure, &[])
            .unwrap();
    let mut dst = vec![10.0, 99.0, 30.0];
    tree_transform_structure_with_structural_recoupling_raw(
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
    assert_eq!(dst, [5.0, 99.0, 15.0]);
}

#[test]
fn nan_beta_reaches_active_and_inactive_destinations() {
    let (mut dst, src, structure) = fixture();
    tree_transform_structure_with_structural_recoupling_raw(
        &mut StridedHostKernelAdapter::default(),
        &mut DefaultDenseExecutor::new(),
        &mut TreeTransformWorkspace::default(),
        &structure,
        &Arc::clone(dst.structure()),
        &Arc::clone(src.structure()),
        dst.data_mut(),
        src.data(),
        1.0,
        f64::NAN,
        4,
    )
    .unwrap();
    assert!(dst.data().iter().all(|value| value.is_nan()));
}

/// What: `beta = 0` scales an inactive destination by VectorInterface's
/// `scale(x, 0) = zero(x) * 0`, so a NaN there becomes zero rather than
/// `0 * NaN` (#1438), as TensorKit's `scale!(tdst, β)` does.
#[test]
fn generic_beta_zero_wipes_nan_in_inactive_destinations() {
    let (mut dst, src, structure) = fixture();
    dst.data_mut().fill(f64::NAN);

    tree_transform_structure_with_strided_kernel_raw(
        &mut StridedHostKernelAdapter::default(),
        &mut TreeTransformWorkspace::default(),
        &structure,
        &Arc::clone(dst.structure()),
        &Arc::clone(src.structure()),
        dst.data_mut(),
        src.data(),
        1.0,
        0.0,
    )
    .unwrap();

    assert_eq!(dst.data()[0], 6.0);
    assert_eq!(dst.data()[1].to_bits(), 0.0f64.to_bits());
}

#[test]
fn overwrite_single_does_not_read_nan_destinations_on_any_replay_route() {
    // The one-block fixture never reaches the threaded driver; threaded
    // overwrite is covered by threaded.rs's multi-block recoupling oracle.
    let (mut dst, src, structure) = fixture();
    dst.data_mut().fill(f64::NAN);
    tree_transform_structure_overwrite_with_structural_recoupling_raw(
        &mut StridedHostKernelAdapter::default(),
        &mut DefaultDenseExecutor::new(),
        &mut TreeTransformWorkspace::default(),
        &structure,
        &Arc::clone(dst.structure()),
        &Arc::clone(src.structure()),
        dst.data_mut(),
        src.data(),
        1.0,
        &[],
        1,
    )
    .unwrap();
    assert_eq!(dst.data(), &[6.0, 0.0]);

    let (mut dst, src, structure) = fixture();
    dst.data_mut().fill(f64::NAN);
    tree_transform_structure_overwrite_with_strided_kernel_raw(
        &mut StridedHostKernelAdapter::default(),
        &mut TreeTransformWorkspace::default(),
        &structure,
        &Arc::clone(dst.structure()),
        &Arc::clone(src.structure()),
        dst.data_mut(),
        src.data(),
        1.0,
    )
    .unwrap();
    assert_eq!(dst.data(), &[6.0, 0.0]);

    let (mut dst, src, structure) = fixture();
    dst.data_mut().fill(f64::NAN);
    tree_transform_structure_overwrite_with_structural_recoupling_raw_profiled(
        &mut StridedHostKernelAdapter::default(),
        &mut DefaultDenseExecutor::new(),
        &mut TreeTransformWorkspace::default(),
        &structure,
        &Arc::clone(dst.structure()),
        &Arc::clone(src.structure()),
        dst.data_mut(),
        src.data(),
        1.0,
        &[],
        1,
        &mut TreeTransformReplayProfile::default(),
    )
    .unwrap();
    assert_eq!(dst.data(), &[6.0, 0.0]);
}

#[test]
fn overwrite_multi_does_not_read_nan_active_or_inactive_destinations() {
    let (dst_structure, src_structure, structure) = identity_multi_fixture();
    let mut dst = vec![f64::NAN; 3];

    tree_transform_structure_overwrite_with_strided_kernel_raw(
        &mut StridedHostKernelAdapter::default(),
        &mut TreeTransformWorkspace::default(),
        &structure,
        &dst_structure,
        &src_structure,
        &mut dst,
        &[3.0, 4.0],
        2.0,
    )
    .unwrap();

    assert_eq!(dst, [6.0, 8.0, 0.0]);
}

#[test]
fn overwrite_multi_c64_does_not_read_nan_destinations() {
    let (dst_structure, src_structure, structure) = identity_multi_fixture();
    let fused_rank = structure.layouts().max_fused_rank();
    // Two pack columns: two workers is the most the threaded driver can use.
    for threads in [1, 2] {
        let nan = num_complex::Complex64::new(f64::NAN, f64::NAN);
        let mut dst = vec![nan; 3];
        let mut workspace = TreeTransformWorkspace::default();
        let pool = crate::host_pool::test_pool(threads);
        pool.install(|| {
            tree_transform_structure_overwrite_with_structural_recoupling_raw(
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut workspace,
                &structure,
                &dst_structure,
                &src_structure,
                &mut dst,
                &[
                    num_complex::Complex64::new(3.0, 1.0),
                    num_complex::Complex64::new(4.0, -1.0),
                ],
                num_complex::Complex64::new(2.0, 0.0),
                &[],
                threads,
            )
        })
        .unwrap();
        assert_eq!(workspace.fused_indices.len(), threads * fused_rank);
        assert_eq!(
            dst,
            [
                num_complex::Complex64::new(6.0, 2.0),
                num_complex::Complex64::new(8.0, -2.0),
                num_complex::Complex64::new(0.0, 0.0),
            ]
        );
    }
}

#[test]
fn overwrite_threaded_single_and_multi_ignore_destination_bits() {
    let src_structure = Arc::new(
        BlockStructure::packed_column_major(1, [vec![1], vec![1], vec![1], vec![1]]).unwrap(),
    );
    let dst_structure = Arc::new(
        BlockStructure::packed_column_major(1, [vec![1], vec![1], vec![1], vec![1], vec![1]])
            .unwrap(),
    );
    let structure = TreeTransformStructure::compile_structures(
        &dst_structure,
        &src_structure,
        &[
            TreeTransformBlockSpec::single(0, 0, 2.0),
            TreeTransformBlockSpec::single(1, 1, -1.0),
            TreeTransformBlockSpec::multi(vec![2, 3], vec![2, 3], vec![1.0, 0.0, 0.0, 1.0]),
        ],
    )
    .unwrap();
    let src = [3.0, 4.0, 5.0, 6.0];
    let expected = [6.0, -4.0, 5.0, 6.0, 0.0];

    let fused_rank = structure.layouts().max_fused_rank();
    // Two runnable Singles and two pack columns: two workers is the most the
    // threaded driver can use.
    for threads in [1, 2] {
        let mut dst = [f64::NAN; 5];
        let mut workspace = TreeTransformWorkspace::default();
        let pool = crate::host_pool::test_pool(threads);
        pool.install(|| {
            tree_transform_structure_overwrite_with_structural_recoupling_raw(
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut workspace,
                &structure,
                &dst_structure,
                &src_structure,
                &mut dst,
                &src,
                1.0,
                &[],
                threads,
            )
        })
        .unwrap();
        assert_eq!(workspace.fused_indices.len(), threads * fused_rank);
        assert_eq!(dst, expected);
    }
}

#[test]
fn overwrite_multi_recovers_from_dirty_packed_scratch_after_failure() {
    let (dst_structure, src_structure, replay) = identity_multi_fixture();
    let mut workspace = TreeTransformWorkspace::default();
    let mut dense = FailFirstBatchExecutor::new();
    let mut dst = [f64::NAN; 3];

    assert!(
        tree_transform_structure_overwrite_with_structural_recoupling_raw(
            &mut StridedHostKernelAdapter::default(),
            &mut dense,
            &mut workspace,
            &replay,
            &dst_structure,
            &src_structure,
            &mut dst,
            &[3.0, 4.0],
            2.0,
            &[],
            1,
        )
        .is_err()
    );
    assert!(workspace
        .packed
        .destination()
        .as_slice()
        .iter()
        .all(|value| value.is_nan()));

    tree_transform_structure_overwrite_with_structural_recoupling_raw(
        &mut StridedHostKernelAdapter::default(),
        &mut dense,
        &mut workspace,
        &replay,
        &dst_structure,
        &src_structure,
        &mut dst,
        &[3.0, 4.0],
        2.0,
        &[],
        1,
    )
    .unwrap();
    assert_eq!(dst, [6.0, 8.0, 0.0]);
}

#[test]
fn overwrite_validates_before_mutation_and_accepts_rank_boundaries() {
    let (mut dst, src, structure) = fixture();
    dst.data_mut().fill(f64::NAN);
    let before = dst.data().to_vec();
    let result = tree_transform_structure_overwrite_with_strided_kernel_raw(
        &mut StridedHostKernelAdapter::default(),
        &mut TreeTransformWorkspace::default(),
        &structure,
        &Arc::clone(dst.structure()),
        &Arc::clone(src.structure()),
        dst.data_mut(),
        &[],
        1.0,
    );
    assert!(result.is_err());
    assert_eq!(
        dst.data()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        before
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );

    let result = tree_transform_structure_overwrite_with_structural_recoupling_raw(
        &mut StridedHostKernelAdapter::default(),
        &mut DefaultDenseExecutor::new(),
        &mut TreeTransformWorkspace::default(),
        &structure,
        &Arc::clone(dst.structure()),
        &Arc::clone(src.structure()),
        dst.data_mut(),
        &[],
        1.0,
        &[],
        1,
    );
    assert!(result.is_err());
    assert_eq!(
        dst.data()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        before
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );

    let scalar_structure =
        Arc::new(BlockStructure::packed_column_major(0, [Vec::<usize>::new()]).unwrap());
    let scalar_replay = TreeTransformStructure::compile_structures(
        &scalar_structure,
        &scalar_structure,
        &[TreeTransformBlockSpec::single(0, 0, 2.0)],
    )
    .unwrap();
    let mut scalar_dst = [f64::NAN];
    tree_transform_structure_overwrite_with_strided_kernel_raw(
        &mut StridedHostKernelAdapter::default(),
        &mut TreeTransformWorkspace::default(),
        &scalar_replay,
        &scalar_structure,
        &scalar_structure,
        &mut scalar_dst,
        &[3.0],
        1.0,
    )
    .unwrap();
    assert_eq!(scalar_dst, [6.0]);

    let empty_structure = Arc::new(BlockStructure::packed_column_major(1, [vec![0]]).unwrap());
    let empty_replay = TreeTransformStructure::compile_structures(
        &empty_structure,
        &empty_structure,
        &[TreeTransformBlockSpec::single(0, 0, 1.0)],
    )
    .unwrap();
    tree_transform_structure_overwrite_with_strided_kernel_raw(
        &mut StridedHostKernelAdapter::default(),
        &mut TreeTransformWorkspace::default(),
        &empty_replay,
        &empty_structure,
        &empty_structure,
        &mut [],
        &[],
        1.0,
    )
    .unwrap();
}

#[test]
fn structural_serial_replay_scales_inactive_destinations() {
    for beta in [0.0, 0.5, 1.0] {
        let (mut dst, src, structure) = fixture();
        let dst_structure = Arc::clone(dst.structure());
        tree_transform_structure_with_structural_recoupling_raw(
            &mut StridedHostKernelAdapter::default(),
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            &structure,
            &dst_structure,
            src.structure(),
            dst.data_mut(),
            src.data(),
            1.0,
            beta,
            1,
        )
        .unwrap();
        assert_eq!(dst.data(), &expected(beta));
    }
}

#[test]
fn strided_replay_scales_inactive_destinations() {
    for beta in [0.0, 0.5, 1.0] {
        let (mut dst, src, structure) = fixture();
        let dst_structure = Arc::clone(dst.structure());
        let src_structure = Arc::clone(src.structure());
        tree_transform_structure_with_strided_kernel_raw(
            &mut StridedHostKernelAdapter::default(),
            &mut TreeTransformWorkspace::default(),
            &structure,
            &dst_structure,
            &src_structure,
            dst.data_mut(),
            src.data(),
            1.0,
            beta,
        )
        .unwrap();
        assert_eq!(dst.data(), &expected(beta));
    }
}
