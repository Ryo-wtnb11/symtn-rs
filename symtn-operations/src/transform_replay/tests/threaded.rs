use super::*;

#[test]
fn finite_task_view_serial_matches_per_block_replay() -> Result<(), OperationError> {
    let block_structure =
        Arc::new(BlockStructure::packed_column_major(2, vec![vec![2, 2]; 4]).unwrap());
    let transform = TreeTransformStructure::compile_structures(
        &block_structure,
        &block_structure,
        &[
            TreeTransformBlockSpec::single(0, 0, -2.0).with_source_axes([1, 0]),
            TreeTransformBlockSpec::multi(vec![1, 2], vec![1, 2], vec![1.0, 2.0, 3.0, 4.0])
                .with_source_axes([1, 0]),
        ],
    )
    .unwrap();
    let source = (1..=16).map(f64::from).collect::<Vec<_>>();

    for overwrite in [false, true] {
        let mut expected = (17..=32).map(f64::from).collect::<Vec<_>>();
        let mut actual = expected.clone();
        let mut kernels = StridedHostKernelAdapter::default();
        if overwrite {
            tree_transform_structure_overwrite_with_strided_kernel_raw(
                &mut kernels,
                &mut TreeTransformWorkspace::default(),
                &transform,
                &block_structure,
                &block_structure,
                &mut expected,
                &source,
                0.5,
            )?;
            tree_transform_structure_overwrite_with_structural_recoupling_raw(
                &mut kernels,
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &transform,
                &block_structure,
                &block_structure,
                &mut actual,
                &source,
                0.5,
                &[],
                1,
            )?;
        } else {
            tree_transform_structure_with_strided_kernel_raw(
                &mut kernels,
                &mut TreeTransformWorkspace::default(),
                &transform,
                &block_structure,
                &block_structure,
                &mut expected,
                &source,
                0.5,
                -0.25,
            )?;
            tree_transform_structure_with_structural_recoupling_raw(
                &mut kernels,
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &transform,
                &block_structure,
                &block_structure,
                &mut actual,
                &source,
                0.5,
                -0.25,
                1,
            )?;
        }
        assert_eq!(actual, expected);
    }
    Ok(())
}

#[test]
fn threaded_active_interleaved_layout_uses_the_serial_fallback() {
    let src_structure = BlockStructure::packed_column_major(1, [vec![2], vec![2]]).unwrap();
    let dst_structure = custom_structure(vec![block(0, 2, 2, 0), block(1, 2, 2, 1)]);
    let structure = TreeTransformStructure::compile_structures(
        &dst_structure,
        &src_structure,
        &[
            TreeTransformBlockSpec::single(0, 0, 2.0),
            TreeTransformBlockSpec::single(1, 1, -1.0),
        ],
    )
    .unwrap();
    assert!(!structure.parallel_schedule().singles_slice_disjoint);
    let src = [1.0, 2.0, 3.0, 4.0];
    let mut serial = [10.0, 20.0, 30.0, 40.0];
    let mut threaded = serial;
    let dst_structure = Arc::new(dst_structure);
    let src_structure = Arc::new(src_structure);
    for (dst, threads) in [(&mut serial[..], 1), (&mut threaded[..], 4)] {
        tree_transform_structure_with_structural_recoupling_raw(
            &mut StridedHostKernelAdapter::default(),
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            &structure,
            &dst_structure,
            &src_structure,
            dst,
            &src,
            1.0,
            0.5,
            threads,
        )
        .unwrap();
    }

    assert_eq!(threaded, serial);
    assert_eq!(threaded, [7.0, 7.0, 19.0, 16.0]);
}

#[test]
fn threaded_replay_handles_rank_zero_blocks() {
    let structure = Arc::new(
        BlockStructure::packed_column_major(0, [Vec::<usize>::new(), Vec::new()]).unwrap(),
    );
    let transform = TreeTransformStructure::compile_structures(
        &structure,
        &structure,
        &[
            TreeTransformBlockSpec::single(0, 0, 2.0),
            TreeTransformBlockSpec::single(1, 1, -1.0),
        ],
    )
    .unwrap();
    let src = [3.0, 5.0];
    let mut serial = [10.0, 20.0];
    let mut threaded = serial;
    for (dst, threads) in [(&mut serial[..], 1), (&mut threaded[..], 4)] {
        tree_transform_structure_with_structural_recoupling_raw(
            &mut StridedHostKernelAdapter::default(),
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            &transform,
            &structure,
            &structure,
            dst,
            &src,
            1.0,
            0.5,
            threads,
        )
        .unwrap();
    }
    assert_eq!(threaded, serial);
    assert_eq!(threaded, [11.0, 5.0]);
}

#[test]
fn threaded_replay_ignores_zero_extent_work() {
    let structure = Arc::new(BlockStructure::packed_column_major(1, [vec![0]]).unwrap());
    let transform = TreeTransformStructure::compile_structures(
        &structure,
        &structure,
        &[TreeTransformBlockSpec::single(0, 0, 1.0)],
    )
    .unwrap();
    assert!(transform.parallel_schedule().singles.is_empty());
    tree_transform_structure_with_structural_recoupling_raw(
        &mut StridedHostKernelAdapter::default(),
        &mut DefaultDenseExecutor::new(),
        &mut TreeTransformWorkspace::default(),
        &transform,
        &structure,
        &structure,
        &mut [],
        &[],
        1.0,
        f64::NAN,
        4,
    )
    .unwrap();
}

#[test]
fn threaded_multi_scatter_falls_back_for_interleaved_destinations() {
    let src_structure =
        Arc::new(BlockStructure::packed_column_major(1, [vec![2], vec![2]]).unwrap());
    let dst_structure = Arc::new(custom_structure(vec![block(0, 2, 2, 0), block(1, 2, 2, 1)]));
    let transform = TreeTransformStructure::compile_structures(
        &dst_structure,
        &src_structure,
        &[TreeTransformBlockSpec::multi(
            vec![0, 1],
            vec![0, 1],
            vec![1.0, 0.0, 0.0, 1.0],
        )],
    )
    .unwrap();
    assert_eq!(transform.parallel_schedule().scatter_groups.len(), 1);
    assert!(!transform.parallel_schedule().scatter_groups[0].slice_disjoint);
    let src = [1.0, 2.0, 3.0, 4.0];
    let mut serial = [10.0, 20.0, 30.0, 40.0];
    let mut threaded = serial;
    for (dst, threads) in [(&mut serial[..], 1), (&mut threaded[..], 4)] {
        tree_transform_structure_with_structural_recoupling_raw(
            &mut StridedHostKernelAdapter::default(),
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            &transform,
            &dst_structure,
            &src_structure,
            dst,
            &src,
            1.0,
            0.5,
            threads,
        )
        .unwrap();
    }

    assert_eq!(threaded, serial);
    assert_eq!(threaded, [6.0, 13.0, 17.0, 24.0]);
}

#[test]
fn huge_thread_request_is_capped_by_pool_and_runnable_work() {
    // What: normal and profiled replay use the same effective worker count,
    // preserve serial beta semantics, and never size scratch from a caller's
    // unbounded thread request.
    let block_structure =
        Arc::new(BlockStructure::packed_column_major(2, vec![vec![2, 2]; 4]).unwrap());
    let specs = (0..3)
        .map(|block| TreeTransformBlockSpec::single(block, block, 1.0).with_source_axes([1, 0]))
        .collect::<Vec<_>>();
    let transform =
        TreeTransformStructure::compile_structures(&block_structure, &block_structure, &specs)
            .unwrap();
    assert_eq!(transform.layouts().max_fused_rank(), 2);
    let source = (1..=16).map(f64::from).collect::<Vec<_>>();
    let initial = (10..=25).map(f64::from).collect::<Vec<_>>();

    let mut expected = initial.clone();
    tree_transform_structure_with_structural_recoupling_raw(
        &mut StridedHostKernelAdapter::default(),
        &mut DefaultDenseExecutor::new(),
        &mut TreeTransformWorkspace::default(),
        &transform,
        &block_structure,
        &block_structure,
        &mut expected,
        &source,
        1.0,
        0.5,
        1,
    )
    .unwrap();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(3)
        .build()
        .unwrap();
    let mut destination = initial.clone();
    let mut workspace = TreeTransformWorkspace::default();
    pool.install(|| {
        tree_transform_structure_with_structural_recoupling_raw(
            &mut StridedHostKernelAdapter::default(),
            &mut DefaultDenseExecutor::new(),
            &mut workspace,
            &transform,
            &block_structure,
            &block_structure,
            &mut destination,
            &source,
            1.0,
            0.5,
            usize::MAX,
        )
    })
    .unwrap();
    assert_eq!(workspace.fused_indices.len(), 3 * 2);
    assert_eq!(destination, expected);

    let mut profiled_destination = initial;
    let mut profiled_workspace = TreeTransformWorkspace::default();
    let mut profile = TreeTransformReplayProfile::default();
    pool.install(|| {
        tree_transform_structure_with_structural_recoupling_raw_profiled(
            &mut StridedHostKernelAdapter::default(),
            &mut DefaultDenseExecutor::new(),
            &mut profiled_workspace,
            &transform,
            &block_structure,
            &block_structure,
            &mut profiled_destination,
            &source,
            1.0,
            0.5,
            usize::MAX,
            &mut profile,
        )
    })
    .unwrap();
    assert_eq!(profiled_workspace.fused_indices.len(), 3 * 2);
    assert_eq!(profiled_destination, expected);
}

/// Destination block `d` of [`recoupling_fixture`] is
/// `sum_(s, c) c * transpose(source block s)` over `RECOUPLING_TERMS[d]`; an
/// empty term list is an inactive destination. The two Multi groups have
/// non-symmetric (and non-square) `U`, row `r` writing `dst_blocks[r]`.
const RECOUPLING_TERMS: [&[(usize, f64)]; 6] = [
    &[(0, 2.0)],
    &[(1, -1.5)],
    &[(2, 0.5), (3, -3.0)],
    &[(2, 1.0), (3, 2.0)],
    &[(4, 0.25), (5, -4.0)],
    &[],
];

/// Six 3x2 source blocks, six 2x3 destination blocks: two Singles, two
/// Multi groups and an inactive destination, so threaded replay has two
/// runnable Singles and four pack columns.
fn recoupling_fixture() -> (
    Arc<BlockStructure>,
    Arc<BlockStructure>,
    TreeTransformStructure<f64>,
) {
    let src = Arc::new(BlockStructure::packed_column_major(2, vec![vec![3, 2]; 6]).unwrap());
    let dst = Arc::new(BlockStructure::packed_column_major(2, vec![vec![2, 3]; 6]).unwrap());
    let replay = TreeTransformStructure::compile_structures(
        &dst,
        &src,
        &[
            TreeTransformBlockSpec::single(0, 0, 2.0).with_source_axes([1, 0]),
            TreeTransformBlockSpec::single(1, 1, -1.5).with_source_axes([1, 0]),
            TreeTransformBlockSpec::multi(vec![3, 2], vec![2, 3], vec![1.0, 2.0, 0.5, -3.0])
                .with_source_axes([1, 0]),
            TreeTransformBlockSpec::multi(vec![4], vec![4, 5], vec![0.25, -4.0])
                .with_source_axes([1, 0]),
        ],
    )
    .unwrap();
    (dst, src, replay)
}

/// Explicit-index oracle for [`recoupling_fixture`]: column-major blocks of
/// six elements, `dst_d(i, j) = alpha * sum c * src_s(j, i) + beta * dst_d(i, j)`
/// (`beta = None` is overwrite).
fn recoupling_expected<D>(src: &[D], dst: &[D], alpha: D, beta: Option<D>) -> Vec<D>
where
    D: Copy + num_traits::Zero + std::ops::Mul<Output = D> + From<f64>,
{
    let mut expected = vec![D::zero(); 36];
    for (d, terms) in RECOUPLING_TERMS.iter().enumerate() {
        for i in 0..2 {
            for j in 0..3 {
                let at = 6 * d + i + 2 * j;
                let sum = terms.iter().fold(D::zero(), |sum, &(s, c)| {
                    sum + D::from(c) * src[6 * s + j + 3 * i]
                });
                expected[at] = alpha * sum + beta.map_or(D::zero(), |beta| beta * dst[at]);
            }
        }
    }
    expected
}

/// Runs both replay modes with `threads` workers in a pool of exactly that
/// size and returns `(axpby, overwrite)`. A fresh workspace sizes its
/// fused-index scratch per worker, so its length shows the driver used.
fn replay_recoupling_fixture<D>(
    threads: usize,
    src: &[D],
    dst: &[D],
    alpha: D,
    beta: D,
) -> [Vec<D>; 2]
where
    D: DenseRecouplingScalar + RecouplingCoefficientAction<f64> + ConjugateValue,
    StridedHostKernelAdapter: HostKernelAdapter<D>,
{
    let (dst_structure, src_structure, replay) = recoupling_fixture();
    let fused_rank = replay.layouts().max_fused_rank();
    assert!(fused_rank > 0);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    pool.install(|| {
        let mut axpby = dst.to_vec();
        let mut workspace = TreeTransformWorkspace::default();
        tree_transform_structure_with_structural_recoupling_raw(
            &mut StridedHostKernelAdapter::default(),
            &mut DefaultDenseExecutor::new(),
            &mut workspace,
            &replay,
            &dst_structure,
            &src_structure,
            &mut axpby,
            src,
            alpha,
            beta,
            threads,
        )
        .unwrap();
        assert_eq!(workspace.fused_indices.len(), threads * fused_rank);

        let mut overwrite = dst.to_vec();
        let mut workspace = TreeTransformWorkspace::default();
        tree_transform_structure_overwrite_with_structural_recoupling_raw(
            &mut StridedHostKernelAdapter::default(),
            &mut DefaultDenseExecutor::new(),
            &mut workspace,
            &replay,
            &dst_structure,
            &src_structure,
            &mut overwrite,
            src,
            alpha,
            &[],
            threads,
        )
        .unwrap();
        assert_eq!(workspace.fused_indices.len(), threads * fused_rank);
        [axpby, overwrite]
    })
}

/// What: serial and threaded replay both equal the explicit-index oracle on a
/// plan with several runnable Singles and non-identity Multi recoupling, in
/// accumulate (inactive destination scaled by beta) and overwrite (NaN
/// destination never read) modes. All values are dyadic, so equality is exact.
#[test]
fn threaded_non_identity_recoupling_matches_explicit_index_oracle() {
    let src = (1..=36).map(f64::from).collect::<Vec<_>>();
    let dst = (101..=136).map(f64::from).collect::<Vec<_>>();
    let nan_dst = vec![f64::NAN; 36];
    for threads in [1, 2] {
        let [axpby, _] = replay_recoupling_fixture(threads, &src, &dst, 0.5, -0.25);
        assert_eq!(axpby, recoupling_expected(&src, &dst, 0.5, Some(-0.25)));
        let [_, overwrite] = replay_recoupling_fixture(threads, &src, &nan_dst, 0.5, -0.25);
        assert_eq!(overwrite, recoupling_expected(&src, &dst, 0.5, None));
    }
}

#[test]
fn threaded_non_identity_recoupling_matches_oracle_for_complex_data() {
    use num_complex::Complex64;
    let src = (1..=36)
        .map(|k| Complex64::new(f64::from(k), f64::from(37 - k)))
        .collect::<Vec<_>>();
    let dst = (1..=36)
        .map(|k| Complex64::new(-f64::from(k), 0.5 * f64::from(k)))
        .collect::<Vec<_>>();
    let alpha = Complex64::new(0.5, -1.25);
    let beta = Complex64::new(-0.25, 0.75);
    for threads in [1, 2] {
        let [axpby, overwrite] = replay_recoupling_fixture(threads, &src, &dst, alpha, beta);
        assert_eq!(axpby, recoupling_expected(&src, &dst, alpha, Some(beta)));
        assert_eq!(overwrite, recoupling_expected(&src, &dst, alpha, None));
    }
}
