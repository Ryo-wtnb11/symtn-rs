use super::*;

#[test]
fn finite_task_view_serial_matches_per_block_replay() -> Result<(), OperationError> {
    let engine = &mut symtn_dense::CpuEngine::serial();
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
                strided_kernel::Exec::serial(),
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
                engine,
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
                strided_kernel::Exec::serial(),
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
                engine,
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
        crate::host_pool::test_pool(threads)
            .install(|engine| {
                tree_transform_structure_with_structural_recoupling_raw(
                    engine,
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
            })
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
        crate::host_pool::test_pool(threads)
            .install(|engine| {
                tree_transform_structure_with_structural_recoupling_raw(
                    engine,
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
            })
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
    crate::host_pool::test_pool(4)
        .install(|engine| {
            tree_transform_structure_with_structural_recoupling_raw(
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
                f64::NAN,
                4,
            )
        })
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
        crate::host_pool::test_pool(threads)
            .install(|engine| {
                tree_transform_structure_with_structural_recoupling_raw(
                    engine,
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
            })
            .unwrap();
    }

    assert_eq!(threaded, serial);
    assert_eq!(threaded, [6.0, 13.0, 17.0, 24.0]);
}

#[test]
fn huge_thread_request_is_capped_by_pool_and_runnable_work() {
    let engine = &mut symtn_dense::CpuEngine::serial();
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
        engine,
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

    let mut pool = crate::host_pool::test_pool(3);
    let mut destination = initial.clone();
    let mut workspace = TreeTransformWorkspace::default();
    pool.install(|engine| {
        tree_transform_structure_with_structural_recoupling_raw(
            engine,
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
    assert_eq!(
        workspace.fused_indices.len(),
        2,
        "only the calling lane is shared"
    );
    assert_eq!(destination, expected);

    let mut profiled_destination = initial;
    let mut profiled_workspace = TreeTransformWorkspace::default();
    let mut profile = TreeTransformReplayProfile::default();
    pool.install(|engine| {
        tree_transform_structure_with_structural_recoupling_raw_profiled(
            engine,
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
    assert_eq!(profiled_workspace.fused_indices.len(), 2);
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

/// Destination block shape `[ROWS, COLS]`; source blocks are its transpose.
/// 24 elements per block keeps Strided's copy estimate nonzero on every host
/// class (at least 0.09 ns per element), so each overwrite role retains a
/// plan for the wide pool to widen.
const ROWS: usize = 4;
const COLS: usize = 6;
const BLOCK: usize = ROWS * COLS;

/// Six COLSxROWS source blocks, six ROWSxCOLS destination blocks: two
/// Singles, two Multi groups and an inactive destination, so threaded replay
/// has two runnable Singles and four pack columns.
fn recoupling_fixture() -> (
    Arc<BlockStructure>,
    Arc<BlockStructure>,
    TreeTransformStructure<f64>,
) {
    let src = Arc::new(BlockStructure::packed_column_major(2, vec![vec![COLS, ROWS]; 6]).unwrap());
    let dst = Arc::new(BlockStructure::packed_column_major(2, vec![vec![ROWS, COLS]; 6]).unwrap());
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
/// `BLOCK` elements, `dst_d(i, j) = alpha * sum c * src_s(j, i) + beta * dst_d(i, j)`
/// (`beta = None` is overwrite).
fn recoupling_expected<D>(src: &[D], dst: &[D], alpha: D, beta: Option<D>) -> Vec<D>
where
    D: Copy + num_traits::Zero + std::ops::Mul<Output = D> + From<f64>,
{
    let mut expected = vec![D::zero(); 6 * BLOCK];
    for (d, terms) in RECOUPLING_TERMS.iter().enumerate() {
        for i in 0..ROWS {
            for j in 0..COLS {
                let at = BLOCK * d + i + ROWS * j;
                let sum = terms.iter().fold(D::zero(), |sum, &(s, c)| {
                    sum + D::from(c) * src[BLOCK * s + j + COLS * i]
                });
                expected[at] = alpha * sum + beta.map_or(D::zero(), |beta| beta * dst[at]);
            }
        }
    }
    expected
}

/// Runs both replay modes with `threads` replay workers and returns
/// `(axpby, overwrite)`. A fresh workspace sizes its fused-index scratch per
/// worker, so its length shows the driver used. The pool has two workers per
/// replay worker and declares free dispatch, so every block an overwrite
/// visits is widened onto its compiled plan.
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
    // Precondition: both Single roles and the three scatter columns retain a
    // plan, so a wide pool reaches the planned leaf on every host class.
    assert_eq!(replay.layouts().retained_plan_count(), 5);
    let mut pool = crate::host_pool::eager_test_pool(2 * threads);
    pool.install(|engine| {
        let mut axpby = dst.to_vec();
        let mut workspace = TreeTransformWorkspace::default();
        tree_transform_structure_with_structural_recoupling_raw(
            engine,
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
        // Below the inline rank only the calling lane's index is shared;
        // forked lanes keep their own on their stacks.
        assert_eq!(workspace.fused_indices.len(), fused_rank);

        let mut overwrite = dst.to_vec();
        let mut workspace = TreeTransformWorkspace::default();
        tree_transform_structure_overwrite_with_structural_recoupling_raw(
            engine,
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
        // Below the inline rank only the calling lane's index is shared;
        // forked lanes keep their own on their stacks.
        assert_eq!(workspace.fused_indices.len(), fused_rank);
        [axpby, overwrite]
    })
}

/// What: serial and threaded replay both equal the explicit-index oracle on a
/// plan with several runnable Singles and non-identity Multi recoupling, in
/// accumulate (inactive destination scaled by beta) and overwrite (NaN
/// destination never read) modes. All values are dyadic, so equality is exact.
#[test]
fn threaded_non_identity_recoupling_matches_explicit_index_oracle() {
    let len = 6 * BLOCK as u32;
    let src = (1..=len).map(f64::from).collect::<Vec<_>>();
    let dst = (101..=100 + len).map(f64::from).collect::<Vec<_>>();
    let nan_dst = vec![f64::NAN; src.len()];
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
    let len = 6 * BLOCK as u32;
    let src = (1..=len)
        .map(|k| Complex64::new(f64::from(k), f64::from(len + 1 - k)))
        .collect::<Vec<_>>();
    let dst = (1..=len)
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

/// What: the leaf gate's dispatch-cost scan runs only when an overwrite of a
/// role with a retained plan reaches admission, once per call on one token:
/// planless and accumulating calls scan nothing, and an overwrite of planned
/// roles scans exactly once on a wide engine, however many block lists (one
/// Single list, one scatter list per Multi block) the call visits.
#[test]
fn leaf_gate_scans_only_for_planned_overwrites() {
    use crate::kernel_adapter::sealed::GATE_SCANS;
    let scans = |replay: &mut dyn FnMut(&mut dyn Engine)| {
        let mut pool = crate::host_pool::eager_test_pool(4);
        pool.install(|engine| {
            let before = GATE_SCANS.with(|scans| scans.get());
            replay(engine);
            GATE_SCANS.with(|scans| scans.get()) - before
        })
    };

    // Eight Multi groups of one-element blocks: Strided estimates their copy
    // at zero time, so no scatter role retains a plan.
    let tiny = Arc::new(BlockStructure::packed_column_major(1, vec![vec![1]; 16]).unwrap());
    let specs = (0..8)
        .map(|g| {
            TreeTransformBlockSpec::multi(
                vec![2 * g, 2 * g + 1],
                vec![2 * g, 2 * g + 1],
                vec![1.0, 0.0, 0.0, 1.0],
            )
        })
        .collect::<Vec<_>>();
    let planless = TreeTransformStructure::compile_structures(&tiny, &tiny, &specs).unwrap();
    assert_eq!(planless.layouts().retained_plan_count(), 0);
    let source = (1..=16).map(f64::from).collect::<Vec<_>>();
    let mut destination = vec![f64::NAN; 16];
    let planless_scans = scans(&mut |engine| {
        tree_transform_structure_overwrite_with_structural_recoupling_raw(
            engine,
            &mut StridedHostKernelAdapter::default(),
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            &planless,
            &tiny,
            &tiny,
            &mut destination,
            &source,
            1.0,
            &[],
            1,
        )
        .unwrap();
    });
    assert_eq!(destination, source);
    assert_eq!(planless_scans, 0);

    // Three planned Singles of `BLOCK` elements, accumulated, then overwritten.
    let blocks =
        Arc::new(BlockStructure::packed_column_major(2, vec![vec![ROWS, COLS]; 3]).unwrap());
    let singles = TreeTransformStructure::compile_structures(
        &blocks,
        &blocks,
        &(0..3)
            .map(|b| TreeTransformBlockSpec::single(b, b, 2.0))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    assert_eq!(singles.layouts().retained_plan_count(), 3);
    let source = (1..=3 * BLOCK as u32).map(f64::from).collect::<Vec<_>>();
    let mut destination = vec![1.0; source.len()];
    let accumulate_scans = scans(&mut |engine| {
        tree_transform_structure_with_structural_recoupling_raw(
            engine,
            &mut StridedHostKernelAdapter::default(),
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            &singles,
            &blocks,
            &blocks,
            &mut destination,
            &source,
            1.0,
            0.5,
            1,
        )
        .unwrap();
    });
    let accumulated = source.iter().map(|&v| 2.0 * v + 0.5).collect::<Vec<_>>();
    assert_eq!(destination, accumulated);
    assert_eq!(accumulate_scans, 0);

    let overwrite_scans = scans(&mut |engine| {
        tree_transform_structure_overwrite_with_structural_recoupling_raw(
            engine,
            &mut StridedHostKernelAdapter::default(),
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            &singles,
            &blocks,
            &blocks,
            &mut destination,
            &source,
            1.0,
            &[],
            1,
        )
        .unwrap();
    });
    let overwritten = source.iter().map(|&v| 2.0 * v).collect::<Vec<_>>();
    assert_eq!(destination, overwritten);
    assert_eq!(overwrite_scans, 1);

    // Eight Multi groups of planned `BLOCK`-element scatter columns: the
    // serial initialized replay and the owned writer each visit one scatter
    // list per group and scan once.
    // A canonical U(1) layout (sixteen charges, one `ROWS x COLS` block each),
    // so the owned writer also admits it.
    let leg = |degeneracy: usize| {
        symtn_core::SectorLeg::new(
            (0..16).map(|charge| (symtn_core::U1Irrep::new(charge), degeneracy)),
            false,
        )
    };
    let groups = symtn_core::FusionTreeHomSpace::new(
        symtn_core::FusionProductSpace::new([leg(ROWS)]),
        symtn_core::FusionProductSpace::new([leg(COLS)]),
    )
    .coupled_subblock_structure_from_leg_degeneracies(&symtn_core::U1FusionRule)
    .unwrap();
    assert_eq!(groups.block_count(), 16);
    let specs = (0..8)
        .map(|g| {
            TreeTransformBlockSpec::multi(
                vec![2 * g, 2 * g + 1],
                vec![2 * g, 2 * g + 1],
                vec![1.0, 0.0, 0.0, 1.0],
            )
        })
        .collect::<Vec<_>>();
    let multi = TreeTransformStructure::compile_structures(&groups, &groups, &specs).unwrap();
    assert_eq!(multi.layouts().retained_plan_count(), 16);
    let source = (1..=16 * BLOCK as u32).map(f64::from).collect::<Vec<_>>();
    let mut destination = vec![f64::NAN; source.len()];
    let initialized_scans = scans(&mut |engine| {
        tree_transform_structure_overwrite_with_structural_recoupling_raw(
            engine,
            &mut StridedHostKernelAdapter::default(),
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            &multi,
            &groups,
            &groups,
            &mut destination,
            &source,
            1.0,
            &[],
            1,
        )
        .unwrap();
    });
    assert_eq!(destination, source);
    assert_eq!(initialized_scans, 1);
    let mut owned = None;
    let owned_scans = scans(&mut |engine| {
        owned = try_tree_transform_structure_overwrite_owned_raw(
            engine,
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            &multi,
            &groups,
            &groups,
            1,
            &source,
            1.0,
            1,
            None,
        )
        .unwrap();
    });
    assert_eq!(owned, Some(source));
    assert_eq!(owned_scans, 1);
}

/// Seven one-axis destination blocks of two elements (`offset; stride`):
/// `0;3`, `1;3`, `2;3` interleave in one atom `[0, 5]`; `6;1` alone; `8;2`
/// and `9;2` interleave in `[8, 11]`; `12;1` alone. Block 0 is a Single,
/// blocks 1 and 6 one Multi group (one destination in each of two
/// disconnected atoms), block 2 inactive, block 3 a Single, and blocks 4 and
/// 5 the destinations of two Multi groups sharing their atom.
fn joint_fixture() -> (
    Arc<BlockStructure>,
    Arc<BlockStructure>,
    TreeTransformStructure<f64>,
) {
    let dst = Arc::new(custom_structure(vec![
        block(0, 2, 3, 0),
        block(1, 2, 3, 1),
        block(2, 2, 3, 2),
        block(3, 2, 1, 6),
        block(4, 2, 2, 8),
        block(5, 2, 2, 9),
        block(6, 2, 1, 12),
    ]));
    let src = Arc::new(BlockStructure::packed_column_major(1, vec![vec![2]; 8]).unwrap());
    let transform = TreeTransformStructure::compile_structures(
        &dst,
        &src,
        &[
            TreeTransformBlockSpec::single(0, 0, 2.0),
            TreeTransformBlockSpec::multi(vec![1, 6], vec![1, 2], vec![1.0, 2.0, 3.0, 4.0]),
            TreeTransformBlockSpec::single(3, 3, -1.0),
            TreeTransformBlockSpec::multi(vec![4], vec![4, 5], vec![1.0, -1.0]),
            TreeTransformBlockSpec::multi(vec![5], vec![6, 7], vec![2.0, 1.0]),
        ],
    )
    .unwrap();
    (dst, src, transform)
}

#[test]
fn joint_components_own_every_writer_of_their_atoms() {
    // What: the Single, the Multi group and the inactive layout writing the
    // first atom form one component, which also owns the group's
    // disconnected atom; the two groups sharing an atom form another; the
    // lone Single a third. Every item lies inside the atom of its slot.
    let (dst, _, transform) = joint_fixture();
    let schedule = transform.parallel_schedule();
    assert_eq!(schedule.component_count(), 3);
    let atoms = |component: usize| {
        schedule.atoms[schedule.component_atoms(component)]
            .iter()
            .map(|atom| (atom.lo, atom.len))
            .collect::<Vec<_>>()
    };
    assert_eq!(atoms(0), [(0, 6), (12, 2)]);
    assert_eq!(atoms(1), [(6, 2)]);
    assert_eq!(atoms(2), [(8, 4)]);
    let kinds = |component: usize| {
        let mut kinds = schedule
            .component_direct(component)
            .iter()
            .map(|item| match *item {
                crate::transform_structure::TreeTransformDirectItem::Inactive { .. } => "inactive",
                crate::transform_structure::TreeTransformDirectItem::Single { .. } => "single",
            })
            .collect::<Vec<_>>();
        kinds.sort_unstable();
        kinds
    };
    // Items of one atom are in no particular order: they write disjoint
    // elements.
    assert_eq!(kinds(0), ["inactive", "single"]);
    assert_eq!(kinds(1), ["single"]);
    assert!(kinds(2).is_empty());
    let groups = (0..3)
        .map(|component| schedule.component_group_list(component).len())
        .collect::<Vec<_>>();
    assert_eq!(groups, [1, 0, 2]);
    for group in &schedule.scatter_groups {
        for column in &schedule.scatter_columns[group.columns.clone()] {
            let atom = &schedule.atoms[column.slot as usize];
            let offset = transform.layouts().entry(column.dst_layout).offset;
            assert!(offset >= atom.lo && offset < atom.lo + atom.len as isize);
            assert!(schedule
                .component_atoms(group.component as usize)
                .contains(&(column.slot as usize)));
        }
    }
    assert_eq!(dst.required_len().unwrap(), 14);
}

#[test]
fn component_weight_beyond_usize_still_compiles() {
    // What: a component's moved elements (`n` written plus `3n` packed) may
    // exceed `usize` while every buffer extent fits; compilation accepts it.
    // Metadata only, no tensor data.
    let n = 1usize << (usize::BITS - 2);
    let structure = BlockStructure::packed_column_major(1, [vec![n]]).unwrap();
    let transform = TreeTransformStructure::<f64>::compile_structures(
        &structure,
        &structure,
        &[TreeTransformBlockSpec::multi(
            vec![0],
            vec![0, 0, 0],
            vec![1.0, 1.0, 1.0],
        )],
    )
    .unwrap();
    assert_eq!(
        transform.parallel_schedule().components[0].moved,
        4 * n as u128
    );
}

#[test]
fn inactive_block_ending_beyond_isize_still_compiles() {
    // What: an inactive canonical `[n, 2]` block whose last element `2n - 1`
    // exceeds `isize::MAX` inside the `usize` storage compiles to one atom
    // of its `2n` elements. Metadata only, no tensor data.
    use symtn_core::{FusionTreeHomSpace, U1FusionRule, U1Irrep};
    let n = (1usize << (usize::BITS - 2)) + 1;
    let structure =
        FusionTreeHomSpace::from_sectors([(U1Irrep::new(0), n)], [(U1Irrep::new(0), 2)])
            .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
            .unwrap();
    let transform =
        TreeTransformStructure::<f64>::compile_structures(&structure, &structure, &[]).unwrap();
    let atoms = &transform.parallel_schedule().atoms;
    assert_eq!(atoms.len(), 1);
    assert_eq!((atoms[0].lo, atoms[0].len), (0, 2 * n));
}

#[test]
fn joint_component_replay_matches_the_block_oracle() {
    // What: at one and three workers, an overwrite writes `U * source` per
    // block and zeroes the inactive block that interleaves with active
    // ones; an accumulation scales that block by beta.
    let (dst, src, transform) = joint_fixture();
    let source = (1..=16).map(f64::from).collect::<Vec<_>>();
    // Blocks at 0;3, 1;3, 2;3 (inactive), 6;1, 8;2, 9;2, 12;1.
    let expected = [
        2.0, 13.0, 0.0, 4.0, 16.0, 0.0, -7.0, -8.0, -2.0, 41.0, -2.0, 44.0, 29.0, 36.0,
    ];
    let initial = (0..14).map(|at| f64::from(at) - 3.0).collect::<Vec<_>>();
    let axpby = expected
        .iter()
        .zip(&initial)
        .map(|(value, old)| value + 0.5 * old)
        .collect::<Vec<_>>();
    for threads in [1, 3] {
        let mut overwritten = vec![f64::NAN; 14];
        let mut accumulated = initial.clone();
        crate::host_pool::eager_test_pool(threads).install(|engine| {
            tree_transform_structure_overwrite_with_structural_recoupling_raw(
                engine,
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &transform,
                &dst,
                &src,
                &mut overwritten,
                &source,
                1.0,
                &[],
                threads,
            )
            .unwrap();
            tree_transform_structure_with_structural_recoupling_raw(
                engine,
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &transform,
                &dst,
                &src,
                &mut accumulated,
                &source,
                1.0,
                0.5,
                threads,
            )
            .unwrap();
        });
        assert_eq!(overwritten, expected, "threads {threads}");
        assert_eq!(accumulated, axpby, "threads {threads}");
    }
}

#[test]
fn atom_bands_certify_a_common_outer_slab() {
    // What: an atom holding a Single's and a scatter column's interleaved
    // blocks (two codomain trees of two rows over six columns, rows 4) is
    // bandable by its six columns, and the group's other column, whose
    // block fuses with its packed column to one axis, by its twelve
    // elements; a one-worker replay writes the dense oracle. A
    // unit-row-and-column atom and the injective `[3, 2]` / `[2, 3]`
    // layout, whose inner reach leaves its first slab, stay indivisible, as
    // does an atom whose Single and inactive block disagree on the outer
    // extent.
    let key = |sector| BlockKey::ordinal(sector);
    let dst = Arc::new(
        BlockStructure::from_blocks_with_rank(
            2,
            vec![
                BlockSpec::with_key(key(0), vec![2, 6], vec![1, 4], 0).unwrap(),
                BlockSpec::with_key(key(1), vec![2, 6], vec![1, 4], 2).unwrap(),
                BlockSpec::with_key(key(2), vec![2, 6], vec![1, 2], 24).unwrap(),
            ],
        )
        .unwrap(),
    );
    let src = Arc::new(BlockStructure::packed_column_major(2, vec![vec![2, 6]; 3]).unwrap());
    let transform = TreeTransformStructure::compile_structures(
        &dst,
        &src,
        &[
            TreeTransformBlockSpec::single(0, 0, -2.0),
            TreeTransformBlockSpec::multi(vec![1, 2], vec![1, 2], vec![1.0, 2.0, 3.0, 4.0]),
        ],
    )
    .unwrap();
    let schedule = transform.parallel_schedule();
    assert_eq!(schedule.component_count(), 1);
    let bands = schedule
        .atoms
        .iter()
        .zip(&schedule.bands)
        .map(|(atom, band)| (atom.lo, band.stride, band.extent))
        .collect::<Vec<_>>();
    assert_eq!(bands, [(0, 4, 6), (24, 1, 12)]);
    assert_eq!(schedule.components[0].units, 18);

    let source = (1..=36).map(f64::from).collect::<Vec<_>>();
    let mut expected = vec![0.0; 36];
    for column in 0..6 {
        for row in 0..2 {
            let (a, b) = (source[12 + row + 2 * column], source[24 + row + 2 * column]);
            expected[row + 4 * column] = -2.0 * source[row + 2 * column];
            expected[2 + row + 4 * column] = a + 2.0 * b;
            expected[24 + row + 2 * column] = 3.0 * a + 4.0 * b;
        }
    }
    let mut destination = vec![f64::NAN; 36];
    crate::host_pool::eager_test_pool(1)
        .install(|engine| {
            tree_transform_structure_overwrite_with_structural_recoupling_raw(
                engine,
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &transform,
                &dst,
                &src,
                &mut destination,
                &source,
                1.0,
                &[],
                1,
            )
        })
        .unwrap();
    assert_eq!(destination, expected);

    let single = |shape: Vec<usize>, strides: Vec<usize>| {
        let rank = shape.len();
        let dst = Arc::new(
            BlockStructure::from_blocks_with_rank(
                rank,
                vec![BlockSpec::with_key(key(0), shape.clone(), strides, 0).unwrap()],
            )
            .unwrap(),
        );
        let src = Arc::new(BlockStructure::packed_column_major(rank, vec![shape]).unwrap());
        TreeTransformStructure::compile_structures(
            &dst,
            &src,
            &[TreeTransformBlockSpec::single(0, 0, 1.0)],
        )
        .unwrap()
    };
    let bandable = |transform: &TreeTransformStructure<f64>| {
        transform
            .parallel_schedule()
            .bands
            .iter()
            .map(|band| band.stride != 0)
            .collect::<Vec<_>>()
    };
    assert_eq!(bandable(&single(vec![1, 1], vec![1, 2])), [false]);
    assert_eq!(bandable(&single(vec![3, 2], vec![2, 3])), [false]);
    // A Single `{0, 4, 8}` and an inactive block `{2, 6}` share one atom
    // with outer extents 3 and 2, so the atom stays indivisible.
    let dst = BlockStructure::from_blocks_with_rank(
        2,
        vec![
            BlockSpec::with_key(key(0), vec![1, 3], vec![1, 4], 0).unwrap(),
            BlockSpec::with_key(key(1), vec![1, 2], vec![1, 4], 2).unwrap(),
        ],
    )
    .unwrap();
    let src = BlockStructure::packed_column_major(2, vec![vec![1, 3]]).unwrap();
    let with_inactive = TreeTransformStructure::compile_structures(
        &dst,
        &src,
        &[TreeTransformBlockSpec::single(0, 0, 1.0)],
    )
    .unwrap();
    assert_eq!(bandable(&with_inactive), [false]);
}

#[test]
fn component_replay_matches_the_block_oracle_on_interleaved_su2_blocks() {
    // What: on interleaved canonical blocks whose Multi groups own
    // disconnected atoms, replay writes exactly the per-block oracle for the
    // initialised overwrite, the initialised axpby and the owned writer,
    // at one worker and at an odd and an even width.
    let (structure, transform) = su2_interleaved_fixture();
    let source = (1..=60)
        .map(|value| f64::from(value) * 0.25)
        .collect::<Vec<_>>();
    let initial = (1..=60).map(|value| -f64::from(value)).collect::<Vec<_>>();
    let alpha = 1.5;
    let expected = su2_oracle(&source, alpha);
    let axpby = expected
        .iter()
        .zip(&initial)
        .map(|(value, old)| value + 0.5 * old)
        .collect::<Vec<_>>();
    for threads in [1, 3, 4] {
        let mut overwritten = vec![f64::NAN; 60];
        let mut accumulated = initial.clone();
        let owned = crate::host_pool::test_pool(threads).install(|engine| {
            tree_transform_structure_overwrite_with_structural_recoupling_raw(
                engine,
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &transform,
                &structure,
                &structure,
                &mut overwritten,
                &source,
                alpha,
                &[],
                threads,
            )
            .unwrap();
            tree_transform_structure_with_structural_recoupling_raw(
                engine,
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &transform,
                &structure,
                &structure,
                &mut accumulated,
                &source,
                alpha,
                0.5,
                threads,
            )
            .unwrap();
            try_tree_transform_structure_overwrite_owned_raw(
                engine,
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &transform,
                &structure,
                &structure,
                2,
                &source,
                alpha,
                threads,
                None,
            )
            .unwrap()
        });
        assert_eq!(overwritten, expected, "threads {threads}");
        assert_eq!(accumulated, axpby, "threads {threads}");
        assert_eq!(owned, Some(expected.clone()), "threads {threads}");
    }
}
