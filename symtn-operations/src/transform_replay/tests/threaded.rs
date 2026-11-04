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
    assert_eq!(structure.parallel_schedule().component_count(), 1);
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
    assert!(transform.parallel_schedule().direct.is_empty());
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
    assert_eq!(transform.parallel_schedule().component_count(), 1);
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

/// What: in a serial replay, which admits no threaded chunk, the leaf gate's
/// dispatch-cost scan runs only when an overwrite of a role with a retained
/// plan reaches leaf admission, once per call on one token: planless and
/// accumulating calls scan nothing, and an overwrite of planned roles scans
/// exactly once on a wide engine, however many block lists (one Single list,
/// one scatter list per Multi block) the call visits. (A threaded chunk
/// scatter also asks the gate whether its moved elements pay a worker.)
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
            &mut super::Staged::new(),
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
            &mut super::Staged::new(),
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
            &mut super::Staged::new(),
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
            &mut super::Staged::new(),
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
            &mut super::Staged::new(),
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
fn joint_component_replay_matches_the_block_oracle() {
    // What: on both routes and at one and three workers, an overwrite
    // writes `U * source` per block and zeroes the inactive block that
    // interleaves with active ones; an accumulation scales that block by
    // beta in its lane.
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
        let mut pool = crate::host_pool::eager_test_pool(threads);
        for lanes in [true, false] {
            fn replay<E: DenseExecutor>(
                engine: &mut dyn Engine,
                dense: &mut E,
                (transform, dst, src, source): (
                    &TreeTransformStructure<f64>,
                    &Arc<BlockStructure>,
                    &Arc<BlockStructure>,
                    &[f64],
                ),
                dst_data: &mut [f64],
                beta: Option<f64>,
                threads: usize,
            ) {
                match beta {
                    None => tree_transform_structure_overwrite_with_structural_recoupling_raw(
                        engine,
                        &mut StridedHostKernelAdapter::default(),
                        dense,
                        &mut TreeTransformWorkspace::default(),
                        transform,
                        dst,
                        src,
                        dst_data,
                        source,
                        1.0,
                        &[],
                        threads,
                    ),
                    Some(beta) => tree_transform_structure_with_structural_recoupling_raw(
                        engine,
                        &mut StridedHostKernelAdapter::default(),
                        dense,
                        &mut TreeTransformWorkspace::default(),
                        transform,
                        dst,
                        src,
                        dst_data,
                        source,
                        1.0,
                        beta,
                        threads,
                    ),
                }
                .unwrap();
            }
            let parts = (&transform, &dst, &src, &source[..]);
            let run = |engine: &mut dyn Engine, dst_data: &mut [f64], beta: Option<f64>| {
                if lanes {
                    replay(
                        engine,
                        &mut DefaultDenseExecutor::new(),
                        parts,
                        dst_data,
                        beta,
                        threads,
                    );
                } else {
                    replay(
                        engine,
                        &mut super::Staged::new(),
                        parts,
                        dst_data,
                        beta,
                        threads,
                    );
                }
            };
            let mut overwritten = vec![f64::NAN; 14];
            let mut accumulated = initial.clone();
            pool.install(|engine| {
                run(engine, &mut overwritten, None);
                run(engine, &mut accumulated, Some(0.5));
            });
            assert_eq!(overwritten, expected, "threads {threads}, lanes {lanes}");
            assert_eq!(accumulated, axpby, "threads {threads}, lanes {lanes}");
        }
    }
}

/// Records, for every element a copy writes, its address in the
/// destination starting at `base` and the writing thread.
#[derive(Clone)]
struct WriterRecorder {
    inner: StridedHostKernelAdapter,
    base: usize,
    writes: Arc<std::sync::Mutex<Vec<(usize, std::thread::ThreadId)>>>,
}

impl crate::kernel_adapter::sealed::Sealed for WriterRecorder {}

impl HostKernelAdapter<f64> for WriterRecorder {
    fn add_strided_baked(
        &mut self,
        exec: Exec<'_>,
        zero_strides: &mut Vec<isize>,
        dst_data: &mut [f64],
        src_data: &[f64],
        shape: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        dst_offset: isize,
        src_offset: isize,
        source_conjugate: bool,
        alpha: f64,
        beta: f64,
        baked: Option<crate::BakedFusedLayout<'_>>,
        index: Option<&mut [usize]>,
    ) -> Result<(), OperationError> {
        self.inner.add_strided_baked(
            exec,
            zero_strides,
            dst_data,
            src_data,
            shape,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            source_conjugate,
            alpha,
            beta,
            baked,
            index,
        )
    }

    fn axpby_strided_baked(
        &mut self,
        exec: Exec<'_>,
        dst_data: &mut [f64],
        src_data: &[f64],
        shape: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        dst_offset: isize,
        src_offset: isize,
        alpha: f64,
        beta: f64,
        baked: Option<crate::BakedFusedLayout<'_>>,
        index: Option<&mut [usize]>,
    ) -> Result<(), OperationError> {
        self.inner.axpby_strided_baked(
            exec,
            dst_data,
            src_data,
            shape,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            alpha,
            beta,
            baked,
            index,
        )
    }

    fn copy_scale_strided_baked(
        &mut self,
        exec: Exec<'_>,
        dst_data: &mut [f64],
        src_data: &[f64],
        shape: &[usize],
        dst_strides: &[isize],
        src_strides: &[isize],
        dst_offset: isize,
        src_offset: isize,
        source_conjugate: bool,
        alpha: f64,
        baked: Option<crate::BakedFusedLayout<'_>>,
        index: Option<&mut [usize]>,
    ) -> Result<(), OperationError> {
        let start = (dst_data.as_ptr() as usize - self.base) / std::mem::size_of::<f64>();
        let thread = std::thread::current().id();
        let mut addresses = vec![0isize];
        for (&dim, &stride) in shape.iter().zip(dst_strides) {
            addresses = addresses
                .iter()
                .flat_map(|&at| (0..dim as isize).map(move |step| at + step * stride))
                .collect();
        }
        self.writes.lock().unwrap().extend(
            addresses
                .into_iter()
                .map(|at| ((start as isize + dst_offset + at) as usize, thread)),
        );
        self.inner.copy_scale_strided_baked(
            exec,
            dst_data,
            src_data,
            shape,
            dst_strides,
            src_strides,
            dst_offset,
            src_offset,
            source_conjugate,
            alpha,
            baked,
            index,
        )
    }

    fn scale_strided(
        &mut self,
        exec: Exec<'_>,
        dst_data: &mut [f64],
        shape: &[usize],
        dst_strides: &[isize],
        dst_offset: isize,
        beta: f64,
    ) -> Result<(), OperationError> {
        self.inner
            .scale_strided(exec, dst_data, shape, dst_strides, dst_offset, beta)
    }

    fn recoupling_src_times_u_transpose<C>(
        &mut self,
        exec: Exec<'_>,
        destination: &mut [f64],
        source: &[f64],
        recoupling_coefficients_dst_src: &[C],
        element_count: usize,
        src_count: usize,
        dst_count: usize,
    ) -> Result<(), OperationError>
    where
        C: Copy,
        f64: RecouplingCoefficientAction<C>,
    {
        self.inner.recoupling_src_times_u_transpose(
            exec,
            destination,
            source,
            recoupling_coefficients_dst_src,
            element_count,
            src_count,
            dst_count,
        )
    }
}

#[test]
#[cfg_attr(
    any(
        feature = "cpu-blas",
        feature = "blas-accelerate",
        feature = "blas-openblas",
        feature = "blas-mkl",
        feature = "provider-inject"
    ),
    ignore = "lane GEMMs are the native provider's; a BLAS provider runs the staged route"
)]
fn banded_components_write_each_band_from_one_slot_in_every_stage() {
    // What: a component whose atom holds a Single's and a scatter column's
    // interleaved blocks (two codomain trees of two rows over six columns,
    // rows 4) and whose group's other column has its own atom runs banded
    // on three slots: every band (column) of each atom, interior ones
    // included, is written from one thread by both the Single and the
    // scatter, three threads in all, and the values are the dense oracle.
    // A unit-row-and-column atom and the injective `[3, 2]` / `[2, 3]`
    // layout, whose inner reach leaves its first slab, stay indivisible.
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
        .map(|atom| (atom.lo, atom.band_stride, atom.band_extent))
        .collect::<Vec<_>>();
    // The second atom's block fuses with its packed column to one axis.
    assert_eq!(bands, [(0, 4, 6), (24, 1, 12)]);

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
    let writes = Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut kernels = WriterRecorder {
        inner: StridedHostKernelAdapter::default(),
        base: destination.as_ptr() as usize,
        writes: Arc::clone(&writes),
    };
    crate::host_pool::eager_test_pool(3)
        .install(|engine| {
            tree_transform_structure_overwrite_with_structural_recoupling_raw(
                engine,
                &mut kernels,
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &transform,
                &dst,
                &src,
                &mut destination,
                &source,
                1.0,
                &[],
                3,
            )
        })
        .unwrap();
    assert_eq!(destination, expected);
    let writes = writes.lock().unwrap().clone();
    assert_eq!(writes.len(), 36);
    let band = |at: usize| if at < 24 { at / 4 } else { 6 + (at - 24) / 2 };
    let mut writer_of_band = std::collections::HashMap::new();
    for &(at, thread) in &writes {
        assert_eq!(
            *writer_of_band.entry(band(at)).or_insert(thread),
            thread,
            "band {}",
            band(at)
        );
    }
    let threads = writer_of_band
        .values()
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(threads.len(), 3);

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
            .atoms
            .iter()
            .map(|atom| atom.bandable())
            .collect::<Vec<_>>()
    };
    assert_eq!(bandable(&single(vec![1, 1], vec![1, 2])), [false]);
    assert_eq!(bandable(&single(vec![3, 2], vec![2, 3])), [false]);
}

#[test]
fn component_replay_matches_the_block_oracle_at_odd_and_even_widths() {
    // What: on interleaved canonical blocks whose Multi groups own
    // disconnected atoms, threaded replay over ownership components writes
    // exactly the per-block oracle for the initialised overwrite, the
    // initialised axpby and the owned writer, at an odd and an even width.
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
    for threads in [3, 4] {
        let mut pool = crate::host_pool::test_pool(threads);
        assert!(
            pool.install(|engine| {
                effective_tree_transform_threads(engine, transform.parallel_schedule(), threads)
            }) == threads
        );
        let mut overwritten = vec![f64::NAN; 60];
        let mut accumulated = initial.clone();
        let owned = pool.install(|engine| {
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
        assert_eq!(overwritten, expected);
        assert_eq!(accumulated, axpby);
        assert_eq!(owned, Some(expected.clone()));
    }
}

#[test]
fn component_splits_balance_moved_elements_not_component_count() {
    // What: a split gives the left `threads / 2` lanes the legal component
    // boundary whose moved elements are closest to that fraction of the
    // total; ties go to the earlier boundary and a component is never
    // divided.
    let split = |weights: &[usize], threads: usize| {
        let before = std::iter::once(0)
            .chain(weights.iter().scan(0, |sum, &weight| {
                *sum += weight;
                Some(*sum)
            }))
            .collect::<Vec<_>>();
        balanced_boundary(weights.len(), threads, |part| before[part])
    };
    // Odd width: one of three lanes targets a third of 9 elements, which
    // the first three components hold; a count split would take two.
    assert_eq!(split(&[1, 1, 1, 6], 3), 3);
    // One heavy component goes left whole rather than past the target.
    assert_eq!(split(&[10, 1, 1], 2), 1);
    assert_eq!(split(&[1, 1, 10], 2), 2);
    // 1 and 3 are equally far from half of 4: the earlier boundary.
    assert_eq!(split(&[1, 2, 1], 2), 1);
}

#[test]
fn only_components_that_use_every_worker_run_on_the_whole_token() {
    use crate::kernel_adapter::observed::ObservedKernels;
    use strided_kernel::execution::{parallel_width, KernelClass};
    // What: in a threaded overwrite on eight workers, a Single or Multi
    // component whose blocks Strided would give all eight workers runs alone
    // on the whole token, while a smaller widening component and the tiny
    // ones of the same phase split the token and each runs on its lane's
    // share; every block holds its oracle value. The moldable size is the first count Strided widens at
    // all, so the expectation for it follows Strided's own answer. The tiny
    // blocks hold the largest count Strided moves on one worker, and each
    // phase has two or three of them, so the split is cheaper than running
    // the narrow components one after another on the whole token
    // (`split_pays`).
    const THREADS: usize = 8;
    let large = first_count(THREADS, |width| width == THREADS);
    let mut pool = crate::host_pool::test_pool(THREADS);
    let (moldable, moldable_saturates) = pool.with_exec(|exec| {
        let width = |n| parallel_width(&exec, KernelClass::Copy, n).get();
        assert_eq!(width(large), THREADS);
        let (mut below, mut at) = (0usize, large);
        while at - below > 1 {
            let middle = below + (at - below) / 2;
            if width(middle) > 1 {
                at = middle;
            } else {
                below = middle;
            }
        }
        (at, width(at) == THREADS)
    });
    let tiny = moldable - 1;
    let structure = Arc::new(
        BlockStructure::packed_column_major(
            1,
            [
                tiny, tiny, tiny, tiny, moldable, moldable, large, large, tiny, tiny, tiny,
                moldable, large,
            ]
            .map(|len| vec![len]),
        )
        .unwrap(),
    );
    let matrix = [2.0, -1.0, 0.5, 3.0];
    let transform = TreeTransformStructure::compile_structures(
        &structure,
        &structure,
        &[0, 2, 4, 6]
            .map(|first| {
                TreeTransformBlockSpec::multi(
                    vec![first, first + 1],
                    vec![first, first + 1],
                    matrix.to_vec(),
                )
            })
            .into_iter()
            .chain((8..13).map(|block| TreeTransformBlockSpec::single(block, block, 1.5)))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    assert_eq!(transform.parallel_schedule().component_count(), 9);
    let len = structure.required_len().unwrap();
    let source = (0..len)
        .map(|value| (value % 97) as f64 - 48.0)
        .collect::<Vec<_>>();
    let mut expected = vec![0.0; len];
    let mut start = 0;
    for block_len in [tiny, tiny, moldable, large] {
        for element in 0..block_len {
            let (a, b) = (source[start + element], source[start + block_len + element]);
            expected[start + element] = matrix[0] * a + matrix[1] * b;
            expected[start + block_len + element] = matrix[2] * a + matrix[3] * b;
        }
        start += 2 * block_len;
    }
    for at in start..len {
        expected[at] = 1.5 * source[at];
    }
    let mut kernels = ObservedKernels::recording();
    let budgets = std::sync::Arc::clone(&kernels.write_budgets);
    let mut destination = vec![f64::NAN; len];
    pool.install(|engine| {
        tree_transform_structure_overwrite_with_structural_recoupling_raw(
            engine,
            &mut kernels,
            &mut super::Staged::new(),
            &mut TreeTransformWorkspace::default(),
            &transform,
            &structure,
            &structure,
            &mut destination,
            &source,
            1.0,
            &[],
            THREADS,
        )
    })
    .unwrap();
    assert!(destination == expected);
    let budgets = budgets.lock().unwrap().clone();
    assert_eq!(budgets.len(), 13);
    for (elements, budget) in budgets {
        let whole = elements == large || (elements == moldable && moldable_saturates);
        assert_eq!(budget == THREADS, whole, "{elements} elements on {budget}");
        // No lane holds more workers than Strided would use for its block.
        let width = pool.with_exec(|exec| parallel_width(&exec, KernelClass::Copy, elements).get());
        assert!(
            budget <= width,
            "{elements} elements on {budget} of width {width}"
        );
    }
}

#[test]
fn chunks_too_small_to_pay_a_worker_scatter_without_forking() {
    // What: a threaded Multi replay whose every chunk moves too few elements
    // for Strided's rule to give a second worker scatters each chunk inline,
    // so the number of scatter forks does not grow with the chunk count;
    // the output still holds `U * source`.
    let matrix = [2.0, -1.0, 0.5, 3.0];
    let splits = |groups: usize| {
        let structure =
            Arc::new(BlockStructure::packed_column_major(1, vec![vec![2]; 2 * groups]).unwrap());
        let transform = TreeTransformStructure::compile_structures(
            &structure,
            &structure,
            &(0..groups)
                .map(|group| {
                    let blocks = vec![2 * group, 2 * group + 1];
                    TreeTransformBlockSpec::multi(blocks.clone(), blocks, matrix.to_vec())
                })
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let source = (0..4 * groups)
            .map(|value| value as f64)
            .collect::<Vec<_>>();
        let expected = source
            .chunks(4)
            .flat_map(|block| {
                let (a, b) = ([block[0], block[1]], [block[2], block[3]]);
                [
                    matrix[0] * a[0] + matrix[1] * b[0],
                    matrix[0] * a[1] + matrix[1] * b[1],
                    matrix[2] * a[0] + matrix[3] * b[0],
                    matrix[2] * a[1] + matrix[3] * b[1],
                ]
            })
            .collect::<Vec<_>>();
        let mut destination = vec![f64::NAN; 4 * groups];
        let before = SCATTER_SPLITS.with(|splits| splits.get());
        crate::host_pool::test_pool(2)
            .install(|engine| {
                tree_transform_structure_overwrite_with_structural_recoupling_raw(
                    engine,
                    &mut StridedHostKernelAdapter::default(),
                    &mut super::Staged::new(),
                    &mut TreeTransformWorkspace::default(),
                    &transform,
                    &structure,
                    &structure,
                    &mut destination,
                    &source,
                    1.0,
                    &[],
                    2,
                )
            })
            .unwrap();
        assert_eq!(destination, expected);
        SCATTER_SPLITS.with(|splits| splits.get()) - before
    };
    // Two workers chunk two groups at a time, 2 and 6 chunks, each moving
    // eight elements: fewer than pay a second worker.
    assert!(first_count(2, |width| width > 1) > 8);
    assert_eq!(splits(4), splits(12));
}

#[test]
fn a_split_uses_no_more_lanes_than_its_components_can() {
    // What: two components that move enough to pay a split but whose blocks
    // never widen share two of three workers, one each.
    let moved = first_count(3, |width| width == 3);
    let (lanes, widths) = crate::host_pool::test_pool(3).with_exec(|exec| {
        let mut widths = [0, 0];
        let [first, second] = &mut widths;
        let lanes = split_workers(
            &exec,
            &LeafGate::new(1),
            &mut Vec::new(),
            3,
            2 * moved,
            [(1, first), (1, second)],
        );
        (lanes, widths)
    });
    assert_eq!((lanes, widths), (2, [1, 1]));
}

/// The smallest count whose copy Strided gives a width satisfying `holds` on
/// a `threads`-worker test pool, by doubling then bisection (the predicates
/// used here are monotone on the pool's ordinary cost row).
pub(super) fn first_count(threads: usize, holds: impl Fn(usize) -> bool) -> usize {
    use strided_kernel::execution::{parallel_width, KernelClass};
    crate::host_pool::test_pool(threads).with_exec(|exec| {
        let meets = |n| holds(parallel_width(&exec, KernelClass::Copy, n).get());
        let mut at = 1usize;
        while !meets(at) {
            at = at
                .checked_mul(2)
                .expect("Strided reaches the width at some count");
        }
        let mut below = at / 2;
        if below == 0 {
            return at;
        }
        while at - below > 1 {
            let middle = below + (at - below) / 2;
            if meets(middle) {
                at = middle;
            } else {
                below = middle;
            }
        }
        at
    })
}

/// The eight-worker cost row under which a tiny column widens to two
/// workers but no further, and the smallest column counts Strided gives two
/// and all eight workers under it.
fn intermediate_width_pool() -> (crate::host_pool::TestPool, usize, usize) {
    use std::time::Duration;
    use strided_kernel::execution::{parallel_width, KernelClass};
    use tenferro_cpu::threadteam::WidthCosts;
    let row = WidthCosts::from_anchors(&[
        (2, Duration::from_nanos(1)),
        (3, Duration::from_nanos(400)),
        (8, Duration::from_nanos(400)),
    ])
    .unwrap();
    let mut pool = crate::host_pool::test_pool_with_costs(8, row);
    let (two, eight) = pool.with_exec(|exec| {
        let width = |n| parallel_width(&exec, KernelClass::Copy, n).get();
        let first = |workers: usize| {
            let mut at = 1usize;
            while width(at) < workers {
                at *= 2;
            }
            let (mut below, mut at) = (at / 2, at);
            while at - below > 1 {
                let middle = below + (at - below) / 2;
                if width(middle) >= workers {
                    at = middle;
                } else {
                    below = middle;
                }
            }
            at
        };
        let (two, eight) = (first(2), first(8));
        assert_eq!((width(two), width(16 * two), width(eight)), (2, 2, 8));
        (two, eight)
    });
    (pool, two, eight)
}

/// Replays `chunks` identical chunks of eight two-column Multi groups whose
/// columns hold `columns[g]` elements on `pool`'s eight workers, checks
/// `U * source`, and returns the Strided width searches and gate threshold
/// setups the calling thread ran.
fn replay_chunks(
    pool: &mut crate::host_pool::TestPool,
    columns: [usize; 8],
    chunks: usize,
) -> (usize, usize) {
    use crate::kernel_adapter::sealed::{GATE_SCANS, WIDTH_QUERIES};
    let matrix = [2.0, -1.0, 0.5, 3.0];
    let sizes = (0..chunks)
        .flat_map(|_| columns)
        .flat_map(|column| [column, column])
        .collect::<Vec<_>>();
    let structure = Arc::new(
        BlockStructure::packed_column_major(1, sizes.iter().map(|&len| vec![len])).unwrap(),
    );
    let groups = 8 * chunks;
    let transform = TreeTransformStructure::compile_structures(
        &structure,
        &structure,
        &(0..groups)
            .map(|group| {
                let blocks = vec![2 * group, 2 * group + 1];
                TreeTransformBlockSpec::multi(blocks.clone(), blocks, matrix.to_vec())
            })
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let len = structure.required_len().unwrap();
    let source = (0..len)
        .map(|value| (value % 31) as f64)
        .collect::<Vec<_>>();
    let mut expected = Vec::with_capacity(len);
    let mut start = 0;
    for pair in sizes.chunks(2) {
        let column = pair[0];
        let (a, b) = source[start..start + 2 * column].split_at(column);
        expected.extend(a.iter().zip(b).map(|(a, b)| matrix[0] * a + matrix[1] * b));
        expected.extend(a.iter().zip(b).map(|(a, b)| matrix[2] * a + matrix[3] * b));
        start += 2 * column;
    }
    let mut destination = vec![f64::NAN; len];
    let before = (
        WIDTH_QUERIES.with(|queries| queries.get()),
        GATE_SCANS.with(|scans| scans.get()),
    );
    pool.install(|engine| {
        tree_transform_structure_overwrite_with_structural_recoupling_raw(
            engine,
            &mut StridedHostKernelAdapter::default(),
            &mut super::Staged::new(),
            &mut TreeTransformWorkspace::default(),
            &transform,
            &structure,
            &structure,
            &mut destination,
            &source,
            1.0,
            &[],
            8,
        )
    })
    .unwrap();
    assert!(destination == expected);
    (
        WIDTH_QUERIES.with(|queries| queries.get()) - before.0,
        GATE_SCANS.with(|scans| scans.get()) - before.1,
    )
}

#[test]
fn repeated_chunks_reuse_the_calls_width_decisions() {
    // What: 1 or 10 identical chunks of components that widen to two of
    // eight workers ask Strided for widths, and set up the gate's
    // thresholds, equally often: the call's width answers are reused.
    let (mut pool, two, _) = intermediate_width_pool();
    let one = replay_chunks(&mut pool, [two; 8], 1);
    assert!(one.0 > 0 && one.1 > 0);
    assert_eq!(replay_chunks(&mut pool, [two; 8], 10), one);
}

#[test]
fn a_narrowed_lane_keeps_the_whole_token_gate() {
    use crate::kernel_adapter::sealed::GATE_SCANS;
    // What: a gate asked alternately on the call's whole token and on a lane
    // narrowed to two workers (as a chunk's wide components and its one
    // narrowed remainder ask it) sets up each token shape's thresholds once,
    // however many chunks alternate. (Driven directly: the recoupling plan
    // orders Multi groups by size, so compiled chunks cannot repeat a mix of
    // saturating groups and one narrow remainder.)
    let (mut pool, two, eight) = intermediate_width_pool();
    let scans = pool.with_exec(|mut exec| {
        let gate = LeafGate::new(eight);
        let before = GATE_SCANS.with(|scans| scans.get());
        for _ in 0..10 {
            assert!(gate.saturates(&exec, eight));
            let narrowed = granted_width(exec.reborrow(), 2);
            assert!(gate.widens(&narrowed, two));
        }
        GATE_SCANS.with(|scans| scans.get()) - before
    });
    assert_eq!(scans, 2);
}

#[test]
fn whole_token_widths_from_the_call_table_match_strided() {
    use std::time::Duration;
    use strided_kernel::execution::{parallel_width, KernelClass};
    use tenferro_cpu::threadteam::WidthCosts;
    // What: the call's width-transition table answers every count's copy
    // width exactly as Strided does, for a plain, a free-dispatch and an
    // intermediate-width cost row, below, around and past the gate's bound.
    let row = WidthCosts::from_anchors(&[
        (2, Duration::from_nanos(1)),
        (3, Duration::from_nanos(400)),
        (8, Duration::from_nanos(400)),
    ])
    .unwrap();
    let counts = (0..2048)
        .chain((11..=34).flat_map(|shift| [(1usize << shift) - 1, 1 << shift, (1 << shift) + 1]))
        .collect::<Vec<_>>();
    for mut pool in [
        crate::host_pool::test_pool(8),
        crate::host_pool::eager_test_pool(4),
        crate::host_pool::test_pool_with_costs(8, row),
    ] {
        pool.with_exec(|exec| {
            for bound in [0, 4096, usize::MAX] {
                let gate = LeafGate::new(bound);
                let mut levels = Vec::new();
                for &n in &counts {
                    assert_eq!(
                        whole_width(&exec, &gate, &mut levels, n),
                        parallel_width(&exec, KernelClass::Copy, n).get(),
                        "elements {n}, bound {bound}"
                    );
                }
            }
        });
    }
}

#[test]
fn independent_components_split_up_to_their_summed_widths() {
    // What: eight components that each use one worker split over all eight
    // lanes even though Strided gives their total moved elements, as one
    // copy, only two: the phase's width is that of its components side by
    // side, not of one copy.
    let (mut pool, two, _) = intermediate_width_pool();
    let (lanes, total_width) = pool.with_exec(|exec| {
        let gate = LeafGate::new(1);
        let mut levels = Vec::new();
        let mut widths = [0; 8];
        let total = 8 * two;
        let lanes = split_workers(
            &exec,
            &gate,
            &mut levels,
            8,
            total,
            widths.iter_mut().map(|width| (1, width)),
        );
        (lanes, whole_width(&exec, &gate, &mut levels, total))
    });
    assert_eq!((lanes, total_width), (8, 2));
}

/// The native executor whose lane GEMMs of `failing` jobs fail with an
/// error naming the job.
struct FailingLanes {
    inner: DefaultDenseExecutor,
    failing: &'static [usize],
}

struct FailOn<'g> {
    inner: &'g dyn symtn_dense::LaneGemms,
    failing: &'static [usize],
}

impl symtn_dense::LaneGemms for FailOn<'_> {
    fn run<'d>(
        &self,
        token: symtn_dense::LaneToken<'_>,
        job: usize,
        io: symtn_dense::LaneGemmIo<'_, 'd>,
    ) -> Result<(), DenseError> {
        if self.failing.contains(&job) {
            return Err(DenseError::Backend {
                backend: DenseBackend::Tenferro,
                op: "lane",
                message: format!("job {job}"),
            });
        }
        self.inner.run(token, job, io)
    }

    fn time(
        &self,
        token: &mut symtn_dense::LaneToken<'_>,
        job: usize,
        width: core::num::NonZeroUsize,
    ) -> Duration {
        self.inner.time(token, job, width)
    }
}

impl DenseExecutor for FailingLanes {
    fn svd(
        &mut self,
        e: &mut dyn Engine,
        i: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.svd(e, i)
    }
    fn qr(&mut self, e: &mut dyn Engine, i: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.qr(e, i)
    }
    fn eigh(
        &mut self,
        e: &mut dyn Engine,
        i: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.eigh(e, i)
    }
    fn dot_general_into(
        &mut self,
        e: &mut dyn Engine,
        o: DenseWrite<'_>,
        l: DenseRead<'_>,
        r: DenseRead<'_>,
        c: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        self.inner.dot_general_into(e, o, l, r, c)
    }
    fn supports_lane_gemms(&self) -> bool {
        true
    }
    fn with_lane_gemms(
        &mut self,
        engine: &mut dyn Engine,
        dtype: symtn_dense::DenseDType,
        jobs: &[symtn_dense::LaneGemmShape],
        body: &mut symtn_dense::LaneGemmBody<'_>,
    ) -> Result<(), DenseError> {
        let failing = self.failing;
        self.inner
            .with_lane_gemms(engine, dtype, jobs, &mut |token, gemms| {
                body(
                    token,
                    &FailOn {
                        inner: gemms,
                        failing,
                    },
                )
            })
    }
}

#[test]
#[cfg_attr(
    any(
        feature = "cpu-blas",
        feature = "blas-accelerate",
        feature = "blas-openblas",
        feature = "blas-mkl",
        feature = "provider-inject"
    ),
    ignore = "lane GEMMs are the native provider's; a BLAS provider runs the staged route"
)]
fn a_failing_lane_replay_returns_the_first_error_by_tier() {
    // What: on four workers, three one-element groups split over lanes and
    // an eight-element group runs alone on the whole token (first). Two
    // failing split groups return the lower one's error; a failing alone
    // group's error wins over a lower failing split group's. A failing
    // native owned replay returns the error and publishes nothing.
    let structure = Arc::new(
        BlockStructure::packed_column_major(1, [1, 1, 1, 1, 1, 1, 8, 8].map(|len| vec![len]))
            .unwrap(),
    );
    let transform = TreeTransformStructure::compile_structures(
        &structure,
        &structure,
        &[0, 2, 4, 6].map(|first| {
            TreeTransformBlockSpec::multi(
                vec![first, first + 1],
                vec![first, first + 1],
                vec![1.0, 0.0, 0.0, 1.0],
            )
        }),
    )
    .unwrap();
    let source = (1..=22).map(f64::from).collect::<Vec<_>>();
    let mut pool = crate::host_pool::eager_test_pool(4);
    const SPLIT: &[usize] = &[1, 2];
    const TIERS: &[usize] = &[0, 3];
    for (failing, error) in [(SPLIT, "job 1"), (TIERS, "job 3")] {
        let mut destination = vec![0.0; 22];
        let result = pool.install(|engine| {
            tree_transform_structure_overwrite_with_structural_recoupling_raw(
                engine,
                &mut StridedHostKernelAdapter::default(),
                &mut FailingLanes {
                    inner: DefaultDenseExecutor::new(),
                    failing,
                },
                &mut TreeTransformWorkspace::default(),
                &transform,
                &structure,
                &structure,
                &mut destination,
                &source,
                1.0,
                &[],
                4,
            )
        });
        match result {
            Err(OperationError::Dense(DenseError::Backend { message, .. })) => {
                assert_eq!(message, error, "failing {failing:?}")
            }
            other => panic!("failing {failing:?}: {other:?}"),
        }
    }
    // A native owned replay whose lane GEMM fails publishes nothing.
    let (su2, su2_transform) = su2_interleaved_fixture();
    let su2_source = (1..=60).map(f64::from).collect::<Vec<_>>();
    const FIRST: &[usize] = &[0];
    let owned = pool.install(|engine| {
        try_tree_transform_structure_overwrite_owned_raw(
            engine,
            &mut FailingLanes {
                inner: DefaultDenseExecutor::new(),
                failing: FIRST,
            },
            &mut TreeTransformWorkspace::default(),
            &su2_transform,
            &su2,
            &su2,
            2,
            &su2_source,
            1.0,
            4,
            None,
        )
    });
    match owned {
        Err(OperationError::Dense(DenseError::Backend { message, .. })) => {
            assert_eq!(message, "job 0")
        }
        other => panic!("owned: {other:?}"),
    }
}
