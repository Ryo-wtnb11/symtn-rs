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
    assert_eq!(structure.parallel_schedule().single_component_count(), 1);
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
    assert_eq!(transform.parallel_schedule().scatter_component_count(), 1);
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

#[test]
fn ownership_components_own_disconnected_atoms_exclusively() {
    // What: overlapping destination intervals merge into address-ordered
    // atoms, and a Multi group owns every atom its columns write, however
    // far apart: each group's two blocks give two atoms with other groups'
    // atoms between them, every atom has one slot inside its own component's
    // slot range, and every column lies inside its atom.
    let (_, transform) = su2_interleaved_fixture();
    let schedule = transform.parallel_schedule();

    // Singles: (0, 0) [4, 9] and (1, 0) [6, 11] share one atom.
    let single_atoms = schedule
        .single_atoms
        .iter()
        .map(|atom| (atom.lo, atom.item, atom.weight_before))
        .collect::<Vec<_>>();
    assert_eq!(
        single_atoms,
        [
            (4, 0, 0),
            (12, 2, 8),
            (22, 3, 12),
            (28, 4, 16),
            (38, 5, 20),
            (44, 6, 24),
            (50, 7, 28),
        ]
    );

    let atoms = schedule
        .scatter_atoms
        .iter()
        .map(|atom| (atom.lo, atom.len))
        .collect::<Vec<_>>();
    assert_eq!(
        atoms,
        [
            (0, 4),
            (14, 6),
            (20, 6),
            (30, 6),
            (36, 6),
            (46, 6),
            (52, 4),
            (56, 4)
        ]
    );
    let mut slots = schedule
        .scatter_atoms
        .iter()
        .map(|atom| atom.slot)
        .collect::<Vec<_>>();
    slots.sort_unstable();
    assert_eq!(slots, (0..8).collect::<Vec<u32>>());
    assert_eq!(schedule.scatter_component_count(), 4);

    let layouts = transform.layouts();
    let mut components = Vec::new();
    for group in &schedule.scatter_groups {
        let slot_range = schedule.component_slots[group.component as usize]
            ..schedule.component_slots[group.component as usize + 1];
        let mut owned = Vec::new();
        for column in &schedule.scatter_columns[group.columns.clone()] {
            assert!(slot_range.contains(&column.slot));
            let atom = schedule
                .scatter_atoms
                .iter()
                .find(|atom| atom.slot == column.slot)
                .unwrap();
            let offset = layouts.entry(column.dst_layout).offset;
            assert_eq!(offset - atom.lo, column.local_base as isize);
            let [.., last] = su2_addresses(offset as usize);
            assert!(last < atom.lo as usize + atom.len);
            owned.push((offset, atom.lo));
        }
        owned.sort_unstable();
        assert_eq!(slot_range.len(), owned.len());
        components.push((group.component, owned));
    }
    components.sort_unstable();
    let expected = SU2_MULTIS
        .iter()
        .map(|(offsets, _)| offsets.map(|offset| offset as isize))
        .map(|[left, right]| {
            let atom = |offset: isize| atoms.iter().rfind(|&&(lo, _)| lo <= offset).unwrap().0;
            vec![(left, atom(left)), (right, atom(right))]
        })
        .collect::<Vec<_>>();
    assert_eq!(
        components,
        expected
            .into_iter()
            .enumerate()
            .map(|(component, owned)| (component as u32, owned))
            .collect::<Vec<_>>()
    );
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
fn multi_groups_sharing_an_atom_replay_as_one_component() {
    // What: two Multi groups whose interleaved blocks share one atom own it
    // together, apart from a third group's separate atom, and threaded
    // replay writes the independent physical-output oracle.
    let src_structure = Arc::new(
        BlockStructure::packed_column_major(1, [vec![2], vec![2], vec![2], vec![2]]).unwrap(),
    );
    let dst_structure = Arc::new(custom_structure(vec![
        block(0, 2, 3, 0),
        block(1, 2, 3, 1),
        block(2, 2, 3, 2),
        block(3, 2, 1, 6),
    ]));
    let transform = TreeTransformStructure::compile_structures(
        &dst_structure,
        &src_structure,
        &[
            TreeTransformBlockSpec::multi(vec![0], vec![0, 1], vec![1.0, 2.0]),
            TreeTransformBlockSpec::multi(vec![1, 2], vec![2], vec![3.0, -1.0]),
            TreeTransformBlockSpec::multi(vec![3], vec![3, 0], vec![0.5, 1.0]),
        ],
    )
    .unwrap();
    assert_eq!(transform.parallel_schedule().scatter_component_count(), 2);
    let source = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut destination = [f64::NAN; 8];
    crate::host_pool::test_pool(4)
        .install(|engine| {
            tree_transform_structure_overwrite_with_structural_recoupling_raw(
                engine,
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &transform,
                &dst_structure,
                &src_structure,
                &mut destination,
                &source,
                1.0,
                &[],
                4,
            )
        })
        .unwrap();
    assert_eq!(destination, [7.0, 15.0, -5.0, 10.0, 18.0, -6.0, 4.5, 6.0]);
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
    // The executing split runs each side on the token share it was
    // balanced for: at three workers the three light components get one
    // slot and the heavy one two.
    let leaves = record_single_leaves(&[(0, 1, 1), (1, 1, 1), (2, 1, 1), (3, 6, 6)]);
    assert_eq!(leaves, [(vec![0, 1, 2], 1), (vec![3], 2)]);
    // The fixture's Single atoms at three lanes: 8 of 28 elements is the
    // closest left share to a third.
    let (_, transform) = su2_interleaved_fixture();
    let atoms = &transform.parallel_schedule().single_atoms;
    assert_eq!(
        balanced_boundary(atoms.len() - 1, 3, |part| atoms[part].weight_before),
        1
    );
}

/// Runs a three-worker threaded Single phase over atoms `(lo, weight,
/// widest)` of one Single each and returns every leaf's Singles with the
/// token budget it ran on, sorted.
fn record_single_leaves(atoms: &[(isize, usize, usize)]) -> Vec<(Vec<u32>, usize)> {
    use crate::transform_structure::TreeTransformSingleAtom;
    let mut schedule = TreeTransformParallelSchedule::default();
    let mut weight_before = 0;
    for (item, &(lo, weight, widest)) in atoms.iter().enumerate() {
        schedule.singles.push(item as u32);
        schedule.single_atoms.push(TreeTransformSingleAtom {
            item: item as u32,
            lo,
            weight_before,
            widest,
        });
        weight_before += weight;
        schedule.widest_single = schedule.widest_single.max(widest);
    }
    let end = atoms.last().map_or(0, |&(lo, _, _)| lo + 1);
    schedule.single_atoms.push(TreeTransformSingleAtom {
        item: atoms.len() as u32,
        lo: end,
        weight_before,
        widest: 0,
    });
    let leaves = std::sync::Mutex::new(Vec::new());
    let mut dst = vec![0.0f64; end as usize];
    let mut fused_indices = [0usize; 3];
    let mut narrow = Vec::new();
    let gate = LeafGate::new(schedule.widest_single);
    crate::host_pool::test_pool(3)
        .install(|engine| {
            replay_region(engine, |exec| {
                replay_single_phase(
                    &schedule,
                    &mut narrow,
                    &mut dst,
                    &mut fused_indices,
                    1,
                    3,
                    exec,
                    &gate,
                    schedule.widest_single,
                    true,
                    &|items, _, _, _, exec, _| {
                        leaves
                            .lock()
                            .unwrap()
                            .push((items.to_vec(), exec.budget().get()));
                        Ok(())
                    },
                )
            })
        })
        .unwrap()
        .unwrap();
    let mut leaves = leaves.into_inner().unwrap();
    leaves.sort_unstable();
    leaves
}

#[test]
fn wide_components_run_alone_on_the_whole_token_before_narrow_ones_split() {
    // What: a Single phase with one component whose planned block widens on
    // the whole token and three that do not runs the wide one alone on all
    // three workers, then splits the narrow ones one per worker.
    const LARGE: usize = 1 << 24;
    let leaves = record_single_leaves(&[(0, 1, 1), (1, LARGE, LARGE), (2, 1, 1), (3, 1, 1)]);
    assert_eq!(
        leaves,
        [(vec![0], 1), (vec![1], 3), (vec![2], 1), (vec![3], 1)]
    );
}
