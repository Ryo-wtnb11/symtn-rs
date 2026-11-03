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
    // slot and the heavy one two. (An accumulating phase, so no component
    // runs alone; the blocks widen on any share and the total pays.)
    // Each atom is one block that fills either share, so its weight is its
    // largest block, as compiled metadata has it.
    let wide = first_count(3, |width| width == 3).max(first_count(2, |width| width == 2));
    let unit = wide;
    let leaves = record_single_phase(
        &[
            (0, unit, wide),
            (1, unit, wide),
            (2, unit, wide),
            (3, 6 * unit, 6 * unit),
        ],
        false,
    )
    .0;
    assert_eq!(
        leaves,
        [(vec![0], 1), (vec![1], 1), (vec![2], 1), (vec![3], 2)]
    );
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
    record_single_phase(atoms, true).0
}

/// [`record_single_leaves`] and the Strided width searches the phase ran on
/// the calling thread.
fn record_single_phase(
    atoms: &[(isize, usize, usize)],
    widen: bool,
) -> (Vec<(Vec<u32>, usize)>, usize) {
    use crate::kernel_adapter::sealed::WIDTH_QUERIES;
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
    let mut pool = crate::host_pool::test_pool(3);
    let queries = WIDTH_QUERIES.with(|queries| queries.get());
    pool.install(|engine| {
        replay_region(engine, |exec| {
            replay_single_phase(
                &schedule,
                &mut narrow,
                &mut Vec::new(),
                &mut dst,
                &mut fused_indices,
                1,
                3,
                exec,
                &gate,
                schedule.widest_single,
                widen,
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
    let queries = WIDTH_QUERIES.with(|queries| queries.get()) - queries;
    let mut leaves = leaves.into_inner().unwrap();
    leaves.sort_unstable();
    (leaves, queries)
}

#[test]
fn wide_components_run_alone_on_the_whole_token_before_narrow_ones_split() {
    // What: a Single phase with one component whose planned block widens on
    // the whole token and three that do not runs the wide one alone on all
    // three workers, then splits the narrow ones one per worker.
    let large = first_count(3, |width| width == 3);
    // The narrow ones move enough in total to fill the token, so the split
    // pays; their blocks never widen, so each runs on one worker.
    let leaves = record_single_leaves(&[
        (0, large, 1),
        (1, large, large),
        (2, large, 1),
        (3, large, 1),
    ]);
    assert_eq!(
        leaves,
        [(vec![0], 1), (vec![1], 3), (vec![2], 1), (vec![3], 1)]
    );
}

#[test]
fn classifying_more_wide_components_asks_strided_no_more_often() {
    // What: deciding which components widen takes one threshold search per
    // token shape, so a phase with three times the wide components asks
    // Strided for a width exactly as often.
    let large = first_count(3, |width| width == 3);
    let phase = |wide: isize| {
        let atoms = std::iter::once((0, large, 1))
            .chain((1..=wide).map(|lo| (lo, large, large)))
            .chain([(wide + 1, large, 1)])
            .collect::<Vec<_>>();
        record_single_phase(&atoms, true)
    };
    let (few, few_queries) = phase(2);
    let (many, many_queries) = phase(6);
    assert_eq!(few.iter().filter(|(_, budget)| *budget == 3).count(), 2);
    assert_eq!(many.iter().filter(|(_, budget)| *budget == 3).count(), 6);
    assert!(few_queries > 0);
    assert_eq!(many_queries, few_queries);
}

#[test]
fn narrow_single_scratch_is_charged_by_capacity() {
    // What: the Single phase's narrow-atom list counts in the workspace's
    // retained bytes by its capacity.
    let mut workspace = TreeTransformWorkspace::<f64>::default();
    let before = workspace.retained_bytes();
    workspace.narrow_atoms.reserve_exact(4);
    assert_eq!(
        workspace.retained_bytes() - before,
        workspace.narrow_atoms.capacity() * std::mem::size_of::<NarrowAtom>()
    );
}

#[test]
fn only_components_that_use_every_worker_run_on_the_whole_token() {
    use crate::kernel_adapter::observed::ObservedKernels;
    use strided_kernel::execution::{parallel_width, KernelClass};
    // What: in a threaded overwrite on eight workers, a Single or Multi
    // component whose blocks Strided would give all eight workers runs alone
    // on the whole token, while a smaller widening component and two tiny
    // ones of the same phase split the token and each runs on its lane's
    // share; every block holds its oracle value. The moldable size is the first count Strided widens at
    // all, so the expectation for it follows Strided's own answer.
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
    let structure = Arc::new(
        BlockStructure::packed_column_major(
            1,
            [
                2, 2, moldable, moldable, large, large, 2, 2, 2, moldable, large, 2,
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
            .chain((8..12).map(|block| TreeTransformBlockSpec::single(block, block, 1.5)))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    assert_eq!(transform.parallel_schedule().scatter_component_count(), 4);
    let len = structure.required_len().unwrap();
    let source = (0..len)
        .map(|value| (value % 97) as f64 - 48.0)
        .collect::<Vec<_>>();
    let mut expected = vec![0.0; len];
    let mut start = 0;
    for block_len in [2, moldable, large, 2] {
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
    let budgets = std::sync::Arc::clone(&kernels.copy_budgets);
    let mut destination = vec![f64::NAN; len];
    pool.install(|engine| {
        tree_transform_structure_overwrite_with_structural_recoupling_raw(
            engine,
            &mut kernels,
            &mut DefaultDenseExecutor::new(),
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
    assert_eq!(budgets.len(), 12);
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
                    &mut DefaultDenseExecutor::new(),
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
fn a_saturating_single_runs_alone_even_where_a_larger_one_does_not() {
    use std::time::Duration;
    use tenferro_cpu::threadteam::WidthCosts;
    // What: with a cost row under which threadteam's rounded comparison
    // gives the larger of two compiled Single blocks three of four workers
    // and the smaller all four, the smaller still runs alone on the whole
    // token: the largest block's answer settles nothing outside the gate's
    // exact domain. A zero-size destination element replays the schedule
    // without allocating the blocks' payload.
    let costs = WidthCosts::from_anchors(&[
        (2, Duration::ZERO),
        (3, Duration::ZERO),
        (4, Duration::from_nanos(1 << 50)),
    ])
    .unwrap();
    let mut pool = crate::host_pool::test_pool_with_costs(4, costs);
    // The counts around the estimate where rounding ties widths 3 and 4, by
    // bisection on this host's copy estimate: a count Strided gives four
    // workers, then the next larger count it gives three.
    let width = |exec: &Exec<'_>, n| {
        strided_kernel::execution::parallel_width(
            exec,
            strided_kernel::execution::KernelClass::Copy,
            n,
        )
        .get()
    };
    let tie = Duration::from_nanos(13_510_798_882_111_491);
    let (mut below, mut at) = (0usize, usize::MAX);
    while at - below > 1 {
        let middle = below + (at - below) / 2;
        if strided_kernel::execution::KernelClass::Copy.estimate(middle) >= tie {
            at = middle;
        } else {
            below = middle;
        }
    }
    let lengths = pool.with_exec(|exec| {
        let window = below.saturating_sub(64)..=at + 64;
        let narrower = window
            .clone()
            .find(|&n| {
                width(&exec, n) == 3 && window.clone().any(|m| m < n && width(&exec, m) == 4)
            })
            .expect("the cost row ties widths 3 and 4 near this estimate");
        let wider = (below.saturating_sub(64)..narrower)
            .rev()
            .find(|&n| width(&exec, n) == 4)
            .expect("a smaller count gets four workers");
        [wider, narrower]
    });
    let structure =
        Arc::new(BlockStructure::packed_column_major(1, lengths.map(|len| vec![len])).unwrap());
    let transform = TreeTransformStructure::compile_structures(
        &structure,
        &structure,
        &[0, 1].map(|block| TreeTransformBlockSpec::single(block, block, 1.0)),
    )
    .unwrap();
    let schedule = transform.parallel_schedule();
    assert_eq!(schedule.single_component_count(), 2);
    assert_eq!(schedule.widest_single, lengths[1]);
    let gate = LeafGate::new(transform.layouts().max_planned_elements());
    let widths = pool.with_exec(|exec| lengths.map(|len| width(&exec, len)));
    assert_eq!(widths, [4, 3]);
    let leaves = std::sync::Mutex::new(Vec::new());
    let mut dst = vec![(); structure.required_len().unwrap()];
    let mut fused_indices = [0usize; 4];
    let mut narrow = Vec::new();
    pool.with_exec(|exec| {
        replay_single_phase(
            schedule,
            &mut narrow,
            &mut Vec::new(),
            &mut dst,
            &mut fused_indices,
            1,
            4,
            exec,
            &gate,
            transform.layouts().max_planned_elements(),
            true,
            &|items, _, _, _, exec, _| {
                for &item in items {
                    let (dst_layout, _, _) = scheduled_single(transform.blocks(), item)?;
                    leaves.lock().unwrap().push((
                        transform.layouts().entry(dst_layout).element_count,
                        exec.budget().get(),
                    ));
                }
                Ok(())
            },
        )
    })
    .unwrap();
    let mut leaves = leaves.into_inner().unwrap();
    leaves.sort_unstable();
    // The saturating block alone on four workers; the other, the only one
    // left, on the three Strided gives it.
    assert_eq!(leaves, [(lengths[0], 4), (lengths[1], 3)]);
}

#[test]
fn a_single_phase_whose_moves_do_not_pay_a_worker_runs_inline() {
    // What: many tiny Single components whose total moved elements Strided
    // gives one worker run in one list on the caller's whole token: no fork.
    assert!(first_count(3, |width| width > 1) > 32);
    let atoms = (0..32).map(|lo| (lo, 1, 1)).collect::<Vec<_>>();
    let leaves = record_single_leaves(&atoms);
    assert_eq!(leaves, [((0..32).collect::<Vec<u32>>(), 3)]);
}

#[test]
fn a_split_uses_no_more_lanes_than_its_components_can() {
    // What: two components that move enough to pay a split but whose blocks
    // never widen use two lanes of three, one worker each.
    // Each moves enough to fill the token on its own.
    let moved = first_count(3, |width| width == 3);
    let leaves = record_single_leaves(&[(0, moved, 1), (1, moved, 1)]);
    assert_eq!(leaves, [(vec![0], 1), (vec![1], 1)]);
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
fn first_count(threads: usize, holds: impl Fn(usize) -> bool) -> usize {
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
            &mut DefaultDenseExecutor::new(),
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

#[test]
fn independent_widening_components_share_the_token_by_their_strided_widths() {
    use strided_kernel::execution::{parallel_width, KernelClass};
    // What: nine independent compiled Single components whose blocks Strided
    // gives more than one but fewer than all workers of a 16- and a 96-worker
    // token run on nine lanes (the replay's worker count, capped by its
    // component count) that together hold as much of the token as their
    // widths can use: each lane at most its component's width, all lanes
    // `min(budget, sum of widths)`.
    for budget in [16, 96] {
        let mut pool = crate::host_pool::test_pool(budget);
        let (count, width) = pool.with_exec(|exec| {
            let width = |n| parallel_width(&exec, KernelClass::Copy, n).get();
            let mut at = 1usize;
            while width(at) < 2 {
                at *= 2;
            }
            let (mut below, mut at) = (at / 2, at);
            while at - below > 1 {
                let middle = below + (at - below) / 2;
                if width(middle) >= 2 {
                    at = middle;
                } else {
                    below = middle;
                }
            }
            (at, width(at))
        });
        assert!(
            width > 1 && width < budget,
            "budget {budget}: width {width}"
        );
        let structure =
            Arc::new(BlockStructure::packed_column_major(2, vec![vec![count, 1]; 9]).unwrap());
        let transform = TreeTransformStructure::compile_structures(
            &structure,
            &structure,
            &(0..9)
                .map(|block| TreeTransformBlockSpec::single(block, block, 2.0))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let schedule = transform.parallel_schedule();
        assert_eq!(schedule.single_component_count(), 9);
        let threads =
            pool.install(|engine| effective_tree_transform_threads(engine, schedule, budget));
        assert_eq!(threads, 9);
        let leaves = std::sync::Mutex::new(Vec::new());
        let mut dst = vec![(); structure.required_len().unwrap()];
        let mut fused_indices = [0usize; 2];
        let gate = LeafGate::new(transform.layouts().max_planned_elements());
        pool.with_exec(|exec| {
            replay_single_phase(
                schedule,
                &mut Vec::new(),
                &mut Vec::new(),
                &mut dst,
                &mut fused_indices,
                2,
                threads,
                exec,
                &gate,
                transform.layouts().max_planned_elements(),
                true,
                &|_, _, _, _, exec, _| {
                    leaves.lock().unwrap().push(exec.budget().get());
                    Ok(())
                },
            )
        })
        .unwrap();
        let leaves = leaves.into_inner().unwrap();
        assert_eq!(leaves.len(), 9, "budget {budget}");
        assert!(
            leaves.iter().all(|&lane| lane <= width),
            "budget {budget}: {leaves:?}"
        );
        assert_eq!(
            leaves.iter().sum::<usize>(),
            budget.min(9 * width),
            "budget {budget}: {leaves:?}"
        );
    }
}
