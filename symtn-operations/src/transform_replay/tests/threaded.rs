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
            )?;
        }
        assert_eq!(actual, expected);
    }
    Ok(())
}

#[test]
fn threaded_interleaved_singles_write_their_own_elements() {
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
    assert!(!structure.parallel_schedule().singles_slice_disjoint());
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
            )
        })
        .unwrap();
}

#[test]
fn threaded_multi_scatter_writes_interleaved_destinations() {
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
    // One group whose two columns interleave: one hull over both.
    let span = transform.parallel_schedule().group_spans[0];
    assert_eq!((span.lo, span.hi), (0, 3));
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
                )
            })
            .unwrap();
    }

    assert_eq!(threaded, serial);
    assert_eq!(threaded, [6.0, 13.0, 17.0, 24.0]);
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

/// What: the compiled replay metadata of [`recoupling_fixture`], by hand.
/// Destination block `d` spans elements `24 d..=24 d + 23`: the Singles
/// (blocks 0, 1) in address order and slice-disjoint (one run each), the
/// group writing blocks 2 and 3 spans `48..=95` and the one writing block 4
/// `96..=119` (first in plan order), the inactive block 5 starts at 120 (one
/// run), and the four pack columns outnumber the two Singles.
#[test]
fn replay_metadata_lists_direct_items_in_address_order() {
    let (_, _, replay) = recoupling_fixture();
    let schedule = replay.parallel_schedule();
    let layouts = replay.layouts();
    let single_starts: Vec<isize> = schedule
        .singles
        .iter()
        .map(|&block| {
            let (dst_layout, _, _) =
                crate::transform_replay::writers::scheduled_single(replay.blocks(), block).unwrap();
            layouts.entry(dst_layout).offset
        })
        .collect();
    assert_eq!(single_starts, [0, 24]);
    assert!(schedule.singles_slice_disjoint());
    assert_eq!(schedule.single_runs, [0, 1]);
    let inactive: Vec<isize> = schedule
        .inactive
        .iter()
        .map(|&layout| layouts.entry(layout as usize).offset)
        .collect();
    assert_eq!(inactive, [120]);
    // In plan order: the plan sorts the group writing block 4 first.
    let spans: Vec<_> = schedule
        .group_spans
        .iter()
        .map(|span| (span.lo, span.hi))
        .collect();
    assert_eq!(spans, [(96, 119), (48, 95)]);
    assert_eq!(schedule.inactive_runs, [0]);
    assert_eq!(schedule.replay_runnable, 4);
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
        )
        .unwrap();

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
        )
        .unwrap();
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

/// Seven one-axis destination blocks of two elements (`offset; stride`):
/// `0;3`, `1;3`, `2;3` interleave in `[0, 5]`; `6;1` alone; `8;2` and `9;2`
/// interleave in `[8, 11]`; `12;1` alone. Block 0 is a Single, blocks 1 and
/// 6 one Multi group (one destination in each of two disconnected ranges),
/// block 2 inactive, block 3 a Single, and blocks 4 and 5 the destinations
/// of two Multi groups that interleave.
pub(super) fn joint_fixture() -> (
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
fn a_multi_moving_more_than_usize_elements_still_compiles() {
    // What: a Multi group whose written plus packed elements (`n` plus `3n`)
    // exceed `usize` while every buffer extent fits compiles. Metadata only,
    // no tensor data.
    let n = 1usize << (usize::BITS - 2);
    let structure = BlockStructure::packed_column_major(1, [vec![n]]).unwrap();
    TreeTransformStructure::<f64>::compile_structures(
        &structure,
        &structure,
        &[TreeTransformBlockSpec::multi(
            vec![0],
            vec![0, 0, 0],
            vec![1.0, 1.0, 1.0],
        )],
    )
    .unwrap();
}

#[test]
fn inactive_block_ending_beyond_isize_still_compiles() {
    // What: an inactive canonical `[n, 2]` block whose last element `2n - 1`
    // exceeds `isize::MAX` inside the `usize` storage compiles, as one
    // inactive layout. Metadata only, no tensor data.
    use symtn_core::{FusionTreeHomSpace, U1FusionRule, U1Irrep};
    let n = (1usize << (usize::BITS - 2)) + 1;
    let structure = FusionTreeHomSpace::from_sectors(
        [(U1Irrep::new(0).unwrap(), n)],
        [(U1Irrep::new(0).unwrap(), 2)],
    )
    .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
    .unwrap();
    let transform =
        TreeTransformStructure::<f64>::compile_structures(&structure, &structure, &[]).unwrap();
    assert_eq!(transform.parallel_schedule().inactive, [0]);
}

#[test]
fn joint_fixture_replay_matches_the_block_oracle() {
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
            )
            .unwrap();
        });
        assert_eq!(overwritten, expected, "threads {threads}");
        assert_eq!(accumulated, axpby, "threads {threads}");
    }
}

#[test]
fn interleaved_single_and_scatter_blocks_replay_the_dense_oracle() {
    // What: a Single's and a scatter column's interleaved blocks (two
    // codomain trees of two rows over six columns, rows 4) and the group's
    // other column, whose block fuses with its packed column to one axis: a
    // one-worker replay writes the dense oracle.
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
            )
        })
        .unwrap();
    assert_eq!(destination, expected);
}

#[test]
fn replay_matches_the_block_oracle_on_interleaved_su2_blocks() {
    // What: on interleaved canonical blocks whose Multi groups write
    // disconnected ranges, replay writes exactly the per-block oracle for the
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
                None,
            )
            .unwrap()
        });
        assert_eq!(overwritten, expected, "threads {threads}");
        assert_eq!(accumulated, axpby, "threads {threads}");
        assert_eq!(owned, Some(expected.clone()), "threads {threads}");
    }
}

/// Coupled column-major matrices of `(cols, trees)`, each tree a row range
/// `(rows, role)` whose role is `'S'` (a Single), `'I'` (inactive) or a
/// digit (a member of that matrix's Multi group); every tree of a matrix
/// interleaves with the others in its columns. Returns the structure
/// (destination = source), the transform, and the specs as
/// `(destination blocks, source blocks, coefficients)` for the oracle.
#[allow(clippy::type_complexity)]
fn coupled_matrices(
    mats: &[(usize, Vec<(usize, char)>)],
) -> (
    Arc<BlockStructure>,
    TreeTransformStructure<f64>,
    Vec<(Vec<usize>, Vec<usize>, Vec<f64>)>,
) {
    let coefficient = |i: usize| 0.25 + ((i * 7919) % 97) as f64 / 97.0;
    let (mut blocks, mut specs, mut offset, mut next) = (Vec::new(), Vec::new(), 0, 0);
    for (cols, trees) in mats {
        let rows: usize = trees.iter().map(|tree| tree.0).sum();
        let mut row = 0;
        let mut groups = std::collections::BTreeMap::<char, Vec<usize>>::new();
        for &(r, role) in trees {
            let b = blocks.len();
            blocks.push(
                BlockSpec::with_key(
                    BlockKey::ordinal(b),
                    vec![r, *cols],
                    vec![1, rows],
                    offset + row,
                )
                .unwrap(),
            );
            row += r;
            match role {
                'S' => {
                    specs.push((vec![b], vec![b], vec![coefficient(next)]));
                    next += 1;
                }
                'I' => {}
                g => groups.entry(g).or_default().push(b),
            }
        }
        for members in groups.into_values() {
            let m = members.len();
            let c = (0..m * m)
                .map(|i| {
                    next += 1;
                    if i / m == i % m {
                        1.0
                    } else {
                        0.1 * coefficient(next)
                    }
                })
                .collect();
            specs.push((members.clone(), members, c));
        }
        offset += rows * cols;
    }
    let structure = Arc::new(BlockStructure::from_blocks_with_rank(2, blocks).unwrap());
    let compiled = specs
        .iter()
        .map(|(dst, src, c)| {
            if dst.len() == 1 && src.len() == 1 {
                TreeTransformBlockSpec::single(dst[0], src[0], c[0])
            } else {
                TreeTransformBlockSpec::multi(dst.clone(), src.clone(), c.clone())
            }
        })
        .collect::<Vec<_>>();
    let transform =
        TreeTransformStructure::compile_structures(&structure, &structure, &compiled).unwrap();
    (structure, transform, specs)
}

/// The dense oracle of [`coupled_matrices`]: destination block `d` of a
/// spec is `sum_s c[d, s] src_s` element by element (members share their
/// layout shape), an inactive block is zero (overwrite) or `beta dst`.
fn coupled_matrices_expected(
    structure: &BlockStructure,
    specs: &[(Vec<usize>, Vec<usize>, Vec<f64>)],
    source: &[f64],
    initial: &[f64],
    beta: Option<f64>,
) -> Vec<f64> {
    let addresses = |block: usize| {
        let block = structure.block(block).unwrap();
        let (shape, strides, offset) = (block.shape(), block.strides(), block.offset());
        (0..shape[1])
            .flat_map(move |j| (0..shape[0]).map(move |i| offset + i * strides[0] + j * strides[1]))
            .collect::<Vec<_>>()
    };
    let mut expected = match beta {
        Some(beta) => initial.iter().map(|value| beta * value).collect::<Vec<_>>(),
        None => vec![0.0; initial.len()],
    };
    for (dst, src, c) in specs {
        for (d, &dst_block) in dst.iter().enumerate() {
            let out = addresses(dst_block);
            for (s, &src_block) in src.iter().enumerate() {
                for (&at, &from) in out.iter().zip(&addresses(src_block)) {
                    expected[at] += c[d * src.len() + s] * source[from];
                }
            }
        }
    }
    expected
}

/// What: on coupled matrices whose trees interleave in their
/// columns (inactive trees beside Singles and a group pair, as `inact`;
/// two-, three-member groups and a Single, as `hetero`), the hull-run
/// splits write the dense oracle for an overwrite and a `beta = 0.5`
/// accumulation, at one, two, three and eight cheap-dispatch workers under
/// `Auto`, and under `Outer` where the executor honors it.
#[test]
fn interleaved_coupled_matrices_replay_the_dense_oracle() {
    let inact = (0..6)
        .map(|_| (4, vec![(3, 'S'), (3, '0'), (3, '0'), (3, 'I')]))
        .collect::<Vec<_>>();
    let hetero = (0..5)
        .map(|m| {
            (
                3 + m,
                vec![(2, '0'), (5, 'S'), (1, '1'), (2, '0'), (1, '1'), (1, '1')],
            )
        })
        .collect::<Vec<_>>();
    let honors = DefaultDenseExecutor::new().honors_forced_batch_policy();
    for mats in [inact, hetero] {
        let (structure, transform, specs) = coupled_matrices(&mats);
        let len = structure.required_len().unwrap();
        let source = (0..len).map(|v| v as f64 * 0.5 - 3.0).collect::<Vec<_>>();
        let initial = (0..len).map(|v| 1.0 - v as f64).collect::<Vec<_>>();
        let overwrite = coupled_matrices_expected(&structure, &specs, &source, &initial, None);
        let axpby = coupled_matrices_expected(&structure, &specs, &source, &initial, Some(0.5));
        for threads in [1, 2, 3, 8] {
            for strategy in [
                tenferro_cpu::CpuBatchStrategy::Auto,
                tenferro_cpu::CpuBatchStrategy::Outer,
            ] {
                if strategy != tenferro_cpu::CpuBatchStrategy::Auto && !honors {
                    continue;
                }
                let flat = tenferro_cpu::threadteam::WidthCosts::flat(Duration::from_nanos(1));
                let mut engine = symtn_dense::CpuEngine::new(
                    symtn_dense::Pool::builder()
                        .threads(threads)
                        .dispatch_costs(tenferro_cpu::threadteam::DispatchCosts::new(
                            flat, flat, flat,
                        ))
                        .build()
                        .unwrap(),
                )
                .with_batch_policy(tenferro_cpu::CpuBatchPolicy::new(strategy));
                let mut overwritten = vec![f64::NAN; len];
                let mut accumulated = initial.clone();
                tree_transform_structure_overwrite_with_structural_recoupling_raw(
                    &mut engine,
                    &mut StridedHostKernelAdapter::default(),
                    &mut DefaultDenseExecutor::new(),
                    &mut TreeTransformWorkspace::default(),
                    &transform,
                    &structure,
                    &structure,
                    &mut overwritten,
                    &source,
                    1.0,
                    &[],
                )
                .unwrap();
                tree_transform_structure_with_structural_recoupling_raw(
                    &mut engine,
                    &mut StridedHostKernelAdapter::default(),
                    &mut DefaultDenseExecutor::new(),
                    &mut TreeTransformWorkspace::default(),
                    &transform,
                    &structure,
                    &structure,
                    &mut accumulated,
                    &source,
                    1.0,
                    0.5,
                )
                .unwrap();
                let what = format!("{} matrices, {threads} threads, {strategy:?}", mats.len());
                let close = |a: &[f64], b: &[f64]| {
                    a.iter()
                        .zip(b)
                        .all(|(a, b)| (a - b).abs() <= 1e-12 * a.abs().max(1.0))
                };
                assert!(close(&overwritten, &overwrite), "{what}");
                assert!(close(&accumulated, &axpby), "{what}");
            }
        }
    }
}

/// A canonical U(1) structure (four codomain legs of charges `0, 1`, one
/// domain leg of charges `0..=4` and degeneracy 2): each coupled matrix's
/// codomain trees interleave in its columns. In every matrix, by offset,
/// the trees take the roles Single, group, group, inactive in turn (a
/// group short of its pair is inactive instead). Returns the structure, the
/// specs as `(destination blocks, source blocks, coefficients)`, the
/// groups and the inactive blocks.
#[allow(clippy::type_complexity)]
fn canonical_runs() -> (
    Arc<BlockStructure>,
    Vec<(Vec<usize>, Vec<usize>, Vec<f64>)>,
    Vec<[usize; 2]>,
    Vec<usize>,
) {
    use symtn_core::{FusionProductSpace, FusionTreeHomSpace, SectorLeg, U1FusionRule, U1Irrep};
    let bit = || {
        SectorLeg::new(
            [(U1Irrep::new(0).unwrap(), 1), (U1Irrep::new(1).unwrap(), 1)],
            false,
        )
    };
    let out = SectorLeg::new((0..=4).map(|q| (U1Irrep::new(q).unwrap(), 2)), false);
    let structure = FusionTreeHomSpace::new(
        FusionProductSpace::new([bit(), bit(), bit(), bit()]),
        FusionProductSpace::new([out]),
    )
    .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
    .unwrap();
    let regions = structure.coupled_sector_regions(4).unwrap().unwrap();
    let (mut specs, mut groups, mut inactive) = (Vec::new(), Vec::new(), Vec::new());
    for region in regions.iter() {
        let mut blocks = (0..structure.block_count())
            .filter(|&b| {
                region
                    .range()
                    .contains(&structure.block(b).unwrap().offset())
            })
            .collect::<Vec<_>>();
        blocks.sort_by_key(|&b| structure.block(b).unwrap().offset());
        let mut position = 0;
        while position < blocks.len() {
            let b = blocks[position];
            match position % 4 {
                0 => specs.push((vec![b], vec![b], vec![0.5 + b as f64])),
                1 if position + 1 < blocks.len() => {
                    let pair = [b, blocks[position + 1]];
                    specs.push((pair.to_vec(), pair.to_vec(), vec![1.0, -0.5, 0.25, 2.0]));
                    groups.push(pair);
                    position += 1;
                }
                _ => inactive.push(b),
            }
            position += 1;
        }
    }
    (structure, specs, groups, inactive)
}

/// The elements of `structure`'s block `block`, in its iteration order.
fn block_addresses(structure: &BlockStructure, block: usize) -> Vec<usize> {
    let block = structure.block(block).unwrap();
    let mut at = vec![block.offset()];
    for (&extent, &stride) in block.shape().iter().zip(block.strides()) {
        at = at
            .iter()
            .flat_map(|&base| (0..extent).map(move |i| base + i * stride))
            .collect();
    }
    at
}

/// The dense oracle over `specs` with source map `map` (identity or
/// conjugation): destination block `d` is `sum_s c[d, s] map(src_s)`, an
/// inactive block zero, or the old value times `beta` when given.
fn spec_oracle<D>(
    structure: &BlockStructure,
    specs: &[(Vec<usize>, Vec<usize>, Vec<f64>)],
    source: &[D],
    initial: &[D],
    beta: Option<D>,
    map: fn(D) -> D,
) -> Vec<D>
where
    D: Copy + core::ops::Add<Output = D> + core::ops::Mul<Output = D> + From<f64>,
{
    let mut expected = match beta {
        Some(beta) => initial
            .iter()
            .map(|&value| beta * value)
            .collect::<Vec<_>>(),
        None => vec![D::from(0.0); initial.len()],
    };
    for (dst, src, c) in specs {
        for (d, &dst_block) in dst.iter().enumerate() {
            let out = block_addresses(structure, dst_block);
            for (s, &src_block) in src.iter().enumerate() {
                for (&at, &from) in out.iter().zip(&block_addresses(structure, src_block)) {
                    expected[at] = expected[at] + D::from(c[d * src.len() + s]) * map(source[from]);
                }
            }
        }
    }
    expected
}

/// What: on canonical coupled matrices whose trees interleave (Singles that
/// are not slice-disjoint, groups whose two blocks overlap, inactive trees
/// beside them), eight cheap-dispatch workers split the Singles, the
/// inactive trees and each chunk's scatter over their hull runs, which
/// main wrote inline: the Singles and the inactive trees are each written
/// by more than one worker and every scatter column is offered fewer than
/// the eight slots. On one, three and eight workers under `Auto` (and
/// `Outer` where the executor honors it), f64 and conjugated c64 replays
/// write the dense oracle for an initialized overwrite, a `beta = 0.5`
/// accumulation and the owned output.
#[test]
fn canonical_interleaved_runs_replay_the_dense_oracle() {
    use num_complex::Complex64;
    use tenferro_cpu::CpuBatchStrategy;
    let (structure, specs, groups, inactive) = canonical_runs();
    let compiled = |conjugate| {
        let specs = specs
            .iter()
            .map(|(dst, src, c)| {
                if dst.len() == 1 {
                    TreeTransformBlockSpec::single(dst[0], src[0], c[0])
                } else {
                    TreeTransformBlockSpec::multi(dst.clone(), src.clone(), c.clone())
                }
            })
            .collect::<Vec<_>>();
        TreeTransformStructure::compile_structures_with_storage_conjugation(
            &structure, &structure, &specs, conjugate,
        )
        .unwrap()
    };
    let (real, complex) = (compiled(false), compiled(true));
    let schedule = real.parallel_schedule();
    assert!(!schedule.singles_slice_disjoint());
    assert!(schedule.single_runs.len() >= 2 && schedule.inactive_runs.len() >= 2);
    let hull = |block| {
        let at = block_addresses(&structure, block);
        (*at.iter().min().unwrap(), *at.iter().max().unwrap())
    };
    assert!(groups.len() >= 2);
    for &[left, right] in &groups {
        let ((lo, hi), (other_lo, other_hi)) = (hull(left), hull(right));
        assert!(
            lo <= other_hi && other_lo <= hi,
            "the group's blocks overlap"
        );
    }

    let mut engine = super::recording::flat_engine(8, 1, CpuBatchStrategy::Auto);
    let (offers, marks) = super::recording::recorded(&mut engine, &real, &structure, &structure);
    assert!(offers.iter().all(|&offer| offer < 8), "{offers:?}");
    let workers = |blocks: &mut dyn Iterator<Item = usize>| {
        let mut seen = blocks
            .flat_map(|block| block_addresses(&structure, block))
            .filter_map(|at| marks[at].worker)
            .collect::<Vec<_>>();
        seen.sort_unstable_by_key(|worker| format!("{worker:?}"));
        seen.dedup();
        seen.len()
    };
    let singles = specs
        .iter()
        .filter(|spec| spec.0.len() == 1)
        .map(|spec| spec.0[0]);
    assert!(workers(&mut singles.into_iter()) >= 2, "the Singles split");
    assert!(
        workers(&mut inactive.iter().copied()) >= 2,
        "the inactive trees split"
    );

    let len = structure.required_len().unwrap();
    let source = (0..len).map(|v| v as f64 * 0.5 - 3.0).collect::<Vec<_>>();
    let initial = (0..len).map(|v| 1.0 - v as f64).collect::<Vec<_>>();
    let csource = source
        .iter()
        .map(|&v| Complex64::new(v, 0.25 * v - 1.0))
        .collect::<Vec<_>>();
    let cinitial = initial
        .iter()
        .map(|&v| Complex64::new(v, -v))
        .collect::<Vec<_>>();
    let half = Complex64::new(0.5, 0.0);
    let expected = (
        spec_oracle(&structure, &specs, &source, &initial, None, |v| v),
        spec_oracle(&structure, &specs, &source, &initial, Some(0.5), |v| v),
        spec_oracle(
            &structure,
            &specs,
            &csource,
            &cinitial,
            None,
            |v: Complex64| v.conj(),
        ),
        spec_oracle(
            &structure,
            &specs,
            &csource,
            &cinitial,
            Some(half),
            |v: Complex64| v.conj(),
        ),
    );
    let close = |a: &[f64], b: &[f64]| {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(a, b)| (a - b).abs() <= 1e-12 * a.abs().max(1.0))
    };
    let cclose = |a: &[Complex64], b: &[Complex64]| {
        a.len() == b.len()
            && a.iter()
                .zip(b)
                .all(|(a, b)| (a - b).norm() <= 1e-12 * a.norm().max(1.0))
    };
    let honors = DefaultDenseExecutor::new().honors_forced_batch_policy();
    for threads in [1, 3, 8] {
        for strategy in [CpuBatchStrategy::Auto, CpuBatchStrategy::Outer] {
            if strategy != CpuBatchStrategy::Auto && !honors {
                continue;
            }
            let what = format!("{threads} threads, {strategy:?}");
            let mut engine = super::recording::flat_engine(threads, 1, strategy);
            let mut overwritten = vec![f64::NAN; len];
            let mut accumulated = initial.clone();
            tree_transform_structure_overwrite_with_structural_recoupling_raw(
                &mut engine,
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &real,
                &structure,
                &structure,
                &mut overwritten,
                &source,
                1.0,
                &[],
            )
            .unwrap();
            tree_transform_structure_with_structural_recoupling_raw(
                &mut engine,
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &real,
                &structure,
                &structure,
                &mut accumulated,
                &source,
                1.0,
                0.5,
            )
            .unwrap();
            let owned = try_tree_transform_structure_overwrite_owned_raw(
                &mut engine,
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &real,
                &structure,
                &structure,
                4,
                &source,
                1.0,
                None,
            )
            .unwrap()
            .expect("canonical storage admits the owned output");
            assert!(close(&overwritten, &expected.0), "f64 overwrite, {what}");
            assert!(close(&accumulated, &expected.1), "f64 accumulation, {what}");
            assert!(close(&owned, &expected.0), "f64 owned, {what}");

            let mut overwritten = vec![Complex64::new(f64::NAN, 0.0); len];
            let mut accumulated = cinitial.clone();
            tree_transform_structure_overwrite_with_structural_recoupling_raw(
                &mut engine,
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &complex,
                &structure,
                &structure,
                &mut overwritten,
                &csource,
                Complex64::new(1.0, 0.0),
                &[],
            )
            .unwrap();
            tree_transform_structure_with_structural_recoupling_raw(
                &mut engine,
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &complex,
                &structure,
                &structure,
                &mut accumulated,
                &csource,
                Complex64::new(1.0, 0.0),
                half,
            )
            .unwrap();
            let owned = try_tree_transform_structure_overwrite_owned_raw(
                &mut engine,
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &complex,
                &structure,
                &structure,
                4,
                &csource,
                Complex64::new(1.0, 0.0),
                None,
            )
            .unwrap()
            .expect("canonical storage admits the owned output");
            assert!(cclose(&overwritten, &expected.2), "c64 overwrite, {what}");
            assert!(
                cclose(&accumulated, &expected.3),
                "c64 accumulation, {what}"
            );
            assert!(cclose(&owned, &expected.2), "c64 owned, {what}");
        }
    }
}
