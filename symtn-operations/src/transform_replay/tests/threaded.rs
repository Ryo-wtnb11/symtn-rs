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
    assert!(!transform.parallel_schedule().group_spans[0].slice_disjoint);
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
/// (blocks 0, 1) in address order and slice-disjoint, the group writing
/// blocks 2 and 3 spans `48..=95` and the one writing block 4 `96..=119`,
/// each disjoint, the inactive block 5 starts at 120, and the four pack
/// columns outnumber the two Singles.
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
    assert!(schedule.singles_slice_disjoint);
    let inactive: Vec<isize> = schedule
        .inactive
        .iter()
        .map(|&layout| layouts.entry(layout as usize).offset)
        .collect();
    assert_eq!(inactive, [120]);
    let mut spans: Vec<_> = schedule
        .group_spans
        .iter()
        .map(|span| (span.lo, span.hi, span.slice_disjoint))
        .collect();
    spans.sort_unstable();
    assert_eq!(spans, [(48, 95, true), (96, 119, true)]);
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
/// `0;3`, `1;3`, `2;3` interleave in one atom `[0, 5]`; `6;1` alone; `8;2`
/// and `9;2` interleave in `[8, 11]`; `12;1` alone. Block 0 is a Single,
/// blocks 1 and 6 one Multi group (one destination in each of two
/// disconnected atoms), block 2 inactive, block 3 a Single, and blocks 4 and
/// 5 the destinations of two Multi groups sharing their atom.
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
    for component in 0..3 {
        for &group in schedule.component_group_list(component) {
            let group = &schedule.scatter_groups[group as usize];
            for column in &schedule.scatter_columns[group.columns.clone()] {
                let atom = &schedule.atoms[column.slot as usize];
                let offset = transform.layouts().entry(column.dst_layout).offset;
                assert!(offset >= atom.lo && offset < atom.lo + atom.len as isize);
                assert!(schedule
                    .component_atoms(component)
                    .contains(&(column.slot as usize)));
            }
        }
    }
    assert_eq!(dst.required_len().unwrap(), 14);
}

#[test]
fn component_moving_more_than_usize_elements_still_compiles() {
    // What: a component whose written plus packed elements (`n` plus `3n`)
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

/// What: one caller gate serves every planned role a call admits, with one
/// dispatch-cost scan per token, and rejects each role below the paying size
/// without a width search: `G` Multi groups of planned `BLOCK`-element
/// scatter columns on a wide host token scan once and search no width,
/// whether `G` is 8, 32 or 128.
#[test]
fn one_caller_gate_scans_once_and_rejects_without_a_width_search() {
    use crate::kernel_adapter::sealed::{LeafGate, GATE_SCANS, GATE_WIDTHS};
    let count = |counter: &'static std::thread::LocalKey<core::cell::Cell<usize>>| {
        counter.with(|c| c.get())
    };
    for groups in [8, 32, 128] {
        let blocks = Arc::new(
            BlockStructure::packed_column_major(2, vec![vec![ROWS, COLS]; 2 * groups]).unwrap(),
        );
        let specs = (0..groups)
            .map(|g| {
                TreeTransformBlockSpec::multi(
                    vec![2 * g, 2 * g + 1],
                    vec![2 * g, 2 * g + 1],
                    vec![1.0, 0.0, 0.0, 1.0],
                )
            })
            .collect::<Vec<_>>();
        let structure =
            TreeTransformStructure::compile_structures(&blocks, &blocks, &specs).unwrap();
        let layouts = structure.layouts();
        assert_eq!(layouts.retained_plan_count(), 2 * groups);
        let mut pool = crate::host_pool::test_pool(4);
        let (scans, widths, admitted) = pool.with_exec(|exec| {
            let (scans, widths) = (count(&GATE_SCANS), count(&GATE_WIDTHS));
            let gate = LeafGate::new(layouts.max_planned_elements());
            let admitted = (0..layouts.entry_count())
                .filter(|&entry| {
                    let elements = layouts.entry(entry).element_count;
                    let role = layouts.role(entry).unwrap();
                    crate::kernel_adapter::admitted_plan(role, elements, &exec, &gate).is_some()
                })
                .count();
            (
                count(&GATE_SCANS) - scans,
                count(&GATE_WIDTHS) - widths,
                admitted,
            )
        });
        assert_eq!((scans, widths, admitted), (1, 0, 0), "{groups} groups");
    }
}
