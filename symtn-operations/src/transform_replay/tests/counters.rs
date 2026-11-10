//! Structural counters of the replay's dispatch: grouped submissions, the
//! budgets each item is offered, and the leaf-gate work of every gate a
//! call builds on any thread. Counts, not timings; expected values follow
//! from the fixtures and main's recursion by hand, or from main's gate scan
//! replayed on the same token.

use super::recording::*;
use super::*;
use crate::kernel_adapter::sealed::gate_counts::{self, Counts};
use symtn_dense::CpuEngine;
use tenferro_cpu::CpuBatchStrategy;

/// A rank-two structure of `(dims, strides, offset)` blocks.
fn blocks(blocks: &[([usize; 2], [usize; 2], usize)]) -> Arc<BlockStructure> {
    Arc::new(
        BlockStructure::from_blocks_with_rank(
            2,
            blocks
                .iter()
                .enumerate()
                .map(|(key, &(dims, strides, offset))| {
                    BlockSpec::with_key(
                        BlockKey::ordinal(key),
                        dims.to_vec(),
                        strides.to_vec(),
                        offset,
                    )
                    .unwrap()
                })
                .collect(),
        )
        .unwrap(),
    )
}

/// The initialized overwrite of `transform` through the built-in executor,
/// and the counts of the gates of bound `M` it built.
fn overwrite_counts(
    engine: &mut CpuEngine,
    transform: &TreeTransformStructure<f64>,
    dst: &Arc<BlockStructure>,
    src: &Arc<BlockStructure>,
) -> Counts {
    let source = (0..src.required_len().unwrap())
        .map(|v| v as f64 + 1.0)
        .collect::<Vec<_>>();
    let mut out = vec![f64::NAN; dst.required_len().unwrap()];
    let watch = gate_counts::watch(transform.layouts().max_planned_elements());
    tree_transform_structure_overwrite_with_structural_recoupling_raw(
        engine,
        &mut StridedHostKernelAdapter::default(),
        &mut DefaultDenseExecutor::new(),
        &mut TreeTransformWorkspace::default(),
        transform,
        dst,
        src,
        &mut out,
        &source,
        1.0,
        &[],
    )
    .unwrap();
    watch.take()
}

/// Main's `LeafGate::scan` on `exec`'s token for `bound`: its `pays`
/// queries.
fn main_scan_probes(exec: &crate::host_pool::Exec<'_>, bound: usize) -> usize {
    use strided_kernel::execution::KernelClass;
    let cheapest = (2..=exec.budget().get())
        .filter_map(core::num::NonZeroUsize::new)
        .min_by_key(|&width| (exec.dispatch_cost(width), width))
        .unwrap();
    let mut probes = 0;
    let mut pays = |elements: usize| {
        probes += 1;
        exec.pays(KernelClass::Copy.estimate(elements), 1, cheapest)
    };
    if pays(bound) && !pays(0) {
        let (mut below, mut at) = (0usize, bound);
        while at - below > 1 {
            let middle = below + (at - below) / 2;
            if pays(middle) {
                at = middle;
            } else {
                below = middle;
            }
        }
    }
    probes
}

/// What (caller-scan trace): `G` Multi groups of planned `[7, 11]` columns
/// on a four-worker engine whose dispatch never pays replay serially, and
/// the one caller gate serves every role: one scan of one `pays` query,
/// which rejects them all without a width search, for `G` 8, 32 and 128.
#[test]
fn the_caller_gate_scans_once_whatever_the_group_count() {
    for groups in [8, 32, 128] {
        let structure = Arc::new(
            BlockStructure::packed_column_major(2, vec![vec![7, 11]; 2 * groups]).unwrap(),
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
        let transform =
            TreeTransformStructure::compile_structures(&structure, &structure, &specs).unwrap();
        assert_eq!(transform.layouts().retained_plan_count(), 2 * groups);
        let mut engine = flat_engine(4, 1_000_000_000, CpuBatchStrategy::Auto);
        let counts = overwrite_counts(&mut engine, &transform, &structure, &structure);
        let expected = Counts {
            scans: 1,
            pays: 1,
            widths: 0,
        };
        assert_eq!(counts, expected, "{groups} groups");
    }
}

/// The r5 partial-overlap fixture: `G` groups of one `[3, 32]` destination
/// (strides `[4, 6]`, at `196 floor(j / 2) + j mod 2`, so each pair's hulls
/// overlap) from two packed sources, then one `[2, m]` transposing Single
/// after them; `M = 2m`.
fn partial_overlap(
    groups: usize,
    m: usize,
) -> (
    Arc<BlockStructure>,
    Arc<BlockStructure>,
    TreeTransformStructure<f64>,
) {
    let mut dst = (0..groups)
        .map(|j| ([3, 32], [4, 6], 196 * (j / 2) + j % 2))
        .collect::<Vec<_>>();
    dst.push(([2, m], [m, 1], 196 * (groups / 2)));
    let mut src = vec![vec![3, 32]; 2 * groups];
    src.push(vec![2, m]);
    let src = Arc::new(BlockStructure::packed_column_major(2, src).unwrap());
    let mut specs = (0..groups)
        .map(|j| TreeTransformBlockSpec::multi(vec![j], vec![2 * j, 2 * j + 1], vec![1.0, 1.0]))
        .collect::<Vec<_>>();
    specs.push(TreeTransformBlockSpec::single(groups, 2 * groups, 1.0));
    let dst = blocks(&dst);
    let transform = TreeTransformStructure::compile_structures(&dst, &src, &specs).unwrap();
    (dst, src, transform)
}

/// What: on the partial-overlap fixture at four workers and
/// flat 100 ns dispatch, each chunk of four groups forms two hull runs, so
/// its scatter splits over them: each column is offered a two-slot leaf
/// token, the Single its whole four-slot leaf (the caller's gate rejects
/// every planned `[3, 32]` column on the caller's token, so no chunk keeps
/// main's inline route). Every token's cheapest dispatch costs 100 ns, one
/// cost key: the split leaves search it once between them and the caller's
/// gate once more, each within `log2 M + 2` `pays` queries, for `G` 8, 32
/// and 128 and two bounds `M` alike.
#[test]
fn partially_overlapping_scatters_search_once_per_cost_key() {
    for m in [8192usize, 65536] {
        let bound = 2 * ((2 * m).ilog2() as usize + 2);
        for groups in [8, 32, 128] {
            let (dst, src, transform) = partial_overlap(groups, m);
            assert_eq!(transform.layouts().retained_plan_count(), groups + 1);
            assert_eq!(transform.layouts().max_planned_elements(), 2 * m);
            let mut engine = flat_engine(4, 100, CpuBatchStrategy::Auto);
            let counts = overwrite_counts(&mut engine, &transform, &dst, &src);
            assert_eq!(counts.scans, 2, "m {m}, {groups} groups");
            assert!(counts.pays <= bound, "m {m}, {groups} groups: {counts:?}");

            let (offers, _) = recorded(&mut engine, &transform, &dst, &src);
            let mut expected = vec![2; groups];
            expected.push(4);
            assert_eq!(offers, expected, "m {m}, {groups} groups");
        }
    }
}

/// What (r6 Singles trace): a `[2, m]` transposing Single, two scalar
/// Singles and two scalar Multis, consecutive, on four workers at flat
/// 100 ns. The three Singles split as main's recursion (two left on two
/// slots, the right one on two; then one slot each on the left): offers
/// `[1, 1, 2]` in destination order, so the planned Single runs on one slot
/// and no gate probes. The two groups' scatter splits two and two.
#[test]
fn singles_are_offered_the_tokens_of_mains_recursion() {
    let m = 8192;
    let dst = blocks(&[
        ([2, m], [m, 1], 0),
        ([1, 1], [1, 1], 2 * m),
        ([1, 1], [1, 1], 2 * m + 1),
        ([1, 1], [1, 1], 2 * m + 2),
        ([1, 1], [1, 1], 2 * m + 3),
    ]);
    let mut src = vec![vec![2, m]];
    src.extend(vec![vec![1, 1]; 6]);
    let src = Arc::new(BlockStructure::packed_column_major(2, src).unwrap());
    let transform = TreeTransformStructure::compile_structures(
        &dst,
        &src,
        &[
            TreeTransformBlockSpec::single(0, 0, 1.0),
            TreeTransformBlockSpec::single(1, 1, 1.0),
            TreeTransformBlockSpec::single(2, 2, 1.0),
            TreeTransformBlockSpec::multi(vec![3], vec![3, 4], vec![1.0, 1.0]),
            TreeTransformBlockSpec::multi(vec![4], vec![5, 6], vec![1.0, 1.0]),
        ],
    )
    .unwrap();
    assert_eq!(transform.layouts().retained_plan_count(), 1);
    let mut engine = flat_engine(4, 100, CpuBatchStrategy::Auto);
    let (offers, _) = recorded(&mut engine, &transform, &dst, &src);
    assert_eq!(offers, [1, 1, 2, 2, 2]);
    let counts = overwrite_counts(&mut engine, &transform, &dst, &src);
    assert_eq!(counts, Counts::default());
}

/// What (r6 short scatter and P2 trace): two slice-disjoint groups of
/// `[2, 1666]` and `[2, 1667]` transposes (four pack columns, so four
/// workers) at flat 100 ns: the chunk's two groups split over four logical
/// threads, each offered two slots; the two leaf gates share one cost key,
/// so one of them scans with main's global bound, making main's `pays`
/// queries on a two-slot token, and both roles are admitted.
#[test]
fn a_short_scatter_phase_follows_mains_recursion_and_gate() {
    let dst = blocks(&[([2, 1666], [1666, 1], 0), ([2, 1667], [1667, 1], 3332)]);
    let src = Arc::new(
        BlockStructure::packed_column_major(
            2,
            vec![vec![2, 1666], vec![2, 1666], vec![2, 1667], vec![2, 1667]],
        )
        .unwrap(),
    );
    let transform = TreeTransformStructure::compile_structures(
        &dst,
        &src,
        &[
            TreeTransformBlockSpec::multi(vec![0], vec![0, 1], vec![1.0, 1.0]),
            TreeTransformBlockSpec::multi(vec![1], vec![2, 3], vec![1.0, 1.0]),
        ],
    )
    .unwrap();
    let bound = transform.layouts().max_planned_elements();
    assert_eq!(bound, 3334);
    let mut engine = flat_engine(4, 100, CpuBatchStrategy::Auto);
    let (offers, _) = recorded(&mut engine, &transform, &dst, &src);
    assert_eq!(offers, [2, 2]);
    let probes = crate::host_pool::region(
        &mut engine,
        crate::host_pool::HostPoolSite::Replay,
        |exec| main_scan_probes(&exec.limit(core::num::NonZeroUsize::new(2).unwrap()), bound),
    )
    .unwrap();
    let counts = overwrite_counts(&mut engine, &transform, &dst, &src);
    let expected = Counts {
        scans: 1,
        pays: probes,
        widths: 2,
    };
    assert_eq!(counts, expected);
}

/// What (serial token): one planned `[2, m]` transposing Single replays on
/// one worker (one runnable part) on the caller's whole token: under `Auto`
/// on eight workers it is offered all eight and its plan is admitted (one
/// width search); under a forced `Sequential` it is offered one slot and
/// the gate does not scan.
#[test]
fn an_unsplit_move_gets_the_whole_caller_token() {
    let m = 8192;
    let dst = blocks(&[([2, m], [m, 1], 0)]);
    let src = Arc::new(BlockStructure::packed_column_major(2, vec![vec![2, m]]).unwrap());
    let transform = TreeTransformStructure::compile_structures(
        &dst,
        &src,
        &[TreeTransformBlockSpec::single(0, 0, 1.0)],
    )
    .unwrap();
    for (strategy, offer, widths) in [
        (CpuBatchStrategy::Auto, 8, 1),
        (CpuBatchStrategy::Sequential, 1, 0),
    ] {
        let mut engine = flat_engine(8, 100, strategy);
        let (offers, _) = recorded(&mut engine, &transform, &dst, &src);
        assert_eq!(offers, [offer], "{strategy:?}");
        let counts = overwrite_counts(&mut engine, &transform, &dst, &src);
        assert_eq!(counts.widths, widths, "{strategy:?}");
        if strategy == CpuBatchStrategy::Sequential {
            assert_eq!(counts, Counts::default());
        }
    }
}

/// A custom executor that records each grouped submission's jobs and runs
/// and forwards it to the built-in executor's.
#[derive(Default)]
struct Batches {
    inner: DefaultDenseExecutor,
    submissions: Vec<(Vec<DenseGemmBatchJob>, Vec<usize>)>,
}

impl DenseExecutor for Batches {
    fn svd(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.svd(engine, input)
    }

    fn qr(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.qr(engine, input)
    }

    fn eigh(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.eigh(engine, input)
    }

    fn dot_general_into(
        &mut self,
        engine: &mut dyn Engine,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        config: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        self.inner
            .dot_general_into(engine, output, lhs, rhs, config)
    }

    #[allow(clippy::too_many_arguments)]
    fn matmul_batch_axpby_into(
        &mut self,
        engine: &mut dyn Engine,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        jobs: &[DenseGemmBatchJob],
        runs: &[usize],
        alpha: DenseScalar,
        beta: DenseScalar,
    ) -> Result<(), DenseError> {
        self.submissions.push((jobs.to_vec(), runs.to_vec()));
        self.inner
            .matmul_batch_axpby_into(engine, output, lhs, rhs, jobs, runs, alpha, beta)
    }
}

/// What (audit 1): ten identity groups on four free-dispatch workers replay
/// in chunks of four consecutive groups, one grouped submission each:
/// `ceil(10 / 4)` submissions of 4, 4 and 2 jobs, and the values copy.
#[test]
fn each_chunk_is_one_grouped_submission() {
    let groups = 10;
    let structure =
        Arc::new(BlockStructure::packed_column_major(2, vec![vec![4, 6]; 2 * groups]).unwrap());
    let specs = (0..groups)
        .map(|g| {
            TreeTransformBlockSpec::multi(
                vec![2 * g, 2 * g + 1],
                vec![2 * g, 2 * g + 1],
                vec![1.0, 0.0, 0.0, 1.0],
            )
        })
        .collect::<Vec<_>>();
    let transform =
        TreeTransformStructure::compile_structures(&structure, &structure, &specs).unwrap();
    let source = (0..structure.required_len().unwrap())
        .map(|v| v as f64)
        .collect::<Vec<_>>();
    let mut out = vec![f64::NAN; source.len()];
    let mut dense = Batches::default();
    tree_transform_structure_overwrite_with_structural_recoupling_raw(
        &mut engine(4, CpuBatchStrategy::Auto),
        &mut StridedHostKernelAdapter::default(),
        &mut dense,
        &mut TreeTransformWorkspace::default(),
        &transform,
        &structure,
        &structure,
        &mut out,
        &source,
        1.0,
        &[],
    )
    .unwrap();
    let jobs = dense
        .submissions
        .iter()
        .map(|(jobs, _)| jobs.len())
        .collect::<Vec<_>>();
    assert_eq!(jobs, [4, 4, 2]);
    assert_eq!(out, source);
}

/// What (member batch): a three-member overwrite on four free-dispatch
/// workers is one grouped submission of `G N` jobs, plan job outer and
/// member inner, each member's operands at its member stride of the packed
/// scratch and every member sharing its job's coefficients; with one group
/// the members form one affine run; on cheap-dispatch workers, whose
/// members split into member ranges around the submission, too.
#[test]
fn a_wide_member_batch_is_one_member_strided_submission() {
    let members = 3;
    for cheap in [false, true] {
        for (dst, src, transform) in [two_lane_fixture(), identity_multi_fixture()] {
            let plan = transform.recoupling_plan();
            let expected = plan
                .jobs()
                .iter()
                .flat_map(|job| {
                    (0..members).map(move |member| DenseGemmBatchJob {
                        dst_offset: member * plan.destination_len() + job.dst_offset,
                        lhs_offset: member * plan.source_len() + job.lhs_offset,
                        ..*job
                    })
                })
                .collect::<Vec<_>>();
            let source = (0..members * src.required_len().unwrap())
                .map(|v| v as f64 + 1.0)
                .collect::<Vec<_>>();
            let mut out = vec![f64::NAN; members * dst.required_len().unwrap()];
            let mut dense = Batches::default();
            tree_transform_members_overwrite_raw(
                &mut if cheap {
                    flat_engine(4, 1, CpuBatchStrategy::Auto)
                } else {
                    engine(4, CpuBatchStrategy::Auto)
                },
                &mut StridedHostKernelAdapter::default(),
                &mut dense,
                &mut TreeTransformWorkspace::default(),
                &transform,
                &dst,
                &src,
                &mut out,
                &source,
                members,
                &[],
            )
            .unwrap();
            let groups = plan.jobs().len();
            assert_eq!(dense.submissions.len(), 1, "{groups} groups");
            let (jobs, runs) = &dense.submissions[0];
            assert_eq!(jobs, &expected, "{groups} groups");
            if groups == 1 {
                assert_eq!(runs, &[members]);
            }
        }
    }
}

/// The ten-group fixture's dimensions: groups of two packed `ROWS x COLS`
/// blocks.
const GROUPS: usize = 10;
const ROWS: usize = 4;
const COLS: usize = 6;

/// Ten identity groups of two packed `ROWS x COLS` blocks each.
fn ten_identity_groups() -> (Arc<BlockStructure>, TreeTransformStructure<f64>) {
    let structure = Arc::new(
        BlockStructure::packed_column_major(2, vec![vec![ROWS, COLS]; 2 * GROUPS]).unwrap(),
    );
    let specs = (0..GROUPS)
        .map(|g| {
            TreeTransformBlockSpec::multi(
                vec![2 * g, 2 * g + 1],
                vec![2 * g, 2 * g + 1],
                vec![1.0, 0.0, 0.0, 1.0],
            )
        })
        .collect::<Vec<_>>();
    let transform =
        TreeTransformStructure::compile_structures(&structure, &structure, &specs).unwrap();
    (structure, transform)
}

/// An overwrite of `transform` on `budget` workers under `strategy` through
/// the built-in executor, into `workspace`.
fn try_overwrite(
    budget: usize,
    strategy: CpuBatchStrategy,
    workspace: &mut TreeTransformWorkspace<f64>,
    transform: &TreeTransformStructure<f64>,
    structure: &Arc<BlockStructure>,
) -> Result<Vec<f64>, OperationError> {
    let source = (0..structure.required_len().unwrap())
        .map(|v| v as f64)
        .collect::<Vec<_>>();
    let mut out = vec![f64::NAN; source.len()];
    tree_transform_structure_overwrite_with_structural_recoupling_raw(
        &mut engine(budget, strategy),
        &mut StridedHostKernelAdapter::default(),
        &mut DefaultDenseExecutor::new(),
        workspace,
        transform,
        structure,
        structure,
        &mut out,
        &source,
        1.0,
        &[],
    )?;
    Ok(out)
}

/// [`try_overwrite`] of an identity transform, which succeeds and copies.
fn overwrite(
    budget: usize,
    strategy: CpuBatchStrategy,
    workspace: &mut TreeTransformWorkspace<f64>,
    transform: &TreeTransformStructure<f64>,
    structure: &Arc<BlockStructure>,
) -> Vec<f64> {
    let out = try_overwrite(budget, strategy, workspace, transform, structure).unwrap();
    assert_eq!(out, (0..out.len()).map(|v| v as f64).collect::<Vec<_>>());
    out
}

/// What (H fill, audit 2): the ten-group overwrite on `t = 4` free-dispatch
/// workers replays `Q = ceil(10 / 4)` chunks of consecutive groups in
/// `1 + 2 Q` parallel movement phases (one direct, a pack and a scatter per
/// chunk) on every call. A fresh workspace initializes the largest chunk's
/// packed destination, `t` groups of two `ROWS x COLS` columns, inside the
/// first pack; a warm repeat initializes none. Serial replays (one worker)
/// add no parallel phase: after one of a `2 x (32 x 32)` group and one of a
/// `2 x (1 x 1)` group, which leave the scratch short of that chunk with the
/// capacity for it, the same overwrite initializes only the missing tail,
/// on the caller.
#[test]
fn a_parallel_replay_fills_its_destination_scratch_once_in_its_phases() {
    let threads = 4;
    let chunks = GROUPS.div_ceil(threads);
    let phases = 1 + 2 * chunks;
    let largest_chunk = threads.min(GROUPS) * 2 * ROWS * COLS;
    let (structure, transform) = ten_identity_groups();
    let mut workspace = TreeTransformWorkspace::default();
    let call = |workspace: &mut TreeTransformWorkspace<f64>| {
        let (fills, phases) = (workspace.destination_fills(), workspace.movement_phases());
        overwrite(
            threads,
            CpuBatchStrategy::Auto,
            workspace,
            &transform,
            &structure,
        );
        (
            workspace.destination_fills() - fills,
            workspace.movement_phases() - phases,
        )
    };
    assert_eq!(call(&mut workspace), (largest_chunk, phases), "fresh");
    assert_eq!(call(&mut workspace), (0, phases), "warm");

    let structure_of = |dims: [usize; 2]| {
        Arc::new(BlockStructure::packed_column_major(2, vec![dims.to_vec(); 2]).unwrap())
    };
    let one_group = |structure: &Arc<BlockStructure>| {
        let spec = TreeTransformBlockSpec::multi(vec![0, 1], vec![0, 1], vec![1.0, 0.0, 0.0, 1.0]);
        TreeTransformStructure::compile_structures(structure, structure, &[spec]).unwrap()
    };
    let mut history = TreeTransformWorkspace::default();
    for dims in [[32, 32], [1, 1]] {
        let structure = structure_of(dims);
        overwrite(
            1,
            CpuBatchStrategy::Auto,
            &mut history,
            &one_group(&structure),
            &structure,
        );
    }
    assert_eq!(
        (history.movement_phases(), history.destination_fills()),
        (0, 0),
        "serial"
    );
    let prefix = history.destination_len();
    assert!(prefix < largest_chunk && 2 * 32 * 32 >= largest_chunk);
    assert_eq!(call(&mut history), (largest_chunk - prefix, phases), "tail");
}

/// What (step-5 review P1-1): a forced policy's replay width is capped by
/// the compiled runnable parts, so one rank-nine Single (fused rank nine,
/// above the inline index width) retains the same traversal scratch on two
/// workers and on sixty-four, under `Inner` and under `Split`, as main's
/// runnable clamp does. An executor that cannot honor a forced policy
/// refuses it instead, before any write.
#[test]
fn a_forced_width_sizes_no_scratch_beyond_the_runnable_parts() {
    let shape = vec![2; 9];
    let strides = (0..9).map(|axis| 1usize << axis).collect::<Vec<_>>();
    let block = BlockSpec::with_key(BlockKey::ordinal(0), shape, strides, 0).unwrap();
    let structure = Arc::new(BlockStructure::from_blocks_with_rank(9, vec![block]).unwrap());
    let specs = vec![TreeTransformBlockSpec::single(0, 0, 1.0_f64).with_source_axes((0..9).rev())];
    let transform =
        TreeTransformStructure::compile_structures(&structure, &structure, &specs).unwrap();
    let split = CpuBatchStrategy::Split {
        lanes: core::num::NonZeroUsize::new(2).unwrap(),
    };
    let honors = DefaultDenseExecutor::new().honors_forced_batch_policy();
    for strategy in [CpuBatchStrategy::Inner, split] {
        let retained = |budget: usize| {
            let mut workspace = TreeTransformWorkspace::default();
            try_overwrite(budget, strategy, &mut workspace, &transform, &structure)
                .map(|_| workspace.retained_bytes())
        };
        if honors {
            assert_eq!(retained(2).unwrap(), retained(64).unwrap(), "{strategy:?}");
        } else {
            for budget in [2, 64] {
                assert!(
                    matches!(
                        retained(budget),
                        Err(OperationError::Dense(DenseError::Unsupported {
                            op: "tree_transform_replay",
                            ..
                        }))
                    ),
                    "{strategy:?} on {budget}"
                );
            }
        }
    }
}

/// Three slice-disjoint Singles (two scalars, then a `[2, m]` block written
/// transposed) followed by `inactive` scalar destinations, and the Singles'
/// coefficients.
fn singles_then_inactive(
    m: usize,
    scalars: bool,
    inactive: usize,
) -> (
    Arc<BlockStructure>,
    Arc<BlockStructure>,
    TreeTransformStructure<f64>,
    Vec<f64>,
) {
    let mut dst = Vec::new();
    let mut src = Vec::new();
    if scalars {
        dst.extend([([1, 1], [1, 1], 0), ([1, 1], [1, 1], 1)]);
        src.extend([vec![1, 1], vec![1, 1]]);
    }
    let big = dst.len();
    dst.push(([2, m], [m, 1], big));
    src.push(vec![2, m]);
    let end = big + 2 * m;
    dst.extend((0..inactive).map(|k| ([1, 1], [1, 1], end + k)));
    let coefficients: Vec<f64> = [2.0, -1.0, 0.5][3 - (big + 1)..].to_vec();
    let specs = (0..=big)
        .map(|b| TreeTransformBlockSpec::single(b, b, coefficients[b]))
        .collect::<Vec<_>>();
    let src = Arc::new(BlockStructure::packed_column_major(2, src).unwrap());
    let dst = blocks(&dst);
    let transform = TreeTransformStructure::compile_structures(&dst, &src, &specs).unwrap();
    (dst, src, transform, coefficients)
}

/// The overwrite of [`singles_then_inactive`] from source `v + 1` by hand:
/// scalar Single `b` is `c_b src_b`, the big block `(i, j)` (destination
/// strides `[m, 1]`) is `c src(i, j)` of its column-major source, the
/// inactive scalars are zero.
fn singles_then_inactive_expected(m: usize, coefficients: &[f64], inactive: usize) -> Vec<f64> {
    let big = coefficients.len() - 1;
    let source = |at: usize| at as f64 + 1.0;
    let mut expected = vec![0.0; big + 2 * m + inactive];
    for b in 0..big {
        expected[b] = coefficients[b] * source(b);
    }
    for i in 0..2 {
        for j in 0..m {
            expected[big + i * m + j] = coefficients[big] * source(big + i + 2 * j);
        }
    }
    expected
}

/// An overwrite of `transform` from source `v + 1` on `engine`.
fn overwritten(
    engine: &mut CpuEngine,
    transform: &TreeTransformStructure<f64>,
    dst: &Arc<BlockStructure>,
    src: &Arc<BlockStructure>,
) -> Vec<f64> {
    let source = (0..src.required_len().unwrap())
        .map(|v| v as f64 + 1.0)
        .collect::<Vec<_>>();
    let mut out = vec![f64::NAN; dst.required_len().unwrap()];
    tree_transform_structure_overwrite_with_structural_recoupling_raw(
        engine,
        &mut StridedHostKernelAdapter::default(),
        &mut DefaultDenseExecutor::new(),
        &mut TreeTransformWorkspace::default(),
        transform,
        dst,
        src,
        &mut out,
        &source,
        1.0,
        &[],
    )
    .unwrap();
    out
}

/// What: three slice-disjoint Singles keep main's
/// split at its width `t_0 = 3` on four flat-cost workers, though the four
/// inactive scalars after them form four runs (run width 4): the Singles
/// are offered `[2, 1, 1]` in destination order, so the planned `[2, m]`
/// Single runs on one slot and no gate searches; the inactive runs split
/// and the values are the hand oracle's.
#[test]
fn inactive_runs_leave_mains_singles_split_alone() {
    // A bound no other gate-counting test uses.
    let m = 8200;
    let (dst, src, transform, coefficients) = singles_then_inactive(m, true, 4);
    assert_eq!(transform.parallel_schedule().inactive_runs, [0, 1, 2, 3]);
    assert_eq!(transform.parallel_schedule().replay_runnable, 3);
    let mut engine = flat_engine(4, 100, CpuBatchStrategy::Auto);
    let (offers, _) = recorded(&mut engine, &transform, &dst, &src);
    assert_eq!(offers, [2, 1, 1]);
    let counts = overwrite_counts(&mut engine, &transform, &dst, &src);
    assert_eq!(counts, Counts::default());
    assert_eq!(
        overwritten(&mut engine, &transform, &dst, &src),
        singles_then_inactive_expected(m, &coefficients, 4)
    );
}

/// What: one planned `[2, m]` Single and four
/// inactive scalars, no pack columns: the replay width is one, but the
/// inactive layouts form four runs and split (one parallel movement phase),
/// while the Single keeps main's inline route on the caller's whole
/// four-slot token; the values are the hand oracle's.
#[test]
fn inactive_runs_split_under_a_serial_replay_width() {
    let m = 8300;
    let (dst, src, transform, coefficients) = singles_then_inactive(m, false, 4);
    assert_eq!(transform.parallel_schedule().replay_runnable, 1);
    let mut engine = flat_engine(4, 100, CpuBatchStrategy::Auto);
    let (offers, _) = recorded(&mut engine, &transform, &dst, &src);
    assert_eq!(offers, [4]);
    let source = (0..src.required_len().unwrap())
        .map(|v| v as f64 + 1.0)
        .collect::<Vec<_>>();
    let mut out = vec![f64::NAN; dst.required_len().unwrap()];
    let mut workspace = TreeTransformWorkspace::default();
    tree_transform_structure_overwrite_with_structural_recoupling_raw(
        &mut engine,
        &mut StridedHostKernelAdapter::default(),
        &mut DefaultDenseExecutor::new(),
        &mut workspace,
        &transform,
        &dst,
        &src,
        &mut out,
        &source,
        1.0,
        &[],
    )
    .unwrap();
    assert_eq!(workspace.movement_phases(), 1);
    assert_eq!(out, singles_then_inactive_expected(m, &coefficients, 4));
}

/// What: on three workers, three disjoint inactive layouts
/// and three one-source, two-destination groups of `0`, `m` and `m` rows
/// (the empty group first in plan order): the two pack columns admit main's
/// width `t_0 = 2` for the groups, whatever the three inactive runs admit
/// for their own phase, so the chunks, the grouped submissions and the
/// packed scratch are those of consecutive chunks of two jobs, derived here
/// from the jobs' geometry, and the copy is exact.
#[test]
fn inactive_runs_leave_the_chunk_width_alone() {
    let m = 4096;
    let mut layout = Vec::new();
    let mut offset = 0;
    for rows in [0, m, m] {
        for _ in 0..2 {
            layout.push(([rows, 1], [1, rows.max(1)], offset));
            offset += rows;
        }
    }
    for _ in 0..3 {
        layout.push(([1, 1], [1, 1], offset));
        offset += 1;
    }
    let dst = blocks(&layout);
    let src = Arc::new(
        BlockStructure::packed_column_major(2, [0, m, m].map(|rows| vec![rows, 1])).unwrap(),
    );
    let specs = (0..3)
        .map(|g| TreeTransformBlockSpec::multi(vec![2 * g, 2 * g + 1], vec![g], vec![1.0, 2.0]))
        .collect::<Vec<_>>();
    let transform = TreeTransformStructure::compile_structures(&dst, &src, &specs).unwrap();
    let schedule = transform.parallel_schedule();
    assert_eq!(schedule.pack_columns.len(), 2);
    assert_eq!(schedule.inactive_runs.len(), 3);
    let width = 2;
    let jobs = transform.recoupling_plan().jobs();
    let span = |chunk: &[DenseGemmBatchJob], of: fn(&DenseGemmBatchJob) -> (usize, usize)| {
        let (first, _) = of(&chunk[0]);
        let (offset, len) = of(&chunk[chunk.len() - 1]);
        offset + len - first
    };
    let source_len = jobs
        .chunks(width)
        .map(|chunk| span(chunk, |job| (job.lhs_offset, job.rows * job.contracted)))
        .max()
        .unwrap();
    let destination_len = jobs
        .chunks(width)
        .map(|chunk| span(chunk, |job| (job.dst_offset, job.rows * job.cols)))
        .max()
        .unwrap();
    let source = (0..src.required_len().unwrap())
        .map(|v| v as f64 + 1.0)
        .collect::<Vec<_>>();
    let mut out = vec![f64::NAN; dst.required_len().unwrap()];
    let mut workspace = TreeTransformWorkspace::default();
    let mut dense = Batches::default();
    tree_transform_structure_overwrite_with_structural_recoupling_raw(
        &mut flat_engine(3, 1, CpuBatchStrategy::Auto),
        &mut StridedHostKernelAdapter::default(),
        &mut dense,
        &mut workspace,
        &transform,
        &dst,
        &src,
        &mut out,
        &source,
        1.0,
        &[],
    )
    .unwrap();
    let submitted = dense
        .submissions
        .iter()
        .map(|(jobs, _)| jobs.len())
        .collect::<Vec<_>>();
    let expected = jobs.chunks(width).map(<[_]>::len).collect::<Vec<_>>();
    assert_eq!(submitted, expected);
    assert_eq!(workspace.packed.source().capacity(), source_len);
    assert_eq!(workspace.packed.destination().len(), destination_len);
    let mut values = vec![0.0; out.len()];
    let mut at = 0;
    for (g, rows) in [0, m, m].into_iter().enumerate() {
        let first = [0, 0, m][g];
        for (column, coefficient) in [1.0, 2.0].into_iter().enumerate() {
            for row in 0..rows {
                values[at + column * rows + row] = coefficient * source[first + row];
            }
        }
        at += 2 * rows;
    }
    assert_eq!(out, values);
}

/// What: pairs of `[1, 2]` destinations of stride 2 interleave
/// at `4 i` and `4 i + 1`, and group `i` writes the second block of pair
/// `i` and the first of pair `i + 1`, so neighbouring group hulls overlap
/// and every chunk of four groups is one hull run: each scatter column
/// stays inline on the caller's whole four-slot token.
#[test]
fn a_chain_of_overlapping_groups_scatters_inline() {
    let groups = 12;
    let dst = blocks(
        &(0..=groups)
            .flat_map(|i| [([1, 2], [1, 2], 4 * i), ([1, 2], [1, 2], 4 * i + 1)])
            .collect::<Vec<_>>(),
    );
    let src =
        Arc::new(BlockStructure::packed_column_major(2, vec![vec![1, 2]; 2 * groups + 2]).unwrap());
    let mut specs = (0..groups)
        .map(|i| {
            TreeTransformBlockSpec::multi(
                vec![2 * i + 1, 2 * i + 2],
                vec![2 * i + 1, 2 * i + 2],
                vec![1.0, 0.5, 0.5, 1.0],
            )
        })
        .collect::<Vec<_>>();
    specs.push(TreeTransformBlockSpec::single(0, 0, 1.0));
    specs.push(TreeTransformBlockSpec::single(
        2 * groups + 1,
        2 * groups + 1,
        1.0,
    ));
    let transform = TreeTransformStructure::compile_structures(&dst, &src, &specs).unwrap();
    let mut engine = flat_engine(4, 1, CpuBatchStrategy::Auto);
    let (offers, _) = recorded(&mut engine, &transform, &dst, &src);
    // The two Singles first and last in address order; every scatter
    // column between them inline.
    assert_eq!(offers.len(), 2 * groups + 2);
    assert!(
        offers[1..offers.len() - 1].iter().all(|&offer| offer == 4),
        "{offers:?}"
    );
}

/// `pairs` coupled pairs of `[2, cols]` Singles whose destinations
/// interleave (strides `[1, 4]`, offsets `4 cols p` and `4 cols p + 2`):
/// the Singles are not slice-disjoint and form one hull run per pair.
fn interleaved_single_pairs(
    pairs: usize,
    cols: usize,
) -> (
    Arc<BlockStructure>,
    Arc<BlockStructure>,
    TreeTransformStructure<f64>,
) {
    let dst = blocks(
        &(0..pairs)
            .flat_map(|p| {
                [
                    ([2, cols], [1, 4], 4 * cols * p),
                    ([2, cols], [1, 4], 4 * cols * p + 2),
                ]
            })
            .collect::<Vec<_>>(),
    );
    let src =
        Arc::new(BlockStructure::packed_column_major(2, vec![vec![2, cols]; 2 * pairs]).unwrap());
    let specs = (0..2 * pairs)
        .map(|b| TreeTransformBlockSpec::single(b, b, 1.0))
        .collect::<Vec<_>>();
    let transform = TreeTransformStructure::compile_structures(&dst, &src, &specs).unwrap();
    let schedule = transform.parallel_schedule();
    assert!(!schedule.singles_slice_disjoint());
    assert_eq!(schedule.single_runs.len(), pairs);
    assert_eq!(schedule.singles_planned, 2 * cols);
    (dst, src, transform)
}

/// What: when the caller's gate admits the plan of the largest planned
/// Single on the caller's token, Singles that are not slice-disjoint keep
/// main's inline route even though they form two runs: on four flat-cost
/// workers each `[2, m]` Single is offered the caller's whole four-slot
/// token and its plan is admitted.
#[test]
fn an_admitted_planned_single_keeps_mains_inline_route() {
    // A bound no other gate-counting test uses.
    let (dst, src, transform) = interleaved_single_pairs(2, 8250);
    let mut engine = flat_engine(4, 100, CpuBatchStrategy::Auto);
    let (offers, _) = recorded(&mut engine, &transform, &dst, &src);
    assert_eq!(offers, [4; 4]);
    let counts = overwrite_counts(&mut engine, &transform, &dst, &src);
    assert!(counts.widths > 0, "{counts:?}");
}

/// What: when the caller's gate rejects the plan of the largest planned
/// Single on the caller's token, Singles that are not slice-disjoint split
/// over their runs: on four flat-cost workers every one of 64 interleaved
/// pairs of `[2, 64]` Singles is written on a split leaf's smaller token.
#[test]
fn a_rejected_planned_single_splits_over_its_runs() {
    let (dst, src, transform) = interleaved_single_pairs(64, 64);
    let mut engine = flat_engine(4, 100, CpuBatchStrategy::Auto);
    let (offers, _) = recorded(&mut engine, &transform, &dst, &src);
    assert_eq!(offers.len(), 128);
    assert!(offers.iter().all(|&offer| offer < 4), "{offers:?}");
}
