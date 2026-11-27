//! Structural replay contracts: grouped submissions, token ownership,
//! scratch first writes, and independent dense-value checks.

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
/// and the counts of the gates it built.
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
    let watch = gate_counts::watch();
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
    fn honors_forced_batch_policy(&self) -> bool {
        self.inner.honors_forced_batch_policy()
    }

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

/// Each public Outer-policy chunk makes one grouped dense submission and
/// preserves every copied value.
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
        &mut engine(4, CpuBatchStrategy::Outer),
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
    let expected = transform
        .recoupling_plan()
        .jobs()
        .chunks(4)
        .map(<[_]>::len)
        .collect::<Vec<_>>();
    assert_eq!(jobs, expected);
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

/// Cold packed destinations are initialized once by the pack workers; a
/// warm repeat adds no fill. Growing within retained capacity fills only the tail.
#[test]
fn a_parallel_replay_fills_its_destination_scratch_once_in_its_phases() {
    let threads = 4;
    let largest_chunk = threads.min(GROUPS) * 2 * ROWS * COLS;
    let (structure, transform) = ten_identity_groups();
    // Auto may choose a different chunk width after calibration; only the
    // initialized scratch length is contractual, not its scheduling choice.
    let mut auto = TreeTransformWorkspace::default();
    overwrite(
        threads,
        CpuBatchStrategy::Auto,
        &mut auto,
        &transform,
        &structure,
    );
    let cold_fills = auto.destination_fills();
    assert_eq!(cold_fills, auto.destination_len());
    overwrite(
        threads,
        CpuBatchStrategy::Auto,
        &mut auto,
        &transform,
        &structure,
    );
    assert_eq!(auto.destination_fills(), cold_fills);
    let mut workspace = TreeTransformWorkspace::default();
    let call = |workspace: &mut TreeTransformWorkspace<f64>| {
        let fills = workspace.destination_fills();
        overwrite(
            threads,
            CpuBatchStrategy::Outer,
            workspace,
            &transform,
            &structure,
        );
        workspace.destination_fills() - fills
    };
    assert_eq!(call(&mut workspace), largest_chunk, "fresh");
    assert_eq!(call(&mut workspace), 0, "warm");

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
    assert_eq!(call(&mut history), largest_chunk - prefix, "tail");
}

/// A one-item forced phase retains the same traversal scratch at different
/// engine budgets; an executor unable to honor the policy refuses it.
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

/// Inactive zero fills preserve the hand-computed values of scalar Singles
/// and a transposed Single sharing the destination.
#[test]
fn inactive_fills_preserve_transposed_single_values() {
    // A bound no other gate-counting test uses.
    let m = 8200;
    let (dst, src, transform, coefficients) = singles_then_inactive(m, true, 4);
    assert_eq!(transform.parallel_schedule().inactive_runs, [0, 1, 2, 3]);
    let mut engine = flat_engine(4, 100, CpuBatchStrategy::Auto);
    assert_eq!(
        overwritten(&mut engine, &transform, &dst, &src),
        singles_then_inactive_expected(m, &coefficients, 4)
    );
}

/// Inactive zero fills preserve a dominant transposed Single's values.
#[test]
fn inactive_fills_keep_a_dominant_single_correct() {
    let m = 8300;
    let (dst, src, transform, coefficients) = singles_then_inactive(m, false, 4);
    let mut engine = flat_engine(4, 100, CpuBatchStrategy::Auto);
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
    assert_eq!(out, singles_then_inactive_expected(m, &coefficients, 4));
}

/// Forced chunking preserves empty groups, inactive zeros and the hand
/// matrix values, with scratch derived from the submitted geometry.
#[test]
fn forced_chunks_preserve_empty_groups_and_inactive_values() {
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
        &mut flat_engine(
            3,
            1,
            CpuBatchStrategy::Split {
                lanes: core::num::NonZeroUsize::new(width).unwrap(),
            },
        ),
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

/// `groups` Multi groups whose hulls overlap in a chain (pairs of `[1, 2]`
/// destinations of stride 2 at `4 i` and `4 i + 1`; group `i` writes the
/// second block of pair `i` and the first of pair `i + 1`), between two
/// Singles at the first and last pairs: one ownership run.
fn overlapping_group_chain(
    groups: usize,
) -> (
    Arc<BlockStructure>,
    Arc<BlockStructure>,
    TreeTransformStructure<f64>,
) {
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
    (dst, src, transform)
}

/// Purpose: one transitive group ownership run keeps the whole token. Pairs of `[1, 2]` destinations of stride 2 interleave
/// at `4 i` and `4 i + 1`, and group `i` writes the second block of pair
/// `i` and the first of pair `i + 1`, so neighbouring group hulls overlap
/// and every chunk of four groups is one hull run: each scatter column
/// stays inline on the caller's whole four-slot token.
#[test]
fn a_chain_of_overlapping_groups_scatters_inline() {
    let groups = 12;
    let (dst, src, transform) = overlapping_group_chain(groups);
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

/// Purpose: an Auto engine whose calibration failed fails closed in
/// execution: a planned Single and every scatter column are offered one slot,
/// so nothing forks inside a modeled phase and no plan is widened; the same
/// engine with calibrated constants is the control that offers the whole
/// four-slot token and widens the planned Single.
#[test]
fn a_failed_calibration_runs_modeled_movement_on_one_slot() {
    let (single_dst, single_src, single) = interleaved_single_pairs(1, 8250);
    let (chain_dst, chain_src, chain) = overlapping_group_chain(12);
    for calibrated in [true, false] {
        let mut engine = flat_engine(4, 100, CpuBatchStrategy::Auto);
        if !calibrated {
            symtn_dense::engine_slot::clear(&mut engine);
            super::super::movement::install_test_constants(&mut engine, None);
        }
        let offer = if calibrated { 4 } else { 1 };
        let (offers, _) = recorded(&mut engine, &single, &single_dst, &single_src);
        assert_eq!(offers, [offer; 2], "calibrated: {calibrated}");
        let counts = overwrite_counts(&mut engine, &single, &single_dst, &single_src);
        assert_eq!(counts.widths > 0, calibrated, "{counts:?}");
        let (offers, _) = recorded(&mut engine, &chain, &chain_dst, &chain_src);
        assert!(
            offers[1..offers.len() - 1].iter().all(|&o| o == offer),
            "calibrated: {calibrated}, {offers:?}"
        );
    }
}

/// Purpose: calibrated constants evaluate the width model for a cold
/// decision of several jobs; a failed calibration is the control that
/// decides without evaluating it.
#[test]
fn calibrated_constants_evaluate_the_width_model() {
    let (dst, src, transform) = interleaved_single_pairs(4, 64);
    for calibrated in [true, false] {
        let mut engine = flat_engine(4, 100, CpuBatchStrategy::Auto);
        if !calibrated {
            symtn_dense::engine_slot::clear(&mut engine);
            super::super::movement::install_test_constants(&mut engine, None);
        }
        crate::probe::take(crate::probe::Event::MovementModelEvaluation);
        recorded(&mut engine, &transform, &dst, &src);
        assert_eq!(
            crate::probe::take(crate::probe::Event::MovementModelEvaluation) > 0,
            calibrated,
            "calibrated: {calibrated}"
        );
    }
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
    (dst, src, transform)
}

/// Purpose: one indivisible hull job keeps every planned role on the whole
/// token, so the inline gate can still admit its plans.
#[test]
fn an_admitted_planned_single_keeps_mains_inline_route() {
    let (dst, src, transform) = interleaved_single_pairs(1, 8250);
    let mut engine = flat_engine(4, 100, CpuBatchStrategy::Auto);
    let (offers, _) = recorded(&mut engine, &transform, &dst, &src);
    assert_eq!(offers, [4; 2]);
    let counts = overwrite_counts(&mut engine, &transform, &dst, &src);
    assert!(counts.widths > 0, "{counts:?}");
}

/// Purpose: all forced split leaves share one search per dispatch-cost key,
/// including balanced three-slot leaves absent from a halving-only key list.
#[test]
fn balanced_split_leaves_share_their_gate_cost_key() {
    let budget = 8;
    let lanes = 3;
    let m = 8193;
    let dst = blocks(
        &(0..lanes)
            .map(|i| ([2, m], [m, 1], 2 * m * i))
            .collect::<Vec<_>>(),
    );
    let src =
        Arc::new(BlockStructure::packed_column_major(2, vec![vec![2, m]; 2 * lanes]).unwrap());
    let specs = (0..lanes)
        .map(|i| TreeTransformBlockSpec::multi(vec![i], vec![2 * i, 2 * i + 1], vec![1.0, 1.0]))
        .collect::<Vec<_>>();
    let transform = TreeTransformStructure::compile_structures(&dst, &src, &specs).unwrap();
    let mut engine = flat_engine(budget, 100, CpuBatchStrategy::Outer);
    let (offers, _) = recorded(&mut engine, &transform, &dst, &src);
    let expected = (0..lanes)
        .map(|i| budget / lanes + usize::from(i < budget % lanes))
        .collect::<Vec<_>>();
    assert_eq!(offers, expected);
    let counts = overwrite_counts(&mut engine, &transform, &dst, &src);
    // Flat dispatch makes every nonserial leaf's cheapest cost one shared key.
    let keys = expected
        .iter()
        .filter(|&&share| share > 1)
        .map(|_| 100)
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(counts.scans, keys.len());
    assert_eq!(counts.widths, lanes);
    // A serial engine cannot search a team cost key.
    let counts = overwrite_counts(
        &mut flat_engine(1, 100, CpuBatchStrategy::Outer),
        &transform,
        &dst,
        &src,
    );
    assert_eq!(counts.scans, 0);
}
