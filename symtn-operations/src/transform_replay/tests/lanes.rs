//! The lane replay's contracts: one writer per destination element, each
//! stage at its selected width, results independent of the engine width,
//! and job-local errors with the lowest-indexed component reported.

use super::recording::*;
use super::*;
use std::sync::Mutex;
use std::time::Duration;
use strided_kernel::execution::KernelClass;
use symtn_dense::lanes::{Estimate, Exec as LaneExec, LaneGemmIo, LaneGemmShape, LaneGemms};
use symtn_dense::{CpuEngine, DenseDType, Pool};
use tenferro_cpu::threadteam::{DispatchCosts, WidthCosts};
use tenferro_cpu::{CpuBatchPolicy, CpuBatchStrategy};

use super::super::writers::ReplayParts;

/// An engine of `threads` workers whose dispatch costs `d` ns at every
/// width, for every dispatch kind.
fn costed_engine(threads: usize, d: u64) -> CpuEngine {
    let costs = WidthCosts::new([Duration::from_nanos(d); 3]);
    CpuEngine::new(
        Pool::builder()
            .threads(threads)
            .dispatch_costs(DispatchCosts::new(costs, costs, costs))
            .build()
            .unwrap(),
    )
}

/// The default executor, observed: each lane GEMM records the budget its
/// context offers (a one-worker GEMM records one), `estimate` (when set)
/// replaces every GEMM's estimate, and the groups in `fail` fail on either
/// route.
struct Spy {
    inner: DefaultDenseExecutor,
    estimate: Option<Duration>,
    fail: Vec<usize>,
    /// The recoupling plan's `rhs_offset` per group, which names the group
    /// of a one-worker submission.
    rhs_offsets: Vec<usize>,
    budgets: Mutex<Vec<(usize, usize)>>,
}

impl Spy {
    fn new(transform: &TreeTransformStructure<f64>) -> Self {
        Self {
            inner: DefaultDenseExecutor::new(),
            estimate: None,
            fail: Vec::new(),
            rhs_offsets: transform
                .recoupling_plan()
                .jobs()
                .iter()
                .map(|job| job.rhs_offset)
                .collect(),
            budgets: Mutex::new(Vec::new()),
        }
    }

    fn budgets(&self) -> Vec<(usize, usize)> {
        let mut budgets = self.budgets.lock().unwrap().clone();
        budgets.sort_unstable();
        budgets
    }
}

struct SpyGemms<'a> {
    inner: &'a dyn LaneGemms,
    estimate: Option<Duration>,
    fail: &'a [usize],
    budgets: &'a Mutex<Vec<(usize, usize)>>,
}

impl LaneGemms for SpyGemms<'_> {
    fn run(&self, exec: LaneExec<'_>, job: usize, io: LaneGemmIo<'_>) -> Result<(), DenseError> {
        self.budgets.lock().unwrap().push((job, exec.budget()));
        if self.fail.contains(&job) {
            return Err(injected(job));
        }
        self.inner.run(exec, job, io)
    }

    fn estimate(&self, job: usize) -> Estimate {
        match self.estimate {
            Some(serial) => Estimate {
                serial,
                payback: None,
                sections: 1,
                widen: true,
            },
            None => self.inner.estimate(job),
        }
    }
}

impl DenseExecutor for Spy {
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
        for job in jobs {
            let Some(group) = self.rhs_offsets.iter().position(|&o| o == job.rhs_offset) else {
                continue;
            };
            self.budgets.lock().unwrap().push((group, 1));
            if self.fail.contains(&group) {
                return Err(injected(group));
            }
        }
        self.inner
            .matmul_batch_axpby_into(engine, output, lhs, rhs, jobs, runs, alpha, beta)
    }

    fn supports_lane_gemms(&self) -> bool {
        true
    }

    fn with_lane_gemms(
        &mut self,
        engine: &mut dyn Engine,
        dtype: DenseDType,
        jobs: &[LaneGemmShape],
        body: &mut symtn_dense::lanes::LaneGemmBody<'_>,
    ) -> Result<(), DenseError> {
        let Self {
            inner,
            estimate,
            fail,
            budgets,
            ..
        } = self;
        inner.with_lane_gemms(engine, dtype, jobs, &mut |exec, gemms| {
            body(
                exec,
                &SpyGemms {
                    inner: gemms,
                    estimate: *estimate,
                    fail,
                    budgets,
                },
            )
        })
    }
}

/// Runs the lane replay of `transform` on `engine` through the recording
/// writer and returns every element's mark.
fn record(
    engine: &mut dyn Engine,
    dense: &mut Spy,
    transform: &TreeTransformStructure<f64>,
    len: usize,
    source: &[f64],
) -> (Vec<Mark>, TreeTransformWorkspace<f64>) {
    let (ran, marks, workspace) = try_record(engine, dense, transform, len, source);
    assert!(ran, "placement left the lane runner");
    (marks, workspace)
}

/// [`record`], also returning whether placement kept the lane runner.
fn try_record(
    engine: &mut dyn Engine,
    dense: &mut Spy,
    transform: &TreeTransformStructure<f64>,
    len: usize,
    source: &[f64],
) -> (bool, Vec<Mark>, TreeTransformWorkspace<f64>) {
    let task = transform.task_view().unwrap();
    let mut workspace = TreeTransformWorkspace::default();
    ensure_recoupling_coefficients(&mut workspace, task, transform.identity_marker()).unwrap();
    let mut marks = vec![Mark::default(); len];
    let recorder = Recorder {
        parts: ReplayParts::new(task),
    };
    let ran = super::super::lanes::lane_replay(
        engine,
        dense,
        &mut workspace,
        task,
        transform.parallel_schedule(),
        &mut marks,
        source,
        &recorder,
        None,
        1,
    )
    .unwrap();
    (ran, marks, workspace)
}

/// What: on a four-worker engine, every destination element is written
/// exactly once, by one worker; a band of the interleaved atom is never
/// split, its bands go to more than one worker, and the injective
/// `[3, 2]` / `[2, 3]` block is written whole by one worker.
#[test]
fn every_destination_element_has_one_writer() {
    let (blocks, dst, _, transform) = interleaved_bandable(true);
    let len = dst.required_len().unwrap();
    let source = (0..len).map(|v| v as f64).collect::<Vec<_>>();
    let mut engine = crate::host_pool::eager_test_engine(4);
    let mut dense = Spy::new(&transform);
    let (marks, _) = record(&mut engine, &mut dense, &transform, len, &source);

    assert_one_writer_per_element(&blocks, &marks);
}

/// What: a lone bandable component whose total estimate pays for no
/// parallel section is admitted on one slot, so it leaves the lane runner
/// for the one-worker replay at budgets six and eight.
#[test]
fn a_call_whose_total_pays_for_no_section_leaves_the_lane_runner() {
    let (_, dst, _, transform) = interleaved_bandable(false);
    assert_eq!(transform.parallel_schedule().component_count(), 1);
    let len = dst.required_len().unwrap();
    let source = (0..len).map(|v| v as f64).collect::<Vec<_>>();
    for budget in [6, 8] {
        let mut engine = costed_engine(budget, 60);
        let mut dense = Spy::new(&transform);
        let (ran, _, _) = try_record(&mut engine, &mut dense, &transform, len, &source);
        assert!(!ran, "budget {budget}");
    }
}

/// What: one Multi group whose GEMM pays for every width (606 ns against a
/// 60 ns dispatch) and whose moves pay for none: the moves run on one
/// slot, while the GEMM's context offers six slots at budget six and four
/// at budget four.
#[test]
fn moves_run_at_one_slot_and_the_gemm_at_its_own_width() {
    let (dst, src, transform) = identity_multi_fixture();
    let len = dst.required_len().unwrap();
    let source = vec![3.0, 4.0];
    assert_eq!(src.required_len().unwrap(), source.len());
    for budget in [6, 4] {
        let mut engine = costed_engine(budget, 60);
        let mut dense = Spy::new(&transform);
        dense.estimate = Some(Duration::from_nanos(606));
        let (marks, _) = record(&mut engine, &mut dense, &transform, len, &source);
        let caller = Some(std::thread::current().id());
        assert!(marks
            .iter()
            .all(|mark| mark.writes == 1 && mark.worker == caller));
        assert_eq!(dense.budgets(), [(0, budget)]);
    }
}

/// What: a zero-estimate bandable Single queued on the lane of a GEMM wide
/// enough for six slots still runs its moves at its own width, one slot,
/// while the GEMM is offered its width.
#[test]
fn a_narrow_job_on_a_wide_lane_runs_at_its_own_width() {
    // Precondition: Strided estimates a two-element copy at zero time, so
    // the Single is a zero-estimate job.
    assert!(KernelClass::Copy.estimate(2).is_zero());
    let dst = Arc::new(
        BlockStructure::from_blocks_with_rank(
            1,
            vec![
                BlockSpec::with_key(BlockKey::ordinal(0), vec![1], vec![1], 0).unwrap(),
                BlockSpec::with_key(BlockKey::ordinal(1), vec![1], vec![1], 1).unwrap(),
                BlockSpec::with_key(BlockKey::ordinal(2), vec![2], vec![1], 2).unwrap(),
            ],
        )
        .unwrap(),
    );
    let src =
        Arc::new(BlockStructure::packed_column_major(1, [vec![1], vec![1], vec![2]]).unwrap());
    let transform = TreeTransformStructure::compile_structures(
        &dst,
        &src,
        &[
            TreeTransformBlockSpec::multi(vec![0, 1], vec![0, 1], vec![1.0, 0.0, 0.0, 1.0]),
            TreeTransformBlockSpec::single(2, 2, 1.0),
        ],
    )
    .unwrap();
    let schedule = transform.parallel_schedule();
    let single = (0..schedule.component_count())
        .find(|&c| schedule.component_group_list(c).is_empty())
        .unwrap();
    assert!(schedule.components[single].units >= 2);
    let mut engine = costed_engine(6, 60);
    let mut dense = Spy::new(&transform);
    dense.estimate = Some(Duration::from_nanos(6000));
    let (marks, workspace) = record(
        &mut engine,
        &mut dense,
        &transform,
        4,
        &[1.0, 2.0, 3.0, 4.0],
    );
    let placement = workspace.lanes.placement();
    let lane = (0..placement.lanes())
        .find(|&l| placement.jobs(l).contains(&single))
        .unwrap();
    assert!(placement.share(lane) >= 2);
    assert_eq!(marks[2].worker, marks[3].worker);
    assert_eq!(marks[2].worker, Some(std::thread::current().id()));
    assert!(dense.budgets().iter().all(|&(_, offered)| offered >= 2));
}

/// What: an overwritten Single that retains a compiled plan Strided does not
/// admit on the stage's four slots (a 24-element copy against 60 ns
/// dispatch) is banded like any other component under forced `Inner`, so
/// more than one worker writes it, each element once.
#[test]
fn an_unadmitted_planned_single_is_banded() {
    let structure = Arc::new(BlockStructure::packed_column_major(2, [vec![4, 6]]).unwrap());
    let transform = TreeTransformStructure::compile_structures(
        &structure,
        &structure,
        &[TreeTransformBlockSpec::single(0, 0, -2.0)],
    )
    .unwrap();
    // Precondition: the role keeps its plan.
    assert_eq!(transform.layouts().retained_plan_count(), 1);
    let source = (0..24).map(f64::from).collect::<Vec<_>>();
    let mut engine =
        costed_engine(4, 60).with_batch_policy(CpuBatchPolicy::new(CpuBatchStrategy::Inner));
    let (marks, _) = record(
        &mut engine,
        &mut Spy::new(&transform),
        &transform,
        24,
        &source,
    );
    assert!(marks.iter().all(|mark| mark.writes == 1));
    let mut workers = marks
        .iter()
        .map(|mark| format!("{:?}", mark.worker))
        .collect::<Vec<_>>();
    workers.sort_unstable();
    workers.dedup();
    assert!(workers.len() > 1, "the planned Single ran whole");
}

/// What: the interleaved SU(2) transform, initialized and owned, gives the
/// same destination at budgets one, two, three and eight: every Single
/// element bit for bit (movement only), every Multi element within the
/// GEMM's tolerance.
#[test]
fn results_do_not_depend_on_the_engine_width() {
    let (structure, transform) = su2_interleaved_fixture();
    let source = (1..=60)
        .map(|value| f64::from(value) * 0.375 - 7.0)
        .collect::<Vec<_>>();
    let initial = (1..=60)
        .map(|value| f64::from(value) - 30.0)
        .collect::<Vec<_>>();
    let run = |budget: usize| {
        let mut engine = crate::host_pool::eager_test_engine(budget);
        let mut accumulated = initial.clone();
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
            1.25,
            -0.5,
        )
        .unwrap();
        let owned = try_tree_transform_structure_overwrite_owned_raw(
            &mut engine,
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            &transform,
            &structure,
            &structure,
            2,
            &source,
            1.25,
            None,
        )
        .unwrap()
        .unwrap();
        (accumulated, owned)
    };
    let single = super::su2_interleaved::SINGLES
        .iter()
        .flat_map(|&(offset, _)| super::su2_interleaved::addresses(offset))
        .collect::<Vec<_>>();
    let (accumulated, owned) = run(1);
    for budget in [2, 3, 8] {
        let (other_accumulated, other_owned) = run(budget);
        for (expected, actual) in [(&accumulated, &other_accumulated), (&owned, &other_owned)] {
            for at in 0..60 {
                if single.contains(&at) {
                    assert_eq!(
                        expected[at].to_bits(),
                        actual[at].to_bits(),
                        "budget {budget}"
                    );
                } else {
                    let scale = expected[at].abs().max(1.0);
                    assert!(
                        (expected[at] - actual[at]).abs() <= 8.0 * f64::EPSILON * scale,
                        "budget {budget}, element {at}"
                    );
                }
            }
        }
    }
}

/// What: two failing GEMMs in different components: with lanes every
/// component runs and the lowest-indexed failing component's error is
/// reported, at every budget; one worker (a serial engine, or forced
/// `Sequential` on four threads) replays in plan order, reports the first
/// failure there and never attempts the later group; the owned output
/// publishes nothing.
#[test]
fn the_lowest_failing_component_is_reported() {
    let (dst, src, transform) = super::threaded::joint_fixture();
    let schedule = transform.parallel_schedule();
    let low = schedule.component_group_list(0)[0] as usize;
    let high = *schedule.component_group_list(2).first().unwrap() as usize;
    // The two orders disagree, so the test tells them apart.
    assert!(high < low);
    let source = (1..=16).map(f64::from).collect::<Vec<_>>();
    for (budget, strategy) in [
        (1, CpuBatchStrategy::Outer),
        (2, CpuBatchStrategy::Outer),
        (3, CpuBatchStrategy::Outer),
        (4, CpuBatchStrategy::Sequential),
    ] {
        let one_worker = budget == 1 || matches!(strategy, CpuBatchStrategy::Sequential);
        let mut engine = if budget == 1 {
            CpuEngine::serial()
        } else {
            CpuEngine::new(Pool::builder().threads(budget).build().unwrap())
                .with_batch_policy(CpuBatchPolicy::new(strategy))
        };
        let mut dense = Spy::new(&transform);
        dense.fail = vec![low, high];
        // A real GEMM estimate, so the forced lanes place the components by
        // it rather than all on the first lane.
        dense.estimate = Some(Duration::from_micros(1));
        let mut workspace = TreeTransformWorkspace::default();
        let error = tree_transform_structure_overwrite_with_structural_recoupling_raw(
            &mut engine,
            &mut StridedHostKernelAdapter::default(),
            &mut dense,
            &mut workspace,
            &transform,
            &dst,
            &src,
            &mut [f64::NAN; 14],
            &source,
            1.0,
            &[],
        )
        .unwrap_err();
        let expected = if one_worker { high } else { low };
        assert_eq!(
            group_of(error),
            format!("group {expected}"),
            "budget {budget}"
        );
        let attempted = dense
            .budgets()
            .iter()
            .map(|&(job, _)| job)
            .collect::<Vec<_>>();
        if one_worker {
            assert!(!attempted.contains(&low), "budget {budget}");
        }
        if budget == 3 {
            let placement = workspace.lanes.placement();
            let lane_of =
                |c: usize| (0..placement.lanes()).find(|&l| placement.jobs(l).contains(&c));
            assert_ne!(lane_of(0), lane_of(2));
            // Both failing GEMMs were attempted.
            assert!(attempted.contains(&low) && attempted.contains(&high));
        }
    }

    let (structure, su2) = su2_interleaved_fixture();
    let mut engine = CpuEngine::new(Pool::builder().threads(3).build().unwrap())
        .with_batch_policy(CpuBatchPolicy::new(CpuBatchStrategy::Outer));
    let mut dense = Spy::new(&su2);
    dense.fail = vec![0, 1];
    let owned = try_tree_transform_structure_overwrite_owned_raw(
        &mut engine,
        &mut dense,
        &mut TreeTransformWorkspace::default(),
        &su2,
        &structure,
        &structure,
        2,
        &(1..=60).map(f64::from).collect::<Vec<_>>(),
        1.0,
        None,
    );
    assert!(matches!(
        owned,
        Err(OperationError::Dense(DenseError::Unsupported {
            op: "spy",
            ..
        }))
    ));
}

#[path = "../../../tests/support/warm_tree_lanes.rs"]
mod warm_tree_lanes;

/// What: the small fixture of the process-wide warm allocation test
/// (`tests/warm_tree_lanes.rs`), eight Multi groups of two `[8, 3]` blocks
/// and a Single, runs under `Outer` on one lane per component up to the
/// engine's width.
#[test]
fn the_warm_allocation_fixture_runs_on_four_lanes() {
    let (_, transform, source) = warm_tree_lanes::fixture(8, 8);
    let threads = warm_tree_lanes::THREADS;
    let mut engine = CpuEngine::new(Pool::builder().threads(threads).build().unwrap())
        .with_batch_policy(CpuBatchPolicy::new(CpuBatchStrategy::Outer));
    let (_, workspace) = record(
        &mut engine,
        &mut Spy::new(&transform),
        &transform,
        source.len(),
        &source,
    );
    assert_eq!(
        workspace.lanes.placement().lanes(),
        transform.parallel_schedule().component_count().min(threads)
    );
}

/// One component of two interleaved `[512, 2]` Singles, one atom of two
/// band units whose Direct estimate pays for a section of six parts and not
/// of two at dispatch costs of 1 us at two slots and 1 ns from four; replays
/// it through the recording writer on six slots under `strategy` and returns
/// the marks and the first lane's share.
fn two_band_component(strategy: CpuBatchStrategy) -> (Vec<Mark>, usize) {
    let rows = 512;
    let blocks = [
        (0, [rows, 2], [1, 2 * rows]),
        (rows, [rows, 2], [1, 2 * rows]),
    ];
    let dst = Arc::new(
        BlockStructure::from_blocks_with_rank(
            2,
            blocks
                .iter()
                .enumerate()
                .map(|(key, &(offset, dims, strides))| {
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
    );
    let src =
        Arc::new(BlockStructure::packed_column_major(2, [vec![rows, 2], vec![rows, 2]]).unwrap());
    let transform = TreeTransformStructure::compile_structures(
        &dst,
        &src,
        &[
            TreeTransformBlockSpec::single(0, 0, 1.0),
            TreeTransformBlockSpec::single(1, 1, 1.0),
        ],
    )
    .unwrap();
    let schedule = transform.parallel_schedule();
    assert_eq!(
        (schedule.component_count(), schedule.components[0].units),
        (1, 2)
    );
    let estimate = KernelClass::Copy.estimate(4 * rows);
    assert!(
        estimate >= Duration::from_nanos(3) && estimate < Duration::from_nanos(3000),
        "fixture premise: {estimate:?} pays at width six, not two"
    );
    let costs = WidthCosts::new([1000, 1, 1].map(Duration::from_nanos));
    let mut engine = CpuEngine::new(
        Pool::builder()
            .threads(6)
            .dispatch_costs(DispatchCosts::new(costs, costs, costs))
            .build()
            .unwrap(),
    )
    .with_batch_policy(CpuBatchPolicy::new(strategy));
    let len = dst.required_len().unwrap();
    let source = (0..len).map(|v| v as f64).collect::<Vec<_>>();
    let mut dense = Spy::new(&transform);
    let (marks, workspace) = record(&mut engine, &mut dense, &transform, len, &source);
    (marks, workspace.lanes.placement().share(0))
}

/// What: under `Auto`, the two-band component takes width six, and its
/// Direct stage, capped to two bands, a section its estimate does not pay
/// for, runs inline: the calling thread writes every element once.
#[test]
fn a_band_fork_that_does_not_pay_at_its_part_count_runs_inline() {
    let (marks, share) = two_band_component(CpuBatchStrategy::Auto);
    assert_eq!(share, 6);
    let caller = Some(std::thread::current().id());
    assert!(marks
        .iter()
        .all(|mark| mark.writes == 1 && mark.worker == caller));
}

/// What: a forced `Inner` keeps the two-band component's fork whatever it
/// costs: its two bands are written by two workers, each element once.
#[test]
fn a_forced_policy_keeps_its_band_fork() {
    let (marks, _) = two_band_component(CpuBatchStrategy::Inner);
    assert!(marks.iter().all(|mark| mark.writes == 1));
    let workers: std::collections::HashSet<_> = marks.iter().map(|mark| mark.worker).collect();
    assert_eq!(workers.len(), 2);
}

/// What: a fresh workspace's lane replay allocates the lanes' packed source
/// and destination buffers on the caller, at most one of each per lane and
/// more than lane 0's own, on two and on four lanes; the same workspace's
/// next replay allocates neither.
#[test]
fn lane_packed_scratch_is_allocated_on_the_caller_once_per_lane() {
    let rows = 1024;
    let (_, transform, source) = warm_tree_lanes::fixture(8, rows);
    let task = transform.task_view().unwrap();
    for threads in [2, 4] {
        let mut engine = CpuEngine::new(Pool::builder().threads(threads).build().unwrap())
            .with_batch_policy(CpuBatchPolicy::new(CpuBatchStrategy::Outer));
        let mut dense = Spy::new(&transform);
        let fresh_workspace = || {
            let mut workspace = TreeTransformWorkspace::default();
            ensure_recoupling_coefficients(&mut workspace, task, transform.identity_marker())
                .unwrap();
            workspace
        };
        let mut marks = vec![Mark::default(); source.len()];
        let recorder = Recorder {
            parts: ReplayParts::new(task),
        };
        // Every lane's packed buffers hold at least one `rows`-long column.
        let mut replay = |workspace: &mut TreeTransformWorkspace<f64>| {
            crate::counting_alloc::measure_matching(
                rows * core::mem::size_of::<f64>()..=usize::MAX,
                || {
                    super::super::lanes::lane_replay(
                        &mut engine,
                        &mut dense,
                        workspace,
                        task,
                        transform.parallel_schedule(),
                        &mut marks,
                        &source,
                        &recorder,
                        None,
                        1,
                    )
                    .unwrap()
                },
            )
        };
        // Why a throwaway replay first: lane 0 runs on the caller, and the
        // GEMM provider's first use on a thread allocates its own large
        // state there (faer's x86 kernels: per-thread scratch and a cache
        // probe), which is not lane scratch.
        replay(&mut fresh_workspace());
        let mut workspace = fresh_workspace();
        let (ran, fresh) = replay(&mut workspace);
        assert!(ran);
        assert_eq!(workspace.lanes.placement().lanes(), threads);
        let per_lane_on_caller = 3..=2 * threads as u64;
        assert!(
            per_lane_on_caller.contains(&fresh.matched_calls),
            "{threads} lanes: {}",
            fresh.matched_calls
        );
        let (_, warm) = replay(&mut workspace);
        assert_eq!(warm.matched_calls, 0, "{threads} lanes, warm");
    }
}
