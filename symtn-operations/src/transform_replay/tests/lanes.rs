//! The lane replay's contracts: one writer per destination element, each
//! stage at its selected width, results independent of the engine width,
//! and job-local errors with the lowest-indexed component reported.

use super::*;
use crate::kernel_adapter::movement::{Store, Write};
use crate::transform_structure::TreeTransformDirectItem;
use std::sync::Mutex;
use std::thread::ThreadId;
use std::time::Duration;
use strided_kernel::execution::KernelClass;
use symtn_dense::lanes::{Estimate, Exec as LaneExec, LaneGemmIo, LaneGemmShape, LaneGemms};
use symtn_dense::{CpuEngine, DenseDType, Pool};
use tenferro_cpu::threadteam::{DispatchCosts, WidthCosts};
use tenferro_cpu::{CpuBatchPolicy, CpuBatchStrategy};

use super::super::writers::{crop_role, inactive_role, DestinationWriter, ReplayParts};

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

/// One destination element as the recording writer leaves it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Mark {
    writes: u32,
    worker: Option<ThreadId>,
}

/// A movement store that records each write's thread instead of a value.
enum Record {}

impl Store<f64> for Record {
    type Slot = Mark;

    fn put(slot: &mut Mark, _: f64) {
        slot.writes += 1;
        slot.worker = Some(std::thread::current().id());
    }
}

/// A writer that addresses exactly the elements the real writers address,
/// whole or cropped to a band, and records who wrote each.
struct Recorder<'a> {
    parts: ReplayParts<'a, f64>,
    src: &'a [f64],
}

impl DestinationWriter<Mark, f64> for Recorder<'_> {
    type Kernels = ();

    fn kernels(&self) {}

    fn overwrites(&self) -> bool {
        true
    }

    fn direct(
        &self,
        _: &mut (),
        _: &mut Vec<isize>,
        item: TreeTransformDirectItem,
        dst: &mut [Mark],
        dst_start: isize,
        band: Option<super::super::writers::Band>,
        _: crate::host_pool::Exec<'_>,
        index: &mut [usize],
    ) -> Result<(), OperationError> {
        let layouts = self.parts.layouts;
        let mut cropped = [0; crate::kernel_adapter::FUSED_LAYOUT_INLINE_RANK];
        match item {
            TreeTransformDirectItem::Inactive { layout, .. } => {
                let (role, shift) = inactive_role(layouts, layout, band, &mut cropped)?;
                span::<Record, _, f64>(
                    dst,
                    &[0.0],
                    role,
                    layouts.entry(layout).offset - dst_start + shift,
                    0,
                    &mut index[..role.dims().len()],
                    &Write(|value: f64| value),
                );
            }
            TreeTransformDirectItem::Single { block, .. } => {
                let (dst_layout, src_layout, _) = scheduled_single(self.parts.blocks, block)?;
                let role = layouts.role(dst_layout)?;
                let (role, dst_shift, src_shift) = match band {
                    None => (role, 0, 0),
                    Some(band) => crop_role(role, band, &mut cropped)?,
                };
                span::<Record, _, f64>(
                    dst,
                    self.src,
                    role,
                    layouts.entry(dst_layout).offset - dst_start + dst_shift,
                    layouts.entry(src_layout).offset + src_shift,
                    &mut index[..role.dims().len()],
                    &Write(|value: f64| value),
                );
            }
        }
        Ok(())
    }

    fn scatter(
        &self,
        _: &mut (),
        _: &mut Vec<isize>,
        dst_layout: usize,
        packed: &[f64],
        packed_offset: usize,
        dst: &mut [Mark],
        dst_start: isize,
        band: Option<super::super::writers::Band>,
        _: crate::host_pool::Exec<'_>,
        index: &mut [usize],
    ) -> Result<(), OperationError> {
        let layouts = self.parts.layouts;
        let mut cropped = [0; crate::kernel_adapter::FUSED_LAYOUT_INLINE_RANK];
        let role = layouts.role(dst_layout)?;
        let (role, dst_shift, src_shift) = match band {
            None => (role, 0, 0),
            Some(band) => crop_role(role, band, &mut cropped)?,
        };
        span::<Record, _, f64>(
            dst,
            packed,
            role,
            layouts.entry(dst_layout).offset - dst_start + dst_shift,
            offset_to_isize(packed_offset)? + src_shift,
            &mut index[..role.dims().len()],
            &Write(|value: f64| value),
        );
        Ok(())
    }
}

/// The error a [`Spy`] injects for recoupling group `group`.
fn injected(group: usize) -> DenseError {
    DenseError::Unsupported {
        op: "spy",
        message: format!("group {group}"),
    }
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
    let task = transform.task_view().unwrap();
    let mut workspace = TreeTransformWorkspace::default();
    ensure_recoupling_coefficients(&mut workspace, task, transform.identity_marker()).unwrap();
    let mut marks = vec![Mark::default(); len];
    let recorder = Recorder {
        parts: ReplayParts::new(task),
        src: source,
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
    )
    .unwrap();
    assert!(ran, "placement left the lane runner");
    (marks, workspace)
}

/// Destination element addresses of a block `(offset, dims, strides)`.
fn addresses(offset: usize, dims: &[usize], strides: &[usize]) -> Vec<usize> {
    let mut out = vec![offset];
    for (&dim, &stride) in dims.iter().zip(strides) {
        out = out
            .iter()
            .flat_map(|&base| (0..dim).map(move |i| base + i * stride))
            .collect();
    }
    out
}

/// Columns of the interleaved bandable blocks.
const COLUMNS: usize = 32;

/// Two `[2, COLUMNS]` blocks of row stride 4 interleave in one atom
/// bandable by its columns, a Single and a Multi group's first column; the
/// group's second column, `[2, COLUMNS]` of row stride 2, fuses to one
/// contiguous atom; with `injective`, a Single over the `[3, 2]` / `[2, 3]`
/// layout, which no band certificate can split, forms a second component.
/// Returns the destination blocks `(offset, dims, strides)`, the structures
/// and the transform.
#[allow(clippy::type_complexity)]
fn interleaved_bandable(
    injective: bool,
) -> (
    Vec<(usize, [usize; 2], [usize; 2])>,
    Arc<BlockStructure>,
    Arc<BlockStructure>,
    TreeTransformStructure<f64>,
) {
    let mut blocks = vec![
        (0, [2, COLUMNS], [1, 4]),
        (2, [2, COLUMNS], [1, 4]),
        (4 * COLUMNS, [2, COLUMNS], [1, 2]),
    ];
    if injective {
        blocks.push((6 * COLUMNS, [3, 2], [2, 3]));
    }
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
    let src = Arc::new(
        BlockStructure::packed_column_major(2, blocks.iter().map(|&(_, dims, _)| dims.to_vec()))
            .unwrap(),
    );
    let mut specs = vec![
        TreeTransformBlockSpec::single(0, 0, -2.0),
        TreeTransformBlockSpec::multi(vec![1, 2], vec![1, 2], vec![1.0, 2.0, 3.0, 4.0]),
    ];
    if injective {
        specs.push(TreeTransformBlockSpec::single(3, 3, 1.0));
    }
    let transform = TreeTransformStructure::compile_structures(&dst, &src, &specs).unwrap();
    (blocks, dst, src, transform)
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

    let mut addressed = vec![false; len];
    for &(offset, dims, strides) in &blocks {
        for at in addresses(offset, &dims, &strides) {
            addressed[at] = true;
        }
    }
    for (at, mark) in marks.iter().enumerate() {
        assert_eq!(mark.writes, u32::from(addressed[at]), "element {at}");
    }
    // Atom `[0, 4 COLUMNS)`: band `k` is elements `4k..4k + 4`.
    let mut atom_workers = Vec::new();
    for band in 0..COLUMNS {
        let workers = (4 * band..4 * band + 4)
            .map(|at| marks[at].worker)
            .collect::<Vec<_>>();
        assert!(workers.iter().all(|w| *w == workers[0]), "band {band}");
        atom_workers.push(workers[0]);
    }
    atom_workers.sort_unstable_by_key(|w| format!("{w:?}"));
    atom_workers.dedup();
    assert!(atom_workers.len() > 1, "the bandable atom is not banded");
    let injective = addresses(6 * COLUMNS, &[3, 2], &[2, 3])
        .into_iter()
        .map(|at| marks[at].worker)
        .collect::<Vec<_>>();
    assert!(injective.iter().all(|w| w.is_some() && *w == injective[0]));
}

/// What: a lone bandable component whose movement and GEMM estimates pay
/// for no parallel section runs every stage at one slot, so one worker,
/// the calling thread, writes the whole destination at budgets six and
/// eight.
#[test]
fn cheap_stages_of_one_component_run_on_one_slot() {
    let (blocks, dst, _, transform) = interleaved_bandable(false);
    assert_eq!(transform.parallel_schedule().component_count(), 1);
    let len = dst.required_len().unwrap();
    let source = (0..len).map(|v| v as f64).collect::<Vec<_>>();
    for budget in [6, 8] {
        let mut engine = costed_engine(budget, 60);
        let mut dense = Spy::new(&transform);
        let (marks, _) = record(&mut engine, &mut dense, &transform, len, &source);
        let caller = Some(std::thread::current().id());
        for &(offset, dims, strides) in &blocks {
            for at in addresses(offset, &dims, &strides) {
                assert_eq!(marks[at].worker, caller, "budget {budget}, element {at}");
            }
        }
        assert!(dense.budgets().iter().all(|&(_, offered)| offered == 1));
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

/// The group message of an injected error.
fn group_of(error: OperationError) -> String {
    match error {
        OperationError::Dense(DenseError::Unsupported { op: "spy", message }) => message,
        other => panic!("not an injected error: {other:?}"),
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

/// What: the fixture of the process-wide warm allocation test
/// (`tests/warm_tree_lanes.rs`), eight Multi groups of two `[8, 3]` blocks
/// and a Single, runs on four lanes under `Outer` on a four-worker engine.
#[test]
fn the_warm_allocation_fixture_runs_on_four_lanes() {
    let structure = Arc::new(BlockStructure::packed_column_major(2, vec![vec![8, 3]; 17]).unwrap());
    let mut specs = (0..8)
        .map(|g| {
            TreeTransformBlockSpec::multi(
                vec![2 * g, 2 * g + 1],
                vec![2 * g, 2 * g + 1],
                vec![1.0, 2.0, -0.5, 1.5],
            )
        })
        .collect::<Vec<_>>();
    specs.push(TreeTransformBlockSpec::single(16, 16, -2.0));
    let transform =
        TreeTransformStructure::compile_structures(&structure, &structure, &specs).unwrap();
    let mut engine = CpuEngine::new(Pool::builder().threads(4).build().unwrap())
        .with_batch_policy(CpuBatchPolicy::new(CpuBatchStrategy::Outer));
    let source = vec![1.0; 17 * 24];
    let (_, workspace) = record(
        &mut engine,
        &mut Spy::new(&transform),
        &transform,
        source.len(),
        &source,
    );
    assert_eq!(workspace.lanes.placement().lanes(), 4);
}
