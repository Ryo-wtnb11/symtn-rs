//! The barrier form's contracts, through a custom executor without lane
//! GEMMs: its values equal the built-in executor's replay, every destination
//! element has one writer, and a failure is job-local with the lowest-indexed
//! component reported at every lane count. A custom executor honors no
//! forced batch policy, so its public replays refuse one; the forced lanes
//! below are the barrier route's own.

use super::recording::*;
use super::*;
use symtn_dense::{CpuEngine, Pool};
use tenferro_cpu::{CpuBatchPolicy, CpuBatchStrategy};

use super::super::writers::ReplayParts;

/// `alpha T(src) + beta dst` on `budget` workers under `strategy`.
#[allow(clippy::too_many_arguments)]
fn update<E: DenseExecutor>(
    dense: &mut E,
    budget: usize,
    strategy: CpuBatchStrategy,
    transform: &TreeTransformStructure<f64>,
    dst: &Arc<BlockStructure>,
    src: &Arc<BlockStructure>,
    initial: &[f64],
    source: &[f64],
) -> Vec<f64> {
    let mut out = initial.to_vec();
    tree_transform_structure_with_structural_recoupling_raw(
        &mut engine(budget, strategy),
        &mut StridedHostKernelAdapter::default(),
        dense,
        &mut TreeTransformWorkspace::default(),
        transform,
        dst,
        src,
        &mut out,
        source,
        1.25,
        -0.5,
    )
    .unwrap();
    out
}

/// What: on the joint fixture, the SU(2) fixture and the interleaved
/// bandable family, at budgets 1, 2, 3 and 8 under `Auto`, the barrier
/// form writes the built-in executor's replay: its movement-only
/// elements (Singles and inactive blocks) bit for bit, the recoupled ones
/// within GEMM rounding; the SU(2) update and owned overwrite also match
/// the hand oracle.
#[test]
fn barrier_values_equal_the_built_in_replay() {
    let (joint_dst, joint_src, joint) = super::threaded::joint_fixture();
    let (su2_structure, su2) = su2_interleaved_fixture();
    let (bandable_blocks, bandable_dst, bandable_src, bandable) = interleaved_bandable(false);
    let su2_moved = super::su2_interleaved::SINGLES
        .iter()
        .flat_map(|&(offset, _)| super::su2_interleaved::addresses(offset))
        .collect::<Vec<_>>();
    let (offset, dims, strides) = bandable_blocks[0];
    let cases = [
        (&joint, &joint_dst, &joint_src, vec![0, 3, 2, 5, 6, 7]),
        (&su2, &su2_structure, &su2_structure, su2_moved),
        (
            &bandable,
            &bandable_dst,
            &bandable_src,
            addresses(offset, &dims, &strides),
        ),
    ];
    let close = |expected: f64, actual: f64| {
        (expected - actual).abs() <= 8.0 * f64::EPSILON * expected.abs().max(1.0)
    };
    for (transform, dst, src, moved) in cases {
        let len = dst.required_len().unwrap();
        let source = (0..src.required_len().unwrap())
            .map(|v| v as f64 * 0.375 - 7.0)
            .collect::<Vec<_>>();
        let initial = (0..len).map(|v| v as f64 - 30.0).collect::<Vec<_>>();
        for budget in [1, 2, 3, 8] {
            {
                let strategy = CpuBatchStrategy::Auto;
                let what = format!("len {len}, budget {budget}, {strategy:?}");
                let args = (transform, dst, src, &initial[..], &source[..]);
                let built_in = update(
                    &mut DefaultDenseExecutor::new(),
                    budget,
                    strategy,
                    args.0,
                    args.1,
                    args.2,
                    args.3,
                    args.4,
                );
                let barrier = update(
                    &mut Forwarding::default(),
                    budget,
                    strategy,
                    args.0,
                    args.1,
                    args.2,
                    args.3,
                    args.4,
                );
                for at in 0..len {
                    if moved.contains(&at) {
                        assert_eq!(built_in[at].to_bits(), barrier[at].to_bits(), "{what}");
                    } else {
                        assert!(close(built_in[at], barrier[at]), "{what}, element {at}");
                    }
                }
                if std::ptr::eq(transform, &su2) {
                    let oracle = su2_oracle(&source, 1.25);
                    for at in 0..len {
                        assert!(close(oracle[at] - 0.5 * initial[at], barrier[at]), "{what}");
                    }
                    let owned = try_tree_transform_structure_overwrite_owned_raw(
                        &mut engine(budget, strategy),
                        &mut Forwarding::default(),
                        &mut TreeTransformWorkspace::default(),
                        &su2,
                        &su2_structure,
                        &su2_structure,
                        2,
                        &source,
                        1.25,
                        None,
                    )
                    .unwrap()
                    .unwrap();
                    for at in 0..len {
                        assert!(close(oracle[at], owned[at]), "{what}");
                    }
                }
            }
        }
    }
}

/// What: in the barrier form on a four-worker engine, every destination
/// element is written exactly once, by one worker; a band of the
/// interleaved atom (the interior one included) is never split, its bands
/// go to more than one worker, and the `[3, 2]` / `[2, 3]` block is
/// written whole by one worker.
#[test]
fn every_destination_element_has_one_writer_in_the_barrier_form() {
    let (blocks, dst, _, transform) = interleaved_bandable(true);
    let len = dst.required_len().unwrap();
    let source = (0..len).map(|v| v as f64).collect::<Vec<_>>();
    let mut engine = crate::host_pool::eager_test_engine(4);
    let task = transform.task_view().unwrap();
    let mut workspace = TreeTransformWorkspace::default();
    ensure_recoupling_coefficients(&mut workspace, task, transform.identity_marker()).unwrap();
    let mut marks = vec![Mark::default(); len];
    let recorder = Recorder {
        parts: ReplayParts::new(task),
    };
    let ran = super::super::lanes::barrier_replay(
        &mut engine,
        &mut Forwarding::default(),
        &mut workspace,
        task,
        transform.parallel_schedule(),
        &mut marks,
        &source,
        &recorder,
        None,
        1,
    )
    .unwrap();
    assert!(ran, "placement left the lane runner");

    assert_one_writer_per_element(&blocks, &marks);
}

/// What: component 0's second GEMM (`e0`) and component 1's only GEMM
/// (`e1`) fail. On two, three and four forced `Outer` one-slot lanes of the
/// barrier route, `e0` is reported and both GEMMs are attempted, `e0` after
/// `e1` failed; one worker (a serial engine, or forced `Sequential` on four
/// workers, through the public replay) replays in plan order, reports the
/// first failure there and attempts nothing after it; no failed GEMM is
/// submitted twice.
#[test]
fn a_barrier_failure_ends_only_its_component_and_the_lowest_wins() {
    let (dst, src, transform) = two_lane_fixture();
    let schedule = transform.parallel_schedule();
    assert_eq!(schedule.component_count(), 2);
    assert_eq!(schedule.component_group_list(0).len(), 2);
    let source = (1..=192).map(f64::from).collect::<Vec<_>>();
    // Each plan job's label: its Multi block's spec position plus one.
    let label: Vec<usize> = transform
        .recoupling_plan()
        .entries()
        .map(|(block, _)| block + 1)
        .collect();
    let e0 = label[schedule.component_group_list(0)[1] as usize];
    let e1 = label[schedule.component_group_list(1)[0] as usize];
    for (budget, strategy) in [
        (1, CpuBatchStrategy::Auto),
        (2, CpuBatchStrategy::Outer),
        (3, CpuBatchStrategy::Outer),
        (4, CpuBatchStrategy::Outer),
        (4, CpuBatchStrategy::Sequential),
    ] {
        let one_worker = budget == 1 || matches!(strategy, CpuBatchStrategy::Sequential);
        let mut engine = if budget == 1 {
            CpuEngine::serial()
        } else {
            CpuEngine::new(Pool::builder().threads(budget).build().unwrap())
                .with_batch_policy(CpuBatchPolicy::new(strategy))
        };
        let mut dense = Forwarding {
            fail: vec![e0, e1],
            ..Forwarding::default()
        };
        let what = format!("budget {budget}, {strategy:?}");
        if one_worker {
            let error = tree_transform_structure_overwrite_with_structural_recoupling_raw(
                &mut engine,
                &mut StridedHostKernelAdapter::default(),
                &mut dense,
                &mut TreeTransformWorkspace::default(),
                &transform,
                &dst,
                &src,
                &mut [f64::NAN; 96],
                &source,
                1.0,
                &[],
            )
            .unwrap_err();
            // Plan order through the first failing job, nothing after it.
            let first = label.iter().position(|l| [e0, e1].contains(l)).unwrap();
            assert_eq!(dense.attempts(), label[..=first], "{what}");
            assert_eq!(group_of(error), format!("group {}", label[first]), "{what}");
        } else {
            let task = transform.task_view().unwrap();
            let mut workspace = TreeTransformWorkspace::default();
            ensure_recoupling_coefficients(&mut workspace, task, transform.identity_marker())
                .unwrap();
            let recorder = Recorder {
                parts: ReplayParts::new(task),
            };
            let error = super::super::lanes::barrier_replay(
                &mut engine,
                &mut dense,
                &mut workspace,
                task,
                schedule,
                &mut [Mark::default(); 96],
                &source,
                &recorder,
                None,
                1,
            )
            .unwrap_err();
            assert_eq!(group_of(error), format!("group {e0}"), "{what}");
            for failed in [e0, e1] {
                let tries = dense.attempts().iter().filter(|&&l| l == failed).count();
                assert_eq!(tries, 1, "{what}, group {failed} retried");
            }
            let attempts = dense.attempts();
            let at = |label| attempts.iter().position(|&l| l == label);
            assert!(at(e1).unwrap() < at(e0).unwrap(), "{what}");
        }
    }
}

/// A coefficient whose action on the source value `40` panics.
#[derive(Clone, Copy)]
struct Panicking(f64);

impl RecouplingCoefficientAction<Panicking> for f64 {
    fn scale_by_coefficient(self, coefficient: Panicking) -> Self {
        assert!(self != 40.0, "a failing movement");
        self * coefficient.0
    }

    fn coefficient_as_data(coefficient: Panicking) -> Self {
        coefficient.0
    }
}

/// What: a movement panic inside a barrier round of the barrier route (the
/// second of two Singles, each its own component on one of two forced
/// `Outer` lanes) unwinds out of the replay after the fork joined, leaves
/// the engine broken, and the engine refuses its next entry.
#[test]
fn a_barrier_round_panic_breaks_the_engine() {
    let structure = Arc::new(BlockStructure::packed_column_major(1, vec![vec![32]; 2]).unwrap());
    let transform = TreeTransformStructure::compile_structures(
        &structure,
        &structure,
        &[
            TreeTransformBlockSpec::single(0, 0, Panicking(2.0)),
            TreeTransformBlockSpec::single(1, 1, Panicking(3.0)),
        ],
    )
    .unwrap();
    let source = (1..=64).map(f64::from).collect::<Vec<_>>();
    let mut engine = CpuEngine::new(Pool::builder().threads(2).build().unwrap())
        .with_batch_policy(CpuBatchPolicy::new(CpuBatchStrategy::Outer));
    let mut workspace = TreeTransformWorkspace::default();
    let task = transform.task_view().unwrap();
    let writer = super::super::writers::InitializedWriter {
        prototype: StridedHostKernelAdapter::default(),
        task: ReplayParts::new(task),
        alpha: DestinationAlpha::new(1.0, &[]).unwrap(),
        mode: DestinationMode::Overwrite,
    };
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        super::super::lanes::barrier_replay(
            &mut engine,
            &mut Forwarding::default(),
            &mut workspace,
            task,
            transform.parallel_schedule(),
            &mut [0.0; 64],
            &source,
            &writer,
            None,
            1,
        )
    }));
    assert!(unwind.is_err());
    assert_eq!(workspace.lanes.placement().lanes(), 2);
    assert!(engine.is_broken());
    assert!(engine.with_exec(|exec| exec.budget()).is_err());
}

/// What: a forced `Inner`, `Outer` or `Split` policy, which a custom
/// executor cannot honor, is refused with a typed `Unsupported` before any
/// write or executor call: on the initialized replay, the owned overwrite
/// and a two-member batch; on a serial engine with the policy installed and
/// on four workers; for a call `Auto` would replay on one worker (one tiny
/// Multi group), a provider-free one (Singles only) and one with several
/// Multi groups. The initialized destination keeps its bits, the owned
/// overwrite publishes nothing and the executor sees no GEMM.
#[test]
fn a_forced_policy_a_custom_executor_cannot_honor_is_refused_before_any_write() {
    let tiny = identity_multi_fixture();
    let singles = {
        let blocks = Arc::new(BlockStructure::packed_column_major(1, vec![vec![32]; 2]).unwrap());
        let transform = TreeTransformStructure::compile_structures(
            &blocks,
            &blocks,
            &[
                TreeTransformBlockSpec::single(0, 0, 2.0),
                TreeTransformBlockSpec::single(1, 1, 3.0),
            ],
        )
        .unwrap();
        (blocks.clone(), blocks, transform)
    };
    let groups = two_lane_fixture();
    let forced = [
        CpuBatchStrategy::Inner,
        CpuBatchStrategy::Outer,
        CpuBatchStrategy::Split {
            lanes: core::num::NonZeroUsize::new(2).unwrap(),
        },
    ];
    let unsupported = |error: OperationError| {
        matches!(
            error,
            OperationError::Dense(DenseError::Unsupported {
                op: "tree_transform_replay",
                ..
            })
        )
    };
    for (dst, src, transform) in [&tiny, &singles, &groups] {
        let (dst_len, src_len) = (dst.required_len().unwrap(), src.required_len().unwrap());
        for strategy in forced {
            for threads in [1, 4] {
                let what = format!("dst {dst_len}, {strategy:?}, {threads} threads");
                let mut engine = if threads == 1 {
                    CpuEngine::serial()
                } else {
                    CpuEngine::new(Pool::builder().threads(threads).build().unwrap())
                }
                .with_batch_policy(CpuBatchPolicy::new(strategy));
                let mut dense = Forwarding::default();
                let source = (0..2 * src_len).map(|v| v as f64 + 1.0).collect::<Vec<_>>();
                let initial = (0..2 * dst_len).map(|v| v as f64 - 7.5).collect::<Vec<_>>();

                let mut out = initial[..dst_len].to_vec();
                let error = tree_transform_structure_with_structural_recoupling_raw(
                    &mut engine,
                    &mut StridedHostKernelAdapter::default(),
                    &mut dense,
                    &mut TreeTransformWorkspace::default(),
                    transform,
                    dst,
                    src,
                    &mut out,
                    &source[..src_len],
                    1.25,
                    -0.5,
                )
                .unwrap_err();
                assert!(unsupported(error), "{what}, initialized");
                assert_eq!(out, initial[..dst_len], "{what}, initialized");

                let owned = try_tree_transform_structure_overwrite_owned_raw(
                    &mut engine,
                    &mut dense,
                    &mut TreeTransformWorkspace::default(),
                    transform,
                    dst,
                    src,
                    dst.rank(),
                    &source[..src_len],
                    1.25,
                    None,
                );
                assert!(unsupported(owned.unwrap_err()), "{what}, owned");

                let mut out = initial.clone();
                let error = tree_transform_members_overwrite_raw(
                    &mut engine,
                    &mut StridedHostKernelAdapter::default(),
                    &mut dense,
                    &mut TreeTransformWorkspace::default(),
                    transform,
                    dst,
                    src,
                    &mut out,
                    &source,
                    2,
                    &[],
                )
                .unwrap_err();
                assert!(unsupported(error), "{what}, members");
                assert_eq!(out, initial, "{what}, members");
                assert!(dense.attempts().is_empty(), "{what}");
            }
        }
    }
}
