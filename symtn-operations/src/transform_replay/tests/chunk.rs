//! The chunk replay's contracts, through a custom executor on the default
//! batch entry: its values equal the built-in executor's replay, every
//! destination element is written once, its first failing GEMM in plan
//! order stops it, a panicking worker breaks the engine, and a forced batch
//! policy it cannot honor is refused before any write.

use super::recording::*;
use super::*;
use symtn_dense::{CpuEngine, Pool};
use tenferro_cpu::{CpuBatchPolicy, CpuBatchStrategy};

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
/// bandable family, at budgets 1, 2, 3 and 8 under `Auto`, the custom
/// executor's replay writes the built-in executor's: its movement-only
/// elements (Singles and inactive blocks) bit for bit, the recoupled ones
/// within GEMM rounding; the SU(2) update and owned overwrite also match
/// the hand oracle.
#[test]
fn custom_executor_values_equal_the_built_in_replay() {
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

/// What: on four workers, under `Auto` (the eager engine's free dispatch
/// admits every phase on all four), a forced `Sequential` and, where the
/// built-in executor honors it, a forced `Outer`, every destination element
/// of the interleaved fixture is written exactly once and nothing else is.
#[test]
fn every_destination_element_is_written_once() {
    let (blocks, dst, _, transform) = interleaved_bandable(true);
    let len = dst.required_len().unwrap();
    let source = (0..len).map(|v| v as f64).collect::<Vec<_>>();
    let task = transform.task_view().unwrap();
    for strategy in [
        CpuBatchStrategy::Auto,
        CpuBatchStrategy::Outer,
        CpuBatchStrategy::Sequential,
    ] {
        if strategy == CpuBatchStrategy::Outer
            && !DefaultDenseExecutor::new().honors_forced_batch_policy()
        {
            continue;
        }
        let mut engine = engine(4, strategy);
        let mut workspace = TreeTransformWorkspace::default();
        ensure_recoupling_coefficients(&mut workspace, task, transform.identity_marker()).unwrap();
        let mut marks = vec![Mark::default(); len];
        let recorder = Recorder::new(task);
        super::super::replay(
            &mut engine,
            &mut DefaultDenseExecutor::new(),
            &mut workspace,
            task,
            transform.parallel_schedule(),
            &mut marks,
            &source,
            &recorder,
            &mut (),
            None,
        )
        .unwrap();
        assert_written_once(&blocks, &marks);
    }
}

/// What: two of three Multi groups' GEMMs fail. On one worker and on two,
/// three and four (`Auto` on free dispatch: chunks of two, three and three
/// groups), the replay stops at the first failing GEMM in plan order: the
/// executor saw exactly the plan's GEMMs up to it, and its error is
/// returned; no failed GEMM is submitted twice.
#[test]
fn the_first_failing_gemm_in_plan_order_stops_the_replay() {
    let (dst, src, transform) = two_lane_fixture();
    let source = (1..=192).map(f64::from).collect::<Vec<_>>();
    // Each plan job's label: its Multi block's spec position plus one.
    let label: Vec<usize> = transform
        .recoupling_plan()
        .entries()
        .map(|(block, _)| block + 1)
        .collect();
    let failing = [label[1], label[2]];
    for budget in [1, 2, 3, 4] {
        let mut engine = if budget == 1 {
            CpuEngine::serial()
        } else {
            engine(budget, CpuBatchStrategy::Auto)
        };
        let mut dense = Forwarding {
            fail: failing.to_vec(),
            ..Forwarding::default()
        };
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
        assert_eq!(dense.attempts(), label[..=1], "budget {budget}");
        assert_eq!(
            group_of(error),
            format!("group {}", label[1]),
            "budget {budget}"
        );
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

/// What: a movement panic on a split worker (the second of two
/// slice-disjoint Singles, split over two workers of a free-dispatch
/// engine) unwinds out of the replay after the fork joined, leaves the
/// engine broken, and the engine refuses its next entry.
#[test]
fn a_split_worker_panic_breaks_the_engine() {
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
    assert!(transform.parallel_schedule().singles_slice_disjoint);
    let source = (1..=64).map(f64::from).collect::<Vec<_>>();
    let mut engine = engine(2, CpuBatchStrategy::Auto);
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        tree_transform_structure_overwrite_with_structural_recoupling_raw(
            &mut engine,
            &mut StridedHostKernelAdapter::default(),
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            &transform,
            &structure,
            &structure,
            &mut [0.0; 64],
            &source,
            1.0,
            &[],
        )
    }));
    assert!(unwind.is_err());
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
