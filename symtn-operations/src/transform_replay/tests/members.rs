//! A member replay's contracts, on the lane runner natively and in the
//! barrier form and on one worker: it writes what the single-member replay
//! writes into each member, every element of every member has one writer,
//! and errors follow job `member * components + component` on lanes and
//! the plan-job-outer, member-inner order on one worker.

use super::recording::*;
use super::*;
use symtn_dense::{CpuEngine, Pool};
use tenferro_cpu::{CpuBatchPolicy, CpuBatchStrategy};

use super::super::writers::ReplayParts;

/// The member overwrite of `members` members on `engine` through `dense`.
#[allow(clippy::too_many_arguments)]
fn overwrite_members<E: DenseExecutor>(
    engine: &mut CpuEngine,
    dense: &mut E,
    transform: &TreeTransformStructure<f64>,
    dst: &Arc<BlockStructure>,
    src: &Arc<BlockStructure>,
    source: &[f64],
    members: usize,
    scales: &[(usize, f64)],
) -> Result<Vec<f64>, OperationError> {
    let mut out = vec![f64::NAN; members * dst.required_len().unwrap()];
    tree_transform_members_overwrite_raw(
        engine,
        &mut StridedHostKernelAdapter::default(),
        dense,
        &mut TreeTransformWorkspace::default(),
        transform,
        dst,
        src,
        &mut out,
        source,
        members,
        scales,
    )?;
    Ok(out)
}

/// What: with two and three members at budgets 1, 2 and 4 under `Auto` and
/// `Outer`, through the built-in executor and through a custom executor
/// without lane GEMMs (the barrier form), a member overwrite (with
/// destination scales on a Single and on a Multi group) writes into each member what the single-member
/// overwrite of that member's source writes: its movement-only elements bit
/// for bit, the recoupled ones within GEMM rounding.
#[test]
fn members_equal_the_single_member_replay_of_each_member() {
    let (joint_dst, joint_src, joint) = super::threaded::joint_fixture();
    let (su2_structure, su2) = su2_interleaved_fixture();
    let (bandable_blocks, bandable_dst, bandable_src, bandable) = interleaved_bandable(false);
    let su2_moved = super::su2_interleaved::SINGLES
        .iter()
        .flat_map(|&(offset, _)| super::su2_interleaved::addresses(offset))
        .collect::<Vec<_>>();
    let (offset, dims, strides) = bandable_blocks[0];
    // Destination scales on a Single and on both destinations of a Multi
    // group (uniform per group, as the contraction compiler emits them).
    let cases = [
        (
            &joint,
            &joint_dst,
            &joint_src,
            vec![0, 3, 2, 5, 6, 7],
            vec![(0, -1.5), (1, 0.5), (12, 0.5)],
        ),
        (
            &su2,
            &su2_structure,
            &su2_structure,
            su2_moved,
            vec![(0, 0.5), (4, -1.5), (20, 0.5)],
        ),
        (
            &bandable,
            &bandable_dst,
            &bandable_src,
            addresses(offset, &dims, &strides),
            vec![(0, -1.5), (2, 0.5), (4 * COLUMNS, 0.5)],
        ),
    ];
    let close = |expected: f64, actual: f64| {
        (expected - actual).abs() <= 8.0 * f64::EPSILON * expected.abs().max(1.0)
    };
    for (transform, dst, src, moved, scales) in cases {
        let (dst_len, src_len) = (dst.required_len().unwrap(), src.required_len().unwrap());
        for members in [2, 3] {
            let source = (0..members * src_len)
                .map(|v| v as f64 * 0.375 - 7.0)
                .collect::<Vec<_>>();
            for budget in [1, 2, 4] {
                for strategy in [CpuBatchStrategy::Auto, CpuBatchStrategy::Outer] {
                    for barrier in [false, true] {
                        let what = format!(
                            "dst {dst_len}, members {members}, budget {budget}, \
                             {strategy:?}, barrier {barrier}"
                        );
                        let mut engine = engine(budget, strategy);
                        let run = |engine: &mut CpuEngine, source: &[f64], members| {
                            if barrier {
                                overwrite_members(
                                    engine,
                                    &mut Forwarding::default(),
                                    transform,
                                    dst,
                                    src,
                                    source,
                                    members,
                                    &scales,
                                )
                            } else {
                                overwrite_members(
                                    engine,
                                    &mut DefaultDenseExecutor::new(),
                                    transform,
                                    dst,
                                    src,
                                    source,
                                    members,
                                    &scales,
                                )
                            }
                            .unwrap()
                        };
                        let batch = run(&mut engine, &source, members);
                        for member in 0..members {
                            let one = run(&mut engine, &source[member * src_len..][..src_len], 1);
                            let got = &batch[member * dst_len..][..dst_len];
                            for at in 0..dst_len {
                                if moved.contains(&at) {
                                    assert_eq!(one[at].to_bits(), got[at].to_bits(), "{what}");
                                } else {
                                    assert!(close(one[at], got[at]), "{what}, element {at}");
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// What: on a four-worker engine, natively (when the provider runs lane
/// GEMMs) and in the barrier form, a two-member replay of the interleaved
/// bandable family writes every addressed element of each member exactly
/// once and nothing else, and more than one worker writes.
#[test]
fn every_element_of_every_member_has_one_writer() {
    let (blocks, dst, _, transform) = interleaved_bandable(true);
    let len = dst.required_len().unwrap();
    let members = 2;
    let source = (0..members * len).map(|v| v as f64).collect::<Vec<_>>();
    let mut addressed = vec![false; len];
    for &(offset, dims, strides) in &blocks {
        for at in addresses(offset, &dims, &strides) {
            addressed[at] = true;
        }
    }
    for barrier in [false, true] {
        let native = DefaultDenseExecutor::new();
        if !barrier && !native.supports_lane_gemms() {
            continue;
        }
        let mut engine = crate::host_pool::eager_test_engine(4);
        let task = transform.task_view().unwrap();
        let mut workspace = TreeTransformWorkspace::default();
        ensure_recoupling_coefficients(&mut workspace, task, transform.identity_marker()).unwrap();
        let mut marks = vec![Mark::default(); members * len];
        let recorder = Recorder {
            parts: ReplayParts::new(task),
        };
        let route = if barrier {
            super::super::lanes::barrier_replay
        } else {
            super::super::lanes::lane_replay
        };
        let ran = route(
            &mut engine,
            &mut Forwarding {
                inner: native,
                lanes: !barrier,
                ..Forwarding::default()
            },
            &mut workspace,
            task,
            transform.parallel_schedule(),
            &mut marks,
            &source,
            &recorder,
            None,
            members,
        );
        assert!(ran.unwrap(), "placement left the lane runner");
        for (at, mark) in marks.iter().enumerate() {
            assert_eq!(
                mark.writes,
                u32::from(addressed[at % len]),
                "barrier {barrier}, element {at}"
            );
        }
        let mut workers = marks.iter().filter_map(|m| m.worker).collect::<Vec<_>>();
        workers.sort_unstable_by_key(|w| format!("{w:?}"));
        workers.dedup();
        assert!(workers.len() > 1, "barrier {barrier}: one worker wrote");
    }
}

/// What: in a two-member replay of the two-lane fixture, member 0's
/// component 1 GEMM (job 1) and member 1's component 0 first GEMM (job 2)
/// fail. On two and four one-slot lanes, natively and in the barrier form,
/// both are attempted once and job 1's error is reported, the lowest job
/// index. One worker (a serial engine, or forced `Sequential` on four
/// workers) submits the GEMMs plan job outer, member inner (each plan job
/// for member 0, then member 1), stops at the first failure in that order
/// (member 1's, which comes first there) and attempts nothing after it.
#[test]
fn member_failures_report_the_lowest_member_major_job() {
    let (dst, src, transform) = two_lane_fixture();
    let schedule = transform.parallel_schedule();
    let label: Vec<usize> = transform
        .recoupling_plan()
        .entries()
        .map(|(block, _)| block + 1)
        .collect();
    let first = label[schedule.component_group_list(1)[0] as usize];
    let second = label[schedule.component_group_list(0)[0] as usize] + 10;
    let src_len = src.required_len().unwrap();
    let source = (0..2 * src_len)
        .map(|at| (at / src_len) as f64 * MEMBER_SPAN + (at % src_len) as f64 + 1.0)
        .collect::<Vec<_>>();
    for (budget, strategy) in [
        (1, CpuBatchStrategy::Outer),
        (2, CpuBatchStrategy::Outer),
        (4, CpuBatchStrategy::Outer),
        (4, CpuBatchStrategy::Sequential),
    ] {
        for barrier in [false, true] {
            let native = DefaultDenseExecutor::new();
            if !barrier && !native.supports_lane_gemms() {
                continue;
            }
            let one_worker = budget == 1 || matches!(strategy, CpuBatchStrategy::Sequential);
            let mut engine = if budget == 1 {
                CpuEngine::serial()
            } else {
                CpuEngine::new(Pool::builder().threads(budget).build().unwrap())
                    .with_batch_policy(CpuBatchPolicy::new(strategy))
            };
            let mut dense = Forwarding {
                inner: native,
                lanes: !barrier,
                fail: vec![first, second],
                ..Forwarding::default()
            };
            let what = format!("budget {budget}, {strategy:?}, barrier {barrier}");
            let error = overwrite_members(
                &mut engine,
                &mut dense,
                &transform,
                &dst,
                &src,
                &source,
                2,
                &[],
            )
            .unwrap_err();
            let attempts = dense.attempts();
            let tried = |l: usize| attempts.iter().filter(|&&a| a == l).count();
            if one_worker {
                let order = label.iter().flat_map(|&l| [l, l + 10]).collect::<Vec<_>>();
                let at = order
                    .iter()
                    .position(|l| [first, second].contains(l))
                    .unwrap();
                assert_eq!(order[at], second, "the fixture tells the orders apart");
                assert_eq!(attempts, order[..=at], "{what}");
                assert_eq!(group_of(error), format!("group {second}"), "{what}");
            } else {
                assert_eq!(group_of(error), format!("group {first}"), "{what}");
                assert_eq!((tried(first), tried(second)), (1, 1), "{what}");
            }
        }
    }
}
