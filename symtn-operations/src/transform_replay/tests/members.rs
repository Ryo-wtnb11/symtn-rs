//! A member batch's contracts, through the built-in and a custom executor
//! at every engine width: it writes what the single-member replay writes
//! into each member, every element of every member exactly once, and its
//! errors follow the plan-job-outer, member-inner order of its one dense
//! submission.

use super::recording::*;
use super::*;
use crate::kernel_adapter::observed::ObservedKernels;
use symtn_dense::{CpuEngine, Pool};
use tenferro_cpu::{CpuBatchPolicy, CpuBatchStrategy};

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

/// What: with two and three members at budgets 1, 2 and 4 under `Auto`,
/// and under `Outer` where the executor honors forced policies, through the
/// built-in executor and through a custom executor, a member overwrite
/// (with destination scales on a Single and on a Multi group) writes into
/// each member what the single-member overwrite of that member's source
/// writes: its movement-only elements bit for bit, the recoupled ones
/// within GEMM rounding.
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
                        let honors =
                            !barrier && DefaultDenseExecutor::new().honors_forced_batch_policy();
                        if strategy != CpuBatchStrategy::Auto && !honors {
                            continue;
                        }
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

/// What: on a four-worker engine, through the built-in executor and a
/// custom one, a two-member overwrite of the interleaved bandable family
/// writes every addressed element of each member exactly once and nothing
/// else, counted at the kernel adapter.
#[test]
fn every_element_of_every_member_is_written_once() {
    let (blocks, dst, src, transform) = interleaved_bandable(true);
    let len = dst.required_len().unwrap();
    let members = 2;
    let source = (0..members * src.required_len().unwrap())
        .map(|v| v as f64)
        .collect::<Vec<_>>();
    let mut addressed = vec![0u32; len];
    for &(offset, dims, strides) in &blocks {
        for at in addresses(offset, &dims, &strides) {
            addressed[at] += 1;
        }
    }
    for custom in [false, true] {
        let mut engine = crate::host_pool::eager_test_engine(4);
        let mut kernels = ObservedKernels::addressing();
        let mut out = vec![f64::NAN; members * len];
        let ran = if custom {
            let mut dense = Forwarding::default();
            tree_transform_members_overwrite_raw(
                &mut engine,
                &mut kernels,
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
        } else {
            tree_transform_members_overwrite_raw(
                &mut engine,
                &mut kernels,
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &transform,
                &dst,
                &src,
                &mut out,
                &source,
                members,
                &[],
            )
        };
        ran.unwrap();
        let mut writes = vec![0u32; members * len];
        let buffer = out.as_ptr() as usize;
        for &(written, at) in kernels.writes.as_deref().unwrap() {
            if written == buffer {
                writes[usize::try_from(at).unwrap()] += 1;
            }
        }
        for (at, &count) in writes.iter().enumerate() {
            assert_eq!(count, addressed[at % len], "custom {custom}, element {at}");
        }
    }
}

/// What: in a two-member overwrite of the two-lane fixture, member 0's
/// component 1 GEMM and member 1's component 0 first GEMM fail. At every
/// width (a serial engine, `Auto` on two and four workers, forced
/// `Sequential` on four), the batch submits the GEMMs plan job outer,
/// member inner (each plan job for member 0, then member 1), stops at the
/// first failure in that order (member 1's, which comes first there),
/// reports it and attempts nothing after it.
#[test]
fn member_failures_follow_the_plan_job_outer_member_inner_order() {
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
    let order = label.iter().flat_map(|&l| [l, l + 10]).collect::<Vec<_>>();
    let at = order
        .iter()
        .position(|l| [first, second].contains(l))
        .unwrap();
    assert_eq!(order[at], second, "the fixture tells the orders apart");
    for (budget, strategy) in [
        (1, CpuBatchStrategy::Auto),
        (2, CpuBatchStrategy::Auto),
        (4, CpuBatchStrategy::Auto),
        (4, CpuBatchStrategy::Sequential),
    ] {
        let mut engine = if budget == 1 {
            CpuEngine::serial()
        } else {
            CpuEngine::new(Pool::builder().threads(budget).build().unwrap())
                .with_batch_policy(CpuBatchPolicy::new(strategy))
        };
        let mut dense = Forwarding {
            fail: vec![first, second],
            ..Forwarding::default()
        };
        let what = format!("budget {budget}, {strategy:?}");
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
        assert_eq!(dense.attempts(), order[..=at], "{what}");
        assert_eq!(group_of(error), format!("group {second}"), "{what}");
    }
}
