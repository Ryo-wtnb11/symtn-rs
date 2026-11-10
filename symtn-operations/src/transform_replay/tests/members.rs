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
    let (interleaved_layout, interleaved_dst, interleaved_src, interleaved) =
        interleaved_blocks(false);
    let su2_moved = super::su2_interleaved::SINGLES
        .iter()
        .flat_map(|&(offset, _)| super::su2_interleaved::addresses(offset))
        .collect::<Vec<_>>();
    let (offset, dims, strides) = interleaved_layout[0];
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
            &interleaved,
            &interleaved_dst,
            &interleaved_src,
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
                    for custom in [false, true] {
                        let honors =
                            !custom && DefaultDenseExecutor::new().honors_forced_batch_policy();
                        if strategy != CpuBatchStrategy::Auto && !honors {
                            continue;
                        }
                        let what = format!(
                            "dst {dst_len}, members {members}, budget {budget}, \
                             {strategy:?}, custom {custom}"
                        );
                        let mut engine = engine(budget, strategy);
                        let run = |engine: &mut CpuEngine, source: &[f64], members| {
                            if custom {
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
/// custom one, a two-member overwrite of the interleaved family
/// writes every addressed element of each member exactly once and nothing
/// else, counted at the kernel adapter and its clones.
#[test]
fn every_element_of_every_member_is_written_once() {
    let (blocks, dst, src, transform) = interleaved_blocks(true);
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
        let base = out.as_ptr() as usize / core::mem::size_of::<f64>();
        for &at in kernels.writes.as_ref().unwrap().lock().unwrap().iter() {
            if let Some(index) = at.checked_sub(base).filter(|&index| index < writes.len()) {
                writes[index] += 1;
            }
        }
        for (at, &count) in writes.iter().enumerate() {
            assert_eq!(count, addressed[at % len], "custom {custom}, element {at}");
        }
    }
}

/// What: in a two-member overwrite of the two-lane fixture, member 0's GEMM
/// of the lone group and member 1's GEMM of the first interleaved group fail. At every
/// width (a serial engine, `Auto` on two and four workers, forced
/// `Sequential` on four), the batch submits the GEMMs plan job outer,
/// member inner (each plan job for member 0, then member 1), stops at the
/// first failure in that order (member 1's, which comes first there),
/// reports it and attempts nothing after it.
#[test]
fn member_failures_follow_the_plan_job_outer_member_inner_order() {
    let (dst, src, transform) = two_lane_fixture();
    let label: Vec<usize> = transform
        .recoupling_plan()
        .entries()
        .map(|(block, _)| block + 1)
        .collect();
    // Member 0's lone group (block 2) and member 1's first interleaved
    // group (block 0).
    let (first, second) = (3, 1 + 10);
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

/// What: a member batch resolves each call-dependent destination
/// scale once per role for the whole batch, before its member ranges fork:
/// with a scale on every block of the injective interleaved family (two
/// Singles and one two-destination group), batches of 2, 5 and 8 members
/// on four workers under `Auto` and, where the executor honors it, `Outer`
/// look the table up `Singles + scatter columns = 4` times, whatever the
/// members and their split.
#[test]
fn member_scales_are_resolved_once_per_role() {
    let (layout, dst, src, transform) = interleaved_blocks(true);
    let scales = layout
        .iter()
        .enumerate()
        .map(|(block, &(offset, _, _))| (offset, 0.5 + block as f64))
        .collect::<Vec<_>>();
    let roles = 2 + 2;
    let honors = DefaultDenseExecutor::new().honors_forced_batch_policy();
    for strategy in [CpuBatchStrategy::Auto, CpuBatchStrategy::Outer] {
        if strategy != CpuBatchStrategy::Auto && !honors {
            continue;
        }
        for members in [2, 5, 8] {
            let source = (0..members * src.required_len().unwrap())
                .map(|v| v as f64)
                .collect::<Vec<_>>();
            let mut out = vec![f64::NAN; members * dst.required_len().unwrap()];
            let watch = crate::transform_replay::alpha_lookups::watch(scales.as_ptr() as usize);
            tree_transform_members_overwrite_raw(
                &mut engine(4, strategy),
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &transform,
                &dst,
                &src,
                &mut out,
                &source,
                members,
                &scales,
            )
            .unwrap();
            assert_eq!(watch.take(), roles, "{strategy:?}, {members} members");
        }
    }
}

/// What: a batch of no-element members large enough that `j N` overflows
/// `usize` at a range boundary (`N = isize::MAX`) succeeds when its moves
/// split over four member ranges (forced `Outer` on four workers) and
/// leaves the engine usable.
#[test]
#[cfg_attr(
    any(
        feature = "cpu-blas",
        feature = "blas-accelerate",
        feature = "blas-openblas",
        feature = "blas-mkl",
        feature = "provider-inject"
    ),
    ignore = "only a forced policy splits empty members (no copy estimate pays), and a BLAS provider refuses forced policies"
)]
fn a_huge_batch_of_empty_members_splits_without_overflow() {
    let blocks = Arc::new(BlockStructure::packed_column_major(1, [vec![0]]).unwrap());
    let transform =
        TreeTransformStructure::<f64>::compile_structures(&blocks, &blocks, &[]).unwrap();
    let mut engine = engine(4, CpuBatchStrategy::Outer);
    let mut workspace = TreeTransformWorkspace::<f64>::default();
    tree_transform_members_overwrite_raw(
        &mut engine,
        &mut StridedHostKernelAdapter::default(),
        &mut DefaultDenseExecutor::new(),
        &mut workspace,
        &transform,
        &blocks,
        &blocks,
        &mut [],
        &[],
        isize::MAX as usize,
        &[],
    )
    .unwrap();
    assert!(workspace.member_scratch.len() >= 2, "the moves split");
    assert!(!engine.is_broken());
    assert!(engine.with_exec(|exec| exec.budget()).is_ok());
}

/// Four packed `[8]` blocks, each a Single of coefficient `c`.
fn four_singles<C: Copy>(c: C) -> (Arc<BlockStructure>, TreeTransformStructure<C>) {
    let structure = Arc::new(BlockStructure::packed_column_major(1, vec![vec![8]; 4]).unwrap());
    let specs = (0..4)
        .map(|b| TreeTransformBlockSpec::single(b, b, c))
        .collect::<Vec<_>>();
    let transform =
        TreeTransformStructure::compile_structures(&structure, &structure, &specs).unwrap();
    (structure, transform)
}

/// What: when a member-range move fails (the write of member 2's first
/// Single, injected at the kernel adapter) on four cheap-dispatch workers
/// split into four member ranges, the batch returns that error; the
/// sibling range of member 3 still runs its moves, no grouped GEMM is
/// submitted and no scatter writes a group destination; the engine stays
/// usable.
#[test]
fn a_failing_member_range_move_stops_the_batch_before_its_gemm() {
    let (layout, dst, src, transform) = interleaved_blocks(true);
    let members = 4;
    let len = dst.required_len().unwrap();
    let source = (0..members * src.required_len().unwrap())
        .map(|v| v as f64)
        .collect::<Vec<_>>();
    let mut engine = flat_engine(4, 1, CpuBatchStrategy::Auto);
    let mut workspace = TreeTransformWorkspace::default();
    let mut out = vec![f64::NAN; members * len];
    tree_transform_members_overwrite_raw(
        &mut engine,
        &mut StridedHostKernelAdapter::default(),
        &mut DefaultDenseExecutor::new(),
        &mut workspace,
        &transform,
        &dst,
        &src,
        &mut out,
        &source,
        members,
        &[],
    )
    .unwrap();
    assert_eq!(
        workspace.member_scratch.len(),
        members,
        "four member ranges"
    );

    let base = out.as_ptr() as usize / core::mem::size_of::<f64>();
    let element = |member: usize, block: usize| base + member * len + layout[block].0;
    let mut kernels = ObservedKernels::failing_at(element(2, 0));
    let mut dense = Forwarding::default();
    let error = tree_transform_members_overwrite_raw(
        &mut engine,
        &mut kernels,
        &mut dense,
        &mut workspace,
        &transform,
        &dst,
        &src,
        &mut out,
        &source,
        members,
        &[],
    )
    .unwrap_err();
    assert!(
        matches!(error, OperationError::InvalidArgument { message } if message.contains("injected")),
        "{error:?}"
    );
    assert!(dense.attempts().is_empty());
    let writes = kernels.writes.as_ref().unwrap().lock().unwrap().clone();
    assert!(writes.contains(&element(3, 0)), "member 3's range ran");
    let group = (1..=2)
        .flat_map(|block| {
            let (offset, dims, strides) = layout[block];
            addresses(offset, &dims, &strides)
        })
        .collect::<Vec<_>>();
    for member in 0..members {
        for &at in &group {
            assert!(
                !writes.contains(&(base + member * len + at)),
                "a scatter ran"
            );
        }
    }
    assert!(!engine.is_broken());
    assert!(engine.with_exec(|exec| exec.budget()).is_ok());
}

/// What: a movement panic in one member range (member 2's Single meets
/// the source value 40) on four cheap-dispatch workers, split into four
/// member ranges, unwinds out of the batch after the fork joined and
/// leaves the engine broken.
#[test]
fn a_member_range_panic_breaks_the_engine() {
    let (structure, transform) = four_singles(Panicking(2.0));
    let members = 4;
    let len = structure.required_len().unwrap();
    let mut engine = flat_engine(4, 1, CpuBatchStrategy::Auto);
    let mut workspace = TreeTransformWorkspace::default();
    let run =
        |engine: &mut CpuEngine, workspace: &mut TreeTransformWorkspace<f64>, source: &[f64]| {
            tree_transform_members_overwrite_raw(
                engine,
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                workspace,
                &transform,
                &structure,
                &structure,
                &mut vec![0.0; members * len],
                source,
                members,
                &[],
            )
        };
    let calm = vec![1.0; members * len];
    run(&mut engine, &mut workspace, &calm).unwrap();
    assert_eq!(
        workspace.member_scratch.len(),
        members,
        "four member ranges"
    );
    let mut source = calm;
    source[2 * len + 5] = 40.0;
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run(&mut engine, &mut workspace, &source)
    }));
    assert!(unwind.is_err());
    assert!(engine.is_broken());
    assert!(engine.with_exec(|exec| exec.budget()).is_err());
}
