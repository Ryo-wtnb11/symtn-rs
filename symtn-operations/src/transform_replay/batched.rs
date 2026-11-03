use super::*;
use core::ops::Range;
use core::time::Duration;

pub(super) fn effective_tree_transform_threads(
    engine: &mut dyn Engine,
    schedule: &TreeTransformParallelSchedule,
    requested: usize,
) -> usize {
    let runnable = schedule
        .single_component_count()
        .max(schedule.pack_columns.len())
        .max(1);
    let requested = requested.max(1);

    // Why answer before reading the engine width: a serial replay needs no
    // engine lookup at all.
    if requested == 1 || runnable == 1 {
        return 1;
    }

    // Why not trust the requested count: the caller's engine cannot execute
    // more workers than it has, and scratch for phantom workers turns a harmless
    // large hint into an allocation overflow before any replay begins.
    requested
        .min(crate::host_pool::threads(engine))
        .min(runnable)
}

/// Whether replaying `structure` over `members` members on `engine` submits
/// a recoupling GEMM batch the dense executor places by its lane search
/// ([`symtn_dense::batch_searches_lanes`]). It mirrors the replay's own
/// submissions: one member at a serial width submits each Multi block alone;
/// at a parallel width the compiled jobs go in chunks of the effective worker
/// count; a member replay submits every member's jobs as one batch. The
/// effective width is the replays' own rule over the destination length.
/// Unstable plumbing for symtn's lane-allocation admission, which decides
/// participation before any write.
#[doc(hidden)]
pub fn tree_transform_searches_lanes<C: Copy>(
    engine: &mut dyn Engine,
    structure: &TreeTransformStructure<C>,
    members: usize,
) -> bool {
    let jobs = structure.recoupling_plan().jobs();
    let identity = [symtn_dense::MatrixOp::Identity; 2];
    if jobs.is_empty() {
        return false;
    }
    if members > 1 {
        let dst_len = structure.recoupling_plan().destination_len();
        let src_len = structure.recoupling_plan().source_len();
        let expanded: Vec<DenseGemmBatchJob> = jobs
            .iter()
            .flat_map(|job| {
                (0..members).map(move |member| DenseGemmBatchJob {
                    dst_offset: member * dst_len + job.dst_offset,
                    lhs_offset: member * src_len + job.lhs_offset,
                    ..*job
                })
            })
            .collect();
        let runs = symtn_dense::strided_batch_runs(&expanded);
        return symtn_dense::batch_searches_lanes(&expanded, &runs, identity, false);
    }
    let Ok(requested) = structure
        .dst_structure()
        .required_len()
        .map_err(|_| ())
        .and_then(|len| crate::host_pool::copy_width(engine, len).map_err(|_| ()))
    else {
        return false;
    };
    let threads =
        effective_tree_transform_threads(engine, structure.parallel_schedule(), requested);
    threads > 1
        && jobs.chunks(threads).any(|chunk| {
            symtn_dense::batch_searches_lanes(
                chunk,
                &symtn_dense::strided_batch_runs(chunk),
                identity,
                false,
            )
        })
}

/// Runs a replay phase with the operation's whole engine token, however
/// many workers the phase splits its items over: an unsplit phase still
/// hands its leaves the full token, whose own cost rule decides whether one
/// large block's move fans out. A broken or CUDA engine never reaches Host
/// payload.
pub(super) fn replay_region<R>(
    engine: &mut dyn Engine,
    op: impl FnOnce(Exec<'_>) -> R,
) -> Result<R, OperationError> {
    crate::host_pool::region(engine, crate::host_pool::HostPoolSite::Replay, op)
}

/// One replay fork: both halves run as jobs of `exec`. `lanes` picks the
/// token shares ([`crate::host_pool::join_split`]): `None` halves the token
/// evenly (the Multi pack's item split), `Some((left_threads, threads))`
/// gives the left the share its lanes stand for (the ownership-component
/// splits, whose lanes share a token wider than their count). The split
/// itself (and so every per-worker scratch slice and the left-before-right
/// error order) is fixed by the replay's `threads`, not by where the token
/// places a job.
pub(super) fn replay_join<A, B, RA, RB>(
    exec: Exec<'_>,
    lanes: Option<(usize, usize)>,
    left: A,
    right: B,
) -> (RA, RB)
where
    A: FnOnce(Exec<'_>) -> RA + Send + Sync,
    B: FnOnce(Exec<'_>) -> RB + Send + Sync,
    RA: Send + Sync,
    RB: Send + Sync,
{
    pool_join(exec, lanes, left, right)
}

/// Executes a validated tree-transform block list against a dense executor.
/// Single blocks apply directly; each Multi block reuses one source/destination
/// pack pair for `destination = source * U^T`.
///
/// Inlined into both the plain and profiled entry points so the
/// `Option<&mut profile>` checks constant-fold away in the unprofiled copy.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(super) fn tree_transform_blocks_with_batched_recoupling<A, E, D, C>(
    engine: &mut dyn Engine,
    kernels: &mut A,
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    task: TreeTransformTaskView<'_, C>,
    structure_identity: &Arc<()>,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: DestinationAlpha<'_, D, C>,
    mode: DestinationMode<D>,
    mut profile: Option<&mut TreeTransformReplayProfile>,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    let layouts = task.layouts();
    // One gate for every block list this call visits on the calling
    // thread; forked leaves build their own.
    let gate = LeafGate::new(layouts.max_planned_elements());
    let recoupling_plan = task.recoupling_plan();
    workspace.prepare_fused_indices(1, layouts.max_fused_rank())?;

    // All-Single structures (abelian recoupling is diagonal) skip the batch
    // machinery entirely: no pack scratch, no job list, no scatter pass.
    if recoupling_plan.is_empty() {
        return crate::host_pool::with_exec(engine, |mut exec| {
            for block in task.blocks() {
                let TreeTransformBlock::Single {
                    dst_layout,
                    src_layout,
                    coefficient,
                } = *block
                else {
                    unreachable!("checked above: no Multi blocks");
                };
                let start = profile.as_ref().map(|_| std::time::Instant::now());
                tree_transform_single_on_token(
                    exec.reborrow(),
                    &gate,
                    kernels,
                    &mut workspace.zero_strides,
                    Some(workspace.fused_indices.as_mut_slice()),
                    layouts,
                    dst_layout,
                    src_layout,
                    task.single_coefficients()[coefficient],
                    task.storage_conjugate(),
                    dst_data,
                    0,
                    src_data,
                    alpha.at(layouts.entry(dst_layout).offset),
                    mode,
                )?;
                if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
                    let elapsed = start.elapsed();
                    profile.single_blocks += 1;
                    profile.single_total += elapsed;
                    profile.strided_kernel += elapsed;
                }
            }
            Ok(())
        });
    }

    let start = profile.as_ref().map(|_| std::time::Instant::now());
    let converted = ensure_recoupling_coefficients(workspace, task, structure_identity)?;
    if converted {
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            profile.multi_coefficient_prepare += start.elapsed();
        }
    }

    // Singles apply directly in replay order. Multi blocks are packed through
    // the compile-time recoupling entries, whose order is chosen to form
    // same-shape strided GEMM runs.
    crate::host_pool::with_exec(engine, |mut exec| {
        for block in task.blocks() {
            let TreeTransformBlock::Single {
                dst_layout,
                src_layout,
                coefficient,
            } = *block
            else {
                continue;
            };
            // Timestamps only under profiling: the per-block clock reads are
            // measurable against microsecond replays.
            let start = profile.as_ref().map(|_| std::time::Instant::now());
            tree_transform_single_on_token(
                exec.reborrow(),
                &gate,
                kernels,
                &mut workspace.zero_strides,
                Some(workspace.fused_indices.as_mut_slice()),
                layouts,
                dst_layout,
                src_layout,
                task.single_coefficients()[coefficient],
                task.storage_conjugate(),
                dst_data,
                0,
                src_data,
                alpha.at(layouts.entry(dst_layout).offset),
                mode,
            )?;
            if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
                let elapsed = start.elapsed();
                profile.single_blocks += 1;
                profile.single_total += elapsed;
                profile.strided_kernel += elapsed;
            }
        }
        Ok(())
    })?;

    // Size the pack scratch for the largest Multi block once, so a fresh
    // per-call workspace allocates each buffer once instead of growing it
    // block by block.
    let mut largest = (0usize, 0usize);
    for (block_index, _) in recoupling_plan.entries() {
        if let TreeTransformBlock::Multi {
            dst_count,
            src_count,
            element_count,
            ..
        } = *recoupling_multi_block(task, block_index)?
        {
            largest.0 = largest.0.max(
                element_count
                    .checked_mul(src_count)
                    .ok_or_else(|| OperationError::ElementCountOverflow)?,
            );
            largest.1 = largest.1.max(
                element_count
                    .checked_mul(dst_count)
                    .ok_or_else(|| OperationError::ElementCountOverflow)?,
            );
        }
    }
    workspace.packed.source_mut().reserve_overwrite(largest.0);
    workspace.prepare_packed_destination(largest.1, D::zero());

    for (block_index, job) in recoupling_plan.entries() {
        let TreeTransformBlock::Multi {
            dst_layout_start,
            dst_count,
            src_layout_start,
            src_count,
            element_count,
            ..
        } = *recoupling_multi_block(task, block_index)?
        else {
            unreachable!("recoupling_multi_block only returns Multi blocks");
        };
        debug_assert_eq!(job.rows, element_count);
        debug_assert_eq!(job.contracted, src_count);
        debug_assert_eq!(job.cols, dst_count);
        let source_len = element_count
            .checked_mul(src_count)
            .ok_or_else(|| OperationError::ElementCountOverflow)?;
        let destination_len = element_count
            .checked_mul(dst_count)
            .ok_or_else(|| OperationError::ElementCountOverflow)?;
        let start = profile.as_ref().map(|_| std::time::Instant::now());
        workspace.prepare_packed_destination(destination_len, D::zero());
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            profile.multi_workspace_prepare += start.elapsed();
        }
        let start = profile.as_ref().map(|_| std::time::Instant::now());
        let HostTreeTransformWorkspace {
            packed,
            fused_indices,
            ..
        } = &mut *workspace;
        let storage_conjugate = task.storage_conjugate();
        packed
            .source_mut()
            .overwrite_with_columns(source_len, |mut writer| {
                crate::host_pool::with_exec(engine, |_| {
                    for src_index in 0..src_count {
                        let entry = src_layout_start + src_index;
                        writer.write_column(
                            src_index * element_count,
                            layouts.role(entry)?,
                            src_data,
                            layouts.entry(entry).offset,
                            fused_indices.as_mut_slice(),
                            |value: D| value.maybe_conj(storage_conjugate),
                        )?;
                    }
                    Ok(())
                })
            })?;
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            profile.multi_blocks += 1;
            profile.packed_columns += src_count;
            profile.multi_pack += start.elapsed();
        }

        let start = profile.as_ref().map(|_| std::time::Instant::now());
        {
            let local_job = DenseGemmBatchJob {
                dst_offset: 0,
                lhs_offset: 0,
                ..*job
            };
            let (source, destination) = workspace.packed.source_and_destination_mut();
            recoupling_gemm_batch(
                engine,
                dense,
                destination.as_mut_slice(),
                source.as_slice(),
                &workspace.coefficient_scratch,
                core::slice::from_ref(&local_job),
                &[1],
            )?;
        }
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            let elapsed = start.elapsed();
            profile.multi_dense_matmul_call += elapsed;
            profile.multi_matmul_total += elapsed;
        }

        let start = profile.as_ref().map(|_| std::time::Instant::now());
        crate::host_pool::with_exec(engine, |mut exec| {
            for dst_index in 0..dst_count {
                scatter_column_on_token(
                    exec.reborrow(),
                    &gate,
                    kernels,
                    &mut workspace.zero_strides,
                    Some(workspace.fused_indices.as_mut_slice()),
                    layouts,
                    dst_layout_start + dst_index,
                    workspace.packed.destination().as_slice(),
                    dst_index * element_count,
                    dst_data,
                    0,
                    alpha.at(layouts.entry(dst_layout_start + dst_index).offset),
                    mode,
                )?;
            }
            Ok(())
        })?;
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            profile.scattered_columns += dst_count;
            profile.multi_scatter += start.elapsed();
        }
    }
    Ok(())
}

pub(super) fn parallel_split(items: usize, threads: usize) -> usize {
    let left_threads = threads / 2;
    items
        .saturating_mul(left_threads)
        .div_ceil(threads)
        .clamp(1, items - 1)
}

/// Packs `items` through `packed`, whose slot 0 is packed offset
/// `packed_start`, split over `threads` lanes on item boundaries.
#[allow(clippy::too_many_arguments)]
fn replay_pack_columns<D>(
    fused_indices: &mut [usize],
    max_fused_rank: usize,
    layouts: &TreeTransformLayoutTable,
    items: &[TreeTransformPackReplay],
    mut packed: PackColumnWriter<'_, D>,
    packed_start: usize,
    src_data: &[D],
    storage_conjugate: bool,
    threads: usize,
    exec: Exec<'_>,
) -> Result<(), OperationError>
where
    D: DenseRecouplingScalar + ConjugateValue,
{
    if items.is_empty() {
        return Ok(());
    }
    if threads <= 1 || items.len() == 1 {
        let fused_index = &mut fused_indices[..max_fused_rank];
        for item in items {
            packed.write_column(
                item.packed_offset - packed_start,
                layouts.role(item.src_layout)?,
                src_data,
                layouts.entry(item.src_layout).offset,
                fused_index,
                |value: D| value.maybe_conj(storage_conjugate),
            )?;
        }
        return Ok(());
    }

    let middle = parallel_split(items.len(), threads);
    let boundary = items[middle].packed_offset;
    let (left_packed, right_packed) = packed.split_at(boundary - packed_start)?;
    let (left_items, right_items) = items.split_at(middle);
    let left_threads = threads / 2;
    let right_threads = threads - left_threads;
    let (left_indices, mut right_indices) =
        split_lane_indices(fused_indices, left_threads, threads, max_fused_rank);
    let (left, right) = replay_join(
        exec,
        None,
        |exec| {
            replay_pack_columns(
                left_indices,
                max_fused_rank,
                layouts,
                left_items,
                left_packed,
                packed_start,
                src_data,
                storage_conjugate,
                left_threads,
                exec,
            )
        },
        |exec| {
            replay_pack_columns(
                right_indices.get(),
                max_fused_rank,
                layouts,
                right_items,
                right_packed,
                boundary,
                src_data,
                storage_conjugate,
                right_threads,
                exec,
            )
        },
    );
    left?;
    right
}

/// The split of `parts >= 2` indivisible parts, in order, into a nonempty
/// left and right list whose left side's weight is closest to the fraction
/// `(threads / 2) / threads` of the total; a tie goes to the earlier
/// boundary. `weight_before(k)` is the weight of parts `0..k` plus a common
/// base, nondecreasing in `k`.
///
/// Why weight rather than part count: replay moves are memory-bound, so a
/// lane's time follows the bytes its parts move, and a part (an ownership
/// component) cannot be divided between lanes.
pub(super) fn balanced_boundary(
    parts: usize,
    threads: usize,
    weight_before: impl Fn(usize) -> usize,
) -> usize {
    debug_assert!(parts >= 2 && threads >= 2);
    let base = weight_before(0);
    let target = (weight_before(parts) - base) as u128 * (threads / 2) as u128;
    let scaled = |boundary: usize| (weight_before(boundary) - base) as u128 * threads as u128;
    // The first legal boundary whose left weight reaches the target, or the
    // last legal one.
    let (mut lo, mut hi) = (1, parts - 1);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if scaled(mid) >= target {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    if lo > 1 && target.abs_diff(scaled(lo - 1)) <= target.abs_diff(scaled(lo)) {
        lo - 1
    } else {
        lo
    }
}

/// `exec` narrowed to `width`, the whole-token width of its one component
/// ([`split_workers`]): a lane never holds more workers than its component's
/// largest planned block would use. O(1).
pub(super) fn granted_width(exec: Exec<'_>, width: usize) -> Exec<'_> {
    exec.limit(core::num::NonZeroUsize::new(width).unwrap_or(core::num::NonZeroUsize::MIN))
}

/// Strided's width for a copy of `elements` on the call's whole token
/// `exec`, exactly: the gate's O(1) answer where it decides one worker or
/// the whole budget; otherwise, inside the gate's exact domain, from what
/// this call has learned about the width transitions, asking Strided only
/// when that knowledge does not decide the count; outside the domain
/// Strided's own answer.
///
/// `levels` holds, for each width `k` of the engine budget `B`, the
/// smallest count known to reach width `k` (`reach`, `usize::MAX` when
/// unknown) and one more than the largest count known not to (`miss`, zero
/// when unknown). Why that is exact: inside the domain `width >= k` is
/// monotone in the count for every `k` (the argument of
/// `LeafGate::scan_full`: a wider width's objective falls faster, a width
/// that starts paying later has a larger cost and cannot undercut the
/// current choice, and the domain makes threadteam's rounded comparisons
/// agree), so one Strided answer `w` at count `m` proves `width >= k` for
/// every count from `m` on when `k <= w`, and `width < k` up to `m` when
/// `k > w`.
///
/// Cost: an undecided count costs one Strided search (O(B)) and an O(B)
/// update; a count the knowledge decides costs O(log B). A call therefore
/// searches at most once per distinct count its decisions need, however
/// many chunks or components repeat it.
pub(super) fn whole_width(
    exec: &Exec<'_>,
    gate: &LeafGate,
    levels: &mut Vec<usize>,
    elements: usize,
) -> usize {
    let budget = exec.budget().get();
    match gate.decided(exec, elements) {
        (Some(false), _) => return 1,
        (_, Some(true)) => return budget,
        _ => {}
    }
    if gate.exact_end(exec).is_none_or(|exact| elements > exact) {
        return LeafGate::width(exec, elements);
    }
    if levels.len() != 2 * (budget + 1) {
        levels.clear();
        levels.resize(budget + 1, usize::MAX);
        levels.resize(2 * (budget + 1), 0);
        // Every count reaches one worker.
        levels[1] = 0;
    }
    let (reach, miss) = levels.split_at_mut(budget + 1);
    // Both lists are nondecreasing in `k`: the largest width the count is
    // known to reach, and the smallest it is known to miss.
    let reached = reach[1..].partition_point(|&count| count <= elements);
    let missed = 1 + miss[1..].partition_point(|&above| above <= elements);
    if reached + 1 == missed {
        return reached;
    }
    let width = LeafGate::width(exec, elements);
    for count in &mut reach[1..=width] {
        *count = (*count).min(elements);
    }
    for above in &mut miss[width + 1..] {
        *above = (*above).max(elements + 1);
    }
    width
}

/// The workers of the call's whole token `exec` a split of components
/// shares, recording each component's whole-token width in its slot
/// ([`whole_width`] of its largest planned block): one (run inline on the
/// whole token) when the replay has one lane or Strided's own rule gives
/// the components' `total` moved elements one worker; otherwise the
/// smaller of the token's budget and the sum of the components' widths.
/// The caller splits over `min(threads, workers)` lanes, which share those
/// workers in proportion, each lane narrowed to its component's width when
/// it holds only that component, provided the split [`split_pays`].
///
/// Why the token budget and not the lane count: the replay's lane count is
/// capped by its component count, while each component may use several
/// workers; lanes holding only one worker each would leave the rest of the
/// token idle. Why the total only admits: its copy width is that of one
/// copy, not of a phase of independent components, which can run side by
/// side up to their summed widths. Cost: one [`whole_width`] answer for the
/// total and one per component.
pub(super) fn split_workers<'w>(
    exec: &Exec<'_>,
    gate: &LeafGate,
    levels: &mut Vec<usize>,
    threads: usize,
    total: usize,
    components: impl IntoIterator<Item = (usize, &'w mut usize)>,
) -> usize {
    if threads <= 1 || whole_width(exec, gate, levels, total) <= 1 {
        return 1;
    }
    let mut sum = 0usize;
    for (widest, width) in components {
        *width = whole_width(exec, gate, levels, widest);
        sum = sum.saturating_add(*width);
    }
    exec.budget().get().min(sum).max(1)
}

/// Whether splitting `parts` components over `lanes` lanes that share
/// `workers` slots of the call's whole token `exec` is cheaper than running
/// them one after another on the whole token. `weight_before(p)` is the
/// moved elements of the components before `p` (defined up to `parts`) and
/// `width(p)` component `p`'s whole-token width ([`split_workers`]).
///
/// Both sides cost a component by Strided's own objective for a copy, the
/// one its width minimises: the serial estimate of its moved elements over
/// the width it runs at, plus the token's dispatch cost of a section that
/// wide (no dispatch at one worker). On the whole token a component runs at
/// `width(p)`; in the split at `width(p)` capped by its lane's share of
/// `workers`, which follows [`split_join_parts`] and
/// [`split_join_components`] exactly ([`balanced_boundary`] and
/// [`crate::host_pool::lane_share`]). The split costs its slowest lane plus
/// one two-wide section per fork on the way to it.
///
/// Only an overwrite asks: an accumulating phase moves every block on one
/// worker, where the split's one-worker lanes lose nothing.
///
/// Why: lanes that share a token narrower than the components' summed
/// widths can each hold fewer workers than Strided would give a component
/// alone, so an indivisible large component on a narrow grant, plus the
/// forks that reach it, can bound the split above the sequential
/// whole-token run. Cost: O(parts + lanes log parts), no allocation.
pub(super) fn split_pays(
    exec: &Exec<'_>,
    parts: usize,
    lanes: usize,
    workers: usize,
    weight_before: &impl Fn(usize) -> usize,
    width: &impl Fn(usize) -> usize,
) -> bool {
    let time = |part: usize, at: usize| {
        let estimate = strided_kernel::execution::KernelClass::Copy
            .estimate(weight_before(part + 1) - weight_before(part));
        match core::num::NonZeroUsize::new(width(part).min(at)) {
            Some(width) if width.get() > 1 => (estimate
                / u32::try_from(width.get()).unwrap_or(u32::MAX))
            .saturating_add(exec.dispatch_cost(width)),
            _ => estimate,
        }
    };
    let fork =
        exec.dispatch_cost(core::num::NonZeroUsize::new(2).unwrap_or(core::num::NonZeroUsize::MIN));
    let sequential = (0..parts).fold(Duration::ZERO, |sum, part| {
        sum.saturating_add(time(part, usize::MAX))
    });
    split_time(0..parts, lanes, workers, weight_before, &time, fork) < sequential
}

/// The slowest lane of [`split_pays`]'s split of `parts` over `lanes` lanes
/// holding `budget` slots, each fork adding `fork`.
fn split_time(
    parts: Range<usize>,
    lanes: usize,
    budget: usize,
    weight_before: &impl Fn(usize) -> usize,
    time: &impl Fn(usize, usize) -> Duration,
    fork: Duration,
) -> Duration {
    if lanes <= 1 || parts.len() <= 1 {
        return parts.fold(Duration::ZERO, |sum, part| {
            sum.saturating_add(time(part, budget))
        });
    }
    let middle = parts.start
        + balanced_boundary(parts.len(), lanes, |part| weight_before(parts.start + part));
    let left_lanes = lanes / 2;
    let right_lanes = lanes - left_lanes;
    if budget < 2 {
        // A one-slot token runs both sides inline, one after the other.
        return split_time(
            parts.start..middle,
            left_lanes,
            budget,
            weight_before,
            time,
            fork,
        )
        .saturating_add(split_time(
            middle..parts.end,
            right_lanes,
            budget,
            weight_before,
            time,
            fork,
        ));
    }
    let left_budget = crate::host_pool::lane_share(budget, left_lanes, lanes);
    let left = split_time(
        parts.start..middle,
        left_lanes,
        left_budget,
        weight_before,
        time,
        fork,
    );
    let right = split_time(
        middle..parts.end,
        right_lanes,
        budget - left_budget,
        weight_before,
        time,
        fork,
    );
    fork.saturating_add(left.max(right))
}

/// Recursive `replay_join` over the address-ordered parts `parts` (Single
/// ownership atoms or a subset of them): `lo(p)` is part `p`'s first
/// destination element and `weight_before(p)` the destination elements of
/// the parts before it, defined up to `parts.end`. `dst` is the destination
/// from absolute offset `dst_start`. A split lands on a part start chosen by
/// [`balanced_boundary`], with `threads / 2` lanes and the matching share of
/// the token to the left; `leaf` writes a range of parts into its part of
/// `dst` with one worker's traversal scratch and that worker's share of
/// `exec`. An unsplit list keeps the caller's `gate` and the whole of
/// `exec`, narrowed to `part_width(p)` (the part's whole-token width) when
/// it is a single part `p`; forked leaves build their own gate bounded by
/// `gate_bound`.
///
/// No `unsafe` is needed: atoms are maximal runs of overlapping item
/// intervals, so every item lies inside one atom and `split_at_mut` at an
/// atom start leaves each item wholly in one half. The same recursion serves
/// the initialised replay (`T = D`) and the owned writer
/// (`T = MaybeUninit<D>`), whose publication waits for every join.
#[allow(clippy::too_many_arguments)]
pub(super) fn split_join_parts<T: Send + Sync>(
    parts: Range<usize>,
    lo: &(impl Fn(usize) -> isize + Sync),
    weight_before: &(impl Fn(usize) -> usize + Sync),
    part_width: &(impl Fn(usize) -> usize + Sync),
    dst: &mut [T],
    dst_start: isize,
    fused_indices: &mut [usize],
    max_fused_rank: usize,
    threads: usize,
    exec: Exec<'_>,
    gate: Option<&LeafGate>,
    gate_bound: usize,
    leaf: &(impl Fn(
        Range<usize>,
        &mut [T],
        isize,
        &mut [usize],
        Exec<'_>,
        &LeafGate,
    ) -> Result<(), OperationError>
          + Sync),
) -> Result<(), OperationError> {
    if parts.is_empty() {
        return Ok(());
    }
    if threads <= 1 || parts.len() == 1 {
        let own_gate;
        let gate = match gate {
            Some(gate) => gate,
            None => {
                own_gate = LeafGate::new(gate_bound);
                &own_gate
            }
        };
        let exec = if parts.len() == 1 {
            granted_width(exec, part_width(parts.start))
        } else {
            exec
        };
        return leaf(
            parts,
            dst,
            dst_start,
            &mut fused_indices[..max_fused_rank],
            exec,
            gate,
        );
    }

    let middle = parts.start
        + balanced_boundary(parts.len(), threads, |part| {
            weight_before(parts.start + part)
        });
    let boundary = lo(middle);
    let split =
        usize::try_from(boundary - dst_start).map_err(|_| OperationError::ElementCountOverflow)?;
    let (left_data, right_data) = dst.split_at_mut(split);
    let left_threads = threads / 2;
    let right_threads = threads - left_threads;
    let (left_indices, mut right_indices) =
        split_lane_indices(fused_indices, left_threads, threads, max_fused_rank);
    let (left, right) = replay_join(
        exec,
        Some((left_threads, threads)),
        |exec| {
            split_join_parts(
                parts.start..middle,
                lo,
                weight_before,
                part_width,
                left_data,
                dst_start,
                left_indices,
                max_fused_rank,
                left_threads,
                exec,
                None,
                gate_bound,
                leaf,
            )
        },
        |exec| {
            split_join_parts(
                middle..parts.end,
                lo,
                weight_before,
                part_width,
                right_data,
                boundary,
                right_indices.get(),
                max_fused_rank,
                right_threads,
                exec,
                None,
                gate_bound,
                leaf,
            )
        },
    );
    left?;
    right
}

/// A Single atom the split runs: its index, the destination elements of the
/// listed atoms before it, and its whole-token width ([`split_workers`]).
#[derive(Clone, Copy, Debug)]
pub(super) struct NarrowAtom {
    atom: usize,
    weight_before: usize,
    width: usize,
}

/// The Single phase of a threaded replay over its ownership atoms, on the
/// phase's whole token `exec` by Strided's own width rule (`gate`):
///
/// 1. Inline admission: when Strided gives the phase's total moved elements
///    one worker (or there is one atom or one lane), every Single runs in
///    one list on the whole token, as an unsplit list does.
/// 2. Wide tier (overwrites only; an accumulating `widen = false` replay
///    never runs a plan): an atom whose largest planned Single would use
///    every worker of the token runs alone on it, one after another.
/// 3. Capped split: the remaining atoms split by moved elements
///    ([`split_join_parts`]) over `min(threads, workers)` lanes sharing
///    [`split_workers`] workers: the token's budget capped by the sum of
///    their own widths. Their total is admitted again, so a small remainder
///    runs inline, and a split that is not cheaper than running them one
///    after another on the whole token ([`split_pays`]) runs that way.
/// 4. Singleton grant: a lane left with one atom is narrowed to that atom's
///    whole-token width.
///
/// `leaf` writes one list of Singles into `dst` from absolute offset
/// `dst_start`. `narrow` is the split's atom list (O(atoms) workspace
/// scratch, kept and charged; untouched by an inline phase) and `levels`
/// what the call has learned of widths ([`whole_width`], O(B) workspace
/// scratch, which even the inline admission may fill).
///
/// Cost: inside the gate's exact domain, O(1) per atom for admission and
/// classification, and per split atom one [`whole_width`] answer: O(log B)
/// for engine budget B when the gate or what the call has learned decides
/// it, otherwise one Strided search of O(B) (once per distinct count); past
/// the domain each answer is a Strided search. The split decision adds
/// O(atoms + lanes log atoms) ([`split_pays`]).
#[allow(clippy::too_many_arguments)]
pub(super) fn replay_single_phase<T: Send + Sync>(
    schedule: &TreeTransformParallelSchedule,
    narrow: &mut Vec<NarrowAtom>,
    levels: &mut Vec<usize>,
    dst: &mut [T],
    fused_indices: &mut [usize],
    max_fused_rank: usize,
    threads: usize,
    mut exec: Exec<'_>,
    gate: &LeafGate,
    gate_bound: usize,
    widen: bool,
    leaf: &(impl Fn(
        &[u32],
        &mut [T],
        isize,
        &mut [usize],
        Exec<'_>,
        &LeafGate,
    ) -> Result<(), OperationError>
          + Sync),
) -> Result<(), OperationError> {
    let atoms = &schedule.single_atoms;
    let items = &schedule.singles;
    let parts = schedule.single_component_count();
    if parts == 0 {
        return Ok(());
    }
    let atom_items = |atom: usize| &items[atoms[atom].item as usize..atoms[atom + 1].item as usize];
    if threads <= 1
        || parts == 1
        || whole_width(&exec, gate, levels, atoms[parts].weight_before) <= 1
    {
        return leaf(
            &items[atoms[0].item as usize..atoms[parts].item as usize],
            dst,
            0,
            &mut fused_indices[..max_fused_rank],
            exec,
            gate,
        );
    }
    // Why the largest atom may decide for all: inside the gate's exact
    // domain saturation is monotone, so a largest atom that does not fill
    // the token proves no smaller one does; outside it every atom is asked.
    let tiered = widen
        && gate.may_saturate_up_to(&exec, schedule.widest_single)
        && (0..parts).any(|atom| gate.saturates(&exec, atoms[atom].widest));

    narrow.clear();
    let mut weight = 0usize;
    for atom in 0..parts {
        if tiered && gate.saturates(&exec, atoms[atom].widest) {
            leaf(
                atom_items(atom),
                dst,
                0,
                &mut fused_indices[..max_fused_rank],
                exec.reborrow(),
                gate,
            )?;
        } else {
            narrow.push(NarrowAtom {
                atom,
                weight_before: weight,
                width: 1,
            });
            weight += atoms[atom + 1].weight_before - atoms[atom].weight_before;
        }
    }
    if narrow.is_empty() {
        return Ok(());
    }
    let listed = narrow.len();
    narrow.push(NarrowAtom {
        atom: parts,
        weight_before: weight,
        width: 1,
    });
    let workers = split_workers(
        &exec,
        gate,
        levels,
        threads,
        weight,
        narrow[..listed]
            .iter_mut()
            .map(|part| (atoms[part.atom].widest, &mut part.width)),
    );
    // Lanes are the replay's split width; the workers they share are the
    // part of the token the components can use. A split that is not cheaper
    // than the sequential whole-token run keeps that run; an accumulating
    // phase moves every block on one worker, so its split loses nothing.
    let mut lanes = threads.min(workers);
    if lanes > 1
        && widen
        && !split_pays(
            &exec,
            listed,
            lanes,
            workers,
            &|part| narrow[part].weight_before,
            &|part| narrow[part].width,
        )
    {
        lanes = 1;
    }
    let exec = if lanes > 1 {
        granted_width(exec, workers)
    } else {
        exec
    };
    let narrow = &*narrow;
    split_join_parts(
        0..listed,
        &|part| atoms[narrow[part].atom].lo,
        &|part| narrow[part].weight_before,
        &|part| narrow[part].width,
        dst,
        0,
        fused_indices,
        max_fused_rank,
        lanes,
        exec,
        Some(gate),
        gate_bound,
        &|range: Range<usize>,
          dst: &mut [T],
          dst_start,
          fused_index: &mut [usize],
          mut exec: Exec<'_>,
          gate| {
            for part in range {
                leaf(
                    atom_items(narrow[part].atom),
                    &mut *dst,
                    dst_start,
                    &mut *fused_index,
                    exec.reborrow(),
                    gate,
                )?;
            }
            Ok(())
        },
    )
}

/// One ownership component's share of a Multi chunk's scatter: the groups
/// `chunk_scatter_groups[groups]`. A chunk's unit list is ordered by
/// component and ends with a sentinel holding the total weight.
#[derive(Clone, Copy, Debug)]
pub(super) struct ScatterUnit {
    groups: (usize, usize),
    component: u32,
    /// Destination elements the unit writes.
    weight: usize,
    /// Destination elements the list's earlier units write.
    weight_before: usize,
    /// Elements of the unit's largest planned column.
    widest: usize,
    /// The unit's whole-token width, recorded by [`split_workers`].
    width: usize,
}

impl ScatterUnit {
    /// The unit's groups within `chunk_scatter_groups`.
    pub(super) fn groups<'g>(&self, chunk_groups: &'g [usize]) -> &'g [usize] {
        &chunk_groups[self.groups.0..self.groups.1]
    }
}

/// The wide tier of a chunk's component scatter (see
/// [`replay_single_phase`]): runs every unit whose largest planned column
/// would use every worker of the whole token `exec` alone on that token, one after
/// another, then leaves only the narrow units in `units` (component order
/// kept, weights re-summed, sentinel restored) for [`split_join_components`].
/// An accumulating replay (`widen = false`), a serial width or a single unit
/// leaves the list as it is.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_wide_units<T>(
    units: &mut Vec<ScatterUnit>,
    slots: &mut [&mut [T]],
    fused_index: &mut [usize],
    threads: usize,
    exec: &mut Exec<'_>,
    gate: &LeafGate,
    widen: bool,
    leaf: &impl Fn(
        &[ScatterUnit],
        &mut [&mut [T]],
        usize,
        &mut [usize],
        Exec<'_>,
        &LeafGate,
    ) -> Result<(), OperationError>,
) -> Result<(), OperationError> {
    if !widen || threads <= 1 || units.len() <= 2 {
        return Ok(());
    }
    let Some(mut sentinel) = units.pop() else {
        return Ok(());
    };
    let mut any_wide = false;
    for unit in 0..units.len() {
        if gate.saturates(exec, units[unit].widest) {
            any_wide = true;
            leaf(
                &units[unit..unit + 1],
                slots,
                0,
                fused_index,
                exec.reborrow(),
                gate,
            )?;
        }
    }
    if any_wide {
        units.retain(|unit| !gate.saturates(exec, unit.widest));
        let mut weight = 0usize;
        for unit in units.iter_mut() {
            unit.weight_before = weight;
            weight += unit.weight;
        }
        sentinel.weight_before = weight;
    }
    if !units.is_empty() {
        units.push(sentinel);
    }
    Ok(())
}

/// Where a Multi scatter phase writes: the whole destination on a serial
/// schedule, or its ownership atoms, split once per replay in address order
/// and held in component order (`TreeTransformScatterAtom::slot`).
pub(super) enum ScatterTarget<'a, T> {
    Whole(&'a mut [T]),
    Atoms(Vec<&'a mut [T]>),
}

impl<'a, T> ScatterTarget<'a, T> {
    /// Splits `dst` into the schedule's scatter atoms when more than one
    /// component can run at once; O(atoms), reusing the workspace's slice
    /// capacity.
    ///
    /// The scatter split's scratch: the atom list and the chunk unit list,
    /// O(atoms + chunk groups) metadata kept by a reused workspace and
    /// allocated once by a call-local one. A serial or one-component scatter
    /// allocates neither. (The other split scratch is the Single phase's
    /// narrow list, [`replay_single_phase`], and the width levels both
    /// phases learn, [`whole_width`], O(engine width).)
    pub(super) fn new<D>(
        workspace: &mut TreeTransformWorkspace<D>,
        schedule: &TreeTransformParallelSchedule,
        threads: usize,
        jobs: usize,
        dst: &'a mut [T],
    ) -> Result<Self, OperationError> {
        if threads <= 1 || schedule.scatter_component_count() <= 1 {
            return Ok(Self::Whole(dst));
        }
        // A chunk's units and their sentinel.
        workspace.chunk_units.clear();
        workspace.chunk_units.reserve(threads.min(jobs) + 1);
        let mut slots = workspace.take_atom_slices();
        slots.resize_with(schedule.scatter_atoms.len(), Default::default);
        let mut rest = dst;
        let mut rest_start = 0isize;
        for atom in &schedule.scatter_atoms {
            let skip = usize::try_from(atom.lo - rest_start)
                .ok()
                .filter(|&skip| {
                    skip.checked_add(atom.len)
                        .is_some_and(|end| end <= rest.len())
                })
                .ok_or(OperationError::InvalidArgument {
                    message: "tree transform scatter atom lies outside the destination",
                })?;
            let (_, tail) = core::mem::take(&mut rest).split_at_mut(skip);
            let (atom_data, tail) = tail.split_at_mut(atom.len);
            slots[atom.slot as usize] = atom_data;
            rest = tail;
            rest_start = atom.lo + atom.len as isize;
        }
        Ok(Self::Atoms(slots))
    }

    /// Returns the atom-slice capacity to `workspace`.
    pub(super) fn finish<D>(self, workspace: &mut TreeTransformWorkspace<D>) {
        if let Self::Atoms(slots) = self {
            workspace.keep_atom_slices(slots);
        }
    }
}

/// The absolute destination offset of the first element of `item`'s atom:
/// the base its layout offset is rebased against inside that atom's slice.
#[inline]
pub(super) fn atom_start(
    layouts: &TreeTransformLayoutTable,
    item: &TreeTransformScatterReplay,
) -> Result<isize, OperationError> {
    Ok(layouts.entry(item.dst_layout).offset - offset_to_isize(item.local_base)?)
}

/// One chunk's scatter over its ownership components (`units`, with their
/// sentinel) on the phase's whole token `exec`, in the order of
/// [`replay_single_phase`]: inline admission when Strided gives the chunk's
/// moved elements one worker (or there is one unit or one lane); then the
/// saturating units alone ([`run_wide_units`]); then the remaining units over
/// `min(threads, workers)` lanes sharing [`split_workers`] workers,
/// re-admitted on their own total and against the sequential whole-token
/// run ([`split_pays`], [`split_join_components`]); finally a lane left with one unit narrowed to
/// its whole-token width. Admission asks the gate for every threaded
/// multi-component chunk, planned or not and accumulating or not.
///
/// Cost per chunk: O(1) gate answers inside its exact domain and one
/// [`whole_width`] answer per split unit and for the total: O(log B) for
/// engine budget B when the gate or what the call has learned in `levels`
/// decides it, otherwise one Strided search, O(B), plus O(units + lanes log
/// units) for the split decision ([`split_pays`]). Identical chunks repeat
/// no search; a new undecided count searches once. Past the exact domain
/// each answer is a Strided search.
///
/// Why per chunk and not once per phase: a chunk's packed destination exists
/// only between its GEMM and the next chunk's, so a split cannot outlive it;
/// the inline rule keeps a call of small chunks from paying a fork each.
#[allow(clippy::too_many_arguments)]
pub(super) fn scatter_chunk_components<T: Send + Sync>(
    units: &mut Vec<ScatterUnit>,
    levels: &mut Vec<usize>,
    component_slots: &[u32],
    slots: &mut [&mut [T]],
    fused_indices: &mut [usize],
    max_fused_rank: usize,
    threads: usize,
    mut exec: Exec<'_>,
    gate: &LeafGate,
    gate_bound: usize,
    widen: bool,
    leaf: &(impl Fn(
        &[ScatterUnit],
        &mut [&mut [T]],
        usize,
        &mut [usize],
        Exec<'_>,
        &LeafGate,
    ) -> Result<(), OperationError>
          + Sync),
) -> Result<(), OperationError> {
    let parts = units.len().saturating_sub(1);
    if parts == 0 {
        return Ok(());
    }
    let total = units[parts].weight_before;
    if threads <= 1 || parts == 1 || whole_width(&exec, gate, levels, total) <= 1 {
        return leaf(
            &units[..parts],
            slots,
            0,
            &mut fused_indices[..max_fused_rank],
            exec,
            gate,
        );
    }
    #[cfg(test)]
    SCATTER_SPLITS.with(|splits| splits.set(splits.get() + 1));
    run_wide_units(
        units,
        slots,
        &mut fused_indices[..max_fused_rank],
        threads,
        &mut exec,
        gate,
        widen,
        leaf,
    )?;
    let parts = units.len().saturating_sub(1);
    if parts == 0 {
        return Ok(());
    }
    let total = units[parts].weight_before;
    let workers = split_workers(
        &exec,
        gate,
        levels,
        threads,
        total,
        units[..parts]
            .iter_mut()
            .map(|unit| (unit.widest, &mut unit.width)),
    );
    // Lanes are the replay's split width; the workers they share are the
    // part of the token the components can use. A split that is not cheaper
    // than the sequential whole-token run keeps that run; an accumulating
    // phase moves every block on one worker, so its split loses nothing.
    let mut lanes = threads.min(workers);
    if lanes > 1
        && widen
        && !split_pays(
            &exec,
            parts,
            lanes,
            workers,
            &|part| units[part].weight_before,
            &|part| units[part].width,
        )
    {
        lanes = 1;
    }
    let exec = if lanes > 1 {
        granted_width(exec, workers)
    } else {
        exec
    };
    split_join_components(
        units,
        component_slots,
        slots,
        0,
        fused_indices,
        max_fused_rank,
        lanes,
        exec,
        Some(gate),
        gate_bound,
        leaf,
    )
}

#[cfg(test)]
thread_local! {
    /// Chunk scatters this thread split between lanes (test observation).
    pub(super) static SCATTER_SPLITS: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}

/// Recursive `replay_join` over one chunk's scatter `units` (ending with a
/// sentinel): `slots` are the atoms of components `component_slots`-indexed
/// from slot `slot_base`. A split lands between two units at the boundary
/// [`balanced_boundary`] picks over their destination elements; the slots
/// split at the right unit's first atom, so each side holds exactly the atoms
/// of its own components. `leaf` scatters its units' groups into their
/// atoms; lanes, tokens and gates follow [`split_join_parts`].
///
/// No `unsafe` is needed: every atom belongs to one component and units are
/// ordered by component, so a component's atoms are one run of `slots` and
/// the halves are disjoint borrows even where components interleave in
/// address order.
#[allow(clippy::too_many_arguments)]
pub(super) fn split_join_components<T: Send + Sync>(
    units: &[ScatterUnit],
    component_slots: &[u32],
    slots: &mut [&mut [T]],
    slot_base: usize,
    fused_indices: &mut [usize],
    max_fused_rank: usize,
    threads: usize,
    exec: Exec<'_>,
    gate: Option<&LeafGate>,
    gate_bound: usize,
    leaf: &(impl Fn(
        &[ScatterUnit],
        &mut [&mut [T]],
        usize,
        &mut [usize],
        Exec<'_>,
        &LeafGate,
    ) -> Result<(), OperationError>
          + Sync),
) -> Result<(), OperationError> {
    let parts = units.len().saturating_sub(1);
    if parts == 0 {
        return Ok(());
    }
    if threads <= 1 || parts == 1 {
        let own_gate;
        let gate = match gate {
            Some(gate) => gate,
            None => {
                own_gate = LeafGate::new(gate_bound);
                &own_gate
            }
        };
        let exec = if parts == 1 {
            granted_width(exec, units[0].width)
        } else {
            exec
        };
        return leaf(
            &units[..parts],
            slots,
            slot_base,
            &mut fused_indices[..max_fused_rank],
            exec,
            gate,
        );
    }

    let middle = balanced_boundary(parts, threads, |part| units[part].weight_before);
    let boundary = component_slots[units[middle].component as usize] as usize;
    let (left_slots, right_slots) = slots.split_at_mut(boundary - slot_base);
    let left_threads = threads / 2;
    let right_threads = threads - left_threads;
    let (left_indices, mut right_indices) =
        split_lane_indices(fused_indices, left_threads, threads, max_fused_rank);
    let (left, right) = replay_join(
        exec,
        Some((left_threads, threads)),
        |exec| {
            split_join_components(
                &units[..=middle],
                component_slots,
                left_slots,
                slot_base,
                left_indices,
                max_fused_rank,
                left_threads,
                exec,
                None,
                gate_bound,
                leaf,
            )
        },
        |exec| {
            split_join_components(
                &units[middle..],
                component_slots,
                right_slots,
                boundary,
                right_indices.get(),
                max_fused_rank,
                right_threads,
                exec,
                None,
                gate_bound,
                leaf,
            )
        },
    );
    left?;
    right
}

/// The `(dst_layout, src_layout, coefficient)` of the Single block a
/// parallel schedule entry names.
#[inline]
pub(super) fn scheduled_single(
    blocks: &[TreeTransformBlock],
    item: u32,
) -> Result<(usize, usize, usize), OperationError> {
    match blocks.get(item as usize) {
        Some(&TreeTransformBlock::Single {
            dst_layout,
            src_layout,
            coefficient,
        }) => Ok((dst_layout, src_layout, coefficient)),
        _ => Err(OperationError::InvalidArgument {
            message: "tree transform single schedule names a non-Single block",
        }),
    }
}

/// Sizes the chunk scratch once for the largest chunk of `jobs`, so the
/// chunks [`replay_multi_chunk`] then visits never grow it.
///
/// Why not let each chunk grow its scratch: chunks are visited in GEMM-run
/// order, not by size, so growth reallocates once per larger chunk and a warm
/// call's allocation count would scale with the block count.
pub(super) fn reserve_multi_chunk_scratch<D: Copy + Zero>(
    workspace: &mut TreeTransformWorkspace<D>,
    jobs: &[DenseGemmBatchJob],
    chunk_size: usize,
) -> Result<(), OperationError> {
    let span = |start: usize, end: Option<usize>| {
        end.and_then(|end| end.checked_sub(start))
            .ok_or(OperationError::ElementCountOverflow)
    };
    let (mut source_len, mut destination_len) = (0usize, 0usize);
    for chunk in jobs.chunks(chunk_size) {
        let (first, last) = (&chunk[0], &chunk[chunk.len() - 1]);
        let source_end = last
            .rows
            .checked_mul(last.contracted)
            .and_then(|len| last.lhs_offset.checked_add(len));
        let destination_end = last
            .rows
            .checked_mul(last.cols)
            .and_then(|len| last.dst_offset.checked_add(len));
        source_len = source_len.max(span(first.lhs_offset, source_end)?);
        destination_len = destination_len.max(span(first.dst_offset, destination_end)?);
    }
    workspace.packed.source_mut().reserve_overwrite(source_len);
    workspace.prepare_packed_destination(destination_len, D::zero());
    let chunk_len = chunk_size.min(jobs.len());
    workspace.chunk_jobs.clear();
    workspace.chunk_jobs.reserve(chunk_len);
    workspace.chunk_runs.clear();
    workspace.chunk_runs.reserve(chunk_len);
    workspace.chunk_scatter_groups.clear();
    workspace.chunk_scatter_groups.reserve(chunk_len);
    Ok(())
}

/// One concurrency-bounded chunk of Multi jobs up to and including its grouped
/// GEMM: rebases the jobs onto chunk scratch, packs the chunk's source columns
/// (in parallel when `threads > 1`), runs the GEMM into
/// `workspace.packed.destination()`, and lists the chunk's non-empty scatter
/// groups in `workspace.chunk_scatter_groups`. With `form_units` (a scatter
/// split over components) it orders them by ownership component, one
/// `workspace.chunk_units` entry per component.
///
/// Shared by the initialised parallel replay and the owned uninitialised
/// writer, which differ only in how they scatter the packed destination.
/// Returns the chunk's packed destination base offset.
#[allow(clippy::too_many_arguments)]
pub(super) fn replay_multi_chunk<E, D, C>(
    engine: &mut dyn Engine,
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    task: TreeTransformTaskView<'_, C>,
    schedule: &TreeTransformParallelSchedule,
    first_group: usize,
    chunk: &[DenseGemmBatchJob],
    pack_cursor: &mut usize,
    src_data: &[D],
    threads: usize,
    form_units: bool,
    mut profile: Option<&mut TreeTransformReplayProfile>,
) -> Result<usize, OperationError>
where
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    let layouts = task.layouts();
    let max_fused_rank = layouts.max_fused_rank();
    let storage_conjugate = task.storage_conjugate();
    let source_start = chunk[0].lhs_offset;
    let destination_start = chunk[0].dst_offset;
    let source_end = chunk
        .last()
        .and_then(|job| {
            job.rows
                .checked_mul(job.contracted)
                .and_then(|len| job.lhs_offset.checked_add(len))
        })
        .ok_or_else(|| OperationError::ElementCountOverflow)?;
    let destination_end = chunk
        .last()
        .and_then(|job| {
            job.rows
                .checked_mul(job.cols)
                .and_then(|len| job.dst_offset.checked_add(len))
        })
        .ok_or_else(|| OperationError::ElementCountOverflow)?;

    workspace.chunk_jobs.clear();
    workspace
        .chunk_jobs
        .extend(chunk.iter().map(|job| DenseGemmBatchJob {
            dst_offset: job.dst_offset - destination_start,
            lhs_offset: job.lhs_offset - source_start,
            ..*job
        }));
    strided_batch_runs_into(&workspace.chunk_jobs, &mut workspace.chunk_runs);

    let start = profile.as_ref().map(|_| std::time::Instant::now());
    workspace.prepare_packed_destination(destination_end - destination_start, D::zero());
    if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
        profile.multi_workspace_prepare += start.elapsed();
    }

    let pack_start = *pack_cursor;
    while *pack_cursor < schedule.pack_columns.len()
        && schedule.pack_columns[*pack_cursor].packed_offset < source_end
    {
        *pack_cursor += 1;
    }
    let start = profile.as_ref().map(|_| std::time::Instant::now());
    let HostTreeTransformWorkspace {
        packed,
        fused_indices,
        ..
    } = &mut *workspace;
    packed
        .source_mut()
        .overwrite_with_columns(source_end - source_start, |writer| {
            replay_region(engine, |exec| {
                replay_pack_columns(
                    fused_indices.as_mut_slice(),
                    max_fused_rank,
                    layouts,
                    &schedule.pack_columns[pack_start..*pack_cursor],
                    writer,
                    source_start,
                    src_data,
                    storage_conjugate,
                    threads,
                    exec,
                )
            })?
        })?;
    if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
        profile.packed_columns += chunk.iter().map(|job| job.contracted).sum::<usize>();
        profile.multi_pack += start.elapsed();
    }

    let start = profile.as_ref().map(|_| std::time::Instant::now());
    let (source, destination) = workspace.packed.source_and_destination_mut();
    recoupling_gemm_batch(
        engine,
        dense,
        destination.as_mut_slice(),
        source.as_slice(),
        &workspace.coefficient_scratch,
        &workspace.chunk_jobs,
        &workspace.chunk_runs,
    )?;
    if let (Some(profile), Some(start)) = (profile, start) {
        let elapsed = start.elapsed();
        profile.multi_dense_matmul_call += elapsed;
        profile.multi_matmul_total += elapsed;
    }

    workspace.chunk_scatter_groups.clear();
    workspace.chunk_scatter_groups.extend(
        (first_group..first_group + chunk.len())
            .filter(|&group| !schedule.scatter_groups[group].columns.is_empty()),
    );
    if !form_units {
        return Ok(destination_start);
    }
    // Why not sort/copy every scatter descriptor: only the at-most-T group
    // indices need component order to form the chunk's scatter units.
    workspace
        .chunk_scatter_groups
        .sort_unstable_by_key(|&group| (schedule.scatter_groups[group].component, group));
    workspace.chunk_units.clear();
    let mut weight = 0usize;
    for (position, &group) in workspace.chunk_scatter_groups.iter().enumerate() {
        let replay = &schedule.scatter_groups[group];
        let group_weight = chunk[group - first_group]
            .rows
            .saturating_mul(replay.columns.len());
        match workspace.chunk_units.last_mut() {
            Some(unit) if unit.component == replay.component => {
                unit.groups.1 = position + 1;
                unit.weight = unit.weight.saturating_add(group_weight);
                unit.widest = unit.widest.max(replay.widest);
            }
            _ => workspace.chunk_units.push(ScatterUnit {
                groups: (position, position + 1),
                component: replay.component,
                weight: group_weight,
                weight_before: weight,
                widest: replay.widest,
                width: 1,
            }),
        }
        weight = weight.saturating_add(group_weight);
    }
    let end = workspace.chunk_scatter_groups.len();
    workspace.chunk_units.push(ScatterUnit {
        groups: (end, end),
        component: u32::MAX,
        weight: 0,
        weight_before: weight,
        widest: 0,
        width: 1,
    });
    Ok(destination_start)
}

/// Threaded variant of [`tree_transform_blocks_with_batched_recoupling`]
/// (TensorKit `_add_abelian_kernel_threaded!` / `_add_general_kernel_threaded!`
/// precedent, indexmanipulations.jl:520-738):
///
/// - Singles and Multi scatter split over ownership components, and Multi
///   blocks replay in concurrency-bounded chunks whose pack and scatter phases
///   each take the engine token once. Work items are independent because the
///   compile step rejects duplicate destination blocks
///   (`OperationError::DuplicateTransformDestination`) and pack columns are
///   disjoint scratch ranges by construction; the workspace forbids `unsafe`,
///   so disjointness is realized structurally with `split_at_mut` at compiled
///   atom boundaries instead of TensorKit-style shared writes. Overlapping
///   destination intervals (canonical coupled-sector blocks interleave) merge
///   into one atom, and owners sharing an atom form one component, which a
///   lane replays whole; a phase with one component stays serial. An
///   overwrite first runs its wide components (whose largest planned block
///   would use every worker of the whole token) alone on that token, then
///   splits the rest; a chunk whose moved elements Strided would give one
///   worker scatters inline.
///   Components are balanced by the destination elements they move.
/// - Each chunk is one serial grouped GEMM call between its pack and scatter
///   phases. The dense executor owns its own parallelism, so no nesting arises.
///
/// Parallel copy scheduling forks the operation's engine token recursively
/// (`replay_join`), capped by the configured worker count. Replay descriptors and safe split
/// boundaries are compiled into the structure. Each block keeps its worker's
/// token share, which an overwrite may spend on the block's compiled plan.
///
/// Per-task traversal indices are disjoint slices of one workspace arena, split
/// by the same worker budget as the data. Kernel-adapter clones therefore carry
/// configuration only on compiled replay; numerical and chunk-descriptor
/// storage also stays in the reused workspace.
///
/// Profiling attribution is phase-level; per-item clocks across workers would
/// measure contention, not work.
#[allow(clippy::too_many_arguments)]
pub(super) fn tree_transform_blocks_with_batched_recoupling_parallel<A, E, D, C>(
    engine: &mut dyn Engine,
    kernels: &mut A,
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    task: TreeTransformTaskView<'_, C>,
    schedule: &TreeTransformParallelSchedule,
    structure_identity: &Arc<()>,
    dst_data: &mut [D],
    src_data: &[D],
    alpha: DestinationAlpha<'_, D, C>,
    mode: DestinationMode<D>,
    threads: usize,
    mut profile: Option<&mut TreeTransformReplayProfile>,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D> + Clone + Send + Sync,
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    let layouts = task.layouts();
    // One gate for every block list this call visits on the calling
    // thread; forked leaves build their own.
    let gate = LeafGate::new(layouts.max_planned_elements());
    // The width table belongs to this call's engine and token.
    workspace.width_levels.clear();
    let recoupling_plan = task.recoupling_plan();
    let max_fused_rank = layouts.max_fused_rank();
    workspace.prepare_fused_indices(threads, max_fused_rank)?;

    let single_count = schedule.single_block_count;
    let multi_count = recoupling_plan.jobs().len();
    let start = profile.as_ref().map(|_| std::time::Instant::now());
    let converted = ensure_recoupling_coefficients(workspace, task, structure_identity)?;
    if converted {
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            profile.multi_coefficient_prepare += start.elapsed();
        }
    }
    if let Some(profile) = profile.as_deref_mut() {
        profile.single_blocks += single_count;
        profile.multi_blocks += multi_count;
    }

    let storage_conjugate = task.storage_conjugate();

    // Single destinations are independent of every Multi destination.
    {
        let start = profile.as_ref().map(|_| std::time::Instant::now());
        if schedule.single_component_count() > 1 {
            let blocks = task.blocks();
            let coefficients = task.single_coefficients();
            let kernels = &*kernels;
            let HostTreeTransformWorkspace {
                fused_indices,
                narrow_atoms,
                width_levels,
                ..
            } = &mut *workspace;
            replay_region(engine, |exec| {
                replay_single_phase(
                    schedule,
                    narrow_atoms,
                    width_levels,
                    dst_data,
                    fused_indices.as_mut_slice(),
                    max_fused_rank,
                    threads,
                    exec,
                    &gate,
                    layouts.max_planned_elements(),
                    mode.overwrites(),
                    &|items, dst, dst_start, fused_index, mut exec, gate| {
                        let mut kernels = kernels.clone();
                        let mut zero_strides = Vec::new();
                        for &item in items {
                            let (dst_layout, src_layout, coefficient) =
                                scheduled_single(blocks, item)?;
                            tree_transform_single_on_token(
                                exec.reborrow(),
                                gate,
                                &mut kernels,
                                &mut zero_strides,
                                Some(&mut *fused_index),
                                layouts,
                                dst_layout,
                                src_layout,
                                coefficients[coefficient],
                                storage_conjugate,
                                dst,
                                dst_start,
                                src_data,
                                alpha.at(layouts.entry(dst_layout).offset),
                                mode,
                            )?;
                        }
                        Ok(())
                    },
                )
            })??;
        } else {
            let mut zero_strides = Vec::new();
            crate::host_pool::with_exec(engine, |mut exec| {
                for &item in &schedule.singles {
                    let (dst_layout, src_layout, coefficient) =
                        scheduled_single(task.blocks(), item)?;
                    tree_transform_single_on_token(
                        exec.reborrow(),
                        &gate,
                        kernels,
                        &mut zero_strides,
                        Some(workspace.fused_indices.as_mut_slice()),
                        layouts,
                        dst_layout,
                        src_layout,
                        task.single_coefficients()[coefficient],
                        storage_conjugate,
                        dst_data,
                        0,
                        src_data,
                        alpha.at(layouts.entry(dst_layout).offset),
                        mode,
                    )?;
                }
                Ok(())
            })?;
        }

        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            let elapsed = start.elapsed();
            profile.single_total += elapsed;
            profile.strided_kernel += elapsed;
        }
    }

    // Why not retain one pack arena for the whole plan: TensorKit owns scratch
    // per concurrently executing fusion block. The existing replay-thread
    // budget is also the natural bound for a grouped dense submission.
    let chunk_size = threads.max(1);
    reserve_multi_chunk_scratch(workspace, recoupling_plan.jobs(), chunk_size)?;
    let mut target = ScatterTarget::new(
        workspace,
        schedule,
        threads,
        recoupling_plan.jobs().len(),
        dst_data,
    )?;
    let mut pack_cursor = 0;
    // Why a closure: an error still returns the atom-list capacity to
    // the workspace before it propagates.
    let mut replay_chunks = || -> Result<(), OperationError> {
        for (chunk_index, chunk) in recoupling_plan.jobs().chunks(chunk_size).enumerate() {
            let scattered_column_count = chunk.iter().map(|job| job.cols).sum::<usize>();
            let destination_start = replay_multi_chunk(
                engine,
                dense,
                workspace,
                task,
                schedule,
                chunk_index * chunk_size,
                chunk,
                &mut pack_cursor,
                src_data,
                threads,
                matches!(target, ScatterTarget::Atoms(_)),
                profile.as_deref_mut(),
            )?;
            let HostTreeTransformWorkspace {
                packed,
                fused_indices,
                chunk_scatter_groups,
                chunk_units,
                width_levels,
                ..
            } = &mut *workspace;
            let packed_destination = packed.destination().as_slice();
            let scatter_columns = &schedule.scatter_columns;
            let scatter_groups = &schedule.scatter_groups;

            let start = profile.as_ref().map(|_| std::time::Instant::now());
            match &mut target {
                ScatterTarget::Atoms(slots) => {
                    let kernels = &*kernels;
                    let leaf = |units: &[ScatterUnit],
                                slots: &mut [&mut [D]],
                                slot_base: usize,
                                fused_index: &mut [usize],
                                mut exec: Exec<'_>,
                                gate: &LeafGate| {
                        let mut kernels = kernels.clone();
                        let mut zero_strides = Vec::new();
                        for unit in units {
                            for &group in unit.groups(chunk_scatter_groups) {
                                for item in &scatter_columns[scatter_groups[group].columns.clone()]
                                {
                                    let atom = &mut *slots[item.slot as usize - slot_base];
                                    scatter_column_on_token(
                                        exec.reborrow(),
                                        gate,
                                        &mut kernels,
                                        &mut zero_strides,
                                        Some(&mut *fused_index),
                                        layouts,
                                        item.dst_layout,
                                        packed_destination,
                                        item.packed_offset - destination_start,
                                        atom,
                                        atom_start(layouts, item)?,
                                        alpha.at(layouts.entry(item.dst_layout).offset),
                                        mode,
                                    )?;
                                }
                            }
                        }
                        Ok(())
                    };
                    replay_region(engine, |exec| {
                        scatter_chunk_components(
                            chunk_units,
                            width_levels,
                            &schedule.component_slots,
                            slots,
                            fused_indices.as_mut_slice(),
                            max_fused_rank,
                            threads,
                            exec,
                            &gate,
                            layouts.max_planned_elements(),
                            mode.overwrites(),
                            &leaf,
                        )
                    })??;
                }
                ScatterTarget::Whole(dst_data) => {
                    let mut zero_strides = Vec::new();
                    crate::host_pool::with_exec(engine, |mut exec| {
                        for &group in chunk_scatter_groups.iter() {
                            for item in &scatter_columns[scatter_groups[group].columns.clone()] {
                                scatter_column_on_token(
                                    exec.reborrow(),
                                    &gate,
                                    kernels,
                                    &mut zero_strides,
                                    Some(fused_indices.as_mut_slice()),
                                    layouts,
                                    item.dst_layout,
                                    packed_destination,
                                    item.packed_offset - destination_start,
                                    dst_data,
                                    0,
                                    alpha.at(layouts.entry(item.dst_layout).offset),
                                    mode,
                                )?;
                            }
                        }
                        Ok(())
                    })?;
                }
            }
            if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
                profile.scattered_columns += scattered_column_count;
                profile.multi_scatter += start.elapsed();
            }
        }
        Ok(())
    };
    let replayed = replay_chunks();
    target.finish(workspace);
    replayed?;
    debug_assert_eq!(pack_cursor, schedule.pack_columns.len());
    Ok(())
}

#[cfg(test)]
mod bounded_workspace_tests {
    use super::*;
    use crate::{StridedHostKernelAdapter, TreeTransformBlockSpec};
    use num_complex::Complex64;

    fn many_group_fixture() -> (Arc<BlockStructure>, TreeTransformStructure<f64>, Vec<f64>) {
        const GROUPS: usize = 32;
        let mut shapes = Vec::with_capacity(2 * GROUPS);
        let mut specs = Vec::with_capacity(GROUPS);
        let mut source = Vec::new();
        for group in 0..GROUPS {
            let elements = group % 8 + 1;
            shapes.push(vec![elements]);
            shapes.push(vec![elements]);
            let first = 2 * group;
            specs.push(TreeTransformBlockSpec::multi(
                vec![first, first + 1],
                vec![first, first + 1],
                vec![1.0, 0.0, 0.0, 1.0],
            ));
            let base = source.len();
            source.extend((0..2 * elements).map(|index| (base + index + 1) as f64));
        }
        let structure = Arc::new(BlockStructure::packed_column_major(1, shapes).unwrap());
        let transform =
            TreeTransformStructure::compile_structures(&structure, &structure, &specs).unwrap();
        (structure, transform, source)
    }

    #[test]
    fn fused_index_arena_is_grow_only_and_reports_overflow() {
        // What: execution scratch retains its largest checked worker/rank arena
        // and reports impossible products without allocating or panicking.
        // Up to the inline rank it holds the calling lane only, at any width:
        // forked lanes keep their own index, so a wide eager call stays inline.
        let mut workspace = TreeTransformWorkspace::<f64>::default();
        workspace.prepare_fused_indices(4, 8).unwrap();
        assert_eq!(workspace.fused_indices.len(), 8);
        assert!(!workspace.fused_indices.spilled());
        workspace.prepare_fused_indices(4, 9).unwrap();
        assert_eq!(workspace.fused_indices.len(), 36);
        workspace.prepare_fused_indices(1, 2).unwrap();
        assert_eq!(workspace.fused_indices.len(), 36);
        assert_eq!(
            workspace.prepare_fused_indices(usize::MAX, 9),
            Err(OperationError::ElementCountOverflow)
        );
        assert_eq!(
            workspace.prepare_fused_indices(usize::MAX / 8, 10),
            Err(OperationError::ElementCountOverflow)
        );
    }

    #[test]
    fn many_groups_bound_serial_and_parallel_pack_scratch_by_concurrency() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let (structure, transform, source) = many_group_fixture();
        let total_source = transform.recoupling_plan().source_len();
        let total_destination = transform.recoupling_plan().destination_len();

        for threads in [1, 3] {
            let mut destination = vec![0.0; source.len()];
            let mut workspace = TreeTransformWorkspace::default();
            tree_transform_structure_with_structural_recoupling_raw(
                engine,
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut workspace,
                &transform,
                &structure,
                &structure,
                &mut destination,
                &source,
                1.0,
                0.0,
                threads,
            )
            .unwrap();

            // What: 32 independent groups replay exactly, while retained pack
            // capacity follows at most `threads` largest groups rather than all.
            assert_eq!(destination, source);
            let (source_capacity, destination_capacity) = workspace.packed_capacities();
            let chunk_source_bound = (threads * 2 * 8).next_power_of_two();
            let chunk_destination_bound = (threads * 2 * 8).next_power_of_two();
            assert!(source_capacity <= chunk_source_bound);
            assert!(destination_capacity <= chunk_destination_bound);
            assert!(source_capacity < total_source);
            assert!(destination_capacity < total_destination);
        }
    }

    #[test]
    fn bounded_group_replay_preserves_complex_alpha_beta() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let (structure, transform, source) = many_group_fixture();
        let source = source
            .into_iter()
            .map(|value| Complex64::new(value, -value))
            .collect::<Vec<_>>();
        let initial = Complex64::new(2.0, -3.0);
        let alpha = Complex64::new(0.5, 0.25);
        let beta = Complex64::new(-0.5, 0.75);
        let mut destination = vec![initial; source.len()];

        tree_transform_structure_with_structural_recoupling_raw(
            engine,
            &mut StridedHostKernelAdapter::default(),
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            &transform,
            &structure,
            &structure,
            &mut destination,
            &source,
            alpha,
            beta,
            3,
        )
        .unwrap();

        // What: chunk boundaries do not change complex recoupling or axpby.
        let expected = source
            .iter()
            .map(|&value| alpha * value + beta * initial)
            .collect::<Vec<_>>();
        assert_eq!(destination, expected);
    }
}
