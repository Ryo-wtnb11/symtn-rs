use super::*;

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

/// One replay fork: both halves run as jobs of `exec`. The split itself (and
/// so every per-worker scratch slice and the left-before-right error order)
/// is fixed by the replay's `threads`, not by where the token places a job.
pub(super) fn replay_join<A, B, RA, RB>(exec: Exec<'_>, left: A, right: B) -> (RA, RB)
where
    A: FnOnce(Exec<'_>) -> RA + Send + Sync,
    B: FnOnce(Exec<'_>) -> RB + Send + Sync,
    RA: Send + Sync,
    RB: Send + Sync,
{
    pool_join(exec, left, right)
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

/// Recursive `replay_join` over the Single ownership atoms `atoms` (address
/// ordered, ending with a sentinel): `dst` is the destination from absolute
/// offset `dst_start`, `items` the whole Single schedule, which each atom
/// indexes. A split lands on an atom start chosen by [`balanced_boundary`]
/// over the atoms' destination elements, with `threads / 2` lanes to the
/// left; `leaf` writes its atoms' items into its part of `dst` with one
/// worker's traversal scratch and that worker's share of `exec`. An unsplit
/// list keeps the whole of `exec` and the caller's `gate`; forked leaves
/// build their own gate bounded by `gate_bound`.
///
/// No `unsafe` is needed: atoms are maximal runs of overlapping item
/// intervals, so every item lies inside one atom and `split_at_mut` at an
/// atom start leaves each item wholly in one half. The same recursion serves
/// the initialised replay (`T = D`) and the owned writer
/// (`T = MaybeUninit<D>`), whose publication waits for every join.
#[allow(clippy::too_many_arguments)]
pub(super) fn split_join_atoms<T: Send + Sync>(
    items: &[u32],
    atoms: &[TreeTransformSingleAtom],
    dst: &mut [T],
    dst_start: isize,
    fused_indices: &mut [usize],
    max_fused_rank: usize,
    threads: usize,
    exec: Exec<'_>,
    gate: Option<&LeafGate>,
    gate_bound: usize,
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
    let parts = atoms.len().saturating_sub(1);
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
        return leaf(
            &items[atoms[0].item as usize..atoms[parts].item as usize],
            dst,
            dst_start,
            &mut fused_indices[..max_fused_rank],
            exec,
            gate,
        );
    }

    let middle = balanced_boundary(parts, threads, |part| atoms[part].weight_before);
    let boundary = atoms[middle].lo;
    let split =
        usize::try_from(boundary - dst_start).map_err(|_| OperationError::ElementCountOverflow)?;
    let (left_data, right_data) = dst.split_at_mut(split);
    let left_threads = threads / 2;
    let right_threads = threads - left_threads;
    let (left_indices, mut right_indices) =
        split_lane_indices(fused_indices, left_threads, threads, max_fused_rank);
    let (left, right) = replay_join(
        exec,
        |exec| {
            split_join_atoms(
                items,
                &atoms[..=middle],
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
            split_join_atoms(
                items,
                &atoms[middle..],
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

/// One ownership component's share of a Multi chunk's scatter: the groups
/// `chunk_scatter_groups[group..]` up to the next unit's `group`. A chunk's
/// unit list is ordered by component and ends with a sentinel.
#[derive(Clone, Copy, Debug)]
pub(super) struct ScatterUnit {
    group: usize,
    component: u32,
    /// Destination elements the chunk's earlier units write.
    weight_before: usize,
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
    pub(super) fn new<D>(
        workspace: &mut TreeTransformWorkspace<D>,
        schedule: &TreeTransformParallelSchedule,
        threads: usize,
        dst: &'a mut [T],
    ) -> Result<Self, OperationError> {
        if threads <= 1 || schedule.scatter_component_count() <= 1 {
            return Ok(Self::Whole(dst));
        }
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

/// Recursive `replay_join` over one chunk's scatter `units` (ending with a
/// sentinel): `slots` are the atoms of components `component_slots`-indexed
/// from slot `slot_base`. A split lands between two units at the boundary
/// [`balanced_boundary`] picks over their destination elements; the slots
/// split at the right unit's first atom, so each side holds exactly the atoms
/// of its own components. `leaf` scatters its units' groups (`groups`
/// entries) into their atoms; lanes, tokens and gates follow
/// [`split_join_atoms`].
///
/// No `unsafe` is needed: every atom belongs to one component and units are
/// ordered by component, so a component's atoms are one run of `slots` and
/// the halves are disjoint borrows even where components interleave in
/// address order.
#[allow(clippy::too_many_arguments)]
pub(super) fn split_join_components<T: Send + Sync>(
    units: &[ScatterUnit],
    groups: &[usize],
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
        &[usize],
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
        return leaf(
            &groups[units[0].group..units[parts].group],
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
        |exec| {
            split_join_components(
                &units[..=middle],
                groups,
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
                groups,
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
    workspace.chunk_units.clear();
    if chunk_len != 0 {
        // The chunk's units and their sentinel.
        workspace.chunk_units.reserve(chunk_len + 1);
    }
    Ok(())
}

/// One concurrency-bounded chunk of Multi jobs up to and including its grouped
/// GEMM: rebases the jobs onto chunk scratch, packs the chunk's source columns
/// (in parallel when `threads > 1`), runs the GEMM into
/// `workspace.packed.destination()`, and orders the chunk's non-empty scatter
/// groups by ownership component in `workspace.chunk_scatter_groups`, one
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
    // Why not sort/copy every scatter descriptor: only the at-most-T group
    // indices need component order to form the chunk's scatter units.
    workspace
        .chunk_scatter_groups
        .sort_unstable_by_key(|&group| (schedule.scatter_groups[group].component, group));
    workspace.chunk_units.clear();
    let mut weight = 0usize;
    for (position, &group) in workspace.chunk_scatter_groups.iter().enumerate() {
        let replay = &schedule.scatter_groups[group];
        if workspace
            .chunk_units
            .last()
            .is_none_or(|unit| unit.component != replay.component)
        {
            workspace.chunk_units.push(ScatterUnit {
                group: position,
                component: replay.component,
                weight_before: weight,
            });
        }
        weight = weight.saturating_add(
            chunk[group - first_group]
                .rows
                .saturating_mul(replay.columns.len()),
        );
    }
    workspace.chunk_units.push(ScatterUnit {
        group: workspace.chunk_scatter_groups.len(),
        component: u32::MAX,
        weight_before: weight,
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
///   lane replays whole; a phase with one component stays serial.
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
            replay_region(engine, |exec| {
                split_join_atoms(
                    &schedule.singles,
                    &schedule.single_atoms,
                    dst_data,
                    0,
                    workspace.fused_indices.as_mut_slice(),
                    max_fused_rank,
                    threads,
                    exec,
                    Some(&gate),
                    layouts.max_planned_elements(),
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
    let mut target = ScatterTarget::new(workspace, schedule, threads, dst_data)?;
    let mut pack_cursor = 0;
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
            profile.as_deref_mut(),
        )?;
        let HostTreeTransformWorkspace {
            packed,
            fused_indices,
            chunk_scatter_groups,
            chunk_units,
            ..
        } = &mut *workspace;
        let packed_destination = packed.destination().as_slice();
        let scatter_columns = &schedule.scatter_columns;
        let scatter_groups = &schedule.scatter_groups;

        let start = profile.as_ref().map(|_| std::time::Instant::now());
        match &mut target {
            ScatterTarget::Atoms(slots) => {
                let kernels = &*kernels;
                replay_region(engine, |exec| {
                    split_join_components(
                        chunk_units,
                        chunk_scatter_groups,
                        &schedule.component_slots,
                        slots,
                        0,
                        fused_indices.as_mut_slice(),
                        max_fused_rank,
                        threads,
                        exec,
                        Some(&gate),
                        layouts.max_planned_elements(),
                        &|groups, slots, slot_base, fused_index, mut exec, gate| {
                            let mut kernels = kernels.clone();
                            let mut zero_strides = Vec::new();
                            for &group in groups {
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
                            Ok(())
                        },
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
    target.finish(workspace);
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
