use super::*;

fn checked_len<D>(len: usize) -> Result<isize, OperationError> {
    core::alloc::Layout::array::<D>(len).map_err(|_| OperationError::ElementCountOverflow)?;
    offset_to_isize(len)
}

fn checked_view(
    offset: isize,
    shape: &[usize],
    strides: &[isize],
    member_len: usize,
) -> Result<(), OperationError> {
    if shape.len() != strides.len() {
        return Err(OperationError::InvalidArgument {
            message: "member transform layout rank mismatch",
        });
    }
    if shape.contains(&0) {
        return Ok(());
    }
    let (mut lo, mut hi) = (offset, offset);
    for (&extent, &stride) in shape.iter().zip(strides) {
        let span = isize::try_from(extent - 1)
            .map_err(|_| OperationError::ElementCountOverflow)?
            .checked_mul(stride)
            .ok_or(OperationError::ElementCountOverflow)?;
        if span < 0 {
            lo = lo
                .checked_add(span)
                .ok_or(OperationError::ElementCountOverflow)?;
        } else {
            hi = hi
                .checked_add(span)
                .ok_or(OperationError::ElementCountOverflow)?;
        }
    }
    if lo < 0 || hi >= checked_len::<u8>(member_len)? {
        return Err(OperationError::InvalidArgument {
            message: "member transform layout exceeds its member",
        });
    }
    Ok(())
}

fn checked_range(start: usize, len: usize, total: usize) -> Result<(), OperationError> {
    if start.checked_add(len).is_some_and(|end| end <= total) {
        Ok(())
    } else {
        Err(OperationError::InvalidArgument {
            message: "member transform packed job exceeds its buffer",
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn admit<D, C>(
    task: TreeTransformTaskView<'_, C>,
    dst_structure: &Arc<BlockStructure>,
    src_structure: &Arc<BlockStructure>,
    dst_len: usize,
    src_len: usize,
    members: usize,
    destinations: &mut Vec<(usize, usize)>,
) -> Result<(), OperationError>
where
    D: DenseRecouplingScalar,
    C: Copy,
{
    if members == 0 {
        return Err(OperationError::InvalidArgument {
            message: "member transform requires at least one member",
        });
    }
    offset_to_isize(members)?;
    task.validate_structures(dst_structure, src_structure)?;
    let dst_member = dst_structure.required_len()?;
    let src_member = src_structure.required_len()?;
    task.validate_lengths(dst_member, src_member)?;
    for (actual, per_member) in [(dst_len, dst_member), (src_len, src_member)] {
        let expected = per_member
            .checked_mul(members)
            .ok_or(OperationError::ElementCountOverflow)?;
        checked_len::<D>(expected)?;
        if actual != expected {
            return Err(OperationError::ElementCountMismatch { expected, actual });
        }
    }
    // Compilation already proves destination block injectivity. Exact
    // structure admission above carries that proof to every member; the
    // checked member stride keeps their physical ranges separate.
    let layouts = task.layouts();
    let plan = task.recoupling_plan();
    task.validate_workspace_requirements::<D>()?;
    checked_len::<D>(
        plan.source_len()
            .checked_mul(members)
            .ok_or(OperationError::ElementCountOverflow)?,
    )?;
    checked_len::<D>(
        plan.destination_len()
            .checked_mul(members)
            .ok_or(OperationError::ElementCountOverflow)?,
    )?;
    core::alloc::Layout::array::<DenseGemmBatchJob>(
        plan.jobs()
            .len()
            .checked_mul(members)
            .ok_or(OperationError::ElementCountOverflow)?,
    )
    .map_err(|_| OperationError::ElementCountOverflow)?;
    // Every view below is a compiled normalized role: it addresses exactly
    // the raw block's element set, so its minimum and maximum, and hence
    // these bounds, are the raw layout's.
    for &index in task.inactive_destination_layouts() {
        let (dims, strides) = layouts.inactive_role(index)?;
        checked_view(layouts.entry(index).offset, dims, strides, dst_member)?;
    }
    for block in task.blocks() {
        if let TreeTransformBlock::Single {
            dst_layout,
            src_layout,
            coefficient,
        } = *block
        {
            let role = layouts.role(dst_layout)?;
            checked_view(
                layouts.entry(dst_layout).offset,
                role.dims(),
                role.dst_strides(),
                dst_member,
            )?;
            checked_view(
                layouts.entry(src_layout).offset,
                role.dims(),
                role.src_strides(),
                src_member,
            )?;
            if coefficient >= task.single_coefficients().len() {
                return Err(OperationError::CoefficientCountMismatch {
                    expected: coefficient + 1,
                    actual: task.single_coefficients().len(),
                });
            }
        }
    }
    destinations.clear();
    destinations.reserve(plan.jobs().len());
    for (block_index, job) in plan.entries() {
        let &TreeTransformBlock::Multi {
            dst_layout_start,
            dst_count,
            src_layout_start,
            src_count,
            element_count,
            coefficient_start,
            ..
        } = recoupling_multi_block(task, block_index)?
        else {
            unreachable!()
        };
        if (job.rows, job.contracted, job.cols) != (element_count, src_count, dst_count) {
            return Err(OperationError::InvalidArgument {
                message: "member transform job geometry mismatch",
            });
        }
        let src_span = element_count
            .checked_mul(src_count)
            .ok_or(OperationError::ElementCountOverflow)?;
        let dst_span = element_count
            .checked_mul(dst_count)
            .ok_or(OperationError::ElementCountOverflow)?;
        let coefficient_span = src_count
            .checked_mul(dst_count)
            .ok_or(OperationError::ElementCountOverflow)?;
        checked_range(job.lhs_offset, src_span, plan.source_len())?;
        checked_range(job.dst_offset, dst_span, plan.destination_len())?;
        checked_range(job.rhs_offset, coefficient_span, plan.coefficient_len())?;
        checked_range(coefficient_start, coefficient_span, task.coefficient_len())?;
        destinations.push((job.dst_offset, job.dst_offset + dst_span));
        for column in 0..src_count {
            let entry = src_layout_start + column;
            let role = layouts.role(entry)?;
            checked_view(
                layouts.entry(entry).offset,
                role.dims(),
                role.src_strides(),
                src_member,
            )?;
            checked_view(0, role.dims(), role.dst_strides(), element_count)?;
            checked_range(
                job.lhs_offset + column * element_count,
                element_count,
                plan.source_len(),
            )?;
        }
        for column in 0..dst_count {
            let entry = dst_layout_start + column;
            let role = layouts.role(entry)?;
            checked_view(
                layouts.entry(entry).offset,
                role.dims(),
                role.dst_strides(),
                dst_member,
            )?;
            checked_view(0, role.dims(), role.src_strides(), element_count)?;
            checked_range(
                job.dst_offset + column * element_count,
                element_count,
                plan.destination_len(),
            )?;
        }
    }
    destinations.sort_unstable();
    if destinations.windows(2).any(|pair| pair[0].1 > pair[1].0) {
        return Err(OperationError::InvalidArgument {
            message: "member transform packed destinations overlap",
        });
    }
    Ok(())
}

/// Read-only admission for a member replay. It validates bounds and prepares
/// fallible coefficient conversion before the caller starts any stage.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn admit_tree_transform_members_overwrite_raw<D, C>(
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst_structure: &Arc<BlockStructure>,
    src_structure: &Arc<BlockStructure>,
    dst_len: usize,
    src_len: usize,
    members: usize,
) -> Result<(), OperationError>
where
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C>,
    C: Copy,
{
    let task = structure.task_view()?;
    admit::<D, C>(
        task,
        dst_structure,
        src_structure,
        dst_len,
        src_len,
        members,
        &mut workspace.member_ranges,
    )?;
    ensure_recoupling_coefficients(workspace, task, structure.identity_marker())?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn move_members<A, D>(
    exec: crate::host_pool::Exec<'_>,
    kernels: &mut A,
    shape: &mut Vec<usize>,
    dst_strides: &mut Vec<isize>,
    src_strides: &mut Vec<isize>,
    dst: &mut [D],
    src: &[D],
    local_shape: &[usize],
    local_dst_strides: &[isize],
    local_src_strides: &[isize],
    dst_member_stride: isize,
    src_member_stride: isize,
    dst_offset: isize,
    src_offset: isize,
    members: usize,
    conjugate: bool,
    scale: D,
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D>,
    D: Copy,
{
    shape.clear();
    shape.extend_from_slice(local_shape);
    shape.push(members);
    dst_strides.clear();
    dst_strides.extend_from_slice(local_dst_strides);
    dst_strides.push(dst_member_stride);
    src_strides.clear();
    src_strides.extend_from_slice(local_src_strides);
    src_strides.push(src_member_stride);
    kernels.copy_scale_strided_baked(
        exec,
        dst,
        src,
        shape,
        dst_strides,
        src_strides,
        dst_offset,
        src_offset,
        conjugate,
        scale,
        None,
        None,
    )
}

/// One worker's member-axis move scratch, kept by the workspace: the moved
/// shape, its two stride lists, and the zero source strides of a fill.
#[derive(Clone, Debug, Default)]
pub(super) struct MemberScratch {
    shape: Vec<usize>,
    dst_strides: Vec<isize>,
    src_strides: Vec<isize>,
    zero_strides: Vec<isize>,
}

impl MemberScratch {
    pub(super) fn retained_bytes(&self) -> usize {
        self.shape
            .capacity()
            .saturating_mul(core::mem::size_of::<usize>())
            .saturating_add(
                (self.dst_strides.capacity()
                    + self.src_strides.capacity()
                    + self.zero_strides.capacity())
                .saturating_mul(core::mem::size_of::<isize>()),
            )
    }
}

/// The read-only parts of a member batch's moves: the task, the strides
/// between members of every buffer, and the destination scale of every
/// Single and scatter column, resolved once per call.
#[derive(Clone, Copy)]
struct MemberMoves<'a, D, C> {
    parts: super::writers::ReplayParts<'a, C>,
    plan: &'a crate::transform_structure::TreeTransformRecouplingPlan,
    inactive: &'a [usize],
    src: &'a [D],
    /// The scale of each Single, in block order, then of each scatter
    /// column, in plan order.
    alpha: &'a [D],
    singles: usize,
    src_stride: isize,
    dst_stride: isize,
    packed_src_stride: isize,
    packed_dst_stride: isize,
}

impl<D, C> MemberMoves<'_, D, C>
where
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy,
{
    /// The moves before the grouped GEMM for `members` members from member
    /// `first`: inactive fills and Singles into `dst`, and the pack columns
    /// into `packed`; `dst` and `packed` start at member `first`.
    #[allow(clippy::too_many_arguments)]
    fn pack<A: HostKernelAdapter<D>>(
        &self,
        mut exec: Exec<'_>,
        kernels: &mut A,
        scratch: &mut MemberScratch,
        dst: &mut [D],
        packed: &mut [D],
        first: usize,
        members: usize,
    ) -> Result<(), OperationError> {
        let layouts = self.parts.layouts;
        let src_shift = offset_to_isize(first)?
            .checked_mul(self.src_stride)
            .ok_or(OperationError::ElementCountOverflow)?;
        let MemberScratch {
            shape,
            dst_strides,
            src_strides,
            zero_strides,
        } = scratch;
        for &index in self.inactive {
            let (dims, strides) = layouts.inactive_role(index)?;
            let zero = [D::zero()];
            zero_strides.clear();
            zero_strides.resize(dims.len(), 0);
            move_members(
                exec.reborrow(),
                kernels,
                shape,
                dst_strides,
                src_strides,
                dst,
                &zero,
                dims,
                strides,
                zero_strides,
                self.dst_stride,
                0,
                layouts.entry(index).offset,
                0,
                members,
                false,
                D::one(),
            )?;
        }
        let mut single = 0;
        for block in self.parts.blocks {
            let TreeTransformBlock::Single {
                dst_layout,
                src_layout,
                coefficient,
            } = *block
            else {
                continue;
            };
            let role = layouts.role(dst_layout)?;
            let dst_entry = layouts.entry(dst_layout);
            let src_entry = layouts.entry(src_layout);
            shape.clear();
            shape.extend_from_slice(role.dims());
            shape.push(members);
            dst_strides.clear();
            dst_strides.extend_from_slice(role.dst_strides());
            dst_strides.push(self.dst_stride);
            src_strides.clear();
            src_strides.extend_from_slice(role.src_strides());
            src_strides.push(self.src_stride);
            kernels.transform_strided_baked(
                exec.reborrow(),
                zero_strides,
                dst,
                self.src,
                shape,
                dst_strides,
                src_strides,
                dst_entry.offset,
                src_entry.offset + src_shift,
                self.parts.storage_conjugate,
                TransformScale::new(
                    self.alpha[single],
                    self.parts.single_coefficients[coefficient],
                ),
                None,
                None,
                None,
            )?;
            single += 1;
        }
        for (block_index, job) in self.plan.entries() {
            let &TreeTransformBlock::Multi {
                src_layout_start,
                src_count,
                element_count,
                ..
            } = multi_block(self.parts.blocks, block_index)?
            else {
                unreachable!()
            };
            for column in 0..src_count {
                let entry = src_layout_start + column;
                let role = layouts.role(entry)?;
                move_members(
                    exec.reborrow(),
                    kernels,
                    shape,
                    dst_strides,
                    src_strides,
                    packed,
                    self.src,
                    role.dims(),
                    role.dst_strides(),
                    role.src_strides(),
                    self.packed_src_stride,
                    self.src_stride,
                    offset_to_isize(job.lhs_offset + column * element_count)?,
                    layouts.entry(entry).offset + src_shift,
                    members,
                    self.parts.storage_conjugate,
                    D::one(),
                )?;
            }
        }
        Ok(())
    }

    /// The scatter after the grouped GEMM for `members` members from member
    /// `first`, from the whole packed destination into `dst`, which starts
    /// at member `first`.
    #[allow(clippy::too_many_arguments)]
    fn scatter<A: HostKernelAdapter<D>>(
        &self,
        mut exec: Exec<'_>,
        kernels: &mut A,
        scratch: &mut MemberScratch,
        dst: &mut [D],
        packed: &[D],
        first: usize,
        members: usize,
    ) -> Result<(), OperationError> {
        let layouts = self.parts.layouts;
        let packed_shift = offset_to_isize(first)?
            .checked_mul(self.packed_dst_stride)
            .ok_or(OperationError::ElementCountOverflow)?;
        let mut column_index = self.singles;
        for (block_index, job) in self.plan.entries() {
            let &TreeTransformBlock::Multi {
                dst_layout_start,
                dst_count,
                element_count,
                ..
            } = multi_block(self.parts.blocks, block_index)?
            else {
                unreachable!()
            };
            for column in 0..dst_count {
                let entry = dst_layout_start + column;
                let role = layouts.role(entry)?;
                move_members(
                    exec.reborrow(),
                    kernels,
                    &mut scratch.shape,
                    &mut scratch.dst_strides,
                    &mut scratch.src_strides,
                    dst,
                    packed,
                    role.dims(),
                    role.dst_strides(),
                    role.src_strides(),
                    self.dst_stride,
                    self.packed_dst_stride,
                    layouts.entry(entry).offset,
                    offset_to_isize(job.dst_offset + column * element_count)? + packed_shift,
                    members,
                    false,
                    self.alpha[column_index],
                )?;
                column_index += 1;
            }
        }
        Ok(())
    }
}

/// The Multi block `index` of `blocks`.
fn multi_block(
    blocks: &[TreeTransformBlock],
    index: usize,
) -> Result<&TreeTransformBlock, OperationError> {
    match blocks.get(index) {
        Some(block @ TreeTransformBlock::Multi { .. }) => Ok(block),
        _ => Err(OperationError::InvalidArgument {
            message: "tree transform recoupling plan references a single block",
        }),
    }
}

/// Resolves the destination scale of every Single (block order) and every
/// scatter column (plan order) into `out`, once per call: `θ_b` of the
/// role's within-member destination offset.
fn resolve_member_alpha<D, C>(
    task: TreeTransformTaskView<'_, C>,
    alpha: DestinationAlpha<'_, D, C>,
    out: &mut Vec<D>,
) -> Result<usize, OperationError>
where
    D: Copy + RecouplingCoefficientAction<C>,
    C: Copy,
{
    let layouts = task.layouts();
    out.clear();
    // Why one exact reserve: a fresh workspace would otherwise grow the
    // table by doubling, a few calls more per call as the roles grow.
    let roles = task
        .blocks()
        .iter()
        .map(|block| match *block {
            TreeTransformBlock::Single { .. } => 1,
            TreeTransformBlock::Multi { dst_count, .. } => dst_count,
        })
        .sum::<usize>();
    out.reserve_exact(roles);
    for block in task.blocks() {
        if let TreeTransformBlock::Single { dst_layout, .. } = *block {
            out.push(alpha.at(layouts.entry(dst_layout).offset));
        }
    }
    let singles = out.len();
    for (block_index, _) in task.recoupling_plan().entries() {
        let &TreeTransformBlock::Multi {
            dst_layout_start,
            dst_count,
            ..
        } = recoupling_multi_block(task, block_index)?
        else {
            unreachable!()
        };
        for column in 0..dst_count {
            out.push(alpha.at(layouts.entry(dst_layout_start + column).offset));
        }
    }
    Ok(singles)
}

/// Overwrite a fixed-stride Host batch with one completed tree transform.
/// Each member uses the exact structure passed at compilation; the workspace
/// owns only execution scratch and can be reused at a different batch size.
/// One member replays on the ordinary route. A larger batch refuses a
/// forced batch policy `dense` cannot honor, as the ordinary route does,
/// and is admitted whole (every member's ranges) before any write; then the
/// member axis is one more strided dimension of each move around one dense
/// submission of the member-expanded GEMM jobs, plan job outer and member
/// inner, stopping at the first failure. `destination_scales` folds `θ_b`
/// into the moves writing destination block `b` of every member, as in
/// [`tree_transform_structure_overwrite_with_structural_recoupling_raw`],
/// each scale resolved once per role for the whole batch.
///
/// The moves before and after the GEMM each run in one engine entry at the
/// width [`symtn_dense::lanes::flat_width`] admits over the members: under
/// `Auto` the engine's width when a copy of the whole batch destination
/// pays for it, under a forced policy its batch decision, and one under
/// `Sequential`. At width one every member moves on the caller's token;
/// above it the members split into at most that many contiguous ranges,
/// each moving its members as one member-axis move per role.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn tree_transform_members_overwrite_raw<A, E, D, C>(
    engine: &mut dyn Engine,
    kernels: &mut A,
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst_structure: &Arc<BlockStructure>,
    src_structure: &Arc<BlockStructure>,
    dst_data: &mut [D],
    src_data: &[D],
    members: usize,
    destination_scales: &[(usize, C)],
) -> Result<(), OperationError>
where
    A: HostKernelAdapter<D> + Clone + Send + Sync,
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    if members == 1 {
        // The ordinary entry runs its own route, refuses an unhonored forced
        // policy first and admits the single member before writing its
        // destination.
        return tree_transform_structure_overwrite_with_structural_recoupling_raw(
            engine,
            kernels,
            dense,
            workspace,
            structure,
            dst_structure,
            src_structure,
            dst_data,
            src_data,
            D::one(),
            destination_scales,
        );
    }
    refuse_unhonored_policy(engine, dense)?;
    let alpha = DestinationAlpha::new(D::one(), destination_scales)?;
    let task = structure.task_view()?;
    admit::<D, C>(
        task,
        dst_structure,
        src_structure,
        dst_data.len(),
        src_data.len(),
        members,
        &mut workspace.member_ranges,
    )?;
    ensure_recoupling_coefficients(workspace, task, structure.identity_marker())?;
    // Why the member axis, not the one-member replay per member: the call
    // then makes one dense submission for every member, as the provider's
    // batch entry expects, instead of one per member and group.
    let plan = task.recoupling_plan();
    let dst_len = dst_structure.required_len()?;
    let packed_src_len = plan.source_len() * members;
    let packed_dst_len = plan.destination_len() * members;
    workspace.prepare_packed_buffers(packed_src_len, packed_dst_len, D::zero());
    workspace.chunk_jobs.clear();
    for (_, job) in plan.entries() {
        for member in 0..members {
            workspace.chunk_jobs.push(DenseGemmBatchJob {
                dst_offset: member * plan.destination_len() + job.dst_offset,
                lhs_offset: member * plan.source_len() + job.lhs_offset,
                ..*job
            });
        }
    }
    strided_batch_runs_into(&workspace.chunk_jobs, &mut workspace.chunk_runs);
    let singles = resolve_member_alpha(task, alpha, &mut workspace.member_alpha)?;
    if workspace.member_scratch.is_empty() {
        workspace.member_scratch.push(MemberScratch::default());
    }
    let strides = [
        offset_to_isize(src_structure.required_len()?)?,
        offset_to_isize(dst_len)?,
        offset_to_isize(plan.source_len())?,
        offset_to_isize(plan.destination_len())?,
    ];
    let geometry = MemberMoves {
        parts: super::writers::ReplayParts::new(task),
        plan,
        inactive: task.inactive_destination_layouts(),
        src: src_data,
        alpha: &[],
        singles,
        src_stride: strides[0],
        dst_stride: strides[1],
        packed_src_stride: strides[2],
        packed_dst_stride: strides[3],
    };
    let HostTreeTransformWorkspace {
        packed,
        member_alpha,
        member_scratch,
        ..
    } = &mut *workspace;
    let lens = MemberLens {
        dst: dst_len,
        packed_src: plan.source_len(),
    };
    let threads = member_moves(
        engine,
        &MemberMoves {
            alpha: member_alpha,
            ..geometry
        },
        kernels,
        member_scratch,
        dst_data,
        packed.source_mut().as_mut_slice(),
        &[],
        lens,
        members,
        None,
    )?;
    if !workspace.chunk_jobs.is_empty() {
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
    }
    let HostTreeTransformWorkspace {
        packed,
        member_alpha,
        member_scratch,
        ..
    } = &mut *workspace;
    member_moves(
        engine,
        &MemberMoves {
            alpha: member_alpha,
            ..geometry
        },
        kernels,
        member_scratch,
        dst_data,
        &mut [],
        packed.destination().as_slice(),
        lens,
        members,
        Some(threads),
    )?;
    Ok(())
}

/// Elements per member of the destination and of the packed source.
#[derive(Clone, Copy)]
struct MemberLens {
    dst: usize,
    packed_src: usize,
}

/// One member phase in one engine entry: the pack phase when `packed_dst`
/// is empty and `threads` is `None` (which admits the width and returns
/// it), the scatter phase from `packed_dst` at that width otherwise. At
/// width one every member moves on the caller's token with the caller's
/// kernels; above it, the members split into `min(threads, members)`
/// contiguous ranges over a [`lanes::Split`] of `threads`.
#[allow(clippy::too_many_arguments)]
fn member_moves<A, D, C>(
    engine: &mut dyn Engine,
    moves: &MemberMoves<'_, D, C>,
    kernels: &mut A,
    scratch: &mut Vec<MemberScratch>,
    dst: &mut [D],
    packed_src: &mut [D],
    packed_dst: &[D],
    lens: MemberLens,
    members: usize,
    threads: Option<usize>,
) -> Result<usize, OperationError>
where
    A: HostKernelAdapter<D> + Clone + Send + Sync,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    let scatter = threads.is_some();
    #[cfg(feature = "tenferro")]
    {
        use symtn_dense::lanes;
        let estimate = strided_kernel::execution::KernelClass::Copy.estimate(dst.len());
        crate::host_pool::lane_region(engine, crate::host_pool::HostPoolSite::Strided, |exec| {
            let width = threads.unwrap_or_else(|| lanes::flat_width(&exec, estimate, members));
            if width <= 1 {
                let token = exec.into_token();
                return if scatter {
                    moves.scatter(token, kernels, &mut scratch[0], dst, packed_dst, 0, members)
                } else {
                    moves.pack(token, kernels, &mut scratch[0], dst, packed_src, 0, members)
                }
                .map(|()| width);
            }
            if scratch.len() < width {
                scratch.resize_with(width, MemberScratch::default);
            }
            let ranges = width.min(members);
            let work = MemberPhase {
                moves,
                packed_dst,
                lens,
                members,
                ranges,
                scatter,
                kernels: &*kernels,
                lane: core::marker::PhantomData,
            };
            let mut lane = MemberLane {
                dst,
                packed: packed_src,
                first: 0,
                kernels: kernels.clone(),
                scratch: &mut scratch[..width],
            };
            let mut states = vec![(); ranges];
            super::chunk::outcome(lanes::run_native(
                exec,
                &work,
                super::chunk::split(width, |lane, cut| work.cut(lane, cut)),
                &mut states,
                core::slice::from_mut(&mut lane),
            ))
            .map(|()| width)
        })?
    }
    #[cfg(not(feature = "tenferro"))]
    {
        let _ = (packed_dst, lens);
        crate::host_pool::with_exec(engine, |exec| {
            if scatter {
                moves.scatter(exec, kernels, &mut scratch[0], dst, packed_dst, 0, members)
            } else {
                moves.pack(exec, kernels, &mut scratch[0], dst, packed_src, 0, members)
            }
        })?;
        Ok(1)
    }
}

/// A member phase split over contiguous member ranges: range `j` of
/// `ranges` holds members `floor(j N / ranges)..floor((j + 1) N / ranges)`.
#[cfg(feature = "tenferro")]
struct MemberPhase<'a, 'm, 'd, D, C, A> {
    moves: &'a MemberMoves<'m, D, C>,
    packed_dst: &'a [D],
    lens: MemberLens,
    members: usize,
    ranges: usize,
    scatter: bool,
    kernels: &'a A,
    lane: core::marker::PhantomData<fn(&'d mut [D])>,
}

/// One worker of a member phase: its members' destination and packed
/// source (the pack phase only) from member `first`, its kernels and its
/// scratch (one per logical thread of its part; it uses the first).
#[cfg(feature = "tenferro")]
struct MemberLane<'d, D, A> {
    dst: &'d mut [D],
    packed: &'d mut [D],
    first: usize,
    kernels: A,
    scratch: &'d mut [MemberScratch],
}

#[cfg(feature = "tenferro")]
impl<'d, D, C, A: Clone> MemberPhase<'_, '_, 'd, D, C, A> {
    /// The first member of range `job`: `floor(job N / ranges)`.
    ///
    /// Why the quotient-remainder form: `job N` can exceed `usize` for an
    /// admitted batch of empty members, while each term below stays at most
    /// `N`.
    fn start(&self, job: usize) -> usize {
        let (quotient, remainder) = (self.members / self.ranges, self.members % self.ranges);
        job * quotient + job * remainder / self.ranges
    }

    /// Cuts `lane` before range `cut.job`: `lane` keeps the members before
    /// it and its first `cut.left` scratch entries.
    fn cut(
        &self,
        lane: &mut MemberLane<'d, D, A>,
        cut: symtn_dense::lanes::Cut,
    ) -> Result<MemberLane<'d, D, A>, OperationError> {
        let first = self.start(cut.job);
        let members = first
            .checked_sub(lane.first)
            .ok_or(OperationError::ElementCountOverflow)?;
        let (dst_left, dst_right) =
            core::mem::take(&mut lane.dst).split_at_mut(members * self.lens.dst);
        lane.dst = dst_left;
        let packed_at = if self.scatter {
            0
        } else {
            members * self.lens.packed_src
        };
        let (packed_left, packed_right) = core::mem::take(&mut lane.packed).split_at_mut(packed_at);
        lane.packed = packed_left;
        let (scratch_left, scratch_right) =
            core::mem::take(&mut lane.scratch).split_at_mut(cut.left);
        lane.scratch = scratch_left;
        Ok(MemberLane {
            dst: dst_right,
            packed: packed_right,
            first,
            kernels: self.kernels.clone(),
            scratch: scratch_right,
        })
    }
}

#[cfg(feature = "tenferro")]
impl<'d, D, C, A> symtn_dense::lanes::Work for MemberPhase<'_, '_, 'd, D, C, A>
where
    A: HostKernelAdapter<D> + Clone + Send + Sync,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    type Job = ();
    type Lane = MemberLane<'d, D, A>;
    type Error = OperationError;

    fn count(&self) -> usize {
        self.ranges
    }

    fn run(
        &self,
        job: usize,
        _: &mut (),
        lane: &mut Self::Lane,
        exec: symtn_dense::lanes::Exec<'_>,
    ) -> Result<(), OperationError> {
        let (first, end) = (self.start(job), self.start(job + 1));
        let skip = first - lane.first;
        let dst = &mut lane.dst[skip * self.lens.dst..(end - lane.first) * self.lens.dst];
        let scratch = &mut lane.scratch[0];
        if self.scatter {
            self.moves.scatter(
                exec.into_token(),
                &mut lane.kernels,
                scratch,
                dst,
                self.packed_dst,
                first,
                end - first,
            )
        } else {
            let packed = &mut lane.packed
                [skip * self.lens.packed_src..(end - lane.first) * self.lens.packed_src];
            self.moves.pack(
                exec.into_token(),
                &mut lane.kernels,
                scratch,
                dst,
                packed,
                first,
                end - first,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{StridedHostKernelAdapter, TreeTransformBlockSpec};
    use num_complex::Complex64;
    use symtn_dense::{
        DenseDotConfig, DenseError, DenseRead, DenseScalar, DenseTensor, DenseWrite,
    };

    fn fixture() -> (Arc<BlockStructure>, TreeTransformStructure<f64>) {
        let blocks = Arc::new(
            BlockStructure::packed_column_major(1, [vec![1], vec![1], vec![1], vec![1]]).unwrap(),
        );
        let transform = TreeTransformStructure::compile_structures(
            &blocks,
            &blocks,
            &[
                TreeTransformBlockSpec::multi(vec![0, 1], vec![0, 1], vec![1.0, 2.0, 3.0, 4.0]),
                TreeTransformBlockSpec::single(2, 2, -2.0),
            ],
        )
        .unwrap();
        (blocks, transform)
    }

    /// The built-in executor, counting the products it is handed (the
    /// default batch entry submits each job through the single product).
    struct CountingDense {
        inner: DefaultDenseExecutor,
        products: usize,
    }

    impl Default for CountingDense {
        fn default() -> Self {
            Self {
                inner: DefaultDenseExecutor::new(),
                products: 0,
            }
        }
    }

    /// The engines a member batch runs on: one worker, and four workers
    /// whose token its moves take.
    fn routes() -> [(symtn_dense::CpuEngine, CountingDense); 2] {
        [
            (symtn_dense::CpuEngine::serial(), CountingDense::default()),
            (
                crate::host_pool::eager_test_engine(4),
                CountingDense::default(),
            ),
        ]
    }

    /// The member overwrite of `source` into a destination filled with
    /// NaN on each of [`routes`].
    fn on_every_route(
        transform: &TreeTransformStructure<f64>,
        blocks: &Arc<BlockStructure>,
        source: &[Complex64],
        members: usize,
    ) -> Vec<Vec<Complex64>> {
        routes()
            .into_iter()
            .map(|(mut engine, mut dense)| {
                let mut workspace = TreeTransformWorkspace::default();
                let mut destination = vec![Complex64::new(f64::NAN, f64::NAN); source.len()];
                tree_transform_members_overwrite_raw(
                    &mut engine,
                    &mut StridedHostKernelAdapter::default(),
                    &mut dense,
                    &mut workspace,
                    transform,
                    blocks,
                    blocks,
                    &mut destination,
                    source,
                    members,
                    &[],
                )
                .unwrap();
                destination
            })
            .collect()
    }

    impl DenseExecutor for CountingDense {
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
        fn matmul_axpby_into(
            &mut self,
            engine: &mut dyn Engine,
            output: DenseWrite<'_>,
            lhs: DenseRead<'_>,
            rhs: DenseRead<'_>,
            alpha: DenseScalar,
            beta: DenseScalar,
        ) -> Result<(), DenseError> {
            self.products += 1;
            self.inner
                .matmul_axpby_into(engine, output, lhs, rhs, alpha, beta)
        }
    }

    #[test]
    fn member_overwrite_matches_hand_recoupling_and_clears_inactive() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let (blocks, transform) = fixture();
        let mut workspace = TreeTransformWorkspace::<f64>::default();
        for members in [1, 2, 17, 2] {
            let source: Vec<f64> = (0..members)
                .flat_map(|member| {
                    let base = member as f64 + 1.0;
                    [base, base + 2.0, base + 4.0, base + 6.0]
                })
                .collect();
            let mut destination = vec![f64::NAN; source.len()];
            let mut dense = CountingDense::default();
            tree_transform_members_overwrite_raw(
                engine,
                &mut StridedHostKernelAdapter::default(),
                &mut dense,
                &mut workspace,
                &transform,
                &blocks,
                &blocks,
                &mut destination,
                &source,
                members,
                &[],
            )
            .unwrap();
            for member in 0..members {
                let [a, b, c, _] = source[4 * member..][..4].try_into().unwrap();
                assert_eq!(
                    &destination[4 * member..4 * member + 4],
                    &[a + 2.0 * b, 3.0 * a + 4.0 * b, -2.0 * c, 0.0]
                );
            }
        }
    }

    #[test]
    fn complex_members_keep_real_structural_scale_and_grouped_rhs() {
        let (blocks, transform) = fixture();
        let source = [
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, -1.0),
            Complex64::new(5.0, 4.0),
            Complex64::new(7.0, 1.0),
            Complex64::new(2.0, -3.0),
            Complex64::new(-1.0, 5.0),
            Complex64::new(4.0, 2.0),
            Complex64::new(6.0, 2.0),
        ];
        for dst in on_every_route(&transform, &blocks, &source, 2) {
            for member in 0..2 {
                let a = source[4 * member];
                let b = source[4 * member + 1];
                let c = source[4 * member + 2];
                let expected = [
                    a + 2.0 * b,
                    3.0 * a + 4.0 * b,
                    -2.0 * c,
                    Complex64::new(0.0, 0.0),
                ];
                assert_eq!(&dst[4 * member..4 * member + 4], &expected);
            }
        }
    }

    #[test]
    fn short_last_member_rejects_before_any_write_or_dense_submission() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let (blocks, transform) = fixture();
        let source = [1.0; 7];
        let mut dst = [f64::NAN; 8];
        let mut dense = CountingDense::default();
        let error = tree_transform_members_overwrite_raw(
            engine,
            &mut StridedHostKernelAdapter::default(),
            &mut dense,
            &mut TreeTransformWorkspace::default(),
            &transform,
            &blocks,
            &blocks,
            &mut dst,
            &source,
            2,
            &[],
        )
        .unwrap_err();
        assert!(matches!(
            error,
            OperationError::ElementCountMismatch {
                expected: 8,
                actual: 7
            }
        ));
        assert!(dst.iter().all(|value| value.is_nan()));
        assert_eq!(dense.products, 0);
    }

    #[test]
    fn heterogeneous_multi_blocks_recouple_every_member() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let blocks = Arc::new(
            BlockStructure::packed_column_major(
                1,
                [vec![2], vec![2], vec![1], vec![1], vec![1], vec![1]],
            )
            .unwrap(),
        );
        let transform = TreeTransformStructure::compile_structures(
            &blocks,
            &blocks,
            &[
                TreeTransformBlockSpec::multi(vec![0, 1], vec![0, 1], vec![1.0, 2.0, -1.0, 3.0]),
                TreeTransformBlockSpec::multi(vec![2, 3], vec![2, 3], vec![2.0, -1.0, 4.0, 1.0]),
                TreeTransformBlockSpec::single(4, 4, 5.0),
            ],
        )
        .unwrap();
        let members = 17;
        let source: Vec<f64> = (0..members)
            .flat_map(|member| {
                let t = member as f64;
                [
                    1.0 + t,
                    2.0 + t,
                    3.0 + t,
                    4.0 + t,
                    5.0 + t,
                    6.0 + t,
                    7.0 + t,
                    8.0 + t,
                ]
            })
            .collect();
        let mut destination = vec![f64::NAN; source.len()];
        let mut dense = CountingDense::default();
        tree_transform_members_overwrite_raw(
            engine,
            &mut StridedHostKernelAdapter::default(),
            &mut dense,
            &mut TreeTransformWorkspace::default(),
            &transform,
            &blocks,
            &blocks,
            &mut destination,
            &source,
            members,
            &[],
        )
        .unwrap();
        for member in 0..members {
            let [a, b, c, d, e, f, g, _] = source[8 * member..8 * member + 8].try_into().unwrap();
            assert_eq!(
                &destination[8 * member..8 * member + 8],
                &[
                    a + 2.0 * c,
                    b + 2.0 * d,
                    -a + 3.0 * c,
                    -b + 3.0 * d,
                    2.0 * e - f,
                    4.0 * e + f,
                    5.0 * g,
                    0.0,
                ]
            );
        }
    }

    #[test]
    fn conjugated_complex_source_uses_the_task_view_flag() {
        let blocks = Arc::new(BlockStructure::packed_column_major(1, [vec![1]]).unwrap());
        let transform = TreeTransformStructure::compile_structures_with_storage_conjugation(
            &blocks,
            &blocks,
            &[TreeTransformBlockSpec::single(0, 0, -2.0_f64)],
            true,
        )
        .unwrap();
        let source = [Complex64::new(1.0, 2.0), Complex64::new(-3.0, 4.0)];
        let mut destination = [Complex64::new(f64::NAN, f64::NAN); 2];
        tree_transform_members_overwrite_raw(
            &mut symtn_dense::CpuEngine::serial(),
            &mut StridedHostKernelAdapter::default(),
            &mut CountingDense::default(),
            &mut TreeTransformWorkspace::default(),
            &transform,
            &blocks,
            &blocks,
            &mut destination,
            &source,
            2,
            &[],
        )
        .unwrap();
        assert_eq!(
            destination,
            [Complex64::new(-2.0, 4.0), Complex64::new(6.0, 8.0)]
        );
    }

    #[test]
    fn conjugated_complex_multi_pack_matches_hand_matrix_action() {
        let blocks = Arc::new(BlockStructure::packed_column_major(1, [vec![1], vec![1]]).unwrap());
        let transform = TreeTransformStructure::compile_structures_with_storage_conjugation(
            &blocks,
            &blocks,
            &[TreeTransformBlockSpec::multi(
                vec![0, 1],
                vec![0, 1],
                vec![1.0_f64, 2.0, 3.0, 4.0],
            )],
            true,
        )
        .unwrap();
        let source = [
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, -1.0),
            Complex64::new(-2.0, 4.0),
            Complex64::new(5.0, 3.0),
        ];
        for destination in on_every_route(&transform, &blocks, &source, 2) {
            for member in 0..2 {
                let a = source[2 * member].conj();
                let b = source[2 * member + 1].conj();
                assert_eq!(
                    &destination[2 * member..2 * member + 2],
                    &[a + 2.0 * b, 3.0 * a + 4.0 * b]
                );
            }
        }
    }

    #[test]
    fn invalid_batch_count_rejects_before_writes() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let (blocks, transform) = fixture();
        let mut destination = [f64::NAN; 8];
        let mut dense = CountingDense::default();
        for members in [0, usize::MAX] {
            assert!(tree_transform_members_overwrite_raw(
                engine,
                &mut StridedHostKernelAdapter::default(),
                &mut dense,
                &mut TreeTransformWorkspace::default(),
                &transform,
                &blocks,
                &blocks,
                &mut destination,
                &[1.0; 8],
                members,
                &[],
            )
            .is_err());
        }
        assert!(destination.iter().all(|value| value.is_nan()));
        assert_eq!(dense.products, 0);
    }
}
