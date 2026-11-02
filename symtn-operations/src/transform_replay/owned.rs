use super::*;

/// Internal, unstable owned-output path for the built-in host executor:
/// allocates the destination uninitialised and writes every element exactly
/// once. Returns `Ok(None)` without allocating output when the destination
/// does not have a proof of exact physical overwrite coverage.
///
/// `threads` is the requested replay worker count; the effective count follows
/// the same schedule rule as the initialised replay
/// (`effective_tree_transform_threads`), so the writer runs under the parallel
/// schedule whenever the initialised path would, and serially otherwise.
/// Either way each block keeps the engine-token share of the worker that
/// writes it, so one large block can still widen its own move.
///
/// `profile` accumulates the same phases as the initialised profiled replay
/// (benches and tests only; production callers pass `None`).
///
/// Why public: `symtn-tensors` is a separate crate. This executor seam is not
/// a general backend API; downstream callers must not rely on it.
#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn try_tree_transform_structure_overwrite_owned_raw<E, D, C>(
    engine: &mut dyn Engine,
    dense: &mut E,
    workspace: &mut TreeTransformWorkspace<D>,
    structure: &TreeTransformStructure<C>,
    dst_structure: &Arc<BlockStructure>,
    src_structure: &Arc<BlockStructure>,
    nout: usize,
    src_data: &[D],
    alpha: D,
    threads: usize,
    mut profile: Option<&mut TreeTransformReplayProfile>,
) -> Result<Option<Vec<D>>, OperationError>
where
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    let total_start = profile.as_ref().map(|_| std::time::Instant::now());
    let Some(proof) = PhysicalOverwriteProof::new(
        structure,
        dst_structure,
        src_structure,
        src_data.len(),
        nout,
    )?
    else {
        return Ok(None);
    };
    debug_assert_eq!(proof.required_len, dst_structure.required_len()?);
    debug_assert_eq!(proof.nout, nout);
    debug_assert!(Arc::ptr_eq(proof.dst_structure, dst_structure));
    debug_assert!(core::ptr::eq(proof.structure, structure));
    let task = structure.task_view()?;
    let schedule = structure.parallel_schedule();
    let threads = effective_tree_transform_threads(engine, schedule, threads);
    let layouts = structure.layouts();
    let max_fused_rank = layouts.max_fused_rank();
    // One gate for every unsplit block list on the calling thread; forked
    // leaves build their own.
    let gate = LeafGate::new(layouts.max_planned_elements());
    workspace.prepare_fused_indices(threads, max_fused_rank)?;
    if let (Some(profile), Some(start)) = (profile.as_deref_mut(), total_start) {
        profile.validate += start.elapsed();
        profile.single_blocks += schedule.single_block_count;
        profile.multi_blocks += structure.recoupling_plan().jobs().len();
    }

    let output = initialize_owned::<D, OperationError>(proof.required_len, |dst_data| {
        let recoupling_plan = structure.recoupling_plan();
        let storage_conjugate = structure.storage_conjugate();
        let coefficients = task.single_coefficients();

        let start = profile.as_ref().map(|_| std::time::Instant::now());
        for &layout_index in structure.inactive_destination_layouts() {
            write_uninit_layout_zero(layouts, layout_index, dst_data)?;
        }
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            profile.strided_kernel += start.elapsed();
        }
        let blocks = structure.blocks();
        // Singles split over their ownership atoms exactly as the initialised
        // parallel replay does; one atom stays on the calling lane.
        let start = profile.as_ref().map(|_| std::time::Instant::now());
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
                    for &item in items {
                        let (dst_layout, src_layout, coefficient) = scheduled_single(blocks, item)?;
                        write_uninit_layout_from_source(
                            exec.reborrow(),
                            gate,
                            layouts,
                            dst_layout,
                            src_layout,
                            dst,
                            dst_start,
                            src_data,
                            storage_conjugate,
                            TransformScale::new(alpha, coefficients[coefficient]),
                            fused_index,
                        )?;
                    }
                    Ok(())
                },
            )
        })??;
        if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
            let elapsed = start.elapsed();
            profile.single_total += elapsed;
            profile.strided_kernel += elapsed;
        }

        if recoupling_plan.is_empty() {
            return Ok(());
        }
        let start = profile.as_ref().map(|_| std::time::Instant::now());
        if ensure_recoupling_coefficients(workspace, task, structure.identity_marker())? {
            if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
                profile.multi_coefficient_prepare += start.elapsed();
            }
        }
        let chunk_size = threads.max(1);
        reserve_multi_chunk_scratch(workspace, recoupling_plan.jobs(), chunk_size)?;
        let mut target = ScatterTarget::new(workspace, schedule, threads, dst_data)?;
        let mut pack_cursor = 0;
        for (chunk_index, chunk) in recoupling_plan.jobs().chunks(chunk_size).enumerate() {
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
            let packed = packed.destination().as_slice();
            let scatter_columns = &schedule.scatter_columns;
            let scatter_groups = &schedule.scatter_groups;
            let start = profile.as_ref().map(|_| std::time::Instant::now());
            match &mut target {
                ScatterTarget::Atoms(slots) => replay_region(engine, |exec| {
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
                            for &group in groups {
                                for item in &scatter_columns[scatter_groups[group].columns.clone()]
                                {
                                    write_uninit_layout_from_packed(
                                        exec.reborrow(),
                                        gate,
                                        layouts,
                                        item.dst_layout,
                                        &mut *slots[item.slot as usize - slot_base],
                                        atom_start(layouts, item)?,
                                        packed,
                                        item.packed_offset - destination_start,
                                        alpha,
                                        fused_index,
                                    )?;
                                }
                            }
                            Ok(())
                        },
                    )
                })??,
                ScatterTarget::Whole(dst) => replay_region(engine, |mut exec| {
                    let fused_index = &mut fused_indices[..max_fused_rank];
                    for &group in chunk_scatter_groups.iter() {
                        for item in &scatter_columns[scatter_groups[group].columns.clone()] {
                            write_uninit_layout_from_packed(
                                exec.reborrow(),
                                &gate,
                                layouts,
                                item.dst_layout,
                                dst,
                                0,
                                packed,
                                item.packed_offset - destination_start,
                                alpha,
                                fused_index,
                            )?;
                        }
                    }
                    Ok::<(), OperationError>(())
                })??,
            }
            if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
                profile.scattered_columns += chunk.iter().map(|job| job.cols).sum::<usize>();
                profile.multi_scatter += start.elapsed();
            }
        }
        target.finish(workspace);
        debug_assert_eq!(pack_cursor, schedule.pack_columns.len());
        Ok(())
    })?;
    if let (Some(profile), Some(start)) = (profile, total_start) {
        profile.total += start.elapsed();
    }
    Ok(Some(output))
}

fn layout_linear_offset(
    mut linear: usize,
    shape: &[usize],
    strides: &[isize],
    base: isize,
) -> Result<usize, OperationError> {
    let mut offset = base;
    for (&dim, &stride) in shape.iter().zip(strides) {
        let coordinate = if dim == 0 { 0 } else { linear % dim };
        if let Some(quotient) = linear.checked_div(dim) {
            linear = quotient;
        }
        let coordinate =
            isize::try_from(coordinate).map_err(|_| OperationError::ElementCountOverflow)?;
        offset = offset
            .checked_add(
                coordinate
                    .checked_mul(stride)
                    .ok_or_else(|| OperationError::ElementCountOverflow)?,
            )
            .ok_or_else(|| OperationError::ElementCountOverflow)?;
    }
    usize::try_from(offset).map_err(|_| OperationError::OffsetOverflow { value: usize::MAX })
}

// Why-not fuse the zero writer: it has no paired source view, and a pure
// permute — the deg=1 U(1) owned-path regime this optimization targets — always
// touches every destination block, so `inactive_destination_layouts` is empty
// and this writer never runs on the hot path. It walks the inactive entry's
// compiled one-sided role, which addresses the same element set as the raw
// block, on the per-element odometer (issue #232, condition 2).
fn write_uninit_layout_zero<D: Zero + Copy>(
    layouts: &TreeTransformLayoutTable,
    layout_index: usize,
    dst: &mut [MaybeUninit<D>],
) -> Result<(), OperationError> {
    let (dims, strides) = layouts.inactive_role(layout_index)?;
    let layout = layouts.entry(layout_index);
    for linear in 0..layout.element_count {
        let index = layout_linear_offset(linear, dims, strides, layout.offset)?;
        dst[index].write(D::zero());
    }
    Ok(())
}

/// `dst` may be a split of the owned destination starting at absolute offset
/// `dst_start`; layout offsets are rebased exactly as the initialised parallel
/// replay rebases them.
#[allow(clippy::too_many_arguments)]
fn write_uninit_layout_from_source<D, C>(
    exec: Exec<'_>,
    gate: &LeafGate,
    layouts: &TreeTransformLayoutTable,
    dst_index: usize,
    src_index: usize,
    dst: &mut [MaybeUninit<D>],
    dst_start: isize,
    src: &[D],
    conjugate: bool,
    scale: TransformScale<D, C>,
    fused_index: &mut [usize],
) -> Result<(), OperationError>
where
    D: Copy
        + Send
        + Sync
        + Mul<D, Output = D>
        + Zero
        + One
        + PartialEq
        + ConjugateValue
        + RecouplingCoefficientAction<C>,
    C: Copy + Sync,
{
    // The element op is chosen once per block, with the structural coefficient
    // still in its own type; see `TransformScale`.
    if scale.is_identity() {
        return write_uninit_layout_mapped(
            exec,
            gate,
            layouts,
            dst_index,
            src_index,
            dst,
            dst_start,
            src,
            fused_index,
            move |value: D| value.maybe_conj(conjugate),
        );
    }
    if scale.is_zero() {
        let zero = scale.apply(D::zero());
        return write_uninit_layout_mapped(
            exec,
            gate,
            layouts,
            dst_index,
            src_index,
            dst,
            dst_start,
            src,
            fused_index,
            move |_: D| zero,
        );
    }
    match scale {
        TransformScale::Structural(coefficient) => {
            let coefficient = &coefficient;
            write_uninit_layout_mapped(
                exec,
                gate,
                layouts,
                dst_index,
                src_index,
                dst,
                dst_start,
                src,
                fused_index,
                move |value: D| {
                    value
                        .maybe_conj(conjugate)
                        .scale_by_coefficient(*coefficient)
                },
            )
        }
        TransformScale::Data(scale) => write_uninit_layout_mapped(
            exec,
            gate,
            layouts,
            dst_index,
            src_index,
            dst,
            dst_start,
            src,
            fused_index,
            move |value: D| scale * value.maybe_conj(conjugate),
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn write_uninit_layout_mapped<D, F>(
    exec: Exec<'_>,
    gate: &LeafGate,
    layouts: &TreeTransformLayoutTable,
    dst_index: usize,
    src_index: usize,
    dst: &mut [MaybeUninit<D>],
    dst_start: isize,
    src: &[D],
    fused_index: &mut [usize],
    map: F,
) -> Result<(), OperationError>
where
    D: Copy + Send + Sync + Mul<D, Output = D> + One + PartialEq,
    F: Fn(D) -> D + Send + Sync,
{
    write_role_uninit(
        exec,
        gate,
        layouts.role(dst_index)?,
        layouts.entry(dst_index).element_count,
        dst,
        src,
        layouts.entry(dst_index).offset - dst_start,
        layouts.entry(src_index).offset,
        fused_index,
        map,
    )
}

/// Writes one role into uninitialized memory: its compiled plan across
/// `exec` when [`admitted_plan`] admits one, the baked walk otherwise.
#[allow(clippy::too_many_arguments)]
fn write_role_uninit<D, F>(
    exec: Exec<'_>,
    gate: &LeafGate,
    role: crate::BakedFusedLayout<'_>,
    elements: usize,
    dst: &mut [MaybeUninit<D>],
    src: &[D],
    dst_offset: isize,
    src_offset: isize,
    fused_index: &mut [usize],
    map: F,
) -> Result<(), OperationError>
where
    D: Copy + Send + Sync + Mul<D, Output = D> + One + PartialEq,
    F: Fn(D) -> D + Send + Sync,
{
    match admitted_plan(role, elements, &exec, gate) {
        Some(plan) => planned_write_uninit(plan, role, dst, src, dst_offset, src_offset, exec, map),
        None => write_fused_uninit(role, dst, src, dst_offset, src_offset, fused_index, map),
    }
}

#[allow(clippy::too_many_arguments)]
fn write_uninit_layout_from_packed<D>(
    exec: Exec<'_>,
    gate: &LeafGate,
    layouts: &TreeTransformLayoutTable,
    dst_index: usize,
    dst: &mut [MaybeUninit<D>],
    dst_start: isize,
    packed: &[D],
    packed_offset: usize,
    alpha: D,
    fused_index: &mut [usize],
) -> Result<(), OperationError>
where
    D: Copy + Send + Sync + Mul<D, Output = D> + Zero + One + PartialEq,
{
    // Why the identity arm: `1 * (inf + 0i)` is `inf + NaN i`, so the scatter
    // must copy rather than multiply when the caller's alpha is one. A zero
    // alpha gives `scale_value`'s exact zero even where the recoupling GEMM
    // left a NaN in `packed`, as TensorKit's scatter `tensoradd!` does.
    // The scatter role bakes src = packed (column-major) strides, so the
    // fused walk over `packed` starting at `packed_offset` reproduces the
    // column-major gather `packed[packed_offset + linear]`.
    let role = layouts.role(dst_index)?;
    let elements = layouts.entry(dst_index).element_count;
    let dst_offset = layouts.entry(dst_index).offset - dst_start;
    let src_offset = offset_to_isize(packed_offset)?;
    if alpha.is_one() {
        write_role_uninit(
            exec,
            gate,
            role,
            elements,
            dst,
            packed,
            dst_offset,
            src_offset,
            fused_index,
            |v| v,
        )
    } else {
        write_role_uninit(
            exec,
            gate,
            role,
            elements,
            dst,
            packed,
            dst_offset,
            src_offset,
            fused_index,
            move |value| scale_value(value, alpha),
        )
    }
}

#[cfg(test)]
mod owned_overwrite_tests {
    use super::*;
    use crate::{StridedHostKernelAdapter, TreeTransformBlockSpec};
    use num_complex::Complex64;
    use symtn_core::{
        BlockKey, BlockSpec, FusionProductSpace, FusionTreeHomSpace, FusionTreePairKey, SectorLeg,
        Z2FusionRule, Z2Irrep,
    };
    use symtn_dense::{
        DenseBackend, DenseDotConfig, DenseError, DenseRead, DenseScalar, DenseTensor, DenseWrite,
    };

    fn canonical_structure(offset: usize) -> Arc<BlockStructure> {
        canonical_block(offset, 2, 3)
    }

    /// Rows x cols elements of the planned-leaf fixtures: 24 keeps Strided's
    /// copy estimate nonzero on every host class (at least 0.09 ns per
    /// element), so the role retains a plan for a wide engine to widen.
    const PLANNED_ROWS: usize = 4;
    const PLANNED_COLS: usize = 6;
    const PLANNED_LEN: usize = PLANNED_ROWS * PLANNED_COLS;

    fn canonical_block(offset: usize, rows: usize, cols: usize) -> Arc<BlockStructure> {
        let key = BlockKey::from(
            FusionTreePairKey::try_pair_from_sector_ids(
                [1],
                [1],
                1,
                [false],
                [false],
                [],
                [],
                [],
                [],
            )
            .unwrap(),
        );
        Arc::new(
            BlockStructure::from_blocks_with_rank(
                2,
                vec![BlockSpec::with_key(key, vec![rows, cols], vec![1, rows], offset).unwrap()],
            )
            .unwrap(),
        )
    }

    #[test]
    fn owned_writer_matches_initialized_oracle_for_real_and_complex() {
        // What: canonical owned replay writes every physical value and is
        // byte-for-byte equal to the initialized overwrite oracle; at
        // `alpha = 1` both scale a complex payload by the real structural
        // coefficient componentwise, so `±inf`, NaN and `-0` components keep
        // exactly the bits of `(-2 * re, -2 * im)`. The second engine declares
        // free dispatch, so the block's move is widened onto its compiled plan.
        for mut engine in [
            symtn_dense::CpuEngine::serial(),
            crate::host_pool::eager_test_engine(2),
        ] {
            let engine = &mut engine;
            let structure = canonical_block(0, PLANNED_ROWS, PLANNED_COLS);
            let transform = TreeTransformStructure::compile_structures(
                &structure,
                &structure,
                &[TreeTransformBlockSpec::single(0, 0, -2.0)],
            )
            .unwrap();
            // Precondition: the Single role retains its plan on every host.
            assert_eq!(transform.layouts().retained_plan_count(), 1);
            let real_src = (1..=PLANNED_LEN).map(|v| v as f64).collect::<Vec<_>>();

            let mut expected = vec![f64::NAN; PLANNED_LEN];
            tree_transform_structure_overwrite_with_structural_recoupling_raw(
                engine,
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &transform,
                &structure,
                &structure,
                &mut expected,
                &real_src,
                3.0,
                &[],
                1,
            )
            .unwrap();
            let actual = try_tree_transform_structure_overwrite_owned_raw(
                engine,
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &transform,
                &structure,
                &structure,
                1,
                &real_src,
                3.0,
                1,
                None,
            )
            .unwrap()
            .unwrap();
            assert_eq!(actual, expected);

            let complex_src = (1..=PLANNED_LEN)
                .map(|value| Complex64::new(value as f64, -(value as f64)))
                .collect::<Vec<_>>();
            let complex = try_tree_transform_structure_overwrite_owned_raw(
                engine,
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &transform,
                &structure,
                &structure,
                1,
                &complex_src,
                Complex64::new(3.0, 1.0),
                1,
                None,
            )
            .unwrap()
            .unwrap();
            let scale = Complex64::new(3.0, 1.0) * -2.0;
            assert_eq!(
                complex,
                complex_src.iter().map(|&v| scale * v).collect::<Vec<_>>()
            );

            let nan = f64::from_bits(0x7ff8_0000_0000_1398);
            let non_finite = [
                Complex64::new(f64::INFINITY, 0.0),
                Complex64::new(-0.0, 1.0),
                Complex64::new(nan, -0.0),
                Complex64::new(1.5, f64::NEG_INFINITY),
                Complex64::new(0.0, -0.0),
                Complex64::new(-3.0, 0.5),
            ]
            .repeat(PLANNED_LEN / 6);
            let bits = |values: &[Complex64]| {
                values
                    .iter()
                    .map(|v| (v.re.to_bits(), v.im.to_bits()))
                    .collect::<Vec<_>>()
            };
            let componentwise = non_finite
                .iter()
                .map(|v| Complex64::new(-2.0 * v.re, -2.0 * v.im))
                .collect::<Vec<_>>();
            let owned = try_tree_transform_structure_overwrite_owned_raw(
                engine,
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &transform,
                &structure,
                &structure,
                1,
                &non_finite,
                Complex64::new(1.0, 0.0),
                1,
                None,
            )
            .unwrap()
            .unwrap();
            let mut initialized = vec![Complex64::new(f64::NAN, f64::NAN); PLANNED_LEN];
            tree_transform_structure_overwrite_with_structural_recoupling_raw(
                engine,
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &transform,
                &structure,
                &structure,
                &mut initialized,
                &non_finite,
                Complex64::new(1.0, 0.0),
                &[],
                1,
            )
            .unwrap();
            assert_eq!(bits(&owned), bits(&componentwise));
            assert_eq!(bits(&initialized), bits(&componentwise));
        }
    }

    #[test]
    fn owned_writer_multi_matches_direct_matrix_oracle() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        // What: the uninitialized Multi writer applies every destination-by-source
        // recoupling coefficient and writes the final owned payload exactly once.
        let space = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new(
                [(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 1)],
                false,
            )]),
            FusionProductSpace::new([SectorLeg::new(
                [(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 1)],
                false,
            )]),
        )
        .coupled_subblock_structure(&Z2FusionRule, 1, [vec![1, 1], vec![1, 1]])
        .unwrap();
        let structure = Arc::clone(&space);
        let transform = TreeTransformStructure::compile_structures(
            &structure,
            &structure,
            &[TreeTransformBlockSpec::multi(
                vec![0, 1],
                vec![0, 1],
                vec![2.0, 3.0, 5.0, 7.0],
            )],
        )
        .unwrap();

        let actual = try_tree_transform_structure_overwrite_owned_raw(
            engine,
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            &transform,
            &structure,
            &structure,
            1,
            &[11.0, 13.0],
            2.0,
            1,
            None,
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            actual,
            [
                2.0 * (2.0 * 11.0 + 3.0 * 13.0),
                2.0 * (5.0 * 11.0 + 7.0 * 13.0)
            ]
        );
    }

    #[test]
    fn zero_scales_write_zeros_over_non_finite_sources() {
        // What: the uninitialized writer and the initialized overwrite follow
        // VectorInterface's `scale(x, 0) = zero(x) * 0` for a zero caller
        // alpha and for a zero structural coefficient in a Single block, and
        // for a zero alpha in a Multi scatter whose recoupling GEMM saw
        // `inf`: every output is `0.0`, as TensorKit's
        // `permute!(tdst, tsrc, p, 0, 0)` (and `scale(Inf, 0.0) == 0.0`)
        // gives. The second engine declares free dispatch, so every block's
        // move is widened onto its compiled plan.
        for mut engine in [
            symtn_dense::CpuEngine::serial(),
            crate::host_pool::eager_test_engine(2),
        ] {
            let engine = &mut engine;
            let non_finite = [f64::INFINITY, f64::NAN, f64::NEG_INFINITY, 1.0, -0.0, 2.0]
                .repeat(PLANNED_LEN / 6);
            let structure = canonical_block(0, PLANNED_ROWS, PLANNED_COLS);
            for (alpha, coefficient) in [(0.0, -2.0), (1.0, 0.0), (3.0, 0.0)] {
                let transform = TreeTransformStructure::compile_structures(
                    &structure,
                    &structure,
                    &[TreeTransformBlockSpec::single(0, 0, coefficient)],
                )
                .unwrap();
                // Precondition: the Single role retains its plan on every host.
                assert_eq!(transform.layouts().retained_plan_count(), 1);
                let owned = try_tree_transform_structure_overwrite_owned_raw(
                    engine,
                    &mut DefaultDenseExecutor::new(),
                    &mut TreeTransformWorkspace::default(),
                    &transform,
                    &structure,
                    &structure,
                    1,
                    &non_finite,
                    alpha,
                    1,
                    None,
                )
                .unwrap()
                .unwrap();
                let mut initialized = vec![f64::NAN; non_finite.len()];
                tree_transform_structure_overwrite_with_structural_recoupling_raw(
                    engine,
                    &mut StridedHostKernelAdapter::default(),
                    &mut DefaultDenseExecutor::new(),
                    &mut TreeTransformWorkspace::default(),
                    &transform,
                    &structure,
                    &structure,
                    &mut initialized,
                    &non_finite,
                    alpha,
                    &[],
                    1,
                )
                .unwrap();
                for actual in [owned, initialized] {
                    assert!(
                        actual.iter().all(|&value| value == 0.0),
                        "alpha {alpha}, coefficient {coefficient}: {actual:?}"
                    );
                }
            }

            // Scatter columns of `PLANNED_LEN` elements, so the wide engine
            // plans them too.
            let space = FusionTreeHomSpace::new(
                FusionProductSpace::new([z2_leg(PLANNED_ROWS, PLANNED_ROWS)]),
                FusionProductSpace::new([z2_leg(PLANNED_COLS, PLANNED_COLS)]),
            )
            .coupled_subblock_structure(
                &Z2FusionRule,
                1,
                [
                    vec![PLANNED_ROWS, PLANNED_COLS],
                    vec![PLANNED_ROWS, PLANNED_COLS],
                ],
            )
            .unwrap();
            let structure = Arc::clone(&space);
            let transform = TreeTransformStructure::compile_structures(
                &structure,
                &structure,
                &[TreeTransformBlockSpec::multi(
                    vec![0, 1],
                    vec![0, 1],
                    vec![2.0, 3.0, 5.0, 7.0],
                )],
            )
            .unwrap();
            // Precondition: both scatter columns retain their plans.
            assert_eq!(transform.layouts().retained_plan_count(), 2);
            let source = [f64::INFINITY, f64::NEG_INFINITY, f64::NAN].repeat(2 * PLANNED_LEN / 3);
            let owned = try_tree_transform_structure_overwrite_owned_raw(
                engine,
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &transform,
                &structure,
                &structure,
                1,
                &source,
                0.0,
                1,
                None,
            )
            .unwrap()
            .unwrap();
            let mut initialized = vec![f64::NAN; source.len()];
            tree_transform_structure_overwrite_with_structural_recoupling_raw(
                engine,
                &mut StridedHostKernelAdapter::default(),
                &mut DefaultDenseExecutor::new(),
                &mut TreeTransformWorkspace::default(),
                &transform,
                &structure,
                &structure,
                &mut initialized,
                &source,
                0.0,
                &[],
                1,
            )
            .unwrap();
            assert_eq!(owned, [0.0; 2 * PLANNED_LEN]);
            assert_eq!(initialized, [0.0; 2 * PLANNED_LEN]);
        }
    }

    #[test]
    fn owned_writer_covers_canonical_interleaved_blocks() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        // What: in a canonical layout with two codomain trees per coupled
        // sector, the blocks' columns interleave inside the sector matrix;
        // the owned writer still returns every element, Singles, Multi
        // scatter and an inactive block alike, equal to the initialized
        // overwrite.
        let structure = FusionTreeHomSpace::new(
            FusionProductSpace::new([z2_leg(1, 1), z2_leg(1, 1)]),
            FusionProductSpace::new([z2_leg(2, 2)]),
        )
        .coupled_subblock_structure_from_leg_degeneracies(&Z2FusionRule)
        .unwrap();
        assert_eq!(structure.block_count(), 4);
        let inactive = structure.block(3).unwrap();
        assert_eq!(inactive.shape(), [1, 1, 2]);
        let inactive_offsets = [inactive.offset(), inactive.offset() + inactive.strides()[2]];
        let transform = TreeTransformStructure::compile_structures(
            &structure,
            &structure,
            &[
                TreeTransformBlockSpec::single(0, 0, -2.0),
                TreeTransformBlockSpec::multi(vec![1, 2], vec![1, 2], vec![-2.0, 0.0, 0.0, -2.0]),
            ],
        )
        .unwrap();
        let source = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let expected = (0..source.len())
            .map(|at| {
                if inactive_offsets.contains(&at) {
                    0.0
                } else {
                    -3.0 * source[at]
                }
            })
            .collect::<Vec<_>>();

        let actual = try_tree_transform_structure_overwrite_owned_raw(
            engine,
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            &transform,
            &structure,
            &structure,
            2,
            &source,
            1.5,
            1,
            None,
        )
        .unwrap();
        assert_eq!(actual, Some(expected.clone()));

        let mut initialized = vec![f64::NAN; source.len()];
        tree_transform_structure_overwrite_with_structural_recoupling_raw(
            engine,
            &mut StridedHostKernelAdapter::default(),
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            &transform,
            &structure,
            &structure,
            &mut initialized,
            &source,
            1.5,
            &[],
            1,
        )
        .unwrap();
        assert_eq!(initialized, expected);
    }

    fn pool(threads: usize) -> crate::host_pool::TestPool {
        crate::host_pool::test_pool(threads)
    }

    fn z2_leg(even: usize, odd: usize) -> SectorLeg {
        SectorLeg::new([(Z2Irrep::EVEN, even), (Z2Irrep::ODD, odd)], false)
    }

    /// Z2 rank (1, 3): one codomain tree per coupled sector, so the eight
    /// 24-element blocks are contiguous column slices (slice-disjoint).
    /// Singles and Multi groups interleave in destination order: three
    /// runnable singles and five pack columns.
    fn mixed_fixture() -> (Arc<BlockStructure>, TreeTransformStructure<f64>, Vec<f64>) {
        let space = FusionTreeHomSpace::new(
            FusionProductSpace::new([z2_leg(3, 3)]),
            FusionProductSpace::new([z2_leg(2, 2), z2_leg(2, 2), z2_leg(2, 2)]),
        )
        .coupled_subblock_structure(&Z2FusionRule, 1, vec![vec![3, 2, 2, 2]; 8])
        .unwrap();
        let structure = Arc::clone(&space);
        assert_eq!(structure.block_count(), 8);
        let specs = vec![
            TreeTransformBlockSpec::single(0, 0, -2.0),
            TreeTransformBlockSpec::multi(
                vec![1, 2, 3],
                vec![1, 2, 3],
                vec![1.0, -1.0, 2.0, 0.5, 4.0, -3.0, 1.5, 2.5, -0.5],
            ),
            TreeTransformBlockSpec::single(4, 4, 0.5),
            TreeTransformBlockSpec::multi(vec![5, 6], vec![5, 6], vec![2.0, 3.0, 5.0, 7.0]),
            TreeTransformBlockSpec::single(7, 7, 3.0),
        ];
        let transform =
            TreeTransformStructure::compile_structures(&structure, &structure, &specs).unwrap();
        let source = (1..=structure.required_len().unwrap())
            .map(|value| value as f64)
            .collect::<Vec<_>>();
        (structure, transform, source)
    }

    #[test]
    fn owned_writer_under_the_parallel_schedule_is_bit_identical_to_serial() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        // What: with a multi-worker budget the owned writer splits the
        // uninitialised destination over its ownership components and
        // produces exactly the serial owned bytes for real and complex
        // data, which in turn equal the initialised overwrite oracle.
        let (structure, transform, source) = mixed_fixture();
        let schedule = transform.parallel_schedule();
        assert_eq!(schedule.single_component_count(), 3);
        assert_eq!(schedule.scatter_component_count(), 2);
        assert_eq!(schedule.singles.len(), 3);
        assert_eq!(schedule.pack_columns.len(), 5);

        let mut pool = pool(3);
        assert_eq!(
            pool.install(|engine| effective_tree_transform_threads(engine, schedule, 3)),
            3
        );

        let mut oracle = vec![f64::NAN; source.len()];
        tree_transform_structure_overwrite_with_structural_recoupling_raw(
            engine,
            &mut StridedHostKernelAdapter::default(),
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            &transform,
            &structure,
            &structure,
            &mut oracle,
            &source,
            1.5,
            &[],
            1,
        )
        .unwrap();
        let serial = try_tree_transform_structure_overwrite_owned_raw(
            engine,
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            &transform,
            &structure,
            &structure,
            1,
            &source,
            1.5,
            1,
            None,
        )
        .unwrap()
        .unwrap();
        assert_eq!(serial, oracle);
        let mut workspace = TreeTransformWorkspace::default();
        let parallel = pool
            .install(|engine| {
                try_tree_transform_structure_overwrite_owned_raw(
                    engine,
                    &mut DefaultDenseExecutor::new(),
                    &mut workspace,
                    &transform,
                    &structure,
                    &structure,
                    1,
                    &source,
                    1.5,
                    3,
                    None,
                )
            })
            .unwrap()
            .unwrap();
        assert!(serial
            .iter()
            .zip(&parallel)
            .all(|(serial, parallel)| serial.to_bits() == parallel.to_bits()));

        let complex_source = source
            .iter()
            .map(|&value| Complex64::new(value, -0.5 * value))
            .collect::<Vec<_>>();
        let alpha = Complex64::new(1.5, -1.0);
        let serial = try_tree_transform_structure_overwrite_owned_raw(
            engine,
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            &transform,
            &structure,
            &structure,
            1,
            &complex_source,
            alpha,
            1,
            None,
        )
        .unwrap()
        .unwrap();
        let parallel = pool
            .install(|engine| {
                try_tree_transform_structure_overwrite_owned_raw(
                    engine,
                    &mut DefaultDenseExecutor::new(),
                    &mut TreeTransformWorkspace::default(),
                    &transform,
                    &structure,
                    &structure,
                    1,
                    &complex_source,
                    alpha,
                    3,
                    None,
                )
            })
            .unwrap()
            .unwrap();
        assert!(serial.iter().zip(&parallel).all(|(serial, parallel)| {
            serial.re.to_bits() == parallel.re.to_bits()
                && serial.im.to_bits() == parallel.im.to_bits()
        }));
    }

    #[test]
    fn owned_writer_without_a_disjoint_split_uses_the_serial_writer() {
        // What: a schedule with one runnable item has no disjoint split, so a
        // multi-worker budget collapses to the serial writer, which still
        // returns the owned result.
        let structure = canonical_structure(0);
        let transform = TreeTransformStructure::compile_structures(
            &structure,
            &structure,
            &[TreeTransformBlockSpec::single(0, 0, -2.0)],
        )
        .unwrap();
        let schedule = transform.parallel_schedule();
        let mut pool = pool(3);
        assert_eq!(
            pool.install(|engine| effective_tree_transform_threads(engine, schedule, 3)),
            1
        );
        let source = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let actual = pool
            .install(|engine| {
                try_tree_transform_structure_overwrite_owned_raw(
                    engine,
                    &mut DefaultDenseExecutor::new(),
                    &mut TreeTransformWorkspace::default(),
                    &transform,
                    &structure,
                    &structure,
                    1,
                    &source,
                    1.5,
                    3,
                    None,
                )
            })
            .unwrap()
            .unwrap();
        assert_eq!(actual, source.map(|value| -3.0 * value));
    }

    struct FailingGemm;

    impl DenseExecutor for FailingGemm {
        fn svd(
            &mut self,
            _engine: &mut dyn Engine,
            _: DenseRead<'_>,
        ) -> Result<Vec<DenseTensor>, DenseError> {
            unreachable!("owned transform replay never factorizes")
        }

        fn qr(
            &mut self,
            _engine: &mut dyn Engine,
            _: DenseRead<'_>,
        ) -> Result<Vec<DenseTensor>, DenseError> {
            unreachable!("owned transform replay never factorizes")
        }

        fn eigh(
            &mut self,
            _engine: &mut dyn Engine,
            _: DenseRead<'_>,
        ) -> Result<Vec<DenseTensor>, DenseError> {
            unreachable!("owned transform replay never factorizes")
        }

        fn dot_general_into(
            &mut self,
            _engine: &mut dyn Engine,
            _: DenseWrite<'_>,
            _: DenseRead<'_>,
            _: DenseRead<'_>,
            _: &DenseDotConfig,
        ) -> Result<(), DenseError> {
            unreachable!("owned transform replay uses the grouped GEMM")
        }

        fn matmul_batch_axpby_into(
            &mut self,
            _engine: &mut dyn Engine,
            _: DenseWrite<'_>,
            _: DenseRead<'_>,
            _: DenseRead<'_>,
            _: &[DenseGemmBatchJob],
            _: &[usize],
            _: DenseScalar,
            _: DenseScalar,
        ) -> Result<(), DenseError> {
            Err(DenseError::Backend {
                backend: DenseBackend::Tenferro,
                op: "matmul_batch_axpby_into",
                message: "injected recoupling failure".to_string(),
            })
        }
    }

    #[test]
    fn owned_writer_error_after_parallel_singles_returns_err_without_publishing() {
        // What: a GEMM failure after the parallel singles phase has written
        // part of the uninitialised buffer surfaces as `Err`; no owned payload
        // escapes and nothing panics on the joined workers.
        let (structure, transform, source) = mixed_fixture();
        let result = pool(3).install(|engine| {
            try_tree_transform_structure_overwrite_owned_raw(
                engine,
                &mut FailingGemm,
                &mut TreeTransformWorkspace::default(),
                &transform,
                &structure,
                &structure,
                1,
                &source,
                1.0,
                3,
                None,
            )
        });
        assert!(matches!(
            result,
            Err(OperationError::Dense(DenseError::Backend { .. }))
        ));
    }

    #[test]
    fn owned_writer_rejects_padding_and_out_of_range_split_before_allocation() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        // What: a holey destination and an out-of-range split receive no owned
        // result, leaving the caller on the initialized fallback path.
        let canonical = canonical_structure(0);
        let padded = canonical_structure(1);
        let padded_transform = TreeTransformStructure::compile_structures(
            &padded,
            &canonical,
            &[TreeTransformBlockSpec::single(0, 0, 1.0)],
        )
        .unwrap();
        let mut dense = DefaultDenseExecutor::new();
        let mut workspace = TreeTransformWorkspace::default();
        assert!(try_tree_transform_structure_overwrite_owned_raw(
            engine,
            &mut dense,
            &mut workspace,
            &padded_transform,
            &padded,
            &canonical,
            1,
            &[1.0; 6],
            1.0,
            1,
            None,
        )
        .unwrap()
        .is_none());

        let canonical_transform = TreeTransformStructure::compile_structures(
            &canonical,
            &canonical,
            &[TreeTransformBlockSpec::single(0, 0, 1.0)],
        )
        .unwrap();
        assert!(try_tree_transform_structure_overwrite_owned_raw(
            engine,
            &mut dense,
            &mut workspace,
            &canonical_transform,
            &canonical,
            &canonical,
            3,
            &[1.0; 6],
            1.0,
            1,
            None,
        )
        .unwrap()
        .is_none());
    }
}
