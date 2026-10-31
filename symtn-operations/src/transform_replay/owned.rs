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
) -> Result<Option<Vec<D>>, OperationError>
where
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
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
    workspace.prepare_fused_indices(threads, max_fused_rank)?;

    initialize_owned(proof.required_len, |dst_data| {
        let recoupling_plan = structure.recoupling_plan();
        let storage_conjugate = structure.storage_conjugate();
        let coefficients = task.single_coefficients();

        for &layout_index in structure.inactive_destination_layouts() {
            write_uninit_layout_zero(layouts, layout_index, dst_data)?;
        }
        let blocks = structure.blocks();
        // Singles split on the schedule's slice-disjoint boundaries exactly as
        // `replay_single_blocks` does; a non-disjoint schedule stays serial.
        let singles_threads = if schedule.singles_slice_disjoint {
            threads
        } else {
            1
        };
        replay_region(engine, singles_threads, |exec| {
            split_join_uninit(
                &schedule.singles,
                dst_data,
                0,
                workspace.fused_indices.as_mut_slice(),
                max_fused_rank,
                singles_threads,
                exec,
                &|&item| layouts.destination_lo(scheduled_single(blocks, item)?.0),
                &|items, dst, dst_start, fused_index| {
                    for &item in items {
                        let (dst_layout, src_layout, coefficient) = scheduled_single(blocks, item)?;
                        write_uninit_layout_from_source(
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

        if recoupling_plan.is_empty() {
            return Ok(());
        }
        ensure_recoupling_coefficients(workspace, task, structure.identity_marker())?;
        let chunk_size = threads.max(1);
        let mut pack_cursor = 0;
        for (chunk_index, chunk) in recoupling_plan.jobs().chunks(chunk_size).enumerate() {
            let (destination_start, scatter_slice_disjoint) = replay_multi_chunk(
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
                None,
            )?;
            let packed = workspace.packed.destination().as_slice();
            let scatter_columns = &schedule.scatter_columns;
            let scatter_groups = &schedule.scatter_groups;
            // Groups split on their ordered destination ranges exactly as
            // `replay_scatter_groups` does; columns inside a group stay serial.
            let scatter_threads = if scatter_slice_disjoint { threads } else { 1 };
            replay_region(engine, scatter_threads, |exec| {
                split_join_uninit(
                    &workspace.chunk_scatter_groups,
                    dst_data,
                    0,
                    workspace.fused_indices.as_mut_slice(),
                    max_fused_rank,
                    scatter_threads,
                    exec,
                    &|&group| Ok(scatter_columns[scatter_groups[group].columns.start].dst_lo),
                    &|groups, dst, dst_start, fused_index| {
                        for &group in groups {
                            for item in &scatter_columns[scatter_groups[group].columns.clone()] {
                                write_uninit_layout_from_packed(
                                    layouts,
                                    item.dst_layout,
                                    dst,
                                    dst_start,
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
            })??;
        }
        debug_assert_eq!(pack_cursor, schedule.pack_columns.len());
        Ok(())
    })
    .map(Some)
}

/// Recursive `replay_join` over an uninitialised destination, mirroring the
/// split tree of `replay_single_blocks` / `replay_scatter_groups`: `items`
/// are sorted by destination offset and `dst_lo(item)` is the absolute start
/// of each item's destination range; `leaf` writes `items` into `dst`, whose
/// first element is absolute offset `dst_start`, using one worker's traversal
/// scratch.
///
/// No `unsafe` is needed here: `split_at_mut` at `dst_lo(items[middle])`
/// yields two non-overlapping `&mut [MaybeUninit<D>]`, which the compiler
/// proves independent. The callers pass `threads > 1` only when the compiled
/// schedule proved the items' destination ranges pairwise slice-disjoint
/// (`singles_slice_disjoint` / `scatter_groups[i].slice_disjoint` plus the
/// per-chunk group ordering check), so the boundary lies inside `dst` and no
/// item of one half touches the other half. Each leaf writes only its own
/// half; full coverage of the buffer is the caller's `PhysicalOverwriteProof`,
/// and `initialize_owned` exposes the initialised length only after every
/// join has returned `Ok`.
#[allow(clippy::too_many_arguments)]
fn split_join_uninit<I, D>(
    items: &[I],
    dst: &mut [MaybeUninit<D>],
    dst_start: isize,
    fused_indices: &mut [usize],
    max_fused_rank: usize,
    threads: usize,
    exec: Exec<'_>,
    dst_lo: &(impl Fn(&I) -> Result<isize, OperationError> + Sync),
    leaf: &(impl Fn(&[I], &mut [MaybeUninit<D>], isize, &mut [usize]) -> Result<(), OperationError>
          + Sync),
) -> Result<(), OperationError>
where
    I: Sync,
    D: Send + Sync,
{
    if items.is_empty() {
        return Ok(());
    }
    if threads <= 1 || items.len() == 1 {
        return leaf(items, dst, dst_start, &mut fused_indices[..max_fused_rank]);
    }

    let middle = parallel_split(items.len(), threads);
    let boundary = dst_lo(&items[middle])?;
    let split =
        usize::try_from(boundary - dst_start).map_err(|_| OperationError::ElementCountOverflow)?;
    let (left_data, right_data) = dst.split_at_mut(split);
    let (left_items, right_items) = items.split_at(middle);
    let left_threads = threads / 2;
    let right_threads = threads - left_threads;
    let (left_indices, mut right_indices) =
        split_lane_indices(fused_indices, left_threads, threads, max_fused_rank);
    let (left, right) = replay_join(
        exec,
        |exec| {
            split_join_uninit(
                left_items,
                left_data,
                dst_start,
                left_indices,
                max_fused_rank,
                left_threads,
                exec,
                dst_lo,
                leaf,
            )
        },
        |exec| {
            split_join_uninit(
                right_items,
                right_data,
                boundary,
                right_indices.get(),
                max_fused_rank,
                right_threads,
                exec,
                dst_lo,
                leaf,
            )
        },
    );
    left?;
    right
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
        + Mul<D, Output = D>
        + Zero
        + One
        + PartialEq
        + ConjugateValue
        + RecouplingCoefficientAction<C>,
    C: Copy,
{
    // The element op is chosen once per block, with the structural coefficient
    // still in its own type; see `TransformScale`.
    if scale.is_identity() {
        return write_uninit_layout_mapped(
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
        TransformScale::Structural(coefficient) => write_uninit_layout_mapped(
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
                    .scale_by_coefficient(coefficient)
            },
        ),
        TransformScale::Data(scale) => write_uninit_layout_mapped(
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
    D: Copy,
    F: Fn(D) -> D,
{
    write_fused_uninit(
        layouts.role(dst_index)?,
        dst,
        src,
        layouts.entry(dst_index).offset - dst_start,
        layouts.entry(src_index).offset,
        fused_index,
        map,
    )
}

#[allow(clippy::too_many_arguments)]
fn write_uninit_layout_from_packed<D>(
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
    D: Copy + Mul<D, Output = D> + Zero + One + PartialEq,
{
    // Why the identity arm: `1 * (inf + 0i)` is `inf + NaN i`, so the scatter
    // must copy rather than multiply when the caller's alpha is one. A zero
    // alpha gives `scale_value`'s exact zero even where the recoupling GEMM
    // left a NaN in `packed`, as TensorKit's scatter `tensoradd!` does.
    // The scatter role bakes src = packed (column-major) strides, so the
    // fused walk over `packed` starting at `packed_offset` reproduces the
    // column-major gather `packed[packed_offset + linear]`.
    let role = layouts.role(dst_index)?;
    let dst_offset = layouts.entry(dst_index).offset - dst_start;
    let src_offset = offset_to_isize(packed_offset)?;
    if alpha.is_one() {
        write_fused_uninit(
            role,
            dst,
            packed,
            dst_offset,
            src_offset,
            fused_index,
            |v| v,
        )
    } else {
        write_fused_uninit(
            role,
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
                vec![BlockSpec::with_key(key, vec![2, 3], vec![1, 2], offset).unwrap()],
            )
            .unwrap(),
        )
    }

    #[test]
    fn owned_writer_matches_initialized_oracle_for_real_and_complex() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        // What: canonical serial owned replay writes every physical value and
        // is byte-for-byte equal to the initialized overwrite oracle.
        let structure = canonical_structure(0);
        let transform = TreeTransformStructure::compile_structures(
            &structure,
            &structure,
            &[TreeTransformBlockSpec::single(0, 0, -2.0)],
        )
        .unwrap();

        let mut expected = vec![f64::NAN; 6];
        tree_transform_structure_overwrite_with_structural_recoupling_raw(
            engine,
            &mut StridedHostKernelAdapter::default(),
            &mut DefaultDenseExecutor::new(),
            &mut TreeTransformWorkspace::default(),
            &transform,
            &structure,
            &structure,
            &mut expected,
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
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
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            3.0,
            1,
        )
        .unwrap()
        .unwrap();
        assert_eq!(actual, expected);

        let complex_src = (1..=6)
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
        )
        .unwrap()
        .unwrap();
        let scale = Complex64::new(3.0, 1.0) * -2.0;
        assert_eq!(
            complex,
            complex_src.iter().map(|&v| scale * v).collect::<Vec<_>>()
        );
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
    fn owned_writer_zero_scales_write_zeros_over_non_finite_sources() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        // What: the uninitialized writer follows VectorInterface's
        // `scale(x, 0) = zero(x) * 0` (#1438) for a zero caller alpha and for
        // a zero structural coefficient in a Single block, and for a zero
        // alpha in a Multi scatter whose recoupling GEMM saw `inf`: every
        // output is `0.0`, as TensorKit's `permute!(tdst, tsrc, p, 0, 0)`
        // (and `scale(Inf, 0.0) == 0.0`) gives.
        let non_finite = [f64::INFINITY, f64::NAN, f64::NEG_INFINITY, 1.0, -0.0, 2.0];
        let structure = canonical_structure(0);
        for (alpha, coefficient) in [(0.0, -2.0), (1.0, 0.0), (3.0, 0.0)] {
            let transform = TreeTransformStructure::compile_structures(
                &structure,
                &structure,
                &[TreeTransformBlockSpec::single(0, 0, coefficient)],
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
                &non_finite,
                alpha,
                1,
            )
            .unwrap()
            .unwrap();
            assert!(
                actual.iter().all(|&value| value == 0.0),
                "alpha {alpha}, coefficient {coefficient}: {actual:?}"
            );
        }

        let space = FusionTreeHomSpace::new(
            FusionProductSpace::new([z2_leg(1, 1)]),
            FusionProductSpace::new([z2_leg(1, 1)]),
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
            &[f64::INFINITY, f64::NEG_INFINITY],
            0.0,
            1,
        )
        .unwrap()
        .unwrap();
        assert_eq!(actual, [0.0, 0.0]);
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
        // uninitialised destination on the compiled slice-disjoint boundaries
        // and produces exactly the serial owned bytes for real and complex
        // data, which in turn equal the initialised overwrite oracle.
        let (structure, transform, source) = mixed_fixture();
        let schedule = transform.parallel_schedule();
        assert!(schedule.singles_slice_disjoint);
        assert!(schedule
            .scatter_groups
            .iter()
            .all(|group| group.slice_disjoint));
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
        // returns the owned result. Interleaved (non-slice-disjoint) layouts
        // cannot reach this writer at all: the physical overwrite proof
        // requires every destination layout to be contiguous.
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
        )
        .unwrap()
        .is_none());
    }
}
