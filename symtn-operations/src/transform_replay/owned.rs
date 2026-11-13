use super::*;

/// Internal, unstable owned-output path for the built-in host executor:
/// allocates the destination uninitialised and writes every element exactly
/// once. Returns `Ok(None)` without allocating output when the destination
/// does not have a proof of exact physical overwrite coverage.
///
/// The writer runs on the same route as the initialised replay ([`replay`]).
/// Nothing is published unless every write succeeded. A forced batch policy `dense` cannot honor is refused first,
/// as the initialised replay refuses it, even where `Ok(None)` would follow.
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
    mut profile: Option<&mut TreeTransformReplayProfile>,
) -> Result<Option<Vec<D>>, OperationError>
where
    E: DenseExecutor,
    D: DenseRecouplingScalar + RecouplingCoefficientAction<C> + ConjugateValue,
    C: Copy + Sync,
{
    refuse_unhonored_policy(engine, dense)?;
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
    let task = structure.task_view()?;
    task.validate_workspace_requirements::<D>()?;
    if let (Some(profile), Some(start)) = (profile.as_deref_mut(), total_start) {
        profile.validate += start.elapsed();
    }
    if !task.recoupling_plan().is_empty() {
        let start = profile.as_ref().map(|_| std::time::Instant::now());
        if ensure_recoupling_coefficients(workspace, task, structure.identity_marker())? {
            if let (Some(profile), Some(start)) = (profile.as_deref_mut(), start) {
                profile.multi_coefficient_prepare += start.elapsed();
            }
        }
    }
    let writer = OwnedWriter {
        task: ReplayParts::new(task),
        alpha,
    };
    let output = initialize_owned::<D, OperationError>(proof.required_len, |dst_data| {
        replay(
            engine,
            dense,
            workspace,
            task,
            structure.parallel_schedule(),
            dst_data,
            src_data,
            &writer,
            &mut (),
            profile.as_deref_mut(),
        )
    })?;
    if let (Some(profile), Some(start)) = (profile, total_start) {
        profile.total += start.elapsed();
    }
    Ok(Some(output))
}

/// Writes one role into uninitialized memory: its compiled plan across
/// `exec` when [`admitted_plan`] admits one, the span leaf otherwise.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn write_role_uninit<D, F>(
    exec: Exec<'_>,
    gate: &mut LeafGate<'_>,
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
    if let Some(plan) = admitted_plan(role, elements, &exec, gate) {
        return planned_write_uninit(plan, role, dst, src, dst_offset, src_offset, exec, map);
    }
    let Some(index) = fused_index.get_mut(..role.dims().len()) else {
        return Err(OperationError::InvalidArgument {
            message: "fused traversal scratch is shorter than the normalized rank",
        });
    };
    span::<Uninit, _, D>(dst, src, role, dst_offset, src_offset, index, &Write(map));
    Ok(())
}

#[cfg(test)]
mod owned_overwrite_tests {
    use super::*;
    use crate::test_numerics::numerics::nan_folded_bits;
    use crate::{StridedHostKernelAdapter, TreeTransformBlockSpec};
    use num_complex::Complex64;
    use symtn_core::{
        BlockKey, BlockSpec, FusionProductSpace, FusionTreeHomSpace, FusionTreePairKey, SectorLeg,
        ZNFusionRule, ZNIrrep,
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
                    .map(|v| (nan_folded_bits(v.re), nan_folded_bits(v.im)))
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
                [
                    (ZNIrrep::new(0, 2).unwrap(), 1),
                    (ZNIrrep::new(1, 2).unwrap(), 1),
                ],
                false,
            )]),
            FusionProductSpace::new([SectorLeg::new(
                [
                    (ZNIrrep::new(0, 2).unwrap(), 1),
                    (ZNIrrep::new(1, 2).unwrap(), 1),
                ],
                false,
            )]),
        )
        .coupled_subblock_structure(&ZNFusionRule::new(2).unwrap(), 1, [vec![1, 1], vec![1, 1]])
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
                &ZNFusionRule::new(2).unwrap(),
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
        .coupled_subblock_structure_from_leg_degeneracies(&ZNFusionRule::new(2).unwrap())
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
        )
        .unwrap();
        assert_eq!(initialized, expected);
    }

    fn pool(threads: usize) -> crate::host_pool::TestPool {
        crate::host_pool::test_pool(threads)
    }

    fn z2_leg(even: usize, odd: usize) -> SectorLeg {
        SectorLeg::new(
            [
                (ZNIrrep::new(0, 2).unwrap(), even),
                (ZNIrrep::new(1, 2).unwrap(), odd),
            ],
            false,
        )
    }

    /// Z2 rank (1, 3): one codomain tree per coupled sector, so the eight
    /// 24-element blocks are contiguous column slices.
    /// Singles and Multi groups interleave in destination order: three
    /// runnable singles and five pack columns.
    fn mixed_fixture() -> (Arc<BlockStructure>, TreeTransformStructure<f64>, Vec<f64>) {
        let space = FusionTreeHomSpace::new(
            FusionProductSpace::new([z2_leg(3, 3)]),
            FusionProductSpace::new([z2_leg(2, 2), z2_leg(2, 2), z2_leg(2, 2)]),
        )
        .coupled_subblock_structure(&ZNFusionRule::new(2).unwrap(), 1, vec![vec![3, 2, 2, 2]; 8])
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
    fn owned_writer_on_many_workers_is_bit_identical_to_one_worker() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        // What: on a multi-worker engine the owned writer runs the chunk
        // replay and produces exactly the one-worker owned bytes for real
        // and complex data, which in turn equal the initialised overwrite
        // oracle.
        let (structure, transform, source) = mixed_fixture();
        let mut pool = pool(3);

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
    fn owned_writer_of_one_single_on_a_wide_engine_returns_the_result() {
        // What: a structure of one Single replays on a multi-worker engine
        // and still returns the owned result.
        let structure = canonical_structure(0);
        let transform = TreeTransformStructure::compile_structures(
            &structure,
            &structure,
            &[TreeTransformBlockSpec::single(0, 0, -2.0)],
        )
        .unwrap();
        let mut pool = pool(3);
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
                    None,
                )
            })
            .unwrap()
            .unwrap();
        assert_eq!(actual, source.map(|value| -3.0 * value));
    }

    #[test]
    fn owned_overwrite_of_only_inactive_blocks_returns_zeros() {
        // What: an owned overwrite whose one canonical U(1) charge-zero
        // `[2, 3]` block has no active spec returns six zeros at one and at
        // three workers: the inactive role's rank sizes the traversal
        // scratch even though no active role has one.
        use symtn_core::{U1FusionRule, U1Irrep};
        let structure = FusionTreeHomSpace::from_sectors(
            [(U1Irrep::new(0).unwrap(), 2)],
            [(U1Irrep::new(0).unwrap(), 3)],
        )
        .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
        .unwrap();
        let transform =
            TreeTransformStructure::<f64>::compile_structures(&structure, &structure, &[]).unwrap();
        let source = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        for threads in [1, 3] {
            let actual = pool(threads)
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
                        1.0,
                        None,
                    )
                })
                .unwrap();
            assert_eq!(actual, Some(vec![0.0; 6]), "threads {threads}");
        }
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
            unreachable!("owned transform replay uses the accumulate-form GEMM")
        }

        fn matmul_axpby_into(
            &mut self,
            _engine: &mut dyn Engine,
            _: DenseWrite<'_>,
            _: DenseRead<'_>,
            _: DenseRead<'_>,
            _: DenseScalar,
            _: DenseScalar,
        ) -> Result<(), DenseError> {
            Err(DenseError::Backend {
                backend: DenseBackend::Tenferro,
                op: "matmul_axpby_into",
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
            None,
        )
        .unwrap()
        .is_none());
    }
}
