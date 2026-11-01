//! Left/right null-space tests.

use super::*;

fn checked_spy_null(
    left: bool,
    dense: &mut impl DenseExecutor,
    input: &BoundDynamicTensorRef<'_, LateGenericSpy, f64>,
) -> Result<BoundDynFactor<LateGenericSpy, f64>, CheckedGenericFactorPlanError<LateGenericError>> {
    let engine = &mut symtn_dense::CpuEngine::serial();
    if left {
        left_null_dyn_checked_generic(engine, dense, input)
    } else {
        right_null_dyn_checked_generic(engine, dense, input)
    }
}

/// A checked Generic input whose blocks are longer on the requested null
/// side, so every coupled sector runs one QR.
fn null_side_generic_input(left: bool) -> BoundDynamicFusionMapSpace<FactorGenericRule> {
    if left {
        generic_factorization_input().0
    } else {
        wide_generic_factorization_input()
    }
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_null_admits_only_after_all_dense_work_and_keeps_exact_authority() {
    // What: both null sides finish every sector QR before their output
    // admission, whose late error publishes no factor.
    for left in [true, false] {
        let source = null_side_generic_input(left);
        let data = vec![0.0; source.space().required_len().unwrap()];
        let complete_provider = Arc::new(LateGenericSpy {
            rule: FactorGenericRule,
            fail_at: usize::MAX,
            calls: Cell::new(0),
            identity: RuleIdentity::new_unique::<LateGenericSpy>(),
        });
        let complete_space = bind_to_spy(&source, &complete_provider);
        let complete_input = BoundDynamicTensorRef::try_new(&complete_space, &data).unwrap();
        let mut complete_dense = ScriptedExecutor::<CountingDense>::default();
        let factor = checked_spy_null(left, &mut complete_dense, &complete_input).unwrap();
        assert!(Arc::ptr_eq(
            factor.space().provider_arc(),
            &complete_provider
        ));
        let final_call = complete_provider.calls.get();
        assert!(final_call > 1);
        assert_eq!(complete_dense.counts().of(&[Op::Svd, Op::SvdInto]), 0);
        assert!(complete_dense.counts().qr_into > 1);

        let failing_provider = Arc::new(LateGenericSpy {
            rule: FactorGenericRule,
            fail_at: final_call,
            calls: Cell::new(0),
            identity: RuleIdentity::new_unique::<LateGenericSpy>(),
        });
        let failing_space = bind_to_spy(&source, &failing_provider);
        let failing_input = BoundDynamicTensorRef::try_new(&failing_space, &data).unwrap();
        let before = failing_input.data().to_vec();
        let mut failing_dense = ScriptedExecutor::<CountingDense>::default();
        assert!(matches!(
            checked_spy_null(left, &mut failing_dense, &failing_input),
            Err(CheckedGenericFactorPlanError::Provider(LateGenericError(call)))
                if call == final_call
        ));
        assert_eq!(
            failing_dense.counts().qr_into,
            complete_dense.counts().qr_into
        );
        assert_eq!(failing_input.data(), before);
        assert!(Arc::ptr_eq(
            failing_input.space().provider_arc(),
            &failing_provider
        ));
    }
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_null_dense_failure_never_reaches_output_admission() {
    // What: a later sector QR failure stops after structural preflight and
    // before the checked output builder or any scatter can run.
    for left in [true, false] {
        let source = null_side_generic_input(left);
        let data = vec![0.0; source.space().required_len().unwrap()];
        let provider = Arc::new(LateGenericSpy {
            rule: FactorGenericRule,
            fail_at: usize::MAX,
            calls: Cell::new(0),
            identity: RuleIdentity::new_unique::<LateGenericSpy>(),
        });
        let checked = bind_to_spy(&source, &provider);
        let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
        let before = input.data().to_vec();
        let expected_preflight = {
            let probe = LateGenericSpy {
                rule: FactorGenericRule,
                fail_at: usize::MAX,
                calls: Cell::new(0),
                identity: RuleIdentity::new_unique::<LateGenericSpy>(),
            };
            let side = if left {
                source.space().homspace().codomain()
            } else {
                source.space().homspace().domain()
            };
            coupled_sector_block_dimensions_generic_checked(side, &probe).unwrap();
            probe.calls.get()
        };
        assert!(matches!(
            checked_spy_null(
                left,
                &mut ScriptedExecutor::<FailSecondNullQr>::default(),
                &input
            ),
            Err(CheckedGenericFactorPlanError::Operation(
                OperationError::Dense(_)
            ))
        ));
        assert_eq!(provider.calls.get(), expected_preflight);
        assert_eq!(input.data(), before);
    }
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_disjoint_null_is_identity_without_dense_calls() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: disjoint support returns complete identity bases for both sides
    // and never enters dense work.
    let x = SectorId::new(1);
    let vacuum = SectorId::new(0);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(x, 2)], false)]),
        FusionProductSpace::new([SectorLeg::new([(vacuum, 3)], false)]),
    );
    let source = BoundDynamicFusionMapSpace::from_final_homspace_generic(
        Arc::new(FactorGenericRule),
        homspace,
    )
    .unwrap();
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
        identity: RuleIdentity::new_unique::<LateGenericSpy>(),
    });
    let checked = bind_to_spy(&source, &provider);
    let data = vec![0.0; checked.space().required_len().unwrap()];
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let left = left_null_dyn_checked_generic(
        engine,
        &mut ScriptedExecutor::new(RejectExecutorCalls),
        &input,
    )
    .unwrap();
    let right = right_null_dyn_checked_generic(
        engine,
        &mut ScriptedExecutor::new(RejectExecutorCalls),
        &input,
    )
    .unwrap();
    assert_eq!(left.data(), &[1.0, 0.0, 0.0, 1.0]);
    assert_eq!(right.data().len(), 9);
    for column in 0..3 {
        for row in 0..3 {
            assert_eq!(right.data()[row + 3 * column], f64::from(row == column));
        }
    }
    assert!(Arc::ptr_eq(left.space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(right.space().provider_arc(), &provider));
}

#[test]
fn null_spaces_are_orthonormal_and_annihilate_the_tensor() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rule = Z2FusionRule;
    let sectors = [SectorId::new(0), SectorId::new(1)];
    let degeneracy = 2usize;
    let leg = || SectorLeg::new(sectors.iter().map(|&sector| (sector, degeneracy)), false);

    // Tall map (2 codomain legs, 1 domain leg): nontrivial left null space.
    let tall_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg()]),
    );
    let tall_space = DynamicFusionMapSpace::from_final_homspace(&rule, tall_hom).unwrap();
    let len = tall_space.required_len().unwrap();
    let tall = TestTensor::<f64>::from_vec_with_fusion_space(
        (0..len).map(|i| ((i * 3 + 1) % 13) as f64 - 6.0).collect(),
        tall_space,
    )
    .unwrap();
    let mut dense_executor = symtn_dense::DefaultDenseExecutor::new();
    let null = left_null(
        engine,
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &tall),
    )
    .unwrap();

    let null_matrices = dense_sector_matrices(2, &null);
    assert!(!null_matrices.is_empty());
    assert_orthonormal_columns(&null_matrices);
    let tensor_matrices = dense_sector_matrices(2, &tall);
    for (sector, n_rows, n_cols, n) in &null_matrices {
        let (_, a_rows, a_cols, a) = tensor_matrices
            .iter()
            .find(|(candidate, ..)| candidate == sector)
            .expect("tensor sector present");
        assert_eq!(n_rows, a_rows);
        assert_eq!(*n_cols, a_rows - (*a_rows).min(*a_cols));
        // N^T A = 0.
        for null_col in 0..*n_cols {
            for a_col in 0..*a_cols {
                let mut dot = 0.0;
                for row in 0..*a_rows {
                    dot += n[row + n_rows * null_col] * a[row + a_rows * a_col];
                }
                assert!(dot.abs() < 1e-9, "left null failed: {dot}");
            }
        }
    }

    // Wide map (1 codomain leg, 2 domain legs): nontrivial right null space.
    let wide_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let wide_space = DynamicFusionMapSpace::from_final_homspace(&rule, wide_hom).unwrap();
    let len = wide_space.required_len().unwrap();
    let wide = TestTensor::<f64>::from_vec_with_fusion_space(
        (0..len).map(|i| ((i * 5 + 2) % 11) as f64 - 5.0).collect(),
        wide_space,
    )
    .unwrap();
    let null = right_null(
        engine,
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &wide),
    )
    .unwrap();

    let null_matrices = dense_sector_matrices(1, &null);
    assert!(!null_matrices.is_empty());
    let tensor_matrices = dense_sector_matrices(1, &wide);
    for (sector, n_rows, n_cols, n) in &null_matrices {
        let (_, a_rows, a_cols, a) = tensor_matrices
            .iter()
            .find(|(candidate, ..)| candidate == sector)
            .expect("tensor sector present");
        assert_eq!(n_cols, a_cols);
        assert_eq!(*n_rows, a_cols - (*a_cols).min(*a_rows));
        // Rows of N are orthonormal: N N^T = I.
        for left in 0..*n_rows {
            for right in 0..*n_rows {
                let mut dot = 0.0;
                for col in 0..*n_cols {
                    dot += n[left + n_rows * col] * n[right + n_rows * col];
                }
                let expected = if left == right { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1e-9);
            }
        }
        // A N^T = 0 (rows of N span the kernel).
        for a_row in 0..*a_rows {
            for null_row in 0..*n_rows {
                let mut dot = 0.0;
                for col in 0..*a_cols {
                    dot += a[a_row + a_rows * col] * n[null_row + n_rows * col];
                }
                assert!(dot.abs() < 1e-9, "right null failed: {dot}");
            }
        }
    }
}

#[test]
fn square_null_spaces_are_empty_whatever_the_rank() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: the default null dimension is `(rows - cols)+` from the space
    // alone (MatrixAlgebraKit `qr_null!`/`lq_null!`), so square sectors,
    // including zero and rank-one ones, have no null directions and run no
    // dense work.
    let rule = Z2FusionRule;
    let mut dense = ScriptedExecutor::new(RejectExecutorCalls);
    for matrix in [
        one_sector_matrix(vec![0.0; 4]),
        one_sector_matrix(vec![1.0, 2.0, 1.0, 2.0]),
        one_sector_matrix(vec![1.0, 0.0, 0.0, 1.0e-300]),
    ] {
        let input = bound_tensor(Arc::new(rule), &matrix);
        let left = left_null(engine, &mut dense, &input.as_ref()).unwrap();
        let right = right_null(engine, &mut dense, &input.as_ref()).unwrap();
        assert_eq!(left.structure().block_count(), 0);
        assert!(left.data().is_empty());
        assert_eq!(right.structure().block_count(), 0);
        assert!(right.data().is_empty());
    }
}

/// Left null of a tall `3 x 2` and right null of a wide `2 x 3` sector in
/// dtype `D`: one null direction each, orthonormal and annihilating the
/// input under the conjugate-transpose pairing, whatever the rank.
fn assert_rectangular_null_spaces_ignore_rank<D: FactorScalar>(
    tall: Vec<D>,
    wide: Vec<D>,
    tol: f64,
) {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rule = Z2FusionRule;
    let mut dense = ScriptedExecutor::<CountingDense>::default();
    let tall = one_sector_rectangular_matrix(tall, 3, 2);
    let input = bound_tensor(Arc::new(rule), &tall);
    let n = left_null(engine, &mut dense, &input.as_ref()).unwrap();
    assert_eq!(n.structure().block(0).unwrap().shape(), &[3, 1]);
    let norm: f64 = n.data().iter().map(|v| v.widen_complex().norm_sqr()).sum();
    assert!((norm - 1.0).abs() < tol);
    for col in 0..2 {
        let dot: num_complex::Complex64 = (0..3)
            .map(|row| {
                n.data()[row].widen_complex().conj() * tall.data()[row + 3 * col].widen_complex()
            })
            .sum();
        assert!(dot.norm() < tol, "left null failed: {dot}");
    }
    assert_eq!(
        right_null(engine, &mut dense, &input.as_ref())
            .unwrap()
            .structure()
            .block_count(),
        0
    );

    let wide = one_sector_rectangular_matrix(wide, 2, 3);
    let input = bound_tensor(Arc::new(rule), &wide);
    let n = right_null(engine, &mut dense, &input.as_ref()).unwrap();
    assert_eq!(n.structure().block(0).unwrap().shape(), &[1, 3]);
    let norm: f64 = n.data().iter().map(|v| v.widen_complex().norm_sqr()).sum();
    assert!((norm - 1.0).abs() < tol);
    for row in 0..2 {
        // A N^H = 0.
        let dot: num_complex::Complex64 = (0..3)
            .map(|col| {
                wide.data()[row + 2 * col].widen_complex() * n.data()[col].widen_complex().conj()
            })
            .sum();
        assert!(dot.norm() < tol, "right null failed: {dot}");
    }
    assert_eq!(
        left_null(engine, &mut dense, &input.as_ref())
            .unwrap()
            .structure()
            .block_count(),
        0
    );
    assert_eq!(dense.counts().of(&[Op::Svd, Op::SvdInto]), 0);
    assert_eq!(dense.counts().qr_into, 2);
}

#[test]
fn rectangular_null_spaces_keep_the_shape_deficit_only() {
    // What: tall/wide sectors keep exactly `|rows - cols|` directions on the
    // longer side for full-rank, rank-one, near-cutoff and zero inputs, in
    // every factorization dtype; the rank deficit adds none.
    for (tall, wide) in [
        (
            vec![1.0, 2.0, 3.0, -2.0, 0.5, 1.0],
            vec![1.0, -2.0, 2.0, 0.5, 3.0, 1.0],
        ),
        (
            vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0],
            vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0],
        ),
        (
            vec![1.0, 0.0, 0.0, 0.0, 1.0e-20, 0.0],
            vec![1.0, 0.0, 0.0, 1.0e-20, 0.0, 0.0],
        ),
        (vec![0.0; 6], vec![0.0; 6]),
    ] {
        assert_rectangular_null_spaces_ignore_rank::<f64>(tall.clone(), wide.clone(), 1e-12);
        let single = |v: &Vec<f64>| v.iter().map(|&x| x as f32).collect::<Vec<_>>();
        assert_rectangular_null_spaces_ignore_rank::<f32>(single(&tall), single(&wide), 1e-5);
        // A complex payload with phases on both factors, so the pairing
        // must conjugate.
        let phased = |v: &Vec<f64>| {
            v.iter()
                .enumerate()
                .map(|(i, &x)| Complex64::new(x, 0.5 * x * (i as f64 - 2.0)))
                .collect::<Vec<_>>()
        };
        assert_rectangular_null_spaces_ignore_rank::<Complex64>(
            phased(&tall),
            phased(&wide),
            1e-12,
        );
        let phased32 = |v: &Vec<f64>| {
            phased(v)
                .into_iter()
                .map(|z| Complex32::new(z.re as f32, z.im as f32))
                .collect::<Vec<_>>()
        };
        assert_rectangular_null_spaces_ignore_rank::<Complex32>(
            phased32(&tall),
            phased32(&wide),
            1e-5,
        );
    }
}

fn assert_disjoint_null_spaces_keep_structural_directions<D: FactorScalar>() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let assert_identity = |data: &[D], order: usize| {
        for col in 0..order {
            for row in 0..order {
                let expected = if row == col { 1.0 } else { 0.0 };
                assert!((data[row + order * col].widen_complex().re - expected).abs() < 1e-12);
                assert!(data[row + order * col].widen_complex().im.abs() < 1e-12);
            }
        }
    };
    let provider = Arc::new(U1FusionRule);
    let tensor = u1_cross_space_map::<D>(&[(1, 2)], &[(0, 3)]);
    let input = bound_tensor(Arc::clone(&provider), &tensor);

    let buffer_builds_rec = crate::probe::record();
    let left = left_null(
        engine,
        &mut ScriptedExecutor::new(RejectExecutorCalls),
        &input.as_ref(),
    )
    .unwrap();
    assert_eq!(buffer_builds_rec.factor_buffer_builds(), (1, 0));
    assert_eq!(left.structure().block_count(), 1);
    assert_eq!(left.structure().block(0).unwrap().shape(), &[2, 2]);
    assert_eq!(left.data().len(), 4);
    assert_identity(left.data(), 2);
    assert!(Arc::ptr_eq(left.space().provider_arc(), &provider));

    let buffer_builds_rec = crate::probe::record();
    let right = right_null(
        engine,
        &mut ScriptedExecutor::new(RejectExecutorCalls),
        &input.as_ref(),
    )
    .unwrap();
    assert_eq!(buffer_builds_rec.factor_buffer_builds(), (0, 1));
    assert_eq!(right.structure().block_count(), 1);
    assert_eq!(right.structure().block(0).unwrap().shape(), &[3, 3]);
    assert_eq!(right.data().len(), 9);
    assert_identity(right.data(), 3);
    assert!(Arc::ptr_eq(right.space().provider_arc(), &provider));
}

#[test]
fn disjoint_null_spaces_keep_all_structural_directions_without_dense_work() {
    // What: a zero map between disjoint supports has the whole codomain/domain
    // as its left/right null space and builds only the requested factor.
    assert_disjoint_null_spaces_keep_structural_directions::<f64>();
    assert_disjoint_null_spaces_keep_structural_directions::<Complex64>();
}

#[test]
fn null_zero_only_input_normalizes_to_empty_without_dense_work() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let tensor = rectangular_svd_tensor(0, 0);
    let input = bound_tensor(Arc::new(Z2FusionRule), &tensor);
    let mut dense = ScriptedExecutor::new(RejectExecutorCalls);

    let left = left_null(engine, &mut dense, &input.as_ref()).unwrap();
    let right = right_null(engine, &mut dense, &input.as_ref()).unwrap();

    assert_eq!(left.structure().block_count(), 0);
    assert!(left.data().is_empty());
    assert_eq!(right.structure().block_count(), 0);
    assert!(right.data().is_empty());
}

#[test]
fn unmatched_null_sectors_coexist_with_a_full_rank_matched_sector() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: a matched square sector has no null directions and runs no
    // dense work, while side-only sectors survive as identity bases.
    let provider = Arc::new(U1FusionRule);
    let tensor = u1_cross_space_map::<f64>(&[(0, 1), (1, 2)], &[(0, 1), (2, 3)]);
    let input = bound_tensor(Arc::clone(&provider), &tensor);

    let mut dense = ScriptedExecutor::new(RejectExecutorCalls);
    let left = left_null(engine, &mut dense, &input.as_ref()).unwrap();
    assert_eq!(left.structure().block_count(), 1);
    assert_eq!(left.structure().block(0).unwrap().shape(), &[2, 2]);

    let mut dense = ScriptedExecutor::new(RejectExecutorCalls);
    let right = right_null(engine, &mut dense, &input.as_ref()).unwrap();
    assert_eq!(right.structure().block_count(), 1);
    assert_eq!(right.structure().block(0).unwrap().shape(), &[3, 3]);
}

#[test]
fn null_space_second_sector_failure_builds_no_factor() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: all dense work finishes before the one requested factor is built.
    let provider = Arc::new(U1FusionRule);
    let tall = u1_cross_space_map::<f64>(&[(0, 3), (1, 3)], &[(0, 2), (1, 2)]);
    let wide = u1_cross_space_map::<f64>(&[(0, 2), (1, 2)], &[(0, 3), (1, 3)]);
    for (left, tensor) in [(true, &tall), (false, &wide)] {
        let before = tensor.data().to_vec();
        let input = bound_tensor(Arc::clone(&provider), tensor);
        let buffer_builds_rec = crate::probe::record();
        let mut dense = ScriptedExecutor::<FailSecondNullQr>::default();
        let result = if left {
            left_null(engine, &mut dense, &input.as_ref())
        } else {
            right_null(engine, &mut dense, &input.as_ref())
        };
        assert!(matches!(result, Err(OperationError::Dense(_))));
        assert_eq!(dense.counts().qr_into, 2);
        assert_eq!(buffer_builds_rec.factor_buffer_builds(), (0, 0));
        assert_eq!(tensor.data(), before);
    }
}

// Caller-thread allocation bytes.
#[allow(unsafe_code)]
#[path = "../../../tests/support/counting_alloc.rs"]
pub(crate) mod counting_alloc;

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

/// Runs every multiplicity-free full, null and eigen op on admitted input
/// regions; `general` must be an endomorphism.
fn multiplicity_free_full_family<R, D>(
    general: &BoundDynamicTensorRef<'_, R, D>,
    tall: &BoundDynamicTensorRef<'_, R, D>,
    hermitian: &BoundDynamicTensorRef<'_, R, D>,
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    for input in [general, tall] {
        qr_full_dyn(engine, &mut dense, input).unwrap();
        lq_full_dyn(engine, &mut dense, input).unwrap();
        left_null_dyn(engine, &mut dense, input).unwrap();
        right_null_dyn(engine, &mut dense, input).unwrap();
    }
    eig_full_dyn(engine, &mut dense, general).unwrap();
    eigh_full_dyn(engine, &mut dense, hermitian, HermitianTol::DEFAULT).unwrap();
}

fn multiplicity_free_tall_test_tensor<R>(rule: &R, sectors: &[SectorId]) -> TestTensor<f64>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
{
    let degeneracy = 2usize;
    let leg = || SectorLeg::new(sectors.iter().map(|&sector| (sector, degeneracy)), false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg()]),
    );
    let space = DynamicFusionMapSpace::from_final_homspace(rule, homspace).unwrap();
    let len = space.required_len().unwrap();
    TestTensor::<f64>::from_vec_with_fusion_space(
        (0..len)
            .map(|index| ((index * 13 + 7) % 31) as f64 * 0.25 - 3.5)
            .collect(),
        space,
    )
    .unwrap()
}

/// `source` in `D`; complex copies get a nonzero imaginary part unless
/// `real_only` (a Hermitian source stays Hermitian).
fn scalar_copy<D: FactorScalar>(source: &TestTensor<f64>, real_only: bool) -> TestTensor<D> {
    TestTensor::from_vec_with_fusion_space(
        source
            .data()
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                let imaginary = if real_only {
                    0.0
                } else {
                    ((index * 5 + 2) % 7) as f64 * 0.3 - 0.9
                };
                D::from_complex64(Complex64::new(value, imaginary))
            })
            .collect(),
        source.fusion_space().clone(),
    )
    .unwrap()
}

/// What: canonical inputs to the multiplicity-free full QR/LQ, null and
/// eigen ops pack nothing.
fn assert_multiplicity_free_full_family_borrows_input<R, D>(rule: R, sectors: &[SectorId])
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + Clone,
    D: FactorScalar,
{
    let provider = Arc::new(rule.clone());
    let general = scalar_copy::<D>(&tsvd_test_tensor(&rule, sectors), false);
    let tall = scalar_copy::<D>(&multiplicity_free_tall_test_tensor(&rule, sectors), false);
    let hermitian = scalar_copy::<D>(&hermitian_test_tensor(&rule, sectors), true);
    let bind = |tensor: &TestTensor<D>| bound_tensor(Arc::clone(&provider), tensor);
    let tall_bound = bound_tensor(Arc::clone(&provider), &tall);
    let tall_ref = tall_bound.as_ref();
    let tall_input = tall_ref.dynamic();

    let input_pack_bytes = |general: &TestTensor<D>, hermitian: &TestTensor<D>| {
        let (general, hermitian) = (bind(general), bind(hermitian));
        let (general, hermitian) = (general.as_ref(), hermitian.as_ref());
        let (general, hermitian) = (general.dynamic(), hermitian.dynamic());
        let input_pack_bytes_rec = crate::probe::record();
        multiplicity_free_full_family(&general, &tall_input, &hermitian);
        input_pack_bytes_rec.get(crate::probe::Event::InputPackBytes)
    };

    assert_eq!(input_pack_bytes(&general, &hermitian), 0);
}

#[test]
fn multiplicity_free_full_null_eigen_ops_borrow_canonical_input() {
    let u1 = [-1, 0, 1].map(|charge| U1Irrep::new(charge).sector_id());
    let su2 = [0, 1].map(|twice| SU2Irrep::from_twice_spin(twice).sector_id());
    let product =
        symtn_core::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule)
            .unwrap();
    let product_sectors = [
        product.encode_component_ids(SectorId::new(0), U1Irrep::new(0).sector_id()),
        product.encode_component_ids(SectorId::new(1), U1Irrep::new(1).sector_id()),
        product.encode_component_ids(SectorId::new(1), U1Irrep::new(-1).sector_id()),
    ];
    assert_multiplicity_free_full_family_borrows_input::<_, f64>(U1FusionRule, &u1);
    assert_multiplicity_free_full_family_borrows_input::<_, Complex64>(U1FusionRule, &u1);
    assert_multiplicity_free_full_family_borrows_input::<_, f64>(SU2FusionRule, &su2);
    assert_multiplicity_free_full_family_borrows_input::<_, Complex64>(SU2FusionRule, &su2);
    assert_multiplicity_free_full_family_borrows_input::<_, f64>(product.clone(), &product_sectors);
    assert_multiplicity_free_full_family_borrows_input::<_, Complex64>(product, &product_sectors);
}

/// Runs every checked-Generic full, null and eigen op on admitted input
/// regions.
fn checked_full_family<R, D>(
    general: &BoundDynamicTensorRef<'_, R, D>,
    tall: &BoundDynamicTensorRef<'_, R, D>,
    hermitian: &BoundDynamicTensorRef<'_, R, D>,
) where
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    for input in [general, tall] {
        qr_full_dyn_checked_generic(engine, &mut dense, input).unwrap();
        lq_full_dyn_checked_generic(engine, &mut dense, input).unwrap();
        left_null_dyn_checked_generic(engine, &mut dense, input).unwrap();
        right_null_dyn_checked_generic(engine, &mut dense, input).unwrap();
    }
    eig_full_dyn_checked_generic(engine, &mut dense, general).unwrap();
    eigh_full_dyn_checked_generic(engine, &mut dense, hermitian, HermitianTol::DEFAULT).unwrap();
}

#[cfg(feature = "cgc-gen")]
fn su3_checked_input<D: FactorScalar>(
    provider: &Arc<symtn_core::LieGroupFusionRule>,
    nin: usize,
    seed: usize,
) -> (
    BoundDynamicFusionMapSpace<symtn_core::LieGroupFusionRule>,
    Vec<D>,
) {
    let adjoint = provider.encode_irrep(&[1, 1]).unwrap();
    let trivial = provider.encode_irrep(&[0, 0]).unwrap();
    let fundamental = provider.encode_irrep(&[1, 0]).unwrap();
    let leg = || SectorLeg::new([(adjoint, 2), (trivial, 1), (fundamental, 1)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new((0..nin).map(|_| leg()).collect::<Vec<_>>()),
    );
    let space = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(provider),
        homspace,
    )
    .unwrap();
    // Diagonal `1, 2, ..., n` plus off-diagonal entries of modulus below
    // `0.36 / n`: the Gershgorin discs (radius < 0.36) are disjoint, so every
    // square sector has distinct eigenvalues and is diagonalizable by
    // construction, independent of the dense provider.
    let mut data = vec![D::zero(); space.space().required_len().unwrap()];
    let regions = space
        .space()
        .structure()
        .coupled_sector_regions(space.space().nout())
        .unwrap()
        .unwrap();
    for region in regions.iter() {
        let rows = region.rows();
        let scale = 0.25 / rows.max(region.cols()) as f64;
        let matrix = &mut data[region.range()];
        for (index, value) in matrix.iter_mut().enumerate() {
            let (row, col) = (index % rows, index / rows);
            *value = D::from_complex64(if row == col {
                Complex64::new((row + 1) as f64, 0.0)
            } else {
                let hash = (index * 11 + seed) as f64;
                Complex64::new((hash * 0.7).sin() * scale, (hash * 1.3).cos() * scale)
            });
        }
    }
    (space, data)
}

#[cfg(feature = "cgc-gen")]
fn hermitian_regions_copy<D: FactorScalar>(
    space: &BoundDynamicFusionMapSpace<symtn_core::LieGroupFusionRule>,
    data: &[D],
) -> Vec<D> {
    let mut hermitian = data.to_vec();
    for region in space
        .space()
        .structure()
        .coupled_sector_regions(space.space().nout())
        .unwrap()
        .unwrap()
        .iter()
    {
        let n = region.rows();
        let matrix = &data[region.range()];
        let target = &mut hermitian[region.range()];
        for col in 0..n {
            for row in 0..n {
                let value: Complex64 = (matrix[row + col * n].widen_complex()
                    + matrix[col + row * n].widen_complex().conj())
                .scale(0.5);
                target[row + col * n] = D::from_complex64(value);
            }
        }
    }
    hermitian
}

#[cfg(feature = "cgc-gen")]
fn assert_checked_su3_full_family_borrows_input<D: FactorScalar>() {
    let provider = Arc::new(symtn_core::testing::su_lie_rule(3));
    let (square, general) = su3_checked_input::<D>(&provider, 2, 3);
    let (tall_space, tall) = su3_checked_input::<D>(&provider, 1, 5);
    let hermitian = hermitian_regions_copy(&square, &general);
    assert_checked_full_family_borrows_input(
        &BoundDynamicTensorRef::try_new(&square, &general).unwrap(),
        &BoundDynamicTensorRef::try_new(&tall_space, &tall).unwrap(),
        &BoundDynamicTensorRef::try_new(&square, &hermitian).unwrap(),
    );
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_generic_full_null_eigen_ops_borrow_canonical_su3_input() {
    // Vertex multiplicity: 8 ⊗ 8 contains 8 twice.
    assert_checked_su3_full_family_borrows_input::<f64>();
    assert_checked_su3_full_family_borrows_input::<Complex64>();
}

fn assert_checked_full_family_borrows_input<R, D>(
    general: &BoundDynamicTensorRef<'_, R, D>,
    tall: &BoundDynamicTensorRef<'_, R, D>,
    hermitian: &BoundDynamicTensorRef<'_, R, D>,
) where
    R: CheckedGenericFusion,
    D: FactorScalar,
{
    let input_pack_bytes_rec = crate::probe::record();
    checked_full_family(general, tall, hermitian);
    assert_eq!(
        input_pack_bytes_rec.get(crate::probe::Event::InputPackBytes),
        0
    );
}

#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn assert_checked_factor_generic_full_family_borrows_input<D: FactorScalar>() {
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
        identity: FactorGenericRule.rule_identity(),
    });
    let rebind = |space: &BoundDynamicFusionMapSpace<FactorGenericRule>| {
        BoundDynamicFusionMapSpace::bind_generic(space.space().clone(), Arc::clone(&provider))
            .unwrap()
    };
    let convert = |data: &[Complex64]| {
        data.iter()
            .map(|&value| D::from_complex64(value))
            .collect::<Vec<_>>()
    };
    let (square, hermitian, general) = generic_values_endomorphism_input();
    let (tall_space, tall) = generic_factorization_input();
    let (square, tall_space) = (rebind(&square), rebind(&tall_space));
    let (hermitian, general) = (convert(&hermitian), convert(&general));
    let tall = tall
        .iter()
        .enumerate()
        .map(|(index, &value)| D::from_complex64(Complex64::new(value, index as f64 * 0.125 - 0.5)))
        .collect::<Vec<_>>();
    assert_checked_full_family_borrows_input(
        &BoundDynamicTensorRef::try_new(&square, &general).unwrap(),
        &BoundDynamicTensorRef::try_new(&tall_space, &tall).unwrap(),
        &BoundDynamicTensorRef::try_new(&square, &hermitian).unwrap(),
    );
}

#[test]
fn checked_generic_full_null_eigen_ops_borrow_canonical_multiplicity_input() {
    // What: checked-Generic inputs with a two-dimensional vertex
    // space borrow their regions.
    assert_checked_factor_generic_full_family_borrows_input::<f64>();
    assert_checked_factor_generic_full_family_borrows_input::<Complex64>();
}

#[test]
fn mf_null_factor_space_is_staged_once_per_side_and_cached() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: the multiplicity-free authority stages each null factor's space
    // exactly once, and a repeated call reuses the cached layout (the same
    // structure `Arc`).
    let tensor = rectangular_svd_tensor(5, 2);
    let bound = bound_tensor(Arc::new(Z2FusionRule), &tensor);
    let layout = bound.as_ref();
    let input = layout.dynamic();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let stage_rec = crate::probe::record();
    let stages = || stage_rec.get(crate::probe::Event::MfFactorSpaceStage);
    for left in [true, false] {
        let mut null = |dense: &mut symtn_dense::DefaultDenseExecutor| {
            if left {
                left_null_dyn(engine, dense, &input).unwrap()
            } else {
                right_null_dyn(engine, dense, &input).unwrap()
            }
        };
        let before = stages();
        let first = null(&mut dense);
        assert_eq!(stages() - before, 1);
        let second = null(&mut dense);
        assert_eq!(stages() - before, 2);
        assert!(Arc::ptr_eq(
            first.space().space().structure(),
            second.space().space().structure()
        ));
    }
}
