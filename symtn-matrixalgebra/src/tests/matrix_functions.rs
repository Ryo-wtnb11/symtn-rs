//! `inv`/`pinv`/`exp`/`solve` matrix-function tests (#1596 split of tests.rs).

use super::*;

#[test]
fn square_matrix_function_rejects_noncanonical_admitted_output_before_kernel() {
    let (canonical_space, canonical_data) = generic_factorization_input();
    let (padded_space, _) = padded_generic_factorization_input(&canonical_space, &canonical_data);
    let input = BoundDynamicTensorRef::try_new(&canonical_space, &canonical_data).unwrap();
    let called = Cell::new(false);
    let result = map_square_sectors_dyn_into(
        &mut ScriptedExecutor::new(RejectExecutorCalls),
        &input,
        padded_space,
        |_| -> Result<(), OperationError> {
            called.set(true);
            unreachable!("layout rejection precedes kernel initialization")
        },
        |_, _, _, _, _| unreachable!("layout rejection precedes dense work"),
    );
    assert!(matches!(
        result,
        Err(OperationError::UnsupportedTensorContractScope { .. })
    ));
    assert!(!called.get());
}

#[test]
fn generic_exp_direct_reuses_the_exact_input_provider_and_layout() {
    let provider = Arc::new(FactorGenericRule);
    let x = SectorId::new(1);
    let leg = SectorLeg::new([(x, 1)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone(), leg.clone()]),
        FusionProductSpace::new([leg.clone(), leg]),
    );
    let space =
        BoundDynamicFusionMapSpace::from_final_homspace_generic(provider, homspace).unwrap();
    let data = vec![0.0; space.space().required_len().unwrap()];
    let input = BoundDynamicTensorRef::try_new(&space, &data).unwrap();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let output = exp_direct_into_dyn(&mut dense, &input, space.clone()).unwrap();

    assert!(Arc::ptr_eq(
        output.space().provider_arc(),
        space.provider_arc()
    ));
    assert_eq!(
        output.space().space().structure(),
        space.space().structure()
    );
    assert_eq!(output.space().space().homspace(), space.space().homspace());
}

#[test]
fn pinv_rejects_invalid_rcond_before_dense_execution() {
    // What: invalid cutoff policy is rejected without entering the factorization backend.
    let rule = Z2FusionRule;
    let tensor = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let input = bound_tensor(Arc::new(rule), &tensor);
    for rcond in [-1.0, f64::NAN, f64::INFINITY] {
        let mut dense = ScriptedExecutor::new(RejectExecutorCalls);
        let error = pinv(&mut dense, &input.as_ref(), rcond).unwrap_err();
        assert!(
            matches!(error, OperationError::InvalidArgument { message } if message.contains("rcond"))
        );
    }
}

#[test]
fn exp_of_a_hermitian_tensor_inverts_under_negation() {
    let rule = Z2FusionRule;
    let raw = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    // Keep the spectrum modest so exp(t) exp(-t) stays well conditioned.
    let tensor = TestTensor::<f64>::from_vec_with_fusion_space(
        raw.data().iter().map(|value| 0.1 * value).collect(),
        raw.fusion_space().clone(),
    )
    .unwrap();
    let negated = TestTensor::<f64>::from_vec_with_fusion_space(
        tensor.data().iter().map(|value| -value).collect(),
        tensor.fusion_space().clone(),
    )
    .unwrap();
    let mut dense_executor = symtn_dense::DefaultDenseExecutor::new();
    let mut context = default_context();

    let forward = exp(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap();
    let backward = exp(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &negated),
    )
    .unwrap();
    let identity = crate::compose::compose(&mut context, &rule, &forward, &backward).unwrap();
    assert_identity_matrices(&dense_sector_matrices(2, &identity));
}

#[test]
fn pinv_satisfies_the_moore_penrose_identity() {
    let rule = Z2FusionRule;
    let sectors = [SectorId::new(0), SectorId::new(1)];
    let degeneracy = 2usize;
    let leg = || SectorLeg::new(sectors.iter().map(|&sector| (sector, degeneracy)), false);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg()]),
    );
    let space = DynamicFusionMapSpace::from_final_homspace(&rule, hom).unwrap();
    let len = space.required_len().unwrap();
    let canonical = TestTensor::<f64>::from_vec_with_fusion_space(
        (0..len).map(|i| ((i * 3 + 2) % 11) as f64 - 5.0).collect(),
        space,
    )
    .unwrap();
    let tensor = padded_copy(&rule, &canonical);
    let mut dense_executor = symtn_dense::DefaultDenseExecutor::new();
    let mut context = default_context();
    let compact_svd_copy_rec = crate::probe::record();
    let input_pack_bytes_rec = crate::probe::record();

    let plus = pinv(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
        1e-12,
    )
    .unwrap();
    let tp = crate::compose::compose(&mut context, &rule, &tensor, &plus).unwrap();
    let tpt = crate::compose::compose(&mut context, &rule, &tp, &tensor).unwrap();
    assert_eq!(tpt.structure(), canonical.structure());
    assert_eq!(tpt.data().len(), canonical.data().len());
    for (index, (lhs, rhs)) in tpt.data().iter().zip(canonical.data()).enumerate() {
        assert!(
            (lhs - rhs).abs() < 1e-8,
            "Moore-Penrose violated at raw position {index}: {lhs} != {rhs}"
        );
    }
    // The padded input is packed once; no SVD factor is laid out.
    assert!(input_pack_bytes_rec.get(crate::probe::Event::InputPackBytes) > 0);
    assert_eq!(compact_svd_copy_rec.compact_svd_copy(), Default::default());
}

#[test]
fn inv_composes_to_the_identity() {
    let rule = Z2FusionRule;
    let tensor = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    assert!(tensor
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .is_some());
    let expected_sectors = dense_sector_matrices(2, &tensor).len();
    let mut dense_executor = ScriptedExecutor::<SolveCallSpy>::default();
    let mut context = default_context();
    let inverse = inv(
        &mut dense_executor,
        &mut context,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap();
    assert_eq!(dense_executor.counts().solve, expected_sectors);
    let identity = crate::compose::compose(&mut context, &rule, &tensor, &inverse).unwrap();
    assert_identity_matrices(&dense_sector_matrices(2, &identity));
}

#[test]
fn solve_left_uses_one_direct_solve_per_sector_for_rectangular_rhs() {
    // What: A \ B accepts multiple RHS columns and never forms an inverse.
    let divisor = u1_block_endomorphism(&[(0, 2, vec![2.0_f64, 0.0, 1.0, 3.0]), (1, 1, vec![4.0])]);
    let rhs = u1_block_map(&[
        (0, 2, 3, vec![4.0_f64, 6.0, 10.0, 12.0, 16.0, 18.0]),
        (1, 1, 2, vec![28.0, 32.0]),
    ]);
    let divisor_provider = Arc::new(U1FusionRule);
    let rhs_provider = Arc::new(U1FusionRule);
    let divisor = bound_tensor(Arc::clone(&divisor_provider), &divisor);
    let rhs = bound_tensor(Arc::clone(&rhs_provider), &rhs);
    let mut dense = ScriptedExecutor::<SolveCallSpy>::default();

    let solved = solve_left_into_mf(
        &mut dense,
        &divisor.as_ref().dynamic(),
        &rhs.as_ref().dynamic(),
    )
    .unwrap();

    assert!(Arc::ptr_eq(
        solved.space().provider_arc(),
        &divisor_provider
    ));
    assert!(!Arc::ptr_eq(solved.space().provider_arc(), &rhs_provider));
    let solved: BoundTensorMap<_, _> = typed_from_bound_factor(solved).unwrap();
    let sectors = dense_sector_matrices(1, solved.tensor());
    let even = sectors
        .iter()
        .find(|(sector, _, _, _)| *sector == U1Irrep::new(0).sector_id())
        .unwrap();
    let odd = sectors
        .iter()
        .find(|(sector, _, _, _)| *sector == U1Irrep::new(1).sector_id())
        .unwrap();
    assert_eq!((even.1, even.2), (2, 3));
    // Hand quotients; the triangular solve may divide or multiply by a
    // reciprocal, so they are compared under the rule (2 = the sector order).
    numerics::assert_slices_close("even", &even.3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2);
    assert_eq!((odd.1, odd.2), (1, 2));
    numerics::assert_slices_close("odd", &odd.3, &[7.0, 8.0], 1);
    assert_eq!(dense.counts().solve, 2);
    // What: the backend wrote the first final sector in the returned payload;
    // no owned solution twin followed by a full-result copy can satisfy this
    // pointer identity.
    assert_eq!(dense.destination_ptrs[0], solved.data().as_ptr() as usize);
}

#[test]
fn solve_left_preserves_complex_values_without_adjointing() {
    // What: A \ B is the ordinary complex linear solve, not a Hermitian route.
    let divisor_data = [
        Complex64::new(2.0, 1.0),
        Complex64::new(0.0, 0.5),
        Complex64::new(1.0, -1.0),
        Complex64::new(3.0, 0.0),
    ];
    let expected = [
        Complex64::new(1.0, 2.0),
        Complex64::new(-1.0, 0.5),
        Complex64::new(0.25, -0.75),
        Complex64::new(2.0, 1.0),
    ];
    let mut rhs_data = vec![Complex64::zero(); 4];
    for col in 0..2 {
        for row in 0..2 {
            rhs_data[row + 2 * col] = divisor_data[row] * expected[2 * col]
                + divisor_data[row + 2] * expected[1 + 2 * col];
        }
    }
    let divisor = u1_block_endomorphism(&[(0, 2, divisor_data.to_vec())]);
    let rhs = u1_block_map(&[(0, 2, 2, rhs_data)]);
    let divisor = bound_tensor(Arc::new(U1FusionRule), &divisor);
    let rhs = bound_tensor(Arc::new(U1FusionRule), &rhs);
    let mut dense = ScriptedExecutor::<SolveCallSpy>::default();

    let solved = solve_left_into_mf(
        &mut dense,
        &divisor.as_ref().dynamic(),
        &rhs.as_ref().dynamic(),
    )
    .unwrap();

    for (&actual, expected) in solved.data().iter().zip(expected) {
        assert!((actual - expected).norm() < 1.0e-12);
    }
    assert_eq!(dense.counts().solve, 1);
}

#[test]
fn solve_left_returns_the_backend_error_when_a_later_sector_fails() {
    // What: an earlier successful sector cannot publish a partial solution.
    let divisor = u1_block_endomorphism(&[(0, 1, vec![2.0_f64]), (1, 1, vec![3.0])]);
    let rhs = u1_block_map(&[(0, 1, 1, vec![4.0_f64]), (1, 1, 1, vec![9.0_f64])]);
    let divisor = bound_tensor(Arc::new(U1FusionRule), &divisor);
    let rhs = bound_tensor(Arc::new(U1FusionRule), &rhs);
    let mut dense = ScriptedExecutor::<FailSecondSolve>::default();

    let error = solve_left_into_mf(
        &mut dense,
        &divisor.as_ref().dynamic(),
        &rhs.as_ref().dynamic(),
    )
    .unwrap_err();

    assert!(matches!(
        error,
        OperationError::Dense(DenseError::Backend {
            op: "solve_into",
            ..
        })
    ));
    assert_eq!(dense.counts().solve, 2);
}

#[test]
fn solve_left_validates_spaces_before_backend_execution() {
    // What: a codomain mismatch is a structural failure, not a backend call.
    let divisor = u1_block_endomorphism(&[(0, 2, vec![1.0_f64, 0.0, 0.0, 1.0])]);
    let wrong_codomain = u1_cross_space_map::<f64>(&[(0, 3)], &[(0, 1)]);
    let divisor = bound_tensor(Arc::new(U1FusionRule), &divisor);
    let wrong_codomain = bound_tensor(Arc::new(U1FusionRule), &wrong_codomain);
    let mut dense = ScriptedExecutor::new(RejectExecutorCalls);
    let error = solve_left_into_mf(
        &mut dense,
        &divisor.as_ref().dynamic(),
        &wrong_codomain.as_ref().dynamic(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        OperationError::SpaceMismatch {
            message: "solve requires equal divisor and right-hand-side codomains"
        }
    ));
    // The divisor's isomorphism is the facade preflight's (#1995): this
    // seam requires an admitted divisor.
}

#[test]
fn solve_left_preserves_dense_singularity_and_capability_errors() {
    // What: solve reports the backend's stable error class without falling
    // back to SVD, inverse formation, or a pseudoinverse.
    let divisor = u1_block_endomorphism(&[(0, 2, vec![1.0_f64, 0.0, 0.0, 0.0])]);
    let rhs = u1_block_map(&[(0, 2, 1, vec![1.0_f64, 1.0])]);
    let divisor = bound_tensor(Arc::new(U1FusionRule), &divisor);
    let rhs = bound_tensor(Arc::new(U1FusionRule), &rhs);
    let error = solve_left_into_mf(
        &mut symtn_dense::DefaultDenseExecutor::new(),
        &divisor.as_ref().dynamic(),
        &rhs.as_ref().dynamic(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        OperationError::Dense(DenseError::NumericalFailure {
            op: "solve_into",
            ..
        })
    ));

    let divisor = u1_block_endomorphism(&[(0, 1, vec![2.0_f64])]);
    let rhs = u1_block_map(&[(0, 1, 1, vec![1.0_f64])]);
    let divisor = bound_tensor(Arc::new(U1FusionRule), &divisor);
    let rhs = bound_tensor(Arc::new(U1FusionRule), &rhs);
    let error = solve_left_into_mf(
        &mut ScriptedExecutor::new(RejectExecutorCalls),
        &divisor.as_ref().dynamic(),
        &rhs.as_ref().dynamic(),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        OperationError::Dense(DenseError::Unsupported {
            op: "solve_into",
            ..
        })
    ));
}

#[test]
fn solve_left_direct_into_rejects_foreign_authority_and_wrong_output_before_execution() {
    // What: caller-admitted output metadata is an input contract, not a hint.
    let divisor = u1_block_endomorphism(&[(0, 1, vec![2.0_f64])]);
    let rhs = u1_block_map(&[(0, 1, 1, vec![1.0_f64])]);
    let provider = Arc::new(U1FusionRule);
    let divisor = bound_tensor(Arc::clone(&provider), &divisor);
    let rhs = bound_tensor(Arc::new(U1FusionRule), &rhs);
    let expected = FusionTreeHomSpace::new(
        divisor.space().space().homspace().domain().clone(),
        rhs.space().space().homspace().domain().clone(),
    );

    let foreign = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::new(U1FusionRule),
        expected.clone(),
    )
    .unwrap();
    let error = solve_left_direct_into_dyn(
        &mut ScriptedExecutor::new(RejectExecutorCalls),
        &divisor.as_ref().dynamic(),
        &rhs.as_ref().dynamic(),
        foreign,
    )
    .unwrap_err();
    assert!(
        matches!(error, OperationError::StructureMismatch { .. }),
        "{error:?}"
    );

    let wrong = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        provider,
        FusionTreeHomSpace::new(
            divisor.space().space().homspace().domain().clone(),
            FusionProductSpace::new([
                SectorLeg::new([(U1Irrep::new(0).sector_id(), 1)], false),
                SectorLeg::new([(U1Irrep::new(0).sector_id(), 1)], false),
            ]),
        ),
    )
    .unwrap();
    let error = solve_left_direct_into_dyn(
        &mut ScriptedExecutor::new(RejectExecutorCalls),
        &divisor.as_ref().dynamic(),
        &rhs.as_ref().dynamic(),
        wrong,
    )
    .unwrap_err();
    assert!(
        matches!(error, OperationError::StructureMismatch { .. }),
        "{error:?}"
    );
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn pinv_direct_into_rejects_foreign_authority_and_wrong_output_before_execution() {
    // What: the checked-Generic pinv seam treats its admitted destination as
    // a sealed input, before it allocates staging or invokes SVD/GEMM.
    let (base, data) = generic_factorization_input();
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
        identity: FactorGenericRule.rule_identity(),
    });
    let source =
        BoundDynamicFusionMapSpace::bind_generic(base.space().clone(), Arc::clone(&provider))
            .unwrap();
    let input = BoundDynamicTensorRef::try_new(&source, &data).unwrap();
    let expected = FusionTreeHomSpace::new(
        source.space().homspace().domain().clone(),
        source.space().homspace().codomain().clone(),
    );

    // Same identity, another allocation: the foreign authority.
    let foreign = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::new(LateGenericSpy {
            rule: FactorGenericRule,
            fail_at: usize::MAX,
            calls: Cell::new(0),
            identity: provider.identity.clone(),
        }),
        expected.clone(),
    )
    .unwrap();
    let error = pinv_direct_into_dyn(
        &mut ScriptedExecutor::new(RejectExecutorCalls),
        &input,
        foreign,
        0.0,
    )
    .unwrap_err();
    assert!(
        matches!(error, OperationError::StructureMismatch { .. }),
        "{error:?}"
    );

    let wrong = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        provider,
        FusionTreeHomSpace::new(
            source.space().homspace().domain().clone(),
            FusionProductSpace::new([
                SectorLeg::new([(SectorId::new(1), 1)], false),
                SectorLeg::new([(SectorId::new(1), 1)], false),
            ]),
        ),
    )
    .unwrap();
    let error = pinv_direct_into_dyn(
        &mut ScriptedExecutor::new(RejectExecutorCalls),
        &input,
        wrong,
        0.0,
    )
    .unwrap_err();
    assert!(
        matches!(error, OperationError::StructureMismatch { .. }),
        "{error:?}"
    );
}

#[test]
fn solve_left_direct_into_rejects_late_tree_route_before_execution() {
    // What: an admitted output with the right identity and HomSpace still
    // cannot reinterpret a different coupled-tree basis.
    let source = hermitian_test_tensor(&Z2FusionRule, &[SectorId::new(0), SectorId::new(1)]);
    let output = reversed_coupled_tree_basis_copy(&Z2FusionRule, &source);
    let provider = Arc::new(Z2FusionRule);
    let divisor = bound_tensor(Arc::clone(&provider), &source);
    let rhs = bound_tensor(Arc::clone(&provider), &source);
    let output_space = BoundDynamicFusionMapSpace::bind_multiplicity_free(
        output.fusion_space().clone(),
        Arc::clone(&provider),
    )
    .unwrap();
    let mut dense = ScriptedExecutor::<SolveCallSpy>::default();

    let error = solve_left_direct_into_dyn(
        &mut dense,
        &divisor.as_ref().dynamic(),
        &rhs.as_ref().dynamic(),
        output_space,
    )
    .unwrap_err();

    assert!(
        matches!(
            error,
            OperationError::SpaceMismatch {
                message: "solve coupled-sector tree bases are incompatible"
            }
        ),
        "{error:?}"
    );
    assert_eq!(dense.counts().solve, 0);
}

fn u1_block_map<D>(blocks: &[(i32, usize, usize, Vec<D>)]) -> TestTensor<D>
where
    D: Copy + Zero,
{
    let codomain = SectorLeg::new(
        blocks
            .iter()
            .map(|(charge, rows, _, _)| (U1Irrep::new(*charge).sector_id(), *rows)),
        false,
    );
    let domain = SectorLeg::new(
        blocks
            .iter()
            .map(|(charge, _, cols, _)| (U1Irrep::new(*charge).sector_id(), *cols)),
        false,
    );
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([codomain]),
        FusionProductSpace::new([domain]),
    );
    let space = DynamicFusionMapSpace::from_final_homspace(&U1FusionRule, homspace).unwrap();
    TestTensor::from_block_fn_with_fusion_space(space, D::zero(), |key, indices| {
        let BlockKey::FusionTree(tree) = key else {
            return D::zero();
        };
        let coupled = tree.codomain_tree().coupled();
        let (_, rows, _, data) = blocks
            .iter()
            .find(|(charge, _, _, _)| U1Irrep::new(*charge).sector_id() == coupled)
            .unwrap();
        data[indices[0] + rows * indices[1]]
    })
    .unwrap()
}

fn scalar_block_endomorphism<R>(rule: &R, blocks: &[(SectorId, f64)]) -> TestTensor<f64>
where
    R: MultiplicityFreeRigidSymbols,
{
    let leg = SectorLeg::new(blocks.iter().map(|&(sector, _)| (sector, 1)), false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone()]),
        FusionProductSpace::new([leg]),
    );
    let space = DynamicFusionMapSpace::from_final_homspace(rule, homspace).unwrap();
    TestTensor::from_block_fn_with_fusion_space(space, 0.0, |key, _| {
        let BlockKey::FusionTree(tree) = key else {
            return 0.0;
        };
        let sector = tree.codomain_tree().coupled();
        blocks
            .iter()
            .find_map(|&(candidate, value)| (candidate == sector).then_some(value))
            .unwrap()
    })
    .unwrap()
}

fn assert_scale_separated_inverse<R>(rule: R, sectors: [SectorId; 2])
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey,
{
    let values = [3.0, 1e-14];
    let tensor =
        scalar_block_endomorphism(&rule, &[(sectors[0], values[0]), (sectors[1], values[1])]);
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let mut context = TensorContractFusionExecutionContext::<f64, R::Key>::default();
    let inverse = inv(
        &mut dense,
        &mut context,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap();

    for (sector, value) in sectors.into_iter().zip(values) {
        assert!((value * scalar_block(inverse.tensor(), sector) - 1.0).abs() < 1e-12);
    }
}

#[test]
fn inv_uses_each_u1_sector_scale_for_f64_rank_and_value() {
    // What: an invertible scalar sector remains invertible regardless of another sector's scale.
    for dominant in [1.0, 1e12] {
        let tensor = u1_block_endomorphism(&[(0, 1, vec![dominant]), (1, 1, vec![1e-14_f64])]);
        let mut dense = symtn_dense::DefaultDenseExecutor::new();
        let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();

        let inverse = inv(
            &mut dense,
            &mut context,
            &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
        )
        .unwrap();

        let dominant_inverse = scalar_u1_block(inverse.tensor(), 0);
        let small_inverse = scalar_u1_block(inverse.tensor(), 1);
        assert!((dominant * dominant_inverse - 1.0).abs() < 1e-12);
        assert!((1e-14 * small_inverse - 1.0).abs() < 1e-12);
        assert!((small_inverse / 1e14 - 1.0).abs() < 1e-12);
    }
}

#[test]
fn inv_uses_sector_local_scale_for_su2_fz2_and_product_rules() {
    // What: sector-local rank and inversion apply uniformly to non-Abelian,
    // fermionic, and nested product rules rather than only the U1 fixture.
    assert_scale_separated_inverse(
        SU2FusionRule,
        [
            SU2Irrep::from_twice_spin(0).sector_id(),
            SU2Irrep::from_twice_spin(1).sector_id(),
        ],
    );
    assert_scale_separated_inverse(
        FermionParityFusionRule,
        [SectorId::new(0), SectorId::new(1)],
    );

    let fz2_u1 = product_fusion_rule(FermionParityFusionRule, U1FusionRule);
    let product_sectors = [
        fz2_u1.encode_component_ids(SectorId::new(0), U1Irrep::new(0).sector_id()),
        fz2_u1.encode_component_ids(SectorId::new(1), U1Irrep::new(1).sector_id()),
    ];
    let fz2_u1_su2 = product_fusion_rule(fz2_u1, SU2FusionRule);
    let nested_sectors = [
        fz2_u1_su2
            .encode_component_ids(product_sectors[0], SU2Irrep::from_twice_spin(0).sector_id()),
        fz2_u1_su2
            .encode_component_ids(product_sectors[1], SU2Irrep::from_twice_spin(1).sector_id()),
    ];
    assert_scale_separated_inverse(fz2_u1_su2, nested_sectors);
}

#[test]
fn inv_uses_each_u1_sector_scale_for_phased_c64_values() {
    // What: complex phases do not couple numerical-rank decisions across sectors.
    let large = Complex64::from_polar(1e8, 0.37);
    let small = Complex64::from_polar(1e-14, -0.91);
    let tensor = u1_block_endomorphism(&[(0, 1, vec![large]), (1, 1, vec![small])]);
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let mut context = TensorContractFusionExecutionContext::<Complex64, RuleIdentity>::default();

    let inverse = inv(
        &mut dense,
        &mut context,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
    )
    .unwrap();

    assert!(
        (large * scalar_u1_block(inverse.tensor(), 0) - Complex64::new(1.0, 0.0)).norm() < 1e-12
    );
    assert!(
        (small * scalar_u1_block(inverse.tensor(), 1) - Complex64::new(1.0, 0.0)).norm() < 1e-12
    );
}

#[test]
fn inv_solves_padded_u1_sectors_and_matches_the_dense_oracle() {
    // What: a legal expert layout uses the same sector solves and restores each
    // dense inverse to the destination block layout.
    let tensor = u1_block_endomorphism(&[
        (0, 2, vec![2.0_f64, 1.0, 3.0, 4.0]),
        (1, 2, vec![1.0_f64, 2.0, 0.0, 3.0]),
    ]);
    let padded = padded_copy(&U1FusionRule, &tensor);
    assert!(padded
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .is_none());
    let mut dense = ScriptedExecutor::<SolveCallSpy>::default();
    let mut context = default_context();

    let inverse = inv(
        &mut dense,
        &mut context,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &padded),
    )
    .unwrap();

    let expected = [
        (U1Irrep::new(0).sector_id(), vec![0.8, -0.2, -0.6, 0.4]),
        (
            U1Irrep::new(1).sector_id(),
            vec![1.0, -2.0 / 3.0, 0.0, 1.0 / 3.0],
        ),
    ];
    for (sector, _, _, values) in dense_sector_matrices(1, inverse.tensor()) {
        let oracle = expected
            .iter()
            .find_map(|(candidate, values)| (*candidate == sector).then_some(values))
            .unwrap();
        for (&actual, &expected) in values.iter().zip(oracle) {
            assert!((actual - expected).abs() < 1.0e-12);
        }
    }
    assert_eq!(dense.counts().solve, 2);
}

#[test]
fn inv_reorders_a_complete_expert_tree_grid_by_key() {
    // What: an expert layout's block order cannot transpose matrix axes or
    // replace fusion-tree identity as the inverse routing authority.
    let rule = Z2FusionRule;
    let leg = || SectorLeg::new([(SectorId::new(0), 2), (SectorId::new(1), 2)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let space = DynamicFusionMapSpace::from_final_homspace(&rule, homspace).unwrap();
    let tree_code = |tree: &FusionTreeKey| {
        tree.uncoupled()
            .iter()
            .fold(0usize, |value, sector| value * 2 + sector.id())
    };
    let diagonal = |tree: &FusionTreeKey, row: usize| 2.0 + tree_code(tree) as f64 + row as f64;
    let upper = |tree: &FusionTreeKey| 0.25 + tree_code(tree) as f64 * 0.125;
    let canonical = TestTensor::from_block_fn_with_fusion_space(space, 0.0, |key, indices| {
        let BlockKey::FusionTree(key) = key else {
            return 0.0;
        };
        if key.codomain_tree() != key.domain_tree() {
            return 0.0;
        }
        let row = indices[0] + 2 * indices[1];
        let col = indices[2] + 2 * indices[3];
        if row == col {
            diagonal(key.codomain_tree(), row)
        } else if row == 0 && col == 1 {
            upper(key.codomain_tree())
        } else {
            0.0
        }
    })
    .unwrap();
    let reordered = reversed_complete_grid_copy(&rule, &canonical);
    assert!(reordered
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .is_none());
    let mut dense = ScriptedExecutor::<SolveCallSpy>::default();
    let mut context = default_context();

    let inverse = inv(
        &mut dense,
        &mut context,
        &bound_tensor_ref!(Arc::new(rule), &reordered),
    )
    .unwrap();

    for index in 0..inverse.structure().block_count() {
        let block = inverse.structure().block(index).unwrap();
        let BlockKey::FusionTree(key) = block.key() else {
            panic!("inverse output must retain fusion-tree blocks")
        };
        for row in 0..4 {
            for col in 0..4 {
                let position = block.offset()
                    + (row % 2) * block.strides()[0]
                    + (row / 2) * block.strides()[1]
                    + (col % 2) * block.strides()[2]
                    + (col / 2) * block.strides()[3];
                let expected = if key.codomain_tree() != key.domain_tree() {
                    0.0
                } else if row == col {
                    diagonal(key.codomain_tree(), row).recip()
                } else if row == 0 && col == 1 {
                    -upper(key.codomain_tree())
                        / (diagonal(key.codomain_tree(), 0) * diagonal(key.codomain_tree(), 1))
                } else {
                    0.0
                };
                assert!(
                    (inverse.data()[position] - expected).abs() < 1.0e-12,
                    "tree={key:?} row={row} col={col}"
                );
            }
        }
    }
    assert_eq!(dense.counts().solve, 2);
}

#[test]
fn inv_preserves_genuinely_complex_nonhermitian_sector_values() {
    // What: inverse is an ordinary solve, not an adjoint or Hermitian spectral operation.
    let a = Complex64::new(2.0, 1.0);
    let b = Complex64::new(3.0, -2.0);
    let c = Complex64::new(1.0, 4.0);
    let d = Complex64::new(5.0, -1.0);
    let tensor = u1_block_endomorphism(&[(0, 2, vec![a, b, c, d])]);
    let mut dense = ScriptedExecutor::<SolveCallSpy>::default();
    let bound = bound_tensor(Arc::new(U1FusionRule), &tensor);

    let inverse = inv_into_mf(&mut dense, &bound.as_ref().dynamic()).unwrap();

    let determinant = a * d - b * c;
    let oracle = [
        d / determinant,
        -b / determinant,
        -c / determinant,
        a / determinant,
    ];
    for (&actual, expected) in inverse.data().iter().zip(oracle) {
        assert!((actual - expected).norm() < 1.0e-12);
    }
    assert_eq!(dense.counts().solve, 1);
}

#[test]
fn inv_dyn_reverses_isomorphic_spaces_with_different_tree_ranks() {
    // What: sector identity routes an inverse between isomorphic reduced spaces
    // even when their external tree ranks differ.
    let charge = U1Irrep::new(0).sector_id();
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(charge, 2)], false),
            SectorLeg::new([(charge, 3)], false),
        ]),
        FusionProductSpace::new([SectorLeg::new([(charge, 6)], false)]),
    );
    let space = DynamicFusionMapSpace::from_final_homspace(&U1FusionRule, homspace).unwrap();
    let mut data = vec![0.0_f64; 36];
    for index in 0..6 {
        data[index + 6 * index] = index as f64 + 1.0;
    }
    let tensor = TestTensor::<f64>::from_vec_with_fusion_space(data, space).unwrap();
    let bound = bound_tensor(Arc::new(U1FusionRule), &tensor);
    let mut dense = ScriptedExecutor::<SolveCallSpy>::default();

    let inverse = inv_into_mf(&mut dense, &bound.as_ref().dynamic()).unwrap();
    let inverse: BoundTensorMap<_, _> = typed_from_bound_factor(inverse).unwrap();

    assert_eq!(
        inverse.tensor().fusion_space().homspace().codomain(),
        tensor.fusion_space().homspace().domain()
    );
    assert_eq!(
        inverse.tensor().fusion_space().homspace().domain(),
        tensor.fusion_space().homspace().codomain()
    );
    for row in 0..6 {
        for col in 0..6 {
            let expected = if row == col {
                1.0 / (row as f64 + 1.0)
            } else {
                0.0
            };
            assert!((inverse.data()[row + 6 * col] - expected).abs() < 1.0e-12);
        }
    }
    assert_eq!(dense.counts().solve, 1);
}

#[test]
fn inv_route_preflight_rejects_a_missing_later_sector_before_execution() {
    // What: every sector route is checked before the first dense solve.
    let source = u1_block_endomorphism(&[(0, 1, vec![1.0_f64]), (1, 1, vec![2.0])]);
    let incomplete_output = u1_block_endomorphism(&[(0, 1, vec![1.0_f64])]);
    let source_regions = source
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    let output_regions = incomplete_output
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();

    let error =
        validate_inverse_region_routes_for_test(&source_regions, &output_regions).unwrap_err();

    assert!(matches!(
        error,
        OperationError::SpaceMismatch {
            message: "inverse output is missing a source coupled sector"
        }
    ));
}

#[test]
fn inv_accepts_tiny_nonzero_pivots_for_all_factor_dtypes() {
    // What: ordinary inverse follows provider LU singularity semantics rather
    // than rejecting a nonzero pivot using a dtype-relative SVD cutoff.
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let tensor = u1_block_endomorphism(&[(0, 2, vec![1.0_f32, 0.0, 0.0, 1.0e-8])]);
    let bound = bound_tensor(Arc::new(U1FusionRule), &tensor);
    let inverse = inv_into_mf(&mut dense, &bound.as_ref().dynamic()).unwrap();
    assert!((inverse.data()[0] - 1.0).abs() < 1.0e-6);
    assert!((inverse.data()[3] * 1.0e-8 - 1.0).abs() < 1.0e-5);

    let tensor = u1_block_endomorphism(&[(0, 2, vec![1.0_f64, 0.0, 0.0, 1.0e-16])]);
    let bound = bound_tensor(Arc::new(U1FusionRule), &tensor);
    let inverse = inv_into_mf(&mut dense, &bound.as_ref().dynamic()).unwrap();
    assert!((inverse.data()[0] - 1.0).abs() < 1.0e-12);
    assert!((inverse.data()[3] * 1.0e-16 - 1.0).abs() < 1.0e-12);

    let phase = Complex32::from_polar(1.0e-8, -0.41);
    let tensor = u1_block_endomorphism(&[(
        0,
        2,
        vec![
            Complex32::new(1.0, 0.0),
            Complex32::zero(),
            Complex32::zero(),
            phase,
        ],
    )]);
    let bound = bound_tensor(Arc::new(U1FusionRule), &tensor);
    let inverse = inv_into_mf(&mut dense, &bound.as_ref().dynamic()).unwrap();
    assert!((inverse.data()[3] * phase - Complex32::new(1.0, 0.0)).norm() < 1.0e-5);

    let phase = Complex64::from_polar(1.0e-16, 0.23);
    let tensor = u1_block_endomorphism(&[(
        0,
        2,
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::zero(),
            Complex64::zero(),
            phase,
        ],
    )]);
    let bound = bound_tensor(Arc::new(U1FusionRule), &tensor);
    let inverse = inv_into_mf(&mut dense, &bound.as_ref().dynamic()).unwrap();
    assert!((inverse.data()[3] * phase - Complex64::new(1.0, 0.0)).norm() < 1.0e-12);
}

#[test]
fn inv_rejects_a_genuinely_singular_sector_without_an_output() {
    // What: the dense solve reports an exact singular direction as a typed numerical failure.
    let tensor = u1_block_endomorphism(&[(0, 2, vec![1.0_f64, 0.0, 0.0, 0.0])]);
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();

    let error = inv(
        &mut dense,
        &mut context,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
    )
    .unwrap_err();

    assert!(matches!(
        error,
        OperationError::Dense(DenseError::NumericalFailure {
            op: "solve_into",
            ..
        })
    ));
}

#[test]
fn inv_returns_the_backend_error_when_a_later_sector_fails() {
    // What: success in an earlier sector cannot publish a partial inverse when
    // a later backend solve fails.
    let tensor = u1_block_endomorphism(&[(0, 1, vec![2.0_f64]), (1, 1, vec![3.0])]);
    let bound = bound_tensor(Arc::new(U1FusionRule), &tensor);
    let mut dense = ScriptedExecutor::<FailSecondSolve>::default();

    let error = inv_into_mf(&mut dense, &bound.as_ref().dynamic()).unwrap_err();

    assert!(matches!(
        error,
        OperationError::Dense(DenseError::Backend {
            op: "solve_into",
            ..
        })
    ));
    assert_eq!(dense.counts().solve, 2);
}

#[test]
fn inv_accepts_a_zero_dimensional_endomorphism_without_dense_execution() {
    // What: the inverse of the legal empty endomorphism is the empty endomorphism.
    let tensor = rectangular_svd_tensor(0, 0);
    let mut dense = ScriptedExecutor::new(RejectExecutorCalls);
    let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();

    let inverse = inv(
        &mut dense,
        &mut context,
        &bound_tensor_ref!(Arc::new(Z2FusionRule), &tensor),
    )
    .unwrap();

    assert!(inverse.data().is_empty());
    assert_eq!(inverse.tensor().fusion_space(), tensor.fusion_space());
}

#[test]
fn inv_solves_the_rank_zero_scalar_sector() {
    // What: a rank-zero scalar is one 1x1 vacuum-sector solve, not an empty tensor.
    let rule = Z2FusionRule;
    let homspace =
        FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([]));
    let space = DynamicFusionMapSpace::from_final_homspace(&rule, homspace).unwrap();
    let scalar = TestTensor::<f64>::from_vec_with_fusion_space(vec![-4.0], space).unwrap();
    let mut dense = ScriptedExecutor::<SolveCallSpy>::default();
    let mut context = default_context();

    let inverse = inv(
        &mut dense,
        &mut context,
        &bound_tensor_ref!(Arc::new(rule), &scalar),
    )
    .unwrap();

    assert_eq!(inverse.data(), &[-0.25]);
    assert_eq!(dense.counts().solve, 1);
}

#[test]
fn pinv_keeps_its_global_rcond_cutoff() {
    // What: public pinv still drops singular values relative to the global maximum.
    let canonical = u1_block_endomorphism(&[(0, 1, vec![1.0_f64]), (1, 1, vec![1e-14])]);
    let tensor = padded_copy(&U1FusionRule, &canonical);
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let compact_svd_copy_rec = crate::probe::record();
    let input_pack_bytes_rec = crate::probe::record();

    let inverse = pinv(
        &mut dense,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
        1e-12,
    )
    .unwrap();

    assert!((scalar_u1_block(inverse.tensor(), 0) - 1.0).abs() < 1e-12);
    assert_eq!(scalar_u1_block(inverse.tensor(), 1), 0.0);
    // The padded input is packed once; no SVD factor is laid out.
    assert!(input_pack_bytes_rec.get(crate::probe::Event::InputPackBytes) > 0);
    assert_eq!(compact_svd_copy_rec.compact_svd_copy(), Default::default());
}

#[test]
fn pinv_adjoint_parent_reconstructs_complex_padded_rectangular_sectors() {
    let rule = Z2FusionRule;
    let source = mixed_rectangular_c32_tensor();
    let canonical = TestTensor::<Complex64>::from_vec_with_fusion_space(
        source
            .data()
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect(),
        source.fusion_space().clone(),
    )
    .unwrap();
    let parent = padded_copy(&rule, &canonical);
    let provider = Arc::new(rule);
    let bound = bound_tensor(Arc::clone(&provider), &parent);
    assert!(
        crate::factorize::compact_factor_plan_for_test(bound.space())
            .unwrap()
            .is_none()
    );
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let mut context = TensorContractFusionExecutionContext::<Complex64, RuleIdentity>::default();
    let compact_svd_copy_rec = crate::probe::record();
    let input_pack_bytes_rec = crate::probe::record();
    let output = pinv_adjoint_parent_into_mf(&mut dense, &bound.as_ref().dynamic(), 0.0).unwrap();
    let output: BoundTensorMap<_, _> = typed_from_bound_factor(output).unwrap();
    let adjoint = adjoint(provider.as_ref(), &canonical).unwrap();
    let first = crate::compose::compose(&mut context, provider.as_ref(), &adjoint, output.tensor())
        .unwrap();
    let reconstructed =
        crate::compose::compose(&mut context, provider.as_ref(), &first, &adjoint).unwrap();

    assert_eq!(reconstructed.structure(), adjoint.structure());
    for (&actual, &expected) in reconstructed.data().iter().zip(adjoint.data()) {
        assert!((actual - expected).norm() < 1.0e-10);
    }
    // The padded input is packed once; no SVD factor is laid out.
    assert!(input_pack_bytes_rec.get(crate::probe::Event::InputPackBytes) > 0);
    assert_eq!(compact_svd_copy_rec.compact_svd_copy(), Default::default());
}

#[test]
fn pinv_adjoint_parent_discards_unpublished_output_on_recomposition_failure() {
    // What: after a successful parent SVD and scaling, a failed per-sector
    // recomposition GEMM returns no partial output.
    let tensor = u1_block_endomorphism(&[(0, 2, vec![2.0, 0.0, 0.0, 3.0])]);
    let bound = bound_tensor(Arc::new(U1FusionRule), &tensor);
    let mut dense = ScriptedExecutor::<MatrixFunctionCallSpy>::default();
    dense.script.fail(
        &[Op::DotGeneral],
        None,
        "dot_general_into",
        "injected recomposition failure",
    );

    assert!(matches!(
        pinv_adjoint_parent_into_mf(&mut dense, &bound.as_ref().dynamic(), 0.0),
        Err(OperationError::Dense(DenseError::Backend {
            op: "dot_general_into",
            ..
        }))
    ));
    assert_eq!(dense.counts().svd, 1, "the parent SVD ran");
}

// ============================================================================
// Issue #577: general (non-Hermitian) matrix exponential.
//
// Oracle. `series_exp` evaluates the defining series and shares nothing with
// the Padé implementation; `exp_of_the_oracle_tensor_matches_tensorkit_anchors`
// keeps two TensorKit blocks of the same fixture as convention anchors.
//
// Fixture certification: the fill is a closed formula in the coupled charge and
// the two degeneracy indices, and `u1_block_endomorphism` places one fusion
// tree per coupled sector, so `u1_block_matrix` reads back exactly the matrix
// Julia's `blocks(t)` iterated over. `exp_fixture_blocks_reproduce_the_oracle_input`
// pins that correspondence through the pre-existing reader before any exponential
// is taken.
// ============================================================================

/// Relative Frobenius agreement with the series oracle and the anchors, per the
/// #577 design.
const EXP_ORACLE_RTOL: f64 = 1e-12;

/// Residual of `exp(A) exp(-A) - 1`, per the #577 design.
const EXP_INVERSE_RTOL: f64 = 1e-11;

/// Column-major coupled-sector matrix of a single-tree `1 <- 1` U(1) tensor.
fn u1_block_matrix<D: Copy + Zero>(tensor: &TestTensor<D>, charge: i32) -> Vec<D> {
    let sector = U1Irrep::new(charge).sector_id();
    let structure = tensor.structure();
    let block = (0..structure.block_count())
        .map(|index| structure.block(index).unwrap())
        .find(|block| match block.key() {
            BlockKey::FusionTree(key) => key.codomain_tree().coupled() == sector,
            _ => false,
        })
        .unwrap();
    let order = block.shape()[0];
    assert_eq!(block.shape(), &[order, order]);
    let mut matrix = vec![D::zero(); order * order];
    for column in 0..order {
        for row in 0..order {
            matrix[row + order * column] = tensor.data()
                [block.offset() + row * block.strides()[0] + column * block.strides()[1]];
        }
    }
    matrix
}

/// Relative Frobenius comparison against a row-major `(re, im)` oracle block.
fn assert_sector_matrix_matches<D: FactorScalar>(
    actual: &[D],
    expected_rows: &[&[(f64, f64)]],
    rtol: f64,
    what: &str,
) {
    let order = expected_rows.len();
    assert_eq!(actual.len(), order * order, "{what}: unexpected block size");
    let mut residual = 0.0_f64;
    let mut reference = 0.0_f64;
    for (row, entries) in expected_rows.iter().enumerate() {
        assert_eq!(entries.len(), order, "{what}: oracle block is not square");
        for (column, &(real, imaginary)) in entries.iter().enumerate() {
            let expected = Complex64::new(real, imaginary);
            let got = actual[row + order * column].widen_complex();
            residual += (got - expected).norm_sqr();
            reference += expected.norm_sqr();
        }
    }
    let relative = residual.sqrt() / reference.sqrt();
    assert!(
        relative <= rtol,
        "{what}: relative Frobenius error {relative:e} exceeds {rtol:e}"
    );
}

/// An executor whose selected backend supplies GEMM but no dense solve — the
/// trait default for `solve_into` is what must reach the caller.
#[derive(Default)]
struct SolvelessExecutor;

impl Observer for SolvelessExecutor {}

#[test]
fn exp_fixture_blocks_reproduce_the_oracle_input() {
    // What: fixture certification. Before any exponential is taken, the blocks
    // symtn stores are the blocks Julia's `blocks(t)` filled — read back with
    // the pre-existing block reader, not with anything #577 introduced.
    let tensor = exp_oracle_tensor::<f64>(4.0);
    for (charge, order) in [(0, 3usize), (1, 2usize)] {
        let block = u1_block_matrix(&tensor, charge);
        for column in 0..order {
            for row in 0..order {
                assert_eq!(
                    block[row + order * column],
                    exp_oracle_fill(charge, row, column, 4.0),
                    "fixture entry ({row}, {column}) of charge {charge}"
                );
            }
        }
    }
    // Julia `opnorm(b, 1)` of the same blocks: 9.0 and 6.0, both above
    // theta_13, so the fixture exercises the scaling-and-squaring phase.
    for (charge, order, expected) in [(0, 3usize, 9.0), (1, 2usize, 6.0)] {
        let block = u1_block_matrix(&tensor, charge);
        let norm1 = (0..order)
            .map(|column| {
                (0..order)
                    .map(|row| block[row + order * column].abs())
                    .sum::<f64>()
            })
            .fold(0.0_f64, f64::max);
        assert_eq!(norm1, expected, "one-norm of charge {charge}");
    }
}

#[test]
fn exp_of_a_nilpotent_jordan_block_is_the_terminating_series() {
    // What: a nonnormal, non-diagonalizable block the spectral route cannot
    // touch. exp(J) = 1 + J + J^2/2 + J^3/6 exactly, so the approximant, the
    // scaling choice and the solve are all pinned against closed form.
    let mut jordan = vec![0.0_f64; 16];
    for row in 0..3 {
        jordan[row + 4 * (row + 1)] = 1.0;
    }
    let tensor = u1_block_endomorphism(&[(0, 4, jordan)]);
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let exponential = exp(
        &mut dense,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
    )
    .unwrap();

    let expected = [
        [(1.0, 0.0), (1.0, 0.0), (0.5, 0.0), (1.0 / 6.0, 0.0)],
        [(0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.5, 0.0)],
        [(0.0, 0.0), (0.0, 0.0), (1.0, 0.0), (1.0, 0.0)],
        [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.0)],
    ];
    assert_sector_matrix_matches(
        &u1_block_matrix(exponential.tensor(), 0),
        &expected.iter().map(|row| &row[..]).collect::<Vec<_>>(),
        1e-14,
        "nilpotent Jordan block",
    );
}

#[test]
fn exp_of_a_real_skew_symmetric_block_is_the_analytic_rotation() {
    // What: skew-symmetric is anti-Hermitian, so the eigh gate refuses it while
    // its exponential is the exact rotation by the same angle — at any angle.
    // The large one is the value-level gate on scaling and squaring: at
    // ||A||_1 = 20 the [13/13] approximant is far outside its accuracy range,
    // so an unscaled evaluation misses the rotation by orders of magnitude.
    for angle in [0.7_f64, 20.0] {
        let tensor = u1_block_endomorphism(&[(0, 2, vec![0.0_f64, angle, -angle, 0.0])]);
        let mut dense = symtn_dense::DefaultDenseExecutor::new();

        let exponential = exp(
            &mut dense,
            &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
        )
        .unwrap();

        let expected = [
            [(angle.cos(), 0.0), (-angle.sin(), 0.0)],
            [(angle.sin(), 0.0), (angle.cos(), 0.0)],
        ];
        assert_sector_matrix_matches(
            &u1_block_matrix(exponential.tensor(), 0),
            &expected.iter().map(|row| &row[..]).collect::<Vec<_>>(),
            1e-13,
            &format!("skew-symmetric rotation by {angle}"),
        );
    }
}

/// `exp(A) = (Σ_{k≤30} (A/8)^k / k!)^8`.
///
/// The defining series ([`taylor_exp`]) at `||A/8||_1 ≤ 3/2`, where 30 terms
/// are exact to the last bit, followed by three squarings. It shares no Padé
/// table, no balancing, no `theta` dispatch and no dense solve with the
/// implementation.
fn series_exp(matrix: &[Complex64], order: usize) -> Vec<Complex64> {
    let eighth: Vec<Complex64> = matrix.iter().map(|&entry| entry / 8.0).collect();
    let mut result = taylor_exp(&eighth, order, 30);
    for _ in 0..3 {
        result = matmul(&result, &result, order);
    }
    result
}

/// Asserts `exp` of both coupled sectors of `exp_oracle_tensor::<D>(scale)`
/// against [`series_exp`] of the blocks symtn stores.
fn assert_exp_oracle_tensor_matches_the_series<D: FactorScalar>(scale: f64, what: &str) {
    let tensor = exp_oracle_tensor::<D>(scale);
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let exponential = exp(
        &mut dense,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
    )
    .unwrap();
    for (charge, order) in [(0, 3usize), (1, 2usize)] {
        let input: Vec<Complex64> = u1_block_matrix(&tensor, charge)
            .into_iter()
            .map(FactorScalar::widen_complex)
            .collect();
        let expected = series_exp(&input, order);
        let rows: Vec<Vec<(f64, f64)>> = (0..order)
            .map(|row| {
                (0..order)
                    .map(|column| {
                        let entry = expected[row + order * column];
                        (entry.re, entry.im)
                    })
                    .collect()
            })
            .collect();
        assert_sector_matrix_matches(
            &u1_block_matrix(exponential.tensor(), charge),
            &rows.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            EXP_ORACLE_RTOL,
            &format!("{what} scale {scale} charge {charge}"),
        );
    }
}

#[test]
fn exp_of_a_multisector_u1_endomorphism_matches_the_series() {
    // What: two coupled sectors of different order, f64 and nonnormal,
    // non-Hermitian c64, at a scale below theta_13 (1) and above it (4, where
    // the scaling-and-squaring phase runs), against the defining series.
    for scale in [1.0, 4.0] {
        assert_exp_oracle_tensor_matches_the_series::<f64>(scale, "f64");
        assert_exp_oracle_tensor_matches_the_series::<Complex64>(scale, "c64");
    }
}

#[test]
// Verbatim `%.17g` TensorKit output.
#[allow(clippy::excessive_precision)]
fn exp_of_the_oracle_tensor_matches_tensorkit_anchors() {
    // What: the convention anchor for the series oracle above — the charge-1
    // blocks as printed by TensorKit 0.16.2 (`exp!` hands every block to Julia
    // 1.11.6 `LinearAlgebra.exp!`) for the fill of `exp_oracle_block`.
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let real = exp_oracle_tensor::<f64>(4.0);
    let real = exp(
        &mut dense,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &real),
    )
    .unwrap();
    assert_sector_matrix_matches(
        &u1_block_matrix(real.tensor(), 1),
        &[
            &[(6.8456187388272456, 0.0), (-1.9710572969428048, 0.0)],
            &[(13.797401078599625, 0.0), (-1.0386104489439727, 0.0)],
        ],
        EXP_ORACLE_RTOL,
        "f64 scale 4 charge 1",
    );
    let complex = exp_oracle_tensor::<Complex64>(1.0);
    let complex = exp(
        &mut dense,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &complex),
    )
    .unwrap();
    assert_sector_matrix_matches(
        &u1_block_matrix(complex.tensor(), 1),
        &[
            &[
                (1.7454107392461176, -0.35809087573900988),
                (-0.17904543786950494, 0.179045437869505),
            ],
            &[
                (1.2533180650865348, -0.17904543786950497),
                (1.0292289877680976, 0.35809087573900988),
            ],
        ],
        EXP_ORACLE_RTOL,
        "c64 scale 1 charge 1",
    );
}

#[test]
fn exp_agrees_between_the_direct_region_and_packed_layouts() {
    // What: the same blocks reached through the canonical coupled-sector
    // regions and through the packed matricization fall-back produce the same
    // tensor — the kernel sees identical matrices on both routes.
    let tensor = exp_oracle_tensor::<f64>(4.0);
    let packed = padded_copy(&U1FusionRule, &tensor);
    assert!(tensor
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .is_some());
    assert!(packed
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .is_none());
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let direct = exp(
        &mut dense,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
    )
    .unwrap();
    let fallback = exp(
        &mut dense,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &packed),
    )
    .unwrap();

    for charge in [0, 1] {
        assert_eq!(
            u1_block_matrix(direct.tensor(), charge),
            u1_block_matrix(fallback.tensor(), charge),
            "packed layout changed the exponential of charge {charge}"
        );
    }
}

// ============================================================================
// Multi-tree coupled sectors.
//
// Every fixture above places exactly one fusion tree per coupled sector, so a
// permutation of whole tree-aligned row blocks inside a sector is invisible to
// them: the sector matrix has only one block to permute. `tsvd_test_tensor` at
// rank 2 <- 2 with three U(1) charges and degeneracy 2 gives coupled sectors
// carrying one, two and three trees (orders 4, 8, 12), so the tree layout
// inside a sector matrix has something to get wrong. The oracle below is the
// defining series, which shares no scaling rule, no Padé table, no solve and no
// squaring loop with the implementation, and is read back through the
// pre-existing `coupled_sector_regions` reader.
// ============================================================================

/// Scale of the multi-tree fixture. Chosen so the largest sector 1-norm sits
/// just above `theta_13` — the squaring phase runs, and the series oracle is
/// still deep inside its convergent range at 60 terms.
const EXP_MULTITREE_SCALE: f64 = 0.3;

/// Entrywise agreement with the series oracle, relative to the largest entry of
/// the sector. Loose next to the Padé error because the series is summed in the
/// fixture's own (non-normal) basis; still orders of magnitude below any
/// misplaced tree block, which moves entries by O(1).
const EXP_MULTITREE_TOL: f64 = 1e-11;

/// Column-major `order x order` product, by hand.
fn matmul(left: &[Complex64], right: &[Complex64], order: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::zero(); order * order];
    for column in 0..order {
        for row in 0..order {
            out[row + order * column] = (0..order)
                .map(|inner| left[row + order * inner] * right[inner + order * column])
                .sum();
        }
    }
    out
}

/// `exp(A)` by its defining series `Σ_k A^k / k!`, summed to `terms`.
///
/// Deliberately shares nothing with the implementation: no scaling, no
/// squaring, no Padé coefficients, no dense solve — only GEMM by hand.
fn taylor_exp(matrix: &[Complex64], order: usize, terms: usize) -> Vec<Complex64> {
    let mut term = vec![Complex64::zero(); order * order];
    for index in 0..order {
        term[index + order * index] = Complex64::new(1.0, 0.0);
    }
    let mut result = term.clone();
    for step in 1..=terms {
        term = matmul(&term, matrix, order)
            .into_iter()
            .map(|entry| entry / step as f64)
            .collect();
        for (accumulated, &value) in result.iter_mut().zip(&term) {
            *accumulated += value;
        }
    }
    result
}

/// Column-major coupled-sector matrices of a canonical layout, read through the
/// pre-existing region reader.
fn coupled_sector_matrices(tensor: &TestTensor<f64>) -> Vec<(SectorId, usize, Vec<f64>)> {
    tensor
        .structure()
        .coupled_sector_regions(tensor.fusion_space().nout())
        .unwrap()
        .expect("canonical coupled-sector storage")
        .iter()
        .map(|region| {
            assert_eq!(
                region.rows(),
                region.cols(),
                "endomorphism sector must be square"
            );
            (
                region.coupled(),
                region.rows(),
                tensor.data()[region.range()].to_vec(),
            )
        })
        .collect()
}

/// Rank 2 <- 2 U(1) endomorphism whose coupled sectors carry one, two and three
/// fusion trees, scaled into the oracle's comfortable range.
fn exp_multitree_fixture() -> TestTensor<f64> {
    let sectors: Vec<SectorId> = (0..3).map(|c| U1Irrep::new(c).sector_id()).collect();
    let tensor = tsvd_test_tensor(&U1FusionRule, &sectors);
    TestTensor::<f64>::from_vec_with_fusion_space(
        tensor
            .data()
            .iter()
            .map(|value| value * EXP_MULTITREE_SCALE)
            .collect(),
        tensor.fusion_space().clone(),
    )
    .unwrap()
}

/// Entrywise comparison of a whole tensor's sectors against the series oracle.
fn assert_multitree_exp_matches_the_series(
    exponential: &TestTensor<f64>,
    sources: &[(SectorId, usize, Vec<f64>)],
    what: &str,
) {
    let images = coupled_sector_matrices(exponential);
    assert_eq!(images.len(), sources.len(), "{what}: sector count changed");
    for ((sector, order, source), (image_sector, image_order, image)) in
        sources.iter().zip(images.iter())
    {
        assert_eq!(
            (sector, order),
            (image_sector, image_order),
            "{what}: layout"
        );
        let widened: Vec<Complex64> = source.iter().map(|&v| Complex64::new(v, 0.0)).collect();
        let expected: Vec<f64> = taylor_exp(&widened, *order, 60)
            .into_iter()
            .map(|entry| entry.re)
            .collect();
        let tolerance = EXP_MULTITREE_TOL * expected.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
        for column in 0..*order {
            for row in 0..*order {
                let index = row + order * column;
                let residual = (image[index] - expected[index]).abs();
                assert!(
                    residual <= tolerance,
                    "{what}: sector {sector:?} entry ({row}, {column}) is {} against series {}, \
                     residual {residual:e}",
                    image[index],
                    expected[index]
                );
            }
        }
    }
}

#[test]
fn exp_of_a_multi_tree_sector_matches_the_series_entrywise() {
    // What: the tree layout *inside* a coupled sector. Permuting whole
    // tree-aligned row blocks of a multi-tree sector preserves every norm, the
    // sector count, the solve count and the round trip through exp(-A), so only
    // an entrywise comparison against an independently computed exponential of
    // the *same* source matrix can see it.
    let tensor = exp_multitree_fixture();
    let sources = coupled_sector_matrices(&tensor);
    // Certify the fixture actually is multi-tree, and that it exercises the
    // general arm at a scale where squaring happens.
    assert_eq!(
        sources
            .iter()
            .map(|(_, order, _)| *order)
            .collect::<Vec<_>>(),
        vec![4, 8, 12, 8, 4],
        "fixture no longer has one-, two- and three-tree coupled sectors"
    );
    let widest = sources
        .iter()
        .map(|(_, order, matrix)| {
            (0..*order)
                .map(|column| {
                    (0..*order)
                        .map(|row| matrix[row + order * column].abs())
                        .sum::<f64>()
                })
                .fold(0.0_f64, f64::max)
        })
        .fold(0.0_f64, f64::max);
    assert!(
        // theta_13 = 5.3719..., so this window means s >= 1.
        widest > 5.372 && widest < 12.0,
        "fixture 1-norm {widest} is outside the scaling-and-squaring window"
    );

    let mut spy = ScriptedExecutor::<MatrixFunctionCallSpy>::default();
    let exponential = exp(
        &mut spy,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
    )
    .unwrap();
    assert_eq!(
        spy.counts().of(MATRIX_FUNCTION_EIGH),
        0,
        "the fixture must reach the general arm"
    );
    assert_eq!(spy.counts().solve, sources.len(), "one solve per sector");

    assert_multitree_exp_matches_the_series(exponential.tensor(), &sources, "direct regions");
}

#[test]
fn exp_of_a_multi_tree_sector_matches_the_series_through_the_packed_layout() {
    // What: the same gate on the matricization fall-back, where the sector
    // matrix is assembled tree by tree and written back through
    // `reorder_inverse_solution` — the route that has to rebuild the tree
    // layout rather than inherit it.
    let tensor = exp_multitree_fixture();
    let packed = padded_copy(&U1FusionRule, &tensor);
    assert!(
        packed
            .structure()
            .coupled_sector_regions(2)
            .unwrap()
            .is_none(),
        "the padded copy must take the matricization fall-back"
    );
    let sources = coupled_sector_matrices(&tensor);

    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let exponential = exp(
        &mut dense,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &packed),
    )
    .unwrap();

    // The output layout is always canonical, so the same reader applies.
    assert_multitree_exp_matches_the_series(exponential.tensor(), &sources, "packed layout");
}

#[test]
fn exp_of_a_general_endomorphism_inverts_under_negation() {
    // What: exp(A) exp(-A) = 1 for a non-Hermitian A, the check that catches a
    // sign or a transposition the oracle blocks would also have to agree with.
    let tensor = exp_oracle_tensor::<f64>(1.0);
    let negated = TestTensor::<f64>::from_vec_with_fusion_space(
        tensor.data().iter().map(|value| -value).collect(),
        tensor.fusion_space().clone(),
    )
    .unwrap();
    let rule = U1FusionRule;
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let mut context = default_context();

    let forward = exp(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor)).unwrap();
    let backward = exp(&mut dense, &bound_tensor_ref!(Arc::new(rule), &negated)).unwrap();

    let identity =
        crate::compose::compose(&mut context, &rule, forward.tensor(), backward.tensor()).unwrap();
    for (charge, order) in [(0, 3usize), (1, 2usize)] {
        let block = u1_block_matrix(&identity, charge);
        for column in 0..order {
            for row in 0..order {
                let expected = if row == column { 1.0 } else { 0.0 };
                let residual = (block[row + order * column] - expected).abs();
                assert!(
                    residual <= EXP_INVERSE_RTOL,
                    "charge {charge} entry ({row}, {column}) residual {residual:e}"
                );
            }
        }
    }
}

fn hermitian_exp_fixture() -> TestTensor<f64> {
    u1_block_endomorphism(&[
        (
            0,
            3,
            vec![0.5, 0.25, -0.125, 0.25, -0.75, 0.375, -0.125, 0.375, 1.25],
        ),
        (1, 2, vec![0.25, 0.5, 0.5, -0.125]),
    ])
}

#[test]
fn exp_of_a_hermitian_endomorphism_is_the_spectral_route() {
    // What: the retained route, pinned two ways — the dispatch (one EIGH and
    // one recomposition GEMM per sector, no solve) and the published values,
    // which agree with `v exp(d) v^H` recomposed here by the eigendecomposition
    // seam and the fusion contraction engine on the same backend under the
    // workspace rule (3 = the largest sector order). The result is exactly
    // Hermitian, as Julia's `exp!` publishes it.
    //
    // The reference is computed rather than frozen because a frozen one is a
    // pin on the platform's LAPACK: frozen constants right on macOS are a few
    // ULP off on Linux, and CI would fail on values that are not the point. The spy counters above are what catch a reroute onto
    // Pade; the values check that the route publishes `v exp(d) v^H`.
    let tensor = hermitian_exp_fixture();
    let mut spy = ScriptedExecutor::<MatrixFunctionCallSpy>::default();

    let exponential = exp(
        &mut spy,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
    )
    .unwrap();

    assert_eq!(
        spy.counts().of(MATRIX_FUNCTION_EIGH),
        2,
        "one eigendecomposition per coupled sector"
    );
    assert_eq!(spy.counts().solve, 0, "the Hermitian route must not solve");
    assert_eq!(
        spy.counts().dot_general,
        2,
        "one V exp(D) V^H GEMM per coupled sector"
    );
    for charge in [0, 1] {
        let block = u1_block_matrix(exponential.tensor(), charge);
        let order = (block.len() as f64).sqrt() as usize;
        for column in 0..order {
            for row in 0..order {
                assert_eq!(
                    block[row + order * column].to_bits(),
                    block[column + order * row].to_bits(),
                    "exactly symmetric block {charge}"
                );
            }
        }
    }

    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let mut context = default_context();
    let bound = bound_tensor(Arc::new(U1FusionRule), &tensor);
    let input = bound.as_ref();
    let input = input.dynamic();
    let eigh = eigh_full_dyn(&mut dense, &input, HermitianTol::DEFAULT).unwrap();
    let vh = crate::factorize::adjoint_bound_factor(eigh.v()).unwrap();
    let mut vd = eigh.v().clone();
    let exponentials = eigh
        .eigenvalues()
        .iter()
        .map(|entry| SectorSpectrum {
            sector: entry.sector,
            values: entry.values.iter().map(|value| value.exp()).collect(),
        })
        .collect::<Vec<_>>();
    let vd_space = vd.space().space().clone();
    scale_axis_by_spectrum(&vd_space, vd.data_mut(), None, &exponentials).unwrap();
    let spectral = crate::compose::compose_bound_dyn(&mut context, &vd, &vh).unwrap();
    let spectral: BoundTensorMap<_, _> = typed_from_bound_factor(spectral).unwrap();
    numerics::assert_slices_close(
        "exp",
        exponential.tensor().data(),
        spectral.tensor().data(),
        3,
    );
}

#[test]
fn exp_of_a_hermitian_c64_endomorphism_takes_the_spectral_route() {
    // What: the *dispatch*, not the values. The retained Hermitian route is
    // pinned by value only on U(1)/f64 above, so a hermiticity predicate that
    // misclassified complex input would silently reroute it onto Pade with
    // nothing failing. A solve on Hermitian input is the observable.
    let half = Complex64::new(0.5, 0.0);
    let charge_zero = vec![
        half,
        Complex64::new(0.25, -0.125),
        Complex64::new(-0.125, 0.375),
        Complex64::new(0.25, 0.125),
        Complex64::new(-0.75, 0.0),
        Complex64::new(0.375, -0.25),
        Complex64::new(-0.125, -0.375),
        Complex64::new(0.375, 0.25),
        Complex64::new(1.25, 0.0),
    ];
    let charge_one = vec![
        Complex64::new(1.0, 0.0),
        Complex64::new(0.25, 0.5),
        Complex64::new(0.25, -0.5),
        Complex64::new(-0.75, 0.0),
    ];
    let tensor = u1_block_endomorphism(&[(0, 3, charge_zero), (1, 2, charge_one)]);
    let mut spy = ScriptedExecutor::<MatrixFunctionCallSpy>::default();

    exp(
        &mut spy,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
    )
    .unwrap();

    assert_eq!(
        spy.counts().of(MATRIX_FUNCTION_EIGH),
        2,
        "one eigendecomposition per coupled sector"
    );
    assert_eq!(
        spy.counts().solve,
        0,
        "a Hermitian c64 input must not reach the Pade solve"
    );
}

#[test]
fn exp_of_a_hermitian_su2_endomorphism_takes_the_spectral_route() {
    // What: the same dispatch gate on a non-abelian rule, where the coupled
    // sector matrix carries the recoupling structure the predicate reads.
    let rule = SU2FusionRule;
    let symmetric = |a: f64, b: f64, c: f64| vec![a, b, b, c];
    let blocks = [
        (
            SU2Irrep::from_twice_spin(0).sector_id(),
            2usize,
            symmetric(0.5, 0.25, -0.75),
        ),
        (
            SU2Irrep::from_twice_spin(1).sector_id(),
            2usize,
            symmetric(1.25, -0.375, 0.625),
        ),
        (
            SU2Irrep::from_twice_spin(2).sector_id(),
            2usize,
            symmetric(-0.25, 0.5, 0.125),
        ),
    ];
    let tensor = block_endomorphism(&rule, &blocks);
    let mut spy = ScriptedExecutor::<MatrixFunctionCallSpy>::default();

    exp(&mut spy, &bound_tensor_ref!(Arc::new(rule), &tensor)).unwrap();

    assert_eq!(
        spy.counts().of(MATRIX_FUNCTION_EIGH),
        3,
        "one eigendecomposition per coupled sector"
    );
    assert_eq!(
        spy.counts().solve,
        0,
        "a Hermitian SU(2) input must not reach the Pade solve"
    );
}

#[test]
fn exp_sector_work_scales_with_the_sector_count_and_the_scaling_count() {
    // What: complexity parity. Doubling the number of equal-size sectors
    // doubles the block work instead of coupling them into one dense cube, and
    // a block above theta_13 pays exactly its squarings.
    let block = |charge| exp_oracle_block::<f64>(charge, 2, 1.0);
    let two = u1_block_endomorphism(&[(0, 2, block(0)), (1, 2, block(1))]);
    let four = u1_block_endomorphism(&[
        (0, 2, block(0)),
        (1, 2, block(1)),
        (2, 2, block(2)),
        (3, 2, block(3)),
    ]);

    let mut two_spy = ScriptedExecutor::<MatrixFunctionCallSpy>::default();
    exp(
        &mut two_spy,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &two),
    )
    .unwrap();
    let mut four_spy = ScriptedExecutor::<MatrixFunctionCallSpy>::default();
    exp(
        &mut four_spy,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &four),
    )
    .unwrap();

    assert_eq!(four_spy.counts().solve, 2 * two_spy.counts().solve);
    assert_eq!(
        four_spy.counts().dot_general,
        2 * two_spy.counts().dot_general
    );

    // ||A||_1 = 36 for this block, so s = ceil(log2(36 / theta_13)) = 3 and the
    // squaring loop adds exactly three GEMMs on top of the six.
    let scaled = u1_block_endomorphism(&[(0, 3, exp_oracle_block::<f64>(0, 3, 16.0))]);
    let mut scaled_spy = ScriptedExecutor::<MatrixFunctionCallSpy>::default();
    exp(
        &mut scaled_spy,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &scaled),
    )
    .unwrap();
    assert_eq!(scaled_spy.counts().dot_general, 9);
    assert_eq!(scaled_spy.counts().solve, 1);
}

#[test]
fn exp_balances_a_badly_scaled_block_before_the_pade_evaluation() {
    // What: the balancing step Julia's `exp!` runs before the approximant
    // (`LAPACK.gebal!('B', A)`, stdlib v1.11 `dense.jl:684`) and undoes after
    // it, pinned two ways on `A = [0 1e16; 1e-16 0]`, whose square is the
    // identity and whose exponential is therefore `cosh(1) I + sinh(1) A`.
    // Unbalanced the block has `||A||_1 = 1e16` and pays 51 squarings, each one
    // squaring the approximant's error along with it; balanced it has norm
    // ~1.11, below theta_13, so the *dispatch* — six GEMMs and no squaring —
    // is the sharpest observable there is.
    let tensor = u1_block_endomorphism(&[(0, 2, vec![0.0_f64, 1e-16, 1e16, 0.0])]);
    let mut spy = ScriptedExecutor::<MatrixFunctionCallSpy>::default();

    let exponential = exp(
        &mut spy,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
    )
    .unwrap();

    assert_eq!(
        spy.counts().dot_general,
        6,
        "the balanced block is below theta_13: six GEMMs, no squaring"
    );
    let cosh = 1.0_f64.cosh();
    let sinh = 1.0_f64.sinh();
    let expected = [
        [(cosh, 0.0), (sinh * 1e16, 0.0)],
        [(sinh * 1e-16, 0.0), (cosh, 0.0)],
    ];
    assert_sector_matrix_matches(
        &u1_block_matrix(exponential.tensor(), 0),
        &expected.iter().map(|row| &row[..]).collect::<Vec<_>>(),
        1e-14,
        "balanced [0 1e16; 1e-16 0]",
    );
}

#[test]
fn exp_undoes_the_balancing_permutation_and_scaling_like_julia() {
    // What: the *undo* half of balancing, on a block that exercises both halves
    // of `gebal('B')` at once — a column that isolates an eigenvalue, so the
    // permutation moves it to the front, and a badly scaled remainder, so the
    // diagonal similarity acts on the window behind it. The regression fixture
    // above needs no permutation and so cannot see an undo applied in the wrong
    // order; this one can.
    //
    // That both halves fired is asserted directly on `balance_in_place`, against
    // Julia 1.11.6:
    //
    // ```julia
    // d = [1e8, 1.0, 1e-8]
    // B = [0.0 0.0 1.0; 2.0 3.0 5.0; 1.0 0.0 0.0]
    // A = [B[i, j] * d[i] / d[j] for i in 1:3, j in 1:3]
    // LinearAlgebra.LAPACK.gebal!('B', copy(A))
    // # (2, 3, [2.0, 9.007199254740992e15, 1.0])       # 9.007...e15 == 2^53
    // ```
    //
    // `ilo, ihi = 2, 3` is the permutation, `scale[2] = 2^53` the scaling.
    //
    // The *values* are checked against an exact oracle rather than against a
    // recorded `exp(A)`: `A = D B D^-1`, so `exp(A) = D exp(B) D^-1` entry by
    // entry, and `exp(B)` is available in closed form — its `{1,3}` corner is
    // `[cosh 1, sinh 1; sinh 1, cosh 1]`, its `(2,2)` entry `e^3`, and the two
    // remaining entries are the integrals `e^3 (7 (1 - e^-2) -/+ 1.5 (1 - e^-4)) / 4`.
    // `expected` below is that closed form, cross-checked to 17 digits against a
    // 400-bit BigFloat scaling-and-squaring Taylor evaluation of `exp(B)` in
    // Julia. Recording a `Float64` `exp(A)` instead is what broke CI: the
    // fixture's own conditioning makes the answer platform-dependent in the
    // ninth digit, so no recorded double is portable at a tight tolerance.
    //
    // The tolerance is that conditioning, made explicit: balancing equalizes the
    // window's norms without seeing the isolated column, so the balanced block
    // still has `||A||_1 ~ 7e8` and takes 27 squarings, each of which roughly
    // doubles the relative error. `2^27 * eps ~ 3.0e-8` bounds it; `1e-7` is
    // that bound with a factor of three of room. Julia's own answer sits at
    // 2.24e-8 on macOS and 2.15e-8 on Linux, i.e. the *whole* macOS/Linux spread
    // is 1.1e-9, two orders inside the gate. The structural zeros stay EXACT.
    let diagonal = [1e8_f64, 1.0, 1e-8];
    let rows = [[0.0_f64, 0.0, 1.0], [2.0, 3.0, 5.0], [1.0, 0.0, 0.0]];
    let mut block = vec![0.0_f64; 9];
    for (row, entries) in rows.iter().enumerate() {
        for (column, &entry) in entries.iter().enumerate() {
            block[row + 3 * column] = entry * diagonal[row] / diagonal[column];
        }
    }
    assert_gebal_matches(
        &block,
        3,
        (1, 2),
        &[2.0, 2.0_f64.powi(53), 1.0],
        "the permuting-and-scaling fixture",
    );

    let tensor = u1_block_endomorphism(&[(0, 3, block)]);
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let exponential = exp(
        &mut dense,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
    )
    .unwrap();

    let (cosh, sinh, e3) = (1.0_f64.cosh(), 1.0_f64.sinh(), 3.0_f64.exp());
    let (even, odd) = (
        7.0 * (1.0 - (-2.0_f64).exp()),
        1.5 * (1.0 - (-4.0_f64).exp()),
    );
    let exp_b = [
        [cosh, 0.0, sinh],
        [e3 * (even - odd) / 4.0, e3, e3 * (even + odd) / 4.0],
        [sinh, 0.0, cosh],
    ];
    let expected: [[f64; 3]; 3] = std::array::from_fn(|row| {
        std::array::from_fn(|column| exp_b[row][column] * diagonal[row] / diagonal[column])
    });
    let actual = u1_block_matrix(exponential.tensor(), 0);
    for (row, entries) in expected.iter().enumerate() {
        for (column, &want) in entries.iter().enumerate() {
            let got = actual[row + 3 * column];
            // The two structural zeros are the permutation's own signature: the
            // isolated column stays exactly isolated, to the last bit. Every
            // other entry gets the squaring-error budget derived above.
            let tolerance = if want == 0.0 { 0.0 } else { 1e-7 * want.abs() };
            assert!(
                (got - want).abs() <= tolerance,
                "entry ({row}, {column}): {got:.17e} differs from the oracle's {want:.17e}"
            );
        }
    }
}

/// Column-major `order x order` block from row-major rows.
fn column_major<D: Copy + Zero>(rows: &[&[D]]) -> Vec<D> {
    let order = rows.len();
    let mut block = vec![D::zero(); order * order];
    for (row, entries) in rows.iter().enumerate() {
        assert_eq!(entries.len(), order);
        for (column, &entry) in entries.iter().enumerate() {
            block[row + order * column] = entry;
        }
    }
    block
}

/// Asserts `balance_in_place` reproduces `LAPACK.gebal!('B', A)`, whose output
/// is quoted 0-based here: LAPACK's `ilo`/`ihi` are 1-based and inclusive,
/// symtn's are 0-based and inclusive.
fn assert_gebal_matches<D: FactorScalar>(
    block: &[D],
    order: usize,
    expected_window: (usize, usize),
    expected_scale: &[f64],
    what: &str,
) {
    let mut matrix = block.to_vec();
    let mut scale = vec![0.0_f64; order];
    let window = crate::matrix_functions::balance_in_place(&mut matrix, order, &mut scale);
    assert_eq!(window, expected_window, "{what}: window");
    assert_eq!(scale, expected_scale, "{what}: scale");
}

#[test]
fn balancing_measures_rows_and_columns_with_the_euclidean_norm_like_gebal() {
    // What: `gebal` measures its rows and columns with `DNRM2`
    // (`dgebal.f:341-342`), not with a sum of `abs1`. The fixture is chosen so
    // the two disagree — the `abs1` sum reaches `scale = [1, 1/2, 1]` here,
    // where LAPACK reaches `[2, 1, 1]` — so it is the norm itself under test
    // and not merely the loop around it.
    //
    // Oracle, Julia 1.11.6:
    //
    // ```julia
    // LinearAlgebra.LAPACK.gebal!('B', Float64[0 4 0; 1 0 1; 1 1 0])
    // # (1, 3, [2.0, 1.0, 1.0])
    // ```
    //
    // `Float32` and `ComplexF64` of the same matrix give the same triple, so
    // all three are pinned to the one certified answer.
    let rows: [&[f64]; 3] = [&[0.0, 4.0, 0.0], &[1.0, 0.0, 1.0], &[1.0, 1.0, 0.0]];
    let block = column_major(&rows);
    assert_gebal_matches(
        &block,
        3,
        (0, 2),
        &[2.0, 1.0, 1.0],
        "f64 [0 4 0; 1 0 1; 1 1 0]",
    );

    let single = block.iter().map(|&entry| entry as f32).collect::<Vec<_>>();
    assert_gebal_matches(&single, 3, (0, 2), &[2.0, 1.0, 1.0], "f32 of the same");

    let complex = block
        .iter()
        .map(|&entry| Complex64::new(entry, 0.0))
        .collect::<Vec<_>>();
    assert_gebal_matches(&complex, 3, (0, 2), &[2.0, 1.0, 1.0], "c64 of the same");

    // And the span is the whole window *including* the diagonal:
    // `DNRM2(L-K+1, A(K,I), 1)` is contiguous, it does not skip `A(I,I)`.
    // Dropping the diagonal reaches `[4, 1, 1]` on this second fixture.
    //
    // ```julia
    // LinearAlgebra.LAPACK.gebal!('B', Float64[-4 -1 -9; -1 1 -7; 0 -9 -7])
    // # (1, 3, [2.0, 1.0, 1.0])
    // ```
    let with_diagonal: [&[f64]; 3] = [&[-4.0, -1.0, -9.0], &[-1.0, 1.0, -7.0], &[0.0, -9.0, -7.0]];
    assert_gebal_matches(
        &column_major(&with_diagonal),
        3,
        (0, 2),
        &[2.0, 1.0, 1.0],
        "f64 [-4 -1 -9; -1 1 -7; 0 -9 -7]",
    );
}

#[test]
fn balancing_sizes_a_complex_iamax_element_by_its_modulus() {
    // What: `zgebal` *selects* `CA`/`RA` with `IZAMAX`, which compares
    // `|Re| + |Im|`, but then takes the **modulus** of the element it selected
    // (`zgebal.f:348-351`). The two differ by up to `sqrt(2)`, most of a radix
    // step, so they can stop the scaling loop one factor of two apart.
    //
    // The fixture makes that visible: column 1 isolates an eigenvalue, so the
    // window starts at row 2 while `CA` still scans row 1, letting the entry
    // `(3 + 4i) * 1e271` — `abs1` 7e271, modulus 5e271 — dominate `CA` and land
    // between `sfmax2 / 2` and `sfmax2` after the loop's doublings.
    //
    // Oracle, Julia 1.11.6:
    //
    // ```julia
    // A = ComplexF64[7 (3+4im)*1e271 0; 0 0 1e41; 0 1 0]
    // LinearAlgebra.LAPACK.gebal!('B', copy(A))
    // # (2, 3, [1.0, 1.4757395258967641e20, 0.5])   # 1.4757...e20 == 2^67
    // ```
    //
    // Sizing `CA`/`RA` by `abs1` instead gives `[1, 2^66, 0.25]`.
    let huge = Complex64::new(3.0, 4.0) * 1e271;
    let zero = Complex64::new(0.0, 0.0);
    let rows: [&[Complex64]; 3] = [
        &[Complex64::new(7.0, 0.0), huge, zero],
        &[zero, zero, Complex64::new(1e41, 0.0)],
        &[zero, Complex64::new(1.0, 0.0), zero],
    ];
    assert_gebal_matches(
        &column_major(&rows),
        3,
        (1, 2),
        &[1.0, 2.0_f64.powi(67), 0.5],
        "c64 modulus-vs-abs1 fixture",
    );
}

#[test]
fn balancing_takes_its_machine_bounds_from_the_single_precision_component() {
    // What: `sgebal`/`cgebal` derive `SFMIN1` and friends from `SLAMCH`
    // (`sgebal.f:330`), `dgebal`/`zgebal` from `DLAMCH` (`dgebal.f:330`). Under
    // the double bounds this fixture's radix loop runs to a factor around
    // `2^970`, which is `inf` in `f32`; under the single bounds it stops at
    // `2^102`, which is not.
    //
    // Oracle, Julia 1.11.6:
    //
    // ```julia
    // LinearAlgebra.LAPACK.gebal!('B', Float32[0 3f38; 1f-45 0])
    // # (1, 2, Float32[5.0706024f30, 1.4551915f-11])  # 2^102 and 2^-36
    // ```
    //
    // The same triple certifies `ComplexF32`.
    let rows: [&[f32]; 2] = [&[0.0, 3e38], &[1e-45, 0.0]];
    let expected = [2.0_f64.powi(102), 2.0_f64.powi(-36)];
    assert_gebal_matches(&column_major(&rows), 2, (0, 1), &expected, "f32 edge block");

    let complex = column_major(&rows)
        .iter()
        .map(|&entry| Complex32::new(entry, 0.0))
        .collect::<Vec<_>>();
    assert_gebal_matches(&complex, 2, (0, 1), &expected, "c32 edge block");
}

#[test]
fn exp_of_a_single_precision_block_spanning_the_whole_f32_range() {
    // What: through the public facade, every entry of
    // `A = [0 3e38; 1e-45 0]` and of its exponential is a finite `f32`.
    // Balancing with the *double* machine bounds would produce a factor that
    // is `inf` in `f32` and send the Pade evaluation to all-NaN.
    //
    // Oracle, Julia 1.11.6: `exp(Float32[0 3f38; 1f-45 0])` is
    // `Float32[1.0000002 3.0000004f38; 1.0f-45 1.0000002]`, and the `ComplexF32`
    // matrix gives the same with zero imaginary parts. The tolerance is a
    // single-precision one: the oracle is quoted to the eight digits `f32`
    // carries.
    let rows: [&[f32]; 2] = [&[0.0, 3e38], &[1e-45, 0.0]];
    let expected: [&[(f64, f64)]; 2] = [
        &[(1.0000002, 0.0), (3.0000004e38, 0.0)],
        &[(1e-45, 0.0), (1.0000002, 0.0)],
    ];
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let tensor = u1_block_endomorphism(&[(0, 2, column_major(&rows))]);
    let exponential = exp(
        &mut dense,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
    )
    .unwrap();
    assert_sector_matrix_matches(
        &u1_block_matrix(exponential.tensor(), 0),
        &expected,
        1e-6,
        "f32 [0 3e38; 1e-45 0]",
    );

    let complex_block = column_major(&rows)
        .iter()
        .map(|&entry| Complex32::new(entry, 0.0))
        .collect::<Vec<_>>();
    let tensor = u1_block_endomorphism(&[(0, 2, complex_block)]);
    let exponential = exp(
        &mut dense,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
    )
    .unwrap();
    assert_sector_matrix_matches(
        &u1_block_matrix(exponential.tensor(), 0),
        &expected,
        1e-6,
        "c32 [0 3e38; 1e-45 0]",
    );
}

#[test]
fn exp_reports_unsupported_when_the_executor_cannot_solve() {
    // What: device storage whose backend supplies GEMM but no solve is refused
    // in the backend's own words, not silently routed onto the host.
    let tensor = exp_oracle_tensor::<f64>(1.0);
    let mut dense = ScriptedExecutor::<SolvelessExecutor>::default();

    let error = exp(
        &mut dense,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
    )
    .unwrap_err();

    assert!(
        matches!(
            error,
            OperationError::Dense(DenseError::Unsupported {
                op: "solve_into",
                ..
            })
        ),
        "unexpected error {error:?}"
    );
}

#[test]
fn exp_rejects_a_non_endomorphism() {
    // What: TensorKit's own precondition (`domain == codomain`) still holds —
    // #577 widens which endomorphisms are accepted, not which maps are.
    let rule = U1FusionRule;
    let codomain = SectorLeg::new([(U1Irrep::new(0).sector_id(), 3)], false);
    let domain = SectorLeg::new([(U1Irrep::new(0).sector_id(), 2)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([codomain]),
        FusionProductSpace::new([domain]),
    );
    let space = DynamicFusionMapSpace::from_final_homspace(&rule, homspace).unwrap();
    let tensor = TestTensor::<f64>::from_vec_with_fusion_space(
        (0..space.required_len().unwrap())
            .map(|index| index as f64)
            .collect(),
        space,
    )
    .unwrap();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let error = exp(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor)).unwrap_err();

    assert!(
        matches!(
            error,
            OperationError::SpaceMismatch {
                message: "exp requires an endomorphism (codomain == domain)"
            }
        ),
        "an exp caller must be refused in exp's words, not eigh's: {error:?}"
    );
}

#[test]
fn exp_rejects_a_nonfinite_general_block() {
    // What: a NaN block is not Hermitian to MatrixAlgebraKit's predicate, so it
    // arrives at the general arm; it must be named there rather than handed to
    // the backend as a silent NaN.
    let tensor = u1_block_endomorphism(&[(0, 2, vec![1.0_f64, 0.5, f64::NAN, 2.0])]);
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let error = exp(
        &mut dense,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
    )
    .unwrap_err();

    assert!(
        matches!(
            error,
            OperationError::InvalidArgument {
                message: "exp requires finite coupled-sector blocks"
            }
        ),
        "unexpected error {error:?}"
    );
}

#[test]
fn exp_rejects_a_block_whose_column_norm_overflows() {
    // What: every entry is finite but the column 1-norm is not. The squaring
    // count is `ceil(log2(inf / theta_13))` cast to `u32`, which saturates to
    // `u32::MAX`; read back as `i32` that is -1, so the block would be scaled
    // *up* and then squared ~4.3e9 times — a finite input that never returns.
    // The overflowing norm has to be refused where it is computed.
    //
    // The norm in question is the *balanced* one, since balancing runs first,
    // so the fixture is one balancing cannot rescue: every entry is within a
    // factor of five of `f64::MAX`, which puts LAPACK's overflow guards (`ca`
    // and `ra` against `sfmax2`) in the way of any radix step, and the column
    // sum stays infinite. A block whose imbalance balancing *can* undo —
    // `[1e308 1; 1e308 2]` — is not refused (#577 balancing), and is not
    // meant to be: Julia reaches the same finite
    // balanced norm on it and exponentiates it.
    let tensor = u1_block_endomorphism(&[(0, 2, vec![1e308_f64, 1e308, 2e307, 1e308])]);
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let error = exp(
        &mut dense,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
    )
    .unwrap_err();

    assert!(
        matches!(
            error,
            OperationError::InvalidArgument {
                message: "exp requires coupled-sector blocks with a finite 1-norm"
            }
        ),
        "unexpected error {error:?}"
    );
}

#[test]
fn exp_publishes_nothing_when_a_later_sector_fails() {
    // What: failure atomicity. A backend failure on the second sector leaves no
    // tensor and does not touch the input's storage.
    let tensor = exp_oracle_tensor::<f64>(1.0);
    let before = tensor.data().to_vec();
    let mut spy = matrix_function_spy(Some(2));

    let error = exp(
        &mut spy,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
    )
    .unwrap_err();

    assert!(
        matches!(
            error,
            OperationError::Dense(DenseError::Backend {
                op: "solve_into",
                ..
            })
        ),
        "unexpected error {error:?}"
    );
    assert_eq!(
        spy.counts().solve,
        2,
        "the failing sector must have been tried"
    );
    assert_eq!(tensor.data(), &before[..], "input storage was mutated");
}

#[test]
fn pinv_cutoff_rejects_a_non_finite_singular_value_instead_of_dropping_it() {
    // What: both dense pinv routes (`inverted_pinv_spectrum`, the staged
    // checked-Generic fold) take their cutoff from `pinv_cutoff`. A NaN in a
    // later sector must not vanish into `f64::max` and leave the finite
    // `rcond * 4` cutoff the other sectors alone would give.
    let sectors: [&[f64]; 3] = [&[4.0, 1.0], &[f64::NAN, 0.5], &[2.0]];
    let flat = || sectors.iter().flat_map(|values| values.iter().copied());
    assert!(matches!(
        pinv_cutoff(flat(), 0.25),
        Err(OperationError::InvalidArgument { .. })
    ));
    for poison in [f64::INFINITY, f64::NEG_INFINITY] {
        assert!(pinv_cutoff([1.0, poison], 0.0).is_err(), "{poison}");
    }
    // Finite spectra keep the global `rcond * sigma_max`, max in any sector.
    assert_eq!(pinv_cutoff([1.0, 0.5, 4.0, 2.0], 0.25).unwrap(), 1.0);
    assert_eq!(pinv_cutoff(std::iter::empty(), 0.25).unwrap(), 0.0);
}

#[test]
fn hermitian_exp_accepts_consistently_reordered_tree_stacking_with_facade_values() {
    // What: reversing rows and columns together is the same operator in a
    // permuted basis, so exp's spectral route passes the stacking guard and
    // publishes the facade's values block by block: `V f(D) V^H` composes
    // eigenvectors published in the fresh factor layout by tree identity
    // (#1518). The Padé route refuses the layout explicitly instead
    // ("inverse output tree basis does not transpose the source basis").
    let rule = U1FusionRule;
    let sectors = [-1, 0, 1].map(|charge| U1Irrep::new(charge).sector_id());
    let scaled = |tensor: TestTensor<f64>| {
        TestTensor::<f64>::from_vec_with_fusion_space(
            tensor.data().iter().map(|value| 0.1 * value).collect(),
            tensor.fusion_space().clone(),
        )
        .unwrap()
    };
    let provider = Arc::new(rule);
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    {
        let facade = scaled(hermitian_test_tensor(&rule, &sectors));
        let reordered = reversed_coupled_tree_basis_copy(&rule, &facade);
        let expected = exp(
            &mut dense,
            &bound_tensor_ref!(Arc::clone(&provider), &facade),
        )
        .unwrap();
        let actual = exp(
            &mut dense,
            &bound_tensor_ref!(Arc::clone(&provider), &reordered),
        )
        .unwrap();
        let (expected, actual) = (expected.tensor(), actual.tensor());
        let scale = expected
            .data()
            .iter()
            .fold(1.0f64, |max, value| max.max(value.abs()));
        let structure = expected.structure();
        for index in 0..structure.block_count() {
            let key = structure.block(index).unwrap().key().clone();
            let (left, right) = (
                expected.block_by_key(&key).unwrap(),
                actual.block_by_key(&key).unwrap(),
            );
            assert_eq!(left.shape(), right.shape());
            let mut multi = vec![0usize; left.shape().len()];
            loop {
                let at = |view: &symtn_core::BlockView<'_, f64>| {
                    view.data()[view.offset()
                        + multi
                            .iter()
                            .zip(view.strides())
                            .map(|(&index, &stride)| index * stride)
                            .sum::<usize>()]
                };
                let (a, b) = (at(&left), at(&right));
                assert!(
                    (a - b).abs() <= 1e-12 * scale,
                    "{key:?} {multi:?}: {a} vs {b}"
                );
                let Some(axis) = (0..multi.len()).find(|&axis| {
                    multi[axis] += 1;
                    if multi[axis] < left.shape()[axis] {
                        true
                    } else {
                        multi[axis] = 0;
                        false
                    }
                }) else {
                    break;
                };
                let _ = axis;
            }
        }
    }
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_factor_isomorphism_queries_codomain_then_domain_without_shortcut() {
    // What: the checked inverse/solve admission compares the coupled
    // dimensions of both sides, codomain first, even when the two sides are
    // equal, and reports a provider failure as a plan error.
    // The sources are admitted for `FactorGenericRule`, and this admission
    // reads no structure cache, so the spies may answer under its identity.
    let spy = |fail_at| {
        Arc::new(LateGenericSpy {
            rule: FactorGenericRule,
            fail_at,
            calls: Cell::new(0),
            identity: FactorGenericRule.rule_identity(),
        })
    };
    let (square, _, _) = generic_values_endomorphism_input();
    let (rectangular, _) = generic_factorization_input();
    for (source, expected) in [(&square, true), (&rectangular, false)] {
        let space =
            BoundDynamicFusionMapSpace::bind_generic(source.space().clone(), spy(usize::MAX))
                .unwrap();
        assert_eq!(factor_isomorphic_checked_generic(&space).unwrap(), expected);
        let total = space.provider().calls.get();
        assert!(total > 0);

        let probe = spy(usize::MAX);
        crate::factorize::coupled_sector_block_dimensions_generic_checked(
            source.space().homspace().codomain(),
            probe.as_ref(),
        )
        .unwrap();
        let codomain_calls = probe.calls.get();
        assert!(codomain_calls < total);

        // The first domain query fails: the codomain was queried in full.
        let failing = BoundDynamicFusionMapSpace::bind_generic(
            source.space().clone(),
            spy(codomain_calls + 1),
        )
        .unwrap();
        assert!(matches!(
            factor_isomorphic_checked_generic(&failing),
            Err(CheckedGenericFactorPlanError::Provider(LateGenericError(call)))
                if call == codomain_calls + 1
        ));
        assert_eq!(failing.provider().calls.get(), codomain_calls + 1);
    }
}

#[test]
fn mf_pinv_exp_and_polar_publish_no_factor_layout() {
    // What (#1752): the multiplicity-free pseudo-inverse, Hermitian exponential
    // and polar factors recompose each coupled sector into their output by one
    // GEMM: no factor space is staged, no SVD factor is scattered into a
    // factor layout, and no fusion contraction runs (the seams take no
    // contraction context). Region and packed inputs, U(1)-like Z2 and SU(2)
    // with degeneracy 2.
    fn check<R>(rule: R, sectors: &[SectorId])
    where
        R: MultiplicityFreeRigidSymbols<Scalar = f64> + Clone,
    {
        let provider = Arc::new(rule.clone());
        for padded in [false, true] {
            let general = tsvd_test_tensor(&rule, sectors);
            let hermitian = hermitian_test_tensor(&rule, sectors);
            let (general, hermitian) = if padded {
                (padded_copy(&rule, &general), padded_copy(&rule, &hermitian))
            } else {
                (general, hermitian)
            };
            let general = bound_tensor(Arc::clone(&provider), &general);
            let hermitian = bound_tensor(Arc::clone(&provider), &hermitian);
            let (general, hermitian) = (general.as_ref(), hermitian.as_ref());
            let (general, hermitian) = (general.dynamic(), hermitian.dynamic());
            let mut dense = ScriptedExecutor::<MatrixFunctionCallSpy>::default();
            let stage_rec = crate::probe::record();
            let stages = || stage_rec.get(crate::probe::Event::MfFactorSpaceStage);
            let compact_svd_copy_rec = crate::probe::record();
            let before = stages();

            pinv_into_mf(&mut dense, &general, 1e-12).unwrap();
            pinv_adjoint_parent_into_mf(&mut dense, &general, 1e-12).unwrap();
            left_polar_dyn(&mut dense, &general).unwrap();
            right_polar_dyn(&mut dense, &general).unwrap();
            let solves = dense.counts().solve;
            exp_into_mf(&mut dense, &hermitian).unwrap();
            assert_eq!(dense.counts().solve, solves, "Hermitian exp is spectral");

            assert_eq!(stages(), before, "padded={padded}: factor spaces staged");
            assert_eq!(
                compact_svd_copy_rec.compact_svd_copy(),
                Default::default(),
                "padded={padded}: SVD factor layout"
            );
        }
    }
    check(Z2FusionRule, &[SectorId::new(0), SectorId::new(1)]);
    check(
        SU2FusionRule,
        &[
            SU2Irrep::from_twice_spin(0).sector_id(),
            SU2Irrep::from_twice_spin(1).sector_id(),
        ],
    );
}

#[test]
fn exp_of_a_general_endomorphism_runs_one_solve_per_sector_and_no_eigh() {
    // What: the general arm's per-sector budget — six GEMMs and one solve for a
    // block that needs no squaring, and never an eigendecomposition.
    let tensor = exp_oracle_tensor::<f64>(1.0);
    let mut spy = ScriptedExecutor::<MatrixFunctionCallSpy>::default();

    exp(
        &mut spy,
        &bound_tensor_ref!(Arc::new(U1FusionRule), &tensor),
    )
    .unwrap();

    assert_eq!(
        spy.counts().of(MATRIX_FUNCTION_EIGH),
        0,
        "the general arm must not eigendecompose"
    );
    assert_eq!(
        spy.counts().solve,
        2,
        "one solve per nonempty coupled sector"
    );
    assert_eq!(
        spy.counts().dot_general,
        12,
        "six GEMMs per sector at s = 0"
    );
}

#[test]
fn inv_propagates_unsupported_solve_without_svd_fallback() {
    // What: a dense executor without solve support returns its typed capability
    // error instead of silently changing inverse algorithms.
    let tensor = u1_block_endomorphism(&[(0, 1, vec![2.0_f64])]);
    let mut dense = ScriptedExecutor::new(RejectExecutorCalls);
    let bound = bound_tensor(Arc::new(U1FusionRule), &tensor);

    let error = inv_into_mf(&mut dense, &bound.as_ref().dynamic()).unwrap_err();

    assert!(matches!(
        error,
        OperationError::Dense(DenseError::Unsupported {
            op: "solve_into",
            ..
        })
    ));
}
