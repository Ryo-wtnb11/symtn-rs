use super::*;

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

#[test]
fn pinv_adjoint_parent_uses_one_parent_svd_and_the_shared_global_cutoff() {
    // What: the largest singular value is in the later sector, and the first
    // sector sits exactly on the strict global cutoff. The parent-native seam
    // runs one SVD per stored sector, preserves provider identity, and emits
    // the final logical-adjoint orientation directly.
    let canonical = u1_block_endomorphism(&[(0, 1, vec![0.5_f64]), (1, 1, vec![1.0])]);
    let tensor = padded_copy(&U1FusionRule, &canonical);
    let provider = Arc::new(U1FusionRule);
    let bound = bound_tensor(Arc::clone(&provider), &tensor);
    let mut dense = ScriptedExecutor::<SvdCallSpy>::default();
    let compact_svd_copy_rec = crate::probe::record();
    let input_pack_bytes_rec = crate::probe::record();

    let output = pinv_adjoint_parent_into_mf(&mut dense, &bound.as_ref().dynamic(), 0.5).unwrap();
    assert_eq!(dense.counts().svd, 2);
    assert!(Arc::ptr_eq(output.space().provider_arc(), &provider));
    let output: BoundTensorMap<_, _> = typed_from_bound_factor(output).unwrap();
    assert_eq!(scalar_u1_block(output.tensor(), 0), 0.0);
    assert!((scalar_u1_block(output.tensor(), 1) - 1.0).abs() < 1e-12);
    // The padded input is packed once; no SVD factor is laid out.
    assert!(input_pack_bytes_rec.get(crate::probe::Event::InputPackBytes) > 0);
    assert_eq!(compact_svd_copy_rec.compact_svd_copy(), Default::default());
}

#[test]
fn pinv_adjoint_parent_rejects_invalid_rcond_before_svd() {
    // What: the hidden seam owns the same validation precedence as ordinary
    // pinv, independently of either facade.
    let tensor = u1_block_endomorphism(&[(0, 1, vec![1.0_f64])]);
    let bound = bound_tensor(Arc::new(U1FusionRule), &tensor);
    for rcond in [-1.0, f64::NAN, f64::INFINITY] {
        let mut dense = ScriptedExecutor::new(RejectExecutorCalls);
        assert!(matches!(
            pinv_adjoint_parent_into_mf(&mut dense, &bound.as_ref().dynamic(), rcond,),
            Err(OperationError::InvalidArgument { .. })
        ));
    }
}

#[test]
fn pinv_adjoint_parent_discards_unpublished_factors_on_late_svd_failure() {
    // What: a successful first sector cannot publish factors or an output when
    // the second sector's SVD fails.
    let canonical = u1_block_endomorphism(&[(0, 1, vec![2.0_f64]), (1, 1, vec![3.0])]);
    let tensor = padded_copy(&U1FusionRule, &canonical);
    let before = tensor.data().to_vec();
    let bound = bound_tensor(Arc::new(U1FusionRule), &tensor);
    let mut dense = ScriptedExecutor::<FailSecondSvd>::default();
    let compact_svd_copy_rec = crate::probe::record();
    let input_pack_bytes_rec = crate::probe::record();

    assert!(matches!(
        pinv_adjoint_parent_into_mf(&mut dense, &bound.as_ref().dynamic(), 0.0,),
        Err(OperationError::Dense(DenseError::Backend {
            op: "svd_into",
            ..
        }))
    ));
    assert_eq!(dense.counts().of(&[Op::Svd, Op::SvdInto]), 2);
    assert_eq!(tensor.data(), before);
    // The padded input is packed once; no SVD factor is laid out.
    assert!(input_pack_bytes_rec.get(crate::probe::Event::InputPackBytes) > 0);
    assert_eq!(compact_svd_copy_rec.compact_svd_copy(), Default::default());
}

#[test]
fn polar_validates_every_sector_before_direct_or_fallback_svd_execution() {
    // What: a later invalid sector prevents SVD of an earlier valid sector on both layouts.
    let rule = Z2FusionRule;
    let direct = mixed_rectangular_tensor((4, 2), (1, 3));
    let direct_bound = bound_tensor(Arc::new(rule), &direct);
    assert!(
        crate::factorize::compact_factor_plan_for_test(direct_bound.space())
            .unwrap()
            .is_some()
    );
    let mut dense = ScriptedExecutor::<SvdCallSpy>::default();
    let direct_error = left_polar(&mut dense, &direct_bound.as_ref()).unwrap_err();
    assert!(matches!(
        direct_error,
        OperationError::InvalidArgument { message }
            if message.contains("left_polar")
                && message.contains("coupled-sector")
    ));
    assert_eq!(dense.counts().svd, 0);

    let fallback_source = mixed_rectangular_tensor((2, 4), (3, 1));
    let fallback_bound = bound_tensor(Arc::new(rule), &fallback_source);
    let fallback_space = fallback_bound.space().adjoint_view().unwrap();
    assert!(
        crate::factorize::compact_factor_plan_for_test(&fallback_space)
            .unwrap()
            .is_none()
    );
    let fallback_input =
        BoundDynamicTensorRef::try_new(&fallback_space, fallback_bound.data()).unwrap();
    let mut dense = ScriptedExecutor::<SvdCallSpy>::default();
    let fallback_error = left_polar_dyn(&mut dense, &fallback_input).unwrap_err();
    assert!(matches!(
        fallback_error,
        OperationError::InvalidArgument { message }
            if message.contains("left_polar")
                && message.contains("coupled-sector")
    ));
    assert_eq!(dense.counts().svd, 0);
}

#[test]
fn single_precision_svd_and_eig_work_end_to_end() {
    use num_complex::Complex32;
    let rule = Z2FusionRule;
    let sectors = [SectorId::new(0), SectorId::new(1)];
    let degeneracy = 2usize;
    let leg = || SectorLeg::new(sectors.iter().map(|&sector| (sector, degeneracy)), false);
    let homspace = || {
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg()]),
            FusionProductSpace::new([leg(), leg()]),
        )
    };
    let space = || {
        let hom = homspace();
        DynamicFusionMapSpace::from_final_homspace(&rule, hom).unwrap()
    };
    let f32_space = space();
    let len = f32_space.required_len().unwrap();
    let tensor_f32 = TestTensor::<f32>::from_vec_with_fusion_space(
        (0..len)
            .map(|i| ((i * 7 + 3) % 23) as f32 * 0.5 - 5.0)
            .collect(),
        f32_space,
    )
    .unwrap();

    let mut dense_executor = symtn_dense::DefaultDenseExecutor::new();
    let svd = svd_compact(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &tensor_f32),
    )
    .unwrap();

    // Reconstruct through an f32 contraction at single precision.
    let mut scaled_vh = svd.vh.tensor().clone();
    {
        let structure = std::sync::Arc::clone(scaled_vh.structure());
        for index in 0..structure.block_count() {
            let block = structure.block(index).unwrap();
            let BlockKey::FusionTree(key) = block.key() else {
                continue;
            };
            let sector = key.codomain_tree().coupled();
            let values = &svd
                .singular_values
                .iter()
                .find(|entry| entry.sector == sector)
                .unwrap()
                .values;
            let shape = block.shape().to_vec();
            let strides = block.strides().to_vec();
            let offset = block.offset();
            let count = shape.iter().product::<usize>();
            let mut indices = vec![0usize; shape.len()];
            for _ in 0..count {
                let position = offset
                    + indices
                        .iter()
                        .zip(&strides)
                        .map(|(&i, &s)| i * s)
                        .sum::<usize>();
                scaled_vh.data_mut()[position] *= values[indices[0]] as f32;
                for axis in 0..shape.len() {
                    indices[axis] += 1;
                    if indices[axis] < shape[axis] {
                        break;
                    }
                    indices[axis] = 0;
                }
            }
        }
    }
    let mut context = TensorContractFusionExecutionContext::<f32, RuleIdentity>::default();
    let reconstructed = crate::compose::compose(&mut context, &rule, &svd.u, &scaled_vh).unwrap();
    let distance = tensor_f32
        .data()
        .iter()
        .zip(reconstructed.data())
        .map(|(lhs, rhs)| ((lhs - rhs) as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(distance < 1e-3, "f32 reconstruction distance {distance}");

    // Complex32 general eigendecomposition returns Complex32 factors.
    let c32_space = space();
    let len = c32_space.required_len().unwrap();
    let tensor_c32 = TestTensor::<Complex32>::from_vec_with_fusion_space(
        (0..len)
            .map(|i| {
                Complex32::new(
                    ((i * 3 + 1) % 13) as f32 - 6.0,
                    ((i * 5 + 2) % 11) as f32 - 5.0,
                )
            })
            .collect(),
        c32_space,
    )
    .unwrap();
    let eig = eig_full(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &tensor_c32),
    )
    .unwrap();
    assert!(!eig.eigenvalues.is_empty());
    for entry in &eig.eigenvalues {
        for pair in entry.values.windows(2) {
            assert!((pair[0].re, pair[0].im) <= (pair[1].re, pair[1].im));
        }
    }
    let _: &TestTensor<Complex32> = &eig.v;
}
