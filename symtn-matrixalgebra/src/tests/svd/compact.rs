use super::*;

fn assert_compact_svd_direct_copy_probe(recording: &crate::probe::Recording) {
    let probe = recording.compact_svd_copy();
    assert_eq!(probe.input_pack_calls, 0);
    assert_eq!(probe.input_pack_bytes, 0);
    assert_eq!(probe.output_scatter_calls, 0);
    assert_eq!(probe.output_scatter_bytes, 0);
}

#[test]
fn compact_svd_canonical_layout_skips_input_pack_and_factor_scatter() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: canonical coupled storage reaches final factor destinations without numerical copies.
    let rule = Z2FusionRule;
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let compact_svd_copy_rec = crate::probe::record();
    svd_compact(
        engine,
        &mut dense,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap();

    assert_compact_svd_direct_copy_probe(&compact_svd_copy_rec);
}

#[test]
fn direct_compact_svd_uses_owned_executor_outputs_only() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let tensor = rectangular_svd_tensor(3, 2);
    let mut direct = ScriptedExecutor::<RejectSvdInto>::default();
    svd_compact(
        engine,
        &mut direct,
        &bound_tensor_ref!(Arc::new(Z2FusionRule), &tensor),
    )
    .unwrap();
    assert_eq!(direct.counts().svd, 1);
    assert_eq!(direct.counts().svd_into, 0);

    let (space, data) = generic_factorization_input();
    let (_, layout_space) = bind_checked_layout(&space);
    let input = BoundDynamicTensorRef::try_new(&layout_space, &data).unwrap();
    let mut generic = ScriptedExecutor::<RejectSvdInto>::default();
    svd_compact_factors_with_spectrum_dyn_checked_generic(engine, &mut generic, &input).unwrap();
    assert!(generic.counts().svd > 0);
    assert_eq!(generic.counts().svd_into, 0);

    let (_, checked_space) = bind_checked_only(&space);
    let checked_input = BoundDynamicTensorRef::try_new(&checked_space, &data).unwrap();
    let mut checked = ScriptedExecutor::<RejectSvdInto>::default();
    svd_compact_dyn_checked_generic(engine, &mut checked, &checked_input).unwrap();
    assert!(checked.counts().svd > 0);
    assert_eq!(checked.counts().svd_into, 0);

    let bound = bound_tensor(Arc::new(Z2FusionRule), &tensor);
    let mut polar = ScriptedExecutor::<RejectSvdInto>::default();
    left_polar(engine, &mut polar, &bound.as_ref()).unwrap();
    assert_eq!(polar.counts().svd, 1);
    assert_eq!(polar.counts().svd_into, 0);

    let fallback_space = bound.space().adjoint_view().unwrap();
    let fallback = BoundDynamicTensorRef::try_new(&fallback_space, bound.data()).unwrap();
    let mut adjoint_executor = ScriptedExecutor::<RejectSvdInto>::default();
    let compact_svd_copy_rec = crate::probe::record();
    svd_compact_dyn(engine, &mut adjoint_executor, &fallback).unwrap();
    assert!(adjoint_executor.counts().svd > 0);
    assert_eq!(adjoint_executor.counts().svd_into, 0);
    let probe = compact_svd_copy_rec.compact_svd_copy();
    assert!(probe.input_pack_calls > 0);
    assert!(probe.input_pack_bytes > 0);
    assert!(probe.output_scatter_calls > 0);
    assert!(probe.output_scatter_bytes > 0);
}

fn assert_mf_compact_svd_fallback_live_owners<D: crate::factorize::FactorScalar>() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let source = mixed_rectangular_c32_tensor();
    let tensor = TestTensor::<D>::from_vec_with_fusion_space(
        source
            .data()
            .iter()
            .map(|value| D::from_complex64(Complex64::new(value.re as f64, value.im as f64)))
            .collect(),
        source.fusion_space().clone(),
    )
    .unwrap();
    let bound = bound_tensor(Arc::new(Z2FusionRule), &tensor);
    let fallback_space = bound.space().adjoint_view().unwrap();
    let fallback = BoundDynamicTensorRef::try_new(&fallback_space, bound.data()).unwrap();
    let mut dense = ScriptedExecutor::<RejectSvdInto>::default();

    let compact_svd_copy_rec = crate::probe::record();
    let mf_svd_fallback_rec = crate::probe::record();
    svd_compact_dyn(engine, &mut dense, &fallback).unwrap();
    let stage = mf_svd_fallback_rec.mf_svd_fallbacks();

    assert_eq!(dense.counts().svd_into, 0);
    assert_eq!(dense.counts().svd, 2);
    assert_eq!(dense.output_ptrs, stage);
    assert_eq!(stage.len(), 2);
    assert!(stage.iter().all(|&(u, vt)| u != 0 && vt != 0 && u != vt));
    let probe = compact_svd_copy_rec.compact_svd_copy();
    assert!(probe.input_pack_calls > 0);
    assert!(probe.output_scatter_calls > 0);

    let mut adjoint_dense = ScriptedExecutor::<RejectSvdInto>::default();
    let compact_svd_copy_rec = crate::probe::record();
    let mf_svd_fallback_rec = crate::probe::record();
    svd_compact_adjoint_factors_dyn(engine, &mut adjoint_dense, &fallback).unwrap();
    let adjoint_stage = mf_svd_fallback_rec.mf_svd_fallbacks();
    assert_eq!(adjoint_dense.counts().svd_into, 0);
    assert_eq!(adjoint_dense.counts().svd, 2);
    assert_eq!(adjoint_dense.output_ptrs, adjoint_stage);
    assert_eq!(adjoint_stage.len(), 2);
    assert!(adjoint_stage
        .iter()
        .all(|&(u, vt)| u != 0 && vt != 0 && u != vt));
    let adjoint_probe = compact_svd_copy_rec.compact_svd_copy();
    assert!(adjoint_probe.input_pack_calls > 0);
    assert!(adjoint_probe.output_scatter_calls > 0);
}

#[test]
fn mf_compact_svd_fallback_keeps_live_owners_for_every_dtype() {
    assert_mf_compact_svd_fallback_live_owners::<f64>();
    assert_mf_compact_svd_fallback_live_owners::<f32>();
    assert_mf_compact_svd_fallback_live_owners::<Complex32>();
    assert_mf_compact_svd_fallback_live_owners::<Complex64>();
}

#[test]
fn generic_compact_svd_padded_fallback_uses_owned_outputs_and_scatter() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let (canonical_space, canonical_data) = generic_factorization_input();
    let (padded_space, padded_data) =
        padded_generic_factorization_input(&canonical_space, &canonical_data);
    let (_, padded_space) = bind_checked_layout(&padded_space);
    let padded = BoundDynamicTensorRef::try_new(&padded_space, &padded_data).unwrap();
    let mut dense = ScriptedExecutor::<RejectSvdInto>::default();

    let compact_svd_copy_rec = crate::probe::record();
    let generic_pair_rec = crate::probe::record();
    svd_compact_factors_with_spectrum_dyn_checked_generic(engine, &mut dense, &padded).unwrap();

    assert!(dense.counts().svd > 0);
    assert_eq!(dense.counts().svd_into, 0);
    let probe = compact_svd_copy_rec.compact_svd_copy();
    assert!(probe.input_pack_calls > 0);
    // Padding lives only in the source: the checked tier packs it away
    // and publishes the fresh factors canonically, without an output scatter.
    let publication = generic_pair_rec.generic_pair_publication();
    assert_eq!(
        (
            publication.canonical_publications,
            publication.fallback_publications
        ),
        (1, 0)
    );
    assert_eq!(
        (
            publication.left_scatter_calls,
            publication.right_scatter_calls
        ),
        (0, 0)
    );
}

fn assert_generic_compact_svd_fallback_live_owners<D: crate::factorize::FactorScalar>() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let (canonical_space, canonical_data) = generic_factorization_input();
    let (padded_space, padded_data) =
        padded_generic_factorization_input(&canonical_space, &canonical_data);
    let data = padded_data
        .into_iter()
        .map(D::from_real)
        .collect::<Vec<_>>();
    let (_, padded_space) = bind_checked_layout(&padded_space);
    let padded = BoundDynamicTensorRef::try_new(&padded_space, &data).unwrap();
    let mut dense = ScriptedExecutor::<RejectSvdInto>::default();

    let compact_svd_copy_rec = crate::probe::record();
    let generic_pair_rec = crate::probe::record();
    let svd_stage_rec = crate::probe::record();
    svd_compact_factors_with_spectrum_dyn_checked_generic(engine, &mut dense, &padded).unwrap();
    let stage = svd_stage_rec.checked_svd_stages();

    assert_eq!(dense.counts().svd_into, 0);
    assert_eq!(dense.output_ptrs, stage);
    assert_eq!(stage.len(), 2);
    assert!(stage.iter().all(|&(u, vt)| u != 0 && vt != 0 && u != vt));
    let probe = compact_svd_copy_rec.compact_svd_copy();
    assert!(probe.input_pack_calls > 0);
    // Padding lives only in the source: the checked tier packs it away
    // and publishes the fresh factors canonically, without an output scatter.
    let publication = generic_pair_rec.generic_pair_publication();
    assert_eq!(
        (
            publication.canonical_publications,
            publication.fallback_publications
        ),
        (1, 0)
    );
    assert_eq!(
        (
            publication.left_scatter_calls,
            publication.right_scatter_calls
        ),
        (0, 0)
    );
}

#[test]
fn generic_compact_svd_fallback_keeps_live_owners_for_every_dtype() {
    assert_generic_compact_svd_fallback_live_owners::<f64>();
    assert_generic_compact_svd_fallback_live_owners::<f32>();
    assert_generic_compact_svd_fallback_live_owners::<Complex32>();
    assert_generic_compact_svd_fallback_live_owners::<Complex64>();
}

#[test]
fn generic_compact_svd_interleaved_complex_fallback_preserves_source_order() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let (canonical_space, canonical_data, _) = generic_values_endomorphism_input();
    let (interleaved_space, interleaved_data) =
        interleaved_generic_endomorphism_input(&canonical_space, &canonical_data);
    let (_, canonical_space) = bind_checked_layout(&canonical_space);
    let (_, interleaved_space) = bind_checked_layout(&interleaved_space);
    let canonical = BoundDynamicTensorRef::try_new(&canonical_space, &canonical_data).unwrap();
    let interleaved =
        BoundDynamicTensorRef::try_new(&interleaved_space, &interleaved_data).unwrap();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let canonical_svd =
        svd_compact_factors_with_spectrum_dyn_checked_generic(engine, &mut dense, &canonical)
            .unwrap();
    let compact_svd_copy_rec = crate::probe::record();
    let generic_pair_rec = crate::probe::record();
    let fallback_svd =
        svd_compact_factors_with_spectrum_dyn_checked_generic(engine, &mut dense, &interleaved)
            .unwrap();

    let fallback_s = generic_diagonal_factor(&fallback_svd.0, &fallback_svd.2);
    assert_compact_factors_reconstruct_input(
        &interleaved,
        &fallback_svd.0,
        Some(&fallback_s),
        &fallback_svd.1,
    );
    assert_eq!(
        fallback_svd
            .2
            .iter()
            .map(|entry| entry.sector)
            .collect::<Vec<_>>(),
        [SectorId::new(1), SectorId::new(0)]
    );
    for actual in &fallback_svd.2 {
        let expected = canonical_svd
            .2
            .iter()
            .find(|entry| entry.sector == actual.sector)
            .unwrap();
        assert_eq!(actual.values.len(), expected.values.len());
        for (&actual, &expected) in actual.values.iter().zip(&expected.values) {
            assert!((actual - expected).abs() < 1.0e-10);
        }
    }
    let probe = compact_svd_copy_rec.compact_svd_copy();
    assert!(probe.input_pack_calls > 0);
    let publication = generic_pair_rec.generic_pair_publication();
    assert_eq!(
        (
            publication.canonical_publications,
            publication.fallback_publications
        ),
        (0, 1)
    );
    assert!(publication.left_scatter_calls > 0 && publication.right_scatter_calls > 0);
}

#[test]
fn generic_compact_svd_padded_complex_rectangular_fallback_matches_canonical_gauge() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let (canonical_space, canonical_data) = generic_svd_truncation_input::<Complex64>(true);
    let (padded_space, padded_data) =
        padded_generic_svd_truncation_input(&canonical_space, &canonical_data);
    let (_, canonical_space) = bind_checked_layout(&canonical_space);
    let (_, padded_space) = bind_checked_layout(&padded_space);
    let canonical = BoundDynamicTensorRef::try_new(&canonical_space, &canonical_data).unwrap();
    let padded = BoundDynamicTensorRef::try_new(&padded_space, &padded_data).unwrap();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let canonical_svd =
        svd_compact_factors_with_spectrum_dyn_checked_generic(engine, &mut dense, &canonical)
            .unwrap();
    let compact_svd_copy_rec = crate::probe::record();
    let generic_pair_rec = crate::probe::record();
    let mut reject = ScriptedExecutor::<RejectSvdInto>::default();
    let padded_svd =
        svd_compact_factors_with_spectrum_dyn_checked_generic(engine, &mut reject, &padded)
            .unwrap();

    assert_eq!(reject.counts().svd_into, 0);
    let padded_s = generic_diagonal_factor(&padded_svd.0, &padded_svd.2);
    assert_compact_factors_reconstruct_input(
        &padded,
        &padded_svd.0,
        Some(&padded_s),
        &padded_svd.1,
    );
    assert_generic_complex_factor_close(&padded_svd.0, &canonical_svd.0);
    assert_generic_complex_factor_close(&padded_svd.1, &canonical_svd.1);
    assert_real_spectra_close(&padded_svd.2, &canonical_svd.2);
    let probe = compact_svd_copy_rec.compact_svd_copy();
    assert!(probe.input_pack_calls > 0);
    // Padding lives only in the source: the checked tier packs it away
    // and publishes the fresh factors canonically, without an output scatter.
    let publication = generic_pair_rec.generic_pair_publication();
    assert_eq!(
        (
            publication.canonical_publications,
            publication.fallback_publications
        ),
        (1, 0)
    );
    assert_eq!(
        (
            publication.left_scatter_calls,
            publication.right_scatter_calls
        ),
        (0, 0)
    );
}

#[test]
fn generic_compact_svd_second_dense_failure_preserves_source() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let (source_space, source_data) = generic_svd_truncation_input::<f64>(false);
    let (space, data) = padded_generic_svd_truncation_input(&source_space, &source_data);
    let (_, space) = bind_checked_layout(&space);
    let input = BoundDynamicTensorRef::try_new(&space, &data).unwrap();
    let before = input.data().to_vec();
    let mut dense = ScriptedExecutor::<FailSecondSvd>::default();

    let compact_svd_copy_rec = crate::probe::record();
    let generic_pair_rec = crate::probe::record();
    match svd_compact_factors_with_spectrum_dyn_checked_generic(engine, &mut dense, &input) {
        Err(CheckedGenericFactorPlanError::Operation(OperationError::Dense(
            DenseError::Backend { op: "svd_into", .. },
        ))) => {}
        Err(error) => panic!("unexpected Generic SVD failure: {error:?}"),
        Ok(_) => panic!("second compact SVD must fail"),
    }
    assert_eq!(dense.counts().of(&[Op::Svd, Op::SvdInto]), 2);
    assert_eq!(input.data(), before);
    let probe = compact_svd_copy_rec.compact_svd_copy();
    assert!(probe.input_pack_calls > 0);
    // The checked tier stages every sector before publishing, so a dense
    // failure publishes no factor block at all.
    assert_eq!(
        generic_pair_rec.generic_pair_publication(),
        crate::probe::GenericPairPublicationProbe::default()
    );
}

#[test]
fn generic_compact_svd_empty_input_skips_dense_execution() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let x = SectorId::new(1);
    let empty_leg = SectorLeg::new([(x, 0)], false);
    let space = BoundDynamicFusionMapSpace::from_final_homspace_generic(
        Arc::new(FactorGenericRule),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([empty_leg.clone()]),
            FusionProductSpace::new([empty_leg]),
        ),
    )
    .unwrap();
    let (_, space) = bind_checked_layout(&space);
    let data: [f64; 0] = [];
    let input = BoundDynamicTensorRef::try_new(&space, &data).unwrap();
    let mut reject = ScriptedExecutor::new(RejectExecutorCalls);

    let (u, vh, singular_values) =
        svd_compact_factors_with_spectrum_dyn_checked_generic(engine, &mut reject, &input).unwrap();

    assert!(u.data().is_empty());
    assert!(vh.data().is_empty());
    assert!(singular_values.is_empty());
}

#[test]
fn compact_svd_adjoint_error_preserves_borrowed_input_and_publishes_no_factors() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: an SVD backend failure leaves the parent storage unchanged and
    // returns no partially constructed adjoint factors.
    let rule = Z2FusionRule;
    let tensor = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let before = tensor.data().to_vec();
    let bound = bound_tensor(Arc::new(rule), &tensor);
    let mut adjoint_dense = ScriptedExecutor::<FailAfterObservingSvdInput>::default();
    let result =
        svd_compact_adjoint_factors_dyn(engine, &mut adjoint_dense, &bound.as_ref().dynamic());
    assert!(matches!(result, Err(OperationError::Dense(_))));
    assert_eq!(tensor.data(), before);
    assert!(!adjoint_dense.observed.is_empty());
}

#[test]
fn compact_svd_adjoint_late_error_preserves_borrowed_input_and_publishes_no_factors() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rule = Z2FusionRule;
    let canonical = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let tensor = padded_copy(&rule, &canonical);
    let before = tensor.data().to_vec();
    let bound = bound_tensor(Arc::new(rule), &tensor);
    let mut dense = ScriptedExecutor::<FailSecondSvd>::default();
    let compact_svd_copy_rec = crate::probe::record();

    let result = svd_compact_adjoint_factors_dyn(engine, &mut dense, &bound.as_ref().dynamic());

    assert!(matches!(result, Err(OperationError::Dense(_))));
    assert_eq!(tensor.data(), before);
    assert_eq!(dense.counts().of(&[Op::Svd, Op::SvdInto]), 2);
    let probe = compact_svd_copy_rec.compact_svd_copy();
    assert!(probe.input_pack_calls > 0);
    assert!(probe.output_scatter_calls > 0);
}

fn f64_svd_outputs(rows: usize, cols: usize) -> Vec<DenseTensor> {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let data = vec![1.0; rows * cols];
    dense
        .svd(
            engine,
            DenseRead::F64(
                symtn_dense::DenseView::new(&data, &[rows, cols], &[1, rows], 0).unwrap(),
            ),
        )
        .unwrap()
}

fn c64_svd_outputs(rows: usize, cols: usize) -> Vec<DenseTensor> {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let data = vec![Complex64::new(1.0, 1.0); rows * cols];
    dense
        .svd(
            engine,
            DenseRead::C64(
                symtn_dense::DenseView::new(&data, &[rows, cols], &[1, rows], 0).unwrap(),
            ),
        )
        .unwrap()
}

#[test]
fn compact_owned_svd_preserves_svd_into_output_precedence() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let tensor = rectangular_svd_tensor(2, 2);
    let bound = bound_tensor(Arc::new(Z2FusionRule), &tensor);
    let input = bound.as_ref();
    let mut check = |outputs: Vec<DenseTensor>, expected: DenseError| {
        let mut dense = ScriptedExecutor::new(FailAfterObservingSvdInput {
            outputs: Some(outputs),
            ..Default::default()
        });
        let error = svd_compact(engine, &mut dense, &input).unwrap_err();
        assert_eq!(error, OperationError::Dense(expected));
    };

    check(
        vec![f64_svd_outputs(2, 2).remove(0)],
        arity_mismatch("svd_into", 3, 1),
    );
    check(
        f64_svd_outputs(1, 1),
        DenseError::ShapeMismatch {
            op: "svd_into",
            expected: vec![2, 2],
            actual: vec![1, 1],
        },
    );
    let mut outputs = f64_svd_outputs(2, 2);
    outputs[1] = f64_svd_outputs(1, 1).remove(1);
    check(
        outputs,
        DenseError::ShapeMismatch {
            op: "svd_into",
            expected: vec![2],
            actual: vec![1],
        },
    );
    let mut outputs = f64_svd_outputs(2, 2);
    outputs[2] = f64_svd_outputs(1, 1).remove(2);
    check(
        outputs,
        DenseError::ShapeMismatch {
            op: "svd_into",
            expected: vec![2, 2],
            actual: vec![1, 1],
        },
    );
    let outputs = c64_svd_outputs(2, 2);
    let expected = outputs[0].as_f64_slice().unwrap_err();
    assert!(matches!(
        expected,
        DenseError::DTypeMismatch {
            actual: symtn_dense::DenseDType::C64,
            ..
        }
    ));
    let mut dense = ScriptedExecutor::new(FailAfterObservingSvdInput {
        outputs: Some(outputs),
        ..Default::default()
    });
    let error = svd_compact(engine, &mut dense, &input).unwrap_err();
    assert!(matches!(error, OperationError::Dense(actual) if actual == expected));

    let mut outputs = f64_svd_outputs(2, 2);
    outputs[1] = c64_svd_outputs(2, 2).remove(0);
    let expected = outputs[1].as_f64_slice().unwrap_err();
    assert!(matches!(
        expected,
        DenseError::DTypeMismatch {
            actual: symtn_dense::DenseDType::C64,
            ..
        }
    ));
    let mut dense = ScriptedExecutor::new(FailAfterObservingSvdInput {
        outputs: Some(outputs),
        ..Default::default()
    });
    let error = svd_compact(engine, &mut dense, &input).unwrap_err();
    assert!(matches!(error, OperationError::Dense(actual) if actual == expected));

    let mut outputs = f64_svd_outputs(2, 2);
    outputs[2] = c64_svd_outputs(2, 2).remove(0);
    let expected = outputs[2].as_f64_slice().unwrap_err();
    assert!(matches!(
        expected,
        DenseError::DTypeMismatch {
            actual: symtn_dense::DenseDType::C64,
            ..
        }
    ));
    let mut dense = ScriptedExecutor::new(FailAfterObservingSvdInput {
        outputs: Some(outputs),
        ..Default::default()
    });
    let error = svd_compact(engine, &mut dense, &input).unwrap_err();
    assert!(matches!(error, OperationError::Dense(actual) if actual == expected));
}

#[test]
fn compact_svd_error_preserves_borrowed_input_and_publishes_no_factors() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: a provider failure cannot mutate borrowed tensor storage or return partial factors.
    let rule = Z2FusionRule;
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let before = tensor.data().to_vec();
    let mut dense = ScriptedExecutor::<FailAfterObservingSvdInput>::default();

    let diagonal_bond_build_rec = crate::probe::record();
    let result = svd_compact(
        engine,
        &mut dense,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    );

    assert!(matches!(result, Err(OperationError::Dense(_))));
    assert_eq!(tensor.data(), before);
    assert!(!dense.observed.is_empty());
    assert!(dense
        .observed
        .iter()
        .all(|sector| before.windows(sector.len()).any(|window| window == sector)));
    assert_eq!(
        diagonal_bond_build_rec.diagonal_bond_build(),
        crate::probe::DiagonalBondBuildProbe::default()
    );
}

fn assert_rectangular_direct_svd(rows: usize, cols: usize) {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rule = Z2FusionRule;
    let tensor = rectangular_svd_tensor(rows, cols);
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let bound = bound_tensor(Arc::new(rule), &tensor);
    let compact_svd_copy_rec = crate::probe::record();
    let svd = svd_compact(engine, &mut dense, &bound.as_ref()).unwrap();
    assert_factor_layout_matches_per_tree_shapes(svd.u.space());
    assert_factor_layout_matches_per_tree_shapes(svd.s.space());
    assert_factor_layout_matches_per_tree_shapes(svd.vh.space());
    assert_compact_svd_direct_copy_probe(&compact_svd_copy_rec);
    let rank = rows.min(cols);
    if rank == 0 {
        assert!(svd.u.space().space().homspace().domain().legs()[0]
            .sectors()
            .is_empty());
        assert!(svd.vh.space().space().homspace().codomain().legs()[0]
            .sectors()
            .is_empty());
    }
    let singular = svd
        .singular_values
        .first()
        .map(|entry| entry.values.as_slice())
        .unwrap_or_default();
    assert_eq!(singular.len(), rank);
    for col in 0..cols {
        for row in 0..rows {
            let reconstructed = (0..rank)
                .map(|bond| {
                    svd.u.data()[row + rows * bond]
                        * singular[bond]
                        * svd.vh.data()[bond + rank * col]
                })
                .sum::<f64>();
            assert!((reconstructed - tensor.data()[row + rows * col]).abs() < 1e-10);
        }
    }

    let compact_svd_copy_rec = crate::probe::record();
    let adjoint =
        svd_compact_adjoint_factors_dyn(engine, &mut dense, &bound.as_ref().dynamic()).unwrap();
    let adjoint_probe = compact_svd_copy_rec.compact_svd_copy();
    assert_eq!(adjoint_probe.input_pack_calls, 0);
    assert_eq!(adjoint_probe.output_scatter_calls, 0);
    if rank != 0 {
        assert_eq!(adjoint_probe.owned_output_owner_reused, 2);
    }
    let adjoint_singular = adjoint
        .2
        .first()
        .map(|entry| entry.values.as_slice())
        .unwrap_or_default();
    assert_eq!(adjoint_singular.len(), rank);
    for col in 0..rows {
        for row in 0..cols {
            let reconstructed = (0..rank)
                .map(|bond| {
                    adjoint.0.data()[row + cols * bond]
                        * adjoint_singular[bond]
                        * adjoint.1.data()[bond + rank * col]
                })
                .sum::<f64>();
            assert!((reconstructed - tensor.data()[col + rows * row]).abs() < 1e-10);
        }
    }
}

#[test]
fn compact_svd_direct_spans_reconstruct_tall_and_wide_matrices() {
    // What: exact final-factor spans work for both compact rectangular shapes.
    assert_rectangular_direct_svd(5, 3);
    assert_rectangular_direct_svd(3, 5);
}

#[test]
fn compact_svd_direct_outputs_keep_executor_factor_owners() {
    fn check<D: crate::factorize::FactorScalar>(tensor: &TestTensor<D>) {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let mut dense = ScriptedExecutor::<RejectSvdInto>::default();
        let bound = bound_tensor(Arc::new(Z2FusionRule), tensor);
        let compact_svd_copy_rec = crate::probe::record();
        let svd = svd_compact(engine, &mut dense, &bound.as_ref()).unwrap();
        assert_eq!(dense.output_ptrs.len(), 1);
        let (u, vh) = dense.output_ptrs[0];
        assert_eq!(u, svd.u.data().as_ptr() as usize);
        assert_eq!(vh, svd.vh.data().as_ptr() as usize);
        let probe = compact_svd_copy_rec.compact_svd_copy();
        assert_compact_svd_direct_copy_probe(&compact_svd_copy_rec);
        assert_eq!(probe.owned_output_publications, 2);
        assert_eq!(probe.owned_output_owner_reused, 2);
    }

    let tensor = rectangular_svd_tensor(3, 2);
    check(&tensor);
    let space = tensor.fusion_space().clone();
    check(
        &TestTensor::<f32>::from_vec_with_fusion_space(
            tensor.data().iter().map(|&value| value as f32).collect(),
            space.clone(),
        )
        .unwrap(),
    );
    check(
        &TestTensor::<Complex32>::from_vec_with_fusion_space(
            tensor
                .data()
                .iter()
                .map(|&value| Complex32::new(value as f32, value as f32 * 0.25))
                .collect(),
            space.clone(),
        )
        .unwrap(),
    );
    check(
        &TestTensor::<Complex64>::from_vec_with_fusion_space(
            tensor
                .data()
                .iter()
                .map(|&value| Complex64::new(value, value * 0.25))
                .collect(),
            space,
        )
        .unwrap(),
    );
}

#[test]
fn compact_svd_zero_only_input_normalizes_to_an_empty_factorization_result() {
    // What: a zero-only row or column produces empty factors and no phantom
    // spectrum entry or factor route.
    assert_rectangular_direct_svd(0, 3);
    assert_rectangular_direct_svd(3, 0);
    assert_zero_axis_svd_compact(0, 3);
    assert_zero_axis_svd_compact(3, 0);
}

#[test]
fn compact_svd_direct_and_fallback_apply_the_same_gauge() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: direct writes do not change the canonical phase chosen by the fallback.
    let rule = Z2FusionRule;
    let tensor = rectangular_svd_tensor(3, 3);
    let mut transposed_data = vec![0.0; 9];
    for col in 0..3 {
        for row in 0..3 {
            transposed_data[row + 3 * col] = tensor.data()[col + 3 * row];
        }
    }
    let transposed =
        TestTensor::from_vec_with_fusion_space(transposed_data, tensor.fusion_space().clone())
            .unwrap();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let direct = svd_compact(
        engine,
        &mut dense,
        &bound_tensor_ref!(Arc::new(rule), &transposed),
    )
    .unwrap();
    let bound = bound_tensor(Arc::new(rule), &tensor);
    let adjoint_space = bound.space().adjoint_view().unwrap();
    let fallback_input = BoundDynamicTensorRef::try_new(&adjoint_space, bound.data()).unwrap();
    let fallback = svd_compact_dyn(engine, &mut dense, &fallback_input).unwrap();

    for (left, right) in direct.u.data().iter().zip(fallback.u().data()) {
        assert!((left - right).abs() < 1e-12);
    }
    for (left, right) in direct.vh.data().iter().zip(fallback.vh().data()) {
        assert!((left - right).abs() < 1e-12);
    }
    assert_spectra_agree(
        "direct vs adjoint-view fallback",
        &direct.singular_values,
        fallback.singular_values(),
    );

    let adjoint_fallback =
        svd_compact_adjoint_factors_dyn(engine, &mut dense, &fallback_input).unwrap();
    let expected = svd_compact_factors_dyn(engine, &mut dense, &bound.as_ref().dynamic()).unwrap();
    for (actual, expected) in adjoint_fallback.0.data().iter().zip(expected.0.data()) {
        assert!((actual - expected).abs() < 1e-12);
    }
    for (actual, expected) in adjoint_fallback.1.data().iter().zip(expected.1.data()) {
        assert!((actual - expected).abs() < 1e-12);
    }
    for (actual, expected) in adjoint_fallback.2.iter().zip(&expected.2) {
        assert_eq!(actual.sector, expected.sector);
        for (actual, expected) in actual.values.iter().zip(&expected.values) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }
}

#[test]
fn compact_svd_adjoint_accepts_padded_parent_layout() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: the optimized adjoint path uses the existing packed fallback for
    // custom offsets instead of materializing a canonical adjoint input.
    let rule = Z2FusionRule;
    let parent = padded_copy(&rule, &rectangular_svd_tensor(5, 3));
    let bound = bound_tensor(Arc::new(rule), &parent);
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let actual =
        svd_compact_adjoint_factors_dyn(engine, &mut dense, &bound.as_ref().dynamic()).unwrap();
    let singular = &actual.2[0].values;
    let source = parent.structure().block(0).unwrap();
    for col in 0..5 {
        for row in 0..3 {
            let reconstructed = (0..3)
                .map(|bond| {
                    actual.0.data()[row + 3 * bond]
                        * singular[bond]
                        * actual.1.data()[bond + 3 * col]
                })
                .sum::<f64>();
            let expected = parent.data()
                [source.offset() + col * source.strides()[0] + row * source.strides()[1]];
            assert!((reconstructed - expected).abs() < 1e-10);
        }
    }
}

#[test]
fn compact_svd_adjoint_c64_padded_parent_reconstructs_literal_adjoint() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rule = Z2FusionRule;
    let source = rectangular_svd_tensor(5, 3);
    let complex = TestTensor::<Complex64>::from_vec_with_fusion_space(
        source
            .data()
            .iter()
            .enumerate()
            .map(|(index, &value)| Complex64::new(value, index as f64 * 0.125 - 0.5))
            .collect(),
        source.fusion_space().clone(),
    )
    .unwrap();
    let parent = padded_copy(&rule, &complex);
    let bound = bound_tensor(Arc::new(rule), &parent);
    assert!(
        crate::factorize::compact_factor_plan_for_test(bound.space())
            .unwrap()
            .is_none()
    );
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let compact_svd_copy_rec = crate::probe::record();
    let actual =
        svd_compact_adjoint_factors_dyn(engine, &mut dense, &bound.as_ref().dynamic()).unwrap();
    let singular = &actual.2[0].values;
    let source = parent.structure().block(0).unwrap();
    for col in 0..5 {
        for row in 0..3 {
            let reconstructed = (0..3)
                .map(|bond| {
                    actual.0.data()[row + 3 * bond]
                        * singular[bond]
                        * actual.1.data()[bond + 3 * col]
                })
                .sum::<Complex64>();
            let expected = parent.data()
                [source.offset() + col * source.strides()[0] + row * source.strides()[1]]
                .conj();
            assert!((reconstructed - expected).norm() < 1e-10);
        }
    }
    let probe = compact_svd_copy_rec.compact_svd_copy();
    assert!(probe.input_pack_calls > 0);
    assert!(probe.output_scatter_calls > 0);
}

#[test]
fn compact_svd_c64_reconstructs_mixed_tall_and_wide_sectors_without_copies() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    use num_complex::Complex64;

    // What: one call reconstructs mixed rectangular complex sectors directly in final storage.
    let rule = Z2FusionRule;
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(even, 5), (odd, 2)], false)]),
        FusionProductSpace::new([SectorLeg::new([(even, 3), (odd, 4)], false)]),
    );
    let space = DynamicFusionMapSpace::from_final_homspace(&rule, homspace).unwrap();
    let tensor = TestTensor::<Complex64>::from_vec_with_fusion_space(
        (0..space.required_len().unwrap())
            .map(|index| {
                Complex64::new(
                    ((index * 7 + 2) % 17) as f64 - 6.0,
                    ((index * 5 + 3) % 13) as f64 * 0.25 - 1.0,
                )
            })
            .collect(),
        space,
    )
    .unwrap();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let compact_svd_copy_rec = crate::probe::record();

    let svd = svd_compact(
        engine,
        &mut dense,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap();

    assert_compact_svd_direct_copy_probe(&compact_svd_copy_rec);
    let input_regions = tensor
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    let u_regions = svd
        .u
        .tensor()
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    let vh_regions = svd
        .vh
        .tensor()
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    for input_region in input_regions.iter() {
        let sector = input_region.coupled();
        let u_region = u_regions
            .iter()
            .find(|region| region.coupled() == sector)
            .unwrap();
        let vh_region = vh_regions
            .iter()
            .find(|region| region.coupled() == sector)
            .unwrap();
        let singular = &svd
            .singular_values
            .iter()
            .find(|values| values.sector == sector)
            .unwrap()
            .values;
        let rows = input_region.rows();
        let cols = input_region.cols();
        let rank = rows.min(cols);
        for col in 0..cols {
            for row in 0..rows {
                let reconstructed = (0..rank)
                    .map(|bond| {
                        svd.u.data()[u_region.range().start + row + rows * bond]
                            * singular[bond]
                            * svd.vh.data()[vh_region.range().start + bond + rank * col]
                    })
                    .sum::<Complex64>();
                let expected = tensor.data()[input_region.range().start + row + rows * col];
                assert!((reconstructed - expected).norm() < 1e-10);
            }
        }
    }
}

#[test]
fn compact_svd_adjoint_c64_padded_fallback_matches_canonical_gauge() {
    let engine = &mut symtn_dense::CpuEngine::serial();
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
    let padded = padded_copy(&rule, &canonical);
    let canonical_bound = bound_tensor(Arc::new(rule), &canonical);
    let padded_bound = bound_tensor(Arc::new(rule), &padded);
    assert!(
        crate::factorize::compact_factor_plan_for_test(canonical_bound.space())
            .unwrap()
            .is_some()
    );
    assert!(
        crate::factorize::compact_factor_plan_for_test(padded_bound.space())
            .unwrap()
            .is_none()
    );
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let expected =
        svd_compact_adjoint_factors_dyn(engine, &mut dense, &canonical_bound.as_ref().dynamic())
            .unwrap();
    let compact_svd_copy_rec = crate::probe::record();
    let actual =
        svd_compact_adjoint_factors_dyn(engine, &mut dense, &padded_bound.as_ref().dynamic())
            .unwrap();

    assert_eq!(
        actual.0.space().space().structure(),
        expected.0.space().space().structure()
    );
    assert_eq!(
        actual.1.space().space().structure(),
        expected.1.space().space().structure()
    );
    for (&actual, &expected) in actual.0.data().iter().zip(expected.0.data()) {
        assert!((actual - expected).norm() < 1.0e-10);
    }
    for (&actual, &expected) in actual.1.data().iter().zip(expected.1.data()) {
        assert!((actual - expected).norm() < 1.0e-10);
    }
    assert_real_spectra_close(&actual.2, &expected.2);
    let probe = compact_svd_copy_rec.compact_svd_copy();
    assert!(probe.input_pack_calls > 0);
    assert!(probe.output_scatter_calls > 0);
}

#[test]
fn compact_svd_c32_reconstructs_mixed_tall_and_wide_sectors_without_copies() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: single-precision complex direct spans reconstruct both rectangular orientations.
    let rule = Z2FusionRule;
    let tensor = mixed_rectangular_c32_tensor();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let compact_svd_copy_rec = crate::probe::record();

    let svd = svd_compact(
        engine,
        &mut dense,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap();

    assert_compact_svd_direct_copy_probe(&compact_svd_copy_rec);
    let input_regions = tensor
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    let u_regions = svd
        .u
        .tensor()
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    let vh_regions = svd
        .vh
        .tensor()
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    for input_region in input_regions.iter() {
        let sector = input_region.coupled();
        let u_region = u_regions
            .iter()
            .find(|region| region.coupled() == sector)
            .unwrap();
        let vh_region = vh_regions
            .iter()
            .find(|region| region.coupled() == sector)
            .unwrap();
        let singular = &svd
            .singular_values
            .iter()
            .find(|values| values.sector == sector)
            .unwrap()
            .values;
        let rows = input_region.rows();
        let cols = input_region.cols();
        let rank = rows.min(cols);
        for col in 0..cols {
            for row in 0..rows {
                let reconstructed = (0..rank)
                    .map(|bond| {
                        svd.u.data()[u_region.range().start + row + rows * bond]
                            * singular[bond] as f32
                            * svd.vh.data()[vh_region.range().start + bond + rank * col]
                    })
                    .sum::<Complex32>();
                let expected = tensor.data()[input_region.range().start + row + rows * col];
                assert!((reconstructed - expected).norm() < 2e-4);
            }
        }
    }
}

#[test]
fn compact_svd_c32_direct_and_fallback_apply_the_same_gauge() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: the single-precision direct path preserves the fallback's canonical complex phase.
    let rule = Z2FusionRule;
    let real = rectangular_svd_tensor(3, 3);
    let data = real
        .data()
        .iter()
        .enumerate()
        .map(|(index, &value)| Complex32::new(value as f32, (index as f32 - 3.0) * 0.25))
        .collect::<Vec<_>>();
    let tensor = TestTensor::from_vec_with_fusion_space(data, real.fusion_space().clone()).unwrap();
    let mut transposed_data = vec![Complex32::new(0.0, 0.0); 9];
    for col in 0..3 {
        for row in 0..3 {
            transposed_data[row + 3 * col] = tensor.data()[col + 3 * row];
        }
    }
    let transposed =
        TestTensor::from_vec_with_fusion_space(transposed_data, tensor.fusion_space().clone())
            .unwrap();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let compact_svd_copy_rec = crate::probe::record();
    let direct = svd_compact(
        engine,
        &mut dense,
        &bound_tensor_ref!(Arc::new(rule), &transposed),
    )
    .unwrap();
    assert_compact_svd_direct_copy_probe(&compact_svd_copy_rec);
    let bound = bound_tensor(Arc::new(rule), &tensor);
    let adjoint_space = bound.space().adjoint_view().unwrap();
    let fallback_input = BoundDynamicTensorRef::try_new(&adjoint_space, bound.data()).unwrap();
    let compact_svd_copy_rec = crate::probe::record();
    let fallback = svd_compact_dyn(engine, &mut dense, &fallback_input).unwrap();
    let fallback_probe = compact_svd_copy_rec.compact_svd_copy();
    assert!(fallback_probe.input_pack_calls > 0);
    assert!(fallback_probe.output_scatter_calls > 0);
    for entry in &direct.singular_values {
        assert!(entry.values.last().is_some_and(|value| *value > 1e-3));
        assert!(entry
            .values
            .windows(2)
            .all(|pair| (pair[0] - pair[1]).abs() > 1e-3));
    }

    assert_eq!(direct.u.data().len(), fallback.u().data().len());
    for (left, right) in direct.u.data().iter().zip(fallback.u().data()) {
        assert!((*left - *right).norm() < 2e-5);
    }
    assert_eq!(direct.vh.data().len(), fallback.vh().data().len());
    for (left, right) in direct.vh.data().iter().zip(fallback.vh().data()) {
        assert!((*left - *right).norm() < 2e-5);
    }
    assert_eq!(
        direct.singular_values.len(),
        fallback.singular_values().len()
    );
    for (left_entry, right_entry) in direct
        .singular_values
        .iter()
        .zip(fallback.singular_values())
    {
        assert_eq!(left_entry.sector, right_entry.sector);
        assert_eq!(left_entry.values.len(), right_entry.values.len());
        for (left, right) in left_entry.values.iter().zip(&right_entry.values) {
            assert!((left - right).abs() < 1e-5);
        }
    }
}

#[test]
fn svd_compact_factor_dims_include_sectors_without_populated_trees() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: public compact SVD retains the complete original leg space,
    // including a sector absent from every populated fusion block.
    let rule = U1FusionRule;
    let neutral = U1Irrep::new(0).sector_id();
    let positive = U1Irrep::new(1).sector_id();
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(neutral, 2), (positive, 3)], false)]),
        FusionProductSpace::new([SectorLeg::new([(neutral, 2)], false)]),
    );
    let space = DynamicFusionMapSpace::from_final_homspace(&rule, homspace).unwrap();
    let tensor = TestTensor::from_vec_with_fusion_space(vec![1.0, 2.0, 3.0, 4.0], space).unwrap();
    let original_homspace = tensor.fusion_space().homspace().clone();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let result = svd_compact(
        engine,
        &mut dense,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap();

    assert_factor_layout_matches_per_tree_shapes(result.u.space());
    assert_factor_layout_matches_per_tree_shapes(result.s.space());
    assert_factor_layout_matches_per_tree_shapes(result.vh.space());
    assert_eq!(result.u.tensor().dims(), &[5, 2]);
    assert_eq!(result.vh.tensor().dims(), &[2, 2]);
    assert_eq!(
        result.u.tensor().fusion_space().homspace().codomain(),
        original_homspace.codomain()
    );
    assert_eq!(
        result.vh.tensor().fusion_space().homspace().domain(),
        original_homspace.domain()
    );
}

#[test]
fn svd_compact_preserves_asymmetric_non_self_dual_u1_factor_layouts() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: canonical factors retain unequal degeneracies and the dual U(1)
    // domain convention while reconstructing both coupled sectors.
    let rule = U1FusionRule;
    let neutral = U1Irrep::new(0).sector_id();
    let positive = U1Irrep::new(1).sector_id();
    let negative = U1Irrep::new(-1).sector_id();
    let codomain = SectorLeg::new([(neutral, 3), (positive, 2)], false);
    let domain = SectorLeg::new([(neutral, 1), (negative, 4)], true);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([codomain]),
        FusionProductSpace::new([domain]),
    );
    let space = DynamicFusionMapSpace::from_final_homspace(&rule, homspace).unwrap();
    let tensor = TestTensor::<f64>::from_vec_with_fusion_space(
        (0..space.required_len().unwrap())
            .map(|index| ((index * 7 + 3) % 17) as f64 - 6.0)
            .collect(),
        space,
    )
    .unwrap();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let svd = svd_compact(
        engine,
        &mut dense,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap();

    assert_factor_layout_matches_per_tree_shapes(svd.u.space());
    assert_factor_layout_matches_per_tree_shapes(svd.s.space());
    assert_factor_layout_matches_per_tree_shapes(svd.vh.space());
    let input_regions = tensor
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    let u_regions = svd
        .u
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    let vh_regions = svd
        .vh
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    for input_region in input_regions.iter() {
        let sector = input_region.coupled();
        let u_region = u_regions
            .iter()
            .find(|region| region.coupled() == sector)
            .unwrap();
        let vh_region = vh_regions
            .iter()
            .find(|region| region.coupled() == sector)
            .unwrap();
        let singular = &svd
            .singular_values
            .iter()
            .find(|entry| entry.sector == sector)
            .unwrap()
            .values;
        for col in 0..input_region.cols() {
            for row in 0..input_region.rows() {
                let reconstructed = (0..singular.len())
                    .map(|bond| {
                        svd.u.data()[u_region.range().start + row + input_region.rows() * bond]
                            * singular[bond]
                            * svd.vh.data()[vh_region.range().start + bond + singular.len() * col]
                    })
                    .sum::<f64>();
                let expected =
                    tensor.data()[input_region.range().start + row + input_region.rows() * col];
                assert!((reconstructed - expected).abs() < 1.0e-10);
            }
        }
    }
}

#[test]
fn svd_rejects_a_different_provider_before_dense_execution() {
    let tensor = hermitian_test_tensor(&Z2FusionRule, &[SectorId::new(0), SectorId::new(1)]);

    let _backend = ScriptedExecutor::new(RejectExecutorCalls);
    let error = match BoundTensorMap::try_new(Arc::new(U1FusionRule), tensor) {
        Ok(_) => panic!("mismatched provider must not produce an authority"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        OperationError::Core(CoreError::FusionRuleMismatch { .. })
    ));
}

#[test]
fn public_svd_authority_rejects_same_type_with_different_identity_and_qdim() {
    // What: provider provenance, not the Rust type or sector ids, owns qdim.
    let source_rule = IdentityQdimRule::new(1.0);
    let other_rule = IdentityQdimRule::new((1.0 + 5.0_f64.sqrt()) / 2.0);
    let tensor = tsvd_test_tensor(&source_rule, &[SectorId::new(0)]);

    let _backend = ScriptedExecutor::new(RejectExecutorCalls);
    let error = match BoundTensorMap::try_new(Arc::new(other_rule), tensor) {
        Ok(_) => panic!("different provider identity must not produce an authority"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        OperationError::Core(CoreError::FusionRuleMismatch { .. })
    ));
}

#[test]
fn svd_input_rejects_short_storage_before_dense_execution() {
    let tensor = tsvd_test_tensor(&Z2FusionRule, &[SectorId::new(0), SectorId::new(1)]);
    let bound = BoundDynamicFusionMapSpace::bind_multiplicity_free(
        tensor.fusion_space().clone(),
        Arc::new(Z2FusionRule),
    )
    .unwrap();
    let short = &tensor.data()[..tensor.data().len() - 1];

    let error = match BoundDynamicTensorRef::try_new(&bound, short) {
        Ok(_) => panic!("short storage must be rejected"),
        Err(error) => error,
    };

    assert!(matches!(
        error,
        OperationError::Core(CoreError::DimensionMismatch { .. })
    ));
}

#[test]
fn typed_svd_borrows_input_authority_and_retains_its_exact_allocation() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: borrowed typed input creates no replacement authority, and every SVD factor inherits it.
    let rule = Z2FusionRule;
    let provider = Arc::new(rule);
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let input = bound_tensor(Arc::clone(&provider), &tensor);
    let first = input.as_ref();
    let second = input.as_ref();

    assert!(std::ptr::eq(first.space(), input.space()));
    assert!(std::ptr::eq(second.space(), input.space()));
    assert!(std::ptr::eq(first.tensor(), input.tensor()));
    assert!(std::ptr::eq(second.tensor(), input.tensor()));

    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let factors = svd_compact(engine, &mut dense, &first).unwrap();
    assert!(Arc::ptr_eq(&provider, factors.u.space().provider_arc()));
    assert!(Arc::ptr_eq(&provider, factors.s.space().provider_arc()));
    assert!(Arc::ptr_eq(&provider, factors.vh.space().provider_arc()));
}

fn assert_zero_axis_svd_compact(rows: usize, cols: usize) {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rule = Z2FusionRule;
    let tensor = rectangular_svd_tensor(rows, cols);
    let input = bound_tensor(Arc::new(rule), &tensor);
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    {
        let diagonal_bond_build_rec = crate::probe::record();
        let result = svd_compact_dyn(engine, &mut dense, &input.as_ref().dynamic()).unwrap();
        assert_eq!(
            result
                .singular_values()
                .iter()
                .map(|entry| entry.values.len())
                .sum::<usize>(),
            0
        );
        assert!(result.u().data().is_empty());
        assert!(result.s().data().is_empty());
        assert!(result.vh().data().is_empty());
        for factor in [result.u(), result.s(), result.vh()] {
            assert_eq!(factor.space().space().structure().block_count(), 0);
            assert_factor_layout_matches_per_tree_shapes(factor.space());
        }
        assert_eq!(
            diagonal_bond_build_rec.diagonal_bond_build(),
            crate::probe::DiagonalBondBuildProbe {
                calls: 1,
                values: 0,
            }
        );
    }
}

#[test]
fn svd_compact_gauge_matches_matrixalgebrakit_phase_rule() {
    use num_complex::Complex64;
    let c = Complex64::new;
    let mut u = vec![
        c(3.0, 4.0),
        c(1.0, -1.0),
        c(-2.0, 0.5),
        c(0.25, -0.5),
        c(-4.0, 0.0),
        c(1.0, 2.0),
    ];
    let mut vh = vec![
        c(0.5, -1.0),
        c(-0.25, 0.75),
        c(1.0, 0.0),
        c(0.0, -2.0),
        c(-1.5, 0.25),
        c(0.75, -0.5),
    ];
    let sigma = [2.0, 0.75];
    let product = |u: &[Complex64], vh: &[Complex64]| -> Vec<Complex64> {
        let mut out = vec![c(0.0, 0.0); 9];
        for col in 0..3 {
            for row in 0..3 {
                for k in 0..2 {
                    out[row + 3 * col] += u[row + 3 * k] * sigma[k] * vh[k + 2 * col];
                }
            }
        }
        out
    };
    let before = product(&u, &vh);
    crate::factorize::svd_compact_gauge(&mut u, 3, 3, &mut vh, 2, 3, 2);
    for &(row, col) in &[(0, 0), (1, 1)] {
        let pivot = u[row + 3 * col];
        assert!(pivot.im.abs() < 1e-14, "pivot {pivot} not real");
        assert!(pivot.re >= 0.0, "pivot {pivot} negative");
    }
    let after = product(&u, &vh);
    for (lhs, rhs) in after.iter().zip(&before) {
        assert!(
            (lhs - rhs).norm() < 1e-13,
            "product changed: {lhs} vs {rhs}"
        );
    }
}

#[test]
fn svd_compact_adjoint_gauge_fixes_final_left_factor() {
    use num_complex::Complex64;
    let c = Complex64::new;
    let mut u = vec![
        c(3.0, 4.0),
        c(1.0, -1.0),
        c(-2.0, 0.5),
        c(0.25, -0.5),
        c(-4.0, 0.0),
        c(1.0, 2.0),
    ];
    let mut vh = vec![
        c(0.5, -1.0),
        c(-0.25, 0.75),
        c(1.0, 0.0),
        c(0.0, -2.0),
        c(-0.5, 1.0),
        c(0.75, -0.5),
    ];
    let sigma = [2.0, 0.75];
    let product = |u: &[Complex64], vh: &[Complex64]| -> Vec<Complex64> {
        let mut out = vec![c(0.0, 0.0); 9];
        for col in 0..3 {
            for row in 0..3 {
                for k in 0..2 {
                    out[row + 3 * col] += u[row + 3 * k] * sigma[k] * vh[k + 2 * col];
                }
            }
        }
        out
    };
    let before = product(&u, &vh);
    crate::factorize::svd_compact_adjoint_gauge(&mut u, 3, 3, &mut vh, 2, 3, 2);

    // These rows become the columns of final U = V after adjointing Vh.
    for &(row, col) in &[(0, 0), (1, 1)] {
        let pivot = vh[row + 2 * col];
        assert!(pivot.im.abs() < 1e-14, "pivot {pivot} not real");
        assert!(pivot.re >= 0.0, "pivot {pivot} negative");
    }
    let after = product(&u, &vh);
    for (lhs, rhs) in after.iter().zip(&before) {
        assert!(
            (lhs - rhs).norm() < 1e-13,
            "product changed: {lhs} vs {rhs}"
        );
    }
}

#[test]
fn noncanonical_mf_svd_and_eigh_scatter_each_output_block_once() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: on a four-sector noncanonical MF input, the compact SVD and EIGH
    // fallbacks group each factor side once and iterate only the scattered
    // blocks (F = B = 16 per side), instead of G_s * B = 64 visits per side.
    use crate::probe::ScatterVisitProbe;
    let sectors = (0..4).map(SectorId::new).collect::<Vec<_>>();
    let rule = || symtn_core::ZNFusionRule::new(4).unwrap();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let tensor = tsvd_test_tensor(&rule(), &sectors);
    let bound = bound_tensor(Arc::new(rule()), &tensor);
    let adjoint_space = bound.space().adjoint_view().unwrap();
    let input = BoundDynamicTensorRef::try_new(&adjoint_space, bound.data()).unwrap();
    let scatter_visit_rec = crate::probe::record();
    let (u, vh, _) = svd_compact_factors_dyn(engine, &mut dense, &input).unwrap();
    let b_u = u.space().space().structure().block_count();
    let b_vh = vh.space().space().structure().block_count();
    assert_eq!((b_u, b_vh), (16, 16));
    let probe = scatter_visit_rec.scatter_visit();
    assert_eq!(
        probe,
        ScatterVisitProbe {
            left_grouped: b_u,
            right_grouped: b_vh,
            left_groups_built: 1,
            right_groups_built: 1,
            left_visits: b_u,
            right_visits: b_vh,
        }
    );
    assert!(probe.left_grouped + probe.left_visits < 4 * b_u);
    assert!(probe.right_grouped + probe.right_visits < 4 * b_vh);

    let tensor = hermitian_test_tensor(&rule(), &sectors);
    let bound = bound_tensor(Arc::new(rule()), &tensor);
    let adjoint_space = bound.space().adjoint_view().unwrap();
    let input = BoundDynamicTensorRef::try_new(&adjoint_space, bound.data()).unwrap();
    let scatter_visit_rec = crate::probe::record();
    let full = eigh_full_dyn(engine, &mut dense, &input, HermitianTol::DEFAULT).unwrap();
    let b_v = full.v().space().space().structure().block_count();
    assert_eq!(b_v, 16);
    let probe = scatter_visit_rec.scatter_visit();
    assert_eq!(
        probe,
        ScatterVisitProbe {
            left_grouped: b_v,
            left_groups_built: 1,
            left_visits: b_v,
            ..ScatterVisitProbe::default()
        }
    );
    assert!(probe.left_grouped + probe.left_visits < 4 * b_v);
}
