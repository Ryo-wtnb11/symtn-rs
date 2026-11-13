//! QR and LQ factorization tests.

use super::*;

#[test]
fn compact_qr_canonical_layout_skips_input_pack_and_factor_scatter() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: canonical compact QR reads source regions and writes final factor regions directly.
    let rule = ZNFusionRule::new(2).unwrap();
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let compact_qr_copy_rec = crate::probe::record();
    qr_compact(
        engine,
        &mut dense,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap();

    let probe = compact_qr_copy_rec.compact_qr_copy();
    assert_eq!(probe.input_pack_bytes, 0);
    assert_eq!(probe.output_scatter_bytes, 0);
}

#[test]
fn compact_qr_noncanonical_layout_uses_copy_fallback() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: expert noncanonical compact QR retains positive pack-and-scatter copy evidence.
    let rule = ZNFusionRule::new(2).unwrap();
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let bound = bound_tensor(Arc::new(rule), &tensor);
    let adjoint_space = bound.space().adjoint_view().unwrap();
    let input = BoundDynamicTensorRef::try_new(&adjoint_space, bound.data()).unwrap();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let compact_qr_copy_rec = crate::probe::record();
    qr_compact_dyn(engine, &mut dense, &input).unwrap();
    let probe = compact_qr_copy_rec.compact_qr_copy();

    assert!(probe.input_pack_bytes > 0);
    assert!(probe.output_scatter_bytes > 0);
}

#[test]
fn compact_qr_lq_noncanonical_layout_does_not_call_qr_into() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // The fallback still consumes `qr` outputs; its scatter is symtn-owned.
    let rule = ZNFusionRule::new(2).unwrap();
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let bound = bound_tensor(Arc::new(rule), &tensor);
    let adjoint_space = bound.space().adjoint_view().unwrap();
    let input = BoundDynamicTensorRef::try_new(&adjoint_space, bound.data()).unwrap();
    let mut dense = ScriptedExecutor::new(FailAfterObservingQrInput {
        qr_succeeds: true,
        ..Default::default()
    });

    qr_compact_dyn(engine, &mut dense, &input).unwrap();
    lq_compact_dyn(engine, &mut dense, &input).unwrap();

    assert!(!dense.observed.is_empty());
}

#[test]
fn compact_lq_canonical_layout_avoids_pack_and_scatter() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: canonical compact LQ reads and writes the sector regions in place, without general pack/scatter.
    let rule = ZNFusionRule::new(2).unwrap();
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let compact_lq_copy_rec = crate::probe::record();
    lq_compact(
        engine,
        &mut dense,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap();
    let probe = compact_lq_copy_rec.compact_lq_copy();

    assert_eq!(probe.input_pack_calls, 0);
    assert_eq!(probe.input_pack_bytes, 0);
    assert_eq!(probe.output_scatter_calls, 0);
    assert_eq!(probe.output_scatter_bytes, 0);
}

#[test]
fn generic_compact_qr_lq_paths_do_not_call_qr_into() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let (canonical_space, canonical_data) = generic_factorization_input();
    let (_, canonical_space) = bind_checked_layout(&canonical_space);
    // The adjoint view is the noncanonical (packed) source.
    let fallback_space = canonical_space.adjoint_view().unwrap();
    let canonical = BoundDynamicTensorRef::try_new(&canonical_space, &canonical_data).unwrap();
    let fallback = BoundDynamicTensorRef::try_new(&fallback_space, &canonical_data).unwrap();

    let mut direct_dense = ScriptedExecutor::new(FailAfterObservingQrInput {
        qr_succeeds: true,
        ..Default::default()
    });
    qr_compact_dyn_checked_generic(engine, &mut direct_dense, &canonical).unwrap();
    lq_compact_dyn_checked_generic(engine, &mut direct_dense, &canonical).unwrap();
    assert!(!direct_dense.observed.is_empty());

    let mut fallback_dense = ScriptedExecutor::new(FailAfterObservingQrInput {
        qr_succeeds: true,
        ..Default::default()
    });
    qr_compact_dyn_checked_generic(engine, &mut fallback_dense, &fallback).unwrap();
    lq_compact_dyn_checked_generic(engine, &mut fallback_dense, &fallback).unwrap();
    assert!(!fallback_dense.observed.is_empty());
}

#[test]
fn compact_lq_noncanonical_layout_uses_copy_fallback() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: noncanonical (adjoint-view) compact LQ retains positive general pack-and-scatter evidence.
    let rule = ZNFusionRule::new(2).unwrap();
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let bound = bound_tensor(Arc::new(rule), &tensor);
    let adjoint_space = bound.space().adjoint_view().unwrap();
    let input = BoundDynamicTensorRef::try_new(&adjoint_space, bound.data()).unwrap();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let compact_lq_copy_rec = crate::probe::record();
    lq_compact_dyn(engine, &mut dense, &input).unwrap();
    let probe = compact_lq_copy_rec.compact_lq_copy();

    assert!(probe.input_pack_bytes > 0);
    assert!(probe.output_scatter_bytes > 0);
}

#[test]
fn compact_qr_error_preserves_borrowed_input_and_publishes_no_factors() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: a QR backend failure leaves borrowed storage unchanged and returns no factor pair.
    let rule = ZNFusionRule::new(2).unwrap();
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let before = tensor.data().to_vec();
    let mut dense = ScriptedExecutor::<FailAfterObservingQrInput>::default();

    let result = qr_compact(
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
}

#[test]
fn compact_qr_uses_owned_executor_outputs_not_qr_into() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rule = ZNFusionRule::new(2).unwrap();
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let mut dense = ScriptedExecutor::new(FailAfterObservingQrInput {
        qr_succeeds: true,
        ..Default::default()
    });
    let input = bound_tensor(Arc::new(rule), &tensor);

    let Qr { q, r } = qr_compact(engine, &mut dense, &input.as_ref()).unwrap();

    assert!(!dense.observed.is_empty());
    assert!(!q.data().is_empty());
    assert!(!r.data().is_empty());
}

#[test]
fn compact_owned_qr_preserves_qr_into_output_precedence() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let tensor = rectangular_svd_tensor(2, 2);
    let bound = bound_tensor(Arc::new(ZNFusionRule::new(2).unwrap()), &tensor);
    let input = bound.as_ref();
    let mut check = |outputs: Vec<DenseTensor>, expected: DenseError| {
        let mut dense = ScriptedExecutor::new(FailAfterObservingQrInput {
            outputs: Some(outputs),
            ..Default::default()
        });
        let error = qr_compact(engine, &mut dense, &input).unwrap_err();
        assert_eq!(error, OperationError::Dense(expected));
    };

    check(
        vec![f64_qr_outputs(2, 2).remove(0)],
        arity_mismatch("qr_into", 2, 1),
    );
    check(
        f64_qr_outputs(1, 1),
        DenseError::ShapeMismatch {
            op: "qr_into",
            expected: vec![2, 2],
            actual: vec![1, 1],
        },
    );
    let mut outputs = f64_qr_outputs(2, 2);
    outputs[1] = f64_qr_outputs(1, 1).remove(1);
    check(
        outputs,
        DenseError::ShapeMismatch {
            op: "qr_into",
            expected: vec![2, 2],
            actual: vec![1, 1],
        },
    );
    let outputs = c64_qr_outputs(2, 2);
    let expected = outputs[0].as_f64_slice().unwrap_err();
    assert!(matches!(
        expected,
        DenseError::DTypeMismatch {
            actual: symtn_dense::DenseDType::C64,
            ..
        }
    ));
    let mut dense = ScriptedExecutor::new(FailAfterObservingQrInput {
        outputs: Some(outputs),
        ..Default::default()
    });
    let error = qr_compact(engine, &mut dense, &input).unwrap_err();
    assert!(matches!(error, OperationError::Dense(actual) if actual == expected));
}

#[test]
fn compact_lq_error_preserves_borrowed_input_and_publishes_no_factors() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: an LQ backend failure leaves borrowed storage unchanged and returns no factor pair.
    let rule = ZNFusionRule::new(2).unwrap();
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let before = tensor.data().to_vec();
    let mut dense = ScriptedExecutor::<FailAfterObservingQrInput>::default();

    let result = lq_compact(
        engine,
        &mut dense,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    );

    assert!(matches!(result, Err(OperationError::Dense(_))));
    assert_eq!(tensor.data(), before);
    assert!(!dense.observed.is_empty());
}

fn assert_rectangular_direct_qr(rows: usize, cols: usize) {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rule = ZNFusionRule::new(2).unwrap();
    let tensor = rectangular_svd_tensor(rows, cols);
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let compact_qr_copy_rec = crate::probe::record();
    let Qr { q, r } = qr_compact(
        engine,
        &mut dense,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap();
    assert_factor_layout_matches_per_tree_shapes(q.space());
    assert_factor_layout_matches_per_tree_shapes(r.space());
    let probe = compact_qr_copy_rec.compact_qr_copy();
    assert_eq!(probe.input_pack_bytes, 0);
    assert_eq!(probe.output_scatter_bytes, 0);
    let rank = rows.min(cols);
    if rank == 0 {
        assert!(q.space().space().homspace().domain().legs()[0]
            .sectors()
            .is_empty());
        assert!(r.space().space().homspace().codomain().legs()[0]
            .sectors()
            .is_empty());
    }
    for col in 0..cols {
        for row in 0..rows {
            let reconstructed = (0..rank)
                .map(|bond| q.data()[row + rows * bond] * r.data()[bond + rank * col])
                .sum::<f64>();
            assert!((reconstructed - tensor.data()[row + rows * col]).abs() < 1e-10);
        }
    }
}

#[test]
fn compact_qr_direct_spans_reconstruct_tall_and_wide_matrices() {
    // What: exact final Q/R spans reconstruct both compact rectangular orientations.
    assert_rectangular_direct_qr(5, 3);
    assert_rectangular_direct_qr(3, 5);
}

#[test]
fn compact_qr_zero_only_input_normalizes_to_an_empty_factorization_result() {
    // What: a zero-only row or column produces empty Q/R spaces without
    // calling an invalid factor route.
    assert_rectangular_direct_qr(0, 3);
    assert_rectangular_direct_qr(3, 0);
}

fn assert_rectangular_direct_lq(rows: usize, cols: usize) {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rule = ZNFusionRule::new(2).unwrap();
    let tensor = rectangular_svd_tensor(rows, cols);
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let compact_lq_copy_rec = crate::probe::record();
    let Lq { l: left, q: right } = lq_compact(
        engine,
        &mut dense,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap();
    assert_factor_layout_matches_per_tree_shapes(left.space());
    assert_factor_layout_matches_per_tree_shapes(right.space());
    let probe = compact_lq_copy_rec.compact_lq_copy();
    assert_eq!(probe.input_pack_bytes, 0);
    assert_eq!(probe.output_scatter_bytes, 0);
    let rank = rows.min(cols);
    if rank == 0 {
        assert!(left.space().space().homspace().domain().legs()[0]
            .sectors()
            .is_empty());
        assert!(right.space().space().homspace().codomain().legs()[0]
            .sectors()
            .is_empty());
    }
    for col in 0..cols {
        for row in 0..rows {
            let reconstructed = (0..rank)
                .map(|bond| left.data()[row + rows * bond] * right.data()[bond + rank * col])
                .sum::<f64>();
            assert!((reconstructed - tensor.data()[row + rows * col]).abs() < 1e-10);
        }
    }
}

#[test]
fn compact_lq_direct_spans_reconstruct_zero_unit_tall_wide_and_square() {
    // What: direct LQ spans reconstruct every rectangular edge orientation without general pack/scatter.
    for (rows, cols) in [(0, 3), (3, 0), (1, 1), (5, 3), (3, 5), (4, 4)] {
        assert_rectangular_direct_lq(rows, cols);
    }
}

#[test]
fn compact_qr_c64_reconstructs_mixed_tall_and_wide_sectors_without_copies() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    use num_complex::Complex64;

    // What: one complex QR call reconstructs mixed rectangular sectors in final storage.
    let rule = ZNFusionRule::new(2).unwrap();
    let source = mixed_rectangular_c32_tensor();
    let tensor = TestTensor::<Complex64>::from_vec_with_fusion_space(
        source
            .data()
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect(),
        source.fusion_space().clone(),
    )
    .unwrap();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let compact_qr_copy_rec = crate::probe::record();

    let Qr { q, r } = qr_compact(
        engine,
        &mut dense,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap();

    let probe = compact_qr_copy_rec.compact_qr_copy();
    assert_eq!(probe.input_pack_bytes, 0);
    assert_eq!(probe.output_scatter_bytes, 0);
    let input_regions = tensor
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    let q_regions = q.structure().coupled_sector_regions(1).unwrap().unwrap();
    let r_regions = r.structure().coupled_sector_regions(1).unwrap().unwrap();
    for input_region in input_regions.iter() {
        let sector = input_region.coupled();
        let q_region = q_regions
            .iter()
            .find(|region| region.coupled() == sector)
            .unwrap();
        let r_region = r_regions
            .iter()
            .find(|region| region.coupled() == sector)
            .unwrap();
        let rows = input_region.rows();
        let cols = input_region.cols();
        let rank = rows.min(cols);
        for col in 0..cols {
            for row in 0..rows {
                let reconstructed = (0..rank)
                    .map(|bond| {
                        q.data()[q_region.range().start + row + rows * bond]
                            * r.data()[r_region.range().start + bond + rank * col]
                    })
                    .sum::<Complex64>();
                let expected = tensor.data()[input_region.range().start + row + rows * col];
                assert!((reconstructed - expected).norm() < 1e-10);
            }
        }
    }
}

#[test]
fn compact_lq_c64_reconstructs_mixed_tall_and_wide_sectors() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    use num_complex::Complex64;

    // What: one complex LQ call reconstructs mixed rectangular sectors in place, without general pack/scatter.
    let rule = ZNFusionRule::new(2).unwrap();
    let source = mixed_rectangular_c32_tensor();
    let tensor = TestTensor::<Complex64>::from_vec_with_fusion_space(
        source
            .data()
            .iter()
            .map(|value| Complex64::new(value.re as f64, value.im as f64))
            .collect(),
        source.fusion_space().clone(),
    )
    .unwrap();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let compact_lq_copy_rec = crate::probe::record();

    let Lq { l: left, q: right } = lq_compact(
        engine,
        &mut dense,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap();

    let probe = compact_lq_copy_rec.compact_lq_copy();
    assert_eq!(probe.input_pack_bytes, 0);
    assert_eq!(probe.output_scatter_bytes, 0);
    let input_regions = tensor
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    let left_regions = left.structure().coupled_sector_regions(1).unwrap().unwrap();
    let right_regions = right
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    for input_region in input_regions.iter() {
        let sector = input_region.coupled();
        let left_region = left_regions
            .iter()
            .find(|region| region.coupled() == sector)
            .unwrap();
        let right_region = right_regions
            .iter()
            .find(|region| region.coupled() == sector)
            .unwrap();
        let rows = input_region.rows();
        let cols = input_region.cols();
        let rank = rows.min(cols);
        for col in 0..cols {
            for row in 0..rows {
                let reconstructed = (0..rank)
                    .map(|bond| {
                        left.data()[left_region.range().start + row + rows * bond]
                            * right.data()[right_region.range().start + bond + rank * col]
                    })
                    .sum::<Complex64>();
                let expected = tensor.data()[input_region.range().start + row + rows * col];
                assert!((reconstructed - expected).norm() < 1e-10);
            }
        }
    }
}

fn assert_compact_qr_reconstructs_rule<R>(rule: &R, sectors: &[SectorId])
where
    R: Clone + MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let tensor = tsvd_test_tensor(rule, sectors);
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let Qr { q, r } = qr_compact(
        engine,
        &mut dense,
        &bound_tensor_ref!(Arc::new((*rule).clone()), &tensor),
    )
    .unwrap();
    assert_factor_layout_matches_per_tree_shapes(q.space());
    assert_factor_layout_matches_per_tree_shapes(r.space());
    let reconstructed = contract_pair(rule, &tensor, &q, &r);
    assert_svd_blocks_match(&tensor, &reconstructed);
}

#[test]
fn compact_qr_reconstructs_u1_fermion_parity_and_product_rules() {
    // What: direct Q/R routes preserve abelian, fermionic, and encoded product sector labels.
    assert_compact_qr_reconstructs_rule(
        &U1FusionRule,
        &[
            SectorId::from(U1Irrep::new(-1).unwrap()),
            SectorId::from(U1Irrep::new(0).unwrap()),
            SectorId::from(U1Irrep::new(1).unwrap()),
        ],
    );
    assert_compact_qr_reconstructs_rule(
        &FermionParityFusionRule::new(),
        &[SectorId::new(0), SectorId::new(1)],
    );
    let product = symtn_core::ProductFusionRule::<_, _>::try_new(
        FermionParityFusionRule::new(),
        U1FusionRule,
    )
    .unwrap();
    let product_sectors = [
        product.encode_component_ids(SectorId::new(0), SectorId::from(U1Irrep::new(0).unwrap())),
        product.encode_component_ids(SectorId::new(1), SectorId::from(U1Irrep::new(1).unwrap())),
    ];
    assert_compact_qr_reconstructs_rule(&product, &product_sectors);

    let nested = symtn_core::ProductFusionRule::<_, _>::try_new(product, SU2FusionRule).unwrap();
    let nested_sectors = [
        nested.encode_component_ids(product_sectors[0], SU2Irrep::from_twice_spin(0).sector_id()),
        nested.encode_component_ids(product_sectors[1], SU2Irrep::from_twice_spin(1).sector_id()),
    ];
    let compact_qr_copy_rec = crate::probe::record();
    assert_compact_qr_reconstructs_rule(&nested, &nested_sectors);
    let probe = compact_qr_copy_rec.compact_qr_copy();
    assert_eq!(probe.input_pack_bytes, 0);
    assert_eq!(probe.output_scatter_bytes, 0);
}

fn assert_compact_lq_reconstructs_rule<R>(rule: &R, sectors: &[SectorId])
where
    R: Clone + MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let tensor = tsvd_test_tensor(rule, sectors);
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let Lq { l: left, q: right } = lq_compact(
        engine,
        &mut dense,
        &bound_tensor_ref!(Arc::new((*rule).clone()), &tensor),
    )
    .unwrap();
    assert_factor_layout_matches_per_tree_shapes(left.space());
    assert_factor_layout_matches_per_tree_shapes(right.space());
    let reconstructed = contract_pair(rule, &tensor, &left, &right);
    assert_svd_blocks_match(&tensor, &reconstructed);
}

#[test]
fn compact_lq_reconstructs_u1_fermion_parity_and_product_rules() {
    // What: direct LQ routes preserve non-Abelian, abelian, fermionic, and nested product sector labels.
    assert_compact_lq_reconstructs_rule(
        &U1FusionRule,
        &[
            SectorId::from(U1Irrep::new(-1).unwrap()),
            SectorId::from(U1Irrep::new(0).unwrap()),
            SectorId::from(U1Irrep::new(1).unwrap()),
        ],
    );
    assert_compact_lq_reconstructs_rule(
        &SU2FusionRule,
        &[
            SU2Irrep::from_twice_spin(0).sector_id(),
            SU2Irrep::from_twice_spin(1).sector_id(),
        ],
    );
    assert_compact_lq_reconstructs_rule(
        &FermionParityFusionRule::new(),
        &[SectorId::new(0), SectorId::new(1)],
    );
    let product = symtn_core::ProductFusionRule::<_, _>::try_new(
        FermionParityFusionRule::new(),
        U1FusionRule,
    )
    .unwrap();
    let product_sectors = [
        product.encode_component_ids(SectorId::new(0), SectorId::from(U1Irrep::new(0).unwrap())),
        product.encode_component_ids(SectorId::new(1), SectorId::from(U1Irrep::new(1).unwrap())),
    ];
    assert_compact_lq_reconstructs_rule(&product, &product_sectors);

    let nested = symtn_core::ProductFusionRule::<_, _>::try_new(product, SU2FusionRule).unwrap();
    let nested_sectors = [
        nested.encode_component_ids(product_sectors[0], SU2Irrep::from_twice_spin(0).sector_id()),
        nested.encode_component_ids(product_sectors[1], SU2Irrep::from_twice_spin(1).sector_id()),
    ];
    let compact_lq_copy_rec = crate::probe::record();
    assert_compact_lq_reconstructs_rule(&nested, &nested_sectors);
    let probe = compact_lq_copy_rec.compact_lq_copy();
    assert_eq!(probe.input_pack_bytes, 0);
    assert_eq!(probe.output_scatter_bytes, 0);
}

fn bound_factor_matrices<R>(
    factor: &BoundDynFactor<R, f64>,
) -> Vec<(SectorId, usize, usize, Vec<f64>)> {
    factor
        .space()
        .space()
        .structure()
        .coupled_sector_regions(factor.space().space().nout())
        .unwrap()
        .unwrap()
        .iter()
        .map(|region| {
            assert_eq!(region.range().len(), region.rows() * region.cols());
            (
                region.coupled(),
                region.rows(),
                region.cols(),
                factor.data()[region.range()].to_vec(),
            )
        })
        .collect()
}

fn assert_orthonormal_rows(matrices: &[(SectorId, usize, usize, Vec<f64>)]) {
    for (sector, rows, cols, matrix) in matrices {
        for upper in 0..*rows {
            for lower in 0..*rows {
                let dot = (0..*cols)
                    .map(|col| matrix[upper + rows * col] * matrix[lower + rows * col])
                    .sum::<f64>();
                let expected = if upper == lower { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1.0e-9,
                    "sector {sector:?}: row dot ({upper},{lower}) = {dot}"
                );
            }
        }
    }
}

fn assert_nonnegative_diagonal(matrices: &[(SectorId, usize, usize, Vec<f64>)]) {
    for (sector, rows, cols, matrix) in matrices {
        for index in 0..(*rows).min(*cols) {
            assert!(
                matrix[index + rows * index] >= 0.0,
                "sector {sector:?}: diagonal {index} is negative"
            );
        }
    }
}

fn assert_full_qr_observation(
    observation: &FullQrObservation,
    input: &[Complex64],
    rows: usize,
    cols: usize,
) {
    let dense_cols = if rows <= cols { cols } else { cols + rows };
    assert_eq!(observation.input_shape, [rows, dense_cols]);
    assert_eq!(observation.q_shape, [rows, rows]);
    assert_eq!(observation.r_shape, [rows, dense_cols]);
    let mut expected = vec![Complex64::new(0.0, 0.0); rows * dense_cols];
    expected[..rows * cols].copy_from_slice(input);
    if rows > cols {
        for row in 0..rows {
            expected[rows * cols + row * rows + row] = Complex64::new(1.0, 0.0);
        }
    }
    assert_eq!(observation.values, expected);
}

/// The transpose, not the adjoint: full LQ factors `X^T`, read in place.
fn transpose_complex(input: &[Complex64], rows: usize, cols: usize) -> Vec<Complex64> {
    let mut output = vec![Complex64::new(0.0, 0.0); input.len()];
    for col in 0..cols {
        for row in 0..rows {
            output[col + cols * row] = input[row + rows * col];
        }
    }
    output
}

#[test]
fn full_qr_and_lq_use_original_input_only_when_economy_q_is_full() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rule = ZNFusionRule::new(2).unwrap();
    let tensor = mixed_rectangular_tensor((2, 4), (3, 1));
    let matrices = dense_sector_matrices(1, &tensor);
    let input = bound_tensor(Arc::new(rule), &tensor);
    let input_ref = input.as_ref();
    let input = input_ref.dynamic();

    let mut qr_dense = ScriptedExecutor::<FullQrInputSpy>::default();
    let Qr { q, r } = qr_full_dyn(engine, &mut qr_dense, &input).unwrap();
    assert_eq!(qr_dense.observations.len(), matrices.len());
    for (observation, (_, rows, cols, matrix)) in qr_dense.observations.iter().zip(matrices.iter())
    {
        let matrix = matrix
            .iter()
            .map(|&value| Complex64::new(value, 0.0))
            .collect::<Vec<_>>();
        assert_full_qr_observation(observation, &matrix, *rows, *cols);
    }
    assert_orthonormal_columns(&bound_factor_matrices(&q));
    assert_nonnegative_diagonal(&bound_factor_matrices(&r));
    assert_compact_factors_reconstruct_input(&input, &q, None, &r);

    let mut lq_dense = ScriptedExecutor::<FullQrInputSpy>::default();
    let Lq { l, q } = lq_full_dyn(engine, &mut lq_dense, &input).unwrap();
    assert_eq!(lq_dense.observations.len(), matrices.len());
    for (observation, (_, rows, cols, matrix)) in lq_dense.observations.iter().zip(matrices.iter())
    {
        let matrix = matrix
            .iter()
            .map(|&value| Complex64::new(value, 0.0))
            .collect::<Vec<_>>();
        let transpose = transpose_complex(&matrix, *rows, *cols);
        assert_full_qr_observation(observation, &transpose, *cols, *rows);
    }
    assert_nonnegative_diagonal(&bound_factor_matrices(&l));
    assert_orthonormal_rows(&bound_factor_matrices(&q));
    assert_compact_factors_reconstruct_input(&input, &l, None, &q);
}

fn checked_fixture_matrices(
    space: &BoundDynamicFusionMapSpace<LateGenericSpy>,
    data: &[Complex64],
) -> Vec<(usize, usize, Vec<Complex64>)> {
    (0..space.space().structure().block_count())
        .map(|index| {
            let block = space.space().structure().block(index).unwrap();
            let (rows, cols) = (block.shape()[0], block.shape()[1]);
            let mut matrix = vec![Complex64::new(0.0, 0.0); rows * cols];
            for col in 0..cols {
                for row in 0..rows {
                    matrix[row + rows * col] =
                        data[block.offset() + row * block.strides()[0] + col * block.strides()[1]];
                }
            }
            (rows, cols, matrix)
        })
        .collect()
}

fn assert_checked_full_qr_lq_inputs(
    provider: Arc<LateGenericSpy>,
    space: BoundDynamicFusionMapSpace<LateGenericSpy>,
    data: Vec<Complex64>,
) {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let matrices = checked_fixture_matrices(&space, &data);
    let input = BoundDynamicTensorRef::try_new(&space, &data).unwrap();

    let mut qr_dense = ScriptedExecutor::<FullQrInputSpy>::default();
    let Qr { q, r } = qr_full_dyn_checked_generic(engine, &mut qr_dense, &input).unwrap();
    assert_eq!(qr_dense.observations.len(), matrices.len());
    for (observation, (rows, cols, matrix)) in qr_dense.observations.iter().zip(matrices.iter()) {
        assert_full_qr_observation(observation, matrix, *rows, *cols);
    }
    assert_compact_factors_reconstruct_input(&input, &q, None, &r);
    assert!(Arc::ptr_eq(q.space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(r.space().provider_arc(), &provider));

    let mut lq_dense = ScriptedExecutor::<FullQrInputSpy>::default();
    let Lq { l, q } = lq_full_dyn_checked_generic(engine, &mut lq_dense, &input).unwrap();
    assert_eq!(lq_dense.observations.len(), matrices.len());
    for (observation, (rows, cols, matrix)) in lq_dense.observations.iter().zip(matrices.iter()) {
        let transpose = transpose_complex(matrix, *rows, *cols);
        assert_full_qr_observation(observation, &transpose, *cols, *rows);
    }
    assert_compact_factors_reconstruct_input(&input, &l, None, &q);
    assert!(Arc::ptr_eq(l.space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(q.space().provider_arc(), &provider));
}

#[test]
fn checked_full_qr_and_lq_preserve_complex_inputs_and_provider_identity() {
    let (provider, space, data) = checked_svd_truncation_input::<Complex64>(true);
    assert_checked_full_qr_lq_inputs(provider, space, data);
    let (provider, space, data) = checked_svd_wide_input::<Complex64>();
    assert_checked_full_qr_lq_inputs(provider, space, data);
}

#[test]
fn full_and_compact_qr_lq_match_for_rank_deficient_no_completion_shapes() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: with no completion columns the full and compact factorizations
    // agree (path agreement under the workspace rule, 3 = the long side), and
    // the full factors are orthonormal and reconstruct the input.
    let rule = ZNFusionRule::new(2).unwrap();
    let wide_space = rectangular_svd_tensor(2, 3).fusion_space().clone();
    let wide =
        TestTensor::from_vec_with_fusion_space(vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0], wide_space)
            .unwrap();
    let wide_input = bound_tensor(Arc::new(rule.clone()), &wide);
    let wide_input_ref = wide_input.as_ref();
    let wide_input = wide_input_ref.dynamic();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let compact = qr_compact_dyn(engine, &mut dense, &wide_input).unwrap();
    let full = qr_full_dyn(engine, &mut dense, &wide_input).unwrap();
    assert_eq!(full.q.space().space(), compact.q.space().space());
    assert_eq!(full.r.space().space(), compact.r.space().space());
    numerics::assert_slices_close("full vs compact", full.q.data(), compact.q.data(), 3);
    numerics::assert_slices_close("full vs compact", full.r.data(), compact.r.data(), 3);
    assert_orthonormal_columns(&bound_factor_matrices(&full.q));
    assert_nonnegative_diagonal(&bound_factor_matrices(&full.r));
    assert_compact_factors_reconstruct_input(&wide_input, &full.q, None, &full.r);

    let tall = transposed_rectangular_tensor(&wide, 2, 3);
    let tall_input = bound_tensor(Arc::new(rule), &tall);
    let tall_input_ref = tall_input.as_ref();
    let tall_input = tall_input_ref.dynamic();
    let compact = lq_compact_dyn(engine, &mut dense, &tall_input).unwrap();
    let full = lq_full_dyn(engine, &mut dense, &tall_input).unwrap();
    assert_eq!(full.l.space().space(), compact.l.space().space());
    assert_eq!(full.q.space().space(), compact.q.space().space());
    numerics::assert_slices_close("full vs compact", full.l.data(), compact.l.data(), 3);
    numerics::assert_slices_close("full vs compact", full.q.data(), compact.q.data(), 3);
    assert_nonnegative_diagonal(&bound_factor_matrices(&full.l));
    assert_orthonormal_rows(&bound_factor_matrices(&full.q));
    assert_compact_factors_reconstruct_input(&tall_input, &full.l, None, &full.q);
}

#[test]
fn qr_full_gives_square_unitary_and_reconstructs() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rule = SU2FusionRule;
    let tensor = tsvd_test_tensor(
        &rule,
        &[
            SU2Irrep::from_twice_spin(0).sector_id(),
            SU2Irrep::from_twice_spin(1).sector_id(),
        ],
    );
    let mut dense_executor = symtn_dense::DefaultDenseExecutor::new();
    let Qr { q, r } = qr_full(
        engine,
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap();

    let matrices = dense_sector_matrices(2, &q);
    for (_, rows, cols, _) in &matrices {
        assert_eq!(rows, cols, "full Q must be square per sector");
    }
    assert_orthonormal_columns(&matrices);

    let reconstructed = contract_pair(&rule, &tensor, &q, &r);
    assert_svd_blocks_match(&tensor, &reconstructed);
}

#[test]
fn lq_full_reconstructs() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rule = ZNFusionRule::new(2).unwrap();
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let mut dense_executor = symtn_dense::DefaultDenseExecutor::new();
    let Lq { l, q } = lq_full(
        engine,
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule.clone()), &tensor),
    )
    .unwrap();
    let reconstructed = contract_pair(&rule, &tensor, &l, &q);
    assert_svd_blocks_match(&tensor, &reconstructed);
}

#[test]
fn null_qr_failure_preserves_input_and_builds_no_factor() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    for (left, rows, cols) in [(true, 3, 2), (false, 2, 3)] {
        let tensor =
            one_sector_rectangular_matrix(vec![1.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0], rows, cols);
        let before = tensor.data().to_vec();
        let input = bound_tensor(Arc::new(ZNFusionRule::new(2).unwrap()), &tensor);
        let mut dense = ScriptedExecutor::<FailNullQr>::default();

        let buffer_builds_rec = crate::probe::record();
        let result = if left {
            left_null(engine, &mut dense, &input.as_ref())
        } else {
            right_null(engine, &mut dense, &input.as_ref())
        };

        assert!(
            matches!(
                &result,
                Err(OperationError::Dense(DenseError::Backend { op: "qr_into", message, .. }))
                    if message == "injected null QR failure"
            ),
            "{result:?}"
        );
        assert_eq!(dense.counts().qr_into, 1);
        assert_eq!(buffer_builds_rec.factor_buffer_builds(), (0, 0));
        assert_eq!(tensor.data(), before);
    }
}

#[test]
fn positive_diagonal_gauge_matches_tensorkit_qr_reference() {
    // TensorKit 0.17.0 / MatrixAlgebraKit 0.6.8 crosscheck:
    //   A = [-1 2; 3 4; 5 -6]; Q, R = MatrixAlgebraKit.qr_compact(A)
    // (default `positive = true` since MAK 0.6.8). Column-major reference:
    let q_ref = [
        -0.16903085094570325,
        0.50709255283711,
        0.8451542547285166,
        0.21398024625545642,
        0.8559209850218259,
        -0.4707565417620042,
    ];
    let r_ref = [
        5.916079783099615,
        0.0,
        -3.380617018914066,
        6.676183683170241,
    ];
    // Start from the equally valid un-gauged QR with both diagonal signs
    // flipped (Q -> -Q, R -> -R); the gauge must restore the reference.
    let mut q: Vec<f64> = q_ref.iter().map(|v| -v).collect();
    let mut r: Vec<f64> = r_ref.iter().map(|v| -v).collect();
    crate::factorize::positive_diagonal_gauge(&mut q, 3, &mut r, 2, 2);
    for (value, reference) in q.iter().zip(&q_ref) {
        assert!(
            (value - reference).abs() < 1e-14,
            "Q {value} != {reference}"
        );
    }
    for (value, reference) in r.iter().zip(&r_ref) {
        assert!(
            (value - reference).abs() < 1e-14,
            "R {value} != {reference}"
        );
    }
}

#[test]
fn qr_compact_positive_gauge_idempotent_on_isometry() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    for rule_case in [0usize, 1usize] {
        if rule_case == 0 {
            let rule = ZNFusionRule::new(2).unwrap();
            let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
            let mut dense_executor = symtn_dense::DefaultDenseExecutor::new();
            let input = bound_tensor(Arc::new(rule), &tensor);
            let Qr { q, .. } = qr_compact(engine, &mut dense_executor, &input.as_ref()).unwrap();
            let Qr { q: q2, r: r2 } = qr_compact(engine, &mut dense_executor, &q.as_ref()).unwrap();
            assert_svd_blocks_match(&q, &q2);
            assert_identity_sector_matrices(&dense_sector_matrices(1, &r2));
        } else {
            let rule = SU2FusionRule;
            let tensor = tsvd_test_tensor(
                &rule,
                &[
                    SU2Irrep::from_twice_spin(0).sector_id(),
                    SU2Irrep::from_twice_spin(1).sector_id(),
                ],
            );
            let mut dense_executor = symtn_dense::DefaultDenseExecutor::new();
            let input = bound_tensor(Arc::new(rule), &tensor);
            let Qr { q, .. } = qr_compact(engine, &mut dense_executor, &input.as_ref()).unwrap();
            let Qr { q: q2, r: r2 } = qr_compact(engine, &mut dense_executor, &q.as_ref()).unwrap();
            assert_svd_blocks_match(&q, &q2);
            assert_identity_sector_matrices(&dense_sector_matrices(1, &r2));
        }
    }
}

#[test]
fn lq_compact_positive_gauge_idempotent_on_isometry() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rule = ZNFusionRule::new(2).unwrap();
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let mut dense_executor = symtn_dense::DefaultDenseExecutor::new();
    let input = bound_tensor(Arc::new(rule), &tensor);
    let Lq { q, .. } = lq_compact(engine, &mut dense_executor, &input.as_ref()).unwrap();
    let Lq { l: l2, q: q2 } = lq_compact(engine, &mut dense_executor, &q.as_ref()).unwrap();
    assert_svd_blocks_match(&q, &q2);
    assert_identity_sector_matrices(&dense_sector_matrices(1, &l2));
}

#[test]
fn lazy_adjoint_lq_tree_route_factors_the_logical_source() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: LQ of the lazy adjoint of a parent that is not region-tiled (an
    // adjoint view) packs it, publishes on the factor spaces LQ of the
    // materialized adjoint publishes on, and factors the logical source `X`.
    let rule = ZNFusionRule::new(2).unwrap();
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let bound = bound_tensor(Arc::new(rule), &tensor);
    let bound = bound.as_ref();
    let materialized = bound.dynamic();
    let view_space = materialized.space().adjoint_view().unwrap();
    let parent = BoundDynamicTensorRef::try_new(&view_space, materialized.data()).unwrap();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    for full in [false, true] {
        let input_pack_bytes_rec = crate::probe::record();
        let by_tree = if full {
            crate::factorize::lq_full_adjoint_dyn(engine, &mut dense, &parent, materialized.space())
        } else {
            crate::factorize::lq_compact_adjoint_dyn(
                engine,
                &mut dense,
                &parent,
                materialized.space(),
            )
        }
        .unwrap();
        assert!(
            input_pack_bytes_rec.get(crate::probe::Event::InputPackBytes) > 0,
            "full {full}"
        );
        let eager = if full {
            crate::factorize::lq_full_dyn(engine, &mut dense, &materialized)
        } else {
            crate::factorize::lq_compact_dyn(engine, &mut dense, &materialized)
        }
        .unwrap();
        for (expected, tree) in [(&eager.l, &by_tree.l), (&eager.q, &by_tree.q)] {
            assert_eq!(
                expected.space().space(),
                tree.space().space(),
                "full {full}"
            );
        }
        // Oracle: the logical source `X` itself. `L Q` reconstructs it and
        // `Q` has orthonormal rows, within the fixture helpers' f64
        // tolerances (1e-10 per entry, 1e-9 per row dot product).
        assert_compact_factors_reconstruct_input(&materialized, &by_tree.l, None, &by_tree.q);
        assert_orthonormal_rows(&bound_factor_matrices(&by_tree.q));
    }
}

#[test]
fn checked_lazy_adjoint_lq_tree_route_factors_the_logical_source() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: the checked-Generic counterpart of the multiplicity-free test
    // above.
    let (space, data) = generic_factorization_input();
    let (_, space) = bind_checked_layout(&space);
    let materialized = BoundDynamicTensorRef::try_new(&space, &data).unwrap();
    let view_space = space.adjoint_view().unwrap();
    let parent = BoundDynamicTensorRef::try_new(&view_space, &data).unwrap();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    for full in [false, true] {
        let input_pack_bytes_rec = crate::probe::record();
        let by_tree = if full {
            crate::factorize::lq_full_adjoint_dyn_checked_generic(
                engine, &mut dense, &parent, &space,
            )
        } else {
            crate::factorize::lq_compact_adjoint_dyn_checked_generic(
                engine, &mut dense, &parent, &space,
            )
        }
        .unwrap();
        assert!(
            input_pack_bytes_rec.get(crate::probe::Event::InputPackBytes) > 0,
            "full {full}"
        );
        let eager = if full {
            crate::factorize::lq_full_dyn_checked_generic(engine, &mut dense, &materialized)
        } else {
            crate::factorize::lq_compact_dyn_checked_generic(engine, &mut dense, &materialized)
        }
        .unwrap();
        for (expected, tree) in [(&eager.l, &by_tree.l), (&eager.q, &by_tree.q)] {
            assert_eq!(
                expected.space().space(),
                tree.space().space(),
                "full {full}"
            );
        }
        // Oracle as above: `L Q` reconstructs `X` and `Q` has orthonormal rows.
        assert_compact_factors_reconstruct_input(&materialized, &by_tree.l, None, &by_tree.q);
        assert_orthonormal_rows(&bound_factor_matrices(&by_tree.q));
    }
}

/// U(1) fusion declared under Generic style, so the checked route admits the
/// exact geometry the multiplicity-free route factors.
struct U1AsGeneric;

impl symtn_core::testing::Provider for U1AsGeneric {}

impl FusionRule for U1AsGeneric {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Generic
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        U1FusionRule.braiding_style()
    }
    fn vacuum(&self) -> SectorId {
        U1FusionRule.vacuum()
    }
    fn dual(&self, sector: SectorId) -> SectorId {
        U1FusionRule.dual(sector)
    }
    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        U1FusionRule.fusion_channels(left, right)
    }
    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        U1FusionRule.nsymbol(left, right, coupled)
    }
    fn sector_order_key(&self, sector: SectorId) -> symtn_core::SectorOrderKey {
        U1FusionRule.sector_order_key(sector)
    }
}

/// Warm allocation calls of checked compact LQ (eager, lazy adjoint) minus
/// those of the multiplicity-free direct-region route on the same U(1) data
/// and layout, with charges `-m..=m` on every leg (`4m + 1` coupled sectors).
fn checked_lq_excess_calls<D: FactorScalar>(m: i32) -> [i64; 2] {
    use super::null_space::counting_alloc;
    static RULE: U1AsGeneric = U1AsGeneric;
    let leg = |degeneracy: usize| {
        SectorLeg::new(
            (-m..=m).map(|charge| (SectorId::from(U1Irrep::new(charge).unwrap()), degeneracy)),
            false,
        )
    };
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(2), leg(2)]),
        FusionProductSpace::new([leg(1), leg(1)]),
    );
    let mf = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::new(U1FusionRule),
        homspace.clone(),
    )
    .unwrap();
    let checked = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::new(InfallibleGeneric::new(&RULE)),
        homspace,
    )
    .unwrap();
    let data = (0..mf.space().required_len().unwrap())
        .map(|index| D::from_real(((index * 7 + 3) as f64 * 0.37).sin()))
        .collect::<Vec<_>>();
    let mf_input = BoundDynamicTensorRef::try_new(&mf, &data).unwrap();
    let checked_input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let mf_adjoint =
        <symtn_core::MultiplicityFreeAdmissionMode as symtn_tensors::CoefficientAlgebra<
            U1FusionRule,
        >>::adjoint_space(mf_input.space())
        .unwrap();
    let checked_adjoint = <CheckedGenericAdmissionMode as symtn_tensors::CoefficientAlgebra<
        InfallibleGeneric<'static, U1AsGeneric>,
    >>::adjoint_space(checked_input.space())
    .unwrap();
    let mut engine = symtn_dense::CpuEngine::serial();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let mut warm = |call: &mut dyn FnMut(
        &mut symtn_dense::DefaultDenseExecutor,
        &mut dyn symtn_dense::Engine,
    )| {
        call(&mut dense, &mut engine);
        call(&mut dense, &mut engine);
        counting_alloc::measure(|| call(&mut dense, &mut engine))
            .1
            .calls as i64
    };
    let eager = warm(&mut |dense, engine| {
        lq_compact_dyn_checked_generic(engine, dense, &checked_input).unwrap();
    }) - warm(&mut |dense, engine| {
        lq_compact_dyn(engine, dense, &mf_input).unwrap();
    });
    let lazy = warm(&mut |dense, engine| {
        crate::factorize::lq_compact_adjoint_dyn_checked_generic(
            engine,
            dense,
            &checked_input,
            &checked_adjoint,
        )
        .unwrap();
    }) - warm(&mut |dense, engine| {
        crate::factorize::lq_compact_adjoint_dyn(engine, dense, &mf_input, &mf_adjoint).unwrap();
    });
    [eager, lazy]
}

/// Each coupled sector's column-major payload of `factor`.
fn factor_sector_data<R, D: Copy>(
    factor: &BoundDynFactor<R, D>,
) -> std::collections::BTreeMap<SectorId, Vec<D>> {
    factor
        .space()
        .space()
        .structure()
        .coupled_sector_regions(factor.space().space().nout())
        .unwrap()
        .unwrap()
        .iter()
        .map(|region| (region.coupled(), factor.data()[region.range()].to_vec()))
        .collect()
}

#[test]
fn lazy_adjoint_compact_lq_is_the_gauged_parent_qr_adjointed() {
    use num_complex::Complex64;
    // What: compact LQ of a lazy adjoint `X = A^H` on the region-tiled
    // routes (multiplicity-free and checked) is, bit for bit, the
    // positive-diagonal-gauged QR `A = Q R` of each parent block followed by
    // `l = R^H`, `q = Q^H`: the gauge applied on the final placement equals
    // the gauge applied before the adjoint. Complex entries, so a dropped or
    // doubled conjugation shows.
    static RULE: U1AsGeneric = U1AsGeneric;
    let leg = |degeneracy: usize| {
        SectorLeg::new(
            (-1..=1).map(|charge| (SectorId::from(U1Irrep::new(charge).unwrap()), degeneracy)),
            false,
        )
    };
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(2), leg(2)]),
        FusionProductSpace::new([leg(1), leg(1)]),
    );
    let mf = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::new(U1FusionRule),
        homspace.clone(),
    )
    .unwrap();
    let checked = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::new(InfallibleGeneric::new(&RULE)),
        homspace,
    )
    .unwrap();
    let data = (0..mf.space().required_len().unwrap())
        .map(|index| {
            let x = (index * 7 + 3) as f64;
            Complex64::new((x * 0.37).sin(), (x * 0.53).cos())
        })
        .collect::<Vec<_>>();
    let mf_input = BoundDynamicTensorRef::try_new(&mf, &data).unwrap();
    let checked_input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let mf_adjoint =
        <symtn_core::MultiplicityFreeAdmissionMode as symtn_tensors::CoefficientAlgebra<
            U1FusionRule,
        >>::adjoint_space(mf_input.space())
        .unwrap();
    let checked_adjoint = <CheckedGenericAdmissionMode as symtn_tensors::CoefficientAlgebra<
        InfallibleGeneric<'static, U1AsGeneric>,
    >>::adjoint_space(checked_input.space())
    .unwrap();
    let engine = &mut symtn_dense::CpuEngine::serial();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let adjoint = |data: &[Complex64], rows: usize, cols: usize| {
        let mut out = vec![Complex64::default(); data.len()];
        for col in 0..cols {
            for row in 0..rows {
                out[col + cols * row] = data[row + rows * col].conj();
            }
        }
        out
    };
    let mut want_l = std::collections::BTreeMap::new();
    let mut want_q = std::collections::BTreeMap::new();
    for region in mf
        .space()
        .structure()
        .coupled_sector_regions(mf.space().nout())
        .unwrap()
        .unwrap()
        .iter()
    {
        let (rows, cols) = (region.rows(), region.cols());
        let rank = rows.min(cols);
        let shape = [rows, cols];
        let strides = [1, rows];
        let view = symtn_dense::DenseView::new(&data[region.range()], &shape, &strides, 0).unwrap();
        let [q, r]: [symtn_dense::DenseTensor; 2] = dense
            .qr(engine, DenseRead::C64(view))
            .unwrap()
            .try_into()
            .unwrap();
        let (mut q, mut r) = (
            q.as_c64_slice().unwrap().to_vec(),
            r.as_c64_slice().unwrap().to_vec(),
        );
        crate::factorize::positive_diagonal_gauge(&mut q, rows, &mut r, rank, cols);
        want_l.insert(region.coupled(), adjoint(&r, rank, cols));
        want_q.insert(region.coupled(), adjoint(&q, rows, rank));
    }

    let mf_lq =
        crate::factorize::lq_compact_adjoint_dyn(engine, &mut dense, &mf_input, &mf_adjoint)
            .unwrap();
    let checked_lq = crate::factorize::lq_compact_adjoint_dyn_checked_generic(
        engine,
        &mut dense,
        &checked_input,
        &checked_adjoint,
    )
    .unwrap();
    for (route, l, q) in [
        (
            "multiplicity-free",
            factor_sector_data(&mf_lq.l),
            factor_sector_data(&mf_lq.q),
        ),
        (
            "checked",
            factor_sector_data(&checked_lq.l),
            factor_sector_data(&checked_lq.q),
        ),
    ] {
        assert_eq!(l, want_l, "{route} l");
        assert_eq!(q, want_q, "{route} q");
    }
}

/// Fails its first QR solve when `fail` (recording that it did in
/// `injected`), and refuses every linear-algebra scope at exit.
struct LqRefusingScope {
    fail: bool,
    qr_calls: usize,
    injected: bool,
}

impl crate::tests::scripted_executor::Observer for LqRefusingScope {
    fn call(
        &mut self,
        call: crate::tests::scripted_executor::Call<'_, '_>,
    ) -> Option<crate::tests::scripted_executor::Reply> {
        if let crate::tests::scripted_executor::Call::Read(
            crate::tests::scripted_executor::Op::Qr,
            _,
        ) = call
        {
            self.qr_calls += 1;
            if self.fail && self.qr_calls == 1 {
                self.injected = true;
                return Some(Err(DenseError::OutOfBounds));
            }
        }
        None
    }

    fn scope_exit(&mut self, _: crate::tests::scripted_executor::Counts) -> Result<(), DenseError> {
        Err(scope_refusal("lq scope"))
    }
}

#[test]
fn compact_lq_runs_in_one_scope_whose_refusal_wins() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: compact LQ, ordinary and of a lazy adjoint, runs its dense batch
    // inside the executor's linear-algebra scope: a scope that refuses at
    // exit fails an otherwise successful call, and its refusal takes
    // precedence over a solver failure inside it.
    let rule = ZNFusionRule::new(2).unwrap();
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let bound = bound_tensor(Arc::new(rule), &tensor);
    let bound = bound.as_ref();
    let materialized = bound.dynamic();
    let view_space = materialized.space().adjoint_view().unwrap();
    let parent = BoundDynamicTensorRef::try_new(&view_space, materialized.data()).unwrap();
    type Route<'r> = dyn Fn(
            &mut dyn symtn_dense::Engine,
            &mut crate::tests::scripted_executor::ScriptedExecutor<LqRefusingScope>,
        ) -> Result<(), OperationError>
        + 'r;
    let ordinary: &Route<'_> = &|engine, dense| lq_compact(engine, dense, &bound).map(|_| ());
    let lazy: &Route<'_> = &|engine, dense| {
        crate::factorize::lq_compact_adjoint_dyn(engine, dense, &parent, materialized.space())
            .map(|_| ())
    };
    for fail in [false, true] {
        for (route, run) in [("ordinary", ordinary), ("lazy", lazy)] {
            // A fresh executor per route, so each route's first QR solve is
            // the one that fails.
            let mut dense =
                crate::tests::scripted_executor::ScriptedExecutor::new(LqRefusingScope {
                    fail,
                    qr_calls: 0,
                    injected: false,
                });
            let outcome = run(engine, &mut dense);
            assert_eq!(
                dense.observer.injected, fail,
                "{route}: solver failure injected"
            );
            assert_eq!(
                outcome,
                Err(OperationError::Dense(scope_refusal("lq scope"))),
                "{route}, solver failure {fail}"
            );
        }
    }
}

#[test]
fn checked_compact_lq_adds_no_per_sector_allocation_over_the_multiplicity_free_route() {
    // What: checked compact LQ, eager and lazy adjoint, runs the same
    // executor batch and places each sector's final factors once into the
    // buffers it publishes, as the multiplicity-free route does. Its
    // allocation excess over that route is then the per-call checked
    // staging alone, so it must not grow with the coupled-sector count.
    fn check<D: FactorScalar>() {
        let few = checked_lq_excess_calls::<D>(1);
        let many = checked_lq_excess_calls::<D>(2);
        eprintln!(
            "checked - MF LQ warm calls (eager, lazy): 5 sectors {few:?}, 9 sectors {many:?}"
        );
        for (few, many) in few.into_iter().zip(many) {
            assert!(many <= few, "5 sectors {few}, 9 sectors {many}");
        }
    }
    check::<f64>();
    check::<Complex64>();
}

#[test]
fn positive_diagonal_gauge_complex_phase_and_zero_diagonal() {
    use num_complex::Complex64;
    let c = Complex64::new;
    // q: 3 x 3, r: 3 x 3 upper triangular with complex diagonal phases and a
    // zero diagonal entry (row 1), column-major.
    let q: Vec<Complex64> = (0..9)
        .map(|i| c((i as f64 * 0.7 - 2.0).sin(), (i as f64 * 1.3 + 0.5).cos()))
        .collect();
    let r = vec![
        c(-3.0, 4.0),
        c(0.0, 0.0),
        c(0.0, 0.0),
        c(1.0, -2.0),
        c(0.0, 0.0),
        c(0.0, 0.0),
        c(0.5, 0.25),
        c(2.0, 1.0),
        c(0.0, -7.0),
    ];
    let product = |q: &[Complex64], r: &[Complex64]| -> Vec<Complex64> {
        let mut out = vec![c(0.0, 0.0); 9];
        for col in 0..3 {
            for row in 0..3 {
                for k in 0..3 {
                    out[row + 3 * col] += q[row + 3 * k] * r[k + 3 * col];
                }
            }
        }
        out
    };
    let before = product(&q, &r);
    let mut q_gauged = q.clone();
    let mut r_gauged = r.clone();
    crate::factorize::positive_diagonal_gauge(&mut q_gauged, 3, &mut r_gauged, 3, 3);
    // Diagonal of R is real non-negative; the zero entry keeps phase 1.
    for j in 0..3 {
        let diagonal = r_gauged[j + 3 * j];
        assert!(
            diagonal.im.abs() < 1e-14,
            "R[{j},{j}] = {diagonal} not real"
        );
        assert!(diagonal.re >= 0.0, "R[{j},{j}] = {diagonal} negative");
    }
    let zero_diagonal_row = 1;
    let leading_dimension = 3;
    let zero_diagonal_index = zero_diagonal_row + leading_dimension * zero_diagonal_row;
    assert_eq!(r_gauged[zero_diagonal_index], c(0.0, 0.0));
    assert_eq!(q_gauged[3], q[3], "zero diagonal must not rescale Q column");
    // Q * R is unchanged.
    let after = product(&q_gauged, &r_gauged);
    for (lhs, rhs) in after.iter().zip(&before) {
        assert!(
            (lhs - rhs).norm() < 1e-13,
            "product changed: {lhs} vs {rhs}"
        );
    }
}

/// Each coupled sector's `(sector, rows, cols, column-major entries)` of a
/// one-block-per-sector bound layout, widened to `Complex64`.
fn sector_matrices_c64<D: FactorScalar>(
    structure: &symtn_core::BlockStructure,
    nout: usize,
    data: &[D],
) -> Vec<(SectorId, usize, usize, Vec<Complex64>)> {
    structure
        .coupled_sector_regions(nout)
        .unwrap()
        .unwrap()
        .iter()
        .map(|region| {
            let entries = data[region.range()].iter().map(|v| v.widen_complex());
            (
                region.coupled(),
                region.rows(),
                region.cols(),
                entries.collect(),
            )
        })
        .collect()
}

/// One-block-per-sector U(1) map of sector shapes 2x5, 4x2, 3x3 and 1x4
/// (wide, tall, square, wide) with distinct nonzero entries.
fn unequal_sector_tensor<D: FactorScalar>() -> TestTensor<D> {
    let codomain = SectorLeg::new(
        [(0, 2), (1, 4), (2, 3), (3, 1)]
            .map(|(c, d)| (SectorId::from(U1Irrep::new(c).unwrap()), d)),
        false,
    );
    let domain = SectorLeg::new(
        [(0, 5), (1, 2), (2, 3), (3, 4)]
            .map(|(c, d)| (SectorId::from(U1Irrep::new(c).unwrap()), d)),
        false,
    );
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([codomain]),
        FusionProductSpace::new([domain]),
    );
    let space = DynamicFusionMapSpace::from_final_homspace(&U1FusionRule, homspace).unwrap();
    let mut k = 0.0_f64;
    TestTensor::from_block_fn_with_fusion_space(space, D::zero(), |_, _| {
        k += 1.0;
        D::from_complex64(Complex64::new((0.7 * k).sin(), (1.3 * k).cos()))
    })
    .unwrap()
}

fn full_lq_and_right_null_lane_case<D: FactorScalar>() {
    // What: full LQ and the right null basis of several unequal tall, wide
    // and square sectors solved together on a four-thread engine: per sector
    // `L Q = X`, `Q Q^H = I`, `L` lower trapezoidal with a real
    // non-negative diagonal; `X N^H = 0`, `N N^H = I` with `(n - m)+` rows,
    // and no block for a sector without null directions.
    let tol = 1e-12;
    let engine =
        &mut symtn_dense::CpuEngine::new(symtn_dense::Pool::builder().threads(4).build().unwrap());
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let tensor = unequal_sector_tensor::<D>();
    let input = bound_tensor(Arc::new(U1FusionRule), &tensor);
    let input_ref = input.as_ref();
    let input = input_ref.dynamic();
    let x = sector_matrices_c64(tensor.structure(), 1, tensor.data());
    assert_eq!(x.len(), 4);

    let Lq { l, q } = lq_full_dyn(engine, &mut dense, &input).unwrap();
    let l = sector_matrices_c64(l.space().space().structure(), 1, l.data());
    let q = sector_matrices_c64(q.space().space().structure(), 1, q.data());
    assert_eq!((l.len(), q.len()), (x.len(), x.len()));
    for ((sector, m, n, x), ((l_sector, l_rows, l_cols, l), (q_sector, q_rows, q_cols, q))) in
        x.iter().zip(l.iter().zip(&q))
    {
        let (m, n) = (*m, *n);
        assert_eq!((l_sector, *l_rows, *l_cols), (sector, m, n), "L block");
        assert_eq!((q_sector, *q_rows, *q_cols), (sector, n, n), "Q block");
        for i in 0..m {
            for j in 0..n {
                let lq: Complex64 = (0..n).map(|p| l[i + m * p] * q[p + n * j]).sum();
                assert!(
                    (lq - x[i + m * j]).norm() < tol,
                    "sector {sector:?}: L Q != X"
                );
                if j > i {
                    assert!(l[i + m * j].norm() < tol, "sector {sector:?}: L not lower");
                }
            }
            if i < n {
                let d = l[i + m * i];
                assert!(
                    d.re >= 0.0 && d.im.abs() < tol,
                    "sector {sector:?}: L diagonal {d}"
                );
            }
        }
        for i in 0..n {
            for j in 0..n {
                let gram: Complex64 = (0..n).map(|p| q[i + n * p] * q[j + n * p].conj()).sum();
                let one = if i == j { 1.0 } else { 0.0 };
                assert!((gram - one).norm() < tol, "sector {sector:?}: Q Q^H != I");
            }
        }
    }

    let null = right_null_dyn(engine, &mut dense, &input).unwrap();
    let null = sector_matrices_c64(null.space().space().structure(), 1, null.data());
    let wide = x.iter().filter(|(_, m, n, _)| n > m).collect::<Vec<_>>();
    assert_eq!(null.len(), wide.len());
    for ((sector, m, n, x), (null_sector, rows, cols, basis)) in wide.into_iter().zip(&null) {
        let (m, n) = (*m, *n);
        assert_eq!((null_sector, *rows, *cols), (sector, n - m, n));
        for i in 0..m {
            for r in 0..n - m {
                let product: Complex64 = (0..n)
                    .map(|p| x[i + m * p] * basis[r + (n - m) * p].conj())
                    .sum();
                assert!(product.norm() < tol, "sector {sector:?}: X N^H != 0");
            }
        }
        for a in 0..n - m {
            for b in 0..n - m {
                let gram: Complex64 = (0..n)
                    .map(|p| basis[a + (n - m) * p] * basis[b + (n - m) * p].conj())
                    .sum();
                let one = if a == b { 1.0 } else { 0.0 };
                assert!((gram - one).norm() < tol, "sector {sector:?}: N N^H != I");
            }
        }
    }
}

#[test]
fn full_lq_and_right_null_solve_unequal_sectors_on_lanes_real() {
    full_lq_and_right_null_lane_case::<f64>();
}

#[test]
fn full_lq_and_right_null_solve_unequal_sectors_on_lanes_complex() {
    full_lq_and_right_null_lane_case::<Complex64>();
}

/// Records the destination shapes each forwarded `factorize_batch_into`
/// delivers; any per-matrix QR or owned batch is a failure.
#[derive(Default)]
struct DestShapeSpy {
    shapes: Vec<Vec<Vec<usize>>>,
}

impl crate::tests::scripted_executor::Observer for DestShapeSpy {
    fn script(script: &mut crate::tests::scripted_executor::Script) {
        script
            .set(Op::FactorizeBatchInto, Action::Forward)
            .set(Op::FactorizeBatch, Action::Panic("one destination batch"))
            .set(Op::Qr, Action::Panic("one destination batch"))
            .set(Op::QrInto, Action::Panic("one destination batch"));
    }

    fn batch_into_outcome(
        &mut self,
        index: usize,
        dest: &mut symtn_dense::FactorDest<'_>,
        outcome: Result<(), symtn_dense::FactorFailure>,
        sink: &mut crate::tests::scripted_executor::BatchIntoSink<'_>,
    ) -> Result<(), DenseError> {
        let shapes = dest
            .outputs_mut()
            .iter()
            .flatten()
            .map(|output| output.shape().to_vec())
            .collect();
        self.shapes.push(shapes);
        sink(index, dest, outcome)
    }
}

#[test]
fn full_lq_and_right_null_each_submit_one_transposed_destination_batch() {
    // What: each operation solves every eligible sector in one destination
    // batch of the transposed problem (full QR `[n x n, n x m]` per sector;
    // the null basis `[n x (n - m)]` for each wide sector only), with no
    // per-matrix QR.
    let engine = &mut symtn_dense::CpuEngine::serial();
    let tensor = unequal_sector_tensor::<f64>();
    let input = bound_tensor(Arc::new(U1FusionRule), &tensor);
    let input_ref = input.as_ref();
    let input = input_ref.dynamic();
    let sectors = sector_matrices_c64(tensor.structure(), 1, tensor.data());

    let mut dense = ScriptedExecutor::<DestShapeSpy>::default();
    lq_full_dyn(engine, &mut dense, &input).unwrap();
    assert_eq!(dense.counts().factorize_batch_into, 1);
    let expected = sectors
        .iter()
        .map(|&(_, m, n, _)| vec![vec![n, n], vec![n, m]])
        .collect::<Vec<_>>();
    assert_eq!(dense.observer.shapes, expected);

    let mut dense = ScriptedExecutor::<DestShapeSpy>::default();
    right_null_dyn(engine, &mut dense, &input).unwrap();
    assert_eq!(dense.counts().factorize_batch_into, 1);
    let expected = sectors
        .iter()
        .filter(|&&(_, m, n, _)| n > m)
        .map(|&(_, m, n, _)| vec![vec![n, n - m]])
        .collect::<Vec<_>>();
    assert_eq!(expected.len(), 2);
    assert_eq!(dense.observer.shapes, expected);
}
