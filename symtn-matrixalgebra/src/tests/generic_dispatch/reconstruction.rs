use super::*;

fn scale_vt_rows_by_singular_values(vt: &mut TestTensor<f64>, singular_values: &[SectorSpectrum]) {
    let structure = std::sync::Arc::clone(vt.structure());
    for index in 0..structure.block_count() {
        let block = structure.block(index).unwrap();
        let BlockKey::FusionTree(key) = block.key() else {
            continue;
        };
        let sector = key.codomain_tree().coupled();
        let values = &singular_values
            .iter()
            .find(|entry| entry.sector == sector)
            .expect("singular values for every Vt sector")
            .values;
        let shape = block.shape().to_vec();
        let count = shape.iter().product::<usize>();
        let mut multi_index = vec![0usize; shape.len()];
        for _ in 0..count {
            let position = block.offset()
                + multi_index
                    .iter()
                    .zip(block.strides())
                    .map(|(&i, &s)| i * s)
                    .sum::<usize>();
            vt.data_mut()[position] *= values[multi_index[0]];
            for axis in 0..shape.len() {
                multi_index[axis] += 1;
                if multi_index[axis] < shape[axis] {
                    break;
                }
                multi_index[axis] = 0;
            }
        }
    }
}

fn run_tsvd_reconstruction_case<R>(rule: &R, sectors: &[SectorId])
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey + Clone,
{
    let degeneracy = 2usize;
    let leg = || SectorLeg::new(sectors.iter().map(|&sector| (sector, degeneracy)), false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let space = DynamicFusionMapSpace::from_final_homspace(rule, homspace).unwrap();
    let len = space.required_len().unwrap();
    let tensor = TestTensor::<f64>::from_vec_with_fusion_space(
        (0..len)
            .map(|index| ((index * 7 + 3) % 23) as f64 * 0.5 - 5.0)
            .collect(),
        space,
    )
    .unwrap();

    let mut dense_executor = symtn_dense::DefaultDenseExecutor::new();
    let svd = svd_compact(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule.clone()), &tensor),
    )
    .unwrap();
    assert_factor_layout_matches_per_tree_shapes(svd.u.space());
    assert_factor_layout_matches_per_tree_shapes(svd.s.space());
    assert_factor_layout_matches_per_tree_shapes(svd.vh.space());

    for entry in &svd.singular_values {
        for pair in entry.values.windows(2) {
            assert!(
                pair[0] >= pair[1] - 1e-12,
                "singular values must be descending in sector {:?}",
                entry.sector
            );
        }
        assert!(entry.values.iter().all(|&value| value >= -1e-12));
    }

    let mut scaled_vt = svd.vh.tensor().clone();
    scale_vt_rows_by_singular_values(&mut scaled_vt, &svd.singular_values);

    let mut reconstructed = TestTensor::<f64>::from_vec_with_fusion_space(
        vec![0.0; len],
        tensor.fusion_space().clone(),
    )
    .unwrap();
    let mut context = TensorContractFusionExecutionContext::<f64, R::Key>::default();
    context_contract_into(
        &mut context,
        rule,
        &mut reconstructed,
        &svd.u,
        &scaled_vt,
        TensorContractSpec::new(&[2], &[0], OutputAxisOrder::from_axes(&[0, 1, 2, 3])),
        1.0,
        0.0,
    )
    .unwrap();

    assert_svd_blocks_match(&tensor, &reconstructed);
}

#[test]
fn tsvd_fusion_reconstructs_z2_tensor_coupled_layout() {
    run_tsvd_reconstruction_case(&Z2FusionRule, &[SectorId::new(0), SectorId::new(1)]);
}

#[test]
fn tsvd_fusion_reconstructs_su2_tensor() {
    run_tsvd_reconstruction_case(
        &SU2FusionRule,
        &[
            SU2Irrep::from_twice_spin(0).sector_id(),
            SU2Irrep::from_twice_spin(1).sector_id(),
        ],
    );
}

#[test]
fn tsvd_fusion_reconstructs_u1_tensor() {
    run_tsvd_reconstruction_case(
        &U1FusionRule,
        &[
            U1Irrep::new(-1).sector_id(),
            U1Irrep::new(0).sector_id(),
            U1Irrep::new(1).sector_id(),
        ],
    );
}

#[test]
fn tsvd_fusion_reconstructs_fermion_parity_tensor() {
    // What: the canonical direct SVD preserves both fermion-parity sectors.
    run_tsvd_reconstruction_case(
        &FermionParityFusionRule,
        &[SectorId::new(0), SectorId::new(1)],
    );
}

#[test]
fn tsvd_fusion_reconstructs_product_rule_tensor() {
    // What: direct sector spans are keyed by the encoded product SectorId.
    let rule =
        symtn_core::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule)
            .unwrap();
    let sectors = [
        rule.encode_component_ids(SectorId::new(0), U1Irrep::new(0).sector_id()),
        rule.encode_component_ids(SectorId::new(1), U1Irrep::new(1).sector_id()),
    ];
    run_tsvd_reconstruction_case(&rule, &sectors);
}

fn u1_lowest_label_matrix(rows: usize, cols: usize) -> TestTensor<f64> {
    let rule = U1FusionRule;
    let minimum = U1Irrep::new(i32::MIN + 1).sector_id();
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(minimum, rows)], false)]),
        FusionProductSpace::new([SectorLeg::new([(minimum, cols)], false)]),
    );
    let space = DynamicFusionMapSpace::from_final_homspace(&rule, homspace).unwrap();
    TestTensor::from_vec_with_fusion_space(
        (0..rows * cols)
            .map(|index| ((index * 7 + 3) % 17) as f64 - 5.0)
            .collect(),
        space,
    )
    .unwrap()
}

fn assert_matrix_product(
    expected: &[f64],
    rows: usize,
    inner: usize,
    cols: usize,
    left: &[f64],
    right: &[f64],
) {
    for col in 0..cols {
        for row in 0..rows {
            let actual = (0..inner)
                .map(|index| left[row + rows * index] * right[index + inner * col])
                .sum::<f64>();
            assert!(
                (actual - expected[row + rows * col]).abs() < 1e-10,
                "matrix product differs at ({row}, {col}): {actual} != {}",
                expected[row + rows * col]
            );
        }
    }
}

#[test]
fn compact_factors_do_not_relabel_the_lowest_u1_sector() {
    // What: compact SVD, QR, and LQ return correctly oriented factors at the lowest U(1) label.
    let rule = U1FusionRule;
    let tensor = u1_lowest_label_matrix(3, 2);
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let svd = svd_compact(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor)).unwrap();
    assert_eq!(svd.u.tensor().dims(), &[3, 2]);
    assert_eq!(svd.s.tensor().dims(), &[2, 2]);
    assert_eq!(svd.vh.tensor().dims(), &[2, 2]);
    let singular = &svd.singular_values[0].values;
    let mut scaled_vh = svd.vh.data().to_vec();
    for col in 0..2 {
        for row in 0..2 {
            scaled_vh[row + 2 * col] *= singular[row];
        }
    }
    assert_matrix_product(tensor.data(), 3, 2, 2, svd.u.data(), &scaled_vh);

    let Qr { q, r } = qr_compact(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor)).unwrap();
    assert_eq!(q.tensor().dims(), &[3, 2]);
    assert_eq!(r.tensor().dims(), &[2, 2]);
    assert_matrix_product(tensor.data(), 3, 2, 2, q.data(), r.data());

    let Lq { l, q } = lq_compact(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor)).unwrap();
    assert_eq!(l.tensor().dims(), &[3, 2]);
    assert_eq!(q.tensor().dims(), &[2, 2]);
    assert_matrix_product(tensor.data(), 3, 2, 2, l.data(), q.data());
}

#[cfg(target_pointer_width = "64")]
#[test]
fn compact_factorizations_do_not_relabel_product_lowest_u1_sectors() {
    // What: product-sector factors inherit the no-relabel contract for compact SVD, QR, LQ, and full EIGH.
    let rule =
        symtn_core::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule)
            .unwrap();
    let minimum = rule
        .try_encode_component_ids(SectorId::new(1), U1Irrep::new(i32::MIN + 1).sector_id())
        .unwrap();
    let matrix = |rows: usize, cols: usize, data: Vec<f64>| {
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([SectorLeg::new([(minimum, rows)], false)]),
            FusionProductSpace::new([SectorLeg::new([(minimum, cols)], false)]),
        );
        let space = DynamicFusionMapSpace::from_final_homspace(&rule, homspace).unwrap();
        TestTensor::from_vec_with_fusion_space(data, space).unwrap()
    };
    let rectangular = matrix(3, 2, vec![-2.0, 5.0, 1.0, 4.0, -3.0, 2.0]);
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let svd = svd_compact(
        &mut dense,
        &bound_tensor_ref!(Arc::new(rule.clone()), &rectangular),
    )
    .unwrap();
    assert_eq!(svd.u.tensor().dims(), &[3, 2]);
    assert_eq!(svd.s.tensor().dims(), &[2, 2]);
    assert_eq!(svd.vh.tensor().dims(), &[2, 2]);

    let Qr { q, r } = qr_compact(
        &mut dense,
        &bound_tensor_ref!(Arc::new(rule.clone()), &rectangular),
    )
    .unwrap();
    assert_eq!(q.tensor().dims(), &[3, 2]);
    assert_eq!(r.tensor().dims(), &[2, 2]);
    assert_matrix_product(rectangular.data(), 3, 2, 2, q.data(), r.data());

    let Lq { l, q } = lq_compact(
        &mut dense,
        &bound_tensor_ref!(Arc::new(rule.clone()), &rectangular),
    )
    .unwrap();
    assert_eq!(l.tensor().dims(), &[3, 2]);
    assert_eq!(q.tensor().dims(), &[2, 2]);
    assert_matrix_product(rectangular.data(), 3, 2, 2, l.data(), q.data());

    let hermitian = matrix(2, 2, vec![4.0, 1.0, 1.0, 3.0]);
    let result = eigh_full(&mut dense, &bound_tensor_ref!(Arc::new(rule), &hermitian)).unwrap();
    assert_eq!(result.v.tensor().dims(), &[2, 2]);
    assert_eq!(result.d.tensor().dims(), &[2, 2]);
    let values = &result.eigenvalues[0].values;
    assert_eq!(values.len(), 2);
    for (col, &value) in values.iter().enumerate() {
        for row in 0..2 {
            let lhs = (0..2)
                .map(|index| hermitian.data()[row + 2 * index] * result.v.data()[index + 2 * col])
                .sum::<f64>();
            let rhs = result.v.data()[row + 2 * col] * value;
            assert!((lhs - rhs).abs() < 1e-10);
        }
    }
}

#[test]
fn leftorth_fusion_reconstructs_z2_and_su2_tensors() {
    for (rule_case, sectors) in [
        (0usize, vec![SectorId::new(0), SectorId::new(1)]),
        (
            1usize,
            vec![
                SU2Irrep::from_twice_spin(0).sector_id(),
                SU2Irrep::from_twice_spin(1).sector_id(),
            ],
        ),
    ] {
        if rule_case == 0 {
            let rule = Z2FusionRule;
            let tensor = tsvd_test_tensor(&rule, &sectors);
            let mut dense_executor = symtn_dense::DefaultDenseExecutor::new();
            let Qr { q, r } = qr_compact(
                &mut dense_executor,
                &bound_tensor_ref!(Arc::new(rule), &tensor),
            )
            .unwrap();
            let reconstructed = contract_pair(&rule, &tensor, &q, &r);
            assert_svd_blocks_match(&tensor, &reconstructed);
        } else {
            let rule = SU2FusionRule;
            let tensor = tsvd_test_tensor(&rule, &sectors);
            let mut dense_executor = symtn_dense::DefaultDenseExecutor::new();
            let Qr { q, r } = qr_compact(
                &mut dense_executor,
                &bound_tensor_ref!(Arc::new(rule), &tensor),
            )
            .unwrap();
            let reconstructed = contract_pair(&rule, &tensor, &q, &r);
            assert_svd_blocks_match(&tensor, &reconstructed);
        }
    }
}

#[test]
fn rightorth_fusion_reconstructs_z2_and_su2_tensors() {
    {
        let rule = Z2FusionRule;
        let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
        let mut dense_executor = symtn_dense::DefaultDenseExecutor::new();
        let Lq { l, q } = lq_compact(
            &mut dense_executor,
            &bound_tensor_ref!(Arc::new(rule), &tensor),
        )
        .unwrap();
        let reconstructed = contract_pair(&rule, &tensor, &l, &q);
        assert_svd_blocks_match(&tensor, &reconstructed);
    }
    {
        let rule = SU2FusionRule;
        let tensor = tsvd_test_tensor(
            &rule,
            &[
                SU2Irrep::from_twice_spin(0).sector_id(),
                SU2Irrep::from_twice_spin(1).sector_id(),
            ],
        );
        let mut dense_executor = symtn_dense::DefaultDenseExecutor::new();
        let Lq { l, q } = lq_compact(
            &mut dense_executor,
            &bound_tensor_ref!(Arc::new(rule), &tensor),
        )
        .unwrap();
        let reconstructed = contract_pair(&rule, &tensor, &l, &q);
        assert_svd_blocks_match(&tensor, &reconstructed);
    }
}

#[test]
fn tsvd_singular_tensor_composes_u_s_vt() {
    let rule = SU2FusionRule;
    let tensor = tsvd_test_tensor(
        &rule,
        &[
            SU2Irrep::from_twice_spin(0).sector_id(),
            SU2Irrep::from_twice_spin(1).sector_id(),
        ],
    );
    let mut dense_executor = symtn_dense::DefaultDenseExecutor::new();
    let svd = svd_compact(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap();
    let s_tensor = svd.s.clone();

    let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
    let mut u_s = TestTensor::<f64>::from_vec_with_fusion_space(
        vec![0.0; svd.u.data().len()],
        svd.u.fusion_space().clone(),
    )
    .unwrap();
    context_contract_into(
        &mut context,
        &rule,
        &mut u_s,
        &svd.u,
        &s_tensor,
        TensorContractSpec::new(&[2], &[0], OutputAxisOrder::from_axes(&[0, 1, 2])),
        1.0,
        0.0,
    )
    .unwrap();

    let mut reconstructed = TestTensor::<f64>::from_vec_with_fusion_space(
        vec![0.0; tensor.data().len()],
        tensor.fusion_space().clone(),
    )
    .unwrap();
    context_contract_into(
        &mut context,
        &rule,
        &mut reconstructed,
        &u_s,
        &svd.vh,
        TensorContractSpec::new(&[2], &[0], OutputAxisOrder::from_axes(&[0, 1, 2, 3])),
        1.0,
        0.0,
    )
    .unwrap();

    assert_svd_blocks_match(&tensor, &reconstructed);
}

#[test]
fn full_factorizations_agree_with_compact_on_matching_square_support() {
    // What: on square support the full factorizations need no completion and
    // agree with the compact ones (path agreement under the workspace rule,
    // 2 = the block dimension); each builds only its returned buffers.
    let rule = U1FusionRule;
    let neutral = U1Irrep::new(0).sector_id();
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(neutral, 2)], false)]),
        FusionProductSpace::new([SectorLeg::new([(neutral, 2)], false)]),
    );
    let space = DynamicFusionMapSpace::from_final_homspace(&rule, homspace).unwrap();
    let tensor = TestTensor::from_vec_with_fusion_space(vec![-1.0, 3.0, 2.0, 4.0], space).unwrap();
    let input = bound_tensor(Arc::new(rule), &tensor);
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let compact = svd_compact(&mut dense, &input.as_ref()).unwrap();
    let buffer_builds_rec = crate::probe::record();
    let full = svd_full(&mut dense, &input.as_ref()).unwrap();
    assert_eq!(
        buffer_builds_rec.factor_buffer_builds(),
        (1, 1),
        "full SVD must build exactly its returned U and Vh buffers"
    );
    numerics::assert_slices_close("U", full.u.data(), compact.u.data(), 2);
    numerics::assert_slices_close("S", full.s.data(), compact.s.data(), 2);
    numerics::assert_slices_close("Vh", full.vh.data(), compact.vh.data(), 2);

    let Qr {
        q: q_compact,
        r: r_compact,
    } = qr_compact(&mut dense, &input.as_ref()).unwrap();
    let Qr {
        q: q_full,
        r: r_full,
    } = qr_full(&mut dense, &input.as_ref()).unwrap();
    numerics::assert_slices_close("Q", q_full.data(), q_compact.data(), 2);
    numerics::assert_slices_close("R", r_full.data(), r_compact.data(), 2);

    let Lq {
        l: l_compact,
        q: q_compact,
    } = lq_compact(&mut dense, &input.as_ref()).unwrap();
    let Lq {
        l: l_full,
        q: q_full,
    } = lq_full(&mut dense, &input.as_ref()).unwrap();
    numerics::assert_slices_close("L", l_full.data(), l_compact.data(), 2);
    numerics::assert_slices_close("Q", q_full.data(), q_compact.data(), 2);
}

#[test]
fn full_factorizations_skip_dense_backend_for_disjoint_support() {
    let rule = U1FusionRule;
    let positive = U1Irrep::new(1).sector_id();
    let neutral = U1Irrep::new(0).sector_id();
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(positive, 2)], false)]),
        FusionProductSpace::new([SectorLeg::new([(neutral, 3)], false)]),
    );
    let space = DynamicFusionMapSpace::from_final_homspace(&rule, homspace).unwrap();
    let tensor = TestTensor::from_vec_with_fusion_space(Vec::<f64>::new(), space).unwrap();
    let input = bound_tensor(Arc::new(rule), &tensor);

    svd_full(
        &mut ScriptedExecutor::new(RejectExecutorCalls),
        &input.as_ref(),
    )
    .unwrap();
    qr_full(
        &mut ScriptedExecutor::new(RejectExecutorCalls),
        &input.as_ref(),
    )
    .unwrap();
    lq_full(
        &mut ScriptedExecutor::new(RejectExecutorCalls),
        &input.as_ref(),
    )
    .unwrap();
}

fn assert_value_region_paths_match<R, D>(
    rule: Arc<R>,
    general: &TestTensor<D>,
    hermitian: &TestTensor<D>,
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let general_padded = padded_copy(rule.as_ref(), general);
    let hermitian_padded = padded_copy(rule.as_ref(), hermitian);
    let general_bound = bound_tensor(Arc::clone(&rule), general);
    let hermitian_bound = bound_tensor(Arc::clone(&rule), hermitian);
    let general_fallback = bound_tensor(Arc::clone(&rule), &general_padded);
    let hermitian_fallback = bound_tensor(rule, &hermitian_padded);
    assert!(general_bound
        .space()
        .space()
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .is_some());
    assert!(general_fallback
        .space()
        .space()
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .is_none());
    assert!(hermitian_fallback
        .space()
        .space()
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .is_none());
    let general_before = general_bound.data().to_vec();
    let hermitian_before = hermitian_bound.data().to_vec();
    let general_fallback_before = general_fallback.data().to_vec();
    let hermitian_fallback_before = hermitian_fallback.data().to_vec();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let values_fallbacks_rec = crate::probe::record();
    let direct_svd = svd_vals_dyn(&mut dense, &general_bound.as_ref().dynamic()).unwrap();
    let direct_eigh = eigh_vals_dyn(
        &mut dense,
        &hermitian_bound.as_ref().dynamic(),
        HermitianTol::DEFAULT,
    )
    .unwrap();
    let direct_eig = eig_vals_dyn(&mut dense, &general_bound.as_ref().dynamic()).unwrap();
    assert_eq!(
        values_fallbacks_rec.get(crate::probe::Event::ValuesMatricizationFallback),
        0
    );

    let values_fallbacks_rec = crate::probe::record();
    let packed_svd = svd_vals_dyn(&mut dense, &general_fallback.as_ref().dynamic()).unwrap();
    let packed_eigh = eigh_vals_dyn(
        &mut dense,
        &hermitian_fallback.as_ref().dynamic(),
        HermitianTol::DEFAULT,
    )
    .unwrap();
    let packed_eig = eig_vals_dyn(&mut dense, &general_fallback.as_ref().dynamic()).unwrap();
    assert_eq!(
        values_fallbacks_rec.get(crate::probe::Event::ValuesMatricizationFallback),
        3
    );

    assert_real_spectra_close(&direct_svd, &packed_svd);
    assert_real_spectra_close(&direct_eigh, &packed_eigh);
    assert_complex_spectra_close(&direct_eig, &packed_eig);
    assert!(general_bound.data() == general_before);
    assert!(hermitian_bound.data() == hermitian_before);
    assert!(general_fallback.data() == general_fallback_before);
    assert!(hermitian_fallback.data() == hermitian_fallback_before);
}

#[test]
fn value_region_paths_match_packed_oracles_across_supported_rules() {
    // What: canonical region borrowing and noncanonical packing return the same
    // ordered spectra for Abelian, non-Abelian, fermionic, and product rules.
    let u1 = [
        U1Irrep::new(-1).sector_id(),
        U1Irrep::new(0).sector_id(),
        U1Irrep::new(1).sector_id(),
    ];
    assert_value_region_paths_match(
        Arc::new(U1FusionRule),
        &tsvd_test_tensor(&U1FusionRule, &u1),
        &hermitian_test_tensor(&U1FusionRule, &u1),
    );

    let su2 = [
        SU2Irrep::from_twice_spin(0).sector_id(),
        SU2Irrep::from_twice_spin(1).sector_id(),
    ];
    assert_value_region_paths_match(
        Arc::new(SU2FusionRule),
        &tsvd_test_tensor(&SU2FusionRule, &su2),
        &hermitian_test_tensor(&SU2FusionRule, &su2),
    );

    let fz2 = [SectorId::new(0), SectorId::new(1)];
    assert_value_region_paths_match(
        Arc::new(FermionParityFusionRule),
        &tsvd_test_tensor(&FermionParityFusionRule, &fz2),
        &hermitian_test_tensor(&FermionParityFusionRule, &fz2),
    );

    let product =
        symtn_core::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule)
            .unwrap();
    let product_sectors = [
        product.encode_component_ids(SectorId::new(0), U1Irrep::new(0).sector_id()),
        product.encode_component_ids(SectorId::new(1), U1Irrep::new(1).sector_id()),
    ];
    assert_value_region_paths_match(
        Arc::new(product.clone()),
        &tsvd_test_tensor(&product, &product_sectors),
        &hermitian_test_tensor(&product, &product_sectors),
    );
}

#[test]
fn value_region_paths_match_packed_oracles_for_complex64() {
    // What: borrowed C64 spans preserve nonreal general matrices and conjugate
    // off-diagonal Hermitian matrices across all three values-only operations.
    let rule = Z2FusionRule;
    let sectors = [SectorId::new(0), SectorId::new(1)];
    let general = tsvd_test_tensor(&rule, &sectors);
    let hermitian = hermitian_test_tensor(&rule, &sectors);
    let general = TestTensor::<Complex64>::from_vec_with_fusion_space(
        general
            .data()
            .iter()
            .enumerate()
            .map(|(index, &value)| Complex64::new(value, (index % 7) as f64 * 0.125 - 0.25))
            .collect(),
        general.fusion_space().clone(),
    )
    .unwrap();
    let regions = hermitian
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .unwrap();
    let mut hermitian_data = hermitian
        .data()
        .iter()
        .map(|&value| Complex64::new(value, 0.0))
        .collect::<Vec<_>>();
    for region in regions.iter().filter(|region| region.rows() >= 2) {
        let start = region.range().start;
        hermitian_data[start + 1].im = -0.75;
        hermitian_data[start + region.rows()].im = 0.75;
    }
    let hermitian = TestTensor::<Complex64>::from_vec_with_fusion_space(
        hermitian_data,
        hermitian.fusion_space().clone(),
    )
    .unwrap();

    assert_value_region_paths_match(Arc::new(rule), &general, &hermitian);
}

fn lowered_z2_binding(tensor: &TestTensor<f64>) -> BoundDynamicFusionMapSpace<Z2FusionRule> {
    let provider = Arc::new(Z2FusionRule);
    let raw = tensor.fusion_space().clone();
    let hom = raw.homspace().clone();
    // Why not caller-supplied per-tree shapes: the #586 sweep narrowed the
    // shape-admission bridge to symtn-tensors; the kept public installer
    // derives the identical blocks from the final homspace's leg
    // degeneracies for these dense-leg fixtures.
    BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(provider, hom)
        .unwrap()
}

#[test]
fn ordinary_factorizations_and_composition_inherit_lowered_layout_strategy() {
    // What: cold compact SVD, compact QR, full EIGH, adjoint, and factor
    // composition all retain the ordinary built-in layout-build strategy.
    let tensor = hermitian_test_tensor(&Z2FusionRule, &[SectorId::new(0), SectorId::new(1)]);
    let bound = lowered_z2_binding(&tensor);
    let expert = BoundDynamicFusionMapSpace::bind_multiplicity_free(
        tensor.fusion_space().clone(),
        Arc::new(Z2FusionRule),
    )
    .unwrap();
    let malformed = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(SectorId::new(99), 1)], false)]),
        FusionProductSpace::new([]),
    );
    assert!(expert.prime_derived_homspace(&malformed).is_ok());
    let input = BoundDynamicTensorRef::try_new(&bound, tensor.data()).unwrap();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let rejects_invalid_sector = |result| {
        assert!(matches!(result, Err(OperationError::FusionAlgebra(e))
            if matches!(*e, symtn_core::FusionAlgebraError::InvalidSector { sector }
                if sector == SectorId::new(99))));
    };
    let svd = svd_compact_dyn(&mut dense, &input).unwrap();
    for factor in [svd.u(), svd.s(), svd.vh()] {
        rejects_invalid_sector(factor.space().prime_derived_homspace(&malformed));
    }
    let Qr { q, r } = qr_compact_dyn(&mut dense, &input).unwrap();
    rejects_invalid_sector(q.space().prime_derived_homspace(&malformed));
    rejects_invalid_sector(r.space().prime_derived_homspace(&malformed));
    let eigh = eigh_full_dyn(&mut dense, &input, HermitianTol::DEFAULT).unwrap();
    rejects_invalid_sector(eigh.v().space().prime_derived_homspace(&malformed));

    let adjoint = crate::factorize::adjoint_bound_factor(svd.u()).unwrap();
    rejects_invalid_sector(adjoint.space().prime_derived_homspace(&malformed));
    let mut context = default_context();
    let composed = crate::compose::compose_bound_dyn(&mut context, svd.u(), svd.s()).unwrap();
    rejects_invalid_sector(composed.space().prime_derived_homspace(&malformed));
}

#[test]
fn every_factorization_and_matrix_function_retains_each_callers_exact_provider_arc() {
    // What: each owned output keeps the caller's provider allocation, not an
    // equal provider of the same type seen by an earlier call.
    let hermitian = hermitian_test_tensor(&Z2FusionRule, &[SectorId::new(0), SectorId::new(1)]);
    let general = tsvd_test_tensor(&Z2FusionRule, &[SectorId::new(0), SectorId::new(1)]);
    let rectangular = rectangular_svd_tensor(7, 5);
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let mut output_providers = |provider: &Arc<Z2FusionRule>| {
        let hermitian = bound_tensor(Arc::clone(provider), &hermitian);
        let general = bound_tensor(Arc::clone(provider), &general);
        let rectangular = bound_tensor(Arc::clone(provider), &rectangular);
        let (hermitian, general, rectangular) =
            (hermitian.as_ref(), general.as_ref(), rectangular.as_ref());
        let (hermitian_dyn, general_dyn) = (hermitian.dynamic(), general.dynamic());
        let eigh = eigh_full(&mut dense, &hermitian).unwrap();
        let eigh_dyn = eigh_full_dyn(&mut dense, &hermitian_dyn, HermitianTol::DEFAULT).unwrap();
        let eig = eig_full_dyn(&mut dense, &general_dyn).unwrap();
        let exponential = exp_into_mf(&mut dense, &hermitian_dyn).unwrap();
        let Qr { q: qr_q, r: qr_r } = qr_compact(&mut dense, &rectangular).unwrap();
        let Lq { l: lq_l, q: lq_q } = lq_compact(&mut dense, &rectangular).unwrap();
        let LeftPolar {
            w: w_left,
            p: p_left,
        } = left_polar_dyn(&mut dense, &hermitian_dyn).unwrap();
        let RightPolar {
            p: p_right,
            wh: w_right,
        } = right_polar_dyn(&mut dense, &hermitian_dyn).unwrap();
        let inverse = inv_into_mf(&mut dense, &hermitian_dyn).unwrap();
        let pseudo_inverse = pinv_into_mf(&mut dense, &hermitian_dyn, 1.0e-13).unwrap();
        [
            ("eigh", eigh.v.space().provider_arc()),
            ("eigh_dyn", eigh_dyn.v().space().provider_arc()),
            ("eig", eig.v().space().provider_arc()),
            ("exp", exponential.space().provider_arc()),
            ("qr.q", qr_q.space().provider_arc()),
            ("qr.r", qr_r.space().provider_arc()),
            ("lq.l", lq_l.space().provider_arc()),
            ("lq.q", lq_q.space().provider_arc()),
            ("left_polar.w", w_left.space().provider_arc()),
            ("left_polar.p", p_left.space().provider_arc()),
            ("right_polar.p", p_right.space().provider_arc()),
            ("right_polar.wh", w_right.space().provider_arc()),
            ("inv", inverse.space().provider_arc()),
            ("pinv", pseudo_inverse.space().provider_arc()),
        ]
        .map(|(op, arc)| (op, Arc::clone(arc)))
    };

    let first_provider = Arc::new(Z2FusionRule);
    let second_provider = Arc::new(Z2FusionRule);
    for provider in [&first_provider, &second_provider] {
        for (op, output) in output_providers(provider) {
            assert!(Arc::ptr_eq(&output, provider), "{op}");
        }
    }
}

#[test]
fn adjoint_composition_gives_the_identity_on_the_bond() {
    let rule = SU2FusionRule;
    let tensor = tsvd_test_tensor(
        &rule,
        &[
            SU2Irrep::from_twice_spin(0).sector_id(),
            SU2Irrep::from_twice_spin(1).sector_id(),
        ],
    );
    let mut dense_executor = symtn_dense::DefaultDenseExecutor::new();
    let Qr { q, .. } = qr_compact(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap();
    let qh = adjoint(&rule, &q).unwrap();
    let mut context = default_context();
    let identity = crate::compose::compose(&mut context, &rule, &qh, &q).unwrap();
    assert_identity_matrices(&dense_sector_matrices(1, &identity));
}

/// Returns a batch that violates the `factorize_batch` contract: one input's
/// entry missing, or one entry missing a factor.
struct MalformedBatch {
    drop_entry: bool,
}

impl Observer for MalformedBatch {
    fn script(script: &mut Script) {
        script.set(Op::FactorizeBatch, Action::Forward).set(
            Op::DotGeneral,
            Action::Panic("test only exercises factorizations"),
        );
    }

    fn batch_outputs(&mut self, outputs: &mut Vec<Vec<DenseTensor>>) {
        if self.drop_entry {
            outputs.pop();
        } else {
            outputs[0].pop();
        }
    }
}

#[test]
fn compact_factorizations_reject_a_malformed_executor_batch() {
    // What: a batch with a missing input entry, or an entry missing a factor,
    // is symtn's own typed arity error, never a dropped sector, a panic or a
    // backend error. A missing factor keeps the per-matrix path's error.
    let rule = Z2FusionRule;
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let bound = bound_tensor(Arc::new(rule), &tensor);
    let input = bound.as_ref();
    for (drop_entry, qr_op, svd_op) in [
        (true, "factorize_batch", "factorize_batch"),
        (false, "qr_into", "svd_into"),
    ] {
        let mut dense = ScriptedExecutor::new(MalformedBatch { drop_entry });
        let is_arity_error = |error: &OperationError, expected: &str| {
            matches!(
                error,
                OperationError::Dense(DenseError::ShapeMismatch { op, .. }) if *op == expected
            )
        };
        let qr = qr_compact(&mut dense, &input).map(|_| ()).unwrap_err();
        assert!(is_arity_error(&qr, qr_op), "qr_compact: {qr:?}");
        let svd = svd_compact(&mut dense, &input).map(|_| ()).unwrap_err();
        assert!(is_arity_error(&svd, svd_op), "svd_compact: {svd:?}");
    }
}
