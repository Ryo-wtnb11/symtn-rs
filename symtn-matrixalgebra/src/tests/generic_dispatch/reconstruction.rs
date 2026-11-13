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
    let engine = &mut symtn_dense::CpuEngine::serial();
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
        engine,
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
    run_tsvd_reconstruction_case(
        &ZNFusionRule::new(2).unwrap(),
        &[SectorId::new(0), SectorId::new(1)],
    );
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
            SectorId::from(U1Irrep::new(-1).unwrap()),
            SectorId::from(U1Irrep::new(0).unwrap()),
            SectorId::from(U1Irrep::new(1).unwrap()),
        ],
    );
}

#[test]
fn tsvd_fusion_reconstructs_fermion_parity_tensor() {
    // What: the canonical direct SVD preserves both fermion-parity sectors.
    run_tsvd_reconstruction_case(
        &FermionParityFusionRule::new(),
        &[SectorId::new(0), SectorId::new(1)],
    );
}

#[test]
fn tsvd_fusion_reconstructs_product_rule_tensor() {
    // What: direct sector spans are keyed by the encoded product SectorId.
    let rule = symtn_core::ProductFusionRule::<_, _>::try_new(
        FermionParityFusionRule::new(),
        U1FusionRule,
    )
    .unwrap();
    let sectors = [
        rule.encode_component_ids(SectorId::new(0), SectorId::from(U1Irrep::new(0).unwrap())),
        rule.encode_component_ids(SectorId::new(1), SectorId::from(U1Irrep::new(1).unwrap())),
    ];
    run_tsvd_reconstruction_case(&rule, &sectors);
}

fn u1_lowest_label_matrix(rows: usize, cols: usize) -> TestTensor<f64> {
    let rule = U1FusionRule;
    let minimum = SectorId::from(U1Irrep::new(i32::MIN + 1).unwrap());
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
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: compact SVD, QR, and LQ return correctly oriented factors at the lowest U(1) label.
    let rule = U1FusionRule;
    let tensor = u1_lowest_label_matrix(3, 2);
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let svd = svd_compact(
        engine,
        &mut dense,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap();
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

    let Qr { q, r } = qr_compact(
        engine,
        &mut dense,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap();
    assert_eq!(q.tensor().dims(), &[3, 2]);
    assert_eq!(r.tensor().dims(), &[2, 2]);
    assert_matrix_product(tensor.data(), 3, 2, 2, q.data(), r.data());

    let Lq { l, q } = lq_compact(
        engine,
        &mut dense,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap();
    assert_eq!(l.tensor().dims(), &[3, 2]);
    assert_eq!(q.tensor().dims(), &[2, 2]);
    assert_matrix_product(tensor.data(), 3, 2, 2, l.data(), q.data());
}

#[cfg(target_pointer_width = "64")]
#[test]
fn compact_factorizations_do_not_relabel_product_lowest_u1_sectors() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: product-sector factors inherit the no-relabel contract for compact SVD, QR, LQ, and full EIGH.
    let rule = symtn_core::ProductFusionRule::<_, _>::try_new(
        FermionParityFusionRule::new(),
        U1FusionRule,
    )
    .unwrap();
    let minimum = rule
        .try_encode_component_ids(
            SectorId::new(1),
            SectorId::from(U1Irrep::new(i32::MIN + 1).unwrap()),
        )
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
        engine,
        &mut dense,
        &bound_tensor_ref!(Arc::new(rule.clone()), &rectangular),
    )
    .unwrap();
    assert_eq!(svd.u.tensor().dims(), &[3, 2]);
    assert_eq!(svd.s.tensor().dims(), &[2, 2]);
    assert_eq!(svd.vh.tensor().dims(), &[2, 2]);

    let Qr { q, r } = qr_compact(
        engine,
        &mut dense,
        &bound_tensor_ref!(Arc::new(rule.clone()), &rectangular),
    )
    .unwrap();
    assert_eq!(q.tensor().dims(), &[3, 2]);
    assert_eq!(r.tensor().dims(), &[2, 2]);
    assert_matrix_product(rectangular.data(), 3, 2, 2, q.data(), r.data());

    let Lq { l, q } = lq_compact(
        engine,
        &mut dense,
        &bound_tensor_ref!(Arc::new(rule.clone()), &rectangular),
    )
    .unwrap();
    assert_eq!(l.tensor().dims(), &[3, 2]);
    assert_eq!(q.tensor().dims(), &[2, 2]);
    assert_matrix_product(rectangular.data(), 3, 2, 2, l.data(), q.data());

    let hermitian = matrix(2, 2, vec![4.0, 1.0, 1.0, 3.0]);
    let result = eigh_full(
        engine,
        &mut dense,
        &bound_tensor_ref!(Arc::new(rule), &hermitian),
    )
    .unwrap();
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
    let engine = &mut symtn_dense::CpuEngine::serial();
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
            let rule = ZNFusionRule::new(2).unwrap();
            let tensor = tsvd_test_tensor(&rule, &sectors);
            let mut dense_executor = symtn_dense::DefaultDenseExecutor::new();
            let Qr { q, r } = qr_compact(
                engine,
                &mut dense_executor,
                &bound_tensor_ref!(Arc::new(rule.clone()), &tensor),
            )
            .unwrap();
            let reconstructed = contract_pair(&rule, &tensor, &q, &r);
            assert_svd_blocks_match(&tensor, &reconstructed);
        } else {
            let rule = SU2FusionRule;
            let tensor = tsvd_test_tensor(&rule, &sectors);
            let mut dense_executor = symtn_dense::DefaultDenseExecutor::new();
            let Qr { q, r } = qr_compact(
                engine,
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
    let engine = &mut symtn_dense::CpuEngine::serial();
    {
        let rule = ZNFusionRule::new(2).unwrap();
        let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
        let mut dense_executor = symtn_dense::DefaultDenseExecutor::new();
        let Lq { l, q } = lq_compact(
            engine,
            &mut dense_executor,
            &bound_tensor_ref!(Arc::new(rule.clone()), &tensor),
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
            engine,
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
    let svd = svd_compact(
        engine,
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
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: on square support the full factorizations need no completion and
    // agree with the compact ones (path agreement under the workspace rule,
    // 2 = the block dimension); each builds only its returned buffers.
    let rule = U1FusionRule;
    let neutral = SectorId::from(U1Irrep::new(0).unwrap());
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(neutral, 2)], false)]),
        FusionProductSpace::new([SectorLeg::new([(neutral, 2)], false)]),
    );
    let space = DynamicFusionMapSpace::from_final_homspace(&rule, homspace).unwrap();
    let tensor = TestTensor::from_vec_with_fusion_space(vec![-1.0, 3.0, 2.0, 4.0], space).unwrap();
    let input = bound_tensor(Arc::new(rule), &tensor);
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let compact = svd_compact(engine, &mut dense, &input.as_ref()).unwrap();
    let buffer_builds_rec = crate::probe::record();
    let full = svd_full(engine, &mut dense, &input.as_ref()).unwrap();
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
    } = qr_compact(engine, &mut dense, &input.as_ref()).unwrap();
    let Qr {
        q: q_full,
        r: r_full,
    } = qr_full(engine, &mut dense, &input.as_ref()).unwrap();
    numerics::assert_slices_close("Q", q_full.data(), q_compact.data(), 2);
    numerics::assert_slices_close("R", r_full.data(), r_compact.data(), 2);

    let Lq {
        l: l_compact,
        q: q_compact,
    } = lq_compact(engine, &mut dense, &input.as_ref()).unwrap();
    let Lq {
        l: l_full,
        q: q_full,
    } = lq_full(engine, &mut dense, &input.as_ref()).unwrap();
    numerics::assert_slices_close("L", l_full.data(), l_compact.data(), 2);
    numerics::assert_slices_close("Q", q_full.data(), q_compact.data(), 2);
}

#[test]
fn full_factorizations_skip_dense_backend_for_disjoint_support() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rule = U1FusionRule;
    let positive = SectorId::from(U1Irrep::new(1).unwrap());
    let neutral = SectorId::from(U1Irrep::new(0).unwrap());
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(positive, 2)], false)]),
        FusionProductSpace::new([SectorLeg::new([(neutral, 3)], false)]),
    );
    let space = DynamicFusionMapSpace::from_final_homspace(&rule, homspace).unwrap();
    let tensor = TestTensor::from_vec_with_fusion_space(Vec::<f64>::new(), space).unwrap();
    let input = bound_tensor(Arc::new(rule), &tensor);

    svd_full(
        engine,
        &mut ScriptedExecutor::new(RejectExecutorCalls),
        &input.as_ref(),
    )
    .unwrap();
    qr_full(
        engine,
        &mut ScriptedExecutor::new(RejectExecutorCalls),
        &input.as_ref(),
    )
    .unwrap();
    lq_full(
        engine,
        &mut ScriptedExecutor::new(RejectExecutorCalls),
        &input.as_ref(),
    )
    .unwrap();
}

/// Values of the eager transposes of `general` and `hermitian` (their
/// adjoints, conjugated back) on the borrowed region path against the
/// adjoint views of the originals on the packed path: a zero-copy view swaps
/// axes without conjugating, so both paths read the same matrices.
fn assert_value_region_paths_match<R, D>(
    rule: Arc<R>,
    general: &TestTensor<D>,
    hermitian: &TestTensor<D>,
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + Clone,
    D: FactorScalar + symtn_tensors::ConjugateValue,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let transpose = |tensor: &TestTensor<D>| {
        let adjoint = adjoint(rule.as_ref(), tensor).unwrap();
        TestTensor::from_vec_with_fusion_space(
            adjoint
                .data()
                .iter()
                .map(|&value| value.maybe_conj(true))
                .collect(),
            adjoint.fusion_space().clone(),
        )
        .unwrap()
    };
    let general_bound = bound_tensor(Arc::clone(&rule), &transpose(general));
    let hermitian_bound = bound_tensor(Arc::clone(&rule), &transpose(hermitian));
    let general_parent = bound_tensor(Arc::clone(&rule), general);
    let hermitian_parent = bound_tensor(rule, hermitian);
    let general_view_space = general_parent.space().adjoint_view().unwrap();
    let hermitian_view_space = hermitian_parent.space().adjoint_view().unwrap();
    let general_view =
        BoundDynamicTensorRef::try_new(&general_view_space, general_parent.data()).unwrap();
    let hermitian_view =
        BoundDynamicTensorRef::try_new(&hermitian_view_space, hermitian_parent.data()).unwrap();
    assert!(general_bound
        .space()
        .space()
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .is_some());
    let general_before = general_parent.data().to_vec();
    let hermitian_before = hermitian_parent.data().to_vec();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let values_fallbacks_rec = crate::probe::record();
    let direct_svd = svd_vals_dyn(engine, &mut dense, &general_bound.as_ref().dynamic()).unwrap();
    let direct_eigh = eigh_vals_dyn(
        engine,
        &mut dense,
        &hermitian_bound.as_ref().dynamic(),
        HermitianTol::DEFAULT,
    )
    .unwrap();
    let direct_eig = eig_vals_dyn(engine, &mut dense, &general_bound.as_ref().dynamic()).unwrap();
    assert_eq!(
        values_fallbacks_rec.get(crate::probe::Event::ValuesMatricizationFallback),
        0
    );

    let values_fallbacks_rec = crate::probe::record();
    let packed_svd = svd_vals_dyn(engine, &mut dense, &general_view).unwrap();
    let packed_eigh =
        eigh_vals_dyn(engine, &mut dense, &hermitian_view, HermitianTol::DEFAULT).unwrap();
    let packed_eig = eig_vals_dyn(engine, &mut dense, &general_view).unwrap();
    assert_eq!(
        values_fallbacks_rec.get(crate::probe::Event::ValuesMatricizationFallback),
        3
    );

    assert_real_spectra_close(&direct_svd, &packed_svd);
    assert_real_spectra_close(&direct_eigh, &packed_eigh);
    assert_complex_spectra_close(&direct_eig, &packed_eig);
    assert!(general_parent.data() == general_before);
    assert!(hermitian_parent.data() == hermitian_before);
}

#[test]
fn value_region_paths_match_packed_oracles_across_supported_rules() {
    // What: canonical region borrowing and noncanonical (adjoint-view)
    // packing agree for Abelian, non-Abelian, fermionic, and product rules.
    let u1 = [
        SectorId::from(U1Irrep::new(-1).unwrap()),
        SectorId::from(U1Irrep::new(0).unwrap()),
        SectorId::from(U1Irrep::new(1).unwrap()),
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
        Arc::new(FermionParityFusionRule::new()),
        &tsvd_test_tensor(&FermionParityFusionRule::new(), &fz2),
        &hermitian_test_tensor(&FermionParityFusionRule::new(), &fz2),
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
    let rule = ZNFusionRule::new(2).unwrap();
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

fn lowered_z2_binding(tensor: &TestTensor<f64>) -> BoundDynamicFusionMapSpace<ZNFusionRule> {
    let provider = Arc::new(ZNFusionRule::new(2).unwrap());
    let raw = tensor.fusion_space().clone();
    let hom = raw.homspace().clone();
    // Why not caller-supplied per-tree shapes: the shape-admission bridge is
    // narrowed to symtn-tensors; the kept public installer
    // derives the identical blocks from the final homspace's leg
    // degeneracies for these dense-leg fixtures.
    BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(provider, hom)
        .unwrap()
}

#[test]
fn ordinary_factorizations_and_composition_inherit_lowered_layout_strategy() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: cold compact SVD, compact QR, full EIGH, adjoint, and factor
    // composition all retain the ordinary built-in layout-build strategy.
    let tensor = hermitian_test_tensor(
        &ZNFusionRule::new(2).unwrap(),
        &[SectorId::new(0), SectorId::new(1)],
    );
    let bound = lowered_z2_binding(&tensor);
    let expert = BoundDynamicFusionMapSpace::bind_multiplicity_free(
        tensor.fusion_space().clone(),
        Arc::new(ZNFusionRule::new(2).unwrap()),
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
    let svd = svd_compact_dyn(engine, &mut dense, &input).unwrap();
    for factor in [svd.u(), svd.s(), svd.vh()] {
        rejects_invalid_sector(factor.space().prime_derived_homspace(&malformed));
    }
    let Qr { q, r } = qr_compact_dyn(engine, &mut dense, &input).unwrap();
    rejects_invalid_sector(q.space().prime_derived_homspace(&malformed));
    rejects_invalid_sector(r.space().prime_derived_homspace(&malformed));
    let eigh = eigh_full_dyn(engine, &mut dense, &input, HermitianTol::DEFAULT).unwrap();
    rejects_invalid_sector(eigh.v().space().prime_derived_homspace(&malformed));

    let adjoint = crate::factorize::adjoint_bound_factor(svd.u()).unwrap();
    rejects_invalid_sector(adjoint.space().prime_derived_homspace(&malformed));
    let mut context = default_context();
    let composed =
        crate::compose::compose_bound_dyn(engine, &mut context, svd.u(), svd.s()).unwrap();
    rejects_invalid_sector(composed.space().prime_derived_homspace(&malformed));
}

#[test]
fn every_factorization_and_matrix_function_retains_each_callers_exact_provider_arc() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: each owned output keeps the caller's provider allocation, not an
    // equal provider of the same type seen by an earlier call.
    let hermitian = hermitian_test_tensor(
        &ZNFusionRule::new(2).unwrap(),
        &[SectorId::new(0), SectorId::new(1)],
    );
    let general = tsvd_test_tensor(
        &ZNFusionRule::new(2).unwrap(),
        &[SectorId::new(0), SectorId::new(1)],
    );
    let rectangular = rectangular_svd_tensor(7, 5);
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let mut output_providers = |provider: &Arc<ZNFusionRule>| {
        let hermitian = bound_tensor(Arc::clone(provider), &hermitian);
        let general = bound_tensor(Arc::clone(provider), &general);
        let rectangular = bound_tensor(Arc::clone(provider), &rectangular);
        let (hermitian, general, rectangular) =
            (hermitian.as_ref(), general.as_ref(), rectangular.as_ref());
        let (hermitian_dyn, general_dyn) = (hermitian.dynamic(), general.dynamic());
        let eigh = eigh_full(engine, &mut dense, &hermitian).unwrap();
        let eigh_dyn =
            eigh_full_dyn(engine, &mut dense, &hermitian_dyn, HermitianTol::DEFAULT).unwrap();
        let eig = eig_full_dyn(engine, &mut dense, &general_dyn).unwrap();
        let exponential = exp_into_mf(engine, &mut dense, &hermitian_dyn).unwrap();
        let Qr { q: qr_q, r: qr_r } = qr_compact(engine, &mut dense, &rectangular).unwrap();
        let Lq { l: lq_l, q: lq_q } = lq_compact(engine, &mut dense, &rectangular).unwrap();
        let LeftPolar {
            w: w_left,
            p: p_left,
        } = left_polar_dyn(engine, &mut dense, &hermitian_dyn).unwrap();
        let RightPolar {
            p: p_right,
            wh: w_right,
        } = right_polar_dyn(engine, &mut dense, &hermitian_dyn).unwrap();
        let inverse = inv_into_mf(engine, &mut dense, &hermitian_dyn).unwrap();
        let pseudo_inverse = pinv_into_mf(engine, &mut dense, &hermitian_dyn, 1.0e-13).unwrap();
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

    let first_provider = Arc::new(ZNFusionRule::new(2).unwrap());
    let second_provider = Arc::new(ZNFusionRule::new(2).unwrap());
    for provider in [&first_provider, &second_provider] {
        for (op, output) in output_providers(provider) {
            assert!(Arc::ptr_eq(&output, provider), "{op}");
        }
    }
}

#[test]
fn adjoint_composition_gives_the_identity_on_the_bond() {
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
    let Qr { q, .. } = qr_compact(
        engine,
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap();
    let qh = adjoint(&rule, &q).unwrap();
    let mut context = default_context();
    let identity = crate::compose::compose(engine, &mut context, &rule, &qh, &q).unwrap();
    assert_identity_matrices(&dense_sector_matrices(1, &identity));
}

/// How a forwarded two-input batch reaches the caller.
#[derive(Clone, Copy, Debug)]
enum Delivery {
    /// The second input is never delivered.
    DropLast,
    /// The first input's outputs miss a factor.
    DropFactor,
    /// The first input fails in the solver; delivery stops there.
    FailFirst,
    /// The first input's outputs miss a factor, then the second fails.
    DropFactorThenFail,
    /// The first input is delivered twice.
    Duplicate,
    /// The first input is delivered under the second's index.
    OutOfOrder,
}

struct MalformedBatch {
    delivery: Delivery,
}

fn solver_failure() -> DenseError {
    DenseError::Backend {
        backend: DenseBackend::Tenferro,
        op: "injected solver",
        message: "injected failure".to_string(),
    }
}

impl Observer for MalformedBatch {
    fn script(script: &mut Script) {
        script.set(Op::FactorizeBatch, Action::Forward).set(
            Op::DotGeneral,
            Action::Panic("test only exercises factorizations"),
        );
    }

    fn batch_outcome(
        &mut self,
        index: usize,
        mut outcome: Result<Vec<DenseTensor>, DenseError>,
        sink: &mut BatchSink<'_>,
    ) -> Result<(), DenseError> {
        use Delivery::*;
        match (self.delivery, index) {
            (DropLast, 1) => Ok(()),
            (DropFactor | DropFactorThenFail, 0) => {
                outcome.as_mut().unwrap().pop();
                sink(index, outcome)
            }
            (FailFirst, 0) | (DropFactorThenFail, 1) => sink(index, Err(solver_failure())),
            (Duplicate, 0) => {
                sink(index, outcome)?;
                sink(index, Err(solver_failure()))
            }
            (OutOfOrder, 0) => sink(1, outcome),
            _ => sink(index, outcome),
        }
    }
}

#[test]
fn compact_factorizations_reject_a_malformed_executor_batch() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: the planned compact QR/LQ/SVD batch is solver-first: a delivered
    // solver error is the result even after an earlier malformed factor,
    // which is checked only once the batch succeeded. A missing,
    // repeated or misplaced delivery is symtn's own typed arity error,
    // never a dropped sector, a panic or a backend error; a missing factor
    // keeps the per-matrix path's error.
    let rule = ZNFusionRule::new(2).unwrap();
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let bound = bound_tensor(Arc::new(rule), &tensor);
    let input = bound.as_ref();
    let arity = |op| {
        move |error: &OperationError| {
            matches!(
                error,
                OperationError::Dense(DenseError::ShapeMismatch { op: actual, .. }) if *actual == op
            )
        }
    };
    let solver = |error: &OperationError| *error == OperationError::Dense(solver_failure());
    type Expect = Box<dyn Fn(&OperationError) -> bool>;
    let cases: [(Delivery, Expect, Expect, Expect); 6] = [
        (
            Delivery::DropLast,
            Box::new(arity("factorize_batch")),
            Box::new(arity("factorize_batch")),
            Box::new(arity("factorize_batch")),
        ),
        (
            Delivery::DropFactor,
            Box::new(arity("qr_into")),
            Box::new(arity("lq")),
            Box::new(arity("svd_into")),
        ),
        (
            Delivery::FailFirst,
            Box::new(solver),
            Box::new(solver),
            Box::new(solver),
        ),
        (
            Delivery::DropFactorThenFail,
            Box::new(solver),
            Box::new(solver),
            Box::new(solver),
        ),
        (
            Delivery::Duplicate,
            Box::new(arity("factorize_batch")),
            Box::new(arity("factorize_batch")),
            Box::new(arity("factorize_batch")),
        ),
        (
            Delivery::OutOfOrder,
            Box::new(arity("factorize_batch")),
            Box::new(arity("factorize_batch")),
            Box::new(arity("factorize_batch")),
        ),
    ];
    for (delivery, qr_expect, lq_expect, svd_expect) in cases {
        let mut dense = ScriptedExecutor::new(MalformedBatch { delivery });
        let qr = qr_compact(engine, &mut dense, &input)
            .map(|_| ())
            .unwrap_err();
        assert!(qr_expect(&qr), "{delivery:?} qr_compact: {qr:?}");
        let lq = lq_compact(engine, &mut dense, &input)
            .map(|_| ())
            .unwrap_err();
        assert!(lq_expect(&lq), "{delivery:?} lq_compact: {lq:?}");
        let svd = svd_compact(engine, &mut dense, &input)
            .map(|_| ())
            .unwrap_err();
        assert!(svd_expect(&svd), "{delivery:?} svd_compact: {svd:?}");
    }
}

use crate::tests::scripted_executor::BatchIntoSink;
use symtn_dense::{FactorDest, FactorFailure};

/// How a forwarded two-input destination batch reaches the caller.
#[derive(Clone, Copy, Debug)]
enum IntoDelivery {
    /// The second input is never delivered.
    DropLast,
    /// The first input fails in the solver.
    FailFirst,
    /// The first input's outputs do not fit, then the second fails.
    OutputThenFail,
    /// The first input is delivered twice.
    Duplicate,
    /// The inputs are delivered in reversed order.
    Swapped,
    /// The first input is delivered under an index past the batch.
    OutOfRange,
    /// The first input is delivered, then the batch fails.
    PrefixThenError,
    /// The first input fails, the executor ignores the sink's stop, delivers
    /// the second and fails the batch with another error.
    IgnoresStop,
}

/// Overrides `factorize_batch_into` itself (the native destination lanes),
/// standing between them and the caller as `delivery` says.
struct MalformedIntoBatch {
    delivery: IntoDelivery,
}

fn batch_failure() -> DenseError {
    DenseError::Backend {
        backend: DenseBackend::Tenferro,
        op: "injected batch",
        message: "injected batch failure".to_string(),
    }
}

impl Observer for MalformedIntoBatch {
    fn script(script: &mut Script) {
        script
            .set(
                Op::FactorizeBatch,
                Action::Panic("the destination override bypasses the owned batch"),
            )
            .set(Op::FactorizeBatchInto, Action::Forward)
            .set(
                Op::DotGeneral,
                Action::Panic("test only exercises factorizations"),
            );
    }

    fn batch_into_outcome(
        &mut self,
        index: usize,
        dest: &mut FactorDest<'_>,
        outcome: Result<(), FactorFailure>,
        sink: &mut BatchIntoSink<'_>,
    ) -> Result<(), DenseError> {
        use IntoDelivery::*;
        let output = || Err(FactorFailure::Output(arity_mismatch("qr_into", 2, 1)));
        let solver = || Err(FactorFailure::Solver(solver_failure()));
        match (self.delivery, index) {
            (DropLast, 1) => Ok(()),
            (FailFirst, 0) | (OutputThenFail, 1) => sink(index, dest, solver()),
            (OutputThenFail, 0) => sink(index, dest, output()),
            (Duplicate, 0) => {
                sink(index, dest, outcome.clone())?;
                sink(index, dest, outcome)
            }
            // The lanes deliver in order: the first delivery is withheld and
            // the second is delivered first, then again under the first
            // index, as an executor that reorders its deliveries would.
            (Swapped, 0) => Ok(()),
            (Swapped, 1) => {
                sink(1, dest, outcome.clone())?;
                sink(0, dest, outcome)
            }
            (OutOfRange, 0) => sink(5, dest, outcome),
            (PrefixThenError, 1) => Err(batch_failure()),
            (IgnoresStop, 0) => {
                let _ = sink(index, dest, solver());
                Ok(())
            }
            _ => sink(index, dest, outcome),
        }
    }

    fn batch_into_result(&mut self, result: Result<(), DenseError>) -> Result<(), DenseError> {
        match self.delivery {
            IntoDelivery::PrefixThenError => result.and(Err(batch_failure())),
            IntoDelivery::IgnoresStop => Err(batch_failure()),
            _ => result,
        }
    }
}

#[test]
fn compact_factorizations_reject_a_malformed_destination_batch() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: an executor that overrides the destination entry itself, below
    // the owned-batch adapter, gets the same delivery contract. A missing,
    // repeated, reordered or out-of-range delivery is symtn's typed
    // batch-arity error, and no misdelivered destination is staged as a
    // sector; a batch error after a delivered prefix is the result; a
    // staging stop wins over whatever the executor returns after ignoring
    // it; the solver-first QR/SVD batches let a later solver failure win
    // over an earlier malformed output.
    let rule = ZNFusionRule::new(2).unwrap();
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let bound = bound_tensor(Arc::new(rule), &tensor);
    let input = bound.as_ref();
    let arity = |error: &OperationError| {
        matches!(
            error,
            OperationError::Dense(DenseError::ShapeMismatch {
                op: "factorize_batch",
                ..
            })
        )
    };
    let solver = |error: &OperationError| *error == OperationError::Dense(solver_failure());
    let batch = |error: &OperationError| *error == OperationError::Dense(batch_failure());
    type Expect = fn(&OperationError) -> bool;
    let cases: [(IntoDelivery, Expect); 8] = [
        (IntoDelivery::DropLast, arity),
        (IntoDelivery::FailFirst, solver),
        (IntoDelivery::OutputThenFail, solver),
        (IntoDelivery::Duplicate, arity),
        (IntoDelivery::Swapped, arity),
        (IntoDelivery::OutOfRange, arity),
        (IntoDelivery::PrefixThenError, batch),
        (IntoDelivery::IgnoresStop, solver),
    ];
    for (delivery, expect) in cases {
        // Only in-order, in-range deliveries reach a sector: the staged
        // indices are a prefix of `0, 1`.
        let staged_in_order = |recording: &crate::probe::Recording| {
            let staged = recording.staged_destinations();
            staged.len() <= 2 && staged.iter().enumerate().all(|(n, &(index, _))| n == index)
        };
        let mut dense = ScriptedExecutor::new(MalformedIntoBatch { delivery });
        let recording = crate::probe::record();
        let qr = qr_compact(engine, &mut dense, &input)
            .map(|_| ())
            .unwrap_err();
        assert!(expect(&qr), "{delivery:?} qr_compact: {qr:?}");
        assert!(staged_in_order(&recording), "{delivery:?} qr_compact");
        let recording = crate::probe::record();
        let svd = svd_compact(engine, &mut dense, &input)
            .map(|_| ())
            .unwrap_err();
        assert!(expect(&svd), "{delivery:?} svd_compact: {svd:?}");
        assert!(staged_in_order(&recording), "{delivery:?} svd_compact");
        assert_eq!(dense.counts().factorize_batch, 0, "{delivery:?}");
    }
}

/// Whether every staged payload address of output `output` lies in
/// `published`, the first one at its start.
fn staged_into(
    staged: &[(usize, [Option<usize>; 3])],
    output: usize,
    published: &[impl Sized],
) -> bool {
    let start = published.as_ptr() as usize;
    let end = start + std::mem::size_of_val(published);
    !staged.is_empty()
        && staged[0].1[output] == Some(start)
        && staged
            .iter()
            .all(|(_, outputs)| outputs[output].is_some_and(|at| (start..end).contains(&at)))
}

#[test]
fn direct_factor_destinations_are_the_published_allocations() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: the direct-region QR, LQ, SVD and eigh batches hand each sector a
    // destination inside the factor buffer the operation publishes, so a
    // delivered factor reaches the result with no further payload copy.
    let rule = ZNFusionRule::new(2).unwrap();
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let bound = bound_tensor(Arc::new(rule), &tensor);
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let recording = crate::probe::record();
    let qr = qr_compact(engine, &mut dense, &bound.as_ref()).unwrap();
    let staged = recording.staged_destinations();
    assert_eq!(staged.len(), 2);
    assert!(staged_into(&staged, 0, qr.q.data()));
    assert!(staged_into(&staged, 1, qr.r.data()));

    let recording = crate::probe::record();
    let lq = lq_compact(engine, &mut dense, &bound.as_ref()).unwrap();
    let staged = recording.staged_destinations();
    assert_eq!(staged.len(), 2);
    assert!(staged_into(&staged, 0, lq.l.data()));
    assert!(staged_into(&staged, 1, lq.q.data()));

    let recording = crate::probe::record();
    let svd = svd_compact(engine, &mut dense, &bound.as_ref()).unwrap();
    let staged = recording.staged_destinations();
    assert_eq!(staged.len(), 2);
    assert!(staged_into(&staged, 0, svd.u.data()));
    assert!(staged_into(&staged, 2, svd.vh.data()));

    let hermitian = one_sector_matrix(vec![2.0_f64, 1.0, 1.0, 3.0]);
    let recording = crate::probe::record();
    let eigh = crate::factorize::eigh_full(
        engine,
        &mut dense,
        &bound_tensor_ref!(Arc::new(ZNFusionRule::new(2).unwrap()), &hermitian),
    )
    .unwrap();
    let staged = recording.staged_destinations();
    assert_eq!(staged.len(), 1);
    assert!(staged_into(&staged, 1, eigh.v.data()));
}
