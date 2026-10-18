//! Hermitian and general eigendecomposition tests (#1596 split of tests.rs).

use super::*;

#[test]
fn eigh_canonical_layout_skips_input_pack_and_vector_scatter() {
    // What: canonical EIGH reads source regions and writes final eigenvector regions directly.
    let rule = Z2FusionRule;
    let tensor = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let eigh_copy_rec = crate::probe::record();
    eigh_full(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor)).unwrap();

    assert_eq!(
        eigh_copy_rec.eigh_copy(),
        crate::probe::EighCopyProbe::default()
    );
}

#[test]
fn checked_only_generic_eigh_validates_every_region_before_dense_work() {
    let (space, mut hermitian, _) = generic_values_endomorphism_input();
    let (provider, space) = bind_checked_only(&space);
    let regions = space
        .space()
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .unwrap();
    let later = regions.last().unwrap();
    hermitian[later.range().start + 1] += Complex64::new(1.0, 0.0);
    let before = hermitian.clone();
    let provider_calls = provider.calls.get();
    let mut dense = ScriptedExecutor::<EighCallSpy>::default();
    let values_fallbacks_rec = crate::probe::record();

    let error = eigh_vals_dyn_checked_generic(
        &mut dense,
        &BoundDynamicTensorRef::try_new(&space, &hermitian).unwrap(),
    )
    .unwrap_err();

    assert!(matches!(
        error,
        CheckedGenericFactorPlanError::Operation(OperationError::InvalidArgument {
            message: "eigh requires Hermitian coupled-sector blocks",
        })
    ));
    assert_eq!(dense.counts().of(EIGH_ENTRIES), 0);
    assert_eq!(
        values_fallbacks_rec.get(crate::probe::Event::ValuesMatricizationFallback),
        0
    );
    assert_eq!(provider.calls.get(), provider_calls);
    assert_eq!(hermitian, before);
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_eigh_stages_dense_work_before_checked_factor_admission() {
    let x = SectorId::new(1);
    let leg = SectorLeg::new([(x, 1)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone(), leg.clone()]),
        FusionProductSpace::new([leg.clone(), leg]),
    );
    let source = BoundDynamicFusionMapSpace::from_final_homspace_generic(
        Arc::new(FactorGenericRule),
        homspace,
    )
    .unwrap();
    let data = vec![0.0; source.space().required_len().unwrap()];

    let failing = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: 1,
        calls: Cell::new(0),
        identity: RuleIdentity::new_unique::<LateGenericSpy>(),
    });
    let checked = bind_to_spy(&source, &failing);
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let mut dense = ScriptedExecutor::<CountingDense>::default();
    assert!(matches!(
        eigh_full_dyn_checked_generic(&mut dense, &input, HermitianTol::DEFAULT),
        Err(CheckedGenericFactorPlanError::Provider(LateGenericError(1)))
    ));
    assert_eq!(dense.counts().of(&[Op::Eigh, Op::EighInto]), 2);
    assert_eq!(input.data(), data);

    let complete = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
        identity: RuleIdentity::new_unique::<LateGenericSpy>(),
    });
    let checked = bind_to_spy(&source, &complete);
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let mut dense = ScriptedExecutor::<CountingDense>::default();
    let full = eigh_full_dyn_checked_generic(&mut dense, &input, HermitianTol::DEFAULT).unwrap();
    assert_eq!(dense.counts().of(&[Op::Eigh, Op::EighInto]), 2);
    assert!(Arc::ptr_eq(full.v().space().provider_arc(), &complete));
    assert!(full
        .eigenvalues()
        .iter()
        .flat_map(|entry| &entry.values)
        .all(|value| *value == 0.0));
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_eigh_late_dense_failure_publishes_no_factors() {
    let x = SectorId::new(1);
    let leg = SectorLeg::new([(x, 1)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone(), leg.clone()]),
        FusionProductSpace::new([leg.clone(), leg]),
    );
    let source = BoundDynamicFusionMapSpace::from_final_homspace_generic(
        Arc::new(FactorGenericRule),
        homspace,
    )
    .unwrap();
    let data = vec![0.0; source.space().required_len().unwrap()];
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
        identity: RuleIdentity::new_unique::<LateGenericSpy>(),
    });
    let checked = bind_to_spy(&source, &provider);
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    provider.calls.set(0);
    let mut dense = ScriptedExecutor::new(FailAfterObservingEighInput {
        outputs: Some(f64_eigh_outputs(1)),
        ..Default::default()
    });
    let one_sided_rec = crate::probe::record();

    let result = eigh_full_dyn_checked_generic(&mut dense, &input, HermitianTol::DEFAULT);

    assert!(matches!(
        result,
        Err(CheckedGenericFactorPlanError::Operation(
            OperationError::Dense(_)
        ))
    ));
    assert_eq!(dense.observed.len(), 2);
    assert_eq!(provider.calls.get(), 0);
    assert_eq!(input.data(), data);
    assert_eq!(
        one_sided_rec.one_sided_publication(),
        crate::probe::OneSidedPublicationProbe::default()
    );
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_eigh_uses_owned_dense_output() {
    let x = SectorId::new(1);
    let leg = SectorLeg::new([(x, 1)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone(), leg.clone()]),
        FusionProductSpace::new([leg.clone(), leg]),
    );
    let source = BoundDynamicFusionMapSpace::from_final_homspace_generic(
        Arc::new(FactorGenericRule),
        homspace,
    )
    .unwrap();
    let data = vec![0.0; source.space().required_len().unwrap()];
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
        identity: RuleIdentity::new_unique::<LateGenericSpy>(),
    });
    let checked = bind_to_spy(&source, &provider);
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let mut dense = ScriptedExecutor::<RejectEighInto>::default();

    let full = eigh_full_dyn_checked_generic(&mut dense, &input, HermitianTol::DEFAULT).unwrap();

    assert_eq!(dense.counts().eigh, 2);
    assert_eq!(dense.counts().eigh_into, 0);
    assert!(Arc::ptr_eq(full.v().space().provider_arc(), &provider));
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_eigh_keeps_owned_vectors_in_live_pairs_before_publication() {
    let x = SectorId::new(1);
    let leg = SectorLeg::new([(x, 1)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone(), leg.clone()]),
        FusionProductSpace::new([leg.clone(), leg]),
    );
    let source = BoundDynamicFusionMapSpace::from_final_homspace_generic(
        Arc::new(FactorGenericRule),
        homspace,
    )
    .unwrap();
    let data = vec![0.0; source.space().required_len().unwrap()];
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
        identity: RuleIdentity::new_unique::<LateGenericSpy>(),
    });
    let checked = bind_to_spy(&source, &provider);
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let mut dense = ScriptedExecutor::<RejectEighInto>::default();
    let eigh_pairs_rec = crate::probe::record();

    let full = eigh_full_dyn_checked_generic(&mut dense, &input, HermitianTol::DEFAULT).unwrap();
    let before_publication = eigh_pairs_rec.checked_eigh_pairs();

    assert_eq!(dense.counts().eigh_into, 0);
    assert_eq!(dense.vector_ptrs, before_publication);
    assert_eq!(dense.vector_ptrs.len(), 2);
    assert_eq!(dense.vector_ptrs.len(), full.eigenvalues().len());
}

#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn assert_checked_generic_eigh_live_pair_owners<D: crate::factorize::FactorScalar>() {
    let x = SectorId::new(1);
    let leg = SectorLeg::new([(x, 1)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone(), leg.clone()]),
        FusionProductSpace::new([leg.clone(), leg]),
    );
    let source = BoundDynamicFusionMapSpace::from_final_homspace_generic(
        Arc::new(FactorGenericRule),
        homspace,
    )
    .unwrap();
    let data = vec![D::zero(); source.space().required_len().unwrap()];
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
        identity: RuleIdentity::new_unique::<LateGenericSpy>(),
    });
    let checked = bind_to_spy(&source, &provider);
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let mut dense = ScriptedExecutor::<RejectEighInto>::default();
    let eigh_pairs_rec = crate::probe::record();

    let full = eigh_full_dyn_checked_generic(&mut dense, &input, HermitianTol::DEFAULT).unwrap();

    assert_eq!(dense.counts().eigh_into, 0);
    assert_eq!(dense.vector_ptrs, eigh_pairs_rec.checked_eigh_pairs());
    assert_eq!(dense.vector_ptrs.len(), 2);
    assert_eq!(dense.vector_ptrs.len(), full.eigenvalues().len());
    assert!(Arc::ptr_eq(full.v().space().provider_arc(), &provider));
}

#[test]
fn checked_generic_eigh_keeps_live_pair_owners_for_every_dtype() {
    assert_checked_generic_eigh_live_pair_owners::<f64>();
    assert_checked_generic_eigh_live_pair_owners::<f32>();
    assert_checked_generic_eigh_live_pair_owners::<Complex32>();
    assert_checked_generic_eigh_live_pair_owners::<Complex64>();
}

fn assert_complex_checked_eigh_reconstruction<R>(
    source_regions: &[symtn_core::CoupledSectorRegion],
    hermitian: &[Complex64],
    full: &EighFullDyn<R, Complex64>,
) {
    let vector_regions = full
        .v()
        .space()
        .space()
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .unwrap();

    for source_region in source_regions {
        let vectors = vector_regions
            .iter()
            .find(|region| region.coupled() == source_region.coupled())
            .unwrap();
        let values = &full
            .eigenvalues()
            .iter()
            .find(|spectrum| spectrum.sector == source_region.coupled())
            .unwrap()
            .values;
        let n = source_region.rows();
        assert_eq!((vectors.rows(), vectors.cols(), values.len()), (n, n, n));
        for column in 0..n {
            for row in 0..n {
                let reconstructed = (0..n)
                    .map(|bond| {
                        full.v().data()[vectors.range().start + row + n * bond]
                            * values[bond]
                            * full.v().data()[vectors.range().start + column + n * bond].conj()
                    })
                    .sum::<Complex64>();
                let expected = hermitian[source_region.range().start + row + n * column];
                assert!((reconstructed - expected).norm() < 1.0e-10);
            }
        }
    }
}

#[test]
fn checked_generic_eigh_reconstructs_complex_unequal_multi_tree_sectors() {
    let (source, hermitian, _) = generic_values_endomorphism_input();
    let source_regions = source
        .space()
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .unwrap();
    assert_eq!(
        source_regions
            .iter()
            .map(|region| region.rows())
            .collect::<Vec<_>>(),
        [1, 2]
    );
    let multi_tree = source_regions
        .iter()
        .find(|region| region.rows() == 2)
        .unwrap();
    assert_eq!(
        (multi_tree.row_trees().len(), multi_tree.col_trees().len()),
        (2, 2)
    );
    assert!(hermitian.iter().any(|value| value.im != 0.0));

    let (provider, checked) = bind_checked_only(&source);
    let input = BoundDynamicTensorRef::try_new(&checked, &hermitian).unwrap();
    let full = eigh_full_dyn_checked_generic(
        &mut symtn_dense::DefaultDenseExecutor::new(),
        &input,
        HermitianTol::DEFAULT,
    )
    .unwrap();
    assert_complex_checked_eigh_reconstruction(&source_regions, &hermitian, &full);
    assert!(Arc::ptr_eq(full.v().space().provider_arc(), &provider));
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_eigh_reconstructs_padded_reordered_complex_input() {
    let (source, hermitian, _) = generic_values_endomorphism_input();
    let source_regions = source
        .space()
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .unwrap();
    let (expert, data) = padded_reordered_generic_endomorphism_input(&source, &hermitian);
    assert!(expert
        .space()
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .is_none());
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
        identity: FactorGenericRule.rule_identity(),
    });
    let checked =
        BoundDynamicFusionMapSpace::bind_generic(expert.space().clone(), Arc::clone(&provider))
            .unwrap();
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let before = input.data().to_vec();
    let full = eigh_full_dyn_checked_generic(
        &mut symtn_dense::DefaultDenseExecutor::new(),
        &input,
        HermitianTol::DEFAULT,
    )
    .unwrap();
    assert_complex_checked_eigh_reconstruction(&source_regions, &hermitian, &full);
    assert_eq!(input.data(), before);
    assert!(Arc::ptr_eq(full.v().space().provider_arc(), &provider));
}

#[test]
fn checked_generic_eigh_stably_keeps_raw_exact_signed_ties() {
    let (source, mut hermitian, _) = generic_values_endomorphism_input();
    let regions = source
        .space()
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .unwrap();
    for region in regions.iter() {
        let values = match region.rows() {
            1 => &[Complex64::new(1.0, 0.0)][..],
            2 => &[
                Complex64::new(-2.0, 0.0),
                Complex64::zero(),
                Complex64::zero(),
                Complex64::new(2.0, 0.0),
            ],
            rows => panic!("unexpected checked EIGH tie fixture size {rows}"),
        };
        hermitian[region.range()].copy_from_slice(values);
    }
    let (_, checked) = bind_checked_only(&source);
    let input = BoundDynamicTensorRef::try_new(&checked, &hermitian).unwrap();
    let mut dense = ScriptedExecutor::<RecordingEigh>::default();

    let full = eigh_full_dyn_checked_generic(&mut dense, &input, HermitianTol::DEFAULT).unwrap();

    let tied_sector = full
        .eigenvalues()
        .iter()
        .find(|spectrum| spectrum.values.len() == 2)
        .unwrap();
    let raw_tied = dense
        .raw_values
        .iter()
        .find(|values| values.len() == 2)
        .unwrap()
        .iter()
        .copied()
        .filter(|value| value.abs() == 2.0)
        .collect::<Vec<_>>();
    let published_tied = tied_sector
        .values
        .iter()
        .copied()
        .filter(|value| value.abs() == 2.0)
        .collect::<Vec<_>>();
    assert_eq!(raw_tied.len(), 2);
    assert!(raw_tied.iter().any(|value| *value < 0.0));
    assert!(raw_tied.iter().any(|value| *value > 0.0));
    assert_eq!(published_tied, raw_tied);
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_eig_stages_dense_work_before_checked_factor_admission() {
    let x = SectorId::new(1);
    let leg = SectorLeg::new([(x, 1)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone(), leg.clone()]),
        FusionProductSpace::new([leg.clone(), leg]),
    );
    let source = BoundDynamicFusionMapSpace::from_final_homspace_generic(
        Arc::new(FactorGenericRule),
        homspace,
    )
    .unwrap();
    let data = vec![0.0; source.space().required_len().unwrap()];
    let failing = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: 1,
        calls: Cell::new(0),
        identity: RuleIdentity::new_unique::<LateGenericSpy>(),
    });
    let checked = bind_to_spy(&source, &failing);
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let mut dense = ScriptedExecutor::<CountingDense>::default();
    assert!(matches!(
        eig_full_dyn_checked_generic(&mut dense, &input),
        Err(CheckedGenericFactorPlanError::Provider(LateGenericError(1)))
    ));
    assert_eq!(dense.counts().eig, 2);
    assert_eq!(dense.counts().svd_vals, 0);
    assert_eq!(input.data(), data);
}

#[test]
fn eigh_noncanonical_layout_uses_copy_fallback() {
    // What: expert noncanonical EIGH retains positive pack-and-vector-scatter copy evidence.
    let rule = Z2FusionRule;
    let tensor = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let bound = bound_tensor(Arc::new(rule), &tensor);
    let adjoint_space = bound.space().adjoint_view().unwrap();
    let input = BoundDynamicTensorRef::try_new(&adjoint_space, bound.data()).unwrap();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let eigh_copy_rec = crate::probe::record();
    eigh_full_dyn(&mut dense, &input, HermitianTol::DEFAULT).unwrap();
    let probe = eigh_copy_rec.eigh_copy();

    assert!(probe.input_pack_bytes > 0);
    assert!(probe.output_scatter_bytes > 0);
}

#[test]
fn eigh_direct_rejects_a_later_nonhermitian_sector_before_any_dense_call() {
    // What: canonical EIGH validates every coupled sector without packing before any driver call.
    let rule = Z2FusionRule;
    let tensor = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let regions = tensor
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .unwrap();
    let later = regions.last().unwrap();
    let mut data = tensor.data().to_vec();
    data[later.range().start + 1] += 1.0;
    let nonhermitian =
        TestTensor::<f64>::from_vec_with_fusion_space(data, tensor.fusion_space().clone()).unwrap();
    let mut dense = ScriptedExecutor::<EighCallSpy>::default();

    let eigh_copy_rec = crate::probe::record();
    let error = eigh_full(
        &mut dense,
        &bound_tensor_ref!(Arc::new(rule), &nonhermitian),
    )
    .unwrap_err();

    assert_eq!(
        error,
        OperationError::InvalidArgument {
            message: "eigh requires Hermitian coupled-sector blocks",
        }
    );
    assert_eq!(dense.counts().of(EIGH_ENTRIES), 0);
    assert_eq!(
        eigh_copy_rec.eigh_copy(),
        crate::probe::EighCopyProbe::default()
    );
}

#[test]
fn eigh_fallback_rejects_nonhermitian_complex_input_before_dense_execution() {
    // What: a valid noncanonical layout receives the same complex Hermitian preflight after packing.
    let rule = Z2FusionRule;
    let real = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let regions = real.structure().coupled_sector_regions(2).unwrap().unwrap();
    let later = regions.last().unwrap();
    let mut data = real
        .data()
        .iter()
        .map(|&value| Complex64::new(value, 0.0))
        .collect::<Vec<_>>();
    data[later.range().start + 1] += Complex64::new(1.0, 2.0);
    let tensor =
        TestTensor::<Complex64>::from_vec_with_fusion_space(data, real.fusion_space().clone())
            .unwrap();
    let bound = bound_tensor(Arc::new(rule), &tensor);
    let adjoint_space = bound.space().adjoint_view().unwrap();
    let input = BoundDynamicTensorRef::try_new(&adjoint_space, bound.data()).unwrap();
    let mut dense = ScriptedExecutor::<EighCallSpy>::default();

    let eigh_copy_rec = crate::probe::record();
    let error = eigh_full_dyn(&mut dense, &input, HermitianTol::DEFAULT).unwrap_err();

    assert_eq!(
        error,
        OperationError::InvalidArgument {
            message: "eigh requires Hermitian coupled-sector blocks",
        }
    );
    assert_eq!(dense.counts().of(EIGH_ENTRIES), 0);
    assert!(eigh_copy_rec.eigh_copy().input_pack_bytes > 0);
}

#[test]
fn eigh_vals_rejects_a_later_nonhermitian_sector_before_any_dense_call() {
    // What: values-only EIGH validates every borrowed sector before its first no-vector driver.
    let rule = Z2FusionRule;
    let tensor = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let regions = tensor
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .unwrap();
    let later = regions.last().unwrap();
    let mut data = tensor.data().to_vec();
    data[later.range().start + 1] += 1.0;
    let nonhermitian =
        TestTensor::<f64>::from_vec_with_fusion_space(data, tensor.fusion_space().clone()).unwrap();
    let before = nonhermitian.data().to_vec();
    let mut dense = ScriptedExecutor::<EighCallSpy>::default();

    let error = eigh_vals(
        &mut dense,
        &bound_tensor_ref!(Arc::new(rule), &nonhermitian),
    )
    .unwrap_err();

    assert_eq!(
        error,
        OperationError::InvalidArgument {
            message: "eigh requires Hermitian coupled-sector blocks",
        }
    );
    assert_eq!(dense.counts().of(EIGH_ENTRIES), 0);
    assert_eq!(nonhermitian.data(), before);
}

/// `[[1, δ], [0, 2]]` passes when `|δ|/√2 ≤ tol·√(5 + δ²)`.
/// Use the small-δ threshold `tol·√10` with a 3% margin.
fn hermitian_threshold_pair<D: FactorScalar>(tol: f64, scale: f64) -> [TestTensor<D>; 2] {
    let threshold = tol * 10.0_f64.sqrt();
    let matrix = |delta: f64| {
        one_sector_matrix(
            [1.0, delta, 0.0, 2.0]
                .into_iter()
                .map(|value| D::from_real(value * scale))
                .collect(),
        )
    };
    [matrix(0.97 * threshold), matrix(1.03 * threshold)]
}

#[test]
fn eigh_uses_eps_three_quarters_relative_tolerance_for_every_factor_dtype() {
    // What: by default (#1987) normalized residuals below eps^(3/4) pass and
    // those above it fail, at the payload's own precision.
    fn check<D: FactorScalar + std::fmt::Debug>() {
        let [within, outside] = hermitian_threshold_pair::<D>(D::epsilon().powf(0.75), 1.0);
        assert_eigh_preflight(&within, true);
        assert_eigh_preflight(&outside, false);
    }
    check::<f32>();
    check::<Complex32>();
    check::<f64>();
    check::<Complex64>();
}

#[test]
fn eigh_explicit_hermitian_tol_moves_the_threshold() {
    // What: an explicit tolerance replaces the default in both directions;
    // a 64·eps rule is one such tolerance.
    for tol in [64.0 * f64::EPSILON, 1.0e-6] {
        let hermitian_tol = HermitianTol::relative(tol).unwrap();
        let [within, outside] = hermitian_threshold_pair::<f64>(tol, 1.0);
        assert_eigh_preflight_at(&within, true, hermitian_tol);
        assert_eigh_preflight_at(&outside, false, hermitian_tol);
    }
    // Zero admits only exactly Hermitian blocks.
    let exact = HermitianTol::relative(0.0).unwrap();
    assert_eigh_preflight_at(
        &one_sector_matrix(vec![1.0_f64, 0.5, 0.5, 2.0]),
        true,
        exact,
    );
    let [nearly, _] = hermitian_threshold_pair::<f64>(f64::EPSILON, 1.0);
    assert_eigh_preflight_at(&nearly, false, exact);
    for invalid in [-1.0e-12, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            HermitianTol::relative(invalid),
            Err(OperationError::InvalidArgument {
                message: "hermitian_tol must be finite and non-negative"
            })
        ));
    }
}

#[test]
fn eigh_hermitian_preflight_is_invariant_under_finite_rescaling() {
    // What: multiplying a block cannot change a fixed relative perturbation's classification.
    for scale in [1.0e-200, 1.0, 1.0e200] {
        let [accepted, rejected] = hermitian_threshold_pair::<f64>(f64::EPSILON.powf(0.75), scale);
        assert_eigh_preflight(&accepted, true);
        assert_eigh_preflight(&rejected, false);
    }
    for scale in [1.0e-30, 1.0, 1.0e30] {
        let [accepted, rejected] =
            hermitian_threshold_pair::<f32>(f64::from(f32::EPSILON).powf(0.75), scale);
        assert_eigh_preflight(&accepted, true);
        assert_eigh_preflight(&rejected, false);
    }
}

#[test]
fn eigh_hermitian_preflight_preserves_subnormal_relative_defects() {
    // What: normalization precedes subtraction, so dividing by two cannot erase a minimum subnormal defect.
    let s32 = f32::from_bits(1);
    let s64 = f64::from_bits(1);
    assert_eigh_preflight(&one_sector_matrix(vec![s32, s32, s32, 0.0]), true);
    assert_eigh_preflight(&one_sector_matrix(vec![s64, s64, s64, 0.0]), true);
    assert_eigh_preflight(&one_sector_matrix(vec![s32, 0.0, s32, 0.0]), false);
    assert_eigh_preflight(&one_sector_matrix(vec![s64, 0.0, s64, 0.0]), false);
}

#[test]
fn eigh_accepts_exact_hermitian_max_magnitude_inputs() {
    // What: stable Frobenius scaling accepts exact Hermitian matrices at finite maxima.
    let max_f32 = one_sector_matrix(vec![f32::MAX, 0.0, 0.0, f32::MAX]);
    let max_f64 = one_sector_matrix(vec![f64::MAX, 0.0, 0.0, f64::MAX]);
    let max_c32 = one_sector_matrix(vec![
        Complex32::new(f32::MAX, 0.0),
        Complex32::new(0.0, 0.0),
        Complex32::new(0.0, 0.0),
        Complex32::new(f32::MAX, 0.0),
    ]);
    let max_c64 = one_sector_matrix(vec![
        Complex64::new(f64::MAX, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(f64::MAX, 0.0),
    ]);

    assert_eigh_preflight(&max_f32, true);
    assert_eigh_preflight(&max_f64, true);
    assert_eigh_preflight(&max_c32, true);
    assert_eigh_preflight(&max_c64, true);
}

#[test]
fn eigh_rejects_a_large_nonhermitian_input() {
    // What: overflow-safe tolerance comparison still rejects large finite asymmetry.
    let tensor = one_sector_matrix(vec![f64::MAX, f64::MAX, 0.0, f64::MAX]);

    assert_eigh_preflight(&tensor, false);
}

#[test]
fn eigh_relative_hermitian_preflight_is_block_size_independent() {
    // What: repeating the same relative diagonal defect cannot change acceptance with block size.
    let rtol = f64::EPSILON.powf(0.75);
    for n in [2, 64] {
        for (delta, accepted) in [(rtol / 2.0, true), (2.0 * rtol, false)] {
            let mut data = vec![Complex64::new(0.0, 0.0); n * n];
            for diagonal in 0..n {
                data[diagonal + n * diagonal] = Complex64::new(1.0, delta);
            }
            assert_eigh_preflight(&one_sector_rectangular_matrix(data, n, n), accepted);
        }
    }
}

#[test]
fn eigh_relative_hermitian_preflight_counts_cross_block_pairs_twice() {
    // What: a defect spanning the 32x32 traversal boundary contributes both conjugate positions.
    const N: usize = 33;
    let rtol = f64::EPSILON.powf(0.75);
    for (factor, accepted) in [(1.3, true), (1.6, false)] {
        let mut data = vec![0.0; N * N];
        for diagonal in 0..N {
            data[diagonal + N * diagonal] = 1.0;
        }
        data[N * 32] = factor * rtol * (N as f64).sqrt();
        assert_eigh_preflight(&one_sector_rectangular_matrix(data, N, N), accepted);
    }
}

#[test]
fn eigh_rejects_a_nonreal_complex_diagonal_before_dense_execution() {
    // What: complex Hermitian validation checks diagonal reality as well as off-diagonal conjugacy.
    let tensor = one_sector_matrix(vec![
        Complex64::new(1.0, 1.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(2.0, 0.0),
    ]);
    let mut dense = ScriptedExecutor::<EighCallSpy>::default();

    let error = eigh_full(
        &mut dense,
        &bound_tensor_ref!(Arc::new(Z2FusionRule), &tensor),
    )
    .unwrap_err();

    assert!(matches!(error, OperationError::InvalidArgument { .. }));
    assert_eq!(dense.counts().of(EIGH_ENTRIES), 0);
}

#[test]
fn eigh_rejects_nonfinite_input_before_dense_execution() {
    // What: NaN and infinity cannot satisfy the Hermitian EIGH input contract.
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let tensor = one_sector_matrix(vec![value, 0.0, 0.0, 2.0]);
        let mut dense = ScriptedExecutor::<EighCallSpy>::default();
        let error = eigh_full(
            &mut dense,
            &bound_tensor_ref!(Arc::new(Z2FusionRule), &tensor),
        )
        .unwrap_err();
        assert!(matches!(error, OperationError::InvalidArgument { .. }));
        assert_eq!(dense.counts().of(EIGH_ENTRIES), 0);
    }
}

#[test]
fn eigh_preserves_endomorphism_error_precedence() {
    // What: a non-endomorphism retains its structural error before numeric Hermitian inspection.
    let tensor = one_sector_rectangular_matrix(vec![f64::NAN; 6], 2, 3);
    let mut dense = ScriptedExecutor::<EighCallSpy>::default();

    let error = eigh_full(
        &mut dense,
        &bound_tensor_ref!(Arc::new(Z2FusionRule), &tensor),
    )
    .unwrap_err();

    assert_eq!(
        error,
        OperationError::SpaceMismatch {
            message: "eigh requires an endomorphism (codomain == domain)",
        }
    );
    assert_eq!(dense.counts().of(EIGH_ENTRIES), 0);
}

#[test]
fn eigh_error_preserves_borrowed_input_and_publishes_no_output() {
    // What: an EIGH backend failure leaves borrowed storage unchanged and returns no vectors.
    let rule = Arc::new(Z2FusionRule);
    let canonical = hermitian_test_tensor(rule.as_ref(), &[SectorId::new(0), SectorId::new(1)]);
    let padded = padded_copy(rule.as_ref(), &canonical);
    for (tensor, is_fallback) in [(&canonical, false), (&padded, true)] {
        let before = tensor.data().to_vec();
        let mut dense = ScriptedExecutor::<FailAfterObservingEighInput>::default();

        let eigh_copy_rec = crate::probe::record();
        let result = eigh_full(&mut dense, &bound_tensor_ref!(Arc::clone(&rule), tensor));

        assert!(matches!(result, Err(OperationError::Dense(_))));
        assert_eq!(tensor.data(), before);
        assert!(!dense.observed.is_empty());
        if is_fallback {
            assert!(eigh_copy_rec.eigh_copy().input_pack_bytes > 0);
        } else {
            assert!(dense
                .observed
                .iter()
                .all(|sector| before.windows(sector.len()).any(|window| window == sector)));
        }
    }
}

fn unequal_fallback_eigh_fixtures() -> (TestTensor<f64>, TestTensor<Complex64>) {
    let source = mixed_rectangular_tensor((3, 3), (2, 2));
    let source_regions = source
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    assert_eq!(
        source_regions
            .iter()
            .map(|region| region.rows())
            .collect::<Vec<_>>(),
        vec![3, 2]
    );
    let mut data = vec![0.0; source.data().len()];
    for region in source_regions.iter() {
        for diagonal in 0..region.rows() {
            data[region.range().start + diagonal + region.rows() * diagonal] = match diagonal {
                0 => -2.0,
                1 => 2.0,
                _ => 1.0,
            };
        }
    }
    let real =
        TestTensor::<f64>::from_vec_with_fusion_space(data, source.fusion_space().clone()).unwrap();
    let complex = TestTensor::<Complex64>::from_vec_with_fusion_space(
        real.data()
            .iter()
            .map(|&value| Complex64::new(value, 0.0))
            .collect(),
        real.fusion_space().clone(),
    )
    .unwrap();
    let mut complex_data = complex.data().to_vec();
    for region in source_regions.iter() {
        complex_data[region.range().start + 1] = Complex64::new(1.2, 1.6);
        complex_data[region.range().start + region.rows()] = Complex64::new(1.2, -1.6);
    }
    let complex = TestTensor::<Complex64>::from_vec_with_fusion_space(
        complex_data,
        complex.fusion_space().clone(),
    )
    .unwrap();
    (real, complex)
}

#[test]
fn eigh_fallback_publishes_the_provider_ascending_order() {
    // What: the noncanonical fallback publishes the solver's ascending order,
    // ±x ties included, unchanged.
    let rule = Arc::new(Z2FusionRule);
    let (source, _) = unequal_fallback_eigh_fixtures();
    let source_regions = source
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    let padded = padded_copy(rule.as_ref(), &source);
    assert!(padded
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .is_none());
    let mut dense = ScriptedExecutor::<RecordingEigh>::default();

    let eigh = eigh_full(&mut dense, &bound_tensor_ref!(Arc::clone(&rule), &padded)).unwrap();

    assert_eq!(dense.raw_values.len(), source_regions.len());
    assert_eq!(
        eigh.eigenvalues
            .iter()
            .map(|spectrum| spectrum.sector)
            .collect::<Vec<_>>(),
        source_regions
            .iter()
            .map(|region| region.coupled())
            .collect::<Vec<_>>(),
    );
    for (spectrum, raw) in eigh.eigenvalues.iter().zip(&dense.raw_values) {
        // The provider's ascending spectrum (with its ±2 pair) is published
        // as returned, without a permutation.
        assert!(raw.windows(2).all(|pair| pair[0] <= pair[1]));
        assert!(raw.contains(&-2.0) && raw.contains(&2.0));
        assert_eq!(&spectrum.values, raw);
    }
}

#[test]
fn eigh_fallback_reconstructs_complex_unequal_sectors() {
    // What: the padded fallback scatters each complex owned V into the public
    // factor structure without changing any sector's V D V^H reconstruction.
    let rule = Arc::new(Z2FusionRule);
    let (_, source) = unequal_fallback_eigh_fixtures();
    let source_regions = source
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    let padded = padded_copy(rule.as_ref(), &source);
    assert!(padded
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .is_none());
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let eigh = eigh_full(&mut dense, &bound_tensor_ref!(Arc::clone(&rule), &padded)).unwrap();
    let vector_regions = eigh
        .v
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();
    for source_region in source_regions.iter() {
        let vector_region = vector_regions
            .iter()
            .find(|region| region.coupled() == source_region.coupled())
            .unwrap();
        let values = &eigh
            .eigenvalues
            .iter()
            .find(|spectrum| spectrum.sector == source_region.coupled())
            .unwrap()
            .values;
        let n = source_region.rows();
        assert_eq!(vector_region.rows(), n);
        for column in 0..n {
            for row in 0..n {
                let reconstructed = (0..n)
                    .map(|bond| {
                        eigh.v.data()[vector_region.range().start + row + n * bond]
                            * values[bond]
                            * eigh.v.data()[vector_region.range().start + column + n * bond].conj()
                    })
                    .sum::<Complex64>();
                let expected = source.data()[source_region.range().start + row + n * column];
                assert!((reconstructed - expected).norm() < 1e-9);
            }
        }
    }
}

#[test]
fn eigh_direct_column_reorder_preserves_the_literal_three_cycle() {
    // What: direct owned vectors reorder in place by destination columns [1, 2, 0].
    let mut vectors = vec![
        10.0, 11.0, 12.0, // column 0
        20.0, 21.0, 22.0, // column 1
        30.0, 31.0, 32.0, // column 2
    ];
    let mut visited = vec![false; 3];
    let mut scratch = vec![0.0; 3];
    crate::factorize::reorder_columns_in_place_for_test(
        &mut vectors,
        3,
        &[1, 2, 0],
        &mut visited,
        &mut scratch,
    );
    assert_eq!(
        vectors,
        vec![20.0, 21.0, 22.0, 30.0, 31.0, 32.0, 10.0, 11.0, 12.0]
    );
}

#[test]
fn eigh_rejects_non_finite_owned_eigenvalues_before_sorting() {
    // What: the private owned-output validator rejects malformed spectra before sorting.
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            crate::factorize::validate_real_eigenvalues_for_test(&[value]),
            Err(OperationError::Dense(DenseError::NumericalFailure {
                backend: DenseBackend::Tenferro,
                op: crate::factorize::EIGH_EIGENVALUE_CHECK,
                message: "eigenvalues must be finite".to_string(),
            }))
        );
    }
}

#[test]
fn eigh_vectors_retain_each_callers_exact_provider_arc() {
    // What: per-call EIGH factor construction preserves each caller's provider allocation.
    let tensor = hermitian_test_tensor(&Z2FusionRule, &[SectorId::new(0), SectorId::new(1)]);
    let first_provider = Arc::new(Z2FusionRule);
    let second_provider = Arc::new(Z2FusionRule);
    let first = bound_tensor(Arc::clone(&first_provider), &tensor);
    let second = bound_tensor(Arc::clone(&second_provider), &tensor);
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let first_eigh = eigh_full(&mut dense, &first.as_ref()).unwrap();
    let second_eigh = eigh_full(&mut dense, &second.as_ref()).unwrap();

    assert!(Arc::ptr_eq(
        first_eigh.v.space().provider_arc(),
        &first_provider
    ));
    assert!(Arc::ptr_eq(
        second_eigh.v.space().provider_arc(),
        &second_provider
    ));
}

#[test]
fn eigh_direct_outputs_keep_executor_vector_owner() {
    // What: a one-region direct EIGH publishes the executor-returned V buffer.
    fn check<D: crate::factorize::FactorScalar>(tensor: &TestTensor<D>) {
        let mut dense = ScriptedExecutor::<RejectEighInto>::default();
        let eigh = eigh_full(
            &mut dense,
            &bound_tensor_ref!(Arc::new(Z2FusionRule), tensor),
        )
        .unwrap();
        assert_eq!(dense.counts().eigh, 1);
        assert_eq!(dense.counts().eigh_into, 0);
        assert_eq!(dense.vector_ptrs, vec![eigh.v.data().as_ptr() as usize]);
    }

    let tensor = one_sector_matrix(vec![2.0_f64, 0.0, 0.0, 3.0]);
    let space = tensor.fusion_space().clone();
    check(&tensor);
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
                .map(|&value| Complex32::new(value as f32, 0.0))
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
                .map(|&value| Complex64::new(value, 0.0))
                .collect(),
            space,
        )
        .unwrap(),
    );
}

#[test]
fn eigh_fallback_keeps_owned_vectors_until_the_final_scatter_for_every_dtype() {
    // What: the fallback's executor V is the same allocation immediately before
    // symtn's required structural scatter, not the final factor allocation.
    fn check<D: crate::factorize::FactorScalar>(tensor: &TestTensor<D>) {
        let rule = Arc::new(Z2FusionRule);
        let bound = bound_tensor(Arc::clone(&rule), tensor);
        let adjoint = bound.space().adjoint_view().unwrap();
        assert!(crate::factorize::compact_factor_plan_for_test(&adjoint)
            .unwrap()
            .is_none());
        let input = BoundDynamicTensorRef::try_new(&adjoint, bound.data()).unwrap();
        let mut dense = ScriptedExecutor::<RejectEighInto>::default();
        let eigh_vectors_rec = crate::probe::record();

        let eigh = eigh_full_dyn(&mut dense, &input, HermitianTol::DEFAULT).unwrap();
        let before_scatter = eigh_vectors_rec.eigh_owned_vectors();

        assert!(!before_scatter.is_empty());
        assert_eq!(dense.counts().eigh_into, 0);
        assert_eq!(dense.vector_ptrs, before_scatter);
        assert_eq!(dense.counts().eigh, before_scatter.len());
        assert!(Arc::ptr_eq(
            input.space().provider_arc(),
            eigh.v().space().provider_arc()
        ));
        assert!(before_scatter
            .iter()
            .all(|&pointer| pointer != eigh.v().data().as_ptr() as usize));
    }

    let (real, complex) = unequal_fallback_eigh_fixtures();
    check(&real);
    let space = real.fusion_space().clone();
    check(
        &TestTensor::<f32>::from_vec_with_fusion_space(
            real.data().iter().map(|&value| value as f32).collect(),
            space.clone(),
        )
        .unwrap(),
    );
    check(
        &TestTensor::<Complex32>::from_vec_with_fusion_space(
            complex
                .data()
                .iter()
                .map(|value| Complex32::new(value.re as f32, value.im as f32))
                .collect(),
            complex.fusion_space().clone(),
        )
        .unwrap(),
    );
    check(&complex);
}

#[test]
fn eigh_zero_only_input_normalizes_to_an_empty_factorization_result() {
    // What: a zero-only endomorphism has no phantom output sector or spectrum
    // entry and does not invoke the dense executor.
    let tensor = rectangular_svd_tensor(0, 0);
    let mut dense = ScriptedExecutor::new(RejectExecutorCalls);

    let eigh = eigh_full(
        &mut dense,
        &bound_tensor_ref!(Arc::new(Z2FusionRule), &tensor),
    )
    .unwrap();

    assert!(eigh.v.data().is_empty());
    assert!(eigh.d.data().is_empty());
    assert!(eigh.eigenvalues.is_empty());
    assert!(eigh.v.space().space().homspace().domain().legs()[0]
        .sectors()
        .is_empty());
    assert!(eigh.d.space().space().homspace().codomain().legs()[0]
        .sectors()
        .is_empty());
}

#[test]
fn compact_owned_eigh_preserves_eigh_into_output_precedence() {
    let tensor = one_sector_matrix(vec![2.0, 0.0, 0.0, 3.0]);
    let bound = bound_tensor(Arc::new(Z2FusionRule), &tensor);
    let input = bound.as_ref();
    let check = |outputs: Vec<DenseTensor>, expected: DenseError| {
        let mut dense = ScriptedExecutor::new(FailAfterObservingEighInput {
            outputs: Some(outputs),
            ..Default::default()
        });
        let error = eigh_full(&mut dense, &input).unwrap_err();
        assert_eq!(error, OperationError::Dense(expected));
    };

    check(
        vec![f64_eigh_outputs(2).remove(0)],
        arity_mismatch("eigh_into", 2, 1),
    );
    check(
        f64_eigh_outputs(1),
        DenseError::ShapeMismatch {
            op: "eigh_into",
            expected: vec![2],
            actual: vec![1],
        },
    );
    let mut outputs = f64_eigh_outputs(2);
    outputs[1] = f64_eigh_outputs(1).remove(1);
    check(
        outputs,
        DenseError::ShapeMismatch {
            op: "eigh_into",
            expected: vec![2, 2],
            actual: vec![1, 1],
        },
    );
    let mut outputs = f64_eigh_outputs(2);
    outputs[0] = c64_eigh_outputs(2).remove(1);
    let expected = outputs[0].as_f64_slice().unwrap_err();
    assert!(matches!(
        expected,
        DenseError::DTypeMismatch {
            actual: symtn_dense::DenseDType::C64,
            ..
        }
    ));
    let mut dense = ScriptedExecutor::new(FailAfterObservingEighInput {
        outputs: Some(outputs),
        ..Default::default()
    });
    let error = eigh_full(&mut dense, &input).unwrap_err();
    assert!(matches!(error, OperationError::Dense(actual) if actual == expected));

    let mut outputs = f64_eigh_outputs(2);
    outputs[1] = c64_eigh_outputs(2).remove(1);
    let expected = outputs[1].as_f64_slice().unwrap_err();
    assert!(matches!(
        expected,
        DenseError::DTypeMismatch {
            actual: symtn_dense::DenseDType::C64,
            ..
        }
    ));
    let mut dense = ScriptedExecutor::new(FailAfterObservingEighInput {
        outputs: Some(outputs),
        ..Default::default()
    });
    let error = eigh_full(&mut dense, &input).unwrap_err();
    assert!(matches!(error, OperationError::Dense(actual) if actual == expected));
}

#[test]
fn eigh_full_does_not_relabel_the_lowest_u1_sector() {
    // What: full EIGH preserves the eigen equation and factor orientation at the lowest U(1) label.
    let rule = U1FusionRule;
    let minimum = U1Irrep::new(i32::MIN + 1).sector_id();
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(minimum, 2)], false)]),
        FusionProductSpace::new([SectorLeg::new([(minimum, 2)], false)]),
    );
    let space =
        DynamicFusionMapSpace::from_degeneracy_shapes(&rule, homspace, [vec![2, 2]]).unwrap();
    let tensor = TestTensor::from_vec_with_fusion_space(vec![4.0, 1.0, 1.0, 3.0], space).unwrap();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let result = eigh_full(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor)).unwrap();

    assert_eq!(result.v.tensor().dims(), &[2, 2]);
    assert_eq!(result.d.tensor().dims(), &[2, 2]);
    let values = &result.eigenvalues[0].values;
    assert_eq!(values.len(), 2);
    for (col, &value) in values.iter().enumerate() {
        for row in 0..2 {
            let lhs = (0..2)
                .map(|index| tensor.data()[row + 2 * index] * result.v.data()[index + 2 * col])
                .sum::<f64>();
            let rhs = result.v.data()[row + 2 * col] * value;
            assert!((lhs - rhs).abs() < 1e-10);
        }
    }
}

#[test]
fn eigh_full_satisfies_the_eigen_equation() {
    let rule = SU2FusionRule;
    let tensor = hermitian_test_tensor(
        &rule,
        &[
            SU2Irrep::from_twice_spin(0).sector_id(),
            SU2Irrep::from_twice_spin(1).sector_id(),
        ],
    );
    let mut dense_executor = symtn_dense::DefaultDenseExecutor::new();
    let eigh = eigh_full(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap();

    for entry in &eigh.eigenvalues {
        for pair in entry.values.windows(2) {
            assert!(pair[0] <= pair[1], "eigenvalues must be stored ascending");
        }
    }
    assert_eigen_equation(&rule, &tensor, &eigh.v, &eigh.d);
}

fn assert_eigh_reconstructs_rule<R>(rule: &R, sectors: &[SectorId])
where
    R: Clone + MultiplicityFreeRigidSymbols<Scalar = f64> + TreeTransformRuleCacheKey,
{
    let tensor = hermitian_test_tensor(rule, sectors);
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let eigh = eigh_full(
        &mut dense,
        &bound_tensor_ref!(Arc::new((*rule).clone()), &tensor),
    )
    .unwrap();
    assert_factor_layout_matches_per_tree_shapes(eigh.v.space());
    assert_factor_layout_matches_per_tree_shapes(eigh.d.space());
    for entry in &eigh.eigenvalues {
        for pair in entry.values.windows(2) {
            assert!(pair[0] <= pair[1]);
        }
    }
    assert_eigen_equation(rule, &tensor, &eigh.v, &eigh.d);
}

#[test]
fn eigh_reconstructs_u1_fermion_parity_and_product_rules() {
    // What: direct EIGH preserves abelian, fermionic, product, and nested sector identities.
    assert_eigh_reconstructs_rule(
        &U1FusionRule,
        &[
            U1Irrep::new(-1).sector_id(),
            U1Irrep::new(0).sector_id(),
            U1Irrep::new(1).sector_id(),
        ],
    );
    assert_eigh_reconstructs_rule(
        &FermionParityFusionRule,
        &[SectorId::new(0), SectorId::new(1)],
    );
    let product = product_fusion_rule(FermionParityFusionRule, U1FusionRule);
    let product_sectors = [
        product.encode_component_ids(SectorId::new(0), U1Irrep::new(0).sector_id()),
        product.encode_component_ids(SectorId::new(1), U1Irrep::new(1).sector_id()),
    ];
    assert_eigh_reconstructs_rule(&product, &product_sectors);

    let nested = product_fusion_rule(product, SU2FusionRule);
    let nested_sectors = [
        nested.encode_component_ids(product_sectors[0], SU2Irrep::from_twice_spin(0).sector_id()),
        nested.encode_component_ids(product_sectors[1], SU2Irrep::from_twice_spin(1).sector_id()),
    ];
    let eigh_copy_rec = crate::probe::record();
    assert_eigh_reconstructs_rule(&nested, &nested_sectors);
    assert_eq!(
        eigh_copy_rec.eigh_copy(),
        crate::probe::EighCopyProbe::default()
    );
}

#[test]
fn eigh_c64_reconstructs_multi_sector_hermitian_input_and_fixes_gauge() {
    use num_complex::Complex64;

    // What: complex direct vectors reconstruct every sector and use the canonical phase gauge.
    let rule = Z2FusionRule;
    let real = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let tensor = TestTensor::<Complex64>::from_vec_with_fusion_space(
        real.data()
            .iter()
            .map(|&value| Complex64::new(value, 0.0))
            .collect(),
        real.fusion_space().clone(),
    )
    .unwrap();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let eigh = eigh_full(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor)).unwrap();
    let input_regions = tensor
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .unwrap();
    let vector_regions = eigh
        .v
        .structure()
        .coupled_sector_regions(2)
        .unwrap()
        .unwrap();
    for input_region in input_regions.iter() {
        let sector = input_region.coupled();
        let vector_region = vector_regions
            .iter()
            .find(|region| region.coupled() == sector)
            .unwrap();
        let values = &eigh
            .eigenvalues
            .iter()
            .find(|entry| entry.sector == sector)
            .unwrap()
            .values;
        let n = input_region.rows();
        for bond in 0..n {
            let column = &eigh.v.data()[vector_region.range().start + bond * n
                ..vector_region.range().start + (bond + 1) * n];
            let pivot = column
                .iter()
                .max_by(|a, b| a.norm_sqr().partial_cmp(&b.norm_sqr()).unwrap())
                .unwrap();
            assert!(pivot.im.abs() < 1e-12);
            assert!(pivot.re >= 0.0);
        }
        for col in 0..n {
            for row in 0..n {
                let reconstructed = (0..n)
                    .map(|bond| {
                        eigh.v.data()[vector_region.range().start + row + n * bond]
                            * values[bond]
                            * eigh.v.data()[vector_region.range().start + col + n * bond].conj()
                    })
                    .sum::<Complex64>();
                let expected = tensor.data()[input_region.range().start + row + n * col];
                assert!((reconstructed - expected).norm() < 1e-9);
            }
        }
    }
}

#[test]
fn eigh_direct_regions_publish_vectors_by_factor_order_and_spectra_by_source_order() {
    // What: reversed source spans leave spectra in source traversal while V is
    // interpreted through the published factor's sector regions.
    fn check<D: crate::factorize::FactorScalar>(tensor: &TestTensor<D>) {
        let rule = Arc::new(Z2FusionRule);
        let bound = bound_tensor(Arc::clone(&rule), tensor);
        let plan = crate::factorize::compact_factor_plan_for_test(bound.space())
            .unwrap()
            .unwrap();
        assert!(crate::factorize::compact_factor_plan_routes_for_test(&plan)
            .iter()
            .any(|route| {
                let (source, left, _) = route.factor_regions_for_test();
                left.is_some_and(|left| left != source)
            }));

        let mut dense = symtn_dense::DefaultDenseExecutor::new();
        let eigh = eigh_full(&mut dense, &bound.as_ref()).unwrap();
        let source_regions = tensor
            .structure()
            .coupled_sector_regions(1)
            .unwrap()
            .unwrap();
        assert_eq!(
            eigh.eigenvalues
                .iter()
                .map(|entry| entry.sector)
                .collect::<Vec<_>>(),
            source_regions
                .iter()
                .map(|region| region.coupled())
                .collect::<Vec<_>>()
        );
        let vector_regions = eigh
            .v
            .structure()
            .coupled_sector_regions(1)
            .unwrap()
            .unwrap();
        for source in source_regions.iter() {
            let sector = source.coupled();
            let vector = vector_regions
                .iter()
                .find(|region| region.coupled() == sector)
                .unwrap();
            let values = &eigh
                .eigenvalues
                .iter()
                .find(|entry| entry.sector == sector)
                .unwrap()
                .values;
            let n = source.rows();
            for col in 0..n {
                for row in 0..n {
                    let reconstructed = (0..n)
                        .map(|bond| {
                            eigh.v.data()[vector.range().start + row + n * bond].widen_complex()
                                * values[bond]
                                * eigh.v.data()[vector.range().start + col + n * bond]
                                    .widen_complex()
                                    .conj()
                        })
                        .sum::<Complex64>();
                    let expected =
                        tensor.data()[source.range().start + row + n * col].widen_complex();
                    assert!(
                        (reconstructed - expected).norm() < 2e-5,
                        "sector {sector:?}, entry ({row}, {col})"
                    );
                }
            }
        }
    }

    let rule = Z2FusionRule;
    let source = mixed_rectangular_tensor((3, 3), (2, 2));
    let mut data = source.data().to_vec();
    for region in source
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap()
        .iter()
    {
        let n = region.rows();
        for col in 0..n {
            for row in 0..n {
                data[region.range().start + row + n * col] = if row == col {
                    (row + 2) as f64
                } else {
                    (row + col + 1) as f64 * 0.25
                };
            }
        }
    }
    let real = reversed_complete_grid_copy(
        &rule,
        &TestTensor::<f64>::from_vec_with_fusion_space(data, source.fusion_space().clone())
            .unwrap(),
    );
    check(&real);
    check(
        &TestTensor::<Complex64>::from_vec_with_fusion_space(
            real.data()
                .iter()
                .map(|&value| Complex64::new(value, 0.0))
                .collect(),
            real.fusion_space().clone(),
        )
        .unwrap(),
    );
}

#[test]
fn eig_full_satisfies_the_eigen_equation_for_real_input() {
    use num_complex::Complex64;
    let rule = Z2FusionRule;
    // Non-symmetric endomorphism.
    let tensor = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let mut dense_executor = symtn_dense::DefaultDenseExecutor::new();
    let eig = eig_full(
        &mut dense_executor,
        &bound_tensor_ref!(Arc::new(rule), &tensor),
    )
    .unwrap();

    for entry in &eig.eigenvalues {
        for pair in entry.values.windows(2) {
            assert!((pair[0].re, pair[0].im) <= (pair[1].re, pair[1].im));
        }
    }

    // Promote t to complex (same space => same layout => elementwise cast).
    let tensor_c = TestTensor::<Complex64>::from_vec_with_fusion_space(
        tensor
            .data()
            .iter()
            .map(|&value| Complex64::new(value, 0.0))
            .collect(),
        tensor.fusion_space().clone(),
    )
    .unwrap();

    let mut context = TensorContractFusionExecutionContext::<Complex64, RuleIdentity>::default();
    let mut tv = TestTensor::<Complex64>::from_vec_with_fusion_space(
        vec![Complex64::new(0.0, 0.0); eig.v.data().len()],
        eig.v.fusion_space().clone(),
    )
    .unwrap();
    context_contract_into(
        &mut context,
        &rule,
        &mut tv,
        &tensor_c,
        &eig.v,
        TensorContractSpec::new(&[2, 3], &[0, 1], OutputAxisOrder::from_axes(&[0, 1, 2])),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap();
    let mut vd = TestTensor::<Complex64>::from_vec_with_fusion_space(
        vec![Complex64::new(0.0, 0.0); eig.v.data().len()],
        eig.v.fusion_space().clone(),
    )
    .unwrap();
    context_contract_into(
        &mut context,
        &rule,
        &mut vd,
        &eig.v,
        &eig.d,
        TensorContractSpec::new(&[2], &[0], OutputAxisOrder::from_axes(&[0, 1, 2])),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.0),
    )
    .unwrap();
    for (index, (lhs, rhs)) in tv.data().iter().zip(vd.data()).enumerate() {
        assert!(
            (lhs - rhs).norm() < 1e-8,
            "eigen equation violated at raw position {index}: {lhs} != {rhs}"
        );
    }
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
fn eigh_refuses_a_block_that_stays_hermitian_after_the_column_swap() {
    // What: canonical sector blocks [[2,1],[1,2]] (spectrum {3, 1}) stored
    // with reversed columns read as [[1,2],[2,1]], which is Hermitian with
    // spectrum {3, -1}. The Hermitian preflight cannot catch it; only the
    // stacking guard does.
    let rule = Z2FusionRule;
    let leg = || SectorLeg::new([(SectorId::new(0), 1), (SectorId::new(1), 1)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let key_count = homspace.fusion_tree_keys(&rule).len();
    let space =
        DynamicFusionMapSpace::from_degeneracy_shapes(&rule, homspace, vec![vec![1; 4]; key_count])
            .unwrap();
    let canonical = TestTensor::<f64>::from_block_fn_with_fusion_space(space, 0.0, |key, _| {
        let BlockKey::FusionTree(tree) = key else {
            unreachable!("fusion-tree blocks")
        };
        if tree.codomain_tree() == tree.domain_tree() {
            2.0
        } else {
            1.0
        }
    })
    .unwrap();
    let swapped = mis_stacked_endomorphism_copy(&rule, &canonical);
    let provider = Arc::new(rule);
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let reference = eigh_vals(
        &mut dense,
        &bound_tensor_ref!(Arc::clone(&provider), &canonical),
    )
    .unwrap();
    assert_eq!(reference.len(), 2);
    for spectrum in &reference {
        assert_eq!(spectrum.values.len(), 2);
        assert!((spectrum.values[0] - 1.0).abs() < 1e-12, "{reference:?}");
        assert!((spectrum.values[1] - 3.0).abs() < 1e-12, "{reference:?}");
    }
    let swapped = bound_tensor(provider, &swapped);
    assert_stacking_refusal(eigh_vals(&mut dense, &swapped.as_ref()), "eigh_vals ");
    assert_stacking_refusal(eigh_full(&mut dense, &swapped.as_ref()), "eigh_full ");
}

#[test]
fn scripted_default_eig_vals_reaches_the_forwarded_eig() {
    // What: the trait's default `eig_vals` calls `eig`; with `EigVals` at
    // `Default` and `Eig` at `Forward`, the scripted executor routes it back
    // through its own `eig`, so the values come from the wrapped backend and
    // both entries are counted (as the hand-written doubles did).
    let mut dense = ScriptedExecutor::<CountingDense>::default();
    let data = [2.0_f64, 0.0, 0.0, -1.0];
    let input = DenseRead::F64(symtn_dense::DenseView::new(&data, &[2, 2], &[1, 2], 0).unwrap());
    let values = dense.eig_vals(input).unwrap();
    assert_eq!(values.shape(), [2]);
    assert_eq!(dense.counts().eig_vals, 1);
    assert_eq!(dense.counts().eig, 1);
}

#[test]
fn eigh_admits_gemm_built_gram_and_congruence_matrices_at_n_1000() {
    // What (#1987): `AᵀA` and `Aᵀ(HA)` (`H = B + Bᵀ`) formed by the dense
    // backend's gemm are admitted by the default tolerance. Independent
    // oracle: their relative anti-Hermitian residual by direct summation,
    // which must sit below `eps^(3/4)`.
    let n = 1000;
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut random = || {
        (0..n * n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            })
            .collect::<Vec<f64>>()
    };
    let (a, b) = (random(), random());
    let product = |lhs: &[f64], lhs_strides: [usize; 2], rhs: &[f64]| {
        let mut out = vec![0.0; n * n];
        symtn_dense::DefaultDenseExecutor::new()
            .matmul_into(
                symtn_dense::DenseWrite::F64(
                    symtn_dense::DenseViewMut::new(&mut out, &[n, n], &[1, n], 0).unwrap(),
                ),
                symtn_dense::DenseRead::F64(
                    symtn_dense::DenseView::new(lhs, &[n, n], &lhs_strides, 0).unwrap(),
                ),
                symtn_dense::DenseRead::F64(
                    symtn_dense::DenseView::new(rhs, &[n, n], &[1, n], 0).unwrap(),
                ),
            )
            .unwrap();
        out
    };
    let transposed = [n, 1];
    let h: Vec<f64> = (0..n * n)
        .map(|index| b[index] + b[(index % n) * n + index / n])
        .collect();
    let gram = product(&a, transposed, &a);
    let congruence = product(&a, transposed, &product(&h, [1, n], &a));
    for matrix in [gram, congruence] {
        let (mut residual, mut norm) = (0.0_f64, 0.0_f64);
        for col in 0..n {
            for row in 0..n {
                let value = matrix[row + n * col];
                let half = 0.5 * (value - matrix[col + n * row]);
                residual += half * half;
                norm += value * value;
            }
        }
        let relative = residual.sqrt() / norm.sqrt();
        assert!(relative < f64::EPSILON.powf(0.75), "{relative:e}");
        assert_eigh_preflight(&one_sector_rectangular_matrix(matrix, n, n), true);
    }
}

#[test]
fn ascending_provider_eigh_order_is_published_without_a_permutation() {
    // What: the LAPACK/faer `syev` spectrum is already ascending, so full and
    // values-only EIGH publish it after the O(n) check alone: no sort, no
    // value copy, no eigenvector column permutation (#1985).
    let rule = Z2FusionRule;
    let tensor = hermitian_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let permutations = crate::probe::record();
    let full = eigh_full(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor)).unwrap();
    let values = eigh_vals(&mut dense, &bound_tensor_ref!(Arc::new(rule), &tensor)).unwrap();
    assert_eq!(
        permutations.get(crate::probe::Event::EigOrderPermutation),
        0
    );
    // Full and values-only EIGH may use different LAPACK drivers (Accelerate
    // does), which agree to rounding, not bitwise.
    assert_eq!(full.eigenvalues.len(), values.len());
    for (full, values) in full.eigenvalues.iter().zip(&values) {
        assert_eq!(full.sector, values.sector);
        assert_eq!(full.values.len(), values.values.len());
        assert!(full.values.windows(2).all(|pair| pair[0] <= pair[1]));
        assert!(values.values.windows(2).all(|pair| pair[0] <= pair[1]));
        let norm = full.values.iter().map(|x| x * x).sum::<f64>().sqrt();
        let tol = 64.0 * full.values.len() as f64 * f64::EPSILON * norm.max(1.0);
        for (a, b) in full.values.iter().zip(&values.values) {
            assert!((a - b).abs() <= tol, "{a} vs {b}");
        }
    }
    assert!(full
        .eigenvalues
        .iter()
        .any(|spectrum| spectrum.values.iter().any(|&x| x < 0.0)
            && spectrum.values.iter().any(|&x| x > 0.0)));
}

#[test]
fn eigenvalue_order_authority_is_stable_and_skips_sorted_input() {
    // What: the order authority returns the stable sorting permutation, and
    // `false` with the identity for already-ordered input (#1985).
    use crate::factorize::{ascending_eigh_order, lexicographic_eig_order};
    let mut order = [9; 4];
    let permutations = crate::probe::record();
    assert!(!ascending_eigh_order(&[-3.0, -1.0, -1.0, 2.0], &mut order));
    assert_eq!(order, [0, 1, 2, 3]);
    assert_eq!(
        permutations.get(crate::probe::Event::EigOrderPermutation),
        0
    );
    // Ties keep stored order.
    assert!(ascending_eigh_order(&[2.0, -1.0, 0.5, -1.0], &mut order));
    assert_eq!(order, [1, 3, 2, 0]);
    let c = Complex64::new;
    // A conjugate pair as `geev` returns it (`+` first), equal real parts,
    // and a value of larger magnitude that sorts first.
    let values = [c(1.0, 2.0), c(1.0, -2.0), c(-3.0, 0.0), c(1.0, 2.0)];
    assert!(lexicographic_eig_order(&values, &mut order));
    assert_eq!(order, [2, 1, 0, 3]);
    assert_eq!(
        permutations.get(crate::probe::Event::EigOrderPermutation),
        2
    );
    let mut order = [9; 3];
    assert!(!lexicographic_eig_order(
        &[c(-1.0, 5.0), c(0.0, -1.0), c(0.0, 1.0)],
        &mut order
    ));
    assert_eq!(order, [0, 1, 2]);
}
