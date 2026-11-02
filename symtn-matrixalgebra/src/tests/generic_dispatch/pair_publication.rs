use super::*;

fn one_sector_generic_factorization_input(
) -> (BoundDynamicFusionMapSpace<FactorGenericRule>, Vec<f64>) {
    let provider = Arc::new(FactorGenericRule);
    let vacuum = SectorId::new(0);
    let x = SectorId::new(1);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(x, 2)], false),
            SectorLeg::new([(x, 1)], false),
        ]),
        FusionProductSpace::new([
            SectorLeg::new([(x, 3)], false),
            SectorLeg::new([(vacuum, 1)], false),
        ]),
    );
    let space =
        BoundDynamicFusionMapSpace::from_final_homspace_generic(provider, homspace).unwrap();
    let data = (0..space.space().required_len().unwrap())
        .map(|index| 1.0 + index as f64 / 8.0)
        .collect();
    (space, data)
}

#[test]
fn provider_neutral_generic_factorizations_keep_the_strided_fallback() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: the adjoint view, a strided noncanonical source, factorizes on
    // the packed path with the source's provider; its singular values are
    // the parent's, and every compact factor pair reconstructs it.
    let (canonical_space, canonical_data) = generic_factorization_input();
    let (provider, canonical_space) = bind_checked_layout(&canonical_space);
    let view_space = canonical_space.adjoint_view().unwrap();
    let canonical = BoundDynamicTensorRef::try_new(&canonical_space, &canonical_data).unwrap();
    let view = BoundDynamicTensorRef::try_new(&view_space, &canonical_data).unwrap();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let canonical_values = svd_vals_dyn_checked_generic(&mut dense, &canonical).unwrap();
    assert!(!canonical_values.is_empty());
    let view_values = svd_vals_dyn_checked_generic(&mut dense, &view).unwrap();
    assert_real_spectra_close(&view_values, &canonical_values);

    let svd =
        svd_compact_factors_with_spectrum_dyn_checked_generic(engine, &mut dense, &view).unwrap();
    let s = generic_diagonal_factor(&svd.0, &svd.2);
    assert_compact_factors_reconstruct_input(&view, &svd.0, Some(&s), &svd.1);
    assert!(Arc::ptr_eq(svd.0.space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(svd.1.space().provider_arc(), &provider));

    let qr = qr_compact_dyn_checked_generic(engine, &mut dense, &view).unwrap();
    assert_compact_factors_reconstruct_input(&view, &qr.q, None, &qr.r);
    let lq = lq_compact_dyn_checked_generic(engine, &mut dense, &view).unwrap();
    assert_compact_factors_reconstruct_input(&view, &lq.l, None, &lq.q);
}

#[test]
fn checked_generic_pair_publication_reuses_one_sector_owners() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let (source, data) = one_sector_generic_factorization_input();
    let (provider, checked) = bind_checked_only(&source);
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let generic_pair_rec = crate::probe::record();

    let Qr { q: left, r: right } =
        qr_compact_dyn_checked_generic(engine, &mut dense, &input).unwrap();

    let probe = generic_pair_rec.generic_pair_publication();
    let blocks = left.space().space().structure().block_count()
        + right.space().space().structure().block_count();
    assert_eq!(probe.canonical_publications, 1);
    assert_eq!(probe.fallback_publications, 0);
    assert_eq!((probe.left_owner_reused, probe.right_owner_reused), (1, 1));
    assert_eq!(
        (probe.left_appended_elements, probe.right_appended_elements),
        (0, 0)
    );
    assert_eq!(
        (
            probe.left_scattered_elements,
            probe.right_scattered_elements
        ),
        (0, 0)
    );
    // The checked route validates each staged key once (one cursor event per
    // block) and commits the structure it validated, so no admitted-key
    // equality of a two-source construction (`blocks` visits, `2 * blocks`
    // events) runs.
    assert_eq!(probe.ordered_key_validation_events, blocks);
    assert_eq!(
        (probe.fallback_row_lookups, probe.fallback_col_lookups),
        (0, 0)
    );
    assert!(Arc::ptr_eq(left.space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(right.space().provider_arc(), &provider));
}

fn assert_pair_reconstructs_checked_literal<R, D>(
    left: &BoundDynFactor<R, D>,
    right: &BoundDynFactor<R, D>,
    complex: bool,
) where
    D: FactorScalar,
{
    let left_regions = left
        .space()
        .space()
        .structure()
        .coupled_sector_regions(left.space().space().nout())
        .unwrap()
        .unwrap();
    let right_regions = right
        .space()
        .space()
        .structure()
        .coupled_sector_regions(right.space().space().nout())
        .unwrap()
        .unwrap();
    for sector in [SectorId::new(0), SectorId::new(1)] {
        let left_region = left_regions
            .iter()
            .find(|region| region.coupled() == sector)
            .unwrap();
        let right_region = right_regions
            .iter()
            .find(|region| region.coupled() == sector)
            .unwrap();
        let (rows, cols, expected) = checked_svd_matrix(sector, complex);
        assert_eq!((left_region.rows(), right_region.cols()), (rows, cols));
        assert_eq!(left_region.cols(), right_region.rows());
        let kept = left_region.cols();
        for col in 0..cols {
            for row in 0..rows {
                let actual = (0..kept).fold(D::zero(), |sum, bond| {
                    sum + left.data()[left_region.range().start + row + rows * bond]
                        * right.data()[right_region.range().start + bond + kept * col]
                });
                assert!((actual.widen_complex() - expected[row + rows * col]).norm() < 1.0e-10);
            }
        }
    }
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn staged_generic_pair_callers_publish_canonical_owned_payloads() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let (source, data) = one_sector_generic_factorization_input();
    let (_, bound_space) = bind_checked_layout(&source);
    let bound = BoundDynamicTensorRef::try_new(&bound_space, &data).unwrap();
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
        identity: RuleIdentity::new_unique::<LateGenericSpy>(),
    });
    let checked = bind_to_spy(&source, &provider);
    let checked_input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let generic_pair_rec = crate::probe::record();
    let one_sided_rec = crate::probe::record();

    let mut compact_calls = 0;
    let mut full_calls = 0;
    qr_compact_dyn_checked_generic(engine, &mut dense, &bound).unwrap();
    compact_calls += 1;
    lq_compact_dyn_checked_generic(engine, &mut dense, &bound).unwrap();
    compact_calls += 1;
    qr_compact_dyn_checked_generic(engine, &mut dense, &checked_input).unwrap();
    compact_calls += 1;
    svd_compact_dyn_checked_generic(engine, &mut dense, &checked_input).unwrap();
    compact_calls += 1;
    lq_compact_dyn_checked_generic(engine, &mut dense, &checked_input).unwrap();
    compact_calls += 1;
    qr_full_dyn_checked_generic(engine, &mut dense, &checked_input).unwrap();
    full_calls += 1;
    lq_full_dyn_checked_generic(engine, &mut dense, &checked_input).unwrap();
    full_calls += 1;

    // Full QR/LQ publish each side through the side-aware builder.
    let one_sided = one_sided_rec.one_sided_publication();
    assert_eq!(
        (
            one_sided.canonical_publications,
            one_sided.fallback_publications
        ),
        (2 * full_calls, 0)
    );
    let probe = generic_pair_rec.generic_pair_publication();
    assert_eq!(probe.canonical_publications, compact_calls);
    assert_eq!(probe.fallback_publications, 0);
    assert_eq!(
        (probe.left_owner_reused, probe.right_owner_reused),
        (compact_calls, compact_calls)
    );
    assert_eq!(
        (probe.left_appended_elements, probe.right_appended_elements),
        (0, 0)
    );
    assert_eq!(
        (
            probe.left_scattered_elements,
            probe.right_scattered_elements
        ),
        (0, 0)
    );
}

fn assert_checked_generic_pair_publication_appends_multiple_literal_sectors<D>(complex: bool)
where
    D: FactorScalar,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let (provider, space, data) = checked_svd_truncation_input::<D>(complex);
    let input = BoundDynamicTensorRef::try_new(&space, &data).unwrap();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let generic_pair_rec = crate::probe::record();
    let one_sided_rec = crate::probe::record();

    let qr = qr_compact_dyn_checked_generic(engine, &mut dense, &input).unwrap();
    assert_pair_reconstructs_checked_literal(&qr.q, &qr.r, complex);
    let qr_probe = generic_pair_rec.generic_pair_publication();
    let qr_blocks = qr.q.space().space().structure().block_count()
        + qr.r.space().space().structure().block_count();
    // Not `qr_blocks` visits and `2 * qr_blocks` events: the checked route
    // does not cross-check a separate key list against its structure.
    assert_eq!(qr_probe.ordered_key_validation_events, qr_blocks);
    assert_eq!(
        (qr_probe.fallback_row_lookups, qr_probe.fallback_col_lookups),
        (0, 0)
    );
    svd_compact_dyn_checked_generic(engine, &mut dense, &input).unwrap();
    let lq = lq_compact_dyn_checked_generic(engine, &mut dense, &input).unwrap();
    assert_pair_reconstructs_checked_literal(&lq.l, &lq.q, complex);
    let full_qr = qr_full_dyn_checked_generic(engine, &mut dense, &input).unwrap();
    assert_pair_reconstructs_checked_literal(&full_qr.q, &full_qr.r, complex);
    let full_lq = lq_full_dyn_checked_generic(engine, &mut dense, &input).unwrap();
    assert_pair_reconstructs_checked_literal(&full_lq.l, &full_lq.q, complex);
    // qr, svd and lq compact publish pairs; full qr and lq publish one side
    // per output.
    let (compact_calls, full_calls) = (3, 2);

    // Full QR/LQ publish each side through the side-aware builder.
    let one_sided = one_sided_rec.one_sided_publication();
    assert_eq!(
        (
            one_sided.canonical_publications,
            one_sided.fallback_publications
        ),
        (2 * full_calls, 0)
    );
    let probe = generic_pair_rec.generic_pair_publication();
    assert_eq!(probe.canonical_publications, compact_calls);
    assert_eq!(probe.fallback_publications, 0);
    assert!(probe.left_appended_elements > 0);
    assert!(probe.right_appended_elements > 0);
    assert_eq!(
        (
            probe.left_scattered_elements,
            probe.right_scattered_elements
        ),
        (0, 0)
    );
    assert!(Arc::ptr_eq(qr.q.space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(qr.r.space().provider_arc(), &provider));
}

#[test]
fn checked_generic_pair_publication_appends_multiple_literal_sectors() {
    assert_checked_generic_pair_publication_appends_multiple_literal_sectors::<f64>(false);
    assert_checked_generic_pair_publication_appends_multiple_literal_sectors::<Complex64>(true);
}

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_compact_pair_builder_failure_preserves_provider_context() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: the compact plan issues no preflight query and enumerates the
    // left (Q) space once, then the right (R) space once; the last call of
    // the left enumeration and the first and last calls of the right one
    // propagate their exact provider error without publishing either factor
    // or touching the input. The boundaries come from each factor's own
    // single enumeration.
    let (source, data) = generic_factorization_input();
    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
        identity: RuleIdentity::new_unique::<LateGenericSpy>(),
    });
    let checked = bind_to_spy(&source, &provider);
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let Qr { q, r } = qr_compact_dyn_checked_generic(
        engine,
        &mut ScriptedExecutor::<CountingDense>::default(),
        &input,
    )
    .unwrap();
    let (left, right) = (checked_enumeration_calls(&q), checked_enumeration_calls(&r));
    assert!(left > 0 && right > 0);
    assert_eq!(provider.calls.get(), left + right);

    for fail_at in [left, left + 1, left + right] {
        let provider = Arc::new(LateGenericSpy {
            rule: FactorGenericRule,
            fail_at,
            calls: Cell::new(0),
            identity: RuleIdentity::new_unique::<LateGenericSpy>(),
        });
        let checked = bind_to_spy(&source, &provider);
        let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
        let mut dense = ScriptedExecutor::<CountingDense>::default();
        let result = qr_compact_dyn_checked_generic(engine, &mut dense, &input);
        assert!(matches!(
            result,
            Err(CheckedGenericFactorPlanError::Provider(LateGenericError(call))) if call == fail_at
        ));
        assert_eq!(provider.calls.get(), fail_at);
        assert_eq!(dense.counts().qr, 2);
        assert_eq!(input.data(), data);
    }
}
