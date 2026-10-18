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

fn assert_generic_factor_close<R>(
    actual: &BoundDynFactor<R, f64>,
    expected: &BoundDynFactor<R, f64>,
) {
    assert_eq!(
        actual.space().space().homspace(),
        expected.space().space().homspace()
    );
    assert_eq!(actual.data().len(), expected.data().len());
    for (&actual, &expected) in actual.data().iter().zip(expected.data()) {
        assert!((actual - expected).abs() < 1.0e-12);
    }
}

#[test]
fn provider_neutral_generic_compact_factorizations_remain_covered() {
    let (space, data) = generic_factorization_input();
    let (provider, space) = bind_checked_layout(&space);
    let input = BoundDynamicTensorRef::try_new(&space, &data).unwrap();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    assert!(!svd_vals_dyn_checked_generic(&mut dense, &input)
        .unwrap()
        .is_empty());
    let (u, vh, values) =
        svd_compact_factors_with_spectrum_dyn_checked_generic(&mut dense, &input).unwrap();
    assert!(!values.is_empty());
    assert!(Arc::ptr_eq(u.space().provider_arc(), &provider));
    assert!(Arc::ptr_eq(vh.space().provider_arc(), &provider));
    qr_compact_dyn_checked_generic(&mut dense, &input).unwrap();
    lq_compact_dyn_checked_generic(&mut dense, &input).unwrap();
}

#[test]
fn provider_neutral_generic_factorizations_keep_the_strided_fallback() {
    let (canonical_space, canonical_data) = generic_factorization_input();
    let (padded_space, padded_data) =
        padded_generic_factorization_input(&canonical_space, &canonical_data);
    let (_, canonical_space) = bind_checked_layout(&canonical_space);
    let (_, padded_space) = bind_checked_layout(&padded_space);
    let canonical = BoundDynamicTensorRef::try_new(&canonical_space, &canonical_data).unwrap();
    let padded = BoundDynamicTensorRef::try_new(&padded_space, &padded_data).unwrap();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let canonical_values = svd_vals_dyn_checked_generic(&mut dense, &canonical).unwrap();
    let padded_values = svd_vals_dyn_checked_generic(&mut dense, &padded).unwrap();
    assert_real_spectra_close(&padded_values, &canonical_values);

    let canonical_svd =
        svd_compact_factors_with_spectrum_dyn_checked_generic(&mut dense, &canonical).unwrap();
    let padded_svd =
        svd_compact_factors_with_spectrum_dyn_checked_generic(&mut dense, &padded).unwrap();
    assert_generic_factor_close(&padded_svd.0, &canonical_svd.0);
    assert_generic_factor_close(&padded_svd.1, &canonical_svd.1);
    assert_real_spectra_close(&padded_svd.2, &canonical_svd.2);

    let canonical_qr = qr_compact_dyn_checked_generic(&mut dense, &canonical).unwrap();
    let padded_qr = qr_compact_dyn_checked_generic(&mut dense, &padded).unwrap();
    assert_generic_factor_close(&padded_qr.q, &canonical_qr.q);
    assert_generic_factor_close(&padded_qr.r, &canonical_qr.r);

    let canonical_lq = lq_compact_dyn_checked_generic(&mut dense, &canonical).unwrap();
    let padded_lq = lq_compact_dyn_checked_generic(&mut dense, &padded).unwrap();
    assert_generic_factor_close(&padded_lq.l, &canonical_lq.l);
    assert_generic_factor_close(&padded_lq.q, &canonical_lq.q);
}

#[test]
fn generic_pair_publication_keeps_reordered_tree_scatter_fallback() {
    let (canonical_space, canonical_data) = generic_factorization_input();
    let (reordered_space, reordered_data) =
        expert_generic_factorization_input(&canonical_space, &canonical_data, true);
    let (_, reordered_space) = bind_checked_layout(&reordered_space);
    let reordered = BoundDynamicTensorRef::try_new(&reordered_space, &reordered_data).unwrap();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();

    let generic_pair_rec = crate::probe::record();
    let actual_qr = qr_compact_dyn_checked_generic(&mut dense, &reordered).unwrap();
    assert!(!actual_qr.q.data().is_empty());
    assert!(!actual_qr.r.data().is_empty());
    let probe = generic_pair_rec.generic_pair_publication();
    assert_eq!(
        (probe.canonical_publications, probe.fallback_publications),
        (0, 1)
    );
    assert!(probe.left_scattered_elements > 0);
    assert!(probe.right_scattered_elements > 0);
    assert_eq!(
        probe.left_scattered_elements + probe.right_scattered_elements,
        actual_qr.q.data().len() + actual_qr.r.data().len()
    );

    let generic_pair_rec = crate::probe::record();
    let actual_lq = lq_compact_dyn_checked_generic(&mut dense, &reordered).unwrap();
    assert!(!actual_lq.l.data().is_empty());
    assert!(!actual_lq.q.data().is_empty());
    let probe = generic_pair_rec.generic_pair_publication();
    assert_eq!(
        (probe.canonical_publications, probe.fallback_publications),
        (0, 1)
    );
    assert!(probe.left_scattered_elements > 0);
    assert!(probe.right_scattered_elements > 0);
    assert_eq!(
        probe.left_scattered_elements + probe.right_scattered_elements,
        actual_lq.l.data().len() + actual_lq.q.data().len()
    );
}

#[test]
fn checked_generic_pair_publication_reuses_one_sector_owners() {
    let (source, data) = one_sector_generic_factorization_input();
    let (provider, checked) = bind_checked_only(&source);
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let generic_pair_rec = crate::probe::record();

    let Qr { q: left, r: right } = qr_compact_dyn_checked_generic(&mut dense, &input).unwrap();

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
    let (source, data) = one_sector_generic_factorization_input();
    let (padded_space, padded_data) = padded_generic_factorization_input(&source, &data);
    let (_, padded_space) = bind_checked_layout(&padded_space);
    let padded = BoundDynamicTensorRef::try_new(&padded_space, &padded_data).unwrap();
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

    qr_compact_dyn_checked_generic(&mut dense, &padded).unwrap();
    lq_compact_dyn_checked_generic(&mut dense, &padded).unwrap();
    qr_compact_dyn_checked_generic(&mut dense, &checked_input).unwrap();
    svd_compact_dyn_checked_generic(&mut dense, &checked_input).unwrap();
    lq_compact_dyn_checked_generic(&mut dense, &checked_input).unwrap();
    qr_full_dyn_checked_generic(&mut dense, &checked_input).unwrap();
    lq_full_dyn_checked_generic(&mut dense, &checked_input).unwrap();

    // Full QR/LQ publish each side through the side-aware builder (#1524).
    let one_sided = one_sided_rec.one_sided_publication();
    assert_eq!(
        (
            one_sided.canonical_publications,
            one_sided.fallback_publications
        ),
        (4, 0)
    );
    let probe = generic_pair_rec.generic_pair_publication();
    assert_eq!(probe.canonical_publications, 5);
    assert_eq!(probe.fallback_publications, 0);
    assert_eq!((probe.left_owner_reused, probe.right_owner_reused), (5, 5));
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
    let (provider, space, data) = checked_svd_truncation_input::<D>(complex);
    let input = BoundDynamicTensorRef::try_new(&space, &data).unwrap();
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let generic_pair_rec = crate::probe::record();
    let one_sided_rec = crate::probe::record();

    let qr = qr_compact_dyn_checked_generic(&mut dense, &input).unwrap();
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
    svd_compact_dyn_checked_generic(&mut dense, &input).unwrap();
    let lq = lq_compact_dyn_checked_generic(&mut dense, &input).unwrap();
    assert_pair_reconstructs_checked_literal(&lq.l, &lq.q, complex);
    let full_qr = qr_full_dyn_checked_generic(&mut dense, &input).unwrap();
    assert_pair_reconstructs_checked_literal(&full_qr.q, &full_qr.r, complex);
    let full_lq = lq_full_dyn_checked_generic(&mut dense, &input).unwrap();
    assert_pair_reconstructs_checked_literal(&full_lq.l, &full_lq.q, complex);

    // Full QR/LQ publish each side through the side-aware builder (#1524).
    let one_sided = one_sided_rec.one_sided_publication();
    assert_eq!(
        (
            one_sided.canonical_publications,
            one_sided.fallback_publications
        ),
        (4, 0)
    );
    let probe = generic_pair_rec.generic_pair_publication();
    assert_eq!(probe.canonical_publications, 3);
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

// Checked compact-QR provider sequence for `generic_factorization_input`: the
// compact plan issues no preflight query, the left (Q) space enumerates once
// (calls 1-3), then the right (R) space enumerates once (calls 4-6). Before
// the paired builder enumerated each side once, each side enumerated twice
// (left keys 1-3, left space 4-6, right keys 7-9, right space 10-12) and the
// first right-side call was the seventh.
const COMPACT_PAIR_LEFT_LAST_CALL: usize = 3;

const COMPACT_PAIR_RIGHT_FIRST_CALL: usize = 4;

const COMPACT_PAIR_RIGHT_LAST_CALL: usize = 6;

#[test]
#[expect(
    clippy::arc_with_non_send_sync,
    reason = "the checked Generic API requires Arc identity while Cell is a single-threaded call spy"
)]
fn checked_generic_compact_pair_builder_failure_preserves_provider_context() {
    // What: the last call of the left enumeration and the first call of the
    // right enumeration propagate their exact provider error without
    // publishing either factor or touching the input.
    let (source, data) = generic_factorization_input();
    for fail_at in [
        COMPACT_PAIR_LEFT_LAST_CALL,
        COMPACT_PAIR_RIGHT_FIRST_CALL,
        COMPACT_PAIR_RIGHT_LAST_CALL,
    ] {
        let provider = Arc::new(LateGenericSpy {
            rule: FactorGenericRule,
            fail_at,
            calls: Cell::new(0),
            identity: RuleIdentity::new_unique::<LateGenericSpy>(),
        });
        let checked = bind_to_spy(&source, &provider);
        let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
        let mut dense = ScriptedExecutor::<CountingDense>::default();
        let result = qr_compact_dyn_checked_generic(&mut dense, &input);
        assert!(matches!(
            result,
            Err(CheckedGenericFactorPlanError::Provider(LateGenericError(call))) if call == fail_at
        ));
        assert_eq!(provider.calls.get(), fail_at);
        assert_eq!(dense.counts().qr, 2);
        assert_eq!(input.data(), data);
    }

    let provider = Arc::new(LateGenericSpy {
        rule: FactorGenericRule,
        fail_at: usize::MAX,
        calls: Cell::new(0),
        identity: RuleIdentity::new_unique::<LateGenericSpy>(),
    });
    let checked = bind_to_spy(&source, &provider);
    let input = BoundDynamicTensorRef::try_new(&checked, &data).unwrap();
    let Qr { q, r } =
        qr_compact_dyn_checked_generic(&mut ScriptedExecutor::<CountingDense>::default(), &input)
            .unwrap();
    assert_eq!(provider.calls.get(), COMPACT_PAIR_RIGHT_LAST_CALL);
    assert_eq!(checked_enumeration_calls(&q), COMPACT_PAIR_LEFT_LAST_CALL);
    assert_eq!(
        checked_enumeration_calls(&r),
        COMPACT_PAIR_RIGHT_LAST_CALL - COMPACT_PAIR_LEFT_LAST_CALL
    );
}
