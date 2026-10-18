use super::*;

#[test]
fn hermitian_region_validation_rejects_short_storage_without_panicking() {
    // What: the cross-crate region validator reports malformed storage as a typed structural error.
    let tensor = one_sector_matrix(vec![1.0_f64, 0.0, 0.0, 2.0]);
    let regions = tensor
        .structure()
        .coupled_sector_regions(1)
        .unwrap()
        .unwrap();

    let error = validate_hermitian_regions(&tensor.data()[..3], &regions, HermitianTol::DEFAULT)
        .unwrap_err();

    assert_eq!(
        error,
        OperationError::ElementCountMismatch {
            expected: 4,
            actual: 3,
        }
    );
}

#[test]
fn compact_factor_plan_does_not_retain_provider() {
    // What: plan construction does not retain the input space's provider.
    let tensor = rectangular_svd_tensor(19, 11);
    let provider = Arc::new(Z2FusionRule);
    let weak = Arc::downgrade(&provider);
    let bound = bound_tensor(Arc::clone(&provider), &tensor);
    let plan = crate::factorize::compact_factor_plan_for_test(bound.space())
        .unwrap()
        .unwrap();

    drop(bound);
    drop(provider);

    assert!(weak.upgrade().is_none());
    drop(plan);
}

#[test]
fn compact_factor_plan_is_identical_across_calls_on_one_space() {
    // What: rebuilding the per-call plan on the same bound space yields the
    // same routes and the same region tables (shared `Arc`s), with the first
    // plan still alive.
    let charges =
        [U1Irrep::new(-1), U1Irrep::new(0), U1Irrep::new(1)].map(|charge| charge.sector_id());
    let tensor = tsvd_test_tensor(&U1FusionRule, &charges);
    let bound = bound_tensor(Arc::new(U1FusionRule), &tensor);
    let first = crate::factorize::compact_factor_plan_for_test(bound.space())
        .unwrap()
        .unwrap();
    let second = crate::factorize::compact_factor_plan_for_test(bound.space())
        .unwrap()
        .unwrap();

    let (first_source, first_u, first_vh) =
        crate::factorize::compact_factor_plan_regions_for_test(&first);
    let (second_source, second_u, second_vh) =
        crate::factorize::compact_factor_plan_regions_for_test(&second);
    assert!(first_source.len() >= 3);
    assert!(Arc::ptr_eq(&first_source, &second_source));
    assert!(Arc::ptr_eq(&first_u, &second_u));
    assert!(Arc::ptr_eq(&first_vh, &second_vh));
    assert_eq!(
        crate::factorize::compact_factor_plan_routes_for_test(&first),
        crate::factorize::compact_factor_plan_routes_for_test(&second)
    );
}

#[test]
fn compact_factor_routes_agree_between_sorted_and_unsorted_region_tables() {
    // What: canonical factor regions are strictly sorted by coupled sector and
    // route without a map; an expert (unsorted) region table routes through
    // the map path to the same regions in the same source order.
    let charges =
        [U1Irrep::new(-1), U1Irrep::new(0), U1Irrep::new(1)].map(|charge| charge.sector_id());
    let tensor = tsvd_test_tensor(&U1FusionRule, &charges);
    let bound = bound_tensor(Arc::new(U1FusionRule), &tensor);
    let plan = crate::factorize::compact_factor_plan_for_test(bound.space())
        .unwrap()
        .unwrap();
    let (source, u, vh) = crate::factorize::compact_factor_plan_regions_for_test(&plan);
    assert!(u
        .windows(2)
        .all(|pair| pair[0].coupled() < pair[1].coupled()));
    assert!(vh
        .windows(2)
        .all(|pair| pair[0].coupled() < pair[1].coupled()));

    let mut shuffled_u = u.to_vec();
    let mut shuffled_vh = vh.to_vec();
    shuffled_u.rotate_left(2);
    shuffled_vh.reverse();
    assert!(!shuffled_u
        .windows(2)
        .all(|pair| pair[0].coupled() < pair[1].coupled()));
    let sorted =
        crate::factorize::validate_compact_factor_routes_for_test(&source, &u, &vh).unwrap();
    let unsorted = crate::factorize::validate_compact_factor_routes_for_test(
        &source,
        &shuffled_u,
        &shuffled_vh,
    )
    .unwrap();
    assert_eq!(
        sorted,
        crate::factorize::compact_factor_plan_routes_for_test(&plan)
    );
    assert_eq!(sorted.len(), unsorted.len());
    for (sorted_route, unsorted_route) in sorted.iter().zip(&unsorted) {
        let (sorted_source, sorted_left, sorted_right) = sorted_route.factor_regions_for_test();
        let (unsorted_source, unsorted_left, unsorted_right) =
            unsorted_route.factor_regions_for_test();
        assert_eq!(sorted_source, unsorted_source);
        assert_eq!(
            sorted_left.map(|index| &u[index]),
            unsorted_left.map(|index| &shuffled_u[index])
        );
        assert_eq!(
            sorted_right.map(|index| &vh[index]),
            unsorted_right.map(|index| &shuffled_vh[index])
        );
    }
}

#[test]
fn compact_factor_plan_rejects_duplicate_missing_mismatched_and_extra_routes() {
    // What: every nonzero source sector has one shape-correct left/right route and no extras.
    let rule = Z2FusionRule;
    let tensor = rectangular_svd_tensor(17, 13);
    let bound = bound_tensor(Arc::new(rule), &tensor);
    let plan = crate::factorize::compact_factor_plan_for_test(bound.space())
        .unwrap()
        .unwrap();
    let (source, u, vh) = crate::factorize::compact_factor_plan_regions_for_test(&plan);

    let mut duplicate = u.to_vec();
    duplicate.push(u[0].clone());
    assert!(
        crate::factorize::validate_compact_factor_routes_for_test(&source, &duplicate, &vh,)
            .is_err()
    );
    assert!(crate::factorize::validate_compact_factor_routes_for_test(&source, &[], &vh,).is_err());
    assert!(crate::factorize::validate_compact_factor_routes_for_test(&source, &vh, &vh,).is_err());

    let multi = tsvd_test_tensor(&rule, &[SectorId::new(0), SectorId::new(1)]);
    let multi_bound = bound_tensor(Arc::new(rule), &multi);
    let multi_plan = crate::factorize::compact_factor_plan_for_test(multi_bound.space())
        .unwrap()
        .unwrap();
    let (multi_source, multi_u, multi_vh) =
        crate::factorize::compact_factor_plan_regions_for_test(&multi_plan);
    let mut reversed_u = multi_u.to_vec();
    let mut reversed_vh = multi_vh.to_vec();
    reversed_u.reverse();
    reversed_vh.reverse();
    crate::factorize::validate_compact_factor_routes_for_test(
        &multi_source,
        &reversed_u,
        &reversed_vh,
    )
    .unwrap();
    let mut extra = u.to_vec();
    extra.push(
        multi_u
            .iter()
            .find(|region| region.coupled() == SectorId::new(1))
            .unwrap()
            .clone(),
    );
    assert!(
        crate::factorize::validate_compact_factor_routes_for_test(&source, &extra, &vh,).is_err()
    );
}

#[test]
fn typed_factor_axis_sum_overflow_is_exact_without_storage_materialization() {
    // What: an axis whose structural-zero degeneracies exceed usize reports
    // the exact checked error without allocating storage for those dimensions.
    let rule = U1FusionRule;
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new(
            [
                (U1Irrep::new(1).sector_id(), usize::MAX),
                (U1Irrep::new(2).sector_id(), 1),
            ],
            false,
        )]),
        FusionProductSpace::new([]),
    );
    let space = DynamicFusionMapSpace::from_final_homspace(&rule, homspace).unwrap();

    let error = TestTensor::<f64>::from_dyn(Vec::new(), space).unwrap_err();

    assert_eq!(error, CoreError::ElementCountOverflow);
}
