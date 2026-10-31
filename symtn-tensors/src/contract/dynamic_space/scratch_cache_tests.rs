use super::*;
use crate::probe::{self, Event};
use crate::test_support::CACHE_TEST_LOCK;
use crate::tests::GenericMultiplicityRule;
use std::sync::atomic::{AtomicUsize, Ordering};
use symtn_core::{
    BlockSpec, BraidingStyleKind, FusionProductSpace, FusionStyleKind,
    MultiplicityFreeFusionSymbols, RuleIdentity, SectorId, SectorLeg, SectorVec, U1FusionRule,
    U1Irrep, Z2FusionRule, Z2Irrep,
};

#[derive(Clone)]
struct CountingRule {
    identity: RuleIdentity,
    calls: Arc<AtomicUsize>,
}

impl CountingRule {
    fn new() -> Self {
        Self {
            identity: RuleIdentity::new_unique::<Self>(),
            calls: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl symtn_core::testing::Provider for CountingRule {}

impl FusionRule for CountingRule {
    fn rule_identity(&self) -> RuleIdentity {
        self.identity.clone()
    }
    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Unique
    }
    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }
    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }
    fn fusion_channels(&self, _: SectorId, _: SectorId) -> SectorVec {
        self.calls.fetch_add(1, Ordering::Relaxed);
        [SectorId::new(0)].into_iter().collect()
    }
}

impl symtn_core::MultiplicityFreeFusionRule for CountingRule {}
impl MultiplicityFreeFusionSymbols for CountingRule {
    type Scalar = f64;
    fn f_symbol_scalar(
        &self,
        _: SectorId,
        _: SectorId,
        _: SectorId,
        _: SectorId,
        _: SectorId,
        _: SectorId,
    ) -> f64 {
        1.0
    }
    fn r_symbol_scalar(&self, _: SectorId, _: SectorId, _: SectorId) -> f64 {
        1.0
    }
}
impl MultiplicityFreeRigidSymbols for CountingRule {
    fn dim_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
    fn inv_dim_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
    fn sqrt_dim_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
    fn inv_sqrt_dim_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
    fn twist_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
    fn frobenius_schur_phase_scalar(&self, _: SectorId) -> f64 {
        1.0
    }
}

#[test]
fn raw_transform_rejects_a_different_rule_identity() {
    // What: crate-internal derivation still rejects a provider other than
    // the one recorded by the source, even though public callers can only
    // reach this operation through BoundDynamicFusionMapSpace.
    let first = CountingRule::new();
    let second = CountingRule::new();
    let space = DynamicFusionMapSpace::from_final_homspace(
        &first,
        FusionTreeHomSpace::from_sector_ids([(0, 1)], []),
    )
    .unwrap();

    let error = space
        .transformed(&second, &TreeTransformOperation::permute([0], []))
        .unwrap_err();
    assert!(matches!(
        error,
        OperationError::Core(CoreError::FusionRuleMismatch { .. })
    ));
}

#[test]
fn partitioned_contract_destination_rejects_distinct_provider_identities() {
    let bind = |rule: Arc<CountingRule>| {
        let space = DynamicFusionMapSpace::from_final_homspace(
            rule.as_ref(),
            FusionTreeHomSpace::from_sector_ids([(0, 1)], [(0, 1)]),
        )
        .unwrap();
        BoundDynamicFusionMapSpace::bind_multiplicity_free(space, rule).unwrap()
    };
    let lhs = bind(Arc::new(CountingRule::new()));
    let rhs = bind(Arc::new(CountingRule::new()));
    let error = BoundDynamicFusionMapSpace::contracted_multiplicity_free_partitioned(
        &lhs,
        &rhs,
        &[1],
        &[0],
        OutputAxisOrder::identity(),
        1,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        OperationError::Core(CoreError::FusionRuleMismatch { .. })
    ));
}

// Single-charge U(1) source in a chosen bond dimension (the last leg of each
// block shape); one coupled sector, block shape [deg, deg].
fn u1_space(charge: i32, deg: usize) -> DynamicFusionMapSpace {
    let rule = U1FusionRule;
    let sid = U1Irrep::new(charge).sector_id();
    let leg = || FusionProductSpace::new([SectorLeg::new([(sid, deg)], false)]);
    let hom = FusionTreeHomSpace::new(leg(), leg());
    DynamicFusionMapSpace::from_final_homspace(&rule, hom).unwrap()
}

fn reset_final_result_layout_test_state() -> probe::Recording {
    symtn_core::clear_structure_caches();
    probe::record()
}

#[test]
fn transformed_space_is_derived_eagerly_per_call() {
    // What: every transform eagerly derives one content-identical result
    // space instead of retaining a source-plus-operation result.
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = U1FusionRule;
    let source = u1_space(301, 3);
    let operation = TreeTransformOperation::permute([1], [0]);
    let final_result_layout_builds_rec = reset_final_result_layout_test_state();

    let first = source.transformed(&rule, &operation).unwrap();
    assert_eq!(
        final_result_layout_builds_rec.get(Event::FinalResultLayoutBuild),
        1
    );

    let second = source.transformed(&rule, &operation).unwrap();
    assert_eq!(
        final_result_layout_builds_rec.get(Event::FinalResultLayoutBuild),
        2
    );
    assert_eq!(first, second);
}

#[test]
fn contracted_space_is_derived_eagerly_per_call() {
    // What: every contraction eagerly derives one content-identical result
    // space instead of retaining a source-plus-operation result.
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = U1FusionRule;
    let source = u1_space(901, 4);
    let final_result_layout_builds_rec = reset_final_result_layout_test_state();

    let first = DynamicFusionMapSpace::contracted(&rule, &source, &source, &[1], &[0]).unwrap();
    assert_eq!(
        final_result_layout_builds_rec.get(Event::FinalResultLayoutBuild),
        1
    );

    let second = DynamicFusionMapSpace::contracted(&rule, &source, &source, &[1], &[0]).unwrap();
    assert_eq!(
        final_result_layout_builds_rec.get(Event::FinalResultLayoutBuild),
        2
    );
    assert_eq!(first, second);
}

#[test]
fn ordered_contraction_eagerly_builds_only_the_final_layout() {
    use symtn_operations::OutputAxisOrder;

    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = U1FusionRule;
    let source = u1_space(902, 4);
    let axes = TensorContractSpec::new(&[1], &[0], OutputAxisOrder::from_axes(&[1, 0]));
    let expected_homspace = FusionTreeHomSpace::tensorcontract_homspace(
        &rule,
        source.homspace(),
        source.homspace(),
        &[1],
        &[0],
        &[1, 0],
        1,
    )
    .unwrap();
    let final_result_layout_builds_rec = reset_final_result_layout_test_state();

    let first = DynamicFusionMapSpace::contracted_with_spec(&rule, &source, &source, axes).unwrap();
    assert_eq!(
        final_result_layout_builds_rec.get(Event::FinalResultLayoutBuild),
        1
    );
    assert_eq!(first.homspace(), &expected_homspace);

    let second =
        DynamicFusionMapSpace::contracted_with_spec(&rule, &source, &source, axes).unwrap();
    assert_eq!(
        final_result_layout_builds_rec.get(Event::FinalResultLayoutBuild),
        2
    );
    assert_eq!(first, second);
}

#[test]
fn validation_only_contraction_never_builds_a_default_layout() {
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = U1FusionRule;
    let compatible = u1_space(903, 4);
    let incompatible = u1_space(903, 5);
    let final_result_layout_builds_rec = reset_final_result_layout_test_state();

    DynamicFusionMapSpace::validate_contracted_homspace(
        &rule,
        &compatible,
        &compatible,
        &[1],
        &[0],
    )
    .unwrap();
    assert_eq!(
        final_result_layout_builds_rec.get(Event::FinalResultLayoutBuild),
        0
    );

    assert!(DynamicFusionMapSpace::validate_contracted_homspace(
        &rule,
        &compatible,
        &incompatible,
        &[1],
        &[0],
    )
    .is_err());
    assert_eq!(
        final_result_layout_builds_rec.get(Event::FinalResultLayoutBuild),
        0
    );
}

#[test]
fn generic_transform_builds_only_the_final_layout() {
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = GenericMultiplicityRule;
    let homspace = FusionTreeHomSpace::from_sector_ids([(0, 2)], [(0, 3)]);
    let key_count = homspace.fusion_tree_keys_generic(&rule).unwrap().len();
    let source = DynamicFusionMapSpace::from_final_homspace_generic(&rule, homspace).unwrap();
    let final_result_layout_builds_rec = reset_final_result_layout_test_state();

    let transformed = source
        .transformed_generic(&rule, &TreeTransformOperation::permute([1], [0]))
        .unwrap();
    assert_eq!(
        final_result_layout_builds_rec.get(Event::FinalResultLayoutBuild),
        1
    );
    assert_eq!(transformed.structure().block_count(), key_count);
}

#[test]
fn generic_contraction_builds_only_the_final_layout() {
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = GenericMultiplicityRule;
    let homspace = FusionTreeHomSpace::from_sector_ids([(0, 2)], [(0, 2)]);
    let key_count = homspace.fusion_tree_keys_generic(&rule).unwrap().len();
    let source = DynamicFusionMapSpace::from_final_homspace_generic(&rule, homspace).unwrap();
    let final_result_layout_builds_rec = reset_final_result_layout_test_state();

    let contracted =
        DynamicFusionMapSpace::contracted_generic(&rule, &source, &source, &[1], &[0]).unwrap();
    assert_eq!(
        final_result_layout_builds_rec.get(Event::FinalResultLayoutBuild),
        1
    );
    assert_eq!(contracted.structure().block_count(), key_count);
}

#[test]
fn layout_authority_distinguishes_rules_and_reuses_semantic_spaces() {
    // What: opaque layout identity aliases equal bound layouts but never distinct rules.
    let first_provider = Arc::new(Z2FusionRule);
    let second_provider = Arc::new(Z2FusionRule);
    let raw = z2_matrix_space();
    let first = BoundDynamicFusionMapSpace::bind_multiplicity_free(
        raw.clone(),
        Arc::clone(&first_provider),
    )
    .unwrap();
    let second =
        BoundDynamicFusionMapSpace::bind_multiplicity_free(raw, Arc::clone(&second_provider))
            .unwrap();
    assert_eq!(first.validated_layout(), second.validated_layout());

    let wrong =
        BoundDynamicFusionMapSpace::bind_multiplicity_free(u1_space(0, 1), Arc::new(U1FusionRule))
            .unwrap();
    assert_ne!(first.validated_layout(), wrong.validated_layout());
    let error = first
        .rebind_validated(&wrong.validated_layout())
        .unwrap_err();
    assert!(matches!(
        error,
        OperationError::Core(CoreError::FusionRuleMismatch { .. })
    ));
    assert!(Arc::ptr_eq(first.provider_arc(), &first_provider));
}

#[test]
fn layout_authority_rebinds_derived_space_to_exact_provider_arc() {
    // What: a cached raw derived layout inherits the current caller's provider allocation.
    let first_provider = Arc::new(Z2FusionRule);
    let second_provider = Arc::new(Z2FusionRule);
    let raw = z2_matrix_space();
    let first = BoundDynamicFusionMapSpace::bind_multiplicity_free(
        raw.clone(),
        Arc::clone(&first_provider),
    )
    .unwrap();
    let second =
        BoundDynamicFusionMapSpace::bind_multiplicity_free(raw, Arc::clone(&second_provider))
            .unwrap();
    let validated = first.validated_layout();

    let rebound = second.rebind_validated(&validated).unwrap();

    assert!(Arc::ptr_eq(rebound.provider_arc(), &second_provider));
    assert!(!Arc::ptr_eq(rebound.provider_arc(), &first_provider));
}

#[test]
fn layout_authority_normalizes_zero_legs_but_not_storage_geometry() {
    // What: explicit zero sectors share one mathematical layout authority,
    // while a genuinely different storage geometry remains distinct.
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let provider = Arc::new(Z2FusionRule);
    let even = Z2Irrep::EVEN.sector_id();
    let odd = Z2Irrep::ODD.sector_id();
    let leg = |include_zero| {
        let sectors = if include_zero {
            vec![(even, 1), (odd, 0)]
        } else {
            vec![(even, 1)]
        };
        FusionProductSpace::new([SectorLeg::new(sectors, false)])
    };
    let without_zero_hom = FusionTreeHomSpace::new(leg(false), leg(false));
    let with_zero_hom = FusionTreeHomSpace::new(leg(true), leg(true));
    let without_zero = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::clone(&provider),
        without_zero_hom,
    )
    .unwrap();
    let with_zero = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::clone(&provider),
        with_zero_hom,
    )
    .unwrap();
    assert_eq!(
        without_zero.validated_layout(),
        with_zero.validated_layout()
    );

    let raw = z2_matrix_space();
    let shifted_blocks = (0..raw.structure().block_count())
        .map(|index| {
            let block = raw.structure().block(index).unwrap();
            BlockSpec::with_key(
                block.key().clone(),
                block.shape().to_vec(),
                block.strides().to_vec(),
                block.offset() + 1,
            )
            .unwrap()
        })
        .collect();
    let shifted_raw = DynamicFusionMapSpace {
        subblock_structure: Arc::new(BlockStructure::from_blocks(shifted_blocks).unwrap()),
        adjoint: OnceLock::new(),
        ..raw.clone()
    };
    let canonical =
        BoundDynamicFusionMapSpace::bind_multiplicity_free(raw.clone(), Arc::clone(&provider))
            .unwrap();
    let shifted =
        BoundDynamicFusionMapSpace::bind_multiplicity_free(shifted_raw, Arc::clone(&provider))
            .unwrap();
    assert_ne!(canonical.validated_layout(), shifted.validated_layout());
}

#[test]
fn validated_layout_does_not_retain_provider_allocation() {
    // What: erasing a bound layout drops the provider allocation while preserving layout data.
    let provider = Arc::new(Z2FusionRule);
    let weak = Arc::downgrade(&provider);
    let bound = BoundDynamicFusionMapSpace::bind_multiplicity_free(
        z2_matrix_space(),
        Arc::clone(&provider),
    )
    .unwrap();
    let validated = bound.validated_layout();

    drop(bound);
    drop(provider);

    assert!(weak.upgrade().is_none());
    assert_eq!(validated.required_len().unwrap(), 2);
}

#[test]
fn validated_layout_does_not_retain_the_adjoint_memo() {
    // What: a layout taken after a lazy-adjoint contraction filled the
    // parent's adjoint memo neither keeps that memo nor charges differently
    // from a never-adjointed layout (`detach_runtime`).
    let provider = Arc::new(Z2FusionRule);
    let bound = BoundDynamicFusionMapSpace::bind_multiplicity_free(
        z2_matrix_space(),
        Arc::clone(&provider),
    )
    .unwrap();
    let cold_charge = bound.validated_layout().charged_retained_bytes();
    FusionOperand::adjoint(bound.space())
        .prepare(&Z2FusionRule, encoded_layout_primer::<Z2FusionRule>)
        .unwrap();
    bound.space().adjoint_view().unwrap();
    let memo = Arc::downgrade(bound.space().adjoint.get().expect("prepare fills the memo"));

    let validated = bound.validated_layout();
    drop(bound);

    assert!(memo.upgrade().is_none());
    assert_eq!(validated.charged_retained_bytes(), cold_charge);
}

fn z2_matrix_space() -> DynamicFusionMapSpace {
    let leg = || SectorLeg::new([(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 1)], false);
    DynamicFusionMapSpace::from_final_homspace(
        &Z2FusionRule,
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg()]),
            FusionProductSpace::new([leg()]),
        ),
    )
    .unwrap()
}

#[test]
fn bound_space_rejects_wrong_and_missing_identity_before_provider_enumeration() {
    let source_rule = CountingRule::new();
    let homspace = FusionTreeHomSpace::from_sector_ids([(0, 1), (0, 1)], []);
    let space = DynamicFusionMapSpace::from_final_homspace(&source_rule, homspace).unwrap();
    assert!(source_rule.calls.load(Ordering::Relaxed) > 0);

    let wrong_rule = Arc::new(CountingRule::new());
    for error in [
        BoundDynamicFusionMapSpace::bind_multiplicity_free(space.clone(), Arc::clone(&wrong_rule))
            .unwrap_err(),
        BoundDynamicFusionMapSpace::bind_generic(space.clone(), Arc::clone(&wrong_rule))
            .unwrap_err(),
    ] {
        // What: a known rule mismatch is reported before either binding
        // mode can inspect provider style or enumerate the HomSpace.
        assert!(matches!(
            error,
            OperationError::Core(CoreError::FusionRuleMismatch { .. })
        ));
        assert_eq!(wrong_rule.calls.load(Ordering::Relaxed), 0);
    }

    source_rule.calls.store(0, Ordering::Relaxed);
    let unbound = DynamicFusionMapSpace {
        admission: FusionSpaceAdmission::Unbound,
        ..space
    };
    let matching_rule = Arc::new(source_rule);
    for error in [
        BoundDynamicFusionMapSpace::bind_multiplicity_free(
            unbound.clone(),
            Arc::clone(&matching_rule),
        )
        .unwrap_err(),
        BoundDynamicFusionMapSpace::bind_generic(unbound, Arc::clone(&matching_rule)).unwrap_err(),
    ] {
        assert!(matches!(
            error,
            OperationError::Core(CoreError::MissingFusionRuleIdentity)
        ));
        assert_eq!(matching_rule.calls.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn bound_space_requires_the_complete_tree_grid() {
    let complete = z2_matrix_space();
    let first = complete.structure().block(0).unwrap();
    let incomplete_structure = BlockStructure::from_blocks_with_rank(
        complete.rank(),
        vec![BlockSpec::with_key(
            first.key().clone(),
            first.shape().to_vec(),
            first.strides().to_vec(),
            first.offset(),
        )
        .unwrap()],
    )
    .unwrap();
    let incomplete = DynamicFusionMapSpace {
        subblock_structure: Arc::new(incomplete_structure),
        admission: FusionSpaceAdmission::Subset(Z2FusionRule.rule_identity()),
        adjoint: OnceLock::new(),
        ..complete
    };

    let error =
        BoundDynamicFusionMapSpace::bind_multiplicity_free(incomplete, Arc::new(Z2FusionRule))
            .unwrap_err();
    assert!(matches!(
        error,
        OperationError::Core(CoreError::BlockCountMismatch { .. })
    ));
}

#[test]
fn binding_mode_mismatch_is_rejected_by_provider_style() {
    let space = z2_matrix_space();
    let multiplicity_free =
        BoundDynamicFusionMapSpace::bind_multiplicity_free(space.clone(), Arc::new(Z2FusionRule))
            .unwrap();
    let generic_error =
        BoundDynamicFusionMapSpace::bind_generic(space, Arc::new(Z2FusionRule)).unwrap_err();

    assert!(matches!(
        generic_error,
        OperationError::Core(CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Generic,
            actual: FusionStyleKind::Unique,
        })
    ));
    assert_eq!(multiplicity_free.clone().space(), multiplicity_free.space());
}

#[test]
fn every_generic_root_rejects_multiplicity_free_provider_before_input_validation() {
    // What: provider style is the single Generic capability authority for
    // raw binding, shape/final-HomSpace roots, and derived operations.
    let raw = z2_matrix_space();
    let homspace = raw.homspace().clone();
    let bound =
        BoundDynamicFusionMapSpace::bind_multiplicity_free(raw.clone(), Arc::new(Z2FusionRule))
            .unwrap();
    let expected = |error| {
        assert!(matches!(
            error,
            OperationError::Core(CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Generic,
                actual: FusionStyleKind::Unique,
            })
        ));
    };

    expected(BoundDynamicFusionMapSpace::bind_generic(raw, Arc::new(Z2FusionRule)).unwrap_err());
    expected(
        DynamicFusionMapSpace::from_final_homspace_generic(&Z2FusionRule, homspace.clone())
            .unwrap_err(),
    );
    expected(
        BoundDynamicFusionMapSpace::from_final_homspace_generic(
            Arc::new(Z2FusionRule),
            homspace.clone(),
        )
        .unwrap_err(),
    );
    expected(
        BoundDynamicFusionMapSpace::from_final_homspace_generic(Arc::new(Z2FusionRule), homspace)
            .unwrap_err(),
    );
    expected(
        bound
            .transformed_generic(&TreeTransformOperation::permute([0], [1]))
            .unwrap_err(),
    );
    expected(
        BoundDynamicFusionMapSpace::contracted_generic(&bound, &bound, &[99], &[]).unwrap_err(),
    );
}

#[test]
fn direct_bound_construction_enumerates_no_more_than_raw_construction() {
    // What: canonical construction and later Complete admission add no
    // second fusion-tree enumeration.
    let hom = || FusionTreeHomSpace::from_sector_ids([(0, 1)], [(0, 1)]);
    let raw_rule = CountingRule::new();
    let raw = DynamicFusionMapSpace::from_final_homspace(&raw_rule, hom()).unwrap();
    let raw_calls = raw_rule.calls.load(Ordering::Relaxed);
    raw_rule.calls.store(0, Ordering::Relaxed);
    let provider = Arc::new(raw_rule);
    let rebound =
        BoundDynamicFusionMapSpace::bind_multiplicity_free(raw, Arc::clone(&provider)).unwrap();
    assert_eq!(provider.calls.load(Ordering::Relaxed), 0);
    assert!(matches!(
        rebound.space().admission(),
        FusionSpaceAdmission::Complete(_)
    ));

    let bound_rule = Arc::new(CountingRule::new());
    BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::clone(&bound_rule),
        hom(),
    )
    .unwrap();

    assert_eq!(bound_rule.calls.load(Ordering::Relaxed), raw_calls);
}

#[test]
fn derived_bound_contract_and_transform_do_not_reenumerate_for_binding() {
    // What: a derived proof adds no tree-grid pass beyond raw output build.
    let hom = || FusionTreeHomSpace::from_sector_ids([(0, 1)], [(0, 1)]);

    let raw_rule = CountingRule::new();
    let raw = DynamicFusionMapSpace::from_final_homspace(&raw_rule, hom()).unwrap();
    raw_rule.calls.store(0, Ordering::Relaxed);
    let _ = DynamicFusionMapSpace::contracted(&raw_rule, &raw, &raw, &[1], &[0]).unwrap();
    let raw_contract_calls = raw_rule.calls.load(Ordering::Relaxed);

    let bound_rule = Arc::new(CountingRule::new());
    let bound = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::clone(&bound_rule),
        hom(),
    )
    .unwrap();
    bound_rule.calls.store(0, Ordering::Relaxed);
    let contracted =
        BoundDynamicFusionMapSpace::contracted_multiplicity_free(&bound, &bound, &[1], &[0])
            .unwrap();
    assert_eq!(bound_rule.calls.load(Ordering::Relaxed), raw_contract_calls);
    assert!(Arc::ptr_eq(contracted.provider_arc(), &bound_rule));

    let raw_transform_rule = CountingRule::new();
    let raw_transform =
        DynamicFusionMapSpace::from_final_homspace(&raw_transform_rule, hom()).unwrap();
    raw_transform_rule.calls.store(0, Ordering::Relaxed);
    let operation = TreeTransformOperation::permute([0], [1]);
    let _ = raw_transform
        .transformed(&raw_transform_rule, &operation)
        .unwrap();
    let raw_transform_calls = raw_transform_rule.calls.load(Ordering::Relaxed);

    let bound_transform_rule = Arc::new(CountingRule::new());
    let bound_transform = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::clone(&bound_transform_rule),
        hom(),
    )
    .unwrap();
    bound_transform_rule.calls.store(0, Ordering::Relaxed);
    let transformed = bound_transform
        .transformed_multiplicity_free(&operation)
        .unwrap();
    assert_eq!(
        bound_transform_rule.calls.load(Ordering::Relaxed),
        raw_transform_calls
    );
    assert!(Arc::ptr_eq(
        transformed.provider_arc(),
        &bound_transform_rule
    ));
}

#[test]
fn bound_contract_normalizes_equal_identity_to_lhs_provider() {
    // What: independently allocated providers with one semantic identity
    // compose, while a distinct identity is rejected before execution.
    let rule = CountingRule::new();
    let lhs_provider = Arc::new(rule.clone());
    let rhs_provider = Arc::new(rule);
    assert!(!Arc::ptr_eq(&lhs_provider, &rhs_provider));
    let hom = || FusionTreeHomSpace::from_sector_ids([(0, 1)], [(0, 1)]);
    let lhs = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::clone(&lhs_provider),
        hom(),
    )
    .unwrap();
    let rhs = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::clone(&rhs_provider),
        hom(),
    )
    .unwrap();
    let output =
        BoundDynamicFusionMapSpace::contracted_multiplicity_free(&lhs, &rhs, &[1], &[0]).unwrap();
    assert!(Arc::ptr_eq(output.provider_arc(), &lhs_provider));

    let other_provider = Arc::new(CountingRule::new());
    let other =
        BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(other_provider, hom())
            .unwrap();
    let error = BoundDynamicFusionMapSpace::contracted_multiplicity_free(&lhs, &other, &[1], &[0])
        .unwrap_err();
    assert!(matches!(
        error,
        OperationError::Core(CoreError::FusionRuleMismatch { .. })
    ));
}

#[test]
fn equal_layout_and_shapes_share_frozen_content_not_regions() {
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let a = u1_space(1, 3);
    let b = u1_space(1, 3);
    assert_eq!(a.structure().content_id(), b.structure().content_id());
}

#[test]
fn different_bond_dimension_is_not_shared() {
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    // The shapes carry chi, so a differently-truncated build keys separately.
    let a = u1_space(1, 3);
    let b = u1_space(1, 4);
    assert_ne!(a.structure().content_id(), b.structure().content_id());
}

#[test]
fn different_homspace_is_not_shared() {
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let a = u1_space(1, 3);
    let b = u1_space(2, 3);
    assert_ne!(a.structure().content_id(), b.structure().content_id());
}

#[test]
fn canonical_explicit_shapes_preserve_content_after_homspace_intern_eviction() {
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    symtn_core::clear_structure_caches();
    let before = u1_space(73, 3);
    for charge in 10_000..19_000 {
        let sid = U1Irrep::new(charge).sector_id();
        let leg = || FusionProductSpace::new([SectorLeg::new([(sid, 1)], false)]);
        let _ = FusionTreeHomSpace::new(leg(), leg());
    }
    let after = u1_space(73, 3);
    assert_eq!(before.homspace().id(), after.homspace().id());
    assert_eq!(before.structure(), after.structure());
}

#[test]
fn large_canonical_shapes_reuse_frozen_content_without_tensor_storage() {
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = U1FusionRule;
    let sid = U1Irrep::new(91).sector_id();
    let leg = || FusionProductSpace::new([SectorLeg::new([(sid, 4096)], false)]);
    let hom = FusionTreeHomSpace::new(leg(), leg());
    let first = DynamicFusionMapSpace::from_final_homspace(&rule, hom.clone()).unwrap();
    let second = DynamicFusionMapSpace::from_final_homspace(&rule, hom).unwrap();
    assert_eq!(first.required_len().unwrap(), 4096 * 4096);
    assert_eq!(first.structure(), second.structure());
}

#[test]
fn reset_and_concurrent_rebuild_keep_structure_semantics() {
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    symtn_core::clear_structure_caches();
    let spaces = std::thread::scope(|scope| {
        let resetter = scope.spawn(|| {
            for _ in 0..32 {
                symtn_core::clear_structure_caches();
            }
        });
        let builders = (0..4)
            .map(|_| scope.spawn(|| (0..32).map(|_| u1_space(111, 5)).collect::<Vec<_>>()))
            .collect::<Vec<_>>();
        resetter.join().unwrap();
        builders
            .into_iter()
            .flat_map(|builder| builder.join().unwrap())
            .collect::<Vec<_>>()
    });
    let expected = spaces[0].structure().as_ref();
    assert!(spaces
        .iter()
        .all(|space| space.structure().as_ref() == expected));
    let rebuilt = u1_space(111, 5);
    let cached = u1_space(111, 5);
    assert_eq!(
        rebuilt.structure().content_id(),
        cached.structure().content_id()
    );
}

#[test]
fn bound_layout_strategy_propagates_without_heap_ownership() {
    // What: clone, validated-layout rebind, derived construction, transform,
    // and contraction retain one opaque lowered strategy, while an expert
    // encoded root remains distinct and a mixed binary result follows lhs.
    assert!(
        std::mem::size_of::<LayoutBuildCapability<symtn_core::SU2FusionRule>>()
            <= std::mem::size_of::<usize>()
    );
    let provider = Arc::new(symtn_core::SU2FusionRule);
    let half = symtn_core::SU2Irrep::from_twice_spin(1).sector_id();
    let leg = || FusionProductSpace::new([symtn_core::SectorLeg::new([(half, 1)], false)]);
    let make_hom = || FusionTreeHomSpace::new(leg(), leg());
    let hom = make_hom();
    checked_layout_primer(provider.as_ref(), &hom).unwrap();
    let lowered = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(
        Arc::clone(&provider),
        hom.clone(),
    )
    .unwrap();
    let encoded = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::clone(&provider),
        hom,
    )
    .unwrap();

    let cloned = lowered.clone();
    let rebound = lowered
        .rebind_validated(&lowered.validated_layout())
        .unwrap();
    let derived = lowered.derive_from_final_homspace(make_hom()).unwrap();
    let transformed = lowered
        .transformed_multiplicity_free(&TreeTransformOperation::permute([0], [1]))
        .unwrap();
    let contracted =
        BoundDynamicFusionMapSpace::contracted_multiplicity_free(&lowered, &cloned, &[], &[])
            .unwrap();

    let malformed = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(SectorId::new(255), 1)], false)]),
        FusionProductSpace::new([]),
    );
    for output in [&cloned, &rebound, &derived, &transformed, &contracted] {
        // The checked enumerator reports the codec cause rather than the
        // lowered path's flattened static message.
        assert!(matches!(
            output.prime_derived_homspace(&malformed),
            Err(OperationError::FusionAlgebra(_))
        ));
    }
    assert!(encoded.prime_derived_homspace(&malformed).is_ok());
    let mixed =
        BoundDynamicFusionMapSpace::contracted_multiplicity_free(&lowered, &encoded, &[], &[])
            .unwrap();
    assert!(matches!(
        mixed.prime_derived_homspace(&malformed),
        Err(OperationError::FusionAlgebra(_))
    ));
}

#[test]
fn root_constructor_builds_from_checked_keys() {
    // What: the lowered root builds the complete key set of a non-abelian
    // space, here spin-1/2 x spin-1/2 coupling to spin 0 and spin 1 on both
    // sides with one tree each, and rejects a sector its codec cannot name.
    let provider = Arc::new(symtn_core::SU2FusionRule);
    let half = symtn_core::SU2Irrep::from_twice_spin(1).sector_id();
    let side = || {
        FusionProductSpace::new([
            SectorLeg::new([(half, 1)], false),
            SectorLeg::new([(half, 1)], false),
        ])
    };
    let hom = FusionTreeHomSpace::new(side(), side());
    let bound = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(
        Arc::clone(&provider),
        hom,
    )
    .unwrap();
    let mut actual = (0..bound.space().structure().block_count())
        .map(|index| {
            bound
                .space()
                .structure()
                .block(index)
                .unwrap()
                .key()
                .clone()
        })
        .collect::<Vec<_>>();
    actual.sort();
    // A rank-2 tree has one vertex (label 1, multiplicity free) and no
    // inner line.
    let mut expected = [0, 2]
        .map(|twice_spin| {
            let coupled = symtn_core::SU2Irrep::from_twice_spin(twice_spin).sector_id();
            BlockKey::FusionTree(
                FusionTreePairKey::try_pair_from_sector_ids(
                    [half.id(), half.id()],
                    [half.id(), half.id()],
                    coupled.id(),
                    [false, false],
                    [false, false],
                    [],
                    [],
                    [1],
                    [1],
                )
                .unwrap(),
            )
        })
        .to_vec();
    expected.sort();
    assert_eq!(actual, expected);

    let malformed = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(SectorId::new(255), 1)], false)]),
        FusionProductSpace::new([]),
    );
    assert!(matches!(
        BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(
            provider, malformed,
        ),
        Err(OperationError::FusionAlgebra(_))
    ));
}

#[test]
fn direct_and_adjoint_permutes_memoize_under_distinct_keys() {
    // What: the direct and the adjoint orientation of one resident source
    // memoize their permuted destinations separately; each warm call takes
    // its own memo without deriving, and equals the space derived from the
    // explicitly permuted HomSpace.
    let _guard = CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    symtn_core::clear_structure_caches();
    let rule = U1FusionRule;
    let leg = |dual| {
        SectorLeg::new(
            [-1, 0, 1].map(|q| (U1Irrep::new(q).sector_id(), q.unsigned_abs() as usize + 1)),
            dual,
        )
    };
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(false), leg(true)]),
        FusionProductSpace::new([leg(false), leg(false)]),
    );
    let space = DynamicFusionMapSpace::from_final_homspace(&rule, hom.clone()).unwrap();
    let operation = TreeTransformOperation::permute([1, 2], [3, 0]);
    let transform = |operand: FusionOperand<'_>| {
        operand
            .prepare(&rule, encoded_layout_primer::<U1FusionRule>)
            .unwrap()
            .transformed_space(&rule, &operation, encoded_layout_primer::<U1FusionRule>)
            .unwrap()
    };
    let derived = |hom: &FusionTreeHomSpace| {
        DynamicFusionMapSpace::from_final_homspace(
            &rule,
            hom.permute(&rule, &[1, 2], &[3, 0]).unwrap(),
        )
        .unwrap()
    };
    let direct = derived(&hom);
    let adjoint = derived(&FusionTreeHomSpace::new(
        hom.domain().clone(),
        hom.codomain().clone(),
    ));
    assert_ne!(direct, adjoint);
    assert_eq!(transform(FusionOperand::direct(&space)), direct);
    assert_eq!(transform(FusionOperand::adjoint(&space)), adjoint);
    let warm = probe::record();
    for _ in 0..2 {
        assert_eq!(transform(FusionOperand::direct(&space)), direct);
        assert_eq!(transform(FusionOperand::adjoint(&space)), adjoint);
    }
    assert_eq!(warm.get(Event::DerivedHomspaceBuild), 0);
    assert_eq!(warm.get(Event::FinalResultLayoutBuild), 0);
}
