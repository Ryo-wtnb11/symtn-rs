use super::*;
use crate::probe::{self, Event};

#[test]
fn checked_generic_shared_provider_builds_identical_plans_concurrently() {
    let raw = dense_generic_dynamic_space();
    let data = (0..raw.required_len().unwrap())
        .map(|index| index as f64)
        .collect::<Vec<_>>();
    let provider = Arc::new(SynchronizedCheckedGeneric::new());
    let source = crate::BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&provider),
        raw.homspace().clone(),
    )
    .unwrap();
    *provider.calls.lock().unwrap() = 0;
    let operation = TreeTransformOperation::braid([0, 2], [1], [0, 1], [2]);

    let outputs = std::thread::scope(|scope| {
        (0..4)
            .map(|_| {
                let source = source.clone();
                let operation = operation.clone();
                let data = &data;
                scope.spawn(move || {
                    crate::TreeTransformExecutionContext::<f64, RuleIdentity, f64>::default()
                        .tree_transform_owned_checked_generic_in(
                            &source, None, data, &operation, 1.0,
                        )
                        .unwrap()
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });
    for output in outputs.iter().skip(1) {
        assert_eq!(output.0.space(), outputs[0].0.space());
        assert_eq!(output.1, outputs[0].1);
        assert!(Arc::ptr_eq(output.0.provider_arc(), &provider));
    }
    assert!(*provider.calls.lock().unwrap() > 0);
}

#[test]
#[allow(clippy::arc_with_non_send_sync)]
fn checked_generic_runtime_reuses_completed_structure_after_checked_admission() {
    // Isolated: a sibling test may publish this shared canonical key between
    // the two calls, turning the cold build into a hit.
    if crate::test_support::run_isolated_or_return(
        "SYMTN_TENSORS_CHECKED_RUNTIME_REUSE_ISOLATED",
        "tests::tree_transform_plan::checked_generic::checked_generic_runtime_reuses_completed_structure_after_checked_admission",
    ) {
        return;
    }
    // What: the first successful checked transform publishes one completed
    // structure, while a repeat still performs provider admission but skips
    // the F/R coefficient build and reuses the same Runtime entry.
    let rule = DenseGenericRule;
    let provider = Arc::new(CheckedPlanSpy::new(&rule));
    let raw = dense_generic_dynamic_space();
    let source = crate::BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&provider),
        raw.homspace().clone(),
    )
    .unwrap();
    let data = (0..source.space().required_len().unwrap())
        .map(|index| index as f64 + 1.0)
        .collect::<Vec<_>>();
    let operation = TreeTransformOperation::braid([0, 2], [1], [0, 1], [2]);
    let mut context = crate::TreeTransformExecutionContext::<f64, RuleIdentity>::default();
    owner_activity();

    provider.calls.set([0; CheckedPlanCall::COUNT]);
    let first = context
        .tree_transform_owned_checked_generic_in(&source, None, &data, &operation, 1.0)
        .unwrap();
    let cold = owner_activity();
    assert_eq!((cold.hits, cold.publications), (0, 1), "{cold:?}");
    assert!(provider.call_count(CheckedPlanCall::F) > 0);
    assert!(provider.call_count(CheckedPlanCall::R) > 0);

    provider.calls.set([0; CheckedPlanCall::COUNT]);
    let repeated = context
        .tree_transform_owned_checked_generic_in(&source, None, &data, &operation, 1.0)
        .unwrap();
    let warm = owner_activity();

    assert_eq!((warm.hits, warm.builds), (1, 0));
    assert_eq!(provider.call_count(CheckedPlanCall::F), 0);
    assert_eq!(provider.call_count(CheckedPlanCall::R), 0);
    assert_eq!(first.0.space(), repeated.0.space());
    assert_eq!(first.1, repeated.1);
    assert!(Arc::ptr_eq(first.0.provider_arc(), &provider));
    assert!(Arc::ptr_eq(repeated.0.provider_arc(), &provider));

    owner_activity();
    *provider.identity.borrow_mut() = Some(RuleIdentity::of_type::<ToyGenericRule>());
    let identity_error = context
        .tree_transform_owned_checked_generic_in(
            &source,
            None,
            &data,
            &TreeTransformOperation::permute([1, 0], [2]),
            1.0,
        )
        .unwrap_err();
    assert!(matches!(
        identity_error,
        CheckedGenericPlanError::Core(CoreError::FusionRuleMismatch { .. })
    ));
    assert_owner_untouched(owner_activity());
    *provider.identity.borrow_mut() = None;

    provider.calls.set([0; CheckedPlanCall::COUNT]);
    provider.fail.set(Some((CheckedPlanCall::F, 1)));
    let late_error = context
        .tree_transform_owned_checked_generic_in(
            &source,
            None,
            &data,
            &TreeTransformOperation::transpose([0], [2, 1]),
            1.0,
        )
        .unwrap_err();
    assert!(matches!(
        late_error,
        CheckedGenericPlanError::Provider(CheckedPlanSpyError(CheckedPlanCall::F))
    ));
    // What: a failed build publishes nothing.
    assert_eq!(owner_activity().publications, 0);
}

#[test]
#[allow(clippy::arc_with_non_send_sync)]
fn checked_generic_adjoint_storage_matches_literal_dense_braid() {
    let rule = DenseGenericRule;
    let provider = Arc::new(CheckedPlanSpy::new(&rule));
    let canonical = crate::BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&provider),
        dense_generic_dynamic_space().homspace().clone(),
    )
    .unwrap();
    let parent = crate::adjoint_bound_space_dyn_generic_checked(&canonical).unwrap();
    let logical = crate::adjoint_bound_space_dyn_generic_checked(&parent).unwrap();
    let parent_data = (0..parent.space().required_len().unwrap())
        .map(|index| Complex64::new(index as f64 + 1.0, 0.25 - index as f64))
        .collect::<Vec<_>>();
    let operation = TreeTransformOperation::braid([1, 0], [2], [0, 1], [2]);
    let alpha = Complex64::new(0.5, -1.25);
    let mut context =
        crate::TreeTransformExecutionContext::<Complex64, RuleIdentity, f64>::default();

    let actual = context
        .tree_transform_owned_checked_generic_in(
            &logical,
            Some(&parent),
            &parent_data,
            &operation,
            alpha,
        )
        .unwrap();

    assert_eq!(
        actual.1,
        literal_dense_generic_adjoint_braid(&parent_data, alpha)
    );
    assert_eq!(actual.0.space().structure().block_count(), 2);
    for (index, offset) in [0, 6].into_iter().enumerate() {
        let block = actual.0.space().structure().block(index).unwrap();
        assert_eq!(block.shape(), &[3, 2, 5]);
        assert_eq!(block.strides(), &[1, 3, 12]);
        assert_eq!(block.offset(), offset);
    }
}

#[test]
fn checked_generic_adjoint_storage_keeps_distinct_logical_layout_cache_identity() {
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let provider = Arc::new(SynchronizedCheckedGeneric::new());
    let canonical = crate::BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&provider),
        dense_generic_dynamic_space().homspace().clone(),
    )
    .unwrap();
    let parent = crate::adjoint_bound_space_dyn_generic_checked(&canonical).unwrap();
    let canonical_logical = crate::adjoint_bound_space_dyn_generic_checked(&parent).unwrap();
    let canonical_structure = canonical_logical.space().structure();
    let reordered_structure = BlockStructure::from_blocks_with_rank(
        3,
        [1, 0]
            .into_iter()
            .enumerate()
            .map(|(position, index)| {
                let block = canonical_structure.block(index).unwrap();
                BlockSpec::with_key(
                    block.key().clone(),
                    block.shape().to_vec(),
                    vec![1, 2, 6],
                    position * 30,
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap();
    let reordered_typed = DynamicFusionMapSpace::try_from_layout(
        canonical_logical.space().homspace().clone(),
        reordered_structure,
    )
    .unwrap()
    .try_bind_rule(provider.as_ref())
    .unwrap();
    let custom_bound = crate::BoundDynamicFusionMapSpace::bind_generic(
        reordered_typed.clone(),
        Arc::clone(&provider),
    )
    .unwrap();
    let reordered_logical = canonical_logical
        .rebind_validated(&custom_bound.validated_layout())
        .unwrap();
    assert_ne!(
        canonical_structure.content_id(),
        reordered_logical.space().structure().content_id()
    );

    let parent_data = (0..parent.space().required_len().unwrap())
        .map(|index| Complex64::new(index as f64 + 1.0, 0.25 - index as f64))
        .collect::<Vec<_>>();
    let operation = TreeTransformOperation::braid([1, 0], [2], [0, 1], [2]);
    let alpha = Complex64::new(0.5, -1.25);
    let expected = literal_dense_generic_adjoint_braid(&parent_data, alpha);
    let mut context =
        crate::TreeTransformExecutionContext::<Complex64, RuleIdentity, f64>::default();
    owner_activity();
    let mut execute = |logical| {
        context
            .tree_transform_owned_checked_generic_in(
                logical,
                Some(&parent),
                &parent_data,
                &operation,
                alpha,
            )
            .unwrap()
    };

    assert_eq!(execute(&canonical_logical).1, expected);
    assert_eq!(execute(&reordered_logical).1, expected);
    assert_eq!(execute(&reordered_logical).1, expected);
    // What: the canonical logical layout and a reordered expert one key
    // distinct entries. The expert layout is never admitted, so its key is
    // lookup-only: each of its calls builds, and nothing aliases the
    // canonical entry.
    // The canonical call may hit an entry a sibling test published.
    let activity = owner_activity();
    assert!(activity.builds >= 2, "{activity:?}");
    assert_eq!(activity.builds + activity.hits, 3, "{activity:?}");
    assert_eq!(activity.publications + activity.hits, 1, "{activity:?}");
}

#[test]
fn checked_generic_adjoint_storage_reads_reordered_padded_parent() {
    let provider = Arc::new(SynchronizedCheckedGeneric::new());
    let canonical = crate::BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&provider),
        dense_generic_dynamic_space().homspace().clone(),
    )
    .unwrap();
    let canonical_parent = crate::adjoint_bound_space_dyn_generic_checked(&canonical).unwrap();
    let parent_structure = canonical_parent.space().structure();
    let padded_structure = BlockStructure::from_blocks_with_rank(
        3,
        [1, 0]
            .into_iter()
            .enumerate()
            .map(|(position, index)| {
                let block = parent_structure.block(index).unwrap();
                BlockSpec::with_key(
                    block.key().clone(),
                    block.shape().to_vec(),
                    vec![1, 7, 19],
                    3 + position * 60,
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap();
    let padded_typed = DynamicFusionMapSpace::try_from_layout(
        canonical_parent.space().homspace().clone(),
        padded_structure,
    )
    .unwrap()
    .try_bind_rule(provider.as_ref())
    .unwrap();
    let custom_bound = crate::BoundDynamicFusionMapSpace::bind_generic(
        padded_typed.clone(),
        Arc::clone(&provider),
    )
    .unwrap();
    let parent = canonical_parent
        .rebind_validated(&custom_bound.validated_layout())
        .unwrap();
    let logical = crate::adjoint_bound_space_dyn_generic_checked(&parent).unwrap();

    let canonical_data = (0..60)
        .map(|index| Complex64::new(index as f64 + 1.0, 0.25 - index as f64))
        .collect::<Vec<_>>();
    let mut padded_data = vec![Complex64::new(-99.0, 77.0); parent.space().required_len().unwrap()];
    for source_vertex in 0..2 {
        let padded_block = parent.space().structure().block(1 - source_vertex).unwrap();
        for source_axis_0 in 0..2 {
            for source_axis_1 in 0..3 {
                for source_axis_2 in 0..5 {
                    let canonical_position =
                        source_vertex * 30 + source_axis_2 + 5 * source_axis_0 + 10 * source_axis_1;
                    let padded_position = padded_block.offset()
                        + source_axis_2
                        + 7 * source_axis_0
                        + 19 * source_axis_1;
                    padded_data[padded_position] = canonical_data[canonical_position];
                }
            }
        }
    }
    let alpha = Complex64::new(0.5, -1.25);
    let mut context =
        crate::TreeTransformExecutionContext::<Complex64, RuleIdentity, f64>::default();
    let actual = context
        .tree_transform_owned_checked_generic_in(
            &logical,
            Some(&parent),
            &padded_data,
            &TreeTransformOperation::braid([1, 0], [2], [0, 1], [2]),
            alpha,
        )
        .unwrap();

    assert_eq!(
        actual.1,
        literal_dense_generic_adjoint_braid(&canonical_data, alpha)
    );
}

#[test]
#[allow(clippy::arc_with_non_send_sync)]
fn checked_generic_adjoint_failures_do_not_consume_or_publish_cache_entries() {
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = DenseGenericRule;
    let provider = Arc::new(CheckedPlanSpy::new(&rule));
    let canonical = crate::BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&provider),
        dense_generic_dynamic_space().homspace().clone(),
    )
    .unwrap();
    let parent = crate::adjoint_bound_space_dyn_generic_checked(&canonical).unwrap();
    let logical = crate::adjoint_bound_space_dyn_generic_checked(&parent).unwrap();
    let data = vec![1.0; parent.space().required_len().unwrap()];
    let operation = TreeTransformOperation::braid([1, 0], [2], [0, 1], [2]);
    let mut context = crate::TreeTransformExecutionContext::<f64, RuleIdentity, f64>::default();
    owner_activity();
    context
        .tree_transform_owned_checked_generic_in(&logical, Some(&parent), &data, &operation, 1.0)
        .unwrap();
    let first = owner_activity();
    assert_eq!(first.hits + first.publications, 1, "{first:?}");

    *provider.identity.borrow_mut() = Some(RuleIdentity::of_type::<ToyGenericRule>());
    let error = context
        .tree_transform_owned_checked_generic_in(&logical, Some(&parent), &data, &operation, 1.0)
        .unwrap_err();
    assert!(matches!(
        error,
        CheckedGenericPlanError::Core(CoreError::FusionRuleMismatch { .. })
    ));
    assert_owner_untouched(owner_activity());

    *provider.identity.borrow_mut() = None;
    let error = context
        .tree_transform_owned_checked_generic_in(
            &logical,
            Some(&parent),
            &data[..data.len() - 1],
            &operation,
            1.0,
        )
        .unwrap_err();
    assert!(matches!(
        error,
        CheckedGenericPlanError::Operation(OperationError::ElementCountMismatch { .. })
    ));
    assert_owner_untouched(owner_activity());
}

#[test]
#[allow(clippy::arc_with_non_send_sync)]
fn checked_generic_adjoint_rejects_equal_length_wrong_parent_relation_without_cache_change() {
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = DenseGenericRule;
    let provider = Arc::new(CheckedPlanSpy::new(&rule));
    let canonical = crate::BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&provider),
        dense_generic_dynamic_space().homspace().clone(),
    )
    .unwrap();
    let parent = crate::adjoint_bound_space_dyn_generic_checked(&canonical).unwrap();
    let logical = crate::adjoint_bound_space_dyn_generic_checked(&parent).unwrap();
    let data = vec![1.0; parent.space().required_len().unwrap()];
    let operation = TreeTransformOperation::braid([1, 0], [2], [0, 1], [2]);
    let mut context = crate::TreeTransformExecutionContext::<f64, RuleIdentity, f64>::default();
    owner_activity();
    context
        .tree_transform_owned_checked_generic_in(&logical, Some(&parent), &data, &operation, 1.0)
        .unwrap();
    let first = owner_activity();
    assert_eq!(first.hits + first.publications, 1, "{first:?}");

    let sector = SectorId::new(1);
    let wrong_logical = crate::BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&provider),
        FusionTreeHomSpace::new(
            FusionProductSpace::new(
                [3usize, 2].map(|degeneracy| SectorLeg::new([(sector, degeneracy)], false)),
            ),
            FusionProductSpace::new([SectorLeg::new([(sector, 5)], false)]),
        ),
    )
    .unwrap();
    assert_eq!(
        wrong_logical.space().required_len().unwrap(),
        logical.space().required_len().unwrap()
    );

    let error = context
        .tree_transform_owned_checked_generic_in(
            &wrong_logical,
            Some(&parent),
            &data,
            &operation,
            1.0,
        )
        .unwrap_err();
    assert!(matches!(
        error,
        CheckedGenericPlanError::Operation(OperationError::StructureMismatch {
            tensor: "checked adjoint relation"
        })
    ));
    assert_owner_untouched(owner_activity());
}

#[test]
#[allow(clippy::arc_with_non_send_sync)]
fn checked_generic_direct_transform_does_not_prepare_fusion_operand_projection() {
    let rule = DenseGenericRule;
    let provider = Arc::new(CheckedPlanSpy::new(&rule));
    let source = crate::BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        provider,
        dense_generic_dynamic_space().homspace().clone(),
    )
    .unwrap();
    let data = vec![1.0; source.space().required_len().unwrap()];
    let fusion_operand_projection_prepares_rec = probe::record();

    crate::TreeTransformExecutionContext::<f64, RuleIdentity, f64>::default()
        .tree_transform_owned_checked_generic_in(
            &source,
            None,
            &data,
            &TreeTransformOperation::braid([1, 0], [2], [0, 1], [2]),
            1.0,
        )
        .unwrap();

    assert_eq!(
        fusion_operand_projection_prepares_rec.get(Event::FusionOperandProjectionPrepare),
        0
    );
}

#[test]
#[allow(clippy::arc_with_non_send_sync)]
fn checked_generic_adjoint_storage_handles_empty_layouts() {
    let rule = DenseGenericRule;
    let provider = Arc::new(CheckedPlanSpy::new(&rule));
    let empty_leg = || SectorLeg::new(std::iter::empty::<(SectorId, usize)>(), false);
    let canonical = crate::BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&provider),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([empty_leg(), empty_leg()]),
            FusionProductSpace::new([empty_leg()]),
        ),
    )
    .unwrap();
    let parent = crate::adjoint_bound_space_dyn_generic_checked(&canonical).unwrap();
    let logical = crate::adjoint_bound_space_dyn_generic_checked(&parent).unwrap();
    assert_eq!(parent.space().required_len().unwrap(), 0);
    let mut context =
        crate::TreeTransformExecutionContext::<Complex64, RuleIdentity, f64>::default();

    for operation in [
        TreeTransformOperation::braid([1, 0], [2], [0, 1], [2]),
        TreeTransformOperation::transpose([2], [1, 0]),
    ] {
        let output = context
            .tree_transform_owned_checked_generic_in(
                &logical,
                Some(&parent),
                &[],
                &operation,
                Complex64::new(0.5, -1.25),
            )
            .unwrap();
        assert!(output.1.is_empty());
    }
}

#[test]
#[allow(clippy::arc_with_non_send_sync)]
fn checked_generic_adjoint_storage_rejects_distinct_provider_allocation() {
    let rule = DenseGenericRule;
    let logical_provider = Arc::new(CheckedPlanSpy::new(&rule));
    let canonical = crate::BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&logical_provider),
        dense_generic_dynamic_space().homspace().clone(),
    )
    .unwrap();
    let parent = crate::adjoint_bound_space_dyn_generic_checked(&canonical).unwrap();
    let logical = crate::adjoint_bound_space_dyn_generic_checked(&parent).unwrap();
    let storage_provider = Arc::new(CheckedPlanSpy::new(&rule));
    let distinct_parent = crate::BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        storage_provider,
        parent.space().homspace().clone(),
    )
    .unwrap();
    let data = vec![1.0; distinct_parent.space().required_len().unwrap()];

    let error = crate::TreeTransformExecutionContext::<f64, RuleIdentity, f64>::default()
        .tree_transform_owned_checked_generic_in(
            &logical,
            Some(&distinct_parent),
            &data,
            &TreeTransformOperation::braid([1, 0], [2], [0, 1], [2]),
            1.0,
        )
        .unwrap_err();

    assert!(matches!(
        error,
        CheckedGenericPlanError::Operation(OperationError::StructureMismatch {
            tensor: "checked adjoint provider"
        })
    ));
}

/// What: a checked transform whose commit fails after its groups were
/// composed publishes none of them (publication is deferred to the commit), so a
/// retry composes them again; the committed retry publishes them.
#[test]
#[allow(clippy::arc_with_non_send_sync)]
fn checked_generic_commit_failure_publishes_no_composed_coefficients() {
    struct CommitFailureIdentity;
    let rule = DenseGenericRule;
    let provider = Arc::new(CheckedPlanSpy::new(&rule));
    *provider.identity.borrow_mut() = Some(RuleIdentity::of_type::<CommitFailureIdentity>());
    let source = crate::BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&provider),
        dense_generic_dynamic_space().homspace().clone(),
    )
    .unwrap();
    let data = vec![1.0; source.space().required_len().unwrap()];
    let operation = TreeTransformOperation::braid([0, 2], [1], [0, 1], [2]);
    let mut context = crate::TreeTransformExecutionContext::<f64, RuleIdentity>::default();
    crate::probe::take_coefficient_group_activity();

    provider.calls.set([0; CheckedPlanCall::COUNT]);
    provider
        .restyle_at
        .set(Some((CheckedPlanCall::R, FusionStyleKind::Unique)));
    let error = context
        .tree_transform_owned_checked_generic_in(&source, None, &data, &operation, 1.0)
        .unwrap_err();
    assert!(
        matches!(
            error,
            CheckedGenericPlanError::Core(CoreError::UnsupportedFusionStyle { .. })
        ),
        "{error:?}"
    );
    assert!(provider.call_count(CheckedPlanCall::R) > 0);
    let failed = crate::probe::take_coefficient_group_activity();
    assert!(failed.misses > 0);
    assert_eq!((failed.hits, failed.publications), (0, 0));

    provider.restyle_at.set(None);
    provider.fusion_style.set(None);
    provider.calls.set([0; CheckedPlanCall::COUNT]);
    context
        .tree_transform_owned_checked_generic_in(&source, None, &data, &operation, 1.0)
        .unwrap();
    let committed = crate::probe::take_coefficient_group_activity();
    assert_eq!(
        (committed.hits, committed.misses, committed.publications),
        (0, failed.misses, failed.misses)
    );
    assert!(provider.call_count(CheckedPlanCall::F) > 0);
}

#[test]
#[allow(clippy::arc_with_non_send_sync)]
fn checked_generic_adjoint_late_provider_failure_does_not_publish_cache() {
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = DenseGenericRule;
    let provider = Arc::new(CheckedPlanSpy::new(&rule));
    let canonical = crate::BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&provider),
        dense_generic_dynamic_space().homspace().clone(),
    )
    .unwrap();
    let parent = crate::adjoint_bound_space_dyn_generic_checked(&canonical).unwrap();
    let logical = crate::adjoint_bound_space_dyn_generic_checked(&parent).unwrap();
    let data = vec![1.0; parent.space().required_len().unwrap()];
    let mut context = crate::TreeTransformExecutionContext::<f64, RuleIdentity, f64>::default();
    owner_activity();
    provider.calls.set([0; CheckedPlanCall::COUNT]);
    provider.fail.set(Some((CheckedPlanCall::F, 1)));

    let error = context
        .tree_transform_owned_checked_generic_in(
            &logical,
            Some(&parent),
            &data,
            &TreeTransformOperation::transpose([0], [2, 1]),
            1.0,
        )
        .unwrap_err();

    assert!(matches!(
        error,
        CheckedGenericPlanError::Provider(CheckedPlanSpyError(CheckedPlanCall::F))
    ));
    assert!(provider.call_count(CheckedPlanCall::F) > 0);
    assert_eq!(owner_activity().publications, 0);
}

#[test]
#[allow(clippy::arc_with_non_send_sync)] // The API requires Arc; this single-threaded spy uses Cells for deterministic failures.
fn checked_generic_owned_failure_does_not_publish_destination_state() {
    use symtn_core::{structure_cache_info, StructureCacheKind};

    const ISOLATED: &str = "SYMTN_CHECKED_GENERIC_OWNED_FAILURE_ISOLATED";
    // What: exact global-cache snapshots run outside the parallel unit-test process.
    if std::env::var_os(ISOLATED).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "tests::tree_transform_plan::checked_generic::checked_generic_owned_failure_does_not_publish_destination_state",
            ])
            .env(ISOLATED, "1")
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            output.status.success() && stdout.contains("test result: ok. 1 passed; 0 failed;"),
            "isolated test did not execute exactly once: {}\nstdout:\n{stdout}\nstderr:\n{stderr}",
            output.status
        );
        return;
    }

    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = DenseGenericRule;
    let src_space = dense_generic_dynamic_space();
    let src_data = vec![1.0; src_space.required_len().unwrap()];
    let source_before = src_space.clone();
    let data_before = src_data.clone();
    let operation = TreeTransformOperation::braid([0, 2], [1], [0, 1], [2]);

    for mismatch in [
        Some(symtn_core::RuleIdentity::of_type::<ToyGenericRule>()),
        None,
    ] {
        let style_mismatch = mismatch.is_none();
        let provider = Arc::new(CheckedPlanSpy::new(&rule));
        let test_space = crate::BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
            Arc::clone(&provider),
            src_space.homspace().clone(),
        )
        .unwrap();
        provider.calls.set([0; CheckedPlanCall::COUNT]);
        if let Some(identity) = mismatch {
            *provider.identity.borrow_mut() = Some(identity);
        } else {
            provider.fusion_style.set(Some(FusionStyleKind::Unique));
        }
        provider.fail.set(Some((CheckedPlanCall::N, 1)));
        let layout_before = structure_cache_info(StructureCacheKind::SectorStructure);
        let complete_before = structure_cache_info(StructureCacheKind::DegeneracyStructure);
        owner_activity();

        let error = crate::TreeTransformExecutionContext::<f64, RuleIdentity, f64>::default()
            .tree_transform_owned_checked_generic_in(&test_space, None, &src_data, &operation, 1.0)
            .unwrap_err();
        if style_mismatch {
            // The checked transform's own admission reports the tensors
            // variant (#1950), not the core one.
            assert!(
                matches!(
                    &error,
                    CheckedGenericPlanError::Operation(OperationError::UnsupportedFusionStyle {
                        operation: rejected,
                        style: FusionStyleKind::Unique,
                    }) if **rejected == operation
                ),
                "{error:?}"
            );
        } else {
            assert!(matches!(
                error,
                CheckedGenericPlanError::Core(CoreError::FusionRuleMismatch { .. })
            ));
        }
        assert_eq!(provider.calls.get(), [0; CheckedPlanCall::COUNT]);
        // A layout walk that succeeded before the failure may publish its
        // pure-data layout (#2030); nothing else is published.
        let layout_after = structure_cache_info(StructureCacheKind::SectorStructure);
        assert!(layout_after.admissions() <= layout_before.admissions() + 1);
        assert_eq!(layout_after.rejections(), layout_before.rejections());
        let complete_after = structure_cache_info(StructureCacheKind::DegeneracyStructure);
        assert_eq!(complete_after.entries(), complete_before.entries());
        assert_eq!(
            complete_after.charged_bytes(),
            complete_before.charged_bytes()
        );
        assert_eq!(complete_after.admissions(), complete_before.admissions());
        assert_owner_untouched(owner_activity());
        assert_eq!(src_space, source_before);
        assert_eq!(src_data, data_before);
    }

    for failure in [
        Err((CheckedPlanCall::Dual, 1)),
        Err((CheckedPlanCall::N, 1)),
        Err((CheckedPlanCall::SqrtDim, 1)),
        Err((CheckedPlanCall::InvSqrtDim, 1)),
        Err((CheckedPlanCall::F, 2)),
        Err((CheckedPlanCall::R, 2)),
        Ok(MalformedCheckedSymbol::F),
        Ok(MalformedCheckedSymbol::R),
    ] {
        let provider = Arc::new(CheckedPlanSpy::new(&rule));
        let bound_src = crate::BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
            Arc::clone(&provider),
            src_space.homspace().clone(),
        )
        .unwrap();
        provider.calls.set([0; CheckedPlanCall::COUNT]);
        match failure {
            Err((call, nth)) => provider.fail.set(Some((call, nth))),
            Ok(symbol) => provider.malformed.set(Some(symbol)),
        }
        let layout_before = structure_cache_info(StructureCacheKind::SectorStructure);
        let complete_before = structure_cache_info(StructureCacheKind::DegeneracyStructure);
        owner_activity();

        let error = crate::TreeTransformExecutionContext::<f64, RuleIdentity, f64>::default()
            .tree_transform_owned_checked_generic_in(&bound_src, None, &src_data, &operation, 1.0)
            .unwrap_err();
        match failure {
            Err((call, _)) => assert!(matches!(
                error,
                CheckedGenericPlanError::Provider(CheckedPlanSpyError(found)) if found == call
            )),
            Ok(MalformedCheckedSymbol::F) => assert!(matches!(
                error,
                CheckedGenericPlanError::SymbolShape { symbol: "F", .. }
            )),
            Ok(MalformedCheckedSymbol::R) => assert!(matches!(
                error,
                CheckedGenericPlanError::SymbolShape { symbol: "R", .. }
            )),
        }
        // A layout walk that succeeded before the failure may publish its
        // pure-data layout (#2030); nothing else is published.
        let layout_after = structure_cache_info(StructureCacheKind::SectorStructure);
        assert!(layout_after.admissions() <= layout_before.admissions() + 1);
        assert_eq!(layout_after.rejections(), layout_before.rejections());
        let complete_after = structure_cache_info(StructureCacheKind::DegeneracyStructure);
        assert_eq!(complete_after.entries(), complete_before.entries());
        assert_eq!(
            complete_after.charged_bytes(),
            complete_before.charged_bytes()
        );
        assert_eq!(complete_after.admissions(), complete_before.admissions());
        assert_owner_untouched(owner_activity());
        assert_eq!(src_space, source_before);
        assert_eq!(src_data, data_before);
    }

    let dense_rule = DenseGenericRule;
    let dense_homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(SectorId::new(1), 7)], false),
            SectorLeg::new([(SectorId::new(1), 11)], true),
        ]),
        FusionProductSpace::new([SectorLeg::new([(SectorId::new(1), 13)], false)]),
    );
    let provider = Arc::new(CheckedPlanSpy::new(&dense_rule));
    let dense_space = crate::BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&provider),
        dense_homspace,
    )
    .unwrap();
    let dense_data = vec![1.0; dense_space.space().required_len().unwrap()];
    provider.calls.set([0; CheckedPlanCall::COUNT]);
    let layout_before = structure_cache_info(StructureCacheKind::SectorStructure);
    let complete_before = structure_cache_info(StructureCacheKind::DegeneracyStructure);
    provider
        .fail
        .set(Some((CheckedPlanCall::FrobeniusSchur, 1)));
    let error = crate::TreeTransformExecutionContext::<f64, RuleIdentity, f64>::default()
        .tree_transform_owned_checked_generic_in(
            &dense_space,
            None,
            &dense_data,
            &TreeTransformOperation::permute([1, 0], [2]),
            1.0,
        )
        .unwrap_err();
    assert!(matches!(
        error,
        CheckedGenericPlanError::Provider(CheckedPlanSpyError(CheckedPlanCall::FrobeniusSchur))
    ));
    // The destination layout walk succeeded before the failure and may
    // publish its pure-data layout (#2030); nothing else is published.
    let layout_after = structure_cache_info(StructureCacheKind::SectorStructure);
    assert!(layout_after.admissions() <= layout_before.admissions() + 1);
    assert_eq!(layout_after.rejections(), layout_before.rejections());
    let complete_after = structure_cache_info(StructureCacheKind::DegeneracyStructure);
    assert_eq!(complete_after.entries(), complete_before.entries());
    assert_eq!(
        complete_after.charged_bytes(),
        complete_before.charged_bytes()
    );
    assert_eq!(complete_after.admissions(), complete_before.admissions());
}

#[test]
fn checked_generic_plan_preserves_provider_sources_and_rejects_bad_f_r_shapes() {
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = DenseGenericRule;
    let pairs = dense_generic_rank3_pairs(&rule);
    let structure = packed_fixture_structure(
        3,
        pairs
            .into_iter()
            .map(BlockKey::from)
            .map(|key| (key, vec![1usize; 3])),
    )
    .unwrap();
    let r_only = TreeTransformOperation::permute([1, 0, 2], []);
    let inner_f_r = TreeTransformOperation::braid([0, 2, 1], [], [0, 1, 2], []);
    owner_activity();

    for (operation, call) in [
        (r_only.clone(), CheckedPlanCall::R),
        (inner_f_r.clone(), CheckedPlanCall::F),
    ] {
        let provider = CheckedPlanSpy::new(&rule);
        // Fail after the first lookup has succeeded: compilation must still
        // return no partially assembled plan.
        provider.fail.set(Some((call, 2)));
        let error =
            build_checked_generic_tree_pair_transform_group_plan(&provider, operation, &structure)
                .unwrap_err();
        assert!(
            matches!(
                error,
                CheckedGenericPlanError::Provider(CheckedPlanSpyError(found)) if found == call
            ),
            "{error:?}"
        );
        assert_eq!(
            std::error::Error::source(&error)
                .and_then(|source| source.downcast_ref::<CheckedPlanSpyError>()),
            Some(&CheckedPlanSpyError(call))
        );
        // Checked planning is a standalone compile step: a failed provider
        // query cannot publish a partial Runtime cache entry or statistic.
        assert_owner_untouched(owner_activity());
    }

    for (operation, malformed, symbol) in [
        (r_only, MalformedCheckedSymbol::R, "R"),
        (inner_f_r, MalformedCheckedSymbol::F, "F"),
    ] {
        let provider = CheckedPlanSpy::new(&rule);
        provider.malformed.set(Some(malformed));
        let error =
            build_checked_generic_tree_pair_transform_group_plan(&provider, operation, &structure)
                .unwrap_err();
        assert!(matches!(
            error,
            CheckedGenericPlanError::SymbolShape {
                symbol: found,
                ..
            } if found == symbol
        ));
        assert_owner_untouched(owner_activity());
    }
}

#[test]
fn checked_generic_plan_rejects_style_before_structure_or_symbol_queries() {
    let rule = DenseGenericRule;
    let pairs = dense_generic_rank3_pairs(&rule);
    let structure = packed_fixture_structure(
        3,
        pairs
            .into_iter()
            .map(BlockKey::from)
            .map(|key| (key, vec![1usize; 3])),
    )
    .unwrap();
    let operation = TreeTransformOperation::permute([1, 0, 2], []);

    let wrong_fusion_style = CheckedPlanSpy::new(&rule);
    wrong_fusion_style
        .fusion_style
        .set(Some(FusionStyleKind::Simple));
    wrong_fusion_style
        .braiding_style
        .set(Some(BraidingStyleKind::NoBraiding));
    let error = build_checked_generic_tree_pair_transform_group_plan(
        &wrong_fusion_style,
        operation.clone(),
        &structure,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        CheckedGenericPlanError::Operation(OperationError::UnsupportedFusionStyle {
            style: FusionStyleKind::Simple,
            ..
        })
    ));
    assert_eq!(wrong_fusion_style.calls.get(), [0; CheckedPlanCall::COUNT]);

    // "Planar" providers have `NoBraiding`: a permutation is rejected before
    // either structural N/dual access or any F/R/rigidity lookup.
    let planar = CheckedPlanSpy::new(&rule);
    planar
        .braiding_style
        .set(Some(BraidingStyleKind::NoBraiding));
    let error =
        build_checked_generic_tree_pair_transform_group_plan(&planar, operation, &structure)
            .unwrap_err();
    assert!(matches!(
        error,
        CheckedGenericPlanError::Operation(OperationError::UnsupportedBraidingStyle {
            style: BraidingStyleKind::NoBraiding,
            ..
        })
    ));
    assert_eq!(planar.calls.get(), [0; CheckedPlanCall::COUNT]);

    let noncyclic = CheckedPlanSpy::new(&rule);
    let error = build_checked_generic_tree_pair_transform_group_plan(
        &noncyclic,
        TreeTransformOperation::transpose([1, 0], [2]),
        &structure,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        CheckedGenericPlanError::Operation(OperationError::InvalidPermutation { .. })
    ));
    // Syntax rejection precedes structural (Dual/N), rigidity, and F/R
    // provider queries.
    assert_eq!(noncyclic.calls.get(), [0; CheckedPlanCall::COUNT]);

    let cyclic = CheckedPlanSpy::new(&rule);
    let cyclic_plan = build_checked_generic_tree_pair_transform_group_plan(
        &cyclic,
        TreeTransformOperation::transpose([2], [1, 0]),
        &structure,
    )
    .unwrap();
    assert!(!cyclic_plan.specs().is_empty());
    assert!(cyclic.call_count(CheckedPlanCall::F) > 0);

    let empty_cyclic = CheckedPlanSpy::new(&rule);
    let empty_plan = build_checked_generic_tree_pair_transform_group_plan(
        &empty_cyclic,
        TreeTransformOperation::transpose([2], [1, 0]),
        &BlockStructure::empty(3),
    )
    .unwrap();
    assert!(empty_plan.specs().is_empty());
    assert_eq!(empty_cyclic.calls.get(), [0; CheckedPlanCall::COUNT]);

    let second_tree_fails_structure = CheckedPlanSpy::new(&rule);
    // Each rank-3 tree has two vertices. Failure 3 is therefore the first N
    // query of the second source tree in the all-source structural preflight.
    second_tree_fails_structure
        .fail
        .set(Some((CheckedPlanCall::N, 3)));
    let error = build_checked_generic_tree_pair_transform_group_plan(
        &second_tree_fails_structure,
        TreeTransformOperation::permute([1, 0, 2], []),
        &structure,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        CheckedGenericPlanError::Provider(CheckedPlanSpyError(CheckedPlanCall::N))
    ));
    for call in [
        CheckedPlanCall::SqrtDim,
        CheckedPlanCall::InvSqrtDim,
        CheckedPlanCall::FrobeniusSchur,
        CheckedPlanCall::F,
        CheckedPlanCall::R,
    ] {
        assert_eq!(second_tree_fails_structure.call_count(call), 0);
    }
}

#[test]
fn checked_generic_plan_repartition_queries_checked_rigid_primitives() {
    let rule = DenseGenericRule;
    let pair = dense_generic_dual_source_pair(&rule);
    let structure = packed_fixture_structure(3, [(BlockKey::from(pair), vec![1usize; 3])]).unwrap();
    let operation = TreeTransformOperation::permute([1, 0], [2]);
    for call in [
        CheckedPlanCall::Dual,
        CheckedPlanCall::N,
        CheckedPlanCall::SqrtDim,
        CheckedPlanCall::InvSqrtDim,
        CheckedPlanCall::FrobeniusSchur,
        // On this repartition, the F lookup derives the B move; B is not a
        // separate provider method.
        CheckedPlanCall::F,
        CheckedPlanCall::R,
    ] {
        let provider = CheckedPlanSpy::new(&rule);
        provider.fail.set(Some((call, 1)));
        let error = build_checked_generic_tree_pair_transform_group_plan(
            &provider,
            operation.clone(),
            &structure,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            CheckedGenericPlanError::Provider(CheckedPlanSpyError(found)) if found == call
        ));
    }

    let provider = CheckedPlanSpy::new(&rule);
    build_checked_generic_tree_pair_transform_group_plan(&provider, operation, &structure).unwrap();
    for call in [
        CheckedPlanCall::Dual,
        CheckedPlanCall::N,
        CheckedPlanCall::SqrtDim,
        CheckedPlanCall::InvSqrtDim,
        CheckedPlanCall::FrobeniusSchur,
        CheckedPlanCall::F,
        CheckedPlanCall::R,
    ] {
        assert!(provider.call_count(call) > 0, "{call:?} was not live");
    }
}

#[test]
fn generic_transpose_accepts_nonsymmetric_braiding_but_permute_rejects_it() {
    let rule = AnyonicGenericRule;
    let src_pair = vacuum_rank2_src_pair_for_rule(&rule);
    let structure =
        packed_fixture_structure(2, [(BlockKey::from(src_pair.clone()), vec![1, 1])]).unwrap();

    let transpose = TreeTransformOperation::transpose([1, 0], []);
    let expected = generic_transpose_tree_pair(&rule, &src_pair, &[1, 0], &[]).unwrap();
    let plan = build_generic_tree_pair_transform_group_plan(&rule, transpose, &structure).unwrap();
    assert_eq!(plan.specs().len(), 1);
    assert_eq!(plan.specs()[0].dst_keys(), &[expected[0].0.clone()]);
    assert_eq!(
        plan.specs()[0].recoupling_coefficients_dst_src(),
        &[expected[0].1]
    );

    let permute = TreeTransformOperation::permute([1, 0], []);
    let err = build_generic_tree_pair_transform_group_plan(&rule, permute.clone(), &structure)
        .unwrap_err();
    assert_eq!(
        err,
        OperationError::UnsupportedBraidingStyle {
            operation: Box::new(permute),
            style: BraidingStyleKind::Anyonic,
        }
    );
}
