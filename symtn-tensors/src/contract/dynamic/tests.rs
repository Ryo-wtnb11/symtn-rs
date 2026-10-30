use super::super::scratch::DynamicFusionScratchWorkspace;
use super::*;
use crate::probe::{self, Event};
use std::cell::Cell;
use symtn_core::{
    FusionProductSpace, FusionRule, SU2FusionRule, SectorLeg, U1FusionRule, U1Irrep, Z2FusionRule,
};
use symtn_operations::TensorContractSpec;

use crate::tree_context::TreeTransformExecutionContext;
use crate::{BoundDynamicFusionMapSpace, DenseTreeTransformOperations};
use symtn_operations::OutputAxisOrder;

use super::super::dynamic_space::{checked_metadata_dispatcher, MetadataOutput, MetadataRequest};

thread_local! {
    static EXECUTION_PRIMER_CALLS: Cell<usize> = const { Cell::new(0) };
}

fn counting_su2_primer(
    rule: &SU2FusionRule,
    request: MetadataRequest<'_>,
) -> Result<MetadataOutput, OperationError> {
    EXECUTION_PRIMER_CALLS.with(|calls| calls.set(calls.get() + 1));
    checked_metadata_dispatcher(rule, request)
}

fn reset_execution_primer_calls() {
    EXECUTION_PRIMER_CALLS.with(|calls| calls.set(0));
}

fn execution_primer_calls() -> usize {
    EXECUTION_PRIMER_CALLS.with(Cell::get)
}

fn run_nan_poisoned_destination_case(
    scratch: &mut DynamicFusionScratchWorkspace<f64>,
    open_degeneracy: usize,
) -> (Vec<f64>, usize, usize) {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let provider = Arc::new(U1FusionRule);
    let wide = || {
        SectorLeg::new(
            [
                (U1Irrep::new(0).sector_id(), 1),
                (U1Irrep::new(1).sector_id(), open_degeneracy),
                (U1Irrep::new(2).sector_id(), 1),
            ],
            false,
        )
    };
    let narrow = || {
        SectorLeg::new(
            [
                (U1Irrep::new(0).sector_id(), 2),
                (U1Irrep::new(1).sector_id(), 1),
            ],
            false,
        )
    };
    let space = |codomain, domain| {
        BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
            Arc::clone(&provider),
            FusionTreeHomSpace::new(
                FusionProductSpace::new(codomain),
                FusionProductSpace::new(domain),
            ),
        )
        .unwrap()
    };
    let lhs = space(vec![wide(), wide()], vec![narrow()]);
    let rhs = space(vec![narrow()], vec![wide()]);
    let output_order = OutputAxisOrder::from_axes(&[1, 0, 2]);
    let axes = TensorContractSpec::new(&[2], &[0], output_order);
    let dst = BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
        &lhs,
        &rhs,
        &[2],
        &[0],
        output_order,
    )
    .unwrap();
    let plan = super::super::fusion::prepare_tensorcontract_fusion_plan_dyn_raw(
        provider.as_ref(),
        dst.space(),
        lhs.space(),
        rhs.space(),
        axes,
    )
    .unwrap();
    assert!(!plan.output_transform_is_identity());

    let mut tree_context =
        TreeTransformExecutionContext::new(DenseTreeTransformOperations::default_executor());
    let artifact =
        compile_dynamic_tree_execution_artifact::<symtn_core::MultiplicityFreeAdmissionMode, _>(
            engine,
            tree_context.planning(),
            provider.as_ref(),
            encoded_layout_primer::<U1FusionRule>,
            &plan,
            dst.space(),
            lhs.space(),
            lhs.space().structure(),
            rhs.space(),
            rhs.space().structure(),
        )
        .unwrap();
    let core_dst = artifact.core_dst.as_ref().unwrap();
    let core_len = core_dst.space.required_len().unwrap();
    let inactive = artifact.block_plan.inactive_destination_regions().len();
    assert!(inactive > 0, "fixture must include an inactive core block");
    let lhs_data = (0..lhs.space().required_len().unwrap())
        .map(|index| (index as f64 * 0.37 + 0.1).sin())
        .collect::<Vec<_>>();
    let rhs_data = (0..rhs.space().required_len().unwrap())
        .map(|index| (index as f64 * 0.23 + 0.2).cos())
        .collect::<Vec<_>>();
    let mut execute =
        |tree_context: &mut TreeTransformExecutionContext<f64, crate::RuleIdentity>,
         scratch: &mut DynamicFusionScratchWorkspace<f64>,
         output: &mut [f64]| {
            execute_dynamic_tree_execution_artifact(
                engine,
                tree_context,
                &mut DenseTreeTransformOperations::default(),
                &mut super::super::backend::TensorContractWorkspace::default(),
                &mut FusionBlockContractWorkspace::default(),
                scratch,
                &artifact,
                dst.space().structure(),
                output,
                &lhs_data,
                &rhs_data,
                1.0,
                0.0,
            )
            .unwrap();
        };
    let mut expected = vec![0.0; dst.space().required_len().unwrap()];
    execute(
        &mut tree_context,
        &mut DynamicFusionScratchWorkspace::default(),
        &mut expected,
    );
    let mut actual = vec![0.0; expected.len()];
    execute(&mut tree_context, scratch, &mut actual);
    assert!(
        scratch
            .dst_data_mut()
            .expect("replay retains destination scratch")
            .iter()
            .all(|value| value.is_finite()),
        "strong zero must initialize active and inactive core blocks"
    );
    assert_eq!(
        actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    (actual, core_len, inactive)
}

#[test]
fn dynamic_core_strong_zero_overwrites_nan_scratch_after_shape_change() {
    let mut scratch = DynamicFusionScratchWorkspace::default();
    let (_, first_len, _) = run_nan_poisoned_destination_case(&mut scratch, 4);
    scratch
        .dst_data_mut()
        .expect("first replay retains destination scratch")
        .fill(f64::NAN);
    let (same_shape, same_len, same_inactive) = run_nan_poisoned_destination_case(&mut scratch, 4);
    assert_eq!(first_len, same_len);
    assert!(same_inactive > 0);
    assert!(same_shape.iter().all(|value| value.is_finite()));
    scratch
        .dst_data_mut()
        .expect("same-shape replay retains destination scratch")
        .fill(f64::NAN);

    let (actual, second_len, inactive) = run_nan_poisoned_destination_case(&mut scratch, 2);

    assert_ne!(first_len, second_len, "fixture must change scratch shape");
    assert!(inactive > 0);
    assert!(actual.iter().all(|value| value.is_finite()));
}

#[test]
fn direct_prelowered_source_reuses_ordinary_transform_entry() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rule = U1FusionRule;
    let charges = [-1, 0, 1].map(|charge| U1Irrep::new(charge).sector_id());
    let codomain = SectorLeg::new(charges.map(|sector| (sector, 1)), false);
    let domain = SectorLeg::new(charges.map(|sector| (rule.dual(sector), 1)), false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([codomain]),
        FusionProductSpace::new([domain]),
    );
    let canonical = DynamicFusionMapSpace::from_final_homspace(&rule, homspace.clone()).unwrap();
    let reversed_keys = (0..canonical.structure().block_count())
        .rev()
        .map(|index| canonical.structure().block(index).unwrap().key().clone())
        .collect::<Vec<_>>();
    let reordered = DynamicFusionMapSpace::try_from_layout(
        homspace,
        crate::tests::packed_fixture_structure(
            2,
            reversed_keys.into_iter().map(|key| (key, vec![1, 1])),
        )
        .unwrap(),
    )
    .unwrap()
    .try_bind_rule(&rule)
    .unwrap();
    let source = reordered;
    let layout = super::super::dynamic_space::FusionOperand::direct(&source)
        .prepare(
            &rule,
            super::super::dynamic_space::encoded_layout_primer::<U1FusionRule>,
        )
        .unwrap();
    assert!(layout.is_direct());
    let operation = TreeTransformOperation::transpose([1], [0]);
    let mut tree_context: TreeTransformExecutionContext<
        f64,
        <U1FusionRule as TreeTransformRuleCacheKey>::Key,
        f64,
        _,
    > = TreeTransformExecutionContext::new(DenseTreeTransformOperations::default_executor());
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    symtn_core::testing::mark_structure_canonical(source.structure());
    let ordinary_cold = compile_transformed_source(
        engine,
        tree_context.planning(),
        &rule,
        &source,
        source.structure(),
        &operation,
        false,
        super::super::dynamic_space::encoded_layout_primer::<U1FusionRule>,
    )
    .unwrap();
    let direct_hit = compile_prelowered_source_transform(
        engine,
        tree_context.planning(),
        &rule,
        &layout,
        &operation,
        super::super::dynamic_space::encoded_layout_primer::<U1FusionRule>,
    )
    .unwrap();

    // What: the Direct side of a mixed prelowered contraction uses the
    // ordinary parent-storage compiler and therefore the exact same entry.
    assert!(Arc::ptr_eq(
        ordinary_cold
            .transform_structure
            .as_ref()
            .unwrap()
            .replay_core(),
        direct_hit
            .transform_structure
            .as_ref()
            .unwrap()
            .replay_core()
    ));
    assert_eq!(ordinary_cold.space, direct_hit.space);
}

#[test]
fn execution_layout_primer_runs_once_per_derivation() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: transformed-source and nonidentity-output core spaces invoke
    // the selected primer once per derivation.
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = SU2FusionRule;
    let homspace = FusionTreeHomSpace::from_sector_ids([(7, 3); 4], []);
    let source = DynamicFusionMapSpace::from_final_homspace(&rule, homspace).unwrap();
    let scalar_homspace = FusionTreeHomSpace::from_sector_ids([], []);
    let scalar = DynamicFusionMapSpace::from_final_homspace(&rule, scalar_homspace).unwrap();
    let operation = TreeTransformOperation::permute([0, 2, 1, 3], []);
    let axes = TensorContractSpec::new(&[], &[], OutputAxisOrder::from_axes(&[0, 2, 1, 3]));
    let provider = Arc::new(rule);
    let source_bound =
        super::super::dynamic_space::BoundDynamicFusionMapSpace::bind_multiplicity_free(
            source.clone(),
            Arc::clone(&provider),
        )
        .unwrap();
    let scalar_bound =
        super::super::dynamic_space::BoundDynamicFusionMapSpace::bind_multiplicity_free(
            scalar.clone(),
            Arc::clone(&provider),
        )
        .unwrap();
    let output_bound = super::super::dynamic_space::BoundDynamicFusionMapSpace::contracted_multiplicity_free_ordered(
        &source_bound,
        &scalar_bound,
        axes.lhs_contracting_axes(),
        axes.rhs_contracting_axes(),
        axes.output_permutation(),
    )
    .unwrap();
    let output = output_bound.space().clone();
    let plan = super::super::fusion::prepare_tensorcontract_fusion_plan_dyn_raw(
        &rule, &output, &source, &scalar, axes,
    )
    .unwrap();
    assert!(!plan.output_transform_is_identity());

    symtn_core::clear_structure_caches();
    let mut tree_context =
        TreeTransformExecutionContext::new(DenseTreeTransformOperations::default_executor());
    reset_execution_primer_calls();
    let mut derive_source =
        |tree_context: &mut TreeTransformExecutionContext<f64, crate::RuleIdentity>| {
            compile_transformed_source(
                engine,
                tree_context.planning(),
                &rule,
                &source,
                source.structure(),
                &operation,
                false,
                counting_su2_primer,
            )
            .unwrap()
        };
    let cold_transform = derive_source(&mut tree_context);
    assert_eq!(execution_primer_calls(), 1);
    let warm_transform = derive_source(&mut tree_context);
    // What: no context retains the derivation, so the warm call primes its layout key again; the transformer
    // itself is the published one.
    assert_eq!(execution_primer_calls(), 2);
    assert_eq!(cold_transform.space, warm_transform.space);
    assert!(Arc::ptr_eq(
        cold_transform
            .transform_structure
            .as_ref()
            .unwrap()
            .replay_core(),
        warm_transform
            .transform_structure
            .as_ref()
            .unwrap()
            .replay_core()
    ));

    symtn_core::clear_structure_caches();
    reset_execution_primer_calls();
    let mut derive_core =
        |tree_context: &mut TreeTransformExecutionContext<f64, crate::RuleIdentity>| {
            compile_core_dst(
                engine,
                tree_context.planning(),
                &rule,
                &source,
                &scalar,
                &plan,
                &output,
                counting_su2_primer,
            )
            .unwrap()
        };
    let cold_core = derive_core(&mut tree_context);
    assert_eq!(execution_primer_calls(), 1);
    let warm_core = derive_core(&mut tree_context);
    assert_eq!(execution_primer_calls(), 2);
    assert_eq!(cold_core.space, warm_core.space);

    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();
    reset_execution_primer_calls();
    // The DynamicTree artifact compile primes layouts with the destination's
    // capability (#1858). The typed route takes `copyC` for this
    // permute-only contraction, so the artifact compile is driven directly.
    let output_bound = output_bound
        .clone()
        .with_test_layout_primer(counting_su2_primer);
    let mut compile =
        |context: &mut crate::TensorContractFusionExecutionContext<f64, crate::RuleIdentity>,
         output: &super::super::dynamic_space::BoundDynamicFusionMapSpace<_>,
         axes| {
            let resolution = context
                .compile_storage_contract_resolution(
                    engine,
                    output,
                    crate::FusionOperand::direct(source_bound.space()),
                    crate::FusionOperand::direct(scalar_bound.space()),
                    axes,
                )
                .unwrap();
            assert!(resolution.is_dynamic_tree());
        };
    let mut context =
        crate::TensorContractFusionExecutionContext::<f64, crate::RuleIdentity>::default();
    compile(&mut context, &output_bound, axes);
    let cold_calls = execution_primer_calls();
    assert!(cold_calls > 0);
    compile(&mut context, &output_bound, axes);
    // What: each compile derives its spaces again (nothing is retained per
    // context), priming the same layouts.
    assert_eq!(execution_primer_calls(), 2 * cold_calls);
}

#[test]
fn borrowable_core_layout_accepts_equal_structure_across_intern_reset() {
    let _guard = crate::test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let rule = Z2FusionRule;
    let build = || {
        DynamicFusionMapSpace::from_final_homspace(
            &rule,
            FusionTreeHomSpace::from_sector_ids([(0, 1)], [(0, 1)]),
        )
        .unwrap()
        .try_bind_rule(&rule)
        .unwrap()
    };
    let source = build();
    let source_structure = Arc::clone(source.structure());

    symtn_core::clear_structure_caches();
    let core = build();

    // What: equal live layouts created on opposite sides of an intern reset
    // remain borrowable even though their process-local content ids differ.
    assert_ne!(source_structure.content_id(), core.structure().content_id());
    assert!(source_is_borrowable_core_layout(
        &source,
        &source_structure,
        &core,
        &TreeTransformOperation::permute([0], [1]),
        false,
    ));
}

#[test]
fn borrowable_core_layout_defers_homspace_identity_until_after_cheap_gates() {
    // What: nonidentity and conjugating sources skip HomSpace identity,
    // while an otherwise borrowable identity layout performs one comparison.
    let rule = Z2FusionRule;
    let layout = DynamicFusionMapSpace::from_final_homspace(
        &rule,
        FusionTreeHomSpace::from_sector_ids([(0, 1)], [(0, 1)]),
    )
    .unwrap()
    .try_bind_rule(&rule)
    .unwrap();
    let source = layout;
    let source_structure = Arc::clone(source.structure());

    let source_layout_homspace_id_comparisons_rec = probe::record();
    assert!(!source_is_borrowable_core_layout(
        &source,
        &source_structure,
        &source,
        &TreeTransformOperation::permute([1], [0]),
        false,
    ));
    assert_eq!(
        source_layout_homspace_id_comparisons_rec.get(Event::SourceLayoutHomspaceIdComparison),
        0
    );

    let source_layout_homspace_id_comparisons_rec = probe::record();
    assert!(!source_is_borrowable_core_layout(
        &source,
        &source_structure,
        &source,
        &TreeTransformOperation::permute([0], [1]),
        true,
    ));
    assert_eq!(
        source_layout_homspace_id_comparisons_rec.get(Event::SourceLayoutHomspaceIdComparison),
        0
    );

    let source_layout_homspace_id_comparisons_rec = probe::record();
    assert!(source_is_borrowable_core_layout(
        &source,
        &source_structure,
        &source,
        &TreeTransformOperation::permute([0], [1]),
        false,
    ));
    assert_eq!(
        source_layout_homspace_id_comparisons_rec.get(Event::SourceLayoutHomspaceIdComparison),
        1
    );
}
