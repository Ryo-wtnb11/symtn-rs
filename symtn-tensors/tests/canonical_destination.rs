use std::sync::Arc;
use symtn_core::{
    FusionProductSpace, FusionTreeHomSpace, RuleIdentity, SectorId, SectorLeg, ZNFusionRule,
};
use symtn_tensors::{
    plan_compose, prepare_tensorcontract_fusion_plan_dyn, tensorcontract_fusion_block_specs,
    try_compile_storage_contract_core_route, BoundDynamicFusionMapSpace, ContractDestinationInit,
    FusionOperand, OperationError, OutputAxisOrder, TensorContractFusionExecutionContext,
    TensorContractSpec,
};

fn is_non_canonical_destination(error: &OperationError) -> bool {
    matches!(
        error,
        OperationError::InvalidArgument { message } if message.contains("canonical coupled-sector layout")
    )
}

/// A canonical Z2 `1 <- 1` space with two sectors of degeneracy 2, and its
/// lazy adjoint view, whose storage has no canonical coupled-sector regions
/// at the view's split.
fn storage_and_adjoint_view() -> (
    BoundDynamicFusionMapSpace<ZNFusionRule>,
    BoundDynamicFusionMapSpace<ZNFusionRule>,
) {
    let leg = || SectorLeg::new([(SectorId::new(0), 2), (SectorId::new(1), 2)], false);
    let storage = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::new(ZNFusionRule::new(2).unwrap()),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg()]),
            FusionProductSpace::new([leg()]),
        ),
    )
    .unwrap();
    let view = storage.adjoint_view().unwrap();
    (storage, view)
}

const LHS: [f64; 8] = [1., 2., 3., 4., 2., 0., 0., 3.];
const RHS: [f64; 8] = [5., 6., 7., 8., 4., 0., 0., 5.];

/// A composition writes coupled-sector matrices in place, so a destination
/// without its space's canonical layout is the caller's argument error,
/// reported before planning and before any write.
#[test]
fn composition_into_an_adjoint_view_destination_is_an_argument_error() {
    let (storage, dst) = storage_and_adjoint_view();
    let operand = || FusionOperand::adjoint(storage.space());

    let planned = plan_compose(&dst, operand(), operand()).unwrap_err();
    assert!(is_non_canonical_destination(&planned), "{planned:?}");

    let mut output = [11.; 8];
    let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
    let executed = context
        .tensorcompose_planned_into(
            &mut symtn_dense::CpuEngine::serial(),
            &dst,
            &mut output,
            (operand(), &LHS),
            (operand(), &RHS),
            1.,
            ContractDestinationInit::Axpby(0.),
        )
        .unwrap_err();
    assert!(is_non_canonical_destination(&executed), "{executed:?}");
    assert_eq!(output, [11.; 8]);
}

/// Every contraction entry, planning and compile-only ones included, rejects
/// the same destination before planning: with an identity output its core
/// would write the view's storage.
#[test]
fn contraction_into_an_adjoint_view_destination_is_an_argument_error() {
    let (storage, dst) = storage_and_adjoint_view();
    let operand = || FusionOperand::adjoint(storage.space());

    let mut output = [11.; 8];
    let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
    let executed = context
        .tensorcontract_planned_into(
            &mut symtn_dense::CpuEngine::serial(),
            &dst,
            &mut output,
            (operand(), &LHS),
            (operand(), &RHS),
            &[1],
            &[0],
            &[0, 1],
            1.,
            ContractDestinationInit::Axpby(0.),
        )
        .unwrap_err();
    assert!(is_non_canonical_destination(&executed), "{executed:?}");
    assert_eq!(output, [11.; 8]);

    let conjugated = TensorContractSpec::new_with_conjugation(
        &[1],
        &[0],
        OutputAxisOrder::from_axes(&[0, 1]),
        true,
        true,
    );
    let Err(core) = try_compile_storage_contract_core_route(&dst, operand(), operand(), conjugated)
    else {
        panic!("the core route must reject the destination");
    };
    assert!(is_non_canonical_destination(&core), "{core:?}");

    let plan =
        prepare_tensorcontract_fusion_plan_dyn(&dst, operand(), operand(), &[1], &[0], &[0, 1])
            .unwrap_err();
    assert!(is_non_canonical_destination(&plan), "{plan:?}");

    let specs = tensorcontract_fusion_block_specs(
        storage.provider(),
        dst.space(),
        storage.space(),
        storage.space(),
        TensorContractSpec::new(&[1], &[0], OutputAxisOrder::from_axes(&[0, 1])),
    )
    .unwrap_err();
    assert!(is_non_canonical_destination(&specs), "{specs:?}");
}
