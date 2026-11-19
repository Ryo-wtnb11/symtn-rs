use std::sync::Arc;
use symtn_core::{
    FusionProductSpace, FusionTreeHomSpace, RuleIdentity, SectorId, SectorLeg, ZNFusionRule,
};
use symtn_tensors::{
    plan_compose, BoundDynamicFusionMapSpace, ContractDestinationInit, FusionOperand,
    OperationError, TensorContractFusionExecutionContext,
};

fn is_non_canonical_destination(error: &OperationError) -> bool {
    matches!(
        error,
        OperationError::InvalidArgument { message } if message.contains("canonical coupled-sector layout")
    )
}

/// A composition writes coupled-sector matrices in place, so a destination
/// without its space's canonical layout (here a lazy adjoint view, whose
/// two-sector Z2 storage has no canonical regions at the view's split) is
/// the caller's argument error, reported before planning and before any
/// write.
#[test]
fn composition_into_an_adjoint_view_destination_is_an_argument_error() {
    let leg = || SectorLeg::new([(SectorId::new(0), 2), (SectorId::new(1), 2)], false);
    let storage = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::new(ZNFusionRule::new(2).unwrap()),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg()]),
            FusionProductSpace::new([leg()]),
        ),
    )
    .unwrap();
    let dst = storage.adjoint_view().unwrap();
    let operand = || FusionOperand::adjoint(storage.space());

    let planned = plan_compose(&dst, operand(), operand()).unwrap_err();
    assert!(is_non_canonical_destination(&planned), "{planned:?}");

    let lhs = [1., 2., 3., 4., 2., 0., 0., 3.];
    let rhs = [5., 6., 7., 8., 4., 0., 0., 5.];
    let mut output = [11.; 8];
    let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
    let executed = context
        .tensorcompose_planned_into(
            &mut symtn_dense::CpuEngine::serial(),
            &dst,
            &mut output,
            (operand(), &lhs),
            (operand(), &rhs),
            1.,
            ContractDestinationInit::Axpby(0.),
        )
        .unwrap_err();
    assert!(is_non_canonical_destination(&executed), "{executed:?}");
    assert_eq!(output, [11.; 8]);
}

/// The contraction planner rejects the same destination before planning:
/// with an identity output its core would also write the view's storage.
#[test]
fn contraction_into_an_adjoint_view_destination_is_an_argument_error() {
    let leg = || SectorLeg::new([(SectorId::new(0), 2), (SectorId::new(1), 2)], false);
    let storage = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::new(ZNFusionRule::new(2).unwrap()),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg()]),
            FusionProductSpace::new([leg()]),
        ),
    )
    .unwrap();
    let dst = storage.adjoint_view().unwrap();
    let operand = || FusionOperand::adjoint(storage.space());
    let lhs = [1., 2., 3., 4., 2., 0., 0., 3.];
    let rhs = [5., 6., 7., 8., 4., 0., 0., 5.];
    let mut output = [11.; 8];
    let mut context = TensorContractFusionExecutionContext::<f64, RuleIdentity>::default();
    let error = context
        .tensorcontract_planned_into(
            &mut symtn_dense::CpuEngine::serial(),
            &dst,
            &mut output,
            (operand(), &lhs),
            (operand(), &rhs),
            &[1],
            &[0],
            &[0, 1],
            1.,
            ContractDestinationInit::Axpby(0.),
        )
        .unwrap_err();
    assert!(is_non_canonical_destination(&error), "{error:?}");
    assert_eq!(output, [11.; 8]);
}
