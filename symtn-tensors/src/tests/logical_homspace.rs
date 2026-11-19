//! The logical HomSpace of a lazy-adjoint operand and the TensorKit
//! `adjointtensorindex` axis map.

use super::*;
use crate::FusionOperand;

#[test]
fn adjoint_operand_logical_homspace_swaps_sides_without_dualizing() {
    // What: an adjoint operand's logical HomSpace is TensorKit's
    // `adjoint(space)`: codomain and domain swap, leg duality is kept, and the
    // value is memoized on the storage space.
    let u1 = |charge| SectorId::from(U1Irrep::new(charge).unwrap());
    let a = || SectorLeg::new([(u1(1), 2), (u1(-1), 1)], false);
    let b = || SectorLeg::new([(u1(2), 1), (u1(0), 3)], true);
    let c = || SectorLeg::new([(u1(3), 1), (u1(1), 2)], false);
    let parent = DynamicFusionMapSpace::from_final_homspace(
        &U1FusionRule,
        FusionTreeHomSpace::new(
            FusionProductSpace::new([a(), b()]),
            FusionProductSpace::new([c()]),
        ),
    )
    .unwrap();
    let expected = FusionTreeHomSpace::new(
        FusionProductSpace::new([c()]),
        FusionProductSpace::new([a(), b()]),
    );

    let logical = FusionOperand::adjoint(&parent).logical_homspace();

    assert_eq!(logical, &expected);
    assert!(std::ptr::eq(
        logical,
        FusionOperand::adjoint(&parent).logical_homspace()
    ));
    assert!(std::ptr::eq(
        FusionOperand::direct(&parent).logical_homspace(),
        parent.homspace()
    ));
}

#[test]
fn categorical_adjoint_axis_map_matches_tensorkit_adjointtensorindex() {
    assert_eq!(crate::lowering::adjoint_tensor_axis(2, 1, 0).unwrap(), 1);
    assert_eq!(crate::lowering::adjoint_tensor_axis(2, 1, 1).unwrap(), 2);
    assert_eq!(crate::lowering::adjoint_tensor_axis(2, 1, 2).unwrap(), 0);

    let err = crate::lowering::adjoint_tensor_axis(2, 1, 3).unwrap_err();
    assert!(matches!(err, OperationError::InvalidAxisSet { .. }));
}
