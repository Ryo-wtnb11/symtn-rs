//! Categorical adjoint views of fusion-map spaces and the TensorKit
//! `adjointtensorindex` axis map.

use super::*;

#[test]
fn categorical_adjoint_view_swaps_homspace_and_block_layout_without_dualizing() {
    // What: the view swaps categorical sides and block axes while keeping the
    // parent's offsets and strides; applying it twice is exact.
    let sector = SectorId::new(1);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(sector, 2)], false)]),
        FusionProductSpace::new([SectorLeg::new([(sector, 3)], false)]),
    );
    let space = DynamicFusionMapSpace::from_final_homspace(&Z2FusionRule, hom).unwrap();

    let adjoint = space
        .validate_rule(&Z2FusionRule)
        .and_then(|()| space.adjoint_view())
        .unwrap();

    assert_eq!(
        adjoint.homspace().codomain().legs(),
        space.homspace().domain().legs()
    );
    assert_eq!(
        adjoint.homspace().domain().legs(),
        space.homspace().codomain().legs()
    );
    assert_eq!(adjoint.homspace().codomain().legs()[0].degeneracies(), &[3]);
    assert_eq!(adjoint.homspace().domain().legs()[0].degeneracies(), &[2]);
    let block = adjoint.structure().block(0).unwrap();
    assert_eq!(block.shape(), &[3, 2]);
    assert_eq!(block.strides(), &[2, 1]);
    assert_eq!(block.offset(), 0);
    let key = expect_tree_key(block.key());
    assert_eq!(key.codomain_tree().uncoupled(), &[sector]);
    assert_eq!(key.domain_tree().uncoupled(), &[sector]);
    assert_eq!(adjoint.adjoint_view().unwrap(), space);
}

#[test]
fn categorical_adjoint_view_does_not_dualize_nonselfdual_u1_stored_legs() {
    let q1 = U1Irrep::new(1).sector_id();
    let q2 = U1Irrep::new(2).sector_id();
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(q1, 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(q2, 1)], false)]),
    );
    let space = DynamicFusionMapSpace::from_final_homspace(&U1FusionRule, hom).unwrap();

    let adjoint = space
        .validate_rule(&U1FusionRule)
        .and_then(|()| space.adjoint_view())
        .unwrap();

    assert_eq!(
        adjoint.homspace().codomain().legs(),
        &[SectorLeg::new([(q2, 1)], false)]
    );
    assert_eq!(
        adjoint.homspace().domain().legs(),
        &[SectorLeg::new([(q1, 1)], false)]
    );
    assert_ne!(
        adjoint.homspace().codomain().legs(),
        &[SectorLeg::new([(U1Irrep::new(-2).sector_id(), 1)], true)]
    );
}

#[test]
fn categorical_adjoint_axis_map_matches_tensorkit_adjointtensorindex() {
    assert_eq!(crate::lowering::adjoint_tensor_axis(2, 1, 0).unwrap(), 1);
    assert_eq!(crate::lowering::adjoint_tensor_axis(2, 1, 1).unwrap(), 2);
    assert_eq!(crate::lowering::adjoint_tensor_axis(2, 1, 2).unwrap(), 0);

    let err = crate::lowering::adjoint_tensor_axis(2, 1, 3).unwrap_err();
    assert!(matches!(err, OperationError::InvalidAxisSet { .. }));
}
