//! Categorical adjoint views of fusion-map spaces and the TensorKit
//! `adjointtensorindex` axis map.

use super::*;

#[test]
fn categorical_adjoint_view_swaps_homspace_and_block_layout_without_dualizing() {
    let cod_sector = SectorId::new(1);
    let dom_sector = SectorId::new(1);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(cod_sector, 2)], false)]),
        FusionProductSpace::new([SectorLeg::new([(dom_sector, 3)], false)]),
    );
    let structure = BlockStructure::from_blocks_with_rank(
        2,
        vec![BlockSpec::with_key(
            BlockKey::from(FusionTreePairKey::pair(
                FusionTreeKey::try_new_for_rule(
                    &Z2FusionRule,
                    [cod_sector],
                    cod_sector,
                    [false],
                    [],
                    [],
                )
                .unwrap(),
                FusionTreeKey::try_new_for_rule(
                    &Z2FusionRule,
                    [dom_sector],
                    dom_sector,
                    [false],
                    [],
                    [],
                )
                .unwrap(),
            )),
            vec![2, 3],
            vec![1, 2],
            5,
        )
        .unwrap()],
    )
    .unwrap();
    let space = DynamicFusionMapSpace::try_from_layout(hom, structure)
        .unwrap()
        .try_bind_rule(&Z2FusionRule)
        .unwrap();

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
    assert_eq!(block.offset(), 5);
    let key = expect_tree_key(block.key());
    assert_eq!(key.codomain_tree().uncoupled(), &[dom_sector]);
    assert_eq!(key.domain_tree().uncoupled(), &[cod_sector]);
}

#[test]
fn fusion_space_adjoint_view_preserves_the_strided_block_footprint() {
    // What: adjoint swaps categorical sides and block axes while retaining
    // the already-admitted physical footprint; applying it twice is exact.
    let homspace = FusionTreeHomSpace::from_sector_ids([(0, 2)], [(0, 3)]);
    let structure = BlockStructure::from_blocks(vec![BlockSpec::with_key(
        BlockKey::ordinal(0),
        vec![2, 3],
        vec![1, 4],
        2,
    )
    .unwrap()])
    .unwrap();
    let source = DynamicFusionMapSpace::try_from_layout(homspace, structure).unwrap();

    let adjoint = source.adjoint_view().unwrap();
    assert_eq!(adjoint.homspace().codomain(), source.homspace().domain());
    assert_eq!(adjoint.homspace().domain(), source.homspace().codomain());
    let block = adjoint.structure().block(0).unwrap();
    assert_eq!(block.shape(), &[3, 2]);
    assert_eq!(block.strides(), &[4, 1]);
    assert_eq!(block.offset(), 2);
    assert_eq!(adjoint.adjoint_view().unwrap(), source);
}

#[test]
fn categorical_adjoint_view_does_not_dualize_nonselfdual_u1_stored_legs() {
    let q1 = U1Irrep::new(1).sector_id();
    let q2 = U1Irrep::new(2).sector_id();
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(q1, 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(q2, 1)], false)]),
    );
    let space = DynamicFusionMapSpace::try_from_layout(hom, BlockStructure::empty(2))
        .unwrap()
        .try_bind_rule(&U1FusionRule)
        .unwrap();

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
