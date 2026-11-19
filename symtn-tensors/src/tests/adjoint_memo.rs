use super::*;
use std::hash::BuildHasher;
use std::sync::Arc;
use symtn_core::{CheckedFusionAlgebra, SectorId, U1FusionRule, U1Irrep};

use crate::{BoundDynamicFusionMapSpace, FusionOperand};

fn u1_legs() -> Vec<(SectorId, usize)> {
    [(-1, 2), (0, 3), (1, 4)]
        .into_iter()
        .map(|(charge, deg)| (SectorId::from(U1Irrep::new(charge).unwrap()), deg))
        .collect()
}

fn bound<R>(rule: &Arc<R>, legs: &[(SectorId, usize)]) -> BoundDynamicFusionMapSpace<R>
where
    R: MultiplicityFreeRigidSymbols + CheckedFusionAlgebra,
{
    let leg = || SectorLeg::new(legs.iter().copied(), false);
    BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(
        Arc::clone(rule),
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg()]),
            FusionProductSpace::new([leg()]),
        ),
    )
    .unwrap()
}

#[test]
fn adjoint_memo_leaves_space_equality_and_layout_hash_unchanged() {
    let rule = Arc::new(U1FusionRule);
    let filled = bound(&rule, &u1_legs());
    // A separate construction can cross an intern reset and mint a different layout id.
    let empty = filled.rebind_validated(&filled.validated_layout()).unwrap();
    let hash = |space: &BoundDynamicFusionMapSpace<U1FusionRule>| {
        rustc_hash::FxBuildHasher.hash_one(space.validated_layout())
    };
    let empty_hash = hash(&empty);
    let adjoint = FusionOperand::adjoint(filled.space()).logical_homspace();

    assert_eq!(filled.space(), empty.space());
    assert_eq!(filled.validated_layout(), empty.validated_layout());
    assert_eq!(hash(&filled), empty_hash);
    // A clone of a filled space shares the memo and still compares equal.
    let cloned = filled.space().clone();
    assert!(std::ptr::eq(
        FusionOperand::adjoint(&cloned).logical_homspace(),
        adjoint
    ));
    assert_eq!(&cloned, empty.space());
}
