use std::sync::Arc;

use symtn_core::{
    structure_cache_info, CoreError, FermionParityFusionRule, FusionProductSpace,
    FusionTreeHomSpace, SectorLeg, StructureCacheKind, U1FusionRule, U1Irrep, Z2Irrep,
};
use symtn_tensors::{BoundDynamicFusionMapSpace, OperationError};

#[test]
fn lowered_complete_cache_preflight_preserves_statistics_and_hits() {
    let _serial = crate::counting_alloc::serial();
    // What: a lowered leg-extent overflow is rejected without changing any
    // complete structure cache statistic or resource.
    symtn_core::clear_structure_caches();
    symtn_core::clear_structure_caches();
    let vacuum = U1Irrep::new(0).sector_id();
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(vacuum, usize::MAX)], false)]),
        FusionProductSpace::new([SectorLeg::new([(vacuum, 2)], false)]),
    );
    let before = structure_cache_info(StructureCacheKind::DegeneracyStructure);

    let error = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(
        Arc::new(U1FusionRule),
        homspace,
    )
    .unwrap_err();

    assert_eq!(error, OperationError::Core(CoreError::ElementCountOverflow));
    assert_eq!(
        structure_cache_info(StructureCacheKind::DegeneracyStructure),
        before
    );

    let provider = Arc::new(FermionParityFusionRule);
    let odd = Z2Irrep::ODD.sector_id();
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(odd, 1)], false)]),
        FusionProductSpace::new([SectorLeg::new([(odd, 1)], false)]),
    );
    let first = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(
        Arc::clone(&provider),
        homspace.clone(),
    )
    .unwrap();
    let after_first = structure_cache_info(StructureCacheKind::DegeneracyStructure);
    assert_eq!(after_first.admissions(), 1);

    let second = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free_checked(
        provider, homspace,
    )
    .unwrap();
    let after_second = structure_cache_info(StructureCacheKind::DegeneracyStructure);

    assert_eq!(
        first.space().structure().content_id(),
        second.space().structure().content_id()
    );
    assert_eq!(after_second.hits(), after_first.hits() + 1);
    assert_eq!(after_second.admissions(), after_first.admissions());
}
