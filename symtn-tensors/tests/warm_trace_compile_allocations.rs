//! A warm non-Unique trace compile replays its groups' lowered terms from the
//! process-global caches: it misses no structure cache, and its allocations
//! do not depend on the degeneracies (the per-row split it replaces would
//! scale with neither but cost a cache miss per group).

use std::sync::Arc;

use symtn_core::{
    FusionProductSpace, FusionTreeHomSpace, MultiplicityFreeAdmissionMode, SU2FusionRule, SU2Irrep,
    SectorId, SectorLeg,
};
use symtn_tensors::{BoundDynamicFusionMapSpace, PivotalCoefficientAlgebra, TensorTraceAxisSpec};

use crate::counting_alloc;

const OUTPUT: [usize; 3] = [1, 2, 4];
const LHS: [usize; 1] = [0];
const RHS: [usize; 1] = [3];

fn axes() -> TensorTraceAxisSpec<'static> {
    TensorTraceAxisSpec::new(&OUTPUT, &LHS, &RHS)
}

/// `[v, v̄, v] ← [v, v̄]` over `sectors`, each of degeneracy `degeneracy`.
fn hom(sectors: &[SectorId], degeneracy: usize) -> FusionTreeHomSpace {
    let leg = |dual| SectorLeg::new(sectors.iter().map(|&sector| (sector, degeneracy)), dual);
    FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(false), leg(true), leg(false)]),
        FusionProductSpace::new([leg(false), leg(true)]),
    )
}

/// Structure-cache misses so far, over every kind.
fn structure_misses() -> u64 {
    symtn_core::StructureCacheKind::ALL
        .into_iter()
        .map(|kind| symtn_core::structure_cache_info(kind).misses())
        .sum()
}

/// Compiled-trace hits so far (the completed-transformer cache holds the
/// compiled traces).
fn compiled_trace_hits() -> u64 {
    symtn_core::structure_cache_info(symtn_core::StructureCacheKind::CompletedTreeTransformer)
        .hits()
}

/// Allocation calls, structure-cache misses and compiled-trace hits of one
/// warm compile, after two warm-ups.
fn warm_compile<M, R>(
    dst: &BoundDynamicFusionMapSpace<R>,
    src: &BoundDynamicFusionMapSpace<R>,
) -> (u64, u64, u64)
where
    M: PivotalCoefficientAlgebra<R>,
    M::Error: std::fmt::Debug,
{
    for _ in 0..2 {
        drop(M::trace_terms(dst, src, axes()).unwrap());
    }
    let (misses, hits) = (structure_misses(), compiled_trace_hits());
    counting_alloc::start();
    let structure = M::trace_terms(dst, src, axes()).unwrap();
    let calls = counting_alloc::stop().calls;
    let (misses, hits) = (structure_misses() - misses, compiled_trace_hits() - hits);
    drop(structure);
    (calls, misses, hits)
}

/// The contract over a degeneracy-one and a degeneracy-three twin: no
/// cache miss, one compiled-trace hit, equal allocation calls.
fn assert_warm_replay(label: &str, [small, large]: [(u64, u64, u64); 2]) {
    assert_eq!(
        (small.1, large.1),
        (0, 0),
        "{label}: a warm compile missed a cache"
    );
    assert_eq!(
        (small.2, large.2),
        (1, 1),
        "{label}: a warm compile rebuilt its trace instead of hitting it"
    );
    counting_alloc::assert_same_calls(label, small.0, large.0);
}

/// Purpose: a warm multiplicity-free SU(2) trace compile misses no cache and
/// costs the same at every degeneracy.
#[test]
fn warm_multiplicity_free_trace_compile_replays_from_cache() {
    let _serial = crate::counting_alloc::serial();
    let sectors = [0, 1, 2, 3].map(|spin| SU2Irrep::from_twice_spin(spin).sector_id());
    let sizes = [1, 3].map(|degeneracy| {
        let src = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
            Arc::new(SU2FusionRule),
            hom(&sectors, degeneracy),
        )
        .unwrap();
        let dst_hom = symtn_tensors::tensortrace_fusion_dyn_preflight_checked(&src, axes(), 2)
            .unwrap()
            .into_selected_homspace();
        let dst = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
            Arc::new(SU2FusionRule),
            dst_hom,
        )
        .unwrap();
        warm_compile::<MultiplicityFreeAdmissionMode, _>(&dst, &src)
    });
    assert_warm_replay("MF SU(2)", sizes);
}

/// Purpose: the checked-Generic SU(3) twin of the contract above.
#[cfg(feature = "cgc-gen")]
#[test]
fn warm_checked_generic_trace_compile_replays_from_cache() {
    let _serial = crate::counting_alloc::serial();
    use symtn_core::{CheckedGenericAdmissionMode, LieGroupFusionRule};
    let rule =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let sectors = [[0, 0], [1, 0], [1, 1]].map(|dynkin| rule.encode_irrep(&dynkin).unwrap());
    let sizes = [1, 3].map(|degeneracy| {
        let src = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
            Arc::clone(&rule),
            hom(&sectors, degeneracy),
        )
        .unwrap();
        let dst_hom =
            symtn_tensors::tensortrace_fusion_dyn_preflight_generic_checked(&src, axes(), 2)
                .unwrap()
                .into_selected_homspace();
        let dst = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
            Arc::clone(&rule),
            dst_hom,
        )
        .unwrap();
        warm_compile::<CheckedGenericAdmissionMode, _>(&dst, &src)
    });
    assert_warm_replay("checked SU(3)", sizes);
}
