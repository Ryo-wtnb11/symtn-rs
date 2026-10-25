//! #2149: a warm non-Unique trace compile replays its groups' lowered terms
//! from cache 4. Its allocations are bounded by the destination blocks and
//! terms it returns, not by the permutation rows it would otherwise split.

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

/// `[v, v̄, v] ← [v, v̄]` over `sectors`, each of degeneracy one.
fn hom(sectors: &[SectorId]) -> FusionTreeHomSpace {
    let leg = |dual| SectorLeg::new(sectors.iter().map(|&sector| (sector, 1)), dual);
    FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(false), leg(true), leg(false)]),
        FusionProductSpace::new([leg(false), leg(true)]),
    )
}

/// Warm calls of one compile against its returned size, after two warm-ups.
fn warm_compile<M, R>(
    dst: &BoundDynamicFusionMapSpace<R>,
    src: &BoundDynamicFusionMapSpace<R>,
) -> (u64, usize, usize)
where
    M: PivotalCoefficientAlgebra<R>,
    M::Error: std::fmt::Debug,
{
    for _ in 0..2 {
        drop(M::trace_terms(dst, src, axes()).unwrap());
    }
    counting_alloc::start();
    let structure = M::trace_terms(dst, src, axes()).unwrap();
    let calls = counting_alloc::stop().calls;
    (
        calls,
        dst.space().structure().block_count(),
        structure.terms().len(),
    )
}

/// The structural gate: a fixed overhead plus at most one allocation per
/// destination block and per term. A per-call split of every permutation row
/// (tens of allocations per row) is far above this bound.
fn assert_bounded(label: &str, sizes: &[(u64, usize, usize)]) {
    eprintln!("{label}: (warm calls, destination blocks, terms) = {sizes:?}");
    for &(calls, dst_blocks, terms) in sizes {
        assert!(
            calls <= 64 + (dst_blocks + terms) as u64,
            "{label}: {calls} warm calls for {dst_blocks} destination blocks, {terms} terms"
        );
    }
    let (small, large) = (sizes[0], sizes[sizes.len() - 1]);
    assert!(
        large.2 > 2 * small.2,
        "{label}: sizes must differ: {sizes:?}"
    );
}

#[test]
fn warm_multiplicity_free_trace_compile_is_bounded_by_its_terms() {
    let _serial = crate::counting_alloc::serial();
    let sizes = [vec![0, 1], vec![0, 1, 2], vec![0, 1, 2, 3]].map(|spins| {
        let sectors = spins
            .iter()
            .map(|&spin| SU2Irrep::from_twice_spin(spin).sector_id())
            .collect::<Vec<_>>();
        let src = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
            Arc::new(SU2FusionRule),
            hom(&sectors),
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
    assert_bounded("MF SU(2)", &sizes);
}

#[cfg(feature = "cgc-gen")]
#[test]
fn warm_checked_generic_trace_compile_is_bounded_by_its_terms() {
    let _serial = crate::counting_alloc::serial();
    use symtn_core::{CheckedGenericAdmissionMode, LieGroupFusionRule};
    let rule =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let sizes = [&[[0, 0], [1, 0]][..], &[[0, 0], [1, 0], [1, 1]][..]].map(|dynkins| {
        let sectors = dynkins
            .iter()
            .map(|dynkin| rule.encode_dynkin(dynkin).unwrap())
            .collect::<Vec<_>>();
        let src = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
            Arc::clone(&rule),
            hom(&sectors),
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
    assert_bounded("checked SU(3)", &sizes);
}
