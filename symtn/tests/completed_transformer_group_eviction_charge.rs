//! A completed transformer (cache 3) bound from cached coefficient groups
//! (cache 4) must not keep those groups' allocations alive once cache 4
//! evicts them unless it is charged for them: its charge bounds what
//! evicting it frees in either eviction order.

use std::sync::Arc;

use symtn::cache::StructureCacheKind;
use symtn::sector::{SU2FusionRule, SU2Irrep};
use symtn::typed::{GradedSpace, Runtime, TensorMap};

use crate::counting_alloc;

fn info(kind: StructureCacheKind) -> symtn::cache::StructureCacheInfo {
    symtn::cache::stats()
        .into_iter()
        .find(|info| info.kind() == kind)
        .unwrap()
}

fn su2(degeneracies: [usize; 4]) -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        (0..4).map(|twice| (SU2Irrep::from_twice_spin(twice), degeneracies[twice])),
    )
    .unwrap()
}

#[test]
fn completed_transformer_charge_covers_what_it_frees_after_group_eviction() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let (groups, completed) = (
        StructureCacheKind::TreeTransformCoefficients,
        StructureCacheKind::CompletedTreeTransformer,
    );
    let budgets = [groups, completed].map(|kind| (kind, info(kind).byte_budget()));
    symtn::cache::clear();

    // An SU(2) permute with recoupling (Multi specs), so cache 3's core is
    // bound from cache-4 groups.
    let runtime = Runtime::builder().build().unwrap();
    let phys =
        GradedSpace::try_new(Arc::new(SU2FusionRule), [(SU2Irrep::from_twice_spin(1), 1)]).unwrap();
    let (l, u, r, d) = (
        su2([4, 5, 2, 1]),
        su2([3, 4, 3, 1]),
        su2([5, 4, 2, 2]),
        su2([3, 5, 2, 1]),
    );
    let a = TensorMap::<_, f64>::rand_with_seed(engine, &runtime, [&l, &u, &phys], [&r, &d], 3)
        .unwrap();
    drop(a.permute(engine, &[0, 2, 3], &[1, 4]).unwrap());
    drop(a);
    assert!(info(groups).entries() > 0);
    let resident = info(completed);
    assert!(resident.entries() > 0);

    // What: once cache 4 has evicted the groups, cache 3's charge still
    // covers all that evicting cache 3 frees.
    symtn::cache::configure_budgets([(groups, 0)]);
    let ((), freed) = counting_alloc::measure(|| symtn::cache::configure_budgets([(completed, 0)]));
    assert_eq!(info(completed).entries(), 0);
    let freed = u64::try_from(-freed.live_bytes).unwrap();
    assert!(
        resident.charged_bytes() >= freed,
        "cache 3 charged {} B but freed {freed} B",
        resident.charged_bytes()
    );

    symtn::cache::configure_budgets(budgets);
}
