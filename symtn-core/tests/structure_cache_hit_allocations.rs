use crate::su2::su2_rule;
use std::hint::black_box;
use std::sync::Arc;

use symtn_core::FusionTreeHomSpace;

use crate::counting_alloc;

#[test]
fn complete_structure_cache_hit_returns_canonical_arc_without_allocating() {
    let su2_provider_rule = su2_rule();
    let _serial = crate::counting_alloc::serial();
    // What: while a wrapper is live, a complete-structure lookup returns the
    // canonical Arc itself; the key is built on the stack and no wrapper,
    // region state, or key Arc is constructed. The shared serial guard prevents
    // another test from evicting the warmed bounded-cache entry.
    let spin = |twice: usize| symtn_core::SectorId::new(twice);
    let hom = FusionTreeHomSpace::from_sectors(
        [(spin(1), 2), (spin(3), 1)],
        [(spin(1), 3), (spin(2), 2)],
    );
    let first = hom
        .coupled_subblock_structure_from_leg_degeneracies(&su2_provider_rule)
        .unwrap();

    let (second, allocs) = counting_alloc::measure(|| {
        black_box(&hom)
            .coupled_subblock_structure_from_leg_degeneracies(&su2_provider_rule)
            .unwrap()
    });

    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(allocs.calls, 0);
}

#[test]
fn warm_layout_cache_hit_builds_its_lookup_key_without_allocating() {
    let su2_provider_rule = su2_rule();
    let _serial = crate::counting_alloc::serial();
    // What: a warm fusion-tree layout lookup constructs its cache key from the
    // HomSpace content in place, so neither the read-side lookup nor the commit
    // allocates. Uses a HomSpace of its own so the shared bounded layout cache
    // holds the warmed entry regardless of what other tests admit.
    let spin = |twice: usize| symtn_core::SectorId::new(twice);
    let hom = FusionTreeHomSpace::from_sectors(
        [(spin(2), 2), (spin(4), 3)],
        [(spin(2), 1), (spin(3), 2)],
    );
    let first = hom.fusion_tree_keys(&su2_provider_rule);

    let (second, allocs) =
        counting_alloc::measure(|| black_box(&hom).fusion_tree_keys(&su2_provider_rule));

    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(allocs.calls, 0);
}
