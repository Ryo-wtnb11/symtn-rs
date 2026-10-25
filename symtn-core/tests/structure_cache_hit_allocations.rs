use std::hint::black_box;
use std::sync::Arc;

use symtn_core::{FusionTreeHomSpace, SU2FusionRule, SU2Irrep};

use crate::counting_alloc;

#[test]
fn complete_structure_cache_hit_returns_canonical_arc_without_allocating() {
    let _serial = crate::counting_alloc::serial();
    // What: while a wrapper is live, a complete-structure lookup returns the
    // canonical Arc itself; the key is built on the stack and no wrapper,
    // region state, or key Arc is constructed. The shared serial guard prevents
    // another test from evicting the warmed bounded-cache entry.
    let spin = |twice: usize| SU2Irrep::from_twice_spin(twice).sector_id();
    let hom = FusionTreeHomSpace::from_sectors(
        [(spin(1), 2), (spin(3), 1)],
        [(spin(1), 3), (spin(2), 2)],
    );
    let first = hom
        .coupled_subblock_structure_from_leg_degeneracies(&SU2FusionRule)
        .unwrap();

    let (second, allocs) = counting_alloc::measure(|| {
        black_box(&hom)
            .coupled_subblock_structure_from_leg_degeneracies(&SU2FusionRule)
            .unwrap()
    });

    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(allocs.calls, 0);
}

#[test]
fn warm_layout_cache_hit_builds_its_lookup_key_without_allocating() {
    let _serial = crate::counting_alloc::serial();
    // What: a warm fusion-tree layout lookup constructs its cache key from the
    // HomSpace content in place, so neither the read-side lookup nor the commit
    // allocates. Uses a HomSpace of its own so the shared bounded layout cache
    // holds the warmed entry regardless of what other tests admit.
    let spin = |twice: usize| SU2Irrep::from_twice_spin(twice).sector_id();
    let hom = FusionTreeHomSpace::from_sectors(
        [(spin(2), 2), (spin(4), 3)],
        [(spin(2), 1), (spin(3), 2)],
    );
    let first = hom.fusion_tree_keys(&SU2FusionRule);

    let (second, allocs) =
        counting_alloc::measure(|| black_box(&hom).fusion_tree_keys(&SU2FusionRule));

    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(allocs.calls, 0);
}
