//! Warm category adaptation reuses SUN's existing interned object handles.

#![cfg(feature = "cgc-gen")]

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

use symtn_sectors::{CheckedGenericFusion, CheckedGenericRigidSymbols, SUNFusionRule};

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

#[test]
fn warm_sun_scalar_queries_reuse_interned_handles() {
    for (n, labels, dimension, multiplicity) in [
        (2, &[2][..], 3.0, 1),
        (3, &[1, 1][..], 8.0, 2),
        (4, &[1, 0, 1][..], 15.0, 2),
    ] {
        let rule = SUNFusionRule::new(n).unwrap();
        let adjoint = rule.encode_dynkin(labels).unwrap();
        let query = || {
            (
                rule.try_dim_scalar(adjoint).unwrap(),
                rule.try_frobenius_schur_phase_scalar(adjoint).unwrap(),
                rule.try_dual(adjoint).unwrap(),
                rule.try_nsymbol(adjoint, adjoint, adjoint).unwrap(),
            )
        };
        assert_eq!(query(), (dimension, 1.0, adjoint, multiplicity));
        let (values, allocs) = counting_alloc::measure(query);
        assert_eq!(values, (dimension, 1.0, adjoint, multiplicity));
        assert_eq!((allocs.calls, allocs.bytes), (0, 0), "SU({n}): {allocs:?}");
    }
}
