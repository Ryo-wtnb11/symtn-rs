//! Warm category adaptation reuses SUN's existing interned object handles.

#![cfg(feature = "cgc-gen")]

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

#[path = "../../tests/support/numerics.rs"]
mod numerics;
use num_complex::{Complex32, Complex64};

use symtn_sectors::{
    CheckedGenericFusion, CheckedGenericRigidSymbols, GenericRMatrix, SUNFusionRule,
};

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

#[test]
fn warm_sun_symbol_queries() {
    let rule = SUNFusionRule::new(3).unwrap();
    let octet = rule.encode_dynkin(&[1, 1]).unwrap();
    let f = || {
        rule.try_f_symbol_generic(octet, octet, octet, octet, octet, octet)
            .unwrap()
    };
    f();
    let (f, allocations) = counting_alloc::measure(f);
    assert_eq!(f.data().len(), 16);
    assert_eq!((allocations.calls, allocations.bytes), (1, 128));
    println!(
        "SU3 F: {} calls/{} bytes",
        allocations.calls, allocations.bytes
    );
    let queries: [(&str, &dyn Fn() -> GenericRMatrix<f64>); 3] = [
        ("R", &|| {
            rule.try_r_symbol_generic(octet, octet, octet).unwrap()
        }),
        ("A", &|| {
            rule.try_a_symbol_generic(octet, octet, octet).unwrap()
        }),
        ("B", &|| {
            rule.try_b_symbol_generic(octet, octet, octet).unwrap()
        }),
    ];
    for (name, query) in queries {
        let expected = query();
        let (actual, allocations) = counting_alloc::measure(query);
        assert_eq!(actual.data(), expected.data());
        assert_eq!(actual.shape(), (2, 2));
        let budget = if name == "R" { (22, 6168) } else { (6, 192) };
        assert_eq!(
            (allocations.calls, allocations.bytes),
            budget,
            "{name}: {allocations:?}"
        );
        println!(
            "SU3 {name}: {} calls/{} bytes",
            allocations.calls, allocations.bytes
        );
    }
    // Tensor-unit vertices distinguish which input A and B dualize.
    let three = rule.encode_dynkin(&[1, 0]).unwrap();
    let unit = rule.encode_dynkin(&[0, 0]).unwrap();
    for block in [
        rule.try_a_symbol_generic(three, unit, three).unwrap(),
        rule.try_b_symbol_generic(unit, three, three).unwrap(),
    ] {
        assert_eq!(block.shape(), (1, 1));
        numerics::assert_close("fundamental A/B", block.data()[0], 1.0, 3);
    }
}
