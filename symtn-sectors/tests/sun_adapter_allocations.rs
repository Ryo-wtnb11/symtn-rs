//! Warm category adaptation reuses SUN's existing interned object handles.

#![cfg(feature = "cgc-gen")]

#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

#[path = "../../tests/support/numerics.rs"]
mod numerics;
use num_complex::{Complex32, Complex64};

use fusion_symbols::category::{FusionCategory, SymbolBlock};
use symtn_sectors::{
    CategoryProvider, CheckedGenericFusion, CheckedGenericRigidSymbols, SUNFusionRule,
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
    let object = rule.object(octet).unwrap();
    let category = CategoryProvider::category(&rule);
    let direct_f = || {
        category
            .f_symbol(&object, &object, &object, &object, &object, &object)
            .unwrap()
    };
    let adapted_f = || {
        rule.try_f_symbol_generic(octet, octet, octet, octet, octet, octet)
            .unwrap()
    };
    let direct_r = || category.r_symbol(&object, &object, &object).unwrap();
    let adapted_r = || rule.try_r_symbol_generic(octet, octet, octet).unwrap();
    let direct_a = || category.a_symbol(&object, &object, &object).unwrap();
    let adapted_a = || rule.try_a_symbol_generic(octet, octet, octet).unwrap();
    let direct_b = || category.b_symbol(&object, &object, &object).unwrap();
    let adapted_b = || rule.try_b_symbol_generic(octet, octet, octet).unwrap();
    direct_f();
    adapted_f();
    direct_r();
    adapted_r();
    direct_a();
    adapted_a();
    direct_b();
    adapted_b();

    let (fs_f, fs_f_cost) = counting_alloc::measure(direct_f);
    let (f, f_cost) = counting_alloc::measure(adapted_f);
    let (fs_r, fs_r_cost) = counting_alloc::measure(direct_r);
    let (r, r_cost) = counting_alloc::measure(adapted_r);
    let (fs_a, fs_a_cost) = counting_alloc::measure(direct_a);
    let (a, a_cost) = counting_alloc::measure(adapted_a);
    let (fs_b, fs_b_cost) = counting_alloc::measure(direct_b);
    let (b, b_cost) = counting_alloc::measure(adapted_b);
    assert_eq!(f.data(), fs_f.as_slice());
    assert_eq!(r.data(), fs_r.as_slice());
    assert_eq!(a.data(), fs_a.data());
    assert_eq!(b.data(), fs_b.data());
    assert_eq!(f.data().len(), 16);
    for block in [&r, &a, &b] {
        assert_eq!(block.shape(), (2, 2));
    }
    for (name, direct, adapted, len) in [
        ("F", fs_f_cost, f_cost, f.data().len()),
        ("R", fs_r_cost, r_cost, r.data().len()),
        ("A", fs_a_cost, a_cost, a.data().len()),
        ("B", fs_b_cost, b_cost, b.data().len()),
    ] {
        println!(
            "SU3 {name}: fs {} calls/{} bytes; adapter {} calls/{} bytes; len {len}",
            direct.calls, direct.bytes, adapted.calls, adapted.bytes
        );
        assert_eq!(
            (adapted.calls, adapted.bytes),
            (
                direct.calls + 1,
                direct.bytes + (len * size_of::<f64>()) as u64
            ),
            "{name}: fs {direct:?}; adapter {adapted:?}"
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
