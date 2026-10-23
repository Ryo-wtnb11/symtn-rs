//! Eager warm-query allocation evidence; run alone with `--nocapture`.
#[path = "../../tests/support/counting_alloc.rs"]
mod counting_alloc;

#[cfg(feature = "cgc-gen")]
#[path = "../../tests/support/numerics.rs"]
mod numerics;
#[cfg(feature = "cgc-gen")]
use num_complex::{Complex32, Complex64};

use symtn_sectors::{
    CheckedFusionAlgebra, FermionParityFusionRule, MultiplicityFreeFusionSymbols,
    MultiplicityFreeRigidSymbols, ProductFusionRule, ProductSector, SectorCodec, U1FusionRule,
    U1Irrep, Z2Irrep,
};

#[global_allocator]
static ALLOCATOR: counting_alloc::CountingAllocator = counting_alloc::CountingAllocator;

#[test]
fn warm_product_queries() {
    let rule = ProductFusionRule::<_, _>::try_new(U1FusionRule, FermionParityFusionRule).unwrap();
    let odd = rule
        .encode_sector(&ProductSector::new(U1Irrep::new(0), Z2Irrep::ODD))
        .unwrap();
    let even = rule
        .encode_sector(&ProductSector::new(U1Irrep::new(0), Z2Irrep::EVEN))
        .unwrap();
    let queries: [(&str, &dyn Fn() -> f64); 4] = [
        ("N", &|| rule.try_nsymbol(odd, odd, even).unwrap() as f64),
        ("F", &|| {
            rule.f_symbol_scalar(odd, odd, odd, odd, even, even)
        }),
        ("R", &|| rule.r_symbol_scalar(odd, odd, even)),
        ("dim", &|| rule.dim_scalar(odd)),
    ];
    for (name, query) in queries {
        let expected = query();
        let (actual, allocations) = counting_alloc::measure(query);
        assert_eq!(actual, expected);
        println!(
            "U1xfZ2 {name}: {} calls, {} bytes",
            allocations.calls, allocations.bytes
        );
        assert_eq!(
            (allocations.calls, allocations.bytes),
            (0, 0),
            "{name}: {allocations:?}"
        );
    }
}

#[cfg(feature = "cgc-gen")]
#[test]
fn warm_generic_product_queries() {
    use fusion_symbols::category::{FusionCategory, SymbolBlock};
    use symtn_sectors::{
        CategoryProvider, CheckedGenericRigidSymbols, SUNFusionRule, TypedSectorAdmission,
    };

    let rule = ProductFusionRule::<_, _>::try_new(
        SUNFusionRule::new(3).unwrap(),
        SUNFusionRule::new(3).unwrap(),
    )
    .unwrap();
    let octet = vec![1, 1];
    let id = rule
        .try_encode_label(&ProductSector::new(octet.clone(), octet))
        .unwrap();
    let object = rule.object(id).unwrap();
    let category = CategoryProvider::category(&rule);
    let direct_f = || {
        category
            .f_symbol(&object, &object, &object, &object, &object, &object)
            .unwrap()
    };
    let adapted_f = || rule.try_f_symbol_generic(id, id, id, id, id, id).unwrap();
    let direct_r = || category.r_symbol(&object, &object, &object).unwrap();
    let adapted_r = || rule.try_r_symbol_generic(id, id, id).unwrap();
    let direct_a = || category.a_symbol(&object, &object, &object).unwrap();
    let adapted_a = || rule.try_a_symbol_generic(id, id, id).unwrap();
    let direct_b = || category.b_symbol(&object, &object, &object).unwrap();
    let adapted_b = || rule.try_b_symbol_generic(id, id, id).unwrap();
    direct_f();
    adapted_f();
    direct_r();
    adapted_r();
    direct_a();
    adapted_a();
    direct_b();
    adapted_b();

    let (fs_f, fs_f_cost) = counting_alloc::measure(direct_f);
    let (f, f_cost) = counting_alloc::measure_matching(2048..=2048, adapted_f);
    let (fs_r, fs_r_cost) = counting_alloc::measure(direct_r);
    let (r, r_cost) = counting_alloc::measure_matching(128..=128, adapted_r);
    let (fs_a, fs_a_cost) = counting_alloc::measure(direct_a);
    let (a, a_cost) = counting_alloc::measure_matching(128..=128, adapted_a);
    let (fs_b, fs_b_cost) = counting_alloc::measure(direct_b);
    let (b, b_cost) = counting_alloc::measure_matching(128..=128, adapted_b);
    assert_eq!(f.data(), fs_f.as_slice());
    assert_eq!(r.data(), fs_r.as_slice());
    assert_eq!(a.data(), fs_a.data());
    assert_eq!(b.data(), fs_b.data());
    assert_eq!(f.data().len(), 256);
    assert_eq!(r.data().len(), 16);
    assert_eq!(a.data().len(), 16);
    assert_eq!(b.data().len(), 16);
    // The adapter's owned block copies are separate from fs product storage.
    assert!(f_cost.matched_calls >= 1);
    assert!(r_cost.matched_calls >= 1);
    assert!(a_cost.matched_calls >= 1);
    assert!(b_cost.matched_calls >= 1);
    for (name, direct, adapted, copied) in [
        ("F", fs_f_cost, f_cost, 2048),
        ("R", fs_r_cost, r_cost, 128),
        ("A", fs_a_cost, a_cost, 128),
        ("B", fs_b_cost, b_cost, 128),
    ] {
        println!("SU3xSU3 {name}: fs {} calls/{} bytes; adapter {} calls/{} bytes; copied {copied} bytes",
            direct.calls, direct.bytes, adapted.calls, adapted.bytes);
    }
}

#[cfg(feature = "cgc-gen")]
#[test]
fn warm_nested_generic_product_queries() {
    use symtn_sectors::{
        CheckedGenericRigidSymbols, GenericRMatrix, SUNFusionRule, TypedSectorAdmission,
    };

    let rule = ProductFusionRule::<_, _>::try_new(
        ProductFusionRule::<_, _>::try_new(
            SUNFusionRule::new(3).unwrap(),
            SUNFusionRule::new(3).unwrap(),
        )
        .unwrap(),
        SUNFusionRule::new(3).unwrap(),
    )
    .unwrap();
    let octet = vec![1, 1];
    let id = rule
        .try_encode_label(&ProductSector::new(
            ProductSector::new(octet.clone(), octet.clone()),
            octet,
        ))
        .unwrap();
    let f = || rule.try_f_symbol_generic(id, id, id, id, id, id).unwrap();
    f();
    let (f, allocations) = counting_alloc::measure(f);
    assert_eq!(f.data().len(), 4096);
    assert_eq!((allocations.calls, allocations.bytes), (5, 69824));
    println!(
        "(SU3xSU3)xSU3 F: {} calls/{} bytes",
        allocations.calls, allocations.bytes
    );
    let queries: [(&str, &dyn Fn() -> GenericRMatrix<f64>); 3] = [
        ("R", &|| rule.try_r_symbol_generic(id, id, id).unwrap()),
        ("A", &|| rule.try_a_symbol_generic(id, id, id).unwrap()),
        ("B", &|| rule.try_b_symbol_generic(id, id, id).unwrap()),
    ];
    for (name, query) in queries {
        let expected = query();
        let (actual, allocations) = counting_alloc::measure(query);
        assert_eq!(actual.data(), expected.data());
        assert_eq!(actual.shape(), (8, 8));
        if name == "R" {
            assert_eq!((allocations.calls, allocations.bytes), (68, 19784));
        } else {
            assert!(
                allocations.calls <= 38 && allocations.bytes <= 2360,
                "{name}: {allocations:?}"
            );
        }
        println!(
            "(SU3xSU3)xSU3 {name}: {} calls/{} bytes",
            allocations.calls, allocations.bytes
        );
    }
    // The non-self-dual first/second input selects different retained duals.
    let label = |x: Vec<i64>| ProductSector::new(ProductSector::new(x.clone(), x.clone()), x);
    let three = rule.try_encode_label(&label(vec![1, 0])).unwrap();
    let unit = rule.try_encode_label(&label(vec![0, 0])).unwrap();
    for block in [
        rule.try_a_symbol_generic(three, unit, three).unwrap(),
        rule.try_b_symbol_generic(unit, three, three).unwrap(),
    ] {
        assert_eq!(block.shape(), (1, 1));
        numerics::assert_close("fundamental A/B", block.data()[0], 1.0, 27);
    }
}
