//! Eager warm-query allocation evidence; run alone with `--nocapture`.
use crate::counting_alloc;

#[cfg(feature = "cgc-gen")]
use crate::numerics;
use symtn_sectors::{
    CheckedFusionAlgebra, FermionParityFusionRule, MultiplicityFreeFusionSymbols,
    MultiplicityFreeRigidSymbols, ProductFusionRule, ProductSector, SectorCodec, U1FusionRule,
    U1Irrep, Z2Irrep,
};

#[test]
fn warm_product_queries() {
    let _serial = crate::counting_alloc::serial();
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
    let _serial = crate::counting_alloc::serial();
    use fusion_symbols::category::{FusionCategory, SymbolBlock};
    use symtn_sectors::{
        CategoryProvider, CheckedGenericRigidSymbols, LieGroupFusionRule, TypedSectorAdmission,
    };

    let rule = ProductFusionRule::<_, _>::try_new(
        LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap(),
        LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap(),
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
    assert_eq!(a.data(), fs_a.as_slice());
    assert_eq!(b.data(), fs_b.as_slice());
    assert_eq!(f.data().len(), 256);
    assert_eq!(r.data().len(), 16);
    assert_eq!(a.data().len(), 16);
    assert_eq!(b.data().len(), 16);
    // The adapter's owned block copies are separate from fs product storage.
    assert!(f_cost.matched_calls >= 1);
    assert!(r_cost.matched_calls >= 1);
    assert!(a_cost.matched_calls >= 1);
    assert!(b_cost.matched_calls >= 1);
    for (name, direct, adapted, len) in [
        ("F", fs_f_cost, f_cost, f.data().len()),
        ("R", fs_r_cost, r_cost, r.data().len()),
        ("A", fs_a_cost, a_cost, a.data().len()),
        ("B", fs_b_cost, b_cost, b.data().len()),
    ] {
        println!(
            "SU3xSU3 {name}: fs {} calls/{} bytes; adapter {} calls/{} bytes; len {len}",
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
}

#[cfg(feature = "cgc-gen")]
#[test]
fn warm_nested_generic_product_queries() {
    let _serial = crate::counting_alloc::serial();
    use fusion_symbols::category::{FusionCategory, SymbolBlock};
    use symtn_sectors::{
        CategoryProvider, CheckedGenericRigidSymbols, LieGroupFusionRule, TypedSectorAdmission,
    };

    let rule = ProductFusionRule::<_, _>::try_new(
        ProductFusionRule::<_, _>::try_new(
            LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap(),
            LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap(),
        )
        .unwrap(),
        LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap(),
    )
    .unwrap();
    let octet = vec![1, 1];
    let id = rule
        .try_encode_label(&ProductSector::new(
            ProductSector::new(octet.clone(), octet.clone()),
            octet,
        ))
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
    let (f, f_cost) = counting_alloc::measure(adapted_f);
    let (fs_r, fs_r_cost) = counting_alloc::measure(direct_r);
    let (r, r_cost) = counting_alloc::measure(adapted_r);
    let (fs_a, fs_a_cost) = counting_alloc::measure(direct_a);
    let (a, a_cost) = counting_alloc::measure(adapted_a);
    let (fs_b, fs_b_cost) = counting_alloc::measure(direct_b);
    let (b, b_cost) = counting_alloc::measure(adapted_b);
    assert_eq!(f.data(), fs_f.as_slice());
    assert_eq!(r.data(), fs_r.as_slice());
    assert_eq!(a.data(), fs_a.as_slice());
    assert_eq!(b.data(), fs_b.as_slice());
    assert_eq!(f.data().len(), 4096);
    for block in [&r, &a, &b] {
        assert_eq!(block.shape(), (8, 8));
    }
    for (name, direct, adapted, len) in [
        ("F", fs_f_cost, f_cost, f.data().len()),
        ("R", fs_r_cost, r_cost, r.data().len()),
        ("A", fs_a_cost, a_cost, a.data().len()),
        ("B", fs_b_cost, b_cost, b.data().len()),
    ] {
        println!(
            "(SU3xSU3)xSU3 {name}: fs {} calls/{} bytes; adapter {} calls/{} bytes; len {len}",
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
