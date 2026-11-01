use super::*;

#[cfg(feature = "cgc-gen")]
#[test]
fn sun_checked_generic_compact_qr_preserves_provider_and_reconstructs() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    use symtn::sector::LieGroupFusionRule;

    let runtime = Runtime::builder().build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |trees, _| {
            trees.coupled().iter().sum::<i64>() as f64 + 1.0
        })
        .unwrap();

    let Qr { q, r } = source.qr_compact(engine, &[0], &[1]).unwrap();
    assert!(std::ptr::eq(q.provider(), provider.as_ref()));
    assert!(std::ptr::eq(r.provider(), provider.as_ref()));
    let rebuilt = q.compose(engine, &r).unwrap();
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(source.dense_data().unwrap())
        .all(|(actual, expected)| (actual - expected).abs() < 1.0e-10));

    let complex = source.convert::<Complex64>(engine).unwrap();
    let Qr {
        q: complex_q,
        r: complex_r,
    } = complex.qr_compact(engine, &[0], &[1]).unwrap();
    assert!(std::ptr::eq(complex_q.provider(), provider.as_ref()));
    assert!(std::ptr::eq(complex_r.provider(), provider.as_ref()));
    let complex_rebuilt = complex_q.compose(engine, &complex_r).unwrap();
    assert!(complex_rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(complex.dense_data().unwrap())
        .all(|(actual, expected)| (*actual - *expected).norm() < 1.0e-10));
}

#[cfg(feature = "cgc-gen")]
#[test]
fn sun_checked_generic_compact_lq_preserves_provider_and_reconstructs() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    use symtn::sector::LieGroupFusionRule;

    let runtime = Runtime::builder().build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |trees, _| {
            trees.coupled().iter().sum::<i64>() as f64 + 1.0
        })
        .unwrap();

    let Lq { l, q } = source.lq_compact(engine, &[0], &[1]).unwrap();
    assert!(std::ptr::eq(l.provider(), provider.as_ref()));
    assert!(std::ptr::eq(q.provider(), provider.as_ref()));
    let rebuilt = l.compose(engine, &q).unwrap();
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(source.dense_data().unwrap())
        .all(|(actual, expected)| (actual - expected).abs() < 1.0e-10));

    let complex = source.convert::<Complex64>(engine).unwrap();
    let Lq {
        l: complex_l,
        q: complex_q,
    } = complex.lq_compact(engine, &[0], &[1]).unwrap();
    assert!(std::ptr::eq(complex_l.provider(), provider.as_ref()));
    assert!(std::ptr::eq(complex_q.provider(), provider.as_ref()));
    let complex_rebuilt = complex_l.compose(engine, &complex_q).unwrap();
    assert!(complex_rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(complex.dense_data().unwrap())
        .all(|(actual, expected)| (*actual - *expected).norm() < 1.0e-10));
}

#[cfg(feature = "cgc-gen")]
#[test]
fn sun_checked_generic_full_qr_preserves_provider_and_reconstructs() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    use symtn::sector::LieGroupFusionRule;

    let runtime = Runtime::builder().build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |trees, _| {
            trees.coupled().iter().sum::<i64>() as f64 + 1.0
        })
        .unwrap();

    let Qr { q, r } = source.qr_full(engine, &[0], &[1]).unwrap();
    assert!(std::ptr::eq(q.provider(), provider.as_ref()));
    assert!(std::ptr::eq(r.provider(), provider.as_ref()));
    let rebuilt = q.compose(engine, &r).unwrap();
    assert!(rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(source.dense_data().unwrap())
        .all(|(actual, expected)| (actual - expected).abs() < 1.0e-10));

    let complex = source.convert::<Complex64>(engine).unwrap();
    let Qr {
        q: complex_q,
        r: complex_r,
    } = complex.qr_full(engine, &[0], &[1]).unwrap();
    assert!(std::ptr::eq(complex_q.provider(), provider.as_ref()));
    assert!(std::ptr::eq(complex_r.provider(), provider.as_ref()));
    let complex_rebuilt = complex_q.compose(engine, &complex_r).unwrap();
    assert!(complex_rebuilt
        .dense_data()
        .unwrap()
        .iter()
        .zip(complex.dense_data().unwrap())
        .all(|(actual, expected)| (*actual - *expected).norm() < 1.0e-10));
}

#[test]
fn checked_generic_compact_qr_failure_is_typed_and_nonpublishing() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let _cache = cache_exclusive();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(120));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg], [&leg], |_, _| 2.0).unwrap();
    let before = source.dense_data().unwrap().to_vec();
    // The construction published the layouts this operation walks; a
    // provider whose answers change is not one identity, so the cached
    // layouts go first. The tag is this test's alone.
    forget_cached_structures();
    provider.fail_algebra.store(true, Ordering::Relaxed);
    let error = source.qr_compact(engine, &[0, 1], &[2]).unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Plan(symtn::typed::CheckedGenericPlanError::Provider(
            ToyError::Algebra
        ))
    ));
    assert_eq!(source.dense_data().unwrap(), before.as_slice());
}

#[test]
fn checked_generic_compact_lq_failure_is_typed_and_nonpublishing() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let _cache = cache_exclusive();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(121));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg], [&leg], |_, _| 2.0).unwrap();
    let before = source.dense_data().unwrap().to_vec();
    // The construction published the layouts this operation walks; a
    // provider whose answers change is not one identity, so the cached
    // layouts go first. The tag is this test's alone.
    forget_cached_structures();
    provider.fail_algebra.store(true, Ordering::Relaxed);
    let error = source.lq_compact(engine, &[0, 1], &[2]).unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Plan(symtn::typed::CheckedGenericPlanError::Provider(
            ToyError::Algebra
        ))
    ));
    assert_eq!(source.dense_data().unwrap(), before.as_slice());
}

#[test]
fn checked_generic_full_qr_failure_is_typed_and_nonpublishing() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg], [&leg], |_, _| 2.0).unwrap();
    let before = source.dense_data().unwrap().to_vec();
    provider.fail_algebra.store(true, Ordering::Relaxed);
    let error = source.qr_full(engine, &[0, 1], &[2]).unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Plan(symtn::typed::CheckedGenericPlanError::Provider(
            ToyError::Algebra
        ))
    ));
    assert_eq!(source.dense_data().unwrap(), before.as_slice());
}

#[test]
fn checked_generic_full_lq_failure_is_typed_and_nonpublishing() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg], [&leg], |_, _| 2.0).unwrap();
    let before = source.dense_data().unwrap().to_vec();
    provider.fail_algebra.store(true, Ordering::Relaxed);
    let error = source.lq_full(engine, &[0, 1], &[2]).unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Plan(symtn::typed::CheckedGenericPlanError::Provider(
            ToyError::Algebra
        ))
    ));
    assert_eq!(source.dense_data().unwrap(), before.as_slice());
}
