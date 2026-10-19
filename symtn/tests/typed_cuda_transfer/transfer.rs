use super::*;

fn assert_roundtrip<R>(source: TensorMap<R, f64>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    let provider = source.provider() as *const R;
    let runtime = symtn::typed::__network::runtime_identity(source.runtime());
    let structure = structural_snapshot(&source);
    let expected = source.dense_data().unwrap().to_vec();

    let device = source.to_cuda().unwrap();
    assert_eq!(device.placement(), symtn::expert::Placement::Cuda(0));
    let device_clone = device.clone();
    let restored = device_clone.to_host().unwrap();

    assert!(std::ptr::eq(restored.provider(), provider));
    assert!(runtime.matches(restored.runtime()));
    assert_eq!(restored.dense_data().unwrap(), expected);
    assert_eq!(structural_snapshot(&restored), structure);
}

#[test]
#[ignore]
fn builtin_and_simple_product_providers_round_trip_through_the_device() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();

    let u1_rule = Arc::new(U1FusionRule);
    let u1 = GradedSpace::try_new(
        Arc::clone(&u1_rule),
        [(U1Irrep::new(-1), 1), (U1Irrep::new(0), 2)],
    )
    .unwrap();
    assert_roundtrip(
        TensorMap::from_subblock_fn(&runtime, [&u1], [&u1], |_, indices| indices[0] as f64 + 1.0)
            .unwrap(),
    );

    let fz2_rule = Arc::new(FermionParityFusionRule);
    let fz2 = GradedSpace::try_new(
        Arc::clone(&fz2_rule),
        [(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 2)],
    )
    .unwrap();
    assert_roundtrip(
        TensorMap::from_subblock_fn(&runtime, [&fz2], [&fz2], |_, indices| {
            indices[0] as f64 + 2.0
        })
        .unwrap(),
    );

    let su2_rule = Arc::new(SU2FusionRule);
    let su2 = GradedSpace::try_new(
        Arc::clone(&su2_rule),
        [
            (SU2Irrep::from_twice_spin(0), 1),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap();
    assert_roundtrip(
        TensorMap::from_subblock_fn(&runtime, [&su2], [&su2], |_, indices| {
            indices[0] as f64 + 3.0
        })
        .unwrap(),
    );

    let product_rule = Arc::new(U1FusionRule.product(FermionParityFusionRule));
    let product = GradedSpace::try_new(
        Arc::clone(&product_rule),
        [
            (product_sector(U1Irrep::new(0), Z2Irrep::EVEN), 1),
            (product_sector(U1Irrep::new(1), Z2Irrep::ODD), 2),
        ],
    )
    .unwrap();
    assert_roundtrip(
        TensorMap::from_subblock_fn(&runtime, [&product], [&product], |_, indices| {
            indices[0] as f64 + 4.0
        })
        .unwrap(),
    );
}

fn assert_c64_roundtrip<R>(source: TensorMap<R, Complex64>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    let provider = source.provider() as *const R;
    let runtime = symtn::typed::__network::runtime_identity(source.runtime());
    let structure = structural_snapshot(&source);
    let expected = source.dense_data().unwrap().to_vec();
    assert!(
        expected.iter().all(|value| value.im != 0.0),
        "fixture must have nonzero imaginary parts"
    );

    let device = source.to_cuda().unwrap();
    assert_eq!(device.placement(), symtn::expert::Placement::Cuda(0));
    let restored = device.clone().to_host().unwrap();

    assert!(std::ptr::eq(restored.provider(), provider));
    assert!(runtime.matches(restored.runtime()));
    assert_eq!(restored.dense_data().unwrap(), expected);
    assert_eq!(structural_snapshot(&restored), structure);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_c64_roundtrip_is_bit_exact_across_providers() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();

    let u1 = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(-1), 1), (U1Irrep::new(0), 2)],
    )
    .unwrap();
    assert_c64_roundtrip(
        TensorMap::<_, Complex64>::from_subblock_fn(&runtime, [&u1], [&u1], |_, indices| {
            complex_entry(indices, 1.0)
        })
        .unwrap(),
    );

    let fz2 = GradedSpace::try_new(
        Arc::new(FermionParityFusionRule),
        [(Z2Irrep::EVEN, 1), (Z2Irrep::ODD, 2)],
    )
    .unwrap();
    assert_c64_roundtrip(
        TensorMap::<_, Complex64>::from_subblock_fn(&runtime, [&fz2], [&fz2], |_, indices| {
            complex_entry(indices, 2.0)
        })
        .unwrap(),
    );

    let su2 = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 1),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap();
    assert_c64_roundtrip(
        TensorMap::<_, Complex64>::from_subblock_fn(
            &runtime,
            [&su2, &su2, &su2],
            [&su2, &su2],
            |_, indices| complex_entry(indices, 3.0),
        )
        .unwrap(),
    );

    let product = GradedSpace::try_new(
        Arc::new(U1FusionRule.product(FermionParityFusionRule)),
        [
            (product_sector(U1Irrep::new(0), Z2Irrep::EVEN), 1),
            (product_sector(U1Irrep::new(1), Z2Irrep::ODD), 2),
        ],
    )
    .unwrap();
    assert_c64_roundtrip(
        TensorMap::<_, Complex64>::from_subblock_fn(
            &runtime,
            [&product],
            [&product],
            |_, indices| complex_entry(indices, 4.0),
        )
        .unwrap(),
    );
}
