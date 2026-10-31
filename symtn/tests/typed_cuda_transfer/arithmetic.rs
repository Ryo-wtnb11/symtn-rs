use super::*;

#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_arithmetic_matches_host_lazy_ownership_and_concurrency() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut crate::common::cuda_engine();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let lhs: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();
    let rhs_provider = Arc::new(U1FusionRule);
    let rhs_leg = GradedSpace::try_new(
        Arc::clone(&rhs_provider),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let rhs: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&rhs_leg], [&rhs_leg], |_, indices| {
            2.0 * indices.iter().sum::<usize>() as f64 - 1.0
        })
        .unwrap();
    let lhs_data = lhs
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .to_vec();
    let rhs_data = rhs
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .to_vec();
    let lhs_provider = lhs.provider() as *const U1FusionRule;
    let rhs_provider = rhs.provider() as *const U1FusionRule;
    let runtime_id = symtn::typed::__network::runtime_identity(&runtime);
    let lhs_device = lhs.to_cuda(cuda).unwrap();
    let rhs_device = rhs.to_cuda(cuda).unwrap();

    for factor in [2.5, -1.75] {
        let expected = lhs.scale(engine, factor).unwrap();
        let actual = lhs_device
            .scale(cuda, factor)
            .unwrap()
            .to_host(cuda)
            .unwrap();
        assert_eq!(
            actual.materialize(engine).unwrap().dense_data().unwrap(),
            expected.materialize(engine).unwrap().dense_data().unwrap()
        );
        assert!(std::ptr::eq(actual.provider(), lhs_provider));
        assert!(runtime_id.matches(actual.runtime()));
        assert_eq!(structural_snapshot(&actual), structural_snapshot(&lhs));
    }

    let (alpha, beta) = (2.25, -3.5);
    let expected_add = lhs.axpby(engine, alpha, &rhs, beta).unwrap();
    let actual_add = lhs_device
        .axpby(cuda, alpha, &rhs_device, beta)
        .unwrap()
        .to_host(cuda)
        .unwrap();
    assert_eq!(
        actual_add
            .materialize(engine)
            .unwrap()
            .dense_data()
            .unwrap(),
        expected_add
            .materialize(engine)
            .unwrap()
            .dense_data()
            .unwrap()
    );
    assert!(std::ptr::eq(actual_add.provider(), lhs_provider));
    assert!(!std::ptr::eq(actual_add.provider(), rhs_provider));
    assert!(runtime_id.matches(actual_add.runtime()));
    assert_eq!(structural_snapshot(&actual_add), structural_snapshot(&lhs));
    assert_eq!(
        lhs_device.to_host(cuda).unwrap().dense_data().unwrap(),
        lhs_data
    );
    assert_eq!(
        rhs_device.to_host(cuda).unwrap().dense_data().unwrap(),
        rhs_data
    );
    for _ in 0..3 {
        assert_eq!(
            lhs_device
                .axpby(cuda, alpha, &rhs_device, beta)
                .unwrap()
                .to_host(cuda)
                .unwrap()
                .materialize(engine)
                .unwrap()
                .dense_data()
                .unwrap(),
            expected_add
                .materialize(engine)
                .unwrap()
                .dense_data()
                .unwrap()
        );
    }

    let nonfinite_values = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -0.0];
    let nonfinite_index = std::cell::Cell::new(0usize);
    let nonfinite = TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, _| {
        let index = nonfinite_index.get();
        nonfinite_index.set(index + 1);
        nonfinite_values[index % nonfinite_values.len()]
    })
    .unwrap();
    let nonfinite_bits: Vec<_> = nonfinite
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .map(|value| value.to_bits())
        .collect();
    let nonfinite_device = nonfinite.to_cuda(cuda).unwrap();
    let exact_zero = nonfinite_device
        .zeros_like(cuda)
        .unwrap()
        .to_host(cuda)
        .unwrap();
    assert!(exact_zero
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .all(|value| value.to_bits() == 0));

    let finite_values = [0.0, -0.0, 2.0, -3.0, 1.0];
    let finite_index = std::cell::Cell::new(0usize);
    let finite = TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, _| {
        let index = finite_index.get();
        finite_index.set(index + 1);
        finite_values[index % finite_values.len()]
    })
    .unwrap();
    let finite_device = finite.to_cuda(cuda).unwrap();
    let assert_nonfinite_numeric_parity = |actual: &[f64], expected: &[f64]| {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected) {
            if expected.is_nan() {
                assert!(actual.is_nan(), "expected NaN, got {actual:?}");
            } else {
                assert_eq!(actual, expected);
            }
        }
    };

    for zero_factor in [0.0, -0.0] {
        let expected_scale_zero = nonfinite.scale(engine, zero_factor).unwrap();
        let actual_scale_zero = nonfinite_device
            .scale(cuda, zero_factor)
            .unwrap()
            .to_host(cuda)
            .unwrap();
        assert_nonfinite_numeric_parity(
            actual_scale_zero
                .materialize(engine)
                .unwrap()
                .dense_data()
                .unwrap(),
            expected_scale_zero
                .materialize(engine)
                .unwrap()
                .dense_data()
                .unwrap(),
        );

        let expected_zero_alpha = nonfinite.axpby(engine, zero_factor, &finite, 1.0).unwrap();
        let actual_zero_alpha = nonfinite_device
            .axpby(cuda, zero_factor, &finite_device, 1.0)
            .unwrap()
            .to_host(cuda)
            .unwrap();
        assert_nonfinite_numeric_parity(
            actual_zero_alpha
                .materialize(engine)
                .unwrap()
                .dense_data()
                .unwrap(),
            expected_zero_alpha
                .materialize(engine)
                .unwrap()
                .dense_data()
                .unwrap(),
        );

        let expected_zero_beta = finite.axpby(engine, 1.0, &nonfinite, zero_factor).unwrap();
        let actual_zero_beta = finite_device
            .axpby(cuda, 1.0, &nonfinite_device, zero_factor)
            .unwrap()
            .to_host(cuda)
            .unwrap();
        assert_nonfinite_numeric_parity(
            actual_zero_beta
                .materialize(engine)
                .unwrap()
                .dense_data()
                .unwrap(),
            expected_zero_beta
                .materialize(engine)
                .unwrap()
                .dense_data()
                .unwrap(),
        );
        // TensorKit's `scale(x, 0)` drops a zero-scaled operand, NaN and Inf
        // included.
        assert!(actual_scale_zero
            .materialize(engine)
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .all(|&value| value == 0.0));
        assert_eq!(
            actual_zero_alpha
                .materialize(engine)
                .unwrap()
                .dense_data()
                .unwrap(),
            finite.materialize(engine).unwrap().dense_data().unwrap()
        );
        assert_eq!(
            actual_zero_beta
                .materialize(engine)
                .unwrap()
                .dense_data()
                .unwrap(),
            finite.materialize(engine).unwrap().dense_data().unwrap()
        );
    }
    assert_eq!(
        nonfinite_device
            .to_host(cuda)
            .unwrap()
            .materialize(engine)
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        nonfinite_bits
    );

    let lhs_lazy = lhs_device.adjoint(cuda).unwrap();
    let rhs_lazy = rhs_device.adjoint(cuda).unwrap();
    let lazy_scale = lhs_lazy.scale(cuda, alpha).unwrap().to_host(cuda).unwrap();
    let lazy_add = lhs_lazy
        .axpby(cuda, alpha, &rhs_lazy, beta)
        .unwrap()
        .to_host(cuda)
        .unwrap();
    let lazy_zero = lhs_lazy.zeros_like(cuda).unwrap().to_host(cuda).unwrap();
    assert_eq!(
        lazy_scale
            .materialize(engine)
            .unwrap()
            .dense_data()
            .unwrap(),
        lhs.adjoint(engine)
            .unwrap()
            .scale(engine, alpha)
            .unwrap()
            .materialize(engine)
            .unwrap()
            .dense_data()
            .unwrap()
    );
    let rhs_adjoint = rhs.adjoint(engine).unwrap();
    assert_eq!(
        lazy_add.materialize(engine).unwrap().dense_data().unwrap(),
        lhs.adjoint(engine)
            .unwrap()
            .axpby(engine, alpha, &rhs_adjoint, beta)
            .unwrap()
            .materialize(engine)
            .unwrap()
            .dense_data()
            .unwrap()
    );
    assert!(std::ptr::eq(lazy_add.provider(), lhs_provider));
    assert!(!std::ptr::eq(lazy_add.provider(), rhs_provider));
    assert!(lazy_zero
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .all(|value| value.to_bits() == 0));
    // One lazy and one owned operand: the Host rule, an owned
    // result on the receiver's provider.
    let mixed = lhs_lazy.axpby(cuda, alpha, &rhs_device, beta).unwrap();
    assert!(std::ptr::eq(mixed.provider(), lhs_provider));
    let expected = lhs
        .adjoint(engine)
        .unwrap()
        .axpby(engine, alpha, &rhs, beta)
        .unwrap();
    for (device, host) in mixed
        .to_host(cuda)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(expected.dense_data().unwrap())
    {
        assert!((device - host).abs() <= 1e-12 * (1.0 + host.abs()));
    }

    let su2_provider = Arc::new(SU2FusionRule);
    let su2_leg = GradedSpace::try_new(
        Arc::clone(&su2_provider),
        [
            (SU2Irrep::from_twice_spin(0), 1),
            (SU2Irrep::from_twice_spin(1), 1),
        ],
    )
    .unwrap();
    let su2: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&su2_leg], [&su2_leg], |trees, _| {
            if trees.coupled() == &SU2Irrep::from_twice_spin(0) {
                1.0
            } else {
                2.0
            }
        })
        .unwrap();
    // SU(2) weights the norm by dim(c); the device reduction must too.
    assert!(
        (su2.to_cuda(cuda).unwrap().norm(cuda, 2.0).unwrap() - su2.norm(engine, 2.0).unwrap())
            .abs()
            < 1e-12
    );

    let zn3 = Arc::new(ZNFusionRule::new(3).unwrap());
    let charge0 = GradedSpace::try_new(Arc::clone(&zn3), [(zn3.irrep(0), 1)]).unwrap();
    let charge1 = GradedSpace::try_new(Arc::clone(&zn3), [(zn3.irrep(1), 1)]).unwrap();
    let empty: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&charge0], [&charge1], |_, _| f64::NAN)
            .unwrap();
    let empty_device = empty.to_cuda(cuda).unwrap();
    assert!(empty_device
        .scale(cuda, 2.0)
        .unwrap()
        .to_host(cuda)
        .unwrap()
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .is_empty());
    assert!(empty_device
        .axpby(cuda, alpha, &empty_device, beta)
        .unwrap()
        .to_host(cuda)
        .unwrap()
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .is_empty());
    assert!(empty_device
        .zeros_like(cuda)
        .unwrap()
        .to_host(cuda)
        .unwrap()
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .is_empty());

    let other_runtime = Runtime::builder().build().unwrap();
    let other_leg = GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0), 1)]).unwrap();
    let foreign = TensorMap::from_subblock_fn(
        engine,
        &other_runtime,
        [&other_leg],
        [&other_leg],
        |_, _| 1.0,
    )
    .unwrap()
    .to_cuda(cuda)
    .unwrap();
    assert_eq!(
        lhs_device.axpby(cuda, alpha, &foreign, beta).unwrap_err(),
        symtn::typed::Error::RuntimeMismatch
    );
    let mismatched =
        TensorMap::from_subblock_fn(engine, &runtime, [&other_leg], [&other_leg], |_, _| 1.0)
            .unwrap()
            .to_cuda(cuda)
            .unwrap();
    assert!(matches!(
        lhs_device.axpby(cuda, alpha, &mismatched, beta),
        Err(symtn::typed::Error::Operation(operation))
            if matches!(*operation, symtn::typed::OperationError::SpaceMismatch { .. })
    ));
    assert_eq!(
        lhs_device.to_host(cuda).unwrap().dense_data().unwrap(),
        lhs_data
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_c64_inner_is_conjugate_linear_in_the_first_argument() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let cuda = &mut crate::common::cuda_engine();
    let runtime = Runtime::builder().build().unwrap();
    let i = Complex64::new(0.0, 1.0);

    for (sectors, seed) in [(0_usize, 1.0_f64), (1, 7.0)] {
        let su2 = GradedSpace::try_new(
            Arc::new(SU2FusionRule),
            [
                (SU2Irrep::from_twice_spin(0), 1 + sectors),
                (SU2Irrep::from_twice_spin(1), 2),
            ],
        )
        .unwrap();
        let a = TensorMap::<_, Complex64>::from_subblock_fn(
            engine,
            &runtime,
            [&su2],
            [&su2],
            |_, indices| complex_entry(indices, seed),
        )
        .unwrap();
        let b = TensorMap::<_, Complex64>::from_subblock_fn(
            engine,
            &runtime,
            [&su2],
            [&su2],
            |_, indices| complex_entry(indices, seed + 2.0),
        )
        .unwrap();
        let host_inner = a.inner(engine, &b).unwrap();
        let host_norm = a.norm(engine, 2.0).unwrap();
        assert!(
            host_inner.im != 0.0,
            "the fixture must exercise a complex inner product"
        );

        let a_device = a.to_cuda(cuda).unwrap();
        let b_device = b.to_cuda(cuda).unwrap();
        let inner = a_device.inner(cuda, &b_device).unwrap();
        let tolerance = 1e-12 * (1.0 + host_inner.norm());
        assert!((inner - host_inner).norm() <= tolerance);
        assert!((b_device.inner(cuda, &a_device).unwrap() - inner.conj()).norm() <= tolerance);
        let scaled = a_device.scale(cuda, i).unwrap();
        numerics::assert_slices_close(
            "scaled",
            scaled.to_host(cuda).unwrap().dense_data().unwrap(),
            a.scale(engine, i).unwrap().dense_data().unwrap(),
            ELEMENTWISE_TERMS,
        );
        assert!((scaled.inner(cuda, &b_device).unwrap() - (-i) * inner).norm() <= tolerance);
        let b_scaled = b_device.scale(cuda, i).unwrap();
        assert!((a_device.inner(cuda, &b_scaled).unwrap() - i * inner).norm() <= tolerance);

        let norm = a_device.norm(cuda, 2.0).unwrap();
        assert!((norm - host_norm).abs() <= 1e-12 * (1.0 + host_norm));
        let self_inner = a_device.inner(cuda, &a_device).unwrap();
        assert!(self_inner.im.abs() <= 1e-12 * (1.0 + self_inner.norm()));
        assert!((self_inner.re - norm * norm).abs() <= 1e-12 * (1.0 + self_inner.norm()));
    }
}
