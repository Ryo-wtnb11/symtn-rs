use super::*;

#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_arithmetic_matches_host_lazy_ownership_and_concurrency() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let lhs: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
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
        TensorMap::from_subblock_fn(&runtime, [&rhs_leg], [&rhs_leg], |_, indices| {
            2.0 * indices.iter().sum::<usize>() as f64 - 1.0
        })
        .unwrap();
    let lhs_data = lhs.materialize().unwrap().dense_data().unwrap().to_vec();
    let rhs_data = rhs.materialize().unwrap().dense_data().unwrap().to_vec();
    let lhs_provider = lhs.provider() as *const U1FusionRule;
    let rhs_provider = rhs.provider() as *const U1FusionRule;
    let runtime_id = symtn::typed::__network::runtime_identity(&runtime);
    let lhs_device = lhs.to_cuda().unwrap();
    let rhs_device = rhs.to_cuda().unwrap();

    for factor in [2.5, -1.75] {
        let expected = lhs.scale(factor).unwrap();
        let actual = lhs_device.scale(factor).unwrap().to_host().unwrap();
        assert_eq!(
            actual.materialize().unwrap().dense_data().unwrap(),
            expected.materialize().unwrap().dense_data().unwrap()
        );
        assert!(std::ptr::eq(actual.provider(), lhs_provider));
        assert!(runtime_id.matches(actual.runtime()));
        assert_eq!(structural_snapshot(&actual), structural_snapshot(&lhs));
    }

    let (alpha, beta) = (2.25, -3.5);
    let expected_add = lhs.axpby(alpha, &rhs, beta).unwrap();
    let actual_add = lhs_device
        .axpby(alpha, &rhs_device, beta)
        .unwrap()
        .to_host()
        .unwrap();
    assert_eq!(
        actual_add.materialize().unwrap().dense_data().unwrap(),
        expected_add.materialize().unwrap().dense_data().unwrap()
    );
    assert!(std::ptr::eq(actual_add.provider(), lhs_provider));
    assert!(!std::ptr::eq(actual_add.provider(), rhs_provider));
    assert!(runtime_id.matches(actual_add.runtime()));
    assert_eq!(structural_snapshot(&actual_add), structural_snapshot(&lhs));
    assert_eq!(
        lhs_device.to_host().unwrap().dense_data().unwrap(),
        lhs_data
    );
    assert_eq!(
        rhs_device.to_host().unwrap().dense_data().unwrap(),
        rhs_data
    );
    for _ in 0..3 {
        assert_eq!(
            lhs_device
                .axpby(alpha, &rhs_device, beta)
                .unwrap()
                .to_host()
                .unwrap()
                .materialize()
                .unwrap()
                .dense_data()
                .unwrap(),
            expected_add.materialize().unwrap().dense_data().unwrap()
        );
    }

    let nonfinite_values = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -0.0];
    let nonfinite_index = std::cell::Cell::new(0usize);
    let nonfinite = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, _| {
        let index = nonfinite_index.get();
        nonfinite_index.set(index + 1);
        nonfinite_values[index % nonfinite_values.len()]
    })
    .unwrap();
    let nonfinite_bits: Vec<_> = nonfinite
        .materialize()
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .map(|value| value.to_bits())
        .collect();
    let nonfinite_device = nonfinite.to_cuda().unwrap();
    let exact_zero = nonfinite_device.zeros_like().unwrap().to_host().unwrap();
    assert!(exact_zero
        .materialize()
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .all(|value| value.to_bits() == 0));

    let finite_values = [0.0, -0.0, 2.0, -3.0, 1.0];
    let finite_index = std::cell::Cell::new(0usize);
    let finite = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, _| {
        let index = finite_index.get();
        finite_index.set(index + 1);
        finite_values[index % finite_values.len()]
    })
    .unwrap();
    let finite_device = finite.to_cuda().unwrap();
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
        let expected_scale_zero = nonfinite.scale(zero_factor).unwrap();
        let actual_scale_zero = nonfinite_device
            .scale(zero_factor)
            .unwrap()
            .to_host()
            .unwrap();
        assert_nonfinite_numeric_parity(
            actual_scale_zero
                .materialize()
                .unwrap()
                .dense_data()
                .unwrap(),
            expected_scale_zero
                .materialize()
                .unwrap()
                .dense_data()
                .unwrap(),
        );

        let expected_zero_alpha = nonfinite.axpby(zero_factor, &finite, 1.0).unwrap();
        let actual_zero_alpha = nonfinite_device
            .axpby(zero_factor, &finite_device, 1.0)
            .unwrap()
            .to_host()
            .unwrap();
        assert_nonfinite_numeric_parity(
            actual_zero_alpha
                .materialize()
                .unwrap()
                .dense_data()
                .unwrap(),
            expected_zero_alpha
                .materialize()
                .unwrap()
                .dense_data()
                .unwrap(),
        );

        let expected_zero_beta = finite.axpby(1.0, &nonfinite, zero_factor).unwrap();
        let actual_zero_beta = finite_device
            .axpby(1.0, &nonfinite_device, zero_factor)
            .unwrap()
            .to_host()
            .unwrap();
        assert_nonfinite_numeric_parity(
            actual_zero_beta
                .materialize()
                .unwrap()
                .dense_data()
                .unwrap(),
            expected_zero_beta
                .materialize()
                .unwrap()
                .dense_data()
                .unwrap(),
        );
        // TensorKit's `scale(x, 0)` drops a zero-scaled operand, NaN and Inf
        // included (#1442).
        assert!(actual_scale_zero
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .all(|&value| value == 0.0));
        assert_eq!(
            actual_zero_alpha
                .materialize()
                .unwrap()
                .dense_data()
                .unwrap(),
            finite.materialize().unwrap().dense_data().unwrap()
        );
        assert_eq!(
            actual_zero_beta
                .materialize()
                .unwrap()
                .dense_data()
                .unwrap(),
            finite.materialize().unwrap().dense_data().unwrap()
        );
    }
    assert_eq!(
        nonfinite_device
            .to_host()
            .unwrap()
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        nonfinite_bits
    );

    let lhs_lazy = lhs_device.adjoint().unwrap();
    let rhs_lazy = rhs_device.adjoint().unwrap();
    let lazy_scale = lhs_lazy.scale(alpha).unwrap().to_host().unwrap();
    let lazy_add = lhs_lazy
        .axpby(alpha, &rhs_lazy, beta)
        .unwrap()
        .to_host()
        .unwrap();
    let lazy_zero = lhs_lazy.zeros_like().unwrap().to_host().unwrap();
    assert_eq!(
        lazy_scale.materialize().unwrap().dense_data().unwrap(),
        lhs.adjoint()
            .unwrap()
            .scale(alpha)
            .unwrap()
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap()
    );
    assert_eq!(
        lazy_add.materialize().unwrap().dense_data().unwrap(),
        lhs.adjoint()
            .unwrap()
            .axpby(alpha, &rhs.adjoint().unwrap(), beta)
            .unwrap()
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap()
    );
    assert!(std::ptr::eq(lazy_add.provider(), lhs_provider));
    assert!(!std::ptr::eq(lazy_add.provider(), rhs_provider));
    assert!(lazy_zero
        .materialize()
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .all(|value| value.to_bits() == 0));
    // One lazy and one owned operand (#1874): the Host rule, an owned
    // result on the receiver's provider.
    let mixed = lhs_lazy.axpby(alpha, &rhs_device, beta).unwrap();
    assert!(std::ptr::eq(mixed.provider(), lhs_provider));
    let expected = lhs.adjoint().unwrap().axpby(alpha, &rhs, beta).unwrap();
    for (device, host) in mixed
        .to_host()
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(expected.dense_data().unwrap())
    {
        assert!((device - host).abs() <= 1e-12 * (1.0 + host.abs()));
    }

    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    lhs_device
                        .axpby(alpha, &rhs_device, beta)
                        .unwrap()
                        .to_host()
                        .unwrap()
                        .materialize()
                        .unwrap()
                        .dense_data()
                        .unwrap()
                        .to_vec()
                })
            })
            .collect();
        for worker in workers {
            assert_eq!(
                worker.join().unwrap(),
                expected_add.materialize().unwrap().dense_data().unwrap()
            );
        }
    });

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
        TensorMap::from_subblock_fn(&runtime, [&su2_leg], [&su2_leg], |trees, _| {
            if trees.coupled() == &SU2Irrep::from_twice_spin(0) {
                1.0
            } else {
                2.0
            }
        })
        .unwrap();
    // SU(2) weights the norm by dim(c); the device reduction must too.
    assert!((su2.to_cuda().unwrap().norm(2.0).unwrap() - su2.norm(2.0).unwrap()).abs() < 1e-12);

    let zn3 = Arc::new(ZNFusionRule::new(3).unwrap());
    let charge0 = GradedSpace::try_new(Arc::clone(&zn3), [(zn3.irrep(0), 1)]).unwrap();
    let charge1 = GradedSpace::try_new(Arc::clone(&zn3), [(zn3.irrep(1), 1)]).unwrap();
    let empty: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&charge0], [&charge1], |_, _| f64::NAN).unwrap();
    let empty_device = empty.to_cuda().unwrap();
    assert!(empty_device
        .scale(2.0)
        .unwrap()
        .to_host()
        .unwrap()
        .materialize()
        .unwrap()
        .dense_data()
        .unwrap()
        .is_empty());
    assert!(empty_device
        .axpby(alpha, &empty_device, beta)
        .unwrap()
        .to_host()
        .unwrap()
        .materialize()
        .unwrap()
        .dense_data()
        .unwrap()
        .is_empty());
    assert!(empty_device
        .zeros_like()
        .unwrap()
        .to_host()
        .unwrap()
        .materialize()
        .unwrap()
        .dense_data()
        .unwrap()
        .is_empty());

    let other_runtime = Runtime::builder().cuda(0).build().unwrap();
    let other_leg = GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0), 1)]).unwrap();
    let foreign =
        TensorMap::from_subblock_fn(&other_runtime, [&other_leg], [&other_leg], |_, _| 1.0)
            .unwrap()
            .to_cuda()
            .unwrap();
    assert_eq!(
        lhs_device.axpby(alpha, &foreign, beta).unwrap_err(),
        symtn::typed::Error::RuntimeMismatch
    );
    let mismatched = TensorMap::from_subblock_fn(&runtime, [&other_leg], [&other_leg], |_, _| 1.0)
        .unwrap()
        .to_cuda()
        .unwrap();
    assert!(matches!(
        lhs_device.axpby(alpha, &mismatched, beta),
        Err(symtn::typed::Error::Operation(operation))
            if matches!(*operation, symtn::typed::OperationError::SpaceMismatch { .. })
    ));
    assert_eq!(
        lhs_device.to_host().unwrap().dense_data().unwrap(),
        lhs_data
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_c64_inner_is_conjugate_linear_in_the_first_argument() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
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
        let a =
            TensorMap::<_, Complex64>::from_subblock_fn(&runtime, [&su2], [&su2], |_, indices| {
                complex_entry(indices, seed)
            })
            .unwrap();
        let b =
            TensorMap::<_, Complex64>::from_subblock_fn(&runtime, [&su2], [&su2], |_, indices| {
                complex_entry(indices, seed + 2.0)
            })
            .unwrap();
        let host_inner = a.inner(&b).unwrap();
        let host_norm = a.norm(2.0).unwrap();
        assert!(
            host_inner.im != 0.0,
            "the fixture must exercise a complex inner product"
        );

        let a_device = a.to_cuda().unwrap();
        let b_device = b.to_cuda().unwrap();
        let inner = a_device.inner(&b_device).unwrap();
        let tolerance = 1e-12 * (1.0 + host_inner.norm());
        assert!((inner - host_inner).norm() <= tolerance);
        assert!((b_device.inner(&a_device).unwrap() - inner.conj()).norm() <= tolerance);
        let scaled = a_device.scale(i).unwrap();
        numerics::assert_slices_close(
            "scaled",
            scaled.to_host().unwrap().dense_data().unwrap(),
            a.scale(i).unwrap().dense_data().unwrap(),
            ELEMENTWISE_TERMS,
        );
        assert!((scaled.inner(&b_device).unwrap() - (-i) * inner).norm() <= tolerance);
        assert!(
            (a_device.inner(&b_device.scale(i).unwrap()).unwrap() - i * inner).norm() <= tolerance
        );

        let norm = a_device.norm(2.0).unwrap();
        assert!((norm - host_norm).abs() <= 1e-12 * (1.0 + host_norm));
        let self_inner = a_device.inner(&a_device).unwrap();
        assert!(self_inner.im.abs() <= 1e-12 * (1.0 + self_inner.norm()));
        assert!((self_inner.re - norm * norm).abs() <= 1e-12 * (1.0 + self_inner.norm()));
    }
}
