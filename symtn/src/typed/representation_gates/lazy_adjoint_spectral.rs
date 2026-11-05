use super::*;

/// Solve only; every solve fails with `failure` when one is given.
#[cfg(feature = "cgc-gen")]
fn solve_spy(counts: &Arc<SpyCounts>, failure: Option<&'static str>) -> SpyExecutor {
    let spy = SpyExecutor::counting(counts).only(&[Kernel::Solve], "test only exercises solve");
    match failure {
        Some(message) => spy.failing(&[Kernel::Solve], None, message),
        None => spy,
    }
}

fn assert_eigh_uses_a_cold_logical_copy(
    engine: &mut dyn Engine,
    source: &TensorMap<U1FusionRule, f64>,
) {
    let eager = eager_adjoint_oracle(source);
    let expected_vals = eager
        .eigh_vals(engine, &[0], &[1], HermitianTol::DEFAULT)
        .unwrap();
    let expected_full = eager
        .eigh_full(engine, &[0], &[1], HermitianTol::DEFAULT)
        .unwrap();
    let parent_body = Arc::clone(owned(source));
    let parent_data = Arc::clone(&parent_body.data);
    let lazy = source.adjoint(engine).unwrap();

    for _ in 0..2 {
        assert_eq!(
            lazy.clone()
                .eigh_vals(engine, &[0], &[1], HermitianTol::DEFAULT)
                .unwrap(),
            expected_vals
        );
        let full = lazy
            .clone()
            .eigh_full(engine, &[0], &[1], HermitianTol::DEFAULT)
            .unwrap();
        assert_eq!(
            full.d.materialize(engine).unwrap().dense_data().unwrap(),
            expected_full
                .d
                .materialize(engine)
                .unwrap()
                .dense_data()
                .unwrap()
        );
        assert_eq!(
            full.v.dense_data().unwrap(),
            expected_full.v.dense_data().unwrap()
        );
        for output in [&full.d, &full.v] {
            assert!(output.owned_body().is_some());
            assert!(Arc::ptr_eq(
                output.logical_space().provider_arc(),
                source.logical_space().provider_arc()
            ));
        }
    }

    let calls = (0..4)
        .map(|_| {
            let clone = lazy.clone();
            std::thread::spawn(move || {
                let engine = &mut symtn_dense::CpuEngine::serial();
                let vals = clone
                    .eigh_vals(engine, &[0], &[1], HermitianTol::DEFAULT)
                    .unwrap();
                let full = clone
                    .eigh_full(engine, &[0], &[1], HermitianTol::DEFAULT)
                    .unwrap();
                (
                    vals,
                    full.d
                        .materialize(engine)
                        .unwrap()
                        .dense_data()
                        .unwrap()
                        .to_vec(),
                )
            })
        })
        .collect::<Vec<_>>();
    for call in calls {
        let (vals, diagonal) = call.join().unwrap();
        assert_eq!(vals, expected_vals);
        assert_eq!(
            diagonal,
            expected_full
                .d
                .materialize(engine)
                .unwrap()
                .dense_data()
                .unwrap()
        );
    }
    assert!(Arc::ptr_eq(owned(source), &parent_body));
    assert!(Arc::ptr_eq(&owned(source).data, &parent_data));
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}

#[test]
fn eigh_dense_lazy_near_hermitian_uses_logical_triangle_and_stays_cold() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 2)]).unwrap();
    let source =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, indices| {
            match (indices[0], indices[1]) {
                (0, 0) | (1, 1) => 1.0,
                (0, 1) => 4.0e-15,
                _ => 0.0,
            }
        })
        .unwrap();
    let logical = eager_adjoint_oracle(&source);
    let logical_vals = logical
        .eigh_vals(engine, &[0], &[1], HermitianTol::DEFAULT)
        .unwrap();
    let parent_vals = source
        .eigh_vals(engine, &[0], &[1], HermitianTol::DEFAULT)
        .unwrap();
    assert!(logical_vals[0]
        .values
        .iter()
        .zip(&parent_vals[0].values)
        .any(|(logical, parent)| (logical - parent).abs() > 1.0e-15));

    assert_eigh_uses_a_cold_logical_copy(engine, &source);
}

#[test]
fn eigh_dense_lazy_complex_orientation_and_failures_match_logical_oracles() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 2)]).unwrap();
    let hermitian = TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, indices| {
        match (indices[0], indices[1]) {
            (0, 0) => num_complex::Complex64::new(2.0, 0.0),
            (1, 1) => num_complex::Complex64::new(3.0, 0.0),
            (0, 1) => num_complex::Complex64::new(0.0, 1.0),
            (1, 0) => num_complex::Complex64::new(0.0, -1.0),
            _ => unreachable!(),
        }
    })
    .unwrap();
    let eager = eager_adjoint_oracle(&hermitian);
    let expected = eager
        .eigh_full(engine, &[0], &[1], HermitianTol::DEFAULT)
        .unwrap();
    let lazy = hermitian.adjoint(engine).unwrap();
    let actual = lazy
        .eigh_full(engine, &[0], &[1], HermitianTol::DEFAULT)
        .unwrap();
    assert_eq!(
        actual.d.materialize(engine).unwrap().dense_data().unwrap(),
        expected
            .d
            .materialize(engine)
            .unwrap()
            .dense_data()
            .unwrap()
    );
    assert_eq!(
        actual.v.dense_data().unwrap(),
        expected.v.dense_data().unwrap()
    );
    let actual_v = &actual.v.adjoint(engine).unwrap();
    let reconstructed = actual
        .v
        .compose(engine, &actual.d)
        .unwrap()
        .compose(engine, actual_v)
        .unwrap();
    assert_typed_map_close(engine, &reconstructed, &eager, 1.0e-12);
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };

    let nonhermitian =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, indices| {
            match (indices[0], indices[1]) {
                (0, 0) => 1.0,
                (1, 1) => 2.0,
                (0, 1) => 1.0,
                _ => 0.0,
            }
        })
        .unwrap();
    let eager = eager_adjoint_oracle(&nonhermitian);
    let expected = [
        eager
            .eigh_vals(engine, &[0], &[1], HermitianTol::DEFAULT)
            .unwrap_err()
            .to_string(),
        eager
            .eigh_full(engine, &[0], &[1], HermitianTol::DEFAULT)
            .unwrap_err()
            .to_string(),
    ];
    let lazy = nonhermitian.adjoint(engine).unwrap();
    for _ in 0..2 {
        assert_eq!(
            lazy.eigh_vals(engine, &[0], &[1], HermitianTol::DEFAULT)
                .unwrap_err()
                .to_string(),
            expected[0]
        );
        assert_eq!(
            lazy.eigh_full(engine, &[0], &[1], HermitianTol::DEFAULT)
                .unwrap_err()
                .to_string(),
            expected[1]
        );
    }
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}

fn assert_eig_uses_a_cold_logical_copy(
    engine: &mut dyn Engine,
    source: &TensorMap<U1FusionRule, f64>,
) {
    let eager = eager_adjoint_oracle(source);
    let expected_vals = eager.eig_vals(engine, &[0], &[1]).unwrap();
    let expected_full = eager.eig_full(engine, &[0], &[1]).unwrap();
    let parent_body = Arc::clone(owned(source));
    let parent_data = Arc::clone(&parent_body.data);
    let lazy = source.adjoint(engine).unwrap();

    for _ in 0..2 {
        assert_eq!(
            lazy.clone().eig_vals(engine, &[0], &[1]).unwrap(),
            expected_vals
        );
        let full = lazy.clone().eig_full(engine, &[0], &[1]).unwrap();
        assert_eq!(
            full.d.materialize(engine).unwrap().dense_data().unwrap(),
            expected_full
                .d
                .materialize(engine)
                .unwrap()
                .dense_data()
                .unwrap()
        );
        assert_eq!(
            full.v.dense_data().unwrap(),
            expected_full.v.dense_data().unwrap()
        );
        for output in [&full.d, &full.v] {
            assert!(output.owned_body().is_some());
            assert!(Arc::ptr_eq(
                output.logical_space().provider_arc(),
                source.logical_space().provider_arc()
            ));
        }
    }

    let calls = (0..4)
        .map(|_| {
            let clone = lazy.clone();
            std::thread::spawn(move || {
                let engine = &mut symtn_dense::CpuEngine::serial();
                let vals = clone.eig_vals(engine, &[0], &[1]).unwrap();
                let full = clone.eig_full(engine, &[0], &[1]).unwrap();
                (
                    vals,
                    full.d
                        .materialize(engine)
                        .unwrap()
                        .dense_data()
                        .unwrap()
                        .to_vec(),
                )
            })
        })
        .collect::<Vec<_>>();
    for call in calls {
        let (vals, diagonal) = call.join().unwrap();
        assert_eq!(vals, expected_vals);
        assert_eq!(
            diagonal,
            expected_full
                .d
                .materialize(engine)
                .unwrap()
                .dense_data()
                .unwrap()
        );
    }
    assert!(Arc::ptr_eq(owned(source), &parent_body));
    assert!(Arc::ptr_eq(&owned(source).data, &parent_data));
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };

    // Positive control: the probe sees an implicit materialization.
    let _ = lazy.materialized_tensor_uncached(engine).unwrap();
}

#[test]
fn eig_dense_lazy_nonnormal_is_logical_owned_repeatable_and_cold() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 2)]).unwrap();
    let source =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, indices| {
            match (indices[0], indices[1]) {
                (0, 0) => 1.0_f64,
                (1, 1) => 2.0,
                (0, 1) => 1.0,
                _ => 0.0,
            }
        })
        .unwrap();
    let logical = eager_adjoint_oracle(&source);
    let Eig { d, v } = source
        .adjoint(engine)
        .unwrap()
        .eig_full(engine, &[0], &[1])
        .unwrap();
    let lhs = logical
        .convert::<Complex64>(engine)
        .unwrap()
        .compose(engine, &v)
        .unwrap();
    let rhs = v.compose(engine, &d).unwrap();
    assert_typed_map_close(engine, &lhs, &rhs, 1.0e-12);

    assert_eig_uses_a_cold_logical_copy(engine, &source);
}

#[test]
fn eig_dense_lazy_real_order_signed_zero_and_defective_cases_match_logical_oracles() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let scalar_leg = GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0), 1)]).unwrap();
    let negative =
        TensorMap::from_subblock_fn(engine, &runtime, [&scalar_leg], [&scalar_leg], |_, _| -2.0)
            .unwrap();
    let lazy = negative.adjoint(engine).unwrap();
    let value = lazy.eig_vals(engine, &[0], &[1]).unwrap()[0].values[0];
    assert_eq!(value, num_complex::Complex64::new(-2.0, 0.0));
    assert_eq!(value.im.to_bits(), 0.0f64.to_bits());

    let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 2)]).unwrap();
    let rotation =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, indices| {
            match (indices[0], indices[1]) {
                (0, 1) => -1.0,
                (1, 0) => 1.0,
                _ => 0.0,
            }
        })
        .unwrap();
    let eager = eager_adjoint_oracle(&rotation);
    let expected = eager.eig_vals(engine, &[0], &[1]).unwrap();
    let lazy = rotation.adjoint(engine).unwrap();
    assert_eq!(lazy.eig_vals(engine, &[0], &[1]).unwrap(), expected);
    // Lexicographic (re, im): the conjugate pair's `-i` precedes `+i`.
    assert_eq!(expected[0].values[0].im, -1.0);
    assert_eq!(expected[0].values[1].im, 1.0);

    for epsilon in [0.0, 1.0e-12] {
        let jordan = TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, indices| {
            match (indices[0], indices[1]) {
                (0, 0) | (1, 1) => 1.0,
                (0, 1) => 1.0,
                (1, 0) => epsilon,
                _ => unreachable!(),
            }
        })
        .unwrap();
        let eager = eager_adjoint_oracle(&jordan);
        let lazy = jordan.adjoint(engine).unwrap();
        assert_eq!(
            lazy.eig_vals(engine, &[0], &[1]).unwrap(),
            eager.eig_vals(engine, &[0], &[1]).unwrap()
        );
        let actual = lazy.eig_full(engine, &[0], &[1]).unwrap();
        let expected = eager.eig_full(engine, &[0], &[1]).unwrap();
        assert_eq!(
            actual.d.materialize(engine).unwrap().dense_data().unwrap(),
            expected
                .d
                .materialize(engine)
                .unwrap()
                .dense_data()
                .unwrap()
        );
        assert_eq!(
            actual.v.dense_data().unwrap(),
            expected.v.dense_data().unwrap()
        );
    }
}

#[test]
fn eig_dense_lazy_failures_match_logical_oracle_and_stay_cold() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let left = GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0), 2)]).unwrap();
    let right = GradedSpace::try_new(provider, [(U1Irrep::new(0), 3)]).unwrap();
    let source = TensorMap::from_subblock_fn(engine, &runtime, [&left], [&right], |_, indices| {
        (indices[0] + indices[1]) as f64
    })
    .unwrap();
    let eager = eager_adjoint_oracle(&source);
    let expected = [
        eager.eig_vals(engine, &[0], &[1]).unwrap_err().to_string(),
        eager.eig_full(engine, &[0], &[1]).unwrap_err().to_string(),
    ];
    let lazy = source.adjoint(engine).unwrap();
    for _ in 0..2 {
        assert_eq!(
            lazy.eig_vals(engine, &[0], &[1]).unwrap_err().to_string(),
            expected[0]
        );
        assert_eq!(
            lazy.eig_full(engine, &[0], &[1]).unwrap_err().to_string(),
            expected[1]
        );
    }
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}

#[test]
fn exp_of_a_near_hermitian_adjoint_uses_the_logical_orientation_and_stays_cold() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: the fixed approximate-Hermitian dispatch must see logical A^H,
    // whose lower triangle is the conjugated parent upper triangle. A
    // parent-exp redirect feeds EIGH the other triangle and changes values.
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 2)]).unwrap();
    let delta = 4.0e-15;
    let parent =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, indices| {
            match (indices[0], indices[1]) {
                (0, 0) => num_complex::Complex64::new(0.25, 0.0),
                (1, 1) => num_complex::Complex64::new(-0.5, 0.0),
                (0, 1) => num_complex::Complex64::new(delta, 0.0),
                _ => num_complex::Complex64::new(0.0, 0.0),
            }
        })
        .unwrap();
    let eager = eager_adjoint_oracle(&parent);
    let expected = eager.exp(engine, &[0], &[1]).unwrap();
    let parent_redirect = parent
        .exp(engine, &[0], &[1])
        .unwrap()
        .adjoint(engine)
        .unwrap()
        .materialized_tensor_uncached(engine)
        .unwrap();
    let lazy = parent.adjoint(engine).unwrap();
    let actual = lazy.exp(engine, &[0], &[1]).unwrap();

    assert_typed_map_close(engine, &actual, &expected, 1.0e-20);
    assert!(parent_redirect
        .dense_data()
        .unwrap()
        .iter()
        .zip(expected.dense_data().unwrap())
        .any(|(&actual, &expected)| {
            (actual.widen_complex() - expected.widen_complex()).norm() > 1.0e-16
        }));
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}

fn assert_exp_uses_a_cold_logical_copy<R, D>(engine: &mut dyn Engine, source: &TensorMap<R, D>)
where
    R: TypedSectorAdmission<
            Sector = <R as SectorCodec>::Sector,
            Error = FusionAlgebraError,
            Mode = MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send
        + Sync
        + 'static,
    D: AdvancedLinalgScalar + core::fmt::Debug + Send + Sync + 'static,
{
    let eager = eager_adjoint_oracle(source);
    let expected = eager
        .exp(engine, &codomain_axes(&eager), &domain_axes(&eager))
        .unwrap();
    let parent_body = Arc::clone(owned(source));
    let parent_data = Arc::clone(&parent_body.data);
    let lazy = source.adjoint(engine).unwrap();

    for _ in 0..2 {
        let actual = lazy
            .clone()
            .exp(engine, &codomain_axes(&lazy), &domain_axes(&lazy))
            .unwrap();
        assert_typed_map_close(engine, &actual, &expected, 1.0e-9);
        assert!(actual.owned_body().is_some());
        assert!(Arc::ptr_eq(
            actual.logical_space().provider_arc(),
            source.logical_space().provider_arc()
        ));
        assert!(!Arc::ptr_eq(&owned(&actual).data, &parent_data));
        let _ = actual.dense_data().unwrap();
    }
    let calls = (0..4)
        .map(|_| {
            let clone = lazy.clone();
            std::thread::spawn(move || {
                let engine = &mut symtn_dense::CpuEngine::serial();
                clone
                    .exp(engine, &codomain_axes(&clone), &domain_axes(&clone))
                    .unwrap()
            })
        })
        .collect::<Vec<_>>();
    for call in calls {
        assert_typed_map_close(engine, &call.join().unwrap(), &expected, 1.0e-9);
    }
    assert!(Arc::ptr_eq(owned(source), &parent_body));
    assert!(Arc::ptr_eq(&owned(source).data, &parent_data));
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}

#[test]
fn exp_uses_owned_provider_native_outputs_without_warming_lazy_receivers() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: real/complex non-self-dual U(1) and a genuine SU(2) multitree
    // remain deterministic across repeats, clones, and concurrent calls.
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(
        provider,
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 3),
            (U1Irrep::new(2), 1),
        ],
    )
    .unwrap();
    let u1 = TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, indices| {
        if indices[0] == indices[1] {
            0.25 + indices[0] as f64 / 10.0
        } else {
            (indices[0] + 2 * indices[1] + 1) as f64 / 100.0
        }
    })
    .unwrap();
    assert_exp_uses_a_cold_logical_copy(engine, &u1);
    assert_exp_uses_a_cold_logical_copy(engine, &genuinely_complex(&u1));

    let provider = Arc::new(SU2FusionRule);
    let half = GradedSpace::try_new(provider, [(SU2Irrep::from_twice_spin(1), 1)]).unwrap();
    let su2 = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&half, &half, &half],
        [&half, &half, &half],
        |_, indices| (indices.iter().sum::<usize>() + 1) as f64 / 20.0,
    )
    .unwrap();
    assert!(su2.logical_space().space().structure().block_count() > 1);
    assert_exp_uses_a_cold_logical_copy(engine, &genuinely_complex(&su2));
}

#[test]
fn exp_failure_leaves_the_lazy_receiver_and_parent_untouched() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 2)]).unwrap();
    let source = TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, indices| {
        if indices == [0, 1] {
            num_complex::Complex64::new(f64::NAN, 0.0)
        } else {
            num_complex::Complex64::new((indices[0] + indices[1] + 1) as f64, 0.0)
        }
    })
    .unwrap();
    let before = source.dense_data().unwrap().to_vec();
    let parent = Arc::clone(owned(&source));
    let data = Arc::clone(&parent.data);
    let lazy = source.adjoint(engine).unwrap();

    assert!(matches!(
        lazy.exp(engine, &[0], &[1]),
        Err(Error::Operation(_))
    ));
    assert!(source
        .dense_data()
        .unwrap()
        .iter()
        .zip(&before)
        .all(|(actual, expected)| {
            actual.re.to_bits() == expected.re.to_bits()
                && actual.im.to_bits() == expected.im.to_bits()
        }));
    assert!(Arc::ptr_eq(owned(&source), &parent));
    assert!(Arc::ptr_eq(&owned(&source).data, &data));
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}

fn assert_inverse_redirect<R, D>(engine: &mut dyn Engine, source: &TensorMap<R, D>)
where
    R: TypedSectorAdmission<
            Sector = <R as SectorCodec>::Sector,
            Error = FusionAlgebraError,
            Mode = MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send
        + Sync
        + 'static,
    D: AdvancedLinalgScalar + core::fmt::Debug + Send + Sync + 'static,
{
    let eager = eager_adjoint_oracle(source);
    let expected = eager
        .inv(engine, &codomain_axes(&eager), &domain_axes(&eager))
        .unwrap();
    let parent_body = Arc::clone(owned(source));
    let parent_data = Arc::clone(&parent_body.data);
    let lazy = source.adjoint(engine).unwrap();

    for _ in 0..2 {
        let actual = lazy
            .clone()
            .inv(engine, &codomain_axes(&lazy), &domain_axes(&lazy))
            .unwrap();
        assert_typed_map_close(engine, &actual, &expected, 1e-10);
        let codomain = eager.codomain();
        let domain = eager.domain();
        let eager_compose = &eager.compose(engine, &actual).unwrap();
        let operand =
            &TensorMap::isomorphism(engine, source.runtime(), codomain.iter(), codomain.iter())
                .unwrap();
        assert_typed_map_close(engine, eager_compose, operand, 1e-9);
        let actual_compose = &actual.compose(engine, &eager).unwrap();
        let operand =
            &TensorMap::isomorphism(engine, source.runtime(), domain.iter(), domain.iter())
                .unwrap();
        assert_typed_map_close(engine, actual_compose, operand, 1e-9);
        assert!(actual.owned_body().is_some());
        assert!(Arc::ptr_eq(
            actual.logical_space().provider_arc(),
            source.logical_space().provider_arc()
        ));
        assert!(!Arc::ptr_eq(&owned(&actual).data, &parent_data));
        let _ = actual.dense_data().unwrap();
    }

    let calls = (0..4)
        .map(|_| {
            let clone = lazy.clone();
            std::thread::spawn(move || {
                let engine = &mut symtn_dense::CpuEngine::serial();
                clone
                    .inv(engine, &codomain_axes(&clone), &domain_axes(&clone))
                    .unwrap()
            })
        })
        .collect::<Vec<_>>();
    for call in calls {
        assert_typed_map_close(engine, &call.join().unwrap(), &expected, 1e-10);
    }

    assert!(Arc::ptr_eq(owned(source), &parent_body));
    assert!(Arc::ptr_eq(&owned(source).data, &parent_data));
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}

#[test]
fn inverse_redirect_is_owned_provider_native_repeatable_and_cold() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: U(1) complex blocks and a genuine SU(2) multitree use the
    // inverse identity without materializing the lazy receiver, including
    // cloned and concurrent calls, and return detached provider-native data.
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(
        provider,
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 3),
            (U1Irrep::new(2), 1),
        ],
    )
    .unwrap();
    let u1 = TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, indices| {
        let re = if indices[0] == indices[1] {
            20.0 + indices[0] as f64
        } else {
            (indices[0] + 2 * indices[1] + 1) as f64 / 100.0
        };
        num_complex::Complex64::new(re, (indices[0] + indices[1] + 1) as f64 / 200.0)
    })
    .unwrap();
    let identity =
        TensorMap::<_, num_complex::Complex64>::isomorphism(engine, &runtime, [&leg], [&leg])
            .unwrap();
    let u1 = u1
        .axpby(
            engine,
            num_complex::Complex64::new(1.0, 0.0),
            &identity,
            num_complex::Complex64::new(100.0, 0.0),
        )
        .unwrap();
    assert_inverse_redirect(engine, &u1);

    let provider = Arc::new(U1FusionRule);
    let wide = GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0), 4)]).unwrap();
    let narrow = GradedSpace::try_new(provider, [(U1Irrep::new(0), 2)]).unwrap();
    let unequal = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&wide],
        [&narrow, &narrow],
        |_, indices| {
            let column = 2 * indices[1] + indices[2];
            if indices[0] == column {
                10.0 + indices[0] as f64
            } else {
                (indices[0] + column + 1) as f64 / 100.0
            }
        },
    )
    .unwrap();
    assert_ne!(unequal.codomain_rank(), unequal.domain_rank());
    assert_inverse_redirect(engine, &unequal);

    let provider = Arc::new(SU2FusionRule);
    let half = GradedSpace::try_new(provider, [(SU2Irrep::from_twice_spin(1), 1)]).unwrap();
    let su2 = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&half, &half, &half],
        [&half, &half, &half],
        |_, indices| {
            if indices[..3] == indices[3..] {
                20.0 + indices.iter().sum::<usize>() as f64
            } else {
                (indices.iter().sum::<usize>() + 1) as f64 / 100.0
            }
        },
    )
    .unwrap();
    let identity = TensorMap::<_, f64>::isomorphism(
        engine,
        &runtime,
        [&half, &half, &half],
        [&half, &half, &half],
    )
    .unwrap();
    let su2 = su2.axpby(engine, 1.0, &identity, 100.0).unwrap();
    assert!(su2.logical_space().space().structure().block_count() > 1);
    assert_inverse_redirect(engine, &su2);
}

#[test]
fn inverse_redirect_failure_leaves_the_receiver_cold() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: a singular solve changes neither parent Arc/bytes nor the lazy
    // receiver.
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 3)]).unwrap();
    let source = TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, indices| {
        if indices[0] == indices[1] {
            4.0 + indices[0] as f64
        } else {
            (indices[0] + indices[1] + 1) as f64 / 20.0
        }
    })
    .unwrap();
    let singular = source.scale(engine, 0.0).unwrap();
    let before = singular.dense_data().unwrap().to_vec();
    let body = Arc::clone(owned(&singular));
    let data = Arc::clone(&body.data);
    let cold = singular.adjoint(engine).unwrap();
    assert!(matches!(
        cold.inv(engine, &[0], &[1]),
        Err(Error::Operation(_))
    ));
    assert_eq!(singular.dense_data().unwrap(), before);
    assert!(Arc::ptr_eq(owned(&singular), &body));
    assert!(Arc::ptr_eq(&owned(&singular).data, &data));
    let TypedTensorRepr::Adjoint(_) = &cold.repr else {
        unreachable!()
    };

    // The first U(1) sector solves before the second singular sector
    // fails, pinning atomicity after partial backend progress.
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 1), (U1Irrep::new(1), 2)]).unwrap();
    let late = TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |trees, indices| {
        if trees.codomain_uncoupled[0] == U1Irrep::new(0) && indices[0] == indices[1] {
            2.0
        } else {
            0.0
        }
    })
    .unwrap();
    let before = late.dense_data().unwrap().to_vec();
    let data = Arc::clone(&owned(&late).data);
    let cold = late.adjoint(engine).unwrap();
    assert!(matches!(
        cold.inv(engine, &[0], &[1]),
        Err(Error::Operation(_))
    ));
    assert_eq!(late.dense_data().unwrap(), before);
    assert!(Arc::ptr_eq(&owned(&late).data, &data));
    let TypedTensorRepr::Adjoint(_) = &cold.repr else {
        unreachable!()
    };
}

#[test]
fn solve_is_transactional_provider_native_and_cache_cold() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let densified = probe::record();
    let runtime = Runtime::builder().build().unwrap();
    let lhs_provider = Arc::new(U1FusionRule);
    let rhs_provider = Arc::new(U1FusionRule);
    let lhs_leg = GradedSpace::try_new(
        Arc::clone(&lhs_provider),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let rhs_codomain = GradedSpace::try_new(
        Arc::clone(&rhs_provider),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let rhs_domain =
        GradedSpace::try_new(rhs_provider, [(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)]).unwrap();
    let divisor =
        TensorMap::from_subblock_fn(engine, &runtime, [&lhs_leg], [&lhs_leg], |_, indices| {
            if indices[0] == indices[1] {
                4.0 + indices[0] as f64
            } else {
                0.25
            }
        })
        .unwrap();
    let rhs = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&rhs_codomain],
        [&rhs_domain],
        |_, indices| (indices[0] + 2 * indices[1] + 1) as f64,
    )
    .unwrap();

    let solution = divisor.solve(engine, &[0], &[1], &rhs, &[0], &[1]).unwrap();
    let divisor_compose = &divisor.compose(engine, &solution).unwrap();
    assert_typed_map_close(engine, divisor_compose, &rhs, 1e-11);
    assert!(Arc::ptr_eq(
        solution.logical_space().provider_arc(),
        &lhs_provider
    ));

    let complex_divisor = divisor.convert::<Complex64>(engine).unwrap();
    let complex_rhs = rhs
        .convert::<Complex64>(engine)
        .unwrap()
        .scale(engine, num_complex::Complex64::new(1.0, 0.25))
        .unwrap();
    let complex_solution = complex_divisor
        .solve(engine, &[0], &[1], &complex_rhs, &[0], &[1])
        .unwrap();
    let complex_divisor_compose = &complex_divisor.compose(engine, &complex_solution).unwrap();
    assert_typed_map_close(engine, complex_divisor_compose, &complex_rhs, 1e-11);
    assert!(Arc::ptr_eq(
        complex_solution.logical_space().provider_arc(),
        &lhs_provider
    ));

    let lazy = divisor.adjoint(engine).unwrap();
    let expected = eager_adjoint_oracle(&divisor)
        .solve(engine, &[0], &[1], &rhs, &[0], &[1])
        .unwrap();
    let lazy_solve = &lazy.solve(engine, &[0], &[1], &rhs, &[0], &[1]).unwrap();
    assert_typed_map_close(engine, lazy_solve, &expected, 1e-11);

    let square_rhs = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&rhs_codomain],
        [&rhs_codomain],
        |_, indices| (2 * indices[0] + indices[1] + 1) as f64,
    )
    .unwrap();
    let lazy_rhs = square_rhs.adjoint(engine).unwrap();
    let expected = divisor
        .solve(
            engine,
            &[0],
            &[1],
            &eager_adjoint_oracle(&square_rhs),
            &[0],
            &[1],
        )
        .unwrap();
    let divisor_solve = &divisor
        .solve(engine, &[0], &[1], &lazy_rhs, &[0], &[1])
        .unwrap();
    assert_typed_map_close(engine, divisor_solve, &expected, 1e-11);

    let bad_leg = GradedSpace::try_new(Arc::clone(&lhs_provider), [(U1Irrep::new(7), 1)]).unwrap();
    let bad = TensorMap::from_subblock_fn(engine, &runtime, [&bad_leg], [&bad_leg], |_, _| 1.0)
        .unwrap()
        .adjoint(engine)
        .unwrap();
    assert!(matches!(
        lazy.solve(engine, &[0], &[1], &bad, &[0], &[1]),
        Err(Error::Operation(error))
            if matches!(*error, symtn_tensors::OperationError::SpaceMismatch { .. })
    ));

    let singular_dense = divisor.scale(engine, 0.0).unwrap().adjoint(engine).unwrap();
    assert!(matches!(
        singular_dense.solve(engine, &[0], &[1], &lazy_rhs, &[0], &[1]),
        Err(Error::Operation(_))
    ));

    let compact_rhs = TensorMap::diagonal(
        engine,
        &runtime,
        &lhs_leg,
        [
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![2.0, 3.0],
            },
            SectorSpectrum {
                sector: U1Irrep::new(1),
                values: vec![5.0],
            },
        ],
    )
    .unwrap();
    let compact_solution = divisor
        .solve(engine, &[0], &[1], &compact_rhs, &[0], &[1])
        .unwrap();
    // A dense divisor densifies the compact RHS into its solve buffer once.
    assert_eq!(densified.get(Event::DiagonalMaterialization), 1);
    let densified = probe::record();

    let compact_divisor = TensorMap::diagonal(
        engine,
        &runtime,
        &lhs_leg,
        [
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![2.0, 4.0],
            },
            SectorSpectrum {
                sector: U1Irrep::new(1),
                values: vec![8.0],
            },
        ],
    )
    .unwrap();
    let compact_compact = compact_divisor
        .solve(engine, &[0], &[1], &compact_rhs, &[0], &[1])
        .unwrap();
    assert!(compact_compact.spectrum().is_some());
    assert_eq!(compact_compact.spectrum().unwrap()[0].values, [1.0, 0.75]);
    assert_eq!(compact_compact.spectrum().unwrap()[1].values, [0.625]);
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    let divisor_compose = &divisor.compose(engine, &compact_solution).unwrap();
    assert_typed_map_close(engine, divisor_compose, &compact_rhs, 1e-11);

    let scaled = compact_divisor
        .solve(engine, &[0], &[1], &rhs, &[0], &[1])
        .unwrap();
    let compact_divisor_compose = &compact_divisor.compose(engine, &scaled).unwrap();
    assert_typed_map_close(engine, compact_divisor_compose, &rhs, 1e-11);
    assert!(Arc::ptr_eq(
        scaled.logical_space().provider_arc(),
        compact_divisor.logical_space().provider_arc()
    ));
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);

    let singular = TensorMap::diagonal(
        engine,
        &runtime,
        &lhs_leg,
        [
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![2.0, 0.0],
            },
            SectorSpectrum {
                sector: U1Irrep::new(1),
                values: vec![8.0],
            },
        ],
    )
    .unwrap();
    assert!(matches!(
        singular.solve(engine, &[0], &[1], &rhs, &[0], &[1]),
        Err(Error::Operation(error))
            if matches!(*error, symtn_tensors::OperationError::Dense(
                symtn_dense::DenseError::NumericalFailure { op: "solve_into", .. }
            ))
    ));
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
}

#[cfg(feature = "cgc-gen")]
fn assert_checked_generic_solve_acceptance<D>(engine: &mut dyn Engine)
where
    D: AdvancedLinalgScalar + core::fmt::Debug,
{
    use symtn_core::LieGroupFusionRule;

    let calls = Arc::new(SpyCounts::default());
    let _dense = probe::inject_dense(Box::new(solve_spy(&calls, None)));
    let runtime = Runtime::builder().build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 2)]).unwrap();
    let divisor: TensorMap<_, D> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg], [&leg, &leg], |trees, ij| {
            let row = ij[0] + 2 * ij[1];
            let col = ij[2] + 2 * ij[3];
            D::from_real(if row == col {
                7.0 + trees.codomain_vertices()[0].get() as f64
            } else {
                0.125 * (1 + trees.domain_vertices()[0].get()) as f64
            })
        })
        .unwrap();
    let rhs: TensorMap<_, D> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg], [&leg, &leg], |trees, ij| {
            D::from_real(
                (1 + ij.iter().sum::<usize>() + 3 * trees.domain_vertices()[0].get()) as f64,
            )
        })
        .unwrap();
    assert!((0..divisor.subblock_count()).any(|i| {
        divisor
            .subblock_fusion_trees(i)
            .unwrap()
            .codomain_vertices()[0]
            .get()
            == 2
    }));

    for (lazy_lhs, lazy_rhs) in [(false, false), (true, false), (false, true), (true, true)] {
        let lhs = if lazy_lhs {
            divisor.adjoint(engine).unwrap()
        } else {
            divisor.clone()
        };
        let right = if lazy_rhs {
            rhs.adjoint(engine).unwrap()
        } else {
            rhs.clone()
        };
        let lhs_before = divisor.dense_data().unwrap().to_vec();
        let rhs_before = rhs.dense_data().unwrap().to_vec();
        calls.reset();
        let solution = lhs
            .solve(engine, &[0, 1], &[2, 3], &right, &[0, 1], &[2, 3])
            .unwrap();
        assert!(matches!(solution.repr, TypedTensorRepr::Owned(_)));
        assert_eq!(
            calls.total(),
            5,
            "the SU(3) μ=2 fixture has five nonempty coupled-sector solve routes"
        );
        let lhs_oracle = lhs.materialized_tensor_uncached(engine).unwrap();
        let rhs_oracle = right.materialized_tensor_uncached(engine).unwrap();
        let reconstructed = lhs_oracle.compose(engine, &solution).unwrap();
        assert!(reconstructed
            .dense_data()
            .unwrap()
            .iter()
            .zip(rhs_oracle.dense_data().unwrap())
            .all(|(&actual, &expected)| {
                (actual.widen_complex() - expected.widen_complex()).norm() < 2e-10
            }));
        for i in 0..solution.subblock_count() {
            assert_eq!(
                reconstructed.subblock_fusion_trees(i).unwrap(),
                rhs_oracle.subblock_fusion_trees(i).unwrap(),
            );
        }
        assert_eq!(divisor.dense_data().unwrap(), lhs_before.as_slice());
        assert_eq!(rhs.dense_data().unwrap(), rhs_before.as_slice());
    }
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_generic_solves_are_owned_uncached_and_one_call_per_route() {
    let engine = &mut crate::typed::CpuEngine::serial();
    // What: the solve keeps all four lazy input pairs cold on
    // nondegenerate real and complex cross-multiplicity matrices.
    assert_checked_generic_solve_acceptance::<f64>(engine);
    assert_checked_generic_solve_acceptance::<num_complex::Complex64>(engine);
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_generic_left_solve_preserves_injected_backend_provenance() {
    let engine = &mut crate::typed::CpuEngine::serial();
    use symtn_core::LieGroupFusionRule;

    let calls = Arc::new(SpyCounts::default());
    let _dense = probe::inject_dense(Box::new(solve_spy(
        &calls,
        Some("injected checked solve failure"),
    )));
    let runtime = Runtime::builder().build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 1)]).unwrap();
    let lhs: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, ij| {
            f64::from(ij[0] == ij[1])
        })
        .unwrap();
    let rhs = lhs.scale(engine, 2.0).unwrap();
    let before_lhs = lhs.dense_data().unwrap().to_vec();
    let before_rhs = rhs.dense_data().unwrap().to_vec();
    assert!(matches!(
        lhs.solve(engine, &[0], &[1], &rhs, &[0], &[1]),
        Err(GenericTensorError::Facade(Error::Operation(error)))
            if matches!(*error, symtn_tensors::OperationError::Dense(
                DenseError::Backend { op: "solve_into", ref message, .. }
            ) if message == "injected checked solve failure")
    ));
    assert_eq!(calls.total(), 1);
    assert_eq!(lhs.dense_data().unwrap(), before_lhs.as_slice());
    assert_eq!(rhs.dense_data().unwrap(), before_rhs.as_slice());
}

fn assert_pinv_redirect<R, D>(
    engine: &mut dyn Engine,
    source: &TensorMap<R, D>,
    rcond: f64,
    exact_original: bool,
) where
    R: TypedSectorAdmission<
            Sector = <R as SectorCodec>::Sector,
            Error = FusionAlgebraError,
            Mode = MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec
        + Send
        + Sync
        + 'static,
    D: AdvancedLinalgScalar + core::fmt::Debug + Send + Sync + 'static,
{
    let eager = eager_adjoint_oracle(source);
    let expected = eager
        .pinv(engine, &codomain_axes(&eager), &domain_axes(&eager), rcond)
        .unwrap();
    let parent_body = Arc::clone(owned(source));
    let parent_data = Arc::clone(&parent_body.data);
    let lazy = source.adjoint(engine).unwrap();

    for _ in 0..2 {
        let actual = lazy
            .clone()
            .pinv(engine, &codomain_axes(&lazy), &domain_axes(&lazy), rcond)
            .unwrap();
        assert_typed_map_close(engine, &actual, &expected, 1e-9);
        let pap = actual
            .compose(engine, &eager)
            .unwrap()
            .compose(engine, &actual)
            .unwrap();
        assert_typed_map_close(engine, &pap, &actual, 1e-8);
        assert!(is_hermitian!(
            engine,
            eager.compose(engine, &actual).unwrap(),
            1e-9
        ));
        assert!(is_hermitian!(
            engine,
            actual.compose(engine, &eager).unwrap(),
            1e-9
        ));
        if exact_original {
            let apa = eager
                .compose(engine, &actual)
                .unwrap()
                .compose(engine, &eager)
                .unwrap();
            assert_typed_map_close(engine, &apa, &eager, 1e-8);
        }
        assert!(actual.owned_body().is_some());
        assert!(Arc::ptr_eq(
            actual.logical_space().provider_arc(),
            source.logical_space().provider_arc()
        ));
        assert!(!Arc::ptr_eq(&owned(&actual).data, &parent_data));
        let _ = actual.dense_data().unwrap();
    }

    let calls = (0..4)
        .map(|_| {
            let clone = lazy.clone();
            std::thread::spawn(move || {
                let engine = &mut symtn_dense::CpuEngine::serial();
                clone
                    .pinv(engine, &codomain_axes(&clone), &domain_axes(&clone), rcond)
                    .unwrap()
            })
        })
        .collect::<Vec<_>>();
    for call in calls {
        assert_typed_map_close(engine, &call.join().unwrap(), &expected, 1e-9);
    }
    assert!(Arc::ptr_eq(owned(source), &parent_body));
    assert!(Arc::ptr_eq(&owned(source).data, &parent_data));
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}

#[test]
fn pinv_redirect_preserves_semantics_ownership_and_cold_concurrency() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: full-rank, rectangular/rank-deficient, non-self-dual U(1),
    // complex data, empty support, and a genuine SU(2) multitree all use
    // the parent-factor seam and return detached provider-native outputs.
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg =
        GradedSpace::try_new(provider, [(U1Irrep::new(-1), 2), (U1Irrep::new(0), 3)]).unwrap();
    let full = TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, indices| {
        let re = if indices[0] == indices[1] {
            30.0 + indices[0] as f64
        } else {
            (indices[0] + indices[1] + 1) as f64 / 100.0
        };
        num_complex::Complex64::new(re, (indices[0] + 2 * indices[1] + 1) as f64 / 200.0)
    })
    .unwrap();
    assert_pinv_redirect(engine, &full, 1e-12, true);
    let operand = &genuinely_complex(&u1_matrix_fixture(engine, [(0, 3), (1, 2)], [(0, 2)]));

    assert_pinv_redirect(engine, operand, 1e-10, false);
    let operand = &u1_matrix_fixture(engine, [(1, 2)], [(0, 3)]);
    assert_pinv_redirect(engine, operand, 1e-10, false);
    let su2 = genuinely_complex(&su2_lazy_fixture(engine));
    assert!(su2.logical_space().space().structure().block_count() > 1);
    assert_pinv_redirect(engine, &su2, 1e-10, false);
}

#[test]
fn pinv_redirect_late_svd_failure_keeps_parent_and_receiver_cold() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: a successful first-sector SVD cannot publish factors, mutate
    // parent bytes, or initialize the lazy receiver when sector two fails.
    let _dense = probe::inject_dense(Box::new(fail_second_svd(&Arc::default())));
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 2), (U1Irrep::new(1), 2)]).unwrap();
    let source = TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, indices| {
        (indices.iter().sum::<usize>() + 1) as f64
    })
    .unwrap();
    let before = source.dense_data().unwrap().to_vec();
    let data = Arc::clone(&owned(&source).data);
    let lazy = source.adjoint(engine).unwrap();
    assert!(matches!(
        lazy.pinv(engine, &[0], &[1], 0.0),
        Err(Error::Operation(_))
    ));
    assert_eq!(source.dense_data().unwrap(), before);
    assert!(Arc::ptr_eq(&owned(&source).data, &data));
    let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
        unreachable!()
    };
}
