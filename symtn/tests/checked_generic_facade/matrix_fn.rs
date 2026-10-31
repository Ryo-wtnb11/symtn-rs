use super::*;

#[test]
fn checked_generic_compose_and_inv_powers_match_explicit_real_and_complex_oracles() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: integer powers are compositions (`inv` first for a negative
    // exponent); the oracles are hand-computed 2x2 matrix powers.
    fn power<D: symtn::typed::AdvancedLinalgScalar>(
        tensor: &TensorMap<CheckedOnlyToy, D>,
        exponent: i32,
    ) -> TensorMap<CheckedOnlyToy, D> {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let base = if exponent < 0 {
            tensor.inv(engine, &[0], &[1]).unwrap()
        } else {
            tensor.clone()
        };
        let mut result = base.clone();
        for _ in 1..exponent.unsigned_abs() {
            result = result.compose(engine, &base).unwrap();
        }
        result
    }
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let real: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, ij| {
            [[2.0, 1.0], [3.0, 4.0]][ij[0]][ij[1]]
        })
        .unwrap();
    let complex = real
        .convert::<Complex64>(engine)
        .unwrap()
        .scale(engine, Complex64::new(1.0, 1.0))
        .unwrap();
    let real_oracles: &[(i32, &[f64])] = &[
        (1, &[2.0, 3.0, 1.0, 4.0]),
        (2, &[7.0, 18.0, 6.0, 19.0]),
        (3, &[32.0, 93.0, 31.0, 94.0]),
        (-1, &[0.8, -0.6, -0.2, 0.4]),
        (-2, &[0.76, -0.72, -0.24, 0.28]),
    ];
    let complex_oracles: &[(i32, &[Complex64])] = &[
        (
            1,
            &[
                Complex64::new(2.0, 2.0),
                Complex64::new(3.0, 3.0),
                Complex64::new(1.0, 1.0),
                Complex64::new(4.0, 4.0),
            ],
        ),
        (
            2,
            &[
                Complex64::new(0.0, 14.0),
                Complex64::new(0.0, 36.0),
                Complex64::new(0.0, 12.0),
                Complex64::new(0.0, 38.0),
            ],
        ),
        (
            3,
            &[
                Complex64::new(-64.0, 64.0),
                Complex64::new(-186.0, 186.0),
                Complex64::new(-62.0, 62.0),
                Complex64::new(-188.0, 188.0),
            ],
        ),
        (
            -1,
            &[
                Complex64::new(0.4, -0.4),
                Complex64::new(-0.3, 0.3),
                Complex64::new(-0.1, 0.1),
                Complex64::new(0.2, -0.2),
            ],
        ),
        (
            -2,
            &[
                Complex64::new(0.0, -0.38),
                Complex64::new(0.0, 0.36),
                Complex64::new(0.0, 0.12),
                Complex64::new(0.0, -0.14),
            ],
        ),
    ];
    for &(exponent, expected) in real_oracles {
        assert!(power(&real, exponent)
            .dense_data()
            .unwrap()
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (actual - expected).abs() < 1e-12));
    }
    for &(exponent, expected) in complex_oracles {
        assert!(power(&complex, exponent)
            .dense_data()
            .unwrap()
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (*actual - *expected).norm() < 1e-12));
    }
}

#[cfg(feature = "cgc-gen")]
fn assert_sun_checked_generic_inverse_outer_multiplicity<D>(
    engine: &mut dyn Engine,
    n: usize,
    adjoint: Vec<i64>,
) where
    D: symtn::typed::AdvancedLinalgScalar + fmt::Debug + PartialEq + numerics::Numeric,
{
    use symtn::sector::LieGroupFusionRule;

    let runtime = Runtime::builder().build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(n).unwrap()).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(adjoint.clone(), 2)]).unwrap();
    let source: TensorMap<_, D> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg], [&leg, &leg], |trees, ij| {
            let row = ij[0] + 2 * ij[1];
            let col = ij[2] + 2 * ij[3];
            D::from_real(
                if trees.codomain_vertices() == trees.domain_vertices() && row == col {
                    2.0
                } else if trees.codomain_vertices()[0].get() == 1
                    && trees.domain_vertices()[0].get() == 2
                    && row == col
                {
                    1.0
                } else {
                    0.0
                },
            )
        })
        .unwrap();
    assert!((0..source.subblock_count()).any(|index| {
        let trees = source.subblock_fusion_trees(index).unwrap();
        trees.codomain_vertices()[0].get() == 2 || trees.domain_vertices()[0].get() == 2
    }));
    assert!((0..source.subblock_count()).any(|index| {
        let trees = source.subblock_fusion_trees(index).unwrap();
        trees.codomain_vertices()[0].get() == 1
            && trees.domain_vertices()[0].get() == 2
            && source.dense_data().unwrap()[source.subblock(index).unwrap().offset()]
                == D::from_real(1.0)
    }));

    let identity: TensorMap<_, D> =
        TensorMap::isomorphism(engine, &runtime, [&leg, &leg], [&leg, &leg]).unwrap();
    let terms = endomorphism_terms(source.dense_data().unwrap().len());
    let inverse = source.inv(engine, &[0, 1], &[2, 3]).unwrap();
    assert!(std::ptr::eq(inverse.provider(), provider.as_ref()));
    assert!(symtn::typed::__network::runtime_identity(source.runtime()).matches(inverse.runtime()));
    assert_eq!(inverse.codomain(), source.domain());
    assert_eq!(inverse.domain(), source.codomain());
    for product in [
        source.compose(engine, &inverse).unwrap(),
        inverse.compose(engine, &source).unwrap(),
    ] {
        assert_eq!(product.codomain(), source.codomain());
        numerics::assert_slices_close(
            "inverse against the identity",
            product.dense_data().unwrap(),
            identity.dense_data().unwrap(),
            terms,
        );
    }
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_generic_inv_sun_outer_multiplicity_satisfies_inverse_laws() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    for (n, adjoint) in [(3, vec![1, 1]), (4, vec![1, 0, 1])] {
        assert_sun_checked_generic_inverse_outer_multiplicity::<f64>(engine, n, adjoint.clone());
        assert_sun_checked_generic_inverse_outer_multiplicity::<Complex64>(engine, n, adjoint);
    }
}

#[cfg(feature = "cgc-gen")]
fn assert_sun_checked_generic_left_solve(engine: &mut dyn Engine, n: usize, label: Vec<i64>) {
    use symtn::sector::LieGroupFusionRule;

    let runtime = Runtime::builder().build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(n).unwrap()).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(label, 2)]).unwrap();
    let divisor: TensorMap<_, f64> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&leg, &leg],
        [&leg, &leg],
        |trees, indices| {
            let row = indices[0] + 2 * indices[1];
            let col = indices[2] + 2 * indices[3];
            if row == col {
                7.0 + trees.codomain_vertices()[0].get() as f64
                    + 0.25 * trees.domain_vertices()[0].get() as f64
            } else {
                0.05 * (1
                    + trees.codomain_vertices()[0].get()
                    + 2 * trees.domain_vertices()[0].get()) as f64
            }
        },
    )
    .unwrap();
    assert!((0..divisor.subblock_count()).any(|index| {
        divisor
            .subblock_fusion_trees(index)
            .unwrap()
            .codomain_vertices()
            .iter()
            .chain(
                divisor
                    .subblock_fusion_trees(index)
                    .unwrap()
                    .domain_vertices(),
            )
            .any(|vertex| vertex.get() > 1)
    }));
    let rhs: TensorMap<_, f64> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&leg, &leg],
        [&leg, &leg],
        |trees, indices| {
            (indices.iter().sum::<usize>()
                + 1
                + 3 * trees.codomain_vertices()[0].get()
                + 5 * trees.domain_vertices()[0].get()) as f64
        },
    )
    .unwrap();
    let route_swapped_rhs: TensorMap<_, f64> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&leg, &leg],
        [&leg, &leg],
        |trees, indices| {
            (indices.iter().sum::<usize>()
                + 1
                + 3 * trees.domain_vertices()[0].get()
                + 5 * trees.codomain_vertices()[0].get()) as f64
        },
    )
    .unwrap();

    let solution = divisor
        .solve(engine, &[0, 1], &[2, 3], &rhs, &[0, 1], &[2, 3])
        .unwrap();
    assert!(std::ptr::eq(solution.provider(), provider.as_ref()));
    let reconstructed = divisor.compose(engine, &solution).unwrap();
    for index in 0..rhs.subblock_count() {
        assert_eq!(
            reconstructed.subblock_fusion_trees(index).unwrap(),
            rhs.subblock_fusion_trees(index).unwrap()
        );
        assert_eq!(
            reconstructed.subblock(index).unwrap().shape(),
            rhs.subblock(index).unwrap().shape()
        );
    }
    assert!(reconstructed
        .dense_data()
        .unwrap()
        .iter()
        .zip(rhs.dense_data().unwrap())
        .all(|(actual, expected)| (*actual - *expected).abs() < 2e-10));
    assert!(reconstructed
        .dense_data()
        .unwrap()
        .iter()
        .zip(route_swapped_rhs.dense_data().unwrap())
        .any(|(actual, swapped)| (*actual - *swapped).abs() > 1e-7));

    let moved = divisor
        .solve(engine, &[1, 0], &[3, 2], &rhs, &[1, 0], &[3, 2])
        .unwrap();
    let permuted_divisor = divisor.permute(engine, &[1, 0], &[3, 2]).unwrap();
    let permuted_rhs = rhs.permute(engine, &[1, 0], &[3, 2]).unwrap();
    let composed = permuted_divisor
        .solve(engine, &[0, 1], &[2, 3], &permuted_rhs, &[0, 1], &[2, 3])
        .unwrap();
    assert_eq!(moved.dense_data().unwrap(), composed.dense_data().unwrap());

    let complex_divisor = divisor.convert::<Complex64>(engine).unwrap();
    let complex_rhs = rhs
        .convert::<Complex64>(engine)
        .unwrap()
        .scale(engine, Complex64::new(1.0, 0.25))
        .unwrap();
    let complex_solution = complex_divisor
        .solve(engine, &[0, 1], &[2, 3], &complex_rhs, &[0, 1], &[2, 3])
        .unwrap();
    let complex_reconstructed = complex_divisor.compose(engine, &complex_solution).unwrap();
    for index in 0..complex_rhs.subblock_count() {
        assert_eq!(
            complex_reconstructed.subblock_fusion_trees(index).unwrap(),
            complex_rhs.subblock_fusion_trees(index).unwrap()
        );
        assert_eq!(
            complex_reconstructed.subblock(index).unwrap().shape(),
            complex_rhs.subblock(index).unwrap().shape()
        );
    }
    assert!(complex_reconstructed
        .dense_data()
        .unwrap()
        .iter()
        .zip(complex_rhs.dense_data().unwrap())
        .all(|(actual, expected)| (*actual - *expected).norm() < 2e-10));

    let moved = complex_divisor
        .solve(engine, &[1, 0], &[3, 2], &complex_rhs, &[1, 0], &[3, 2])
        .unwrap();
    let permuted_divisor = complex_divisor.permute(engine, &[1, 0], &[3, 2]).unwrap();
    let permuted_rhs = complex_rhs.permute(engine, &[1, 0], &[3, 2]).unwrap();
    let composed = permuted_divisor
        .solve(engine, &[0, 1], &[2, 3], &permuted_rhs, &[0, 1], &[2, 3])
        .unwrap();
    assert_eq!(moved.dense_data().unwrap(), composed.dense_data().unwrap());
}

#[cfg(feature = "cgc-gen")]
#[test]
fn sun_checked_generic_left_solve_preserves_outer_multiplicity() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    for (n, label) in [(3, vec![1, 1]), (4, vec![1, 0, 1])] {
        assert_sun_checked_generic_left_solve(engine, n, label);
    }
}

#[cfg(feature = "cgc-gen")]
#[test]
fn sun_checked_generic_inv_preflight_counts_outer_multiplicity() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    use symtn::sector::LieGroupFusionRule;

    let runtime = Runtime::builder().build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let adjoint = vec![1, 1];
    let codomain_leg = GradedSpace::try_new(Arc::clone(&provider), [(adjoint.clone(), 1)]).unwrap();
    let isomorphic_domain = GradedSpace::try_new(
        Arc::clone(&provider),
        [
            (vec![0, 0], 1),
            (adjoint.clone(), 2),
            (vec![3, 0], 1),
            (vec![0, 3], 1),
            (vec![2, 2], 1),
        ],
    )
    .unwrap();
    let accepted: TensorMap<_, f64> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&codomain_leg, &codomain_leg],
        [&isomorphic_domain],
        |trees, indices| {
            if trees.coupled() == &adjoint {
                let row = trees.codomain_vertices()[0].get() - 1;
                if row == indices[2] {
                    2.0
                } else {
                    0.0
                }
            } else {
                2.0
            }
        },
    )
    .unwrap();
    let inverse = accepted.inv(engine, &[0, 1], &[2]).unwrap();
    assert_eq!(inverse.codomain(), accepted.domain());
    assert_eq!(inverse.domain(), accepted.codomain());

    let nonisomorphic_domain = GradedSpace::try_new(
        Arc::clone(&provider),
        [
            (vec![0, 0], 1),
            (adjoint, 1),
            (vec![3, 0], 1),
            (vec![0, 3], 1),
            (vec![2, 2], 1),
        ],
    )
    .unwrap();
    let rejected: TensorMap<_, f64> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&codomain_leg, &codomain_leg],
        [&nonisomorphic_domain],
        |_, _| 1.0,
    )
    .unwrap();
    let before = rejected.dense_data().unwrap().to_vec();
    assert!(
        matches!(rejected.inv(engine, &[0, 1], &[2]), Err(GenericTensorError::Facade(symtn::typed::Error::Operation(e))) if matches!(*e, symtn::typed::OperationError::SpaceMismatch { message } if message.contains("inv requires isomorphic")))
    );
    assert_eq!(rejected.dense_data().unwrap(), before.as_slice());
}

#[test]
fn checked_generic_exp_uses_general_pade_for_nonhermitian_dense_blocks() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, ij| {
            f64::from(ij == [0, 1])
        })
        .unwrap();
    let expected = [1.0, 0.0, 1.0, 1.0];
    reset_provider_queries(&provider);
    let direct = source.exp(engine, &[0], &[1]).unwrap();
    assert_no_provider_queries(&provider);
    assert!(std::ptr::eq(direct.provider(), provider.as_ref()));
    assert!(symtn::typed::__network::runtime_identity(direct.runtime()).matches(source.runtime()));
    assert_eq!(direct.codomain(), source.codomain());
    assert_eq!(direct.domain(), source.domain());
    assert_eq!(direct.subblock_count(), source.subblock_count());
    assert!(direct
        .dense_data()
        .unwrap()
        .iter()
        .zip(expected)
        .all(|(a, b)| (*a - b).abs() < 1e-12));
    let lazy = source.adjoint(engine).unwrap();
    let lazy_exp = lazy.exp(engine, &[0], &[1]).unwrap();
    assert!(std::ptr::eq(lazy_exp.provider(), provider.as_ref()));
    assert!(
        symtn::typed::__network::runtime_identity(lazy_exp.runtime()).matches(source.runtime())
    );
    assert_eq!(lazy_exp.codomain(), lazy.codomain());
    assert_eq!(lazy_exp.domain(), lazy.domain());
    assert_eq!(lazy_exp.subblock_count(), lazy.subblock_count());
    // Hand oracle: exp(Nᵀ) = 1 + Nᵀ for the nilpotent N above.
    numerics::assert_slices_close(
        "exp of the lazy adjoint",
        lazy_exp.dense_data().unwrap(),
        &[1.0, 1.0, 0.0, 1.0],
        2,
    );
    let complex = source
        .convert::<Complex64>(engine)
        .unwrap()
        .scale(engine, Complex64::new(1.0, 0.25))
        .unwrap();
    reset_provider_queries(&provider);
    let complex_exp = complex.exp(engine, &[0], &[1]).unwrap();
    assert_no_provider_queries(&provider);
    assert!(std::ptr::eq(complex_exp.provider(), provider.as_ref()));
    assert!(
        symtn::typed::__network::runtime_identity(complex_exp.runtime()).matches(source.runtime())
    );
    assert_eq!(complex_exp.codomain(), complex.codomain());
    assert_eq!(complex_exp.domain(), complex.domain());
    assert!(complex_exp
        .dense_data()
        .unwrap()
        .iter()
        .zip([
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.25),
            Complex64::new(1.0, 0.0)
        ])
        .all(|(a, b)| (*a - b).norm() < 1e-12));
}

#[test]
fn checked_generic_exp_rejects_nonendomorphism_before_provider_work() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let wide = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let narrow = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&wide], [&narrow], |_, _| 1.0).unwrap();
    let before = source.dense_data().unwrap().to_vec();
    reset_provider_queries(&provider);
    assert!(
        matches!(source.exp(engine, &[0], &[1]), Err(GenericTensorError::Facade(symtn::typed::Error::Operation(e))) if matches!(*e, symtn::typed::OperationError::SpaceMismatch { message } if message.contains("exp requires an endomorphism")))
    );
    assert_no_provider_queries(&provider);
    assert_eq!(source.dense_data().unwrap(), before.as_slice());
}

#[test]
fn checked_generic_exp_rejects_early_and_late_nonfinite_sectors_without_publication() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 1), (Label::X, 1)]).unwrap();
    for target in [Label::Vacuum, Label::X] {
        let source: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |trees, _| {
                if trees.coupled() == &target {
                    f64::NAN
                } else {
                    0.0
                }
            })
            .unwrap();
        let before = source.dense_data().unwrap().to_vec();
        assert!(
            matches!(source.exp(engine, &[0], &[1]), Err(GenericTensorError::Facade(symtn::typed::Error::Operation(e))) if matches!(*e, symtn::typed::OperationError::InvalidArgument { message } if message.contains("finite coupled-sector blocks")))
        );
        assert_eq!(source.dense_data().unwrap().len(), before.len());
        assert!(source
            .dense_data()
            .unwrap()
            .iter()
            .zip(&before)
            .all(|(a, b)| a.to_bits() == b.to_bits()));
    }
}

#[cfg(feature = "cgc-gen")]
fn assert_sun_checked_generic_exp_outer_multiplicity(
    engine: &mut dyn Engine,
    n: usize,
    adjoint: Vec<i64>,
) {
    use symtn::sector::LieGroupFusionRule;

    let runtime = Runtime::builder().build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(n).unwrap()).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(adjoint.clone(), 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg], [&leg, &leg], |trees, ij| {
            let left = trees.codomain_vertices()[0].get();
            let right = trees.domain_vertices()[0].get();
            if ij.iter().all(|&index| index == 0) {
                if left == right {
                    left as f64 / 5.0
                } else if left == 1 && right == 2 {
                    0.3
                } else {
                    0.0
                }
            } else {
                0.0
            }
        })
        .unwrap();
    assert!(
        (0..source.subblock_count()).any(|index| source
            .subblock_fusion_trees(index)
            .unwrap()
            .codomain_vertices()[0]
            .get()
            > 1),
        "[adj, adj] -> adj must retain its outer-multiplicity key"
    );
    let real_output = source.exp(engine, &[0, 1], &[2, 3]).unwrap();
    assert!(std::ptr::eq(real_output.provider(), provider.as_ref()));
    assert!(
        symtn::typed::__network::runtime_identity(real_output.runtime()).matches(source.runtime())
    );
    assert_eq!(real_output.codomain(), source.codomain());
    assert_eq!(real_output.domain(), source.domain());
    for index in 0..source.subblock_count() {
        assert_eq!(
            real_output.subblock_fusion_trees(index).unwrap(),
            source.subblock_fusion_trees(index).unwrap()
        );
        assert_eq!(
            real_output.subblock(index).unwrap(),
            source.subblock(index).unwrap()
        );
    }
    let real_inverse = source
        .scale(engine, -1.0)
        .unwrap()
        .exp(engine, &[0, 1], &[2, 3])
        .unwrap();
    let real_identity: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg], [&leg, &leg], |trees, _| {
            f64::from(trees.codomain_vertices() == trees.domain_vertices())
        })
        .unwrap();
    for product in [
        real_output.compose(engine, &real_inverse).unwrap(),
        real_inverse.compose(engine, &real_output).unwrap(),
    ] {
        assert!(product
            .dense_data()
            .unwrap()
            .iter()
            .zip(real_identity.dense_data().unwrap())
            .all(|(a, b)| (*a - *b).abs() < 2e-10));
    }
    let input = source
        .convert::<Complex64>(engine)
        .unwrap()
        .scale(engine, Complex64::new(1.0, 0.2))
        .unwrap();
    let output = input.exp(engine, &[0, 1], &[2, 3]).unwrap();
    assert!(std::ptr::eq(output.provider(), provider.as_ref()));
    assert!(symtn::typed::__network::runtime_identity(output.runtime()).matches(source.runtime()));
    assert_eq!(output.codomain(), input.codomain());
    assert_eq!(output.domain(), input.domain());
    assert_eq!(output.subblock_count(), input.subblock_count());
    for index in 0..input.subblock_count() {
        assert_eq!(
            output.subblock_fusion_trees(index).unwrap(),
            input.subblock_fusion_trees(index).unwrap()
        );
        assert_eq!(
            output.subblock(index).unwrap(),
            input.subblock(index).unwrap()
        );
    }
    let inverse = input
        .scale(engine, Complex64::new(-1.0, 0.0))
        .unwrap()
        .exp(engine, &[0, 1], &[2, 3])
        .unwrap();
    let identity: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg], [&leg, &leg], |trees, _| {
            Complex64::new(
                f64::from(trees.codomain_vertices() == trees.domain_vertices()),
                0.0,
            )
        })
        .unwrap();
    for product in [
        output.compose(engine, &inverse).unwrap(),
        inverse.compose(engine, &output).unwrap(),
    ] {
        assert!(product
            .dense_data()
            .unwrap()
            .iter()
            .zip(identity.dense_data().unwrap())
            .all(|(a, b)| (*a - *b).norm() < 2e-10));
    }
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_generic_exp_sun_outer_multiplicity_preserves_layout() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    assert_sun_checked_generic_exp_outer_multiplicity(engine, 3, vec![1, 1]);
    assert_sun_checked_generic_exp_outer_multiplicity(engine, 4, vec![1, 0, 1]);
}

#[test]
fn checked_generic_inv_isomorphism_preflight_failure_is_typed_and_nonpublishing() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    // Unequal sides, so the isomorphism is a provider question: equal sides
    // are isomorphic without one.
    let dual = leg.try_dual().unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&dual], |_, _| 2.0).unwrap();
    let before = source.dense_data().unwrap().to_vec();
    provider.fail_algebra.store(true, Ordering::Relaxed);
    assert!(matches!(
        source.inv(engine, &[0], &[1]),
        Err(GenericTensorError::Plan(
            symtn::typed::CheckedGenericPlanError::Provider(ToyError::Algebra)
        ))
    ));
    assert_eq!(source.dense_data().unwrap(), before.as_slice());
}

#[test]
fn checked_generic_inv_destination_admission_failure_is_typed_and_nonpublishing() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, _| 2.0).unwrap();
    let before = source.dense_data().unwrap().to_vec();
    provider.invalid_style.store(true, Ordering::Relaxed);
    assert!(matches!(
        source.inv(engine, &[0], &[1]),
        Err(GenericTensorError::Structure(
            CheckedGenericStructureError::Core(symtn_core::CoreError::UnsupportedFusionStyle {
                actual: symtn::sector::FusionStyleKind::Unique,
                ..
            })
        ))
    ));
    assert_eq!(source.dense_data().unwrap(), before.as_slice());
}

#[test]
fn checked_generic_inv_accepts_unequal_isomorphic_spaces_and_rejects_nonisomorphic() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let x = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let unit = GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&x], [&x, &unit], |_, indices| {
            if indices[0] == indices[1] {
                2.0
            } else {
                0.0
            }
        })
        .unwrap();
    let inverse = source.inv(engine, &[0], &[1, 2]).unwrap();
    assert_eq!((inverse.codomain_rank(), inverse.domain_rank()), (2, 1));
    assert_eq!(inverse.codomain(), source.domain());
    assert_eq!(inverse.domain(), source.codomain());
    assert!(std::ptr::eq(inverse.provider(), provider.as_ref()));
    assert!(symtn::typed::__network::runtime_identity(source.runtime()).matches(inverse.runtime()));
    // Hand oracle: `source` is `2·1`, so both products are `source / 2`.
    let terms = endomorphism_terms(source.dense_data().unwrap().len());
    numerics::assert_slices_close(
        "source ∘ inverse",
        source
            .compose(engine, &inverse)
            .unwrap()
            .dense_data()
            .unwrap(),
        source.scale(engine, 0.5).unwrap().dense_data().unwrap(),
        terms,
    );
    numerics::assert_slices_close(
        "inverse ∘ source",
        inverse
            .compose(engine, &source)
            .unwrap()
            .dense_data()
            .unwrap(),
        source.scale(engine, 0.5).unwrap().dense_data().unwrap(),
        terms,
    );

    let narrow = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let nonisomorphic: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&narrow], [&x], |_, _| 1.0).unwrap();
    let before = nonisomorphic.dense_data().unwrap().to_vec();
    assert!(
        matches!(nonisomorphic.inv(engine, &[0], &[1]), Err(GenericTensorError::Facade(symtn::typed::Error::Operation(e))) if matches!(*e, symtn::typed::OperationError::SpaceMismatch { message } if message.contains("inv requires isomorphic")))
    );
    assert_eq!(nonisomorphic.dense_data().unwrap(), before.as_slice());
}

#[test]
fn checked_generic_inv_singular_early_and_late_sectors_preserve_source() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let bond =
        GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 1), (Label::X, 1)]).unwrap();
    for target in [Label::Vacuum, Label::X] {
        let source: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&bond], [&bond], |trees, _| {
                if trees.coupled() == &target {
                    0.0
                } else {
                    1.0
                }
            })
            .unwrap();
        let labels = (0..source.subblock_count())
            .map(|index| *source.subblock_fusion_trees(index).unwrap().coupled())
            .collect::<Vec<_>>();
        assert_eq!(labels, [Label::Vacuum, Label::X]);
        let before = source.dense_data().unwrap().to_vec();
        assert!(
            matches!(source.inv(engine, &[0], &[1]), Err(GenericTensorError::Facade(symtn::typed::Error::Operation(e))) if matches!(*e, symtn::typed::OperationError::Dense(symtn_dense::DenseError::NumericalFailure { op: "solve_into", .. })))
        );
        assert_eq!(source.dense_data().unwrap(), before.as_slice());
    }
}

#[test]
fn checked_generic_left_solve_accepts_distinct_provider_arcs_and_rectangular_rhs() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let lhs_provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let rhs_provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let lhs_codomain = GradedSpace::try_new(Arc::clone(&lhs_provider), [(Label::X, 2)]).unwrap();
    let lhs_domain_x = GradedSpace::try_new(Arc::clone(&lhs_provider), [(Label::X, 2)]).unwrap();
    let lhs_domain_unit =
        GradedSpace::try_new(Arc::clone(&lhs_provider), [(Label::Vacuum, 1)]).unwrap();
    let rhs_codomain = GradedSpace::try_new(Arc::clone(&rhs_provider), [(Label::X, 2)]).unwrap();
    let rhs_domain = GradedSpace::try_new(rhs_provider, [(Label::X, 3)]).unwrap();
    let divisor: TensorMap<_, f64> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&lhs_codomain],
        [&lhs_domain_x, &lhs_domain_unit],
        |_, indices| {
            if indices[0] == indices[1] {
                2.0 + indices[0] as f64
            } else {
                0.0
            }
        },
    )
    .unwrap();
    let rhs: TensorMap<_, f64> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&rhs_codomain],
        [&rhs_domain],
        |_, indices| (indices[0] + 2 * indices[1] + 1) as f64,
    )
    .unwrap();

    let solution = divisor
        .solve(engine, &[0], &[1, 2], &rhs, &[0], &[1])
        .unwrap();
    assert!(std::ptr::eq(solution.provider(), lhs_provider.as_ref()));
    assert_eq!(solution.codomain(), divisor.domain());
    assert_eq!(solution.domain(), rhs.domain());
    assert!(divisor
        .compose(engine, &solution)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(rhs.dense_data().unwrap())
        .all(|(actual, expected)| (*actual - *expected).abs() < 1e-11));

    let complex_divisor = divisor.convert::<Complex64>(engine).unwrap();
    let complex_rhs = rhs
        .convert::<Complex64>(engine)
        .unwrap()
        .scale(engine, Complex64::new(1.0, 0.25))
        .unwrap();
    let complex_solution = complex_divisor
        .solve(engine, &[0], &[1, 2], &complex_rhs, &[0], &[1])
        .unwrap();
    assert!(std::ptr::eq(
        complex_solution.provider(),
        lhs_provider.as_ref()
    ));
    assert!(complex_divisor
        .compose(engine, &complex_solution)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(complex_rhs.dense_data().unwrap())
        .all(|(actual, expected)| (*actual - *expected).norm() < 1e-11));
}

#[test]
fn checked_generic_left_solve_preflight_failures_are_nonpublishing() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let x = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let narrow = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let lhs: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&x], [&narrow], |_, _| 1.0).unwrap();
    let rhs: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&x], [&x], |_, _| 1.0).unwrap();
    let before = lhs.dense_data().unwrap().to_vec();
    assert!(
        matches!(lhs.solve(engine, &[0], &[1], &rhs, &[0], &[1]), Err(GenericTensorError::Facade(symtn::typed::Error::Operation(e))) if matches!(*e, symtn::typed::OperationError::SpaceMismatch { message } if message.contains("isomorphic divisor")))
    );
    assert_eq!(lhs.dense_data().unwrap(), before.as_slice());

    let other_runtime = Runtime::builder().build().unwrap();
    let runtime_rhs: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &other_runtime, [&x], [&x], |_, _| 1.0).unwrap();
    reset_provider_queries(&provider);
    assert!(matches!(
        lhs.solve(engine, &[0], &[1], &runtime_rhs, &[0], &[1]),
        Err(GenericTensorError::Facade(
            symtn::typed::Error::RuntimeMismatch
        ))
    ));
    assert_no_provider_queries(&provider);

    let foreign_provider = Arc::new(CheckedOnlyToy::new_product_probe(1));
    let foreign_x = GradedSpace::try_new(Arc::clone(&foreign_provider), [(Label::X, 2)]).unwrap();
    let foreign_rhs: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&foreign_x], [&foreign_x], |_, _| 1.0)
            .unwrap();
    reset_provider_queries(&provider);
    reset_provider_queries(&foreign_provider);
    assert!(matches!(
        lhs.solve(engine, &[0], &[1], &foreign_rhs, &[0], &[1]),
        Err(GenericTensorError::Facade(
            symtn::typed::Error::RuleMismatch
        ))
    ));
    assert_no_provider_queries(&provider);
    assert_no_provider_queries(&foreign_provider);

    let wrong_codomain = GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 1)]).unwrap();
    let codomain_rhs: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&wrong_codomain], [&x], |_, _| 1.0).unwrap();
    reset_provider_queries(&provider);
    assert!(matches!(
        lhs.solve(engine, &[0], &[1], &codomain_rhs, &[0], &[1]),
        Err(GenericTensorError::Facade(symtn::typed::Error::Operation(error)))
            if matches!(*error, symtn::typed::OperationError::SpaceMismatch { .. })
    ));
    assert_no_provider_queries(&provider);
}

#[test]
fn checked_generic_left_solve_singular_sectors_are_nonpublishing() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let bond =
        GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 1), (Label::X, 1)]).unwrap();
    let rhs: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&bond], [&bond], |_, _| 1.0).unwrap();
    for target in [Label::Vacuum, Label::X] {
        let divisor: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&bond], [&bond], |trees, _| {
                f64::from(trees.coupled() != &target)
            })
            .unwrap();
        let before = divisor.dense_data().unwrap().to_vec();
        assert!(
            matches!(divisor.solve(engine, &[0], &[1], &rhs, &[0], &[1]), Err(GenericTensorError::Facade(symtn::typed::Error::Operation(e))) if matches!(*e, symtn::typed::OperationError::Dense(symtn_dense::DenseError::NumericalFailure { op: "solve_into", .. })))
        );
        assert_eq!(divisor.dense_data().unwrap(), before.as_slice());
    }
}

#[test]
fn checked_generic_left_solve_covers_all_lazy_input_pairs() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let _cache = cache_shared();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let divisor: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, indices| {
            if indices[0] == indices[1] {
                2.0 + indices[0] as f64
            } else {
                0.0
            }
        })
        .unwrap();
    let rhs: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, indices| {
            if indices[0] == indices[1] {
                2.0 + indices[0] as f64
            } else {
                1.0
            }
        })
        .unwrap();
    // Hand oracle: the divisor is diagonal and both fixtures are symmetric,
    // so every lazy/eager combination solves to `rhs[i][j] / (2 + i)`.
    let expected: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, indices| {
            let rhs = if indices[0] == indices[1] {
                2.0 + indices[0] as f64
            } else {
                1.0
            };
            rhs / (2.0 + indices[0] as f64)
        })
        .unwrap();
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
        reset_provider_queries(&provider);
        let solution = lhs.solve(engine, &[0], &[1], &right, &[0], &[1]).unwrap();
        assert!(std::ptr::eq(solution.provider(), provider.as_ref()));
        numerics::assert_slices_close(
            "solve",
            solution.dense_data().unwrap(),
            expected.dense_data().unwrap(),
            2,
        );
        // Includes the one identity query that both admits the root and keys
        // the sector-structure cache. The divisor's equal
        // sides are isomorphic without a coupled-dimension query.
        assert_eq!(provider.queries_since_reset.load(Ordering::Relaxed), 2);
    }
}

#[cfg(feature = "cgc-gen")]
#[test]
fn sun_checked_generic_map_diagonal_keeps_svd_bond_and_principal_branch() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // Checked compact SVD publishes `s` as a compact diagonal on its bond.
    use symtn::sector::LieGroupFusionRule;

    let runtime = Runtime::builder().build().unwrap();
    for n in [3, 4] {
        let provider = Arc::new(
            LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(n).unwrap()).unwrap(),
        );
        let label = if n == 3 { vec![1, 1] } else { vec![1, 0, 1] };
        let leg = GradedSpace::try_new(Arc::clone(&provider), [(label, 1)]).unwrap();
        let source: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg], [&leg], |trees, _| {
                trees.codomain_vertices()[0].get() as f64 + 1.0
            })
            .unwrap();
        let Svd { s, .. } = source.svd_compact(engine, &[0, 1], &[2]).unwrap();
        assert!(s.dense_data().is_err());
        let root = s.map_diagonal(engine, f64::sqrt).unwrap();
        assert!(std::ptr::eq(root.provider(), s.provider()));
        assert_eq!(root.codomain(), s.codomain());
        assert_eq!(root.domain(), s.domain());
        assert!(symtn::typed::__network::runtime_identity(root.runtime()).matches(s.runtime()));
        let square = root.compose(engine, &root).unwrap();
        assert!(square.dense_data().is_err(), "D * D stays compact");
        assert!(square
            .materialize(engine)
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .zip(s.materialize(engine).unwrap().dense_data().unwrap())
            .all(|(actual, expected)| (actual - expected).abs() < 1.0e-10));

        let negative: TensorMap<_, Complex64> = TensorMap::diagonal(
            engine,
            &runtime,
            &leg,
            leg.sectors()
                .unwrap()
                .into_iter()
                .map(|sector| SectorSpectrum {
                    sector,
                    values: vec![Complex64::new(-1.0, 0.0)],
                }),
        )
        .unwrap();
        let principal = negative.map_diagonal(engine, |value| value.sqrt()).unwrap();
        assert!(principal
            .diagview(engine)
            .unwrap()
            .iter()
            .flat_map(|entry| &entry.values)
            .all(|value| *value == Complex64::new(0.0, 1.0)));
    }
}

#[test]
fn checked_generic_map_diagonal_rejects_dense_before_queries_and_preserves_source() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: dense storage is refused before any provider query, whether or not
    // it is bond shaped, and the source is untouched.
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let bond_leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let non_bond: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg], [&leg], |_, _| 2.0).unwrap();
    let dense_diagonal: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&bond_leg], [&bond_leg], |_, ij| {
            if ij[0] == ij[1] {
                4.0
            } else {
                0.0
            }
        })
        .unwrap();
    for tensor in [non_bond, dense_diagonal] {
        let before = tensor.dense_data().unwrap().to_vec();
        reset_provider_queries(&provider);
        match tensor.map_diagonal(engine, f64::sqrt) {
            Err(symtn::typed::Error::InvalidArgument(message)) => {
                assert!(message.contains("compact diagonal"), "{message}");
            }
            other => panic!("expected dense rejection, got {other:?}"),
        }
        assert_no_provider_queries(&provider);
        assert_eq!(tensor.dense_data().unwrap(), before.as_slice());
    }

    let compact: TensorMap<_, f64> = TensorMap::diagonal(
        engine,
        &runtime,
        &bond_leg,
        [SectorSpectrum {
            sector: Label::X,
            values: vec![4.0, 9.0],
        }],
    )
    .unwrap();
    reset_provider_queries(&provider);
    let root = compact.map_diagonal(engine, f64::sqrt).unwrap();
    assert_no_provider_queries(&provider);
    assert_eq!(root.diagview(engine).unwrap()[0].values, [2.0, 3.0]);
}
