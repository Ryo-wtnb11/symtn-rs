use super::*;

#[test]
fn checked_generic_compose_and_inv_powers_match_explicit_real_and_complex_oracles() {
    // What: integer powers are compositions (`inv` first for a negative
    // exponent); the oracles are hand-computed 2x2 matrix powers.
    fn power<D: symtn::typed::AdvancedLinalgScalar>(
        tensor: &TensorMap<CheckedOnlyToy, D>,
        exponent: i32,
    ) -> TensorMap<CheckedOnlyToy, D> {
        let base = if exponent < 0 {
            tensor.inv(&[0], &[1]).unwrap()
        } else {
            tensor.clone()
        };
        let mut result = base.clone();
        for _ in 1..exponent.unsigned_abs() {
            result = result.compose(&base).unwrap();
        }
        result
    }
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let real: TensorMap<_, f64> = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, ij| {
        [[2.0, 1.0], [3.0, 4.0]][ij[0]][ij[1]]
    })
    .unwrap();
    let complex = real
        .convert::<Complex64>()
        .scale(Complex64::new(1.0, 1.0))
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
fn assert_sun_checked_generic_inverse_outer_multiplicity<D>(n: usize, adjoint: Vec<i64>)
where
    D: symtn::typed::AdvancedLinalgScalar + fmt::Debug + PartialEq + numerics::Numeric,
{
    use symtn::sector::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(n).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(adjoint.clone(), 2)]).unwrap();
    let source: TensorMap<_, D> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, ij| {
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
        TensorMap::isomorphism(&runtime, [&leg, &leg], [&leg, &leg]).unwrap();
    let terms = endomorphism_terms(source.dense_data().unwrap().len());
    let inverse = source.inv(&[0, 1], &[2, 3]).unwrap();
    assert!(std::ptr::eq(inverse.provider(), provider.as_ref()));
    assert!(symtn::typed::__network::runtime_identity(source.runtime()).matches(inverse.runtime()));
    assert_eq!(inverse.codomain(), source.domain());
    assert_eq!(inverse.domain(), source.codomain());
    for product in [
        source.compose(&inverse).unwrap(),
        inverse.compose(&source).unwrap(),
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
    for (n, adjoint) in [(3, vec![1, 1]), (4, vec![1, 0, 1])] {
        assert_sun_checked_generic_inverse_outer_multiplicity::<f64>(n, adjoint.clone());
        assert_sun_checked_generic_inverse_outer_multiplicity::<Complex64>(n, adjoint);
    }
}

#[cfg(feature = "cgc-gen")]
fn assert_sun_checked_generic_left_solve(n: usize, label: Vec<i64>) {
    use symtn::sector::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(n).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(label, 2)]).unwrap();
    let divisor: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, indices| {
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
        })
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
    let rhs: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, indices| {
            (indices.iter().sum::<usize>()
                + 1
                + 3 * trees.codomain_vertices()[0].get()
                + 5 * trees.domain_vertices()[0].get()) as f64
        })
        .unwrap();
    let route_swapped_rhs: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, indices| {
            (indices.iter().sum::<usize>()
                + 1
                + 3 * trees.domain_vertices()[0].get()
                + 5 * trees.codomain_vertices()[0].get()) as f64
        })
        .unwrap();

    let solution = divisor
        .solve(&[0, 1], &[2, 3], &rhs, &[0, 1], &[2, 3])
        .unwrap();
    assert!(std::ptr::eq(solution.provider(), provider.as_ref()));
    let reconstructed = divisor.compose(&solution).unwrap();
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
        .solve(&[1, 0], &[3, 2], &rhs, &[1, 0], &[3, 2])
        .unwrap();
    let permuted_divisor = divisor.permute(&[1, 0], &[3, 2]).unwrap();
    let permuted_rhs = rhs.permute(&[1, 0], &[3, 2]).unwrap();
    let composed = permuted_divisor
        .solve(&[0, 1], &[2, 3], &permuted_rhs, &[0, 1], &[2, 3])
        .unwrap();
    assert_eq!(moved.dense_data().unwrap(), composed.dense_data().unwrap());

    let complex_divisor = divisor.convert::<Complex64>();
    let complex_rhs = rhs
        .convert::<Complex64>()
        .scale(Complex64::new(1.0, 0.25))
        .unwrap();
    let complex_solution = complex_divisor
        .solve(&[0, 1], &[2, 3], &complex_rhs, &[0, 1], &[2, 3])
        .unwrap();
    let complex_reconstructed = complex_divisor.compose(&complex_solution).unwrap();
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
        .solve(&[1, 0], &[3, 2], &complex_rhs, &[1, 0], &[3, 2])
        .unwrap();
    let permuted_divisor = complex_divisor.permute(&[1, 0], &[3, 2]).unwrap();
    let permuted_rhs = complex_rhs.permute(&[1, 0], &[3, 2]).unwrap();
    let composed = permuted_divisor
        .solve(&[0, 1], &[2, 3], &permuted_rhs, &[0, 1], &[2, 3])
        .unwrap();
    assert_eq!(moved.dense_data().unwrap(), composed.dense_data().unwrap());
}

#[cfg(feature = "cgc-gen")]
#[test]
fn sun_checked_generic_left_solve_preserves_outer_multiplicity() {
    for (n, label) in [(3, vec![1, 1]), (4, vec![1, 0, 1])] {
        assert_sun_checked_generic_left_solve(n, label);
    }
}

#[cfg(feature = "cgc-gen")]
#[test]
fn sun_checked_generic_inv_preflight_counts_outer_multiplicity() {
    use symtn::sector::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
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
    let inverse = accepted.inv(&[0, 1], &[2]).unwrap();
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
        &runtime,
        [&codomain_leg, &codomain_leg],
        [&nonisomorphic_domain],
        |_, _| 1.0,
    )
    .unwrap();
    let before = rejected.dense_data().unwrap().to_vec();
    assert!(matches!(
        rejected.inv(&[0, 1], &[2]),
        Err(GenericTensorError::Facade(symtn::typed::Error::Operation(
            _
        )))
    ));
    assert_eq!(rejected.dense_data().unwrap(), before.as_slice());
}

#[test]
fn checked_generic_exp_uses_general_pade_for_nonhermitian_dense_blocks() {
    let _cache = cache_shared();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, ij| f64::from(ij == [0, 1]))
            .unwrap();
    let expected = [1.0, 0.0, 1.0, 1.0];
    reset_provider_queries(&provider);
    let direct = source.exp(&[0], &[1]).unwrap();
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
    let lazy = source.adjoint().unwrap();
    let lazy_exp = lazy.exp(&[0], &[1]).unwrap();
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
        .convert::<Complex64>()
        .scale(Complex64::new(1.0, 0.25))
        .unwrap();
    reset_provider_queries(&provider);
    let complex_exp = complex.exp(&[0], &[1]).unwrap();
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
    let _cache = cache_shared();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let wide = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let narrow = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&wide], [&narrow], |_, _| 1.0).unwrap();
    let before = source.dense_data().unwrap().to_vec();
    reset_provider_queries(&provider);
    assert!(matches!(
        source.exp(&[0], &[1]),
        Err(GenericTensorError::Facade(symtn::typed::Error::Operation(
            _
        )))
    ));
    assert_no_provider_queries(&provider);
    assert_eq!(source.dense_data().unwrap(), before.as_slice());
}

#[test]
fn checked_generic_exp_rejects_early_and_late_nonfinite_sectors_without_publication() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 1), (Label::X, 1)]).unwrap();
    for target in [Label::Vacuum, Label::X] {
        let source: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |trees, _| {
                if trees.coupled() == &target {
                    f64::NAN
                } else {
                    0.0
                }
            })
            .unwrap();
        let before = source.dense_data().unwrap().to_vec();
        assert!(matches!(
            source.exp(&[0], &[1]),
            Err(GenericTensorError::Facade(symtn::typed::Error::Operation(
                _
            )))
        ));
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
fn assert_sun_checked_generic_exp_outer_multiplicity(n: usize, adjoint: Vec<i64>) {
    use symtn::sector::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(n).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(adjoint.clone(), 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, ij| {
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
    let real_output = source.exp(&[0, 1], &[2, 3]).unwrap();
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
    let real_inverse = source.scale(-1.0).unwrap().exp(&[0, 1], &[2, 3]).unwrap();
    let real_identity: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, _| {
            f64::from(trees.codomain_vertices() == trees.domain_vertices())
        })
        .unwrap();
    for product in [
        real_output.compose(&real_inverse).unwrap(),
        real_inverse.compose(&real_output).unwrap(),
    ] {
        assert!(product
            .dense_data()
            .unwrap()
            .iter()
            .zip(real_identity.dense_data().unwrap())
            .all(|(a, b)| (*a - *b).abs() < 2e-10));
    }
    let input = source
        .convert::<Complex64>()
        .scale(Complex64::new(1.0, 0.2))
        .unwrap();
    let output = input.exp(&[0, 1], &[2, 3]).unwrap();
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
        .scale(Complex64::new(-1.0, 0.0))
        .unwrap()
        .exp(&[0, 1], &[2, 3])
        .unwrap();
    let identity: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, _| {
            Complex64::new(
                f64::from(trees.codomain_vertices() == trees.domain_vertices()),
                0.0,
            )
        })
        .unwrap();
    for product in [
        output.compose(&inverse).unwrap(),
        inverse.compose(&output).unwrap(),
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
    assert_sun_checked_generic_exp_outer_multiplicity(3, vec![1, 1]);
    assert_sun_checked_generic_exp_outer_multiplicity(4, vec![1, 0, 1]);
}

#[test]
fn checked_generic_inv_isomorphism_preflight_failure_is_typed_and_nonpublishing() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    // Unequal sides, so the isomorphism is a provider question: equal sides
    // are isomorphic without one (#1995).
    let dual = leg.try_dual().unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&dual], |_, _| 2.0).unwrap();
    let before = source.dense_data().unwrap().to_vec();
    provider.fail_algebra.store(true, Ordering::Relaxed);
    assert!(matches!(
        source.inv(&[0], &[1]),
        Err(GenericTensorError::Plan(
            symtn::typed::CheckedGenericPlanError::Provider(ToyError::Algebra)
        ))
    ));
    assert_eq!(source.dense_data().unwrap(), before.as_slice());
}

#[test]
fn checked_generic_inv_destination_admission_failure_is_typed_and_nonpublishing() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, _| 2.0).unwrap();
    let before = source.dense_data().unwrap().to_vec();
    provider.invalid_style.store(true, Ordering::Relaxed);
    assert!(matches!(
        source.inv(&[0], &[1]),
        Err(GenericTensorError::Structure(_))
    ));
    assert_eq!(source.dense_data().unwrap(), before.as_slice());
}

#[test]
fn checked_generic_inv_accepts_unequal_isomorphic_spaces_and_rejects_nonisomorphic() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let x = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let unit = GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&x], [&x, &unit], |_, indices| {
            if indices[0] == indices[1] {
                2.0
            } else {
                0.0
            }
        })
        .unwrap();
    let inverse = source.inv(&[0], &[1, 2]).unwrap();
    assert_eq!((inverse.codomain_rank(), inverse.domain_rank()), (2, 1));
    assert_eq!(inverse.codomain(), source.domain());
    assert_eq!(inverse.domain(), source.codomain());
    assert!(std::ptr::eq(inverse.provider(), provider.as_ref()));
    assert!(symtn::typed::__network::runtime_identity(source.runtime()).matches(inverse.runtime()));
    // Hand oracle: `source` is `2·1`, so both products are `source / 2`.
    let terms = endomorphism_terms(source.dense_data().unwrap().len());
    numerics::assert_slices_close(
        "source ∘ inverse",
        source.compose(&inverse).unwrap().dense_data().unwrap(),
        source.scale(0.5).unwrap().dense_data().unwrap(),
        terms,
    );
    numerics::assert_slices_close(
        "inverse ∘ source",
        inverse.compose(&source).unwrap().dense_data().unwrap(),
        source.scale(0.5).unwrap().dense_data().unwrap(),
        terms,
    );

    let narrow = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let nonisomorphic: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&narrow], [&x], |_, _| 1.0).unwrap();
    let before = nonisomorphic.dense_data().unwrap().to_vec();
    assert!(matches!(
        nonisomorphic.inv(&[0], &[1]),
        Err(GenericTensorError::Facade(symtn::typed::Error::Operation(
            _
        )))
    ));
    assert_eq!(nonisomorphic.dense_data().unwrap(), before.as_slice());
}

#[test]
fn checked_generic_inv_singular_early_and_late_sectors_preserve_source() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let bond =
        GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 1), (Label::X, 1)]).unwrap();
    for target in [Label::Vacuum, Label::X] {
        let source: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(&runtime, [&bond], [&bond], |trees, _| {
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
        assert!(matches!(
            source.inv(&[0], &[1]),
            Err(GenericTensorError::Facade(symtn::typed::Error::Operation(
                _
            )))
        ));
        assert_eq!(source.dense_data().unwrap(), before.as_slice());
    }
}

#[test]
fn checked_generic_left_solve_accepts_distinct_provider_arcs_and_rectangular_rhs() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let lhs_provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let rhs_provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let lhs_codomain = GradedSpace::try_new(Arc::clone(&lhs_provider), [(Label::X, 2)]).unwrap();
    let lhs_domain_x = GradedSpace::try_new(Arc::clone(&lhs_provider), [(Label::X, 2)]).unwrap();
    let lhs_domain_unit =
        GradedSpace::try_new(Arc::clone(&lhs_provider), [(Label::Vacuum, 1)]).unwrap();
    let rhs_codomain = GradedSpace::try_new(Arc::clone(&rhs_provider), [(Label::X, 2)]).unwrap();
    let rhs_domain = GradedSpace::try_new(rhs_provider, [(Label::X, 3)]).unwrap();
    let divisor: TensorMap<_, f64> = TensorMap::from_subblock_fn(
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
    let rhs: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&rhs_codomain], [&rhs_domain], |_, indices| {
            (indices[0] + 2 * indices[1] + 1) as f64
        })
        .unwrap();

    let solution = divisor.solve(&[0], &[1, 2], &rhs, &[0], &[1]).unwrap();
    assert!(std::ptr::eq(solution.provider(), lhs_provider.as_ref()));
    assert_eq!(solution.codomain(), divisor.domain());
    assert_eq!(solution.domain(), rhs.domain());
    assert!(divisor
        .compose(&solution)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(rhs.dense_data().unwrap())
        .all(|(actual, expected)| (*actual - *expected).abs() < 1e-11));

    let complex_divisor = divisor.convert::<Complex64>();
    let complex_rhs = rhs
        .convert::<Complex64>()
        .scale(Complex64::new(1.0, 0.25))
        .unwrap();
    let complex_solution = complex_divisor
        .solve(&[0], &[1, 2], &complex_rhs, &[0], &[1])
        .unwrap();
    assert!(std::ptr::eq(
        complex_solution.provider(),
        lhs_provider.as_ref()
    ));
    assert!(complex_divisor
        .compose(&complex_solution)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(complex_rhs.dense_data().unwrap())
        .all(|(actual, expected)| (*actual - *expected).norm() < 1e-11));
}

#[test]
fn checked_generic_left_solve_preflight_failures_are_nonpublishing() {
    let _cache = cache_shared();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let x = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let narrow = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let lhs: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&x], [&narrow], |_, _| 1.0).unwrap();
    let rhs: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&x], [&x], |_, _| 1.0).unwrap();
    let before = lhs.dense_data().unwrap().to_vec();
    assert!(matches!(
        lhs.solve(&[0], &[1], &rhs, &[0], &[1]),
        Err(GenericTensorError::Facade(symtn::typed::Error::Operation(
            _
        )))
    ));
    assert_eq!(lhs.dense_data().unwrap(), before.as_slice());

    let other_runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let runtime_rhs: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&other_runtime, [&x], [&x], |_, _| 1.0).unwrap();
    reset_provider_queries(&provider);
    assert!(matches!(
        lhs.solve(&[0], &[1], &runtime_rhs, &[0], &[1]),
        Err(GenericTensorError::Facade(
            symtn::typed::Error::RuntimeMismatch
        ))
    ));
    assert_no_provider_queries(&provider);

    let foreign_provider = Arc::new(CheckedOnlyToy::new_product_probe(1));
    let foreign_x = GradedSpace::try_new(Arc::clone(&foreign_provider), [(Label::X, 2)]).unwrap();
    let foreign_rhs: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&foreign_x], [&foreign_x], |_, _| 1.0).unwrap();
    reset_provider_queries(&provider);
    reset_provider_queries(&foreign_provider);
    assert!(matches!(
        lhs.solve(&[0], &[1], &foreign_rhs, &[0], &[1]),
        Err(GenericTensorError::Facade(
            symtn::typed::Error::RuleMismatch
        ))
    ));
    assert_no_provider_queries(&provider);
    assert_no_provider_queries(&foreign_provider);

    let wrong_codomain = GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 1)]).unwrap();
    let codomain_rhs: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&wrong_codomain], [&x], |_, _| 1.0).unwrap();
    reset_provider_queries(&provider);
    assert!(matches!(
        lhs.solve(&[0], &[1], &codomain_rhs, &[0], &[1]),
        Err(GenericTensorError::Facade(symtn::typed::Error::Operation(error)))
            if matches!(*error, symtn::typed::OperationError::SpaceMismatch { .. })
    ));
    assert_no_provider_queries(&provider);
}

#[test]
fn checked_generic_left_solve_singular_sectors_are_nonpublishing() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let bond =
        GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 1), (Label::X, 1)]).unwrap();
    let rhs: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&bond], [&bond], |_, _| 1.0).unwrap();
    for target in [Label::Vacuum, Label::X] {
        let divisor: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(&runtime, [&bond], [&bond], |trees, _| {
                f64::from(trees.coupled() != &target)
            })
            .unwrap();
        let before = divisor.dense_data().unwrap().to_vec();
        assert!(matches!(
            divisor.solve(&[0], &[1], &rhs, &[0], &[1]),
            Err(GenericTensorError::Facade(symtn::typed::Error::Operation(
                _
            )))
        ));
        assert_eq!(divisor.dense_data().unwrap(), before.as_slice());
    }
}

#[test]
fn checked_generic_left_solve_covers_all_lazy_input_pairs() {
    let _cache = cache_shared();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let divisor: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
            if indices[0] == indices[1] {
                2.0 + indices[0] as f64
            } else {
                0.0
            }
        })
        .unwrap();
    let rhs: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
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
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
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
            divisor.adjoint().unwrap()
        } else {
            divisor.clone()
        };
        let right = if lazy_rhs {
            rhs.adjoint().unwrap()
        } else {
            rhs.clone()
        };
        reset_provider_queries(&provider);
        let solution = lhs.solve(&[0], &[1], &right, &[0], &[1]).unwrap();
        assert!(std::ptr::eq(solution.provider(), provider.as_ref()));
        numerics::assert_slices_close(
            "solve",
            solution.dense_data().unwrap(),
            expected.dense_data().unwrap(),
            2,
        );
        // Includes the one identity query that both admits the root and keys
        // the sector-structure cache (#2030, #2046). The divisor's equal
        // sides are isomorphic without a coupled-dimension query (#1995).
        assert_eq!(provider.queries_since_reset.load(Ordering::Relaxed), 2);
    }
}
