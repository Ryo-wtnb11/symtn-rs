use super::*;

#[cfg(feature = "cgc-gen")]
fn assert_sun_checked_generic_eigh<D>(
    n: usize,
    label: Vec<i64>,
    off_diagonal: D,
    adjoint: impl Fn(D) -> D + Copy,
    real: impl Fn(D) -> f64 + Copy,
    norm_squared: f64,
    close: impl Fn(D, D) -> f64 + Copy,
) where
    D: symtn::typed::FactorizationScalar + fmt::Debug,
{
    use std::cell::Cell;

    use symtn::sector::LieGroupFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(n).unwrap()).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(label.clone(), 2)]).unwrap();
    let cross_sector = label.clone();
    let source: TensorMap<_, D> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, index| {
            let row = index[0] + 2 * index[1];
            let column = index[2] + 2 * index[3];
            if row != column {
                return D::from_real(0.0);
            }
            if trees.coupled() == &cross_sector {
                let mu_row = trees.codomain_vertices()[0].get();
                let mu_column = trees.domain_vertices()[0].get();
                match (mu_row, mu_column) {
                    (1, 1) => D::from_real(4.0 + 10.0 * row as f64),
                    (2, 2) => D::from_real(-1.0 + 10.0 * row as f64),
                    (1, 2) => off_diagonal,
                    (2, 1) => adjoint(off_diagonal),
                    _ => D::from_real(0.0),
                }
            } else if trees.codomain_vertices() == trees.domain_vertices() {
                D::from_real(50.0 + 10.0 * row as f64)
            } else {
                D::from_real(0.0)
            }
        })
        .unwrap();
    assert!((0..source.subblock_count()).any(|index| {
        let trees = source.subblock_fusion_trees(index).unwrap();
        trees.coupled() == &label
            && trees.codomain_vertices()[0].get() == 1
            && trees.domain_vertices()[0].get() == 2
            && source.subblock(index).unwrap().shape() == [2, 2, 2, 2]
            && source.dense_data().unwrap()[source.subblock(index).unwrap().offset()]
                == off_diagonal
    }));

    let Eigh { d, v } = source
        .eigh_full(&[0, 1], &[2, 3], HermitianTol::DEFAULT)
        .unwrap();
    assert!(std::ptr::eq(d.provider(), provider.as_ref()));
    assert!(std::ptr::eq(v.provider(), provider.as_ref()));
    assert!(symtn::expert::diagonal_spectrum(&d).unwrap().is_some());
    assert_eq!(v.codomain(), source.codomain());
    assert_eq!(v.domain(), d.codomain());
    for (actual, expected) in source
        .compose(&v)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(v.compose(&d).unwrap().dense_data().unwrap())
    {
        assert!(close(*actual, *expected) < 1e-8);
    }

    let lazy_vh = v.adjoint().unwrap();
    let logical_vh = lazy_vh
        .materialize()
        .unwrap()
        .dense_data()
        .unwrap()
        .to_vec();
    let position = Cell::new(0usize);
    let codomain = v.domain();
    let domain = v.codomain();
    let vh: TensorMap<_, D> =
        TensorMap::from_subblock_fn(&runtime, codomain.iter(), domain.iter(), |_, _| {
            let index = position.get();
            position.set(index + 1);
            logical_vh[index]
        })
        .unwrap();
    assert_same_checked_generic_layout_and_close(
        &v.compose(&d).unwrap().compose(&vh).unwrap(),
        &source,
        close,
    );

    let root = (6.25 + norm_squared).sqrt();
    let lambda_plus = 1.5 + root;
    let lambda_minus = 1.5 - root;
    let identity_codomain = d.codomain();
    let identity_domain = d.domain();
    let identity: TensorMap<_, D> = TensorMap::from_subblock_fn(
        &runtime,
        identity_codomain.iter(),
        identity_domain.iter(),
        |_, index| D::from_real(f64::from(index[0] == index[1])),
    )
    .unwrap();
    let mut selector = identity.clone();
    let mut skipped_target = false;
    for index in 0..d.subblock_count() {
        let block = d.subblock(index).unwrap();
        for diagonal in 0..block.shape()[0] {
            let value = d.materialize().unwrap().dense_data().unwrap()
                [block.offset() + diagonal * block.strides()[0] + diagonal * block.strides()[1]];
            let scalar = real(value);
            if (scalar - lambda_plus).abs() < 1e-8 {
                assert!(!skipped_target, "target eigenvalue must be nondegenerate");
                skipped_target = true;
                continue;
            }
            let shifted = d
                .axpby(D::from_real(1.0), &identity, D::from_real(-scalar))
                .unwrap()
                .scale(D::from_real(1.0 / (lambda_plus - scalar)))
                .unwrap();
            selector = selector.compose(&shifted).unwrap();
        }
    }
    assert!(skipped_target);
    let projector = v.compose(&selector).unwrap().compose(&vh).unwrap();
    let expected: TensorMap<_, D> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, index| {
            let row = index[0] + 2 * index[1];
            let column = index[2] + 2 * index[3];
            if trees.coupled() != &label || row != 0 || column != 0 {
                return D::from_real(0.0);
            }
            let mu_row = trees.codomain_vertices()[0].get();
            let mu_column = trees.domain_vertices()[0].get();
            let inverse_denominator = D::from_real(1.0 / (lambda_plus - lambda_minus));
            match (mu_row, mu_column) {
                (1, 1) => D::from_real(4.0 - lambda_minus) * inverse_denominator,
                (2, 2) => D::from_real(-1.0 - lambda_minus) * inverse_denominator,
                (1, 2) => off_diagonal * inverse_denominator,
                (2, 1) => adjoint(off_diagonal) * inverse_denominator,
                _ => D::from_real(0.0),
            }
        })
        .unwrap();
    assert_same_checked_generic_layout_and_close(&projector, &expected, close);
}

#[cfg(feature = "cgc-gen")]
#[test]
fn sun_checked_generic_eigh_cross_mu_projectors_for_both_dtypes() {
    let _serial = crate::counting_alloc::serial();
    for (n, label) in [(3, vec![1, 1]), (4, vec![1, 0, 1])] {
        assert_sun_checked_generic_eigh(
            n,
            label.clone(),
            1.0,
            |value| value,
            |value| value,
            1.0,
            |actual, expected| (actual - expected).abs(),
        );
        assert_sun_checked_generic_eigh(
            n,
            label,
            Complex64::new(1.0, 1.0),
            |value| value.conj(),
            |value| value.re,
            2.0,
            |actual, expected| (actual - expected).norm(),
        );
    }
}

#[cfg(feature = "cgc-gen")]
fn assert_sun_checked_generic_eig<D>(n: usize, label: Vec<i64>)
where
    D: SunEigInput,
{
    use symtn::sector::LieGroupFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(n).unwrap()).unwrap());
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(label.clone(), 2)]).unwrap();
    let source: TensorMap<_, D> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, index| {
            let row = index[0] + 2 * index[1];
            let column = index[2] + 2 * index[3];
            if row != column {
                return D::from_real(0.0);
            }
            if trees.coupled() == &label {
                let mu_row = trees.codomain_vertices()[0].get();
                let mu_column = trees.domain_vertices()[0].get();
                let shift = 10.0 * row as f64;
                D::from_real(match (mu_row, mu_column) {
                    (1, 1) | (2, 2) => 1.0 + shift,
                    (1, 2) => -3.0,
                    (2, 1) => 1.0,
                    _ => 0.0,
                })
            } else if trees.codomain_vertices() == trees.domain_vertices() {
                D::from_real(50.0 + 10.0 * row as f64)
            } else {
                D::from_real(0.0)
            }
        })
        .unwrap();
    let Eig { d, v } = source.eig_full(&[0, 1], &[2, 3]).unwrap();
    assert!(std::ptr::eq(d.provider(), provider.as_ref()));
    assert!(std::ptr::eq(v.provider(), provider.as_ref()));
    assert!(symtn::expert::diagonal_spectrum(&d).unwrap().is_some());
    let complex_source = D::to_complex(&source);
    for (sector, a) in complex_source.blocks().unwrap() {
        let (v, d) = (v.block(&sector).unwrap(), d.block(&sector).unwrap());
        assert_unit_gauged_eigenpairs(
            a.rows(),
            |i, j| a.get(i, j).unwrap(),
            |i, j| v.get(i, j).unwrap(),
            |k| d.get(k, k).unwrap(),
            f64::EPSILON,
        );
    }
    let av = complex_source.compose(&v).unwrap();
    let vd = v.compose(&d).unwrap();
    assert!(av
        .dense_data()
        .unwrap()
        .iter()
        .zip(vd.dense_data().unwrap())
        .all(|(actual, expected)| (*actual - *expected).norm() < 1.0e-8));
    let rebuilt = v
        .compose(&d)
        .unwrap()
        .compose(&v.inv(&[0, 1], &[2]).unwrap())
        .unwrap();
    assert_same_checked_generic_layout_and_close(&rebuilt, &complex_source, |actual, expected| {
        (actual - expected).norm()
    });

    let lambda_plus = Complex64::new(1.0, 3.0_f64.sqrt());
    let identity: TensorMap<_, Complex64> = TensorMap::from_subblock_fn(
        &runtime,
        d.codomain().iter(),
        d.domain().iter(),
        |_, index| Complex64::new(f64::from(index[0] == index[1]), 0.0),
    )
    .unwrap();
    let mut selector = identity.clone();
    let mut found_target = false;
    for index in 0..d.subblock_count() {
        let block = d.subblock(index).unwrap();
        for diagonal in 0..block.shape()[0] {
            let value = d.materialize().unwrap().dense_data().unwrap()
                [block.offset() + diagonal * block.strides()[0] + diagonal * block.strides()[1]];
            if (value - lambda_plus).norm() < 1.0e-8 {
                assert!(!found_target);
                found_target = true;
            } else {
                selector = selector
                    .compose(
                        &d.axpby(Complex64::new(1.0, 0.0), &identity, -value)
                            .unwrap()
                            .scale(Complex64::new(1.0, 0.0) / (lambda_plus - value))
                            .unwrap(),
                    )
                    .unwrap();
            }
        }
    }
    assert!(found_target);
    let projector = v
        .compose(&selector)
        .unwrap()
        .compose(&v.inv(&[0, 1], &[2]).unwrap())
        .unwrap();
    let root = 3.0_f64.sqrt();
    let expected: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, index| {
            let row = index[0] + 2 * index[1];
            let column = index[2] + 2 * index[3];
            if trees.coupled() != &label || row != 0 || column != 0 {
                return Complex64::new(0.0, 0.0);
            }
            match (
                trees.codomain_vertices()[0].get(),
                trees.domain_vertices()[0].get(),
            ) {
                (1, 1) | (2, 2) => Complex64::new(0.5, 0.0),
                (1, 2) => Complex64::new(0.0, root / 2.0),
                (2, 1) => Complex64::new(0.0, -1.0 / (2.0 * root)),
                _ => Complex64::new(0.0, 0.0),
            }
        })
        .unwrap();
    assert_same_checked_generic_layout_and_close(&projector, &expected, |actual, expected| {
        (actual - expected).norm()
    });
}

#[cfg(feature = "cgc-gen")]
#[test]
fn sun_checked_generic_eig_nonnormal_cross_mu_projectors_for_both_dtypes() {
    let _serial = crate::counting_alloc::serial();
    for (n, label) in [(3, vec![1, 1]), (4, vec![1, 0, 1])] {
        assert_sun_checked_generic_eig::<f64>(n, label.clone());
        assert_sun_checked_generic_eig::<Complex64>(n, label);
    }
}

#[test]
fn checked_generic_eigh_vals_preserves_spectrum_and_dtype() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, _| 2.5).unwrap();

    let spectra = source.eigh_vals(&[0], &[1], HermitianTol::DEFAULT).unwrap();
    assert_eq!(spectra.len(), 1);
    assert_eq!(spectra[0].sector, Label::X);
    assert_eq!(spectra[0].values, vec![2.5]);

    let complex = source.convert::<Complex64>();
    assert_eq!(
        complex
            .eigh_vals(&[0], &[1], HermitianTol::DEFAULT)
            .unwrap(),
        spectra
    );

    let compact: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: Label::X,
            values: vec![2.5],
        }],
    )
    .unwrap();
    assert_eq!(
        compact
            .eigh_vals(&[0], &[1], HermitianTol::DEFAULT)
            .unwrap(),
        spectra
    );
    provider.fail_decode.store(true, Ordering::Relaxed);
    assert!(matches!(
        compact.eigh_vals(&[0], &[1], HermitianTol::DEFAULT),
        Err(GenericTensorError::Plan(
            symtn::typed::CheckedGenericPlanError::Provider(ToyError::Decode)
        ))
    ));
    provider.fail_decode.store(false, Ordering::Relaxed);
}

fn assert_checked_generic_eigh_factors<D>(
    source: &TensorMap<CheckedOnlyToy, D>,
    close: impl Fn(D, D) -> f64 + Copy,
    adjoint: impl Fn(D) -> D + Copy,
) where
    D: symtn::typed::FactorizationScalar + symtn::typed::SpectrumMagnitude + fmt::Debug,
{
    let Eigh { d, v } = source.eigh_full(&[0], &[1], HermitianTol::DEFAULT).unwrap();
    assert!(std::ptr::eq(d.provider(), source.provider()));
    assert!(std::ptr::eq(v.provider(), source.provider()));
    assert!(symtn::typed::__network::runtime_identity(d.runtime()).matches(source.runtime()));
    assert!(symtn::typed::__network::runtime_identity(v.runtime()).matches(source.runtime()));
    assert_eq!(v.codomain(), source.codomain());
    assert_eq!(d.codomain(), d.domain());
    assert_eq!(v.domain(), d.codomain());
    let spectrum = symtn::expert::diagonal_spectrum(&d).unwrap().unwrap();
    assert_eq!(spectrum.iter().map(|s| s.values.len()).sum::<usize>(), 3);
    assert_eq!(d.materialize().unwrap().dense_data().unwrap().len(), 9);

    for (actual, expected) in source
        .compose(&v)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(v.compose(&d).unwrap().dense_data().unwrap())
    {
        assert!(close(*actual, *expected) < 1e-10);
    }
    let vectors = v.dense_data().unwrap();
    let diagonal_materialized = d.materialize().unwrap();
    let diagonal = diagonal_materialized.dense_data().unwrap();
    for column in 0..3 {
        for row in 0..3 {
            let gram = (0..3).fold(D::from_real(0.0), |sum, inner| {
                sum + adjoint(vectors[inner + row * 3]) * vectors[inner + column * 3]
            });
            assert!(close(gram, D::from_real(f64::from(row == column))) < 1e-10);
            let rebuilt = (0..3).fold(D::from_real(0.0), |sum, inner| {
                sum + vectors[row + inner * 3]
                    * diagonal[inner + inner * 3]
                    * adjoint(vectors[column + inner * 3])
            });
            assert!(close(rebuilt, source.dense_data().unwrap()[row + column * 3]) < 1e-10);
        }
    }

    // The truncated eigendecomposition is a composition (#1534).
    let found = d.domain()[0]
        .find_truncated(&d.diagview().unwrap(), &Truncation::rank(5))
        .unwrap();
    let truncated_d = d
        .restrict_leg(&[(0, &found.selection), (1, &found.selection)])
        .unwrap();
    let truncated_v = v
        .restrict_leg(&[(v.codomain_rank(), &found.selection)])
        .unwrap();
    assert!(std::ptr::eq(truncated_d.provider(), source.provider()));
    assert!(std::ptr::eq(truncated_v.provider(), source.provider()));
    // The kept values carry the same values in the same order, exactly — no
    // rounding, and no reordering of the values inside a sector.
    let compact = truncated_d.diagview().unwrap();
    assert_eq!(compact.len(), 1);
    assert_eq!(compact[0].sector, Label::X);
    for (actual, expected) in compact[0]
        .values
        .iter()
        .zip([D::from_real(-3.0), D::from_real(2.0)])
    {
        assert_eq!(close(*actual, expected), 0.0);
    }
    let expected_error = (1.0 + 2.0_f64.sqrt()).sqrt();
    assert!((found.error - expected_error).abs() < 1e-12);
}

#[test]
fn checked_generic_eigh_full_and_trunc_preserve_contract_for_both_dtypes() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 3)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
            [-3.0, 2.0, 1.0][index[0]] * f64::from(index[0] == index[1])
        })
        .unwrap();
    assert_checked_generic_eigh_factors(
        &source,
        |actual, expected| (actual - expected).abs(),
        |value| value,
    );
    let complex = source.convert::<Complex64>();
    assert_checked_generic_eigh_factors(
        &complex,
        |actual, expected| (actual - expected).norm(),
        |value| value.conj(),
    );
    let Eigh { d: complex_d, .. } = complex
        .eigh_full(&[0], &[1], HermitianTol::DEFAULT)
        .unwrap();
    assert!(complex_d
        .materialize()
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .all(|value| value.im == 0.0));
}

#[test]
fn checked_generic_eigh_lazy_success_and_failure_leave_the_view_lazy() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let hermitian: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
            [[2.0, 1.0], [1.0, -1.0]][index[0]][index[1]]
        })
        .unwrap();
    let lazy = hermitian.adjoint().unwrap();
    assert!(
        symtn::typed::__network::network_reuse_class(&lazy, false)
            == symtn::typed::__network::NetworkReuseClass::LazyAdjoint
    );
    assert!(lazy.eigh_full(&[0], &[1], HermitianTol::DEFAULT).is_ok());
    assert!(
        symtn::typed::__network::network_reuse_class(&lazy, false)
            == symtn::typed::__network::NetworkReuseClass::LazyAdjoint
    );

    let nonhermitian: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
            [[0.0, 1.0], [0.0, 0.0]][index[0]][index[1]]
        })
        .unwrap();
    let lazy = nonhermitian.adjoint().unwrap();
    assert!(lazy.eigh_full(&[0], &[1], HermitianTol::DEFAULT).is_err());
    assert!(
        symtn::typed::__network::network_reuse_class(&lazy, false)
            == symtn::typed::__network::NetworkReuseClass::LazyAdjoint
    );
}

#[test]
fn checked_generic_eigh_rejects_invalid_inputs_before_publication() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let wide = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let narrow = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let nonendomorphism: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&wide], [&narrow], |_, _| 1.0).unwrap();
    assert!(matches!(nonendomorphism
        .eigh_full(&[0], &[1], HermitianTol::DEFAULT), Err(GenericTensorError::Plan(symtn::typed::CheckedGenericPlanError::Operation(symtn::typed::OperationError::SpaceMismatch { message }))) if message.contains("endomorphism")));

    let nonhermitian: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&wide], [&wide], |_, index| {
            [[0.0, 1.0], [0.0, 0.0]][index[0]][index[1]]
        })
        .unwrap();
    assert!(matches!(nonhermitian
        .eigh_full(&[0], &[1], HermitianTol::DEFAULT), Err(GenericTensorError::Plan(symtn::typed::CheckedGenericPlanError::Operation(symtn::typed::OperationError::InvalidArgument { message }))) if message.contains("Hermitian coupled-sector blocks")));
    let nonfinite: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&wide], [&wide], |_, index| {
            if index[0] == index[1] {
                f64::NAN
            } else {
                0.0
            }
        })
        .unwrap();
    assert!(matches!(nonfinite
        .eigh_full(&[0], &[1], HermitianTol::DEFAULT), Err(GenericTensorError::Plan(symtn::typed::CheckedGenericPlanError::Operation(symtn::typed::OperationError::InvalidArgument { message }))) if message.contains("input components must be finite")));
}

/// Owned EIGH entries; the destination form is refused, so a checked
/// Generic EIGH must use owned output.
const EIGH_OWNED: &[Kernel] = &[Kernel::Eigh, Kernel::EighVals];

/// Counts owned EIGH calls; the `fail_at`-th fails.
fn eigh_fault_spy(counts: &Arc<SpyCounts>, fail_at: Option<usize>) -> SpyExecutor {
    let spy = SpyExecutor::counting(counts).failing(
        &[Kernel::EighInto],
        None,
        "checked Generic EIGH must use owned output",
    );
    match fail_at {
        Some(nth) => spy.failing(
            EIGH_OWNED,
            Some(nth),
            "injected checked Generic EIGH failure",
        ),
        None => spy,
    }
}

/// Fails every dense EIG call.
fn eig_fault_spy() -> SpyExecutor {
    SpyExecutor::default().failing(
        &[Kernel::Eig, Kernel::EigVals],
        None,
        "injected checked Generic EIG failure",
    )
}

#[test]
fn checked_generic_eigh_preflights_all_sectors_and_runs_once_per_sector() {
    let _serial = crate::counting_alloc::serial();
    let calls = Arc::new(SpyCounts::default());
    let runtime = Runtime::builder()
        .dense_threads(1)
        .with_dense_executor(Box::new(eigh_fault_spy(&calls, None)))
        .build()
        .unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 1), (Label::X, 2)]).unwrap();
    let hermitian: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |trees, index| {
            if index[0] == index[1] {
                if trees.coupled() == &Label::Vacuum {
                    1.0
                } else {
                    2.0
                }
            } else {
                0.0
            }
        })
        .unwrap();
    hermitian
        .eigh_full(&[0], &[1], HermitianTol::DEFAULT)
        .unwrap();
    assert_eq!(calls.of(EIGH_OWNED), 2);

    calls.reset();
    let nonhermitian: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |trees, index| {
            if trees.coupled() == &Label::X && index == [0, 1] {
                1.0
            } else {
                0.0
            }
        })
        .unwrap();
    assert!(matches!(nonhermitian
        .eigh_full(&[0], &[1], HermitianTol::DEFAULT), Err(GenericTensorError::Plan(symtn::typed::CheckedGenericPlanError::Operation(symtn::typed::OperationError::InvalidArgument { message }))) if message.contains("Hermitian coupled-sector blocks")));
    assert_eq!(calls.of(EIGH_OWNED), 0);
    assert_eq!(calls.total(), 0);
}

#[test]
fn checked_generic_eigh_dense_failure_preserves_the_source() {
    let _serial = crate::counting_alloc::serial();
    let calls = Arc::new(SpyCounts::default());
    let runtime = Runtime::builder()
        .dense_threads(1)
        .with_dense_executor(Box::new(eigh_fault_spy(&calls, Some(1))))
        .build()
        .unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
            [[2.0, 1.0], [1.0, -1.0]][index[0]][index[1]]
        })
        .unwrap();
    let before = source.dense_data().unwrap().to_vec();
    assert!(source.eigh_full(&[0], &[1], HermitianTol::DEFAULT).is_err());
    assert_eq!(calls.of(EIGH_OWNED), 1);
    assert_eq!(source.dense_data().unwrap(), before);
    assert!(std::ptr::eq(source.provider(), provider.as_ref()));
}

#[test]
fn checked_generic_eig_dense_failure_preserves_the_source() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder()
        .dense_threads(1)
        .with_dense_executor(Box::new(eig_fault_spy()))
        .build()
        .unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
            [[1.0, -3.0], [1.0, 1.0]][index[0]][index[1]]
        })
        .unwrap();
    let before = source.dense_data().unwrap().to_vec();
    assert!(source.eig_full(&[0], &[1]).is_err());
    assert_eq!(source.dense_data().unwrap(), before);
    assert!(std::ptr::eq(source.provider(), provider.as_ref()));
}

#[test]
fn checked_generic_eigh_qdim_and_decode_failures_publish_no_pair() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
            [2.0, -1.0][index[0]] * f64::from(index[0] == index[1])
        })
        .unwrap();
    let before = source.dense_data().unwrap().to_vec();

    // The truncation composition decodes labels in `diagview` and reads the
    // quantum dimension in `find_truncated`; both surface the provider error.
    let Eigh { d, .. } = source.eigh_full(&[0], &[1], HermitianTol::DEFAULT).unwrap();
    provider.fail_decode.store(true, Ordering::Relaxed);
    assert!(matches!(
        d.diagview(),
        Err(GenericTensorError::Structure(
            CheckedGenericStructureError::Provider(ToyError::Decode)
        ))
    ));
    provider.fail_decode.store(false, Ordering::Relaxed);

    let spectra = d.diagview().unwrap();
    provider.fail_dim.store(true, Ordering::Relaxed);
    assert!(matches!(
        d.domain()[0].find_truncated(&spectra, &Truncation::rank(1)),
        Err(GenericTensorError::Plan(
            symtn::typed::CheckedGenericPlanError::Provider(ToyError::Algebra)
        ))
    ));
    provider.fail_dim.store(false, Ordering::Relaxed);

    assert_eq!(source.dense_data().unwrap(), before);
    assert!(std::ptr::eq(source.provider(), provider.as_ref()));
}

#[test]
fn checked_generic_eig_qdim_and_decode_failures_publish_no_pair() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
            [[1.0, -3.0], [1.0, 1.0]][index[0]][index[1]]
        })
        .unwrap();
    let before = source.dense_data().unwrap().to_vec();

    // The truncation composition decodes labels in `diagview` and reads the
    // quantum dimension in `find_truncated`; both surface the provider error.
    let Eig { d, .. } = source.eig_full(&[0], &[1]).unwrap();
    provider.fail_decode.store(true, Ordering::Relaxed);
    assert!(matches!(
        d.diagview(),
        Err(GenericTensorError::Structure(
            CheckedGenericStructureError::Provider(ToyError::Decode)
        ))
    ));
    provider.fail_decode.store(false, Ordering::Relaxed);

    let spectra = d.diagview().unwrap();
    provider.fail_dim.store(true, Ordering::Relaxed);
    assert!(matches!(
        d.domain()[0].find_truncated(&spectra, &Truncation::rank(1)),
        Err(GenericTensorError::Plan(
            symtn::typed::CheckedGenericPlanError::Provider(ToyError::Algebra)
        ))
    ));
    provider.fail_dim.store(false, Ordering::Relaxed);

    assert_eq!(source.dense_data().unwrap(), before);
    assert!(std::ptr::eq(source.provider(), provider.as_ref()));
}

#[test]
fn checked_generic_eigh_signed_ties_are_stable_and_degenerate_projectors_are_invariant() {
    let _serial = crate::counting_alloc::serial();
    use std::cell::Cell;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 3)]).unwrap();
    let tied: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
            [-2.0, 2.0, 1.0][index[0]] * f64::from(index[0] == index[1])
        })
        .unwrap();
    let Eigh { d, .. } = tied.eigh_full(&[0], &[1], HermitianTol::DEFAULT).unwrap();
    // The ascending order of the ±2 pair is the contract; the values carry
    // the eigensolver's rounding (`terms` = the block size 3).
    numerics::assert_slices_close(
        "eigh_full tied spectrum",
        &[
            d.materialize().unwrap().dense_data().unwrap()[0],
            d.materialize().unwrap().dense_data().unwrap()[4],
            d.materialize().unwrap().dense_data().unwrap()[8],
        ],
        &[-2.0, 1.0, 2.0],
        3,
    );

    let degenerate: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
            [2.0, 2.0, -1.0][index[0]] * f64::from(index[0] == index[1])
        })
        .unwrap();
    let Eigh { d, v } = degenerate
        .eigh_full(&[0], &[1], HermitianTol::DEFAULT)
        .unwrap();
    let selector_codomain = d.codomain();
    let selector_domain = d.domain();
    let selector: TensorMap<_, f64> = TensorMap::from_subblock_fn(
        &runtime,
        selector_codomain.iter(),
        selector_domain.iter(),
        |_, index| {
            f64::from(
                index[0] == index[1]
                    && d.materialize().unwrap().dense_data().unwrap()[index[0] * 4] == 2.0,
            )
        },
    )
    .unwrap();
    let lazy_vh = v.adjoint().unwrap();
    let data = lazy_vh
        .materialize()
        .unwrap()
        .dense_data()
        .unwrap()
        .to_vec();
    let position = Cell::new(0usize);
    let codomain = v.domain();
    let domain = v.codomain();
    let vh: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, codomain.iter(), domain.iter(), |_, _| {
            let index = position.get();
            position.set(index + 1);
            data[index]
        })
        .unwrap();
    let projector = v.compose(&selector).unwrap().compose(&vh).unwrap();
    numerics::assert_slices_close(
        "eigh degenerate projector",
        projector.dense_data().unwrap(),
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        3,
    );
}

#[test]
fn checked_generic_eig_vals_preserves_spectrum_and_dtype() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
            if indices[0] == indices[1] {
                (indices[0] + 2) as f64
            } else {
                0.0
            }
        })
        .unwrap();

    let spectra = source.eig_vals(&[0], &[1]).unwrap();
    assert_eq!(spectra.len(), 1);
    assert_eq!(spectra[0].sector, Label::X);
    assert_eq!(
        spectra[0].values,
        vec![Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0)]
    );

    let complex = source.convert::<Complex64>();
    assert_eq!(complex.eig_vals(&[0], &[1]).unwrap(), spectra);

    let compact: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: Label::X,
            values: vec![Complex64::new(1.0, -1.0), Complex64::new(0.0, 3.0)],
        }],
    )
    .unwrap();
    assert_eq!(
        compact.eig_vals(&[0], &[1]).unwrap(),
        vec![SectorSpectrum {
            sector: Label::X,
            values: vec![Complex64::new(0.0, 3.0), Complex64::new(1.0, -1.0)],
        }]
    );
    provider.fail_decode.store(true, Ordering::Relaxed);
    assert!(matches!(
        compact.eig_vals(&[0], &[1]),
        Err(GenericTensorError::Plan(
            symtn::typed::CheckedGenericPlanError::Provider(ToyError::Decode)
        ))
    ));
    provider.fail_decode.store(false, Ordering::Relaxed);
}

fn assert_checked_generic_eig_reconstruction(
    source: &TensorMap<CheckedOnlyToy, Complex64>,
    d: &TensorMap<CheckedOnlyToy, Complex64>,
    v: &TensorMap<CheckedOnlyToy, Complex64>,
) {
    assert!(std::ptr::eq(d.provider(), source.provider()));
    assert!(std::ptr::eq(v.provider(), source.provider()));
    assert!(symtn::typed::__network::runtime_identity(d.runtime()).matches(source.runtime()));
    assert!(symtn::typed::__network::runtime_identity(v.runtime()).matches(source.runtime()));
    assert_eq!(v.codomain(), source.codomain());
    assert_eq!(v.domain(), d.codomain());
    let av = source.compose(v).unwrap();
    let vd = v.compose(d).unwrap();
    let scale = source
        .dense_data()
        .unwrap()
        .iter()
        .map(|value| value.norm())
        .fold(1.0_f64, f64::max);
    assert!(av
        .dense_data()
        .unwrap()
        .iter()
        .zip(vd.dense_data().unwrap())
        .all(|(actual, expected)| (*actual - *expected).norm() <= 1.0e-11 * scale));
    let rebuilt = v
        .compose(d)
        .unwrap()
        .compose(&v.inv(&[0], &[1]).unwrap())
        .unwrap();
    assert_same_checked_generic_layout_and_close(&rebuilt, source, |actual, expected| {
        (actual - expected).norm()
    });
}

#[test]
fn checked_generic_eig_full_is_complex_and_reconstructs_nonnormal_inputs() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let real: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
            [[1.0, -3.0], [1.0, 1.0]][index[0]][index[1]]
        })
        .unwrap();
    let Eig {
        d: real_d,
        v: real_v,
    } = real.eig_full(&[0], &[1]).unwrap();
    assert!(real_d
        .materialize()
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .any(|value| value.im.abs() > 1.0));
    for column in 0..2 {
        let pivot = (0..2)
            .max_by(|&left, &right| {
                real_v.dense_data().unwrap()[left + column * 2]
                    .norm()
                    .total_cmp(&real_v.dense_data().unwrap()[right + column * 2].norm())
            })
            .unwrap();
        let pivot = real_v.dense_data().unwrap()[pivot + column * 2];
        assert!(pivot.im.abs() < 1.0e-12);
        assert!(pivot.re >= 0.0);
    }
    assert_checked_generic_eig_reconstruction(&real.convert::<Complex64>(), &real_d, &real_v);

    let complex = real
        .convert::<Complex64>()
        .scale(Complex64::new(1.0, 0.25))
        .unwrap();
    let Eig {
        d: complex_d,
        v: complex_v,
    } = complex.eig_full(&[0], &[1]).unwrap();
    assert!(complex_d
        .materialize()
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .any(|value| value.im.abs() > 1.0));
    assert_checked_generic_eig_reconstruction(&complex, &complex_d, &complex_v);

    let compact: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: Label::X,
            values: vec![Complex64::new(1.0, 1.0), Complex64::new(0.0, -3.0)],
        }],
    )
    .unwrap();
    let Eig { d, v } = compact.eig_full(&[0], &[1]).unwrap();
    assert_eq!(
        d.diagview().unwrap()[0].values,
        [Complex64::new(0.0, -3.0), Complex64::new(1.0, 1.0)]
    );
    let dense = compact.materialize().unwrap();
    assert_checked_generic_eig_reconstruction(&dense, &d, &v);
    let fault_runtime = Runtime::builder()
        .dense_threads(1)
        .with_dense_executor(Box::new(eig_fault_spy()))
        .build()
        .unwrap();
    let fault_provider = Arc::new(CheckedOnlyToy::new(1));
    let fault_leg = GradedSpace::try_new(Arc::clone(&fault_provider), [(Label::X, 2)]).unwrap();
    let compact_on = |leg: &GradedSpace<_>| -> TensorMap<_, Complex64> {
        TensorMap::diagonal(
            &fault_runtime,
            leg,
            [SectorSpectrum {
                sector: Label::X,
                values: vec![Complex64::new(1.0, 1.0), Complex64::new(0.0, -3.0)],
            }],
        )
        .unwrap()
    };
    let nondual_compact = compact_on(&fault_leg);
    let fault_compact = compact_on(&fault_leg.try_dual().unwrap());
    fault_provider.invalid_style.store(true, Ordering::Relaxed);
    // A nondual bond is its own eigenbasis bond (TensorKit `fuse(V) = V`): `v`
    // and `d` live on the input space, so the provider is never queried and
    // the dense eigensolver never runs.
    reset_provider_queries(&fault_provider);
    let Eig { d, v } = nondual_compact.eig_full(&[0], &[1]).unwrap();
    assert_eq!(
        fault_provider.queries_since_reset.load(Ordering::Relaxed),
        0
    );
    assert_eq!(v.codomain(), nondual_compact.codomain());
    assert_eq!(d.codomain(), nondual_compact.codomain());
    // A dual bond's eigenbasis bond is the fresh nondual `fuse(V)`: the
    // eligible compact path reaches checked V publication; falling back to
    // dense EIG would return the injected dense error instead.
    let compact_error = fault_compact.eig_full(&[0], &[1]).unwrap_err();
    assert!(
        matches!(
            compact_error,
            GenericTensorError::Plan(symtn::typed::CheckedGenericPlanError::Operation(
                symtn::typed::OperationError::Core(
                    symtn::typed::CoreError::UnsupportedFusionStyle { .. }
                )
            ))
        ),
        "{compact_error:?}"
    );
}

#[test]
fn checked_generic_eig_ties_are_stable_and_degenerate_projectors_are_invariant() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 3)]).unwrap();
    let tied: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
            [-2.0, 2.0, 1.0][index[0]] * f64::from(index[0] == index[1])
        })
        .unwrap();
    let Eig { d, .. } = tied.eig_full(&[0], &[1]).unwrap();
    numerics::assert_slices_close(
        "eig_full tied spectrum",
        &[
            d.materialize().unwrap().dense_data().unwrap()[0],
            d.materialize().unwrap().dense_data().unwrap()[4],
            d.materialize().unwrap().dense_data().unwrap()[8],
        ],
        &[
            Complex64::new(-2.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
        ],
        3,
    );

    let degenerate: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
            [2.0, 2.0, -1.0][index[0]] * f64::from(index[0] == index[1])
        })
        .unwrap();
    let Eig { d, v } = degenerate.eig_full(&[0], &[1]).unwrap();
    let selector: TensorMap<_, Complex64> = TensorMap::from_subblock_fn(
        &runtime,
        d.codomain().iter(),
        d.domain().iter(),
        |_, index| {
            Complex64::new(
                f64::from(
                    index[0] == index[1]
                        && (d.materialize().unwrap().dense_data().unwrap()[index[0] * 4]
                            - Complex64::new(2.0, 0.0))
                        .norm()
                            < 1.0e-12,
                ),
                0.0,
            )
        },
    )
    .unwrap();
    let projector = v
        .compose(&selector)
        .unwrap()
        .compose(&v.inv(&[0], &[1]).unwrap())
        .unwrap();
    numerics::assert_slices_close(
        "eig degenerate projector",
        projector.dense_data().unwrap(),
        &[
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        3,
    );
}

#[test]
fn checked_generic_eig_truncation_reports_discarded_spectrum_norm_only() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 3)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
            [-3.0, 2.0, 1.0][index[0]] * f64::from(index[0] == index[1])
        })
        .unwrap();
    // The truncated eigendecomposition is a composition (#1534).
    let Eig { d, v } = source.eig_full(&[0], &[1]).unwrap();
    let found = d.domain()[0]
        .find_truncated(&d.diagview().unwrap(), &Truncation::rank(5))
        .unwrap();
    let truncated_d = d
        .restrict_leg(&[(0, &found.selection), (1, &found.selection)])
        .unwrap();
    let truncated_v = v
        .restrict_leg(&[(v.codomain_rank(), &found.selection)])
        .unwrap();
    assert!(std::ptr::eq(truncated_d.provider(), provider.as_ref()));
    assert!(std::ptr::eq(truncated_v.provider(), provider.as_ref()));
    let kept = truncated_d.diagview().unwrap();
    assert_eq!(kept[0].sector, Label::X);
    assert_eq!(
        kept[0].values,
        vec![Complex64::new(-3.0, 0.0), Complex64::new(2.0, 0.0)]
    );
    assert!((found.error - (1.0 + 2.0_f64.sqrt()).sqrt()).abs() < 1.0e-12);
}

/// Independent dense oracle for one coupled block: every returned pair
/// satisfies `A v = lambda v` to backward error relative to `|A| |v|` and no
/// eigenvector vanishes. No inverse of `V` is formed: a defective block has
/// no eigenbasis.
fn assert_defective_eigenpairs(
    n: usize,
    a: impl Fn(usize, usize) -> Complex64,
    v: impl Fn(usize, usize) -> Complex64,
    lambda: impl Fn(usize) -> Complex64,
    exact: Complex64,
) {
    let scale = (0..n)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .map(|(i, j)| a(i, j).norm())
        .fold(1.0_f64, f64::max);
    // A size-n Jordan block's eigenvalue moves by about (eps |A|)^(1/n) under a
    // backward-stable solver, so the spectrum bound follows that theory; the
    // eigenpair residual below is the main oracle.
    let spectrum_tol = 10.0 * (f64::EPSILON * scale).powf(1.0 / n as f64);
    for k in 0..n {
        assert!((lambda(k) - exact).norm() <= spectrum_tol);
        let norm = (0..n).map(|i| v(i, k).norm_sqr()).sum::<f64>().sqrt();
        assert!(norm > 0.5);
        for i in 0..n {
            let av = (0..n).map(|j| a(i, j) * v(j, k)).sum::<Complex64>();
            assert!((av - lambda(k) * v(i, k)).norm() <= 1.0e-12 * n as f64 * scale * norm);
        }
    }
}

#[test]
fn eig_full_accepts_finite_defective_sectors_in_both_modes_without_a_rank_svd() {
    let _serial = crate::counting_alloc::serial();
    use symtn::sector::{SU2FusionRule, SU2Irrep};

    let counts = Arc::new(SpyCounts::default());
    let runtime = Runtime::builder()
        .dense_threads(1)
        .with_dense_executor(Box::new(SpyExecutor::counting(&counts)))
        .build()
        .unwrap();
    // `lambda I + N` with `N` the upper shift: a single Jordan chain, so each
    // block of dimension two or more is defective.
    let jordan = |lambda: Complex64, row: usize, col: usize| {
        if row == col {
            lambda
        } else if col == row + 1 {
            Complex64::new(1.0, 0.0)
        } else {
            Complex64::new(0.0, 0.0)
        }
    };
    let checked_leg = GradedSpace::try_new(
        Arc::new(CheckedOnlyToy::new(0)),
        [(Label::Vacuum, 2), (Label::X, 3)],
    )
    .unwrap();
    let mf_leg = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(1), 2),
            (SU2Irrep::from_twice_spin(2), 3),
        ],
    )
    .unwrap();
    macro_rules! check {
        ($leg:expr, $first:expr, $dtype:ty, $from:expr, $lambda:expr) => {{
            let from: fn(Complex64) -> $dtype = $from;
            let lambda = |sector: &_| $lambda[usize::from(sector != &$first)];
            let source: TensorMap<_, $dtype> =
                TensorMap::from_subblock_fn(&runtime, [&$leg], [&$leg], |trees, index| {
                    from(jordan(lambda(trees.coupled()), index[0], index[1]))
                })
                .unwrap();
            counts.reset();
            let Eig { d, v } = source.eig_full(&[0], &[1]).unwrap();
            // One dense EIG per coupled sector and nothing else: no
            // eigenvector SVD runs in either mode.
            assert_eq!(counts.get(Kernel::Eig), 2);
            assert_eq!(counts.others(&[Kernel::Eig]), 0);
            let mut sectors = 0;
            for (sector, a) in source.blocks().unwrap() {
                let (v, d) = (v.block(&sector).unwrap(), d.block(&sector).unwrap());
                assert_defective_eigenpairs(
                    a.rows(),
                    |i, j| Complex64::from(a.get(i, j).unwrap()),
                    |i, j| v.get(i, j).unwrap(),
                    |k| d.get(k, k).unwrap(),
                    lambda(&sector),
                );
                sectors += 1;
            }
            assert_eq!(sectors, 2);
        }};
    }
    let real = [Complex64::new(2.0, 0.0), Complex64::new(-0.5, 0.0)];
    let complex = [Complex64::new(2.0, 1.0), Complex64::new(-0.5, -0.25)];
    check!(checked_leg, Label::Vacuum, f64, |z| z.re, real);
    check!(checked_leg, Label::Vacuum, Complex64, |z| z, complex);
    let spin_half = SU2Irrep::from_twice_spin(1);
    check!(mf_leg, spin_half, f64, |z| z.re, real);
    check!(mf_leg, spin_half, Complex64, |z| z, complex);
}

/// Independent oracle for the published eigenvector gauge (#2073), per
/// column `k` of one coupled block: `A v = lambda v` to backward error, unit
/// 2-norm (LAPACK `geev`, which MatrixAlgebraKit `eig_full!` publishes), and
/// the first largest-magnitude component real and non-negative (MAK
/// `gaugefix!`). `eps` is the payload's machine epsilon.
fn assert_unit_gauged_eigenpairs(
    n: usize,
    a: impl Fn(usize, usize) -> Complex64,
    v: impl Fn(usize, usize) -> Complex64,
    lambda: impl Fn(usize) -> Complex64,
    eps: f64,
) {
    let scale = (0..n)
        .flat_map(|i| (0..n).map(move |j| (i, j)))
        .map(|(i, j)| a(i, j).norm())
        .fold(1.0_f64, f64::max);
    let tol = 64.0 * n as f64 * eps;
    for k in 0..n {
        let norm = (0..n).map(|i| v(i, k).norm_sqr()).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() <= tol, "column {k} has norm {norm}");
        let pivot = (0..n).fold(0, |best, i| {
            if v(best, k).norm() < v(i, k).norm() {
                i
            } else {
                best
            }
        });
        assert!(v(pivot, k).im.abs() <= tol && v(pivot, k).re >= 0.0);
        for i in 0..n {
            let av = (0..n).map(|j| a(i, j) * v(j, k)).sum::<Complex64>();
            assert!((av - lambda(k) * v(i, k)).norm() <= tol * scale);
        }
    }
}

#[test]
fn eig_full_publishes_unit_norm_phase_gauged_columns_in_both_modes() {
    let _serial = crate::counting_alloc::serial();
    use symtn::sector::{SU2FusionRule, SU2Irrep};

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    // `lambda I + N` (a single Jordan chain: defective, and on base the dense
    // backend returned a column like `[9.007e14, -1]`), and an upper
    // triangular block with distinct eigenvalues `lambda + k` whose
    // eigenvectors have no magnitude ties.
    let jordan = |lambda: Complex64, row: usize, col: usize| {
        if row == col {
            lambda
        } else if col == row + 1 {
            Complex64::new(1.0, 0.0)
        } else {
            Complex64::new(0.0, 0.0)
        }
    };
    let triangular = |lambda: Complex64, row: usize, col: usize| {
        if row == col {
            lambda + row as f64
        } else if col == row + 1 {
            Complex64::new(0.5, 0.0)
        } else if col == row + 2 {
            Complex64::new(0.25, 0.0)
        } else {
            Complex64::new(0.0, 0.0)
        }
    };
    let checked_leg = GradedSpace::try_new(
        Arc::new(CheckedOnlyToy::new(0)),
        [(Label::Vacuum, 2), (Label::X, 3)],
    )
    .unwrap();
    // Non-Abelian sectors with degeneracy 2 and 3.
    let mf_leg = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(1), 2),
            (SU2Irrep::from_twice_spin(2), 3),
        ],
    )
    .unwrap();
    macro_rules! check {
        ($leg:expr, $first:expr, $dtype:ty, $from:expr, $widen:expr, $widen_eig:expr, $eps:expr, $lambda:expr) => {{
            let from: fn(Complex64) -> $dtype = $from;
            let lambda = |sector: &_| $lambda[usize::from(sector != &$first)];
            for fixture in [jordan, triangular] {
                let source: TensorMap<_, $dtype> =
                    TensorMap::from_subblock_fn(&runtime, [&$leg], [&$leg], |trees, index| {
                        from(fixture(lambda(trees.coupled()), index[0], index[1]))
                    })
                    .unwrap();
                let Eig { d, v } = source.eig_full(&[0], &[1]).unwrap();
                let mut sectors = 0;
                for (sector, a) in source.blocks().unwrap() {
                    let (v, d) = (v.block(&sector).unwrap(), d.block(&sector).unwrap());
                    assert_unit_gauged_eigenpairs(
                        a.rows(),
                        |i, j| $widen(a.get(i, j).unwrap()),
                        |i, j| $widen_eig(v.get(i, j).unwrap()),
                        |k| $widen_eig(d.get(k, k).unwrap()),
                        $eps,
                    );
                    sectors += 1;
                }
                assert_eq!(sectors, 2);
            }
        }};
    }
    let real = [Complex64::new(2.0, 0.0), Complex64::new(-0.5, 0.0)];
    let complex = [Complex64::new(2.0, 1.0), Complex64::new(-0.5, -0.25)];
    let c32 = |z: Complex32| Complex64::new(z.re.into(), z.im.into());
    let spin_half = SU2Irrep::from_twice_spin(1);
    macro_rules! both_modes {
        ($($args:tt)*) => {
            check!(checked_leg, Label::Vacuum, $($args)*);
            check!(mf_leg, spin_half, $($args)*);
        };
    }
    both_modes!(
        f64,
        |z| z.re,
        |x: f64| Complex64::new(x, 0.0),
        |z| z,
        f64::EPSILON,
        real
    );
    both_modes!(Complex64, |z| z, |z| z, |z| z, f64::EPSILON, complex);
    both_modes!(
        f32,
        |z| z.re as f32,
        |x: f32| Complex64::new(x.into(), 0.0),
        c32,
        f64::from(f32::EPSILON),
        real
    );
    both_modes!(
        Complex32,
        |z| Complex32::new(z.re as f32, z.im as f32),
        c32,
        c32,
        f64::from(f32::EPSILON),
        complex
    );
}

#[test]
fn eig_full_rejects_nonfinite_eigenvectors_of_finite_input_in_both_modes() {
    let _serial = crate::counting_alloc::serial();
    use symtn::sector::{SU2FusionRule, SU2Irrep};

    // A finite Jordan block `2 I + N` of size n in one sector. faer's
    // back-substitution overflows here (f32 from n = 8, f64 from n = 26)
    // while the eigenvalues stay finite; LAPACK `xTREVC` rescales and
    // returns finite columns. Either outcome is allowed, but never `Ok` with
    // a nonfinite column: the eigenvalue check runs first, then the
    // eigenvector check, both a `NumericalFailure`.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let jordan = |row: usize, col: usize| {
        if row == col {
            2.0
        } else if col == row + 1 {
            1.0
        } else {
            0.0
        }
    };
    let faer_only = crate::backend::faer_is_default_backend();
    macro_rules! check {
        ($leg:expr, $dtype:ty, $from:expr) => {{
            let from: fn(f64) -> $dtype = $from;
            let source: TensorMap<_, $dtype> =
                TensorMap::from_subblock_fn(&runtime, [&$leg], [&$leg], |_, index| {
                    from(jordan(index[0], index[1]))
                })
                .unwrap();
            match source.eig_full(&[0], &[1]) {
                Ok(Eig { v, .. }) => {
                    assert!(
                        v.dense_data()
                            .unwrap()
                            .iter()
                            .all(|value| value.re.is_finite() && value.im.is_finite()),
                        "Ok with a nonfinite eigenvector column"
                    );
                    assert!(!faer_only, "faer is expected to overflow on this block");
                }
                Err(error) => {
                    assert!(
                        matches!(
                            crate::report::cause::<symtn::typed::OperationError>(&error),
                            Some(symtn::typed::OperationError::Dense(
                                symtn::expert::DenseError::NumericalFailure {
                                    op: "eig eigenvector check",
                                    ..
                                }
                            ))
                        ),
                        "{error:?}"
                    );
                }
            }
        }};
    }
    for (n, single) in [(26, false), (30, false), (8, true), (10, true)] {
        let checked_leg =
            GradedSpace::try_new(Arc::new(CheckedOnlyToy::new(0)), [(Label::Vacuum, n)]).unwrap();
        let mf_leg =
            GradedSpace::try_new(Arc::new(SU2FusionRule), [(SU2Irrep::from_twice_spin(1), n)])
                .unwrap();
        if single {
            check!(checked_leg, f32, |x| x as f32);
            check!(checked_leg, Complex32, |x| Complex32::new(x as f32, 0.0));
            check!(mf_leg, f32, |x| x as f32);
            check!(mf_leg, Complex32, |x| Complex32::new(x as f32, 0.0));
        } else {
            check!(checked_leg, f64, |x| x);
            check!(checked_leg, Complex64, |x| Complex64::new(x, 0.0));
            check!(mf_leg, f64, |x| x);
            check!(mf_leg, Complex64, |x| Complex64::new(x, 0.0));
        }
    }
}

#[test]
fn checked_generic_eig_rejects_nonfinite_input() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let nonfinite: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
            if index == [0, 0] {
                f64::NAN
            } else {
                0.0
            }
        })
        .unwrap();
    let error = nonfinite.eig_full(&[0], &[1]).unwrap_err();
    assert_eq!(
        crate::report::cause::<symtn::typed::OperationError>(&error),
        Some(&symtn::typed::OperationError::InvalidArgument {
            message: "eig input components must be finite"
        }),
        "{error:?}"
    );
}

#[test]
fn checked_generic_eig_lazy_calls_leave_the_source_view_lazy() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
            [[1.0, -3.0], [1.0, 1.0]][index[0]][index[1]]
        })
        .unwrap();
    let lazy = source.adjoint().unwrap();
    assert!(
        symtn::typed::__network::network_reuse_class(&lazy, false)
            == symtn::typed::__network::NetworkReuseClass::LazyAdjoint
    );
    assert!(lazy.eig_full(&[0], &[1]).is_ok());
    assert!(
        symtn::typed::__network::network_reuse_class(&lazy, false)
            == symtn::typed::__network::NetworkReuseClass::LazyAdjoint
    );
}

/// A nondual compact diagonal's Hermitian eigenbasis bond is its own bond
/// (TensorKit `fuse(V) = V`), so `eigh_full` makes no provider query and a
/// failing or style-changing provider cannot fail it.
#[test]
fn checked_nondual_compact_diagonal_eigh_full_makes_no_provider_query() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    for fault in ["fail_algebra", "invalid_style"] {
        let provider = Arc::new(CheckedOnlyToy::new_product_probe(1));
        let bond = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
        let diagonal: TensorMap<_, f64> = TensorMap::diagonal(
            &runtime,
            &bond,
            [SectorSpectrum {
                sector: Label::X,
                values: vec![2.0, -1.0],
            }],
        )
        .unwrap();
        match fault {
            "fail_algebra" => provider.fail_algebra.store(true, Ordering::Relaxed),
            _ => provider.invalid_style.store(true, Ordering::Relaxed),
        }
        reset_provider_queries(&provider);
        let Eigh { d, v } = diagonal
            .eigh_full(&[0], &[1], HermitianTol::DEFAULT)
            .unwrap();
        assert_eq!(provider.queries_since_reset.load(Ordering::Relaxed), 0);
        assert_eq!(d.codomain(), diagonal.codomain());
        assert_eq!(v.codomain(), diagonal.codomain());
    }
}

/// The checked Generic eigh admission is the same shared check at the same
/// tolerance (#1987): `[[1, δ], [0, 2]]` passes exactly when `δ ≤ tol·√10`,
/// and `diag(1 + iδ, 2)` exactly when `δ ≤ tol·√5`.
#[test]
fn checked_generic_eigh_admits_at_the_given_hermitian_tolerance() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    for (tol, hermitian_tol) in [
        (f64::EPSILON.powf(0.75), HermitianTol::DEFAULT),
        (1.0e-3, HermitianTol::relative(1.0e-3).unwrap()),
    ] {
        for (factor, admitted) in [(0.97, true), (1.03, false)] {
            let delta = factor * tol * 10.0_f64.sqrt();
            let dense: TensorMap<_, f64> =
                TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| match index {
                    [0, 0] => 1.0,
                    [1, 0] => delta,
                    [1, 1] => 2.0,
                    _ => 0.0,
                })
                .unwrap();
            assert_eq!(dense.eigh_vals(&[0], &[1], hermitian_tol).is_ok(), admitted);
            assert_eq!(dense.eigh_full(&[0], &[1], hermitian_tol).is_ok(), admitted);
            let delta = factor * tol * 5.0_f64.sqrt();
            let diagonal: TensorMap<_, Complex64> = TensorMap::diagonal(
                &runtime,
                &leg,
                [SectorSpectrum {
                    sector: Label::X,
                    values: vec![Complex64::new(1.0, delta), Complex64::new(2.0, 0.0)],
                }],
            )
            .unwrap();
            assert_eq!(
                diagonal.eigh_vals(&[0], &[1], hermitian_tol).is_ok(),
                admitted
            );
            assert_eq!(
                diagonal.eigh_full(&[0], &[1], hermitian_tol).is_ok(),
                admitted
            );
        }
    }
}

#[test]
fn checked_generic_eig_order_matches_the_lexicographic_oracle_and_the_mf_route() {
    let _serial = crate::counting_alloc::serial();
    // What: #1985 on checked Generic. Dense and compact EIG/EIGH, full and
    // values-only, publish the order sorted here independently (EIG
    // lexicographic `(re, im)`, EIGH ascending), and dense EIG agrees with
    // the multiplicity-free route on the same block.
    use std::cmp::Ordering::{Equal, Greater, Less};
    use symtn::sector::{U1FusionRule, U1Irrep};

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 4)]).unwrap();
    let c = Complex64::new;
    // Equal real parts in a conjugate pair; |λ| order would be
    // [1+2i, 1-2i, -1.5, 0.5i].
    let diagonal = [c(1.0, 2.0), c(-1.5, 0.0), c(1.0, -2.0), c(0.0, 0.5)];
    let entry = |i: usize, j: usize| match i.cmp(&j) {
        Equal => diagonal[i],
        Less => c(0.25 * (i + 2 * j) as f64, -0.1 * j as f64),
        Greater => c(0.0, 0.0),
    };
    let mut want = diagonal.to_vec();
    want.sort_by(|a, b| (a.re, a.im).partial_cmp(&(b.re, b.im)).unwrap());

    let general: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
            entry(index[0], index[1])
        })
        .unwrap();
    let Eig { d, v } = general.eig_full(&[0], &[1]).unwrap();
    let got = d.diagview().unwrap().remove(0).values;
    numerics::assert_slices_close("checked eig_full order", &got, &want, 16);
    numerics::assert_slices_close(
        "checked eig_vals order",
        &general.eig_vals(&[0], &[1]).unwrap().remove(0).values,
        &want,
        16,
    );
    numerics::assert_slices_close(
        "checked t v = v d",
        general.compose(&v).unwrap().dense_data().unwrap(),
        v.compose(&d).unwrap().dense_data().unwrap(),
        16,
    );
    let u1 = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 4)]).unwrap();
    let mf: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&u1], [&u1], |_, index| entry(index[0], index[1]))
            .unwrap();
    let Eig { d: mf_d, .. } = mf.eig_full(&[0], &[1]).unwrap();
    numerics::assert_slices_close(
        "checked vs multiplicity-free eig",
        &got,
        &mf_d.diagview().unwrap().remove(0).values,
        16,
    );

    let compact: TensorMap<_, Complex64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: Label::X,
            values: diagonal.to_vec(),
        }],
    )
    .unwrap();
    let Eig { d, v } = compact.eig_full(&[0], &[1]).unwrap();
    assert_eq!(d.diagview().unwrap().remove(0).values, want);
    assert_eq!(compact.eig_vals(&[0], &[1]).unwrap().remove(0).values, want);
    assert_eq!(
        compact.compose(&v).unwrap().dense_data().unwrap(),
        v.compose(&d).unwrap().dense_data().unwrap()
    );

    let real = [2.0, -3.0, 0.5, -2.0];
    let compact: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: Label::X,
            values: real.to_vec(),
        }],
    )
    .unwrap();
    let ascending = vec![-3.0, -2.0, 0.5, 2.0];
    let Eigh { d, v } = compact
        .eigh_full(&[0], &[1], HermitianTol::DEFAULT)
        .unwrap();
    assert_eq!(d.diagview().unwrap().remove(0).values, ascending);
    assert_eq!(
        compact
            .eigh_vals(&[0], &[1], HermitianTol::DEFAULT)
            .unwrap()
            .remove(0)
            .values,
        ascending
    );
    assert_eq!(
        compact.compose(&v).unwrap().dense_data().unwrap(),
        v.compose(&d).unwrap().dense_data().unwrap()
    );
    let dense = compact.materialize().unwrap();
    numerics::assert_slices_close(
        "checked dense eigh order",
        &dense
            .eigh_vals(&[0], &[1], HermitianTol::DEFAULT)
            .unwrap()
            .remove(0)
            .values,
        &ascending,
        4,
    );
}
