use super::*;

#[test]
fn checked_generic_reductions_cover_real_complex_dense_payloads() {
    let _cache = cache_shared();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
            (indices.iter().sum::<usize>() + 1) as f64
        })
        .unwrap();
    let complex = source.convert::<Complex64>();
    // The 2x2 block of `X` holds `1, 2, 2, 3`, weighted by
    // `dim(X) = 1 + sqrt(2)` (TensorKit `linalg.jl:dot`, `tr`, `_norm`):
    // inner = dim(X) * (1 + 4 + 4 + 9), tr = dim(X) * (1 + 3).
    let dim_x = 1.0 + 2.0_f64.sqrt();
    numerics::assert_close("inner", source.inner(&source).unwrap(), dim_x * 18.0, 4);
    numerics::assert_close("tr", source.tr().unwrap(), dim_x * 4.0, 2);
    numerics::assert_close(
        "c64 inner",
        complex.inner(&complex).unwrap(),
        Complex64::new(dim_x * 18.0, 0.0),
        4,
    );
    numerics::assert_close(
        "c64 tr",
        complex.tr().unwrap(),
        Complex64::new(dim_x * 4.0, 0.0),
        2,
    );
    assert!(provider.coefficient_queries.load(Ordering::Relaxed) > 0);
    // Every TensorKit exponent (`linalg.jl:_norm`); `Inf` is not weighted.
    // Only an exponent outside TensorKit's domain is rejected.
    for (p, want) in [
        (1.0, dim_x * 8.0),
        (2.0, (dim_x * 18.0).sqrt()),
        (3.0, (dim_x * 44.0_f64).cbrt()),
        (f64::INFINITY, 3.0),
    ] {
        numerics::assert_close(&format!("norm({p})"), source.norm(p).unwrap(), want, 4);
        numerics::assert_close(&format!("c64 norm({p})"), complex.norm(p).unwrap(), want, 4);
    }
    for p in [0.0, f64::NAN] {
        assert!(
            matches!(
                source.norm(p),
                Err(GenericTensorError::Facade(
                    symtn::typed::Error::InvalidArgument(_)
                ))
            ),
            "checked Generic norm({p})"
        );
    }
}

#[test]
fn checked_generic_compact_reductions_read_only_the_spectrum() {
    // What: a checked Generic compact operand reduces from its stored values,
    // weighted by `dim(X) = 1 + sqrt(2)`, against compact, dense and lazy
    // partners; the dense partner's off-diagonal entries are not read.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let compact = TensorMap::<_, Complex64>::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: Label::X,
            values: vec![Complex64::new(2.0, 1.0), Complex64::new(-1.0, 0.5)],
        }],
    )
    .unwrap();
    let dense: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
            match (indices[0], indices[1]) {
                (0, 0) => Complex64::new(1.0, -1.0),
                (1, 1) => Complex64::new(0.5, 2.0),
                _ => Complex64::new(f64::NAN, f64::NAN),
            }
        })
        .unwrap();
    let dim_x = 1.0 + 2.0_f64.sqrt();
    let (a, d) = (
        [Complex64::new(2.0, 1.0), Complex64::new(-1.0, 0.5)],
        [Complex64::new(1.0, -1.0), Complex64::new(0.5, 2.0)],
    );
    let inner = dim_x * (a[0].conj() * d[0] + a[1].conj() * d[1]);
    let terms = 4;
    numerics::assert_close(
        "compact.inner(dense)",
        compact.inner(&dense).unwrap(),
        inner,
        terms,
    );
    numerics::assert_close(
        "dense.inner(compact)",
        dense.inner(&compact).unwrap(),
        inner.conj(),
        terms,
    );
    // `dense^H` has diagonal `conj(d)`.
    let lazy = dense.adjoint().unwrap();
    numerics::assert_close(
        "compact.inner(lazy)",
        compact.inner(&lazy).unwrap(),
        dim_x * (a[0].conj() * d[0].conj() + a[1].conj() * d[1].conj()),
        terms,
    );
    let self_inner = dim_x * (a[0].norm_sqr() + a[1].norm_sqr());
    numerics::assert_close(
        "compact.inner(compact)",
        compact.inner(&compact).unwrap(),
        Complex64::from(self_inner),
        terms,
    );
    numerics::assert_close(
        "norm(2)",
        compact.norm(2.0).unwrap(),
        self_inner.sqrt(),
        terms,
    );
    numerics::assert_close(
        "norm(1)",
        compact.norm(1.0).unwrap(),
        dim_x * (a[0].norm() + a[1].norm()),
        terms,
    );
    numerics::assert_close(
        "norm(3)",
        compact.norm(3.0).unwrap(),
        (dim_x * (a[0].norm().powi(3) + a[1].norm().powi(3))).cbrt(),
        terms,
    );
    numerics::assert_close(
        "norm(Inf)",
        compact.norm(f64::INFINITY).unwrap(),
        a[0].norm(),
        terms,
    );
    numerics::assert_close("tr", compact.tr().unwrap(), dim_x * (a[0] + a[1]), terms);

    // A failing `dim` surfaces as the provider's error, as on dense input.
    provider.fail_dim.store(true, Ordering::Relaxed);
    for result in [
        compact.inner(&compact).map(|_| ()),
        compact.inner(&dense).map(|_| ()),
        compact.norm(2.0).map(|_| ()),
        compact.norm(1.0).map(|_| ()),
        compact.tr().map(|_| ()),
    ] {
        assert!(matches!(
            result.unwrap_err(),
            GenericTensorError::Structure(CheckedGenericStructureError::Provider(
                ToyError::Algebra
            ))
        ));
    }
    provider.fail_dim.store(false, Ordering::Relaxed);
}

#[test]
fn checked_generic_reductions_do_not_requery_the_admitted_fusion_style() {
    let _cache = cache_shared();
    // Admission fixed the Generic style; a provider whose answer changes
    // afterwards is neither asked again nor able to drop the `dim` weights.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
            (indices.iter().sum::<usize>() + 1) as f64
        })
        .unwrap();
    let inner = source.inner(&source).unwrap();
    let norm = source.norm(2.0).unwrap();
    provider.style_queries.store(0, Ordering::Relaxed);
    provider.invalid_style.store(true, Ordering::Relaxed);
    assert_eq!(source.inner(&source).unwrap().to_bits(), inner.to_bits());
    assert_eq!(source.norm(2.0).unwrap().to_bits(), norm.to_bits());
    assert_eq!(provider.style_queries.load(Ordering::Relaxed), 0);
}

#[test]
fn checked_generic_host_add_scale_cover_real_and_complex_payloads() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
            (indices.iter().sum::<usize>() + 1) as f64
        })
        .unwrap();
    let added = source.axpby(2.0, &source, -1.0).unwrap();
    assert_eq!(added.dense_data().unwrap(), source.dense_data().unwrap());
    let scaled = source.scale(3.0).unwrap();
    assert!(scaled
        .dense_data()
        .unwrap()
        .iter()
        .zip(source.dense_data().unwrap())
        .all(|(a, b)| (*a - 3.0 * *b).abs() < 1e-12));

    let complex = source.convert::<Complex64>();
    let added = complex
        .axpby(
            Complex64::new(2.0, 0.0),
            &complex,
            Complex64::new(-1.0, 0.0),
        )
        .unwrap();
    assert_eq!(added.dense_data().unwrap(), complex.dense_data().unwrap());
    let scaled = complex.scale(Complex64::new(0.5, -1.0)).unwrap();
    assert!(scaled
        .dense_data()
        .unwrap()
        .iter()
        .zip(complex.dense_data().unwrap())
        .all(|(a, b)| (*a - *b * Complex64::new(0.5, -1.0)).norm() < 1e-12));
}

#[test]
fn checked_generic_add_rejects_runtime_before_layout_without_queries() {
    let _cache = cache_shared();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let foreign_runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let narrow = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let wide = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let left: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&narrow], [&narrow], |_, _| 1.0).unwrap();
    let right: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&foreign_runtime, [&wide], [&wide], |_, _| 2.0).unwrap();
    reset_provider_queries(&provider);
    let before = left.dense_data().unwrap().to_vec();
    let error = left.axpby(1.0, &right, 1.0).unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Facade(symtn::typed::Error::RuntimeMismatch)
    ));
    assert_eq!(left.dense_data().unwrap(), before.as_slice());
    assert_eq!(provider.algebra_queries.load(Ordering::Relaxed), 0);
    assert_eq!(provider.coefficient_queries.load(Ordering::Relaxed), 0);
}

#[test]
fn checked_generic_add_rejects_layout_mismatch_without_queries() {
    let _cache = cache_shared();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let narrow = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let wide = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let left: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&narrow], [&narrow], |_, _| 1.0).unwrap();
    let right: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&wide], [&wide], |_, _| 2.0).unwrap();
    reset_provider_queries(&provider);
    let before = left.dense_data().unwrap().to_vec();
    let error = left.axpby(1.0, &right, 1.0).unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Facade(symtn::typed::Error::Operation(operation))
            if matches!(*operation, symtn::typed::OperationError::SpaceMismatch { .. })
    ));
    assert_eq!(left.dense_data().unwrap(), before.as_slice());
    assert_eq!(provider.algebra_queries.load(Ordering::Relaxed), 0);
    assert_eq!(provider.coefficient_queries.load(Ordering::Relaxed), 0);
}

#[test]
fn checked_generic_add_assign_rejects_runtime_before_layout_and_preserves_receiver() {
    let _cache = cache_shared();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let foreign_runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let narrow = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let wide = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let mut left: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&narrow], [&narrow], |_, _| 1.0).unwrap();
    let right: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&foreign_runtime, [&wide], [&wide], |_, _| 2.0).unwrap();
    let before_data = left.dense_data().unwrap().to_vec();
    let before_trees = (0..left.subblock_count())
        .map(|index| left.subblock_fusion_trees(index).unwrap())
        .collect::<Vec<_>>();
    reset_provider_queries(&provider);
    let error = right.axpby_into(&mut left, 1.0, 1.0).unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Facade(symtn::typed::Error::RuntimeMismatch)
    ));
    assert_eq!(left.dense_data().unwrap(), before_data.as_slice());
    assert_eq!(
        (0..left.subblock_count())
            .map(|index| left.subblock_fusion_trees(index).unwrap())
            .collect::<Vec<_>>(),
        before_trees
    );
    assert_eq!(provider.algebra_queries.load(Ordering::Relaxed), 0);
    assert_eq!(provider.coefficient_queries.load(Ordering::Relaxed), 0);
}

#[test]
fn checked_generic_add_assign_rejects_layout_mismatch_and_preserves_receiver() {
    let _cache = cache_shared();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let narrow = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let wide = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let mut left: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&narrow], [&narrow], |_, _| 1.0).unwrap();
    let right: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&wide], [&wide], |_, _| 2.0).unwrap();
    let before_data = left.dense_data().unwrap().to_vec();
    let before_trees = (0..left.subblock_count())
        .map(|index| left.subblock_fusion_trees(index).unwrap())
        .collect::<Vec<_>>();
    reset_provider_queries(&provider);
    let error = right.axpby_into(&mut left, 1.0, 1.0).unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Facade(symtn::typed::Error::Operation(operation))
            if matches!(*operation, symtn::typed::OperationError::SpaceMismatch { .. })
    ));
    assert_eq!(left.dense_data().unwrap(), before_data.as_slice());
    assert_eq!(
        (0..left.subblock_count())
            .map(|index| left.subblock_fusion_trees(index).unwrap())
            .collect::<Vec<_>>(),
        before_trees
    );
    assert_eq!(provider.algebra_queries.load(Ordering::Relaxed), 0);
    assert_eq!(provider.coefficient_queries.load(Ordering::Relaxed), 0);
}

#[test]
fn checked_generic_lazy_adjoint_preserves_provider_and_reductions() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
            (indices.iter().sum::<usize>() + 1) as f64
        })
        .unwrap();

    let adjoint = source.adjoint().unwrap();
    assert!(std::ptr::eq(adjoint.provider(), provider.as_ref()));
    assert!((adjoint.norm(2.0).unwrap() - source.norm(2.0).unwrap()).abs() < 1.0e-12);
    assert!((adjoint.tr().unwrap() - source.tr().unwrap()).abs() < 1.0e-12);

    let complex = source.convert::<Complex64>();
    let complex_adjoint = complex.adjoint().unwrap();
    assert!(std::ptr::eq(complex_adjoint.provider(), provider.as_ref()));
    assert!((complex_adjoint.tr().unwrap() - complex.tr().unwrap().conj()).norm() < 1.0e-12);
}

#[test]
fn checked_generic_reduction_dimension_failure_is_typed_and_nonpublishing() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, _| 2.0).unwrap();
    let before = source.dense_data().unwrap().to_vec();
    provider.fail_algebra.store(true, Ordering::Relaxed);
    let error = source.norm(2.0).unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Structure(CheckedGenericStructureError::Provider(ToyError::Algebra))
    ));
    assert_eq!(source.dense_data().unwrap(), before.as_slice());
}

#[cfg(feature = "cgc-gen")]
#[test]
fn sun_checked_generic_adjoint_and_reductions_preserve_provider_and_errors() {
    use symtn::sector::SUNFusionRule;
    use symtn::sector::SUNFusionRuleError;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    for (n, adjoint) in [(3, vec![1, 1]), (4, vec![1, 0, 1])] {
        let provider = Arc::new(SUNFusionRule::new(n).unwrap());
        let leg = GradedSpace::try_new(Arc::clone(&provider), [(adjoint, 1)]).unwrap();
        let source: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |trees, _| {
                trees.coupled().iter().sum::<i64>() as f64 + 1.0
            })
            .unwrap();

        let dagger = source.adjoint().unwrap();
        assert!(std::ptr::eq(dagger.provider(), provider.as_ref()));
        assert!((dagger.norm(2.0).unwrap() - source.norm(2.0).unwrap()).abs() < 1.0e-12);
        assert!((dagger.inner(&dagger).unwrap() - source.inner(&source).unwrap()).abs() < 1.0e-12);
        assert!((dagger.tr().unwrap() - source.tr().unwrap()).abs() < 1.0e-12);

        let complex = source.convert::<Complex64>();
        let complex_dagger = complex.adjoint().unwrap();
        assert!(std::ptr::eq(complex_dagger.provider(), provider.as_ref()));
        assert!((complex_dagger.tr().unwrap() - complex.tr().unwrap().conj()).norm() < 1.0e-12);
    }

    let provider = SUNFusionRule::new(3).unwrap();
    let three = provider.encode_dynkin(&[1, 0]).unwrap();
    let eight = provider.encode_dynkin(&[1, 1]).unwrap();
    assert!(matches!(
        provider.try_r_symbol_generic(three, three, eight),
        Err(SUNFusionRuleError::Racah(_))
    ));
}

/// TensorKit's zero-scale rule on the checked-generic `add`/`scale`
/// (VectorInterface `scale(x, α) = (iszero(α) ? zero(x) : x) * α`, observed on
/// TensorKit 0.17.1 as in `typed_zero_scale.rs`): a zero coefficient drops its
/// operand, NaN and `Inf` included (#1442).
#[test]
fn checked_generic_add_and_scale_drop_zero_scaled_operands_as_tensorkit() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let x: TensorMap<_, f64> = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, ij| {
        [[f64::INFINITY, f64::NAN], [1.5, -2.0]][ij[0]][ij[1]]
    })
    .unwrap();
    let y: TensorMap<_, f64> = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, ij| {
        [[3.0, 4.0], [f64::NAN, 0.5]][ij[0]][ij[1]]
    })
    .unwrap();
    let scale = |value: f64, factor: f64| if factor == 0.0 { 0.0 } else { value * factor };
    let same = |got: &[f64], want: &[f64]| {
        assert_eq!(got.len(), want.len());
        for (&got, &want) in got.iter().zip(want) {
            assert!(
                (got.is_nan() && want.is_nan()) || got == want,
                "{got} != {want}"
            );
        }
    };
    for (alpha, beta) in [(0.0, 1.0), (1.0, 0.0), (0.0, 0.0), (2.0, -1.0)] {
        let want: Vec<f64> = x
            .dense_data()
            .unwrap()
            .iter()
            .zip(y.dense_data().unwrap())
            .map(|(&a, &b)| scale(a, alpha) + scale(b, beta))
            .collect();
        same(
            x.axpby(alpha, &y, beta).unwrap().dense_data().unwrap(),
            &want,
        );
        let mut assigned = x.materialize().unwrap();
        y.axpby_into(&mut assigned, beta, alpha).unwrap();
        same(assigned.dense_data().unwrap(), &want);
    }
    for factor in [0.0, 2.0] {
        let want: Vec<f64> = x
            .dense_data()
            .unwrap()
            .iter()
            .map(|&a| scale(a, factor))
            .collect();
        same(x.scale(factor).unwrap().dense_data().unwrap(), &want);
        let mut assigned = x.materialize().unwrap();
        assigned.scale_assign(factor).unwrap();
        same(assigned.dense_data().unwrap(), &want);
    }
}

#[test]
fn checked_tr_matches_the_literal_weighted_sum_and_keeps_error_precedence() {
    let _cache = cache_shared();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 1), (Label::X, 2)]).unwrap();
    // 2+2 endomorphism: diagonal blocks exist per (coupled, vertices) pair and
    // off-diagonal multiplicity blocks must not contribute.
    let complex =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], lazy_oracle_value)
            .unwrap();
    assert!((0..complex.subblock_count()).any(|index| {
        let trees = complex.subblock_fusion_trees(index).unwrap();
        trees.codomain_vertices() != trees.domain_vertices()
    }));
    let dim = |label: &Label| match label {
        Label::Vacuum => 1.0,
        Label::X => 1.0 + 2.0_f64.sqrt(),
        _ => unreachable!(),
    };

    let mut expected = Complex64::new(0.0, 0.0);
    common::literal_weighted_trace(&snapshot!(complex), dim, |value, weight| {
        expected += value * weight
    });
    assert!(expected.im.abs() > 1e-6);
    let lazy = complex.adjoint().unwrap();
    let sectors = (0..complex.subblock_count())
        .map(|index| *complex.subblock_fusion_trees(index).unwrap().coupled())
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    provider.coefficient_queries.store(0, Ordering::Relaxed);
    assert!(lazy_close_c64(complex.tr().unwrap(), expected));
    assert_eq!(
        provider.coefficient_queries.swap(0, Ordering::Relaxed),
        sectors
    );
    assert!(lazy_close_c64(lazy.tr().unwrap(), expected.conj()));
    assert_eq!(
        provider.coefficient_queries.swap(0, Ordering::Relaxed),
        sectors
    );

    let real = complex.re();
    let mut expected_real = 0.0;
    common::literal_weighted_trace(&snapshot!(real), dim, |value, weight| {
        expected_real += value * weight
    });
    let real_lazy = real.adjoint().unwrap();
    provider.coefficient_queries.store(0, Ordering::Relaxed);
    assert!(lazy_close_f64(real.tr().unwrap(), expected_real));
    assert_eq!(
        provider.coefficient_queries.swap(0, Ordering::Relaxed),
        sectors
    );
    assert!(lazy_close_f64(real_lazy.tr().unwrap(), expected_real));
    assert_eq!(
        provider.coefficient_queries.swap(0, Ordering::Relaxed),
        sectors
    );
    let mut unweighted = 0.0;
    common::literal_weighted_trace(
        &snapshot!(real),
        |_| 1.0,
        |value, weight| unweighted += value * weight,
    );
    assert!((unweighted - expected_real).abs() > 1e-6);

    // Error precedence: endomorphism check before any provider query, then the
    // provider's dim failure, for owned and lazy inputs alike.
    let non_endomorphism =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg], lazy_oracle_value).unwrap();
    provider.fail_dim.store(true, Ordering::Relaxed);
    provider.coefficient_queries.store(0, Ordering::Relaxed);
    for tensor in [
        non_endomorphism.clone(),
        non_endomorphism.adjoint().unwrap(),
    ] {
        assert!(matches!(
            tensor.tr().unwrap_err(),
            GenericTensorError::Facade(symtn::typed::Error::Operation(operation))
                if matches!(*operation, symtn::typed::OperationError::SpaceMismatch { .. })
        ));
    }
    assert_eq!(provider.coefficient_queries.load(Ordering::Relaxed), 0);
    for tensor in [complex.clone(), lazy.clone()] {
        assert!(matches!(
            tensor.tr().unwrap_err(),
            GenericTensorError::Structure(CheckedGenericStructureError::Provider(
                ToyError::Algebra
            ))
        ));
    }
    provider.fail_dim.store(false, Ordering::Relaxed);
    assert!(lazy_close_c64(complex.tr().unwrap(), expected));
}

#[test]
fn checked_inner_and_norm_take_one_weight_per_sector_and_keep_error_precedence() {
    let _cache = cache_shared();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 1), (Label::X, 2)]).unwrap();
    // 2+2 with outer multiplicity: every coupled sector owns several blocks
    // (B > G), so a per-block weight lookup would be observable as extra
    // provider queries below.
    let lhs = TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], lazy_oracle_value)
        .unwrap();
    let rhs =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, indices| {
            lazy_oracle_value(trees, indices) * Complex64::new(0.5, -1.5) + Complex64::new(1.0, 2.0)
        })
        .unwrap();
    let sectors = (0..lhs.subblock_count())
        .map(|index| *lhs.subblock_fusion_trees(index).unwrap().coupled())
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    assert_eq!(sectors, 2);
    assert!(lhs.subblock_count() > sectors);
    let dim = |label: &Label| match label {
        Label::Vacuum => 1.0,
        Label::X => 1.0 + 2.0_f64.sqrt(),
        _ => unreachable!(),
    };
    let literal_c64 = |a: &TensorMap<CheckedOnlyToy, Complex64>,
                       b: &TensorMap<CheckedOnlyToy, Complex64>| {
        let mut expected = Complex64::new(0.0, 0.0);
        literal_weighted_inner(&snapshot!(a), &snapshot!(b), dim, |x, y, weight| {
            expected += x.conj() * y * weight
        });
        expected
    };
    let literal_f64 = |a: &TensorMap<CheckedOnlyToy, f64>, b: &TensorMap<CheckedOnlyToy, f64>| {
        let mut expected = 0.0;
        literal_weighted_inner(&snapshot!(a), &snapshot!(b), dim, |x, y, weight| {
            expected += x * y * weight
        });
        expected
    };

    // Owned, complex and real.
    let expected = literal_c64(&lhs, &rhs);
    assert!(expected.im.abs() > 1e-6);
    assert!(lazy_close_c64(lhs.inner(&rhs).unwrap(), expected));
    assert!(lazy_close_c64(rhs.inner(&lhs).unwrap(), expected.conj()));
    assert!(lazy_close_f64(
        lhs.norm(2.0).unwrap(),
        literal_c64(&lhs, &lhs).re.sqrt()
    ));
    let real_lhs = lhs.re();
    let real_rhs = rhs.re();
    assert!(lazy_close_f64(
        real_lhs.inner(&real_rhs).unwrap(),
        literal_f64(&real_lhs, &real_rhs)
    ));
    assert!(lazy_close_f64(
        real_lhs.norm(2.0).unwrap(),
        literal_f64(&real_lhs, &real_lhs).sqrt()
    ));
    let mut unweighted = Complex64::new(0.0, 0.0);
    literal_weighted_inner(
        &snapshot!(lhs),
        &snapshot!(rhs),
        |_| 1.0,
        |x, y, w| unweighted += x.conj() * y * w,
    );
    assert!((unweighted - expected).norm() > 1e-6);

    // Lazy adjoints in every orientation: the literal oracle on the
    // materialized lazy payload, and <a†, b†> = conj(<a, b>) against owned.
    let lazy_lhs = lhs.adjoint().unwrap();
    let lazy_rhs = rhs.adjoint().unwrap();
    let expected_lazy = literal_c64(&lazy_lhs, &lazy_rhs);
    assert!(lazy_close_c64(expected_lazy, expected.conj()));
    assert!(lazy_close_c64(
        lazy_lhs.inner(&lazy_rhs).unwrap(),
        expected_lazy
    ));
    let expected_mixed = literal_c64(&lazy_lhs, &rhs);
    assert!(lazy_close_c64(
        lazy_lhs.inner(&rhs).unwrap(),
        expected_mixed
    ));
    assert!(lazy_close_c64(
        rhs.inner(&lazy_lhs).unwrap(),
        expected_mixed.conj()
    ));
    assert!(lazy_close_f64(
        lazy_lhs.norm(2.0).unwrap(),
        lhs.norm(2.0).unwrap()
    ));
    let real_lazy = real_lhs.adjoint().unwrap();
    assert!(lazy_close_f64(
        real_lazy.inner(&real_rhs).unwrap(),
        literal_f64(&real_lazy, &real_rhs)
    ));

    // Exactly G `dim` queries per call on the dense and the oriented owners.
    // The former per-call map hashed 3G (dense) or 2G + B (lazy) times; hashes
    // are not provider queries, so only the query count is pinned here.
    for (row, call) in [
        (
            "owned inner",
            Box::new(|| lhs.inner(&rhs).map(|_| ())) as Box<dyn Fn() -> _>,
        ),
        ("owned norm", Box::new(|| lhs.norm(2.0).map(|_| ()))),
        (
            "lazy-lazy inner",
            Box::new(|| lazy_lhs.inner(&lazy_rhs).map(|_| ())),
        ),
        (
            "lazy-owned inner",
            Box::new(|| lazy_lhs.inner(&rhs).map(|_| ())),
        ),
        (
            "owned-lazy inner",
            Box::new(|| rhs.inner(&lazy_lhs).map(|_| ())),
        ),
        ("lazy norm", Box::new(|| lazy_lhs.norm(2.0).map(|_| ()))),
    ] {
        provider.coefficient_queries.store(0, Ordering::Relaxed);
        call().unwrap();
        assert_eq!(
            provider.coefficient_queries.load(Ordering::Relaxed),
            sectors,
            "{row}"
        );
    }

    // Error precedence: a space mismatch comes before any weight query even
    // while `dim` is failing; then the provider failure surfaces for compact,
    // owned and lazy inputs on both reductions.
    let wide = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 3)]).unwrap();
    let mismatched =
        TensorMap::from_subblock_fn(&runtime, [&wide, &wide], [&wide, &wide], lazy_oracle_value)
            .unwrap();
    let diagonal = TensorMap::<_, Complex64>::diagonal(
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: Label::Vacuum,
                values: vec![Complex64::new(3.0, 0.0)],
            },
            SectorSpectrum {
                sector: Label::X,
                values: vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            },
        ],
    )
    .unwrap();
    provider.fail_dim.store(true, Ordering::Relaxed);
    provider.coefficient_queries.store(0, Ordering::Relaxed);
    for (a, b) in [
        (&lhs, &mismatched),
        (&mismatched, &lhs),
        (&lazy_lhs, &mismatched),
        (&mismatched, &lazy_lhs),
    ] {
        assert!(matches!(
            a.inner(b).unwrap_err(),
            GenericTensorError::Facade(symtn::typed::Error::Operation(operation))
                if matches!(*operation, symtn::typed::OperationError::SpaceMismatch { .. })
        ));
    }
    assert_eq!(provider.coefficient_queries.load(Ordering::Relaxed), 0);
    // A compact diagonal reduces through the same weights (#1867), so it
    // fails exactly as the dense and lazy inputs do.
    assert!(matches!(
        diagonal.inner(&diagonal).unwrap_err(),
        GenericTensorError::Structure(CheckedGenericStructureError::Provider(ToyError::Algebra))
    ));
    assert!(matches!(
        diagonal.norm(2.0).unwrap_err(),
        GenericTensorError::Structure(CheckedGenericStructureError::Provider(ToyError::Algebra))
    ));
    for tensor in [&lhs, &lazy_lhs] {
        assert!(matches!(
            tensor.inner(&rhs).unwrap_err(),
            GenericTensorError::Structure(CheckedGenericStructureError::Provider(
                ToyError::Algebra
            ))
        ));
        assert!(matches!(
            tensor.norm(2.0).unwrap_err(),
            GenericTensorError::Structure(CheckedGenericStructureError::Provider(
                ToyError::Algebra
            ))
        ));
    }
    provider.fail_dim.store(false, Ordering::Relaxed);
    assert!(lazy_close_c64(lhs.inner(&rhs).unwrap(), expected));
    assert!(lazy_close_c64(
        lazy_lhs.inner(&lazy_rhs).unwrap(),
        expected_lazy
    ));
}

/// TensorKit `inner` (`vectorinterface.jl`): `sum_c dim(c) * <a_c, b_c>` over
/// every stored block with the first argument conjugated, walked with an
/// explicit odometer over public block geometry so the expected value never
/// passes through the coupled-region or oriented owners under test.
fn literal_weighted_inner<D: Copy>(
    lhs: &common::Snapshot<Label, D>,
    rhs: &common::Snapshot<Label, D>,
    dim: impl Fn(&Label) -> f64,
    mut accumulate: impl FnMut(D, D, f64),
) {
    assert_eq!(lhs.blocks.len(), rhs.blocks.len());
    for ((trees, geometry), (other_trees, other_geometry)) in lhs.blocks.iter().zip(&rhs.blocks) {
        assert_eq!(trees, other_trees);
        assert_eq!(geometry.shape, other_geometry.shape);
        let weight = dim(trees.coupled());
        common::for_each_index(&geometry.shape, |index| {
            accumulate(
                lhs.data[common::linear(geometry, index)],
                rhs.data[common::linear(other_geometry, index)],
                weight,
            );
        });
    }
}
