//! `inner`, `norm`, `tr`, `axpby`, `scale` and `adjoint` on the bond
//! `unit ⊕ charged²`. The charged sector's quantum dimension `d` is 1 for Z2
//! and `1 + √2` for the toy, so the checked mode separates a weighted
//! reduction from an unweighted one.

use super::*;

/// `unit` block `[3]`, charged block `[[1, 2], [2, 3]]`.
fn hand_value<S: PartialEq>(unit: &S, coupled: &S, ij: &[usize]) -> f64 {
    if coupled == unit {
        3.0
    } else {
        (ij[0] + ij[1] + 1) as f64
    }
}

in_both_modes! {
    /// TensorKit's `dot`, `tr` and `norm(p)` weight each coupled block by its
    /// quantum dimension (`p = ∞` excepted), real and complex; `inner` is
    /// conjugate-linear in its first argument; an exponent outside
    /// TensorKit's domain is an invalid argument. Oracle: hand sums.
    fn inner_tr_and_norm_are_dimension_weighted<M>() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let provider = M::provider();
        let bond = GradedSpace::try_new(Arc::clone(&provider), [(M::unit(), 1), (M::charged(), 2)])
            .unwrap();
        let source: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&bond], [&bond], |trees, ij| {
                hand_value(&M::unit(), trees.coupled(), ij)
            })
            .unwrap();
        let complex = source.convert::<Complex64>(engine).unwrap();
        let d = M::dim(M::charged());
        // Squares 9 | 1 + 4 + 4 + 9, diagonal 3 | 1 + 3, cubes 27 | 1 + 8 + 8 + 27.
        let squared = 9.0 + 18.0 * d;
        numerics::assert_close("inner", source.inner(engine, &source).unwrap(), squared, 4);
        numerics::assert_close("tr", source.tr(engine).unwrap(), 3.0 + 4.0 * d, 4);
        numerics::assert_close(
            "c64 tr",
            complex.tr(engine).unwrap(),
            Complex64::new(3.0 + 4.0 * d, 0.0),
            4,
        );
        for (p, want) in [
            (1.0, 3.0 + 8.0 * d),
            (2.0, squared.sqrt()),
            (3.0, (27.0 + 44.0 * d).cbrt()),
            (f64::INFINITY, 3.0),
        ] {
            numerics::assert_close(&format!("norm({p})"), source.norm(engine, p).unwrap(), want, 4);
            numerics::assert_close(&format!("c64 norm({p})"), complex.norm(engine, p).unwrap(), want, 4);
        }
        for p in [0.0, f64::NAN] {
            assert!(
                matches!(facade_err(source.norm(engine, p)), symtn::typed::Error::InvalidArgument(_)),
                "norm({p})"
            );
        }

        let i = Complex64::new(0.0, 1.0);
        let rotated = complex.scale(engine, i).unwrap();
        numerics::assert_close(
            "<a, i a> = i |a|²",
            complex.inner(engine, &rotated).unwrap(),
            i * squared,
            4,
        );
        numerics::assert_close(
            "<i a, a> = -i |a|²",
            rotated.inner(engine, &complex).unwrap(),
            -i * squared,
            4,
        );
    }
}

in_both_modes! {
    /// `axpby(α, y, β)` is `α x + β y` and `scale(λ)` is `λ x` entry by entry,
    /// for real and genuinely complex coefficients; like TensorKit, a zero
    /// coefficient drops its operand, so a NaN or infinity under it does not
    /// propagate, also for `axpby_into` and `scale_assign`.
    fn axpby_and_scale_are_entrywise_and_drop_zero_scaled_operands<M>() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let provider = M::provider();
        let bond = GradedSpace::try_new(Arc::clone(&provider), [(M::unit(), 1), (M::charged(), 2)])
            .unwrap();
        let x: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&bond], [&bond], |trees, ij| {
                if trees.coupled() == &M::unit() {
                    -0.5
                } else {
                    [[f64::INFINITY, f64::NAN], [1.5, -2.0]][ij[0]][ij[1]]
                }
            })
            .unwrap();
        let y: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&bond], [&bond], |trees, ij| {
                if trees.coupled() == &M::unit() {
                    4.0
                } else {
                    [[3.0, 4.0], [f64::NAN, 0.5]][ij[0]][ij[1]]
                }
            })
            .unwrap();
        let scaled = |value: f64, factor: f64| if factor == 0.0 { 0.0 } else { value * factor };
        let same = |got: &[f64], want: &[f64]| {
            assert_eq!(got.len(), want.len());
            for (&got, &want) in got.iter().zip(want) {
                assert!((got.is_nan() && want.is_nan()) || got == want, "{got} != {want}");
            }
        };
        for (alpha, beta) in [(0.0, 1.0), (1.0, 0.0), (0.0, 0.0), (2.0, -1.0)] {
            let want: Vec<f64> = x
                .dense_data()
                .unwrap()
                .iter()
                .zip(y.dense_data().unwrap())
                .map(|(&a, &b)| scaled(a, alpha) + scaled(b, beta))
                .collect();
            same(x.axpby(engine, alpha, &y, beta).unwrap().dense_data().unwrap(), &want);
            let mut assigned = x.materialize(engine).unwrap();
            y.axpby_into(engine, &mut assigned, beta, alpha).unwrap();
            same(assigned.dense_data().unwrap(), &want);
        }
        for factor in [0.0, -2.5] {
            let want: Vec<f64> =
                x.dense_data().unwrap().iter().map(|&a| scaled(a, factor)).collect();
            same(x.scale(engine, factor).unwrap().dense_data().unwrap(), &want);
            let mut assigned = x.materialize(engine).unwrap();
            assigned.scale_assign(engine, factor).unwrap();
            same(assigned.dense_data().unwrap(), &want);
        }

        let finite: TensorMap<_, Complex64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&bond], [&bond], |_, ij| {
                Complex64::new(ij[0] as f64 + 1.0, 2.0 - ij[1] as f64)
            })
            .unwrap();
        let other = finite.scale(engine, Complex64::new(0.0, 1.0)).unwrap();
        let (alpha, beta) = (Complex64::new(0.5, -2.0), Complex64::new(-1.5, 0.25));
        let want: Vec<Complex64> = finite
            .dense_data()
            .unwrap()
            .iter()
            .zip(other.dense_data().unwrap())
            .map(|(&a, &b)| alpha * a + beta * b)
            .collect();
        assert_eq!(finite.axpby(engine, alpha, &other, beta).unwrap().dense_data().unwrap(), want);
        let factor = Complex64::new(0.25, 3.0);
        let want: Vec<Complex64> =
            finite.dense_data().unwrap().iter().map(|&a| factor * a).collect();
        assert_eq!(finite.scale(engine, factor).unwrap().dense_data().unwrap(), want);
    }
}

in_both_modes! {
    /// `axpby`, `axpby_into` and `inner` check the runtime before the space,
    /// and `tr` needs an endomorphism; every refusal precedes provider work
    /// and leaves the receiver's payload and trees untouched.
    fn binary_reductions_reject_runtime_then_space_and_tr_needs_an_endomorphism<M>() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let foreign_runtime = Runtime::builder().build().unwrap();
        let provider = M::provider();
        let narrow = GradedSpace::try_new(Arc::clone(&provider), [(M::charged(), 1)]).unwrap();
        let wide = GradedSpace::try_new(Arc::clone(&provider), [(M::charged(), 2)]).unwrap();
        let mut left: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&narrow], [&narrow], |_, _| 1.0)
                .unwrap();
        let foreign: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &foreign_runtime, [&wide], [&wide], |_, _| 2.0)
                .unwrap();
        let mismatched: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&wide], [&wide], |_, _| 2.0).unwrap();
        let non_endomorphism: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&wide], [&narrow], |_, _| 2.0).unwrap();
        let lazy_non_endomorphism = non_endomorphism.adjoint(engine).unwrap();
        let before = left.dense_data().unwrap().to_vec();
        let trees = |tensor: &TensorMap<_, f64>| {
            (0..tensor.subblock_count())
                .map(|index| tensor.subblock_fusion_trees(index).unwrap())
                .collect::<Vec<_>>()
        };
        let before_trees = trees(&left);

        M::reset_queries(&provider);
        for error in [
            facade_err(left.axpby(engine, 1.0, &foreign, 1.0)),
            facade_err(foreign.axpby_into(engine, &mut left, 1.0, 1.0)),
            facade_err(left.inner(engine, &foreign)),
        ] {
            assert!(matches!(error, symtn::typed::Error::RuntimeMismatch), "{error:?}");
        }
        for error in [
            facade_err(left.axpby(engine, 1.0, &mismatched, 1.0)),
            facade_err(mismatched.axpby_into(engine, &mut left, 1.0, 1.0)),
            facade_err(left.inner(engine, &mismatched)),
            facade_err(non_endomorphism.tr(engine)),
            facade_err(lazy_non_endomorphism.tr(engine)),
        ] {
            assert!(is_space_mismatch(&error, ""), "{error:?}");
        }
        M::assert_no_queries(&provider);
        assert_eq!(left.dense_data().unwrap(), before.as_slice());
        assert_eq!(trees(&left), before_trees);
    }
}

in_both_modes! {
    /// `adjoint` swaps codomain and domain, keeps the provider, is an
    /// involution on the payload, conjugates complex values (compared as
    /// multisets, since the transpose moves them), keeps the norm and
    /// conjugates the trace.
    fn adjoint_swaps_spaces_conjugates_and_keeps_reductions<M>() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let provider = M::provider();
        let leg = GradedSpace::try_new(Arc::clone(&provider), [(M::unit(), 1), (M::charged(), 2)])
            .unwrap();
        let mut next = 0.0;
        let tall: TensorMap<_, Complex64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg], [&leg], |_, _| {
                next += 1.0;
                Complex64::new(next, 0.5 - next * next / 7.0)
            })
            .unwrap();
        let adjoint = tall.adjoint(engine).unwrap();
        assert!(std::ptr::eq(adjoint.provider(), provider.as_ref()));
        assert_eq!(adjoint.codomain(), tall.domain());
        assert_eq!(adjoint.domain(), tall.codomain());
        let back = adjoint.adjoint(engine).unwrap();
        assert_eq!(back.codomain(), tall.codomain());
        assert_eq!(back.dense_data().unwrap(), tall.dense_data().unwrap());
        numerics::assert_close(
            "norm(A†)",
            adjoint.norm(engine, 2.0).unwrap(),
            tall.norm(engine, 2.0).unwrap(),
            4,
        );
        let sorted = |mut values: Vec<Complex64>| {
            values.sort_by(|a, b| a.re.total_cmp(&b.re).then(a.im.total_cmp(&b.im)));
            values
        };
        let got = sorted(adjoint.materialize(engine).unwrap().dense_data().unwrap().to_vec());
        let conjugated = sorted(tall.dense_data().unwrap().iter().map(|v| v.conj()).collect());
        assert_eq!(got, conjugated);
        assert_ne!(got, sorted(tall.dense_data().unwrap().to_vec()));

        let bond = leg;
        let endomorphism: TensorMap<_, Complex64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&bond], [&bond], |trees, ij| {
                Complex64::new(hand_value(&M::unit(), trees.coupled(), ij), ij[0] as f64 + 0.5)
            })
            .unwrap();
        let trace = endomorphism.tr(engine).unwrap();
        assert!(trace.im.abs() > 0.1);
        numerics::assert_close(
            "tr(A†) = conj tr(A)",
            endomorphism.adjoint(engine).unwrap().tr(engine).unwrap(),
            trace.conj(),
            4,
        );
    }
}
