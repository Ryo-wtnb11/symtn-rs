//! `inv`, `pinv`, `exp` and `map_diagonal`. Every bond has a unit and a
//! charged sector, stored in that order, each block column-major.

use super::*;

fn dense_close<T: numerics::Numeric>(what: &str, got: &[T], want: &[T]) {
    numerics::assert_slices_close(what, got, want, 8);
}

fn is_numerical_failure(error: &symtn::typed::Error) -> bool {
    matches!(
        error,
        symtn::typed::Error::Operation(error) if matches!(
            **error,
            OperationError::Dense(symtn_dense::DenseError::NumericalFailure { .. })
        )
    )
}

fn is_space_mismatch(error: &symtn::typed::Error, reason: &str) -> bool {
    matches!(
        error,
        symtn::typed::Error::Operation(error) if matches!(
            &**error,
            OperationError::SpaceMismatch { message } if message.contains(reason)
        )
    )
}

/// The four Moore-Penrose identities, by dense payload.
macro_rules! assert_moore_penrose {
    ($engine:ident, $a:expr, $p:expr) => {{
        let a = $a;
        let p = $p;
        let a_p = a.compose($engine, p).unwrap();
        let p_a = p.compose($engine, a).unwrap();
        dense_close(
            "A A+ A = A",
            a_p.compose($engine, a).unwrap().dense_data().unwrap(),
            a.dense_data().unwrap(),
        );
        dense_close(
            "A+ A A+ = A+",
            p_a.compose($engine, p).unwrap().dense_data().unwrap(),
            p.dense_data().unwrap(),
        );
        for (name, projector) in [("(A A+)† = A A+", &a_p), ("(A+ A)† = A+ A", &p_a)] {
            dense_close(
                name,
                projector
                    .adjoint($engine)
                    .unwrap()
                    .materialize($engine)
                    .unwrap()
                    .dense_data()
                    .unwrap(),
                projector.dense_data().unwrap(),
            );
        }
    }};
}

// ---------------------------------------------------------------------------
// `inv`.
// ---------------------------------------------------------------------------

in_both_modes! {
    /// `inv` is the exact inverse of every coupled block, so negative integer
    /// powers are compositions of it. Oracle: hand-computed powers of `2·1`
    /// (unit) and `[[2,1],[3,4]]` (charged), real and times `1 + i`.
    fn inv_and_integer_powers_match_hand_oracles<M>() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let provider = M::provider();
        let bond =
            GradedSpace::try_new(Arc::clone(&provider), [(M::unit(), 2), (M::charged(), 2)])
                .unwrap();
        let real: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&bond], [&bond], |trees, ij| {
                if trees.coupled() == &M::unit() {
                    if ij[0] == ij[1] { 2.0 } else { 0.0 }
                } else {
                    [[2.0, 1.0], [3.0, 4.0]][ij[0]][ij[1]]
                }
            })
            .unwrap();
        let real_oracles: [(i32, [f64; 8]); 5] = [
            (1, [2.0, 0.0, 0.0, 2.0, 2.0, 3.0, 1.0, 4.0]),
            (2, [4.0, 0.0, 0.0, 4.0, 7.0, 18.0, 6.0, 19.0]),
            (3, [8.0, 0.0, 0.0, 8.0, 32.0, 93.0, 31.0, 94.0]),
            (-1, [0.5, 0.0, 0.0, 0.5, 0.8, -0.6, -0.2, 0.4]),
            (-2, [0.25, 0.0, 0.0, 0.25, 0.76, -0.72, -0.24, 0.28]),
        ];
        let inverse = real.inv(engine, &[0], &[1]).unwrap();
        assert_eq!(inverse.codomain(), real.domain());
        for (exponent, expected) in real_oracles {
            let base = if exponent < 0 { &inverse } else { &real };
            let mut power = base.clone();
            for _ in 1..exponent.unsigned_abs() {
                power = power.compose(engine, base).unwrap();
            }
            dense_close(&format!("A^{exponent}"), power.dense_data().unwrap(), &expected);
        }

        let c = Complex64::new;
        let complex = real
            .convert::<Complex64>(engine)
            .unwrap()
            .scale(engine, c(1.0, 1.0))
            .unwrap();
        // `(zA)^k = z^k A^k`: the complex oracle is the real one times `z^k`.
        let complex_oracles = real_oracles.map(|(exponent, expected)| {
            (exponent, expected.map(|value| c(1.0, 1.0).powi(exponent) * value))
        });
        let inverse = complex.inv(engine, &[0], &[1]).unwrap();
        for (exponent, expected) in complex_oracles {
            let base = if exponent < 0 { &inverse } else { &complex };
            let mut power = base.clone();
            for _ in 1..exponent.unsigned_abs() {
                power = power.compose(engine, base).unwrap();
            }
            dense_close(&format!("((1+i)A)^{exponent}"), power.dense_data().unwrap(), &expected);
        }
    }
}

in_both_modes! {
    /// TensorKit's `inv` asks for `codomain ≅ domain`, not `==`, and returns
    /// `domain <- codomain` on the source provider; a non-isomorphic pair is
    /// refused without touching the source.
    fn inv_accepts_isomorphic_unequal_spaces_and_rejects_nonisomorphic<M>() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let provider = M::provider();
        let x = GradedSpace::try_new(Arc::clone(&provider), [(M::unit(), 1), (M::charged(), 2)])
            .unwrap();
        let unit = GradedSpace::try_new(Arc::clone(&provider), [(M::unit(), 1)]).unwrap();
        let source: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&x], [&x, &unit], |_, ij| {
                if ij[0] == ij[1] { 2.0 } else { 0.0 }
            })
            .unwrap();
        let inverse = source.inv(engine, &[0], &[1, 2]).unwrap();
        assert_eq!((inverse.codomain_rank(), inverse.domain_rank()), (2, 1));
        assert_eq!(inverse.codomain(), source.domain());
        assert_eq!(inverse.domain(), source.codomain());
        assert!(std::ptr::eq(inverse.provider(), provider.as_ref()));
        // Hand oracle: the inverse of `2·1` is `1/2·1` on the swapped spaces.
        let expected: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&x, &unit], [&x], |_, ij| {
                if ij[0] == ij[2] { 0.5 } else { 0.0 }
            })
            .unwrap();
        dense_close(
            "inv(2·1)",
            inverse.dense_data().unwrap(),
            expected.dense_data().unwrap(),
        );

        let narrow =
            GradedSpace::try_new(Arc::clone(&provider), [(M::unit(), 1), (M::charged(), 1)])
                .unwrap();
        let nonisomorphic: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&narrow], [&x], |_, _| 1.0).unwrap();
        let before = nonisomorphic.dense_data().unwrap().to_vec();
        let error = facade_err(nonisomorphic.inv(engine, &[0], &[1]));
        assert!(is_space_mismatch(&error, "inv requires isomorphic"), "{error:?}");
        assert_eq!(nonisomorphic.dense_data().unwrap(), before.as_slice());
    }
}

in_both_modes! {
    /// A singular coupled block, first or last in storage order, is a
    /// `NumericalFailure` on the dense (LU) arm and on the compact
    /// (zero-entry) arm, and the source is left untouched.
    fn inv_of_a_singular_sector_fails_on_both_storages_and_keeps_the_source<M>() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let provider = M::provider();
        let bond =
            GradedSpace::try_new(Arc::clone(&provider), [(M::unit(), 1), (M::charged(), 1)])
                .unwrap();
        for target in [M::unit(), M::charged()] {
            let dense: TensorMap<_, f64> =
                TensorMap::from_subblock_fn(engine, &runtime, [&bond], [&bond], |trees, _| {
                    if trees.coupled() == &target { 0.0 } else { 1.0 }
                })
                .unwrap();
            let before = dense.dense_data().unwrap().to_vec();
            let error = facade_err(dense.inv(engine, &[0], &[1]));
            assert!(is_numerical_failure(&error), "dense {target:?}: {error:?}");
            assert_eq!(dense.dense_data().unwrap(), before.as_slice());

            let compact: TensorMap<_, f64> = TensorMap::diagonal(
                engine,
                &runtime,
                &bond,
                [M::unit(), M::charged()].map(|sector| SectorSpectrum {
                    values: vec![if sector == target { 0.0 } else { 1.0 }],
                    sector,
                }),
            )
            .unwrap();
            let error = facade_err(compact.inv(engine, &[0], &[1]));
            assert!(is_numerical_failure(&error), "compact {target:?}: {error:?}");
        }
    }
}

// ---------------------------------------------------------------------------
// `pinv`.
// ---------------------------------------------------------------------------

in_both_modes! {
    /// `rcond` must be finite and non-negative; it is validated before any
    /// provider work and before the finite-payload scan, for dense and
    /// compact receivers alike, and the receiver is untouched.
    fn pinv_rejects_a_bad_rcond_first_on_every_storage<M>() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let provider = M::provider();
        let bond =
            GradedSpace::try_new(Arc::clone(&provider), [(M::unit(), 1), (M::charged(), 2)])
                .unwrap();
        let dense: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&bond], [&bond], |_, ij| {
                (1 + ij[0] + 2 * ij[1]) as f64
            })
            .unwrap();
        let nan_payload = dense.scale(engine, f64::NAN).unwrap();
        let compact: TensorMap<_, f64> = TensorMap::diagonal(
            engine,
            &runtime,
            &bond,
            [
                SectorSpectrum { sector: M::unit(), values: vec![2.0] },
                SectorSpectrum { sector: M::charged(), values: vec![1.0, 0.0] },
            ],
        )
        .unwrap();
        for (name, receiver) in [("dense", &dense), ("NaN payload", &nan_payload)] {
            let before = receiver.dense_data().unwrap().to_vec();
            for rcond in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                M::reset_queries(&provider);
                match facade_err(receiver.pinv(engine, &[0], &[1], rcond)) {
                    symtn::typed::Error::InvalidArgument(message) => {
                        assert!(message.contains("rcond"), "{name}, {rcond}: {message}")
                    }
                    other => panic!("{name}, {rcond}: {other:?}"),
                }
                M::assert_no_queries(&provider);
            }
            let after = receiver.dense_data().unwrap();
            assert!(after.iter().zip(&before).all(|(a, b)| a.to_bits() == b.to_bits()));
        }
        for rcond in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                matches!(
                    facade_err(compact.pinv(engine, &[0], &[1], rcond)),
                    symtn::typed::Error::InvalidArgument(_)
                ),
                "compact accepted rcond {rcond}"
            );
        }
    }
}

in_both_modes! {
    /// The cutoff is `sigma > rcond * sigma_max` with one `sigma_max` over the
    /// whole tensor (not per sector, unlike TensorKit's per-block `rtol`).
    /// The global maximum, 4, lives in the second (charged) sector and the
    /// unit sector's only value, 1, sits exactly on the cutoff `0.25 * 4`:
    /// a per-sector, first-sector-only or `>=` cutoff would keep it.
    fn pinv_cuts_strictly_at_one_global_sigma_max_on_both_storages<M>() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let provider = M::provider();
        let bond =
            GradedSpace::try_new(Arc::clone(&provider), [(M::unit(), 1), (M::charged(), 2)])
                .unwrap();
        let dense: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&bond], [&bond], |trees, ij| {
                match (trees.coupled() == &M::unit(), ij[0] == ij[1], ij[0]) {
                    (true, _, _) => 1.0,
                    (false, true, 0) => 4.0,
                    (false, true, _) => 2.0,
                    (false, false, _) => 0.0,
                }
            })
            .unwrap();
        let pseudo = dense.pinv(engine, &[0], &[1], 0.25).unwrap();
        dense_close(
            "dense pinv",
            pseudo.dense_data().unwrap(),
            &[0.0, 0.25, 0.0, 0.0, 0.5],
        );

        let compact: TensorMap<_, f64> = TensorMap::diagonal(
            engine,
            &runtime,
            &bond,
            [
                SectorSpectrum { sector: M::unit(), values: vec![1.0] },
                SectorSpectrum { sector: M::charged(), values: vec![4.0, 2.0] },
            ],
        )
        .unwrap();
        let pseudo = compact.pinv(engine, &[0], &[1], 0.25).unwrap();
        let values: Vec<Vec<f64>> = pseudo
            .diagview(engine)
            .unwrap()
            .into_iter()
            .map(|entry| entry.values)
            .collect();
        assert_eq!(values, [vec![0.0], vec![0.25, 0.5]]);
    }
}

in_both_modes! {
    /// `pinv` of a wide, a tall and a complex map, each with a numerically
    /// rank-one charged block (so a broken cutoff cannot pass as an ordinary
    /// inverse), satisfies the four Moore-Penrose identities; a lazy adjoint's
    /// `pinv` is the adjoint of the parent's.
    fn pinv_satisfies_the_moore_penrose_identities<M>() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let provider = M::provider();
        let narrow =
            GradedSpace::try_new(Arc::clone(&provider), [(M::unit(), 2), (M::charged(), 2)])
                .unwrap();
        let wide_leg =
            GradedSpace::try_new(Arc::clone(&provider), [(M::unit(), 3), (M::charged(), 3)])
                .unwrap();
        let entry = |unit: bool, row: usize, col: usize| {
            if unit {
                [[1.0, 0.0, 1.0], [0.0, 2.0, 1.0]][row][col]
            } else {
                [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]][row][col]
            }
        };
        let wide: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&narrow], [&wide_leg], |trees, ij| {
                entry(trees.coupled() == &M::unit(), ij[0], ij[1])
            })
            .unwrap();
        let tall: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&wide_leg], [&narrow], |trees, ij| {
                entry(trees.coupled() == &M::unit(), ij[1], ij[0])
            })
            .unwrap();
        // Above the rank-one block's numerically-zero singular value, far
        // below every real one.
        let rcond = 1e-6;
        for source in [&wide, &tall] {
            let pseudo = source.pinv(engine, &[0], &[1], rcond).unwrap();
            assert_eq!(pseudo.codomain(), source.domain());
            assert_eq!(pseudo.domain(), source.codomain());
            assert!(std::ptr::eq(pseudo.provider(), provider.as_ref()));
            assert_moore_penrose!(engine, source, &pseudo);
        }
        let complex = wide
            .convert::<Complex64>(engine)
            .unwrap()
            .scale(engine, Complex64::new(1.0, 0.25))
            .unwrap();
        let complex_pseudo = complex.pinv(engine, &[0], &[1], rcond).unwrap();
        assert_moore_penrose!(engine, &complex, &complex_pseudo);

        let pseudo = wide.pinv(engine, &[0], &[1], rcond).unwrap();
        let lazy_pseudo = wide
            .adjoint(engine)
            .unwrap()
            .pinv(engine, &[0], &[1], rcond)
            .unwrap();
        dense_close(
            "pinv(A†) = pinv(A)†",
            lazy_pseudo.dense_data().unwrap(),
            pseudo
                .adjoint(engine)
                .unwrap()
                .materialize(engine)
                .unwrap()
                .dense_data()
                .unwrap(),
        );
    }
}

// ---------------------------------------------------------------------------
// Compact (diagonal) `inv` / `pinv`.
// ---------------------------------------------------------------------------

in_both_modes! {
    /// TensorKit's `inv` / `pinv(::DiagonalTensorMap)`: elementwise reciprocal
    /// (with the global cutoff for `pinv`) on the stored values, staying
    /// compact on the input bond. Real reciprocals are the IEEE ones; complex
    /// ones are within 1 ulp of `1 / z` (the port of Julia's complex `inv`
    /// orders the same formula differently).
    fn compact_inv_and_pinv_are_elementwise_reciprocals<M>() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let provider = M::provider();
        let bond =
            GradedSpace::try_new(Arc::clone(&provider), [(M::unit(), 2), (M::charged(), 2)])
                .unwrap();
        let values = |tensor: &TensorMap<_, f64>, engine: &mut symtn_dense::CpuEngine| {
            assert!(tensor.dense_data().is_err(), "the result is compact");
            tensor
                .diagview(engine)
                .unwrap()
                .into_iter()
                .map(|entry| entry.values)
                .collect::<Vec<_>>()
        };
        let real: TensorMap<_, f64> = TensorMap::diagonal(
            engine,
            &runtime,
            &bond,
            [
                SectorSpectrum { sector: M::unit(), values: vec![4.0, -3.0] },
                SectorSpectrum { sector: M::charged(), values: vec![0.5, 8.0] },
            ],
        )
        .unwrap();
        let inverse = real.inv(engine, &[0], &[1]).unwrap();
        assert_eq!(inverse.codomain(), real.codomain());
        assert_eq!(
            values(&inverse, engine),
            [vec![0.25, (-3.0f64).recip()], vec![2.0, 0.125]]
        );
        // sigma_max = 8: cutoffs 0, 0.8 and 3.6.
        for (rcond, expected) in [
            (0.0, [vec![0.25, (-3.0f64).recip()], vec![2.0, 0.125]]),
            (0.1, [vec![0.25, (-3.0f64).recip()], vec![0.0, 0.125]]),
            (0.45, [vec![0.25, 0.0], vec![0.0, 0.125]]),
        ] {
            let pseudo = real.pinv(engine, &[0], &[1], rcond).unwrap();
            assert_eq!(values(&pseudo, engine), expected, "rcond {rcond}");
        }

        let c = Complex64::new;
        let source = [c(3.0, 4.0), c(0.5, -2.0), c(1e-3, 7.0), c(-6.0, 0.25)];
        let complex: TensorMap<_, Complex64> = TensorMap::diagonal(
            engine,
            &runtime,
            &bond,
            [
                SectorSpectrum { sector: M::unit(), values: source[..2].to_vec() },
                SectorSpectrum { sector: M::charged(), values: source[2..].to_vec() },
            ],
        )
        .unwrap();
        let read = |tensor: &TensorMap<_, Complex64>, engine: &mut symtn_dense::CpuEngine| {
            assert!(tensor.dense_data().is_err(), "the result is compact");
            tensor
                .diagview(engine)
                .unwrap()
                .into_iter()
                .flat_map(|entry| entry.values)
                .collect::<Vec<_>>()
        };
        let inverse = read(&complex.inv(engine, &[0], &[1]).unwrap(), engine);
        // sigma_max = |1e-3 + 7i|: rcond 0.5 cuts only |0.5 - 2i| < 3.5.
        let pseudo = read(&complex.pinv(engine, &[0], &[1], 0.5).unwrap(), engine);
        for (index, &value) in source.iter().enumerate() {
            let reciprocal = c(1.0, 0.0) / value;
            crate::ulp::assert_complex_within_ulps(inverse[index], reciprocal, 1, "inv");
            let expected = if index == 1 { c(0.0, 0.0) } else { reciprocal };
            crate::ulp::assert_complex_within_ulps(pseudo[index], expected, 1, "pinv");
        }
    }
}

// ---------------------------------------------------------------------------
// `exp`.
// ---------------------------------------------------------------------------

in_both_modes! {
    /// Like TensorKit's `exp`, a general per-block exponential with no
    /// hermiticity gate, for owned, lazy-adjoint and complex receivers, on the
    /// source provider and without provider queries. Oracle: `exp(N) = 1 + N`
    /// for the nilpotent unit block and `exp(θJ) = cos θ + J sin θ` for the
    /// charged rotation generator, whose norm 8 forces Padé scaling and
    /// squaring.
    fn exp_matches_closed_forms_for_non_hermitian_blocks<M>() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let provider = M::provider();
        let bond =
            GradedSpace::try_new(Arc::clone(&provider), [(M::unit(), 2), (M::charged(), 2)])
                .unwrap();
        let theta = 8.0;
        let source: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&bond], [&bond], |trees, ij| {
                match (trees.coupled() == &M::unit(), ij[0], ij[1]) {
                    (true, 0, 1) => 1.0,
                    (false, 0, 1) => -theta,
                    (false, 1, 0) => theta,
                    _ => 0.0,
                }
            })
            .unwrap();
        let (cos, sin) = (theta.cos(), theta.sin());

        M::reset_queries(&provider);
        let direct = source.exp(engine, &[0], &[1]).unwrap();
        M::assert_no_queries(&provider);
        assert!(std::ptr::eq(direct.provider(), provider.as_ref()));
        assert_eq!(direct.codomain(), source.codomain());
        assert_eq!(direct.domain(), source.domain());
        dense_close(
            "exp(A)",
            direct.dense_data().unwrap(),
            &[1.0, 0.0, 1.0, 1.0, cos, sin, -sin, cos],
        );

        let lazy = source.adjoint(engine).unwrap().exp(engine, &[0], &[1]).unwrap();
        dense_close(
            "exp(A†)",
            lazy.dense_data().unwrap(),
            &[1.0, 1.0, 0.0, 1.0, cos, -sin, sin, cos],
        );

        let z = Complex64::new(1.0, 0.25);
        let complex = source
            .convert::<Complex64>(engine)
            .unwrap()
            .scale(engine, z)
            .unwrap();
        M::reset_queries(&provider);
        let complex_exp = complex.exp(engine, &[0], &[1]).unwrap();
        M::assert_no_queries(&provider);
        let (one, zero) = (Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0));
        let (cos, sin) = ((z * theta).cos(), (z * theta).sin());
        dense_close(
            "exp(zA)",
            complex_exp.dense_data().unwrap(),
            &[one, zero, z, one, cos, sin, -sin, cos],
        );
    }
}

in_both_modes! {
    /// `exp` needs `codomain == domain`; anything else is refused before
    /// provider work, with the source untouched.
    fn exp_rejects_a_non_endomorphism_before_provider_work<M>() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let provider = M::provider();
        let wide = GradedSpace::try_new(Arc::clone(&provider), [(M::charged(), 2)]).unwrap();
        let narrow = GradedSpace::try_new(Arc::clone(&provider), [(M::charged(), 1)]).unwrap();
        let source: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&wide], [&narrow], |_, _| 1.0).unwrap();
        let before = source.dense_data().unwrap().to_vec();
        M::reset_queries(&provider);
        let error = facade_err(source.exp(engine, &[0], &[1]));
        assert!(is_space_mismatch(&error, "exp requires an endomorphism"), "{error:?}");
        M::assert_no_queries(&provider);
        assert_eq!(source.dense_data().unwrap(), before.as_slice());
    }
}

in_both_modes! {
    /// A nonfinite block, first or last in storage order, is refused as an
    /// invalid argument and nothing is published into the source.
    fn exp_rejects_a_nonfinite_sector_without_publication<M>() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let provider = M::provider();
        let leg =
            GradedSpace::try_new(Arc::clone(&provider), [(M::unit(), 1), (M::charged(), 1)])
                .unwrap();
        for target in [M::unit(), M::charged()] {
            let source: TensorMap<_, f64> =
                TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |trees, _| {
                    if trees.coupled() == &target { f64::NAN } else { 0.0 }
                })
                .unwrap();
            let before = source.dense_data().unwrap().to_vec();
            let error = facade_err(source.exp(engine, &[0], &[1]));
            assert!(
                matches!(
                    &error,
                    symtn::typed::Error::Operation(e) if matches!(
                        &**e,
                        OperationError::InvalidArgument { message }
                            if message.contains("finite coupled-sector blocks")
                    )
                ),
                "{target:?}: {error:?}"
            );
            assert!(source
                .dense_data()
                .unwrap()
                .iter()
                .zip(&before)
                .all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }
}

in_both_modes! {
    /// TensorKit's `exp(::DiagonalTensorMap)` is unconditionally elementwise
    /// and stays compact, real or complex; the dense storage of the same
    /// (non-Hermitian) complex diagonal goes through the general dense arm
    /// and agrees with it.
    fn compact_exp_is_elementwise_and_agrees_with_the_dense_arm<M>() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let provider = M::provider();
        let bond =
            GradedSpace::try_new(Arc::clone(&provider), [(M::unit(), 2), (M::charged(), 1)])
                .unwrap();
        let real_values = [-1.0, 0.5, 2.0];
        let real: TensorMap<_, f64> = TensorMap::diagonal(
            engine,
            &runtime,
            &bond,
            [
                SectorSpectrum { sector: M::unit(), values: real_values[..2].to_vec() },
                SectorSpectrum { sector: M::charged(), values: real_values[2..].to_vec() },
            ],
        )
        .unwrap();
        let image = real.exp(engine, &[0], &[1]).unwrap();
        assert!(image.dense_data().is_err(), "real exp stays compact");
        let got: Vec<f64> =
            image.diagview(engine).unwrap().into_iter().flat_map(|e| e.values).collect();
        dense_close("exp(d)", &got, &real_values.map(f64::exp));

        let c = Complex64::new;
        let complex_values = [c(0.0, 1.0), c(0.5, -2.0), c(-1.0, 3.0)];
        let complex: TensorMap<_, Complex64> = TensorMap::diagonal(
            engine,
            &runtime,
            &bond,
            [
                SectorSpectrum { sector: M::unit(), values: complex_values[..2].to_vec() },
                SectorSpectrum { sector: M::charged(), values: complex_values[2..].to_vec() },
            ],
        )
        .unwrap();
        let image = complex.exp(engine, &[0], &[1]).unwrap();
        assert!(image.dense_data().is_err(), "complex exp stays compact");
        let got: Vec<Complex64> =
            image.diagview(engine).unwrap().into_iter().flat_map(|e| e.values).collect();
        dense_close("exp(z)", &got, &complex_values.map(|z| z.exp()));

        let dense_exp = complex
            .materialize(engine)
            .unwrap()
            .exp(engine, &[0], &[1])
            .unwrap();
        dense_close(
            "dense exp of the same diagonal",
            dense_exp.dense_data().unwrap(),
            image.materialize(engine).unwrap().dense_data().unwrap(),
        );
    }
}

// ---------------------------------------------------------------------------
// `map_diagonal`.
// ---------------------------------------------------------------------------

in_both_modes! {
    /// `map_diagonal` applies `f` to the stored values of a compact diagonal
    /// (also through a lazy adjoint, which conjugates them) and stays compact,
    /// without provider queries. Dense storage is refused before any query,
    /// even when its blocks happen to be diagonal, and the source is untouched.
    fn map_diagonal_maps_compact_values_and_refuses_dense_storage<M>() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let provider = M::provider();
        let bond =
            GradedSpace::try_new(Arc::clone(&provider), [(M::unit(), 2), (M::charged(), 1)])
                .unwrap();
        let compact: TensorMap<_, f64> = TensorMap::diagonal(
            engine,
            &runtime,
            &bond,
            [
                SectorSpectrum { sector: M::unit(), values: vec![4.0, 9.0] },
                SectorSpectrum { sector: M::charged(), values: vec![0.25] },
            ],
        )
        .unwrap();
        M::reset_queries(&provider);
        let root = compact.map_diagonal(engine, f64::sqrt).unwrap();
        M::assert_no_queries(&provider);
        assert!(root.dense_data().is_err(), "the result is compact");
        assert_eq!(root.codomain(), compact.codomain());
        let values: Vec<Vec<f64>> =
            root.diagview(engine).unwrap().into_iter().map(|e| e.values).collect();
        assert_eq!(values, [vec![2.0, 3.0], vec![0.5]]);

        let c = Complex64::new;
        let complex: TensorMap<_, Complex64> = TensorMap::diagonal(
            engine,
            &runtime,
            &bond,
            [
                SectorSpectrum { sector: M::unit(), values: vec![c(1.0, 2.0), c(0.0, -1.0)] },
                SectorSpectrum { sector: M::charged(), values: vec![c(3.0, 0.0)] },
            ],
        )
        .unwrap();
        let squared = complex
            .adjoint(engine)
            .unwrap()
            .map_diagonal(engine, |value| value * value)
            .unwrap();
        assert!(squared.dense_data().is_err(), "the adjoint result is compact");
        let values: Vec<Complex64> =
            squared.diagview(engine).unwrap().into_iter().flat_map(|e| e.values).collect();
        assert_eq!(values, [c(-3.0, -4.0), c(-1.0, 0.0), c(9.0, 0.0)]);

        let dense_diagonal: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&bond], [&bond], |_, ij| {
                f64::from(ij[0] == ij[1])
            })
            .unwrap();
        let non_bond: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&bond, &bond], [&bond], |_, _| 2.0)
                .unwrap();
        for (name, tensor) in [
            ("dense diagonal", dense_diagonal.clone()),
            ("lazy adjoint", dense_diagonal.adjoint(engine).unwrap()),
            ("non-bond", non_bond),
        ] {
            let before = tensor.materialize(engine).unwrap().dense_data().unwrap().to_vec();
            M::reset_queries(&provider);
            match tensor.map_diagonal(engine, f64::sqrt) {
                Err(symtn::typed::Error::InvalidArgument(message)) => assert!(
                    message.contains("compact diagonal") && message.contains("dense storage"),
                    "{name}: {message}"
                ),
                other => panic!("{name}: dense input was accepted: {other:?}"),
            }
            M::assert_no_queries(&provider);
            assert_eq!(
                tensor.materialize(engine).unwrap().dense_data().unwrap(),
                before.as_slice(),
                "{name}"
            );
        }
    }
}
