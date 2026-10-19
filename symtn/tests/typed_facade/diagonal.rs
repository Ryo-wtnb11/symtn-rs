use super::*;

#[test]
fn compact_reductions_carry_the_su2_dimension_weight() {
    // What: the compact reductions apply `dim(c)` exactly where the dense ones
    // do. Z2 alone cannot see this — every `dim(c)` is one there.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = su2_tensor(&runtime);
    let typed = typed.svd_compact(&[0], &[1]).unwrap().s;
    let dense = forced_dense(&typed);

    // Compact and dense routes may sum in different orders; see above.
    let terms = dense.dense_data().unwrap().len();
    numerics::assert_close(
        "norm",
        typed.norm(2.0).unwrap(),
        dense.norm(2.0).unwrap(),
        terms,
    );
    numerics::assert_close("tr", typed.tr().unwrap(), dense.tr().unwrap(), terms);
    numerics::assert_close(
        "inner",
        typed.inner(&typed).unwrap(),
        dense.inner(&dense).unwrap(),
        terms,
    );
    // `<s, s>` is the squared norm: the identity that pins the weighting.
    let norm = typed.norm(2.0).unwrap();
    assert!((typed.inner(&typed).unwrap() - norm * norm).abs() < 1e-9 * norm * norm);
    // The unweighted sum, for contrast: a dropped `dim(c)` would make `tr` this.
    let unweighted: f64 = typed
        .materialize()
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .sum();
    assert!(
        (typed.tr().unwrap() - unweighted).abs() > 1e-6,
        "the SU(2) spectrum trace is not dimension weighted"
    );
    // `norm(Inf)` is deliberately *not* weighted.
    assert_eq!(
        typed.norm(f64::INFINITY).unwrap(),
        dense.norm(f64::INFINITY).unwrap()
    );
}

#[test]
fn compact_add_matches_pointwise_values_on_both_arms() {
    // What: diagonal + diagonal stays diagonal, and diagonal + dense goes dense
    // — and both arms obey the pointwise linear-combination law.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_bond(&runtime);

    let diagonal_sum = typed.axpby(0.75, &typed, -0.5).unwrap();
    for (&actual, &source) in diagonal_sum
        .materialize()
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(typed.materialize().unwrap().dense_data().unwrap())
    {
        let expected = 0.25 * source;
        assert!((actual - expected).abs() <= 1e-12 * expected.abs().max(1.0));
    }

    // A dense operand on the same bond space: `id` is the cheapest one, and it
    // is not diagonal *storage* even though its values are diagonal.
    let typed_dense = TensorMap::isomorphism(&runtime, &typed.domain(), &typed.domain())
        .unwrap()
        .scale(3.0)
        .unwrap();

    for (alpha, beta) in [(0.75, -0.5), (1.0, 1.0)] {
        let diagonal_dense = typed.axpby(alpha, &typed_dense, beta).unwrap();
        let dense_diagonal = typed_dense.axpby(alpha, &typed, beta).unwrap();
        for (((&diagonal_dense, &dense_diagonal), &diagonal), &dense) in diagonal_dense
            .dense_data()
            .unwrap()
            .iter()
            .zip(dense_diagonal.dense_data().unwrap())
            .zip(typed.materialize().unwrap().dense_data().unwrap())
            .zip(typed_dense.dense_data().unwrap())
        {
            let expected = alpha * diagonal + beta * dense;
            assert!((diagonal_dense - expected).abs() <= 1e-12 * expected.abs().max(1.0));
            let expected = alpha * dense + beta * diagonal;
            assert!((dense_diagonal - expected).abs() <= 1e-12 * expected.abs().max(1.0));
        }
    }
}

#[test]
fn compose_takes_the_compact_paths_and_reconstructs_the_source() {
    // What: `u * s * vh` runs through the bond-scaling arms rather than a
    // dense GEMM, and still reproduces the source — plus `s * s`, the
    // diagonal-times-diagonal arm.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_tensor(&runtime);
    let Svd {
        u: tu,
        s: ts,
        vh: tvh,
    } = typed.svd_compact(&[0, 1], &[2]).unwrap();

    let tus = tu.compose(&ts).unwrap();
    let svh = ts.compose(&tvh).unwrap();
    let squared = ts.compose(&ts).unwrap();
    // `s * s` is still a spectrum: its entries are the squares.
    for (squared, original) in squared
        .materialize()
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(ts.materialize().unwrap().dense_data().unwrap())
    {
        assert!((squared - original * original).abs() < 1e-12);
    }

    // Both associations reconstruct, separately pinning `t * D` and `D * t`.
    for recon in [tus.compose(&tvh).unwrap(), tu.compose(&svh).unwrap()] {
        assert_data_close_f64(recon.dense_data().unwrap(), typed.dense_data().unwrap());
    }
}

#[test]
fn compose_declines_a_compact_arm_it_cannot_prove() {
    // What: the compact arms fire on a proved destination, not on the storage
    // alone. Two spectra on different bond spaces are not composable at all,
    // and the guard is what leaves that verdict to the expert layer instead of
    // multiplying two unrelated spectra elementwise.
    let _guard = cache_lock();
    let runtime = runtime();
    let wide = z2_tensor_split(&runtime, 2);
    // A second endomorphism on a leg with different degeneracies, so its bond
    // space genuinely differs from `wide`'s rather than merely being a second
    // allocation of the same one.
    let narrow_leg = GradedSpace::try_new(
        Arc::new(symtn::sector::Z2FusionRule),
        [
            (symtn::sector::Z2Irrep::EVEN, 1),
            (symtn::sector::Z2Irrep::ODD, 1),
        ],
    )
    .unwrap();
    let mut next = 0.0;
    let narrow = TensorMap::from_subblock_fn(&runtime, [&narrow_leg], [&narrow_leg], |_, _| {
        next += 1.0;
        next
    })
    .unwrap();
    let wide_s = wide.svd_compact(&[0, 1], &[2]).unwrap().s;
    let narrow_s = narrow.svd_compact(&[0], &[1]).unwrap().s;

    assert_ne!(
        wide_s.materialize().unwrap().dense_data().unwrap().len(),
        narrow_s.materialize().unwrap().dense_data().unwrap().len(),
        "the fixture's two bond spaces must differ for this to test anything"
    );
    assert!(
        wide_s
            .compose(&narrow_s)
            .err()
            .unwrap()
            .report()
            .contains("leg degeneracy mismatch"),
        "composing spectra on mismatched bond spaces must be refused"
    );
    // And the same for a dense operand whose contracted leg does not match the
    // spectrum's bond.
    assert!(wide
        .compose(&narrow_s)
        .err()
        .unwrap()
        .report()
        .contains("leg degeneracy mismatch"));
    let wide_vh = wide.svd_compact(&[0, 1], &[2]).unwrap().vh;
    let result = narrow_s.compose(&wide_vh);
    assert!(
        matches!(result, Err(symtn::typed::Error::Operation(_)))
            && result
                .err()
                .unwrap()
                .report()
                .contains("leg degeneracy mismatch")
    );
}

// ---------------------------------------------------------------------------
// Issue #584: the compact diagonal arm of `contract`.
// ---------------------------------------------------------------------------

/// The three axis patterns the diagonal `contract` arm claims, plus one it must
/// decline, as `(name, spectrum on the right, spec)` on a `[v, v] <- [v]`
/// tensor `t` and its own SVD spectrum `s`.
///
/// `t · s` contracts `t`'s domain axis against `s`'s codomain axis (the
/// compose-shaped pairing, which is the only one the engine admits: contracted
/// legs must agree on their duality flag), and `s · t` the mirror.
const DIAGONAL_CONTRACT_CASES: &[(&str, bool, ContractSpec<'static>)] = &[
    // `t · s`, identity output order: the scaled leg stays last.
    (
        "t*s",
        true,
        ContractSpec {
            lhs: &[2],
            rhs: &[0],
            codomain: &[0, 1],
            domain: &[2],
        },
    ),
    // The same, with the output order moving the scaled leg across the split.
    (
        "t*s reordered",
        true,
        ContractSpec {
            lhs: &[2],
            rhs: &[0],
            codomain: &[2, 0],
            domain: &[1],
        },
    ),
    // A split of another size: one codomain leg (#1549).
    (
        "t*s resplit",
        true,
        ContractSpec {
            lhs: &[2],
            rhs: &[0],
            codomain: &[1],
            domain: &[2, 0],
        },
    ),
    // `s · t`: `s`'s domain axis against `t`'s leading codomain axis, so the
    // scaled leg comes first and the destination is `[v] <- [v, v]`.
    (
        "s*t",
        false,
        ContractSpec {
            lhs: &[1],
            rhs: &[0],
            codomain: &[0],
            domain: &[1, 2],
        },
    ),
    (
        "s*t reordered",
        false,
        ContractSpec {
            lhs: &[1],
            rhs: &[0],
            codomain: &[1],
            domain: &[2, 0],
        },
    ),
    // Every open leg in the codomain (#1549).
    (
        "s*t resplit",
        false,
        ContractSpec {
            lhs: &[1],
            rhs: &[0],
            codomain: &[2, 0, 1],
            domain: &[],
        },
    ),
    // `s · t` on `t`'s second codomain axis: the arm's leg position is free
    // within the preserved side.
    (
        "s*t inner leg",
        false,
        ContractSpec {
            lhs: &[1],
            rhs: &[1],
            codomain: &[0],
            domain: &[1, 2],
        },
    ),
    // Declined by the arm — `s`'s *codomain* axis against `t`'s domain axis is
    // admissible but is not one of the two proved geometries — and computed
    // densely. The expected bytes are pinned below.
    (
        "s*t dense fallback",
        false,
        ContractSpec {
            lhs: &[0],
            rhs: &[2],
            codomain: &[0],
            domain: &[1, 2],
        },
    ),
];

#[test]
fn compact_contract_identity_output_does_not_publish_a_transform_cache_entry() {
    // Isolated: the completed-transformer counters are process-global.
    if crate::run_isolated_or_return(
        "SYMTN_TYPED_FACADE_COMPACT_CONTRACT_IDENTITY_OUTPUT_DOES_NO",
        "diagonal::compact_contract_identity_output_does_not_publish_a_transform_cache_entry",
    ) {
        return;
    }
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(symtn::sector::U1FusionRule);
    let leg = u1_leg(&provider, &[(-1, 1), (0, 2), (1, 1)]);
    let tensor: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&runtime, [&leg, &leg], [&leg], 592_302).unwrap();
    let spectrum = TensorMap::<_, f64>::isomorphism(&runtime, [&leg], [&leg])
        .unwrap()
        .svd_compact(&[0], &[1])
        .unwrap()
        .s;
    symtn::cache::clear();
    let before = crate::completed_transformers();

    let result = tensor
        .contract(
            &spectrum,
            &ContractSpec {
                lhs: &[2],
                rhs: &[0],
                codomain: &[0, 1],
                domain: &[2],
            },
        )
        .unwrap();

    assert_eq!(
        result.dense_data().unwrap().len(),
        tensor.dense_data().unwrap().len()
    );
    assert_eq!(crate::completed_transformers(), before);
}

#[test]
fn diagonal_contract_preserves_left_provider_authority_on_every_compact_arm() {
    let _guard = cache_lock();
    let runtime = runtime();

    macro_rules! exercise {
        ($name:literal, $rule:expr, $sectors:expr) => {{
            let left_leg = GradedSpace::try_new(Arc::new($rule), $sectors).unwrap();
            let right_leg = GradedSpace::try_new(Arc::new($rule), $sectors).unwrap();
            let left = TensorMap::<_, f64>::isomorphism(&runtime, [&left_leg], [&left_leg])
                .unwrap()
                .scale(2.0)
                .unwrap();
            let right = TensorMap::<_, f64>::isomorphism(&runtime, [&right_leg], [&right_leg])
                .unwrap()
                .scale(3.0)
                .unwrap();
            let left_d = left.svd_compact(&[0], &[1]).unwrap().s;
            let right_d = right.svd_compact(&[0], &[1]).unwrap().s;
            assert!(!std::ptr::eq(left.provider(), right.provider()));

            let left_dense = forced_dense(&left_d);
            let right_dense = forced_dense(&right_d);

            let actual = [
                left.contract(
                    &right_d,
                    &ContractSpec {
                        lhs: &[1],
                        rhs: &[0],
                        codomain: &[0],
                        domain: &[1],
                    },
                )
                .unwrap(),
                right_d
                    .contract(
                        &left,
                        &ContractSpec {
                            lhs: &[1],
                            rhs: &[0],
                            codomain: &[0],
                            domain: &[1],
                        },
                    )
                    .unwrap(),
                right_d
                    .contract(
                        &left_d,
                        &ContractSpec {
                            lhs: &[1],
                            rhs: &[0],
                            codomain: &[0],
                            domain: &[1],
                        },
                    )
                    .unwrap(),
            ];
            let expected = [
                left.contract(
                    &right_dense,
                    &ContractSpec {
                        lhs: &[1],
                        rhs: &[0],
                        codomain: &[0],
                        domain: &[1],
                    },
                )
                .unwrap(),
                right_dense
                    .contract(
                        &left,
                        &ContractSpec {
                            lhs: &[1],
                            rhs: &[0],
                            codomain: &[0],
                            domain: &[1],
                        },
                    )
                    .unwrap(),
                right_dense
                    .contract(
                        &left_dense,
                        &ContractSpec {
                            lhs: &[1],
                            rhs: &[0],
                            codomain: &[0],
                            domain: &[1],
                        },
                    )
                    .unwrap(),
            ];
            let providers = [left.provider(), right_d.provider(), right_d.provider()];

            for (index, arm) in ["t*D", "D*t", "D*D"].into_iter().enumerate() {
                let (actual, expected) = (&actual[index], &expected[index]);
                assert_same_legs(&actual.codomain(), &expected.codomain());
                assert_same_legs(&actual.domain(), &expected.domain());
                assert_eq!(
                    actual.materialize().unwrap().dense_data().unwrap(),
                    expected.dense_data().unwrap(),
                    "{} {arm} values",
                    $name
                );
                assert!(
                    std::ptr::eq(actual.provider(), providers[index]),
                    "{} {arm} lost left authority",
                    $name
                );
                assert_eq!(
                    symtn::expert::diagonal_spectrum(&actual).unwrap().is_some(),
                    index == 2,
                    "{} {arm} storage",
                    $name
                );
            }
        }};
    }

    exercise!(
        "Z2",
        symtn::sector::Z2FusionRule,
        [
            (symtn::sector::Z2Irrep::EVEN, 2),
            (symtn::sector::Z2Irrep::ODD, 1),
        ]
    );
    exercise!(
        "U1",
        symtn::sector::U1FusionRule,
        [
            (symtn::sector::U1Irrep::new(0), 2),
            (symtn::sector::U1Irrep::new(1), 1),
        ]
    );
}

#[test]
fn complex_diagonal_contract_matches_the_typed_dense_route() {
    // What: every complex compact arm agrees with the ordinary typed engine
    // route on the same spaces, including a compact `D · D` result.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_complex_tensor(&runtime);
    let typed_s = typed.svd_compact(&[0, 1], &[2]).unwrap().s;
    let dense_s = forced_dense(&typed_s);

    for &(_name, spectrum_on_the_right, spec) in DIAGONAL_CONTRACT_CASES {
        let (fast_lhs, fast_rhs, dense_lhs, dense_rhs) = if spectrum_on_the_right {
            (&typed, &typed_s, &typed, &dense_s)
        } else {
            (&typed_s, &typed, &dense_s, &typed)
        };
        let expected = dense_lhs.contract(dense_rhs, &spec).unwrap();
        let got = fast_lhs.contract(fast_rhs, &spec).unwrap();
        assert_same_legs(&got.codomain(), &expected.codomain());
        assert_same_legs(&got.domain(), &expected.domain());
        assert_data_close_c64(got.dense_data().unwrap(), expected.dense_data().unwrap());
    }

    let expected = dense_s
        .contract(
            &dense_s,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    let got = typed_s
        .contract(
            &typed_s,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    assert_same_legs(&got.codomain(), &expected.codomain());
    assert_same_legs(&got.domain(), &expected.domain());
    assert_data_close_c64(
        got.materialize().unwrap().dense_data().unwrap(),
        expected.dense_data().unwrap(),
    );
    assert!(symtn::expert::diagonal_spectrum(&got).unwrap().is_some());
}

#[test]
fn the_diagonal_contract_arm_keeps_fermionic_signs() {
    // What: bends inside the compact arm's `permute` pick up the same parity
    // signs as the ordinary typed engine route.
    //
    // The supertrace twist `contract` applies to a **dual** contracted leg of
    // the right operand cannot be reached from this facade, so it is not
    // asserted here: a compact spectrum's bond leg is built non-dual
    // (`seam::spectrum_bond`), the engine admits a contraction only
    // when the two contracted legs agree on their duality flag, and the arm's
    // right-operand leg is codomain-side, where external duality *is* that
    // flag. `TensorMap::try_contract_diagonal` declines rather than assumes it,
    // and that guard is what would have to be tested if a dual bond leg ever
    // became constructible here.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = fermionic_endo(&runtime);
    let typed_s = typed.svd_compact(&[0], &[1]).unwrap().s;
    let dense_s = forced_dense(&typed_s);

    for &(name, spectrum_on_the_right, output_axes) in &[
        ("t*s", true, &[0usize, 1][..]),
        ("t*s reordered", true, &[1, 0][..]),
        ("s*t", false, &[0, 1][..]),
    ] {
        let (fast_lhs, fast_rhs, dense_lhs, dense_rhs) = if spectrum_on_the_right {
            (&typed, &typed_s, &typed, &dense_s)
        } else {
            (&typed_s, &typed, &dense_s, &typed)
        };
        let expected = dense_lhs
            .contract(
                dense_rhs,
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &output_axes[..1],
                    domain: &output_axes[1..],
                },
            )
            .unwrap();
        let got = fast_lhs
            .contract(
                fast_rhs,
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &output_axes[..1],
                    domain: &output_axes[1..],
                },
            )
            .unwrap();
        assert_same_legs(&got.codomain(), &expected.codomain());
        assert_same_legs(&got.domain(), &expected.domain());
        assert_eq!(
            got.dense_data().unwrap(),
            expected.dense_data().unwrap(),
            "fermionic {name}"
        );
    }

    let expected = dense_s
        .contract(
            &dense_s,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    let got = typed_s
        .contract(
            &typed_s,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    assert_same_legs(&got.codomain(), &expected.codomain());
    assert_same_legs(&got.domain(), &expected.domain());
    assert_eq!(
        got.materialize().unwrap().dense_data().unwrap(),
        expected.dense_data().unwrap(),
        "fermionic s*s"
    );
    assert!(symtn::expert::diagonal_spectrum(&got).unwrap().is_some());
}

#[test]
fn the_diagonal_contract_arm_declines_an_illegal_contraction() {
    // What: the arm must not answer where the dense route would refuse. A
    // contracted leg whose duality flag does not match the spectrum's bond leg
    // is inadmissible, and the error is the expert layer's, not a scaled tensor
    // on a made-up space.
    let _guard = cache_lock();
    let runtime = runtime();
    let leg = GradedSpace::try_new(
        Arc::new(symtn::sector::Z2FusionRule),
        [
            (symtn::sector::Z2Irrep::EVEN, 2),
            (symtn::sector::Z2Irrep::ODD, 3),
        ],
    )
    .unwrap();
    let dual = leg.try_dual().unwrap();
    let typed = TensorMap::from_subblock_fn(&runtime, [&dual], [&leg], typed_fill_value).unwrap();
    let s = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], typed_fill_value)
        .unwrap()
        .svd_compact(&[0], &[1])
        .unwrap()
        .s;

    // `s`'s domain leg is non-dual, `typed`'s leading codomain leg is dual.
    assert!(s
        .contract(
            &typed,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1]
            }
        )
        .err()
        .unwrap()
        .report()
        .contains("contracted fusion leg duality flags do not match"));
    // The mirror direction needs its own case, because it is the `t · D` arm's
    // own leg comparison that has to reject it: `leg <- dual` contracted on its
    // domain axis against `s`, whose bond leg is non-dual by construction, so
    // the two raw flags differ and the engine refuses the pair.
    let dual_domain =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&dual], typed_fill_value).unwrap();
    assert!(dual_domain
        .contract(
            &s,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1]
            }
        )
        .err()
        .unwrap()
        .report()
        .contains("contracted fusion leg duality flags do not match"));
    // A degeneracy mismatch on an otherwise well-oriented pair is the other way
    // the comparison earns its keep: nothing about the axis pattern is wrong, so
    // only the legs themselves say this is not a contraction.
    let narrow = GradedSpace::try_new(
        Arc::new(symtn::sector::Z2FusionRule),
        [
            (symtn::sector::Z2Irrep::EVEN, 2),
            (symtn::sector::Z2Irrep::ODD, 2),
        ],
    )
    .unwrap();
    let narrow_bond = TensorMap::from_subblock_fn(&runtime, [&narrow], [&narrow], typed_fill_value)
        .unwrap()
        .svd_compact(&[0], &[1])
        .unwrap()
        .s;
    let wide = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], typed_fill_value).unwrap();
    assert!(wide
        .contract(
            &narrow_bond,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1]
            }
        )
        .err()
        .unwrap()
        .report()
        .contains("leg degeneracy mismatch"));
    assert!(narrow_bond
        .contract(
            &wide,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1]
            }
        )
        .err()
        .unwrap()
        .report()
        .contains("leg degeneracy mismatch"));
    // And a wrong-length axis list or a non-permutation output order is still
    // the expert layer's error rather than a fast-path answer. `[0, 2]` is the
    // out-of-range case, which the arm has to reject *before* indexing its own
    // source order with it.
    assert!(matches!(typed
        .contract(
            &s,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[]
            }
        ), Err(symtn::typed::Error::Operation(e)) if matches!(*e, symtn::typed::OperationError::InvalidPermutation { rank: 2, .. })));
    assert!(matches!(typed
        .contract(
            &s,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[1],
                domain: &[1]
            }
        ), Err(symtn::typed::Error::Operation(e)) if matches!(*e, symtn::typed::OperationError::InvalidPermutation { rank: 2, .. })));
    assert!(matches!(wide
        .contract(
            &s,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[2]
            }
        ), Err(symtn::typed::Error::Operation(e)) if matches!(*e, symtn::typed::OperationError::InvalidPermutation { rank: 2, .. })));
    assert!(matches!(typed
        .contract(
            &s,
            &ContractSpec {
                lhs: &[9],
                rhs: &[0],
                codomain: &[0],
                domain: &[1]
            }
        ), Err(symtn::typed::Error::Operation(e)) if matches!(*e, symtn::typed::OperationError::InvalidAxisSet { tensor: "lhs", .. })));
}

/// The codomain/domain split plus every leg's sectors, degeneracies and dual
/// flag. Comparing the split alone is too weak: a reordered output can leave the
/// rank and the codomain length intact and still land on legs with the opposite
/// duality flag, which is exactly what the `D · D` output-order guard refuses.
#[allow(clippy::type_complexity)]
fn space_shape(
    t: &TensorMap<symtn::sector::Z2FusionRule, f64>,
) -> (usize, Vec<(Vec<symtn::sector::Z2Irrep>, Vec<usize>, bool)>) {
    let legs = t
        .codomain()
        .iter()
        .chain(t.domain().iter())
        .map(|leg| {
            (
                leg.sectors().unwrap(),
                leg.degeneracies().to_vec(),
                leg.is_dual(),
            )
        })
        .collect();
    (t.codomain().len(), legs)
}

/// Every permutation of `0..n`, for the exhaustive output-order sweep below.
fn all_output_orders(n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![Vec::new()];
    }
    let mut orders = Vec::new();
    for head in 0..n {
        for mut rest in all_output_orders(n - 1) {
            for axis in &mut rest {
                if *axis >= head {
                    *axis += 1;
                }
            }
            let mut order = vec![head];
            order.extend(rest);
            orders.push(order);
        }
    }
    orders
}

#[test]
fn the_diagonal_contract_arm_is_its_own_dense_route_on_every_axis_pattern() {
    // What: the compact arm never differs from the dense route it replaces.
    // This direct oracle catches a shared fast-path mistake or a divergence
    // between the codomain rank derived by the arm (`self.rank() - 1` for
    // `t · D`, `1` for `D · t`) and the one the engine would build.
    //
    // Here the comparison is fast against dense *inside this facade*:
    // `forced_dense` adds an exact dense zero on the same space, which preserves
    // the values while forcing an ordinary dense payload. Every single-axis
    // pattern and output order is swept on both codomain/domain splits, so
    // `t · D` is covered at an inner domain axis (which `[v, v] <- [v]` cannot
    // reach: it has one domain
    // leg) as well as at the trailing one, and an inadmissible pattern must be
    // refused by both routes rather than answered by one.
    let _guard = cache_lock();
    let runtime = runtime();
    for split in [1, 2] {
        let t = z2_tensor_split(&runtime, split);
        let s = t
            .svd_compact(&codomain_axes(&t), &domain_axes(&t))
            .unwrap()
            .s;
        let s_dense = forced_dense(&s);

        let mut fired = 0usize;
        for orders in [&all_output_orders(3)] {
            for output_axes in orders {
                for lhs_axis in 0..3 {
                    for rhs_axis in 0..2 {
                        // `t · s`
                        let dense = t.contract(
                            &s_dense,
                            &ContractSpec {
                                lhs: &[lhs_axis],
                                rhs: &[rhs_axis],
                                codomain: &output_axes[..2],
                                domain: &output_axes[2..],
                            },
                        );
                        let fast = t.contract(
                            &s,
                            &ContractSpec {
                                lhs: &[lhs_axis],
                                rhs: &[rhs_axis],
                                codomain: &output_axes[..2],
                                domain: &output_axes[2..],
                            },
                        );
                        let label =
                            format!("t*s split={split} {lhs_axis}/{rhs_axis} {output_axes:?}");
                        match (dense, fast) {
                            (Ok(dense), Ok(fast)) => {
                                assert_eq!(
                                    space_shape(&fast),
                                    space_shape(&dense),
                                    "{label} space"
                                );
                                assert_eq!(
                                    fast.dense_data().unwrap(),
                                    dense.dense_data().unwrap(),
                                    "{label} payload"
                                );
                                fired += 1;
                            }
                            (Err(_), Err(_)) => {}
                            (dense, fast) => panic!(
                                "{label}: dense {:?} but fast {:?}",
                                dense.map(|_| ()),
                                fast.map(|_| ())
                            ),
                        }
                        // `s · t`
                        let dense = s_dense.contract(
                            &t,
                            &ContractSpec {
                                lhs: &[rhs_axis],
                                rhs: &[lhs_axis],
                                codomain: &output_axes[..1],
                                domain: &output_axes[1..],
                            },
                        );
                        let fast = s.contract(
                            &t,
                            &ContractSpec {
                                lhs: &[rhs_axis],
                                rhs: &[lhs_axis],
                                codomain: &output_axes[..1],
                                domain: &output_axes[1..],
                            },
                        );
                        let label =
                            format!("s*t split={split} {rhs_axis}/{lhs_axis} {output_axes:?}");
                        match (dense, fast) {
                            (Ok(dense), Ok(fast)) => {
                                assert_eq!(
                                    space_shape(&fast),
                                    space_shape(&dense),
                                    "{label} space"
                                );
                                assert_eq!(
                                    fast.dense_data().unwrap(),
                                    dense.dense_data().unwrap(),
                                    "{label} payload"
                                );
                                fired += 1;
                            }
                            (Err(_), Err(_)) => {}
                            (dense, fast) => panic!(
                                "{label}: dense {:?} but fast {:?}",
                                dense.map(|_| ()),
                                fast.map(|_| ())
                            ),
                        }
                    }
                }
            }
        }
        assert!(fired > 0, "split={split} swept no admissible pattern");

        // `s · s`, where the surviving bond may stay compact.
        for output_axes in all_output_orders(2) {
            for lhs_axis in 0..2 {
                for rhs_axis in 0..2 {
                    let dense = s_dense.contract(
                        &s_dense,
                        &ContractSpec {
                            lhs: &[lhs_axis],
                            rhs: &[rhs_axis],
                            codomain: &output_axes[..1],
                            domain: &output_axes[1..],
                        },
                    );
                    let fast = s.contract(
                        &s,
                        &ContractSpec {
                            lhs: &[lhs_axis],
                            rhs: &[rhs_axis],
                            codomain: &output_axes[..1],
                            domain: &output_axes[1..],
                        },
                    );
                    let label = format!("s*s {lhs_axis}/{rhs_axis} {output_axes:?}");
                    match (dense, fast) {
                        (Ok(dense), Ok(fast)) => {
                            assert_eq!(space_shape(&fast), space_shape(&dense), "{label} space");
                            assert_eq!(
                                fast.materialize().unwrap().dense_data().unwrap(),
                                dense.dense_data().unwrap(),
                                "{label} payload"
                            );
                        }
                        (Err(_), Err(_)) => {}
                        (dense, fast) => panic!(
                            "{label}: dense {:?} but fast {:?}",
                            dense.map(|_| ()),
                            fast.map(|_| ())
                        ),
                    }
                }
            }
        }
    }
}

/// The three rank-(1,1) re-orderings that reduce to the proved swap, plus the
/// two repartitions that do not — the latter are here so a compact arm that
/// fires too widely is caught by the same sweep.
fn rank_one_reorderings<R, D>(tensor: &TensorMap<R, D>) -> Vec<(&'static str, TensorMap<R, D>)>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: symtn::typed::TensorScalar + std::fmt::Debug,
{
    vec![
        ("permute", tensor.permute(&[1], &[0]).unwrap()),
        ("full transpose", full_transpose!(tensor).unwrap()),
        ("transpose", tensor.transpose(&[1], &[0]).unwrap()),
        ("repartition(1)", tensor.repartition(1).unwrap()),
        ("repartition(0)", tensor.repartition(0).unwrap()),
        ("repartition(2)", tensor.repartition(2).unwrap()),
    ]
}

#[test]
fn compact_rank_one_swaps_match_the_forced_dense_route() {
    // What: every re-ordering of a compact bond factor returns the tensor the
    // dense tree transform returns — same legs and same bytes.
    //
    // What this test cannot see: Z2 is self-dual and bosonic, so every swap
    // coefficient is exactly 1 — dropping the coefficient entirely still passes
    // here. The coefficient is carried by the Z3 and fZ2 sweep below, which is
    // where a mutation to it dies.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_bond(&runtime);
    let dense = forced_dense(&typed);

    for ((name, compact), (_, oracle)) in rank_one_reorderings(&typed)
        .into_iter()
        .zip(rank_one_reorderings(&dense))
    {
        assert_eq!(
            typed_leg_shapes(&compact),
            typed_leg_shapes(&oracle),
            "{name} legs"
        );
        assert_eq!(
            compact.materialize().unwrap().dense_data().unwrap(),
            oracle.dense_data().unwrap(),
            "{name} payload"
        );
    }
}

#[test]
fn compact_rank_one_swaps_match_the_dense_route_for_dual_and_fermionic_legs() {
    // What: the swap's per-sector coefficient is only observable where the two
    // ends of the bond are not interchangeable. Z3 is not self-dual, so the
    // destination block of every sector is a *different* sector; the fermionic
    // provider adds a braiding phase the bosonic cases cannot show. A dual leg
    // is swept on both, since bending is what fixes which sector labels the
    // destination structure carries.
    let _guard = cache_lock();
    let runtime = runtime();

    for is_dual in [false, true] {
        let provider = Arc::new(ExternalZ3::new());
        let leg = z3_leg(&provider, is_dual);
        let mut next = 0.0;
        let source = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, _| {
            next += 1.0;
            next
        })
        .unwrap();
        let compact = source.svd_compact(&[0], &[1]).unwrap().s;
        let dense = forced_dense(&compact);
        for ((name, actual), (_, expected)) in rank_one_reorderings(&compact)
            .into_iter()
            .zip(rank_one_reorderings(&dense))
        {
            assert_eq!(
                typed_leg_shapes(&actual),
                typed_leg_shapes(&expected),
                "z3 dual={is_dual} {name} legs"
            );
            assert_eq!(
                actual.materialize().unwrap().dense_data().unwrap(),
                expected.dense_data().unwrap(),
                "z3 dual={is_dual} {name} payload"
            );
        }

        let leg = GradedSpace::try_new(
            Arc::new(symtn::sector::FermionParityFusionRule),
            [
                (symtn::sector::Z2Irrep::EVEN, 2),
                (symtn::sector::Z2Irrep::ODD, 3),
            ],
        )
        .and_then(|space| if is_dual { space.try_dual() } else { Ok(space) })
        .unwrap();
        let mut next = 0.0;
        let source = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, _| {
            next += 1.0;
            next
        })
        .unwrap();
        let compact = source.svd_compact(&[0], &[1]).unwrap().s;
        let dense = forced_dense(&compact);
        for ((name, actual), (_, expected)) in rank_one_reorderings(&compact)
            .into_iter()
            .zip(rank_one_reorderings(&dense))
        {
            assert_eq!(
                typed_leg_shapes(&actual),
                typed_leg_shapes(&expected),
                "fZ2 dual={is_dual} {name} legs"
            );
            assert_eq!(
                actual.materialize().unwrap().dense_data().unwrap(),
                expected.dense_data().unwrap(),
                "fZ2 dual={is_dual} {name} payload"
            );
        }
    }
}

/// A compact bond factor whose spectrum is exactly `values` per sector, built
/// by rescaling a singular-value factor: `scale` stays compact, so the fixture
/// never leaves compact storage on its way to the assertion.
fn z2_spectrum_fixture(
    runtime: &Runtime,
    rank_deficient: bool,
) -> TensorMap<symtn::sector::Z2FusionRule, f64> {
    let leg = GradedSpace::try_new(
        Arc::new(symtn::sector::Z2FusionRule),
        [
            (symtn::sector::Z2Irrep::EVEN, 2),
            (symtn::sector::Z2Irrep::ODD, 3),
        ],
    )
    .unwrap();
    // A constant block is rank one, so all but one singular value per sector is
    // zero — the positive *semi*definite fixture `isposdef` must still reject.
    let source = TensorMap::from_subblock_fn(runtime, [&leg], [&leg], |sectors, indices| {
        if rank_deficient {
            1.0
        } else {
            typed_fill_value(sectors, indices)
        }
    })
    .unwrap();
    let s = source.svd_compact(&[0], &[1]).unwrap().s;
    if rank_deficient {
        // The trailing singular values of a rank-one block are zero only up to
        // the provider's rounding (exactly zero on faer, ~1e-17 on Accelerate).
        // Snap them so the fixture is the exactly semidefinite spectrum the
        // tests name.
        s.map_diagonal(|v| if v > 1e-8 { v } else { 0.0 }).unwrap()
    } else {
        s
    }
}

#[test]
fn compact_is_posdef_matches_the_forced_dense_route() {
    // What: reading the stored spectrum answers exactly what eigendecomposing
    // the materialization answers, on every sign pattern and at every
    // tolerance — including the strictness at zero, which is the one place a
    // `>=` would pass a positive-semidefinite spectrum the dense route rejects.
    let _guard = cache_lock();
    let runtime = runtime();

    let positive = z2_spectrum_fixture(&runtime, false);
    let semidefinite = z2_spectrum_fixture(&runtime, true);
    let negative = positive.scale(-1.0).unwrap();
    let indefinite = positive.axpby(1.0, &semidefinite, -3.0).unwrap();

    for (name, tensor) in [
        ("positive", &positive),
        ("semidefinite", &semidefinite),
        ("negative", &negative),
        ("indefinite", &indefinite),
    ] {
        let oracle = forced_dense(tensor);
        for tol in [0.0, 1e-14, 1e-8, 1e-3, 0.5] {
            assert_eq!(
                is_posdef_compact!(tensor, tol, |v: f64| v),
                is_posdef!(oracle, tol),
                "{name} at tol {tol}"
            );
        }
    }

    // The one case whose answer is asserted absolutely rather than only against
    // the oracle: a rank-deficient spectrum is positive semidefinite, and
    // `isposdef` is strict.
    assert!(is_posdef_compact!(positive, 0.0, |v: f64| v));
    assert!(!is_posdef_compact!(semidefinite, 0.0, |v: f64| v));
    assert!(!is_posdef_compact!(negative, 0.0, |v: f64| v));
}

#[test]
fn compact_is_posdef_matches_the_forced_dense_route_for_a_hermitian_c64_spectrum() {
    // What: a c64 payload whose stored values are real up to rounding is the
    // case where a compact branch could disagree with the dense one — the dense
    // route Hermitian-eigendecomposes and so reads only the real part, and the
    // compact branch must make the same choice rather than, say, comparing a
    // modulus. The `d` of `eigh_full` is exactly that payload.
    let _guard = cache_lock();
    let runtime = runtime();
    let real = z2_complex_tensor(&runtime);
    let square = real
        .contract(
            &real.adjoint().unwrap(),
            &ContractSpec {
                lhs: &[2],
                rhs: &[0],
                codomain: &[0, 1],
                domain: &[2, 3],
            },
        )
        .unwrap();
    let hermitian = square.repartition(2).unwrap();
    assert!(is_hermitian!(hermitian, 1e-10));

    let d = hermitian
        .eigh_full(&[0, 1], &[2, 3], HermitianTol::DEFAULT)
        .unwrap()
        .d;

    // The Hermiticity gate is load-bearing and is checked here, because every
    // other fixture in this file answers the same with or without it. Rotating
    // the spectrum by `1 + i` keeps every real part positive — so a predicate
    // that reads only the stored real parts calls it positive definite — while
    // making the tensor not Hermitian at all. The gate is the only thing
    // standing between the compact arm and a wrong `true`.
    // Built from singular values, not from the Gram's eigenvalues: the latter
    // can hold an exact zero, which the compact predicate rejects on its own
    // and which would therefore hide the gate rather than test it.
    let skewed = real
        .svd_compact(&[0, 1], &[2])
        .unwrap()
        .s
        .scale(Complex64::new(1.0, 1.0))
        .unwrap();
    assert!(!is_hermitian!(skewed, 1e-10));
    for tol in [0.0, 1e-14, 1e-8, 1e-3] {
        assert!(
            !is_posdef_compact!(skewed, tol, |v: Complex64| v.re),
            "skewed at tol {tol}"
        );
        assert_eq!(
            is_posdef_compact!(skewed, tol, |v: Complex64| v.re),
            is_posdef!(forced_dense(&skewed), tol),
            "skewed at tol {tol}"
        );
    }

    let oracle = forced_dense(&d);
    for tol in [0.0, 1e-14, 1e-8, 1e-3] {
        assert_eq!(
            is_posdef_compact!(d, tol, |v: Complex64| v.re),
            is_posdef!(oracle, tol),
            "gram at tol {tol}"
        );
        let flipped = d.scale(Complex64::new(-1.0, 0.0)).unwrap();
        assert_eq!(
            is_posdef_compact!(flipped, tol, |v: Complex64| v.re),
            is_posdef!(forced_dense(&flipped), tol),
            "negated gram at tol {tol}"
        );
    }
}

// ---------------------------------------------------------------------------
// Issue #604: the compact full-pair trace arm. The typed `trace_pairs` keeps
// a compact spectrum factor compact. The value sweep here
// covers the same rules, orientations, and variants against a forced-dense
// typed route. The storage claim lives in `typed_diagonal_allocations.rs`.
// ---------------------------------------------------------------------------

/// A compact bond factor from a counter-filled `[v] <- [v]` endomorphism.
fn compact_bond_trace<R, D>(
    runtime: &Runtime,
    typed_leg: &GradedSpace<R>,
    fill: impl Fn(f64) -> D + Copy,
) -> TensorMap<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: symtn::typed::FactorizationScalar + std::fmt::Debug,
{
    let mut next: f64 = 0.0;
    let typed: TensorMap<R, D> =
        TensorMap::from_subblock_fn(runtime, [typed_leg], [typed_leg], |_, _| {
            next += 1.0;
            fill(next)
        })
        .unwrap();
    typed.svd_compact(&[0], &[1]).unwrap().s
}

/// The typed compact trace against the typed forced-dense engine route, both
/// pair orders. Close, not byte-for-byte: the engine reduces block by block in
/// its own order, so this is the independent value oracle.
fn assert_compact_trace_matches_forced_dense<R, D>(
    label: &str,
    typed: &TensorMap<R, D>,
    widen: impl Fn(D) -> Complex64 + Copy,
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: symtn::typed::TensorScalar + std::fmt::Debug,
{
    let dense: TensorMap<R, D> = forced_dense(typed);
    for pairs in [[(0usize, 1usize)], [(1usize, 0usize)]] {
        let actual: Complex64 = widen(typed.trace_pairs(&pairs).unwrap().scalar().unwrap());
        let oracle: Complex64 = widen(dense.trace_pairs(&pairs).unwrap().scalar().unwrap());
        assert!(
            (actual - oracle).norm() <= 1e-12 * oracle.norm().max(1.0),
            "{label} {pairs:?}: compact {actual:?} vs dense route {oracle:?}"
        );
    }
}

/// The freshly factorized bond, both compact swaps, and the compact adjoint.
fn sweep_compact_trace_variants<R, D>(
    label: &str,
    typed_s: &TensorMap<R, D>,
    widen: impl Fn(D) -> Complex64 + Copy,
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: symtn::typed::TensorScalar + std::fmt::Debug,
{
    let variants: Vec<(&str, TensorMap<R, D>)> = vec![
        ("plain", typed_s.clone()),
        ("transposed", full_transpose!(typed_s).unwrap()),
        ("permuted", typed_s.permute(&[1], &[0]).unwrap()),
        ("adjoint", typed_s.adjoint().unwrap()),
    ];
    for (which, typed) in &variants {
        assert_compact_trace_matches_forced_dense(&format!("{label} {which}"), typed, widen);
    }
}

#[test]
fn compact_full_trace_matches_the_forced_dense_route() {
    // What: the full trace of a rank-(1,1) compact spectrum factor over its
    // only pair is the same number on compact and forced-dense storage — U(1),
    // SU(2) (non-unit quantum dimensions), fZ2 (the twist makes it a
    // supertrace) and the packed U(1) x fZ2 product route, on plain and dual
    // bond legs, f64 and c64, both pair orders, and on legs the compact swaps
    // bent themselves.
    let _guard = cache_lock();
    let runtime = runtime();
    let real = |value: f64| value;
    let complex = |value: f64| Complex64::new(value, 0.25 + value % 3.0);
    let widen_real = |value: f64| Complex64::new(value, 0.0);
    let widen_complex = |value: Complex64| value;

    for is_dual in [false, true] {
        // U(1).
        let mut typed_leg: GradedSpace<symtn::sector::U1FusionRule> = GradedSpace::try_new(
            Arc::new(symtn::sector::U1FusionRule),
            [
                (symtn::sector::U1Irrep::new(-1), 2),
                (symtn::sector::U1Irrep::new(0), 3),
                (symtn::sector::U1Irrep::new(1), 2),
            ],
        )
        .unwrap();
        if is_dual {
            typed_leg = typed_leg.try_dual().unwrap();
        }
        let typed_s = compact_bond_trace(&runtime, &typed_leg, real);
        sweep_compact_trace_variants(&format!("u1 dual={is_dual} f64"), &typed_s, widen_real);
        let typed_s = compact_bond_trace(&runtime, &typed_leg, complex);
        sweep_compact_trace_variants(&format!("u1 dual={is_dual} c64"), &typed_s, widen_complex);

        // SU(2): dim(c) takes the values 1 and 2, so a coefficient-free
        // reduction cannot pass.
        let mut typed_leg: GradedSpace<symtn::sector::SU2FusionRule> = GradedSpace::try_new(
            Arc::new(symtn::sector::SU2FusionRule),
            [
                (SU2Irrep::from_twice_spin(0), 2),
                (SU2Irrep::from_twice_spin(1), 3),
            ],
        )
        .unwrap();
        if is_dual {
            typed_leg = typed_leg.try_dual().unwrap();
        }
        let typed_s = compact_bond_trace(&runtime, &typed_leg, real);
        sweep_compact_trace_variants(&format!("su2 dual={is_dual} f64"), &typed_s, widen_real);
        let typed_s = compact_bond_trace(&runtime, &typed_leg, complex);
        sweep_compact_trace_variants(&format!("su2 dual={is_dual} c64"), &typed_s, widen_complex);

        // fZ2: the twist is -1 on the odd sector, so this is where the
        // supertrace coefficient and its orientation live.
        let mut typed_leg: GradedSpace<symtn::sector::FermionParityFusionRule> =
            GradedSpace::try_new(
                Arc::new(symtn::sector::FermionParityFusionRule),
                [
                    (symtn::sector::Z2Irrep::EVEN, 2),
                    (symtn::sector::Z2Irrep::ODD, 3),
                ],
            )
            .unwrap();
        if is_dual {
            typed_leg = typed_leg.try_dual().unwrap();
        }
        let typed_s = compact_bond_trace(&runtime, &typed_leg, real);
        sweep_compact_trace_variants(&format!("fz2 dual={is_dual} f64"), &typed_s, widen_real);
        let typed_s = compact_bond_trace(&runtime, &typed_leg, complex);
        sweep_compact_trace_variants(&format!("fz2 dual={is_dual} c64"), &typed_s, widen_complex);

        // U(1) x fZ2, using the packed codec from the #589 section's rule
        // aliases: both the charge and parity factor must survive into the
        // coefficient.
        let product_label = |charge: i32, parity: u8| {
            symtn::sector::ProductSector::new(
                symtn::sector::U1Irrep::new(charge),
                parity_irrep(parity),
            )
        };
        let mut typed_leg: GradedSpace<U1Fz2Rule> = GradedSpace::try_new(
            Arc::new(U1Fz2Rule::new(
                symtn::sector::U1FusionRule,
                symtn::sector::FermionParityFusionRule,
            )),
            [(product_label(0, 0), 2), (product_label(1, 1), 3)],
        )
        .unwrap();
        if is_dual {
            typed_leg = typed_leg.try_dual().unwrap();
        }
        let typed_s = compact_bond_trace(&runtime, &typed_leg, real);
        sweep_compact_trace_variants(&format!("u1xfz2 dual={is_dual} f64"), &typed_s, widen_real);
        let typed_s = compact_bond_trace(&runtime, &typed_leg, complex);
        sweep_compact_trace_variants(
            &format!("u1xfz2 dual={is_dual} c64"),
            &typed_s,
            widen_complex,
        );
    }

    // Genuinely complex stored values (a spectrum has none out of `svd`, and
    // the adjoint variant only conjugates): a compact complex rotation stays
    // compact, and the forced-dense oracle covers it on the rule where the
    // twist could interact with the phase.
    let typed_leg: GradedSpace<symtn::sector::FermionParityFusionRule> = GradedSpace::try_new(
        Arc::new(symtn::sector::FermionParityFusionRule),
        [
            (symtn::sector::Z2Irrep::EVEN, 2),
            (symtn::sector::Z2Irrep::ODD, 3),
        ],
    )
    .unwrap();
    let typed_s = compact_bond_trace(&runtime, &typed_leg, complex);
    let rotated: TensorMap<symtn::sector::FermionParityFusionRule, Complex64> =
        typed_s.scale(Complex64::new(0.8, -0.6)).unwrap();
    assert_compact_trace_matches_forced_dense("fz2 rotated c64", &rotated, widen_complex);
    assert_compact_trace_matches_forced_dense(
        "fz2 rotated transposed c64",
        &full_transpose!(rotated).unwrap(),
        widen_complex,
    );
}

#[test]
fn compact_full_trace_is_the_supertrace_and_the_transpose_flips_it() {
    // What: on a single fermion-parity sector, `trace_pairs` and `tr` differ by
    // exactly the twist: an odd fZ2 bond flips the sign, an even one does not,
    // and the bosonic Z2 twin of the odd bond pins that the sign is the *twist*
    // and not the parity label. The transpose duals both bond legs, and the
    // traced channel is twisted only where its leg is not dual, so the same
    // tensor traces without the fermionic sign once transposed — which kills a
    // coefficient that reads the sector alone, or swaps the dual and non-dual
    // arms.
    let _guard = cache_lock();
    let runtime = runtime();

    let assert_supertrace =
        |name: &str, sign: f64, s: &TensorMap<symtn::sector::FermionParityFusionRule, f64>| {
            let positive: f64 = s.tr().unwrap();
            let traced: f64 = s.trace_pairs(&[(0, 1)]).unwrap().scalar().unwrap();
            assert_eq!(traced, sign * positive, "{name}");
            let transposed: TensorMap<symtn::sector::FermionParityFusionRule, f64> =
                full_transpose!(s).unwrap();
            let transposed_positive: f64 = transposed.tr().unwrap();
            let transposed_traced: f64 =
                transposed.trace_pairs(&[(0, 1)]).unwrap().scalar().unwrap();
            assert_eq!(transposed_traced, transposed_positive, "{name} transposed");
        };

    for (name, parity, sign) in [
        ("fz2 even", symtn::sector::Z2Irrep::EVEN, 1.0),
        ("fz2 odd", symtn::sector::Z2Irrep::ODD, -1.0),
    ] {
        let leg: GradedSpace<symtn::sector::FermionParityFusionRule> = GradedSpace::try_new(
            Arc::new(symtn::sector::FermionParityFusionRule),
            [(parity, 3)],
        )
        .unwrap();
        let mut next: f64 = 0.0;
        let source: TensorMap<symtn::sector::FermionParityFusionRule, f64> =
            TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, _| {
                next += 1.0;
                next
            })
            .unwrap();
        let s: TensorMap<symtn::sector::FermionParityFusionRule, f64> =
            source.svd_compact(&[0], &[1]).unwrap().s;
        assert_supertrace(name, sign, &s);
    }

    // The bosonic twin of the odd fixture: same parity label, twist +1, so the
    // supertrace *is* the positive trace here.
    let leg: GradedSpace<symtn::sector::Z2FusionRule> = GradedSpace::try_new(
        Arc::new(symtn::sector::Z2FusionRule),
        [(symtn::sector::Z2Irrep::ODD, 3)],
    )
    .unwrap();
    let mut next: f64 = 0.0;
    let source: TensorMap<symtn::sector::Z2FusionRule, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, _| {
            next += 1.0;
            next
        })
        .unwrap();
    let s: TensorMap<symtn::sector::Z2FusionRule, f64> = source.svd_compact(&[0], &[1]).unwrap().s;
    let positive: f64 = s.tr().unwrap();
    let traced: f64 = s.trace_pairs(&[(0, 1)]).unwrap().scalar().unwrap();
    assert_eq!(traced, positive, "z2 odd");
}

#[test]
fn compact_trace_of_no_pairs_returns_the_source_and_malformed_pairs_fail_first() {
    // What: the compact arm's boundaries. Tracing nothing on a compact factor
    // returns the source (the pre-guard short-circuit), and a malformed pair
    // list errors before the arm can run — the same validation order as on
    // dense storage. The dense geometries
    // outside the guard (rank > 2, partial pairs) are pinned by the
    // `trace_pairs` tests above.
    let _guard = cache_lock();
    let runtime = runtime();
    let s = z2_bond(&runtime);

    let untouched: TensorMap<symtn::sector::Z2FusionRule, f64> = s.trace_pairs(&[]).unwrap();
    assert_eq!(
        untouched.materialize().unwrap().dense_data().unwrap(),
        s.materialize().unwrap().dense_data().unwrap()
    );

    for pairs in [vec![(0usize, 9usize)], vec![(0, 0)], vec![(0, 1), (1, 0)]] {
        assert!(matches!(
            s.trace_pairs(&pairs).unwrap_err(),
            symtn::typed::Error::Operation(operation)
                if matches!(
                    *operation,
                    symtn::typed::OperationError::InvalidAxisSet { tensor: "trace pairs", .. }
                )
        ));
    }
}

/// `0` even, `1` odd, matching the public fermion-parity label contract.
// Strict on purpose: an out-of-range parity must not fold onto a valid label.
fn parity_irrep(parity: u8) -> symtn::sector::Z2Irrep {
    match parity {
        0 => symtn::sector::Z2Irrep::EVEN,
        1 => symtn::sector::Z2Irrep::ODD,
        other => panic!("not a fermion parity: {other} (parity is exactly 0 or 1)"),
    }
}
