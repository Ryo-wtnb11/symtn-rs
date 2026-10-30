use super::*;

// ---------------------------------------------------------------------------
// Decompositions (issue #567).
//
// Factorizations are checked through their mathematical contracts.
// ---------------------------------------------------------------------------

fn typed_z2_spectrum(
    spectrum: &[symtn::typed::SectorSpectrum<symtn::sector::Z2Irrep>],
) -> Vec<(symtn::sector::Z2Irrep, Vec<f64>)> {
    spectrum
        .iter()
        .map(|entry| (entry.sector, entry.values.clone()))
        .collect()
}

/// `u * s * vh` through the typed `contract`, for a `[2] <- [1]` factor chain:
/// `u`'s last axis is its bond, `s` is `bond <- bond`, `vh` is `bond <- rest`.
fn recompose(
    u: &TensorMap<symtn::sector::Z2FusionRule, f64>,
    s: &TensorMap<symtn::sector::Z2FusionRule, f64>,
    vh: &TensorMap<symtn::sector::Z2FusionRule, f64>,
) -> TensorMap<symtn::sector::Z2FusionRule, f64> {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let us = u
        .contract(
            engine,
            s,
            &ContractSpec {
                lhs: &[2],
                rhs: &[0],
                codomain: &[0, 1],
                domain: &[2],
            },
        )
        .unwrap();
    us.contract(
        engine,
        vh,
        &ContractSpec {
            lhs: &[2],
            rhs: &[0],
            codomain: &[0, 1],
            domain: &[2],
        },
    )
    .unwrap()
}

#[test]
fn svd_compact_reconstructs_the_source_through_the_typed_contract() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: the factors really are a factorization in this facade's own
    // vocabulary. The composition runs through `contract`, which agrees with
    // `compose` for this bosonic rule.
    let runtime = runtime();
    let typed = z2_tensor(&runtime);

    let Svd { u, s, vh } = typed.svd_compact(engine, &[0, 1], &[2]).unwrap();
    let recon = recompose(&u, &s, &vh);

    assert_eq!(
        recon.dense_data().unwrap().len(),
        typed.dense_data().unwrap().len()
    );
    for (got, want) in recon
        .dense_data()
        .unwrap()
        .iter()
        .zip(typed.dense_data().unwrap())
    {
        assert!(
            (got - want).abs() <= 1e-12 * want.abs().max(1.0),
            "{got} vs {want}"
        );
    }
}

#[test]
fn svd_compact_reconstructs_a_complex_payload() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: c64 takes the complex factorization route. The imaginary part is
    // deliberately not proportional to the real one, so a stray conjugation
    // or a real-only path is visible.
    let runtime = runtime();
    let typed = z2_complex_tensor(&runtime);
    let Svd {
        u: tu,
        s: ts,
        vh: tvh,
    } = typed.svd_compact(engine, &[0, 1], &[2]).unwrap();

    let recon = tu
        .contract(
            engine,
            &ts,
            &ContractSpec {
                lhs: &[2],
                rhs: &[0],
                codomain: &[0, 1],
                domain: &[2],
            },
        )
        .unwrap()
        .contract(
            engine,
            &tvh,
            &ContractSpec {
                lhs: &[2],
                rhs: &[0],
                codomain: &[0, 1],
                domain: &[2],
            },
        )
        .unwrap();
    for (got, want) in recon
        .dense_data()
        .unwrap()
        .iter()
        .zip(typed.dense_data().unwrap())
    {
        assert!(
            (got - want).norm() <= 1e-12 * want.norm().max(1.0),
            "{got} vs {want}"
        );
    }
}

#[test]
fn svd_full_reconstructs_with_unitary_outer_factors() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = runtime();
    for num_codomain in [2, 1] {
        let typed = z2_tensor_split(&runtime, num_codomain);
        let Svd { u, s, vh } = typed
            .svd_full(engine, &codomain_axes(&typed), &domain_axes(&typed))
            .unwrap();

        let recon = u.compose(engine, &s).unwrap().compose(engine, &vh).unwrap();
        assert_data_close_f64(recon.dense_data().unwrap(), typed.dense_data().unwrap());
        assert!(is_isometric!(engine, u, 1e-12));
        assert!(is_isometric!(engine, vh.adjoint(engine).unwrap(), 1e-12));
        assert_same_legs(&u.codomain(), &typed.codomain());
        assert_same_legs(&vh.domain(), &typed.domain());
        assert_same_legs(&u.domain(), &s.codomain());
        assert_same_legs(&s.domain(), &vh.codomain());
    }
}

/// The truncated eigendecomposition `(d, v)` of `eigh_full` / `eig_full`
/// output, composed the same way. Evaluates to `(d, v, error)`.
macro_rules! truncated_eigen {
    ($engine:ident, $full:expr, $truncation:expr) => {{
        // `$full` is an `Eigh` or an `Eig`; both name their factors `d`, `v`.
        let full = $full.unwrap();
        let (d, v) = (full.d, full.v);
        let found = d.domain()[0]
            .find_truncated(&d.diagview($engine).unwrap(), &$truncation)
            .unwrap();
        (
            d.restrict_leg($engine, &[(0, &found.selection), (1, &found.selection)])
                .unwrap(),
            v.restrict_leg($engine, &[(v.codomain_rank(), &found.selection)])
                .unwrap(),
            found.error,
        )
    }};
}

#[test]
fn truncated_svd_reconstructs_and_reports_the_discarded_weight() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = runtime();
    let typed = z2_tensor(&runtime);

    let truncation = symtn::typed::Truncation::rank(2);
    let (u, s, vh, error) = truncated_svd!(engine, typed, truncation);

    assert_same_legs(&u.codomain(), &typed.codomain());
    assert_same_legs(&vh.domain(), &typed.domain());
    assert_same_legs(&u.domain(), &s.codomain());
    assert_same_legs(&s.domain(), &vh.codomain());

    // The reported error is the 2-norm of everything the truncation dropped.
    // Z2 is a group, so every quantum dimension is one and the weighting is
    // the identity — the check is then a plain sum of squares.
    let full = typed.svd_vals(engine, &[0, 1], &[2]).unwrap();
    let kept = typed_z2_spectrum(&s.diagview(engine).unwrap());
    let mut discarded = 0.0;
    for entry in &full {
        let kept_here = kept
            .iter()
            .find(|(sector, _)| sector == &entry.sector)
            .map_or(0, |(_, values)| values.len());
        for value in &entry.values[kept_here..] {
            discarded += value * value;
        }
    }
    assert!(discarded > 0.0, "the fixture must actually truncate");
    assert!((error - discarded.sqrt()).abs() < 1e-12);

    let recon = recompose(&u, &s, &vh);
    let reconstruction_error = recon
        .dense_data()
        .unwrap()
        .iter()
        .zip(typed.dense_data().unwrap())
        .map(|(got, want)| (got - want) * (got - want))
        .sum::<f64>()
        .sqrt();
    assert!((error - reconstruction_error).abs() < 1e-12);

    // A degenerate but well-formed policy is a policy, not an error: keeping
    // nothing succeeds and discards the whole spectrum.
    let (_, empty_s, _, _) = truncated_svd!(engine, typed, symtn::typed::Truncation::Rank(0));
    assert!(empty_s
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .is_empty());
    assert!(empty_s
        .diagview(engine)
        .unwrap()
        .iter()
        .all(|entry| entry.values.is_empty()));
}

#[test]
fn a_spectrum_decode_failure_comes_back_as_the_codec_error() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: a codec that cannot decode a coupled sector the engine produced
    // fails the call with the provider's own error instead of panicking inside
    // the label map.
    //
    // This is the Err path worth testing because a degenerate `Truncation` is
    // not one: `Rank(0)`, `Rank(usize::MAX)` and `All(vec![])` are all
    // constructible and all legitimately succeed (see the `Rank(0)` assertion
    // in the truncation test), and the states that would fail validation are
    // unreachable from outside `symtn-matrixalgebra` — their variants are
    // `#[non_exhaustive]` and their constructors are fallible. So the provider
    // is the only input to these methods a caller can actually malform.
    let runtime = runtime();
    let provider = Arc::new(external_z3_with(Quirk::NarrowDecode));
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(Z3Charge(0), 1), (Z3Charge(1), 2)]).unwrap();
    // Two charge-1 codomain legs couple to charge 2, the id this codec refuses.
    // `zeros` never decodes, so the tensor builds and the failure lands in the
    // spectrum decode.
    let tensor =
        TensorMap::<ExternalZ3Rule, f64>::zeros(engine, &runtime, [&leg, &leg], [&leg, &leg])
            .unwrap();

    assert!(matches!(
        tensor.svd_vals(engine, &[0, 1], &[2, 3]).unwrap_err(),
        symtn::typed::Error::FusionAlgebra(_)
    ));
    assert!(matches!(
        tensor
            .svd_compact(engine, &[0, 1], &[2, 3])
            .unwrap()
            .s
            .diagview(engine)
            .unwrap_err(),
        symtn::typed::Error::FusionAlgebra(_)
    ));
}

#[test]
fn svd_vals_reports_exact_per_label_spectra() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = runtime();
    let leg = GradedSpace::try_new(
        Arc::new(symtn::sector::Z2FusionRule),
        [
            (symtn::sector::Z2Irrep::EVEN, 3),
            (symtn::sector::Z2Irrep::ODD, 2),
        ],
    )
    .unwrap();
    // Each block is diagonal, so its singular values are the absolute diagonal
    // entries. Both sectors are deliberately signed and out of magnitude order.
    let typed = TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |trees, indices| {
        if indices[0] != indices[1] {
            return 0.0;
        }
        if *trees.coupled() == symtn::sector::Z2Irrep::EVEN {
            [-2.0, 5.0, -1.0][indices[0]]
        } else {
            assert_eq!(*trees.coupled(), symtn::sector::Z2Irrep::ODD);
            [3.0, -4.0][indices[0]]
        }
    })
    .unwrap();

    let spectrum = typed.svd_vals(engine, &[0], &[1]).unwrap();
    assert_eq!(
        typed_z2_spectrum(&spectrum),
        [
            (symtn::sector::Z2Irrep::EVEN, vec![5.0, 2.0, 1.0]),
            (symtn::sector::Z2Irrep::ODD, vec![4.0, 3.0]),
        ]
    );
    // `Z2Irrep` orders exactly as its ids do, so this fixture cannot tell a
    // label sort from an id sort. The next test does.
}

#[test]
fn svd_vals_sorts_by_label_where_that_differs_from_the_id_order() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: the O2' promise is *label* order, and the only way to see it is a
    // provider whose codec does not order its labels the way the engine orders
    // its ids. `Quirk::ReversedLabels` is exactly that — a valid codec, not a
    // broken one — so the seam hands the spectrum back in the reversed order
    // and only the facade's own sort puts it right.
    let runtime = runtime();
    let provider = Arc::new(external_z3_with(Quirk::ReversedLabels));
    let tensor = z3_rank_four(&runtime, &provider);

    let labels: Vec<u8> = tensor
        .svd_vals(engine, &[0, 1], &[2, 3])
        .unwrap()
        .iter()
        .map(|entry| entry.sector.0)
        .collect();

    assert_eq!(labels, [0, 1, 2]);
}

#[test]
fn qr_and_lq_reconstruct_with_the_expected_isometries_and_spaces() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // Both splits are exercised deliberately: `2 <- 1` is tall in every coupled
    // sector, where LQ-compact and LQ-full return the same factors and so
    // cannot tell the two seams apart; `1 <- 2` is wide, where they differ (and
    // symmetrically for QR).
    let runtime = runtime();
    for num_codomain in [2, 1] {
        let typed = z2_tensor_split(&runtime, num_codomain);

        let qr_compact = typed
            .qr_compact(engine, &codomain_axes(&typed), &domain_axes(&typed))
            .unwrap();
        let qr_full = typed
            .qr_full(engine, &codomain_axes(&typed), &domain_axes(&typed))
            .unwrap();
        for Qr { q, r } in [&qr_compact, &qr_full] {
            assert_data_close_f64(
                q.compose(engine, r).unwrap().dense_data().unwrap(),
                typed.dense_data().unwrap(),
            );
            assert!(is_isometric!(engine, q, 1e-12));
            assert_same_legs(&q.codomain(), &typed.codomain());
            assert_same_legs(&r.domain(), &typed.domain());
            assert_same_legs(&q.domain(), &r.codomain());
        }

        let lq_compact = typed
            .lq_compact(engine, &codomain_axes(&typed), &domain_axes(&typed))
            .unwrap();
        let lq_full = typed
            .lq_full(engine, &codomain_axes(&typed), &domain_axes(&typed))
            .unwrap();
        for Lq { l, q } in [&lq_compact, &lq_full] {
            assert_data_close_f64(
                l.compose(engine, q).unwrap().dense_data().unwrap(),
                typed.dense_data().unwrap(),
            );
            assert!(is_isometric!(engine, q.adjoint(engine).unwrap(), 1e-12));
            assert_same_legs(&l.codomain(), &typed.codomain());
            assert_same_legs(&q.domain(), &typed.domain());
            assert_same_legs(&l.domain(), &q.codomain());
        }

        // Each orientation distinguishes one compact seam from its full
        // sibling through the internal bond space.
        if num_codomain == 2 {
            assert_ne!(
                typed_leg_shapes(&qr_compact.q),
                typed_leg_shapes(&qr_full.q)
            );
        } else {
            assert_ne!(
                typed_leg_shapes(&lq_compact.q),
                typed_leg_shapes(&lq_full.q)
            );
        }
    }
}

#[test]
fn left_and_right_null_spaces_annihilate_the_source() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = runtime();
    for num_codomain in [2, 1] {
        let typed = z2_tensor_split(&runtime, num_codomain);
        let left = typed
            .left_null(engine, &codomain_axes(&typed), &domain_axes(&typed))
            .unwrap();
        let right = typed
            .right_null(engine, &codomain_axes(&typed), &domain_axes(&typed))
            .unwrap();

        assert_same_legs(&left.codomain(), &typed.codomain());
        assert_same_legs(&right.domain(), &typed.domain());
        assert!(is_isometric!(engine, left, 1e-12));
        assert!(is_isometric!(engine, right.adjoint(engine).unwrap(), 1e-12));
        assert!(left
            .adjoint(engine)
            .unwrap()
            .compose(engine, &typed)
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .all(|value| value.abs() < 1e-12));
        let right_adjoint = right.adjoint(engine).unwrap();
        assert!(typed
            .compose(engine, &right_adjoint)
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .all(|value| value.abs() < 1e-12));

        let left_null_dimensions = left.domain()[0].degeneracies().to_vec();
        let right_null_dimensions = right.codomain()[0].degeneracies().to_vec();
        // The null dimension is `(rows - cols)+` per sector, from the space
        // alone: the fixture's rank-deficient sector adds none. The tall form
        // leaves 11 and 9 on the left bond and nothing on the right;
        // repartitioning to the wide form exchanges the two sides.
        if num_codomain == 2 {
            assert_eq!(left_null_dimensions, [11, 9]);
            assert!(right_null_dimensions.is_empty());
        } else {
            assert!(left_null_dimensions.is_empty());
            assert_eq!(right_null_dimensions, [11, 9]);
        }
    }
}

#[test]
fn decompositions_carry_a_fermionic_provider() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // The one provider this facade can host whose braiding is not symmetric.
    // Every quantum dimension is one for `FermionParity`, and the fusion-tree
    // storage of a block *is* its coupled-sector matricization, so the sum of
    // squared singular values is the sum of squared stored elements — a
    // hand-checkable identity that no gauge convention can move.
    let runtime = runtime();
    let tensor = fermionic_rank_three(&runtime);

    let spectrum = tensor.svd_vals(engine, &[0, 1], &[2]).unwrap();
    let from_spectrum: f64 = spectrum
        .iter()
        .flat_map(|entry| entry.values.iter())
        .map(|value| value * value)
        .sum();
    let from_data: f64 = tensor
        .dense_data()
        .unwrap()
        .iter()
        .map(|value| value * value)
        .sum();
    assert!((from_spectrum - from_data).abs() < 1e-12);

    // And the seam is reachable at all for this provider, in both directions.
    let Qr { q, r } = tensor.qr_compact(engine, &[0, 1], &[2]).unwrap();
    assert_eq!(q.codomain().len(), 2);
    assert_eq!(r.domain().len(), 1);
}

#[test]
fn decompositions_carry_an_external_provider_with_its_own_labels() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // A downstream provider drives the same surface, and its spectrum comes
    // back in its own labels rather than raw ids.
    let runtime = runtime();
    let provider = Arc::new(external_z3());
    let tensor = z3_rank_four(&runtime, &provider);

    let spectrum = tensor.svd_vals(engine, &[0, 1], &[2, 3]).unwrap();
    assert!(!spectrum.is_empty());
    assert!(spectrum.iter().all(|entry| entry.sector.0 < 3));
    assert!(spectrum.windows(2).all(|w| w[0].sector < w[1].sector));

    let (_, s, _, error) = truncated_svd!(engine, tensor, symtn::typed::Truncation::Full);
    assert_eq!(error, 0.0);
    assert_eq!(
        s.diagview(engine)
            .unwrap()
            .iter()
            .map(|entry| entry.sector)
            .collect::<Vec<_>>(),
        spectrum
            .iter()
            .map(|entry| entry.sector)
            .collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// The Hermitian eigendecompositions.
// ---------------------------------------------------------------------------

/// A typed Hermitian endomorphism: `p = t + t†`, so `eigh` is defined
/// without projecting first.
fn z2_hermitian(runtime: &Runtime) -> TensorMap<symtn::sector::Z2FusionRule, f64> {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let typed = z2_endomorphism(runtime);
    let typed_adjoint = typed.adjoint(engine).unwrap();
    typed.axpby(engine, 1.0, &typed_adjoint, 1.0).unwrap()
}

#[test]
fn eigh_full_reconstructs_the_source_through_compose() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: `v * d * v†` is the source. The middle composition takes the
    // compact bond-scaling arm, so this exercises the storage as well as the
    // factorization.
    let runtime = runtime();
    let typed = z2_hermitian(&runtime);

    let Eigh { d, v } = typed
        .eigh_full(engine, &[0], &[1], HermitianTol::DEFAULT)
        .unwrap();
    let v_adjoint = v.adjoint(engine).unwrap();
    let recon = v
        .compose(engine, &d)
        .unwrap()
        .compose(engine, &v_adjoint)
        .unwrap();

    assert_eq!(
        recon.dense_data().unwrap().len(),
        typed.dense_data().unwrap().len()
    );
    for (got, want) in recon
        .dense_data()
        .unwrap()
        .iter()
        .zip(typed.dense_data().unwrap())
    {
        assert!(
            (got - want).abs() <= 1e-10 * want.abs().max(1.0),
            "{got} vs {want}"
        );
    }
}

#[test]
fn eigh_vals_follow_the_provider_label_order() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: caller-facing spectra use provider labels in canonical order.
    let runtime = runtime();
    let typed = z2_hermitian(&runtime);
    let spectrum = typed
        .eigh_vals(engine, &[0], &[1], HermitianTol::DEFAULT)
        .unwrap();

    assert_eq!(
        spectrum
            .iter()
            .map(|entry| entry.sector)
            .collect::<Vec<_>>(),
        [symtn::sector::Z2Irrep::EVEN, symtn::sector::Z2Irrep::ODD]
    );
    assert!(spectrum.iter().all(|entry| !entry.values.is_empty()));
}

#[test]
fn truncated_eigh_reports_the_discarded_eigenvalue_norm() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = runtime();
    let typed = z2_hermitian(&runtime);
    let truncation = Truncation::rank(3);

    let mut magnitudes: Vec<_> = typed
        .eigh_vals(engine, &[0], &[1], HermitianTol::DEFAULT)
        .unwrap()
        .into_iter()
        .flat_map(|entry| entry.values)
        .map(f64::abs)
        .collect();
    magnitudes.sort_by(|lhs, rhs| rhs.total_cmp(lhs));
    let expected_error = magnitudes[3..]
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let (d, _, error) = truncated_eigen!(
        engine,
        typed.eigh_full(engine, &[0], &[1], HermitianTol::DEFAULT),
        truncation
    );

    assert_eq!(
        d.diagview(engine)
            .unwrap()
            .iter()
            .map(|entry| entry.values.len())
            .sum::<usize>(),
        3
    );
    assert!(
        expected_error > 0.0,
        "the fixture must discard nonzero values"
    );
    assert!((error - expected_error).abs() < 1e-12 * expected_error.max(1.0));
    assert!(
        d.materialize(engine).unwrap().dense_data().unwrap().len()
            < typed
                .eigh_full(engine, &[0], &[1], HermitianTol::DEFAULT)
                .unwrap()
                .d
                .materialize(engine)
                .unwrap()
                .dense_data()
                .unwrap()
                .len()
    );
}

#[test]
fn eigh_reports_a_non_hermitian_input_rather_than_a_wrong_answer() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = runtime();
    let typed = z2_endomorphism(&runtime);

    assert!(typed
        .eigh_full(engine, &[0], &[1], HermitianTol::DEFAULT)
        .err()
        .unwrap()
        .report()
        .contains("eigh requires Hermitian coupled-sector blocks"));
    assert!(typed
        .eigh_vals(engine, &[0], &[1], HermitianTol::DEFAULT)
        .err()
        .unwrap()
        .report()
        .contains("eigh requires Hermitian coupled-sector blocks"));
}

// ---------------------------------------------------------------------------
// The general eigendecompositions.
// ---------------------------------------------------------------------------

/// The typed c64 endomorphism `eig` needs: [`z2_complex_tensor`] is rank
/// three, and `eig` is defined on square maps only.
fn z2_complex_endo(runtime: &Runtime) -> TensorMap<symtn::sector::Z2FusionRule, Complex64> {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let complex = |value: f64| Complex64::new(value, 1.0 + value % 5.0);
    let leg = GradedSpace::try_new(
        Arc::new(symtn::sector::Z2FusionRule),
        [
            (symtn::sector::Z2Irrep::EVEN, 2),
            (symtn::sector::Z2Irrep::ODD, 3),
        ],
    )
    .unwrap();
    TensorMap::from_subblock_fn(engine, runtime, [&leg], [&leg], |trees, indices| {
        complex(typed_fill_value(trees, indices))
    })
    .unwrap()
}

#[test]
fn eig_full_satisfies_the_eigen_equation_for_a_real_payload() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: a real input promotes to complex factors satisfying A V = V D.
    let runtime = runtime();
    let typed = z2_endomorphism(&runtime);

    let Eig { d, v } = typed.eig_full(engine, &[0], &[1]).unwrap();
    let av = typed
        .convert::<Complex64>(engine)
        .unwrap()
        .compose(engine, &v)
        .unwrap();
    let vd = v.compose(engine, &d).unwrap();

    assert_data_close_c64(av.dense_data().unwrap(), vd.dense_data().unwrap());
}

#[test]
fn complex_eig_satisfies_the_eigen_equation_and_conjugates_its_spectrum() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: the native complex route satisfies A V = V D, and the compact
    // spectrum's adjoint conjugates genuinely nonreal eigenvalues.
    let runtime = runtime();
    let typed = z2_complex_endo(&runtime);

    let Eig { d, v } = typed.eig_full(engine, &[0], &[1]).unwrap();
    assert_data_close_c64(
        typed.compose(engine, &v).unwrap().dense_data().unwrap(),
        v.compose(engine, &d).unwrap().dense_data().unwrap(),
    );

    // Genuinely complex, so a missing conjugation is observable.
    assert!(
        d.materialize(engine)
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .any(|value| value.im.abs() > 1e-6),
        "the eig spectrum must be off the real axis for this to test anything"
    );
    let adjoint = d.adjoint(engine).unwrap();
    for (conjugated, original) in adjoint
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(d.materialize(engine).unwrap().dense_data().unwrap())
    {
        assert_eq!(*conjugated, original.conj());
    }
}

#[test]
fn eig_vals_are_label_ordered_and_trunc_reports_the_discarded_norm() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = runtime();
    let typed = z2_endomorphism(&runtime);
    let spectrum = typed.eig_vals(engine, &[0], &[1]).unwrap();
    assert_eq!(
        spectrum
            .iter()
            .map(|entry| entry.sector)
            .collect::<Vec<_>>(),
        [symtn::sector::Z2Irrep::EVEN, symtn::sector::Z2Irrep::ODD]
    );

    let truncation = Truncation::rank(3);
    let mut magnitudes: Vec<_> = spectrum
        .into_iter()
        .flat_map(|entry| entry.values)
        .map(|value| value.norm())
        .collect();
    magnitudes.sort_by(|lhs, rhs| rhs.total_cmp(lhs));
    let expected_error = magnitudes[3..]
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let (d, _, error) = truncated_eigen!(engine, typed.eig_full(engine, &[0], &[1]), truncation);

    assert_eq!(
        d.diagview(engine)
            .unwrap()
            .iter()
            .map(|entry| entry.values.len())
            .sum::<usize>(),
        3
    );
    assert!(
        expected_error > 0.0,
        "the fixture must discard nonzero values"
    );
    assert!((error - expected_error).abs() < 1e-12 * expected_error.max(1.0));
}

// ---------------------------------------------------------------------------
// The `is_hermitian` / `project_*` family.
// ---------------------------------------------------------------------------

#[test]
fn hermitian_projections_satisfy_their_identities_and_predicate_truth_table() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: the complementary projections reconstruct the source and their
    // adjoint symmetries agree with the typed predicate family.
    let runtime = runtime();
    let typed = z2_endomorphism(&runtime);
    let tol = 1e-10;

    let hermitian = project_hermitian!(engine, typed).unwrap();
    let antihermitian = project_antihermitian!(engine, typed).unwrap();
    assert_data_close_f64(
        hermitian
            .axpby(engine, 1.0, &antihermitian, 1.0)
            .unwrap()
            .dense_data()
            .unwrap(),
        typed.dense_data().unwrap(),
    );
    assert_data_close_f64(
        hermitian
            .adjoint(engine)
            .unwrap()
            .materialize(engine)
            .unwrap()
            .dense_data()
            .unwrap(),
        hermitian.dense_data().unwrap(),
    );
    assert_data_close_f64(
        antihermitian
            .adjoint(engine)
            .unwrap()
            .materialize(engine)
            .unwrap()
            .dense_data()
            .unwrap(),
        antihermitian
            .scale(engine, -1.0)
            .unwrap()
            .dense_data()
            .unwrap(),
    );
    assert_data_close_f64(
        project_hermitian!(engine, hermitian)
            .unwrap()
            .dense_data()
            .unwrap(),
        hermitian.dense_data().unwrap(),
    );
    assert_data_close_f64(
        project_antihermitian!(engine, antihermitian)
            .unwrap()
            .dense_data()
            .unwrap(),
        antihermitian.dense_data().unwrap(),
    );

    let cases = [
        (&typed, [false, false, false, false, false]),
        (&hermitian, [true, false, false, false, false]),
        (&antihermitian, [false, true, false, false, false]),
    ];
    for (tensor, expected) in cases {
        assert_eq!(
            [
                is_hermitian!(engine, tensor, tol),
                is_antihermitian!(engine, tensor, tol),
                is_isometric!(engine, tensor, tol),
                is_unitary!(engine, tensor, tol),
                is_posdef!(engine, tensor, tol),
            ],
            expected
        );
    }
}

#[test]
fn isometry_and_posdef_see_their_positive_cases() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: the two members the fixture above only ever answers `false` for.
    // `u` from an SVD is isometric by construction, and `t† t` is positive
    // definite when `t` has full rank.
    let runtime = runtime();
    let typed = z2_tensor(&runtime);
    let tol = 1e-9;

    let tu = typed.svd_compact(engine, &[0, 1], &[2]).unwrap().u;
    assert!(is_isometric!(engine, tu, tol));
    // Isometric but not unitary: `u` is tall here.
    assert!(!is_unitary!(engine, tu, tol));

    // `2 * id` is Hermitian with every eigenvalue at 2: positive definite on
    // any provider, and the cheapest tensor that is.
    let typed_positive = TensorMap::isomorphism(engine, &runtime, &typed.domain(), &typed.domain())
        .unwrap()
        .scale(engine, 2.0)
        .unwrap();
    assert!(is_hermitian!(engine, typed_positive, tol));
    assert!(is_posdef!(engine, typed_positive, tol));
    // Hermitian but not positive definite: the same tensor negated.
    let negated = typed_positive.scale(engine, -1.0).unwrap();
    assert!(is_hermitian!(engine, negated, tol));
    assert!(!is_posdef!(engine, negated, tol));
    // Positive *semi*definite is `false`, not `true`: TensorKit's `isposdef` is
    // Cholesky-based and strict, and this facade's rustdoc promises the same.
    // A real diagonal endomorphism with one entry at exactly zero is the case
    // that separates `>` from `>=` — `eigh` on it returns that zero exactly, so
    // the comparison is not floating-point weather.
    let semidefinite_leg = GradedSpace::try_new(
        Arc::new(symtn::sector::Z2FusionRule),
        [
            (symtn::sector::Z2Irrep::EVEN, 2),
            (symtn::sector::Z2Irrep::ODD, 3),
        ],
    )
    .unwrap();
    let semidefinite = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&semidefinite_leg],
        [&semidefinite_leg],
        |_, indices: &[usize]| {
            if indices[0] != indices[1] {
                0.0
            } else {
                // Row 0 of every block is the zero eigenvalue.
                indices[0] as f64
            }
        },
    )
    .unwrap();
    assert!(is_hermitian!(engine, semidefinite, 0.0));
    assert!(semidefinite
        .eigh_vals(engine, &[0], &[1], HermitianTol::DEFAULT)
        .unwrap()
        .iter()
        .any(|entry| entry.values.contains(&0.0)));
    assert!(
        !is_posdef!(engine, semidefinite, 0.0),
        "a positive semidefinite tensor must not be reported positive definite"
    );

    // This rank-deficient fixture's Gram matrix is Hermitian but not strictly
    // positive definite.
    let tgram = typed
        .adjoint(engine)
        .unwrap()
        .compose(engine, &typed)
        .unwrap();
    assert!(is_hermitian!(engine, tgram, tol));
    assert!(!is_posdef!(engine, tgram, tol));
}

#[test]
fn a_non_endomorphism_is_never_hermitian_and_never_errors() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: predicates are total and false; projections have no endomorphism to
    // return and therefore reject the input.
    let runtime = runtime();
    let typed = z2_tensor(&runtime);

    assert!(!is_hermitian!(engine, typed, 1e-9));
    assert!(!is_antihermitian!(engine, typed, 1e-9));
    assert!(!is_posdef!(engine, typed, 1e-9));
    assert!(project_hermitian!(engine, typed)
        .err()
        .unwrap()
        .report()
        .contains("tensors live on different spaces or block layouts"));
    assert!(project_antihermitian!(engine, typed)
        .err()
        .unwrap()
        .report()
        .contains("tensors live on different spaces or block layouts"));
}

#[test]
fn typed_polar_reconstructs_the_input_f64_u1_and_c64_fz2() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // Gate 1: `t = w ∘ p` (left) and `t = p ∘ w` (right) at factorization
    // tolerance, on both dtypes and on both a bosonic and a fermionic rule.
    let runtime = runtime();

    let leg = u1_typed_leg();
    let tall: TensorMap<symtn::sector::U1FusionRule, f64> =
        TensorMap::rand_with_seed(engine, &runtime, [&leg, &leg], [&leg], 3).unwrap();
    let LeftPolar { w, p } = tall.left_polar(engine, &[0, 1], &[2]).unwrap();
    assert_data_close_f64(
        w.compose(engine, &p).unwrap().dense_data().unwrap(),
        tall.dense_data().unwrap(),
    );
    let wide: TensorMap<symtn::sector::U1FusionRule, f64> =
        TensorMap::rand_with_seed(engine, &runtime, [&leg], [&leg, &leg], 5).unwrap();
    let RightPolar { p, wh: w } = wide.right_polar(engine, &[0], &[1, 2]).unwrap();
    assert_data_close_f64(
        p.compose(engine, &w).unwrap().dense_data().unwrap(),
        wide.dense_data().unwrap(),
    );

    let leg = fz2_typed_leg();
    let tall: TensorMap<symtn::sector::FermionParityFusionRule, Complex64> =
        TensorMap::rand_with_seed(engine, &runtime, [&leg, &leg], [&leg], 7).unwrap();
    let LeftPolar { w, p } = tall.left_polar(engine, &[0, 1], &[2]).unwrap();
    assert_data_close_c64(
        w.compose(engine, &p).unwrap().dense_data().unwrap(),
        tall.dense_data().unwrap(),
    );
    let wide: TensorMap<symtn::sector::FermionParityFusionRule, Complex64> =
        TensorMap::rand_with_seed(engine, &runtime, [&leg], [&leg, &leg], 11).unwrap();
    let RightPolar { p, wh: w } = wide.right_polar(engine, &[0], &[1, 2]).unwrap();
    assert_data_close_c64(
        p.compose(engine, &w).unwrap().dense_data().unwrap(),
        wide.dense_data().unwrap(),
    );
}

#[test]
fn typed_polar_factor_laws_hold() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // Gate 2: `w† ∘ w = id(domain)` for `left_polar` (resp. `w ∘ w† = id` on
    // the rows for `right_polar`); `p` Hermitian with non-negative spectrum,
    // read through `eigh_vals`.
    let runtime = runtime();

    // c64 fermionic left arm: a stray conjugation or sign is visible here.
    let leg = fz2_typed_leg();
    let tall: TensorMap<symtn::sector::FermionParityFusionRule, Complex64> =
        TensorMap::rand_with_seed(engine, &runtime, [&leg, &leg], [&leg], 13).unwrap();
    let LeftPolar { w, p } = tall.left_polar(engine, &[0, 1], &[2]).unwrap();
    let id: TensorMap<symtn::sector::FermionParityFusionRule, Complex64> =
        TensorMap::isomorphism(engine, &runtime, [&leg], [&leg]).unwrap();
    assert_data_close_c64(
        w.adjoint(engine)
            .unwrap()
            .compose(engine, &w)
            .unwrap()
            .dense_data()
            .unwrap(),
        id.dense_data().unwrap(),
    );
    assert!(is_hermitian!(engine, p, 1e-12));
    for entry in p
        .eigh_vals(engine, &[0], &[1], HermitianTol::DEFAULT)
        .unwrap()
    {
        assert!(entry.values.iter().all(|&value| value >= -1e-12));
    }

    // f64 U(1) right arm.
    let leg = u1_typed_leg();
    let wide: TensorMap<symtn::sector::U1FusionRule, f64> =
        TensorMap::rand_with_seed(engine, &runtime, [&leg], [&leg, &leg], 17).unwrap();
    let RightPolar { p, wh: w } = wide.right_polar(engine, &[0], &[1, 2]).unwrap();
    let id: TensorMap<symtn::sector::U1FusionRule, f64> =
        TensorMap::isomorphism(engine, &runtime, [&leg], [&leg]).unwrap();
    let w_adjoint = w.adjoint(engine).unwrap();
    assert_data_close_f64(
        w.compose(engine, &w_adjoint).unwrap().dense_data().unwrap(),
        id.dense_data().unwrap(),
    );
    assert!(is_hermitian!(engine, p, 1e-12));
    for entry in p
        .eigh_vals(engine, &[0], &[1], HermitianTol::DEFAULT)
        .unwrap()
    {
        assert!(entry.values.iter().all(|&value| value >= -1e-12));
    }
}

#[test]
fn typed_polar_factor_spaces_match_tensorkit() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // Gate 3: factor *spaces*, not just shapes — TK 0.17
    // `factorizations/matrixalgebrakit.jl:204-214`: left `W` lives on
    // `space(t)` and `P` on `domain ← domain`; right `P` on
    // `codomain ← codomain` and `Wᴴ` on `space(t)`. A dual leg sits on each
    // side so a dropped duality flag is visible.
    let runtime = runtime();
    let leg = u1_typed_leg();
    let dual = leg.try_dual().unwrap();

    let tall: TensorMap<symtn::sector::U1FusionRule, f64> =
        TensorMap::rand_with_seed(engine, &runtime, [&leg, &dual], [&dual], 19).unwrap();
    let LeftPolar { w, p } = tall.left_polar(engine, &[0, 1], &[2]).unwrap();
    assert_same_legs(&w.codomain(), &tall.codomain());
    assert_same_legs(&w.domain(), &tall.domain());
    assert_same_legs(&p.codomain(), &tall.domain());
    assert_same_legs(&p.domain(), &tall.domain());

    let wide: TensorMap<symtn::sector::U1FusionRule, f64> =
        TensorMap::rand_with_seed(engine, &runtime, [&dual], [&leg, &dual], 23).unwrap();
    let RightPolar { p, wh: w } = wide.right_polar(engine, &[0], &[1, 2]).unwrap();
    assert_same_legs(&p.codomain(), &wide.codomain());
    assert_same_legs(&p.domain(), &wide.codomain());
    assert_same_legs(&w.codomain(), &wide.codomain());
    assert_same_legs(&w.domain(), &wide.domain());
}

#[test]
fn typed_polar_wrong_side_rectangular_reports_the_requested_direction() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // Gate 4: the split-2 fixture is tall in every coupled sector, the split-1
    // one wide — so `right_polar` on the former and `left_polar` on the latter
    // return the seam's exact wrong-side errors, unfiltered.
    let runtime = runtime();

    let typed_tall = z2_tensor_split(&runtime, 2);
    assert!(matches!(
        typed_tall.right_polar(engine, &[0, 1], &[2]).unwrap_err(),
        symtn::typed::Error::Operation(error)
            if matches!(
                error.as_ref(),
                symtn::typed::OperationError::InvalidArgument { message }
                    if *message
                        == "right_polar requires columns >= rows in every coupled-sector matrix"
            )
    ));

    let typed_wide = z2_tensor_split(&runtime, 1);
    assert!(matches!(
        typed_wide.left_polar(engine, &[0], &[1, 2]).unwrap_err(),
        symtn::typed::Error::Operation(error)
            if matches!(
                error.as_ref(),
                symtn::typed::OperationError::InvalidArgument { message }
                    if *message
                        == "left_polar requires rows >= columns in every coupled-sector matrix"
            )
    ));
}

#[test]
fn typed_polar_carries_an_external_provider() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // Gate 5: `ExternalZ3` is a downstream-style provider, so this is a
    // provider-generic law check —
    // reconstruction plus the isometry law — driving the same context lane an
    // external provider reaches through `multiplicity_free_lane`.
    let runtime = runtime();
    let provider = Arc::new(external_z3());
    // Same legs on both sides (no duals): every coupled-sector matrix is
    // square, so both polars are defined. `z3_rank_four`'s dual domain is not
    // usable here — a dual flips each charge, so some coupled sectors come out
    // wider than tall and `left_polar` rightly refuses them.
    let wide = z3_leg(&provider, false);
    let narrow = z3_other_leg(&provider, false);
    let mut counter = 0.0;
    let tensor: TensorMap<ExternalZ3Rule, f64> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&wide, &narrow],
        [&wide, &narrow],
        |_, _| {
            counter += 1.0;
            counter
        },
    )
    .unwrap();

    let LeftPolar { w, p } = tensor.left_polar(engine, &[0, 1], &[2, 3]).unwrap();
    assert_data_close_f64(
        w.compose(engine, &p).unwrap().dense_data().unwrap(),
        tensor.dense_data().unwrap(),
    );
    let gram = w.adjoint(engine).unwrap().compose(engine, &w).unwrap();
    let id: TensorMap<ExternalZ3Rule, f64> =
        TensorMap::isomorphism(engine, &runtime, [&wide, &narrow], [&wide, &narrow]).unwrap();
    assert_data_close_f64(gram.dense_data().unwrap(), id.dense_data().unwrap());
}

// ---------------------------------------------------------------------------
// Typed inspection, scalar, zeros_like and dtype conversions.
// ---------------------------------------------------------------------------

#[test]
fn typed_rank_accessors_on_a_mixed_fixture() {
    let _serial = crate::counting_alloc::serial();
    // Gate 1: ranks on a mixed 2 <- 1 fixture.
    let runtime = runtime();
    let typed = z2_tensor(&runtime);
    assert_eq!(typed.codomain_rank(), 2);
    assert_eq!(typed.domain_rank(), 1);
    assert_eq!(typed.rank(), 3);
}
