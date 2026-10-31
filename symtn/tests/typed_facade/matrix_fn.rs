use super::*;
use symtn::typed::Engine;

// ---------------------------------------------------------------------------
// `inv`.
// ---------------------------------------------------------------------------

/// An endomorphism whose every coupled-sector block is nonsingular: the fill
/// used by [`z2_endomorphism`] is position-weighted and produces rank-one
/// blocks, so `inv` on it would be testing the singular path instead. Adding a
/// multiple of the identity is the cheapest fix.
fn z2_invertible(runtime: &Runtime) -> TensorMap<symtn::sector::Z2FusionRule, f64> {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let typed = z2_endomorphism(runtime);
    let typed_id =
        TensorMap::isomorphism(engine, runtime, &typed.domain(), &typed.domain()).unwrap();
    typed.axpby(engine, 1.0, &typed_id, 100.0).unwrap()
}

#[test]
fn inv_is_a_two_sided_inverse() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = runtime();
    let typed = z2_invertible(&runtime);

    let typed_inverse = typed.inv(engine, &[0], &[1]).unwrap();
    let expected =
        TensorMap::<_, f64>::isomorphism(engine, &runtime, &typed.domain(), &typed.domain())
            .unwrap();
    for (name, identity) in [
        ("t * inv(t)", typed.compose(engine, &typed_inverse).unwrap()),
        ("inv(t) * t", typed_inverse.compose(engine, &typed).unwrap()),
    ] {
        let error = identity
            .dense_data()
            .unwrap()
            .iter()
            .zip(expected.dense_data().unwrap())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(error < 1e-9, "{name} is not the identity: {error}");
    }
}

#[test]
fn inv_of_a_compact_spectrum_is_the_elementwise_reciprocal() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: the O(rank) arm. A spectrum's inverse is `1/s_i` on the stored
    // values.
    let runtime = runtime();
    let typed = z2_endomorphism(&runtime);

    let typed_s = typed.svd_compact(engine, &[0], &[1]).unwrap().s;
    // The fixture is rank deficient, so the full spectrum contains zeros that
    // `inv` must refuse; keep only the nonzero part.
    let (_, typed_s, _, _) = truncated_svd!(engine, typed_s, Truncation::Rank(2));

    let inverse = typed_s.inv(engine, &[0], &[1]).unwrap();
    let source_spectrum = symtn::expert::diagonal_spectrum(engine, &typed_s)
        .unwrap()
        .unwrap();
    let inverse_spectrum = symtn::expert::diagonal_spectrum(engine, &inverse)
        .unwrap()
        .unwrap();
    assert_eq!(source_spectrum.len(), inverse_spectrum.len());
    for (source, image) in source_spectrum.iter().zip(&inverse_spectrum) {
        assert_eq!(source.sector, image.sector);
        for (&value, &reciprocal) in source.values.iter().zip(&image.values) {
            assert_eq!(reciprocal, value.recip());
        }
    }
    // `s * s^-1` is the identity on the bond.
    let product = typed_s.compose(engine, &inverse).unwrap();
    let expected =
        TensorMap::<_, f64>::isomorphism(engine, &runtime, &typed_s.domain(), &typed_s.domain())
            .unwrap();
    let error = product
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(expected.dense_data().unwrap())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(error < 1e-9, "s * inv(s) is not the identity: {error}");
}

#[test]
fn inv_reports_a_singular_input_as_a_typed_error() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: singular input is a `Result`, never a panic, and both storages
    // report it as the same numerical failure although the compact arm
    // detects it by inspecting the stored value and the dense one inside the
    // LAPACK solve.
    let runtime = runtime();
    let typed = z2_endomorphism(&runtime);

    // Compact: a spectrum scaled to exactly zero. Why not the tail of a
    // rank-deficient SVD: those singular values come back tiny but nonzero, and
    // the arm under test compares against exact zero, not a tolerance.
    let spectrum = typed
        .svd_compact(engine, &[0], &[1])
        .unwrap()
        .s
        .scale(engine, 0.0)
        .unwrap();
    let singular = |result: Result<_, symtn::typed::Error>| {
        matches!(
            result,
            Err(symtn::typed::Error::Operation(error)) if matches!(
                *error,
                symtn::typed::OperationError::Dense(symtn_dense::DenseError::NumericalFailure { .. })
            )
        )
    };
    assert!(singular(spectrum.inv(engine, &[0], &[1])));

    // Dense: an all-zero endomorphism.
    let zeros = typed.scale(engine, 0.0).unwrap();
    assert!(singular(zeros.inv(engine, &[0], &[1])));
}

#[test]
fn inv_accepts_isomorphic_but_unequal_codomain_and_domain() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: TensorKit's `inv` asks for `codomain ≅ domain`, not `==`, and
    // returns `domain <- codomain`. The seam agrees: a rank-one codomain and a
    // rank-two domain with the same coupled-sector dimensions is accepted, and
    // the result carries the swapped spaces. This is a behavior pin — the
    // rustdoc states it, so a seam that tightened to equality must fail here.
    let runtime = runtime();
    let provider = Arc::new(symtn::sector::Z2FusionRule);
    let wide = GradedSpace::try_new(
        provider.clone(),
        [
            (symtn::sector::Z2Irrep::EVEN, 2),
            (symtn::sector::Z2Irrep::ODD, 2),
        ],
    )
    .unwrap();
    // `narrow ⊗ narrow` has coupled dimensions (even 2, odd 2) as well, so the
    // two sides are isomorphic while the hom spaces differ in rank.
    let narrow = GradedSpace::try_new(
        provider,
        [
            (symtn::sector::Z2Irrep::EVEN, 1),
            (symtn::sector::Z2Irrep::ODD, 1),
        ],
    )
    .unwrap();
    let mut next = 0.0;
    let tensor =
        TensorMap::from_subblock_fn(engine, &runtime, [&wide], [&narrow, &narrow], |_, _| {
            next += 1.0;
            next * next
        })
        .unwrap();

    let inverse = tensor.inv(engine, &[0], &[1, 2]).unwrap();
    assert_eq!(inverse.codomain().len(), 2);
    assert_eq!(inverse.domain().len(), 1);
    let identity = tensor.compose(engine, &inverse).unwrap();
    let expected = TensorMap::<_, f64>::isomorphism(engine, &runtime, [&wide], [&wide]).unwrap();
    let error = identity
        .dense_data()
        .unwrap()
        .iter()
        .zip(expected.dense_data().unwrap())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(error < 1e-9, "t * inv(t) is not the identity: {error}");
}

// ---------------------------------------------------------------------------
// `pinv`.
// ---------------------------------------------------------------------------

#[test]
fn pinv_satisfies_the_moore_penrose_identities() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // The fixture is deliberately rank deficient: a full-rank one would let a
    // broken cutoff pass as an ordinary inverse.
    let runtime = runtime();
    let typed = z2_endomorphism(&runtime);

    // `rcond` is well above the fixture's numerically-zero singular values and
    // well below its real ones, so the cutoff drops exactly the null directions
    // — inverting those instead would amplify rounding into the millions of ulp.
    let pseudo = typed.pinv(engine, &[0], &[1], 1e-6).unwrap();
    let left_support = typed.compose(engine, &pseudo).unwrap();
    let right_support = pseudo.compose(engine, &typed).unwrap();
    let assert_close =
        |engine: &mut dyn Engine,
         name: &str,
         actual: &TensorMap<symtn::sector::Z2FusionRule, f64>,
         expected: &TensorMap<symtn::sector::Z2FusionRule, f64>| {
            let error = actual
                .materialize(engine)
                .unwrap()
                .dense_data()
                .unwrap()
                .iter()
                .zip(expected.dense_data().unwrap())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
            assert!(
                error < 1e-9 * expected.norm(engine, 2.0).unwrap().max(1.0),
                "{name}: {error}"
            );
        };
    let left_support_compose = &left_support.compose(engine, &typed).unwrap();
    assert_close(engine, "t t^+ t != t", left_support_compose, &typed);
    let right_support_compose = &right_support.compose(engine, &pseudo).unwrap();
    assert_close(engine, "t^+ t t^+ != t^+", right_support_compose, &pseudo);
    let left_support_adjoint = &left_support.adjoint(engine).unwrap();
    assert_close(
        engine,
        "(t t^+)† != t t^+",
        left_support_adjoint,
        &left_support,
    );
    let right_support_adjoint = &right_support.adjoint(engine).unwrap();
    assert_close(
        engine,
        "(t^+ t)† != t^+ t",
        right_support_adjoint,
        &right_support,
    );
}

#[test]
fn pinv_of_a_compact_spectrum_is_the_elementwise_cutoff_reciprocal() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: the O(rank) arm — an elementwise cutoff and reciprocal, whose own
    // singular values are `|entry|`, so no SVD runs at all.
    let runtime = runtime();
    let typed = z2_endomorphism(&runtime);
    let typed_s = typed.svd_compact(engine, &[0], &[1]).unwrap().s;
    let source = symtn::expert::diagonal_spectrum(engine, &typed_s)
        .unwrap()
        .unwrap();
    let sigma_max = source
        .iter()
        .flat_map(|entry| &entry.values)
        .map(|value| value.abs())
        .fold(0.0, f64::max);

    for rcond in [0.0, 1e-12, 1e-3] {
        let typed_s_pinv = typed_s.pinv(engine, &[0], &[1], rcond).unwrap();
        let image = symtn::expert::diagonal_spectrum(engine, &typed_s_pinv)
            .unwrap()
            .unwrap();
        let cutoff = rcond * sigma_max;
        for (source, image) in source.iter().zip(&image) {
            assert_eq!(source.sector, image.sector);
            for (&value, &actual) in source.values.iter().zip(&image.values) {
                let expected = if value.abs() > cutoff {
                    value.recip()
                } else {
                    0.0
                };
                assert_eq!(actual, expected, "rcond {rcond}, value {value}");
            }
        }
    }
}

#[test]
fn pinv_cuts_a_singular_value_sitting_exactly_on_the_cutoff() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: the boundary. The comparison is `sigma > rcond * sigma_max`, so a
    // singular value at *exactly* the cutoff is discarded, not kept — on both
    // storages. This is the one bit of the cutoff policy a
    // mutation to `>=` would otherwise slip past.
    let runtime = runtime();
    let leg = GradedSpace::try_new(
        Arc::new(symtn::sector::Z2FusionRule),
        [(symtn::sector::Z2Irrep::EVEN, 2)],
    )
    .unwrap();
    // Diagonal with entries 4 and 1: sigma_max is 4, so rcond = 0.25 puts the
    // second singular value exactly on the cutoff. Both are powers of two, so
    // the product is exact and the comparison is not floating-point weather.
    let tensor =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, indices: &[usize]| {
            if indices[0] != indices[1] {
                0.0
            } else if indices[0] == 0 {
                4.0
            } else {
                1.0
            }
        })
        .unwrap();
    assert_eq!(0.25 * 4.0, 1.0, "the fixture's cutoff must be exact");

    let dense_pinv = tensor.pinv(engine, &[0], &[1], 0.25).unwrap();
    // Kept: 1/4 for the surviving value. Cut: an exact 0 where 1/1 would be.
    let mut kept: Vec<f64> = dense_pinv
        .dense_data()
        .unwrap()
        .iter()
        .copied()
        .filter(|v| *v != 0.0)
        .collect();
    kept.sort_by(f64::total_cmp);
    assert_eq!(kept, vec![0.25], "the boundary singular value survived");

    // A discarded nonzero mode makes this the exact Moore-Penrose inverse of
    // the hard-thresholded effective-rank tensor, not of the original tensor.
    let thresholded =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, indices: &[usize]| {
            if indices[0] == 0 && indices[1] == 0 {
                4.0
            } else {
                0.0
            }
        })
        .unwrap();
    let triple = tensor
        .compose(engine, &dense_pinv)
        .unwrap()
        .compose(engine, &tensor)
        .unwrap();
    assert_eq!(
        triple.dense_data().unwrap(),
        thresholded.dense_data().unwrap()
    );
    assert_ne!(triple.dense_data().unwrap(), tensor.dense_data().unwrap());

    // And on the compact arm.
    let spectrum = tensor.svd_compact(engine, &[0], &[1]).unwrap().s;
    let compact_pinv = spectrum.pinv(engine, &[0], &[1], 0.25).unwrap();
    let mut kept: Vec<f64> = compact_pinv
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .copied()
        .filter(|v| *v != 0.0)
        .collect();
    kept.sort_by(f64::total_cmp);
    assert_eq!(
        kept,
        vec![0.25],
        "the boundary value survived the compact arm"
    );
}

#[test]
fn pinv_rejects_a_nonfinite_or_negative_rcond_before_any_work() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: `rcond` is validated at the facade, so a bad one never reaches the
    // SVD — and the compact arm validates it too, which a guard placed only on
    // the dense route would miss.
    let runtime = runtime();
    let typed = z2_endomorphism(&runtime);
    let spectrum = typed.svd_compact(engine, &[0], &[1]).unwrap().s;

    for rcond in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(
            matches!(
                typed.pinv(engine, &[0], &[1], rcond),
                Err(symtn::typed::Error::InvalidArgument(_))
            ),
            "dense pinv accepted rcond {rcond}"
        );
        assert!(
            matches!(
                spectrum.pinv(engine, &[0], &[1], rcond),
                Err(symtn::typed::Error::InvalidArgument(_))
            ),
            "compact pinv accepted rcond {rcond}"
        );
    }
}

#[test]
fn pinv_uses_one_global_sigma_max_across_every_sector() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: the cutoff is relative to the largest singular value of the *whole*
    // tensor, not of each coupled sector — the deliberate divergence from
    // TensorKit's per-block `rtol`.
    //
    // The global maximum deliberately lives in the **second** sector. A fold
    // that only ever looks at the first sector reads `sigma_max = 1` here, which
    // puts the cutoff at 0.5 and keeps everything — so that weaker mutant fails
    // this test, as does the per-sector one, which cannot cut anything in a 1x1
    // sector at all. Both were run by hand against this fixture and both fail
    // it; with the maximum in the first sector, the first-sector-only mutant
    // survived, because there the two folds happen to agree.
    let runtime = runtime();
    let leg = GradedSpace::try_new(
        Arc::new(symtn::sector::Z2FusionRule),
        [
            (symtn::sector::Z2Irrep::EVEN, 1),
            (symtn::sector::Z2Irrep::ODD, 1),
        ],
    )
    .unwrap();
    // Even sector (stored first): 1. Odd sector: 1024. Each sector is 1x1, so
    // per-sector sigma_max would be the entry itself and nothing could ever be
    // cut; and the global maximum is not in the sector a first-sector-only fold
    // would find.
    let tensor = TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |trees, _| {
        if *trees.coupled() == symtn::sector::Z2Irrep::EVEN {
            1.0
        } else {
            1024.0
        }
    })
    .unwrap();

    let pseudo = tensor.pinv(engine, &[0], &[1], 0.5).unwrap();
    let mut kept: Vec<f64> = pseudo
        .dense_data()
        .unwrap()
        .iter()
        .copied()
        .filter(|v| *v != 0.0)
        .collect();
    kept.sort_by(f64::total_cmp);
    assert_eq!(
        kept,
        vec![1.0 / 1024.0],
        "a per-sector cutoff kept the small sector"
    );
    // The compact arm's own `max|entry|` is global for the same reason.
    let spectrum = tensor.svd_compact(engine, &[0], &[1]).unwrap().s;
    let kept = spectrum
        .pinv(engine, &[0], &[1], 0.5)
        .unwrap()
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .copied()
        .filter(|v| *v != 0.0)
        .count();
    assert_eq!(kept, 1, "the compact arm used a per-sector cutoff");
}

// ---------------------------------------------------------------------------
// `exp`.
// ---------------------------------------------------------------------------

#[test]
fn exp_accepts_a_non_hermitian_endomorphism_and_inverts_under_negation() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: like TensorKit's `exp`, a general per-block Pade approximant
    // with no hermiticity gate. `exp(A) exp(-A) = id` pins the actual exponential.
    let runtime = runtime();
    let typed = z2_endomorphism(&runtime);

    assert!(!is_hermitian!(engine, typed, 1e-9));
    let typed_exp = typed.exp(engine, &[0], &[1]).unwrap();

    // exp(A) exp(-A) = id, evaluated through the typed composition.
    let inverse = typed
        .scale(engine, -1.0)
        .unwrap()
        .exp(engine, &[0], &[1])
        .unwrap();
    let identity =
        TensorMap::<_, f64>::isomorphism(engine, &runtime, &typed.domain(), &typed.domain())
            .unwrap();
    let residual = typed_exp
        .compose(engine, &inverse)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(identity.dense_data().unwrap())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(residual < 1e-11, "exp(A) exp(-A) != id: {residual}");
}

#[test]
fn exp_of_a_compact_spectrum_stays_compact_and_is_elementwise() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: `exp(s_i)` on the `Σ_c k_c` stored values, matching TensorKit's
    // `exp(::DiagonalTensorMap)` law.
    let runtime = runtime();
    let typed = z2_endomorphism(&runtime);
    // Scaled down: the fixture's largest singular value is in the thousands and
    // `exp` of it overflows to infinity, which no comparison can separate from
    // a wrong infinity.
    let typed_s = typed
        .svd_compact(engine, &[0], &[1])
        .unwrap()
        .s
        .scale(engine, 1e-3)
        .unwrap();

    let typed_exp = typed_s.exp(engine, &[0], &[1]).unwrap();
    // Every stored value is `exp` of the source's: the elementwise claim, read
    // off the materialized diagonal so it does not need a compact accessor.
    for (index, (source, image)) in typed_s
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(typed_exp.materialize(engine).unwrap().dense_data().unwrap())
        .enumerate()
    {
        let expected = if *source == 0.0 && !on_diagonal(&typed_s, index) {
            // Off-diagonal of the block-diagonal materialization: `exp` of a
            // diagonal is diagonal, so these stay zero rather than becoming 1.
            0.0
        } else {
            source.exp()
        };
        assert!(
            (image - expected).abs() < 1e-12,
            "entry {index}: {image} is not exp({source})"
        );
    }
}

/// Whether storage position `index` sits on a block's own diagonal. Used to
/// read a compact tensor's elementwise claim off its dense materialization.
fn on_diagonal<R, D>(tensor: &TensorMap<R, D>, index: usize) -> bool
where
    R: symtn::sector::MultiplicityFreeRigidSymbols<Scalar = f64>
        + symtn::sector::CheckedFusionAlgebra
        + symtn::sector::SectorCodec,
    D: symtn::typed::TensorScalar,
{
    (0..tensor.subblock_count()).any(|block| {
        let block = tensor.subblock(block).unwrap();
        let shape = block.shape();
        (0..shape[0]).any(|row| {
            index == block.offset() + row * block.strides()[0] + row * block.strides()[1]
        })
    })
}

#[test]
fn exp_of_a_complex_compact_spectrum_takes_the_complex_elementwise_branch() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: the compact arm is TensorKit's `exp(::DiagonalTensorMap)`, which is
    // unconditionally elementwise — so a c64 spectrum with a nonreal entry, the
    // case the Hermitian dense arm would refuse, comes back as `exp` of that
    // entry. Storage therefore *does* change what `exp` accepts, exactly as it
    // does in TensorKit; the rustdoc says so and this is the pin.
    let runtime = runtime();
    let leg = GradedSpace::try_new(
        Arc::new(symtn::sector::Z2FusionRule),
        [(symtn::sector::Z2Irrep::EVEN, 2)],
    )
    .unwrap();
    let dense =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, indices: &[usize]| {
            if indices[0] == indices[1] {
                Complex64::new(0.0, indices[0] as f64)
            } else {
                Complex64::new(0.0, 0.0)
            }
        })
        .unwrap();
    // Dense storage of the very same matrix: it is accepted
    // too, through the general Pade arm — and because this particular matrix is
    // already diagonal, the two arms must agree entry for entry. Storage no
    // longer decides *whether* `exp` is defined, only how it is computed.
    assert!(!is_hermitian!(engine, dense, 1e-9));
    let dense_exponential = dense.exp(engine, &[0], &[1]).unwrap();
    for (index, (source, value)) in dense
        .dense_data()
        .unwrap()
        .iter()
        .zip(dense_exponential.dense_data().unwrap())
        .enumerate()
    {
        let expected = if on_diagonal(&dense, index) {
            source.exp()
        } else {
            Complex64::new(0.0, 0.0)
        };
        assert!(
            (value - expected).norm() < 1e-12,
            "dense entry {index}: {value} is not exp(engine, {source})"
        );
    }

    // Compact storage of the same values: accepted, elementwise.
    let spectrum = dense
        .eig_full(engine, &[0], &[1])
        .unwrap()
        .d
        .scale(engine, Complex64::new(1.0, 0.0))
        .unwrap();
    let image = spectrum.exp(engine, &[0], &[1]).unwrap();
    for (index, (source, value)) in spectrum
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(image.materialize(engine).unwrap().dense_data().unwrap())
        .enumerate()
    {
        let expected = if *source == Complex64::new(0.0, 0.0) && !on_diagonal(&spectrum, index) {
            Complex64::new(0.0, 0.0)
        } else {
            source.exp()
        };
        assert!(
            (value - expected).norm() < 1e-12,
            "entry {index}: {value} is not exp({source})"
        );
    }
}

fn bits_f64(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn map_diagonal_matches_hand_built_diagonals_on_a_dual_u1_bond() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: the result is the compact diagonal whose values are `f` of the
    // stored values, sector by sector, on the receiver's own (dual) bond.
    // Oracle: diagonals built from hand-computed literals.
    let runtime = runtime();
    let bond = GradedSpace::try_new(
        Arc::new(symtn::sector::U1FusionRule),
        [
            (symtn::sector::U1Irrep::new(0), 2),
            (symtn::sector::U1Irrep::new(1), 1),
            (symtn::sector::U1Irrep::new(-2), 2),
        ],
    )
    .unwrap()
    .try_dual()
    .unwrap();
    assert!(bond.is_dual());
    let q = symtn::sector::U1Irrep::new;
    let source: TensorMap<_, f64> = TensorMap::diagonal(
        engine,
        &runtime,
        &bond,
        spectra(&[
            (q(0), &[4.0, 9.0][..]),
            (q(-1), &[0.25][..]),
            (q(2), &[16.0, 1.0][..]),
        ]),
    )
    .unwrap();

    let root = source.map_diagonal(engine, f64::sqrt).unwrap();
    let expected: TensorMap<_, f64> = TensorMap::diagonal(
        engine,
        &runtime,
        &bond,
        spectra(&[
            (q(0), &[2.0, 3.0][..]),
            (q(-1), &[0.5][..]),
            (q(2), &[4.0, 1.0][..]),
        ]),
    )
    .unwrap();
    assert_eq!(root.codomain(), vec![bond.clone()]);
    assert_eq!(root.domain(), vec![bond.clone()]);
    assert_eq!(
        root.diagview(engine).unwrap(),
        expected.diagview(engine).unwrap()
    );
    assert_eq!(
        bits_f64(root.materialize(engine).unwrap().dense_data().unwrap()),
        bits_f64(expected.materialize(engine).unwrap().dense_data().unwrap())
    );

    let reciprocal = source.map_diagonal(engine, |value| 1.0 / value).unwrap();
    let expected: TensorMap<_, f64> = TensorMap::diagonal(
        engine,
        &runtime,
        &bond,
        spectra(&[
            (q(0), &[0.25, 1.0 / 9.0][..]),
            (q(-1), &[4.0][..]),
            (q(2), &[0.0625, 1.0][..]),
        ]),
    )
    .unwrap();
    assert_eq!(
        bits_f64(
            reciprocal
                .materialize(engine)
                .unwrap()
                .dense_data()
                .unwrap()
        ),
        bits_f64(expected.materialize(engine).unwrap().dense_data().unwrap())
    );

    // Complex payload: `f` decides the branch; `Complex64::sqrt` is principal.
    let source: TensorMap<_, Complex64> = TensorMap::diagonal(
        engine,
        &runtime,
        &bond,
        spectra(&[
            (
                q(0),
                &[Complex64::new(-1.0, 0.0), Complex64::new(4.0, 0.0)][..],
            ),
            (q(-1), &[Complex64::new(0.0, 2.0)][..]),
            (
                q(2),
                &[Complex64::new(9.0, 0.0), Complex64::new(-4.0, 0.0)][..],
            ),
        ]),
    )
    .unwrap();
    let root = source.map_diagonal(engine, |value| value.sqrt()).unwrap();
    let expected = [
        Complex64::new(0.0, 1.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(1.0, 1.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(0.0, 2.0),
    ];
    let actual: Vec<_> = root
        .diagview(engine)
        .unwrap()
        .into_iter()
        .flat_map(|entry| entry.values)
        .collect();
    let sorted = |mut values: Vec<Complex64>| {
        values.sort_by(|a, b| (a.re, a.im).partial_cmp(&(b.re, b.im)).unwrap());
        values
    };
    let (actual, expected) = (sorted(actual), sorted(expected.to_vec()));
    for (value, expected) in actual.iter().zip(&expected) {
        assert!((value - expected).norm() < 1e-15, "{value} != {expected}");
    }
}

#[test]
fn map_diagonal_on_su2_repeats_each_value_over_its_carrier_dimension() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: `f` acts on the reduced values once per degeneracy index, and the
    // physical-basis expansion repeats each result `2j + 1` times on the
    // diagonal. Oracle: the hand-listed physical diagonal.
    let runtime = runtime();
    let j = SU2Irrep::from_twice_spin;
    let bond =
        GradedSpace::try_new(Arc::new(SU2FusionRule), [(j(0), 2), (j(1), 1), (j(2), 2)]).unwrap();
    let source: TensorMap<_, f64> = TensorMap::diagonal(
        engine,
        &runtime,
        &bond,
        spectra(&[
            (j(0), &[4.0, 9.0][..]),
            (j(1), &[0.25][..]),
            (j(2), &[16.0, 1.0][..]),
        ]),
    )
    .unwrap();
    let root = source.map_diagonal(engine, f64::sqrt).unwrap();
    let expected: TensorMap<_, f64> = TensorMap::diagonal(
        engine,
        &runtime,
        &bond,
        spectra(&[
            (j(0), &[2.0, 3.0][..]),
            (j(1), &[0.5][..]),
            (j(2), &[4.0, 1.0][..]),
        ]),
    )
    .unwrap();
    assert_eq!(
        root.diagview(engine).unwrap(),
        expected.diagview(engine).unwrap()
    );
    assert_eq!(
        bits_f64(root.materialize(engine).unwrap().dense_data().unwrap()),
        bits_f64(expected.materialize(engine).unwrap().dense_data().unwrap())
    );

    let physical = root.to_physical_dense(engine).unwrap();
    assert_eq!(physical.shape, vec![10, 10]);
    let mut diagonal = Vec::new();
    for row in 0..10 {
        for col in 0..10 {
            let value = physical.data[row + 10 * col];
            if row == col {
                diagonal.push(value);
            } else {
                assert_eq!(value, 0.0, "off-diagonal ({row}, {col})");
            }
        }
    }
    diagonal.sort_by(f64::total_cmp);
    let mut expected = vec![2.0, 3.0, 0.5, 0.5, 4.0, 4.0, 4.0, 1.0, 1.0, 1.0];
    expected.sort_by(f64::total_cmp);
    for (value, expected) in diagonal.iter().zip(&expected) {
        assert!((value - expected).abs() < 1e-14, "{value} != {expected}");
    }
}

#[test]
fn map_diagonal_keeps_the_result_compact() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: the result is stored as a compact diagonal (no dense payload), for
    // the `s` of `svd_compact` and for the compact adjoint of a c64 diagonal.
    let runtime = runtime();
    let s = z2_endomorphism(&runtime)
        .svd_compact(engine, &[0], &[1])
        .unwrap()
        .s;
    assert!(symtn::expert::diagonal_spectrum(engine, &s)
        .unwrap()
        .is_some());
    let root = s.map_diagonal(engine, f64::sqrt).unwrap();
    let stored = symtn::expert::diagonal_spectrum(engine, &root)
        .unwrap()
        .unwrap();
    let source = symtn::expert::diagonal_spectrum(engine, &s)
        .unwrap()
        .unwrap();
    for (root, source) in stored.iter().zip(&source) {
        assert_eq!(root.sector, source.sector);
        let expected: Vec<_> = source.values.iter().map(|value| value.sqrt()).collect();
        assert_eq!(bits_f64(&root.values), bits_f64(&expected));
    }

    let complex = s
        .convert::<Complex64>(engine)
        .unwrap()
        .scale(engine, Complex64::new(0.0, 1.0))
        .unwrap();
    let adjoint = complex.adjoint(engine).unwrap();
    let mapped = adjoint.map_diagonal(engine, |value| value * value).unwrap();
    assert!(symtn::expert::diagonal_spectrum(engine, &mapped)
        .unwrap()
        .is_some());
}

#[test]
fn map_diagonal_rejects_dense_storage() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: dense storage is refused with a typed error naming the expected
    // input, even when the dense blocks happen to be diagonal, and a lazy
    // adjoint of a dense tensor is refused the same way.
    let runtime = runtime();
    let leg = GradedSpace::try_new(
        Arc::new(symtn::sector::Z2FusionRule),
        [(symtn::sector::Z2Irrep::EVEN, 2)],
    )
    .unwrap();
    let dense_diagonal: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, ij: &[usize]| {
            f64::from(ij[0] == ij[1])
        })
        .unwrap();
    for (name, tensor) in [
        ("dense diagonal", dense_diagonal.clone()),
        ("lazy adjoint", dense_diagonal.adjoint(engine).unwrap()),
        ("dense endomorphism", z2_endomorphism(&runtime)),
        ("non-bond", z2_tensor(&runtime)),
    ] {
        match tensor.map_diagonal(engine, f64::sqrt) {
            Err(symtn::typed::Error::InvalidArgument(message)) => assert!(
                message.contains("compact diagonal") && message.contains("dense storage"),
                "{name}: {message}"
            ),
            other => panic!("{name}: dense input was accepted: {other:?}"),
        }
    }
}

#[test]
fn c64_compact_inv_and_pinv_are_elementwise_reciprocals() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // A c64 tensor's singular values are stored in the payload dtype, so both
    // compact matrix functions must apply their laws directly to those values.
    let runtime = runtime();
    // A full-rank c64 `[v] <- [v]` map with a wide spectrum.
    let complex = |value: f64| Complex64::new(value, 1.0 + value % 5.0);
    let leg = GradedSpace::try_new(
        Arc::new(symtn::sector::Z2FusionRule),
        [
            (symtn::sector::Z2Irrep::EVEN, 16),
            (symtn::sector::Z2Irrep::ODD, 17),
        ],
    )
    .unwrap();
    let mut state = 0x05ee_dc64_u64;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        complex(((state >> 33) as f64) / (u32::MAX as f64) + 0.5)
    };
    let typed =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, _| next()).unwrap();
    let typed_s = typed.svd_compact(engine, &[0], &[1]).unwrap().s;
    let source = symtn::expert::diagonal_spectrum(engine, &typed_s)
        .unwrap()
        .unwrap();
    let sigma_max = source
        .iter()
        .flat_map(|entry| &entry.values)
        .map(|value| value.norm())
        .fold(0.0, f64::max);
    let rcond = 1e-12;
    let typed_s_inv = typed_s.inv(engine, &[0], &[1]).unwrap();
    let inverse = symtn::expert::diagonal_spectrum(engine, &typed_s_inv)
        .unwrap()
        .unwrap();
    let typed_s_pinv = typed_s.pinv(engine, &[0], &[1], rcond).unwrap();
    let pseudo = symtn::expert::diagonal_spectrum(engine, &typed_s_pinv)
        .unwrap()
        .unwrap();
    for ((source, inverse), pseudo) in source.iter().zip(&inverse).zip(&pseudo) {
        assert_eq!(source.sector, inverse.sector);
        assert_eq!(source.sector, pseudo.sector);
        for ((&value, &inverse), &pseudo) in source
            .values
            .iter()
            .zip(&inverse.values)
            .zip(&pseudo.values)
        {
            // Within 1 ulp, not bitwise, per `docs/testing_numerics.md`: the
            // compact reciprocal runs a literal port of Julia's
            // `inv(::ComplexF64)` (`symtn/src/typed.rs`
            // `julia_complex64_reciprocal`), whose fast path uses `mul_add`
            // where the naive `1/z` computed here does not, so this is a
            // different valid floating-point order of the same formula, not
            // an exact-equality contract. `scaled_complex64_reciprocal.rs`
            // is the bitwise-against-Julia-itself oracle for this port.
            let reciprocal = Complex64::new(1.0, 0.0) / value;
            ulp::assert_complex_within_ulps(inverse, reciprocal, 1, "inv");
            let expected = if value.norm() > rcond * sigma_max {
                reciprocal
            } else {
                Complex64::new(0.0, 0.0)
            };
            ulp::assert_complex_within_ulps(pseudo, expected, 1, "pinv");
        }
    }
}
