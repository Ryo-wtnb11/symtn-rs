use super::*;

// ---------------------------------------------------------------------------
// Decompositions.
//
// Factorizations are checked through their mathematical contracts.
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// The general eigendecompositions.
// ---------------------------------------------------------------------------

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
