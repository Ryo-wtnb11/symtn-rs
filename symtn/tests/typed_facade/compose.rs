use super::*;

// ---------------------------------------------------------------------------
// `TensorMap::compose`.
// ---------------------------------------------------------------------------

/// A position-weighted fill whose distinct entries expose individual signs.
fn fermionic_typed_fill(
    sectors: &symtn::typed::BlockFusionTrees<symtn::sector::Z2Irrep>,
    indices: &[usize],
) -> f64 {
    let mut value = f64::from(sectors.coupled().parity()) * 1000.0;
    for (position, label) in sectors.codomain_uncoupled().iter().enumerate() {
        value += f64::from(label.parity()) * 100.0 * (position + 1) as f64;
    }
    for (position, label) in sectors.domain_uncoupled().iter().enumerate() {
        value += f64::from(label.parity()) * 10.0 * (position + 1) as f64;
    }
    value + 1.0 + indices.iter().sum::<usize>() as f64
}

/// `a : [v] <- [v*, v]` and `b : [v*, v] <- [v]`.
///
/// Composition contracts two legs here, exactly one of which is **dual** —
/// and a dual leg is the only place a fermionic twist can act. The mixed pair
/// is deliberate: with every contracted leg dual, "twist the dual contracted
/// legs" and "twist all contracted legs" would be the same statement, and the
/// twist identity below would not be pinned to a leg set at all.
fn fermionic_compose_pair(
    runtime: &Runtime,
) -> (
    TensorMap<symtn::sector::FermionParityFusionRule, f64>,
    TensorMap<symtn::sector::FermionParityFusionRule, f64>,
) {
    let leg = fermionic_leg_with(&[1, 2]);
    let leg_dual = leg.try_dual().unwrap();
    let typed_a =
        TensorMap::from_subblock_fn(runtime, [&leg], [&leg_dual, &leg], fermionic_typed_fill)
            .unwrap();
    let typed_b =
        TensorMap::from_subblock_fn(runtime, [&leg_dual, &leg], [&leg], fermionic_typed_fill)
            .unwrap();
    (typed_a, typed_b)
}

/// [`fermionic_leg`] with explicit `[even, odd]` degeneracies.
fn fermionic_leg_with(
    degeneracies: &[usize; 2],
) -> GradedSpace<symtn::sector::FermionParityFusionRule> {
    GradedSpace::try_new(
        Arc::new(symtn::sector::FermionParityFusionRule),
        [
            (symtn::sector::Z2Irrep::EVEN, degeneracies[0]),
            (symtn::sector::Z2Irrep::ODD, degeneracies[1]),
        ],
    )
    .expect("fermionic leg is well formed")
}

#[test]
fn fermionic_compose_is_contract_against_a_twisted_right_operand() {
    // What: the exact relation between the two contraction semantics —
    // `compose(a, b) == contract(a, twist(b, b's dual codomain legs))`. The
    // twisted operand is built through `from_subblock_fn` rather than through the
    // public `twist`: theta is -1 on an odd sector and +1 otherwise, so the
    // twisted tensor is a one-line independent oracle here.
    let _guard = cache_lock();
    let runtime = runtime();
    let (typed_a, typed_b) = fermionic_compose_pair(&runtime);

    let leg = fermionic_leg_with(&[1, 2]);
    let leg_dual = leg.try_dual().unwrap();
    let twisted_b =
        TensorMap::from_subblock_fn(&runtime, [&leg_dual, &leg], [&leg], |sectors, indices| {
            // Codomain leg 0 is the dual one; leg 1 is contracted too but is
            // not dual, so theta does not act on it.
            let theta = if sectors.codomain_uncoupled()[0] == symtn::sector::Z2Irrep::ODD {
                -1.0
            } else {
                1.0
            };
            theta * fermionic_typed_fill(sectors, indices)
        })
        .unwrap();

    assert_eq!(
        typed_a.compose(&typed_b).unwrap().dense_data().unwrap(),
        typed_a
            .contract(
                &twisted_b,
                &ContractSpec {
                    lhs: &[1, 2],
                    rhs: &[0, 1],
                    codomain: &[0],
                    domain: &[1]
                }
            )
            .unwrap()
            .dense_data()
            .unwrap()
    );
}

#[test]
fn fermionic_compose_and_contract_disagree() {
    // What: the sign is real. If `compose` were routed through the ordinary
    // contraction the two would coincide and this assertion would fail.
    let _guard = cache_lock();
    let runtime = runtime();
    let (typed_a, typed_b) = fermionic_compose_pair(&runtime);

    assert_ne!(
        typed_a.compose(&typed_b).unwrap().dense_data().unwrap(),
        typed_a
            .contract(
                &typed_b,
                &ContractSpec {
                    lhs: &[1, 2],
                    rhs: &[0, 1],
                    codomain: &[0],
                    domain: &[1]
                }
            )
            .unwrap()
            .dense_data()
            .unwrap()
    );
}

#[test]
fn bosonic_compose_is_contract_with_the_identity_output_order() {
    // What: for a symmetric braiding the supertrace twist is the identity, so
    // the two semantics coincide.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_endomorphism(&runtime);

    let composed = typed.compose(&typed).unwrap();

    assert_eq!(
        composed.dense_data().unwrap(),
        typed
            .contract(
                &typed,
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &[0],
                    domain: &[1]
                }
            )
            .unwrap()
            .dense_data()
            .unwrap()
    );
}

#[test]
fn compose_contracts_the_whole_domain_against_the_whole_codomain() {
    // What: the derived axes. `[v, v] <- [v]` composed with `[v] <- [v, v]`
    // contracts exactly the one shared leg and leaves `[v, v] <- [v, v]`, and
    // the same tensor comes back from the explicit contraction. Perturbing
    // either axis derivation changes the shape or the numbers here.
    let _guard = cache_lock();
    let runtime = runtime();
    let tall = z2_tensor_split(&runtime, 2);
    let wide = z2_tensor_split(&runtime, 1);

    let composed = tall.compose(&wide).unwrap();

    assert_eq!(composed.codomain().len(), 2);
    assert_eq!(composed.domain().len(), 2);
    assert_eq!(
        composed.dense_data().unwrap(),
        tall.contract(
            &wide,
            &ContractSpec {
                lhs: &[2],
                rhs: &[0],
                codomain: &[0, 1],
                domain: &[2, 3]
            }
        )
        .unwrap()
        .dense_data()
        .unwrap()
    );
}

#[test]
fn compose_rejects_operands_from_different_runtimes() {
    let _guard = cache_lock();
    let left = z2_endomorphism(&runtime());
    let right = z2_endomorphism(&runtime());

    assert!(matches!(
        left.compose(&right).unwrap_err(),
        symtn::typed::Error::RuntimeMismatch
    ));
}

#[test]
fn compose_rejects_operands_whose_domain_and_codomain_do_not_meet() {
    // A rank mismatch and a matching-rank but non-dual leg mismatch, both from
    // the expert layer rather than from a pre-check here.
    let _guard = cache_lock();
    let runtime = runtime();
    let endo = z2_endomorphism(&runtime);
    let tall = z2_tensor_split(&runtime, 2);
    let wide = z2_tensor_split(&runtime, 1);

    // Rank mismatch in both directions: one domain leg against two codomain
    // legs, and two domain legs against one codomain leg.
    assert!(endo.compose(&tall).is_err());
    assert!(wide.compose(&endo).is_err());

    // Matching ranks, mismatched legs: the degeneracies differ, so the two do
    // not meet even though the shapes line up.
    let z2 = Arc::new(symtn::sector::Z2FusionRule);
    let narrow_leg = GradedSpace::try_new(
        Arc::clone(&z2),
        [
            (symtn::sector::Z2Irrep::EVEN, 1),
            (symtn::sector::Z2Irrep::ODD, 1),
        ],
    )
    .unwrap();
    let narrow =
        TensorMap::from_subblock_fn(&runtime, [&narrow_leg], [&narrow_leg], typed_fill_value)
            .unwrap();
    assert!(endo
        .compose(&narrow)
        .unwrap_err()
        .report()
        .contains("leg degeneracy mismatch"));

    // Matching ranks and matching degeneracies, opposite dual flags.
    let wide_leg = GradedSpace::try_new(
        Arc::clone(&z2),
        [
            (symtn::sector::Z2Irrep::EVEN, 2),
            (symtn::sector::Z2Irrep::ODD, 3),
        ],
    )
    .and_then(|space| space.try_dual())
    .unwrap();
    let dual_endo =
        TensorMap::from_subblock_fn(&runtime, [&wide_leg], [&wide_leg], typed_fill_value).unwrap();
    // The error names both operands' axes and external-axis flags: `endo`'s
    // domain axis 1 is `V'` and `dual_endo`'s codomain axis 0 is `V'` too.
    assert!(endo.compose(&dual_endo).unwrap_err().report().ends_with(
        "contracted fusion leg duality flags do not match: lhs axis 1 (is_dual = true) \
         and rhs axis 0 (is_dual = true) must have opposite duality flags"
    ));
    // `contract` reports the axes the caller named.
    let tall = z2_tensor_split(&runtime, 2);
    assert!(tall
        .contract(&dual_endo, &ContractSpec { lhs: &[2], rhs: &[0], codomain: &[0, 1], domain: &[2] })
        .unwrap_err()
        .report()
        .ends_with("lhs axis 2 (is_dual = true) and rhs axis 0 (is_dual = true) must have opposite duality flags"));

    // Matching ranks and flags, different sector content.
    let even_only =
        GradedSpace::try_new(Arc::clone(&z2), [(symtn::sector::Z2Irrep::EVEN, 2)]).unwrap();
    let even_endo =
        TensorMap::from_subblock_fn(&runtime, [&even_only], [&even_only], typed_fill_value)
            .unwrap();
    assert!(endo
        .compose(&even_endo)
        .unwrap_err()
        .report()
        .contains("dimension mismatch"));
}

// ---------------------------------------------------------------------------
// The identity `TensorMap::isomorphism(V, V)`.
// ---------------------------------------------------------------------------

#[test]
fn id_writes_the_nonuniform_fused_diagonal() {
    // What: the identity on a leg list whose per-sector degeneracies differ
    // from leg to leg — the case where the
    // diagonal offsets inside a coupled-sector block are not all the same.
    let _guard = cache_lock();
    let runtime = runtime();
    let z2 = Arc::new(symtn::sector::Z2FusionRule);
    let typed_leg = |even, odd| {
        GradedSpace::try_new(
            Arc::clone(&z2),
            [
                (symtn::sector::Z2Irrep::EVEN, even),
                (symtn::sector::Z2Irrep::ODD, odd),
            ],
        )
        .unwrap()
    };

    let wide = typed_leg(2, 3);
    let narrow = typed_leg(1, 4);
    let typed =
        TensorMap::<_, f64>::isomorphism(&runtime, [&wide, &narrow], [&wide, &narrow]).unwrap();

    // Not the zero tensor, and not the all-ones one either: a genuine diagonal.
    // 14 even + 11 odd fused states: `2*1 + 3*4` and `2*4 + 3*1`.
    assert_eq!(
        typed
            .dense_data()
            .unwrap()
            .iter()
            .filter(|&&v| v == 1.0)
            .count(),
        25
    );
    assert!(typed.dense_data().unwrap().contains(&0.0));
}

#[test]
fn id_composes_as_the_identity_on_both_sides() {
    // What: the defining property, as a byte oracle against the source tensor.
    // `[v, v] <- [v]`, so the two sides exercise different ranks.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_tensor(&runtime);

    let left = TensorMap::isomorphism(&runtime, &typed.codomain(), &typed.codomain()).unwrap();
    let right = TensorMap::isomorphism(&runtime, &typed.domain(), &typed.domain()).unwrap();

    assert_eq!(
        left.compose(&typed).unwrap().dense_data().unwrap(),
        typed.dense_data().unwrap()
    );
    assert_eq!(
        typed.compose(&right).unwrap().dense_data().unwrap(),
        typed.dense_data().unwrap()
    );
}

#[test]
fn id_composes_as_the_identity_for_a_fermionic_provider() {
    // What: no stray sign. A fermionic identity is still the plain diagonal —
    // the twist question belongs to the contracted legs, and composition does
    // not ask it.
    let _guard = cache_lock();
    let runtime = runtime();
    let (typed_a, _) = fermionic_compose_pair(&runtime);

    let left = TensorMap::isomorphism(&runtime, &typed_a.codomain(), &typed_a.codomain()).unwrap();
    let right = TensorMap::isomorphism(&runtime, &typed_a.domain(), &typed_a.domain()).unwrap();

    assert_eq!(
        left.compose(&typed_a).unwrap().dense_data().unwrap(),
        typed_a.dense_data().unwrap()
    );
    assert_eq!(
        typed_a.compose(&right).unwrap().dense_data().unwrap(),
        typed_a.dense_data().unwrap()
    );
}

#[test]
fn id_needs_at_least_one_leg() {
    // The provider is inferred from the legs, exactly as for `zeros`.
    let _guard = cache_lock();
    let runtime = runtime();
    let legs: [&GradedSpace<symtn::sector::Z2FusionRule>; 0] = [];

    assert!(matches!(
        TensorMap::<_, f64>::isomorphism(&runtime, legs, legs).unwrap_err(),
        symtn::typed::Error::InvalidArgument(message)
            if message.contains("at least one leg")
    ));
}

// ---------------------------------------------------------------------------
// Compact diagonal storage and the operations
// that consume it.
//
// Storage claims are measured in `tests/typed_diagonal_allocations.rs`; the
// value gates here compare compact routes with independent dense or pointwise
// oracles.
// ---------------------------------------------------------------------------

#[test]
fn compact_reductions_match_the_forced_dense_route() {
    // What: `norm`, `norm(Inf)`, `tr` and `inner` read the stored spectrum
    // instead of its materialization, and land on the same numbers.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = z2_bond(&runtime);
    let dense = forced_dense(&typed);

    // Two reductions of the same singular values that may sum in different
    // orders: agreement within the tolerance rule over the dense payload is
    // the contract. A maximum is order-independent, so `norm(Inf)` is exact.
    let terms = dense.dense_data().unwrap().len();
    numerics::assert_close(
        "norm",
        typed.norm(2.0).unwrap(),
        dense.norm(2.0).unwrap(),
        terms,
    );
    assert_eq!(
        typed.norm(f64::INFINITY).unwrap(),
        dense.norm(f64::INFINITY).unwrap()
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
}
