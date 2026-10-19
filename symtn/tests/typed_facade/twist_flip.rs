use super::*;

/// Position-weighted value from typed fZ2 labels.
fn fz2_typed_fill(
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
    value
        + indices
            .iter()
            .enumerate()
            .map(|(a, &i)| (a + 1) * i)
            .sum::<usize>() as f64
}

// ---------------------------------------------------------------------------
// Typed twist / flip / unit-leg insert & remove.
// ---------------------------------------------------------------------------

/// One typed fZ2 operand: rank `2 <- 1` with a dual codomain leg. The codomain carries a non-dual
/// and a dual leg and the domain a non-dual one, so twist/flip gates can pick
/// each (side, duality) combination off one fixture.
fn fz2_index(runtime: &Runtime) -> TensorMap<symtn::sector::FermionParityFusionRule, f64> {
    let provider = Arc::new(symtn::sector::FermionParityFusionRule);
    let leg = |pairs: &[(u8, usize)]| {
        GradedSpace::try_new(
            Arc::clone(&provider),
            pairs.iter().map(|&(parity, degeneracy)| {
                (
                    if parity == 0 {
                        symtn::sector::Z2Irrep::EVEN
                    } else {
                        symtn::sector::Z2Irrep::ODD
                    },
                    degeneracy,
                )
            }),
        )
        .unwrap()
    };
    let l1 = leg(&[(0, 1), (1, 2)]);
    let l2 = leg(&[(0, 2), (1, 1)]).try_dual().unwrap();
    let lv = leg(&[(0, 1), (1, 1)]);
    TensorMap::from_subblock_fn(runtime, [&l1, &l2], [&lv], fz2_typed_fill).unwrap()
}

/// The typed flip-doctest fixture: fZ2 `V <- V`, even block 2.0, odd block 3.0.
fn typed_fz2_doctest(runtime: &Runtime) -> TensorMap<symtn::sector::FermionParityFusionRule, f64> {
    let provider = Arc::new(symtn::sector::FermionParityFusionRule);
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [
            (symtn::sector::Z2Irrep::EVEN, 1),
            (symtn::sector::Z2Irrep::ODD, 1),
        ],
    )
    .unwrap();
    TensorMap::from_subblock_fn(runtime, [&leg], [&leg], |sectors, _| {
        if sectors.coupled() == &symtn::sector::Z2Irrep::EVEN {
            2.0
        } else {
            3.0
        }
    })
    .unwrap()
}

fn typed_fz2_two_block(
    runtime: &Runtime,
    dual: bool,
) -> TensorMap<symtn::sector::FermionParityFusionRule, f64> {
    let provider = Arc::new(symtn::sector::FermionParityFusionRule);
    let leg = GradedSpace::try_new(
        provider,
        [
            (symtn::sector::Z2Irrep::EVEN, 1),
            (symtn::sector::Z2Irrep::ODD, 1),
        ],
    )
    .unwrap();
    let leg = if dual { leg.try_dual().unwrap() } else { leg };
    TensorMap::from_subblock_fn(runtime, [&leg], [&leg], |sectors, _| {
        if sectors.coupled() == &symtn::sector::Z2Irrep::EVEN {
            2.0
        } else {
            3.0
        }
    })
    .unwrap()
}

macro_rules! assert_same_typed_block_structure {
    ($got:expr, $source:expr) => {{
        let (got, source) = ($got, $source);
        assert!(std::ptr::eq(got.provider(), source.provider()));
        assert_eq!(got.subblock_count(), source.subblock_count());
        for index in 0..source.subblock_count() {
            let (after, before) = (
                got.subblock(index).unwrap(),
                source.subblock(index).unwrap(),
            );
            assert_eq!(
                (after.offset(), after.shape(), after.strides()),
                (before.offset(), before.shape(), before.strides())
            );
            let (after, before) = (
                got.subblock_fusion_trees(index).unwrap(),
                source.subblock_fusion_trees(index).unwrap(),
            );
            assert_eq!(after.coupled(), before.coupled());
            assert_eq!(after.codomain_uncoupled(), before.codomain_uncoupled());
            assert_eq!(after.codomain_innerlines(), before.codomain_innerlines());
            assert_eq!(after.codomain_vertices(), before.codomain_vertices());
            assert_eq!(after.domain_uncoupled(), before.domain_uncoupled());
            assert_eq!(after.domain_innerlines(), before.domain_innerlines());
            assert_eq!(after.domain_vertices(), before.domain_vertices());
        }
    }};
}

#[test]
fn typed_inverse_index_ops_pin_values_and_preserve_structure() {
    let _guard = cache_lock();
    let runtime = runtime();
    let simple = typed_fz2_two_block(&runtime, false);
    let dual = typed_fz2_two_block(&runtime, true);
    assert_eq!(
        simple
            .flip(&[0], Direction::Inverse)
            .unwrap()
            .dense_data()
            .unwrap(),
        &[2.0, -3.0]
    );
    assert_eq!(
        dual.flip(&[0], Direction::Inverse)
            .unwrap()
            .dense_data()
            .unwrap(),
        &[2.0, 3.0]
    );
    assert_eq!(
        simple
            .flip(&[1], Direction::Inverse)
            .unwrap()
            .dense_data()
            .unwrap(),
        &[2.0, 3.0]
    );
    assert_eq!(
        dual.flip(&[1], Direction::Inverse)
            .unwrap()
            .dense_data()
            .unwrap(),
        &[2.0, -3.0]
    );
    assert_eq!(
        simple
            .twist(&[0], Direction::Inverse)
            .unwrap()
            .dense_data()
            .unwrap(),
        &[2.0, -3.0]
    );
    assert_ne!(
        simple
            .flip(&[1], Direction::Forward)
            .unwrap()
            .flip(&[1], Direction::Forward)
            .unwrap()
            .dense_data()
            .unwrap(),
        simple.dense_data().unwrap()
    );
    for restored in [
        simple
            .flip(&[1], Direction::Forward)
            .unwrap()
            .flip(&[1], Direction::Inverse)
            .unwrap(),
        simple
            .flip(&[1], Direction::Inverse)
            .unwrap()
            .flip(&[1], Direction::Forward)
            .unwrap(),
    ] {
        assert_eq!(restored.dense_data().unwrap(), simple.dense_data().unwrap());
        assert_same_legs(&restored.codomain(), &simple.codomain());
        assert_same_legs(&restored.domain(), &simple.domain());
    }
    assert_eq!(
        simple
            .flip(&[1, 1], Direction::Inverse)
            .unwrap()
            .dense_data()
            .unwrap(),
        simple
            .flip(&[1], Direction::Inverse)
            .unwrap()
            .flip(&[1], Direction::Inverse)
            .unwrap()
            .dense_data()
            .unwrap()
    );

    let complex = simple.convert::<Complex64>();
    assert_eq!(
        complex
            .flip(&[0], Direction::Inverse)
            .unwrap()
            .dense_data()
            .unwrap(),
        &[2.0.into(), (-3.0).into()]
    );
    assert_eq!(
        complex
            .twist(&[0], Direction::Inverse)
            .unwrap()
            .dense_data()
            .unwrap(),
        &[2.0.into(), (-3.0).into()]
    );

    let su2_provider = Arc::new(SU2FusionRule);
    let spin_half =
        GradedSpace::try_new(su2_provider, [(SU2Irrep::from_twice_spin(1), 1)]).unwrap();
    let spin_half_dual = spin_half.try_dual().unwrap();
    let su2: TensorMap<SU2FusionRule, f64> =
        TensorMap::from_subblock_fn(&runtime, [&spin_half_dual], [&spin_half], |_, _| 5.0).unwrap();
    assert_eq!(
        su2.flip(&[0], Direction::Inverse)
            .unwrap()
            .dense_data()
            .unwrap(),
        &[5.0]
    );
    assert_eq!(
        su2.flip(&[1], Direction::Inverse)
            .unwrap()
            .dense_data()
            .unwrap(),
        &[-5.0]
    );

    let structured = fz2_index(&runtime);
    let twisted = structured.twist(&[0, 1, 2], Direction::Inverse).unwrap();
    assert!(std::ptr::eq(twisted.provider(), structured.provider()));
    assert_same_legs(&twisted.codomain(), &structured.codomain());
    assert_same_legs(&twisted.domain(), &structured.domain());
    let codomain_flip = structured.flip(&[0], Direction::Inverse).unwrap();
    assert_same_typed_block_structure!(&codomain_flip, &structured);
    assert_eq!(
        codomain_flip.codomain()[0].is_dual(),
        !structured.codomain()[0].is_dual()
    );
    assert_eq!(
        codomain_flip.codomain()[1].is_dual(),
        structured.codomain()[1].is_dual()
    );
    assert_same_legs(&codomain_flip.domain(), &structured.domain());
    let domain_flip = structured.flip(&[2], Direction::Inverse).unwrap();
    assert_same_typed_block_structure!(&domain_flip, &structured);
    assert_same_legs(&domain_flip.codomain(), &structured.codomain());
    assert_eq!(
        domain_flip.domain()[0].is_dual(),
        !structured.domain()[0].is_dual()
    );

    let u1_provider = Arc::new(symtn::sector::U1FusionRule);
    let u1_leg = GradedSpace::try_new(
        u1_provider,
        [
            (symtn::sector::U1Irrep::new(-1), 1),
            (symtn::sector::U1Irrep::new(0), 1),
            (symtn::sector::U1Irrep::new(1), 1),
        ],
    )
    .unwrap();
    let u1: TensorMap<symtn::sector::U1FusionRule, f64> =
        TensorMap::from_subblock_fn(&runtime, [&u1_leg], [&u1_leg], |_, _| 7.0).unwrap();
    let u1_twist = u1.twist(&[0, 1], Direction::Inverse).unwrap();
    assert_eq!(
        u1_twist.dense_data().unwrap().as_ptr(),
        u1.dense_data().unwrap().as_ptr()
    );
    let u1_flip = u1.flip(&[0], Direction::Inverse).unwrap();
    assert_eq!(u1_flip.dense_data().unwrap(), u1.dense_data().unwrap());
    assert_same_typed_block_structure!(&u1_flip, &u1);
    assert_eq!(u1_flip.codomain()[0].is_dual(), !u1.codomain()[0].is_dual());
    assert_same_legs(&u1_flip.domain(), &u1.domain());
}

#[test]
fn inverse_index_ops_cover_the_fermionic_simple_product() {
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = fz2_u1_su2_oracle(&runtime, 1.0);
    for legs in [&[0usize][..], &[1][..], &[0, 1][..]] {
        assert_eq!(
            typed
                .twist(legs, Direction::Forward)
                .unwrap()
                .twist(legs, Direction::Inverse)
                .unwrap()
                .dense_data()
                .unwrap(),
            typed.dense_data().unwrap()
        );
        assert_eq!(
            typed
                .flip(legs, Direction::Forward)
                .unwrap()
                .flip(legs, Direction::Inverse)
                .unwrap()
                .dense_data()
                .unwrap(),
            typed.dense_data().unwrap()
        );
    }
}

#[test]
fn typed_multi_leg_dense_twist_is_the_per_leg_product_by_value() {
    // What: the multi-leg dense twist coefficient is pinned by a hand-computed
    // value. A mutation that drops the per-leg product in the shared
    // `twist_block_factor` (e.g. keeping only the first leg's θ) survives
    // ordinary cross-operation parity gates, which share the same helper.
    // Fixture: fZ2 `V <- V`, even block 2.0, odd block 3.0, θ(odd) = −1;
    // both legs carry the block's coupled sector.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = typed_fz2_doctest(&runtime);
    // One leg: θ bites, the odd block negates (sanity that the factor is
    // live at all on this fixture).
    let one: TensorMap<symtn::sector::FermionParityFusionRule, f64> =
        typed.twist(&[0], Direction::Forward).unwrap();
    assert_eq!(one.dense_data().unwrap(), &[2.0, -3.0]);
    let domain_leg: TensorMap<symtn::sector::FermionParityFusionRule, f64> =
        typed.twist(&[1], Direction::Forward).unwrap();
    assert_eq!(domain_leg.dense_data().unwrap(), &[2.0, -3.0]);
    // Two *different* legs: the odd block scales by θ·θ = (−1)² = +1 — the
    // per-leg product, not a single factor.
    let both: TensorMap<symtn::sector::FermionParityFusionRule, f64> =
        typed.twist(&[0, 1], Direction::Forward).unwrap();
    assert_eq!(both.dense_data().unwrap(), &[2.0, 3.0]);
    // The same leg listed twice: identity by value, for the same θ² reason.
    let twice: TensorMap<symtn::sector::FermionParityFusionRule, f64> =
        typed.twist(&[1, 1], Direction::Forward).unwrap();
    assert_eq!(twice.dense_data().unwrap(), &[2.0, 3.0]);
}

#[test]
fn typed_flip_is_a_fourth_root_of_identity_and_flip_squared_scales_odd_blocks() {
    // What (gate 2): the TensorKit non-involution law on the typed facade —
    // flip² returns to the original spaces but scales the odd block by
    // θ = −1; only flip⁴ = id.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = typed_fz2_doctest(&runtime);
    let f1: TensorMap<symtn::sector::FermionParityFusionRule, f64> =
        typed.flip(&[1], Direction::Forward).unwrap();
    assert_eq!(f1.dense_data().unwrap(), &[2.0, -3.0]);
    assert_eq!(f1.domain()[0].is_dual(), !typed.domain()[0].is_dual());
    let f2: TensorMap<symtn::sector::FermionParityFusionRule, f64> =
        f1.flip(&[1], Direction::Forward).unwrap();
    assert_same_legs(&f2.codomain(), &typed.codomain());
    assert_same_legs(&f2.domain(), &typed.domain());
    assert_eq!(f2.dense_data().unwrap(), &[2.0, -3.0]);
    let f4: TensorMap<symtn::sector::FermionParityFusionRule, f64> = f2
        .flip(&[1], Direction::Forward)
        .unwrap()
        .flip(&[1], Direction::Forward)
        .unwrap();
    assert_eq!(f4.dense_data().unwrap(), typed.dense_data().unwrap());
}

#[test]
fn typed_flip_repeated_leg_in_one_call_is_sequential() {
    // What: the same leg listed twice in one call flips it twice sequentially;
    // the second occurrence sees the duality the first one left.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = typed_fz2_doctest(&runtime);
    let typed_twice: TensorMap<symtn::sector::FermionParityFusionRule, f64> =
        typed.flip(&[1, 1], Direction::Forward).unwrap();
    assert_eq!(typed_twice.dense_data().unwrap(), &[2.0, -3.0]);
    let typed_stepwise: TensorMap<symtn::sector::FermionParityFusionRule, f64> = typed
        .flip(&[1], Direction::Forward)
        .unwrap()
        .flip(&[1], Direction::Forward)
        .unwrap();
    assert_eq!(
        typed_twice.dense_data().unwrap(),
        typed_stepwise.dense_data().unwrap()
    );
    assert_same_legs(&typed_twice.domain(), &typed.domain());
}

#[test]
fn typed_insert_unit_round_trips_at_every_position_and_shares_the_payload() {
    // What (gate 4): `insert_unit` at every legal
    // position `0..=rank` followed by `remove_unit` at the inserted axis
    // restores the spaces *and* the payload allocation — `data()` returns the
    // same buffer address, the O(1) reuse the #613 contract promises for a
    // dense payload. (The `Arc`-level gate lives with the body layout tests
    // in `typed.rs`, which can see the private fields.)
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = fz2_index(&runtime);
    for position in 0..=typed.rank() {
        for left in [true, false] {
            for dual in [false, true] {
                let inserted: TensorMap<symtn::sector::FermionParityFusionRule, f64> = if left {
                    typed
                        .insert_unit(
                            position,
                            Side::Domain,
                            if dual { Duality::Dual } else { Duality::Plain },
                        )
                        .unwrap()
                } else {
                    typed
                        .insert_unit(
                            position,
                            Side::Codomain,
                            if dual { Duality::Dual } else { Duality::Plain },
                        )
                        .unwrap()
                };
                assert_eq!(
                    inserted.dense_data().unwrap().as_ptr(),
                    typed.dense_data().unwrap().as_ptr(),
                    "left={left} position={position} dual={dual}"
                );
                let legs: Vec<_> = inserted
                    .codomain()
                    .into_iter()
                    .chain(inserted.domain())
                    .collect();
                assert_eq!(legs[position].is_dual(), dual);
                assert_eq!(legs[position].degeneracies(), &[1]);
                let removed: TensorMap<symtn::sector::FermionParityFusionRule, f64> =
                    inserted.remove_unit(position).unwrap();
                assert_eq!(
                    removed.dense_data().unwrap().as_ptr(),
                    typed.dense_data().unwrap().as_ptr()
                );
                assert_same_legs(&removed.codomain(), &typed.codomain());
                assert_same_legs(&removed.domain(), &typed.domain());
            }
        }
    }
}

#[test]
fn typed_index_op_error_classes_and_empty_shortcuts_are_stable() {
    // What: each rejected input returns the typed error class and operation-
    // specific message, while an empty leg list is a buffer-sharing clone.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = fz2_index(&runtime);

    // Twist / flip: out-of-range leg.
    for (error, message) in [
        (
            typed.twist(&[5], Direction::Forward).unwrap_err(),
            "invalid argument: twist leg 5 out of range for rank 3",
        ),
        (
            typed.flip(&[5], Direction::Forward).unwrap_err(),
            "invalid argument: flip leg 5 out of range for rank 3",
        ),
        (
            typed.twist(&[5], Direction::Inverse).unwrap_err(),
            "invalid argument: inverse twist leg 5 out of range for rank 3",
        ),
        (
            typed.flip(&[5], Direction::Inverse).unwrap_err(),
            "invalid argument: inverse flip leg 5 out of range for rank 3",
        ),
    ] {
        assert!(matches!(error, symtn::typed::Error::InvalidArgument(_)));
        assert_eq!(error.to_string(), message);
    }

    // Empty leg list: identical clone, shared buffer typed-side.
    let typed_untwisted: TensorMap<symtn::sector::FermionParityFusionRule, f64> =
        typed.twist(&[], Direction::Forward).unwrap();
    assert_eq!(
        typed_untwisted.dense_data().unwrap().as_ptr(),
        typed.dense_data().unwrap().as_ptr()
    );
    let typed_unflipped: TensorMap<symtn::sector::FermionParityFusionRule, f64> =
        typed.flip(&[], Direction::Forward).unwrap();
    assert_eq!(
        typed_unflipped.dense_data().unwrap().as_ptr(),
        typed.dense_data().unwrap().as_ptr()
    );
    let typed_untwisted_inverse: TensorMap<symtn::sector::FermionParityFusionRule, f64> =
        typed.twist(&[], Direction::Inverse).unwrap();
    assert_eq!(
        typed_untwisted_inverse.dense_data().unwrap().as_ptr(),
        typed.dense_data().unwrap().as_ptr()
    );
    let typed_unflipped_inverse: TensorMap<symtn::sector::FermionParityFusionRule, f64> =
        typed.flip(&[], Direction::Inverse).unwrap();
    assert_eq!(
        typed_unflipped_inverse.dense_data().unwrap().as_ptr(),
        typed.dense_data().unwrap().as_ptr()
    );

    // Insert: position past the rank.
    assert_eq!(
        typed
            .insert_unit(4, Side::Domain, Duality::Plain)
            .unwrap_err()
            .to_string(),
        "invalid argument: TensorMap::insert_unit: position 4 exceeds rank 3"
    );
    assert_eq!(
        typed
            .insert_unit(4, Side::Codomain, Duality::Plain)
            .unwrap_err()
            .to_string(),
        "invalid argument: TensorMap::insert_unit: position 4 exceeds rank 3"
    );

    // Remove: out-of-range axis, then a non-unit leg.
    assert_eq!(
        typed.remove_unit(3).unwrap_err().to_string(),
        "invalid argument: TensorMap::remove_unit: axis 3 is out of range for rank 3"
    );
    let error = typed.remove_unit(0).unwrap_err();
    assert!(
        matches!(error, symtn::typed::Error::InvalidArgument(_)),
        "{error:?}"
    );
    assert_eq!(
        error.to_string(),
        "invalid argument: TensorMap::remove_unit: axis 0 is not a canonical unit leg"
    );
}

#[test]
fn typed_twist_on_a_compact_spectrum_matches_the_dense_route() {
    // What: the compact twist arm matches the ordinary dense tree transform
    // on an SVD spectrum whose coupled sectors include the odd one.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed = fz2_index(&runtime);
    let typed_s: TensorMap<symtn::sector::FermionParityFusionRule, f64> =
        typed.svd_compact(&[0, 1], &[2]).unwrap().s;
    let dense = forced_dense(&typed_s);
    let typed_twisted: TensorMap<symtn::sector::FermionParityFusionRule, f64> =
        typed_s.twist(&[0], Direction::Forward).unwrap();
    assert_eq!(
        typed_twisted.materialize().unwrap().dense_data().unwrap(),
        dense
            .twist(&[0], Direction::Forward)
            .unwrap()
            .dense_data()
            .unwrap()
    );
    // And the two-leg twist is the identity on the bond (θ² = 1 per sector).
    let typed_both: TensorMap<symtn::sector::FermionParityFusionRule, f64> =
        typed_s.twist(&[0, 1], Direction::Forward).unwrap();
    assert_eq!(
        typed_both.materialize().unwrap().dense_data().unwrap(),
        typed_s.materialize().unwrap().dense_data().unwrap()
    );
    let typed_inverse: TensorMap<symtn::sector::FermionParityFusionRule, f64> =
        typed_s.twist(&[0], Direction::Inverse).unwrap();
    assert_eq!(
        typed_inverse.materialize().unwrap().dense_data().unwrap(),
        typed_twisted.materialize().unwrap().dense_data().unwrap()
    );
}

#[test]
fn external_z3_twist_flip_and_units_hold_by_value() {
    // What (gate 7): typed-only checks on the external Z3 provider. Coverage
    // limit, on purpose: Z3 is bosonic (θ ≡ 1, χ ≡ 1), so `twist` exercises
    // only the identity short-circuit and `flip` only the structural toggle
    // with factor 1 — the θ/χ-bearing arms are covered by the built-in fZ2
    // parity gates above; the harness has no fermionic external provider.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalZ3::new());
    let w = z3_leg(&provider, false);
    let v = z3_leg(&provider, true);
    let t: TensorMap<ExternalZ3, f64> =
        TensorMap::from_subblock_fn(&runtime, [&w], [&v], |sectors, indices| {
            f64::from(sectors.coupled().0) * 10.0 + (indices[0] * 3 + indices[1]) as f64
        })
        .unwrap();

    // Twist: identity, shared buffer.
    let twisted: TensorMap<ExternalZ3, f64> = t.twist(&[0, 1], Direction::Forward).unwrap();
    assert_eq!(
        twisted.dense_data().unwrap().as_ptr(),
        t.dense_data().unwrap().as_ptr()
    );

    // Flip: values unchanged, duality flags toggled, non-self-dual sector
    // sets preserved as stored (flip toggles the flag, not the labels).
    let flipped: TensorMap<ExternalZ3, f64> = t.flip(&[0, 1], Direction::Forward).unwrap();
    assert_eq!(flipped.dense_data().unwrap(), t.dense_data().unwrap());
    assert!(flipped.codomain()[0].is_dual());
    assert!(!flipped.domain()[0].is_dual());
    assert_eq!(
        flipped.codomain()[0].sectors().unwrap(),
        t.codomain()[0].sectors().unwrap()
    );

    // Units: insert -> remove round trip on the external provider, O(1)
    // payload reuse observable through `data()`.
    let inserted: TensorMap<ExternalZ3, f64> =
        t.insert_unit(1, Side::Codomain, Duality::Dual).unwrap();
    assert_eq!(
        inserted.dense_data().unwrap().as_ptr(),
        t.dense_data().unwrap().as_ptr()
    );
    assert_eq!(inserted.codomain()[1].sectors().unwrap(), vec![Z3Charge(0)]);
    let removed: TensorMap<ExternalZ3, f64> = inserted.remove_unit(1).unwrap();
    assert_eq!(
        removed.dense_data().unwrap().as_ptr(),
        t.dense_data().unwrap().as_ptr()
    );
    assert_same_legs(&removed.codomain(), &t.codomain());
    assert_same_legs(&removed.domain(), &t.domain());
}

#[test]
fn external_nobraiding_twist_and_flip_reject_nontrivial_sectors() {
    // What: under `BraidingStyleKind::NoBraiding` the
    // twist eigenvalue is undefined, so twist/flip on a leg carrying any
    // non-unit sector must fail — TensorKit `has_shared_twist`
    // (`tensors/indexmanipulations.jl:34-41`) throws `SectorMismatch` there
    // — instead of silently applying θ ≡ 1. The compact spectrum arm must
    // hit the same preflight before its own dispatch.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(PlanarZ2);
    let mixed = GradedSpace::try_new(
        Arc::clone(&provider),
        [(PlanarParity(0), 1), (PlanarParity(1), 2)],
    )
    .unwrap();
    let t: TensorMap<PlanarZ2, f64> =
        TensorMap::from_subblock_fn(&runtime, [&mixed], [&mixed], |sectors, indices| {
            f64::from(sectors.coupled().0) * 10.0 + (indices[0] * 3 + indices[1]) as f64
        })
        .unwrap();

    for error in [
        t.twist(&[0], Direction::Forward).unwrap_err(),
        t.flip(&[1], Direction::Forward).unwrap_err(),
        t.twist(&[0], Direction::Inverse).unwrap_err(),
        t.flip(&[1], Direction::Inverse).unwrap_err(),
    ] {
        assert!(
            matches!(error, symtn::typed::Error::InvalidArgument(_)),
            "{error:?}"
        );
        assert!(error.to_string().contains("no braiding"), "{error}");
    }
    // The compact diagonal arm rejects too: an SVD spectrum factor lives on
    // the mixed bond space, so its twist must fail before the compact
    // per-sector scaling ever runs.
    let s: TensorMap<PlanarZ2, f64> = t.svd_compact(&[0], &[1]).unwrap().s;
    let compact_error = s.twist(&[0], Direction::Forward).unwrap_err();
    assert!(
        matches!(compact_error, symtn::typed::Error::InvalidArgument(_)),
        "{compact_error:?}"
    );
    assert!(matches!(
        s.twist(&[0], Direction::Inverse),
        Err(symtn::typed::Error::InvalidArgument(_))
    ));
}

#[test]
fn external_nobraiding_vacuum_only_legs_twist_passes_flip_rejects() {
    // What: the TK asymmetry on
    // vacuum-only legs under NoBraiding. `twist` carries an explicit
    // unit-sector carve-out (`has_shared_twist`,
    // `tensors/indexmanipulations.jl:34-41`) and is the identity (shared
    // buffer). `flip` has NO such exception — TK's fusion-tree flip
    // unconditionally evaluates `frobenius_schur_phase(a)` and `twist(a)`
    // (`fusiontrees/braiding_manipulations.jl:384-412`), neither of which a
    // NoBraiding sector defines, so flip fails in TK even on the vacuum and
    // must fail here. The boundary stays: `flip(&[])` is the empty-list
    // identical clone and never reaches the guard.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(PlanarZ2);
    let unit_only = GradedSpace::try_new(Arc::clone(&provider), [(PlanarParity(0), 2)]).unwrap();
    let t: TensorMap<PlanarZ2, f64> =
        TensorMap::from_subblock_fn(&runtime, [&unit_only], [&unit_only], |_, indices| {
            (indices[0] * 2 + indices[1]) as f64
        })
        .unwrap();

    let twisted: TensorMap<PlanarZ2, f64> = t.twist(&[0, 1], Direction::Forward).unwrap();
    assert_eq!(
        twisted.dense_data().unwrap().as_ptr(),
        t.dense_data().unwrap().as_ptr()
    );
    let twisted_inverse: TensorMap<PlanarZ2, f64> = t.twist(&[0, 1], Direction::Inverse).unwrap();
    assert_eq!(
        twisted_inverse.dense_data().unwrap().as_ptr(),
        t.dense_data().unwrap().as_ptr()
    );

    let flip_error = t.flip(&[0], Direction::Forward).unwrap_err();
    assert!(
        matches!(flip_error, symtn::typed::Error::InvalidArgument(_)),
        "{flip_error:?}"
    );
    assert!(
        flip_error.to_string().contains("no braiding"),
        "{flip_error}"
    );
    assert!(matches!(
        t.flip(&[0], Direction::Inverse),
        Err(symtn::typed::Error::InvalidArgument(_))
    ));

    let unflipped: TensorMap<PlanarZ2, f64> = t.flip(&[], Direction::Forward).unwrap();
    assert_eq!(
        unflipped.dense_data().unwrap().as_ptr(),
        t.dense_data().unwrap().as_ptr()
    );
    let unflipped_inverse: TensorMap<PlanarZ2, f64> = t.flip(&[], Direction::Inverse).unwrap();
    assert_eq!(
        unflipped_inverse.dense_data().unwrap().as_ptr(),
        t.dense_data().unwrap().as_ptr()
    );
}

#[test]
fn cu1_typed_rank_three_permutation_pins_the_gauge_contract_and_recoupling_values() {
    // What: CU(1) does not certify a trivial associator gauge, and this
    // rank-three charged fixture independently pins its recoupled payload.
    let _guard = cache_lock();
    let runtime = runtime();
    let rule = Arc::new(CU1FusionRule);
    assert!(!rule.has_trivial_associator_gauge());
    let q = CU1Irrep::from_twice_charge(1);
    let leg = GradedSpace::try_new(Arc::clone(&rule), [(q, 1)]).unwrap();
    let tensor: TensorMap<CU1FusionRule, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg, &leg], [&leg], |_, _| 1.0).unwrap();
    assert_eq!(tensor.codomain().len(), 3);
    assert_eq!(tensor.domain().len(), 1);
    assert!(tensor
        .codomain()
        .iter()
        .chain(tensor.domain().iter())
        .all(|space| space.degeneracies() == [1]));
    assert_eq!(tensor.dense_data().unwrap(), [1.0, 1.0, 1.0]);
    let permuted = tensor.permute(&[2, 0, 1], &[3]).unwrap();
    assert_eq!(permuted.codomain().len(), 3);
    assert_eq!(permuted.domain().len(), 1);
    assert_eq!(permuted.dense_data().unwrap().len(), 3);
    for (got, expected) in permuted.dense_data().unwrap().iter().zip([
        2.0_f64.sqrt() / 2.0,
        -2.0_f64.sqrt() / 2.0,
        2.0_f64.sqrt(),
    ]) {
        assert!((got - expected).abs() <= 1e-12, "{got} vs {expected}");
    }
}

// ---------------------------------------------------------------------------
// Issue #580, group 6: ordered contraction through canonical `contract`.
// ---------------------------------------------------------------------------

#[test]
#[expect(
    clippy::type_complexity,
    reason = "the test table records axes and expected error class in one fixture row"
)]
fn contract_error_classes_and_their_both_defect_precedence() {
    // What: the typed contract reports
    // a bad output order before a simultaneous contracted-leg mismatch.
    let _guard = cache_lock();
    let second = runtime();
    let runtime = runtime();
    let typed_fixture = |runtime: &Runtime| {
        let leg = GradedSpace::try_new(
            Arc::new(symtn::sector::Z2FusionRule),
            [
                (symtn::sector::Z2Irrep::EVEN, 2),
                (symtn::sector::Z2Irrep::ODD, 3),
            ],
        )
        .unwrap();
        TensorMap::from_subblock_fn(runtime, [&leg, &leg], [&leg], typed_fill_value).unwrap()
    };
    let typed: TensorMap<symtn::sector::Z2FusionRule, f64> = typed_fixture(&runtime);

    let cases: &[(&str, &[usize], &[usize], &[usize], &str)] = &[
        (
            "len mismatch",
            &[2],
            &[0, 1],
            &[0, 1, 2, 3],
            "ContractAxisCountMismatch",
        ),
        (
            "lhs out of range",
            &[9],
            &[0],
            &[0, 1, 2, 3],
            "InvalidAxisSet",
        ),
        (
            "rhs out of range",
            &[2],
            &[9],
            &[0, 1, 2, 3],
            "InvalidAxisSet",
        ),
        (
            "output wrong length",
            &[2],
            &[0],
            &[0, 1, 2],
            "InvalidPermutation",
        ),
        (
            "output duplicate",
            &[2],
            &[0],
            &[0, 0, 1, 2],
            "InvalidPermutation",
        ),
        (
            "output out of range",
            &[2],
            &[0],
            &[0, 1, 2, 9],
            "InvalidPermutation",
        ),
    ];
    for &(name, lhs_axes, rhs_axes, output_axes, class) in cases {
        // `typed` is rank (2, 1) and every case contracts one of its legs, so
        // two open legs of it form the default codomain.
        let (codomain, domain) = output_axes.split_at(2.min(output_axes.len()));
        let spec = ContractSpec {
            lhs: lhs_axes,
            rhs: rhs_axes,
            codomain,
            domain,
        };
        let typed_error = typed.contract(&typed, &spec).unwrap_err();
        assert!(
            matches!(typed_error, symtn::typed::Error::Operation(_)),
            "{name}: {typed_error:?}"
        );
        assert!(
            format!("{typed_error:?}").contains(class),
            "{name}: {typed_error:?} does not carry {class}"
        );
    }

    // Mismatched contracted legs retain the expert layer's class.
    let narrow_leg = GradedSpace::try_new(
        Arc::new(symtn::sector::Z2FusionRule),
        [
            (symtn::sector::Z2Irrep::EVEN, 2),
            (symtn::sector::Z2Irrep::ODD, 2),
        ],
    )
    .unwrap();
    let typed_narrow: TensorMap<symtn::sector::Z2FusionRule, f64> = TensorMap::from_subblock_fn(
        &runtime,
        [&narrow_leg, &narrow_leg],
        [&narrow_leg],
        typed_fill_value,
    )
    .unwrap();
    let typed_error = typed
        .contract(
            &typed_narrow,
            &ContractSpec {
                lhs: &[2],
                rhs: &[0],
                codomain: &[0, 1],
                domain: &[2, 3],
            },
        )
        .unwrap_err();
    assert!(
        format!("{typed_error:?}").contains("LegDegeneracyMismatch"),
        "{typed_error:?}"
    );

    // Both defects at once: mismatched legs AND a non-permutation output order.
    let typed_both = typed
        .contract(
            &typed_narrow,
            &ContractSpec {
                lhs: &[2],
                rhs: &[0],
                codomain: &[0, 0],
                domain: &[1, 2],
            },
        )
        .unwrap_err();
    assert!(
        format!("{typed_both:?}").contains("InvalidPermutation"),
        "typed both-defect precedence moved: {typed_both:?}"
    );

    // Runtime mismatch stays a facade-level error.
    let typed_second: TensorMap<symtn::sector::Z2FusionRule, f64> = typed_fixture(&second);
    assert!(matches!(
        typed
            .contract(
                &typed_second,
                &ContractSpec {
                    lhs: &[2],
                    rhs: &[0],
                    codomain: &[0, 1],
                    domain: &[2, 3]
                }
            )
            .unwrap_err(),
        symtn::typed::Error::RuntimeMismatch
    ));

    // Rule-identity mismatch stays the expert layer's rejection through the
    // canonical name too.
    let first = Arc::new(ExternalZ3::tagged(0));
    let second_rule = Arc::new(ExternalZ3::tagged(1));
    let lhs = counting_z3(
        &runtime,
        &z3_dense_leg(&first, 2),
        &z3_dense_leg(&first, 3),
        1.0,
    );
    let rhs = counting_z3(
        &runtime,
        &z3_dense_leg(&second_rule, 3),
        &z3_dense_leg(&second_rule, 4),
        1.0,
    );
    let identity_error = lhs
        .contract(
            &rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap_err();
    assert!(
        format!("{identity_error:?}").contains("FusionRuleMismatch"),
        "{identity_error:?}"
    );
}
