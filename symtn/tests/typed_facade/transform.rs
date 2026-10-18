use super::*;

#[test]
fn permute_moves_legs_of_a_multi_block_external_provider_tensor() {
    // What: a non-identity permute of a runtime-rank multi-block external
    // provider tensor produces the reordered spaces, keeps the element count,
    // and moves the payload (this rule's F/R symbols are all 1, so the permuted
    // buffer is a rearrangement of the source, never a rescaling).
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalZ3::new());
    let tensor = z3_rank_four(&runtime, &provider);

    // The source is `[wide, narrow] <- [wide', narrow']`; this order sends one
    // leg of each shape across the codomain/domain split, so the degeneracies
    // pin which source leg landed where. Flags alone would not: they are
    // symmetric under swapping the split.
    let permuted = tensor.permute(&[1, 2], &[3, 0]).unwrap();

    assert_eq!(permuted.codomain().len(), 2);
    assert_eq!(permuted.domain().len(), 2);
    // Axis 1 (`narrow`) stays in the codomain unchanged; axis 2 (`wide'`) is
    // bent round from the domain, which conjugates it, so it arrives as the
    // non-dual `wide`. Symmetrically axis 0 (`wide`) bent into the domain
    // arrives as `wide'`, while axis 3 (`narrow'`) is carried across as is.
    assert_eq!(permuted.codomain()[0].degeneracies(), &[1, 2, 4]);
    assert_eq!(permuted.codomain()[1].degeneracies(), &[2, 3, 1]);
    assert_eq!(permuted.domain()[0].degeneracies(), &[1, 4, 2]);
    assert_eq!(permuted.domain()[1].degeneracies(), &[2, 1, 3]);
    assert!(!permuted.codomain()[0].is_dual());
    assert!(!permuted.codomain()[1].is_dual());
    assert!(permuted.domain()[0].is_dual());
    assert!(permuted.domain()[1].is_dual());
    assert_eq!(
        permuted.dense_data().unwrap().len(),
        tensor.dense_data().unwrap().len()
    );
    assert_ne!(permuted.dense_data().unwrap(), tensor.dense_data().unwrap());
    let mut moved: Vec<f64> = permuted.dense_data().unwrap().to_vec();
    let mut original: Vec<f64> = tensor.dense_data().unwrap().to_vec();
    moved.sort_by(f64::total_cmp);
    original.sort_by(f64::total_cmp);
    assert_eq!(moved, original);
}

#[test]
fn permute_round_trips_back_to_the_source_layout() {
    // What: permuting and permuting back is the identity on both the spaces and
    // the payload — a stronger statement than "the multiset survived", since it
    // pins where each element landed.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalZ3::new());
    let tensor = z3_rank_four(&runtime, &provider);

    let there = tensor.permute(&[1, 3], &[0, 2]).unwrap();
    let back = there.permute(&[2, 0], &[3, 1]).unwrap();

    assert_eq!(back.dense_data().unwrap(), tensor.dense_data().unwrap());
    assert_eq!(back.subblock_count(), tensor.subblock_count());
    for index in 0..tensor.subblock_count() {
        assert_eq!(
            back.subblock_fusion_trees(index).unwrap(),
            tensor.subblock_fusion_trees(index).unwrap()
        );
    }
}

#[test]
fn permute_carries_a_simple_fusion_provider_with_a_complex_payload() {
    // What: nothing in the typed transform is abelian- or real-specific; the
    // SU(2) recoupling coefficients reach a `Complex64` payload.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalSu2);
    let leg = su2_leg(&provider, false);
    let tensor: TensorMap<ExternalSu2, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], |sectors, indices| {
            Complex64::new(
                sectors.coupled().twice_spin() as f64 + 1.0,
                indices.iter().sum::<usize>() as f64 + 1.0,
            )
        })
        .unwrap();

    let permuted = tensor.permute(&[0, 2], &[1, 3]).unwrap();

    assert_eq!(permuted.codomain().len(), 2);
    assert_eq!(permuted.domain().len(), 2);
    assert!(permuted.subblock_count() >= 1);
    assert!(permuted
        .dense_data()
        .unwrap()
        .iter()
        .any(|value| *value != Complex64::new(0.0, 0.0)));
}

#[test]
fn permute_rejects_malformed_axes_without_panicking() {
    // What: the expert layer's typed errors are the contract — an out-of-range
    // axis, a repeated axis and a wrong-length axis list all come back as `Err`.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalZ3::new());
    let tensor = z3_rank_four(&runtime, &provider);

    assert!(tensor.permute(&[0, 9], &[2, 3]).is_err());
    assert!(tensor.permute(&[0, 0], &[2, 3]).is_err());
    assert!(tensor.permute(&[0], &[2, 3]).is_err());
}

#[test]
fn braid_rejects_a_wrong_length_levels_list() {
    // What: the public pre-check — `levels` must name every source axis — in
    // both directions.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalZ3::new());
    let tensor = z3_rank_four(&runtime, &provider);

    let short = tensor.braid(&[1, 2], &[3, 0], &[0, 1, 2]).unwrap_err();
    let message = short.to_string();
    assert!(message.contains("one level per source axis"), "{message}");
    assert!(message.contains("expected 4"), "{message}");
    assert!(tensor.braid(&[1, 2], &[3, 0], &[0; 5]).is_err());
    // Malformed axes still come back from the expert layer, not from here.
    assert!(tensor.braid(&[0, 0], &[2, 3], &[0, 1, 2, 3]).is_err());
}

#[test]
fn transpose_twice_returns_the_source_layout() {
    // What: the planar transpose is an involution — it rotates every leg once
    // round the boundary, so applying it twice restores the source spaces,
    // block order and bytes exactly.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalZ3::new());
    let tensor = z3_rank_four(&runtime, &provider);

    let once = full_transpose!(tensor).unwrap();
    let twice = full_transpose!(once).unwrap();

    assert_ne!(once.dense_data().unwrap(), tensor.dense_data().unwrap());
    assert_eq!(typed_leg_shapes(&twice), typed_leg_shapes(&tensor));
    assert_eq!(twice.dense_data().unwrap(), tensor.dense_data().unwrap());
    assert_eq!(twice.subblock_count(), tensor.subblock_count());
    for index in 0..tensor.subblock_count() {
        assert_eq!(
            twice.subblock_fusion_trees(index).unwrap(),
            tensor.subblock_fusion_trees(index).unwrap()
        );
    }
}

#[test]
fn transpose_rejects_malformed_axes_without_panicking() {
    // What: out-of-range axes, a wrong-length list and a non-planar
    // re-arrangement (a permute, which `transpose` must refuse rather than
    // silently braid) all come back as `Err`.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalZ3::new());
    let tensor = z3_rank_four(&runtime, &provider);

    assert!(tensor.transpose(&[0, 9], &[2, 3]).is_err());
    assert!(tensor.transpose(&[0], &[2, 3]).is_err());
    assert!(tensor.transpose(&[1, 2], &[3, 0]).is_err());
}

// ---------------------------------------------------------------------------
// `TensorMap::repartition`.
// ---------------------------------------------------------------------------

#[test]
fn repartition_moves_the_boundary_and_round_trips_at_every_split() {
    // What: every split point of a rank-4 tensor map is reachable, reports the
    // requested codomain/domain sizes, and comes back to the source layout —
    // spaces, block identities and bytes — when repartitioned to the original
    // split.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalZ3::new());
    let tensor = z3_rank_four(&runtime, &provider);

    for num_codomain in 0..=4 {
        let moved = tensor.repartition(num_codomain).unwrap();
        assert_eq!(moved.codomain().len(), num_codomain);
        assert_eq!(moved.domain().len(), 4 - num_codomain);
        assert_eq!(
            moved.dense_data().unwrap().len(),
            tensor.dense_data().unwrap().len()
        );

        let back = moved.repartition(2).unwrap();
        assert_eq!(typed_leg_shapes(&back), typed_leg_shapes(&tensor));
        assert_eq!(back.dense_data().unwrap(), tensor.dense_data().unwrap());
        for index in 0..tensor.subblock_count() {
            assert_eq!(
                back.subblock_fusion_trees(index).unwrap(),
                tensor.subblock_fusion_trees(index).unwrap()
            );
        }
    }
}

#[test]
fn repartition_rejects_a_split_beyond_the_rank() {
    // What: `num_codomain > rank` has no planar reading at all, so it is
    // rejected rather than clamped.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalZ3::new());
    let tensor = z3_rank_four(&runtime, &provider);

    let error = tensor.repartition(5).unwrap_err();

    assert!(error.to_string().contains("exceeds rank 4"), "{error}");
}

#[test]
fn planar_transposes_bend_where_permute_braids_for_a_fermionic_provider() {
    // What: for a provider whose braiding is not symmetric, a planar transpose
    // and a permute of the *same* axes are different morphisms — the permute
    // crosses strands and picks up the fermionic sign, the planar bend does
    // not. Every other test in this suite is blind to the difference, because
    // every other provider it can host is bosonic; spelling `transpose` as a
    // `permute` would pass all of them and be wrong here.
    let _guard = cache_lock();
    let runtime = runtime();
    let tensor = fermionic_rank_three(&runtime);
    assert_eq!(tensor.dense_data().unwrap(), [1.0, 2.0, 3.0, 4.0]);

    // Full transpose: same element motion either way, opposite signs.
    assert_eq!(
        full_transpose!(tensor).unwrap().dense_data().unwrap(),
        [1.0, 2.0, 4.0, 3.0]
    );
    assert_eq!(
        tensor.permute(&[2], &[1, 0]).unwrap().dense_data().unwrap(),
        [1.0, 2.0, -4.0, -3.0]
    );

    // The explicit form, on a different rotation of the planar order.
    assert_eq!(
        tensor
            .transpose(&[1, 2], &[0])
            .unwrap()
            .dense_data()
            .unwrap(),
        [1.0, 4.0, 2.0, 3.0]
    );
    assert_eq!(
        tensor.permute(&[1, 2], &[0]).unwrap().dense_data().unwrap(),
        [1.0, 4.0, -2.0, -3.0]
    );
}

#[test]
fn repartition_is_sign_free_even_for_a_fermionic_provider() {
    // What: moving the planar boundary never crosses two strands — the cyclic
    // order of the legs is what `repartition` preserves by definition — so
    // unlike the transposes above, its result carries no braiding phase and
    // coincides with the permute of the same axes even fermionically.
    //
    // Recorded because it is not obvious and it bounds what a test can prove:
    // no provider this facade can host makes a `repartition`-only substitution
    // of the planar transform by a braided one observable. The two transposes
    // above are what guard the shared planar helper the three methods route
    // through.
    let _guard = cache_lock();
    let runtime = runtime();
    let tensor = fermionic_rank_three(&runtime);

    assert_eq!(
        tensor.repartition(0).unwrap().dense_data().unwrap(),
        [1.0, 4.0, 3.0, 2.0]
    );
    assert_eq!(
        tensor.repartition(1).unwrap().dense_data().unwrap(),
        [1.0, 4.0, 3.0, 2.0]
    );
    assert_eq!(
        tensor.repartition(3).unwrap().dense_data().unwrap(),
        [1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(
        tensor.repartition(0).unwrap().dense_data().unwrap(),
        tensor
            .permute(&[], &[2, 1, 0])
            .unwrap()
            .dense_data()
            .unwrap()
    );
}

// ---------------------------------------------------------------------------
// Typed cat / absorb.
// ---------------------------------------------------------------------------

/// Position-weighted value from typed U(1) labels.
fn u1_typed_fill(
    sectors: &symtn::typed::BlockFusionTrees<symtn::sector::U1Irrep>,
    indices: &[usize],
) -> f64 {
    let mut value = f64::from(sectors.coupled().charge()) * 1000.0;
    for (position, label) in sectors.codomain_uncoupled().iter().enumerate() {
        value += f64::from(label.charge()) * 100.0 * (position + 1) as f64;
    }
    for (position, label) in sectors.domain_uncoupled().iter().enumerate() {
        value += f64::from(label.charge()) * 10.0 * (position + 1) as f64;
    }
    value
        + indices
            .iter()
            .enumerate()
            .map(|(a, &i)| (a + 1) * i)
            .sum::<usize>() as f64
}

/// One typed U(1) cat operand: rank `2 <- 1` with a dual codomain leg.
fn u1_cat_tensor(
    runtime: &Runtime,
    domain_pairs: &[(i32, usize)],
) -> TensorMap<symtn::sector::U1FusionRule, f64> {
    let provider = Arc::new(symtn::sector::U1FusionRule);
    let l1 = u1_leg(&provider, &[(-1, 1), (0, 2), (1, 1)]);
    let l2 = u1_leg(&provider, &[(0, 1), (1, 2)]).try_dual().unwrap();
    let lv = u1_leg(&provider, domain_pairs);
    TensorMap::from_subblock_fn(runtime, [&l1, &l2], [&lv], u1_typed_fill).unwrap()
}

#[test]
fn typed_cat_preserves_a_dual_changed_leg_and_its_slabs() {
    // What: the typed direct sum preserves the changed leg's duality, sector
    // union, degeneracies, and lhs-then-rhs slab order directly.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(symtn::sector::FermionParityFusionRule);
    let lw = GradedSpace::try_new(
        Arc::clone(&provider),
        [
            (symtn::sector::Z2Irrep::EVEN, 1),
            (symtn::sector::Z2Irrep::ODD, 1),
        ],
    )
    .unwrap();
    let build = |pairs: &[(u8, usize)]| {
        let leg = GradedSpace::try_new(
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
        .try_dual()
        .unwrap();
        TensorMap::from_subblock_fn(&runtime, [&lw], [&leg], |sectors, indices| {
            10.0 * f64::from(sectors.coupled().parity()) + indices[1] as f64
        })
        .unwrap()
    };
    let typed_lhs = build(&[(0, 2)]);
    let typed_rhs = build(&[(0, 1), (1, 2)]);
    let typed_joined: TensorMap<symtn::sector::FermionParityFusionRule, f64> =
        typed_lhs.cat(&typed_rhs, Side::Domain).unwrap();
    let changed = &typed_joined.domain()[0];
    assert!(changed.is_dual());
    assert_eq!(
        changed.sectors().unwrap(),
        vec![symtn::sector::Z2Irrep::EVEN, symtn::sector::Z2Irrep::ODD]
    );
    assert_eq!(changed.degeneracies(), &[3, 2]);
    assert_eq!(
        typed_joined.dense_data().unwrap(),
        &[0.0, 1.0, 0.0, 10.0, 11.0]
    );
}

#[test]
fn typed_cat_pins_the_slab_order_by_value() {
    // What (gate 2): hand-computed payloads pin adjacent column slabs for
    // catdomain and adjacent
    // row slabs for catcodomain — so the slab order is pinned by value, not
    // only by structural parity.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(symtn::sector::U1FusionRule);
    let w = u1_leg(&provider, &[(0, 2)]);
    let v1 = u1_leg(&provider, &[(0, 1)]);
    let v2 = u1_leg(&provider, &[(0, 2)]);

    let a: TensorMap<symtn::sector::U1FusionRule, f64> =
        TensorMap::from_subblock_fn(&runtime, [&w], [&v1], |_, i| (i[0] + 1) as f64).unwrap();
    let b: TensorMap<symtn::sector::U1FusionRule, f64> =
        TensorMap::from_subblock_fn(&runtime, [&w], [&v2], |_, i| (i[0] + 2 * i[1] + 3) as f64)
            .unwrap();
    let joined: TensorMap<symtn::sector::U1FusionRule, f64> = a.cat(&b, Side::Domain).unwrap();
    // Column-major: lhs column [1, 2], then rhs columns [3, 4] and [5, 6].
    assert_eq!(
        joined.dense_data().unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );

    let at: TensorMap<symtn::sector::U1FusionRule, f64> =
        TensorMap::from_subblock_fn(&runtime, [&v1], [&w], |_, i| (i[1] + 1) as f64).unwrap();
    let bt: TensorMap<symtn::sector::U1FusionRule, f64> =
        TensorMap::from_subblock_fn(&runtime, [&v2], [&w], |_, i| (i[0] + 2 * i[1] + 3) as f64)
            .unwrap();
    let stacked: TensorMap<symtn::sector::U1FusionRule, f64> = at.cat(&bt, Side::Codomain).unwrap();
    // Row slabs: lhs row first within each column.
    assert_eq!(
        stacked.dense_data().unwrap(),
        &[1.0, 3.0, 4.0, 2.0, 5.0, 6.0]
    );
}

#[test]
fn typed_absorb_pins_the_common_prefix_by_value() {
    // What (gate 3): absorb's common-prefix semantics against a hand-computed
    // payload. Destination block is 2x3, source block is 3x2; the shared
    // prefix is 2x2, so exactly those four column-major entries change.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(symtn::sector::U1FusionRule);
    let destination: TensorMap<symtn::sector::U1FusionRule, f64> = TensorMap::from_subblock_fn(
        &runtime,
        [&u1_leg(&provider, &[(0, 2)])],
        [&u1_leg(&provider, &[(0, 3)])],
        |_, i| (10 * (i[0] + 1) + i[1] + 1) as f64,
    )
    .unwrap();
    let source: TensorMap<symtn::sector::U1FusionRule, f64> = TensorMap::from_subblock_fn(
        &runtime,
        [&u1_leg(&provider, &[(0, 3)])],
        [&u1_leg(&provider, &[(0, 2)])],
        |_, i| -((10 * (i[0] + 1) + i[1] + 1) as f64),
    )
    .unwrap();
    // destination (column-major 2x3): [11, 21, 12, 22, 13, 23]
    // source (column-major 3x2): [-11, -21, -31, -12, -22, -32]
    let absorbed: TensorMap<symtn::sector::U1FusionRule, f64> =
        destination.absorb(&source).unwrap();
    assert_eq!(
        absorbed.dense_data().unwrap(),
        &[-11.0, -21.0, -12.0, -22.0, 13.0, 23.0]
    );
}

#[test]
fn typed_absorb_is_total_for_disjoint_zero_extent_and_rank_zero() {
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(symtn::sector::U1FusionRule);
    let zero = u1_leg(&provider, &[(0, 2)]);
    let one = u1_leg(&provider, &[(1, 3)]);
    let destination: TensorMap<symtn::sector::U1FusionRule, f64> =
        TensorMap::from_subblock_fn(&runtime, [&zero], [&zero], |_, i| (i[0] + 10 * i[1]) as f64)
            .unwrap();
    let disjoint: TensorMap<symtn::sector::U1FusionRule, f64> =
        TensorMap::from_subblock_fn(&runtime, [&one], [&one], |_, _| 7.0).unwrap();
    assert_eq!(
        destination.absorb(&disjoint).unwrap().dense_data().unwrap(),
        destination.dense_data().unwrap()
    );

    let zero_extent = u1_leg(&provider, &[(0, 0)]);
    let empty: TensorMap<symtn::sector::U1FusionRule, f64> =
        TensorMap::zeros(&runtime, [&zero_extent], [&zero_extent]).unwrap();
    assert!(empty
        .absorb(&destination)
        .unwrap()
        .dense_data()
        .unwrap()
        .is_empty());

    let vector: TensorMap<symtn::sector::U1FusionRule, f64> =
        TensorMap::from_subblock_fn(&runtime, [&zero], [], |_, _| 2.0).unwrap();
    let covector: TensorMap<symtn::sector::U1FusionRule, f64> =
        TensorMap::from_subblock_fn(&runtime, [], [&zero], |_, _| 3.0).unwrap();
    let source_vector: TensorMap<symtn::sector::U1FusionRule, f64> =
        TensorMap::from_subblock_fn(&runtime, [&zero], [], |_, _| 5.0).unwrap();
    let source_covector: TensorMap<symtn::sector::U1FusionRule, f64> =
        TensorMap::from_subblock_fn(&runtime, [], [&zero], |_, _| 7.0).unwrap();
    let scalar_destination = covector.compose(&vector).unwrap();
    let scalar_source = source_covector.compose(&source_vector).unwrap();
    assert_eq!(
        scalar_destination
            .absorb(&scalar_source)
            .unwrap()
            .scalar()
            .unwrap(),
        scalar_source.scalar().unwrap()
    );
}

#[test]
fn typed_cat_and_absorb_validation_and_precedence_are_stable() {
    // What: each operation keeps its own validation classes and order.
    let _guard = cache_lock();
    let runtime = runtime();
    let typed_lhs = u1_cat_tensor(&runtime, &[(0, 1), (1, 1)]);
    let assert_invalid = |error: symtn::typed::Error, fragment: &str| {
        assert!(
            matches!(&error, symtn::typed::Error::InvalidArgument(_)),
            "{error:?}"
        );
        assert!(format!("{error:?}").contains(fragment), "{error:?}");
    };
    let assert_mismatch = |error: symtn::typed::Error, fragment: &str| {
        assert!(
            matches!(&error, symtn::typed::Error::Operation(operation)
                if matches!(**operation, symtn::typed::OperationError::SpaceMismatch { .. })),
            "{error:?}"
        );
        assert!(format!("{error:?}").contains(fragment), "{error:?}");
    };

    // Wrong rank: a multi-leg changed side.
    let provider = Arc::new(symtn::sector::U1FusionRule);
    let two_legs_typed: TensorMap<symtn::sector::U1FusionRule, f64> = TensorMap::from_subblock_fn(
        &runtime,
        [
            &u1_leg(&provider, &[(-1, 1), (0, 2), (1, 1)]),
            &u1_leg(&provider, &[(0, 1), (1, 2)]).try_dual().unwrap(),
        ],
        [&u1_leg(&provider, &[(0, 1)]), &u1_leg(&provider, &[(0, 1)])],
        u1_typed_fill,
    )
    .unwrap();
    assert_invalid(
        typed_lhs.cat(&two_legs_typed, Side::Domain).unwrap_err(),
        "exactly one domain leg",
    );

    // Mismatched unchanged side, catdomain: rank `2 <- 1` on both operands
    // (the one-domain-leg check passes) with different codomain product
    // spaces, so the codomain-equality check itself is what fires.
    let other_codomain_typed: TensorMap<symtn::sector::U1FusionRule, f64> =
        TensorMap::from_subblock_fn(
            &runtime,
            [
                &u1_leg(&provider, &[(-1, 1), (0, 3), (1, 1)]),
                &u1_leg(&provider, &[(0, 1), (1, 2)]).try_dual().unwrap(),
            ],
            [&u1_leg(&provider, &[(0, 1), (1, 1)])],
            u1_typed_fill,
        )
        .unwrap();
    assert_mismatch(
        typed_lhs
            .cat(&other_codomain_typed, Side::Domain)
            .unwrap_err(),
        "identical codomain product spaces",
    );

    // Mismatched unchanged side, catcodomain: rank `1 <- 2` on both operands
    // (the one-codomain-leg check passes) with different domain product
    // spaces, so the domain-equality check itself is what fires.
    let stack_pair = |domain1: &[(i32, usize)]| {
        TensorMap::from_subblock_fn(
            &runtime,
            [&u1_leg(&provider, &[(0, 2), (1, 1)])],
            [
                &u1_leg(&provider, &[(-1, 1), (0, 1), (1, 1)]),
                &u1_leg(&provider, domain1),
            ],
            u1_typed_fill,
        )
        .unwrap()
    };
    let typed_stack_lhs: TensorMap<symtn::sector::U1FusionRule, f64> =
        stack_pair(&[(0, 1), (1, 1)]);
    let typed_stack_rhs: TensorMap<symtn::sector::U1FusionRule, f64> =
        stack_pair(&[(0, 2), (1, 1)]);
    assert_mismatch(
        typed_stack_lhs
            .cat(&typed_stack_rhs, Side::Codomain)
            .unwrap_err(),
        "identical domain product spaces",
    );

    // Duality mismatch on the changed leg: the direct sum refuses.
    let dual_domain_typed: TensorMap<symtn::sector::U1FusionRule, f64> =
        TensorMap::from_subblock_fn(
            &runtime,
            [
                &u1_leg(&provider, &[(-1, 1), (0, 2), (1, 1)]),
                &u1_leg(&provider, &[(0, 1), (1, 2)]).try_dual().unwrap(),
            ],
            [&u1_leg(&provider, &[(0, 1), (1, 1)]).try_dual().unwrap()],
            u1_typed_fill,
        )
        .unwrap();
    assert_mismatch(
        typed_lhs.cat(&dual_domain_typed, Side::Domain).unwrap_err(),
        "opposite duality",
    );

    assert_invalid(
        typed_lhs.absorb(&two_legs_typed).unwrap_err(),
        "equal codomain/domain ranks",
    );
    assert_invalid(
        typed_lhs.absorb(&dual_domain_typed).unwrap_err(),
        "equal duality",
    );

    // Runtime mismatch, for all three operations.
    let other_runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let typed_other = u1_cat_tensor(&other_runtime, &[(0, 1), (1, 1)]);
    for error in [
        typed_lhs.cat(&typed_other, Side::Domain).unwrap_err(),
        typed_lhs.cat(&typed_other, Side::Codomain).unwrap_err(),
        typed_lhs.absorb(&typed_other).unwrap_err(),
    ] {
        assert!(matches!(error, symtn::typed::Error::RuntimeMismatch));
    }
}

#[test]
fn typed_cat_and_absorb_reject_a_foreign_rule_identity_first() {
    // What (gate 4): the rule-identity check fires before any space validation:
    // two providers of the same Rust type but different identities cannot be
    // concatenated or absorbed and report `RuleMismatch`.
    let _guard = cache_lock();
    let runtime = runtime();
    let build = |provider: &Arc<ExternalZ3>| {
        let leg = GradedSpace::try_new(Arc::clone(provider), [(Z3Charge(0), 1), (Z3Charge(1), 1)])
            .unwrap();
        let tensor: TensorMap<ExternalZ3, f64> =
            TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, _| 1.0).unwrap();
        tensor
    };
    let ours = build(&Arc::new(ExternalZ3::new()));
    let theirs = build(&Arc::new(ExternalZ3::tagged(7)));
    for error in [
        ours.cat(&theirs, Side::Domain).unwrap_err(),
        ours.cat(&theirs, Side::Codomain).unwrap_err(),
        ours.absorb(&theirs).unwrap_err(),
    ] {
        assert!(matches!(error, symtn::typed::Error::RuleMismatch));
    }

    let other_runtime = Runtime::builder().build().unwrap();
    let foreign = Arc::new(ExternalZ3::tagged(7));
    let foreign_leg =
        GradedSpace::try_new(Arc::clone(&foreign), [(Z3Charge(0), 1), (Z3Charge(1), 1)]).unwrap();
    let foreign_runtime: TensorMap<ExternalZ3, f64> =
        TensorMap::from_subblock_fn(&other_runtime, [&foreign_leg], [&foreign_leg], |_, _| 1.0)
            .unwrap();
    assert!(matches!(
        ours.absorb(&foreign_runtime).unwrap_err(),
        symtn::typed::Error::RuleMismatch
    ));

    let foreign_bad_rank: TensorMap<ExternalZ3, f64> = TensorMap::from_subblock_fn(
        &other_runtime,
        [&foreign_leg, &foreign_leg],
        [&foreign_leg],
        |_, _| 1.0,
    )
    .unwrap();
    assert!(matches!(
        ours.absorb(&foreign_bad_rank).unwrap_err(),
        symtn::typed::Error::InvalidArgument(_)
    ));
}

#[test]
fn external_z3_cat_and_absorb_hold_by_value() {
    // What (gate 6): typed-only law/value checks on the external Z3 provider.
    // The expected payloads are computed by hand rather than through another
    // facade.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalZ3::new());
    let w =
        GradedSpace::try_new(Arc::clone(&provider), [(Z3Charge(0), 1), (Z3Charge(1), 1)]).unwrap();
    let v1 =
        GradedSpace::try_new(Arc::clone(&provider), [(Z3Charge(0), 1), (Z3Charge(1), 1)]).unwrap();
    let v2 =
        GradedSpace::try_new(Arc::clone(&provider), [(Z3Charge(1), 1), (Z3Charge(2), 1)]).unwrap();
    let a: TensorMap<ExternalZ3, f64> =
        TensorMap::from_subblock_fn(&runtime, [&w], [&v1], |sectors, _| {
            1.0 + f64::from(sectors.coupled().0)
        })
        .unwrap();
    let b: TensorMap<ExternalZ3, f64> =
        TensorMap::from_subblock_fn(&runtime, [&w], [&v2], |sectors, _| {
            10.0 + f64::from(sectors.coupled().0)
        })
        .unwrap();

    let joined: TensorMap<ExternalZ3, f64> = a.cat(&b, Side::Domain).unwrap();
    // Blocks in coupled-sector order: charge 0 holds only the lhs value, the
    // shared charge 1 holds the lhs column then the rhs column; charge 2 has
    // no codomain sector, so the merged leg carries it without a block.
    assert_eq!(joined.dense_data().unwrap(), &[1.0, 2.0, 11.0]);
    assert_eq!(
        joined.domain()[0].sectors().unwrap(),
        vec![Z3Charge(0), Z3Charge(1), Z3Charge(2)]
    );
    assert_eq!(joined.domain()[0].degeneracies(), &[1, 2, 1]);

    // Absorb: destination charge-0 block is 1x1, source block 1x1 — the
    // shared charge-0 and charge-1 blocks are overwritten; nothing else
    // exists. Then a non-shared key: absorb from a tensor whose only block is
    // charge 1.
    let c: TensorMap<ExternalZ3, f64> =
        TensorMap::from_subblock_fn(&runtime, [&w], [&v2], |sectors, _| {
            100.0 + f64::from(sectors.coupled().0)
        })
        .unwrap();
    let absorbed: TensorMap<ExternalZ3, f64> = a.absorb(&c).unwrap();
    // a's blocks: charge 0 -> 1.0, charge 1 -> 2.0; c has only charge 1
    // (value 101.0). The non-shared charge-0 block is untouched.
    assert_eq!(absorbed.dense_data().unwrap(), &[1.0, 101.0]);
}

fn probe_leg<const ANYONIC: bool>() -> GradedSpace<RealBraidingProbe<ANYONIC>> {
    GradedSpace::try_new(Arc::new(RealBraidingProbe::<ANYONIC>), [(ProbeSector, 2)]).unwrap()
}

/// Entry `(i, j)` of the one 2x2 block is `offset + 2 i + j`.
fn probe_value(offset: f64, row: usize, column: usize) -> f64 {
    offset + (2 * row + column) as f64
}

fn probe_matrix<const ANYONIC: bool>(
    runtime: &Runtime,
    offset: f64,
) -> TensorMap<RealBraidingProbe<ANYONIC>, f64> {
    let leg = probe_leg::<ANYONIC>();
    TensorMap::from_subblock_fn(runtime, [&leg], [&leg], |_, index| {
        probe_value(offset, index[0], index[1])
    })
    .unwrap()
}

/// Hand oracle of `lhs.compose(rhs)` for the offsets 1 and 5: the one block
/// is the 2x2 matrix product.
fn probe_composition(row: usize, column: usize) -> f64 {
    (0..2)
        .map(|k| probe_value(1.0, row, k) * probe_value(5.0, k, column))
        .sum()
}

fn is_unsupported_contract_scope(error: &symtn::typed::Error) -> bool {
    matches!(
        error,
        symtn::typed::Error::Operation(operation)
            if matches!(**operation, symtn::typed::OperationError::UnsupportedTensorContractScope { .. })
    )
}

fn is_non_symmetric_contraction(error: &symtn::typed::Error) -> bool {
    matches!(
        error,
        symtn::typed::Error::Operation(operation)
            if matches!(
                **operation,
                symtn::typed::OperationError::UnsupportedTensorContractScope {
                    message: symtn::typed::NON_SYMMETRIC_CONTRACTION_UNSUPPORTED
                }
            )
    )
}

/// Every Host contraction entry rejects a non-symmetric braiding with one
/// error, even on the canonical axes, before touching the destination, while
/// `compose` of the same operands is admitted (TensorKit `blas_contract!`
/// versus `mul!`).
fn assert_host_contract_entries_reject_but_compose_admits<const ANYONIC: bool>() {
    let runtime = runtime();
    let lhs = probe_matrix::<ANYONIC>(&runtime, 1.0);
    let rhs = probe_matrix::<ANYONIC>(&runtime, 5.0);
    let mut destination = probe_matrix::<ANYONIC>(&runtime, 9.0);
    let before = destination.dense_data().unwrap().to_vec();

    let contract = lhs
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
    assert!(is_non_symmetric_contraction(&contract), "{contract:?}");
    let ordered = lhs
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
    let overwrite = lhs
        .contract_into(
            &rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
            &mut destination,
            1.0,
            0.0,
        )
        .unwrap_err();
    let ordered_overwrite = lhs
        .contract_into(
            &rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
            &mut destination,
            1.0,
            0.0,
        )
        .unwrap_err();
    for error in [&ordered, &overwrite, &ordered_overwrite] {
        assert!(is_non_symmetric_contraction(error), "{error:?}");
    }
    assert_eq!(destination.dense_data().unwrap(), &before[..]);

    let composed = lhs.compose(&rhs).unwrap();
    let leg = probe_leg::<ANYONIC>();
    let expected: TensorMap<RealBraidingProbe<ANYONIC>, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
            probe_composition(index[0], index[1])
        })
        .unwrap();
    assert_eq!(
        composed.dense_data().unwrap(),
        expected.dense_data().unwrap()
    );
}

#[test]
fn non_symmetric_contract_entries_reject_canonical_axes_but_compose_admits() {
    let _guard = cache_lock();
    assert_host_contract_entries_reject_but_compose_admits::<false>();
    assert_host_contract_entries_reject_but_compose_admits::<true>();
}

#[test]
fn symmetric_canonical_contract_is_admitted_as_the_negative_control() {
    // What: the same one-sector 2x2 operands under a bosonic rule contract on
    // the canonical axes to the composition oracle, so the rejections above
    // are the braiding boundary alone.
    let _guard = cache_lock();
    let runtime = runtime();
    let leg = GradedSpace::try_new(
        Arc::new(symtn::sector::U1FusionRule),
        [(symtn::sector::U1Irrep::new(0), 2)],
    )
    .unwrap();
    let matrix = |offset: f64| -> TensorMap<symtn::sector::U1FusionRule, f64> {
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
            probe_value(offset, index[0], index[1])
        })
        .unwrap()
    };
    let expected: TensorMap<symtn::sector::U1FusionRule, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
            probe_composition(index[0], index[1])
        })
        .unwrap();
    let contracted = matrix(1.0)
        .contract(
            &matrix(5.0),
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    assert_eq!(
        contracted.dense_data().unwrap(),
        expected.dense_data().unwrap()
    );
}

fn assert_compact_trace_rejects_like_dense_trace<const ANYONIC: bool>() {
    let runtime = runtime();
    let dense = probe_matrix::<ANYONIC>(&runtime, 1.0);
    let compact: TensorMap<RealBraidingProbe<ANYONIC>, f64> = TensorMap::diagonal(
        &runtime,
        &probe_leg::<ANYONIC>(),
        [symtn::typed::SectorSpectrum {
            sector: ProbeSector,
            values: vec![1.0, 2.0],
        }],
    )
    .unwrap();
    let compact_error = compact.trace_pairs(&[(0, 1)]).unwrap_err();
    let dense_error = dense.trace_pairs(&[(0, 1)]).unwrap_err();
    assert!(
        is_unsupported_contract_scope(&compact_error),
        "{compact_error:?}"
    );
    assert_eq!(compact_error.to_string(), dense_error.to_string());
}

#[test]
fn non_symmetric_compact_trace_rejects_like_dense_trace() {
    // What: the rank-(1,1) compact-spectrum trace is rejected with the dense
    // trace's error instead of answering from the spectrum (#1355).
    let _guard = cache_lock();
    assert_compact_trace_rejects_like_dense_trace::<false>();
    assert_compact_trace_rejects_like_dense_trace::<true>();
}

#[test]
fn compact_braid_matches_dense_replay_when_r_overflows_or_underflows_f32() {
    fn compare<const R_SCALE: u8>(values: [f32; 2]) {
        let runtime = runtime();
        let leg = GradedSpace::try_new(
            Arc::new(RealBraidingProbe::<true, R_SCALE>),
            [(ProbeSector, 2)],
        )
        .unwrap();
        let compact: TensorMap<RealBraidingProbe<true, R_SCALE>, f32> = TensorMap::diagonal(
            &runtime,
            &leg,
            [symtn::typed::SectorSpectrum {
                sector: ProbeSector,
                values: values.to_vec(),
            }],
        )
        .unwrap();
        let actual = compact.braid(&[1], &[0], &[0, 1]).unwrap();
        let expected = compact
            .materialize()
            .unwrap()
            .braid(&[1], &[0], &[0, 1])
            .unwrap();
        if R_SCALE == 1 {
            assert!(expected
                .dense_data()
                .unwrap()
                .iter()
                .all(|&value| value == 0.0));
        } else {
            assert!(expected
                .dense_data()
                .unwrap()
                .iter()
                .any(|value| value.is_nan()));
        }
        for (&actual, &expected) in actual
            .dense_data()
            .unwrap()
            .iter()
            .zip(expected.dense_data().unwrap())
        {
            if expected.is_nan() {
                assert!(actual.is_nan());
            } else {
                assert_eq!(actual, expected);
            }
        }
    }
    let _guard = cache_lock();
    compare::<1>([f32::INFINITY, 2.0]);
    compare::<2>([1.0, 2.0]);
}

/// The device entries follow the Host order: the rejection comes before any
/// device work and leaves the destination as it was; device `compose` is
/// admitted and agrees with the Host.
#[cfg(feature = "cuda")]
fn assert_device_contract_entries_reject_but_compose_admits<const ANYONIC: bool>() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let host_lhs = probe_matrix::<ANYONIC>(&runtime, 1.0);
    let host_rhs = probe_matrix::<ANYONIC>(&runtime, 5.0);
    let lhs = host_lhs.to_cuda().unwrap();
    let rhs = host_rhs.to_cuda().unwrap();
    let mut destination = probe_matrix::<ANYONIC>(&runtime, 9.0).to_cuda().unwrap();
    let before = destination
        .to_host()
        .unwrap()
        .dense_data()
        .unwrap()
        .to_vec();

    let transfers = symtn::expert::cuda_transfer_stats();
    let contract = lhs
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
    let ordered = lhs
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
    let overwrite = lhs
        .contract_into(
            &rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
            &mut destination,
            1.0,
            0.0,
        )
        .unwrap_err();
    assert_eq!(symtn::expert::cuda_transfer_stats(), transfers);
    for error in [&contract, &ordered, &overwrite] {
        assert!(is_non_symmetric_contraction(error), "{error:?}");
    }
    assert_eq!(
        destination.to_host().unwrap().dense_data().unwrap(),
        &before[..]
    );

    let composed = lhs.compose(&rhs).unwrap().to_host().unwrap();
    assert_eq!(
        composed.dense_data().unwrap(),
        host_lhs.compose(&host_rhs).unwrap().dense_data().unwrap()
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn non_symmetric_device_contract_entries_reject_but_compose_admits() {
    let _guard = cache_lock();
    assert_device_contract_entries_reject_but_compose_admits::<false>();
    assert_device_contract_entries_reject_but_compose_admits::<true>();
}
