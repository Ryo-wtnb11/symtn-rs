use super::*;
use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::SectorSpectrum;

#[test]
fn permute_moves_legs_of_a_multi_block_external_provider_tensor() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: a non-identity permute of a runtime-rank multi-block external
    // provider tensor produces the reordered spaces, keeps the element count,
    // and moves the payload (this rule's F/R symbols are all 1, so the permuted
    // buffer is a rearrangement of the source, never a rescaling).
    let runtime = runtime();
    let provider = Arc::new(external_z3());
    let tensor = z3_rank_four(&runtime, &provider);

    // The source is `[wide, narrow] <- [wide', narrow']`; this order sends one
    // leg of each shape across the codomain/domain split, so the degeneracies
    // pin which source leg landed where. Flags alone would not: they are
    // symmetric under swapping the split.
    let permuted = tensor.permute(engine, &[1, 2], &[3, 0]).unwrap();

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
fn permute_carries_a_simple_fusion_provider_with_a_complex_payload() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: nothing in the typed transform is abelian- or real-specific; the
    // SU(2) recoupling coefficients reach a `Complex64` payload.
    let runtime = runtime();
    let provider = Arc::new(external_su2());
    let leg = su2_leg(&provider, false);
    let tensor: TensorMap<ExternalSu2Rule, Complex64> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&leg, &leg],
        [&leg, &leg],
        |sectors, indices| {
            Complex64::new(
                sectors.coupled().twice_spin() as f64 + 1.0,
                indices.iter().sum::<usize>() as f64 + 1.0,
            )
        },
    )
    .unwrap();

    let permuted = tensor.permute(engine, &[0, 2], &[1, 3]).unwrap();

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
fn transpose_twice_returns_the_source_layout() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: the planar transpose is an involution — it rotates every leg once
    // round the boundary, so applying it twice restores the source spaces,
    // block order and bytes exactly.
    let runtime = runtime();
    let provider = Arc::new(external_z3());
    let tensor = z3_rank_four(&runtime, &provider);

    let once = full_transpose!(engine, tensor).unwrap();
    let twice = full_transpose!(engine, once).unwrap();

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

// ---------------------------------------------------------------------------
// `TensorMap::repartition`.
// ---------------------------------------------------------------------------

#[test]
fn repartition_moves_the_boundary_and_round_trips_at_every_split() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: every split point of a rank-4 tensor map is reachable, reports the
    // requested codomain/domain sizes, and comes back to the source layout —
    // spaces, block identities and bytes — when repartitioned to the original
    // split.
    let runtime = runtime();
    let provider = Arc::new(external_z3());
    let tensor = z3_rank_four(&runtime, &provider);

    for num_codomain in 0..=4 {
        let moved = tensor.repartition(engine, num_codomain).unwrap();
        assert_eq!(moved.codomain().len(), num_codomain);
        assert_eq!(moved.domain().len(), 4 - num_codomain);
        assert_eq!(
            moved.dense_data().unwrap().len(),
            tensor.dense_data().unwrap().len()
        );

        let back = moved.repartition(engine, 2).unwrap();
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
fn planar_transposes_bend_where_permute_braids_for_a_fermionic_provider() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: for a provider whose braiding is not symmetric, a planar transpose
    // and a permute of the *same* axes are different morphisms — the permute
    // crosses strands and picks up the fermionic sign, the planar bend does
    // not. Every other test in this suite is blind to the difference, because
    // every other provider it can host is bosonic; spelling `transpose` as a
    // `permute` would pass all of them and be wrong here.
    let runtime = runtime();
    let tensor = fermionic_rank_three(&runtime);
    assert_eq!(tensor.dense_data().unwrap(), [1.0, 2.0, 3.0, 4.0]);

    // Full transpose: same element motion either way, opposite signs.
    assert_eq!(
        full_transpose!(engine, tensor)
            .unwrap()
            .dense_data()
            .unwrap(),
        [1.0, 2.0, 4.0, 3.0]
    );
    assert_eq!(
        tensor
            .permute(engine, &[2], &[1, 0])
            .unwrap()
            .dense_data()
            .unwrap(),
        [1.0, 2.0, -4.0, -3.0]
    );

    // The explicit form, on a different rotation of the planar order.
    assert_eq!(
        tensor
            .transpose(engine, &[1, 2], &[0])
            .unwrap()
            .dense_data()
            .unwrap(),
        [1.0, 4.0, 2.0, 3.0]
    );
    assert_eq!(
        tensor
            .permute(engine, &[1, 2], &[0])
            .unwrap()
            .dense_data()
            .unwrap(),
        [1.0, 4.0, -2.0, -3.0]
    );
}

#[test]
fn repartition_is_sign_free_even_for_a_fermionic_provider() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
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
    let runtime = runtime();
    let tensor = fermionic_rank_three(&runtime);

    assert_eq!(
        tensor.repartition(engine, 0).unwrap().dense_data().unwrap(),
        [1.0, 4.0, 3.0, 2.0]
    );
    assert_eq!(
        tensor.repartition(engine, 1).unwrap().dense_data().unwrap(),
        [1.0, 4.0, 3.0, 2.0]
    );
    assert_eq!(
        tensor.repartition(engine, 3).unwrap().dense_data().unwrap(),
        [1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(
        tensor.repartition(engine, 0).unwrap().dense_data().unwrap(),
        tensor
            .permute(engine, &[], &[2, 1, 0])
            .unwrap()
            .dense_data()
            .unwrap()
    );
}

// ---------------------------------------------------------------------------
// Typed cat / absorb.
// ---------------------------------------------------------------------------

#[test]
fn typed_cat_preserves_a_dual_changed_leg_and_its_slabs() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: the typed direct sum preserves the changed leg's duality, sector
    // union, degeneracies, and lhs-then-rhs slab order directly.
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
    let mut build = |pairs: &[(u8, usize)]| {
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
        TensorMap::from_subblock_fn(engine, &runtime, [&lw], [&leg], |sectors, indices| {
            10.0 * f64::from(sectors.coupled().parity()) + indices[1] as f64
        })
        .unwrap()
    };
    let typed_lhs = build(&[(0, 2)]);
    let typed_rhs = build(&[(0, 1), (1, 2)]);
    let typed_joined: TensorMap<symtn::sector::FermionParityFusionRule, f64> =
        typed_lhs.cat(engine, &typed_rhs, Side::Domain).unwrap();
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
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What (gate 2): hand-computed payloads pin adjacent column slabs for
    // catdomain and adjacent
    // row slabs for catcodomain — so the slab order is pinned by value, not
    // only by structural parity.
    let runtime = runtime();
    let provider = Arc::new(symtn::sector::U1FusionRule);
    let w = u1_leg(&provider, &[(0, 2)]);
    let v1 = u1_leg(&provider, &[(0, 1)]);
    let v2 = u1_leg(&provider, &[(0, 2)]);

    let a: TensorMap<symtn::sector::U1FusionRule, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&w], [&v1], |_, i| (i[0] + 1) as f64)
            .unwrap();
    let b: TensorMap<symtn::sector::U1FusionRule, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&w], [&v2], |_, i| {
            (i[0] + 2 * i[1] + 3) as f64
        })
        .unwrap();
    let joined: TensorMap<symtn::sector::U1FusionRule, f64> =
        a.cat(engine, &b, Side::Domain).unwrap();
    // Column-major: lhs column [1, 2], then rhs columns [3, 4] and [5, 6].
    assert_eq!(
        joined.dense_data().unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );

    let at: TensorMap<symtn::sector::U1FusionRule, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&v1], [&w], |_, i| (i[1] + 1) as f64)
            .unwrap();
    let bt: TensorMap<symtn::sector::U1FusionRule, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&v2], [&w], |_, i| {
            (i[0] + 2 * i[1] + 3) as f64
        })
        .unwrap();
    let stacked: TensorMap<symtn::sector::U1FusionRule, f64> =
        at.cat(engine, &bt, Side::Codomain).unwrap();
    // Row slabs: lhs row first within each column.
    assert_eq!(
        stacked.dense_data().unwrap(),
        &[1.0, 3.0, 4.0, 2.0, 5.0, 6.0]
    );
}

#[test]
fn typed_absorb_pins_the_common_prefix_by_value() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What (gate 3): absorb's common-prefix semantics against a hand-computed
    // payload. Destination block is 2x3, source block is 3x2; the shared
    // prefix is 2x2, so exactly those four column-major entries change.
    let runtime = runtime();
    let provider = Arc::new(symtn::sector::U1FusionRule);
    let destination: TensorMap<symtn::sector::U1FusionRule, f64> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&u1_leg(&provider, &[(0, 2)])],
        [&u1_leg(&provider, &[(0, 3)])],
        |_, i| (10 * (i[0] + 1) + i[1] + 1) as f64,
    )
    .unwrap();
    let source: TensorMap<symtn::sector::U1FusionRule, f64> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&u1_leg(&provider, &[(0, 3)])],
        [&u1_leg(&provider, &[(0, 2)])],
        |_, i| -((10 * (i[0] + 1) + i[1] + 1) as f64),
    )
    .unwrap();
    // destination (column-major 2x3): [11, 21, 12, 22, 13, 23]
    // source (column-major 3x2): [-11, -21, -31, -12, -22, -32]
    let absorbed: TensorMap<symtn::sector::U1FusionRule, f64> =
        destination.absorb(engine, &source).unwrap();
    assert_eq!(
        absorbed.dense_data().unwrap(),
        &[-11.0, -21.0, -12.0, -22.0, 13.0, 23.0]
    );
}

#[test]
fn typed_absorb_is_total_for_disjoint_zero_extent_and_rank_zero() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = runtime();
    let provider = Arc::new(symtn::sector::U1FusionRule);
    let zero = u1_leg(&provider, &[(0, 2)]);
    let one = u1_leg(&provider, &[(1, 3)]);
    let destination: TensorMap<symtn::sector::U1FusionRule, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&zero], [&zero], |_, i| {
            (i[0] + 10 * i[1]) as f64
        })
        .unwrap();
    let disjoint: TensorMap<symtn::sector::U1FusionRule, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&one], [&one], |_, _| 7.0).unwrap();
    assert_eq!(
        destination
            .absorb(engine, &disjoint)
            .unwrap()
            .dense_data()
            .unwrap(),
        destination.dense_data().unwrap()
    );

    let zero_extent = u1_leg(&provider, &[(0, 0)]);
    let empty: TensorMap<symtn::sector::U1FusionRule, f64> =
        TensorMap::zeros(engine, &runtime, [&zero_extent], [&zero_extent]).unwrap();
    assert!(empty
        .absorb(engine, &destination)
        .unwrap()
        .dense_data()
        .unwrap()
        .is_empty());

    let vector: TensorMap<symtn::sector::U1FusionRule, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&zero], [], |_, _| 2.0).unwrap();
    let covector: TensorMap<symtn::sector::U1FusionRule, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [], [&zero], |_, _| 3.0).unwrap();
    let source_vector: TensorMap<symtn::sector::U1FusionRule, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&zero], [], |_, _| 5.0).unwrap();
    let source_covector: TensorMap<symtn::sector::U1FusionRule, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [], [&zero], |_, _| 7.0).unwrap();
    let scalar_destination = covector.compose(engine, &vector).unwrap();
    let scalar_source = source_covector.compose(engine, &source_vector).unwrap();
    assert_eq!(
        scalar_destination
            .absorb(engine, &scalar_source)
            .unwrap()
            .scalar(engine)
            .unwrap(),
        scalar_source.scalar(engine).unwrap()
    );
}

#[test]
fn typed_cat_and_absorb_reject_a_foreign_rule_identity_first() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What (gate 4): the rule-identity check fires before any space validation:
    // two providers of the same Rust type but different identities cannot be
    // concatenated or absorbed and report `RuleMismatch`.
    let runtime = runtime();
    let mut build = |provider: &Arc<ExternalZ3Rule>| {
        let leg = GradedSpace::try_new(Arc::clone(provider), [(Z3Charge(0), 1), (Z3Charge(1), 1)])
            .unwrap();
        let tensor: TensorMap<ExternalZ3Rule, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, _| 1.0).unwrap();
        tensor
    };
    let ours = build(&Arc::new(external_z3()));
    let theirs = build(&Arc::new(external_z3_tagged(7)));
    for error in [
        ours.cat(engine, &theirs, Side::Domain).unwrap_err(),
        ours.cat(engine, &theirs, Side::Codomain).unwrap_err(),
        ours.absorb(engine, &theirs).unwrap_err(),
    ] {
        assert!(matches!(error, symtn::typed::Error::RuleMismatch));
    }

    let other_runtime = Runtime::builder().build().unwrap();
    let foreign = Arc::new(external_z3_tagged(7));
    let foreign_leg =
        GradedSpace::try_new(Arc::clone(&foreign), [(Z3Charge(0), 1), (Z3Charge(1), 1)]).unwrap();
    let foreign_runtime: TensorMap<ExternalZ3Rule, f64> = TensorMap::from_subblock_fn(
        engine,
        &other_runtime,
        [&foreign_leg],
        [&foreign_leg],
        |_, _| 1.0,
    )
    .unwrap();
    assert!(matches!(
        ours.absorb(engine, &foreign_runtime).unwrap_err(),
        symtn::typed::Error::RuleMismatch
    ));

    let foreign_bad_rank: TensorMap<ExternalZ3Rule, f64> = TensorMap::from_subblock_fn(
        engine,
        &other_runtime,
        [&foreign_leg, &foreign_leg],
        [&foreign_leg],
        |_, _| 1.0,
    )
    .unwrap();
    assert!(matches!(
        ours.absorb(engine, &foreign_bad_rank).unwrap_err(),
        symtn::typed::Error::InvalidArgument(_)
    ));
}

#[test]
fn external_z3_cat_and_absorb_hold_by_value() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What (gate 6): typed-only law/value checks on the external Z3 provider.
    // The expected payloads are computed by hand rather than through another
    // facade.
    let runtime = runtime();
    let provider = Arc::new(external_z3());
    let w =
        GradedSpace::try_new(Arc::clone(&provider), [(Z3Charge(0), 1), (Z3Charge(1), 1)]).unwrap();
    let v1 =
        GradedSpace::try_new(Arc::clone(&provider), [(Z3Charge(0), 1), (Z3Charge(1), 1)]).unwrap();
    let v2 =
        GradedSpace::try_new(Arc::clone(&provider), [(Z3Charge(1), 1), (Z3Charge(2), 1)]).unwrap();
    let a: TensorMap<ExternalZ3Rule, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&w], [&v1], |sectors, _| {
            1.0 + f64::from(sectors.coupled().0)
        })
        .unwrap();
    let b: TensorMap<ExternalZ3Rule, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&w], [&v2], |sectors, _| {
            10.0 + f64::from(sectors.coupled().0)
        })
        .unwrap();

    let joined: TensorMap<ExternalZ3Rule, f64> = a.cat(engine, &b, Side::Domain).unwrap();
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
    let c: TensorMap<ExternalZ3Rule, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&w], [&v2], |sectors, _| {
            100.0 + f64::from(sectors.coupled().0)
        })
        .unwrap();
    let absorbed: TensorMap<ExternalZ3Rule, f64> = a.absorb(engine, &c).unwrap();
    // a's blocks: charge 0 -> 1.0, charge 1 -> 2.0; c has only charge 1
    // (value 101.0). The non-shared charge-0 block is untouched.
    assert_eq!(absorbed.dense_data().unwrap(), &[1.0, 101.0]);
}

#[test]
fn compact_braid_matches_dense_replay_when_r_overflows_or_underflows_f32() {
    let _serial = crate::counting_alloc::serial();
    fn compare<const R_SCALE: u8>(values: [f32; 2]) {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let runtime = runtime();
        let leg = GradedSpace::try_new(
            Arc::new(RealBraidingProbe::<true, R_SCALE>),
            [(ProbeSector, 2)],
        )
        .unwrap();
        let compact: TensorMap<RealBraidingProbe<true, R_SCALE>, f32> = TensorMap::diagonal(
            engine,
            &runtime,
            &leg,
            [symtn::typed::SectorSpectrum {
                sector: ProbeSector,
                values: values.to_vec(),
            }],
        )
        .unwrap();
        let actual = compact.braid(engine, &[1], &[0], &[0, 1]).unwrap();
        let expected = compact
            .materialize(engine)
            .unwrap()
            .braid(engine, &[1], &[0], &[0, 1])
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
    compare::<1>([f32::INFINITY, 2.0]);
    compare::<2>([1.0, 2.0]);
}

fn catcodomain_oracle<D: Copy>(lhs: &[D], lhs_rows: usize, rhs: &[D], rhs_rows: usize) -> Vec<D> {
    let columns = lhs.len() / lhs_rows;
    let mut output = Vec::with_capacity(lhs.len() + rhs.len());
    for column in 0..columns {
        output.extend_from_slice(&lhs[column * lhs_rows..(column + 1) * lhs_rows]);
        output.extend_from_slice(&rhs[column * rhs_rows..(column + 1) * rhs_rows]);
    }
    output
}

#[test]
fn compact_cat_ignores_zero_degeneracy_bond_sectors() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let leg = u1_leg(&Arc::new(U1FusionRule), &[(0, 0), (1, 2)]);
    let diagonal: TensorMap<U1FusionRule, f64> = TensorMap::diagonal(
        engine,
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(1),
            values: vec![2.0, 3.0],
        }],
    )
    .unwrap();
    let actual = diagonal.cat(engine, &diagonal, Side::Domain).unwrap();
    let dense = diagonal.materialize(engine).unwrap();
    let expected = dense.cat(engine, &dense, Side::Domain).unwrap();
    assert_eq!(actual.dense_data(), expected.dense_data());
}

#[test]
fn typed_lazy_adjoint_cat_matches_column_major_oracles_in_every_orientation() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let common = u1_leg(&provider, &[(0, 3)]);
    let left = u1_leg(&provider, &[(0, 2)]);
    let right = u1_leg(&provider, &[(0, 4)]);

    let lhs_parent: TensorMap<U1FusionRule, Complex64> =
        TensorMap::rand_with_seed(engine, &runtime, [&left], [&common], 773_001).unwrap();
    let rhs_parent: TensorMap<U1FusionRule, Complex64> =
        TensorMap::rand_with_seed(engine, &runtime, [&right], [&common], 773_002).unwrap();
    let lhs_owned: TensorMap<U1FusionRule, Complex64> =
        TensorMap::rand_with_seed(engine, &runtime, [&common], [&left], 773_003).unwrap();
    let rhs_owned: TensorMap<U1FusionRule, Complex64> =
        TensorMap::rand_with_seed(engine, &runtime, [&common], [&right], 773_004).unwrap();

    let lhs_adjoint = lhs_parent.adjoint(engine).unwrap();
    let rhs_adjoint = rhs_parent.adjoint(engine).unwrap();
    for (lhs, rhs, expected) in [
        (
            &lhs_adjoint,
            &rhs_owned,
            lhs_parent
                .adjoint(engine)
                .unwrap()
                .materialize(engine)
                .unwrap()
                .dense_data()
                .unwrap()
                .iter()
                .chain(rhs_owned.dense_data().unwrap())
                .copied()
                .collect::<Vec<_>>(),
        ),
        (
            &lhs_owned,
            &rhs_adjoint,
            lhs_owned
                .dense_data()
                .unwrap()
                .iter()
                .chain(
                    rhs_parent
                        .adjoint(engine)
                        .unwrap()
                        .materialize(engine)
                        .unwrap()
                        .dense_data()
                        .unwrap(),
                )
                .copied()
                .collect(),
        ),
        (
            &lhs_adjoint,
            &rhs_adjoint,
            lhs_parent
                .adjoint(engine)
                .unwrap()
                .materialize(engine)
                .unwrap()
                .dense_data()
                .unwrap()
                .iter()
                .chain(
                    rhs_parent
                        .adjoint(engine)
                        .unwrap()
                        .materialize(engine)
                        .unwrap()
                        .dense_data()
                        .unwrap(),
                )
                .copied()
                .collect(),
        ),
    ] {
        assert_eq!(
            lhs.cat(engine, rhs, Side::Domain)
                .unwrap()
                .dense_data()
                .unwrap(),
            expected
        );
    }

    let upper_parent: TensorMap<U1FusionRule, Complex64> =
        TensorMap::rand_with_seed(engine, &runtime, [&common], [&left], 773_005).unwrap();
    let lower_parent: TensorMap<U1FusionRule, Complex64> =
        TensorMap::rand_with_seed(engine, &runtime, [&common], [&right], 773_006).unwrap();
    let upper_owned: TensorMap<U1FusionRule, Complex64> =
        TensorMap::rand_with_seed(engine, &runtime, [&left], [&common], 773_007).unwrap();
    let lower_owned: TensorMap<U1FusionRule, Complex64> =
        TensorMap::rand_with_seed(engine, &runtime, [&right], [&common], 773_008).unwrap();
    let upper_adjoint = upper_parent.adjoint(engine).unwrap();
    let lower_adjoint = lower_parent.adjoint(engine).unwrap();
    for (upper, lower, expected) in [
        (
            &upper_adjoint,
            &lower_owned,
            catcodomain_oracle(
                upper_parent
                    .adjoint(engine)
                    .unwrap()
                    .materialize(engine)
                    .unwrap()
                    .dense_data()
                    .unwrap(),
                2,
                lower_owned.dense_data().unwrap(),
                4,
            ),
        ),
        (
            &upper_owned,
            &lower_adjoint,
            catcodomain_oracle(
                upper_owned.dense_data().unwrap(),
                2,
                lower_parent
                    .adjoint(engine)
                    .unwrap()
                    .materialize(engine)
                    .unwrap()
                    .dense_data()
                    .unwrap(),
                4,
            ),
        ),
        (
            &upper_adjoint,
            &lower_adjoint,
            catcodomain_oracle(
                upper_parent
                    .adjoint(engine)
                    .unwrap()
                    .materialize(engine)
                    .unwrap()
                    .dense_data()
                    .unwrap(),
                2,
                lower_parent
                    .adjoint(engine)
                    .unwrap()
                    .materialize(engine)
                    .unwrap()
                    .dense_data()
                    .unwrap(),
                4,
            ),
        ),
    ] {
        assert_eq!(
            upper
                .cat(engine, lower, Side::Codomain)
                .unwrap()
                .dense_data()
                .unwrap(),
            expected
        );
    }
}
