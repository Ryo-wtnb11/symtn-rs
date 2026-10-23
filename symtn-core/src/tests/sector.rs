use super::*;

#[test]
fn sector_leg_treats_zero_degeneracy_as_an_absent_sector() {
    // What: explicit zero degeneracies and omitted sectors identify the same
    // mathematical leg, fusion-tree keys, and hash identity.
    let even = z2_even();
    let odd = z2_odd();
    let omitted = SectorLeg::new([(even, 2)], false);
    let explicit_zero = SectorLeg::new([(odd, 0), (even, 2)], false);

    assert_eq!(explicit_zero, omitted);
    assert_eq!(explicit_zero.degeneracy(odd), None);

    let hash = |leg: &SectorLeg| {
        let mut state = std::collections::hash_map::DefaultHasher::new();
        leg.hash(&mut state);
        state.finish()
    };
    assert_eq!(hash(&explicit_zero), hash(&omitted));

    let hom = |leg| {
        FusionTreeHomSpace::new(
            FusionProductSpace::new([leg]),
            FusionProductSpace::new([SectorLeg::new([(even, 2)], false)]),
        )
    };
    assert_eq!(
        hom(explicit_zero).fusion_tree_keys(&Z2FusionRule),
        hom(omitted).fusion_tree_keys(&Z2FusionRule)
    );
}

#[test]
fn sector_leg_try_new_reports_every_duplicate_sector_declaration() {
    // What: duplicate sector declarations are construction errors
    // independent of degeneracy and input order.
    let sector = z2_even();
    for pairs in [
        [(sector, 2), (sector, 2)],
        [(sector, 2), (sector, 3)],
        [(sector, 0), (sector, 2)],
        [(sector, 2), (sector, 0)],
        [(sector, 0), (sector, 0)],
    ] {
        assert_eq!(
            SectorLeg::try_new(pairs, false),
            Err(SectorLegConstructionError::DuplicateSector { sector })
        );
    }
}

#[test]
#[should_panic(expected = "appears multiple times")]
fn sector_leg_new_preserves_the_infallible_panic_boundary() {
    // What: the compatibility constructor still rejects duplicate positive
    // sectors instead of silently selecting one declaration.
    let sector = z2_even();
    let _ = SectorLeg::new([(sector, 2), (sector, 2)], false);
}

#[test]
fn coupled_sector_dimensions_include_outer_multiplicity_and_check_overflow() {
    // What: a generic fusion channel contributes N(a,b,c), and dimension
    // arithmetic reports overflow rather than wrapping.
    let sector = SectorId::new(1);
    let product = FusionProductSpace::new([
        SectorLeg::new([(sector, 3)], false),
        SectorLeg::new([(sector, 5)], false),
    ]);
    assert_eq!(
        product
            .coupled_sector_block_dimensions(&IsomorphismMultiplicityRule)
            .unwrap(),
        BTreeMap::from([(SectorId::new(2), 30)])
    );

    let overflowing = FusionProductSpace::new([
        SectorLeg::new([(sector, usize::MAX)], false),
        SectorLeg::new([(sector, 1)], false),
    ]);
    assert_eq!(
        overflowing.coupled_sector_block_dimensions(&IsomorphismMultiplicityRule),
        Err(CoreError::ElementCountOverflow)
    );
}

fn assert_checked_contract_all_orientations<R>(
    rule: &R,
    matched: SectorLeg,
    mismatched: SectorLeg,
    open: SectorLeg,
) where
    R: CheckedFusionAlgebra,
{
    let assert_same_result = |lhs: &FusionTreeHomSpace, rhs: &FusionTreeHomSpace| {
        let infallible =
            FusionTreeHomSpace::tensorcontract_homspace(rule, lhs, rhs, &[0], &[0], &[], 0);
        let checked = FusionTreeHomSpace::try_tensorcontract_homspace_checked(
            rule,
            lhs,
            rhs,
            &[0],
            &[0],
            &[],
            0,
        );
        match (infallible, checked) {
            (Ok(expected), Ok(actual)) => assert_eq!(actual, expected),
            (Err(expected), Err(CheckedFusionSpaceError::Core(actual))) => {
                assert_eq!(*actual, expected)
            }
            (_, Err(CheckedFusionSpaceError::FusionAlgebra(error))) => {
                panic!("closed fixture unexpectedly failed checked algebra: {error}")
            }
            (expected, actual) => {
                panic!("checked/infallible contraction results differ: {expected:?} vs {actual:?}")
            }
        }
    };

    for lhs_axis in 0..2 {
        for rhs_axis in 0..2 {
            let lhs_stored = if lhs_axis == 0 {
                matched.dual(rule)
            } else {
                matched.clone()
            };
            let rhs_stored = if rhs_axis == 0 {
                matched.clone()
            } else {
                matched.dual(rule)
            };
            let lhs = if lhs_axis == 0 {
                FusionTreeHomSpace::new(
                    FusionProductSpace::new([lhs_stored]),
                    FusionProductSpace::new([open.clone()]),
                )
            } else {
                FusionTreeHomSpace::new(
                    FusionProductSpace::new([open.clone()]),
                    FusionProductSpace::new([lhs_stored]),
                )
            };
            let rhs = if rhs_axis == 0 {
                FusionTreeHomSpace::new(
                    FusionProductSpace::new([rhs_stored]),
                    FusionProductSpace::new([open.clone()]),
                )
            } else {
                FusionTreeHomSpace::new(
                    FusionProductSpace::new([open.clone()]),
                    FusionProductSpace::new([rhs_stored]),
                )
            };
            assert_direct_contract_matches_reference(
                rule,
                &lhs,
                &rhs,
                &[lhs_axis],
                &[rhs_axis],
                &[1, 0],
                1,
            );

            let bad_rhs_stored = if rhs_axis == 0 {
                mismatched.clone()
            } else {
                mismatched.dual(rule)
            };
            let bad_rhs = if rhs_axis == 0 {
                FusionTreeHomSpace::new(
                    FusionProductSpace::new([bad_rhs_stored]),
                    FusionProductSpace::new([open.clone()]),
                )
            } else {
                FusionTreeHomSpace::new(
                    FusionProductSpace::new([open.clone()]),
                    FusionProductSpace::new([bad_rhs_stored]),
                )
            };
            let lhs_contract = if lhs_axis == 0 { 0 } else { 1 };
            let rhs_contract = if rhs_axis == 0 { 0 } else { 1 };
            let infallible = FusionTreeHomSpace::tensorcontract_homspace(
                rule,
                &lhs,
                &bad_rhs,
                &[lhs_contract],
                &[rhs_contract],
                &[1, 0],
                1,
            );
            let checked = FusionTreeHomSpace::try_tensorcontract_homspace_checked(
                rule,
                &lhs,
                &bad_rhs,
                &[lhs_contract],
                &[rhs_contract],
                &[1, 0],
                1,
            );
            match (infallible, checked) {
                (Err(expected), Err(CheckedFusionSpaceError::Core(actual))) => {
                    assert_eq!(*actual, expected)
                }
                (expected, actual) => {
                    panic!("checked/infallible mismatch error differs: {expected:?} vs {actual:?}")
                }
            }
        }
    }

    // Exercise the direct codomain/domain form in addition to the four
    // stored-side combinations above.
    let direct_lhs = FusionTreeHomSpace::new(
        FusionProductSpace::new([matched.clone()]),
        FusionProductSpace::new([]),
    );
    let direct_rhs = FusionTreeHomSpace::new(
        FusionProductSpace::new([]),
        FusionProductSpace::new([matched]),
    );
    assert_same_result(&direct_lhs, &direct_rhs);
}

#[test]
fn checked_tensorcontract_matches_all_closed_rules_and_leg_orientations() {
    // What: checked contraction preserves valid HomSpaces and structural
    // errors across all four stored-side orientations and multi-sector
    // membership for every built-in multiplicity-free family.
    let fixture = |sectors: &[(SectorId, usize)], mismatch: &[(SectorId, usize)]| {
        (
            SectorLeg::new(sectors.iter().copied(), false),
            SectorLeg::new(mismatch.iter().copied(), false),
        )
    };
    let (z2, z2_bad) = fixture(
        &[(z2_even(), 1), (z2_odd(), 2)],
        &[(z2_even(), 1), (z2_odd(), 3)],
    );
    assert_checked_contract_all_orientations(
        &Z2FusionRule,
        z2,
        z2_bad,
        SectorLeg::new([(z2_even(), 1)], false),
    );
    let (fz2, fz2_bad) = fixture(
        &[(z2_even(), 2), (z2_odd(), 1)],
        &[(z2_even(), 3), (z2_odd(), 1)],
    );
    assert_checked_contract_all_orientations(
        &FermionParityFusionRule,
        fz2,
        fz2_bad,
        SectorLeg::new([(z2_even(), 1)], false),
    );
    let (u1_leg, u1_bad) = fixture(&[(u1(-2), 1), (u1(1), 2)], &[(u1(-2), 1), (u1(1), 3)]);
    assert_checked_contract_all_orientations(
        &U1FusionRule,
        u1_leg,
        u1_bad,
        SectorLeg::new([(u1(0), 1)], false),
    );
    let spin0 = su2(0);
    let spin_half = su2(1);
    let (su2_leg, su2_bad) = fixture(&[(spin0, 1), (spin_half, 2)], &[(spin0, 1), (spin_half, 3)]);
    assert_checked_contract_all_orientations(
        &SU2FusionRule,
        su2_leg,
        su2_bad,
        SectorLeg::new([(spin0, 1)], false),
    );
    let (fibonacci, fibonacci_bad) = fixture(
        &[(SectorId::new(0), 1), (SectorId::new(1), 2)],
        &[(SectorId::new(0), 1), (SectorId::new(1), 3)],
    );
    assert_checked_contract_all_orientations(
        &TableRule::fibonacci(),
        fibonacci,
        fibonacci_bad,
        SectorLeg::new([(SectorId::new(0), 1)], false),
    );

    #[cfg(target_pointer_width = "64")]
    {
        type Rule = ProductFusionRule<U1FusionRule, Z2FusionRule, TensorKitProductCodec>;
        let rule = Rule::try_new(U1FusionRule, Z2FusionRule).unwrap();
        let first = TensorKitProductCodec::encode(u1(-2), z2_even());
        let second = TensorKitProductCodec::encode(u1(1), z2_odd());
        let vacuum = TensorKitProductCodec::encode(u1(0), z2_even());
        let (product, product_bad) =
            fixture(&[(first, 1), (second, 2)], &[(first, 1), (second, 3)]);
        assert_checked_contract_all_orientations(
            &rule,
            product,
            product_bad,
            SectorLeg::new([(vacuum, 1)], false),
        );
    }
}

fn assert_direct_leg_degeneracy_structure_matches_reference<R>(
    rule: &R,
    homspace: &FusionTreeHomSpace,
) where
    R: MultiplicityFreeFusionRule,
{
    let expected = reference_leg_degeneracy_structure(rule, homspace);
    let actual = homspace
        .coupled_subblock_structure_from_leg_degeneracies(rule)
        .unwrap();
    assert_eq!(actual, expected);
    assert_ne!(actual.content_id(), expected.content_id());
    assert_eq!(
        actual.required_len().unwrap(),
        expected.required_len().unwrap()
    );
}

fn assert_coupled_grid_layout_matches_key_reconstruction<R>(rule: &R, homspace: &FusionTreeHomSpace)
where
    R: MultiplicityFreeFusionRule,
{
    let reconstructed =
        reconstructed_fusion_tree_layout_data_from_keys(homspace.fusion_tree_keys_uncached(rule));
    let direct = homspace.fusion_tree_layout_data_uncached(rule);

    assert_eq!(direct.keys, reconstructed.keys);
    assert_eq!(direct.sectors.len(), reconstructed.sectors.len());
    for (actual, expected) in direct.sectors.iter().zip(&reconstructed.sectors) {
        assert_eq!(actual.start, expected.start);
        assert_eq!(actual.row_count, expected.row_count);
        assert_eq!(actual.col_count, expected.col_count);
        assert_eq!(
            expected.row_key_offsets,
            (0..actual.row_count).collect::<Vec<_>>()
        );
        assert_eq!(
            expected.col_key_offsets,
            (0..actual.col_count)
                .map(|col| col * actual.row_count)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            expected.entries,
            (0..actual.col_count)
                .flat_map(|col| {
                    (0..actual.row_count).map(move |row| FusionTreeBlockLayoutEntry { row, col })
                })
                .collect::<Vec<_>>()
        );
    }

    let direct_parts = coupled_subblock_parts_from_leg_degeneracies(homspace, &direct).unwrap();
    let reconstructed_parts = reference_leg_degeneracy_structure(rule, homspace);
    assert_eq!(*direct_parts.0, *reconstructed_parts.sector_structure());
    assert_eq!(direct_parts.1, *reconstructed_parts.degeneracy_structure());
    assert_eq!(
        direct_parts.1.required_len().unwrap(),
        reconstructed_parts.required_len().unwrap()
    );
}

#[test]
fn direct_leg_degeneracy_layout_matches_reference_for_supported_rules() {
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mixed_leg =
        |sectors: &[(SectorId, usize)], dual| SectorLeg::new(sectors.iter().copied(), dual);
    let build = |sectors: &[(SectorId, usize)]| {
        FusionTreeHomSpace::new(
            FusionProductSpace::new([mixed_leg(sectors, false), mixed_leg(sectors, true)]),
            FusionProductSpace::new([mixed_leg(sectors, true), mixed_leg(sectors, false)]),
        )
    };

    let u1_hom = build(&[(u1(-2), 2), (u1(1), 1)]);
    assert_direct_leg_degeneracy_structure_matches_reference(&U1FusionRule, &u1_hom);
    assert_coupled_grid_layout_matches_key_reconstruction(&U1FusionRule, &u1_hom);

    let parity_hom = build(&[(SectorId::new(0), 3), (SectorId::new(1), 2)]);
    assert_direct_leg_degeneracy_structure_matches_reference(&FermionParityFusionRule, &parity_hom);
    assert_coupled_grid_layout_matches_key_reconstruction(&FermionParityFusionRule, &parity_hom);

    let su2_hom = build(&[
        (SU2Irrep::from_twice_spin(0).sector_id(), 2),
        (SU2Irrep::from_twice_spin(1).sector_id(), 1),
    ]);
    assert_direct_leg_degeneracy_structure_matches_reference(&SU2FusionRule, &su2_hom);
    assert_coupled_grid_layout_matches_key_reconstruction(&SU2FusionRule, &su2_hom);

    let product_rule =
        symtn_sectors::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule)
            .unwrap();
    let product_hom = build(&[
        (
            product_rule.encode_component_ids(SectorId::new(0), u1(-1)),
            2,
        ),
        (
            product_rule.encode_component_ids(SectorId::new(1), u1(2)),
            1,
        ),
    ]);
    assert_direct_leg_degeneracy_structure_matches_reference(&product_rule, &product_hom);
    assert_coupled_grid_layout_matches_key_reconstruction(&product_rule, &product_hom);

    let scalar = FusionTreeHomSpace::new(FusionProductSpace::new([]), FusionProductSpace::new([]));
    assert_direct_leg_degeneracy_structure_matches_reference(&U1FusionRule, &scalar);
    assert_coupled_grid_layout_matches_key_reconstruction(&U1FusionRule, &scalar);

    let rank_one = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(u1(0), 5)], true)]),
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
    );
    assert_direct_leg_degeneracy_structure_matches_reference(&U1FusionRule, &rank_one);
    assert_coupled_grid_layout_matches_key_reconstruction(&U1FusionRule, &rank_one);
    let rank_one_block = rank_one
        .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
        .unwrap();
    assert_eq!(rank_one_block.required_len().unwrap(), 5);
    assert_eq!(rank_one_block.block(0).unwrap().shape(), &[5]);
    assert_eq!(rank_one_block.block(0).unwrap().strides(), &[1]);
    assert_eq!(rank_one_block.block(0).unwrap().offset(), 0);

    let rank_one_domain = FusionTreeHomSpace::new(
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
        FusionProductSpace::new([SectorLeg::new([(u1(0), 7)], true)]),
    );
    let rank_one_domain_block = rank_one_domain
        .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
        .unwrap();
    assert_eq!(rank_one_domain_block.required_len().unwrap(), 7);
    assert_eq!(rank_one_domain_block.block(0).unwrap().shape(), &[7]);
    assert_eq!(rank_one_domain_block.block(0).unwrap().strides(), &[1]);
    assert_eq!(rank_one_domain_block.block(0).unwrap().offset(), 0);

    let empty_domain = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(u1(-1), 2), (u1(0), 1)], false),
            SectorLeg::new([(u1(0), 3), (u1(1), 1)], false),
        ]),
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
    );
    assert_direct_leg_degeneracy_structure_matches_reference(&U1FusionRule, &empty_domain);
    assert_coupled_grid_layout_matches_key_reconstruction(&U1FusionRule, &empty_domain);
}

#[test]
fn external_sector_lookup_rejects_a_short_sector_tuple() {
    // What: both the all-keys and the unique-key lookup reject a sector tuple
    // whose length is not the hom space's leg count.
    let sectors = [Z2Irrep::EVEN.sector_id()];
    let metadata_error = CoreError::DimensionMismatch {
        expected: 2,
        actual: 1,
    };

    let homspace = FusionTreeHomSpace::from_sectors([(Z2Irrep::EVEN, 1)], [(Z2Irrep::EVEN, 1)]);

    assert_eq!(
        homspace
            .fusion_tree_keys_from_external_sectors(&Z2FusionRule, &sectors)
            .unwrap_err(),
        metadata_error
    );
    assert_eq!(
        homspace
            .unique_fusion_tree_key_from_external_sectors(&Z2FusionRule, &sectors)
            .unwrap_err(),
        metadata_error
    );
}

#[test]
fn sector_leg_dual_shares_the_sector_data_the_dual_fixes() {
    // What: #1403. A leg whose sector -> degeneracy map the rule's dual
    // fixes shares its storage with its dual, as TensorKit's `dual(V)`
    // shares `V.dims`; the dual is otherwise content-identical to the
    // eagerly built leg, including equality and hashing.
    let hash = |leg: &SectorLeg| {
        let mut state = std::collections::hash_map::DefaultHasher::new();
        leg.hash(&mut state);
        state.finish()
    };
    let rule = U1FusionRule;

    let fixed = SectorLeg::new([(u1(-1), 2), (u1(0), 3), (u1(1), 2)], false);
    let dual = fixed.dual(&rule);
    assert!(fixed.shares_sector_data_with(&dual));
    assert!(dual.is_dual());
    let built = SectorLeg::new([(u1(1), 2), (u1(0), 3), (u1(-1), 2)], true);
    assert_eq!(dual, built);
    assert_eq!(hash(&dual), hash(&built));
    assert_ne!(dual, fixed);
    assert_eq!(dual.dual(&rule), fixed);
    assert!(dual.dual(&rule).shares_sector_data_with(&fixed));
    assert_eq!(fixed.try_dual(&rule), Ok(dual.clone()));
    assert!(fixed
        .try_dual(&rule)
        .unwrap()
        .shares_sector_data_with(&fixed));

    // What: a leg whose map the dual moves also shares its storage, with
    // the dual map the hand-computed `q -> -q` gives.
    let moved = SectorLeg::new([(u1(-1), 2), (u1(1), 4)], false);
    let moved_dual = moved.dual(&rule);
    assert!(moved.shares_sector_data_with(&moved_dual));
    assert_eq!(moved_dual.sectors(), &[u1(-1), u1(1)]);
    assert_eq!(moved_dual.degeneracies(), &[4, 2]);
    assert_eq!(moved_dual.degeneracy(u1(1)), Some(2));
    assert!(moved_dual.is_dual());
    assert_eq!(moved_dual.dual(&rule), moved);
    assert!(moved_dual.dual(&rule).shares_sector_data_with(&moved));
}

fn sector_leg_hash(leg: &SectorLeg) -> u64 {
    let mut state = std::collections::hash_map::DefaultHasher::new();
    leg.hash(&mut state);
    state.finish()
}

/// `leg.dual(rule)` equals, hashes and prints like `expected` built
/// eagerly, shares `leg`'s storage, and dualizes back to `leg`.
fn assert_shared_dual<R: FusionRule>(rule: &R, leg: &SectorLeg, expected: &SectorLeg) {
    for _ in 0..2 {
        let dual = leg.dual(rule);
        assert_eq!(&dual, expected);
        assert_eq!(sector_leg_hash(&dual), sector_leg_hash(expected));
        assert_eq!(format!("{dual:?}"), format!("{expected:?}"));
        assert!(dual.shares_sector_data_with(leg));
        let back = dual.dual(rule);
        assert_eq!(&back, leg);
        assert_eq!(sector_leg_hash(&back), sector_leg_hash(leg));
        assert!(back.shares_sector_data_with(leg));
    }
}

#[test]
fn sector_leg_dual_is_shared_for_non_self_dual_self_dual_and_product_rules() {
    // What: #1403. The dual shares storage for every map, and its content
    // is the hand-computed dual map with the flag flipped.
    for is_dual in [false, true] {
        let u1_leg = SectorLeg::new([(u1(-2), 1), (u1(0), 3), (u1(1), 5)], is_dual);
        let u1_expected = SectorLeg::new([(u1(2), 1), (u1(0), 3), (u1(-1), 5)], !is_dual);
        assert_shared_dual(&U1FusionRule, &u1_leg, &u1_expected);
        assert_eq!(u1_leg.try_dual(&U1FusionRule), Ok(u1_expected.clone()));
        assert!(u1_leg
            .try_dual(&U1FusionRule)
            .unwrap()
            .shares_sector_data_with(&u1_leg));

        let su2_leg = SectorLeg::new([(su2(0), 2), (su2(1), 1), (su2(2), 4)], is_dual);
        let su2_expected = SectorLeg::new([(su2(0), 2), (su2(1), 1), (su2(2), 4)], !is_dual);
        assert_shared_dual(&SU2FusionRule, &su2_leg, &su2_expected);

        type Fz2U1 = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
        let product = Fz2U1::try_new(FermionParityFusionRule, U1FusionRule).unwrap();
        let sector = |parity, charge| product.encode_component_ids(parity, u1(charge));
        let product_leg = SectorLeg::new(
            [
                (sector(z2_even(), 0), 2),
                (sector(z2_odd(), 1), 3),
                (sector(z2_even(), -2), 1),
            ],
            is_dual,
        );
        let product_expected = SectorLeg::new(
            [
                (sector(z2_even(), 0), 2),
                (sector(z2_odd(), -1), 3),
                (sector(z2_even(), 2), 1),
            ],
            !is_dual,
        );
        assert_shared_dual(&product, &product_leg, &product_expected);
        assert_eq!(product_leg.try_dual(&product), Ok(product_expected));
    }
}

#[test]
fn sector_leg_dual_under_a_different_rule_ignores_the_shared_map() {
    // What: sector ids are rule-relative. A rule whose dual differs from
    // the rule that filled a leg's dual map gets its own dual, and the
    // shared map keeps serving the filling rule.
    let leg = SectorLeg::new([(u1(-1), 2), (u1(1), 4)], false);
    let u1_dual = leg.dual(&U1FusionRule);
    let identity_dual = leg.dual(&SU2FusionRule);
    assert_eq!(
        identity_dual,
        SectorLeg::new([(u1(-1), 2), (u1(1), 4)], true)
    );
    assert_eq!(
        u1_dual.dual(&SU2FusionRule),
        SectorLeg::new([(u1(-1), 4), (u1(1), 2)], false)
    );
    assert_eq!(leg.dual(&U1FusionRule), u1_dual);
    assert!(leg.dual(&U1FusionRule).shares_sector_data_with(&leg));
    assert_eq!(u1_dual.degeneracies(), &[4, 2]);

    // What: the other fill order.
    let leg = SectorLeg::new([(u1(-1), 2), (u1(1), 4)], false);
    assert_eq!(
        leg.dual(&SU2FusionRule),
        SectorLeg::new([(u1(-1), 2), (u1(1), 4)], true)
    );
    assert_eq!(
        leg.dual(&U1FusionRule),
        SectorLeg::new([(u1(-1), 4), (u1(1), 2)], true)
    );
}

#[test]
fn sector_leg_dual_race_loser_shares_only_an_agreeing_map() {
    // What: the arm a thread takes when another filled the shared dual
    // map between its check and its install. The loser's maps are written
    // out by hand: `q -> -q` moves this leg's map, the identity keeps it.
    let pairs = [(u1(-1), 2), (u1(1), 4)];
    let u1_map = || DualSectorMap {
        images: Box::new([u1(1), u1(-1)]),
        moved: Some(MovedSectorMap {
            sectors: Box::new([u1(-1), u1(1)]),
            degeneracies: Box::new([4, 2]),
            images: Box::new([u1(1), u1(-1)]),
        }),
    };
    let identity_map = || DualSectorMap {
        images: Box::new([u1(-1), u1(1)]),
        moved: None,
    };
    let u1_dual = SectorLeg::new([(u1(-1), 4), (u1(1), 2)], true);
    let identity_dual = SectorLeg::new(pairs, true);

    // Loser agrees with the winner: shares the winner's map.
    let leg = SectorLeg::new(pairs, false);
    leg.dual(&U1FusionRule);
    let dual = leg.install_dual_map(u1_map());
    assert_eq!(dual, u1_dual);
    assert!(dual.shares_sector_data_with(&leg));

    // Loser's moved map, winner's identity map: the loser's own leg.
    let leg = SectorLeg::new(pairs, false);
    leg.dual(&SU2FusionRule);
    let dual = leg.install_dual_map(u1_map());
    assert_eq!(dual, u1_dual);
    assert!(!dual.shares_sector_data_with(&leg));
    assert_eq!(leg.dual(&SU2FusionRule), identity_dual);

    // Loser's identity map, winner's moved map: the source storage with
    // the flag flipped, not the winner's moved map.
    let leg = SectorLeg::new(pairs, false);
    leg.dual(&U1FusionRule);
    let dual = leg.install_dual_map(identity_map());
    assert_eq!(dual, identity_dual);
    assert!(dual.shares_sector_data_with(&leg));
    assert_eq!(leg.dual(&U1FusionRule), u1_dual);
}

#[test]
fn sector_leg_charge_does_not_grow_when_the_dual_map_is_filled() {
    // What: a cache charges a leg when it admits it; filling the shared
    // dual map later, including spilled storage, stays inside that charge.
    let rule = U1FusionRule;
    let leg = SectorLeg::new(
        (0..12).map(|charge| (u1(charge), 1 + charge as usize)),
        false,
    );
    let charge = leg.charged_retained_bytes();
    let dual = leg.dual(&rule);
    assert_eq!(leg.charged_retained_bytes(), charge);
    assert_eq!(dual.charged_retained_bytes(), charge);
    assert_eq!(
        dual,
        SectorLeg::new(
            (0..12).map(|charge| (u1(-charge), 1 + charge as usize)),
            true
        )
    );
    assert_eq!(dual.dual(&rule), leg);
}
