use super::*;

/// Three `{0, 1}` legs per side under the multiplicity-two test rule:
/// coupled sector 1 has 14 trees per side and sector 0 has 6, counting
/// inner lines and vertices, so F_c = T_c per sector (unit degeneracies).
#[test]
fn checked_one_sided_many_tree_publication_reuses_one_index_per_sector() {
    fn run<D: FactorScalar + fmt::Debug>(values: impl Fn(usize) -> D) {
        let rule = TestGenericRule;
        let provider = Arc::new(InfallibleGeneric::new(&rule));
        let leg = || SectorLeg::new([(SectorId::new(0), 1), (SectorId::new(1), 1)], false);
        let homspace = FusionTreeHomSpace::new(
            FusionProductSpace::new([leg(), leg(), leg()]),
            FusionProductSpace::new([leg(), leg(), leg()]),
        );
        let space = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
            Arc::clone(&provider),
            homspace.clone(),
        )
        .unwrap();
        let zeros = vec![D::zero(); space.space().required_len().unwrap()];
        let fresh = || sector_matricizations(space.space().structure(), &zeros, 3).unwrap();
        let matrices = fresh();
        assert_eq!(
            matrices
                .iter()
                .map(|m| (m.sector.id(), m.row_trees.len(), m.col_trees.len()))
                .collect::<Vec<_>>(),
            [(0, 6, 6), (1, 14, 14)]
        );
        for side in [FactorSide::Left, FactorSide::Right] {
            let dimensions = matrices
                .iter()
                .map(|m| (m.sector, source_extent(m, side)))
                .collect::<BTreeMap<_, _>>();
            let build = |matrices: &[SectorMatricization<D>], pairs: &mut [FactorPair<D>]| {
                let one_sided_rec = crate::probe::record();
                let placement_index_rec = crate::probe::record();
                let factor = build_bound_factor_generic_checked(
                    &provider,
                    &homspace,
                    matrices,
                    pairs,
                    &dimensions,
                    side,
                )
                .unwrap();
                (
                    factor.data().to_vec(),
                    one_sided_rec.one_sided_publication(),
                    placement_index_rec.placement_index(),
                )
            };
            let staged = |matrices: &[SectorMatricization<D>]| {
                staged_one_sided_pairs(matrices, &dimensions, side, side, &values)
            };

            // Canonical transfer: prevalidation alone, one lookup per key,
            // and no plan allocation for the all-populated input.
            let mut pairs = staged(&matrices);
            let (reference, probe, index) = build(&matrices, &mut pairs);
            assert_eq!(probe.canonical_publications, 1);
            assert_eq!(probe.plan_bytes, 0);
            assert_eq!(
                index,
                PlacementIndexProbe {
                    index_builds: 1,
                    indexed_sides: 2,
                    indexed_trees: 20,
                    lookups: 20,
                }
            );

            // Fallback: the single prevalidation table is reused (rather than
            // one table per sector, G_s = 2), so lookups double without a
            // rebuild; 60 hashes instead of the per-sector route's
            // 2 * (6 * 6 + 14 * 14) = 464 key comparisons.
            let mut reversed = fresh();
            reversed.reverse();
            let mut pairs = staged(&reversed);
            let (data, probe, index) = build(&reversed, &mut pairs);
            assert_eq!(probe.fallback_publications, 1);
            assert_eq!(data, reference);
            assert_eq!(
                index,
                PlacementIndexProbe {
                    index_builds: 1,
                    indexed_sides: 2,
                    indexed_trees: 20,
                    lookups: 40,
                }
            );
            assert!(index.indexed_trees + index.lookups < 2 * (6 * 6 + 14 * 14));

            // Identity-only sector after (keep 0) and before (keep 1) the
            // populated sector: only the populated sector is indexed.
            let lens = matrices
                .iter()
                .map(|m| source_extent(m, side) * dimensions[&m.sector])
                .collect::<Vec<_>>();
            for keep in [0usize, 1] {
                let kept = fresh()
                    .into_iter()
                    .filter(|m| m.sector.id() == keep)
                    .collect::<Vec<_>>();
                let mut pairs = staged(&kept);
                let selected = pairs
                    .iter()
                    .map(|pair| selected_of(pair, side).clone())
                    .collect::<Vec<_>>();
                let (data, probe, index) = build(&kept, &mut pairs);
                assert_eq!(
                    (probe.canonical_publications, probe.fallback_publications),
                    (1, 0)
                );
                assert_eq!(
                    index,
                    PlacementIndexProbe {
                        index_builds: 1,
                        indexed_sides: 1,
                        indexed_trees: kept[0].row_trees.len(),
                        lookups: kept[0].row_trees.len(),
                    }
                );
                let (first, second) = data.split_at(lens[0]);
                let (reference_first, reference_second) = reference.split_at(lens[0]);
                assert_eq!(second.len(), lens[1]);
                if keep == 0 {
                    assert_eq!(first, reference_first);
                } else {
                    assert_eq!(second, reference_second);
                }
                let checked_factor = build_bound_factor_generic_checked(
                    &provider,
                    &homspace,
                    &kept,
                    &mut staged(&kept),
                    &dimensions,
                    side,
                )
                .unwrap();
                assert_literal_one_sided_layout(
                    checked_factor.space().space().structure(),
                    &data,
                    &kept,
                    &selected,
                    &dimensions,
                    side,
                    side,
                );
            }
        }
    }
    run(|k| k as f64 + 0.25);
    run(|k| Complex64::new(k as f64 + 0.25, 0.5 - k as f64));
}

#[test]
fn checked_one_sided_placement_preserves_both_sides_and_tree_orders() {
    let rule = TestGenericRule;
    let provider = Arc::new(InfallibleGeneric::new(&rule));
    let dimensions = BTreeMap::from([(SectorId::new(1), 2)]);

    for reverse in [false, true] {
        let (homspace, matrix, pair) = vertex_tree_factor_fixture(reverse);
        for side in [FactorSide::Left, FactorSide::Right] {
            let (_, _, mut staged) = vertex_tree_factor_fixture(reverse);
            let factor = build_bound_factor_generic_checked(
                &provider,
                &homspace,
                std::slice::from_ref(&matrix),
                std::slice::from_mut(&mut staged),
                &dimensions,
                side,
            )
            .unwrap();
            let expected = match (side, reverse) {
                (FactorSide::Left, false) => pair.left.clone(),
                (FactorSide::Right, false) => pair.right.clone(),
                (FactorSide::Left, true) => vec![
                    pair.left[2],
                    pair.left[3],
                    pair.left[0],
                    pair.left[1],
                    pair.left[6],
                    pair.left[7],
                    pair.left[4],
                    pair.left[5],
                ],
                (FactorSide::Right, true) => vec![
                    pair.right[6],
                    pair.right[7],
                    pair.right[8],
                    pair.right[9],
                    pair.right[10],
                    pair.right[11],
                    pair.right[0],
                    pair.right[1],
                    pair.right[2],
                    pair.right[3],
                    pair.right[4],
                    pair.right[5],
                ],
            };
            assert_eq!(factor.data(), expected);
            assert!(Arc::ptr_eq(factor.space().provider_arc(), &provider));
            let selected_vertices = (0..factor.space().space().structure().block_count())
                .map(|index| factor.space().space().structure().block(index).unwrap())
                .map(|block| match block.key() {
                    BlockKey::FusionTree(key) => match side {
                        FactorSide::Left => key.codomain_tree().vertices()[0].get(),
                        FactorSide::Right => key.domain_tree().vertices()[0].get(),
                    },
                    _ => unreachable!("checked factor has fusion-tree keys"),
                })
                .collect::<Vec<_>>();
            assert_eq!(selected_vertices, [1, 2]);
        }
    }
}

#[test]
fn checked_one_sided_reports_missing_pairs_and_full_trees() {
    let rule = TestGenericRule;
    let provider = Arc::new(InfallibleGeneric::new(&rule));
    let (homspace, matrix, _) = vertex_tree_factor_fixture(false);
    let dimensions = BTreeMap::from([(SectorId::new(1), 2)]);
    for side in [FactorSide::Left, FactorSide::Right] {
        let placement_index_rec = crate::probe::record();
        let error = build_bound_factor_generic_checked::<_, Complex64, _>(
            &provider,
            &homspace,
            std::slice::from_ref(&matrix),
            &mut [],
            &dimensions,
            side,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            CheckedGenericFactorPlanError::Operation(OperationError::SpaceMismatch {
                message: "factor rank absent for a populated source sector"
            })
        ));
        // Prevalidation indexed the sole sector and looked up both keys;
        // the fallback rejected the missing pair before its own lookup.
        assert_eq!(
            placement_index_rec.placement_index(),
            PlacementIndexProbe {
                index_builds: 1,
                indexed_sides: 1,
                indexed_trees: 2,
                lookups: 2,
            }
        );
    }

    for side in [FactorSide::Left, FactorSide::Right] {
        let (_, mut wrong, mut pair) = vertex_tree_factor_fixture(false);
        match side {
            FactorSide::Left => wrong.row_trees[0].0 = wrong.row_trees[1].0.clone(),
            FactorSide::Right => wrong.col_trees[0].0 = wrong.col_trees[1].0.clone(),
        }
        let error = build_bound_factor_generic_checked(
            &provider,
            &homspace,
            std::slice::from_ref(&wrong),
            std::slice::from_mut(&mut pair),
            &dimensions,
            side,
        )
        .unwrap_err();
        let expected = match side {
            FactorSide::Left => "factor codomain tree absent from the source matricization",
            FactorSide::Right => "factor domain tree absent from the source matricization",
        };
        assert!(matches!(
            error,
            CheckedGenericFactorPlanError::Operation(
                OperationError::SpaceMismatch { message }
            ) if message == expected
        ));
    }
}

#[test]
fn checked_one_sided_canonical_transfer_across_sectors_and_extra_pairs() {
    let rule = TestGenericRule;
    let provider = Arc::new(InfallibleGeneric::new(&rule));
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(even, 2), (odd, 1)], false),
            SectorLeg::new([(even, 1), (odd, 3)], false),
        ]),
        FusionProductSpace::new([
            SectorLeg::new([(even, 1), (odd, 2)], false),
            SectorLeg::new([(even, 2), (odd, 1)], false),
        ]),
    );
    let space = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::clone(&provider),
        homspace.clone(),
    )
    .unwrap();
    let zeros = vec![Complex64::new(0.0, 0.0); space.space().required_len().unwrap()];
    let matrices = sector_matricizations(space.space().structure(), &zeros, 2).unwrap();
    assert_eq!(matrices.len(), 2);
    let values = |k: usize| Complex64::new(0.5 * k as f64, 2.0 - k as f64);

    for side in [FactorSide::Left, FactorSide::Right] {
        let dimensions = matrices
            .iter()
            .map(|matrix| (matrix.sector, source_extent(matrix, side)))
            .collect::<BTreeMap<_, _>>();
        let mut pairs = staged_one_sided_pairs(&matrices, &dimensions, side, side, &values);
        let selected = pairs
            .iter()
            .map(|pair| selected_of(pair, side).clone())
            .collect::<Vec<_>>();
        let opposite = pairs
            .iter()
            .map(|pair| opposite_of(pair, side).clone())
            .collect::<Vec<_>>();
        let one_sided_rec = crate::probe::record();
        let factor = build_bound_factor_generic_checked(
            &provider,
            &homspace,
            &matrices,
            &mut pairs,
            &dimensions,
            side,
        )
        .unwrap();
        let probe = one_sided_rec.one_sided_publication();
        assert_eq!(
            (probe.canonical_publications, probe.fallback_publications),
            (1, 0)
        );
        assert_eq!(probe.appended_elements, selected[1].len());
        // All-populated input allocates no plan, as the boolean proof did.
        assert_eq!(probe.plan_bytes, 0);
        assert!(Arc::ptr_eq(factor.space().provider_arc(), &provider));
        assert_literal_one_sided_layout(
            factor.space().space().structure(),
            factor.data(),
            &matrices,
            &selected,
            &dimensions,
            side,
            side,
        );
        for (index, pair) in pairs.iter().enumerate() {
            assert!(selected_of(pair, side).is_empty());
            assert_eq!(opposite_of(pair, side), &opposite[index]);
        }

        // An extra pair is rejected as in the multiplicity-free mode
        // (mode divergences take MF semantics). The bound space
        // is already enumerated and committed when the error fires, but no
        // factor data is published. Unreachable from the public API: pairs
        // come from the same matricization.
        let mut pairs = staged_one_sided_pairs(&matrices, &dimensions, side, side, &values);
        pairs.push(FactorPair {
            sector: SectorId::new(7),
            kept: 1,
            left: vec![Complex64::new(1.0, 1.0)],
            left_rows: 1,
            right: vec![Complex64::new(1.0, 1.0)],
            right_leading: 1,
        });
        let one_sided_rec = crate::probe::record();
        let error = build_bound_factor_generic_checked(
            &provider,
            &homspace,
            &matrices,
            &mut pairs,
            &dimensions,
            side,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            CheckedGenericFactorPlanError::Operation(OperationError::SpaceMismatch {
                message: "factor sector absent from the source tensor"
            })
        ));
        let probe = one_sided_rec.one_sided_publication();
        assert_eq!(
            (probe.canonical_publications, probe.fallback_publications),
            (0, 0)
        );
        assert_buffers_intact(&pairs, side);
    }
}

#[test]
fn checked_one_sided_vertex_trees_transfer_sole_owner_or_fall_back() {
    let rule = TestGenericRule;
    let provider = Arc::new(InfallibleGeneric::new(&rule));
    let dimensions = BTreeMap::from([(SectorId::new(1), 2)]);
    for reverse in [false, true] {
        for side in [FactorSide::Left, FactorSide::Right] {
            let (homspace, matrix, mut pair) = vertex_tree_factor_fixture(reverse);
            let (_, _, reference) = vertex_tree_factor_fixture(reverse);
            let selected_ptr = selected_of(&pair, side).as_ptr();
            let one_sided_rec = crate::probe::record();
            let factor = build_bound_factor_generic_checked(
                &provider,
                &homspace,
                std::slice::from_ref(&matrix),
                std::slice::from_mut(&mut pair),
                &dimensions,
                side,
            )
            .unwrap();
            let probe = one_sided_rec.one_sided_publication();
            assert_eq!(opposite_of(&pair, side), opposite_of(&reference, side));
            if reverse {
                assert_eq!(
                    (probe.canonical_publications, probe.fallback_publications),
                    (0, 1)
                );
                assert_eq!(selected_of(&pair, side), selected_of(&reference, side));
            } else {
                assert_eq!(
                    (
                        probe.canonical_publications,
                        probe.fallback_publications,
                        probe.owner_reused,
                        probe.appended_elements,
                    ),
                    (1, 0, 1, 0)
                );
                assert!(std::ptr::eq(selected_ptr, factor.data().as_ptr()));
                assert!(selected_of(&pair, side).is_empty());
                assert_eq!(factor.data(), selected_of(&reference, side));
            }
        }
    }
}

/// The output HomSpace exactly as `build_bound_factor_generic_checked`
/// derives it, built independently of the returned factor.
fn one_sided_output_hom(
    homspace: &FusionTreeHomSpace,
    dimensions: &BTreeMap<SectorId, usize>,
    side: FactorSide,
) -> FusionTreeHomSpace {
    let bond = SectorLeg::new(
        dimensions.iter().map(|(&sector, &dim)| (sector, dim)),
        false,
    );
    match side {
        FactorSide::Left => {
            FusionTreeHomSpace::new(homspace.codomain().clone(), FusionProductSpace::new([bond]))
        }
        FactorSide::Right => {
            FusionTreeHomSpace::new(FusionProductSpace::new([bond]), homspace.domain().clone())
        }
    }
}

/// Two-sector checked geometry with `keep` selecting which sectors carry a
/// matricization; the others become identity-only output sectors.
type TwoSectorCheckedFixture = (
    FusionTreeHomSpace,
    Vec<SectorMatricization<Complex64>>,
    BTreeMap<SectorId, usize>,
    BTreeMap<SectorId, usize>,
);

fn two_sector_checked_fixture(keep: &dyn Fn(SectorId) -> bool) -> TwoSectorCheckedFixture {
    let rule = TestGenericRule;
    let provider = Arc::new(InfallibleGeneric::new(&rule));
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(even, 2), (odd, 1)], false),
            SectorLeg::new([(even, 1), (odd, 3)], false),
        ]),
        FusionProductSpace::new([
            SectorLeg::new([(even, 1), (odd, 2)], false),
            SectorLeg::new([(even, 2), (odd, 1)], false),
        ]),
    );
    let space =
        BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(provider, homspace.clone())
            .unwrap();
    let zeros = vec![Complex64::new(0.0, 0.0); space.space().required_len().unwrap()];
    let matrices = sector_matricizations(space.space().structure(), &zeros, 2).unwrap();
    assert_eq!(matrices.len(), 2);
    let row_dimensions = matrices
        .iter()
        .map(|matrix| (matrix.sector, matrix.rows))
        .collect();
    let col_dimensions = matrices
        .iter()
        .map(|matrix| (matrix.sector, matrix.cols))
        .collect();
    let matrices = matrices
        .into_iter()
        .filter(|matrix| keep(matrix.sector))
        .collect();
    (homspace, matrices, row_dimensions, col_dimensions)
}

#[test]
#[allow(clippy::arc_with_non_send_sync)] // The checked API needs Arc identity; the recorder is a single-threaded RefCell log.
fn checked_one_sided_enumerates_the_output_layout_once() {
    // What: one publication issues exactly the provider queries of a
    // single layout enumeration (not keys and then a bound space, which
    // would issue that identical sequence twice), and the committed
    // block order equals the enumerated key order.
    let dimensions = BTreeMap::from([(SectorId::new(1), 2)]);
    for reverse in [false, true] {
        for side in [FactorSide::Left, FactorSide::Right] {
            let (homspace, matrix, mut pair) = vertex_tree_factor_fixture(reverse);
            let recorder = Arc::new(RecordingGeneric::new());
            let factor = build_bound_factor_generic_checked(
                &recorder,
                &homspace,
                std::slice::from_ref(&matrix),
                std::slice::from_mut(&mut pair),
                &dimensions,
                side,
            )
            .unwrap();
            let once = recorder.log();
            // Vertex fixture: the one-leg bond side folds once, the
            // two-leg side queries channels and the multiplicity of its
            // single vertex.
            assert_eq!(once.len(), 3, "{once:?}");

            let former = Arc::new(RecordingGeneric::new());
            let output_hom = one_sided_output_hom(&homspace, &dimensions, side);
            let keys = output_hom
                .fusion_tree_keys_generic_checked(former.as_ref())
                .unwrap();
            let expected = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
                Arc::clone(&former),
                output_hom,
            )
            .unwrap();
            let twice = former.log();
            assert_eq!(twice.len(), 2 * once.len());
            assert_eq!(&twice[..once.len()], once.as_slice());
            assert_eq!(&twice[once.len()..], once.as_slice());

            let block_keys = block_rows(factor.space().space().structure())
                .into_iter()
                .map(|(key, ..)| key)
                .collect::<Vec<_>>();
            let key_order = keys.into_iter().map(BlockKey::from).collect::<Vec<_>>();
            assert_eq!(block_keys, key_order);
            assert_eq!(
                block_rows(factor.space().space().structure()),
                block_rows(expected.space().structure())
            );
            assert!(Arc::ptr_eq(factor.space().provider_arc(), &recorder));
        }
    }
}

#[test]
#[allow(clippy::arc_with_non_send_sync)] // The checked API needs Arc identity; the recorder is a single-threaded RefCell log.
fn checked_one_sided_placement_error_precedes_bound_space() {
    // What: a populated output key whose tree the matricization lacks
    // fails after the single enumeration and before any bound space or
    // publication exists; the staged factor buffers stay untouched.
    let dimensions = BTreeMap::from([(SectorId::new(1), 2)]);
    for side in [FactorSide::Left, FactorSide::Right] {
        let (homspace, mut matrix, mut pair) = vertex_tree_factor_fixture(false);
        let (_, _, reference) = vertex_tree_factor_fixture(false);
        match side {
            FactorSide::Left => matrix.row_trees.truncate(1),
            FactorSide::Right => matrix.col_trees.truncate(1),
        }
        let recorder = Arc::new(RecordingGeneric::new());
        let one_sided_rec = crate::probe::record();
        let placement_index_rec = crate::probe::record();
        let error = build_bound_factor_generic_checked(
            &recorder,
            &homspace,
            std::slice::from_ref(&matrix),
            std::slice::from_mut(&mut pair),
            &dimensions,
            side,
        )
        .unwrap_err();
        let expected_message = match side {
            FactorSide::Left => "factor codomain tree absent from the source matricization",
            FactorSide::Right => "factor domain tree absent from the source matricization",
        };
        assert!(matches!(
            error,
            CheckedGenericFactorPlanError::Operation(
                OperationError::SpaceMismatch { message }
            ) if message == expected_message
        ));
        assert_eq!(recorder.log().len(), 3);
        let probe = one_sided_rec.one_sided_publication();
        assert_eq!(
            (probe.canonical_publications, probe.fallback_publications),
            (0, 0)
        );
        let index = placement_index_rec.placement_index();
        assert_eq!(
            (index.index_builds, index.indexed_sides, index.indexed_trees),
            (1, 1, 1)
        );
        assert_eq!(index.lookups, 2);
        assert_eq!(pair.left, reference.left);
        assert_eq!(pair.right, reference.right);
    }
}

#[test]
fn checked_one_sided_structure_matches_two_enumeration_construction() {
    // What: with identity-only sectors before or after the populated one,
    // the committed structure, required length, data length and provider
    // binding equal those of a space built by a separate key-then-space
    // enumeration, for both sides and both scalar types.
    let rule = TestGenericRule;
    let provider = Arc::new(InfallibleGeneric::new(&rule));
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let keeps: [&dyn Fn(SectorId) -> bool; 3] =
        [&|_| true, &|sector| sector == odd, &|sector| sector == even];
    for keep in keeps {
        let (homspace, matrices, row_dimensions, col_dimensions) = two_sector_checked_fixture(keep);
        for side in [FactorSide::Left, FactorSide::Right] {
            let dimensions = match side {
                FactorSide::Left => &row_dimensions,
                FactorSide::Right => &col_dimensions,
            };
            let expected = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
                Arc::clone(&provider),
                one_sided_output_hom(&homspace, dimensions, side),
            )
            .unwrap();
            let expected_len = expected.space().required_len().unwrap();

            let complex = |k: usize| Complex64::new(0.5 * k as f64, 2.0 - k as f64);
            let mut pairs = staged_one_sided_pairs(&matrices, dimensions, side, side, &complex);
            let factor = build_bound_factor_generic_checked(
                &provider, &homspace, &matrices, &mut pairs, dimensions, side,
            )
            .unwrap();
            assert_eq!(
                block_rows(factor.space().space().structure()),
                block_rows(expected.space().structure())
            );
            assert_eq!(factor.space().space().required_len().unwrap(), expected_len);
            assert_eq!(factor.data().len(), expected_len);
            assert_eq!(
                factor.space().space().homspace(),
                expected.space().homspace()
            );
            assert!(Arc::ptr_eq(factor.space().provider_arc(), &provider));

            let real_matrices = matrices
                .iter()
                .map(|matrix| SectorMatricization {
                    sector: matrix.sector,
                    rows: matrix.rows,
                    cols: matrix.cols,
                    row_trees: matrix.row_trees.clone(),
                    col_trees: matrix.col_trees.clone(),
                    data: Vec::<f64>::new(),
                })
                .collect::<Vec<_>>();
            let real = |k: usize| 0.25 * k as f64;
            let mut pairs = staged_one_sided_pairs(&real_matrices, dimensions, side, side, &real);
            let factor = build_bound_factor_generic_checked(
                &provider,
                &homspace,
                &real_matrices,
                &mut pairs,
                dimensions,
                side,
            )
            .unwrap();
            assert_eq!(
                block_rows(factor.space().space().structure()),
                block_rows(expected.space().structure())
            );
            assert_eq!(factor.data().len(), expected_len);
        }
    }
}
