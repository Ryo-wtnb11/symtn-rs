use super::*;

#[derive(Clone, Debug)]
struct ProbeTreeKey(usize);

static PROBE_TREE_KEY_EQ_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

static PROBE_TREE_KEY_HASH_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

impl PartialEq for ProbeTreeKey {
    fn eq(&self, other: &Self) -> bool {
        PROBE_TREE_KEY_EQ_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.0 == other.0
    }
}

impl Eq for ProbeTreeKey {}

impl std::hash::Hash for ProbeTreeKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        PROBE_TREE_KEY_HASH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::hash::Hash::hash(&self.0, state);
    }
}

#[test]
fn fusion_term_accumulator_keeps_singleton_path_and_hashes_multi_terms() {
    PROBE_TREE_KEY_EQ_CALLS.store(0, std::sync::atomic::Ordering::Relaxed);
    PROBE_TREE_KEY_HASH_CALLS.store(0, std::sync::atomic::Ordering::Relaxed);
    let mut singleton = FusionTermAccumulator::new();
    singleton.push(ProbeTreeKey(7), 3usize);
    let singleton_terms = singleton.into_vec();
    assert_eq!(singleton_terms.len(), 1);
    let (singleton_key, singleton_coefficient) = &singleton_terms[0];
    assert_eq!(singleton_key.0, 7);
    assert_eq!(*singleton_coefficient, 3);
    assert_eq!(
        PROBE_TREE_KEY_HASH_CALLS.load(std::sync::atomic::Ordering::Relaxed),
        0
    );

    const DISTINCT: usize = 512;
    const ROUNDS: usize = 4;
    PROBE_TREE_KEY_EQ_CALLS.store(0, std::sync::atomic::Ordering::Relaxed);
    PROBE_TREE_KEY_HASH_CALLS.store(0, std::sync::atomic::Ordering::Relaxed);
    let mut accumulator = FusionTermAccumulator::new();
    for _ in 0..ROUNDS {
        for key in 0..DISTINCT {
            accumulator.push(ProbeTreeKey(key), 1usize);
        }
    }
    let terms = accumulator.into_vec();
    assert_eq!(terms.len(), DISTINCT);
    for (index, (key, coefficient)) in terms.iter().enumerate() {
        assert_eq!(key.0, index);
        assert_eq!(*coefficient, ROUNDS);
    }
    let eq_calls = PROBE_TREE_KEY_EQ_CALLS.load(std::sync::atomic::Ordering::Relaxed);
    assert!(
        eq_calls < DISTINCT * ROUNDS * 8,
        "HashMap-backed accumulation should stay linear; saw {eq_calls} equality checks"
    );
    assert!(
        PROBE_TREE_KEY_HASH_CALLS.load(std::sync::atomic::Ordering::Relaxed) > DISTINCT,
        "multi-term accumulation should use the hash path"
    );
}

#[test]
fn multiplicity_free_su2_braid_expands_innerline_channels() {
    let rule = SU2FusionRule;
    let tree = FusionTreeKey::try_from_sector_ids(
        [1, 1, 1, 1],
        0,
        [false, false, false, false],
        [0, 1],
        [1, 1, 1],
    )
    .unwrap();

    let braided = multiplicity_free_braid_tree(&rule, &tree, &[0, 2, 1, 3], &[0, 1, 2, 3]).unwrap();

    assert_eq!(braided.len(), 2);
    assert_eq!(braided[0].0.uncoupled(), &[SectorId::new(1); 4]);
    assert_eq!(
        braided[0].0.innerlines(),
        &[SectorId::new(0), SectorId::new(1)]
    );
    assert!((braided[0].1 - 0.5).abs() < 1.0e-12);
    assert_eq!(
        braided[1].0.innerlines(),
        &[SectorId::new(2), SectorId::new(1)]
    );
    assert!((braided[1].1 - 0.866_025_403_784_438_6).abs() < 1.0e-12);
}

#[test]
fn bendright_preserves_local_error_precedence_before_missing_duality() {
    // What: local coupled/innerline errors remain observable before a
    // missing final duality flag when several malformed fields coexist.
    let missing_innerline = FusionTreePairKey::pair(
        FusionTreeKey::try_from_sector_ids([1, 1, 1], 1, [false, false], [], [1, 1]).unwrap(),
        FusionTreeKey::try_from_sector_ids([], 0, [], [], []).unwrap(),
    );
    assert_eq!(
        multiplicity_free_bendright_tree_pair(&SU2FusionRule, &missing_innerline).unwrap_err(),
        CoreError::MalformedFusionTree {
            message: "bendright requires the last codomain innerline",
        }
    );

    let mismatched_coupled = FusionTreePairKey::pair(
        FusionTreeKey::try_from_sector_ids([1, 1], 0, [false], [], [1]).unwrap(),
        FusionTreeKey::try_from_sector_ids([1], 1, [false], [], []).unwrap(),
    );
    assert_eq!(
        multiplicity_free_bendright_tree_pair(&SU2FusionRule, &mismatched_coupled).unwrap_err(),
        CoreError::MalformedFusionTree {
            message: "fusion tree pair requires matching coupled sectors",
        }
    );
}

#[test]
fn fusion_tree_group_key_records_external_sector_tuples_and_duality() {
    let group = FusionTreeGroupKey::from_sector_ids([2, 3], [5], [false, true], [true]);

    assert_eq!(
        group.codomain_uncoupled(),
        &[SectorId::new(2), SectorId::new(3)]
    );
    assert_eq!(group.domain_uncoupled(), &[SectorId::new(5)]);
    assert_eq!(group.codomain_is_dual(), &[false, true]);
    assert_eq!(group.domain_is_dual(), &[true]);

    let same = FusionTreeGroupKey::new(
        [SectorId::new(2), SectorId::new(3)],
        [SectorId::new(5)],
        [false, true],
        [true],
    );
    assert_eq!(group, same);
}

#[test]
fn selected_leg_tuple_visitor_is_fallible_and_restartable() {
    // What: an early visitor error stops traversal without corrupting the
    // reusable scratch used by a later traversal.
    let space = FusionProductSpace::new([
        SectorLeg::new([(SectorId::new(0), 1), (SectorId::new(1), 1)], false),
        SectorLeg::new([(SectorId::new(2), 1), (SectorId::new(3), 1)], true),
    ]);
    let expected = materialized_leg_tuple_oracle(&space)
        .into_iter()
        .map(|tuple| {
            tuple
                .into_iter()
                .map(|leg| (leg.sector(), leg.is_dual()))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    let mut partial = Vec::new();
    let err = space
        .try_visit_selected_leg_tuples(&mut |tuple| {
            partial.push(
                tuple
                    .iter()
                    .map(|leg| (leg.sector(), leg.is_dual()))
                    .collect::<Vec<_>>(),
            );
            if partial.len() == 2 {
                Err("stop")
            } else {
                Ok(())
            }
        })
        .unwrap_err();
    assert_eq!(err, "stop");
    assert_eq!(partial, expected[..2]);

    let mut restarted = Vec::new();
    space
        .try_visit_selected_leg_tuples::<(), _>(&mut |tuple| {
            restarted.push(
                tuple
                    .iter()
                    .map(|leg| (leg.sector(), leg.is_dual()))
                    .collect::<Vec<_>>(),
            );
            Ok(())
        })
        .unwrap();
    assert_eq!(restarted, expected);
}

fn assert_literal_binary_choice_sector_grids<R>(
    rule: &R,
    vacuum: SectorId,
    external: SectorId,
    mut groups: Vec<(SectorId, Vec<[SectorId; 2]>)>,
) where
    R: MultiplicityFreeFusionRule,
{
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(vacuum, 2), (external, 3)], false),
            SectorLeg::new([(vacuum, 5), (external, 7)], true),
        ]),
        FusionProductSpace::new([
            SectorLeg::new([(vacuum, 11), (external, 13)], true),
            SectorLeg::new([(vacuum, 17), (external, 19)], false),
        ]),
    );

    groups.sort_by_key(|(coupled, _)| *coupled);
    let layout = homspace.fusion_tree_layout_data_uncached(rule);
    assert_eq!(layout.sectors.len(), groups.len());
    let expected_key_count = groups
        .iter()
        .map(|(_, trees)| trees.len() * trees.len())
        .sum::<usize>();
    assert_eq!(layout.keys.len(), expected_key_count);

    let structure = homspace
        .coupled_subblock_structure_from_leg_degeneracies(rule)
        .unwrap();
    let regions = structure.coupled_sector_regions(2).unwrap().unwrap();
    assert_eq!(regions.len(), groups.len());
    let mut block_index = 0usize;
    let mut sector_offset = 0usize;
    for (sector_index, ((expected_coupled, trees), sector)) in
        groups.iter().zip(&layout.sectors).enumerate()
    {
        assert_eq!(
            (sector.start, sector.row_count, sector.col_count),
            (block_index, trees.len(), trees.len())
        );

        let row_shapes = trees
            .iter()
            .map(|tree| {
                [
                    if tree[0] == vacuum { 2 } else { 3 },
                    if tree[1] == vacuum { 5 } else { 7 },
                ]
            })
            .collect::<Vec<_>>();
        let col_shapes = trees
            .iter()
            .map(|tree| {
                [
                    if tree[0] == vacuum { 11 } else { 13 },
                    if tree[1] == vacuum { 17 } else { 19 },
                ]
            })
            .collect::<Vec<_>>();
        let row_dims = row_shapes
            .iter()
            .map(|shape| shape[0] * shape[1])
            .collect::<Vec<_>>();
        let col_dims = col_shapes
            .iter()
            .map(|shape| shape[0] * shape[1])
            .collect::<Vec<_>>();
        let matrix_rows = row_dims.iter().sum::<usize>();
        let matrix_cols = col_dims.iter().sum::<usize>();
        let mut col_offset = 0usize;
        for (col, domain_tree) in trees.iter().enumerate() {
            let mut row_offset = 0usize;
            for (row, codomain_tree) in trees.iter().enumerate() {
                let key = &layout.keys[block_index];
                assert_eq!(key.codomain_uncoupled(), codomain_tree);
                assert_eq!(key.domain_uncoupled(), domain_tree);
                assert_eq!(key.codomain_is_dual(), &[false, true]);
                assert_eq!(key.domain_is_dual(), &[true, false]);
                assert_eq!(key.coupled(), *expected_coupled);

                let block = structure.block(block_index).unwrap();
                assert_eq!(
                    block.shape(),
                    &[
                        row_shapes[row][0],
                        row_shapes[row][1],
                        col_shapes[col][0],
                        col_shapes[col][1],
                    ]
                );
                assert_eq!(
                    block.strides(),
                    &[
                        1,
                        row_shapes[row][0],
                        matrix_rows,
                        matrix_rows * col_shapes[col][0],
                    ]
                );
                assert_eq!(
                    block.offset(),
                    sector_offset + row_offset + matrix_rows * col_offset
                );
                row_offset += row_dims[row];
                block_index += 1;
            }
            col_offset += col_dims[col];
        }

        let sector_end = sector_offset + matrix_rows * matrix_cols;
        assert_eq!(regions[sector_index].coupled(), *expected_coupled);
        assert_eq!(regions[sector_index].rows(), matrix_rows);
        assert_eq!(regions[sector_index].cols(), matrix_cols);
        assert_eq!(regions[sector_index].range(), sector_offset..sector_end);
        sector_offset = sector_end;
    }
    assert_eq!(block_index, expected_key_count);
    assert_eq!(structure.required_len().unwrap(), sector_offset);
}

#[test]
fn canonical_coupled_grid_has_literal_nonabelian_and_product_metadata() {
    assert_literal_binary_choice_sector_grids(
        &FermionParityFusionRule,
        z2_even(),
        z2_odd(),
        vec![
            (
                z2_even(),
                vec![[z2_even(), z2_even()], [z2_odd(), z2_odd()]],
            ),
            (z2_odd(), vec![[z2_odd(), z2_even()], [z2_even(), z2_odd()]]),
        ],
    );
    assert_literal_binary_choice_sector_grids(
        &SU2FusionRule,
        su2(0),
        su2(1),
        vec![
            (su2(0), vec![[su2(0), su2(0)], [su2(1), su2(1)]]),
            (su2(1), vec![[su2(1), su2(0)], [su2(0), su2(1)]]),
            (su2(2), vec![[su2(1), su2(1)]]),
        ],
    );

    type Fz2U1Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
    type Fz2U1Layout = ProductSectorLayout<Fz2SectorLayout, U1SectorLayout>;
    type Fz2U1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule, Fz2U1Codec>;
    type TripleCodec = PackedProductCodec<Fz2U1Layout, Su2SectorLayout>;
    type TripleRule = ProductFusionRule<Fz2U1Rule, SU2FusionRule, TripleCodec>;

    let triple_rule = TripleRule::new(
        Fz2U1Rule::new(FermionParityFusionRule, U1FusionRule),
        SU2FusionRule,
    );
    let vacuum = TripleCodec::encode(Fz2U1Codec::encode(z2_even(), u1(0)), su2(0));
    let external = TripleCodec::encode(Fz2U1Codec::encode(z2_odd(), u1(0)), su2(1));
    let second_channel = TripleCodec::encode(Fz2U1Codec::encode(z2_even(), u1(0)), su2(2));
    assert_literal_binary_choice_sector_grids(
        &triple_rule,
        vacuum,
        external,
        vec![
            (vacuum, vec![[vacuum, vacuum], [external, external]]),
            (external, vec![[external, vacuum], [vacuum, external]]),
            (second_channel, vec![[external, external]]),
        ],
    );
}

#[test]
fn generic_tree_cannot_enter_multiplicity_free_projection() {
    // What: a genuine multiplicity-two tree is rejected
    // before the compact path can erase its vertex identity.
    let rule = UnitaryToyOmRule;
    let a = SectorId::new(UnitaryToyOmRule::A);
    let coupled = SectorId::new(UnitaryToyOmRule::C);
    let tree = FusionTreeKey::try_new_for_rule(
        &rule,
        [a, a],
        coupled,
        [false, false],
        [],
        [MultiplicityIndex::new(2).unwrap()],
    )
    .unwrap();

    let error = match project_multiplicity_free_tree(&rule, &tree) {
        Ok(_) => panic!("Generic tree entered multiplicity-free projection"),
        Err(error) => error,
    };
    assert_eq!(
        error,
        CoreError::UnsupportedFusionStyle {
            expected: FusionStyleKind::Simple,
            actual: FusionStyleKind::Generic,
        }
    );

    // A proof is an indexed view of the exact slice it checked. Creating
    // another key later cannot extend that authority.
    let checked = FusionTreeKey::try_from_sector_ids([1, 1], 0, [false; 2], [], [1]).unwrap();
    let separate = FusionTreeKey::try_from_sector_ids([1, 1], 0, [false; 2], [], [2]).unwrap();
    let projection =
        MultiplicityFreeTreeProjection::checked(&SU2FusionRule, std::slice::from_ref(&checked))
            .unwrap();
    assert!(projection.tree_at(0).is_some());
    assert!(projection.tree_at(1).is_none());
    assert_eq!(separate.vertices(), &[MultiplicityIndex::new(2).unwrap()]);
}

#[test]
fn compact_block_error_does_not_publish_partial_rows() {
    let rule = SU2FusionRule;
    let sources = compact_operator_cohort_fixture(&rule, su2(2), su2(2));
    let valid = sources[..4].to_vec();
    let permutation = [7usize, 6, 5, 4, 3, 2, 1, 0];
    let domain = [8usize];
    let baseline =
        multiplicity_free_permute_tree_pair_block(&rule, &valid, &permutation, &domain).unwrap();

    let mut malformed = valid.clone();
    let source = &valid[1];
    let codomain = source.codomain_tree();
    let shortened = FusionTreeKey::new(
        codomain.uncoupled().iter().copied(),
        codomain.coupled(),
        codomain.is_dual().iter().copied(),
        codomain.innerlines()[..codomain.innerlines().len() - 1]
            .iter()
            .copied(),
        codomain.vertices().iter().copied(),
    );
    malformed[1] = FusionTreePairKey::pair(shortened, source.domain_tree().clone());
    let snapshot = malformed.clone();

    // What: an error after earlier source rows were staged leaves caller
    // keys unchanged and cannot affect a later successful block transform.
    assert!(
        multiplicity_free_permute_tree_pair_block(&rule, &malformed, &permutation, &domain,)
            .is_err()
    );
    assert_eq!(malformed, snapshot);
    assert_eq!(
        multiplicity_free_permute_tree_pair_block(&rule, &valid, &permutation, &domain,).unwrap(),
        baseline
    );
}

#[test]
fn compact_repartition_preserves_source_major_error_precedence() {
    let codomain = |coupled, innerlines: &[SectorId]| {
        FusionTreeKey::new(
            [u1(1), u1(1), u1(1), u1(1)],
            coupled,
            [false; 4],
            innerlines.iter().copied(),
            [MultiplicityIndex::ONE; 3],
        )
    };
    let domain = |coupled| FusionTreeKey::new([u1(4)], coupled, [false], [], []);
    let sources = [
        FusionTreePairKey::pair(codomain(u1(4), &[u1(2)]), domain(u1(4))),
        FusionTreePairKey::pair(codomain(u1(4), &[]), domain(u1(99))),
    ];

    // What: the public categorical boundary rejects source 0's malformed
    // tree before compact bend-local validation can inspect later sources.
    assert_eq!(
        multiplicity_free_permute_tree_pair_block(&U1FusionRule, &sources, &[0, 1], &[4, 3, 2],)
            .unwrap_err(),
        CoreError::MalformedFusionTree {
            message: "fusion tree has an invalid number of innerlines",
        }
    );
}

#[test]
fn ordered_block_linear_storage_preserves_absent_and_zero_structure() {
    let rule = SU2FusionRule;
    let half = SectorLeg::new([(su2(1), 1)], false);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([half.clone(), half.clone(), half]),
        FusionProductSpace::new([SectorLeg::new([(su2(1), 1)], false)]),
    );
    let keys = hom.fusion_tree_keys(&rule);
    assert_eq!(keys.len(), 2);
    let group = validate_tree_pair_block_group_for_rule(&rule, &keys)
        .unwrap()
        .expect("SU2 cohort is nonempty");

    let singleton_basis = CompactMultiplicityFreeTreePairBasis::from_group(group).unwrap();
    let mut singleton_columns = DenseColumns::with_capacity(2, 2);
    let row0 = singleton_columns.push_empty_row();
    let row1 = singleton_columns.push_empty_row();
    singleton_columns.row_mut(row0)[0] = Some(0.0);
    singleton_columns.row_mut(row1)[1] = Some(2.0);
    let singleton = order_compact_block(singleton_basis, singleton_columns);
    assert_eq!(singleton.destinations(), keys.as_ref());
    assert_eq!(
        singleton.storage(),
        &OrderedBlockLinearStorage::SingletonColumns {
            destination_rows: vec![0, 1],
            coefficients: vec![0.0, 2.0],
        }
    );

    let dense_basis = CompactMultiplicityFreeTreePairBasis::from_group(group).unwrap();
    let mut dense_columns = DenseColumns::with_capacity(2, 2);
    let row0 = dense_columns.push_empty_row();
    let row1 = dense_columns.push_empty_row();
    dense_columns.row_mut(row0)[0] = Some(0.0);
    dense_columns.row_mut(row1)[0] = Some(1.0);
    dense_columns.row_mut(row1)[1] = Some(2.0);
    let dense = order_compact_block(dense_basis, dense_columns);
    assert_eq!(dense.destinations(), keys.as_ref());
    assert_eq!(
        dense.storage(),
        &OrderedBlockLinearStorage::DenseDstSrc(vec![Some(0.0), None, Some(1.0), Some(2.0),])
    );
}

#[test]
fn transpose_tree_pair_block_matches_complex_f_oracle() {
    let rule = FibonacciFAdmissibilityProbe::with_complex_f_phase();
    let tau = || SectorLeg::new([(SectorId::new(1), 1)], false);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([tau(), tau()]),
        FusionProductSpace::new([tau(), tau()]),
    );
    let sources = hom.fusion_tree_keys(&rule);
    assert!(sources.len() > 1);
    for (codomain_permutation, domain_permutation, expected_direction) in [
        ([1usize, 3], [0usize, 2], PreparedCycleDirection::Clockwise),
        (
            [2usize, 0],
            [3usize, 1],
            PreparedCycleDirection::Anticlockwise,
        ),
    ] {
        let prepared = PreparedTreePairOperation::prepare_transpose(
            2,
            2,
            &codomain_permutation,
            &domain_permutation,
        )
        .unwrap();
        assert!(matches!(
            prepared.plan,
            PreparedTreePairPlan::Transpose { direction, .. }
                if direction == expected_direction
        ));
        let rows = multiplicity_free_transpose_tree_pair_block(
            &rule,
            &sources,
            &codomain_permutation,
            &domain_permutation,
        )
        .unwrap();
        assert!(rows
            .iter()
            .flatten()
            .any(|(_, coefficient)| coefficient.im.abs() > 1.0e-12));

        // What: both compact cycle directions preserve ordered multi-row
        // non-real F products and conjugation against the full-key oracle.
        assert_compact_transpose_matches_full_key_oracle(
            &rule,
            &sources,
            &codomain_permutation,
            &domain_permutation,
            true,
        );
    }
}
