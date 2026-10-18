use super::*;

#[test]
fn canonical_coupled_grid_matches_literal_u1_sector_matrices() {
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(u1(0), 2), (u1(1), 3)], false),
            SectorLeg::new([(u1(0), 5), (u1(1), 7)], true),
        ]),
        FusionProductSpace::new([
            SectorLeg::new([(u1(0), 11), (u1(1), 13)], true),
            SectorLeg::new([(u1(0), 17), (u1(1), 19)], false),
        ]),
    );

    let structure = homspace
        .coupled_subblock_structure_from_leg_degeneracies(&U1FusionRule)
        .unwrap();
    let expected = [
        (
            [u1(0), u1(0)],
            [u1(0), u1(0)],
            u1(0),
            [2, 5, 11, 17],
            [1, 2, 10, 110],
            0,
        ),
        (
            [u1(1), u1(0)],
            [u1(1), u1(0)],
            u1(1),
            [3, 5, 13, 17],
            [1, 3, 29, 377],
            1870,
        ),
        (
            [u1(0), u1(1)],
            [u1(1), u1(0)],
            u1(1),
            [2, 7, 13, 17],
            [1, 2, 29, 377],
            1885,
        ),
        (
            [u1(1), u1(0)],
            [u1(0), u1(1)],
            u1(1),
            [3, 5, 11, 19],
            [1, 3, 29, 319],
            8279,
        ),
        (
            [u1(0), u1(1)],
            [u1(0), u1(1)],
            u1(1),
            [2, 7, 11, 19],
            [1, 2, 29, 319],
            8294,
        ),
        (
            [u1(1), u1(1)],
            [u1(1), u1(1)],
            u1(2),
            [3, 7, 13, 19],
            [1, 3, 21, 273],
            14340,
        ),
    ];

    assert_eq!(structure.block_count(), expected.len());
    for (index, (codomain, domain, coupled, shape, strides, offset)) in expected.iter().enumerate()
    {
        let block = structure.block(index).unwrap();
        let BlockKey::FusionTree(key) = block.key() else {
            panic!("canonical symmetric block must use a fusion-tree key");
        };
        assert_eq!(key.codomain_uncoupled(), codomain);
        assert_eq!(key.domain_uncoupled(), domain);
        assert_eq!(key.codomain_is_dual(), &[false, true]);
        assert_eq!(key.domain_is_dual(), &[true, false]);
        assert_eq!(key.coupled(), *coupled);
        assert_eq!(block.shape(), shape);
        assert_eq!(block.strides(), strides);
        assert_eq!(block.offset(), *offset);
    }
    assert_eq!(structure.required_len().unwrap(), 19527);

    let regions = structure.coupled_sector_regions(2).unwrap().unwrap();
    assert_eq!(regions.len(), 3);
    assert_eq!(
        regions
            .iter()
            .map(|region| (
                region.coupled(),
                region.rows(),
                region.cols(),
                region.range(),
            ))
            .collect::<Vec<_>>(),
        vec![
            (u1(0), 10, 187, 0..1870),
            (u1(1), 29, 430, 1870..14340),
            (u1(2), 21, 247, 14340..19527),
        ]
    );
}

#[test]
fn canonical_su2_innerline_grid_matches_literal_sector_matrices() {
    let half = su2(1);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(half, 2)], false),
            SectorLeg::new([(half, 3)], true),
            SectorLeg::new([(half, 5)], false),
        ]),
        FusionProductSpace::new([
            SectorLeg::new([(half, 7)], true),
            SectorLeg::new([(half, 11)], false),
            SectorLeg::new([(half, 13)], true),
        ]),
    );

    let layout = homspace.fusion_tree_layout_data_uncached(&SU2FusionRule);
    assert_eq!(layout.keys.len(), 5);
    assert_eq!(layout.sectors.len(), 2);
    assert_eq!(
        (
            layout.sectors[0].start,
            layout.sectors[0].row_count,
            layout.sectors[0].col_count,
        ),
        (0, 2, 2)
    );
    assert_eq!(
        (
            layout.sectors[1].start,
            layout.sectors[1].row_count,
            layout.sectors[1].col_count,
        ),
        (4, 1, 1)
    );
    let expected_innerlines = [
        (su2(0), su2(0), su2(1)),
        (su2(2), su2(0), su2(1)),
        (su2(0), su2(2), su2(1)),
        (su2(2), su2(2), su2(1)),
        (su2(2), su2(2), su2(3)),
    ];
    for (key, &(codomain_inner, domain_inner, coupled)) in
        layout.keys.iter().zip(&expected_innerlines)
    {
        assert_eq!(key.codomain_uncoupled(), &[half, half, half]);
        assert_eq!(key.domain_uncoupled(), &[half, half, half]);
        assert_eq!(key.codomain_is_dual(), &[false, true, false]);
        assert_eq!(key.domain_is_dual(), &[true, false, true]);
        assert_eq!(key.codomain_innerlines(), &[codomain_inner]);
        assert_eq!(key.domain_innerlines(), &[domain_inner]);
        assert_eq!(key.coupled(), coupled);
    }

    let structure = homspace
        .coupled_subblock_structure_from_leg_degeneracies(&SU2FusionRule)
        .unwrap();
    let expected_offsets = [0, 30, 60060, 60090, 120120];
    for (index, &offset) in expected_offsets.iter().enumerate() {
        let block = structure.block(index).unwrap();
        assert_eq!(block.shape(), &[2, 3, 5, 7, 11, 13]);
        if index < 4 {
            assert_eq!(block.strides(), &[1, 2, 6, 60, 420, 4620]);
        } else {
            assert_eq!(block.strides(), &[1, 2, 6, 30, 210, 2310]);
        }
        assert_eq!(block.offset(), offset);
    }
    assert_eq!(structure.required_len().unwrap(), 150150);
    let regions = structure.coupled_sector_regions(3).unwrap().unwrap();
    assert_eq!(regions.len(), 2);
    assert_eq!(
        (
            regions[0].coupled(),
            regions[0].rows(),
            regions[0].cols(),
            regions[0].range(),
        ),
        (su2(1), 60, 2002, 0..120120)
    );
    assert_eq!(
        (
            regions[1].coupled(),
            regions[1].rows(),
            regions[1].cols(),
            regions[1].range(),
        ),
        (su2(3), 30, 1001, 120120..150150)
    );
}

#[test]
fn expert_cantor_product_keeps_the_encoded_fallback() {
    // What: custom-codec/expert product construction remains available via
    // the encoded public entry and keeps its Cantor-pairing IDs.
    type Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
    let rule = Rule::new(FermionParityFusionRule, U1FusionRule);
    let sector = TensorKitProductCodec::encode(z2_odd(), u1(2));
    let hom = singleton_rank_hom(sector, 3);
    assert_eq!(
        hom.fusion_tree_keys(&rule).as_ref(),
        hom.fusion_tree_keys_uncached(&rule)
    );
}

#[test]
fn fusion_layout_lookup_and_reset_are_concurrent_safe() {
    // What: concurrent lookup/reset cannot publish partial layout content or
    // return a structure with the wrong keys/shapes.
    let _guard = test_support::CACHE_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    clear_structure_caches();

    let workers = (0..4)
        .map(|worker| {
            std::thread::spawn(move || {
                let rule = U1FusionRule;
                for iteration in 0..64 {
                    if worker == 0 && iteration % 8 == 0 {
                        clear_structure_caches();
                    }
                    let charge = worker * 100 + iteration;
                    let hom = FusionTreeHomSpace::from_sectors(
                        [(U1Irrep::new(charge), 2)],
                        [(U1Irrep::new(charge), 3)],
                    );
                    let keys = hom.fusion_tree_keys(&rule);
                    assert_eq!(keys.len(), 1);
                    assert_eq!(keys[0].coupled(), U1Irrep::new(charge).sector_id());
                    let structure = hom
                        .coupled_subblock_structure(&rule, 1, [vec![2, 3]])
                        .unwrap();
                    assert_eq!(structure.block_count(), 1);
                    assert_eq!(structure.block(0).unwrap().shape(), &[2, 3]);
                }
            })
        })
        .collect::<Vec<_>>();
    for worker in workers {
        worker.join().unwrap();
    }
}

#[test]
fn braid_tree_pair_block_matches_per_source() {
    use std::collections::BTreeMap;
    let rule = SU2FusionRule;
    let leg = || {
        SectorLeg::new(
            [
                (SectorId::new(0), 1),
                (SectorId::new(1), 1),
                (SectorId::new(2), 1),
            ],
            false,
        )
    };
    // (V⊗V⊗V) ← V spans many uncoupled blocks; test each block.
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg(), leg()]),
        FusionProductSpace::new([leg()]),
    );
    let keys = hom.fusion_tree_keys(&rule);

    // Group source tree-pairs by their uncoupled block (the batching unit).
    let mut blocks: BTreeMap<Vec<usize>, Vec<FusionTreePairKey>> = BTreeMap::new();
    for key in keys.iter() {
        let tag: Vec<usize> = key
            .codomain_tree()
            .uncoupled()
            .iter()
            .chain(key.domain_tree().uncoupled())
            .map(|s| s.id())
            .collect();
        blocks.entry(tag).or_default().push(key.clone());
    }

    // Global leg indices: codomain legs 0,1,2 and domain leg 3. Reverse the
    // codomain, keep the domain leg in place.
    let codomain_permutation = [2usize, 1, 0];
    let domain_permutation = [3usize];
    let mut checked_blocks = 0;
    for src_keys in blocks.values() {
        let batched = multiplicity_free_permute_tree_pair_block(
            &rule,
            src_keys,
            &codomain_permutation,
            &domain_permutation,
        )
        .unwrap();
        assert_eq!(batched.len(), src_keys.len());
        for (src, batched_rows) in src_keys.iter().zip(&batched) {
            let per_source = multiplicity_free_permute_tree_pair(
                &rule,
                src,
                &codomain_permutation,
                &domain_permutation,
            )
            .unwrap();
            // Compare as key -> coefficient maps within double-precision tol.
            let mut want: BTreeMap<FusionTreePairKey, f64> = BTreeMap::new();
            for (k, c) in &per_source {
                *want.entry(k.clone()).or_insert(0.0) += c;
            }
            let mut got: BTreeMap<FusionTreePairKey, f64> = BTreeMap::new();
            for (k, c) in batched_rows {
                *got.entry(k.clone()).or_insert(0.0) += c;
            }
            assert_eq!(
                want.keys().collect::<Vec<_>>(),
                got.keys().collect::<Vec<_>>(),
                "destination trees differ for a source in block"
            );
            for (k, wc) in &want {
                let gc = got[k];
                assert!(
                    (wc - gc).abs() <= 1e-12 * (1.0 + wc.abs()),
                    "coefficient mismatch {wc} vs {gc}"
                );
            }
        }
        checked_blocks += 1;
    }
    assert!(checked_blocks > 0, "expected at least one block");
}

fn assert_compact_operator_cohorts<R>(rule: &R, sources: &[FusionTreePairKey])
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + MultiplicityFreeFusionRule,
{
    for cohort_len in [1usize, 2, 4, 8, 16] {
        let cohort = &sources[..cohort_len];

        // What: a direct compact bend-left produces the exact public
        // full-key rows in source and first-appearance destination order.
        let group = validate_tree_pair_block_group_for_rule(rule, cohort)
            .unwrap()
            .unwrap();
        let basis = CompactMultiplicityFreeTreePairBasis::from_group(group).unwrap();
        let (basis, columns) = compact_bendleft_block(rule, basis, None).unwrap();
        let got = scatter_compact_block(basis, columns);
        let want = cohort
            .iter()
            .map(|source| {
                multiplicity_free_bendleft_tree_pair(rule, source)
                    .unwrap()
                    .into_vec()
            })
            .collect::<Vec<_>>();
        assert_eq!(got, want);

        // What: a direct compact bend-right preserves the same one-row
        // oracle and ordering for every requested block cohort size.
        let group = validate_tree_pair_block_group_for_rule(rule, cohort)
            .unwrap()
            .unwrap();
        let basis = CompactMultiplicityFreeTreePairBasis::from_group(group).unwrap();
        let (basis, columns) = compact_bendright_block(rule, basis, None).unwrap();
        let got = scatter_compact_block(basis, columns);
        let want = cohort
            .iter()
            .map(|source| {
                multiplicity_free_bendright_tree_pair(rule, source)
                    .unwrap()
                    .into_vec()
            })
            .collect::<Vec<_>>();
        assert_eq!(got, want);

        // What: compact non-first Artin rows use the public F/R kernel and
        // retain its channel order for SU(2)-branching source cohorts.
        let group = validate_tree_pair_block_group_for_rule(rule, cohort)
            .unwrap()
            .unwrap();
        let basis = CompactMultiplicityFreeTreePairBasis::from_group(group).unwrap();
        let (basis, columns) = compact_codomain_artin_block(rule, basis, None, 3, false).unwrap();
        let got = scatter_compact_block(basis, columns);
        let want = cohort
            .iter()
            .map(|source| {
                let domain = source.domain_tree().clone();
                multiplicity_free_artin_braid_at_with_inverse(
                    rule,
                    source.codomain_tree(),
                    3,
                    false,
                )
                .unwrap()
                .into_iter()
                .map(|(codomain, coefficient)| {
                    (
                        FusionTreePairKey::pair(codomain, domain.clone()),
                        coefficient,
                    )
                })
                .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(got, want);
    }
}

#[test]
fn compact_block_operators_match_su2_cohort_oracles() {
    let rule = SU2FusionRule;
    let sources = compact_operator_cohort_fixture(&rule, su2(2), su2(2));

    assert_compact_operator_cohorts(&rule, &sources);
}

#[test]
fn compact_block_operators_match_fermionic_product_cohort_oracles() {
    type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
    type ProductRule = ProductFusionRule<FpU1Rule, SU2FusionRule>;
    let left = FpU1Rule::default();
    let rule = ProductRule::default();
    let external = rule.encode_component_ids(left.encode_component_ids(z2_odd(), u1(0)), su2(2));
    let coupled = rule.encode_component_ids(left.encode_component_ids(z2_even(), u1(0)), su2(2));
    let sources = compact_operator_cohort_fixture(&rule, external, coupled);

    assert_compact_operator_cohorts(&rule, &sources);
}

fn assert_all_codomain_compact_cohorts<R>(rule: &R, sources: &[FusionTreeKey])
where
    R: MultiplicityFreeFusionSymbols<Scalar = f64> + MultiplicityFreeFusionRule,
{
    let cases = [
        ([1usize, 0, 2, 3, 4, 5, 6, 7], [0usize, 1, 2, 3, 4, 5, 6, 7]),
        ([2usize, 0, 1, 3, 4, 5, 6, 7], [2usize, 0, 1, 3, 4, 5, 6, 7]),
    ];
    for cohort_len in [1usize, 2, 4, 8, 16] {
        let cohort = &sources[..cohort_len];
        for (permutation, levels) in cases {
            let got =
                multiplicity_free_braid_tree_block(rule, cohort, &permutation, &levels).unwrap();
            let want = cohort
                .iter()
                .map(|source| {
                    multiplicity_free_braid_tree(rule, source, &permutation, &levels).unwrap()
                })
                .collect::<Vec<_>>();
            assert_eq!(got.len(), want.len());
            for (got_row, want_row) in got.iter().zip(&want) {
                // What: compact all-codomain execution preserves the scalar
                // kernel's destination order as well as every full key.
                assert_eq!(
                    got_row.iter().map(|(key, _)| key).collect::<Vec<_>>(),
                    want_row.iter().map(|(key, _)| key).collect::<Vec<_>>()
                );
                assert_eq!(got_row.len(), want_row.len());
                for ((_, got_coefficient), (_, want_coefficient)) in got_row.iter().zip(want_row) {
                    assert!(
                        (got_coefficient - want_coefficient).abs()
                            <= 1.0e-12 * (1.0 + want_coefficient.abs())
                    );
                }
            }
        }
    }
}

#[test]
fn all_codomain_compact_block_matches_su2_cohorts() {
    let rule = SU2FusionRule;
    let sources = compact_operator_cohort_fixture(&rule, su2(2), su2(2))
        .into_iter()
        .map(|source| source.codomain_tree().clone())
        .collect::<Vec<_>>();

    assert_all_codomain_compact_cohorts(&rule, &sources);
}

#[test]
fn all_codomain_compact_block_matches_fermionic_product_cohorts() {
    type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
    type ProductRule = ProductFusionRule<FpU1Rule, SU2FusionRule>;
    let left = FpU1Rule::default();
    let rule = ProductRule::default();
    let external = rule.encode_component_ids(left.encode_component_ids(z2_odd(), u1(0)), su2(2));
    let coupled = rule.encode_component_ids(left.encode_component_ids(z2_even(), u1(0)), su2(2));
    let sources = compact_operator_cohort_fixture(&rule, external, coupled)
        .into_iter()
        .map(|source| source.codomain_tree().clone())
        .collect::<Vec<_>>();

    assert_all_codomain_compact_cohorts(&rule, &sources);
}

#[test]
fn transpose_tree_pair_block_matches_full_key_su2_cycles_and_repartition() {
    use std::collections::BTreeMap;
    let rule = SU2FusionRule;
    let leg = || {
        SectorLeg::new(
            [
                (SectorId::new(0), 1),
                (SectorId::new(1), 1),
                (SectorId::new(2), 1),
            ],
            false,
        )
    };
    // (V⊗V⊗V) ← V spans many uncoupled blocks; test each block.
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg(), leg()]),
        FusionProductSpace::new([leg()]),
    );
    let keys = hom.fusion_tree_keys(&rule);

    // Group source tree-pairs by their uncoupled block (the batching unit).
    let mut blocks: BTreeMap<Vec<usize>, Vec<FusionTreePairKey>> = BTreeMap::new();
    for key in keys.iter() {
        let tag: Vec<usize> = key
            .codomain_tree()
            .uncoupled()
            .iter()
            .chain(key.domain_tree().uncoupled())
            .map(|s| s.id())
            .collect();
        blocks.entry(tag).or_default().push(key.clone());
    }

    let mut checked_blocks = 0;
    for (codomain_permutation, domain_permutation, uses_dense_block) in [
        (vec![3usize], vec![2usize, 1, 0], true),
        (vec![1usize, 2, 3], vec![0usize], true),
        (vec![0usize, 1], vec![3usize, 2], false),
        (vec![0usize, 1, 2, 3], vec![], false),
    ] {
        for src_keys in blocks.values() {
            assert_compact_transpose_matches_full_key_oracle(
                &rule,
                src_keys,
                &codomain_permutation,
                &domain_permutation,
                uses_dense_block,
            );
            checked_blocks += 1;
        }
    }
    assert!(checked_blocks > 0, "expected at least one block");
}

// Regression (#1921): a transpose whose last compact step is a bend leaves the
// output basis in first-appearance order, not HomSpace order. For rank-5
// SU(2) and Fibonacci `[4,3,2 | 1,0]` from (2,3) that order differs from the
// HomSpace keys in some uncoupled blocks; values must still match per key.
fn assert_rank5_transpose_matches_per_pair_by_key<R>(rule: &R, sectors: &[SectorId])
where
    R: MultiplicityFreeRigidSymbols + MultiplicityFreeFusionRule,
    R::Scalar: Clone
        + Add<Output = R::Scalar>
        + Mul<Output = R::Scalar>
        + std::fmt::Debug
        + TransposeOracleScalar,
{
    use std::collections::BTreeMap;
    let leg = || SectorLeg::new(sectors.iter().map(|&sector| (sector, 1)), false);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg(), leg()]),
    );
    let mut blocks: BTreeMap<Vec<usize>, Vec<FusionTreePairKey>> = BTreeMap::new();
    for key in hom.fusion_tree_keys(rule).iter() {
        let tag = key
            .codomain_tree()
            .uncoupled()
            .iter()
            .chain(key.domain_tree().uncoupled())
            .map(|sector| sector.id())
            .collect();
        blocks.entry(tag).or_default().push(key.clone());
    }
    let (codomain_permutation, domain_permutation) = ([4usize, 3, 2], [1usize, 0]);
    let tolerance = |expected: &R::Scalar| 1.0e-12 * (1.0 + expected.oracle_magnitude());
    for sources in blocks.values() {
        assert_compact_transpose_matches_full_key_oracle(
            rule,
            sources,
            &codomain_permutation,
            &domain_permutation,
            true,
        );
        let compact = multiplicity_free_transpose_tree_pair_block(
            rule,
            sources,
            &codomain_permutation,
            &domain_permutation,
        )
        .unwrap();
        for (source, compact_rows) in sources.iter().zip(&compact) {
            // Independent oracle: the keyed per-pair transpose.
            let per_pair = crate::testing::multiplicity_free_transpose_tree_pair(
                rule,
                source,
                &codomain_permutation,
                &domain_permutation,
            )
            .unwrap();
            for (key, expected) in &per_pair {
                let actual = compact_rows
                    .iter()
                    .find(|(candidate, _)| candidate == key)
                    .map(|(_, coefficient)| coefficient);
                match actual {
                    Some(actual) => assert!(
                        actual.oracle_distance(expected) <= tolerance(expected),
                        "{key:?}: compact {actual:?} vs per-pair {expected:?}"
                    ),
                    None => assert!(
                        expected.oracle_magnitude() <= 1.0e-12,
                        "compact omits {key:?} = {expected:?}"
                    ),
                }
            }
            for (key, actual) in compact_rows {
                if !per_pair.iter().any(|(candidate, _)| candidate == key) {
                    assert!(
                        actual.oracle_magnitude() <= 1.0e-12,
                        "compact adds {key:?} = {actual:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn rank5_transpose_with_trailing_bend_matches_per_pair_by_key() {
    assert_rank5_transpose_matches_per_pair_by_key(&SU2FusionRule, &[su2(0), su2(1), su2(2)]);
    assert_rank5_transpose_matches_per_pair_by_key(
        &TableRule::fibonacci(),
        &[SectorId::new(0), SectorId::new(1)],
    );
}

// An empty source domain makes the codomain Artin swap the first compact move
// of a block braid (no bendleft precedes it), so it starts from the source
// columns rather than composing through a previous matrix.
fn assert_domainless_block_braid_matches_per_pair<R>(rule: &R, sectors: &[SectorId])
where
    R: MultiplicityFreeRigidSymbols + MultiplicityFreeFusionRule,
    R::Scalar: Clone
        + Add<Output = R::Scalar>
        + Mul<Output = R::Scalar>
        + std::fmt::Debug
        + TransposeOracleScalar,
{
    use std::collections::BTreeMap;
    let leg = || SectorLeg::new(sectors.iter().map(|&sector| (sector, 1)), false);
    let hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg(), leg()]),
        FusionProductSpace::new(Vec::<SectorLeg>::new()),
    );
    let mut blocks: BTreeMap<Vec<usize>, Vec<FusionTreePairKey>> = BTreeMap::new();
    for key in hom.fusion_tree_keys(rule).iter() {
        let tag = key
            .codomain_tree()
            .uncoupled()
            .iter()
            .map(|s| s.id())
            .collect();
        blocks.entry(tag).or_default().push(key.clone());
    }
    assert!(!blocks.is_empty(), "fixture must have domainless blocks");
    for (codomain_permutation, domain_permutation, levels) in [
        (vec![1usize, 0, 2], vec![], vec![0usize, 1, 2]),
        (vec![2usize, 1, 0], vec![], vec![2usize, 0, 1]),
        (vec![1usize, 2], vec![0usize], vec![0usize, 1, 2]),
    ] {
        for sources in blocks.values() {
            let compact = multiplicity_free_braid_tree_pair_block(
                rule,
                sources,
                &codomain_permutation,
                &domain_permutation,
                &levels,
                &[],
            )
            .unwrap();
            for (source, compact_rows) in sources.iter().zip(&compact) {
                let per_pair = multiplicity_free_braid_tree_pair(
                    rule,
                    source,
                    &codomain_permutation,
                    &domain_permutation,
                    &levels,
                    &[],
                )
                .unwrap();
                assert_eq!(compact_rows.len(), per_pair.len(), "{source:?}");
                for (key, expected) in &per_pair {
                    let actual = compact_rows
                        .iter()
                        .find(|(candidate, _)| candidate == key)
                        .map(|(_, coefficient)| coefficient)
                        .unwrap_or_else(|| panic!("compact braid omits {key:?}"));
                    assert!(
                        actual.oracle_distance(expected)
                            <= 1.0e-12 * (1.0 + expected.oracle_magnitude()),
                        "{key:?}: compact {actual:?} vs per-pair {expected:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn domainless_block_braid_matches_per_pair() {
    assert_domainless_block_braid_matches_per_pair(&SU2FusionRule, &[su2(0), su2(1), su2(2)]);
    assert_domainless_block_braid_matches_per_pair(
        &TableRule::fibonacci(),
        &[SectorId::new(0), SectorId::new(1)],
    );
}

#[test]
fn transpose_tree_pair_block_matches_full_key_fermionic_product_cycle() {
    type FpU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
    type ProductRule = ProductFusionRule<FpU1Rule, SU2FusionRule>;
    let left = FpU1Rule::default();
    let rule = ProductRule::default();
    let coupled = rule.encode_component_ids(left.encode_component_ids(z2_even(), u1(0)), su2(1));
    let odd_half = rule.encode_component_ids(left.encode_component_ids(z2_odd(), u1(1)), su2(1));
    let odd_one = rule.encode_component_ids(left.encode_component_ids(z2_odd(), u1(-1)), su2(2));
    let source = FusionTreePairKey::pair(
        FusionTreeKey::try_new_for_rule(&rule, [coupled], coupled, [false], [], []).unwrap(),
        FusionTreeKey::try_new_for_rule(
            &rule,
            [odd_half, odd_one],
            coupled,
            [false, true],
            [],
            [MultiplicityIndex::ONE],
        )
        .unwrap(),
    );

    // What: nested fZ2 x U1 x SU2 keeps the fermionic pivotal phase,
    // non-self-dual charge, and non-Abelian channel through a cycle.
    assert_compact_transpose_matches_full_key_oracle(&rule, &[source], &[2, 1], &[0], true);
}

#[test]
fn transpose_tree_pair_block_matches_low_rank_and_nonselfdual_u1_oracles() {
    let empty = FusionTreeKey::new([], u1(0), [], [], []);
    let rank_zero = [FusionTreePairKey::pair(empty.clone(), empty)];
    // What: scalar transpose remains the symbol-free identity operation.
    assert_compact_transpose_matches_full_key_oracle(&U1FusionRule, &rank_zero, &[], &[], false);

    let rank_one_hom = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new([(u1(0), 1)], false)]),
        FusionProductSpace::new([]),
    );
    let rank_one = rank_one_hom.fusion_tree_keys(&U1FusionRule);
    assert_eq!(rank_one.len(), 1);
    // What: moving one vacuum leg across the partition uses only the final
    // full-key reconstruction boundary.
    assert_compact_transpose_matches_full_key_oracle(&U1FusionRule, &rank_one, &[], &[0], false);

    // What: a non-self-dual U1 cycle preserves sector dualization and flags.
    assert_compact_transpose_matches_full_key_oracle(
        &U1FusionRule,
        &[u1_nonselfdual_tree_pair_fixture()],
        &[1, 2],
        &[0],
        true,
    );
}

#[test]
fn external_sectors_select_all_su2_simple_innerline_blocks() {
    let rule = SU2FusionRule;
    let half = SectorId::new(1);
    let hom = FusionTreeHomSpace::from_sector_ids([(1, 1), (1, 1), (1, 1)], [(1, 1)]);
    let structure = hom
        .coupled_subblock_structure(&rule, 3, [vec![1, 1, 1, 1], vec![1, 1, 1, 1]])
        .unwrap();

    let offsets = hom
        .fusion_tree_keys_from_external_sectors(&rule, &[half, half, half, half])
        .unwrap()
        .iter()
        .map(|key| structure.fusion_tree_pair_block(key).unwrap().offset())
        .collect::<Vec<_>>();

    assert_eq!(offsets, [0, 1]);

    let err = hom
        .unique_fusion_tree_key_from_external_sectors(&rule, &[half, half, half, half])
        .unwrap_err();
    assert_eq!(
        err,
        CoreError::BlockCountMismatch {
            expected: 1,
            actual: 2,
        }
    );
}

#[test]
fn fusion_tree_groups_preserve_structure_order_and_ignore_internal_tree_data() {
    let first = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [10, 20],
            [30],
            5,
            [false, true],
            [true],
            [101],
            [201],
            [301, 302],
            [401],
        )
        .unwrap(),
    );
    let second = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [1],
            [2, 3],
            4,
            [true],
            [false, true],
            [],
            [202],
            [303],
            [402, 403],
        )
        .unwrap(),
    );
    let same_group_as_first = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids(
            [10, 20],
            [30],
            6,
            [false, true],
            [true],
            [102],
            [203],
            [304, 305],
            [404],
        )
        .unwrap(),
    );

    let keys = vec![first.clone(), second.clone(), same_group_as_first.clone()];
    let sector = SectorStructure::from_keys(2, keys.clone()).unwrap();
    let block_structure =
        packed_fixture_structure(2, keys.into_iter().map(|key| (key, vec![1, 1]))).unwrap();

    let sector_groups = sector.fusion_tree_groups();
    let block_groups = block_structure.fusion_tree_groups();
    assert_eq!(sector_groups, block_groups);
    let mut first_appearance_groups = Vec::<FusionTreeBlockGroup>::new();
    for (index, key) in [first, second, same_group_as_first].iter().enumerate() {
        let group_key = key.fusion_tree_group_key().unwrap();
        if let Some(group) = first_appearance_groups
            .iter_mut()
            .find(|group| group.group_key() == &group_key)
        {
            group.block_indices.push(index);
        } else {
            first_appearance_groups.push(FusionTreeBlockGroup::new(group_key, vec![index]));
        }
    }
    // What: construction-time metadata is exactly the eager
    // first-appearance grouping, including interleaved storage indices.
    assert_eq!(sector_groups, first_appearance_groups);
    assert_eq!(sector_groups.len(), 2);
    assert_eq!(sector_groups[0].block_indices(), &[0, 2]);
    assert_eq!(sector_groups[1].block_indices(), &[1]);
    assert_eq!(
        sector_groups[0].group_key(),
        &FusionTreeGroupKey::from_sector_ids([10, 20], [30], [false, true], [true])
    );
    assert_eq!(
        sector_groups[1].group_key(),
        &FusionTreeGroupKey::from_sector_ids([1], [2, 3], [true], [false, true])
    );
}

#[test]
fn sector_structure_rejects_every_mixed_key_kind_pair() {
    let fusion = BlockKey::from(
        FusionTreePairKey::try_pair_from_sector_ids([7], [8], 9, [false], [true], [], [], [], [])
            .unwrap(),
    );
    let keys = [BlockKey::trivial(), BlockKey::opaque([7]), fusion];
    for expected in 0..keys.len() {
        for actual in 0..keys.len() {
            if expected == actual {
                continue;
            }
            assert_eq!(
                SectorStructure::from_keys(2, [keys[expected].clone(), keys[actual].clone()])
                    .unwrap_err(),
                CoreError::MixedBlockKeyKinds {
                    expected: keys[expected].kind(),
                    actual: keys[actual].kind(),
                }
            );
        }
    }

    let dense = BlockStructure::trivial(&[2, 3]).unwrap();
    let empty = BlockStructure::empty(2);
    assert!(dense.fusion_tree_groups().is_empty());
    assert!(empty.fusion_tree_groups().is_empty());
}

#[test]
fn opaque_block_key_words_are_rank_independent_application_identity() {
    let key = OpaqueBlockKey::from_words([3, 5, 8, 13, 21]);
    let block_key = BlockKey::from(key.clone());
    let rank_one = SectorStructure::from_keys(1, [block_key.clone()]).unwrap();
    let rank_seven = SectorStructure::from_keys(7, [block_key.clone()]).unwrap();

    // What: opaque word count has no relationship to tensor rank, and the
    // public constructors preserve all application routing words.
    assert_eq!(key.words(), &[3, 5, 8, 13, 21]);
    assert_eq!(OpaqueBlockKey::new(vec![3, 5, 8, 13, 21]), key);
    assert_eq!(rank_one.key(0).unwrap(), &block_key);
    assert_eq!(rank_seven.key(0).unwrap(), &block_key);
    assert_eq!(OpaqueBlockKey::ordinal(34).words(), &[34]);
}

#[test]
fn empty_sector_structure_has_one_canonical_namespace_free_form() {
    let constructed = SectorStructure::from_keys(3, std::iter::empty::<BlockKey>()).unwrap();
    let canonical = SectorStructure::empty(3);

    // What: both public empty constructors produce the same namespace-free
    // structure and never allocate a meaningless compact lookup.
    assert_eq!(constructed, canonical);
    assert_eq!(constructed.key_kind(), None);
    assert!(!constructed.has_compact_lookup());
}

#[test]
fn block_structure_separates_sector_and_degeneracy_data() {
    let sector =
        SectorStructure::from_keys(2, [BlockKey::opaque([0, 1]), BlockKey::opaque([1, 0])])
            .unwrap();
    let degeneracy = DegeneracyStructure::packed_column_major(2, [vec![2, 3], vec![3, 2]]).unwrap();
    let structure = BlockStructure::from_parts(sector, degeneracy).unwrap();

    assert_eq!(structure.rank(), 2);
    assert_eq!(
        structure.sector_structure().key(0).unwrap(),
        &BlockKey::opaque([0, 1])
    );
    assert_eq!(
        structure.sector_structure().key(1).unwrap(),
        &BlockKey::opaque([1, 0])
    );
    assert_eq!(
        structure.degeneracy_structure().block(0).unwrap().shape(),
        &[2, 3]
    );
    assert_eq!(
        structure.degeneracy_structure().block(1).unwrap().offset(),
        6
    );
    assert_eq!(structure.required_len().unwrap(), 12);
}

#[test]
fn sector_structure_pairs_compact_keys_without_map_lookup() {
    let dst = SectorStructure::from_keys(
        2,
        [
            BlockKey::opaque([2]),
            BlockKey::opaque([0]),
            BlockKey::opaque([1]),
        ],
    )
    .unwrap();
    let src = SectorStructure::from_keys(
        2,
        [
            BlockKey::opaque([0]),
            BlockKey::opaque([1]),
            BlockKey::opaque([2]),
        ],
    )
    .unwrap();

    assert!(src.has_compact_lookup());
    assert_eq!(dst.find_index(&BlockKey::opaque([0])), Some(1));
    assert_eq!(src.find_index(&BlockKey::opaque([2])), Some(2));
    assert_eq!(dst.pair_indices_from(&src).unwrap(), vec![2, 0, 1]);
}

#[test]
fn lookup_never_aliases_different_key_namespaces() {
    let dense = SectorStructure::dense(0);
    let opaque_zero = SectorStructure::from_keys(0, [BlockKey::ordinal(0)]).unwrap();
    let fusion_pair =
        FusionTreePairKey::try_pair_from_sector_ids([0], [], 0, [false], [], [], [], [], [])
            .unwrap();
    let fusion = SectorStructure::from_keys(1, [fusion_pair.clone()]).unwrap();
    let opaque_one = SectorStructure::from_keys(1, [BlockKey::ordinal(1)]).unwrap();

    // What: compact integer routing is only an accelerator inside the
    // Dense/Opaque namespaces and never establishes categorical identity.
    assert_eq!(dense.find_index(&BlockKey::ordinal(0)), None);
    assert_eq!(opaque_zero.find_index(&BlockKey::Dense), None);
    assert_eq!(
        opaque_one.find_index(&BlockKey::from(fusion_pair.clone())),
        None
    );
    assert_eq!(opaque_one.find_fusion_tree_pair_index(&fusion_pair), None);
    assert!(!fusion.has_compact_lookup());
    assert_eq!(
        opaque_one.pair_indices_from(&fusion),
        Err(CoreError::MixedBlockKeyKinds {
            expected: BlockKeyKind::Opaque,
            actual: BlockKeyKind::FusionTree,
        })
    );
}

#[test]
fn sector_structure_pairs_general_opaque_keys_by_sorted_merge() {
    let key_a = BlockKey::opaque([0, 1]);
    let key_b = BlockKey::opaque([1, 0]);
    let dst = SectorStructure::from_keys(2, [key_b.clone(), key_a.clone()]).unwrap();
    let src = SectorStructure::from_keys(2, [key_a.clone(), key_b.clone()]).unwrap();

    assert!(!src.has_compact_lookup());
    assert_eq!(dst.find_index(&key_a), Some(1));
    assert_eq!(src.find_index(&key_b), Some(1));
    assert_eq!(dst.pair_indices_from(&src).unwrap(), vec![1, 0]);
}

fn side_symmetric_u1_keys() -> Arc<[FusionTreePairKey]> {
    let leg = || SectorLeg::new([(u1(-1), 1), (u1(0), 1), (u1(1), 1)], false);
    FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    )
    .fusion_tree_keys(&U1FusionRule)
}

fn tree_arena_of(structure: &SectorStructure, index: usize) -> Arc<TreeArena> {
    let pair = structure.key(index).unwrap().as_fusion_tree_pair().unwrap();
    Arc::clone(&pair.codomain_tree().arena)
}

#[test]
fn a_structure_owns_its_tree_arena_and_reuses_only_a_complete_canonical_one() {
    let keys = side_symmetric_u1_keys();
    let layout_arena = Arc::clone(&keys[0].codomain_tree().arena);

    // What: the layout keys already use their whole canonical arena, so a
    // structure over all of them takes it without a copy (single-arena ownership, #2192).
    let full = SectorStructure::from_keys(4, keys.iter().cloned()).unwrap();
    assert!(Arc::ptr_eq(&tree_arena_of(&full, 0), &layout_arena));

    // What: a structure over three of those keys copies them into an arena
    // of its own instead of pinning the large one, and finds the same keys.
    let subset = SectorStructure::from_keys(4, keys.iter().take(3).cloned()).unwrap();
    let subset_arena = tree_arena_of(&subset, 0);
    assert!(!Arc::ptr_eq(&subset_arena, &layout_arena));
    assert!(subset_arena.len() < layout_arena.len());
    for index in 0..subset.block_count() {
        let pair = subset.key(index).unwrap().as_fusion_tree_pair().unwrap();
        assert!(Arc::ptr_eq(&pair.domain_tree().arena, &subset_arena));
        assert_eq!(pair, &keys[index]);
        assert_eq!(
            subset.find_fusion_tree_pair_index(&keys[index]),
            Some(index)
        );
    }
    assert_eq!(subset.find_fusion_tree_pair_index(&keys[3]), None);
}

#[test]
fn same_arena_lookup_of_swapped_trees_takes_the_content_search() {
    let keys = side_symmetric_u1_keys();
    let structure = SectorStructure::from_keys(4, keys.iter().cloned()).unwrap();
    for index in 0..structure.block_count() {
        let own = structure.key(index).unwrap().as_fusion_tree_pair().unwrap();
        // What: both handles point into the structure's arena but onto the
        // other side's rows, so the side-rank fast path must not apply; the
        // answer is the block whose trees equal them by content.
        let swapped =
            FusionTreePairKey::pair(own.domain_tree().clone(), own.codomain_tree().clone());
        let expected = (0..structure.block_count()).find(|&candidate| {
            structure.key(candidate).unwrap() == &BlockKey::from(swapped.clone())
        });
        assert_eq!(structure.find_fusion_tree_pair_index(&swapped), expected);
        assert_eq!(structure.find_fusion_tree_pair_index(own), Some(index));
        assert_eq!(
            structure.find_adjoint_fusion_tree_pair_index(&swapped),
            Some(index)
        );
    }
}

#[test]
fn same_arena_lookup_with_one_tree_on_the_wrong_side_takes_the_content_search() {
    let keys = side_symmetric_u1_keys();
    let structure = SectorStructure::from_keys(4, keys.iter().cloned()).unwrap();
    let content_index = |pair: &FusionTreePairKey| {
        (0..structure.block_count())
            .find(|&candidate| structure.key(candidate).unwrap() == &BlockKey::from(pair.clone()))
    };
    for index in 0..structure.block_count() {
        let own = structure.key(index).unwrap().as_fusion_tree_pair().unwrap();
        // What: both handles point into the structure's arena, but one of
        // them sits on the other side's rows (domain, domain) or (codomain,
        // codomain); neither half of the side guard may be skipped.
        for probe in [
            FusionTreePairKey::pair(own.domain_tree().clone(), own.domain_tree().clone()),
            FusionTreePairKey::pair(own.codomain_tree().clone(), own.codomain_tree().clone()),
        ] {
            let expected = content_index(&probe);
            assert_eq!(structure.find_fusion_tree_pair_index(&probe), expected);
        }
    }
    // The fixture must exercise hits, not only misses.
    assert!((0..structure.block_count()).any(|index| {
        let own = structure.key(index).unwrap().as_fusion_tree_pair().unwrap();
        let probe = FusionTreePairKey::pair(own.domain_tree().clone(), own.domain_tree().clone());
        structure.find_fusion_tree_pair_index(&probe).is_some()
    }));
}

#[test]
fn a_structure_of_side_swapped_layout_keys_does_not_reuse_their_arena() {
    let keys = side_symmetric_u1_keys();
    let layout_arena = Arc::clone(&keys[0].codomain_tree().arena);
    // What: every arena row is used, but each codomain tree sits on a domain
    // row and vice versa, so the arena is not this structure's canonical
    // one; it is re-homed and every swapped key is found by content.
    let swapped = keys
        .iter()
        .map(|key| FusionTreePairKey::pair(key.domain_tree().clone(), key.codomain_tree().clone()))
        .collect::<Vec<_>>();
    let structure = SectorStructure::from_keys(4, swapped.iter().cloned()).unwrap();
    assert!(!Arc::ptr_eq(&tree_arena_of(&structure, 0), &layout_arena));
    for (index, key) in swapped.iter().enumerate() {
        assert_eq!(structure.find_fusion_tree_pair_index(key), Some(index));
        assert_eq!(structure.key(index).unwrap(), &BlockKey::from(key.clone()));
    }
}

/// Keys of the SU(2) `½^⊗8 ← ½^⊗8` layout: many trees per external tuple.
fn su2_half_keys() -> Arc<[FusionTreePairKey]> {
    let half = || SectorLeg::new([(SU2Irrep::from_twice_spin(1).sector_id(), 1)], false);
    FusionTreeHomSpace::new(
        FusionProductSpace::new((0..4).map(|_| half())),
        FusionProductSpace::new((0..4).map(|_| half())),
    )
    .fusion_tree_keys(&SU2FusionRule)
}

#[test]
fn trees_of_one_external_tuple_store_their_externals_once() {
    let keys = su2_half_keys();
    // What: every tree of either side has the same uncoupled sectors,
    // duality flags and vertex labels, so the arena stores them once and
    // otherwise only each tree's coupled sector and inner lines (#2192),
    // whatever order the keys arrive in.
    for order in [
        keys.iter().cloned().collect::<Vec<_>>(),
        keys.iter().rev().cloned().collect(),
        keys.iter()
            .map(|key| {
                FusionTreePairKey::pair(key.domain_tree().clone(), key.codomain_tree().clone())
            })
            .collect(),
    ] {
        let structure = SectorStructure::from_keys(8, order.iter().cloned()).unwrap();
        let arena = tree_arena_of(&structure, 0);
        let lines = (0..arena.len())
            .map(|row| 1 + arena.innerlines(row).len())
            .sum::<usize>();
        // One run of 4 uncoupled sectors, 3 vertices and 4 flags.
        assert_eq!(arena.stored_entries(), (lines + 4, 3, 4));
        for (index, key) in order.iter().enumerate() {
            assert_eq!(structure.find_fusion_tree_pair_index(key), Some(index));
        }
    }
}

fn u1_raw_tree(uncoupled: &[usize], innerlines: &[usize], coupled: usize) -> FusionTreeKey {
    FusionTreeKey::try_from_sector_ids_for_rule(
        &U1FusionRule,
        uncoupled.iter().copied(),
        coupled,
        uncoupled.iter().map(|_| false),
        innerlines.iter().copied(),
        std::iter::repeat_n(1usize, uncoupled.len().saturating_sub(1)),
    )
    .unwrap()
}

/// U(1) zigzag sector id of `charge`, and back.
fn u1_id(charge: i64) -> usize {
    (if charge >= 0 {
        2 * charge
    } else {
        -2 * charge - 1
    }) as usize
}

fn u1_charge(id: usize) -> i64 {
    if id.is_multiple_of(2) {
        (id / 2) as i64
    } else {
        -(id.div_ceil(2) as i64)
    }
}

#[test]
fn shared_runs_survive_a_content_hash_collision_of_different_lengths() {
    // What: [a] and [b1, b2, b3] are valid U(1) uncoupled runs whose run
    // hashes collide (found by lattice reduction). Sharing must
    // compare stored length and content, never read past the stored run:
    // the structure builds and every key is found as itself.
    let (a, b1, b2, b3) = (
        536870912usize,
        268711625usize,
        267632009usize,
        536388608usize,
    );
    let sum = u1_charge(b1) + u1_charge(b2) + u1_charge(b3);
    let keys = [
        FusionTreePairKey::pair(
            u1_raw_tree(&[a], &[], a),
            u1_raw_tree(&[a, u1_id(0), u1_id(0)], &[a], a),
        ),
        FusionTreePairKey::pair(
            u1_raw_tree(&[u1_id(sum)], &[], u1_id(sum)),
            u1_raw_tree(
                &[b1, b2, b3],
                &[u1_id(u1_charge(b1) + u1_charge(b2))],
                u1_id(sum),
            ),
        ),
    ];
    let structure = SectorStructure::from_keys(4, keys.iter().cloned()).unwrap();
    for key in &keys {
        let found = structure.find_fusion_tree_pair_index(key).unwrap();
        assert_eq!(
            structure.key(found).unwrap().as_fusion_tree_pair().unwrap(),
            key
        );
    }
}

#[test]
fn shared_runs_survive_a_content_hash_collision_of_equal_lengths() {
    // What: these two raw two-sector runs have colliding run hashes (a
    // two-dimensional lattice solution); each tree keeps its own content.
    let raw = |uncoupled: [usize; 2], coupled: usize| {
        FusionTreeKey::try_from_sector_ids(uncoupled, coupled, [false, false], [], [1]).unwrap()
    };
    let keys = [
        FusionTreePairKey::pair(
            raw([8589934592, 8589934592], 0),
            raw([8589934592, 8589934592], 0),
        ),
        FusionTreePairKey::pair(
            raw([5538824234, 5310866862], 0),
            raw([5538824234, 5310866862], 0),
        ),
    ];
    let structure = SectorStructure::from_keys(4, keys.iter().cloned()).unwrap();
    for key in &keys {
        let found = structure.find_fusion_tree_pair_index(key).unwrap();
        assert_eq!(
            structure.key(found).unwrap().as_fusion_tree_pair().unwrap(),
            key
        );
    }
}
