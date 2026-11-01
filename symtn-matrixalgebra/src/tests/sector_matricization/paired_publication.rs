use super::*;

/// The checked pair publisher over `provider`'s infallible checked view, with
/// its operation errors unwrapped: the view never fails, so this is the paired
/// publication every checked Generic factorization uses.
fn publish_pair<'r, R, D, M>(
    provider: &'r Arc<R>,
    homspace: &FusionTreeHomSpace,
    matricizations: &[M],
    pairs: Vec<FactorPair<D>>,
) -> Result<DynamicFactorPair<InfallibleGeneric<'r, R>, D>, OperationError>
where
    R: FusionRule,
    D: FactorScalar,
    M: SectorGeometry,
{
    let checked = Arc::new(InfallibleGeneric::new(provider.as_ref()));
    build_left_right_bound_pair_generic_checked(&checked, homspace, matricizations, pairs).map_err(
        |error| match error {
            CheckedGenericFactorPlanError::Provider(never) => match never {},
            CheckedGenericFactorPlanError::Operation(error) => error,
        },
    )
}

#[test]
fn generic_pair_publication_falls_back_for_padded_staged_geometry() {
    let (homspace, matrix) = z2_single_sector_matrix(2, 1);
    let provider = Arc::new(TestGenericRule);
    let generic_pair_rec = crate::probe::record();
    let scatter_visit_rec = crate::probe::record();

    let (left, right) = publish_pair(
        &provider,
        &homspace,
        &[matrix],
        vec![FactorPair {
            sector: SectorId::new(0),
            kept: 1,
            left: vec![2.0, 3.0],
            left_rows: 2,
            right: vec![5.0, 99.0],
            right_leading: 2,
        }],
    )
    .unwrap();

    assert_eq!(left.data(), [2.0, 3.0]);
    assert_eq!(right.data(), [5.0]);
    // G_s = 1 still pays the grouping pass (B = 1 per side) on top of the
    // F = 1 scattered block; disclosed, not dispatched on.
    assert_eq!(
        scatter_visit_rec.scatter_visit(),
        ScatterVisitProbe {
            left_grouped: 1,
            right_grouped: 1,
            left_groups_built: 1,
            right_groups_built: 1,
            left_visits: 1,
            right_visits: 1,
        }
    );
    let probe = generic_pair_rec.generic_pair_publication();
    assert_eq!(
        (probe.canonical_publications, probe.fallback_publications),
        (0, 1)
    );
    assert_eq!(
        (
            probe.left_scattered_elements,
            probe.right_scattered_elements
        ),
        (2, 1)
    );
}

#[test]
fn generic_pair_fallback_validation_indexes_each_sector_once() {
    let (sector_count, trees_per_sector) = (4usize, 8usize);
    let mut matrices = Vec::new();
    let mut left_keys = Vec::new();
    let mut right_keys = Vec::new();
    for sector in 0..sector_count {
        let bond = generic_pair(sector, 1, 1).codomain_tree().clone();
        let mut row_trees = Vec::new();
        let mut col_trees = Vec::new();
        for tree_index in 0..trees_per_sector {
            let source = generic_pair(sector, tree_index + 1, tree_index + 1);
            left_keys.push(FusionTreePairKey::pair(
                source.codomain_tree().clone(),
                bond.clone(),
            ));
            right_keys.push(FusionTreePairKey::pair(
                bond.clone(),
                source.domain_tree().clone(),
            ));
            row_trees.push((source.codomain_tree().clone(), tree_index, vec![1, 1]));
            col_trees.push((source.domain_tree().clone(), tree_index, vec![1, 1]));
        }
        matrices.push(SectorMatricization {
            sector: SectorId::new(sector),
            rows: trees_per_sector,
            cols: trees_per_sector,
            row_trees,
            col_trees,
            data: Vec::<f64>::new(),
        });
    }
    // Interleave the key order across sectors so each sector's index is
    // reused across non-adjacent keys.
    left_keys.reverse();
    right_keys.reverse();
    let total = sector_count * trees_per_sector;
    for (side, keys) in [
        (FactorSide::Left, &left_keys),
        (FactorSide::Right, &right_keys),
    ] {
        let generic_pair_rec = crate::probe::record();
        let placement_index_rec = crate::probe::record();
        let mut matrix_by_sector = None;
        assert!(
            validate_generic_factor_keys(keys, side, None, &matrices, &mut matrix_by_sector)
                .unwrap()
        );
        let probe = generic_pair_rec.generic_pair_publication();
        let lookups = match side {
            FactorSide::Left => probe.fallback_row_lookups,
            FactorSide::Right => probe.fallback_col_lookups,
        };
        assert_eq!(lookups, total);
        let index = placement_index_rec.placement_index();
        assert_eq!(
            index,
            PlacementIndexProbe {
                index_builds: 1,
                indexed_sides: sector_count,
                indexed_trees: total,
                lookups: total,
            }
        );
        // F + T = 64 hashes versus a per-sector F * T_c = 32 * 8 = 256
        // comparisons.
        assert!(index.indexed_trees + index.lookups < total * trees_per_sector);
    }
}

#[test]
fn generic_pair_publication_preserves_vertex_tree_payload_placement() {
    let provider = Arc::new(TestGenericRule);
    let (homspace, matrix, pair) = vertex_tree_factor_fixture(false);
    let expected_left = pair.left.clone();
    let expected_right = pair.right.clone();
    let generic_pair_rec = crate::probe::record();
    let (left, right) = publish_pair(&provider, &homspace, &[matrix], vec![pair]).unwrap();
    assert_eq!(left.data(), expected_left);
    assert_eq!(right.data(), expected_right);
    let probe = generic_pair_rec.generic_pair_publication();
    assert_eq!(
        (probe.canonical_publications, probe.fallback_publications),
        (1, 0)
    );
    assert_eq!(
        (probe.fallback_row_lookups, probe.fallback_col_lookups),
        (0, 0)
    );
    assert_eq!(
        (probe.left_scatter_calls, probe.right_scatter_calls),
        (0, 0)
    );
    // The checked publisher validates each published key once, in order.
    assert_eq!(
        probe.ordered_key_validation_events,
        left.space().space().structure().block_count()
            + right.space().space().structure().block_count()
    );

    let (homspace, matrix, pair) = vertex_tree_factor_fixture(true);
    let left_source = pair.left.clone();
    let right_source = pair.right.clone();
    let generic_pair_rec = crate::probe::record();
    let placement_index_rec = crate::probe::record();
    let (left, right) = publish_pair(&provider, &homspace, &[matrix], vec![pair]).unwrap();
    assert_eq!(
        left.data(),
        [
            left_source[2],
            left_source[3],
            left_source[0],
            left_source[1],
            left_source[6],
            left_source[7],
            left_source[4],
            left_source[5],
        ]
    );
    assert_eq!(
        right.data(),
        [
            right_source[6],
            right_source[7],
            right_source[8],
            right_source[9],
            right_source[10],
            right_source[11],
            right_source[0],
            right_source[1],
            right_source[2],
            right_source[3],
            right_source[4],
            right_source[5],
        ]
    );
    let probe = generic_pair_rec.generic_pair_publication();
    assert_eq!(
        (probe.canonical_publications, probe.fallback_publications),
        (0, 1)
    );
    assert_eq!(
        (probe.fallback_row_lookups, probe.fallback_col_lookups),
        (2, 2)
    );
    assert_eq!(
        (probe.left_scatter_calls, probe.right_scatter_calls),
        (2, 2)
    );
    // Row and column key validation each build a one-sided table; the
    // scatter loop shares one two-sided table (not four tables).
    assert_eq!(
        placement_index_rec.placement_index(),
        PlacementIndexProbe {
            index_builds: 3,
            indexed_sides: 4,
            indexed_trees: 8,
            lookups: 8,
        }
    );
}

#[test]
fn generic_pair_canonical_validation_scales_linearly_in_sectors_and_trees() {
    for (sector_count, trees_per_sector) in [(1, 1), (1, 4), (4, 1), (4, 3)] {
        let mut matrices = Vec::with_capacity(sector_count);
        let mut pairs = Vec::with_capacity(sector_count);
        let mut left_keys = Vec::with_capacity(sector_count * trees_per_sector);
        let mut right_keys = Vec::with_capacity(sector_count * trees_per_sector);
        let mut left_blocks = Vec::with_capacity(sector_count * trees_per_sector);
        let mut right_blocks = Vec::with_capacity(sector_count * trees_per_sector);
        let mut output_offset = 0usize;
        for sector in 0..sector_count {
            let bond = FusionTreePairKey::try_pair_from_sector_ids(
                [sector],
                [sector],
                sector,
                [false],
                [false],
                [],
                [],
                [],
                [],
            )
            .unwrap()
            .codomain_tree()
            .clone();
            let mut row_trees = Vec::with_capacity(trees_per_sector);
            let mut col_trees = Vec::with_capacity(trees_per_sector);
            for tree_index in 0..trees_per_sector {
                let source = generic_pair(sector, tree_index + 1, tree_index + 1);
                let row_tree = source.codomain_tree().clone();
                let col_tree = source.domain_tree().clone();
                let left_key = FusionTreePairKey::pair(row_tree.clone(), bond.clone());
                let right_key = FusionTreePairKey::pair(bond.clone(), col_tree.clone());
                left_blocks.push(
                    BlockSpec::with_key(
                        left_key.clone().into(),
                        vec![1, 1, 1],
                        vec![1, 1, trees_per_sector],
                        output_offset + tree_index,
                    )
                    .unwrap(),
                );
                right_blocks.push(
                    BlockSpec::with_key(
                        right_key.clone().into(),
                        vec![1, 1, 1],
                        vec![1, 1, 1],
                        output_offset + tree_index,
                    )
                    .unwrap(),
                );
                left_keys.push(left_key);
                right_keys.push(right_key);
                row_trees.push((row_tree, tree_index, vec![1, 1]));
                col_trees.push((col_tree, tree_index, vec![1, 1]));
            }
            let sector = SectorId::new(sector);
            matrices.push(SectorMatricization {
                sector,
                rows: trees_per_sector,
                cols: trees_per_sector,
                row_trees,
                col_trees,
                data: vec![0.0; trees_per_sector * trees_per_sector],
            });
            pairs.push(FactorPair {
                sector,
                kept: 1,
                left: (0..trees_per_sector)
                    .map(|index| 100.0 * sector.id() as f64 + index as f64)
                    .collect(),
                left_rows: trees_per_sector,
                right: (0..trees_per_sector)
                    .map(|index| -100.0 * sector.id() as f64 - index as f64)
                    .collect(),
                right_leading: 1,
            });
            output_offset += trees_per_sector;
        }
        let ranks = pairs
            .iter()
            .map(|pair| SectorRank {
                sector: pair.sector,
                kept: pair.kept,
            })
            .collect::<Vec<_>>();
        let left_structure = BlockStructure::from_blocks_with_rank(3, left_blocks).unwrap();
        let right_structure = BlockStructure::from_blocks_with_rank(3, right_blocks).unwrap();
        let expected_left = pairs
            .iter()
            .flat_map(|pair| pair.left.iter().copied())
            .collect::<Vec<_>>();
        let expected_right = pairs
            .iter()
            .flat_map(|pair| pair.right.iter().copied())
            .collect::<Vec<_>>();

        let generic_pair_rec = crate::probe::record();
        let mut matrix_by_sector = None;
        let mut left_cursor = FactorTreeCursor::new(&matrices, &ranks);
        assert!(validate_generic_factor_keys(
            &left_keys,
            FactorSide::Left,
            Some(&mut left_cursor),
            &matrices,
            &mut matrix_by_sector,
        )
        .unwrap());
        let mut right_cursor = FactorTreeCursor::new(&matrices, &ranks);
        assert!(validate_generic_factor_keys(
            &right_keys,
            FactorSide::Right,
            Some(&mut right_cursor),
            &matrices,
            &mut matrix_by_sector,
        )
        .unwrap());
        assert!(factor_output_is_canonical(
            &left_structure,
            &matrices,
            &pairs,
            expected_left.len(),
            FactorSide::Left,
        ));
        assert!(factor_output_is_canonical(
            &right_structure,
            &matrices,
            &pairs,
            expected_right.len(),
            FactorSide::Right,
        ));
        let output_blocks = left_keys.len() + right_keys.len();
        let probe = generic_pair_rec.generic_pair_publication();
        assert_eq!(probe.ordered_key_validation_events, output_blocks);
        assert_eq!(
            (probe.fallback_row_lookups, probe.fallback_col_lookups),
            (0, 0)
        );
        let (left, right) =
            publish_generic_factor_pairs(pairs, expected_left.len(), expected_right.len());
        assert_eq!(left, expected_left);
        assert_eq!(right, expected_right);
    }
}

#[test]
fn generic_pair_validation_preserves_missing_sector_and_tree_errors() {
    let (homspace, matrix) = z2_single_sector_matrix(1, 1);
    let provider = Arc::new(TestGenericRule);
    let pair = || FactorPair {
        sector: SectorId::new(0),
        kept: 1,
        left: vec![1.0],
        left_rows: 1,
        right: vec![1.0],
        right_leading: 1,
    };

    let missing_sector = publish_pair(
        &provider,
        &homspace,
        &[] as &[SectorMatricization<f64>],
        vec![pair()],
    )
    .unwrap_err();
    assert!(matches!(
        missing_sector,
        OperationError::SpaceMismatch {
            message: "factor tree references a coupled sector absent from the source tensor"
        }
    ));

    let wrong =
        FusionTreePairKey::try_pair_from_sector_ids([1], [0], 0, [false], [false], [], [], [], [])
            .unwrap()
            .codomain_tree()
            .clone();
    let mut wrong_row = matrix;
    wrong_row.row_trees[0].0 = wrong.clone();
    wrong_row.col_trees[0].0 = wrong.clone();
    let missing_row = publish_pair(&provider, &homspace, &[wrong_row], vec![pair()]).unwrap_err();
    assert!(matches!(
        missing_row,
        OperationError::SpaceMismatch {
            message: "factor codomain tree absent from the source matricization"
        }
    ));

    let (_, mut wrong_col) = z2_single_sector_matrix(1, 1);
    wrong_col.col_trees[0].0 = wrong;
    let missing_col = publish_pair(&provider, &homspace, &[wrong_col], vec![pair()]).unwrap_err();
    assert!(matches!(
        missing_col,
        OperationError::SpaceMismatch {
            message: "factor domain tree absent from the source matricization"
        }
    ));
}

#[test]
fn generic_pair_publication_handles_zero_kept_sector() {
    let provider = Arc::new(TestGenericRule);
    let (homspace, matrix) = z2_single_sector_matrix(2, 1);
    let generic_pair_rec = crate::probe::record();
    let (left, right) = publish_pair(
        &provider,
        &homspace,
        &[matrix],
        vec![FactorPair {
            sector: SectorId::new(0),
            kept: 0,
            left: Vec::<f64>::new(),
            left_rows: 2,
            right: Vec::new(),
            right_leading: 0,
        }],
    )
    .unwrap();
    assert!(left.data().is_empty());
    assert!(right.data().is_empty());
    assert_eq!(
        generic_pair_rec.generic_pair_publication(),
        GenericPairPublicationProbe {
            canonical_publications: 1,
            ..GenericPairPublicationProbe::default()
        }
    );

    let empty = FusionProductSpace::new(std::iter::empty::<SectorLeg>());
    let empty_hom = FusionTreeHomSpace::new(empty.clone(), empty);
    let (left, right) =
        publish_pair::<_, f64, SectorMatricization<f64>>(&provider, &empty_hom, &[], Vec::new())
            .unwrap();
    assert!(left.data().is_empty());
    assert!(right.data().is_empty());
}

#[test]
fn generic_pair_publication_handles_scalar_matrix() {
    let provider = Arc::new(TestGenericRule);
    let empty = FusionProductSpace::new(std::iter::empty::<SectorLeg>());
    let homspace = FusionTreeHomSpace::new(empty.clone(), empty);
    let key = homspace.fusion_tree_keys_generic(&TestGenericRule).unwrap()[0].clone();
    let generic_pair_rec = crate::probe::record();
    let (left, right) = publish_pair(
        &provider,
        &homspace,
        &[SectorMatricization {
            sector: SectorId::new(0),
            rows: 1,
            cols: 1,
            row_trees: vec![(key.codomain_tree().clone(), 0, Vec::new())],
            col_trees: vec![(key.domain_tree().clone(), 0, Vec::new())],
            data: vec![7.0],
        }],
        vec![FactorPair {
            sector: SectorId::new(0),
            kept: 1,
            left: vec![2.0],
            left_rows: 1,
            right: vec![3.5],
            right_leading: 1,
        }],
    )
    .unwrap();
    assert_eq!(left.data(), [2.0]);
    assert_eq!(right.data(), [3.5]);
    let probe = generic_pair_rec.generic_pair_publication();
    assert_eq!(probe.canonical_publications, 1);
    assert_eq!((probe.left_owner_reused, probe.right_owner_reused), (1, 1));
}

#[test]
fn generic_pair_fallback_scatters_rank1_right_block_with_right_layout() {
    let b = 3usize;
    let padding = 1usize;
    let (homspace, matrix) = zero_leg_side_matrix::<f64>(FactorSide::Right, 2, padding);
    let provider = Arc::new(TestGenericRule);
    let left = (0..2 * b).map(|k| 10.0 + k as f64).collect::<Vec<_>>();
    let right = (0..b * (1 + padding))
        .map(|k| 20.0 + k as f64)
        .collect::<Vec<_>>();
    let generic_pair_rec = crate::probe::record();

    let (left_factor, right_factor) = publish_pair(
        &provider,
        &homspace,
        std::slice::from_ref(&matrix),
        vec![FactorPair {
            sector: SectorId::new(0),
            kept: b,
            left: left.clone(),
            left_rows: 2,
            right: right.clone(),
            right_leading: b,
        }],
    )
    .unwrap();

    let probe = generic_pair_rec.generic_pair_publication();
    assert_eq!(
        (probe.canonical_publications, probe.fallback_publications),
        (0, 1)
    );
    assert_eq!(left_factor.data(), left);
    let block = right_factor.space().space().structure().block(0).unwrap();
    assert_eq!(block.shape(), [b]);
    assert_eq!(
        right_factor.data(),
        (0..b).map(|j| right[j + b * padding]).collect::<Vec<_>>()
    );
}

/// Z_4 fusion declared under Generic style so the generic paired builders
/// accept it; multiplicity-free, so trees and blocks match the abelian
/// geometry.
struct Z4GenericRule;

static Z4_GENERIC: Z4GenericRule = Z4GenericRule;

impl symtn_core::testing::Provider for Z4GenericRule {}

impl FusionRule for Z4GenericRule {
    fn rule_identity(&self) -> symtn_core::RuleIdentity {
        symtn_core::RuleIdentity::of_type::<Self>()
    }

    fn fusion_style(&self) -> symtn_core::FusionStyleKind {
        symtn_core::FusionStyleKind::Generic
    }

    fn braiding_style(&self) -> symtn_core::BraidingStyleKind {
        symtn_core::BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        SectorId::new((4 - sector.id() % 4) % 4)
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> symtn_core::SectorVec {
        [SectorId::new((left.id() + right.id()) % 4)]
            .into_iter()
            .collect()
    }

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        usize::from((left.id() + right.id()) % 4 == coupled.id())
    }
}

/// Two-leg codomain/domain hom space under [`Z4GenericRule`], matricized
/// exactly as the generic production path does.
fn z4_generic_geometry<D: FactorScalar>(
    legs: [Vec<(SectorId, usize)>; 4],
) -> (FusionTreeHomSpace, Vec<SectorMatricization<D>>) {
    let [a, b, c, d] = legs;
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new(a, false), SectorLeg::new(b, false)]),
        FusionProductSpace::new([SectorLeg::new(c, false), SectorLeg::new(d, false)]),
    );
    let space = BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(
        Arc::new(InfallibleGeneric::new(&Z4_GENERIC)),
        homspace.clone(),
    )
    .unwrap();
    let zeros = vec![D::zero(); space.space().required_len().unwrap()];
    let matrices = sector_matricizations(space.space().structure(), &zeros, 2).unwrap();
    (homspace, matrices)
}

/// Every leg carries all four charges: G_s = 4 coupled sectors with four
/// row and four column trees each, so each factor side has B = F = 16
/// output blocks.
fn z4_all_charge_legs() -> [Vec<(SectorId, usize)>; 4] {
    let all = (0..4).map(|s| (SectorId::new(s), 1)).collect::<Vec<_>>();
    [all.clone(), all.clone(), all.clone(), all]
}

/// Stages `rows x kept` left and `kept x cols` right factors per sector
/// with `kept = min(rows, cols)` and sector-tagged values.
fn staged_pairs<D: FactorScalar>(
    matrices: &[SectorMatricization<D>],
    values: &dyn Fn(usize) -> D,
) -> Vec<FactorPair<D>> {
    matrices
        .iter()
        .map(|matrix| {
            let kept = matrix.rows.min(matrix.cols);
            let tag = matrix.sector.id() + 1;
            FactorPair {
                sector: matrix.sector,
                kept,
                left: (0..matrix.rows * kept)
                    .map(|k| values(1000 * tag + k))
                    .collect(),
                left_rows: matrix.rows,
                right: (0..kept * matrix.cols)
                    .map(|k| values(5000 * tag + k))
                    .collect(),
                right_leading: kept,
            }
        })
        .collect()
}

#[test]
fn generic_pair_fallback_scatters_each_output_block_once() {
    // What: a paired fallback publication over G_s = 4 matricizations
    // groups each factor side once and iterates only the F blocks of the
    // matricization being scattered, publishing the same data as the
    // canonical path, for both paired builders and both scalar types.
    fn run<D: FactorScalar + fmt::Debug>(values: impl Fn(usize) -> D) {
        let fresh = || z4_generic_geometry::<D>(z4_all_charge_legs());
        let (homspace, matrices) = fresh();
        assert_eq!(matrices.len(), 4);
        assert!(matrices
            .iter()
            .all(|m| m.row_trees.len() == 4 && m.col_trees.len() == 4));
        let checked_provider = Arc::new(InfallibleGeneric::new(&Z4_GENERIC));

        let build = |matrices: &[SectorMatricization<D>]| {
            let generic_pair_rec = crate::probe::record();
            let scatter_visit_rec = crate::probe::record();
            let (left, right) = build_left_right_bound_pair_generic_checked(
                &checked_provider,
                &homspace,
                matrices,
                staged_pairs(matrices, &values),
            )
            .unwrap();
            assert_eq!(
                (
                    left.space().space().structure().block_count(),
                    right.space().space().structure().block_count()
                ),
                (16, 16)
            );
            (
                left.data().to_vec(),
                right.data().to_vec(),
                generic_pair_rec.generic_pair_publication(),
                scatter_visit_rec.scatter_visit(),
            )
        };

        let reference = build(&matrices);
        {
            let (_, _, probe, visits) = &reference;
            assert_eq!(probe.canonical_publications, 1);
            assert_eq!(*visits, ScatterVisitProbe::default());
        }

        let (_, mut reversed) = fresh();
        reversed.reverse();
        let checked = build(&reversed);
        {
            let ((left, right, probe, visits), (left_ref, right_ref, ..)) = (&checked, &reference);
            assert_eq!(probe.fallback_publications, 1);
            assert_eq!((left, right), (left_ref, right_ref));
            assert_eq!(
                (probe.left_scatter_calls, probe.right_scatter_calls),
                (16, 16)
            );
            // One grouping pass over B = 16 plus the F = 16 scattered
            // blocks, not G_s * B = 4 * 16 = 64 block visits per side.
            assert_eq!(
                *visits,
                ScatterVisitProbe {
                    left_grouped: 16,
                    right_grouped: 16,
                    left_groups_built: 1,
                    right_groups_built: 1,
                    left_visits: 16,
                    right_visits: 16,
                }
            );
            assert!(visits.left_grouped + visits.left_visits < 4 * 16);
            assert!(visits.right_grouped + visits.right_visits < 4 * 16);
        }
    }
    run(|k| k as f64 + 0.125);
    run(|k| Complex64::new(k as f64 + 0.125, 0.75 - k as f64));
}

#[test]
fn generic_pair_validation_reports_left_defect_before_right_across_sectors() {
    // What: validation precedence, not scatter order. With a defective
    // domain tree in the first sector and a defective codomain tree in a
    // later sector, the codomain error fires for both paired builders
    // because `validate_generic_factor_keys` checks every left key before
    // any right key. Scatter-time placement errors are unreachable in the
    // paired builders: validation has already checked every key.
    let two = || {
        let leg = || vec![(SectorId::new(0), 1), (SectorId::new(1), 1)];
        z4_generic_geometry::<f64>([leg(), leg(), leg(), leg()])
    };
    let (homspace, matrices) = two();
    assert_eq!(
        matrices.iter().map(|m| m.sector.id()).collect::<Vec<_>>(),
        [0, 1, 2]
    );
    let checked_provider = Arc::new(InfallibleGeneric::new(&Z4_GENERIC));
    let foreign_row = matrices[0].row_trees[0].0.clone();
    let foreign_col = matrices[1].col_trees[0].0.clone();
    let corrupt = |bad_first_col: bool, bad_later_row: bool| {
        let (_, mut matrices) = two();
        if bad_first_col {
            matrices[0].col_trees[0].0 = foreign_col.clone();
        }
        if bad_later_row {
            matrices[1].row_trees[0].0 = foreign_row.clone();
        }
        matrices
    };
    let codomain = "factor codomain tree absent from the source matricization";
    let domain = "factor domain tree absent from the source matricization";
    for (bad_first_col, bad_later_row, expected) in [
        (true, true, codomain),
        (true, false, domain),
        (false, true, codomain),
    ] {
        let matrices = corrupt(bad_first_col, bad_later_row);
        let error = build_left_right_bound_pair_generic_checked(
            &checked_provider,
            &homspace,
            &matrices,
            staged_pairs(&matrices, &|k| k as f64),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            CheckedGenericFactorPlanError::Operation(
                OperationError::SpaceMismatch { message }
            ) if message == expected
        ));
    }
}

/// The paired output HomSpaces exactly as
/// `build_left_right_bound_pair_generic_checked` derives them, built
/// independently of the returned factors.
fn paired_output_homs<D>(
    homspace: &FusionTreeHomSpace,
    pairs: &[FactorPair<D>],
) -> (FusionTreeHomSpace, FusionTreeHomSpace) {
    let bond = SectorLeg::new(pairs.iter().map(|pair| (pair.sector, pair.kept)), false);
    (
        FusionTreeHomSpace::new(
            homspace.codomain().clone(),
            FusionProductSpace::new([bond.clone()]),
        ),
        FusionTreeHomSpace::new(FusionProductSpace::new([bond]), homspace.domain().clone()),
    )
}

/// One checked factor space built in two passes: keys enumerated, then
/// the bound space enumerated again.
fn two_enumeration_space<R: CheckedGenericFusion>(
    provider: &Arc<R>,
    hom: FusionTreeHomSpace,
) -> (Vec<FusionTreePairKey>, BoundDynamicFusionMapSpace<R>) {
    let keys = hom
        .fusion_tree_keys_generic_checked(provider.as_ref())
        .unwrap();
    let space =
        BoundDynamicFusionMapSpace::from_final_homspace_generic_checked(Arc::clone(provider), hom)
            .unwrap();
    (keys, space)
}

fn assert_factor_matches_space<R: CheckedGenericFusion, D: FactorScalar + fmt::Debug>(
    factor: &BoundDynFactor<R, D>,
    keys: &[FusionTreePairKey],
    expected: &BoundDynamicFusionMapSpace<R>,
    provider: &Arc<R>,
) {
    let rows = block_rows(factor.space().space().structure());
    assert_eq!(rows, block_rows(expected.space().structure()));
    assert_eq!(
        rows.into_iter().map(|(key, ..)| key).collect::<Vec<_>>(),
        keys.iter().cloned().map(BlockKey::from).collect::<Vec<_>>()
    );
    let expected_len = expected.space().required_len().unwrap();
    assert_eq!(factor.space().space().required_len().unwrap(), expected_len);
    assert_eq!(factor.data().len(), expected_len);
    assert_eq!(
        factor.space().space().homspace(),
        expected.space().homspace()
    );
    assert!(Arc::ptr_eq(factor.space().provider_arc(), provider));
}

/// Checked paired publication equals the two-enumeration construction in
/// structure, key order, length, homspace and provider binding, and in data
/// the same publication over the plain rule's infallible view.
fn assert_checked_pair_matches_two_enumeration_construction<
    P: FusionRule,
    R: CheckedGenericFusion,
    D: FactorScalar + fmt::Debug,
    M: SectorGeometry,
>(
    plain: &Arc<P>,
    checked: &Arc<R>,
    homspace: &FusionTreeHomSpace,
    matrices: &[M],
    pairs: &dyn Fn() -> Vec<FactorPair<D>>,
    expected_publications: (usize, usize),
) {
    let (left_hom, right_hom) = paired_output_homs(homspace, &pairs());
    let (left_keys, left_expected) = two_enumeration_space(checked, left_hom);
    let (right_keys, right_expected) = two_enumeration_space(checked, right_hom);
    let (plain_left, plain_right) = publish_pair(plain, homspace, matrices, pairs()).unwrap();
    let generic_pair_rec = crate::probe::record();
    let (left, right) =
        build_left_right_bound_pair_generic_checked(checked, homspace, matrices, pairs()).unwrap();
    let probe = generic_pair_rec.generic_pair_publication();
    assert_eq!(
        (probe.canonical_publications, probe.fallback_publications),
        expected_publications
    );
    assert_eq!(
        probe.ordered_key_validation_events,
        left_keys.len() + right_keys.len()
    );
    assert_factor_matches_space(&left, &left_keys, &left_expected, checked);
    assert_factor_matches_space(&right, &right_keys, &right_expected, checked);
    assert_eq!(left.data(), plain_left.data());
    assert_eq!(right.data(), plain_right.data());
}

#[test]
#[allow(clippy::arc_with_non_send_sync)] // The checked API needs Arc identity; the recorder is a single-threaded RefCell log.
fn checked_paired_enumerates_each_factor_layout_once() {
    // What: one paired publication issues exactly the provider queries of
    // one left enumeration followed by one right enumeration (not keys and
    // then a bound space per side, which would issue each sequence twice). Canonical (`reverse = false`) and fallback
    // (`reverse = true`) geometries publish the same structure and data
    // as the two-enumeration construction.
    for reverse in [false, true] {
        let (homspace, matrix, pair) = vertex_tree_factor_fixture(reverse);
        let pairs = || vec![vertex_tree_factor_fixture(reverse).2];
        let recorder = Arc::new(RecordingGeneric::new());
        build_left_right_bound_pair_generic_checked(
            &recorder,
            &homspace,
            std::slice::from_ref(&matrix),
            vec![pair],
        )
        .unwrap();
        let once = recorder.log();

        let (left_hom, right_hom) = paired_output_homs(&homspace, &pairs());
        let side_log = |hom: &FusionTreeHomSpace| {
            let probe = RecordingGeneric::new();
            hom.prepare_coupled_subblock_structure_from_leg_degeneracies_generic_checked(&probe)
                .unwrap();
            probe.log()
        };
        let (once_left, once_right) = (side_log(&left_hom), side_log(&right_hom));
        // Vertex fixture: each side folds its one-leg bond once and
        // queries channels for its fold, in-table channels for its tree
        // walk and the multiplicity of each channel.
        assert_eq!((once_left.len(), once_right.len()), (4, 4));
        assert_eq!(once, [once_left.clone(), once_right.clone()].concat());

        let former = Arc::new(RecordingGeneric::new());
        two_enumeration_space(&former, left_hom);
        two_enumeration_space(&former, right_hom);
        assert_eq!(
            former.log(),
            [once_left.clone(), once_left, once_right.clone(), once_right].concat()
        );

        assert_checked_pair_matches_two_enumeration_construction(
            &Arc::new(TestGenericRule),
            &recorder,
            &homspace,
            std::slice::from_ref(&matrix),
            &pairs,
            if reverse { (0, 1) } else { (1, 0) },
        );
    }
}

#[test]
fn checked_paired_structure_matches_two_enumeration_construction() {
    // What: over G_s = 4 Z_4 matricizations with F = 16 blocks per side,
    // canonical and reversed (fallback) geometries, `f64` and
    // `Complex64`, the checked pair equals the two-enumeration
    // construction and, in data, the same publication over the plain rule.
    fn run<D: FactorScalar + fmt::Debug>(values: impl Fn(usize) -> D) {
        let fresh = || z4_generic_geometry::<D>(z4_all_charge_legs());
        let (homspace, matrices) = fresh();
        let (_, mut reversed) = fresh();
        reversed.reverse();
        let plain = Arc::new(Z4GenericRule);
        let checked = Arc::new(InfallibleGeneric::new(&Z4_GENERIC));
        for (matrices, publications) in [(&matrices, (1, 0)), (&reversed, (0, 1))] {
            assert_checked_pair_matches_two_enumeration_construction(
                &plain,
                &checked,
                &homspace,
                matrices,
                &|| staged_pairs(matrices, &values),
                publications,
            );
        }
    }
    run(|k| k as f64 + 0.125);
    run(|k| Complex64::new(k as f64 + 0.125, 0.75 - k as f64));
}

#[test]
#[allow(clippy::arc_with_non_send_sync)] // The checked API needs Arc identity; the recorder is a single-threaded RefCell log.
fn checked_paired_placement_error_precedes_bound_space() {
    // What: a factor key whose tree the matricization lacks fails after
    // that side's single enumeration and before any publication; the
    // left side is validated completely before the right side is
    // enumerated: the left error surfaces after 4 queries and the right
    // error after 8 (a two-pass route would take 4 and 12: left keys, left
    // space, right keys).
    for side in [FactorSide::Left, FactorSide::Right] {
        let (homspace, mut matrix, pair) = vertex_tree_factor_fixture(false);
        match side {
            FactorSide::Left => matrix.row_trees.truncate(1),
            FactorSide::Right => matrix.col_trees.truncate(1),
        }
        let recorder = Arc::new(RecordingGeneric::new());
        let generic_pair_rec = crate::probe::record();
        let error = build_left_right_bound_pair_generic_checked(
            &recorder,
            &homspace,
            std::slice::from_ref(&matrix),
            vec![pair],
        )
        .unwrap_err();
        let (expected_message, expected_calls) = match side {
            FactorSide::Left => (
                "factor codomain tree absent from the source matricization",
                4,
            ),
            FactorSide::Right => ("factor domain tree absent from the source matricization", 8),
        };
        assert!(matches!(
            error,
            CheckedGenericFactorPlanError::Operation(
                OperationError::SpaceMismatch { message }
            ) if message == expected_message
        ));
        assert_eq!(recorder.log().len(), expected_calls);
        let probe = generic_pair_rec.generic_pair_publication();
        assert_eq!(
            (probe.canonical_publications, probe.fallback_publications),
            (0, 0)
        );
    }
}
