use super::*;

struct PayloadFreeGeometry {
    sector: SectorId,
    rows: usize,
    cols: usize,
    row_tree: FusionTreeKey,
    row_shape: Vec<usize>,
    col_tree: FusionTreeKey,
    col_shape: Vec<usize>,
}

impl SectorGeometry for PayloadFreeGeometry {
    fn sector(&self) -> SectorId {
        self.sector
    }

    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn tree_count(&self, _side: FactorSide) -> usize {
        1
    }

    fn tree(&self, side: FactorSide, index: usize) -> Option<TreeExtentRef<'_>> {
        if index != 0 {
            return None;
        }
        let (tree, shape) = match side {
            FactorSide::Left => (&self.row_tree, self.row_shape.as_slice()),
            FactorSide::Right => (&self.col_tree, self.col_shape.as_slice()),
        };
        Some(TreeExtentRef {
            tree,
            offset: 0,
            shape,
        })
    }
}

#[test]
fn left_factor_publication_borrows_real_geometry_for_complex_output() {
    let x = SectorId::new(1);
    let vacuum = SectorId::new(0);
    let source_key = FusionTreePairKey::try_pair_from_sector_ids(
        [1, 1, 1],
        [1],
        1,
        [false; 3],
        [false],
        [0],
        std::iter::empty::<usize>(),
        [1, 1],
        std::iter::empty::<usize>(),
    )
    .unwrap();
    source_key.validate_for_rule(&Z2FusionRule).unwrap();
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(x, 2)], false),
            SectorLeg::new([(x, 1)], false),
            SectorLeg::new([(x, 3)], false),
        ]),
        FusionProductSpace::new([SectorLeg::new([(x, 1)], false)]),
    );
    let authority = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::new(Z2FusionRule),
        homspace.clone(),
    )
    .unwrap();
    let geometry = [PayloadFreeGeometry {
        sector: x,
        rows: 6,
        cols: 1,
        row_tree: source_key.codomain_tree().clone(),
        row_shape: vec![2, 1, 3],
        col_tree: source_key.domain_tree().clone(),
        col_shape: vec![1],
    }];
    let row_tree_before = geometry[0].row_tree.clone();
    let col_tree_before = geometry[0].col_tree.clone();
    let expected = (0..12)
        .map(|index| Complex64::new(index as f64 + 0.5, 10.0 - index as f64))
        .collect::<Vec<_>>();
    let mut pairs = [FactorPair {
        sector: x,
        kept: 2,
        left: expected.clone(),
        left_rows: 6,
        right: Vec::new(),
        right_leading: 0,
    }];

    let selected_ptr = pairs[0].left.as_ptr();
    let one_sided_rec = crate::probe::record();
    let factor = build_left_bound_factor(&authority, &homspace, &geometry, &mut pairs).unwrap();

    let probe = one_sided_rec.one_sided_publication();
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
    assert!(pairs[0].left.is_empty());
    assert_eq!(geometry[0].row_tree, row_tree_before);
    assert_eq!(geometry[0].col_tree, col_tree_before);
    assert_eq!(geometry[0].row_shape, [2, 1, 3]);
    assert_eq!(geometry[0].col_shape, [1]);
    let block = factor.space().space().structure().block(0).unwrap();
    assert_eq!(block.shape(), [2, 1, 3, 2]);
    let BlockKey::FusionTree(key) = block.key() else {
        panic!("left factor must retain the complete fusion-tree key")
    };
    assert_eq!(key.codomain_tree(), &row_tree_before);
    assert_eq!(key.domain_tree().uncoupled(), &[x]);
    assert_eq!(key.domain_tree().coupled(), x);
    assert_eq!(key.domain_tree().vertices(), &[]);
    assert_eq!(key.domain_tree().innerlines(), &[]);
    assert_eq!(factor.data().len(), expected.len());
    for bond in 0..2 {
        for third in 0..3 {
            for first in 0..2 {
                let output = block.offset()
                    + first * block.strides()[0]
                    + third * block.strides()[2]
                    + bond * block.strides()[3];
                let matrix_row = first + 2 * third;
                assert_eq!(factor.data()[output], expected[matrix_row + 6 * bond]);
            }
        }
    }
    assert_eq!(vacuum, row_tree_before.innerlines()[0]);
}

#[test]
fn mf_one_sided_placement_uses_direct_and_adjoint_source_sides() {
    let (homspace, mut matrix) = z2_single_sector_matrix(2, 3);
    matrix.rows = 3;
    matrix.row_trees[0].1 = 1;
    matrix.cols = 5;
    matrix.col_trees[0].1 = 2;
    matrix.data = Vec::new();
    let authority = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::new(Z2FusionRule),
        homspace.clone(),
    )
    .unwrap();
    let adjoint = symtn_tensors::adjoint_bound_space_dyn(&authority).unwrap();
    let values = |len: usize, base: f64| {
        (0..len)
            .map(|index| Complex64::new(base + index as f64, -base - index as f64 / 10.0))
            .collect::<Vec<_>>()
    };

    let cases = [
        (
            &authority,
            authority.space().homspace(),
            FactorSide::Left,
            FactorPlacement::Direct,
            2,
            FactorPair {
                sector: SectorId::new(0),
                kept: 99,
                left: values(6, 10.0),
                left_rows: 3,
                right: Vec::new(),
                right_leading: 0,
            },
            vec![1, 2, 4, 5],
        ),
        (
            &authority,
            authority.space().homspace(),
            FactorSide::Right,
            FactorPlacement::Direct,
            3,
            FactorPair {
                sector: SectorId::new(0),
                kept: 99,
                left: Vec::new(),
                left_rows: 0,
                right: values(15, 20.0),
                right_leading: 3,
            },
            (6..15).collect(),
        ),
        (
            &adjoint,
            adjoint.space().homspace(),
            FactorSide::Left,
            FactorPlacement::Adjoint,
            4,
            FactorPair {
                sector: SectorId::new(0),
                kept: 99,
                left: values(20, 30.0),
                left_rows: 5,
                right: Vec::new(),
                right_leading: 0,
            },
            vec![2, 3, 4, 7, 8, 9, 12, 13, 14, 17, 18, 19],
        ),
        (
            &adjoint,
            adjoint.space().homspace(),
            FactorSide::Right,
            FactorPlacement::Adjoint,
            2,
            FactorPair {
                sector: SectorId::new(0),
                kept: 99,
                left: Vec::new(),
                left_rows: 0,
                right: values(6, 40.0),
                right_leading: 2,
            },
            (2..6).collect(),
        ),
    ];

    for (authority, output_hom, side, placement, bond, mut pair, selected) in cases {
        let source = match side {
            FactorSide::Left => &pair.left,
            FactorSide::Right => &pair.right,
        };
        let expected = selected
            .into_iter()
            .map(|index| source[index])
            .collect::<Vec<_>>();
        let factor = build_bound_factor_with_placement(
            authority,
            output_hom,
            std::slice::from_ref(&matrix),
            std::slice::from_mut(&mut pair),
            &BTreeMap::from([(SectorId::new(0), bond)]),
            side,
            placement,
        )
        .unwrap();
        assert_eq!(factor.data(), expected);
    }
}

#[test]
fn mf_one_sided_pair_errors_precede_tree_traversal() {
    let (homspace, mut matrix) = z2_single_sector_matrix(1, 1);
    let authority = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::new(Z2FusionRule),
        homspace.clone(),
    )
    .unwrap();
    matrix.row_trees.clear();
    matrix.col_trees.clear();
    let dimensions = BTreeMap::from([(SectorId::new(0), 1)]);
    let pair = |sector| FactorPair {
        sector,
        kept: 1,
        left: vec![1.0],
        left_rows: 1,
        right: vec![2.0],
        right_leading: 1,
    };

    let placement_index_rec = crate::probe::record();
    let error = build_bound_factor_with_placement(
        &authority,
        &homspace,
        std::slice::from_ref(&matrix),
        &mut [pair(SectorId::new(0)), pair(SectorId::new(1))],
        &dimensions,
        FactorSide::Left,
        FactorPlacement::Direct,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        OperationError::SpaceMismatch {
            message: "factor sector absent from the source tensor"
        }
    ));
    assert_eq!(
        placement_index_rec.placement_index(),
        PlacementIndexProbe::default()
    );

    let error = build_bound_factor_with_placement::<_, f64, _>(
        &authority,
        &homspace,
        std::slice::from_ref(&matrix),
        &mut [],
        &dimensions,
        FactorSide::Right,
        FactorPlacement::Direct,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        OperationError::SpaceMismatch {
            message: "factor rank absent for a populated source sector"
        }
    ));
}

#[test]
fn one_sided_row_placement_uses_the_first_matching_duplicate() {
    let a = generic_pair(1, 1, 1).codomain_tree().clone();
    let b = generic_pair(1, 2, 1).codomain_tree().clone();
    let c = generic_pair(0, 1, 1).codomain_tree().clone();
    let matrix = SectorMatricization {
        sector: SectorId::new(1),
        rows: 7,
        cols: 0,
        row_trees: vec![
            (a.clone(), 0, vec![2]),
            (b.clone(), 2, vec![2]),
            (a.clone(), 4, vec![2]),
            (c.clone(), 6, vec![1]),
        ],
        col_trees: Vec::new(),
        data: Vec::<f64>::new(),
    };
    let index = PlacementIndex::new(std::slice::from_ref(&matrix), &[FactorSide::Left]);
    let offsets = [&b, &c, &a].map(|tree| {
        index
            .placement(SectorId::new(1), FactorSide::Left, tree)
            .unwrap()
            .0
    });

    assert_eq!(offsets, [2, 6, 0]);
}

#[test]
fn placement_index_distinguishes_dual_innerline_and_vertex_trees() {
    let base = full_identity_pair(7, false, 11, false);
    let dual = full_identity_pair(7, true, 11, true);
    let inner = full_identity_pair(8, false, 12, false);
    // Same sectors, duals and inner lines as `base`; only the vertices differ.
    let vertex = FusionTreePairKey::try_pair_from_sector_ids(
        [1, 2, 3],
        [4, 5, 6],
        9,
        [false, false, false],
        [true, false, false],
        [7],
        [11],
        [2, 2],
        [1, 1],
    )
    .unwrap();
    let (row_vertex, col_vertex) = (vertex.codomain_tree().clone(), vertex.domain_tree().clone());
    let matrix = SectorMatricization {
        sector: SectorId::new(9),
        rows: 4,
        cols: 4,
        row_trees: vec![
            (base.codomain_tree().clone(), 0, vec![1]),
            (dual.codomain_tree().clone(), 1, vec![1]),
            (inner.codomain_tree().clone(), 2, vec![1]),
            (row_vertex.clone(), 3, vec![1]),
        ],
        col_trees: vec![
            (base.domain_tree().clone(), 0, vec![1]),
            (dual.domain_tree().clone(), 1, vec![1]),
            (inner.domain_tree().clone(), 2, vec![1]),
            (col_vertex.clone(), 3, vec![1]),
        ],
        data: Vec::<f64>::new(),
    };
    let index = PlacementIndex::new(
        std::slice::from_ref(&matrix),
        &[FactorSide::Left, FactorSide::Right],
    );
    let sector = SectorId::new(9);
    assert_eq!(
        [
            base.codomain_tree(),
            dual.codomain_tree(),
            inner.codomain_tree(),
            &row_vertex
        ]
        .map(|tree| index.placement(sector, FactorSide::Left, tree).unwrap().0),
        [0, 1, 2, 3]
    );
    assert_eq!(
        [
            base.domain_tree(),
            dual.domain_tree(),
            inner.domain_tree(),
            &col_vertex
        ]
        .map(|tree| index.placement(sector, FactorSide::Right, tree).unwrap().0),
        [0, 1, 2, 3]
    );
    // A domain tree is never a codomain tree of this matrix and vice versa.
    assert!(matches!(
        index.placement(sector, FactorSide::Left, base.domain_tree()),
        Err(OperationError::SpaceMismatch {
            message: "factor codomain tree absent from the source matricization"
        })
    ));
    assert!(matches!(
        index.placement(sector, FactorSide::Right, base.codomain_tree()),
        Err(OperationError::SpaceMismatch {
            message: "factor domain tree absent from the source matricization"
        })
    ));
}

#[test]
fn mf_one_sided_duplicate_source_trees_publish_the_first_entry() {
    fn run<D: FactorScalar + fmt::Debug>(values: impl Fn(usize) -> D) {
        let (homspace, mut matrix) = z2_single_sector_matrix(2, 3);
        let authority = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
            Arc::new(Z2FusionRule),
            homspace.clone(),
        )
        .unwrap();
        let matrix = {
            let row = matrix.row_trees[0].0.clone();
            let col = matrix.col_trees[0].0.clone();
            matrix.rows = 4;
            matrix.row_trees = vec![(row.clone(), 0, vec![2]), (row, 2, vec![2])];
            matrix.cols = 6;
            matrix.col_trees = vec![(col.clone(), 0, vec![3]), (col, 3, vec![3])];
            SectorMatricization {
                sector: matrix.sector,
                rows: matrix.rows,
                cols: matrix.cols,
                row_trees: matrix.row_trees,
                col_trees: matrix.col_trees,
                data: Vec::<D>::new(),
            }
        };
        let source = (0..12).map(&values).collect::<Vec<_>>();
        let cases = [
            (FactorSide::Left, 2, 4, vec![0, 1, 4, 5], vec![2, 3, 6, 7]),
            (FactorSide::Right, 2, 2, (0..6).collect(), (6..12).collect()),
        ];
        for (side, bond, leading, first, second) in cases {
            let mut pair = match side {
                FactorSide::Left => FactorPair {
                    sector: SectorId::new(0),
                    kept: 99,
                    left: source.clone(),
                    left_rows: leading,
                    right: Vec::new(),
                    right_leading: 0,
                },
                FactorSide::Right => FactorPair {
                    sector: SectorId::new(0),
                    kept: 99,
                    left: Vec::new(),
                    left_rows: 0,
                    right: source.clone(),
                    right_leading: leading,
                },
            };
            let one_sided_rec = crate::probe::record();
            let placement_index_rec = crate::probe::record();
            let factor = build_bound_factor_with_placement(
                &authority,
                &homspace,
                std::slice::from_ref(&matrix),
                std::slice::from_mut(&mut pair),
                &BTreeMap::from([(SectorId::new(0), bond)]),
                side,
                FactorPlacement::Direct,
            )
            .unwrap();
            let pick = |indices: &[usize]| indices.iter().map(|&i| source[i]).collect::<Vec<_>>();
            assert_eq!(factor.data(), pick(&first));
            assert_ne!(pick(&first), pick(&second));
            assert_eq!(
                one_sided_rec.one_sided_publication().fallback_publications,
                1
            );
            assert_eq!(
                placement_index_rec.placement_index(),
                PlacementIndexProbe {
                    index_builds: 1,
                    indexed_sides: 1,
                    indexed_trees: 2,
                    lookups: 1,
                }
            );
        }
    }
    run(|k| k as f64 + 0.5);
    run(|k| Complex64::new(k as f64 + 0.5, -(k as f64)));
}

/// Z_8 with every leg carrying all eight charges: every coupled sector has
/// eight row trees and eight column trees, so F_c = T_c = 8 per sector.
fn z8_many_tree_legs() -> [Vec<(SectorId, usize)>; 4] {
    let all = (0..8).map(|s| (SectorId::new(s), 1)).collect::<Vec<_>>();
    [all.clone(), all.clone(), all.clone(), all]
}

#[test]
fn mf_one_sided_many_tree_fallback_indexes_each_populated_sector_once() {
    fn run<D: FactorScalar + fmt::Debug>(values: impl Fn(usize) -> D) {
        let fresh = || {
            two_leg_geometry::<_, D>(
                Arc::new(symtn_core::ZNFusionRule::new(8).unwrap()),
                z8_many_tree_legs(),
            )
        };
        let (authority, matrices) = fresh();
        assert_eq!(matrices.len(), 8);
        assert!(matrices
            .iter()
            .all(|m| m.row_trees.len() == 8 && m.col_trees.len() == 8));
        for (side, placement) in [
            (FactorSide::Left, FactorPlacement::Direct),
            (FactorSide::Right, FactorPlacement::Direct),
        ] {
            let source_trees = source_trees_for(side, placement);
            let dimensions = matrices
                .iter()
                .map(|m| (m.sector, source_extent(m, source_trees)))
                .collect::<BTreeMap<_, _>>();
            let build = |matrices: &[SectorMatricization<D>], pairs: &mut [FactorPair<D>]| {
                let one_sided_rec = crate::probe::record();
                let placement_index_rec = crate::probe::record();
                let factor = build_bound_factor_with_placement(
                    &authority,
                    authority.space().homspace(),
                    matrices,
                    pairs,
                    &dimensions,
                    side,
                    placement,
                )
                .unwrap();
                (
                    factor.data().to_vec(),
                    one_sided_rec.one_sided_publication(),
                    placement_index_rec.placement_index(),
                )
            };
            let staged = |matrices: &[SectorMatricization<D>]| {
                staged_one_sided_pairs(matrices, &dimensions, side, source_trees, &values)
            };

            let mut pairs = staged(&matrices);
            let (reference, probe, index) = build(&matrices, &mut pairs);
            assert_eq!(probe.canonical_publications, 1);
            assert_eq!(index, PlacementIndexProbe::default());

            // All eight sectors populated, reversed: F = T = 64 over
            // G_s = 8 in one table (not one per sector), so 128 hashes
            // instead of 8 * 8 * 8 = 512 key comparisons.
            let (_, mut reversed) = fresh();
            reversed.reverse();
            let mut pairs = staged(&reversed);
            let (data, probe, index) = build(&reversed, &mut pairs);
            assert_eq!(probe.fallback_publications, 1);
            assert_eq!(data, reference);
            assert_eq!(
                index,
                PlacementIndexProbe {
                    index_builds: 1,
                    indexed_sides: 8,
                    indexed_trees: 64,
                    lookups: 64,
                }
            );
            assert!(index.indexed_trees + index.lookups < 8 * 8 * 8);

            // Identity-only sectors 0 (before), 3 (between) and 7 (after)
            // publish canonically without any placement index.
            let (_, kept) = fresh();
            let kept = kept
                .into_iter()
                .filter(|m| ![0, 3, 7].contains(&m.sector.id()))
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
            assert_eq!(index, PlacementIndexProbe::default());
            let mut start = 0usize;
            for matrix in &matrices {
                let len = source_extent(matrix, source_trees) * dimensions[&matrix.sector];
                if ![0, 3, 7].contains(&matrix.sector.id()) {
                    assert_eq!(&data[start..start + len], &reference[start..start + len]);
                }
                start += len;
            }
            assert_eq!(start, data.len());
            let (authority, _) = fresh();
            let factor_space = build_bound_factor_space(
                &authority,
                authority.space().homspace(),
                SectorLeg::new(dimensions.iter().map(|(&s, &d)| (s, d)), false),
                side,
            )
            .unwrap();
            assert_literal_one_sided_layout(
                factor_space.space().structure(),
                &data,
                &kept,
                &selected,
                &dimensions,
                side,
                source_trees,
            );
        }
    }
    run(|k| k as f64 + 0.25);
    run(|k| Complex64::new(k as f64 + 0.25, 0.5 - k as f64));
}

/// Two-leg codomain/domain hom space matricized exactly as production
/// does, so tree orders and offsets are the real ones.
fn two_leg_geometry<R, D>(
    rule: Arc<R>,
    legs: [Vec<(SectorId, usize)>; 4],
) -> (BoundDynamicFusionMapSpace<R>, Vec<SectorMatricization<D>>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64>,
    D: FactorScalar,
{
    let [a, b, c, d] = legs;
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([SectorLeg::new(a, false), SectorLeg::new(b, false)]),
        FusionProductSpace::new([SectorLeg::new(c, false), SectorLeg::new(d, false)]),
    );
    let authority =
        BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(rule, homspace).unwrap();
    let data = vec![D::zero(); authority.space().required_len().unwrap()];
    let matrices = sector_matricizations(authority.space().structure(), &data, 2).unwrap();
    (authority, matrices)
}

fn z2_two_sector_geometry<D: FactorScalar>() -> (
    BoundDynamicFusionMapSpace<Z2FusionRule>,
    Vec<SectorMatricization<D>>,
) {
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    two_leg_geometry(
        Arc::new(Z2FusionRule),
        [
            vec![(even, 2), (odd, 1)],
            vec![(even, 1), (odd, 3)],
            vec![(even, 1), (odd, 2)],
            vec![(even, 2), (odd, 1)],
        ],
    )
}

fn source_trees_for(side: FactorSide, placement: FactorPlacement) -> FactorSide {
    match (side, placement) {
        (FactorSide::Left, FactorPlacement::Direct)
        | (FactorSide::Right, FactorPlacement::Adjoint) => FactorSide::Left,
        (FactorSide::Right, FactorPlacement::Direct)
        | (FactorSide::Left, FactorPlacement::Adjoint) => FactorSide::Right,
    }
}

fn mf_one_sided_canonical_transfer_case<D>(values: &dyn Fn(usize) -> D)
where
    D: FactorScalar + PartialEq + fmt::Debug,
{
    let (authority, matrices) = z2_two_sector_geometry::<D>();
    let adjoint = symtn_tensors::adjoint_bound_space_dyn(&authority).unwrap();
    let row_dimensions = matrices
        .iter()
        .map(|matrix| (matrix.sector, matrix.rows))
        .collect::<BTreeMap<_, _>>();
    let col_dimensions = matrices
        .iter()
        .map(|matrix| (matrix.sector, matrix.cols))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(matrices.len(), 2);
    assert!(matrices.iter().all(|matrix| matrix.rows != matrix.cols
        && matrix.row_trees.len() == 2
        && matrix.col_trees.len() == 2));
    assert_ne!(matrices[0].rows, matrices[1].rows);

    let cases = [
        (&authority, FactorSide::Left, FactorPlacement::Direct),
        (&authority, FactorSide::Right, FactorPlacement::Direct),
        (&adjoint, FactorSide::Left, FactorPlacement::Adjoint),
        (&adjoint, FactorSide::Right, FactorPlacement::Adjoint),
    ];
    for (space, side, placement) in cases {
        let source_trees = source_trees_for(side, placement);
        let dimensions = match source_trees {
            FactorSide::Left => &row_dimensions,
            FactorSide::Right => &col_dimensions,
        };
        let mut pairs = staged_one_sided_pairs(&matrices, dimensions, side, source_trees, values);
        let selected = pairs
            .iter()
            .map(|pair| selected_of(pair, side).clone())
            .collect::<Vec<_>>();
        let opposite = pairs
            .iter()
            .map(|pair| opposite_of(pair, side).clone())
            .collect::<Vec<_>>();

        let one_sided_rec = crate::probe::record();
        let factor = build_bound_factor_with_placement(
            space,
            space.space().homspace(),
            &matrices,
            &mut pairs,
            dimensions,
            side,
            placement,
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
        assert_literal_one_sided_layout(
            factor.space().space().structure(),
            factor.data(),
            &matrices,
            &selected,
            dimensions,
            side,
            source_trees,
        );
        for (index, pair) in pairs.iter().enumerate() {
            assert!(selected_of(pair, side).is_empty());
            assert_eq!(opposite_of(pair, side), &opposite[index]);
            assert_eq!((pair.kept, pair.left_rows, pair.right_leading).0, 99);
        }
    }
}

#[test]
fn mf_one_sided_canonical_transfer_real_full_svd_bonds() {
    mf_one_sided_canonical_transfer_case(&|k| k as f64 * 0.5 - 7.0);
}

#[test]
fn mf_one_sided_canonical_transfer_complex_full_svd_bonds() {
    mf_one_sided_canonical_transfer_case(&|k| Complex64::new(k as f64, -(k as f64) / 3.0));
}

#[test]
fn mf_one_sided_fallbacks_match_canonical_output_and_keep_buffers() {
    let (authority, matrices) = z2_two_sector_geometry::<f64>();
    let homspace = authority.space().homspace();
    let side = FactorSide::Left;
    let placement = FactorPlacement::Direct;
    let row_dimensions = matrices
        .iter()
        .map(|matrix| (matrix.sector, matrix.rows))
        .collect::<BTreeMap<_, _>>();
    let staged = |scale: f64| {
        staged_one_sided_pairs(&matrices, &row_dimensions, side, side, &|k| {
            scale * (k as f64 + 1.0)
        })
    };
    let build = |matrices: &[SectorMatricization<f64>],
                 pairs: &mut [FactorPair<f64>],
                 dimensions: &BTreeMap<SectorId, usize>| {
        let one_sided_rec = crate::probe::record();
        let result = build_bound_factor_with_placement(
            &authority, homspace, matrices, pairs, dimensions, side, placement,
        );
        (result, one_sided_rec.one_sided_publication())
    };
    let canonical = |scale: f64| {
        let mut pairs = staged(scale);
        let (factor, probe) = build(&matrices, &mut pairs, &row_dimensions);
        assert_eq!(
            (probe.canonical_publications, probe.fallback_publications),
            (1, 0)
        );
        factor.unwrap().data().to_vec()
    };
    let reference = canonical(1.0);
    let sector_len = matrices[0].rows * matrices[0].rows;

    // Reordered matricizations: same admitted output, scatter path.
    let (_, mut reversed) = z2_two_sector_geometry::<f64>();
    reversed.reverse();
    let mut pairs = staged(1.0);
    pairs.reverse();
    let (factor, probe) = build(&reversed, &mut pairs, &row_dimensions);
    assert_eq!(
        (probe.canonical_publications, probe.fallback_publications),
        (0, 1)
    );
    assert_eq!(factor.unwrap().data(), reference);
    assert_buffers_intact(&pairs, side);

    // Reordered pairs alone.
    let mut pairs = staged(1.0);
    pairs.reverse();
    let (factor, probe) = build(&matrices, &mut pairs, &row_dimensions);
    assert_eq!(probe.fallback_publications, 1);
    assert_eq!(factor.unwrap().data(), reference);
    assert_buffers_intact(&pairs, side);

    // Padded source geometry: the even factor carries one extra leading
    // row that no tree covers, so tree offsets do not start at zero.
    let (_, mut padded) = z2_two_sector_geometry::<f64>();
    padded[0].rows += 1;
    for tree in &mut padded[0].row_trees {
        tree.1 += 1;
    }
    let mut pairs =
        staged_one_sided_pairs(&padded, &row_dimensions, side, side, &|k| k as f64 + 1.0);
    let selected = pairs
        .iter()
        .map(|pair| pair.left.clone())
        .collect::<Vec<_>>();
    let (factor, probe) = build(&padded, &mut pairs, &row_dimensions);
    assert_eq!(probe.fallback_publications, 1);
    let factor = factor.unwrap();
    assert_eq!(factor.data().len(), reference.len());
    assert_literal_one_sided_layout(
        factor.space().space().structure(),
        factor.data(),
        &padded,
        &selected,
        &row_dimensions,
        side,
        side,
    );
    assert_buffers_intact(&pairs, side);

    // Identity-only sector after (drop odd) and before (drop even) the
    // populated sector publishes canonically: the populated region equals
    // the canonical one and the identity matches the literal oracle.
    for kept in [0..1, 1..2] {
        let mut pairs = staged(1.0);
        let selected = pairs[kept.clone()]
            .iter()
            .map(|pair| pair.left.clone())
            .collect::<Vec<_>>();
        let (factor, probe) = build(
            &matrices[kept.clone()],
            &mut pairs[kept.clone()],
            &row_dimensions,
        );
        assert_eq!(
            (probe.canonical_publications, probe.fallback_publications),
            (1, 0)
        );
        let factor = factor.unwrap();
        let populated = if kept.start == 0 {
            0..sector_len
        } else {
            sector_len..reference.len()
        };
        assert_eq!(&factor.data()[populated.clone()], &reference[populated]);
        assert_literal_one_sided_layout(
            factor.space().space().structure(),
            factor.data(),
            &matrices[kept],
            &selected,
            &row_dimensions,
            side,
            side,
        );
    }

    // Missing pair for a populated sector: unchanged error.
    let mut pairs = staged(1.0);
    let (result, probe) = build(&matrices, &mut pairs[..1], &row_dimensions);
    assert!(matches!(
        result.unwrap_err(),
        OperationError::SpaceMismatch {
            message: "factor rank absent for a populated source sector"
        }
    ));
    assert_eq!(probe.fallback_publications, 1);
    assert_buffers_intact(&pairs, side);

    // Extraneous pair: unchanged error before any publication decision.
    let mut pairs = staged(1.0);
    pairs.push(FactorPair {
        sector: SectorId::new(5),
        kept: 1,
        left: vec![1.0],
        left_rows: 1,
        right: Vec::new(),
        right_leading: 0,
    });
    let (result, probe) = build(&matrices, &mut pairs, &row_dimensions);
    assert!(matches!(
        result.unwrap_err(),
        OperationError::SpaceMismatch {
            message: "factor sector absent from the source tensor"
        }
    ));
    assert_eq!(
        (probe.canonical_publications, probe.fallback_publications),
        (0, 0)
    );
    assert_buffers_intact(&pairs, side);

    // Duplicate pair records for the even sector with different payloads:
    // the last record wins, on the scatter path.
    let alternate = canonical(2.0);
    let mut pairs = staged(1.0);
    let mut duplicate = staged(2.0);
    pairs.push(duplicate.swap_remove(0));
    let (factor, probe) = build(&matrices, &mut pairs, &row_dimensions);
    assert_eq!(probe.fallback_publications, 1);
    let factor = factor.unwrap();
    assert_eq!(&factor.data()[..sector_len], &alternate[..sector_len]);
    assert_eq!(&factor.data()[sector_len..], &reference[sector_len..]);
    assert_ne!(&alternate[..sector_len], &reference[..sector_len]);
    assert_buffers_intact(&pairs, side);
}

#[test]
fn mf_one_sided_identity_sector_between_populated_sectors_publishes_canonically() {
    let rule = Arc::new(symtn_core::ZNFusionRule::new(3).unwrap());
    let sectors = [SectorId::new(0), SectorId::new(1), SectorId::new(2)];
    let [s0, s1, s2] = sectors;
    let (authority, matrices) = two_leg_geometry::<_, f64>(
        rule,
        [
            vec![(s0, 1), (s1, 2), (s2, 1)],
            vec![(s0, 2), (s1, 1), (s2, 1)],
            vec![(s0, 1), (s1, 1), (s2, 2)],
            vec![(s0, 2), (s1, 1), (s2, 1)],
        ],
    );
    assert_eq!(
        matrices
            .iter()
            .map(|matrix| matrix.sector)
            .collect::<Vec<_>>(),
        sectors
    );
    let side = FactorSide::Right;
    let col_dimensions = matrices
        .iter()
        .map(|matrix| (matrix.sector, matrix.cols))
        .collect::<BTreeMap<_, _>>();
    let build = |matrices: &[SectorMatricization<f64>], pairs: &mut [FactorPair<f64>]| {
        let one_sided_rec = crate::probe::record();
        let factor = build_bound_factor_with_placement(
            &authority,
            authority.space().homspace(),
            matrices,
            pairs,
            &col_dimensions,
            side,
            FactorPlacement::Direct,
        )
        .unwrap();
        (
            factor.data().to_vec(),
            one_sided_rec.one_sided_publication(),
        )
    };
    let mut pairs =
        staged_one_sided_pairs(&matrices, &col_dimensions, side, side, &|k| k as f64 + 0.25);
    let (reference, probe) = build(&matrices, &mut pairs);
    assert_eq!(
        (probe.canonical_publications, probe.fallback_publications),
        (1, 0)
    );

    let (_, mut without_middle) = two_leg_geometry::<_, f64>(
        Arc::new(symtn_core::ZNFusionRule::new(3).unwrap()),
        [
            vec![(s0, 1), (s1, 2), (s2, 1)],
            vec![(s0, 2), (s1, 1), (s2, 1)],
            vec![(s0, 1), (s1, 1), (s2, 2)],
            vec![(s0, 2), (s1, 1), (s2, 1)],
        ],
    );
    without_middle.remove(1);
    let mut pairs = staged_one_sided_pairs(&without_middle, &col_dimensions, side, side, &|k| {
        k as f64 + 0.25
    });
    let selected = pairs
        .iter()
        .map(|pair| pair.right.clone())
        .collect::<Vec<_>>();
    let one_sided_rec = crate::probe::record();
    let factor = build_bound_factor_with_placement(
        &authority,
        authority.space().homspace(),
        &without_middle,
        &mut pairs,
        &col_dimensions,
        side,
        FactorPlacement::Direct,
    )
    .unwrap();
    let probe = one_sided_rec.one_sided_publication();
    assert_eq!(
        (probe.canonical_publications, probe.fallback_publications),
        (1, 0)
    );
    let first = matrices[0].cols * matrices[0].cols;
    let middle = matrices[1].cols * matrices[1].cols;
    // The first factor owns the output; the identity is written in place
    // and the last factor appended.
    assert_eq!(probe.appended_elements, middle + selected[1].len());
    let data = factor.data();
    assert_eq!(&data[..first], &reference[..first]);
    assert_eq!(&data[first + middle..], &reference[first + middle..]);
    assert_literal_one_sided_layout(
        factor.space().space().structure(),
        data,
        &without_middle,
        &selected,
        &col_dimensions,
        side,
        side,
    );

    // Reordered records around the identity sector keep the tree-identity
    // scatter and publish the same output.
    let (_, mut reversed) = two_leg_geometry::<_, f64>(
        Arc::new(symtn_core::ZNFusionRule::new(3).unwrap()),
        [
            vec![(s0, 1), (s1, 2), (s2, 1)],
            vec![(s0, 2), (s1, 1), (s2, 1)],
            vec![(s0, 1), (s1, 1), (s2, 2)],
            vec![(s0, 2), (s1, 1), (s2, 1)],
        ],
    );
    reversed.remove(1);
    reversed.reverse();
    let mut pairs =
        staged_one_sided_pairs(&reversed, &col_dimensions, side, side, &|k| k as f64 + 0.25);
    let (scattered, probe) = build(&reversed, &mut pairs);
    assert_eq!(
        (probe.canonical_publications, probe.fallback_publications),
        (0, 1)
    );
    assert_buffers_intact(&pairs, side);
    assert_eq!(scattered, data);
}

/// U(1) MPS site `(left x phys) <- right` with a codomain-only coupled
/// charge 2 and a domain-only charge 3; charge 0 is square (full rank,
/// omitted from both null spaces) and charge 1 is `6 x 4`.
fn u1_mps_side_only_space() -> BoundDynamicFusionMapSpace<symtn_core::U1FusionRule> {
    let q = |charge| symtn_core::U1Irrep::new(charge).sector_id();
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([
            SectorLeg::new([(q(0), 3), (q(1), 3)], false),
            SectorLeg::new([(q(0), 1), (q(1), 1)], false),
        ]),
        FusionProductSpace::new([SectorLeg::new([(q(0), 3), (q(1), 4), (q(3), 2)], false)]),
    );
    BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::new(symtn_core::U1FusionRule),
        homspace,
    )
    .unwrap()
}

/// Byte-traffic contract for side-only publication: every full/null factor
/// of the MPS site is published canonically, and the only elements
/// written beyond the moved first owner are the appended factors and the
/// in-place identities. The scatter fallback instead zero-fills all `P`
/// output elements and then scatters every populated block.
fn side_only_publication_bytes_case<D>(values: impl Fn(usize) -> D)
where
    D: FactorScalar + PartialEq + fmt::Debug,
{
    let engine = &mut symtn_dense::CpuEngine::serial();
    let space = u1_mps_side_only_space();
    let data = (0..space.space().required_len().unwrap())
        .map(&values)
        .collect::<Vec<_>>();
    let input = BoundDynamicTensorRef::try_new(&space, &data).unwrap();
    let matrices = sector_matricizations(space.space().structure(), &data, 2).unwrap();
    assert_eq!(
        matrices
            .iter()
            .map(|matrix| (matrix.rows, matrix.cols))
            .collect::<Vec<_>>(),
        [(3, 3), (6, 4)]
    );
    let q = |charge| symtn_core::U1Irrep::new(charge).sector_id();
    let codomain = BTreeMap::from([(q(0), 3), (q(1), 6), (q(2), 3)]);
    let domain = BTreeMap::from([(q(0), 3), (q(1), 4), (q(3), 2)]);
    let mut dense = symtn_dense::DefaultDenseExecutor::new();
    let bytes = |elements: usize| elements * std::mem::size_of::<D>();
    let check = |factor: &BoundDynFactor<_, D>,
                 dimensions: &BTreeMap<SectorId, usize>,
                 side: FactorSide| {
        assert_literal_one_sided_blocks(
            factor.space().space().structure(),
            factor.data(),
            &matrices,
            None,
            dimensions,
            side,
            side,
        );
    };
    let one_sided_rec = std::cell::RefCell::new(crate::probe::record());
    let publication = |expected_written: usize| {
        let probe = one_sided_rec.borrow().one_sided_publication();
        *one_sided_rec.borrow_mut() = crate::probe::record();
        assert_eq!(
            (probe.canonical_publications, probe.fallback_publications),
            (1, 0)
        );
        // An identity sector is the only case that allocates a plan.
        assert_ne!(probe.plan_bytes, 0);
        assert_eq!(bytes(probe.appended_elements), bytes(expected_written));
    };

    // Left null: charge 1 null basis 6 x 2 owns the buffer, charge 2 adds
    // a 3 x 3 identity, charge 0 is absent (P = 12 + 9, fallback 21 + 12 + 3).
    let null = left_null_dyn(engine, &mut dense, &input).unwrap();
    publication(9);
    assert_eq!(null.data().len(), 21);
    check(
        &null,
        &BTreeMap::from([(q(1), 2), (q(2), 3)]),
        FactorSide::Left,
    );
    // Right null: charges 0 and 1 have full column rank and are absent;
    // the charge-3 identity (2 x 2) is written into a fresh buffer.
    let null = right_null_dyn(engine, &mut dense, &input).unwrap();
    publication(4);
    assert_eq!(null.data().len(), 4);
    check(&null, &BTreeMap::from([(q(3), 2)]), FactorSide::Right);

    let Qr { q: left, r: right } = qr_full_dyn(engine, &mut dense, &input).unwrap();
    let _ = one_sided_rec.borrow().one_sided_publication();
    check(&left, &codomain, FactorSide::Left);
    check(&right, &codomain, FactorSide::Right);
    let Lq { l: left, q: right } = lq_full_dyn(engine, &mut dense, &input).unwrap();
    check(&left, &domain, FactorSide::Left);
    check(&right, &domain, FactorSide::Right);
    let svd = svd_full_dyn(engine, &mut dense, &input).unwrap();
    check(svd.u(), &codomain, FactorSide::Left);
    check(svd.vh(), &domain, FactorSide::Right);
    let probe = one_sided_rec.borrow().one_sided_publication();
    assert_eq!(
        (probe.canonical_publications, probe.fallback_publications),
        (6, 0)
    );
    // Q: 36 + 9 identity; R: 24 (charge 2 has no domain block); L: 24;
    // Q': 16 + 4 identity; U: 36 + 9; Vh: 16 + 4 (first owners moved).
    assert_eq!(
        bytes(probe.appended_elements),
        bytes(45 + 24 + 24 + 20 + 45 + 20)
    );
}

#[test]
fn side_only_publication_writes_each_output_element_once_real() {
    side_only_publication_bytes_case(|k| ((k * k) as f64 * 0.37 + k as f64).sin());
}

#[test]
fn side_only_publication_writes_each_output_element_once_complex() {
    side_only_publication_bytes_case(|k| {
        Complex64::new(
            ((k * k) as f64 * 0.37 + k as f64).sin(),
            ((k * k) as f64 * 0.23 - k as f64).cos(),
        )
    });
}

#[test]
fn mf_one_sided_zero_bond_sector_between_populated_sectors() {
    // Pinned outcome (a): the admitted structure omits zero-extent blocks,
    // so a zero-bond sector contributes no blocks and an empty factor; the
    // populated neighbours still transfer canonically.
    let legs = |s0, s1, s2| {
        [
            vec![(s0, 1), (s1, 2), (s2, 1)],
            vec![(s0, 2), (s1, 1), (s2, 1)],
            vec![(s0, 1), (s1, 1), (s2, 2)],
            vec![(s0, 2), (s1, 1), (s2, 1)],
        ]
    };
    let [s0, s1, s2] = [SectorId::new(0), SectorId::new(1), SectorId::new(2)];
    let (authority, matrices) = two_leg_geometry::<_, f64>(
        Arc::new(symtn_core::ZNFusionRule::new(3).unwrap()),
        legs(s0, s1, s2),
    );
    assert_eq!(
        matrices
            .iter()
            .map(|matrix| matrix.sector)
            .collect::<Vec<_>>(),
        [s0, s1, s2]
    );
    let side = FactorSide::Right;
    let dimensions = BTreeMap::from([(s0, matrices[0].cols), (s1, 0), (s2, matrices[2].cols)]);
    let mut pairs = staged_one_sided_pairs(&matrices, &dimensions, side, side, &|k| k as f64 - 0.5);
    pairs[1].kept = 0;
    assert!(pairs[1].right.is_empty());
    let selected = pairs
        .iter()
        .map(|pair| pair.right.clone())
        .collect::<Vec<_>>();
    let opposite = pairs
        .iter()
        .map(|pair| pair.left.clone())
        .collect::<Vec<_>>();

    let one_sided_rec = crate::probe::record();
    let factor = build_bound_factor_with_placement(
        &authority,
        authority.space().homspace(),
        &matrices,
        &mut pairs,
        &dimensions,
        side,
        FactorPlacement::Direct,
    )
    .unwrap();

    let probe = one_sided_rec.one_sided_publication();
    assert_eq!(
        (probe.canonical_publications, probe.fallback_publications),
        (1, 0)
    );
    assert_eq!(probe.appended_elements, selected[2].len());
    assert_eq!(factor.data().len(), selected[0].len() + selected[2].len());
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
        assert!(pair.right.is_empty());
        assert_eq!(pair.left, opposite[index]);
    }
}

/// Rank-1 blocks on both sides through the MF and the checked one-sided
/// owners (#1197). Right: `b x cols` column-major, the block at column
/// `o` is `F[j + b*o]`; Left: `a x b`, the block at row `o` is
/// `F[o + a*j]`. Before the fix the Right block was read with the Left
/// stride (`F[o + b*j]`), which for `b = 3, cols = 2, o = 1` reaches
/// `F[7]` past the six-element factor and panicked on the slice bound.
fn one_sided_rank1_fallback_case<D>(values: &dyn Fn(usize) -> D)
where
    D: FactorScalar + PartialEq + fmt::Debug,
{
    let even = SectorId::new(0);
    let b = 3usize;
    let padding = 1usize;
    let factor = (0..2 * b).map(values).collect::<Vec<_>>();
    let pair = |side| match side {
        FactorSide::Right => FactorPair {
            sector: even,
            kept: 99,
            left: vec![values(500)],
            left_rows: 1,
            right: factor.clone(),
            right_leading: b,
        },
        FactorSide::Left => FactorPair {
            sector: even,
            kept: 99,
            left: factor.clone(),
            left_rows: 1 + padding,
            right: vec![values(500)],
            right_leading: 1,
        },
    };
    let cases = [
        (
            FactorSide::Right,
            (0..b).map(|j| factor[j + b * padding]).collect::<Vec<_>>(),
        ),
        (
            FactorSide::Left,
            (0..b)
                .map(|j| factor[padding + (1 + padding) * j])
                .collect::<Vec<_>>(),
        ),
    ];
    for (side, expected) in cases {
        let (homspace, matrix) = zero_leg_side_matrix::<D>(side, 2, padding);
        let dimensions = BTreeMap::from([(even, b)]);
        let selected = factor.clone();

        let authority = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
            Arc::new(Z2FusionRule),
            homspace.clone(),
        )
        .unwrap();
        let mut pairs = [pair(side)];
        let one_sided_rec = crate::probe::record();
        let mf = build_bound_factor_with_placement(
            &authority,
            &homspace,
            std::slice::from_ref(&matrix),
            &mut pairs,
            &dimensions,
            side,
            FactorPlacement::Direct,
        )
        .unwrap();
        let probe = one_sided_rec.one_sided_publication();
        assert_eq!(
            (probe.canonical_publications, probe.fallback_publications),
            (0, 1)
        );
        let block = mf.space().space().structure().block(0).unwrap();
        assert_eq!(block.shape(), [b]);
        assert_eq!(mf.data(), expected);
        assert_eq!(selected_of(&pairs[0], side), &selected);

        let provider = Arc::new(InfallibleGeneric::new(&TestGenericRule));
        let mut pairs = [pair(side)];
        let one_sided_rec = crate::probe::record();
        let checked = build_bound_factor_generic_checked(
            &provider,
            &homspace,
            std::slice::from_ref(&matrix),
            &mut pairs,
            &dimensions,
            side,
        )
        .unwrap();
        let probe = one_sided_rec.one_sided_publication();
        assert_eq!(
            (probe.canonical_publications, probe.fallback_publications),
            (0, 1)
        );
        assert_eq!(checked.data(), expected);
        assert_eq!(selected_of(&pairs[0], side), &selected);
    }
}

#[test]
fn one_sided_rank1_fallback_scatters_right_and_left_layouts_real() {
    one_sided_rank1_fallback_case(&|k| k as f64 + 0.5);
}

#[test]
fn one_sided_rank1_fallback_scatters_right_and_left_layouts_complex() {
    one_sided_rank1_fallback_case(&|k| Complex64::new(k as f64, -(k as f64) - 0.25));
}
