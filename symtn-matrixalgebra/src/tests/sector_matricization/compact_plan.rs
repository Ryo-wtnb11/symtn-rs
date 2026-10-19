use super::*;

#[test]
fn compact_routes_reject_an_extra_positive_output_region() {
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let leg = SectorLeg::new([(even, 1), (odd, 1)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone()]),
        FusionProductSpace::new([leg]),
    );
    let authority = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::new(Z2FusionRule),
        homspace,
    )
    .unwrap();
    let regions = checked_sector_regions(authority.space().structure(), 1)
        .unwrap()
        .unwrap();

    let error = compile_compact_factor_routes(&regions[..1], &regions, &regions).unwrap_err();

    assert!(matches!(
        error,
        OperationError::SpaceMismatch {
            message: "compact left factor contains an unused nonzero sector"
        }
    ));
}

fn z2_pair(codomain: [usize; 2], domain: [usize; 2], coupled: usize) -> FusionTreePairKey {
    let pair = FusionTreePairKey::try_pair_from_sector_ids(
        codomain,
        domain,
        coupled,
        [false; 2],
        [false; 2],
        std::iter::empty::<usize>(),
        std::iter::empty::<usize>(),
        [1],
        [1],
    )
    .unwrap();
    pair.validate_for_rule(&Z2FusionRule).unwrap();
    pair
}

#[test]
fn aligned_diagonal_spectrum_admission_is_exact_and_order_independent() {
    let even = SectorId::new(0);
    let odd = SectorId::new(1);
    let leg = SectorLeg::new([(even, 2), (odd, 1)], false);
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg.clone()]),
        FusionProductSpace::new([leg]),
    );
    let authority = BoundDynamicFusionMapSpace::from_final_homspace_multiplicity_free(
        Arc::new(Z2FusionRule),
        homspace,
    )
    .unwrap();
    let regions = checked_sector_regions(authority.space().structure(), 1)
        .unwrap()
        .unwrap();
    let reordered = [
        SectorSpectrum {
            sector: odd,
            values: vec![3.0],
        },
        SectorSpectrum {
            sector: even,
            values: vec![1.0, 2.0],
        },
    ];
    let admitted = aligned_diagonal_spectrum_by_sector(&regions, &reordered).unwrap();
    assert_eq!(admitted[&even].values, [1.0, 2.0]);
    assert_eq!(admitted[&odd].values, [3.0]);
    assert!(
        aligned_diagonal_spectrum_by_sector(&regions, &reordered[..1])
            .unwrap_err()
            .contains("sector count")
    );
    let missing = [
        SectorSpectrum {
            sector: SectorId::new(9),
            values: vec![3.0],
        },
        SectorSpectrum {
            sector: even,
            values: vec![1.0, 2.0],
        },
    ];
    assert!(aligned_diagonal_spectrum_by_sector(&regions, &missing)
        .unwrap_err()
        .contains("missing"));
    let duplicate = [
        SectorSpectrum {
            sector: even,
            values: vec![1.0, 2.0],
        },
        SectorSpectrum {
            sector: even,
            values: vec![3.0, 4.0],
        },
    ];
    assert!(aligned_diagonal_spectrum_by_sector(&regions, &duplicate)
        .unwrap_err()
        .contains("twice"));
    let wrong_length = [
        SectorSpectrum {
            sector: odd,
            values: vec![3.0],
        },
        SectorSpectrum {
            sector: even,
            values: vec![1.0],
        },
    ];
    assert!(aligned_diagonal_spectrum_by_sector(&regions, &wrong_length)
        .unwrap_err()
        .contains("length"));

    let rectangular = BlockStructure::from_blocks_with_rank(
        4,
        vec![BlockSpec::with_key(
            z2_pair([0, 0], [0, 0], 0).into(),
            vec![2, 1, 1, 1],
            vec![1, 2, 2, 2],
            0,
        )
        .unwrap()],
    )
    .unwrap();
    let rectangular_regions = checked_sector_regions(&rectangular, 2).unwrap().unwrap();
    let rectangular_spectrum = [SectorSpectrum {
        sector: even,
        values: vec![1.0, 2.0],
    }];
    assert!(
        aligned_diagonal_spectrum_by_sector(&rectangular_regions, &rectangular_spectrum)
            .unwrap_err()
            .contains("aligned square")
    );

    let trees = [z2_pair([1, 1], [1, 1], 0), z2_pair([0, 0], [0, 0], 0)];
    let mut blocks = Vec::new();
    for (column, col) in trees.iter().enumerate() {
        for (row, row_tree) in trees.iter().enumerate() {
            blocks.push(
                BlockSpec::with_key(
                    FusionTreePairKey::pair(
                        row_tree.codomain_tree().clone(),
                        col.domain_tree().clone(),
                    )
                    .into(),
                    vec![1, 1, 1, 1],
                    vec![1, 1, 2, 2],
                    row + 2 * column,
                )
                .unwrap(),
            );
        }
    }
    let structure = BlockStructure::from_blocks_with_rank(4, blocks).unwrap();
    let noncanonical = checked_sector_regions(&structure, 2).unwrap().unwrap();
    assert!(noncanonical[0].row_trees()[0].tree() > noncanonical[0].row_trees()[1].tree());
    let spectrum = [SectorSpectrum {
        sector: even,
        values: vec![1.0, 2.0],
    }];
    assert!(aligned_diagonal_spectrum_by_sector(&noncanonical, &spectrum).is_ok());
}

#[test]
fn packed_and_region_geometry_preserve_dual_and_innerline_tree_identity() {
    let mut row_trees = vec![
        full_identity_pair(7, false, 11, false)
            .codomain_tree()
            .clone(),
        full_identity_pair(8, true, 11, false)
            .codomain_tree()
            .clone(),
    ];
    let mut col_trees = vec![
        full_identity_pair(7, false, 10, false)
            .domain_tree()
            .clone(),
        full_identity_pair(7, false, 11, true).domain_tree().clone(),
    ];
    row_trees.sort();
    col_trees.sort();
    assert_ne!(row_trees[0].innerlines(), row_trees[1].innerlines());
    assert_ne!(row_trees[0].is_dual(), row_trees[1].is_dual());
    assert_ne!(col_trees[0].innerlines(), col_trees[1].innerlines());
    assert_ne!(col_trees[0].is_dual(), col_trees[1].is_dual());

    let row_shapes = [vec![1, 2, 1], vec![2, 1, 1]];
    let col_shapes = [vec![1, 3, 1], vec![1, 1, 1]];
    let row_extents = row_shapes
        .iter()
        .map(|shape| shape.iter().product::<usize>())
        .collect::<Vec<_>>();
    let col_extents = col_shapes
        .iter()
        .map(|shape| shape.iter().product::<usize>())
        .collect::<Vec<_>>();
    let rows = row_extents.iter().sum::<usize>();
    let cols = col_extents.iter().sum::<usize>();
    let mut blocks = Vec::new();
    let mut col_offset = 0;
    for (col, (col_tree, col_shape)) in col_trees.iter().zip(&col_shapes).enumerate() {
        let mut row_offset = 0;
        for (row, (row_tree, row_shape)) in row_trees.iter().zip(&row_shapes).enumerate() {
            let mut shape = row_shape.clone();
            shape.extend_from_slice(col_shape);
            let mut strides = Vec::with_capacity(shape.len());
            let mut stride = 1;
            for &dimension in row_shape {
                strides.push(stride);
                stride *= dimension;
            }
            stride = rows;
            for &dimension in col_shape {
                strides.push(stride);
                stride *= dimension;
            }
            blocks.push(
                BlockSpec::with_key(
                    FusionTreePairKey::pair(row_tree.clone(), col_tree.clone()).into(),
                    shape,
                    strides,
                    row_offset + rows * col_offset,
                )
                .unwrap(),
            );
            row_offset += row_extents[row];
        }
        col_offset += col_extents[col];
    }
    let structure = BlockStructure::from_blocks_with_rank(6, blocks).unwrap();
    let data = (0..rows * cols)
        .map(|value| value as f64)
        .collect::<Vec<_>>();
    let regions = structure.coupled_sector_regions(3).unwrap().unwrap();
    let packed = sector_matricizations(&structure, &data, 3).unwrap();
    assert_eq!(regions.len(), 1);
    assert_eq!(packed.len(), 1);
    assert!(matches!(
        generic_input_matricizations(&structure, &data, 3).unwrap(),
        InputMatricizations::Regions { .. }
    ));
    let region = &regions[0];
    let matrix = &packed[0];
    assert_eq!(matrix.data, data);
    assert_eq!(region.sector(), SectorId::new(9));
    assert_eq!((region.rows(), region.cols()), (4, 4));
    assert_eq!(region.sector(), matrix.sector());
    assert_eq!(
        (region.rows(), region.cols()),
        (matrix.rows(), matrix.cols())
    );
    for side in [FactorSide::Left, FactorSide::Right] {
        let (expected_trees, expected_offsets, expected_shapes) = match side {
            FactorSide::Left => (&row_trees, [0, 2], &row_shapes),
            FactorSide::Right => (&col_trees, [0, 3], &col_shapes),
        };
        assert_eq!(region.tree_count(side), matrix.tree_count(side));
        for index in 0..region.tree_count(side) {
            let borrowed = region.tree(side, index).unwrap();
            let owned = matrix.tree(side, index).unwrap();
            assert_eq!(borrowed.tree, &expected_trees[index]);
            assert_eq!(borrowed.offset, expected_offsets[index]);
            assert_eq!(borrowed.shape, expected_shapes[index]);
            assert_eq!(owned.tree, &expected_trees[index]);
            assert_eq!(owned.offset, expected_offsets[index]);
            assert_eq!(owned.shape, expected_shapes[index]);
            assert_eq!(borrowed.tree, owned.tree);
            assert_eq!(borrowed.offset, owned.offset);
            assert_eq!(borrowed.shape, owned.shape);
        }
    }
}

#[test]
fn sector_matricizations_preserve_encounter_order_and_padded_block_values() {
    // What: noncanonical storage packs repeated row/column trees into
    // first-encounter sector geometry without changing block copy order.
    let structure = BlockStructure::from_blocks_with_rank(
        4,
        vec![
            BlockSpec::with_key(
                z2_pair([0, 1], [1, 0], 1).into(),
                vec![1, 2, 3, 1],
                vec![2, 9, 1, 20],
                3,
            )
            .unwrap(),
            BlockSpec::with_key(
                z2_pair([0, 0], [1, 1], 0).into(),
                vec![2, 1, 1, 2],
                vec![2, 20, 1, 7],
                30,
            )
            .unwrap(),
            BlockSpec::with_key(
                z2_pair([0, 0], [0, 0], 0).into(),
                vec![2, 1, 2, 1],
                vec![3, 20, 1, 50],
                50,
            )
            .unwrap(),
            BlockSpec::with_key(
                z2_pair([1, 1], [1, 1], 0).into(),
                vec![1, 2, 1, 2],
                vec![4, 1, 20, 7],
                70,
            )
            .unwrap(),
            BlockSpec::with_key(
                z2_pair([1, 1], [0, 0], 0).into(),
                vec![1, 2, 2, 1],
                vec![4, 1, 5, 20],
                90,
            )
            .unwrap(),
        ],
    )
    .unwrap();
    let mut data = vec![0.0; structure.required_len().unwrap()];
    for (position, value) in [
        (3, 101.0),
        (12, 102.0),
        (4, 103.0),
        (13, 104.0),
        (5, 105.0),
        (14, 106.0),
        (30, 1.0),
        (32, 2.0),
        (37, 5.0),
        (39, 6.0),
        (50, 3.0),
        (53, 4.0),
        (51, 7.0),
        (54, 8.0),
        (70, 9.0),
        (71, 10.0),
        (77, 13.0),
        (78, 14.0),
        (90, 11.0),
        (91, 12.0),
        (95, 15.0),
        (96, 16.0),
    ] {
        data[position] = value;
    }

    let matrices = sector_matricizations::<f64>(&structure, &data, 2).unwrap();

    assert_eq!(
        matrices
            .iter()
            .map(|matrix| matrix.sector)
            .collect::<Vec<_>>(),
        [SectorId::new(1), SectorId::new(0)]
    );
    assert_eq!(matrices[0].rows, 2);
    assert_eq!(matrices[0].cols, 3);
    assert_eq!(matrices[0].data, [101.0, 102.0, 103.0, 104.0, 105.0, 106.0]);
    assert_eq!(
        matrices[0].row_trees,
        vec![(
            z2_pair([0, 1], [1, 0], 1).codomain_tree().clone(),
            0,
            vec![1, 2],
        )]
    );
    assert_eq!(
        matrices[0].col_trees,
        vec![(
            z2_pair([0, 1], [1, 0], 1).domain_tree().clone(),
            0,
            vec![3, 1],
        )]
    );

    assert_eq!(matrices[1].rows, 4);
    assert_eq!(matrices[1].cols, 4);
    assert_eq!(
        matrices[1].data,
        [1.0, 2.0, 9.0, 10.0, 5.0, 6.0, 13.0, 14.0, 3.0, 4.0, 11.0, 12.0, 7.0, 8.0, 15.0, 16.0,]
    );
    assert_eq!(
        matrices[1].row_trees,
        vec![
            (
                z2_pair([0, 0], [1, 1], 0).codomain_tree().clone(),
                0,
                vec![2, 1],
            ),
            (
                z2_pair([1, 1], [1, 1], 0).codomain_tree().clone(),
                2,
                vec![1, 2],
            ),
        ]
    );
    assert_eq!(
        matrices[1].col_trees,
        vec![
            (
                z2_pair([0, 0], [1, 1], 0).domain_tree().clone(),
                0,
                vec![1, 2],
            ),
            (
                z2_pair([0, 0], [0, 0], 0).domain_tree().clone(),
                2,
                vec![2, 1],
            ),
        ]
    );
}

#[test]
fn generic_sector_matricizations_preserve_full_tree_identity_and_exact_layout() {
    let structure = BlockStructure::from_blocks_with_rank(
        4,
        vec![
            BlockSpec::with_key(
                generic_pair(1, 2, 1).into(),
                vec![1, 2, 2, 1],
                vec![1, 17, 3, 40],
                5,
            )
            .unwrap(),
            BlockSpec::with_key(
                generic_pair(0, 1, 1).into(),
                vec![1, 1, 1, 1],
                vec![1, 7, 5, 3],
                80,
            )
            .unwrap(),
            BlockSpec::with_key(
                generic_pair(1, 1, 2).into(),
                vec![1, 1, 1, 3],
                vec![1, 8, 2, 4],
                60,
            )
            .unwrap(),
            BlockSpec::with_key(
                generic_pair(1, 1, 1).into(),
                vec![1, 1, 2, 1],
                vec![1, 9, 5, 30],
                40,
            )
            .unwrap(),
        ],
    )
    .unwrap();
    let mut real = vec![0.0; structure.required_len().unwrap()];
    for (index, value) in [
        (5, 11.0),
        (22, 12.0),
        (8, 13.0),
        (25, 14.0),
        (80, 90.0),
        (60, 31.0),
        (64, 32.0),
        (68, 33.0),
        (40, 21.0),
        (45, 22.0),
    ] {
        real[index] = value;
    }
    let expected = [
        11.0, 12.0, 21.0, 13.0, 14.0, 22.0, 0.0, 0.0, 31.0, 0.0, 0.0, 32.0, 0.0, 0.0, 33.0,
    ];

    let matrices = sector_matricizations(&structure, &real, 2).unwrap();
    assert_eq!(
        matrices
            .iter()
            .map(|matrix| matrix.sector)
            .collect::<Vec<_>>(),
        [SectorId::new(1), SectorId::new(0)]
    );
    assert_eq!((matrices[0].rows, matrices[0].cols), (3, 5));
    assert_eq!(matrices[0].data, expected);
    assert_eq!(matrices[1].data, [90.0]);
    assert_eq!(
        matrices[0]
            .row_trees
            .iter()
            .map(|(tree, offset, shape)| (tree.vertices()[0].get(), *offset, shape.clone()))
            .collect::<Vec<_>>(),
        [(2, 0, vec![1, 2]), (1, 2, vec![1, 1])]
    );
    assert_eq!(
        matrices[0]
            .col_trees
            .iter()
            .map(|(tree, offset, shape)| (tree.vertices()[0].get(), *offset, shape.clone()))
            .collect::<Vec<_>>(),
        [(1, 0, vec![2, 1]), (2, 2, vec![1, 3])]
    );

    let complex = real
        .iter()
        .map(|&value| Complex64::new(value, -value / 10.0))
        .collect::<Vec<_>>();
    let complex_matrices = sector_matricizations(&structure, &complex, 2).unwrap();
    assert_eq!(
        complex_matrices[0].data,
        expected.map(|value| Complex64::new(value, -value / 10.0))
    );
    assert_eq!(complex_matrices[1].data, [Complex64::new(90.0, -9.0)]);

    let empty = BlockStructure::from_blocks_with_rank(4, Vec::new()).unwrap();
    assert!(sector_matricizations::<f64>(&empty, &[], 2)
        .unwrap()
        .is_empty());

    let scalar_key =
        FusionTreePairKey::try_pair_from_sector_ids([], [], 0, [], [], [], [], [], []).unwrap();
    let scalar = BlockStructure::from_blocks_with_rank(
        0,
        vec![BlockSpec::with_key(scalar_key.into(), vec![], vec![], 1).unwrap()],
    )
    .unwrap();
    assert_eq!(
        sector_matricizations(&scalar, &[0.0, 7.0], 0).unwrap()[0].data,
        [7.0]
    );

    let zero_extent = BlockStructure::from_blocks_with_rank(
        4,
        vec![BlockSpec::with_key(
            generic_pair(1, 1, 1).into(),
            vec![0, 1, 1, 1],
            vec![1, 1, 1, 1],
            0,
        )
        .unwrap()],
    )
    .unwrap();
    let zero_matrix = sector_matricizations::<f64>(&zero_extent, &[], 2).unwrap();
    assert_eq!((zero_matrix[0].rows, zero_matrix[0].cols), (0, 1));
    assert!(zero_matrix[0].data.is_empty());

    let non_fusion = BlockStructure::from_blocks_with_rank(
        1,
        vec![BlockSpec::with_key(BlockKey::opaque([7]), vec![1], vec![1], 0).unwrap()],
    )
    .unwrap();
    assert!(matches!(
        sector_matricizations::<f64>(&non_fusion, &[1.0], 1),
        Err(OperationError::ExpectedFusionTreeBlock {
            tensor: "tsvd",
            index: 0
        })
    ));
}

#[test]
fn generic_sector_matricizations_keep_tree_offsets_matrix_local() {
    // Raw block structures can pair trees whose coupled sectors differ. A
    // domain tree shared by two codomain-selected matrices therefore has
    // an independent column offset in each matrix.
    let sector_one = generic_pair(1, 1, 1);
    let sector_zero = generic_pair(0, 1, 1);
    let row_one = sector_one.codomain_tree().clone();
    let row_zero = sector_zero.codomain_tree().clone();
    let shared_domain = sector_one.domain_tree().clone();
    let other_domain = sector_zero.domain_tree().clone();
    let structure = BlockStructure::from_blocks_with_rank(
        4,
        vec![
            BlockSpec::with_key(
                FusionTreePairKey::pair(row_one.clone(), other_domain.clone()).into(),
                vec![1, 1, 1, 1],
                vec![1, 1, 1, 1],
                1,
            )
            .unwrap(),
            BlockSpec::with_key(
                FusionTreePairKey::pair(row_zero.clone(), shared_domain.clone()).into(),
                vec![1, 1, 1, 2],
                vec![1, 1, 1, 1],
                4,
            )
            .unwrap(),
            BlockSpec::with_key(
                FusionTreePairKey::pair(row_one.clone(), shared_domain.clone()).into(),
                vec![1, 1, 1, 2],
                vec![1, 1, 1, 1],
                8,
            )
            .unwrap(),
        ],
    )
    .unwrap();
    let mut data = vec![0.0; structure.required_len().unwrap()];
    data[1] = 11.0;
    data[4..6].copy_from_slice(&[21.0, 22.0]);
    data[8..10].copy_from_slice(&[31.0, 32.0]);

    let matrices = sector_matricizations(&structure, &data, 2).unwrap();

    assert_eq!(matrices.len(), 2);
    assert_eq!(matrices[0].sector, SectorId::new(1));
    assert_eq!((matrices[0].rows, matrices[0].cols), (1, 3));
    assert_eq!(matrices[0].data, [11.0, 31.0, 32.0]);
    assert_eq!(
        matrices[0]
            .col_trees
            .iter()
            .map(|(tree, offset, _)| (tree, *offset))
            .collect::<Vec<_>>(),
        [(&other_domain, 0), (&shared_domain, 1)]
    );
    assert_eq!(matrices[1].sector, SectorId::new(0));
    assert_eq!((matrices[1].rows, matrices[1].cols), (1, 2));
    assert_eq!(matrices[1].data, [21.0, 22.0]);
    assert_eq!(matrices[1].col_trees[0].0, shared_domain);
    assert_eq!(matrices[1].col_trees[0].1, 0);
}
