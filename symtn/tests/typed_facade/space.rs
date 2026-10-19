use super::*;

const VACUUM: TableSector = TableSector::new(1);
const TAU: TableSector = TableSector::new(2);

#[test]
fn fibonacci_codec_reaches_public_graded_space_construction() {
    // What: the table provider's public labels cross the ordinary typed
    // codec boundary and return in engine-sector order. Tensor execution has a
    // separate complex-coefficient admission boundary and is not claimed here.
    let space =
        GradedSpace::try_new(Arc::new(TableRule::fibonacci()), [(TAU, 3), (VACUUM, 2)]).unwrap();

    assert_eq!(space.sectors().unwrap(), vec![VACUUM, TAU]);
    assert_eq!(space.degeneracies(), &[2, 3]);
    assert!(!space.is_dual());
}

#[test]
fn fibonacci_complex_tensor_reaches_all_checked_constructors() {
    let runtime = runtime();
    let space =
        GradedSpace::try_new(Arc::new(TableRule::fibonacci()), [(VACUUM, 1), (TAU, 1)]).unwrap();

    let zeros: TensorMap<_, Complex64> = TensorMap::zeros(&runtime, [&space], [&space]).unwrap();
    assert!(zeros
        .dense_data()
        .unwrap()
        .iter()
        .all(|&value| value == Complex64::new(0.0, 0.0)));

    let filled: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&space], [&space], |trees, indices| {
            Complex64::new(
                (indices.iter().sum::<usize>() + 1) as f64,
                if trees.coupled() == &TAU { 1.0 } else { -1.0 },
            )
        })
        .unwrap();
    assert!(filled
        .dense_data()
        .unwrap()
        .iter()
        .any(|value| value.im != 0.0));

    let random: TensorMap<_, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&space], [&space], 0x9E37_79B9_7F4A_7C15).unwrap();
    let seeded: TensorMap<_, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&space], [&space], 592_301).unwrap();
    assert_eq!(
        random.dense_data().unwrap().len(),
        zeros.dense_data().unwrap().len()
    );
    assert_eq!(
        seeded.dense_data().unwrap().len(),
        zeros.dense_data().unwrap().len()
    );
    assert!(seeded
        .dense_data()
        .unwrap()
        .iter()
        .any(|value| value.im != 0.0));
}

fn fibonacci_tau_braid_fixture(runtime: &Runtime) -> TensorMap<TableRule, Complex64> {
    let tau = GradedSpace::try_new(Arc::new(TableRule::fibonacci()), [(TAU, 1)]).unwrap();
    TensorMap::from_subblock_fn(runtime, [&tau, &tau, &tau], [&tau], |trees, _| match trees
        .codomain_innerlines()
    {
        [VACUUM] => Complex64::new(1.0, 0.0),
        [TAU] => Complex64::new(0.0, 0.0),
        inner => panic!("unexpected Fibonacci fusion path {inner:?}"),
    })
    .unwrap()
}

fn fibonacci_tau_channel(tensor: &TensorMap<TableRule, Complex64>) -> [Complex64; 2] {
    let mut values = [Complex64::new(0.0, 0.0); 2];
    for (trees, block) in tensor.subblocks().unwrap() {
        if trees.coupled() != &TAU {
            continue;
        }
        let slot = match trees.codomain_innerlines() {
            [VACUUM] => 0,
            [TAU] => 1,
            inner => panic!("unexpected Fibonacci fusion path {inner:?}"),
        };
        values[slot] = *block.get(&[0, 0, 0, 0]).unwrap();
    }
    values
}

#[test]
fn fibonacci_forward_braid_matches_closed_form_and_tensorkit_fixture() {
    // Isolated: the completed-transformer counters are process-global.
    if crate::run_isolated_or_return(
        "SYMTN_TYPED_FACADE_FIBONACCI_FORWARD_BRAID_MATCHES_CLOSED_F",
        "space::fibonacci_forward_braid_matches_closed_form_and_tensorkit_fixture",
    ) {
        return;
    }
    let runtime = runtime();
    let source = fibonacci_tau_braid_fixture(&runtime);
    symtn::cache::clear();
    let braided = source.braid(&[0, 2, 1], &[3], &[0, 1, 2, 3]).unwrap();
    let cold = crate::completed_transformers();
    assert_eq!((cold.entries(), cold.misses(), cold.hits()), (1, 1, 0));
    let warm = source.braid(&[0, 2, 1], &[3], &[0, 1, 2, 3]).unwrap();
    assert_eq!(warm.dense_data().unwrap(), braided.dense_data().unwrap());
    let warm_info = crate::completed_transformers();
    assert_eq!(
        (warm_info.entries(), warm_info.misses(), warm_info.hits()),
        (1, 1, 1)
    );
    let actual = fibonacci_tau_channel(&braided);

    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let f = [
        [1.0 / phi, 1.0 / phi.sqrt()],
        [1.0 / phi.sqrt(), -1.0 / phi],
    ];
    let r = [
        Complex64::from_polar(1.0, 4.0 * std::f64::consts::PI / 5.0),
        Complex64::from_polar(1.0, -3.0 * std::f64::consts::PI / 5.0),
    ];
    let expected = [
        f[0][0] * r[0] * f[0][0] + f[0][1] * r[1] * f[1][0],
        f[1][0] * r[0] * f[0][0] + f[1][1] * r[1] * f[1][0],
    ];
    for (value, oracle) in actual.into_iter().zip(expected) {
        assert!((value - oracle).norm() < 1e-12, "{value:?} != {oracle:?}");
    }

    // TensorKit 0.17 / TensorKitSectors Fibonacci, pinned independently from
    // `treebraider` on the same all-tau, total-tau basis and braid convention.
    let tensorkit = [
        Complex64::new(-0.5, -0.3632712640026805),
        Complex64::new(-0.2429341358783228, 0.7476743906106103),
    ];
    for (value, oracle) in fibonacci_tau_channel(&braided).into_iter().zip(tensorkit) {
        assert!((value - oracle).norm() < 1e-12, "{value:?} != {oracle:?}");
    }

    let back = braided.braid(&[0, 2, 1], &[3], &[0, 2, 1, 3]).unwrap();
    for (value, oracle) in fibonacci_tau_channel(&back)
        .into_iter()
        .zip(fibonacci_tau_channel(&source))
    {
        assert!((value - oracle).norm() < 1e-12);
    }
}

#[test]
fn fibonacci_planar_transforms_roundtrip_without_braiding() {
    let runtime = runtime();
    let source = fibonacci_tau_braid_fixture(&runtime);
    assert_eq!(
        source
            .codomain()
            .iter()
            .map(GradedSpace::is_dual)
            .collect::<Vec<_>>(),
        [false, false, false]
    );
    assert_eq!(
        source
            .domain()
            .iter()
            .map(GradedSpace::is_dual)
            .collect::<Vec<_>>(),
        [false]
    );

    let repartitioned = source.repartition(2).unwrap();
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    // Pinned TensorKit `repartition` fixture; the same value is the
    // independent rigidity oracle sqrt(d_tau).
    assert_data_close_c64(
        repartitioned.dense_data().unwrap(),
        &[Complex64::new(phi.sqrt(), 0.0), Complex64::new(0.0, 0.0)],
    );
    assert_eq!(
        (repartitioned.codomain_rank(), repartitioned.domain_rank()),
        (2, 2)
    );
    assert_eq!(repartitioned.codomain()[0].sectors().unwrap(), [TAU]);
    assert_eq!(repartitioned.domain()[1].sectors().unwrap(), [TAU]);
    assert_eq!(
        repartitioned
            .codomain()
            .iter()
            .map(GradedSpace::is_dual)
            .collect::<Vec<_>>(),
        [false, false]
    );
    assert_eq!(
        repartitioned
            .domain()
            .iter()
            .map(GradedSpace::is_dual)
            .collect::<Vec<_>>(),
        [false, true]
    );
    assert_data_close_c64(
        repartitioned.repartition(3).unwrap().dense_data().unwrap(),
        source.dense_data().unwrap(),
    );

    let transposed = full_transpose!(source).unwrap();
    // Pinned TensorKit planar-transpose fixture and the closed-form first row
    // of the Fibonacci F matrix.
    assert_data_close_c64(
        transposed.dense_data().unwrap(),
        &[
            Complex64::new(1.0 / phi, 0.0),
            Complex64::new(1.0 / phi.sqrt(), 0.0),
        ],
    );
    assert_eq!(
        (transposed.codomain_rank(), transposed.domain_rank()),
        (1, 3)
    );
    assert_eq!(
        transposed
            .codomain()
            .iter()
            .map(GradedSpace::is_dual)
            .collect::<Vec<_>>(),
        [true]
    );
    assert_eq!(
        transposed
            .domain()
            .iter()
            .map(GradedSpace::is_dual)
            .collect::<Vec<_>>(),
        [true, true, true]
    );
    assert_data_close_c64(
        full_transpose!(transposed).unwrap().dense_data().unwrap(),
        source.dense_data().unwrap(),
    );

    let explicit = source.transpose(&[3], &[2, 1, 0]).unwrap();
    assert_data_close_c64(
        explicit.dense_data().unwrap(),
        transposed.dense_data().unwrap(),
    );
    for (trees, _) in explicit.subblocks().unwrap() {
        assert_eq!(trees.codomain_uncoupled(), &[TAU]);
        assert_eq!(trees.domain_uncoupled(), &[TAU; 3]);
    }
}

#[test]
fn fibonacci_otimes_matches_the_nontrivial_tensorkit_fixture() {
    let runtime = runtime();
    let tau = GradedSpace::try_new(Arc::new(TableRule::fibonacci()), [(TAU, 1)]).unwrap();
    let dual_tau = tau.try_dual().unwrap();
    let alpha = Complex64::new(2.0, 3.0);
    let beta = Complex64::new(-1.0, 2.0);
    let gamma = alpha * beta;
    assert_eq!(gamma, Complex64::new(-8.0, 1.0));

    let lhs: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&tau], [&tau], |_, _| alpha).unwrap();
    let rhs: TensorMap<_, Complex64> = TensorMap::from_subblock_fn(
        &runtime,
        [&dual_tau, &tau],
        [&dual_tau, &tau],
        |trees, _| {
            if trees.coupled() == &TAU {
                beta
            } else {
                Complex64::new(0.0, 0.0)
            }
        },
    )
    .unwrap();
    let product = lhs.otimes(&rhs).unwrap();

    assert_eq!(
        product
            .codomain()
            .iter()
            .map(GradedSpace::is_dual)
            .collect::<Vec<_>>(),
        [false, true, false]
    );
    assert_eq!(
        product
            .domain()
            .iter()
            .map(GradedSpace::is_dual)
            .collect::<Vec<_>>(),
        [false, true, false]
    );

    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let tau_oracle = [
        [1.0 / phi, -1.0 / (phi * phi.sqrt())],
        [-1.0 / (phi * phi.sqrt()), 1.0 / phi.powi(2)],
    ];
    let mut seen = [[false; 2]; 2];
    let mut vacuum = 0;
    for (trees, block) in product.subblocks().unwrap() {
        let value = *block.get(&[0; 6]).unwrap();
        if trees.coupled() == &VACUUM {
            vacuum += 1;
            assert_data_close_c64(&[value], &[gamma]);
            continue;
        }
        let slot = |innerlines: &[TableSector]| match innerlines {
            [VACUUM] => 0,
            [TAU] => 1,
            innerlines => panic!("unexpected Fibonacci innerlines {innerlines:?}"),
        };
        let (row, column) = (
            slot(trees.codomain_innerlines()),
            slot(trees.domain_innerlines()),
        );
        seen[row][column] = true;
        assert_data_close_c64(&[value], &[gamma * tau_oracle[row][column]]);
    }
    assert_eq!(vacuum, 1);
    assert_eq!(seen, [[true; 2]; 2]);
}

fn fibonacci_matrix_entry(seed: f64, sector: &TableSector, row: usize, column: usize) -> Complex64 {
    let channel = match *sector {
        VACUUM => 0.0,
        TAU => 7.0,
        other => panic!("not a Fibonacci sector: {other:?}"),
    };
    Complex64::new(
        seed + channel + 2.0 * row as f64 - column as f64,
        0.5 * seed - channel + row as f64 + 3.0 * column as f64,
    )
}

#[test]
fn fibonacci_compose_is_complex_coupled_sector_matrix_multiplication() {
    // Isolated: the completed-transformer counters are process-global.
    if crate::run_isolated_or_return(
        "SYMTN_TYPED_FACADE_FIBONACCI_COMPOSE_IS_COMPLEX_COUPLED_SEC",
        "space::fibonacci_compose_is_complex_coupled_sector_matrix_multiplication",
    ) {
        return;
    }
    // What: fixed-boundary Fibonacci composition multiplies every coupled
    // sector block, preserves left authority, and is identical cold and warm.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(TableRule::fibonacci());
    let rows = GradedSpace::try_new(Arc::clone(&provider), [(VACUUM, 1), (TAU, 2)]).unwrap();
    let shared = GradedSpace::try_new(Arc::clone(&provider), [(VACUUM, 2), (TAU, 3)]).unwrap();
    let columns = GradedSpace::try_new(Arc::clone(&provider), [(VACUUM, 2), (TAU, 1)]).unwrap();
    let lhs: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&rows], [&shared], |trees, indices| {
            fibonacci_matrix_entry(1.0, trees.coupled(), indices[0], indices[1])
        })
        .unwrap();
    let rhs: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&shared], [&columns], |trees, indices| {
            fibonacci_matrix_entry(5.0, trees.coupled(), indices[0], indices[1])
        })
        .unwrap();
    symtn::cache::clear();
    let before = crate::completed_transformers();
    let cold = lhs.compose(&rhs).unwrap();
    let after_cold = crate::completed_transformers();
    let warm = lhs.compose(&rhs).unwrap();
    let expected: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&rows], [&columns], |trees, indices| {
            let inner = match *trees.coupled() {
                VACUUM => 2,
                TAU => 3,
                other => panic!("not a Fibonacci sector: {other:?}"),
            };
            (0..inner)
                .map(|k| {
                    fibonacci_matrix_entry(1.0, trees.coupled(), indices[0], k)
                        * fibonacci_matrix_entry(5.0, trees.coupled(), k, indices[1])
                })
                .sum()
        })
        .unwrap();

    assert_data_close_c64(cold.dense_data().unwrap(), expected.dense_data().unwrap());
    assert_data_close_c64(warm.dense_data().unwrap(), expected.dense_data().unwrap());
    assert!(std::ptr::eq(cold.provider(), lhs.provider()));
    assert_eq!(after_cold, before);
    assert_eq!(crate::completed_transformers(), before);

    // What: the same genuinely complex blocks are associative, while ordinary
    // contraction refuses even this crossing-free boundary.
    let matrix = |seed| {
        TensorMap::from_subblock_fn(&runtime, [&shared], [&shared], |trees, indices| {
            fibonacci_matrix_entry(seed, trees.coupled(), indices[0], indices[1])
        })
        .unwrap()
    };
    let (a, b, c): (_, _, TensorMap<_, Complex64>) = (matrix(1.0), matrix(3.0), matrix(8.0));

    let left = a.compose(&b).unwrap().compose(&c).unwrap();
    let right = a.compose(&b.compose(&c).unwrap()).unwrap();
    assert_data_close_c64(left.dense_data().unwrap(), right.dense_data().unwrap());

    symtn::cache::clear();
    let before = crate::completed_transformers();
    let error = a
        .contract(
            &b,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap_err();
    assert!(matches!(
        error,
        symtn::typed::Error::Operation(operation)
            if matches!(*operation, symtn::typed::OperationError::UnsupportedTensorContractScope { .. })
    ));
    assert_eq!(crate::completed_transformers(), before);

    let other_runtime = Runtime::builder().build().unwrap();
    let other: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&other_runtime, [&shared], [&shared], |trees, indices| {
            fibonacci_matrix_entry(2.0, trees.coupled(), indices[0], indices[1])
        })
        .unwrap();
    assert!(matches!(
        a.compose(&other).unwrap_err(),
        symtn::typed::Error::RuntimeMismatch
    ));
    assert!(matches!(
        a.contract(
            &other,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1]
            }
        )
        .unwrap_err(),
        symtn::typed::Error::RuntimeMismatch
    ));
}

#[test]
fn fibonacci_ordinary_permute_rejection_does_not_publish_a_cache_entry() {
    // Isolated: the completed-transformer counters are process-global.
    if crate::run_isolated_or_return(
        "SYMTN_TYPED_FACADE_FIBONACCI_ORDINARY_PERMUTE_REJECTION_DOE",
        "space::fibonacci_ordinary_permute_rejection_does_not_publish_a_cache_entry",
    ) {
        return;
    }
    let runtime = runtime();
    let source = fibonacci_tau_braid_fixture(&runtime);
    symtn::cache::clear();
    let before = crate::completed_transformers();
    let error = source.permute(&[0, 2, 1], &[3]).unwrap_err();
    assert!(
        matches!(
            crate::report::cause::<symtn::typed::OperationError>(&error),
            Some(symtn::typed::OperationError::UnsupportedBraidingStyle { .. })
        ),
        "{error:?}"
    );
    assert_eq!(crate::completed_transformers(), before);
}

#[test]
fn graded_space_drops_zero_degeneracy_sectors() {
    // What: the leg invariant (a zero-degeneracy sector is absent) reaches the
    // typed surface unchanged.
    let provider = Arc::new(ExternalZ3::new());
    let space = GradedSpace::try_new(provider, [(Z3Charge(0), 2), (Z3Charge(1), 0)]).unwrap();

    assert_eq!(space.sectors().unwrap(), vec![Z3Charge(0)]);
    assert_eq!(space.degeneracies(), &[2]);
}

#[test]
fn graded_space_constructor_queries_and_algebra_match_tensorkit() {
    fn generic_dim<R>(space: &GradedSpace<R>) -> f64
    where
        R: symtn::sector::TypedSectorAdmission,
        R::Mode: symtn::typed::TypedSpaceModeDispatch<R>,
    {
        space.dim().ok().unwrap()
    }

    let ordinary = GradedSpace::<symtn::sector::U1FusionRule>::try_new(
        Arc::new(symtn::sector::U1FusionRule),
        [(symtn::sector::U1Irrep::new(0), 1)],
    )
    .unwrap();
    assert_eq!(
        ordinary.sectors().unwrap(),
        [symtn::sector::U1Irrep::new(0)]
    );

    let provider = Arc::new(symtn::sector::U1FusionRule);
    let shared =
        GradedSpace::try_new(Arc::clone(&provider), [(symtn::sector::U1Irrep::new(0), 1)]).unwrap();
    assert!(std::ptr::eq(shared.provider(), provider.as_ref()));
    assert_eq!(shared.sectors().unwrap(), ordinary.sectors().unwrap());
    assert_eq!(shared.degeneracies(), ordinary.degeneracies());
    assert_eq!(shared.is_dual(), ordinary.is_dual());
    let dual = GradedSpace::try_new(Arc::clone(&provider), [(symtn::sector::U1Irrep::new(1), 2)])
        .and_then(|space| space.try_dual())
        .unwrap();
    assert!(std::ptr::eq(dual.provider(), provider.as_ref()));
    assert_eq!(dual.sectors().unwrap(), [symtn::sector::U1Irrep::new(-1)]);
    assert_eq!(
        dual.degeneracy(&symtn::sector::U1Irrep::new(-1)).unwrap(),
        2
    );
    assert_eq!(dual.degeneracy(&symtn::sector::U1Irrep::new(1)).unwrap(), 0);
    let back = dual.try_dual().unwrap();
    assert!(!back.is_dual());
    assert_eq!(back.sectors().unwrap(), [symtn::sector::U1Irrep::new(1)]);
    assert_eq!(
        back.try_dual().unwrap().sectors().unwrap(),
        dual.sectors().unwrap()
    );

    let unit = dual.unitspace().unwrap();
    assert!(std::ptr::eq(unit.provider(), provider.as_ref()));
    assert!(!unit.is_dual());
    assert_eq!(unit.sectors().unwrap(), [symtn::sector::U1Irrep::new(0)]);
    assert_eq!(unit.degeneracies(), [1]);

    let left = GradedSpace::try_new(
        Arc::clone(&provider),
        [
            (symtn::sector::U1Irrep::new(0), 1),
            (symtn::sector::U1Irrep::new(1), 2),
        ],
    )
    .unwrap();
    let right = GradedSpace::try_new(
        Arc::clone(&provider),
        [(symtn::sector::U1Irrep::new(-1), 3)],
    )
    .unwrap();
    let fused = left.fuse(&right).unwrap();
    assert_eq!(
        fused.degeneracy(&symtn::sector::U1Irrep::new(-1)).unwrap(),
        3
    );
    assert_eq!(
        fused.degeneracy(&symtn::sector::U1Irrep::new(0)).unwrap(),
        6
    );
    assert!(!fused.is_dual());
    assert_eq!(generic_dim(&left), 3.0);

    let summed = left.oplus(&right).unwrap();
    assert_eq!(
        summed.degeneracy(&symtn::sector::U1Irrep::new(-1)).unwrap(),
        3
    );
    assert_eq!(
        summed.degeneracy(&symtn::sector::U1Irrep::new(0)).unwrap(),
        1
    );
    assert_eq!(
        summed.degeneracy(&symtn::sector::U1Irrep::new(1)).unwrap(),
        2
    );
    assert!(left.oplus(&right.try_dual().unwrap()).is_err());
    let huge = GradedSpace::try_new(
        Arc::clone(&provider),
        [(symtn::sector::U1Irrep::new(0), usize::MAX)],
    )
    .unwrap();
    assert!(huge.oplus(&ordinary).is_err());
    assert!(huge.fuse(&left).is_err());

    let su2_provider = Arc::new(SU2FusionRule);
    let half = SU2Irrep::from_twice_spin(1);
    let su2_left = GradedSpace::try_new(Arc::clone(&su2_provider), [(half, 2)])
        .and_then(|space| space.try_dual())
        .unwrap();
    let su2_right = GradedSpace::try_new(Arc::clone(&su2_provider), [(half, 3)]).unwrap();
    assert_eq!(su2_left.sectors().unwrap(), [half]);
    assert!(su2_left.is_dual());
    assert_eq!(su2_left.dim().unwrap(), 4.0);
    let su2_fused = su2_left.fuse(&su2_right).unwrap();
    assert_eq!(
        su2_fused.degeneracy(&SU2Irrep::from_twice_spin(0)).unwrap(),
        6
    );
    assert_eq!(
        su2_fused.degeneracy(&SU2Irrep::from_twice_spin(2)).unwrap(),
        6
    );

    let z2_provider = Arc::new(symtn::sector::Z2FusionRule);
    let z2_dual = GradedSpace::try_new(z2_provider, [(symtn::sector::Z2Irrep::ODD, 1)])
        .and_then(|space| space.try_dual())
        .unwrap();
    assert_eq!(z2_dual.sectors().unwrap(), [symtn::sector::Z2Irrep::ODD]);
    assert!(z2_dual.is_dual());

    let product_provider =
        Arc::new(symtn::sector::U1FusionRule.product(symtn::sector::FermionParityFusionRule));
    let product_left = GradedSpace::try_new(
        Arc::clone(&product_provider),
        [(
            symtn::sector::product_sector(
                symtn::sector::U1Irrep::new(1),
                symtn::sector::Z2Irrep::ODD,
            ),
            2,
        )],
    )
    .unwrap();
    let product_right = GradedSpace::try_new(
        product_provider,
        [(
            symtn::sector::product_sector(
                symtn::sector::U1Irrep::new(-1),
                symtn::sector::Z2Irrep::ODD,
            ),
            3,
        )],
    )
    .unwrap();
    let product_fused = product_left.fuse(&product_right).unwrap();
    assert_eq!(
        product_fused
            .degeneracy(&symtn::sector::product_sector(
                symtn::sector::U1Irrep::new(0),
                symtn::sector::Z2Irrep::EVEN,
            ))
            .unwrap(),
        6
    );
}

#[test]
fn graded_space_rejects_a_duplicate_label_by_name() {
    // What: the duplicate is reported as the caller's own label, which needs
    // the check to run before the label is encoded away into a `SectorId`.
    let provider = Arc::new(ExternalZ3::new());
    let error = GradedSpace::try_new(provider, [(Z3Charge(1), 2), (Z3Charge(1), 3)]).unwrap_err();

    let message = error.to_string();
    assert!(message.contains("Z3Charge(1)"), "{message}");
    assert!(message.contains("more than once"), "{message}");
}

#[test]
fn graded_space_reports_aliased_labels_as_a_codec_law_violation() {
    // What: two distinct labels encoding to one id is the provider breaking
    // codec injectivity, not the caller declaring a sector twice, and the two
    // cases must not be conflated in the diagnosis.
    let provider = Arc::new(ExternalZ3::with(Quirk::AliasLabels));
    let error = GradedSpace::try_new(provider, [(Z3Charge(0), 2), (Z3Charge(1), 3)]).unwrap_err();

    let message = error.to_string();
    assert!(message.contains("SectorCodec"), "{message}");
    assert!(message.contains("Z3Charge(0)"), "{message}");
    assert!(message.contains("Z3Charge(1)"), "{message}");
}

#[test]
fn graded_space_reports_an_unrepresentable_label() {
    // What: an out-of-domain label surfaces the provider's own encode error.
    let provider = Arc::new(ExternalZ3::new());
    let error = GradedSpace::try_new(provider, [(Z3Charge(7), 2)]).unwrap_err();

    assert!(error.report().contains("Z3 charge 7"), "{error}");
    let provider = Arc::new(ExternalZ3::new());
    let space = z3_leg(&provider, false);
    assert!(space.degeneracy(&Z3Charge(7)).is_err());
}

#[test]
fn graded_space_dual_flips_the_leg_and_dualizes_non_self_dual_labels() {
    // What: `try_dual` uses the provider's checked dual, so Z3's charge 1 and
    // charge 2 swap (with their degeneracies) and the dual flag flips.
    let provider = Arc::new(ExternalZ3::new());
    let space = z3_leg(&provider, false);
    let dual = space.try_dual().unwrap();

    assert!(dual.is_dual());
    assert_eq!(
        dual.sectors().unwrap(),
        vec![Z3Charge(0), Z3Charge(1), Z3Charge(2)]
    );
    // Charge 1 (degeneracy 3) became charge 2 and vice versa.
    assert_eq!(dual.degeneracies(), &[2, 1, 3]);
    assert_eq!(
        dual.try_dual().unwrap().degeneracies(),
        space.degeneracies()
    );
}

#[test]
fn graded_space_dual_surfaces_a_failing_checked_dual() {
    // What: a provider that cannot dual a sector reports its own typed error
    // rather than producing a partially dualized leg.
    let provider = Arc::new(ExternalZ3::with(Quirk::FailDual));
    let space = z3_leg(&provider, false);

    assert!(space.try_dual().is_err());
}

#[test]
fn graded_space_dual_reports_a_non_injective_dual_instead_of_panicking() {
    // What: a provider whose dual collapses two sectors onto one id is a
    // broken rigidity structure. The leg cannot hold the result, and the
    // failure must come back as a typed error, not a panic inside this
    // `Result`-returning API.
    let provider = Arc::new(ExternalZ3::with(Quirk::CollapsingDual));
    let space = z3_leg(&provider, false);

    let error = space.try_dual().unwrap_err();

    assert!(error.to_string().contains("not injective"), "{error}");
    assert!(
        GradedSpace::try_new(provider, [(Z3Charge(0), 1), (Z3Charge(1), 1)])
            .and_then(|space| space.try_dual())
            .is_err()
    );
}

#[test]
fn graded_space_carries_a_simple_fusion_provider_too() {
    // Isolated: the completed-transformer counters are process-global.
    if crate::run_isolated_or_return(
        "SYMTN_TYPED_FACADE_GRADED_SPACE_CARRIES_A_SIMPLE_FUSION_PRO",
        "space::graded_space_carries_a_simple_fusion_provider_too",
    ) {
        return;
    }
    // What: nothing in the typed space is abelian-specific.
    let provider = Arc::new(ExternalSu2);
    let space = su2_leg(&provider, true);

    assert_eq!(space.sectors().unwrap(), vec![SU2Irrep::from_twice_spin(1)]);
    assert!(space.is_dual());
    assert_eq!(crate::completed_transformers().entries(), 0);
}

// ---------------------------------------------------------------------------
// `TensorMap<R, D>` ownership and `zeros`.
// ---------------------------------------------------------------------------

#[test]
fn graded_space_reports_labels_in_provider_sector_id_order() {
    // What: `sectors()` decodes back to the caller's labels, ordered by the
    // provider's sector id (not by label order), with degeneracies parallel.
    let provider = Arc::new(ExternalZ3::new());
    let space = GradedSpace::try_new(
        Arc::clone(&provider),
        [(Z3Charge(2), 1), (Z3Charge(0), 2), (Z3Charge(1), 3)],
    )
    .unwrap();

    assert_eq!(
        space.sectors().unwrap(),
        vec![Z3Charge(0), Z3Charge(1), Z3Charge(2)]
    );
    assert_eq!(space.degeneracies(), &[2, 3, 1]);
    assert!(!space.is_dual());
}
