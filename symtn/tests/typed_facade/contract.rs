use super::*;

#[test]
fn contract_matches_a_hand_computed_product_with_a_reordered_output() {
    // What: a 2x3 by 3x4 contraction with `output_axes = [1, 0]` is the
    // transpose of the matrix product, element for element. The values are the
    // ones the expert-layer helper is pinned to, so the facade is proved to
    // pass the axes and the output order through unaltered.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalZ3::new());
    let (rows, shared, columns) = (
        z3_dense_leg(&provider, 2),
        z3_dense_leg(&provider, 3),
        z3_dense_leg(&provider, 4),
    );
    let lhs = counting_z3(&runtime, &rows, &shared, 1.0);
    let rhs = counting_z3(&runtime, &shared, &columns, 7.0);
    assert_eq!(lhs.dense_data().unwrap(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(rhs.dense_data().unwrap().len(), 12);

    let contracted = lhs
        .contract(
            &rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[1],
                domain: &[0],
            },
        )
        .unwrap();

    assert_eq!(
        contracted.dense_data().unwrap(),
        [76.0, 103.0, 130.0, 157.0, 100.0, 136.0, 172.0, 208.0]
    );
    assert_eq!(contracted.codomain().len(), 1);
    assert_eq!(contracted.domain().len(), 1);
}

#[test]
fn contract_with_the_default_output_order_keeps_the_open_axes_in_place() {
    // What: `0..open_rank` is the identity output order, so the same product
    // comes back untransposed.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalZ3::new());
    let (rows, shared, columns) = (
        z3_dense_leg(&provider, 2),
        z3_dense_leg(&provider, 3),
        z3_dense_leg(&provider, 4),
    );
    let lhs = counting_z3(&runtime, &rows, &shared, 1.0);
    let rhs = counting_z3(&runtime, &shared, &columns, 7.0);

    let contracted = lhs
        .contract(
            &rhs,
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
        [76.0, 100.0, 103.0, 136.0, 130.0, 172.0, 157.0, 208.0]
    );
}

#[test]
fn contract_carries_a_simple_fusion_provider_with_a_complex_payload() {
    // What: the contraction seam is neither abelian- nor real-specific.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalSu2);
    let leg = su2_leg(&provider, false);
    let mut next = Complex64::new(0.0, 0.0);
    let build = |next: &mut Complex64| {
        let mut step = *next;
        let tensor: TensorMap<ExternalSu2, Complex64> =
            TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], |_, _| {
                step += Complex64::new(1.0, 0.5);
                step
            })
            .unwrap();
        *next = step;
        tensor
    };
    let lhs = build(&mut next);
    let rhs = build(&mut next);

    // Two spin-1/2 legs on each side, so both operands carry the spin-0 and
    // spin-1 blocks; the pair of contracted legs makes the result rank 4.
    let contracted = lhs
        .contract(
            &rhs,
            &ContractSpec {
                lhs: &[2, 3],
                rhs: &[0, 1],
                codomain: &[2, 0],
                domain: &[3, 1],
            },
        )
        .unwrap();

    assert_eq!(contracted.codomain().len() + contracted.domain().len(), 4);
    assert!(contracted
        .dense_data()
        .unwrap()
        .iter()
        .any(|value| *value != Complex64::new(0.0, 0.0)));
}

#[test]
fn contract_rejects_operands_from_different_runtimes() {
    // What: the runtime is a trust boundary — two runtimes own separate
    // execution state, so mixing them is refused before any provider work.
    let _guard = cache_lock();
    let first = runtime();
    let second = runtime();
    let provider = Arc::new(ExternalZ3::new());
    let (rows, shared, columns) = (
        z3_dense_leg(&provider, 2),
        z3_dense_leg(&provider, 3),
        z3_dense_leg(&provider, 4),
    );
    let lhs = counting_z3(&first, &rows, &shared, 1.0);
    let rhs = counting_z3(&second, &shared, &columns, 1.0);

    let error = lhs
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

    assert!(matches!(error, symtn::typed::Error::RuntimeMismatch));
}

#[test]
fn contract_accepts_separately_allocated_equal_identity_providers() {
    // What: the counterpart of the distinct-identity rejection — two
    // independent allocations of one rule interoperate in an operation, not
    // just in construction, and produce the same values a single allocation
    // does.
    let _guard = cache_lock();
    let runtime = runtime();
    let first = Arc::new(ExternalZ3::new());
    let second = Arc::new(ExternalZ3::new());
    assert!(!Arc::ptr_eq(&first, &second));
    let lhs = counting_z3(
        &runtime,
        &z3_dense_leg(&first, 2),
        &z3_dense_leg(&first, 3),
        1.0,
    );
    let rhs = counting_z3(
        &runtime,
        &z3_dense_leg(&second, 3),
        &z3_dense_leg(&second, 4),
        7.0,
    );

    let contracted = lhs
        .contract(
            &rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[1],
                domain: &[0],
            },
        )
        .unwrap();

    assert_eq!(
        contracted.dense_data().unwrap(),
        [76.0, 103.0, 130.0, 157.0, 100.0, 136.0, 172.0, 208.0]
    );
}

#[test]
fn contract_rejects_operands_with_distinct_rule_identities() {
    // What: two providers of one Rust type but different identities are a
    // different algebra. The rejection comes from the expert layer, which is
    // why the facade does not pre-check it.
    let _guard = cache_lock();
    let runtime = runtime();
    let first = Arc::new(ExternalZ3::tagged(0));
    let second = Arc::new(ExternalZ3::tagged(1));
    let lhs = counting_z3(
        &runtime,
        &z3_dense_leg(&first, 2),
        &z3_dense_leg(&first, 3),
        1.0,
    );
    let rhs = counting_z3(
        &runtime,
        &z3_dense_leg(&second, 3),
        &z3_dense_leg(&second, 4),
        1.0,
    );

    assert!(lhs
        .contract(
            &rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1]
            }
        )
        .is_err());
}

#[test]
fn contract_rejects_malformed_axes_without_panicking() {
    // What: mismatched axis-list lengths, out-of-range axes and a wrong-length
    // output order all come back as `Err`.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalZ3::new());
    let (rows, shared, columns) = (
        z3_dense_leg(&provider, 2),
        z3_dense_leg(&provider, 3),
        z3_dense_leg(&provider, 4),
    );
    let lhs = counting_z3(&runtime, &rows, &shared, 1.0);
    let rhs = counting_z3(&runtime, &shared, &columns, 1.0);

    assert!(lhs
        .contract(
            &rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[],
                codomain: &[0],
                domain: &[1]
            }
        )
        .is_err());
    assert!(lhs
        .contract(
            &rhs,
            &ContractSpec {
                lhs: &[9],
                rhs: &[0],
                codomain: &[0],
                domain: &[1]
            }
        )
        .is_err());
    assert!(lhs
        .contract(
            &rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[9],
                codomain: &[0],
                domain: &[1]
            }
        )
        .is_err());
    assert!(lhs
        .contract(
            &rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[]
            }
        )
        .is_err());
}

#[test]
fn otimes_rejects_runtime_and_rule_identity_mismatches() {
    let _guard = cache_lock();
    let first = runtime();
    let second = runtime();
    let provider = Arc::new(ExternalZ3::new());
    let lhs = counting_z3(
        &first,
        &z3_dense_leg(&provider, 2),
        &z3_dense_leg(&provider, 3),
        1.0,
    );
    let other_runtime = counting_z3(
        &second,
        &z3_dense_leg(&provider, 2),
        &z3_dense_leg(&provider, 3),
        1.0,
    );
    assert!(matches!(
        lhs.otimes(&other_runtime).unwrap_err(),
        symtn::typed::Error::RuntimeMismatch
    ));

    let other_rule = Arc::new(ExternalZ3::tagged(1));
    let other_rule = counting_z3(
        &first,
        &z3_dense_leg(&other_rule, 2),
        &z3_dense_leg(&other_rule, 3),
        1.0,
    );
    assert!(lhs.otimes(&other_rule).is_err());
}

#[test]
fn otimes_matches_tensorkit_planar_trivial_without_requesting_braiding() {
    // What: the #595 NoBraiding oracle is TensorKit's own PlanarTrivial
    // category. The previous contract-plus-output-permute route reached its
    // NoBraiding boundary; the monoidal merge succeeds without an R symbol.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(PlanarTrivial);
    let lhs_cod = GradedSpace::try_new(Arc::clone(&provider), [(PlanarTrivialSector, 2)]).unwrap();
    let lhs_dom = GradedSpace::try_new(Arc::clone(&provider), [(PlanarTrivialSector, 3)])
        .and_then(|space| space.try_dual())
        .unwrap();
    let rhs_cod = GradedSpace::try_new(Arc::clone(&provider), [(PlanarTrivialSector, 4)])
        .and_then(|space| space.try_dual())
        .unwrap();
    let rhs_dom = GradedSpace::try_new(Arc::clone(&provider), [(PlanarTrivialSector, 2)]).unwrap();
    let lhs: TensorMap<PlanarTrivial, f64> =
        TensorMap::from_subblock_fn(&runtime, [&lhs_cod], [&lhs_dom], |_, indices| {
            (1 + indices[0] + 10 * indices[1]) as f64
        })
        .unwrap();
    let rhs: TensorMap<PlanarTrivial, f64> =
        TensorMap::from_subblock_fn(&runtime, [&rhs_cod], [&rhs_dom], |_, indices| {
            (2 + indices[0] + 10 * indices[1]) as f64
        })
        .unwrap();
    let expected: TensorMap<PlanarTrivial, f64> = TensorMap::from_subblock_fn(
        &runtime,
        [&lhs_cod, &rhs_cod],
        [&lhs_dom, &rhs_dom],
        |_, indices| {
            (1 + indices[0] + 10 * indices[2]) as f64 * (2 + indices[1] + 10 * indices[3]) as f64
        },
    )
    .unwrap();

    // The pre-#595 lowering encoded the same operation as empty-axis
    // contraction plus this interleaving output permutation. NoBraiding still
    // rejects that braid-requiring route.
    assert!(lhs
        .contract(
            &rhs,
            &ContractSpec {
                lhs: &[],
                rhs: &[],
                codomain: &[0, 2],
                domain: &[1, 3]
            }
        )
        .is_err());

    let actual = lhs.otimes(&rhs).unwrap();

    assert_eq!(actual.dense_data().unwrap(), expected.dense_data().unwrap());
    assert_eq!(
        actual
            .codomain()
            .into_iter()
            .chain(actual.domain())
            .map(|space| space.is_dual())
            .collect::<Vec<_>>(),
        [false, true, true, false]
    );
}

#[test]
fn otimes_fz2_complex_oracle_has_no_crossing_phase() {
    // Built-in TensorKit-equivalent FermionParity semantics: complex payload
    // multiplication is sign-free because otimes performs no leg crossing.
    let _guard = cache_lock();
    let runtime = runtime();
    let rule = Arc::new(symtn::sector::FermionParityFusionRule);
    let leg = GradedSpace::try_new(
        Arc::clone(&rule),
        [
            (symtn::sector::Z2Irrep::EVEN, 1),
            (symtn::sector::Z2Irrep::ODD, 1),
        ],
    )
    .unwrap();
    let lhs: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |sectors, _| {
            if *sectors.coupled() == symtn::sector::Z2Irrep::EVEN {
                Complex64::new(2.0, 1.0)
            } else {
                Complex64::new(-3.0, 2.0)
            }
        })
        .unwrap();
    let rhs: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |sectors, _| {
            if *sectors.coupled() == symtn::sector::Z2Irrep::EVEN {
                Complex64::new(5.0, -1.0)
            } else {
                Complex64::new(1.0, 4.0)
            }
        })
        .unwrap();
    let expected: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], |sectors, _| {
            let codomain = sectors.codomain_uncoupled();
            let domain = sectors.domain_uncoupled();
            if codomain != domain {
                return Complex64::new(0.0, 0.0);
            }
            let lhs = if codomain[0] == symtn::sector::Z2Irrep::EVEN {
                Complex64::new(2.0, 1.0)
            } else {
                Complex64::new(-3.0, 2.0)
            };
            let rhs = if codomain[1] == symtn::sector::Z2Irrep::EVEN {
                Complex64::new(5.0, -1.0)
            } else {
                Complex64::new(1.0, 4.0)
            };
            lhs * rhs
        })
        .unwrap();

    assert_eq!(
        lhs.otimes(&rhs).unwrap().dense_data().unwrap(),
        expected.dense_data().unwrap()
    );
}

#[test]
fn typed_deligne_product_uses_the_explicit_component_order() {
    let _guard = cache_lock();
    let runtime = runtime();
    let u1_rule = Arc::new(symtn::sector::U1FusionRule);
    let fz2_rule = Arc::new(symtn::sector::FermionParityFusionRule);
    let charge =
        GradedSpace::try_new(Arc::clone(&u1_rule), [(symtn::sector::U1Irrep::new(1), 1)]).unwrap();
    let parity =
        GradedSpace::try_new(Arc::clone(&fz2_rule), [(symtn::sector::Z2Irrep::ODD, 1)]).unwrap();
    let lhs = TensorMap::from_subblock_fn(&runtime, [&charge], [&charge], |_, _| 2.0).unwrap();
    let rhs = TensorMap::from_subblock_fn(&runtime, [&parity], [&parity], |_, _| 3.0).unwrap();
    let product =
        Arc::new(symtn::sector::U1FusionRule.product(symtn::sector::FermionParityFusionRule));

    let result = lhs.deligne_product(&rhs, product).unwrap();

    assert_eq!((result.codomain_rank(), result.domain_rank()), (2, 2));
    assert_eq!(result.dense_data().unwrap(), [6.0]);
    let codomain = result.codomain();
    assert_eq!(
        codomain[0].sectors().unwrap(),
        [symtn::sector::product_sector(
            symtn::sector::U1Irrep::new(1),
            symtn::sector::Z2Irrep::EVEN
        )]
    );
    assert_eq!(
        codomain[1].sectors().unwrap(),
        [symtn::sector::product_sector(
            symtn::sector::U1Irrep::new(0),
            symtn::sector::Z2Irrep::ODD
        )]
    );
}

#[test]
fn typed_deligne_product_rejects_a_component_identity_mismatch() {
    let _guard = cache_lock();
    let runtime = runtime();
    let lhs_rule = Arc::new(ExternalZ3::tagged(0));
    let lhs = counting_z3(
        &runtime,
        &z3_dense_leg(&lhs_rule, 1),
        &z3_dense_leg(&lhs_rule, 1),
        2.0,
    );
    let u1_rule = Arc::new(symtn::sector::U1FusionRule);
    let u1 =
        GradedSpace::try_new(Arc::clone(&u1_rule), [(symtn::sector::U1Irrep::new(0), 1)]).unwrap();
    let rhs = TensorMap::from_subblock_fn(&runtime, [&u1], [&u1], |_, _| 3.0).unwrap();
    let wrong = Arc::new(ExternalZ3::tagged(1).product(symtn::sector::U1FusionRule));

    assert!(matches!(
        lhs.deligne_product(&rhs, wrong).unwrap_err(),
        symtn::typed::Error::RuleMismatch
    ));
}

#[test]
fn typed_deligne_product_checks_runtime_before_both_component_identities() {
    let _guard = cache_lock();
    let first = runtime();
    let second = runtime();
    let left_rule = Arc::new(ExternalZ3::tagged(0));
    let right_rule = Arc::new(ExternalZ3::tagged(2));
    let lhs = counting_z3(
        &first,
        &z3_dense_leg(&left_rule, 1),
        &z3_dense_leg(&left_rule, 1),
        2.0,
    );
    let rhs = counting_z3(
        &second,
        &z3_dense_leg(&right_rule, 1),
        &z3_dense_leg(&right_rule, 1),
        3.0,
    );
    let wrong_both = Arc::new(ExternalZ3::tagged(1).product(ExternalZ3::tagged(3)));

    assert!(matches!(
        lhs.deligne_product(&rhs, wrong_both).unwrap_err(),
        symtn::typed::Error::RuntimeMismatch
    ));

    let rhs = counting_z3(
        &first,
        &z3_dense_leg(&right_rule, 1),
        &z3_dense_leg(&right_rule, 1),
        3.0,
    );
    let wrong_right = Arc::new(ExternalZ3::tagged(0).product(ExternalZ3::tagged(3)));
    assert!(matches!(
        lhs.deligne_product(&rhs, wrong_right).unwrap_err(),
        symtn::typed::Error::RuleMismatch
    ));
}

#[test]
fn typed_deligne_product_preserves_duals_multiblocks_and_complex_values() {
    let _guard = cache_lock();
    let runtime = runtime();
    let u1_rule = Arc::new(symtn::sector::U1FusionRule);
    let fz2_rule = Arc::new(symtn::sector::FermionParityFusionRule);
    let charge_cod = GradedSpace::try_new(
        Arc::clone(&u1_rule),
        [
            (symtn::sector::U1Irrep::new(-1), 1),
            (symtn::sector::U1Irrep::new(1), 1),
        ],
    )
    .unwrap();
    let charge_dom = GradedSpace::try_new(
        Arc::clone(&u1_rule),
        [
            (symtn::sector::U1Irrep::new(-1), 1),
            (symtn::sector::U1Irrep::new(1), 1),
        ],
    )
    .and_then(|space| space.try_dual())
    .unwrap();
    let parity_cod = GradedSpace::try_new(
        Arc::clone(&fz2_rule),
        [
            (symtn::sector::Z2Irrep::EVEN, 1),
            (symtn::sector::Z2Irrep::ODD, 1),
        ],
    )
    .and_then(|space| space.try_dual())
    .unwrap();
    let parity_dom = GradedSpace::try_new(
        Arc::clone(&fz2_rule),
        [
            (symtn::sector::Z2Irrep::EVEN, 1),
            (symtn::sector::Z2Irrep::ODD, 1),
        ],
    )
    .unwrap();
    let lhs: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&charge_cod], [&charge_dom], |sectors, _| {
            Complex64::new(sectors.coupled().charge() as f64, 2.0)
        })
        .unwrap();
    let rhs: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&parity_cod], [&parity_dom], |sectors, _| {
            if *sectors.coupled() == symtn::sector::Z2Irrep::EVEN {
                Complex64::new(3.0, -1.0)
            } else {
                Complex64::new(-2.0, 4.0)
            }
        })
        .unwrap();
    let product =
        Arc::new(symtn::sector::U1FusionRule.product(symtn::sector::FermionParityFusionRule));

    let actual = lhs.deligne_product(&rhs, product).unwrap();
    let codomain = actual.codomain();
    let domain = actual.domain();
    let expected =
        TensorMap::from_subblock_fn(&runtime, codomain.iter(), domain.iter(), |sectors, _| {
            let codomain = sectors.codomain_uncoupled();
            let domain = sectors.domain_uncoupled();
            if codomain[0].left() == domain[0].left() && codomain[1].right() == domain[1].right() {
                let lhs = Complex64::new(codomain[0].left().charge() as f64, 2.0);
                let rhs = if *codomain[1].right() == symtn::sector::Z2Irrep::EVEN {
                    Complex64::new(3.0, -1.0)
                } else {
                    Complex64::new(-2.0, 4.0)
                };
                lhs * rhs
            } else {
                Complex64::new(0.0, 0.0)
            }
        })
        .unwrap();

    assert_eq!(actual.dense_data().unwrap(), expected.dense_data().unwrap());
    assert!(actual.subblock_count() > 1);
    assert_eq!(
        codomain
            .into_iter()
            .chain(domain)
            .map(|space| space.is_dual())
            .collect::<Vec<_>>(),
        [false, true, true, false]
    );
}

#[test]
fn typed_deligne_product_accepts_a_nondefault_product_codec() {
    type Codec = symtn::sector::PackedProductCodec<
        symtn::sector::U1SectorLayout,
        symtn::sector::Fz2SectorLayout,
    >;
    let _guard = cache_lock();
    let runtime = runtime();
    let u1_rule = Arc::new(symtn::sector::U1FusionRule);
    let fz2_rule = Arc::new(symtn::sector::FermionParityFusionRule);
    let charge =
        GradedSpace::try_new(Arc::clone(&u1_rule), [(symtn::sector::U1Irrep::new(2), 1)]).unwrap();
    let parity =
        GradedSpace::try_new(Arc::clone(&fz2_rule), [(symtn::sector::Z2Irrep::ODD, 1)]).unwrap();
    let lhs = TensorMap::from_subblock_fn(&runtime, [&charge], [&charge], |_, _| 2.0).unwrap();
    let rhs = TensorMap::from_subblock_fn(&runtime, [&parity], [&parity], |_, _| 5.0).unwrap();
    let product = Arc::new(symtn::sector::ProductFusionRule::<_, _, Codec>::new(
        symtn::sector::U1FusionRule,
        symtn::sector::FermionParityFusionRule,
    ));

    let result = lhs.deligne_product(&rhs, product).unwrap();

    assert_eq!(result.dense_data().unwrap(), [10.0]);
    assert_eq!(
        result.codomain()[0].sectors().unwrap(),
        [symtn::sector::product_sector(
            symtn::sector::U1Irrep::new(2),
            symtn::sector::Z2Irrep::EVEN
        )]
    );
}

#[test]
fn typed_deligne_product_maps_component_innerlines_into_the_product_tree() {
    let _guard = cache_lock();
    let runtime = runtime();
    let u1_rule = Arc::new(symtn::sector::U1FusionRule);
    let fz2_rule = Arc::new(symtn::sector::FermionParityFusionRule);
    let charges = [1, 2, 3].map(|charge| {
        GradedSpace::try_new(
            Arc::clone(&u1_rule),
            [(symtn::sector::U1Irrep::new(charge), 1)],
        )
        .unwrap()
    });
    let charge_total =
        GradedSpace::try_new(Arc::clone(&u1_rule), [(symtn::sector::U1Irrep::new(6), 1)]).unwrap();
    let odd =
        GradedSpace::try_new(Arc::clone(&fz2_rule), [(symtn::sector::Z2Irrep::ODD, 1)]).unwrap();
    let lhs =
        TensorMap::from_subblock_fn(&runtime, charges.iter(), [&charge_total], |_, _| 2.0).unwrap();
    let rhs =
        TensorMap::from_subblock_fn(&runtime, [&odd, &odd, &odd], [&odd], |_, _| 3.0).unwrap();
    let product =
        Arc::new(symtn::sector::U1FusionRule.product(symtn::sector::FermionParityFusionRule));

    let result = lhs.deligne_product(&rhs, product).unwrap();
    let block = result.subblock_fusion_trees(0).unwrap();

    assert_eq!(result.dense_data().unwrap(), [6.0]);
    assert_eq!(
        block.codomain_innerlines(),
        [
            symtn::sector::product_sector(
                symtn::sector::U1Irrep::new(3),
                symtn::sector::Z2Irrep::EVEN
            ),
            symtn::sector::product_sector(
                symtn::sector::U1Irrep::new(6),
                symtn::sector::Z2Irrep::EVEN
            ),
            symtn::sector::product_sector(
                symtn::sector::U1Irrep::new(6),
                symtn::sector::Z2Irrep::ODD
            ),
            symtn::sector::product_sector(
                symtn::sector::U1Irrep::new(6),
                symtn::sector::Z2Irrep::EVEN
            ),
        ]
    );
}

#[test]
fn typed_deligne_product_prepares_both_embeddings_before_publishing_either() {
    type WrongCodec = symtn::sector::PackedProductCodec<
        symtn::sector::U1SectorLayout,
        symtn::sector::Fz2SectorLayout,
    >;
    let _guard = cache_lock();
    let runtime = runtime();
    let rule = Arc::new(symtn::sector::U1FusionRule);
    let charge_one =
        GradedSpace::try_new(Arc::clone(&rule), [(symtn::sector::U1Irrep::new(1), 1)]).unwrap();
    let lhs =
        TensorMap::from_subblock_fn(&runtime, [&charge_one], [&charge_one], |_, _| 2.0).unwrap();
    let rhs =
        TensorMap::from_subblock_fn(&runtime, [&charge_one], [&charge_one], |_, _| 3.0).unwrap();
    let product = Arc::new(symtn::sector::ProductFusionRule::<
        symtn::sector::U1FusionRule,
        symtn::sector::U1FusionRule,
        WrongCodec,
    >::new(
        symtn::sector::U1FusionRule, symtn::sector::U1FusionRule
    ));
    let before = (
        structure_cache_info(StructureCacheKind::SectorStructure),
        structure_cache_info(StructureCacheKind::DegeneracyStructure),
    );

    assert!(lhs.deligne_product(&rhs, product).is_err());

    assert_eq!(
        (
            structure_cache_info(StructureCacheKind::SectorStructure),
            structure_cache_info(StructureCacheKind::DegeneracyStructure),
        ),
        before
    );
}

// ---------------------------------------------------------------------------
// `TensorMap::braid`.
// ---------------------------------------------------------------------------

#[test]
fn braid_moves_legs_of_a_multi_block_external_provider_tensor() {
    // What: an explicit braid with a full level assignment produces the same
    // reordered spaces a permute of the same axes does, and moves the payload.
    //
    // This symmetric fixture pins how levels are split and validated; the
    // Fibonacci tests above pin genuinely anyonic level values.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalZ3::new());
    let tensor = z3_rank_four(&runtime, &provider);

    let braided = tensor.braid(&[1, 2], &[3, 0], &[0, 1, 2, 3]).unwrap();
    let permuted = tensor.permute(&[1, 2], &[3, 0]).unwrap();

    assert_eq!(
        braided.dense_data().unwrap(),
        permuted.dense_data().unwrap()
    );
    assert_ne!(braided.dense_data().unwrap(), tensor.dense_data().unwrap());
    assert_eq!(braided.codomain()[0].degeneracies(), &[1, 2, 4]);
    assert_eq!(braided.domain()[1].degeneracies(), &[2, 1, 3]);
    // A different level assignment is the same morphism for a bosonic rule.
    let reversed = tensor.braid(&[1, 2], &[3, 0], &[3, 2, 1, 0]).unwrap();
    assert_eq!(
        reversed.dense_data().unwrap(),
        braided.dense_data().unwrap()
    );
}

#[test]
fn contract_ordered_delegates_with_a_nonidentity_output_order() {
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(symtn::sector::U1FusionRule);
    let leg = u1_leg(&provider, &[(-1, 1), (0, 2), (1, 1)]);
    let lhs: TensorMap<symtn::sector::U1FusionRule, f64> =
        TensorMap::rand_with_seed(&runtime, [], [&leg], 224_303).unwrap();
    let rhs: TensorMap<symtn::sector::U1FusionRule, f64> =
        TensorMap::rand_with_seed(&runtime, [&leg], [&leg, &leg], 224_304).unwrap();

    let actual = lhs
        .contract(
            &rhs,
            &ContractSpec {
                lhs: &[0],
                rhs: &[0],
                codomain: &[],
                domain: &[1, 0],
            },
        )
        .unwrap();
    let expected = lhs
        .contract(
            &rhs,
            &ContractSpec {
                lhs: &[0],
                rhs: &[0],
                codomain: &[],
                domain: &[0, 1],
            },
        )
        .unwrap()
        .permute(&[], &[1, 0])
        .unwrap();
    // The ordered route may run its GEMM on another layout; the contracted
    // leg has dimension 4.
    numerics::assert_slices_close(
        "contract",
        actual.dense_data().unwrap(),
        expected.dense_data().unwrap(),
        4,
    );
}

#[test]
fn typed_contract_parallel_su2_replay_matches_serial() {
    let _guard = cache_lock();

    fn run(runtime: &Runtime) -> Vec<f64> {
        let provider = Arc::new(SU2FusionRule);
        let leg = GradedSpace::try_new(
            provider,
            [
                (SU2Irrep::from_twice_spin(0), 2),
                (SU2Irrep::from_twice_spin(1), 3),
                (SU2Irrep::from_twice_spin(2), 2),
            ],
        )
        .unwrap();
        let lhs: TensorMap<SU2FusionRule, f64> =
            TensorMap::rand_with_seed(runtime, [&leg, &leg], [&leg, &leg], 224_401).unwrap();
        let rhs: TensorMap<SU2FusionRule, f64> =
            TensorMap::rand_with_seed(runtime, [&leg, &leg], [&leg, &leg], 224_402).unwrap();
        lhs.contract(
            &rhs,
            &ContractSpec {
                lhs: &[3, 2],
                rhs: &[0, 1],
                codomain: &[2, 0],
                domain: &[3, 1],
            },
        )
        .unwrap()
        .dense_data()
        .unwrap()
        .to_vec()
    }

    let serial = Runtime::builder().recoupling_threads(1).build().unwrap();
    let parallel = Runtime::builder().recoupling_threads(2).build().unwrap();
    let serial = run(&serial);
    let parallel = run(&parallel);
    assert_eq!(serial.len(), parallel.len());
    assert!(serial
        .iter()
        .zip(parallel)
        .all(|(&serial, parallel)| (serial - parallel).abs() < 1e-12));
}

#[test]
fn contract_on_the_external_z3_provider_matches_the_hand_product() {
    // What (gate 5): a typed-only ordered-contraction value check on the
    // external provider. Same fixture as the
    // `contract` hand-product gate: `output_axes = [1, 0]` is the transpose of
    // the 2x3 · 3x4 counting product, `[0, 1]` the product itself.
    let _guard = cache_lock();
    let runtime = runtime();
    let provider = Arc::new(ExternalZ3::new());
    let (rows, shared, columns) = (
        z3_dense_leg(&provider, 2),
        z3_dense_leg(&provider, 3),
        z3_dense_leg(&provider, 4),
    );
    let lhs = counting_z3(&runtime, &rows, &shared, 1.0);
    let rhs = counting_z3(&runtime, &shared, &columns, 7.0);

    let transposed: TensorMap<ExternalZ3, f64> = lhs
        .contract(
            &rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[1],
                domain: &[0],
            },
        )
        .unwrap();
    assert_eq!(
        transposed.dense_data().unwrap(),
        [76.0, 103.0, 130.0, 157.0, 100.0, 136.0, 172.0, 208.0]
    );
    let identity: TensorMap<ExternalZ3, f64> = lhs
        .contract(
            &rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    assert_eq!(
        identity.dense_data().unwrap(),
        [76.0, 100.0, 103.0, 136.0, 130.0, 172.0, 157.0, 208.0]
    );
}
