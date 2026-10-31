use super::*;

#[cfg(feature = "cgc-gen")]
#[test]
fn su3_octet_vertex_moves_preserve_unequal_degeneracy_blocks() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    use symtn::sector::LieGroupFusionRule;

    let runtime = Runtime::builder().build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let legs =
        [2, 3, 5].map(|n| GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], n)]).unwrap());
    let source = TensorMap::<_, f64>::from_subblock_fn(
        engine,
        &runtime,
        [&legs[0], &legs[1]],
        [&legs[2]],
        |trees, i| {
            100.0 * trees.codomain_vertices()[0].get() as f64
                + (15 * i[0] + 5 * i[1] + i[2] + 1) as f64
        },
    )
    .unwrap();
    // TensorKit 0.17.1 bendright/foldright(FusionTreeBlock) exported I₂.
    // Its repartition wrapper failed on Julia 1.12; these dense literals compose
    // those exported maps with indexmanipulations.jl's explicit axis permutation.
    let bend = [
        101., 106., 111., 102., 107., 112., 103., 108., 113., 104., 109., 114., 105., 110., 115.,
        116., 121., 126., 117., 122., 127., 118., 123., 128., 119., 124., 129., 120., 125., 130.,
    ];
    let fold = [
        101., 102., 103., 104., 105., 116., 117., 118., 119., 120., 106., 107., 108., 109., 110.,
        121., 122., 123., 124., 125., 111., 112., 113., 114., 115., 126., 127., 128., 129., 130.,
    ];
    for (output, shape, dual, literal) in [
        (
            source.transpose(engine, &[0], &[2, 1]).unwrap(),
            [2, 5, 3],
            [false, true],
            bend,
        ),
        (
            source.repartition(engine, 1).unwrap(),
            [2, 5, 3],
            [false, true],
            bend,
        ),
        (
            source.transpose(engine, &[1], &[0, 2]).unwrap(),
            [3, 2, 5],
            [true, false],
            fold,
        ),
    ] {
        assert_eq!(output.subblock_count(), 2);
        assert_eq!(
            output
                .domain()
                .iter()
                .map(|leg| leg.is_dual())
                .collect::<Vec<_>>(),
            dual
        );
        for b in 0..output.subblock_count() {
            let mu = output.subblock_fusion_trees(b).unwrap().domain_vertices()[0].get();
            let block = output.subblock(b).unwrap();
            assert_eq!(block.shape(), shape);
            let actual = (0..30)
                .map(|linear| {
                    let i = [
                        linear / (shape[1] * shape[2]),
                        linear / shape[2] % shape[1],
                        linear % shape[2],
                    ];
                    let offset = block.offset()
                        + i.iter()
                            .zip(block.strides())
                            .map(|(i, s)| i * s)
                            .sum::<usize>();
                    output.dense_data().unwrap()[offset]
                })
                .collect::<Vec<_>>();
            let expected = literal.map(|x| x + 100.0 * (mu - 1) as f64);
            // A linear transform has at most source.len() terms per entry.
            numerics::assert_slices_close("SU3 vertex move", &actual, &expected, 60);
            assert!(expected.windows(2).any(|pair| pair[1] != pair[0] + 1.0));
            assert!((actual[0] - (literal[0] + 100.0 * (2 - mu) as f64)).abs() > 50.0);
        }
    }
}

/// #1449: like TensorKit `adjoint(::DiagonalTensorMap)` and the
/// multiplicity-free path, the checked-Generic adjoint of a compact diagonal is
/// the owned conjugated diagonal, and its unstored zeros stay `+0 + 0i`.
#[cfg(feature = "cgc-gen")]
#[test]
fn checked_generic_complex_diagonal_adjoint_is_the_owned_conjugated_diagonal() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    use symtn::sector::LieGroupFusionRule;
    use symtn::sector::{U1FusionRule, U1Irrep};

    fn bits<R: TypedSectorAdmission>(
        engine: &mut dyn Engine,
        tensor: &TensorMap<R, Complex64>,
    ) -> Vec<(u64, u64)> {
        tensor
            .materialize(engine)
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .map(|z| (z.re.to_bits(), z.im.to_bits()))
            .collect()
    }
    // Each structural zero must be `+0 + 0i`; an `im` of `-0` fails.
    fn assert_positive_zeros<R: TypedSectorAdmission>(
        engine: &mut dyn Engine,
        tensor: &TensorMap<R, Complex64>,
        count: usize,
    ) {
        let zeros: Vec<_> = bits(engine, tensor)
            .into_iter()
            .filter(|&(re, im)| f64::from_bits(re) == 0.0 && f64::from_bits(im) == 0.0)
            .collect();
        assert_eq!(zeros, vec![(0, 0); count]);
    }
    let conj_if = |conj: bool, values: &[(f64, f64)]| -> Vec<Complex64> {
        values
            .iter()
            .map(|&(re, im)| Complex64::new(re, if conj { -im } else { im }))
            .collect()
    };

    let runtime = Runtime::builder().build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg = GradedSpace::try_new(provider, [(vec![2i64, 1], 3), (vec![0, 0], 2)]).unwrap();
    let big = [(1.5, 0.5), (-2.0, -1.0), (0.25, 3.0)];
    let small = [(3.0, -0.75), (-0.5, 2.0)];
    let su3 = |engine: &mut dyn Engine, conj: bool| {
        TensorMap::<_, Complex64>::diagonal(
            engine,
            &runtime,
            &leg,
            [
                SectorSpectrum {
                    sector: vec![2i64, 1],
                    values: conj_if(conj, &big),
                },
                SectorSpectrum {
                    sector: vec![0, 0],
                    values: conj_if(conj, &small),
                },
            ],
        )
        .unwrap()
    };
    let s = su3(engine, false);
    let adjoint = s.adjoint(engine).unwrap();
    // Hand-conjugated values are the oracle.
    let expected = su3(engine, true);
    // 3x3 [2,1] and 2x2 [0,0] blocks: 6 + 2 off-diagonal entries.
    assert_positive_zeros(engine, &adjoint, 8);
    assert_eq!(bits(engine, &adjoint), bits(engine, &expected));
    assert!(
        symtn::typed::__network::network_reuse_class(&adjoint, false) == NetworkReuseClass::Compact
    );
    assert_eq!(
        symtn::expert::diagonal_spectrum(engine, &adjoint).unwrap(),
        symtn::expert::diagonal_spectrum(engine, &expected).unwrap()
    );
    let adjoint_adjoint = &adjoint.adjoint(engine).unwrap();
    assert_eq!(
        symtn::expert::diagonal_spectrum(engine, adjoint_adjoint).unwrap(),
        symtn::expert::diagonal_spectrum(engine, &s).unwrap()
    );

    let assert_close = |got: &[Complex64], want: &[Complex64], what: &str| {
        assert_eq!(got.len(), want.len(), "{what}");
        for (got, want) in got.iter().zip(want) {
            assert!((got - want).norm() <= 1e-12, "{what}");
        }
    };
    // Consumers take the owned compact form as they take `s` itself.
    let Qr { q, r } = adjoint.qr_compact(engine, &[0], &[1]).unwrap();
    let qr = q.compose(engine, &r).unwrap();
    assert!(qr.dense_data().is_err(), "D * D stays compact");
    assert_close(
        qr.materialize(engine).unwrap().dense_data().unwrap(),
        adjoint.materialize(engine).unwrap().dense_data().unwrap(),
        "qr",
    );
    let Svd { u, s: sigma, vh } = adjoint.svd_compact(engine, &[0], &[1]).unwrap();
    let reconstructed = u
        .compose(engine, &sigma)
        .unwrap()
        .compose(engine, &vh)
        .unwrap();
    assert_close(
        reconstructed.dense_data().unwrap(),
        adjoint.materialize(engine).unwrap().dense_data().unwrap(),
        "svd",
    );
    // s^† s is diagonal with |s|^2 entries.
    let gram = adjoint.compose(engine, &s).unwrap();
    let norms = |values: &[(f64, f64)]| -> Vec<Complex64> {
        values
            .iter()
            .map(|&(re, im)| Complex64::new(re * re + im * im, 0.0))
            .collect()
    };
    let expected_gram = TensorMap::<_, Complex64>::diagonal(
        engine,
        &runtime,
        &leg,
        [
            SectorSpectrum {
                sector: vec![2i64, 1],
                values: norms(&big),
            },
            SectorSpectrum {
                sector: vec![0, 0],
                values: norms(&small),
            },
        ],
    )
    .unwrap();
    // D * D stays compact (TensorKit `compose_dest(::Diagonal, ::Diagonal)`).
    assert_close(
        &symtn::expert::diagonal_spectrum(engine, &gram)
            .unwrap()
            .expect("compact")
            .into_iter()
            .flat_map(|entry| entry.values)
            .collect::<Vec<_>>(),
        &symtn::expert::diagonal_spectrum(engine, &expected_gram)
            .unwrap()
            .expect("compact")
            .into_iter()
            .flat_map(|entry| entry.values)
            .collect::<Vec<_>>(),
        "s^† s",
    );

    // Non-self-dual SU(3) irreps on a dual leg (`[1,0]^2 + [0,1]` dualized to
    // `[0,1]^2 + [1,0]`): the adjoint keeps the space, not its dual, and
    // conjugates the values.
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let chiral = GradedSpace::try_new(provider, [(vec![1i64, 0], 2), (vec![0, 1], 1)])
        .unwrap()
        .try_dual()
        .unwrap();
    let triplets = [(0.5, -1.25), (2.0, 0.75)];
    let antitriplet = [(-1.0, 4.0)];
    let chiral_diagonal = |engine: &mut dyn Engine, conj: bool| {
        TensorMap::<_, Complex64>::diagonal(
            engine,
            &runtime,
            &chiral,
            [
                SectorSpectrum {
                    sector: vec![0i64, 1],
                    values: conj_if(conj, &triplets),
                },
                SectorSpectrum {
                    sector: vec![1, 0],
                    values: conj_if(conj, &antitriplet),
                },
            ],
        )
        .unwrap()
    };
    let source = chiral_diagonal(engine, false);
    let chiral_adjoint = source.adjoint(engine).unwrap();
    assert_eq!(chiral_adjoint.codomain(), source.codomain());
    assert_eq!(chiral_adjoint.domain(), source.domain());
    assert_eq!(chiral_adjoint.codomain()[0], chiral);
    let operand = &chiral_diagonal(engine, true);
    assert_eq!(
        symtn::expert::diagonal_spectrum(engine, &chiral_adjoint).unwrap(),
        symtn::expert::diagonal_spectrum(engine, operand).unwrap()
    );
    let operand = &chiral_diagonal(engine, true);
    assert_eq!(bits(engine, &chiral_adjoint), bits(engine, operand));
    // 2x2 [0,1] block and 1x1 [1,0] block: 2 off-diagonal entries.
    assert_positive_zeros(engine, &chiral_adjoint, 2);

    // A real diagonal is its own adjoint (TensorKit returns `d`).
    let real = TensorMap::<_, f64>::diagonal(
        engine,
        &runtime,
        &chiral,
        [
            SectorSpectrum {
                sector: vec![0i64, 1],
                values: vec![0.5, 2.0],
            },
            SectorSpectrum {
                sector: vec![1, 0],
                values: vec![-1.0],
            },
        ],
    )
    .unwrap();
    let real_adjoint = real.adjoint(engine).unwrap();
    assert!(
        symtn::typed::__network::network_reuse_class(&real_adjoint, false)
            == NetworkReuseClass::Compact
    );
    assert_eq!(real_adjoint.codomain(), real.codomain());
    assert_eq!(
        symtn::expert::diagonal_spectrum(engine, &real_adjoint).unwrap(),
        symtn::expert::diagonal_spectrum(engine, &real).unwrap()
    );

    // The multiplicity-free path gives the same representation and zeros.
    let u1 = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(1), 3), (U1Irrep::new(0), 2)],
    )
    .unwrap();
    let u1_diagonal = |engine: &mut dyn Engine, conj: bool| {
        TensorMap::<_, Complex64>::diagonal(
            engine,
            &runtime,
            &u1,
            [
                SectorSpectrum {
                    sector: U1Irrep::new(1),
                    values: conj_if(conj, &big),
                },
                SectorSpectrum {
                    sector: U1Irrep::new(0),
                    values: conj_if(conj, &small),
                },
            ],
        )
        .unwrap()
    };
    let mf_adjoint = u1_diagonal(engine, false).adjoint(engine).unwrap();
    assert!(
        symtn::typed::__network::network_reuse_class(&mf_adjoint, false)
            == NetworkReuseClass::Compact
    );
    let operand = &u1_diagonal(engine, true);
    assert_eq!(bits(engine, &mf_adjoint), bits(engine, operand));
    assert_positive_zeros(engine, &mf_adjoint, 8);
}

#[test]
fn checked_only_multiplicity_two_transforms_keep_the_source_authority() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg, &leg], [], |trees, _| {
            trees.codomain_vertices()[0].get() as f64
        })
        .unwrap();
    let snapshot = |tensor: &TensorMap<CheckedOnlyToy, f64>| {
        (0..tensor.subblock_count())
            .map(|index| tensor.subblock_fusion_trees(index).unwrap())
            .collect::<Vec<_>>()
    };
    let source_snapshot = snapshot(&source);
    provider.coefficient_queries.store(0, Ordering::Relaxed);
    let error = source.braid(engine, &[1, 0, 2], &[], &[0, 1]).unwrap_err();
    assert!(matches!(error, GenericTensorError::Facade(_)));
    assert_eq!(provider.coefficient_queries.load(Ordering::Relaxed), 0);

    let permuted = source.permute(engine, &[1, 0, 2], &[]).unwrap();
    assert!(std::ptr::eq(permuted.provider(), provider.as_ref()));
    let restored = permuted.permute(engine, &[1, 0, 2], &[]).unwrap();
    assert_eq!(snapshot(&restored), source_snapshot);
    for (actual, expected) in restored
        .dense_data()
        .unwrap()
        .iter()
        .zip(source.dense_data().unwrap())
    {
        assert!((actual - expected).abs() <= 1e-12);
    }

    let braided = source.braid(engine, &[1, 0, 2], &[], &[0, 1, 2]).unwrap();
    assert!(std::ptr::eq(braided.provider(), provider.as_ref()));

    // A permutation this source has not taken: a warm one answers from its
    // destination memo and completed transformer without the provider
    // (#2154), so only a cold call reaches the failing algebra.
    provider.fail_algebra.store(true, Ordering::Relaxed);
    let error = source.permute(engine, &[2, 1, 0], &[]).unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Plan(symtn::typed::CheckedGenericPlanError::Provider(
            ToyError::Algebra
        ))
    ));
}

#[cfg(feature = "cgc-gen")]
#[test]
fn sun_checked_generic_unit_insert_remove_preserves_authority_and_payload() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    use symtn::sector::LieGroupFusionRule;

    let runtime = Runtime::builder().build().unwrap();
    for n in [3, 4] {
        let provider = Arc::new(
            LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(n).unwrap()).unwrap(),
        );
        let label = if n == 3 { vec![1, 1] } else { vec![1, 0, 1] };
        let leg = GradedSpace::try_new(Arc::clone(&provider), [(label, 1)]).unwrap();
        let source: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, _| 2.0).unwrap();

        assert!(source.remove_unit(engine, 0).is_err());
        let inserted = source
            .insert_unit(engine, 0, Side::Domain, symtn::typed::Duality::Plain)
            .unwrap();
        assert!(std::ptr::eq(inserted.provider(), provider.as_ref()));
        assert_eq!(
            inserted.dense_data().unwrap().as_ptr(),
            source.dense_data().unwrap().as_ptr()
        );
        let removed = inserted.remove_unit(engine, 0).unwrap();
        assert!(std::ptr::eq(removed.provider(), provider.as_ref()));
        assert_eq!(
            removed.dense_data().unwrap().as_ptr(),
            source.dense_data().unwrap().as_ptr()
        );
        assert_eq!(removed.dense_data().unwrap(), source.dense_data().unwrap());

        // The seam decides only the boundary slot `position == codomain_rank`.
        for (seam, codomain_rank) in [(Side::Codomain, 2), (Side::Domain, 1)] {
            let inserted = source
                .insert_unit(engine, 1, seam, symtn::typed::Duality::Dual)
                .unwrap();
            assert_eq!(inserted.codomain_rank(), codomain_rank, "{seam:?}");
            let unit = match seam {
                Side::Codomain => inserted.codomain()[1].clone(),
                Side::Domain => inserted.domain()[0].clone(),
            };
            assert!(unit.is_dual(), "{seam:?}");
            assert!(std::ptr::eq(inserted.provider(), provider.as_ref()));
            assert_eq!(
                inserted.dense_data().unwrap().as_ptr(),
                source.dense_data().unwrap().as_ptr()
            );
        }
    }
}

#[test]
fn checked_unit_insert_correspondence_failure_does_not_publish_destination() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    if run_isolated_or_return(
        "SYMTN_CHECKED_UNIT_INSERT_CORRESPONDENCE_FAILURE_ISOLATED",
        "transforms::checked_unit_insert_correspondence_failure_does_not_publish_destination",
    ) {
        return;
    }
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(241));
    let x = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&x], [&x, &x], |_, _| 3.0).unwrap();

    // Negative fixture only: after the source exists, claim a second channel
    // for 1 x X. Destination preparation succeeds, but the larger layout no
    // longer corresponds to inserting an identity leg into `source`.
    provider.unit_layout_fault.store(1, Ordering::Relaxed);
    forget_cached_structures();
    let before = crate::checked_generic_facade::structure_cache_info(
        symtn::cache::StructureCacheKind::DegeneracyStructure,
    );
    assert_eq!(before.entries(), 0);

    let error = source
        .insert_unit(engine, 0, Side::Codomain, symtn::typed::Duality::Plain)
        .unwrap_err();

    assert!(matches!(
        error,
        GenericTensorError::Structure(CheckedGenericStructureError::Core(
            symtn::typed::CoreError::UnitLayoutCorrespondence
        ))
    ));
    let after = crate::checked_generic_facade::structure_cache_info(
        symtn::cache::StructureCacheKind::DegeneracyStructure,
    );
    assert_eq!(after.admissions(), before.admissions());
    assert_eq!(after.entries(), before.entries());
    assert_eq!(after.charged_bytes(), before.charged_bytes());
}

#[test]
fn checked_unit_remove_correspondence_failure_does_not_publish_destination() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    if run_isolated_or_return(
        "SYMTN_CHECKED_UNIT_REMOVE_CORRESPONDENCE_FAILURE_ISOLATED",
        "transforms::checked_unit_remove_correspondence_failure_does_not_publish_destination",
    ) {
        return;
    }
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(242));
    let x = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let vacuum = GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 1)]).unwrap();
    let base: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&x, &x], [&vacuum], |_, _| 5.0).unwrap();
    let source = base
        .insert_unit(engine, 0, Side::Codomain, symtn::typed::Duality::Plain)
        .unwrap();

    // Negative fixture only: rebuild the destination after suppressing the
    // X x X -> 1 channel that the already-built source still contains.
    provider.unit_layout_fault.store(2, Ordering::Relaxed);
    forget_cached_structures();
    let before = crate::checked_generic_facade::structure_cache_info(
        symtn::cache::StructureCacheKind::DegeneracyStructure,
    );
    assert_eq!(before.entries(), 0);

    let error = source.remove_unit(engine, 0).unwrap_err();

    assert!(matches!(
        error,
        GenericTensorError::Structure(CheckedGenericStructureError::Core(
            symtn::typed::CoreError::UnitLayoutCorrespondence
        ))
    ));
    let after = crate::checked_generic_facade::structure_cache_info(
        symtn::cache::StructureCacheKind::DegeneracyStructure,
    );
    assert_eq!(after.admissions(), before.admissions());
    assert_eq!(after.entries(), before.entries());
    assert_eq!(after.charged_bytes(), before.charged_bytes());
}

#[test]
fn checked_unit_final_provider_guard_remains_structure_typed() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(243));
    let x = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&x], [&x], |_, _| 7.0).unwrap();
    let payload = source.dense_data().unwrap().as_ptr();

    reset_provider_queries(&provider);
    provider
        .invalid_style_after_first_query
        .store(true, Ordering::Relaxed);
    let error = source
        .insert_unit(engine, 0, Side::Codomain, symtn::typed::Duality::Plain)
        .unwrap_err();

    assert!(matches!(
        error,
        GenericTensorError::Structure(CheckedGenericStructureError::Core(
            symtn::typed::CoreError::UnsupportedFusionStyle {
                expected: FusionStyleKind::Generic,
                actual: FusionStyleKind::Unique,
            }
        ))
    ));
    assert_eq!(provider.style_queries.load(Ordering::Relaxed), 2);
    assert_eq!(source.dense_data().unwrap().as_ptr(), payload);
    assert_eq!(source.dense_data().unwrap(), &[7.0]);
}

#[test]
fn checked_only_contract_and_compose_keep_left_authority() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let _cache = cache_exclusive();
    let runtime = Runtime::builder().build().unwrap();
    let left_provider = Arc::new(CheckedOnlyToy::new(124));
    let right_provider = Arc::new(CheckedOnlyToy::new(124));
    let left_leg = GradedSpace::try_new(Arc::clone(&left_provider), [(Label::X, 1)]).unwrap();
    let right_leg = GradedSpace::try_new(Arc::clone(&right_provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&left_leg], [&left_leg], |_, _| 1.0)
            .unwrap();
    let nontrivial: TensorMap<_, f64> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&left_leg, &left_leg],
        [&left_leg],
        |_, _| 1.0,
    )
    .unwrap();
    let identity: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&right_leg], [&right_leg], |_, _| 1.0)
            .unwrap();
    left_provider.r_queries.store(0, Ordering::Relaxed);

    for output in [
        source
            .contract(
                engine,
                &identity,
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &[0],
                    domain: &[1],
                },
            )
            .unwrap(),
        source.compose(engine, &identity).unwrap(),
    ] {
        assert!(std::ptr::eq(output.provider(), left_provider.as_ref()));
        assert_eq!(output.dense_data().unwrap(), source.dense_data().unwrap());
        for index in 0..source.subblock_count() {
            assert_eq!(
                output.subblock_fusion_trees(index).unwrap(),
                source.subblock_fusion_trees(index).unwrap()
            );
        }
    }
    assert_eq!(left_provider.r_queries.load(Ordering::Relaxed), 0);

    let other_runtime = Runtime::builder().build().unwrap();
    let foreign_runtime_identity: TensorMap<_, f64> = TensorMap::from_subblock_fn(
        engine,
        &other_runtime,
        [&right_leg],
        [&right_leg],
        |_, _| 1.0,
    )
    .unwrap();
    left_provider.algebra_queries.store(0, Ordering::Relaxed);
    right_provider.algebra_queries.store(0, Ordering::Relaxed);
    assert!(matches!(
        source.contract(
            engine,
            &foreign_runtime_identity,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1]
            }
        ),
        Err(GenericTensorError::Facade(
            symtn::typed::Error::RuntimeMismatch
        ))
    ));
    assert_eq!(left_provider.algebra_queries.load(Ordering::Relaxed), 0);
    assert_eq!(right_provider.algebra_queries.load(Ordering::Relaxed), 0);

    let wrong_provider = Arc::new(CheckedOnlyToy::new(1));
    let wrong_leg = GradedSpace::try_new(Arc::clone(&wrong_provider), [(Label::X, 1)]).unwrap();
    let wrong_identity: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&wrong_leg], [&wrong_leg], |_, _| 1.0)
            .unwrap();
    left_provider.algebra_queries.store(0, Ordering::Relaxed);
    wrong_provider.algebra_queries.store(0, Ordering::Relaxed);
    assert!(matches!(
        source.contract(
            engine,
            &wrong_identity,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1]
            }
        ),
        Err(GenericTensorError::Plan(
            symtn::typed::CheckedGenericPlanError::Core(
                symtn_core::CoreError::FusionRuleMismatch { .. }
            )
        ))
    ));
    assert_eq!(left_provider.algebra_queries.load(Ordering::Relaxed), 0);
    assert_eq!(wrong_provider.algebra_queries.load(Ordering::Relaxed), 0);

    // See `forget_cached_structures`: this tag is this test's alone.
    forget_cached_structures();
    left_provider.fail_algebra.store(true, Ordering::Relaxed);
    let error = nontrivial
        .contract(
            engine,
            &identity,
            &ContractSpec {
                lhs: &[2],
                rhs: &[0],
                codomain: &[0, 1],
                domain: &[2],
            },
        )
        .unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Plan(symtn::typed::CheckedGenericPlanError::Provider(
            ToyError::Algebra
        ))
    ));
}

#[test]
fn checked_only_identity_transforms_make_no_provider_queries() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg], [&leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();

    for counter in [
        &provider.identity_queries,
        &provider.style_queries,
        &provider.algebra_queries,
        &provider.coefficient_queries,
        &provider.f_queries,
        &provider.r_queries,
        &provider.postcommit_queries,
    ] {
        counter.store(0, Ordering::Relaxed);
    }

    for output in [
        source.permute(engine, &[0, 1], &[2]).unwrap(),
        source.braid(engine, &[0, 1], &[2], &[2, 1, 0]).unwrap(),
        source.transpose(engine, &[0, 1], &[2]).unwrap(),
        source.repartition(engine, 2).unwrap(),
    ] {
        assert!(std::ptr::eq(output.provider(), provider.as_ref()));
        assert_eq!(
            output.dense_data().unwrap().as_ptr(),
            source.dense_data().unwrap().as_ptr()
        );
    }
    for counter in [
        &provider.identity_queries,
        &provider.style_queries,
        &provider.algebra_queries,
        &provider.coefficient_queries,
        &provider.f_queries,
        &provider.r_queries,
        &provider.postcommit_queries,
    ] {
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn checked_only_otimes_preserves_typed_late_f_failures() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg], [&leg, &leg], |trees, _| {
            trees.codomain_vertices()[0].get() as f64
                - 2.0 * trees.domain_vertices()[0].get() as f64
        })
        .unwrap();

    provider.f_queries.store(0, Ordering::Relaxed);
    source.otimes(engine, &source).unwrap();
    let final_f_query = provider.f_queries.load(Ordering::Relaxed);
    assert!(final_f_query > 1);
    provider.f_queries.store(0, Ordering::Relaxed);
    provider.reset_commit_spy();
    provider
        .fail_f_on_query
        .store(final_f_query, Ordering::Relaxed);
    let error = source.otimes(engine, &source).unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::TensorProduct(CheckedGenericTensorProductError::Provider(
            ToyError::Algebra
        ))
    ));
    assert_eq!(provider.f_queries.load(Ordering::Relaxed), final_f_query);
    assert!(!provider.final_style_guard_ran_last());
    provider.fail_f_on_query.store(0, Ordering::Relaxed);

    provider.f_queries.store(0, Ordering::Relaxed);
    provider.reset_commit_spy();
    provider.malformed_f.store(true, Ordering::Relaxed);
    let error = source.otimes(engine, &source).unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::TensorProduct(CheckedGenericTensorProductError::SymbolShape {
            symbol: "F",
            ..
        })
    ));
    assert!(!provider.final_style_guard_ran_last());
    assert_eq!(provider.r_queries.load(Ordering::Relaxed), 0);
}

#[test]
fn checked_only_otimes_rejects_runtime_identity_and_style_before_algebra() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let other_runtime = Runtime::builder().build().unwrap();
    let first = Arc::new(CheckedOnlyToy::new(0));
    let mismatched = Arc::new(CheckedOnlyToy::new(1));
    let first_leg = GradedSpace::try_new(Arc::clone(&first), [(Label::X, 1)]).unwrap();
    let mismatched_leg = GradedSpace::try_new(Arc::clone(&mismatched), [(Label::X, 1)]).unwrap();
    let lhs: TensorMap<_, f64> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&first_leg, &first_leg],
        [&first_leg],
        |_, _| 1.0,
    )
    .unwrap();
    let wrong_runtime: TensorMap<_, f64> = TensorMap::from_subblock_fn(
        engine,
        &other_runtime,
        [&first_leg, &first_leg],
        [&first_leg],
        |_, _| 1.0,
    )
    .unwrap();
    let wrong_identity: TensorMap<_, f64> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&mismatched_leg, &mismatched_leg],
        [&mismatched_leg],
        |_, _| 1.0,
    )
    .unwrap();
    first.algebra_queries.store(0, Ordering::Relaxed);
    first.coefficient_queries.store(0, Ordering::Relaxed);
    mismatched.algebra_queries.store(0, Ordering::Relaxed);
    mismatched.coefficient_queries.store(0, Ordering::Relaxed);

    assert!(matches!(
        lhs.otimes(engine, &wrong_runtime),
        Err(GenericTensorError::Facade(
            symtn::typed::Error::RuntimeMismatch
        ))
    ));
    assert!(matches!(
        lhs.otimes(engine, &wrong_identity),
        Err(GenericTensorError::TensorProduct(
            CheckedGenericTensorProductError::Core(
                symtn_core::CoreError::FusionRuleMismatch { .. }
            )
        ))
    ));
    first.invalid_style.store(true, Ordering::Relaxed);
    assert!(matches!(
        lhs.otimes(engine, &lhs),
        Err(GenericTensorError::TensorProduct(
            CheckedGenericTensorProductError::Core(symtn_core::CoreError::UnsupportedFusionStyle {
                actual: symtn::sector::FusionStyleKind::Unique,
                ..
            })
        ))
    ));
    assert_eq!(first.algebra_queries.load(Ordering::Relaxed), 0);
    assert_eq!(first.coefficient_queries.load(Ordering::Relaxed), 0);
    assert_eq!(mismatched.algebra_queries.load(Ordering::Relaxed), 0);
    assert_eq!(mismatched.coefficient_queries.load(Ordering::Relaxed), 0);
}

#[test]
fn checked_only_otimes_matches_fixed_heterogeneous_nonunit_oracle() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let _cache = cache_shared();
    let runtime = Runtime::builder().build().unwrap();
    let first = Arc::new(CheckedOnlyToy::new_product_probe(9));
    let second = Arc::new(CheckedOnlyToy::new_product_probe(9));
    let x2 = GradedSpace::try_new(Arc::clone(&first), [(Label::X, 2)]).unwrap();
    let x1 = GradedSpace::try_new(Arc::clone(&first), [(Label::X, 1)]).unwrap();
    let y3 = GradedSpace::try_new(Arc::clone(&second), [(Label::One, 3)]).unwrap();
    let y1 = GradedSpace::try_new(Arc::clone(&second), [(Label::One, 1)]).unwrap();
    let rhs_x1 = GradedSpace::try_new(Arc::clone(&second), [(Label::X, 1)]).unwrap();
    let lhs: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&x2, &x1], [&x1, &x1], |trees, index| {
            100.0 * trees.codomain_vertices()[0].get() as f64
                + 10.0 * trees.domain_vertices()[0].get() as f64
                + index[0] as f64
                + 1.0
        })
        .unwrap();
    let rhs: TensorMap<_, f64> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&y3, &rhs_x1],
        [&y1, &rhs_x1],
        |_, index| index[0] as f64 + 1.0,
    )
    .unwrap();
    for provider in [&first, &second] {
        provider.identity_queries.store(0, Ordering::Relaxed);
        provider.style_queries.store(0, Ordering::Relaxed);
        provider.algebra_queries.store(0, Ordering::Relaxed);
        provider.coefficient_queries.store(0, Ordering::Relaxed);
        provider.r_queries.store(0, Ordering::Relaxed);
        provider.f_queries.store(0, Ordering::Relaxed);
        provider.reset_commit_spy();
    }
    let output = lhs.otimes(engine, &rhs).unwrap();
    assert!(std::ptr::eq(output.provider(), first.as_ref()));
    // The sector-structure cache key (#2030) and the final commit guard
    // reuse the identity the checked walk admitted (#2046).
    assert_eq!(first.identity_queries.load(Ordering::Relaxed), 2);
    assert_eq!(second.identity_queries.load(Ordering::Relaxed), 1);
    assert!(first.final_style_guard_ran_last());
    assert!(first.algebra_queries.load(Ordering::Relaxed) > 0);
    assert!(first.f_queries.load(Ordering::Relaxed) > 0);
    assert_eq!(first.r_queries.load(Ordering::Relaxed), 0);
    assert_eq!(second.algebra_queries.load(Ordering::Relaxed), 0);
    assert_eq!(second.coefficient_queries.load(Ordering::Relaxed), 0);
    assert_eq!(second.f_queries.load(Ordering::Relaxed), 0);
    assert_eq!(second.r_queries.load(Ordering::Relaxed), 0);

    const EXPECTED_KEYS: [([usize; 3], [usize; 3]); 16] = [
        ([1, 1, 1], [1, 1, 1]),
        ([2, 1, 1], [1, 1, 1]),
        ([1, 1, 2], [1, 1, 1]),
        ([2, 1, 2], [1, 1, 1]),
        ([1, 1, 1], [2, 1, 1]),
        ([2, 1, 1], [2, 1, 1]),
        ([1, 1, 2], [2, 1, 1]),
        ([2, 1, 2], [2, 1, 1]),
        ([1, 1, 1], [1, 1, 2]),
        ([2, 1, 1], [1, 1, 2]),
        ([1, 1, 2], [1, 1, 2]),
        ([2, 1, 2], [1, 1, 2]),
        ([1, 1, 1], [2, 1, 2]),
        ([2, 1, 1], [2, 1, 2]),
        ([1, 1, 2], [2, 1, 2]),
        ([2, 1, 2], [2, 1, 2]),
    ];
    let keys = (0..output.subblock_count())
        .map(|index| {
            let trees = output.subblock_fusion_trees(index).unwrap();
            assert_eq!(
                trees.codomain_uncoupled(),
                &[Label::X, Label::X, Label::One, Label::X]
            );
            assert_eq!(
                trees.domain_uncoupled(),
                &[Label::X, Label::X, Label::One, Label::X]
            );
            assert_eq!(
                output.subblock(index).unwrap().shape(),
                &[2, 1, 3, 1, 1, 1, 1, 1]
            );
            (
                trees
                    .codomain_vertices()
                    .iter()
                    .map(|vertex| vertex.get())
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),
                trees
                    .domain_vertices()
                    .iter()
                    .map(|vertex| vertex.get())
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(keys, EXPECTED_KEYS);

    const EXPECTED_DATA: [f64; 96] = [
        555.0, 560.0, 1110.0, 1120.0, 1665.0, 1680.0, 1055.0, 1060.0, 2110.0, 2120.0, 3165.0,
        3180.0, 1221.0, 1232.0, 2442.0, 2464.0, 3663.0, 3696.0, 2321.0, 2332.0, 4642.0, 4664.0,
        6963.0, 6996.0, 605.0, 610.0, 1210.0, 1220.0, 1815.0, 1830.0, 1105.0, 1110.0, 2210.0,
        2220.0, 3315.0, 3330.0, 1331.0, 1342.0, 2662.0, 2684.0, 3993.0, 4026.0, 2431.0, 2442.0,
        4862.0, 4884.0, 7293.0, 7326.0, 1221.0, 1232.0, 2442.0, 2464.0, 3663.0, 3696.0, 2321.0,
        2332.0, 4642.0, 4664.0, 6963.0, 6996.0, 2775.0, 2800.0, 5550.0, 5600.0, 8325.0, 8400.0,
        5275.0, 5300.0, 10550.0, 10600.0, 15825.0, 15900.0, 1331.0, 1342.0, 2662.0, 2684.0, 3993.0,
        4026.0, 2431.0, 2442.0, 4862.0, 4884.0, 7293.0, 7326.0, 3025.0, 3050.0, 6050.0, 6100.0,
        9075.0, 9150.0, 5525.0, 5550.0, 11050.0, 11100.0, 16575.0, 16650.0,
    ];
    assert_eq!(output.dense_data().unwrap(), EXPECTED_DATA);

    // The first stored value has two nonzero root-multiplicity paths:
    // μ=1 contributes 111*1 and μ=2 contributes 111*4. The fixed result
    // therefore kills overwrite-instead-of-accumulate mutations.
    let colliding_path_coefficients = [1.0, 4.0];
    assert_eq!(colliding_path_coefficients.len(), 2);
    assert_eq!(
        111.0 * colliding_path_coefficients.iter().sum::<f64>(),
        EXPECTED_DATA[0]
    );

    let complex_lhs: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&x2, &x1], [&x1, &x1], |trees, index| {
            Complex64::new(1.0, 1.0)
                * (100.0 * trees.codomain_vertices()[0].get() as f64
                    + 10.0 * trees.domain_vertices()[0].get() as f64
                    + index[0] as f64
                    + 1.0)
        })
        .unwrap();
    let complex_rhs: TensorMap<_, Complex64> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&y3, &rhs_x1],
        [&y1, &rhs_x1],
        |_, index| Complex64::new(2.0, -3.0) * (index[0] as f64 + 1.0),
    )
    .unwrap();
    first.f_queries.store(0, Ordering::Relaxed);
    first.reset_commit_spy();
    let complex = complex_lhs.otimes(engine, &complex_rhs).unwrap();
    assert!(std::ptr::eq(complex.provider(), first.as_ref()));
    for (actual, expected) in complex.dense_data().unwrap().iter().zip(EXPECTED_DATA) {
        assert!((*actual - Complex64::new(5.0, -1.0) * expected).norm() <= 1e-12);
    }
}

#[test]
fn checked_generic_cat_admits_once_and_queries_only_left_before_commit() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: successful catdomain admission uses the left provider Arc once;
    // admitted identity stamps keep the equal-identity right provider cold,
    // and copy planning performs no provider query after commit.
    //
    // Why a fresh process: the spy commits after as many queries as `zeros`
    // made on the same admission, which depends on the process-wide
    // structure caches. A sibling test's `forget_cached_structures` between
    // the two would make `cat` re-derive what `zeros` found cached, and those
    // extra admission queries would land after the spy's commit point.
    if run_isolated_or_return(
        "SYMTN_CHECKED_CAT_ADMITS_ONCE_ISOLATED",
        "transforms::checked_generic_cat_admits_once_and_queries_only_left_before_commit",
    ) {
        return;
    }
    let runtime = Runtime::builder().build().unwrap();
    let left_provider = Arc::new(CheckedOnlyToy::new(0));
    let right_provider = Arc::new(CheckedOnlyToy::new(0));
    let left_common = GradedSpace::try_new(Arc::clone(&left_provider), [(Label::X, 1)]).unwrap();
    let right_common = GradedSpace::try_new(Arc::clone(&right_provider), [(Label::X, 1)]).unwrap();
    let left_changed = GradedSpace::try_new(Arc::clone(&left_provider), [(Label::X, 1)]).unwrap();
    let right_changed = GradedSpace::try_new(Arc::clone(&right_provider), [(Label::X, 2)]).unwrap();
    let lhs: TensorMap<_, f64> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&left_common, &left_common],
        [&left_changed],
        |trees, indices| {
            10.0 + trees.codomain_vertices()[0].get() as f64 + indices.iter().sum::<usize>() as f64
        },
    )
    .unwrap();
    let rhs: TensorMap<_, f64> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&right_common, &right_common],
        [&right_changed],
        |trees, indices| {
            20.0 + trees.codomain_vertices()[0].get() as f64 + indices.iter().sum::<usize>() as f64
        },
    )
    .unwrap();
    let combined = left_changed.oplus(&right_changed).unwrap();
    for provider in [&left_provider, &right_provider] {
        for counter in [
            &provider.identity_queries,
            &provider.style_queries,
            &provider.algebra_queries,
            &provider.coefficient_queries,
            &provider.f_queries,
            &provider.r_queries,
        ] {
            counter.store(0, Ordering::Relaxed);
        }
        provider.reset_commit_spy();
    }
    let _: TensorMap<_, f64> =
        TensorMap::zeros(engine, &runtime, [&left_common, &left_common], [&combined]).unwrap();
    let mut admission_queries = [
        left_provider.identity_queries.load(Ordering::Relaxed),
        left_provider.style_queries.load(Ordering::Relaxed),
        left_provider.algebra_queries.load(Ordering::Relaxed),
        left_provider.coefficient_queries.load(Ordering::Relaxed),
        left_provider.f_queries.load(Ordering::Relaxed),
        left_provider.r_queries.load(Ordering::Relaxed),
    ];
    let admission_query_count = left_provider.queries_since_reset.load(Ordering::Relaxed) - 3;
    // `zeros` first checks the three supplied leg authorities; cat starts from
    // already-admitted stamps, so remove exactly those three identity reads.
    admission_queries[0] -= 3;
    for counter in [
        &left_provider.identity_queries,
        &left_provider.style_queries,
        &left_provider.algebra_queries,
        &left_provider.coefficient_queries,
        &left_provider.f_queries,
        &left_provider.r_queries,
    ] {
        counter.store(0, Ordering::Relaxed);
    }
    left_provider.arm_commit_spy_after_queries(admission_query_count);

    let output = lhs.cat(engine, &rhs, Side::Domain).unwrap();

    assert!(std::ptr::eq(output.provider(), left_provider.as_ref()));
    assert!(!std::ptr::eq(output.provider(), right_provider.as_ref()));
    assert_eq!(left_provider.commit_count.load(Ordering::Relaxed), 1);
    assert_eq!(left_provider.postcommit_queries.load(Ordering::Relaxed), 0);
    assert_eq!(
        [
            left_provider.identity_queries.load(Ordering::Relaxed),
            left_provider.style_queries.load(Ordering::Relaxed),
            left_provider.algebra_queries.load(Ordering::Relaxed),
            left_provider.coefficient_queries.load(Ordering::Relaxed),
            left_provider.f_queries.load(Ordering::Relaxed),
            left_provider.r_queries.load(Ordering::Relaxed),
        ],
        admission_queries
    );
    for counter in [
        &right_provider.identity_queries,
        &right_provider.style_queries,
        &right_provider.algebra_queries,
        &right_provider.coefficient_queries,
        &right_provider.f_queries,
        &right_provider.r_queries,
        &right_provider.postcommit_queries,
    ] {
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }
}

#[test]
fn checked_generic_cat_precedence_and_admission_failure_are_typed_nonpublishing() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let _cache = cache_exclusive();
    // What: admission stamps, runtime, cat arguments, then output admission
    // reject in order; every failure leaves both admitted input payloads alone.
    let runtime = Runtime::builder().build().unwrap();
    let other_runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(123));
    let equal = Arc::new(CheckedOnlyToy::new(123));
    let wrong = Arc::new(CheckedOnlyToy::new(1));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let equal_leg = GradedSpace::try_new(Arc::clone(&equal), [(Label::X, 1)]).unwrap();
    let wrong_leg = GradedSpace::try_new(Arc::clone(&wrong), [(Label::X, 1)]).unwrap();
    let lhs: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg], [&leg], |_, indices| {
            1.0 + indices.iter().sum::<usize>() as f64
        })
        .unwrap();
    let valid_rhs: TensorMap<_, f64> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&equal_leg, &equal_leg],
        [&equal_leg],
        |_, _| 2.0,
    )
    .unwrap();
    let wrong_identity: TensorMap<_, f64> = TensorMap::zeros(
        engine,
        &other_runtime,
        [&wrong_leg, &wrong_leg],
        [&wrong_leg, &wrong_leg],
    )
    .unwrap();
    let wrong_runtime: TensorMap<_, f64> = TensorMap::zeros(
        engine,
        &other_runtime,
        [&equal_leg, &equal_leg],
        [&equal_leg, &equal_leg],
    )
    .unwrap();
    let bad_arguments: TensorMap<_, f64> = TensorMap::zeros(
        engine,
        &runtime,
        [&equal_leg, &equal_leg],
        [&equal_leg, &equal_leg],
    )
    .unwrap();
    let lhs_before = lhs.dense_data().unwrap().to_vec();
    let rhs_before = valid_rhs.dense_data().unwrap().to_vec();
    // See `forget_cached_structures`: this tag is this test's alone.
    forget_cached_structures();
    provider.fail_algebra.store(true, Ordering::Relaxed);
    for counter in [
        &provider.identity_queries,
        &provider.style_queries,
        &provider.algebra_queries,
        &provider.coefficient_queries,
        &provider.f_queries,
        &provider.r_queries,
    ] {
        counter.store(0, Ordering::Relaxed);
    }
    provider.reset_commit_spy();

    assert!(matches!(
        lhs.cat(engine, &wrong_identity, Side::Domain),
        Err(GenericTensorError::Facade(
            symtn::typed::Error::RuleMismatch
        ))
    ));
    assert!(matches!(
        lhs.cat(engine, &wrong_runtime, Side::Domain),
        Err(GenericTensorError::Facade(
            symtn::typed::Error::RuntimeMismatch
        ))
    ));
    assert!(
        matches!(lhs.cat(engine, &bad_arguments, Side::Domain), Err(GenericTensorError::Facade(symtn::typed::Error::InvalidArgument(message))) if message.contains("exactly one domain leg"))
    );
    assert_eq!(provider.identity_queries.load(Ordering::Relaxed), 0);
    assert_eq!(provider.algebra_queries.load(Ordering::Relaxed), 0);

    assert!(matches!(
        lhs.cat(engine, &valid_rhs, Side::Domain),
        Err(GenericTensorError::Structure(
            CheckedGenericStructureError::Provider(ToyError::Algebra)
        ))
    ));
    assert!(!provider.final_style_guard_ran_last());
    assert_eq!(lhs.dense_data().unwrap(), lhs_before);
    assert_eq!(valid_rhs.dense_data().unwrap(), rhs_before);
}

#[cfg(feature = "cgc-gen")]
fn sun_cat_marker(trees: &symtn::typed::BlockFusionTrees<Vec<i64>>) -> usize {
    trees
        .codomain_vertices()
        .iter()
        .enumerate()
        .map(|(index, vertex)| (index + 1) * 100 * vertex.get())
        .chain(
            trees
                .domain_vertices()
                .iter()
                .enumerate()
                .map(|(index, vertex)| (index + 1) * 1_000 * vertex.get()),
        )
        .sum()
}

#[cfg(feature = "cgc-gen")]
fn assert_sun_cat_values<D>(
    output: &TensorMap<symtn::sector::LieGroupFusionRule, D>,
    lhs: &TensorMap<symtn::sector::LieGroupFusionRule, D>,
    rhs: &TensorMap<symtn::sector::LieGroupFusionRule, D>,
    changed_axis: usize,
    lhs_extent: usize,
    value: impl Fn(usize) -> D,
) where
    D: Copy + fmt::Debug + PartialEq + symtn::typed::TensorScalar,
{
    let mut saw_mu_two = false;
    for output_index in 0..output.subblock_count() {
        let trees = output.subblock_fusion_trees(output_index).unwrap();
        saw_mu_two |= trees
            .codomain_vertices()
            .iter()
            .chain(trees.domain_vertices())
            .any(|vertex| vertex.get() == 2);
        assert!((0..lhs.subblock_count())
            .any(|index| lhs.subblock_fusion_trees(index).unwrap() == trees));
        assert!((0..rhs.subblock_count())
            .any(|index| rhs.subblock_fusion_trees(index).unwrap() == trees));
        let block = output.subblock(output_index).unwrap();
        let elements = block.shape().iter().product::<usize>();
        for linear in 0..elements {
            let mut remainder = linear;
            let mut indices = Vec::with_capacity(block.shape().len());
            let mut position = block.offset();
            for (&extent, &stride) in block.shape().iter().zip(block.strides()) {
                let index = remainder % extent;
                remainder /= extent;
                indices.push(index);
                position += index * stride;
            }
            let (base, local_changed) = if indices[changed_axis] < lhs_extent {
                (10_000, indices[changed_axis])
            } else {
                (20_000, indices[changed_axis] - lhs_extent)
            };
            indices[changed_axis] = local_changed;
            let index_marker = indices
                .iter()
                .enumerate()
                .map(|(axis, index)| (axis + 1) * index)
                .sum::<usize>();
            assert_eq!(
                output.dense_data().unwrap()[position],
                value(base + sun_cat_marker(&trees) + index_marker)
            );
        }
    }
    assert!(saw_mu_two, "SU(N) cat fixture must carry a μ=2 full key");
}

#[cfg(feature = "cgc-gen")]
fn assert_sun_cat_case<D>(
    engine: &mut dyn Engine,
    n: usize,
    label: Vec<i64>,
    value: impl Fn(usize) -> D + Copy,
) where
    D: Copy + fmt::Debug + PartialEq + symtn::typed::TensorScalar,
{
    use symtn::sector::LieGroupFusionRule;

    let runtime = Runtime::builder().build().unwrap();
    let left_provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(n).unwrap()).unwrap());
    let right_provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(n).unwrap()).unwrap());
    let left_common =
        GradedSpace::try_new(Arc::clone(&left_provider), [(label.clone(), 1)]).unwrap();
    let right_common =
        GradedSpace::try_new(Arc::clone(&right_provider), [(label.clone(), 1)]).unwrap();
    let left_changed =
        GradedSpace::try_new(Arc::clone(&left_provider), [(label.clone(), 1)]).unwrap();
    let right_changed =
        GradedSpace::try_new(Arc::clone(&right_provider), [(label.clone(), 2)]).unwrap();
    let fill = |base, trees: &symtn::typed::BlockFusionTrees<Vec<i64>>, indices: &[usize]| {
        value(
            base + sun_cat_marker(trees)
                + indices
                    .iter()
                    .enumerate()
                    .map(|(axis, index)| (axis + 1) * index)
                    .sum::<usize>(),
        )
    };

    let domain_lhs: TensorMap<_, D> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&left_common, &left_common],
        [&left_changed],
        |trees, indices| fill(10_000, trees, indices),
    )
    .unwrap();
    let domain_rhs: TensorMap<_, D> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&right_common, &right_common],
        [&right_changed],
        |trees, indices| fill(20_000, trees, indices),
    )
    .unwrap();
    let domain = domain_lhs.cat(engine, &domain_rhs, Side::Domain).unwrap();
    assert!(std::ptr::eq(domain.provider(), left_provider.as_ref()));
    assert!(!std::ptr::eq(domain.provider(), right_provider.as_ref()));
    assert_eq!(domain.domain()[0].degeneracy(&label).unwrap(), 3);
    assert_sun_cat_values(&domain, &domain_lhs, &domain_rhs, 2, 1, value);
    let domain_rhs_adjoint = domain_rhs.adjoint(engine).unwrap();
    let lazy_domain = domain_lhs
        .adjoint(engine)
        .unwrap()
        .cat(engine, &domain_rhs_adjoint, Side::Codomain)
        .unwrap();
    assert_eq!(
        lazy_domain.dense_data().unwrap(),
        domain
            .adjoint(engine)
            .unwrap()
            .materialize(engine)
            .unwrap()
            .dense_data()
            .unwrap()
    );

    let codomain_lhs: TensorMap<_, D> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&left_changed],
        [&left_common, &left_common],
        |trees, indices| fill(10_000, trees, indices),
    )
    .unwrap();
    let codomain_rhs: TensorMap<_, D> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&right_changed],
        [&right_common, &right_common],
        |trees, indices| fill(20_000, trees, indices),
    )
    .unwrap();
    let codomain = codomain_lhs
        .cat(engine, &codomain_rhs, Side::Codomain)
        .unwrap();
    assert!(std::ptr::eq(codomain.provider(), left_provider.as_ref()));
    assert_eq!(codomain.codomain()[0].degeneracy(&label).unwrap(), 3);
    assert_sun_cat_values(&codomain, &codomain_lhs, &codomain_rhs, 0, 1, value);
    let codomain_rhs_adjoint = codomain_rhs.adjoint(engine).unwrap();
    let lazy_codomain = codomain_lhs
        .adjoint(engine)
        .unwrap()
        .cat(engine, &codomain_rhs_adjoint, Side::Domain)
        .unwrap();
    assert_eq!(
        lazy_codomain.dense_data().unwrap(),
        codomain
            .adjoint(engine)
            .unwrap()
            .materialize(engine)
            .unwrap()
            .dense_data()
            .unwrap()
    );
}

#[cfg(feature = "cgc-gen")]
#[test]
fn sun_checked_generic_cat_covers_both_directions_dtypes_and_mu_two_keys() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: exact TensorKit direct-sum slab values, μ=2 full-key matching,
    // distinct equal-identity Arcs, left authority, and lazy-adjoint parity.
    for (n, label) in [(3, vec![1, 1]), (4, vec![1, 0, 1])] {
        assert_sun_cat_case(engine, n, label.clone(), |value| value as f64);
        assert_sun_cat_case(engine, n, label, |value| {
            Complex64::new(value as f64, -(value as f64))
        });
    }
}

#[cfg(feature = "cgc-gen")]
#[test]
fn sun_adjoint_multiplicity_transforms_round_trip_labels_vertices_and_payload() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    use symtn::sector::LieGroupFusionRule;

    let runtime = Runtime::builder().build().unwrap();
    for (n, adjoint) in [(3, vec![1, 1]), (4, vec![1, 0, 1])] {
        let provider = Arc::new(
            LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(n).unwrap()).unwrap(),
        );
        let leg = GradedSpace::try_new(Arc::clone(&provider), [(adjoint.clone(), 1)]).unwrap();
        let tensor: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg], [&leg], |trees, _| {
                trees.codomain_vertices()[0].get() as f64
            })
            .unwrap();
        assert_eq!(tensor.subblock_count(), 2);
        for index in 0..2 {
            let trees = tensor.subblock_fusion_trees(index).unwrap();
            assert_eq!(trees.coupled(), &adjoint);
            assert_eq!(
                trees.codomain_uncoupled(),
                &[adjoint.clone(), adjoint.clone()]
            );
            assert_eq!(trees.domain_uncoupled(), std::slice::from_ref(&adjoint));
            assert_eq!(trees.codomain_vertices()[0].get(), index + 1);
            assert_eq!(tensor.subblock(index).unwrap().shape(), &[1, 1, 1]);
        }
        assert_eq!(tensor.dense_data().unwrap(), &[1.0, 2.0]);

        let identity: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, _| 1.0).unwrap();
        for output in [
            tensor
                .contract(
                    engine,
                    &identity,
                    &ContractSpec {
                        lhs: &[2],
                        rhs: &[0],
                        codomain: &[0, 1],
                        domain: &[2],
                    },
                )
                .unwrap(),
            tensor.compose(engine, &identity).unwrap(),
        ] {
            assert!(std::ptr::eq(output.provider(), provider.as_ref()));
            assert_eq!(output.dense_data().unwrap(), tensor.dense_data().unwrap());
            for index in 0..tensor.subblock_count() {
                assert_eq!(
                    output.subblock_fusion_trees(index).unwrap(),
                    tensor.subblock_fusion_trees(index).unwrap()
                );
            }
        }

        let product = tensor.otimes(engine, &tensor).unwrap();
        assert!(std::ptr::eq(product.provider(), provider.as_ref()));
        let (expected_len, expected_sum, expected_weighted, expected_prefix): (
            usize,
            f64,
            f64,
            &[f64],
        ) = match n {
            3 => (
                145,
                9.468_841_418_575_323,
                39.231_504_693_264_13,
                &[
                    0.0,
                    1.0,
                    2.0,
                    2.0,
                    4.0,
                    0.0,
                    0.0,
                    0.0,
                    0.353_553_390_593_273_6,
                    0.707_106_781_186_547_2,
                    0.0,
                    0.857_142_857_142_857,
                ],
            ),
            4 => (
                245,
                8.608_165_620_335_726,
                -1.317_392_645_553_582,
                &[
                    0.0,
                    0.0,
                    1.0,
                    2.0,
                    2.0,
                    4.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.338_873_675_850_995_87,
                    0.677_747_351_701_991_7,
                ],
            ),
            _ => unreachable!(),
        };
        assert_eq!(product.dense_data().unwrap().len(), expected_len);
        let sum = product.dense_data().unwrap().iter().sum::<f64>();
        let weighted = product
            .dense_data()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(index, value)| (index + 1) as f64 * value)
            .sum::<f64>();
        assert!((sum - expected_sum).abs() <= 1e-10);
        assert!((weighted - expected_weighted).abs() <= 1e-9);
        for (&actual, &expected) in product.dense_data().unwrap().iter().zip(expected_prefix) {
            assert!((actual - expected).abs() <= 1e-10);
        }
        let mut adjoint_root_vertices = Vec::new();
        for index in 0..product.subblock_count() {
            let trees = product.subblock_fusion_trees(index).unwrap();
            if trees.coupled() == &adjoint
                && product.dense_data().unwrap()[product.subblock(index).unwrap().offset()].abs()
                    > 1e-10
            {
                assert_eq!(trees.codomain_uncoupled(), vec![adjoint.clone(); 4]);
                assert_eq!(trees.domain_uncoupled(), vec![adjoint.clone(); 2]);
                adjoint_root_vertices.push(trees.domain_vertices().last().unwrap().get());
            }
        }
        adjoint_root_vertices.sort_unstable();
        adjoint_root_vertices.dedup();
        assert_eq!(adjoint_root_vertices, [1, 2]);

        let snapshot = |tensor: &TensorMap<LieGroupFusionRule, f64>| {
            (0..tensor.subblock_count())
                .map(|index| tensor.subblock_fusion_trees(index).unwrap())
                .collect::<Vec<_>>()
        };
        let source_snapshot = snapshot(&tensor);
        for restored in [
            tensor
                .permute(engine, &[1, 0], &[2])
                .unwrap()
                .permute(engine, &[1, 0], &[2])
                .unwrap(),
            tensor
                .braid(engine, &[0, 2], &[1], &[0, 1, 2])
                .unwrap()
                .braid(engine, &[0, 2], &[1], &[0, 1, 2])
                .unwrap(),
            tensor
                .repartition(engine, 1)
                .unwrap()
                .repartition(engine, 2)
                .unwrap(),
            tensor
                .transpose(engine, &[2], &[1, 0])
                .unwrap()
                .transpose(engine, &[2, 1], &[0])
                .unwrap(),
        ] {
            assert!(std::ptr::eq(restored.provider(), provider.as_ref()));
            assert_eq!(snapshot(&restored), source_snapshot);
            for (actual, expected) in restored
                .dense_data()
                .unwrap()
                .iter()
                .zip(tensor.dense_data().unwrap())
            {
                assert!((actual - expected).abs() <= 1e-10);
            }
        }
    }
}

#[cfg(feature = "cgc-gen")]
#[test]
fn sun_checked_generic_transforms_reuse_the_runtime_completed_store() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    use symtn::sector::LieGroupFusionRule;
    if crate::checked_generic_facade::run_isolated_or_return(
        "SYMTN_CHECKED_FACADE_COMPLETED_STORE",
        "transforms::sun_checked_generic_transforms_reuse_the_runtime_completed_store",
    ) {
        return;
    }

    let runtime = Runtime::builder().build().unwrap();
    for (n, adjoint) in [(3, vec![1, 1]), (4, vec![1, 0, 1])] {
        let provider = Arc::new(
            LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(n).unwrap()).unwrap(),
        );
        let leg = GradedSpace::try_new(Arc::clone(&provider), [(adjoint, 1)]).unwrap();
        let source: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg], [&leg], |trees, _| {
                trees.codomain_vertices()[0].get() as f64
            })
            .unwrap();

        for operation in ["permute", "braid", "repartition"] {
            symtn::cache::clear();
            let mut apply = |tensor: &TensorMap<LieGroupFusionRule, f64>| match operation {
                "permute" => tensor.permute(engine, &[1, 0], &[2]),
                "braid" => tensor.braid(engine, &[1, 0], &[2], &[0, 1, 2]),
                "repartition" => tensor.repartition(engine, 1),
                _ => unreachable!(),
            };
            let first = apply(&source).unwrap();
            let cold = crate::checked_generic_facade::completed_transformers();
            let repeated = apply(&source).unwrap();
            let warm = crate::checked_generic_facade::completed_transformers();

            assert_eq!(cold.entries(), 1);
            assert_eq!(cold.misses(), 1);
            assert_eq!(warm.entries(), 1);
            assert_eq!(warm.misses(), 1);
            assert_eq!(warm.hits(), 1);
            assert_eq!(repeated.dense_data().unwrap(), first.dense_data().unwrap());
            assert!(std::ptr::eq(first.provider(), provider.as_ref()));
            assert!(std::ptr::eq(repeated.provider(), provider.as_ref()));
            for index in 0..first.subblock_count() {
                assert_eq!(
                    repeated.subblock_fusion_trees(index).unwrap(),
                    first.subblock_fusion_trees(index).unwrap()
                );
            }
        }
    }
}

/// Rank 2+2 checked fixture: `X (x) X -> X` carries outer multiplicity two,
/// the coupled sector ranges over `{Vacuum, X}`, degeneracies two and three.
fn checked_multiplicity_lazy_fixture(
    runtime: &Runtime,
    provider: &Arc<CheckedOnlyToy>,
) -> TensorMap<CheckedOnlyToy, Complex64> {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let x = GradedSpace::try_new(Arc::clone(provider), [(Label::X, 2)]).unwrap();
    let mixed =
        GradedSpace::try_new(Arc::clone(provider), [(Label::Vacuum, 1), (Label::X, 3)]).unwrap();
    let tensor =
        TensorMap::from_subblock_fn(engine, runtime, [&x, &x], [&x, &mixed], lazy_oracle_value)
            .unwrap();
    let coupled: std::collections::BTreeSet<_> = (0..tensor.subblock_count())
        .map(|index| *tensor.subblock_fusion_trees(index).unwrap().coupled())
        .collect();
    assert_eq!(coupled.len(), 2);
    assert!((0..tensor.subblock_count()).any(|index| {
        tensor
            .subblock_fusion_trees(index)
            .unwrap()
            .codomain_vertices()[0]
            .get()
            == 2
    }));
    tensor
}

/// Asserts `lazy.data()` against the literal pre-#1201 formula, builds an
/// Owned twin from that literal payload, and checks that every listed
/// transform gives the same blocks (or the same error) on both. `$ops` is a
/// closure `(lazy, owned) -> [(name, Result, Result); N]`.
macro_rules! assert_checked_lazy_adjoint_matches_literal {
    ($engine:ident, $parent:expr, $conj:expr, $close:expr, $ops:expr) => {{
        let parent = $parent;
        let lazy = parent.adjoint($engine).unwrap();
        let parent_snapshot = snapshot!($engine, parent);
        let lazy_snapshot = snapshot!($engine, lazy);
        let literal = common::literal_adjoint_payload(&parent_snapshot, &lazy_snapshot, $conj);
        assert_eq!(
            literal.len(),
            lazy.materialize($engine)
                .unwrap()
                .dense_data()
                .unwrap()
                .len()
        );
        for (actual, expected) in lazy
            .materialize($engine)
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .zip(&literal)
        {
            assert!($close(*actual, *expected), "{actual:?} != {expected:?}");
        }
        let codomain = lazy.codomain();
        let domain = lazy.domain();
        let owned = TensorMap::from_subblock_fn(
            $engine,
            parent.runtime(),
            &codomain,
            &domain,
            |trees, indices| {
                let (_, geometry) = lazy_snapshot
                    .blocks
                    .iter()
                    .find(|(candidate, _)| candidate == trees)
                    .unwrap();
                literal[common::linear(geometry, indices)]
            },
        )
        .unwrap();
        common::assert_same_tensor(&snapshot!($engine, owned), &lazy_snapshot, $close);

        let mut succeeded = 0usize;
        for (name, actual, expected) in $ops($engine, &lazy, &owned) {
            match (actual, expected) {
                (Ok(actual), Ok(expected)) => {
                    succeeded += 1;
                    assert!(std::ptr::eq(actual.provider(), parent.provider()));
                    common::assert_same_tensor(
                        &snapshot!($engine, actual),
                        &snapshot!($engine, expected),
                        $close,
                    );
                }
                (Err(actual), Err(expected)) => {
                    assert_eq!(format!("{actual:?}"), format!("{expected:?}"), "{name}")
                }
                (actual, expected) => panic!("{name}: lazy {actual:?} vs owned {expected:?}"),
            }
        }
        succeeded
    }};
}

#[test]
fn checked_multiplicity_lazy_adjoint_matches_the_literal_kernel_for_real_and_complex() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    // The toy's F-symbols are the identity on `e == f` and zero otherwise, so
    // most recouplings are unrepresentable and fail identically for the lazy
    // and the Owned input (error parity is asserted); the real SU(3) sibling
    // below covers every transform. Bends on a one-sided tree do succeed and
    // exercise the lazy adjoint's parent route.
    type ToyStep<D> = (
        &'static str,
        Result<TensorMap<CheckedOnlyToy, D>, GenericTensorError<ToyError>>,
        Result<TensorMap<CheckedOnlyToy, D>, GenericTensorError<ToyError>>,
    );
    fn two_sided_ops<D: symtn::typed::TensorScalar>(
        engine: &mut dyn Engine,
        lazy: &TensorMap<CheckedOnlyToy, D>,
        owned: &TensorMap<CheckedOnlyToy, D>,
    ) -> [ToyStep<D>; 5] {
        [
            (
                "permute",
                lazy.permute(engine, &[1, 0], &[2, 3]),
                owned.permute(engine, &[1, 0], &[2, 3]),
            ),
            (
                "braid",
                lazy.braid(engine, &[1, 0], &[2, 3], &[1, 0, 2, 3]),
                owned.braid(engine, &[1, 0], &[2, 3], &[1, 0, 2, 3]),
            ),
            (
                "repartition",
                lazy.repartition(engine, 1),
                owned.repartition(engine, 1),
            ),
            (
                "full transpose",
                lazy.transpose(engine, &[3, 2], &[1, 0]),
                owned.transpose(engine, &[3, 2], &[1, 0]),
            ),
            (
                "transpose",
                lazy.transpose(engine, &[1, 3], &[0, 2]),
                owned.transpose(engine, &[1, 3], &[0, 2]),
            ),
        ]
    }
    // Lazy space `[] <- X, X, X`. Its permute and braid are the reference
    // steps `adjoint(permute(parent, adjointtensorindices))` (TensorKit
    // `indexmanipulations.jl:260-264, 343-352`): a codomain braid of the
    // parent `X, X, X <- []`, which the toy represents, where the Owned twin's
    // domain braid needs the toy's empty recouplings.
    fn one_sided_ops<D: symtn::typed::TensorScalar>(
        engine: &mut dyn Engine,
        lazy: &TensorMap<CheckedOnlyToy, D>,
        owned: &TensorMap<CheckedOnlyToy, D>,
    ) -> [ToyStep<D>; 5] {
        let parent = lazy.adjoint(engine).unwrap();
        assert!(owned.permute(engine, &[], &[1, 0, 2]).is_err());
        assert!(owned.braid(engine, &[], &[1, 0, 2], &[0, 1, 2]).is_err());
        [
            (
                "permute",
                lazy.permute(engine, &[], &[1, 0, 2]),
                parent
                    .permute(engine, &[1, 0, 2], &[])
                    .and_then(|t| t.adjoint(engine)),
            ),
            (
                "braid",
                lazy.braid(engine, &[], &[1, 0, 2], &[0, 1, 2]),
                parent
                    .braid(engine, &[1, 0, 2], &[], &[0, 1, 2])
                    .and_then(|t| t.adjoint(engine)),
            ),
            (
                "repartition",
                lazy.repartition(engine, 1),
                owned.repartition(engine, 1),
            ),
            (
                "full transpose",
                lazy.transpose(engine, &[2, 1, 0], &[]),
                owned.transpose(engine, &[2, 1, 0], &[]),
            ),
            (
                "transpose",
                lazy.transpose(engine, &[2], &[0, 1]),
                owned.transpose(engine, &[2], &[0, 1]),
            ),
        ]
    }

    let two_sided = checked_multiplicity_lazy_fixture(&runtime, &provider);
    assert!(two_sided
        .dense_data()
        .unwrap()
        .iter()
        .any(|value| value.im != 0.0));
    assert_checked_lazy_adjoint_matches_literal!(
        engine,
        two_sided.clone(),
        |z: Complex64| z.conj(),
        lazy_close_c64,
        two_sided_ops
    );
    let two_sided_real = two_sided.re(engine).unwrap();
    assert_checked_lazy_adjoint_matches_literal!(
        engine,
        two_sided_real,
        |x: f64| x,
        lazy_close_f64,
        two_sided_ops
    );

    let x = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let one_sided =
        TensorMap::from_subblock_fn(engine, &runtime, [&x, &x, &x], [], lazy_oracle_value).unwrap();
    assert!((0..one_sided.subblock_count()).any(|index| {
        one_sided
            .subblock_fusion_trees(index)
            .unwrap()
            .codomain_vertices()[0]
            .get()
            == 2
    }));
    let complex_succeeded = assert_checked_lazy_adjoint_matches_literal!(
        engine,
        one_sided.clone(),
        |z: Complex64| z.conj(),
        lazy_close_c64,
        one_sided_ops
    );
    let one_sided_real = one_sided.re(engine).unwrap();
    let real_succeeded = assert_checked_lazy_adjoint_matches_literal!(
        engine,
        one_sided_real,
        |x: f64| x,
        lazy_close_f64,
        one_sided_ops
    );
    assert!(complex_succeeded >= 2 && real_succeeded >= 2);
}

/// Real SU(3) multiplicity: all five transforms on a lazy adjoint against the
/// literal-payload Owned twin.
#[cfg(feature = "cgc-gen")]
#[test]
fn sun_lazy_adjoint_matches_the_literal_kernel_under_all_transforms() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    use symtn::sector::LieGroupFusionRule;

    let runtime = Runtime::builder().build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let adjoint = vec![1, 1];
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(adjoint.clone(), 2)]).unwrap();
    let other =
        GradedSpace::try_new(Arc::clone(&provider), [(vec![0, 0], 1), (adjoint, 1)]).unwrap();
    let complex = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&leg, &leg],
        [&other, &leg],
        |trees, indices| {
            let tree = (format!("{trees:?}").bytes().fold(0u32, |acc, byte| {
                acc.wrapping_mul(31).wrapping_add(byte as u32)
            }) % 97) as f64;
            let mut re = tree;
            let mut im = -0.5 * tree + 0.25;
            for (axis, &index) in indices.iter().enumerate() {
                re += (index as f64 + 1.0) * (axis as f64 + 1.0);
                im += (index as f64 + 1.0) * (axis as f64 + 1.0) * (axis as f64 + 1.0) * 0.5;
            }
            Complex64::new(re, im)
        },
    )
    .unwrap();
    assert!((0..complex.subblock_count()).any(|index| {
        complex
            .subblock_fusion_trees(index)
            .unwrap()
            .codomain_vertices()[0]
            .get()
            == 2
    }));
    type SunStep<D> = (
        &'static str,
        Result<
            TensorMap<LieGroupFusionRule, D>,
            GenericTensorError<symtn::sector::LieGroupFusionRuleError>,
        >,
        Result<
            TensorMap<LieGroupFusionRule, D>,
            GenericTensorError<symtn::sector::LieGroupFusionRuleError>,
        >,
    );
    fn ops<D: symtn::typed::TensorScalar>(
        engine: &mut dyn Engine,
        lazy: &TensorMap<LieGroupFusionRule, D>,
        owned: &TensorMap<LieGroupFusionRule, D>,
    ) -> [SunStep<D>; 5] {
        [
            (
                "permute",
                lazy.permute(engine, &[1, 3], &[0, 2]),
                owned.permute(engine, &[1, 3], &[0, 2]),
            ),
            (
                "braid",
                lazy.braid(engine, &[1, 3], &[0, 2], &[3, 1, 2, 0]),
                owned.braid(engine, &[1, 3], &[0, 2], &[3, 1, 2, 0]),
            ),
            (
                "repartition",
                lazy.repartition(engine, 1),
                owned.repartition(engine, 1),
            ),
            (
                "full transpose",
                lazy.transpose(engine, &[3, 2], &[1, 0]),
                owned.transpose(engine, &[3, 2], &[1, 0]),
            ),
            (
                "transpose",
                lazy.transpose(engine, &[1, 3], &[0, 2]),
                owned.transpose(engine, &[1, 3], &[0, 2]),
            ),
        ]
    }
    let succeeded = assert_checked_lazy_adjoint_matches_literal!(
        engine,
        complex.clone(),
        |z: Complex64| z.conj(),
        lazy_close_c64,
        ops
    );
    assert_eq!(succeeded, 5);
    let real = complex.re(engine).unwrap();
    let succeeded =
        assert_checked_lazy_adjoint_matches_literal!(engine, real, |x: f64| x, lazy_close_f64, ops);
    assert_eq!(succeeded, 5);
}
