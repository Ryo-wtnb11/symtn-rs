use super::*;

#[test]
fn checked_generic_diagonal_is_compact_canonical_and_provider_owned() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let bond = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2), (Label::Vacuum, 1)])
        .unwrap()
        .try_dual()
        .unwrap();
    let real = TensorMap::<_, f64>::diagonal(
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: Label::X,
                values: vec![2.0, 3.0],
            },
            SectorSpectrum {
                sector: Label::Vacuum,
                values: vec![1.0],
            },
        ],
    )
    .unwrap();
    assert!(std::ptr::eq(real.provider(), provider.as_ref()));
    assert_eq!(real.codomain()[0], bond);
    assert_eq!(real.domain()[0], bond);
    assert_eq!(
        symtn::expert::diagonal_spectrum(&real).unwrap().unwrap(),
        [
            SectorSpectrum {
                sector: Label::Vacuum,
                values: vec![1.0],
            },
            SectorSpectrum {
                sector: Label::X,
                values: vec![2.0, 3.0],
            },
        ]
    );
    let adjoint = real.adjoint().unwrap();
    assert!(std::ptr::eq(adjoint.provider(), provider.as_ref()));
    assert!(
        symtn::typed::__network::network_reuse_class(&adjoint, false) == NetworkReuseClass::Compact
    );
    assert_eq!(
        real.materialize().unwrap().dense_data().unwrap(),
        &[1.0, 2.0, 0.0, 0.0, 3.0]
    );
    assert_eq!(
        adjoint.materialize().unwrap().dense_data().unwrap(),
        real.materialize().unwrap().dense_data().unwrap()
    );
    assert_eq!(
        symtn::expert::diagonal_spectrum(&adjoint.adjoint().unwrap())
            .unwrap()
            .unwrap(),
        symtn::expert::diagonal_spectrum(&real).unwrap().unwrap()
    );

    let complex = TensorMap::<_, Complex64>::diagonal(
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: Label::Vacuum,
                values: vec![Complex64::new(1.0, 1.0)],
            },
            SectorSpectrum {
                sector: Label::X,
                values: vec![Complex64::new(2.0, -1.0), Complex64::new(3.0, 2.0)],
            },
        ],
    )
    .unwrap();
    assert!(std::ptr::eq(complex.provider(), provider.as_ref()));
    assert_eq!(
        complex
            .adjoint()
            .unwrap()
            .materialize()
            .unwrap()
            .dense_data()
            .unwrap()[0],
        Complex64::new(1.0, -1.0)
    );
}

#[test]
fn checked_generic_diagonal_rejects_before_layout_and_preserves_error_precedence() {
    let _cache = cache_shared();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new_product_probe(0));
    let bond =
        GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 1), (Label::X, 2)]).unwrap();

    let compact = TensorMap::<_, f64>::diagonal(
        &runtime,
        &bond,
        [
            SectorSpectrum {
                sector: Label::Vacuum,
                values: vec![1.0],
            },
            SectorSpectrum {
                sector: Label::X,
                values: vec![2.0, 3.0],
            },
        ],
    )
    .unwrap();
    provider.fail_decode.store(true, Ordering::Relaxed);
    assert!(matches!(
        symtn::expert::diagonal_spectrum(&compact),
        Err(GenericTensorError::Structure(
            CheckedGenericStructureError::Provider(ToyError::Decode)
        ))
    ));
    provider.fail_decode.store(false, Ordering::Relaxed);

    reset_provider_queries(&provider);
    assert!(matches!(
        TensorMap::<_, f64>::diagonal(
            &runtime,
            &bond,
            [
                SectorSpectrum { sector: Label::Invalid, values: vec![1.0] },
                SectorSpectrum { sector: Label::Invalid, values: vec![2.0] },
            ],
        ),
        Err(GenericTensorError::Facade(symtn::typed::Error::InvalidArgument(message)))
            if message.contains("more than once")
    ));
    assert_no_provider_queries(&provider);

    assert!(matches!(
        TensorMap::<_, f64>::diagonal(
            &runtime,
            &bond,
            [
                SectorSpectrum {
                    sector: Label::X,
                    values: vec![1.0, 2.0]
                },
                SectorSpectrum {
                    sector: Label::Invalid,
                    values: vec![3.0]
                },
            ],
        ),
        Err(GenericTensorError::Structure(
            CheckedGenericStructureError::Provider(ToyError::InvalidSector)
        ))
    ));
    assert!(matches!(
        TensorMap::<_, f64>::diagonal(
            &runtime,
            &bond,
            [
                SectorSpectrum { sector: Label::X, values: vec![1.0, 2.0] },
                SectorSpectrum { sector: Label::AliasX, values: vec![3.0, 4.0] },
            ],
        ),
        Err(GenericTensorError::Facade(symtn::typed::Error::InvalidArgument(message)))
            if message.contains("both encode")
    ));

    provider.invalid_style.store(true, Ordering::Relaxed);
    for (spectra, expected) in [
        (
            vec![SectorSpectrum {
                sector: Label::Vacuum,
                values: vec![1.0],
            }],
            "missing",
        ),
        (
            vec![
                SectorSpectrum {
                    sector: Label::Vacuum,
                    values: vec![1.0],
                },
                SectorSpectrum {
                    sector: Label::One,
                    values: vec![2.0],
                },
            ],
            "missing",
        ),
        (
            vec![
                SectorSpectrum {
                    sector: Label::Vacuum,
                    values: vec![1.0],
                },
                SectorSpectrum {
                    sector: Label::X,
                    values: vec![2.0],
                },
                SectorSpectrum {
                    sector: Label::One,
                    values: vec![4.0],
                },
            ],
            "unknown",
        ),
        (
            vec![
                SectorSpectrum {
                    sector: Label::Vacuum,
                    values: vec![1.0],
                },
                SectorSpectrum {
                    sector: Label::X,
                    values: vec![2.0],
                },
            ],
            "length",
        ),
    ] {
        assert!(matches!(
            TensorMap::<_, f64>::diagonal(&runtime, &bond, spectra),
            Err(GenericTensorError::Facade(symtn::typed::Error::InvalidArgument(message)))
                if message.contains(expected)
        ));
    }

    assert!(matches!(
        TensorMap::<_, f64>::diagonal(
            &runtime,
            &bond,
            [
                SectorSpectrum {
                    sector: Label::Vacuum,
                    values: vec![1.0]
                },
                SectorSpectrum {
                    sector: Label::X,
                    values: vec![2.0, 3.0]
                },
            ],
        ),
        Err(GenericTensorError::Structure(_))
    ));
}

#[cfg(feature = "cgc-gen")]
#[test]
fn sun_checked_generic_diagonal_constructs_standalone_compact_blocks() {
    use symtn::sector::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    for (n, adjoint) in [(3, vec![1, 1]), (4, vec![1, 0, 1])] {
        let provider = Arc::new(SUNFusionRule::new(n).unwrap());
        let adjoint_id = provider.encode_dynkin(&adjoint).unwrap();
        assert_eq!(
            provider
                .try_nsymbol(adjoint_id, adjoint_id, adjoint_id)
                .unwrap(),
            2
        );
        let bond = GradedSpace::try_new(Arc::clone(&provider), [(adjoint.clone(), 2)]).unwrap();
        let diagonal = TensorMap::<_, f64>::diagonal(
            &runtime,
            &bond,
            [SectorSpectrum {
                sector: adjoint,
                values: vec![2.0, 3.0],
            }],
        )
        .unwrap();
        assert!(std::ptr::eq(diagonal.provider(), provider.as_ref()));
        assert!(symtn::expert::diagonal_spectrum(&diagonal)
            .unwrap()
            .is_some());
        // This provider has outer multiplicity two, but a rank-one spectrum has no μ axis.
        assert_eq!(diagonal.subblock_count(), 1);
        assert_eq!(diagonal.subblock(0).unwrap().shape(), &[2, 2]);
        assert_eq!(diagonal.subblock(0).unwrap().strides(), &[1, 2]);
        assert_eq!(
            diagonal.materialize().unwrap().dense_data().unwrap(),
            &[2.0, 0.0, 0.0, 3.0]
        );
    }
}

#[test]
fn checked_generic_space_algebra_keeps_multiplicity_dimensions_and_failures_typed() {
    let _cache = cache_shared();
    let provider = Arc::new(CheckedOnlyToy::new_space_probe(7));
    let rhs_provider = Arc::new(CheckedOnlyToy::new_space_probe(7));
    let left = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let right = GradedSpace::try_new(Arc::clone(&rhs_provider), [(Label::X, 3)]).unwrap();

    let dim = left.dim().unwrap();
    assert!((dim - 5.0).abs() < 1.0e-12);
    assert_ne!(dim, 6.0);
    let fused = left.fuse(&right).unwrap();
    assert_eq!(fused.degeneracy(&Label::X).unwrap(), 12);
    assert!(std::ptr::eq(fused.provider(), provider.as_ref()));
    assert!(!std::ptr::eq(fused.provider(), rhs_provider.as_ref()));
    let summed = left.oplus(&right).unwrap();
    assert_eq!(summed.degeneracy(&Label::X).unwrap(), 5);
    assert!(std::ptr::eq(summed.provider(), provider.as_ref()));
    assert!(!std::ptr::eq(summed.provider(), rhs_provider.as_ref()));
    let unit = left.unitspace().unwrap();
    assert!(std::ptr::eq(unit.provider(), provider.as_ref()));
    assert_eq!(unit.degeneracy(&Label::Vacuum).unwrap(), 1);

    let foreign_provider = Arc::new(CheckedOnlyToy::new_space_probe(8));
    let foreign = GradedSpace::try_new(Arc::clone(&foreign_provider), [(Label::X, 1)]).unwrap();
    let before = provider.algebra_queries.load(Ordering::Relaxed)
        + foreign_provider.algebra_queries.load(Ordering::Relaxed);
    assert!(matches!(
        left.oplus(&foreign),
        Err(GenericTensorError::Facade(
            symtn::typed::Error::RuleMismatch
        ))
    ));
    assert!(matches!(
        left.fuse(&foreign),
        Err(GenericTensorError::Facade(
            symtn::typed::Error::RuleMismatch
        ))
    ));
    assert_eq!(
        provider.algebra_queries.load(Ordering::Relaxed)
            + foreign_provider.algebra_queries.load(Ordering::Relaxed),
        before
    );

    provider.fail_algebra.store(true, Ordering::Relaxed);
    assert!(matches!(
        left.fuse(&right),
        Err(GenericTensorError::Structure(
            CheckedGenericStructureError::Provider(ToyError::Algebra)
        ))
    ));
    assert_eq!(left.degeneracy(&Label::X).unwrap(), 2);
    assert_eq!(right.degeneracy(&Label::X).unwrap(), 3);
}

#[test]
fn checked_only_provider_uses_ordinary_typed_ownership_and_vertices() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let first = Arc::new(CheckedOnlyToy::new(0));
    let second = Arc::new(CheckedOnlyToy::new(0));
    let left = GradedSpace::try_new(Arc::clone(&first), [(Label::X, 2)]).unwrap();
    let right = GradedSpace::try_new(Arc::clone(&second), [(Label::X, 2)]).unwrap();

    let tensor: TensorMap<_, f64> = TensorMap::zeros(&runtime, [&left, &right], [&right]).unwrap();
    assert!(std::ptr::eq(tensor.provider(), first.as_ref()));
    assert_eq!(tensor.rank(), 3);
    assert_eq!(tensor.subblock_count(), 2);
    let vertices: Vec<_> = (0..tensor.subblock_count())
        .map(|index| {
            let trees = tensor.subblock_fusion_trees(index).unwrap();
            assert_eq!(trees.coupled(), &Label::X);
            assert_eq!(trees.codomain_uncoupled(), &[Label::X, Label::X]);
            assert!(trees.codomain_innerlines().is_empty());
            assert_eq!(trees.domain_uncoupled(), &[Label::X]);
            assert!(trees.domain_vertices().is_empty());
            trees.codomain_vertices()[0].get()
        })
        .collect();
    assert_eq!(vertices, [1, 2]);
    assert_eq!(left.sectors().unwrap(), [Label::X]);

    let clone = tensor.clone();
    assert!(std::ptr::eq(clone.provider(), first.as_ref()));
    assert_eq!(
        clone.dense_data().unwrap().as_ptr(),
        tensor.dense_data().unwrap().as_ptr()
    );
}

#[test]
fn checked_generic_subblocks_decode_transactionally_and_keep_outer_multiplicity() {
    let _cache = cache_shared();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let tensor: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg, &leg], |trees, _| {
            100.0 * trees.codomain_vertices()[0].get() as f64
                + 10.0 * trees.domain_vertices()[0].get() as f64
        })
        .unwrap();

    let mut vacuum_blocks = 0;
    let mut observed = tensor
        .subblocks()
        .unwrap()
        .filter_map(|(trees, values)| {
            assert_eq!(trees.codomain_uncoupled(), &[Label::X, Label::X]);
            assert_eq!(trees.domain_uncoupled(), &[Label::X, Label::X]);
            assert_eq!(values.shape(), &[1, 1, 1, 1]);
            assert_eq!(
                values.data().as_ptr(),
                tensor.dense_data().unwrap().as_ptr()
            );
            if trees.coupled() == &Label::Vacuum {
                vacuum_blocks += 1;
                assert_eq!(trees.codomain_vertices()[0].get(), 1);
                assert_eq!(trees.domain_vertices()[0].get(), 1);
                assert_eq!(values.get(&[0, 0, 0, 0]), Some(&110.0));
                return None;
            }
            assert_eq!(trees.coupled(), &Label::X);
            Some((
                trees.codomain_vertices()[0].get(),
                trees.domain_vertices()[0].get(),
                *values.get(&[0, 0, 0, 0]).unwrap() as usize,
            ))
        })
        .collect::<Vec<_>>();
    assert_eq!(vacuum_blocks, 1);
    observed.sort_unstable();
    assert_eq!(
        observed,
        [(1, 1, 110), (1, 2, 120), (2, 1, 210), (2, 2, 220)]
    );

    reset_provider_queries(&provider);
    // One rank-(2, 2) block decodes coupled + two codomain + two domain labels.
    // Each decode also queries the toy vacuum, so query eleven starts the
    // second block. The first block decoded fully, but no iterator containing
    // that prefix escapes.
    provider.fail_decode_on_query.store(11, Ordering::Relaxed);
    let result = tensor.subblocks();
    let queries = provider.queries_since_reset.load(Ordering::Relaxed);
    let error = result
        .err()
        .unwrap_or_else(|| panic!("late decode must fail; observed {queries} queries"));
    assert!(matches!(
        error,
        GenericTensorError::Structure(CheckedGenericStructureError::Provider(ToyError::Decode))
    ));
    assert_eq!(provider.queries_since_reset.load(Ordering::Relaxed), 11);
    provider.fail_decode_on_query.store(0, Ordering::Relaxed);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn checked_only_provider_roundtrips_through_typed_cuda_without_algebra_dispatch() {
    let _cache = cache_shared();
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let codomain =
        GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 1), (Label::X, 2)]).unwrap();
    let domain = GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 1), (Label::X, 3)])
        .and_then(|space| space.try_dual())
        .unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&codomain], [&domain], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();
    let vertex_leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let vertex_source: TensorMap<_, f64> = TensorMap::from_subblock_fn(
        &runtime,
        [&vertex_leg, &vertex_leg],
        [&vertex_leg],
        |trees, indices| {
            trees.codomain_vertices()[0].get() as f64 + indices.iter().sum::<usize>() as f64
        },
    )
    .unwrap();
    let block_structure = |tensor: &TensorMap<CheckedOnlyToy, f64>| {
        (0..tensor.subblock_count())
            .map(|index| {
                let block = tensor.subblock(index).unwrap();
                (
                    block.key().clone(),
                    tensor.subblock_fusion_trees(index).unwrap(),
                    block.offset(),
                    block.shape().to_vec(),
                    block.strides().to_vec(),
                )
            })
            .collect::<Vec<_>>()
    };
    let structure = |tensor: &TensorMap<CheckedOnlyToy, f64>| {
        let mut codomain_legs = Vec::new();
        let mut domain_legs = Vec::new();
        for index in 0..tensor.subblock_count() {
            let block = tensor.subblock(index).unwrap();
            let trees = tensor.subblock_fusion_trees(index).unwrap();
            let symtn::typed::BlockKey::FusionTree(raw_trees) = block.key() else {
                panic!("checked Generic tensors use fusion-tree block keys")
            };
            assert_eq!(trees.codomain_uncoupled().len(), 1);
            assert_eq!(trees.domain_uncoupled().len(), 1);
            assert_eq!(block.shape().len(), 2);
            codomain_legs.push((
                trees.codomain_uncoupled()[0],
                block.shape()[0],
                raw_trees.codomain_tree().is_dual()[0],
            ));
            domain_legs.push((
                trees.domain_uncoupled()[0],
                block.shape()[1],
                raw_trees.domain_tree().is_dual()[0],
            ));
        }
        codomain_legs.sort_unstable();
        codomain_legs.dedup();
        domain_legs.sort_unstable();
        domain_legs.dedup();
        (codomain_legs, domain_legs, block_structure(tensor))
    };
    let expected_structure = structure(&source);
    let expected_vertex_structure = block_structure(&vertex_source);
    assert_eq!(
        expected_structure.0,
        [(Label::Vacuum, 1, false), (Label::X, 2, false)]
    );
    assert_eq!(
        expected_structure.1,
        [(Label::Vacuum, 1, true), (Label::X, 3, true)]
    );
    let expected = source.dense_data().unwrap().to_vec();
    let expected_vertex_data = vertex_source.dense_data().unwrap().to_vec();
    provider.algebra_queries.store(0, Ordering::Relaxed);
    provider.coefficient_queries.store(0, Ordering::Relaxed);

    let device = source.to_cuda().unwrap();
    let restored = device.to_host().unwrap();
    let vertex_device = vertex_source.to_cuda().unwrap();
    let vertex_restored = vertex_device.to_host().unwrap();

    assert!(std::ptr::eq(restored.provider(), provider.as_ref()));
    assert_eq!(restored.dense_data().unwrap(), expected);
    assert_eq!(structure(&restored), expected_structure);
    assert_eq!(vertex_restored.dense_data().unwrap(), expected_vertex_data);
    assert_eq!(block_structure(&vertex_restored), expected_vertex_structure);
    assert_eq!(provider.algebra_queries.load(Ordering::Relaxed), 0);
    assert_eq!(provider.coefficient_queries.load(Ordering::Relaxed), 0);
}

/// QR/LQ/SVD and the value-only spectra of a facade-layout source (#1494),
/// each against an oracle that does not share the factorization's layout:
/// reconstruction, isometry against an explicit tree-basis identity, `S`
/// against `svd_vals`, and `eigh_vals`/`eig_vals` of `A Aᴴ` against the
/// squared singular values.
#[cfg(feature = "cgc-gen")]
macro_rules! assert_sun_compact_laws {
    ($runtime:expr, $source:expr) => {{
        let source = $source;
        let assert_close = |actual: &TensorMap<_, _>, expected: &TensorMap<_, _>, what: &str| {
            let error = actual
                .axpby(1.0.into(), expected, (-1.0).into())
                .unwrap()
                .norm(2.0)
                .unwrap();
            assert!(
                error < 1e-9 * (1.0 + expected.norm(2.0).unwrap()),
                "{what}: {error}"
            );
        };
        let owned_adjoint =
            |tensor: &TensorMap<_, _>| tensor.adjoint().unwrap().materialize().unwrap();
        let assert_identity = |gram: &TensorMap<_, _>, what: &str| {
            let (codomain, domain) = (gram.codomain(), gram.domain());
            let rank = codomain.iter().count();
            let identity = TensorMap::from_subblock_fn(
                $runtime,
                codomain.iter(),
                domain.iter(),
                |trees, ij| {
                    let same_tree = trees.codomain_uncoupled() == trees.domain_uncoupled()
                        && trees.codomain_innerlines() == trees.domain_innerlines()
                        && trees.codomain_vertices() == trees.domain_vertices();
                    let (row, col) = ij.split_at(rank);
                    if same_tree && row == col { 1.0 } else { 0.0 }.into()
                },
            )
            .unwrap();
            assert_close(gram, &identity, what);
        };
        let by_sector = |spectra: Vec<SectorSpectrum<_>>| {
            spectra
                .into_iter()
                .filter(|spectrum| !spectrum.values.is_empty())
                .map(|spectrum| {
                    let mut values = spectrum.values;
                    values.sort_by(|a, b| b.total_cmp(a));
                    (spectrum.sector, values)
                })
                .collect::<Vec<_>>()
        };
        fn find<S: PartialEq>(spectra: &[(S, Vec<f64>)], sector: &S) -> Vec<f64> {
            spectra
                .iter()
                .find(|(key, _)| key == sector)
                .map_or_else(Vec::new, |(_, values)| values.clone())
        }
        let assert_values = |actual: &[f64], expected: &[f64], what: &str| {
            let scale = 1.0 + expected.first().copied().unwrap_or(0.0).abs();
            for (index, &actual) in actual.iter().enumerate() {
                let expected = expected.get(index).copied().unwrap_or(0.0);
                assert!((actual - expected).abs() < 1e-9 * scale, "{what}");
            }
        };

        let Qr { q, r } = source
            .qr_compact(&codomain_axes(&source), &domain_axes(&source))
            .unwrap();
        assert_close(&q.compose(&r).unwrap(), source, "A = QR");
        assert_identity(&owned_adjoint(&q).compose(&q).unwrap(), "Qᴴ Q = 1");
        let Lq { l, q } = source
            .lq_compact(&codomain_axes(&source), &domain_axes(&source))
            .unwrap();
        assert_close(&l.compose(&q).unwrap(), source, "A = LQ");
        assert_identity(&q.compose(&owned_adjoint(&q)).unwrap(), "Q Qᴴ = 1");
        let Svd { u, s, vh } = source
            .svd_compact(&codomain_axes(&source), &domain_axes(&source))
            .unwrap();
        assert_close(
            &u.compose(&s).unwrap().compose(&vh).unwrap(),
            source,
            "A = U S Vh",
        );
        assert_identity(&owned_adjoint(&u).compose(&u).unwrap(), "Uᴴ U = 1");
        assert_identity(&vh.compose(&owned_adjoint(&vh)).unwrap(), "Vh Vhᴴ = 1");

        let singular = by_sector(
            source
                .svd_vals(&codomain_axes(&source), &domain_axes(&source))
                .unwrap(),
        );
        let diagonal = by_sector(s.eigh_vals(&[0], &[1], HermitianTol::DEFAULT).unwrap());
        assert_eq!(singular.len(), diagonal.len());
        for (sector, values) in &singular {
            assert_eq!(find(&diagonal, sector).len(), values.len());
            assert_values(&find(&diagonal, sector), values, "S = svd_vals");
        }
        let gram = source.compose(&owned_adjoint(source)).unwrap();
        let squared = singular
            .iter()
            .map(|(sector, values)| (sector.clone(), values.iter().map(|v| v * v).collect()))
            .collect::<Vec<_>>();
        for (sector, values) in by_sector(
            gram.eigh_vals(
                &codomain_axes(&gram),
                &domain_axes(&gram),
                HermitianTol::DEFAULT,
            )
            .unwrap(),
        ) {
            assert_values(&values, &find(&squared, &sector), "eigh_vals(A Aᴴ) = s²");
        }
        for spectrum in gram
            .eig_vals(&codomain_axes(&gram), &domain_axes(&gram))
            .unwrap()
        {
            let mut values = spectrum.values.iter().map(|v| v.re).collect::<Vec<_>>();
            assert!(spectrum
                .values
                .iter()
                .all(|v| v.im.abs() < 1e-9 * (1.0 + v.norm())));
            values.sort_by(|a, b| b.total_cmp(a));
            assert_values(
                &values,
                &find(&squared, &spectrum.sector),
                "eig_vals(A Aᴴ) = s²",
            );
        }
    }};
}

#[cfg(feature = "cgc-gen")]
#[test]
fn sun_checked_generic_compact_factors_and_spectra_on_facade_layouts() {
    use symtn::sector::SUNFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SUNFusionRule::new(3).unwrap());
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 1], 2), (vec![0, 0], 1)]).unwrap();
    let t =
        GradedSpace::try_new(Arc::clone(&provider), [(vec![1, 0], 2), (vec![0, 1], 1)]).unwrap();
    let t_dual = t.try_dual().unwrap();
    let value = |salt: u64| ((salt.wrapping_mul(2_654_435_761) % 1009) as f64) / 1009.0 - 0.5;
    let cases: [(&[&GradedSpace<_>], &[&GradedSpace<_>]); 7] = [
        (&[&leg, &leg], &[&leg]),
        (&[&leg], &[&leg, &leg]),
        (&[&leg, &leg], &[&leg, &leg]),
        (&[&t, &t_dual], &[&t]),
        (&[&t], &[&t, &t_dual]),
        (&[&t_dual, &t_dual], &[&t_dual]),
        (&[&t_dual, &t_dual, &t_dual], &[]),
    ];
    for (codomain, domain) in cases {
        let mut salt = 0;
        let real: TensorMap<_, f64> = TensorMap::from_subblock_fn(
            &runtime,
            codomain.iter().copied(),
            domain.iter().copied(),
            |_, _| {
                salt += 1;
                value(salt)
            },
        )
        .unwrap();
        assert_sun_compact_laws!(&runtime, &real);
        let complex: TensorMap<_, Complex64> = TensorMap::from_subblock_fn(
            &runtime,
            codomain.iter().copied(),
            domain.iter().copied(),
            |_, _| {
                salt += 2;
                Complex64::new(value(salt), value(salt + 1))
            },
        )
        .unwrap();
        assert_sun_compact_laws!(&runtime, &complex);
    }
}

#[test]
fn checked_errors_stay_typed_and_callback_waits_for_all_decodes() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let error = GradedSpace::try_new(Arc::clone(&provider), [(Label::Invalid, 1)]).unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Structure(CheckedGenericStructureError::Provider(
            ToyError::InvalidSector
        ))
    ));

    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    provider.fail_decode.store(true, Ordering::Relaxed);
    let callbacks = AtomicUsize::new(0);
    let error = TensorMap::<_, f64>::from_subblock_fn(&runtime, [&leg, &leg], [&leg], |_, _| {
        callbacks.fetch_add(1, Ordering::Relaxed);
        1.0
    })
    .unwrap_err();
    assert!(matches!(
        error,
        GenericTensorError::Structure(CheckedGenericStructureError::Provider(ToyError::Decode))
    ));
    assert_eq!(callbacks.load(Ordering::Relaxed), 0);
}

#[test]
fn identity_mismatch_precedes_algebra_queries_and_both_dtypes_fill() {
    let _cache = cache_shared();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let first = Arc::new(CheckedOnlyToy::new(0));
    let other = Arc::new(CheckedOnlyToy::new(1));
    let left = GradedSpace::try_new(Arc::clone(&first), [(Label::X, 1)]).unwrap();
    let right = GradedSpace::try_new(Arc::clone(&other), [(Label::X, 1)]).unwrap();
    let error = TensorMap::<_, f64>::zeros(&runtime, [&left], [&right]).unwrap_err();
    assert!(matches!(error, GenericTensorError::Facade(_)));
    assert_eq!(first.algebra_queries.load(Ordering::Relaxed), 0);
    assert_eq!(other.algebra_queries.load(Ordering::Relaxed), 0);

    let real: TensorMap<_, f64> =
        TensorMap::rand_with_seed(&runtime, [&left, &left], [&left], 7).unwrap();
    let complex: TensorMap<_, Complex64> =
        TensorMap::rand_with_seed(&runtime, [&left, &left], [&left], 7).unwrap();
    assert_eq!(
        real.dense_data().unwrap().len(),
        complex.dense_data().unwrap().len()
    );
    assert!(complex
        .dense_data()
        .unwrap()
        .iter()
        .any(|value| value.im != 0.0));
}

#[test]
fn failed_checked_admission_leaves_the_provider_usable() {
    // A failed admission must not leave provider-side state behind: the same
    // provider then builds exactly what a provider that never failed builds.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(7));
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let fresh = GradedSpace::try_new(Arc::new(CheckedOnlyToy::new(7)), [(Label::X, 1)]).unwrap();

    provider.fail_algebra.store(true, Ordering::Relaxed);
    assert!(TensorMap::<_, f64>::rand_with_seed(&runtime, [&leg, &leg], [&leg], 7).is_err());
    provider.fail_algebra.store(false, Ordering::Relaxed);

    let after_failure =
        TensorMap::<_, f64>::rand_with_seed(&runtime, [&leg, &leg], [&leg], 7).unwrap();
    let control =
        TensorMap::<_, f64>::rand_with_seed(&runtime, [&fresh, &fresh], [&fresh], 7).unwrap();
    assert_eq!(
        after_failure.dense_data().unwrap(),
        control.dense_data().unwrap()
    );
    assert_eq!(after_failure.codomain(), control.codomain());
}

/// The identity written sector by sector from its definition, independent of
/// `isomorphism`: `1` exactly where the codomain and domain fusion trees are
/// equal (uncoupled sectors, inner lines and vertex labels) and the codomain
/// and domain degeneracy indices agree.
fn equal_tree_identity<R>(runtime: &Runtime, legs: &[&GradedSpace<R>]) -> TensorMap<R, f64>
where
    R: TypedSectorAdmission,
    R::Mode: TypedTensorConstructionDispatch<R, f64>,
{
    let rank = legs.len();
    TensorMap::from_subblock_fn(
        runtime,
        legs.iter().copied(),
        legs.iter().copied(),
        |trees, ij| {
            let same_tree = trees.codomain_uncoupled() == trees.domain_uncoupled()
                && trees.codomain_innerlines() == trees.domain_innerlines()
                && trees.codomain_vertices() == trees.domain_vertices();
            f64::from(u8::from(same_tree && ij[..rank] == ij[rank..]))
        },
    )
    .unwrap()
}

/// #1559: `isomorphism(V, V)` is the identity on a checked-Generic provider
/// with fusion multiplicity, and composing it on either side returns the
/// operand; `isometry` satisfies `w† ∘ w = isomorphism(domain, domain)`.
fn check_generic_structural_constructors<R>(
    runtime: &Runtime,
    leg: &GradedSpace<R>,
    operand: impl Fn(&[usize]) -> f64,
) where
    R: TypedSectorAdmission<
            Error = <R as CheckedGenericFusion>::Error,
            Mode = CheckedGenericAdmissionMode,
        > + CheckedGenericRigidSymbols<Scalar = f64>,
{
    // The fixture must carry a subblock whose trees differ only in their
    // vertex labels; otherwise a vertex-blind oracle would pass unnoticed.
    let product: TensorMap<R, f64> = TensorMap::zeros(runtime, [leg, leg], [leg, leg]).unwrap();
    assert!(product.subblocks().unwrap().any(|(trees, _)| {
        trees.codomain_uncoupled() == trees.domain_uncoupled()
            && trees.codomain_innerlines() == trees.domain_innerlines()
            && trees.codomain_vertices() != trees.domain_vertices()
    }));
    for legs in [vec![leg], vec![leg, leg], vec![leg, leg, leg]] {
        let oracle = equal_tree_identity(runtime, &legs);
        let real: TensorMap<R, f64> =
            TensorMap::isomorphism(runtime, legs.iter().copied(), legs.iter().copied()).unwrap();
        assert_eq!(real.dense_data().unwrap(), oracle.dense_data().unwrap());
        let complex: TensorMap<R, Complex64> =
            TensorMap::isomorphism(runtime, legs.iter().copied(), legs.iter().copied()).unwrap();
        assert_eq!(
            complex.dense_data().unwrap(),
            oracle.convert::<Complex64>().dense_data().unwrap()
        );
    }

    let t: TensorMap<R, f64> =
        TensorMap::from_subblock_fn(runtime, [leg, leg], [leg], |_, ij| operand(ij)).unwrap();
    let left: TensorMap<R, f64> = TensorMap::isomorphism(runtime, [leg, leg], [leg, leg]).unwrap();
    let right: TensorMap<R, f64> = TensorMap::isomorphism(runtime, [leg], [leg]).unwrap();
    assert_eq!(
        left.compose(&t).unwrap().dense_data().unwrap(),
        t.dense_data().unwrap()
    );
    assert_eq!(
        t.compose(&right).unwrap().dense_data().unwrap(),
        t.dense_data().unwrap()
    );

    let w: TensorMap<R, f64> = TensorMap::isometry(runtime, [leg, leg], [leg]).unwrap();
    // Checked-Generic compose takes owned operands, so the lazy adjoint's
    // logical payload is copied into an owned tensor in stored order.
    let logical = w
        .adjoint()
        .unwrap()
        .materialize()
        .unwrap()
        .dense_data()
        .unwrap()
        .to_vec();
    let mut next = logical.into_iter();
    let w_dagger: TensorMap<R, f64> =
        TensorMap::from_subblock_fn(runtime, [leg], [leg, leg], |_, _| next.next().unwrap())
            .unwrap();
    let gram = w_dagger.compose(&w).unwrap();
    assert_eq!(gram.dense_data().unwrap(), right.dense_data().unwrap());
    let error = TensorMap::<R, f64>::isomorphism(runtime, [leg, leg], [leg]).unwrap_err();
    assert!(
        matches!(
            &error,
            GenericTensorError::Facade(symtn::typed::Error::Operation(operation))
                if matches!(**operation, symtn::typed::OperationError::SpaceMismatch { .. })
        ),
        "{error}"
    );
    let error = TensorMap::<R, f64>::isometry(runtime, [leg], [leg, leg]).unwrap_err();
    assert!(
        matches!(
            &error,
            GenericTensorError::Facade(symtn::typed::Error::Operation(operation))
                if matches!(**operation, symtn::typed::OperationError::SpaceMismatch { .. })
        ),
        "{error}"
    );
}

#[test]
fn checked_generic_isomorphism_is_the_equal_tree_identity_with_multiplicity() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    // X ⊗ X = 1 ⊕ 2X: every rank >= 2 side carries a multiplicity vertex.
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg = GradedSpace::try_new(provider, [(Label::Vacuum, 1), (Label::X, 2)]).unwrap();
    check_generic_structural_constructors(&runtime, &leg, |ij| {
        0.25 + (ij[0] * 5 + ij[1] * 3 + ij[2]) as f64
    });
}

#[cfg(feature = "cgc-gen")]
#[test]
fn su3_isomorphism_is_the_equal_tree_identity_with_multiplicity() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    // 8 ⊗ 8 contains 8 twice.
    let provider = Arc::new(symtn::sector::SUNFusionRule::new(3).unwrap());
    let leg = GradedSpace::try_new(provider, [(vec![1, 1], 2), (vec![0, 0], 1)]).unwrap();
    check_generic_structural_constructors(&runtime, &leg, |ij| {
        0.5 - (ij[0] * 7 + ij[1] * 2 + ij[2]) as f64 * 0.125
    });
}
