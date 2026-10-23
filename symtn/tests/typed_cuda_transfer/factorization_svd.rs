use super::*;

fn host_svd_trunc<R, D>(
    source: &TensorMap<R, D>,
    truncation: &Truncation,
    to_f64: impl Fn(D) -> f64,
) -> HostSvdTrunc<R, D>
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: symtn::typed::FactorizationScalar + symtn::typed::SpectrumMagnitude,
{
    let Svd { u, s, vh } = source
        .svd_compact(&codomain_axes(source), &domain_axes(source))
        .unwrap();
    let found = s.domain()[0]
        .find_truncated(&s.diagview().unwrap(), truncation)
        .unwrap();
    let s = s
        .restrict_leg(&[(0, &found.selection), (1, &found.selection)])
        .unwrap();
    HostSvdTrunc {
        u: u.restrict_leg(&[(u.codomain_rank(), &found.selection)])
            .unwrap(),
        vh: vh.restrict_leg(&[(0, &found.selection)]).unwrap(),
        singular_values: labelled_f64(&s, to_f64),
        s,
        error: found.error,
    }
}

/// The device truncated SVD: compact SVD on the device, one download per
/// factor, then the Host truncation primitives. It must reproduce the same
/// composition on Host in the kept bond space and in the kept spectrum, and
/// reconstruct `u s vh` to dtype tolerance. The device factors keep the raw
/// cuSOLVER gauge, so `u`/`vh` payloads are never compared bit for bit.
fn assert_typed_cuda_svd_trunc_composition_matches_host<R>(
    source: &TensorMap<R, f64>,
    truncation: &Truncation,
) where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
{
    let provider = source.provider() as *const R;
    let runtime = symtn::typed::__network::runtime_identity(source.runtime());
    let source_bits: Vec<_> = source
        .dense_data()
        .unwrap()
        .iter()
        .map(|value| value.to_bits())
        .collect();
    let expected = host_svd_trunc(source, truncation, |value: f64| value);
    let source_device = source.to_cuda().unwrap();

    let Svd {
        u: u_device,
        s: s_device,
        vh: vh_device,
    } = source_device
        .svd_compact(&codomain_axes(&source_device), &domain_axes(&source_device))
        .unwrap();
    for factor in [&u_device, &s_device, &vh_device] {
        assert!(std::ptr::eq(factor.provider(), provider));
        assert!(runtime.matches(factor.runtime()));
        assert_eq!(factor.placement(), symtn::expert::Placement::Cuda(0));
    }
    let u = u_device.to_host().unwrap();
    let s = s_device.to_host().unwrap();
    let vh = vh_device.to_host().unwrap();

    let found = s.domain()[0]
        .find_truncated(&s.diagview().unwrap(), truncation)
        .unwrap();
    let selection = &found.selection;
    let u = u.restrict_leg(&[(u.codomain_rank(), selection)]).unwrap();
    let s = s.restrict_leg(&[(0, selection), (1, selection)]).unwrap();
    let vh = vh.restrict_leg(&[(0, selection)]).unwrap();

    // The kept bond space is exactly Host's, including the empty bond of a
    // discard-all policy.
    assert_eq!(*selection.subspace(), expected.s.domain()[0]);
    for (actual, expected) in [(&u, &expected.u), (&s, &expected.s), (&vh, &expected.vh)] {
        assert_eq!(structural_snapshot(actual), structural_snapshot(expected));
    }
    // Per sector and matched by provider label: `diagview` orders by encoded
    // `SectorId`, the Host oracle by decoded label, and those two orders are
    // not the same contract.
    let mut kept = s.diagview().unwrap();
    kept.sort_by(|left, right| left.sector.cmp(&right.sector));
    assert_eq!(kept.len(), expected.singular_values.len());
    for (actual, expected) in kept.iter().zip(&expected.singular_values) {
        assert_eq!(actual.sector, expected.sector);
        numerics::assert_slices_close(
            "actual.values",
            &actual.values,
            &expected.values,
            FACTOR_TERMS,
        );
    }
    assert!((found.error - expected.error).abs() <= 1e-10 * (1.0 + expected.error));

    assert!(is_isometric!(u, 1e-10));
    assert!(is_isometric!(vh.adjoint().unwrap(), 1e-10));
    let actual_rebuilt = u.compose(&s).unwrap().compose(&vh).unwrap();
    let expected_rebuilt = expected
        .u
        .compose(&expected.s)
        .unwrap()
        .compose(&expected.vh)
        .unwrap();
    numerics::assert_slices_close(
        "actual_rebuilt",
        actual_rebuilt.dense_data().unwrap(),
        expected_rebuilt.dense_data().unwrap(),
        FACTOR_TERMS,
    );
    assert_eq!(
        source_device
            .to_host()
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        source_bits
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_svd_compact_streams_dense_multiplicity_free_f64_factors() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let u1 = Arc::new(U1FusionRule);
    let tall = GradedSpace::try_new(
        Arc::clone(&u1),
        [(U1Irrep::new(0), 3), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let wide = GradedSpace::try_new(
        Arc::clone(&u1),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 3)],
    )
    .unwrap();
    let rectangular = TensorMap::from_subblock_fn(&runtime, [&tall], [&wide], |_, indices| {
        if indices[0] == indices[1] {
            6.0 + indices[0] as f64
        } else {
            (1 + indices[0] + 2 * indices[1]) as f64
        }
    })
    .unwrap();
    assert_typed_cuda_svd_matches_host(&rectangular);

    let rank_deficient = TensorMap::from_subblock_fn(&runtime, [&tall], [&tall], |_, indices| {
        (indices[0] + 1) as f64 * (indices[1] + 1) as f64
    })
    .unwrap();
    assert_typed_cuda_svd_matches_host(&rank_deficient);

    let su2 = Arc::new(SU2FusionRule);
    let su2_leg = GradedSpace::try_new(
        Arc::clone(&su2),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap();
    let multi_tree =
        TensorMap::from_subblock_fn(&runtime, [&su2_leg, &su2_leg], [&su2_leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();
    let multi_tree_snapshot = structural_snapshot(&multi_tree);
    assert!(multi_tree_snapshot
        .blocks
        .iter()
        .enumerate()
        .any(|(index, left)| {
            multi_tree_snapshot.blocks[index + 1..].iter().any(|right| {
                left.fusion_trees.coupled() == right.fusion_trees.coupled()
                    && left.fusion_trees != right.fusion_trees
            })
        }));
    assert_typed_cuda_svd_matches_host(&multi_tree);

    let all_zero: TensorMap<_, f64> = TensorMap::zeros(&runtime, [&wide], [&wide]).unwrap();
    assert!(all_zero
        .dense_data()
        .unwrap()
        .iter()
        .all(|value| *value == 0.0));
    assert_typed_cuda_svd_matches_host(&all_zero);

    let fermion = Arc::new(FermionParityFusionRule);
    let fermion_leg = GradedSpace::try_new(
        Arc::clone(&fermion),
        [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 1)],
    )
    .unwrap();
    let fermion_tensor =
        TensorMap::from_subblock_fn(&runtime, [&fermion_leg], [&fermion_leg], |_, indices| {
            (1 + indices[0] + 3 * indices[1]) as f64
        })
        .unwrap();
    assert_typed_cuda_svd_matches_host(&fermion_tensor);

    let product = Arc::new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(U1FusionRule, FermionParityFusionRule)
            .unwrap(),
    );
    let product_leg = GradedSpace::try_new(
        Arc::clone(&product),
        [
            (product_sector(U1Irrep::new(0), Z2Irrep::EVEN), 2),
            (product_sector(U1Irrep::new(1), Z2Irrep::ODD), 1),
        ],
    )
    .unwrap();
    let product_tensor =
        TensorMap::from_subblock_fn(&runtime, [&product_leg], [&product_leg], |_, indices| {
            (2 + indices[0] + indices[1]) as f64
        })
        .unwrap();
    assert_typed_cuda_svd_matches_host(&product_tensor);

    let zn3 = Arc::new(ZNFusionRule::new(3).unwrap());
    let charge0 = GradedSpace::try_new(Arc::clone(&zn3), [(zn3.irrep(0), 2)]).unwrap();
    let charge1 = GradedSpace::try_new(Arc::clone(&zn3), [(zn3.irrep(1), 3)]).unwrap();
    let empty: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&charge0], [&charge1], |_, _| 1.0).unwrap();
    assert_typed_cuda_svd_matches_host(&empty);

    let device = rectangular.to_cuda().unwrap();
    let source_bits: Vec<_> = device
        .to_host()
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .map(|value| value.to_bits())
        .collect();
    assert!(matches!(
        device.adjoint().unwrap().svd_compact(&[0], &[1]),
        Err(symtn::typed::Error::UnsupportedOnDevice(_))
    ));
    let expected = rectangular
        .svd_compact(&codomain_axes(&rectangular), &domain_axes(&rectangular))
        .unwrap();
    for _ in 0..3 {
        assert_cuda_svd_result(
            &rectangular,
            &expected,
            device
                .svd_compact(&codomain_axes(&device), &domain_axes(&device))
                .unwrap(),
        );
    }
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..2)
            .map(|_| {
                scope.spawn(|| {
                    device
                        .svd_compact(&codomain_axes(&device), &domain_axes(&device))
                        .unwrap()
                })
            })
            .collect();
        for worker in workers {
            assert_cuda_svd_result(&rectangular, &expected, worker.join().unwrap());
        }
    });
    assert_eq!(
        device
            .to_host()
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        source_bits
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_svd_compact_handles_large_wide_sector() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let rows = GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0), 8)]).unwrap();
    let cols = GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0), 1025)]).unwrap();
    let source = TensorMap::from_subblock_fn(&runtime, [&rows], [&cols], |_, indices| {
        let row = indices[0];
        let col = indices[1];
        ((row * 17 + col * 13 + 3) % 31) as f64 / 31.0 - 0.5 + if row == col { 2.0 } else { 0.0 }
    })
    .unwrap();

    assert_typed_cuda_svd_matches_host(&source);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_svd_trunc_composition_handles_large_wide_sector() {
    // What: the 8 x 1025 cuSOLVER path composes with a kept prefix on the Host.
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let rows = GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0), 8)]).unwrap();
    let cols = GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0), 1025)]).unwrap();
    let source = TensorMap::from_subblock_fn(&runtime, [&rows], [&cols], |_, indices| {
        let row = indices[0];
        let col = indices[1];
        ((row * 17 + col * 13 + 3) % 31) as f64 / 31.0 - 0.5 + if row == col { 2.0 } else { 0.0 }
    })
    .unwrap();

    assert_typed_cuda_svd_trunc_composition_matches_host(&source, &Truncation::rank(4));
}

#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_svd_trunc_composition_matches_host_policies_structure_and_ownership() {
    // What: all truncation policies and supported provider families match the
    // Host semantic oracle without comparing backend-dependent U/Vh gauges.
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let rows = GradedSpace::try_new(
        Arc::clone(&provider),
        [(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    let cols = GradedSpace::try_new(
        Arc::clone(&provider),
        [(U1Irrep::new(0), 2), (U1Irrep::new(2), 1)],
    )
    .unwrap();
    let mixed = TensorMap::from_subblock_fn(&runtime, [&rows], [&cols], |_, indices| {
        if indices[0] == indices[1] {
            5.0 + indices[0] as f64
        } else {
            (1 + indices[0] + 2 * indices[1]) as f64
        }
    })
    .unwrap();
    let same_identity_space = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 1), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let policies = [
        Truncation::Full,
        Truncation::rank(0),
        Truncation::rank(2),
        Truncation::rank(usize::MAX),
        Truncation::absolute_cutoff(0.25).unwrap(),
        Truncation::relative_inf_cutoff(0.2).unwrap(),
        Truncation::relative_error(0.1).unwrap(),
        Truncation::space(same_identity_space.truncspace()),
        Truncation::rank(3).and(Truncation::absolute_cutoff(0.1).unwrap()),
    ];
    for policy in &policies {
        assert_typed_cuda_svd_trunc_composition_matches_host(&mixed, policy);
    }

    let dimension_calls = Arc::new(AtomicUsize::new(0));
    let reentrant = Arc::new(ReentrantDimensionRule {
        runtime: runtime.clone(),
        calls: Arc::clone(&dimension_calls),
    });
    let reentrant_leg = GradedSpace::try_new(Arc::clone(&reentrant), [(ProbeSector, 2)]).unwrap();
    let reentrant_source = TensorMap::from_subblock_fn(
        &runtime,
        [&reentrant_leg],
        [&reentrant_leg],
        |_, indices| (1 + indices[0] + indices[1]) as f64,
    )
    .unwrap();
    assert_typed_cuda_svd_trunc_composition_matches_host(&reentrant_source, &Truncation::rank(1));
    assert!(dimension_calls.load(Ordering::SeqCst) > 0);

    // `[3, 2] <- [3, 2]` factors put a 2x2 block at element offset 9, an
    // unaligned destination for the device factor copy (#1320); these two fixtures cover that layout through the composition.
    let rank_deficient = TensorMap::from_subblock_fn(&runtime, [&rows], [&rows], |_, indices| {
        (indices[0] + 1) as f64 * (indices[1] + 1) as f64
    })
    .unwrap();
    assert_typed_cuda_svd_trunc_composition_matches_host(&rank_deficient, &Truncation::rank(1));
    let all_zero: TensorMap<_, f64> = TensorMap::zeros(&runtime, [&rows], [&rows]).unwrap();
    assert_typed_cuda_svd_trunc_composition_matches_host(
        &all_zero,
        &Truncation::relative_error(0.0).unwrap(),
    );

    let no_intersection_rows =
        GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(4), 2)]).unwrap();
    let no_intersection: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&no_intersection_rows], [&cols], |_, _| 1.0)
            .unwrap();
    assert!(no_intersection.dense_data().unwrap().is_empty());
    assert_typed_cuda_svd_trunc_composition_matches_host(&no_intersection, &Truncation::Full);

    let su2 = Arc::new(SU2FusionRule);
    let su2_leg = GradedSpace::try_new(
        Arc::clone(&su2),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap();
    let multi_tree =
        TensorMap::from_subblock_fn(&runtime, [&su2_leg, &su2_leg], [&su2_leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();
    assert_typed_cuda_svd_trunc_composition_matches_host(&multi_tree, &Truncation::rank(3));

    let fermion = Arc::new(FermionParityFusionRule);
    let fermion_leg = GradedSpace::try_new(
        Arc::clone(&fermion),
        [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 1)],
    )
    .unwrap();
    let fermion_tensor =
        TensorMap::from_subblock_fn(&runtime, [&fermion_leg], [&fermion_leg], |_, indices| {
            (1 + indices[0] + 3 * indices[1]) as f64
        })
        .unwrap();
    assert_typed_cuda_svd_trunc_composition_matches_host(&fermion_tensor, &Truncation::rank(2));

    let product = Arc::new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(U1FusionRule, FermionParityFusionRule)
            .unwrap(),
    );
    let product_leg = GradedSpace::try_new(
        Arc::clone(&product),
        [
            (product_sector(U1Irrep::new(0), Z2Irrep::EVEN), 2),
            (product_sector(U1Irrep::new(1), Z2Irrep::ODD), 1),
        ],
    )
    .unwrap();
    let product_tensor =
        TensorMap::from_subblock_fn(&runtime, [&product_leg], [&product_leg], |_, indices| {
            (2 + indices[0] + indices[1]) as f64
        })
        .unwrap();
    assert_typed_cuda_svd_trunc_composition_matches_host(&product_tensor, &Truncation::rank(2));

    // Repetition and concurrency bear on the device half of the
    // composition, which is `svd_compact`.
    let device = mixed.to_cuda().unwrap();
    let expected = device
        .svd_compact(&codomain_axes(&device), &domain_axes(&device))
        .unwrap()
        .s
        .to_host()
        .unwrap();
    let expected_spectrum = expected.diagview().unwrap();
    for _ in 0..2 {
        let actual = device
            .svd_compact(&codomain_axes(&device), &domain_axes(&device))
            .unwrap()
            .s
            .to_host()
            .unwrap();
        assert_eq!(actual.diagview().unwrap(), expected_spectrum);
    }
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..2)
            .map(|_| {
                scope.spawn(|| {
                    device
                        .svd_compact(&codomain_axes(&device), &domain_axes(&device))
                        .unwrap()
                        .s
                        .to_host()
                        .unwrap()
                })
            })
            .collect();
        for worker in workers {
            assert_eq!(
                worker.join().unwrap().diagview().unwrap(),
                expected_spectrum
            );
        }
    });
}

fn assert_c64_svd_trunc_composition_matches_host<R>(
    source: &TensorMap<R, Complex64>,
    truncation: &Truncation,
) where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
{
    let source_data = source.dense_data().unwrap().to_vec();
    let expected = host_svd_trunc(source, truncation, |value: Complex64| value.re);
    let device = source.to_cuda().unwrap();

    let Svd {
        u: u_device,
        s: s_device,
        vh: vh_device,
    } = device
        .svd_compact(&codomain_axes(&device), &domain_axes(&device))
        .unwrap();
    assert_device_factor_handles(source, [&u_device, &s_device, &vh_device]);
    let u = u_device.to_host().unwrap();
    let s = s_device.to_host().unwrap();
    let vh = vh_device.to_host().unwrap();

    let found = s.domain()[0]
        .find_truncated(&s.diagview().unwrap(), truncation)
        .unwrap();
    let selection = &found.selection;
    let u = u.restrict_leg(&[(u.codomain_rank(), selection)]).unwrap();
    let s = s.restrict_leg(&[(0, selection), (1, selection)]).unwrap();
    let vh = vh.restrict_leg(&[(0, selection)]).unwrap();

    assert_eq!(*selection.subspace(), expected.s.domain()[0]);
    for (actual, expected) in [(&u, &expected.u), (&s, &expected.s), (&vh, &expected.vh)] {
        assert_eq!(structural_snapshot(actual), structural_snapshot(expected));
    }
    let mut kept = s.diagview().unwrap();
    kept.sort_by(|left, right| left.sector.cmp(&right.sector));
    assert_eq!(kept.len(), expected.singular_values.len());
    for (actual, expected) in kept.iter().zip(&expected.singular_values) {
        assert_eq!(actual.sector, expected.sector);
        let values: Vec<f64> = actual.values.iter().map(|value| value.re).collect();
        numerics::assert_slices_close("values", &values, &expected.values, FACTOR_TERMS);
    }
    assert!((found.error - expected.error).abs() <= 1e-9 * (1.0 + expected.error));

    assert!(is_isometric!(u, 1e-10), "U^H U = I");
    assert!(is_isometric!(vh.adjoint().unwrap(), 1e-10), "V^H V = I");
    let actual_rebuilt = u.compose(&s).unwrap().compose(&vh).unwrap();
    let expected_rebuilt = expected
        .u
        .compose(&expected.s)
        .unwrap()
        .compose(&expected.vh)
        .unwrap();
    numerics::assert_slices_close(
        "actual_rebuilt",
        actual_rebuilt.dense_data().unwrap(),
        expected_rebuilt.dense_data().unwrap(),
        FACTOR_TERMS,
    );
    assert_eq!(device.to_host().unwrap().dense_data().unwrap(), source_data);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_c64_svd_matches_host_spectra_and_truncation_policies() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let u1 = Arc::new(U1FusionRule);
    let rows = GradedSpace::try_new(
        Arc::clone(&u1),
        [(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    let cols = GradedSpace::try_new(
        Arc::clone(&u1),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 3)],
    )
    .unwrap();
    let rectangular = TensorMap::<_, Complex64>::from_subblock_fn(
        &runtime,
        [&rows],
        [&cols],
        distinct_c64_fill(),
    )
    .unwrap();
    assert_c64_svd_matches_host(&rectangular);

    let su2 = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap();
    let su2_tensor = TensorMap::<_, Complex64>::from_subblock_fn(
        &runtime,
        [&su2, &su2, &su2],
        [&su2, &su2],
        distinct_c64_fill(),
    )
    .unwrap();
    assert_c64_svd_matches_host(&su2_tensor);

    let unmatched_space = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 1), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let policies = [
        // Full, discard-all, and mixed keep/drop decisions.
        Truncation::Full,
        Truncation::rank(0),
        Truncation::rank(2),
        Truncation::rank(usize::MAX),
        Truncation::absolute_cutoff(0.25).unwrap(),
        Truncation::relative_inf_cutoff(0.2).unwrap(),
        Truncation::relative_error(0.1).unwrap(),
        Truncation::space(unmatched_space.truncspace()),
        Truncation::rank(3).and(Truncation::absolute_cutoff(0.1).unwrap()),
    ];
    for policy in &policies {
        assert_c64_svd_trunc_composition_matches_host(&rectangular, policy);
    }
    assert_c64_svd_trunc_composition_matches_host(&su2_tensor, &Truncation::rank(2));
}

#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_su2_svd_trunc_matches_host_at_every_weighted_rank_budget() {
    // What: SU(2) spectra with nontrivial degeneracies and quantum dimensions
    // 1, 2 and 3 take the Host decision at every integer rank budget, most of
    // which land exactly on a cumulative `dim(c)` sum (#1871), in f64 and c64.
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 2),
            (SU2Irrep::from_twice_spin(2), 1),
        ],
    )
    .unwrap();
    // Hashed fills keep singular values apart, so device rounding cannot
    // reorder a near-tie that the budget then splits.
    let mut ordinal = 0usize;
    let real = TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg], |_, _| {
        ordinal += 1;
        pseudo_random_c64(ordinal).re
    })
    .unwrap();
    let complex = TensorMap::<_, Complex64>::from_subblock_fn(
        &runtime,
        [&leg, &leg],
        [&leg],
        distinct_c64_fill(),
    )
    .unwrap();
    for budget in 0..=12 {
        assert_typed_cuda_svd_trunc_composition_matches_host(&real, &Truncation::rank(budget));
        assert_c64_svd_trunc_composition_matches_host(&complex, &Truncation::rank(budget));
    }
}
