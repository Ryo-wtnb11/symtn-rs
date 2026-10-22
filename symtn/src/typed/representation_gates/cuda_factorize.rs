use super::*;
#[cfg(feature = "cuda")]
use crate::typed::probe::{Fault, Gauge, RoutePolicy};

/// `(qr calls, factor copies, selector uploads, output uploads, assembly
/// GEMMs, live route scratch, peak route scratch)` in `recording`.
#[cfg(feature = "cuda")]
fn qr_counts(recording: &probe::Recording) -> (usize, usize, usize, usize, usize, usize, usize) {
    let (live, peak) = recording.gauge(Gauge::CudaQrScratch);
    (
        recording.get(Event::CudaQrDecomposition),
        recording.get(Event::CudaFactorCopy),
        recording.get(Event::CudaSelectorUpload),
        recording.get(Event::CudaQrOutputUpload),
        recording.get(Event::CudaAssemblyGemm),
        live,
        peak,
    )
}

/// `(successful results, spectrum scalars, output uploads, live route
/// scratch, peak route scratch)` of device SVD in `recording`.
#[cfg(feature = "cuda")]
fn svd_counts(recording: &probe::Recording) -> (usize, usize, usize, usize, usize) {
    let (live, peak) = recording.gauge(Gauge::CudaSvdScratch);
    (
        recording.get(Event::CudaSvdDecomposition),
        recording.get(Event::CudaSvdSpectrumScalar),
        recording.get(Event::CudaSvdOutputUpload),
        live,
        peak,
    )
}

#[cfg(feature = "cuda")]
fn svd_routes(treewise: bool) -> RoutePolicy {
    RoutePolicy {
        svd_treewise: treewise,
        ..RoutePolicy::DEFAULT
    }
}

/// `(factor_copies, selector_uploads, assembly_gemms)` a compact QR or
/// compact SVD assembly must perform for this plan: an aligned side is one
/// whole-factor copy and no selector upload, a non-aligned side is one GEMM
/// per nonempty target tree.
#[cfg(feature = "cuda")]
fn cuda_route_assembly_counts(plan: &CompactFactorPlan) -> (usize, usize, usize) {
    let nonempty_trees = |trees: &[CoupledTreeExtent]| {
        trees
            .iter()
            .filter(|tree| tree.extent().is_ok_and(|extent| extent != 0))
            .count()
    };
    let mut counts = (0, 0, 0);
    for (route, left, right) in executed_routes(plan) {
        let aligned_left = plan.left_preserves_trees(route);
        let aligned_right = plan.right_preserves_trees(route);
        if aligned_left {
            counts.0 += 1;
        } else {
            counts.2 += nonempty_trees(plan.left_regions()[left].row_trees());
        }
        if aligned_right {
            counts.0 += 1;
        } else {
            counts.2 += nonempty_trees(plan.right_regions()[right].col_trees());
        }
        if !(aligned_left && aligned_right) {
            counts.1 += 1;
        }
    }
    counts
}

/// The factor spaces and routes every device factorization publishes through
/// are the Host plan's, checked without a device.
///
/// A device-typed tensor (a compact diagonal needs no device allocation)
/// builds its plan through the same `compact_factor_plan` the device QR, SVD
/// and EIGH call, and a multi-tree source with a dual leg and one-sided
/// sectors goes through `compact_factor_routes`, `executed_routes` and the SVD
/// diagonal validation. Every space must equal the Host eager factorization's
/// output space, so a device-local bond, space or route rule fails here.
#[cfg(feature = "cuda")]
#[test]
fn typed_cuda_factor_plan_publishes_the_host_factor_spaces_without_a_device() {
    fn assert_host_spaces<R>(source: &TensorMap<R, f64>, plan: &CompactFactorPlan)
    where
        R: symtn_core::TypedSectorAdmission<
                Sector = <R as symtn_core::SectorCodec>::Sector,
                Error = symtn_core::FusionAlgebraError,
                Mode = symtn_core::MultiplicityFreeAdmissionMode,
            > + MultiplicityFreeRigidSymbols<Scalar = f64>
            + CheckedFusionAlgebra
            + SectorCodec,
    {
        let space = source.logical_space();
        let (rows, cols) = (codomain_axes(source), domain_axes(source));
        let Qr { q, r } = source.qr_compact(&rows, &cols).unwrap();
        let Svd { u, s, vh } = source.svd_compact(&rows, &cols).unwrap();
        let left = plan.left_space(space).unwrap();
        let right = plan.right_space(space).unwrap();
        let bond = plan.bond_space(space).unwrap();
        for (host, device, what) in [
            (&q, &left, "qr q"),
            (&r, &right, "qr r"),
            (&u, &left, "svd u"),
            (&s, &bond, "svd s"),
            (&vh, &right, "svd vh"),
        ] {
            assert_eq!(host.logical_space().space(), device.space(), "{what}");
        }
        let bond_regions = sector_regions(bond.space().structure(), bond.space().nout()).unwrap();
        let diagonals = validate_cuda_svd_middle_regions(plan, &bond_regions).unwrap();
        let routes: Vec<_> = executed_routes(plan).collect();
        assert_eq!(diagonals.len(), routes.len());
        assert_eq!(routes.len(), bond_regions.len());
        for ((route, left_region, right_region), diagonal) in routes.iter().zip(&diagonals) {
            let source_region = &plan.source_regions()[route.source_region()];
            assert_eq!(route.rank(), source_region.rows().min(source_region.cols()));
            assert_eq!(plan.left_regions()[*left_region].coupled(), route.sector());
            assert_eq!(
                plan.right_regions()[*right_region].coupled(),
                route.sector()
            );
            let middle = bond_regions
                .iter()
                .find(|region| region.coupled() == route.sector())
                .unwrap();
            assert_eq!(*diagonal, middle.range().start);
        }
        if space.space().homspace().codomain() == space.space().homspace().domain() {
            let Eigh { d, v } = source
                .eigh_full(&rows, &cols, HermitianTol::DEFAULT)
                .unwrap();
            assert_eq!(v.logical_space().space(), left.space(), "eigh v");
            assert_eq!(d.logical_space().space(), bond.space(), "eigh d");
        }
    }

    // Multi-tree sectors, a dual codomain leg, sectors on one side only.
    let source = u1_lazy_fixture();
    let plan = compact_factor_routes(source.logical_space()).unwrap();
    assert!(executed_routes(&plan).count() > 1);
    assert!(plan
        .source_regions()
        .iter()
        .any(|region| region.row_trees().len() > 1));
    assert_host_spaces(&source, &plan);

    // The device method itself, on a device-typed compact diagonal (a valid
    // device tensor that needs no allocation) over a multi-sector bond; its
    // dense Host twin has the same space.
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 3),
            (U1Irrep::new(2), 1),
        ],
    )
    .unwrap();
    let dense =
        TensorMap::<U1FusionRule, f64>::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
            if index[0] == index[1] {
                2.0 + index[0] as f64
            } else {
                0.25
            }
        })
        .unwrap();
    let spectrum = dense
        .svd_compact(&codomain_axes(&dense), &domain_axes(&dense))
        .unwrap()
        .s;
    let TypedData::Diagonal(values) = owned(&spectrum).data.as_ref() else {
        unreachable!("SVD factor is compact")
    };
    let device: TensorMap<_, f64, CudaStorage> = TensorMap {
        runtime: runtime.clone(),
        repr: owned_repr(TypedTensorBody::new(
            spectrum.logical_space().clone(),
            TypedData::<f64, CudaStorage>::Diagonal(values.clone()),
        )),
    };
    let host = spectrum.materialize().unwrap();
    assert_eq!(host.logical_space().space(), device.logical_space().space());
    assert_host_spaces(&host, &device.compact_factor_plan().unwrap());
}

#[cfg(feature = "cuda")]
#[test]
fn typed_cuda_factorizations_reject_compact_lazy_and_truncation_before_runtime_work() {
    let diagonal = u1_lazy_fixture().svd_compact(&[0, 1], &[2]).unwrap().s;
    let TypedData::Diagonal(spectrum) = owned(&diagonal).data.as_ref() else {
        unreachable!("SVD factor is compact")
    };
    let device_diagonal: TensorMap<_, f64, CudaStorage> = TensorMap {
        runtime: diagonal.runtime.clone(),
        repr: owned_repr(TypedTensorBody::new(
            diagonal.logical_space().clone(),
            TypedData::<f64, CudaStorage>::Diagonal(spectrum.clone()),
        )),
    };
    assert!(matches!(
        device_diagonal.qr_compact(&codomain_axes(&device_diagonal), &domain_axes(&device_diagonal)),
        Err(Error::UnsupportedOnDevice(message)) if message.contains("dense CUDA storage")
    ));
    assert!(matches!(
        device_diagonal.svd_compact(&codomain_axes(&device_diagonal), &domain_axes(&device_diagonal)),
        Err(Error::UnsupportedOnDevice(message)) if message.contains("dense CUDA storage")
    ));
    assert!(matches!(
        device_diagonal.eigh_full(&codomain_axes(&device_diagonal), &domain_axes(&device_diagonal), HermitianTol::DEFAULT),
        Err(Error::UnsupportedOnDevice(message)) if message.contains("dense CUDA storage")
    ));
    // #1452: the adjoint of a compact diagonal is the owned (conjugated)
    // diagonal on every storage, never a lazy view, so it is rejected as
    // compact storage. The lazy-operand rejection needs a dense device
    // parent and is covered by
    // `typed_cuda_factorizations_reject_lazy_adjoint_before_runtime_work`.
    let adjoint = device_diagonal.adjoint().unwrap();
    assert!(Arc::ptr_eq(owned(&adjoint), owned(&device_diagonal)));
    assert!(matches!(
        adjoint.qr_compact(&codomain_axes(&adjoint), &domain_axes(&adjoint)),
        Err(Error::UnsupportedOnDevice(message)) if message.contains("dense CUDA storage")
    ));
    assert!(matches!(
        adjoint.svd_compact(&codomain_axes(&adjoint), &domain_axes(&adjoint)),
        Err(Error::UnsupportedOnDevice(message)) if message.contains("dense CUDA storage")
    ));
    assert!(matches!(
        adjoint.eigh_full(&codomain_axes(&adjoint), &domain_axes(&adjoint), HermitianTol::DEFAULT),
        Err(Error::UnsupportedOnDevice(message)) if message.contains("dense CUDA storage")
    ));

    let complex_spectrum: Vec<_> = spectrum
        .iter()
        .map(|entry| symtn_matrixalgebra::SectorSpectrum {
            sector: entry.sector,
            values: entry
                .values
                .iter()
                .map(|&value| num_complex::Complex64::new(value, value + 1.0))
                .collect(),
        })
        .collect();
    let complex_diagonal: TensorMap<_, num_complex::Complex64, CudaStorage<_>> = TensorMap {
        runtime: diagonal.runtime.clone(),
        repr: owned_repr(TypedTensorBody::diagonal(
            diagonal.logical_space().clone(),
            complex_spectrum.clone(),
        )),
    };
    let complex_adjoint = complex_diagonal.adjoint().unwrap();
    let TypedData::Diagonal(conjugated) = owned(&complex_adjoint).data.as_ref() else {
        unreachable!("the adjoint of a compact diagonal is compact")
    };
    for (actual, source) in conjugated.iter().zip(&complex_spectrum) {
        assert_eq!(actual.sector, source.sector);
        let expected: Vec<_> = source.values.iter().map(|value| value.conj()).collect();
        assert_eq!(actual.values, expected);
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_factorizations_reject_lazy_adjoint_before_runtime_work() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();
    let lazy = source.to_cuda().unwrap().adjoint().unwrap();
    assert!(matches!(&lazy.repr, TypedTensorRepr::Adjoint(_)));
    assert!(matches!(
        lazy.qr_compact(&codomain_axes(&lazy), &domain_axes(&lazy)),
        Err(Error::UnsupportedOnDevice(message)) if message.contains("lazy adjoint")
    ));
    assert!(matches!(
        lazy.svd_compact(&codomain_axes(&lazy), &domain_axes(&lazy)),
        Err(Error::UnsupportedOnDevice(message)) if message.contains("lazy adjoint")
    ));
    assert!(matches!(
        lazy.eigh_full(&codomain_axes(&lazy), &domain_axes(&lazy), HermitianTol::DEFAULT),
        Err(Error::UnsupportedOnDevice(message)) if message.contains("lazy adjoint")
    ));
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_eigh_full_matches_host_without_hidden_materialization() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [(U1Irrep::new(-1), 2), (U1Irrep::new(0), 3)],
    )
    .unwrap();
    let source = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        let (row, col) = (indices[0], indices[1]);
        if row == col {
            row as f64 + 1.0
        } else if row.abs_diff(col) == 1 {
            0.125
        } else {
            0.0
        }
    })
    .unwrap();
    let device = source.to_cuda().unwrap();

    let expected_full = source
        .eigh_full(
            &codomain_axes(&source),
            &domain_axes(&source),
            HermitianTol::DEFAULT,
        )
        .unwrap();
    let Eigh {
        d: d_device,
        v: v_device,
    } = device
        .eigh_full(
            &codomain_axes(&device),
            &domain_axes(&device),
            HermitianTol::DEFAULT,
        )
        .unwrap();
    assert_eq!(d_device.placement(), Placement::Cuda(0));
    assert_eq!(v_device.placement(), Placement::Cuda(0));
    assert!(Arc::ptr_eq(
        v_device.logical_space().provider_arc(),
        source.logical_space().provider_arc()
    ));
    let d = d_device.to_host().unwrap();
    let v = v_device.to_host().unwrap();
    assert_typed_map_close(&d, &expected_full.d, 1.0e-10);
    assert_typed_map_close(
        &source.compose(&v).unwrap(),
        &v.compose(&d).unwrap(),
        1.0e-10,
    );

    let su2_provider = Arc::new(SU2FusionRule);
    let su2_leg = GradedSpace::try_new(
        Arc::clone(&su2_provider),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap();
    let su2_source = TensorMap::from_subblock_fn(&runtime, [&su2_leg], [&su2_leg], |_, indices| {
        if indices[0] == indices[1] {
            indices[0] as f64 + 1.0
        } else {
            0.25
        }
    })
    .unwrap();
    assert!(su2_source.subblock_count() >= 2);
    let su2_device = su2_source.to_cuda().unwrap();
    let Eigh { d: su2_d, v: su2_v } = su2_device
        .eigh_full(
            &codomain_axes(&su2_device),
            &domain_axes(&su2_device),
            HermitianTol::DEFAULT,
        )
        .unwrap();
    assert!(Arc::ptr_eq(
        su2_v.logical_space().provider_arc(),
        su2_source.logical_space().provider_arc()
    ));
    let su2_d = su2_d.to_host().unwrap();
    let su2_v = su2_v.to_host().unwrap();
    assert_typed_map_close(
        &su2_source.compose(&su2_v).unwrap(),
        &su2_v.compose(&su2_d).unwrap(),
        1.0e-10,
    );

    let input_before_failure = device.to_host().unwrap();
    for failure in [("decomposition", 2), ("assembly", 2)] {
        probe::arm(Fault::CudaEigh(failure.0), failure.1);
        assert!(device
            .eigh_full(
                &codomain_axes(&device),
                &domain_axes(&device),
                HermitianTol::DEFAULT
            )
            .is_err());
        probe::disarm();
        assert_typed_map_close(
            &device.to_host().unwrap(),
            &input_before_failure,
            f64::EPSILON,
        );
    }

    let nonhermitian = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        match (indices[0], indices[1]) {
            (0, 1) => 1.0,
            _ => 0.0,
        }
    })
    .unwrap()
    .to_cuda()
    .unwrap();
    assert!(matches!(
        nonhermitian.eigh_full(&codomain_axes(&nonhermitian), &domain_axes(&nonhermitian), HermitianTol::DEFAULT),
        Err(Error::Operation(error))
            if matches!(
                error.as_ref(),
                symtn_tensors::OperationError::InvalidArgument { .. }
            )
    ));
}

/// Z2 endomorphism whose canonical sector blocks are `[[2,1],[1,2]]`
/// (spectrum {3, 1}), stored with rows stacked by ascending codomain tree
/// and columns by descending domain tree, so each block reads
/// `[[1,2],[2,1]]`: still Hermitian, but with spectrum {3, -1}.
fn mis_stacked_hermitian_z2(runtime: &Runtime) -> TensorMap<Z2FusionRule, f64> {
    let rule = Z2FusionRule;
    let leg = || {
        SectorLeg::new(
            [
                (Z2Irrep::new(0).sector_id(), 1),
                (Z2Irrep::new(1).sector_id(), 1),
            ],
            false,
        )
    };
    let homspace = FusionTreeHomSpace::new(
        FusionProductSpace::new([leg(), leg()]),
        FusionProductSpace::new([leg(), leg()]),
    );
    let mut blocks: Vec<(FusionTreePairKey, Vec<usize>)> = homspace
        .fusion_tree_keys(&rule)
        .iter()
        .map(|key| (key.clone(), vec![1; 4]))
        .collect();
    blocks.sort_by(|(a, _), (b, _)| {
        a.codomain_tree()
            .cmp(b.codomain_tree())
            .then(b.domain_tree().cmp(a.domain_tree()))
    });
    let structure = BlockStructure::coupled_sector_matrix_with_keys(&rule, 2, 4, blocks).unwrap();
    let regions = structure.coupled_sector_regions(2).unwrap().unwrap();
    assert!(regions
        .iter()
        .all(|region| region.row_trees() != region.col_trees()));
    let space = DynamicFusionMapSpace::try_from_layout(homspace, structure)
        .unwrap()
        .try_bind_rule(&rule)
        .unwrap();
    let space =
        BoundDynamicFusionMapSpace::bind_multiplicity_free(space.clone(), Arc::new(rule)).unwrap();
    let mut fill = |key: &BlockKey, _: &[usize]| -> f64 {
        let BlockKey::FusionTree(tree) = key else {
            unreachable!("fusion-tree blocks")
        };
        if tree.codomain_tree() == tree.domain_tree() {
            2.0
        } else {
            1.0
        }
    };
    let data = crate::typed::block_layout::apply_fill(
        space.space(),
        crate::typed::block_layout::Fill::BlockFn(&mut fill),
    )
    .unwrap();
    TensorMap {
        runtime: runtime.clone(),
        repr: owned_repr(TypedTensorBody {
            space,
            data: Arc::new(TypedData::Dense(data)),
        }),
    }
}

#[test]
fn host_eigh_refuses_a_mis_stacked_block_that_stays_hermitian() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let tensor = mis_stacked_hermitian_z2(&runtime);
    let error = format!(
        "{:?}",
        tensor
            .eigh_full(&[0, 1], &[2, 3], HermitianTol::DEFAULT)
            .err()
    );
    assert!(error.contains("eigh_full requires identical"), "{error}");
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_eigh_refuses_a_mis_stacked_block_that_stays_hermitian() {
    // What: the device path reads the same tiling as the host and must
    // refuse it too, instead of returning the {3, -1} spectrum of the
    // column-permuted block.
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let device = mis_stacked_hermitian_z2(&runtime).to_cuda().unwrap();
    assert!(matches!(
        device.eigh_full(&codomain_axes(&device), &domain_axes(&device), HermitianTol::DEFAULT),
        Err(Error::Operation(error))
            if matches!(
                error.as_ref(),
                symtn_tensors::OperationError::UnsupportedTensorContractScope { message }
                    if message.starts_with("eigh_full requires identical")
            )
    ));
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_eigh_aligned_assembly_matches_the_per_tree_path_bitwise() {
    // What: cuSOLVER's ascending order is the published one (#1985), so an
    // aligned route copies its eigenvectors once and the general path once
    // per codomain tree, with no selector. The selector-GEMM assembly (one
    // GEMM per route or per tree, one selector upload per call), still the
    // path for a non-identity order, is forced for comparison. All four are
    // exact data movement and publish the same values; the GEMM's `x * 1 + 0`
    // may turn a `-0.0` into `+0.0`, so zeros are compared by value.
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 1),
            (U1Irrep::new(1), 2),
        ],
    )
    .unwrap();
    let source =
        TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&leg, &leg], [&leg, &leg], 11)
            .unwrap();
    let source = source.axpby(1.0, &source.adjoint().unwrap(), 1.0).unwrap();
    let device = source.to_cuda().unwrap();
    let sectors = sector_regions(
        device.logical_space().space().structure(),
        device.logical_space().space().nout(),
    )
    .unwrap();
    let trees: usize = sectors.iter().map(|region| region.row_trees().len()).sum();
    assert!(
        trees > sectors.len(),
        "the fixture needs multi-tree sectors"
    );

    let mut outputs = Vec::new();
    for (treewise, forced, uploads, copies, gemms) in [
        (false, false, 0, sectors.len(), 0),
        (true, false, 0, trees, 0),
        (false, true, 1, 0, sectors.len()),
        (true, true, 1, 0, trees),
    ] {
        let routes = RoutePolicy {
            eigh_treewise: treewise,
            eigh_force_selector: forced,
            ..RoutePolicy::DEFAULT
        };
        let recording = probe::record();
        let Eigh { d, v } = probe::with_routes(routes, || {
            device
                .eigh_full(
                    &codomain_axes(&device),
                    &domain_axes(&device),
                    HermitianTol::DEFAULT,
                )
                .unwrap()
        });
        assert_eq!(recording.get(Event::CudaSelectorUpload), uploads);
        let (_, factor_copies, _, _, assembly_gemms, _, _) = qr_counts(&recording);
        assert_eq!((factor_copies, assembly_gemms), (copies, gemms));
        outputs.push((d.to_host().unwrap(), v.to_host().unwrap()));
    }
    let bits = |map: &TensorMap<U1FusionRule, f64>| {
        map.dense_data()
            .unwrap()
            .iter()
            .map(|value| (value + 0.0).to_bits())
            .collect::<Vec<_>>()
    };
    for (d, v) in &outputs[1..] {
        assert_eq!(bits(&outputs[0].0), bits(d));
        assert_eq!(bits(&outputs[0].1), bits(v));
    }
    let (d, v) = &outputs[0];
    assert_typed_map_close(&source.compose(v).unwrap(), &v.compose(d).unwrap(), 1.0e-10);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_eigh_selector_applies_a_non_identity_order() {
    // What: cuSOLVER's order is already ascending, so the selector assembly
    // only runs for a non-identity order. Reversing every odd sector's order
    // (test toggle) makes that permutation real next to identity sectors:
    // values follow the injected order, every column pairs with its value
    // (`t v = v d`), and one selector is uploaded, on aligned and per-tree
    // routes alike.
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 1),
            (U1Irrep::new(1), 2),
        ],
    )
    .unwrap();
    let source =
        TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&leg, &leg], [&leg, &leg], 13)
            .unwrap();
    let source = source.axpby(1.0, &source.adjoint().unwrap(), 1.0).unwrap();
    let device = source.to_cuda().unwrap();
    let sectors = sector_regions(
        device.logical_space().space().structure(),
        device.logical_space().space().nout(),
    )
    .unwrap();
    let trees: usize = sectors.iter().map(|region| region.row_trees().len()).sum();
    let reversed: Vec<bool> = sectors
        .iter()
        .enumerate()
        .map(|(index, region)| index % 2 == 1 && region.rows() >= 2)
        .collect();
    assert!(reversed.iter().any(|&r| r) && reversed.iter().any(|&r| !r));
    let eigh = |device: &TensorMap<U1FusionRule, f64, CudaStorage<f64>>| {
        let Eigh { d, v } = device
            .eigh_full(
                &codomain_axes(device),
                &domain_axes(device),
                HermitianTol::DEFAULT,
            )
            .unwrap();
        (d.to_host().unwrap(), v.to_host().unwrap())
    };
    let (ascending, _) = eigh(&device);
    let ascending = ascending.diagview().unwrap();
    // The spectra list the sectors in source-region order.
    assert_eq!(
        ascending
            .iter()
            .map(|entry| entry.values.len())
            .collect::<Vec<_>>(),
        sectors
            .iter()
            .map(|region| region.rows())
            .collect::<Vec<_>>()
    );

    for (treewise, gemms) in [(false, sectors.len()), (true, trees)] {
        let routes = RoutePolicy {
            eigh_treewise: treewise,
            eigh_reverse_odd_sectors: true,
            ..RoutePolicy::DEFAULT
        };
        let recording = probe::record();
        let (d, v) = probe::with_routes(routes, || eigh(&device));
        assert_eq!(recording.get(Event::CudaSelectorUpload), 1);
        let (_, factor_copies, _, _, assembly_gemms, _, _) = qr_counts(&recording);
        assert_eq!((factor_copies, assembly_gemms), (0, gemms));
        for ((entry, want), &reverse) in d.diagview().unwrap().iter().zip(&ascending).zip(&reversed)
        {
            let mut want = want.values.clone();
            if reverse {
                want.reverse();
            }
            assert_eq!(entry.values, want, "treewise {treewise}");
        }
        assert_typed_map_close(
            &source.compose(&v).unwrap(),
            &v.compose(&d).unwrap(),
            1.0e-10,
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_qr_work_and_preflight_are_streamed_and_transactional() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let provider = Arc::new(SU2FusionRule);
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap();
    let source = TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg], |_, indices| {
        indices.iter().sum::<usize>() as f64 + 1.0
    })
    .unwrap();
    let regions = sector_regions(
        source.logical_space().space().structure(),
        source.logical_space().space().nout(),
    )
    .unwrap();
    let nonempty = regions
        .iter()
        .filter(|region| region.rows() != 0 && region.cols() != 0)
        .count();
    let source_device = source.to_cuda().unwrap();
    // Per-route transfer and kernel counts follow the proved layout flag.
    let plan = source_device.compact_factor_plan().unwrap();
    let (factor_copies, selector_uploads, assembly_gemms) = cuda_route_assembly_counts(&plan);
    let routes = executed_routes(&plan).count();
    assert_eq!(routes, nonempty);
    // Both factor spaces of this fixture reproduce the source tree layout,
    // so every route takes the whole-factor copy and the assembly uploads
    // and downloads nothing. The non-aligned fallback is a layout
    // property, not a workload one: the Host plan's tree-order predicate
    // decides it, and the test `RoutePolicy` forces it on the device.
    assert!(
        executed_routes(&plan)
            .all(|(route, _, _)| plan.left_preserves_trees(route)
                && plan.right_preserves_trees(route)),
        "expected an all-aligned route mix, got {:?}",
        executed_routes(&plan)
            .map(|(route, _, _)| (
                plan.left_preserves_trees(route),
                plan.right_preserves_trees(route)
            ))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        (factor_copies, selector_uploads, assembly_gemms),
        (2 * routes, 0, 0)
    );

    let recording = probe::record();
    source_device
        .qr_compact(&codomain_axes(&source_device), &domain_axes(&source_device))
        .unwrap()
        .pair();
    assert_eq!(
        qr_counts(&recording),
        (
            nonempty,
            factor_copies,
            selector_uploads,
            2,
            assembly_gemms,
            0,
            usize::from(nonempty != 0),
        )
    );

    let malformed_storage = {
        let lease = runtime.lease_cuda().unwrap();
        CudaStorage::<f64>::upload(&lease, &[]).unwrap()
    };
    let malformed = TensorMap {
        runtime: runtime.clone(),
        repr: owned_repr(TypedTensorBody::dense(
            source.logical_space().clone(),
            malformed_storage,
        )),
    };
    let recording = probe::record();
    assert!(matches!(
        malformed.qr_compact(&codomain_axes(&malformed), &domain_axes(&malformed)),
        Err(Error::InvalidArgument(message)) if message.contains("payload length")
    ));
    assert_eq!(qr_counts(&recording), (0, 0, 0, 0, 0, 0, 0));

    let stranded_storage = {
        let lease = runtime.lease_cuda().unwrap();
        CudaStorage::upload(&lease, source.dense_data().unwrap()).unwrap()
    };
    let stranded = TensorMap {
        runtime: Runtime::builder().build().unwrap(),
        repr: owned_repr(TypedTensorBody::dense(
            source.logical_space().clone(),
            stranded_storage,
        )),
    };
    let recording = probe::record();
    assert!(matches!(
        stranded.qr_compact(&codomain_axes(&stranded), &domain_axes(&stranded)),
        Err(Error::InvalidArgument(message)) if message.contains("without a CUDA device")
    ));
    assert_eq!(qr_counts(&recording), (0, 0, 0, 0, 0, 0, 0));

    let zn3 = Arc::new(ZNFusionRule::new(3).unwrap());
    let charge0 = GradedSpace::try_new(Arc::clone(&zn3), [(zn3.irrep(0), 1)]).unwrap();
    let charge1 = GradedSpace::try_new(Arc::clone(&zn3), [(zn3.irrep(1), 1)]).unwrap();
    let empty: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&charge0], [&charge1], |_, _| 1.0).unwrap();
    let recording = probe::record();
    empty
        .to_cuda()
        .unwrap()
        .qr_compact(&codomain_axes(&empty), &domain_axes(&empty))
        .unwrap()
        .pair();
    assert_eq!(qr_counts(&recording), (0, 0, 0, 2, 0, 0, 0));
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_svd_work_is_streamed_and_preflight_is_transactional() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let provider = Arc::new(SU2FusionRule);
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap();
    let source = TensorMap::from_subblock_fn(&runtime, [&leg, &leg], [&leg], |_, indices| {
        indices.iter().sum::<usize>() as f64 + 1.0
    })
    .unwrap();
    let regions = sector_regions(
        source.logical_space().space().structure(),
        source.logical_space().space().nout(),
    )
    .unwrap();
    let nonempty = regions
        .iter()
        .filter(|region| region.rows() != 0 && region.cols() != 0)
        .count();
    let singular_values = regions
        .iter()
        .map(|region| region.rows().min(region.cols()))
        .sum();
    let source_device = source.to_cuda().unwrap();
    // Compact SVD assembles through the same aligned-copy dispatch as QR
    // and shares its copy/selector/GEMM observation.
    let plan = source_device.compact_factor_plan().unwrap();
    let (factor_copies, _, assembly_gemms) = cuda_route_assembly_counts(&plan);
    let recording = probe::record();
    source_device
        .svd_compact(&codomain_axes(&source_device), &domain_axes(&source_device))
        .unwrap();
    assert_eq!(
        svd_counts(&recording),
        (nonempty, singular_values, 3, 0, usize::from(nonempty != 0))
    );
    // No device QR and no QR output upload happen here; the shared events
    // record only this assembly's copies and GEMMs. The SVD builds its gauge
    // selectors on the device, so it uploads none.
    assert_eq!(
        qr_counts(&recording),
        (0, factor_copies, 0, 0, assembly_gemms, 0, 0)
    );

    let malformed_storage = {
        let lease = runtime.lease_cuda().unwrap();
        CudaStorage::<f64>::upload(&lease, &[]).unwrap()
    };
    let malformed = TensorMap {
        runtime: runtime.clone(),
        repr: owned_repr(TypedTensorBody::dense(
            source.logical_space().clone(),
            malformed_storage,
        )),
    };
    for lazy in [false, true] {
        let recording = probe::record();
        let rejected = if lazy {
            source_device.adjoint().unwrap().svd_compact(&[0], &[1, 2])
        } else {
            malformed.svd_compact(&codomain_axes(&malformed), &domain_axes(&malformed))
        };
        assert!(rejected.is_err());
        assert_eq!(svd_counts(&recording), (0, 0, 0, 0, 0));
    }

    let stranded_storage = {
        let lease = runtime.lease_cuda().unwrap();
        CudaStorage::upload(&lease, source.dense_data().unwrap()).unwrap()
    };
    let stranded = TensorMap {
        runtime: Runtime::builder().build().unwrap(),
        repr: owned_repr(TypedTensorBody::dense(
            source.logical_space().clone(),
            stranded_storage,
        )),
    };
    let recording = probe::record();
    assert!(stranded
        .svd_compact(&codomain_axes(&stranded), &domain_axes(&stranded))
        .is_err());
    assert_eq!(svd_counts(&recording), (0, 0, 0, 0, 0));
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_svd_non_aligned_routes_upload_no_selector_and_the_same_diagonal() {
    // What: the rustdoc's upload count, three zero-initialized factors
    // and no selector upload even on non-aligned routes (the gauge
    // selector is built on the device), and a diagonal that does not
    // depend on the assembly path.
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 1),
            (U1Irrep::new(1), 2),
        ],
    )
    .unwrap();
    let source =
        TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&leg, &leg], [&leg], 13).unwrap();
    let device = source.to_cuda().unwrap();
    let routes = sector_regions(
        source.logical_space().space().structure(),
        source.logical_space().space().nout(),
    )
    .unwrap()
    .iter()
    .filter(|region| region.rows() != 0 && region.cols() != 0)
    .count();
    let mut diagonals = Vec::new();
    for (treewise, selectors) in [(false, 0), (true, 0)] {
        let recording = probe::record();
        let Svd { s, .. } = probe::with_routes(svd_routes(treewise), || {
            device
                .svd_compact(&codomain_axes(&device), &domain_axes(&device))
                .unwrap()
        });
        let (_, _, creations, _, _) = svd_counts(&recording);
        let (_, _, selector_uploads, _, _, _, _) = qr_counts(&recording);
        assert_eq!(
            (creations, selector_uploads),
            (3, selectors),
            "treewise {treewise}"
        );
        diagonals.push(
            s.to_host()
                .unwrap()
                .dense_data()
                .unwrap()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
        );
    }
    assert!(routes > 1);
    assert_eq!(diagonals[0], diagonals[1]);
}

/// Each nonempty coupled sector of a host tensor as `(sector, rows, cols,
/// column-major widened values)`.
#[cfg(feature = "cuda")]
fn gauge_sector_matrices<R, D>(
    tensor: &TensorMap<R, D>,
) -> Vec<(SectorId, usize, usize, Vec<Complex64>)>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: CudaFactorizationPayload,
{
    let tensor = tensor.materialize().unwrap();
    let data = tensor.dense_data().unwrap();
    let space = tensor.logical_space().space();
    sector_regions(space.structure(), space.nout())
        .unwrap()
        .iter()
        .filter(|region| region.rows() != 0 && region.cols() != 0)
        .map(|region| {
            let values = data[region.range()]
                .iter()
                .map(|value| value.widen_complex())
                .collect();
            (region.coupled(), region.rows(), region.cols(), values)
        })
        .collect()
}

/// `A = U diag(g) Vh` from the Host SVD of `host`, with every sector's
/// spectrum replaced by `g(i) = 2` for `i < 2` and `1 / (2 + i)` after:
/// an exactly degenerate top pair in every sector of dimension >= 2.
#[cfg(feature = "cuda")]
fn with_degenerate_spectrum<R, D>(host: &TensorMap<R, D>) -> TensorMap<R, D>
where
    R: symtn_core::TypedSectorAdmission<
            Sector = <R as symtn_core::SectorCodec>::Sector,
            Error = symtn_core::FusionAlgebraError,
            Mode = symtn_core::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: CudaFactorizationPayload,
{
    let Svd { u, s, vh } = host
        .svd_compact(&codomain_axes(host), &domain_axes(host))
        .unwrap();
    let bond = s.domain();
    let spectrum = TensorMap::<R, D>::from_subblock_fn(host.runtime(), &bond, &bond, |_, index| {
        if index[0] != index[1] {
            D::zero()
        } else if index[0] < 2 {
            D::from_real(2.0)
        } else {
            D::from_real(1.0 / (2 + index[0]) as f64)
        }
    })
    .unwrap();
    u.compose(&spectrum).unwrap().compose(&vh).unwrap()
}

/// Consistency contract of #1552, Host SVD as the oracle: (1) singular
/// values agree; (2) a vector of a non-degenerate singular value with a
/// clear pivot agrees entrywise (same sign/phase gauge); (3) a degenerate
/// group, or a vector whose pivot is ambiguous at solver rounding, agrees
/// by its projector. Also: the pivot of every device `u` column is real
/// and non-negative. Returns how many vectors were compared entrywise.
#[cfg(feature = "cuda")]
fn assert_device_svd_gauge_matches_host<R, D>(host: &TensorMap<R, D>) -> usize
where
    R: symtn_core::TypedSectorAdmission<
            Sector = <R as symtn_core::SectorCodec>::Sector,
            Error = symtn_core::FusionAlgebraError,
            Mode = symtn_core::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: CudaFactorizationPayload,
{
    let tolerance = 1.0e3 * D::epsilon();
    let expected = host
        .svd_compact(&codomain_axes(host), &domain_axes(host))
        .unwrap();
    let actual = host
        .to_cuda()
        .unwrap()
        .svd_compact(&codomain_axes(host), &domain_axes(host))
        .unwrap();
    let actual = Svd {
        u: actual.u.to_host().unwrap(),
        s: actual.s.to_host().unwrap(),
        vh: actual.vh.to_host().unwrap(),
    };
    let matrices = |svd: &Svd<TensorMap<R, D>>| {
        (
            gauge_sector_matrices(&svd.u),
            gauge_sector_matrices(&svd.s),
            gauge_sector_matrices(&svd.vh),
        )
    };
    let (host_u, host_s, host_vh) = matrices(&expected);
    let (device_u, device_s, device_vh) = matrices(&actual);
    assert_eq!(host_u.len(), device_u.len());
    let find = |list: &[(SectorId, usize, usize, Vec<Complex64>)], sector: SectorId| {
        list.iter()
            .find(|entry| entry.0 == sector)
            .cloned()
            .expect("every sector has all three factors")
    };
    let mut entrywise = 0;
    for (sector, rows, k, u_host) in &host_u {
        let (rows, k) = (*rows, *k);
        let (_, _, _, u_device) = find(&device_u, *sector);
        let (_, _, cols, vh_host) = find(&host_vh, *sector);
        let (_, _, _, vh_device) = find(&device_vh, *sector);
        let s_host = find(&host_s, *sector).3;
        let s_device = find(&device_s, *sector).3;
        let singular = |s: &[Complex64], i: usize| s[i * (k + 1)].re;
        // (1) singular values.
        for i in 0..k {
            let (d, h) = (singular(&s_device, i), singular(&s_host, i));
            assert!(
                (d - h).abs() <= tolerance * h.abs().max(1.0),
                "s: {d} vs {h}"
            );
        }
        let u_col = |u: &[Complex64], j: usize| u[j * rows..(j + 1) * rows].to_vec();
        let vh_row =
            |vh: &[Complex64], j: usize| (0..cols).map(|c| vh[j + k * c]).collect::<Vec<_>>();
        // Device pivot: the first largest-|u| entry is real and >= 0.
        for j in 0..k {
            let column = u_col(&u_device, j);
            let top = column.iter().map(|x| x.norm()).fold(0.0, f64::max);
            assert!(
                column
                    .iter()
                    .any(|x| x.norm() >= top - tolerance && x.im.abs() <= tolerance && x.re >= 0.0),
                "device u column {j} of {sector:?} has no real non-negative pivot: {column:?}"
            );
        }
        // Group singular values into degenerate runs.
        let mut start = 0;
        while start < k {
            let mut end = start + 1;
            while end < k
                && (singular(&s_host, end - 1) - singular(&s_host, end)).abs()
                    <= 1.0e-6 * singular(&s_host, start).max(1.0)
            {
                end += 1;
            }
            let clear_pivot = |j: usize| {
                let mut magnitudes: Vec<f64> = u_col(u_host, j).iter().map(|x| x.norm()).collect();
                magnitudes.sort_by(|a, b| b.partial_cmp(a).unwrap());
                magnitudes.len() < 2 || magnitudes[0] - magnitudes[1] > 1.0e-6
            };
            if end - start == 1 && clear_pivot(start) {
                // (2) same gauge, entrywise.
                for (d, h) in u_col(&u_device, start).iter().zip(u_col(u_host, start)) {
                    assert!(
                        (d - h).norm() <= tolerance * 10.0,
                        "u {sector:?}[{start}]: {d} vs {h}"
                    );
                }
                for (d, h) in vh_row(&vh_device, start)
                    .iter()
                    .zip(vh_row(&vh_host, start))
                {
                    assert!(
                        (d - h).norm() <= tolerance * 10.0,
                        "vh {sector:?}[{start}]: {d} vs {h}"
                    );
                }
                entrywise += 1;
            } else {
                // (3) projectors of the run.
                let projector = |vectors: Vec<Vec<Complex64>>| {
                    let n = vectors[0].len();
                    let mut p = vec![Complex64::new(0.0, 0.0); n * n];
                    for v in &vectors {
                        for a in 0..n {
                            for b in 0..n {
                                p[a + n * b] += v[a] * v[b].conj();
                            }
                        }
                    }
                    p
                };
                let run = start..end;
                for (host_vectors, device_vectors) in [
                    (
                        run.clone().map(|j| u_col(u_host, j)).collect::<Vec<_>>(),
                        run.clone().map(|j| u_col(&u_device, j)).collect::<Vec<_>>(),
                    ),
                    (
                        run.clone().map(|j| vh_row(&vh_host, j)).collect(),
                        run.clone().map(|j| vh_row(&vh_device, j)).collect(),
                    ),
                ] {
                    let (p, q) = (projector(host_vectors), projector(device_vectors));
                    let difference = p
                        .iter()
                        .zip(&q)
                        .map(|(a, b)| (a - b).norm_sqr())
                        .sum::<f64>()
                        .sqrt();
                    assert!(
                        difference <= tolerance * 10.0,
                        "projector {sector:?}[{run:?}]: {difference:e}"
                    );
                }
            }
            start = end;
        }
    }
    entrywise
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_svd_follows_the_host_gauge_with_and_without_degenerate_spectra() {
    // What: #1552's three-part consistency contract against the Host SVD
    // over U(1) and SU(2), f64 and c64, generic and exactly degenerate
    // spectra, and aligned as well as forced per-tree (non-aligned)
    // assembly, where the gauge rides the selector GEMM instead.
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let u1 = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 3),
            (U1Irrep::new(1), 2),
        ],
    )
    .unwrap();
    let su2 = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap();
    fn cases<R>(runtime: &Runtime, leg: &GradedSpace<R>) -> usize
    where
        R: symtn_core::TypedSectorAdmission<
                Sector = <R as symtn_core::SectorCodec>::Sector,
                Error = symtn_core::FusionAlgebraError,
                Mode = symtn_core::MultiplicityFreeAdmissionMode,
            > + MultiplicityFreeRigidSymbols<Scalar = f64>
            + CheckedFusionAlgebra
            + SectorCodec,
    {
        let real = TensorMap::<R, f64>::rand_with_seed(runtime, [leg, leg], [leg], 21).unwrap();
        let complex =
            TensorMap::<R, Complex64>::rand_with_seed(runtime, [leg, leg], [leg], 22).unwrap();
        let mut entrywise = 0;
        for treewise in [false, true] {
            probe::with_routes(svd_routes(treewise), || {
                entrywise += assert_device_svd_gauge_matches_host(&real);
                entrywise += assert_device_svd_gauge_matches_host(&complex);
                assert_device_svd_gauge_matches_host(&with_degenerate_spectrum(&real));
                assert_device_svd_gauge_matches_host(&with_degenerate_spectrum(&complex));
            });
        }
        entrywise
    }
    // The entrywise (sign/phase) statement must not be vacuous.
    assert!(cases(&runtime, &u1) > 8);
    assert!(cases(&runtime, &su2) > 8);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_svd_of_a_near_tie_follows_the_first_largest_entry() {
    // What: A = [[1, -1], [-1, 1]] has singular vectors whose entries tie
    // in exact arithmetic. cuSOLVER returns them 1 ulp apart
    // (0.7071067811865475 vs ...476, observed on the A100), so the rule
    // is checked exactly on the device's own output: in every column the
    // first entry of largest magnitude is positive. The ±1 scaling is
    // exact, so these magnitudes are cuSOLVER's. No column or `vh` row may
    // be zeroed (a tie-break that summed tied entries would do that).
    // The exact-tie contract (first row wins) is pinned on hand-built
    // factors in `symtn-dense/tests/cuda_svd_gauge.rs`.
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let host =
        TensorMap::<U1FusionRule, f64>::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
            if index[0] == index[1] {
                1.0
            } else {
                -1.0
            }
        })
        .unwrap();
    let Svd { u, s, vh } = host
        .to_cuda()
        .unwrap()
        .svd_compact(&codomain_axes(&host), &domain_axes(&host))
        .unwrap();
    let (u, s, vh) = (
        u.to_host().unwrap(),
        s.to_host().unwrap(),
        vh.to_host().unwrap(),
    );
    let half = std::f64::consts::FRAC_1_SQRT_2;
    let u = u.dense_data().unwrap();
    for column in u.chunks(2) {
        let pivot = if column[1].abs() > column[0].abs() {
            column[1]
        } else {
            column[0]
        };
        assert!(pivot > 0.0, "u = {u:?}");
    }
    let s = s.dense_data().unwrap();
    assert!((s[0] - 2.0).abs() <= 1e-12 && s[3].abs() <= 1e-12);
    // Column 0 spans [1, -1], column 1 spans [1, 1], none zeroed.
    assert!(
        (u[0] + u[1]).abs() <= 1e-12 && (u[2] - u[3]).abs() <= 1e-12,
        "u = {u:?}"
    );
    let vh = vh.dense_data().unwrap();
    for value in u.iter().chain(vh) {
        assert!(
            (value.abs() - half).abs() <= 1e-12,
            "u = {u:?}, vh = {vh:?}"
        );
    }
    // vh row 0 takes u column 0's phase: A = 2 u0 vh0 with vh0 = u0.
    assert!(
        (vh[0] - u[0]).abs() <= 1e-12 && (vh[2] - u[1]).abs() <= 1e-12,
        "vh = {vh:?}"
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_svd_gauge_costs_the_documented_ops_and_only_the_finite_scan_download() {
    // What: the rustdoc's cost of the gauge. Per nonempty route, 13 ops for
    // the phases, 2 per aligned side (broadcast, mul) or 1 per non-aligned
    // side (the selector's embed_diagonal), and 1 conj for a complex left
    // side. Per call, one gauge-weight upload (no per-route identity
    // selectors); nothing is downloaded.
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 1),
            (U1Irrep::new(1), 2),
        ],
    )
    .unwrap();
    fn check<D: CudaFactorizationPayload>(
        runtime: &Runtime,
        leg: &GradedSpace<U1FusionRule>,
        seed: u64,
    ) {
        let host =
            TensorMap::<U1FusionRule, D>::rand_with_seed(runtime, [leg, leg], [leg], seed).unwrap();
        let device = host.to_cuda().unwrap();
        let regions = sector_regions(
            host.logical_space().space().structure(),
            host.logical_space().space().nout(),
        )
        .unwrap();
        let plan = device.compact_factor_plan().unwrap();
        let max_rows = executed_routes(&plan)
            .map(|(route, _, _)| regions[route.source_region()].rows())
            .max()
            .unwrap();
        for treewise in [false, true] {
            let expected_ops: u64 = executed_routes(&plan)
                .map(|(route, _, _)| {
                    let side = |aligned: bool| if aligned && !treewise { 2 } else { 1 };
                    13 + side(plan.left_preserves_trees(route))
                        + side(plan.right_preserves_trees(route))
                        + u64::from(<D as symtn_dense::CudaScalar>::IS_COMPLEX)
                })
                .sum();
            let recording = probe::record();
            let before = symtn_dense::cuda_transfer_stats();
            let Svd { u, s, vh } = probe::with_routes(svd_routes(treewise), || {
                device
                    .svd_compact(&codomain_axes(&device), &domain_axes(&device))
                    .unwrap()
            });
            let after = symtn_dense::cuda_transfer_stats();
            let (_, _, selector_uploads, _, _, _, _) = qr_counts(&recording);
            assert_eq!(selector_uploads, 0, "treewise {treewise}");
            assert_eq!(
                after.gauge_ops - before.gauge_ops,
                expected_ops,
                "treewise {treewise}"
            );
            // The finite-input scan's one scalar (#2038); nothing else.
            assert_eq!(after.d2h_calls - before.d2h_calls, 1, "treewise {treewise}");
            assert_eq!(after.h2d_calls - before.h2d_calls, 4, "treewise {treewise}");
            let len = |t: &TensorMap<U1FusionRule, D, CudaStorage<D>>| {
                t.logical_space().space().required_len().unwrap()
            };
            let payload = |len: usize| (len * std::mem::size_of::<D>()) as u64;
            assert_eq!(
                after.h2d_bytes - before.h2d_bytes,
                payload(len(&u) + len(&s) + len(&vh))
                    + (max_rows * std::mem::size_of::<i64>()) as u64,
                "treewise {treewise}"
            );
        }
    }
    check::<f64>(&runtime, &leg, 31);
    check::<Complex64>(&runtime, &leg, 32);
}

/// An independent reference for `s` (#1536): each nonempty
/// sector's cuSOLVER spectrum, downloaded and placed on a host zero
/// buffer by [`fill_diagonal_values`]. Returned as widened bits.
#[cfg(feature = "cuda")]
fn downloaded_svd_diagonal_bits<R, D>(
    device: &TensorMap<R, D, CudaStorage<D>>,
    s_len: usize,
    s_structure: &BlockStructure,
) -> Vec<(u64, u64)>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: CudaFactorizationPayload,
{
    let source = device.direct_cuda_storage("test").unwrap();
    let space = device.logical_space().space();
    let regions = sector_regions(space.structure(), space.nout()).unwrap();
    let mut lease = device.runtime.lease_cuda().unwrap();
    let cuda = &mut *lease;
    let mut sectors = Vec::new();
    let mut spectra = Vec::new();
    for region in regions.iter() {
        if region.rows() == 0 || region.cols() == 0 {
            continue;
        }
        let (_, spectrum, _) = cuda_svd_region::<D>(
            cuda,
            &source.0,
            region.range().start,
            region.rows(),
            region.cols(),
        )
        .unwrap();
        sectors.push(region.coupled());
        spectra.push(spectrum);
    }
    let spectra: Vec<_> = sectors
        .into_iter()
        .zip(cuda_download_spectra::<D>(cuda, &spectra).unwrap())
        .map(|(sector, values)| symtn_matrixalgebra::SectorSpectrum { sector, values })
        .collect();
    let mut host = vec![<D as symtn_dense::CudaScalar>::ZERO; s_len];
    fill_diagonal_values(s_structure, &mut host, &spectra).unwrap();
    host.into_iter().map(widened_bits).collect()
}

#[cfg(feature = "cuda")]
fn widened_bits<D: FactorScalar>(value: D) -> (u64, u64) {
    let value = value.widen_complex();
    (value.re.to_bits(), value.im.to_bits())
}

#[cfg(feature = "cuda")]
fn assert_device_svd_diagonal_matches_the_downloaded_one<R, D>(host: &TensorMap<R, D>)
where
    R: symtn_core::TypedSectorAdmission<
            Sector = <R as symtn_core::SectorCodec>::Sector,
            Error = symtn_core::FusionAlgebraError,
            Mode = symtn_core::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: CudaFactorizationPayload,
{
    let regions = sector_regions(
        host.logical_space().space().structure(),
        host.logical_space().space().nout(),
    )
    .unwrap();
    let nonempty = regions
        .iter()
        .filter(|region| region.rows() != 0 && region.cols() != 0)
        .count();
    assert!(nonempty > 1, "the fixture needs several blocks");
    let device = host.to_cuda().unwrap();
    let Svd { s, .. } = device
        .svd_compact(&codomain_axes(&device), &domain_axes(&device))
        .unwrap();
    let s = s.to_host().unwrap();
    let expected = downloaded_svd_diagonal_bits(
        &device,
        s.materialize().unwrap().dense_data().unwrap().len(),
        s.logical_space().space().structure(),
    );
    let actual: Vec<_> = s
        .materialize()
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .copied()
        .map(widened_bits)
        .collect();
    assert_eq!(actual, expected);
    // The spectra themselves agree with the Host SVD to dtype tolerance.
    let Svd { s: host_s, .. } = host
        .svd_compact(&codomain_axes(host), &domain_axes(host))
        .unwrap();
    let tolerance = 1.0e3 * D::epsilon();
    for (device, host) in s
        .materialize()
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(host_s.materialize().unwrap().dense_data().unwrap())
    {
        let (device, host) = (device.widen_complex(), host.widen_complex());
        assert!(
            (device - host).norm() <= tolerance * host.norm().max(1.0),
            "{device} vs {host}"
        );
    }
}

#[cfg(feature = "cuda")]
/// Every fixture below has a coupled sector on one side only: it has no
/// block, no route and no diagonal region, and must not shift the others.
fn assert_device_svd_diagonal_every_dtype<R>(
    runtime: &Runtime,
    codomain: &[&GradedSpace<R>],
    domain: &[&GradedSpace<R>],
) where
    R: symtn_core::TypedSectorAdmission<
            Sector = <R as symtn_core::SectorCodec>::Sector,
            Error = symtn_core::FusionAlgebraError,
            Mode = symtn_core::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
{
    let fused = |legs: &[&GradedSpace<R>]| {
        legs[1..]
            .iter()
            .fold(legs[0].clone(), |acc, leg| acc.fuse(leg).unwrap())
            .sectors()
            .unwrap()
    };
    let (coupled_codomain, coupled_domain) = (fused(codomain), fused(domain));
    assert!(
        coupled_codomain
            .iter()
            .any(|sector| !coupled_domain.contains(sector))
            || coupled_domain
                .iter()
                .any(|sector| !coupled_codomain.contains(sector)),
        "the fixture needs a coupled sector on one side only"
    );
    let host = |seed| {
        TensorMap::<R, f64>::rand_with_seed(
            runtime,
            codomain.iter().copied(),
            domain.iter().copied(),
            seed,
        )
        .unwrap()
    };
    assert_device_svd_diagonal_matches_the_downloaded_one(&host(3));
    assert_device_svd_diagonal_matches_the_downloaded_one(
        &TensorMap::<R, Complex64>::rand_with_seed(
            runtime,
            codomain.iter().copied(),
            domain.iter().copied(),
            5,
        )
        .unwrap(),
    );
    assert_device_svd_diagonal_matches_the_downloaded_one(
        &TensorMap::<R, f32>::rand_with_seed(
            runtime,
            codomain.iter().copied(),
            domain.iter().copied(),
            7,
        )
        .unwrap(),
    );
    assert_device_svd_diagonal_matches_the_downloaded_one(
        &TensorMap::<R, num_complex::Complex32>::rand_with_seed(
            runtime,
            codomain.iter().copied(),
            domain.iter().copied(),
            11,
        )
        .unwrap(),
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_svd_diagonal_written_on_device_equals_the_downloaded_diagonal_bitwise() {
    // What: `s` is written by a device strided copy of each sector's
    // spectrum; it must equal, bit for bit, the host-filled `s` built from
    // the same solver's downloaded spectra, across symmetries, dtypes,
    // multi-tree (non-aligned) sectors and an empty coupled sector.
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();

    let u1 = Arc::new(U1FusionRule);
    let u1_leg = |charges: &[(i32, usize)]| {
        GradedSpace::try_new(
            Arc::clone(&u1),
            charges
                .iter()
                .map(|&(charge, dim)| (U1Irrep::new(charge), dim)),
        )
        .unwrap()
    };
    let small = u1_leg(&[(-1, 2), (0, 1), (1, 2)]);
    // Coupled sector 3 exists in the domain only: an empty route.
    let wide = u1_leg(&[(-1, 3), (0, 4), (1, 2), (3, 2)]);
    assert_device_svd_diagonal_every_dtype(&runtime, &[&small, &small], &[&wide]);

    let su2 = Arc::new(SU2FusionRule);
    let su2_leg = GradedSpace::try_new(
        Arc::clone(&su2),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap();
    // Spin 3/2 occurs in the codomain only.
    assert_device_svd_diagonal_every_dtype(&runtime, &[&su2_leg, &su2_leg, &su2_leg], &[&su2_leg]);

    let fermion = Arc::new(U1FusionRule.product(FermionParityFusionRule));
    let fermion_leg = GradedSpace::try_new(
        Arc::clone(&fermion),
        [
            (product_sector(U1Irrep::new(0), Z2Irrep::EVEN), 2),
            (product_sector(U1Irrep::new(1), Z2Irrep::ODD), 2),
            (product_sector(U1Irrep::new(-1), Z2Irrep::ODD), 1),
        ],
    )
    .unwrap();
    assert_device_svd_diagonal_every_dtype(
        &runtime,
        &[&fermion_leg, &fermion_leg],
        &[&fermion_leg],
    );
}

/// A payload built from widened components; the imaginary part is dropped
/// for a real payload.
#[cfg(feature = "cuda")]
trait NonfiniteFixture: CudaFactorizationPayload {
    const MAX: f64;
    const SUBNORMAL: f64;
    fn from_parts(re: f64, im: f64) -> Self;
}
#[cfg(feature = "cuda")]
impl NonfiniteFixture for f32 {
    const MAX: f64 = f32::MAX as f64;
    const SUBNORMAL: f64 = (f32::MIN_POSITIVE / 4.0) as f64;
    fn from_parts(re: f64, _: f64) -> Self {
        re as f32
    }
}
#[cfg(feature = "cuda")]
impl NonfiniteFixture for f64 {
    const MAX: f64 = f64::MAX;
    const SUBNORMAL: f64 = f64::MIN_POSITIVE / 4.0;
    fn from_parts(re: f64, _: f64) -> Self {
        re
    }
}
#[cfg(feature = "cuda")]
impl NonfiniteFixture for num_complex::Complex32 {
    const MAX: f64 = f32::MAX as f64;
    const SUBNORMAL: f64 = (f32::MIN_POSITIVE / 4.0) as f64;
    fn from_parts(re: f64, im: f64) -> Self {
        num_complex::Complex32::new(re as f32, im as f32)
    }
}
#[cfg(feature = "cuda")]
impl NonfiniteFixture for Complex64 {
    const MAX: f64 = f64::MAX;
    const SUBNORMAL: f64 = f64::MIN_POSITIVE / 4.0;
    fn from_parts(re: f64, im: f64) -> Self {
        Complex64::new(re, im)
    }
}

/// Three U(1) sectors of `rows x cols` blocks, real symmetric and
/// diagonally dominant when square; `entry` replaces the last element of the
/// last sector, so a check that stopped early would miss it.
#[cfg(feature = "cuda")]
fn finite_input_fixture<D: NonfiniteFixture>(
    runtime: &Runtime,
    rows: usize,
    cols: usize,
    entry: Option<(f64, f64)>,
) -> TensorMap<U1FusionRule, D> {
    let leg = |dim: usize| {
        GradedSpace::try_new(
            Arc::new(U1FusionRule),
            (0..3).map(|charge| (U1Irrep::new(charge), dim)),
        )
        .unwrap()
    };
    let last = U1Irrep::new(2);
    TensorMap::from_subblock_fn(runtime, [&leg(rows)], [&leg(cols)], |trees, index| {
        let (i, j) = (index[0], index[1]);
        if let Some((re, im)) = entry {
            if *trees.coupled() == last && i == rows - 1 && j == cols - 1 {
                return D::from_parts(re, im);
            }
        }
        let value = if i == j {
            (rows + i) as f64
        } else {
            1.0 / (1.0 + i.abs_diff(j) as f64)
        };
        D::from_parts(value, 0.0)
    })
    .unwrap()
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug)]
enum FiniteInputOp {
    Qr,
    Svd,
    Eigh,
}

/// Runs `op` on the current split; the error is rendered for comparison.
#[cfg(feature = "cuda")]
fn run_finite_input_op<D, S>(
    tensor: &TensorMap<U1FusionRule, D, S>,
    op: FiniteInputOp,
) -> Result<(), String>
where
    D: FactorizationScalar,
    S: FactorExec<U1FusionRule, D>,
{
    let (rows, cols) = (codomain_axes(tensor), domain_axes(tensor));
    match op {
        FiniteInputOp::Qr => tensor.qr_compact(&rows, &cols).map(drop),
        FiniteInputOp::Svd => tensor.svd_compact(&rows, &cols).map(drop),
        FiniteInputOp::Eigh => tensor
            .eigh_full(&rows, &cols, HermitianTol::DEFAULT)
            .map(drop),
    }
    .map_err(|err| format!("{err:?}"))
}

#[cfg(feature = "cuda")]
fn assert_device_refuses_nonfinite_input_like_the_host<D: NonfiniteFixture>(runtime: &Runtime) {
    let mut entries = vec![
        (f64::NAN, 0.0),
        (f64::INFINITY, 0.0),
        (f64::NEG_INFINITY, 0.0),
    ];
    if <D as symtn_dense::CudaScalar>::IS_COMPLEX {
        entries.extend([(0.0, f64::INFINITY), (0.0, f64::NAN)]);
    }
    let mut cases = vec![
        (FiniteInputOp::Qr, 8, 8),
        (FiniteInputOp::Svd, 8, 8),
        (FiniteInputOp::Eigh, 8, 8),
    ];
    if <D as symtn_dense::CudaScalar>::IS_COMPLEX && <D as FactorScalar>::epsilon() < 1e-10 {
        // Above 1024 rows cuSOLVER's SVD is `gesvd`, which returns NaN
        // factors for a nonfinite input unless symtn rejects it first (#2038).
        cases.push((FiniteInputOp::Svd, 1100, 4));
    }
    for (op, rows, cols) in cases {
        for entry in &entries {
            let host = finite_input_fixture::<D>(runtime, rows, cols, Some(*entry));
            let expected = run_finite_input_op(&host, op).unwrap_err();
            assert!(
                expected.contains("input components must be finite"),
                "{op:?} {entry:?}: Host {expected}"
            );
            let device = host.to_cuda().unwrap();
            let before = symtn_dense::cuda_transfer_stats();
            let refused = run_finite_input_op(&device, op).unwrap_err();
            let after = symtn_dense::cuda_transfer_stats();
            assert_eq!(refused, expected, "{op:?} {rows}x{cols} {entry:?}");
            // No solver ran and no output was uploaded or allocated: the
            // scan's one scalar is the only transfer.
            assert_eq!(
                (
                    after.solver_calls - before.solver_calls,
                    after.h2d_calls - before.h2d_calls,
                    after.device_allocs - before.device_allocs,
                    after.d2h_calls - before.d2h_calls,
                ),
                (0, 0, 0, 1),
                "{op:?} {rows}x{cols} {entry:?}"
            );
        }
        // The extremes of the finite range pass the check; whatever the
        // solver then does, the call is not refused as nonfinite input.
        let mut finite = vec![(D::MAX, 0.0), (-D::MAX, 0.0), (D::SUBNORMAL, 0.0)];
        if <D as symtn_dense::CudaScalar>::IS_COMPLEX {
            finite.push((D::MAX, D::MAX));
        }
        for entry in finite {
            let device = finite_input_fixture::<D>(runtime, rows, cols, Some(entry))
                .to_cuda()
                .unwrap();
            if let Err(err) = run_finite_input_op(&device, op) {
                assert!(
                    !err.contains("input components must be finite"),
                    "{op:?} {entry:?}: {err}"
                );
            }
        }
        // A subnormal entry is an ordinary finite input: the device result
        // is the Host's.
        let host = finite_input_fixture::<D>(runtime, rows, cols, Some((D::SUBNORMAL, 0.0)));
        let device = host.to_cuda().unwrap();
        let before = symtn_dense::cuda_transfer_stats();
        run_finite_input_op(&device, op).unwrap();
        let after = symtn_dense::cuda_transfer_stats();
        assert!(after.solver_calls > before.solver_calls, "{op:?}");
        run_finite_input_op(&host, op).unwrap();
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_factorizations_refuse_nonfinite_input_with_the_host_error() {
    // What: eager device QR, SVD and EIGH refuse a NaN or infinite real or
    // imaginary component with the Host's error (#1986, #2038), before any
    // solver call or output upload; EIGH reports it ahead of Hermiticity.
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    assert_device_refuses_nonfinite_input_like_the_host::<f32>(&runtime);
    assert_device_refuses_nonfinite_input_like_the_host::<f64>(&runtime);
    assert_device_refuses_nonfinite_input_like_the_host::<num_complex::Complex32>(&runtime);
    assert_device_refuses_nonfinite_input_like_the_host::<Complex64>(&runtime);
}
