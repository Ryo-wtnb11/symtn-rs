use super::*;

/// `(zero uploads, kernels)` of the device arithmetic in `recording`.
#[cfg(feature = "cuda")]
fn arithmetic(recording: &probe::Recording) -> (usize, usize) {
    (
        recording.get(Event::CudaZeroUpload),
        recording.get(Event::CudaArithmeticKernel),
    )
}

/// `(downloads, device partials, host partials)` of the device reductions
/// in `recording`.
#[cfg(feature = "cuda")]
fn reduction(recording: &probe::Recording) -> (usize, usize, usize) {
    (
        recording.get(Event::CudaReductionDownload),
        recording.get(Event::CudaReductionDevicePartials),
        recording.get(Event::CudaReductionHostPartials),
    )
}

#[cfg(feature = "cuda")]
fn assert_cuda_tensor_matches_host<R>(
    actual: &TensorMap<R, f64>,
    expected: &TensorMap<R, f64>,
    provider: *const R,
    runtime: &crate::runtime::RuntimeIdentity,
) where
    R: symtn_core::TypedSectorAdmission<
            Sector = <R as symtn_core::SectorCodec>::Sector,
            Error = symtn_core::FusionAlgebraError,
            Mode = symtn_core::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
{
    assert!(std::ptr::eq(actual.provider(), provider));
    assert!(runtime.matches(actual.runtime()));
    assert_eq!(
        actual.logical_space().space(),
        expected.logical_space().space()
    );
    assert_eq!(actual.dense_data().unwrap(), expected.dense_data().unwrap());
    assert_eq!(actual.subblock_count(), expected.subblock_count());
    for index in 0..actual.subblock_count() {
        let actual_block = actual.subblock(index).unwrap();
        let expected_block = expected.subblock(index).unwrap();
        assert_eq!(actual_block.key(), expected_block.key());
        assert_eq!(actual_block.offset(), expected_block.offset());
        assert_eq!(actual_block.shape(), expected_block.shape());
        assert_eq!(actual_block.strides(), expected_block.strides());
        assert_eq!(
            actual.subblock_fusion_trees(index).unwrap(),
            expected.subblock_fusion_trees(index).unwrap()
        );
    }
}

#[cfg(feature = "cuda")]
fn assert_cuda_lazy_contract_orientations<R>(lhs: &TensorMap<R, f64>, rhs: &TensorMap<R, f64>)
where
    R: symtn_core::TypedSectorAdmission<
            Sector = <R as symtn_core::SectorCodec>::Sector,
            Error = symtn_core::FusionAlgebraError,
            Mode = symtn_core::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
{
    let lhs_axes: Vec<_> = (lhs.codomain_rank()..lhs.rank()).collect();
    let rhs_axes: Vec<_> = (0..rhs.codomain_rank()).collect();
    let output_axes: Vec<_> = (0..lhs.codomain_rank() + rhs.domain_rank()).collect();
    let (codomain, domain) = output_axes.split_at(lhs.codomain_rank());
    let spec = ContractSpec {
        lhs: &lhs_axes,
        rhs: &rhs_axes,
        codomain,
        domain,
    };
    let expected_contract = lhs.contract(rhs, &spec).unwrap();
    let expected_compose = lhs.compose(rhs).unwrap();
    let provider = lhs.provider() as *const R;
    let runtime = lhs.runtime().identity();

    for (lhs_adjoint, rhs_adjoint) in [(false, false), (true, false), (false, true), (true, true)] {
        let device_operand = |logical: &TensorMap<R, f64>, adjoint: bool| {
            if adjoint {
                eager_adjoint_oracle(logical)
                    .to_cuda()
                    .unwrap()
                    .adjoint()
                    .unwrap()
            } else {
                logical.to_cuda().unwrap()
            }
        };
        let lhs_device = device_operand(lhs, lhs_adjoint);
        let rhs_device = device_operand(rhs, rhs_adjoint);
        let contract = lhs_device
            .contract(&rhs_device, &spec)
            .unwrap()
            .to_host()
            .unwrap();
        let compose = lhs_device.compose(&rhs_device).unwrap().to_host().unwrap();

        assert_cuda_tensor_matches_host(&contract, &expected_contract, provider, &runtime);
        assert_cuda_tensor_matches_host(&compose, &expected_compose, provider, &runtime);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn typed_cuda_owned_metadata_validation_orders_ordinal_before_length() {
    type DeviceTensor = TensorMap<U1FusionRule, f64, CudaStorage>;
    assert!(DeviceTensor::validate_cuda_owned_metadata(
        Placement::Cuda(0),
        Placement::Cuda(0),
        7,
        7
    )
    .is_ok());
    assert_eq!(
        DeviceTensor::validate_cuda_owned_metadata(Placement::Cuda(1), Placement::Cuda(0), 7, 6)
            .unwrap_err(),
        Error::PlacementMismatch
    );
    assert!(matches!(
        DeviceTensor::validate_cuda_owned_metadata(
            Placement::Cuda(0),
            Placement::Cuda(0),
            7,
            6
        ),
        Err(Error::InvalidArgument(message)) if message.contains("payload length")
    ));
}

#[cfg(feature = "cuda")]
#[test]
fn missing_cuda_context_precedes_compact_expansion_and_lazy_materialization() {
    let densified = probe::record();
    let source = u1_lazy_fixture();
    let diagonal = source
        .svd_compact(&codomain_axes(&source), &domain_axes(&source))
        .unwrap()
        .s;
    let lazy = source.adjoint().unwrap();
    let TypedData::Diagonal(spectrum) = owned(&diagonal).data.as_ref() else {
        unreachable!("SVD factor is compact")
    };
    let mut malformed_spectrum = spectrum.clone();
    for entry in &mut malformed_spectrum {
        entry.values.clear();
    }
    let malformed_expansion = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        symtn_matrixalgebra::seam::diagonal_bond_data(
            diagonal.logical_space().space(),
            &malformed_spectrum,
            &|value| value,
        )
    }));
    assert!(
        malformed_expansion.is_err() || matches!(malformed_expansion, Ok(Err(_))),
        "the fixture must fail if compact expansion runs"
    );
    let malformed = TensorMap {
        runtime: diagonal.runtime.clone(),
        repr: owned_repr(TypedTensorBody::diagonal(
            diagonal.logical_space().clone(),
            malformed_spectrum,
        )),
    };
    let missing_context = Error::InvalidArgument(
        "this runtime was built without a CUDA device; use Runtime::builder().cuda(device)"
            .to_string(),
    );

    assert_eq!(malformed.to_cuda().unwrap_err(), missing_context);
    assert!(matches!(lazy.to_cuda(), Err(error) if error == missing_context));
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn typed_cuda_compact_and_lazy_roundtrips_keep_source_caches_cold() {
    let densified = probe::record();
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();

    let diagonal = source
        .svd_compact(&codomain_axes(&source), &domain_axes(&source))
        .unwrap()
        .s;
    let TypedData::Diagonal(spectrum) = owned(&diagonal).data.as_ref() else {
        unreachable!("SVD factor is compact")
    };
    let expected_diagonal = symtn_matrixalgebra::seam::diagonal_bond_data(
        diagonal.logical_space().space(),
        spectrum,
        &|value| value,
    )
    .unwrap();
    let diagonal_device = diagonal.to_cuda().unwrap();
    assert_eq!(diagonal_device.placement(), Placement::Cuda(0));
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    let diagonal_host = diagonal_device.to_host().unwrap();
    assert!(matches!(
        owned(&diagonal_host).data.as_ref(),
        TypedData::Dense(_)
    ));
    assert_eq!(diagonal_host.dense_data().unwrap(), expected_diagonal);
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);

    let lazy = source.adjoint().unwrap();
    let expected_lazy = symtn_tensors::materialize_adjoint_data_dyn(
        source.logical_space().space(),
        lazy.logical_space().space(),
        source.dense_data().unwrap(),
    )
    .unwrap();
    let lazy_device = lazy.to_cuda().unwrap();
    let TypedTensorRepr::Adjoint(device_view) = &lazy_device.repr else {
        unreachable!("transfer preserves the lazy view")
    };
    let expected_norm = source.norm(2.0).unwrap();
    let recording = probe::record();
    assert!(
        (lazy_device.norm(2.0).unwrap() - expected_norm).abs() <= 1e-12 * (1.0 + expected_norm)
    );
    let observed = reduction(&recording);
    let sector_count = sector_regions(
        source.logical_space().space().structure(),
        source.logical_space().space().nout(),
    )
    .unwrap()
    .len();
    assert_eq!(observed, (1, sector_count.max(1), sector_count.max(1)));
    assert!(source.dense_data().unwrap().len() > sector_count.max(1));

    // `$arithmetic` is `(zero uploads, kernels)`. The thread's transfer
    // counters pin that the output's zero initialization is the only upload:
    // no coefficient is uploaded (#2187) and nothing is downloaded.
    macro_rules! observed_arithmetic {
        ($expression:expr, $arithmetic:expr, $reduction:expr) => {{
            let recording = probe::record();
            let before = symtn_dense::cuda_transfer_stats();
            let result = $expression;
            let after = symtn_dense::cuda_transfer_stats();
            let (zero_uploads, _): (usize, usize) = $arithmetic;
            assert_eq!(after.h2d_calls - before.h2d_calls, zero_uploads as u64);
            assert_eq!(after.d2h_calls, before.d2h_calls);
            assert_eq!(arithmetic(&recording), $arithmetic);
            assert_eq!(reduction(&recording), $reduction);
            result
        }};
    }

    let source_device = source.to_cuda().unwrap();
    let empty_storage = {
        let lease = runtime.lease_cuda().unwrap();
        CudaStorage::<f64>::upload(&lease, &[]).unwrap()
    };
    let malformed_length = TensorMap {
        runtime: runtime.clone(),
        repr: owned_repr(TypedTensorBody::dense(
            source.logical_space().clone(),
            empty_storage,
        )),
    };
    let recording = probe::record();
    assert!(matches!(
        malformed_length.scale(2.0),
        Err(Error::InvalidArgument(message)) if message.contains("payload length")
    ));
    assert_eq!(arithmetic(&recording), (0, 0));

    // One `combine` body for every representation (#2187): one zero upload,
    // then one strided read per stored block of each nonzero-coefficient
    // operand, an owned operand being one packed block. The first call
    // uploads the context's shared `1` once per dtype; warm it outside the
    // windows.
    let owned_reads = 1;
    source_device.scale(1.5).unwrap();
    observed_arithmetic!(source_device.scale(-2.0), (1, owned_reads), (0, 0, 0)).unwrap();
    observed_arithmetic!(
        source_device.axpby(2.0, &source_device, -3.0),
        (1, 2 * owned_reads),
        (0, 0, 0)
    )
    .unwrap();
    observed_arithmetic!(
        source_device.axpby(0.0, &source_device, -3.0),
        (1, owned_reads),
        (0, 0, 0)
    )
    .unwrap();
    observed_arithmetic!(source_device.zeros_like(), (1, 0), (0, 0, 0)).unwrap();

    let lazy_scale =
        observed_arithmetic!(lazy_device.scale(-2.0), (1, owned_reads), (0, 0, 0)).unwrap();
    let lazy_add = observed_arithmetic!(
        lazy_device.axpby(2.0, &lazy_device, -3.0),
        (1, 2 * owned_reads),
        (0, 0, 0)
    )
    .unwrap();
    let lazy_zero = observed_arithmetic!(lazy_device.zeros_like(), (1, 0), (0, 0, 0)).unwrap();
    for result in [&lazy_scale, &lazy_add, &lazy_zero] {
        assert!(matches!(result.repr, TypedTensorRepr::Adjoint(_)));
    }
    // Mixed lazy/owned (#1874): one zero output upload, no coefficient
    // upload, one strided read per lazy block plus one for the owned operand.
    let blocks = lazy_device
        .logical_space()
        .space()
        .structure()
        .block_count();
    let mixed = observed_arithmetic!(
        lazy_device.axpby(2.0, &source_device, -3.0),
        (1, blocks + owned_reads),
        (0, 0, 0)
    )
    .unwrap();
    assert!(matches!(mixed.repr, TypedTensorRepr::Owned(_)));
    let expected_mixed = lazy.axpby(2.0, &source, -3.0).unwrap();
    for (device, host) in mixed
        .to_host()
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .zip(expected_mixed.dense_data().unwrap())
    {
        assert!((device - host).abs() <= 1e-12 * (1.0 + host.abs()));
    }
    // A warm lazy-source `axpby_into` (#1874) reads the parent in place:
    // no transfer in either direction and no device allocation.
    let mut destination = mixed;
    lazy_device.axpby_into(&mut destination, 2.0, 0.0).unwrap();
    let before = symtn_dense::cuda_transfer_stats();
    lazy_device.axpby_into(&mut destination, -1.5, 0.5).unwrap();
    let after = symtn_dense::cuda_transfer_stats();
    assert_eq!(after.h2d_calls, before.h2d_calls);
    assert_eq!(after.d2h_calls, before.d2h_calls);
    assert_eq!(after.device_allocs, before.device_allocs);

    assert!(matches!(
        lazy_device.inner(&lazy_device),
        Err(Error::UnsupportedOnDevice(_))
    ));
    assert!(matches!(
        lazy_device.inner(&lazy_device),
        Err(Error::UnsupportedOnDevice(_))
    ));

    let mut missing_context = lazy_device.clone();
    missing_context.runtime = Runtime::builder().build().unwrap();
    let recording = probe::record();
    assert!(matches!(
        missing_context.norm(2.0),
        Err(Error::InvalidArgument(message)) if message.contains("without a CUDA device")
    ));
    assert_eq!(reduction(&recording), (0, 0, 0));
    let recording = probe::record();
    assert!(matches!(
        missing_context.zeros_like(),
        Err(Error::InvalidArgument(message)) if message.contains("without a CUDA device")
    ));
    assert_eq!(arithmetic(&recording), (0, 0));
    let device_clone = lazy_device.clone();
    let TypedTensorRepr::Adjoint(clone_view) = &device_clone.repr else {
        unreachable!("clone preserves the lazy view")
    };
    assert!(Arc::ptr_eq(device_view, clone_view));

    let lazy_host = device_clone.to_host().unwrap();
    let TypedTensorRepr::Adjoint(_) = &lazy_host.repr else {
        unreachable!("roundtrip preserves the lazy view")
    };
    assert_eq!(
        lazy_host.materialize().unwrap().dense_data().unwrap(),
        expected_lazy
    );
}

/// #1268: a `Complex64` device payload must perform exactly the same
/// number of device allocations, zero uploads, kernels, and
/// reduction downloads as the `f64` payload on the same structure. Only
/// the bytes per element change (covered in `symtn-dense`).
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_complex_payload_costs_the_same_device_calls_as_f64() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let real: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();
    let complex: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
            let ramp = indices.iter().sum::<usize>() as f64;
            Complex64::new(ramp + 1.0, -(ramp + 1.75))
        })
        .unwrap();

    fn observe<T>(run: impl FnOnce() -> T) -> ((usize, usize), (usize, usize, usize)) {
        let recording = probe::record();
        drop(run());
        (arithmetic(&recording), reduction(&recording))
    }

    let real_device = real.to_cuda().unwrap();
    let complex_device = complex.to_cuda().unwrap();
    assert_eq!(
        observe(|| real_device.scale(-2.0).unwrap()),
        observe(|| complex_device.scale(Complex64::new(-2.0, 0.5)).unwrap())
    );
    assert_eq!(
        observe(|| real_device.axpby(2.0, &real_device, -3.0).unwrap()),
        observe(|| complex_device
            .axpby(
                Complex64::new(2.0, 1.0),
                &complex_device,
                Complex64::new(-3.0, 0.25)
            )
            .unwrap())
    );
    assert_eq!(
        observe(|| real_device.zeros_like().unwrap()),
        observe(|| complex_device.zeros_like().unwrap())
    );
    assert_eq!(
        observe(|| real_device.inner(&real_device).unwrap()),
        observe(|| complex_device.inner(&complex_device).unwrap())
    );
    // Contraction allocates its destination and runs its kernels inside
    // the replay seam, which has no arithmetic/reduction hooks: both
    // dtypes must leave those counters untouched rather than falling back
    // to the axpby or reduction paths.
    let untouched = ((0, 0), (0, 0, 0));
    assert_eq!(
        observe(|| real_device
            .contract(
                &real_device,
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &[0],
                    domain: &[1]
                }
            )
            .unwrap()),
        untouched
    );
    assert_eq!(
        observe(|| complex_device
            .contract(
                &complex_device,
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &[0],
                    domain: &[1]
                }
            )
            .unwrap()),
        untouched
    );
}

#[cfg(feature = "cuda")]
#[test]
fn typed_cuda_reduction_placement_validation_is_exact() {
    assert!(validate_cuda_reduction_placement(
        Placement::Cuda(0),
        Placement::Cuda(0),
        Placement::Cuda(0)
    )
    .is_ok());
    assert_eq!(
        validate_cuda_reduction_placement(
            Placement::Cuda(1),
            Placement::Cuda(0),
            Placement::Cuda(0)
        )
        .unwrap_err(),
        Error::PlacementMismatch
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_lazy_adjoint_contract_and_compose_match_rectangular_host_oracles() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = |degeneracy| {
        GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0), degeneracy)]).unwrap()
    };
    let (m, k, n) = (2, 3, 4);
    let lhs = TensorMap::from_subblock_fn(&runtime, [&leg(m)], [&leg(k)], |_, indices| {
        (indices[0] + m * indices[1]) as f64 + 1.0
    })
    .unwrap();
    let rhs = TensorMap::from_subblock_fn(&runtime, [&leg(k)], [&leg(n)], |_, indices| {
        (2 * indices[0] + indices[1]) as f64 + 1.0
    })
    .unwrap();
    let expected_contract = lhs
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
    let expected_compose = lhs.compose(&rhs).unwrap();

    for upload_parent_first in [false, true] {
        for (lhs_adjoint, rhs_adjoint) in
            [(false, false), (true, false), (false, true), (true, true)]
        {
            let device_operand = |logical: &TensorMap<U1FusionRule, f64>, adjoint: bool| {
                if !adjoint {
                    return logical.to_cuda().unwrap();
                }
                let parent = eager_adjoint_oracle(logical);
                if upload_parent_first {
                    parent.to_cuda().unwrap().adjoint().unwrap()
                } else {
                    parent.adjoint().unwrap().to_cuda().unwrap()
                }
            };
            let lhs_device = device_operand(&lhs, lhs_adjoint);
            let rhs_device = device_operand(&rhs, rhs_adjoint);
            let contracted = lhs_device
                .contract(
                    &rhs_device,
                    &ContractSpec {
                        lhs: &[1],
                        rhs: &[0],
                        codomain: &[0],
                        domain: &[1],
                    },
                )
                .unwrap();
            let composed = lhs_device.compose(&rhs_device).unwrap();
            let contracted = contracted.to_host().unwrap();
            let composed = composed.to_host().unwrap();

            assert_eq!(
                contracted.logical_space().space(),
                expected_contract.logical_space().space()
            );
            assert_eq!(
                composed.logical_space().space(),
                expected_compose.logical_space().space()
            );
            assert_eq!(
                contracted.dense_data().unwrap(),
                expected_contract.dense_data().unwrap()
            );
            assert_eq!(
                composed.dense_data().unwrap(),
                expected_compose.dense_data().unwrap()
            );
            assert!(Arc::ptr_eq(
                contracted.logical_space().provider_arc(),
                lhs.logical_space().provider_arc()
            ));
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_lazy_adjoint_preserves_fermionic_contract_sign() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let provider = Arc::new(FermionParityFusionRule);
    let odd = |is_dual| {
        GradedSpace::try_new(Arc::clone(&provider), [(Z2Irrep::ODD, 1)])
            .and_then(|space| if is_dual { space.try_dual() } else { Ok(space) })
            .unwrap()
    };
    let lhs =
        TensorMap::from_subblock_fn(&runtime, [&odd(false)], [&odd(true)], |_, _| 2.0).unwrap();
    let rhs =
        TensorMap::from_subblock_fn(&runtime, [&odd(true)], [&odd(false)], |_, _| 3.0).unwrap();

    for (lhs_adjoint, rhs_adjoint) in [(false, false), (true, false), (false, true), (true, true)] {
        let device_operand = |logical: &TensorMap<FermionParityFusionRule, f64>, adjoint: bool| {
            if adjoint {
                eager_adjoint_oracle(logical)
                    .to_cuda()
                    .unwrap()
                    .adjoint()
                    .unwrap()
            } else {
                logical.to_cuda().unwrap()
            }
        };
        let lhs_device = device_operand(&lhs, lhs_adjoint);
        let rhs_device = device_operand(&rhs, rhs_adjoint);
        let contract = lhs_device
            .contract(
                &rhs_device,
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &[0],
                    domain: &[1],
                },
            )
            .unwrap()
            .to_host()
            .unwrap();
        let compose = lhs_device.compose(&rhs_device).unwrap().to_host().unwrap();

        assert_eq!(contract.dense_data().unwrap(), &[-6.0]);
        assert_eq!(compose.dense_data().unwrap(), &[6.0]);
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_lazy_adjoint_covers_su2_rank_five_and_simple_product() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();

    let su2_provider = Arc::new(SU2FusionRule);
    let su2 = GradedSpace::try_new(
        Arc::clone(&su2_provider),
        [
            (SU2Irrep::from_twice_spin(0), 1),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap();
    let su2_lhs =
        TensorMap::from_subblock_fn(&runtime, [&su2, &su2, &su2], [&su2, &su2], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();
    let su2_rhs =
        TensorMap::from_subblock_fn(&runtime, [&su2, &su2], [&su2, &su2, &su2], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 3.0
        })
        .unwrap();
    assert_cuda_lazy_contract_orientations(&su2_lhs, &su2_rhs);

    let product_provider = Arc::new(U1FusionRule.product(FermionParityFusionRule));
    let product = GradedSpace::try_new(
        Arc::clone(&product_provider),
        [
            (product_sector(U1Irrep::new(0), Z2Irrep::EVEN), 2),
            (product_sector(U1Irrep::new(1), Z2Irrep::ODD), 1),
        ],
    )
    .unwrap();
    let product_lhs =
        TensorMap::from_subblock_fn(&runtime, [&product], [&product], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();
    let product_rhs =
        TensorMap::from_subblock_fn(&runtime, [&product], [&product], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 4.0
        })
        .unwrap();
    assert_cuda_lazy_contract_orientations(&product_lhs, &product_rhs);
}

/// GL-3 (#1281): a device operation must not need the coarse Runtime
/// state mutex. The reverse direction is the observable one: Host
/// standalone contraction never takes `state`, so only a parked holder of
/// that lock can prove a device operation is independent of it.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn device_work_completes_while_another_thread_holds_the_runtime_state_lock() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc;
    use std::sync::Barrier;
    use std::time::Duration;

    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    let lhs = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        indices[0] as f64 + 1.0
    })
    .unwrap();
    let rhs = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        indices[1] as f64 + 2.0
    })
    .unwrap();

    // Independent oracle for the device result, computed on Host before
    // the state lock is parked.
    let host_expected = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        indices[0] as f64 + 1.0
    })
    .unwrap()
    .contract(
        &TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
            indices[1] as f64 + 2.0
        })
        .unwrap(),
        &ContractSpec {
            lhs: &[1],
            rhs: &[0],
            codomain: &[0],
            domain: &[1],
        },
    )
    .unwrap();

    let holding = Arc::new(Barrier::new(2));
    let release = Arc::new(AtomicBool::new(false));
    let (sender, receiver) = mpsc::channel();

    let values = std::thread::scope(|scope| {
        let holder_runtime = runtime.clone();
        let holder_barrier = Arc::clone(&holding);
        let holder_release = Arc::clone(&release);
        scope.spawn(move || {
            let _state = holder_runtime.lock();
            holder_barrier.wait();
            while !holder_release.load(Ordering::SeqCst) {
                std::thread::yield_now();
            }
        });

        holding.wait();
        scope.spawn(move || {
            let device = lhs
                .to_cuda()
                .and_then(|lhs| Ok((lhs, rhs.to_cuda()?)))
                .and_then(|(lhs, rhs)| {
                    lhs.contract(
                        &rhs,
                        &ContractSpec {
                            lhs: &[1],
                            rhs: &[0],
                            codomain: &[0],
                            domain: &[1],
                        },
                    )
                })
                .and_then(|out| out.to_host());
            let _ = sender.send(device.map(|out| out.dense_data().unwrap().to_vec()));
        });

        let outcome = receiver.recv_timeout(Duration::from_secs(30));
        release.store(true, Ordering::SeqCst);
        outcome
            .expect("device transfer and contraction blocked on the Runtime state lock")
            .expect("device contraction failed")
    });

    // The device result under the parked state lock is the Host result.
    assert_eq!(values.len(), host_expected.dense_data().unwrap().len());
    assert!(values.iter().any(|value| *value != 0.0));
    for (actual, expected) in values.iter().zip(host_expected.dense_data().unwrap()) {
        assert!(
            (actual - expected).abs() < 1.0e-12,
            "device value {actual} differs from the Host oracle {expected}"
        );
    }
}

/// GL-3 (#1281): the device lock nests with the CPU leases in either
/// order without deadlocking, because it is a separate mutex.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn device_lease_nests_with_cpu_leases_in_both_orders() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();

    {
        let cuda = runtime.lease_cuda().unwrap();
        let context = runtime.lease_context().unwrap();
        let dense = runtime.lease_dense();
        drop(dense);
        // A CPU lease released under the device guard must not deadlock.
        drop(context);
        drop(cuda);
    }

    let context = runtime.lease_context().unwrap();
    let cuda = runtime.lease_cuda().unwrap();
    drop(context);
    drop(cuda);
}

/// GL-3 (#1281): device lowering touches no execution context, so a
/// device contraction leaves the Runtime tree-transform cache untouched.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a real CUDA device"]
fn device_contraction_leaves_the_tree_transform_cache_unchanged() {
    if crate::test_cache::run_isolated_or_return(
        "SYMTN_DEVICE_CONTRACTION_CACHE_ISOLATED",
        "typed::representation_gates::cuda::device_contraction_leaves_the_tree_transform_cache_unchanged",
    ) {
        return;
    }
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    let lhs = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        indices[0] as f64 + 1.0
    })
    .unwrap()
    .to_cuda()
    .unwrap();
    let rhs = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        indices[1] as f64 + 2.0
    })
    .unwrap()
    .to_cuda()
    .unwrap();

    let before = crate::test_cache::completed();
    let product = lhs
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
    assert_eq!(product.placement(), Placement::Cuda(0));
    assert_eq!(crate::test_cache::completed(), before);
}

/// A destination whose device is not the Runtime's is rejected before any
/// device work, its bytes untouched (#1551). The public API cannot build
/// one — a tensor's storage lives on its own Runtime's device, and the
/// Runtime check comes first — so this in-crate gate forges it from a
/// second device's tensor.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires two real CUDA devices"]
fn typed_cuda_into_rejects_a_foreign_device_destination_untouched() {
    let rt0 = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let rt1 = Runtime::builder().cuda(1).dense_threads(1).build().unwrap();
    let v = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(0), 2),
            (U1Irrep::new(1), 1),
            (U1Irrep::new(-1), 2),
        ],
    )
    .unwrap();
    let w = v.try_dual().unwrap();
    let host_t = TensorMap::<_, f64>::rand_with_seed(&rt0, [&v, &w], [&v, &w], 1).unwrap();
    let host_rhs = TensorMap::<_, f64>::rand_with_seed(&rt0, [&v, &w], [&v], 2).unwrap();
    let (t, rhs) = (host_t.to_cuda().unwrap(), host_rhs.to_cuda().unwrap());
    let spec = ContractSpec {
        lhs: &[2, 3],
        rhs: &[0, 1],
        codomain: &[2, 0],
        domain: &[1],
    };
    let foreign = |like: TensorMap<U1FusionRule, f64>| {
        let (codomain, domain) = (like.codomain(), like.domain());
        let mut next = 0u64;
        let mut device = TensorMap::<_, f64>::from_subblock_fn(&rt1, &codomain, &domain, |_, _| {
            next += 1;
            std::f64::consts::PI * next as f64
        })
        .unwrap()
        .to_cuda()
        .unwrap();
        let before: Vec<u64> = device
            .to_host()
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect();
        device.runtime = rt0.clone();
        (device, before)
    };
    type Device = TensorMap<U1FusionRule, f64, CudaStorage<f64>>;
    let check = |what: &str,
                 (mut device, before): (Device, Vec<u64>),
                 call: &dyn Fn(&mut Device) -> Result<(), Error>| {
        assert_eq!(call(&mut device), Err(Error::PlacementMismatch), "{what}");
        device.runtime = rt1.clone();
        let after: Vec<u64> = device
            .to_host()
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect();
        assert_eq!(after, before, "{what}: destination changed");
    };
    check(
        "permute_into",
        foreign(host_t.permute(&[2, 0], &[1, 3]).unwrap()),
        &|d| t.permute_into(&[2, 0], &[1, 3], d, 1.0, 0.5),
    );
    check(
        "braid_into",
        foreign(host_t.braid(&[1, 0], &[3, 2], &[0, 1, 2, 3]).unwrap()),
        &|d| t.braid_into(&[1, 0], &[3, 2], &[0, 1, 2, 3], d, 1.0, 0.5),
    );
    check(
        "transpose_into",
        foreign(host_t.transpose(&[1, 3], &[0, 2]).unwrap()),
        &|d| t.transpose_into(&[1, 3], &[0, 2], d, 1.0, 0.5),
    );
    check(
        "repartition_into",
        foreign(host_t.repartition(1).unwrap()),
        &|d| t.repartition_into(d, 1.0, 0.5),
    );
    check(
        "trace_pairs_into",
        foreign(host_t.trace_pairs(&[(0, 2)]).unwrap()),
        &|d| t.trace_pairs_into(&[(0, 2)], d, 1.0, 0.5),
    );
    check(
        "contract_into",
        foreign(host_t.contract(&host_rhs, &spec).unwrap()),
        &|d| t.contract_into(&rhs, &spec, d, 1.0, 0.5),
    );
    check("axpby_into", foreign(host_t.clone()), &|d| {
        t.axpby_into(d, 1.0, 0.5)
    });
}

/// The device twist lowering (`cuda_twist_regions` fed by the body's
/// `twist_block_factor`) against the Host θ authority, without a device: one
/// region per block on that block's geometry, and per block the factor
/// `Π θ(uncoupled sector)` of the mode's `twist_values` (conjugated for the
/// inverse), which is also the ratio the Host `twist` applies to that block.
/// A lazy input lowers onto its parent with the inverse direction.
#[cfg(feature = "cuda")]
fn assert_cuda_twist_lowering_matches_host<R, D>(
    t: &TensorMap<R, D>,
    legs: &[usize],
    direction: Direction,
) -> usize
where
    R: symtn_core::TypedSectorAdmission<
            Sector = <R as symtn_core::SectorCodec>::Sector,
            Error = symtn_core::FusionAlgebraError,
            Mode = symtn_core::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: CudaPayload + std::fmt::Debug,
{
    let what = format!("{legs:?} {direction:?}");
    let expected = t.twist(legs, direction).unwrap();
    let (source, twisted, legs, inverse) = match (&t.repr, &expected.repr) {
        (TypedTensorRepr::Adjoint(view), TypedTensorRepr::Adjoint(result)) => (
            t.parent_of(view),
            expected.parent_of(result),
            logical_adjoint_axes_to_parent(
                view.parent.space.space().nout(),
                view.parent.space.space().nin(),
                legs,
            ),
            !direction.is_inverse(),
        ),
        (TypedTensorRepr::Owned(_), TypedTensorRepr::Owned(_)) => (
            t.clone(),
            expected.clone(),
            legs.to_vec(),
            direction.is_inverse(),
        ),
        _ => panic!("{what}: twist keeps the representation"),
    };
    let space = source.logical_space();
    let provider = space.provider();
    let nout = source.codomain_rank();
    let theta = <MultiplicityFreeAdmissionMode as TypedTensorTwistDispatch<R, D>>::twist_values(
        provider,
        space.space().structure(),
        nout,
        &legs,
    )
    .unwrap()
    .unwrap_or_else(|| panic!("{what}: fixture is not the identity"));
    let lowered = super::super::cuda_transform::cuda_twist_regions::<D, _>(
        space.space(),
        &super::super::twist_flip::twist_block_factor(&theta, nout, &legs, inverse),
        "twist",
    )
    .unwrap();
    let structure = space.space().structure();
    assert_eq!(
        lowered.len(),
        structure.block_count(),
        "{what}: one region per block"
    );
    let (before, after) = (source.dense_data().unwrap(), twisted.dense_data().unwrap());
    let mut negated = 0;
    for (index, (region, factor)) in lowered.iter().enumerate() {
        let block = structure.block(index).unwrap();
        assert_eq!(region.dims(), block.shape(), "{what}: block {index} shape");
        assert_eq!(
            region.strides(),
            block.strides(),
            "{what}: block {index} strides"
        );
        assert_eq!(
            region.offset(),
            block.offset(),
            "{what}: block {index} offset"
        );
        let BlockKey::FusionTree(key) = block.key() else {
            panic!("{what}: fusion-tree block");
        };
        let theta_product: f64 = legs
            .iter()
            .map(|&leg| provider.twist_scalar(uncoupled_sector_of_leg(key, nout, leg)))
            .product();
        assert_eq!(
            *factor,
            D::from_real(theta_product),
            "{what}: block {index} θ"
        );
        negated += usize::from(theta_product == -1.0);
        let mut offsets = vec![block.offset()];
        for (&extent, &stride) in block.shape().iter().zip(block.strides()) {
            offsets = offsets
                .iter()
                .flat_map(|&base| (0..extent).map(move |i| base + i * stride))
                .collect();
        }
        for offset in offsets {
            assert_eq!(
                after[offset],
                before[offset] * *factor,
                "{what}: block {index} element {offset} is the Host twist"
            );
        }
    }
    negated
}

#[cfg(feature = "cuda")]
#[test]
fn typed_cuda_twist_lowering_is_the_host_theta_authority() {
    fn fill<S>(_: &crate::typed::BlockFusionTrees<S>, ix: &[usize]) -> f64 {
        ix.iter().enumerate().fold(0.37, |value, (axis, &i)| {
            value * 1.7 + (i as f64 + 1.0) * (axis as f64 + 2.0)
        })
    }
    fn complex_fill<S>(trees: &crate::typed::BlockFusionTrees<S>, ix: &[usize]) -> Complex64 {
        let re = fill(trees, ix);
        Complex64::new(re, 0.25 - 0.5 * re)
    }
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let directions = [Direction::Forward, Direction::Inverse];

    // fZ2 with a dual codomain leg: both sides, both dualities, repeated legs.
    let fz2 = GradedSpace::try_new(
        Arc::new(FermionParityFusionRule),
        [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 1)],
    )
    .unwrap();
    let dual = fz2.try_dual().unwrap();
    let real: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&fz2, &dual], [&fz2, &fz2], fill).unwrap();
    let complex: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&fz2, &dual], [&fz2, &fz2], complex_fill).unwrap();
    for legs in [
        &[0usize][..],
        &[1],
        &[3],
        &[0, 3],
        &[1, 2, 3],
        &[0, 1, 1, 3],
    ] {
        for direction in directions {
            assert!(assert_cuda_twist_lowering_matches_host(&real, legs, direction) > 0);
            assert!(assert_cuda_twist_lowering_matches_host(&complex, legs, direction) > 0);
        }
    }

    // A lazy adjoint input lowers onto its parent with the inverse direction.
    let lazy =
        TensorMap::<_, Complex64>::from_subblock_fn(&runtime, [&fz2, &dual], [&fz2], complex_fill)
            .unwrap()
            .adjoint()
            .unwrap();
    for legs in [&[0usize][..], &[1], &[2], &[0, 1]] {
        for direction in directions {
            assert!(assert_cuda_twist_lowering_matches_host(&lazy, legs, direction) > 0);
        }
    }

    // Fermionic products: fZ2 x U(1), and fZ2 x SU(2) with non-Abelian
    // degeneracies.
    let fu1 = GradedSpace::try_new(
        Arc::new(FermionParityFusionRule.product(U1FusionRule)),
        [
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
            (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 1),
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(2)), 1),
        ],
    )
    .unwrap();
    let t: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&fu1, &fu1], [&fu1, &fu1], fill).unwrap();
    for legs in [&[1usize][..], &[2], &[0, 2]] {
        for direction in directions {
            assert!(assert_cuda_twist_lowering_matches_host(&t, legs, direction) > 0);
        }
    }
    let fsu2 = GradedSpace::try_new(
        Arc::new(FermionParityFusionRule.product(SU2FusionRule)),
        [
            (
                product_sector(Z2Irrep::EVEN, SU2Irrep::from_twice_spin(0)),
                2,
            ),
            (
                product_sector(Z2Irrep::ODD, SU2Irrep::from_twice_spin(1)),
                1,
            ),
            (
                product_sector(Z2Irrep::EVEN, SU2Irrep::from_twice_spin(2)),
                1,
            ),
        ],
    )
    .unwrap();
    let t: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&fsu2, &fsu2], [&fsu2, &fsu2], complex_fill)
            .unwrap();
    for legs in [&[0usize][..], &[3], &[1, 3]] {
        for direction in directions {
            assert!(assert_cuda_twist_lowering_matches_host(&t, legs, direction) > 0);
        }
    }

    // A bosonic provider is the identity before any block walk: no lowering.
    let su2 = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 1),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap();
    let t: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&su2, &su2], [&su2], fill).unwrap();
    let twisted = t.twist(&[0, 2], Direction::Forward).unwrap();
    assert!(Arc::ptr_eq(
        &t.owned_body().unwrap().data,
        &twisted.owned_body().unwrap().data
    ));
}

/// `space` with every block keyed opaquely instead of by its fusion-tree
/// pair (a structure cannot mix key kinds): same geometry, no adjoint
/// partner.
#[cfg(feature = "cuda")]
fn with_opaque_keys(
    space: &symtn_tensors::DynamicFusionMapSpace,
) -> symtn_tensors::DynamicFusionMapSpace {
    let structure = space.structure();
    let specs = (0..structure.block_count())
        .map(|i| {
            let block = structure.block(i).unwrap();
            BlockSpec::with_key(
                BlockKey::opaque([i as u64]),
                block.shape().to_vec(),
                block.strides().to_vec(),
                block.offset(),
            )
            .unwrap()
        })
        .collect();
    DynamicFusionMapSpace::try_from_layout(
        space.homspace().clone(),
        BlockStructure::from_blocks_with_rank(space.rank(), specs).unwrap(),
    )
    .unwrap()
}

/// The device lowering of the one [`symtn_tensors::AdjointBlockMap`]
/// (`cuda_adjoint_lowering`) against hand-computed copies and tables, without
/// a device. Per block of the adjoint layout the parent block is looked up by
/// hand (the pair with its trees swapped) and its strides are permuted by
/// hand (adjoint axis `a` is parent axis `nout + a` for `a < nin`, else
/// `a - nin`). With `zero_blocks` the adjoint layout keys its blocks
/// opaquely, so none has an adjoint source and each must read as zero. Applying the
/// lowering to the parent on the Host must equal the Host materialization.
#[cfg(feature = "cuda")]
fn assert_cuda_adjoint_lowering_matches_hand<R, D>(t: &TensorMap<R, D>, zero_blocks: bool)
where
    R: symtn_core::TypedSectorAdmission<
            Sector = <R as symtn_core::SectorCodec>::Sector,
            Error = symtn_core::FusionAlgebraError,
            Mode = symtn_core::MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: CudaPayload + std::fmt::Debug,
{
    let lazy = t.adjoint().unwrap();
    let TypedTensorRepr::Adjoint(view) = &lazy.repr else {
        panic!("dense adjoint is lazy");
    };
    let parent_space = view.parent.space.space();
    let logical = view.logical_space.space();
    let adjoint_space = if zero_blocks {
        with_opaque_keys(logical)
    } else {
        logical.clone()
    };
    let map = symtn_tensors::AdjointBlockMap::new(parent_space, &adjoint_space);
    let lowering = super::super::cuda_transfer::cuda_adjoint_lowering(&map, D::IS_COMPLEX).unwrap();
    // The elementwise kernels read every dtype through the copies (#1874).
    let regions = super::super::cuda_transfer::cuda_adjoint_lowering(&map, false).unwrap();

    let (nout, nin) = (parent_space.nout(), parent_space.nin());
    let parent_structure = parent_space.structure();
    let structure = adjoint_space.structure();
    let len = adjoint_space.required_len().unwrap();
    let elements = |shape: &[usize], strides: &[usize], offset: usize| {
        let mut offsets = vec![offset];
        for (&extent, &stride) in shape.iter().zip(strides) {
            offsets = offsets
                .iter()
                .flat_map(|&base| (0..extent).map(move |i| base + i * stride))
                .collect();
        }
        offsets
    };
    let mut copies = Vec::new();
    let mut table = vec![0usize; len];
    let mut zeros = Vec::new();
    for index in 0..structure.block_count() {
        let block = structure.block(index).unwrap();
        let destination = (
            block.shape().to_vec(),
            block.strides().to_vec(),
            block.offset(),
        );
        let BlockKey::FusionTree(key) = block.key() else {
            zeros.push(destination);
            continue;
        };
        let swapped = BlockKey::FusionTree(FusionTreePairKey::pair(
            key.domain_tree().clone(),
            key.codomain_tree().clone(),
        ));
        let parent = parent_structure
            .block(parent_structure.find_block_index_by_key(&swapped).unwrap())
            .unwrap();
        let strides: Vec<usize> = (0..block.shape().len())
            .map(|axis| parent.strides()[if axis < nin { nout + axis } else { axis - nin }])
            .collect();
        let dst = elements(block.shape(), block.strides(), block.offset());
        let src = elements(block.shape(), &strides, parent.offset());
        for (dst, src) in dst.into_iter().zip(src) {
            table[dst] = src;
        }
        copies.push((
            (block.shape().to_vec(), strides, parent.offset()),
            destination,
        ));
    }
    let geometry = |region: &symtn_dense::CudaRegion| {
        (
            region.dims().to_vec(),
            region.strides().to_vec(),
            region.offset(),
        )
    };
    let lowered_zeros: Vec<_> = lowering.zeros.iter().map(geometry).collect();
    assert_eq!(lowered_zeros, zeros, "zero regions");
    assert_eq!(
        zeros.len(),
        if zero_blocks {
            structure.block_count()
        } else {
            0
        }
    );
    if D::IS_COMPLEX {
        assert!(lowering.copies.is_empty(), "complex lowers to a table");
        assert_eq!(lowering.table, table, "element table");
    } else {
        assert!(lowering.table.is_empty(), "real lowers to copies");
    }
    assert!(regions.table.is_empty(), "copies only");
    let lowered: Vec<_> = regions
        .copies
        .iter()
        .map(|(source, destination)| (geometry(source), geometry(destination)))
        .collect();
    assert_eq!(lowered, copies, "strided copies");
    let lowered_zeros: Vec<_> = regions.zeros.iter().map(geometry).collect();
    assert_eq!(lowered_zeros, zeros, "zero regions without a table");

    // The lowering, run on the Host, is the Host materialization.
    let parent = t.dense_data().unwrap();
    let mut output = parent.to_vec();
    if D::IS_COMPLEX {
        for (dst, &src) in table.iter().enumerate() {
            output[dst] = parent[src].conj();
        }
    } else {
        for ((shape, source_strides, source_offset), (_, strides, offset)) in &copies {
            let dst = elements(shape, strides, *offset);
            let src = elements(shape, source_strides, *source_offset);
            for (dst, src) in dst.into_iter().zip(src) {
                output[dst] = parent[src];
            }
        }
    }
    for (shape, strides, offset) in &zeros {
        for dst in elements(shape, strides, *offset) {
            assert_ne!(output[dst], D::zero(), "the zero block is not vacuous");
            output[dst] = D::zero();
        }
    }
    let expected =
        symtn_tensors::materialize_adjoint_data_dyn(parent_space, &adjoint_space, parent).unwrap();
    assert_eq!(output, expected);
    if !zero_blocks {
        assert_eq!(output, lazy.materialize().unwrap().dense_data().unwrap());
    }
}

#[cfg(feature = "cuda")]
#[test]
fn typed_cuda_materialize_lowers_the_host_adjoint_block_map() {
    fn fill<S>(_: &crate::typed::BlockFusionTrees<S>, ix: &[usize]) -> f64 {
        ix.iter().enumerate().fold(0.37, |value, (axis, &i)| {
            value * 1.7 + (i as f64 + 1.0) * (axis as f64 + 2.0)
        })
    }
    fn complex_fill<S>(trees: &crate::typed::BlockFusionTrees<S>, ix: &[usize]) -> Complex64 {
        let re = fill(trees, ix);
        Complex64::new(re, 0.25 - 0.5 * re)
    }
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();

    // U(1) `[V, V*] <- [V]`, with and without zero blocks.
    let u1 = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(0), 2),
            (U1Irrep::new(1), 3),
            (U1Irrep::new(-1), 1),
        ],
    )
    .unwrap();
    let u1_dual = u1.try_dual().unwrap();
    let real: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&u1, &u1_dual], [&u1], fill).unwrap();
    let complex: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&u1, &u1_dual], [&u1], complex_fill).unwrap();
    for zero_blocks in [false, true] {
        assert_cuda_adjoint_lowering_matches_hand(&real, zero_blocks);
        assert_cuda_adjoint_lowering_matches_hand(&complex, zero_blocks);
    }

    // SU(2): several fusion-tree blocks per coupled sector, each a strided
    // view into its coupled-sector matrix.
    let su2 = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 3),
            (SU2Irrep::from_twice_spin(2), 1),
        ],
    )
    .unwrap();
    let real: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&su2, &su2], [&su2], fill).unwrap();
    let complex: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&su2, &su2], [&su2], complex_fill).unwrap();
    for zero_blocks in [false, true] {
        assert_cuda_adjoint_lowering_matches_hand(&real, zero_blocks);
        assert_cuda_adjoint_lowering_matches_hand(&complex, zero_blocks);
    }

    // Fermion parity with a dual leg.
    let fz2 = GradedSpace::try_new(
        Arc::new(FermionParityFusionRule),
        [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 3)],
    )
    .unwrap();
    let fz2_dual = fz2.try_dual().unwrap();
    let complex: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&fz2, &fz2_dual], [&fz2], complex_fill).unwrap();
    assert_cuda_adjoint_lowering_matches_hand(&complex, false);
    let real: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&fz2, &fz2_dual], [&fz2], fill).unwrap();
    assert_cuda_adjoint_lowering_matches_hand(&real, true);

    // A parent without the swapped pair: the map's `MissingBlockKey`, the
    // Host error, on the device lowering too (§4 L6).
    let lazy = complex.adjoint().unwrap();
    let TypedTensorRepr::Adjoint(view) = &lazy.repr else {
        panic!("dense adjoint is lazy");
    };
    let parent = with_opaque_keys(view.parent.space.space());
    let logical = view.logical_space.space();
    let host = symtn_tensors::materialize_adjoint_data_dyn(
        &parent,
        logical,
        complex.dense_data().unwrap(),
    );
    assert!(matches!(host, Err(OperationError::MissingBlockKey { .. })));
    let device = super::super::cuda_transfer::cuda_adjoint_lowering(
        &symtn_tensors::AdjointBlockMap::new(&parent, logical),
        true,
    );
    assert!(matches!(
        device,
        Err(Error::Operation(op)) if matches!(*op, OperationError::MissingBlockKey { .. })
    ));
}
