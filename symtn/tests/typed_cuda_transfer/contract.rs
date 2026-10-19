use super::*;

fn assert_direct_contract_and_compose<R>(lhs: &TensorMap<R, f64>, rhs: &TensorMap<R, f64>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
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
    let runtime = symtn::typed::__network::runtime_identity(lhs.runtime());
    let lhs_device = lhs.to_cuda().unwrap();
    let rhs_device = rhs.to_cuda().unwrap();

    let contract = lhs_device
        .contract(&rhs_device, &spec)
        .unwrap()
        .to_host()
        .unwrap();
    let compose = lhs_device.compose(&rhs_device).unwrap().to_host().unwrap();

    for (actual, expected) in [
        (&contract, &expected_contract),
        (&compose, &expected_compose),
    ] {
        assert!(std::ptr::eq(actual.provider(), provider));
        assert!(runtime.matches(actual.runtime()));
        assert_eq!(actual.dense_data().unwrap(), expected.dense_data().unwrap());
        assert_eq!(structural_snapshot(actual), structural_snapshot(expected));
    }
}

#[test]
#[ignore]
fn typed_cuda_direct_execution_matches_host_providers_and_structure() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();

    let u1_rule = Arc::new(U1FusionRule);
    let u1 = GradedSpace::try_new(
        Arc::clone(&u1_rule),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let u1_lhs = TensorMap::from_subblock_fn(&runtime, [&u1], [&u1], |_, indices| {
        indices.iter().sum::<usize>() as f64 + 1.0
    })
    .unwrap();
    let u1_rhs = TensorMap::from_subblock_fn(&runtime, [&u1], [&u1], |_, indices| {
        indices.iter().sum::<usize>() as f64 + 2.0
    })
    .unwrap();
    assert_direct_contract_and_compose(&u1_lhs, &u1_rhs);

    let su2_rule = Arc::new(SU2FusionRule);
    let su2 = GradedSpace::try_new(
        Arc::clone(&su2_rule),
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
    assert_direct_contract_and_compose(&su2_lhs, &su2_rhs);

    let product_rule = Arc::new(U1FusionRule.product(FermionParityFusionRule));
    let product = GradedSpace::try_new(
        Arc::clone(&product_rule),
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
    assert_direct_contract_and_compose(&product_lhs, &product_rhs);

    let product_dual = GradedSpace::try_new(
        Arc::clone(&product_rule),
        [
            (product_sector(U1Irrep::new(0), Z2Irrep::EVEN), 2),
            (product_sector(U1Irrep::new(1), Z2Irrep::ODD), 1),
        ],
    )
    .and_then(|space| space.try_dual())
    .unwrap();
    let product_multileg_lhs = TensorMap::from_subblock_fn(
        &runtime,
        [&product],
        [&product_dual, &product_dual],
        |_, indices| indices.iter().sum::<usize>() as f64 + 1.0,
    )
    .unwrap();
    let product_multileg_rhs = TensorMap::from_subblock_fn(
        &runtime,
        [&product_dual, &product_dual],
        [&product],
        |_, indices| indices.iter().sum::<usize>() as f64 + 2.0,
    )
    .unwrap();
    assert_direct_contract_and_compose(&product_multileg_lhs, &product_multileg_rhs);
}

#[test]
#[ignore]
fn typed_cuda_fermionic_contract_is_minus_six_and_compose_stays_plus_six() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let provider = Arc::new(FermionParityFusionRule);
    let lhs_codomain = GradedSpace::try_new(Arc::clone(&provider), [(Z2Irrep::ODD, 1)]).unwrap();
    let lhs_domain = GradedSpace::try_new(Arc::clone(&provider), [(Z2Irrep::ODD, 1)])
        .and_then(|space| space.try_dual())
        .unwrap();
    let rhs_codomain = GradedSpace::try_new(Arc::clone(&provider), [(Z2Irrep::ODD, 1)])
        .and_then(|space| space.try_dual())
        .unwrap();
    let rhs_domain = GradedSpace::try_new(Arc::clone(&provider), [(Z2Irrep::ODD, 1)]).unwrap();
    let lhs =
        TensorMap::from_subblock_fn(&runtime, [&lhs_codomain], [&lhs_domain], |_, _| 2.0).unwrap();
    let rhs =
        TensorMap::from_subblock_fn(&runtime, [&rhs_codomain], [&rhs_domain], |_, _| 3.0).unwrap();
    assert_eq!(
        lhs.contract(
            &rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1]
            }
        )
        .unwrap()
        .dense_data()
        .unwrap(),
        [-6.0]
    );
    assert_eq!(lhs.compose(&rhs).unwrap().dense_data().unwrap(), [6.0]);

    let lhs_device = lhs.to_cuda().unwrap();
    let rhs_device = rhs.to_cuda().unwrap();
    assert_eq!(
        lhs_device
            .compose(&rhs_device)
            .unwrap()
            .to_host()
            .unwrap()
            .dense_data()
            .unwrap(),
        [6.0]
    );
    assert_eq!(
        lhs_device
            .contract(
                &rhs_device,
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &[0],
                    domain: &[1]
                }
            )
            .unwrap()
            .to_host()
            .unwrap()
            .dense_data()
            .unwrap(),
        [-6.0]
    );
}

#[test]
#[ignore]
fn typed_cuda_direct_supports_canonical_lazy_and_rejects_other_scopes_before_mutation() {
    let runtime = Runtime::builder().cuda(0).build().unwrap();
    let other_runtime = Runtime::builder().cuda(0).build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0), 2)]).unwrap();
    let host = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
        indices.iter().sum::<usize>() as f64 + 1.0
    })
    .unwrap();
    let other_host =
        TensorMap::from_subblock_fn(&other_runtime, [&leg], [&leg], |_, _| 1.0).unwrap();
    let expected = host.dense_data().unwrap().to_vec();
    let device = host.to_cuda().unwrap();
    let other_device = other_host.to_cuda().unwrap();

    assert_eq!(
        device.compose(&other_device).unwrap_err(),
        symtn::typed::Error::RuntimeMismatch
    );
    // General axes are admitted (#1345): a mismatched pairing is
    // the Host's own error, and a permuted output is the Host's result.
    assert_eq!(
        device
            .contract(
                &device,
                &ContractSpec {
                    lhs: &[0],
                    rhs: &[0],
                    codomain: &[0],
                    domain: &[1]
                }
            )
            .unwrap_err()
            .to_string(),
        host.contract(
            &host,
            &ContractSpec {
                lhs: &[0],
                rhs: &[0],
                codomain: &[0],
                domain: &[1]
            }
        )
        .unwrap_err()
        .to_string()
    );
    let close = |actual: &[f64], expected: &[f64]| {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() <= 1e-12 * (1.0 + expected.abs()));
        }
    };
    close(
        device
            .contract(
                &device,
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &[1],
                    domain: &[0],
                },
            )
            .unwrap()
            .to_host()
            .unwrap()
            .dense_data()
            .unwrap(),
        host.contract(
            &host,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[1],
                domain: &[0],
            },
        )
        .unwrap()
        .dense_data()
        .unwrap(),
    );
    let lazy_host = host.adjoint().unwrap();
    let expected_lazy_compose = lazy_host.compose(&host).unwrap();
    let lazy = lazy_host.to_cuda().unwrap();
    let lazy_compose = lazy.compose(&device).unwrap().to_host().unwrap();
    assert!(std::ptr::eq(lazy_compose.provider(), host.provider()));
    assert!(symtn::typed::__network::runtime_identity(&runtime).matches(lazy_compose.runtime()));
    assert_eq!(
        lazy_compose.dense_data().unwrap(),
        expected_lazy_compose.dense_data().unwrap()
    );
    assert_eq!(
        structural_snapshot(&lazy_compose),
        structural_snapshot(&expected_lazy_compose)
    );
    let lazy_general = lazy
        .contract(
            &device,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[1],
                domain: &[0],
            },
        )
        .unwrap();
    let expected_lazy_general = lazy_host
        .contract(
            &host,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[1],
                domain: &[0],
            },
        )
        .unwrap();
    close(
        lazy_general.to_host().unwrap().dense_data().unwrap(),
        expected_lazy_general.dense_data().unwrap(),
    );
    assert_eq!(device.to_host().unwrap().dense_data().unwrap(), expected);

    let zn3 = Arc::new(ZNFusionRule::new(3).unwrap());
    let zn4 = Arc::new(ZNFusionRule::new(4).unwrap());
    let zn3_leg = GradedSpace::try_new(Arc::clone(&zn3), [(zn3.irrep(0), 1)]).unwrap();
    let zn4_leg = GradedSpace::try_new(Arc::clone(&zn4), [(zn4.irrep(0), 1)]).unwrap();
    let zn3_tensor: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&zn3_leg], [&zn3_leg], |_, _| 1.0).unwrap();
    let zn4_tensor: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&zn4_leg], [&zn4_leg], |_, _| 1.0).unwrap();
    let zn3_device = zn3_tensor.to_cuda().unwrap();
    let zn4_device = zn4_tensor.to_cuda().unwrap();
    assert!(zn3_device.compose(&zn4_device).is_err());
    assert_eq!(zn3_device.to_host().unwrap().dense_data().unwrap(), [1.0]);
    assert_eq!(zn4_device.to_host().unwrap().dense_data().unwrap(), [1.0]);

    let left_open = GradedSpace::try_new(Arc::clone(&zn3), [(zn3.irrep(0), 1)]).unwrap();
    let seam = GradedSpace::try_new(Arc::clone(&zn3), [(zn3.irrep(1), 1)]).unwrap();
    let right_open = GradedSpace::try_new(Arc::clone(&zn3), [(zn3.irrep(2), 1)]).unwrap();
    let zero_lhs: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&left_open], [&seam], |_, _| 1.0).unwrap();
    let zero_rhs: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&seam], [&right_open], |_, _| 1.0).unwrap();
    assert_eq!(zero_lhs.subblock_count(), 0);
    assert_eq!(zero_rhs.subblock_count(), 0);
    let zero_output = zero_lhs
        .to_cuda()
        .unwrap()
        .compose(&zero_rhs.to_cuda().unwrap())
        .unwrap()
        .to_host()
        .unwrap();
    assert_eq!(zero_output.subblock_count(), 0);
    assert!(zero_output.dense_data().unwrap().is_empty());
}

fn assert_c64_contract_and_compose<R>(lhs: &TensorMap<R, Complex64>, rhs: &TensorMap<R, Complex64>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
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
    let lhs_device = lhs.to_cuda().unwrap();
    let rhs_device = rhs.to_cuda().unwrap();

    let contract = lhs_device
        .contract(&rhs_device, &spec)
        .unwrap()
        .to_host()
        .unwrap();
    let compose = lhs_device.compose(&rhs_device).unwrap().to_host().unwrap();
    for (actual, expected) in [
        (&contract, &expected_contract),
        (&compose, &expected_compose),
    ] {
        // Why not `numerics`: these entries reach magnitude 271 next to exact
        // zeros, where the rule's slice-wide bound would exceed the per-entry
        // `1e-12 * (1 + |want|)` this gate holds.
        let (got, want) = (actual.dense_data().unwrap(), expected.dense_data().unwrap());
        assert_eq!(got.len(), want.len());
        for (got, want) in got.iter().zip(want) {
            assert!(
                (got - want).norm() <= 1e-12 * (1.0 + want.norm()),
                "{got} vs {want}"
            );
        }
        assert_eq!(structural_snapshot(actual), structural_snapshot(expected));
    }

    // Lazy conjugate-transpose operands on either side must route the
    // conjugation through the GEMM flag, never a materialized buffer.
    // `A^H . A` and `B . B^H` are composable for every split.
    let host_left = lhs.adjoint().unwrap().compose(lhs).unwrap();
    let device_left = lhs_device
        .adjoint()
        .unwrap()
        .compose(&lhs_device)
        .unwrap()
        .to_host()
        .unwrap();
    numerics::assert_slices_close(
        "device_left",
        device_left.dense_data().unwrap(),
        host_left.dense_data().unwrap(),
        CONTRACT_TERMS,
    );
    let host_right = rhs.compose(&rhs.adjoint().unwrap()).unwrap();
    let device_right = rhs_device
        .compose(&rhs_device.adjoint().unwrap())
        .unwrap()
        .to_host()
        .unwrap();
    numerics::assert_slices_close(
        "device_right",
        device_right.dense_data().unwrap(),
        host_right.dense_data().unwrap(),
        CONTRACT_TERMS,
    );
}

#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_c64_contract_and_compose_match_host() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();

    let u1 = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    assert_c64_contract_and_compose(
        &TensorMap::<_, Complex64>::from_subblock_fn(&runtime, [&u1], [&u1], |_, indices| {
            complex_entry(indices, 1.0)
        })
        .unwrap(),
        &TensorMap::<_, Complex64>::from_subblock_fn(&runtime, [&u1], [&u1], |_, indices| {
            complex_entry(indices, 2.0)
        })
        .unwrap(),
    );

    let su2 = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 1),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap();
    assert_c64_contract_and_compose(
        &TensorMap::<_, Complex64>::from_subblock_fn(
            &runtime,
            [&su2, &su2, &su2],
            [&su2, &su2],
            |_, indices| complex_entry(indices, 1.0),
        )
        .unwrap(),
        &TensorMap::<_, Complex64>::from_subblock_fn(
            &runtime,
            [&su2, &su2],
            [&su2, &su2, &su2],
            |_, indices| complex_entry(indices, 3.0),
        )
        .unwrap(),
    );

    // Fermionic provider: contract carries the twist, compose does not, so
    // the two results stay distinct for complex payloads too.
    let fz2 = GradedSpace::try_new(
        Arc::new(FermionParityFusionRule),
        [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 1)],
    )
    .unwrap();
    let fz2_dual = GradedSpace::try_new(
        Arc::new(FermionParityFusionRule),
        [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 1)],
    )
    .and_then(|space| space.try_dual())
    .unwrap();
    assert_c64_contract_and_compose(
        &TensorMap::<_, Complex64>::from_subblock_fn(
            &runtime,
            [&fz2],
            [&fz2_dual, &fz2_dual],
            |_, indices| complex_entry(indices, 1.0),
        )
        .unwrap(),
        &TensorMap::<_, Complex64>::from_subblock_fn(
            &runtime,
            [&fz2_dual, &fz2_dual],
            [&fz2],
            |_, indices| complex_entry(indices, 2.0),
        )
        .unwrap(),
    );

    // General axes are admitted (#1345): these scopes agree with the Host.
    let host_lhs =
        TensorMap::<_, Complex64>::from_subblock_fn(&runtime, [&u1], [&u1], |_, indices| {
            complex_entry(indices, 5.0)
        })
        .unwrap();
    let host_rhs =
        TensorMap::<_, Complex64>::from_subblock_fn(&runtime, [&u1], [&u1], |_, indices| {
            complex_entry(indices, 6.0)
        })
        .unwrap();
    let lhs = host_lhs.to_cuda().unwrap();
    let rhs = host_rhs.to_cuda().unwrap();
    for spec in [
        ContractSpec {
            lhs: &[0],
            rhs: &[1],
            codomain: &[0],
            domain: &[1],
        },
        ContractSpec {
            lhs: &[1],
            rhs: &[0],
            codomain: &[1],
            domain: &[0],
        },
    ] {
        let expected = host_lhs.contract(&host_rhs, &spec).unwrap();
        let actual = lhs.contract(&rhs, &spec).unwrap().to_host().unwrap();
        assert_eq!(
            actual.dense_data().unwrap().len(),
            expected.dense_data().unwrap().len()
        );
        for (actual, expected) in actual
            .dense_data()
            .unwrap()
            .iter()
            .zip(expected.dense_data().unwrap())
        {
            assert!((actual - expected).norm() <= 1e-12 * (1.0 + expected.norm()));
        }
    }
}

/// Independent oracle for the conjugation flag itself.
///
/// Host and device both lower operand conjugation to the same Tenferro
/// `DotGeneralAccumulation` flags, so a Host-vs-device comparison alone cannot
/// catch a shared misreading of those flags. This computes `A^H . B` for a
/// one-sector U(1) fixture by explicit conjugate-transpose loops over the
/// reduced block and compares both the Host and the device result against it.
#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_c64_lazy_adjoint_contract_matches_a_hand_expansion() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let u1 = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let a = TensorMap::<_, Complex64>::from_subblock_fn(&runtime, [&u1], [&u1], |_, indices| {
        complex_entry(indices, 1.0) * Complex64::new(1.0, indices[1] as f64 + 1.0)
    })
    .unwrap();
    let b = TensorMap::<_, Complex64>::from_subblock_fn(&runtime, [&u1], [&u1], |_, indices| {
        complex_entry(indices, 3.0) * Complex64::new(indices[0] as f64 + 1.0, -1.0)
    })
    .unwrap();
    assert_eq!(a.subblock_count(), 1);

    fn reduced_block(
        tensor: &TensorMap<U1FusionRule, Complex64>,
    ) -> impl Fn(usize, usize) -> Complex64 + use<> {
        let block = tensor.subblock(0).unwrap();
        assert_eq!(block.shape(), [2, 2]);
        let offset = block.offset();
        let strides = block.strides().to_vec();
        let data = tensor.dense_data().unwrap().to_vec();
        move |row: usize, col: usize| data[offset + row * strides[0] + col * strides[1]]
    }
    let a_block = reduced_block(&a);
    let b_block = reduced_block(&b);

    // (A^H B)[i, j] = sum_k conj(A[k, i]) * B[k, j].
    let mut expected = [[Complex64::new(0.0, 0.0); 2]; 2];
    for (i, row) in expected.iter_mut().enumerate() {
        for (j, entry) in row.iter_mut().enumerate() {
            for k in 0..2 {
                *entry += a_block(k, i).conj() * b_block(k, j);
            }
        }
    }

    let a_device = a.to_cuda().unwrap();
    let b_device = b.to_cuda().unwrap();
    // Both device entry points: `compose` (twist-free compiler) and `contract`
    // (which lowers the lazy operand through `TensorContractSpec::
    // new_with_conjugation`). U(1) is bosonic, so both must equal the same
    // conjugate-transposed product.
    let results = [
        (
            "compose",
            a.adjoint().unwrap().compose(&b).unwrap(),
            a_device
                .adjoint()
                .unwrap()
                .compose(&b_device)
                .unwrap()
                .to_host()
                .unwrap(),
        ),
        (
            "contract",
            a.adjoint()
                .unwrap()
                .contract(
                    &b,
                    &ContractSpec {
                        lhs: &[1],
                        rhs: &[0],
                        codomain: &[0],
                        domain: &[1],
                    },
                )
                .unwrap(),
            a_device
                .adjoint()
                .unwrap()
                .contract(
                    &b_device,
                    &ContractSpec {
                        lhs: &[1],
                        rhs: &[0],
                        codomain: &[0],
                        domain: &[1],
                    },
                )
                .unwrap()
                .to_host()
                .unwrap(),
        ),
    ];
    for (label, host, device) in &results {
        let result_block = host.subblock(0).unwrap();
        let (offset, strides) = (result_block.offset(), result_block.strides().to_vec());
        for (i, row) in expected.iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                let index = offset + i * strides[0] + j * strides[1];
                assert!(
                    (host.dense_data().unwrap()[index] - value).norm()
                        <= 1e-12 * (1.0 + value.norm()),
                    "host {label} {:?} != hand {value:?}",
                    host.dense_data().unwrap()[index]
                );
                assert!(
                    (device.dense_data().unwrap()[index] - value).norm()
                        <= 1e-12 * (1.0 + value.norm()),
                    "device {label} {:?} != hand {value:?}",
                    device.dense_data().unwrap()[index]
                );
            }
        }
    }
    assert!(
        expected.iter().flatten().any(|value| value.im != 0.0),
        "the oracle must distinguish conjugation from transposition"
    );
}

/// The device destination-overwrite entry writes exactly what
/// the returning device contraction returns, including the `+0.0` of every
/// destination block no GEMM reaches, and every rejection leaves the
/// destination's bytes untouched (#1276).
#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_contract_into_matches_the_returning_contraction() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    // `k` carries only charge 0, so the destination's charge-1 block has no
    // contributing GEMM and must come out as exactly `+0.0` from the zeroing
    // of the plan's inactive blocks.
    let outer = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 3)],
    )
    .unwrap();
    let inner = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let lhs = TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&outer], [&inner], 761_000)
        .unwrap()
        .to_cuda()
        .unwrap();
    let rhs = TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&inner], [&outer], 761_001)
        .unwrap()
        .to_cuda()
        .unwrap();

    let expected = lhs
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
    let expected_data = expected.to_host().unwrap().dense_data().unwrap().to_vec();
    assert!(
        expected_data.contains(&0.0),
        "fixture must contain a destination block no GEMM reaches"
    );

    let poisoned = || {
        TensorMap::<U1FusionRule, f64>::from_subblock_fn(&runtime, [&outer], [&outer], |_, _| 7.5)
            .unwrap()
            .to_cuda()
            .unwrap()
    };
    let mut destination = poisoned();
    lhs.contract_into(
        &rhs,
        &ContractSpec {
            lhs: &[1],
            rhs: &[0],
            codomain: &[0],
            domain: &[1],
        },
        &mut destination,
        1.0,
        0.0,
    )
    .unwrap();
    let written = destination
        .to_host()
        .unwrap()
        .dense_data()
        .unwrap()
        .to_vec();
    assert_eq!(written.len(), expected_data.len());
    // Reached entries are two-term GEMM sums that the two entry points may
    // accumulate differently: within the tolerance rule. Unreached entries
    // are the zero fill, which is exact.
    numerics::assert_slices_close(
        "contract_into against contract",
        &written,
        &expected_data,
        2,
    );
    for (index, (&actual, &expected)) in written.iter().zip(&expected_data).enumerate() {
        if expected == 0.0 {
            assert_eq!(
                actual.to_bits(),
                0,
                "unreachable element {index} must be +0.0"
            );
        }
    }

    // Every rejection is refused before a device write: the poisoned bytes
    // survive each one unchanged.
    let poison_data = poisoned().to_host().unwrap().dense_data().unwrap().to_vec();
    let assert_rejected =
        |label: &str,
         result: Result<(), symtn::typed::Error>,
         dst: &TensorMap<U1FusionRule, f64, CudaStorage>| {
            assert!(result.is_err(), "{label} must be rejected");
            assert_eq!(
                dst.to_host().unwrap().dense_data().unwrap(),
                poison_data.as_slice(),
                "{label} wrote to the destination before rejecting"
            );
        };

    // Alias: the destination shares the lhs payload body, directly and through
    // a lazy adjoint view of the same parent.
    let lhs_data = lhs.to_host().unwrap().dense_data().unwrap().to_vec();
    let mut lhs_alias = lhs.clone();
    assert!(
        lhs.contract_into(
            &rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1]
            },
            &mut lhs_alias,
            1.0,
            0.0,
        )
        .is_err(),
        "an lhs alias must be rejected"
    );
    assert_eq!(
        lhs_alias.to_host().unwrap().dense_data().unwrap(),
        lhs_data.as_slice()
    );
    let mut adjoint_alias = lhs.clone();
    assert!(lhs
        .adjoint()
        .unwrap()
        .contract_into(
            &rhs,
            &ContractSpec {
                lhs: &[0],
                rhs: &[0],
                codomain: &[0],
                domain: &[1]
            },
            &mut adjoint_alias,
            1.0,
            0.0,
        )
        .is_err());
    assert_eq!(
        adjoint_alias.to_host().unwrap().dense_data().unwrap(),
        lhs_data.as_slice()
    );

    // Same required length, different block charges: the space check must
    // catch what the length check cannot.
    let drifted = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 2), (U1Irrep::new(4), 3)],
    )
    .unwrap();
    let mut wrong_space = TensorMap::<U1FusionRule, f64>::from_subblock_fn(
        &runtime,
        [&drifted],
        [&drifted],
        |_, _| 7.5,
    )
    .unwrap()
    .to_cuda()
    .unwrap();
    let wrong_space_before = wrong_space
        .to_host()
        .unwrap()
        .dense_data()
        .unwrap()
        .to_vec();
    assert_eq!(
        wrong_space_before.len(),
        expected_data.len(),
        "the drifted destination must have the same required length"
    );
    assert!(
        lhs.contract_into(
            &rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1]
            },
            &mut wrong_space,
            1.0,
            0.0,
        )
        .is_err(),
        "a destination whose block layout differs must be rejected"
    );
    assert_eq!(
        wrong_space.to_host().unwrap().dense_data().unwrap(),
        wrong_space_before.as_slice()
    );

    let mut destination = poisoned();
    let shared = destination.clone();
    assert_rejected(
        "shared ownership",
        lhs.contract_into(
            &rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
            &mut destination,
            1.0,
            0.0,
        ),
        &destination,
    );
    drop(shared);
    // Every alpha and beta is supported (#1550): `2 * contract + 0.5 * dst`,
    // with the unreached block scaled rather than cleared.
    lhs.contract_into(
        &rhs,
        &ContractSpec {
            lhs: &[1],
            rhs: &[0],
            codomain: &[0],
            domain: &[1],
        },
        &mut destination,
        2.0,
        0.5,
    )
    .unwrap();
    let accumulated: Vec<f64> = expected_data
        .iter()
        .zip(&poison_data)
        .map(|(&value, &prior)| 2.0 * value + 0.5 * prior)
        .collect();
    numerics::assert_slices_close(
        "contract_into alpha = 2, beta = 0.5",
        destination.to_host().unwrap().dense_data().unwrap(),
        &accumulated,
        3,
    );

    // A destination whose blocks all exist but none of which any GEMM reaches:
    // the inactive-block zeroing alone produces the whole result.
    let disjoint = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(7), 2)]).unwrap();
    let empty_lhs =
        TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&outer], [&disjoint], 761_002)
            .unwrap()
            .to_cuda()
            .unwrap();
    let empty_rhs =
        TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&disjoint], [&outer], 761_003)
            .unwrap()
            .to_cuda()
            .unwrap();
    let empty_expected = empty_lhs
        .contract(
            &empty_rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    let mut empty_destination = poisoned();
    empty_lhs
        .contract_into(
            &empty_rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
            &mut empty_destination,
            1.0,
            0.0,
        )
        .unwrap();
    assert_eq!(
        empty_destination.to_host().unwrap().dense_data().unwrap(),
        empty_expected.to_host().unwrap().dense_data().unwrap()
    );

    // A destination space with no coupled sector at all: `required_len` is 0,
    // so the replay writes nothing.
    let single = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let zero_lhs =
        TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&single], [&single], 761_004)
            .unwrap()
            .to_cuda()
            .unwrap();
    let zero_rhs =
        TensorMap::<U1FusionRule, f64>::rand_with_seed(&runtime, [&single], [&disjoint], 761_005)
            .unwrap()
            .to_cuda()
            .unwrap();
    let mut zero_destination = TensorMap::<U1FusionRule, f64>::from_subblock_fn(
        &runtime,
        [&single],
        [&disjoint],
        |_, _| 7.5,
    )
    .unwrap()
    .to_cuda()
    .unwrap();
    assert!(
        zero_destination
            .to_host()
            .unwrap()
            .dense_data()
            .unwrap()
            .is_empty(),
        "the fixture must have a zero-length destination"
    );
    zero_lhs
        .contract_into(
            &zero_rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
            &mut zero_destination,
            1.0,
            0.0,
        )
        .unwrap();
    assert_eq!(
        zero_destination.to_host().unwrap().dense_data().unwrap(),
        zero_lhs
            .contract(
                &zero_rhs,
                &ContractSpec {
                    lhs: &[1],
                    rhs: &[0],
                    codomain: &[0],
                    domain: &[1]
                }
            )
            .unwrap()
            .to_host()
            .unwrap()
            .dense_data()
            .unwrap()
    );
}
