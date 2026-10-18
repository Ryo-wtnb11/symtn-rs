use super::*;

/// [`HostSvdTrunc`] for the Hermitian eigendecomposition.
struct HostEighTrunc<R: SectorCodec, D> {
    d: TensorMap<R, D>,
    v: TensorMap<R, D>,
    eigenvalues: Vec<symtn::typed::SectorSpectrum<R::Sector, f64>>,
    error: f64,
}

fn host_eigh_trunc<R, D>(
    source: &TensorMap<R, D>,
    truncation: &Truncation,
    to_f64: impl Fn(D) -> f64,
) -> HostEighTrunc<R, D>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
    D: symtn::typed::FactorizationScalar + symtn::typed::SpectrumMagnitude,
{
    let Eigh { d, v } = source
        .eigh_full(
            &codomain_axes(source),
            &domain_axes(source),
            HermitianTol::DEFAULT,
        )
        .unwrap();
    let found = d.domain()[0]
        .find_truncated(&d.diagview().unwrap(), truncation)
        .unwrap();
    let d = d
        .restrict_leg(&[(0, &found.selection), (1, &found.selection)])
        .unwrap();
    HostEighTrunc {
        v: v.restrict_leg(&[(v.codomain_rank(), &found.selection)])
            .unwrap(),
        eigenvalues: labelled_f64(&d, to_f64),
        d,
        error: found.error,
    }
}

fn assert_reduction_parity<R>(lhs: &TensorMap<R, f64>, rhs: &TensorMap<R, f64>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    let expected_inner = lhs.inner(rhs).unwrap();
    let expected_norm = lhs.norm(2.0).unwrap();
    let lhs_device = lhs.to_cuda().unwrap();
    let rhs_device = rhs.to_cuda().unwrap();
    let inner = lhs_device.inner(&rhs_device).unwrap();
    let norm = lhs_device.norm(2.0).unwrap();
    let tolerance = 1e-12 * (1.0 + expected_inner.abs().max(expected_norm));

    assert!((inner - expected_inner).abs() <= tolerance);
    assert!((norm - expected_norm).abs() <= tolerance);
    let self_inner = lhs_device.inner(&lhs_device).unwrap();
    assert!((self_inner - norm.powi(2)).abs() <= 1e-12 * (1.0 + self_inner.abs()));
}

fn assert_typed_cuda_qr_matches_host<R>(source: &TensorMap<R, f64>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    let provider = source.provider() as *const R;
    let runtime = symtn::typed::__network::runtime_identity(source.runtime());
    let source_bits: Vec<_> = source
        .dense_data()
        .unwrap()
        .iter()
        .map(|value| value.to_bits())
        .collect();
    let Qr {
        q: expected_left,
        r: expected_right,
    } = source
        .qr_compact(&codomain_axes(source), &domain_axes(source))
        .unwrap();
    let source_device = source.to_cuda().unwrap();
    let Qr {
        q: left_device,
        r: right_device,
    } = source_device
        .qr_compact(&codomain_axes(&source_device), &domain_axes(&source_device))
        .unwrap();

    for factor in [&left_device, &right_device] {
        assert!(std::ptr::eq(factor.provider(), provider));
        assert!(runtime.matches(factor.runtime()));
        assert_eq!(factor.placement(), symtn::expert::Placement::Cuda(0));
    }
    let left = left_device.to_host().unwrap();
    let right = right_device.to_host().unwrap();
    numerics::assert_slices_close(
        "left",
        left.dense_data().unwrap(),
        expected_left.dense_data().unwrap(),
        FACTOR_TERMS,
    );
    numerics::assert_slices_close(
        "right",
        right.dense_data().unwrap(),
        expected_right.dense_data().unwrap(),
        FACTOR_TERMS,
    );
    assert_eq!(
        structural_snapshot(&left),
        structural_snapshot(&expected_left)
    );
    assert_eq!(
        structural_snapshot(&right),
        structural_snapshot(&expected_right)
    );
    let rebuilt = left.compose(&right).unwrap();
    numerics::assert_slices_close(
        "rebuilt",
        rebuilt.dense_data().unwrap(),
        source.dense_data().unwrap(),
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

fn assert_finite_r_diagonal_nonnegative<R>(right: &TensorMap<R, f64>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    for block_index in 0..right.subblock_count() {
        let block = right.subblock(block_index).unwrap();
        let diagonal_len = block.shape()[0].min(block.shape()[1]);
        for index in 0..diagonal_len {
            let value = right.dense_data().unwrap()
                [block.offset() + index * block.strides()[0] + index * block.strides()[1]];
            if value.is_finite() {
                assert!(value >= 0.0, "negative finite R diagonal {value:?}");
            }
        }
    }
}

#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_reductions_cover_weights_providers_lazy_and_preflight() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();

    let u1_provider = Arc::new(U1FusionRule);
    let u1_leg = GradedSpace::try_new(
        Arc::clone(&u1_provider),
        [(U1Irrep::new(0), 1), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let u1_lhs: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&u1_leg], [&u1_leg], |trees, _| {
            if trees.coupled() == &U1Irrep::new(0) {
                1.0
            } else {
                2.0
            }
        })
        .unwrap();
    let u1_rhs: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&u1_leg], [&u1_leg], |trees, _| {
            if trees.coupled() == &U1Irrep::new(0) {
                3.0
            } else {
                4.0
            }
        })
        .unwrap();
    assert_eq!(u1_lhs.inner(&u1_rhs).unwrap(), 11.0);
    assert_reduction_parity(&u1_lhs, &u1_rhs);

    let u1_device = u1_lhs.to_cuda().unwrap();
    let u1_norm = u1_device.norm(2.0).unwrap();
    let lazy = u1_device.adjoint().unwrap();
    assert!((lazy.norm(2.0).unwrap() - u1_norm).abs() < 1e-12);
    assert!(matches!(
        lazy.inner(&lazy),
        Err(symtn::typed::Error::UnsupportedOnDevice(_))
    ));
    let u1_rhs_device = u1_rhs.to_cuda().unwrap();
    let repeated = u1_device.inner(&u1_rhs_device).unwrap();
    for _ in 0..3 {
        assert_eq!(u1_device.inner(&u1_rhs_device).unwrap(), repeated);
    }
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..4)
            .map(|_| scope.spawn(|| u1_device.inner(&u1_rhs_device).unwrap()))
            .collect();
        for worker in workers {
            assert_eq!(worker.join().unwrap(), repeated);
        }
    });

    let su2_provider = Arc::new(SU2FusionRule);
    let spin0 = SU2Irrep::from_twice_spin(0);
    let spin_half = SU2Irrep::from_twice_spin(1);
    let su2_leg =
        GradedSpace::try_new(Arc::clone(&su2_provider), [(spin0, 1), (spin_half, 1)]).unwrap();
    let su2_lhs: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&su2_leg], [&su2_leg], |trees, _| {
            if trees.coupled() == &spin0 {
                1.0
            } else {
                2.0
            }
        })
        .unwrap();
    let su2_rhs: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&su2_leg], [&su2_leg], |trees, _| {
            if trees.coupled() == &spin0 {
                3.0
            } else {
                4.0
            }
        })
        .unwrap();
    // dim(j=0) * 1 * 3 + dim(j=1/2) * 2 * 4 = 1 * 3 + 2 * 8.
    assert_eq!(su2_lhs.inner(&su2_rhs).unwrap(), 19.0);
    assert_reduction_parity(&su2_lhs, &su2_rhs);

    let su2_rank5_lhs = TensorMap::from_subblock_fn(
        &runtime,
        [&su2_leg, &su2_leg, &su2_leg],
        [&su2_leg, &su2_leg],
        |_, indices| indices.iter().sum::<usize>() as f64 + 1.0,
    )
    .unwrap();
    let su2_rank5_rhs = TensorMap::from_subblock_fn(
        &runtime,
        [&su2_leg, &su2_leg, &su2_leg],
        [&su2_leg, &su2_leg],
        |_, indices| indices.iter().sum::<usize>() as f64 + 3.0,
    )
    .unwrap();
    assert_reduction_parity(&su2_rank5_lhs, &su2_rank5_rhs);

    let fz2_provider = Arc::new(FermionParityFusionRule);
    let fz2_leg = GradedSpace::try_new(
        Arc::clone(&fz2_provider),
        [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 1)],
    )
    .unwrap();
    let fz2_lhs = TensorMap::from_subblock_fn(&runtime, [&fz2_leg], [&fz2_leg], |_, indices| {
        indices.iter().sum::<usize>() as f64 + 1.0
    })
    .unwrap();
    let fz2_rhs = TensorMap::from_subblock_fn(&runtime, [&fz2_leg], [&fz2_leg], |_, indices| {
        indices.iter().sum::<usize>() as f64 + 3.0
    })
    .unwrap();
    assert_reduction_parity(&fz2_lhs, &fz2_rhs);

    let product_provider = Arc::new(U1FusionRule.product(FermionParityFusionRule));
    let product_leg = GradedSpace::try_new(
        Arc::clone(&product_provider),
        [
            (product_sector(U1Irrep::new(0), Z2Irrep::EVEN), 2),
            (product_sector(U1Irrep::new(1), Z2Irrep::ODD), 1),
        ],
    )
    .unwrap();
    let product_lhs =
        TensorMap::from_subblock_fn(&runtime, [&product_leg], [&product_leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 2.0
        })
        .unwrap();
    let product_rhs =
        TensorMap::from_subblock_fn(&runtime, [&product_leg], [&product_leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 5.0
        })
        .unwrap();
    assert_reduction_parity(&product_lhs, &product_rhs);

    let callback_count = Arc::new(AtomicUsize::new(0));
    let probe_provider = Arc::new(ReentrantDimensionRule {
        runtime: runtime.clone(),
        calls: Arc::clone(&callback_count),
    });
    let probe_leg = GradedSpace::try_new(Arc::clone(&probe_provider), [(ProbeSector, 1)]).unwrap();
    let probe_lhs: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&probe_leg], [&probe_leg], |_, _| 2.0).unwrap();
    let probe_rhs: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&probe_leg], [&probe_leg], |_, _| 3.0).unwrap();
    let calls_before_reduction = callback_count.load(Ordering::SeqCst);
    assert_eq!(
        probe_lhs
            .to_cuda()
            .unwrap()
            .inner(&probe_rhs.to_cuda().unwrap())
            .unwrap(),
        6.0
    );
    assert!(callback_count.load(Ordering::SeqCst) > calls_before_reduction);

    // Same canary for a device structural transform (#1322). This provider's
    // `dim_scalar` re-enters `lease_cuda`, so any provider callback made while
    // the device lease is held deadlocks on a non-re-entrant mutex. The
    // transform's compile needs F and R symbols but no dimension for this
    // one-sector rule, so the counter is not asserted to move — what is
    // guarded is that a future edit which does need a dimension cannot make
    // that call under the lease.
    let probe_device = probe_lhs.to_cuda().unwrap();
    assert_eq!(
        probe_device
            .permute(&[1], &[0])
            .unwrap()
            .to_host()
            .unwrap()
            .dense_data()
            .unwrap(),
        probe_lhs.permute(&[1], &[0]).unwrap().dense_data().unwrap()
    );

    let other_runtime = Runtime::builder().cuda(0).build().unwrap();
    let foreign = TensorMap::from_subblock_fn(&other_runtime, [&u1_leg], [&u1_leg], |_, _| 1.0)
        .unwrap()
        .to_cuda()
        .unwrap();
    assert_eq!(
        u1_device.inner(&foreign).unwrap_err(),
        symtn::typed::Error::RuntimeMismatch
    );

    let wider = GradedSpace::try_new(
        Arc::clone(&u1_provider),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let mismatched = TensorMap::from_subblock_fn(&runtime, [&wider], [&wider], |_, _| 1.0)
        .unwrap()
        .to_cuda()
        .unwrap();
    assert!(matches!(
        u1_device.inner(&mismatched),
        Err(symtn::typed::Error::Operation(operation))
            if matches!(*operation, symtn::typed::OperationError::SpaceMismatch { .. })
    ));
    assert_eq!(
        u1_device.to_host().unwrap().dense_data().unwrap(),
        u1_lhs.dense_data().unwrap()
    );

    let zn3 = Arc::new(ZNFusionRule::new(3).unwrap());
    let charge0 = GradedSpace::try_new(Arc::clone(&zn3), [(zn3.irrep(0), 1)]).unwrap();
    let charge1 = GradedSpace::try_new(Arc::clone(&zn3), [(zn3.irrep(1), 1)]).unwrap();
    let empty: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&charge0], [&charge1], |_, _| 1.0).unwrap();
    assert!(empty.dense_data().unwrap().is_empty());
    let empty_device = empty.to_cuda().unwrap();
    assert_eq!(empty_device.norm(2.0).unwrap(), 0.0);
    assert_eq!(empty_device.inner(&empty_device).unwrap(), 0.0);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_qr_compact_streams_multiplicity_free_f64_factors() {
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
    let mixed = TensorMap::from_subblock_fn(&runtime, [&tall], [&wide], |_, indices| {
        if indices[0] == indices[1] {
            6.0 + indices[0] as f64
        } else {
            (1 + indices[0] + 2 * indices[1]) as f64
        }
    })
    .unwrap();
    assert_typed_cuda_qr_matches_host(&mixed);

    let charge_zero_only = GradedSpace::try_new(Arc::clone(&u1), [(U1Irrep::new(0), 2)]).unwrap();
    let unmatched =
        TensorMap::from_subblock_fn(&runtime, [&tall], [&charge_zero_only], |_, indices| {
            if indices[0] == indices[1] {
                4.0 + indices[0] as f64
            } else {
                1.0
            }
        })
        .unwrap();
    assert_typed_cuda_qr_matches_host(&unmatched);

    let square = TensorMap::from_subblock_fn(&runtime, [&wide], [&wide], |_, indices| {
        if indices[0] == indices[1] {
            8.0 + indices[0] as f64
        } else {
            (1 + indices[0] + indices[1]) as f64
        }
    })
    .unwrap();
    assert_typed_cuda_qr_matches_host(&square);

    let su2 = Arc::new(SU2FusionRule);
    let su2_leg = GradedSpace::try_new(
        Arc::clone(&su2),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap();
    let su2_tensor = TensorMap::from_subblock_fn(&runtime, [&su2_leg], [&su2_leg], |_, indices| {
        if indices[0] == indices[1] {
            7.0 + indices[0] as f64
        } else {
            1.0
        }
    })
    .unwrap();
    assert_typed_cuda_qr_matches_host(&su2_tensor);

    let product = Arc::new(U1FusionRule.product(FermionParityFusionRule));
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
            if indices[0] == indices[1] {
                5.0 + indices[0] as f64
            } else {
                1.0
            }
        })
        .unwrap();
    assert_typed_cuda_qr_matches_host(&product_tensor);

    let multi_tree =
        TensorMap::from_subblock_fn(&runtime, [&su2_leg, &su2_leg], [&su2_leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();
    let Qr {
        q: expected_multi_left,
        r: expected_multi_right,
    } = multi_tree
        .qr_compact(&codomain_axes(&multi_tree), &domain_axes(&multi_tree))
        .unwrap();
    let multi_tree_device = multi_tree.to_cuda().unwrap();
    let Qr { q: left, r: right } = multi_tree_device
        .qr_compact(
            &codomain_axes(&multi_tree_device),
            &domain_axes(&multi_tree_device),
        )
        .unwrap();
    let left = left.to_host().unwrap();
    let right = right.to_host().unwrap();
    assert_eq!(
        structural_snapshot(&left),
        structural_snapshot(&expected_multi_left)
    );
    assert_eq!(
        structural_snapshot(&right),
        structural_snapshot(&expected_multi_right)
    );
    let rebuilt = left.compose(&right).unwrap();
    numerics::assert_slices_close(
        "rebuilt",
        rebuilt.dense_data().unwrap(),
        multi_tree.dense_data().unwrap(),
        FACTOR_TERMS,
    );

    let zero = TensorMap::from_subblock_fn(&runtime, [&wide], [&wide], |_, _| 0.0).unwrap();
    let zero_device = zero.to_cuda().unwrap();
    let Qr {
        q: zero_left,
        r: zero_right,
    } = zero_device
        .qr_compact(&codomain_axes(&zero_device), &domain_axes(&zero_device))
        .unwrap();
    let zero_left = zero_left.to_host().unwrap();
    let zero_right = zero_right.to_host().unwrap();
    let zero_rebuilt = zero_left.compose(&zero_right).unwrap();
    numerics::assert_slices_close(
        "zero_rebuilt",
        zero_rebuilt.dense_data().unwrap(),
        zero.dense_data().unwrap(),
        FACTOR_TERMS,
    );
    assert_finite_r_diagonal_nonnegative(&zero_right);

    let rank_deficient_leg = GradedSpace::try_new(Arc::clone(&u1), [(U1Irrep::new(0), 3)]).unwrap();
    let rank_deficient = TensorMap::from_subblock_fn(
        &runtime,
        [&rank_deficient_leg],
        [&rank_deficient_leg],
        |_, indices| (indices[0] + 1) as f64 * (indices[1] + 1) as f64,
    )
    .unwrap();
    let Qr {
        q: rank_left,
        r: rank_right,
    } = rank_deficient
        .to_cuda()
        .unwrap()
        .qr_compact(
            &codomain_axes(&rank_deficient),
            &domain_axes(&rank_deficient),
        )
        .unwrap();
    let rank_left = rank_left.to_host().unwrap();
    let rank_right = rank_right.to_host().unwrap();
    assert!(is_isometric!(rank_left, 1e-10));
    numerics::assert_slices_close(
        "rank_left .compose(rank_right)",
        rank_left
            .compose(&rank_right)
            .unwrap()
            .dense_data()
            .unwrap(),
        rank_deficient.dense_data().unwrap(),
        FACTOR_TERMS,
    );
    assert_finite_r_diagonal_nonnegative(&rank_right);

    let tiny_negative = TensorMap::from_subblock_fn(
        &runtime,
        [&charge_zero_only],
        [&charge_zero_only],
        |_, indices| {
            if indices[0] == indices[1] {
                -1.0e-300 * (indices[0] + 1) as f64
            } else {
                0.0
            }
        },
    )
    .unwrap();
    let Qr {
        q: tiny_left,
        r: tiny_right,
    } = tiny_negative
        .to_cuda()
        .unwrap()
        .qr_compact(&codomain_axes(&tiny_negative), &domain_axes(&tiny_negative))
        .unwrap();
    let tiny_left = tiny_left.to_host().unwrap();
    let tiny_right = tiny_right.to_host().unwrap();
    numerics::assert_slices_close(
        "tiny_left .compose(tiny_right)",
        tiny_left
            .compose(&tiny_right)
            .unwrap()
            .dense_data()
            .unwrap(),
        tiny_negative.dense_data().unwrap(),
        FACTOR_TERMS,
    );
    assert_finite_r_diagonal_nonnegative(&tiny_right);

    let zn3 = Arc::new(ZNFusionRule::new(3).unwrap());
    let charge0 = GradedSpace::try_new(Arc::clone(&zn3), [(zn3.irrep(0), 2)]).unwrap();
    let charge1 = GradedSpace::try_new(Arc::clone(&zn3), [(zn3.irrep(1), 3)]).unwrap();
    let empty: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&charge0], [&charge1], |_, _| 1.0).unwrap();
    assert!(empty.dense_data().unwrap().is_empty());
    assert_typed_cuda_qr_matches_host(&empty);

    let device = mixed.to_cuda().unwrap();
    let lazy = device.adjoint().unwrap();
    assert!(matches!(
        lazy.qr_compact(&codomain_axes(&lazy), &domain_axes(&lazy)),
        Err(symtn::typed::Error::UnsupportedOnDevice(_))
    ));
    let expected = device
        .qr_compact(&codomain_axes(&device), &domain_axes(&device))
        .unwrap();
    let expected_left = expected.q.to_host().unwrap();
    let expected_right = expected.r.to_host().unwrap();
    for _ in 0..3 {
        let actual = device
            .qr_compact(&codomain_axes(&device), &domain_axes(&device))
            .unwrap();
        numerics::assert_slices_close(
            "actual.q",
            actual.q.to_host().unwrap().dense_data().unwrap(),
            expected_left.dense_data().unwrap(),
            FACTOR_TERMS,
        );
        numerics::assert_slices_close(
            "actual.r",
            actual.r.to_host().unwrap().dense_data().unwrap(),
            expected_right.dense_data().unwrap(),
            FACTOR_TERMS,
        );
    }
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    device
                        .qr_compact(&codomain_axes(&device), &domain_axes(&device))
                        .unwrap()
                })
            })
            .collect();
        for worker in workers {
            let actual = worker.join().unwrap();
            numerics::assert_slices_close(
                "actual.q",
                actual.q.to_host().unwrap().dense_data().unwrap(),
                expected_left.dense_data().unwrap(),
                FACTOR_TERMS,
            );
            numerics::assert_slices_close(
                "actual.r",
                actual.r.to_host().unwrap().dense_data().unwrap(),
                expected_right.dense_data().unwrap(),
                FACTOR_TERMS,
            );
        }
    });
}

/// Device QR is gauge-fixed like Host QR (positive diagonal), so the factors
/// compare pointwise, within the `numerics` rule
/// `32 eps(f64) sqrt(n) max(1, L) kappa`: `n` the stored entries, `L` the
/// largest compared magnitude, `kappa = sigma_max / sigma_min` of the source
/// from its `svd_vals`.
fn assert_c64_qr_matches_host<R>(source: &TensorMap<R, Complex64>)
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    let source_data = source.dense_data().unwrap().to_vec();
    let (mut largest, mut smallest) = (0.0_f64, f64::INFINITY);
    for entry in &source
        .svd_vals(&codomain_axes(source), &domain_axes(source))
        .unwrap()
    {
        for &value in &entry.values {
            largest = largest.max(value);
            smallest = smallest.min(value);
        }
    }
    assert!(smallest > 0.0, "the QR fixture must have full rank");
    // `kappa` conditions the rule by the source's singular-value spread. The
    // slice scale `L` (largest |entry| of Q, R or QR) never exceeds `||A||_F`,
    // so the bound stays within the former `64 sqrt(n) eps ||A|| kappa`.
    let terms = source_data.len().max(1);
    let kappa = largest / smallest;
    let Qr {
        q: host_q,
        r: host_r,
    } = source
        .qr_compact(&codomain_axes(source), &domain_axes(source))
        .unwrap();
    let device = source.to_cuda().unwrap();
    let Qr {
        q: q_device,
        r: r_device,
    } = device
        .qr_compact(&codomain_axes(&device), &domain_axes(&device))
        .unwrap();
    assert_device_factor_handles(source, [&q_device, &r_device]);
    let q = q_device.to_host().unwrap();
    let r = r_device.to_host().unwrap();
    assert_eq!(structural_snapshot(&q), structural_snapshot(&host_q));
    assert_eq!(structural_snapshot(&r), structural_snapshot(&host_r));
    numerics::assert_slices_close_scaled(
        "q",
        q.dense_data().unwrap(),
        host_q.dense_data().unwrap(),
        terms,
        kappa,
    );
    numerics::assert_slices_close_scaled(
        "r",
        r.dense_data().unwrap(),
        host_r.dense_data().unwrap(),
        terms,
        kappa,
    );
    numerics::assert_slices_close_scaled(
        "q.compose(r)",
        q.compose(&r).unwrap().dense_data().unwrap(),
        &source_data,
        terms,
        kappa,
    );
    assert_eq!(device.to_host().unwrap().dense_data().unwrap(), source_data);
}

/// `a + a^H` as a Hermitian fixture with nonzero imaginary off-diagonals: a
/// transpose-only admission rule would reject it.
fn hermitian_c64<R>(source: &TensorMap<R, Complex64>) -> TensorMap<R, Complex64>
where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    let hermitian = source
        .axpby(
            Complex64::new(1.0, 0.0),
            &source.adjoint().unwrap(),
            Complex64::new(1.0, 0.0),
        )
        .unwrap();
    assert!(
        hermitian.subblock_count() > 0
            && (0..hermitian.subblock_count()).any(|index| {
                let block = hermitian.subblock(index).unwrap();
                (0..block.shape()[0]).any(|row| {
                    (0..block.shape()[1]).any(|col| {
                        row != col
                            && hermitian.dense_data().unwrap()[block.offset()
                                + row * block.strides()[0]
                                + col * block.strides()[1]]
                                .im
                                .abs()
                                > 1e-6
                    })
                })
            }),
        "Hermitian fixture must have nonzero imaginary off-diagonal entries"
    );
    hermitian
}

fn assert_c64_eigh_trunc_composition_matches_host<R>(
    source: &TensorMap<R, Complex64>,
    truncation: &Truncation,
) where
    R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec,
{
    let source_data = source.dense_data().unwrap().to_vec();
    let expected = host_eigh_trunc(source, truncation, |value: Complex64| value.re);
    let device = source.to_cuda().unwrap();

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
    assert_device_factor_handles(source, [&d_device, &v_device]);
    let d = d_device.to_host().unwrap();
    let v = v_device.to_host().unwrap();

    let found = d.domain()[0]
        .find_truncated(&d.diagview().unwrap(), truncation)
        .unwrap();
    let selection = &found.selection;
    let d = d.restrict_leg(&[(0, selection), (1, selection)]).unwrap();
    let v = v.restrict_leg(&[(v.codomain_rank(), selection)]).unwrap();

    assert_eq!(*selection.subspace(), expected.d.domain()[0]);
    assert_eq!(structural_snapshot(&d), structural_snapshot(&expected.d));
    assert_eq!(structural_snapshot(&v), structural_snapshot(&expected.v));
    // Hermitian eigenvalues are real for a complex payload too.
    assert!(
        d.dense_data()
            .unwrap()
            .iter()
            .all(|value| value.im.abs() <= 1e-10),
        "eigenvalues must be real"
    );
    let mut kept = d.diagview().unwrap();
    kept.sort_by(|left, right| left.sector.cmp(&right.sector));
    assert_eq!(kept.len(), expected.eigenvalues.len());
    for (actual, expected) in kept.iter().zip(&expected.eigenvalues) {
        assert_eq!(actual.sector, expected.sector);
        let values: Vec<f64> = actual.values.iter().map(|value| value.re).collect();
        numerics::assert_slices_close("values", &values, &expected.values, FACTOR_TERMS);
    }
    assert!((found.error - expected.error).abs() <= 1e-9 * (1.0 + expected.error));

    assert!(is_isometric!(v, 1e-10), "V^H V = I");
    if matches!(truncation, Truncation::Full) {
        let rebuilt = v
            .compose(&d)
            .unwrap()
            .compose(&v.adjoint().unwrap())
            .unwrap();
        numerics::assert_slices_close(
            "rebuilt",
            rebuilt.dense_data().unwrap(),
            &source_data,
            FACTOR_TERMS,
        );
    }
    assert_eq!(device.to_host().unwrap().dense_data().unwrap(), source_data);
}

#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_c64_eigh_admits_hermitian_and_rejects_complex_symmetric_input() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let u1 = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    let hermitian = hermitian_c64(
        &TensorMap::<_, Complex64>::from_subblock_fn(&runtime, [&u1], [&u1], distinct_c64_fill())
            .unwrap(),
    );
    assert_c64_eigh_trunc_composition_matches_host(&hermitian, &Truncation::Full);
    assert_c64_eigh_trunc_composition_matches_host(&hermitian, &Truncation::rank(2));

    let Eigh { d, v } = hermitian
        .to_cuda()
        .unwrap()
        .eigh_full(
            &codomain_axes(&hermitian),
            &domain_axes(&hermitian),
            HermitianTol::DEFAULT,
        )
        .unwrap();
    let d = d.to_host().unwrap();
    let v = v.to_host().unwrap();
    assert!(is_isometric!(v, 1e-10), "V^H V = I");
    numerics::assert_slices_close(
        "v.compose(d)  .compose(v.adjoint())",
        v.compose(&d)
            .unwrap()
            .compose(&v.adjoint().unwrap())
            .unwrap()
            .dense_data()
            .unwrap(),
        hermitian.dense_data().unwrap(),
        FACTOR_TERMS,
    );

    let su2 = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap();
    let su2_hermitian = hermitian_c64(
        &TensorMap::<_, Complex64>::from_subblock_fn(
            &runtime,
            [&su2, &su2],
            [&su2, &su2],
            distinct_c64_fill(),
        )
        .unwrap(),
    );
    assert_c64_eigh_trunc_composition_matches_host(&su2_hermitian, &Truncation::Full);

    // `a^T = a` but `a^H != a`: a transpose-only admission rule would accept.
    let complex_symmetric =
        TensorMap::<_, Complex64>::from_subblock_fn(&runtime, [&u1], [&u1], |_, indices| {
            let (row, col) = (indices[0].min(indices[1]), indices[0].max(indices[1]));
            Complex64::new(1.0 + row as f64, 1.0 + col as f64)
        })
        .unwrap();
    assert!(
        complex_symmetric
            .eigh_full(
                &codomain_axes(&complex_symmetric),
                &domain_axes(&complex_symmetric),
                HermitianTol::DEFAULT
            )
            .is_err(),
        "the Host oracle rejects a complex-symmetric non-Hermitian input"
    );
    let device_error = complex_symmetric
        .to_cuda()
        .unwrap()
        .eigh_full(
            &codomain_axes(&complex_symmetric),
            &domain_axes(&complex_symmetric),
            HermitianTol::DEFAULT,
        )
        .expect_err("device EIGH must reject a complex-symmetric non-Hermitian input");
    assert!(
        matches!(
            &device_error,
            symtn::typed::Error::Operation(error)
                if matches!(
                    **error,
                    symtn::typed::OperationError::InvalidArgument { .. }
                )
        ),
        "unexpected error: {device_error:?}"
    );

    // A 2x2 hand case: `[[1, i], [-i, 1]]` is Hermitian, `[[1, i], [i, 1]]` is
    // complex-symmetric and must be rejected.
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let hand_hermitian =
        TensorMap::<_, Complex64>::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| match (
            indices[0], indices[1],
        ) {
            (0, 1) => Complex64::new(0.0, 1.0),
            (1, 0) => Complex64::new(0.0, -1.0),
            _ => Complex64::new(1.0, 0.0),
        })
        .unwrap();
    let Eigh { d, .. } = hand_hermitian
        .to_cuda()
        .unwrap()
        .eigh_full(
            &codomain_axes(&hand_hermitian),
            &domain_axes(&hand_hermitian),
            HermitianTol::DEFAULT,
        )
        .unwrap();
    let mut values: Vec<f64> = d
        .to_host()
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .map(|z| z.re)
        .collect();
    values.sort_by(f64::total_cmp);
    numerics::assert_slices_close("values", &values, &[0.0, 0.0, 0.0, 2.0], FACTOR_TERMS);

    let hand_symmetric =
        TensorMap::<_, Complex64>::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
            if indices[0] == indices[1] {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 1.0)
            }
        })
        .unwrap();
    assert!(hand_symmetric
        .to_cuda()
        .unwrap()
        .eigh_full(
            &codomain_axes(&hand_symmetric),
            &domain_axes(&hand_symmetric),
            HermitianTol::DEFAULT
        )
        .is_err());
}

/// #1320: a factor block that starts at an odd element offset of the flat
/// device buffer.
///
/// The aligned whole-factor route copies each block straight into its sector
/// region, so the copy must report each offset destination view's true
/// alignment to cuTENSOR; claiming 256-byte alignment for an unaligned view
/// faults with `cudaErrorMisalignedAddress` at the next synchronizing call. Every
/// degeneracy pair here puts the second sector's block at an offset whose byte
/// product is not a multiple of 256: `(3, 2)` at element 9, `(5, 2)` at 25,
/// `(3, 3)` at 9. The `(4, 2)` and `(2, 2)` pairs are aligned controls.
#[test]
#[ignore = "requires a real CUDA device"]
fn typed_cuda_factorizations_handle_blocks_at_unaligned_offsets() {
    let runtime = Runtime::builder().cuda(0).dense_threads(1).build().unwrap();
    let u1 = Arc::new(U1FusionRule);

    for (d0, d1) in [(3usize, 2usize), (5, 2), (3, 3), (4, 2), (2, 2)] {
        let leg = GradedSpace::try_new(
            Arc::clone(&u1),
            [(U1Irrep::new(0), d0), (U1Irrep::new(1), d1)],
        )
        .unwrap();
        let square = TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
            if indices[0] == indices[1] {
                4.0 + indices[0] as f64
            } else {
                (1 + indices[0] + 2 * indices[1]) as f64
            }
        })
        .unwrap();
        assert_typed_cuda_svd_matches_host(&square);
        assert_typed_cuda_qr_matches_host(&square);

        // Complex64 doubles the element size, so the same block offsets are a
        // different multiple of 256.
        let complex = TensorMap::<_, Complex64>::from_subblock_fn(
            &runtime,
            [&leg],
            [&leg],
            distinct_c64_fill(),
        )
        .unwrap();
        assert_c64_svd_matches_host(&complex);
        assert_c64_qr_matches_host(&complex);
    }
}
