use super::*;

fn poison_destination<R, D>(destination: &mut TensorMap<R, D>)
where
    D: TensorScalar,
{
    let TypedTensorRepr::Owned(body) = &mut destination.repr else {
        panic!("overwrite destination fixture must be owned")
    };
    let body = Arc::get_mut(body).expect("overwrite destination body must be unique");
    let data = Arc::get_mut(&mut body.data).expect("overwrite payload must be unique");
    let TypedData::Dense(data) = data else {
        panic!("overwrite destination fixture must be dense")
    };
    data.fill(D::from_real(f64::NAN));
}

fn assert_overwrite_matches<R, D>(
    engine: &mut dyn Engine,
    source: &TensorMap<R, D>,
    expected: TensorMap<R, D>,
    alpha: D,
    overwrite: impl FnOnce(&mut dyn Engine, &mut TensorMap<R, D>) -> Result<(), Error>,
) where
    R: TypedSectorAdmission<
            Sector = <R as SectorCodec>::Sector,
            Error = FusionAlgebraError,
            Mode = MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: TensorScalar + core::fmt::Debug + crate::test_numerics::numerics::Numeric,
{
    let source_before = source.dense_data().unwrap().to_vec();
    let mut destination = expected.zeros_like(engine).unwrap();
    poison_destination(&mut destination);
    let provider = Arc::as_ptr(destination.logical_space().provider_arc());
    let body = Arc::as_ptr(owned(&destination));
    let space = destination.logical_space().space() as *const DynamicFusionMapSpace;
    let storage = destination.dense_data().unwrap().as_ptr();

    overwrite(engine, &mut destination).unwrap();

    // The overwrite folds `alpha` into the recoupling while the oracle
    // scales afterwards; a recoupled entry sums at most one term per
    // source block.
    let scaled = expected.scale(engine, alpha).unwrap();
    crate::test_numerics::numerics::assert_slices_close(
        "overwrite against the scaled owned route",
        destination.dense_data().unwrap(),
        scaled.dense_data().unwrap(),
        source.subblock_count(),
    );
    assert_eq!(source.dense_data().unwrap(), source_before);
    assert_eq!(
        Arc::as_ptr(destination.logical_space().provider_arc()),
        provider
    );
    assert_eq!(Arc::as_ptr(owned(&destination)), body);
    assert_eq!(
        destination.logical_space().space() as *const DynamicFusionMapSpace,
        space
    );
    assert_eq!(destination.dense_data().unwrap().as_ptr(), storage);
}

#[allow(clippy::too_many_arguments)]
fn assert_contract_overwrite_matches<R, D>(
    engine: &mut dyn Engine,
    label: &str,
    lhs: &TensorMap<R, D>,
    rhs: &TensorMap<R, D>,
    lhs_axes: &[usize],
    rhs_axes: &[usize],
    output_axes: &[usize],
    alpha: D,
    ordered_alias: bool,
) where
    R: TypedSectorAdmission<
            Sector = <R as SectorCodec>::Sector,
            Error = FusionAlgebraError,
            Mode = MultiplicityFreeAdmissionMode,
        > + MultiplicityFreeRigidSymbols<Scalar = f64>
        + CheckedFusionAlgebra
        + SectorCodec,
    D: TensorScalar + core::fmt::Debug + crate::test_numerics::numerics::Numeric,
{
    let (codomain, domain) = output_axes.split_at(lhs.rank() - lhs_axes.len());
    let spec = ContractSpec {
        lhs: lhs_axes,
        rhs: rhs_axes,
        codomain,
        domain,
    };
    let expected = lhs
        .contract(engine, rhs, &spec)
        .unwrap_or_else(|error| panic!("{label} returning oracle failed: {error:?}"));
    let lhs_before = lhs.dense_data().unwrap().to_vec();
    let rhs_before = rhs.dense_data().unwrap().to_vec();
    let mut destination = expected.zeros_like(engine).unwrap();
    poison_destination(&mut destination);
    let provider = Arc::as_ptr(destination.logical_space().provider_arc());
    let body = Arc::as_ptr(owned(&destination));
    let space = destination.logical_space().space() as *const DynamicFusionMapSpace;
    let storage = destination.dense_data().unwrap().as_ptr();

    if ordered_alias {
        lhs.contract_into(
            engine,
            rhs,
            &spec,
            &mut destination,
            alpha,
            D::from_real(0.0),
        )
        .unwrap_or_else(|error| panic!("{label} ordered overwrite failed: {error:?}"));
    } else {
        lhs.contract_into(
            engine,
            rhs,
            &spec,
            &mut destination,
            alpha,
            D::from_real(0.0),
        )
        .unwrap_or_else(|error| panic!("{label} overwrite failed: {error:?}"));
    }

    // Owned and destination routes of one contraction. An entry is
    // bilinear in the operands, so `len(lhs) * len(rhs)` bounds its
    // terms, recoupled fusion trees (the cu1 output order) included.
    let terms = lhs.dense_data().unwrap().len() * rhs.dense_data().unwrap().len();
    crate::test_numerics::numerics::assert_slices_close(
        label,
        destination.dense_data().unwrap(),
        expected.scale(engine, alpha).unwrap().dense_data().unwrap(),
        terms,
    );
    assert_eq!(lhs.dense_data().unwrap(), lhs_before);
    assert_eq!(rhs.dense_data().unwrap(), rhs_before);
    assert_eq!(
        Arc::as_ptr(destination.logical_space().provider_arc()),
        provider
    );
    assert_eq!(Arc::as_ptr(owned(&destination)), body);
    assert_eq!(
        destination.logical_space().space() as *const DynamicFusionMapSpace,
        space
    );
    assert_eq!(destination.dense_data().unwrap().as_ptr(), storage);
}

#[test]
fn typed_tree_overwrite_matches_owned_provider_and_scalar_matrix() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();

    let u1_provider = Arc::new(U1FusionRule);
    let u1_leg = GradedSpace::try_new(
        Arc::clone(&u1_provider),
        [
            (U1Irrep::new(-1), 1),
            (U1Irrep::new(0), 2),
            (U1Irrep::new(1), 1),
        ],
    )
    .unwrap();
    let u1 = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&u1_leg, &u1_leg],
        [&u1_leg],
        |_, indices| indices.iter().sum::<usize>() as f64 + 1.0,
    )
    .unwrap();
    let u1_permuted = u1.permute(engine, &[1], &[2, 0]).unwrap();
    assert_overwrite_matches(engine, &u1, u1_permuted, -1.5, |engine, destination| {
        u1.permute_into(engine, &[1], &[2, 0], destination, -1.5, 0.0)
    });
    let u1_identity = u1.zeros_like(engine).unwrap();
    assert_overwrite_matches(engine, &u1, u1_identity, 0.0, |engine, destination| {
        u1.permute_into(engine, &[0, 1], &[2], destination, 0.0, 0.0)
    });

    let independent_leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 1),
            (U1Irrep::new(0), 2),
            (U1Irrep::new(1), 1),
        ],
    )
    .unwrap();
    let mut independent_destination = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&independent_leg, &independent_leg],
        [&independent_leg],
        |_, _| f64::NAN,
    )
    .unwrap();
    assert!(!Arc::ptr_eq(
        u1.logical_space().provider_arc(),
        independent_destination.logical_space().provider_arc()
    ));
    let destination_provider = Arc::as_ptr(independent_destination.logical_space().provider_arc());
    u1.permute_into(
        engine,
        &[0, 1],
        &[2],
        &mut independent_destination,
        2.0,
        0.0,
    )
    .unwrap();
    assert_eq!(
        independent_destination.dense_data().unwrap(),
        u1.scale(engine, 2.0).unwrap().dense_data().unwrap()
    );
    assert_eq!(
        Arc::as_ptr(independent_destination.logical_space().provider_arc()),
        destination_provider
    );

    let su2_provider = Arc::new(SU2FusionRule);
    let su2_leg = GradedSpace::try_new(
        su2_provider,
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 1),
        ],
    )
    .unwrap();
    let su2 = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&su2_leg, &su2_leg],
        [&su2_leg],
        |_, indices| indices.iter().sum::<usize>() as f64 + 1.0,
    )
    .unwrap()
    .convert::<Complex64>(engine)
    .unwrap();
    let alpha = num_complex::Complex64::new(0.75, -0.25);
    let su2_permuted = su2.permute(engine, &[1], &[2, 0]).unwrap();
    assert_overwrite_matches(engine, &su2, su2_permuted, alpha, |engine, destination| {
        su2.permute_into(engine, &[1], &[2, 0], destination, alpha, alpha * 0.0)
    });

    let product_provider = Arc::new(
        ProductFusionRule::<_, _>::try_new(U1FusionRule, FermionParityFusionRule).unwrap(),
    );
    let product_leg = GradedSpace::try_new(
        product_provider,
        [
            (product_sector(U1Irrep::new(0), Z2Irrep::EVEN), 2),
            (product_sector(U1Irrep::new(1), Z2Irrep::ODD), 1),
        ],
    )
    .unwrap();
    let product = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&product_leg, &product_leg],
        [&product_leg],
        |_, indices| indices.iter().sum::<usize>() as f64 + 1.0,
    )
    .unwrap();
    let product_permuted = product.permute(engine, &[1], &[2, 0]).unwrap();
    assert_overwrite_matches(
        engine,
        &product,
        product_permuted,
        2.0,
        |engine, destination| product.permute_into(engine, &[1], &[2, 0], destination, 2.0, 0.0),
    );
}

#[test]
fn typed_planar_overwrite_matches_fermionic_owned_routes() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(FermionParityFusionRule);
    let odd = GradedSpace::try_new(provider, [(Z2Irrep::ODD, 2)]).unwrap();
    let source = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&odd, &odd],
        [&odd, &odd],
        |_, indices| indices.iter().sum::<usize>() as f64 + 1.0,
    )
    .unwrap();
    let alpha = -1.25;
    let source_transpose = source.transpose(engine, &[3, 2], &[1, 0]).unwrap();

    assert_overwrite_matches(
        engine,
        &source,
        source_transpose,
        alpha,
        |engine, destination| {
            source.transpose_into(engine, &[3, 2], &[1, 0], destination, alpha, alpha * 0.0)
        },
    );
    let source_transpose = source.transpose(engine, &[1, 3], &[0, 2]).unwrap();
    assert_overwrite_matches(
        engine,
        &source,
        source_transpose,
        alpha,
        |engine, destination| {
            source.transpose_into(engine, &[1, 3], &[0, 2], destination, alpha, alpha * 0.0)
        },
    );
    let right = source.repartition(engine, 3).unwrap();
    assert_overwrite_matches(engine, &source, right, alpha, |engine, destination| {
        source.repartition_into(engine, destination, alpha, alpha * 0.0)
    });
    let left = source.repartition(engine, 1).unwrap();
    assert_overwrite_matches(engine, &source, left, alpha, |engine, destination| {
        source.repartition_into(engine, destination, alpha, alpha * 0.0)
    });
}

fn f64_bits<R>(engine: &mut dyn Engine, tensor: &TensorMap<R, f64>) -> Vec<u64> {
    tensor
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .map(|value| value.to_bits())
        .collect()
}

fn f64_destination_state<R>(
    engine: &mut dyn Engine,
    tensor: &TensorMap<R, f64>,
) -> (Vec<u64>, [usize; 4]) {
    (
        f64_bits(engine, tensor),
        [
            Arc::as_ptr(tensor.logical_space().provider_arc()) as usize,
            Arc::as_ptr(owned(tensor)) as usize,
            tensor.logical_space().space() as *const DynamicFusionMapSpace as usize,
            tensor.dense_data().unwrap().as_ptr() as usize,
        ],
    )
}

fn pop_dense_element<R, D>(tensor: &mut TensorMap<R, D>) {
    let TypedTensorRepr::Owned(body) = &mut tensor.repr else {
        panic!("malformed-storage fixture must be owned")
    };
    let body = Arc::get_mut(body).expect("malformed-storage body must be unique");
    let data = Arc::get_mut(&mut body.data).expect("malformed-storage payload must be unique");
    let TypedData::Dense(data) = data else {
        panic!("malformed-storage fixture must be dense")
    };
    data.pop();
}

#[test]
fn typed_tree_overwrite_rejections_leave_destination_unchanged() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)]).unwrap();
    let source =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg], [&leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();
    let expected = source.permute(engine, &[1], &[2, 0]).unwrap();

    let assert_unchanged =
        |engine: &mut dyn Engine, destination: &TensorMap<U1FusionRule, f64>, before: &[u64]| {
            assert_eq!(f64_bits(engine, destination), before);
        };

    let mut wrong_layout = source.zeros_like(engine).unwrap();
    poison_destination(&mut wrong_layout);
    let before = f64_bits(engine, &wrong_layout);
    assert!(source
        .permute_into(engine, &[1], &[2, 0], &mut wrong_layout, 1.0, 0.0)
        .is_err());
    assert_unchanged(engine, &wrong_layout, &before);

    for (codomain_axes, domain_axes) in [(&[1, 1][..], &[2][..]), (&[1][..], &[2, 3][..])] {
        let mut destination = expected.zeros_like(engine).unwrap();
        poison_destination(&mut destination);
        let before = f64_bits(engine, &destination);
        assert!(source
            .permute_into(
                engine,
                codomain_axes,
                domain_axes,
                &mut destination,
                1.0,
                0.0
            )
            .is_err());
        assert_unchanged(engine, &destination, &before);
    }

    let mut nonplanar = source
        .transpose(engine, &[2], &[1, 0])
        .unwrap()
        .zeros_like(engine)
        .unwrap();
    poison_destination(&mut nonplanar);
    let before = f64_bits(engine, &nonplanar);
    assert!(source
        .transpose_into(engine, &[0, 2], &[1], &mut nonplanar, 1.0, 0.0)
        .is_err());
    assert_unchanged(engine, &nonplanar, &before);

    let other_runtime = Runtime::builder().build().unwrap();
    let mut foreign = expected.zeros_like(engine).unwrap();
    foreign.runtime = other_runtime;
    poison_destination(&mut foreign);
    let before = f64_bits(engine, &foreign);
    assert_eq!(
        source
            .permute_into(engine, &[1], &[2, 0], &mut foreign, 1.0, 0.0)
            .unwrap_err(),
        Error::RuntimeMismatch
    );
    assert_unchanged(engine, &foreign, &before);

    let mut shared_body = expected.zeros_like(engine).unwrap();
    poison_destination(&mut shared_body);
    let before = f64_bits(engine, &shared_body);
    let shared_body_handle = shared_body.clone();
    assert!(source
        .permute_into(engine, &[1], &[2, 0], &mut shared_body, 1.0, 0.0)
        .is_err());
    assert_unchanged(engine, &shared_body, &before);
    drop(shared_body_handle);

    let mut shared_payload = expected.zeros_like(engine).unwrap();
    poison_destination(&mut shared_payload);
    let before = f64_bits(engine, &shared_payload);
    let payload_handle = shared_payload
        .insert_unit(engine, 0, Side::Domain, Duality::Plain)
        .unwrap();
    assert!(source
        .permute_into(engine, &[1], &[2, 0], &mut shared_payload, 1.0, 0.0)
        .is_err());
    assert_unchanged(engine, &shared_payload, &before);
    drop(payload_handle);

    let mut alias = source
        .insert_unit(engine, 0, Side::Domain, Duality::Plain)
        .unwrap()
        .remove_unit(engine, 0)
        .unwrap();
    let before = f64_bits(engine, &alias);
    assert!(source
        .permute_into(engine, &[0, 1], &[2], &mut alias, 1.0, 0.0)
        .is_err());
    assert_unchanged(engine, &alias, &before);

    let mut bad_len = expected.zeros_like(engine).unwrap();
    poison_destination(&mut bad_len);
    let before = {
        let TypedTensorRepr::Owned(body) = &mut bad_len.repr else {
            unreachable!()
        };
        let body = Arc::get_mut(body).unwrap();
        let data = Arc::get_mut(&mut body.data).unwrap();
        let TypedData::Dense(data) = data else {
            unreachable!()
        };
        data.pop();
        f64_bits(engine, &bad_len)
    };
    assert!(source
        .permute_into(engine, &[1], &[2, 0], &mut bad_len, 1.0, 0.0)
        .is_err());
    assert_unchanged(engine, &bad_len, &before);

    let mut lazy_destination = expected.adjoint(engine).unwrap();
    let before = f64_bits(engine, &lazy_destination);
    assert!(source
        .permute_into(engine, &[1], &[2, 0], &mut lazy_destination, 1.0, 0.0)
        .is_err());
    assert_eq!(f64_bits(engine, &lazy_destination), before);

    let lazy_source = source.adjoint(engine).unwrap();
    let mut destination = expected.zeros_like(engine).unwrap();
    poison_destination(&mut destination);
    let before = f64_bits(engine, &destination);
    assert!(lazy_source
        .permute_into(engine, &[1], &[2, 0], &mut destination, 1.0, 0.0)
        .is_err());
    assert_unchanged(engine, &destination, &before);

    let square = TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, indices| {
        indices.iter().sum::<usize>() as f64 + 1.0
    })
    .unwrap();
    let compact = square.svd_compact(engine, &[0], &[1]).unwrap().s;
    let mut compact_destination = compact.zeros_like(engine).unwrap();
    let before = compact_destination
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .to_vec();
    assert!(square
        .permute_into(engine, &[0], &[1], &mut compact_destination, 1.0, 0.0)
        .is_err());
    assert_eq!(
        compact_destination
            .materialize(engine)
            .unwrap()
            .dense_data()
            .unwrap(),
        before
    );

    let z2 = Arc::new(ZNFusionRule::new(2).unwrap());
    let z3 = Arc::new(ZNFusionRule::new(3).unwrap());
    let z2_leg = GradedSpace::try_new(Arc::clone(&z2), [(z2.irrep(0), 1)]).unwrap();
    let z3_leg = GradedSpace::try_new(Arc::clone(&z3), [(z3.irrep(0), 1)]).unwrap();
    let z2_source =
        TensorMap::from_subblock_fn(engine, &runtime, [&z2_leg], [&z2_leg], |_, _| 2.0).unwrap();
    let mut z3_destination =
        TensorMap::from_subblock_fn(engine, &runtime, [&z3_leg], [&z3_leg], |_, _| f64::NAN)
            .unwrap();
    let before = f64_bits(engine, &z3_destination);
    assert_eq!(
        z2_source
            .permute_into(engine, &[0], &[1], &mut z3_destination, 1.0, 0.0)
            .unwrap_err(),
        Error::RuleMismatch
    );
    assert_eq!(f64_bits(engine, &z3_destination), before);
}

#[test]
fn typed_tree_overwrite_covers_boundary_ranks_and_runtime_cache_reuse() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    if crate::test_cache::run_isolated_or_return(
        "SYMTN_TREE_OVERWRITE_REUSE_ISOLATED",
        "typed::representation_gates::overwrite::typed_tree_overwrite_covers_boundary_ranks_and_runtime_cache_reuse",
    ) {
        return;
    }
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 1)]).unwrap();

    let one_sided = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&leg],
        std::iter::empty::<&GradedSpace<U1FusionRule>>(),
        |_, _| 3.0,
    )
    .unwrap();
    let moved = one_sided.repartition(engine, 0).unwrap();
    assert_overwrite_matches(engine, &one_sided, moved, 2.0, |engine, destination| {
        one_sided.repartition_into(engine, destination, 2.0, 0.0)
    });

    let square = TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, _| 4.0).unwrap();
    let scalar = square.trace_pairs(engine, &[(0, 1)]).unwrap();
    let scalar_destination = scalar.transpose(engine, &[], &[]).unwrap();
    assert_overwrite_matches(
        engine,
        &scalar,
        scalar_destination,
        -0.5,
        |engine, destination| scalar.transpose_into(engine, &[], &[], destination, -0.5, 0.0),
    );

    let high_rank = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        (0..9).map(|_| &leg),
        (0..8).map(|_| &leg),
        |_, _| 1.0,
    )
    .unwrap();
    let mut high_rank_destination = high_rank.zeros_like(engine).unwrap();
    poison_destination(&mut high_rank_destination);
    let before = f64_bits(engine, &high_rank_destination);
    assert!(high_rank
        .permute_into(
            engine,
            &[0, 1, 2, 3, 4, 5, 6, 7, 17],
            &[8, 9, 10, 11, 12, 13, 14, 15],
            &mut high_rank_destination,
            1.0,
            0.0,
        )
        .is_err());
    assert_eq!(f64_bits(engine, &high_rank_destination), before);

    crate::cache::clear();
    let source =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg], [&leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();
    let expected = source.permute(engine, &[1], &[2, 0]).unwrap();
    crate::cache::clear();
    let mut first = expected.zeros_like(engine).unwrap();
    source
        .permute_into(engine, &[1], &[2, 0], &mut first, 1.0, 0.0)
        .unwrap();
    let cold = crate::test_cache::completed();
    let mut second = expected.zeros_like(engine).unwrap();
    source
        .permute_into(engine, &[1], &[2, 0], &mut second, 1.0, 0.0)
        .unwrap();
    let warm = crate::test_cache::completed();
    assert_eq!(warm.entries(), cold.entries());
    assert!(warm.hits() > cold.hits());
    assert_eq!(first.dense_data().unwrap(), second.dense_data().unwrap());
}

#[test]
fn typed_tree_overwrite_shared_runtime_is_concurrent_and_deterministic() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // What: exact-layout admission and completed replay share one Runtime
    // without serializing execution or changing results across callers.
    let runtime = Runtime::builder().build().unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 3),
            (U1Irrep::new(1), 2),
        ],
    )
    .unwrap();
    let source =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg, &leg], [&leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap();
    let expected = source.permute(engine, &[1], &[2, 0]).unwrap();
    let mut warm = expected.zeros_like(engine).unwrap();
    source
        .permute_into(engine, &[1], &[2, 0], &mut warm, 1.0, 0.0)
        .unwrap();

    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|_| {
                scope.spawn(|| {
                    let engine = &mut symtn_dense::CpuEngine::serial();
                    let mut destination = expected.zeros_like(engine).unwrap();
                    source
                        .permute_into(engine, &[1], &[2, 0], &mut destination, 1.0, 0.0)
                        .unwrap();
                    destination
                })
            })
            .collect();
        for handle in handles {
            assert_eq!(
                handle.join().unwrap().dense_data().unwrap(),
                expected.dense_data().unwrap()
            );
        }
    });
}

#[test]
fn typed_contract_overwrite_matches_provider_scalar_and_order_matrix() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();

    let u1_leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 1),
            (U1Irrep::new(0), 2),
            (U1Irrep::new(1), 1),
        ],
    )
    .unwrap();
    let u1 = TensorMap::from_subblock_fn(engine, &runtime, [&u1_leg], [&u1_leg], |_, indices| {
        indices.iter().sum::<usize>() as f64 + 1.0
    })
    .unwrap();
    assert_contract_overwrite_matches(engine, "u1", &u1, &u1, &[1], &[0], &[0, 1], -1.5, false);

    let su2_leg = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 1),
        ],
    )
    .unwrap();
    let su2 =
        TensorMap::from_subblock_fn(engine, &runtime, [&su2_leg], [&su2_leg], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 1.0
        })
        .unwrap()
        .convert::<Complex64>(engine)
        .unwrap();
    assert_contract_overwrite_matches(
        engine,
        "su2",
        &su2,
        &su2,
        &[1],
        &[0],
        &[1, 0],
        num_complex::Complex64::new(0.75, -0.25),
        true,
    );

    let odd = GradedSpace::try_new(Arc::new(FermionParityFusionRule), [(Z2Irrep::ODD, 2)]).unwrap();
    let fermionic = TensorMap::from_subblock_fn(engine, &runtime, [&odd], [&odd], |_, indices| {
        indices.iter().sum::<usize>() as f64 + 1.0
    })
    .unwrap();
    assert_contract_overwrite_matches(
        engine,
        "fz2",
        &fermionic,
        &fermionic,
        &[0, 1],
        &[1, 0],
        &[],
        0.0,
        false,
    );

    let product_leg = GradedSpace::try_new(
        Arc::new(
            ProductFusionRule::<_, _>::try_new(U1FusionRule, FermionParityFusionRule).unwrap(),
        ),
        [
            (product_sector(U1Irrep::new(0), Z2Irrep::EVEN), 2),
            (product_sector(U1Irrep::new(1), Z2Irrep::ODD), 1),
        ],
    )
    .unwrap();
    let product = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&product_leg],
        [&product_leg],
        |_, indices| indices.iter().sum::<usize>() as f64 + 1.0,
    )
    .unwrap();
    assert_contract_overwrite_matches(
        engine,
        "product",
        &product,
        &product,
        &[1],
        &[0],
        &[0, 1],
        2.0,
        false,
    );

    let q = GradedSpace::try_new(
        Arc::new(CU1FusionRule),
        [(CU1Irrep::from_twice_charge(1), 1)],
    )
    .unwrap();
    let cu1 =
        TensorMap::from_subblock_fn(engine, &runtime, [&q, &q, &q], [&q], |_, _| 1.0).unwrap();
    let cu1_expected = cu1
        .contract(
            engine,
            &cu1,
            &ContractSpec {
                lhs: &[3],
                rhs: &[0],
                codomain: &[5, 1, 3],
                domain: &[0, 4, 2],
            },
        )
        .unwrap();
    assert!(cu1_expected.dense_data().unwrap().contains(&0.0));
    assert_contract_overwrite_matches(
        engine,
        "cu1",
        &cu1,
        &cu1,
        &[3],
        &[0],
        &[5, 1, 3, 0, 4, 2],
        1.0,
        false,
    );
}

#[test]
fn typed_contract_overwrite_keeps_distinct_destination_provider_authority() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let mut build = |provider: Arc<U1FusionRule>, offset: f64| {
        let leg = GradedSpace::try_new(
            provider,
            [
                (U1Irrep::new(-1), 1),
                (U1Irrep::new(0), 2),
                (U1Irrep::new(1), 1),
            ],
        )
        .unwrap();
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, indices| {
            offset + indices.iter().sum::<usize>() as f64
        })
        .unwrap()
    };

    let lhs = build(Arc::new(U1FusionRule), 1.0);
    let rhs = build(Arc::new(U1FusionRule), 10.0);
    let destination_provider = Arc::new(U1FusionRule);
    let destination_lhs = build(Arc::clone(&destination_provider), 0.0);
    let destination_rhs = build(destination_provider, 0.0);
    let expected = lhs
        .contract(
            engine,
            &rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    let mut destination = destination_lhs
        .contract(
            engine,
            &destination_rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap()
        .zeros_like(engine)
        .unwrap();
    poison_destination(&mut destination);

    assert!(!Arc::ptr_eq(
        lhs.logical_space().provider_arc(),
        rhs.logical_space().provider_arc()
    ));
    assert!(!Arc::ptr_eq(
        lhs.logical_space().provider_arc(),
        destination.logical_space().provider_arc()
    ));
    let provider = Arc::as_ptr(destination.logical_space().provider_arc());
    let body = Arc::as_ptr(owned(&destination));
    let space = destination.logical_space().space() as *const DynamicFusionMapSpace;
    let storage = destination.dense_data().unwrap().as_ptr();

    lhs.contract_into(engine, &rhs, &RANK_TWO_COMPOSE, &mut destination, 1.0, 0.0)
        .unwrap();

    assert_eq!(
        destination.dense_data().unwrap(),
        expected.dense_data().unwrap()
    );
    assert_eq!(
        Arc::as_ptr(destination.logical_space().provider_arc()),
        provider
    );
    assert_eq!(Arc::as_ptr(owned(&destination)), body);
    assert_eq!(
        destination.logical_space().space() as *const DynamicFusionMapSpace,
        space
    );
    assert_eq!(destination.dense_data().unwrap().as_ptr(), storage);
}

#[test]
fn typed_contract_overwrite_accepts_lazy_and_compact_inputs_without_warming_adjoint() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let densified = probe::record();
    let runtime = Runtime::builder().build().unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 3), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let lhs = TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, indices| {
        (indices[0] + 2 * indices[1] + 1) as f64
    })
    .unwrap();
    let rhs = TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, indices| {
        (2 * indices[0] + indices[1] + 1) as f64
    })
    .unwrap();
    let lazy_lhs = lhs.adjoint(engine).unwrap();
    let lazy_rhs = rhs.adjoint(engine).unwrap();
    let expected = lazy_lhs
        .contract(
            engine,
            &lazy_rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    let mut destination = expected.zeros_like(engine).unwrap();
    poison_destination(&mut destination);
    lazy_lhs
        .contract_into(
            engine,
            &lazy_rhs,
            &RANK_TWO_COMPOSE,
            &mut destination,
            1.0,
            0.0,
        )
        .unwrap();
    assert_eq!(
        destination.dense_data().unwrap(),
        expected.dense_data().unwrap()
    );
    for lazy in [&lazy_lhs, &lazy_rhs] {
        let TypedTensorRepr::Adjoint(_) = &lazy.repr else {
            unreachable!()
        };
    }

    let Svd { u, s, .. } = lhs.svd_compact(engine, &[0], &[1]).unwrap();
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    let expected = u
        .contract(
            engine,
            &s,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    let mut destination = expected.zeros_like(engine).unwrap();
    poison_destination(&mut destination);
    u.contract_into(engine, &s, &RANK_TWO_COMPOSE, &mut destination, 1.0, 0.0)
        .unwrap();
    assert_eq!(
        destination.dense_data().unwrap(),
        expected.dense_data().unwrap()
    );
    // #1870: the compact operand scales `u`'s leg into the destination,
    // as eager's arm does, and is never densified.
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
}

#[test]
fn typed_contract_overwrite_rejections_are_preclear_and_atomic() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let densified = probe::record();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let leg = GradedSpace::try_new(provider, [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)]).unwrap();
    let lhs = TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, indices| {
        (indices[0] + 2 * indices[1] + 1) as f64
    })
    .unwrap();
    let rhs = TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, indices| {
        (2 * indices[0] + indices[1] + 1) as f64
    })
    .unwrap();
    let expected = lhs
        .contract(
            engine,
            &rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap();
    let destination = |engine: &mut dyn Engine| {
        let mut destination = expected.zeros_like(engine).unwrap();
        poison_destination(&mut destination);
        destination
    };

    let mut foreign = destination(engine);
    foreign.runtime = Runtime::builder().build().unwrap();
    let before = f64_destination_state(engine, &foreign);
    assert_eq!(
        lhs.contract_into(
            engine,
            &rhs,
            &ContractSpec {
                lhs: &[9],
                ..RANK_TWO_COMPOSE
            },
            &mut foreign,
            1.0,
            0.0,
        )
        .unwrap_err(),
        Error::RuntimeMismatch
    );
    assert_eq!(f64_destination_state(engine, &foreign), before);

    let other_runtime = Runtime::builder().build().unwrap();
    let foreign_rhs =
        TensorMap::from_subblock_fn(engine, &other_runtime, [&leg], [&leg], |_, indices| {
            (indices[0] + indices[1] + 1) as f64
        })
        .unwrap();
    let mut rejected = destination(engine);
    let before = f64_destination_state(engine, &rejected);
    assert_eq!(
        lhs.contract_into(
            engine,
            &foreign_rhs,
            &ContractSpec {
                lhs: &[9],
                ..RANK_TWO_COMPOSE
            },
            &mut rejected,
            1.0,
            0.0,
        )
        .unwrap_err(),
        Error::RuntimeMismatch
    );
    assert_eq!(f64_destination_state(engine, &rejected), before);

    for spec in [
        ContractSpec {
            lhs: &[1, 1],
            rhs: &[0, 0],
            codomain: &[],
            domain: &[],
        },
        ContractSpec {
            lhs: &[2],
            ..RANK_TWO_COMPOSE
        },
        ContractSpec {
            domain: &[0],
            ..RANK_TWO_COMPOSE
        },
    ] {
        let mut rejected = destination(engine);
        let before = f64_destination_state(engine, &rejected);
        assert!(lhs
            .contract_into(engine, &rhs, &spec, &mut rejected, 1.0, 0.0)
            .is_err());
        assert_eq!(f64_destination_state(engine, &rejected), before);
    }

    let bad_leg = GradedSpace::try_new(
        Arc::clone(lhs.logical_space().provider_arc()),
        [(U1Irrep::new(7), 1)],
    )
    .unwrap();
    let bad_rhs =
        TensorMap::from_subblock_fn(engine, &runtime, [&bad_leg], [&bad_leg], |_, _| 1.0).unwrap();
    let mut rejected = destination(engine);
    let before = f64_destination_state(engine, &rejected);
    assert!(lhs
        .contract_into(engine, &bad_rhs, &RANK_TWO_COMPOSE, &mut rejected, 1.0, 0.0)
        .is_err());
    assert_eq!(f64_destination_state(engine, &rejected), before);

    let mut wrong_layout = lhs
        .insert_unit(engine, 0, Side::Domain, Duality::Plain)
        .unwrap()
        .zeros_like(engine)
        .unwrap();
    poison_destination(&mut wrong_layout);
    let before = f64_destination_state(engine, &wrong_layout);
    assert!(lhs
        .contract_into(engine, &rhs, &RANK_TWO_COMPOSE, &mut wrong_layout, 1.0, 0.0)
        .is_err());
    assert_eq!(f64_destination_state(engine, &wrong_layout), before);

    for malformed in ["lhs", "rhs", "destination"] {
        let mut bad_lhs = lhs.scale(engine, 1.0).unwrap();
        let mut bad_rhs = rhs.scale(engine, 1.0).unwrap();
        let mut rejected = destination(engine);
        match malformed {
            "lhs" => pop_dense_element(&mut bad_lhs),
            "rhs" => pop_dense_element(&mut bad_rhs),
            "destination" => pop_dense_element(&mut rejected),
            _ => unreachable!(),
        }
        let before = f64_destination_state(engine, &rejected);
        assert!(bad_lhs
            .contract_into(engine, &bad_rhs, &RANK_TWO_COMPOSE, &mut rejected, 1.0, 0.0)
            .is_err());
        assert_eq!(f64_destination_state(engine, &rejected), before);
    }

    let mut lhs_alias = lhs
        .insert_unit(engine, 0, Side::Domain, Duality::Plain)
        .unwrap()
        .remove_unit(engine, 0)
        .unwrap();
    let before = f64_destination_state(engine, &lhs_alias);
    assert!(lhs
        .contract_into(engine, &rhs, &RANK_TWO_COMPOSE, &mut lhs_alias, 1.0, 0.0)
        .is_err());
    assert_eq!(f64_destination_state(engine, &lhs_alias), before);

    let mut rhs_alias = rhs
        .insert_unit(engine, 0, Side::Domain, Duality::Plain)
        .unwrap()
        .remove_unit(engine, 0)
        .unwrap();
    let before = f64_destination_state(engine, &rhs_alias);
    assert!(lhs
        .contract_into(engine, &rhs, &RANK_TWO_COMPOSE, &mut rhs_alias, 1.0, 0.0)
        .is_err());
    assert_eq!(f64_destination_state(engine, &rhs_alias), before);

    let mut shared_body = destination(engine);
    let shared_body_handle = shared_body.clone();
    let before = f64_destination_state(engine, &shared_body);
    assert!(lhs
        .contract_into(engine, &rhs, &RANK_TWO_COMPOSE, &mut shared_body, 1.0, 0.0)
        .is_err());
    assert_eq!(f64_destination_state(engine, &shared_body), before);
    drop(shared_body_handle);

    let mut shared_payload = destination(engine);
    let shared_payload_handle = shared_payload
        .insert_unit(engine, 0, Side::Domain, Duality::Plain)
        .unwrap();
    let before = f64_destination_state(engine, &shared_payload);
    assert!(lhs
        .contract_into(
            engine,
            &rhs,
            &RANK_TWO_COMPOSE,
            &mut shared_payload,
            1.0,
            0.0
        )
        .is_err());
    assert_eq!(f64_destination_state(engine, &shared_payload), before);
    drop(shared_payload_handle);

    let mut lazy_destination = expected.adjoint(engine).unwrap();
    let TypedTensorRepr::Adjoint(view) = &lazy_destination.repr else {
        unreachable!()
    };
    let view = Arc::as_ptr(view);
    assert!(lhs
        .contract_into(
            engine,
            &rhs,
            &RANK_TWO_COMPOSE,
            &mut lazy_destination,
            1.0,
            0.0
        )
        .is_err());
    let TypedTensorRepr::Adjoint(after) = &lazy_destination.repr else {
        unreachable!()
    };
    assert_eq!(Arc::as_ptr(after), view);

    let mut compact_destination = lhs.svd_compact(engine, &[0], &[1]).unwrap().s;
    let payload = Arc::clone(&owned(&compact_destination).data);
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
    assert!(lhs
        .contract_into(
            engine,
            &rhs,
            &RANK_TWO_COMPOSE,
            &mut compact_destination,
            1.0,
            0.0
        )
        .is_err());
    assert!(Arc::ptr_eq(&owned(&compact_destination).data, &payload));
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);

    let z2 = Arc::new(ZNFusionRule::new(2).unwrap());
    let z3 = Arc::new(ZNFusionRule::new(3).unwrap());
    let z2_leg = GradedSpace::try_new(Arc::clone(&z2), [(z2.irrep(0), 1)]).unwrap();
    let z3_leg = GradedSpace::try_new(Arc::clone(&z3), [(z3.irrep(0), 1)]).unwrap();
    let z2_lhs =
        TensorMap::from_subblock_fn(engine, &runtime, [&z2_leg], [&z2_leg], |_, _| 1.0).unwrap();
    let z2_rhs =
        TensorMap::from_subblock_fn(engine, &runtime, [&z2_leg], [&z2_leg], |_, _| 2.0).unwrap();
    let z3_rhs =
        TensorMap::from_subblock_fn(engine, &runtime, [&z3_leg], [&z3_leg], |_, _| 2.0).unwrap();
    let mut rejected = z2_lhs
        .contract(
            engine,
            &z2_rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap()
        .zeros_like(engine)
        .unwrap();
    poison_destination(&mut rejected);
    let before = f64_destination_state(engine, &rejected);
    assert_eq!(
        z2_lhs
            .contract_into(engine, &z3_rhs, &RANK_TWO_COMPOSE, &mut rejected, 1.0, 0.0)
            .unwrap_err(),
        Error::RuleMismatch
    );
    assert_eq!(f64_destination_state(engine, &rejected), before);

    let z3_lhs =
        TensorMap::from_subblock_fn(engine, &runtime, [&z3_leg], [&z3_leg], |_, _| 1.0).unwrap();
    let mut z3_destination = z3_lhs
        .contract(
            engine,
            &z3_rhs,
            &ContractSpec {
                lhs: &[1],
                rhs: &[0],
                codomain: &[0],
                domain: &[1],
            },
        )
        .unwrap()
        .zeros_like(engine)
        .unwrap();
    poison_destination(&mut z3_destination);
    let before = f64_destination_state(engine, &z3_destination);
    assert_eq!(
        z2_lhs
            .contract_into(
                engine,
                &z2_rhs,
                &RANK_TWO_COMPOSE,
                &mut z3_destination,
                1.0,
                0.0
            )
            .unwrap_err(),
        Error::RuleMismatch
    );
    assert_eq!(f64_destination_state(engine, &z3_destination), before);
}

#[test]
fn typed_contract_overwrite_handles_unmatched_sectors_and_reuses_runtime_cache() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    if crate::test_cache::run_isolated_or_return(
        "SYMTN_CONTRACT_OVERWRITE_REUSE_ISOLATED",
        "typed::representation_gates::overwrite::typed_contract_overwrite_handles_unmatched_sectors_and_reuses_runtime_cache",
    ) {
        return;
    }
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let bond = GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(0), 2)]).unwrap();
    let left = GradedSpace::try_new(
        Arc::clone(&provider),
        [(U1Irrep::new(0), 1), (U1Irrep::new(1), 1)],
    )
    .unwrap();
    let partly_disjoint = GradedSpace::try_new(
        Arc::clone(&provider),
        [(U1Irrep::new(0), 1), (U1Irrep::new(2), 1)],
    )
    .unwrap();
    let disjoint = GradedSpace::try_new(provider, [(U1Irrep::new(3), 1)]).unwrap();
    let lhs = TensorMap::from_subblock_fn(engine, &runtime, [&left], [&bond], |_, indices| {
        indices.iter().sum::<usize>() as f64 + 1.0
    })
    .unwrap();
    for open in [&partly_disjoint, &disjoint] {
        let rhs = TensorMap::from_subblock_fn(engine, &runtime, [&bond], [open], |_, indices| {
            indices.iter().sum::<usize>() as f64 + 2.0
        })
        .unwrap();
        assert_contract_overwrite_matches(
            engine,
            "unmatched sectors",
            &lhs,
            &rhs,
            &[1],
            &[0],
            &[0, 1],
            1.0,
            false,
        );
    }

    let provider = Arc::new(CU1FusionRule);
    let q = GradedSpace::try_new(provider, [(CU1Irrep::from_twice_charge(1), 1)]).unwrap();
    let source =
        TensorMap::from_subblock_fn(engine, &runtime, [&q, &q, &q], [&q], |_, _| 1.0).unwrap();
    let axes = [5, 1, 3, 0, 4, 2];
    let expected = source
        .contract(
            engine,
            &source,
            &ContractSpec {
                lhs: &[3],
                rhs: &[0],
                codomain: &axes[..3],
                domain: &axes[3..],
            },
        )
        .unwrap();
    crate::cache::clear();
    let mut first = expected.zeros_like(engine).unwrap();
    poison_destination(&mut first);
    source
        .contract_into(
            engine,
            &source,
            &ContractSpec {
                lhs: &[3],
                rhs: &[0],
                codomain: &axes[..3],
                domain: &axes[3..],
            },
            &mut first,
            1.0,
            0.0,
        )
        .unwrap();
    let cold = crate::test_cache::completed();
    let mut second = expected.zeros_like(engine).unwrap();
    poison_destination(&mut second);
    source
        .contract_into(
            engine,
            &source,
            &ContractSpec {
                lhs: &[3],
                rhs: &[0],
                codomain: &axes[..3],
                domain: &axes[3..],
            },
            &mut second,
            1.0,
            0.0,
        )
        .unwrap();
    let warm = crate::test_cache::completed();
    assert_eq!(first.dense_data().unwrap(), second.dense_data().unwrap());
    assert_eq!(warm.entries(), cold.entries());
    assert!(warm.hits() > cold.hits());
}
