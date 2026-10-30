use super::*;

#[test]
fn network_degeneracy_restriction_copies_nonprefix_rectangles_and_lazy_adjoint() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let zero = U1Irrep::new(0);
    let rows = GradedSpace::try_new(Arc::clone(&provider), [(zero, 3)]).unwrap();
    let columns = GradedSpace::try_new(Arc::clone(&provider), [(zero, 4)]).unwrap();
    let source =
        TensorMap::from_subblock_fn(engine, &runtime, [&rows], [&columns], |_, indices| {
            (indices[0] + 10 * indices[1]) as f64
        })
        .unwrap();
    let zero_id = TypedSectorAdmission::try_encode_label(provider.as_ref(), &zero).unwrap();

    let direct = source
        .network_restrict_degeneracies(
            false,
            &[
                NetworkDegeneracyRestriction {
                    effective_axis: 0,
                    authority_sector: zero_id,
                    range: 1..3,
                    partner: false,
                },
                NetworkDegeneracyRestriction {
                    effective_axis: 1,
                    authority_sector: zero_id,
                    range: 2..4,
                    partner: false,
                },
            ],
        )
        .unwrap();
    assert_eq!(direct.dense_data().unwrap(), &[21.0, 22.0, 31.0, 32.0]);
    assert!(Arc::ptr_eq(
        direct.logical_space().provider_arc(),
        &provider
    ));

    let lazy = source.adjoint(engine).unwrap();
    let restricted = lazy
        .network_restrict_degeneracies(
            false,
            &[
                NetworkDegeneracyRestriction {
                    effective_axis: 0,
                    authority_sector: zero_id,
                    range: 1..3,
                    partner: false,
                },
                NetworkDegeneracyRestriction {
                    effective_axis: 1,
                    authority_sector: zero_id,
                    range: 1..3,
                    partner: false,
                },
            ],
        )
        .unwrap();
    assert_eq!(restricted.dense_data().unwrap(), &[11.0, 21.0, 12.0, 22.0]);
}

#[test]
fn network_degeneracy_restriction_keeps_its_validation_order_after_the_shared_kernel() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    // The shared per-sector kernel must not move any of this path's own
    // checks: axis/duplicate, then empty range, then sector presence, then
    // the degeneracy bound, all before a destination exists.
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let zero = U1Irrep::new(0);
    let leg = GradedSpace::try_new(Arc::clone(&provider), [(zero, 3)]).unwrap();
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], |_, indices| {
            indices[0] as f64
        })
        .unwrap();
    let zero_id = TypedSectorAdmission::try_encode_label(provider.as_ref(), &zero).unwrap();
    let absent_id =
        TypedSectorAdmission::try_encode_label(provider.as_ref(), &U1Irrep::new(4)).unwrap();
    let message = |restrictions: &[NetworkDegeneracyRestriction]| {
        let Err(error) = source.network_restrict_degeneracies(false, restrictions) else {
            panic!("request must be rejected");
        };
        error.to_string()
    };
    let restriction = |effective_axis, authority_sector, range: std::ops::Range<usize>| {
        NetworkDegeneracyRestriction {
            effective_axis,
            authority_sector,
            range,
            partner: false,
        }
    };

    // Out-of-range axis outranks an empty range and an absent sector.
    assert!(message(&[restriction(2, absent_id, 1..1)])
        .contains("invalid or duplicate effective restriction axis"));
    // A duplicate axis outranks everything that follows it.
    assert!(
        message(&[restriction(0, zero_id, 0..1), restriction(0, zero_id, 0..1)])
            .contains("invalid or duplicate effective restriction axis")
    );
    // An empty range outranks an absent sector.
    assert!(message(&[restriction(0, absent_id, 1..1)]).contains("must be nonempty"));
    // An absent sector outranks the degeneracy bound.
    assert!(message(&[restriction(0, absent_id, 0..9)]).contains("is absent from effective axis"));
    assert!(message(&[restriction(0, zero_id, 0..9)]).contains("exceeds axis"));
    // The tensor itself is untouched and a valid request still works.
    assert_eq!(
        source.dense_data().unwrap(),
        &[0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0]
    );
    assert_eq!(
        source
            .network_restrict_degeneracies(false, &[restriction(0, zero_id, 1..3)])
            .unwrap()
            .dense_data()
            .unwrap(),
        &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    );
}

#[test]
fn network_degeneracy_restriction_maps_effective_nonselfdual_domain_sector() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let plus = U1Irrep::new(1);
    let space = GradedSpace::try_new(Arc::clone(&provider), [(plus, 3)]).unwrap();
    let source = TensorMap::from_subblock_fn(engine, &runtime, [&space], [&space], |_, indices| {
        (indices[0] + 10 * indices[1]) as f64
    })
    .unwrap();
    let plus_id = TypedSectorAdmission::try_encode_label(provider.as_ref(), &plus).unwrap();
    let restricted = source
        .network_restrict_degeneracies(
            false,
            &[
                NetworkDegeneracyRestriction {
                    effective_axis: 0,
                    authority_sector: plus_id,
                    range: 1..3,
                    partner: false,
                },
                NetworkDegeneracyRestriction {
                    effective_axis: 1,
                    authority_sector: plus_id,
                    range: 1..3,
                    partner: true,
                },
            ],
        )
        .unwrap();
    assert_eq!(restricted.dense_data().unwrap(), &[11.0, 12.0, 21.0, 22.0]);
}

#[test]
fn network_degeneracy_restriction_conjugates_complex_lazy_adjoint() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let zero = U1Irrep::new(0);
    let rows = GradedSpace::try_new(Arc::clone(&provider), [(zero, 2)]).unwrap();
    let columns = GradedSpace::try_new(Arc::clone(&provider), [(zero, 3)]).unwrap();
    let source =
        TensorMap::from_subblock_fn(engine, &runtime, [&rows], [&columns], |_, indices| {
            Complex64::new(
                indices[0] as f64 + 10.0 * indices[1] as f64,
                indices[1] as f64 + 1.0,
            )
        })
        .unwrap();
    let sector = TypedSectorAdmission::try_encode_label(provider.as_ref(), &zero).unwrap();
    let lazy = source.adjoint(engine).unwrap();
    let restricted = lazy
        .network_restrict_degeneracies(
            false,
            &[
                NetworkDegeneracyRestriction {
                    effective_axis: 0,
                    authority_sector: sector,
                    range: 1..3,
                    partner: false,
                },
                NetworkDegeneracyRestriction {
                    effective_axis: 1,
                    authority_sector: sector,
                    range: 0..2,
                    partner: false,
                },
            ],
        )
        .unwrap();
    assert_eq!(
        restricted.dense_data().unwrap(),
        &[
            Complex64::new(10.0, -2.0),
            Complex64::new(20.0, -3.0),
            Complex64::new(11.0, -2.0),
            Complex64::new(21.0, -3.0),
        ]
    );
}

#[test]
fn network_scatter_seals_authority_split_and_zero_block_legs_before_mutation() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let zero = U1Irrep::new(0);
    let full = GradedSpace::try_new(Arc::clone(&provider), [(zero, 2)]).unwrap();
    let mut destination =
        TensorMap::<_, f64>::from_subblock_fn(engine, &runtime, [&full], [&full], |_, ij| {
            (1 + ij[0] + 2 * ij[1]) as f64
        })
        .unwrap();
    let before = destination.dense_data().unwrap().to_vec();

    let other_provider = Arc::new(U1FusionRule);
    let other = GradedSpace::try_new(other_provider, [(zero, 2)]).unwrap();
    let wrong_authority = TensorMap::<_, f64>::zeros(engine, &runtime, [&other], [&other]).unwrap();
    assert!(matches!(destination
        .network_scatter_add_assign(engine, &wrong_authority, &[None, None]), Err(Error::InvalidArgument(message)) if message.contains("different sector provider")));
    assert_eq!(destination.dense_data().unwrap(), before);

    assert!(
        matches!(destination.network_scatter_add_assign(engine, &destination.clone(), &[None]), Err(Error::InvalidArgument(message)) if message.contains("1 ranges") && message.contains("rank-2"))
    );

    let wrong_split = TensorMap::<_, f64>::zeros(engine, &runtime, [&full, &full], []).unwrap();
    assert!(matches!(destination
        .network_scatter_add_assign(engine, &wrong_split, &[None, None]), Err(Error::InvalidArgument(message)) if message.contains("codomain/domain split (2, 0)") && message.contains("(1, 1)")));
    assert_eq!(destination.dense_data().unwrap(), before);

    // A non-vacuum rank-one map has no admissible blocks, so only logical
    // leg validation can reject malformed scatter metadata.
    let charged_full = GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(1), 2)]).unwrap();
    let charged_piece =
        GradedSpace::try_new(Arc::clone(&provider), [(U1Irrep::new(1), 1)]).unwrap();
    let mut empty_destination =
        TensorMap::<_, f64>::zeros(engine, &runtime, [&charged_full], []).unwrap();
    let empty_piece = TensorMap::<_, f64>::zeros(engine, &runtime, [&charged_piece], []).unwrap();
    assert_eq!(empty_destination.subblock_count(), 0);
    empty_destination
        .network_scatter_add_assign(engine, &empty_piece, &[Some(1..2)])
        .unwrap();
    assert!(matches!(empty_destination
        .network_scatter_add_assign(engine, &empty_piece, &[Some(2..3)]), Err(Error::InvalidArgument(message)) if message.contains("destination range")));

    let two_sectors =
        GradedSpace::try_new(provider, [(U1Irrep::new(1), 1), (U1Irrep::new(2), 1)]).unwrap();
    let empty_two = TensorMap::<_, f64>::zeros(engine, &runtime, [&two_sectors], []).unwrap();
    assert_eq!(empty_two.subblock_count(), 0);
    assert!(matches!(empty_destination
        .network_scatter_add_assign(engine, &empty_two, &[Some(0..1)]), Err(Error::InvalidArgument(message)) if message.contains("destination range")));
}

#[test]
fn network_scatter_reads_complex_lazy_adjoint_parent_without_materializing() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let densified = probe::record();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(U1FusionRule);
    let plus = U1Irrep::new(1);
    let rows = GradedSpace::try_new(Arc::clone(&provider), [(plus, 2)]).unwrap();
    let columns = GradedSpace::try_new(provider, [(plus, 3)]).unwrap();
    let source = TensorMap::from_subblock_fn(engine, &runtime, [&rows], [&columns], |_, ij| {
        Complex64::new((ij[0] + 2 * ij[1]) as f64, (1 + ij[0] + ij[1]) as f64)
    })
    .unwrap();
    let lazy = source.adjoint(engine).unwrap();
    let codomain = lazy.codomain();
    let domain = lazy.domain();
    let mut destination =
        TensorMap::zeros(engine, &runtime, codomain.iter(), domain.iter()).unwrap();
    destination
        .network_scatter_add_assign(engine, &lazy, &[None, None])
        .unwrap();
    let expected = (0..2)
        .flat_map(|column| {
            (0..3).map(move |row| {
                Complex64::new((column + 2 * row) as f64, -((1 + column + row) as f64))
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(destination.dense_data().unwrap(), expected);
    assert_eq!(densified.get(Event::DiagonalMaterialization), 0);
}
