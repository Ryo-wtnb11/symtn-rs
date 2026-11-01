use super::*;

#[test]
fn scaling_by_the_inverse_norm_divides_by_the_dimension_weighted_norm() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = runtime();

    // SU(2): normalizing by a dimension-weighted norm is what a plain
    // Frobenius normalization would get wrong, and only a non-abelian fixture
    // can see it.
    let typed = su2_tensor(&runtime);
    let norm = typed.norm(engine, 2.0).unwrap();
    let unit = typed.scale(engine, 1.0 / norm).unwrap();
    for (&actual, &source) in unit
        .dense_data()
        .unwrap()
        .iter()
        .zip(typed.dense_data().unwrap())
    {
        assert!((actual - source / norm).abs() <= 1e-12 * source.abs().max(1.0));
    }
    assert!((unit.norm(engine, 2.0).unwrap() - 1.0).abs() < 1e-12);
}

#[test]
fn inner_uses_the_same_dimension_weight_as_norm() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: `inner` is TensorKit's `dot(x, y)` — conjugate-linear in the first
    // argument and quantum-dimension weighted, and it comes back as a `D`.
    // SU(2) is what
    // exercises the weighted branch; Z2 alone would take the abelian fast path.
    let runtime = runtime();

    let z2_typed = z2_tensor(&runtime);
    let su2_typed = su2_tensor(&runtime);
    let agree = |typed_value: f64, norm: f64| {
        // `<t, t>` is the squared norm, which is the identity that pins this
        // weighting to `norm`'s.
        assert!((typed_value - norm * norm).abs() < 1e-9 * norm * norm);
    };
    agree(
        z2_typed.inner(engine, &z2_typed).unwrap(),
        z2_typed.norm(engine, 2.0).unwrap(),
    );
    agree(
        su2_typed.inner(engine, &su2_typed).unwrap(),
        su2_typed.norm(engine, 2.0).unwrap(),
    );
}

#[test]
fn tr_uses_the_nonabelian_dimension_weight() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: TensorKit's positive trace `Σ_c dim(c) * tr(b_c)`. SU(2) separates
    // it from the unweighted diagonal sum.
    let runtime = runtime();

    let typed = su2_tensor(&runtime);
    let trace = typed.tr(engine).unwrap();
    // The unweighted diagonal sum of the same blocks, for contrast: `tr` is
    // not it, which is what a dropped `dim(c)` would make it.
    let unweighted: f64 = (0..typed.subblock_count())
        .map(|index| {
            let block = typed.subblock(index).unwrap();
            let size = block.shape()[0];
            (0..size)
                .map(|i| {
                    typed.dense_data().unwrap()
                        [block.offset() + i * (block.strides()[0] + block.strides()[1])]
                })
                .sum::<f64>()
        })
        .sum();
    assert!(
        (trace - unweighted).abs() > 1.0,
        "the SU(2) fixture must separate the weighted trace from the unweighted one"
    );
}

// ---------------------------------------------------------------------------
// `TensorMap::adjoint`.
// ---------------------------------------------------------------------------

#[test]
fn adjoint_carries_an_external_provider() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = runtime();
    let provider = Arc::new(external_z3());
    let tensor = z3_rank_four(&runtime, &provider);

    let adjoint = tensor.adjoint(engine).unwrap();

    assert_eq!(
        adjoint
            .materialize(engine)
            .unwrap()
            .dense_data()
            .unwrap()
            .len(),
        tensor.dense_data().unwrap().len()
    );
    assert_eq!(
        adjoint.adjoint(engine).unwrap().dense_data().unwrap(),
        tensor.dense_data().unwrap()
    );
    // A dagger preserves the dimension-weighted norm.
    assert!((adjoint.norm(engine, 2.0).unwrap() - tensor.norm(engine, 2.0).unwrap()).abs() < 1e-12);
}

#[test]
fn trace_pairs_preserves_partial_trace_geometry() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: the full trace to a rank-0 tensor and a partial trace that leaves a
    // leg open. The partial case is the one
    // that exercises the output-axis derivation and the destination's
    // codomain rank; the full case is the degenerate one.
    let runtime = runtime();

    let typed = z2_endomorphism(&runtime);
    let full = typed.trace_pairs(engine, &[(0, 1)]).unwrap();
    assert_eq!(full.dense_data().unwrap().len(), 1);
    assert_eq!(full.scalar(engine).unwrap(), typed.tr(engine).unwrap());

    // `[v, v] <- [v]`: tracing axis 1 against axis 2 leaves axis 0 open, so the
    // result is `[v] <- []` and the open axis keeps its side.
    let typed = z2_tensor(&runtime);
    let partial = typed.trace_pairs(engine, &[(1, 2)]).unwrap();
    assert_eq!(partial.codomain().len(), 1);
    assert_eq!(partial.domain().len(), 0);
    assert!(partial
        .dense_data()
        .unwrap()
        .iter()
        .any(|&value| value != 0.0));

    // The two cases above leave at most one survivor, and it is codomain-side,
    // so neither can see the order of `output_axes` nor the codomain-rank
    // filter that splits the destination. These two can.
    //
    // `[v] <- [v, v]`, tracing (0, 1): the survivor is axis 2, a domain-side
    // leg, so the destination is `[] <- [v]` — a dropped codomain-rank filter
    // would put it in the codomain instead.
    let typed = z2_tensor_split(&runtime, 1);
    let survivor = typed.trace_pairs(engine, &[(0, 1)]).unwrap();
    assert_eq!(survivor.codomain().len(), 0);
    assert_eq!(survivor.domain().len(), 1);
    assert!(survivor
        .dense_data()
        .unwrap()
        .iter()
        .any(|&value| value != 0.0));

    // `[v, v] <- [v, v]`, tracing (0, 3): two survivors, axes 1 and 2, one on
    // each side — so their relative order in `output_axes` is observable, and
    // reversing it changes the bytes.
    let leg = GradedSpace::try_new(
        Arc::new(symtn::sector::Z2FusionRule),
        [
            (symtn::sector::Z2Irrep::EVEN, 2),
            (symtn::sector::Z2Irrep::ODD, 3),
        ],
    )
    .unwrap();
    let typed: TensorMap<symtn::sector::Z2FusionRule, f64> = TensorMap::from_subblock_fn(
        engine,
        &runtime,
        [&leg, &leg],
        [&leg, &leg],
        typed_fill_value,
    )
    .unwrap();
    let two_survivors = typed.trace_pairs(engine, &[(0, 3)]).unwrap();
    assert_eq!(two_survivors.codomain().len(), 1);
    assert_eq!(two_survivors.domain().len(), 1);
    assert!(two_survivors
        .dense_data()
        .unwrap()
        .iter()
        .any(|&value| value != 0.0));
}

#[test]
fn trace_pairs_of_nothing_is_the_source() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = runtime();
    let typed = z2_tensor(&runtime);

    let traced = typed.trace_pairs(engine, &[]).unwrap();

    assert_eq!(traced.dense_data().unwrap(), typed.dense_data().unwrap());
    assert_eq!(traced.codomain().len(), 2);
}

#[test]
fn trace_pairs_rejects_malformed_pairs() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // Out-of-range and repeated axes are an invalid axis subset.
    let runtime = runtime();
    let typed = z2_endomorphism(&runtime);

    for pairs in [vec![(0usize, 9usize)], vec![(0, 0)], vec![(0, 1), (1, 0)]] {
        assert!(matches!(
            typed.trace_pairs(engine, &pairs).unwrap_err(),
            symtn::typed::Error::Operation(operation)
                if matches!(
                    *operation,
                    symtn::typed::OperationError::InvalidAxisSet { tensor: "trace pairs", .. }
                )
        ));
    }
}

#[test]
fn fermionic_trace_pairs_is_the_supertrace_and_tr_is_not() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    // What: the documented divergence. `tr` is TensorKit's positive trace
    // (`Σ_c dim(c) tr(b_c)`), `trace_pairs` is the tensor-contraction trace,
    // which for a fermionic rule carries the twist — the supertrace. On this
    // fixture the odd sector contributes with opposite signs, so the two
    // numbers differ.
    let runtime = runtime();
    let typed = fermionic_endo(&runtime);

    let positive = typed.tr(engine).unwrap();
    let super_trace = typed.trace_pairs(engine, &[(0, 1)]).unwrap();

    assert_ne!(
        super_trace.dense_data().unwrap(),
        [positive],
        "the fermionic supertrace must not coincide with the positive trace"
    );
}
