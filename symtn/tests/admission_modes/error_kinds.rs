//! One variant per error kind at the facade: a hom-space misuse is
//! `SpaceMismatch`, a value precondition `InvalidArgument`, and a numerical
//! failure `Dense(NumericalFailure)`, never a backend error.

use super::*;
use symtn::typed::HermitianTol;

in_both_modes! {
    /// Each misuse class maps to its one `OperationError` variant, whether
    /// the facade or a checked plan stage refuses it.
    fn misuse_has_one_variant_per_kind<M>() {
        let engine = &mut symtn_dense::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let provider = M::provider();
        let wide = GradedSpace::try_new(Arc::clone(&provider), [(M::charged(), 2)]).unwrap();
        let narrow = GradedSpace::try_new(Arc::clone(&provider), [(M::charged(), 1)]).unwrap();
        let rectangular: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&wide], [&narrow], |_, _| 1.0).unwrap();
        let skew: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&wide], [&wide], |_, ij| {
                (2 * ij[0] + ij[1]) as f64
            })
            .unwrap();
        let singular: TensorMap<_, f64> = TensorMap::diagonal(
            engine,
            &runtime,
            &wide,
            [SectorSpectrum { sector: M::charged(), values: vec![0.0, 1.0] }],
        )
        .unwrap();
        let narrow_square: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&narrow], [&narrow], |_, _| 1.0)
                .unwrap();
        let space_mismatch = |error: OperationError| {
            assert!(matches!(error, OperationError::SpaceMismatch { .. }), "{error:?}")
        };

        let error = operation_err(skew.eigh_full(engine, &[0], &[1], HermitianTol::DEFAULT));
        assert!(matches!(error, OperationError::InvalidArgument { .. }), "{error:?}");
        let error = operation_err(singular.inv(engine, &[0], &[1]));
        assert!(
            matches!(error, OperationError::Dense(symtn_dense::DenseError::NumericalFailure { .. })),
            "{error:?}"
        );

        let mut other = skew.clone();
        let mut wrong_destination = rectangular.clone();
        space_mismatch(operation_err(
            rectangular.eigh_full(engine, &[0], &[1], HermitianTol::DEFAULT),
        ));
        space_mismatch(operation_err(rectangular.exp(engine, &[0], &[1])));
        space_mismatch(operation_err(rectangular.tr(engine)));
        space_mismatch(operation_err(skew.axpby(engine, 1.0, &rectangular, 1.0)));
        space_mismatch(operation_err(skew.inner(engine, &rectangular)));
        space_mismatch(operation_err(rectangular.axpby_into(engine, &mut other, 1.0, 1.0)));
        // The identity permutation: the toy cannot recouple a bend, and the
        // destination check is the same for every permutation.
        space_mismatch(operation_err(skew.permute_into(
            engine,
            &[0],
            &[1],
            &mut wrong_destination,
            1.0,
            0.0,
        )));
        space_mismatch(operation_err(TensorMap::<_, f64>::isomorphism(
            engine,
            &runtime,
            [&wide],
            [&narrow],
        )));
        space_mismatch(operation_err(TensorMap::<_, f64>::isometry(
            engine,
            &runtime,
            [&narrow],
            [&wide],
        )));
        space_mismatch(operation_err(narrow_square.cat(
            engine,
            &rectangular,
            symtn::typed::Side::Domain,
        )));
        space_mismatch(operation_err(wide.oplus(&wide.try_dual().unwrap())));
    }
}
