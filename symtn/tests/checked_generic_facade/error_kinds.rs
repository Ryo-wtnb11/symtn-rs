//! One variant per error kind at the facade, in both fusion modes (#1765):
//! a hom-space misuse is `SpaceMismatch`, a value precondition
//! `InvalidArgument`, and a numerical failure `Dense(NumericalFailure)`,
//! never a backend error.

use super::*;
use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{CheckedGenericPlanError, Error, OperationError};
use symtn_dense::Engine;

fn is_space_mismatch(error: &OperationError) -> bool {
    matches!(error, OperationError::SpaceMismatch { .. })
}

fn is_invalid_argument(error: &OperationError) -> bool {
    matches!(error, OperationError::InvalidArgument { .. })
}

fn is_numerical_failure(error: &OperationError) -> bool {
    matches!(
        error,
        OperationError::Dense(DenseError::NumericalFailure { .. })
    )
}

fn multiplicity_free<T: fmt::Debug>(
    result: Result<T, Error>,
    kind: fn(&OperationError) -> bool,
) -> bool {
    matches!(result, Err(Error::Operation(error)) if kind(&error))
}

fn checked<T: fmt::Debug, E>(
    result: Result<T, GenericTensorError<E>>,
    kind: fn(&OperationError) -> bool,
) -> bool {
    match result {
        Err(GenericTensorError::Facade(Error::Operation(error))) => kind(&error),
        Err(GenericTensorError::Plan(CheckedGenericPlanError::Operation(error))) => kind(&error),
        _ => false,
    }
}

#[test]
fn multiplicity_free_misuse_has_one_variant_per_kind() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let rule = Arc::new(U1FusionRule);
    let wide = GradedSpace::try_new(Arc::clone(&rule), [(U1Irrep::new(0), 2)]).unwrap();
    let narrow = GradedSpace::try_new(Arc::clone(&rule), [(U1Irrep::new(0), 1)]).unwrap();
    let rectangular: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&wide], [&narrow], |_, _| 1.0).unwrap();
    let skew: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&wide], [&wide], |_, index| {
            (2 * index[0] + index[1]) as f64
        })
        .unwrap();
    let singular: TensorMap<_, f64> = TensorMap::diagonal(
        engine,
        &runtime,
        &wide,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![0.0, 1.0],
        }],
    )
    .unwrap();

    let eigh = |engine: &mut dyn Engine, t: &TensorMap<_, f64>| {
        t.eigh_full(engine, &[0], &[1], HermitianTol::DEFAULT)
    };
    assert!(multiplicity_free(
        eigh(engine, &rectangular),
        is_space_mismatch
    ));
    assert!(multiplicity_free(
        rectangular.exp(engine, &[0], &[1]),
        is_space_mismatch
    ));
    assert!(multiplicity_free(eigh(engine, &skew), is_invalid_argument));
    assert!(multiplicity_free(
        singular.inv(engine, &[0], &[1]),
        is_numerical_failure
    ));
    // #2097: the remaining hom-space relations, each TensorKit `SpaceMismatch`.
    let mut other = skew.clone();
    let mut wrong_destination = rectangular.clone();
    assert!(multiplicity_free(rectangular.tr(engine), is_space_mismatch));
    assert!(multiplicity_free(
        skew.axpby(engine, 1.0, &rectangular, 1.0),
        is_space_mismatch
    ));
    assert!(multiplicity_free(
        skew.inner(engine, &rectangular),
        is_space_mismatch
    ));
    assert!(multiplicity_free(
        rectangular.axpby_into(engine, &mut other, 1.0, 1.0),
        is_space_mismatch
    ));
    assert!(multiplicity_free(
        skew.permute_into(engine, &[1], &[0], &mut wrong_destination, 1.0, 0.0),
        is_space_mismatch
    ));
    assert!(multiplicity_free(
        TensorMap::<_, f64>::isomorphism(engine, &runtime, [&wide], [&narrow]),
        is_space_mismatch
    ));
    assert!(multiplicity_free(
        TensorMap::<_, f64>::isometry(engine, &runtime, [&narrow], [&wide]),
        is_space_mismatch
    ));
    let narrow_cod: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&narrow], [&narrow], |_, _| 1.0).unwrap();
    assert!(multiplicity_free(
        narrow_cod.cat(engine, &rectangular, symtn::typed::Side::Domain),
        is_space_mismatch
    ));
    assert!(multiplicity_free(
        wide.oplus(&wide.try_dual().unwrap()),
        is_space_mismatch
    ));
}

#[test]
fn checked_misuse_has_one_variant_per_kind() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let wide = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let narrow = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let rectangular: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&wide], [&narrow], |_, _| 1.0).unwrap();
    let skew: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&wide], [&wide], |_, index| {
            (2 * index[0] + index[1]) as f64
        })
        .unwrap();
    let singular: TensorMap<_, f64> = TensorMap::diagonal(
        engine,
        &runtime,
        &wide,
        [SectorSpectrum {
            sector: Label::X,
            values: vec![0.0, 1.0],
        }],
    )
    .unwrap();

    let eigh = |engine: &mut dyn Engine, t: &TensorMap<_, f64>| {
        t.eigh_full(engine, &[0], &[1], HermitianTol::DEFAULT)
    };
    assert!(checked(eigh(engine, &rectangular), is_space_mismatch));
    assert!(checked(
        rectangular.exp(engine, &[0], &[1]),
        is_space_mismatch
    ));
    assert!(checked(eigh(engine, &skew), is_invalid_argument));
    assert!(checked(
        singular.inv(engine, &[0], &[1]),
        is_numerical_failure
    ));
    // #2097: the remaining hom-space relations, each TensorKit `SpaceMismatch`.
    let mut other = skew.clone();
    assert!(checked(rectangular.tr(engine), is_space_mismatch));
    assert!(checked(
        skew.axpby(engine, 1.0, &rectangular, 1.0),
        is_space_mismatch
    ));
    assert!(checked(skew.inner(engine, &rectangular), is_space_mismatch));
    assert!(checked(
        rectangular.axpby_into(engine, &mut other, 1.0, 1.0),
        is_space_mismatch
    ));
    assert!(checked(
        TensorMap::<_, f64>::isomorphism(engine, &runtime, [&wide], [&narrow]),
        is_space_mismatch
    ));
    assert!(checked(
        TensorMap::<_, f64>::isometry(engine, &runtime, [&narrow], [&wide]),
        is_space_mismatch
    ));
    let narrow_cod: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&narrow], [&narrow], |_, _| 1.0).unwrap();
    assert!(checked(
        narrow_cod.cat(engine, &rectangular, symtn::typed::Side::Domain),
        is_space_mismatch
    ));
    assert!(checked(
        wide.oplus(&wide.try_dual().unwrap()),
        is_space_mismatch
    ));
}
