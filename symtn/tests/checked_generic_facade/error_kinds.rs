//! One variant per error kind at the facade, in both fusion modes (#1765):
//! a hom-space misuse is `SpaceMismatch`, a value precondition
//! `InvalidArgument`, a numerical failure `Dense(NumericalFailure)`, and a
//! malformed executor answer symtn's own `Dense(ShapeMismatch)` or
//! `Dense(DTypeMismatch)`, never a backend error.

use super::*;
use symtn::expert::DenseView;
use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{CheckedGenericPlanError, Error, OperationError};

/// Answers QR with one factor too few, or with `f32` factors.
struct MalformedQr {
    inner: DefaultDenseExecutor,
    f32_factors: bool,
}

impl MalformedQr {
    fn runtime(f32_factors: bool) -> Runtime {
        Runtime::builder()
            .dense_threads(1)
            .with_dense_executor(Box::new(Self {
                inner: DefaultDenseExecutor::new(),
                f32_factors,
            }))
            .build()
            .unwrap()
    }
}

impl DenseExecutor for MalformedQr {
    fn svd(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.svd(input)
    }

    fn eigh(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.eigh(input)
    }

    fn qr(&mut self, input: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        if self.f32_factors {
            let one = [1.0_f32];
            return self
                .inner
                .qr(DenseRead::F32(DenseView::new(&one, &[1, 1], &[1, 1], 0)?));
        }
        let mut factors = self.inner.qr(input)?;
        factors.pop();
        Ok(factors)
    }

    fn dot_general_into(
        &mut self,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        config: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        self.inner.dot_general_into(output, lhs, rhs, config)
    }
}

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

fn is_shape_mismatch(error: &OperationError) -> bool {
    matches!(
        error,
        OperationError::Dense(DenseError::ShapeMismatch { .. })
    )
}

fn is_dtype_mismatch(error: &OperationError) -> bool {
    matches!(
        error,
        OperationError::Dense(DenseError::DTypeMismatch { .. })
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
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let rule = Arc::new(U1FusionRule);
    let wide = GradedSpace::try_new(Arc::clone(&rule), [(U1Irrep::new(0), 2)]).unwrap();
    let narrow = GradedSpace::try_new(Arc::clone(&rule), [(U1Irrep::new(0), 1)]).unwrap();
    let rectangular: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&wide], [&narrow], |_, _| 1.0).unwrap();
    let skew: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&wide], [&wide], |_, index| {
            (2 * index[0] + index[1]) as f64
        })
        .unwrap();
    let singular: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &wide,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![0.0, 1.0],
        }],
    )
    .unwrap();

    let eigh = |t: &TensorMap<_, f64>| t.eigh_full(&[0], &[1], HermitianTol::DEFAULT);
    assert!(multiplicity_free(eigh(&rectangular), is_space_mismatch));
    assert!(multiplicity_free(
        rectangular.exp(&[0], &[1]),
        is_space_mismatch
    ));
    assert!(multiplicity_free(eigh(&skew), is_invalid_argument));
    assert!(multiplicity_free(
        singular.inv(&[0], &[1]),
        is_numerical_failure
    ));
    // #2097: the remaining hom-space relations, each TensorKit `SpaceMismatch`.
    let mut other = skew.clone();
    let mut wrong_destination = rectangular.clone();
    assert!(multiplicity_free(rectangular.tr(), is_space_mismatch));
    assert!(multiplicity_free(
        skew.axpby(1.0, &rectangular, 1.0),
        is_space_mismatch
    ));
    assert!(multiplicity_free(
        skew.inner(&rectangular),
        is_space_mismatch
    ));
    assert!(multiplicity_free(
        rectangular.axpby_into(&mut other, 1.0, 1.0),
        is_space_mismatch
    ));
    assert!(multiplicity_free(
        skew.permute_into(&[1], &[0], &mut wrong_destination, 1.0, 0.0),
        is_space_mismatch
    ));
    assert!(multiplicity_free(
        TensorMap::<_, f64>::isomorphism(&runtime, [&wide], [&narrow]),
        is_space_mismatch
    ));
    assert!(multiplicity_free(
        TensorMap::<_, f64>::isometry(&runtime, [&narrow], [&wide]),
        is_space_mismatch
    ));
    let narrow_cod: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&narrow], [&narrow], |_, _| 1.0).unwrap();
    assert!(multiplicity_free(
        narrow_cod.cat(&rectangular, symtn::typed::Side::Domain),
        is_space_mismatch
    ));
    assert!(multiplicity_free(
        wide.oplus(&wide.try_dual().unwrap()),
        is_space_mismatch
    ));
    for (f32_factors, kind) in [
        (false, is_shape_mismatch as fn(&OperationError) -> bool),
        (true, is_dtype_mismatch),
    ] {
        let runtime = MalformedQr::runtime(f32_factors);
        let square: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(&runtime, [&wide], [&wide], |_, _| 1.0).unwrap();
        assert!(multiplicity_free(square.qr_compact(&[0], &[1]), kind));
    }
}

#[test]
fn checked_misuse_has_one_variant_per_kind() {
    let _serial = crate::counting_alloc::serial();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let wide = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let narrow = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let rectangular: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&wide], [&narrow], |_, _| 1.0).unwrap();
    let skew: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&wide], [&wide], |_, index| {
            (2 * index[0] + index[1]) as f64
        })
        .unwrap();
    let singular: TensorMap<_, f64> = TensorMap::diagonal(
        &runtime,
        &wide,
        [SectorSpectrum {
            sector: Label::X,
            values: vec![0.0, 1.0],
        }],
    )
    .unwrap();

    let eigh = |t: &TensorMap<_, f64>| t.eigh_full(&[0], &[1], HermitianTol::DEFAULT);
    assert!(checked(eigh(&rectangular), is_space_mismatch));
    assert!(checked(rectangular.exp(&[0], &[1]), is_space_mismatch));
    assert!(checked(eigh(&skew), is_invalid_argument));
    assert!(checked(singular.inv(&[0], &[1]), is_numerical_failure));
    // #2097: the remaining hom-space relations, each TensorKit `SpaceMismatch`.
    let mut other = skew.clone();
    assert!(checked(rectangular.tr(), is_space_mismatch));
    assert!(checked(
        skew.axpby(1.0, &rectangular, 1.0),
        is_space_mismatch
    ));
    assert!(checked(skew.inner(&rectangular), is_space_mismatch));
    assert!(checked(
        rectangular.axpby_into(&mut other, 1.0, 1.0),
        is_space_mismatch
    ));
    assert!(checked(
        TensorMap::<_, f64>::isomorphism(&runtime, [&wide], [&narrow]),
        is_space_mismatch
    ));
    assert!(checked(
        TensorMap::<_, f64>::isometry(&runtime, [&narrow], [&wide]),
        is_space_mismatch
    ));
    let narrow_cod: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&narrow], [&narrow], |_, _| 1.0).unwrap();
    assert!(checked(
        narrow_cod.cat(&rectangular, symtn::typed::Side::Domain),
        is_space_mismatch
    ));
    assert!(checked(
        wide.oplus(&wide.try_dual().unwrap()),
        is_space_mismatch
    ));
    for (f32_factors, kind) in [
        (false, is_shape_mismatch as fn(&OperationError) -> bool),
        (true, is_dtype_mismatch),
    ] {
        let runtime = MalformedQr::runtime(f32_factors);
        let square: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(&runtime, [&wide], [&wide], |_, _| 1.0).unwrap();
        assert!(checked(square.qr_compact(&[0], &[1]), kind));
    }
}
