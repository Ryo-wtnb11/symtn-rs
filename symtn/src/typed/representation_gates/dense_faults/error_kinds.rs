//! A malformed executor answer is symtn's own `Dense(ShapeMismatch)` or
//! `Dense(DTypeMismatch)` at the facade, in both fusion modes (#1765), never
//! a backend error.

use super::*;
use symtn_dense::DenseView;

/// Answers QR with one factor too few, or with `f32` factors.
struct MalformedQr {
    inner: DefaultDenseExecutor,
    f32_factors: bool,
}

impl DenseExecutor for MalformedQr {
    fn svd(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.svd(engine, input)
    }

    fn eigh(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        self.inner.eigh(engine, input)
    }

    fn qr(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        if self.f32_factors {
            let one = [1.0_f32];
            return self.inner.qr(
                engine,
                DenseRead::F32(DenseView::new(&one, &[1, 1], &[1, 1], 0)?),
            );
        }
        let mut factors = self.inner.qr(engine, input)?;
        factors.pop();
        Ok(factors)
    }

    fn dot_general_into(
        &mut self,
        engine: &mut dyn Engine,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        config: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        self.inner
            .dot_general_into(engine, output, lhs, rhs, config)
    }
}

/// Injects a `MalformedQr` for as long as the returned guard lives.
fn inject_malformed_qr(f32_factors: bool) -> probe::DenseInjection {
    probe::inject_dense(Box::new(MalformedQr {
        inner: DefaultDenseExecutor::new(),
        f32_factors,
    }))
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

type ErrorKind = fn(&OperationError) -> bool;

const MALFORMED: [(bool, ErrorKind); 2] = [(false, is_shape_mismatch), (true, is_dtype_mismatch)];

#[test]
fn multiplicity_free_malformed_qr_answer_is_a_typed_dense_error() {
    let engine = &mut crate::typed::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let wide = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    for (f32_factors, kind) in MALFORMED {
        let _dense = inject_malformed_qr(f32_factors);
        let square: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&wide], [&wide], |_, _| 1.0).unwrap();
        assert!(matches!(
            square.qr_compact(engine, &[0], &[1]),
            Err(Error::Operation(error)) if kind(&error)
        ));
    }
}

#[test]
fn checked_malformed_qr_answer_is_a_typed_dense_error() {
    let engine = &mut crate::typed::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0xa0));
    let wide = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    for (f32_factors, kind) in MALFORMED {
        let _dense = inject_malformed_qr(f32_factors);
        let square: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&wide], [&wide], |_, _| 1.0).unwrap();
        assert!(match square.qr_compact(engine, &[0], &[1]) {
            Err(GenericTensorError::Facade(Error::Operation(error))) => kind(&error),
            Err(GenericTensorError::Plan(CheckedGenericPlanError::Operation(error))) =>
                kind(&error),
            _ => false,
        });
    }
}
