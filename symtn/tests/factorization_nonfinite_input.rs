//! Multiplicity-free factorizations refuse nonfinite input through the shared
//! finite-input stage (#1986), for dense, lazy-adjoint and compact diagonal
//! input in real and complex dtypes. The checked Generic counterpart is in
//! `checked_generic_facade/nonfinite.rs`.

use std::sync::Arc;
use symtn::typed::HermitianTol;

use num_complex::Complex64;
use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{
    Error, GradedSpace, OperationError, Runtime, SectorSpectrum, TensorMap, TensorScalar,
};

use crate::nonfinite_factor_input;
use nonfinite_factor_input::assert_finite_input_refusals;

fn leg() -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 2)],
    )
    .unwrap()
}

/// The identity with `bad` at one entry of the last coupled-sector block, so
/// a finite first block does not hide it.
fn dense_with<D: TensorScalar>(
    runtime: &Runtime,
    bad: D,
    one: D,
    zero: D,
) -> TensorMap<U1FusionRule, D> {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let leg = leg();
    TensorMap::from_subblock_fn(engine, runtime, [&leg], [&leg], |trees, index| {
        let last = *trees.coupled() == U1Irrep::new(1);
        match (last && index == [1, 0], index[0] == index[1]) {
            (true, _) => bad,
            (false, true) => one,
            (false, false) => zero,
        }
    })
    .unwrap()
}

fn diagonal_with<D: TensorScalar>(runtime: &Runtime, bad: D, one: D) -> TensorMap<U1FusionRule, D> {
    let engine = &mut symtn_dense::CpuEngine::serial();
    TensorMap::diagonal(
        engine,
        runtime,
        &leg(),
        [
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![one, one],
            },
            SectorSpectrum {
                sector: U1Irrep::new(1),
                values: vec![one, bad],
            },
        ],
    )
    .unwrap()
}

#[test]
fn multiplicity_free_factorizations_refuse_nonfinite_real_input() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let dense = dense_with(&runtime, bad, 1.0, 0.0);
        assert_finite_input_refusals(
            nonfinite_factor_errors!(engine, &dense),
            &format!("dense {bad}"),
        );
        let adjoint = dense.adjoint(engine).unwrap();
        assert_finite_input_refusals(
            nonfinite_factor_errors!(engine, &adjoint),
            &format!("lazy adjoint {bad}"),
        );
        let diagonal = diagonal_with(&runtime, bad, 1.0);
        assert_finite_input_refusals(
            nonfinite_factor_errors!(engine, &diagonal),
            &format!("diagonal {bad}"),
        );
    }
}

#[test]
fn multiplicity_free_factorizations_refuse_nonfinite_complex_input() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    for bad in [
        Complex64::new(f64::NAN, 0.0),
        Complex64::new(0.0, f64::INFINITY),
        Complex64::new(f64::NEG_INFINITY, 1.0),
    ] {
        let dense = dense_with(&runtime, bad, one, zero);
        assert_finite_input_refusals(
            nonfinite_factor_errors!(engine, &dense),
            &format!("dense {bad}"),
        );
        let adjoint = dense.adjoint(engine).unwrap();
        assert_finite_input_refusals(
            nonfinite_factor_errors!(engine, &adjoint),
            &format!("lazy adjoint {bad}"),
        );
        let diagonal = diagonal_with(&runtime, bad, one);
        assert_finite_input_refusals(
            nonfinite_factor_errors!(engine, &diagonal),
            &format!("diagonal {bad}"),
        );
    }
}

/// Structure is reported before values, as on the diagonal route: a
/// nonfinite non-endomorphism is refused as a non-endomorphism by eig/eigh,
/// and a nonfinite wide block by left polar as a shape error.
#[test]
fn structural_admission_precedes_the_finite_input_stage() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let row = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 1)]).unwrap();
    let col = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)]).unwrap();
    let wide: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &runtime, [&row], [&col], |_, _| f64::NAN).unwrap();
    let operation = |error: Error| match error {
        Error::Operation(operation) => *operation,
        other => panic!("expected an operation error, got {other:?}"),
    };
    let space_mismatch = |message| OperationError::SpaceMismatch { message };
    for (error, family) in [
        (
            wide.eigh_vals(engine, &[0], &[1], HermitianTol::DEFAULT)
                .unwrap_err(),
            "eigh",
        ),
        (
            wide.eigh_full(engine, &[0], &[1], HermitianTol::DEFAULT)
                .unwrap_err(),
            "eigh",
        ),
        (wide.eig_vals(engine, &[0], &[1]).unwrap_err(), "eig"),
        (wide.eig_full(engine, &[0], &[1]).unwrap_err(), "eig"),
    ] {
        let expected = match family {
            "eigh" => space_mismatch("eigh requires an endomorphism (codomain == domain)"),
            _ => space_mismatch("eig requires an endomorphism (codomain == domain)"),
        };
        assert_eq!(operation(error), expected);
    }
    assert_eq!(
        operation(wide.left_polar(engine, &[0], &[1]).unwrap_err()),
        OperationError::InvalidArgument {
            message: "left_polar requires rows >= columns in every coupled-sector matrix"
        }
    );
    // Families with no structural admission refuse the values.
    assert_eq!(
        operation(wide.svd_vals(engine, &[0], &[1]).unwrap_err()),
        OperationError::InvalidArgument {
            message: "svd input components must be finite"
        }
    );
}

/// Negative control: a finite value of extreme magnitude passes the input
/// stage, whatever the factorization then makes of it.
#[test]
fn finite_extreme_input_passes_the_finite_input_stage() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let runtime = Runtime::builder().build().unwrap();
    let extreme = Complex64::new(f64::MAX, f64::MAX);
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    for t in [
        dense_with(&runtime, extreme, one, zero),
        diagonal_with(&runtime, extreme, one),
    ] {
        let rows: &[usize] = &[0];
        let cols: &[usize] = &[1];
        let outcomes = [
            t.svd_vals(engine, rows, cols).map(drop),
            t.qr_compact(engine, rows, cols).map(drop),
            t.lq_compact(engine, rows, cols).map(drop),
            t.left_null(engine, rows, cols).map(drop),
            t.left_polar(engine, rows, cols).map(drop),
            t.eigh_vals(engine, rows, cols, HermitianTol::DEFAULT)
                .map(drop),
            t.eig_vals(engine, rows, cols).map(drop),
        ];
        for outcome in outcomes {
            if let Err(error) = outcome {
                let (message, text) = nonfinite_factor_input::refusal(&error);
                assert!(
                    !message.is_some_and(|message| message.ends_with("input components must be finite")),
                    "{text}"
                );
            }
        }
    }
}
