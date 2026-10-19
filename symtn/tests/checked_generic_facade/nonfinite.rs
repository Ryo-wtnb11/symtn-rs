use super::*;

#[macro_use]
#[path = "../support/nonfinite_factor_input.rs"]
mod nonfinite_factor_input;
use nonfinite_factor_input::assert_finite_input_refusals;

/// Checked Generic factorizations refuse nonfinite input through the shared
/// finite-input stage (#1986), dense and compact diagonal, real and complex;
/// the multiplicity-free counterpart is `factorization_nonfinite_input.rs`.
#[test]
fn checked_generic_factorizations_refuse_nonfinite_input() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(Label::Vacuum, 2), (Label::X, 2)]).unwrap();
    // The bad entry sits in the last block, so a finite first block does not
    // hide it.
    fn cases<D: symtn::typed::TensorScalar>(
        runtime: &Runtime,
        leg: &GradedSpace<CheckedOnlyToy>,
        bad: D,
        one: D,
        zero: D,
    ) -> [TensorMap<CheckedOnlyToy, D>; 2] {
        let dense = TensorMap::from_subblock_fn(runtime, [leg], [leg], |trees, index| {
            match (
                *trees.coupled() == Label::X && index == [1, 0],
                index[0] == index[1],
            ) {
                (true, _) => bad,
                (false, true) => one,
                (false, false) => zero,
            }
        })
        .unwrap();
        let diagonal = TensorMap::diagonal(
            runtime,
            leg,
            [
                SectorSpectrum {
                    sector: Label::Vacuum,
                    values: vec![one, one],
                },
                SectorSpectrum {
                    sector: Label::X,
                    values: vec![one, bad],
                },
            ],
        )
        .unwrap();
        [dense, diagonal]
    }
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let [dense, diagonal] = cases(&runtime, &leg, bad, 1.0, 0.0);
        for (storage, t) in [("dense", &dense), ("diagonal", &diagonal)] {
            assert_finite_input_refusals(nonfinite_factor_errors!(t), &format!("{storage} {bad}"));
        }
        // The lazy adjoints checked mode accepts: polar reads the parent,
        // null runs the opposite null space of the parent.
        let adjoint = dense.adjoint().unwrap();
        let errors = vec![
            (
                "polar",
                nonfinite_factor_input::refusal(&adjoint.left_polar(&[0], &[1]).unwrap_err()),
            ),
            (
                "polar",
                nonfinite_factor_input::refusal(&adjoint.right_polar(&[0], &[1]).unwrap_err()),
            ),
            (
                "null",
                nonfinite_factor_input::refusal(&adjoint.left_null(&[0], &[1]).unwrap_err()),
            ),
            (
                "null",
                nonfinite_factor_input::refusal(&adjoint.right_null(&[0], &[1]).unwrap_err()),
            ),
        ];
        assert_finite_input_refusals(errors, &format!("lazy adjoint {bad}"));
    }
    for bad in [
        Complex64::new(f64::NAN, 0.0),
        Complex64::new(0.0, f64::INFINITY),
        Complex64::new(f64::NEG_INFINITY, 1.0),
    ] {
        let (one, zero) = (Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0));
        for (storage, t) in ["dense", "diagonal"]
            .into_iter()
            .zip(cases(&runtime, &leg, bad, one, zero))
        {
            assert_finite_input_refusals(nonfinite_factor_errors!(&t), &format!("{storage} {bad}"));
        }
    }
}

/// The polar block-shape check precedes the finite-input stage in checked
/// mode too.
#[test]
fn checked_generic_polar_reports_shape_before_values() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(CheckedOnlyToy::new(0));
    let row = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 1)]).unwrap();
    let col = GradedSpace::try_new(Arc::clone(&provider), [(Label::X, 2)]).unwrap();
    let wide: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&row], [&col], |_, _| f64::NAN).unwrap();
    let (message, error) =
        nonfinite_factor_input::refusal(&wide.left_polar(&[0], &[1]).unwrap_err());
    assert_eq!(
        message,
        Some("left_polar requires rows >= columns in every coupled-sector matrix"),
        "{error}"
    );
}
