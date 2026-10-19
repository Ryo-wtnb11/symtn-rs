//! A NaN reaching a truncation or pseudo-inverse cutoff is a typed error, never
//! a silent finite decision (#1295).
//!
//! Neither reference defines this: MatrixAlgebraKit `findtruncated` /
//! `_truncerr_impl` compare against a NaN-poisoned norm and keep nothing, and
//! TensorKit `pinv(::DiagonalTensorMap)` maps a NaN entry to zero through
//! `pinv(x::Number)`. symtn rejects the spectrum instead. Each fixture puts the
//! NaN in the *second* sector so a fold that ignores NaN would still see a
//! finite maximum from the first.

use report::Report;

use std::sync::Arc;
use symtn::typed::HermitianTol;

use num_complex::Complex64;
use symtn::sector::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use symtn::typed::{Error, OperationError, TensorMap, Truncation};
use symtn::typed::{GradedSpace, SectorSpectrum};

#[path = "../../tests/support/fixtures.rs"]
mod fixtures;
#[path = "support/report.rs"]
mod report;
use report::cause;

use fixtures::host_runtime;

/// pinv's finite-input refusal, the same typed operation error on compact and
/// dense storage.
const PINV_NONFINITE: OperationError = OperationError::InvalidArgument {
    message: "pinv singular values must be finite",
};

fn policies() -> Vec<Truncation> {
    vec![
        Truncation::Full,
        Truncation::rank(2),
        Truncation::relative_cutoff(0.1).unwrap(),
        Truncation::relative_inf_cutoff(0.1).unwrap(),
        Truncation::relative_error(0.1).unwrap(),
        Truncation::rank(3).and(Truncation::relative_inf_cutoff(0.1).unwrap()),
    ]
}

macro_rules! assert_nan_spectrum_is_rejected {
    ($leg:expr, $sectors:expr, $scalar:expr) => {{
        let leg = &$leg;
        let [first, second] = $sectors;
        let spectra = vec![
            SectorSpectrum {
                sector: first,
                values: vec![$scalar(4.0), $scalar(1.0)],
            },
            SectorSpectrum {
                sector: second,
                values: vec![$scalar(f64::NAN), $scalar(0.5)],
            },
        ];
        for policy in policies() {
            assert!(
                matches!(leg.find_truncated(&spectra, &policy), Err(Error::Operation(error)) if matches!(*error, OperationError::InvalidArgument { message } if message.contains("finite") && message.contains("non-negative"))),
                "find_truncated accepted a NaN spectrum under {policy:?}"
            );
        }
        let diagonal = TensorMap::diagonal(&host_runtime(), leg, spectra).unwrap();
        for rcond in [0.0, 0.1] {
            assert!(
                matches!(
                    diagonal.pinv(&[0], &[1], rcond),
                    Err(Error::Operation(error)) if *error == PINV_NONFINITE
                ),
                "compact pinv hid a NaN entry at rcond {rcond}"
            );
        }
        // The dense routes: a NaN payload never yields a finite factor either.
        // The truncated compositions fail at the factorization or at the
        // decision; neither publishes a factor.
        for policy in policies() {
            let svd =
                diagonal
                    .svd_compact(&[0], &[1])
                    .and_then(|symtn::typed::Svd { s, .. }| {
                        s.domain()[0].find_truncated(&s.diagview()?, &policy)
                    });
            assert!(svd.err().unwrap().report().contains("svd input components must be finite"), "svd composition {policy:?}");
            let eigh = diagonal
                .eigh_full(&[0], &[1], HermitianTol::DEFAULT)
                .and_then(|symtn::typed::Eigh { d, .. }| {
                    d.domain()[0].find_truncated(&d.diagview()?, &policy)
                });
            assert!(eigh.err().unwrap().report().contains("eigh input components must be finite"), "eigh composition {policy:?}");
        }
    }};
}

fn real(value: f64) -> f64 {
    value
}

fn complex(value: f64) -> Complex64 {
    Complex64::new(value, 0.0)
}

fn u1() -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 2), (U1Irrep::new(1), 2)],
    )
    .unwrap()
}

fn su2() -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap()
}

#[test]
fn u1_real_nan_spectrum_is_rejected() {
    assert_nan_spectrum_is_rejected!(u1(), [U1Irrep::new(0), U1Irrep::new(1)], real);
}

#[test]
fn u1_complex_nan_spectrum_is_rejected() {
    assert_nan_spectrum_is_rejected!(u1(), [U1Irrep::new(0), U1Irrep::new(1)], complex);
}

#[test]
fn su2_real_nan_spectrum_is_rejected() {
    assert_nan_spectrum_is_rejected!(
        su2(),
        [SU2Irrep::from_twice_spin(0), SU2Irrep::from_twice_spin(1)],
        real
    );
}

#[test]
fn su2_complex_nan_spectrum_is_rejected() {
    assert_nan_spectrum_is_rejected!(
        su2(),
        [SU2Irrep::from_twice_spin(0), SU2Irrep::from_twice_spin(1)],
        complex
    );
}

#[test]
fn compact_svd_rejects_infinite_input_without_changing_spectrum() {
    // The shared finite-input stage refuses a nonfinite diagonal before any
    // work, as the dense route refuses nonfinite input (#1986); the input is
    // unchanged.
    let input: TensorMap<_, f64> = TensorMap::diagonal(
        &host_runtime(),
        &u1(),
        [
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![1.0, -2.0],
            },
            SectorSpectrum {
                sector: U1Irrep::new(1),
                values: vec![f64::INFINITY, 0.0],
            },
        ],
    )
    .unwrap();
    let before = input.diagview().unwrap();
    let error = input.svd_compact(&[0], &[1]).err().unwrap();
    assert_eq!(
        cause::<OperationError>(&error),
        Some(&OperationError::InvalidArgument {
            message: "svd input components must be finite"
        }),
        "{error:?}"
    );
    assert_eq!(input.diagview().unwrap(), before);
}

#[test]
fn compact_pinv_of_a_finite_diagonal_is_unchanged() {
    // Hand-computed: sigma_max = 4, rcond 0.2 -> cutoff 0.8; 0.5 is cut.
    let diagonal: TensorMap<_, f64> = TensorMap::diagonal(
        &host_runtime(),
        &u1(),
        [
            SectorSpectrum {
                sector: U1Irrep::new(0),
                values: vec![4.0, 1.0],
            },
            SectorSpectrum {
                sector: U1Irrep::new(1),
                values: vec![2.0, 0.5],
            },
        ],
    )
    .unwrap();
    let image = symtn::expert::diagonal_spectrum(&diagonal.pinv(&[0], &[1], 0.2).unwrap())
        .unwrap()
        .unwrap();
    assert_eq!(image[0].values, vec![0.25, 1.0]);
    assert_eq!(image[1].values, vec![0.5, 0.0]);
}

/// A NaN tensor through the dense multiplicity-free pinv routes (owned and
/// lazy adjoint): the finite-input stage refuses it before any provider SVD,
/// so every provider answers the compact route's typed operation error.
#[test]
fn dense_pinv_of_a_nan_tensor_is_invalid_argument() {
    let runtime = host_runtime();
    let leg = u1();
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let real: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |trees, index: &[usize]| {
                if *trees.coupled() == U1Irrep::new(1) && index == [0, 0] {
                    bad
                } else if index[0] == index[1] {
                    2.0
                } else {
                    0.5
                }
            })
            .unwrap();
        let complex = real.convert::<Complex64>();
        for (case, result) in [
            ("f64 owned", real.pinv(&[0], &[1], 0.1).map(|_| ())),
            (
                "f64 adjoint",
                real.adjoint().unwrap().pinv(&[0], &[1], 0.1).map(|_| ()),
            ),
            ("c64 owned", complex.pinv(&[0], &[1], 0.1).map(|_| ())),
            (
                "c64 adjoint",
                complex.adjoint().unwrap().pinv(&[0], &[1], 0.1).map(|_| ()),
            ),
        ] {
            match result {
                Err(Error::Operation(error)) => assert_eq!(*error, PINV_NONFINITE, "{case}"),
                other => panic!("{case}: expected the finite-input refusal, got {other:?}"),
            }
        }
        // rcond validation precedes the finite scan.
        match real.pinv(&[0], &[1], -1.0) {
            Err(Error::InvalidArgument(message)) => {
                assert!(message.contains("rcond"), "{message}")
            }
            other => panic!("expected the rcond error, got {:?}", other.map(|_| ())),
        }
    }
}
