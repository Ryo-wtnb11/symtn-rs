//! Non-finite payloads, one table per operation family, over dense, lazy
//! adjoint and compact diagonal storage, `f64` and `Complex64`, U(1) and
//! SU(2). Every fixture puts the non-finite component in the *last*
//! coupled sector, so a fold that skipped it would still see finite values
//! from the earlier blocks.
//!
//! - `norm(p)` follows Julia, which TensorKit `cfaa073e` reaches through
//!   `LinearAlgebra.norm` / `_norm` (`src/tensors/linalg.jl:257-282`): a NaN
//!   gives NaN for every `p`, `±Inf` gives `Inf`, an `Inf` beside a NaN gives
//!   NaN, and `|NaN + Inf i| = hypot(NaN, Inf) = Inf`. QSpace has no
//!   non-finite policy (`wbarray<T>::norm2` is an unscaled `Σ x²`).
//! - Every factorization refuses non-finite input with its family's
//!   `"<family> input components must be finite"` through the shared
//!   finite-input stage; structure is reported before values.
//! - `pinv` refuses non-finite singular values (`pinv singular values must
//!   be finite`) after validating `rcond`.
//! - `find_truncated` refuses a NaN spectrum under every policy.
//!
//! Neither reference defines the last three: MatrixAlgebraKit `findtruncated`
//! compares against a NaN-poisoned norm and keeps nothing, and TensorKit
//! `pinv(::DiagonalTensorMap)` maps a NaN entry to zero. symtn returns a
//! typed error instead of a silent finite decision. The checked Generic
//! factorization column is `checked_generic_facade/nonfinite.rs`.

use symtn::typed::HermitianTol;

use num_complex::Complex64;
use symtn::sector::U1Irrep;
use symtn::typed::{Error, OperationError, SectorSpectrum, TensorMap, Truncation};

use crate::fixtures::host_runtime;
use crate::nonfinite_factor_input;
use nonfinite_factor_input::assert_finite_input_refusals;

use crate::legs::{su2_leg_2_2, u1_leg_2_2};

/// A finite entry of the dense fixtures (row `i`, column `j`).
fn finite(i: usize, j: usize) -> Complex64 {
    Complex64::new(
        1.0 + 0.5 * i as f64 - 0.25 * j as f64,
        0.5 + 0.125 * i as f64 + 0.375 * j as f64,
    )
}

/// The three storages of one endomorphism `[leg] <- [leg]` whose last
/// coupled sector carries `first` at `(0, 0)` and `second` at `(1, 1)`:
/// the dense tensor, its lazy adjoint (the same entries, conjugated and
/// transposed) and the compact diagonal holding the same diagonal.
macro_rules! storages {
    ($engine:ident, $leg:expr, $D:ty, $conv:expr, $first:expr, $second:expr) => {{
        let runtime = host_runtime();
        let leg = $leg;
        let conv = $conv;
        let (first, second): (Complex64, Complex64) = ($first, $second);
        let last = leg.sectors().unwrap().into_iter().last().unwrap();
        let value = |is_last: bool, i: usize, j: usize| -> $D {
            match (is_last, i, j) {
                (true, 0, 0) => conv(first),
                (true, 1, 1) => conv(second),
                _ => conv(finite(i, j)),
            }
        };
        let dense =
            TensorMap::<_, $D>::from_subblock_fn($engine, &runtime, [&leg], [&leg], |trees, ix| {
                value(*trees.coupled() == last, ix[0], ix[1])
            })
            .unwrap();
        let lazy = dense.adjoint($engine).unwrap();
        let compact = TensorMap::<_, $D>::diagonal(
            $engine,
            &runtime,
            &leg,
            leg.sectors().unwrap().into_iter().map(|sector| {
                let is_last = sector == last;
                SectorSpectrum {
                    values: (0..leg.degeneracy(&sector).unwrap())
                        .map(|i| value(is_last, i, i))
                        .collect(),
                    sector,
                }
            }),
        )
        .unwrap();
        [
            ("dense", dense),
            ("lazy adjoint", lazy),
            ("compact", compact),
        ]
    }};
}

// ---- norm(p) ----

/// One poisoned payload: the components written at `(0, 0)` and `(1, 1)` of
/// the last block, and the norm Julia returns for it at every `p`.
struct NormCase {
    what: &'static str,
    first: Complex64,
    second: Complex64,
    complex_only: bool,
    norm: f64,
}

const NAN: f64 = f64::NAN;
const INF: f64 = f64::INFINITY;

fn norm_cases() -> [NormCase; 6] {
    let c = Complex64::new;
    let one = finite(1, 1);
    [
        NormCase {
            what: "NaN",
            first: c(NAN, 0.0),
            second: one,
            complex_only: false,
            norm: NAN,
        },
        NormCase {
            what: "Inf",
            first: c(INF, 0.0),
            second: one,
            complex_only: false,
            norm: INF,
        },
        NormCase {
            what: "-Inf",
            first: c(-INF, 0.0),
            second: one,
            complex_only: false,
            norm: INF,
        },
        NormCase {
            what: "Inf beside NaN",
            first: c(INF, 0.0),
            second: c(NAN, 0.0),
            complex_only: false,
            norm: NAN,
        },
        // Only `|z|` sees a NaN imaginary part beside a finite real part.
        NormCase {
            what: "1 + NaN i",
            first: c(1.0, NAN),
            second: one,
            complex_only: true,
            norm: NAN,
        },
        NormCase {
            what: "NaN + Inf i",
            first: c(NAN, INF),
            second: one,
            complex_only: true,
            norm: INF,
        },
    ]
}

macro_rules! norm_cells {
    ($engine:ident, $label:expr, $leg:expr, $D:ty, $conv:expr, $complex:expr) => {{
        for case in norm_cases() {
            if case.complex_only && !$complex {
                continue;
            }
            for (storage, t) in storages!($engine, $leg, $D, $conv, case.first, case.second) {
                for p in [1.0, 2.0, 3.0, f64::INFINITY] {
                    let got = t.norm($engine, p).unwrap();
                    let agrees = if case.norm.is_nan() {
                        got.is_nan()
                    } else {
                        got == case.norm
                    };
                    assert!(
                        agrees,
                        "{} {storage} {}: norm({p}) = {got}, Julia gives {}",
                        $label, case.what, case.norm
                    );
                }
            }
        }
    }};
}

#[test]
fn norm_of_a_non_finite_payload_follows_julia_on_every_storage() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let real = |z: Complex64| z.re;
    let complex = |z: Complex64| z;
    norm_cells!(engine, "U(1) f64", u1_leg_2_2(), f64, real, false);
    norm_cells!(engine, "U(1) c64", u1_leg_2_2(), Complex64, complex, true);
    norm_cells!(engine, "SU(2) f64", su2_leg_2_2(), f64, real, false);
    norm_cells!(engine, "SU(2) c64", su2_leg_2_2(), Complex64, complex, true);
}

// ---- factorizations and pinv ----

fn real_bads() -> [Complex64; 3] {
    [NAN, INF, -INF].map(|x| Complex64::new(x, 0.0))
}

fn complex_bads() -> [Complex64; 4] {
    [
        Complex64::new(NAN, 0.0),
        Complex64::new(INF, 0.0),
        Complex64::new(0.0, INF),
        Complex64::new(-INF, 1.0),
    ]
}

/// pinv's finite-input refusal, the same typed operation error on every
/// storage.
const PINV_NONFINITE: OperationError = OperationError::InvalidArgument {
    message: "pinv singular values must be finite",
};

macro_rules! factor_and_pinv_cells {
    ($engine:ident, $label:expr, $leg:expr, $D:ty, $conv:expr, $bads:expr) => {{
        for bad in $bads {
            let one = finite(1, 1);
            for (storage, t) in storages!($engine, $leg, $D, $conv, bad, one) {
                let case = format!("{} {storage} {bad}", $label);
                assert_finite_input_refusals(nonfinite_factor_errors!($engine, &t), &case);
                for rcond in [0.0, 0.1] {
                    match t.pinv($engine, &[0], &[1], rcond) {
                        Err(Error::Operation(error)) => {
                            assert_eq!(*error, PINV_NONFINITE, "{case} pinv rcond {rcond}")
                        }
                        other => panic!(
                            "{case} pinv rcond {rcond}: expected the finite-input refusal, got {:?}",
                            other.map(drop)
                        ),
                    }
                }
                // `rcond` is validated before the singular values are scanned.
                match t.pinv($engine, &[0], &[1], -1.0) {
                    Err(Error::InvalidArgument(message)) => {
                        assert!(message.contains("rcond"), "{case}: {message}")
                    }
                    other => panic!("{case}: expected the rcond error, got {:?}", other.map(drop)),
                }
            }
        }
    }};
}

#[test]
fn factorizations_and_pinv_refuse_non_finite_input_on_every_storage() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let real = |z: Complex64| z.re;
    let complex = |z: Complex64| z;
    factor_and_pinv_cells!(engine, "U(1) f64", u1_leg_2_2(), f64, real, real_bads());
    factor_and_pinv_cells!(
        engine,
        "U(1) c64",
        u1_leg_2_2(),
        Complex64,
        complex,
        complex_bads()
    );
    factor_and_pinv_cells!(engine, "SU(2) f64", su2_leg_2_2(), f64, real, real_bads());
    factor_and_pinv_cells!(
        engine,
        "SU(2) c64",
        su2_leg_2_2(),
        Complex64,
        complex,
        complex_bads()
    );
}

/// Negative control: a finite value of extreme magnitude passes the input
/// stage, whatever the factorization then makes of it.
#[test]
fn finite_extreme_input_passes_the_finite_input_stage() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let extreme = Complex64::new(f64::MAX, f64::MAX);
    let complex = |z: Complex64| z;
    for (storage, t) in storages!(
        engine,
        u1_leg_2_2(),
        Complex64,
        complex,
        extreme,
        finite(1, 1)
    ) {
        if storage == "lazy adjoint" {
            continue;
        }
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
                    "{storage}: {text}"
                );
            }
        }
    }
}

/// Negative control for the pinv cells, hand-computed: `σ_max = 4`,
/// `rcond 0.2` gives the cutoff 0.8, so 0.5 is cut.
#[test]
fn compact_pinv_of_a_finite_diagonal_inverts_above_the_cutoff() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let diagonal: TensorMap<_, f64> = TensorMap::diagonal(
        engine,
        &host_runtime(),
        &u1_leg_2_2(),
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
    let diagonal_pinv = &diagonal.pinv(engine, &[0], &[1], 0.2).unwrap();
    let image = symtn::expert::diagonal_spectrum(engine, diagonal_pinv)
        .unwrap()
        .unwrap();
    assert_eq!(image[0].values, vec![0.25, 1.0]);
    assert_eq!(image[1].values, vec![0.5, 0.0]);
}

// ---- find_truncated ----

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

macro_rules! truncation_cells {
    ($label:expr, $leg:expr, $conv:expr) => {{
        let leg = $leg;
        let conv = $conv;
        let sectors = leg.sectors().unwrap();
        // The NaN sits in the second sector, after a finite maximum.
        let spectra: Vec<_> = sectors
            .into_iter()
            .enumerate()
            .map(|(position, sector)| SectorSpectrum {
                sector,
                values: if position == 0 {
                    vec![conv(4.0), conv(1.0)]
                } else {
                    vec![conv(f64::NAN), conv(0.5)]
                },
            })
            .collect();
        for policy in policies() {
            assert!(
                matches!(
                    leg.find_truncated(&spectra, &policy),
                    Err(Error::Operation(error)) if matches!(
                        *error,
                        OperationError::InvalidArgument { message }
                            if message.contains("finite") && message.contains("non-negative")
                    )
                ),
                "{}: find_truncated accepted a NaN spectrum under {policy:?}",
                $label
            );
        }
    }};
}

#[test]
fn find_truncated_refuses_a_nan_spectrum_under_every_policy() {
    let real = |x: f64| x;
    let complex = |x: f64| Complex64::new(x, 0.0);
    truncation_cells!("U(1) f64", u1_leg_2_2(), real);
    truncation_cells!("U(1) c64", u1_leg_2_2(), complex);
    truncation_cells!("SU(2) f64", su2_leg_2_2(), real);
    truncation_cells!("SU(2) c64", su2_leg_2_2(), complex);
}
