//! Truncation norms and decisions stay correct when the spectrum's squares
//! leave the `f64` range.
//!
//! Fixture: SU(2) `j = 0` (dim 1) with `[4, 2] * s` and `j = 1/2` (dim 2) with
//! `[2, 1] * s`. By hand, `norm^2 = (16 + 4 + 2 * (4 + 1)) s^2 = 30 s^2`, and
//! the two `2 s` values tie across sectors; TensorKit's `isless` puts `j = 0`
//! first, so `rank` keeps it first and `relative_error` discards it
//! first.
//!
//! - `rank(3)`: keep `4, 2` of `j = 0` (weight 2); the tied `2` of `j = 1/2`
//!   would need 4. Kept `[2, 0]`, error `sqrt(2 * 5) s = sqrt(10) s`.
//! - `relative_cutoff(0.3)`: threshold `0.3 sqrt(30) s ≈ 1.64 s` drops only
//!   the `1`. Kept `[2, 1]`, error `sqrt(2) s`.
//! - `relative_error(0.5)`: budget `7.5 s^2`; ascending weighted squares are
//!   `2` (the `1`), then `4` (the tied `2` of `j = 0`) for `6`, then `8` for
//!   `14`. Kept `[1, 1]`, error `sqrt(6) s`.
//!
//! TensorKit 0.17.1 / MatrixAlgebraKit 0.6.9 on the same `SectorVector`
//! (Julia 1.11.6) agrees at `s = 1` (`truncrank(3)` error
//! `3.1622776601683795`, `truncerror(atol = sqrt(7.5))` keeps `[1, 1]` with
//! error `2.4494897427831783`, `trunctol(atol = 0.3 sqrt(30))` error
//! `1.4142135623730951`). Its `truncation_error!` -> `norm(::SectorVector)` ->
//! `_norm` adds `dim(c) * norm(b)^2` unscaled, so at `s = 1e200` it reports
//! `norm = Inf`, every error `Inf`, and `truncerror` discards everything; at
//! `s = 1e-200` it reports `norm = 0`, every error `0`, and `truncerror`
//! again discards everything. symtn returns the representable values.

use std::sync::Arc;

use symtn::sector::{SU2FusionRule, SU2Irrep};
use symtn::typed::Truncation;
use symtn::typed::{GradedSpace, SectorSpectrum};

use crate::numerics;

fn spin(twice: usize) -> SU2Irrep {
    SU2Irrep::from_twice_spin(twice)
}

/// Kept counts in `[j = 0, j = 1/2]` order and the reported error.
fn find<V>(values: [[V; 2]; 2], truncation: &Truncation) -> (Vec<usize>, f64)
where
    V: symtn::typed::SpectrumMagnitude + Copy,
{
    let sectors = [spin(0), spin(1)];
    let leg = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        sectors.iter().map(|sector| (*sector, 2)),
    )
    .unwrap();
    let spectra: Vec<SectorSpectrum<SU2Irrep, V>> = sectors
        .iter()
        .zip(values)
        .map(|(sector, values)| SectorSpectrum {
            sector: *sector,
            values: values.to_vec(),
        })
        .collect();
    let found = leg.find_truncated(&spectra, truncation).unwrap();
    let kept = sectors
        .iter()
        .map(|sector| found.selection.subspace().degeneracy(sector).unwrap())
        .collect();
    (kept, found.error)
}

fn cases() -> [(Truncation, [usize; 2], f64); 3] {
    [
        (Truncation::rank(3), [2, 0], 10.0),
        (Truncation::relative_cutoff(0.3).unwrap(), [2, 1], 2.0),
        (Truncation::relative_error(0.5).unwrap(), [1, 1], 6.0),
    ]
}

/// The error is compared as `error / s` against `sqrt(squared)`: the absolute
/// tolerance of `numerics` would accept `0` at `s = 1e-200`. Four values reach
/// the error.
fn check(kept: Vec<usize>, error: f64, s: f64, want: ([usize; 2], f64), what: &str) {
    assert_eq!(kept, want.0, "{what}: kept counts");
    numerics::assert_close(&format!("{what}: error / s"), error / s, want.1.sqrt(), 4);
}

#[test]
fn f64_spectra_near_the_overflow_and_underflow_bounds_keep_the_in_range_decision() {
    for s in [1.0, 1e200, 1e-200, 1e300, 1e-300] {
        for (truncation, kept, squared) in cases() {
            let (got, error) = find([[4.0 * s, 2.0 * s], [2.0 * s, s]], &truncation);
            check(
                got,
                error,
                s,
                (kept, squared),
                &format!("{truncation:?} s {s:e}"),
            );
        }
    }
}

#[test]
fn f32_spectra_at_the_single_precision_range_keep_the_in_range_decision() {
    // Single-precision spectra are widened to `f64` before the decision, so
    // `1e±30` is in range there; this pins that the widening path agrees.
    for s in [1.0_f32, 1e30, 1e-30] {
        for (truncation, kept, squared) in cases() {
            let (got, error) = find([[4.0 * s, 2.0 * s], [2.0 * s, s]], &truncation);
            check(
                got,
                error,
                f64::from(s),
                (kept, squared),
                &format!("{truncation:?} f32 s {s:e}"),
            );
        }
    }
}
