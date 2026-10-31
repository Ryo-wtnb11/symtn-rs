//! Shared ULP-distance helpers for tests that compare a computed `f64`
//! against a correctly-rounded (or otherwise independently derived) oracle
//! value, rather than requiring bitwise equality.
//!
//! Used by `typed_facade.rs`'s
//! `c64_compact_inv_and_pinv_are_elementwise_reciprocals`, whose
//! exact equality assumption does not hold because `Complex64`'s compact
//! reciprocal runs the Julia-ported `inv` (`symtn/src/typed.rs`
//! `julia_complex64_reciprocal`) instead of the naive `1/z` it compares
//! against: the new fast path's `mul_add` gives a different, still valid,
//! rounding for the same normal-range value.

use num_complex::Complex64;

/// Maps an `f64` to a total-ordered `i64` key (Bruce Dawson's
/// "AlmostEqualUlps" trick): adjacent representable `f64` values map to
/// adjacent `i64` keys, with `+0.0` and `-0.0` both mapping to `0`.
fn ordered_key(x: f64) -> i64 {
    let bits = x.to_bits() as i64;
    if bits < 0 {
        i64::MIN.wrapping_sub(bits)
    } else {
        bits
    }
}

/// Distance in ULPs between two finite `f64` values. `NaN` is not comparable
/// and is the caller's responsibility to check separately.
pub fn ulps(a: f64, b: f64) -> u64 {
    debug_assert!(a.is_finite() && b.is_finite(), "ulps() needs finite input");
    ordered_key(a).abs_diff(ordered_key(b))
}

/// Asserts `a` and `b` are within `max_ulps` of each other, treating
/// `+/-0.0` as equal and requiring both non-finite inputs to match exactly
/// (same infinity, or both `NaN`).
pub fn assert_within_ulps(a: f64, b: f64, max_ulps: u64, context: &str) {
    match (a.is_finite(), b.is_finite()) {
        (true, true) => {
            let distance = ulps(a, b);
            assert!(
                distance <= max_ulps,
                "{context}: {a:e} and {b:e} are {distance} ulps apart (limit {max_ulps})"
            );
        }
        (false, false) => {
            assert!(
                (a.is_nan() && b.is_nan()) || a == b,
                "{context}: non-finite mismatch {a:e} vs {b:e}"
            );
        }
        _ => panic!("{context}: finiteness mismatch {a:e} vs {b:e}"),
    }
}

/// Component-wise [`assert_within_ulps`] for a complex pair.
pub fn assert_complex_within_ulps(a: Complex64, b: Complex64, max_ulps: u64, context: &str) {
    assert_within_ulps(a.re, b.re, max_ulps, &format!("{context} (re)"));
    assert_within_ulps(a.im, b.im, max_ulps, &format!("{context} (im)"));
}
