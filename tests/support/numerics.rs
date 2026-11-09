//! The workspace tolerance rule for arithmetic test comparisons
//! (`docs/testing_numerics.md`).
//!
//! Test targets include this file with `#[path = ".../tests/support/numerics.rs"]`
//! so every crate shares one rule without a dev-dependency. The including
//! module must have `Complex32` and `Complex64` in scope.
//!
//! The bound is `K * sqrt(terms) * eps(dtype) * max(1, scale)`, the rule of
//! `symtn/tests/single_precision_oracle/mod.rs`: `terms` is the number of
//! floating terms that reach one compared entry, `scale` the largest oracle
//! magnitude. A call site that needs a conditioning factor states its origin.

#![allow(dead_code)]

use super::{Complex32, Complex64};

/// The bits a bit-exact comparison checks: every NaN maps to one value,
/// because NaN sign and payload are unspecified through Rust arithmetic and
/// are not part of the library contract. Every other bit, the sign of zero
/// included, is kept; an `f32` widens exactly.
pub fn nan_folded_bits(value: impl Into<f64>) -> u64 {
    let value = value.into();
    if value.is_nan() {
        f64::NAN.to_bits()
    } else {
        value.to_bits()
    }
}

/// `K` of the `K * sqrt(terms) * eps * scale` bound.
pub const K: f64 = 32.0;

/// A payload dtype the rule applies to: its unit roundoff and its value as a
/// wide complex number.
pub trait Numeric: Copy + std::fmt::Debug {
    const EPS: f64;
    fn wide(self) -> Complex64;
}

impl Numeric for f32 {
    const EPS: f64 = f32::EPSILON as f64;
    fn wide(self) -> Complex64 {
        Complex64::new(self.into(), 0.0)
    }
}

impl Numeric for f64 {
    const EPS: f64 = f64::EPSILON;
    fn wide(self) -> Complex64 {
        Complex64::new(self, 0.0)
    }
}

impl Numeric for Complex32 {
    const EPS: f64 = f32::EPSILON as f64;
    fn wide(self) -> Complex64 {
        Complex64::new(self.re.into(), self.im.into())
    }
}

impl Numeric for Complex64 {
    const EPS: f64 = f64::EPSILON;
    fn wide(self) -> Complex64 {
        self
    }
}

/// `K * sqrt(terms) * eps(T) * max(1, scale)`.
pub fn tolerance<T: Numeric>(terms: usize, scale: f64) -> f64 {
    K * (terms.max(1) as f64).sqrt() * T::EPS * scale.max(1.0)
}

/// Asserts `|got - want| <= tolerance(terms, |want|)`.
#[track_caller]
pub fn assert_close<T: Numeric>(what: &str, got: T, want: T, terms: usize) {
    assert_close_scaled(what, got, want, terms, 1.0);
}

/// [`assert_close`] with a conditioning factor multiplying the bound.
#[track_caller]
pub fn assert_close_scaled<T: Numeric>(
    what: &str,
    got: T,
    want: T,
    terms: usize,
    conditioning: f64,
) {
    let bound = tolerance::<T>(terms, want.wide().norm()) * conditioning;
    let error = (got.wide() - want.wide()).norm();
    assert!(
        error <= bound,
        "{what}: {got:?} against the oracle {want:?} (error {error:e} > tolerance {bound:e})"
    );
}

/// Asserts equal lengths and every entry within `tolerance(terms, max|want|)`.
#[track_caller]
pub fn assert_slices_close<T: Numeric>(what: &str, got: &[T], want: &[T], terms: usize) {
    assert_slices_close_scaled(what, got, want, terms, 1.0);
}

/// [`assert_slices_close`] with a conditioning factor multiplying the bound.
#[track_caller]
pub fn assert_slices_close_scaled<T: Numeric>(
    what: &str,
    got: &[T],
    want: &[T],
    terms: usize,
    conditioning: f64,
) {
    assert_eq!(got.len(), want.len(), "{what}: payload lengths differ");
    let scale = want.iter().map(|v| v.wide().norm()).fold(0.0, f64::max);
    let bound = tolerance::<T>(terms, scale) * conditioning;
    for (index, (&g, &w)) in got.iter().zip(want).enumerate() {
        let error = (g.wide() - w.wide()).norm();
        assert!(
            error <= bound,
            "{what}: entry {index} is {g:?} against the oracle {w:?} \
             (error {error:e} > tolerance {bound:e})"
        );
    }
}

/// [`assert_slices_close`] for an oracle that must not be all zero: against
/// an all-zero oracle a bounded error proves nothing.
#[track_caller]
pub fn assert_nonzero_slices_close<T: Numeric>(what: &str, got: &[T], want: &[T], terms: usize) {
    assert!(
        want.iter().any(|value| value.wide().norm() > 0.0),
        "{what}: an all-zero oracle proves nothing"
    );
    assert_slices_close(what, got, want, terms);
}
