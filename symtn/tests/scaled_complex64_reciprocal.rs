//! The `Complex64` compact reciprocal (symtn#1463): `inv`/`pinv` of a
//! multiplicity-free compact (diagonal) `Complex64` tensor apply
//! `ScalarOps::recip_value` elementwise. Before this fix that was the naive
//! `1.0 / z`, i.e. `(c - d*i) / (c*c + d*d)`: squaring both components
//! overflows `f64` for `|z| >=~ 2^512` and flushes the denominator to zero
//! for `|z| <=~ 2^-511`, although `1/z` is finite and representable across
//! that whole range.
//!
//! symtn's semantics reference for `inv` is TensorKit, which delegates a
//! diagonal entry's reciprocal to Julia's own scalar `inv`.
//! `julia_complex64_reciprocal` (`symtn/src/typed/numeric.rs`) is a literal
//! port of Julia 1.11.6's `Base.inv(w::ComplexF64)` (`base/complex.jl:479`)
//! and its `robust_cinv` fallback (`base/complex.jl:508`);
//! `julia_complex32_reciprocal_wide` ports `inv(z::Complex{Float32})`
//! (`base/complex.jl:473`).
//!
//! Oracles: (1) a few literal rows, either exact powers of two whose
//! reciprocal is written as a formula, or bit patterns checked against Julia
//! 1.11.6 (aarch64) running `inv` on the same input; (2) identities every
//! correct reciprocal satisfies: `inv(2^k z) = 2^-k inv(z)`,
//! `inv(conj z) = conj(inv z)`, and `z * inv(z) ≈ 1`.

use crate::report;
use report::Report;

use std::f64::consts::{E, FRAC_1_SQRT_2, LN_2, PI, SQRT_2};
use std::sync::Arc;

use num_complex::{Complex32, Complex64};
use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{GradedSpace, Runtime, SectorSpectrum, TensorMap};

fn runtime() -> Runtime {
    Runtime::builder().build().expect("runtime builds")
}

fn leg(dim: usize) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), dim)]).unwrap()
}

fn bits(z: Complex64) -> (u64, u64) {
    (z.re.to_bits(), z.im.to_bits())
}

fn bits32(z: Complex32) -> (u32, u32) {
    (z.re.to_bits(), z.im.to_bits())
}

/// Exact `2^k` for every `k` in `f64`'s range, subnormals included
/// (`powi` would round through an intermediate `inf`/`0`).
fn p2(k: i32) -> f64 {
    if k >= -1022 {
        f64::from_bits(((k + 1023) as u64) << 52)
    } else {
        f64::from_bits(1u64 << (k + 1074))
    }
}

/// Exact `2^k` for every `k` in `f32`'s range, subnormals included.
fn p2_32(k: i32) -> f32 {
    if k >= -126 {
        f32::from_bits(((k + 127) as u32) << 23)
    } else {
        f32::from_bits(1u32 << (k + 149))
    }
}

fn c(re: f64, im: f64) -> Complex64 {
    Complex64::new(re, im)
}

fn diagonal<T>(value: T) -> TensorMap<U1FusionRule, T>
where
    T: symtn::typed::TensorScalar,
{
    TensorMap::diagonal(
        &runtime(),
        &leg(1),
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![value],
        }],
    )
    .unwrap()
}

/// The single stored entry of `t.inv(&[0], &[1])` for a rank-`(1,1)`
/// compact diagonal tensor holding exactly `value`.
fn compact_inv(value: Complex64) -> Complex64 {
    diagonal(value).inv(&[0], &[1]).unwrap().diagview().unwrap()[0].values[0]
}

/// `pinv` with `rcond = 0` (never discards a nonzero entry). Only valid for
/// a finite `value`: `pinv`'s own preflight rejects a non-finite magnitude.
fn compact_pinv(value: Complex64) -> Complex64 {
    diagonal(value)
        .pinv(&[0], &[1], 0.0)
        .unwrap()
        .diagview()
        .unwrap()[0]
        .values[0]
}

fn compact_inv32(value: Complex32) -> Complex32 {
    diagonal(value).inv(&[0], &[1]).unwrap().diagview().unwrap()[0].values[0]
}

/// One branch boundary of `Base.inv(::ComplexF64)` per row. The fast path
/// applies for `2^-511.5 <= max(|re|,|im|) <= 2^511.5`; outside it,
/// `robust_cinv` runs, after scaling down by 1/2 when `max >= floatmax/2`
/// and up by `2/eps^2` when `max <= 2^-969`; an infinite component returns
/// signed zeros. Power-of-two rows have exact reciprocals written as
/// formulas; their bits also agree with Julia 1.11.6's `inv`.
#[test]
fn compact_reciprocal_at_branch_boundaries() {
    let inf = f64::INFINITY;
    let rows: &[(&str, Complex64, Complex64)] = &[
        // fast path, top edge: the naive `1/z` overflowed from here up
        ("2^511", c(p2(511), 0.0), c(p2(-511), -0.0)),
        // just above the fast path: robust_cinv without scaling
        ("2^512(1-i)", c(p2(512), -p2(512)), c(p2(-513), p2(-513))),
        // fast path, bottom edge
        ("2^-511(1+i)", c(p2(-511), p2(-511)), c(p2(510), -p2(510))),
        // just below the fast path; pure imaginary
        ("-2^-512 i", c(0.0, -p2(-512)), c(0.0, p2(512))),
        // largest normal power: reciprocal is the smallest normal
        ("2^1022", c(p2(1022), 0.0), c(p2(-1022), -0.0)),
        // scale-down branch (`max >= floatmax/2`); subnormal result
        (
            "2^1023(1-i)",
            c(p2(1023), -p2(1023)),
            c(p2(-1024), p2(-1024)),
        ),
        // scale-up branch (`max <= 2^-969`)
        (
            "2^-1022(1+i)",
            c(p2(-1022), p2(-1022)),
            c(p2(1021), -p2(1021)),
        ),
        // smallest subnormal: 2^1074 overflows
        ("2^-1074", c(p2(-1074), 0.0), c(inf, -0.0)),
        ("2^-1074(1-i)", c(p2(-1074), -p2(-1074)), c(inf, inf)),
        // infinite components: signed zeros `(copysign(0, re), copysign(0, -im))`
        ("inf", c(inf, 0.0), c(0.0, -0.0)),
        ("-inf - 3i", c(-inf, -3.0), c(-0.0, 0.0)),
        ("-inf i", c(0.0, -inf), c(0.0, 0.0)),
        // Generic mantissas, bits checked against Julia 1.11.6
        // `inv(::ComplexF64)` on aarch64 (where `muladd` fuses). The
        // `mul_add`-sensitive rows change if that fused step is unfused.
        (
            "fast path, |im| > |re|",
            c(
                f64::from_bits(0xbfcb94ada8d8b6e0),
                f64::from_bits(0xbff08b1b9c37253a),
            ),
            c(
                f64::from_bits(0xbfc8b9924520dc4a),
                f64::from_bits(0x3feda928076bfa12),
            ),
        ),
        (
            "fast path; pins the denominator `cd.mul_add(cd, dc * dc)`",
            c(
                f64::from_bits(0x3fc2bf85a17ded20),
                f64::from_bits(0xbfe06c0e79448bf4),
            ),
            c(
                f64::from_bits(0x3fe074cc8c88b823),
                f64::from_bits(0x3ffcd440fa11eccc),
            ),
        ),
        (
            "unscaled robust_cinv near 2^511, swapped (|im| > |re|)",
            c(
                f64::from_bits(0xdfc981ba11426db0),
                f64::from_bits(0x5feef280c2a756ca),
            ),
            c(
                f64::from_bits(0x9fca292e91262ab1),
                f64::from_bits(0x9fefbdad03df1927),
            ),
        ),
        (
            "unscaled robust_cinv near 2^-512, swapped (|im| > |re|); \
             pins robust_cinv's `d.mul_add(r, c)`",
            c(
                f64::from_bits(0x1fed0f254120eaf0),
                f64::from_bits(0x1ff0c4746b0cbf5c),
            ),
            c(
                f64::from_bits(0x5fde398c37f9ffa7),
                f64::from_bits(0xdfe170a2b35e8b5a),
            ),
        ),
    ];
    for &(label, z, expected) in rows {
        let inv = compact_inv(z);
        assert_eq!(bits(inv), bits(expected), "inv({label}) = {inv:?}");
        if z.re.is_finite() && z.im.is_finite() {
            let pinv = compact_pinv(z);
            assert_eq!(bits(pinv), bits(expected), "pinv({label}) = {pinv:?}");
        }
    }
}

/// Generic inputs with comparable components, `|z|` near 1.
fn generic_inputs() -> [Complex64; 4] {
    [
        c(PI, E),
        c(-LN_2, SQRT_2),
        c(FRAC_1_SQRT_2, -0.3),
        c(-1.25, -0.75),
    ]
}

/// Scaling by a power of two commutes with the reciprocal. Within one
/// branch the algorithm is homogeneous, so this holds bit-for-bit; across
/// branches (fast path vs scaled/unscaled `robust_cinv`) only to rounding.
#[test]
fn reciprocal_commutes_with_power_of_two_scaling() {
    for z in generic_inputs() {
        let inv = compact_inv(z);
        // fast path throughout
        for k in [-400, -1, 1, 400] {
            let scaled = compact_inv(z * p2(k)) * p2(k);
            assert_eq!(bits(scaled), bits(inv), "z = {z:?}, k = {k}");
        }
        // unscaled robust_cinv throughout
        let big = z * p2(600);
        let inv_big = compact_inv(big);
        for k in [100, 300] {
            let scaled = compact_inv(big * p2(k)) * p2(k);
            assert_eq!(bits(scaled), bits(inv_big), "z = {big:?}, k = {k}");
        }
        // every branch against the fast path, to rounding
        for k in [-1000, -600, -512, 512, 600, 1000] {
            let scaled = compact_inv(z * p2(k)) * p2(k);
            assert!(
                (scaled - inv).norm() <= 2.0 * f64::EPSILON * inv.norm(),
                "z = {z:?}, k = {k}: {scaled:?} vs {inv:?}"
            );
        }
    }
}

/// `inv(conj z) = conj(inv z)` exactly: every branch treats the sign of the
/// imaginary part symmetrically.
#[test]
fn reciprocal_commutes_with_conjugation() {
    for z in generic_inputs() {
        for k in [-1000, -600, 0, 600, 1000] {
            let w = z * p2(k);
            assert_eq!(
                bits(compact_inv(w.conj())),
                bits(compact_inv(w).conj()),
                "w = {w:?}"
            );
        }
    }
}

/// `z * inv(z) = 1` to a few ulp wherever `inv(z)` is a normal number,
/// including the magnitudes where the naive `1/z` overflowed or underflowed.
#[test]
fn reciprocal_times_input_is_one() {
    for z in generic_inputs() {
        for k in [-1000, -600, -511, 0, 511, 600, 1000] {
            let w = z * p2(k);
            let product = w * compact_inv(w);
            assert!(
                (product - 1.0).norm() <= 4.0 * f64::EPSILON,
                "w = {w:?}: w * inv(w) = {product:?}"
            );
        }
    }
}

/// `Complex32` widens to `Complex64`, divides, and narrows, so its edge
/// cases are narrowing ones: a subnormal or overflowing `f32` result and
/// the infinite-component signed zeros. Bits agree with Julia 1.11.6's
/// `inv(::ComplexF32)`.
#[test]
fn compact_reciprocal_32_at_range_edges() {
    let inf = f32::INFINITY;
    let c32 = Complex32::new;
    let rows: &[(&str, Complex32, Complex32)] = &[
        (
            "2^127(1-i)",
            c32(p2_32(127), -p2_32(127)),
            c32(p2_32(-128), p2_32(-128)),
        ),
        ("2^-126", c32(p2_32(-126), 0.0), c32(p2_32(126), -0.0)),
        ("2^-149(1-i)", c32(p2_32(-149), -p2_32(-149)), c32(inf, inf)),
        ("-inf", c32(-inf, 0.0), c32(-0.0, -0.0)),
        ("3 + inf i", c32(3.0, inf), c32(0.0, -0.0)),
    ];
    for &(label, z, expected) in rows {
        let inv = compact_inv32(z);
        assert_eq!(bits32(inv), bits32(expected), "inv({label}) = {inv:?}");
    }
}

/// `inv` of an exactly zero compact entry is the caller-mistake
/// `InvalidArgument` the compact arm reports
/// (`symtn/src/typed/fusion_tree.rs::inv_spectrum`), whichever reciprocal
/// algorithm the nonzero branch uses (Julia's own `inv(0.0+0.0im)` is
/// `NaN + NaN*im`, not an error).
#[test]
fn zero_entry_is_still_reported_as_a_singular_diagonal() {
    assert!(diagonal(c(0.0, 0.0))
        .inv(&[0], &[1])
        .err()
        .unwrap()
        .report()
        .contains("inv of a singular diagonal"));
}
