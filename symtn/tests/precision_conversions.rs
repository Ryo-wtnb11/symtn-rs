//! Explicit payload precision conversions (#1323).
//!
//! What: `convert` widenings (f32 -> f64/c32/c64, f64 -> c64, c32 -> c64) are
//! exact and the narrowings (f64 -> f32, c64 -> c32) round like Rust `as f32`,
//! entry by entry; spaces, block
//! structure and storage form (dense, compact diagonal, lazy adjoint) are
//! unchanged; a conversion makes one payload-sized allocation.
//!
//! Every comparison is bitwise: conversion is data movement. The oracle for a
//! widening is `f64::from` and for a narrowing `as f32`, applied per element
//! to the source payload — independent of the tensor code under test.

use crate::common::ErrorKind as _;
use crate::single_precision_oracle;

use std::hint::black_box;
use std::sync::Arc;

use num_complex::{Complex32, Complex64};
use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::{Engine, GradedSpace, SectorSpectrum, TensorMap};

use single_precision_oracle::{fermion_su2_leg, runtime, u1_leg, u1_leg_with};

use crate::counting_alloc;
#[cfg(feature = "cgc-gen")]
use crate::report;

// Every test takes the target-wide serial guard so no sibling can warm or
// evict the shared layout cache between allocation measurements.

/// Allocation calls, bytes, and calls of at least `large` bytes.
fn measured<T>(large: usize, operation: impl FnOnce() -> T) -> (T, usize, usize, usize) {
    let (value, allocs) = counting_alloc::measure_matching(large..=usize::MAX, operation);
    (
        value,
        allocs.calls as usize,
        allocs.bytes as usize,
        allocs.matched_calls as usize,
    )
}

/// Every `f32` class: signed zeros, the smallest and largest subnormal, the
/// normal range ends, infinities, a NaN with a payload, and ordinary values.
const F32_SPECIALS: [f32; 12] = [
    0.0,
    -0.0,
    f32::from_bits(1),
    f32::from_bits(0x007f_ffff),
    f32::MIN_POSITIVE,
    f32::MAX,
    -f32::MAX,
    f32::INFINITY,
    f32::NEG_INFINITY,
    f32::from_bits(0x7fc0_1234),
    1.0 / 3.0,
    -2.5e-3,
];

/// `f64` inputs chosen for the rounding and range edges of `as f32`.
fn f64_narrowing_cases() -> Vec<(f64, Option<f32>)> {
    let one = 1.0f64;
    let half_ulp = 2f64.powi(-24);
    let max = f64::from(f32::MAX);
    vec![
        // tie at 1 + 2^-24 rounds to even (down to 1)
        (one + half_ulp, Some(1.0)),
        // just above the tie rounds up
        (one + half_ulp + 2f64.powi(-52), Some(1.0 + f32::EPSILON)),
        // tie at 1 + 3 * 2^-24 rounds to even (up)
        (one + 3.0 * half_ulp, Some(1.0 + 2.0 * f32::EPSILON)),
        (0.1, None),
        (-0.0, Some(-0.0)),
        (0.0, Some(0.0)),
        // below half the ulp of f32::MAX: rounds back to f32::MAX
        (max + 2f64.powi(102), Some(f32::MAX)),
        // the tie above f32::MAX (odd significand) rounds to infinity
        (max + 2f64.powi(103), Some(f32::INFINITY)),
        (-2.0 * max, Some(f32::NEG_INFINITY)),
        (1e300, Some(f32::INFINITY)),
        (f64::INFINITY, Some(f32::INFINITY)),
        (f64::NEG_INFINITY, Some(f32::NEG_INFINITY)),
        // subnormal range: kept, not flushed
        (f64::from(f32::from_bits(1)), Some(f32::from_bits(1))),
        (3e-39, None),
        // below half the smallest subnormal: signed zero
        (1e-46, Some(0.0)),
        (-1e-46, Some(-0.0)),
        (f64::from_bits(1), Some(0.0)),
        (f64::NAN, None),
    ]
}

fn f32_bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn f64_bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

fn c32_bits(values: &[Complex32]) -> Vec<(u32, u32)> {
    values
        .iter()
        .map(|value| (value.re.to_bits(), value.im.to_bits()))
        .collect()
}

fn c64_bits(values: &[Complex64]) -> Vec<(u64, u64)> {
    values
        .iter()
        .map(|value| (value.re.to_bits(), value.im.to_bits()))
        .collect()
}

fn narrow(value: f64) -> f32 {
    black_box(value) as f32
}

/// Asserts the converted tensor carries the source's spaces and blocks.
macro_rules! assert_same_structure {
    ($engine:ident, $what:expr, $got:expr, $source:expr) => {{
        let (got, source) = (&$got, &$source);
        assert_eq!(got.codomain(), source.codomain(), "{}: codomain", $what);
        assert_eq!(got.domain(), source.domain(), "{}: domain", $what);
        assert_eq!(
            got.subblock_count(),
            source.subblock_count(),
            "{}: blocks",
            $what
        );
        for index in 0..source.subblock_count() {
            assert_eq!(
                format!("{:?}", got.subblock_fusion_trees(index).unwrap()),
                format!("{:?}", source.subblock_fusion_trees(index).unwrap()),
                "{}: fusion trees of block {index}",
                $what
            );
        }
        assert_eq!(
            got.materialize($engine)
                .unwrap()
                .dense_data()
                .unwrap()
                .len(),
            source
                .materialize($engine)
                .unwrap()
                .dense_data()
                .unwrap()
                .len(),
            "{}: length",
            $what
        );
        assert!(
            std::ptr::eq(got.provider(), source.provider()),
            "{}: the provider Arc must be shared, not rebuilt",
            $what
        );
    }};
}

/// Runs every conversion on one tensor of any provider and storage form and
/// checks it bitwise against the per-element oracle and the source structure.
/// `$f32` is the single-precision source; the other dtypes are derived from it
/// by the oracle, not by the code under test.
macro_rules! assert_all_conversions {
    ($engine:ident, $what:expr, $f32:expr) => {{
        let what = $what;
        let source: TensorMap<_, f32> = $f32;
        let values = source
            .materialize($engine)
            .unwrap()
            .dense_data()
            .unwrap()
            .to_vec();

        let wide = source.convert::<f64>($engine).unwrap();
        assert_same_structure!($engine, what, wide, source);
        let expected: Vec<f64> = values.iter().map(|&value| f64::from(value)).collect();
        assert_eq!(
            f64_bits(wide.materialize($engine).unwrap().dense_data().unwrap()),
            f64_bits(&expected),
            "{what}: convert::<f64>"
        );
        assert_eq!(
            f32_bits(
                wide.convert::<f32>($engine)
                    .unwrap()
                    .materialize($engine)
                    .unwrap()
                    .dense_data()
                    .unwrap()
            ),
            f32_bits(&values),
            "{what}: convert::<f32>(convert::<f64>(x)) == x"
        );

        let c32 = source.convert::<Complex32>($engine).unwrap();
        assert_same_structure!($engine, what, c32, source);
        let expected: Vec<Complex32> = values
            .iter()
            .map(|&value| Complex32::new(value, 0.0))
            .collect();
        assert_eq!(
            c32_bits(c32.materialize($engine).unwrap().dense_data().unwrap()),
            c32_bits(&expected),
            "{what}: convert::<Complex32>"
        );

        let c64 = source.convert::<Complex64>($engine).unwrap();
        assert_same_structure!($engine, what, c64, source);
        let expected: Vec<Complex64> = values
            .iter()
            .map(|&value| Complex64::new(f64::from(value), 0.0))
            .collect();
        assert_eq!(
            c64_bits(c64.materialize($engine).unwrap().dense_data().unwrap()),
            c64_bits(&expected),
            "{what}: f32 convert::<Complex64>"
        );
        assert_eq!(
            c64_bits(
                wide.convert::<Complex64>($engine)
                    .unwrap()
                    .materialize($engine)
                    .unwrap()
                    .dense_data()
                    .unwrap()
            ),
            c64_bits(&expected),
            "{what}: f64 convert::<Complex64>"
        );
        assert_eq!(
            c64_bits(
                c32.convert::<Complex64>($engine)
                    .unwrap()
                    .materialize($engine)
                    .unwrap()
                    .dense_data()
                    .unwrap()
            ),
            c64_bits(&expected),
            "{what}: Complex32 convert::<Complex64>"
        );
        assert_eq!(
            c32_bits(
                c64.convert::<Complex32>($engine)
                    .unwrap()
                    .materialize($engine)
                    .unwrap()
                    .dense_data()
                    .unwrap()
            ),
            c32_bits(c32.materialize($engine).unwrap().dense_data().unwrap()),
            "{what}: convert::<Complex32>(convert::<Complex64>(x)) == x"
        );

        // A genuinely complex payload: both components move independently.
        let complex = c32.scale($engine, Complex32::new(0.75, -1.5)).unwrap();
        let complex_values = complex
            .materialize($engine)
            .unwrap()
            .dense_data()
            .unwrap()
            .to_vec();
        let widened = complex.convert::<Complex64>($engine).unwrap();
        assert_same_structure!($engine, what, widened, complex);
        let expected: Vec<Complex64> = complex_values
            .iter()
            .map(|value| Complex64::new(f64::from(value.re), f64::from(value.im)))
            .collect();
        assert_eq!(
            c64_bits(widened.materialize($engine).unwrap().dense_data().unwrap()),
            c64_bits(&expected),
            "{what}: complex convert::<Complex64>"
        );
        assert_eq!(
            c32_bits(
                widened
                    .convert::<Complex32>($engine)
                    .unwrap()
                    .materialize($engine)
                    .unwrap()
                    .dense_data()
                    .unwrap()
            ),
            c32_bits(&complex_values),
            "{what}: complex round trip"
        );
        (source, wide)
    }};
}

/// Converts the lazy adjoint of `$owned` at every source dtype and compares
/// each result bitwise with the element-wise conversion of the materialized
/// source view. Returns the converted views for storage-form checks.
macro_rules! assert_adjoint_conversions {
    ($engine:ident, $what:expr, $owned:expr) => {{
        let what = $what;
        let owned: TensorMap<_, f32> = $owned;
        let lazy32 = owned.adjoint($engine).unwrap();
        let lazy64 = owned
            .convert::<f64>($engine)
            .unwrap()
            .adjoint($engine)
            .unwrap();
        let lazyc32 = owned
            .convert::<Complex32>($engine)
            .unwrap()
            .scale($engine, Complex32::new(0.75, -1.5))
            .unwrap()
            .adjoint($engine)
            .unwrap();
        let lazyc64 = owned
            .convert::<Complex64>($engine)
            .unwrap()
            .scale($engine, Complex64::new(0.75, -1.5))
            .unwrap()
            .adjoint($engine)
            .unwrap();

        let widened_f64 = lazy32.convert::<f64>($engine).unwrap();
        let expected: Vec<f64> = lazy32
            .materialize($engine)
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        assert_eq!(
            f64_bits(
                widened_f64
                    .materialize($engine)
                    .unwrap()
                    .dense_data()
                    .unwrap()
            ),
            f64_bits(&expected),
            "{what}: f32 convert::<f64>"
        );
        assert_same_structure!($engine, what, widened_f64, lazy32);

        let to_c32 = lazy32.convert::<Complex32>($engine).unwrap();
        let expected: Vec<Complex32> = lazy32
            .materialize($engine)
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .map(|&v| Complex32::new(v, 0.0))
            .collect();
        assert_eq!(
            c32_bits(to_c32.materialize($engine).unwrap().dense_data().unwrap()),
            c32_bits(&expected),
            "{what}: f32 convert::<Complex32>"
        );

        let f32_to_c64 = lazy32.convert::<Complex64>($engine).unwrap();
        let expected: Vec<Complex64> = lazy32
            .materialize($engine)
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .map(|&v| Complex64::new(f64::from(v), 0.0))
            .collect();
        assert_eq!(
            c64_bits(
                f32_to_c64
                    .materialize($engine)
                    .unwrap()
                    .dense_data()
                    .unwrap()
            ),
            c64_bits(&expected),
            "{what}: f32 convert::<Complex64>"
        );

        let f64_to_c64 = lazy64.convert::<Complex64>($engine).unwrap();
        let expected: Vec<Complex64> = lazy64
            .materialize($engine)
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .map(|&v| Complex64::new(v, 0.0))
            .collect();
        assert_eq!(
            c64_bits(
                f64_to_c64
                    .materialize($engine)
                    .unwrap()
                    .dense_data()
                    .unwrap()
            ),
            c64_bits(&expected),
            "{what}: f64 convert::<Complex64>"
        );
        assert_same_structure!($engine, what, f64_to_c64, lazy64);

        let narrowed = lazy64.convert::<f32>($engine).unwrap();
        let expected: Vec<f32> = lazy64
            .materialize($engine)
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .map(|&v| narrow(v))
            .collect();
        assert_eq!(
            f32_bits(narrowed.materialize($engine).unwrap().dense_data().unwrap()),
            f32_bits(&expected),
            "{what}: convert::<f32>"
        );

        let c32_to_c64 = lazyc32.convert::<Complex64>($engine).unwrap();
        let expected: Vec<Complex64> = lazyc32
            .materialize($engine)
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .map(|v| Complex64::new(f64::from(v.re), f64::from(v.im)))
            .collect();
        assert_eq!(
            c64_bits(
                c32_to_c64
                    .materialize($engine)
                    .unwrap()
                    .dense_data()
                    .unwrap()
            ),
            c64_bits(&expected),
            "{what}: Complex32 convert::<Complex64>"
        );

        let narrowed_c = lazyc64.convert::<Complex32>($engine).unwrap();
        let expected: Vec<Complex32> = lazyc64
            .materialize($engine)
            .unwrap()
            .dense_data()
            .unwrap()
            .iter()
            .map(|v| Complex32::new(narrow(v.re), narrow(v.im)))
            .collect();
        assert_eq!(
            c32_bits(
                narrowed_c
                    .materialize($engine)
                    .unwrap()
                    .dense_data()
                    .unwrap()
            ),
            c32_bits(&expected),
            "{what}: convert::<Complex32>"
        );
        assert_same_structure!($engine, what, narrowed_c, lazyc64);

        (widened_f64, f64_to_c64, c32_to_c64, narrowed_c)
    }};
}

/// A tensor cycling through [`F32_SPECIALS`], scaled per entry so the
/// stored values are distinct.
macro_rules! filled {
    ($engine:ident, [$($codomain:expr),*], [$($domain:expr),*]) => {{
        let mut next = 0usize;
        let tensor: TensorMap<_, f32> =
            TensorMap::from_subblock_fn($engine, &runtime(), [$($codomain),*], [$($domain),*], |_, _| {
                let value = F32_SPECIALS[next % F32_SPECIALS.len()];
                next += 1;
                value * (1.0 + next as f32 / 64.0)
            })
            .unwrap();
        tensor
    }};
}

#[test]
fn u1_dense_conversions_are_exact_and_keep_structure() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let leg = u1_leg();
    let (source, _) =
        assert_all_conversions!(engine, "u1 dense", filled!(engine, [&leg, &leg], [&leg]));
    assert!(source
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .any(|value| value.is_nan()));
    assert!(source
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .any(|value| value.is_subnormal()));
    assert!(source
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .any(|value| *value == 0.0 && value.is_sign_negative()));
}

#[test]
fn fermionic_su2_dense_conversions_are_exact_and_keep_structure() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let leg = fermion_su2_leg();
    assert_all_conversions!(
        engine,
        "fZ2 x U(1) x SU(2) dense",
        filled!(engine, [&leg, &leg], [&leg, &leg])
    );
}

#[test]
fn lazy_adjoint_conversions_match_the_converted_logical_payload() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let leg = fermion_su2_leg();
    let source = filled!(engine, [&leg, &leg], [&leg]);
    // The logical payload of the lazy view is what gets converted.
    assert_all_conversions!(
        engine,
        "fermionic lazy adjoint",
        source.adjoint(engine).unwrap()
    );
    let u1 = u1_leg();
    assert_all_conversions!(
        engine,
        "u1 lazy adjoint",
        filled!(engine, [&u1], [&u1, &u1]).adjoint(engine).unwrap()
    );

    assert_adjoint_conversions!(engine, "fermionic dense adjoint", source.clone());
    assert_adjoint_conversions!(
        engine,
        "u1 dense adjoint",
        filled!(engine, [&u1, &u1], [&u1])
    );

    // A complex lazy adjoint conjugates: the converted view must too.
    let complex = source
        .convert::<Complex32>(engine)
        .unwrap()
        .scale(engine, Complex32::new(0.5, 2.0))
        .unwrap();
    let lazy = complex.adjoint(engine).unwrap();
    let expected: Vec<Complex64> = lazy
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .map(|value| Complex64::new(f64::from(value.re), f64::from(value.im)))
        .collect();
    let widened = complex
        .adjoint(engine)
        .unwrap()
        .convert::<Complex64>(engine)
        .unwrap();
    assert_eq!(
        c64_bits(widened.materialize(engine).unwrap().dense_data().unwrap()),
        c64_bits(&expected)
    );
    assert_eq!(
        c32_bits(
            complex
                .convert::<Complex64>(engine)
                .unwrap()
                .adjoint(engine)
                .unwrap()
                .convert::<Complex32>(engine)
                .unwrap()
                .materialize(engine)
                .unwrap()
                .dense_data()
                .unwrap()
        ),
        c32_bits(lazy.materialize(engine).unwrap().dense_data().unwrap())
    );
}

#[test]
fn compact_diagonal_stays_compact() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let leg = u1_leg();
    let mut next = 0usize;
    let spectra = [(-1, 2), (0, 3), (1, 2)].map(|(charge, degeneracy)| SectorSpectrum {
        sector: U1Irrep::new(charge),
        values: (0..degeneracy)
            .map(|_| {
                next += 1;
                F32_SPECIALS[next % F32_SPECIALS.len()]
            })
            .collect(),
    });
    let source: TensorMap<U1FusionRule, f32> =
        TensorMap::diagonal(engine, &runtime(), &leg, spectra.clone()).unwrap();
    let (_, wide) = assert_all_conversions!(engine, "compact diagonal", source.clone());

    let widened = symtn::expert::diagonal_spectrum(engine, &wide)
        .unwrap()
        .expect("convert::<f64> stays compact");
    assert_eq!(widened.len(), spectra.len());
    for got in &widened {
        let expected = spectra
            .iter()
            .find(|entry| entry.sector == got.sector)
            .unwrap();
        let expected: Vec<f64> = expected
            .values
            .iter()
            .map(|&value| f64::from(value))
            .collect();
        assert_eq!(f64_bits(&got.values), f64_bits(&expected));
    }
    let source_convert = &source.convert::<Complex32>(engine).unwrap();
    assert!(symtn::expert::diagonal_spectrum(engine, source_convert)
        .unwrap()
        .is_some());
    let source_convert = &source.convert::<Complex64>(engine).unwrap();
    assert!(symtn::expert::diagonal_spectrum(engine, source_convert)
        .unwrap()
        .is_some());
    let wide_convert = &wide.convert::<f32>(engine).unwrap();
    assert!(symtn::expert::diagonal_spectrum(engine, wide_convert)
        .unwrap()
        .is_some());
    let wide_convert = &wide
        .convert::<Complex64>(engine)
        .unwrap()
        .convert::<Complex32>(engine)
        .unwrap();
    assert!(symtn::expert::diagonal_spectrum(engine, wide_convert)
        .unwrap()
        .is_some());
    // A multiplicity-free `adjoint` of a compact diagonal is already an owned
    // compact diagonal, so its conversion is the owned case again.
    let adjoint = source
        .adjoint(engine)
        .unwrap()
        .convert::<f64>(engine)
        .unwrap();
    assert!(symtn::expert::diagonal_spectrum(engine, &adjoint)
        .unwrap()
        .is_some());
}

#[test]
fn narrowing_rounds_like_as_f32_per_element() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let cases = f64_narrowing_cases();
    let leg =
        GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), cases.len())]).unwrap();
    let mut entries = cases.iter().map(|&(value, _)| value);
    let source: TensorMap<U1FusionRule, f64> = TensorMap::from_subblock_fn(
        engine,
        &runtime(),
        [&leg],
        [&GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 1)]).unwrap()],
        |_, _| entries.next().unwrap(),
    )
    .unwrap();
    // `from_subblock_fn` fills in the coupled layout; read the order back.
    let values = source.dense_data().unwrap().to_vec();
    assert_eq!(values.len(), cases.len());

    let narrowed = source.convert::<f32>(engine).unwrap();
    assert_same_structure!(engine, "convert::<f32>", narrowed, source);
    for (index, (&got, &value)) in narrowed
        .dense_data()
        .unwrap()
        .iter()
        .zip(&values)
        .enumerate()
    {
        let oracle = narrow(value);
        assert_eq!(got.to_bits(), oracle.to_bits(), "entry {index}: {value:e}");
        assert_eq!(got.is_nan(), value.is_nan(), "entry {index}: NaN stays NaN");
    }
    // Hand values, independent of `as`: the documented rounding and range.
    for &(value, expected) in &cases {
        let position = values
            .iter()
            .position(|stored| stored.to_bits() == value.to_bits())
            .unwrap();
        let got = narrowed.dense_data().unwrap()[position];
        match expected {
            Some(expected) => assert_eq!(got.to_bits(), expected.to_bits(), "{value:e}"),
            None if value.is_nan() => assert!(got.is_nan()),
            None => assert_eq!(f64::from(got), f64::from(narrow(value))),
        }
    }
    assert!(narrowed
        .dense_data()
        .unwrap()
        .iter()
        .any(|value| value.is_subnormal()));
    let source_convert = &source.convert::<Complex64>(engine).unwrap();

    // Complex narrowing is componentwise with the same rounding.
    let complex = source
        .convert::<Complex64>(engine)
        .unwrap()
        .scale(engine, Complex64::new(0.0, 1.0))
        .unwrap()
        .axpby(
            engine,
            Complex64::new(1.0, 0.0),
            source_convert,
            Complex64::new(1.0, 0.0),
        );
    let complex = complex.unwrap();
    let complex_values = complex.dense_data().unwrap().to_vec();
    let narrowed = complex.convert::<Complex32>(engine).unwrap();
    assert_same_structure!(engine, "convert::<Complex32>", narrowed, complex);
    let expected: Vec<Complex32> = complex_values
        .iter()
        .map(|value| Complex32::new(narrow(value.re), narrow(value.im)))
        .collect();
    assert_eq!(
        c32_bits(narrowed.dense_data().unwrap()),
        c32_bits(&expected)
    );
}

/// Measures `convert` on a small and a large source of the same structure:
/// the handle cost must not depend on the payload, the byte difference must
/// be exactly `payloads` buffers of the given entry sizes, and exactly that
/// many allocations are payload-sized.
fn assert_payload_allocations<T, O>(
    engine: &mut dyn Engine,
    what: &str,
    small: &T,
    large: &T,
    lengths: (usize, usize),
    entry_sizes: &[usize],
    convert: impl Fn(&T, &mut dyn Engine) -> O,
) {
    let smallest = *entry_sizes.iter().min().unwrap();
    let per_entry: usize = entry_sizes.iter().sum();
    black_box(convert(small, engine));
    black_box(convert(large, engine));
    let (_, small_calls, small_bytes, small_payloads) =
        measured(lengths.0 * smallest, || black_box(convert(small, engine)));
    let (_, large_calls, large_bytes, large_payloads) =
        measured(lengths.1 * smallest, || black_box(convert(large, engine)));
    assert_eq!(
        small_calls, large_calls,
        "{what}: handle cost depends on the payload"
    );
    assert_eq!(
        large_bytes - small_bytes,
        (lengths.1 - lengths.0) * per_entry,
        "{what}: payload bytes"
    );
    assert_eq!(
        (small_payloads, large_payloads),
        (entry_sizes.len(), entry_sizes.len()),
        "{what}: payload-sized allocations"
    );
}

#[test]
fn conversion_payload_allocations() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let (small_leg, large_leg) = (u1_leg_with([2, 3, 2]), u1_leg_with([9, 10, 9]));
    let small = filled!(engine, [&small_leg, &small_leg], [&small_leg]);
    let large = filled!(engine, [&large_leg, &large_leg], [&large_leg]);
    let lengths = (
        small.dense_data().unwrap().len(),
        large.dense_data().unwrap().len(),
    );
    assert_payload_allocations(
        engine,
        "f32 convert::<f64>",
        &small,
        &large,
        lengths,
        &[std::mem::size_of::<f64>()],
        TensorMap::convert::<f64>,
    );
    assert_payload_allocations(
        engine,
        "f32 convert::<Complex32>",
        &small,
        &large,
        lengths,
        &[std::mem::size_of::<Complex32>()],
        TensorMap::convert::<Complex32>,
    );
    assert_payload_allocations(
        engine,
        "f32 convert::<Complex64>",
        &small,
        &large,
        lengths,
        &[std::mem::size_of::<Complex64>()],
        |t: &TensorMap<U1FusionRule, f32>, engine: &mut dyn Engine| {
            t.convert::<Complex64>(engine).unwrap()
        },
    );

    let (small_wide, large_wide) = (
        small.convert::<f64>(engine).unwrap(),
        large.convert::<f64>(engine).unwrap(),
    );
    assert_payload_allocations(
        engine,
        "f64 convert::<f32>",
        &small_wide,
        &large_wide,
        lengths,
        &[std::mem::size_of::<f32>()],
        TensorMap::convert::<f32>,
    );
    assert_payload_allocations(
        engine,
        "f64 convert::<Complex64>",
        &small_wide,
        &large_wide,
        lengths,
        &[std::mem::size_of::<Complex64>()],
        |t: &TensorMap<U1FusionRule, f64>, engine: &mut dyn Engine| {
            t.convert::<Complex64>(engine).unwrap()
        },
    );
    let (small_c32, large_c32) = (
        small.convert::<Complex32>(engine).unwrap(),
        large.convert::<Complex32>(engine).unwrap(),
    );
    assert_payload_allocations(
        engine,
        "Complex32 convert::<Complex64>",
        &small_c32,
        &large_c32,
        lengths,
        &[std::mem::size_of::<Complex64>()],
        |t: &TensorMap<U1FusionRule, Complex32>, engine: &mut dyn Engine| {
            t.convert::<Complex64>(engine).unwrap()
        },
    );
    let (small_c64, large_c64) = (
        small.convert::<Complex64>(engine).unwrap(),
        large.convert::<Complex64>(engine).unwrap(),
    );
    assert_payload_allocations(
        engine,
        "Complex64 convert::<Complex32>",
        &small_c64,
        &large_c64,
        lengths,
        &[std::mem::size_of::<Complex32>()],
        TensorMap::convert::<Complex32>,
    );

    // A lazy-adjoint source: the result is owned, at the cost of an
    // operation-local materialization in the source dtype plus the output —
    // two payload-sized allocations.
    let (small_lazy, large_lazy) = (
        small.adjoint(engine).unwrap(),
        large.adjoint(engine).unwrap(),
    );
    assert_payload_allocations(
        engine,
        "lazy adjoint convert::<f64>",
        &small_lazy,
        &large_lazy,
        lengths,
        &[std::mem::size_of::<f32>(), std::mem::size_of::<f64>()],
        TensorMap::convert::<f64>,
    );
}

/// A converted lazy adjoint is owned, so operations that reject lazy
/// adjoints accept it (#1446 review: `diagview` on U(1) returned `Ok` on main
/// and must keep doing so).
#[test]
fn converted_lazy_adjoints_are_owned() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    let leg = u1_leg();
    let bond: TensorMap<U1FusionRule, f64> =
        TensorMap::rand_with_seed(engine, &runtime(), [&leg], [&leg], 11).unwrap();
    let lazy = bond.adjoint(engine).unwrap();
    assert!(
        lazy.diagview(engine).is_invalid_argument("lazy adjoint"),
        "the fixture must be a lazy adjoint"
    );
    assert_eq!(
        lazy.convert::<Complex64>(engine)
            .unwrap()
            .diagview(engine)
            .unwrap()
            .len(),
        bond.diagview(engine).unwrap().len()
    );
    lazy.convert::<f32>(engine)
        .unwrap()
        .diagview(engine)
        .unwrap();
    lazy.convert::<f32>(engine)
        .unwrap()
        .convert::<f64>(engine)
        .unwrap()
        .diagview(engine)
        .unwrap();
    lazy.convert::<f32>(engine)
        .unwrap()
        .convert::<Complex32>(engine)
        .unwrap()
        .diagview(engine)
        .unwrap();
    lazy.convert::<Complex64>(engine)
        .unwrap()
        .convert::<Complex32>(engine)
        .unwrap()
        .diagview(engine)
        .unwrap();
    lazy.convert::<Complex64>(engine)
        .unwrap()
        .adjoint(engine)
        .unwrap()
        .convert::<Complex32>(engine)
        .unwrap()
        .convert::<Complex64>(engine)
        .unwrap()
        .diagview(engine)
        .unwrap();

    // A `permute_into` source must be owned.
    let tensor: TensorMap<U1FusionRule, f32> = filled!(engine, [&leg, &leg], [&leg]);
    let source = tensor
        .adjoint(engine)
        .unwrap()
        .convert::<f64>(engine)
        .unwrap();
    let expected = source.permute(engine, &[2], &[0, 1]).unwrap();
    let mut destination = expected.zeros_like(engine).unwrap();
    source
        .permute_into(engine, &[2], &[0, 1], &mut destination, 1.0, 0.0)
        .unwrap();
    assert_eq!(
        f64_bits(destination.dense_data().unwrap()),
        f64_bits(expected.dense_data().unwrap())
    );
}

/// Checked-Generic provider: SU(3) with an outer-multiplicity vertex.
#[cfg(feature = "cgc-gen")]
#[test]
fn checked_generic_su3_conversions_are_exact_and_keep_structure() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    use symtn::sector::LieGroupFusionRule;

    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg = GradedSpace::try_new(provider, [(vec![2i64, 2], 2)]).unwrap();
    let source = filled!(engine, [&leg, &leg], [&leg, &leg]);
    assert!(
        (0..source.subblock_count()).any(|index| source
            .subblock_fusion_trees(index)
            .unwrap()
            .codomain_vertices()
            .iter()
            .any(|vertex| vertex.get() > 1)),
        "the fixture must carry a Generic multiplicity vertex"
    );
    assert_all_conversions!(engine, "SU(3) dense", source.clone());
    assert_all_conversions!(
        engine,
        "SU(3) lazy adjoint",
        source.adjoint(engine).unwrap()
    );
    let (_, complex, _, _) = assert_adjoint_conversions!(engine, "SU(3) dense adjoint", source);
    // The converted adjoint comes out owned (only owned dense storage lends
    // its payload), and QR reaches the finite-input stage, which refuses the
    // fixture's NaN and infinities (#1986).
    complex.dense_data().unwrap();
    let error = complex.qr_compact(engine, &[0, 1], &[2, 3]).unwrap_err();
    assert_eq!(
        report::cause::<symtn::typed::OperationError>(&error),
        Some(&symtn::typed::OperationError::InvalidArgument {
            message: "qr input components must be finite"
        }),
        "{error:?}"
    );
}

/// Checked-Generic compact diagonal adjoint (#1446 review, #1449): the
/// conversion must come out as the owned compact diagonal, keep the
/// structural zeros `+0 + 0i`, and stay factorizable.
#[cfg(feature = "cgc-gen")]
#[test]
fn checked_generic_diagonal_adjoint_converts_to_an_owned_compact_diagonal() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let _serial = crate::counting_alloc::serial();
    use symtn::sector::LieGroupFusionRule;

    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg = GradedSpace::try_new(provider, [(vec![2i64, 1], 3), (vec![0, 0], 2)]).unwrap();
    let spectra = [
        SectorSpectrum {
            sector: vec![2i64, 1],
            values: vec![f32::INFINITY, f32::from_bits(0x7fc0_1234), -0.0],
        },
        SectorSpectrum {
            sector: vec![0, 0],
            values: vec![f32::NEG_INFINITY, f32::from_bits(1)],
        },
    ];
    let source: TensorMap<_, f32> = TensorMap::diagonal(engine, &runtime(), &leg, spectra).unwrap();
    assert!(source
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .any(|value| value.is_nan()));
    assert!(source
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .any(|value| value.is_infinite()));

    let (widened_f64, to_c64, c32_to_c64, narrowed_c) =
        assert_adjoint_conversions!(engine, "SU(3) diagonal adjoint", source.clone());
    for converted in [&widened_f64.convert::<Complex64>(engine).unwrap(), &to_c64] {
        assert!(symtn::expert::diagonal_spectrum(engine, converted)
            .unwrap()
            .is_some());
    }
    // The stored spectrum of the complex conversions is conj(convert(value)),
    // bitwise.
    let factor = Complex32::new(0.75, -1.5);
    let source_convert = &source
        .convert::<Complex32>(engine)
        .unwrap()
        .scale(engine, factor)
        .unwrap();
    let stored = symtn::expert::diagonal_spectrum(engine, source_convert)
        .unwrap()
        .unwrap();
    let got = symtn::expert::diagonal_spectrum(engine, &c32_to_c64)
        .unwrap()
        .unwrap();
    for entry in &got {
        let parent = stored.iter().find(|p| p.sector == entry.sector).unwrap();
        let expected: Vec<Complex64> = parent
            .values
            .iter()
            .map(|v| Complex64::new(f64::from(v.re), f64::from(v.im)).conj())
            .collect();
        assert_eq!(c64_bits(&entry.values), c64_bits(&expected));
    }
    let source_convert = &source
        .convert::<Complex64>(engine)
        .unwrap()
        .scale(engine, Complex64::new(0.75, -1.5))
        .unwrap();
    let stored = symtn::expert::diagonal_spectrum(engine, source_convert)
        .unwrap()
        .unwrap();
    let got = symtn::expert::diagonal_spectrum(engine, &narrowed_c)
        .unwrap()
        .unwrap();
    for entry in &got {
        let parent = stored.iter().find(|p| p.sector == entry.sector).unwrap();
        let expected: Vec<Complex32> = parent
            .values
            .iter()
            .map(|v| Complex32::new(narrow(v.re), narrow(-v.im)))
            .collect();
        assert_eq!(c32_bits(&entry.values), c32_bits(&expected));
    }

    // The review repro: finite f64 values, `convert::<Complex64>` of the adjoint.
    let finite: TensorMap<_, f64> = TensorMap::diagonal(
        engine,
        &runtime(),
        &leg,
        [
            SectorSpectrum {
                sector: vec![2i64, 1],
                values: vec![1.5, -2.0, 0.25],
            },
            SectorSpectrum {
                sector: vec![0, 0],
                values: vec![3.0, -0.5],
            },
        ],
    )
    .unwrap();
    let lazy = finite.adjoint(engine).unwrap();
    let converted = lazy.convert::<Complex64>(engine).unwrap();
    let expected: Vec<Complex64> = lazy
        .materialize(engine)
        .unwrap()
        .dense_data()
        .unwrap()
        .iter()
        .map(|&v| Complex64::new(v, 0.0))
        .collect();
    assert_eq!(
        c64_bits(converted.materialize(engine).unwrap().dense_data().unwrap()),
        c64_bits(&expected)
    );
    converted.qr_compact(engine, &[0], &[1]).unwrap();
}
