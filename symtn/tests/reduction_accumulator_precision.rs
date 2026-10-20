//! Where the reductions accumulate, and what that is allowed to change (#1315).
//!
//! `inner`, `norm`, `tr` and the compact full trace sum over a whole coupled
//! region, so they accumulate in `WideScalar::Wide` — the payload type itself
//! for `f64`/`Complex64`, double precision for `f32`/`Complex32`. Two things
//! need pinning, and neither is reachable through the widened-input oracle in
//! `single_precision_base.rs`, whose tolerance is far larger than the
//! difference between a wide and a narrow accumulator.
//!
//! 1. **Double precision matches an independent oracle.** Each `f64` and
//!    `Complex64` reduction is compared, under the workspace tolerance rule
//!    (`docs/testing_numerics.md`), with a double-double sum of the fixture
//!    entries recorded while `from_subblock_fn` fills them. The oracle does not
//!    depend on the order symtn accumulates in, so a valid reordering passes
//!    and a wrong weight, conjugation or pairing fails.
//!
//!    The fixtures are deliberately **not** exactly representable: the values
//!    come from `0.1`-based sequences. They cover all three storage forms
//!    (dense, compact diagonal, lazy adjoint), the three reduction kinds
//!    (`norm`, `inner`, `tr`/full trace), an abelian provider with
//!    `dim(c) == 1` and a non-abelian one where `dim(c) == 2`, so the
//!    quantum-dimension weight is a real factor rather than a no-op.
//!    `checked_generic` adds the Checked-Generic admission mode behind
//!    `cgc-gen`.
//!
//!    One micro-difference is *not* observable here and is worth recording:
//!    applying the weight componentwise (`re * w`, `im * w`) rather than as
//!    the full complex product with `w + 0i` differs only when the real part
//!    of a block's contribution is a signed zero or non-finite, and a signed
//!    zero is then flattened by the `+0.0` the accumulator starts from — no
//!    public input reaches it. The `negative zero` fixtures check what *is*
//!    observable: `-0.0` components entering the lazy-adjoint kernel, with
//!    the other component non-dyadic so the result is nonzero and the
//!    conjugation direction shows in its sign.
//!
//! 2. **Single precision really does accumulate wide.** The fixture is one
//!    entry of `8192.0f32` and the rest `1.0f32`. `8192^2` is `2^26`, whose
//!    `f32` spacing is 8, so a narrow accumulator swallows every `1.0` that
//!    follows and returns exactly `2^26`, while a double accumulator returns
//!    `2^26 + (n - 1)`. With `n - 1 > 8` the two differ *after* narrowing to
//!    `f32`, so even `inner`, which returns the payload type, tells them
//!    apart. Each assertion is paired with `assert_ne!` against the value a
//!    narrow accumulator would produce, computed in this file, so the test
//!    cannot pass by both paths agreeing. These comparisons stay exact: every
//!    entry and partial sum is an integer below `2^53`, so every summation
//!    order gives the same bits.
//!
//! A third case pins range rather than precision: entries of `1e20f32` square
//! to `1e40`, far past the `f32` maximum of about `3.4e38`. `norm` must be
//! finite in every storage form.

use std::collections::HashMap;
use std::sync::Arc;

use symtn::sector::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use symtn::typed::{Complex32, Complex64, GradedSpace, SectorSpectrum, TensorMap};

#[path = "../../tests/support/numerics.rs"]
mod numerics;

#[path = "../../tests/support/fixtures.rs"]
mod fixtures;

use fixtures::host_runtime;

/// One U(1) charge with degeneracy 4: a single 4x4 dense block, a 4-entry
/// compact spectrum, and `dim(c) == 1` so the weight cannot mask a difference.
fn leg() -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 4)]).unwrap()
}

/// Degeneracy 32: 1024 dense entries and a 32-entry compact spectrum, both
/// long enough for the swallowed `1.0`s below to add up past one `f32` step of
/// `2^26`.
fn wide_leg() -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 32)]).unwrap()
}

// --- 1. double precision matches an independent oracle ------------------------

/// Three U(1) charges with unequal degeneracies: several coupled blocks of
/// different sizes, every `dim(c) == 1`.
fn u1_multi_leg() -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), 2),
            (U1Irrep::new(0), 3),
            (U1Irrep::new(1), 2),
        ],
    )
    .unwrap()
}

/// Spin 0 and spin 1/2, so the reductions weight one block by `dim(c) == 1`
/// and the other by `dim(c) == 2`.
fn su2_leg() -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 2),
        ],
    )
    .unwrap()
}

/// A non-dyadic real: no finite binary expansion, so sums of these depend on
/// the order they are added in.
fn real_at(step: usize) -> f64 {
    (step as f64) * 0.1 - 0.35 + 0.7 / (step as f64 + 1.3)
}

/// A second, independent non-dyadic sequence for the imaginary part and for
/// the right-hand operand.
fn other_at(step: usize) -> f64 {
    0.9 / (step as f64 + 2.1) - (step as f64) * 0.03
}

/// Counts fill calls so the value depends on position in the payload, which is
/// what makes a reordered accumulation visible.
fn stepper() -> impl FnMut() -> usize {
    let mut step = 0usize;
    move || {
        step += 1;
        step
    }
}

/// Double-double accumulator: `hi + lo` carries about 106 bits, so the oracle
/// sums below are exact to far below one `f64` ulp for these few-dozen-term,
/// well-scaled fixtures, independently of any summation order symtn picks.
#[derive(Clone, Copy, Default)]
struct DoubleDouble {
    hi: f64,
    lo: f64,
}

impl DoubleDouble {
    fn add(&mut self, value: f64) {
        let sum = self.hi + value;
        let virtual_value = sum - self.hi;
        let error = (self.hi - (sum - virtual_value)) + (value - virtual_value);
        self.hi = sum;
        self.lo += error;
    }

    /// Adds `a * b` exactly: `mul_add` recovers the rounding error of the
    /// product.
    fn add_product(&mut self, a: f64, b: f64) {
        let product = a * b;
        self.add(product);
        self.add(a.mul_add(b, -product));
    }

    fn value(self) -> f64 {
        self.hi + self.lo
    }
}

/// `sum_k weight_k * x_k * y_k` over complex operands, accumulated exactly
/// enough to serve as the oracle. `weight` is an integer quantum dimension:
/// `weight * x` is exact for the dyadic weights 1 and 2 and rounds once for
/// others (SU(3)'s 27), well inside the bound.
fn oracle_dot(terms: impl IntoIterator<Item = (f64, Complex64, Complex64)>) -> Complex64 {
    let (mut re, mut im) = (DoubleDouble::default(), DoubleDouble::default());
    for (weight, x, y) in terms {
        let x = x * weight;
        re.add_product(x.re, y.re);
        re.add_product(-x.im, y.im);
        im.add_product(x.re, y.im);
        im.add_product(x.im, y.re);
    }
    Complex64::new(re.value(), im.value())
}

/// Entries of a fixture, recorded while `from_subblock_fn` fills it, keyed by
/// coupled sector and (row, column) inside that sector's block.
type Entries = HashMap<(String, usize, usize), (f64, Complex64)>;

/// One reduction: symtn's value, the double-double oracle, and the number of
/// terms that reach it.
type Measurement = (&'static str, Complex64, Complex64, usize);

fn one() -> Complex64 {
    Complex64::new(1.0, 0.0)
}

/// Oracles for `norm`, `inner(a, b)`, `tr(a)`, `inner(a†, b†)`, `inner(a†, b)`
/// from the recorded entries. `(a†)_c[i][j] = conj(a_c[j][i])` in the block of
/// the same coupled sector, weighted by `dim(c)` like every other reduction.
fn dense_oracles(a: &Entries, b: &Entries) -> [Complex64; 5] {
    let transposed = |(sector, i, j): &(String, usize, usize)| (sector.clone(), *j, *i);
    let norm2 = oracle_dot(a.values().map(|&(w, x)| (w, x.conj(), x)));
    let inner = oracle_dot(a.iter().map(|(key, &(w, x))| (w, x.conj(), b[key].1)));
    let tr = oracle_dot(
        a.iter()
            .filter(|((_, i, j), _)| i == j)
            .map(|(_, &(w, x))| (w, x, one())),
    );
    // conj(conj(a_ji)) * conj(b_ji), summed over every (j, i).
    let adjoint_inner = oracle_dot(a.iter().map(|(key, &(w, x))| (w, x, b[key].1.conj())));
    // conj(conj(a_ji)) * b_ij.
    let mixed_inner = oracle_dot(a.iter().map(|(key, &(w, x))| (w, x, b[&transposed(key)].1)));
    [
        Complex64::new(norm2.re.sqrt(), 0.0),
        inner,
        tr,
        adjoint_inner,
        mixed_inner,
    ]
}

/// Records every entry `from_subblock_fn` produces, keyed by coupled sector and
/// block position, with the sector's quantum dimension as its weight.
macro_rules! recorded_fixture {
    ($runtime:expr, $leg:expr, $dtype:ty, $wide:expr, $dim:expr, $entry:expr) => {{
        let mut entries = Entries::new();
        let mut step = stepper();
        let tensor: TensorMap<_, $dtype> =
            TensorMap::from_subblock_fn($runtime, [$leg], [$leg], |trees, index| {
                let value: $dtype = $entry(step());
                let key = (format!("{:?}", trees.coupled()), index[0], index[1]);
                entries.insert(key, ($dim(trees.coupled()), $wide(value)));
                value
            })
            .unwrap();
        (tensor, entries)
    }};
}

macro_rules! dense_reductions {
    ($out:expr, $prefix:literal, $runtime:expr, $leg:expr, $dtype:ty, $wide:expr, $dim:expr) => {{
        let (a, a_entries) = recorded_fixture!($runtime, $leg, $dtype, $wide, $dim, |step| {
            <$dtype>::from_parts(real_at(step), other_at(step))
        });
        let (b, b_entries) = recorded_fixture!($runtime, $leg, $dtype, $wide, $dim, |step| {
            <$dtype>::from_parts(other_at(step + 5), real_at(step + 2))
        });
        let [norm, inner, tr, adjoint_inner, mixed_inner] = dense_oracles(&a_entries, &b_entries);
        let terms = a_entries.len();

        $out.push((
            concat!($prefix, " dense norm"),
            Complex64::new(a.norm(2.0).unwrap(), 0.0),
            norm,
            terms,
        ));
        $out.push((
            concat!($prefix, " dense inner"),
            $wide(a.inner(&b).unwrap()),
            inner,
            terms,
        ));
        $out.push((
            concat!($prefix, " dense tr"),
            $wide(a.tr().unwrap()),
            tr,
            terms,
        ));
        // Both operands lazy, then one lazy and one owned: the oriented kernel
        // is reached with a different conjugation pattern each way.
        $out.push((
            concat!($prefix, " lazy adjoint inner"),
            $wide(a.adjoint().unwrap().inner(&b.adjoint().unwrap()).unwrap()),
            adjoint_inner,
            terms,
        ));
        $out.push((
            concat!($prefix, " lazy adjoint mixed inner"),
            $wide(a.adjoint().unwrap().inner(&b).unwrap()),
            mixed_inner,
            terms,
        ));
    }};
}

/// Compact diagonal storage has its own reductions. Its full-trace row reads
/// the rank-0 result through the multiplicity-free `scalar`, so this is
/// separate from [`dense_reductions`]; checked Generic compact reductions are
/// covered by `compact_reductions_both_modes.rs`.
macro_rules! compact_reductions {
    ($out:expr, $prefix:literal, $runtime:expr, $leg:expr, $dtype:ty, $wide:expr, $dim:expr) => {{
        let mut diagonal = stepper();
        let mut spectrum = || {
            let mut recorded = Vec::new();
            let spectrum = $leg
                .sectors()
                .unwrap()
                .into_iter()
                .zip($leg.degeneracies())
                .map(|(sector, &degeneracy)| {
                    let values: Vec<$dtype> = (0..degeneracy)
                        .map(|_| {
                            let step = diagonal();
                            <$dtype>::from_parts(real_at(step), other_at(step))
                        })
                        .collect();
                    recorded.extend(values.iter().map(|&v| ($dim(&sector), $wide(v))));
                    SectorSpectrum { sector, values }
                })
                .collect::<Vec<_>>();
            (spectrum, recorded)
        };
        let (values, compact_entries) = spectrum();
        let compact: TensorMap<_, $dtype> = TensorMap::diagonal($runtime, $leg, values).unwrap();
        let (values, other_entries) = spectrum();
        let other: TensorMap<_, $dtype> = TensorMap::diagonal($runtime, $leg, values).unwrap();
        let terms = compact_entries.len();
        let norm2 = oracle_dot(compact_entries.iter().map(|&(w, x)| (w, x.conj(), x)));
        let inner = oracle_dot(
            compact_entries
                .iter()
                .zip(&other_entries)
                .map(|(&(w, x), &(_, y))| (w, x.conj(), y)),
        );
        let tr = oracle_dot(compact_entries.iter().map(|&(w, x)| (w, x, one())));

        $out.push((
            concat!($prefix, " compact norm"),
            Complex64::new(compact.norm(2.0).unwrap(), 0.0),
            Complex64::new(norm2.re.sqrt(), 0.0),
            terms,
        ));
        $out.push((
            concat!($prefix, " compact inner"),
            $wide(compact.inner(&other).unwrap()),
            inner,
            terms,
        ));
        $out.push((
            concat!($prefix, " compact tr"),
            $wide(compact.tr().unwrap()),
            tr,
            terms,
        ));
        $out.push((
            concat!($prefix, " compact full trace"),
            $wide(compact.trace_pairs(&[(0, 1)]).unwrap().scalar().unwrap()),
            tr,
            terms,
        ));
    }};
}

/// Bridges the two payload dtypes into one `Complex64` for the oracle.
trait FromParts: Copy {
    fn from_parts(re: f64, im: f64) -> Self;
}

impl FromParts for f64 {
    fn from_parts(re: f64, _im: f64) -> Self {
        re
    }
}

impl FromParts for Complex64 {
    fn from_parts(re: f64, im: f64) -> Self {
        Self::new(re, im)
    }
}

fn u1_dim(_: &U1Irrep) -> f64 {
    1.0
}

fn su2_dim(sector: &SU2Irrep) -> f64 {
    (sector.twice_spin() + 1) as f64
}

fn double_precision_measurements() -> Vec<Measurement> {
    let runtime = host_runtime();
    let u1 = u1_multi_leg();
    let su2 = su2_leg();
    let real = |v: f64| Complex64::new(v, 0.0);
    let complex = |v| v;
    let mut out = Vec::new();
    dense_reductions!(out, "f64 u1", &runtime, &u1, f64, real, u1_dim);
    compact_reductions!(out, "f64 u1", &runtime, &u1, f64, real, u1_dim);
    dense_reductions!(out, "f64 su2", &runtime, &su2, f64, real, su2_dim);
    compact_reductions!(out, "f64 su2", &runtime, &su2, f64, real, su2_dim);
    dense_reductions!(out, "c64 u1", &runtime, &u1, Complex64, complex, u1_dim);
    compact_reductions!(out, "c64 u1", &runtime, &u1, Complex64, complex, u1_dim);
    dense_reductions!(out, "c64 su2", &runtime, &su2, Complex64, complex, su2_dim);
    compact_reductions!(out, "c64 su2", &runtime, &su2, Complex64, complex, su2_dim);

    // Negative zeros in the payload, carried through the lazy-adjoint kernel.
    // Each operand has `-0.0` in one component and a non-dyadic value in the
    // other, so the conjugation the oriented inner applies shows in the sign
    // of a *nonzero* imaginary part.
    let (signed, signed_entries) =
        recorded_fixture!(&runtime, &u1, Complex64, complex, u1_dim, |step| {
            Complex64::new(real_at(step), -0.0)
        });
    let (probe, probe_entries) =
        recorded_fixture!(&runtime, &u1, Complex64, complex, u1_dim, |step| {
            Complex64::new(-0.0, other_at(step))
        });
    let [_, _, tr, adjoint_inner, mixed_inner] = dense_oracles(&signed_entries, &probe_entries);
    let terms = signed_entries.len();
    out.push((
        "c64 negative zero lazy adjoint inner",
        signed
            .adjoint()
            .unwrap()
            .inner(&probe.adjoint().unwrap())
            .unwrap(),
        adjoint_inner,
        terms,
    ));
    out.push((
        "c64 negative zero lazy adjoint mixed inner",
        signed.adjoint().unwrap().inner(&probe).unwrap(),
        mixed_inner,
        terms,
    ));
    out.push(("c64 negative zero tr", signed.tr().unwrap(), tr, terms));
    out
}

fn assert_measurements_match_the_oracle(measurements: &[Measurement]) {
    for &(what, got, want, terms) in measurements {
        numerics::assert_close(what, got, want, terms);
    }
}

#[test]
fn double_precision_reductions_match_a_double_double_oracle() {
    let measurements = double_precision_measurements();
    // 8 reductions per (dtype, provider) pair plus the 3 negative-zero probes.
    assert_eq!(measurements.len(), 4 * 9 + 3);
    assert_measurements_match_the_oracle(&measurements);
}

// --- 2. single precision accumulates wide ------------------------------------

/// The one large entry; `8192^2 == 2^26`, where the `f32` spacing is 8.
const PEAK: f32 = 8192.0;

/// `sum |x|^2` as a narrow accumulator computes it: `2^26` first, then `n - 1`
/// additions of `1.0` that each round straight back to `2^26`.
fn narrow_sum(n: usize) -> f32 {
    let mut sum = PEAK * PEAK;
    for _ in 1..n {
        sum += 1.0f32 * 1.0f32;
    }
    sum
}

/// `sum |x|^2` as a double accumulator computes it: exact.
fn wide_sum(n: usize) -> f64 {
    f64::from(PEAK) * f64::from(PEAK) + (n - 1) as f64
}

#[test]
fn mixed_compact_dense_single_precision_cancellation_accumulates_wide() {
    let runtime = host_runtime();
    let leg = wide_leg();
    let large = PEAK * PEAK;
    let mut narrow = large;
    for _ in 1..31 {
        narrow += 1.0;
    }
    narrow -= large;
    assert_ne!(narrow, 30.0, "fixture must distinguish narrow accumulation");

    let real_dense: TensorMap<U1FusionRule, f32> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
            if index[0] != index[1] {
                0.0
            } else if index[0] == 0 {
                large
            } else if index[0] == 31 {
                -large
            } else {
                1.0
            }
        })
        .unwrap();
    let real_compact: TensorMap<U1FusionRule, f32> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![1.0; 32],
        }],
    )
    .unwrap();
    let real_lazy = real_dense.adjoint().unwrap();
    for value in [
        real_compact.inner(&real_dense).unwrap(),
        real_dense.inner(&real_compact).unwrap(),
        real_compact.inner(&real_lazy).unwrap(),
        real_lazy.inner(&real_compact).unwrap(),
    ] {
        assert_eq!(value, 30.0);
    }

    let complex_dense = real_dense.convert::<Complex32>();
    let complex_compact = real_compact.convert::<Complex32>();
    let complex_lazy = complex_dense.adjoint().unwrap();
    for value in [
        complex_compact.inner(&complex_dense).unwrap(),
        complex_dense.inner(&complex_compact).unwrap(),
        complex_compact.inner(&complex_lazy).unwrap(),
        complex_lazy.inner(&complex_compact).unwrap(),
    ] {
        assert_eq!(value, Complex32::new(30.0, 0.0));
    }
}

#[test]
fn single_precision_reductions_accumulate_in_double() {
    let runtime = host_runtime();
    let leg = wide_leg();
    let dense_entries = 32 * 32;
    let dense: TensorMap<U1FusionRule, f32> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
            if index == [0, 0] {
                PEAK
            } else {
                1.0
            }
        })
        .unwrap();
    assert_eq!(dense.dense_data().unwrap().len(), dense_entries);

    let expected = wide_sum(dense_entries);
    assert_ne!(
        expected as f32,
        narrow_sum(dense_entries),
        "the fixture must distinguish a wide from a narrow accumulator in f32"
    );

    // `inner(self, self)` is `sum |x|^2` unweighted (dim(c) == 1).
    assert_eq!(dense.inner(&dense).unwrap(), expected as f32, "dense inner");
    // `norm` returns f64, so it shows the accumulator without narrowing.
    assert_eq!(dense.norm(2.0).unwrap(), expected.sqrt(), "dense norm");
    assert_ne!(
        dense.norm(2.0).unwrap(),
        f64::from(narrow_sum(dense_entries)).sqrt()
    );

    // The lazy-adjoint operands run `oriented_fusion_inner`, a different
    // kernel with its own per-block and cross-block accumulators.
    let lazy = dense.adjoint().unwrap();
    assert_eq!(
        lazy.inner(&lazy).unwrap(),
        expected as f32,
        "lazy adjoint inner must accumulate like the dense form"
    );

    // Compact diagonal storage: `compact_inner`, a third accumulator.
    let mut values = vec![1.0f32; 32];
    values[0] = PEAK;
    let compact: TensorMap<U1FusionRule, f32> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values,
        }],
    )
    .unwrap();
    let compact_expected = wide_sum(32);
    assert_ne!(compact_expected as f32, narrow_sum(32));
    assert_eq!(
        compact.inner(&compact).unwrap(),
        compact_expected as f32,
        "compact inner"
    );
    assert_eq!(
        compact.norm(2.0).unwrap(),
        compact_expected.sqrt(),
        "compact norm"
    );

    // Compact `tr` and the compact full trace sum the values themselves, which
    // are exact here; what they must not do is disagree with each other.
    let trace = PEAK + 31.0;
    assert_eq!(compact.tr().unwrap(), trace, "compact tr");
    assert_eq!(
        compact.trace_pairs(&[(0, 1)]).unwrap().scalar().unwrap(),
        trace,
        "compact full trace"
    );
}

#[test]
fn complex32_reductions_accumulate_in_double() {
    let runtime = host_runtime();
    let leg = wide_leg();
    let dense_entries = 32 * 32;
    let dense: TensorMap<U1FusionRule, Complex32> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, index| {
            Complex32::new(if index == [0, 0] { PEAK } else { 1.0 }, 0.0)
        })
        .unwrap();
    let expected = Complex32::new(wide_sum(dense_entries) as f32, 0.0);
    assert_ne!(expected.re, narrow_sum(dense_entries));
    assert_eq!(dense.inner(&dense).unwrap(), expected, "dense inner");
    assert_eq!(
        dense.norm(2.0).unwrap(),
        wide_sum(dense_entries).sqrt(),
        "dense norm"
    );
    let lazy = dense.adjoint().unwrap();
    assert_eq!(lazy.inner(&lazy).unwrap(), expected, "lazy adjoint inner");
}

#[test]
fn single_precision_norm_stays_finite_where_a_narrow_accumulator_overflows() {
    // 1e20^2 = 1e40, past the f32 maximum of ~3.4e38: a narrow accumulator
    // saturates to +inf and the storage form would decide the answer.
    let runtime = host_runtime();
    let leg = leg();
    let value = 1e20f32;
    assert!(
        (value * value).is_infinite(),
        "the fixture must overflow an f32 accumulator"
    );

    let dense: TensorMap<U1FusionRule, f32> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, _| value).unwrap();
    let compact: TensorMap<U1FusionRule, f32> = TensorMap::diagonal(
        &runtime,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![value; 4],
        }],
    )
    .unwrap();

    for (what, norm) in [
        ("dense", dense.norm(2.0).unwrap()),
        ("compact", compact.norm(2.0).unwrap()),
        ("lazy adjoint", dense.adjoint().unwrap().norm(2.0).unwrap()),
    ] {
        assert!(norm.is_finite(), "{what} norm overflowed: {norm}");
    }
    // `inner` returns the payload type, and `4 * 4 * 1e40` genuinely exceeds
    // the `f32` range, so `+inf` there is the correct narrowing of a finite
    // wide accumulator rather than an accumulator that saturated. `norm`
    // above is what distinguishes the two, since it returns `f64`.
    let lazy = dense.adjoint().unwrap();
    for (what, value) in [
        ("dense inner", dense.inner(&dense).unwrap()),
        ("lazy adjoint inner", lazy.inner(&lazy).unwrap()),
    ] {
        assert!(value.is_infinite(), "{what}: {value}");
    }
}

/// The Checked-Generic admission mode reaches the same reductions through its
/// own dispatch and its own fallible weight lookup, so it needs its own checks.
/// SU(3) with the adjoint irrep also carries outer multiplicity.
#[cfg(feature = "cgc-gen")]
mod checked_generic {
    use super::*;
    use symtn::sector::SUNFusionRule;

    /// Weyl dimension of the SU(3) irrep with Dynkin labels `(p, q)`:
    /// `(p + 1)(q + 1)(p + q + 2) / 2`.
    fn su3_dim(labels: &[i64]) -> f64 {
        let (p, q) = (labels[0] as f64, labels[1] as f64);
        (p + 1.0) * (q + 1.0) * (p + q + 2.0) / 2.0
    }

    fn measurements() -> Vec<Measurement> {
        let runtime = host_runtime();
        let provider = Arc::new(SUNFusionRule::new(3).unwrap());
        let adjoint = vec![2i64, 2];
        let leg = GradedSpace::try_new(Arc::clone(&provider), [(adjoint.clone(), 2)]).unwrap();
        // Compact rows need the multiplicity-free `scalar` (see
        // `compact_reductions`), so only the dense and lazy-adjoint
        // reductions are checked here.
        let real = |v: f64| Complex64::new(v, 0.0);
        let complex = |v| v;
        let mut out = Vec::new();
        dense_reductions!(out, "f64 su3", &runtime, &leg, f64, real, su3_dim);
        dense_reductions!(out, "c64 su3", &runtime, &leg, Complex64, complex, su3_dim);
        out
    }

    #[test]
    fn checked_generic_reductions_match_a_double_double_oracle() {
        let measurements = measurements();
        assert_eq!(measurements.len(), 2 * 5);
        assert_measurements_match_the_oracle(&measurements);
    }
}

#[test]
fn dense_inner_lane_lengths_match_serial_oracle_all_dtypes() {
    let runtime = host_runtime();
    macro_rules! check {
        ($rule:expr, $sector:expr, $weight:expr, $dtype:ty, $cast:expr) => {
            for len in [0, 8, 13] {
                let rule = Arc::new($rule);
                let out = GradedSpace::try_new(rule.clone(), [($sector, len)]).unwrap();
                let input = GradedSpace::try_new(rule, [($sector, 1)]).unwrap();
                let a: TensorMap<_, $dtype> =
                    TensorMap::from_subblock_fn(&runtime, [&out], [&input], |_, index| {
                        ($cast)(Complex64::new(
                            real_at(index[0] + 1),
                            other_at(index[0] + 2),
                        ))
                    })
                    .unwrap();
                let b: TensorMap<_, $dtype> =
                    TensorMap::from_subblock_fn(&runtime, [&out], [&input], |_, index| {
                        ($cast)(Complex64::new(
                            other_at(index[0] + 3),
                            real_at(index[0] + 4),
                        ))
                    })
                    .unwrap();
                let aa = a.dense_data().unwrap();
                let bb = b.dense_data().unwrap();
                assert_eq!(aa.len(), len);
                let mut serial = Complex64::new(0.0, 0.0);
                for (&a, &b) in aa.iter().zip(bb) {
                    serial += numerics::Numeric::wide(a).conj() * numerics::Numeric::wide(b);
                }
                // At most `len` bilinear terms, weighted by the known irrep dimension.
                numerics::assert_close(
                    "dense lane inner",
                    a.inner(&b).unwrap(),
                    ($cast)(serial * $weight),
                    len,
                );
            }
        };
    }
    macro_rules! providers {
        ($dtype:ty, $cast:expr) => {
            check!(U1FusionRule, U1Irrep::new(0), 1.0, $dtype, $cast);
            check!(
                SU2FusionRule,
                SU2Irrep::from_twice_spin(1),
                2.0,
                $dtype,
                $cast
            );
        };
    }
    providers!(f64, |v: Complex64| v.re);
    providers!(Complex64, |v: Complex64| v);
    providers!(f32, |v: Complex64| v.re as f32);
    providers!(Complex32, |v: Complex64| Complex32::new(
        v.re as f32,
        v.im as f32
    ));
}

#[test]
fn dense_inner_cancellation_accumulates_wide() {
    let runtime = host_runtime();
    macro_rules! check {
        ($rule:expr, $sector:expr, $weight:expr, $dtype:ty, $cast:expr) => {{
            let rule = Arc::new($rule);
            let out = GradedSpace::try_new(rule.clone(), [($sector, 17)]).unwrap();
            let input = GradedSpace::try_new(rule, [($sector, 1)]).unwrap();
            // Each lane sees a large value, a unit value, then cancellation;
            // even an eight-lane f32 accumulator loses the unit in lane zero.
            let mut values = [1.0_f32; 17];
            values[0] = 1e8;
            values[16] = -1e8;
            let narrow = values.iter().copied().sum::<f32>();
            assert!((narrow - 15.0).abs() > 1.0);
            let a: TensorMap<_, $dtype> =
                TensorMap::from_subblock_fn(&runtime, [&out], [&input], |_, _| ($cast)(1.0))
                    .unwrap();
            let b: TensorMap<_, $dtype> =
                TensorMap::from_subblock_fn(&runtime, [&out], [&input], |_, index| {
                    ($cast)(values[index[0]])
                })
                .unwrap();
            numerics::assert_close(
                "wide cancellation",
                a.inner(&b).unwrap(),
                ($cast)(15.0 * $weight),
                17,
            );
        }};
    }
    check!(U1FusionRule, U1Irrep::new(0), 1.0, f32, |v| v);
    check!(SU2FusionRule, SU2Irrep::from_twice_spin(1), 2.0, f32, |v| v);
    check!(U1FusionRule, U1Irrep::new(0), 1.0, Complex32, |v| {
        Complex32::new(v, 0.0)
    });
    check!(
        SU2FusionRule,
        SU2Irrep::from_twice_spin(1),
        2.0,
        Complex32,
        |v| Complex32::new(v, 0.0)
    );
}
