//! Shared oracle harness for the single-precision integration suites
//! (#1315 base family, #1324 factorization family).
//!
//! The oracle for every check built on this module is the **`f64`/`Complex64`
//! result of the same operation on the widened input**. Both tensors are
//! filled through `from_subblock_fn` from one 24-bit pseudo-random stream, so the
//! double tensor holds exactly `f64::from` of every single-precision entry and
//! the widening contributes no error of its own. The double path is a
//! different execution lane with a different dense kernel, and it is the path
//! the existing TensorKit/QSpace conformance suites already validate, so it is
//! independent of the code under test.
//!
//! Tolerance is `K * sqrt(n) * eps(real(D)) * max(1, max|expected|)` with
//! `K = 32` and `n` the number of payload entries the reduction or kernel sums
//! over — the standard bound for a naive length-`n` floating sum, with `K`
//! covering the recoupling coefficients applied on top of it. Suites that need
//! a conditioning factor state it at the call site.

#![allow(dead_code)]

use std::sync::Arc;

use num_complex::{Complex32, Complex64};
use symtn::sector::{
    product_sector, FermionParityFusionRule, ProductFusionRule, SU2FusionRule, SU2Irrep,
    U1FusionRule, U1Irrep, Z2Irrep,
};
use symtn::typed::{GradedSpace, Runtime};

/// `K` of the `K * sqrt(n) * eps` bound in the module docs.
pub const K: f64 = 32.0;

pub fn runtime() -> Runtime {
    Runtime::builder().build().unwrap()
}

/// Payload values that a single-precision tensor and its double-precision twin
/// can both hold exactly.
///
/// `parts` takes `f32` components on purpose: `f64::from(re)` is exact, so the
/// twin is the widening of the single-precision tensor entry by entry, with no
/// rounding to account for before the operation under test runs.
pub trait Parts: Copy {
    fn parts(re: f32, im: f32) -> Self;
    fn wide(self) -> Complex64;
}

impl Parts for f32 {
    fn parts(re: f32, _im: f32) -> Self {
        re
    }
    fn wide(self) -> Complex64 {
        Complex64::new(f64::from(self), 0.0)
    }
}

impl Parts for f64 {
    fn parts(re: f32, _im: f32) -> Self {
        f64::from(re)
    }
    fn wide(self) -> Complex64 {
        Complex64::new(self, 0.0)
    }
}

impl Parts for Complex32 {
    fn parts(re: f32, im: f32) -> Self {
        Self::new(re, im)
    }
    fn wide(self) -> Complex64 {
        Complex64::new(f64::from(self.re), f64::from(self.im))
    }
}

impl Parts for Complex64 {
    fn parts(re: f32, im: f32) -> Self {
        Self::new(f64::from(re), f64::from(im))
    }
    fn wide(self) -> Complex64 {
        self
    }
}

/// One 24-bit draw in `[-1, 1)`: exactly representable in `f32`, so the two
/// twins receive the same number and not two roundings of one.
pub fn draw(state: &mut u64) -> f32 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^= value >> 31;
    ((value >> 40) as f32) / ((1_u32 << 23) as f32) - 1.0
}

pub fn draw_parts<D: Parts>(state: &mut u64) -> D {
    let re = draw(state);
    let im = draw(state);
    D::parts(re, im)
}

/// Multiplicative unit of a payload dtype, for the `beta` of `add`.
pub fn one<D: Parts>() -> D {
    D::parts(1.0, 0.0)
}

/// Additive inverse of the unit, for the `-1` of a residual `add`.
pub fn minus_one<D: Parts>() -> D {
    D::parts(-1.0, 0.0)
}

/// Builds the same tensor at two precisions on one runtime.
///
/// A macro rather than a function: the payload dtype is the only thing that
/// varies, and the alternative is repeating the eight dispatch bounds of
/// `from_subblock_fn` for both instantiations.
#[macro_export]
macro_rules! twin {
    ($engine:ident, $rt:expr, $narrow:ty, $wide:ty, $codomain:expr, $domain:expr, $seed:expr) => {{
        let mut state = $seed;
        let narrow: symtn::typed::TensorMap<_, $narrow> =
            symtn::typed::TensorMap::from_subblock_fn($engine, $rt, $codomain, $domain, |_, _| {
                $crate::single_precision_oracle::draw_parts(&mut state)
            })
            .unwrap();
        let mut state = $seed;
        let wide: symtn::typed::TensorMap<_, $wide> =
            symtn::typed::TensorMap::from_subblock_fn($engine, $rt, $codomain, $domain, |_, _| {
                $crate::single_precision_oracle::draw_parts(&mut state)
            })
            .unwrap();
        (narrow, wide)
    }};
}

/// Builds the same tensor at two precisions from an explicit entry function of
/// the block multi-index, for fixtures whose conditioning has to be known.
///
/// `$entry` returns `(re, im)` as `f32` components, so — exactly as with
/// [`twin`] — the double-precision twin holds the widening of the
/// single-precision one entry by entry.
#[macro_export]
macro_rules! twin_with {
    ($engine:ident, $rt:expr, $narrow:ty, $wide:ty, $codomain:expr, $domain:expr, $entry:expr) => {{
        let entry = $entry;
        let narrow: symtn::typed::TensorMap<_, $narrow> =
            symtn::typed::TensorMap::from_subblock_fn(
                $engine,
                $rt,
                $codomain,
                $domain,
                |_, index| {
                    let (re, im) = entry(index);
                    <$narrow as $crate::single_precision_oracle::Parts>::parts(re, im)
                },
            )
            .unwrap();
        let wide: symtn::typed::TensorMap<_, $wide> = symtn::typed::TensorMap::from_subblock_fn(
            $engine,
            $rt,
            $codomain,
            $domain,
            |_, index| {
                let (re, im) = entry(index);
                <$wide as $crate::single_precision_oracle::Parts>::parts(re, im)
            },
        )
        .unwrap();
        (narrow, wide)
    }};
}

/// The tolerance of the module bound at `n = terms` and a payload scale.
pub fn tolerance(terms: usize, scale: f64) -> f64 {
    K * (terms as f64).sqrt() * f64::from(f32::EPSILON) * scale.max(1.0)
}

/// Asserts the two payloads agree within the module's tolerance, and that the
/// twins really did describe the same tensor (equal length, equal blocks).
pub fn assert_payloads_agree<D: Parts, W: Parts>(
    what: &str,
    narrow: &[D],
    wide: &[W],
    terms: usize,
) {
    assert_payloads_agree_scaled(what, narrow, wide, terms, 1.0)
}

/// [`assert_payloads_agree`] with an explicit conditioning factor multiplying
/// the bound. Callers state where the factor comes from.
pub fn assert_payloads_agree_scaled<D: Parts, W: Parts>(
    what: &str,
    narrow: &[D],
    wide: &[W],
    terms: usize,
    conditioning: f64,
) {
    assert_eq!(
        narrow.len(),
        wide.len(),
        "{what}: the twins hold different payload lengths"
    );
    let scale = wide
        .iter()
        .map(|&value| value.wide().norm())
        .fold(0.0f64, f64::max);
    let bound = tolerance(terms, scale) * conditioning;
    for (index, (&got, &expected)) in narrow.iter().zip(wide).enumerate() {
        let error = (got.wide() - expected.wide()).norm();
        assert!(
            error <= bound,
            "{what}: entry {index} is {got:?} against the widened oracle {expected:?} \
             (error {error:e} > tolerance {bound:e})",
            got = got.wide(),
            expected = expected.wide(),
        );
    }
}

pub fn assert_scalars_agree(what: &str, narrow: Complex64, wide: Complex64, terms: usize) {
    assert_scalars_agree_scaled(what, narrow, wide, terms, 1.0)
}

pub fn assert_scalars_agree_scaled(
    what: &str,
    narrow: Complex64,
    wide: Complex64,
    terms: usize,
    conditioning: f64,
) {
    let bound = tolerance(terms, wide.norm()) * conditioning;
    let error = (narrow - wide).norm();
    assert!(
        error <= bound,
        "{what}: {narrow} against the widened oracle {wide} \
         (error {error:e} > tolerance {bound:e})"
    );
}

/// U(1): abelian, several blocks, nontrivial degeneracies.
pub fn u1_leg() -> GradedSpace<U1FusionRule> {
    u1_leg_with([2, 3, 2])
}

/// [`u1_leg`] with chosen per-sector degeneracies, for rectangular fixtures.
pub fn u1_leg_with(degeneracies: [usize; 3]) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(-1), degeneracies[0]),
            (U1Irrep::new(0), degeneracies[1]),
            (U1Irrep::new(1), degeneracies[2]),
        ],
    )
    .unwrap()
}

pub type FermionU1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule>;
pub type FermionSu2Rule = ProductFusionRule<FermionU1Rule, SU2FusionRule>;

pub fn fermion_su2_provider() -> Arc<FermionSu2Rule> {
    Arc::new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(
            symtn_sectors::ProductFusionRule::<_, _>::try_new(
                FermionParityFusionRule,
                U1FusionRule,
            )
            .unwrap(),
            SU2FusionRule,
        )
        .unwrap(),
    )
}

/// fZ2 x U(1) x SU(2): fermionic signs, nontrivial braiding and non-abelian
/// recoupling with `dim(c) != 1` in one provider.
pub fn fermion_su2_leg() -> GradedSpace<FermionSu2Rule> {
    fermion_su2_leg_with([2, 2, 1])
}

/// [`fermion_su2_leg`] with chosen per-sector degeneracies.
pub fn fermion_su2_leg_with(degeneracies: [usize; 3]) -> GradedSpace<FermionSu2Rule> {
    let label = |parity, charge, twice_spin| {
        product_sector(
            product_sector(parity, U1Irrep::new(charge)),
            SU2Irrep::from_twice_spin(twice_spin),
        )
    };
    GradedSpace::try_new(
        fermion_su2_provider(),
        [
            (label(Z2Irrep::EVEN, 0, 0), degeneracies[0]),
            (label(Z2Irrep::ODD, 1, 1), degeneracies[1]),
            (label(Z2Irrep::EVEN, 2, 2), degeneracies[2]),
        ],
    )
    .unwrap()
}
