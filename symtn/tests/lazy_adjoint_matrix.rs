//! The lazy-adjoint matrix: every operation on a lazy adjoint `t^H` against
//! an independent oracle, over U(1), SU(2), fermion parity times U(1) and
//! checked Generic SU(3) with outer multiplicity, `f64` and `Complex64`.
//!
//! This module: materialization and `add` with a lazy-adjoint operand are
//! exact block copies and per-element `alpha`/`beta` updates; one lazy and one
//! owned operand give an owned result, two lazy operands a lazy adjoint.
//! The `factorizations` submodule factors `t^H` against its materialization.
//! Transforms and the trace are `lazy_adjoint_shared_owner.rs`, contraction
//! `lazy_adjoint_contract_both_modes.rs` and
//! `lazy_adjoint_non_self_dual_contract.rs`, complex symbols
//! `complex_symbol_lazy_adjoint.rs`, and `axpby_into` and transforms `_into`
//! `into_admission_matrix.rs`.
//!
//! The oracle is independent of every strided kernel: each entry is filled
//! with `from_subblock_fn` from a hash of its own fusion-tree pair and degeneracy
//! index, and the adjoint entry is TensorKit's `block(t', c) = block(t, c)'`
//! written per element: the codomain and domain trees swap, the index moves
//! the domain axes first, and the value is conjugated. The `add` oracle spells
//! out the documented update order: `alpha = 1, beta = 0` is a copy, never
//! `1 * x`, so complex `inf + 0i` and signed zeros survive bit for bit.
//! Copies (`(1, 0)`, `(0, 1)`, `(0, 0)`) compare bit for bit; a scaled update,
//! and any update of two lazy operands (the lazy adjoint of the parents' owned
//! update), is arithmetic, so its finite entries compare under the workspace tolerance
//! rule (`docs/testing_numerics.md`) and its nonfinite entries by class.

use std::hash::{Hash, Hasher};
use std::sync::Arc;
use symtn::typed::Engine;

use num_complex::Complex64;

use crate::numerics;
use symtn::sector::{
    product_sector, FermionParityFusionRule, SU2FusionRule, U1FusionRule, U1Irrep, Z2Irrep,
};
use symtn::typed::BlockFusionTrees;
use symtn::typed::{GradedSpace, Runtime, TensorMap};

use crate::legs::{su2_on, u1_on};

trait Exact:
    Copy + PartialEq + std::ops::Add<Output = Self> + std::ops::Mul<Output = Self> + numerics::Numeric
{
    const ZERO: Self;
    const ONE: Self;
    const ALPHA: Self;
    const BETA: Self;
    fn from_hash(re: f64, im: f64) -> Self;
    fn conj(self) -> Self;
    fn bits(self) -> (u64, u64);
}

impl Exact for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const ALPHA: Self = 0.75;
    const BETA: Self = -1.25;
    fn from_hash(re: f64, _: f64) -> Self {
        re
    }
    fn conj(self) -> Self {
        self
    }
    fn bits(self) -> (u64, u64) {
        (self.to_bits(), 0)
    }
}

impl Exact for Complex64 {
    const ZERO: Self = Complex64::new(0.0, 0.0);
    const ONE: Self = Complex64::new(1.0, 0.0);
    const ALPHA: Self = Complex64::new(0.75, -0.5);
    const BETA: Self = Complex64::new(-1.25, 0.375);
    fn from_hash(re: f64, im: f64) -> Self {
        Complex64::new(re, im)
    }
    fn conj(self) -> Self {
        Complex64::conj(&self)
    }
    fn bits(self) -> (u64, u64) {
        (self.re.to_bits(), self.im.to_bits())
    }
}

/// A deterministic entry per `(salt, tree pair, index)`, with signed zeros,
/// infinities and payload-carrying NaNs mixed into ordinary values.
fn component<S: Hash>(trees: &BlockFusionTrees<S>, indices: &[usize], salt: u8) -> f64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    // Codomain side, then domain side, spelled out so the adjoint oracle can
    // swap the two without depending on the struct's field order.
    (
        trees.coupled(),
        trees.codomain_uncoupled(),
        trees.codomain_innerlines(),
        trees.codomain_vertices(),
        trees.domain_uncoupled(),
        trees.domain_innerlines(),
        trees.domain_vertices(),
        indices,
        salt,
    )
        .hash(&mut hasher);
    value_from_hash(hasher.finish())
}

fn value_from_hash(bits: u64) -> f64 {
    const SPECIAL: [u64; 6] = [
        0x8000_0000_0000_0000,
        0x7ff0_0000_0000_0000,
        0xfff0_0000_0000_0000,
        0x7ff8_0000_0000_1399,
        0xfff8_0000_0000_0042,
        0x0000_0000_0000_0001,
    ];
    if bits.is_multiple_of(5) {
        return f64::from_bits(SPECIAL[(bits >> 8) as usize % SPECIAL.len()]);
    }
    let magnitude = 0.5 + (bits >> 11) as f64 / (1u64 << 53) as f64;
    let sign = if bits & 1 == 0 { 1.0 } else { -1.0 };
    sign * magnitude * f64::powi(2.0, ((bits >> 1) % 16) as i32 - 8)
}

fn entry<D: Exact, S: Hash>(trees: &BlockFusionTrees<S>, indices: &[usize], salt: u8) -> D {
    D::from_hash(
        component(trees, indices, 2 * salt),
        component(trees, indices, 2 * salt + 1),
    )
}

/// `entry` of the parent at the element the adjoint entry `(trees, indices)`
/// reads, conjugated: the parent's codomain is the adjoint's domain.
fn adjoint_entry<D: Exact, S: Hash + Clone>(
    trees: &BlockFusionTrees<S>,
    indices: &[usize],
    salt: u8,
) -> D {
    let adjoint_codomain_rank = trees.codomain_uncoupled().len();
    let mut parent_indices = indices[adjoint_codomain_rank..].to_vec();
    parent_indices.extend_from_slice(&indices[..adjoint_codomain_rank]);
    let side = |salt: u8| {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        (
            trees.coupled(),
            trees.domain_uncoupled(),
            trees.domain_innerlines(),
            trees.domain_vertices(),
            trees.codomain_uncoupled(),
            trees.codomain_innerlines(),
            trees.codomain_vertices(),
            parent_indices.as_slice(),
            salt,
        )
            .hash(&mut hasher);
        value_from_hash(hasher.finish())
    };
    D::from_hash(side(2 * salt), side(2 * salt + 1)).conj()
}

/// The documented per-element update of `add(lhs, rhs, alpha, beta)`.
fn updated<D: Exact>(alpha: D, beta: D, lhs: D, rhs: D) -> D {
    let step = |a: D, b: D, dst: D, src: D| {
        if b == D::ZERO {
            if a == D::ONE {
                src
            } else {
                a * src
            }
        } else if b == D::ONE {
            dst + a * src
        } else {
            b * dst + a * src
        }
    };
    let mut value = D::ZERO;
    if alpha != D::ZERO {
        value = step(alpha, D::ZERO, value, lhs);
    }
    if beta != D::ZERO {
        let keep = if alpha == D::ZERO { D::ZERO } else { D::ONE };
        value = step(beta, keep, value, rhs);
    }
    value
}

/// `add` of two lazy adjoints, `(conj(alpha) A + conj(beta) B)^H`, per
/// logical element: the owned update `scale(x, a) + scale(y, b)` on the
/// parents' (conjugated) entries, with VectorInterface's strong zero and a
/// multiply at an exact one.
fn parent_updated<D: Exact>(alpha: D, beta: D, lhs: D, rhs: D) -> D {
    let scale = |a: D, x: D| if a == D::ZERO { D::ZERO * a } else { a * x };
    (scale(alpha.conj(), lhs.conj()) + scale(beta.conj(), rhs.conj())).conj()
}

fn bits<D: Exact>(data: &[D]) -> Vec<(u64, u64)> {
    data.iter().map(|value| value.bits()).collect()
}

/// A scaled `add` entry is at most `beta * rhs + alpha * lhs`, four real
/// products per component (`terms = 4`). Nonfinite oracle components must be
/// reproduced by class (NaN, or the same infinity).
fn assert_update_close<D: Exact>(what: &str, got: &[D], want: &[D]) {
    assert_eq!(got.len(), want.len(), "{what}: payload lengths differ");
    let scale = want
        .iter()
        .flat_map(|value| {
            let value = value.wide();
            [value.re, value.im]
        })
        .filter(|value| value.is_finite())
        .fold(0.0, |max: f64, value| max.max(value.abs()));
    let bound = numerics::tolerance::<D>(4, scale);
    for (index, (got, want)) in got.iter().zip(want).enumerate() {
        let (got, want) = (got.wide(), want.wide());
        for (g, w) in [(got.re, want.re), (got.im, want.im)] {
            let agrees = if w.is_nan() {
                g.is_nan()
            } else if w.is_infinite() {
                g == w
            } else {
                (g - w).abs() <= bound
            };
            assert!(
                agrees,
                "{what}: entry {index} is {got:?} against the oracle {want:?} (tolerance {bound:e})"
            );
        }
    }
}

macro_rules! assert_exact_adjoint {
    ($engine:ident, $dtype:ty, [$($codomain:expr),+], [$($domain:expr),+]) => {{
        let runtime = Runtime::builder().build().unwrap();
        let codomain = vec![$($codomain),+];
        let domain = vec![$($domain),+];
        let parent = |$engine: &mut dyn Engine, salt| {
            TensorMap::<_, $dtype>::from_subblock_fn($engine,
                &runtime,
                codomain.iter().copied(),
                domain.iter().copied(),
                move |trees, indices| entry::<$dtype, _>(trees, indices, salt),
            )
            .unwrap()
        };
        let on_adjoint_space = |$engine: &mut dyn Engine, fill: &dyn Fn(&_, &[usize]) -> $dtype| {
            TensorMap::<_, $dtype>::from_subblock_fn($engine,
                &runtime,
                domain.iter().copied(),
                codomain.iter().copied(),
                |trees, indices| fill(trees, indices),
            )
            .unwrap()
        };
        let (first, second) = (parent($engine, 0), parent($engine, 1));
        let owned = on_adjoint_space($engine, &|trees, indices| entry::<$dtype, _>(trees, indices, 2));
        let expected = on_adjoint_space($engine, &|trees, indices| {
            adjoint_entry::<$dtype, _>(trees, indices, 0)
        });

        let components: Vec<f64> = bits(first.dense_data().unwrap())
            .into_iter()
            .flat_map(|(re, im)| [f64::from_bits(re), f64::from_bits(im)])
            .collect();

        let lazy = first.adjoint($engine).unwrap();
        assert_eq!(lazy.codomain(), expected.codomain());
        assert_eq!(lazy.domain(), expected.domain());
        assert!(expected.subblock_count() > 1);
        assert_eq!(bits(lazy.materialize($engine).unwrap().dense_data().unwrap()), bits(expected.dense_data().unwrap()));

        let lazy = first.adjoint($engine).unwrap();
        let other_lazy = second.adjoint($engine).unwrap();
        let (zero, one, alpha, beta) = (
            <$dtype as Exact>::ZERO,
            <$dtype as Exact>::ONE,
            <$dtype as Exact>::ALPHA,
            <$dtype as Exact>::BETA,
        );
        for (a, b) in [
            (one, zero),
            (zero, one),
            (one, one),
            (alpha, zero),
            (zero, beta),
            (alpha, one),
            (one, beta),
            (alpha, beta),
            (zero, zero),
        ] {
            let cases: [(&TensorMap<_, $dtype>, &TensorMap<_, $dtype>, u8, u8); 3] = [
                (&lazy, &owned, 0, 2),
                (&owned, &lazy, 2, 0),
                (&lazy, &other_lazy, 0, 1),
            ];
            for (lhs, rhs, lhs_salt, rhs_salt) in cases {
                let value = |salt: u8, trees: &_, indices: &[usize]| {
                    if salt == 2 {
                        entry::<$dtype, _>(trees, indices, 2)
                    } else {
                        adjoint_entry::<$dtype, _>(trees, indices, salt)
                    }
                };
                let both_lazy = rhs_salt == 1;
                let expected = on_adjoint_space($engine, &|trees, indices| {
                    let (x, y) = (value(lhs_salt, trees, indices), value(rhs_salt, trees, indices));
                    if both_lazy {
                        parent_updated(a, b, x, y)
                    } else {
                        updated(a, b, x, y)
                    }
                });
                // Two lazy operands give a lazy adjoint; its
                // materialization is the logical sum.
                let sum = lhs.axpby($engine, a, rhs, b).unwrap();
                // One lazy and one owned operand give an owned result.
                assert_eq!(
                    sum.dense_data().is_err(),
                    both_lazy,
                    "salts ({lhs_salt}, {rhs_salt}): storage form"
                );
                let sum = sum.materialize($engine).unwrap();
                // Two lazy operands are the lazy adjoint of their parents'
                // owned update, which multiplies by an exact one: an
                // arithmetic update, not a copy.
                let copy =
                    [(one, zero), (zero, one), (zero, zero)].contains(&(a, b)) && !both_lazy;
                if copy {
                    assert_eq!(
                        bits(sum.dense_data().unwrap()),
                        bits(expected.dense_data().unwrap()),
                        "salts ({lhs_salt}, {rhs_salt})"
                    );
                } else {
                    assert_update_close(
                        &format!("salts ({lhs_salt}, {rhs_salt})"),
                        sum.dense_data().unwrap(),
                        expected.dense_data().unwrap(),
                    );
                }
            }
        }
        // The operands stayed lazy views of untouched parents.
        assert_eq!(bits(first.dense_data().unwrap()), bits(parent($engine, 0).dense_data().unwrap()));
        components
    }};
}

#[test]
fn lazy_adjoint_materialization_and_add_are_exact_on_u1_legs() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let provider = Arc::new(U1FusionRule);
    let leg = u1_on(&provider, &[(-1, 3), (0, 2), (1, 4)]);
    let dual = leg.try_dual().unwrap();
    let other = u1_on(&provider, &[(-1, 1), (0, 3), (1, 2)]);
    let mut real = assert_exact_adjoint!(engine, f64, [&leg, &other], [&leg]);
    let mut complex = assert_exact_adjoint!(engine, Complex64, [&leg, &dual], [&other]);
    complex.extend(assert_exact_adjoint!(
        engine,
        Complex64,
        [&other],
        [&dual, &leg, &other]
    ));
    real.extend(assert_exact_adjoint!(
        engine,
        f64,
        [&dual, &other],
        [&leg, &other]
    ));
    // The bit-exact copies are only meaningful if the parents carry the
    // values a multiply by one would change: NaN, ±Inf and signed zeros.
    for (dtype, components) in [("f64", real), ("c64", complex)] {
        assert!(components.iter().any(|x| x.is_nan()), "{dtype}: no NaN");
        assert!(components.contains(&f64::INFINITY), "{dtype}: no +Inf");
        assert!(components.contains(&f64::NEG_INFINITY), "{dtype}: no -Inf");
        assert!(
            components.iter().any(|x| *x == 0.0 && x.is_sign_negative()),
            "{dtype}: no -0"
        );
    }
}

#[test]
fn lazy_adjoint_materialization_and_add_are_exact_on_su2_legs() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let provider = Arc::new(SU2FusionRule);
    let leg = su2_on(&provider, &[(0, 2), (1, 3), (2, 1)]);
    let other = su2_on(&provider, &[(0, 1), (1, 2)]);
    assert_exact_adjoint!(engine, f64, [&leg, &other], [&leg]);
    assert_exact_adjoint!(engine, Complex64, [&leg, &other, &other], [&leg]);
    assert_exact_adjoint!(engine, Complex64, [&other, &leg], [&leg, &other]);
}

#[test]
fn lazy_adjoint_materialization_and_add_are_exact_on_fz2_u1_legs() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let provider = Arc::new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule)
            .unwrap(),
    );
    let label = |odd: bool, charge: i32| {
        product_sector(
            if odd { Z2Irrep::ODD } else { Z2Irrep::EVEN },
            U1Irrep::new(charge),
        )
    };
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [
            (label(false, 0), 3),
            (label(true, 1), 2),
            (label(true, -1), 2),
        ],
    )
    .unwrap();
    let dual = leg.try_dual().unwrap();
    let other = GradedSpace::try_new(
        Arc::clone(&provider),
        [
            (label(false, 0), 2),
            (label(true, 1), 1),
            (label(false, 1), 3),
        ],
    )
    .unwrap();
    assert_exact_adjoint!(engine, f64, [&leg, &dual], [&leg]);
    assert_exact_adjoint!(engine, Complex64, [&leg, &other], [&dual]);
    assert_exact_adjoint!(engine, Complex64, [&dual], [&leg, &other]);
}

/// Checked-Generic `add` (`host_add_impl`) and materialization on SU(3):
/// `adj ⊗ adj → adj` has outer multiplicity two, so blocks that differ only
/// by their vertex label must each keep their own entries.
#[cfg(feature = "cgc-gen")]
#[test]
fn lazy_adjoint_materialization_and_add_are_exact_on_su3_multiplicity_legs() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    use symtn::sector::LieGroupFusionRule;

    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let adjoint = vec![2i64, 2];
    let trivial = vec![0i64, 0];
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(adjoint.clone(), 2), (trivial, 1)]).unwrap();
    let other = GradedSpace::try_new(Arc::clone(&provider), [(adjoint, 3)]).unwrap();
    let runtime = Runtime::builder().build().unwrap();
    let probe =
        TensorMap::<_, f64>::from_subblock_fn(engine, &runtime, [&leg, &other], [&leg], |_, _| 0.0)
            .unwrap();
    assert!(
        (0..probe.subblock_count()).any(|index| probe
            .subblock_fusion_trees(index)
            .unwrap()
            .codomain_vertices()
            .iter()
            .any(|vertex| vertex.get() == 2)),
        "fixture must carry a Generic vertex key mu = 2"
    );
    assert_exact_adjoint!(engine, f64, [&leg, &other], [&leg]);
    assert_exact_adjoint!(engine, Complex64, [&leg, &other], [&other, &leg]);
}

mod factorizations {
    //! Every factorization of a lazy adjoint equals the same factorization of its
    //! materialized adjoint, in both fusion modes: the same factor
    //! spaces, the same storage form (compact or dense), and the same values in
    //! the documented gauge (dense rounding tolerance).
    //!
    //! Oracle: `lazy.materialize()` is the exact block copy
    //! `block(t^H, c) = block(t, c)^H` (the parent module), factored by the
    //! ordinary owned path. The lazy receiver reaches each op with identity leg
    //! roles, so it is factored as a lazy adjoint rather than permuted into an
    //! owned tensor first.

    use std::sync::Arc;

    use num_complex::Complex64;
    use symtn::sector::{
        FermionParityFusionRule, ProductFusionRule, ProductSector, SU2FusionRule, SU2Irrep,
        U1FusionRule, U1Irrep, Z2Irrep,
    };
    use symtn::typed::{
        Eig, Eigh, GradedSpace, HermitianTol, LeftPolar, Lq, Qr, RightPolar, Runtime,
        SectorSpectrum, Svd, TensorMap,
    };

    const TOL: f64 = 1.0e-10;

    trait Distance: Copy {
        fn distance(self, other: Self) -> f64;
    }

    impl Distance for f64 {
        fn distance(self, other: Self) -> f64 {
            (self - other).abs()
        }
    }

    impl Distance for Complex64 {
        fn distance(self, other: Self) -> f64 {
            (self - other).norm()
        }
    }

    fn assert_spectra_close<S: PartialEq + std::fmt::Debug, V: Distance>(
        what: &str,
        lazy: &[SectorSpectrum<S, V>],
        eager: &[SectorSpectrum<S, V>],
    ) {
        assert_eq!(lazy.len(), eager.len(), "{what}: sector count");
        for (lazy, eager) in lazy.iter().zip(eager) {
            assert_eq!(lazy.sector, eager.sector, "{what}: sector");
            assert_eq!(lazy.values.len(), eager.values.len(), "{what}: length");
            for (&a, &b) in lazy.values.iter().zip(&eager.values) {
                assert!(a.distance(b) < TOL, "{what}: values differ");
            }
        }
    }

    /// Same spaces, same storage form, same values.
    macro_rules! assert_factor_close {
        ($engine:ident, $what:expr, $lazy:expr, $eager:expr) => {{
            let (what, lazy, eager) = ($what, &$lazy, &$eager);
            assert_eq!(lazy.codomain(), eager.codomain(), "{what}: codomain");
            assert_eq!(lazy.domain(), eager.domain(), "{what}: domain");
            assert_eq!(
                lazy.diagview($engine).is_ok(),
                eager.diagview($engine).is_ok(),
                "{what}: storage form"
            );
            let (lazy, eager) = (
                lazy.materialize($engine).unwrap(),
                eager.materialize($engine).unwrap(),
            );
            let (lazy, eager) = (lazy.dense_data().unwrap(), eager.dense_data().unwrap());
            assert_eq!(lazy.len(), eager.len(), "{what}: payload length");
            for (&a, &b) in lazy.iter().zip(eager) {
                assert!(Distance::distance(a, b) < TOL, "{what}: values differ");
            }
        }};
    }

    /// Every one-input factorization on the lazy adjoint of the rectangular
    /// `parent`, against the materialized adjoint.
    macro_rules! assert_rectangular_ops {
        ($engine:ident, $label:expr, $parent:expr) => {{
            let label = $label;
            let lazy = $parent.adjoint($engine).unwrap();
            let eager = lazy.materialize($engine).unwrap();
            let rows: Vec<usize> = (0..lazy.codomain_rank()).collect();
            let cols: Vec<usize> =
                (lazy.codomain_rank()..lazy.codomain_rank() + lazy.domain_rank()).collect();
            let (r, c) = (rows.as_slice(), cols.as_slice());

            for (name, lazy_qr, eager_qr) in [
                (
                    "qr_compact",
                    lazy.qr_compact($engine, r, c),
                    eager.qr_compact($engine, r, c),
                ),
                (
                    "qr_full",
                    lazy.qr_full($engine, r, c),
                    eager.qr_full($engine, r, c),
                ),
            ] {
                let (Qr { q, r: rr }, Qr { q: eq, r: er }) = (lazy_qr.unwrap(), eager_qr.unwrap());
                assert_factor_close!($engine, format!("{label} {name} q"), q, eq);
                assert_factor_close!($engine, format!("{label} {name} r"), rr, er);
            }
            for (name, lazy_lq, eager_lq) in [
                (
                    "lq_compact",
                    lazy.lq_compact($engine, r, c),
                    eager.lq_compact($engine, r, c),
                ),
                (
                    "lq_full",
                    lazy.lq_full($engine, r, c),
                    eager.lq_full($engine, r, c),
                ),
            ] {
                let (Lq { l, q }, Lq { l: el, q: eq }) = (lazy_lq.unwrap(), eager_lq.unwrap());
                assert_factor_close!($engine, format!("{label} {name} l"), l, el);
                assert_factor_close!($engine, format!("{label} {name} q"), q, eq);
            }
            for (name, lazy_svd, eager_svd) in [
                (
                    "svd_compact",
                    lazy.svd_compact($engine, r, c),
                    eager.svd_compact($engine, r, c),
                ),
                (
                    "svd_full",
                    lazy.svd_full($engine, r, c),
                    eager.svd_full($engine, r, c),
                ),
            ] {
                let (
                    Svd { u, s, vh },
                    Svd {
                        u: eu,
                        s: es,
                        vh: evh,
                    },
                ) = (lazy_svd.unwrap(), eager_svd.unwrap());
                assert_factor_close!($engine, format!("{label} {name} u"), u, eu);
                assert_factor_close!($engine, format!("{label} {name} s"), s, es);
                assert_factor_close!($engine, format!("{label} {name} vh"), vh, evh);
            }
            assert_spectra_close(
                &format!("{label} svd_vals"),
                &lazy.svd_vals($engine, r, c).unwrap(),
                &eager.svd_vals($engine, r, c).unwrap(),
            );
            assert_factor_close!(
                $engine,
                format!("{label} left_null"),
                lazy.left_null($engine, r, c).unwrap(),
                eager.left_null($engine, r, c).unwrap()
            );
            assert_factor_close!(
                $engine,
                format!("{label} right_null"),
                lazy.right_null($engine, r, c).unwrap(),
                eager.right_null($engine, r, c).unwrap()
            );
            assert_factor_close!(
                $engine,
                format!("{label} pinv"),
                lazy.pinv($engine, r, c, 1.0e-12).unwrap(),
                eager.pinv($engine, r, c, 1.0e-12).unwrap()
            );
        }};
    }

    /// Every endomorphism factorization on the lazy adjoint of `parent` (square
    /// coupled-sector blocks, so both polar directions apply), and the Hermitian
    /// ones on the lazy adjoint of `hermitian`.
    macro_rules! assert_endomorphism_ops {
        ($engine:ident, $label:expr, $parent:expr, $hermitian:expr) => {{
            let label = $label;
            let lazy = $parent.adjoint($engine).unwrap();
            let eager = lazy.materialize($engine).unwrap();
            let rows: Vec<usize> = (0..lazy.codomain_rank()).collect();
            let cols: Vec<usize> =
                (lazy.codomain_rank()..lazy.codomain_rank() + lazy.domain_rank()).collect();
            let (r, c) = (rows.as_slice(), cols.as_slice());

            let (LeftPolar { w, p }, LeftPolar { w: ew, p: ep }) = (
                lazy.left_polar($engine, r, c).unwrap(),
                eager.left_polar($engine, r, c).unwrap(),
            );
            assert_factor_close!($engine, format!("{label} left_polar w"), w, ew);
            assert_factor_close!($engine, format!("{label} left_polar p"), p, ep);
            let (RightPolar { p, wh }, RightPolar { p: ep, wh: ewh }) = (
                lazy.right_polar($engine, r, c).unwrap(),
                eager.right_polar($engine, r, c).unwrap(),
            );
            assert_factor_close!($engine, format!("{label} right_polar p"), p, ep);
            assert_factor_close!($engine, format!("{label} right_polar wh"), wh, ewh);
            assert_factor_close!(
                $engine,
                format!("{label} inv"),
                lazy.inv($engine, r, c).unwrap(),
                eager.inv($engine, r, c).unwrap()
            );
            assert_factor_close!(
                $engine,
                format!("{label} exp"),
                lazy.exp($engine, r, c).unwrap(),
                eager.exp($engine, r, c).unwrap()
            );
            assert_spectra_close(
                &format!("{label} eig_vals"),
                &lazy.eig_vals($engine, r, c).unwrap(),
                &eager.eig_vals($engine, r, c).unwrap(),
            );
            let (Eig { d, v }, Eig { d: ed, v: ev }) = (
                lazy.eig_full($engine, r, c).unwrap(),
                eager.eig_full($engine, r, c).unwrap(),
            );
            assert_factor_close!($engine, format!("{label} eig_full d"), d, ed);
            assert_factor_close!($engine, format!("{label} eig_full v"), v, ev);

            let lazy = $hermitian.adjoint($engine).unwrap();
            let eager = lazy.materialize($engine).unwrap();
            let tol = HermitianTol::DEFAULT;
            assert_spectra_close(
                &format!("{label} eigh_vals"),
                &lazy.eigh_vals($engine, r, c, tol).unwrap(),
                &eager.eigh_vals($engine, r, c, tol).unwrap(),
            );
            let (Eigh { d, v }, Eigh { d: ed, v: ev }) = (
                lazy.eigh_full($engine, r, c, tol).unwrap(),
                eager.eigh_full($engine, r, c, tol).unwrap(),
            );
            assert_factor_close!($engine, format!("{label} eigh_full d"), d, ed);
            assert_factor_close!($engine, format!("{label} eigh_full v"), v, ev);
        }};
    }

    /// Rectangular `[a, b] <- [a]`, `[a] <- [a, b]` and `[a', b] <- [b, a']`
    /// (dual legs) and endomorphism `[a, a] <- [a, a]` fixtures
    /// with a Hermitian `X X^H`, for both scalar types.
    macro_rules! assert_all_ops {
        ($engine:ident, $label:expr, $runtime:expr, $a:expr, $b:expr) => {{
            let engine = &mut *$engine;
            let (runtime, a, b) = (&$runtime, &$a, &$b);
            macro_rules! for_dtype {
                ($d:ty, $seed:expr) => {{
                    let label = format!("{} {}", $label, stringify!($d));
                    let rectangular =
                        TensorMap::<_, $d>::rand_with_seed(engine, runtime, [a, b], [a], $seed)
                            .unwrap();
                    assert_rectangular_ops!(engine, format!("{label} tall"), rectangular);
                    let wide =
                        TensorMap::<_, $d>::rand_with_seed(engine, runtime, [a], [a, b], $seed + 1)
                            .unwrap();
                    assert_rectangular_ops!(engine, format!("{label} wide"), wide);
                    let dual = a.try_dual().unwrap();
                    let dual_legs = TensorMap::<_, $d>::rand_with_seed(
                        engine,
                        runtime,
                        [&dual, b],
                        [b, &dual],
                        $seed + 3,
                    )
                    .unwrap();
                    assert_rectangular_ops!(engine, format!("{label} dual"), dual_legs);
                    let square = TensorMap::<_, $d>::rand_with_seed(
                        engine,
                        runtime,
                        [a, a],
                        [a, a],
                        $seed + 2,
                    )
                    .unwrap();
                    let square_adjoint =
                        &square.adjoint(engine).unwrap().materialize(engine).unwrap();
                    let hermitian = square.compose(engine, square_adjoint).unwrap();
                    assert_endomorphism_ops!(engine, label, square, hermitian);
                }};
            }
            for_dtype!(f64, 1755);
            for_dtype!(Complex64, 2755);
        }};
    }

    #[test]
    fn u1_lazy_adjoint_factorizations_match_the_materialized_adjoint() {
        let engine = &mut symtn::typed::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let provider = Arc::new(U1FusionRule);
        let leg = |pairs: &[(i32, usize)]| {
            GradedSpace::try_new(
                Arc::clone(&provider),
                pairs.iter().map(|&(q, d)| (U1Irrep::new(q), d)),
            )
            .unwrap()
        };
        let a = leg(&[(-1, 2), (0, 3), (1, 1)]);
        let b = leg(&[(0, 2), (1, 3), (2, 1)]);
        assert_all_ops!(engine, "U(1)", runtime, a, b);
    }

    /// Fermion parity times U(1): the adjoint's swapped trees carry the
    /// fermionic exchange.
    #[test]
    fn fermionic_u1_lazy_adjoint_factorizations_match_the_materialized_adjoint() {
        let engine = &mut symtn::typed::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let provider = Arc::new(
            ProductFusionRule::<FermionParityFusionRule, U1FusionRule>::try_new(
                FermionParityFusionRule,
                U1FusionRule,
            )
            .unwrap(),
        );
        let leg = |pairs: &[(i32, usize)]| {
            GradedSpace::try_new(
                Arc::clone(&provider),
                pairs.iter().map(|&(q, d)| {
                    let parity = if q.rem_euclid(2) == 0 {
                        Z2Irrep::EVEN
                    } else {
                        Z2Irrep::ODD
                    };
                    (ProductSector::new(parity, U1Irrep::new(q)), d)
                }),
            )
            .unwrap()
        };
        let a = leg(&[(-1, 2), (0, 3), (1, 2)]);
        let b = leg(&[(0, 2), (1, 2)]);
        assert_all_ops!(engine, "fZ2xU(1)", runtime, a, b);
    }

    #[test]
    fn multiplicity_free_su2_lazy_adjoint_factorizations_match_the_materialized_adjoint() {
        let engine = &mut symtn::typed::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let provider = Arc::new(SU2FusionRule);
        let su2 = |pairs: &[(usize, usize)]| {
            GradedSpace::try_new(
                Arc::clone(&provider),
                pairs
                    .iter()
                    .map(|&(twice, degeneracy)| (SU2Irrep::from_twice_spin(twice), degeneracy)),
            )
            .unwrap()
        };
        let a = su2(&[(0, 2), (1, 2), (2, 1)]);
        let b = su2(&[(0, 1), (1, 2)]);
        assert_all_ops!(engine, "MF SU(2)", runtime, a, b);
    }

    #[cfg(feature = "cgc-gen")]
    #[test]
    fn checked_su2_lazy_adjoint_factorizations_match_the_materialized_adjoint() {
        let engine = &mut symtn::typed::CpuEngine::serial();
        use symtn::sector::LieGroupFusionRule;

        let runtime = Runtime::builder().build().unwrap();
        let provider = Arc::new(
            LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(2).unwrap()).unwrap(),
        );
        let a = GradedSpace::try_new(
            Arc::clone(&provider),
            [(vec![0i64], 2), (vec![1], 2), (vec![2], 1)],
        )
        .unwrap();
        let b =
            GradedSpace::try_new(Arc::clone(&provider), [(vec![0i64], 1), (vec![1], 2)]).unwrap();
        assert_all_ops!(engine, "checked SU(2)", runtime, a, b);
    }

    /// SU(3) `8 ⊗ 8 ∋ 8` twice: blocks that differ only by their vertex label
    /// carry their own entries, so the swapped factor spaces of a lazy adjoint
    /// must keep the multiplicity index on the right side.
    #[cfg(feature = "cgc-gen")]
    #[test]
    fn checked_su3_multiplicity_lazy_adjoint_factorizations_match_the_materialized_adjoint() {
        let engine = &mut symtn::typed::CpuEngine::serial();
        use symtn::sector::LieGroupFusionRule;

        let runtime = Runtime::builder().build().unwrap();
        let provider = Arc::new(
            LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap(),
        );
        let a = GradedSpace::try_new(Arc::clone(&provider), [(vec![2i64, 2], 1), (vec![0, 0], 1)])
            .unwrap();
        let b = GradedSpace::try_new(Arc::clone(&provider), [(vec![2i64, 2], 2)]).unwrap();
        let probe =
            TensorMap::<_, f64>::rand_with_seed(engine, &runtime, [&a, &a], [&a, &a], 0).unwrap();
        assert!(
            (0..probe.subblock_count()).any(|index| probe
                .subblock_fusion_trees(index)
                .unwrap()
                .codomain_vertices()
                .iter()
                .any(|vertex| vertex.get() == 2)),
            "fixture must carry a Generic vertex key mu = 2"
        );
        assert_all_ops!(engine, "checked SU(3)", runtime, a, b);
        let three =
            GradedSpace::try_new(Arc::clone(&provider), [(vec![1i64, 0], 2), (vec![0, 0], 1)])
                .unwrap();
        let three_bar =
            GradedSpace::try_new(Arc::clone(&provider), [(vec![0i64, 1], 3), (vec![1, 0], 1)])
                .unwrap();
        assert_all_ops!(engine, "checked SU(3) 3", runtime, three, three_bar);
    }

    /// The checked parent-reading seams (compact and full SVD, pinv) on
    /// non-self-dual SU(3) sectors `3`, `3̄` and on dual legs: the swapped hom
    /// space of the adjoint must keep each leg's sector and duality.
    #[cfg(feature = "cgc-gen")]
    #[test]
    fn checked_su3_seams_keep_non_self_dual_sectors_and_dual_legs() {
        let engine = &mut symtn::typed::CpuEngine::serial();
        use symtn::sector::LieGroupFusionRule;

        let runtime = Runtime::builder().build().unwrap();
        let provider = Arc::new(
            LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap(),
        );
        let three =
            GradedSpace::try_new(Arc::clone(&provider), [(vec![1i64, 0], 2), (vec![0, 0], 1)])
                .unwrap();
        let three_bar =
            GradedSpace::try_new(Arc::clone(&provider), [(vec![0i64, 1], 3), (vec![1, 0], 1)])
                .unwrap();
        let dual = three.try_dual().unwrap();
        assert!(dual.is_dual() && !three.is_dual());

        macro_rules! seams {
            ($engine:ident, $d:ty, $seed:expr, [$($codomain:expr),+], [$($domain:expr),+]) => {{
                let parent = TensorMap::<_, $d>::rand_with_seed($engine,
                    &runtime,
                    [$(&$codomain),+],
                    [$(&$domain),+],
                    $seed,
                )
                .unwrap();
                let label = format!("{} {}", stringify!($d), $seed);
                let lazy = parent.adjoint($engine).unwrap();
                let eager = lazy.materialize($engine).unwrap();
                let rows: Vec<usize> = (0..lazy.codomain_rank()).collect();
                let cols: Vec<usize> =
                    (lazy.codomain_rank()..lazy.codomain_rank() + lazy.domain_rank()).collect();
                let (r, c) = (rows.as_slice(), cols.as_slice());
                for (name, lazy_svd, eager_svd) in [
                    ("svd_compact", lazy.svd_compact($engine, r, c), eager.svd_compact($engine, r, c)),
                    ("svd_full", lazy.svd_full($engine, r, c), eager.svd_full($engine, r, c)),
                ] {
                    let (Svd { u, s, vh }, Svd { u: eu, s: es, vh: evh }) =
                        (lazy_svd.unwrap(), eager_svd.unwrap());
                    assert_factor_close!($engine, format!("{label} {name} u"), u, eu);
                    assert_factor_close!($engine, format!("{label} {name} s"), s, es);
                    assert_factor_close!($engine, format!("{label} {name} vh"), vh, evh);
                }
                assert_factor_close!($engine,
                    format!("{label} pinv"),
                    lazy.pinv($engine, r, c, 1.0e-12).unwrap(),
                    eager.pinv($engine, r, c, 1.0e-12).unwrap()
                );
            }};
        }
        seams!(engine, f64, 31, [three, three_bar], [three]);
        seams!(engine, Complex64, 32, [three, three_bar], [three]);
        seams!(engine, f64, 33, [three_bar], [three, dual]);
        seams!(engine, Complex64, 34, [three_bar], [three, dual]);
        seams!(engine, f64, 35, [dual, three_bar], [dual]);
        seams!(engine, Complex64, 36, [dual, three_bar], [dual]);
    }
}
