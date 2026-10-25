//! Lazy-adjoint materialization and `add` with a lazy-adjoint operand are
//! exact block copies and per-element `alpha`/`beta` updates (#1399).
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
//! update, A10), is arithmetic, so its finite entries compare under the workspace tolerance
//! rule (`docs/testing_numerics.md`) and its nonfinite entries by class.

use std::hash::{Hash, Hasher};
use std::sync::Arc;

use num_complex::Complex64;

use crate::numerics;
use symtn::sector::{
    product_sector, FermionParityFusionRule, SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep,
    Z2Irrep,
};
use symtn::typed::BlockFusionTrees;
use symtn::typed::{GradedSpace, Runtime, TensorMap};

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
    ($dtype:ty, [$($codomain:expr),+], [$($domain:expr),+]) => {{
        let runtime = Runtime::builder().dense_threads(1).build().unwrap();
        let codomain = vec![$($codomain),+];
        let domain = vec![$($domain),+];
        let parent = |salt| {
            TensorMap::<_, $dtype>::from_subblock_fn(
                &runtime,
                codomain.iter().copied(),
                domain.iter().copied(),
                move |trees, indices| entry::<$dtype, _>(trees, indices, salt),
            )
            .unwrap()
        };
        let on_adjoint_space = |fill: &dyn Fn(&_, &[usize]) -> $dtype| {
            TensorMap::<_, $dtype>::from_subblock_fn(
                &runtime,
                domain.iter().copied(),
                codomain.iter().copied(),
                |trees, indices| fill(trees, indices),
            )
            .unwrap()
        };
        let (first, second) = (parent(0), parent(1));
        let owned = on_adjoint_space(&|trees, indices| entry::<$dtype, _>(trees, indices, 2));
        let expected = on_adjoint_space(&|trees, indices| {
            adjoint_entry::<$dtype, _>(trees, indices, 0)
        });

        let lazy = first.adjoint().unwrap();
        assert_eq!(lazy.codomain(), expected.codomain());
        assert_eq!(lazy.domain(), expected.domain());
        assert!(expected.subblock_count() > 1);
        assert_eq!(bits(lazy.materialize().unwrap().dense_data().unwrap()), bits(expected.dense_data().unwrap()));

        let lazy = first.adjoint().unwrap();
        let other_lazy = second.adjoint().unwrap();
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
                let expected = on_adjoint_space(&|trees, indices| {
                    let (x, y) = (value(lhs_salt, trees, indices), value(rhs_salt, trees, indices));
                    if both_lazy {
                        parent_updated(a, b, x, y)
                    } else {
                        updated(a, b, x, y)
                    }
                });
                // Two lazy operands give a lazy adjoint (A10); its
                // materialization is the logical sum.
                let sum = lhs.axpby(a, rhs, b).unwrap().materialize().unwrap();
                // Two lazy operands are the lazy adjoint of their parents'
                // owned update (A10), which multiplies by an exact one: an
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
        assert_eq!(bits(first.dense_data().unwrap()), bits(parent(0).dense_data().unwrap()));
    }};
}

fn u1(provider: &Arc<U1FusionRule>, pairs: &[(i32, usize)]) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::clone(provider),
        pairs
            .iter()
            .map(|&(charge, degeneracy)| (U1Irrep::new(charge), degeneracy)),
    )
    .unwrap()
}

fn su2(provider: &Arc<SU2FusionRule>, pairs: &[(usize, usize)]) -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        Arc::clone(provider),
        pairs
            .iter()
            .map(|&(twice_spin, degeneracy)| (SU2Irrep::from_twice_spin(twice_spin), degeneracy)),
    )
    .unwrap()
}

#[test]
fn lazy_adjoint_materialization_and_add_are_exact_on_u1_legs() {
    let provider = Arc::new(U1FusionRule);
    let leg = u1(&provider, &[(-1, 3), (0, 2), (1, 4)]);
    let dual = leg.try_dual().unwrap();
    let other = u1(&provider, &[(-1, 1), (0, 3), (1, 2)]);
    assert_exact_adjoint!(f64, [&leg, &other], [&leg]);
    assert_exact_adjoint!(Complex64, [&leg, &dual], [&other]);
    assert_exact_adjoint!(Complex64, [&other], [&dual, &leg, &other]);
    assert_exact_adjoint!(f64, [&dual, &other], [&leg, &other]);
}

#[test]
fn lazy_adjoint_materialization_and_add_are_exact_on_su2_legs() {
    let provider = Arc::new(SU2FusionRule);
    let leg = su2(&provider, &[(0, 2), (1, 3), (2, 1)]);
    let other = su2(&provider, &[(0, 1), (1, 2)]);
    assert_exact_adjoint!(f64, [&leg, &other], [&leg]);
    assert_exact_adjoint!(Complex64, [&leg, &other, &other], [&leg]);
    assert_exact_adjoint!(Complex64, [&other, &leg], [&leg, &other]);
}

#[test]
fn lazy_adjoint_materialization_and_add_are_exact_on_fz2_u1_legs() {
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
    assert_exact_adjoint!(f64, [&leg, &dual], [&leg]);
    assert_exact_adjoint!(Complex64, [&leg, &other], [&dual]);
    assert_exact_adjoint!(Complex64, [&dual], [&leg, &other]);
}

/// Checked-Generic `add` (`host_add_impl`) and materialization on SU(3):
/// `adj ⊗ adj → adj` has outer multiplicity two, so blocks that differ only
/// by their vertex label must each keep their own entries.
#[cfg(feature = "cgc-gen")]
#[test]
fn lazy_adjoint_materialization_and_add_are_exact_on_su3_multiplicity_legs() {
    use symtn::sector::LieGroupFusionRule;

    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let adjoint = vec![2i64, 2];
    let trivial = vec![0i64, 0];
    let leg =
        GradedSpace::try_new(Arc::clone(&provider), [(adjoint.clone(), 2), (trivial, 1)]).unwrap();
    let other = GradedSpace::try_new(Arc::clone(&provider), [(adjoint, 3)]).unwrap();
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let probe = TensorMap::<_, f64>::from_subblock_fn(&runtime, [&leg, &other], [&leg], |_, _| 0.0)
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
    assert_exact_adjoint!(f64, [&leg, &other], [&leg]);
    assert_exact_adjoint!(Complex64, [&leg, &other], [&other, &leg]);
}
