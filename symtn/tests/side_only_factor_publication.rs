//! Full QR/LQ/SVD and null spaces of MPS-like sites whose codomain and domain
//! each carry a coupled sector the other side lacks (#1477). Those sectors are
//! identity-completed (MatrixAlgebraKit `one!`), and full-rank sectors drop
//! out of the null spaces, so the factors must stay exactly unitary /
//! complementary on the complete fused spaces.

use std::sync::Arc;
use symtn::typed::Engine;
#[cfg(feature = "cgc-gen")]
use symtn::typed::HermitianTol;

use num_complex::Complex64;
use symtn::sector::{
    product_sector, FermionParityFusionRule, ProductFusionRule, SU2FusionRule, SU2Irrep,
    U1FusionRule, U1Irrep, Z2Irrep,
};
use symtn::sector::{PackedProductCodec, U1SectorLayout};
use symtn::typed::{GradedSpace, Runtime, TensorMap};
use symtn::typed::{Lq, Qr, Svd};

/// The receiver's own split as leg roles: `rows = 0..nout`.
#[cfg(feature = "cgc-gen")]
fn codomain_axes<R, D, S>(t: &symtn::typed::TensorMap<R, D, S>) -> Vec<usize> {
    (0..t.codomain_rank()).collect()
}

/// The receiver's own split as leg roles: `cols = nout..rank`.
#[cfg(feature = "cgc-gen")]
fn domain_axes<R, D, S>(t: &symtn::typed::TensorMap<R, D, S>) -> Vec<usize> {
    (t.codomain_rank()..t.rank()).collect()
}

type Fz2U1Codec = PackedProductCodec<symtn::sector::Fz2SectorLayout, U1SectorLayout>;
type Fz2U1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule, Fz2U1Codec>;

macro_rules! assert_close {
    ($engine:ident, $actual:expr, $expected:expr) => {{
        let actual = $actual;
        let expected = $expected;
        let error = actual
            .axpby($engine, 1.0.into(), expected, (-1.0).into())
            .unwrap()
            .norm($engine, 2.0)
            .unwrap();
        assert!(
            error <= 1e-10 * (1.0 + expected.norm($engine, 2.0).unwrap()),
            "residual {error}"
        );
    }};
}

/// `values` must not follow a low-order linear recurrence, or the populated
/// sectors lose rank and the null spaces change.
macro_rules! side_only_case {
    ($engine:ident, $rule:expr, $left:expr, $phys:expr, $right:expr, $scalar:ty, $value:expr) => {{
        let rt = Runtime::builder().build().unwrap();
        let provider = Arc::new($rule);
        let left = GradedSpace::try_new(Arc::clone(&provider), $left).unwrap();
        let phys = GradedSpace::try_new(Arc::clone(&provider), $phys).unwrap();
        let right = GradedSpace::try_new(Arc::clone(&provider), $right).unwrap();
        let mut counter = 0usize;
        let value = $value;
        let t: TensorMap<_, $scalar> =
            TensorMap::from_subblock_fn($engine, &rt, [&left, &phys], [&right], |_, _| {
                counter += 1;
                value(counter)
            })
            .unwrap();
        let codomain_id: TensorMap<_, $scalar> =
            TensorMap::isomorphism($engine, &rt, [&left, &phys], [&left, &phys]).unwrap();
        let domain_id: TensorMap<_, $scalar> =
            TensorMap::isomorphism($engine, &rt, [&right], [&right]).unwrap();
        let unitary = |$engine: &mut dyn Engine, u: &TensorMap<_, $scalar>| {
            let adjoint = u.adjoint($engine).unwrap();
            let inner =
                TensorMap::isomorphism($engine, &rt, u.domain().iter(), u.domain().iter()).unwrap();
            let outer =
                TensorMap::isomorphism($engine, &rt, u.codomain().iter(), u.codomain().iter())
                    .unwrap();
            assert_close!($engine, &adjoint.compose($engine, u).unwrap(), &inner);
            assert_close!($engine, &u.compose($engine, &adjoint).unwrap(), &outer);
        };

        let Qr { q, r } = t.qr_full($engine, &[0, 1], &[2]).unwrap();
        assert_close!($engine, &q.compose($engine, &r).unwrap(), &t);
        unitary($engine, &q);
        let Lq { l, q } = t.lq_full($engine, &[0, 1], &[2]).unwrap();
        assert_close!($engine, &l.compose($engine, &q).unwrap(), &t);
        unitary($engine, &q);
        let Svd { u, s, vh } = t.svd_full($engine, &[0, 1], &[2]).unwrap();
        assert_close!(
            $engine,
            &u.compose($engine, &s)
                .unwrap()
                .compose($engine, &vh)
                .unwrap(),
            &t
        );
        unitary($engine, &u);
        unitary($engine, &vh);

        let pseudo = t.pinv($engine, &[0, 1], &[2], 1e-12).unwrap();
        let null = t.left_null($engine, &[0, 1], &[2]).unwrap();
        let null_adjoint = null.adjoint($engine).unwrap();
        assert!(
            null_adjoint
                .compose($engine, &t)
                .unwrap()
                .norm($engine, 2.0)
                .unwrap()
                <= 1e-10
        );
        assert_close!(
            $engine,
            &null_adjoint.compose($engine, &null).unwrap(),
            &TensorMap::isomorphism($engine, &rt, null.domain().iter(), null.domain().iter())
                .unwrap()
        );
        let t_compose = &t.compose($engine, &pseudo).unwrap();
        assert_close!(
            $engine,
            &null
                .compose($engine, &null_adjoint)
                .unwrap()
                .axpby($engine, 1.0.into(), t_compose, 1.0.into())
                .unwrap(),
            &codomain_id
        );
        let null = t.right_null($engine, &[0, 1], &[2]).unwrap();
        let null_adjoint = null.adjoint($engine).unwrap();
        assert!(
            t.compose($engine, &null_adjoint)
                .unwrap()
                .norm($engine, 2.0)
                .unwrap()
                <= 1e-10
        );
        assert_close!(
            $engine,
            &null.compose($engine, &null_adjoint).unwrap(),
            &TensorMap::isomorphism($engine, &rt, null.codomain().iter(), null.codomain().iter())
                .unwrap()
        );
        let pseudo_compose = &pseudo.compose($engine, &t).unwrap();
        assert_close!(
            $engine,
            &null_adjoint
                .compose($engine, &null)
                .unwrap()
                .axpby($engine, 1.0.into(), pseudo_compose, 1.0.into())
                .unwrap(),
            &domain_id
        );
    }};
}

fn real(k: usize) -> f64 {
    ((k * k) as f64 * 0.37 + k as f64).sin()
}

fn complex(k: usize) -> Complex64 {
    Complex64::new(real(k), ((k * k) as f64 * 0.23 - k as f64).cos())
}

macro_rules! all_symmetries {
    ($engine:ident, $scalar:ty, $value:expr) => {{
        // U(1): codomain-only charge 2, domain-only charge 3.
        let q = U1Irrep::new;
        side_only_case!(
            $engine,
            U1FusionRule,
            [(q(0), 2), (q(1), 2)],
            [(q(0), 1), (q(1), 1)],
            [(q(0), 2), (q(1), 3), (q(3), 2)],
            $scalar,
            $value
        );
        // SU(2): codomain-only spin 1, domain-only spin 2.
        let j = SU2Irrep::from_twice_spin;
        side_only_case!(
            $engine,
            SU2FusionRule,
            [(j(1), 2)],
            [(j(1), 1)],
            [(j(0), 1), (j(4), 2)],
            $scalar,
            $value
        );
        // fZ2 x U(1): codomain-only (even, 2), domain-only (odd, 3).
        let f = |odd: bool, charge| {
            product_sector(
                if odd { Z2Irrep::ODD } else { Z2Irrep::EVEN },
                U1Irrep::new(charge),
            )
        };
        side_only_case!(
            $engine,
            Fz2U1Rule::try_new(FermionParityFusionRule, U1FusionRule).unwrap(),
            [(f(false, 0), 2), (f(true, 1), 2)],
            [(f(false, 0), 1), (f(true, 1), 1)],
            [(f(false, 0), 2), (f(true, 1), 3), (f(true, 3), 2)],
            $scalar,
            $value
        );
    }};
}

#[test]
fn side_only_full_and_null_factors_are_complete_real() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    all_symmetries!(engine, f64, real);
}

#[test]
fn side_only_full_and_null_factors_are_complete_complex() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    all_symmetries!(engine, Complex64, complex);
}

/// TensorKit `initialize_output(qr_full!/lq_full!)`: Q of full QR maps
/// `fuse(codomain)` into the codomain, Q of full LQ maps the domain into
/// `fuse(domain)`, so every side-only sector is identity-completed (#1524).
/// `$qr_bond`/`$lq_bond` are those fused spaces built by `GradedSpace::fuse`,
/// independently of the factorization.
#[cfg(feature = "cgc-gen")]
macro_rules! full_qr_lq_bond_case {
    ($engine:ident, $rt:expr, $scalar:ty, $codomain:expr, $domain:expr, $qr_bond:expr, $lq_bond:expr, $seed:expr) => {{
        let rt = $rt;
        let owned_adjoint = |$engine: &mut dyn Engine, u: &TensorMap<_, $scalar>| {
            let adjoint = u.adjoint($engine).unwrap();
            adjoint
                .axpby($engine, 1.0.into(), &adjoint, 0.0.into())
                .unwrap()
        };
        // Checked Generic has no `id`: a Hermitian map is the identity iff
        // every eigenvalue is one, and a count equal to the expected reduced
        // dimension rules out an unstored (zero) side-only block.
        let identity = |$engine: &mut dyn Engine, gram: TensorMap<_, $scalar>, reduced: usize| {
            let values = gram
                .eigh_vals(
                    $engine,
                    &codomain_axes(&gram),
                    &domain_axes(&gram),
                    HermitianTol::DEFAULT,
                )
                .unwrap()
                .into_iter()
                .flat_map(|spectrum| spectrum.values)
                .collect::<Vec<_>>();
            assert_eq!(values.len(), reduced);
            assert!(
                values.iter().all(|value| (value - 1.0).abs() <= 1e-10),
                "{values:?}"
            );
        };
        let reduced = |space: &GradedSpace<_>| space.degeneracies().iter().sum::<usize>();
        let unitary =
            |$engine: &mut dyn Engine, u: &TensorMap<_, $scalar>, bond: &GradedSpace<_>| {
                let adjoint = owned_adjoint($engine, u);
                let adjoint_compose = adjoint.compose($engine, u).unwrap();
                identity($engine, adjoint_compose, reduced(bond));
                let u_compose = u.compose($engine, &adjoint).unwrap();
                identity($engine, u_compose, reduced(bond));
            };
        let t: TensorMap<_, $scalar> =
            TensorMap::rand_with_seed($engine, rt, $codomain, $domain, $seed).unwrap();
        let Qr { q, r } = t
            .qr_full($engine, &codomain_axes(&t), &domain_axes(&t))
            .unwrap();
        assert_eq!(q.domain(), std::slice::from_ref($qr_bond));
        assert_eq!(r.codomain(), q.domain());
        assert_close!($engine, &q.compose($engine, &r).unwrap(), &t);
        unitary($engine, &q, $qr_bond);
        let Lq { l, q } = t
            .lq_full($engine, &codomain_axes(&t), &domain_axes(&t))
            .unwrap();
        assert_eq!(q.codomain(), std::slice::from_ref($lq_bond));
        assert_eq!(l.domain(), q.codomain());
        assert_close!($engine, &l.compose($engine, &q).unwrap(), &t);
        unitary($engine, &q, $lq_bond);
        // Null spaces already use the side-aware builder: a side-only sector
        // is wholly null, so rank + null dimension is the fused dimension.
        let rank = t
            .svd_vals($engine, &codomain_axes(&t), &domain_axes(&t))
            .unwrap()
            .into_iter()
            .flat_map(|spectrum| spectrum.values)
            .filter(|&value| value > 1e-10)
            .count();
        let n = t
            .left_null($engine, &codomain_axes(&t), &domain_axes(&t))
            .unwrap();
        let n_adjoint = owned_adjoint($engine, &n);
        assert!(
            n_adjoint
                .compose($engine, &t)
                .unwrap()
                .norm($engine, 2.0)
                .unwrap()
                <= 1e-10
        );
        let n_adjoint_compose = n_adjoint.compose($engine, &n).unwrap();
        identity($engine, n_adjoint_compose, reduced($qr_bond) - rank);
        let n = t
            .right_null($engine, &codomain_axes(&t), &domain_axes(&t))
            .unwrap();
        let n_adjoint = owned_adjoint($engine, &n);
        assert!(
            t.compose($engine, &n_adjoint)
                .unwrap()
                .norm($engine, 2.0)
                .unwrap()
                <= 1e-10
        );
        let n_compose = n.compose($engine, &n_adjoint).unwrap();
        identity($engine, n_compose, reduced($lq_bond) - rank);
    }};
}

/// The multiplicity-free comparison fixture: U(1) with codomain-only charge
/// 2 and domain-only charge 3 obeys the same output-space rule.
#[cfg(feature = "cgc-gen")]
macro_rules! multiplicity_free_full_qr_lq_bonds {
    ($engine:ident, $d:ty) => {{
        let rt = Runtime::builder().build().unwrap();
        let provider = Arc::new(U1FusionRule);
        let q = U1Irrep::new;
        let space = |irreps: Vec<(U1Irrep, usize)>| {
            GradedSpace::try_new(Arc::clone(&provider), irreps).unwrap()
        };
        let left = space(vec![(q(0), 2), (q(1), 2)]);
        let phys = space(vec![(q(0), 1), (q(1), 1)]);
        let right = space(vec![(q(0), 2), (q(1), 3), (q(3), 2)]);
        let fused = left.fuse(&phys).unwrap();
        full_qr_lq_bond_case!(
            $engine,
            &rt,
            $d,
            [&left, &phys],
            [&right],
            &fused,
            &right,
            1524
        );
        full_qr_lq_bond_case!(
            $engine,
            &rt,
            $d,
            [&right],
            [&left, &phys],
            &right,
            &fused,
            1525
        );
    }};
}

#[cfg(feature = "cgc-gen")]
mod checked_generic {
    use super::*;
    use symtn::sector::LieGroupFusionRule;

    /// SU(3), `a = 8^2 + 1`, `b = 8^2 + 1 + 6^2`: `a ⊗ a` holds 8 with
    /// multiplicity two and the codomain-only 10, 10̄, 27; `b` holds the
    /// domain-only 6. Both orientations put side-only sectors on each side.
    macro_rules! checked_full_qr_lq_bonds {
        ($engine:ident, $d:ty) => {{
            let rt = Runtime::builder().build().unwrap();
            let provider = Arc::new(
                LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap(),
            );
            let space = |irreps: Vec<(Vec<i64>, usize)>| {
                GradedSpace::try_new(Arc::clone(&provider), irreps).unwrap()
            };
            let a = space(vec![(vec![1, 1], 2), (vec![0, 0], 1)]);
            let b = space(vec![(vec![1, 1], 2), (vec![0, 0], 1), (vec![2, 0], 2)]);
            let fused = a.fuse(&a).unwrap();
            full_qr_lq_bond_case!($engine, &rt, $d, [&a, &a], [&b], &fused, &b, 1526);
            full_qr_lq_bond_case!($engine, &rt, $d, [&b], [&a, &a], &b, &fused, 1527);
        }};
    }

    /// Column-major multi-index of `linear` within `shape`.
    fn unravel(mut linear: usize, shape: &[usize]) -> Vec<usize> {
        shape
            .iter()
            .map(|&extent| {
                let index = linear % extent;
                linear /= extent;
                index
            })
            .collect()
    }

    /// Asserts that a Gram map is the literal identity block by block:
    /// entry = δ(codomain tree, domain tree) δ(row, column), where a tree is
    /// its uncoupled sectors, inner lines and vertex labels. Returns the
    /// number of unit entries, i.e. the reduced dimension it is the identity on.
    macro_rules! assert_literal_identity {
        ($gram:expr) => {{
            let gram = $gram;
            let nc = gram.codomain().len();
            let mut ones = 0usize;
            for (trees, view) in gram.subblocks().unwrap() {
                let same = trees.codomain_uncoupled() == trees.domain_uncoupled()
                    && trees.codomain_innerlines() == trees.domain_innerlines()
                    && trees.codomain_vertices() == trees.domain_vertices();
                let (row_shape, col_shape) = view.shape().split_at(nc);
                let rows = row_shape.iter().product::<usize>();
                let cols = col_shape.iter().product::<usize>();
                for col in 0..cols {
                    for row in 0..rows {
                        let mut index = unravel(row, row_shape);
                        index.extend(unravel(col, col_shape));
                        let value: Complex64 = (*view.get(&index).unwrap()).into();
                        let unit = same && row == col;
                        ones += usize::from(unit);
                        let expected = if unit { 1.0 } else { 0.0 };
                        assert!(
                            (value - expected).norm() <= 1e-10,
                            "{trees:?} [{row}, {col}] = {value}"
                        );
                    }
                }
            }
            ones
        }};
    }

    /// Every side-only block of a full Q is exactly the identity slice in
    /// output tree order: tree-side index `i` of the `n`-th tree of sector c
    /// maps to bond index `offset_n + i` (column-major `i`), and the offsets
    /// tile the fused degeneracy of c. The opposite factor stores no block in
    /// a side-only sector. Returns the largest tree count of one side-only
    /// sector.
    macro_rules! assert_side_only_identity_slices {
        ($scalar:ty, $t:expr, $q:expr, $other:expr, $bond:expr, $bond_first:expr) => {{
            let (t, q, bond) = ($t, $q, $bond);
            let one: $scalar = 1.0.into();
            let zero: $scalar = 0.0.into();
            let populated = t
                .subblocks()
                .unwrap()
                .map(|(trees, _)| trees.coupled().clone())
                .collect::<Vec<_>>();
            let mut offsets = Vec::new();
            for (trees, view) in q.subblocks().unwrap() {
                let sector = trees.coupled().clone();
                if populated.contains(&sector) {
                    continue;
                }
                let shape = view.shape();
                let (bond_axis, tree_shape) = if $bond_first {
                    (0, &shape[1..])
                } else {
                    (shape.len() - 1, &shape[..shape.len() - 1])
                };
                let extent = tree_shape.iter().product::<usize>();
                let position = match offsets.iter().position(|(s, _, _)| *s == sector) {
                    Some(position) => position,
                    None => {
                        offsets.push((sector.clone(), 0usize, 0usize));
                        offsets.len() - 1
                    }
                };
                let offset = offsets[position].1;
                for k in 0..shape[bond_axis] {
                    for i in 0..extent {
                        let mut index = unravel(i, tree_shape);
                        index.insert(bond_axis, k);
                        let value = *view.get(&index).unwrap();
                        let expected = if k == offset + i { one } else { zero };
                        assert!(value == expected, "{trees:?} bond {k} tree index {i}");
                    }
                }
                offsets[position].1 += extent;
                offsets[position].2 += 1;
            }
            assert!(!offsets.is_empty());
            for (sector, offset, _) in &offsets {
                assert_eq!(*offset, bond.degeneracy(sector).unwrap());
            }
            for (trees, _) in $other.subblocks().unwrap() {
                assert!(populated.contains(trees.coupled()), "{trees:?}");
            }
            offsets.iter().map(|&(_, _, n)| n).max().unwrap()
        }};
    }

    macro_rules! literal_side_only_case {
        ($engine:ident, $rt:expr, $scalar:ty, $codomain:expr, $domain:expr, $qr_bond:expr, $lq_bond:expr, $seed:expr) => {{
            let rt = $rt;
            let owned_adjoint = |$engine: &mut dyn Engine, u: &TensorMap<_, $scalar>| {
                let adjoint = u.adjoint($engine).unwrap();
                adjoint
                    .axpby($engine, 1.0.into(), &adjoint, 0.0.into())
                    .unwrap()
            };
            let reduced = |space: &GradedSpace<_>| space.degeneracies().iter().sum::<usize>();
            let t: TensorMap<_, $scalar> =
                TensorMap::rand_with_seed($engine, rt, $codomain, $domain, $seed).unwrap();
            let Qr { q, r } = t
                .qr_full($engine, &codomain_axes(&t), &domain_axes(&t))
                .unwrap();
            let qh = owned_adjoint($engine, &q);
            assert_eq!(
                assert_literal_identity!(qh.compose($engine, &q).unwrap()),
                reduced($qr_bond)
            );
            assert_eq!(
                assert_literal_identity!(q.compose($engine, &qh).unwrap()),
                reduced($qr_bond)
            );
            let qr_trees = assert_side_only_identity_slices!($scalar, &t, &q, &r, $qr_bond, false);
            let Lq { l, q } = t
                .lq_full($engine, &codomain_axes(&t), &domain_axes(&t))
                .unwrap();
            let qh = owned_adjoint($engine, &q);
            assert_eq!(
                assert_literal_identity!(qh.compose($engine, &q).unwrap()),
                reduced($lq_bond)
            );
            assert_eq!(
                assert_literal_identity!(q.compose($engine, &qh).unwrap()),
                reduced($lq_bond)
            );
            let lq_trees = assert_side_only_identity_slices!($scalar, &t, &q, &l, $lq_bond, true);
            (qr_trees, lq_trees)
        }};
    }

    /// Side-only sectors with several fusion trees, including both outer
    /// multiplicity vertices of 8 ⊗ 8 → 8: `a = 8² ⊕ 1 ⊕ 3`, so 8 in `a ⊗ a`
    /// has the four trees (8,8)μ=1, (8,8)μ=2, (8,1), (1,8). `b = 1² ⊕ 15²`
    /// makes 8 codomain-only and 15 = (4,0) domain-only; the mixed
    /// `b = 1² ⊕ 8³ ⊕ 15²` keeps 8 populated.
    macro_rules! checked_literal_side_only {
        ($engine:ident, $d:ty) => {{
            let rt = Runtime::builder().build().unwrap();
            let provider = Arc::new(
                LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap(),
            );
            let space = |irreps: Vec<(Vec<i64>, usize)>| {
                GradedSpace::try_new(Arc::clone(&provider), irreps).unwrap()
            };
            let a = space(vec![(vec![1, 1], 2), (vec![0, 0], 1), (vec![1, 0], 1)]);
            let fused = a.fuse(&a).unwrap();
            let pure = space(vec![(vec![0, 0], 2), (vec![4, 0], 2)]);
            let mixed = space(vec![(vec![0, 0], 2), (vec![1, 1], 3), (vec![4, 0], 2)]);
            for (seed, b) in [(1528, &pure), (1530, &mixed)] {
                let (qr_trees, _) =
                    literal_side_only_case!($engine, &rt, $d, [&a, &a], [b], &fused, b, seed);
                assert!(qr_trees >= 4, "{qr_trees}");
                let (_, lq_trees) =
                    literal_side_only_case!($engine, &rt, $d, [b], [&a, &a], b, &fused, seed + 1);
                assert!(lq_trees >= 4, "{lq_trees}");
            }
        }};
    }

    #[test]
    fn checked_generic_side_only_identity_is_literal_in_tree_order() {
        let engine = &mut symtn::typed::CpuEngine::serial();
        checked_literal_side_only!(engine, f64);
        checked_literal_side_only!(engine, Complex64);
    }

    #[test]
    fn checked_generic_full_qr_lq_bonds_are_fused_sides() {
        let engine = &mut symtn::typed::CpuEngine::serial();
        multiplicity_free_full_qr_lq_bonds!(engine, f64);
        checked_full_qr_lq_bonds!(engine, f64);
        multiplicity_free_full_qr_lq_bonds!(engine, Complex64);
        checked_full_qr_lq_bonds!(engine, Complex64);
    }
}
