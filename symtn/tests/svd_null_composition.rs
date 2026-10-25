//! The rank-revealing null space as a composition (#1984).
//!
//! TensorKit's `left_null(t; alg = :svd, trunc)` / `right_null` is, in symtn,
//! `svd_full` → `diagview` of `s` → `extend_spectrum` on the factor's bond →
//! `find_truncated` with a keep-below rule → `restrict_leg`. The oracle is
//! independent of the SVD: every fixture block is built with a known rank, so
//! its numerical nullity is the rectangular deficit plus the hand-counted rank
//! deficit, and the result is checked by dense relations — it annihilates the
//! input and has orthonormal columns (rows).

use std::sync::Arc;

use num_complex::{Complex32, Complex64};
use symtn::sector::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use symtn::typed::{GradedSpace, Svd, TensorMap, Truncation};

use crate::fixtures;

use fixtures::host_runtime;

/// MatrixAlgebraKit's default null `rtol` (`defaulttol`), at the payload's
/// real precision.
fn default_rtol(epsilon: f64) -> f64 {
    epsilon.powf(2.0 / 3.0)
}

/// Left null space by composition: `u`'s columns kept by `truncation` over
/// `s`'s diagonal padded to `u`'s bond.
macro_rules! composed_left_null {
    ($a:expr, $truncation:expr) => {{
        let Svd { u, s, .. } = $a.svd_full(&[0], &[1]).unwrap();
        let bond = u.domain()[0].clone();
        let spectrum = bond.extend_spectrum(&s.diagview().unwrap()).unwrap();
        let found = bond.find_truncated(&spectrum, &$truncation).unwrap();
        u.restrict_leg(&[(1, &found.selection)]).unwrap()
    }};
}

/// Right null space by composition, on `vh`'s bond.
macro_rules! composed_right_null {
    ($a:expr, $truncation:expr) => {{
        let Svd { s, vh, .. } = $a.svd_full(&[0], &[1]).unwrap();
        let bond = vh.codomain()[0].clone();
        let spectrum = bond.extend_spectrum(&s.diagview().unwrap()).unwrap();
        let found = bond.find_truncated(&spectrum, &$truncation).unwrap();
        vh.restrict_leg(&[(0, &found.selection)]).unwrap()
    }};
}

/// `n^H a = 0` and `n^H n = 1` (left), or `a n^H = 0` and `n n^H = 1` (right).
macro_rules! assert_null_space {
    ($runtime:expr, $a:expr, $n:expr, $left:expr, $tol:expr, $one:expr) => {{
        let (a, n) = (&$a, &$n);
        let n_adjoint = n.adjoint().unwrap().materialize().unwrap();
        let (residual, gram, bond) = if $left {
            (
                n_adjoint.compose(a).unwrap(),
                n_adjoint.compose(n).unwrap(),
                n.domain(),
            )
        } else {
            (
                a.compose(&n_adjoint).unwrap(),
                n.compose(&n_adjoint).unwrap(),
                n.codomain(),
            )
        };
        assert!(
            residual.norm(2.0).unwrap() <= $tol * (1.0 + a.norm(2.0).unwrap()),
            "the null space does not annihilate the input"
        );
        let identity = TensorMap::isomorphism(&$runtime, &bond, &bond).unwrap();
        let defect = gram
            .axpby($one, &identity, -$one)
            .unwrap()
            .norm(2.0)
            .unwrap();
        assert!(
            defect <= $tol,
            "the null basis is not orthonormal: {defect}"
        );
    }};
}

/// U(1): charge 0 is a `4 x 2` rank-one block (nullity `2 + 1`), charge 1 a
/// `3 x 3` block of rank two (nullity `0 + 1`), and charge 2 lives only in the
/// codomain (nullity 2).
macro_rules! u1_rank_deficient {
    ($runtime:expr, $scalar:ty, $value:expr) => {{
        let rule = Arc::new(U1FusionRule);
        let codomain = GradedSpace::try_new(
            Arc::clone(&rule),
            [
                (U1Irrep::new(0), 4),
                (U1Irrep::new(1), 3),
                (U1Irrep::new(2), 2),
            ],
        )
        .unwrap();
        let domain =
            GradedSpace::try_new(rule, [(U1Irrep::new(0), 2), (U1Irrep::new(1), 3)]).unwrap();
        let value = $value;
        let a: TensorMap<_, $scalar> =
            TensorMap::from_subblock_fn(&$runtime, [&codomain], [&domain], |trees, i| {
                if *trees.coupled() == U1Irrep::new(0) {
                    // Outer product: rank one.
                    value((i[0] + 1) as f64, 0.5 * i[0] as f64)
                        * value(1.0 + i[1] as f64, -(i[1] as f64))
                } else if i[0] == i[1] {
                    value([1.0, 2.0, 0.0][i[0]], 0.25) * value(f64::from(i[0] != 2), 0.0)
                } else {
                    value(0.0, 0.0)
                }
            })
            .unwrap();
        a
    }};
}

macro_rules! check_u1 {
    ($scalar:ty, $value:expr, $epsilon:expr, $tol:expr) => {{
        let runtime = host_runtime();
        let value = $value;
        let a = u1_rank_deficient!(runtime, $scalar, value);
        let below = Truncation::below(0.0, default_rtol($epsilon)).unwrap();
        let expected = [(0, 3), (1, 1), (2, 2)];

        let left = composed_left_null!(a, below);
        assert_null_space!(runtime, a, left, true, $tol, value(1.0, 0.0));
        for (charge, nullity) in expected {
            assert_eq!(
                left.domain()[0].degeneracy(&U1Irrep::new(charge)).unwrap(),
                nullity,
                "left nullity of charge {charge}"
            );
        }
        // The shape-based default keeps only the rectangular deficit.
        let shape = a.left_null(&[0], &[1]).unwrap();
        for (charge, nullity) in [(0, 2), (1, 0), (2, 2)] {
            assert_eq!(
                shape.domain()[0].degeneracy(&U1Irrep::new(charge)).unwrap(),
                nullity
            );
        }

        let wide = a.adjoint().unwrap().materialize().unwrap();
        let right = composed_right_null!(wide, below);
        assert_null_space!(runtime, wide, right, false, $tol, value(1.0, 0.0));
        for (charge, nullity) in expected {
            assert_eq!(
                right.codomain()[0]
                    .degeneracy(&U1Irrep::new(charge))
                    .unwrap(),
                nullity,
                "right nullity of charge {charge}"
            );
        }

        // `maxnullity` (`rank_smallest`) caps the weighted count, taking the
        // smallest values first; every kept direction is still null.
        let capped = below.clone().and(Truncation::rank_smallest(4));
        let left = composed_left_null!(a, capped);
        assert_null_space!(runtime, a, left, true, $tol, value(1.0, 0.0));
        assert_eq!(left.domain()[0].degeneracies().iter().sum::<usize>(), 4);

        // An `atol` above every singular value keeps the whole bond.
        let everything = Truncation::below(1.0e6, 0.0).unwrap();
        let left = composed_left_null!(a, everything);
        assert_eq!(
            left.domain()[0].degeneracies(),
            a.codomain()[0].degeneracies()
        );
    }};
}

#[test]
fn u1_composed_null_space_counts_rectangular_and_rank_deficits_for_every_dtype() {
    check_u1!(f64, |re: f64, _im: f64| re, f64::EPSILON, 1e-12);
    check_u1!(
        f32,
        |re: f64, _im: f64| re as f32,
        f64::from(f32::EPSILON),
        1e-5
    );
    check_u1!(
        Complex64,
        |re: f64, im: f64| Complex64::new(re, im),
        f64::EPSILON,
        1e-12
    );
    check_u1!(
        Complex32,
        |re: f64, im: f64| Complex32::new(re as f32, im as f32),
        f64::from(f32::EPSILON),
        1e-5
    );
}

#[test]
fn su2_composed_null_space_counts_reduced_rank_deficits() {
    // What: spin 0 is a `3 x 1` block of rank one (nullity 2), spin 1/2 a
    // `2 x 2` block of rank one (nullity 1), so the shape-based default
    // keeps `[2, 0]` and the composition `[2, 1]` reduced directions.
    let runtime = host_runtime();
    let rule = Arc::new(SU2FusionRule);
    let spin0 = SU2Irrep::from_twice_spin(0);
    let half = SU2Irrep::from_twice_spin(1);
    let codomain = GradedSpace::try_new(Arc::clone(&rule), [(spin0, 3), (half, 2)]).unwrap();
    let domain = GradedSpace::try_new(rule, [(spin0, 1), (half, 2)]).unwrap();
    let a: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&codomain], [&domain], |_, i| {
            ((i[0] + 1) * (i[1] + 2)) as f64
        })
        .unwrap();
    let below = Truncation::below(0.0, default_rtol(f64::EPSILON)).unwrap();
    let left = composed_left_null!(a, below);
    assert_null_space!(runtime, a, left, true, 1e-12, 1.0);
    assert_eq!(left.domain()[0].degeneracy(&spin0).unwrap(), 2);
    assert_eq!(left.domain()[0].degeneracy(&half).unwrap(), 1);
    let shape = a.left_null(&[0], &[1]).unwrap();
    assert_eq!(shape.domain()[0].degeneracy(&spin0).unwrap(), 2);
    assert_eq!(shape.domain()[0].degeneracy(&half).unwrap(), 0);

    // `maxnullity` weighs by quantum dimension: a budget of 3 takes the
    // two spin-0 zeros (weight 1 each) and cannot afford a spin-1/2 state
    // (weight 2).
    let capped = below.and(Truncation::rank_smallest(3));
    let left = composed_left_null!(a, capped);
    assert_eq!(left.domain()[0].degeneracy(&spin0).unwrap(), 2);
    assert_eq!(left.domain()[0].degeneracy(&half).unwrap(), 0);
}

#[test]
fn extend_spectrum_pads_with_rectangular_zeros_and_rejects_malformed_input() {
    use symtn::typed::{Error, SectorSpectrum};
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(U1Irrep::new(0), 3), (U1Irrep::new(1), 2)],
    )
    .unwrap();
    let entry = |charge, values: &[f64]| SectorSpectrum {
        sector: U1Irrep::new(charge),
        values: values.to_vec(),
    };
    let padded = leg.extend_spectrum(&[entry(0, &[2.0, -1.0])]).unwrap();
    assert_eq!(
        padded,
        vec![entry(0, &[2.0, 1.0, 0.0]), entry(1, &[0.0, 0.0])]
    );

    for (bad, message) in [
        (
            vec![entry(0, &[1.0, 1.0, 1.0, 1.0])],
            "more than the leg degeneracy",
        ),
        (
            vec![entry(0, &[1.0]), entry(0, &[1.0])],
            "more than one spectrum",
        ),
        (vec![entry(5, &[1.0])], "absent from the extended leg"),
    ] {
        assert!(matches!(
            leg.extend_spectrum(&bad),
            Err(Error::InvalidArgument(text)) if text.contains(message)
        ));
    }
}

/// The coupled-sector blocks of a rank-(1,1) U(1) map, which for an abelian
/// bosonic rule is its dense expansion: `(sector, rows, cols, column-major
/// entries)`, read through the public block geometry.
fn u1_blocks<D>(t: &TensorMap<U1FusionRule, D>) -> Vec<(U1Irrep, usize, usize, Vec<Complex64>)>
where
    D: symtn::typed::TensorScalar + Into<Complex64>,
{
    let data = t.dense_data().unwrap();
    (0..t.subblock_count())
        .map(|index| {
            let block = t.subblock(index).unwrap();
            let (rows, cols) = (block.shape()[0], block.shape()[1]);
            let (row_stride, col_stride) = (block.strides()[0], block.strides()[1]);
            let entries = (0..cols)
                .flat_map(|col| (0..rows).map(move |row| (row, col)))
                .map(|(row, col)| data[block.offset() + row * row_stride + col * col_stride].into())
                .collect();
            let sector = *t.subblock_fusion_trees(index).unwrap().coupled();
            (sector, rows, cols, entries)
        })
        .collect()
}

/// Hand dense check of a left (`left = true`) or right null space `n` of `a`:
/// per coupled sector `N^H A = 0` (`A N^H = 0`) and orthonormal columns
/// (rows), with the null dimension `(m_c - n_c)+` (`(n_c - m_c)+`) and the
/// whole degeneracy of a sector only on the null side.
fn assert_dense_null<D>(a: &TensorMap<U1FusionRule, D>, n: &TensorMap<U1FusionRule, D>, left: bool)
where
    D: symtn::typed::TensorScalar + Into<Complex64>,
{
    let a_blocks = u1_blocks(a);
    let null_side = if left { a.codomain() } else { a.domain() };
    let leg = &null_side[0];
    for sector in leg.sectors().unwrap() {
        let extent = leg.degeneracy(&sector).unwrap();
        let other = a_blocks
            .iter()
            .find(|(s, ..)| *s == sector)
            .map_or(0, |&(_, rows, cols, _)| if left { cols } else { rows });
        let bond = if left {
            &n.domain()[0]
        } else {
            &n.codomain()[0]
        };
        assert_eq!(
            bond.degeneracy(&sector).unwrap(),
            extent.saturating_sub(other),
            "null dimension of {sector:?}"
        );
    }
    for (sector, n_rows, n_cols, n_data) in u1_blocks(n) {
        // As a matrix on the null side: column `k` is a null vector.
        let (extent, count) = if left {
            (n_rows, n_cols)
        } else {
            (n_cols, n_rows)
        };
        let vector = |k: usize, i: usize| {
            if left {
                n_data[i + n_rows * k]
            } else {
                n_data[k + n_rows * i].conj()
            }
        };
        for k in 0..count {
            for l in 0..count {
                let dot: Complex64 = (0..extent)
                    .map(|i| vector(k, i).conj() * vector(l, i))
                    .sum();
                let expected = if k == l { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).norm() < 1e-12,
                    "orthonormality in {sector:?}"
                );
            }
        }
        if let Some((_, a_rows, a_cols, a_data)) = a_blocks.iter().find(|(s, ..)| *s == sector) {
            let others = if left { *a_cols } else { *a_rows };
            for k in 0..count {
                for j in 0..others {
                    let dot: Complex64 = (0..extent)
                        .map(|i| {
                            let entry = if left {
                                a_data[i + a_rows * j]
                            } else {
                                a_data[j + a_rows * i]
                            };
                            // `vector` is already conjugated on the right.
                            if left {
                                vector(k, i).conj() * entry
                            } else {
                                entry * vector(k, i)
                            }
                        })
                        .sum();
                    assert!(dot.norm() < 1e-12, "annihilation in {sector:?}: {dot}");
                }
            }
        }
    }
}

#[test]
fn shape_based_null_spaces_on_dual_legs_annihilate_the_dense_blocks() {
    // What: the default null spaces of maps with a dual codomain leg and a
    // dual domain leg, real and complex, checked block by block against the
    // input's dense coupled-sector matrices: `(rows - cols)+` orthonormal
    // directions annihilating each block, and a whole side-only sector.
    let runtime = host_runtime();
    let rule = Arc::new(U1FusionRule);
    let w = GradedSpace::try_new(
        Arc::clone(&rule),
        [
            (U1Irrep::new(0), 3),
            (U1Irrep::new(1), 3),
            (U1Irrep::new(2), 1),
        ],
    )
    .unwrap();
    let v = GradedSpace::try_new(rule, [(U1Irrep::new(0), 2), (U1Irrep::new(-1), 1)]).unwrap();
    let w_dual = w.try_dual().unwrap();
    let v_dual = v.try_dual().unwrap();
    // Dual codomain: `w* <- v` couples 0 (3 x 2) and -1 (3 x 1); -2 is
    // codomain-only. Dual domain: `w <- v*` couples 0 (3 x 2) and 1 (3 x 1);
    // 2 is codomain-only.
    for (codomain, domain, seed) in [(&w_dual, &v, 11), (&w, &v_dual, 12)] {
        let real: TensorMap<_, f64> =
            TensorMap::rand_with_seed(&runtime, [codomain], [domain], seed).unwrap();
        let complex: TensorMap<_, Complex64> =
            TensorMap::rand_with_seed(&runtime, [codomain], [domain], seed).unwrap();
        assert_dense_null(&real, &real.left_null(&[0], &[1]).unwrap(), true);
        assert_dense_null(&real, &real.right_null(&[0], &[1]).unwrap(), false);
        assert_dense_null(&complex, &complex.left_null(&[0], &[1]).unwrap(), true);
        assert_dense_null(&complex, &complex.right_null(&[0], &[1]).unwrap(), false);
        // The wide transpose of the same legs: the right null space is the
        // nontrivial one.
        let wide: TensorMap<_, Complex64> =
            TensorMap::rand_with_seed(&runtime, [domain], [codomain], seed).unwrap();
        assert_dense_null(&wide, &wide.right_null(&[0], &[1]).unwrap(), false);
        assert_dense_null(&wide, &wide.left_null(&[0], &[1]).unwrap(), true);
    }
}
