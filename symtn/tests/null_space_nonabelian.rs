//! The shape-based null spaces are orthonormal and annihilate the input on
//! non-Abelian fusion trees with an inner line and a dual leg (SU(2)) and
//! with outer multiplicity (SU(3)) (#2309).
//!
//! Oracle: the defining identities, evaluated by composition rather than by
//! the factorization under test. A left null space `n` of `a` satisfies
//! `n^H a = 0` and `n^H n = 1`; a right null space satisfies `a n^H = 0` and
//! `n n^H = 1`. The null dimension of every coupled sector is the hand-counted
//! rectangular deficit `(rows_c - cols_c)+` of that sector's block.

use std::sync::Arc;

use num_complex::Complex64;
use symtn::sector::{SU2FusionRule, SU2Irrep};
use symtn::typed::{GradedSpace, TensorMap};

use crate::fixtures;

use fixtures::host_runtime;

const TOL: f64 = 1e-12;

/// `n^H a = 0` and `n^H n = 1` (left), or `a n^H = 0` and `n n^H = 1` (right).
macro_rules! assert_null_space {
    ($runtime:expr, $a:expr, $n:expr, $left:expr, $one:expr) => {{
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
        let residual = residual.norm(2.0).unwrap();
        assert!(
            residual <= TOL * (1.0 + a.norm(2.0).unwrap()),
            "the null space does not annihilate the input: {residual}"
        );
        let identity = TensorMap::isomorphism(&$runtime, &bond, &bond).unwrap();
        let defect = gram
            .axpby($one, &identity, -$one)
            .unwrap()
            .norm(2.0)
            .unwrap();
        assert!(defect <= TOL, "the null basis is not orthonormal: {defect}");
    }};
}

/// Left null of the tall `[legs..] <- [w]` and right null of the wide
/// `[w] <- [legs..]`, with the null bond's degeneracy per coupled sector.
macro_rules! check_both_sides {
    ($runtime:expr, $scalar:ty, [$($leg:expr),+], $w:expr, $one:expr, $expected:expr) => {{
        let w = &$w;
        let legs = [$(&$leg),+];
        let many: Vec<usize> = (0..legs.len()).collect();
        let shifted: Vec<usize> = (1..=legs.len()).collect();
        let tall: TensorMap<_, $scalar> =
            TensorMap::rand_with_seed(&$runtime, legs, [w], 2309).unwrap();
        let left = tall.left_null(&many, &[legs.len()]).unwrap();
        assert_null_space!($runtime, tall, left, true, $one);
        let wide: TensorMap<_, $scalar> =
            TensorMap::rand_with_seed(&$runtime, [w], legs, 2310).unwrap();
        let right = wide.right_null(&[0], &shifted).unwrap();
        assert_null_space!($runtime, wide, right, false, $one);
        for (sector, nullity) in $expected {
            assert_eq!(
                left.domain()[0].degeneracy(&sector).unwrap(),
                nullity,
                "left nullity of {sector:?}"
            );
            assert_eq!(
                right.codomain()[0].degeneracy(&sector).unwrap(),
                nullity,
                "right nullity of {sector:?}"
            );
        }
    }};
}

#[test]
fn su2_null_spaces_annihilate_and_are_orthonormal_on_inner_lines_and_dual_legs() {
    // `v = 0^2 + (1/2)^2 + 1`. Its pair `v (x) v*` has the degeneracies
    //   0:   (0,0) 4 + (1/2,1/2) 4 + (1,1) 1                 = 9
    //   1/2: (0,1/2) 4 + (1/2,0) 4 + (1/2,1) 2 + (1,1/2) 2   = 12
    //   1:   (0,1) 2 + (1,0) 2 + (1/2,1/2) 4 + (1,1) 1       = 9
    //   3/2: (1/2,1) 2 + (1,1/2) 2                           = 4
    //   2:   (1,1) 1                                         = 1
    // which is the inner line `a` of the trees of `v (x) v* (x) v`. Fusing
    // `a` with the third leg (`0 x2, 1/2 x2, 1 x1`) gives the rows
    //   0:   a=0 18 + a=1/2 24 + a=1 9                       = 51
    //   1/2: a=0 18 + a=1/2 24+12 + a=1 18 + a=3/2 4         = 76
    //   1:   a=0 9 + a=1/2 24 + a=1 18+9 + a=3/2 8 + a=2 1   = 69
    //   3/2: a=1/2 12 + a=1 18 + a=3/2 8+4 + a=2 2           = 44
    //   2:   a=1 9 + a=3/2 8 + a=2 2+1                       = 20
    //   5/2: a=3/2 4 + a=2 2                                 = 6
    //   3:   a=2 1                                           = 1
    // (total dimension `sum rows_c (2c+1) = 729 = 9^3`), against
    // `w = 0 + (1/2)^2 + 1^3` columns `1, 2, 3, 0, 0, 0, 0`.
    let runtime = host_runtime();
    let rule = Arc::new(SU2FusionRule);
    let spin = SU2Irrep::from_twice_spin;
    let v = GradedSpace::try_new(
        Arc::clone(&rule),
        [(spin(0), 2), (spin(1), 2), (spin(2), 1)],
    )
    .unwrap();
    let v_dual = v.try_dual().unwrap();
    let w = GradedSpace::try_new(rule, [(spin(0), 1), (spin(1), 2), (spin(2), 3)]).unwrap();
    let expected = [
        (spin(0), 50),
        (spin(1), 74),
        (spin(2), 66),
        (spin(3), 44),
        (spin(4), 20),
        (spin(5), 6),
        (spin(6), 1),
    ];
    check_both_sides!(runtime, f64, [v, v_dual, v], w, 1.0, expected);
    check_both_sides!(
        runtime,
        Complex64,
        [v, v_dual, v],
        w,
        Complex64::new(1.0, 0.0),
        expected
    );
}

#[cfg(feature = "cgc-gen")]
#[test]
fn su3_null_spaces_annihilate_and_are_orthonormal_with_outer_multiplicity() {
    use symtn::sector::LieGroupFusionRule;

    // Dynkin labels: 1 = [0,0], 8 = [1,1], 10 = [3,0], 10bar = [0,3],
    // 27 = [2,2]; `8 (x) 8 = 1 + 8 + 8 + 10 + 10bar + 27`. With
    // `a = 8^2 + 1`, `a (x) a*` has the coupled-sector rows
    //   1:  (8,8) 4 + (1,1) 1                                = 5
    //   8:  (8,8) mu=1 4 + (8,8) mu=2 4 + (8,1) 2 + (1,8) 2  = 12
    //   10, 10bar, 27: (8,8) 4 each
    // against `b = 1^2 + 8^3` columns `2, 3, 0, 0, 0`.
    let runtime = host_runtime();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let a =
        GradedSpace::try_new(Arc::clone(&provider), [(vec![1i64, 1], 2), (vec![0, 0], 1)]).unwrap();
    let a_dual = a.try_dual().unwrap();
    let b = GradedSpace::try_new(provider, [(vec![0i64, 0], 2), (vec![1, 1], 3)]).unwrap();
    let probe = TensorMap::<_, f64>::zeros(&runtime, [&a, &a_dual], [&b]).unwrap();
    assert!(
        (0..probe.subblock_count()).any(|index| probe
            .subblock_fusion_trees(index)
            .unwrap()
            .codomain_vertices()
            .iter()
            .any(|vertex| vertex.get() == 2)),
        "fixture must carry the outer multiplicity vertex mu = 2"
    );
    let expected = [
        (vec![0i64, 0], 3),
        (vec![1, 1], 9),
        (vec![3, 0], 4),
        (vec![0, 3], 4),
        (vec![2, 2], 4),
    ];
    check_both_sides!(runtime, f64, [a, a_dual], b, 1.0, expected.clone());
    check_both_sides!(
        runtime,
        Complex64,
        [a, a_dual],
        b,
        Complex64::new(1.0, 0.0),
        expected
    );
}
