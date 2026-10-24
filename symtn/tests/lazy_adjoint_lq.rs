//! LQ of a lazy adjoint equals LQ of its materialized adjoint (#2070): the
//! same factor spaces and the same values in the positive-diagonal gauge,
//! for compact and full LQ, real and complex payloads, non-square coupled
//! blocks with degeneracy above one, dual legs, and Abelian, fermionic and
//! non-Abelian sectors in both fusion modes.
//!
//! Oracle: `lazy.materialize()` is the exact block copy
//! `block(t^H, c) = block(t, c)^H`, factored by the ordinary owned path.
//! Identity leg roles keep the receiver lazy, so it reaches LQ as an adjoint.

use std::sync::Arc;

use num_complex::Complex64;
use symtn::sector::{
    FermionParityFusionRule, ProductFusionRule, ProductSector, SU2FusionRule, SU2Irrep,
    U1FusionRule, U1Irrep, Z2Irrep,
};
use symtn::typed::{GradedSpace, Lq, Runtime, TensorMap};

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

/// Same spaces, same values in the same gauge. Both routes hand the same
/// block to the same dense QR and gauge; the tolerance only admits a dense
/// kernel whose rounding depends on buffer alignment; a different gauge
/// (a phase on any bond column) would differ by O(1) and fail.
macro_rules! assert_factor_close {
    ($what:expr, $lazy:expr, $eager:expr) => {{
        let (what, lazy, eager) = ($what, &$lazy, &$eager);
        assert_eq!(lazy.codomain(), eager.codomain(), "{what}: codomain");
        assert_eq!(lazy.domain(), eager.domain(), "{what}: domain");
        let (lazy, eager) = (lazy.materialize().unwrap(), eager.materialize().unwrap());
        let (lazy, eager) = (lazy.dense_data().unwrap(), eager.dense_data().unwrap());
        assert_eq!(lazy.len(), eager.len(), "{what}: payload length");
        for (&a, &b) in lazy.iter().zip(eager) {
            assert!(Distance::distance(a, b) < 1.0e-12, "{what}: values differ");
        }
    }};
}

/// Compact and full LQ of the lazy adjoint of `parent` against those of the
/// materialized adjoint.
macro_rules! assert_lazy_lq {
    ($label:expr, $parent:expr) => {{
        let label = $label;
        let lazy = $parent.adjoint().unwrap();
        let eager = lazy.materialize().unwrap();
        let rows: Vec<usize> = (0..lazy.codomain_rank()).collect();
        let cols: Vec<usize> =
            (lazy.codomain_rank()..lazy.codomain_rank() + lazy.domain_rank()).collect();
        let (r, c) = (rows.as_slice(), cols.as_slice());
        for (name, lazy_lq, eager_lq) in [
            ("lq_compact", lazy.lq_compact(r, c), eager.lq_compact(r, c)),
            ("lq_full", lazy.lq_full(r, c), eager.lq_full(r, c)),
        ] {
            let (Lq { l, q }, Lq { l: el, q: eq }) = (lazy_lq.unwrap(), eager_lq.unwrap());
            assert_factor_close!(format!("{label} {name} l"), l, el);
            assert_factor_close!(format!("{label} {name} q"), q, eq);
        }
    }};
}

/// Tall, wide and square-with-dual parents on legs `a`, `b`, both dtypes.
macro_rules! assert_fixtures {
    ($label:expr, $runtime:expr, $a:expr, $b:expr) => {{
        let (runtime, a, b) = (&$runtime, &$a, &$b);
        let dual = a.try_dual().unwrap();
        macro_rules! for_dtype {
            ($d:ty, $seed:expr) => {{
                let label = format!("{} {}", $label, stringify!($d));
                let tall = TensorMap::<_, $d>::rand_with_seed(runtime, [a, b], [b], $seed).unwrap();
                assert_lazy_lq!(format!("{label} tall"), tall);
                let wide =
                    TensorMap::<_, $d>::rand_with_seed(runtime, [b], [a, b], $seed + 1).unwrap();
                assert_lazy_lq!(format!("{label} wide"), wide);
                let dual_legs =
                    TensorMap::<_, $d>::rand_with_seed(runtime, [&dual, b], [b, &dual], $seed + 2)
                        .unwrap();
                assert_lazy_lq!(format!("{label} dual"), dual_legs);
            }};
        }
        for_dtype!(f64, 2070);
        for_dtype!(Complex64, 3070);
    }};
}

#[test]
fn u1_lazy_adjoint_lq_matches_the_materialized_adjoint() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
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
    assert_fixtures!("U(1)", runtime, a, b);
}

#[test]
fn su2_lazy_adjoint_lq_matches_the_materialized_adjoint() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider = Arc::new(SU2FusionRule);
    let leg = |pairs: &[(usize, usize)]| {
        GradedSpace::try_new(
            Arc::clone(&provider),
            pairs
                .iter()
                .map(|&(twice, d)| (SU2Irrep::from_twice_spin(twice), d)),
        )
        .unwrap()
    };
    let a = leg(&[(0, 2), (1, 3), (2, 1)]);
    let b = leg(&[(0, 1), (1, 2)]);
    assert_fixtures!("SU(2)", runtime, a, b);
}

#[test]
fn fermionic_u1_lazy_adjoint_lq_matches_the_materialized_adjoint() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
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
    assert_fixtures!("fZ2xU(1)", runtime, a, b);
}

/// Checked Generic: SU(3) with outer multiplicity (`8 ⊗ 8 ∋ 8` twice) and
/// non-self-dual `3`, `3̄` legs.
#[cfg(feature = "cgc-gen")]
#[test]
fn checked_su3_lazy_adjoint_lq_matches_the_materialized_adjoint() {
    use symtn::sector::LieGroupFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let adjoint_irrep =
        GradedSpace::try_new(Arc::clone(&provider), [(vec![2i64, 2], 2), (vec![0, 0], 1)]).unwrap();
    let octet = GradedSpace::try_new(Arc::clone(&provider), [(vec![2i64, 2], 2)]).unwrap();
    assert_fixtures!("checked SU(3) 8", runtime, adjoint_irrep, octet);
    let three =
        GradedSpace::try_new(Arc::clone(&provider), [(vec![1i64, 0], 2), (vec![0, 0], 1)]).unwrap();
    let three_bar =
        GradedSpace::try_new(Arc::clone(&provider), [(vec![0i64, 1], 3), (vec![1, 0], 1)]).unwrap();
    assert_fixtures!("checked SU(3) 3", runtime, three, three_bar);
}
