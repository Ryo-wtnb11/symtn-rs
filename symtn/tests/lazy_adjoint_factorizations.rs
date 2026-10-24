//! Every factorization of a lazy adjoint equals the same factorization of its
//! materialized adjoint, in both fusion modes (#1755): the same factor
//! spaces, the same storage form (compact or dense), and the same values in
//! the documented gauge (dense rounding tolerance).
//!
//! Oracle: `lazy.materialize()` is the exact block copy
//! `block(t^H, c) = block(t, c)^H` (`lazy_adjoint_exact.rs`), factored by the
//! ordinary owned path. The lazy receiver reaches each op with identity leg
//! roles, so it is factored as a lazy adjoint rather than permuted into an
//! owned tensor first.

use std::sync::Arc;

use num_complex::Complex64;
use symtn::sector::{SU2FusionRule, SU2Irrep};
use symtn::typed::{
    Eig, Eigh, GradedSpace, HermitianTol, LeftPolar, Lq, Qr, RightPolar, Runtime, SectorSpectrum,
    Svd, TensorMap,
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
    ($what:expr, $lazy:expr, $eager:expr) => {{
        let (what, lazy, eager) = ($what, &$lazy, &$eager);
        assert_eq!(lazy.codomain(), eager.codomain(), "{what}: codomain");
        assert_eq!(lazy.domain(), eager.domain(), "{what}: domain");
        assert_eq!(
            lazy.diagview().is_ok(),
            eager.diagview().is_ok(),
            "{what}: storage form"
        );
        let (lazy, eager) = (lazy.materialize().unwrap(), eager.materialize().unwrap());
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
    ($label:expr, $parent:expr) => {{
        let label = $label;
        let lazy = $parent.adjoint().unwrap();
        let eager = lazy.materialize().unwrap();
        let rows: Vec<usize> = (0..lazy.codomain_rank()).collect();
        let cols: Vec<usize> =
            (lazy.codomain_rank()..lazy.codomain_rank() + lazy.domain_rank()).collect();
        let (r, c) = (rows.as_slice(), cols.as_slice());

        for (name, lazy_qr, eager_qr) in [
            ("qr_compact", lazy.qr_compact(r, c), eager.qr_compact(r, c)),
            ("qr_full", lazy.qr_full(r, c), eager.qr_full(r, c)),
        ] {
            let (Qr { q, r: rr }, Qr { q: eq, r: er }) = (lazy_qr.unwrap(), eager_qr.unwrap());
            assert_factor_close!(format!("{label} {name} q"), q, eq);
            assert_factor_close!(format!("{label} {name} r"), rr, er);
        }
        for (name, lazy_lq, eager_lq) in [
            ("lq_compact", lazy.lq_compact(r, c), eager.lq_compact(r, c)),
            ("lq_full", lazy.lq_full(r, c), eager.lq_full(r, c)),
        ] {
            let (Lq { l, q }, Lq { l: el, q: eq }) = (lazy_lq.unwrap(), eager_lq.unwrap());
            assert_factor_close!(format!("{label} {name} l"), l, el);
            assert_factor_close!(format!("{label} {name} q"), q, eq);
        }
        for (name, lazy_svd, eager_svd) in [
            (
                "svd_compact",
                lazy.svd_compact(r, c),
                eager.svd_compact(r, c),
            ),
            ("svd_full", lazy.svd_full(r, c), eager.svd_full(r, c)),
        ] {
            let (
                Svd { u, s, vh },
                Svd {
                    u: eu,
                    s: es,
                    vh: evh,
                },
            ) = (lazy_svd.unwrap(), eager_svd.unwrap());
            assert_factor_close!(format!("{label} {name} u"), u, eu);
            assert_factor_close!(format!("{label} {name} s"), s, es);
            assert_factor_close!(format!("{label} {name} vh"), vh, evh);
        }
        assert_spectra_close(
            &format!("{label} svd_vals"),
            &lazy.svd_vals(r, c).unwrap(),
            &eager.svd_vals(r, c).unwrap(),
        );
        assert_factor_close!(
            format!("{label} left_null"),
            lazy.left_null(r, c).unwrap(),
            eager.left_null(r, c).unwrap()
        );
        assert_factor_close!(
            format!("{label} right_null"),
            lazy.right_null(r, c).unwrap(),
            eager.right_null(r, c).unwrap()
        );
        assert_factor_close!(
            format!("{label} pinv"),
            lazy.pinv(r, c, 1.0e-12).unwrap(),
            eager.pinv(r, c, 1.0e-12).unwrap()
        );
    }};
}

/// Every endomorphism factorization on the lazy adjoint of `parent` (square
/// coupled-sector blocks, so both polar directions apply), and the Hermitian
/// ones on the lazy adjoint of `hermitian`.
macro_rules! assert_endomorphism_ops {
    ($label:expr, $parent:expr, $hermitian:expr) => {{
        let label = $label;
        let lazy = $parent.adjoint().unwrap();
        let eager = lazy.materialize().unwrap();
        let rows: Vec<usize> = (0..lazy.codomain_rank()).collect();
        let cols: Vec<usize> =
            (lazy.codomain_rank()..lazy.codomain_rank() + lazy.domain_rank()).collect();
        let (r, c) = (rows.as_slice(), cols.as_slice());

        let (LeftPolar { w, p }, LeftPolar { w: ew, p: ep }) = (
            lazy.left_polar(r, c).unwrap(),
            eager.left_polar(r, c).unwrap(),
        );
        assert_factor_close!(format!("{label} left_polar w"), w, ew);
        assert_factor_close!(format!("{label} left_polar p"), p, ep);
        let (RightPolar { p, wh }, RightPolar { p: ep, wh: ewh }) = (
            lazy.right_polar(r, c).unwrap(),
            eager.right_polar(r, c).unwrap(),
        );
        assert_factor_close!(format!("{label} right_polar p"), p, ep);
        assert_factor_close!(format!("{label} right_polar wh"), wh, ewh);
        assert_factor_close!(
            format!("{label} inv"),
            lazy.inv(r, c).unwrap(),
            eager.inv(r, c).unwrap()
        );
        assert_factor_close!(
            format!("{label} exp"),
            lazy.exp(r, c).unwrap(),
            eager.exp(r, c).unwrap()
        );
        assert_spectra_close(
            &format!("{label} eig_vals"),
            &lazy.eig_vals(r, c).unwrap(),
            &eager.eig_vals(r, c).unwrap(),
        );
        let (Eig { d, v }, Eig { d: ed, v: ev }) =
            (lazy.eig_full(r, c).unwrap(), eager.eig_full(r, c).unwrap());
        assert_factor_close!(format!("{label} eig_full d"), d, ed);
        assert_factor_close!(format!("{label} eig_full v"), v, ev);

        let lazy = $hermitian.adjoint().unwrap();
        let eager = lazy.materialize().unwrap();
        let tol = HermitianTol::DEFAULT;
        assert_spectra_close(
            &format!("{label} eigh_vals"),
            &lazy.eigh_vals(r, c, tol).unwrap(),
            &eager.eigh_vals(r, c, tol).unwrap(),
        );
        let (Eigh { d, v }, Eigh { d: ed, v: ev }) = (
            lazy.eigh_full(r, c, tol).unwrap(),
            eager.eigh_full(r, c, tol).unwrap(),
        );
        assert_factor_close!(format!("{label} eigh_full d"), d, ed);
        assert_factor_close!(format!("{label} eigh_full v"), v, ev);
    }};
}

/// Rectangular `[a, b] <- [a]` and endomorphism `[a, a] <- [a, a]` fixtures
/// with a Hermitian `X X^H`, for both scalar types.
macro_rules! assert_all_ops {
    ($label:expr, $runtime:expr, $a:expr, $b:expr) => {{
        let (runtime, a, b) = (&$runtime, &$a, &$b);
        macro_rules! for_dtype {
            ($d:ty, $seed:expr) => {{
                let label = format!("{} {}", $label, stringify!($d));
                let rectangular =
                    TensorMap::<_, $d>::rand_with_seed(runtime, [a, b], [a], $seed).unwrap();
                assert_rectangular_ops!(format!("{label} tall"), rectangular);
                let wide =
                    TensorMap::<_, $d>::rand_with_seed(runtime, [a], [a, b], $seed + 1).unwrap();
                assert_rectangular_ops!(format!("{label} wide"), wide);
                let square =
                    TensorMap::<_, $d>::rand_with_seed(runtime, [a, a], [a, a], $seed + 2).unwrap();
                let hermitian = square
                    .compose(&square.adjoint().unwrap().materialize().unwrap())
                    .unwrap();
                assert_endomorphism_ops!(label, square, hermitian);
            }};
        }
        for_dtype!(f64, 1755);
        for_dtype!(Complex64, 2755);
    }};
}

#[test]
fn multiplicity_free_su2_lazy_adjoint_factorizations_match_the_materialized_adjoint() {
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
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
    assert_all_ops!("MF SU(2)", runtime, a, b);
}

#[cfg(feature = "cgc-gen")]
#[test]
fn checked_su2_lazy_adjoint_factorizations_match_the_materialized_adjoint() {
    use symtn::sector::LieGroupFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(2).unwrap()).unwrap());
    let a = GradedSpace::try_new(
        Arc::clone(&provider),
        [(vec![0i64], 2), (vec![1], 2), (vec![2], 1)],
    )
    .unwrap();
    let b = GradedSpace::try_new(Arc::clone(&provider), [(vec![0i64], 1), (vec![1], 2)]).unwrap();
    assert_all_ops!("checked SU(2)", runtime, a, b);
}

/// SU(3) `8 ⊗ 8 ∋ 8` twice: blocks that differ only by their vertex label
/// carry their own entries, so the swapped factor spaces of a lazy adjoint
/// must keep the multiplicity index on the right side.
#[cfg(feature = "cgc-gen")]
#[test]
fn checked_su3_multiplicity_lazy_adjoint_factorizations_match_the_materialized_adjoint() {
    use symtn::sector::LieGroupFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let a =
        GradedSpace::try_new(Arc::clone(&provider), [(vec![2i64, 2], 1), (vec![0, 0], 1)]).unwrap();
    let b = GradedSpace::try_new(Arc::clone(&provider), [(vec![2i64, 2], 2)]).unwrap();
    let probe = TensorMap::<_, f64>::rand_with_seed(&runtime, [&a, &a], [&a, &a], 0).unwrap();
    assert!(
        (0..probe.subblock_count()).any(|index| probe
            .subblock_fusion_trees(index)
            .unwrap()
            .codomain_vertices()
            .iter()
            .any(|vertex| vertex.get() == 2)),
        "fixture must carry a Generic vertex key mu = 2"
    );
    assert_all_ops!("checked SU(3)", runtime, a, b);
}

/// The checked parent-reading seams (compact and full SVD, pinv) on
/// non-self-dual SU(3) sectors `3`, `3̄` and on dual legs: the swapped hom
/// space of the adjoint must keep each leg's sector and duality.
#[cfg(feature = "cgc-gen")]
#[test]
fn checked_su3_seams_keep_non_self_dual_sectors_and_dual_legs() {
    use symtn::sector::LieGroupFusionRule;

    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let three =
        GradedSpace::try_new(Arc::clone(&provider), [(vec![1i64, 0], 2), (vec![0, 0], 1)]).unwrap();
    let three_bar =
        GradedSpace::try_new(Arc::clone(&provider), [(vec![0i64, 1], 3), (vec![1, 0], 1)]).unwrap();
    let dual = three.try_dual().unwrap();
    assert!(dual.is_dual() && !three.is_dual());

    macro_rules! seams {
        ($d:ty, $seed:expr, [$($codomain:expr),+], [$($domain:expr),+]) => {{
            let parent = TensorMap::<_, $d>::rand_with_seed(
                &runtime,
                [$(&$codomain),+],
                [$(&$domain),+],
                $seed,
            )
            .unwrap();
            let label = format!("{} {}", stringify!($d), $seed);
            let lazy = parent.adjoint().unwrap();
            let eager = lazy.materialize().unwrap();
            let rows: Vec<usize> = (0..lazy.codomain_rank()).collect();
            let cols: Vec<usize> =
                (lazy.codomain_rank()..lazy.codomain_rank() + lazy.domain_rank()).collect();
            let (r, c) = (rows.as_slice(), cols.as_slice());
            for (name, lazy_svd, eager_svd) in [
                ("svd_compact", lazy.svd_compact(r, c), eager.svd_compact(r, c)),
                ("svd_full", lazy.svd_full(r, c), eager.svd_full(r, c)),
            ] {
                let (Svd { u, s, vh }, Svd { u: eu, s: es, vh: evh }) =
                    (lazy_svd.unwrap(), eager_svd.unwrap());
                assert_factor_close!(format!("{label} {name} u"), u, eu);
                assert_factor_close!(format!("{label} {name} s"), s, es);
                assert_factor_close!(format!("{label} {name} vh"), vh, evh);
            }
            assert_factor_close!(
                format!("{label} pinv"),
                lazy.pinv(r, c, 1.0e-12).unwrap(),
                eager.pinv(r, c, 1.0e-12).unwrap()
            );
        }};
    }
    seams!(f64, 31, [three, three_bar], [three]);
    seams!(Complex64, 32, [three, three_bar], [three]);
    seams!(f64, 33, [three_bar], [three, dual]);
    seams!(Complex64, 34, [three_bar], [three, dual]);
    seams!(f64, 35, [dual, three_bar], [dual]);
    seams!(Complex64, 36, [dual, three_bar], [dual]);
}
