//! Factorization-family oracle for the single-precision payloads (#1324).
//!
//! Oracle, tolerance, fixtures and the exactly-widening draw stream come from
//! `single_precision_oracle`, shared with the base-family suite (#1315).
//!
//! Two kinds of evidence appear here, and they are kept apart on purpose:
//!
//! * **Gauge-fixed factors** are compared entry by entry against the
//!   double-precision factorization of the exactly widened input. `qr_compact`
//!   and `lq_compact` with a positive real diagonal are unique for a
//!   full-column-rank (full-row-rank) block, and the polar factors `W`, `P` are
//!   unique for a full-rank block, so the double-precision payload is a
//!   legitimate pointwise oracle for them. So are the spectra, which are
//!   ordered.
//! * **Gauge-dependent factors** — the singular vectors, the Hermitian
//!   eigenvectors, the extra columns of a `*_full` factorization and the null
//!   bases — are compared only through gauge-independent identities
//!   (reconstruction, isometry, annihilation), because LAPACK's `s`/`c` and
//!   `d`/`z` drivers are under no obligation to pick the same phase.
//!
//! Tolerances are `K * sqrt(n) * eps(f32) * max(1, scale) * kappa` with
//! `K = 32` from the shared module, `n` the payload length and `kappa` the
//! **measured** condition number of the fixture — computed from the
//! double-precision `svd_vals` of that very fixture, printed in every failure
//! message, and applied only where the quantity under test is a forward error
//! (a gauge-fixed factor, a null-space annihilation). Reconstruction and
//! isometry laws are backward stable and take `kappa = 1`.

use crate::single_precision_oracle;

use crate::numerics;

include!("common/predicate_chains.rs");
include!("common/predicate_chain_coefficients.rs");

use num_complex::{Complex32, Complex64};
use symtn::sector::{U1FusionRule, U1Irrep};
use symtn::typed::HermitianTol;
use symtn::typed::{Eigh, LeftPolar, Lq, Qr, RightPolar, Svd};
use symtn::typed::{TensorMap, Truncation};

use single_precision_oracle::{
    assert_payloads_agree_scaled, assert_scalars_agree, fermion_su2_leg_with, minus_one, one,
    runtime, tolerance, u1_leg_with,
};

/// The truncated factorizations are compositions (#1534):
/// `svd_compact`/`eigh_full`/`eig_full` → `diagview` → `find_truncated` →
/// `restrict_leg`.
#[allow(dead_code)] // each case reads the factors it checks
struct SvdTruncated<T> {
    u: T,
    s: T,
    vh: T,
    error: f64,
}

#[allow(dead_code)] // each case reads the factors it checks
struct EigenTruncated<T> {
    d: T,
    v: T,
    error: f64,
}

macro_rules! svd_trunc {
    ($engine:ident, $tensor:expr, $truncation:expr) => {{
        let Svd { u, s, vh } = $tensor.svd_compact($engine, &[0], &[1]).unwrap();
        let found = s.domain()[0]
            .find_truncated(&s.diagview($engine).unwrap(), $truncation)
            .unwrap();
        SvdTruncated {
            u: u.restrict_leg($engine, &[(u.codomain_rank(), &found.selection)])
                .unwrap(),
            s: s.restrict_leg($engine, &[(0, &found.selection), (1, &found.selection)])
                .unwrap(),
            vh: vh.restrict_leg($engine, &[(0, &found.selection)]).unwrap(),
            error: found.error,
        }
    }};
}

macro_rules! eigen_trunc {
    ($engine:ident, $full:expr, $truncation:expr) => {{
        // `$full` is an `Eigh` or an `Eig`; both name their factors `d`, `v`.
        let full = $full.unwrap();
        let (d, v) = (full.d, full.v);
        let found = d.domain()[0]
            .find_truncated(&d.diagview($engine).unwrap(), $truncation)
            .unwrap();
        EigenTruncated {
            d: d.restrict_leg($engine, &[(0, &found.selection), (1, &found.selection)])
                .unwrap(),
            v: v.restrict_leg($engine, &[(v.codomain_rank(), &found.selection)])
                .unwrap(),
            error: found.error,
        }
    }};
}

/// `tol` for the tolerance-taking predicates at single precision: far above
/// `eps(f32)` and far below the fixture's own scale. MatrixAlgebraKit's
/// `defaulttol` (`src/common/defaults.jl`) is `eps(real(T))^(2/3)`, which is
/// `2.4e-5` at `f32`; this is the same order.
const PREDICATE_TOL: f64 = 1e-4;

/// The entry of the shared fixture at the `(row, col)` of a coupled
/// block: a well-separated diagonal with a constant off-diagonal, so the
/// singular values are distinct and the factorizations are gauge-fixed.
fn fixture_entry(row: usize, col: usize) -> (f32, f32) {
    const DIAGONAL: [f32; 4] = [8.0, 4.0, 2.0, 1.0];
    if row == col {
        (DIAGONAL[row % 4], 0.0)
    } else if row < col {
        (0.25, 0.125)
    } else {
        (0.25, -0.125)
    }
}

/// A fixture whose Hermitian part has eigenvalues of both signs, so
/// `is_posdef` has something to answer `false` about.
fn indefinite_entry(row: usize, col: usize) -> (f32, f32) {
    const DIAGONAL: [f32; 4] = [3.0, -1.0, 2.0, -4.0];
    if row == col {
        (DIAGONAL[row % 4], 0.0)
    } else if row < col {
        (0.25, 0.125)
    } else {
        (0.25, -0.125)
    }
}

/// `sigma_max / sigma_min` over every coupled block of a double-precision
/// tensor: the conditioning the forward-error bounds below are scaled by.
///
/// A macro rather than a function so it does not have to repeat the dispatch
/// bounds `svd_vals` carries.
macro_rules! measured_kappa {
    ($engine:ident, $tensor:expr) => {{
        let values = $tensor.svd_vals($engine, &[0], &[1]).unwrap();
        let mut largest = 0.0f64;
        let mut smallest = f64::INFINITY;
        for entry in &values {
            for &value in &entry.values {
                largest = largest.max(value);
                smallest = smallest.min(value);
            }
        }
        assert!(
            smallest > 0.0 && largest.is_finite(),
            "the fixture must have full rank in every coupled block"
        );
        largest / smallest
    }};
}

macro_rules! assert_reconstruction {
    ($engine:ident, $what:expr, $actual:expr, $expected:expr, $terms:expr, $narrow:ty) => {{
        let actual = $actual;
        let expected = $expected;
        let residual = actual
            .axpby($engine, one::<$narrow>(), expected, minus_one::<$narrow>())
            .unwrap()
            .norm($engine, 2.0)
            .unwrap();
        let bound = tolerance($terms, expected.norm($engine, 2.0).unwrap());
        assert!(
            residual <= bound,
            "{}: residual {residual:e} exceeds tolerance {bound:e}",
            $what
        );
    }};
}

/// `q† ∘ q == id` on the domain.
macro_rules! assert_isometry {
    ($engine:ident, $what:expr, $rt:expr, $q:expr, $terms:expr, $narrow:ty) => {{
        let q = $q;
        let identity = TensorMap::<_, $narrow>::isomorphism(
            $engine,
            $rt,
            q.domain().iter(),
            q.domain().iter(),
        )
        .unwrap();
        assert_reconstruction!(
            $engine,
            $what,
            &q.adjoint($engine).unwrap().compose($engine, q).unwrap(),
            &identity,
            $terms,
            $narrow
        );
    }};
}

/// `q ∘ q† == id` on the codomain.
macro_rules! assert_coisometry {
    ($engine:ident, $what:expr, $rt:expr, $q:expr, $terms:expr, $narrow:ty) => {{
        let q = $q;
        let identity = TensorMap::<_, $narrow>::isomorphism(
            $engine,
            $rt,
            q.codomain().iter(),
            q.codomain().iter(),
        )
        .unwrap();
        let q_adjoint = q.adjoint($engine).unwrap();
        let q_compose = q.compose($engine, &q_adjoint).unwrap();
        assert_reconstruction!($engine, $what, &q_compose, &identity, $terms, $narrow);
    }};
}

macro_rules! assert_unitary {
    ($engine:ident, $what:expr, $rt:expr, $q:expr, $terms:expr, $narrow:ty) => {{
        assert_isometry!($engine, $what, $rt, $q, $terms, $narrow);
        assert_coisometry!($engine, $what, $rt, $q, $terms, $narrow);
    }};
}

/// The gauge convention `qr`/`lq` fix: the diagonal of the triangular factor is
/// positive and real in every coupled block. Read through `diagview`, the
/// public per-sector diagonal of a bond endomorphism.
///
/// The positivity is exact — it is the sign (or phase) the gauge step applies,
/// not a computed quantity. The *reality* of a complex diagonal is not: the
/// phase is applied as a multiplication, so the imaginary part comes back at
/// the rounding of that product and is bounded, not zero. The bound is the
/// module tolerance at the entry's own scale, so it carries no absolute
/// constant.
macro_rules! assert_positive_real_diagonal {
    ($engine:ident, $what:expr, $factor:expr) => {{
        let spectra = $factor.diagview($engine).unwrap();
        assert!(
            !spectra.is_empty(),
            "{}: the triangular factor has no coupled blocks",
            $what
        );
        for entry in &spectra {
            for value in &entry.values {
                let value = <_ as single_precision_oracle::Parts>::wide(*value);
                let bound = tolerance(1, value.norm());
                assert!(
                    value.re > 0.0 && value.im.abs() <= bound,
                    "{}: diagonal entry {value} is not positive with an imaginary part \
                     within {bound:e}",
                    $what
                );
            }
        }
    }};
}

/// Spectra are ordered, hence a pointwise oracle. Compared sector by sector.
macro_rules! assert_spectra_agree {
    ($what:expr, $narrow:expr, $wide:expr, $terms:expr) => {{
        let narrow = $narrow;
        let wide = $wide;
        assert_eq!(
            narrow.len(),
            wide.len(),
            "{}: the twins produced a different number of sectors",
            $what
        );
        for (got, expected) in narrow.iter().zip(wide.iter()) {
            assert_eq!(got.sector, expected.sector, "{}: sector order", $what);
            assert_eq!(
                got.values.len(),
                expected.values.len(),
                "{}: sector {:?} has a different spectrum length",
                $what,
                got.sector
            );
            for (index, (&value, &oracle)) in got.values.iter().zip(&expected.values).enumerate() {
                assert_scalars_agree(
                    &format!("{}: sector {:?} value {index}", $what, got.sector),
                    Complex64::new(value, 0.0),
                    Complex64::new(oracle, 0.0),
                    $terms,
                );
            }
        }
    }};
}

/// Every factorization the marker admits, over one provider and one payload
/// dtype, against the double-precision twin.
macro_rules! factor_checks {
    ($engine:ident,
        $name:expr, $narrow:ty, $wide:ty,
        $tall_cod:expr, $tall_dom:expr, $short_cod:expr, $short_dom:expr,
        $min_blocks:expr
    ) => {{
        let rt = runtime();
        let name: &str = $name;
        let tall_cod = $tall_cod;
        let tall_dom = $tall_dom;
        let short_cod = $short_cod;
        let short_dom = $short_dom;
        let (tall, wide_tall) = twin_with!(
            $engine,
            &rt,
            $narrow,
            $wide,
            tall_cod.iter().copied(),
            tall_dom.iter().copied(),
            |index: &[usize]| fixture_entry(index[0], index[1])
        );
        let (short, wide_short) = twin_with!(
            $engine,
            &rt,
            $narrow,
            $wide,
            short_cod.iter().copied(),
            short_dom.iter().copied(),
            |index: &[usize]| fixture_entry(index[0], index[1])
        );
        let terms = wide_tall.dense_data().unwrap().len();
        assert!(
            tall.subblock_count() >= $min_blocks,
            "{name}: the fixture must carry at least {} coupled blocks, has {}",
            $min_blocks,
            tall.subblock_count()
        );

        let kappa = measured_kappa!($engine, &wide_tall);
        let short_kappa = measured_kappa!($engine, &wide_short);
        assert!(
            kappa < 1e4 && short_kappa < 1e4,
            "{name}: the fixture is too ill-conditioned to be a single-precision oracle \
             (kappa {kappa:e}, {short_kappa:e})"
        );

        // ---- QR: gauge-fixed by the positive real diagonal of R. ----------
        let Qr { q, r } = tall.qr_compact($engine, &[0], &[1]).unwrap();
        let Qr { q: wq, r: wr } = wide_tall.qr_compact($engine, &[0], &[1]).unwrap();
        assert_reconstruction!(
            $engine,
            format!("{name}: qr_compact (kappa {kappa:e})"),
            &q.compose($engine, &r).unwrap(),
            &tall,
            terms,
            $narrow
        );
        assert_isometry!(
            $engine,
            format!("{name}: qr_compact Q"),
            &rt,
            &q,
            terms,
            $narrow
        );
        assert_positive_real_diagonal!($engine, format!("{name}: qr_compact R"), r);
        assert_payloads_agree_scaled(
            &format!("{name}: qr_compact Q against the widened oracle (kappa {kappa:e})"),
            q.dense_data().unwrap(),
            wq.dense_data().unwrap(),
            terms,
            kappa,
        );
        assert_payloads_agree_scaled(
            &format!("{name}: qr_compact R against the widened oracle (kappa {kappa:e})"),
            r.dense_data().unwrap(),
            wr.dense_data().unwrap(),
            terms,
            kappa,
        );

        // `*_full` adds null columns whose gauge nothing fixes: identities only.
        let Qr { q, r } = tall.qr_full($engine, &[0], &[1]).unwrap();
        assert_reconstruction!(
            $engine,
            format!("{name}: qr_full"),
            &q.compose($engine, &r).unwrap(),
            &tall,
            terms,
            $narrow
        );
        assert_isometry!(
            $engine,
            format!("{name}: qr_full Q"),
            &rt,
            &q,
            terms,
            $narrow
        );

        // ---- LQ. ---------------------------------------------------
        let Lq { l, q } = short.lq_compact($engine, &[0], &[1]).unwrap();
        let Lq { l: wl, q: wq } = wide_short.lq_compact($engine, &[0], &[1]).unwrap();
        assert_reconstruction!(
            $engine,
            format!("{name}: lq_compact"),
            &l.compose($engine, &q).unwrap(),
            &short,
            terms,
            $narrow
        );
        assert_coisometry!(
            $engine,
            format!("{name}: lq_compact Q"),
            &rt,
            &q,
            terms,
            $narrow
        );
        assert_positive_real_diagonal!($engine, format!("{name}: lq_compact L"), l);
        assert_payloads_agree_scaled(
            &format!("{name}: lq_compact L against the widened oracle (kappa {short_kappa:e})"),
            l.dense_data().unwrap(),
            wl.dense_data().unwrap(),
            terms,
            short_kappa,
        );
        assert_payloads_agree_scaled(
            &format!("{name}: lq_compact Q against the widened oracle (kappa {short_kappa:e})"),
            q.dense_data().unwrap(),
            wq.dense_data().unwrap(),
            terms,
            short_kappa,
        );

        let Lq { l, q } = short.lq_full($engine, &[0], &[1]).unwrap();
        assert_reconstruction!(
            $engine,
            format!("{name}: lq_full"),
            &l.compose($engine, &q).unwrap(),
            &short,
            terms,
            $narrow
        );
        assert_coisometry!(
            $engine,
            format!("{name}: lq_full Q"),
            &rt,
            &q,
            terms,
            $narrow
        );

        // ---- SVD: spectra are ordered, so they are the pointwise oracle. ---
        let Svd { u, s, vh } = tall.svd_compact($engine, &[0], &[1]).unwrap();
        assert_reconstruction!(
            $engine,
            format!("{name}: svd_compact"),
            &u.compose($engine, &s)
                .unwrap()
                .compose($engine, &vh)
                .unwrap(),
            &tall,
            terms,
            $narrow
        );
        assert_isometry!(
            $engine,
            format!("{name}: svd_compact U"),
            &rt,
            &u,
            terms,
            $narrow
        );
        assert_coisometry!(
            $engine,
            format!("{name}: svd_compact Vh"),
            &rt,
            &vh,
            terms,
            $narrow
        );

        let Svd { u, s, vh } = tall.svd_full($engine, &[0], &[1]).unwrap();
        assert_reconstruction!(
            $engine,
            format!("{name}: svd_full"),
            &u.compose($engine, &s)
                .unwrap()
                .compose($engine, &vh)
                .unwrap(),
            &tall,
            terms,
            $narrow
        );
        assert_isometry!(
            $engine,
            format!("{name}: svd_full U"),
            &rt,
            &u,
            terms,
            $narrow
        );

        let values = tall.svd_vals($engine, &[0], &[1]).unwrap();
        let wide_values = wide_tall.svd_vals($engine, &[0], &[1]).unwrap();
        assert_spectra_agree!(format!("{name}: svd_vals"), &values, &wide_values, terms);
        for entry in &values {
            assert!(
                entry.values.windows(2).all(|pair| pair[0] >= pair[1]),
                "{name}: singular values must be descending, got {:?}",
                entry.values
            );
            assert!(
                entry.values.iter().all(|&value| value >= 0.0),
                "{name}: singular values must be non-negative"
            );
        }

        // ---- Truncation away from a tie. ----------------------------------
        // The fixture's singular values are separated by factors of about two,
        // which is far above the `eps(f32) * kappa` noise of the
        // decomposition, so the kept set is reproducible across dtypes.
        let truncated = svd_trunc!($engine, tall, &Truncation::rank(2));
        let wide_truncated = svd_trunc!($engine, wide_tall, &Truncation::rank(2));
        assert_eq!(
            truncated
                .s
                .materialize($engine)
                .unwrap()
                .dense_data()
                .unwrap()
                .len(),
            wide_truncated
                .s
                .materialize($engine)
                .unwrap()
                .dense_data()
                .unwrap()
                .len(),
            "{name}: svd_trunc kept a different number of states than the widened oracle"
        );
        assert!(
            truncated.error > 0.0,
            "{name}: the rank budget must discard something"
        );
        assert_scalars_agree(
            &format!("{name}: svd_trunc error"),
            Complex64::new(truncated.error, 0.0),
            Complex64::new(wide_truncated.error, 0.0),
            terms,
        );
        let reconstructed = truncated
            .u
            .compose($engine, &truncated.s)
            .unwrap()
            .compose($engine, &truncated.vh)
            .unwrap();
        let residual = reconstructed
            .axpby($engine, one::<$narrow>(), &tall, minus_one::<$narrow>())
            .unwrap()
            .norm($engine, 2.0)
            .unwrap();
        assert!(
            (residual - truncated.error).abs() <= tolerance(terms, truncated.error),
            "{name}: svd_trunc reports error {} against residual {residual}",
            truncated.error
        );

        // ---- Hermitian eigendecomposition. --------------------------------
        let (h, wide_h) = twin_with!(
            $engine,
            &rt,
            $narrow,
            $wide,
            short_cod.iter().copied(),
            short_cod.iter().copied(),
            |index: &[usize]| fixture_entry(index[0], index[1])
        );
        let h_terms = wide_h.dense_data().unwrap().len();

        let Eigh { d, v } = h
            .eigh_full($engine, &[0], &[1], HermitianTol::DEFAULT)
            .unwrap();
        let v_adjoint = v.adjoint($engine).unwrap();
        assert_reconstruction!(
            $engine,
            format!("{name}: eigh_full"),
            &v.compose($engine, &d)
                .unwrap()
                .compose($engine, &v_adjoint)
                .unwrap(),
            &h,
            h_terms,
            $narrow
        );
        assert_unitary!(
            $engine,
            format!("{name}: eigh_full V"),
            &rt,
            &v,
            h_terms,
            $narrow
        );

        let eigenvalues = h
            .eigh_vals($engine, &[0], &[1], HermitianTol::DEFAULT)
            .unwrap();
        let wide_eigenvalues = wide_h
            .eigh_vals($engine, &[0], &[1], HermitianTol::DEFAULT)
            .unwrap();
        assert_spectra_agree!(
            format!("{name}: eigh_vals"),
            &eigenvalues,
            &wide_eigenvalues,
            h_terms
        );
        for entry in &eigenvalues {
            assert!(
                entry.values.windows(2).all(|pair| pair[0] <= pair[1]),
                "{name}: Hermitian eigenvalues must be ascending, got {:?}",
                entry.values
            );
        }

        let truncated = eigen_trunc!(
            $engine,
            h.eigh_full($engine, &[0], &[1], HermitianTol::DEFAULT),
            &Truncation::rank(1)
        );
        let wide_truncated = eigen_trunc!(
            $engine,
            wide_h.eigh_full($engine, &[0], &[1], HermitianTol::DEFAULT),
            &Truncation::rank(1)
        );
        assert!(
            truncated.error > 0.0,
            "{name}: eigh_trunc discarded nothing"
        );
        assert_eq!(
            truncated
                .d
                .materialize($engine)
                .unwrap()
                .dense_data()
                .unwrap()
                .len(),
            wide_truncated
                .d
                .materialize($engine)
                .unwrap()
                .dense_data()
                .unwrap()
                .len(),
            "{name}: eigh_trunc kept a different number of states than the widened oracle"
        );
        assert_scalars_agree(
            &format!("{name}: eigh_trunc error"),
            Complex64::new(truncated.error, 0.0),
            Complex64::new(wide_truncated.error, 0.0),
            h_terms,
        );

        // ---- Null spaces. --------------------------------------------------
        // A tall full-column-rank map has a left null space of the leftover
        // rows; the annihilation is a forward error, hence the `kappa`.
        let left = tall.left_null($engine, &[0], &[1]).unwrap();
        assert!(
            left.dense_data().unwrap().len() > 0,
            "{name}: the tall fixture must have a nonempty left null space"
        );
        let annihilated = left
            .adjoint($engine)
            .unwrap()
            .compose($engine, &tall)
            .unwrap()
            .norm($engine, 2.0)
            .unwrap();
        let bound = tolerance(terms, tall.norm($engine, 2.0).unwrap()) * kappa;
        assert!(
            annihilated <= bound,
            "{name}: left_null does not annihilate the fixture, \
             residual {annihilated:e} > {bound:e} (kappa {kappa:e})"
        );
        assert_isometry!(
            $engine,
            format!("{name}: left_null"),
            &rt,
            &left,
            terms,
            $narrow
        );

        let right = short.right_null($engine, &[0], &[1]).unwrap();
        assert!(
            right.dense_data().unwrap().len() > 0,
            "{name}: the wide fixture must have a nonempty right null space"
        );
        let right_adjoint = right.adjoint($engine).unwrap();
        let annihilated = short
            .compose($engine, &right_adjoint)
            .unwrap()
            .norm($engine, 2.0)
            .unwrap();
        let bound = tolerance(terms, short.norm($engine, 2.0).unwrap()) * short_kappa;
        assert!(
            annihilated <= bound,
            "{name}: right_null does not annihilate the fixture, \
             residual {annihilated:e} > {bound:e}"
        );
        assert_coisometry!(
            $engine,
            format!("{name}: right_null"),
            &rt,
            &right,
            terms,
            $narrow
        );

        // ---- Polar: both factors are unique for a full-rank block. ---------
        let LeftPolar { w, p } = tall.left_polar($engine, &[0], &[1]).unwrap();
        let LeftPolar { w: ww, p: wp } = wide_tall.left_polar($engine, &[0], &[1]).unwrap();
        assert_reconstruction!(
            $engine,
            format!("{name}: left_polar"),
            &w.compose($engine, &p).unwrap(),
            &tall,
            terms,
            $narrow
        );
        assert_isometry!(
            $engine,
            format!("{name}: left_polar W"),
            &rt,
            &w,
            terms,
            $narrow
        );
        assert!(
            p.eigh_vals($engine, &[0], &[1], HermitianTol::DEFAULT)
                .unwrap()
                .iter()
                .flat_map(|entry| &entry.values)
                .all(|&value| value >= -tolerance(terms, p.norm($engine, 2.0).unwrap())),
            "{name}: the left polar positive factor must be positive semidefinite"
        );
        assert_payloads_agree_scaled(
            &format!("{name}: left_polar W against the widened oracle (kappa {kappa:e})"),
            w.dense_data().unwrap(),
            ww.dense_data().unwrap(),
            terms,
            kappa,
        );
        assert_payloads_agree_scaled(
            &format!("{name}: left_polar P against the widened oracle (kappa {kappa:e})"),
            p.dense_data().unwrap(),
            wp.dense_data().unwrap(),
            terms,
            kappa,
        );

        let RightPolar { p, wh: w } = short.right_polar($engine, &[0], &[1]).unwrap();
        assert_reconstruction!(
            $engine,
            format!("{name}: right_polar"),
            &p.compose($engine, &w).unwrap(),
            &short,
            terms,
            $narrow
        );
        assert_coisometry!(
            $engine,
            format!("{name}: right_polar W"),
            &rt,
            &w,
            terms,
            $narrow
        );

        // ---- Lazy-adjoint receiver. ----------------------------------------
        // `short.adjoint()` is tall and is not materialized before the
        // factorization asks for its blocks.
        let lazy = short.adjoint($engine).unwrap();
        let Svd { u, s, vh } = lazy.svd_compact($engine, &[0], &[1]).unwrap();
        assert_reconstruction!(
            $engine,
            format!("{name}: svd_compact on a lazy adjoint"),
            &u.compose($engine, &s)
                .unwrap()
                .compose($engine, &vh)
                .unwrap(),
            &lazy,
            terms,
            $narrow
        );
        let lazy_values = lazy.svd_vals($engine, &[0], &[1]).unwrap();
        let wide_lazy_values = wide_short
            .adjoint($engine)
            .unwrap()
            .svd_vals($engine, &[0], &[1])
            .unwrap();
        assert_spectra_agree!(
            format!("{name}: svd_vals on a lazy adjoint"),
            &lazy_values,
            &wide_lazy_values,
            terms
        );
    }};
}

/// `is_hermitian!` / `is_posdef!`: the predicate chains that factorize. They
/// are checked on multiplicity-free fixtures only, so they are a separate macro
/// rather than part of `factor_checks!`, which the Checked-Generic provider
/// shares.
///
/// Both are decided against the *widened* oracle rather than against a
/// hard-coded answer: what is under test is that a single-precision payload
/// reaches the same verdict, at a `tol` scaled to its own epsilon.
macro_rules! hermitian_predicate_checks {
    ($engine:ident, $name:expr, $narrow:ty, $wide:ty, $leg:expr) => {{
        let rt = runtime();
        let name: &str = $name;
        let leg = $leg;
        let tall_leg = $leg;

        let (tall, wide_tall) = twin_with!(
            $engine,
            &rt,
            $narrow,
            $wide,
            [&tall_leg],
            [&leg],
            |index: &[usize]| fixture_entry(index[0], index[1])
        );
        let gram = tall
            .adjoint($engine)
            .unwrap()
            .compose($engine, &tall)
            .unwrap();
        let wide_gram = wide_tall
            .adjoint($engine)
            .unwrap()
            .compose($engine, &wide_tall)
            .unwrap();
        assert!(
            is_hermitian!($engine, gram, PREDICATE_TOL),
            "{name}: a Gram matrix is Hermitian"
        );
        assert!(
            is_posdef!($engine, gram, PREDICATE_TOL),
            "{name}: the Gram matrix of a full-column-rank fixture is positive definite"
        );
        assert_eq!(
            is_posdef!($engine, gram, PREDICATE_TOL),
            is_posdef!($engine, wide_gram, PREDICATE_TOL),
            "{name}: is_posdef disagreed with the widened oracle on the Gram matrix"
        );

        let (indefinite, wide_indefinite) = twin_with!(
            $engine,
            &rt,
            $narrow,
            $wide,
            [&leg],
            [&leg],
            |index: &[usize]| { indefinite_entry(index[0], index[1]) }
        );
        assert!(
            is_hermitian!($engine, indefinite, PREDICATE_TOL),
            "{name}: the indefinite fixture is Hermitian by construction"
        );
        assert!(
            !is_posdef!($engine, indefinite, PREDICATE_TOL),
            "{name}: an indefinite fixture must not be positive definite"
        );
        assert_eq!(
            is_posdef!($engine, indefinite, PREDICATE_TOL),
            is_posdef!($engine, wide_indefinite, PREDICATE_TOL),
            "{name}: is_posdef disagreed with the widened oracle on the indefinite fixture"
        );

        // A compact-diagonal receiver: the spectrum factor *is* its own
        // Hermitian spectrum, so `is_posdef` answers without factorizing.
        let Svd { u, .. } = gram.svd_compact($engine, &[0], &[1]).unwrap();
        let _ = u;
        let Svd { s: spectrum, .. } = tall.svd_compact($engine, &[0], &[1]).unwrap();
        assert!(
            is_posdef_compact!($engine, spectrum, PREDICATE_TOL, |v: $narrow| {
                numerics::Numeric::wide(v).re
            }),
            "{name}: a compact singular-value factor with positive values is positive definite"
        );
    }};
}

/// Instantiated once per admitted single-precision dtype.
macro_rules! factorization_suite {
    ($suite:ident, $narrow:ty, $wide:ty) => {
        mod $suite {
            use super::*;

            #[test]
            fn u1_factorizations_match_the_widened_oracle() {
                let engine = &mut symtn::typed::CpuEngine::serial();
                let tall = u1_leg_with([3, 4, 3]);
                let short = u1_leg_with([2, 3, 2]);
                factor_checks!(
                    engine,
                    concat!("U(1) ", stringify!($narrow)),
                    $narrow,
                    $wide,
                    [&tall],
                    [&short],
                    [&short],
                    [&tall],
                    3
                );
                hermitian_predicate_checks!(
                    engine,
                    concat!("U(1) ", stringify!($narrow)),
                    $narrow,
                    $wide,
                    u1_leg_with([2, 3, 2])
                );
            }

            #[test]
            fn fermion_su2_factorizations_match_the_widened_oracle() {
                let engine = &mut symtn::typed::CpuEngine::serial();
                let tall = fermion_su2_leg_with([3, 3, 2]);
                let short = fermion_su2_leg_with([2, 2, 1]);
                factor_checks!(
                    engine,
                    concat!("fZ2 x U(1) x SU(2) ", stringify!($narrow)),
                    $narrow,
                    $wide,
                    [&tall],
                    [&short],
                    [&short],
                    [&tall],
                    3
                );
                hermitian_predicate_checks!(
                    engine,
                    concat!("fZ2 x U(1) x SU(2) ", stringify!($narrow)),
                    $narrow,
                    $wide,
                    fermion_su2_leg_with([2, 2, 1])
                );
            }
        }
    };
}

factorization_suite!(f32_payload, f32, f64);
factorization_suite!(complex32_payload, Complex32, Complex64);

/// Checked-Generic provider: SU(3), whose factorizations run through the
/// checked dispatch (`decide_bond_truncation` with the checked mode and the
/// checked factor plans) rather than the multiplicity-free one.
///
/// A separate, smaller body rather than an instantiation of `factor_checks!`:
/// the isometry laws are expressed here as "the Gram factor is the identity"
/// read through `diagview` plus a pointwise comparison against the widened
/// oracle, which is gauge-independent because the factor's own gauge cancels in
/// `q† ∘ q`.
#[cfg(feature = "cgc-gen")]
mod checked_generic {
    use super::*;
    use std::sync::Arc;
    use symtn::sector::LieGroupFusionRule;
    use symtn::typed::GradedSpace;

    fn su3_leg(degeneracy: usize) -> GradedSpace<LieGroupFusionRule> {
        GradedSpace::try_new(
            Arc::new(
                LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap(),
            ),
            [(vec![2i64, 2], degeneracy)],
        )
        .unwrap()
    }

    macro_rules! checked_generic_suite {
        ($suite:ident, $narrow:ty, $wide:ty) => {
            mod $suite {
                use super::*;

                #[test]
                fn su3_factorizations_match_the_widened_oracle() {
                    let engine = &mut symtn::typed::CpuEngine::serial();
                    let rt = runtime();
                    let name = concat!("SU(3) ", stringify!($narrow));
                    let tall_leg = su3_leg(3);
                    let short_leg = su3_leg(2);

                    let (tall, wide_tall) = twin_with!(
                        engine,
                        &rt,
                        $narrow,
                        $wide,
                        [&tall_leg],
                        [&short_leg],
                        |index: &[usize]| fixture_entry(index[0], index[1])
                    );
                    let (short, wide_short) = twin_with!(
                        engine,
                        &rt,
                        $narrow,
                        $wide,
                        [&short_leg],
                        [&tall_leg],
                        |index: &[usize]| fixture_entry(index[0], index[1])
                    );
                    let terms = wide_tall.dense_data().unwrap().len();
                    let kappa = measured_kappa!(engine, &wide_tall);
                    let short_kappa = measured_kappa!(engine, &wide_short);
                    assert!(
                        kappa < 1e4 && short_kappa < 1e4,
                        "{name}: the fixture is too ill-conditioned to be a \
                         single-precision oracle (kappa {kappa:e}, {short_kappa:e})"
                    );

                    // QR, gauge-fixed by the positive diagonal of R.
                    let Qr { q, r } = tall.qr_compact(engine, &[0], &[1]).unwrap();
                    let Qr { q: wq, r: wr } = wide_tall.qr_compact(engine, &[0], &[1]).unwrap();
                    assert_reconstruction!(
                        engine,
                        format!("{name}: qr_compact (kappa {kappa:e})"),
                        &q.compose(engine, &r).unwrap(),
                        &tall,
                        terms,
                        $narrow
                    );
                    assert_positive_real_diagonal!(engine, format!("{name}: qr_compact R"), r);
                    assert_payloads_agree_scaled(
                        &format!("{name}: qr_compact Q against the widened oracle"),
                        q.dense_data().unwrap(),
                        wq.dense_data().unwrap(),
                        terms,
                        kappa,
                    );
                    assert_payloads_agree_scaled(
                        &format!("{name}: qr_compact R against the widened oracle"),
                        r.dense_data().unwrap(),
                        wr.dense_data().unwrap(),
                        terms,
                        kappa,
                    );

                    let Lq { l, q } = short.lq_compact(engine, &[0], &[1]).unwrap();
                    assert_reconstruction!(
                        engine,
                        format!("{name}: lq_compact"),
                        &l.compose(engine, &q).unwrap(),
                        &short,
                        terms,
                        $narrow
                    );
                    assert_positive_real_diagonal!(engine, format!("{name}: lq_compact L"), l);

                    // SVD and the checked-Generic truncation decision.
                    let Svd { u, s, vh } = tall.svd_compact(engine, &[0], &[1]).unwrap();
                    assert_reconstruction!(
                        engine,
                        format!("{name}: svd_compact"),
                        &u.compose(engine, &s).unwrap().compose(engine, &vh).unwrap(),
                        &tall,
                        terms,
                        $narrow
                    );
                    assert_spectra_agree!(
                        format!("{name}: svd_vals"),
                        &tall.svd_vals(engine, &[0], &[1]).unwrap(),
                        &wide_tall.svd_vals(engine, &[0], &[1]).unwrap(),
                        terms
                    );

                    let truncated = svd_trunc!(engine, tall, &Truncation::rank(1));
                    let wide_truncated = svd_trunc!(engine, wide_tall, &Truncation::rank(1));
                    assert!(
                        truncated.error > 0.0,
                        "{name}: the rank budget must discard something"
                    );
                    assert_eq!(
                        truncated
                            .s
                            .diagview(engine)
                            .unwrap()
                            .iter()
                            .map(|entry| entry.values.len())
                            .sum::<usize>(),
                        wide_truncated
                            .s
                            .diagview(engine)
                            .unwrap()
                            .iter()
                            .map(|entry| entry.values.len())
                            .sum::<usize>(),
                        "{name}: svd_trunc kept a different number of states than the oracle"
                    );
                    assert_scalars_agree(
                        &format!("{name}: svd_trunc error"),
                        Complex64::new(truncated.error, 0.0),
                        Complex64::new(wide_truncated.error, 0.0),
                        terms,
                    );

                    // Hermitian eigendecomposition on the square fixture,
                    // which `fixture_entry` makes Hermitian by construction.
                    let (h, wide_h) = twin_with!(
                        engine,
                        &rt,
                        $narrow,
                        $wide,
                        [&short_leg],
                        [&short_leg],
                        |index: &[usize]| fixture_entry(index[0], index[1])
                    );
                    let h_terms = wide_h.dense_data().unwrap().len();
                    // `v ∘ d ∘ v†` would need the adjoint operand the checked
                    // dispatch rejects, so `eigh_full` is pinned here by its
                    // spectrum factor instead: `d` must carry exactly the
                    // eigenvalues `eigh_vals` reports, which are themselves
                    // compared against the widened oracle just below.
                    let Eigh { d, v } = h
                        .eigh_full(engine, &[0], &[1], HermitianTol::DEFAULT)
                        .unwrap();
                    assert_spectra_agree!(
                        format!("{name}: eigh_full D against eigh_vals"),
                        &d.diagview(engine)
                            .unwrap()
                            .iter()
                            .map(|entry| symtn::typed::SectorSpectrum {
                                sector: entry.sector.clone(),
                                values: entry
                                    .values
                                    .iter()
                                    .map(|value| {
                                        <_ as single_precision_oracle::Parts>::wide(*value).re
                                    })
                                    .collect::<Vec<f64>>(),
                            })
                            .collect::<Vec<_>>(),
                        &h.eigh_vals(engine, &[0], &[1], HermitianTol::DEFAULT)
                            .unwrap(),
                        h_terms
                    );
                    // `v` is checked by the law that does not need an adjoint
                    // operand: `h ∘ v == v ∘ d`, with `h` and `d` both owned.
                    assert_reconstruction!(
                        engine,
                        format!("{name}: eigh_full h∘v == v∘d"),
                        &h.compose(engine, &v).unwrap(),
                        &v.compose(engine, &d).unwrap(),
                        h_terms,
                        $narrow
                    );
                    assert_spectra_agree!(
                        format!("{name}: eigh_vals"),
                        &h.eigh_vals(engine, &[0], &[1], HermitianTol::DEFAULT)
                            .unwrap(),
                        &wide_h
                            .eigh_vals(engine, &[0], &[1], HermitianTol::DEFAULT)
                            .unwrap(),
                        h_terms
                    );
                    let truncated = eigen_trunc!(
                        engine,
                        h.eigh_full(engine, &[0], &[1], HermitianTol::DEFAULT),
                        &Truncation::rank(1)
                    );
                    let wide_truncated = eigen_trunc!(
                        engine,
                        wide_h.eigh_full(engine, &[0], &[1], HermitianTol::DEFAULT),
                        &Truncation::rank(1)
                    );
                    assert_scalars_agree(
                        &format!("{name}: eigh_trunc error"),
                        Complex64::new(truncated.error, 0.0),
                        Complex64::new(wide_truncated.error, 0.0),
                        h_terms,
                    );

                    // Polar, whose factors are unique for a full-rank
                    // block and therefore compared pointwise. The null spaces
                    // and the isometry laws are not asserted here: they are
                    // unrelated to the payload dtype and covered at `f64` by
                    // `checked_generic_facade.rs`. The pointwise agreement
                    // with the widened oracle below pins `W` and `P` — both
                    // unique — completely, and the `f64` factors are the ones
                    // those suites already prove isometric.
                    let LeftPolar { w, p } = tall.left_polar(engine, &[0], &[1]).unwrap();
                    let LeftPolar { w: ww, p: wp } =
                        wide_tall.left_polar(engine, &[0], &[1]).unwrap();
                    assert_reconstruction!(
                        engine,
                        format!("{name}: left_polar"),
                        &w.compose(engine, &p).unwrap(),
                        &tall,
                        terms,
                        $narrow
                    );
                    assert_payloads_agree_scaled(
                        &format!("{name}: left_polar W against the widened oracle"),
                        w.dense_data().unwrap(),
                        ww.dense_data().unwrap(),
                        terms,
                        kappa,
                    );
                    assert_payloads_agree_scaled(
                        &format!("{name}: left_polar P against the widened oracle"),
                        p.dense_data().unwrap(),
                        wp.dense_data().unwrap(),
                        terms,
                        kappa,
                    );

                    let RightPolar { p, wh: w } = short.right_polar(engine, &[0], &[1]).unwrap();
                    assert_reconstruction!(
                        engine,
                        format!("{name}: right_polar"),
                        &p.compose(engine, &w).unwrap(),
                        &short,
                        terms,
                        $narrow
                    );

                    // A lazy-adjoint receiver reads its parent at single
                    // precision too (#1755): the same singular values as its
                    // materialized adjoint, at the payload's precision.
                    let lazy = short.adjoint(engine).unwrap();
                    let lazy_values = lazy.svd_vals(engine, &[0], &[1]).unwrap();
                    let eager_values = lazy
                        .materialize(engine)
                        .unwrap()
                        .svd_vals(engine, &[0], &[1])
                        .unwrap();
                    assert_eq!(lazy_values.len(), eager_values.len(), "{name}");
                    for (a, b) in lazy_values.iter().zip(&eager_values) {
                        assert_eq!(a.sector, b.sector, "{name}");
                        for (x, y) in a.values.iter().zip(&b.values) {
                            assert!((x - y).abs() <= 1.0e-4 * y.abs().max(1.0), "{name}");
                        }
                    }
                }
            }
        };
    }

    checked_generic_suite!(f32_payload, f32, f64);
    checked_generic_suite!(complex32_payload, Complex32, Complex64);
}

/// A compact-diagonal receiver: the factorizations of a spectrum factor, and
/// `find_truncated` fed from a single-precision `diagview`.
///
/// `GradedSpace::find_truncated` stays on the base marker and takes any
/// `SpectrumMagnitude`, so admitting `f32`/`Complex32` there is what makes
/// `t.diagview()` at single precision usable as a truncation input at all.
mod compact_diagonal {
    use super::*;
    use symtn::typed::SectorSpectrum;

    /// Well-separated, exactly `f32`-representable values, descending in
    /// magnitude within each sector.
    fn spectra<D: single_precision_oracle::Parts>() -> [SectorSpectrum<U1Irrep, D>; 3] {
        [
            (U1Irrep::new(-1), vec![(8.0f32, 0.0f32), (2.0, 0.0)]),
            (U1Irrep::new(0), vec![(4.0f32, 0.0), (1.0, 0.0), (0.5, 0.0)]),
            (U1Irrep::new(1), vec![(3.0f32, 0.0), (0.25, 0.0)]),
        ]
        .map(|(sector, values)| SectorSpectrum {
            sector,
            values: values
                .into_iter()
                .map(|(re, im)| D::parts(re, im))
                .collect(),
        })
    }

    macro_rules! compact_suite {
        ($suite:ident, $narrow:ty, $wide:ty) => {
            mod $suite {
                use super::*;

                #[test]
                fn compact_receivers_match_the_widened_oracle() {
                    let engine = &mut symtn_dense::CpuEngine::serial();
                    let rt = runtime();
                    let leg = u1_leg_with([2, 3, 2]);
                    let narrow: TensorMap<U1FusionRule, $narrow> =
                        TensorMap::diagonal(engine, &rt, &leg, spectra()).unwrap();
                    let wide: TensorMap<U1FusionRule, $wide> =
                        TensorMap::diagonal(engine, &rt, &leg, spectra()).unwrap();
                    let terms = 7;

                    assert_spectra_agree!(
                        "compact svd_vals",
                        &narrow.svd_vals(engine, &[0], &[1]).unwrap(),
                        &wide.svd_vals(engine, &[0], &[1]).unwrap(),
                        terms
                    );
                    assert_spectra_agree!(
                        "compact eigh_vals",
                        &narrow
                            .eigh_vals(engine, &[0], &[1], HermitianTol::DEFAULT)
                            .unwrap(),
                        &wide
                            .eigh_vals(engine, &[0], &[1], HermitianTol::DEFAULT)
                            .unwrap(),
                        terms
                    );
                    assert!(
                        is_posdef_compact!(engine, narrow, PREDICATE_TOL, |v: $narrow| {
                            numerics::Numeric::wide(v).re
                        }),
                        "a compact factor with positive values is positive definite"
                    );
                    assert_eq!(
                        is_posdef_compact!(engine, narrow, PREDICATE_TOL, |v: $narrow| {
                            numerics::Numeric::wide(v).re
                        }),
                        is_posdef_compact!(engine, wide, PREDICATE_TOL, |v: $wide| {
                            numerics::Numeric::wide(v).re
                        })
                    );

                    // `find_truncated` fed from the single-precision
                    // `diagview`: the path that needs `SpectrumMagnitude` for
                    // `f32` / `Complex32`.
                    let narrow_view = narrow.diagview(engine).unwrap();
                    let wide_view = wide.diagview(engine).unwrap();
                    for policy in [
                        Truncation::rank(3),
                        Truncation::relative_cutoff(0.2).unwrap(),
                        Truncation::relative_error(0.1).unwrap(),
                    ] {
                        let got = leg.find_truncated(&narrow_view, &policy).unwrap();
                        let expected = leg.find_truncated(&wide_view, &policy).unwrap();
                        assert_eq!(
                            got.selection.subspace().sectors().unwrap(),
                            expected.selection.subspace().sectors().unwrap(),
                            "find_truncated kept a different subspace than the widened oracle \
                             under {policy:?}"
                        );
                        assert_scalars_agree(
                            &format!("find_truncated error under {policy:?}"),
                            Complex64::new(got.error, 0.0),
                            Complex64::new(expected.error, 0.0),
                            terms,
                        );
                    }
                }
            }
        };
    }

    compact_suite!(f32_payload, f32, f64);
    compact_suite!(complex32_payload, Complex32, Complex64);
}

/// Exact widening makes `find_truncated` dtype-independent on a spectrum both
/// payloads can hold exactly.
///
/// This is the oracle the rest of this file rests on, stated as a test: when a
/// spectrum is exactly `f32`-representable, the single- and double-precision
/// `diagview`s widen to bit-identical `f64` inputs, the decision arithmetic is
/// `f64` at both dtypes, and so every policy must return the *same* selection
/// and the same `error` bits. Nothing about the payload dtype can enter here —
/// which is why a genuine cross-dtype difference needs a spectrum the narrow
/// payload cannot represent, the subject of the next test.
#[test]
fn find_truncated_is_dtype_independent_on_an_exactly_representable_spectrum() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    use symtn::typed::SectorSpectrum;

    let rt = runtime();
    let leg = u1_leg_with([2, 3, 2]);
    let values: [(U1Irrep, &[f32]); 3] = [
        (U1Irrep::new(-1), &[8.0, 2.0]),
        (U1Irrep::new(0), &[4.0, 1.0, 0.5]),
        (U1Irrep::new(1), &[3.0, 0.25]),
    ];
    let narrow: TensorMap<U1FusionRule, f32> = TensorMap::diagonal(
        engine,
        &rt,
        &leg,
        values.map(|(sector, v)| SectorSpectrum {
            sector,
            values: v.to_vec(),
        }),
    )
    .unwrap();
    let wide: TensorMap<U1FusionRule, f64> = TensorMap::diagonal(
        engine,
        &rt,
        &leg,
        values.map(|(sector, v)| SectorSpectrum {
            sector,
            values: v.iter().copied().map(f64::from).collect(),
        }),
    )
    .unwrap();

    let narrow_view = narrow.diagview(engine).unwrap();
    let wide_view = wide.diagview(engine).unwrap();
    for policy in [
        Truncation::rank(3),
        Truncation::relative_cutoff(0.2).unwrap(),
        Truncation::relative_error(0.1).unwrap(),
        Truncation::relative_inf_cutoff(0.4).unwrap(),
    ] {
        let got = leg.find_truncated(&narrow_view, &policy).unwrap();
        let expected = leg.find_truncated(&wide_view, &policy).unwrap();
        assert_eq!(
            got.selection.subspace(),
            expected.selection.subspace(),
            "{policy:?}: exact widening must give the same selection"
        );
        assert_eq!(
            got.error.to_bits(),
            expected.error.to_bits(),
            "{policy:?}: exact widening must give the same error bits \
             ({} vs {})",
            got.error,
            expected.error
        );
    }
}

/// The documented near-tie case: a difference the `f32` payload cannot see.
///
/// Two coupled blocks whose largest singular values differ by a relative
/// `2^-27` — five times finer than `f32::EPSILON`. The `f64` twin resolves the
/// difference and its `rank(1)` budget keeps the larger one; the `f32` twin
/// cannot hold it at all (`4 * (1 + 2^-27)` rounds to exactly `4` in `f32`), so
/// the two candidates are bit-identical to it and the documented positional
/// tie-break picks the other sector. The kept sector therefore *does* differ
/// from the double-precision run of the same physics, deterministically.
///
/// What is asserted is the contract [`FactorizationScalar`] states: the kept
/// *count* is the budget either way, and the discarded weight agrees to the
/// accuracy of the values themselves — a rank budget at a tie swaps two
/// interchangeable states. Which sector wins is reported, not asserted, except
/// for the one assertion that makes this test non-vacuous: the two runs really
/// do disagree about it.
#[test]
fn a_rank_tie_the_single_precision_payload_cannot_resolve_keeps_the_other_sector() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rt = runtime();
    let leg = u1_leg_with([2, 2, 1]);
    let tie = 4.0f64;
    let nudged = tie * (1.0 + f64::powi(2.0, -27));
    assert_eq!(
        nudged as f32, tie as f32,
        "the nudge must vanish in f32 for this fixture to mean anything"
    );
    assert_ne!(nudged, tie, "and must survive in f64");

    // One 2x2 block per coupled sector: sector 0 carries `diag(4, 1)`, sector
    // -1 carries `diag(4 * (1 + 2^-27), 1)`.
    let entry = |block_sector: i32, index: &[usize]| -> f64 {
        if index[0] != index[1] {
            return 0.0;
        }
        let top = if block_sector == 0 { tie } else { nudged };
        if index[0] == 0 {
            top
        } else {
            1.0
        }
    };
    let wide: TensorMap<U1FusionRule, f64> =
        TensorMap::from_subblock_fn(engine, &rt, [&leg], [&leg], |trees, index| {
            entry(trees.coupled().charge(), index)
        })
        .unwrap();
    let narrow: TensorMap<U1FusionRule, f32> =
        TensorMap::from_subblock_fn(engine, &rt, [&leg], [&leg], |trees, index| {
            entry(trees.coupled().charge(), index) as f32
        })
        .unwrap();

    let policy = Truncation::rank(1);
    let got = svd_trunc!(engine, narrow, &policy);
    let expected = svd_trunc!(engine, wide, &policy);

    let kept = |t: &symtn::typed::GradedSpace<U1FusionRule>| -> usize {
        t.sectors()
            .unwrap()
            .iter()
            .map(|sector| t.degeneracy(sector).unwrap())
            .sum()
    };
    assert_eq!(
        kept(&got.s.domain()[0]),
        1,
        "the rank budget keeps exactly one state"
    );
    assert_eq!(
        kept(&got.s.domain()[0]),
        kept(&expected.s.domain()[0]),
        "how many states survive is a property of the budget, not of the dtype"
    );

    let got_sectors = got.s.domain()[0].sectors().unwrap();
    let expected_sectors = expected.s.domain()[0].sectors().unwrap();
    eprintln!("near tie: f32 kept {got_sectors:?}, f64 kept {expected_sectors:?}");
    assert_ne!(
        got_sectors, expected_sectors,
        "the fixture is built so the two runs disagree about which of the two \
         interchangeable states survives; if they now agree the fixture has \
         stopped exercising the documented near-tie behaviour"
    );

    // And the invariant that does hold at a rank tie: the discarded weight.
    // The two candidates differ by a relative 2^-27, so swapping them moves the
    // error by far less than the module tolerance.
    assert_scalars_agree(
        "near-tie truncation error",
        Complex64::new(got.error, 0.0),
        Complex64::new(expected.error, 0.0),
        4,
    );
}

/// Double-precision truncation decisions, pinned exactly, and their errors
/// against hand values.
///
/// The fixtures are non-dyadic (multiples of a tenth) and two of them sit
/// within a rounding of the budget, which is where the rounding slack in
/// `symtn-matrixalgebra/src/truncation.rs` decides (budget-relative since
/// #1333). The kept counts are combinatorial and compared exactly; the error
/// is `sqrt(sum_discarded v^2)` (every `dim(c) == 1`), compared under the
/// workspace tolerance rule with the hand value of the discarded tail.
#[test]
fn double_precision_truncation_decisions_and_errors_match_hand_values() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    use symtn::typed::SectorSpectrum;

    let rt = runtime();
    let leg = u1_leg_with([3, 3, 3]);
    // Non-dyadic, descending, and with a tail whose weights land close to the
    // relative-error budgets below.
    let spectra = || {
        [
            (U1Irrep::new(-1), [0.9f64, 0.3, 0.1]),
            (U1Irrep::new(0), [0.7f64, 0.2, 0.1]),
            (U1Irrep::new(1), [0.5f64, 0.3, 0.1]),
        ]
        .map(|(sector, values)| SectorSpectrum {
            sector,
            values: values.to_vec(),
        })
    };
    let tensor: TensorMap<U1FusionRule, f64> =
        TensorMap::diagonal(engine, &rt, &leg, spectra()).unwrap();
    let view = tensor.diagview(engine).unwrap();

    // Hand errors: the discarded tail of the 9-value spectrum
    // {0.9, 0.3, 0.1 | 0.7, 0.2, 0.1 | 0.5, 0.3, 0.1}, whose squares sum to 1.8.
    // Keeping 4 discards {0.3, 0.2, 0.1, 0.1, 0.1} (0.16); keeping 7 discards
    // two 0.1s (0.02); keeping 3 discards 0.25 of the weight; keeping 8 one
    // 0.1; keeping 5 {0.2, 0.1, 0.1, 0.1} (0.07).
    let cases: [(Truncation, usize, f64); 8] = [
        (Truncation::rank(4), 4, 0.4),
        (Truncation::rank(7), 7, 0.02f64.sqrt()),
        (Truncation::relative_cutoff(0.25).unwrap(), 3, 0.5),
        (Truncation::relative_error(0.1).unwrap(), 8, 0.1),
        (Truncation::relative_error(0.2).unwrap(), 5, 0.07f64.sqrt()),
        (Truncation::relative_error(0.3).unwrap(), 4, 0.4),
        // The budget is exactly the weight of the two smallest tails, so the
        // rounding slack is what decides: `0.1 * 0.1` twice sums to
        // `0.020000000000000004`, four ulps above the budget, and only the
        // budget-relative `(n + 5) * eps` slack (#1333) lets the second tail
        // go. Without it this case keeps 8.
        (
            Truncation::relative_error((1.0f64 / 90.0).sqrt()).unwrap(),
            7,
            0.02f64.sqrt(),
        ),
        // A composite, so the intersection path is pinned too.
        (
            Truncation::rank(4).and(Truncation::relative_error(0.2).unwrap()),
            4,
            0.4,
        ),
    ];
    for (policy, kept, error) in cases {
        let decision = leg.find_truncated(&view, &policy).unwrap();
        let subspace = decision.selection.subspace();
        let kept_total: usize = subspace
            .sectors()
            .unwrap()
            .iter()
            .map(|sector| subspace.degeneracy(sector).unwrap())
            .sum();
        assert_eq!(
            kept_total, kept,
            "{policy:?} kept {kept_total} states, the pin says {kept}"
        );
        numerics::assert_close(
            &format!("{policy:?} truncation error"),
            decision.error,
            error,
            9,
        );
    }
}
