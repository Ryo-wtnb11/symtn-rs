//! Advanced linear-algebra oracle for the single-precision payloads.
//!
//! Every operation `AdvancedLinalgScalar` admits — `inv`, `solve` (also in
//! its adjointed right-solve composition), `pinv`, `exp` (Hermitian spectral
//! route and non-Hermitian Padé route), integer powers composed from `compose`
//! and `inv`, `map_diagonal(sqrt)`, `eig_vals`, `eig_full` and its truncation — is
//! compared against the `f64`/`Complex64` result of the same operation on the
//! exactly widened input (`single_precision_oracle`).
//!
//! Tolerance: `K * sqrt(n) * eps(f32) * max(1, scale) * kappa`, `K = 32`,
//! `n` the payload length. `kappa` multiplies the bound only for a forward
//! error and is **measured** on the double-precision twin:
//!
//! | quantity | `kappa` |
//! | --- | --- |
//! | `inv`, `solve`, right solve, `pinv`, negative powers | `sigma_max / sigma_min` of the divisor |
//! | `exp`, positive powers, `map_diagonal(sqrt)`, `eig_full` residual | 1 (backward stable / elementwise) |
//! | eigenvalues | `sigma_max / sigma_min` of the double-precision eigenvector factor (Bauer–Fike) |

use crate::single_precision_oracle;

use std::sync::Arc;

use num_complex::{Complex32, Complex64};
use symtn::sector::{SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep};
use symtn::typed::{Eig, Engine, Svd};
use symtn::typed::{GradedSpace, SectorSpectrum, TensorMap, Truncation};

use single_precision_oracle::{
    assert_payloads_agree_scaled, assert_scalars_agree_scaled, fermion_su2_leg_with, minus_one,
    one, runtime, tolerance, u1_leg_with, Parts,
};

/// Hermitian, well separated: diagonal `8, 4, 2, 1` with a small constant
/// off-diagonal. `exp` takes the spectral route on it.
fn hermitian_entry(row: usize, col: usize) -> (f32, f32) {
    const DIAGONAL: [f32; 4] = [8.0, 4.0, 2.0, 1.0];
    if row == col {
        (DIAGONAL[row % 4], 0.0)
    } else if row < col {
        (0.25, 0.125)
    } else {
        (0.25, -0.125)
    }
}

/// Non-Hermitian with distinct eigenvalues: `exp` takes the Padé route and
/// `eig_full` has a well-conditioned eigenbasis.
fn nonnormal_entry(row: usize, col: usize) -> (f32, f32) {
    const DIAGONAL: [f32; 4] = [1.0, 0.5, -0.5, 0.25];
    if row == col {
        (DIAGONAL[row % 4], 0.0)
    } else if row < col {
        (0.25, 0.125)
    } else {
        (-0.125, 0.0625)
    }
}

fn su2_leg_with(degeneracies: [usize; 3]) -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), degeneracies[0]),
            (SU2Irrep::from_twice_spin(1), degeneracies[1]),
            (SU2Irrep::from_twice_spin(2), degeneracies[2]),
        ],
    )
    .unwrap()
}

/// `sigma_max / sigma_min` over every coupled block of a double-precision
/// tensor, asserted full rank and below `1e4` so the fixture is a usable
/// single-precision oracle.
macro_rules! measured_kappa {
    ($engine:ident, $tensor:expr) => {{
        let mut largest = 0.0f64;
        let mut smallest = f64::INFINITY;
        for entry in &$tensor.svd_vals($engine, &[0], &[1]).unwrap() {
            for &value in &entry.values {
                largest = largest.max(value);
                smallest = smallest.min(value);
            }
        }
        let kappa = largest / smallest;
        assert!(
            smallest > 0.0 && kappa < 1e4,
            "fixture conditioning {kappa:e} is unusable as a single-precision oracle"
        );
        kappa
    }};
}

/// `‖actual - expected‖ <= tolerance(terms, ‖expected‖)`.
macro_rules! assert_residual {
    ($engine:ident, $what:expr, $actual:expr, $expected:expr, $terms:expr, $d:ty) => {{
        let expected = $expected;
        let residual = $actual
            .axpby($engine, one::<$d>(), expected, minus_one::<$d>())
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

/// Complex spectra sector by sector, each value within the bound scaled by
/// `kappa`.
macro_rules! assert_complex_spectra_agree {
    ($what:expr, $narrow:expr, $wide:expr, $terms:expr, $kappa:expr) => {{
        let (narrow, wide) = ($narrow, $wide);
        assert_eq!(narrow.len(), wide.len(), "{}: sector count", $what);
        for (got, expected) in narrow.iter().zip(wide.iter()) {
            assert_eq!(got.sector, expected.sector, "{}: sector order", $what);
            assert_eq!(got.values.len(), expected.values.len(), "{}", $what);
            for (index, (value, oracle)) in got.values.iter().zip(&expected.values).enumerate() {
                assert_scalars_agree_scaled(
                    &format!("{}: sector {:?} value {index}", $what, got.sector),
                    Parts::wide(*value),
                    Parts::wide(*oracle),
                    $terms,
                    $kappa,
                );
            }
        }
    }};
}

/// The whole admitted family over one provider and one payload dtype.
///
/// `$eig` is `D::Eig` and `$to_eig` embeds the narrow input in it; the typed
/// `let` bindings are the static proof that single-precision eig returns
/// `Complex32`, not a widened `Complex64`.
macro_rules! advanced_checks {
    (
        $engine:ident,
        $name:expr, $narrow:ty, $wide:ty, $eig:ty, $to_eig:expr,
        $square:expr, $tall:expr, $min_blocks:expr, $multiplicity_free:expr
    ) => {{
        let engine = &mut *$engine;
        let rt = runtime();
        let name: &str = $name;
        let square = $square;
        let tall = $tall;
        let (h, wide_h) = twin_with!(
            engine,
            &rt,
            $narrow,
            $wide,
            [&square],
            [&square],
            |i: &[usize]| { hermitian_entry(i[0], i[1]) }
        );
        let (a, wide_a) = twin_with!(
            engine,
            &rt,
            $narrow,
            $wide,
            [&square],
            [&square],
            |i: &[usize]| { nonnormal_entry(i[0], i[1]) }
        );
        let (b, wide_b) = twin_with!(
            engine,
            &rt,
            $narrow,
            $wide,
            [&square],
            [&square],
            |i: &[usize]| { nonnormal_entry(i[1], i[0]) }
        );
        let (t, wide_t) = twin_with!(
            engine,
            &rt,
            $narrow,
            $wide,
            [&tall],
            [&square],
            |i: &[usize]| { hermitian_entry(i[0], i[1]) }
        );
        let n = wide_h.dense_data().unwrap().len();
        assert!(
            h.subblock_count() >= $min_blocks,
            "{name}: the fixture must carry at least {} coupled blocks",
            $min_blocks
        );
        let kappa_h = measured_kappa!(engine, &wide_h);
        let kappa_a = measured_kappa!(engine, &wide_a);
        let kappa_t = measured_kappa!(engine, &wide_t);

        // ---- inv / solve / right solve / pinv: forward errors. -------------
        assert_payloads_agree_scaled(
            &format!("{name}: inv (kappa {kappa_a:e})"),
            a.inv(engine, &[0], &[1]).unwrap().dense_data().unwrap(),
            wide_a
                .inv(engine, &[0], &[1])
                .unwrap()
                .dense_data()
                .unwrap(),
            n,
            kappa_a,
        );
        assert_payloads_agree_scaled(
            &format!("{name}: solve (kappa {kappa_a:e})"),
            a.solve(engine, &[0], &[1], &b, &[0], &[1])
                .unwrap()
                .dense_data()
                .unwrap(),
            wide_a
                .solve(engine, &[0], &[1], &wide_b, &[0], &[1])
                .unwrap()
                .dense_data()
                .unwrap(),
            n,
            kappa_a,
        );
        let b_adjoint = b.adjoint(engine).unwrap();
        let wide_b_adjoint = wide_b.adjoint(engine).unwrap();
        assert_payloads_agree_scaled(
            &format!("{name}: right solve (kappa {kappa_a:e})"),
            a.adjoint(engine)
                .unwrap()
                .solve(engine, &[0], &[1], &b_adjoint, &[0], &[1])
                .unwrap()
                .adjoint(engine)
                .unwrap()
                .materialize(engine)
                .unwrap()
                .dense_data()
                .unwrap(),
            wide_a
                .adjoint(engine)
                .unwrap()
                .solve(engine, &[0], &[1], &wide_b_adjoint, &[0], &[1])
                .unwrap()
                .adjoint(engine)
                .unwrap()
                .materialize(engine)
                .unwrap()
                .dense_data()
                .unwrap(),
            n,
            kappa_a,
        );
        assert_payloads_agree_scaled(
            &format!("{name}: pinv (kappa {kappa_t:e})"),
            t.pinv(engine, &[0], &[1], 1e-4)
                .unwrap()
                .dense_data()
                .unwrap(),
            wide_t
                .pinv(engine, &[0], &[1], 1e-4)
                .unwrap()
                .dense_data()
                .unwrap(),
            wide_t.dense_data().unwrap().len(),
            kappa_t,
        );

        // ---- exp: spectral (Hermitian) and Padé (non-Hermitian) routes. ----
        assert_payloads_agree_scaled(
            &format!("{name}: exp, Hermitian route"),
            h.exp(engine, &[0], &[1]).unwrap().dense_data().unwrap(),
            wide_h
                .exp(engine, &[0], &[1])
                .unwrap()
                .dense_data()
                .unwrap(),
            n,
            1.0,
        );
        assert_payloads_agree_scaled(
            &format!("{name}: exp, Padé route"),
            a.exp(engine, &[0], &[1]).unwrap().dense_data().unwrap(),
            wide_a
                .exp(engine, &[0], &[1])
                .unwrap()
                .dense_data()
                .unwrap(),
            n,
            1.0,
        );

        // ---- Integer powers: repeated composition; a negative power inverts once.
        macro_rules! cube {
            ($t:expr) => {{
                let square = $t.compose(engine, &$t).unwrap();
                $t.compose(engine, &square).unwrap()
            }};
        }
        macro_rules! inverse_square {
            ($t:expr) => {{
                let inverse = $t.inv(engine, &[0], &[1]).unwrap();
                inverse.compose(engine, &inverse).unwrap()
            }};
        }
        assert_payloads_agree_scaled(
            &format!("{name}: a ∘ a ∘ a"),
            cube!(a).dense_data().unwrap(),
            cube!(wide_a).dense_data().unwrap(),
            3 * n,
            1.0,
        );
        assert_payloads_agree_scaled(
            &format!("{name}: h⁻¹ ∘ h⁻¹ (kappa {kappa_h:e})"),
            inverse_square!(h).dense_data().unwrap(),
            inverse_square!(wide_h).dense_data().unwrap(),
            2 * n,
            kappa_h * kappa_h,
        );

        // ---- Elementwise sqrt of a compact spectrum. ------------------------
        // Compact SVD stores `s` diagonally for every Host provider. Checked
        // Generic reductions still need an explicit dense payload.
        let Svd { s, .. } = h.svd_compact(engine, &[0], &[1]).unwrap();
        let Svd { s: wide_s, .. } = wide_h.svd_compact(engine, &[0], &[1]).unwrap();
        let root = s.map_diagonal(engine, |value| value.sqrt()).unwrap();
        assert_payloads_agree_scaled(
            &format!("{name}: sqrt of a compact spectrum"),
            root.materialize(engine).unwrap().dense_data().unwrap(),
            wide_s
                .map_diagonal(engine, |value| value.sqrt())
                .unwrap()
                .materialize(engine)
                .unwrap()
                .dense_data()
                .unwrap(),
            n,
            1.0,
        );
        assert_residual!(
            engine,
            format!("{name}: sqrt ∘ sqrt"),
            root.compose(engine, &root).unwrap(),
            &s.materialize(engine).unwrap(),
            n,
            $narrow
        );
        // ---- General eig: factors in `D::Eig`, spectra `Complex64`. --------
        let Eig { d, v }: Eig<TensorMap<_, $eig>> = a.eig_full(engine, &[0], &[1]).unwrap();
        let Eig { v: wide_v, .. } = wide_a.eig_full(engine, &[0], &[1]).unwrap();
        let kappa_v = measured_kappa!(engine, &wide_v);
        let to_eig = $to_eig;
        let a_eig: TensorMap<_, $eig> = to_eig(&mut *engine, &a);
        let vd = v.compose(engine, &d).unwrap();
        assert_residual!(
            engine,
            format!("{name}: eig_full a∘v == v∘d"),
            a_eig.compose(engine, &v).unwrap(),
            &vd,
            n,
            $eig
        );
        let wide_values = wide_a.eig_vals(engine, &[0], &[1]).unwrap();
        let values: Vec<SectorSpectrum<_, Complex64>> = a.eig_vals(engine, &[0], &[1]).unwrap();
        assert_complex_spectra_agree!(
            format!("{name}: eig_vals (kappa(V) {kappa_v:e})"),
            &values,
            &wide_values,
            n,
            kappa_v
        );
        assert_complex_spectra_agree!(
            format!("{name}: eig_full d (kappa(V) {kappa_v:e})"),
            &{
                let mut spectra = d.diagview(engine).unwrap();
                spectra.sort_by(|left, right| left.sector.cmp(&right.sector));
                spectra
            },
            &wide_values,
            n,
            kappa_v
        );
        // The truncated eigendecomposition is `eig_full` → `diagview` →
        // `find_truncated` → two-leg `restrict_leg`.
        let (kept, error) = {
            let found = d.domain()[0]
                .find_truncated(&d.diagview(engine).unwrap(), &Truncation::rank(2))
                .unwrap();
            (
                d.restrict_leg(engine, &[(0, &found.selection), (1, &found.selection)])
                    .unwrap(),
                found.error,
            )
        };
        let (wide_kept, wide_error) = {
            let Eig { d: wide_d, .. } = wide_a.eig_full(engine, &[0], &[1]).unwrap();
            let found = wide_d.domain()[0]
                .find_truncated(&wide_d.diagview(engine).unwrap(), &Truncation::rank(2))
                .unwrap();
            (
                wide_d
                    .restrict_leg(engine, &[(0, &found.selection), (1, &found.selection)])
                    .unwrap(),
                found.error,
            )
        };
        assert_eq!(
            kept.materialize(engine)
                .unwrap()
                .dense_data()
                .unwrap()
                .len(),
            wide_kept
                .materialize(engine)
                .unwrap()
                .dense_data()
                .unwrap()
                .len(),
            "{name}: truncated eig kept a different number of states than the widened oracle"
        );
        assert!(error > 0.0, "{name}: truncated eig discarded nothing");
        assert_scalars_agree_scaled(
            &format!("{name}: truncated eig error"),
            Complex64::new(error, 0.0),
            Complex64::new(wide_error, 0.0),
            n,
            kappa_v,
        );

        if $multiplicity_free {
            // Lazy-adjoint receivers are materialized for the call.
            let lazy = a.adjoint(engine).unwrap();
            assert_payloads_agree_scaled(
                &format!("{name}: inv of a lazy adjoint"),
                lazy.inv(engine, &[0], &[1]).unwrap().dense_data().unwrap(),
                wide_a
                    .adjoint(engine)
                    .unwrap()
                    .inv(engine, &[0], &[1])
                    .unwrap()
                    .dense_data()
                    .unwrap(),
                n,
                kappa_a,
            );
            assert_payloads_agree_scaled(
                &format!("{name}: exp of a lazy adjoint"),
                lazy.exp(engine, &[0], &[1]).unwrap().dense_data().unwrap(),
                wide_a
                    .adjoint(engine)
                    .unwrap()
                    .exp(engine, &[0], &[1])
                    .unwrap()
                    .dense_data()
                    .unwrap(),
                n,
                1.0,
            );
        }
    }};
}

macro_rules! advanced_suite {
    ($suite:ident, $narrow:ty, $wide:ty, $eig:ty, $to_eig:expr) => {
        mod $suite {
            use super::*;

            #[test]
            fn u1_advanced_linalg_matches_the_widened_oracle() {
                let engine = &mut symtn::typed::CpuEngine::serial();
                advanced_checks!(
                    engine,
                    concat!("U(1) ", stringify!($narrow)),
                    $narrow,
                    $wide,
                    $eig,
                    $to_eig,
                    u1_leg_with([2, 3, 2]),
                    u1_leg_with([3, 4, 3]),
                    3,
                    true
                );
            }

            #[test]
            fn su2_advanced_linalg_matches_the_widened_oracle() {
                let engine = &mut symtn::typed::CpuEngine::serial();
                advanced_checks!(
                    engine,
                    concat!("SU(2) ", stringify!($narrow)),
                    $narrow,
                    $wide,
                    $eig,
                    $to_eig,
                    su2_leg_with([2, 3, 2]),
                    su2_leg_with([3, 4, 3]),
                    3,
                    true
                );
            }

            #[test]
            fn fermion_su2_advanced_linalg_matches_the_widened_oracle() {
                let engine = &mut symtn::typed::CpuEngine::serial();
                advanced_checks!(
                    engine,
                    concat!("fZ2 x U(1) x SU(2) ", stringify!($narrow)),
                    $narrow,
                    $wide,
                    $eig,
                    $to_eig,
                    fermion_su2_leg_with([2, 2, 1]),
                    fermion_su2_leg_with([3, 3, 2]),
                    3,
                    true
                );
            }
        }
    };
}

advanced_suite!(
    f32_payload,
    f32,
    f64,
    Complex32,
    |engine: &mut dyn Engine, t: &TensorMap<_, f32>| t.convert::<Complex32>(engine).unwrap()
);
advanced_suite!(
    complex32_payload,
    Complex32,
    Complex64,
    Complex32,
    |_: &mut dyn Engine, t: &TensorMap<_, Complex32>| t.clone()
);

/// The reciprocal of a compact `Complex32` entry whose squared modulus
/// underflows or overflows `f32` while the reciprocal itself is
/// representable: `2^-80`, `2^-80 i` and `2^100` invert exactly.
#[test]
fn compact_complex32_reciprocal_does_not_underflow() {
    let engine = &mut symtn_dense::CpuEngine::serial();
    let rt = runtime();
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 3)]).unwrap();
    let tiny = 2.0f32.powi(-80);
    let diagonal: TensorMap<_, Complex32> = TensorMap::diagonal(
        engine,
        &rt,
        &leg,
        [SectorSpectrum {
            sector: U1Irrep::new(0),
            values: vec![
                Complex32::new(tiny, 0.0),
                Complex32::new(0.0, tiny),
                Complex32::new(2.0f32.powi(100), 0.0),
            ],
        }],
    )
    .unwrap();
    let expected = [
        Complex32::new(2.0f32.powi(80), 0.0),
        Complex32::new(0.0, -(2.0f32.powi(80))),
        Complex32::new(2.0f32.powi(-100), 0.0),
    ];
    let inverse = diagonal.inv(engine, &[0], &[1]).unwrap();
    assert_eq!(inverse.diagview(engine).unwrap()[0].values, expected, "inv");
    let pseudo = diagonal.pinv(engine, &[0], &[1], 0.0).unwrap();
    assert_eq!(pseudo.diagview(engine).unwrap()[0].values, expected, "pinv");
}

/// Checked-Generic provider: SU(3), outer multiplicity, the checked dispatch.
#[cfg(feature = "cgc-gen")]
mod checked_generic {
    use super::*;
    use symtn::sector::LieGroupFusionRule;

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
        ($suite:ident, $narrow:ty, $wide:ty, $eig:ty, $to_eig:expr) => {
            mod $suite {
                use super::*;

                #[test]
                fn su3_advanced_linalg_matches_the_widened_oracle() {
                    let engine = &mut symtn::typed::CpuEngine::serial();
                    advanced_checks!(
                        engine,
                        concat!("SU(3) ", stringify!($narrow)),
                        $narrow,
                        $wide,
                        $eig,
                        $to_eig,
                        su3_leg(2),
                        su3_leg(3),
                        1,
                        false
                    );
                }
            }
        };
    }

    checked_generic_suite!(
        f32_payload,
        f32,
        f64,
        Complex32,
        |engine: &mut dyn Engine, t: &TensorMap<_, f32>| t.convert::<Complex32>(engine).unwrap()
    );
    checked_generic_suite!(
        complex32_payload,
        Complex32,
        Complex64,
        Complex32,
        |_: &mut dyn Engine, t: &TensorMap<_, Complex32>| t.clone()
    );
}
