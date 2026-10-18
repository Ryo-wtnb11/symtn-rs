//! `inner`, `norm(p)`, `tr` and `axpby` over compact diagonal, dense and lazy
//! operands take one body in both admission modes (#1867, #1868).
//!
//! TensorKit `cfaa073e` reduces a `DiagonalTensorMap` through the same
//! `blocks(t)` loops as any tensor: `LinearAlgebra.norm`/`_norm`
//! (`src/tensors/linalg.jl:257-282`, any `p`), `LinearAlgebra.tr`
//! (`linalg.jl:319`) and `VectorInterface.inner` (`src/tensors/vectorinterface.jl:114`),
//! whose per-block `Diagonal` methods read only the stored diagonal. The style
//! of the provider never enters. So SU(2) admitted multiplicity-free
//! (`SU2FusionRule`) and checked Generic (`SUNFusionRule::new(2)`) must give
//! the same answers, and both must match the hand oracle below:
//!
//! ```text
//! norm(t, p)  = (Σ_c dim(c) Σ_ij |t_c[i,j]|^p)^(1/p),  norm(t, Inf) = max |t_c[i,j]|
//! tr(t)       = Σ_c dim(c) Σ_i t_c[i,i]
//! inner(a, b) = Σ_c dim(c) Σ_ij conj(a_c[i,j]) b_c[i,j]
//! ```
//!
//! with `dim(j) = 2j + 1` and every entry written out from the tables here,
//! never read back from a symtn result.

#![cfg(feature = "cgc-gen")]

use std::sync::Arc;

use symtn::sector::{SU2FusionRule, SU2Irrep, SUNFusionRule};
use symtn::typed::{Complex32, Complex64, GradedSpace, SectorSpectrum, TensorMap};

#[path = "../../tests/support/fixtures.rs"]
mod fixtures;
#[path = "../../tests/support/numerics.rs"]
mod numerics;

use fixtures::host_runtime;

/// Floating terms reaching one reduction: every stored entry of the fixtures
/// (`Σ_c k_c² = 4 + 9 + 4`) plus slack for the weighted cross-sector sum.
const TERMS: usize = 32;

/// `(2j, k)` per coupled sector: nontrivial degeneracies, all distinct.
const SECTORS: [(usize, usize); 3] = [(0, 2), (1, 3), (2, 2)];

const POWERS: [f64; 4] = [1.0, 2.0, 3.0, f64::INFINITY];

/// The stored diagonal of the compact operand `a`.
fn a(twice: usize, i: usize) -> Complex64 {
    let re = [[0.93, -0.41, 0.0], [0.87, 0.52, -0.11], [0.74, 0.29, 0.0]][twice][i];
    Complex64::new(re, 0.2 * re - 0.05 * i as f64 + 0.01 * twice as f64)
}

/// The diagonal of the second operand `b`.
fn b(twice: usize, i: usize) -> Complex64 {
    let re = [[0.3, 1.2, 0.0], [-0.6, 0.25, 0.8], [1.1, -0.45, 0.0]][twice][i];
    Complex64::new(re, -0.3 * re + 0.07 * i as f64)
}

/// Off-diagonal entries of the dense operand, which a compact partner must
/// never read.
fn off(twice: usize, i: usize, j: usize) -> Complex64 {
    Complex64::new(
        0.07 * (i + 1) as f64 - 0.05 * (j + 2) as f64 + 0.01 * twice as f64,
        0.03 * (i + 2 * j + 1) as f64,
    )
}

fn dim(twice: usize) -> f64 {
    (twice + 1) as f64
}

/// Every hand-oracle value the macro below is compared against, in one
/// payload dtype (`conv` drops the imaginary parts for `f64`).
struct Oracle {
    norms_a: [f64; 4],
    tr_a: Complex64,
    inner_ab: Complex64,
    axpby_norm2: f64,
    axpby_tr: Complex64,
}

const ALPHA: f64 = 0.75;
const BETA: f64 = -1.25;

fn oracle(conv: impl Fn(Complex64) -> Complex64) -> Oracle {
    let mut powers = [0.0; 3];
    let mut max = 0.0_f64;
    let (mut tr_a, mut inner_ab, mut axpby_tr) = (
        Complex64::default(),
        Complex64::default(),
        Complex64::default(),
    );
    let mut axpby_norm2 = 0.0;
    for (twice, k) in SECTORS {
        let d = dim(twice);
        for i in 0..k {
            let (ai, bi) = (conv(a(twice, i)), conv(b(twice, i)));
            for (slot, p) in powers.iter_mut().zip([1.0, 2.0, 3.0]) {
                *slot += d * ai.norm().powf(p);
            }
            max = max.max(ai.norm());
            tr_a += d * ai;
            inner_ab += d * ai.conj() * bi;
            // `ALPHA * a + BETA * L` with `L = P^H`, `diag(L) = b`,
            // `L[i, j] = conj(P[j, i]) = conj(off(j, i))`.
            let sum = ALPHA * ai + BETA * bi;
            axpby_tr += d * sum;
            axpby_norm2 += d * sum.norm_sqr();
            for j in (0..k).filter(|&j| j != i) {
                axpby_norm2 += d * (BETA * conv(off(twice, j, i)).conj()).norm_sqr();
            }
        }
    }
    Oracle {
        norms_a: [powers[0], powers[1].sqrt(), powers[2].cbrt(), max],
        tr_a,
        inner_ab,
        axpby_norm2,
        axpby_tr,
    }
}

/// Runs every reduction over one provider and payload dtype and returns the
/// results in a fixed order: `norm(p)` of the compact, dense-twin and lazy-twin
/// `a`; `tr` of each; `inner` across every compact/dense/lazy pairing; then
/// `axpby(compact, lazy)` and `axpby(lazy, compact)` as their norm and trace.
macro_rules! reductions {
    ($rule:expr, $label:expr, $twice_of:expr, $D:ty, $conv:expr) => {{
        let runtime = host_runtime();
        let label = $label;
        let twice_of = $twice_of;
        let conv = $conv;
        let leg = GradedSpace::try_new(
            Arc::new($rule),
            SECTORS.iter().map(|&(twice, k)| (label(twice), k)),
        )
        .unwrap();
        let spectrum = |entry: fn(usize, usize) -> Complex64| {
            SECTORS
                .iter()
                .map(|&(twice, k)| SectorSpectrum {
                    sector: label(twice),
                    values: (0..k).map(|i| conv(entry(twice, i))).collect::<Vec<$D>>(),
                })
                .collect::<Vec<_>>()
        };
        let compact_a = TensorMap::<_, $D>::diagonal(&runtime, &leg, spectrum(a)).unwrap();
        let compact_b = TensorMap::<_, $D>::diagonal(&runtime, &leg, spectrum(b)).unwrap();
        let dense = |fill: &dyn Fn(usize, usize, usize) -> Complex64| -> TensorMap<_, $D> {
            TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |trees, indices| {
                conv(fill(twice_of(trees.coupled()), indices[0], indices[1]))
            })
            .unwrap()
        };
        let twin_a = dense(&|twice, i, j| {
            if i == j {
                a(twice, i)
            } else {
                Complex64::default()
            }
        });
        let lazy_twin_a = dense(&|twice, i, j| {
            if i == j {
                a(twice, i).conj()
            } else {
                Complex64::default()
            }
        })
        .adjoint()
        .unwrap();
        // `diag(dense_b) = b` with nonzero off-diagonal entries, and a lazy
        // `L = P^H` whose diagonal is `b` and whose parent stores `off`.
        let dense_b = dense(&|twice, i, j| {
            if i == j {
                b(twice, i)
            } else {
                off(twice, i, j)
            }
        });
        let lazy_b = dense(&|twice, i, j| {
            if i == j {
                b(twice, i).conj()
            } else {
                off(twice, i, j)
            }
        })
        .adjoint()
        .unwrap();
        let wide = |value: $D| numerics::Numeric::wide(value);
        let mut out: Vec<(String, Complex64)> = Vec::new();
        for (what, tensor) in [
            ("compact", &compact_a),
            ("dense", &twin_a),
            ("lazy", &lazy_twin_a),
        ] {
            for p in POWERS {
                out.push((format!("{what} norm({p})"), tensor.norm(p).unwrap().into()));
            }
            out.push((format!("{what} tr"), wide(tensor.tr().unwrap())));
        }
        for (what, lhs, rhs, conjugate) in [
            ("compact.inner(compact)", &compact_a, &compact_b, false),
            ("compact.inner(dense)", &compact_a, &dense_b, false),
            ("dense.inner(compact)", &dense_b, &compact_a, true),
            ("compact.inner(lazy)", &compact_a, &lazy_b, false),
            ("lazy.inner(compact)", &lazy_b, &compact_a, true),
        ] {
            let value = wide(lhs.inner(rhs).unwrap());
            out.push((
                what.to_string(),
                if conjugate { value.conj() } else { value },
            ));
        }
        let alpha: $D = conv(Complex64::new(ALPHA, 0.0));
        let beta: $D = conv(Complex64::new(BETA, 0.0));
        for (what, sum) in [
            (
                "axpby(compact, lazy)",
                compact_a.axpby(alpha, &lazy_b, beta).unwrap(),
            ),
            (
                "axpby(lazy, compact)",
                lazy_b.axpby(beta, &compact_a, alpha).unwrap(),
            ),
        ] {
            out.push((
                format!("{what} norm(2)^2"),
                sum.norm(2.0).unwrap().powi(2).into(),
            ));
            out.push((format!("{what} tr"), wide(sum.tr().unwrap())));
        }
        out
    }};
}

fn expected(oracle: &Oracle) -> Vec<Complex64> {
    let mut out = Vec::new();
    for _ in ["compact", "dense", "lazy"] {
        out.extend(oracle.norms_a.map(Complex64::from));
        out.push(oracle.tr_a);
    }
    out.extend([oracle.inner_ab; 5]);
    for _ in 0..2 {
        out.push(oracle.axpby_norm2.into());
        out.push(oracle.axpby_tr);
    }
    out
}

fn assert_both_modes<D: numerics::Numeric>(
    dtype: &str,
    oracle: Oracle,
    mf: Vec<(String, Complex64)>,
    checked: Vec<(String, Complex64)>,
) {
    let want = expected(&oracle);
    assert_eq!(mf.len(), want.len());
    assert_eq!(checked.len(), want.len());
    for (((what, mf), (_, checked)), want) in mf.into_iter().zip(checked).zip(want) {
        for (mode, got) in [("multiplicity-free", mf), ("checked Generic", checked)] {
            numerics::assert_close(&format!("{dtype} {mode} {what}"), got, want, TERMS);
        }
    }
}

fn su2(twice: usize) -> SU2Irrep {
    SU2Irrep::from_twice_spin(twice)
}

fn su2_twice(label: &SU2Irrep) -> usize {
    label.twice_spin()
}

fn sun2(twice: usize) -> Vec<i64> {
    vec![twice as i64]
}

fn sun2_twice(label: &[i64]) -> usize {
    label[0] as usize
}

#[test]
fn compact_dense_and_lazy_reductions_agree_across_modes_f64() {
    // What: one reduction body serves both modes; a checked Generic compact
    // operand reduces and `norm(p != 2)` exists on checked Generic.
    let real = |z: Complex64| z.re;
    let mf = reductions!(SU2FusionRule, su2, su2_twice, f64, real);
    let checked = reductions!(SUNFusionRule::new(2).unwrap(), sun2, sun2_twice, f64, real);
    assert_both_modes::<f64>("f64", oracle(|z| z.re.into()), mf, checked);
}

#[test]
fn compact_dense_and_lazy_reductions_agree_across_modes_c64() {
    let complex = |z: Complex64| z;
    let mf = reductions!(SU2FusionRule, su2, su2_twice, Complex64, complex);
    let checked = reductions!(
        SUNFusionRule::new(2).unwrap(),
        sun2,
        sun2_twice,
        Complex64,
        complex
    );
    assert_both_modes::<Complex64>("c64", oracle(|z| z), mf, checked);
}

/// `svd_compact(t).s` is compact in both modes; its weighted Frobenius norm is
/// `t`'s, and every `p` agrees with a dense twin of `s`.
#[test]
fn svd_compact_spectrum_reduces_on_su3_and_on_su2_in_both_modes() {
    let runtime = host_runtime();
    macro_rules! check {
        ($what:expr, $rule:expr, $sectors:expr) => {{
            let leg = GradedSpace::try_new(Arc::new($rule), $sectors).unwrap();
            let t: TensorMap<_, Complex64> =
                TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
                    Complex64::new(
                        1.0 + 0.5 * indices[0] as f64 - 0.3 * indices[1] as f64,
                        0.25 * (indices[0] * indices[1]) as f64 - 0.1,
                    )
                })
                .unwrap();
            let s = t.svd_compact(&[0], &[1]).unwrap().s;
            assert!(s.dense_data().is_err(), "{}: s is stored compact", $what);
            numerics::assert_close(
                &format!("{} |s| = |t|", $what),
                s.norm(2.0).unwrap(),
                t.norm(2.0).unwrap(),
                TERMS,
            );
            numerics::assert_close(
                &format!("{} <s, s> = |t|^2", $what),
                s.inner(&s).unwrap(),
                Complex64::from(t.norm(2.0).unwrap().powi(2)),
                TERMS,
            );
            let twin = s.materialize().unwrap();
            for p in POWERS {
                numerics::assert_close(
                    &format!("{} norm({p})", $what),
                    s.norm(p).unwrap(),
                    twin.norm(p).unwrap(),
                    TERMS,
                );
            }
            numerics::assert_close(
                &format!("{} tr", $what),
                s.tr().unwrap(),
                twin.tr().unwrap(),
                TERMS,
            );
            [POWERS.map(|p| s.norm(p).unwrap())]
        }};
    }
    check!(
        "su3",
        SUNFusionRule::new(3).unwrap(),
        [(vec![0i64, 0], 2), (vec![1, 1], 3), (vec![1, 0], 2)]
    );
    let mf = check!(
        "mf su2",
        SU2FusionRule,
        SECTORS.map(|(twice, k)| (su2(twice), k))
    );
    let checked = check!(
        "checked su2",
        SUNFusionRule::new(2).unwrap(),
        SECTORS.map(|(twice, k)| (sun2(twice), k))
    );
    for (p, (mf, checked)) in POWERS.iter().zip(mf[0].iter().zip(checked[0])) {
        numerics::assert_close(&format!("su2 modes norm({p})"), *mf, checked, TERMS);
    }
}
