//! The quantum-dimension weight of truncation and weighted reductions is the
//! exact `dim(c)` in both admission modes (#1871).
//!
//! TensorKit weights `findtruncated(::SectorVector, ...)` by `dim(c)`
//! (`src/factorizations/truncation.jl:187,239`), an `Int` for SU(N), so its
//! budgets and cross-sector ties compare exactly. The oracles here are the
//! Weyl dimension formula and SU(2) as both a multiplicity-free
//! (`SU2FusionRule`) and a checked Generic (`LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(2).unwrap())`) provider:
//! the same spectra must take the same decision in either mode.

#![cfg(feature = "cgc-gen")]

use std::sync::Arc;

use num_complex::Complex64;
use symtn::sector::{LieGroupFusionRule, SU2FusionRule, SU2Irrep};
use symtn::typed::{GradedSpace, SectorSpectrum, SpectrumMagnitude, Truncation};

/// Weyl dimension of the SU(3) irrep with Dynkin labels `(p, q)`.
fn su3_dim(p: i64, q: i64) -> f64 {
    ((p + 1) * (q + 1) * (p + q + 2) / 2) as f64
}

/// The kept count of one sector in a truncated subspace.
macro_rules! kept {
    ($subspace:expr, $sector:expr) => {
        $subspace.degeneracy($sector).unwrap()
    };
}

#[test]
fn su3_quantum_dimension_is_exact() {
    // What: a weighted dimension is the Weyl dimension, not `sqrt(dim)^2`
    // (`sqrt(8)^2 = 8.000000000000002`).
    let rule =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    for p in 0..4 {
        for q in 0..4 {
            let leg = GradedSpace::try_new(Arc::clone(&rule), [(vec![p, q], 2)]).unwrap();
            assert_eq!(leg.dim().unwrap(), 2.0 * su3_dim(p, q), "({p}, {q})");
        }
    }

    // The discarded weight of a unit spectrum is `sqrt(sum_c dim(c))`.
    let leg = GradedSpace::try_new(Arc::clone(&rule), [(vec![1, 1], 1)]).unwrap();
    let spectra = [SectorSpectrum {
        sector: vec![1, 1],
        values: vec![1.0],
    }];
    let found = leg.find_truncated(&spectra, &Truncation::rank(0)).unwrap();
    assert_eq!(found.error, 8f64.sqrt());
    // A budget of exactly `dim(c)` keeps the state (TensorKit `totaldim >
    // howmany` breaks only past the budget); one less keeps nothing.
    let found = leg.find_truncated(&spectra, &Truncation::rank(8)).unwrap();
    assert_eq!(kept!(found.selection.subspace(), &vec![1, 1]), 1);
    let found = leg.find_truncated(&spectra, &Truncation::rank(7)).unwrap();
    assert_eq!(kept!(found.selection.subspace(), &vec![1, 1]), 0);
}

#[test]
fn su3_exact_dimension_tie_breaks_in_tensorkit_order() {
    // What: `3` and `3bar` have equal dimension and equal values; the tie goes
    // to SUNRepresentations' `isless` order, `(0, 1)` before `(1, 0)`: kept
    // first by a rank budget, discarded first by an error budget, whatever
    // order the caller lists the spectra in.
    let rule =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let (three, three_bar) = (vec![1, 0], vec![0, 1]);
    let leg = GradedSpace::try_new(
        Arc::clone(&rule),
        [(three.clone(), 2), (three_bar.clone(), 2)],
    )
    .unwrap();
    let spectrum = |sector: &Vec<i64>| SectorSpectrum {
        sector: sector.clone(),
        values: vec![1.0, 0.5],
    };
    // Weighted norm^2 = 2 * 3 * (1 + 0.25) = 7.5; discarding one 0.5 costs
    // 0.75 and two cost 1.5, so 0.35^2 * 7.5 = 0.91875 discards exactly one.
    let cases = [
        (
            Truncation::rank(3),
            [0, 1],
            (2.0 * 3.0 * 1.25f64 - 3.0).sqrt(),
        ),
        (Truncation::rank(6), [1, 1], 1.5f64.sqrt()),
        (Truncation::rank(9), [1, 2], 0.75f64.sqrt()),
        (
            Truncation::relative_error(0.35).unwrap(),
            [2, 1],
            0.75f64.sqrt(),
        ),
    ];
    for order in [
        [spectrum(&three), spectrum(&three_bar)],
        [spectrum(&three_bar), spectrum(&three)],
    ] {
        for (policy, [three_kept, three_bar_kept], error) in &cases {
            let found = leg.find_truncated(&order, policy).unwrap();
            let subspace = found.selection.subspace();
            assert_eq!(kept!(subspace, &three), *three_kept, "{policy:?}");
            assert_eq!(kept!(subspace, &three_bar), *three_bar_kept, "{policy:?}");
            assert!(
                (found.error - error).abs() <= 4.0 * f64::EPSILON * error,
                "{policy:?}"
            );
        }
    }
}

/// SU(2) spectra with nontrivial degeneracies, as `(2j, values)`.
fn su2_spectra() -> Vec<(usize, Vec<f64>)> {
    vec![
        (0, vec![0.93, 0.41]),
        (1, vec![0.87, 0.52, 0.11]),
        (2, vec![0.74, 0.29]),
        (3, vec![0.66]),
    ]
}

/// Every boundary at which a policy over `spectra` changes its decision, and
/// one representable step on either side, for each value-driven policy.
fn boundary_policies(spectra: &[(usize, Vec<f64>)]) -> Vec<Truncation> {
    let total: usize = spectra.iter().map(|(twice, v)| (twice + 1) * v.len()).sum();
    let norm = spectra
        .iter()
        .map(|(twice, v)| (twice + 1) as f64 * v.iter().map(|x| x * x).sum::<f64>())
        .sum::<f64>()
        .sqrt();
    let mut policies: Vec<_> = (0..=total + 1).map(Truncation::rank).collect();
    for value in spectra.iter().flat_map(|(_, v)| v.iter().copied()) {
        for scale in [value, value / norm] {
            for step in [scale.next_down(), scale, scale.next_up()] {
                policies.push(Truncation::absolute_cutoff(step).unwrap());
                policies.push(Truncation::relative_cutoff(step).unwrap());
                policies.push(Truncation::relative_error(step).unwrap());
            }
        }
    }
    policies
}

fn assert_modes_agree<V: SpectrumMagnitude + Clone>(to_value: impl Fn(f64) -> V) {
    let spectra = su2_spectra();
    let mf_leg = GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        spectra
            .iter()
            .map(|(twice, v)| (SU2Irrep::from_twice_spin(*twice), v.len())),
    )
    .unwrap();
    let checked_leg = GradedSpace::try_new(
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(2).unwrap()).unwrap()),
        spectra
            .iter()
            .map(|(twice, v)| (vec![*twice as i64], v.len())),
    )
    .unwrap();
    let mf_spectra: Vec<_> = spectra
        .iter()
        .map(|(twice, v)| SectorSpectrum {
            sector: SU2Irrep::from_twice_spin(*twice),
            values: v.iter().copied().map(&to_value).collect(),
        })
        .collect();
    let checked_spectra: Vec<_> = spectra
        .iter()
        .map(|(twice, v)| SectorSpectrum {
            sector: vec![*twice as i64],
            values: v.iter().copied().map(&to_value).collect(),
        })
        .collect();
    assert_eq!(mf_leg.dim().unwrap(), checked_leg.dim().unwrap());
    for policy in boundary_policies(&spectra) {
        let mf = mf_leg.find_truncated(&mf_spectra, &policy).unwrap();
        let checked = checked_leg
            .find_truncated(&checked_spectra, &policy)
            .unwrap();
        for (twice, _) in &spectra {
            assert_eq!(
                kept!(mf.selection.subspace(), &SU2Irrep::from_twice_spin(*twice)),
                kept!(checked.selection.subspace(), &vec![*twice as i64]),
                "2j = {twice} under {policy:?}"
            );
        }
        // One weight and one sector order: the same arithmetic, bit for bit.
        assert_eq!(
            mf.error.to_bits(),
            checked.error.to_bits(),
            "error under {policy:?}"
        );
    }
}

#[test]
fn multiplicity_free_and_checked_su2_take_one_decision_f64() {
    // What: SU(2) through either admission mode keeps the same states and
    // reports the same discarded weight at every decision boundary.
    assert_modes_agree(|value| value);
}

#[test]
fn multiplicity_free_and_checked_su2_take_one_decision_c64() {
    // What: the same for complex spectra, selected by magnitude.
    assert_modes_agree(|value| Complex64::from_polar(value, 0.7));
}
