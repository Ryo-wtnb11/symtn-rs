//! The general eigendecomposition truncated as a composition (#1534).
//!
//! The invariant: `eig_full` → `diagview` → `find_truncated` →
//! `restrict_leg` keeps the eigenpairs a hand selection
//! keeps, and nothing else. Every fixture is triangular in each coupled
//! sector, so its eigenvalues are its diagonal (`triangular_eigenvalues`), and
//! the kept pairs are checked through the gauge-free relation
//! `t * v = v * d`; see `truncation_oracle` for why this expectation shares no
//! code with symtn's factorization or decision.
//!
//! Each case asserts, exactly: the kept bond leg against the hand selection,
//! every factor's spaces, and every factor's block geometry against the
//! canonical layout of its homspace. Then, within the workspace tolerance rule
//! (`docs/testing_numerics.md`): the kept eigenvalues, the eigen relation with
//! nonzero eigenvector columns, and the truncation error against the discarded
//! norm.

use std::sync::Arc;

use num_complex::Complex64;
use symtn::sector::{
    FermionParityFusionRule, SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep, Z2Irrep,
};
use symtn::typed::TensorMap;
use symtn::typed::{Eig, GradedSpace, SectorSpectrum};

use crate::numerics;
use crate::truncation_oracle;

use truncation_oracle::{
    discarded_norm, kept_values, select, triangular_eigenvalues, ClosedFormDim, Offer,
};

use crate::fixtures;

use fixtures::{codomain_axes, domain_axes, host_runtime};

fn fill(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 33) as f64) / (u32::MAX as f64) - 0.5
}

fn u1_leg(pairs: &[(i32, usize)]) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        pairs
            .iter()
            .map(|&(charge, degeneracy)| (U1Irrep::new(charge), degeneracy)),
    )
    .unwrap()
}

fn su2_leg(pairs: &[(usize, usize)]) -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        pairs
            .iter()
            .map(|&(twice_spin, degeneracy)| (SU2Irrep::from_twice_spin(twice_spin), degeneracy)),
    )
    .unwrap()
}

fn fz2_leg(pairs: &[(bool, usize)]) -> GradedSpace<FermionParityFusionRule> {
    GradedSpace::try_new(
        Arc::new(FermionParityFusionRule),
        pairs
            .iter()
            .map(|&(odd, degeneracy)| (if odd { Z2Irrep::ODD } else { Z2Irrep::EVEN }, degeneracy)),
    )
    .unwrap()
}

/// The whole gate for one case. `$complex` lifts the source to the complex
/// payload of the factors.
macro_rules! assert_eig_composition {
    ($source:expr, $complex:expr, $truncation:expr, $policy:expr, $case:expr) => {{
        let source = &$source;
        let case: &str = $case;
        let truncation: &symtn::typed::Truncation = &$truncation;

        let Eig { d, v } = source
            .eig_full(&codomain_axes(&source), &domain_axes(&source))
            .unwrap();
        let bond = d.domain()[0].clone();
        let spectra = d.diagview().unwrap();
        let found = bond.find_truncated(&spectra, truncation).unwrap();
        let got_d = d
            .restrict_leg(&[(0, &found.selection), (1, &found.selection)])
            .unwrap();
        let got_v = v
            .restrict_leg(&[(v.codomain_rank(), &found.selection)])
            .unwrap();

        let blocks = sector_matrices!(source);
        let references: Vec<(_, Vec<Complex64>)> = blocks
            .iter()
            .map(|(sector, matrix)| (sector.clone(), triangular_eigenvalues(matrix)))
            .collect();
        let offers: Vec<Offer<_>> = references
            .iter()
            .map(|(sector, values)| Offer {
                sector: sector.clone(),
                dim: ClosedFormDim::closed_form_dim(sector),
                magnitudes: values.iter().map(|value| value.norm()).collect(),
            })
            .collect();
        let kept = select(&offers, &$policy);

        assert_kept_bond!(got_d.domain()[0], offers, kept, case);
        assert_eq!(got_d.codomain(), got_d.domain(), "{case}: d is a bond map");
        assert_eq!(got_v.codomain(), source.codomain(), "{case}: v codomain");
        assert_eq!(got_v.domain(), got_d.domain(), "{case}: v domain");
        assert!(
            symtn::expert::diagonal_spectrum(&got_d).unwrap().is_some(),
            "{case}: d stays compact"
        );
        assert_canonical_layout!(got_d, format!("{case}: d"));
        assert_canonical_layout!(got_v, format!("{case}: v"));

        let terms = source.materialize().unwrap().dense_data().unwrap().len();
        for entry in got_d.diagview().unwrap() {
            let (_, reference) = references
                .iter()
                .find(|(sector, _)| *sector == entry.sector)
                .expect("a kept sector is an offered sector");
            numerics::assert_slices_close(
                &format!("{case}: kept eigenvalues of {:?}", entry.sector),
                &entry.values,
                &kept_values(&offers, &kept, &entry.sector, reference),
                terms,
            );
        }

        let complex = $complex;
        let left = complex.compose(&got_v).unwrap();
        let right = got_v.compose(&got_d).unwrap();
        numerics::assert_slices_close(
            &format!("{case}: t * v = v * d"),
            left.dense_data().unwrap(),
            right.dense_data().unwrap(),
            terms,
        );
        // The relation alone is met by `v = 0`. An eigenvector's scale is a
        // gauge the API leaves open, so only "nonzero" is asserted.
        for (sector, matrix) in sector_matrices!(got_v) {
            for col in 0..matrix.cols {
                let norm: f64 = (0..matrix.rows)
                    .map(|row| matrix.at(row, col).norm_sqr())
                    .sum::<f64>()
                    .sqrt();
                assert!(norm > 0.5, "{case}: v of {sector:?} column {col} is {norm}");
            }
        }

        numerics::assert_close(
            &format!("{case}: truncation error"),
            found.error,
            discarded_norm(&offers, &kept),
            terms,
        );

        found
    }};
}

macro_rules! eig_policy_sweep {
    ($source:expr, $complex:expr, $target:expr, $tag:expr) => {{
        let complex = $complex;
        let source = $source;
        for (name, truncation, policy) in policies!($target) {
            assert_eig_composition!(
                source,
                complex.clone(),
                truncation,
                policy,
                &format!("{} {name}", $tag)
            );
        }
    }};
}

/// Upper triangular in every coupled sector: `diagonal` on the diagonal,
/// `upper` above it, zero below.
fn triangular<R, D>(
    leg: &GradedSpace<R>,
    mut diagonal: impl FnMut() -> D,
    mut upper: impl FnMut() -> D,
    zero: D,
) -> TensorMap<R, D>
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + symtn::sector::MultiplicityFreeRigidSymbols<Scalar = f64>
        + symtn::sector::CheckedFusionAlgebra
        + symtn::sector::SectorCodec,
    D: symtn::typed::TensorScalar,
{
    TensorMap::from_subblock_fn(&host_runtime(), [leg], [leg], |_, indices| {
        match indices[0].cmp(&indices[1]) {
            std::cmp::Ordering::Equal => diagonal(),
            std::cmp::Ordering::Less => upper(),
            std::cmp::Ordering::Greater => zero,
        }
    })
    .unwrap()
}

fn real_triangular<R>(leg: &GradedSpace<R>, seed: u64) -> TensorMap<R, f64>
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + symtn::sector::MultiplicityFreeRigidSymbols<Scalar = f64>
        + symtn::sector::CheckedFusionAlgebra
        + symtn::sector::SectorCodec,
{
    let state = std::cell::Cell::new(seed);
    let next = || {
        let mut s = state.get();
        let value = fill(&mut s);
        state.set(s);
        value
    };
    triangular(leg, || 4.0 * next(), || 0.5 * next(), 0.0)
}

fn complex_triangular<R>(leg: &GradedSpace<R>, seed: u64) -> TensorMap<R, Complex64>
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + symtn::sector::MultiplicityFreeRigidSymbols<Scalar = f64>
        + symtn::sector::CheckedFusionAlgebra
        + symtn::sector::SectorCodec,
{
    let state = std::cell::Cell::new(seed);
    let next = || {
        let mut s = state.get();
        let value = fill(&mut s);
        state.set(s);
        value
    };
    triangular(
        leg,
        || Complex64::new(4.0 * next(), 4.0 * next()),
        || Complex64::new(0.5 * next(), 0.5 * next()),
        Complex64::new(0.0, 0.0),
    )
}

#[test]
fn u1_eig_composition_matches_the_hand_selection_for_every_policy() {
    let leg = u1_leg(&[(-1, 2), (0, 3), (1, 2)]);
    let source = real_triangular(&leg, 0x0e19_0001);
    eig_policy_sweep!(
        source,
        source.convert::<Complex64>(),
        u1_leg(&[(-1, 1), (0, 2)]),
        "u1 eig f64"
    );
    let source = complex_triangular(&leg, 0x0e19_0002);
    eig_policy_sweep!(
        source,
        source.clone(),
        u1_leg(&[(-1, 1), (0, 2)]),
        "u1 eig c64"
    );
}

#[test]
fn su2_eig_composition_matches_the_hand_selection_for_every_policy() {
    // dim(c) = 2j + 1 weights the rank budget and the discarded norm.
    let leg = su2_leg(&[(0, 3), (1, 2), (2, 2)]);
    let target = su2_leg(&[(0, 2), (2, 1)]);
    let source = real_triangular(&leg, 0x0e19_0003);
    eig_policy_sweep!(source, source.convert::<Complex64>(), target, "su2 eig f64");
    let source = complex_triangular(&leg, 0x0e19_0004);
    eig_policy_sweep!(source, source.clone(), target, "su2 eig c64");
}

#[test]
fn fermionic_eig_composition_matches_the_hand_selection_for_every_policy() {
    let leg = fz2_leg(&[(false, 3), (true, 3)]);
    let target = fz2_leg(&[(false, 2), (true, 1)]);
    let source = real_triangular(&leg, 0x0e19_0005);
    eig_policy_sweep!(source, source.convert::<Complex64>(), target, "fz2 eig f64");
    let source = complex_triangular(&leg, 0x0e19_0006);
    eig_policy_sweep!(source, source.clone(), target, "fz2 eig c64");
}

#[test]
fn lazy_adjoint_eig_composition_matches_the_hand_selection() {
    // The adjoint of an upper-triangular map is lower triangular, with the
    // conjugated diagonal: still a hand answer.
    let leg = u1_leg(&[(0, 3), (1, 2)]);
    let lazy = complex_triangular(&leg, 0x0e19_0007).adjoint().unwrap();
    eig_policy_sweep!(
        lazy,
        lazy.clone(),
        u1_leg(&[(0, 2), (1, 1)]),
        "u1 eig lazy adjoint"
    );
}

#[test]
fn a_whole_sector_is_dropped_from_the_eig_bond() {
    let leg = u1_leg(&[(0, 3), (1, 2)]);
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&host_runtime(), [&leg], [&leg], |trees, indices| {
            let scale = if trees.coupled() == &U1Irrep::new(0) {
                1.0
            } else {
                1.0e-6
            };
            match indices[0].cmp(&indices[1]) {
                std::cmp::Ordering::Equal => scale * (3 - indices[0]) as f64,
                std::cmp::Ordering::Less => scale * 0.25,
                std::cmp::Ordering::Greater => 0.0,
            }
        })
        .unwrap();
    let found = assert_eig_composition!(
        source,
        source.convert::<Complex64>(),
        symtn::typed::Truncation::relative_cutoff(1e-3).unwrap(),
        truncation_oracle::Policy::RelativeCutoff(1e-3),
        "drop"
    );
    assert_eq!(
        found.selection.subspace().sectors().unwrap(),
        vec![U1Irrep::new(0)],
        "the low sector must be gone from the bond, not merely shortened"
    );
}

/// A rank-(2,2) endomorphism with several fusion trees per coupled sector,
/// upper triangular in the oracle's sector-matrix order: a basis state is
/// `(tree labels, degeneracy index with the first axis fastest)`, the order
/// in which `truncation_oracle` lays out rows and columns, so the eigenvalues
/// are still the hand-set diagonal (`triangular_eigenvalues` checks the
/// shape).
fn multi_tree_triangular<R>(codomain: [&GradedSpace<R>; 2], seed: u64) -> TensorMap<R, Complex64>
where
    R: symtn::sector::TypedSectorAdmission<
            Sector = <R as symtn::sector::SectorCodec>::Sector,
            Error = symtn::typed::FusionAlgebraError,
            Mode = symtn::sector::MultiplicityFreeAdmissionMode,
        > + symtn::sector::MultiplicityFreeRigidSymbols<Scalar = f64>
        + symtn::sector::CheckedFusionAlgebra
        + symtn::sector::SectorCodec,
    <R as symtn::sector::SectorCodec>::Sector: Ord + Clone,
{
    let mut state = seed;
    TensorMap::from_subblock_fn(&host_runtime(), codomain, codomain, |trees, indices| {
        let row = (
            trees.codomain_uncoupled().to_vec(),
            trees.codomain_innerlines().to_vec(),
            indices[1],
            indices[0],
        );
        let col = (
            trees.domain_uncoupled().to_vec(),
            trees.domain_innerlines().to_vec(),
            indices[3],
            indices[2],
        );
        let value = Complex64::new(fill(&mut state), fill(&mut state));
        match row.cmp(&col) {
            std::cmp::Ordering::Equal => 4.0 * value,
            std::cmp::Ordering::Less => 0.5 * value,
            std::cmp::Ordering::Greater => Complex64::new(0.0, 0.0),
        }
    })
    .unwrap()
}

#[test]
fn multi_tree_eig_composition_matches_the_hand_selection_for_every_policy() {
    // Two U(1) legs: coupled sector 1 carries the trees (0, 1) and (1, 0).
    let leg = u1_leg(&[(0, 2), (1, 2)]);
    let source = multi_tree_triangular([&leg, &leg], 0x0e19_0008);
    assert!(
        (0..source.subblock_count()).any(|i| {
            let coupled = *source.subblock_fusion_trees(i).unwrap().coupled();
            (0..source.subblock_count())
                .filter(|&j| source.subblock_fusion_trees(j).unwrap().coupled() == &coupled)
                .count()
                > 1
        }),
        "the fixture must stack several trees in one coupled sector"
    );
    eig_policy_sweep!(
        source,
        source.clone(),
        u1_leg(&[(0, 1), (1, 2)]),
        "u1 x u1 eig c64"
    );

    // Fermion parity with a dual leg: leg (x) leg' <- leg (x) leg', so the
    // odd-odd trees and the twist of the dual leg enter the layout.
    let leg = fz2_leg(&[(false, 2), (true, 2)]);
    let dual = leg.try_dual().unwrap();
    let source = multi_tree_triangular([&leg, &dual], 0x0e19_0009);
    eig_policy_sweep!(
        source,
        source.clone(),
        fz2_leg(&[(false, 2), (true, 1)]),
        "fz2 x fz2' eig c64"
    );
}

/// Odd positions, then even: a fixed shuffle of `0..n` that turns the
/// published (ascending) spectrum into one that is neither ascending nor
/// descending.
fn shuffle(n: usize) -> Vec<usize> {
    (1..n).step_by(2).chain((0..n).step_by(2)).collect()
}

/// The decision follows stored order, not a published sort (#2095): the bond
/// of `eig_full` is re-stored in the [`shuffle`]d order — `d' = diag(d[π])`
/// and `v' = v P` with `P[π(j), j] = 1`, so `t v' = v' d'` still holds — and
/// the composition must keep exactly the hand selection's positions over the
/// same shuffled order, gather `v'`'s columns at those positions, and keep
/// the factorization relation.
macro_rules! assert_shuffled_eig_composition {
    ($source:expr, $target:expr, $tag:expr) => {{
        let source = &$source;
        let runtime = source.runtime().clone();
        let Eig { d, v } = source
            .eig_full(&codomain_axes(source), &domain_axes(source))
            .unwrap();
        let bond = d.domain()[0].clone();
        let stored: Vec<SectorSpectrum<_, Complex64>> = d
            .diagview()
            .unwrap()
            .into_iter()
            .map(|entry| SectorSpectrum {
                values: shuffle(entry.values.len())
                    .into_iter()
                    .map(|p| entry.values[p])
                    .collect(),
                sector: entry.sector,
            })
            .collect();
        let d = TensorMap::diagonal(&runtime, &bond, stored).unwrap();
        let permutation: TensorMap<_, Complex64> =
            TensorMap::from_subblock_fn(&runtime, [&bond], [&bond], |trees, indices| {
                let n = bond.degeneracy(trees.coupled()).unwrap();
                if indices[0] == shuffle(n)[indices[1]] {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                }
            })
            .unwrap();
        let v = v.compose(&permutation).unwrap();
        let spectra = d.diagview().unwrap();

        let references: Vec<(_, Vec<Complex64>)> = sector_matrices!(source)
            .iter()
            .map(|(sector, matrix)| {
                let published = triangular_eigenvalues(matrix);
                let stored = shuffle(published.len())
                    .into_iter()
                    .map(|p| published[p])
                    .collect();
                (sector.clone(), stored)
            })
            .collect();
        let offers: Vec<Offer<_>> = references
            .iter()
            .map(|(sector, values)| Offer {
                sector: sector.clone(),
                dim: ClosedFormDim::closed_form_dim(sector),
                magnitudes: values.iter().map(|value| value.norm()).collect(),
            })
            .collect();
        let full_v = sector_matrices!(v);
        let terms = source.materialize().unwrap().dense_data().unwrap().len();
        let mut gapped = false;
        for (name, truncation, policy) in policies!($target) {
            let case = format!("{} {name}", $tag);
            let found = bond.find_truncated(&spectra, &truncation).unwrap();
            let kept = select(&offers, &policy);
            for (offer, positions) in offers.iter().zip(&kept) {
                assert_eq!(
                    found.selection.positions(&offer.sector).unwrap(),
                    *positions,
                    "{case}: kept positions of {:?}",
                    offer.sector
                );
                gapped |= positions.windows(2).any(|pair| pair[1] != pair[0] + 1);
            }
            let selection = &found.selection;
            let got_d = d.restrict_leg(&[(0, selection), (1, selection)]).unwrap();
            let got_v = v.restrict_leg(&[(v.codomain_rank(), selection)]).unwrap();

            // Restriction is a gather at the kept positions, exactly.
            for (sector, matrix) in sector_matrices!(got_v) {
                let (_, parent) = full_v.iter().find(|(s, _)| *s == sector).unwrap();
                let positions = selection.positions(&sector).unwrap();
                assert_eq!(matrix.cols, positions.len(), "{case}: v columns");
                for (col, &position) in positions.iter().enumerate() {
                    for row in 0..matrix.rows {
                        assert_eq!(matrix.at(row, col), parent.at(row, position), "{case}");
                    }
                }
            }
            for entry in got_d.diagview().unwrap() {
                let (_, reference) = references
                    .iter()
                    .find(|(sector, _)| *sector == entry.sector)
                    .unwrap();
                let want: Vec<Complex64> = selection
                    .positions(&entry.sector)
                    .unwrap()
                    .into_iter()
                    .map(|p| reference[p])
                    .collect();
                numerics::assert_slices_close(
                    &format!("{case}: kept eigenvalues of {:?}", entry.sector),
                    &entry.values,
                    &want,
                    terms,
                );
            }
            let left = source.compose(&got_v).unwrap();
            let right = got_v.compose(&got_d).unwrap();
            numerics::assert_slices_close(
                &format!("{case}: t * v = v * d"),
                left.dense_data().unwrap(),
                right.dense_data().unwrap(),
                terms,
            );
            numerics::assert_close(
                &format!("{case}: truncation error"),
                found.error,
                discarded_norm(&offers, &kept),
                terms,
            );
        }
        assert!(
            gapped,
            "{}: some policy must keep a non-contiguous set",
            $tag
        );
    }};
}

#[test]
fn shuffled_eig_bond_is_truncated_at_tensorkits_stored_positions() {
    let leg = u1_leg(&[(-1, 2), (0, 4), (1, 3)]);
    assert_shuffled_eig_composition!(
        complex_triangular(&leg, 0x0e19_2095),
        u1_leg(&[(-1, 1), (0, 2), (1, 2)]),
        "u1 shuffled eig c64"
    );
    let leg = su2_leg(&[(0, 4), (1, 3), (2, 2)]);
    assert_shuffled_eig_composition!(
        complex_triangular(&leg, 0x0e19_2096),
        su2_leg(&[(0, 2), (1, 2)]),
        "su2 shuffled eig c64"
    );
}
