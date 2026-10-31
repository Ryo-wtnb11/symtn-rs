//! The truncation composition on a checked Generic provider.
//!
//! SU(3) with the adjoint irrep is the case the multiplicity-free file cannot
//! reach: the quantum-dimension weight is the fallible exact `try_dim_scalar`.
//! Compact SVD stores `s` diagonally; its explicitly materialized copy also
//! exercises `diagview`'s strided dense arm.
//!
//! The SVD source is rank `(2,1)` on purpose. A `bond <- bond` map has
//! rank-one trees and therefore no outer-multiplicity vertex at all; with two
//! adjoint legs in the codomain, `8 ⊗ 8 -> 8` carries two fusion channels, so
//! `u`'s blocks differ only by their vertex label and the restriction has to
//! leave that part of the key alone.
//!
//! The gate is the multiplicity-free one (`truncation_composition.rs`): the
//! kept bond and every factor's spaces and block geometry exactly, then the
//! kept values, the gauge-free factor relations and the truncation error
//! under the workspace tolerance rule (`docs/testing_numerics.md`), all
//! against the symtn-independent expectation of `truncation_oracle`.

#![cfg(feature = "cgc-gen")]

use std::sync::Arc;
use symtn::typed::HermitianTol;

use num_complex::Complex64;
use symtn::sector::LieGroupFusionRule;
use symtn::typed::TensorMap;
use symtn::typed::{Eig, Eigh, GradedSpace, SectorSpectrum, Svd, Truncation};

use crate::numerics;
use crate::truncation_oracle;

use truncation_oracle::{
    assert_error_close, assert_kept_magnitudes, discarded_norm, kept_values, select,
    triangular_eigenvalues, Offer,
};

use crate::fixtures;

use fixtures::host_runtime;

use crate::su3::su3_dim;

fn fill(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 33) as f64) / (u32::MAX as f64) - 0.5
}

/// `x^H` as an owned tensor: checked-Generic `compose` takes owned operands
/// only, and `add` is the owned copy that accepts a lazy adjoint.
macro_rules! owned_adjoint {
    ($engine:ident, $x:expr, $one:expr) => {{
        let x = &$x;
        let adjoint = x.adjoint($engine).unwrap();
        adjoint.axpby($engine, $one, &adjoint, $one - $one).unwrap()
    }};
}

/// `lhs` and `rhs` are `compose` results on one homspace and agree within the
/// tolerance rule.
macro_rules! assert_relation {
    ($lhs:expr, $rhs:expr, $terms:expr, $what:expr) => {{
        let (lhs, rhs) = (&$lhs, &$rhs);
        assert_eq!(lhs.codomain(), rhs.codomain(), "{} codomain", $what);
        assert_eq!(lhs.domain(), rhs.domain(), "{} domain", $what);
        numerics::assert_slices_close(
            &$what,
            lhs.dense_data().unwrap(),
            rhs.dense_data().unwrap(),
            $terms,
        );
    }};
}

/// `x^H x = 1` on the bond, with the identity spelled by its reduced blocks.
macro_rules! assert_isometry {
    ($engine:ident, $gram:expr, $bond:expr, $one:expr, $terms:expr, $what:expr) => {{
        let gram = $gram;
        let (one, zero) = ($one, $one - $one);
        let identity =
            TensorMap::from_subblock_fn($engine, gram.runtime(), [&$bond], [&$bond], |_, index| {
                if index[0] == index[1] {
                    one
                } else {
                    zero
                }
            })
            .unwrap();
        assert_relation!(gram, identity, $terms, $what);
    }};
}

fn su3_legs() -> (Arc<LieGroupFusionRule>, GradedSpace<LieGroupFusionRule>) {
    let provider =
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap());
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [(vec![1i64, 0], 3), (vec![2i64, 2], 2)],
    )
    .unwrap();
    (provider, leg)
}

/// A smaller space on the same provider, used as a `truncspace` target.
fn su3_target(provider: &Arc<LieGroupFusionRule>) -> GradedSpace<LieGroupFusionRule> {
    GradedSpace::try_new(Arc::clone(provider), [(vec![2i64, 2], 2)]).unwrap()
}

macro_rules! assert_su3_svd_composition {
    ($engine:ident, $source:expr, $one:expr, $target:expr, $tag:expr) => {{
        let source = $source;
        let offers = singular_offers!($engine, source, su3_dim);
        let terms = source.dense_data().unwrap().len();
        for (name, truncation, policy) in policies!($target) {
            let case = format!("{} {name}", $tag);
            let Svd { u, s, vh } = source.svd_compact($engine, &[0, 1], &[2]).unwrap();
            assert!(
                symtn::expert::diagonal_spectrum($engine, &s)
                    .unwrap()
                    .is_some(),
                "{case}: compact SVD must store s diagonally"
            );
            let dense_s = s.materialize($engine).unwrap();
            assert!(symtn::expert::diagonal_spectrum($engine, &dense_s)
                .unwrap()
                .is_none());
            assert_eq!(
                s.diagview($engine).unwrap(),
                dense_s.diagview($engine).unwrap()
            );
            let bond = s.domain()[0].clone();
            let found = bond
                .find_truncated(&s.diagview($engine).unwrap(), &truncation)
                .unwrap();
            let selection = &found.selection;
            let got_u = u
                .restrict_leg($engine, &[(u.codomain_rank(), selection)])
                .unwrap();
            let got_s = s
                .restrict_leg($engine, &[(0, selection), (1, selection)])
                .unwrap();
            let got_vh = vh.restrict_leg($engine, &[(0, selection)]).unwrap();
            let kept = select(&offers, &policy);

            let kept_bond = got_s.domain()[0].clone();
            assert_kept_bond!(kept_bond, offers, kept, case);
            assert_eq!(got_s.codomain(), got_s.domain(), "{case}: s is a bond map");
            assert_eq!(got_u.codomain(), source.codomain(), "{case}: u codomain");
            assert_eq!(got_u.domain(), got_s.domain(), "{case}: u domain");
            assert_eq!(got_vh.codomain(), got_s.domain(), "{case}: vh codomain");
            assert_eq!(got_vh.domain(), source.domain(), "{case}: vh domain");
            assert_canonical_layout!($engine, got_u, format!("{case}: u"));
            assert_canonical_layout!($engine, got_s, format!("{case}: s"));
            assert_canonical_layout!($engine, got_vh, format!("{case}: vh"));

            assert_kept_magnitudes(
                &case,
                &got_s.diagview($engine).unwrap(),
                &offers,
                &kept,
                terms,
            );
            assert_relation!(
                {
                    let adjoint = owned_adjoint!($engine, got_vh, $one);
                    source.compose($engine, &adjoint)
                }
                .unwrap(),
                got_u.compose($engine, &got_s).unwrap(),
                terms,
                format!("{case}: t vh^H = u s")
            );
            assert_relation!(
                owned_adjoint!($engine, got_u, $one)
                    .compose($engine, &source)
                    .unwrap(),
                got_s.compose($engine, &got_vh).unwrap(),
                terms,
                format!("{case}: u^H t = s vh")
            );
            assert_isometry!(
                $engine,
                owned_adjoint!($engine, got_u, $one)
                    .compose($engine, &got_u)
                    .unwrap(),
                kept_bond,
                $one,
                terms,
                format!("{case}: u^H u")
            );
            assert_isometry!(
                $engine,
                {
                    let adjoint = owned_adjoint!($engine, got_vh, $one);
                    got_vh.compose($engine, &adjoint)
                }
                .unwrap(),
                kept_bond,
                $one,
                terms,
                format!("{case}: vh vh^H")
            );
            assert_error_close(
                &case,
                source.dense_data().unwrap(),
                found.error,
                discarded_norm(&offers, &kept),
                terms,
            );
        }
    }};
}

macro_rules! assert_su3_eigh_composition {
    ($engine:ident, $source:expr, $one:expr, $target:expr, $tag:expr) => {{
        let source = $source;
        // A Hermitian block's singular values are its |lambda|.
        let offers = singular_offers!($engine, source, su3_dim);
        let terms = source.dense_data().unwrap().len();
        for (name, truncation, policy) in policies!($target) {
            let case = format!("{} {name}", $tag);
            let Eigh { d, v } = source
                .eigh_full($engine, &[0], &[1], HermitianTol::DEFAULT)
                .unwrap();
            let bond = d.domain()[0].clone();
            let found = bond
                .find_truncated(&d.diagview($engine).unwrap(), &truncation)
                .unwrap();
            let selection = &found.selection;
            let got_d = d
                .restrict_leg($engine, &[(0, selection), (1, selection)])
                .unwrap();
            let got_v = v
                .restrict_leg($engine, &[(v.codomain_rank(), selection)])
                .unwrap();
            let kept = select(&offers, &policy);

            let kept_bond = got_d.domain()[0].clone();
            assert_kept_bond!(kept_bond, offers, kept, case);
            assert_eq!(got_d.codomain(), got_d.domain(), "{case}: d is a bond map");
            assert_eq!(got_v.codomain(), source.codomain(), "{case}: v codomain");
            assert_eq!(got_v.domain(), got_d.domain(), "{case}: v domain");
            assert_canonical_layout!($engine, got_d, format!("{case}: d"));
            assert_canonical_layout!($engine, got_v, format!("{case}: v"));

            assert_kept_magnitudes(
                &case,
                &got_d.diagview($engine).unwrap(),
                &offers,
                &kept,
                terms,
            );
            assert_relation!(
                source.compose($engine, &got_v).unwrap(),
                got_v.compose($engine, &got_d).unwrap(),
                terms,
                format!("{case}: t v = v d")
            );
            assert_isometry!(
                $engine,
                owned_adjoint!($engine, got_v, $one)
                    .compose($engine, &got_v)
                    .unwrap(),
                kept_bond,
                $one,
                terms,
                format!("{case}: v^H v")
            );
            assert_error_close(
                &case,
                source.dense_data().unwrap(),
                found.error,
                discarded_norm(&offers, &kept),
                terms,
            );
        }
    }};
}

/// The general eigendecomposition on a triangular source, whose eigenvalues
/// are its diagonal. `$complex` is the source lifted to the factor dtype.
macro_rules! assert_su3_eig_composition {
    ($engine:ident, $source:expr, $complex:expr, $target:expr, $tag:expr) => {{
        let source = $source;
        let complex = $complex;
        let references: Vec<(Vec<i64>, Vec<Complex64>)> = sector_matrices!($engine, source)
            .iter()
            .map(|(sector, matrix)| (sector.clone(), triangular_eigenvalues(matrix)))
            .collect();
        let offers: Vec<Offer<Vec<i64>>> = references
            .iter()
            .map(|(sector, values)| Offer {
                sector: sector.clone(),
                dim: su3_dim(sector),
                magnitudes: values.iter().map(|value| value.norm()).collect(),
            })
            .collect();
        let terms = source.dense_data().unwrap().len();
        for (name, truncation, policy) in policies!($target) {
            let case = format!("{} {name}", $tag);
            let Eig { d, v } = source.eig_full($engine, &[0], &[1]).unwrap();
            let bond = d.domain()[0].clone();
            let found = bond
                .find_truncated(&d.diagview($engine).unwrap(), &truncation)
                .unwrap();
            let selection = &found.selection;
            let got_d = d
                .restrict_leg($engine, &[(0, selection), (1, selection)])
                .unwrap();
            let got_v = v
                .restrict_leg($engine, &[(v.codomain_rank(), selection)])
                .unwrap();
            let kept = select(&offers, &policy);

            assert_kept_bond!(got_d.domain()[0], offers, kept, case);
            assert_eq!(got_d.codomain(), got_d.domain(), "{case}: d is a bond map");
            assert_eq!(got_v.codomain(), source.codomain(), "{case}: v codomain");
            assert_eq!(got_v.domain(), got_d.domain(), "{case}: v domain");
            assert_canonical_layout!($engine, got_d, format!("{case}: d"));
            assert_canonical_layout!($engine, got_v, format!("{case}: v"));
            for entry in got_d.diagview($engine).unwrap() {
                let (_, reference) = references
                    .iter()
                    .find(|(sector, _)| *sector == entry.sector)
                    .unwrap();
                numerics::assert_slices_close(
                    &format!("{case}: kept eigenvalues of {:?}", entry.sector),
                    &entry.values,
                    &kept_values(&offers, &kept, &entry.sector, reference),
                    terms,
                );
            }
            assert_relation!(
                complex.compose($engine, &got_v).unwrap(),
                got_v.compose($engine, &got_d).unwrap(),
                terms,
                format!("{case}: t v = v d")
            );
            assert_error_close(
                &case,
                source.dense_data().unwrap(),
                found.error,
                discarded_norm(&offers, &kept),
                terms,
            );
        }
    }};
}

#[test]
fn su3_svd_composition_matches_the_oracle_for_every_policy() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let (provider, leg) = su3_legs();
    let mut state = 0x5150_2701u64;
    let source: TensorMap<_, f64> = TensorMap::from_subblock_fn(
        engine,
        &host_runtime(),
        [&leg, &leg],
        [&leg],
        move |_, _| fill(&mut state),
    )
    .unwrap();
    assert!(
        (0..source.subblock_count()).any(|index| source
            .subblock_fusion_trees(index)
            .unwrap()
            .codomain_vertices()
            .iter()
            .any(|vertex| vertex.get() == 2)),
        "the fixture must carry a Generic outer-multiplicity vertex mu = 2"
    );
    assert_su3_svd_composition!(engine, source, 1.0, su3_target(&provider), "su3 f64");
}

#[test]
fn su3_complex_svd_composition_matches_the_oracle_for_every_policy() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let (provider, leg) = su3_legs();
    let mut state = 0x5150_2703u64;
    let source: TensorMap<_, Complex64> = TensorMap::from_subblock_fn(
        engine,
        &host_runtime(),
        [&leg, &leg],
        [&leg],
        move |_, _| Complex64::new(fill(&mut state), fill(&mut state)),
    )
    .unwrap();
    assert_su3_svd_composition!(
        engine,
        source,
        Complex64::ONE,
        su3_target(&provider),
        "su3 c64"
    );
}

#[test]
fn su3_eigh_composition_matches_the_oracle_for_every_policy() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let (provider, leg) = su3_legs();
    let mut state = 0x5150_2702u64;
    let raw: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(engine, &host_runtime(), [&leg], [&leg], move |_, _| {
            fill(&mut state)
        })
        .unwrap();
    let raw_adjoint = &raw.adjoint(engine).unwrap();
    let source = raw.axpby(engine, 1.0, raw_adjoint, 1.0).unwrap();
    assert_su3_eigh_composition!(engine, source, 1.0, su3_target(&provider), "su3 eigh f64");
}

#[test]
fn su3_complex_eigh_composition_matches_the_oracle_for_every_policy() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let (provider, leg) = su3_legs();
    let mut state = 0x5150_2704u64;
    let raw: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(engine, &host_runtime(), [&leg], [&leg], move |_, _| {
            Complex64::new(fill(&mut state), fill(&mut state))
        })
        .unwrap();
    let one = Complex64::new(1.0, 0.0);
    let raw_adjoint = &raw.adjoint(engine).unwrap();
    let source = raw.axpby(engine, one, raw_adjoint, one).unwrap();
    assert_su3_eigh_composition!(
        engine,
        source,
        Complex64::ONE,
        su3_target(&provider),
        "su3 eigh c64"
    );
}

#[test]
fn su3_eig_composition_matches_the_oracle_for_every_policy() {
    let engine = &mut symtn::typed::CpuEngine::serial();
    let (provider, leg) = su3_legs();
    let mut state = 0x5150_2705u64;
    let source: TensorMap<_, f64> = TensorMap::from_subblock_fn(
        engine,
        &host_runtime(),
        [&leg],
        [&leg],
        move |_, indices| match indices[0].cmp(&indices[1]) {
            std::cmp::Ordering::Equal => 4.0 * fill(&mut state),
            std::cmp::Ordering::Less => 0.5 * fill(&mut state),
            std::cmp::Ordering::Greater => 0.0,
        },
    )
    .unwrap();
    let complex = source.convert::<Complex64>(engine).unwrap();
    assert_su3_eig_composition!(
        engine,
        source,
        complex,
        su3_target(&provider),
        "su3 eig f64"
    );
}

#[test]
fn checked_generic_decision_keeps_stored_positions() {
    // What: the checked-Generic decision takes the stored order too.
    // Weights are the exact SU(3) dimensions 3 and 27; no magnitude ties, so
    // the hand answers below need no sector order. norm^2 = 3 (0.25 + 9 + 4)
    // + 27 (2.25 + 1) = 127.5.
    let (_, leg) = su3_legs();
    let spectra = [
        SectorSpectrum {
            sector: vec![1i64, 0],
            values: vec![0.5, -3.0, 2.0],
        },
        SectorSpectrum {
            sector: vec![2i64, 2],
            values: vec![1.5, 1.0],
        },
    ];
    let positions = |truncation: Truncation| {
        let found = leg.find_truncated(&spectra, &truncation).unwrap();
        spectra
            .iter()
            .map(|entry| found.selection.positions(&entry.sector).unwrap())
            .collect::<Vec<_>>()
    };
    // 3 (weight 3), 2 (6), then 1.5 (33) overflows rank 6.
    assert_eq!(positions(Truncation::rank(6)), [vec![1, 2], vec![]]);
    // Budget 1: 0.5 (0.75) goes, 1.0 (27.75 in total) does not.
    assert_eq!(
        positions(Truncation::relative_error(1.0 / 127.5f64.sqrt()).unwrap()),
        [vec![1, 2], vec![0, 1]]
    );
}
