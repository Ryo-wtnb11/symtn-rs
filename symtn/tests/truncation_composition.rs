//! Truncation as a composition (#1300, #1534).
//!
//! The invariant: `svd_compact` → `diagview` → `find_truncated` →
//! `restrict_leg` is a truncated SVD, and `eigh_full` → …
//! a truncated Hermitian eigendecomposition, under every policy. These
//! compositions are the only truncated factorizations the API offers, so the
//! expectation is independent of symtn (`truncation_oracle`): each coupled
//! sector's reduced matrix is read from the input's public blocks, its
//! singular values come from a Jacobi SVD written in the test, and the kept
//! prefix from a hand implementation of the policy.
//!
//! Each case asserts in this order:
//!
//! 1. **spaces and layout** — exact: the kept bond leg against the hand
//!    selection, every factor's codomain/domain, and every factor's block
//!    geometry against the canonical layout of its homspace.
//! 2. **payloads** — within the workspace tolerance rule
//!    (`docs/testing_numerics.md`): the kept values against the oracle's
//!    magnitudes, and the gauge-free relations `t vh^H = u s`,
//!    `u^H t = s vh`, `u^H u = 1`, `vh vh^H = 1` (`t v = v d`, `v^H v = 1`
//!    for eigh), which make the kept factors genuine singular (eigen) pairs
//!    whatever basis a degenerate block was given.
//! 3. **truncation error** — against `sqrt(sum_c dim(c) sum_discarded v^2)`
//!    over the oracle's spectrum.
//!
//! ## Cross-sector exact ties
//!
//! `select_truncation` breaks an exact tie between two sectors in TensorKit's
//! sector order (#1381), sorting the feed itself when a producer hands it
//! another order (#1305). The tie fixtures below use non-negative U(1)
//! charges and SU(2) spins, whose TensorKit order is the label order the hand
//! selection breaks ties in, so the kept bond is part of the exact gate.

use std::sync::Arc;
use symtn::typed::HermitianTol;

use num_complex::{Complex32, Complex64};
use symtn::sector::{
    product_sector, FermionParityFusionRule, ProductFusionRuleExt, SU2FusionRule, SU2Irrep,
    U1FusionRule, U1Irrep, Z2Irrep,
};
use symtn::typed::{
    Eigh, GradedSpace, LegSelection, SectorSpectrum, SpectrumMagnitude, Svd, TruncatedSelection,
};
use symtn::typed::{Error, TensorMap, Truncation};

#[path = "../../tests/support/numerics.rs"]
mod numerics;
#[macro_use]
mod truncation_oracle;

use truncation_oracle::{
    assert_error_close, assert_kept_magnitudes, discarded_norm, select, ClosedFormDim, Policy,
};

#[path = "../../tests/support/fixtures.rs"]
mod fixtures;

use fixtures::{codomain_axes, domain_axes, host_runtime};

/// Asserts that the fixture really is degenerate: every value of every sector
/// is the *same bit pattern*, so a backend change cannot silently de-tie the
/// spectrum and turn the tie sweeps below into ordinary cuts.
macro_rules! assert_every_value_is_the_same_bit_pattern {
    ($tensor:expr) => {{
        let spectra = $tensor.diagview().unwrap();
        let mut all = spectra.iter().flat_map(|entry| entry.values.iter());
        let first = all.next().expect("a nonempty spectrum").to_bits();
        assert!(
            spectra
                .iter()
                .flat_map(|entry| entry.values.iter())
                .all(|value| value.to_bits() == first),
            "the fixture must be exactly degenerate for a tie test, got {spectra:?}"
        );
        assert!(
            spectra.len() > 1,
            "a cross-sector tie needs several sectors"
        );
    }};
}

/// Deterministic structureless fill, so no policy can accidentally reproduce a
/// pattern in the spectrum.
fn fill(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*state >> 33) as f64) / (u32::MAX as f64) - 0.5
}

/// `lhs` and `rhs` live on one homspace (both are `compose` results, so both
/// are canonical) and agree within the tolerance rule; `terms` is the source
/// payload length, every entry of which can reach every compared entry.
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

/// `x^H x = 1` on the bond, with `x` an isometry out of it.
macro_rules! assert_isometry {
    ($gram:expr, $bond:expr, $terms:expr, $what:expr) => {{
        let gram = $gram;
        let identity = TensorMap::isomorphism(gram.runtime(), [&$bond], [&$bond]).unwrap();
        assert_relation!(gram, identity, $terms, $what);
    }};
}

/// The whole gate for one SVD case.
macro_rules! assert_svd_composition {
    ($source:expr, $truncation:expr, $policy:expr, $case:expr) => {{
        let source = &$source;
        let truncation: &Truncation = &$truncation;
        let case: &str = $case;

        let Svd { u, s, vh } = source
            .svd_compact(&codomain_axes(&source), &domain_axes(&source))
            .unwrap();
        let bond = s.domain()[0].clone();
        let spectra = s.diagview().unwrap();
        let found = bond.find_truncated(&spectra, truncation).unwrap();
        let selection = &found.selection;
        let got_u = u.restrict_leg(&[(u.codomain_rank(), selection)]).unwrap();
        let got_s = s.restrict_leg(&[(0, selection), (1, selection)]).unwrap();
        let got_vh = vh.restrict_leg(&[(0, selection)]).unwrap();

        let offers = singular_offers!(source, ClosedFormDim::closed_form_dim);
        let kept = select(&offers, &$policy);
        let terms = source.materialize().unwrap().dense_data().unwrap().len();

        let kept_bond = got_s.domain()[0].clone();
        assert_kept_bond!(kept_bond, offers, kept, case);
        assert_eq!(got_s.codomain(), got_s.domain(), "{case}: s is a bond map");
        assert_eq!(got_u.codomain(), source.codomain(), "{case}: u codomain");
        assert_eq!(got_u.domain(), got_s.domain(), "{case}: u domain");
        assert_eq!(got_vh.codomain(), got_s.domain(), "{case}: vh codomain");
        assert_eq!(got_vh.domain(), source.domain(), "{case}: vh domain");
        assert_eq!(
            symtn::expert::diagonal_spectrum(&got_s).unwrap().is_some(),
            symtn::expert::diagonal_spectrum(&s).unwrap().is_some(),
            "{case}: restriction keeps s's storage"
        );
        assert_canonical_layout!(got_u, format!("{case}: u"));
        assert_canonical_layout!(got_s, format!("{case}: s"));
        assert_canonical_layout!(got_vh, format!("{case}: vh"));

        assert_kept_magnitudes(case, &got_s.diagview().unwrap(), &offers, terms);
        assert_relation!(
            source.compose(&got_vh.adjoint().unwrap()).unwrap(),
            got_u.compose(&got_s).unwrap(),
            terms,
            format!("{case}: t vh^H = u s")
        );
        assert_relation!(
            got_u.adjoint().unwrap().compose(source).unwrap(),
            got_s.compose(&got_vh).unwrap(),
            terms,
            format!("{case}: u^H t = s vh")
        );
        assert_isometry!(
            got_u.adjoint().unwrap().compose(&got_u).unwrap(),
            kept_bond,
            terms,
            format!("{case}: u^H u")
        );
        assert_isometry!(
            got_vh.compose(&got_vh.adjoint().unwrap()).unwrap(),
            kept_bond,
            terms,
            format!("{case}: vh vh^H")
        );
        assert_error_close(
            case,
            source.materialize().unwrap().dense_data().unwrap(),
            found.error,
            discarded_norm(&offers, &kept),
            terms,
        );
        let offered: usize = offers.iter().map(|o| o.magnitudes.len()).sum();
        assert_eq!(
            selection.is_full(),
            kept.iter().map(Vec::len).sum::<usize>() == offered,
            "{case}: is_full agrees with the kept count"
        );
        found
    }};
}

/// The whole gate for one Hermitian eigendecomposition case. The singular
/// values of a Hermitian block are its `|lambda|`; `t v = v d` with an
/// isometric `v` then pins the signs.
macro_rules! assert_eigh_composition {
    ($source:expr, $truncation:expr, $policy:expr, $case:expr) => {{
        let source = &$source;
        let truncation: &Truncation = &$truncation;
        let case: &str = $case;

        let Eigh { d, v } = source.eigh_full(&[0], &[1], HermitianTol::DEFAULT).unwrap();
        let bond = d.domain()[0].clone();
        let spectra = d.diagview().unwrap();
        let found = bond.find_truncated(&spectra, truncation).unwrap();
        let selection = &found.selection;
        let got_d = d.restrict_leg(&[(0, selection), (1, selection)]).unwrap();
        let got_v = v.restrict_leg(&[(v.codomain_rank(), selection)]).unwrap();

        let offers = singular_offers!(source, ClosedFormDim::closed_form_dim);
        let kept = select(&offers, &$policy);
        let terms = source.dense_data().unwrap().len();

        let kept_bond = got_d.domain()[0].clone();
        assert_kept_bond!(kept_bond, offers, kept, case);
        assert_eq!(got_d.codomain(), got_d.domain(), "{case}: d is a bond map");
        assert_eq!(got_v.codomain(), source.codomain(), "{case}: v codomain");
        assert_eq!(got_v.domain(), got_d.domain(), "{case}: v domain");
        assert!(
            symtn::expert::diagonal_spectrum(&got_d).unwrap().is_some(),
            "{case}: d stays compact"
        );
        assert_canonical_layout!(got_d, format!("{case}: d"));
        assert_canonical_layout!(got_v, format!("{case}: v"));

        assert_kept_magnitudes(case, &got_d.diagview().unwrap(), &offers, terms);
        assert_relation!(
            source.compose(&got_v).unwrap(),
            got_v.compose(&got_d).unwrap(),
            terms,
            format!("{case}: t v = v d")
        );
        assert_isometry!(
            got_v.adjoint().unwrap().compose(&got_v).unwrap(),
            kept_bond,
            terms,
            format!("{case}: v^H v")
        );
        assert_error_close(
            case,
            source.dense_data().unwrap(),
            found.error,
            discarded_norm(&offers, &kept),
            terms,
        );
        let offered: usize = offers.iter().map(|o| o.magnitudes.len()).sum();
        assert_eq!(
            selection.is_full(),
            kept.iter().map(Vec::len).sum::<usize>() == offered,
            "{case}: is_full agrees with the kept count"
        );
        found
    }};
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

// ---------------------------------------------------------------------------
// SVD, Host -> Host, every provider, both dtypes, every policy.
// ---------------------------------------------------------------------------

macro_rules! svd_policy_sweep {
    ($source:expr, $target:expr, $tag:expr) => {{
        let source = $source;
        for (name, truncation, policy) in policies!($target) {
            assert_svd_composition!(source, truncation, policy, &format!("{} {name}", $tag));
        }
    }};
}

macro_rules! eigh_policy_sweep {
    ($source:expr, $target:expr, $tag:expr) => {{
        let source = $source;
        for (name, truncation, policy) in policies!($target) {
            assert_eigh_composition!(source, truncation, policy, &format!("{} {name}", $tag));
        }
    }};
}

#[test]
fn u1_svd_composition_matches_the_oracle_for_every_policy() {
    let left = u1_leg(&[(-1, 2), (0, 3), (1, 2)]);
    let right = u1_leg(&[(-1, 3), (0, 2), (1, 3)]);
    let mut state = 0x1234_5678u64;
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&host_runtime(), [&left], [&right], move |_, _| {
            fill(&mut state)
        })
        .unwrap();
    svd_policy_sweep!(source, u1_leg(&[(-1, 1), (0, 2)]), "u1 f64");
}

#[test]
fn u1_complex_svd_composition_matches_the_oracle_for_every_policy() {
    let left = u1_leg(&[(-1, 2), (0, 3), (1, 2)]);
    let right = u1_leg(&[(-1, 3), (0, 2), (1, 3)]);
    let mut state = 0x2222_1111u64;
    let source: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&host_runtime(), [&left], [&right], move |_, _| {
            Complex64::new(fill(&mut state), fill(&mut state))
        })
        .unwrap();
    svd_policy_sweep!(source, u1_leg(&[(-1, 1), (0, 2)]), "u1 c64");
}

/// The same composition at single precision (#1324).
///
/// `real_spectrum` widens the payload's singular values into the `f64` the
/// decision runs on, and `SpectrumMagnitude` for `f32` / `Complex32` is
/// `f64::from(v).abs()` / `hypot(re, 0)`. This is the path the
/// single-precision `SpectrumMagnitude` impls exist for. The oracle factorizes
/// the widened input in `f64`, so its values and relations are compared at
/// `f32` tolerance; no fixture value lies within `f32` noise of a cut.
#[test]
fn u1_single_precision_svd_composition_matches_the_oracle_for_every_policy() {
    let left = u1_leg(&[(-1, 2), (0, 3), (1, 2)]);
    let right = u1_leg(&[(-1, 3), (0, 2), (1, 3)]);
    let mut state = 0x3333_4444u64;
    let source: TensorMap<_, f32> =
        TensorMap::from_subblock_fn(&host_runtime(), [&left], [&right], move |_, _| {
            fill(&mut state) as f32
        })
        .unwrap();
    svd_policy_sweep!(source, u1_leg(&[(-1, 1), (0, 2)]), "u1 f32");
}

#[test]
fn u1_complex32_svd_composition_matches_the_oracle_for_every_policy() {
    let left = u1_leg(&[(-1, 2), (0, 3), (1, 2)]);
    let right = u1_leg(&[(-1, 3), (0, 2), (1, 3)]);
    let mut state = 0x5555_6666u64;
    let source: TensorMap<_, num_complex::Complex32> =
        TensorMap::from_subblock_fn(&host_runtime(), [&left], [&right], move |_, _| {
            num_complex::Complex32::new(fill(&mut state) as f32, fill(&mut state) as f32)
        })
        .unwrap();
    svd_policy_sweep!(source, u1_leg(&[(-1, 1), (0, 2)]), "u1 c32");
}

#[test]
fn su2_svd_composition_matches_the_oracle_for_every_policy() {
    // SU(2): dim(c) = 2j+1, so the rank budget is quantum-dimension weighted
    // and cross-sector selection is not the naive value order.
    let leg = su2_leg(&[(0, 3), (1, 2), (2, 2)]);
    let mut state = 0x3333_4444u64;
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&host_runtime(), [&leg], [&leg], move |_, _| {
            fill(&mut state)
        })
        .unwrap();
    svd_policy_sweep!(source, su2_leg(&[(0, 2), (2, 1)]), "su2 f64");
}

#[test]
fn su2_complex_svd_composition_matches_the_oracle_for_every_policy() {
    let leg = su2_leg(&[(0, 3), (1, 2), (2, 2)]);
    let mut state = 0x4444_5555u64;
    let source: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&host_runtime(), [&leg], [&leg], move |_, _| {
            Complex64::new(fill(&mut state), fill(&mut state))
        })
        .unwrap();
    svd_policy_sweep!(source, su2_leg(&[(0, 2), (2, 1)]), "su2 c64");
}

#[test]
fn fermionic_svd_composition_matches_the_oracle_for_every_policy() {
    let leg = fz2_leg(&[(false, 3), (true, 3)]);
    let dual = leg.try_dual().unwrap();
    let mut state = 0x5555_6666u64;
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&host_runtime(), [&leg, &dual], [&leg], move |_, _| {
            fill(&mut state)
        })
        .unwrap();
    svd_policy_sweep!(source, fz2_leg(&[(false, 2), (true, 1)]), "fz2 f64");
}

#[test]
fn fermionic_u1_product_svd_composition_matches_the_oracle_for_every_policy() {
    let provider = Arc::new(FermionParityFusionRule.product(U1FusionRule));
    let leg = GradedSpace::try_new(
        Arc::clone(&provider),
        [
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
            (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 3),
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(2)), 2),
        ],
    )
    .unwrap();
    let target = GradedSpace::try_new(
        Arc::clone(&provider),
        [
            (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 1),
            (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 2),
        ],
    )
    .unwrap();
    let mut state = 0x6666_7777u64;
    let source: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&host_runtime(), [&leg], [&leg], move |_, _| {
            Complex64::new(fill(&mut state), fill(&mut state))
        })
        .unwrap();
    svd_policy_sweep!(source, target, "fz2xu1 c64");
}

#[test]
fn lazy_adjoint_svd_composition_matches_the_oracle() {
    let left = u1_leg(&[(0, 3), (1, 2)]);
    let right = u1_leg(&[(0, 2), (1, 3)]);
    let mut state = 0x7777_8888u64;
    let owned: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&host_runtime(), [&left], [&right], move |_, _| {
            fill(&mut state)
        })
        .unwrap();
    let lazy = owned.adjoint().unwrap();
    svd_policy_sweep!(lazy, u1_leg(&[(0, 2), (1, 1)]), "u1 lazy adjoint");
}

// ---------------------------------------------------------------------------
// Hermitian eigendecomposition.
// ---------------------------------------------------------------------------

/// A Hermitian endomorphism with a mixed-sign spectrum, so `|lambda|` order is
/// not the signed order and the selection really is magnitude-driven.
fn hermitian_u1(seed: u64) -> TensorMap<U1FusionRule, f64> {
    let leg = u1_leg(&[(-1, 2), (0, 3), (1, 2)]);
    let mut state = seed;
    let raw: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&host_runtime(), [&leg], [&leg], move |_, _| {
            fill(&mut state)
        })
        .unwrap();
    raw.axpby(1.0, &raw.adjoint().unwrap(), 1.0).unwrap()
}

#[test]
fn u1_eigh_composition_matches_the_oracle_for_every_policy() {
    let source = hermitian_u1(0x8888_9999);
    eigh_policy_sweep!(source, u1_leg(&[(-1, 1), (0, 2)]), "u1 eigh f64");
}

#[test]
fn su2_eigh_composition_matches_the_oracle_for_every_policy() {
    let leg = su2_leg(&[(0, 3), (1, 2), (2, 2)]);
    let mut state = 0x9999_aaaau64;
    let raw: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&host_runtime(), [&leg], [&leg], move |_, _| {
            Complex64::new(fill(&mut state), fill(&mut state))
        })
        .unwrap();
    let one = Complex64::new(1.0, 0.0);
    let source = raw.axpby(one, &raw.adjoint().unwrap(), one).unwrap();
    eigh_policy_sweep!(source, su2_leg(&[(0, 2), (2, 1)]), "su2 eigh c64");
}

#[test]
fn fermionic_eigh_composition_matches_the_oracle_for_every_policy() {
    let leg = fz2_leg(&[(false, 3), (true, 3)]);
    let mut state = 0xaaaa_bbbbu64;
    let raw: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&host_runtime(), [&leg], [&leg], move |_, _| {
            fill(&mut state)
        })
        .unwrap();
    let source = raw.axpby(1.0, &raw.adjoint().unwrap(), 1.0).unwrap();
    eigh_policy_sweep!(source, fz2_leg(&[(false, 2), (true, 1)]), "fz2 eigh f64");
}

// ---------------------------------------------------------------------------
// Decision shapes: whole sectors dropped, ties, discard-all, no-op.
// ---------------------------------------------------------------------------

#[test]
fn a_whole_sector_is_dropped_from_the_bond() {
    // Sector 1 carries a spectrum two orders of magnitude below sector 0, so a
    // relative cutoff removes it entirely rather than shortening it.
    let leg = u1_leg(&[(0, 3), (1, 2)]);
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&host_runtime(), [&leg], [&leg], |trees, _| {
            if trees.coupled() == &U1Irrep::new(0) {
                1.0
            } else {
                1.0e-6
            }
        })
        .unwrap();
    let found = assert_svd_composition!(
        source,
        Truncation::relative_cutoff(1e-3).unwrap(),
        Policy::RelativeCutoff(1e-3),
        "drop"
    );
    assert_eq!(
        found.selection.subspace().sectors().unwrap(),
        vec![U1Irrep::new(0)],
        "the low sector must be gone from the bond, not merely shortened"
    );
}

#[test]
fn within_sector_ties_at_the_cut_keep_genuine_singular_pairs() {
    // A scalar multiple of an isometry: every singular value inside a sector is
    // exactly equal, so `Rank` has to cut inside a run of identical values.
    let leg = u1_leg(&[(0, 3), (1, 3)]);
    let source: TensorMap<_, f64> = TensorMap::isomorphism(&host_runtime(), [&leg], [&leg])
        .unwrap()
        .scale(2.5)
        .unwrap();
    assert_every_value_is_the_same_bit_pattern!(source.svd_compact(&[0], &[1]).unwrap().s);
    for rank in [1usize, 2, 3, 4, 5] {
        assert_svd_composition!(
            source,
            Truncation::rank(rank),
            Policy::Rank(rank),
            &format!("tie rank {rank}")
        );
    }
}

#[test]
fn cross_sector_exact_ties_are_broken_in_tensorkit_sector_order() {
    // Two U(1) sectors with bit-identical spectra: every `Rank` cut that is not
    // a multiple of the sector count lands on an exact cross-sector tie.
    let leg = u1_leg(&[(0, 3), (1, 3), (2, 3)]);
    let source: TensorMap<_, f64> =
        TensorMap::isomorphism(&host_runtime(), [&leg], [&leg]).unwrap();
    assert_every_value_is_the_same_bit_pattern!(source.svd_compact(&[0], &[1]).unwrap().s);
    for rank in 0..=9usize {
        assert_svd_composition!(
            source,
            Truncation::rank(rank),
            Policy::Rank(rank),
            &format!("cross tie rank {rank}")
        );
    }
}

#[test]
fn su2_cross_sector_exact_ties_are_broken_in_tensorkit_sector_order() {
    // Same, with dim(c) != 1: the weighted budget overflows mid-tie.
    let leg = su2_leg(&[(0, 2), (1, 2), (2, 2)]);
    let source: TensorMap<_, f64> =
        TensorMap::isomorphism(&host_runtime(), [&leg], [&leg]).unwrap();
    assert_every_value_is_the_same_bit_pattern!(source.svd_compact(&[0], &[1]).unwrap().s);
    for rank in 0..=12usize {
        assert_svd_composition!(
            source,
            Truncation::rank(rank),
            Policy::Rank(rank),
            &format!("su2 cross tie rank {rank}")
        );
    }
}

#[test]
fn signed_cross_sector_eigenvalue_ties_are_broken_in_tensorkit_sector_order() {
    // `|lambda|` ties across sectors with *opposite signs*: the selection is
    // magnitude-driven, so +2 in one sector and -2 in another are an exact tie
    // that only the slice order can break, and the published `eigh_full` order
    // is by descending |lambda|, not by signed value.
    let leg = u1_leg(&[(0, 2), (1, 2), (2, 2)]);
    let runtime = host_runtime();
    let identity: TensorMap<_, f64> = TensorMap::isomorphism(&runtime, [&leg], [&leg]).unwrap();
    // diag(+2, -2) in every sector: three sectors x two magnitudes, all equal.
    // Annotated: the payload dtype is taken solely from float literals, and
    // `f32` reaching `FactorizationScalar` (#1324) means `eigh_full` does not
    // pin `{float}` to `f64` before `value.abs()` below resolves.
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
            if indices[0] != indices[1] {
                0.0
            } else if indices[0] == 0 {
                2.0
            } else {
                -2.0
            }
        })
        .unwrap();
    assert_eq!(identity.subblock_count(), source.subblock_count());
    let magnitudes: Vec<f64> = source
        .eigh_full(&[0], &[1], HermitianTol::DEFAULT)
        .unwrap()
        .d
        .diagview()
        .unwrap()
        .iter()
        .flat_map(|entry| entry.values.iter().map(|value| value.abs()))
        .collect();
    assert!(
        magnitudes
            .iter()
            .all(|value| value.to_bits() == 2.0f64.to_bits()),
        "every |lambda| must be exactly 2.0 for this to be a tie test, got {magnitudes:?}"
    );
    for rank in 0..=6usize {
        assert_eigh_composition!(
            source,
            Truncation::rank(rank),
            Policy::Rank(rank),
            &format!("signed cross tie rank {rank}")
        );
    }
}

#[test]
fn find_truncated_ignores_the_order_the_spectra_arrive_in() {
    // The documented contract: input order is irrelevant because the spectra
    // are ordered by `SectorId` before the decision. Reversing the input must
    // therefore change neither the subspace nor the error bits — and an exact
    // cross-sector tie is where it would show if the sort were missing.
    let leg = u1_leg(&[(0, 3), (1, 3), (2, 3)]);
    let source: TensorMap<_, f64> =
        TensorMap::isomorphism(&host_runtime(), [&leg], [&leg]).unwrap();
    let Svd { s, .. } = source.svd_compact(&[0], &[1]).unwrap();
    let bond = s.domain()[0].clone();
    let canonical = s.diagview().unwrap();
    let mut reversed = canonical.clone();
    reversed.reverse();
    assert_ne!(
        canonical.iter().map(|e| e.sector).collect::<Vec<_>>(),
        reversed.iter().map(|e| e.sector).collect::<Vec<_>>(),
        "the permuted input must actually differ"
    );
    for rank in 0..=9usize {
        let truncation = Truncation::rank(rank);
        let want = bond.find_truncated(&canonical, &truncation).unwrap();
        let got = bond.find_truncated(&reversed, &truncation).unwrap();
        assert_eq!(
            got.selection.subspace(),
            want.selection.subspace(),
            "rank {rank}"
        );
        assert_eq!(got.error.to_bits(), want.error.to_bits(), "rank {rank}");
        assert_eq!(
            s.restrict_leg(&[(0, &got.selection), (1, &got.selection)])
                .unwrap()
                .materialize()
                .unwrap()
                .dense_data()
                .unwrap(),
            s.restrict_leg(&[(0, &want.selection), (1, &want.selection)])
                .unwrap()
                .materialize()
                .unwrap()
                .dense_data()
                .unwrap(),
            "rank {rank}"
        );
    }
}

#[test]
fn a_dense_bond_map_restricted_on_both_legs_equals_two_one_axis_calls() {
    // The two-axis set restricts both axes in one kernel call. Its oracle is
    // the one-axis restriction applied twice.
    let leg = u1_leg(&[(0, 4), (1, 3)]);
    let runtime = host_runtime();
    let mut state = 0x1357_9bdfu64;
    let dense: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], move |_, _| fill(&mut state))
            .unwrap();
    assert!(
        symtn::expert::diagonal_spectrum(&dense).unwrap().is_none(),
        "this fixture must exercise the dense arm"
    );
    for pairs in [
        vec![(U1Irrep::new(0), 0..2), (U1Irrep::new(1), 0..1)],
        // A non-prefix range: `find_truncated` never produces one, but
        // the compact-preserving form must not assume a leading prefix.
        vec![(U1Irrep::new(0), 1..4), (U1Irrep::new(1), 2..3)],
        // A sector dropped entirely.
        vec![(U1Irrep::new(1), 1..3)],
    ] {
        let selection = LegSelection::try_new(&leg, pairs.clone()).unwrap();
        let once = dense
            .restrict_leg(&[(0, &selection), (1, &selection)])
            .unwrap();
        let twice = dense
            .restrict_leg(&[(0, &selection)])
            .unwrap()
            .restrict_leg(&[(1, &selection)])
            .unwrap();
        assert_eq!(once.codomain(), twice.codomain(), "{pairs:?}");
        assert_eq!(once.domain(), twice.domain(), "{pairs:?}");
        assert_eq!(
            once.dense_data().unwrap(),
            twice.dense_data().unwrap(),
            "{pairs:?}"
        );
    }
}

#[test]
fn discarding_everything_yields_the_empty_bond_and_empty_factors() {
    let leg = u1_leg(&[(0, 3), (1, 2)]);
    let mut state = 0xbbbb_ccccu64;
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&host_runtime(), [&leg], [&leg], move |_, _| {
            fill(&mut state)
        })
        .unwrap();

    let found =
        assert_svd_composition!(source, Truncation::rank(0), Policy::Rank(0), "discard-all");
    let selection = &found.selection;
    assert!(
        selection.subspace().sectors().unwrap().is_empty(),
        "discard-all must produce the empty bond leg"
    );
    assert!(!selection.is_full(), "an empty selection of a nonempty leg");

    let Svd { u, s, vh } = source.svd_compact(&[0], &[1]).unwrap();
    let got_u = u.restrict_leg(&[(u.codomain_rank(), selection)]).unwrap();
    let got_s = s.restrict_leg(&[(0, selection), (1, selection)]).unwrap();
    let got_vh = vh.restrict_leg(&[(0, selection)]).unwrap();
    assert!(
        got_u.dense_data().unwrap().is_empty()
            && got_s
                .materialize()
                .unwrap()
                .dense_data()
                .unwrap()
                .is_empty()
            && got_vh.dense_data().unwrap().is_empty()
    );

    // `embed_leg` with the empty selection: the adjoint of restricting to
    // nothing is the zero map back onto the parent leg.
    let embedded = got_u.embed_leg(got_u.codomain_rank(), selection).unwrap();
    assert_eq!(embedded.domain()[0], s.domain()[0]);
    assert!(embedded
        .dense_data()
        .unwrap()
        .iter()
        .all(|&value| value == 0.0));
}

#[test]
fn a_no_op_decision_is_reported_as_full_and_copies_the_same_bits() {
    let leg = u1_leg(&[(0, 3), (1, 2)]);
    let mut state = 0xcccc_ddddu64;
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&host_runtime(), [&leg], [&leg], move |_, _| {
            fill(&mut state)
        })
        .unwrap();
    let Svd { u, s, .. } = source.svd_compact(&[0], &[1]).unwrap();
    let bond = s.domain()[0].clone();
    let found = bond
        .find_truncated(&s.diagview().unwrap(), &Truncation::Full)
        .unwrap();
    assert!(found.selection.is_full());
    assert_eq!(found.error.to_bits(), 0.0_f64.to_bits());
    assert_eq!(
        u.restrict_leg(&[(u.codomain_rank(), &found.selection)])
            .unwrap()
            .dense_data()
            .unwrap(),
        u.dense_data().unwrap(),
        "a full selection copies the payload unchanged"
    );
}

#[test]
fn is_full_holds_for_the_only_selection_of_an_empty_leg() {
    let empty = u1_leg(&[]);
    let found = empty
        .find_truncated(&[] as &[SectorSpectrum<U1Irrep, f64>], &Truncation::rank(5))
        .unwrap();
    assert!(found.selection.is_full());
    assert_eq!(found.error.to_bits(), 0.0_f64.to_bits());
}

// ---------------------------------------------------------------------------
// diagview
// ---------------------------------------------------------------------------

#[test]
fn diagview_reads_the_same_values_from_compact_and_dense_storage() {
    let leg = u1_leg(&[(0, 3), (1, 2)]);
    let runtime = host_runtime();
    let mut state = 0xdddd_eeeeu64;
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], move |_, _| fill(&mut state))
            .unwrap();
    let Svd { s, .. } = source.svd_compact(&[0], &[1]).unwrap();
    assert!(
        symtn::expert::diagonal_spectrum(&s).unwrap().is_some(),
        "s is compact"
    );
    let compact = s.diagview().unwrap();

    // The same map, forced through dense storage. `add` of a compact factor and
    // a dense zero produces the dense twin (`diagonal_spectrum` is then `None`,
    // which is the contract `diagview` deliberately does not change).
    let zero: TensorMap<_, f64> = TensorMap::zeros(&runtime, [&leg], [&leg]).unwrap();
    let dense = zero.axpby(1.0, &s, 1.0).unwrap();
    assert!(
        symtn::expert::diagonal_spectrum(&dense).unwrap().is_none(),
        "the dense twin must still report no compact storage"
    );
    assert_eq!(dense.diagview().unwrap(), compact);

    // Off-diagonal entries are not inspected: a dense map with a nonzero
    // off-diagonal still reports its diagonal.
    let full: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, indices| {
            if indices[0] == indices[1] {
                (indices[0] + 1) as f64
            } else {
                7.0
            }
        })
        .unwrap();
    let read = full.diagview().unwrap();
    assert_eq!(read.len(), 2);
    for entry in &read {
        let expected: Vec<f64> = (0..entry.values.len()).map(|i| (i + 1) as f64).collect();
        assert_eq!(entry.values, expected);
    }
}

#[test]
fn diagview_reads_a_rectangular_map_and_rejects_a_non_bond_map_and_a_lazy_adjoint() {
    let leg = u1_leg(&[(0, 2), (1, 2)]);
    let other = u1_leg(&[(0, 3), (1, 1)]);
    let runtime = host_runtime();

    let rank_three: TensorMap<_, f64> = TensorMap::zeros(&runtime, [&leg, &leg], [&leg]).unwrap();
    assert!(matches!(
        rank_three.diagview(),
        Err(Error::InvalidArgument(message)) if message.contains("rank-(1,1)")
    ));

    // A rectangular one-leg map is MatrixAlgebraKit's `diagview` input too
    // (#1984): each block gives its leading `min(m_c, n_c)` diagonal.
    let rectangular: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&other], |_, indices| {
            if indices[0] == indices[1] {
                (indices[0] + 1) as f64
            } else {
                7.0
            }
        })
        .unwrap();
    let read = rectangular.diagview().unwrap();
    assert_eq!(read.len(), 2);
    assert_eq!(read[0].values, [1.0, 2.0]);
    assert_eq!(read[1].values, [1.0]);

    let square: TensorMap<_, f64> = TensorMap::zeros(&runtime, [&leg], [&leg]).unwrap();
    assert!(matches!(
        square.adjoint().unwrap().diagview(),
        Err(Error::InvalidArgument(message)) if message.contains("lazy adjoint")
    ));
}

// ---------------------------------------------------------------------------
// multi-axis restrict_leg preconditions
// ---------------------------------------------------------------------------

#[test]
fn multi_axis_restrict_leg_rejects_everything_it_documents() {
    let leg = u1_leg(&[(0, 3), (1, 2)]);
    let runtime = host_runtime();
    let selection = LegSelection::try_new(&leg, [(U1Irrep::new(0), 0..2)]).unwrap();

    let square: TensorMap<_, f64> = TensorMap::zeros(&runtime, [&leg], [&leg]).unwrap();
    assert!(matches!(
        square.restrict_leg(&[]),
        Err(Error::InvalidArgument(message)) if message.contains("at least one")
    ));
    assert!(matches!(
        square.restrict_leg(&[(1, &selection), (0, &selection), (1, &selection)]),
        Err(Error::InvalidArgument(message)) if message.contains("axis 1 appears twice")
    ));
    assert!(matches!(
        square.restrict_leg(&[(0, &selection), (2, &selection)]),
        Err(Error::InvalidArgument(message)) if message.contains("axis 2 is out of range")
    ));

    // Each pair keeps its one-axis error, whichever position it has.
    let other = u1_leg(&[(0, 3), (1, 1)]);
    let rectangular: TensorMap<_, f64> = TensorMap::zeros(&runtime, [&leg], [&other]).unwrap();
    let alone = rectangular.restrict_leg(&[(1, &selection)]).err().unwrap();
    let in_set = rectangular
        .restrict_leg(&[(0, &selection), (1, &selection)])
        .err()
        .unwrap();
    assert!(matches!(
        &alone,
        Error::InvalidArgument(message) if message.contains("not the leg this selection was built from")
    ));
    assert_eq!(format!("{alone:?}"), format!("{in_set:?}"));

    // A dense lazy adjoint is read in place: equal to restricting its
    // materialization, bit for bit.
    let mut state = 0x2468_ace0u64;
    let dense: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], move |_, _| fill(&mut state))
            .unwrap();
    let lazy = dense.adjoint().unwrap();
    let both = [(0, &selection), (1, &selection)];
    assert_eq!(
        lazy.restrict_leg(&both).unwrap().dense_data().unwrap(),
        lazy.materialize()
            .unwrap()
            .restrict_leg(&both)
            .unwrap()
            .dense_data()
            .unwrap()
    );

    // A selection from a different *rule* cannot be spelled: `LegSelection<R>`
    // is typed by the provider, so the `RuleMismatch` guard only fires for
    // two providers of the same type with different identities, which is the
    // one-axis case already tested.
}

#[test]
fn one_selection_on_both_legs_keeps_a_compact_payload_compact() {
    let leg = u1_leg(&[(0, 4), (1, 3)]);
    let mut state = 0xeeee_ffffu64;
    let source: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&host_runtime(), [&leg], [&leg], move |_, _| {
            fill(&mut state)
        })
        .unwrap();
    let Svd { s, .. } = source.svd_compact(&[0], &[1]).unwrap();
    let bond = s.domain()[0].clone();
    let selection = LegSelection::try_new(&bond, [(U1Irrep::new(0), 0..2)]).unwrap();
    let restricted = s.restrict_leg(&[(0, &selection), (1, &selection)]).unwrap();
    let spectrum = symtn::expert::diagonal_spectrum(&restricted)
        .unwrap()
        .expect("a compact input must stay compact");
    assert_eq!(spectrum.len(), 1);
    assert_eq!(spectrum[0].values.len(), 2);
    assert_eq!(
        spectrum[0].values,
        s.diagview().unwrap()[0].values[0..2].to_vec()
    );
}

// ---------------------------------------------------------------------------
// find_truncated input validation
// ---------------------------------------------------------------------------

fn spectrum(charge: i32, values: &[f64]) -> SectorSpectrum<U1Irrep, f64> {
    SectorSpectrum {
        sector: U1Irrep::new(charge),
        values: values.to_vec(),
    }
}

#[test]
fn find_truncated_rejects_a_malformed_input_spectrum() {
    let leg = u1_leg(&[(0, 3), (1, 2)]);
    let full = Truncation::Full;

    let missing = [spectrum(0, &[3.0, 2.0, 1.0])];
    assert!(matches!(
        leg.find_truncated(&missing, &full),
        Err(Error::InvalidArgument(message)) if message.contains("one spectrum per leg sector")
    ));

    let duplicated = [spectrum(0, &[3.0, 2.0, 1.0]), spectrum(0, &[3.0, 2.0, 1.0])];
    assert!(matches!(
        leg.find_truncated(&duplicated, &full),
        Err(Error::InvalidArgument(message)) if message.contains("more than one spectrum")
    ));

    let unknown = [spectrum(0, &[3.0, 2.0, 1.0]), spectrum(7, &[2.0, 1.0])];
    assert!(matches!(
        leg.find_truncated(&unknown, &full),
        Err(Error::InvalidArgument(message)) if message.contains("absent from the truncated leg")
    ));

    for wrong in [&[3.0, 2.0][..], &[3.0, 2.0, 1.0, 0.5][..]] {
        let bad = [spectrum(0, wrong), spectrum(1, &[2.0, 1.0])];
        assert!(matches!(
            leg.find_truncated(&bad, &full),
            Err(Error::InvalidArgument(message)) if message.contains("expected the leg degeneracy")
        ));
    }

    for values in [&[f64::NAN, 2.0, 1.0][..], &[f64::INFINITY, 2.0, 1.0][..]] {
        let bad = [spectrum(0, values), spectrum(1, &[2.0, 1.0])];
        assert!(
            leg.find_truncated(&bad, &full).is_err(),
            "NaN and infinite spectra must be rejected: {values:?}"
        );
    }
    // Stored order is not malformed (#2095): an ascending spectrum is a
    // valid input and keeps everything under `Full`.
    let ascending = [spectrum(0, &[1.0, 2.0, 3.0]), spectrum(1, &[2.0, 1.0])];
    assert!(leg
        .find_truncated(&ascending, &full)
        .unwrap()
        .selection
        .is_full());
}

#[test]
fn find_truncated_rejects_a_truncation_space_from_another_rule() {
    let leg = u1_leg(&[(0, 3), (1, 2)]);
    let foreign = su2_leg(&[(0, 1)]);
    let spectra = [spectrum(0, &[3.0, 2.0, 1.0]), spectrum(1, &[2.0, 1.0])];
    assert!(leg
        .find_truncated(&spectra, &Truncation::space(foreign.truncspace()))
        .is_err());
}

#[test]
fn a_selection_from_another_leg_is_rejected_by_both_appliers() {
    let leg = u1_leg(&[(0, 3), (1, 2)]);
    let other = u1_leg(&[(0, 4), (1, 2)]);
    let runtime = host_runtime();
    let selection = LegSelection::try_new(&other, [(U1Irrep::new(0), 0..2)]).unwrap();
    let tensor: TensorMap<_, f64> = TensorMap::zeros(&runtime, [&leg], [&leg]).unwrap();
    assert!(tensor
        .restrict_leg(&[(0, &selection), (1, &selection)])
        .is_err());
    assert!(tensor.restrict_leg(&[(0, &selection)]).is_err());
}

/// A caller generic over the payload can name every bound `find_truncated`
/// needs: `SpectrumMagnitude` is re-exported from `symtn::typed`, so the
/// device recipe does not have to map the spectrum to `f64` magnitudes first
/// (#1300 left it unnameable outside the crate; #1297 exports it).
fn find_truncated_generically<R, D>(
    s: &TensorMap<R, D>,
    truncation: &Truncation,
) -> TruncatedSelection<R>
where
    R: symtn::sector::MultiplicityFreeRigidSymbols<Scalar = f64>
        + symtn::sector::CheckedFusionAlgebra
        + symtn::sector::SectorCodec,
    D: symtn::typed::TensorScalar + SpectrumMagnitude,
{
    s.domain()[0]
        .find_truncated(&s.diagview().unwrap(), truncation)
        .unwrap()
}

#[test]
fn a_payload_generic_caller_can_name_the_find_truncated_bound() {
    let runtime = host_runtime();
    let leg = u1_leg(&[(0, 3), (1, 2)]);
    let truncation = Truncation::rank(2);

    let mut state = 7;
    let real: TensorMap<_, f64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, _| fill(&mut state)).unwrap();
    let Svd { s, .. } = real.svd_compact(&[0], &[1]).unwrap();
    let found = find_truncated_generically(&s, &truncation);
    let want = assert_svd_composition!(real, truncation, Policy::Rank(2), "f64");
    assert_eq!(
        found.selection.subspace(),
        want.selection.subspace(),
        "f64 kept bond"
    );
    assert_eq!(found.error.to_bits(), want.error.to_bits(), "f64 error");

    let complex: TensorMap<_, Complex64> =
        TensorMap::from_subblock_fn(&runtime, [&leg], [&leg], |_, _| {
            Complex64::new(fill(&mut state), fill(&mut state))
        })
        .unwrap();
    let Svd { s, .. } = complex.svd_compact(&[0], &[1]).unwrap();
    let found = find_truncated_generically(&s, &truncation);
    let want = assert_svd_composition!(complex, truncation, Policy::Rank(2), "c64");
    assert_eq!(
        found.selection.subspace(),
        want.selection.subspace(),
        "c64 kept bond"
    );
    assert_eq!(found.error.to_bits(), want.error.to_bits(), "c64 error");
}
