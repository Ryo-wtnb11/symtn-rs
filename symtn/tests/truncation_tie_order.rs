//! Exact cross-sector ties keep the sector TensorKit keeps (#1381).
//!
//! TensorKit 0.17.1 `findtruncated(::SectorVector, ...)`
//! (`src/factorizations/truncation.jl`) sorts the flat `parent(values)` with
//! `sortperm`/`partialsortperm`, whose ties go to the lower flat index. The
//! flat order is the `SectorVector`'s block order, which is sorted by
//! TensorKitSectors `isless`. So `truncrank` keeps the `isless`-earlier sector
//! first and `truncerror` discards it first.
//!
//! Every expected count below was produced by running TensorKit 0.17.1 /
//! TensorKitSectors 0.3.9 `MAK.findtruncated` (and `svd_trunc`/`eigh_trunc`
//! for the end-to-end cases) on the same fixture. It also follows by hand from
//! the `isless` order quoted beside each case. No expectation is read back from
//! symtn.
//!
//! Ties *within* a sector (#2095) go to the lower position in the same flat
//! order, so `truncrank` keeps the earlier position and `truncerror` discards
//! it. Those expected positions were produced by TensorKit `cfaa073e`
//! (0.17.0) with MatrixAlgebraKit 0.6.8, `MAK.findtruncated` over a
//! `SectorVector` holding the fixture's values in the listed (stored) order.

use std::sync::Arc;
use symtn::typed::HermitianTol;

use symtn::sector::{
    product_sector, FermionParityFusionRule, SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep,
    Z2FusionRule, Z2Irrep, ZNFusionRule,
};
use symtn::sector::{CheckedGenericFusion, FusionRule, SectorOrderKey};
use symtn::typed::{Eigh, GradedSpace, SectorSpectrum, Svd};
use symtn::typed::{Runtime, TensorMap, Truncation};
use symtn_core::InfallibleGeneric;

/// Kept count per entry, in entry order, from `GradedSpace::find_truncated`.
macro_rules! kept {
    ($rule:expr, [$(($sector:expr, [$($value:expr),*])),* $(,)?], $truncation:expr) => {{
        let entries = vec![$(($sector, vec![$($value),*])),*];
        let leg = GradedSpace::try_new(
            Arc::new($rule),
            entries.iter().map(|(sector, values)| (sector.clone(), values.len())),
        )
        .unwrap();
        let spectra: Vec<SectorSpectrum<_>> = entries
            .iter()
            .map(|(sector, values)| SectorSpectrum {
                sector: sector.clone(),
                values: values.clone(),
            })
            .collect();
        let found = leg.find_truncated(&spectra, &$truncation).unwrap();
        entries
            .iter()
            .map(|(sector, _)| found.selection.subspace().degeneracy(sector).unwrap())
            .collect::<Vec<usize>>()
    }};
}

/// Kept positions per entry, in entry order, from `GradedSpace::find_truncated`
/// and `LegSelection::positions`.
macro_rules! positions {
    ($rule:expr, [$(($sector:expr, [$($value:expr),*])),* $(,)?], $truncation:expr) => {{
        let entries = vec![$(($sector, vec![$($value),*])),*];
        let leg = GradedSpace::try_new(
            Arc::new($rule),
            entries.iter().map(|(sector, values)| (sector.clone(), values.len())),
        )
        .unwrap();
        let spectra: Vec<SectorSpectrum<_>> = entries
            .iter()
            .map(|(sector, values)| SectorSpectrum {
                sector: sector.clone(),
                values: values.clone(),
            })
            .collect();
        let found = leg.find_truncated(&spectra, &$truncation).unwrap();
        entries
            .iter()
            .map(|(sector, _)| found.selection.positions(sector).unwrap())
            .collect::<Vec<Vec<usize>>>()
    }};
}

/// `relative_error` with symtn budget `(rtol * norm)^2 = atol^2`, i.e.
/// TensorKit's `truncerror(; atol)`, for a spectrum of weighted squared norm
/// `norm_squared`.
fn truncerror_atol(atol: f64, norm_squared: f64) -> Truncation {
    Truncation::relative_error(atol / norm_squared.sqrt()).unwrap()
}

fn u1(charge: i32) -> U1Irrep {
    U1Irrep::new(charge)
}

/// symtn's `relative_error(rtol)` budget is `(rtol * norm)^2`; this `rtol`
/// gives the fixture's budget `0.12^2`, TensorKit's `truncerror(; atol = 0.12)`.
/// One `0.1` fits in it, two do not.
fn one_small_value_budget() -> Truncation {
    Truncation::relative_error(0.12 / 1.02_f64.sqrt()).unwrap()
}

#[test]
fn u1_opposite_charge_tie_keeps_the_positive_charge_under_rank() {
    // isless: 0 < +1 < -1. symtn ids are 0, 2, 1, so id order kept -1.
    let kept = kept!(
        U1FusionRule,
        [(u1(0), [3.0]), (u1(1), [2.0]), (u1(-1), [2.0])],
        Truncation::rank(2)
    );
    assert_eq!(kept, [1, 1, 0]);
}

#[test]
fn u1_opposite_charge_tie_discards_the_positive_charge_under_discard_weight() {
    // Ascending stable sort discards the isless-earlier +1 first.
    let kept = kept!(
        U1FusionRule,
        [(u1(0), [1.0]), (u1(1), [0.1]), (u1(-1), [0.1])],
        one_small_value_budget()
    );
    assert_eq!(kept, [1, 0, 1]);
}

#[test]
fn u1_tie_inside_and_across_sectors_follows_the_flat_order() {
    // Sorted: 3 (0), 2 (+1), 2 (-1), 1 (+1), 1 (-1).
    let kept = kept!(
        U1FusionRule,
        [(u1(0), [3.0]), (u1(1), [2.0, 1.0]), (u1(-1), [2.0, 1.0])],
        Truncation::rank(4)
    );
    assert_eq!(kept, [1, 2, 1]);
}

#[test]
fn u1_x_z2_opposite_charge_tie_follows_the_product_order() {
    // Product isless over findindex - 1: (+1, 0) -> (3, 0), (-1, 0) -> (4, 0).
    let rule =
        || symtn_sectors::ProductFusionRule::<_, _>::try_new(U1FusionRule, Z2FusionRule).unwrap();
    let sector = |charge| product_sector(u1(charge), Z2Irrep::EVEN);
    let rank = kept!(
        rule(),
        [(sector(0), [3.0]), (sector(1), [2.0]), (sector(-1), [2.0])],
        Truncation::rank(2)
    );
    assert_eq!(rank, [1, 1, 0]);
    let error = kept!(
        rule(),
        [(sector(0), [1.0]), (sector(1), [0.1]), (sector(-1), [0.1])],
        one_small_value_budget()
    );
    assert_eq!(error, [1, 0, 1]);
}

#[test]
fn u1_x_u1_tie_uses_tensorkit_positions_not_integer_ranks() {
    // TensorKit's U(1) positions interleave half-integers: +1 -> 3, -1 -> 4.
    // (-1, 0) -> degree 4 precedes (+1, +1) -> degree 6. Ranks over integer
    // charges alone (+1 -> 1, -1 -> 2) would tie the degrees at 2 and put
    // (+1, +1) first.
    let kept = kept!(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(U1FusionRule, U1FusionRule).unwrap(),
        [
            (product_sector(u1(1), u1(1)), [1.0]),
            (product_sector(u1(-1), u1(0)), [1.0]),
        ],
        Truncation::rank(1)
    );
    assert_eq!(kept, [0, 1]);
}

#[test]
fn nested_product_tie_follows_the_flattened_tensorkit_order() {
    // TensorKit flattens fP ⊠ SU(2) ⊠ fP: (even, 1, even) -> (0, 2, 0) and
    // (odd, 0, odd) -> (1, 0, 1) both have degree 2, and (0, 2, 0) is
    // lexicographically first. Comparing the nested pair ((fP, SU2), fP) would
    // put (odd, 0, odd) first. The spin-1 sector weighs 3, so rank 3 keeps
    // exactly the sector that comes first.
    let rule = symtn_sectors::ProductFusionRule::<_, _>::try_new(
        symtn_sectors::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, SU2FusionRule)
            .unwrap(),
        FermionParityFusionRule,
    )
    .unwrap();
    let kept = kept!(
        rule,
        [
            (
                product_sector(
                    product_sector(Z2Irrep::EVEN, SU2Irrep::from_twice_spin(2)),
                    Z2Irrep::EVEN
                ),
                [1.0]
            ),
            (
                product_sector(
                    product_sector(Z2Irrep::ODD, SU2Irrep::from_twice_spin(0)),
                    Z2Irrep::ODD
                ),
                [1.0]
            ),
        ],
        Truncation::rank(3)
    );
    assert_eq!(kept, [1, 0]);
}

#[test]
fn su2_tie_keeps_the_smaller_spin_first() {
    // isless: 1/2 < 1. Weights 2 and 3: rank 3 fits only the first.
    let kept = kept!(
        SU2FusionRule,
        [
            (SU2Irrep::from_twice_spin(1), [1.0]),
            (SU2Irrep::from_twice_spin(2), [1.0]),
        ],
        Truncation::rank(3)
    );
    assert_eq!(kept, [1, 0]);
}

#[test]
fn zn_tie_keeps_the_smaller_charge_first() {
    let rule = ZNFusionRule::new(3).unwrap();
    let kept = kept!(
        rule.clone(),
        [
            (rule.irrep(0), [3.0]),
            (rule.irrep(1), [2.0]),
            (rule.irrep(2), [2.0]),
        ],
        Truncation::rank(2)
    );
    assert_eq!(kept, [1, 1, 0]);
}

#[test]
fn untied_truncation_is_decided_by_magnitude_alone() {
    let kept = kept!(
        U1FusionRule,
        [
            (u1(0), [3.0]),
            (u1(1), [1.5]),
            (u1(-1), [2.0]),
            (u1(2), [1.0]),
        ],
        Truncation::rank(2)
    );
    assert_eq!(kept, [1, 0, 1, 0]);
}

#[test]
fn svd_and_eigh_truncation_keep_tensorkits_sector_at_a_tie() {
    // TensorKit: svd_trunc/eigh_trunc(id(Rep[U1](0 => 1, 1 => 1, -1 => 1)),
    // truncrank(2)) both return Rep[U1](0 => 1, 1 => 1).
    let runtime = Runtime::builder().dense_threads(1).build().unwrap();
    let leg = GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [(u1(0), 1), (u1(1), 1), (u1(-1), 1)],
    )
    .unwrap();
    let source: TensorMap<U1FusionRule, f64> =
        TensorMap::isomorphism(&runtime, [&leg], [&leg]).unwrap();
    let expected = GradedSpace::try_new(Arc::new(U1FusionRule), [(u1(0), 1), (u1(1), 1)]).unwrap();
    // The truncated factorizations are `*_full`/`svd_compact` -> `diagview` ->
    // `find_truncated` -> `restrict_*`; the kept bond is the restricted leg.
    let Svd { u, s, .. } = source.svd_compact(&[0], &[1]).unwrap();
    let found = s.domain()[0]
        .find_truncated(&s.diagview().unwrap(), &Truncation::rank(2))
        .unwrap();
    let u = u
        .restrict_leg(&[(u.codomain_rank(), &found.selection)])
        .unwrap();
    assert_eq!(u.domain()[0], expected);
    let Eigh { d, .. } = source.eigh_full(&[0], &[1], HermitianTol::DEFAULT).unwrap();
    let found = d.domain()[0]
        .find_truncated(&d.diagview().unwrap(), &Truncation::rank(2))
        .unwrap();
    let d = d
        .restrict_leg(&[(0, &found.selection), (1, &found.selection)])
        .unwrap();
    assert_eq!(d.domain()[0], expected);
}

#[test]
fn u1_order_keys_are_tensorkit_findindex_positions() {
    let keys: Vec<SectorOrderKey> = [0, 1, -1, 2, -2]
        .map(|charge| U1FusionRule.sector_order_key(u1(charge).into()))
        .into();
    let expected: Vec<SectorOrderKey> = [0, 3, 4, 7, 8].map(SectorOrderKey::position).into();
    assert_eq!(keys, expected);
    // The checked-Generic truncation path reads the key through this adapter.
    assert_eq!(
        CheckedGenericFusion::sector_order_key(
            &InfallibleGeneric::new(&U1FusionRule),
            u1(-1).into()
        ),
        SectorOrderKey::position(4)
    );
}

#[test]
fn u1_ties_within_and_across_sectors_discard_the_earliest_flat_entry() {
    // Flat ascending order of the three 0.1s: (0, 1), (0, 2), (+1, 0). The
    // budget 0.12^2 fits one, so TensorKit discards position 1 of charge 0.
    let kept = positions!(
        U1FusionRule,
        [(u1(0), [1.0, 0.1, 0.1]), (u1(1), [0.1])],
        truncerror_atol(0.12, 1.03)
    );
    assert_eq!(kept, [vec![0, 2], vec![0]]);
}

#[test]
fn su2_within_sector_tie_discards_the_earlier_position_with_dim_weights() {
    // Spin 0 (weight 1) holds both 0.1s; spin 1/2 weighs 2. norm^2 = 3.02.
    // Discard (0, 0) for 0.01; (0, 2) would bring 0.02 > 0.0144.
    let kept = positions!(
        SU2FusionRule,
        [
            (SU2Irrep::from_twice_spin(0), [0.1, 1.0, 0.1]),
            (SU2Irrep::from_twice_spin(1), [1.0]),
        ],
        truncerror_atol(0.12, 3.02)
    );
    assert_eq!(kept, [vec![1, 2], vec![0]]);
}

#[test]
fn rank_keeps_the_largest_magnitudes_wherever_they_are_stored() {
    // eigh-like ascending storage: the 2.0 of charge 0 is its last entry.
    let kept = positions!(
        U1FusionRule,
        [(u1(0), [0.5, 2.0]), (u1(1), [2.0, 1.0])],
        Truncation::rank(2)
    );
    assert_eq!(kept, [vec![1], vec![0]]);
    // Signed values select by |v|: 3 (0, 2), then the 2s in flat order,
    // (0, 3) before (+1, 0) before (+1, 1).
    let kept = positions!(
        U1FusionRule,
        [(u1(0), [-1.0, 0.5, 3.0, -2.0]), (u1(1), [2.0, -2.0])],
        Truncation::rank(3)
    );
    assert_eq!(kept, [vec![2, 3], vec![0]]);
    // SU(2): 2.0 (spin 0, weight 1) then 2.0 (spin 1/2, weight 2) fill rank 3.
    let kept = positions!(
        SU2FusionRule,
        [
            (SU2Irrep::from_twice_spin(0), [1.0, 2.0]),
            (SU2Irrep::from_twice_spin(1), [2.0, 0.5]),
        ],
        Truncation::rank(3)
    );
    assert_eq!(kept, [vec![1], vec![0]]);
}

#[test]
fn truncspace_keeps_tensorkits_set_in_stored_order() {
    // TensorKit `findtruncated(::SectorVector, truncspace(0 => 3, 1 => 1))`
    // returns `sortperm(d; by = abs, rev = true)[1:k]`: 0-based `[1, 3, 0]`
    // for charge 0 and `[1]` for charge 1, and so also permutes the kept
    // columns into magnitude order. symtn keeps the same set in stored order
    // (`Truncation::Space`); sorting it by the keep order recovers
    // TensorKit's index vector exactly.
    let values = [vec![2.0, -3.0, 0.5, -2.5], vec![1.0, -1.5]];
    let target = GradedSpace::try_new(Arc::new(U1FusionRule), [(u1(0), 3), (u1(1), 1)]).unwrap();
    let kept = positions!(
        U1FusionRule,
        [(u1(0), [2.0, -3.0, 0.5, -2.5]), (u1(1), [1.0, -1.5])],
        Truncation::space(target.truncspace())
    );
    assert_eq!(kept, [vec![0, 1, 3], vec![1]]);
    let tensorkit = [vec![1, 3, 0], vec![1]];
    for ((mut positions, values), expected) in kept.into_iter().zip(&values).zip(tensorkit) {
        // Keep order: |v| descending, then position ascending.
        positions.sort_by(|&a, &b| {
            f64::abs(values[b])
                .total_cmp(&f64::abs(values[a]))
                .then(a.cmp(&b))
        });
        assert_eq!(positions, expected);
    }
}

#[test]
fn every_spectrum_dtype_selects_the_same_tensorkit_positions() {
    // The signed rank-3 fixture above, published as f32, and as Complex32 /
    // Complex64 eigenvalues rotated by the exact phases 1, i, -1, -i, so
    // every magnitude (and so every tie) is exactly the f64 one.
    use num_complex::{Complex32, Complex64};
    let values = [vec![-1.0, 0.5, 3.0, -2.0], vec![2.0, -2.0]];
    let leg = GradedSpace::try_new(Arc::new(U1FusionRule), [(u1(0), 4), (u1(1), 2)]).unwrap();
    let expected = [vec![2, 3], vec![0]];
    macro_rules! check {
        ($convert:expr) => {{
            let spectra: Vec<SectorSpectrum<_, _>> = [u1(0), u1(1)]
                .into_iter()
                .zip(&values)
                .map(|(sector, values)| SectorSpectrum {
                    sector,
                    values: values.iter().enumerate().map($convert).collect(),
                })
                .collect();
            let found = leg.find_truncated(&spectra, &Truncation::rank(3)).unwrap();
            let got: Vec<Vec<usize>> = [u1(0), u1(1)]
                .iter()
                .map(|sector| found.selection.positions(sector).unwrap())
                .collect();
            assert_eq!(got, expected);
        }};
    }
    let phase = |k: usize, v: f64| match k % 4 {
        0 => (v, 0.0),
        1 => (0.0, v),
        2 => (-v, 0.0),
        _ => (0.0, -v),
    };
    check!(|(_, &v): (usize, &f64)| v as f32);
    check!(|(k, &v): (usize, &f64)| {
        let (re, im) = phase(k, v);
        Complex32::new(re as f32, im as f32)
    });
    check!(|(k, &v): (usize, &f64)| {
        let (re, im) = phase(k, v);
        Complex64::new(re, im)
    });
}
