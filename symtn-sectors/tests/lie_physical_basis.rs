//! The multiplicity-free Lie rule's physical carrier basis: TensorKitSectors
//! 0.3.12's `SU2Irrep` `fusiontensor` order (descending `m`) per rank-one
//! factor, factors composed as its `ProductSector` `_kron` (first factor most
//! significant), and one recoupling gauge with the rule's F and A/B symbols.
//! Expected values are hand values or dense CGC contractions.

use fusion_symbols::lie::GroupId;
use symtn_sectors::{
    CheckedFusionAlgebra, FusionRule, LieGroupFusionRule, MultiplicityFreeAdmissionMode,
    MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols, PhysicalBasisError,
    PhysicalFusionBasis, SectorId,
};

use crate::numerics;

type MfLie = LieGroupFusionRule<MultiplicityFreeAdmissionMode>;

fn rule(group: GroupId) -> MfLie {
    MfLie::new(group).unwrap()
}

fn su2() -> MfLie {
    rule(GroupId::su(2).unwrap())
}

fn spin(rule: &MfLie, dj: i64) -> SectorId {
    rule.encode_irrep(&[dj]).unwrap()
}

fn element(rule: &MfLie, [a, b, c]: [SectorId; 3], [i, j, k]: [usize; 3]) -> f64 {
    rule.try_fusion_tensor_element(a, b, c, i, j, k, 0).unwrap()
}

fn dim(rule: &MfLie, sector: SectorId) -> usize {
    rule.try_carrier_dimension(sector).unwrap()
}

#[test]
fn su2_carrier_descends_from_highest_weight_with_tensorkit_odd_phase() {
    // The asymmetric 1 x 1/2 -> 1/2 channel is reversal-odd, so unlike
    // 1/2 x 1/2 -> 1 it fails if ascending and descending are confused.
    let rule = su2();
    let [one, half] = [spin(&rule, 2), spin(&rule, 1)];
    let vertex = [one, half, half];

    assert_eq!(rule.try_carrier_dimension(one), Ok(3));
    assert_eq!(rule.try_carrier_dimension(half), Ok(2));
    // <1 +1; 1/2 -1/2 | 1/2 +1/2> = sqrt(2/3), and its reversal is odd.
    let expected = (2.0_f64 / 3.0).sqrt();
    numerics::assert_close("highest", element(&rule, vertex, [0, 1, 0]), expected, 1);
    numerics::assert_close("reversed", element(&rule, vertex, [2, 0, 1]), -expected, 1);
    assert_eq!(
        rule.try_fusion_tensor_element(one, half, half, 3, 0, 0, 0),
        Err(PhysicalBasisError::CarrierIndexOutOfBounds {
            sector: one,
            index: 3,
            dimension: 3,
        })
    );
    assert_eq!(
        rule.try_fusion_tensor_element(one, half, half, 0, 0, 0, 1),
        Err(PhysicalBasisError::FusionMultiplicityOutOfBounds {
            left: one,
            right: half,
            coupled: half,
            multiplicity: 1,
            dimension: 1,
        })
    );
}

#[test]
fn product_carrier_puts_the_first_factor_most_significant() {
    // Spin(4) (1/2, 0) x (0, 1/2) -> (1/2, 1/2): both factor CGCs are 1, so
    // the coupled index is exactly the Kronecker index 2 i + j.
    let rule = rule(GroupId::spin(4).unwrap());
    let vertex = [[1, 0], [0, 1], [1, 1]].map(|labels| rule.encode_irrep(&labels).unwrap());
    assert_eq!(dim(&rule, vertex[2]), 4);
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..4 {
                let expected = f64::from(k == 2 * i + j);
                assert_eq!(
                    element(&rule, vertex, [i, j, k]),
                    expected,
                    "({i}, {j}, {k})"
                );
            }
        }
    }
}

#[test]
fn su2_local_tensors_are_one_complete_orthonormal_transform() {
    // All 1 x 1/2 product states and both coupled sectors form one unitary
    // change of basis, not merely a few matching coefficients.
    let rule = su2();
    let [left, right] = [spin(&rule, 2), spin(&rule, 1)];
    let coupled = [spin(&rule, 1), spin(&rule, 3)];
    let columns: Vec<(SectorId, usize)> = coupled
        .iter()
        .flat_map(|&c| (0..dim(&rule, c)).map(move |k| (c, k)))
        .collect();
    let rows: Vec<[usize; 2]> = (0..3).flat_map(|i| (0..2).map(move |j| [i, j])).collect();
    assert_eq!(rows.len(), columns.len());
    let entry =
        |[i, j]: [usize; 2], (c, k): (SectorId, usize)| element(&rule, [left, right, c], [i, j, k]);

    for &x in &rows {
        for &y in &rows {
            let inner: f64 = columns.iter().map(|&c| entry(x, c) * entry(y, c)).sum();
            numerics::assert_close("rows", inner, f64::from(x == y), columns.len());
        }
    }
    for &x in &columns {
        for &y in &columns {
            let inner: f64 = rows.iter().map(|&r| entry(r, x) * entry(r, y)).sum();
            numerics::assert_close("columns", inner, f64::from(x == y), rows.len());
        }
    }
}

/// `<left-tree | right-tree>` at one coupled basis state: the dense
/// contraction of the two CGC parenthesizations of `left x middle x right`.
fn recoupling_overlap(
    rule: &MfLie,
    [left, middle, right, total]: [SectorId; 4],
    [left_coupled, right_coupled]: [SectorId; 2],
    total_basis: usize,
) -> f64 {
    let mut overlap = 0.0;
    for a in 0..dim(rule, left) {
        for b in 0..dim(rule, middle) {
            for c in 0..dim(rule, right) {
                let left_tree: f64 = (0..dim(rule, left_coupled))
                    .map(|e| {
                        element(rule, [left, middle, left_coupled], [a, b, e])
                            * element(rule, [left_coupled, right, total], [e, c, total_basis])
                    })
                    .sum();
                let right_tree: f64 = (0..dim(rule, right_coupled))
                    .map(|f| {
                        element(rule, [middle, right, right_coupled], [b, c, f])
                            * element(rule, [left, right_coupled, total], [a, f, total_basis])
                    })
                    .sum();
                overlap += left_tree * right_tree;
            }
        }
    }
    overlap
}

/// Every admissible F for the given externals equals the CGC recoupling at
/// every coupled basis state. Returns the number of F entries compared.
fn assert_recoupling_matches_f(rule: &MfLie, external: [SectorId; 4]) -> usize {
    let [left, middle, right, total] = external;
    let mut compared = 0;
    for left_coupled in rule.fusion_channels(left, middle) {
        if rule.nsymbol(left_coupled, right, total) == 0 {
            continue;
        }
        for right_coupled in rule.fusion_channels(middle, right) {
            if rule.nsymbol(left, right_coupled, total) == 0 {
                continue;
            }
            let expected =
                rule.f_symbol_scalar(left, middle, right, total, left_coupled, right_coupled);
            for total_basis in 0..dim(rule, total) {
                let overlap =
                    recoupling_overlap(rule, external, [left_coupled, right_coupled], total_basis);
                let terms = dim(rule, left) * dim(rule, middle) * dim(rule, right) * 2;
                numerics::assert_close("F from CGCs", overlap, expected, terms);
            }
            compared += 1;
        }
    }
    compared
}

#[test]
fn su2_physical_basis_and_f_symbols_share_one_recoupling_gauge() {
    let rule = su2();
    let half = spin(&rule, 1);
    assert_eq!(assert_recoupling_matches_f(&rule, [half; 4]), 4);
}

#[test]
fn asymmetric_su2_recoupling_fixes_f_symbol_orientation() {
    // All four external spins differ and the left intermediates {1/2, 3/2}
    // differ from the right {3/2, 5/2}, so swapping the F axes is an
    // inadmissible request, not a hidden matrix transpose.
    let rule = su2();
    let external = [1, 2, 3, 4].map(|dj| spin(&rule, dj));
    assert_eq!(assert_recoupling_matches_f(&rule, external), 4);
    let [half, five_halves] = [1, 5].map(|dj| spin(&rule, dj));
    let f = |e, f| rule.f_symbol_scalar(external[0], external[1], external[2], external[3], e, f);
    assert_eq!(rule.nsymbol(external[0], external[1], five_halves), 0);
    assert!((f(half, five_halves) - f(five_halves, half)).abs() > 0.5);
}

#[test]
fn product_physical_bases_share_the_factor_recoupling_gauge() {
    // Spin(4) with both factors nontrivial (a 4 x 4 F), and U(2) with
    // asymmetric spins and nonzero charges (a 2 x 2 F).
    let spin4 = rule(GroupId::spin(4).unwrap());
    let bispinor = spin4.encode_irrep(&[1, 1]).unwrap();
    assert_eq!(assert_recoupling_matches_f(&spin4, [bispinor; 4]), 16);

    let u2 = rule(GroupId::u(2).unwrap());
    let label = |dj, q| u2.encode_irrep_with_charges(&[dj], &[q]).unwrap();
    assert_eq!(
        assert_recoupling_matches_f(&u2, [label(1, 1), label(2, 0), label(1, -1), label(2, 0)]),
        4
    );
}

#[test]
fn su2_bends_match_dense_cgc_contractions_with_negative_fs_phase() {
    let rule = su2();
    let cgc = |a, b, c, i, j, k| element(&rule, [a, b, c].map(|dj| spin(&rule, dj)), [i, j, k]);
    // <j,m;j,-m|0,0> = (-1)^(j-m)/sqrt(2j+1), independently of F/A/B.
    let singlet = |dj: usize, i: usize, j: usize| {
        if i + j == dj {
            (if i.is_multiple_of(2) { 1.0 } else { -1.0 }) / ((dj + 1) as f64).sqrt()
        } else {
            0.0
        }
    };
    for dj in [1, 2, 3] {
        for i in 0..=dj {
            for j in 0..=dj {
                numerics::assert_close(
                    "singlet CGC",
                    cgc(dj as i64, dj as i64, 0, i, j, 0),
                    singlet(dj, i, j),
                    1,
                );
            }
        }
    }
    assert_eq!(rule.frobenius_schur_phase_scalar(spin(&rule, 1)), -1.0);

    // TensorKitSectors src/sectors.jl: Asymbol_from_fusiontensor and
    // Bsymbol_from_fusiontensor, not their production F-symbol formulas.
    for (a, b, c) in [(1, 1, 0), (1, 1, 2), (1, 2, 1), (2, 1, 1), (3, 2, 1)] {
        let scale = (((a + 1) * (b + 1)) as f64 / (c + 1) as f64).sqrt();
        let [ia, ib, ic] = [a, b, c].map(|x| x as i64);
        let mut bend_a = 0.0;
        for i in 0..=a {
            for j in 0..=a {
                for k in 0..=c {
                    bend_a +=
                        singlet(a, i, j) * cgc(ia, ib, ic, i, 0, k) * cgc(ia, ic, ib, j, k, 0);
                }
            }
        }
        let mut bend_b = 0.0;
        for i in 0..=b {
            for j in 0..=b {
                for k in 0..=c {
                    bend_b +=
                        singlet(b, i, j) * cgc(ia, ib, ic, 0, i, k) * cgc(ic, ib, ia, k, j, 0);
                }
            }
        }
        let [a_id, b_id, c_id] = [ia, ib, ic].map(|dj| spin(&rule, dj));
        // Each dense contraction has at most (dj+1)^2 * (dc+1) terms.
        numerics::assert_close(
            "A from dense CGCs",
            rule.a_symbol_scalar(a_id, b_id, c_id),
            scale * bend_a,
            (a + 1) * (a + 1) * (c + 1),
        );
        numerics::assert_close(
            "B from dense CGCs",
            rule.b_symbol_scalar(a_id, b_id, c_id),
            scale * bend_b,
            (b + 1) * (b + 1) * (c + 1),
        );
    }
}

#[test]
fn spins_beyond_the_i32_projection_range_are_unsupported() {
    // fusion-symbols' CGC projections are `i32`; its labels reach `u32`.
    let rule = su2();
    let last = spin(&rule, i64::from(i32::MAX));
    let beyond = spin(&rule, i64::from(i32::MAX) + 1);
    assert_eq!(rule.try_carrier_dimension(last), Ok(1 << 31));
    assert_eq!(
        rule.try_carrier_dimension(beyond),
        Err(PhysicalBasisError::Unsupported { sector: beyond })
    );
    // The fusion algebra still admits the vertex; only the basis is refused.
    assert_eq!(rule.try_nsymbol(beyond, rule.vacuum(), beyond), Ok(1));
    assert_eq!(
        rule.try_fusion_tensor_element(beyond, rule.vacuum(), beyond, 0, 0, 0, 0),
        Err(PhysicalBasisError::Unsupported { sector: beyond })
    );
}
