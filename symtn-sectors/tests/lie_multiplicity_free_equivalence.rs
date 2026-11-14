//! The multiplicity-free Lie rule over SU(2) is the compact SU(2) rule on the
//! compact rule's domain, bit for bit: the same ids, labels, duals,
//! dimensions, twists and Frobenius–Schur phases for every doubled spin
//! `0..=254`, and the same channels, N, F, R, A and B wherever
//! `SU2FusionRule` admits the inputs. That domain is per operation: every
//! label `<= 254`, and for fusion, N and R also the fusion closure
//! `a + b <= 254`; F, A and B admit each label alone, so their vertices may
//! fuse past 254. The Lie rule's own domain is wider (labels up to
//! `u32::MAX`, no closure bound) and is not restricted here. The oracle is
//! `SU2FusionRule`, whose codec (`id = 2j`, no table) is independent of the
//! Lie codec.

use fusion_symbols::lie::GroupId;
use symtn_sectors::{
    CheckedFusionAlgebra, FusionAlgebraError, FusionRule, LieGroupFusionRule,
    MultiplicityFreeAdmissionMode, MultiplicityFreeFusionSymbols, MultiplicityFreeRigidSymbols,
    SU2FusionRule, SU2Irrep, SectorCodec, SectorId, SU2_MAX_DOUBLED_SPIN,
};

type MfLie = LieGroupFusionRule<MultiplicityFreeAdmissionMode>;

fn su2() -> MfLie {
    MfLie::new(GroupId::su(2).unwrap()).unwrap()
}

fn s(id: usize) -> SectorId {
    SectorId::new(id)
}

#[test]
fn labels_ids_and_single_sector_data_match_on_every_compact_spin() {
    let lie = su2();
    assert_eq!(lie.vacuum(), SU2FusionRule.vacuum());
    for dj in 0..=SU2_MAX_DOUBLED_SPIN {
        let label = lie.category().irrep(&[dj as i64]).unwrap();
        let id = SU2FusionRule
            .encode_sector(&SU2Irrep::from_twice_spin(dj))
            .unwrap();
        assert_eq!(lie.encode_sector(&label).unwrap(), id, "2j = {dj}");
        assert_eq!(lie.decode_sector(id).unwrap(), label, "2j = {dj}");
        assert_eq!(lie.dual(id), SU2FusionRule.dual(id));
        for (name, mf, compact) in [
            ("dim", lie.dim_scalar(id), SU2FusionRule.dim_scalar(id)),
            (
                "sqrt_dim",
                lie.sqrt_dim_scalar(id),
                SU2FusionRule.sqrt_dim_scalar(id),
            ),
            (
                "inv_dim",
                lie.inv_dim_scalar(id),
                SU2FusionRule.inv_dim_scalar(id),
            ),
            (
                "inv_sqrt_dim",
                lie.inv_sqrt_dim_scalar(id),
                SU2FusionRule.inv_sqrt_dim_scalar(id),
            ),
            (
                "twist",
                lie.twist_scalar(id),
                SU2FusionRule.twist_scalar(id),
            ),
            (
                "fs",
                lie.frobenius_schur_phase_scalar(id),
                SU2FusionRule.frobenius_schur_phase_scalar(id),
            ),
        ] {
            assert_eq!(mf.to_bits(), compact.to_bits(), "{name}(2j = {dj})");
        }
    }
}

#[test]
fn channels_and_nsymbols_match_on_every_compact_pair() {
    let lie = su2();
    for a in 0..=SU2_MAX_DOUBLED_SPIN {
        for b in 0..=SU2_MAX_DOUBLED_SPIN - a {
            let channels = SU2FusionRule.fusion_channels(s(a), s(b));
            assert_eq!(lie.fusion_channels(s(a), s(b)), channels, "{a} x {b}");
            // Every coupled spin from below the window to past it, of both
            // parities, admitted or not.
            for c in a.abs_diff(b).saturating_sub(1)..=(a + b + 1).min(SU2_MAX_DOUBLED_SPIN) {
                assert_eq!(
                    lie.try_nsymbol(s(a), s(b), s(c)),
                    SU2FusionRule.try_nsymbol(s(a), s(b), s(c)),
                    "N({a}, {b}, {c})"
                );
            }
        }
    }
    // Past the compact closure the Lie rule keeps answering.
    assert_eq!(lie.try_nsymbol(s(254), s(1), s(253)), Ok(1));
    assert_eq!(
        SU2FusionRule.try_nsymbol(s(254), s(1), s(253)),
        Err(FusionAlgebraError::FusionNotRepresentable {
            left: s(254),
            right: s(1)
        })
    );
}

fn assert_bits(name: &str, ids: &[usize], mf: f64, compact: f64) {
    assert_eq!(mf.to_bits(), compact.to_bits(), "{name}{ids:?}");
}

/// The doubled spins in `x ⊗ y` within the compact labels, by the triangle
/// and parity rule; no closure bound, as F, A and B admit labels alone.
fn window(x: usize, y: usize) -> impl Iterator<Item = usize> {
    (x.abs_diff(y)..=(x + y).min(SU2_MAX_DOUBLED_SPIN)).step_by(2)
}

fn assert_r(lie: &MfLie, [a, b, c]: [usize; 3]) {
    // R admits the fusion closure.
    if a + b <= SU2_MAX_DOUBLED_SPIN {
        let [x, y, z] = [a, b, c].map(s);
        assert_bits(
            "R",
            &[a, b, c],
            lie.r_symbol_scalar(x, y, z),
            SU2FusionRule.r_symbol_scalar(x, y, z),
        );
    }
}

fn assert_a_b(lie: &MfLie, [a, b, c]: [usize; 3]) {
    let [x, y, z] = [a, b, c].map(s);
    assert_bits(
        "A",
        &[a, b, c],
        lie.a_symbol_scalar(x, y, z),
        SU2FusionRule.a_symbol_scalar(x, y, z),
    );
    assert_bits(
        "B",
        &[a, b, c],
        lie.b_symbol_scalar(x, y, z),
        SU2FusionRule.b_symbol_scalar(x, y, z),
    );
}

fn assert_f(lie: &MfLie, ids: [usize; 6]) {
    let [a, b, c, d, e, f] = ids.map(s);
    assert_bits(
        "F",
        &ids,
        lie.f_symbol_scalar(a, b, c, d, e, f),
        SU2FusionRule.f_symbol_scalar(a, b, c, d, e, f),
    );
}

/// Every admissible F over `(a, b, c)` with `(d, e, f)` among the window
/// entries `pick` keeps. Returns the number compared.
fn assert_fs(lie: &MfLie, [a, b, c]: [usize; 3], pick: &dyn Fn(Vec<usize>) -> Vec<usize>) -> usize {
    let mut checked = 0;
    for e in pick(window(a, b).collect()) {
        for d in pick(window(e, c).collect()) {
            for f in pick(window(b, c).collect()) {
                if window(a, f).any(|x| x == d) {
                    assert_f(lie, [a, b, c, d, e, f]);
                    checked += 1;
                }
            }
        }
    }
    checked
}

/// Exhaustive over small spins: every triple and every admissible F.
#[test]
fn f_r_a_b_symbols_match_bitwise_on_small_spins() {
    let lie = su2();
    let mut checked = 0;
    for a in 0..=6 {
        for b in 0..=6 {
            for c in 0..=6 {
                assert_r(&lie, [a, b, c]);
                assert_a_b(&lie, [a, b, c]);
                checked += assert_fs(&lie, [a, b, c], &|all| all);
            }
        }
    }
    assert!(checked > 1000, "{checked} F symbols");
}

/// Across the whole compact range up to its bounds: R, A and B on every
/// coupled spin of each pair of a spin grid that reaches `127` and `254`, and
/// F on the first two and last two entries of each channel window, vertices
/// past the fusion closure included.
#[test]
fn f_r_a_b_symbols_match_bitwise_up_to_the_compact_bound() {
    let lie = su2();
    let grid = [
        0, 1, 2, 3, 62, 63, 64, 119, 120, 126, 127, 128, 190, 240, 253, 254,
    ];
    let edges = |all: Vec<usize>| {
        let n = all.len();
        let mut picked: Vec<usize> = [0, 1, n.saturating_sub(2), n.saturating_sub(1)]
            .into_iter()
            .filter(|&index| index < n)
            .collect();
        picked.dedup();
        picked.into_iter().map(|index| all[index]).collect()
    };
    let mut checked = 0;
    for a in grid {
        for b in grid {
            for c in window(a, b).chain(grid) {
                assert_r(&lie, [a, b, c]);
                assert_a_b(&lie, [a, b, c]);
                checked += assert_fs(&lie, [a, b, c], &edges);
            }
        }
    }
    assert!(checked > 1000, "{checked} F symbols");
}

/// Every doubled spin `0..=254` in families valid for every `j`, whose
/// vertices `j ⊗ j` pass the fusion closure once `j > 127`.
#[test]
fn symbol_families_match_bitwise_on_every_compact_spin() {
    let lie = su2();
    for j in 0..=SU2_MAX_DOUBLED_SPIN {
        assert_a_b(&lie, [j, j, 0]);
        assert_a_b(&lie, [j, 0, j]);
        assert_r(&lie, [j, j, 0]);
        assert_r(&lie, [j, 0, j]);
        assert_f(&lie, [j, 0, j, 0, j, j]);
        assert_f(&lie, [j, j, j, j, 0, 0]);
        if j >= 1 {
            assert_a_b(&lie, [j, 1, j - 1]);
            assert_f(&lie, [j, 1, 1, j, j - 1, 0]);
        }
    }
    assert_r(&lie, [120, 1, 119]);
    assert_a_b(&lie, [120, 1, 119]);
    assert_f(&lie, [120, 0, 120, 240, 120, 120]);
    assert_f(&lie, [127, 0, 127, 254, 127, 127]);
}
