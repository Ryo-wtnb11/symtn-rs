//! Independent dense-CGC bends in TensorKitSectors 0.3.9's descending basis.

use num_complex::{Complex32, Complex64};
use symtn_sectors::{MultiplicityFreeRigidSymbols, PhysicalFusionBasis, SU2FusionRule, SectorId};

#[path = "../../tests/support/numerics.rs"]
mod numerics;

#[test]
fn su2_bends_match_dense_cgc_contractions_with_negative_fs_phase() {
    let rule = SU2FusionRule;
    let cgc = |a, b, c, i, j, k| {
        rule.try_fusion_tensor_element(
            SectorId::new(a),
            SectorId::new(b),
            SectorId::new(c),
            i,
            j,
            k,
            0,
        )
        .unwrap()
    };
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
                    cgc(dj, dj, 0, i, j, 0),
                    singlet(dj, i, j),
                    1,
                );
            }
        }
    }
    assert_eq!(rule.frobenius_schur_phase_scalar(SectorId::new(1)), -1.0);

    // TensorKitSectors src/sectors.jl: Asymbol_from_fusiontensor and
    // Bsymbol_from_fusiontensor, not their production F-symbol formulas.
    for (a, b, c) in [(1, 1, 0), (1, 1, 2), (1, 2, 1), (2, 1, 1), (3, 2, 1)] {
        let scale = (((a + 1) * (b + 1)) as f64 / (c + 1) as f64).sqrt();
        let mut bend_a = 0.0;
        for i in 0..=a {
            for j in 0..=a {
                for k in 0..=c {
                    bend_a += singlet(a, i, j) * cgc(a, b, c, i, 0, k) * cgc(a, c, b, j, k, 0);
                }
            }
        }
        let mut bend_b = 0.0;
        for i in 0..=b {
            for j in 0..=b {
                for k in 0..=c {
                    bend_b += singlet(b, i, j) * cgc(a, b, c, 0, i, k) * cgc(c, b, a, k, j, 0);
                }
            }
        }
        let [a_id, b_id, c_id] = [a, b, c].map(SectorId::new);
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
