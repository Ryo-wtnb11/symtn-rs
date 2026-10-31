//! Cross-cutting checked-Generic factor-plan, tree-stacking and provider
//! dispatch tests that span more than one factorization
//! tests.rs).

use super::*;

fn assert_complex_spectra_close(
    lhs: &[SectorSpectrum<Complex64>],
    rhs: &[SectorSpectrum<Complex64>],
) {
    assert_eq!(lhs.len(), rhs.len());
    for (lhs, rhs) in lhs.iter().zip(rhs) {
        assert_eq!(lhs.sector, rhs.sector);
        assert_eq!(lhs.values.len(), rhs.values.len());
        for (&lhs, &rhs) in lhs.values.iter().zip(&rhs.values) {
            assert!((lhs - rhs).norm() <= 1e-10, "{lhs} vs {rhs}");
        }
    }
}

mod factor_plan;
mod input_matricization;
mod pair_publication;
mod reconstruction;
mod tree_stacking;
