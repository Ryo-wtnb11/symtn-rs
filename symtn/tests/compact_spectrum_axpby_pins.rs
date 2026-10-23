//! Compact-spectrum `axpby` against a lazy adjoint operand (#2081 pins).
//!
//! The compact operand's result must equal its dense twin's, which takes the
//! ordinary dense route and never sees a spectrum. The lazy operand's parent
//! carries off-diagonal weight, so a wrong sector pairing in the spectrum
//! scatter changes the result.

use std::sync::Arc;

use symtn::sector::{
    product_sector, FermionParityFusionRule, Fz2SectorLayout, PackedProductCodec,
    ProductFusionRule, U1FusionRule, U1Irrep, U1SectorLayout, Z2Irrep,
};
use symtn::typed::{Complex64, GradedSpace, SectorSpectrum, TensorMap};

#[path = "../../tests/support/fixtures.rs"]
mod fixtures;

use fixtures::host_runtime;

type Fz2U1Codec = PackedProductCodec<Fz2SectorLayout, U1SectorLayout>;
type Fz2U1Rule = ProductFusionRule<FermionParityFusionRule, U1FusionRule, Fz2U1Codec>;

/// Pinned over a bond: compact vs dense-twin `axpby(alpha, lazy, beta)`, for a
/// scalar built by `scalar(re, im)`.
macro_rules! pin {
    ($name:ident, $rule:expr, $labels:expr, $D:ty, $scalar:expr) => {
        #[test]
        fn $name() {
            let runtime = host_runtime();
            let scalar = $scalar;
            let labels: Vec<_> = $labels;
            let degeneracy = |position: usize| 1 + position % 3;
            let bond = GradedSpace::try_new(
                Arc::new($rule),
                labels.iter().enumerate().map(|(p, &l)| (l, degeneracy(p))),
            )
            .unwrap();
            let sectors = bond.sectors().unwrap().to_vec();
            let diag = |position: usize, i: usize| {
                scalar(
                    0.5 + position as f64 * 0.01 + i as f64,
                    0.1 * i as f64 - 0.2,
                )
            };
            let spectrum: Vec<_> = sectors
                .iter()
                .enumerate()
                .map(|(position, &sector)| SectorSpectrum {
                    sector,
                    values: (0..bond.degeneracies()[position])
                        .map(|i| diag(position, i))
                        .collect::<Vec<$D>>(),
                })
                .collect();
            let compact = TensorMap::<_, $D>::diagonal(&runtime, &bond, spectrum).unwrap();
            let twin = TensorMap::from_subblock_fn(&runtime, [&bond], [&bond], |trees, idx| {
                let position = sectors.iter().position(|s| s == trees.coupled()).unwrap();
                if idx[0] == idx[1] {
                    diag(position, idx[0])
                } else {
                    scalar(0.0, 0.0)
                }
            })
            .unwrap();
            // Parent weight is arbitrary; the lazy operand is its adjoint.
            let lazy = TensorMap::from_subblock_fn(&runtime, [&bond], [&bond], |trees, idx| {
                let position = sectors.iter().position(|s| s == trees.coupled()).unwrap();
                scalar(
                    0.7 * idx[0] as f64 + 0.2 * idx[1] as f64 - 0.01 * position as f64,
                    0.3 * idx[1] as f64 - 0.1,
                )
            })
            .unwrap()
            .adjoint()
            .unwrap();
            let (alpha, beta) = (scalar(0.75, 0.25), scalar(-1.25, 0.5));
            for (what, got, want) in [
                (
                    "compact.axpby(lazy)",
                    compact.axpby(alpha, &lazy, beta).unwrap(),
                    twin.axpby(alpha, &lazy, beta).unwrap(),
                ),
                (
                    "lazy.axpby(compact)",
                    lazy.axpby(beta, &compact, alpha).unwrap(),
                    lazy.axpby(beta, &twin, alpha).unwrap(),
                ),
            ] {
                let gap = got
                    .axpby(scalar(1.0, 0.0), &want, scalar(-1.0, 0.0))
                    .unwrap();
                let scale = f64::from(want.norm(2.0).unwrap()).max(1.0);
                assert!(f64::from(gap.norm(2.0).unwrap()) <= 1e-12 * scale, "{what}");
            }
        }
    };
}

fn u1_wide() -> Vec<U1Irrep> {
    (-70..70).map(|k| U1Irrep::new(7 * k)).collect()
}

fn fz2_u1() -> Vec<<Fz2U1Rule as symtn::sector::TypedSectorAdmission>::Sector> {
    (-12..12)
        .map(|k| {
            product_sector(
                if k % 2 == 0 {
                    Z2Irrep::EVEN
                } else {
                    Z2Irrep::ODD
                },
                U1Irrep::new(k * 5),
            )
        })
        .collect()
}

pin!(
    u1_wide_f64,
    U1FusionRule,
    u1_wide(),
    f64,
    |re: f64, _im: f64| re
);
pin!(
    u1_wide_c64,
    U1FusionRule,
    u1_wide(),
    Complex64,
    Complex64::new
);
pin!(
    product_f64,
    Fz2U1Rule::try_new(FermionParityFusionRule, U1FusionRule).unwrap(),
    fz2_u1(),
    f64,
    |re: f64, _im: f64| re
);
pin!(
    product_c64,
    Fz2U1Rule::try_new(FermionParityFusionRule, U1FusionRule).unwrap(),
    fz2_u1(),
    Complex64,
    Complex64::new
);
