//! `zeros_like` is one body for every admission mode and symbol lane (#1869).
//!
//! TensorKit `cfaa073e` has one `zero(t) = zerovector(t)`
//! (`src/tensors/linalg.jl:51`) built on `zerovector!(similar(t, T))`
//! (`src/tensors/vectorinterface.jl:7`) for every sector style, and keeps a
//! `DiagonalTensorMap` diagonal (`src/tensors/diagonal.jl:275`). So the
//! multiplicity-free rules (U(1), SU(2), and Fibonacci with complex symbols)
//! and checked Generic (SU(2) and SU(3), which has fusion multiplicity) must
//! all give: the operand's codomain and domain, a fresh payload of exact
//! positive zeros whatever the operand holds (NaN, Inf and `-0.0` included),
//! and the operand's storage form (dense stays dense, a lazy adjoint stays a
//! lazy adjoint, compact diagonal stays compact).
//!
//! `zeros_like` has no Host error path (neither has `zerovector`); the CUDA
//! admission errors are pinned in `representation_gates::cuda`.

use std::sync::Arc;

use num_complex::Complex64;
use symtn::sector::{SU2FusionRule, SU2Irrep, TableRule, TableSector, U1FusionRule, U1Irrep};
use symtn::typed::__network::NetworkReuseClass;
use symtn::typed::{GradedSpace, SectorSpectrum, TensorMap};

use crate::fixtures;

use fixtures::host_runtime;

/// Source entries cycle through these, so the zeros cannot be a scaled copy.
const SOURCE: [f64; 5] = [f64::NAN, -0.0, f64::INFINITY, 1.5, -2.25];

fn source(i: usize) -> Complex64 {
    Complex64::new(SOURCE[i % 5], SOURCE[(i + 2) % 5])
}

fn exact_zeros<D: Copy + Into<Complex64>>(values: &[D]) -> bool {
    values.iter().all(|&x| {
        let z: Complex64 = x.into();
        z.re.to_bits() == 0 && z.im.to_bits() == 0
    })
}

fn form<R, D>(t: &TensorMap<R, D>) -> NetworkReuseClass {
    symtn::typed::__network::network_reuse_class(t, false)
}

/// Runs the dense, lazy-adjoint (when `adjoint` gives one) and compact rows on `[leg, leg*] <- [leg]`
/// (and the compact bond `leg <- leg`) and returns the dense stored length,
/// which equal reduced-block structures share across modes.
macro_rules! zeros_rows {
    ($leg:expr, $D:ty, $conv:expr) => {
        zeros_rows!($leg, $D, $conv, |t: &TensorMap<_, $D>| Some(
            t.adjoint().unwrap()
        ))
    };
    ($leg:expr, $D:ty, $conv:expr, $adjoint:expr) => {{
        let adjoint = $adjoint;
        let runtime = host_runtime();
        let leg = $leg;
        let dual = leg.try_dual().unwrap();
        let conv = $conv;
        let mut n = 0;
        let dense =
            TensorMap::<_, $D>::from_subblock_fn(&runtime, [&leg, &dual], [&leg], |_, _| {
                n += 1;
                conv(source(n))
            })
            .unwrap();
        assert!(n > 0, "the fixture stores entries");

        // Dense: same spaces, fresh exact-zero payload.
        let zero = dense.zeros_like().unwrap();
        assert_eq!(zero.codomain(), dense.codomain());
        assert_eq!(zero.domain(), dense.domain());
        assert!(form(&zero) == NetworkReuseClass::OwnedDense);
        let data = zero.dense_data().unwrap();
        assert_eq!(data.len(), dense.dense_data().unwrap().len());
        assert!(exact_zeros(data), "dense zeros are exact +0");
        assert_ne!(data.as_ptr(), dense.dense_data().unwrap().as_ptr());

        // Lazy adjoint: stays lazy, over a zero parent.
        if let Some(lazy) = adjoint(&dense) {
            let zero_lazy = lazy.zeros_like().unwrap();
            assert_eq!(zero_lazy.codomain(), lazy.codomain());
            assert_eq!(zero_lazy.domain(), lazy.domain());
            assert!(form(&lazy) == NetworkReuseClass::LazyAdjoint);
            assert!(
                form(&zero_lazy) == NetworkReuseClass::LazyAdjoint,
                "lazy stays lazy"
            );
            assert!(exact_zeros(
                adjoint(&zero_lazy).unwrap().dense_data().unwrap()
            ));
            // Materializing conjugates (and may apply a recoupling sign), so
            // `-0.0` is allowed here; the parent above is bit-exact.
            assert!(zero_lazy
                .materialize()
                .unwrap()
                .dense_data()
                .unwrap()
                .iter()
                .all(|&x| Into::<Complex64>::into(x) == Complex64::default()));
        }

        // Compact diagonal (and its adjoint): stays compact, same blocks.
        let spectra: Vec<_> = leg
            .sectors()
            .unwrap()
            .into_iter()
            .enumerate()
            .map(|(s, sector)| SectorSpectrum {
                values: (0..leg.degeneracy(&sector).unwrap())
                    .map(|i| conv(source(s + i)))
                    .collect::<Vec<$D>>(),
                sector,
            })
            .collect();
        let compact = TensorMap::<_, $D>::diagonal(&runtime, &leg, spectra).unwrap();
        for operand in std::iter::once(compact.clone()).chain(adjoint(&compact)) {
            let zero = operand.zeros_like().unwrap();
            assert_eq!(zero.codomain(), operand.codomain());
            assert_eq!(zero.domain(), operand.domain());
            assert!(form(&operand) == NetworkReuseClass::Compact);
            assert!(
                form(&zero) == NetworkReuseClass::Compact,
                "compact stays compact"
            );
            let (zeros, source) = (zero.materialize().unwrap(), operand.materialize().unwrap());
            assert_eq!(
                zeros.dense_data().unwrap().len(),
                source.dense_data().unwrap().len()
            );
            assert!(
                exact_zeros(zeros.dense_data().unwrap()),
                "compact zeros are exact +0"
            );
        }
        n
    }};
}

fn u1_leg() -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        [
            (U1Irrep::new(0), 2),
            (U1Irrep::new(1), 3),
            (U1Irrep::new(-1), 1),
        ],
    )
    .unwrap()
}

fn su2_leg() -> GradedSpace<SU2FusionRule> {
    GradedSpace::try_new(
        Arc::new(SU2FusionRule),
        [
            (SU2Irrep::from_twice_spin(0), 2),
            (SU2Irrep::from_twice_spin(1), 3),
            (SU2Irrep::from_twice_spin(2), 1),
        ],
    )
    .unwrap()
}

fn fibonacci_leg() -> GradedSpace<TableRule> {
    GradedSpace::try_new(
        Arc::new(TableRule::fibonacci()),
        [(TableSector::new(1), 2), (TableSector::new(2), 3)],
    )
    .unwrap()
}

const REAL: fn(Complex64) -> f64 = |z| z.re;
const COMPLEX: fn(Complex64) -> Complex64 = |z| z;

#[test]
fn multiplicity_free_zeros_like_keeps_space_and_form() {
    zeros_rows!(u1_leg(), f64, REAL);
    zeros_rows!(u1_leg(), Complex64, COMPLEX);
    zeros_rows!(su2_leg(), f64, REAL);
    zeros_rows!(su2_leg(), Complex64, COMPLEX);
}

#[test]
fn complex_symbol_zeros_like_keeps_space_and_form() {
    // Fibonacci has complex F/R symbols (`Scalar = Complex64`). Its payload
    // is complex and it has no `adjoint` (both compile-time absent), so the
    // lane has the dense and compact rows.
    zeros_rows!(fibonacci_leg(), Complex64, COMPLEX, |_: &TensorMap<
        TableRule,
        Complex64,
    >| {
        None::<TensorMap<TableRule, Complex64>>
    });
}

#[cfg(feature = "cgc-gen")]
mod checked {
    use super::*;
    use symtn::sector::LieGroupFusionRule;

    fn sun_leg(n: usize, sectors: &[(Vec<i64>, usize)]) -> GradedSpace<LieGroupFusionRule> {
        GradedSpace::try_new(
            Arc::new(
                LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(n).unwrap()).unwrap(),
            ),
            sectors.to_vec(),
        )
        .unwrap()
    }

    /// The `su2_leg` sectors as checked Generic SU(2) labels `[2j]`.
    fn checked_su2_leg() -> GradedSpace<LieGroupFusionRule> {
        sun_leg(2, &[(vec![0], 2), (vec![1], 3), (vec![2], 1)])
    }

    /// `8 ⊗ 8 ∋ 8` twice: the dense fixture has outer multiplicity.
    fn su3_leg() -> GradedSpace<LieGroupFusionRule> {
        sun_leg(3, &[(vec![1, 1], 2), (vec![0, 0], 1)])
    }

    #[test]
    fn checked_generic_zeros_like_matches_the_multiplicity_free_rule() {
        for (mf, checked) in [
            (
                zeros_rows!(su2_leg(), f64, REAL),
                zeros_rows!(checked_su2_leg(), f64, REAL),
            ),
            (
                zeros_rows!(su2_leg(), Complex64, COMPLEX),
                zeros_rows!(checked_su2_leg(), Complex64, COMPLEX),
            ),
        ] {
            assert_eq!(mf, checked, "same SU(2) reduced blocks in both modes");
        }
    }

    #[test]
    fn su3_zeros_like_keeps_space_and_form() {
        zeros_rows!(su3_leg(), f64, REAL);
        zeros_rows!(su3_leg(), Complex64, COMPLEX);
    }
}
