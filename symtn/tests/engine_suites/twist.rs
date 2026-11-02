//! `twist` against independent oracles: every ribbon-twist factor is a sign
//! (an entry is kept or negated, never otherwise scaled, and both kinds of
//! block occur), the inverse twist undoes the twist exactly, a bosonic
//! provider or an empty leg list is a clone, and a space with no coupled
//! sector or a leg past the rank is handled before any work. Device == Host
//! parity of the same call stays in `typed_cuda_twist.rs`.

use super::transform::{complex_fill, real_fill};
use super::*;
use crate::legs::u1_leg_2_1_2;
use num_complex::Complex64;
use std::sync::Arc;
use symtn::sector::{
    product_sector, FermionParityFusionRule, SU2FusionRule, SU2Irrep, U1FusionRule, U1Irrep,
    Z2Irrep,
};
use symtn::typed::{Direction, GradedSpace, Runtime};

fn fz2_leg(dual: bool) -> GradedSpace<FermionParityFusionRule> {
    let space = GradedSpace::try_new(
        Arc::new(FermionParityFusionRule),
        [(Z2Irrep::EVEN, 2), (Z2Irrep::ODD, 1)],
    )
    .unwrap();
    if dual {
        space.try_dual().unwrap()
    } else {
        space
    }
}

/// Twists `$tensor` on `$backend` over each leg list in both directions and
/// requires a kept and a negated entry in every result.
macro_rules! signs_only {
    ($backend:ty, $device:ident, $what:expr, $tensor:expr, $cases:expr) => {{
        let tensor = &$tensor;
        let source = <$backend>::upload($device, tensor);
        for legs in $cases {
            for direction in [Direction::Forward, Direction::Inverse] {
                let twisted = source.twist($device, legs, direction).unwrap();
                let twisted = <$backend>::download($device, &twisted);
                let (mut kept, mut negated) = (0usize, 0usize);
                for (&before, &after) in tensor
                    .dense_data()
                    .unwrap()
                    .iter()
                    .zip(twisted.dense_data().unwrap())
                {
                    if after == before {
                        kept += 1;
                    } else {
                        assert_eq!(
                            after, -before,
                            "{} twist {legs:?} scaled by something else",
                            $what
                        );
                        negated += 1;
                    }
                }
                assert!(negated > 0, "{} twist {legs:?} changed nothing", $what);
                assert!(
                    kept > 0,
                    "{} twist {legs:?} scaled every entry, so no unscaled block is covered",
                    $what
                );
            }
        }
    }};
}

on_each_engine! {
    /// The factor is a real sign on fZ2 and on its products with U(1) and
    /// SU(2), which is what lets a backend carry it on a descriptor scale.
    fn a_fermionic_twist_only_keeps_or_negates_an_entry<B>() {
        let device = &mut B::engine();
        let engine = &mut symtn::typed::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();

        let leg = fz2_leg(false);
        let dual = fz2_leg(true);
        let real: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&leg, &dual], [&leg, &leg], real_fill)
                .unwrap();
        assert!(real.subblock_count() >= 2, "multi-block fixture");
        signs_only!(
            B, device, "fZ2/f64", real,
            [&[0usize][..], &[1][..], &[3][..], &[0, 3][..], &[1, 2, 3][..], &[0, 1, 1, 3][..]]
        );
        let complex: TensorMap<_, Complex64> = TensorMap::from_subblock_fn(
            engine, &runtime, [&leg, &dual], [&leg, &leg], complex_fill,
        )
        .unwrap();
        signs_only!(B, device, "fZ2/c64", complex, [&[1usize, 2][..]]);

        let rule = Arc::new(
            symtn_sectors::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule)
                .unwrap(),
        );
        let leg_u1 = GradedSpace::try_new(
            Arc::clone(&rule),
            [
                (product_sector(Z2Irrep::EVEN, U1Irrep::new(0)), 2),
                (product_sector(Z2Irrep::ODD, U1Irrep::new(1)), 1),
                (product_sector(Z2Irrep::EVEN, U1Irrep::new(2)), 1),
            ],
        )
        .unwrap();
        let product_u1: TensorMap<_, f64> = TensorMap::from_subblock_fn(
            engine, &runtime, [&leg_u1, &leg_u1], [&leg_u1, &leg_u1], real_fill,
        )
        .unwrap();
        signs_only!(
            B, device, "fZ2 x U(1)", product_u1,
            [&[1usize][..], &[2][..], &[0, 2][..]]
        );

        let rule = Arc::new(
            symtn_sectors::ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, SU2FusionRule)
                .unwrap(),
        );
        let leg_su2 = GradedSpace::try_new(
            Arc::clone(&rule),
            [
                (product_sector(Z2Irrep::EVEN, SU2Irrep::from_twice_spin(0)), 2),
                (product_sector(Z2Irrep::ODD, SU2Irrep::from_twice_spin(1)), 1),
                (product_sector(Z2Irrep::EVEN, SU2Irrep::from_twice_spin(2)), 1),
            ],
        )
        .unwrap();
        let product_su2: TensorMap<_, Complex64> = TensorMap::from_subblock_fn(
            engine, &runtime, [&leg_su2, &leg_su2], [&leg_su2, &leg_su2], complex_fill,
        )
        .unwrap();
        signs_only!(
            B, device, "fZ2 (x) SU(2)", product_su2,
            [&[0usize][..], &[3][..], &[1, 3][..]]
        );
    }
}

on_each_engine! {
    /// Twisting every leg multiplies the factors to the block's parity, which
    /// is even, so the value is unchanged (the detection tests each leg, so
    /// this is not a short circuit); an inverse twist undoes a twist exactly
    /// because both factors are exactly +-1; an empty leg list and a bosonic
    /// provider are clones.
    fn twists_round_trip_exactly_and_trivial_twists_are_clones<B>() {
        let device = &mut B::engine();
        let engine = &mut symtn::typed::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let leg = fz2_leg(false);
        let dual = fz2_leg(true);
        let tensor: TensorMap<_, Complex64> = TensorMap::from_subblock_fn(
            engine, &runtime, [&leg, &dual], [&leg, &leg], complex_fill,
        )
        .unwrap();
        let source = B::upload(device, &tensor);
        macro_rules! values {
            ($moved:expr) => {{
                let moved = $moved;
                B::download(device, &moved).dense_data().unwrap().to_vec()
            }};
        }

        for legs in [&[0usize, 1, 2, 3][..], &[0, 1, 2, 3, 0, 1, 2, 3][..], &[][..]] {
            for direction in [Direction::Forward, Direction::Inverse] {
                assert_eq!(
                    values!(source.twist(device, legs, direction).unwrap()),
                    tensor.dense_data().unwrap(),
                    "twist {legs:?} {direction:?}"
                );
            }
        }
        let twisted = source.twist(device, &[0, 2], Direction::Forward).unwrap();
        assert_ne!(values!(twisted.clone()), tensor.dense_data().unwrap());
        assert_eq!(
            values!(twisted.twist(device, &[0, 2], Direction::Inverse).unwrap()),
            tensor.dense_data().unwrap(),
            "twist round trip is exact"
        );

        let u1 = u1_leg_2_1_2();
        let bosonic: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&u1, &u1], [&u1, &u1], real_fill)
                .unwrap();
        let bosonic_source = B::upload(device, &bosonic);
        for legs in [&[0usize][..], &[1, 3][..]] {
            for direction in [Direction::Forward, Direction::Inverse] {
                assert_eq!(
                    values!(bosonic_source.twist(device, legs, direction).unwrap()),
                    bosonic.dense_data().unwrap(),
                    "bosonic twist {legs:?} {direction:?} must be a clone"
                );
            }
        }
    }
}

on_each_engine! {
    /// A space with no coupled sector has no block, so the twist is the
    /// identity short circuit rather than a zero-length transfer; a leg past
    /// the rank is refused with the Host's error before anything else.
    fn a_twist_without_blocks_is_empty_and_a_leg_past_the_rank_is_refused<B>() {
        let device = &mut B::engine();
        let engine = &mut symtn::typed::CpuEngine::serial();
        let runtime = Runtime::builder().build().unwrap();
        let even =
            GradedSpace::try_new(Arc::new(FermionParityFusionRule), [(Z2Irrep::EVEN, 2)]).unwrap();
        let odd =
            GradedSpace::try_new(Arc::new(FermionParityFusionRule), [(Z2Irrep::ODD, 1)]).unwrap();
        let empty: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&even], [&odd], real_fill).unwrap();
        assert!(empty.dense_data().unwrap().is_empty(), "the fixture carries no element");
        let source = B::upload(device, &empty);
        let twisted = source.twist(device, &[0, 1], Direction::Forward).unwrap();
        assert!(B::download(device, &twisted).dense_data().unwrap().is_empty());

        let leg = fz2_leg(false);
        let host: TensorMap<_, f64> =
            TensorMap::from_subblock_fn(engine, &runtime, [&leg], [&leg], real_fill).unwrap();
        let source = B::upload(device, &host);
        for (axis, direction) in [(2, Direction::Forward), (7, Direction::Inverse)] {
            assert_eq!(
                source.twist(device, &[axis], direction).unwrap_err(),
                host.twist(engine, &[axis], direction).unwrap_err(),
            );
        }
    }
}
