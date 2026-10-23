//! Shared fixtures for the `StackedTensorMap` gates (#1497).

#![allow(dead_code, unused_macros)]

use num_complex::Complex64;

pub use symtn::sector::{
    product_sector, FermionParityFusionRule, ProductFusionRule, SU2FusionRule, SU2Irrep,
    U1FusionRule, U1Irrep, Z2Irrep,
};
pub use symtn::sector::{Fz2SectorLayout, PackedProductCodec, U1SectorLayout};

pub type Fz2U1Rule = ProductFusionRule<
    FermionParityFusionRule,
    U1FusionRule,
    PackedProductCodec<Fz2SectorLayout, U1SectorLayout>,
>;

/// A payload whose bit pattern the gates compare, with deterministic values
/// that include signed zero, a subnormal, an infinity and a NaN payload.
pub trait Payload: Copy {
    fn value(index: usize) -> Self;
    fn bits(self) -> (u64, u64);
}

fn real(index: usize) -> f64 {
    match index % 7 {
        0 => -0.0,
        1 => f64::MIN_POSITIVE / 4.0,
        2 => f64::NEG_INFINITY,
        3 => f64::from_bits(0x7ff8_0000_0000_1234),
        _ => (index as f64 * 0.731_8).sin() * 1.0e3,
    }
}

impl Payload for f64 {
    fn value(index: usize) -> Self {
        real(index)
    }

    fn bits(self) -> (u64, u64) {
        (self.to_bits(), 0)
    }
}

impl Payload for Complex64 {
    fn value(index: usize) -> Self {
        Complex64::new(real(index), real(index + 3))
    }

    fn bits(self) -> (u64, u64) {
        (self.re.to_bits(), self.im.to_bits())
    }
}

pub fn assert_bit_exact<D: Payload>(left: &[D], right: &[D], label: &str) {
    assert_eq!(left.len(), right.len(), "{label}: length");
    for (index, (&l, &r)) in left.iter().zip(right).enumerate() {
        assert_eq!(l.bits(), r.bits(), "{label}: entry {index}");
    }
}

/// `$count` members `[a, a] <- [a]` of payload `$d`, member `k` filled with
/// `Payload::value(1000 k + n)`.
macro_rules! members {
    ($runtime:expr, $a:expr, $d:ty, $count:expr) => {
        (0..$count)
            .map(|member: usize| {
                let mut index = 1000 * member;
                symtn::typed::TensorMap::<_, $d>::from_subblock_fn(
                    $runtime,
                    [$a, $a],
                    [$a],
                    |_, _| {
                        index += 1;
                        <$d as fixtures::Payload>::value(index)
                    },
                )
                .unwrap()
            })
            .collect::<Vec<_>>()
    };
}

/// Calls `$check!(label, leg)` for U(1), SU(2), fZ2xU(1) and, with
/// `cgc-gen`, checked Generic SU(3). `leg(0)` is the base leg and
/// `leg(1)` changes one degeneracy. Every base leg has a sector of
/// degeneracy at least three, so [`selections`] can keep a non-contiguous
/// set.
macro_rules! for_each_symmetry {
    ($check:ident) => {{
        use fixtures::*;
        use std::sync::Arc;
        use symtn::typed::GradedSpace;
        $check!("U1", |variant: usize| {
            let q = U1Irrep::new;
            GradedSpace::try_new(
                Arc::new(U1FusionRule),
                [(q(-1), 2), (q(0), 1 + variant), (q(1), 3)],
            )
            .unwrap()
        });
        $check!("SU2", |variant: usize| {
            let j = SU2Irrep::from_twice_spin;
            GradedSpace::try_new(
                Arc::new(SU2FusionRule),
                [(j(0), 3), (j(1), 2 + variant), (j(2), 1)],
            )
            .unwrap()
        });
        $check!("fZ2xU1", |variant: usize| {
            let rule = Arc::new(Fz2U1Rule::try_new(FermionParityFusionRule, U1FusionRule).unwrap());
            let even = |charge| product_sector(Z2Irrep::EVEN, U1Irrep::new(charge));
            let odd = |charge| product_sector(Z2Irrep::ODD, U1Irrep::new(charge));
            GradedSpace::try_new(rule, [(even(0), 3), (odd(1), 1 + variant), (odd(-1), 2)]).unwrap()
        });
        #[cfg(feature = "cgc-gen")]
        $check!("SU3 checked Generic", |variant: usize| {
            let rule = Arc::new(symtn::sector::SUNFusionRule::new(3).unwrap());
            GradedSpace::try_new(rule, [(vec![0i64, 0], 1), (vec![1, 1], 3 + variant)]).unwrap()
        });
    }};
}

/// Three selections of `leg`, named in its own labels: every sector at a
/// non-leading offset where it has room, with every other one-dimensional
/// sector dropped (several surviving blocks, some removed); one index of
/// the last sector (a one-dimensional charged leg); and the first and last
/// position of every sector of degeneracy at least three, every position of
/// the others (two runs per such sector, #2095).
pub fn selections<R>(leg: &symtn::typed::GradedSpace<R>) -> Vec<symtn::typed::LegSelection<R>>
where
    R: symtn::sector::TypedSectorAdmission,
    R::Mode: symtn::typed::TypedTensorModeDispatch<R>,
{
    let sectors = leg.sectors().unwrap();
    let degeneracies = leg.degeneracies();
    let partial = sectors
        .iter()
        .zip(degeneracies)
        .enumerate()
        .filter(|&(index, (_, &degeneracy))| index == 0 || degeneracy > 1 || index % 2 == 0)
        .map(|(_, (sector, &degeneracy))| {
            let start = usize::from(degeneracy > 1);
            (sector.clone(), start..degeneracy)
        });
    let last = sectors.len() - 1;
    vec![
        symtn::typed::LegSelection::try_new(leg, partial).unwrap(),
        symtn::typed::LegSelection::try_new(leg, [(sectors[last].clone(), 0..1)]).unwrap(),
        symtn::typed::LegSelection::try_new(
            leg,
            sectors
                .iter()
                .zip(degeneracies)
                .map(|(sector, &degeneracy)| {
                    let ends = if degeneracy >= 3 {
                        vec![0, degeneracy - 1]
                    } else {
                        (0..degeneracy).collect()
                    };
                    (sector.clone(), ends)
                }),
        )
        .unwrap(),
    ]
}

/// `$count` members `[a, a*] <- [a, a*]` of payload `$d`: every leg kind
/// (codomain/domain, dual/non-dual) appears once.
macro_rules! mixed_members {
    ($runtime:expr, $a:expr, $dual:expr, $d:ty, $count:expr) => {
        (0..$count)
            .map(|member: usize| {
                let mut index = 1000 * member;
                symtn::typed::TensorMap::<_, $d>::from_subblock_fn(
                    $runtime,
                    [$a, $dual],
                    [$a, $dual],
                    |_, _| {
                        index += 1;
                        <$d as fixtures::Payload>::value(index)
                    },
                )
                .unwrap()
            })
            .collect::<Vec<_>>()
    };
}
