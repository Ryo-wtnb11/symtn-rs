//! Graded spaces ("legs") and spin helpers that several `symtn` test targets
//! build identically, written against the public facade only.
//!
//! Test targets include this file with
//! `#[path = ".../tests/support/legs.rs"] mod legs;`. A fixed leg here is
//! named by its degeneracies in listed sector order (`su2_leg_2_2_1`:
//! twice-spin 0, 1, 2 with degeneracy 2, 2, 1); a target that needs different
//! sectors or degeneracies builds them through [`u1_leg`] / [`su2_leg`].

#![allow(dead_code)]

use std::sync::Arc;

use crate::su2::{su2_irrep, su2_rule};

use symtn::sector::{
    FermionParityFusionRule, FermionParityIrrep, LieGroupFusionRule, LieIrrep,
    MultiplicityFreeAdmissionMode, U1FusionRule, U1Irrep,
};
use symtn::typed::GradedSpace;

/// U(1) leg with `(charge, degeneracy)` sectors.
pub fn u1_leg(pairs: &[(i32, usize)]) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::new(U1FusionRule),
        pairs
            .iter()
            .map(|&(charge, degeneracy)| (U1Irrep::new(charge).unwrap(), degeneracy)),
    )
    .unwrap()
}

/// U(1) leg on a caller-owned rule, for targets that compare legs by rule
/// identity.
pub fn u1_on(provider: &Arc<U1FusionRule>, pairs: &[(i32, usize)]) -> GradedSpace<U1FusionRule> {
    GradedSpace::try_new(
        Arc::clone(provider),
        pairs
            .iter()
            .map(|&(charge, degeneracy)| (U1Irrep::new(charge).unwrap(), degeneracy)),
    )
    .unwrap()
}

/// SU(2) leg with `(twice_spin, degeneracy)` sectors.
pub fn su2_leg(
    pairs: &[(usize, usize)],
) -> GradedSpace<LieGroupFusionRule<MultiplicityFreeAdmissionMode>> {
    GradedSpace::try_new(
        Arc::new(su2_rule()),
        pairs
            .iter()
            .map(|&(twice_spin, degeneracy)| (su2_irrep(twice_spin), degeneracy)),
    )
    .unwrap()
}

/// SU(2) leg on a caller-owned rule, for targets that compare legs by rule
/// identity.
pub fn su2_on(
    provider: &Arc<LieGroupFusionRule<MultiplicityFreeAdmissionMode>>,
    pairs: &[(usize, usize)],
) -> GradedSpace<LieGroupFusionRule<MultiplicityFreeAdmissionMode>> {
    GradedSpace::try_new(
        Arc::clone(provider),
        pairs
            .iter()
            .map(|&(twice_spin, degeneracy)| (su2_irrep(twice_spin), degeneracy)),
    )
    .unwrap()
}

/// Fermion-parity leg with `(is_odd, degeneracy)` sectors.
pub fn fz2_leg(pairs: &[(bool, usize)]) -> GradedSpace<FermionParityFusionRule> {
    GradedSpace::try_new(
        Arc::new(FermionParityFusionRule::new()),
        pairs.iter().map(|&(odd, degeneracy)| {
            (
                if odd {
                    FermionParityIrrep::ODD
                } else {
                    FermionParityIrrep::EVEN
                },
                degeneracy,
            )
        }),
    )
    .unwrap()
}

/// U(1) charges -1, 0, 1 with degeneracies 2, 1, 2.
pub fn u1_leg_2_1_2() -> GradedSpace<U1FusionRule> {
    u1_leg(&[(-1, 2), (0, 1), (1, 2)])
}

/// U(1) charges -1, 0, 1 with degeneracies 2, 3, 2.
pub fn u1_leg_2_3_2() -> GradedSpace<U1FusionRule> {
    u1_leg(&[(-1, 2), (0, 3), (1, 2)])
}

/// U(1) charges 0, 1 with degeneracies 2, 2.
pub fn u1_leg_2_2() -> GradedSpace<U1FusionRule> {
    u1_leg(&[(0, 2), (1, 2)])
}

/// U(1) charges 0, 1, -1 with degeneracies 2, 3, 1.
pub fn u1_leg_2_3_1() -> GradedSpace<U1FusionRule> {
    u1_leg(&[(0, 2), (1, 3), (-1, 1)])
}

/// SU(2) twice-spin 0, 1 with degeneracies 2, 2.
pub fn su2_leg_2_2() -> GradedSpace<LieGroupFusionRule<MultiplicityFreeAdmissionMode>> {
    su2_leg(&[(0, 2), (1, 2)])
}

/// SU(2) twice-spin 0, 1, 2 with degeneracies 2, 2, 1.
pub fn su2_leg_2_2_1() -> GradedSpace<LieGroupFusionRule<MultiplicityFreeAdmissionMode>> {
    su2_leg(&[(0, 2), (1, 2), (2, 1)])
}

/// SU(2) twice-spin 0, 1, 2 with degeneracies 2, 3, 1.
pub fn su2_leg_2_3_1() -> GradedSpace<LieGroupFusionRule<MultiplicityFreeAdmissionMode>> {
    su2_leg(&[(0, 2), (1, 3), (2, 1)])
}

/// U(1) charges -1, 0, 1 with degeneracies 2, 1, 3, and its dual.
pub fn u1_legs() -> (GradedSpace<U1FusionRule>, GradedSpace<U1FusionRule>) {
    let v = u1_leg(&[(-1, 2), (0, 1), (1, 3)]);
    let w = v.try_dual().unwrap();
    (v, w)
}

/// SU(2) twice-spin 0, 1, 2 with degeneracies 2, 2, 1, and its dual.
pub fn su2_legs() -> (
    GradedSpace<LieGroupFusionRule<MultiplicityFreeAdmissionMode>>,
    GradedSpace<LieGroupFusionRule<MultiplicityFreeAdmissionMode>>,
) {
    let v = su2_leg_2_2_1();
    let w = v.try_dual().unwrap();
    (v, w)
}

/// Twice the spin of an SU(2) sector.
pub fn su2_spin(sector: &LieIrrep) -> usize {
    sector.dynkin().next().unwrap() as usize
}

/// Dimension of an SU(2) irrep.
pub fn su2_dim(sector: &LieIrrep) -> f64 {
    (sector.dynkin().next().unwrap() as usize + 1) as f64
}

/// Two U(1) legs on one rule, charges -1, 0, 1, with degeneracies 2, 3, 4 and
/// 5, 1, 2: the operands of the norm tests.
pub fn u1_norm_legs() -> (GradedSpace<U1FusionRule>, GradedSpace<U1FusionRule>) {
    let rule = Arc::new(U1FusionRule);
    (
        u1_on(&rule, &[(-1, 2), (0, 3), (1, 4)]),
        u1_on(&rule, &[(-1, 5), (0, 1), (1, 2)]),
    )
}

/// Two SU(2) legs on one rule, twice-spin 0, 1, 2, with degeneracies 2, 3, 4
/// and 5, 1, 2: the operands of the norm tests.
pub fn su2_norm_legs() -> (
    GradedSpace<LieGroupFusionRule<MultiplicityFreeAdmissionMode>>,
    GradedSpace<LieGroupFusionRule<MultiplicityFreeAdmissionMode>>,
) {
    let rule = Arc::new(su2_rule());
    (
        su2_on(&rule, &[(0, 2), (1, 3), (2, 4)]),
        su2_on(&rule, &[(0, 5), (1, 1), (2, 2)]),
    )
}
