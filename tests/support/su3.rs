//! SU(3) leg and dimension for the single-precision linear-algebra tests.
//!
//! Kept apart from `legs.rs` because it needs `fusion-symbols`, which not
//! every crate that includes `legs.rs` depends on.

#![allow(dead_code)]

use std::sync::Arc;

use symtn::sector::LieGroupFusionRule;
use symtn::typed::GradedSpace;

/// SU(3) leg with the single irrep `[2, 2]` at the given degeneracy.
pub fn su3_leg(degeneracy: usize) -> GradedSpace<LieGroupFusionRule> {
    GradedSpace::try_new(
        Arc::new(LieGroupFusionRule::new(fusion_symbols::lie::GroupId::su(3).unwrap()).unwrap()),
        [(vec![2i64, 2], degeneracy)],
    )
    .unwrap()
}

/// Dimension of the SU(3) irrep with Dynkin labels `labels`.
pub fn su3_dim(labels: &[i64]) -> f64 {
    let (p, q) = (labels[0] as f64, labels[1] as f64);
    (p + 1.0) * (q + 1.0) * (p + q + 2.0) / 2.0
}
