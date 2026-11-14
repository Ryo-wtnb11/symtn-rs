//! SU(3) leg and dimension for the single-precision linear-algebra tests.
//!
//! Kept apart from `legs.rs` because it needs `fusion-symbols`, which not
//! every crate that includes `legs.rs` depends on.

#![allow(dead_code)]

use std::sync::Arc;

use symtn::sector::{CheckedGenericAdmissionMode, LieGroupFusionRule};
use symtn::typed::GradedSpace;

/// SU(3) leg with the single irrep `[2, 2]` at the given degeneracy.
pub fn su3_leg(degeneracy: usize) -> GradedSpace<LieGroupFusionRule> {
    let rule = Arc::new(
        LieGroupFusionRule::<CheckedGenericAdmissionMode>::new(
            fusion_symbols::lie::GroupId::su(3).unwrap(),
        )
        .unwrap(),
    );
    let irrep = rule.category().irrep(&[2, 2]).unwrap();
    GradedSpace::try_new(rule, [(irrep, degeneracy)]).unwrap()
}

/// Dimension of the SU(3) irrep `irrep`.
pub fn su3_dim(irrep: &symtn::sector::LieIrrep) -> f64 {
    let mut labels = irrep.dynkin();
    let (p, q) = (labels.next().unwrap() as f64, labels.next().unwrap() as f64);
    (p + 1.0) * (q + 1.0) * (p + q + 2.0) / 2.0
}
