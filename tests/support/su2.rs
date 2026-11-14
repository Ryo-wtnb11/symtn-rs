//! Shared construction of multiplicity-free SU(2) fixtures using provider labels.
#![allow(dead_code)]

use crate::sector_types::{GroupId, LieGroupFusionRule, LieIrrep, MultiplicityFreeAdmissionMode};
use std::sync::OnceLock;

pub type Su2Rule = LieGroupFusionRule<MultiplicityFreeAdmissionMode>;

pub fn su2_rule() -> Su2Rule {
    // Keep value identities alive for spy providers and raw structures.
    static RULE: OnceLock<Su2Rule> = OnceLock::new();
    RULE.get_or_init(|| LieGroupFusionRule::new(GroupId::su(2).unwrap()).unwrap())
        .clone()
}

pub fn su2_irrep(doubled: usize) -> LieIrrep {
    su2_rule()
        .category()
        .irrep(&[i64::try_from(doubled).unwrap()])
        .unwrap()
}
