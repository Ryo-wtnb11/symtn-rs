//! The smallest Generic toy fusion rule, shared by symtn-tensors and
//! symtn-matrixalgebra tests: one self-dual sector `x` with
//! `x (x) x -> 1 + x` and outer multiplicity two on `x (x) x -> x`. It carries
//! fusion structure only, no symbols. `ID` gives otherwise identical rules
//! distinct rule identities.
//!
//! Test targets include this file with
//! `#[path = ".../tests/support/generic_toy_fusion.rs"] mod generic_toy_fusion;`.

#![allow(dead_code)]

use symtn_core::{
    BraidingStyleKind, FusionRule, FusionStyleKind, RuleIdentity, SectorId, SectorVec,
};

#[derive(Clone, Copy)]
pub struct ToyGenericRule<const ID: u8>;

impl<const ID: u8> symtn_core::testing::Provider for ToyGenericRule<ID> {}

impl<const ID: u8> FusionRule for ToyGenericRule<ID> {
    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::of_type::<Self>()
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Generic
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        SectorId::new(0)
    }

    fn dual(&self, sector: SectorId) -> SectorId {
        sector
    }

    fn fusion_channels(&self, left: SectorId, right: SectorId) -> SectorVec {
        match (left.id(), right.id()) {
            (0, sector) | (sector, 0) => [SectorId::new(sector)].into_iter().collect(),
            (1, 1) => [SectorId::new(0), SectorId::new(1)].into_iter().collect(),
            _ => SectorVec::new(),
        }
    }

    fn nsymbol(&self, left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        if (left.id(), right.id(), coupled.id()) == (1, 1, 1) {
            2
        } else {
            usize::from(self.fusion_channels(left, right).contains(&coupled))
        }
    }
}
