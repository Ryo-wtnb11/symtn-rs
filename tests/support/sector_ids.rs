//! Sector-id shorthands for the built-in irreps, shared by the
//! `symtn-sectors` integration tests and the `symtn-core` in-crate tests.
//!
//! Included with `#[path = ".../tests/support/sector_ids.rs"] mod sector_ids;`.

#![allow(dead_code)]

use symtn_sectors::{SU2Irrep, SectorId, U1Irrep, Z2Irrep};

pub fn u1(charge: i32) -> SectorId {
    U1Irrep::new(charge).sector_id()
}

/// A U(1) sector id outside the codec's range.
pub fn excluded_u1_id() -> SectorId {
    SectorId::new(u32::MAX as usize)
}

pub fn z2_even() -> SectorId {
    Z2Irrep::EVEN.sector_id()
}

pub fn z2_odd() -> SectorId {
    Z2Irrep::ODD.sector_id()
}

pub fn su2(twice_spin: usize) -> SectorId {
    SU2Irrep::from_twice_spin(twice_spin).sector_id()
}
