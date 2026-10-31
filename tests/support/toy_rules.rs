//! A two-sector checked Generic toy rule several symtn test targets share:
//! sectors `Vacuum` and `X`, `X (x) X -> Vacuum + X` with outer
//! multiplicity two on `X (x) X -> X`, an allocation-free `dim`, and a typed
//! error for any other sector id.
//!
//! Test targets include this file with
//! `#[path = ".../tests/support/toy_rules.rs"] mod toy_rules;`. The
//! instrumented checked toys of `checked_generic_facade` and
//! `checked_generic_braiding` stay local: they fuse different sectors.

#![allow(dead_code)]

use std::fmt;
use std::sync::Arc;

use symtn::sector::{
    BraidingStyleKind, CheckedGenericAdmissionMode, CheckedGenericFusion,
    CheckedGenericRigidSymbols, FusionStyleKind, GenericFArray, GenericRMatrix, RuleIdentity,
    SectorId, SectorVec, TypedSectorAdmission,
};

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum GenericLabel {
    Vacuum,
    X,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GenericError;

impl fmt::Display for GenericError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("invalid synthetic Generic sector")
    }
}

impl std::error::Error for GenericError {}

pub struct GenericToy;

impl GenericToy {
    pub const VACUUM: SectorId = SectorId::new(0);
    pub const X: SectorId = SectorId::new(1);

    fn channels(left: SectorId, right: SectorId) -> Result<SectorVec, GenericError> {
        match (left, right) {
            (Self::VACUUM, sector) | (sector, Self::VACUUM)
                if sector == Self::VACUUM || sector == Self::X =>
            {
                Ok([sector].into_iter().collect())
            }
            (Self::X, Self::X) => Ok([Self::VACUUM, Self::X].into_iter().collect()),
            _ => Err(GenericError),
        }
    }

    fn multiplicity(left: SectorId, right: SectorId, coupled: SectorId) -> usize {
        if (left, right, coupled) == (Self::X, Self::X, Self::X) {
            2
        } else {
            usize::from(
                Self::channels(left, right).is_ok_and(|channels| channels.contains(&coupled)),
            )
        }
    }
}

impl symtn_core::testing::Provider for GenericToy {}

impl CheckedGenericFusion for GenericToy {
    type Error = GenericError;

    fn rule_identity(&self) -> RuleIdentity {
        RuleIdentity::from_canonical_bytes::<Self>(0x1003, Arc::<[u8]>::from(*b"generic-toy"))
    }

    fn fusion_style(&self) -> FusionStyleKind {
        FusionStyleKind::Generic
    }

    fn braiding_style(&self) -> BraidingStyleKind {
        BraidingStyleKind::Bosonic
    }

    fn vacuum(&self) -> SectorId {
        Self::VACUUM
    }

    fn try_dual(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        match sector {
            Self::VACUUM | Self::X => Ok(sector),
            _ => Err(GenericError),
        }
    }

    fn try_fusion_channels(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        Self::channels(left, right)
    }

    fn try_fusion_channels_in_table(
        &self,
        left: SectorId,
        right: SectorId,
    ) -> Result<SectorVec, Self::Error> {
        Self::channels(left, right)
    }

    fn try_nsymbol(
        &self,
        left: SectorId,
        right: SectorId,
        coupled: SectorId,
    ) -> Result<usize, Self::Error> {
        Ok(Self::multiplicity(left, right, coupled))
    }
}

impl CheckedGenericRigidSymbols for GenericToy {
    type Scalar = f64;

    fn try_sqrt_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        match sector {
            Self::VACUUM => Ok(1.0),
            Self::X => Ok((1.0 + 2.0_f64.sqrt()).sqrt()),
            _ => Err(GenericError),
        }
    }

    fn try_inv_sqrt_dim_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        Ok(self.try_sqrt_dim_scalar(sector)?.recip())
    }

    fn try_frobenius_schur_phase_scalar(&self, sector: SectorId) -> Result<f64, Self::Error> {
        self.try_dual(sector).map(|_| 1.0)
    }

    fn try_f_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
        d: SectorId,
        e: SectorId,
        f: SectorId,
    ) -> Result<GenericFArray<f64>, Self::Error> {
        let shape = (
            Self::multiplicity(a, b, e),
            Self::multiplicity(e, c, d),
            Self::multiplicity(b, c, f),
            Self::multiplicity(a, f, d),
        );
        let rows = shape.0 * shape.1;
        let cols = shape.2 * shape.3;
        Ok(GenericFArray::new(
            (0..rows * cols)
                .map(|index| f64::from(index / cols == index % cols))
                .collect(),
            shape,
        ))
    }

    fn try_r_symbol_generic(
        &self,
        a: SectorId,
        b: SectorId,
        c: SectorId,
    ) -> Result<GenericRMatrix<f64>, Self::Error> {
        let size = Self::multiplicity(a, b, c);
        Ok(GenericRMatrix::new(
            (0..size * size)
                .map(|index| f64::from(index / size == index % size))
                .collect(),
            size,
            size,
        ))
    }
}

impl TypedSectorAdmission for GenericToy {
    type Sector = GenericLabel;
    type Error = GenericError;
    type Mode = CheckedGenericAdmissionMode;

    fn typed_rule_identity(&self) -> RuleIdentity {
        CheckedGenericFusion::rule_identity(self)
    }

    fn try_encode_label(&self, sector: &Self::Sector) -> Result<SectorId, Self::Error> {
        Ok(match sector {
            GenericLabel::Vacuum => Self::VACUUM,
            GenericLabel::X => Self::X,
        })
    }

    fn try_decode_label(&self, sector: SectorId) -> Result<Self::Sector, Self::Error> {
        match sector {
            Self::VACUUM => Ok(GenericLabel::Vacuum),
            Self::X => Ok(GenericLabel::X),
            _ => Err(GenericError),
        }
    }

    fn try_dual_id(&self, sector: SectorId) -> Result<SectorId, Self::Error> {
        self.try_dual(sector)
    }
}
