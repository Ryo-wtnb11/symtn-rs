#![forbid(unsafe_code)]

//! Structural vocabulary of symtn: spaces and sectors, fusion trees and their
//! keys, block structures and block views, and the structure caches.
//!
//! Lower-level crates may lower the block views to concrete strided kernels,
//! but no external strided or backend type is required to use this crate.

use core::fmt;
use core::ops::{Add, Mul};
use std::collections::{hash_map::Entry, BTreeMap};
use std::hash::Hash;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, Weak};

#[cfg(test)]
use num_complex::Complex64;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
pub use symtn_sectors::CheckedGenericAdmissionMode;
pub use symtn_sectors::{
    product_fusion_rule, product_fusion_rule_with_codec, product_sector, BraidingStyleKind,
    CU1FusionRule, CU1Irrep, CanonicalUnitAdmission, CanonicalUnitFusionRule, CategoricalScalar,
    CategoryAdapterError, CategoryAdmissionError, CategoryCodec, CategoryFailure,
    CategoryOperation, CategoryRule, CheckedCanonicalUnitFusionRule, CheckedFusionAlgebra,
    CheckedGenericFusion, CheckedGenericPivotal, CheckedGenericRigidSymbols, CoupledSectorFold,
    CoupledSectorFoldBuilder, EnumeratedCodec, FermionParityFusionRule, FusionAlgebraError,
    FusionRule, FusionStyleKind, Fz2SectorLayout, GenericFArray, GenericRMatrix, InfallibleGeneric,
    MultiplicityFreeAdmissionMode, MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols,
    MultiplicityFreeRigidSymbols, PackedProductCodec, PackedSectorLayout, PhysicalBasisError,
    PhysicalFusionBasis, ProductFusionRule, ProductFusionRuleExt, ProductSector,
    ProductSectorCodec, ProductSectorCodecError, ProductSectorComponent, ProductSectorLayout,
    PromoteCoefficientScalar, RuleIdentity, SU2CoefficientError, SU2FusionRule, SU2Irrep,
    SectorCodec, SectorId, SectorOrderKey, SectorVec, Su2SectorLayout, SymbolShapeError, TableRule,
    TableRuleError, TableSector, TensorKitProductCodec, TypedSectorAdmission, U1FusionRule,
    U1Irrep, U1SectorLayout, UncertifiedUnitAdmission, Z2FusionRule, Z2Irrep, ZNFusionRule,
    ZNIrrep, CU1_MAX_TWICE_CHARGE, SU2_MAX_DOUBLED_SPIN,
};
#[cfg(any(test, feature = "testing"))]
pub use symtn_sectors::{GenericFusionSymbols, GenericRigidSymbols};
#[cfg(feature = "cgc-gen")]
pub use symtn_sectors::{SUNFusionRule, SUNFusionRuleError, SUNSymbolError};

pub mod axes;
mod storage;
pub use storage::*;
mod sector;
pub use sector::*;
mod fusion_space;
pub use fusion_space::*;
mod fusion_tree;
pub use fusion_tree::*;
mod block_structure;
pub use block_structure::*;
mod cache;
#[doc(hidden)]
pub use cache::{
    register_structure_cache, set_structure_cache_byte_budget, structure_cache_info,
    structure_cache_infos, ErasedStructureCacheControl, StructureCache,
    StructureCacheRegistrationError,
};
pub use cache::{StructureCacheInfo, StructureCacheKind};
/// Borrowed-key lookups into a [`StructureCache`] implement this.
#[doc(hidden)]
pub use quick_cache::Equivalent as StructureCacheEquivalent;
mod block_view;
pub use block_view::*;
mod error;
pub use error::*;
#[cfg(feature = "testing")]
#[doc(hidden)]
pub mod testing;

#[cfg(any(test, feature = "testing"))]
mod probe;
#[cfg(test)]
mod tests;
#[cfg(test)]
pub(crate) use tests::test_support;
