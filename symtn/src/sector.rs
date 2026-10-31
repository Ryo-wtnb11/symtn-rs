//! Symmetry contract: everything a user needs to add a symmetry or to bound
//! generic tensor code by one.
//!
//! A category defined outside this workspace is a `fusion_symbols` category;
//! [`CategoryRule`] with an [`EnumeratedCodec`] (or a [`CategoryCodec`] of
//! its own) makes it a provider that uses the same
//! [`crate::typed::TensorMap`] API as the built-in rules ([`U1FusionRule`],
//! [`SU2FusionRule`], ...) exported here. The provider traits are sealed
//! bounds, not an extension point. This module is the only path to each item
//! it lists.
//!
//! A category's labels pass through the adapter into spaces and tensors:
//!
//! ```
//! use std::sync::Arc;
//! use fusion_symbols::table::TableCategory;
//! use symtn::sector::{CategoryRule, EnumeratedCodec, MultiplicityFreeAdmissionMode};
//! use symtn::typed::{Complex64, GradedSpace, Runtime, TensorMap};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let engine = &mut symtn::typed::CpuEngine::serial();
//! let category = TableCategory::fibonacci();
//! let tau = category.object(2)?;
//! let rule = CategoryRule::try_new(
//!     category,
//!     EnumeratedCodec::<MultiplicityFreeAdmissionMode>::new(),
//! )?;
//! let v = GradedSpace::try_new(Arc::new(rule), [(tau, 2)])?;
//! let runtime = Runtime::builder().build()?;
//! let tensor: TensorMap<_, Complex64> = TensorMap::zeros(engine, &runtime, [&v], [&v])?;
//! assert_eq!(tensor.subblock_count(), 1);
//! # Ok(())
//! # }
//! ```

/// Built-in fusion rules, their sector labels, and product constructors.
pub use symtn_core::{
    product_sector, CU1FusionRule, CU1Irrep, FermionParityFusionRule, ProductAdmission,
    ProductFusionRule, ProductFusionRuleError, ProductSector, SU2FusionRule, SU2Irrep, SectorId,
    TableRule, TableRuleError, TableSector, U1FusionRule, U1Irrep, Z2FusionRule, Z2Irrep,
    ZNFusionRule, ZNIrrep,
};
pub use symtn_core::{
    BraidingStyleKind, FusionStyleKind, GenericFArray, GenericRMatrix, RuleIdentity,
    SectorOrderKey, SectorVec,
};
/// The fusion rule of a `fusion_symbols` category: the extension point for a
/// category defined outside symtn, with its codec and admission vocabulary.
pub use symtn_core::{
    CanonicalUnitAdmission, CategoryAdapterError, CategoryAdmissionError, CategoryCodec,
    CategoryFailure, CategoryOperation, CategoryProvider, CategoryRule, EnumeratedCodec,
    UncertifiedUnitAdmission,
};
/// Rule traits: the sealed fusion, symbol and admission interfaces of every
/// provider, and the bounds generic code names.
pub use symtn_core::{
    CanonicalUnitFusionRule, CategoricalScalar, CheckedCanonicalUnitFusionRule,
    CheckedFusionAlgebra, CheckedGenericFusion, CheckedGenericPivotal, CheckedGenericRigidSymbols,
    CoupledSectorFold, FusionRule, MultiplicityFreeFusionRule, MultiplicityFreeFusionSymbols,
    MultiplicityFreeRigidSymbols, PhysicalFusionBasis, SectorCodec, TypedSectorAdmission,
};
/// Sealed admission modes: the values of `TypedSectorAdmission::Mode`.
pub use symtn_core::{CheckedGenericAdmissionMode, MultiplicityFreeAdmissionMode};
/// Label codecs of a [`ProductFusionRule`], its third type
/// parameter, and the per-factor layouts they pack.
pub use symtn_core::{
    Fz2SectorLayout, PackedProductCodec, PackedSectorLayout, ProductSectorCodec,
    ProductSectorCodecError, ProductSectorLayout, Su2SectorLayout, TensorKitProductCodec,
    U1SectorLayout,
};
#[cfg(feature = "cgc-gen")]
pub use symtn_core::{LieGroupFusionRule, LieGroupFusionRuleError};
/// Vocabulary the rule traits' signatures use.
pub use symtn_core::{
    PhysicalBasisError, ProductSectorComponent, SU2CoefficientError, SymbolShapeError,
};
