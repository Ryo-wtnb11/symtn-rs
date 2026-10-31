//! Provider-typed facade: spaces and tensor maps that keep the concrete
//! fusion-rule type `R` and speak the provider's own sector labels.
//!
//! This is the canonical user facade. `R`
//! stays concrete through monomorphized construction, so any provider —
//! including a downstream `fusion_symbols` category adapted through
//! [`crate::sector::CategoryRule`], when the resulting rule satisfies the
//! operation's admission and capability bounds — can drive it, and the categorical
//! identity of a tensor comes back as [`TypedSectorAdmission::Sector`] labels
//! instead of opaque [`crate::sector::SectorId`] keys. The engine itself never
//! sees a label; the codec is the single boundary where one enters or leaves.
//!
//! The exception is deliberate: [`TensorMap::subblock`] is the engine-level
//! layout view, and the [`crate::expert::BlockRef`] it returns carries the raw
//! [`crate::typed::BlockKey`]. Labels are what [`TensorMap::subblock_fusion_trees`]
//! is for.
//!
//! # Product symmetries
//!
//! A product symmetry needs no new constructor here, and no new type in the
//! engine. A product of providers *is* a provider — build it with
//! [`crate::sector::ProductFusionRule::try_new`] and label it with
//! [`crate::sector::product_sector`] — so this facade drives `fZ2 ⊠ U(1)`, or any
//! other ordered product of admitted components, through the same
//! [`GradedSpace::try_new`] and [`TensorMap::zeros`] as a single symmetry:
//!
//! ```
//! use std::sync::Arc;
//! use symtn::sector::{
//!     product_sector, FermionParityFusionRule, ProductFusionRule, U1FusionRule, U1Irrep, Z2Irrep,
//! };
//! use symtn::typed::{Error, GradedSpace, Runtime, TensorMap};
//!
//! # fn main() -> Result<(), Error> {
//! let engine = &mut symtn::typed::CpuEngine::serial();
//! let runtime = Runtime::builder().build()?;
//! let rule = ProductFusionRule::<_, _>::try_new(FermionParityFusionRule, U1FusionRule).unwrap();
//!
//! let even = product_sector(Z2Irrep::EVEN, U1Irrep::new(0));
//! let odd = product_sector(Z2Irrep::ODD, U1Irrep::new(1));
//! let v = GradedSpace::try_new(Arc::new(rule), [(even, 2), (odd, 1)])?;
//!
//! let t: TensorMap<_, f64> = TensorMap::zeros(engine, &runtime, [&v], [&v])?;
//! assert_eq!(t.subblock_count(), 2);
//! assert_eq!(t.subblock_fusion_trees(0)?.coupled(), &even);
//! # Ok(())
//! # }
//! ```
//!
//! Three or more factors are the same call again, because the product is
//! itself a provider. The spelling
//!
//! ```text
//! ProductFusionRule::try_new(ProductFusionRule::try_new(FermionParityFusionRule, U1FusionRule)?, SU2FusionRule)?
//! ```
//!
//! is the left-associated `(fZ2 ⊠ U(1)) ⊠ SU(2)`, whose labels are
//! `product_sector(product_sector(parity, charge), spin)` — the label nests in
//! step with the provider. **Factor order and association are structure, not
//! an equivalence.** `U(1) ⊠ fZ2` and `fZ2 ⊠ U(1)` are both legal, are
//! different Rust types with different [`crate::sector::RuleIdentity`]s, and
//! converting between them is an explicit component swap; nothing here permutes
//! factors for you. Association is where symtn diverges from TensorKit rather
//! than mirrors it: TK's `⊠` flattens nested products into one
//! `ProductSector{Tuple{…}}`, so association is unobservable there, while
//! symtn keeps the nesting in the Rust type — see
//! [`crate::sector::ProductFusionRule`] for the full statement.
//!
//! The product-provider claim is subject to the capability bounds of the
//! operation being called. Host paths admit complex categorical scalars where
//! the corresponding categorical-scalar and recoupling bounds are present;
//! some decompositions, network paths, and CUDA remain narrower.
//!
//! **`ProductSector` is not `ProductSpace`.** [`crate::sector::ProductSector`] is
//! a *sector label*: one irrep of a Deligne product category, TensorKit's
//! `ProductSector` (from `TensorKitSectors`).
//! TensorKit's `ProductSpace` is the unrelated leg-level notion — an ordered
//! list of vector spaces forming a tensor's codomain or domain — which this
//! facade spells as the `[&v, &w]` leg slices passed to every constructor, not
//! as a public container type.
//!
//! # Capabilities
//!
//! Host storage (the default `Vec<D>` body) carries the full surface:
//! construction ([`TensorMap::zeros`], [`TensorMap::from_subblock_fn`],
//! [`TensorMap::rand_with_seed`], [`TensorMap::isomorphism`],
//! [`TensorMap::isometry`]), inspection ([`TensorMap::codomain`],
//! [`TensorMap::domain`], [`TensorMap::subblock_fusion_trees`],
//! [`TensorMap::subblock`], [`TensorMap::subblock_count`], [`TensorMap::block`],
//! [`TensorMap::blocks`], [`TensorMap::dense_data`], [`TensorMap::runtime`],
//! [`TensorMap::rank`], [`TensorMap::codomain_rank`], [`TensorMap::domain_rank`],
//! [`TensorMap::leg_dims`], [`TensorMap::scalar`]), index manipulation and
//! contraction ([`TensorMap::permute`], [`TensorMap::braid`],
//! [`TensorMap::transpose`], [`TensorMap::repartition`], [`TensorMap::contract`],
//! [`TensorMap::compose`], [`TensorMap::twist`], [`TensorMap::flip`],
//! [`TensorMap::insert_unit`], [`TensorMap::remove_unit`]), scalar operations
//! ([`TensorMap::axpby`], [`TensorMap::scale`], [`TensorMap::norm`],
//! [`TensorMap::inner`], [`TensorMap::tr`], [`TensorMap::trace_pairs`],
//! [`TensorMap::adjoint`]), factorizations ([`TensorMap::svd_compact`],
//! [`TensorMap::svd_full`], [`TensorMap::svd_vals`], [`TensorMap::qr_compact`],
//! [`TensorMap::qr_full`], [`TensorMap::lq_compact`], [`TensorMap::lq_full`],
//! [`TensorMap::left_polar`], [`TensorMap::right_polar`],
//! [`TensorMap::left_null`], [`TensorMap::right_null`]), eigendecompositions
//! ([`TensorMap::eigh_full`], [`TensorMap::eigh_vals`], [`TensorMap::eig_full`],
//! [`TensorMap::eig_vals`]), matrix functions ([`TensorMap::exp`],
//! [`TensorMap::inv`], [`TensorMap::pinv`], and the elementwise
//! [`TensorMap::map_diagonal`] on compact diagonals), the truncation primitives
//! a truncated factorization is composed from ([`TensorMap::diagview`],
//! [`GradedSpace::find_truncated`], [`GradedSpace::truncspace`],
//! [`TensorMap::restrict_leg`] with a [`LegSelection`] of kept degeneracy
//! positions per sector, contiguous or not), concatenation ([`TensorMap::cat`],
//! [`TensorMap::absorb`]) and explicit payload conversion
//! ([`TensorMap::convert`], [`TensorMap::zeros_like`], [`TensorMap::re`],
//! [`TensorMap::im`]; exact widenings and the two lossy narrowings, never
//! implicit).
//!
//! **Device execution** is a storage type, not a runtime switch:
//! `TensorMap<R, D, CudaStorage<D>>` (feature `cuda`, created by an explicit
//! `to_cuda` and left by `to_host`) shares the Host operation bodies and
//! supplies only its own kernels (see "Operation dispatch" below). The device
//! surface is narrower than Host: an operation whose kernel is not implemented
//! for `CudaStorage` does not exist for it rather than falling back to Host.
//! See the crate-level documentation for the device synchronization model. [`TensorMap::placement`] is diagnostic
//! metadata; no `TensorMap` operation dispatches on it.
//!
//! # Operation dispatch
//!
//! Every operation has one typed body (in the family modules of this
//! directory) and is selected on two independent axes:
//!
//! - **Admission mode**, from the rule: `R::Mode` is
//!   [`MultiplicityFreeAdmissionMode`] or the checked `Generic` mode, and the
//!   bound `R::Mode: Typed*Dispatch` (`TypedTensorModeDispatch` and its
//!   subtraits `TypedTensorTransformDispatch`, `TypedTensorContractDispatch`,
//!   `TypedTensorTraceDispatch`, `TypedTruncationDispatch`, and the other
//!   `Typed*Dispatch` traits of `dispatch.rs`) supplies the mode-specific
//!   admission, layout and error type (`FacadeError<R>`). The mode impls live
//!   in `mode_dispatch.rs`.
//! - **Storage**, from the third parameter of [`TensorMap`]: `S` is `Vec<D>`
//!   (Host) or `CudaStorage<D>`, and the sealed family traits
//!   `TransformExec`, `ContractExec`, `PreparedExec`, `TraceExec`,
//!   `TwistExec`, `FactorExec`, `ReduceExec`, `ElementwiseExec`,
//!   `MaterializeExec` and `StackExec` (all subtraits of `TypedStorage`
//!   except `StackExec`, which is payload-only) supply only the storage kernels
//!   the shared body calls.
//!
//! Each kernel takes an `ExecToken` (`storage.rs`), a value only this crate can
//! construct, so a caller outside the crate can name a kernel but not skip the
//! admission its body performs.
//!
//! **CUDA bounds** are stated per impl, not once: each rule-generic `CudaStorage`
//! impl of a family trait (and each inherent CUDA
//! `impl TensorMap<R, D, CudaStorage<D>>`) restates
//! `R: MultiplicityFreeRigidSymbols<Scalar = f64> + CheckedFusionAlgebra + SectorCodec`
//! and bounds `D` by the payload trait of its family
//! (`CudaPayload`, `CudaFactorizationPayload`). A rule or payload outside those
//! bounds therefore has no device method, which is how checked `Generic`
//! providers are excluded from device factorizations.
//!
//! # Compact diagonal storage
//!
//! For Host providers, the `s` factor from `svd_compact` is compact (and stays
//! compact through `restrict_leg` on both legs). The `d` factor of `eigh_full`
//! and `eig_full`, for multiplicity-free and checked `Generic` providers
//! alike, and the `s` factor of `svd_full` when every sector is square, are
//! compact as well.
//! A compact factor holds `Σ_c k_c` values rather than the `Σ_c k_c²`
//! block-diagonal buffer it would fill, which is what TensorKit's
//! `DiagonalTensorMap` is. It is a storage property and not a type: no signature
//! mentions it, [`TensorMap::dense_data`] refuses it (call
//! [`TensorMap::materialize`] for the dense buffer), and the operations that can exploit
//! it — [`TensorMap::compose`], admitted bond-scaling [`TensorMap::contract`],
//! [`TensorMap::cat`], [`TensorMap::absorb`], [`TensorMap::scale`],
//! [`TensorMap::axpby`], [`TensorMap::adjoint`],
//! [`TensorMap::trace_pairs`] on its full-pair arm, and the reductions — do so
//! silently. Rank-(1,1) [`TensorMap::permute`] and [`TensorMap::transpose`]
//! swaps keep the compact result; an admitted [`TensorMap::braid`] reads the
//! compact source directly and publishes a dense result. The transform
//! `*_into` entries write a compact source's transformed diagonal into their
//! destination under the same single-term bijection proof. Other transform and
//! contraction geometries use the documented dense route.
//!
//! # Deliberately absent
//!
//! - The rest of the matrix-function family (trigonometric and hyperbolic
//!   members, `log`, `sylvester`, a general `sqrt`). Right solves and integer
//!   powers are compositions of [`TensorMap::adjoint`], [`TensorMap::solve`],
//!   [`TensorMap::inv`] and [`TensorMap::compose`] (the zeroth power is
//!   [`TensorMap::isomorphism`] of the domain onto itself), and elementwise
//!   maps of a compact spectrum go through [`TensorMap::map_diagonal`].
//!   General endomorphism `sqrt` additionally needs a Schur seam that does not
//!   exist below this facade.
//! - Operator overloads (`impl Add`, `impl Mul`): they cannot return `Result`,
//!   and panicking operators would contradict the passthrough-error contract.
//! - `conj`, which stays design-gated on its open correctness question for
//!   non-self-dual sectors. [`TensorMap::adjoint`] is the TensorKit-style lazy
//!   parent view for dense storage. A compact diagonal, in either admission
//!   mode, keeps its direct `O(Σ_c k_c)` conjugation path.
//! - Some outer multiplicity factorizations. Checked `Generic` providers
//!   have provider-neutral SVD/QR/LQ, shape-based null spaces, and the admitted
//!   matrix-function subset; each operation documents its own lazy-adjoint and
//!   compact-storage boundary.
//!
//! Construction consumes only the transactional checked admission path, so a
//! provider that reports an invalid or unrepresentable algebra fails with a
//! typed error and publishes no layout, cache, or admission state.

#![cfg_attr(
    not(feature = "cuda"),
    doc = "```compile_fail\nuse symtn::typed::CudaStorage;\n```"
)]
#![cfg_attr(
    not(feature = "cuda"),
    doc = "```compile_fail\nuse symtn::typed::TensorMap;\nfn no_cuda<R>(tensor: &TensorMap<R, f64>) { let _ = tensor.to_cuda(); }\n```"
)]

use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::sync::Arc;

use smallvec::SmallVec;
use symtn_core::{
    validate_unit_layout_correspondence_checked,
    validate_unit_layout_correspondence_generic_checked, BlockRef, BlockStructure, BlockView,
    CoupledSectorRegion, CoupledTreeExtent, FusionProductSpace, FusionTreeHomSpace,
    PreparedTreePairOperation, ProductFusionRule, ProductSector, SectorId, SectorLeg,
    UnitLegInsertion,
};
use symtn_core::{HostReadableStorage, Placement, TensorStorage};
#[cfg(feature = "cuda")]
use symtn_dense::{
    cuda_copy_region_into, cuda_eigh_region, cuda_gemm_region_into,
    cuda_hermitian_regions as dense_cuda_hermitian_regions, cuda_qr_region as dense_cuda_qr_region,
    cuda_svd_gauge_phases, cuda_svd_region as dense_cuda_svd_region, CudaDenseContext,
    CudaDenseStorage, CudaSvdGaugeWeights,
};
use symtn_operations::scale_value;
#[cfg(feature = "cuda")]
use symtn_operations::{CudaTreeTransformDestination, StorageGemm};
use symtn_tensors::{
    expand_physical_host, project_physical_host, BoundDynamicFusionMapSpace, BoundDynamicTensorRef,
    DynamicFusionMapSpace, OutputAxisOrder, OwnedCatCopy, OwnedCatSide, TreeTransformOperation,
    TreeTransformOperationKind, ValidatedDynamicFusionLayout,
};

use crate::sector::{
    CanonicalUnitFusionRule, CategoricalScalar, CheckedCanonicalUnitFusionRule,
    CheckedFusionAlgebra, CheckedGenericAdmissionMode, CheckedGenericFusion, CheckedGenericPivotal,
    CheckedGenericRigidSymbols, MultiplicityFreeAdmissionMode, MultiplicityFreeFusionSymbols,
    MultiplicityFreeRigidSymbols, PhysicalFusionBasis, ProductSectorCodec, SectorCodec,
    TypedSectorAdmission,
};
/// Causes carried by [`Error`] and [`GenericTensorError`] variants.
pub use symtn_core::{CheckedGenericStructureError, CoreError, FusionAlgebraError};
/// Flat f64 CUDA storage used by explicit typed ownership transfer.
#[cfg(feature = "cuda")]
pub use symtn_tensors::cuda::CudaStorage;
#[cfg(feature = "cuda")]
use symtn_tensors::cuda::CudaStorageGemm;
pub use symtn_tensors::CheckedGenericPlanError;
pub use symtn_tensors::OperationError;

/// Error returned by physical-basis expansion and projection.
pub type PhysicalDenseError<E> = symtn_tensors::PhysicalConversionError<E>;

/// Every fallible method here returns this error.
pub use crate::error::{Alternative, Error};
/// The runtime every constructor here takes (prepared structural state) and
/// its builder.
pub use crate::runtime::{Runtime, RuntimeBuilder};
/// The complex payload scalars.
pub use num_complex::{Complex32, Complex64};
/// Block and fusion-tree keys and vertex labels read from a [`TensorMap`],
/// and the error of a checked fusion-space query.
pub use symtn_core::{
    BlockKey, BlockKeyKind, CheckedFusionSpaceError, FusionTreeGroupKey, FusionTreeKey,
    FusionTreePairKey, MultiplicityIndex, OpaqueBlockKey,
};
/// The caller-owned execution resources every numerical operation takes as
/// its first argument (`engine: &mut dyn Engine`): tenferro's engine trait,
/// its CPU engine and the CPU engine's thread pool.
///
/// How grouped dense GEMMs (the recoupling and contraction batches) use a
/// pooled `CpuEngine`'s threads is the engine's batch policy, an engine-owned
/// setting symtn never overrides. On the native provider the default
/// (`CpuBatchStrategy::Auto`) runs a batch on symtn's block GEMM kernel,
/// which takes the batch's thread count from the engine's cost rule over its
/// estimated serial time and its lane count from the engine's batch decision;
/// the lanes share those threads, so a batch of a few large blocks gives each
/// lane several threads. A caller can still give each GEMM the whole pool:
/// `CpuEngine::new(pool).with_batch_policy(CpuBatchPolicy::new(CpuBatchStrategy::Inner))`,
/// or `set_batch_policy` on an existing engine, with `CpuBatchPolicy` and
/// `CpuBatchStrategy` from `tenferro_cpu`. A BLAS provider under `Auto` runs
/// the batch's vendor GEMMs sequentially on the calling thread (vendor threads
/// are outside the engine's lanes); forcing `Inner` there runs the GEMMs on
/// the native kernels instead of the vendor library.
pub use symtn_dense::{CpuEngine, Engine, Pool};
/// The caller-owned CUDA engine every device operation borrows, and the
/// device id it opens on.
///
/// Opening an engine compiles tenferro-gpu's device kernels with NVRTC, seconds
/// per kernel module, so hold one long-lived engine per device (and thread)
/// rather than one per task. `TENFERRO_CUDA_ARTIFACT_CACHE=<dir>` persists the
/// compiled kernels in `<dir>` for later engines and processes.
#[cfg(feature = "cuda")]
pub use symtn_dense::{CudaDeviceId, CudaEngine};
/// The device-scalar supertrait of [`CudaPayload`] and its real counterpart.
#[cfg(feature = "cuda")]
pub use symtn_dense::{CudaRealScalar, CudaScalar};
/// The dense-scalar supertrait of the payload traits; its `Eig` is the
/// payload of [`TensorMap::eig_full`].
pub use symtn_matrixalgebra::FactorScalar;
pub use symtn_matrixalgebra::HermitianTol;
/// The spectrum-magnitude bound of [`GradedSpace::find_truncated`]. Concrete
/// `f64`/`Complex64` callers never name it, but a caller generic over the
/// payload must, so it is re-exported here rather than left unnameable
/// outside the crate.
pub use symtn_matrixalgebra::SpectrumMagnitude;
/// Named factor sets returned by the factorization methods of [`TensorMap`].
/// They are defined next to the expert factorizations in `symtn-matrixalgebra`,
/// which return the same types, and re-exported here.
pub use symtn_matrixalgebra::{Eig, Eigh, LeftPolar, Lq, Qr, RightPolar, Svd};
/// [`GradedSpace::find_truncated`] and the truncated factorizations take
/// these.
pub use symtn_matrixalgebra::{Truncation, TruncationError, TruncationSpace};
/// The coefficient action a payload scalar supports; physical-basis
/// expansion and projection name it as a bound.
pub use symtn_tensors::RecouplingCoefficientAction;

use symtn_matrixalgebra::seam::{rescaled_power_norm, CheckedGenericFactorPlanError};
use symtn_matrixalgebra::BoundDynFactor;

use crate::runtime::RuntimeIdentity;
use crate::runtime::{Ctx, Ctxs};
pub use crate::tensor_product::CheckedGenericTensorProductError;
use crate::tensor_product::{
    internal_layout_error, tensorproduct_owned_checked_generic,
    tensorproduct_owned_multiplicity_free,
};

mod batched;
#[doc(hidden)]
pub use batched::StackExec;
pub use batched::{
    BatchError, BatchMemberRepresentation, ComposePlan, ContractPlan, ContractWorkspace,
    EighFullPlan, EighFullWorkspace, EighStackOutput, MemberFault, SignatureField,
    StackedTensorMap, StructureSignature,
};

#[cfg(test)]
mod contract_stacking_tests;
mod serialization;
#[cfg(test)]
mod spectrum_match_tests;
#[cfg(test)]
mod view_tests;
pub use serialization::{
    DecodeError, DecodeLimits, EncodeError, PersistedScalar, TypedPersistenceCodec,
};

mod scalar;
#[doc(hidden)]
pub use linear_ops::ElementwiseExec;
use linear_ops::{
    axpby_into_checked, destination_precheck, destination_slice, destination_storage_mut,
    require_destination_space,
};
pub(crate) use scalar::ScalarOps;
pub use scalar::{AdvancedLinalgScalar, FactorizationScalar, TensorScalar};
#[cfg(feature = "cuda")]
pub use scalar::{CudaFactorizationPayload, CudaPayload};
mod numeric;
pub(crate) use numeric::{
    absorb_compact_source, absorb_mapped, coupled_region_inner, coupled_region_weighted_sum,
    multiplicity_free_dim, sector_regions, validate_norm_p, weighted_trace,
};
use numeric::{max_abs, scaled_power};
mod block_layout;
pub(crate) mod probe;
pub(crate) use block_layout::{
    apply_fill, cat_homspace, check_flip_layout_identity, compile_cat_plan, flip_block_factor,
    flip_toggled_homspace, logical_adjoint_axes_to_parent, lower_adjoint_tree_transform_operation,
    map_checked_unit_layout_error, oplus_sector_legs, scale_blocks_impl, twist_factor_with_inverse,
    twist_is_identity_over_blocks, with_planar_axes, CatOperandData, CatOperandLayout, Fill,
    PlanarRequestKind,
};
use block_layout::{fuse_sector_content, uncoupled_sector_of_leg};
pub use block_layout::{Direction, Duality, Side};
#[cfg(feature = "cuda")]
mod cuda_factor;
#[cfg(feature = "cuda")]
pub(crate) use cuda_factor::{
    assemble_aligned_left_factor, assemble_left_factor, assemble_right_factor,
    copy_left_factor_treewise, copy_whole_factor, cuda_download_spectra, cuda_hermitian_regions,
    cuda_qr_region, cuda_svd_region, dense_err, ensure_cuda_owns, fill_diagonal_values,
    require_finite_cuda_input, typed_cuda_eigh_region, upload_selector,
};
#[cfg(feature = "cuda")]
use cuda_factor::{
    download_cuda_reduction_partials, executed_routes, validate_cuda_svd_middle_regions,
    TypedCudaQrScratch, TypedCudaSvdScratch,
};
#[cfg(feature = "cuda")]
use symtn_matrixalgebra::seam::{compact_factor_routes, CompactFactorPlan, CompactFactorRoute};
mod generic_error;
pub use generic_error::GenericTensorError;
mod dispatch;
pub use dispatch::{
    BondOutput, BondTransform, FacadeError, TypedAdjointSpace, TypedSpaceModeDispatch,
    TypedTensorConstructionDispatch, TypedTensorContractDispatch, TypedTensorFlipDispatch,
    TypedTensorModeDispatch, TypedTensorProductDispatch, TypedTensorRootDispatch,
    TypedTensorTraceDispatch, TypedTensorTransformDispatch, TypedTensorTwistDispatch,
    TypedTensorUnitDispatch, TypedTruncationDispatch,
};
use dispatch::{
    MultiplicityFreeContractExecution, MultiplicityFreeTransformExecution, TwistCoefficient,
};
mod factorize;
#[doc(hidden)]
pub use factorize::FactorExec;
pub use factorize::FusionMode;
mod checked_generic_contract;
mod mode_dispatch;
use checked_generic_contract::write_identity_blocks_generic;
#[doc(hidden)]
pub use symtn_tensors::{reject_non_symmetric_contraction, NON_SYMMETRIC_CONTRACTION_UNSUPPORTED};
mod space;
use space::{
    require_restriction_set, require_selected_leg_of, restricted_space, restriction_runs,
    space_with_replaced_legs,
};
pub use space::{GradedSpace, LegSelection, TruncatedSelection};
mod fusion_tree;
use fusion_tree::{
    add_spectrum_into, decode_block_fusion_trees, diagonal_factor_on_bound, exp_spectrum,
    full_svd_compact_bond, inv_spectrum, is_diagonal_bond_space, map_spectrum,
    prepare_product_operand, reject_singular_compact_divisor, scatter_spectrum, spectra_disagree,
    wrap_factor_on, TypedData,
};
#[cfg(test)]
use fusion_tree::{diagonal_factor_on, full_svd_spectrum_matches_bonds};
pub use fusion_tree::{
    BlockFusionTrees, CoupledBlock, CoupledBlockPayload, FusionTreeLabels, SectorSpectrum,
};
mod tensor_repr;
use tensor_repr::{
    borrowed_view_unsupported, owned_repr, TypedAdjointView, TypedTensorBody, TypedTensorRepr,
};
pub use tensor_repr::{PayloadConversion, PhysicalDense, TensorMap, TensorRef};
mod cat;
mod compact_arms;
mod construction;
mod contract_ops;
#[doc(hidden)]
pub use checked_generic_contract::TraceExec;
use contract_ops::contract_request;
pub use contract_ops::ContractSpec;
pub use contract_ops::{ContractExec, ContractReplay, PreparedExec};
/// The seam `symtn-network` drives; not part of the facade.
#[doc(hidden)]
pub mod __network;
#[cfg(feature = "cuda")]
mod cuda_contract;
#[cfg(feature = "cuda")]
mod cuda_ops;
#[cfg(feature = "cuda")]
mod cuda_scalar;
#[cfg(feature = "cuda")]
mod cuda_transfer;
#[cfg(feature = "cuda")]
mod cuda_transform;
mod inspection;
#[doc(hidden)]
pub use inspection::MaterializeExec;
mod linear_ops;
mod network_seam;
#[cfg(test)]
use network_seam::NetworkDegeneracyRestriction;
mod reduction_ops;
#[doc(hidden)]
pub use reduction_ops::ReduceExec;
mod restrict;
mod storage;
pub(crate) use storage::admit_host_engine;
pub(crate) use storage::sealed::{ExecToken, EXEC};
#[doc(hidden)]
pub use storage::TypedStorage;
#[cfg(feature = "cuda")]
mod cuda_storage;
mod transform_ops;
mod twist_flip;
use transform_ops::{
    checked_generic_exec_error, generic_insert_unit, generic_remove_unit,
    lazy_adjoint_of_transformed_parent, map_spectrum_dtype, scale_by_beta, transform_source_len,
    with_transform_into_source, CheckedGenericTreeLane, HostTreeLane, MultiplicityFreeTreeLane,
};
#[doc(hidden)]
pub use transform_ops::{TransformExec, TransformExecError, TransformIntoSource, TreeReplay};
#[doc(hidden)]
pub use twist_flip::TwistExec;

/// In-module gates on the typed facade's private state.
///
/// These live inside the module on purpose: they assert what `tests/` cannot
/// see without accessors the facade does not otherwise need — which `Arc`
/// holds what ([`TypedTensorBody`], #580), dense-kernel and solver call
/// counts, cache warmth, and device call counts. The module itself keeps the
/// representation gates and the shared fixtures; its children group the
/// remaining gates by operation family: multiplicity-free and checked
/// factorizations, lazy-adjoint redirects, CUDA, `*_into` overwrite, compact
/// transforms/cat/absorb, contraction and network restriction. Public semantic
/// oracles live in `tests/typed_facade.rs`; dense-cache behavior lives in
/// `tests/typed_diagonal_allocations.rs`.
#[cfg(test)]
mod representation_gates;
