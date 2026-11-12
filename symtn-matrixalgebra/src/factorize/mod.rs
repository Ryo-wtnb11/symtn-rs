use std::collections::BTreeMap;

use rustc_hash::{FxHashMap, FxHashSet};
use std::fmt;
use std::sync::Arc;

use num_complex::Complex64;
use num_traits::{Float, Zero};
#[cfg(test)]
use symtn_core::InfallibleGeneric;
use symtn_core::{
    BlockKey, BlockRef, BlockStructure, CheckedGenericFusion, CheckedGenericStructureError,
    CoreError, CoupledSectorRegion, CoupledTreeExtent, FusionProductSpace, FusionRule,
    FusionTreeHomSpace, FusionTreeKey, FusionTreePairKey, MultiplicityFreeRigidSymbols,
    RuleIdentity, SectorId, SectorLeg, SectorOrderKey, SectorStructure,
};
use symtn_dense::{
    arity_mismatch, check_contiguous_output, DenseBackend, DenseDotConfig, DenseError,
    DenseExecutor, DenseFactorization, DenseOwned, DenseRead, DenseTensor, DenseView, DenseViewMut,
    DenseWrite, FactorDest, FactorFailure,
};

use symtn_core::{CheckedGenericAdmissionMode, MultiplicityFreeAdmissionMode};
pub use symtn_tensors::BoundDynamicTensorRef;
use symtn_tensors::{
    zeroed_payload, BoundDynamicFusionMapSpace, CoefficientAlgebra, DenseBlockScalar,
    DenseRecouplingScalar, DynamicFusionMapSpace, RigidCoefficientAlgebra,
    ValidatedDynamicFusionLayout, ZeroBytes,
};

#[cfg(test)]
use crate::results::Svd;
use crate::results::{LeftPolar, Lq, Qr, RightPolar};
use crate::truncation::{select_truncation, Truncation, WeightedSpectrum};
use symtn_tensors::OperationError;

/// Binds `$geometry` to the admitted input's per-sector geometry
/// (`&[CoupledSectorRegion]` or `&[SectorMatricization<D>]`) for the
/// `SectorGeometry`-generic publication helpers.
macro_rules! with_input_geometry {
    ($input:expr, |$geometry:ident| $body:expr) => {
        match $input {
            InputMatricizations::Regions { regions, .. } => {
                let $geometry: &[CoupledSectorRegion] = regions;
                $body
            }
            InputMatricizations::Packed(matrices) => {
                let $geometry = matrices.as_slice();
                $body
            }
        }
    };
}

mod authority;
mod bound;
mod compact_plan;
mod dense_stage;
mod diagonal;
mod eig;
mod inverse;
mod landing;
mod matricize;
mod null_space;
pub mod op;
mod polar;
mod publish_checked;
mod publish_mf;
mod qr_lq;
mod scalar;
mod source;
mod svd;

#[cfg(test)]
mod hermitian_scale_tests;

#[cfg(test)]
#[path = "../tests/sector_matricization/mod.rs"]
mod sector_matricization_tests;

// Blanket re-exports so every existing `crate::factorize::<name>` path
// (used by `compose.rs`, `matrix_functions.rs`, and `lib.rs`'s own
// `pub use factorize::{...}`) keeps resolving unchanged after the move.
// Each child's own `pub`/`pub(crate)` items pass straight through; each
// child's `pub(super)` items (bumped from fully-private during the move)
// become reachable here and, by re-export, throughout the crate.
// `null_space` and `polar` have no crate-internal consumer outside the
// explicit `pub use` blocks below (every item another file needs is
// already named there), so a blanket glob re-export of either would be
// unused. `polar`'s direction helpers, also used by `svd`, are imported by
// name below.
#[cfg(test)]
pub(crate) use authority::multiplicity_free_output_space;
use authority::*;
pub use authority::{spectrum_bond, FactorMode, FactorSpaceAuthority};
pub(crate) use bound::*;
pub(crate) use compact_plan::*;
pub use compact_plan::{compact_factor_routes, CompactFactorPlan, CompactFactorRoute};
// `dense_stage` items are consumed only by sibling factorization modules.
use dense_stage::*;
pub use diagonal::admit_compact_diagonal;
use diagonal::*;
pub(crate) use eig::*;
pub use inverse::pinv_diagonal_spectrum;
pub(crate) use inverse::*;
use landing::*;
pub(crate) use matricize::*;
use publish_checked::*;
// `publish_mf` exposes crate-visible items only to the unit tests.
#[cfg(test)]
pub(crate) use publish_mf::*;
#[cfg(not(test))]
use publish_mf::*;
pub(crate) use qr_lq::*;
pub use source::{factor, ExecutorLease, FactorOp, FactorOutput, FactorSource};
use source::{factor_from_source, no_adjoint};

/// The proof that a factorization hook is called from inside this crate.
///
/// Why not sealing alone: a sealed supertrait stops external impls and
/// qualified paths, not calls through a generic bound on the public trait
/// (`O::run`, `M::svd_compact_dense`), which would skip [`factor`]'s shared
/// driver. The field is private and [`TOKEN`](exec::TOKEN) is `pub(crate)`, so
/// no value of this type exists outside the crate.
mod exec {
    use super::*;

    #[derive(Clone, Copy)]
    pub struct Token(());

    /// The only [`Token`].
    pub(crate) const TOKEN: Token = Token(());

    /// The execution of one [`FactorOp`]; only [`factor`] calls it.
    pub trait Run<R, D: FactorScalar> {
        fn run<M, L>(
            self,
            _: Token,
            engine: &mut dyn symtn_dense::Engine,
            lease: L,
            source: FactorSource<'_, R, D>,
        ) -> Result<<Self as FactorOp<R, D>>::Output, M::Error>
        where
            Self: FactorOp<R, D>,
            M: FactorMode<R>,
            L: ExecutorLease;
    }
}
use exec::{Run, Token, TOKEN};
#[cfg(test)]
pub(crate) use svd::scale_axis_by_spectrum;
pub(crate) use svd::*;

// Explicit overrides: an item re-exported through the blanket globs above
// resolves at `pub(crate)` (the glob's own visibility), even when the item
// is itself `pub` inside its child module -- Rust's re-export visibility is
// the minimum of the two. `lib.rs`'s own `pub use factorize::{...}` needs
// these specific names to be fully `pub` here, so each one is reimported by
// name; an explicit import always shadows the same name brought in by a
// glob.
pub use bound::BoundDynFactor;
pub use compact_plan::CheckedGenericFactorPlanError;
pub use dense_stage::in_linalg_scope;
pub use eig::{
    ascending_eigh_order, eig_full_dyn, eigh_sector_into, require_finite_eigh_input,
    validate_hermitian_regions, EigFullDyn, EighFullDyn, EighScratch, HermitianTol,
    EIGH_EIGENVALUE_CHECK,
};
#[cfg(test)]
pub(crate) use eig::{eig_vals_dyn, eigh_vals_dyn};
pub use matricize::{
    coupled_sector_block_dimensions_generic_checked, validate_endomorphism_region_stacking,
    EndomorphismFamily, EIGH_NOT_HERMITIAN, EIGH_NOT_SQUARE,
};
#[cfg(test)]
pub(crate) use null_space::{
    left_null, left_null_dyn_checked_generic, right_null, right_null_dyn_checked_generic,
};
#[cfg(test)]
pub(crate) use null_space::{left_null_dyn, right_null_dyn};
#[cfg(test)]
pub(crate) use polar::{
    left_polar, left_polar_dyn, left_polar_dyn_checked_generic, right_polar, right_polar_dyn,
    right_polar_dyn_checked_generic,
};
pub(crate) use scalar::require_finite_factor_input;
pub use scalar::{nonfinite_factor_input, FactorFamily};
pub use scalar::{FactorScalar, SectorSpectrum, SpectrumMagnitude};
pub use svd::{
    axpby_axis_by_spectrum_into, decide_bond_truncation, diagonal_bond_data,
    rectangular_diagonal_bond_tensor, scale_axis_by_spectrum_mapped, SvdFactorsDyn,
    SvdFullFactorsDyn,
};
