#![cfg_attr(not(test), forbid(unsafe_code))]
// Tests only: the counting allocator in `tests::allocation_bytes` is the one
// exemption; `forbid` could not be relaxed for it.
#![cfg_attr(test, deny(unsafe_code))]

//! MatrixAlgebraKit-style factorizations for symtn fusion tensors.
//!
//! Crate mapping onto the TensorKit ecosystem: `symtn-core` +
//! `symtn-operations` together play TensorKit.jl's role (structures and
//! symmetric execution, including the fusion-tree transforms); strided-rs and
//! tenferro play Strided.jl / TensorOperations.jl's symmetry-agnostic dense
//! role underneath; and this crate is the MatrixAlgebraKit layer applied at
//! the fusion-tensor level — blockwise factorizations over the coupled-sector
//! matricization, spectrum truncation, and the derived matrix functions.
//!
//! Every operation decomposes into three kinds of work: dense factorizations
//! and GEMM on the device boundary ([`symtn_dense::DenseExecutor`]),
//! scalar decisions over per-sector spectra on the host
//! ([`truncation`], spectrum functions), and mechanical block-data movement
//! (bond slicing, adjoints) that stays behind device-capable seams.

#[cfg(test)]
mod compose;
mod factorize;
mod matrix_functions;
#[cfg(test)]
mod probe;
mod results;
pub mod truncation;

pub use factorize::{
    BoundDynFactor, FactorScalar, HermitianTol, SectorSpectrum, SpectrumMagnitude,
};
pub use results::{Eig, Eigh, LeftPolar, Lq, Qr, RightPolar, Svd};
pub use truncation::{
    select_truncation, Truncation, TruncationDecision, TruncationError, TruncationSpace,
    WeightedSpectrum,
};

/// The dynamic-rank entry points the `symtn` facade dispatches to. Not a
/// stable API: every item here exists because a `symtn` module or example
/// calls it, and the public API snapshot (`api/`) records the exact list.
#[doc(hidden)]
pub mod seam {
    pub use crate::factorize::{
        admit_compact_diagonal, ascending_eigh_order, axpby_axis_by_spectrum_into,
        compact_factor_routes, coupled_sector_block_dimensions_generic_checked,
        decide_bond_truncation, diagonal_bond_data, eig_full_from_source, eig_vals_from_source,
        eigh_full_dyn, eigh_full_from_source, eigh_sector_into, eigh_vals_from_source,
        in_linalg_scope, left_null_from_source, left_polar_adjoint_from_parent,
        left_polar_from_source, lq_compact_adjoint_from_parent, lq_compact_from_source,
        lq_full_adjoint_from_parent, lq_full_from_source, nonfinite_factor_input,
        pinv_diagonal_spectrum, qr_compact_from_source, qr_full_from_source,
        rectangular_diagonal_bond_tensor, require_finite_eigh_input, right_null_from_source,
        right_polar_adjoint_from_parent, right_polar_from_source, scale_axis_by_spectrum_mapped,
        spectrum_bond, svd_compact_adjoint_from_parent, svd_compact_dyn_checked_generic,
        svd_compact_from_source, svd_full_adjoint_from_parent, svd_full_from_source, svd_vals_dyn,
        svd_vals_from_source, validate_endomorphism_region_stacking, validate_hermitian_regions,
        BoundDynamicTensorRef, CheckedGenericFactorPlanError, CompactFactorPlan,
        CompactFactorRoute, EigFullDyn, EighFullDyn, EighScratch, EndomorphismFamily,
        ExecutorLease, FactorFamily, FactorMode, FactorOutput, FactorSource, FactorSpaceAuthority,
        SvdFactorsDyn, EIGH_EIGENVALUE_CHECK, EIGH_NOT_HERMITIAN, EIGH_NOT_SQUARE,
    };
    pub use crate::matrix_functions::{
        exp_direct_into_dyn, inv_direct_into_dyn, pinv_adjoint_parent_direct_into_dyn,
        pinv_direct_into_dyn, solve_left_direct_into_dyn,
    };
    pub use crate::truncation::rescaled_power_norm;
}

// The unit tests drive the typed const-generic wrappers, which have no
// consumer outside them (nor does `scale_axis_by_spectrum` outside this
// crate), and reach the seam through `use crate::*`.
#[cfg(test)]
use factorize::{
    eig_full, eig_vals, eigh_full, eigh_vals, left_null, left_null_dyn,
    left_null_dyn_checked_generic, left_polar, left_polar_dyn, lq_compact, lq_compact_dyn, lq_full,
    lq_full_dyn, qr_compact, qr_compact_dyn, qr_full, qr_full_dyn, right_null, right_null_dyn,
    right_null_dyn_checked_generic, right_polar, right_polar_dyn, scale_axis_by_spectrum,
    svd_compact, svd_compact_adjoint_factors_dyn, svd_compact_dyn, svd_compact_factors_dyn,
    svd_full, svd_full_adjoint_dyn, svd_full_dyn, svd_full_dyn_checked_generic, svd_vals,
};
#[cfg(test)]
use factorize::{eig_full_dyn, eig_vals_dyn, eigh_vals_dyn};
#[cfg(test)]
use factorize::{
    eig_full_dyn_checked_generic, eigh_full_dyn_checked_generic, left_polar_dyn_checked_generic,
    lq_compact_dyn_checked_generic, lq_full_dyn_checked_generic, qr_compact_dyn_checked_generic,
    qr_full_dyn_checked_generic, right_polar_dyn_checked_generic,
    svd_compact_factors_with_spectrum_dyn_checked_generic,
};
#[cfg(test)]
use matrix_functions::{
    exp, exp_into_mf, inv, inv_into_mf, pinv, pinv_adjoint_parent_into_mf, pinv_into_mf,
    solve_left_into_mf,
};
#[cfg(test)]
use seam::*;

#[cfg(test)]
mod tests;

/// The workspace tolerance rule for arithmetic test comparisons
/// (`docs/testing_numerics.md`).
#[cfg(test)]
#[path = "../../tests/support"]
mod test_numerics {
    use num_complex::{Complex32, Complex64};
    pub(crate) mod numerics;
}
