//! The operations of [`factor`](super::factor), one value each; the result
//! type of each is its [`FactorOp::Output`](super::FactorOp::Output).
//!
//! Why not one shape-carrying op for QR and LQ (compact and full share a
//! result type): a marker per operation keeps the selection static and
//! matches the facade's operations one to one.

use super::HermitianTol;

/// Compact SVD: `(u, vh, singular values)`; the spectrum bond is the caller's.
pub struct SvdCompact;
/// Full SVD.
pub struct SvdFull;
/// Singular values, descending per coupled sector.
pub struct SvdVals;
/// Compact QR.
pub struct QrCompact;
/// Full QR.
pub struct QrFull;
/// Compact LQ.
pub struct LqCompact;
/// Full LQ.
pub struct LqFull;
/// Left null space (MatrixAlgebraKit `qr_null!`).
pub struct LeftNull;
/// Right null space (MatrixAlgebraKit `lq_null!`).
pub struct RightNull;
/// Left polar decomposition `W * P`.
pub struct LeftPolar;
/// Right polar decomposition `P * Wh`.
pub struct RightPolar;
/// Hermitian eigendecomposition, ascending per coupled sector.
pub struct EighFull {
    pub hermitian_tol: HermitianTol,
}
/// Hermitian eigenvalues, ascending per coupled sector.
pub struct EighVals {
    pub hermitian_tol: HermitianTol,
}
/// General eigendecomposition.
pub struct EigFull;
/// General eigenvalues.
pub struct EigVals;
