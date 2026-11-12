//! Named factor sets returned by the factorizations.
//!
//! Each type only names the factors; it neither computes nor converts them.
//! The type parameter is the factor type: `symtn::typed::TensorMap` on the
//! user facade (Host or CUDA storage), the storage-tagged factors of
//! `seam::factor` on this crate.

/// Singular-value decomposition `t = u * s * vh`.
///
/// Returned by the facade's `svd_compact` and `svd_full` (Host and CUDA,
/// multiplicity-free and checked-Generic providers).
///
/// # Order
///
/// Order is per coupled sector; no order across sectors is implied. Within
/// each sector the singular values on the diagonal of `s` are descending, and
/// the columns of `u` and the rows of `vh` follow them. This is the
/// [`crate::SectorSpectrum`] contract.
///
/// # `s` is diagonal
///
/// Mathematically, `s` is diagonal in every coupled sector: square for the
/// compact SVD, rectangular (`m_c x n_c`) for the full SVD.
///
/// # Storage of `s`
///
/// The facade's Host `svd_compact` stores `s` as compact diagonal data
/// (`sum_c k_c` values) for multiplicity-free and checked-Generic providers.
/// Its multiplicity-free `svd_full` also retains compact `s` for admitted
/// owned compact-diagonal inputs; U and Vh remain dense. Other full-SVD
/// inputs, CUDA routes, and this crate's expert tensor-factor entry points
/// store S densely. A facade caller needing a dense buffer can explicitly
/// materialize S.
#[derive(Clone, Debug)]
pub struct Svd<T> {
    /// Left singular vectors, `codomain(t) <- W`.
    pub u: T,
    /// Singular values, `W <- W'` (`W' = W` for the compact SVD).
    pub s: T,
    /// Adjoint right singular vectors, `W' <- domain(t)`.
    pub vh: T,
}

/// Hermitian eigendecomposition `t = v * d * v^H` of an endomorphism.
///
/// Returned by the facade's `eigh_full` (Host and CUDA, multiplicity-free and
/// checked-Generic providers).
///
/// # Order
///
/// Order is per coupled sector; no order across sectors is implied. Within
/// each sector the real eigenvalues on the diagonal of `d` are ascending, as
/// LAPACK and MatrixAlgebraKit return them, and the columns of `v` follow
/// them. No basis within a degenerate eigenspace is promised.
///
/// # `d` is diagonal
///
/// Mathematically, `d` is diagonal in every coupled sector.
///
/// # Storage of `d`
///
/// Independently of the mathematics, the facade's Host `eigh_full` stores `d`
/// as compact diagonal data (`sum_c n_c` values) for both multiplicity-free
/// and checked-Generic providers. CUDA `eigh_full` stores the diagonal matrix
/// densely.
#[derive(Clone, Debug)]
pub struct Eigh<T> {
    /// Real eigenvalues, `W <- W`.
    pub d: T,
    /// Unitary eigenvectors, `codomain(t) <- W`, one column per eigenvalue.
    pub v: T,
}

/// General eigendecomposition `t * v = v * d` of an endomorphism, with
/// complex factors.
///
/// Returned by the facade's Host `eig_full` (multiplicity-free and
/// checked-Generic providers).
///
/// # Order
///
/// Order is per coupled sector; no order across sectors is implied. Within
/// each sector the eigenvalues on the diagonal of `d` are ascending in
/// lexicographic `(real(λ), imag(λ))` order (MatrixAlgebraKit's diagonal
/// `eig_sortby`, applied to dense sectors too), and the columns of `v` follow
/// them. No basis within a degenerate eigenspace is promised.
///
/// # `d` is diagonal
///
/// Mathematically, `d` is diagonal in every coupled sector.
///
/// # Storage of `d`
///
/// Independently of the mathematics, the facade's Host `eig_full` stores `d`
/// as compact diagonal data (`sum_c n_c` values) for both multiplicity-free
/// and checked-Generic providers.
#[derive(Clone, Debug)]
pub struct Eig<T> {
    /// Complex eigenvalues, `W <- W`.
    pub d: T,
    /// Right eigenvectors, `codomain(t) <- W`, one column per eigenvalue,
    /// each of unit 2-norm with its largest-magnitude component real and
    /// non-negative (MatrixAlgebraKit `geev!` then `gaugefix!`).
    pub v: T,
}

/// QR factorization `t = q * r` (compact or full).
#[derive(Clone, Debug)]
pub struct Qr<T> {
    /// Factor with orthonormal columns (unitary for the full QR),
    /// `codomain(t) <- W`.
    pub q: T,
    /// Upper-triangular factor with real non-negative diagonal,
    /// `W <- domain(t)`.
    pub r: T,
}

/// LQ factorization `t = l * q` (compact or full).
#[derive(Clone, Debug)]
pub struct Lq<T> {
    /// Lower-triangular factor with real non-negative diagonal,
    /// `codomain(t) <- W`.
    pub l: T,
    /// Factor with orthonormal rows (unitary for the full LQ),
    /// `W <- domain(t)`.
    pub q: T,
}

/// Left polar decomposition `t = w * p` (TensorKit `left_polar`, which
/// returns `W, P`).
///
/// Left and right polar are separate types because their factors differ in
/// order and in which one is the isometry.
#[derive(Clone, Debug)]
pub struct LeftPolar<T> {
    /// Isometry on the input space, `codomain(t) <- domain(t)`.
    pub w: T,
    /// Positive-semidefinite factor, `domain(t) <- domain(t)`.
    pub p: T,
}

/// Right polar decomposition `t = p * wh` (TensorKit `right_polar`, which
/// returns `P, Wᴴ`).
#[derive(Clone, Debug)]
pub struct RightPolar<T> {
    /// Positive-semidefinite factor, `codomain(t) <- codomain(t)`.
    pub p: T,
    /// Coisometry on the input space, `codomain(t) <- domain(t)`.
    pub wh: T,
}
