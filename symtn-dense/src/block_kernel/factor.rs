//! The native provider's factorizations of independent blocks (compact and
//! full SVD, compact and full QR, the QR null basis, LQ, Hermitian and
//! general eigendecomposition and their values-only forms): tlinalg calls per
//! block, the blocks spread over the lane driver's lanes on the caller's
//! engine.
//!
//! Each block reaches tlinalg exactly as tenferro's per-matrix entry
//! (`svd_read`, `svd_full_read`, `qr_read`, `eigh_read`, `eigvalsh_read`,
//! `svdvals_read`, `eig_read`, `eigvals`) does (same provider, same validation, same error
//! values, and for `eig` tenferro's eigenvector normalization), but on its
//! lane's child token limited to the block's placed width instead of the
//! whole engine, so a block's faer width is the engine's decision within
//! that offer.
//!
//! The owned batch ([`factorize`]) returns each factor as a fresh `Vec` from
//! the provider's `Vec` entry: its length is the factor's logical size and it
//! carries no recycled-buffer history (its capacity is the provider's
//! reservation, not necessarily the length). The destination batch
//! ([`factorize_into`], compact QR, LQ, full QR, the QR null basis, compact
//! and full SVD, Hermitian eigh, the polar decompositions, the Hermitian
//! exponential and the inverse) writes
//! each block's factors into its caller-owned destination through the
//! provider's initialized-destination entry (`*_into_init`), with one
//! workspace and one typed scratch per lane, sized before the lane's first
//! block from that lane's blocks and child context and dropped when the lane
//! finishes; a tall full QR or null basis goes through tlinalg's compact
//! Householder kernel, whose own workspace is still allocated per block. A
//! polar job composes the provider's SVD with two products on the lane's
//! context, a Hermitian exponential job the provider's eigh with one, their
//! geometry plans resolved before the lanes; these two also keep the spectrum
//! they consume in a separate real scratch of the lane. SVD and eigh write
//! their real spectra straight into the destination. An inverse job writes
//! the identity into its destination and runs the provider's LU solve there
//! in place, as tenferro's `solve_read_into` does from a copied right-hand
//! side.
//!
//! A single-matrix block's lane model is its provider's own width decision
//! ([`grant`]) over its serial time. A composite job (polar, Hermitian
//! exponential) is priced per sequential stage
//! ([`FactorInto::stage_works`]): its factorization at the work model's rate
//! with [`grant`] on that factorization's own estimates, and each product at
//! tcontract's GEMM rate with `gemm::grant` on the plan the batch resolved;
//! its serial time is the stages' sum and its time on a lane the sum of each
//! stage's time at its own width. The smaller serial passes (scaling, the
//! finite check, the Hermitian projection) are not priced.

use std::collections::HashMap;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::Arc;
use std::time::Duration;

use num_complex::{Complex32, Complex64};
use strided_view::{RawStridedMut, RawStridedRef};
use tcontract_exec::{Exec, WorkModel};
use tenferro_cpu::{with_cpu_engine, CpuEngine};
use tenferro_tensor::{ShapeVec, StrideVec, Tensor, TensorScalar};

use super::driver::{self, BlockOp, Entry, LaneMembers, OpKind};
use super::gemm::{self, ErasedPlan, Geometry};
use crate::factor_dest::{
    adjoint_items, compact_mut, copytri_items, map_transposed_items, polar_direction,
    project_hermitian_items, scale_exp_factor, scale_polar_factor, Element, FactorSink,
};
use crate::lanes::{self, Grant, GrantKey, LaneTables, STAGES};
use crate::layout::strides_to_isize;
use crate::tenferro_adapter::{
    tenferro_error, tenferro_solve_error, tenferro_view, NOT_A_CPU_ENGINE,
};
use crate::{
    DenseError, DenseGemmBatchJob, DenseRead, DenseTensor, DenseView, DenseWrite, FactorDest,
    FactorFailure, FactorPlacement, MatrixOp,
};
use tlinalg::householder::{self, ReflectorShape};

/// One compact factorization the provider runs per block. The shared code
/// owns validation, costing and scheduling; an implementation owns only its
/// provider call and its factors' shapes.
pub(crate) trait Factor {
    /// The per-matrix entry's `DenseError` label.
    const ENTRY: &'static str;
    /// The provider op, whose name labels the provider's error payloads.
    const OP: tlinalg::Op;

    /// The op name of the per-matrix entry's own validation errors (rank,
    /// squareness, dtype, layout).
    const VALIDATION: &'static str = Self::OP.as_str();

    /// Whether the provider requires square matrices: tenferro checks this
    /// after the rank and before the zero-extent shortcut.
    const SQUARE: bool = false;

    /// Whether the provider reads each item's transpose (the first two
    /// extents and strides swapped, no copy) instead of the item itself.
    const TRANSPOSED: bool = false;

    /// tlinalg's work estimates for one `m x n` item of scalar `T`, in
    /// weighted multiply-adds: its total work and the part a team inside the
    /// item can pay back (`estimate_work`, `estimate_payback_work`).
    fn estimates<T: tlinalg::FaerScalar>(m: usize, n: usize) -> [f64; 2];

    /// The op name of the per-matrix entry's rank and squareness errors for
    /// an input of `shape`: [`Self::VALIDATION`] unless the entry routes
    /// inputs of some ranks elsewhere.
    fn validation(_shape: &[usize]) -> &'static str {
        Self::VALIDATION
    }

    /// The per-matrix entry's error for a provider failure (also of its
    /// workspace query) on an input of `shape`.
    fn provider_failure(error: tlinalg::Error, _shape: &[usize]) -> DenseError {
        tenferro_error(Self::ENTRY, provider_error(Self::OP, error))
    }

    /// The factors of one input of matrix extents `dims = [m, n]` and batch
    /// dimensions `batch`, from the provider, or empty ones when `raw` is
    /// `None` (a zero extent).
    fn factor<T: Scalar>(
        raw: Option<RawStridedRef<'_, T>>,
        dims: [usize; 2],
        batch: &[usize],
        exec: &mut Exec<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError>;
}

/// SVD: compact (`FULL` false) `[U, S, Vh]` of shapes `[m, k, b..]`,
/// `[k, b..]`, `[k, n, b..]` with `k = min(m, n)`, or full `[m, m, b..]`,
/// `[k, b..]`, `[n, n, b..]`; `S` is real. Both are tenferro's
/// `svd_typed` (the provider's `svd_into` with the `full` flag, no
/// post-processing), whose full form of an input with a zero extent is
/// identity `U` and `Vh`.
pub(crate) struct SvdJob<const FULL: bool>;

/// Compact SVD ([`SvdJob`]).
pub(crate) type Svd = SvdJob<false>;

/// Full SVD ([`SvdJob`]).
pub(crate) type SvdFull = SvdJob<true>;

impl<const FULL: bool> SvdJob<FULL> {
    /// The extents of `U`'s columns and `Vh`'s rows.
    fn vector_dims(m: usize, n: usize) -> [usize; 2] {
        if FULL {
            [m, n]
        } else {
            [m.min(n); 2]
        }
    }
}

impl<const FULL: bool> Factor for SvdJob<FULL> {
    const ENTRY: &'static str = if FULL { "svd_full_read" } else { "svd_read" };
    // tenferro reports a full SVD's provider failures under `Op::Svd` too,
    // and only its own validation as `svd_full`.
    const OP: tlinalg::Op = tlinalg::Op::Svd;
    const VALIDATION: &'static str = if FULL { "svd_full" } else { Self::OP.as_str() };

    fn estimates<T: tlinalg::FaerScalar>(m: usize, n: usize) -> [f64; 2] {
        use tlinalg::svd::{estimate_payback_work, estimate_work};
        [estimate_work::<T>(m, n), estimate_payback_work::<T>(m, n)]
    }

    fn factor<T: Scalar>(
        raw: Option<RawStridedRef<'_, T>>,
        [m, n]: [usize; 2],
        batch: &[usize],
        exec: &mut Exec<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        let k = m.min(n);
        let [u_cols, v_rows] = Self::vector_dims(m, n);
        let (mut u, mut s, mut vt) = (Vec::new(), Vec::new(), Vec::new());
        match raw {
            Some(raw) => tlinalg::svd::svd(Self::OP, raw, FULL, &mut u, &mut s, &mut vt, exec)
                .map_err(|err| tenferro_error(Self::ENTRY, provider_error(Self::OP, err)))?,
            None if FULL => {
                let Ok(items) = checked_len(batch) else {
                    // The output metadata refuses the batch; an output it
                    // would admit still cannot hold the identities.
                    return Err(output::<Self, T>(&[m, m], batch, Vec::new())
                        .err()
                        .unwrap_or(DenseError::ElementCountOverflow));
                };
                u = vec![T::ZERO; checked_len(&[m, m, items])?];
                vt = vec![T::ZERO; checked_len(&[n, n, items])?];
                identity_columns(&mut u, m, m, 0);
                identity_columns(&mut vt, n, n, 0);
            }
            None => {}
        }
        Ok(vec![
            output::<Self, T>(&[m, u_cols], batch, u)?,
            output::<Self, <T as TensorScalar>::Real>(&[k], batch, s)?,
            output::<Self, T>(&[v_rows, n], batch, vt)?,
        ])
    }
}

/// Compact QR: `[Q, R]` of shapes `[m, k, b..]`, `[k, n, b..]` with
/// `k = min(m, n)`, in the provider's gauge.
pub(crate) struct Qr;

impl Factor for Qr {
    const ENTRY: &'static str = "qr_read";
    const OP: tlinalg::Op = tlinalg::Op::Qr;

    fn estimates<T: tlinalg::FaerScalar>(m: usize, n: usize) -> [f64; 2] {
        use tlinalg::qr::{estimate_payback_work, estimate_work};
        [estimate_work::<T>(m, n), estimate_payback_work::<T>(m, n)]
    }

    fn factor<T: Scalar>(
        raw: Option<RawStridedRef<'_, T>>,
        [m, n]: [usize; 2],
        batch: &[usize],
        exec: &mut Exec<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        let k = m.min(n);
        let (mut q, mut r) = (Vec::new(), Vec::new());
        if let Some(raw) = raw {
            tlinalg::qr::qr(Self::OP, raw, &mut q, &mut r, exec)
                .map_err(|err| tenferro_error(Self::ENTRY, provider_error(Self::OP, err)))?;
        }
        Ok(vec![
            output::<Self, T>(&[m, k], batch, q)?,
            output::<Self, T>(&[k, n], batch, r)?,
        ])
    }
}

/// Compact LQ: `[L, Q]` of shapes `[m, k, b..]`, `[k, n, b..]` with
/// `k = min(m, n)`, from the provider's QR of the transposed read:
/// `X^T = Q' R'` gives `X = R'^T Q'^T`, so `L = R'^T` and `Q = Q'^T`, each
/// transposed inside the job. No adjoint of `X` is staged.
///
/// Why the transpose, not the adjoint, of `X`: a strided read can swap axes
/// but not conjugate, and `Q'^T` has orthonormal rows like `Q'^H`
/// (`Q'^T conj(Q') = conj(Q'^H Q') = I`).
pub(crate) struct Lq;

impl Factor for Lq {
    const ENTRY: &'static str = Qr::ENTRY;
    const OP: tlinalg::Op = Qr::OP;
    const TRANSPOSED: bool = true;

    fn estimates<T: tlinalg::FaerScalar>(m: usize, n: usize) -> [f64; 2] {
        Qr::estimates::<T>(n, m)
    }

    fn factor<T: Scalar>(
        raw: Option<RawStridedRef<'_, T>>,
        [m, n]: [usize; 2],
        batch: &[usize],
        exec: &mut Exec<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        let k = m.min(n);
        let (mut q, mut r) = (Vec::new(), Vec::new());
        if let Some(raw) = raw {
            tlinalg::qr::qr(Self::OP, raw, &mut q, &mut r, exec)
                .map_err(|err| tenferro_error(Self::ENTRY, provider_error(Self::OP, err)))?;
        }
        Ok(vec![
            output::<Self, T>(&[m, k], batch, transpose_items(&r, k, m))?,
            output::<Self, T>(&[k, n], batch, transpose_items(&q, n, k))?,
        ])
    }
}

/// The transpose of each column-major `rows x cols` item of `data`,
/// column-major `cols x rows` per item.
fn transpose_items<T: Copy>(data: &[T], rows: usize, cols: usize) -> Vec<T> {
    let mut transpose = Vec::with_capacity(data.len());
    if rows * cols == 0 {
        return transpose;
    }
    for item in data.chunks_exact(rows * cols) {
        for row in 0..rows {
            transpose.extend((0..cols).map(|col| item[row + rows * col]));
        }
    }
    transpose
}

/// Full QR: `[Q, R]` of shapes `[m, m, b..]`, `[m, n, b..]`. When `m <= n`
/// it is the compact QR ([`Qr`]), whose `Q` is already square. Otherwise
/// the provider's compact Householder factor of `X` has its reflectors
/// applied to `I_m` (MatrixAlgebraKit `qr_full`): `O(m n^2 + m^2 n)`, no
/// `[X I]` augmentation; `R` is the factor's upper trapezoid. Both in the
/// provider's gauge.
pub(crate) struct QrFull;

impl Factor for QrFull {
    const ENTRY: &'static str = Qr::ENTRY;
    const OP: tlinalg::Op = Qr::OP;

    fn estimates<T: tlinalg::FaerScalar>(m: usize, n: usize) -> [f64; 2] {
        if m <= n {
            return Qr::estimates::<T>(m, n);
        }
        householder_estimates::<T>(m, n, m)
    }

    fn factor<T: Scalar>(
        raw: Option<RawStridedRef<'_, T>>,
        [m, n]: [usize; 2],
        batch: &[usize],
        exec: &mut Exec<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        if m <= n {
            return Qr::factor::<T>(raw, [m, n], batch, exec);
        }
        let Ok(items) = checked_len(batch) else {
            // The output metadata refuses the batch as owned QR does; an
            // output it would admit still cannot hold the batch's factors.
            return Err(output::<Self, T>(&[m, m], batch, Vec::new())
                .err()
                .unwrap_or(DenseError::ElementCountOverflow));
        };
        let mut r = vec![T::ZERO; checked_len(&[m, n, items])?];
        let mut q = vec![T::ZERO; checked_len(&[m, m, items])?];
        match raw {
            Some(raw) => {
                compact_copy(raw, &mut r);
                owned_householder::<T>(&mut r, [m, n], items, &mut q, 0, exec)
                    .map_err(|err| tenferro_error(Self::ENTRY, provider_error(Self::OP, err)))?;
                zero_below_diagonal(&mut r, m, n);
            }
            None => identity_columns(&mut q, m, m, 0),
        }
        Ok(vec![
            output::<Self, T>(&[m, m], batch, q)?,
            output::<Self, T>(&[m, n], batch, r)?,
        ])
    }
}

/// The shape-based QR null basis: `[N]` of shape `[m, m - k, b..]` with
/// `k = min(m, n)`, the reflectors of `X`'s compact Householder factor
/// applied to `[0; I_{m-k}]` (MatrixAlgebraKit `qr_null`), written at its
/// final size; nothing is solved when `m <= n`.
pub(crate) struct QrNull;

impl Factor for QrNull {
    const ENTRY: &'static str = Qr::ENTRY;
    const OP: tlinalg::Op = Qr::OP;

    fn estimates<T: tlinalg::FaerScalar>(m: usize, n: usize) -> [f64; 2] {
        householder_estimates::<T>(m, n, m - m.min(n))
    }

    fn factor<T: Scalar>(
        raw: Option<RawStridedRef<'_, T>>,
        [m, n]: [usize; 2],
        batch: &[usize],
        exec: &mut Exec<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        let null = m - m.min(n);
        let Ok(items) = checked_len(batch) else {
            // The output metadata refuses the batch as owned QR does; an
            // output it would admit still cannot hold the batch's factors.
            return Err(output::<Self, T>(&[m, null], batch, Vec::new())
                .err()
                .unwrap_or(DenseError::ElementCountOverflow));
        };
        let mut basis = vec![T::ZERO; checked_len(&[m, null, items])?];
        match raw {
            Some(raw) if null > 0 => {
                let mut state = vec![T::ZERO; checked_len(&[m, n, items])?];
                compact_copy(raw, &mut state);
                owned_householder::<T>(&mut state, [m, n], items, &mut basis, n, exec)
                    .map_err(|err| tenferro_error(Self::ENTRY, provider_error(Self::OP, err)))?;
            }
            Some(_) => {}
            None => identity_columns(&mut basis, m, null, m - null),
        }
        Ok(vec![output::<Self, T>(&[m, null], batch, basis)?])
    }
}

/// tlinalg's estimates of a compact Householder factor of `m x n` plus its
/// reflectors applied to `cols` target columns.
fn householder_estimates<T: tlinalg::FaerScalar>(m: usize, n: usize, cols: usize) -> [f64; 2] {
    let shape = reflector_shape(m, n, cols);
    [
        householder::estimate_work::<T>(m, n)
            + householder::estimate_apply_reflectors_work::<T>(shape),
        householder::estimate_payback_work::<T>(m, n)
            + householder::estimate_apply_reflectors_payback_work::<T>(shape),
    ]
}

/// Every reflector of an `m x n` compact factor applied to `cols` columns.
fn reflector_shape(m: usize, n: usize, cols: usize) -> ReflectorShape {
    ReflectorShape {
        rows: m,
        a_cols: n,
        cols,
        k: m.min(n),
    }
}

/// The product of `extents`, or [`DenseError::ElementCountOverflow`].
fn checked_len(extents: &[usize]) -> Result<usize, DenseError> {
    extents
        .iter()
        .try_fold(1usize, |len, &extent| len.checked_mul(extent))
        .ok_or(DenseError::ElementCountOverflow)
}

/// Copies the nonempty strided `raw` into `out` as compact column-major
/// items, in batch order.
fn compact_copy<T: Copy>(raw: RawStridedRef<'_, T>, out: &mut [T]) {
    let (dims, strides, data) = (raw.dims(), raw.strides(), raw.data());
    let (m, n) = (dims[0], dims[1]);
    for (item, block) in out.chunks_exact_mut(m * n).enumerate() {
        let mut base = raw.offset();
        let mut rest = item;
        for (&extent, &stride) in dims[2..].iter().zip(&strides[2..]) {
            base += (rest % extent) as isize * stride;
            rest /= extent;
        }
        for col in 0..n {
            for row in 0..m {
                let at = base + row as isize * strides[0] + col as isize * strides[1];
                // `RawStridedRef::new` proved every reachable offset in range.
                block[row + m * col] = data[at as usize];
            }
        }
    }
}

/// Writes `[0; I_cols; 0]` into each column-major `m x cols` item of `c`,
/// the identity block starting at row `first`.
fn identity_columns<T: Scalar>(c: &mut [T], m: usize, cols: usize, first: usize) {
    c.fill(T::ZERO);
    if m * cols == 0 {
        return;
    }
    for item in c.chunks_exact_mut(m * cols) {
        for col in 0..cols {
            item[first + col + m * col] = T::ONE;
        }
    }
}

/// Zeroes the strictly lower triangle of each column-major `m x n` item,
/// leaving the upper trapezoid `R` of a compact Householder factor.
fn zero_below_diagonal<T: Scalar>(r: &mut [T], m: usize, n: usize) {
    if m * n == 0 {
        return;
    }
    for item in r.chunks_exact_mut(m * n) {
        for col in 0..n.min(m) {
            item[col + 1 + m * col..m * (col + 1)].fill(T::ZERO);
        }
    }
}

/// Factors each compact `m x n` item of `state` in place (`R` above the
/// diagonal, reflectors below, coefficients into `coeff`), then sets each
/// `m x cols` item of `target` to `[0; I_cols]` with the identity from row
/// `first` and applies `Q` to it: `Q`'s columns `first..first + cols`.
///
/// Why the `Vec` coefficient wrapper: `compact_factor_into` writes them as
/// `MaybeUninit` while `apply_reflectors_into` reads `&[T]`; joining the two
/// would need `unsafe`, so the lane lends a reused `Vec` instead and only the
/// factor's own workspace is allocated per call.
#[allow(clippy::too_many_arguments)]
fn householder_into<T: Scalar>(
    state: &mut [T],
    [m, n]: [usize; 2],
    batch: usize,
    coeff: &mut Vec<T>,
    target: &mut [T],
    first: usize,
    workspace: &mut [MaybeUninit<u8>],
    exec: &mut Exec<'_>,
) -> tlinalg::Result<()> {
    let cols = target.len().checked_div(m * batch).unwrap_or(0);
    householder::compact_factor(tlinalg::Op::HouseholderQr, m, n, batch, state, coeff, exec)?;
    identity_columns(target, m, cols, first);
    householder::apply_reflectors_into(
        tlinalg::Op::HouseholderQrQColumns,
        reflector_shape(m, n, cols),
        batch,
        state,
        coeff,
        target,
        false,
        workspace,
        exec,
    )
}

/// [`householder_into`] with a fresh coefficient `Vec` and reflector
/// workspace: the owned batch's per-block storage.
fn owned_householder<T: Scalar>(
    state: &mut [T],
    dims @ [m, n]: [usize; 2],
    batch: usize,
    target: &mut [T],
    first: usize,
    exec: &mut Exec<'_>,
) -> tlinalg::Result<()> {
    let cols = target.len().checked_div(m * batch).unwrap_or(0);
    let bytes = householder::apply_reflectors_workspace::<T>(
        reflector_shape(m, n, cols),
        batch,
        false,
        exec,
    )?;
    let mut workspace = vec![MaybeUninit::uninit(); bytes];
    householder_into(
        state,
        dims,
        batch,
        &mut Vec::new(),
        target,
        first,
        &mut workspace,
        exec,
    )
}

/// Polar decomposition (MatrixAlgebraKit `left_polar!` / `right_polar!` via
/// `PolarViaSVD`): left (`LEFT`) `[W, P]` of shapes `[m, n, b..]` and
/// `[n, n, b..]` for `m >= n`, right `[P, Wh]` of shapes `[m, m, b..]` and
/// `[m, n, b..]` for `n >= m`. Destination form only: one job runs the
/// compact SVD into lane scratch, `W = U Vh`, the `sqrt(S)` scaling of `Vh`
/// (left) or `U` (right), `P = X^H X` (or `X X^H`) and the Hermitian
/// projection, both products writing the destinations.
pub(crate) struct Polar<const LEFT: bool>;

impl<const LEFT: bool> Factor for Polar<LEFT> {
    const ENTRY: &'static str = Svd::ENTRY;
    const OP: tlinalg::Op = Svd::OP;

    // The SVD's estimates plus the two products' weighted work: the item's
    // whole weighted work, which the batch decision's query reads. The lane
    // model prices the SVD and each product as its own stage
    // (`FactorInto::stage_works`).
    fn estimates<T: tlinalg::FaerScalar>(m: usize, n: usize) -> [f64; 2] {
        composite_estimates(<Self as FactorInto>::stage_works::<T>(m, n))
    }

    fn factor<T: Scalar>(
        _: Option<RawStridedRef<'_, T>>,
        _: [usize; 2],
        _: &[usize],
        _: &mut Exec<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        Err(crate::executor::destination_only_error(if LEFT {
            crate::DenseFactorization::LeftPolar
        } else {
            crate::DenseFactorization::RightPolar
        }))
    }
}

/// Hermitian exponential (MatrixAlgebraKit `exponential!` via
/// `MatrixFunctionViaEigh`): `[exp]` of shape `[n, n, b..]`. Destination
/// form only: one job runs the Hermitian eigendecomposition into lane
/// scratch, checks its eigenvalues are finite, scales `V`'s columns by
/// `exp(lambda / 2)`, writes `exp = Y Y^H` of that scaled `Y` and makes it
/// exactly Hermitian (`copytri`).
pub(crate) struct ExpViaEigh;

impl Factor for ExpViaEigh {
    const ENTRY: &'static str = Eigh::ENTRY;
    const OP: tlinalg::Op = Eigh::OP;
    const SQUARE: bool = true;

    // The eigendecomposition's estimates plus the product's weighted work
    // (see `Polar::estimates`).
    fn estimates<T: tlinalg::FaerScalar>(n: usize, m: usize) -> [f64; 2] {
        composite_estimates(<Self as FactorInto>::stage_works::<T>(n, m))
    }

    fn factor<T: Scalar>(
        _: Option<RawStridedRef<'_, T>>,
        _: [usize; 2],
        _: &[usize],
        _: &mut Exec<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        Err(crate::executor::destination_only_error(
            crate::DenseFactorization::ExpViaEigh,
        ))
    }
}

/// Inverse (TensorKit `inv` per block): `[inv]` of shape `[n, n, b..]`.
/// Destination form only: one job writes `I_n` into each destination item
/// and solves `X Y = I` there in place, tlinalg's partial-pivot LU
/// `solve_into` with no separate right-hand side. tenferro's
/// `solve_read_into` (the per-matrix `solve_into` entry) is the same call on
/// a copied right-hand side, with the same validation, singularity check and
/// errors: those of its direct route for a nonempty rank-two input
/// (`solve_read_into`), and for any other rank or a zero extent those of the
/// `solve_read` it falls back to (`solve`).
pub(crate) struct Inv;

impl Factor for Inv {
    const ENTRY: &'static str = "solve_into";
    const OP: tlinalg::Op = tlinalg::Op::Solve;
    const VALIDATION: &'static str = "solve_read_into";
    const SQUARE: bool = true;

    fn estimates<T: tlinalg::FaerScalar>(n: usize, _: usize) -> [f64; 2] {
        use tlinalg::lu::{estimate_solve_payback_work, estimate_solve_work};
        [
            estimate_solve_work::<T>(n, n),
            estimate_solve_payback_work::<T>(n, n),
        ]
    }

    // tenferro's `solve_read_into` solves a nonempty rank-two input directly
    // and hands any other rank or a zero extent to `solve_read`, whose errors
    // name `solve`.
    fn validation(shape: &[usize]) -> &'static str {
        if shape.len() == 2 && !shape.contains(&0) {
            Self::VALIDATION
        } else {
            Self::OP.as_str()
        }
    }

    // tenferro's `solve_in_place_view` names a singular pivot after the route
    // the caller used and maps every other failure as the provider op; the
    // `solve_into` entry then types a numerical failure.
    fn provider_failure(error: tlinalg::Error, shape: &[usize]) -> DenseError {
        let error = match error {
            tlinalg::Error::Singular { .. } => {
                let op = Self::validation(shape);
                linalg_error(op, tenferro_linalg::Error::Singular { op })
            }
            other => provider_error(Self::OP, other),
        };
        tenferro_solve_error(error)
    }

    fn factor<T: Scalar>(
        _: Option<RawStridedRef<'_, T>>,
        _: [usize; 2],
        _: &[usize],
        _: &mut Exec<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        Err(crate::executor::destination_only_error(
            crate::DenseFactorization::Inv,
        ))
    }
}

/// Hermitian eigendecomposition: `[values, vectors]` of shapes `[n, b..]`
/// (real) and `[n, n, b..]`.
pub(crate) struct Eigh;

impl Factor for Eigh {
    const ENTRY: &'static str = "eigh_read";
    const OP: tlinalg::Op = tlinalg::Op::Eigh;
    const SQUARE: bool = true;

    fn estimates<T: tlinalg::FaerScalar>(n: usize, _: usize) -> [f64; 2] {
        use tlinalg::eigh::{estimate_payback_work, estimate_work};
        [estimate_work::<T>(n), estimate_payback_work::<T>(n)]
    }

    fn factor<T: Scalar>(
        raw: Option<RawStridedRef<'_, T>>,
        [n, _]: [usize; 2],
        batch: &[usize],
        exec: &mut Exec<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        let (mut values, mut vectors) = (Vec::new(), Vec::new());
        if let Some(raw) = raw {
            tlinalg::eigh::eigh(Self::OP, raw, &mut values, &mut vectors, exec)
                .map_err(|err| tenferro_error(Self::ENTRY, provider_error(Self::OP, err)))?;
        }
        Ok(vec![
            output::<Self, <T as TensorScalar>::Real>(&[n], batch, values)?,
            output::<Self, T>(&[n, n], batch, vectors)?,
        ])
    }
}

/// Hermitian eigenvalues only: `[values]` of shape `[n, b..]`, real.
pub(crate) struct EighVals;

impl Factor for EighVals {
    const ENTRY: &'static str = "eigh_values";
    const OP: tlinalg::Op = tlinalg::Op::EighValues;
    const SQUARE: bool = true;

    fn estimates<T: tlinalg::FaerScalar>(n: usize, _: usize) -> [f64; 2] {
        use tlinalg::eigh::{estimate_payback_work, estimate_work};
        [estimate_work::<T>(n), estimate_payback_work::<T>(n)]
    }

    fn factor<T: Scalar>(
        raw: Option<RawStridedRef<'_, T>>,
        [n, _]: [usize; 2],
        batch: &[usize],
        exec: &mut Exec<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        let mut values = Vec::new();
        if let Some(raw) = raw {
            tlinalg::eigh::eigh_values(Self::OP, raw, &mut values, exec)
                .map_err(|err| tenferro_error(Self::ENTRY, provider_error(Self::OP, err)))?;
        }
        Ok(vec![output::<Self, <T as TensorScalar>::Real>(
            &[n],
            batch,
            values,
        )?])
    }
}

/// Singular values only: `[S]` of shape `[k, b..]` with `k = min(m, n)`, real.
pub(crate) struct SvdVals;

impl Factor for SvdVals {
    const ENTRY: &'static str = "svd_values";
    const OP: tlinalg::Op = tlinalg::Op::SvdValues;

    fn estimates<T: tlinalg::FaerScalar>(m: usize, n: usize) -> [f64; 2] {
        use tlinalg::svd::{estimate_payback_work, estimate_work};
        [estimate_work::<T>(m, n), estimate_payback_work::<T>(m, n)]
    }

    fn factor<T: Scalar>(
        raw: Option<RawStridedRef<'_, T>>,
        [m, n]: [usize; 2],
        batch: &[usize],
        exec: &mut Exec<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        let mut s = Vec::new();
        if let Some(raw) = raw {
            tlinalg::svd::svd_values(Self::OP, raw, &mut s, exec)
                .map_err(|err| tenferro_error(Self::ENTRY, provider_error(Self::OP, err)))?;
        }
        Ok(vec![output::<Self, <T as TensorScalar>::Real>(
            &[m.min(n)],
            batch,
            s,
        )?])
    }
}

/// General eigenvalues only: `[values]` of shape `[n, b..]` in the input's
/// complex counterpart.
pub(crate) struct EigVals;

impl Factor for EigVals {
    const ENTRY: &'static str = "eig_values";
    const OP: tlinalg::Op = tlinalg::Op::EigValues;
    const SQUARE: bool = true;

    fn estimates<T: tlinalg::FaerScalar>(n: usize, _: usize) -> [f64; 2] {
        use tlinalg::eig::{estimate_payback_work, estimate_work};
        [estimate_work::<T>(n), estimate_payback_work::<T>(n)]
    }

    fn factor<T: Scalar>(
        raw: Option<RawStridedRef<'_, T>>,
        [n, _]: [usize; 2],
        batch: &[usize],
        exec: &mut Exec<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        let mut values = Vec::new();
        if let Some(raw) = raw {
            T::eig_values(raw, &mut values, exec)
                .map_err(|err| tenferro_error(Self::ENTRY, provider_error(Self::OP, err)))?;
        }
        Ok(vec![output::<Self, T::EigScalar>(&[n], batch, values)?])
    }
}

/// General eigendecomposition: `[values, vectors]` of shapes `[n, b..]` and
/// `[n, n, b..]` in the input's complex counterpart, the eigenvector columns
/// in tenferro's unit 2-norm gauge.
pub(crate) struct Eig;

impl Factor for Eig {
    const ENTRY: &'static str = "eig_read";
    const OP: tlinalg::Op = tlinalg::Op::Eig;
    const SQUARE: bool = true;

    fn estimates<T: tlinalg::FaerScalar>(n: usize, m: usize) -> [f64; 2] {
        EigVals::estimates::<T>(n, m)
    }

    fn factor<T: Scalar>(
        raw: Option<RawStridedRef<'_, T>>,
        [n, _]: [usize; 2],
        batch: &[usize],
        exec: &mut Exec<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        let (mut values, mut vectors) = (Vec::new(), Vec::new());
        if let Some(raw) = raw {
            T::eig(raw, &mut values, &mut vectors, exec)
                .map_err(|err| tenferro_error(Self::ENTRY, provider_error(Self::OP, err)))?;
            T::normalize_eig(&mut vectors, n, checked_len(batch)?)?;
        }
        Ok(vec![
            output::<Self, T::EigScalar>(&[n], batch, values)?,
            output::<Self, T::EigScalar>(&[n, n], batch, vectors)?,
        ])
    }
}

/// A column-major factor of shape `dims` followed by `batch`.
fn output<F: Factor, T: TensorScalar>(
    dims: &[usize],
    batch: &[usize],
    data: Vec<T>,
) -> Result<DenseTensor, DenseError> {
    Tensor::from_vec_col_major([dims, batch].concat(), data)
        .map_err(|err| tenferro_error(F::ENTRY, err))
        .and_then(DenseTensor::from_tenferro)
}

/// A scalar the provider factors; tlinalg returns its singular values and
/// Hermitian eigenvalues in [`TensorScalar::Real`], the type tenferro
/// publishes them in.
pub(crate) trait Scalar:
    tlinalg::FaerScalar<Real = <Self as Element>::Spectrum>
    + TensorScalar<Real = <Self as Element>::Spectrum>
    + Element
{
    /// The complex counterpart, in which the general eigenvalues are
    /// published.
    type EigScalar: TensorScalar;

    /// Zero and one, for identity targets and zeroed scratch.
    const ZERO: Self;
    const ONE: Self;

    /// The lane's typed scratch of this scalar.
    fn scratch(lane: &mut TypedScratch) -> &mut ScalarScratch<Self>;

    /// The geometry plan of `geometry` at this scalar.
    fn plan(geometry: Geometry) -> Result<Arc<ErasedPlan>, DenseError>;

    /// `gemm::grant` of a product of weighted work `work` and serial time
    /// `serial` with `plan` (from [`Self::plan`]) on a lane of `budget`
    /// slots of `exec`.
    fn product_grant(
        plan: &ErasedPlan,
        work: f64,
        exec: &Exec<'_>,
        budget: usize,
        serial: Duration,
    ) -> usize;

    /// `d = a b` with `plan` (from [`Self::plan`]) on `exec`.
    fn gemm(
        plan: &ErasedPlan,
        exec: &mut Exec<'_>,
        a: &[Self],
        b: &[Self],
        d: &mut [Self],
    ) -> Result<(), DenseError>;

    /// The compact payload of a spectrum destination (the real
    /// counterpart), `None` for another dtype.
    fn spectrum_slot<'b>(
        write: &'b mut DenseWrite<'_>,
    ) -> Option<&'b mut [<Self as TensorScalar>::Real]>;

    /// tlinalg's general eigenvalues only, in its order.
    fn eig_values(
        raw: RawStridedRef<'_, Self>,
        values: &mut Vec<Self::EigScalar>,
        exec: &mut Exec<'_>,
    ) -> tlinalg::Result<()>;

    /// tlinalg's general eigenpairs, in its order and gauge.
    fn eig(
        raw: RawStridedRef<'_, Self>,
        values: &mut Vec<Self::EigScalar>,
        vectors: &mut Vec<Self::EigScalar>,
        exec: &mut Exec<'_>,
    ) -> tlinalg::Result<()>;

    /// tenferro's unit 2-norm gauge on `items` column-major `n x n`
    /// eigenvector blocks (`n > 0`), failing as tenferro's `eig_read` does.
    fn normalize_eig(
        vectors: &mut [Self::EigScalar],
        n: usize,
        items: usize,
    ) -> Result<(), DenseError>;
}

// Why one impl per concrete scalar: tlinalg's general-eigenvalue type is an
// associated type of a trait it does not export, so it equals `EigScalar`
// only once the scalar is concrete.
macro_rules! scalar {
    ($ty:ty, $complex:ty, $real_variant:ident, $slot:ident, $zero:expr, $one:expr) => {
        impl Scalar for $ty {
            const ZERO: Self = $zero;
            const ONE: Self = $one;
            type EigScalar = $complex;

            fn scratch(lane: &mut TypedScratch) -> &mut ScalarScratch<Self> {
                &mut lane.$slot
            }

            fn plan(geometry: Geometry) -> Result<Arc<ErasedPlan>, DenseError> {
                gemm::erased_plan::<$ty>(geometry)
            }

            fn product_grant(
                plan: &ErasedPlan,
                work: f64,
                exec: &Exec<'_>,
                budget: usize,
                serial: Duration,
            ) -> usize {
                gemm::erased_grant::<$ty>(plan, work, exec, budget, serial)
            }

            fn gemm(
                plan: &ErasedPlan,
                exec: &mut Exec<'_>,
                a: &[Self],
                b: &[Self],
                d: &mut [Self],
            ) -> Result<(), DenseError> {
                gemm::execute_erased::<$ty>(plan, exec, $one, a, b, d)
            }

            fn spectrum_slot<'b>(
                write: &'b mut DenseWrite<'_>,
            ) -> Option<&'b mut [<Self as TensorScalar>::Real]> {
                match write {
                    DenseWrite::$real_variant(view) => Some(compact_mut(view)),
                    _ => None,
                }
            }

            fn eig_values(
                raw: RawStridedRef<'_, $ty>,
                values: &mut Vec<$complex>,
                exec: &mut Exec<'_>,
            ) -> tlinalg::Result<()> {
                tlinalg::eig::eig_values(tlinalg::Op::EigValues, raw, values, exec)
            }

            fn eig(
                raw: RawStridedRef<'_, $ty>,
                values: &mut Vec<$complex>,
                vectors: &mut Vec<$complex>,
                exec: &mut Exec<'_>,
            ) -> tlinalg::Result<()> {
                tlinalg::eig::eig(tlinalg::Op::Eig, raw, values, vectors, exec)
            }

            fn normalize_eig(
                vectors: &mut [$complex],
                n: usize,
                items: usize,
            ) -> Result<(), DenseError> {
                tenferro_linalg::normalize_eig_vectors(vectors, n, items)
                    .map_err(|err| tenferro_error(Eig::ENTRY, err))
            }
        }
    };
}

scalar!(f32, Complex32, F32, f32, 0.0, 1.0);
scalar!(f64, Complex64, F64, f64, 0.0, 1.0);
scalar!(
    Complex32,
    Complex32,
    F32,
    c32,
    Complex32::new(0.0, 0.0),
    Complex32::new(1.0, 0.0)
);
scalar!(
    Complex64,
    Complex64,
    F64,
    c64,
    Complex64::new(0.0, 0.0),
    Complex64::new(1.0, 0.0)
);

/// One block: its validated input, its signed strides and, once run, its
/// own outcome (factors or that block's error).
pub(crate) struct Job<'a> {
    input: DenseRead<'a>,
    strides: StrideVec,
    outcome: Option<Result<Vec<DenseTensor>, DenseError>>,
}

/// Factorization `F` of each block on the lane driver.
struct Blocks<'a, F>(PhantomData<fn() -> (&'a (), F)>);

impl<'a, F: Factor> BlockOp for Blocks<'a, F> {
    type Job = Job<'a>;
    const ENTRY: Entry = Entry::Linalg;
    const KIND: OpKind = OpKind::Factorization;
    const LABEL: &'static str = F::ENTRY;
    type Lane = ();

    fn lane(&self, _: &mut LaneMembers<'_, Job<'a>>, _: &Exec<'_>) {}

    fn work(&self, _: usize, job: &Job<'a>) -> f64 {
        work::<F>(job.input)
    }

    fn modeled(&self, _: usize, job: &Job<'a>) -> bool {
        single_item::<F>(job.input).is_some()
    }

    fn grant(&self, _: usize, job: &Job<'a>, exec: &Exec<'_>, budget: usize, _: Duration) -> Grant {
        lanes::single(single_item::<F>(job.input).map_or(1, |item| grant(exec, budget, item)))
    }

    fn grant_key(&self, _: usize, job: &Job<'a>) -> GrantKey {
        item_key(single_item::<F>(job.input))
    }

    // A block's error is its own outcome, not the driver's: the caller
    // delivers every block's outcome in input order.
    fn run(
        &self,
        _: usize,
        job: &mut Job<'a>,
        (): &mut (),
        mut exec: Exec<'_>,
    ) -> Result<(), DenseError> {
        job.outcome = Some(factor::<F>(job.input, &job.strides, &mut exec));
        Ok(())
    }
}

/// The batch's outcome sink (see [`crate::DenseExecutor::factorize_batch`]).
pub(crate) type Sink<'s> =
    dyn FnMut(usize, Result<Vec<DenseTensor>, DenseError>) -> Result<(), DenseError> + 's;

/// Factorization `F` of every input, delivering each input's outcome to
/// `sink` in input order exactly as the per-matrix entry
/// ([`crate::DenseExecutor::svd`], `qr`, `eigh`, `eigh_vals`, `svd_vals`,
/// `eig`, `eig_vals`) returns it; LQ, which has no per-matrix entry,
/// reports the QR entry's errors.
///
/// Checks run in the per-matrix entry's order: the engine kind, then per
/// input the view and the dtype (before the engine is entered), then rank,
/// layout and the factorization (inside it). The first input that fails a
/// pre-entry check is delivered after every input before it; inputs above it
/// are not run. A refused engine entry is returned with no outcome
/// delivered.
pub(crate) fn factorize_batch<F: Factor>(
    engine: &mut dyn tenferro_tensor::Engine,
    inputs: &[DenseRead<'_>],
    sink: &mut Sink<'_>,
) -> Result<(), DenseError> {
    with_cpu_engine(engine, |cpu| batch_on::<F>(cpu, inputs, sink)).unwrap_or_else(|| {
        Err(tenferro_error(
            F::ENTRY,
            tenferro_tensor::Error::unsupported(F::ENTRY, NOT_A_CPU_ENGINE),
        ))
    })
}

/// [`factorize_batch`] of the factorization `op` names.
pub(crate) fn factorize(
    engine: &mut dyn tenferro_tensor::Engine,
    op: crate::DenseFactorization,
    inputs: &[DenseRead<'_>],
    sink: &mut Sink<'_>,
) -> Result<(), DenseError> {
    use crate::DenseFactorization as Op;
    match op {
        Op::Svd => factorize_batch::<Svd>(engine, inputs, sink),
        Op::SvdFull => factorize_batch::<SvdFull>(engine, inputs, sink),
        Op::Qr => factorize_batch::<Qr>(engine, inputs, sink),
        Op::Eigh => factorize_batch::<Eigh>(engine, inputs, sink),
        Op::EighVals => factorize_batch::<EighVals>(engine, inputs, sink),
        Op::SvdVals => factorize_batch::<SvdVals>(engine, inputs, sink),
        Op::Eig => factorize_batch::<Eig>(engine, inputs, sink),
        Op::EigVals => factorize_batch::<EigVals>(engine, inputs, sink),
        Op::Lq => factorize_batch::<Lq>(engine, inputs, sink),
        Op::QrFull => factorize_batch::<QrFull>(engine, inputs, sink),
        Op::QrNull => factorize_batch::<QrNull>(engine, inputs, sink),
        Op::LeftPolar | Op::RightPolar | Op::ExpViaEigh | Op::Inv => {
            Err(crate::executor::destination_only_error(op))
        }
    }
}

fn batch_on<F: Factor>(
    cpu: &mut CpuEngine,
    inputs: &[DenseRead<'_>],
    sink: &mut Sink<'_>,
) -> Result<(), DenseError> {
    let mut jobs = Vec::with_capacity(inputs.len());
    let mut refused = None;
    for &input in inputs {
        match preflight::<F>(input) {
            Ok(strides) => jobs.push(Job {
                input,
                strides,
                outcome: None,
            }),
            Err(error) => {
                refused = Some(error);
                break;
            }
        }
    }
    if !jobs.is_empty() {
        driver::drive(
            cpu,
            &Blocks::<F>(PhantomData),
            &mut jobs,
            &mut LaneTables::default(),
        )?;
    }
    let admitted = jobs.len();
    for (index, job) in jobs.into_iter().enumerate() {
        sink(
            index,
            job.outcome
                .unwrap_or_else(|| unreachable!("the driver runs every admitted job")),
        )?;
    }
    match refused {
        Some(error) => sink(admitted, Err(error)),
        None => Ok(()),
    }
}

/// A [`Factor`] with a destination form: its provider workspace query and
/// its initialized-destination entry.
pub(crate) trait FactorInto: Factor {
    /// Nonzero for a composite job, the provider's factorization followed by
    /// the products of [`Self::geometries`], each a stage the lane model
    /// prices on its own; distinct per composite op, it tags the job's grant
    /// key. Zero for a single-stage factorization.
    const COMPOSITE: u64 = 0;

    /// A composite job's stages for one `m x n` item of scalar `T`: its
    /// factorization's tlinalg (total, payback) estimates and the weighted
    /// work of each product of [`Self::geometries`] (zero for an absent
    /// one); `None` for a single-stage factorization.
    fn stage_works<T: tlinalg::FaerScalar>(_m: usize, _n: usize) -> Option<([f64; 2], [f64; 2])> {
        None
    }

    /// tlinalg's workspace bytes for `batch` items of `dims` on `exec`.
    fn workspace<T: Scalar>(
        dims: [usize; 2],
        batch: usize,
        exec: &Exec<'_>,
    ) -> tlinalg::Result<usize>;

    /// The typed scratch elements one input needs: the factors it stages
    /// before writing its destinations, such as the provider-shaped factors
    /// under [`FactorPlacement::Adjoint`].
    fn scratch_len<T: Scalar>(dims: [usize; 2], batch: usize, placement: FactorPlacement) -> usize;

    /// The real scratch elements one input needs: a spectrum it consumes
    /// instead of writing it to a destination.
    fn spectrum_len(_dims: [usize; 2], _batch: usize) -> usize {
        0
    }

    /// The Householder coefficients one input needs (the capacity of the
    /// lane's reused coefficient `Vec`).
    fn coeff_len(_dims: [usize; 2], _batch: usize) -> usize {
        0
    }

    /// Writes the outputs of an input with a zero extent, whose solve is
    /// skipped; most factorizations have nothing to write.
    fn empty_into<T: Scalar>(_dims: [usize; 2], _outputs: &mut [Option<DenseWrite<'_>>; 3]) {}

    /// The refusal of an input of matrix extents `dims` beyond the provider's
    /// own (a polar direction), checked after the rank and squareness and
    /// before the zero-extent shortcut.
    fn admit(_dims: [usize; 2]) -> Result<(), DenseError> {
        Ok(())
    }

    /// The geometries of the products one nonempty input of `dims`
    /// composes, whose plans are resolved before the lanes; a provider-only
    /// factorization has none.
    fn geometries(_dims: [usize; 2]) -> Result<[Option<Geometry>; 2], DenseError> {
        Ok([None, None])
    }

    /// Solves one input into `outputs` (validated destinations of `T`),
    /// with the plans of [`Self::geometries`] at `T`.
    #[allow(clippy::too_many_arguments)]
    fn factor_into<T: Scalar>(
        raw: RawStridedRef<'_, T>,
        dims: [usize; 2],
        batch: usize,
        outputs: &mut [Option<DenseWrite<'_>>; 3],
        placement: FactorPlacement,
        workspace: &mut [MaybeUninit<u8>],
        scratch: &mut [T],
        spectrum: &mut [<T as Element>::Spectrum],
        coeff: &mut Vec<T>,
        plans: &Plans,
        exec: &mut Exec<'_>,
    ) -> Result<(), JobError>;
}

/// The resolved product plans of one job ([`FactorInto::geometries`]).
pub(crate) type Plans = [Option<Arc<ErasedPlan>>; 2];

/// Why one destination job failed: the delivered outcome tags a provider or
/// product failure as [`FactorFailure::Solver`] and an output check as
/// [`FactorFailure::Output`].
#[derive(Debug)]
pub(crate) enum JobError {
    /// The provider's factorization.
    Provider(tlinalg::Error),
    /// A product of the job's composition.
    Gemm(DenseError),
    /// A check of the job's outputs.
    Output(DenseError),
}

impl From<tlinalg::Error> for JobError {
    fn from(error: tlinalg::Error) -> Self {
        Self::Provider(error)
    }
}

fn slot<'b, T: Scalar>(output: &'b mut Option<DenseWrite<'_>>) -> &'b mut [T] {
    output
        .as_mut()
        .and_then(<T as Element>::compact)
        .expect("validation checked the destination dtype")
}

fn spectrum<'b, T: Scalar>(
    output: &'b mut Option<DenseWrite<'_>>,
) -> &'b mut [<T as TensorScalar>::Real] {
    output
        .as_mut()
        .and_then(T::spectrum_slot)
        .expect("validation checked the destination dtype")
}

impl FactorInto for Qr {
    fn workspace<T: Scalar>(
        [m, n]: [usize; 2],
        batch: usize,
        exec: &Exec<'_>,
    ) -> tlinalg::Result<usize> {
        tlinalg::qr::qr_workspace::<T>(m, n, batch, exec)
    }

    fn scratch_len<T: Scalar>(
        [m, n]: [usize; 2],
        batch: usize,
        placement: FactorPlacement,
    ) -> usize {
        match placement {
            FactorPlacement::AsComputed => 0,
            FactorPlacement::Adjoint => (m + n) * m.min(n) * batch,
        }
    }

    fn factor_into<T: Scalar>(
        raw: RawStridedRef<'_, T>,
        [m, n]: [usize; 2],
        batch: usize,
        outputs: &mut [Option<DenseWrite<'_>>; 3],
        placement: FactorPlacement,
        workspace: &mut [MaybeUninit<u8>],
        scratch: &mut [T],
        _: &mut [<T as Element>::Spectrum],
        _: &mut Vec<T>,
        _: &Plans,
        exec: &mut Exec<'_>,
    ) -> Result<(), JobError> {
        let [first, second, _] = outputs;
        match placement {
            FactorPlacement::AsComputed => Ok(tlinalg::qr::qr_into_init(
                Self::OP,
                raw,
                slot::<T>(first),
                slot::<T>(second),
                workspace,
                exec,
            )?),
            FactorPlacement::Adjoint => {
                let k = m.min(n);
                let (q, rest) = scratch.split_at_mut(m * k * batch);
                let r = &mut rest[..k * n * batch];
                tlinalg::qr::qr_into_init(Self::OP, raw, q, r, workspace, exec)?;
                adjoint_items(r, k, n, slot::<T>(first));
                adjoint_items(q, m, k, slot::<T>(second));
                Ok(())
            }
        }
    }
}

impl FactorInto for Lq {
    fn workspace<T: Scalar>(
        [m, n]: [usize; 2],
        batch: usize,
        exec: &Exec<'_>,
    ) -> tlinalg::Result<usize> {
        tlinalg::qr::qr_workspace::<T>(n, m, batch, exec)
    }

    fn scratch_len<T: Scalar>([m, n]: [usize; 2], batch: usize, _: FactorPlacement) -> usize {
        (n + m) * m.min(n) * batch
    }

    // `raw` is the transposed read `X^T` (`n x m`): its QR `Q' R'` lands in
    // the lane scratch, and one plain transpose pass writes `L = R'^T` and
    // `Q = Q'^T`. Why not conjugate: this is the QR of `X^T`, not of `X^H`,
    // so an adjoint would give the factors of `conj(X)`.
    fn factor_into<T: Scalar>(
        raw: RawStridedRef<'_, T>,
        [m, n]: [usize; 2],
        batch: usize,
        outputs: &mut [Option<DenseWrite<'_>>; 3],
        _: FactorPlacement,
        workspace: &mut [MaybeUninit<u8>],
        scratch: &mut [T],
        _: &mut [<T as Element>::Spectrum],
        _: &mut Vec<T>,
        _: &Plans,
        exec: &mut Exec<'_>,
    ) -> Result<(), JobError> {
        let [first, second, _] = outputs;
        let k = m.min(n);
        let (q, rest) = scratch.split_at_mut(n * k * batch);
        let r = &mut rest[..k * m * batch];
        tlinalg::qr::qr_into_init(Self::OP, raw, q, r, workspace, exec)?;
        map_transposed_items(r, k, m, slot::<T>(first), |value| value);
        map_transposed_items(q, n, k, slot::<T>(second), |value| value);
        Ok(())
    }
}

impl FactorInto for QrFull {
    fn workspace<T: Scalar>(
        dims @ [m, n]: [usize; 2],
        batch: usize,
        exec: &Exec<'_>,
    ) -> tlinalg::Result<usize> {
        if m <= n {
            return Qr::workspace::<T>(dims, batch, exec);
        }
        householder::apply_reflectors_workspace::<T>(reflector_shape(m, n, m), batch, false, exec)
    }

    fn scratch_len<T: Scalar>(_: [usize; 2], _: usize, _: FactorPlacement) -> usize {
        0
    }

    fn coeff_len([m, n]: [usize; 2], batch: usize) -> usize {
        if m <= n {
            0
        } else {
            n * batch
        }
    }

    // `[m, 0]`: `Q = I_m`, `R` empty.
    fn empty_into<T: Scalar>([m, _]: [usize; 2], outputs: &mut [Option<DenseWrite<'_>>; 3]) {
        identity_columns(slot::<T>(&mut outputs[0]), m, m, 0);
    }

    // `m <= n`: the compact QR's destinations are the full ones. Otherwise
    // the input is copied once into `r`, factored there in place, and its
    // reflectors applied to `q = I_m`; `R` is the factor's upper trapezoid.
    fn factor_into<T: Scalar>(
        raw: RawStridedRef<'_, T>,
        [m, n]: [usize; 2],
        batch: usize,
        outputs: &mut [Option<DenseWrite<'_>>; 3],
        placement: FactorPlacement,
        workspace: &mut [MaybeUninit<u8>],
        scratch: &mut [T],
        _: &mut [<T as Element>::Spectrum],
        coeff: &mut Vec<T>,
        _: &Plans,
        exec: &mut Exec<'_>,
    ) -> Result<(), JobError> {
        if m <= n {
            return Qr::factor_into::<T>(
                raw,
                [m, n],
                batch,
                outputs,
                placement,
                workspace,
                scratch,
                &mut [],
                coeff,
                &[None, None],
                exec,
            );
        }
        let [first, second, _] = outputs;
        let (q, r) = (slot::<T>(first), slot::<T>(second));
        compact_copy(raw, r);
        householder_into(r, [m, n], batch, coeff, q, 0, workspace, exec)?;
        zero_below_diagonal(r, m, n);
        Ok(())
    }
}

impl FactorInto for QrNull {
    fn workspace<T: Scalar>(
        [m, n]: [usize; 2],
        batch: usize,
        exec: &Exec<'_>,
    ) -> tlinalg::Result<usize> {
        let null = m - m.min(n);
        householder::apply_reflectors_workspace::<T>(
            reflector_shape(m, n, null),
            batch,
            false,
            exec,
        )
    }

    fn scratch_len<T: Scalar>([m, n]: [usize; 2], batch: usize, _: FactorPlacement) -> usize {
        if m > n {
            m * n * batch
        } else {
            0
        }
    }

    fn coeff_len([m, n]: [usize; 2], batch: usize) -> usize {
        if m > n {
            n * batch
        } else {
            0
        }
    }

    // `[m, 0]`: every direction is null, `N = I_m`.
    fn empty_into<T: Scalar>([m, n]: [usize; 2], outputs: &mut [Option<DenseWrite<'_>>; 3]) {
        let null = m - m.min(n);
        identity_columns(slot::<T>(&mut outputs[0]), m, null, m - null);
    }

    // The input is copied once into the lane scratch and factored there;
    // the reflectors go straight onto `n = [0; I_{m-k}]`, so only the final
    // `m x (m - k)` basis is written.
    fn factor_into<T: Scalar>(
        raw: RawStridedRef<'_, T>,
        [m, n]: [usize; 2],
        batch: usize,
        outputs: &mut [Option<DenseWrite<'_>>; 3],
        _: FactorPlacement,
        workspace: &mut [MaybeUninit<u8>],
        scratch: &mut [T],
        _: &mut [<T as Element>::Spectrum],
        coeff: &mut Vec<T>,
        _: &Plans,
        exec: &mut Exec<'_>,
    ) -> Result<(), JobError> {
        if m <= n {
            return Ok(());
        }
        let state = &mut scratch[..m * n * batch];
        compact_copy(raw, state);
        let basis = slot::<T>(&mut outputs[0]);
        Ok(householder_into(
            state,
            [m, n],
            batch,
            coeff,
            basis,
            n,
            workspace,
            exec,
        )?)
    }
}

impl<const FULL: bool> FactorInto for SvdJob<FULL> {
    fn workspace<T: Scalar>(
        [m, n]: [usize; 2],
        batch: usize,
        exec: &Exec<'_>,
    ) -> tlinalg::Result<usize> {
        tlinalg::svd::svd_workspace::<T>(m, n, FULL, batch, exec)
    }

    fn scratch_len<T: Scalar>(
        [m, n]: [usize; 2],
        batch: usize,
        placement: FactorPlacement,
    ) -> usize {
        let [u_cols, v_rows] = Self::vector_dims(m, n);
        match placement {
            FactorPlacement::AsComputed => 0,
            FactorPlacement::Adjoint => (m * u_cols + v_rows * n) * batch,
        }
    }

    // Full: identity `U` and `Vh` (either placement, each square).
    fn empty_into<T: Scalar>(_: [usize; 2], outputs: &mut [Option<DenseWrite<'_>>; 3]) {
        if !FULL {
            return;
        }
        for index in [0, 2] {
            let dim = outputs[index]
                .as_ref()
                .map_or(0, |output| output.shape()[0]);
            identity_columns(slot::<T>(&mut outputs[index]), dim, dim, 0);
        }
    }

    fn factor_into<T: Scalar>(
        raw: RawStridedRef<'_, T>,
        [m, n]: [usize; 2],
        batch: usize,
        outputs: &mut [Option<DenseWrite<'_>>; 3],
        placement: FactorPlacement,
        workspace: &mut [MaybeUninit<u8>],
        scratch: &mut [T],
        _: &mut [<T as Element>::Spectrum],
        _: &mut Vec<T>,
        _: &Plans,
        exec: &mut Exec<'_>,
    ) -> Result<(), JobError> {
        let [first, values, third] = outputs;
        let s = spectrum::<T>(values);
        match placement {
            FactorPlacement::AsComputed => {
                let (u, vt) = (slot::<T>(first), slot::<T>(third));
                Ok(tlinalg::svd::svd_into_init(
                    Self::OP,
                    raw,
                    FULL,
                    u,
                    s,
                    vt,
                    workspace,
                    exec,
                )?)
            }
            FactorPlacement::Adjoint => {
                let [u_cols, v_rows] = Self::vector_dims(m, n);
                let (u, rest) = scratch.split_at_mut(m * u_cols * batch);
                let vt = &mut rest[..v_rows * n * batch];
                tlinalg::svd::svd_into_init(Self::OP, raw, FULL, u, s, vt, workspace, exec)?;
                adjoint_items(vt, v_rows, n, slot::<T>(first));
                adjoint_items(u, m, u_cols, slot::<T>(third));
                Ok(())
            }
        }
    }
}

impl FactorInto for Eigh {
    fn workspace<T: Scalar>(
        [n, _]: [usize; 2],
        batch: usize,
        exec: &Exec<'_>,
    ) -> tlinalg::Result<usize> {
        tlinalg::eigh::eigh_workspace::<T>(n, true, batch, exec)
    }

    fn scratch_len<T: Scalar>(_: [usize; 2], _: usize, _: FactorPlacement) -> usize {
        0
    }

    fn factor_into<T: Scalar>(
        raw: RawStridedRef<'_, T>,
        _: [usize; 2],
        _: usize,
        outputs: &mut [Option<DenseWrite<'_>>; 3],
        _: FactorPlacement,
        workspace: &mut [MaybeUninit<u8>],
        _: &mut [T],
        _: &mut [<T as Element>::Spectrum],
        _: &mut Vec<T>,
        _: &Plans,
        exec: &mut Exec<'_>,
    ) -> Result<(), JobError> {
        let [values, second, _] = outputs;
        let (w, vectors) = (spectrum::<T>(values), slot::<T>(second));
        Ok(tlinalg::eigh::eigh_into_init(
            Self::OP,
            raw,
            w,
            vectors,
            workspace,
            exec,
        )?)
    }
}

impl<const LEFT: bool> FactorInto for Polar<LEFT> {
    const COMPOSITE: u64 = if LEFT { 2 } else { 3 };

    // The compact SVD, `W = U Vh` (`m k n`) and `P` (`n k n` left, `m k m`
    // right), as `geometries` lays them out.
    fn stage_works<T: tlinalg::FaerScalar>(m: usize, n: usize) -> Option<([f64; 2], [f64; 2])> {
        let svd = Svd::estimates::<T>(m, n);
        let (m, n) = (m as f64, n as f64);
        let (k, order) = (m.min(n), if LEFT { n } else { m });
        Some((
            svd,
            [
                T::COMPLEX_WEIGHT * (m * k * n),
                T::COMPLEX_WEIGHT * (order * order * k),
            ],
        ))
    }

    fn workspace<T: Scalar>(
        [m, n]: [usize; 2],
        batch: usize,
        exec: &Exec<'_>,
    ) -> tlinalg::Result<usize> {
        tlinalg::svd::svd_workspace::<T>(m, n, false, batch, exec)
    }

    // `U` and `Vh` of the SVD.
    fn scratch_len<T: Scalar>([m, n]: [usize; 2], batch: usize, _: FactorPlacement) -> usize {
        (m + n) * m.min(n) * batch
    }

    // `S` of the SVD.
    fn spectrum_len([m, n]: [usize; 2], batch: usize) -> usize {
        m.min(n) * batch
    }

    fn admit(dims: [usize; 2]) -> Result<(), DenseError> {
        polar_direction(LEFT, dims)
    }

    // `W = U Vh`, then `P = X^H X` with `X = sqrt(S) Vh` (left, `n x n`) or
    // `P = X X^H` with `X = U sqrt(S)` (right, `m x m`); the adjoint operand
    // is the factor itself read transposed and conjugated.
    fn geometries([m, n]: [usize; 2]) -> Result<[Option<Geometry>; 2], DenseError> {
        let k = m.min(n);
        let job = |rows, contracted, cols| DenseGemmBatchJob {
            dst_offset: 0,
            lhs_offset: 0,
            rhs_offset: 0,
            rows,
            contracted,
            cols,
        };
        let w = Geometry::new(&job(m, k, n), [MatrixOp::Identity; 2])?;
        let p = if LEFT {
            Geometry::new(&job(n, k, n), [MatrixOp::Adjoint, MatrixOp::Identity])?
        } else {
            Geometry::new(&job(m, k, m), [MatrixOp::Identity, MatrixOp::Adjoint])?
        };
        Ok([Some(w), Some(p)])
    }

    // The products read the scaled factor in lane scratch only after `W`
    // read it unscaled; the destinations receive only `W` and `P`.
    fn factor_into<T: Scalar>(
        raw: RawStridedRef<'_, T>,
        [m, n]: [usize; 2],
        batch: usize,
        outputs: &mut [Option<DenseWrite<'_>>; 3],
        _: FactorPlacement,
        workspace: &mut [MaybeUninit<u8>],
        scratch: &mut [T],
        spectrum: &mut [<T as Element>::Spectrum],
        _: &mut Vec<T>,
        plans: &Plans,
        exec: &mut Exec<'_>,
    ) -> Result<(), JobError> {
        let k = m.min(n);
        let (u, rest) = scratch.split_at_mut(m * k * batch);
        let vh = &mut rest[..k * n * batch];
        let s = &mut spectrum[..k * batch];
        tlinalg::svd::svd_into_init(Self::OP, raw, false, u, s, vh, workspace, exec)?;
        let [Some(w_plan), Some(p_plan)] = plans else {
            unreachable!("the batch resolved every admitted job's plans")
        };
        let [first, second, _] = outputs;
        let (w, p) = if LEFT {
            (first, second)
        } else {
            (second, first)
        };
        let (w, p) = (slot::<T>(w), slot::<T>(p));
        for item in 0..batch {
            let (u, vh) = (&u[item * m * k..][..m * k], &vh[item * k * n..][..k * n]);
            product(|| {
                T::gemm(
                    w_plan.as_ref(),
                    exec,
                    u,
                    vh,
                    &mut w[item * m * n..][..m * n],
                )
            })
            .map_err(JobError::Gemm)?;
        }
        scale_polar_factor(LEFT, [m, n], batch, |index| s[index], u, vh);
        let order = if LEFT { n } else { m };
        for item in 0..batch {
            let x = if LEFT {
                &vh[item * k * n..][..k * n]
            } else {
                &u[item * m * k..][..m * k]
            };
            let p = &mut p[item * order * order..][..order * order];
            product(|| T::gemm(p_plan.as_ref(), exec, x, x, p)).map_err(JobError::Gemm)?;
        }
        project_hermitian_items(p, order);
        Ok(())
    }
}

impl FactorInto for ExpViaEigh {
    const COMPOSITE: u64 = 1;

    // The eigendecomposition and `exp = Y Y^H` (`n n n`).
    fn stage_works<T: tlinalg::FaerScalar>(n: usize, m: usize) -> Option<([f64; 2], [f64; 2])> {
        let eigh = Eigh::estimates::<T>(n, m);
        let n = n as f64;
        Some((eigh, [T::COMPLEX_WEIGHT * n * n * n, 0.0]))
    }

    fn workspace<T: Scalar>(
        dims: [usize; 2],
        batch: usize,
        exec: &Exec<'_>,
    ) -> tlinalg::Result<usize> {
        Eigh::workspace::<T>(dims, batch, exec)
    }

    // `V` of the eigendecomposition.
    fn scratch_len<T: Scalar>([n, _]: [usize; 2], batch: usize, _: FactorPlacement) -> usize {
        n * n * batch
    }

    // Its eigenvalues.
    fn spectrum_len([n, _]: [usize; 2], batch: usize) -> usize {
        n * batch
    }

    // `exp = Y Y^H`, the adjoint operand `Y` itself read transposed and
    // conjugated.
    fn geometries([n, _]: [usize; 2]) -> Result<[Option<Geometry>; 2], DenseError> {
        let job = DenseGemmBatchJob {
            dst_offset: 0,
            lhs_offset: 0,
            rhs_offset: 0,
            rows: n,
            contracted: n,
            cols: n,
        };
        let product = Geometry::new(&job, [MatrixOp::Identity, MatrixOp::Adjoint])?;
        Ok([Some(product), None])
    }

    // A nonfinite eigenvalue stops the job before its scaling; the
    // destination receives only `exp`.
    fn factor_into<T: Scalar>(
        raw: RawStridedRef<'_, T>,
        [n, _]: [usize; 2],
        batch: usize,
        outputs: &mut [Option<DenseWrite<'_>>; 3],
        _: FactorPlacement,
        workspace: &mut [MaybeUninit<u8>],
        scratch: &mut [T],
        spectrum: &mut [<T as Element>::Spectrum],
        _: &mut Vec<T>,
        plans: &Plans,
        exec: &mut Exec<'_>,
    ) -> Result<(), JobError> {
        let vectors = &mut scratch[..n * n * batch];
        let values = &mut spectrum[..n * batch];
        tlinalg::eigh::eigh_into_init(Self::OP, raw, values, vectors, workspace, exec)?;
        let [Some(plan), None] = plans else {
            unreachable!("the batch resolved every admitted job's plan")
        };
        let lambda = |index: usize| values[index];
        crate::factor_dest::finite_eigenvalues((0..n * batch).map(lambda))
            .map_err(JobError::Output)?;
        scale_exp_factor(n, batch, lambda, vectors);
        let out = slot::<T>(&mut outputs[0]);
        for item in 0..batch {
            let y = &vectors[item * n * n..][..n * n];
            let out = &mut out[item * n * n..][..n * n];
            product(|| T::gemm(plan.as_ref(), exec, y, y, out)).map_err(JobError::Gemm)?;
        }
        copytri_items(out, n);
        Ok(())
    }
}

impl FactorInto for Inv {
    fn workspace<T: Scalar>(
        [n, _]: [usize; 2],
        batch: usize,
        exec: &Exec<'_>,
    ) -> tlinalg::Result<usize> {
        tlinalg::lu::lu_solve_workspace::<T>(n, n, false, batch, exec)
    }

    fn scratch_len<T: Scalar>(_: [usize; 2], _: usize, _: FactorPlacement) -> usize {
        0
    }

    // A singular item leaves its destination item holding `I_n`.
    fn factor_into<T: Scalar>(
        raw: RawStridedRef<'_, T>,
        [n, _]: [usize; 2],
        _: usize,
        outputs: &mut [Option<DenseWrite<'_>>; 3],
        _: FactorPlacement,
        workspace: &mut [MaybeUninit<u8>],
        _: &mut [T],
        _: &mut [<T as Element>::Spectrum],
        _: &mut Vec<T>,
        _: &Plans,
        exec: &mut Exec<'_>,
    ) -> Result<(), JobError> {
        let out = slot::<T>(&mut outputs[0]);
        identity_columns(out, n, n, 0);
        let mut step = 1usize;
        // The destination is a compact allocation, so every stride fits.
        let strides: StrideVec = raw
            .dims()
            .iter()
            .map(|&extent| {
                let stride = step as isize;
                step = step.saturating_mul(extent);
                stride
            })
            .collect();
        let out = RawStridedMut::new(out, raw.dims(), &strides, 0).map_err(|err| {
            JobError::Output(tenferro_error(
                Self::ENTRY,
                tenferro_tensor::Error::invalid_argument(Self::VALIDATION, "out", err.to_string()),
            ))
        })?;
        Ok(tlinalg::lu::solve_into(
            Self::OP,
            raw,
            None,
            out,
            false,
            workspace,
            exec,
        )?)
    }
}

/// Runs one dependency product of a destination job; tests count its
/// allocations apart from the job's own owners.
fn product<R>(f: impl FnOnce() -> R) -> R {
    #[cfg(test)]
    return owners::product(f);
    #[cfg(not(test))]
    f()
}

/// One lane's typed scratch per scalar, set up once with the lane.
#[derive(Default)]
pub(crate) struct TypedScratch {
    f32: ScalarScratch<f32>,
    f64: ScalarScratch<f64>,
    c32: ScalarScratch<Complex32>,
    c64: ScalarScratch<Complex64>,
}

/// One scalar's lane scratch: zero-filled values and real spectrum, and an
/// empty Householder coefficient `Vec` whose capacity covers every block of
/// the lane.
#[derive(Default)]
pub(crate) struct ScalarScratch<T: Scalar> {
    values: Vec<T>,
    spectrum: Vec<<T as Element>::Spectrum>,
    coeff: Vec<T>,
}

impl<T: Scalar> ScalarScratch<T> {
    fn new(values: usize, spectrum: usize, coeff: usize) -> Self {
        Self {
            values: vec![T::ZERO; values],
            spectrum: vec![Default::default(); spectrum],
            coeff: Vec::with_capacity(coeff),
        }
    }
}

/// One destination lane: its blocks' largest needs, the count of its blocks
/// still to run and, from its first block's run to its last, its buffers.
///
/// Why the buffers are not built in [`BlockOp::lane`]: that runs on the
/// caller before the fork, so every lane's setup and release would be serial
/// there while their pages still fault on the lane threads. The lane's own
/// thread builds them in its first block's run and frees them after its
/// last; a panic drops whatever remains through `Drop`.
pub(crate) struct IntoLane {
    needs: Needs,
    remaining: usize,
    buffers: Option<LaneBuffers>,
}

/// One lane's provider workspace and typed scratch: the largest need of the
/// lane's blocks, allocated once before its first block.
pub(crate) struct LaneBuffers {
    workspace: Vec<MaybeUninit<u8>>,
    typed: TypedScratch,
    /// A process-unique id of this lane setup.
    #[cfg(test)]
    id: usize,
    /// The allocation calls of this lane's buffer setup.
    #[cfg(test)]
    setup_allocs: Option<u64>,
    /// The thread that set the buffers up.
    #[cfg(test)]
    thread: std::thread::ThreadId,
}

/// Records which thread frees each lane setup's buffers.
#[cfg(test)]
impl Drop for LaneBuffers {
    fn drop(&mut self) {
        owners::released(self.id);
    }
}

/// The largest workspace bytes and typed scratch elements (per scalar, in
/// [`TypedScratch`] field order) of one lane's blocks.
#[derive(Default)]
struct Needs {
    bytes: usize,
    typed: [usize; 4],
    spectrum: [usize; 4],
    coeff: [usize; 4],
}

/// One block of a destination batch: its validated input, its signed
/// strides, its destination, its resolved product plans, a plan resolution
/// or workspace query that failed before its solve, and, once run, its
/// outcome.
pub(crate) struct IntoJob<'a, 'd, 'w> {
    input: DenseRead<'a>,
    strides: StrideVec,
    dest: &'d mut FactorDest<'w>,
    plans: Plans,
    sizing: Option<DenseError>,
    outcome: Option<Result<(), FactorFailure>>,
    /// The id, workspace length and setup thread of the lane setup this
    /// block ran with, and the thread of its solve.
    #[cfg(test)]
    lane_seen: Option<(usize, usize, std::thread::ThreadId, std::thread::ThreadId)>,
}

/// Factorization `F` of each block into its destination on the lane driver.
struct IntoBlocks<'a, 'd, 'w, F>(PhantomData<IntoMarker<'a, 'd, 'w, F>>);

/// [`IntoBlocks`]' variance and auto-trait marker.
type IntoMarker<'a, 'd, 'w, F> = fn() -> (&'a (), &'d (), &'w (), F);

impl<'a, 'd, 'w: 'd, F: FactorInto> BlockOp for IntoBlocks<'a, 'd, 'w, F> {
    type Job = IntoJob<'a, 'd, 'w>;
    const ENTRY: Entry = Entry::Linalg;
    const KIND: OpKind = OpKind::Factorization;
    const LABEL: &'static str = F::ENTRY;
    type Lane = IntoLane;

    fn lane(&self, jobs: &mut LaneMembers<'_, Self::Job>, exec: &Exec<'_>) -> IntoLane {
        let mut needs = Needs::default();
        let mut remaining = 0;
        jobs(&mut |job| {
            remaining += 1;
            if let Err(error) = needs_of::<F>(job.input, job.dest.placement(), exec, &mut needs) {
                job.sizing = Some(error);
            }
        });
        IntoLane {
            needs,
            remaining,
            buffers: None,
        }
    }

    fn work(&self, _: usize, job: &Self::Job) -> f64 {
        work::<F>(job.input)
    }

    fn serial(&self, _: usize, job: &Self::Job, model: &WorkModel, work: f64) -> Duration {
        match stages::<F>(job, model, None) {
            Some(stages) => stages
                .iter()
                .fold(Duration::ZERO, |sum, &(time, _)| sum.saturating_add(time)),
            None => driver::cost(OpKind::Factorization, work, model),
        }
    }

    fn modeled(&self, _: usize, job: &Self::Job) -> bool {
        single_item::<F>(job.input).is_some()
    }

    fn grant(
        &self,
        _: usize,
        job: &Self::Job,
        exec: &Exec<'_>,
        budget: usize,
        _: Duration,
    ) -> Grant {
        match stages::<F>(job, &exec.batch_policy().model, Some((exec, budget))) {
            Some(stages) => stages.map(|(_, width)| width),
            None => lanes::single(
                single_item::<F>(job.input).map_or(1, |item| grant(exec, budget, item)),
            ),
        }
    }

    fn item(
        &self,
        _: usize,
        job: &Self::Job,
        exec: &Exec<'_>,
        serial: Duration,
        grant: Grant,
    ) -> Duration {
        match stages::<F>(job, &exec.batch_policy().model, None) {
            Some(stages) => stages
                .iter()
                .zip(grant)
                .fold(Duration::ZERO, |sum, (&(time, _), width)| {
                    sum.saturating_add(lanes::item_time(exec, time, width))
                }),
            None => lanes::item_time(exec, serial, grant[0]),
        }
    }

    // A composite job's stages are a function of its op, its scalar, its
    // extents and whether its plans resolved, given the context: its
    // estimates, product geometries and plans (a plan is a pure function of
    // scalar and geometry).
    fn grant_key(&self, _: usize, job: &Self::Job) -> GrantKey {
        match (F::COMPOSITE, job.input.shape()) {
            (0, _) | (_, [] | [_]) => item_key(single_item::<F>(job.input)),
            (op, &[m, n, ..]) => {
                let complete =
                    u64::from(job.plans.iter().flatten().count() == products::<F>([m, n]));
                [
                    m as u64,
                    n as u64,
                    (op << 3) | (dtype_tag(job.input) << 1) | complete,
                ]
            }
        }
    }

    // A block's error is its own outcome, not the driver's: the caller
    // delivers every block's outcome in input order.
    fn run(
        &self,
        _: usize,
        job: &mut Self::Job,
        into: &mut IntoLane,
        mut exec: Exec<'_>,
    ) -> Result<(), DenseError> {
        let needs = &into.needs;
        let lane = into.buffers.get_or_insert_with(|| LaneBuffers::new(needs));
        #[cfg(test)]
        {
            job.lane_seen = Some((
                lane.id,
                lane.workspace.len(),
                lane.thread,
                std::thread::current().id(),
            ));
        }
        let sizing = job.sizing.take();
        #[cfg(test)]
        let (setup, id, input) = (lane.setup_allocs, lane.id, owners::address(job.input));
        let plans = &job.plans;
        let solve = || match job.input {
            DenseRead::F32(view) => factor_into_typed::<F, _>(
                view,
                &job.strides,
                job.dest,
                plans,
                sizing,
                lane,
                &mut exec,
            ),
            DenseRead::F64(view) => factor_into_typed::<F, _>(
                view,
                &job.strides,
                job.dest,
                plans,
                sizing,
                lane,
                &mut exec,
            ),
            DenseRead::C32(view) => factor_into_typed::<F, _>(
                view,
                &job.strides,
                job.dest,
                plans,
                sizing,
                lane,
                &mut exec,
            ),
            DenseRead::C64(view) => factor_into_typed::<F, _>(
                view,
                &job.strides,
                job.dest,
                plans,
                sizing,
                lane,
                &mut exec,
            ),
            _ => unreachable!("preflight admits the provider's dtypes only"),
        };
        #[cfg(test)]
        let outcome = {
            let (outcome, solve_allocs) = owners::measured(solve);
            owners::record(input, id, setup, solve_allocs, owners::take_products());
            outcome
        };
        #[cfg(not(test))]
        let outcome = solve();
        job.outcome = Some(outcome);
        into.remaining = into.remaining.saturating_sub(1);
        if into.remaining == 0 {
            into.buffers = None;
        }
        Ok(())
    }
}

impl LaneBuffers {
    /// Allocates one lane's buffers at its blocks' largest `needs`.
    fn new(needs: &Needs) -> Self {
        let build = || {
            (
                vec![MaybeUninit::uninit(); needs.bytes],
                TypedScratch {
                    f32: ScalarScratch::new(needs.typed[0], needs.spectrum[0], needs.coeff[0]),
                    f64: ScalarScratch::new(needs.typed[1], needs.spectrum[1], needs.coeff[1]),
                    c32: ScalarScratch::new(needs.typed[2], needs.spectrum[2], needs.coeff[2]),
                    c64: ScalarScratch::new(needs.typed[3], needs.spectrum[3], needs.coeff[3]),
                },
            )
        };
        #[cfg(test)]
        let ((workspace, typed), setup_allocs) = owners::measured(build);
        #[cfg(not(test))]
        let (workspace, typed) = build();
        LaneBuffers {
            workspace,
            typed,
            #[cfg(test)]
            id: {
                static SETUPS: std::sync::atomic::AtomicUsize =
                    std::sync::atomic::AtomicUsize::new(1);
                SETUPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            },
            #[cfg(test)]
            setup_allocs,
            #[cfg(test)]
            thread: std::thread::current().id(),
        }
    }
}

/// Test-only observation of the destination batch's allocation owners: each
/// lane's buffer setup and each block's solve (provider call included), and
/// apart from the solve the allocations of a job's dependency products,
/// measured on the thread that runs them and kept by input payload address.
#[cfg(test)]
pub(crate) mod owners {
    use std::sync::{Mutex, PoisonError};

    use crate::tests::counting_alloc;
    use crate::DenseRead;

    /// One solved block: its input's payload address, its lane setup's id
    /// and allocation calls, the allocation calls inside its solve outside
    /// its products, and those inside its products.
    #[derive(Clone, Copy, Debug)]
    pub(crate) struct Record {
        pub(crate) input: usize,
        pub(crate) lane: usize,
        pub(crate) setup: Option<u64>,
        pub(crate) solve: Option<u64>,
        pub(crate) products: u64,
    }

    static RECORDS: Mutex<Vec<Record>> = Mutex::new(Vec::new());

    thread_local! {
        /// Calls counted before each product of the window in flight, and
        /// the calls inside those products.
        static OUTSIDE: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
        static PRODUCTS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    }

    /// `f`'s allocation calls on this thread outside the products it runs
    /// ([`product`]), `None` inside another measurement window, which a
    /// nested count would reset.
    pub(crate) fn measured<T>(f: impl FnOnce() -> T) -> (T, Option<u64>) {
        if counting_alloc::active() {
            return (f(), None);
        }
        OUTSIDE.set(0);
        PRODUCTS.set(0);
        let (value, allocs) = counting_alloc::measure(f);
        (value, Some(allocs.calls + OUTSIDE.take()))
    }

    /// Runs one dependency product inside a [`measured`] window, counting
    /// its allocations apart from the window's own.
    pub(super) fn product<T>(f: impl FnOnce() -> T) -> T {
        if !counting_alloc::active() {
            return f();
        }
        OUTSIDE.set(OUTSIDE.get() + counting_alloc::stop().calls);
        let (value, allocs) = counting_alloc::measure(f);
        PRODUCTS.set(PRODUCTS.get() + allocs.calls);
        counting_alloc::start();
        value
    }

    /// The product calls counted since the last [`measured`] window began.
    pub(super) fn take_products() -> u64 {
        PRODUCTS.take()
    }

    pub(crate) fn address(input: DenseRead<'_>) -> usize {
        match input {
            DenseRead::F32(view) => view.data()[view.offset()..].as_ptr() as usize,
            DenseRead::F64(view) => view.data()[view.offset()..].as_ptr() as usize,
            DenseRead::C32(view) => view.data()[view.offset()..].as_ptr() as usize,
            DenseRead::C64(view) => view.data()[view.offset()..].as_ptr() as usize,
            DenseRead::I32(view) => view.data()[view.offset()..].as_ptr() as usize,
            DenseRead::I64(view) => view.data()[view.offset()..].as_ptr() as usize,
            DenseRead::Bool(view) => view.data()[view.offset()..].as_ptr() as usize,
        }
    }

    pub(crate) fn record(
        input: usize,
        lane: usize,
        setup: Option<u64>,
        solve: Option<u64>,
        products: u64,
    ) {
        RECORDS
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(Record {
                input,
                lane,
                setup,
                solve,
                products,
            });
    }

    static RELEASES: Mutex<Vec<(usize, std::thread::ThreadId)>> = Mutex::new(Vec::new());

    /// Records that lane setup `lane`'s buffers are freed on this thread.
    pub(super) fn released(lane: usize) {
        RELEASES
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push((lane, std::thread::current().id()));
    }

    /// The threads that freed lane setup `lane`'s buffers.
    pub(super) fn releases_of(lane: usize) -> Vec<std::thread::ThreadId> {
        RELEASES
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .filter(|&&(id, _)| id == lane)
            .map(|&(_, thread)| thread)
            .collect()
    }

    /// The position the next record takes.
    pub(crate) fn mark() -> usize {
        RECORDS.lock().unwrap_or_else(PoisonError::into_inner).len()
    }

    /// The records since `mark` of blocks whose input payload lies in
    /// `range` (other tests record concurrently).
    pub(crate) fn within(mark: usize, range: std::ops::Range<usize>) -> Vec<Record> {
        RECORDS.lock().unwrap_or_else(PoisonError::into_inner)[mark..]
            .iter()
            .filter(|record| range.contains(&record.input))
            .copied()
            .collect()
    }
}

/// Folds `input`'s workspace and typed scratch into `needs` at the lane's
/// context `exec`. An input whose rank, squareness or extents the solve
/// refuses or skips needs nothing; a failed workspace query is that input's
/// own outcome.
fn needs_of<F: FactorInto>(
    input: DenseRead<'_>,
    placement: FactorPlacement,
    exec: &Exec<'_>,
    needs: &mut Needs,
) -> Result<(), DenseError> {
    fn of<F: FactorInto, T: Scalar>(
        shape: &[usize],
        placement: FactorPlacement,
        exec: &Exec<'_>,
        slot: usize,
        needs: &mut Needs,
    ) -> Result<(), DenseError> {
        let [m, n, batch @ ..] = shape else {
            return Ok(());
        };
        if F::SQUARE && m != n || F::admit([*m, *n]).is_err() || shape.contains(&0) {
            return Ok(());
        }
        // The view's element count fits, so its batch product does.
        let items = batch.iter().product::<usize>();
        let bytes = F::workspace::<T>([*m, *n], items, exec)
            .map_err(|err| F::provider_failure(err, shape))?;
        needs.bytes = needs.bytes.max(bytes);
        needs.typed[slot] = needs.typed[slot].max(F::scratch_len::<T>([*m, *n], items, placement));
        needs.spectrum[slot] = needs.spectrum[slot].max(F::spectrum_len([*m, *n], items));
        needs.coeff[slot] = needs.coeff[slot].max(F::coeff_len([*m, *n], items));
        Ok(())
    }
    match input {
        DenseRead::F32(view) => of::<F, f32>(view.shape(), placement, exec, 0, needs),
        DenseRead::F64(view) => of::<F, f64>(view.shape(), placement, exec, 1, needs),
        DenseRead::C32(view) => of::<F, Complex32>(view.shape(), placement, exec, 2, needs),
        DenseRead::C64(view) => of::<F, Complex64>(view.shape(), placement, exec, 3, needs),
        _ => Ok(()),
    }
}

/// [`factor_typed`]'s checks in the same order (rank, square, the op's own
/// admission, the empty-extent shortcut, layout), then the failed plan
/// resolution or workspace query, or the destination job on the lane's
/// buffers; every failure but the job's output check is
/// [`FactorFailure::Solver`].
#[allow(clippy::too_many_arguments)]
fn factor_into_typed<F: FactorInto, T: Scalar>(
    view: DenseView<'_, T>,
    strides: &[isize],
    dest: &mut FactorDest<'_>,
    plans: &Plans,
    sizing: Option<DenseError>,
    lane: &mut LaneBuffers,
    exec: &mut Exec<'_>,
) -> Result<(), FactorFailure> {
    let shape = view.shape();
    let op = F::validation(shape);
    let [m, n, batch @ ..] = shape else {
        return Err(FactorFailure::Solver(tenferro_error(
            F::ENTRY,
            tenferro_tensor::Error::rank_mismatch(op, 2, shape.len()),
        )));
    };
    if F::SQUARE && m != n {
        return Err(FactorFailure::Solver(tenferro_error(
            F::ENTRY,
            tenferro_tensor::Error::shape_mismatch(op, vec![*m], vec![*n]),
        )));
    }
    F::admit([*m, *n]).map_err(FactorFailure::Solver)?;
    if shape.contains(&0) {
        F::empty_into::<T>([*m, *n], &mut dest.outputs);
        return Ok(());
    }
    // A nonempty view's element count fits, so its batch product does.
    let items = batch.iter().product::<usize>();
    with_provider_read::<F, T, _>(view, strides, |raw| {
        if let Some(error) = sizing {
            return Ok(Err(FactorFailure::Solver(error)));
        }
        let placement = dest.placement();
        let scratch = T::scratch(&mut lane.typed);
        let solved = F::factor_into::<T>(
            raw,
            [*m, *n],
            items,
            &mut dest.outputs,
            placement,
            &mut lane.workspace,
            &mut scratch.values,
            &mut scratch.spectrum,
            &mut scratch.coeff,
            plans,
            exec,
        );
        Ok(solved.map_err(|error| match error {
            JobError::Provider(err) => FactorFailure::Solver(F::provider_failure(err, shape)),
            JobError::Gemm(err) => FactorFailure::Solver(err),
            JobError::Output(err) => FactorFailure::Output(err),
        }))
    })
    .map_err(FactorFailure::Solver)?
}

/// The destination batch's dispatch over the ops with a destination form;
/// the caller validated `op` and `dests`.
pub(crate) fn factorize_into(
    engine: &mut dyn tenferro_tensor::Engine,
    op: crate::DenseFactorization,
    inputs: &[DenseRead<'_>],
    dests: &mut [FactorDest<'_>],
    sink: &mut FactorSink<'_>,
) -> Result<(), DenseError> {
    use crate::DenseFactorization as Op;
    match op {
        Op::Svd => factorize_into_batch::<Svd>(engine, inputs, dests, sink),
        Op::SvdFull => factorize_into_batch::<SvdFull>(engine, inputs, dests, sink),
        Op::Qr => factorize_into_batch::<Qr>(engine, inputs, dests, sink),
        Op::Eigh => factorize_into_batch::<Eigh>(engine, inputs, dests, sink),
        Op::Lq => factorize_into_batch::<Lq>(engine, inputs, dests, sink),
        Op::QrFull => factorize_into_batch::<QrFull>(engine, inputs, dests, sink),
        Op::QrNull => factorize_into_batch::<QrNull>(engine, inputs, dests, sink),
        Op::LeftPolar => factorize_into_batch::<Polar<true>>(engine, inputs, dests, sink),
        Op::RightPolar => factorize_into_batch::<Polar<false>>(engine, inputs, dests, sink),
        Op::ExpViaEigh => factorize_into_batch::<ExpViaEigh>(engine, inputs, dests, sink),
        Op::Inv => factorize_into_batch::<Inv>(engine, inputs, dests, sink),
        _ => unreachable!("validation admits the ops with a destination form only"),
    }
}

fn factorize_into_batch<F: FactorInto>(
    engine: &mut dyn tenferro_tensor::Engine,
    inputs: &[DenseRead<'_>],
    dests: &mut [FactorDest<'_>],
    sink: &mut FactorSink<'_>,
) -> Result<(), DenseError> {
    with_cpu_engine(engine, |cpu| batch_into_on::<F>(cpu, inputs, dests, sink)).unwrap_or_else(
        || {
            Err(tenferro_error(
                F::ENTRY,
                tenferro_tensor::Error::unsupported(F::ENTRY, NOT_A_CPU_ENGINE),
            ))
        },
    )
}

/// [`batch_on`] into destinations: the same preflight, admission, lanes and
/// ordered post-join delivery, with each block's outcome kept as
/// `Result<(), FactorFailure>` instead of its factors.
fn batch_into_on<F: FactorInto>(
    cpu: &mut CpuEngine,
    inputs: &[DenseRead<'_>],
    dests: &mut [FactorDest<'_>],
    sink: &mut FactorSink<'_>,
) -> Result<(), DenseError> {
    let mut jobs = Vec::with_capacity(inputs.len());
    let mut slots = dests.iter_mut();
    let mut refused = None;
    for (&input, dest) in inputs.iter().zip(&mut slots) {
        match preflight::<F>(input) {
            Ok(strides) => jobs.push(IntoJob {
                input,
                strides,
                dest,
                plans: [None, None],
                sizing: None,
                outcome: None,
                #[cfg(test)]
                lane_seen: None,
            }),
            Err(error) => {
                refused = Some((error, dest));
                break;
            }
        }
    }
    resolve_plans::<F>(&mut jobs);
    if !jobs.is_empty() {
        driver::drive(
            cpu,
            &IntoBlocks::<F>(PhantomData),
            &mut jobs,
            &mut LaneTables::default(),
        )?;
    }
    let admitted = jobs.len();
    for (index, job) in jobs.into_iter().enumerate() {
        let outcome = job
            .outcome
            .unwrap_or_else(|| unreachable!("the driver runs every admitted job"));
        sink(index, job.dest, outcome)?;
    }
    match refused {
        Some((error, dest)) => sink(admitted, dest, Err(FactorFailure::Solver(error))),
        None => Ok(()),
    }
}

/// [`Scalar::plan`] of one scalar.
type PlanOf = fn(Geometry) -> Result<Arc<ErasedPlan>, DenseError>;

/// Resolves the product plans of every job that reaches its solve (a
/// nonempty input of rank two or more that `F` admits) before the lanes,
/// once per distinct scalar and geometry, from the geometry-plan cache; a
/// failure is that job's own outcome.
fn resolve_plans<F: FactorInto>(jobs: &mut [IntoJob<'_, '_, '_>]) {
    let mut resolved: HashMap<(u8, Geometry), Arc<ErasedPlan>> = HashMap::new();
    for job in jobs {
        let [m, n, ..] = *job.input.shape() else {
            continue;
        };
        if F::SQUARE && m != n || F::admit([m, n]).is_err() || job.input.shape().contains(&0) {
            continue;
        }
        let geometries = match F::geometries([m, n]) {
            Ok(geometries) => geometries,
            Err(error) => {
                job.sizing = Some(error);
                continue;
            }
        };
        let (tag, plan): (u8, PlanOf) = match job.input {
            DenseRead::F32(_) => (0, f32::plan),
            DenseRead::F64(_) => (1, f64::plan),
            DenseRead::C32(_) => (2, Complex32::plan),
            DenseRead::C64(_) => (3, Complex64::plan),
            _ => continue,
        };
        for (slot, geometry) in job.plans.iter_mut().zip(geometries) {
            let Some(geometry) = geometry else {
                continue;
            };
            let found = match resolved.entry((tag, geometry)) {
                std::collections::hash_map::Entry::Occupied(entry) => Ok(entry.get().clone()),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    plan(geometry).map(|plan| entry.insert(plan).clone())
                }
            };
            match found {
                Ok(plan) => *slot = Some(plan),
                Err(error) => {
                    job.sizing = Some(error);
                    break;
                }
            }
        }
    }
}

/// tlinalg's per-item estimates of `F` for `input`'s scalar (see
/// [`Factor::estimates`]) and its batch product, or `None` for an input the
/// provider refuses (rank or dtype).
fn item_estimates<F: Factor>(input: DenseRead<'_>) -> Option<([f64; 2], f64)> {
    // Direct generic calls, not a function pointer: the estimates inline, and
    // `work`, which reads only the total, need not evaluate the payback.
    fn of<F: Factor, T: tlinalg::FaerScalar>(shape: &[usize]) -> Option<([f64; 2], f64)> {
        let [m, n, batch @ ..] = shape else {
            return None;
        };
        let items = batch.iter().map(|&b| b as f64).product();
        Some((F::estimates::<T>(*m, *n), items))
    }
    match input {
        DenseRead::F32(view) => of::<F, f32>(view.shape()),
        DenseRead::F64(view) => of::<F, f64>(view.shape()),
        DenseRead::C32(view) => of::<F, Complex32>(view.shape()),
        DenseRead::C64(view) => of::<F, Complex64>(view.shape()),
        _ => None,
    }
}

/// tlinalg's work estimate of `F` over every batch item of `input`, in
/// weighted multiply-adds (four per complex multiply-add).
pub(crate) fn work<F: Factor>(input: DenseRead<'_>) -> f64 {
    item_estimates::<F>(input).map_or(0.0, |([total, _], items)| total * items)
}

/// tlinalg's (total, payback) estimates of `F` for `input` when its batch
/// product is one, the only inputs whose provider width [`grant`] composes.
///
/// Why not a batched input: tlinalg's route for several items first gates the
/// whole batch, then splits it into lanes and caps each item at
/// `ceil(budget / lanes)` (`batch::decide`), a decision its public queries do
/// not expose; a batch holding such an input keeps the baseline placement.
pub(crate) fn single_item<F: Factor>(input: DenseRead<'_>) -> Option<[f64; 2]> {
    item_estimates::<F>(input).and_then(|(item, items)| (items == 1.0).then_some(item))
}

/// The width tlinalg gives one item of `estimates = [total, payback]`
/// weighted multiply-adds on a lane of `budget` slots of `exec` under Auto:
/// `batch::decide` for a batch of one takes the batch decision's item width
/// (`width_for` over the total's time, one section) and caps it at the width
/// the payback work's time pays for, both under the engine's work model; a
/// token without a team runs the item serially.
pub(crate) fn grant(exec: &Exec<'_>, budget: usize, [total, payback]: [f64; 2]) -> usize {
    if !exec.token_ref().carries_team() {
        return 1;
    }
    let model = &exec.batch_policy().model;
    let width = |work| exec.width_for_within(budget, driver::work_model_ns(model, work), 1);
    width(total).min(width(payback))
}

/// A composite job's whole (total, payback) estimates: its factorization's
/// plus its products' weighted work, which a team inside the item can also
/// use. The batch decision's query reads them; the lane model prices the
/// stages one by one.
fn composite_estimates(stages: Option<([f64; 2], [f64; 2])>) -> [f64; 2] {
    let Some(([total, payback], [first, second])) = stages else {
        unreachable!("a composite job has stages")
    };
    let products = first + second;
    [total + products, payback + products]
}

/// A single-stage factorization's grant key: its (total, payback) estimates
/// by bits, zero for an input without one; the last word, zero, keeps it
/// apart from every composite key.
fn item_key(item: Option<[f64; 2]>) -> GrantKey {
    let [total, payback] = item.map_or([0; 2], |item| item.map(f64::to_bits));
    [total, payback, 0]
}

/// `input`'s scalar, among the provider's.
fn dtype_tag(input: DenseRead<'_>) -> u64 {
    match input {
        DenseRead::F32(_) => 0,
        DenseRead::F64(_) => 1,
        DenseRead::C32(_) => 2,
        _ => 3,
    }
}

/// The products a composite job of matrix extents `[m, n]` runs: those of
/// nonzero work (whether a work is zero does not depend on the scalar).
fn products<F: FactorInto>([m, n]: [usize; 2]) -> usize {
    F::stage_works::<f64>(m, n).map_or(0, |(_, works)| works.iter().filter(|&&w| w > 0.0).count())
}

/// A composite job's sequential stages (see [`FactorInto::stage_works`]),
/// `None` for a single-stage `F` or an input the provider refuses: its
/// factorization, then each product, each with its serial time over every
/// batch item under `model` and, given a `lane` of `budget` slots of
/// `exec`, its provider's width there for one item (one slot otherwise). A
/// product whose plan the batch did not resolve belongs to a job that runs no
/// solve; it keeps its time and takes one slot.
fn stages<F: FactorInto>(
    job: &IntoJob<'_, '_, '_>,
    model: &WorkModel,
    lane: Option<(&Exec<'_>, usize)>,
) -> Option<[(Duration, usize); STAGES]> {
    if F::COMPOSITE == 0 {
        return None;
    }
    let shape = job.input.shape();
    match job.input {
        DenseRead::F32(_) => stages_of::<F, f32>(shape, &job.plans, model, lane),
        DenseRead::F64(_) => stages_of::<F, f64>(shape, &job.plans, model, lane),
        DenseRead::C32(_) => stages_of::<F, Complex32>(shape, &job.plans, model, lane),
        DenseRead::C64(_) => stages_of::<F, Complex64>(shape, &job.plans, model, lane),
        _ => None,
    }
}

/// [`stages`] at scalar `T`.
fn stages_of<F: FactorInto, T: Scalar>(
    shape: &[usize],
    plans: &Plans,
    model: &WorkModel,
    lane: Option<(&Exec<'_>, usize)>,
) -> Option<[(Duration, usize); STAGES]> {
    let [m, n, ref batch @ ..] = *shape else {
        return None;
    };
    let items: f64 = batch.iter().map(|&b| b as f64).product();
    let (factorization, products) = F::stage_works::<T>(m, n)?;
    let mut stages = [(Duration::ZERO, 1); STAGES];
    stages[0] = (
        driver::cost(OpKind::Factorization, factorization[0] * items, model),
        lane.map_or(1, |(exec, budget)| grant(exec, budget, factorization)),
    );
    for ((stage, work), plan) in stages[1..].iter_mut().zip(products).zip(plans) {
        let time = driver::cost(OpKind::Gemm, work * items, model);
        let width = match (lane, plan) {
            (Some((exec, budget)), Some(plan)) if work > 0.0 => {
                T::product_grant(plan.as_ref(), work, exec, budget, time)
            }
            _ => 1,
        };
        *stage = (time, width);
    }
    Some(stages)
}

/// The checks the per-matrix entry makes before it enters the engine: the
/// tenferro view, then the provider's dtype set.
fn preflight<F: Factor>(input: DenseRead<'_>) -> Result<StrideVec, DenseError> {
    let view = tenferro_view(input)?;
    let dtype = view.dtype();
    if !matches!(
        input,
        DenseRead::F32(_) | DenseRead::F64(_) | DenseRead::C32(_) | DenseRead::C64(_)
    ) {
        let op = F::VALIDATION;
        return Err(tenferro_error(
            F::ENTRY,
            linalg_error(op, tenferro_linalg::Error::UnsupportedDType { op, dtype }),
        ));
    }
    strides_to_isize(view_strides(input))
}

fn view_strides(input: DenseRead<'_>) -> &[usize] {
    match input {
        DenseRead::F32(view) => view.strides(),
        DenseRead::F64(view) => view.strides(),
        DenseRead::I32(view) => view.strides(),
        DenseRead::I64(view) => view.strides(),
        DenseRead::Bool(view) => view.strides(),
        DenseRead::C32(view) => view.strides(),
        DenseRead::C64(view) => view.strides(),
    }
}

fn factor<F: Factor>(
    input: DenseRead<'_>,
    strides: &[isize],
    exec: &mut Exec<'_>,
) -> Result<Vec<DenseTensor>, DenseError> {
    match input {
        DenseRead::F32(view) => factor_typed::<F, _>(view, strides, exec),
        DenseRead::F64(view) => factor_typed::<F, _>(view, strides, exec),
        DenseRead::C32(view) => factor_typed::<F, _>(view, strides, exec),
        DenseRead::C64(view) => factor_typed::<F, _>(view, strides, exec),
        _ => unreachable!("preflight admits the provider's dtypes only"),
    }
}

/// tenferro's `svd_impl`/`qr_impl`/`eigh_impl`/`eig_impl` (values-only forms) on
/// one input: rank check, the square check where the provider requires one,
/// the empty-extent shortcut, then the provider's batched factorization.
fn factor_typed<F: Factor, T: Scalar>(
    view: DenseView<'_, T>,
    strides: &[isize],
    exec: &mut Exec<'_>,
) -> Result<Vec<DenseTensor>, DenseError> {
    let op = F::VALIDATION;
    let shape = view.shape();
    let [m, n, batch @ ..] = shape else {
        return Err(tenferro_error(
            F::ENTRY,
            tenferro_tensor::Error::rank_mismatch(op, 2, shape.len()),
        ));
    };
    if F::SQUARE && m != n {
        return Err(tenferro_error(
            F::ENTRY,
            tenferro_tensor::Error::shape_mismatch(op, vec![*m], vec![*n]),
        ));
    }
    let dims = [*m, *n];
    if shape.contains(&0) {
        return F::factor::<T>(None, dims, batch, exec);
    }
    // The provider's safe `Vec` entry: its workspace query and `*_into` on
    // one per-call region. The owned outcome needs fresh factor storage per
    // block anyway; the destination batch (`factor_into_typed`) is the
    // lane-reused-workspace route.
    with_provider_read::<F, T, _>(view, strides, |raw| F::factor(Some(raw), dims, batch, exec))
}

/// Runs `solve` on the provider's read of the nonempty `view`: the view
/// itself, or for a [`Factor::TRANSPOSED`] factorization its transpose (the
/// first two extents and strides swapped, no copy).
fn with_provider_read<F: Factor, T, R>(
    view: DenseView<'_, T>,
    strides: &[isize],
    solve: impl FnOnce(RawStridedRef<'_, T>) -> Result<R, DenseError>,
) -> Result<R, DenseError> {
    // `DenseView` validated the offset against the storage, and the tenferro
    // view (preflight) proved it fits `isize`.
    let offset = isize::try_from(view.offset()).map_err(|_| DenseError::OffsetOverflow {
        value: view.offset(),
    })?;
    let (swapped_shape, swapped_strides): (ShapeVec, StrideVec);
    let (shape, strides) = if F::TRANSPOSED {
        let mut shape: ShapeVec = view.shape().iter().copied().collect();
        let mut strides: StrideVec = strides.iter().copied().collect();
        shape.swap(0, 1);
        strides.swap(0, 1);
        (swapped_shape, swapped_strides) = (shape, strides);
        (&swapped_shape[..], &swapped_strides[..])
    } else {
        (view.shape(), strides)
    };
    let raw = RawStridedRef::new(view.data(), shape, strides, offset).map_err(|err| {
        tenferro_error(
            F::ENTRY,
            tenferro_tensor::Error::invalid_argument(F::VALIDATION, "layout", err.to_string()),
        )
    })?;
    solve(raw)
}

pub(super) fn linalg_error(
    op: &'static str,
    error: tenferro_linalg::Error,
) -> tenferro_tensor::Error {
    tenferro_tensor::Error::extension(
        op,
        tenferro_linalg::LINALG_EXTENSION_FAMILY_ID,
        error.kind(),
        error,
    )
}

/// tenferro-linalg's `map_error` for the faer provider (`cpu/tlinalg_error.rs`),
/// so a provider failure is the same `tenferro_tensor::Error` value either
/// route reports.
fn provider_error(op: tlinalg::Op, error: tlinalg::Error) -> tenferro_tensor::Error {
    use tenferro_linalg::Error as Linalg;
    let op = op.as_str();
    match error {
        tlinalg::Error::NonConvergence { .. } => linalg_error(op, Linalg::NonConvergence { op }),
        tlinalg::Error::NonFinite { role, .. } => linalg_error(
            op,
            Linalg::NonFinite {
                op,
                role: role.as_str(),
            },
        ),
        tlinalg::Error::Singular { .. } => linalg_error(op, Linalg::Singular { op }),
        tlinalg::Error::InvalidArgument { role, detail, .. } => {
            tenferro_tensor::Error::invalid_argument(op, role, detail)
        }
        // tenferro wraps the same fields in its private `BackendError`, whose
        // message is this variant's; the faer provider never reports it.
        error @ tlinalg::Error::InvalidWorkspace { .. } => {
            tenferro_tensor::Error::backend_source(op, error)
        }
        tlinalg::Error::Inconsistent { detail, .. } => {
            tenferro_tensor::Error::Internal(format!("{op}: {detail}"))
        }
        error @ tlinalg::Error::InsufficientWorkspace { .. } => {
            tenferro_tensor::Error::runtime_state_source(op, error)
        }
        other => tenferro_tensor::Error::Internal(format!(
            "{op}: unmapped linalg provider error {other:?}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use tcontract_exec::{BatchPolicy, WorkModel};
    use tenferro_cpu::Pool;

    use super::*;

    /// Factorization `F`'s own op with recording-only execution: cost,
    /// search admission and grant are `Blocks<F>`'s, and each block records
    /// the reserved width of the lane it runs on instead of factoring.
    struct RecordsWidth<'a, F>(Blocks<'a, F>, Vec<AtomicUsize>);

    impl<'a, F: Factor> BlockOp for RecordsWidth<'a, F> {
        type Job = Job<'a>;
        const ENTRY: Entry = Entry::Linalg;
        const KIND: OpKind = OpKind::Factorization;
        const LABEL: &'static str = F::ENTRY;
        type Lane = ();

        fn lane(&self, _: &mut LaneMembers<'_, Job<'a>>, _: &Exec<'_>) {}

        fn work(&self, index: usize, job: &Job<'a>) -> f64 {
            self.0.work(index, job)
        }

        fn modeled(&self, index: usize, job: &Job<'a>) -> bool {
            self.0.modeled(index, job)
        }

        fn grant(
            &self,
            index: usize,
            job: &Job<'a>,
            exec: &Exec<'_>,
            budget: usize,
            serial: Duration,
        ) -> Grant {
            self.0.grant(index, job, exec, budget, serial)
        }

        fn grant_key(&self, index: usize, job: &Job<'a>) -> GrantKey {
            self.0.grant_key(index, job)
        }

        fn run(
            &self,
            index: usize,
            _: &mut Job<'a>,
            (): &mut (),
            exec: Exec<'_>,
        ) -> Result<(), DenseError> {
            self.1[index].store(exec.budget(), Ordering::Relaxed);
            Ok(())
        }
    }

    /// The reserved lane width each real input of `shapes` runs on under
    /// factorization `F`, on `threads` slots of flat 100 ns dispatch at one
    /// multiply-add per ns, no item overhead and a lane work minimum of
    /// `lane_ns`.
    fn lane_widths<F: Factor>(threads: usize, lane_ns: f64, shapes: &[&[usize]]) -> Vec<usize> {
        use tenferro_cpu::threadteam::{DispatchCosts, WidthCosts};
        let row = WidthCosts::flat(Duration::from_nanos(100));
        let pool = Pool::builder()
            .threads(threads)
            .dispatch_costs(DispatchCosts::new(row, row, row))
            .build()
            .unwrap();
        let model = WorkModel::default()
            .with_muladds_per_ns(1.0)
            .with_item_overhead_ns(0.0)
            .with_min_work_per_lane_ns(lane_ns);
        let mut engine =
            CpuEngine::new(pool).with_batch_policy(BatchPolicy::default().with_model(model));
        let data = vec![0.0_f64; 10000];
        // Column-major strides: each the product of the extents before it.
        let strides: Vec<Vec<usize>> = shapes
            .iter()
            .map(|shape| {
                (0..shape.len())
                    .map(|axis| shape[..axis].iter().product())
                    .collect()
            })
            .collect();
        let mut jobs: Vec<Job<'_>> = shapes
            .iter()
            .zip(&strides)
            .map(|(&shape, strides)| {
                let len = shape.iter().product();
                let input =
                    DenseRead::F64(DenseView::new(&data[..len], shape, strides, 0).unwrap());
                Job {
                    input,
                    strides: preflight::<F>(input).unwrap(),
                    outcome: None,
                }
            })
            .collect();
        let op = RecordsWidth::<F>(
            Blocks(PhantomData),
            shapes.iter().map(|_| 0.into()).collect(),
        );
        driver::drive(&mut engine, &op, &mut jobs, &mut LaneTables::default()).unwrap();
        op.1.iter().map(|w| w.load(Ordering::Relaxed)).collect()
    }

    /// The lanes (share, jobs) `driver::place` gives factorization `F`'s
    /// owned blocks of `shapes` on `threads` slots of flat `row_ns` dispatch
    /// under `model` through `cache`, and the placement's comparison sorts.
    fn placed_in<F: Factor>(
        cache: &crate::lanes::LaneAllocationCache,
        threads: usize,
        row_ns: u64,
        model: WorkModel,
        shapes: &[&[usize]],
    ) -> (Vec<(usize, Vec<usize>)>, usize) {
        use tenferro_cpu::threadteam::{DispatchCosts, WidthCosts};
        let row = WidthCosts::flat(Duration::from_nanos(row_ns));
        let pool = Pool::builder()
            .threads(threads)
            .dispatch_costs(DispatchCosts::new(row, row, row))
            .build()
            .unwrap();
        let mut engine =
            CpuEngine::new(pool).with_batch_policy(BatchPolicy::default().with_model(model));
        let data = vec![0.0_f64; 10000];
        let strides: Vec<Vec<usize>> = shapes
            .iter()
            .map(|shape| {
                (0..shape.len())
                    .map(|axis| shape[..axis].iter().product())
                    .collect()
            })
            .collect();
        let jobs: Vec<Job<'_>> = shapes
            .iter()
            .zip(&strides)
            .map(|(&shape, strides)| {
                let len = shape.iter().product();
                let input =
                    DenseRead::F64(DenseView::new(&data[..len], shape, strides, 0).unwrap());
                Job {
                    input,
                    strides: preflight::<F>(input).unwrap(),
                    outcome: None,
                }
            })
            .collect();
        engine
            .with_linalg_pool(|exec, _| {
                let mut tables = LaneTables::default();
                let recording = crate::probe::record();
                driver::place(exec, &Blocks::<F>(PhantomData), &jobs, cache, &mut tables);
                let sorts = recording.get(crate::probe::Event::PlacementSort);
                let p = tables.placement();
                let lanes = (0..p.lanes())
                    .map(|l| (p.share(l), p.jobs(l).to_vec()))
                    .collect();
                Ok((lanes, sorts))
            })
            .unwrap()
    }

    /// [`placed_in`] through a cache of its own that admits nothing.
    fn placed<F: Factor>(
        threads: usize,
        row_ns: u64,
        model: WorkModel,
        shapes: &[&[usize]],
    ) -> (Vec<(usize, Vec<usize>)>, usize) {
        let uncached = crate::lanes::LaneAllocationCache::new(
            symtn_core::StructureCacheKind::LaneAllocations,
            0,
        );
        placed_in::<F>(&uncached, threads, row_ns, model, shapes)
    }

    /// Purpose: the batch decision caps the searched lane count. Four f64
    /// SVDs `10 x 10` on four slots (flat 100 ns dispatch, one multiply-add
    /// per ns, no overhead, a 2000 ns lane minimum, `min_items = 2`,
    /// `min_items_per_lane = 1`): 4000 ns admit four slots but the decision
    /// two lanes; no payback pays, so every grant is one; no equal share
    /// beats the baseline and the reservation's four lanes exceed the cap.
    /// The result is the baseline's two lanes of two slots, jobs `[0, 2]` and
    /// `[1, 3]`.
    #[test]
    fn the_batch_decision_caps_the_searched_lanes() {
        let model = WorkModel::default()
            .with_muladds_per_ns(1.0)
            .with_item_overhead_ns(0.0)
            .with_min_work_per_lane_ns(2000.0)
            .with_min_items(2)
            .with_min_items_per_lane(1);
        let (lanes, _) = placed::<Svd>(4, 100, model, &[&[10, 10][..]; 4]);
        assert_eq!(lanes, [(2, vec![0, 2]), (2, vec![1, 3])]);
    }

    /// Purpose: a one-lane placement returns before ordering its jobs, for
    /// both of main's one-lane reasons: a QR batch whose whole estimate pays
    /// for no width (one slot), and an SVD batch whose every block widens but
    /// whose `min_items` admits one lane. The blocks have distinct
    /// estimates in permuted order, `[2 + pi(j), 1]` and `[128 + pi(j), 1]`.
    #[test]
    fn a_one_lane_placement_sorts_nothing() {
        let pi = [5, 1, 7, 3, 8, 2, 6, 4];
        let qr: Vec<[usize; 2]> = pi.iter().map(|&p| [2 + p, 1]).collect();
        let qr: Vec<&[usize]> = qr.iter().map(|s| &s[..]).collect();
        let unit = WorkModel::default()
            .with_muladds_per_ns(1.0)
            .with_item_overhead_ns(0.0);
        let (lanes, sorts) = placed::<Qr>(2, 1_000_000_000, unit, &qr);
        assert_eq!((lanes, sorts), (vec![(1, (0..8).collect())], 0));

        let svd: Vec<[usize; 2]> = pi.iter().map(|&p| [128 + p, 1]).collect();
        let svd: Vec<&[usize]> = svd.iter().map(|s| &s[..]).collect();
        let gated = WorkModel::default()
            .with_muladds_per_ns(1e-6)
            .with_item_overhead_ns(0.0)
            .with_min_items(usize::MAX);
        let (lanes, sorts) = placed::<Svd>(2, 1, gated, &svd);
        assert_eq!((lanes, sorts), (vec![(2, (0..8).collect())], 0));
    }

    /// Purpose: a searched factorization batch is admitted to the
    /// lane-allocation cache (here a cache of the test's own), and the
    /// repeat, a hit, equals the cold search and sorts nothing: the
    /// review-r3 SVD trace, main's SVD mix (`80 x 80` and three `4 x 4`) and
    /// the same mix of general eigenvalues. The first call of each misses and
    /// the second hits.
    #[test]
    fn a_repeated_searched_batch_hits_its_cold_placement() {
        let capped = WorkModel::default()
            .with_muladds_per_ns(1.0)
            .with_item_overhead_ns(0.0)
            .with_min_work_per_lane_ns(2000.0)
            .with_min_items(2)
            .with_min_items_per_lane(1);
        let unit = WorkModel::default()
            .with_muladds_per_ns(1.0)
            .with_item_overhead_ns(0.0);
        let small: &[usize] = &[4, 4];
        let mix: &[&[usize]] = &[&[80, 80], small, small, small];
        type Placed = (Vec<(usize, Vec<usize>)>, usize);
        let twice = |placed: &dyn Fn(&crate::lanes::LaneAllocationCache) -> Placed| {
            let cache = crate::lanes::LaneAllocationCache::new(
                symtn_core::StructureCacheKind::LaneAllocations,
                1 << 20,
            );
            let (cold, _) = placed(&cache);
            let info = cache.info();
            assert_eq!((info.hits(), info.misses(), info.admissions()), (0, 1, 1));
            let (warm, sorts) = placed(&cache);
            assert_eq!((&warm, sorts), (&cold, 0));
            assert_eq!(cache.info().hits(), 1);
            cold
        };
        let shapes = [&[10, 10][..]; 4];
        let r3 = twice(&|cache| placed_in::<Svd>(cache, 4, 100, capped, &shapes));
        assert_eq!(r3, [(2, vec![0, 2]), (2, vec![1, 3])]);
        let svd = twice(&|cache| placed_in::<Svd>(cache, 4, 100, unit, mix));
        assert_eq!(svd, [(4, vec![0, 1, 2, 3])]);
        let eig = twice(&|cache| placed_in::<EigVals>(cache, 4, 100, unit, mix));
        assert_eq!(eig, [(4, vec![0, 1, 2, 3])]);
    }

    /// Purpose: each lane of a destination batch is set up once, before its
    /// first block, with a workspace sufficient for every block it runs:
    /// every block of a lane runs with that one setup, whose workspace is
    /// never regrown, and the setup and its release happen on the thread
    /// that runs the lane's solves, on uneven blocks over two forced lanes
    /// and on the single-lane path, including a two-item and an empty-batch
    /// input; every block succeeds, so no workspace was short.
    #[test]
    fn each_lane_is_set_up_once_for_all_its_blocks() {
        use crate::factor_dest::validate;
        use std::collections::BTreeMap;
        use tcontract_exec::BatchStrategy;
        let shapes: [&[usize]; 5] = [&[40, 30], &[3, 2], &[17, 9], &[6, 5, 2], &[4, 3, 0]];
        let data = vec![0.25_f64; 2400];
        let strides: Vec<Vec<usize>> = shapes
            .iter()
            .map(|shape| {
                (0..shape.len())
                    .map(|axis| shape[..axis].iter().product())
                    .collect()
            })
            .collect();
        let inputs: Vec<DenseRead<'_>> = shapes
            .iter()
            .zip(&strides)
            .map(|(&shape, strides)| {
                let len = shape.iter().product();
                DenseRead::F64(DenseView::new(&data[..len], shape, strides, 0).unwrap())
            })
            .collect();
        let engines = [
            CpuEngine::serial(),
            CpuEngine::new(Pool::builder().threads(4).build().unwrap()).with_batch_policy(
                BatchPolicy::new(BatchStrategy::Split {
                    lanes: std::num::NonZeroUsize::new(2).unwrap(),
                }),
            ),
        ];
        for (mut engine, lanes) in engines.into_iter().zip([1, 2]) {
            let geometry: Vec<_> = shapes
                .iter()
                .map(|shape| {
                    let (m, n) = (shape[0], shape[1]);
                    let k = m.min(n);
                    let batch = &shape[2..];
                    let q = [&[m, k][..], batch].concat();
                    let r = [&[k, n][..], batch].concat();
                    let q_strides: Vec<usize> =
                        (0..q.len()).map(|a| q[..a].iter().product()).collect();
                    let r_strides: Vec<usize> =
                        (0..r.len()).map(|a| r[..a].iter().product()).collect();
                    (q, q_strides, r, r_strides)
                })
                .collect();
            let mut storage: Vec<(Vec<f64>, Vec<f64>)> = geometry
                .iter()
                .map(|(q, _, r, _)| (vec![0.0; q.iter().product()], vec![0.0; r.iter().product()]))
                .collect();
            let mut dests: Vec<FactorDest<'_>> = storage
                .iter_mut()
                .zip(&geometry)
                .map(|((q, r), (qs, qst, rs, rst))| {
                    FactorDest::new(
                        [
                            Some(DenseWrite::F64(
                                crate::DenseViewMut::new(q, qs, qst, 0).unwrap(),
                            )),
                            Some(DenseWrite::F64(
                                crate::DenseViewMut::new(r, rs, rst, 0).unwrap(),
                            )),
                            None,
                        ],
                        FactorPlacement::AsComputed,
                    )
                })
                .collect();
            validate(crate::DenseFactorization::Qr, &inputs, &dests).unwrap();
            let mut jobs: Vec<IntoJob<'_, '_, '_>> = inputs
                .iter()
                .zip(dests.iter_mut())
                .map(|(&input, dest)| IntoJob {
                    input,
                    strides: preflight::<Qr>(input).unwrap(),
                    dest,
                    plans: [None, None],
                    sizing: None,
                    outcome: None,
                    lane_seen: None,
                })
                .collect();
            driver::drive(
                &mut engine,
                &IntoBlocks::<Qr>(PhantomData),
                &mut jobs,
                &mut LaneTables::default(),
            )
            .unwrap();
            let mut setups = BTreeMap::new();
            for job in &jobs {
                assert_eq!(job.outcome, Some(Ok(())));
                let (id, len, setup, solve) = job.lane_seen.unwrap();
                assert_eq!(setup, solve, "set up on the thread of its solves");
                assert_eq!(
                    *setups.entry(id).or_insert((len, setup)),
                    (len, setup),
                    "one buffer per setup"
                );
            }
            assert_eq!(setups.len(), lanes, "{lanes} lane(s)");
            for (&id, &(_, setup)) in &setups {
                assert_eq!(owners::releases_of(id), [setup], "freed once, there");
            }
        }
    }

    /// Purpose: a factorization batch whose every input is one matrix joins
    /// the lane search with tlinalg's own width for one SVD item, while a
    /// batch holding an input whose batch product is not one (including
    /// zero) keeps the baseline placement. Four slots, flat 100 ns dispatch (a width pays
    /// from 300 ns), one multiply-add per ns:
    /// - `80 x 80` and three `4 x 4`: the large SVD's payback work is
    ///   `80^3 * 6e-4 = 307.2` ns, so it is granted every width; one
    ///   four-slot lane scores `512000 / 4 + 100 + 3 * 64 = 128292` ns
    ///   against the baseline's 512000 ns on four one-slot lanes;
    /// - `79 x 79` instead: payback `79^3 * 6e-4 = 295.8` ns pays at no width
    ///   (its whole time, 493039 ns, would), so every grant is one, no
    ///   candidate scores below the baseline's 493039 ns, and the tie keeps
    ///   the baseline;
    /// - the first batch with one `4 x 4 x 2` input keeps the baseline.
    ///
    /// On six slots with a 10000 ns lane minimum, one `10000 x 1` and 120
    /// `100 x 1` (payback 6 and 0.06 ns: serial at every width) give `L0 = 2`
    /// lanes; the baseline and two equal lanes score 11000 ns, the
    /// reservation's six one-slot lanes 10000 ns but hold more than `L0`
    /// lanes and are discarded, so the baseline's two three-slot lanes run
    /// every block.
    #[test]
    fn single_matrix_factorizations_join_the_lane_search() {
        let small: &[usize] = &[4, 4];
        assert_eq!(
            lane_widths::<Svd>(4, 0.0, &[&[80, 80], small, small, small]),
            [4; 4]
        );
        assert_eq!(
            lane_widths::<Svd>(4, 0.0, &[&[79, 79], small, small, small]),
            [1; 4]
        );
        assert_eq!(
            lane_widths::<Svd>(4, 0.0, &[&[80, 80], small, small, &[4, 4, 2]]),
            [1; 4]
        );
        let mut many: Vec<&[usize]> = vec![&[10000, 1]];
        many.extend([&[100, 1][..]; 120]);
        assert_eq!(lane_widths::<Svd>(6, 10000.0, &many), [3; 121]);
        // General eigenvalues: their whole estimate is their payback, so the
        // large block is granted every width and one four-slot lane wins as
        // for the SVD.
        assert_eq!(
            lane_widths::<EigVals>(4, 0.0, &[&[80, 80], small, small, small]),
            [4; 4]
        );
    }

    /// Four slots of flat 100 ns dispatch at one multiply-add per ns and no
    /// item overhead (a stage pays from 300 ns).
    fn flat_engine(threads: usize) -> CpuEngine {
        use tenferro_cpu::threadteam::{DispatchCosts, WidthCosts};
        let row = WidthCosts::flat(Duration::from_nanos(100));
        let pool = Pool::builder()
            .threads(threads)
            .dispatch_costs(DispatchCosts::new(row, row, row))
            .build()
            .unwrap();
        let model = WorkModel::default()
            .with_muladds_per_ns(1.0)
            .with_item_overhead_ns(0.0);
        CpuEngine::new(pool).with_batch_policy(BatchPolicy::default().with_model(model))
    }

    /// Destination jobs of `F` on `inputs`, their plans resolved as the
    /// batch resolves them.
    fn into_jobs<'a, 'd, 'w, F: FactorInto>(
        inputs: &[DenseRead<'a>],
        dests: &'d mut [FactorDest<'w>],
    ) -> Vec<IntoJob<'a, 'd, 'w>> {
        let mut jobs: Vec<_> = inputs
            .iter()
            .zip(dests)
            .map(|(&input, dest)| IntoJob {
                input,
                strides: preflight::<F>(input).unwrap(),
                dest,
                plans: [None, None],
                sizing: None,
                outcome: None,
                lane_seen: None,
            })
            .collect();
        resolve_plans::<F>(&mut jobs);
        jobs
    }

    /// One composite job of `F` on `input` as the lane model sees it on
    /// [`flat_engine`]: its serial time, its stage widths and modeled time on
    /// a four-slot lane, and its stage widths on a one-slot lane.
    fn composite_model<F: FactorInto>(input: DenseRead<'_>) -> (Duration, Grant, Duration, Grant) {
        let mut engine = flat_engine(4);
        let model = engine.batch_policy().model;
        let mut dests = [FactorDest::new(
            [None, None, None],
            FactorPlacement::AsComputed,
        )];
        let jobs = into_jobs::<F>(&[input], &mut dests);
        let (op, job) = (IntoBlocks::<F>(PhantomData), &jobs[0]);
        engine
            .with_linalg_pool(|exec, _| {
                let serial = op.serial(0, job, &model, op.work(0, job));
                let wide = op.grant(0, job, exec, 4, serial);
                let item = op.item(0, job, exec, serial, wide);
                Ok((serial, wide, item, op.grant(0, job, exec, 1, serial)))
            })
            .unwrap()
    }

    /// Purpose: a composite job's lane model is the sum of its sequential
    /// stages, each priced by its own provider at its own width: the
    /// factorization at the work model's rate with tlinalg's width for its
    /// own estimates, each product at tcontract's GEMM rate with its resolved
    /// plan's width, each parallel stage paying its own dispatch, the serial
    /// time the stages' sum, and a job whose every stage is serial costing
    /// that sum. On [`flat_engine`] a factorization of total work `t` and
    /// payback `p` takes `t` ns and every slot when `p >= 300`, and a
    /// product of weighted work `w` takes `2 w ns_per_flop` ns and every
    /// slot when that time is at least 300 ns:
    /// - `exp` f64 `60 x 60`: eigh `60^3 = 216000`, payback `129.6`
    ///   (serial); product `Y Y^H` of work `216000` (parallel);
    /// - `exp` c64 `60 x 60`: eigh `4 * 216000`, payback `518.4` (parallel);
    ///   the product four times the f64 work (parallel), two dispatches;
    /// - left polar f64 `200 x 16`: SVD `200 * 16 * 16 = 51200`, payback
    ///   `30.72` (serial); `W = U Vh` of work `200 * 16 * 16` (parallel);
    ///   `P = X^H X` of work `16^3` (serial at either host rate);
    /// - right polar c64 `16 x 200`: SVD `4 * 51200`, payback `122.88`
    ///   (serial); `W` of work `4 * 16 * 16 * 200` and `P = X X^H` of work
    ///   `4 * 16^3` (both parallel), where the left orientation's `n x n`
    ///   product would be `4 * 200 * 16 * 200`;
    /// - `exp` f64 `8 x 8`: eigh `512`, product `512` (both serial).
    ///
    /// On a one-slot lane every stage is serial.
    #[test]
    fn composite_jobs_cost_each_stage_at_its_own_width() {
        let npf = tcontract::ns_per_flop();
        let ns = |x: f64| Duration::from_secs_f64(x * 1e-9);
        let gemm = |work: f64| ns(2.0 * work * npf);
        let d = Duration::from_nanos(100);
        let real = vec![0.0_f64; 3600];
        let complex = vec![Complex64::new(0.0, 0.0); 3600];
        let f64_of = |shape: &'static [usize], strides: &'static [usize]| {
            DenseRead::F64(DenseView::new(&real[..shape[0] * shape[1]], shape, strides, 0).unwrap())
        };
        let c64_of = |shape: &'static [usize], strides: &'static [usize]| {
            DenseRead::C64(
                DenseView::new(&complex[..shape[0] * shape[1]], shape, strides, 0).unwrap(),
            )
        };
        let rows = [
            (
                composite_model::<ExpViaEigh>(f64_of(&[60, 60], &[1, 60])),
                (
                    ns(216000.0) + gemm(216000.0),
                    [1, 4, 1],
                    ns(216000.0) + gemm(216000.0) / 4 + d,
                ),
            ),
            (
                composite_model::<ExpViaEigh>(c64_of(&[60, 60], &[1, 60])),
                (
                    ns(864000.0) + gemm(864000.0),
                    [4, 4, 1],
                    ns(864000.0) / 4 + d + gemm(864000.0) / 4 + d,
                ),
            ),
            (
                composite_model::<Polar<true>>(f64_of(&[200, 16], &[1, 200])),
                (
                    ns(51200.0) + gemm(51200.0) + gemm(4096.0),
                    [1, 4, 1],
                    ns(51200.0) + gemm(51200.0) / 4 + d + gemm(4096.0),
                ),
            ),
            (
                composite_model::<Polar<false>>(c64_of(&[16, 200], &[1, 16])),
                (
                    ns(204800.0) + gemm(204800.0) + gemm(16384.0),
                    [1, 4, 4],
                    ns(204800.0) + gemm(204800.0) / 4 + d + gemm(16384.0) / 4 + d,
                ),
            ),
            (
                composite_model::<ExpViaEigh>(f64_of(&[8, 8], &[1, 8])),
                (ns(512.0) + gemm(512.0), [1, 1, 1], ns(512.0) + gemm(512.0)),
            ),
        ];
        for (row, ((serial, wide, item, narrow), expected)) in rows.into_iter().enumerate() {
            assert_eq!((serial, wide, item), expected, "row {row}");
            assert_eq!(narrow, [1; STAGES], "row {row}");
        }
    }

    /// Purpose: the products a composite job's stages price are the ones
    /// its geometries run, `rows * contracted * cols` each.
    #[test]
    fn stage_works_are_the_geometries_products() {
        fn check<F: FactorInto>(dims: [usize; 2]) {
            let (_, works) = F::stage_works::<f64>(dims[0], dims[1]).unwrap();
            let geometries = F::geometries(dims).unwrap();
            for (work, geometry) in works.iter().zip(geometries) {
                let expected = geometry.map_or(0.0, |g| g.mkn.iter().map(|&e| e as f64).product());
                assert_eq!(*work, expected, "{dims:?}");
            }
        }
        for dims in [[7, 3], [40, 40], [5, 1]] {
            check::<Polar<true>>(dims);
        }
        for dims in [[3, 7], [40, 40], [1, 5]] {
            check::<Polar<false>>(dims);
        }
        for dims in [[9, 9], [1, 1]] {
            check::<ExpViaEigh>(dims);
        }
    }

    /// The `[m, n]` destination output extents of one input of
    /// factorization `op` (`exp`, left or right polar).
    fn composite_outputs(op: crate::DenseFactorization, [m, n]: [usize; 2]) -> Vec<[usize; 2]> {
        use crate::DenseFactorization as Op;
        match op {
            Op::ExpViaEigh => vec![[m, m]],
            Op::LeftPolar => vec![[m, n], [n, n]],
            Op::RightPolar => vec![[m, m], [m, n]],
            op => unreachable!("{op:?} is not composite"),
        }
    }

    /// Runs a destination batch of composite `op` (factorization `F`) on
    /// zero f64 inputs of matrix extents `dims`, all held in one buffer, on
    /// [`flat_engine`]`(4)`: through the native destination entry
    /// [`factorize_into`] under its Auto policy (`entry`), or through the lane
    /// driver on jobs planned as that entry plans them. Returns, from the
    /// driver, the placement (each lane's share and jobs), after checking
    /// that the jobs of one lane, and only those, shared its lane setup; from
    /// the entry, the jobs each lane setup solved (the owner records of the
    /// inputs' payloads). Every job must succeed.
    fn composite_lanes<F: FactorInto>(
        op: crate::DenseFactorization,
        dims: &[[usize; 2]],
        entry: bool,
    ) -> Vec<(usize, Vec<usize>)> {
        let starts: Vec<usize> = dims
            .iter()
            .scan(0, |at, [m, n]| {
                let start = *at;
                *at += m * n;
                Some(start)
            })
            .collect();
        let data = vec![0.0_f64; dims.iter().map(|[m, n]| m * n).sum::<usize>()];
        let strides: Vec<[usize; 2]> = dims.iter().map(|&[m, _]| [1, m]).collect();
        let inputs: Vec<DenseRead<'_>> = (0..dims.len())
            .map(|j| {
                let view = DenseView::new(&data, &dims[j], &strides[j], starts[j]).unwrap();
                DenseRead::F64(view)
            })
            .collect();
        let shapes: Vec<Vec<[usize; 2]>> = dims.iter().map(|&d| composite_outputs(op, d)).collect();
        let out_strides: Vec<Vec<[usize; 2]>> = shapes
            .iter()
            .map(|outs| outs.iter().map(|&[m, _]| [1, m]).collect())
            .collect();
        let mut outputs: Vec<Vec<Vec<f64>>> = shapes
            .iter()
            .map(|outs| outs.iter().map(|[m, n]| vec![0.0; m * n]).collect())
            .collect();
        let mut dests: Vec<FactorDest<'_>> = outputs
            .iter_mut()
            .enumerate()
            .map(|(j, outs)| {
                let mut slots: [Option<DenseWrite<'_>>; 3] = [None, None, None];
                for (k, out) in outs.iter_mut().enumerate() {
                    let view = crate::DenseViewMut::new(out, &shapes[j][k], &out_strides[j][k], 0)
                        .unwrap();
                    slots[k] = Some(DenseWrite::F64(view));
                }
                FactorDest::new(slots, FactorPlacement::AsComputed)
            })
            .collect();
        let mut engine = flat_engine(4);
        if entry {
            let base = data.as_ptr() as usize;
            let range = base..base + data.len() * std::mem::size_of::<f64>();
            let mark = owners::mark();
            factorize_into(
                &mut engine,
                op,
                &inputs,
                &mut dests,
                &mut |_, _, outcome| {
                    assert_eq!(outcome, Ok(()));
                    Ok(())
                },
            )
            .unwrap();
            let mut lanes = std::collections::BTreeMap::<usize, Vec<usize>>::new();
            for record in owners::within(mark, range) {
                let offset = (record.input - base) / std::mem::size_of::<f64>();
                let job = starts.iter().position(|&start| start == offset).unwrap();
                lanes.entry(record.lane).or_default().push(job);
            }
            let mut groups: Vec<(usize, Vec<usize>)> = lanes
                .into_values()
                .map(|mut jobs| {
                    jobs.sort_unstable();
                    (0, jobs)
                })
                .collect();
            groups.sort();
            assert_eq!(
                groups.iter().map(|(_, jobs)| jobs.len()).sum::<usize>(),
                dims.len()
            );
            return groups;
        }
        let mut jobs = into_jobs::<F>(&inputs, &mut dests);
        let mut tables = LaneTables::default();
        driver::drive(
            &mut engine,
            &IntoBlocks::<F>(PhantomData),
            &mut jobs,
            &mut tables,
        )
        .unwrap();
        let placement = tables.placement();
        let lanes: Vec<(usize, Vec<usize>)> = (0..placement.lanes())
            .map(|l| (placement.share(l), placement.jobs(l).to_vec()))
            .collect();
        let mut setups = std::collections::BTreeSet::new();
        for (_, members) in &lanes {
            let setup = |j: usize| jobs[j].lane_seen.unwrap().0;
            assert!(members.iter().all(|&j| setup(j) == setup(members[0])));
            assert!(setups.insert(setup(members[0])), "one setup per lane");
        }
        assert!(jobs.iter().all(|job| job.outcome == Some(Ok(()))));
        lanes
    }

    /// Purpose: a destination composite batch is placed under its per-stage
    /// model by every candidate that can win, through the lane driver and
    /// through the native destination entry under Auto: the equal shares,
    /// the shared reservation and the exclusive one each decide one of these
    /// batches, for the Hermitian exponential and both polar orientations,
    /// and each layout differs from the whole-job model's (every job on one
    /// four-slot lane) and from the one the deciding candidate picks when it
    /// prices a job as its serial time over its factorization's grant (stage
    /// pricing lost at that scoring site).
    ///
    /// Four slots of flat 100 ns dispatch (a stage widens to its whole lane
    /// from 300 ns of work time), one multiply-add per ns. On `w` slots a
    /// stage of `t` ns costs `t / w + 100` when it widens, `t` otherwise.
    /// An `exp` of order `n` is an eigh of `n^3` ns, granted the lane only
    /// when its payback `6e-4 n^3` reaches 300 ns (`n >= 80`), then a product
    /// of `P_n = 2 n^3 p` ns at tcontract's `p` ns per flop, `p` in `[0.0185,
    /// 0.032]`. A left polar of a tall `m x n` (a right polar of its
    /// transpose prices the same) is an SVD of `m n^2` ns, granted the lane
    /// only when its payback `6e-4 m n^2` reaches 300 ns, then `W` of `2 m
    /// n^2 p` ns and `P` of `2 n^3 p` ns. Every product here is above 300 ns.
    /// Scores are quoted at `p = 0.0185` / `0.032`:
    /// - `exp [120, 80, 80, 79]`: two two-slot lanes, `[0]` and `[1, 2, 3]`,
    ///   score `2 (256200 + P_80 / 2) + 493039 + P_79 / 2 + 100 = 1033604 /
    ///   1054084`, against three lanes `[0] [1, 3] [2]` at `1042225 /
    ///   1069361` (`P_79 + 493039` and an `80` on one slot) and both
    ///   reservations at the same three-lane score. Pricing each job as its
    ///   serial time over its eigh's grant (the `79` as `493039 + P_79` on
    ///   the two-slot lane, its eigh's grant being one) raises the two-lane
    ///   score to `1042425 / 1069561` and the three lanes win.
    /// - `exp [120, 80, 40, 40]`: the shared reservation, `3: [0, 3]` and
    ///   `1: [1, 2]`, score `576100 + P_120 / 3 + 100 + 64000 + P_40 / 3 +
    ///   100 = 662401 / 678529` (the `40`'s eigh stays serial), against one
    ///   four-slot lane at `710504 / 726488`; the exclusive one puts both
    ///   `40`s beside the `80`. Priced as its serial time on the reserved
    ///   lane, the `40` no longer joins it: `3: [0]`, `1: [1, 2, 3]`.
    /// - `exp [79, 70]`: the exclusive reservation, `3: [0]` and `1: [1]`,
    ///   score `493039 + P_79 / 3 + 100 = 499219 / 503657`, against the two
    ///   two-slot lanes at `493039 + P_79 / 2 + 100 = 502260 / 508916`; the
    ///   shared one reserves all four slots and puts the `70` behind the `79`
    ///   (`843971 / 849615`). Priced as its serial time on its reserved lane,
    ///   the `79` gains nothing there and the two two-slot lanes win.
    /// - left polar `[120 x 60, 60 x 30, 60 x 30]` and right polar of the
    ///   transposes: the exclusive reservation, `3: [0]` and `1: [1, 2]`,
    ///   score `432000 + 2 * 432000 p / 3 + 100 + 2 * 216000 p / 3 + 100 =
    ///   440192 / 446024` (the `120 x 60`'s SVD payback, `259.2` ns, keeps
    ///   it serial), against the baseline's `2: [0]`, `1: [1]`, `1: [2]` at
    ///   `444188 / 452936`; the shared one reserves every slot (`548090 /
    ///   553560`).
    ///
    /// Through the entry the lanes' shares are not observed; the jobs each
    /// lane solved are.
    #[test]
    fn composite_batches_place_under_the_stage_model() {
        use crate::DenseFactorization as Op;
        let square = |ns: &[usize]| ns.iter().map(|&n| [n, n]).collect::<Vec<_>>();
        let tall = [[120, 60], [60, 30], [60, 30]];
        let wide = tall.map(|[m, n]| [n, m]);
        let polar = vec![(3, vec![0]), (1, vec![1, 2])];
        type Lanes = Vec<(usize, Vec<usize>)>;
        let cases: [(Op, Vec<[usize; 2]>, Lanes); 5] = [
            (
                Op::ExpViaEigh,
                square(&[120, 80, 80, 79]),
                vec![(2, vec![0]), (2, vec![1, 2, 3])],
            ),
            (
                Op::ExpViaEigh,
                square(&[120, 80, 40, 40]),
                vec![(3, vec![0, 3]), (1, vec![1, 2])],
            ),
            (
                Op::ExpViaEigh,
                square(&[79, 70]),
                vec![(3, vec![0]), (1, vec![1])],
            ),
            (Op::LeftPolar, tall.to_vec(), polar.clone()),
            (Op::RightPolar, wide.to_vec(), polar),
        ];
        for (op, dims, expected) in cases {
            let run = |entry| match op {
                Op::ExpViaEigh => composite_lanes::<ExpViaEigh>(op, &dims, entry),
                Op::LeftPolar => composite_lanes::<Polar<true>>(op, &dims, entry),
                _ => composite_lanes::<Polar<false>>(op, &dims, entry),
            };
            assert_eq!(run(false), expected, "{op:?} {dims:?} through the driver");
            let mut groups: Vec<(usize, Vec<usize>)> =
                expected.iter().map(|(_, jobs)| (0, jobs.clone())).collect();
            groups.sort();
            assert_eq!(run(true), groups, "{op:?} {dims:?} through the entry");
            assert!(groups.len() > 1, "the whole-job model's one lane");
        }
    }

    /// Counts its `serial` calls; otherwise a modeled single-stage op.
    struct CountsSerial(AtomicUsize);

    impl BlockOp for CountsSerial {
        type Job = ();
        type Lane = ();
        const ENTRY: Entry = Entry::Linalg;
        const KIND: OpKind = OpKind::Factorization;
        const LABEL: &'static str = "counts_serial";

        fn work(&self, _: usize, _: &()) -> f64 {
            1e6
        }

        fn serial(&self, _: usize, _: &(), model: &WorkModel, work: f64) -> Duration {
            self.0.fetch_add(1, Ordering::Relaxed);
            driver::cost(Self::KIND, work, model)
        }

        fn grant(&self, _: usize, _: &(), _: &Exec<'_>, _: usize, _: Duration) -> Grant {
            [1; STAGES]
        }

        fn grant_key(&self, _: usize, _: &()) -> GrantKey {
            [0; 3]
        }

        fn lane(&self, _: &mut LaneMembers<'_, ()>, _: &Exec<'_>) {}

        fn run(&self, _: usize, _: &mut (), (): &mut (), _: Exec<'_>) -> Result<(), DenseError> {
            Ok(())
        }
    }

    /// Purpose: on a context of one slot the driver prices blocks by the
    /// kind's cost alone and never evaluates a block's stage pricing; on a
    /// wider context it prices every block through it.
    #[test]
    fn one_slot_contexts_skip_stage_pricing() {
        use crate::lanes::LaneAllocationCache;
        use symtn_core::StructureCacheKind;
        for (mut engine, priced) in [(CpuEngine::serial(), 0), (flat_engine(2), 5)] {
            let op = CountsSerial(AtomicUsize::new(0));
            let cache = LaneAllocationCache::new(StructureCacheKind::LaneAllocations, 0);
            let mut tables = LaneTables::default();
            engine
                .with_linalg_pool(|exec, _| {
                    driver::place(exec, &op, &[(); 5], &cache, &mut tables);
                    Ok(())
                })
                .unwrap();
            assert_eq!(op.0.load(Ordering::Relaxed), priced);
        }
    }
}
