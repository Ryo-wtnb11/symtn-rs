use super::*;

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R> + TypedTensorTransformDispatch<R, D>,
    D: AdvancedLinalgScalar,
{
    /// Solves `self * x = rhs` independently in every coupled sector, without
    /// forming an inverse.
    ///
    /// `rows` and `cols` select the receiver's matrix view, while
    /// `rhs_rows` and `rhs_cols` select the right-hand side's matrix view.
    /// Each pair names source axes in [`Self::permute`] order. The current
    /// split borrows its operand without a transform.
    ///
    /// The permuted operands must share a runtime and fusion-rule identity,
    /// their codomains must be exactly equal, and `self` must have isomorphic
    /// codomain and domain. The result has space
    /// `domain(permuted self) <- domain(permuted rhs)` and keeps `self`'s exact provider `Arc`.
    /// This is TensorKit's left solve `self \\ rhs`.
    /// TensorKit's right solve `self / rhs` can be composed from adjoints
    /// and this left solve with roles chosen in the adjointed axis order.
    ///
    /// Dense input uses one linear solve per coupled sector. A compact
    /// diagonal divisor instead applies its elementwise reciprocal (TensorKit
    /// `D \ t`): a compact right-hand side on the same bond gives a compact
    /// quotient on the divisor's space, and any other one — dense, in any
    /// layout, or a lazy adjoint read in place — lands in the output space
    /// with its leading (bond) axis scaled, `O(Σ_c k_c m_c)` with no LU, in
    /// both fusion modes. Every compact divisor, in any bond layout, is
    /// admitted through one shared structural check. Nonfinite divisor
    /// entries map through the reciprocal. An exact zero divisor entry is the
    /// dense route's singular-block operation error in every mode. A compact
    /// right-hand side of a dense divisor is densified into the solve buffer.
    /// On the dense route, lazy adjoints are materialized only for this call.
    ///
    /// # Errors
    ///
    /// In one order in every fusion mode (#1995), as TensorKit's `\` checks:
    /// [`Error::RuntimeMismatch`], then [`Error::RuleMismatch`] for
    /// incompatible operands, then an operation `SpaceMismatch` for unequal
    /// codomains, then one when the divisor is not isomorphic,
    /// then the borrowed-view refusal below; an operation
    /// `Dense(NumericalFailure)` when a sector is singular comes after all of
    /// them. If a checked provider rejects the output space,
    /// its original error is available as the source. If any preflight, sector
    /// solve, or output-space creation fails, no result tensor is returned.
    /// Non-identity roles also return the existing [`Self::permute`] errors
    /// for malformed axes or unsupported braiding.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use symtn::sector::{U1FusionRule, U1Irrep};
    /// use symtn::typed::{GradedSpace, Runtime, TensorMap};
    ///
    /// let engine = &mut symtn::typed::CpuEngine::serial();
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
    /// let a: TensorMap<_, f64> = TensorMap::isomorphism(engine, &runtime, [&v], [&v])?;
    /// let b: TensorMap<_, f64> = TensorMap::rand_with_seed(engine, &runtime, [&v], [&v], 1)?;
    /// assert_eq!(a.solve(engine, &[0], &[1], &b, &[0], &[1])?.dense_data()?, b.dense_data()?);
    /// # Ok::<(), symtn::typed::Error>(())
    /// ```
    ///
    /// An adjoint view `rhs` (`t.adjoint_view()`) is accepted by the current
    /// split of a compact divisor, which reads it directly, in both fusion
    /// modes. A dense divisor or either moved role pair would copy the view
    /// and returns [`Error::Unsupported`]. Pass `&t.adjoint()?.materialize()?`
    /// instead. Moved roles refuse the view after the runtime and rule checks
    /// (which precede the leg roles' own [`Self::permute`] errors) but before
    /// the shape checks, which need the permuted operands.
    pub fn solve<'a>(
        &self,
        engine: &mut dyn Engine,
        rows: &[usize],
        cols: &[usize],
        rhs: impl Into<TensorRef<'a, R, D>>,
        rhs_rows: &[usize],
        rhs_cols: &[usize],
    ) -> Result<Self, FacadeError<R>> {
        let rhs = rhs.into().operand()?;
        let rhs = &*rhs;
        // Runtime and rule first, ahead of the leg roles: the permute cannot
        // change them, so the first error does not depend on the roles or on
        // the rhs's ownership. Permuting a borrowed view would copy it, so
        // moved roles refuse it next; the shape checks need the permuted
        // operands. A moved compact divisor also becomes dense.
        self.require_solve_operands(rhs)?;
        if !self.axes_are_identity(rows, cols) || !rhs.axes_are_identity(rhs_rows, rhs_cols) {
            rhs.refuse_borrowed_view("solve")?;
        }
        self.with_leg_roles(engine, rows, cols, |lhs, engine| {
            rhs.with_leg_roles(engine, rhs_rows, rhs_cols, |right, engine| {
                lhs.factor_solve(engine, right)
            })
        })
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R> + TypedAdjointSpace<R> + TypedTensorTransformDispatch<R, D>,
    D: AdvancedLinalgScalar,
{
    /// TensorKit 0.17 / MatrixAlgebraKit `pinv`: the Moore-Penrose
    /// thresholded pseudo-inverse `t⁺ = V S⁺ Uᴴ`, where `t = U S Vᴴ` is the compact SVD and
    /// `S⁺` inverts every singular value above the cutoff and sends the rest to
    /// zero. This is the exact Moore-Penrose inverse of the hard-thresholded
    /// effective-rank tensor `t_r`. It is the Moore-Penrose inverse of `t`
    /// itself only when no genuinely nonzero singular value is discarded, and
    /// then satisfies `t t⁺ t = t`. It reduces to [`Self::inv`] when `t` is
    /// nonsingular and `rcond` keeps every singular value.
    ///
    /// # Tolerance, and the divergence from TensorKit
    ///
    /// The cutoff is `rcond * σ_max` with **one global `σ_max` taken across all
    /// coupled sectors**, and the comparison is strict: a singular value
    /// sitting exactly on the cutoff is discarded. TensorKit instead takes
    /// per-block `atol`/`rtol` keywords, so its relative tolerance is measured
    /// against each block's own largest singular value. That is a deliberate
    /// divergence, not a gap: a per-block relative tolerance cannot cut
    /// anything in a one-dimensional sector however small that sector's
    /// contribution to the tensor is. TensorKit's `DiagonalTensorMap` branch
    /// is deliberately **not** mirrored either: there `rtol` is ignored
    /// whenever `atol` is nonzero, the default is no cutoff at all, and its
    /// comparison (`abs(x) < tol` discards) *keeps* a value sitting exactly on
    /// the cutoff — the opposite of the strict `>` above.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidArgument`] when `rcond` is not finite or is negative,
    ///   checked before any provider work or dense allocation.
    /// - [`Error::Operation`] (an `InvalidArgument`: `pinv singular values
    ///   must be finite`) for a nonfinite entry on dense or compact input, in
    ///   every fusion mode and on every dense provider; the dense route
    ///   refuses it before any provider SVD.
    /// - [`Error::Operation`] / [`Error::Core`] from dense SVD or recomposition.
    ///
    /// There is no singular-input failure: sending the offending directions to
    /// zero is what a pseudo-inverse is for.
    ///
    /// # Complexity and storage
    ///
    /// Dense input uses one compact SVD per nonempty coupled sector,
    /// `O(Σ_c n_c³)`, then folds `S⁺` into a column scaling and recomposes with
    /// one local GEMM. A Host compact diagonal input stays compact in every
    /// fusion mode and bond layout (TensorKit `pinv(::DiagonalTensorMap)`
    /// keeps `d.domain`): its magnitudes are compared unrounded, in `f64`,
    /// against the same global cutoff, and a retained subnormal inverts to its
    /// IEEE reciprocal (`Inf`). The dense route compares singular values at
    /// the payload's precision, so for `f32`/`Complex32` a value whose rounded
    /// magnitude lands on the other side of the cutoff is cut on one route and
    /// kept on the other. For `K` stored values and `G` sectors, its
    /// elementwise cutoff and reciprocal take `O(K + G)` time and result
    /// space.
    ///
    /// Checked Generic admits the dense route's swapped output with the
    /// source's exact provider `Arc` and validates its identity, HomSpace,
    /// rank, and layout before any SVD/GEMM. A lazy adjoint is read through its parent's SVD in the
    /// dense stage, `(A^H)^+ = U S^+ Vh` for `A = U S Vh`, in both modes.
    ///
    /// `rows` and `cols` are the leg roles: the operation acts on the matrix
    /// view `self.permute(rows, cols)`, and the current split costs nothing
    /// extra (see [`Self::svd_compact`]'s *Leg roles*).
    pub fn pinv(
        &self,
        engine: &mut dyn Engine,
        rows: &[usize],
        cols: &[usize],
        rcond: f64,
    ) -> Result<Self, FacadeError<R>> {
        self.with_leg_roles(engine, rows, cols, |t, engine| t.factor_pinv(engine, rcond))
    }
}

impl<R, D, S> TensorMap<R, D, S>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R> + TypedTensorTransformDispatch<R, D>,
    D: FactorizationScalar,
    S: FactorExec<R, D>,
{
    /// Returns the compact QR factorization `self = q * r` as a [`Qr`].
    ///
    /// In coupled sector `c`, with block shape `m_c x n_c`, the bond dimension
    /// is `k_c = min(m_c, n_c)`. Thus `q : codomain(self) <- W` has
    /// orthonormal columns and `r : W <- domain(self)`. The diagonal of `r` is
    /// fixed to be real and non-negative, matching the default positive gauge.
    /// [`Self::qr_full`] instead uses `dim(W_c) = m_c`, making `q` square and
    /// `r` upper trapezoidal in every sector.
    ///
    /// An owned Host compact diagonal on `V <- V` is factorized directly and
    /// returns both factors in compact storage with `W = V`, including dual
    /// orientation, for multiplicity-free and checked-Generic providers alike
    /// (TensorKit's diagonal dispatch). Any other input, including a
    /// materialized diagonal, takes the dense route, whose `W` is a fresh
    /// nondual bond (TensorKit `fuse`); a dual `V` thus yields `W = V` or its
    /// nondual flip depending on storage. Swapped leg roles are one
    /// [`Self::permute`] first, so the rule follows the mode's rank-(1,1)
    /// swap of a compact diagonal: multiplicity-free rules with a real
    /// categorical scalar keep it compact, so the direct arm applies with the
    /// swapped leg as `W`; checked Generic densifies it and takes the dense
    /// route with a fresh nondual `W`.
    /// Phase is +1 at zero; work and output storage are `O(sum_c k_c)` after
    /// sector/layout validation. Use [`Self::diagview`] to read the factors,
    /// or [`Self::materialize`] before [`Self::dense_data`] for a dense buffer.
    /// A nonfinite entry, dense or diagonal, returns [`Error::Operation`] (an
    /// `InvalidArgument`: `qr`/`lq input components must be finite`) before
    /// any work.
    ///
    /// Other inputs run one dense QR per sector, with cost
    /// `O(sum_c m_c * n_c * min(m_c, n_c))`. Lazy adjoints are materialized
    /// only for the operation (LQ is itself the QR of the adjoint, so the
    /// copy is the one LQ would make). Checked factors use the same provider
    /// instance as `self`.
    /// If any sector fails or the provider rejects an output space, no factors
    /// are returned.
    ///
    /// # Errors
    ///
    /// Dense execution returns [`Error::Operation`], while factor-layout
    /// failures return [`Error::Core`] where applicable. If a checked provider
    /// rejects an output space, its original error is available as the source.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use symtn::sector::{U1FusionRule, U1Irrep};
    /// use symtn::typed::{GradedSpace, Qr, Runtime, TensorMap};
    ///
    /// let engine = &mut symtn::typed::CpuEngine::serial();
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
    /// let a: TensorMap<_, f64> = TensorMap::rand_with_seed(engine, &runtime, [&v], [&v], 3)?;
    /// let Qr { q, r } = a.qr_compact(engine, &[0], &[1])?;
    /// let rebuilt = q.compose(engine, &r)?;
    /// assert!(rebuilt.axpby(engine, 1.0, &a, -1.0)?.norm(engine, 2.0)? < 1e-12);
    /// # Ok::<(), symtn::typed::Error>(())
    /// ```
    ///
    /// `rows` and `cols` are the leg roles: the operation acts on the matrix
    /// view `self.permute(rows, cols)`, and the current split costs nothing
    /// extra (see [`Self::svd_compact`]'s *Leg roles*).
    ///
    /// # CUDA storage
    ///
    /// Owned dense device storage is factored as a stream over the coupled
    /// sectors, each gauge-fixed on the device by the backend's
    /// positive-diagonal QR (`R_jj` real and non-negative, phase 1 kept where
    /// `R_jj == 0`), so nothing of either factor crosses to the host. Each
    /// factor of a route is placed by one whole-factor device copy. A moved
    /// leg-role split is one device permute.
    /// A lazy adjoint device operand returns [`Error::UnsupportedOnDevice`]
    /// before any output exists.
    ///
    /// The finite-input check above holds on the device too, after the
    /// structural admission and before any output upload or solver call: it
    /// costs one device scan of the payload and one scalar download, which
    /// synchronizes the host with the device.
    pub fn qr_compact(
        &self,
        engine: &mut dyn Engine,
        rows: &[usize],
        cols: &[usize],
    ) -> Result<Qr<Self>, FacadeError<R>> {
        self.with_leg_roles(engine, rows, cols, |t, engine| {
            S::qr_compact(engine, EXEC, t)
        })
    }

    /// Returns the compact singular-value decomposition
    /// `self = u * s * vh` as an [`Svd`], which states the spectrum order and
    /// the storage routes of `s`.
    ///
    /// Sector `c` uses `k_c = min(m_c, n_c)`, with
    /// `u : codomain(self) <- W`, `s : W <- W`, and
    /// `vh : W <- domain(self)`. `u` has orthonormal columns, `vh` has
    /// orthonormal rows, and each sector's singular values are non-negative and
    /// descending. [`Self::svd_full`] instead returns square outer factors and
    /// a generally rectangular `s`.
    ///
    /// On Host, compact `s` stores only `sum_c k_c` diagonal values for both
    /// multiplicity-free and checked-Generic providers. `s.materialize()`
    /// produces dense storage when needed.
    /// All checked factors retain the source's exact provider `Arc`.
    ///
    /// The bond `W` of `s` is TensorKit's `fuse(codomain)`: a nondual bond of
    /// the singular-value sectors, which a nondual `V <- V` input already is.
    ///
    /// Dense inputs cost `O(sum_c m_c * n_c * min(m_c, n_c))`. An owned
    /// compact diagonal is sorted directly by sector, without
    /// a dense input or dense SVD call; its dense `u` and `vh` still require
    /// `O(sum_c k_c²)` output storage and writes. A lazy adjoint is factored
    /// from its parent without materializing it, in the same gauge as its
    /// materialized adjoint. A compact diagonal uses the
    /// same direct per-sector sorting and factor publication in both modes; a
    /// nonfinite entry, here or in dense input, returns
    /// [`Error::Operation`] (an `InvalidArgument`: `svd input components must be finite`)
    /// before any work.
    /// Any sector, layout, or provider failure returns no factors.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use symtn::sector::{U1FusionRule, U1Irrep};
    /// use symtn::typed::{GradedSpace, Runtime, Svd, TensorMap};
    ///
    /// let engine = &mut symtn::typed::CpuEngine::serial();
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
    /// let a: TensorMap<_, f64> = TensorMap::rand_with_seed(engine, &runtime, [&v], [&v], 4)?;
    /// let Svd { u, s, vh } = a.svd_compact(engine, &[0], &[1])?;
    /// let rebuilt = u.compose(engine, &s)?.compose(engine, &vh)?;
    /// assert!(rebuilt.axpby(engine, 1.0, &a, -1.0)?.norm(engine, 2.0)? < 1e-12);
    /// # Ok::<(), symtn::typed::Error>(())
    /// ```
    ///
    /// # Leg roles
    ///
    /// `rows` and `cols` list source axes (`0..rank`, codomain first) exactly
    /// as for [`Self::permute`]. They are the codomain and the domain of the
    /// matrix this operation acts on, and the result is the operation applied
    /// to `self.permute(rows, cols)`; every space, factor and error stated
    /// here refers to that matrix view. The current split (`rows = 0..nout`,
    /// `cols = nout..rank`) borrows `self` and runs no transform. Any other
    /// split costs that one permute and nothing more: the permuted tensor is
    /// consumed in place of `self`, and its packed coupled-sector regions are
    /// read by the solver directly. Every factorization and matrix function
    /// (`svd_*`, `qr_*`, `lq_*`, the polar and null-space families, `eigh_*`,
    /// `eig_*`, `exp`, `inv`, `pinv`) takes its leg roles this way. A split
    /// other than the current one needs a symmetric braiding; otherwise the
    /// permute's `UnsupportedBraidingStyle` error is returned before any
    /// factorization work, as are its errors for malformed axes.
    ///
    /// # CUDA storage
    ///
    /// Owned dense device storage is factored as a stream over the coupled
    /// sectors: each nonempty route is decomposed and assembled before its
    /// raw device factors are dropped. The returned `s` is a dense CUDA
    /// tensor, because CUDA diagonal storage is not part of the typed storage
    /// contract; each route's singular values are copied into its diagonal on
    /// the device (element stride `k + 1`). A moved leg-role split is one
    /// device permute. A lazy adjoint device operand returns
    /// [`Error::UnsupportedOnDevice`] before any output exists. The
    /// finite-input check above holds on the device too, after the structural
    /// admission and before any output upload or solver call.
    ///
    /// Transfers, exactly: host to device, one zero upload per factor
    /// (`u`, `s`, `vh`: the dense output sizes, `s` being `Σ_c k_c²`
    /// elements, since a host zero buffer is the only device allocation
    /// path, #740) plus one upload of `max_c rows_c` `i64` gauge weights
    /// when any route is nonempty. Device to host, no tensor payload: one
    /// real scalar, the finite-input scan's sum over the payload (a host
    /// sync); the singular values never cross to the host, so the call ends
    /// without a spectrum download. The backend still reads O(1)
    /// solver-status metadata per route, a host barrier each.
    ///
    /// `u` and `vh` carry the Host gauge: in every column of each sector's
    /// `u`, the first largest-magnitude entry is real and non-negative, and
    /// the matching `vh` row takes the inverse phase. The fix-up runs on the
    /// device without a download (`symtn_dense::cuda_svd_gauge_phases`):
    /// 13 Tenferro ops per route for the phases, then per side a scaling of
    /// the factor before its copy (2 ops), plus one `conj` for a complex left
    /// side. They are
    /// counted in `CudaTransferStats::gauge_ops`.
    pub fn svd_compact(
        &self,
        engine: &mut dyn Engine,
        rows: &[usize],
        cols: &[usize],
    ) -> Result<Svd<Self>, FacadeError<R>> {
        self.with_leg_roles(engine, rows, cols, |t, engine| {
            S::svd_compact(engine, EXEC, t)
        })
    }

    /// Returns the Hermitian eigendecomposition `self = v * d * v^H` as an
    /// [`Eigh`], which states the spectrum order and the storage routes of `d`.
    ///
    /// The input must be an endomorphism whose every coupled-sector block `A`
    /// is Hermitian to `hermitian_tol` (MatrixAlgebraKit's `hermitian_tol`):
    /// `‖(A − Aᴴ)/2‖_F ≤ tol · ‖A‖_F`, where [`HermitianTol::DEFAULT`] is
    /// `eps(real(D))^(3/4)`, MatrixAlgebraKit's level at unit norm. Unlike
    /// MatrixAlgebraKit's absolute `atol`, the tolerance is relative, so the
    /// decision does not change when `self` is rescaled. In each coupled sector,
    /// `v : codomain(self) <- W` is unitary and `d : W <- W` holds the signed
    /// real eigenvalues. Eigenvalues are ascending within each sector; each
    /// eigenvector's phase is fixed by making
    /// its largest-magnitude component real and non-negative. Bases inside an
    /// exactly degenerate eigenspace remain backend-dependent.
    ///
    /// Both multiplicity-free and checked-Generic `d` factors use compact
    /// diagonal storage. Checked factors retain the exact source provider
    /// `Arc`. An owned Host compact diagonal is read directly into a
    /// permutation eigenbasis, for both multiplicity-free and checked-Generic
    /// providers, after the same finite-input and Hermiticity checks as
    /// [`Self::eigh_vals`]; its dense output still occupies `Σ_c k_c²`
    /// elements. Lazy adjoints use an operation-local dense payload.
    ///
    /// # Errors and cost
    ///
    /// A non-endomorphism, failed Hermiticity check, or dense numerical failure
    /// returns [`Error::Operation`]. Layout failures retain [`Error::Core`],
    /// and multiplicity-free algebra failures retain [`Error::FusionAlgebra`]
    /// where applicable. For checked Generic, the original layout or provider
    /// error is available as the source. Factors are returned only after every
    /// sector succeeds; otherwise the method returns an error and no factors.
    /// The direct compact path sorts in `O(Σ_c k_c log k_c)` time and writes
    /// `O(Σ_c k_c²)` eigenvector elements; the general dense route costs
    /// `O(Σ_c n_c³)` time.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use symtn::sector::{U1FusionRule, U1Irrep};
    /// use symtn::typed::{Eigh, GradedSpace, HermitianTol, Runtime, TensorMap};
    ///
    /// let engine = &mut symtn::typed::CpuEngine::serial();
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
    /// let a: TensorMap<_, f64> = TensorMap::isomorphism(engine, &runtime, [&v], [&v])?.scale(engine, 2.0)?;
    /// let Eigh { d, v: eigenvectors } = a.eigh_full(engine, &[0], &[1], HermitianTol::DEFAULT)?;
    /// let inverse = eigenvectors.adjoint(engine)?;
    /// let rebuilt = eigenvectors.compose(engine, &d)?.compose(engine, &inverse)?;
    /// assert!(rebuilt.axpby(engine, 1.0, &a, -1.0)?.norm(engine, 2.0)? < 1e-12);
    /// # Ok::<(), symtn::typed::Error>(())
    /// ```
    ///
    /// `rows` and `cols` are the leg roles: the operation acts on the matrix
    /// view `self.permute(rows, cols)`, and the current split costs nothing
    /// extra (see [`Self::svd_compact`]'s *Leg roles*).
    ///
    /// # CUDA storage
    ///
    /// Owned dense device storage is decomposed with both factors left on the
    /// source device; no receiver-sized payload is downloaded or
    /// materialized, and the returned `d` is a dense CUDA tensor, because CUDA
    /// diagonal storage is not part of the typed storage contract. A moved
    /// leg-role split is one device permute. A lazy adjoint device operand
    /// returns [`Error::UnsupportedOnDevice`] before any output exists.
    ///
    /// No truncation decision is made: every eigenpair is kept. Eigenvalues are
    /// the only numerical payload that crosses to the host, where they are
    /// put in the Host order (ascending per sector; cuSOLVER `syevd` already
    /// returns it, so only an O(n) check runs); the eigenvectors stay on the
    /// device and are copied into `v` (one copy per route). Only if some
    /// sector's order is not the identity are the columns gathered by a
    /// selector GEMM instead.
    ///
    /// A nonfinite entry returns [`Error::Operation`] (an `InvalidArgument`:
    /// `eigh input components must be finite`) as on the Host: after the
    /// endomorphism, stacking and placement admission, before the Hermiticity
    /// check, and before any output upload or solver call.
    ///
    /// Transfers, exactly:
    ///
    /// - Finite-input check: one device scan of the payload and one real
    ///   scalar download, a host sync.
    /// - Hermiticity admission (`cuda_hermitian_regions`): at most three
    ///   downloads of gathered real scalars, one per stage that has an
    ///   undecided sector (per-sector maxima, then sums of squares and
    ///   residual maxima, then residual sums of squares; O(1) values per
    ///   sector), and one rank-0 real normalizer upload per sector entering
    ///   stage 2 (nonzero, finite input maximum) and per sector entering
    ///   stage 3 (nonzero, finite residual maximum): at most two per
    ///   sector.
    /// - Device to host: all sectors' eigenvalues (`Σ_c n_c` real values) in
    ///   one download, which the non-finite check, the host ascending order and
    ///   the factor-space plan consume.
    /// - Host to device: `d` as one dense upload of `Σ_c n_c²` elements with
    ///   the sorted eigenvalues already on its diagonal, one zero upload for
    ///   `v`, and a packed selector upload only when some sector's solver
    ///   order is not ascending (none for cuSOLVER `syevd`).
    ///
    /// Why `d` is filled on the host rather than scattered on the
    /// device: the only device allocation path is a zero upload of the same
    /// `Σ_c n_c²` elements (#740), so a device scatter would add a
    /// `Σ_c n_c` upload and move nothing less.
    pub fn eigh_full(
        &self,
        engine: &mut dyn Engine,
        rows: &[usize],
        cols: &[usize],
        hermitian_tol: HermitianTol,
    ) -> Result<Eigh<Self>, FacadeError<R>> {
        self.with_leg_roles(engine, rows, cols, |t, engine| {
            S::eigh_full(engine, EXEC, t, hermitian_tol)
        })
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R> + TypedTensorTransformDispatch<R, D>,
    D: FactorizationScalar,
{
    /// Returns the full SVD `self = u * s * vh` with square outer factors.
    ///
    /// In sector `c`, `u` is `m_c x m_c`, `vh` is `n_c x n_c`, and `s` is
    /// the rectangular `m_c x n_c` diagonal matrix. The spaces are
    /// `u : codomain <- W_out`, `s : W_out <- W_in`, and
    /// `vh : W_in <- domain`. It accepts the same inputs as
    /// [`Self::svd_compact`], but its square outer factors can require more
    /// dense storage. On Host, `s` uses compact diagonal storage on the bond
    /// `fuse(codomain)` exactly when the constructed `W_out` and `W_in` legs
    /// coincide and every positive bond sector has a complete spectrum.
    /// Otherwise it is dense. An owned compact diagonal input returns its
    /// [`Self::svd_compact`], with which its full SVD coincides (TensorKit
    /// `svd_compact!(::DiagonalAlgorithm)` is `svd_full!`), in every fusion
    /// mode: no dense input materialization or solver call.
    /// Call `s.materialize()` before `dense_data()` when needed.
    /// Checked factors use the source provider instance; a failure returns no
    /// factors.
    ///
    /// The factorization is blockwise, so a rule with complex categorical
    /// coefficients factorizes like any other (TensorKit `foreachblock`):
    ///
    /// ```
    /// use std::sync::Arc;
    /// use symtn::sector::{TableRule, TableSector};
    /// use symtn::typed::{Complex64, GradedSpace, Runtime, Svd, TensorMap};
    ///
    /// let engine = &mut symtn::typed::CpuEngine::serial();
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(
    ///     Arc::new(TableRule::fibonacci()),
    ///     [(TableSector::new(1), 1), (TableSector::new(2), 2)],
    /// )?;
    /// let t: TensorMap<TableRule, Complex64> =
    ///     TensorMap::rand_with_seed(engine, &runtime, [&v, &v], [&v], 7)?;
    /// let Svd { u, s, vh } = t.svd_full(engine, &[0, 1], &[2])?;
    /// let rebuilt = u.compose(engine, &s)?.compose(engine, &vh)?;
    /// assert!(rebuilt.axpby(engine, Complex64::ONE, &t, -Complex64::ONE)?.norm(engine, 2.0)? < 1e-12);
    /// # Ok::<(), symtn::typed::Error>(())
    /// ```
    ///
    /// `rows` and `cols` are the leg roles: the operation acts on the matrix
    /// view `self.permute(rows, cols)`, and the current split costs nothing
    /// extra (see [`Self::svd_compact`]'s *Leg roles*).
    pub fn svd_full(
        &self,
        engine: &mut dyn Engine,
        rows: &[usize],
        cols: &[usize],
    ) -> Result<Svd<Self>, FacadeError<R>> {
        self.with_leg_roles(engine, rows, cols, Self::factor_svd_full)
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R> + TypedAdjointSpace<R> + TypedTensorTransformDispatch<R, D>,
    D: FactorizationScalar,
{
    /// Returns the compact LQ factorization `self = l * q` as an [`Lq`].
    ///
    /// Sector `c` uses bond dimension `k_c = min(m_c, n_c)`, with
    /// `l : codomain(self) <- W` and `q : W <- domain(self)`. The rows of `q`
    /// are orthonormal and the diagonal of `l` is real and non-negative.
    /// [`Self::lq_full`] instead uses `dim(W_c) = n_c`, so `q` is square and
    /// `l` is lower trapezoidal.
    ///
    /// Its cost and compact storage contract are the same as [`Self::qr_compact`]:
    /// owned Host compact diagonals preserve `W = V` and both factors
    /// are compact, including on a dual `V`.
    /// Checked factors use the source provider instance, and a failure returns
    /// no factors. A lazy adjoint `A^H` is not materialized: the QR
    /// `A = Q R` of its parent, read in place, gives `l = R^H` and `q = Q^H`
    /// in the gauge LQ of the materialized adjoint has.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use symtn::sector::{U1FusionRule, U1Irrep};
    /// use symtn::typed::{GradedSpace, Lq, Runtime, TensorMap};
    ///
    /// let engine = &mut symtn::typed::CpuEngine::serial();
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
    /// let a: TensorMap<_, f64> = TensorMap::rand_with_seed(engine, &runtime, [&v], [&v], 5)?;
    /// let Lq { l, q } = a.lq_compact(engine, &[0], &[1])?;
    /// assert!(l.compose(engine, &q)?.axpby(engine, 1.0, &a, -1.0)?.norm(engine, 2.0)? < 1e-12);
    /// # Ok::<(), symtn::typed::Error>(())
    /// ```
    ///
    /// `rows` and `cols` are the leg roles: the operation acts on the matrix
    /// view `self.permute(rows, cols)`, and the current split costs nothing
    /// extra (see [`Self::svd_compact`]'s *Leg roles*).
    pub fn lq_compact(
        &self,
        engine: &mut dyn Engine,
        rows: &[usize],
        cols: &[usize],
    ) -> Result<Lq<Self>, FacadeError<R>> {
        self.with_leg_roles(engine, rows, cols, |t, engine| {
            t.factor_lq(engine,
                FactorOp::LqCompact,
                |engine, lease, source| {
                    symtn_matrixalgebra::seam::lq_compact_from_source::<R::Mode, _, _, _, _>(engine,
                        lease, source,
                    )
                },
                |engine, lease, parent, adjoint_space| {
                    symtn_matrixalgebra::seam::lq_compact_adjoint_from_parent::<
                        R::Mode,
                        _,
                        _,
                        _,
                        _,
                    > (engine, lease, parent, adjoint_space)
                },
            )
        })
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    D: TensorScalar,
{
    /// Applies `f` to every stored value of a compact diagonal tensor and
    /// returns a compact diagonal on the same space.
    ///
    /// This is TensorKit 0.17's `DiagonalTensorMap($f.(d.data), d.domain)`
    /// (`tensors/diagonal.jl:384-390`) with the elementwise function supplied
    /// by the caller: `s.map_diagonal(f64::sqrt)` splits singular values for a
    /// Vidal gauge, `s.map_diagonal(|x| 1.0 / x)` inverts them. A spectral map
    /// never mixes sectors or changes a bond dimension, so the result keeps the
    /// receiver's space. `f` sees the values directly; for a real payload it
    /// decides what a negative entry means (`f64::sqrt` yields `NaN`).
    ///
    /// # Domain
    ///
    /// The receiver must store a compact diagonal: a Host
    /// [`Self::svd_compact`] `s`, an eigendecomposition's `d`, or a tensor
    /// built by [`Self::diagonal`]. A dense tensor is rejected even when its
    /// blocks happen to be diagonal; this method never scans or builds a
    /// dense buffer. A diagonal that is stored densely is made compact explicitly with
    /// `TensorMap::diagonal(runtime, bond, t.diagview()?)`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when the receiver has dense storage,
    /// including a lazy adjoint.
    ///
    /// # Complexity
    ///
    /// `Σ_c k_c` calls of `f` over the stored values and one owned compact
    /// output of that length.
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use symtn::sector::{U1FusionRule, U1Irrep};
    /// use symtn::typed::{GradedSpace, Runtime, SectorSpectrum, TensorMap};
    ///
    /// let engine = &mut symtn::typed::CpuEngine::serial();
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
    /// let values = SectorSpectrum { sector: U1Irrep::new(0), values: vec![4.0, 9.0] };
    /// let s: TensorMap<_, f64> = TensorMap::diagonal(engine, &runtime, &v, [values])?;
    /// assert_eq!(s.map_diagonal(engine, f64::sqrt)?.diagview(engine)?[0].values, [2.0, 3.0]);
    /// let dense: TensorMap<_, f64> = TensorMap::isomorphism(engine, &runtime, [&v], [&v])?;
    /// assert!(dense.map_diagonal(engine, f64::sqrt).is_err());
    /// # Ok::<(), symtn::typed::Error>(())
    /// ```
    pub fn map_diagonal(&self, engine: &mut dyn Engine, f: impl Fn(D) -> D) -> Result<Self, Error> {
        admit_host_engine(engine)?;
        let Some(spectrum) = self.spectrum() else {
            return Err(Error::InvalidArgument(
                "map_diagonal requires a compact diagonal tensor (a compact factor or a \
                 TensorMap::diagonal), but this tensor has dense storage"
                    .to_string(),
            ));
        };
        Ok(self.with_spectrum(map_spectrum(spectrum, |value| Ok(f(value)))?))
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R> + TypedTensorTransformDispatch<R, D>,
    D: FactorizationScalar,
{
    /// Returns the full QR factorization `self = q * r` as a [`Qr`].
    ///
    /// Sector `c` has a square `m_c x m_c` unitary `q` and an
    /// `m_c x n_c` upper-trapezoidal `r`; the intermediate bond therefore has
    /// the codomain's coupled-sector dimensions. The diagonal of `r` is real
    /// and non-negative. For sector shape `m_c x n_c`, dense work is
    /// `O(m_c² n_c)` when `m_c <= n_c`; when `m_c > n_c`, completing the
    /// square `q` costs `O(m_c²(n_c + m_c))`. Source packing and owned factor
    /// publication are additional costs. An owned Host compact diagonal
    /// uses `W = V` and two compact factors under the boundary documented by
    /// [`Self::qr_compact`].
    /// See [`Self::qr_compact`] for the
    /// compact alternative, storage and lazy-input behavior, errors, and example.
    ///
    /// `rows` and `cols` are the leg roles: the operation acts on the matrix
    /// view `self.permute(rows, cols)`, and the current split costs nothing
    /// extra (see [`Self::svd_compact`]'s *Leg roles*).
    pub fn qr_full(
        &self,
        engine: &mut dyn Engine,
        rows: &[usize],
        cols: &[usize],
    ) -> Result<Qr<Self>, FacadeError<R>> {
        self.with_leg_roles(engine, rows, cols, |t, engine| {
            t.factor_qr(engine, FactorOp::QrFull, |engine, lease, source| {
                symtn_matrixalgebra::seam::qr_full_from_source::<R::Mode, _, _, _, _>(
                    engine, lease, source,
                )
            })
        })
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R> + TypedAdjointSpace<R> + TypedTensorTransformDispatch<R, D>,
    D: FactorizationScalar,
{
    /// Returns the full LQ factorization `self = l * q` as an [`Lq`].
    ///
    /// Sector `c` has an `m_c x n_c` lower-trapezoidal `l` and a square
    /// `n_c x n_c` unitary `q`; the intermediate bond therefore has the
    /// domain's coupled-sector dimensions. The diagonal of `l` is real and
    /// non-negative. For sector shape `m_c x n_c`, dense work is
    /// `O(n_c² m_c)` when `n_c <= m_c`; when `n_c > m_c`, completing the
    /// square `q` costs `O(n_c²(m_c + n_c))`. Source packing, the sectorwise
    /// adjoint, and owned factor publication are additional costs. An owned
    /// Host compact diagonal uses `W = V` and two compact factors under
    /// the boundary documented by [`Self::lq_compact`]. See [`Self::lq_compact`] for the compact alternative, storage and
    /// lazy-input behavior, errors, and example.
    ///
    /// `rows` and `cols` are the leg roles: the operation acts on the matrix
    /// view `self.permute(rows, cols)`, and the current split costs nothing
    /// extra (see [`Self::svd_compact`]'s *Leg roles*).
    pub fn lq_full(
        &self,
        engine: &mut dyn Engine,
        rows: &[usize],
        cols: &[usize],
    ) -> Result<Lq<Self>, FacadeError<R>> {
        self.with_leg_roles(engine, rows, cols, |t, engine| {
            t.factor_lq(
                engine,
                FactorOp::LqFull,
                |engine, lease, source| {
                    symtn_matrixalgebra::seam::lq_full_from_source::<R::Mode, _, _, _, _>(
                        engine, lease, source,
                    )
                },
                |engine, lease, parent, adjoint_space| {
                    symtn_matrixalgebra::seam::lq_full_adjoint_from_parent::<R::Mode, _, _, _, _>(
                        engine,
                        lease,
                        parent,
                        adjoint_space,
                    )
                },
            )
        })
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R> + TypedTensorTransformDispatch<R, D>,
    D: FactorizationScalar,
{
    /// Returns only the singular values, grouped by provider-labelled coupled
    /// sector and descending within each sector.
    ///
    /// No factor tensor or intermediate bond is built. This is the least
    /// allocating member of the SVD family when only the spectrum is needed.
    /// Lazy adjoints are read through their owned parent,
    /// and an owned compact diagonal input is read and sorted directly without
    /// a dense solver for both multiplicity-free and checked-Generic providers;
    /// a nonfinite entry, here or in dense input, returns
    /// [`Error::Operation`] (an `InvalidArgument`: `svd input components must be finite`).
    /// A dense failure returns
    /// [`Error::Operation`]; if a provider cannot decode a sector label, its
    /// original error is available as the source. See [`Self::svd_compact`] for
    /// the decomposition contract and representative example.
    /// For the direct path, validating `B` source blocks, sorting `k_c` values
    /// in each of `G` sectors and sorting public labels costs
    /// `O(B + Σ_c k_c log k_c + G log G)` time and `O(Σ_c k_c + G)` space.
    ///
    /// `rows` and `cols` are the leg roles: the operation acts on the matrix
    /// view `self.permute(rows, cols)`, and the current split costs nothing
    /// extra (see [`Self::svd_compact`]'s *Leg roles*).
    pub fn svd_vals(
        &self,
        engine: &mut dyn Engine,
        rows: &[usize],
        cols: &[usize],
    ) -> Result<Vec<SectorSpectrum<<R as TypedSectorAdmission>::Sector, f64>>, FacadeError<R>> {
        self.with_leg_roles(engine, rows, cols, |t, engine| {
            t.factor_values(engine, FactorOp::SvdVals, |engine, lease, source| {
                symtn_matrixalgebra::seam::svd_vals_from_source::<R::Mode, _, _, _, _>(
                    engine, lease, source,
                )
            })
        })
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R> + TypedTensorTransformDispatch<R, D>,
    D: FactorizationScalar,
{
    /// Returns only the real Hermitian eigenvalues, grouped by
    /// provider-labelled coupled sector and ascending within each sector.
    ///
    /// No eigenvector factor or bond space is built. The input must be an
    /// endomorphism and every sector must pass the same Hermiticity admission
    /// at `hermitian_tol` as [`Self::eigh_full`]. An owned Host compact diagonal is read directly for
    /// both multiplicity-free and checked-Generic providers: a nonfinite entry
    /// returns [`Error::Operation`] (an `InvalidArgument`: `eigh input components must be
    /// finite`; dense input too, after the endomorphism and stacking checks
    /// and before the Hermiticity check), the dense route's Hermiticity check is applied to the diagonal,
    /// and the eigenvalues are its real parts. Lazy adjoints use an
    /// operation-local dense payload. Dense failures return
    /// [`Error::Operation`],
    /// layout failures return [`Error::Core`], and an original provider or
    /// label-decoding error is available as the source. No spectrum is returned
    /// unless every sector succeeds.
    ///
    /// `rows` and `cols` are the leg roles: the operation acts on the matrix
    /// view `self.permute(rows, cols)`, and the current split costs nothing
    /// extra (see [`Self::svd_compact`]'s *Leg roles*).
    pub fn eigh_vals(
        &self,
        engine: &mut dyn Engine,
        rows: &[usize],
        cols: &[usize],
        hermitian_tol: HermitianTol,
    ) -> Result<Vec<SectorSpectrum<<R as TypedSectorAdmission>::Sector, f64>>, FacadeError<R>> {
        self.with_leg_roles(engine, rows, cols, |t, engine| {
            t.factor_values(engine, FactorOp::EighVals, |engine, lease, source| {
                symtn_matrixalgebra::seam::eigh_vals_from_source::<R::Mode, _, _, _, _>(
                    engine,
                    lease,
                    source,
                    hermitian_tol,
                )
            })
        })
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R> + TypedTensorTransformDispatch<R, D>,
    D: AdvancedLinalgScalar,
    // Carried across the whole eig row even though this member builds no
    // factor: the three are one API surface.
    <D as FactorScalar>::Eig: TensorScalar,
{
    /// Returns only the general eigenvalues as `Complex64` (at every payload
    /// dtype, like every spectrum of this crate), grouped by
    /// provider-labelled sector and ascending in lexicographic
    /// `(real(λ), imag(λ))` order within each sector.
    ///
    /// No eigenvector factor or bond space is built. The input must be an
    /// endomorphism. An owned Host compact diagonal is read directly for both
    /// multiplicity-free and checked-Generic providers: a nonfinite entry
    /// returns [`Error::Operation`] (an `InvalidArgument`: `eig input components must be
    /// finite`; dense input too, after the endomorphism and stacking checks),
    /// and an eigenvalue of infinite magnitude fails the dense
    /// route's eigenvalue check. Lazy adjoints use an operation-local dense
    /// payload.
    ///
    /// `rows` and `cols` are the leg roles: the operation acts on the matrix
    /// view `self.permute(rows, cols)`, and the current split costs nothing
    /// extra (see [`Self::svd_compact`]'s *Leg roles*).
    pub fn eig_vals(
        &self,
        engine: &mut dyn Engine,
        rows: &[usize],
        cols: &[usize],
    ) -> Result<
        Vec<SectorSpectrum<<R as TypedSectorAdmission>::Sector, num_complex::Complex64>>,
        FacadeError<R>,
    > {
        self.with_leg_roles(engine, rows, cols, |t, engine| {
            t.factor_values(engine, FactorOp::EigVals, |engine, lease, source| {
                symtn_matrixalgebra::seam::eig_vals_from_source::<R::Mode, _, _, _, _>(
                    engine, lease, source,
                )
            })
        })
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R> + TypedTensorTransformDispatch<R, D>,
    D: AdvancedLinalgScalar,
    <D as FactorScalar>::Eig: TensorScalar,
{
    /// Returns the general eigendecomposition `self * v = v * d` as an
    /// [`Eig`] of complex factors.
    ///
    /// For a diagonalizable input this gives
    /// `self = v * d * v^-1`. Both factors are complex even when the input is
    /// real, in the payload's own precision: `D::Eig` is `Complex64` for `f64`
    /// and `Complex64`, and `Complex32` for `f32` and `Complex32`. `v : codomain(self) <- W` holds right eigenvectors and compact
    /// `d : W <- W` holds their eigenvalues. Values are ascending in
    /// lexicographic `(real(λ), imag(λ))` order within each sector (dense
    /// sectors are sorted; MatrixAlgebraKit keeps `geev`'s order there).
    /// Each eigenvector column has unit 2-norm, as LAPACK `geev` returns it to
    /// MatrixAlgebraKit, and its largest-magnitude component is then made real
    /// and non-negative. Degenerate eigenbases remain backend-dependent.
    ///
    /// Both fusion modes promise `self * v ≈ v * d` to backward error and
    /// certify no eigenvector rank, as MatrixAlgebraKit `eig_full!`: a finite
    /// defective input returns its computed, possibly nearly parallel,
    /// eigenvectors. Checked Generic retains the exact source provider `Arc`.
    /// An owned Host compact diagonal reads its eigenvalues directly and
    /// builds a dense permutation factor, for both multiplicity-free and
    /// checked-Generic providers, after the same finite-input and eigenvalue
    /// checks as [`Self::eig_vals`]. Lazy adjoints use an operation-local
    /// dense payload.
    ///
    /// A non-endomorphism, non-finite input, invalid/non-finite eigenvalue,
    /// non-finite eigenvector entry, factor-layout failure, or provider
    /// failure returns no factors. The solver's results are checked
    /// eigenvalues first, then eigenvectors; both failures are a
    /// `DenseError::NumericalFailure` (a dense solver whose back-substitution
    /// overflows on a large Jordan block reports it here rather than
    /// returning NaN columns).
    /// The direct compact path sorts in `O(Σ_c n_c log n_c)` time and writes
    /// `O(Σ_c n_c²)` eigenvector elements; the dense route costs
    /// `O(Σ_c n_c³)` time.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use symtn::sector::{U1FusionRule, U1Irrep};
    /// use symtn::typed::{Eig, GradedSpace, Runtime, TensorMap};
    ///
    /// let engine = &mut symtn::typed::CpuEngine::serial();
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
    /// let a: TensorMap<_, f64> = TensorMap::isomorphism(engine, &runtime, [&v], [&v])?.scale(engine, 2.0)?;
    /// let Eig { d, v: eigenvectors } = a.eig_full(engine, &[0], &[1])?;
    /// let inverse = eigenvectors.inv(engine, &[0], &[1])?;
    /// let rebuilt = eigenvectors.compose(engine, &d)?.compose(engine, &inverse)?;
    /// let a = a.convert::<num_complex::Complex64>(engine)?;
    /// assert!(rebuilt.axpby(engine, 1.0.into(), &a, (-1.0).into())?.norm(engine, 2.0)? < 1e-12);
    /// # Ok::<(), symtn::typed::Error>(())
    /// ```
    ///
    /// `rows` and `cols` are the leg roles: the operation acts on the matrix
    /// view `self.permute(rows, cols)`, and the current split costs nothing
    /// extra (see [`Self::svd_compact`]'s *Leg roles*).
    pub fn eig_full(
        &self,
        engine: &mut dyn Engine,
        rows: &[usize],
        cols: &[usize],
    ) -> Result<Eig<TensorMap<R, <D as FactorScalar>::Eig>>, FacadeError<R>> {
        self.with_leg_roles(engine, rows, cols, Self::factor_eig_full)
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R> + TypedAdjointSpace<R> + TypedTensorTransformDispatch<R, D>,
    D: FactorizationScalar,
{
    /// Returns an orthonormal basis `n : codomain(self) <- W` for the left
    /// null space, satisfying `n^H * self ~= 0`.
    ///
    /// The null space is shape-based, as TensorKit/MatrixAlgebraKit's default
    /// `left_null` (`LeftNullViaQR`, `qr_null!`): in coupled sector `c` the
    /// fresh non-dual one-leg bond `W` has `(m_c - n_c)+` directions, the
    /// trailing columns of the sector's full QR, whatever the input's rank.
    /// A rank-deficient input therefore has a null space smaller than its
    /// numerical kernel. Sectors present only in the codomain contribute their
    /// whole degeneracy; sectors with zero nullity are absent.
    /// [`Self::right_null`] returns the corresponding basis on the domain side.
    ///
    /// Only sectors with `m_c > n_c` run dense work: one economy QR of the
    /// `m_c x (n_c + m_c)` block `[A_c | I]`, `O(m_c^2 (n_c + m_c))`. A compact
    /// diagonal is square in every sector, so its null space is empty and is
    /// returned with no dense work. A nonfinite entry, dense or diagonal,
    /// returns [`Error::Operation`] (an `InvalidArgument`: `null input components must be
    /// finite`).
    /// Lazy adjoints use the opposite null space of their
    /// owned parent and return a detached result without filling the receiver
    /// cache. Checked results use the same provider instance as `self`. symtn
    /// creates the output bond only after every sector succeeds; otherwise it
    /// returns an error and no tensor.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use symtn::sector::{U1FusionRule, U1Irrep};
    /// use symtn::typed::{GradedSpace, Runtime, TensorMap};
    ///
    /// let engine = &mut symtn::typed::CpuEngine::serial();
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
    /// let zero: TensorMap<_, f64> = TensorMap::zeros(engine, &runtime, [&v], [&v])?;
    /// let n = zero.left_null(engine, &[0], &[1])?;
    /// assert!(n.adjoint(engine)?.compose(engine, &zero)?.norm(engine, 2.0)? < 1e-12);
    /// # Ok::<(), symtn::typed::Error>(())
    /// ```
    ///
    /// # Rank-revealing null space
    ///
    /// TensorKit's `left_null(t; alg = :svd, trunc = (; atol, rtol, maxnullity))`
    /// is a composition, not a separate operation: [`Self::svd_full`], the
    /// diagonal of its `s` padded with the rectangular zeros to `u`'s bond
    /// ([`GradedSpace::extend_spectrum`](crate::typed::GradedSpace::extend_spectrum)), the keep-below selection
    /// [`Truncation::below`](crate::typed::Truncation::below) (intersected with
    /// [`Truncation::rank_smallest`](crate::typed::Truncation::rank_smallest)
    /// for `maxnullity`), and [`Self::restrict_leg`] on `u`. MatrixAlgebraKit's
    /// default `trunc` is `rtol = eps(real(T))^(2/3)`. Each sector keeps its
    /// `(m_c - n_c)₊` rectangular directions plus one per singular value at
    /// or below `max(atol, rtol * ‖s‖)`, the weighted 2-norm of every sector.
    /// TensorKit's `trunc = notrunc()` keeps only the rectangular zeros of the
    /// sectors `s` has a block in (`MAK.truncate(::typeof(left_null!), (U, S),
    /// ::NoTruncation)` ranges over `blocks(S)`), so it matches this method
    /// only on sectors present in both codomain and domain: a codomain-only
    /// sector is dropped there, while this method keeps its whole degeneracy,
    /// as the QR-based default `qr_null` does.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use symtn::sector::{U1FusionRule, U1Irrep};
    /// use symtn::typed::{GradedSpace, Runtime, Svd, TensorMap, Truncation};
    ///
    /// let engine = &mut symtn::typed::CpuEngine::serial();
    /// let runtime = Runtime::builder().build()?;
    /// let rule = Arc::new(U1FusionRule);
    /// let tall = GradedSpace::try_new(Arc::clone(&rule), [(U1Irrep::new(0), 3)])?;
    /// let short = GradedSpace::try_new(rule, [(U1Irrep::new(0), 2)])?;
    /// // Rank one: the shape gives one null direction, the rank two.
    /// let a: TensorMap<_, f64> = TensorMap::from_subblock_fn(engine,
    ///     &runtime, [&tall], [&short], |_, i| ((i[0] + 1) * (i[1] + 1)) as f64,
    /// )?;
    /// assert_eq!(a.left_null(engine, &[0], &[1])?.domain()[0].degeneracy(&U1Irrep::new(0))?, 1);
    ///
    /// let Svd { u, s, .. } = a.svd_full(engine, &[0], &[1])?;
    /// let bond = &u.domain()[0];
    /// let spectrum = bond.extend_spectrum(&s.diagview(engine)?)?;
    /// let rtol = f64::EPSILON.powf(2.0 / 3.0);
    /// let null = bond.find_truncated(&spectrum, &Truncation::below(0.0, rtol)?)?;
    /// let n = u.restrict_leg(engine, &[(u.codomain_rank(), &null.selection)])?;
    /// assert_eq!(n.domain()[0].degeneracy(&U1Irrep::new(0))?, 2);
    /// assert!(n.adjoint(engine)?.compose(engine, &a)?.norm(engine, 2.0)? < 1e-12);
    /// # Ok::<(), symtn::typed::Error>(())
    /// ```
    ///
    /// `rows` and `cols` are the leg roles: the operation acts on the matrix
    /// view `self.permute(rows, cols)`, and the current split costs nothing
    /// extra (see [`Self::svd_compact`]'s *Leg roles*).
    pub fn left_null(
        &self,
        engine: &mut dyn Engine,
        rows: &[usize],
        cols: &[usize],
    ) -> Result<Self, FacadeError<R>> {
        self.with_leg_roles(engine, rows, cols, |t, engine| {
            t.factor_null(engine,
                FactorOp::LeftNull,
                |parent, engine| {
                    parent.factor_null(engine,
                        FactorOp::RightNull,
                        |_, _| {
                            // A lazy adjoint's parent is owned, so it is
                            // never redirected again.
                            Err(internal_layout_error(
                                "an owned null-space parent was redirected",
                            )
                            .into())
                        },
                        |engine, lease, source| {
                            symtn_matrixalgebra::seam::right_null_from_source::<R::Mode, _, _, _, _>(engine,
                                lease, source,
                            )
                        },
                    )
                },
                |engine, lease, source| {
                    symtn_matrixalgebra::seam::left_null_from_source::<R::Mode, _, _, _, _>(engine,
                        lease, source,
                    )
                },
            )
        })
    }

    /// Returns an orthonormal-row basis `n : W <- domain(self)` for the right
    /// null space, satisfying `self * n^H ~= 0`.
    ///
    /// The fresh bond contains `(n_c - m_c)+` directions per sector, the
    /// trailing rows of the sector's full LQ (TensorKit/MatrixAlgebraKit
    /// `RightNullViaLQ`, `lq_null!`), with the cost of [`Self::left_null`] with
    /// rows and columns exchanged. Compact diagonals return an empty null
    /// space directly; a lazy
    /// adjoint uses the left null space of its owned parent without
    /// materializing the receiver. Checked results use the source provider instance, and a
    /// failure returns no tensor.
    ///
    /// The rank-revealing right null space (TensorKit
    /// `right_null(t; alg = :svd, trunc)`) is the composition shown for
    /// [`Self::left_null`] on `vh`'s bond: pad `s`'s diagonal to
    /// `vh.codomain()[0]` and restrict `vh`'s leg `0`.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use symtn::sector::{U1FusionRule, U1Irrep};
    /// use symtn::typed::{GradedSpace, Runtime, Svd, TensorMap, Truncation};
    ///
    /// let engine = &mut symtn::typed::CpuEngine::serial();
    /// let runtime = Runtime::builder().build()?;
    /// let rule = Arc::new(U1FusionRule);
    /// let short = GradedSpace::try_new(Arc::clone(&rule), [(U1Irrep::new(0), 2)])?;
    /// let wide = GradedSpace::try_new(rule, [(U1Irrep::new(0), 3)])?;
    /// let a: TensorMap<_, f64> = TensorMap::from_subblock_fn(engine,
    ///     &runtime, [&short], [&wide], |_, i| ((i[0] + 1) * (i[1] + 1)) as f64,
    /// )?;
    /// assert_eq!(a.right_null(engine, &[0], &[1])?.codomain()[0].degeneracy(&U1Irrep::new(0))?, 1);
    ///
    /// let Svd { s, vh, .. } = a.svd_full(engine, &[0], &[1])?;
    /// let bond = &vh.codomain()[0];
    /// let spectrum = bond.extend_spectrum(&s.diagview(engine)?)?;
    /// let rtol = f64::EPSILON.powf(2.0 / 3.0);
    /// let null = bond.find_truncated(&spectrum, &Truncation::below(0.0, rtol)?)?;
    /// let n = vh.restrict_leg(engine, &[(0, &null.selection)])?;
    /// assert_eq!(n.codomain()[0].degeneracy(&U1Irrep::new(0))?, 2);
    /// let n_adjoint = n.adjoint(engine)?;
    /// assert!(a.compose(engine, &n_adjoint)?.norm(engine, 2.0)? < 1e-12);
    /// # Ok::<(), symtn::typed::Error>(())
    /// ```
    ///
    /// `rows` and `cols` are the leg roles: the operation acts on the matrix
    /// view `self.permute(rows, cols)`, and the current split costs nothing
    /// extra (see [`Self::svd_compact`]'s *Leg roles*).
    pub fn right_null(
        &self,
        engine: &mut dyn Engine,
        rows: &[usize],
        cols: &[usize],
    ) -> Result<Self, FacadeError<R>> {
        self.with_leg_roles(engine, rows, cols, |t, engine| {
            t.factor_null(
                engine,
                FactorOp::RightNull,
                |parent, engine| {
                    parent.factor_null(
                        engine,
                        FactorOp::LeftNull,
                        |_, _| {
                            // A lazy adjoint's parent is owned, so it is
                            // never redirected again.
                            Err(
                                internal_layout_error("an owned null-space parent was redirected")
                                    .into(),
                            )
                        },
                        |engine, lease, source| {
                            symtn_matrixalgebra::seam::left_null_from_source::<R::Mode, _, _, _, _>(
                                engine, lease, source,
                            )
                        },
                    )
                },
                |engine, lease, source| {
                    symtn_matrixalgebra::seam::right_null_from_source::<R::Mode, _, _, _, _>(
                        engine, lease, source,
                    )
                },
            )
        })
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R> + TypedTensorTransformDispatch<R, D>,
    D: FactorizationScalar,
{
    /// Returns the left polar decomposition `self = w * p` as a [`LeftPolar`].
    ///
    /// `w` lives on the input space `codomain(self) <- domain(self)` and is an
    /// isometry (`w^H * w = id` on the domain). The positive-semidefinite
    /// factor `p : domain(self) <- domain(self)` is `V * S * V^H` from the
    /// compact SVD. Every coupled-sector block must be at least as tall as it
    /// is wide. [`Self::right_polar`] handles wide blocks.
    ///
    /// Dense-input cost is `O(sum_c m_c * n_c * min(m_c, n_c))` plus sectorwise
    /// composition. An owned compact diagonal stores both factors
    /// compactly; use [`Self::materialize`] for dense buffers. A lazy
    /// adjoint runs the opposite decomposition on its owned parent and returns
    /// detached owned factors without materializing the receiver. Checked factors
    /// use the same provider instance as `self`. If that provider rejects an
    /// output space or any sector computation fails, no factors are returned.
    /// After the block-shape check, a nonfinite entry returns
    /// [`Error::Operation`] (an `InvalidArgument`: `polar input components must be finite`).
    ///
    /// ```
    /// use std::sync::Arc;
    /// use symtn::sector::{U1FusionRule, U1Irrep};
    /// use symtn::typed::{GradedSpace, LeftPolar, Runtime, TensorMap};
    ///
    /// let engine = &mut symtn::typed::CpuEngine::serial();
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2)])?;
    /// let a: TensorMap<_, f64> = TensorMap::isomorphism(engine, &runtime, [&v], [&v])?.scale(engine, 2.0)?;
    /// let LeftPolar { w, p } = a.left_polar(engine, &[0], &[1])?;
    /// assert!(w.compose(engine, &p)?.axpby(engine, 1.0, &a, -1.0)?.norm(engine, 2.0)? < 1e-12);
    /// # Ok::<(), symtn::typed::Error>(())
    /// ```
    ///
    /// `rows` and `cols` are the leg roles: the operation acts on the matrix
    /// view `self.permute(rows, cols)`, and the current split costs nothing
    /// extra (see [`Self::svd_compact`]'s *Leg roles*).
    pub fn left_polar(
        &self,
        engine: &mut dyn Engine,
        rows: &[usize],
        cols: &[usize],
    ) -> Result<LeftPolar<Self>, FacadeError<R>> {
        self.with_leg_roles(engine, rows, cols, Self::factor_left_polar)
    }

    /// Returns the right polar decomposition `self = p * wh` as a
    /// [`RightPolar`].
    ///
    /// `p : codomain(self) <- codomain(self)` is positive semidefinite and
    /// `wh` lives on the input space as a coisometry
    /// (`wh * wh^H = id` on the codomain). Every coupled-sector block must be
    /// at least as wide as it is tall. Factor spaces are stated above;
    /// [`Self::left_polar`] describes the corresponding storage routes, lazy
    /// input handling, and cost. Checked factors use the source provider
    /// instance, and a failure returns no factors.
    ///
    /// `rows` and `cols` are the leg roles: the operation acts on the matrix
    /// view `self.permute(rows, cols)`, and the current split costs nothing
    /// extra (see [`Self::svd_compact`]'s *Leg roles*).
    pub fn right_polar(
        &self,
        engine: &mut dyn Engine,
        rows: &[usize],
        cols: &[usize],
    ) -> Result<RightPolar<Self>, FacadeError<R>> {
        self.with_leg_roles(engine, rows, cols, Self::factor_right_polar)
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R> + TypedAdjointSpace<R> + TypedTensorTransformDispatch<R, D>,
    D: AdvancedLinalgScalar,
{
    /// TensorKit 0.17 / MatrixAlgebraKit `inv`: the true inverse `t^-1` of a
    /// nonsingular map, defined by `t * t^-1 = id` on the codomain and
    /// `t^-1 * t = id` on the domain. Computed per coupled sector as the exact
    /// dense solve `t_c X_c = 1`, not as a spectral function — there is no
    /// truncation policy to apply and no factor tensor to build.
    ///
    /// # Domain
    ///
    /// TensorKit asks for `codomain ≅ domain` — **isomorphic, not equal** —
    /// and returns a map `domain <- codomain`. This facade's seam agrees: a
    /// rank-one codomain and a rank-two domain with the same coupled-sector
    /// dimensions are accepted, and the result carries the two spaces swapped.
    /// The pin is `inv_accepts_isomorphic_unequal_spaces_and_rejects_nonisomorphic`.
    ///
    /// # Errors
    ///
    /// - [`Error::Operation`] with `SpaceMismatch` when the two sides are not
    ///   isomorphic (TensorKit `SpaceMismatch`).
    /// - [`Error::Operation`] with `Dense(NumericalFailure)` when a
    ///   coupled-sector block is singular, in both storages: the dense solve
    ///   fails, and the compact arm finds a zero entry before any solve runs.
    ///   Never a panic; pinned by
    ///   `inv_of_a_singular_sector_fails_on_both_storages_and_keeps_the_source`.
    ///
    /// # Complexity
    ///
    /// Dense input: `O(Σ_c n_c³)`, one LU solve per coupled sector. Compact
    /// input (a spectrum factor, TensorKit's
    /// `DiagonalTensorMap`): the
    /// **O(rank) elementwise-reciprocal arm**, `1/s_i` over the `Σ_c k_c`
    /// stored values, and the result stays compact — matching TensorKit's
    /// `inv(::DiagonalTensorMap)`, which is `inv.(d.data)`. Nothing dense is
    /// built on either side of that arm. A host lazy adjoint solves its owned
    /// parent and returns a detached owned adjoint of that inverse; it does not
    /// allocate or publish a separate receiver-materialization payload.
    ///
    /// Checked Generic performs the same isomorphism preflight, admits the
    /// swapped output with the source provider `Arc` before output allocation
    /// or dense work, and preserves provider admission failures as typed errors.
    /// Every compact input, in any bond layout and fusion mode, takes the same
    /// elementwise-reciprocal arm, with the same zero-entry error and the same
    /// NaN/infinity pass-through, and stays compact on its own space.
    ///
    /// `rows` and `cols` are the leg roles: the operation acts on the matrix
    /// view `self.permute(rows, cols)`, and the current split costs nothing
    /// extra (see [`Self::svd_compact`]'s *Leg roles*).
    pub fn inv(
        &self,
        engine: &mut dyn Engine,
        rows: &[usize],
        cols: &[usize],
    ) -> Result<Self, FacadeError<R>> {
        self.with_leg_roles(engine, rows, cols, Self::factor_inv)
    }
}

impl<R, D> TensorMap<R, D>
where
    R: TypedSectorAdmission,
    R::Mode: FusionMode<R> + TypedTensorTransformDispatch<R, D>,
    D: AdvancedLinalgScalar,
{
    /// The matrix exponential `exp(t) = Σ_k t^k / k!`, evaluated per coupled
    /// sector — TensorKit's `exp`, which copies and calls `exp!`: check
    /// `domain == codomain`, then exponentiate every block.
    ///
    /// # Domain
    ///
    /// Any endomorphism, of any dtype. Multiplicity-free tensors retain the
    /// original two dense routes:
    ///
    /// - **Hermitian blocks** take the spectral function `V exp(D) Vᴴ` of the
    ///   Hermitian eigendecomposition.
    /// - **Everything else** takes blockwise scaling-and-squaring Padé [13/13]
    ///   (Higham 2005). Non-normal, defective and complex non-Hermitian blocks
    ///   are all in domain; nothing is symmetrized.
    ///
    /// The mathematical and blockwise contract matches the referenced
    /// implementation, but bitwise output does not. symtn's general dense route
    /// always uses Padé [13/13], while the referenced implementation delegates
    /// dense algorithm selection and may choose a lower degree for a small-norm
    /// block. The results agree only up to their numerical approximation; pinned
    /// source coordinates are recorded in `symtn/references.md`.
    ///
    /// The **compact** arm is TensorKit's
    /// `exp(::DiagonalTensorMap)`: unconditionally elementwise, with no
    /// hermiticity gate, for every compact input in any bond layout and
    /// fusion mode. Nonfinite entries map through `exp` without an error.
    ///
    /// # Errors
    ///
    /// - [`Error::Operation`] when the input is not an endomorphism
    ///   (`codomain != domain`), when a general block holds a nonfinite entry,
    ///   when a general block's column 1-norm overflows to infinity although
    ///   every entry is finite, or when the backend fails. Nothing is published
    ///   unless every coupled sector succeeded.
    /// - [`Error::Core`] / [`Error::FusionAlgebra`] from the multiplicity-free
    ///   Hermitian composition route.
    ///
    /// # Complexity
    ///
    /// Dense input is `O(Σ_c n_c³)`: multiplicity-free Hermitian blocks use one
    /// eigendecomposition plus a composition; all general blocks, including
    /// checked-Generic, use six GEMMs, one solve and the necessary Padé
    /// squarings per sector with `O(max_c n_c²)` workspace. Coupled sectors are
    /// never mixed. A dense lazy adjoint builds one operation-local logical
    /// payload per call, released with the call. Compact input remains
    /// `O(rank)` elementwise in every fusion mode.
    ///
    /// TensorKit's diagonal implementation is the reference for the compact
    /// branch; this method never panics for a supported tensor contract.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use symtn::sector::{U1FusionRule, U1Irrep};
    /// use symtn::typed::{GradedSpace, Runtime, TensorMap};
    /// let engine = &mut symtn::typed::CpuEngine::serial();
    /// let runtime = Runtime::builder().build()?;
    /// let v = GradedSpace::try_new(Arc::new(U1FusionRule), [(U1Irrep::new(0), 2), (U1Irrep::new(1), 1)])?;
    /// let zero: TensorMap<_, f64> = TensorMap::zeros(engine, &runtime, [&v], [&v])?;
    /// let id: TensorMap<_, f64> = TensorMap::isomorphism(engine, &runtime, [&v], [&v])?;
    /// assert!(zero.exp(engine, &[0], &[1])?.dense_data()?.iter().zip(id.dense_data()?).all(|(a, b)| (a - b).abs() < 1e-15));
    /// # Ok::<(), symtn::typed::Error>(())
    /// ```
    ///
    /// `rows` and `cols` are the leg roles: the operation acts on the matrix
    /// view `self.permute(rows, cols)`, and the current split costs nothing
    /// extra (see [`Self::svd_compact`]'s *Leg roles*).
    pub fn exp(
        &self,
        engine: &mut dyn Engine,
        rows: &[usize],
        cols: &[usize],
    ) -> Result<Self, FacadeError<R>> {
        self.with_leg_roles(engine, rows, cols, Self::factor_exp)
    }
}
