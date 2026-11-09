use num_complex::{Complex32, Complex64};

use crate::factor_dest::Element;
use crate::fresh::{dispatch_fresh, fresh_serial, FRESH_OP};
use crate::{
    DenseDType, DenseDotConfig, DenseError, DenseFreshSegments, DenseRead, DenseScalar,
    DenseTensor, DenseView, DenseViewMut, DenseWrite, Engine, FactorDest, FactorFailure,
};

/// One GEMM of a batched matmul over shared flat buffers: the column-major
/// `rows x cols` destination block at `dst_offset` receives
/// `alpha * lhs_block * rhs_block + beta * dst_block`. Offsets are element
/// offsets relative to the corresponding view's own offset. Callers guarantee
/// the destination blocks of a batch are pairwise disjoint, so executors may
/// run jobs in any order or concurrently.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenseGemmBatchJob {
    pub dst_offset: usize,
    pub lhs_offset: usize,
    pub rhs_offset: usize,
    pub rows: usize,
    pub contracted: usize,
    pub cols: usize,
}

/// Matrix interpretation for one operand of a rank-2 batch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MatrixOp {
    #[default]
    Identity,
    Transpose,
    Adjoint,
}

/// A host-owned column-major payload accepted by an executor's full-SVD entry
/// point.
pub enum DenseOwned {
    F32(Vec<f32>),
    F64(Vec<f64>),
    C32(Vec<Complex32>),
    C64(Vec<Complex64>),
}

/// The per-matrix factorization a [`DenseExecutor::factorize_batch`] issues.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DenseFactorization {
    /// [`DenseExecutor::svd`].
    Svd,
    /// [`DenseExecutor::qr`].
    Qr,
    /// [`DenseExecutor::eigh`].
    Eigh,
    /// [`DenseExecutor::eigh_vals`], delivered as a one-tensor output.
    EighVals,
    /// [`DenseExecutor::svd_vals`], delivered as a one-tensor output.
    SvdVals,
    /// [`DenseExecutor::eig`].
    Eig,
    /// [`DenseExecutor::eig_vals`], delivered as a one-tensor output.
    EigVals,
    /// Compact LQ `X = L Q`: `[L, Q]` of shapes `[m, k, b..]` and
    /// `[k, n, b..]` with `k = min(m, n)`, `Q` with orthonormal rows, in the
    /// provider's gauge. It has no per-matrix entry; the default factors
    /// each input's adjoint with [`DenseExecutor::qr`] (`X^H = Q' R'`) and
    /// returns `[R'^H, Q'^H]`.
    Lq,
    /// Full QR `X = Q R`: `[Q, R]` of shapes `[m, m, b..]` and `[m, n, b..]`,
    /// `Q` unitary and `R` upper trapezoidal, in the provider's gauge. Past
    /// the first `min(m, n)`, `Q`'s columns are an orthonormal basis of the
    /// complement and may differ by provider. It has no per-matrix entry;
    /// the default calls [`DenseExecutor::qr_into`] on `X` when `m <= n`,
    /// otherwise on the `m x (n + m)` block `[X I_m]`, keeping the first `n`
    /// columns of its `R`.
    QrFull,
    /// The shape-based null basis of `X`'s columns (MatrixAlgebraKit
    /// `qr_null`): `[N]` of shape `[m, m - min(m, n), b..]` with orthonormal
    /// columns orthogonal to every column of `X`, the trailing columns of a
    /// full QR's `Q`; any orthonormal basis of that complement, which may
    /// differ by provider. It has no per-matrix entry; the default calls
    /// [`DenseExecutor::qr_into`] on `[X I_m]` when `m > n` and keeps its
    /// `Q`'s last `m - n` columns, and solves nothing when `m <= n`.
    QrNull,
    /// Full SVD `X = U S Vh` (MatrixAlgebraKit `svd_full`): `[U, S, Vh]` of
    /// shapes `[m, m, b..]`, `[k, b..]` and `[n, n, b..]` with
    /// `k = min(m, n)`, `U` and `Vh` unitary, `S` real, non-negative and
    /// non-increasing, in the provider's gauge. Past the first `k`, `U`'s
    /// columns and `Vh`'s rows are an orthonormal basis of the complement
    /// and may differ by provider; an input with a zero extent has identity
    /// `U` and `Vh`. It has no per-matrix entry; the default factors one
    /// `m x n` matrix (a batched input is [`DenseError::Unsupported`]) with
    /// [`DenseExecutor::svd_full_owned`] of a compact copy when
    /// [`DenseExecutor::supports_svd_full`], otherwise with
    /// [`DenseExecutor::svd_into`] into compact `U1`, `S`, `Vh1` and
    /// completes `U` and `V` from `[U1 I_m]` and `[V1 I_n]` as
    /// [`Self::QrNull`]'s default does.
    SvdFull,
    /// Left polar decomposition `X = W P` of an `m x n` input with `m >= n`
    /// (MatrixAlgebraKit `left_polar!` via `PolarViaSVD`): `[W, P]` of shapes
    /// `[m, n, b..]` and `[n, n, b..]`, `W` with orthonormal columns and `P`
    /// Hermitian positive semidefinite. From the compact SVD `X = U S Vh`:
    /// `W = U Vh` and `P = (sqrt(S) Vh)^H (sqrt(S) Vh)`, projected exactly
    /// Hermitian. `P` is unique; `W` is unique only when `X` has full column
    /// rank and may otherwise differ by provider or lane width. Destination
    /// form only ([`DenseExecutor::factorize_batch_into`]); an input with
    /// `m < n` is refused. The default calls [`DenseExecutor::svd`] and then
    /// [`DenseExecutor::dot_general_into`] twice, straight into the
    /// destinations.
    LeftPolar,
    /// Right polar decomposition `X = P Wh` of an `m x n` input with
    /// `n >= m` (MatrixAlgebraKit `right_polar!`): `[P, Wh]` of shapes
    /// `[m, m, b..]` and `[m, n, b..]`, `Wh` with orthonormal rows and
    /// `P = (U sqrt(S)) (U sqrt(S))^H`, otherwise as [`Self::LeftPolar`]
    /// (`Wh` unique only for full row rank); an input with `n < m` is
    /// refused.
    RightPolar,
    /// Exponential of a Hermitian `n x n` input (MatrixAlgebraKit
    /// `exponential!` via `MatrixFunctionViaEigh`): `[exp]` of shape
    /// `[n, n, b..]`, exactly Hermitian. From the Hermitian eigendecomposition
    /// `X = V D V^H` (the provider reads one triangle, as for
    /// [`Self::Eigh`]): `exp(X) = Y Y^H` with `Y = V exp(D / 2)`, so a result
    /// is finite whenever it is representable, even where `exp(D)` alone
    /// would overflow. A nonfinite eigenvalue is the input's
    /// [`FactorFailure::Output`] (`NumericalFailure` with op
    /// [`crate::EIGH_EIGENVALUE_CHECK`]) and skips the product. Destination
    /// form only ([`DenseExecutor::factorize_batch_into`]); a non-square
    /// input is refused. The default calls [`DenseExecutor::eigh`] and then
    /// [`DenseExecutor::dot_general_into`] once, straight into the
    /// destination.
    ExpViaEigh,
}

impl DenseFactorization {
    /// Whether `self` has only a destination form, which the owned
    /// [`DenseExecutor::factorize_batch`] refuses.
    pub(crate) fn destination_only(self) -> bool {
        matches!(self, Self::LeftPolar | Self::RightPolar | Self::ExpViaEigh)
    }
}

/// The owned batch's refusal of a destination-only `op`.
pub(crate) fn refuse_destination_only(op: DenseFactorization) -> Result<(), DenseError> {
    if op.destination_only() {
        return Err(destination_only_error(op));
    }
    Ok(())
}

/// The owned batch's error for the destination-only `op`.
pub(crate) fn destination_only_error(op: DenseFactorization) -> DenseError {
    DenseError::Unsupported {
        op: "factorize_batch",
        message: format!("{op:?} has only a destination form"),
    }
}

mod sealed {
    /// Lends any sized executor as a trait object, so that
    /// [`super::DenseExecutor::with_linalg_scope`]'s default can pass `self`
    /// to its body even when it is called through
    /// `E: DenseExecutor + ?Sized`. Blanket-implemented; not nameable outside
    /// this crate.
    pub trait AsDynDenseExecutor {
        fn as_dyn_dense_executor(&mut self) -> &mut dyn super::DenseExecutor;
    }

    impl<T: super::DenseExecutor> AsDynDenseExecutor for T {
        fn as_dyn_dense_executor(&mut self) -> &mut dyn super::DenseExecutor {
            self
        }
    }
}

/// The body of a [`DenseExecutor::with_linalg_scope`].
pub type DenseLinalgScopeBody<'a> =
    dyn FnMut(&mut dyn DenseExecutor, &mut dyn Engine) -> Result<(), DenseError> + Send + 'a;

pub trait DenseExecutor: sealed::AsDynDenseExecutor {
    fn svd(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError>;
    fn qr(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError>;
    fn eigh(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError>;

    /// Runs `body` once with this executor, letting the sequence of per-matrix
    /// calls it makes share one backend admission where the executor has a
    /// per-call admission cost. A streaming per-block loop uses this instead of
    /// [`Self::factorize_batch`] when holding every block's output at once
    /// would raise its working set.
    ///
    /// The body runs exactly once, and its calls reach this executor in the
    /// order it makes them. It may run on another thread than the caller, so
    /// it must not rely on the caller's thread-local state. It must reach
    /// dense work only through the executor it is given: another executor, or
    /// work started on other threads that re-enters this one, can panic in the
    /// backend's re-entry guard. The body receives the same caller engine; it
    /// must pass that engine to every executor call it makes. The default runs
    /// `body(self, engine)` on the calling thread.
    fn with_linalg_scope(
        &mut self,
        engine: &mut dyn Engine,
        body: &mut DenseLinalgScopeBody<'_>,
    ) -> Result<(), DenseError> {
        body(self.as_dyn_dense_executor(), engine)
    }

    /// Applies `op` to each input and hands `sink` each delivered input's
    /// outcome, `Ok` holding that input's outputs exactly as the per-matrix
    /// entry returns them (a values-only entry's tensor as a one-element
    /// vector).
    ///
    /// Outcomes reach `sink` in ascending input order, each at most once. An
    /// `Err` from `sink` stops delivery and is returned. An executor-level
    /// failure (such as an engine the executor cannot use) is returned
    /// without calling `sink` further.
    ///
    /// The default solves input `i` with the per-matrix entry (for
    /// [`DenseFactorization::Lq`], [`Self::qr`] of the input's adjoint,
    /// written into one scratch sized for the largest input; for
    /// [`DenseFactorization::QrFull`] and [`DenseFactorization::QrNull`],
    /// [`Self::qr_into`] as each variant states; for
    /// [`DenseFactorization::SvdFull`], the entries that variant names) and
    /// calls
    /// `sink(i, outcome)` before solving input `i + 1`, so an executor using
    /// it sees the same call sequence as a per-matrix loop that checks each
    /// output before the next solve. An executor whose per-call entry carries
    /// a fixed admission cost overrides this to pay it once per batch; one
    /// may solve the inputs concurrently on the engine's lanes before
    /// delivering their outcomes in order.
    ///
    /// A destination-only `op` ([`DenseFactorization::LeftPolar`],
    /// [`DenseFactorization::RightPolar`], [`DenseFactorization::ExpViaEigh`])
    /// is [`DenseError::Unsupported`] before any solve.
    fn factorize_batch(
        &mut self,
        engine: &mut dyn Engine,
        op: DenseFactorization,
        inputs: &[DenseRead<'_>],
        sink: &mut dyn FnMut(usize, Result<Vec<DenseTensor>, DenseError>) -> Result<(), DenseError>,
    ) -> Result<(), DenseError> {
        refuse_destination_only(op)?;
        let mut lq = AdjointQrLq::for_inputs(inputs);
        for (index, &input) in inputs.iter().enumerate() {
            let outcome = match op {
                DenseFactorization::Svd => self.svd(engine, input),
                DenseFactorization::Qr => self.qr(engine, input),
                DenseFactorization::Eigh => self.eigh(engine, input),
                DenseFactorization::EighVals => self.eigh_vals(engine, input).map(|v| vec![v]),
                DenseFactorization::SvdVals => self.svd_vals(engine, input).map(|v| vec![v]),
                DenseFactorization::Eig => self.eig(engine, input),
                DenseFactorization::EigVals => self.eig_vals(engine, input).map(|v| vec![v]),
                DenseFactorization::Lq => lq.factor(input, |adjoint| self.qr(engine, adjoint)),
                DenseFactorization::QrFull => full_qr_by_qr_into(self, engine, input, false),
                DenseFactorization::QrNull => full_qr_by_qr_into(self, engine, input, true),
                DenseFactorization::SvdFull => full_svd_by_entries(self, engine, input),
                DenseFactorization::LeftPolar
                | DenseFactorization::RightPolar
                | DenseFactorization::ExpViaEigh => {
                    unreachable!("destination-only ops are refused above")
                }
            };
            sink(index, outcome)?;
        }
        Ok(())
    }

    /// [`Self::factorize_batch`] of the compact QR, compact LQ, full QR, QR
    /// null basis, compact or full SVD or Hermitian eigendecomposition `op`
    /// with each
    /// input's outputs written into its
    /// caller-owned destination `dests[i]` (see [`FactorDest`] for the
    /// shapes, dtypes and [`crate::FactorPlacement`]), and the
    /// destination-only left and right polar decompositions and Hermitian
    /// exponential; any other `op` is [`DenseError::Unsupported`].
    ///
    /// Every destination is validated before any input is solved or any
    /// destination written: the count, each destination's outputs, then
    /// each output's dtype ([`DenseError::DTypeMismatch`]), shape and
    /// compact column-major layout ([`DenseError::ShapeMismatch`]); a
    /// rejected batch leaves every destination untouched.
    ///
    /// `sink(i, &mut dests[i], outcome)` is called once per delivered input,
    /// in ascending input order, after that input's solve has completed; it
    /// may read and rewrite that destination in place. A solver failure is
    /// [`FactorFailure::Solver`], outputs that do not fit the destination
    /// [`FactorFailure::Output`]; after either, that destination's contents
    /// are unspecified. An `Err` from `sink` stops delivery and is returned;
    /// an executor-level failure is returned without calling `sink` further.
    ///
    /// The default runs [`Self::factorize_batch`] and copies each delivered
    /// `Ok` outcome into its destination (after the output checks the
    /// matrix-algebra caller makes) before calling `sink`, so an executor
    /// keeps exactly its own owned calls, scopes and stopping points. The
    /// polar decompositions, which have no owned form, instead run per
    /// input [`Self::svd`] (its outputs checked as `svd_into` checks them, a
    /// mismatch [`FactorFailure::Output`]) and two
    /// [`Self::dot_general_into`] calls writing the destinations directly,
    /// then call `sink` before the next input is solved; the Hermitian
    /// exponential likewise runs [`Self::eigh`] (checked as `eigh_into`
    /// checks it, then its eigenvalues' finiteness, each a
    /// [`FactorFailure::Output`]) and one [`Self::dot_general_into`]. An
    /// executor that solves the inputs on lanes may write the destinations
    /// directly and deliver every outcome after the lanes join.
    #[expect(
        clippy::type_complexity,
        reason = "the sink's index, destination and tagged outcome are the documented delivery contract"
    )]
    fn factorize_batch_into(
        &mut self,
        engine: &mut dyn Engine,
        op: DenseFactorization,
        inputs: &[DenseRead<'_>],
        dests: &mut [FactorDest<'_>],
        sink: &mut dyn FnMut(
            usize,
            &mut FactorDest<'_>,
            Result<(), FactorFailure>,
        ) -> Result<(), DenseError>,
    ) -> Result<(), DenseError> {
        crate::factor_dest::factorize_batch_into_default(self, engine, op, inputs, dests, sink)
    }

    /// Whether this executor supports native full SVD for every [`DenseOwned`]
    /// dtype. The default is false.
    fn supports_svd_full(&self) -> bool {
        false
    }

    /// Full-matrices SVD of a column-major owned `rows x cols` payload.
    ///
    /// On success, returns `(U, S, Vh)` with shapes `rows x rows`,
    /// `min(rows, cols)`, and `cols x cols`. `input` is consumed even when the
    /// operation returns an error. The default reports [`DenseError::Unsupported`]
    /// without validating the shape or payload. A capable implementation must
    /// check element-count overflow, exact payload length, then zero extents
    /// before constructing or calling its backend.
    fn svd_full_owned(
        &mut self,
        _engine: &mut dyn Engine,
        _input: DenseOwned,
        _rows: usize,
        _cols: usize,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        Err(DenseError::Unsupported {
            op: "svd_full_owned",
            message: "executor does not implement owned full-matrices SVD".to_string(),
        })
    }

    /// Solve `a * x = b` into caller-owned storage.
    fn solve_into(
        &mut self,
        _engine: &mut dyn Engine,
        _a: DenseRead<'_>,
        _b: DenseRead<'_>,
        _x: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        Err(DenseError::Unsupported {
            op: "solve_into",
            message: "executor does not implement dense linear solve".to_string(),
        })
    }

    fn svd_into(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
        u: DenseWrite<'_>,
        s: DenseWrite<'_>,
        vt: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        let outputs = self.svd(engine, input)?;
        if outputs.len() != 3 {
            return Err(arity_mismatch("svd_into", 3, outputs.len()));
        }
        copy_dense_tensor_into(&outputs[0], u, "svd_into")?;
        copy_dense_tensor_into(&outputs[1], s, "svd_into")?;
        copy_dense_tensor_into(&outputs[2], vt, "svd_into")
    }

    fn qr_into(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
        q: DenseWrite<'_>,
        r: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        let outputs = self.qr(engine, input)?;
        if outputs.len() != 2 {
            return Err(arity_mismatch("qr_into", 2, outputs.len()));
        }
        copy_dense_tensor_into(&outputs[0], q, "qr_into")?;
        copy_dense_tensor_into(&outputs[1], r, "qr_into")
    }

    fn eigh_into(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
        values: DenseWrite<'_>,
        vectors: DenseWrite<'_>,
    ) -> Result<(), DenseError> {
        let outputs = self.eigh(engine, input)?;
        if outputs.len() != 2 {
            return Err(arity_mismatch("eigh_into", 2, outputs.len()));
        }
        copy_dense_tensor_into(&outputs[0], values, "eigh_into")?;
        copy_dense_tensor_into(&outputs[1], vectors, "eigh_into")
    }

    /// General (non-Hermitian) eigendecomposition `(values, vectors)`; both
    /// outputs are complex regardless of the input scalar. The default reports
    /// [`DenseError::Unsupported`] without inspecting the input.
    fn eig(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        let _ = (engine, input);
        Err(DenseError::Unsupported {
            op: "eig",
            message: "executor does not implement the general eigendecomposition".to_string(),
        })
    }

    /// Singular values only: the length-`min(m, n)` real `S`, without computing
    /// `U`/`Vt` (LAPACK `job='N'`, MatrixAlgebraKit `svd_vals`). The default
    /// computes the full SVD and drops the vectors — correct but wasteful;
    /// backends with a no-vector LAPACK path override this.
    fn svd_vals(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<DenseTensor, DenseError> {
        let mut outputs = self.svd(engine, input)?;
        if outputs.len() != 3 {
            return Err(arity_mismatch("svd_vals", 3, outputs.len()));
        }
        Ok(outputs.swap_remove(1))
    }

    /// Hermitian eigenvalues only, without eigenvectors (LAPACK `job='N'`,
    /// MatrixAlgebraKit `eigh_vals`). Default drops the vectors from the full
    /// decomposition; backends override with a no-vector path.
    fn eigh_vals(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<DenseTensor, DenseError> {
        let mut outputs = self.eigh(engine, input)?;
        if outputs.len() != 2 {
            return Err(arity_mismatch("eigh_vals", 2, outputs.len()));
        }
        Ok(outputs.swap_remove(0))
    }

    /// General (non-Hermitian) eigenvalues only, without eigenvectors (LAPACK
    /// `job='N'`, MatrixAlgebraKit `eig_vals`); complex regardless of input
    /// scalar. Default drops the vectors; backends override with a no-vector
    /// path.
    fn eig_vals(
        &mut self,
        engine: &mut dyn Engine,
        input: DenseRead<'_>,
    ) -> Result<DenseTensor, DenseError> {
        let mut outputs = self.eig(engine, input)?;
        if outputs.len() != 2 {
            return Err(arity_mismatch("eig_vals", 2, outputs.len()));
        }
        Ok(outputs.swap_remove(0))
    }

    fn dot_general_into(
        &mut self,
        engine: &mut dyn Engine,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        config: &DenseDotConfig,
    ) -> Result<(), DenseError>;

    fn matmul_into(
        &mut self,
        engine: &mut dyn Engine,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
    ) -> Result<(), DenseError> {
        self.dot_general_into(engine, output, lhs, rhs, &DenseDotConfig::matmul())
    }

    /// Accumulate-form matmul: `output = alpha * lhs * rhs + beta * output`
    /// (BLAS gemm semantics). The default supports only the overwrite case
    /// `alpha = 1, beta = 0`, which it delegates to [`Self::matmul_into`]; any
    /// other `alpha`/`beta` reports [`DenseError::Unsupported`].
    /// Accumulate-capable backends override it.
    fn matmul_axpby_into(
        &mut self,
        engine: &mut dyn Engine,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        alpha: DenseScalar,
        beta: DenseScalar,
    ) -> Result<(), DenseError> {
        if alpha.is_one() && beta.is_zero() {
            return self.matmul_into(engine, output, lhs, rhs);
        }
        Err(DenseError::Unsupported {
            op: "matmul_axpby_into",
            message: "executor does not implement the accumulate-form matmul".to_string(),
        })
    }

    /// Batched accumulate-form matmul over shared flat buffers: for each job,
    /// the destination block receives `alpha * lhs_block * rhs_block + beta *
    /// dst_block` (column-major, BLAS gemm semantics; see
    /// [`DenseGemmBatchJob`]). The default executes the jobs serially through
    /// `matmul_axpby_into`; batch-capable backends override it.
    ///
    /// `runs` is the plan-time run partition of `jobs` (see
    /// [`strided_batch_runs`] and issue #103): consecutive run lengths summing
    /// to `jobs.len()`. It is a routing hint only: the result is the per-job
    /// GEMM above for every `runs`. Backends that route runs differently read
    /// it to avoid recomputing the partition per replay, but must check that a
    /// run they batch is one same-shape, constant-stride run and otherwise
    /// execute its jobs individually; the serial default ignores it.
    #[expect(
        clippy::too_many_arguments,
        reason = "the dense backend boundary keeps three buffers, batch metadata, and BLAS alpha/beta explicit"
    )]
    fn matmul_batch_axpby_into(
        &mut self,
        engine: &mut dyn Engine,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        jobs: &[DenseGemmBatchJob],
        runs: &[usize],
        alpha: DenseScalar,
        beta: DenseScalar,
    ) -> Result<(), DenseError> {
        let _ = runs;
        let (out_dtype, lhs_dtype, rhs_dtype) = (output.dtype(), lhs.dtype(), rhs.dtype());
        match (output, lhs, rhs) {
            (DenseWrite::F32(out), DenseRead::F32(lhs), DenseRead::F32(rhs)) => {
                matmul_batch_axpby_serial(
                    self,
                    engine,
                    out,
                    lhs,
                    rhs,
                    jobs,
                    alpha,
                    beta,
                    |view: DenseViewMut<'_, f32>| DenseWrite::F32(view),
                    |view: DenseView<'_, f32>| DenseRead::F32(view),
                )
            }
            (DenseWrite::F64(out), DenseRead::F64(lhs), DenseRead::F64(rhs)) => {
                matmul_batch_axpby_serial(
                    self,
                    engine,
                    out,
                    lhs,
                    rhs,
                    jobs,
                    alpha,
                    beta,
                    |view: DenseViewMut<'_, f64>| DenseWrite::F64(view),
                    |view: DenseView<'_, f64>| DenseRead::F64(view),
                )
            }
            (DenseWrite::C32(out), DenseRead::C32(lhs), DenseRead::C32(rhs)) => {
                matmul_batch_axpby_serial(
                    self,
                    engine,
                    out,
                    lhs,
                    rhs,
                    jobs,
                    alpha,
                    beta,
                    |view: DenseViewMut<'_, Complex32>| DenseWrite::C32(view),
                    |view: DenseView<'_, Complex32>| DenseRead::C32(view),
                )
            }
            (DenseWrite::C64(out), DenseRead::C64(lhs), DenseRead::C64(rhs)) => {
                matmul_batch_axpby_serial(
                    self,
                    engine,
                    out,
                    lhs,
                    rhs,
                    jobs,
                    alpha,
                    beta,
                    |view: DenseViewMut<'_, Complex64>| DenseWrite::C64(view),
                    |view: DenseView<'_, Complex64>| DenseRead::C64(view),
                )
            }
            _ => Err(batch_operand_error(
                "matmul_batch_axpby_into",
                out_dtype,
                lhs_dtype,
                rhs_dtype,
            )),
        }
    }

    /// Batched rank-2 multiplication with one matrix interpretation shared by
    /// every job for each operand.
    #[allow(clippy::too_many_arguments)]
    fn matmul_batch_axpby_with_ops_into(
        &mut self,
        engine: &mut dyn Engine,
        output: DenseWrite<'_>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        jobs: &[DenseGemmBatchJob],
        runs: &[usize],
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
        alpha: DenseScalar,
        beta: DenseScalar,
    ) -> Result<(), DenseError> {
        if lhs_op == MatrixOp::Identity && rhs_op == MatrixOp::Identity {
            return self.matmul_batch_axpby_into(engine, output, lhs, rhs, jobs, runs, alpha, beta);
        }
        Err(DenseError::Unsupported {
            op: "matmul_batch_axpby_with_ops_into",
            message: "executor does not implement transpose/adjoint rank-2 batches".to_string(),
        })
    }

    /// Batched overwrite matmul into uninitialized destination memory: each
    /// job's column-major `rows x cols` output becomes
    /// `alpha * op(lhs_block) op(rhs_block)` with `op = ops[0], ops[1]`
    /// applied as in [`Self::matmul_batch_axpby_with_ops_into`], written into
    /// its piece of `segments` (see [`DenseFreshSegments`] for the layout
    /// contract). No prior destination value is read.
    ///
    /// On `Ok`, every job has been reported exactly once through the
    /// carrier's `written` callback with its own piece, in job order, after
    /// every job finished; a job with an empty output is reported with an
    /// empty slice. On `Err`, no job is reported. The default zero-fills each
    /// job's piece through the carrier's `zeroed` callback and runs it as a
    /// one-job [`Self::matmul_batch_axpby_with_ops_into`] with `beta = 0`.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    fn matmul_batch_fresh<'a>(
        &mut self,
        engine: &mut dyn Engine,
        segments: DenseFreshSegments<'_, 'a>,
        lhs: DenseRead<'_>,
        rhs: DenseRead<'_>,
        jobs: &[DenseGemmBatchJob],
        ops: [MatrixOp; 2],
        alpha: DenseScalar,
    ) -> Result<(), DenseError> {
        dispatch_fresh!(FRESH_OP, segments, lhs, rhs, |parts, lhs, rhs| {
            fresh_serial(self, engine, parts, lhs, rhs, jobs, ops, alpha)
        })
    }

    /// Whether this executor's batch entries carry out a forced engine batch
    /// policy (`Inner`, `Outer`, `Split`) as the engine's batch decision
    /// places it. A replay that cannot honor such a policy through its
    /// executor refuses it rather than running another placement. The BLAS
    /// provider and every custom executor answer `false`.
    #[doc(hidden)]
    fn honors_forced_batch_policy(&self) -> bool {
        false
    }
}

/// Executor-returned output count disagrees with the decomposition's arity.
///
/// Shared with the matrix-algebra crate: a short or long executor answer is
/// symtn's own check, so it is a [`DenseError::ShapeMismatch`] wherever it is
/// detected, never a backend error.
pub fn arity_mismatch(op: &'static str, expected: usize, actual: usize) -> DenseError {
    DenseError::ShapeMismatch {
        op,
        expected: vec![expected],
        actual: vec![actual],
    }
}

/// Batched matmul operands must share one scalar dtype; report the first
/// operand that disagrees with the destination, or `Unsupported` when all
/// agree on a dtype the batched route does not cover (integers, bool).
pub(crate) fn batch_operand_error(
    op: &'static str,
    out: DenseDType,
    lhs: DenseDType,
    rhs: DenseDType,
) -> DenseError {
    match [lhs, rhs].into_iter().find(|&d| d != out) {
        Some(actual) => DenseError::DTypeMismatch {
            op,
            expected: out,
            actual,
        },
        None => DenseError::Unsupported {
            op,
            message: format!("batched matmul requires f32/f64/c32/c64 operands, got {out:?}"),
        },
    }
}

fn copy_dense_tensor_into(
    tensor: &DenseTensor,
    output: DenseWrite<'_>,
    op: &'static str,
) -> Result<(), DenseError> {
    if tensor.dtype() != output.dtype() {
        return Err(DenseError::DTypeMismatch {
            op,
            expected: tensor.dtype(),
            actual: output.dtype(),
        });
    }
    match output {
        DenseWrite::F32(output) => {
            copy_contiguous_tensor_into_view(tensor.as_f32_slice()?, tensor.shape(), output, op)
        }
        DenseWrite::F64(output) => {
            copy_contiguous_tensor_into_view(tensor.as_f64_slice()?, tensor.shape(), output, op)
        }
        DenseWrite::C32(output) => {
            copy_contiguous_tensor_into_view(tensor.as_c32_slice()?, tensor.shape(), output, op)
        }
        DenseWrite::C64(output) => {
            copy_contiguous_tensor_into_view(tensor.as_c64_slice()?, tensor.shape(), output, op)
        }
        DenseWrite::I32(_) | DenseWrite::I64(_) | DenseWrite::Bool(_) => {
            Err(DenseError::Unsupported {
                op,
                message: format!("{:?} factorization outputs", tensor.dtype()),
            })
        }
    }
}

/// Checks an executor-returned contiguous output against the destination
/// shape the caller derived, returning its element count.
///
/// Shared with the matrix-algebra crate: the comparison is symtn's own, so a
/// mismatch is [`DenseError::ShapeMismatch`] (expected = destination), never a
/// backend error.
pub fn check_contiguous_output(
    op: &'static str,
    source_len: usize,
    source_shape: &[usize],
    expected_shape: &[usize],
) -> Result<usize, DenseError> {
    if source_shape != expected_shape {
        return Err(DenseError::ShapeMismatch {
            op,
            expected: expected_shape.to_vec(),
            actual: source_shape.to_vec(),
        });
    }
    let expected =
        source_shape
            .iter()
            .try_fold(1usize, |acc, &dim| match acc.checked_mul(dim) {
                Some(count) => Ok(count),
                None => Err(DenseError::ElementCountOverflow),
            })?;
    if source_len != expected {
        return Err(DenseError::ShapeMismatch {
            op,
            expected: vec![expected],
            actual: vec![source_len],
        });
    }
    Ok(expected)
}

pub(crate) fn copy_contiguous_tensor_into_view<T: Copy>(
    source: &[T],
    source_shape: &[usize],
    mut output: DenseViewMut<'_, T>,
    op: &'static str,
) -> Result<(), DenseError> {
    let expected = check_contiguous_output(op, source.len(), source_shape, output.shape())?;
    if expected == 0 {
        return Ok(());
    }
    if source_shape.is_empty() {
        let offset = output.offset();
        output.data_mut()[offset] = source[0];
        return Ok(());
    }

    let shape = output.shape().to_vec();
    let strides = output.strides().to_vec();
    let offset = output.offset();
    let run = shape[0];
    let outer_count = shape[1..].iter().product::<usize>();
    let mut index = vec![0usize; shape.len()];
    let data = output.data_mut();
    for outer in 0..outer_count {
        let src_start = outer * run;
        let mut dst_start = offset;
        for axis in 1..shape.len() {
            dst_start += index[axis] * strides[axis];
        }
        if strides[0] == 1 {
            data[dst_start..dst_start + run].copy_from_slice(&source[src_start..src_start + run]);
        } else {
            for lane in 0..run {
                data[dst_start + lane * strides[0]] = source[src_start + lane];
            }
        }
        for axis in 1..shape.len() {
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
    Ok(())
}

pub(crate) fn batch_offset(base: usize, offset: usize) -> Result<usize, DenseError> {
    match base.checked_add(offset) {
        Some(end) => Ok(end),
        None => Err(DenseError::OffsetOverflow { value: offset }),
    }
}

fn same_gemm_shape(lhs: &DenseGemmBatchJob, rhs: &DenseGemmBatchJob) -> bool {
    lhs.rows == rhs.rows && lhs.contracted == rhs.contracted && lhs.cols == rhs.cols
}

pub(crate) fn strided_batch_run_len(jobs: &[DenseGemmBatchJob], start: usize) -> usize {
    let Some(first) = jobs.get(start) else {
        return 0;
    };
    let Some(second) = jobs.get(start + 1) else {
        return 1;
    };
    if !same_gemm_shape(first, second) {
        return 1;
    }
    let Some(dst_stride) = second.dst_offset.checked_sub(first.dst_offset) else {
        return 1;
    };
    if dst_stride == 0 {
        return 1;
    }
    let Some(lhs_stride) = second.lhs_offset.checked_sub(first.lhs_offset) else {
        return 1;
    };
    let Some(rhs_stride) = second.rhs_offset.checked_sub(first.rhs_offset) else {
        return 1;
    };

    let mut len = 2usize;
    while let Some(next) = jobs.get(start + len) {
        let prev = &jobs[start + len - 1];
        if !same_gemm_shape(first, next) {
            break;
        }
        if prev.dst_offset.checked_add(dst_stride) != Some(next.dst_offset)
            || prev.lhs_offset.checked_add(lhs_stride) != Some(next.lhs_offset)
            || prev.rhs_offset.checked_add(rhs_stride) != Some(next.rhs_offset)
        {
            break;
        }
        len += 1;
    }
    len
}

/// Plan-time run partition of a batch: the maximal same-shape, constant-stride
/// run lengths over `jobs`, in order, summing to `jobs.len()`. A run is a
/// contiguous sequence of jobs with identical GEMM shape whose lhs/rhs/dst
/// offsets advance by a constant stride (exactly what the strided-batch seam
/// can dispatch as one call); a shape or stride break, or a singleton, ends the
/// run at length 1.
///
/// This is a backend-agnostic shape fact — it depends only on job shapes and
/// offsets, not on which dense backend runs the batch — so the plan layer
/// computes it once when a batch plan is compiled and stores it alongside the
/// jobs. The executor then reads the partition to route each run (see issue
/// #103) without recomputing it on every replay.
pub fn strided_batch_runs(jobs: &[DenseGemmBatchJob]) -> Vec<usize> {
    let mut runs = Vec::new();
    strided_batch_runs_into(jobs, &mut runs);
    runs
}

/// Whether a batch of `jobs` (partitioned into `runs`) is one strided
/// submission: one job, or one affine run of at least two jobs whose
/// destination step covers a whole block (the O(1) proof that the rank-3
/// destination is self-disjoint). The executor runs such a batch on the
/// strided entry, which places no blocks on lanes; an op-bearing run also
/// needs a layout its views admit.
pub(crate) fn single_strided_batch(jobs: &[DenseGemmBatchJob], runs: &[usize]) -> bool {
    jobs.len() == 1
        || runs == [jobs.len()]
            && jobs.len() >= 2
            && strided_batch_run_len(jobs, 0) == jobs.len()
            && jobs[1]
                .dst_offset
                .checked_sub(jobs[0].dst_offset)
                .zip(jobs[0].rows.checked_mul(jobs[0].cols))
                .is_some_and(|(dst_step, block)| dst_step >= block)
}

/// Writes [`strided_batch_runs`] into caller-owned scratch.
pub fn strided_batch_runs_into(jobs: &[DenseGemmBatchJob], runs: &mut Vec<usize>) {
    runs.clear();
    let mut start = 0usize;
    while start < jobs.len() {
        let run_len = strided_batch_run_len(jobs, start);
        runs.push(run_len);
        start += run_len;
    }
}

/// Serial fallback for [`DenseExecutor::matmul_batch_axpby_into`]: one
/// `matmul_axpby_into` per job over rank-2 sub-views of the shared buffers.
#[allow(clippy::too_many_arguments)]
fn matmul_batch_axpby_serial<E, T, W, R>(
    executor: &mut E,
    engine: &mut dyn Engine,
    mut output: DenseViewMut<'_, T>,
    lhs: DenseView<'_, T>,
    rhs: DenseView<'_, T>,
    jobs: &[DenseGemmBatchJob],
    alpha: DenseScalar,
    beta: DenseScalar,
    wrap_write: W,
    wrap_read: R,
) -> Result<(), DenseError>
where
    E: DenseExecutor + ?Sized,
    W: for<'x> Fn(DenseViewMut<'x, T>) -> DenseWrite<'x>,
    R: for<'x> Fn(DenseView<'x, T>) -> DenseRead<'x>,
{
    for job in jobs {
        let lhs_shape = [job.rows, job.contracted];
        let lhs_strides = [1, job.rows];
        let rhs_shape = [job.contracted, job.cols];
        let rhs_strides = [1, job.contracted];
        let dst_shape = [job.rows, job.cols];
        let dst_strides = [1, job.rows];
        let lhs_view = DenseView::new(
            lhs.data(),
            &lhs_shape,
            &lhs_strides,
            batch_offset(lhs.offset(), job.lhs_offset)?,
        )?;
        let rhs_view = DenseView::new(
            rhs.data(),
            &rhs_shape,
            &rhs_strides,
            batch_offset(rhs.offset(), job.rhs_offset)?,
        )?;
        let dst_offset = batch_offset(output.offset(), job.dst_offset)?;
        let dst_view = DenseViewMut::new(output.data_mut(), &dst_shape, &dst_strides, dst_offset)?;
        executor.matmul_axpby_into(
            engine,
            wrap_write(dst_view),
            wrap_read(lhs_view),
            wrap_read(rhs_view),
            alpha,
            beta,
        )?;
    }
    Ok(())
}

/// [`DenseFactorization::Lq`] through the QR of each input's adjoint: `X^H`
/// (column-major `[n, m, b..]`) is written into one scratch sized for the
/// largest input of the batch, `qr` factors it as `X^H = Q' R'`, and the
/// outcome is `[L, Q] = [R'^H, Q'^H]`, the outputs checked against the
/// adjoint's shapes before they are adjointed. The executor default and
/// providers without a transposed-read LQ use it.
pub(crate) struct AdjointQrLq {
    largest: usize,
    scratch: AdjointScratch,
}

#[derive(Default)]
struct AdjointScratch {
    f32: Vec<f32>,
    f64: Vec<f64>,
    c32: Vec<Complex32>,
    c64: Vec<Complex64>,
}

impl AdjointQrLq {
    /// Allocates nothing until an input is staged.
    pub(crate) fn for_inputs(inputs: &[DenseRead<'_>]) -> Self {
        let largest = inputs
            .iter()
            .filter_map(|input| checked_count(input.shape()).ok())
            .max()
            .unwrap_or(0);
        Self {
            largest,
            scratch: AdjointScratch::default(),
        }
    }

    /// The LQ of one input through `qr` of its adjoint. An input the staging
    /// cannot read (rank below 2, or a dtype without a conjugate) goes to
    /// `qr` unchanged, so it is refused with `qr`'s own error.
    pub(crate) fn factor(
        &mut self,
        input: DenseRead<'_>,
        qr: impl FnOnce(DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        let largest = self.largest;
        let scratch = &mut self.scratch;
        match input {
            DenseRead::F32(view) if view.shape().len() >= 2 => {
                lq_of_adjoint(view, &mut scratch.f32, largest, qr)
            }
            DenseRead::F64(view) if view.shape().len() >= 2 => {
                lq_of_adjoint(view, &mut scratch.f64, largest, qr)
            }
            DenseRead::C32(view) if view.shape().len() >= 2 => {
                lq_of_adjoint(view, &mut scratch.c32, largest, qr)
            }
            DenseRead::C64(view) if view.shape().len() >= 2 => {
                lq_of_adjoint(view, &mut scratch.c64, largest, qr)
            }
            other => {
                qr(other)?;
                Err(DenseError::Unsupported {
                    op: "lq",
                    message: format!(
                        "LQ input of dtype {:?} and rank {}",
                        other.dtype(),
                        other.shape().len()
                    ),
                })
            }
        }
    }
}

/// A scalar the LQ and full-QR defaults stage: its conjugate, its typed
/// reads and writes, and its owned outputs.
trait AdjointScalar: Copy {
    const ZERO: Self;
    const ONE: Self;
    fn conj(self) -> Self;
    fn read(view: DenseView<'_, Self>) -> DenseRead<'_>;
    fn write(view: DenseViewMut<'_, Self>) -> DenseWrite<'_>;
    fn owned(shape: Vec<usize>, data: Vec<Self>) -> Result<DenseTensor, DenseError>;
    fn owned_input(data: Vec<Self>) -> DenseOwned;
    fn slice(tensor: &DenseTensor) -> Result<&[Self], DenseError>;
    fn tensor(
        like: &DenseTensor,
        shape: Vec<usize>,
        data: Vec<Self>,
    ) -> Result<DenseTensor, DenseError>;
}

macro_rules! adjoint_scalar {
    ($ty:ty, $variant:ident, $slice:ident, $zero:expr, $one:expr, |$value:ident| $conj:expr) => {
        impl AdjointScalar for $ty {
            const ZERO: Self = $zero;
            const ONE: Self = $one;
            fn conj(self) -> Self {
                let $value = self;
                $conj
            }
            fn read(view: DenseView<'_, Self>) -> DenseRead<'_> {
                DenseRead::$variant(view)
            }
            fn write(view: DenseViewMut<'_, Self>) -> DenseWrite<'_> {
                DenseWrite::$variant(view)
            }
            fn owned(shape: Vec<usize>, data: Vec<Self>) -> Result<DenseTensor, DenseError> {
                DenseTensor::host_col_major(shape, data)
            }
            fn owned_input(data: Vec<Self>) -> DenseOwned {
                DenseOwned::$variant(data)
            }
            fn slice(tensor: &DenseTensor) -> Result<&[Self], DenseError> {
                tensor.$slice()
            }
            fn tensor(
                like: &DenseTensor,
                shape: Vec<usize>,
                data: Vec<Self>,
            ) -> Result<DenseTensor, DenseError> {
                DenseTensor::host_col_major_like(like, shape, data)
            }
        }
    };
}

adjoint_scalar!(f32, F32, as_f32_slice, 0.0, 1.0, |value| value);
adjoint_scalar!(f64, F64, as_f64_slice, 0.0, 1.0, |value| value);
adjoint_scalar!(
    Complex32,
    C32,
    as_c32_slice,
    Complex32::new(0.0, 0.0),
    Complex32::new(1.0, 0.0),
    |value| num_complex::Complex::conj(&value)
);
adjoint_scalar!(
    Complex64,
    C64,
    as_c64_slice,
    Complex64::new(0.0, 0.0),
    Complex64::new(1.0, 0.0),
    |value| num_complex::Complex::conj(&value)
);

/// Writes the adjoint of each `m x n` item of the nonempty strided
/// `[m, n, b..]` `view` into `output` as column-major `[n, m, b..]`.
fn extend_adjoint_items<T: AdjointScalar>(
    view: DenseView<'_, T>,
    output: &mut Vec<T>,
) -> Result<(), DenseError> {
    let (shape, strides, data) = (view.shape(), view.strides(), view.data());
    let (m, n) = (shape[0], shape[1]);
    // A broadcast (zero-stride) view can describe more items than `usize`
    // counts.
    let items = checked_count(&shape[2..])?;
    for item in 0..items {
        let mut base = view.offset();
        let mut rest = item;
        for (&extent, &stride) in shape[2..].iter().zip(&strides[2..]) {
            base += rest % extent * stride;
            rest /= extent;
        }
        for row in 0..m {
            output
                .extend((0..n).map(|col| data[base + row * strides[0] + col * strides[1]].conj()));
        }
    }
    Ok(())
}

/// The product of `extents`, or [`DenseError::ElementCountOverflow`].
fn checked_count(extents: &[usize]) -> Result<usize, DenseError> {
    extents
        .iter()
        .try_fold(1usize, |count, &extent| count.checked_mul(extent))
        .ok_or(DenseError::ElementCountOverflow)
}

/// The adjoint of each column-major `rows x cols` item of `data`, column-major
/// `cols x rows` per item.
fn adjoint_items<T: AdjointScalar>(data: &[T], rows: usize, cols: usize) -> Vec<T> {
    let mut adjoint = Vec::with_capacity(data.len());
    if rows * cols == 0 {
        return adjoint;
    }
    for item in data.chunks_exact(rows * cols) {
        for row in 0..rows {
            adjoint.extend((0..cols).map(|col| item[row + rows * col].conj()));
        }
    }
    adjoint
}

fn lq_of_adjoint<T: AdjointScalar>(
    view: DenseView<'_, T>,
    scratch: &mut Vec<T>,
    largest: usize,
    qr: impl FnOnce(DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError>,
) -> Result<Vec<DenseTensor>, DenseError> {
    let shape = view.shape();
    let (m, n, batch) = (shape[0], shape[1], &shape[2..]);
    let k = m.min(n);
    let dims = |a: usize, b: usize| [&[a, b][..], batch].concat();
    let adjoint_shape = dims(n, m);
    scratch.clear();
    // An empty input stages nothing; its empty adjoint still goes to `qr`, so
    // QR owns any refusal of its extents. A nonempty input's element count
    // and column-major strides must fit `usize`.
    let empty = shape.contains(&0);
    if !empty {
        if scratch.capacity() == 0 {
            scratch.reserve(largest);
        }
        extend_adjoint_items(view, scratch)?;
        #[cfg(test)]
        crate::probe::add(crate::probe::Event::LqAdjointStaging, 1);
    }
    let mut adjoint_strides = Vec::with_capacity(adjoint_shape.len());
    let mut stride = 1usize;
    for &extent in &adjoint_shape {
        adjoint_strides.push(stride);
        stride = match stride.checked_mul(extent) {
            Some(next) => next,
            None if empty => 0,
            None => return Err(DenseError::ElementCountOverflow),
        };
    }
    let adjoint = DenseView::new(scratch, &adjoint_shape, &adjoint_strides, 0)?;
    let outputs = qr(T::read(adjoint))?;
    if outputs.len() != 2 {
        return Err(arity_mismatch("qr_into", 2, outputs.len()));
    }
    let (q_shape, r_shape) = (dims(n, k), dims(k, m));
    let q_prime = T::slice(&outputs[0])?;
    check_contiguous_output("qr_into", q_prime.len(), outputs[0].shape(), &q_shape)?;
    let r_prime = T::slice(&outputs[1])?;
    check_contiguous_output("qr_into", r_prime.len(), outputs[1].shape(), &r_shape)?;
    Ok(vec![
        T::tensor(&outputs[1], dims(m, k), adjoint_items(r_prime, k, m))?,
        T::tensor(&outputs[0], dims(k, n), adjoint_items(q_prime, n, k))?,
    ])
}

/// [`DenseFactorization::QrFull`] (`null` false) or
/// [`DenseFactorization::QrNull`] (`null` true) of one input through
/// `executor`'s [`DenseExecutor::qr_into`], as the variants document: `X`
/// itself when `m <= n`, else the `m x (n + m)` block `[X I_m]` per batch
/// item, whose `Q` is square and whose upper-trapezoidal `R` makes `Q`'s
/// first `n` columns span a superspace of `X`'s columns whatever its rank.
/// An input the staging cannot read (rank below 2, or a dtype without a
/// factorization) goes to [`DenseExecutor::qr`] unchanged, so it is refused
/// with `qr`'s own error.
pub(crate) fn full_qr_by_qr_into<E: DenseExecutor + ?Sized>(
    executor: &mut E,
    engine: &mut dyn Engine,
    input: DenseRead<'_>,
    null: bool,
) -> Result<Vec<DenseTensor>, DenseError> {
    match input {
        DenseRead::F32(view) if view.shape().len() >= 2 => full_qr_of(executor, engine, view, null),
        DenseRead::F64(view) if view.shape().len() >= 2 => full_qr_of(executor, engine, view, null),
        DenseRead::C32(view) if view.shape().len() >= 2 => full_qr_of(executor, engine, view, null),
        DenseRead::C64(view) if view.shape().len() >= 2 => full_qr_of(executor, engine, view, null),
        other => {
            executor.qr(engine, other)?;
            Err(DenseError::Unsupported {
                op: "qr_full",
                message: format!(
                    "full QR input of dtype {:?} and rank {}",
                    other.dtype(),
                    other.shape().len()
                ),
            })
        }
    }
}

/// Compact column-major strides of `shape`; a nonempty payload's element
/// count must fit `usize` (an empty one never addresses its strides).
fn col_major_strides(shape: &[usize]) -> Result<Vec<usize>, DenseError> {
    let empty = shape.contains(&0);
    let mut strides = Vec::with_capacity(shape.len());
    let mut stride = 1usize;
    for &extent in shape {
        strides.push(stride);
        stride = match stride.checked_mul(extent) {
            Some(next) => next,
            None if empty => 0,
            None => return Err(DenseError::ElementCountOverflow),
        };
    }
    Ok(strides)
}

fn full_qr_of<E: DenseExecutor + ?Sized, T: AdjointScalar>(
    executor: &mut E,
    engine: &mut dyn Engine,
    view: DenseView<'_, T>,
    null: bool,
) -> Result<Vec<DenseTensor>, DenseError> {
    let shape = view.shape();
    let (m, n, batch) = (shape[0], shape[1], &shape[2..]);
    let dims = |a: usize, b: usize| [&[a, b][..], batch].concat();
    if null && m <= n {
        return Ok(vec![T::owned(dims(m, 0), Vec::new())?]);
    }
    let len = |cols: usize| checked_count(&dims(m, cols));
    let q_shape = dims(m, m);
    let q_strides = col_major_strides(&q_shape)?;
    let mut q = vec![T::ZERO; len(m)?];
    if m <= n {
        let r_shape = dims(m, n);
        let r_strides = col_major_strides(&r_shape)?;
        let mut r = vec![T::ZERO; len(n)?];
        executor.qr_into(
            engine,
            T::read(view),
            T::write(DenseViewMut::new(&mut q, &q_shape, &q_strides, 0)?),
            T::write(DenseViewMut::new(&mut r, &r_shape, &r_strides, 0)?),
        )?;
        return Ok(vec![T::owned(q_shape, q)?, T::owned(r_shape, r)?]);
    }
    let width = n + m;
    let augmented_shape = dims(m, width);
    let augmented_strides = col_major_strides(&augmented_shape)?;
    let mut augmented = vec![T::ZERO; len(width)?];
    let mut r = vec![T::ZERO; augmented.len()];
    let (data, strides) = (view.data(), view.strides());
    for (item, block) in augmented.chunks_exact_mut(m * width).enumerate() {
        let mut base = view.offset();
        let mut rest = item;
        for (&extent, &stride) in batch.iter().zip(&strides[2..]) {
            base += rest % extent * stride;
            rest /= extent;
        }
        for col in 0..n {
            for row in 0..m {
                block[row + m * col] = data[base + row * strides[0] + col * strides[1]];
            }
        }
        for row in 0..m {
            block[m * n + row * m + row] = T::ONE;
        }
    }
    executor.qr_into(
        engine,
        T::read(DenseView::new(
            &augmented,
            &augmented_shape,
            &augmented_strides,
            0,
        )?),
        T::write(DenseViewMut::new(&mut q, &q_shape, &q_strides, 0)?),
        T::write(DenseViewMut::new(
            &mut r,
            &augmented_shape,
            &augmented_strides,
            0,
        )?),
    )?;
    // Per item, `Q`'s last `m - n` columns (null) or `R`'s first `n`.
    let (source, cols, skip, keep) = if null {
        (&q, m, m * n, m * (m - n))
    } else {
        (&r, width, 0, m * n)
    };
    let kept = source
        .chunks_exact(m * cols)
        .flat_map(|item| &item[skip..skip + keep])
        .copied()
        .collect();
    if null {
        Ok(vec![T::owned(dims(m, m - n), kept)?])
    } else {
        Ok(vec![T::owned(q_shape, q)?, T::owned(dims(m, n), kept)?])
    }
}

/// [`DenseFactorization::SvdFull`] of one input through `executor`'s
/// per-matrix entries, as the variant documents. An input the staging cannot
/// read (rank below 2, or a dtype without a factorization) goes to
/// [`DenseExecutor::svd`] unchanged, so it is refused with `svd`'s own error.
pub(crate) fn full_svd_by_entries<E: DenseExecutor + ?Sized>(
    executor: &mut E,
    engine: &mut dyn Engine,
    input: DenseRead<'_>,
) -> Result<Vec<DenseTensor>, DenseError> {
    let batched = || DenseError::Unsupported {
        op: "svd_full",
        message: format!("batched full SVD input of shape {:?}", input.shape()),
    };
    match input {
        _ if input.shape().len() > 2 => Err(batched()),
        DenseRead::F32(view) if view.shape().len() == 2 => full_svd_of(executor, engine, view),
        DenseRead::F64(view) if view.shape().len() == 2 => full_svd_of(executor, engine, view),
        DenseRead::C32(view) if view.shape().len() == 2 => full_svd_of(executor, engine, view),
        DenseRead::C64(view) if view.shape().len() == 2 => full_svd_of(executor, engine, view),
        other => {
            executor.svd(engine, other)?;
            Err(DenseError::Unsupported {
                op: "svd_full",
                message: format!(
                    "full SVD input of dtype {:?} and rank {}",
                    other.dtype(),
                    other.shape().len()
                ),
            })
        }
    }
}

fn full_svd_of<E: DenseExecutor + ?Sized, T: AdjointScalar + Element>(
    executor: &mut E,
    engine: &mut dyn Engine,
    view: DenseView<'_, T>,
) -> Result<Vec<DenseTensor>, DenseError>
where
    T::Spectrum: Element,
{
    let (m, n) = (view.shape()[0], view.shape()[1]);
    if executor.supports_svd_full() {
        let (data, strides) = (view.data(), view.strides());
        let mut compact = Vec::with_capacity(checked_count(&[m, n])?);
        for col in 0..n {
            compact.extend(
                (0..m).map(|row| data[view.offset() + row * strides[0] + col * strides[1]]),
            );
        }
        return executor.svd_full_owned(engine, T::owned_input(compact), m, n);
    }
    // The compact factors through the executor's destination entry, which an
    // executor may implement apart from its owned `svd`.
    let k = m.min(n);
    let mut u1 = vec![<T as AdjointScalar>::ZERO; checked_count(&[m, k])?];
    let mut s = vec![T::Spectrum::default(); k];
    let mut vh1 = vec![<T as AdjointScalar>::ZERO; checked_count(&[k, n])?];
    executor.svd_into(
        engine,
        <T as AdjointScalar>::read(view),
        <T as AdjointScalar>::write(DenseViewMut::new(&mut u1, &[m, k], &[1, m], 0)?),
        <T::Spectrum as Element>::write(DenseViewMut::new(&mut s, &[k], &[1], 0)?),
        <T as AdjointScalar>::write(DenseViewMut::new(&mut vh1, &[k, n], &[1, k], 0)?),
    )?;
    let v1 = adjoint_items(&vh1, k, n);
    let u = complete_columns(executor, engine, u1, m, k)?;
    let v = complete_columns(executor, engine, v1, n, k)?;
    let vh = adjoint_items(&v, n, n);
    let s = DenseTensor::host_col_major(vec![k], s)?;
    Ok(vec![T::owned(vec![m, m], u)?, s, T::owned(vec![n, n], vh)?])
}

/// `thin`'s `rank` orthonormal columns (`rows x rank`, column-major)
/// followed by the null basis of `[thin I]` ([`full_qr_of`] with `null`), a
/// square unitary whose first columns are `thin` unchanged.
fn complete_columns<E: DenseExecutor + ?Sized, T: AdjointScalar>(
    executor: &mut E,
    engine: &mut dyn Engine,
    mut thin: Vec<T>,
    rows: usize,
    rank: usize,
) -> Result<Vec<T>, DenseError> {
    if rank == rows {
        return Ok(thin);
    }
    let shape = [rows, rank];
    let strides = [1, rows];
    let null = full_qr_of(
        executor,
        engine,
        DenseView::new(&thin, &shape, &strides, 0)?,
        true,
    )?;
    let null = T::slice(&null[0])?;
    thin.extend_from_slice(null);
    Ok(thin)
}
