use num_complex::{Complex32, Complex64};

use crate::{
    DenseDType, DenseDotConfig, DenseError, DenseRead, DenseScalar, DenseTensor, DenseView,
    DenseViewMut, DenseWrite, Engine,
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

    /// Applies `op` to each input and returns exactly one entry per input,
    /// holding that input's outputs exactly as the per-matrix entry returns
    /// them; on failure, returns the lowest-indexed failing input's error
    /// and no outputs.
    ///
    /// The default issues one per-matrix call per input, in order, and stops
    /// at the first error. An executor whose per-call entry carries a fixed
    /// admission cost overrides this to pay it once per batch, not once per
    /// matrix; one may run the inputs concurrently on the engine's lanes.
    fn factorize_batch(
        &mut self,
        engine: &mut dyn Engine,
        op: DenseFactorization,
        inputs: &[DenseRead<'_>],
    ) -> Result<Vec<Vec<DenseTensor>>, DenseError> {
        inputs
            .iter()
            .map(|&input| match op {
                DenseFactorization::Svd => self.svd(engine, input),
                DenseFactorization::Qr => self.qr(engine, input),
            })
            .collect()
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
