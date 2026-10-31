use super::*;

/// Rank-2 column-major GEMM over host slices: the only capability the replay
/// half needs from a contraction backend. The symmetric layer adapts its
/// contraction backends onto this.
pub trait Rank2Gemm<D> {
    /// `dst = alpha * lhs * rhs + beta * dst` over column-major matrices
    /// (BLAS gemm semantics).
    #[allow(clippy::too_many_arguments)]
    fn matmul_rank2(
        &mut self,
        engine: &mut dyn Engine,
        dst: &mut [D],
        lhs: &[D],
        rhs: &[D],
        rows: usize,
        contracted: usize,
        cols: usize,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>;

    /// Executes a batch of independent GEMMs addressed by offsets into shared
    /// buffers: for each job, the `rows x cols` destination matrix at
    /// `dst[job.dst_offset..]` receives `alpha * lhs_part * rhs_part + beta *
    /// dst_part` (column-major). The plan layer guarantees the destination
    /// ranges of a batch are pairwise disjoint, so implementations may run
    /// jobs in any order or concurrently. The default executes them in order.
    ///
    /// `runs` is the batch's plan-time run partition (see issue #103); backends
    /// that route runs read it, the serial default ignores it.
    #[expect(
        clippy::too_many_arguments,
        reason = "the replay backend boundary keeps shared buffers, jobs, run partition, and alpha/beta explicit"
    )]
    fn matmul_rank2_batch(
        &mut self,
        engine: &mut dyn Engine,
        dst: &mut [D],
        lhs: &[D],
        rhs: &[D],
        jobs: &[Rank2GemmBatchJob],
        runs: &[usize],
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        D: Copy,
    {
        let _ = runs;
        for job in jobs {
            let lhs_slice = direct_slice(lhs, job.lhs_offset, job.rows, job.contracted)?;
            let rhs_slice = direct_slice(rhs, job.rhs_offset, job.contracted, job.cols)?;
            let dst_slice = direct_slice_mut(dst, job.dst_offset, job.rows, job.cols)?;
            self.matmul_rank2(
                engine,
                dst_slice,
                lhs_slice,
                rhs_slice,
                job.rows,
                job.contracted,
                job.cols,
                alpha,
                beta,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn matmul_rank2_batch_with_ops(
        &mut self,
        engine: &mut dyn Engine,
        dst: &mut [D],
        lhs: &[D],
        rhs: &[D],
        jobs: &[Rank2GemmBatchJob],
        runs: &[usize],
        lhs_op: MatrixOp,
        rhs_op: MatrixOp,
        alpha: D,
        beta: D,
    ) -> Result<(), OperationError>
    where
        D: Copy,
    {
        if lhs_op == MatrixOp::Identity && rhs_op == MatrixOp::Identity {
            return self.matmul_rank2_batch(engine, dst, lhs, rhs, jobs, runs, alpha, beta);
        }
        Err(OperationError::UnsupportedTensorContractScope {
            message: "rank-2 GEMM backend does not implement transpose/adjoint batches",
        })
    }

    /// Batched overwrite GEMM into uninitialized memory: each job's
    /// column-major `rows x cols` output at `dst_offset` becomes
    /// `alpha * op(lhs_part) op(rhs_part)` with `op = ops[0], ops[1]`, and no
    /// prior destination value is read.
    ///
    /// `segments` are `(start, memory)` pairs in ascending, non-overlapping
    /// order, `memory` being destination range `start..start + len`; every
    /// job with a nonempty output lies inside one segment, disjoint from the
    /// other jobs. On `Ok`, every job has been reported once through
    /// `written` with its own initialized piece (an empty job with an empty
    /// slice). A reported slice keeps its borrow for the whole `'a`, which is
    /// what lets the caller treat it as proof that the piece is initialized.
    ///
    /// The default zero-fills each job's piece, runs it as a one-job
    /// [`Self::matmul_rank2_batch_with_ops`] with `beta = 0`, and reports
    /// every job once all succeed.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    fn matmul_rank2_batch_fresh<'a>(
        &mut self,
        engine: &mut dyn Engine,
        segments: &mut [(usize, &'a mut [MaybeUninit<D>])],
        lhs: &[D],
        rhs: &[D],
        jobs: &[Rank2GemmBatchJob],
        ops: [MatrixOp; 2],
        alpha: D,
        written: &mut dyn FnMut(usize, &'a mut [D]),
    ) -> Result<(), OperationError>
    where
        D: Copy + Zero,
    {
        let pieces = job_pieces(segments, jobs)?;
        let mut outputs = Vec::with_capacity(jobs.len());
        for (job, piece) in jobs.iter().zip(pieces) {
            let output = crate::owned_overwrite_buffer::write_zeros(piece);
            if !output.is_empty() {
                self.matmul_rank2_batch_with_ops(
                    engine,
                    &mut *output,
                    lhs,
                    rhs,
                    &[Rank2GemmBatchJob {
                        dst_offset: 0,
                        ..*job
                    }],
                    &[1],
                    ops[0],
                    ops[1],
                    alpha,
                    D::zero(),
                )?;
            }
            outputs.push(output);
        }
        for (index, output) in outputs.into_iter().enumerate() {
            written(index, output);
        }
        Ok(())
    }
}

/// Each job's own piece of `segments` (an empty slice for an empty job),
/// by job index. Refuses a job outside every segment or overlapping
/// another before taking any memory.
pub(super) fn job_pieces<'a, E>(
    segments: &mut [(usize, &'a mut [E])],
    jobs: &[Rank2GemmBatchJob],
) -> Result<Vec<&'a mut [E]>, OperationError> {
    let mut order = Vec::with_capacity(jobs.len());
    for (index, job) in jobs.iter().enumerate() {
        let len = direct_matrix_len(job.rows, job.cols)?;
        if len != 0 {
            let end = job
                .dst_offset
                .checked_add(len)
                .ok_or(OperationError::ElementCountOverflow)?;
            order.push((job.dst_offset, end, index));
        }
    }
    order.sort_unstable();
    let outside = OperationError::InvalidArgument {
        message: "fresh GEMM job lies outside its destination segments",
    };
    // Check every job before taking memory.
    let (mut segment, mut covered) = (0, 0);
    for &(start, end, _) in &order {
        while segments
            .get(segment)
            .is_some_and(|(first, memory)| first + memory.len() <= start)
        {
            segment += 1;
        }
        match segments.get(segment) {
            Some((first, memory)) if start >= covered && start >= *first => {
                if end > first + memory.len() {
                    return Err(outside);
                }
            }
            _ => return Err(outside),
        }
        covered = end;
    }
    let mut pieces: Vec<&'a mut [E]> = jobs.iter().map(|_| Default::default()).collect();
    let (mut segment, mut cursor) = (0, 0);
    let mut remaining: &'a mut [E] = &mut [];
    for (start, end, index) in order {
        while cursor + remaining.len() < end {
            let (first, memory) = &mut segments[segment];
            remaining = core::mem::take(memory);
            cursor = *first;
            segment += 1;
        }
        let (_, rest) = core::mem::take(&mut remaining).split_at_mut(start - cursor);
        let (piece, tail) = rest.split_at_mut(end - start);
        pieces[index] = piece;
        remaining = tail;
        cursor = end;
    }
    Ok(pieces)
}

/// One GEMM job addressed inside its selected source and destination buffers.
/// Fully-direct jobs use tensor storage; irregular jobs use group-local scratch
/// for exactly the operands selected by their execution class.
pub type Rank2GemmBatchJob = DenseGemmBatchJob;

/// Placement-aware block GEMM over storage ranges.
///
/// The device-side replay seam for core fusion-block contraction:
/// `dst[dst_offset..][rows x cols] = lhs[lhs_offset..][rows x contracted] *
/// rhs[rhs_offset..][contracted x cols]` as column-major matrices, with no
/// host-slice contract in the trait. The host implementation wraps a
/// a rank-2 GEMM backend; device implementations submit kernels against
/// device storage handles. No method takes an engine: an implementation
/// carries the engine it submits through (a device implementation holds the
/// CUDA context that borrows the caller's engine), so a replay never needs a
/// second borrow of it.
pub trait StorageGemm<D, DDst, DLhs, DRhs> {
    /// Reports whether the generalized entry point accepts these operand views.
    /// Replay checks this once before any unit-coefficient job can mutate `dst`.
    fn supports_matmul_with_ops_scaled(&self, _lhs_op: MatrixOp, _rhs_op: MatrixOp) -> bool {
        false
    }

    #[allow(clippy::too_many_arguments)]
    fn matmul_range_into(
        &mut self,
        dst: &mut DDst,
        dst_offset: usize,
        lhs: &DLhs,
        lhs_offset: usize,
        rhs: &DRhs,
        rhs_offset: usize,
        rows: usize,
        contracted: usize,
        cols: usize,
    ) -> Result<(), OperationError>;

    /// Executes one GEMM with explicit operand views and output coefficient.
    /// The default preserves source compatibility while rejecting the extension.
    #[allow(clippy::too_many_arguments)]
    fn matmul_range_with_ops_scaled_into(
        &mut self,
        _dst: &mut DDst,
        _dst_offset: usize,
        _lhs: &DLhs,
        _lhs_offset: usize,
        _rhs: &DRhs,
        _rhs_offset: usize,
        _rows: usize,
        _contracted: usize,
        _cols: usize,
        _lhs_op: MatrixOp,
        _rhs_op: MatrixOp,
        _alpha: D,
    ) -> Result<(), OperationError> {
        Err(OperationError::UnsupportedTensorContractScope {
            message: "storage GEMM backend does not implement scaled replay",
        })
    }

    /// `dst = alpha * op(lhs) op(rhs) + beta * dst` for one job: the scaled
    /// entry with the caller's `beta` in the GEMM epilogue. A backend without
    /// it rejects the call before writing, so the first job of a replay
    /// decides the capability for all of them.
    #[allow(clippy::too_many_arguments)]
    fn matmul_range_axpby_with_ops_into(
        &mut self,
        _dst: &mut DDst,
        _dst_offset: usize,
        _lhs: &DLhs,
        _lhs_offset: usize,
        _rhs: &DRhs,
        _rhs_offset: usize,
        _rows: usize,
        _contracted: usize,
        _cols: usize,
        _lhs_op: MatrixOp,
        _rhs_op: MatrixOp,
        _alpha: D,
        _beta: D,
    ) -> Result<(), OperationError> {
        Err(OperationError::UnsupportedTensorContractScope {
            message: "storage GEMM backend does not implement a beta-accumulating replay",
        })
    }
}
