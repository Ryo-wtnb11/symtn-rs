//! Owned replay: the core contraction into a fresh, uninitialized output.
//!
//! The plan derives a fresh layout from its GEMM destinations: physically
//! sorted parts tiling `[0, dst_len)`, each owned either by one GEMM call
//! (a maximal run of adjacent job outputs of that call) or by a zero fill
//! (inactive blocks, scatter-destination blocks and padding: the complement
//! of the job outputs). The layout never rejects a plan; padded plans keep
//! their initialized replay and get `+0.0` padding here.
//!
//! Publication: [`FusionBlockContractPlan::execute_owned`] hands each call
//! only its own parts and accepts a job report only at that job's address
//! and length, once. A reported `&'a mut [D]` holds the part's borrow for the
//! whole replay, so the backend cannot have produced it without initializing
//! that memory, and cannot de-initialize it afterwards. The output's length
//! is set only after every zero part was written and every call's reports
//! cover all of its parts; on an error, a panic or a missing report the
//! storage is dropped at length zero.

use std::ops::Range;

use super::*;
use crate::owned_overwrite_buffer::{initialize_owned, write_zeros};

/// One coefficient batch of `direct_batch`: its jobs and their runs.
#[derive(Clone, Debug)]
pub(super) struct DirectCall {
    pub(super) jobs: Range<usize>,
    pub(super) runs: Range<usize>,
}

/// `direct_batch` split into one batch per job coefficient: the jobs of one
/// coefficient are contiguous and its runs end at the next coefficient.
pub(super) fn direct_calls<C: Copy + PartialEq>(
    jobs: usize,
    alphas: &[C],
    runs: &[usize],
) -> Result<Vec<DirectCall>, OperationError> {
    if jobs != alphas.len() {
        return Err(OperationError::UnsupportedTensorContractScope {
            message: "direct plan has misaligned GEMM coefficients",
        });
    }
    let mut calls = Vec::new();
    let (mut start, mut run) = (0, 0);
    while start < jobs {
        let coefficient = alphas[start];
        let mut end = start;
        let first_run = run;
        while end < jobs && alphas[end] == coefficient {
            let len =
                runs.get(run)
                    .copied()
                    .ok_or(OperationError::UnsupportedTensorContractScope {
                        message: "direct plan runs do not cover its jobs",
                    })?;
            end += len;
            run += 1;
        }
        if end > jobs || alphas[start..end].iter().any(|&alpha| alpha != coefficient) {
            return Err(OperationError::UnsupportedTensorContractScope {
                message: "direct plan runs cross a coefficient change",
            });
        }
        calls.push(DirectCall {
            jobs: start..end,
            runs: first_run..run,
        });
        start = end;
    }
    Ok(calls)
}

#[derive(Clone, Debug, Default)]
pub(super) struct FreshLayout {
    len: usize,
    parts: Vec<FreshPart>,
    calls: Vec<FreshCall>,
    segments: usize,
}

#[derive(Clone, Copy, Debug)]
struct FreshPart {
    start: usize,
    len: usize,
    /// The call-major segment slot of a GEMM-owned part; `None` is zero-owned.
    segment: Option<usize>,
}

#[derive(Clone, Debug)]
struct FreshCall {
    source: CallSource,
    segments: Range<usize>,
    /// Total elements of the call's segments.
    len: usize,
}

#[derive(Clone, Copy, Debug)]
enum CallSource {
    /// An index into `direct_calls`.
    Direct(usize),
    /// An index into `irregular` (a direct destination with packed operands).
    Irregular(usize),
}

impl FreshLayout {
    pub(super) fn build(
        len: usize,
        direct_batch: &[Rank2GemmBatchJob],
        direct_calls: &[DirectCall],
        irregular: &[FusionIrregularGroupExecution],
    ) -> Result<Self, OperationError> {
        let sources = direct_calls
            .iter()
            .enumerate()
            .map(|(index, call)| (CallSource::Direct(index), &direct_batch[call.jobs.clone()]))
            .chain(
                irregular
                    .iter()
                    .enumerate()
                    .filter(|(_, execution)| !execution.class.scatters_dst())
                    .map(|(index, execution)| {
                        (
                            CallSource::Irregular(index),
                            std::slice::from_ref(&execution.job),
                        )
                    }),
            );
        let mut calls = Vec::with_capacity(direct_calls.len() + irregular.len());
        let mut outputs = Vec::with_capacity(direct_batch.len() + irregular.len());
        for (source, jobs) in sources {
            let call = calls.len();
            let mut call_len = 0usize;
            for job in jobs {
                let job_len = direct_matrix_len(job.rows, job.cols)?;
                if job_len != 0 {
                    outputs.push((job.dst_offset, job_len, call));
                    call_len = call_len
                        .checked_add(job_len)
                        .ok_or(OperationError::ElementCountOverflow)?;
                }
            }
            calls.push(FreshCall {
                source,
                segments: 0..0,
                len: call_len,
            });
        }
        outputs.sort_unstable();

        // `segment` holds the owning call until slots are assigned below.
        let mut parts: Vec<FreshPart> = Vec::with_capacity(2 * outputs.len() + 1);
        let mut cursor = 0;
        for (start, job_len, call) in outputs {
            let end = start
                .checked_add(job_len)
                .ok_or(OperationError::ElementCountOverflow)?;
            if start < cursor || end > len {
                return Err(OperationError::UnsupportedTensorContractScope {
                    message: "core contraction groups must write disjoint destination ranges",
                });
            }
            if start > cursor {
                parts.push(FreshPart {
                    start: cursor,
                    len: start - cursor,
                    segment: None,
                });
            }
            match parts.last_mut() {
                Some(part) if part.segment == Some(call) && part.start + part.len == start => {
                    part.len += job_len
                }
                _ => parts.push(FreshPart {
                    start,
                    len: job_len,
                    segment: Some(call),
                }),
            }
            cursor = end;
        }
        if cursor < len {
            parts.push(FreshPart {
                start: cursor,
                len: len - cursor,
                segment: None,
            });
        }

        // Call-major slots, ascending within a call: count each call's
        // parts, give each call its first slot, then hand slots out in
        // physical order.
        for call in parts.iter().filter_map(|part| part.segment) {
            calls[call].segments.end += 1;
        }
        let mut slot = 0;
        for call in &mut calls {
            let count = call.segments.end;
            call.segments = slot..slot;
            slot += count;
        }
        for part in &mut parts {
            if let Some(call) = part.segment {
                let segments = &mut calls[call].segments;
                part.segment = Some(segments.end);
                segments.end += 1;
            }
        }
        calls.retain(|call| call.len != 0);
        Ok(Self {
            len,
            parts,
            calls,
            segments: slot,
        })
    }
}

impl<C> FusionBlockContractPlan<C>
where
    C: Copy + PartialEq + One,
{
    /// `alpha * lhs * rhs` into a fresh owned payload of `dst_structure`,
    /// with no allocator-level zero fill of the whole output: each GEMM call
    /// receives only its own uninitialized job outputs, and symtn writes
    /// `+0.0` to the rest (inactive blocks, padding, and the blocks scatter
    /// groups then overwrite). The GEMM backend may still initialize its
    /// outputs before the product (a one-job batch, the [`Rank2Gemm`]
    /// default, and tcontract's Faer strategy do), and a scatter block is
    /// written twice. The caller's engine is admitted before any write.
    ///
    /// The returned payload exists only if every element was written; see
    /// the module documentation for the publication proof.
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn execute_owned<A, G, D>(
        &self,
        engine: &mut dyn Engine,
        kernels: &mut A,
        gemm: &mut G,
        fusion_workspace: &mut FusionBlockContractWorkspace<D>,
        dst_structure: &Arc<BlockStructure>,
        lhs_structure: &Arc<BlockStructure>,
        lhs_data: &[D],
        rhs_structure: &Arc<BlockStructure>,
        rhs_data: &[D],
        alpha: D,
    ) -> Result<Vec<D>, OperationError>
    where
        A: HostKernelAdapter<D>,
        G: Rank2Gemm<D>,
        D: DenseBlockScalar + RecouplingCoefficientAction<C>,
    {
        let len = dst_structure
            .required_len()
            .map_err(OperationError::from_core_preserving_context)?;
        self.validate_replay_inputs(
            dst_structure,
            len,
            lhs_structure,
            lhs_data.len(),
            rhs_structure,
            rhs_data.len(),
        )?;
        if len != self.fresh.len {
            return Err(OperationError::ElementCountMismatch {
                expected: self.fresh.len,
                actual: len,
            });
        }
        // Why not rely on the GEMMs to admit the engine: an all-inactive or
        // empty output, or a zero alpha, runs none, yet the caller's engine
        // must still be refused before anything is written, as initialized
        // replay refuses it in its inactive-block region.
        crate::host_pool::with_exec(engine, |_| Ok(()))?;
        self.prepare_irregular_scratch(fusion_workspace);
        let mut data = initialize_owned(len, |spare| {
            self.write_fresh(
                engine,
                kernels,
                gemm,
                fusion_workspace,
                spare,
                lhs_data,
                rhs_data,
                alpha,
            )
        })?;
        for execution in &self.irregular {
            if execution.class.scatters_dst() {
                self.execute_irregular_group(
                    engine,
                    kernels,
                    gemm,
                    fusion_workspace,
                    execution,
                    &mut data,
                    lhs_data,
                    rhs_data,
                    alpha,
                    D::zero(),
                )?;
            }
        }
        Ok(data)
    }

    /// Initializes every element of `spare` (the `initialize_owned`
    /// obligation): zero parts here, GEMM parts by their calls' reports.
    #[allow(clippy::too_many_arguments)]
    fn write_fresh<'a, A, G, D>(
        &self,
        engine: &mut dyn Engine,
        kernels: &mut A,
        gemm: &mut G,
        fusion_workspace: &mut FusionBlockContractWorkspace<D>,
        spare: &'a mut [MaybeUninit<D>],
        lhs_data: &[D],
        rhs_data: &[D],
        alpha: D,
    ) -> Result<(), OperationError>
    where
        A: HostKernelAdapter<D>,
        G: Rank2Gemm<D>,
        D: DenseBlockScalar + RecouplingCoefficientAction<C>,
    {
        let base = spare.as_ptr().cast::<D>();
        let mut segments: Vec<(usize, &'a mut [MaybeUninit<D>])> = (0..self.fresh.segments)
            .map(|_| (0, Default::default()))
            .collect();
        let mut rest = spare;
        for part in &self.fresh.parts {
            if part.len > rest.len() {
                return Err(OperationError::ElementCountMismatch {
                    expected: part.len,
                    actual: rest.len(),
                });
            }
            let (piece, tail) = std::mem::take(&mut rest).split_at_mut(part.len);
            rest = tail;
            match part.segment {
                None => {
                    write_zeros(piece);
                }
                Some(slot) => segments[slot] = (part.start, piece),
            }
        }
        if !rest.is_empty() {
            return Err(OperationError::InvalidArgument {
                message: "owned contraction layout does not cover its output",
            });
        }
        let FusionBlockContractWorkspace {
            scratch,
            fresh_reported,
            ..
        } = fusion_workspace;
        for call in &self.fresh.calls {
            let segments = &mut segments[call.segments.clone()];
            match call.source {
                CallSource::Direct(index) => {
                    let jobs = self.direct_calls[index].jobs.clone();
                    let coefficient = self.direct_batch_alpha[jobs.start];
                    let job_alpha = if coefficient == C::one() {
                        alpha
                    } else {
                        alpha.scale_by_coefficient(coefficient)
                    };
                    self.fresh_call(
                        engine,
                        gemm,
                        fresh_reported,
                        segments,
                        call.len,
                        base,
                        lhs_data,
                        rhs_data,
                        &self.direct_batch[jobs],
                        job_alpha,
                    )?;
                }
                CallSource::Irregular(index) => {
                    let execution = &self.irregular[index];
                    let scratch = &mut scratch.as_mut_slice()[..execution.scratch.total_len];
                    if !alpha.is_zero() {
                        self.pack_operands(
                            engine, kernels, scratch, execution, lhs_data, rhs_data,
                        )?;
                    }
                    let (lhs, rhs) = execution.operands(scratch, lhs_data, rhs_data);
                    self.fresh_call(
                        engine,
                        gemm,
                        fresh_reported,
                        segments,
                        call.len,
                        base,
                        lhs,
                        rhs,
                        std::slice::from_ref(&execution.job),
                        alpha,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// One GEMM call into its own `segments` (`len` elements in all).
    /// Succeeds only if the reports cover every segment element: each
    /// accepted report is a distinct job's memory at that job's address and
    /// length, and their lengths sum to `len`.
    #[allow(clippy::too_many_arguments)]
    fn fresh_call<'a, G, D>(
        &self,
        engine: &mut dyn Engine,
        gemm: &mut G,
        reported: &mut Vec<bool>,
        segments: &mut [(usize, &'a mut [MaybeUninit<D>])],
        len: usize,
        base: *const D,
        lhs: &[D],
        rhs: &[D],
        jobs: &[Rank2GemmBatchJob],
        alpha: D,
    ) -> Result<(), OperationError>
    where
        G: Rank2Gemm<D>,
        D: Copy + Zero,
    {
        // BLAS `gemm` at `alpha == 0` writes `beta * C = 0` without forming
        // `A * B`; a backend that multiplies through would turn `0 * Inf`
        // into NaN (see `execute_batch`).
        if alpha.is_zero() {
            for (_, segment) in segments.iter_mut() {
                write_zeros(std::mem::take(segment));
            }
            return Ok(());
        }
        reported.clear();
        reported.resize(jobs.len(), false);
        let mut covered = 0usize;
        let mut forged = false;
        let mut written = |index: usize, output: &'a mut [D]| {
            let accepted = jobs.get(index).is_some_and(|job| {
                let job_len = job.rows.wrapping_mul(job.cols);
                !reported[index]
                    && output.len() == job_len
                    && (job_len == 0
                        || std::ptr::eq(output.as_ptr(), base.wrapping_add(job.dst_offset)))
            });
            if accepted {
                reported[index] = true;
                covered += output.len();
            } else {
                forged = true;
            }
        };
        gemm.matmul_rank2_batch_fresh(
            engine,
            segments,
            lhs,
            rhs,
            jobs,
            [self.lhs_op, self.rhs_op],
            alpha,
            &mut written,
        )?;
        if forged {
            return Err(OperationError::InvalidArgument {
                message: "fresh GEMM reported storage other than its job's output",
            });
        }
        if covered != len || reported.iter().any(|&done| !done) {
            return Err(OperationError::InvalidArgument {
                message: "fresh GEMM left a job output unreported",
            });
        }
        Ok(())
    }
}
