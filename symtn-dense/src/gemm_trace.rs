//! Unstable capture of the grouped GEMM calls [`crate::DefaultDenseExecutor`]
//! receives, for replaying them outside symtn.
//!
//! The collector is process-wide: concurrent operations on lane sub-engines
//! all record into it. A call is recorded at entry, before its dispatch, so a
//! call that then fails is still recorded; discard the trace of a failed step.
//! Only the three batch entries record; single GEMMs are not grouped calls.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, PoisonError};

use crate::block_kernel::gemm::Geometry;
use crate::{DenseDType, DenseGemmBatchJob, DenseScalar, MatrixOp};

/// Every call recorded between [`start`] and [`take`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Trace {
    pub calls: Vec<Call>,
    /// Calls not stored because their jobs exceeded the remaining budget.
    pub dropped_calls: usize,
}

/// One grouped GEMM call: `C_j = alpha * op0(A_j) * op1(B_j) + beta * C_j`.
#[derive(Clone, Debug, PartialEq)]
pub struct Call {
    pub dtype: DenseDType,
    pub ops: [MatrixOp; 2],
    pub alpha: DenseScalar,
    /// `None` for a fresh (uninitialized, never read) destination.
    pub beta: Option<DenseScalar>,
    /// The width of the engine the call was offered.
    pub threads: usize,
    /// The caller's strided-run partition; empty for fresh calls.
    pub runs: Vec<usize>,
    pub jobs: Vec<Job>,
}

/// One column-major job; offsets are relative to each view's own offset and
/// leading dimensions are those of the compact layout the executor lowers to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Job {
    pub a_offset: usize,
    pub b_offset: usize,
    pub c_offset: usize,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub lda: usize,
    pub ldb: usize,
    pub ldc: usize,
}

struct State {
    trace: Trace,
    jobs_left: usize,
}

static ARMED: AtomicBool = AtomicBool::new(false);
static STATE: Mutex<Option<State>> = Mutex::new(None);

/// Clears any trace and records until [`take`], storing at most `max_jobs`
/// jobs; a call that does not fit whole is counted in `dropped_calls`.
pub fn start(max_jobs: usize) {
    *STATE.lock().unwrap_or_else(PoisonError::into_inner) = Some(State {
        trace: Trace::default(),
        jobs_left: max_jobs,
    });
    ARMED.store(true, Ordering::Relaxed);
}

/// Stops recording and returns the trace (empty if never started).
pub fn take() -> Trace {
    ARMED.store(false, Ordering::Relaxed);
    STATE
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .take()
        .map(|state| state.trace)
        .unwrap_or_default()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn record(
    dtype: DenseDType,
    ops: [MatrixOp; 2],
    alpha: DenseScalar,
    beta: Option<DenseScalar>,
    threads: usize,
    jobs: &[DenseGemmBatchJob],
    runs: &[usize],
) {
    if !ARMED.load(Ordering::Relaxed) {
        return;
    }
    let mut guard = STATE.lock().unwrap_or_else(PoisonError::into_inner);
    let Some(state) = guard.as_mut() else {
        return;
    };
    if jobs.len() > state.jobs_left {
        state.trace.dropped_calls += 1;
        return;
    }
    // An extent past `isize` fails in the route anyway; nothing to replay.
    let Ok(jobs) = jobs
        .iter()
        .map(|job| {
            let geometry = Geometry::new(job, ops)?;
            let [a, b, c] = geometry.strides;
            let ld = |s: [isize; 2], op: MatrixOp| match op {
                MatrixOp::Identity => s[1] as usize,
                MatrixOp::Transpose | MatrixOp::Adjoint => s[0] as usize,
            };
            Ok::<_, crate::DenseError>(Job {
                a_offset: job.lhs_offset,
                b_offset: job.rhs_offset,
                c_offset: job.dst_offset,
                m: job.rows,
                n: job.cols,
                k: job.contracted,
                lda: ld(a, ops[0]),
                ldb: ld(b, ops[1]),
                ldc: c[1] as usize,
            })
        })
        .collect::<Result<Vec<_>, _>>()
    else {
        return;
    };
    state.jobs_left -= jobs.len();
    state.trace.calls.push(Call {
        dtype,
        ops,
        alpha,
        beta,
        threads,
        runs: runs.to_vec(),
        jobs,
    });
}
