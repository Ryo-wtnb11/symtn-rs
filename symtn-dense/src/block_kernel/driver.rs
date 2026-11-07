//! The block kernels' one lane driver: a kernel describes one block's
//! estimate and execution ([`BlockOp`]); the driver enters the kernel's
//! guarded engine entry, places the blocks with [`lanes::place`] (one
//! provider stage each) and runs them on [`lanes::run_native`], returning the
//! lowest-indexed failure.

use std::time::Duration;

use tcontract_exec::{Exec, WorkModel};
use tenferro_cpu::CpuEngine;

use crate::lanes::{self, Estimate, Form, LaneTables, StageKind};
use crate::tenferro_adapter::tenferro_error;
use crate::DenseError;

/// The guarded engine entry a kernel runs under: a panic inside either marks
/// the engine broken.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Entry {
    /// `CpuEngine::with_contraction_exec`, which also bounds the buffer
    /// pool's retention for the call.
    Contraction,
    /// `CpuEngine::with_linalg_pool`, the per-matrix factorizations' entry.
    Linalg,
}

/// The dependency whose own model turns a block's work into time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OpKind {
    /// tcontract's GEMM rate: the plan that runs the block is tcontract's.
    Gemm,
    /// The engine's `WorkModel`, from which tlinalg makes its own width
    /// decision for a factorization.
    Factorization,
}

/// A nanosecond estimate as the dependency normalizes it (`tcontract-exec`
/// `exec.rs: ns_duration`): NaN and nonpositive are zero, huge saturates.
pub(crate) fn estimate(ns: f64) -> Duration {
    if ns.is_nan() || ns <= 0.0 {
        Duration::ZERO
    } else if ns >= 1e18 {
        Duration::from_secs(u64::MAX / 4)
    } else {
        Duration::from_secs_f64(ns * 1e-9)
    }
}

/// Serial time of `work` weighted multiply-adds under the engine's work model,
/// mirroring `tcontract-exec` `batch.rs: item_ns` (a rate that is not `> 0`,
/// NaN included, counts as 1).
pub(crate) fn work_model_ns(model: &WorkModel, work: f64) -> f64 {
    let rate = if model.muladds_per_ns > 0.0 {
        model.muladds_per_ns
    } else {
        1.0
    };
    model.item_overhead_ns + work / rate
}

/// The estimated serial time of `work` weighted multiply-adds (four per
/// complex multiply-add) of kind `kind`.
///
/// Each kind uses the model of the dependency that executes the block:
/// tcontract's `ns_per_flop` over `2 work` flops for GEMM, the engine work
/// model (as `tcontract-exec` `item_ns`) for a factorization.
pub(crate) fn cost(kind: OpKind, work: f64, model: &WorkModel) -> Duration {
    estimate(match kind {
        OpKind::Gemm => 2.0 * work * tcontract::ns_per_flop(),
        OpKind::Factorization => work_model_ns(model, work),
    })
}

/// A lane's blocks, handed out one by one to the visitor it is called with.
pub(crate) type LaneMembers<'v, J> = dyn FnMut(&mut dyn FnMut(&mut J)) + 'v;

/// One kernel's blocks: what the driver needs to estimate and run each one.
/// Validation that must precede the entry, and the outputs a block produces,
/// stay with the kernel. Each block is one provider stage.
pub(crate) trait BlockOp: Sync {
    /// One block's exclusive state: its validated operands and its output.
    type Job: Send;
    /// One lane's state, made on the caller before the fork; its blocks run
    /// with it in order on the lane's thread.
    type Lane: Send;
    const ENTRY: Entry;
    /// The `DenseError` label of a refused entry.
    const LABEL: &'static str;

    /// Block `index`'s provider estimates under the engine's work model.
    fn estimate(&self, index: usize, job: &Self::Job, model: &WorkModel) -> Estimate;

    /// Builds one lane's [`Lane`](Self::Lane) from that lane's blocks
    /// (`visit` hands out each one) on a context of the lane's share.
    fn lane(&self, jobs: &mut LaneMembers<'_, Self::Job>, exec: &Exec<'_>) -> Self::Lane;

    /// Runs block `index` on `exec`, whose budget is the block's selected
    /// width, with its lane's scratch.
    fn run(
        &self,
        index: usize,
        job: &mut Self::Job,
        lane: &mut Self::Lane,
        exec: Exec<'_>,
    ) -> Result<(), DenseError>;
}

/// Placement's view of a kernel's blocks: one provider stage each, estimated
/// on demand.
struct Costed<'a, B: BlockOp> {
    op: &'a B,
    jobs: &'a [B::Job],
    model: WorkModel,
}

impl<B: BlockOp> lanes::Jobs for Costed<'_, B> {
    fn count(&self) -> usize {
        self.jobs.len()
    }
    fn stages(&self, _: usize) -> usize {
        1
    }
    fn kind(&self, _: usize, _: usize) -> StageKind {
        StageKind::Provider
    }
    fn estimate(&self, job: usize, _: usize) -> Estimate {
        self.op.estimate(job, &self.jobs[job], &self.model)
    }
}

/// The runner's view of a kernel's blocks.
struct Running<'a, B> {
    op: &'a B,
    count: usize,
}

impl<B: BlockOp> lanes::Jobs for Running<'_, B> {
    fn count(&self) -> usize {
        self.count
    }
    fn stages(&self, _: usize) -> usize {
        1
    }
    fn kind(&self, _: usize, _: usize) -> StageKind {
        StageKind::Provider
    }
    // Placement has finished before the run, which reads no estimate.
    fn estimate(&self, _: usize, _: usize) -> Estimate {
        Estimate {
            serial: Duration::ZERO,
            payback: None,
            sections: 1,
            widen: true,
        }
    }
}

impl<B: BlockOp> lanes::Work for Running<'_, B> {
    type Job = B::Job;
    type Lane = B::Lane;
    type Error = DenseError;
    fn run(
        &self,
        job: usize,
        _: usize,
        state: &mut B::Job,
        lane: &mut B::Lane,
        exec: Exec<'_>,
    ) -> Result<(), DenseError> {
        self.op.run(job, state, lane, exec)
    }
}

/// Runs every block of `jobs` on `cpu`'s lanes inside `B::ENTRY`, placed in
/// `tables`, and returns the lowest-indexed block's error. Every block runs
/// even after another fails; a panicking block unwinds after the lanes join
/// and leaves the engine broken.
pub(crate) fn drive<B: BlockOp>(
    cpu: &mut CpuEngine,
    op: &B,
    jobs: &mut [B::Job],
    tables: &mut LaneTables,
) -> Result<(), DenseError> {
    let entered = match B::ENTRY {
        Entry::Contraction => {
            cpu.with_contraction_exec(|exec, _, _| on_lanes(exec, op, jobs, tables))
        }
        Entry::Linalg => cpu.with_linalg_pool(|exec, _| on_lanes(exec, op, jobs, tables)),
    };
    entered
        .map_err(|err| tenferro_error(B::LABEL, err))?
        .map_err(|(_, error)| error)
}

fn on_lanes<B: BlockOp>(
    exec: &mut Exec<'_>,
    op: &B,
    jobs: &mut [B::Job],
    tables: &mut LaneTables,
) -> tenferro_tensor::Result<lanes::Outcome<DenseError>> {
    let costed = Costed {
        op,
        jobs,
        model: exec.batch_policy().model,
    };
    lanes::place(exec, &costed, Form::Native, tables);
    let placement = tables.placement();
    #[cfg(test)]
    crate::probe::add(crate::probe::Event::BlockLanes, placement.lanes());
    let mut lane_states = Vec::with_capacity(placement.lanes());
    for l in 0..placement.lanes() {
        let lane_exec = exec.reborrow().limit(
            std::num::NonZeroUsize::new(placement.share(l)).unwrap_or(std::num::NonZeroUsize::MIN),
        );
        let members = placement.jobs(l);
        let mut visit = |each: &mut dyn FnMut(&mut B::Job)| {
            for &j in members {
                each(&mut jobs[j]);
            }
        };
        lane_states.push(op.lane(&mut visit, &lane_exec));
    }
    let running = Running {
        op,
        count: jobs.len(),
    };
    lanes::run_native(
        exec.reborrow(),
        &running,
        lanes::Placed { tables },
        jobs,
        &mut lane_states,
    )
    .map_err(|refused| tenferro_tensor::Error::backend_source("Exec::fork_sized", refused))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Purpose: a work-model rate that is zero or NaN counts as one
    /// multiply-add per ns, and a NaN or negative time is zero while an
    /// infinite one saturates, as the dependency normalizes them.
    #[test]
    fn rate_zero_or_nan_counts_as_one_and_nan_time_is_zero() {
        let rate = |r: f64| WorkModel::default().with_muladds_per_ns(r);
        let reference = work_model_ns(&rate(1.0), 4096.0);
        assert_eq!(work_model_ns(&rate(0.0), 4096.0), reference);
        assert_eq!(work_model_ns(&rate(f64::NAN), 4096.0), reference);
        assert_eq!(estimate(f64::NAN), Duration::ZERO);
        assert_eq!(estimate(-1.0), Duration::ZERO);
        assert_eq!(estimate(f64::INFINITY), Duration::from_secs(u64::MAX / 4));
    }
}
