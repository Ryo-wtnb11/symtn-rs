//! The block kernels' one lane driver: a kernel describes one block's work
//! and execution ([`BlockOp`]); the driver enters the kernel's guarded engine
//! entry, turns the work into times with the one cost function, allocates the
//! lanes and runs every block, returning the lowest-indexed failure.

use std::time::Duration;

use tcontract_exec::{Exec, WorkModel};
use tenferro_cpu::CpuEngine;

use super::schedule::{self, AllocationScratch, GrantKey};
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

/// The dependency whose own model turns a block's work into time, and the
/// family of its grant: the lane-allocation cache keys it, so two kinds never
/// share an allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum OpKind {
    /// tcontract's GEMM rate: the plan that runs the block is tcontract's.
    Gemm,
    /// The engine's `WorkModel`, from which tlinalg makes its own width
    /// decision for a factorization (see `factor::grant`).
    Factorization,
}

/// The estimated serial time of `work` weighted multiply-adds (four per
/// complex multiply-add) of kind `kind`.
///
/// Each kind uses the model of the dependency that executes the block, so the
/// scheduler and that dependency agree on the job's cost: tcontract's
/// `ns_per_flop` over `2 work` flops for GEMM, the engine work model (as
/// `tcontract-exec` `item_ns`) for a factorization. Neither is calibrated
/// here; both move to the providers' own estimates when they publish them.
pub(crate) fn cost(kind: OpKind, work: f64, model: &WorkModel) -> Duration {
    schedule::estimate(match kind {
        OpKind::Gemm => 2.0 * work * tcontract::ns_per_flop(),
        OpKind::Factorization => schedule::work_model_ns(model, work),
    })
}

/// One kernel's blocks: what the driver needs to cost and run each one.
/// Validation that must precede the entry, and the outputs a block produces,
/// stay with the kernel.
pub(crate) trait BlockOp: Sync {
    /// One block's exclusive state: its validated operands and its output.
    type Job: Send;
    const ENTRY: Entry;
    const KIND: OpKind;
    /// The `DenseError` label of a refused entry.
    const LABEL: &'static str;

    /// The block's weighted multiply-adds, as its provider estimates them.
    fn work(&self, index: usize, job: &Self::Job) -> f64;

    /// Whether [`grant`](Self::grant) models block `index`'s provider
    /// width. A batch holding a block it does not model keeps the baseline
    /// placement, without the search.
    fn modeled(&self, _: usize, _: &Self::Job) -> bool {
        true
    }

    /// The width block `index`'s provider takes on a lane of `budget`
    /// reserved slots of `exec`, for a block of serial time `serial`: the
    /// scheduler's model of that provider's own width decision, composed from
    /// the provider's public decisions. Consulted only for a batch whose
    /// every block is [`modeled`](Self::modeled).
    fn grant(
        &self,
        index: usize,
        job: &Self::Job,
        exec: &Exec<'_>,
        budget: usize,
        serial: Duration,
    ) -> usize;

    /// What block `index`'s [`grant`](Self::grant) depends on besides its
    /// serial time, the context and the op's [`OpKind`] (see [`GrantKey`]).
    fn grant_key(&self, index: usize, job: &Self::Job) -> GrantKey;

    /// Runs block `index` on its lane's execution context.
    fn run(&self, index: usize, job: &mut Self::Job, exec: Exec<'_>) -> Result<(), DenseError>;
}

/// The driver's reusable staging: the per-block times and the lane tables.
/// An owner that submits repeatedly keeps one, so a warm submission reuses
/// every capacity.
#[derive(Debug, Default)]
pub(crate) struct LaneScratch {
    times: Vec<Duration>,
    allocation: AllocationScratch,
}

impl LaneScratch {
    /// The capacities kept, in bytes. Saturates.
    pub(crate) fn retained_bytes(&self) -> usize {
        self.times
            .capacity()
            .saturating_mul(std::mem::size_of::<Duration>())
            .saturating_add(self.allocation.retained_bytes())
    }
}

/// Runs every block of `jobs` on `cpu`'s lanes inside `B::ENTRY`, staged in
/// `scratch`, and returns the lowest-indexed block's error. Every block runs
/// even after another fails; a panicking block unwinds after the lanes join
/// and leaves the engine broken.
pub(crate) fn drive<B: BlockOp>(
    cpu: &mut CpuEngine,
    op: &B,
    jobs: &mut [B::Job],
    scratch: &mut LaneScratch,
) -> Result<(), DenseError> {
    let entered = match B::ENTRY {
        Entry::Contraction => {
            cpu.with_contraction_exec(|exec, _, _| on_lanes(exec, op, jobs, scratch))
        }
        Entry::Linalg => cpu.with_linalg_pool(|exec, _| on_lanes(exec, op, jobs, scratch)),
    };
    entered
        .map_err(|err| tenferro_error(B::LABEL, err))?
        .map_err(|(_, error)| error)
}

fn on_lanes<B: BlockOp>(
    exec: &mut Exec<'_>,
    op: &B,
    jobs: &mut [B::Job],
    scratch: &mut LaneScratch,
) -> tenferro_tensor::Result<Result<(), (usize, DenseError)>> {
    let model = exec.batch_policy().model;
    let mut total_work = 0.0;
    scratch.times.clear();
    // One `extend` reserves from the job iterator's size hint up front, as
    // the former per-kernel cost staging did, instead of growing by pushes.
    scratch
        .times
        .extend(jobs.iter().enumerate().map(|(index, job)| {
            let work = op.work(index, job);
            total_work += work;
            cost(B::KIND, work, &model)
        }));
    let mean_work = total_work / jobs.len().max(1) as f64;
    let shared: &Exec<'_> = exec;
    // Admission is read only where the allocation can search, so a serial or
    // forced allocation computes no grant input for its blocks.
    let modeled = schedule::searchable(shared, &scratch.times)
        && jobs
            .iter()
            .enumerate()
            .all(|(index, job)| op.modeled(index, job));
    let grant = modeled.then_some(|j: usize, serial: Duration, budget: usize| {
        op.grant(j, &jobs[j], shared, budget, serial)
    });
    let allocation = schedule::allocate_in(
        shared,
        &scratch.times,
        (B::KIND, |j| op.grant_key(j, &jobs[j])),
        mean_work,
        &mut scratch.allocation,
        schedule::lane_allocations(),
        grant,
    );
    let result = schedule::run(exec.reborrow(), &allocation, jobs, |index, job, exec| {
        op.run(index, job, exec)
    });
    scratch.allocation.keep(allocation);
    // A refused lane split is the engine's typed failure under this kernel's
    // label, like a refused entry.
    result.map_err(|refused| tenferro_tensor::Error::backend_source("Exec::fork_sized", refused))
}
