//! The block kernels' one lane driver: a kernel describes one block's work,
//! model and execution ([`BlockOp`]); the driver enters the kernel's guarded
//! engine entry, turns the work into times with the one cost function (a
//! composite block summing its stages', see [`BlockOp::serial`]), allocates
//! the lanes with [`lanes::allocate`] under the kernel's model and runs every
//! block on [`lanes::run_native`], returning the lowest-indexed failure.
//!
//! The model reaches the placement along one path: [`BlockOp::serial`] gives
//! the per-block times, and [`BlockOp::grant`] with [`BlockOp::item`] the
//! stage widths and the modeled time at every lane width, which every
//! candidate scores with and the reservations widen by. The weighted work
//! ([`BlockOp::work`]) feeds only the batch decision's query.

use std::time::Duration;

use tcontract_exec::{Exec, WorkModel};
use tenferro_cpu::CpuEngine;

use crate::lanes::{self, Grant, GrantKey, JobModel, LaneAllocationCache, LaneTables};
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
/// family of its grant. A composite factorization block is of the
/// factorization kind; its products are priced at the GEMM rate within it
/// (see [`BlockOp::serial`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum OpKind {
    /// tcontract's GEMM rate: the plan that runs the block is tcontract's.
    Gemm,
    /// The engine's `WorkModel`, from which tlinalg makes its own width
    /// decision for a factorization (see `factor::grant`).
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
/// Each kind uses the model of the dependency that executes the block, so the
/// scheduler and that dependency agree on the job's cost: tcontract's
/// `ns_per_flop` over `2 work` flops for GEMM, the engine work model (as
/// `tcontract-exec` `item_ns`) for a factorization. Neither is calibrated
/// here; both move to the providers' own estimates when they publish them.
pub(crate) fn cost(kind: OpKind, work: f64, model: &WorkModel) -> Duration {
    estimate(match kind {
        OpKind::Gemm => 2.0 * work * tcontract::ns_per_flop(),
        OpKind::Factorization => work_model_ns(model, work),
    })
}

/// A lane's blocks, handed out one by one to the visitor it is called with.
pub(crate) type LaneMembers<'v, J> = dyn FnMut(&mut dyn FnMut(&mut J)) + 'v;

/// One kernel's blocks: what the driver needs to cost and run each one.
/// Validation that must precede the entry, and the outputs a block produces,
/// stay with the kernel.
pub(crate) trait BlockOp: Sync {
    /// One block's exclusive state: its validated operands and its output.
    type Job: Send;
    /// One lane's state, made on the caller before the fork; its blocks run
    /// with it in order on the lane's thread.
    type Lane: Send;
    const ENTRY: Entry;
    const KIND: OpKind;
    /// The `DenseError` label of a refused entry.
    const LABEL: &'static str;

    /// The block's weighted multiply-adds, as its provider estimates them:
    /// the batch decision's mean work, not a time.
    fn work(&self, index: usize, job: &Self::Job) -> f64;

    /// Block `index`'s serial time, from its weighted work `work`: the
    /// kind's [`cost`], or for a composite block of sequential stages the
    /// saturating sum of its stages' own costs.
    fn serial(&self, _: usize, _: &Self::Job, model: &WorkModel, work: f64) -> Duration {
        cost(Self::KIND, work, model)
    }

    /// Whether [`grant`](Self::grant) models block `index`'s provider
    /// width. A batch holding a block it does not model keeps the baseline
    /// placement, without the search.
    fn modeled(&self, _: usize, _: &Self::Job) -> bool {
        true
    }

    /// The widths block `index`'s stages take on a lane of `budget`
    /// reserved slots of `exec`, for a block of serial time `serial`: the
    /// placement's model of each stage provider's own width decision,
    /// composed from that provider's public decisions ([`lanes::single`]
    /// for a single-stage block). Consulted only for a batch whose every
    /// block is [`modeled`](Self::modeled).
    fn grant(
        &self,
        index: usize,
        job: &Self::Job,
        exec: &Exec<'_>,
        budget: usize,
        serial: Duration,
    ) -> Grant;

    /// Block `index`'s modeled time on a lane of `exec` where its stages
    /// take `grant` (see [`lanes::item_time`]): a composite block sums its
    /// stages' times at their own widths, each parallel stage paying its own
    /// dispatch.
    fn item(
        &self,
        _: usize,
        _: &Self::Job,
        exec: &Exec<'_>,
        serial: Duration,
        grant: Grant,
    ) -> Duration {
        lanes::item_time(exec, serial, grant[0])
    }

    /// What block `index`'s [`grant`](Self::grant) depends on besides its
    /// serial time, the context and the op's [`OpKind`] (see [`GrantKey`]).
    fn grant_key(&self, index: usize, job: &Self::Job) -> GrantKey;

    /// Builds one lane's [`Lane`](Self::Lane) from that lane's blocks
    /// (`visit` hands out each one) on a context of the lane's share.
    fn lane(&self, jobs: &mut LaneMembers<'_, Self::Job>, exec: &Exec<'_>) -> Self::Lane;

    /// Runs block `index` on its lane's execution context with its lane's
    /// scratch.
    fn run(
        &self,
        index: usize,
        job: &mut Self::Job,
        lane: &mut Self::Lane,
        exec: Exec<'_>,
    ) -> Result<(), DenseError>;
}

/// The runner's view of a kernel's blocks.
struct Running<'a, B> {
    op: &'a B,
    count: usize,
}

impl<B: BlockOp> lanes::Work for Running<'_, B> {
    type Job = B::Job;
    type Lane = B::Lane;
    type Error = DenseError;
    fn count(&self) -> usize {
        self.count
    }
    fn run(
        &self,
        job: usize,
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

/// Places `jobs` of `op` on `exec` into `tables`: each block's work turned
/// into its serial time by [`BlockOp::serial`], and [`lanes::allocate`]
/// offered the blocks' model only where it can search and every block is
/// modeled, through `cache` (the process-global lane-allocation cache in
/// production).
///
/// On a context of one slot every block is priced by the kind's [`cost`]
/// alone: whatever the policy, its batch decision is one lane, so the
/// placement is one sequential lane and no time is read but the total, whose
/// width is one; a composite block's stages are not priced there.
pub(crate) fn place<B: BlockOp>(
    exec: &Exec<'_>,
    op: &B,
    jobs: &[B::Job],
    cache: &LaneAllocationCache,
    tables: &mut LaneTables,
) {
    let work_model = exec.batch_policy().model;
    let staged = exec.budget() >= 2;
    let mut total_work = 0.0;
    let mut times = tables.take_times();
    // One `extend` reserves from the job iterator's size hint up front.
    times.extend(jobs.iter().enumerate().map(|(index, job)| {
        let work = op.work(index, job);
        total_work += work;
        if staged {
            op.serial(index, job, &work_model, work)
        } else {
            cost(B::KIND, work, &work_model)
        }
    }));
    let mean_work = total_work / jobs.len().max(1) as f64;
    // Admission is read only where the allocation can search, so a serial or
    // forced allocation computes no grant input for its blocks.
    let modeled = lanes::searchable(exec, &times, mean_work)
        && jobs
            .iter()
            .enumerate()
            .all(|(index, job)| op.modeled(index, job));
    let model = modeled.then_some(Modeled { op, jobs, exec });
    lanes::allocate(
        exec,
        &times,
        (B::KIND, |j| op.grant_key(j, &jobs[j])),
        mean_work,
        model,
        cache,
        tables,
    );
    tables.keep_times(times);
}

/// A kernel's blocks as the placement's [`JobModel`].
struct Modeled<'a, 'e, B: BlockOp> {
    op: &'a B,
    jobs: &'a [B::Job],
    exec: &'a Exec<'e>,
}

impl<B: BlockOp> JobModel for Modeled<'_, '_, B> {
    fn grant(&self, j: usize, serial: Duration, width: usize) -> Grant {
        self.op.grant(j, &self.jobs[j], self.exec, width, serial)
    }

    fn item(&self, exec: &Exec<'_>, j: usize, serial: Duration, grant: Grant) -> Duration {
        self.op.item(j, &self.jobs[j], exec, serial, grant)
    }
}

fn on_lanes<B: BlockOp>(
    exec: &mut Exec<'_>,
    op: &B,
    jobs: &mut [B::Job],
    tables: &mut LaneTables,
) -> tenferro_tensor::Result<lanes::Outcome<DenseError>> {
    place(exec, op, jobs, lanes::lane_allocations(), tables);
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
