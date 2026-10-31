//! The native block GEMM kernel: a batch of disjoint column-major GEMMs, one
//! tcontract `Plan` per distinct geometry, run on the shared scheduler.
//!
//! TensorKit (`mul!` per matched sector) and QSpace (`contractDATA_group`)
//! set up each block GEMM per call; this kernel does the same per geometry,
//! so blocks that share a geometry share one prepared plan within the call.

use std::collections::HashMap;
use std::time::Duration;

use tcontract::api::{CSpec, Labels, LayoutSpec, Op, OperandSpec, Problem, Scalar};
use tcontract::{ns_per_flop, Plan, PlanConfig, SliceAccumulationSource};
use tenferro_cpu::CpuEngine;

use super::schedule;
use crate::tenferro_adapter::{grouped_error, grouped_validate, tenferro_error};
use crate::{DenseError, DenseGemmBatchJob, DenseView, DenseViewMut, MatrixOp};

/// Everything a plan depends on besides the scalar type, which a call fixes:
/// logical extents, the effective strides of A, B and D, and conjugation.
/// Transposition lives only in the strides, because tcontract operands carry
/// only Identity or Conjugate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Geometry {
    mkn: [usize; 3],
    strides: [[isize; 2]; 3],
    conj: [bool; 2],
}

impl Geometry {
    /// The existing op lowering: A is `[1, m]` (Identity) or `[k, 1]`, B is
    /// `[1, k]` or `[n, 1]`, D is `[1, m]`; only Adjoint conjugates. A zero
    /// extent strides as one, as the dependency's compact layouts do.
    fn new(job: &DenseGemmBatchJob, ops: [MatrixOp; 2]) -> Result<Self, DenseError> {
        let stride = |extent: usize| {
            isize::try_from(extent.max(1)).map_err(|_| DenseError::ElementCountOverflow)
        };
        let (m, k, n) = (
            stride(job.rows)?,
            stride(job.contracted)?,
            stride(job.cols)?,
        );
        let a = match ops[0] {
            MatrixOp::Identity => [1, m],
            MatrixOp::Transpose | MatrixOp::Adjoint => [k, 1],
        };
        let b = match ops[1] {
            MatrixOp::Identity => [1, k],
            MatrixOp::Transpose | MatrixOp::Adjoint => [n, 1],
        };
        Ok(Self {
            mkn: [job.rows, job.contracted, job.cols],
            strides: [a, b, [1, m]],
            conj: ops.map(|op| op == MatrixOp::Adjoint),
        })
    }

    fn plan<T: Scalar>(&self) -> tcontract::Result<Plan<T>> {
        let [m, k, n] = self.mkn;
        let spec = |dims: [usize; 2], strides: [isize; 2], conj: bool| {
            let op = if conj { Op::Conjugate } else { Op::Identity };
            LayoutSpec::new(&dims, &strides, 0).map(|layout| OperandSpec::new(layout).with_op(op))
        };
        let problem = Problem::from_labels(
            T::STORAGE,
            spec([m, k], self.strides[0], self.conj[0])?,
            spec([k, n], self.strides[1], self.conj[1])?,
            CSpec::Output(Op::Identity),
            spec([m, n], self.strides[2], false)?,
            &Labels::new(&[0, 1], &[1, 2], &[0, 2]),
        )?;
        Plan::new(&problem, &PlanConfig::default())
    }
}

/// Each job's serial time under the dependency's own GEMM model (tcontract's
/// `ns_per_flop` over `2 m k n` real flops, four times that for complex), and
/// the mean weighted work `m k n` (times four for complex) the engine's
/// batch decision takes.
pub(crate) fn costs<T: Scalar>(jobs: &[DenseGemmBatchJob]) -> (Vec<Duration>, f64) {
    let complex = if T::IS_COMPLEX { 4.0 } else { 1.0 };
    let work = |job: &DenseGemmBatchJob| {
        job.rows as f64 * job.contracted as f64 * job.cols as f64 * complex
    };
    let times = jobs
        .iter()
        .map(|job| schedule::estimate(2.0 * work(job) * ns_per_flop()))
        .collect();
    let mean = jobs.iter().map(work).sum::<f64>() / jobs.len().max(1) as f64;
    (times, mean)
}

/// A job's operand range; an empty one may start anywhere, as validation
/// does not bound it.
fn operand<T>(data: &[T], offset: usize, len: usize) -> &[T] {
    if len == 0 {
        &[]
    } else {
        &data[offset..offset + len]
    }
}

/// `D_j = alpha * op(A_j) op(B_j) + beta * D_j` for every job, on the
/// caller's engine.
///
/// Every operand range and destination overlap is checked before the first
/// write. Each distinct geometry is prepared once per call. The jobs then run
/// on the scheduler's lanes, each into its own disjoint destination slice, so
/// lane order cannot change a result; every job runs, and the error returned
/// is the lowest-indexed job's. `beta == 0` reads no destination value.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_batch_into<T: Scalar + Default>(
    cpu: &mut CpuEngine,
    output: &mut DenseViewMut<'_, T>,
    lhs: DenseView<'_, T>,
    rhs: DenseView<'_, T>,
    jobs: &[DenseGemmBatchJob],
    ops: [MatrixOp; 2],
    alpha: T,
    beta: T,
) -> Result<(), DenseError> {
    let [lhs_len, rhs_len, out_len] = grouped_validate(output, lhs, rhs, jobs)?;

    // Sized up front: a filtered collect would regrow about log2(J) times.
    let mut order = Vec::with_capacity(jobs.len());
    order.extend((0..jobs.len()).filter(|&j| jobs[j].rows * jobs[j].cols != 0));
    order.sort_unstable_by_key(|&j| (jobs[j].dst_offset, j));
    for pair in order.windows(2) {
        let (first, second) = (&jobs[pair[0]], &jobs[pair[1]]);
        if first.dst_offset + first.rows * first.cols > second.dst_offset {
            return Err(grouped_error(format!(
                "grouped output jobs {} and {} overlap",
                pair[0].min(pair[1]),
                pair[0].max(pair[1])
            )));
        }
    }

    // Index the distinct geometries first, so the plans are built in one
    // exactly sized allocation, in order of first use.
    let mut geometries = HashMap::new();
    let mut slots: Vec<(usize, &mut [T])> = Vec::with_capacity(jobs.len());
    for job in jobs {
        let next = geometries.len();
        let plan = *geometries.entry(Geometry::new(job, ops)?).or_insert(next);
        slots.push((plan, &mut [][..]));
    }
    let mut plans: Vec<Plan<T>> = Vec::with_capacity(geometries.len());
    for (index, job) in jobs.iter().enumerate() {
        if slots[index].0 == plans.len() {
            plans.push(
                Geometry::new(job, ops)?
                    .plan()
                    .map_err(|err| grouped_error(format!("job {index}: {err}")))?,
            );
            #[cfg(test)]
            crate::probe::add(crate::probe::Event::GemmPlanBuild, 1);
        }
    }
    let (times, mean_work) = costs::<T>(jobs);

    let lhs = &lhs.data()[lhs.offset()..lhs.offset() + lhs_len];
    let rhs = &rhs.data()[rhs.offset()..rhs.offset() + rhs_len];
    let offset = output.offset();
    let mut remaining = &mut output.data_mut()[offset..offset + out_len];
    let mut cursor = 0;
    for &j in &order {
        let job = &jobs[j];
        let (_, rest) = std::mem::take(&mut remaining).split_at_mut(job.dst_offset - cursor);
        let (slice, tail) = rest.split_at_mut(job.rows * job.cols);
        slots[j].1 = slice;
        remaining = tail;
        cursor = job.dst_offset + job.rows * job.cols;
    }

    let source = if beta == T::default() {
        SliceAccumulationSource::Absent
    } else {
        SliceAccumulationSource::Output
    };
    let run = |j: usize, slot: &mut (usize, &mut [T]), mut child: tcontract_exec::Exec<'_>| {
        let job = &jobs[j];
        let (plan, destination) = slot;
        plans[*plan]
            .execute_slices_accum(
                &mut child,
                alpha,
                (operand(lhs, job.lhs_offset, job.rows * job.contracted), 0),
                (operand(rhs, job.rhs_offset, job.contracted * job.cols), 0),
                beta,
                source,
                (&mut **destination, 0),
            )
            .map_err(|err| grouped_error(format!("job {j}: {err}")))
    };
    cpu.with_contraction_exec(|exec, _, _| {
        let allocation = schedule::allocate(exec, &times, mean_work);
        Ok(schedule::run(exec.reborrow(), &allocation, &mut slots, run))
    })
    .map_err(|err| tenferro_error("grouped_gemm", err))?
    .map_err(|(_, error)| error)
}
