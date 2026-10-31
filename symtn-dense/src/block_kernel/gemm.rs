//! The native block GEMM kernel: a batch of disjoint column-major GEMMs, one
//! tcontract `Plan` per distinct geometry, run on the shared scheduler.
//!
//! TensorKit (`mul!` per matched sector) and QSpace (`contractDATA_group`)
//! set up each block GEMM per call. A tcontract plan is immutable data of its
//! geometry alone, so the kernel keeps each in the process-global
//! [`StructureCacheKind::GemmPlans`] cache: a repeated geometry, in this call
//! or a later one, reuses the prepared plan.

use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use symtn_core::{
    ErasedStructureCacheControl, StructureCache, StructureCacheInfo, StructureCacheKind,
};
use tcontract::api::{CSpec, DType, Labels, LayoutSpec, Op, OperandSpec, Problem, Scalar};
use tcontract::{Plan, PlanConfig, SliceAccumulationSource};
use tcontract_exec::Exec;
use tenferro_cpu::CpuEngine;

use super::driver::{self, BlockOp, Entry, LaneScratch, OpKind};
use crate::tenferro_adapter::{grouped_error, grouped_validate};
use crate::{DenseError, DenseGemmBatchJob, DenseView, DenseViewMut, MatrixOp};

/// Everything a plan depends on besides the scalar type, which a call fixes:
/// logical extents, the effective strides of A, B and D, and conjugation.
/// Transposition lives only in the strides, because tcontract operands carry
/// only Identity or Conjugate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Geometry {
    pub(crate) mkn: [usize; 3],
    pub(crate) strides: [[isize; 2]; 3],
    pub(crate) conj: [bool; 2],
}

impl Geometry {
    /// The existing op lowering: A is `[1, m]` (Identity) or `[k, 1]`, B is
    /// `[1, k]` or `[n, 1]`, D is `[1, m]`; only Adjoint conjugates. A zero
    /// extent strides as one, as the dependency's compact layouts do.
    pub(crate) fn new(job: &DenseGemmBatchJob, ops: [MatrixOp; 2]) -> Result<Self, DenseError> {
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

    pub(crate) fn plan<T: Scalar>(&self) -> tcontract::Result<Plan<T>> {
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
        Plan::from_problem(problem, &PlanConfig::default())
    }
}

/// The complete key of a cached plan: a plan is a pure function of its
/// problem and config, and the labels, the in-place accumulation mode and
/// `PlanConfig::default()` are fixed by [`Geometry::plan`], so the scalar
/// type and the geometry determine it. The provider is a build choice and
/// the hardware model is fixed for the process, so neither varies within one
/// cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct GemmPlanKey {
    dtype: DType,
    geometry: Geometry,
}

type ErasedPlan = dyn Any + Send + Sync;

pub(crate) type GemmPlanCache = StructureCache<GemmPlanKey, ErasedPlan>;

/// quick_cache node and the entry's `Arc` control, per entry (as the
/// completed-transformer cache charges).
const ENTRY_OVERHEAD_BYTES: usize = 10 * core::mem::size_of::<usize>();

/// Heap a plan owns besides its scatter vectors: the problem's layouts and
/// roles, the report and diagnostics, a Packed plan's box with its resolved
/// kernel and folded axes. Checked against live allocations for every dtype,
/// strategy and op lowering by the dense `gemm_plan_charge` test.
const PLAN_HEAP_ALLOWANCE: usize = 16 * 1024;

/// The bytes charged for one cached plan: its inline size, the key, the
/// entry overhead and a bound on its heap. A Packed plan owns one `i64`
/// scatter entry per role element (`a_m, c_m, d_m` and `b_n, c_n, d_n` of
/// `m` and `n` entries, `a_k, b_k` of `k`, four batch entries), extents
/// counted as at least one; a Faer plan owns less. Saturates.
pub(crate) fn charge<T: Scalar>(key: &GemmPlanKey) -> u64 {
    let [m, k, n] = key.geometry.mkn.map(|extent| extent.max(1) as u64);
    let scatter = m
        .saturating_add(n)
        .saturating_mul(3)
        .saturating_add(k.saturating_mul(2))
        .saturating_add(4)
        .saturating_mul(core::mem::size_of::<i64>() as u64);
    [
        core::mem::size_of::<Plan<T>>(),
        core::mem::size_of::<GemmPlanKey>(),
        ENTRY_OVERHEAD_BYTES,
        PLAN_HEAP_ALLOWANCE,
    ]
    .into_iter()
    .fold(scatter, |total, bytes| total.saturating_add(bytes as u64))
}

struct GemmPlansControl;

impl ErasedStructureCacheControl for GemmPlansControl {
    fn info(&self) -> StructureCacheInfo {
        gemm_plans().info()
    }

    fn clear(&self) {
        ErasedStructureCacheControl::clear(gemm_plans())
    }

    fn set_byte_budget(&self, bytes: u64) {
        ErasedStructureCacheControl::set_byte_budget(gemm_plans(), bytes)
    }
}

/// The process-global geometry-plan cache, registered with `symtn-core`'s
/// structure-cache registry on first use.
fn gemm_plans() -> &'static GemmPlanCache {
    static CACHE: OnceLock<GemmPlanCache> = OnceLock::new();
    CACHE.get_or_init(|| {
        // Refused only if another crate claimed the slot first. Then
        // `symtn::cache::clear` cannot reach this cache, so it admits
        // nothing: every geometry builds per call, with identical results.
        let budget =
            symtn_core::register_structure_cache(StructureCacheKind::GemmPlans, &GemmPlansControl)
                .unwrap_or(0);
        StructureCache::new(StructureCacheKind::GemmPlans, budget)
    })
}

/// The plan of `geometry` at scalar type `T` from `cache`, built and offered
/// for admission on a miss. A rejected or unadmitted build is returned
/// uncached, so a budget of zero builds per lookup with the same plan.
///
/// Why single-flight here, unlike the transformer caches: a plan build is
/// index analysis on the calling thread and runs no engine region, so a
/// thread waiting on another's build cannot be holding work that build
/// needs.
pub(crate) fn cached_plan<T: Scalar>(
    cache: &GemmPlanCache,
    geometry: Geometry,
) -> tcontract::Result<Arc<Plan<T>>> {
    let key = GemmPlanKey {
        dtype: T::STORAGE,
        geometry,
    };
    let epoch = symtn_core::core_reset_epoch();
    let plan = cache.get_or_try_build(&key, epoch, || {
        let plan: Arc<ErasedPlan> = Arc::new(geometry.plan::<T>()?);
        #[cfg(test)]
        crate::probe::add(crate::probe::Event::GemmPlanBuild, 1);
        Ok::<_, tcontract::Error>((plan, charge::<T>(&key)))
    })?;
    Ok(plan
        .downcast::<Plan<T>>()
        .unwrap_or_else(|_| unreachable!("the key's dtype fixes the plan type")))
}

/// A job's weighted work: `m k n` multiply-adds, four times that for a
/// complex scalar.
pub(crate) fn work<T: Scalar>(job: &DenseGemmBatchJob) -> f64 {
    let complex = if T::IS_COMPLEX { 4.0 } else { 1.0 };
    job.rows as f64 * job.contracted as f64 * job.cols as f64 * complex
}

/// One call's blocks: each job's operands, its plan and the scalars.
struct GemmBlocks<'a, T> {
    jobs: &'a [DenseGemmBatchJob],
    plans: &'a [Arc<ErasedPlan>],
    lhs: &'a [T],
    rhs: &'a [T],
    alpha: T,
    beta: T,
    source: SliceAccumulationSource<'a, T>,
}

impl<'a, T: Scalar> BlockOp for GemmBlocks<'a, T> {
    /// The job's plan index and its destination slice.
    type Job = (usize, &'a mut [T]);
    const ENTRY: Entry = Entry::Contraction;
    const KIND: OpKind = OpKind::Gemm;
    const LABEL: &'static str = "grouped_gemm";

    fn work(&self, index: usize, _: &Self::Job) -> f64 {
        work::<T>(&self.jobs[index])
    }

    fn run(
        &self,
        index: usize,
        (plan, destination): &mut Self::Job,
        mut exec: Exec<'_>,
    ) -> Result<(), DenseError> {
        let job = &self.jobs[index];
        let plan = self.plans[*plan]
            .downcast_ref::<Plan<T>>()
            .unwrap_or_else(|| unreachable!("the key's dtype fixes the plan type"));
        plan.execute_slices_accum(
            &mut exec,
            self.alpha,
            (
                operand(self.lhs, job.lhs_offset, job.rows * job.contracted),
                0,
            ),
            (
                operand(self.rhs, job.rhs_offset, job.contracted * job.cols),
                0,
            ),
            self.beta,
            self.source,
            (&mut **destination, 0),
        )
        .map_err(|err| grouped_error(format!("job {index}: {err}")))
    }
}

/// The kernel's per-call staging: the destination order, the geometry index,
/// the per-job plan slots, the plan handles and the lane driver's staging.
/// An owner that submits repeatedly (an executor of a prepared workspace)
/// keeps one, so a warm submission reuses every capacity and allocates
/// nothing; an eager executor's scratch lives for its one call. Holds no
/// plan between calls: the handles are released when a call ends, and the
/// geometry-plan cache alone retains plans.
#[derive(Debug, Default)]
pub(crate) struct GemmScratch {
    order: Vec<usize>,
    geometries: HashMap<Geometry, usize>,
    slots: Vec<(usize, &'static mut [u8])>,
    plans: Vec<Arc<ErasedPlan>>,
    lanes: LaneScratch,
}

impl GemmScratch {
    /// A conservative charge of the capacities this scratch keeps: each
    /// vector's capacity, and the geometry index's table at its largest
    /// bucket count for its capacity (entries plus one control byte each,
    /// plus one group of trailing control bytes). Saturates.
    pub(crate) fn retained_bytes(&self) -> usize {
        use std::mem::size_of;
        let index_buckets = self
            .geometries
            .capacity()
            .saturating_mul(8)
            .div_ceil(7)
            .max(1)
            .next_power_of_two();
        [
            self.order.capacity().saturating_mul(size_of::<usize>()),
            index_buckets
                .saturating_mul(size_of::<(Geometry, usize)>() + 1)
                .saturating_add(16),
            self.slots
                .capacity()
                .saturating_mul(size_of::<(usize, &mut [u8])>()),
            self.plans
                .capacity()
                .saturating_mul(size_of::<Arc<ErasedPlan>>()),
            self.lanes.retained_bytes(),
        ]
        .into_iter()
        .fold(0, usize::saturating_add)
    }
}

/// `vec`'s allocation, emptied, for elements of another lifetime or type of
/// the same layout.
///
/// Why not a transmute: the crate forbids unsafe code. std collects a mapped
/// `vec::IntoIter` in place when the layouts match; that is an optimization,
/// not a documented guarantee, so the prepared-workspace allocation tests pin
/// it (a regression would reallocate, never misbehave).
fn recycle<A, B>(mut vec: Vec<A>) -> Vec<B> {
    vec.clear();
    vec.into_iter().map(|_| unreachable!("emptied")).collect()
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
/// caller's engine, staged in `scratch`.
///
/// Every operand range and destination overlap is checked before the first
/// write. Each distinct geometry's plan is looked up once per call in the
/// geometry-plan cache. The jobs then run on the scheduler's lanes, each into
/// its own disjoint destination slice, so lane order cannot change a result;
/// every job runs, and the error returned is the lowest-indexed job's.
/// `beta == 0` reads no destination value.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gemm_batch_into<T: Scalar + Default>(
    cpu: &mut CpuEngine,
    scratch: &mut GemmScratch,
    output: &mut DenseViewMut<'_, T>,
    lhs: DenseView<'_, T>,
    rhs: DenseView<'_, T>,
    jobs: &[DenseGemmBatchJob],
    ops: [MatrixOp; 2],
    alpha: T,
    beta: T,
) -> Result<(), DenseError> {
    let [lhs_len, rhs_len, out_len] = grouped_validate(output, lhs, rhs, jobs)?;
    let GemmScratch {
        order,
        geometries,
        slots: kept_slots,
        plans,
        lanes,
    } = scratch;

    order.clear();
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

    // Index the distinct geometries, in order of first use.
    geometries.clear();
    let mut slots: Vec<(usize, &mut [T])> = recycle(std::mem::take(kept_slots));
    slots.reserve(jobs.len());
    for job in jobs {
        let next = geometries.len();
        let plan = *geometries.entry(Geometry::new(job, ops)?).or_insert(next);
        slots.push((plan, &mut [][..]));
    }
    plans.clear();
    for (index, job) in jobs.iter().enumerate() {
        if slots[index].0 == plans.len() {
            let plan: Arc<ErasedPlan> = cached_plan::<T>(gemm_plans(), Geometry::new(job, ops)?)
                .map_err(|err| grouped_error(format!("job {index}: {err}")))?;
            plans.push(plan);
            #[cfg(test)]
            crate::probe::add(crate::probe::Event::GemmPlanLookup, 1);
        }
    }

    let lhs = &lhs.data()[lhs.offset()..lhs.offset() + lhs_len];
    let rhs = &rhs.data()[rhs.offset()..rhs.offset() + rhs_len];
    let offset = output.offset();
    let mut remaining = &mut output.data_mut()[offset..offset + out_len];
    let mut cursor = 0;
    for &j in order.iter() {
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
    let blocks = GemmBlocks {
        jobs,
        plans,
        lhs,
        rhs,
        alpha,
        beta,
        source,
    };
    let result = driver::drive(cpu, &blocks, &mut slots, lanes);
    *kept_slots = recycle(slots);
    plans.clear();
    result
}
