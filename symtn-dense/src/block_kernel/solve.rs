//! The native provider's batch of independent linear solves
//! ([`crate::DenseExecutor::solve_batch_into`]): every valid compact
//! rank-two system on the lane driver's lanes, each through the
//! partial-pivot LU call tenferro's per-matrix `solve_into` makes on its
//! direct route (`solve_in_place_view` with a copied right-hand side), and
//! every other system through the executor's own per-matrix entry.
//!
//! The provider copies each system's right-hand side into its destination
//! only after that system's factorization and singularity check succeed, so
//! the copy runs on the lane and a singular system's destination is left as
//! it was. One provider workspace per lane, sized from the lane's systems on
//! its context, is allocated on the lane's thread before its first system
//! and dropped after its last.

use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::time::Duration;

use num_complex::{Complex32, Complex64};
use strided_view::{RawStridedMut, RawStridedRef};
use tcontract_exec::Exec;
use tenferro_cpu::with_cpu_engine;
use tenferro_tensor::StrideVec;

use super::driver::{self, BlockOp, Entry, LaneMembers, OpKind};
use super::factor::{grant, linalg_error};
use crate::lanes::{self, Grant, GrantKey, LaneTables};
use crate::layout::strides_to_isize;
use crate::tenferro_adapter::{tenferro_solve_error, NOT_A_CPU_ENGINE};
use crate::{DenseError, DenseExecutor, DenseRead, DenseView, DenseViewMut, DenseWrite, Engine};

/// The per-matrix entry's label.
const ENTRY: &str = "solve_into";
/// The route tenferro names a singular pivot after on its direct route.
const DIRECT: &str = "solve_read_into";

/// The lane form of one system: its extents and both reads' signed strides
/// and offsets (`x` is compact `[1, n]`, its offset its own).
struct Admitted {
    n: usize,
    k: usize,
    a: (StrideVec, isize),
    b: (StrideVec, isize),
    x: isize,
}

/// Whether the system `(a, b, x)` lies in the valid compact rank-two subset
/// of tenferro's direct route: one floating dtype, `a` `[n, n]` and `b`, `x`
/// `[n, k]` with `n, k > 0`, `x` strided `[1, n]`, every stride and offset
/// representable. Every other system, including ones tenferro still solves
/// directly (a vector or padded destination), takes the per-matrix entry.
fn admit(a: DenseRead<'_>, b: DenseRead<'_>, x: &DenseWrite<'_>) -> Option<Admitted> {
    fn read(input: DenseRead<'_>) -> Option<(&[usize], &[usize], usize)> {
        match input {
            DenseRead::F32(v) => Some((v.shape(), v.strides(), v.offset())),
            DenseRead::F64(v) => Some((v.shape(), v.strides(), v.offset())),
            DenseRead::C32(v) => Some((v.shape(), v.strides(), v.offset())),
            DenseRead::C64(v) => Some((v.shape(), v.strides(), v.offset())),
            _ => None,
        }
    }
    let (x_strides, x_offset) = match x {
        DenseWrite::F32(v) => (v.strides(), v.offset()),
        DenseWrite::F64(v) => (v.strides(), v.offset()),
        DenseWrite::C32(v) => (v.strides(), v.offset()),
        DenseWrite::C64(v) => (v.strides(), v.offset()),
        _ => return None,
    };
    if a.dtype() != b.dtype() || b.dtype() != x.dtype() {
        return None;
    }
    let (a_shape, a_strides, a_offset) = read(a)?;
    let (b_shape, b_strides, b_offset) = read(b)?;
    let (&[n, n2], &[rows, k]) = (a_shape, b_shape) else {
        return None;
    };
    if n == 0 || k == 0 || n != n2 || rows != n || x.shape() != b_shape || x_strides != [1, n] {
        return None;
    }
    let signed = |strides, offset: usize| {
        Some((
            strides_to_isize(strides).ok()?,
            isize::try_from(offset).ok()?,
        ))
    };
    Some(Admitted {
        n,
        k,
        a: signed(a_strides, a_offset)?,
        b: signed(b_strides, b_offset)?,
        x: isize::try_from(x_offset).ok()?,
    })
}

/// One admitted system: its original index, operands, destination, a failed
/// workspace query, and once run its outcome.
struct SolveJob<'a, 'x, 'w> {
    index: usize,
    a: DenseRead<'a>,
    b: DenseRead<'a>,
    x: &'x mut DenseWrite<'w>,
    form: Admitted,
    sizing: Option<DenseError>,
    outcome: Option<Result<(), DenseError>>,
}

/// One lane: its systems' largest workspace, the count still to run and,
/// from its first system's run to its last, the workspace itself (built on
/// the lane's thread, as [`super::factor::IntoLane`]'s buffers are).
struct SolveLane {
    bytes: usize,
    remaining: usize,
    workspace: Option<Vec<MaybeUninit<u8>>>,
    /// A process-unique id of this lane's setup and its allocation calls.
    #[cfg(test)]
    setup: (usize, Option<u64>),
}

/// The solves of admitted systems on the lane driver.
struct SolveBlocks<'a, 'x, 'w>(PhantomData<SolveMarker<'a, 'x, 'w>>);

/// [`SolveBlocks`]' variance and auto-trait marker.
type SolveMarker<'a, 'x, 'w> = fn() -> (&'a (), &'x (), &'w ());

/// tlinalg's (total, payback) work estimates of one system.
fn estimates(job: &SolveJob<'_, '_, '_>) -> [f64; 2] {
    fn of<T: tlinalg::FaerScalar>(n: usize, k: usize) -> [f64; 2] {
        use tlinalg::lu::{estimate_solve_payback_work, estimate_solve_work};
        [
            estimate_solve_work::<T>(n, k),
            estimate_solve_payback_work::<T>(n, k),
        ]
    }
    let Admitted { n, k, .. } = job.form;
    match job.a {
        DenseRead::F32(_) => of::<f32>(n, k),
        DenseRead::F64(_) => of::<f64>(n, k),
        DenseRead::C32(_) => of::<Complex32>(n, k),
        _ => of::<Complex64>(n, k),
    }
}

/// tenferro's mapping of a provider failure on the direct route.
fn provider_failure(error: tlinalg::Error) -> DenseError {
    tenferro_solve_error(match error {
        tlinalg::Error::Singular { .. } => {
            linalg_error(DIRECT, tenferro_linalg::Error::Singular { op: DIRECT })
        }
        other => tenferro_linalg::error::map_error(tlinalg::Op::Solve, other),
    })
}

impl<'a, 'x, 'w: 'x> BlockOp for SolveBlocks<'a, 'x, 'w> {
    type Job = SolveJob<'a, 'x, 'w>;
    type Lane = SolveLane;
    const ENTRY: Entry = Entry::Linalg;
    const KIND: OpKind = OpKind::Factorization;
    const LABEL: &'static str = ENTRY;

    fn work(&self, _: usize, job: &Self::Job) -> f64 {
        estimates(job)[0]
    }

    fn grant(
        &self,
        _: usize,
        job: &Self::Job,
        exec: &Exec<'_>,
        budget: usize,
        _: Duration,
    ) -> Grant {
        lanes::single(grant(exec, budget, estimates(job)))
    }

    fn grant_key(&self, _: usize, job: &Self::Job) -> GrantKey {
        let [total, payback] = estimates(job).map(f64::to_bits);
        [total, payback, 0]
    }

    fn lane(&self, jobs: &mut LaneMembers<'_, Self::Job>, exec: &Exec<'_>) -> SolveLane {
        fn query<T: tlinalg::FaerScalar>(
            n: usize,
            k: usize,
            exec: &Exec<'_>,
        ) -> tlinalg::Result<usize> {
            tlinalg::lu::lu_solve_workspace::<T>(n, k, false, 1, exec)
        }
        let (mut bytes, mut remaining) = (0, 0);
        jobs(&mut |job| {
            remaining += 1;
            let Admitted { n, k, .. } = job.form;
            let queried = match job.a {
                DenseRead::F32(_) => query::<f32>(n, k, exec),
                DenseRead::F64(_) => query::<f64>(n, k, exec),
                DenseRead::C32(_) => query::<Complex32>(n, k, exec),
                _ => query::<Complex64>(n, k, exec),
            };
            match queried {
                Ok(need) => bytes = bytes.max(need),
                Err(error) => {
                    job.sizing = Some(tenferro_solve_error(tenferro_linalg::error::map_error(
                        tlinalg::Op::Solve,
                        error,
                    )))
                }
            }
        });
        SolveLane {
            bytes,
            remaining,
            workspace: None,
            #[cfg(test)]
            setup: (0, None),
        }
    }

    // A system's failure is its own outcome; the batch reduces the outcomes
    // in index order after the lanes join.
    fn run(
        &self,
        _: usize,
        job: &mut Self::Job,
        lane: &mut SolveLane,
        mut exec: Exec<'_>,
    ) -> Result<(), DenseError> {
        let bytes = lane.bytes;
        let build = || vec![MaybeUninit::uninit(); bytes];
        if lane.workspace.is_none() {
            #[cfg(test)]
            let workspace = {
                static SETUPS: std::sync::atomic::AtomicUsize =
                    std::sync::atomic::AtomicUsize::new(1);
                let (workspace, calls) = super::factor::owners::measured(build);
                lane.setup = (
                    SETUPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                    calls,
                );
                workspace
            };
            #[cfg(not(test))]
            let workspace = build();
            lane.workspace = Some(workspace);
        }
        let workspace = lane.workspace.as_mut().expect("set above");
        let sizing = job.sizing.take();
        let solve = |job: &mut Self::Job| match sizing {
            Some(error) => Err(error),
            None => solve_one(job.a, job.b, job.x, &job.form, workspace, &mut exec),
        };
        #[cfg(test)]
        let outcome = {
            let (outcome, calls) = super::factor::owners::measured(|| solve(job));
            let input = super::factor::owners::address(job.a);
            super::factor::owners::record(input, lane.setup.0, lane.setup.1, calls, 0);
            outcome
        };
        #[cfg(not(test))]
        let outcome = solve(job);
        job.outcome = Some(outcome);
        lane.remaining = lane.remaining.saturating_sub(1);
        if lane.remaining == 0 {
            lane.workspace = None;
        }
        Ok(())
    }
}

/// One admitted system on the lane's workspace and context.
fn solve_one(
    a: DenseRead<'_>,
    b: DenseRead<'_>,
    x: &mut DenseWrite<'_>,
    form: &Admitted,
    workspace: &mut [MaybeUninit<u8>],
    exec: &mut Exec<'_>,
) -> Result<(), DenseError> {
    fn typed<T: tlinalg::FaerScalar>(
        a: DenseView<'_, T>,
        b: DenseView<'_, T>,
        x: &mut DenseViewMut<'_, T>,
        form: &Admitted,
        workspace: &mut [MaybeUninit<u8>],
        exec: &mut Exec<'_>,
    ) -> Result<(), DenseError> {
        let layout = |role: &'static str, err: strided_view::StridedError| {
            tenferro_solve_error(tenferro_tensor::Error::invalid_argument(
                DIRECT,
                role,
                err.to_string(),
            ))
        };
        let x_strides = [1, form.n as isize];
        let a = RawStridedRef::new(a.data(), a.shape(), &form.a.0, form.a.1)
            .map_err(|e| layout("a", e))?;
        let b = RawStridedRef::new(b.data(), b.shape(), &form.b.0, form.b.1)
            .map_err(|e| layout("b", e))?;
        let shape = x.shape();
        let x = RawStridedMut::new(x.data_mut(), shape, &x_strides, form.x)
            .map_err(|e| layout("out", e))?;
        tlinalg::lu::solve_into(tlinalg::Op::Solve, a, Some(b), x, false, workspace, exec)
            .map_err(provider_failure)
    }
    match (a, b, x) {
        (DenseRead::F32(a), DenseRead::F32(b), DenseWrite::F32(x)) => {
            typed(a, b, x, form, workspace, exec)
        }
        (DenseRead::F64(a), DenseRead::F64(b), DenseWrite::F64(x)) => {
            typed(a, b, x, form, workspace, exec)
        }
        (DenseRead::C32(a), DenseRead::C32(b), DenseWrite::C32(x)) => {
            typed(a, b, x, form, workspace, exec)
        }
        (DenseRead::C64(a), DenseRead::C64(b), DenseWrite::C64(x)) => {
            typed(a, b, x, form, workspace, exec)
        }
        _ => unreachable!("admission checked one floating dtype"),
    }
}

/// The native [`DenseExecutor::solve_batch_into`] over length-checked
/// slices: one reduction over the original system indices.
///
/// With `f` the lowest admitted index: the per-matrix systems below `f` run
/// first, in order, the first error returned; then the lanes solve every
/// admitted system, each keeping its own outcome, and a failure to reach
/// them (a wrong engine, a refused engine entry, a refused fork) is the
/// outcome of system `f`, the first that needed them; then the per-matrix
/// systems above `f` and below the lowest failing admitted index run in
/// order. The lowest-indexed failure is returned.
pub(crate) fn solve_batch_into<E: DenseExecutor + ?Sized>(
    executor: &mut E,
    engine: &mut dyn Engine,
    a: &[DenseRead<'_>],
    b: &[DenseRead<'_>],
    x: &mut [DenseWrite<'_>],
) -> Result<(), DenseError> {
    let mut jobs = Vec::new();
    let mut serial = Vec::new();
    for (index, ((&a, &b), x)) in a.iter().zip(b).zip(x.iter_mut()).enumerate() {
        match admit(a, b, x) {
            Some(form) => jobs.push(SolveJob {
                index,
                a,
                b,
                x,
                form,
                sizing: None,
                outcome: None,
            }),
            None => serial.push((index, a, b, x)),
        }
    }
    let Some(first) = jobs.first().map(|job| job.index) else {
        for (_, a, b, x) in serial {
            executor.solve_into(engine, a, b, x.reborrow())?;
        }
        return Ok(());
    };
    let mut serial = serial.into_iter().peekable();
    while let Some((_, a, b, x)) = serial.next_if(|&(index, ..)| index < first) {
        executor.solve_into(engine, a, b, x.reborrow())?;
    }
    let failed = match with_cpu_engine(engine, |cpu| {
        driver::drive(
            cpu,
            &SolveBlocks(PhantomData),
            &mut jobs,
            &mut LaneTables::default(),
        )
    }) {
        None => Some((
            first,
            tenferro_solve_error(tenferro_tensor::Error::unsupported(ENTRY, NOT_A_CPU_ENGINE)),
        )),
        Some(Err(error)) => Some((first, error)),
        Some(Ok(())) => jobs.into_iter().find_map(|job| match job.outcome {
            Some(Err(error)) => Some((job.index, error)),
            _ => None,
        }),
    };
    let bound = failed.as_ref().map_or(usize::MAX, |&(index, _)| index);
    for (_, a, b, x) in serial.take_while(|&(index, ..)| index < bound) {
        executor.solve_into(engine, a, b, x.reborrow())?;
    }
    failed.map_or(Ok(()), |(_, error)| Err(error))
}
