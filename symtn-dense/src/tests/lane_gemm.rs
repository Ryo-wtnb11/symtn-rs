//! The lane-GEMM seam: a caller's body runs prepared GEMMs on lanes it forks
//! from the entry's token, through a `dyn` executor.

use super::*;
use core::mem::MaybeUninit;
use core::num::NonZeroUsize;
use std::sync::Mutex;
use std::time::Duration;
use tcontract_exec::{BatchPolicy, BatchStrategy};

const TWO: NonZeroUsize = match NonZeroUsize::new(2) {
    Some(two) => two,
    None => unreachable!(),
};

fn shape(rows: usize, k: usize, n: usize) -> LaneGemmShape {
    LaneGemmShape { rows, k, n }
}

/// Runs job `job` of `gemms` on `token` into `destination`, returning the
/// initialised slice.
fn run_f64<'d>(
    gemms: &dyn LaneGemms,
    token: LaneToken<'_>,
    job: usize,
    (lhs, rhs): (&[f64], &[f64]),
    destination: &'d mut [MaybeUninit<f64>],
) -> Result<Option<&'d mut [f64]>, DenseError> {
    let mut written = None;
    gemms.run(
        token,
        job,
        LaneGemmIo::F64 {
            lhs,
            rhs,
            destination,
            written: &mut written,
        },
    )?;
    Ok(written)
}

#[test]
fn lane_gemms_run_on_forked_lanes_through_a_dyn_executor() {
    // What: the native executor, called as `dyn DenseExecutor`, runs one
    // job per lane on the two halves of its entry's token into
    // uninitialised destinations: A0 (2x3) B0 (3x2) and A1 (1x2) B1 (2x3),
    // column-major, against hand products; a complex job matches too. Its
    // time is positive, and under a forced policy, which the scheduler does
    // not model, one at every width.
    let (a0, b0) = (
        [1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
        [1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
    );
    let (a1, b1) = ([2.0, 3.0], [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    // The third job only times: a product large enough to cost nanoseconds.
    let jobs = [shape(2, 3, 2), shape(1, 2, 3), shape(64, 64, 64)];
    let mut engine = CpuEngine::new(crate::Pool::builder().threads(2).build().unwrap());
    let mut native = DefaultDenseExecutor::new();
    let executor: &mut dyn DenseExecutor = &mut native;
    assert!(executor.supports_lane_gemms());
    let results = Mutex::new([None, None]);
    let mut times = None;
    let mut ran = 0;
    executor
        .with_lane_gemms(
            &mut engine,
            DenseDType::F64,
            &jobs,
            &mut |mut token, gemms| {
                ran += 1;
                times = Some((
                    gemms.time(&mut token, 2, NonZeroUsize::MIN),
                    gemms.time(&mut token, 2, TWO),
                ));
                token
                    .fork_sized(&[1, 1], |job, child| {
                        let mut destination = vec![MaybeUninit::uninit(); [4, 3][job]];
                        let operands = if job == 0 {
                            (&a0[..], &b0[..])
                        } else {
                            (&a1[..], &b1[..])
                        };
                        let written = run_f64(gemms, child, job, operands, &mut destination)
                            .unwrap()
                            .map(|written| written.to_vec());
                        results.lock().unwrap()[job] = written;
                    })
                    .unwrap();
            },
        )
        .unwrap();
    assert_eq!(ran, 1);
    let [r0, r1] = results.into_inner().unwrap();
    assert_eq!(r0.unwrap(), [4.0, 10.0, 5.0, 11.0]);
    let (one, two) = times.unwrap();
    assert!(one > Duration::ZERO && two > Duration::ZERO);
    assert_eq!(r1.unwrap(), [14.0, 19.0, 24.0]);

    let (lhs, rhs) = ([c64(1.0, 2.0)], [c64(0.0, 1.0)]);
    let mut destination = [MaybeUninit::uninit(); 1];
    let mut value = None;
    native
        .with_lane_gemms(
            &mut engine,
            DenseDType::C64,
            &[shape(1, 1, 1)],
            &mut |token, gemms| {
                let mut written = None;
                gemms
                    .run(
                        token,
                        0,
                        LaneGemmIo::C64 {
                            lhs: &lhs,
                            rhs: &rhs,
                            destination: &mut destination,
                            written: &mut written,
                        },
                    )
                    .unwrap();
                value = written.map(|written| written[0]);
            },
        )
        .unwrap();
    assert_eq!(value, Some(c64(-2.0, 1.0)));

    let mut forced = CpuEngine::new(crate::Pool::builder().threads(2).build().unwrap())
        .with_batch_policy(BatchPolicy::new(BatchStrategy::Outer));
    let mut forced_times = None;
    native
        .with_lane_gemms(
            &mut forced,
            DenseDType::F64,
            &jobs,
            &mut |mut token, gemms| {
                forced_times = Some((
                    gemms.time(&mut token, 2, NonZeroUsize::MIN),
                    gemms.time(&mut token, 2, TWO),
                ));
            },
        )
        .unwrap();
    let (one, two) = forced_times.unwrap();
    assert_eq!(one, two);
}

#[test]
fn a_destination_that_does_not_cover_the_job_is_refused_unwritten() {
    // What: a lane GEMM whose fresh destination is shorter than its job's
    // `rows x n` fails with a typed error and reports no initialised slice.
    let (lhs, rhs) = ([1.0, 2.0], [3.0, 4.0]);
    let mut engine = CpuEngine::serial();
    let mut short = [MaybeUninit::uninit(); 3];
    let mut outcome = None;
    DefaultDenseExecutor::new()
        .with_lane_gemms(
            &mut engine,
            DenseDType::F64,
            &[shape(2, 1, 2)],
            &mut |token, gemms| {
                outcome =
                    Some(run_f64(gemms, token, 0, (&lhs, &rhs), &mut short).map(|w| w.is_some()));
            },
        )
        .unwrap();
    assert!(matches!(outcome, Some(Err(DenseError::Backend { .. }))));
}

struct NoLanes;

impl DenseExecutor for NoLanes {
    fn svd(
        &mut self,
        _: &mut dyn Engine,
        _: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        unreachable!("not a factorization test")
    }
    fn qr(&mut self, _: &mut dyn Engine, _: DenseRead<'_>) -> Result<Vec<DenseTensor>, DenseError> {
        unreachable!("not a factorization test")
    }
    fn eigh(
        &mut self,
        _: &mut dyn Engine,
        _: DenseRead<'_>,
    ) -> Result<Vec<DenseTensor>, DenseError> {
        unreachable!("not a factorization test")
    }
    fn dot_general_into(
        &mut self,
        _: &mut dyn Engine,
        _: DenseWrite<'_>,
        _: DenseRead<'_>,
        _: DenseRead<'_>,
        _: &DenseDotConfig,
    ) -> Result<(), DenseError> {
        unreachable!("not a contraction test")
    }
}

#[test]
fn every_refusal_comes_before_the_body_runs() {
    // What: a custom executor (the provided default), a dtype without
    // GEMMs and a broken engine refuse with a typed error before calling the
    // body, so a caller can still choose another route.
    let mut engine = CpuEngine::new(crate::Pool::builder().threads(2).build().unwrap());
    let mut ran = false;
    let mut body = |_: LaneToken<'_>, _: &dyn LaneGemms| ran = true;
    assert!(!NoLanes.supports_lane_gemms());
    assert!(matches!(
        NoLanes.with_lane_gemms(&mut engine, DenseDType::F64, &[shape(1, 1, 1)], &mut body),
        Err(DenseError::Unsupported { .. })
    ));
    let mut native = DefaultDenseExecutor::new();
    assert!(matches!(
        native.with_lane_gemms(&mut engine, DenseDType::I32, &[shape(1, 1, 1)], &mut body),
        Err(DenseError::Unsupported { .. })
    ));
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        native.with_lane_gemms(&mut engine, DenseDType::F64, &[], &mut |_, _| {
            panic!("a lane body panics")
        })
    }));
    assert!(panicked.is_err());
    assert!(engine.is_broken());
    assert!(native
        .with_lane_gemms(&mut engine, DenseDType::F64, &[], &mut body)
        .is_err());
    assert!(!ran);
}
