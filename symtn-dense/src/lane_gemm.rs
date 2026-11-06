//! The lane-GEMM seam: a Host replay that places its own jobs on the lane
//! runner runs each job's prepared GEMM on that job's lane context, inside
//! one engine entry. See [`crate::DenseExecutor::with_lane_gemms`].

use num_complex::{Complex32, Complex64};

use crate::block_kernel::driver::{self, OpKind};
use crate::lanes::{Estimate, Exec};
use crate::{DenseDType, DenseError};

/// One lane GEMM's geometry: `rows x k` times `k x n`, every operand packed
/// column-major with no transposition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaneGemmShape {
    pub rows: usize,
    pub k: usize,
    pub n: usize,
}

/// One lane GEMM's packed column-major operands and its initialized
/// destination, for one GEMM dtype: `lhs` is `rows x k`, `rhs` `k x n`,
/// `destination` `rows x n`, each exactly that long.
pub enum LaneGemmIo<'a> {
    F32 {
        lhs: &'a [f32],
        rhs: &'a [f32],
        destination: &'a mut [f32],
    },
    F64 {
        lhs: &'a [f64],
        rhs: &'a [f64],
        destination: &'a mut [f64],
    },
    C32 {
        lhs: &'a [Complex32],
        rhs: &'a [Complex32],
        destination: &'a mut [Complex32],
    },
    C64 {
        lhs: &'a [Complex64],
        rhs: &'a [Complex64],
        destination: &'a mut [Complex64],
    },
}

/// The prepared GEMMs of one [`crate::DenseExecutor::with_lane_gemms`] call,
/// shared by every lane of its body.
pub trait LaneGemms: Sync {
    /// `D = A B` of job `job` (`alpha = 1`, `beta = 0`) into `io`'s
    /// initialized destination, which it overwrites without reading, on
    /// `exec`, the stage's own context: the provider selects its width
    /// within `exec`'s budget. Never enters the engine.
    fn run(&self, exec: Exec<'_>, job: usize, io: LaneGemmIo<'_>) -> Result<(), DenseError>;

    /// The serial estimate of job `job` that the block kernels place their
    /// own GEMMs by: one provider stage. An unknown job estimates zero.
    fn estimate(&self, job: usize) -> Estimate;
}

/// The serial estimate a `dtype` GEMM of `shape` is placed by, lane GEMM or
/// engine submission alike: tcontract's rate over its multiply-adds (four
/// per complex one), one provider stage.
pub fn gemm_estimate(dtype: DenseDType, shape: LaneGemmShape) -> Estimate {
    let complex = matches!(dtype, DenseDType::C32 | DenseDType::C64);
    gemm_stage_estimate(shape, complex, &tcontract_exec::WorkModel::default())
}

/// [`gemm_estimate`] for the block kernels' own GEMMs, at tcontract's GEMM
/// rate (`model` prices only factorizations).
pub(crate) fn gemm_stage_estimate(
    shape: LaneGemmShape,
    complex: bool,
    model: &tcontract_exec::WorkModel,
) -> Estimate {
    let work =
        shape.rows as f64 * shape.k as f64 * shape.n as f64 * if complex { 4.0 } else { 1.0 };
    Estimate {
        serial: driver::cost(OpKind::Gemm, work, model),
        payback: None,
        sections: 1,
        widen: true,
    }
}

/// The body of a [`crate::DenseExecutor::with_lane_gemms`]: called at most
/// once, on the calling thread, with the entry's whole context.
pub type LaneGemmBody<'a> = dyn FnMut(Exec<'_>, &dyn LaneGemms) + 'a;

pub(crate) const LANE_GEMM_OP: &str = "with_lane_gemms";
